//! Reading and printing IR: round trips, parse and verifier errors, how much source a
//! diagnostic shows, nesting, and the dialects the tool links.

use std::rc::Rc;

use runic_core::{
    Context, OpPrinter, OpPrintingFlags,
    parse::{ParserConfig, parse_any},
    pass::{Nesting, PassPipeline},
    print::AsmPrinter,
};

use super::{INVALID, LOCALS, REPEATED_ADD, opt};

/// What printing `text` parsed, and then run through `pipeline` with the API, gives.
fn through_the_api(text: &str, pipeline: &str) -> String {
    let context = Rc::new(Context::default());
    let op = parse_any(ParserConfig::new(context.clone()), "api.runic", text).unwrap();
    let mut pm = PassPipeline::parse_elements(&op.borrow().name(), pipeline)
        .unwrap()
        .load(context.clone(), Nesting::Implicit)
        .unwrap();
    pm.run(op).unwrap();
    drop(pm);
    let flags = OpPrintingFlags::default();
    let op = op.borrow();
    let mut printer = AsmPrinter::root(context.clone(), &flags);
    op.print(&mut printer);
    format!("{}\n", printer.finish())
}

#[test]
fn a_module_round_trips() {
    let output = opt(&[], LOCALS);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout, through_the_api(LOCALS, ""));
    // What it prints reads back to the same text.
    let again = opt(&["-"], &output.stdout);
    again.succeeded();
    assert_eq!(again.stdout, output.stdout);
}

#[test]
fn cse_removes_a_repeated_pure_operation_as_the_api_does() {
    let output = opt(&["-p", "cse"], REPEATED_ADD);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout.matches("arith.add").count(), 2, "{}", output.stdout);
    assert_eq!(output.stdout, through_the_api(REPEATED_ADD, "cse"));
}

#[test]
fn a_parse_error_is_rendered_with_its_source_line_and_nothing_is_printed() {
    let text =
        "builtin.module public @m {\n    builtin.function public extern(\"C\") @f() {\n        \
         builtin.ret\n    };\n};\n";
    let output = opt(&[], text);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.starts_with("error: "), "{}", output.stderr);
    assert!(output.stderr.contains("--> <stdin>:4:5"), "{}", output.stderr);
    assert!(output.stderr.contains("4 |     };"), "{}", output.stderr);
}

#[test]
fn a_verifier_failure_on_parse_is_an_error() {
    let output = opt(&[], INVALID);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: invalid operation builtin.function"),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains("this returns (), but the function returns (i32)"),
        "{}",
        output.stderr
    );
}

#[test]
fn a_diagnostic_in_a_long_input_shows_the_lines_it_is_about_and_not_the_file() {
    let mut text = String::from("builtin.module public @m {\n");
    for i in 0..250 {
        text += &format!(
            "    // CHECK: something about function {i}\n    builtin.function public \
             extern(\"C\") @f{i}() {{\n        builtin.ret;\n    }};\n"
        );
        if i == 125 {
            text += "    builtin.function public extern(\"C\") @bad() -> i32 {\n        \
                     builtin.ret;\n    };\n";
        }
    }
    text += "};\n";
    assert!(text.lines().count() >= 1_000);
    let output = opt(&[], &text);
    output.failed();
    assert!(
        output.stderr.contains("this returns (), but the function returns (i32)"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("@bad"), "{}", output.stderr);
    assert!(output.stderr.lines().count() < 15, "{}", output.stderr);
    assert!(!output.stderr.contains("CHECK"), "{}", output.stderr);
}

/// `depth` nested modules.
fn nested_modules(depth: usize) -> String {
    let mut text = String::new();
    for i in 0..depth {
        text += &format!("builtin.module public @m{i} {{\n");
    }
    for _ in 0..depth {
        text += "};\n";
    }
    text
}

#[test]
fn regions_nested_900_deep_parse_and_print() {
    let output = opt(&["-p", "cse"], &nested_modules(900));
    output.succeeded();
    assert!(output.stdout.contains("@m899 {"));
}

#[test]
fn regions_nested_exactly_1000_deep_parse_print_and_go_through_a_pass() {
    let text = nested_modules(1_000);
    let printed = opt(&[], &text);
    printed.succeeded();
    assert_eq!(printed.stderr, "");
    assert!(printed.stdout.contains("@m999 {"));
    let optimized = opt(&["-p", "cse"], &text);
    optimized.succeeded();
    assert_eq!(optimized.stderr, "");
    assert_eq!(optimized.stdout, printed.stdout);
}

#[test]
fn regions_nested_1001_deep_are_the_parsers_diagnostic() {
    let output = opt(&[], &nested_modules(1_001));
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: regions are nested more than 1000 deep\n"),
        "{}",
        output.stderr
    );
    assert!(
        output
            .stderr
            .contains("this region is nested 1001 deep; at most 1000 is accepted"),
        "{}",
        output.stderr
    );
    // The message names no API type.
    assert!(!output.stderr.contains("ParserConfig"), "{}", output.stderr);
}

/// A module of a function with the parameters `params`, which returns `results` and whose body is
/// `body`.
fn function(params: &str, results: &str, body: &str) -> String {
    format!(
        "builtin.module public @m {{\n    builtin.function public extern(\"C\") \
         @f({params}){results} {{\n{body}\n    }};\n}};\n"
    )
}

#[test]
fn an_operation_of_every_linked_dialect_parses() {
    let cases = [
        (
            "arith",
            function(
                "%a: i32, %b: i32",
                " -> i32",
                "%x = arith.add %a, %b;\nbuiltin.ret %x : (i32);",
            ),
        ),
        ("cf", function("%c: i1", "", "cf.assert %c, \"holds\";\nbuiltin.ret;")),
        (
            "mem",
            function("%p: ptr<i32>", " -> i32", "%x = mem.load %p;\nbuiltin.ret %x : (i32);"),
        ),
        (
            "scf",
            function(
                "%c: i1",
                "",
                "scf.if %c then {\nscf.yield : ();\n} else {\nscf.yield : ();\n} : (i1) -> \
                 ();\nbuiltin.ret;",
            ),
        ),
        (
            "ub",
            function(
                "",
                " -> i32",
                "%x = ub.poison <{ value = #ub.poison<i32> }>;\nbuiltin.ret %x : (i32);",
            ),
        ),
        (
            "builtin",
            String::from(
                "builtin.module public @m {\n    builtin.global_variable public @g : i32;\n    \
                 builtin.function public extern(\"C\") @id(%a: i32) -> i32 {\n        builtin.ret \
                 %a : (i32);\n    };\n    builtin.function public extern(\"C\") @f(%a: i32) -> \
                 i32 locals(i32) {\n        builtin.store_local 0, %a;\n        %v = \
                 builtin.load_local 0 : i32;\n        %r = builtin.call @id(%v) : extern(\"C\") \
                 (i32) -> i32;\n        builtin.ret %r : (i32);\n    };\n};\n",
            ),
        ),
        (
            "di",
            function(
                "%a: i32",
                " -> i32",
                "di.debug_declare <{ variable = #di.variable<{ name = \"x\", file = \"x.rs\", \
                 line = 1 }>, expression = #di.expression<[DI_OP_frame_base(local_slot, 0+0)]> \
                 }>;\nbuiltin.ret %a : (i32);",
            ),
        ),
        (
            "test",
            function("%a: i32", " -> i32", "%x = test.add %a, %a;\nbuiltin.ret %x : (i32);"),
        ),
    ];
    for (dialect, text) in cases {
        let output = opt(&[], &text);
        assert_eq!(output.code, std::process::ExitCode::SUCCESS, "{dialect}: {}", output.stderr);
        assert!(output.stdout.contains(&format!("{dialect}.")), "{dialect}: {}", output.stdout);
    }
}

/// The rejection of a value defined twice: the name, at the second definition (`line:column`),
/// with a label at the first, on line `first`.
#[track_caller]
fn assert_redefined(text: &str, name: &str, second: &str, first: usize) {
    let output = opt(&[], text);
    output.failed();
    assert_eq!(output.stdout, "");
    let stderr = &output.stderr;
    let message = format!("error: redefinition of value '{name}'\n --> <stdin>:{second}\n");
    assert!(stderr.starts_with(&message), "{stderr}");
    assert!(stderr.contains(&format!("^ '{name}' is defined again here")), "{stderr}");
    assert!(stderr.contains(&format!("'{name}' is first defined here")), "{stderr}");
    // The line of the first definition is shown.
    assert!(stderr.contains(&format!("\n{first} |")), "{stderr}");
}

#[test]
fn a_result_defined_twice_is_an_error() {
    let body =
        "%x = arith.constant 1 : i32;\n%x = arith.constant 2 : i32;\nbuiltin.ret %x : (i32);";
    assert_redefined(&function("", " -> i32", body), "%x", "4:2", 3);
    // The same in the generic form.
    let body = "%x = \"arith.constant\"() <{ value = #builtin.int<1 : i32> }> : () -> i32;\n%x = \
                \"arith.constant\"() <{ value = #builtin.int<2 : i32> }> : () -> \
                i32;\nbuiltin.ret %x : (i32);";
    assert_redefined(&function("", " -> i32", body), "%x", "4:2", 3);
}

#[test]
fn a_result_with_the_name_of_a_parameter_is_an_error() {
    let body = "%a = arith.constant 1 : i32;\nbuiltin.ret %a : (i32);";
    assert_redefined(&function("%a: i32", " -> i32", body), "%a", "3:2", 2);
}

#[test]
fn two_parameters_of_one_name_are_an_error() {
    assert_redefined(
        &function("%a: i32, %a: i32", " -> i32", "builtin.ret %a : (i32);"),
        "%a",
        "2:54",
        2,
    );
}

#[test]
fn a_block_argument_defined_twice_in_a_region_is_an_error() {
    // With the name of a parameter.
    let body = "cf.br ^b(%a : i32);\n^b(%a: i32):\nbuiltin.ret %a : (i32);";
    assert_redefined(&function("%a: i32", " -> i32", body), "%a", "4:5", 2);
    // With the name of an argument of another block of the region.
    let body = "cf.br ^b(%a : i32);\n^b(%x: i32):\ncf.br ^c(%x : i32);\n^c(%x: i32):\nbuiltin.ret \
                %x : (i32);";
    assert_redefined(&function("%a: i32", " -> i32", body), "%x", "6:5", 4);
}

#[test]
fn a_nested_region_cannot_reuse_a_name_defined_around_it() {
    let body = "%r = scf.if %c then {\n%a = arith.add %a, %a;\nscf.yield %a : (i32);\n} else \
                {\nscf.yield %a : (i32);\n} : (i1) -> (i32);\nbuiltin.ret %r : (i32);";
    assert_redefined(&function("%a: i32, %c: i1", " -> i32", body), "%a", "4:2", 2);
}

/// A name goes out of scope at the end of the region it is defined in, and every function (an
/// operation isolated from above) starts its names afresh.
#[test]
fn a_name_may_be_reused_in_a_sibling_region_after_its_region_and_in_another_function() {
    let body = "%r = scf.if %c then {\n%x = arith.add %a, %a;\nscf.yield %x : (i32);\n} else \
                {\n%x = arith.mul %a, %a;\nscf.yield %x : (i32);\n} : (i1) -> (i32);\n%x = \
                arith.add %r, %r;\nbuiltin.ret %x : (i32);";
    let output = opt(&[], &function("%a: i32, %c: i1", " -> i32", body));
    output.succeeded();
    // `%x` after the `scf.if` is the sum of its result.
    assert!(
        output.stdout.contains("%5 = arith.add %4, %4;\n        builtin.ret %5"),
        "{}",
        output.stdout
    );

    let text = "builtin.module public @m {\n    builtin.function public extern(\"C\") @f(%a: i32) \
                -> i32 {\n        %x = arith.add %a, %a;\n        builtin.ret %x : (i32);\n    \
                };\n    builtin.function public extern(\"C\") @g(%a: i32) -> i32 {\n        %x = \
                arith.mul %a, %a;\n        builtin.ret %x : (i32);\n    };\n};\n";
    opt(&[], text).succeeded();
}

/// A name bound to several results (`%p:2`) is scoped like any other name: two branches of an
/// `scf.if` may each define it, the `scf.if` may take it for its own results, and after the
/// region that defined it a use of it is undeclared.
#[test]
fn a_multi_result_name_leaves_scope_with_its_region() {
    let define = "%p:2 = \"builtin.unrealized_conversion_cast\"(%a) : (i16) -> (i16, i16);";
    let body = format!(
        "%r = scf.if %c then {{\n{define}\nscf.yield %p#0 : (i16);\n}} else \
         {{\n{define}\nscf.yield %p#1 : (i16);\n}} : (i1) -> (i16);\nbuiltin.ret %r : (i16);"
    );
    opt(&[], &function("%a: i16, %c: i1", " -> i16", &body)).succeeded();

    let body = format!(
        "%p:2 = \"scf.if\"(%c) ({{\n{define}\nscf.yield %p#0, %p#1 : (i16, i16);\n}}, \
         {{\nscf.yield %a, %a : (i16, i16);\n}}) : (i1) -> (i16, i16);\nbuiltin.ret %p#1 : (i16);"
    );
    opt(&[], &function("%a: i16, %c: i1", " -> i16", &body)).succeeded();

    let body = format!(
        "%r = scf.if %c then {{\n{define}\nscf.yield %p#0 : (i16);\n}} else {{\nscf.yield %a : \
         (i16);\n}} : (i1) -> (i16);\nbuiltin.ret %p#1 : (i16);"
    );
    let output = opt(&[], &function("%a: i16, %c: i1", " -> i16", &body));
    output.failed();
    assert!(
        output.stderr.starts_with("error: use of undeclared values\n"),
        "{}",
        output.stderr
    );
}

/// A module whose function holds an `arith.add` with the attribute `note = {note}`.
fn with_note(note: &str) -> String {
    format!(
        "builtin.module public @m {{\n    builtin.function public extern(\"C\") @g(%a: i32) -> \
         i32 {{\n        %4 = arith.add %a, %a attributes {{ note = {note} }};\n        \
         builtin.ret %4 : (i32);\n    }};\n}};\n"
    )
}

/// The tool raises the reader's list nesting limit as it raises the region depth: a list 200 deep
/// reads back in both forms, and one 1,000 deep is read; one deeper is the parser's diagnostic.
#[test]
fn lists_nested_up_to_1000_deep_are_read() {
    let list = |depth: usize| format!("{}{}", "[".repeat(depth), "]".repeat(depth));
    let output = opt(&["--verify-roundtrip"], &with_note(&list(200)));
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert!(output.stdout.contains(&format!("note = {}", list(200))), "{}", output.stdout);
    opt(&[], &with_note(&list(1_000))).succeeded();
    let output = opt(&[], &with_note(&list(1_001)));
    output.failed();
    assert!(
        output.stderr.starts_with("error: lists are nested more than 1000 deep\n"),
        "{}",
        output.stderr
    );
    assert!(
        output
            .stderr
            .contains("this list is nested 1001 deep; at most 1000 is accepted"),
        "{}",
        output.stderr
    );
}

/// A bare `@name` is a symbol reference as an attribute value, read and printed as
/// `#builtin.symbol<..>` is; a second reference after it is a syntax error.
#[test]
fn a_bare_symbol_reference_is_an_attribute_value() {
    let bare = opt(&["--verify-roundtrip"], &with_note("@g"));
    bare.succeeded();
    assert_eq!(bare.stderr, "");
    assert!(bare.stdout.contains("note = #builtin.symbol<@g>"), "{}", bare.stdout);
    let spelled = opt(&["--verify-roundtrip"], &with_note("#builtin.symbol<@g>"));
    assert_eq!(bare.stdout, spelled.stdout);
    let output = opt(&[], &with_note("@zzz @g"));
    output.failed();
    assert_eq!(output.stdout, "");
    // A reference that names nothing is the verifier's error, as for the other spelling
    let output = opt(&[], &with_note("@nosuch"));
    output.failed();
    assert!(output.stderr.contains("'@nosuch'"), "{}", output.stderr);
}

/// A bare rooted path, `::@m::@g`, is read and printed as `#builtin.symbol<::@m::@g>` is, alone
/// and in a list, and reads back under `--verify-roundtrip`; one that names nothing is the
/// verifier's error.
#[test]
fn a_bare_rooted_path_is_an_attribute_value() {
    let bare = opt(&["--verify-roundtrip"], &with_note("::@m::@g"));
    bare.succeeded();
    assert_eq!(bare.stderr, "");
    assert!(bare.stdout.contains("note = #builtin.symbol<::@m::@g>"), "{}", bare.stdout);
    let spelled = opt(&["--verify-roundtrip"], &with_note("#builtin.symbol<::@m::@g>"));
    assert_eq!(bare.stdout, spelled.stdout);
    let listed = opt(&["--verify-roundtrip"], &with_note("[::@m::@g, @g]"));
    listed.succeeded();
    assert!(
        listed
            .stdout
            .contains("note = [#builtin.symbol<::@m::@g>, #builtin.symbol<@g>]"),
        "{}",
        listed.stdout
    );
    let output = opt(&[], &with_note("::@m::@nosuch"));
    output.failed();
    assert!(output.stderr.contains("@nosuch"), "{}", output.stderr);
}
