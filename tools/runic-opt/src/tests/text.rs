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
fn regions_nested_1001_deep_are_the_parsers_diagnostic() {
    let output = opt(&[], &nested_modules(1_001));
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: regions are nested more than 1000 deep"),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains("this region is nested more than 1000 deep"),
        "{}",
        output.stderr
    );
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
    ];
    for (dialect, text) in cases {
        let output = opt(&[], &text);
        assert_eq!(output.code, std::process::ExitCode::SUCCESS, "{dialect}: {}", output.stderr);
        assert!(output.stdout.contains(&format!("{dialect}.")), "{dialect}: {}", output.stdout);
    }
}
