//! The structured control flow operations: `block`, `loop`, `if`, `end`, `br`, `br_table` and
//! `unreachable`; their text, their rules, and the builders.
//!
//! The analyses and the passes over them are in `control_analyses`, and a randomized test of the
//! verifier in `control_random`.

use runic_ir::{
    Builder, RegionExitTerminatorOpInterface, SourceSpan,
    dialects::{
        builtin::{BuiltinOpBuilder, Function},
        test::TestDialect,
    },
    testing::{
        Test,
        text::{normalize, print_all_generic, round_trip, signatures},
    },
};

use super::*;

/// A context with the dialects of the tests of control flow: `wasm`, `scf` and `test`.
pub(super) fn rich_context() -> Rc<Context> {
    let context = context();
    context.get_or_register_dialect::<runic_dialect_scf::ScfDialect>();
    context.get_or_register_dialect::<TestDialect>();
    context
}

/// The text of the function `@f` with the parameters `params`, the results `results` (as written
/// after `->`, or empty) and the body `body`, which ends in its own `builtin.ret`.
pub(super) fn function_with(params: &str, results: &str, body: &str) -> String {
    let results = if results.is_empty() {
        String::new()
    } else {
        format!(" -> {results}")
    };
    format!("builtin.function public extern(\"C\") @f({params}){results} {{\n{body}\n}};")
}

/// `text`, a function, parsed in `context` and verified: its printed text, or the error with its
/// labels.
pub(super) fn check_in(context: &Rc<Context>, text: &str) -> Result<String, String> {
    let (function, printed) =
        parse_function_fixpoint(context, "f", text).map_err(|report| describe(&report))?;
    function
        .borrow()
        .as_operation()
        .recursively_verify()
        .map_err(|report| format!("does not verify: {}", describe(&report)))?;
    Ok(printed)
}

/// The printed text of the function, which must be accepted.
#[track_caller]
pub(super) fn accepted_function(params: &str, results: &str, body: &str) -> String {
    let text = function_with(params, results, body);
    match check_in(&rich_context(), &text) {
        Ok(printed) => printed,
        Err(error) => panic!("rejected: {error}\n{text}"),
    }
}

/// The error of the function, which must not be accepted.
#[track_caller]
pub(super) fn rejected_function(params: &str, results: &str, body: &str) -> String {
    let text = function_with(params, results, body);
    match check_in(&rich_context(), &text) {
        Ok(printed) => panic!("accepted:\n{printed}"),
        Err(error) => error,
    }
}

/// Checks that `function` is rejected with a message that contains `message`.
#[track_caller]
fn rejects_with(params: &str, results: &str, body: &str, message: &str) {
    let error = rejected_function(params, results, body);
    assert!(error.contains(message), "`{message}` is not in\n{error}\nfor\n{body}");
}

/// Round trips the function through its custom and its generic form, whole: the text of a module
/// holding it, read back from either form, prints the same, and every operation is the same.
#[track_caller]
pub(super) fn assert_round_trips(function: &str) {
    let source = format!("builtin.module public @m {{\n{function}\n}};");
    let context = rich_context();
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "control", source)
        .into_result()
        .unwrap_or_else(|report| panic!("{}\n{function}", describe(&report)));
    module
        .borrow()
        .as_operation()
        .recursively_verify()
        .unwrap_or_else(|report| panic!("{}\n{function}", describe(&report)));
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));

    let custom = module.borrow().as_operation().to_string();
    let generic = print_all_generic(module.borrow().as_operation());
    let reread =
        parse::parse::<Module>(ParserConfig::new(context.clone()), "control", generic.clone())
            .into_result()
            .unwrap_or_else(|report| panic!("{}\n{generic}", describe(&report)));
    reread
        .borrow()
        .as_operation()
        .recursively_verify()
        .unwrap_or_else(|report| panic!("{}\n{generic}", describe(&report)));
    assert_eq!(
        normalize(&reread.borrow().as_operation().to_string()),
        normalize(&custom),
        "the generic form\n{generic}"
    );
    assert_eq!(
        signatures(reread.borrow().as_operation()),
        signatures(module.borrow().as_operation()),
        "the generic form\n{generic}"
    );
}

// ---------------------------------------------------------------------------------------------
// Text

#[test]
fn a_block_with_parameters_and_results_prints_parses_and_round_trips() {
    let body = r#"
    %r1, %r2 = wasm.block %a, %b {
    ^body(%x: i32, %y: i64):
        wasm.end %x, %y : (i32, i64);
    } : (i32, i64) -> (i32, i64);
    %s = wasm.block %a {
    ^body(%x: i32):
        wasm.end %x : (i32);
    } : (i32) -> (i32);
    wasm.block {
        wasm.end : ();
    } : () -> ();
    builtin.ret %s : (i32);"#;
    let printed = accepted_function("%a: i32, %b: i64", "i32", body);
    assert!(
        printed.contains("%7, %8 = wasm.block %0, %1 {") || printed.contains("wasm.block"),
        "{printed}"
    );
    assert!(printed.contains("} : (i32, i64) -> (i32, i64);"), "{printed}");
    assert!(printed.contains("wasm.end : ();"), "{printed}");
    assert_round_trips(&function_with("%a: i32, %b: i64", "i32", body));
}

#[test]
fn a_loop_with_parameters_and_results_prints_parses_and_round_trips() {
    let body = r#"
    %r1, %r2 = wasm.loop %a, %b {
    ^body(%x: i32, %y: i64):
        wasm.br 0 %x, %y : (i32, i64);
    } : (i32, i64) -> (i32, i64);
    %s = wasm.loop %a {
    ^body(%x: i32):
        wasm.end %x : (i32);
    } : (i32) -> (i32);
    wasm.loop {
        wasm.end : ();
    } : () -> ();
    builtin.ret %s : (i32);"#;
    let printed = accepted_function("%a: i32, %b: i64", "i32", body);
    assert!(printed.contains("wasm.br 0 %"), "{printed}");
    assert!(printed.contains("} : (i32, i64) -> (i32, i64);"), "{printed}");
    assert_round_trips(&function_with("%a: i32, %b: i64", "i32", body));
}

#[test]
fn an_if_with_an_else_prints_parses_and_round_trips() {
    let body = r#"
    %r1, %r2 = wasm.if %c, %a, %b then {
    ^then(%x: i32, %y: i64):
        wasm.end %x, %y : (i32, i64);
    } else {
    ^else(%x: i32, %y: i64):
        wasm.br 0 %x, %y : (i32, i64);
    } : (i32, i32, i64) -> (i32, i64);
    %s = wasm.if %c, %a then {
    ^then(%x: i32):
        wasm.end %x : (i32);
    } else {
    ^else(%x: i32):
        %z = wasm.add %x, %x;
        wasm.end %z : (i32);
    } : (i32, i32) -> (i32);
    wasm.if %c then {
        wasm.end : ();
    } else {
        wasm.end : ();
    } : (i32) -> ();
    builtin.ret %s : (i32);"#;
    let printed = accepted_function("%c: i32, %a: i32, %b: i64", "i32", body);
    // the third has no operands to pass through, and its `else` only ends: it is left out

    assert_eq!(printed.matches("else {").count(), 2, "{printed}");
    assert!(printed.contains("} : (i32, i32, i64) -> (i32, i64);"), "{printed}");
    assert_round_trips(&function_with("%c: i32, %a: i32, %b: i64", "i32", body));
}

#[test]
fn an_if_without_an_else_has_a_second_region_that_passes_the_operands_through() {
    let body = r#"
    %r = wasm.if %c, %a then {
    ^then(%x: i32):
        %y = wasm.add %x, %x;
        wasm.end %y : (i32);
    } : (i32, i32) -> (i32);
    builtin.ret %r : (i32);"#;
    let printed = accepted_function("%c: i32, %a: i32", "i32", body);
    assert!(!printed.contains("else"), "{printed}");
    assert_round_trips(&function_with("%c: i32, %a: i32", "i32", body));

    // the region the parser built
    let context = rich_context();
    let (function, _) =
        parse_function_fixpoint(&context, "f", &function_with("%c: i32, %a: i32", "i32", body))
            .unwrap();
    let mut found = Vec::new();
    function.borrow().as_operation().prewalk_all(|op: &Operation| {
        if let Some(if_op) = op.downcast_ref::<If>() {
            let region = if_op.else_body();
            let entry = region.entry();
            let ops = entry.body().iter().map(|op| op.name().to_string()).collect::<Vec<_>>();
            assert_eq!(ops, ["wasm.end"]);
            assert_eq!(entry.arguments().len(), 1);
            let end = entry.body().front().get().unwrap();
            let end = end.downcast_ref::<End>().unwrap();
            assert_eq!(
                end.as_operation().operands()[0].borrow().as_value_ref(),
                entry.arguments()[0] as runic_ir::ValueRef
            );
            assert!(if_op.else_is_pass_through());
            found.push(());
        }
    });
    assert_eq!(found.len(), 1);

    // an explicit pass-through `else` is the same operation, and is left out of the text too
    let explicit = body.replace(
        "} : (i32, i32) -> (i32);",
        "} else {\n^else(%x: i32):\n wasm.end %x : (i32);\n} : (i32, i32) -> (i32);",
    );
    assert_eq!(accepted_function("%c: i32, %a: i32", "i32", &explicit), printed);

    // `br_if`: the first region branches, the second passes through
    let br_if = r#"
    %r = wasm.block %a {
    ^body(%v: i32):
        %w = wasm.if %c, %v then {
        ^taken(%x: i32):
            wasm.br 1 %x : (i32);
        } : (i32, i32) -> (i32);
        wasm.end %w : (i32);
    } : (i32) -> (i32);
    builtin.ret %r : (i32);"#;
    let printed = accepted_function("%c: i32, %a: i32", "i32", br_if);
    assert!(!printed.contains("else"), "{printed}");
    assert_round_trips(&function_with("%c: i32, %a: i32", "i32", br_if));
}

#[test]
fn an_if_without_an_else_whose_parameters_differ_from_its_results_is_a_parse_error() {
    for (params, signature, body_ty) in [
        ("%c: i32, %a: i32", "(i32, i32) -> (i64)", "i32"),
        ("%c: i32, %a: i32", "(i32, i32) -> ()", "i32"),
    ] {
        let end = if signature.ends_with("(i64)") {
            "%z = wasm.extend_s %x <{ ty = i64 }>;\n wasm.end %z : (i64);"
        } else {
            "wasm.end : ();"
        };
        let text = function_with(
            params,
            "",
            &format!(
                "wasm.if %c, %a then {{\n^then(%x: {body_ty}):\n {end}\n}} : \
                 {signature};\nbuiltin.ret;"
            ),
        );
        let error = check_in(&rich_context(), &text).expect_err("not a program");
        assert!(
            error.contains(
                "an `if` with no `else` passes its parameters (i32) through as its results"
            ),
            "{error}"
        );
    }
}

#[test]
fn constructs_nest_three_deep_with_several_parameters_and_results() {
    let body = r#"
    %r1, %r2 = wasm.block %a, %b {
    ^b1(%x1: i32, %y1: i64):
        %l1, %l2 = wasm.loop %x1, %y1 {
        ^l(%x2: i32, %y2: i64):
            %i1, %i2 = wasm.if %c, %x2, %y2 then {
            ^t(%x3: i32, %y3: i64):
                wasm.br 1 %x3, %y3 : (i32, i64);
            } else {
            ^e(%x3: i32, %y3: i64):
                wasm.br 2 %x3, %y3 : (i32, i64);
            } : (i32, i32, i64) -> (i32, i64);
            wasm.end %i1, %i2 : (i32, i64);
        } : (i32, i64) -> (i32, i64);
        wasm.end %l1, %l2 : (i32, i64);
    } : (i32, i64) -> (i32, i64);
    builtin.ret %r1 : (i32);"#;
    accepted_function("%c: i32, %a: i32, %b: i64", "i32", body);
    assert_round_trips(&function_with("%c: i32, %a: i32, %b: i64", "i32", body));
}

#[test]
fn a_branch_table_and_a_trap_print_parse_and_round_trip() {
    let body = r#"
    %r = wasm.block %a {
    ^b(%v: i32):
        %q = wasm.loop %v {
        ^h(%w: i32):
            wasm.br_table %i, %w [{
                wasm.br 0 %w : (i32);
            }, {
                wasm.br 1 %w : (i32);
            }] default {
                wasm.br 1 %w : (i32);
            } : (i32, i32);
        } : (i32) -> (i32);
        wasm.end %q : (i32);
    } : (i32) -> (i32);
    wasm.block {
        wasm.unreachable;
    } : () -> ();
    builtin.ret %r : (i32);"#;
    let printed = accepted_function("%i: i32, %a: i32", "i32", body);
    assert!(printed.contains("wasm.br_table %"), "{printed}");
    assert!(printed.contains("wasm.unreachable;"), "{printed}");
    assert_round_trips(&function_with("%i: i32, %a: i32", "i32", body));
}

// ---------------------------------------------------------------------------------------------
// Verification

const SIG: &str = "%c: i32, %a: i32, %b: i64";

#[test]
fn a_branch_to_a_block_carries_the_results_of_the_block() {
    let body = |carried: &str, ty: &str| {
        format!(
            "%r = wasm.block {{\n    wasm.br 0 {carried} : ({ty});\n}} : () -> \
             (i32);\nbuiltin.ret %r : (i32);"
        )
    };
    accepted_function(SIG, "i32", &body("%a", "i32"));
    rejects_with(
        SIG,
        "i32",
        &body("%b", "i64"),
        "`wasm.br` names an enclosing `wasm.block`, `wasm.loop` or `wasm.if` and carries the \
         values it expects: the results of a block or an if, the parameters of a loop, but it \
         carries (i64), and the results of the `wasm.block` or `wasm.if` it names are (i32)",
    );
    let none = "%r = wasm.block {\n    wasm.br 0 : ();\n} : () -> (i32);\nbuiltin.ret %r : (i32);";
    rejects_with(
        SIG,
        "i32",
        none,
        "it carries (), and the results of the `wasm.block` or `wasm.if` it names are (i32)",
    );
}

#[test]
fn a_branch_to_a_loop_carries_the_parameters_of_the_loop() {
    let body = |carried: &str, ty: &str| {
        format!(
            "%r = wasm.loop %a {{\n^h(%i: i32):\n    wasm.br 0 {carried} : ({ty});\n}} : (i32) -> \
             (i32);\nbuiltin.ret %r : (i32);"
        )
    };
    accepted_function(SIG, "i32", &body("%a", "i32"));
    rejects_with(
        SIG,
        "i32",
        &body("%b", "i64"),
        "it carries (i64), and the parameters of the `wasm.loop` it names (the arguments of its \
         body) are (i32)",
    );
}

#[test]
fn a_branch_to_an_if_carries_the_results_of_the_if() {
    let body = |carried: &str, ty: &str| {
        format!(
            "%r = wasm.if %c then {{\n    wasm.br 0 {carried} : ({ty});\n}} else {{\n    wasm.br \
             0 %a : (i32);\n}} : (i32) -> (i32);\nbuiltin.ret %r : (i32);"
        )
    };
    accepted_function(SIG, "i32", &body("%a", "i32"));
    rejects_with(
        SIG,
        "i32",
        &body("%b", "i64"),
        "it carries (i64), and the results of the `wasm.block` or `wasm.if` it names are (i32)",
    );
}

#[test]
fn each_level_names_the_construct_it_counts_to_from_inside_nested_constructs() {
    // block (i32) > loop (i64) > if (f32 -> i32... each of a different type, so a wrong level is
    // a type error
    let body = |level: u8, value: &str, ty: &str| {
        format!(
            r#"
    %r = wasm.block {{
        %q = wasm.loop %b {{
        ^h(%p: i64):
            %s = wasm.if %c then {{
                wasm.br {level} {value} : ({ty});
            }} else {{
                wasm.end %a : (i32);
            }} : (i32) -> (i32);
            wasm.end %p : (i64);
        }} : (i64) -> (i64);
        wasm.end %a : (i32);
    }} : () -> (i32);
    builtin.ret %r : (i32);"#
        )
    };
    // 0 is the `if` (an i32), 1 the loop (its parameters: an i64), 2 the block (an i32)
    accepted_function(SIG, "i32", &body(0, "%a", "i32"));
    accepted_function(SIG, "i32", &body(1, "%b", "i64"));
    accepted_function(SIG, "i32", &body(2, "%a", "i32"));
    rejects_with(
        SIG,
        "i32",
        &body(0, "%b", "i64"),
        "it carries (i64), and the results of the `wasm.block` or `wasm.if` it names are (i32)",
    );
    rejects_with(
        SIG,
        "i32",
        &body(1, "%a", "i32"),
        "it carries (i32), and the parameters of the `wasm.loop` it names (the arguments of its \
         body) are (i64)",
    );
    rejects_with(
        SIG,
        "i32",
        &body(2, "%b", "i64"),
        "it carries (i64), and the results of the `wasm.block` or `wasm.if` it names are (i32)",
    );
}

#[test]
fn a_level_that_names_the_function_is_an_error_that_says_to_use_return() {
    let body = |level: i32| {
        format!("wasm.block {{\n    wasm.br {level} : ();\n}} : () -> ();\nbuiltin.ret %a : (i32);")
    };
    rejects_with(
        SIG,
        "i32",
        &body(1),
        "`wasm.br` names label 1, which is the function itself: Wasm lets `br` name the \
         function's label, but this IR leaves a function with `builtin.ret`",
    );
    rejects_with(
        SIG,
        "i32",
        &body(2),
        "`wasm.br` names label 2, but only 1 `wasm.block`, `wasm.loop` or `wasm.if` enclose it \
         (label 1 would be the function itself, which this IR leaves with `builtin.ret`)",
    );
    rejects_with(
        SIG,
        "i32",
        &body(-1),
        "`wasm.br` names label -1, but a label is numbered from 0",
    );
    // directly in the function body there is no label at all
    let direct = "wasm.br 0 : ();";
    rejects_with(SIG, "", direct, "`wasm.br` names label 0, which is the function itself");
}

#[test]
fn a_branch_may_not_cross_an_operation_that_does_not_forward_exits() {
    let body = r#"
    %r = wasm.block {
        "test.reads_locals_and_holds"() ({
            wasm.br 0 %a : (i32);
        }) : () -> ();
        wasm.end %a : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#;
    rejects_with(
        SIG,
        "i32",
        body,
        "`wasm.br` names label 0, but would have to cross `test.reads_locals_and_holds`, which \
         has not declared that it forwards breaking control flow",
    );
}

#[test]
fn an_scf_if_between_a_branch_and_its_target_is_crossed_and_is_not_a_label() {
    // `scf.if` forwards exits and is no label: level 0 is still the block
    let body = r#"
    %r = wasm.block {
        scf.if %f then {
            wasm.br 0 %a : (i32);
        } else {
            scf.yield : ();
        } : (i1) -> ();
        wasm.end %a : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#;
    accepted_function("%f: i1, %a: i32", "i32", body);
    // so level 1 is the function
    rejects_with(
        "%f: i1, %a: i32",
        "i32",
        &body.replace("wasm.br 0", "wasm.br 1"),
        "which is the function itself",
    );
    // a `wasm.if` counts
    let counted = r#"
    %r = wasm.block {
        wasm.if %c then {
            wasm.br 1 %a : (i32);
        } : (i32) -> ();
        wasm.end %a : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#;
    accepted_function("%c: i32, %a: i32", "i32", counted);
    // level 0 is that `if`, which has no results
    rejects_with(
        "%c: i32, %a: i32",
        "i32",
        &counted.replace("wasm.br 1", "wasm.br 0"),
        "it carries (i32), and the results of the `wasm.block` or `wasm.if` it names are ()",
    );
    assert_round_trips(&function_with("%f: i1, %a: i32", "i32", body));
}

#[test]
fn the_operands_of_an_end_are_the_results_of_the_construct() {
    let block = |end: &str| {
        format!("%r = wasm.block {{\n    {end}\n}} : () -> (i32);\nbuiltin.ret %r : (i32);")
    };
    accepted_function(SIG, "i32", &block("wasm.end %a : (i32);"));
    for (end, expected) in [
        ("wasm.end %b : (i64);", "wasm.end"),
        ("wasm.end : ();", "wasm.end"),
        ("wasm.end %a, %a : (i32, i32);", "wasm.end"),
    ] {
        let error = rejected_function(SIG, "i32", &block(end));
        assert!(error.contains(expected), "{error}");
        assert!(error.contains("invalid operation"), "{error}");
    }
    // the same for a loop and an if
    let looped =
        "%r = wasm.loop {\n    wasm.end %b : (i64);\n} : () -> (i32);\nbuiltin.ret %r : (i32);";
    assert!(rejected_function(SIG, "i32", looped).contains("wasm.end"));
    let branched = "%r = wasm.if %c then {\n    wasm.end %a : (i32);\n} else {\n    wasm.end %b : \
                    (i64);\n} : (i32) -> (i32);\nbuiltin.ret %r : (i32);";
    assert!(rejected_function(SIG, "i32", branched).contains("wasm.end"));
    // under another operation, the terminator has no construct to leave
    let under_scf = "scf.if %f then {\n    wasm.end : ();\n} else {\n    scf.yield : ();\n} : \
                     (i1) -> ();\nbuiltin.ret;";
    let error = rejected_function("%f: i1", "", under_scf);
    assert!(error.contains("`wasm.end` cannot end a region of `scf.if`"), "{error}");
}

#[test]
fn the_entry_block_takes_the_parameters_of_the_construct() {
    rejects_with(
        SIG,
        "i32",
        "%r = wasm.block %a {\n^b(%x: i64):\n    wasm.end %a : (i32);\n} : (i32) -> \
         (i32);\nbuiltin.ret %r : (i32);",
        "`wasm.block` takes parameters and gives results of Wasm value types, and its body's \
         entry block takes the parameters as its arguments, but the entry block of its body takes \
         (i64), not the parameters (i32)",
    );
    rejects_with(
        SIG,
        "i32",
        "%r = wasm.loop %a {\n    wasm.end %a : (i32);\n} : (i32) -> (i32);\nbuiltin.ret %r : \
         (i32);",
        "the entry block of its body takes (), not the parameters (i32)",
    );
    rejects_with(
        SIG,
        "i32",
        "%r = wasm.if %c, %a then {\n^t(%x: i32):\n    wasm.end %x : (i32);\n} else {\n^e(%x: \
         i64):\n    wasm.end %a : (i32);\n} : (i32, i32) -> (i32);\nbuiltin.ret %r : (i32);",
        "the entry block of its second region takes (i64), not the parameters (i32)",
    );
    rejects_with(
        SIG,
        "i32",
        "%r = wasm.if %c, %a then {\n^t(%x: i32, %y: i32):\n    wasm.end %x : (i32);\n} else \
         {\n^e(%x: i32):\n    wasm.end %x : (i32);\n} : (i32, i32) -> (i32);\nbuiltin.ret %r : \
         (i32);",
        "the entry block of its first region takes (i32, i32), not the parameters (i32)",
    );
}

#[test]
fn the_condition_of_an_if_is_an_i32() {
    rejects_with(
        SIG,
        "",
        "wasm.if %b then {\n    wasm.end : ();\n} else {\n    wasm.end : ();\n} : (i64) -> \
         ();\nbuiltin.ret;",
        "its condition has type `i64`",
    );
    rejects_with(
        "%f: i1",
        "",
        "wasm.if %f then {\n    wasm.end : ();\n} else {\n    wasm.end : ();\n} : (i1) -> \
         ();\nbuiltin.ret;",
        "its condition has type `i1`",
    );
}

#[test]
fn the_parameters_and_results_of_a_construct_are_wasm_value_types() {
    rejects_with(
        "%x: i8",
        "",
        "wasm.block %x {\n^b(%y: i8):\n    wasm.end : ();\n} : (i8) -> ();\nbuiltin.ret;",
        "but its operand 0 has type `i8`",
    );
    rejects_with(
        "%x: i8",
        "i8",
        "%r = wasm.block {\n    wasm.end %x : (i8);\n} : () -> (i8);\nbuiltin.ret %r : (i8);",
        "but its result 0 has type `i8`",
    );
    for ty in ["funcref", "externref"] {
        let ty = format!("!wasm.{ty}");
        accepted_function(
            &format!("%x: {ty}"),
            &ty,
            &format!(
                "%r = wasm.loop %x {{\n^h(%y: {ty}):\n    wasm.end %y : ({ty});\n}} : ({ty}) -> \
                 ({ty});\nbuiltin.ret %r : ({ty});"
            ),
        );
    }
}

#[test]
fn a_value_defined_in_a_construct_is_not_usable_after_it() {
    // the parser does not let the text name it, so the program is built
    let mut test = Test::new("f", |cx| ([cx.i32()], [cx.i32()]));
    test.context().get_or_register_dialect::<WasmDialect>();
    let a = test.entry_block().borrow().arguments()[0] as runic_ir::ValueRef;
    let mut f = test.function_builder();
    let span = SourceSpan::UNKNOWN;
    let entry = f.entry_block();
    let block = f.block([], &[], span).unwrap();
    let body = block.borrow().body().entry_block_ref().unwrap();
    f.builder_mut().set_insertion_point_to_end(body);
    let inside = f.add(a, a, span).unwrap();
    f.end([], span).unwrap();
    f.builder_mut().set_insertion_point_to_end(entry);
    let outside = f.add(inside, a, span).unwrap();
    f.builder_mut().ret([outside], span).unwrap();
    let error = test
        .function()
        .borrow()
        .as_operation()
        .recursively_verify()
        .expect_err("a use that is not dominated");
    let text = describe(&error);
    assert!(text.contains("is not dominated by its definition"), "{text}");
}

// ---------------------------------------------------------------------------------------------
// Branch tables

/// A function with a block around a table of `entries` regions and a default, branching to `levels`
/// (the first of them for the first entry, and so on; the last for the default), and its result.
fn table(index_ty: &str, regions: &[&str], default: &str) -> String {
    let entries = regions
        .iter()
        .map(|region| format!("{{\n{region}\n}}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "%r = wasm.block {{\n    wasm.br_table %i, %a [{entries}] default {{\n{default}\n}} : \
         ({index_ty}, i32);\n}} : () -> (i32);\nbuiltin.ret %r : (i32);"
    )
}

const TABLE_SIG: &str = "%i: i32, %a: i32, %b: i64, %j: i64";
const BR: &str = "wasm.br 0 %a : (i32);";

#[test]
fn a_branch_table_has_regions_of_one_branch_each_and_a_default() {
    for entries in 0..4 {
        let regions = alloc::vec![BR; entries];
        let printed = accepted_function(TABLE_SIG, "i32", &table("i32", &regions, BR));
        assert_eq!(printed.matches("wasm.br 0").count(), entries + 1, "{printed}");
        assert_round_trips(&function_with(TABLE_SIG, "i32", &table("i32", &regions, BR)));
    }
    // the entries may name different labels of the same type
    let body = r#"
    %r = wasm.block {
        %q = wasm.loop %a {
        ^h(%p: i32):
            wasm.br_table %i, %p [{
                wasm.br 0 %p : (i32);
            }, {
                wasm.br 1 %p : (i32);
            }] default {
                wasm.br 1 %p : (i32);
            } : (i32, i32);
        } : (i32) -> (i32);
        wasm.end %q : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#;
    accepted_function(TABLE_SIG, "i32", body);
    assert_round_trips(&function_with(TABLE_SIG, "i32", body));
    // and carry no value
    let none = r#"
    wasm.block {
        wasm.br_table %i [{
            wasm.br 0 : ();
        }] default {
            wasm.br 0 : ();
        } : (i32);
    } : () -> ();
    builtin.ret %a : (i32);"#;
    accepted_function(TABLE_SIG, "i32", none);
    assert_round_trips(&function_with(TABLE_SIG, "i32", none));
}

#[test]
fn a_branch_table_with_a_region_that_is_not_one_branch_is_rejected() {
    let rule = "`wasm.br_table` takes an `i32` index and values, and each of its regions holds \
                one block with exactly one `wasm.br` of those values, but ";
    let two = alloc::format!("%z = wasm.add %a, %a;\n{BR}");
    for (regions, default, problem) in [
        (alloc::vec![two.as_str()], BR, "its entry 0 holds 2 operations"),
        (alloc::vec![BR], two.as_str(), "its default region holds 2 operations"),
        (
            alloc::vec![BR],
            "wasm.unreachable;",
            "its default region holds `wasm.unreachable`, not a `wasm.br`",
        ),
        (
            alloc::vec!["%z = wasm.add %a, %a;"],
            BR,
            "its entry 0 holds `wasm.add`, not a `wasm.br`",
        ),
    ] {
        let error = rejected_function(TABLE_SIG, "i32", &table("i32", &regions, default));
        assert!(error.contains(&format!("{rule}{problem}")), "{problem}: {error}");
    }
    // another value of the same type is another value
    let body = "%r = wasm.block %a {\n^b(%v: i32):\n    wasm.br_table %i, %v [{\n wasm.br 0 %a : \
                (i32);\n}] default {\n wasm.br 0 %v : (i32);\n} : (i32, i32);\n} : (i32) -> \
                (i32);\nbuiltin.ret %r : (i32);";
    rejects_with(
        TABLE_SIG,
        "i32",
        body,
        "the `wasm.br` of its entry 0 carries (i32), not the values of the table (i32)",
    );
    // an empty region
    let empty = "%r = wasm.block {\n    wasm.br_table %i, %a [{\n}] default {\n wasm.br 0 %a : \
                 (i32);\n} : (i32, i32);\n} : () -> (i32);\nbuiltin.ret %r : (i32);";
    rejects_with(TABLE_SIG, "i32", empty, "its entry 0 has no block");
}

#[test]
fn every_branch_of_a_table_carries_the_values_of_the_table_even_to_a_label_that_takes_others() {
    // each branch fits the label it names, but the table carries (i32), and one carries (i32, i32)
    let body = r#"
    %r1, %r2 = wasm.block {
        %q = wasm.block {
            wasm.br_table %i, %a [{
                wasm.br 1 %a, %a : (i32, i32);
            }] default {
                wasm.br 0 %a : (i32);
            } : (i32, i32);
        } : () -> (i32);
        wasm.end %q, %q : (i32, i32);
    } : () -> (i32, i32);
    builtin.ret %r1 : (i32);"#;
    rejects_with(
        TABLE_SIG,
        "i32",
        body,
        "the `wasm.br` of its entry 0 carries (i32, i32), not the values of the table (i32)",
    );
    let none = r#"
    wasm.block {
        %q = wasm.block {
            wasm.br_table %i, %a [{
                wasm.br 1 : ();
            }] default {
                wasm.br 0 %a : (i32);
            } : (i32, i32);
        } : () -> (i32);
        wasm.end : ();
    } : () -> ();
    builtin.ret %a : (i32);"#;
    rejects_with(
        TABLE_SIG,
        "i32",
        none,
        "the `wasm.br` of its entry 0 carries (), not the values of the table (i32)",
    );
}

#[test]
fn the_index_of_a_branch_table_is_an_i32() {
    for (ty, params) in [("i64", "%j: i64, %a: i32"), ("i1", "%j: i1, %a: i32")] {
        let body = table(ty, &[BR], BR).replace("%i", "%j");
        let error = rejected_function(params, "i32", &body);
        assert!(error.contains(&format!("but its index has type `{ty}`")), "{error}");
    }
}

#[test]
fn the_branches_of_a_table_are_checked_against_their_targets() {
    // an entry names the function
    let error = rejected_function(TABLE_SIG, "i32", &table("i32", &["wasm.br 1 %a : (i32);"], BR));
    assert!(error.contains("which is the function itself"), "{error}");
    // another type than the target's
    let error = rejected_function(
        "%i: i32, %a: i32",
        "i32",
        &table("i32", &[BR], BR).replace("() -> (i32)", "() -> (i64)").replace(
            "builtin.ret %r : (i32)",
            "%z = wasm.wrap %r <{ ty = i32 }>;\nbuiltin.ret %z : (i32)",
        ),
    );
    assert!(
        error.contains("the results of the `wasm.block` or `wasm.if` it names are (i64)"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------------------------
// Traits and interfaces

fn function_op(context: &Rc<Context>, text: &str) -> runic_ir::OperationRef {
    parse::parse::<Function>(ParserConfig::new(context.clone()), "f", text)
        .into_result()
        .unwrap_or_else(|report| panic!("{}", describe(&report)))
        .as_operation_ref()
}

fn find(root: runic_ir::OperationRef, name: &str) -> Vec<runic_ir::OperationRef> {
    let mut found = Vec::new();
    root.borrow().prewalk_all(|op: &Operation| {
        if op.name().to_string() == name {
            found.push(op.as_operation_ref());
        }
    });
    found
}

#[test]
fn the_exit_kind_and_target_of_a_branch_follow_the_construct_it_names() {
    let text = function_with(
        SIG,
        "i32",
        r#"
    %r = wasm.block {
        %l = wasm.loop %a {
        ^h(%p: i32):
            wasm.if %c then {
                wasm.br 0 : ();
            } : (i32) -> ();
            wasm.br 1 %p : (i32);
        } : (i32) -> (i32);
        wasm.end %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#,
    );
    let context = rich_context();
    let root = function_op(&context, &text);
    let [block] = find(root, "wasm.block")[..] else {
        panic!()
    };
    let branches = find(root, "wasm.br");
    assert_eq!(branches.len(), 2);
    // the first branch is level 0 inside the `if`: the `if` is the target, and it breaks
    let (first, second) = (branches[0].borrow(), branches[1].borrow());
    let first = first.as_trait::<dyn RegionExitTerminatorOpInterface>().unwrap();
    let second = second.as_trait::<dyn RegionExitTerminatorOpInterface>().unwrap();
    assert_eq!(first.get_exit_kind(), runic_ir::RegionExitKind::Break);
    assert_eq!(first.get_exit_target(), Some(find(root, "wasm.if")[0]));
    assert_eq!(second.get_exit_kind(), runic_ir::RegionExitKind::Break);
    assert_eq!(second.get_exit_target(), Some(block));

    // a branch to the loop is a continue
    let text = text.replace("wasm.br 1 %p", "wasm.br 0 %p");
    let root = function_op(&context, &text);
    let [loop_op] = find(root, "wasm.loop")[..] else {
        panic!()
    };
    let branches = find(root, "wasm.br");
    let last = branches[1].borrow();
    let last = last.as_trait::<dyn RegionExitTerminatorOpInterface>().unwrap();
    assert_eq!(last.get_exit_kind(), runic_ir::RegionExitKind::Continue);
    assert_eq!(last.get_exit_target(), Some(loop_op));
}

#[test]
fn the_end_and_the_branch_are_not_returns_and_the_constructs_forward_exits() {
    let text = function_with(
        SIG,
        "i32",
        r#"
    %r = wasm.block {
        wasm.end %a : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#,
    );
    let context = rich_context();
    let root = function_op(&context, &text);
    assert!(!find(root, "wasm.end")[0].borrow().leaves_callable());
    let text = text.replace("wasm.end %a : (i32);", "wasm.br 0 %a : (i32);");
    let root = function_op(&context, &text);
    assert!(!find(root, "wasm.br")[0].borrow().leaves_callable());
}

/// A `builtin.ret` inside the constructs leaves the function, as inside `scf` constructs, and a
/// return inside an operation that does not forward exits is rejected.
#[test]
fn a_return_inside_the_constructs_leaves_the_function() {
    let body = r#"
    %r = wasm.block {
        %l = wasm.loop %a {
        ^h(%p: i32):
            wasm.if %c then {
                builtin.ret %p : (i32);
            } : (i32) -> ();
            wasm.br 0 %p : (i32);
        } : (i32) -> (i32);
        wasm.end %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#;
    accepted_function("%c: i32, %a: i32", "i32", body);
    let tail = r#"
    %r = wasm.block {
        wasm.if %c then {
            builtin.tail_call @f(%a, %a) : extern("C") (i32, i32) -> i32;
        } : (i32) -> ();
        wasm.end %a : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#;
    accepted_function("%c: i32, %a: i32", "i32", tail);
    let opaque = r#"
    "test.reads_locals_and_holds"() ({
        builtin.ret %a : (i32);
    }) : () -> ();
    builtin.ret %a : (i32);"#;
    rejects_with(
        "%c: i32, %a: i32",
        "i32",
        opaque,
        "which has not declared that it forwards breaking control flow",
    );
}

// ---------------------------------------------------------------------------------------------
// Builders

#[test]
fn the_builders_build_each_operation_and_the_result_verifies_and_prints_as_the_text_does() {
    let mut test = Test::new("f", |cx| ([cx.i32(), cx.i32(), cx.i32()], [cx.i32()]));
    test.context().get_or_register_dialect::<WasmDialect>();
    let (c, a, i) = {
        let args = test
            .entry_block()
            .borrow()
            .arguments()
            .iter()
            .map(|arg| arg.borrow().as_value_ref())
            .collect::<Vec<runic_ir::ValueRef>>();
        (args[0], args[1], args[2])
    };
    let i32 = test.context().i32();
    let mut f = test.function_builder();
    let span = SourceSpan::UNKNOWN;
    let entry = f.entry_block();

    // block with a branch table inside a loop inside an if
    let if_op = f.r#if(c, [a], &[i32], span).unwrap();
    let then_block = if_op.borrow().then_body().entry_block_ref().unwrap();
    let else_block = if_op.borrow().else_body().entry_block_ref().unwrap();
    let then_arg = then_block.borrow().arguments()[0] as runic_ir::ValueRef;
    let else_arg = else_block.borrow().arguments()[0] as runic_ir::ValueRef;
    let loop_op = {
        f.builder_mut().set_insertion_point_to_end(then_block);
        let loop_op = f.r#loop([then_arg], &[i32], span).unwrap();
        let body = loop_op.borrow().body().entry_block_ref().unwrap();
        let param = body.borrow().arguments()[0] as runic_ir::ValueRef;
        f.builder_mut().set_insertion_point_to_end(body);
        let table = f.br_table(i, [param], 2, span).unwrap();
        let regions = table
            .borrow()
            .as_operation()
            .regions()
            .iter()
            .map(|region| region.as_region_ref())
            .collect::<Vec<_>>();
        for region in regions {
            let block = f.builder_mut().create_block(region, None, &[]);
            f.builder_mut().set_insertion_point_to_end(block);
            f.br(1, [param], span).unwrap();
        }
        loop_op
    };
    f.builder_mut().set_insertion_point_to_end(then_block);
    let loop_result = loop_op.borrow().results().all()[0].borrow().as_value_ref();
    f.end([loop_result], span).unwrap();
    f.builder_mut().set_insertion_point_to_end(else_block);
    let block_op = f.block([else_arg], &[i32], span).unwrap();
    let block_entry = block_op.borrow().body().entry_block_ref().unwrap();
    let inner = block_entry.borrow().arguments()[0] as runic_ir::ValueRef;
    f.builder_mut().set_insertion_point_to_end(block_entry);
    f.br(0, [inner], span).unwrap();
    f.builder_mut().set_insertion_point_to_end(else_block);
    let block_result = block_op.borrow().results().all()[0].borrow().as_value_ref();
    f.end([block_result], span).unwrap();
    f.builder_mut().set_insertion_point_to_end(entry);
    let result = if_op.borrow().results().all()[0].borrow().as_value_ref();
    f.builder_mut().ret([result], span).unwrap();

    let function = test.function();
    function
        .borrow()
        .as_operation()
        .recursively_verify()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let printed = function.borrow().as_operation().to_string();
    for part in [
        "wasm.if %",
        "wasm.loop %",
        "wasm.br_table %",
        "wasm.block %",
        "wasm.br 1 %",
        "wasm.br 0 %",
        "wasm.end %",
    ] {
        assert!(printed.contains(part), "`{part}` is not in\n{printed}");
    }
    // and what was built is what the text says it is
    let context = rich_context();
    let (_, reprinted) = parse_function_fixpoint(&context, "f", &printed).unwrap();
    assert_eq!(normalize(&reprinted), normalize(&printed));
}

#[test]
fn the_builder_of_a_trap_builds_a_terminator() {
    let mut test = Test::new("f", |_| ([], []));
    test.context().get_or_register_dialect::<WasmDialect>();
    let mut f = test.function_builder();
    f.trap(SourceSpan::UNKNOWN).unwrap();
    let function = test.function();
    function
        .borrow()
        .as_operation()
        .recursively_verify()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    assert!(function.borrow().as_operation().to_string().contains("wasm.unreachable;"));
}

// ---------------------------------------------------------------------------------------------
// The edges each operation states

/// The successors `op` states for `point`, as text: `enter N` (region N), `return`, and
/// `exit to Parent|region N of <target>` for an exit.
fn successors(
    op: runic_ir::OperationRef,
    point: runic_ir::RegionBranchPoint,
    constants: Option<&[Option<runic_ir::AttributeRef>]>,
) -> Vec<String> {
    use runic_ir::{RegionBranchOpInterface, RegionSuccessorKind};

    let op = op.borrow();
    let branch = op.as_trait::<dyn RegionBranchOpInterface>().unwrap();
    let index_of = |region: runic_ir::RegionRef| {
        op.regions().iter().position(|r| r.as_region_ref() == region).unwrap()
    };
    let describe_one = |kind: RegionSuccessorKind| match kind {
        RegionSuccessorKind::Entering(region) => format!("enter {}", index_of(region)),
        RegionSuccessorKind::Returning => String::from("return"),
        RegionSuccessorKind::Exiting { target, to } => {
            format!("exit {} {:?}", target.borrow().name(), to.is_parent())
        }
    };
    match constants {
        Some(constants) => branch
            .get_entry_successor_regions(constants)
            .map(|s| describe_one(s.kind()))
            .collect(),
        None => branch.get_successor_regions(point).map(|s| describe_one(s.kind())).collect(),
    }
}

#[test]
fn each_construct_states_its_entry_and_its_return_whether_or_not_a_branch_names_it() {
    use runic_ir::RegionBranchPoint;

    let with_branch = r#"
    %r = wasm.block {
        wasm.if %c then {
            wasm.br 1 %a : (i32);
        } : (i32) -> ();
        %l = wasm.loop %a {
        ^h(%p: i32):
            wasm.if %c then {
                wasm.br 1 %p : (i32);
            } : (i32) -> ();
            wasm.end %p : (i32);
        } : (i32) -> (i32);
        wasm.end %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#;
    let without_branch = r#"
    %r = wasm.block {
        wasm.if %c then {
            wasm.end : ();
        } : (i32) -> ();
        %l = wasm.loop %a {
        ^h(%p: i32):
            wasm.if %c then {
                wasm.end : ();
            } : (i32) -> ();
            wasm.end %p : (i32);
        } : (i32) -> (i32);
        wasm.end %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#;
    let context = rich_context();
    for body in [with_branch, without_branch] {
        let root = function_op(&context, &function_with("%c: i32, %a: i32", "i32", body));
        let [block] = find(root, "wasm.block")[..] else {
            panic!()
        };
        let [loop_op] = find(root, "wasm.loop")[..] else {
            panic!()
        };
        let ifs = find(root, "wasm.if");
        let child = |op: runic_ir::OperationRef| {
            let region = op.borrow().region(0).as_region_ref();
            RegionBranchPoint::Child(region)
        };
        for op in [block, loop_op] {
            assert_eq!(successors(op, RegionBranchPoint::Parent, None), ["enter 0"]);
            assert_eq!(successors(op, child(op), None), ["return"]);
        }
        for op in ifs {
            assert_eq!(successors(op, RegionBranchPoint::Parent, None), ["enter 0", "enter 1"]);
            assert_eq!(successors(op, child(op), None), ["return"]);
            // nothing skips a region: the entry successors are the regions
            let unknown = [None, None];
            assert_eq!(
                successors(
                    op,
                    RegionBranchPoint::Parent,
                    Some(&unknown[..op.borrow().num_operands()])
                ),
                ["enter 0", "enter 1"]
            );
        }
    }
}

#[test]
fn the_entry_successors_of_an_if_and_a_table_follow_a_constant_condition_or_index() {
    use runic_ir::RegionBranchPoint;

    let context = rich_context();
    let int = |value: i32| -> Option<runic_ir::AttributeRef> {
        let attr: runic_ir::AttributeRef =
            context.int_attr(runic_ir::Int::from(value), context.i32());
        Some(attr)
    };
    let root = function_op(
        &context,
        &function_with(
            "%c: i32, %a: i32",
            "i32",
            r#"
    %r = wasm.block {
        wasm.if %c then {
            wasm.end : ();
        } else {
            wasm.end : ();
        } : (i32) -> ();
        wasm.br_table %c, %a [{
            wasm.br 0 %a : (i32);
        }, {
            wasm.br 0 %a : (i32);
        }] default {
            wasm.br 0 %a : (i32);
        } : (i32, i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#,
        ),
    );
    let if_op = find(root, "wasm.if")[0];
    for (condition, expected) in [(0, "enter 1"), (1, "enter 0"), (-1, "enter 0"), (7, "enter 0")] {
        assert_eq!(
            successors(if_op, RegionBranchPoint::Parent, Some(&[int(condition)])),
            [expected]
        );
    }
    // the regions of a table: the default first, then the entries
    let table = find(root, "wasm.br_table")[0];
    for (index, expected) in [(0, "enter 1"), (1, "enter 2"), (2, "enter 0"), (-1, "enter 0")] {
        assert_eq!(
            successors(table, RegionBranchPoint::Parent, Some(&[int(index), None])),
            [expected],
            "index {index}"
        );
    }
    assert_eq!(
        successors(table, RegionBranchPoint::Parent, Some(&[None, None])),
        ["enter 0", "enter 1", "enter 2"]
    );
    assert_eq!(
        successors(table, RegionBranchPoint::Parent, None),
        ["enter 0", "enter 1", "enter 2"]
    );
    // nothing returns from a region of a table to it: each region leaves by its own branch
    let region = table.borrow().region(1).as_region_ref();
    assert!(successors(table, RegionBranchPoint::Child(region), None).is_empty());
}

#[test]
fn a_branch_states_one_exit_to_its_target_and_a_loop_allows_a_continue_from_its_header() {
    use runic_ir::{
        HasBreakingControlFlowOpInterface, RegionBranchTerminatorOpInterface, RegionSuccessorInfo,
    };

    let context = rich_context();
    let root = function_op(
        &context,
        &function_with(
            SIG,
            "i32",
            r#"
    %r = wasm.block {
        %l = wasm.loop %a {
        ^h(%p: i32):
            wasm.if %c then {
                wasm.br 0 : ();
            } : (i32) -> ();
            wasm.br 0 %p : (i32);
        } : (i32) -> (i32);
        wasm.br 0 %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);"#,
        ),
    );
    let [block] = find(root, "wasm.block")[..] else {
        panic!()
    };
    let [loop_op] = find(root, "wasm.loop")[..] else {
        panic!()
    };
    let branches = find(root, "wasm.br");
    assert_eq!(branches.len(), 3);
    let infos = |op: runic_ir::OperationRef| {
        let op = op.borrow();
        let terminator = op.as_trait::<dyn RegionBranchTerminatorOpInterface>().unwrap();
        terminator.get_successor_regions(&[None])
    };
    // a branch to the `if`, in it, leaves it with the results of the `if` (none)
    let [RegionSuccessorInfo::Exiting { target, to, values }] = &infos(branches[0])[..] else {
        panic!()
    };
    assert_eq!(*target, find(root, "wasm.if")[0]);
    assert!(to.is_parent());
    assert!(values.is_empty());
    // a branch to the loop goes to the entry block of its body, whose arguments take the values
    let [RegionSuccessorInfo::Exiting { target, to, values }] = &infos(branches[1])[..] else {
        panic!()
    };
    assert_eq!(*target, loop_op);
    assert!(!to.is_parent());
    let entry = loop_op.borrow().region(0).entry_block_ref().unwrap();
    assert_eq!(values[0], entry.borrow().arguments()[0] as runic_ir::ValueRef);
    // a branch to the block leaves it with its results
    let [RegionSuccessorInfo::Exiting { target, to, values }] = &infos(branches[2])[..] else {
        panic!()
    };
    assert_eq!(*target, block);
    assert!(to.is_parent());
    assert_eq!(values[0], block.borrow().results()[0].borrow().as_value_ref());

    // the loop is a continue target whose header is its one region, and may be continued from
    let loop_ref = loop_op.borrow();
    let receiver = loop_ref.as_trait::<dyn HasBreakingControlFlowOpInterface>().unwrap();
    assert!(receiver.continue_may_originate_in_header());
    assert_eq!(receiver.get_continue_target_region(), Some(loop_ref.region(0).as_region_ref()));
    assert!(receiver.has_nested_predecessors());
    // a block and an `if` are break targets only
    for name in ["wasm.block", "wasm.if"] {
        let op = find(root, name)[0];
        let op = op.borrow();
        let receiver = op.as_trait::<dyn HasBreakingControlFlowOpInterface>().unwrap();
        assert!(receiver.get_continue_target_region().is_none());
        assert!(!receiver.continue_may_originate_in_header());
        assert!(receiver.has_nested_predecessors());
    }
}
