//! The three verification states, each with input the verifier rejects and with a pass that
//! breaks the IR, and how a failed pass is reported.

use runic_ir::{
    EntityMut, Op, Operation, OperationName, Report,
    dialects::builtin::Function,
    pass::{Pass, PassExecutionState, PostPassStatus, registry::register_pass},
};

use super::{INVALID, LOCALS, opt};

/// The tool's `test-fail` at each function, with the message these tests expect.
pub(super) const FAIL: &str =
    r#"builtin.module(builtin.function(test-fail{message="this pass always fails"}))"#;

/// A pass that leaves invalid IR: it removes the operands of every `builtin.ret`.
#[derive(Default)]
struct BreakReturns;

impl Pass for BreakReturns {
    type Target = Function;

    fn name(&self) -> &'static str {
        "break returns"
    }

    fn argument(&self) -> &'static str {
        "test-break-returns"
    }

    fn description(&self) -> &'static str {
        "Removes the operands of every return"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Function>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let function = op.as_operation_ref();
        drop(op);
        let mut rets = Vec::new();
        function.borrow().prewalk_all(|op: &Operation| {
            if op.name().to_string() == "builtin.ret" {
                rets.push(op.as_operation_ref());
            }
        });
        for mut ret in rets {
            ret.borrow_mut().set_operands(core::iter::empty());
        }
        state.set_post_pass_status(PostPassStatus::Changed);
        Ok(())
    }
}

/// Registers the passes of this module (again, which does nothing, when a test before did).
pub(super) fn register() {
    register_pass(|| Box::new(BreakReturns)).unwrap();
}

#[test]
fn a_panic_in_a_pass_keeps_what_the_chunk_wrote_before_it() {
    let output = opt(&["-p", "test-panic", "--print-ir-before", "all"], LOCALS);
    output.failed();
    assert_eq!(output.stdout, "");
    let header = "// ----- IR before test-panic on builtin.module -----\n";
    assert!(output.stderr.starts_with(header), "{}", output.stderr);
    assert!(output.stderr.contains("builtin.load_local 0"), "{}", output.stderr);
    let error = "error: internal error: the test-panic pass panicked\n";
    let report = output.stderr.find(error);
    let report = report.unwrap_or_else(|| panic!("the report is there: {}", output.stderr));
    // The print comes before the error.
    assert!(output.stderr[..report].contains("load_local"), "{}", output.stderr);
    // The report says where the panic happened, and nothing follows it.
    let location = "the panic happened at tools/runic-opt/src/test_passes/panic.rs:";
    let note = output.stderr[report + error.len()..].strip_prefix(" |\n = note: ");
    let note = note.unwrap_or_else(|| panic!("a note follows: {}", output.stderr));
    assert!(note.starts_with(location), "{}", output.stderr);
    assert_eq!(note.lines().count(), 1, "{}", output.stderr);
    assert!(note.ends_with('\n'), "{}", output.stderr);
}

#[test]
fn by_default_invalid_input_is_rejected_by_the_parser() {
    for args in [&[][..], &["-p", "cse", "--print-ir-before", "cse"][..]] {
        let output = opt(args, INVALID);
        output.failed();
        assert_eq!(output.stdout, "");
        assert!(
            output.stderr.starts_with("error: invalid operation builtin.function"),
            "{}",
            output.stderr
        );
        // The parser rejected it: no pass started.
        assert!(!output.stderr.contains("// ----- IR before"), "{}", output.stderr);
    }
}

#[test]
fn by_default_a_pass_that_breaks_the_ir_is_reported_by_its_name() {
    register();
    let output = opt(&["-p", "test-break-returns,cse", "--print-ir-before", "cse"], LOCALS);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: invalid operation builtin.function"),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains(
            "context: the IR that pass 'test-break-returns' left on 'builtin.function' @f is \
             invalid"
        ),
        "{}",
        output.stderr
    );
    // The pipeline stops at the failure.
    assert!(!output.stderr.contains("IR before cse"), "{}", output.stderr);
}

#[test]
fn with_verify_each_false_invalid_input_is_rejected_by_the_parser() {
    let output = opt(&["--verify-each=false", "-p", "cse", "--print-ir-before", "cse"], INVALID);
    output.failed();
    assert!(
        output.stderr.starts_with("error: invalid operation builtin.function"),
        "{}",
        output.stderr
    );
    // The parser rejected it: no pass started.
    assert!(!output.stderr.contains("// ----- IR before"), "{}", output.stderr);
}

#[test]
fn with_verify_each_false_a_pass_that_breaks_the_ir_is_caught_by_the_final_check() {
    register();
    let output = opt(
        &["--verify-each=false", "-p", "test-break-returns,cse", "--print-ir-after", "cse"],
        LOCALS,
    );
    output.failed();
    assert_eq!(output.stdout, "");
    // Every pass ran: the pass manager did not verify.
    assert!(
        output.stderr.starts_with("// ----- IR after cse on builtin.module -----"),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains("error: invalid operation builtin.function"),
        "{}",
        output.stderr
    );
    assert!(!output.stderr.contains("test-break-returns"), "{}", output.stderr);
    // `--verify-each` alone is `--verify-each=true`.
    let output = opt(&["--verify-each", "-p", "test-break-returns"], LOCALS);
    output.failed();
    assert!(output.stderr.contains("pass 'test-break-returns' left"), "{}", output.stderr);
}

#[test]
fn with_no_verify_invalid_input_and_broken_ir_are_printed_as_they_are() {
    register();
    let output = opt(&["--no-verify"], INVALID);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert!(output.stdout.contains("builtin.ret;"), "{}", output.stdout);
    let output = opt(&["--no-verify", "-p", "cse"], INVALID);
    output.succeeded();
    assert!(output.stdout.contains("builtin.ret;"), "{}", output.stdout);
    let output = opt(&["--no-verify", "-p", "test-break-returns"], LOCALS);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert!(output.stdout.contains("builtin.ret;"), "{}", output.stdout);
}

#[test]
fn a_failed_pass_is_named_with_the_operation_it_ran_on() {
    register();
    let output = opt(&["-p", FAIL], LOCALS);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.starts_with("error: this pass always fails"), "{}", output.stderr);
    assert!(
        output
            .stderr
            .contains("context: pass 'test-fail' failed on 'builtin.function' @f"),
        "{}",
        output.stderr
    );
}

#[test]
fn print_ir_after_failure_prints_the_operation_the_failed_pass_ran_on() {
    register();
    let output = opt(&["-p", FAIL, "--print-ir-after-failure"], LOCALS);
    output.failed();
    let header = "// ----- IR after failed test-fail on builtin.function -----\n";
    assert!(output.stderr.starts_with(header), "{}", output.stderr);
    assert!(output.stderr.contains("@f(%0: i32)"), "{}", output.stderr);
    assert!(output.stderr.contains("error: this pass always fails"), "{}", output.stderr);
    // A failure of the verifier after a pass is a failure of the pass.
    let output = opt(&["-p", "test-break-returns", "--print-ir-after-failure"], LOCALS);
    output.failed();
    let header = "// ----- IR after failed test-break-returns on builtin.function -----\n";
    assert!(output.stderr.starts_with(header), "{}", output.stderr);
    // The function is printed on its own, as the pass left it.
    assert!(output.stderr.contains("\n    builtin.ret;\n};\n"), "{}", output.stderr);
}

/// An exit whose loop, or a structured operation whose region, has no block: the loop's (or the
/// operation's) own verifier reports the empty region, in both forms, and nothing panics.
#[test]
fn an_exit_in_a_loop_with_an_empty_region_is_a_diagnostic() {
    let exits = r#"builtin.function public extern("C") @f(%n: i32, %c: i1) -> i32 {
    %8 = scf.while %n before {
    } after {
    ^b3(%6: i32):
        %7 = scf.if %c then {
            scf.continue 1 %6 : (i32);
        } else {
            scf.break 1 %6 : (i32);
        } : (i1) -> (i32);
        scf.yield %7 : (i32);
    } : (i32) -> (i32);
    builtin.ret %8 : (i32);
};
"#;
    let condition = r#"builtin.function public extern("C") @f(%n: i32, %c: i1) -> i32 {
    %8 = scf.while %n before {
    ^b(%i: i32):
        %7 = scf.if %c then {
            scf.break 1 %i : (i32);
        } else {
            scf.yield %i : (i32);
        } : (i1) -> (i32);
        scf.condition %c, %7 : (i1, i32);
    } after {
    } : (i32) -> (i32);
    builtin.ret %8 : (i32);
};
"#;
    let yields = r#"builtin.function public extern("C") @f(%n: i32, %c: i1) -> i32 {
    %r = scf.if %c then {
    } else {
        scf.yield %n : (i32);
    } : (i1) -> (i32);
    builtin.ret %r : (i32);
};
"#;
    for (text, region) in [
        (exits, "region 0 of scf.while"),
        (condition, "region 1 of scf.while"),
        (yields, "region 0 of scf.if"),
    ] {
        let generic = opt(&["--no-verify", "--print-generic"], text);
        generic.succeeded();
        for input in [text, generic.stdout.as_str()] {
            let output = opt(&[], input);
            output.failed();
            assert!(!output.stderr.contains("internal error"), "{}", output.stderr);
            assert!(output.stderr.contains("<stdin>:"), "{}", output.stderr);
            assert!(
                output.stderr.contains(&format!("successor 0 ({region}) has no entry block")),
                "{}",
                output.stderr
            );
        }
    }
}
