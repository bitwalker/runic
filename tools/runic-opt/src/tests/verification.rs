//! The three verification states, each with input the verifier rejects and with a pass that
//! breaks the IR, and how a failed pass is reported.

use runic_core::{
    EntityMut, Op, Operation, OperationName, Report,
    dialects::builtin::Function,
    pass::{Pass, PassExecutionState, PostPassStatus, registry::register_pass},
};

use super::{INVALID, LOCALS, opt};

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

/// A pass that fails.
#[derive(Default)]
struct Fail;

impl Pass for Fail {
    type Target = Function;

    fn name(&self) -> &'static str {
        "fail"
    }

    fn argument(&self) -> &'static str {
        "test-fail"
    }

    fn description(&self) -> &'static str {
        "Fails"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Function>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        Err(Report::msg("this pass always fails"))
    }
}

/// A pass that panics.
#[derive(Default)]
struct Panic;

impl Pass for Panic {
    type Target = Function;

    fn name(&self) -> &'static str {
        "panic"
    }

    fn argument(&self) -> &'static str {
        "test-panic"
    }

    fn description(&self) -> &'static str {
        "Panics"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Function>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        panic!("the test pass panicked")
    }
}

/// Registers the passes of this module (again, which does nothing, when a test before did).
pub(super) fn register() {
    register_pass(|| Box::new(BreakReturns)).unwrap();
    register_pass(|| Box::new(Fail)).unwrap();
    register_pass(|| Box::new(Panic)).unwrap();
}

#[test]
fn a_panic_in_a_pass_keeps_what_the_chunk_wrote_before_it() {
    register();
    let output = opt(&["-p", "test-panic", "--print-ir-before", "all"], LOCALS);
    output.failed();
    assert_eq!(output.stdout, "");
    let header = "// ----- IR before test-panic on builtin.function -----\n";
    assert!(output.stderr.starts_with(header), "{}", output.stderr);
    assert!(output.stderr.contains("builtin.load_local 0"), "{}", output.stderr);
    let error = "error: internal error: the test pass panicked\n";
    assert!(output.stderr.ends_with(error), "{}", output.stderr);
    // The print comes before the error.
    assert!(output.stderr.find("load_local") < output.stderr.find(error));
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
    let output = opt(&["-p", "test-fail"], LOCALS);
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
    let output = opt(&["-p", "test-fail", "--print-ir-after-failure"], LOCALS);
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
