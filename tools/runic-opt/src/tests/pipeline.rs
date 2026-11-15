//! The pipeline named on the command line: how its text is read, what an error in it reports,
//! and the warning for a pass that ran on nothing.

use runic_core::{
    EntityMut, Op, OperationName, Report,
    diagnostics::diagnostic,
    dialects::builtin::Function,
    pass::{Pass, PassExecutionState, PostPassStatus, registry::register_pass},
};

use super::{LOCALS, REPEATED_ADD, opt};

/// A pass that emits an error at the function it runs on and then succeeds.
#[derive(Default)]
struct EmitError;

impl Pass for EmitError {
    type Target = Function;

    fn name(&self) -> &'static str {
        "emit an error"
    }

    fn argument(&self) -> &'static str {
        "test-emit-error"
    }

    fn description(&self) -> &'static str {
        "Emits an error at each function and succeeds"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Function>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let span = op.as_operation().span;
        let context = op.as_operation().context_rc();
        drop(op);
        let _ = context.emit(diagnostic!(
            severity: Error,
            message: ("an error the pass emitted"),
            labels: [primary(span)],
        ));
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}

/// A pass that fails with an error that has no location.
#[derive(Default)]
struct FailNowhere;

impl Pass for FailNowhere {
    type Target = Function;

    fn name(&self) -> &'static str {
        "fail nowhere"
    }

    fn argument(&self) -> &'static str {
        "test-fail-nowhere"
    }

    fn description(&self) -> &'static str {
        "Fails at each function with an error that has no location"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Function>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        Err(Report::msg("a failure without a location"))
    }
}

fn register() {
    register_pass(|| Box::new(EmitError)).unwrap();
    register_pass(|| Box::new(FailNowhere)).unwrap();
}

/// What `-p pipeline` prints for `text`, which must succeed without a diagnostic.
#[track_caller]
fn optimized(pipeline: &str, text: &str) -> String {
    let output = opt(&["-p", pipeline], text);
    output.succeeded();
    assert_eq!(output.stderr, "", "{pipeline}");
    output.stdout
}

/// A module whose function keeps its argument in a local and adds it to itself twice.
pub(super) const LOCALS_AND_REPEATS: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 locals(i32) {
        builtin.store_local 0, %a;
        %v = builtin.load_local 0 : i32;
        %x = arith.add %v, %v;
        %y = arith.add %v, %v;
        %z = arith.add %x, %y;
        builtin.ret %z : (i32);
    };
};
"#;

#[test]
fn an_anchored_pipeline_a_list_and_an_any_pipeline_run_on_a_module() {
    let anchored = optimized("builtin.module(builtin.function(local2reg),cse)", LOCALS_AND_REPEATS);
    assert!(!anchored.contains("load_local"), "{anchored}");
    assert_eq!(anchored.matches("arith.add").count(), 2, "{anchored}");
    // The same passes as a list of elements anchored on the module: the typed pass is nested
    // under the functions directly inside it.
    assert_eq!(optimized("local2reg,cse", LOCALS_AND_REPEATS), anchored);
    let any = optimized("any(cse)", REPEATED_ADD);
    assert_eq!(any.matches("arith.add").count(), 2, "{any}");
}

#[test]
fn an_anchor_that_is_not_the_top_level_operation_is_an_error_that_names_both() {
    let output = opt(&["-p", "builtin.world(cse)"], REPEATED_ADD);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with(
            "error: the pass pipeline is anchored on 'builtin.world', but the input's top-level \
             operation is 'builtin.module'"
        ),
        "{}",
        output.stderr
    );
}

#[test]
fn text_that_starts_with_an_anchor_is_a_list_unless_it_is_one_whole_pipeline() {
    // A list of elements whose first is a nested pipeline: anchored on the module.
    let list = optimized("builtin.function(dce),cse", REPEATED_ADD);
    assert_eq!(list.matches("arith.add").count(), 2, "{list}");
    // One whole pipeline anchored on a function: an error for a module...
    let output = opt(&["-p", "builtin.function(dce)"], REPEATED_ADD);
    output.failed();
    assert!(output.stderr.contains("anchored on 'builtin.function'"), "{}", output.stderr);
    assert!(output.stderr.contains("operation is 'builtin.module'"), "{}", output.stderr);
    // ...and the pipeline of a function.
    let function = "builtin.function public extern(\"C\") @f(%a: i32) -> i32 {\n    %x = \
                    arith.add %a, %a;\n    builtin.ret %a : (i32);\n};\n";
    let dce = optimized("builtin.function(dce)", function);
    assert!(!dce.contains("arith.add"), "{dce}");
}

#[test]
fn an_unknown_pass_is_an_error_that_lists_a_similar_name() {
    let output = opt(&["-p", "builtin.module(csee)"], REPEATED_ADD);
    output.failed();
    assert_eq!(output.stdout, "");
    assert_eq!(output.stderr, "error: unknown pass 'csee'; did you mean 'cse'?\n");
}

#[test]
fn invalid_options_are_an_error() {
    let output = opt(&["-p", "cse{bogus=1}"], REPEATED_ADD);
    output.failed();
    assert_eq!(output.stderr, "error: pass 'cse' takes no options\n");
    let output = opt(&["-p", "cse{"], REPEATED_ADD);
    output.failed();
    assert!(output.stderr.starts_with("error: invalid pass pipeline: "), "{}", output.stderr);
}

#[test]
fn an_unknown_canonicalization_pattern_is_reported_when_the_pipeline_runs_before_any_pass() {
    let output = opt(
        &[
            "-p",
            "cse,canonicalizer{enable-patterns=no-such-pattern}",
            "--dump-pass-pipeline",
            "--print-ir-before",
            "all",
        ],
        REPEATED_ADD,
    );
    output.failed();
    assert_eq!(output.stdout, "");
    // The pipeline loads (it is dumped), and fails before any pass, `cse` included, runs.
    assert!(
        output.stderr.starts_with("builtin.module(cse,canonicalizer{"),
        "{}",
        output.stderr
    );
    assert!(!output.stderr.contains("// ----- IR before"), "{}", output.stderr);
    assert!(
        output.stderr.contains("unknown canonicalization pattern 'no-such-pattern'"),
        "{}",
        output.stderr
    );
}

/// A world whose function is inside a module, out of the reach of implicit nesting from it.
pub(super) const WORLD: &str = r#"builtin.world {
    builtin.module public @m {
        builtin.function public extern("C") @f(%a: i32) -> i32 locals(i32) {
            builtin.store_local 0, %a;
            %v = builtin.load_local 0 : i32;
            builtin.ret %v : (i32);
        };
    };
};
"#;

#[test]
fn a_pass_that_ran_on_no_operation_is_a_warning_that_says_recursive() {
    let output = opt(&["-p", "local2reg"], WORLD);
    output.succeeded();
    assert!(output.stdout.contains("load_local"), "nothing changed: {}", output.stdout);
    assert!(
        output.stderr.starts_with("warning: pass 'local2reg' ran on no operation"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("recursive("), "{}", output.stderr);
    assert_eq!(output.stderr.matches("warning:").count(), 1, "{}", output.stderr);
    // With `recursive(..)` it runs, and nothing is said.
    let reached = optimized("recursive(builtin.function(local2reg))", WORLD);
    assert!(!reached.contains("load_local"), "{reached}");
    // A pass that ran is not warned about.
    assert_eq!(opt(&["-p", "builtin.module(local2reg)"], LOCALS).stderr, "");
    // The help names the pass's own operation.
    assert!(
        output.stderr.contains("'recursive(builtin.function(local2reg))'"),
        "{}",
        output.stderr
    );
}

#[test]
fn a_pass_on_any_operation_that_ran_on_none_is_a_warning_without_recursive() {
    // No function directly inside the world: the `dce` nested under them runs on nothing.
    let output = opt(&["-p", "builtin.function(dce),cse"], WORLD);
    output.succeeded();
    assert!(
        output.stderr.starts_with("warning: pass 'dce' ran on no operation"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("runs on any operation"), "{}", output.stderr);
    assert!(!output.stderr.contains("recursive("), "{}", output.stderr);
    assert_eq!(output.stderr.matches("warning:").count(), 1, "{}", output.stderr);
}

#[test]
fn a_pass_named_twice_that_ran_on_nothing_is_warned_about_once() {
    let output = opt(&["-p", "local2reg,cse,local2reg"], WORLD);
    output.succeeded();
    assert_eq!(output.stderr.matches("warning:").count(), 1, "{}", output.stderr);
}

#[test]
fn dump_pass_pipeline_prints_the_loaded_pipeline_to_stderr_before_running_it() {
    let output = opt(
        &["-p", "builtin.function(local2reg), cse", "--dump-pass-pipeline"],
        LOCALS_AND_REPEATS,
    );
    output.succeeded();
    assert_eq!(output.stderr, "builtin.module(builtin.function(local2reg),cse)\n");
    assert!(!output.stdout.contains("builtin.module("), "{}", output.stdout);
}

#[test]
fn dump_pass_pipeline_without_a_pipeline_prints_the_empty_one_as_the_pass_manager_does() {
    use std::rc::Rc;

    use runic_core::{
        Context,
        parse::{ParserConfig, parse_any},
        pass::{Nesting, PassPipeline},
    };

    let output = opt(&["--dump-pass-pipeline"], LOCALS);
    output.succeeded();
    let context = Rc::new(Context::default());
    let op = parse_any(ParserConfig::new(context.clone()), "api.runic", LOCALS).unwrap();
    let pm = PassPipeline::parse_elements(&op.borrow().name(), "")
        .unwrap()
        .load(context.clone(), Nesting::Implicit)
        .unwrap();
    struct Textual<'a>(&'a runic_core::pass::PassManager);
    impl core::fmt::Display for Textual<'_> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.0.print_as_textual_pipeline(f)
        }
    }
    assert_eq!(output.stderr, format!("{}\n", Textual(&pm)));
}

#[test]
fn an_error_a_pass_emits_fails_the_case() {
    register();
    let output = opt(&["-p", "test-emit-error"], LOCALS);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: an error the pass emitted\n"),
        "{}",
        output.stderr
    );
    // Expected, it is matched; the case still failed, so it is not printed.
    let text =
        LOCALS.replace("locals(i32) {", "locals(i32) { // expected-error {{the pass emitted}}");
    let output = opt(&["--verify-diagnostics", "-p", "test-emit-error"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout, "");
}

#[test]
fn a_pass_error_without_a_location_is_expected_at_unknown() {
    register();
    let text = format!("// expected-error@unknown {{{{a failure without a location}}}}\n{LOCALS}");
    let output = opt(&["--verify-diagnostics", "-p", "test-fail-nowhere"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    // On a line, it does not match.
    let text = format!("// expected-error@+2 {{{{a failure without a location}}}}\n{LOCALS}");
    opt(&["--verify-diagnostics", "-p", "test-fail-nowhere"], &text).failed();
}

/// The pipeline is checked once, before the input is read: one error for a file of three cases,
/// whether or not they parse, and none about the input.
#[test]
fn a_pipeline_error_is_reported_once_before_the_input_is_read() {
    let three = format!("{LOCALS}// -----\n{LOCALS}// -----\nbuiltin.module {{\n");
    for pipeline in ["cze", "builtin.module(cse", "test-fail{reason=x}", "cse,cze"] {
        let output = opt(&["--split-input-file", "-p", pipeline], &three);
        output.failed();
        assert_eq!(output.stdout, "");
        assert_eq!(output.stderr.matches("error: ").count(), 1, "{pipeline}: {}", output.stderr);
        assert!(!output.stderr.contains("<stdin>"), "{pipeline}: {}", output.stderr);
    }
    let output = opt(&["--split-input-file", "-p", "cze"], &three);
    assert!(output.stderr.starts_with("error: unknown pass 'cze'"), "{}", output.stderr);
    // Nor is a missing input read.
    let output = opt(&["-p", "cze", "no/such/file.runic"], "");
    output.failed();
    assert!(output.stderr.starts_with("error: unknown pass 'cze'"), "{}", output.stderr);
    assert!(!output.stderr.contains("could not read"), "{}", output.stderr);
}
