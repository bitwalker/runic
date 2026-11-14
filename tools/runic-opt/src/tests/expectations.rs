//! `--verify-diagnostics`: the diagnostics of each case are checked against the `expected-<kind>`
//! comments in it.

use runic_core::{
    EntityMut, Op, OperationName, Report,
    diagnostics::diagnostic,
    dialects::builtin::Function,
    pass::{Pass, PassExecutionState, registry::register_pass},
};

use super::{LOCALS, opt, verification};

/// A pass that emits, on the context, a warning and a hint at the function it runs on.
#[derive(Default)]
struct EmitWarningAndHint;

impl Pass for EmitWarningAndHint {
    type Target = Function;

    fn name(&self) -> &'static str {
        "emit a warning and a hint"
    }

    fn argument(&self) -> &'static str {
        "test-emit-warning-and-hint"
    }

    fn description(&self) -> &'static str {
        "Emits a warning and a hint at each function"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Function>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let span = op.as_operation().span;
        let context = op.as_operation().context_rc();
        drop(op);
        let _ = context.emit(diagnostic!(
            severity: Warning,
            message: "a warning from the test pass",
            labels: [primary(span, "at this function")],
        ));
        let _ = context.emit(diagnostic!(
            severity: Hint,
            message: ("a hint about {{braces}}"),
            labels: [primary(span)],
        ));
        Ok(())
    }
}

fn register() {
    register_pass(|| Box::new(EmitWarningAndHint)).unwrap();
    verification::register();
}

const CHECK: &str = "--verify-diagnostics";

/// A function whose return lacks its `;`: the parser reports `}` on line 4, column 5.
const MISSING_SEMICOLON: &str = "builtin.module public @m {
    builtin.function public extern(\"C\") @f() {
        builtin.ret
    };
};
";

/// [`MISSING_SEMICOLON`] with `comment` inserted as line `line` (counted from 1).
fn with_line(text: &str, line: usize, comment: &str) -> String {
    let mut lines = text.lines().map(String::from).collect::<Vec<_>>();
    lines.insert(line - 1, String::from(comment));
    lines.join("\n") + "\n"
}

#[test]
fn a_parse_error_expected_on_its_line_above_and_below() {
    let on_its_line = MISSING_SEMICOLON
        .replace("    };\n};", "    }; // expected-error {{unexpected token '}'}}\n};");
    let below = with_line(MISSING_SEMICOLON, 4, "// expected-error@below {{invalid syntax}}");
    let above = with_line(MISSING_SEMICOLON, 5, "    // expected-error@above {{invalid syntax}}");
    let plus = with_line(MISSING_SEMICOLON, 3, "// expected-error@+2 {{invalid syntax}}");
    let minus = with_line(MISSING_SEMICOLON, 5, "// expected-error@-1 {{invalid syntax}}");
    for text in [on_its_line, below, above, plus, minus] {
        let output = opt(&[CHECK], &text);
        output.succeeded();
        assert_eq!(output.stderr, "", "{text}");
        // The case failed, so nothing is printed.
        assert_eq!(output.stdout, "", "{text}");
    }
}

#[test]
fn a_verifier_error_is_matched_by_its_message_or_its_label() {
    let text = "builtin.module public @m {
    builtin.function public extern(\"C\") @f() -> i32 {
        // expected-error@+1 {{invalid operation builtin.function}}
        builtin.ret; // expected-error {{but the function returns (i32)}}
    };
};
";
    // One diagnostic, two expectations: one of them is left over.
    let output = opt(&[CHECK], text);
    output.failed();
    assert!(
        output
            .stderr
            .contains("expected error \"but the function returns (i32)\" was not produced")
            ^ output
                .stderr
                .contains("expected error \"invalid operation builtin.function\" was not produced"),
        "{}",
        output.stderr
    );
    for expectation in [
        "// expected-error@+1 {{invalid operation builtin.function}}",
        "// expected-error@+1 {{this returns (), but the function returns (i32)}}",
    ] {
        let text = format!(
            "builtin.module public @m {{
    builtin.function public extern(\"C\") @f() -> i32 {{
        {expectation}
        builtin.ret;
    }};
}};
"
        );
        let output = opt(&[CHECK], &text);
        output.succeeded();
        assert_eq!(output.stderr, "");
        assert_eq!(output.stdout, "");
    }
}

#[test]
fn a_pass_failure_is_matched_by_its_message_or_its_context() {
    register();
    for expectation in
        ["this pass always fails", "pass 'test-fail' failed on 'builtin.function' @f"]
    {
        // The function is on the third line.
        let text = format!("// expected-error@+2 {{{{{expectation}}}}}\n{LOCALS}");
        let output = opt(&[CHECK, "-p", verification::FAIL], &text);
        output.succeeded();
        assert_eq!(output.stderr, "", "{expectation}");
        assert_eq!(output.stdout, "", "{expectation}");
    }
    // A verifier failure after a pass is the pass's.
    let text = LOCALS.replace(
        "builtin.ret %v : (i32);",
        "builtin.ret %v : (i32); // expected-error {{but the function returns (i32)}}",
    );
    let output = opt(&[CHECK, "-p", "test-break-returns"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
}

/// [`LOCALS`] with `comment` at the end of the function's line.
fn locals_with(comment: &str) -> String {
    LOCALS.replace("locals(i32) {", &format!("locals(i32) {{ {comment}"))
}

#[test]
fn a_warning_and_a_hint_from_a_pass_two_on_one_line() {
    register();
    let text = locals_with(
        "// expected-warning {{a warning from the test pass}} expected-hint {{a hint about}}",
    );
    let output = opt(&[CHECK, "-p", "test-emit-warning-and-hint"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    // The case succeeded: it is printed.
    assert_eq!(output.stdout, opt(&[], LOCALS).stdout);

    // The label's text is searched too, and the expectations may be on lines of their own: the
    // function is on line 5.
    let text = format!(
        "// expected-warning@+4 {{{{at this function}}}}\n// expected-hint@+3 \
         {{{{hint}}}}\n\n{LOCALS}"
    );
    let output = opt(&[CHECK, "-p", "test-emit-warning-and-hint"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");

    // `@above` and `@below` pass over lines that hold only expectations.
    let function = "    builtin.function public extern(\"C\") @f(%a: i32) -> i32 locals(i32) {\n";
    let text = LOCALS.replace(
        function,
        &format!(
            "// expected-warning@below {{{{warning}}}}\n// expected-hint@below \
             {{{{hint}}}}\n{function}// expected-warning@above {{{{warning}}}}\n// \
             expected-hint@above {{{{hint}}}}\n"
        ),
    );
    let output =
        opt(&[CHECK, "-p", "test-emit-warning-and-hint,test-emit-warning-and-hint"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
}

#[test]
fn an_expectation_matches_one_diagnostic_only() {
    register();
    let text = locals_with("// expected-warning {{warning}} expected-hint {{hint}}");
    let output =
        opt(&[CHECK, "-p", "test-emit-warning-and-hint,test-emit-warning-and-hint"], &text);
    output.failed();
    assert_eq!(
        output
            .stderr
            .matches("unexpected warning: a warning from the test pass")
            .count(),
        1,
        "{}",
        output.stderr
    );
    assert_eq!(
        output.stderr.matches("unexpected hint: a hint about").count(),
        1,
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("--> <stdin>:2:5"), "{}", output.stderr);
    // The case itself succeeded.
    assert!(output.stdout.contains("@f"), "{}", output.stdout);
}

#[test]
fn an_expectation_that_is_not_met_is_reported_at_the_comment() {
    let text = locals_with("// expected-error {{nothing like this}}");
    let output = opt(&[CHECK], &text);
    output.failed();
    assert!(
        output
            .stderr
            .starts_with("error: expected error \"nothing like this\" was not produced"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("--> <stdin>:2:"), "{}", output.stderr);
    assert!(
        output.stderr.contains("expected-error {{nothing like this}}"),
        "{}",
        output.stderr
    );
    assert!(output.stdout.contains("@f"), "{}", output.stdout);
}

#[test]
fn a_diagnostic_that_is_not_expected_is_reported_at_its_location() {
    let output = opt(&[CHECK], MISSING_SEMICOLON);
    output.failed();
    assert!(
        output.stderr.starts_with("error: unexpected error: invalid syntax"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("--> <stdin>:4:5"), "{}", output.stderr);
    assert_eq!(output.stdout, "");
    // A warning nobody expected is a mismatch too.
    register();
    let output = opt(&[CHECK, "-p", "test-emit-warning-and-hint"], LOCALS);
    output.failed();
    assert!(output.stderr.contains("unexpected warning: a warning from the test pass"));
    assert!(output.stderr.contains("unexpected hint: a hint about {{braces}}"));
}

#[test]
fn a_diagnostic_of_another_kind_does_not_match() {
    let text = with_line(MISSING_SEMICOLON, 4, "// expected-warning@below {{invalid syntax}}");
    let output = opt(&[CHECK], &text);
    output.failed();
    assert!(output.stderr.contains("unexpected error: invalid syntax"), "{}", output.stderr);
    assert!(
        output.stderr.contains("expected warning \"invalid syntax\" was not produced"),
        "{}",
        output.stderr
    );
    // Nor does one on another line.
    let text = with_line(MISSING_SEMICOLON, 4, "// expected-error {{invalid syntax}}");
    let output = opt(&[CHECK], &text);
    output.failed();
    assert!(output.stderr.contains("unexpected error: invalid syntax"), "{}", output.stderr);
}

#[test]
fn a_regular_expression() {
    for pattern in ["unexpected token '.'", "^invalid syn[a-z]+$"] {
        let text = with_line(
            MISSING_SEMICOLON,
            4,
            &format!("// expected-error-re@below {{{{{pattern}}}}}"),
        );
        let output = opt(&[CHECK], &text);
        output.succeeded();
        assert_eq!(output.stderr, "", "{pattern}");
    }
    // Without `-re` the text is not a pattern.
    let text = with_line(MISSING_SEMICOLON, 4, "// expected-error@below {{invalid syn[a-z]+}}");
    opt(&[CHECK], &text).failed();
}

#[test]
fn a_diagnostic_without_a_location_is_expected_at_unknown() {
    let world = "builtin.world {
    builtin.module public @m {
        builtin.function public extern(\"C\") @f() {
            builtin.ret;
        };
    };
};
";
    let expected = format!(
        "// expected-warning@unknown {{{{pass 'local2reg' ran on no operation}}}}\n{world}"
    );
    let output = opt(&[CHECK, "-p", "local2reg"], &expected);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert!(output.stdout.contains("builtin.world"), "{}", output.stdout);
    // Not expected, the warning is a mismatch.
    let output = opt(&[CHECK, "-p", "local2reg"], world);
    output.failed();
    assert!(
        output
            .stderr
            .starts_with("error: unexpected warning: pass 'local2reg' ran on no operation"),
        "{}",
        output.stderr
    );
    // An expectation of a line does not match a diagnostic without one.
    let text = format!("{world}// expected-warning@-3 {{{{ran on no operation}}}}\n");
    opt(&[CHECK, "-p", "local2reg"], &text).failed();
}

#[test]
fn with_split_input_file_expectations_are_per_case_at_the_files_lines() {
    let good = LOCALS;
    let expected_error =
        with_line(MISSING_SEMICOLON, 4, "// expected-error@below {{invalid syntax}}");
    let text = format!("{good}// -----\n{expected_error}// -----\n{MISSING_SEMICOLON}");
    let output = opt(&[CHECK, "--split-input-file"], &text);
    output.failed();
    // Only the third case's error is unexpected, at its line in the file.
    let line = good.lines().count() + 1 + expected_error.lines().count() + 1 + 4;
    assert_eq!(output.stderr.matches("unexpected error").count(), 1, "{}", output.stderr);
    assert!(output.stderr.contains(&format!("--> <stdin>:{line}:5")), "{}", output.stderr);
    assert!(!output.stderr.contains("was not produced"), "{}", output.stderr);
    // Only the first case succeeded.
    assert_eq!(output.stdout, opt(&[], good).stdout);

    // An expectation of one case is not met by a diagnostic of another.
    let text = format!(
        "{}// -----\n{MISSING_SEMICOLON}",
        locals_with("// expected-error@+2 {{invalid syntax}}")
    );
    let output = opt(&[CHECK, "--split-input-file"], &text);
    output.failed();
    assert!(output.stderr.contains("was not produced"), "{}", output.stderr);
    assert!(output.stderr.contains("unexpected error: invalid syntax"), "{}", output.stderr);
}

#[test]
fn braces_inside_the_text() {
    register();
    let text = locals_with("// expected-hint {{about {{braces}}}} expected-warning {{test pass}}");
    let output = opt(&[CHECK, "-p", "test-emit-warning-and-hint"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    let text = MISSING_SEMICOLON.replace("    };\n};", "    }; // expected-error {{'}'}}\n};");
    opt(&[CHECK], &text).succeeded();
}

#[test]
fn a_malformed_expectation_is_an_error() {
    for (comment, message) in [
        ("// expected-error@sideways {{x}}", "'@sideways'"),
        ("// expected-error no braces", "'{{'"),
        ("// expected-error {{unterminated", "'}}'"),
        ("// expected-error {{   }}", "no text"),
        ("// expected-error-re {{(}}", "regular expression"),
        ("// expected-note {{x}}", "'note'"),
        ("// expected-error@+100 {{x}}", "line"),
        ("// expected-error@-100 {{x}}", "line"),
    ] {
        let text = locals_with(comment);
        let output = opt(&[CHECK], &text);
        output.failed();
        assert!(output.stderr.starts_with("error: "), "{comment}: {}", output.stderr);
        assert!(output.stderr.contains(message), "{comment}: {}", output.stderr);
        assert!(output.stderr.contains("--> <stdin>:2:"), "{comment}: {}", output.stderr);
        // The case is not processed.
        assert_eq!(output.stdout, "", "{comment}");
    }
}

#[test]
fn a_word_that_ends_in_expected_is_not_an_expectation() {
    for comment in [
        "// unexpected-error {{x}}",
        "// not_expected-error {{x}}",
        "// expected-errors are written like this",
    ] {
        let output = opt(&[CHECK], &locals_with(comment));
        output.succeeded();
        assert_eq!(output.stderr, "", "{comment}");
    }
    // Nor is one in a string.
    let text = LOCALS.replace(
        "builtin.store_local 0, %a;",
        "builtin.store_local 0, %a;\n        cf.assert %c, \"// expected-error {{x}}\";",
    );
    let text = text.replace("@f(%a: i32)", "@f(%a: i32, %c: i1)");
    let output = opt(&[CHECK], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
}

#[test]
fn without_the_flag_expectations_are_comments() {
    let output = opt(&[], &locals_with("// expected-error {{nothing}}"));
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout, opt(&[], LOCALS).stdout);
    let text = with_line(MISSING_SEMICOLON, 4, "// expected-error@below {{invalid syntax}}");
    let output = opt(&[], &text);
    output.failed();
    assert!(output.stderr.starts_with("error: invalid syntax"), "{}", output.stderr);
}
