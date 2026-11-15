//! The built binary, run as a process: what only a process shows (its own streams, the logger, a
//! panic that reaches the process's panic hook).

use std::{
    io::Write,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

const BIN: &str = env!("CARGO_BIN_EXE_runic-opt");

/// How long a run may take before the test says it hangs.
const LIMIT: Duration = Duration::from_secs(60);

/// Runs the binary with `args`, `stdin` and the environment variables `env`; fails the test when
/// it has not exited after [`LIMIT`].
fn opt(args: &[&str], stdin: &str, env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(BIN);
    command
        .args(args)
        .env_remove("RUNIC_TRACE")
        .env_remove("RUST_BACKTRACE")
        .env_remove("RUST_LIB_BACKTRACE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in env {
        command.env(key, value);
    }
    let mut child = command.spawn().expect("the binary starts");
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let start = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if start.elapsed() > LIMIT {
            let _ = child.kill();
            let _ = child.wait();
            panic!("runic-opt {args:?} with {env:?} has not exited after {LIMIT:?}");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// A module whose function computes `a + b` twice and adds the two.
const REPEATED_ADD: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32, %b: i32) -> i32 {
        %x = arith.add %a, %b;
        %y = arith.add %a, %b;
        %z = arith.add %x, %y;
        builtin.ret %z : (i32);
    };
};
"#;

#[test]
fn input_read_from_a_pipe_is_printed_to_a_pipe() {
    let output = opt(&[], REPEATED_ADD, &[]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(text(&output.stderr), "");
    let printed = text(&output.stdout);
    assert!(printed.starts_with("builtin.module public @m {"), "{printed}");
    // What was printed reads back to the same text.
    let again = opt(&["-"], &printed, &[]);
    assert!(again.status.success(), "{}", text(&again.stderr));
    assert_eq!(text(&again.stdout), printed);
}

#[test]
fn runic_trace_that_logs_does_not_hang_the_tool() {
    let output = opt(&["-p", "cse"], REPEATED_ADD, &[("RUNIC_TRACE", "trace")]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    // The log went to standard error; standard output holds only the IR.
    assert!(!output.stderr.is_empty());
    let quiet = opt(&["-p", "cse"], REPEATED_ADD, &[]);
    assert!(quiet.status.success(), "{}", text(&quiet.stderr));
    assert_eq!(text(&output.stdout), text(&quiet.stdout));
}

/// The location of the panic in the `test-panic` pass, as the report gives it.
const LOCATION: &str = "the panic happened at tools/runic-opt/src/test_passes/panic.rs:";

#[test]
fn a_panic_is_reported_by_the_tool_alone() {
    let output = opt(&["-p", "test-panic", "--print-ir-before", "all"], REPEATED_ADD, &[]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(text(&output.stdout), "");
    let stderr = text(&output.stderr);
    // What the chunk wrote before the panic, then the report, and nothing else: the default
    // hook's "thread .. panicked at" is not there.
    let header = "// ----- IR before test-panic on builtin.module -----\n";
    assert!(stderr.starts_with(header), "{stderr}");
    let report = stderr.find("error: internal error: the test-panic pass panicked\n");
    let report = report.unwrap_or_else(|| panic!("the report is there: {stderr}"));
    assert!(stderr[..report].contains("arith.add"), "{stderr}");
    assert!(stderr[report..].contains(LOCATION), "{stderr}");
    assert!(!stderr.contains("panicked at"), "{stderr}");
    assert!(!stderr.contains("stack backtrace"), "{stderr}");
}

#[test]
fn a_backtrace_is_in_the_report_when_asked_for() {
    let output = opt(&["-p", "test-panic"], REPEATED_ADD, &[("RUST_BACKTRACE", "1")]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = text(&output.stderr);
    assert!(
        stderr.starts_with("error: internal error: the test-panic pass panicked\n"),
        "{stderr}"
    );
    assert!(stderr.contains(LOCATION), "{stderr}");
    assert!(stderr.contains("stack backtrace:\n"), "{stderr}");
    assert!(!stderr.contains("panicked at"), "{stderr}");
}

#[test]
fn a_panic_that_cannot_unwind_keeps_what_the_chunk_wrote() {
    let output = opt(
        &["-p", "test-panic{cannot-unwind}", "--print-ir-before", "all"],
        REPEATED_ADD,
        &[],
    );
    // The process aborts.
    assert!(!output.status.success());
    assert_ne!(output.status.code(), Some(1));
    let stderr = text(&output.stderr);
    let header = "// ----- IR before test-panic on builtin.module -----\n";
    assert!(stderr.starts_with(header), "{stderr}");
    let report = stderr.find("error: internal error: the test-panic pass panicked\n");
    let report = report.unwrap_or_else(|| panic!("the report is there: {stderr}"));
    assert!(stderr[..report].contains("arith.add"), "{stderr}");
    assert!(stderr[report..].contains(LOCATION), "{stderr}");
}

#[test]
fn test_panic_is_not_listed() {
    let output = opt(&["--list-passes"], "", &[]);
    assert!(output.status.success());
    let listed = text(&output.stdout);
    assert!(listed.contains("test-fail"), "{listed}");
    assert!(!listed.contains("test-panic"), "{listed}");
}

/// A character the lexer rejects, where one of a set of keywords is expected, is the parser's
/// diagnostic at that line, not an internal error.
#[test]
fn a_lexer_error_where_a_keyword_is_expected_is_a_diagnostic() {
    for source in [
        "builtin.module \u{e9} @m {\n};\n",
        "builtin.module \"public @m {\n};\n",
        "builtin.module % @m {\n};\n",
    ] {
        let output = opt(&[], source, &[]);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{source:?}: {stderr}");
        assert!(!stderr.contains("internal error"), "{source:?}: {stderr}");
        assert!(stderr.contains("<stdin>:1:"), "{source:?}: {stderr}");
    }
}

/// An expectation in a case that holds only comments is an error, not dropped with the case.
#[test]
fn expectations_in_a_case_with_no_operation_are_an_error() {
    let source = "// expected-error {{never produced}}\n// -----\nbuiltin.module public @m {\n};\n";
    let output = opt(&["--split-input-file", "--verify-diagnostics"], source, &[]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stdout));
    assert!(text(&output.stderr).contains("an expectation in a case with no operation"));
}

/// A diagnostic located by `loc(..)` in another file, read from the working directory, does not
/// meet an expectation on the line of the same number in the input.
#[test]
fn a_diagnostic_in_another_file_does_not_meet_an_expectation_by_its_line_number() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("other.txt"), "first\nthis returns\nthird\n").unwrap();
    let source = "builtin.module public @m {\n// expected-error {{returns}}\n    builtin.function \
                  public extern(\"C\") @f() -> i32 {\n        builtin.ret loc(\"other.txt\":2:1 \
                  to 2:5);\n    };\n};\n";
    let mut command = Command::new(BIN);
    command
        .arg("--verify-diagnostics")
        .current_dir(dir.path())
        .env_remove("RUNIC_TRACE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    child.stdin.take().unwrap().write_all(source.as_bytes()).unwrap();
    let output = child.wait_with_output().unwrap();
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("unexpected error: "), "{stderr}");
    assert!(stderr.contains("other.txt:2:1"), "{stderr}");
    assert!(stderr.contains("was not produced"), "{stderr}");
}
