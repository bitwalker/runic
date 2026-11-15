//! The built binary, run as a process: what only a process shows (its own streams, the logger, a
//! panic that reaches the process's panic hook).

use std::{
    io::{Read, Write},
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

const BIN: &str = env!("CARGO_BIN_EXE_runic-opt");

/// The variables the logger reads, its own and those by which a terminal's colours are forced or
/// disabled: removed from every run, so that a run sees only those its test sets.
const TRACE_VARIABLES: [&str; 7] = [
    "RUNIC_TRACE",
    "RUNIC_TRACE_FILTER",
    "RUNIC_TRACE_STYLE",
    "RUNIC_TRACE_TIMING",
    "CLICOLOR",
    "CLICOLOR_FORCE",
    "NO_COLOR",
];

/// How long a run may take before the test says it hangs.
const LIMIT: Duration = Duration::from_secs(60);

/// Runs the binary with `args`, `stdin` and the environment variables `env`; fails the test when
/// it has not exited after [`LIMIT`]. Its standard output and error are read while it runs, so
/// that a run which writes more than a pipe holds does not block on the pipe.
fn opt(args: &[&str], stdin: &str, env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(BIN);
    command
        .args(args)
        .env_remove("RUST_BACKTRACE")
        .env_remove("RUST_LIB_BACKTRACE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for variable in TRACE_VARIABLES {
        command.env_remove(variable);
    }
    for (key, value) in env {
        command.env(key, value);
    }
    let mut child = command.spawn().expect("the binary starts");
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            pipe.read_to_end(&mut bytes).unwrap();
            bytes
        })
    };
    let stdout = drain(Box::new(child.stdout.take().unwrap()));
    let stderr = drain(Box::new(child.stderr.take().unwrap()));
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return Output {
                status,
                stdout: stdout.join().unwrap(),
                stderr: stderr.join().unwrap(),
            };
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

/// What the canonicalizer's pattern driver logs at `trace`: some of its records carry the
/// key-value pairs `dialect` and `op` of the operation they are about.
const DRIVER_TRACE: (&str, &str) = ("RUNIC_TRACE", "pattern-rewrite-driver=trace");

#[test]
fn runic_trace_filter_drops_the_records_whose_key_value_it_does_not_match() {
    const KEYED: &str = "op-specific patterns";
    let all = opt(&["-p", "canonicalizer"], REPEATED_ADD, &[DRIVER_TRACE]);
    assert!(all.status.success(), "{}", text(&all.stderr));
    assert!(text(&all.stderr).contains(KEYED), "{}", text(&all.stderr));
    // A filter on `op` that matches the operation keeps its records.
    let kept = opt(
        &["-p", "canonicalizer"],
        REPEATED_ADD,
        &[DRIVER_TRACE, ("RUNIC_TRACE_FILTER", "op:add")],
    );
    assert!(kept.status.success(), "{}", text(&kept.stderr));
    assert!(text(&kept.stderr).contains(KEYED), "{}", text(&kept.stderr));
    // One that matches nothing drops every record with an `op`, and only those.
    let dropped = opt(
        &["-p", "canonicalizer"],
        REPEATED_ADD,
        &[DRIVER_TRACE, ("RUNIC_TRACE_FILTER", "op:nothing")],
    );
    assert!(dropped.status.success(), "{}", text(&dropped.stderr));
    assert!(!text(&dropped.stderr).contains(KEYED), "{}", text(&dropped.stderr));
    assert!(!dropped.stderr.is_empty());
    assert_eq!(text(&dropped.stdout), text(&all.stdout));
}

#[test]
fn an_empty_runic_trace_filter_filters_nothing() {
    let output = opt(&["-p", "cse"], REPEATED_ADD, &[("RUNIC_TRACE_FILTER", "")]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(text(&output.stderr), "");
    assert!(!output.stdout.is_empty());
    let unset = opt(&["-p", "cse"], REPEATED_ADD, &[]);
    assert_eq!(text(&output.stdout), text(&unset.stdout));
}

#[test]
fn a_malformed_runic_trace_filter_is_an_error() {
    // A filter that is not `key:value`, and one whose value is not a regular expression: an error
    // that names the variable, and nothing is run.
    let not_key_value = opt(&["-p", "cse"], REPEATED_ADD, &[("RUNIC_TRACE_FILTER", "op")]);
    assert_eq!(not_key_value.status.code(), Some(1));
    assert_eq!(
        text(&not_key_value.stderr),
        "error: invalid RUNIC_TRACE_FILTER: expected filters of the form \
         `key:value[,key2:value2,..]`, got 'op'\n"
    );
    assert!(not_key_value.stdout.is_empty());
    let bad_pattern = opt(&["-p", "cse"], REPEATED_ADD, &[("RUNIC_TRACE_FILTER", "op:(")]);
    assert_eq!(bad_pattern.status.code(), Some(1));
    let stderr = text(&bad_pattern.stderr);
    assert!(
        stderr.starts_with("error: invalid RUNIC_TRACE_FILTER: invalid pattern '(' of key 'op': "),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(bad_pattern.stdout.is_empty());
}

#[test]
fn help_is_given_whatever_the_trace_variables_hold() {
    // The logger is set up once the arguments are read: a variable it cannot use does not stand
    // between the user and the help text.
    let help = opt(&["--help"], "", &[("RUNIC_TRACE_FILTER", "op")]);
    assert_eq!(help.status.code(), Some(0), "{}", text(&help.stderr));
    assert!(text(&help.stdout).contains("Usage:"), "{}", text(&help.stdout));
    assert!(help.stderr.is_empty(), "{}", text(&help.stderr));
    // Arguments that are wrong are reported as such, not as the variable's error.
    let usage = opt(&["--no-such-option"], "", &[("RUNIC_TRACE_FILTER", "op")]);
    assert_eq!(usage.status.code(), Some(1));
    assert!(!text(&usage.stderr).contains("RUNIC_TRACE_FILTER"), "{}", text(&usage.stderr));
}

#[test]
fn runic_trace_style_selects_whether_records_are_styled() {
    let styled = |style: Option<&str>, forced: bool| {
        let mut env = vec![DRIVER_TRACE];
        env.extend(style.map(|style| ("RUNIC_TRACE_STYLE", style)));
        if forced {
            env.push(("CLICOLOR_FORCE", "1"));
        }
        let output = opt(&["-p", "canonicalizer"], REPEATED_ADD, &env);
        assert!(output.status.success(), "{}", text(&output.stderr));
        assert!(!output.stderr.is_empty());
        output.stderr.contains(&0x1b)
    };
    // Standard error is a pipe: `auto`, the default, writes no styles to it, and `always` does.
    assert!(!styled(None, false));
    assert!(styled(Some("always"), false));
    // With colours forced, `auto` writes styles to the pipe, and `never` does not.
    assert!(styled(None, true));
    assert!(!styled(Some("never"), true));
}

#[test]
fn runic_trace_timing_gives_records_a_timestamp_of_its_precision() {
    let first_record = |timing: Option<&str>| {
        let mut env = vec![DRIVER_TRACE];
        env.extend(timing.map(|timing| ("RUNIC_TRACE_TIMING", timing)));
        let output = opt(&["-p", "canonicalizer"], REPEATED_ADD, &env);
        assert!(output.status.success(), "{}", text(&output.stderr));
        text(&output.stderr).lines().next().unwrap().to_string()
    };
    // Without it, a record starts with its level.
    let plain = first_record(None);
    assert!(plain.starts_with("[DEBUG ") || plain.starts_with("[TRACE "), "{plain}");
    // With it, a record starts with a timestamp of that precision: `[YYYY-MM-DDTHH:MM:SS.mmmZ`.
    let timed = first_record(Some("ms"));
    let stamp = timed.strip_prefix('[').and_then(|rest| rest.split_once(' ')).unwrap().0;
    assert!(
        stamp.len() == "2000-01-01T00:00:00.000Z".len() && stamp.ends_with('Z'),
        "{timed}"
    );
    // Anything else is an error, and nothing is run.
    let wrong = opt(&["-p", "canonicalizer"], REPEATED_ADD, &[("RUNIC_TRACE_TIMING", "hours")]);
    assert_eq!(wrong.status.code(), Some(1));
    assert_eq!(
        text(&wrong.stderr),
        "error: invalid RUNIC_TRACE_TIMING precision, expected one of [s, ms, us, ns], got \
         'hours'\n"
    );
    assert!(wrong.stdout.is_empty());
}

/// The location of the panic in the `test-panic` pass, as the report gives it.
const LOCATION: &str = "the panic happened at tools/runic-opt/src/test_passes/panic.rs:";

/// The first line of the report of the `test-panic` pass's panic.
const REPORT: &str = "error: internal error: the test-panic pass panicked\n";

/// Whether `text` is the whole report of one panic in the `test-panic` pass: the message, then the
/// note with its location, and nothing after it.
fn is_the_whole_report(text: &str) -> bool {
    let Some(rest) = text.strip_prefix(REPORT) else {
        return false;
    };
    let Some(rest) =
        rest.strip_prefix(" |\n = note: ").and_then(|rest| rest.strip_prefix(LOCATION))
    else {
        return false;
    };
    let Some(position) = rest.strip_suffix('\n') else {
        return false;
    };
    let mut parts = position.split(':');
    parts.clone().count() == 2
        && parts.all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

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
    let report = stderr.find(REPORT);
    let report = report.unwrap_or_else(|| panic!("the report is there: {stderr}"));
    assert!(stderr[..report].contains("arith.add"), "{stderr}");
    assert!(is_the_whole_report(&stderr[report..]), "{stderr}");
    assert!(!stderr.contains("panicked at"), "{stderr}");
}

#[test]
fn a_backtrace_is_in_the_report_when_asked_for() {
    let output = opt(&["-p", "test-panic"], REPEATED_ADD, &[("RUST_BACKTRACE", "1")]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = text(&output.stderr);
    assert!(stderr.starts_with(REPORT), "{stderr}");
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
    let report = stderr.find(REPORT);
    let report = report.unwrap_or_else(|| panic!("the report is there: {stderr}"));
    assert!(stderr[..report].contains("arith.add"), "{stderr}");
    assert!(stderr[report..].contains(LOCATION), "{stderr}");
}

/// A panic that cannot unwind in a later case keeps what the cases before it wrote.
#[test]
fn a_panic_that_cannot_unwind_keeps_what_earlier_cases_wrote() {
    let source =
        "builtin.module public @a {\n  bogus\n};\n// -----\nbuiltin.module public @b {\n};\n";
    let output = opt(&["--split-input-file", "-p", "test-panic{cannot-unwind}"], source, &[]);
    assert!(!output.status.success());
    assert_ne!(output.status.code(), Some(1));
    let stderr = text(&output.stderr);
    // The first case's error, then the second case's report.
    assert!(stderr.starts_with("error: invalid operation\n"), "{stderr}");
    let report = stderr.find(REPORT);
    let report = report.unwrap_or_else(|| panic!("the report is there: {stderr}"));
    assert!(stderr[..report].contains("bogus"), "{stderr}");
    assert!(stderr[report..].contains(LOCATION), "{stderr}");
}

/// A destructor that panics while the panic unwinds aborts the process; the report names the
/// first panic, and the destructor's after it.
#[test]
fn the_first_panic_is_reported_when_a_destructor_panics_during_unwinding() {
    let output = opt(&["-p", "test-panic{destructor}"], REPEATED_ADD, &[]);
    assert!(!output.status.success());
    assert_ne!(output.status.code(), Some(1));
    let stderr = text(&output.stderr);
    assert!(stderr.starts_with(REPORT), "{stderr}");
    let first = stderr.find(LOCATION).unwrap_or_else(|| panic!("{stderr}"));
    let destructor = "then a panic at tools/runic-opt/src/test_passes/panic.rs:";
    let then = stderr.find(destructor).unwrap_or_else(|| panic!("{stderr}"));
    assert!(first < then, "{stderr}");
    assert!(
        stderr[then..].contains(": a destructor of the test-panic pass panicked\n"),
        "{stderr}"
    );
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
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for variable in TRACE_VARIABLES {
        command.env_remove(variable);
    }
    let mut child = command.spawn().unwrap();
    child.stdin.take().unwrap().write_all(source.as_bytes()).unwrap();
    let output = child.wait_with_output().unwrap();
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("unexpected error: "), "{stderr}");
    assert!(stderr.contains("other.txt:2:1"), "{stderr}");
    assert!(stderr.contains("was not produced"), "{stderr}");
}
