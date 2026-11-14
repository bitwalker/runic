//! The command line: usage errors and help, files, printing switches and the listings.

use std::{ffi::OsString, io, path::PathBuf, process::ExitCode};

use super::{REPEATED_ADD, opt};

#[test]
fn a_usage_error_goes_to_the_given_stderr() {
    let output = opt(&["--no-such-switch"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: unexpected argument '--no-such-switch'"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("Usage: runic-opt"), "{}", output.stderr);
    // A switch with an optional value takes it with `=` only: `false` is the input file, and the
    // second file is unexpected.
    let output = opt(&["--verify-each", "false", "x.runic"], "");
    output.failed();
    assert!(
        output.stderr.starts_with("error: unexpected argument 'x.runic'"),
        "{}",
        output.stderr
    );
    let output = opt(&["--verify-each=maybe"], "");
    output.failed();
    assert!(output.stderr.contains("invalid value 'maybe'"), "{}", output.stderr);
}

#[test]
fn help_goes_to_the_given_stdout() {
    let output = opt(&["--help"], "");
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert!(output.stdout.contains("Usage: runic-opt [OPTIONS] [FILE]"), "{}", output.stdout);
    for switch in [
        "--pass-pipeline",
        "--no-verify",
        "--verify-each",
        "--print-generic",
        "--print-locations",
        "--print-ir-before",
        "--print-ir-after",
        "--print-ir-after-change",
        "--print-ir-after-failure",
        "--dump-pass-pipeline",
        "--list-passes",
        "--list-dialects",
    ] {
        assert!(output.stdout.contains(switch), "{switch}: {}", output.stdout);
    }
    assert!(output.stdout.contains("1,000 deep"), "{}", output.stdout);
    assert!(output.stdout.contains("not guaranteed to parse back"), "{}", output.stdout);
}

/// A directory for files a test writes, removed when it is dropped.
fn scratch() -> tempfile::TempDir {
    let parent = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
    std::fs::create_dir_all(&parent).unwrap();
    tempfile::Builder::new().prefix("runic-opt-").tempdir_in(parent).unwrap()
}

#[test]
fn output_goes_to_the_file_named_by_o_and_input_comes_from_the_file_named() {
    let dir = scratch();
    let input = dir.path().join("input.runic");
    let output_file = dir.path().join("output.runic");
    std::fs::write(&input, REPEATED_ADD).unwrap();
    let output =
        opt(&["-p", "cse", input.to_str().unwrap(), "-o", output_file.to_str().unwrap()], "");
    output.succeeded();
    assert_eq!(output.stdout, "");
    assert_eq!(output.stderr, "");
    let written = std::fs::read_to_string(&output_file).unwrap();
    assert_eq!(written, opt(&["-p", "cse"], REPEATED_ADD).stdout);
}

#[test]
fn a_missing_input_file_is_an_error() {
    let dir = scratch();
    let missing = dir.path().join("missing.runic");
    let output = opt(&[missing.to_str().unwrap()], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.starts_with("error: could not read '"), "{}", output.stderr);
    assert!(output.stderr.contains("missing.runic"), "{}", output.stderr);
}

#[test]
fn print_generic_prints_the_generic_form() {
    let output = opt(&["--print-generic"], REPEATED_ADD);
    output.succeeded();
    assert!(output.stdout.starts_with("\"builtin.module\"()"), "{}", output.stdout);
    assert!(output.stdout.contains("\"arith.add\"("), "{}", output.stdout);
}

#[test]
fn print_ir_after_writes_the_header_and_the_ir_to_stderr() {
    let output = opt(&["-p", "cse", "--print-ir-after", "cse"], REPEATED_ADD);
    output.succeeded();
    let header = "// ----- IR after cse on builtin.module -----\n";
    assert!(output.stderr.starts_with(header), "{}", output.stderr);
    // The IR printed after the pass is the result.
    assert_eq!(&output.stderr[header.len()..], format!("{}\n", output.stdout));
    let before = opt(&["-p", "cse", "--print-ir-before=all"], REPEATED_ADD);
    before.succeeded();
    assert!(before.stderr.starts_with("// ----- IR before cse on builtin.module -----\n"));
}

#[test]
fn print_ir_switches_naming_an_unregistered_pass_are_errors_that_list_similar_names() {
    for switch in ["--print-ir-before", "--print-ir-after"] {
        let output = opt(&["-p", "cse", switch, "cse,dcee"], REPEATED_ADD);
        output.failed();
        assert_eq!(output.stdout, "");
        assert_eq!(
            output.stderr,
            format!("error: {switch}: unknown pass 'dcee'; did you mean 'dce'?\n")
        );
    }
}

#[test]
fn list_passes_holds_every_pass_with_its_description_sorted() {
    let output = opt(&["--list-passes"], "not read");
    output.succeeded();
    assert_eq!(output.stderr, "");
    let lines = output.stdout.lines().collect::<Vec<_>>();
    let names = lines
        .iter()
        .map(|line| line.split_whitespace().next().unwrap())
        .collect::<Vec<_>>();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
    for pass in [
        "canonicalizer",
        "cfg-to-scf",
        "control-flow-sink",
        "cse",
        "dce",
        "local2reg",
        "sccp",
        "sink-operand-defs",
    ] {
        let line = lines
            .iter()
            .find(|line| line.split_whitespace().next() == Some(pass))
            .unwrap_or_else(|| panic!("{pass} is listed: {}", output.stdout));
        let description = line[pass.len()..].trim();
        assert!(!description.is_empty(), "{pass} has a description");
        assert_eq!(
            description,
            runic_core::pass::PassInfo::lookup(pass).unwrap().description(),
            "{pass}"
        );
    }
}

#[test]
fn list_dialects_holds_every_linked_dialect_with_its_operation_count() {
    let output = opt(&["--list-dialects"], "not read");
    output.succeeded();
    for dialect in ["builtin", "di", "test", "arith", "cf", "mem", "scf", "ub"] {
        let line = output
            .stdout
            .lines()
            .find(|line| line.split_whitespace().next() == Some(dialect))
            .unwrap_or_else(|| panic!("{dialect} is listed: {}", output.stdout));
        assert!(line.ends_with(" operations") || line.ends_with(" operation"), "{line}");
    }
}

#[test]
fn print_ir_after_change_prints_after_the_passes_that_changed_the_ir() {
    let output = opt(&["-p", "cse,cse", "--print-ir-after-change"], REPEATED_ADD);
    output.succeeded();
    // The first `cse` removes an addition; the second finds nothing to remove.
    assert_eq!(
        output.stderr.matches("// ----- IR after cse on builtin.module -----").count(),
        1
    );
}

/// A standard output that fails every write.
struct Broken;

impl io::Write for Broken {
    fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("the stream is closed"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("the stream is closed"))
    }
}

#[test]
fn a_standard_output_that_fails_to_write_is_an_error_on_stderr() {
    let mut input = REPEATED_ADD.as_bytes();
    let mut stderr = Vec::new();
    let code = crate::run(std::iter::empty::<OsString>(), &mut input, &mut Broken, &mut stderr);
    assert_eq!(code, ExitCode::FAILURE);
    let stderr = String::from_utf8(stderr).unwrap();
    assert_eq!(stderr, "error: could not write standard output: the stream is closed\n");
}

#[test]
fn short_help_states_the_region_depth_limit() {
    let output = opt(&["-h"], "");
    output.succeeded();
    assert!(output.stdout.contains("1,000 deep"), "{}", output.stdout);
}
