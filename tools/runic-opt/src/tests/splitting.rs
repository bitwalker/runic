//! `--split-input-file`: several cases in one file.

use super::{INVALID, opt};

/// A module with a function `@f` that returns `value`.
fn case(value: u32) -> String {
    format!(
        "builtin.module public @m {{\n    builtin.function public extern(\"C\") @f() -> i32 {{\n        \
         %x = arith.constant {value} : i32;\n        builtin.ret %x : (i32);\n    }};\n}};\n"
    )
}

const MARKER: &str = "// -----";

#[test]
fn three_cases_print_as_three_separated_by_the_marker() {
    let text = format!("{}{MARKER}\n{}{MARKER}\n{}", case(1), case(2), case(3));
    let output = opt(&["--split-input-file"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    let single = |n| opt(&[], &case(n)).stdout;
    assert_eq!(
        output.stdout,
        format!("{}{MARKER}\n{}{MARKER}\n{}", single(1), single(2), single(3))
    );
    assert!(!output.stdout.starts_with(MARKER));
    assert!(!output.stdout.trim_end().ends_with(MARKER));
}

#[test]
fn the_same_value_and_symbol_names_in_two_cases_do_not_clash() {
    let text = format!("{}{MARKER}\n{}", case(7), case(7));
    let output = opt(&["--split-input-file", "-p", "cse"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout.matches("@f").count(), 2, "{}", output.stdout);
}

#[test]
fn a_parse_error_in_the_second_case_is_at_its_line_in_the_file() {
    let bad =
        "builtin.module public @m {\n    builtin.function public extern(\"C\") @f() {\n        \
         builtin.ret\n    };\n};\n";
    let first = case(1);
    let text = format!("{first}{MARKER}\n{bad}{MARKER}\n{}", case(3));
    let output = opt(&["--split-input-file"], &text);
    output.failed();
    // The bad case's fourth line follows the first case's lines and the marker.
    let line = first.lines().count() + 1 + 4;
    assert!(output.stderr.contains(&format!("--> <stdin>:{line}:5")), "{}", output.stderr);
    assert!(output.stderr.contains(&format!("{line} |     }};")), "{}", output.stderr);
    // Neither the padding nor the other cases are shown.
    assert!(output.stderr.lines().count() < 15, "{}", output.stderr);
    assert!(!output.stderr.contains("arith.constant"), "{}", output.stderr);
    // The first and third are printed, with one marker between them.
    assert_eq!(output.stdout.matches(MARKER).count(), 1, "{}", output.stdout);
    assert!(output.stdout.contains("arith.constant 1"), "{}", output.stdout);
    assert!(output.stdout.contains("arith.constant 3"), "{}", output.stdout);
}

#[test]
fn a_verifier_failure_in_a_case_fails_only_that_case() {
    let text = format!("{INVALID}{MARKER}\n{}", case(5));
    let output = opt(&["--split-input-file"], &text);
    output.failed();
    assert!(output.stderr.contains("--> <stdin>:3:"), "{}", output.stderr);
    assert!(output.stdout.contains("arith.constant 5"), "{}", output.stdout);
    assert!(!output.stdout.contains(MARKER), "{}", output.stdout);
}

#[test]
fn a_custom_marker() {
    let text = format!("{}// ---\n{}{MARKER}\n", case(1), case(2));
    // The default marker is not a split when another is given: it is a comment.
    let output = opt(&["--split-input-file=// ---"], &text);
    output.succeeded();
    assert_eq!(output.stdout.matches("// ---\n").count(), 1, "{}", output.stdout);
    assert!(output.stdout.contains("arith.constant 1"));
    assert!(output.stdout.contains("arith.constant 2"));
}

#[test]
fn an_empty_marker_is_a_usage_error() {
    let output = opt(&["--split-input-file="], &case(1));
    output.failed();
    assert_eq!(output.stdout, "");
}

#[test]
fn a_file_without_a_marker_is_one_case() {
    let split = opt(&["--split-input-file"], &case(1));
    split.succeeded();
    assert_eq!(split.stdout, opt(&[], &case(1)).stdout);
}

#[test]
fn a_case_of_only_whitespace_and_comments_is_skipped() {
    let text = format!(
        "{}{MARKER}\n\n  // nothing here\n{MARKER}\n{MARKER}\n{}{MARKER}\n",
        case(1),
        case(2)
    );
    let output = opt(&["--split-input-file"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout.matches(MARKER).count(), 1, "{}", output.stdout);
}

#[test]
fn a_marker_line_surrounded_by_whitespace_splits() {
    let text = format!("{}   {MARKER}  \n{}", case(1), case(2));
    let output = opt(&["--split-input-file"], &text);
    output.succeeded();
    assert_eq!(output.stdout.matches(MARKER).count(), 1, "{}", output.stdout);
}

#[test]
fn a_marker_in_a_string_or_a_longer_line_is_not_a_split() {
    let text = "builtin.module public @m {\n    builtin.function public extern(\"C\") @f(%c: i1) \
                {\n        cf.assert %c, \"// -----\";\n        builtin.ret;\n    };\n}; // \
                -----\n";
    let output = opt(&["--split-input-file"], text);
    output.succeeded();
    assert!(output.stdout.contains("\"// -----\""), "{}", output.stdout);
    // The text after the module on the last line is a comment: one case, no separator.
    assert_eq!(output.stdout.matches("builtin.module").count(), 1, "{}", output.stdout);
    let longer = format!("{}// ----- and more\n", case(1));
    let output = opt(&["--split-input-file"], &longer);
    output.succeeded();
    assert_eq!(output.stdout, opt(&[], &case(1)).stdout);
}

#[test]
fn the_pass_that_ran_on_no_operation_warning_is_per_case() {
    let world = "builtin.world {\n    builtin.module public @m {\n        builtin.function public \
                 extern(\"C\") @f() {\n            builtin.ret;\n        };\n    };\n};\n";
    let text = format!("{world}{MARKER}\n{world}");
    let output = opt(&["--split-input-file", "-p", "local2reg"], &text);
    output.succeeded();
    assert_eq!(output.stderr.matches("ran on no operation").count(), 2, "{}", output.stderr);
}

#[test]
fn without_the_option_the_marker_is_only_a_comment() {
    let text = format!("{}{MARKER}\n{}", case(1), case(2));
    let output = opt(&[], &text);
    output.failed();
    assert_eq!(output.stdout, "");
}
