//! The test passes the tool links: `test-convert-wide-ints`, `test-fail` and
//! `test-print-int-ranges`.

use super::opt;

/// `--verify-diagnostics`.
const CHECK: &str = "--verify-diagnostics";

/// A module with `i128` parameters and results, an addition over `i128`, a call, a branch that
/// forwards an `i128`, and a local of `i128`.
const WIDE: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @sum(%a: i128, %b: i128) -> i128 {
        %s = arith.add %a, %b;
        builtin.ret %s : (i128);
    };
    builtin.function public extern("C") @pick(%a: i128, %b: i128, %c: i1) -> i128 locals(i128) {
        builtin.store_local 0, %a;
        cf.cond_br %c ^t(%a : i128), ^e : (i1);
    ^t(%x: i128):
        %y = builtin.call @sum(%x, %b) : extern("C") (i128, i128) -> i128;
        cf.br ^j(%y : i128);
    ^e:
        %l = builtin.load_local 0 : i128;
        cf.br ^j(%l : i128);
    ^j(%z: i128):
        builtin.ret %z : (i128);
    };
};
"#;

/// [WIDE] converted: everything but the addition is over `(i64, i64)`; the addition reads two
/// joins of halves, and its result is split for the return.
const WIDE_CONVERTED: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @sum(%10: i64, %11: i64, %12: i64, %13: i64) -> (i64, i64) {
        %37 = arith.zext %10 <{ ty = #builtin.type<i128> }>;
        %38 = arith.zext %11 <{ ty = #builtin.type<i128> }>;
        %39 = arith.constant 64 : i64;
        %40 = arith.zext %39 <{ ty = #builtin.type<i128> }>;
        %41 = arith.shl %38, %40;
        %42 = arith.bor %41, %37;
        %43 = arith.zext %12 <{ ty = #builtin.type<i128> }>;
        %44 = arith.zext %13 <{ ty = #builtin.type<i128> }>;
        %45 = arith.constant 64 : i64;
        %46 = arith.zext %45 <{ ty = #builtin.type<i128> }>;
        %47 = arith.shl %44, %46;
        %48 = arith.bor %47, %43;
        %2 = arith.add %42, %48;
        %49 = arith.trunc %2 <{ ty = #builtin.type<i64> }>;
        %50 = arith.constant 64 : i64;
        %51 = arith.zext %50 <{ ty = #builtin.type<i128> }>;
        %52 = arith.shr %2, %51;
        %53 = arith.trunc %52 <{ ty = #builtin.type<i64> }>;
        builtin.ret %49, %53 : (i64, i64);
    };
    builtin.function public extern("C") @pick(%18: i64, %19: i64, %20: i64, %21: i64, %22: i1) -> (i64, i64) locals(i64, i64) {
        builtin.store_local 0, %18;
        builtin.store_local 1, %19;
        cf.cond_br %22 ^block4(%18, %19 : i64, i64), ^block5 : (i1);
    ^block4(%25: i64, %26: i64):
        %31, %32 = builtin.call @sum(%25, %26, %20, %21) : extern("C") (i64, i64, i64, i64) -> (i64, i64);
        cf.br ^block6(%31, %32 : i64, i64);
    ^block5:
        %34 = builtin.load_local 0 : i64;
        %35 = builtin.load_local 1 : i64;
        cf.br ^block6(%34, %35 : i64, i64);
    ^block6(%28: i64, %29: i64):
        builtin.ret %28, %29 : (i64, i64);
    };
};
"#;

#[test]
fn a_partial_conversion_flattens_i128_and_leaves_arith_reading_joins() {
    let output = opt(&["-p", "test-convert-wide-ints"], WIDE);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout, WIDE_CONVERTED);
    // What is left of `i128` is the addition, the two joins it reads and the split of its result.
    let (sum, pick) = output.stdout.split_once("@pick").unwrap();
    assert!(!pick.contains("i128"), "{pick}");
    assert_eq!(sum.matches("arith.bor").count(), 2, "{sum}");
    assert_eq!(sum.matches("arith.shl").count(), 2, "{sum}");
    assert_eq!(sum.matches("arith.shr").count(), 1, "{sum}");
    assert_eq!(sum.matches("arith.trunc").count(), 2, "{sum}");
    // The result verifies (the tool verified it after the pass) and reads back.
    let again = opt(&[], &output.stdout);
    again.succeeded();
    assert_eq!(again.stderr, "");
    // Converting again finds nothing over `i128` that a pattern converts.
    let twice = opt(&["-p", "test-convert-wide-ints"], &output.stdout);
    twice.succeeded();
    assert_eq!(twice.stdout, again.stdout);
}

#[test]
fn a_full_conversion_fails_on_arith_over_i128_naming_it() {
    let output = opt(&["-p", "test-convert-wide-ints{full}"], WIDE);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: failed to legalize operation 'arith.add'"),
        "{}",
        output.stderr
    );
    assert!(
        output
            .stderr
            .contains("context: pass 'test-convert-wide-ints' failed on 'builtin.module' @m"),
        "{}",
        output.stderr
    );
}

#[test]
fn a_full_conversion_without_arith_over_i128_leaves_no_i128() {
    let text = WIDE.replace("%s = arith.add %a, %b;\n        builtin.ret %s", "builtin.ret %a");
    let output = opt(&["-p", "test-convert-wide-ints{full=true}"], &text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert!(!output.stdout.contains("i128"), "{}", output.stdout);
    assert!(!output.stdout.contains("arith."), "{}", output.stdout);
    // `full=false` is the partial conversion.
    let partial = opt(&["-p", "test-convert-wide-ints{full=false}"], WIDE);
    assert_eq!(partial.stdout, WIDE_CONVERTED);
}

#[test]
fn the_conversion_is_rooted_where_the_pass_runs() {
    // A world: the conversion reaches the module inside it.
    let world = format!("builtin.world {{\n{WIDE}}};\n");
    let output = opt(&["-p", "test-convert-wide-ints"], &world);
    output.succeeded();
    assert!(output.stdout.contains("@pick(%"), "{}", output.stdout);
    assert!(
        !output.stdout.split_once("@pick").unwrap().1.contains("i128"),
        "{}",
        output.stdout
    );
    // A function whose caller is outside it: the conversion's error is the pass's.
    let output = opt(&["-p", "builtin.module(builtin.function(test-convert-wide-ints))"], WIDE);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output
            .stderr
            .starts_with("error: the conversion changes the signature of function"),
        "{}",
        output.stderr
    );
    assert!(
        output
            .stderr
            .contains("context: pass 'test-convert-wide-ints' failed on 'builtin.function' @sum"),
        "{}",
        output.stderr
    );
}

#[test]
fn a_conversion_that_changes_nothing_is_not_printed_after_with_print_ir_after_change() {
    let output = opt(&["-p", "test-convert-wide-ints", "--print-ir-after-change"], WIDE);
    output.succeeded();
    assert!(output.stderr.starts_with("// ----- IR after test-convert-wide-ints"));
    let unchanged =
        opt(&["-p", "test-convert-wide-ints", "--print-ir-after-change"], super::LOCALS);
    unchanged.succeeded();
    assert_eq!(unchanged.stderr, "");
}

#[test]
fn unknown_options_are_errors() {
    for (pipeline, message) in [
        (
            "test-convert-wide-ints{partial}",
            "pass 'test-convert-wide-ints' has no option 'partial'",
        ),
        ("test-convert-wide-ints{full=maybe}", "invalid value 'maybe' of option 'full'"),
        ("test-fail{reason=x}", "pass 'test-fail' has no option 'reason'"),
        ("test-print-int-ranges{x}", "takes no options"),
    ] {
        let output = opt(&["-p", pipeline], super::LOCALS);
        output.failed();
        assert!(output.stderr.contains(message), "{pipeline}: {}", output.stderr);
    }
}

#[test]
fn test_fail_fails_with_its_message_at_the_operation_it_runs_on() {
    let output = opt(&["-p", "test-fail"], super::LOCALS);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: test-fail\n --> <stdin>:1:1"),
        "{}",
        output.stderr
    );
    assert!(
        output
            .stderr
            .contains("context: pass 'test-fail' failed on 'builtin.module' @m"),
        "{}",
        output.stderr
    );
    // A message with spaces and quotes, at a function.
    let output = opt(
        &[
            "-p",
            r#"builtin.module(builtin.function(test-fail{message="no \"good\" here" after-change}))"#,
        ],
        super::LOCALS,
    );
    output.failed();
    assert!(
        output.stderr.starts_with("error: no \"good\" here\n --> <stdin>:2:5"),
        "{}",
        output.stderr
    );
}

#[test]
fn test_fail_is_matched_by_verify_diagnostics() {
    // The failure is at the function.
    let text = super::LOCALS
        .replace("locals(i32) {\n", "locals(i32) { // expected-error {{a planned failure}}\n");
    let output = opt(
        &[
            CHECK,
            "-p",
            r#"builtin.module(builtin.function(test-fail{message="a planned failure"}))"#,
        ],
        &text,
    );
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout, "");
    // Expected at the wrong line, it is a mismatch.
    let output = opt(&[CHECK, "-p", r#"test-fail{message="a planned failure"}"#], &text);
    output.failed();
    assert!(
        output.stderr.contains("unexpected error: a planned failure"),
        "{}",
        output.stderr
    );
}

#[test]
fn the_test_passes_print_their_options_back_as_text_that_loads_the_same_pipeline() {
    let pipeline = r#"builtin.module(test-convert-wide-ints{full},builtin.function(test-print-int-ranges),test-fail{message="a \"quoted\" message" after-change})"#;
    let output = opt(&["-p", pipeline, "--dump-pass-pipeline"], super::LOCALS);
    output.failed();
    let dumped = output.stderr.lines().next().unwrap();
    assert_eq!(dumped, pipeline);
    let again = opt(&["-p", dumped, "--dump-pass-pipeline"], super::LOCALS);
    assert_eq!(again.stderr.lines().next().unwrap(), dumped);
    // Options at their defaults are not printed.
    let output = opt(
        &[
            "-p",
            "test-convert-wide-ints{full=false},test-fail{message=test-fail}",
            "--dump-pass-pipeline",
        ],
        super::LOCALS,
    );
    assert_eq!(
        output.stderr.lines().next().unwrap(),
        "builtin.module(test-convert-wide-ints,test-fail)"
    );
}

/// A message reaches `test-fail` as written, whatever it holds, and prints back as text that loads
/// the same pipeline.
#[test]
fn a_message_may_hold_commas_equals_braces_and_quotes() {
    for message in [
        "expected i32, found i64",
        "a=b",
        "{x}, (y)",
        r#"say "hi", then \ go"#,
        "message=other, after-change",
    ] {
        let mut quoted = String::new();
        runic_core::pass::write_pass_option(&mut quoted, "message", Some(message)).unwrap();
        let pipeline = format!("builtin.module(test-fail{{{quoted}}})");
        let output = opt(&["-p", &pipeline, "--dump-pass-pipeline"], super::LOCALS);
        output.failed();
        let mut lines = output.stderr.lines();
        let dumped = lines.next().unwrap();
        assert_eq!(dumped, pipeline);
        assert_eq!(lines.next(), Some(format!("error: {message}").as_str()), "{}", output.stderr);
        let again = opt(&["-p", dumped, "--dump-pass-pipeline"], super::LOCALS);
        assert_eq!(again.stderr.lines().next().unwrap(), dumped);
    }
}

#[test]
fn a_bare_message_says_a_value_is_required() {
    let output = opt(&["-p", "test-fail{message}"], super::LOCALS);
    output.failed();
    assert_eq!(
        output.stderr,
        "error: option 'message' of pass 'test-fail' requires a value: 'message=\"..\"'\n"
    );
}

#[test]
fn the_test_passes_are_listed() {
    let output = opt(&["--list-passes"], "");
    for pass in ["test-convert-wide-ints", "test-fail", "test-print-int-ranges"] {
        assert!(
            output.stdout.lines().any(|line| line.split_whitespace().next() == Some(pass)),
            "{pass}: {}",
            output.stdout
        );
    }
}

/// Functions of `runic-analysis`'s own tests of the integer range analysis, with the ranges those
/// tests expect: `inferrable_add_uses_operand_ranges` (1 + 2 is exactly 3),
/// `comparisons_remain_unknown_when_ranges_overlap` (an argument below 10 may or may not be) and
/// `a_comparison_decided_by_the_type_is_exact` (every `i32` is `uge 0`); and a block no branch
/// reaches, whose values have no range.
const RANGES: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @inferrable_add_uses_operand_ranges() -> i32 {
        %lhs = arith.constant 1 : i32; // expected-hint {{range of result 0: u[1, 1] s[1, 1] width 32}}
        %rhs = arith.constant 2 : i32; // expected-hint {{range of result 0: u[2, 2] s[2, 2] width 32}}
        %sum = arith.add %lhs, %rhs; // expected-hint {{range of result 0: u[3, 3] s[3, 3] width 32}}
        builtin.ret %sum : (i32);
    };
    builtin.function public extern("C") @comparisons_remain_unknown_when_ranges_overlap(%arg: i32) -> i1 {
        %limit = arith.constant 10 : i32; // expected-hint {{range of result 0: u[10, 10] s[10, 10] width 32}}
        %c = arith.cmpi ult %arg, %limit; // expected-hint {{range of result 0: u[0, 1] s[-1, 0] width 1}}
        builtin.ret %c : (i1);
    };
    builtin.function public extern("C") @a_comparison_decided_by_the_type_is_exact(%arg: i32) -> i1 {
        %zero = arith.constant 0 : i32; // expected-hint {{range of result 0: u[0, 0] s[0, 0] width 32}}
        %c = arith.cmpi uge %arg, %zero; // expected-hint {{range of result 0: u[1, 1] s[-1, -1] width 1}}
        builtin.ret %c : (i1);
    };
    builtin.function public extern("C") @unreached() -> i32 {
        %a = arith.constant 1 : i32; // expected-hint {{range of result 0: u[1, 1] s[1, 1] width 32}}
        builtin.ret %a : (i32);
    ^dead:
        %b = arith.constant 2 : i32; // expected-hint {{range of result 0: unknown}}
        builtin.ret %b : (i32);
    };
};
"#;

#[test]
fn test_print_int_ranges_gives_the_ranges_the_analysis_tests_expect() {
    let output = opt(&[CHECK, "-p", "test-print-int-ranges"], RANGES);
    output.succeeded();
    assert_eq!(output.stderr, "");
    // It changes nothing.
    assert_eq!(output.stdout, opt(&[], RANGES).stdout);
    let changed = opt(&["-p", "test-print-int-ranges", "--print-ir-after-change"], RANGES);
    changed.succeeded();
    assert!(!changed.stderr.contains("// ----- IR after"), "{}", changed.stderr);
    // Without the check, the hints are rendered at the operations.
    assert!(
        changed
            .stderr
            .contains("hint: range of result 0: u[3, 3] s[3, 3] width 32\n --> <stdin>:5:16"),
        "{}",
        changed.stderr
    );
}

/// A `builtin.ret_imm` of an `i128` becomes its halves as constants and a return of them, in
/// both modes; a global variable of `i128` is legal in partial mode and an error in full mode.
#[test]
fn a_return_of_an_i128_immediate_converts_and_a_global_follows_the_mode() {
    let text = |global: &str| {
        format!(
            "builtin.module public @m {{\n    builtin.global_variable public @g : {global};\n    \
             builtin.function public extern(\"C\") @f() -> i128 {{\n        builtin.ret_imm \
             18446744073709551617 : i128;\n    }};\n}};\n"
        )
    };
    let converted = "    builtin.function public extern(\"C\") @f() -> (i64, i64) {\n        %0 = \
                     arith.constant 1 : i64;\n        %1 = arith.constant 1 : i64;\n        \
                     builtin.ret %0, %1 : (i64, i64);\n    };\n";
    for pipeline in ["test-convert-wide-ints", "test-convert-wide-ints{full}"] {
        let output = opt(&["-p", pipeline], &text("i32"));
        output.succeeded();
        assert!(output.stdout.contains(converted), "{pipeline}: {}", output.stdout);
    }
    let output = opt(&["-p", "test-convert-wide-ints"], &text("i128"));
    output.succeeded();
    assert!(output.stdout.contains("@g : i128;"), "{}", output.stdout);
    let output = opt(&["-p", "test-convert-wide-ints{full}"], &text("i128"));
    output.failed();
    assert!(
        output
            .stderr
            .starts_with("error: failed to legalize operation 'builtin.global_variable'"),
        "{}",
        output.stderr
    );
}
