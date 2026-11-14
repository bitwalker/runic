//! A partial conversion converts the operations its target calls illegal, leaves those it calls
//! legal, and offers those it has no rule for to the patterns, leaving them as they are when none
//! applies.

use super::*;
use crate::dialects::builtin::Ret;

mod random;

/// Runs a partial conversion of `function` under `target` with `patterns`.
fn convert_partial(
    function: OperationRef,
    target: ConversionTarget,
    patterns: impl IntoIterator<Item = Rule>,
) -> Result<ConversionResult, Report> {
    let mut set = ConversionPatternSet::new(target.context());
    for pattern in patterns {
        set.push(pattern);
    }
    apply_partial_conversion(function, target, set, ConversionConfig::default())
}

/// A target whose unknown operation policy is the default (illegal), where `builtin.function`
/// and `builtin.ret` are legal and a `test.constant` is legal unless it uses a type
/// [wide_converter] changes.
fn constants_target(context: &Rc<Context>) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target
        .add_legal_op::<Function>()
        .add_legal_op::<Ret>()
        .add_dynamically_legal_op::<Constant, _>(|op: &Operation| {
            DynamicLegalityResult::legal_if(has_no_wide_types(op))
        });
    target
}

/// An illegal `i128` constant, a legal `i32` constant, and a `test.sink`, which the target has no
/// rule for, of both.
const MIXED: &str = "    %0 = test.constant 5 : i128;
    %1 = test.constant 7 : i32;
    test.sink %0, %1 : (i128, i32);
    builtin.ret;";

/// The illegal constant is split; the legal one is untouched; the `test.sink`, which the target
/// has no rule for and no pattern converts, is untouched, and reads the split constant through the
/// cast the replacement left, realized by the source callback as a `test.join`.
#[test]
fn a_partial_conversion_converts_the_illegal_and_leaves_the_legal_and_the_operation_without_a_rule()
{
    let (context, function) = parse_function("()", MIXED);
    let mut converter = wide_converter(&context);
    join_halves(&mut converter, Rc::default());

    let result = convert_partial(
        function,
        constants_target(&context),
        [split_constant(&context, &converter)],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(result.converted_ops(), 1);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %2 = test.constant 5 : i64;
    %3 = test.constant 0 : i64;
    %5 = test.join %2, %3;
    %1 = test.constant 7 : i32;
    test.sink %5, %1 : (i128, i32);
    builtin.ret;"
        )
    );
}

/// The same input under a full conversion: the `test.sink` has no rule, the unknown operation
/// policy calls it illegal, and nothing converts it.
#[test]
fn a_full_conversion_of_the_same_input_has_no_legalization_path() {
    let (context, function) = parse_function("()", MIXED);
    let mut converter = wide_converter(&context);
    join_halves(&mut converter, Rc::default());

    let err = convert(function, constants_target(&context), [split_constant(&context, &converter)])
        .unwrap_err();

    assert_eq!(
        format!("{err}"),
        "failed to legalize operation 'test.sink': no legalization path to target; the target has \
         no rule for the operation, and its unknown operation policy calls it illegal"
    );
}

/// A `test.add` of `i128`s, which the target has no rule for, is converted by the pattern for it;
/// the `i64` adds it creates, also without a rule, are offered to the same pattern, which does not
/// match them, and stay. The sink without a pattern reads a join of the halves.
#[test]
fn an_operation_without_a_rule_that_a_pattern_converts_is_converted() {
    let (context, function) = parse_function(
        "(%0: i128, %1: i128)",
        "    %2 = test.add %0, %1;
    test.sink %2 : (i128);
    builtin.ret;",
    );
    let mut converter = wide_converter(&context);
    join_halves(&mut converter, Rc::default());
    split_halves(&mut converter, Rc::default());
    let adds = Rc::new(RefCell::new(vec![]));

    let result = convert_partial(
        function,
        constants_target(&context),
        [split_add(&context, &converter, adds.clone())],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(result.converted_ops(), 1);
    assert_eq!(adds.borrow().len(), 1, "the i64 adds are offered but do not match");
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "(%0: i128, %1: i128)",
            "    %10 = test.low %0;
    %11 = test.high %0;
    %12 = test.low %1;
    %13 = test.high %1;
    %7 = test.add %10, %12;
    %8 = test.add %11, %13;
    %14 = test.join %7, %8;
    test.sink %14 : (i128);
    builtin.ret;"
        )
    );
}

/// Which way a test conversion runs.
#[derive(Debug, Copy, Clone)]
enum Way {
    Full,
    Partial,
}

const WAYS: [Way; 2] = [Way::Full, Way::Partial];

fn convert_by(
    way: Way,
    function: OperationRef,
    target: ConversionTarget,
    patterns: impl IntoIterator<Item = Rule>,
) -> Result<ConversionResult, Report> {
    match way {
        Way::Full => convert(function, target, patterns),
        Way::Partial => convert_partial(function, target, patterns),
    }
}

/// A `test.add` that a rule of the target calls illegal (its own rule, or an interface rule) and
/// no pattern converts is an error in both conversions.
#[test]
fn an_illegal_operation_without_a_pattern_is_an_error_in_both_conversions() {
    type Rules = fn(&mut ConversionTarget);
    let cases: [(&str, Rules); 2] = [
        ("its own rule", |target| {
            target.add_illegal_op::<Add>();
        }),
        ("an interface rule", |target| {
            target.add_dynamically_legal_op_interface::<dyn crate::traits::Commutative, _>(|_| {
                DynamicLegalityResult::illegal()
            });
        }),
    ];
    for way in WAYS {
        for (name, rules) in cases {
            let (context, function) = parse_function(
                "(%0: i32)",
                "    %1 = test.add %0, %0;
    test.sink %1 : (i32);
    builtin.ret;",
            );
            let before = function.borrow().to_string();
            let mut target = constants_target(&context);
            rules(&mut target);
            target.add_legal_op::<Sink>();

            let err = convert_by(way, function, target, []).unwrap_err();

            let err = format!("{err}");
            assert!(
                err.starts_with(
                    "failed to legalize operation 'test.add': no legalization path to target; "
                ),
                "{way:?}, {name}: {err}"
            );
            assert!(!err.contains("unknown operation policy"), "{way:?}, {name}: {err}");
            assert_str_eq!(function.borrow().to_string(), before, "{way:?}, {name}");
        }
    }
}

/// A pattern that changes the IR and then fails, or answers that it did not match, aborts both
/// conversions, even on an operation the target has no rule for.
#[test]
fn a_pattern_that_fails_after_mutating_is_an_error_in_both_conversions() {
    type Outcome = fn() -> Result<bool, Report>;
    let cases: [(Outcome, &str); 2] = [
        (
            || Err(Report::msg("gave up")),
            "conversion pattern 'mutates-then-fails' failed after mutating operation 'test.add'",
        ),
        (
            || Ok(false),
            "conversion pattern 'mutates-then-fails' mutated operation 'test.add' but returned no \
             match",
        ),
    ];
    for way in WAYS {
        for (outcome, message) in cases {
            let (context, function) = parse_function(
                "(%0: i32)",
                "    %1 = test.add %0, %0;
    test.sink %1 : (i32);
    builtin.ret;",
            );
            let mut target = constants_target(&context);
            target.add_legal_op::<Sink>();
            let pattern = Rule::new::<Add>(
                &context,
                "mutates-then-fails",
                None,
                [registered::<Constant>(&context)],
                move |op, _, rewriter| {
                    let span = op.borrow().span();
                    rewriter.set_insertion_point_before(op);
                    rewriter.create_op::<Constant, _>(span, (Int::from(1i32),))?;
                    outcome()
                },
            );

            let err = convert_by(way, function, target, [pattern]).unwrap_err();

            let err = format!("{err}");
            assert!(err.contains(message), "{way:?}: {err}");
        }
    }
}

/// A split constant whose `test.sink` no pattern converts, with no source callback: the cast the
/// replacement left is not realized.
const UNREALIZED: &str = "    %0 = test.constant 5 : i128;
    test.sink %0 : (i128);
    builtin.ret;";

/// In a partial conversion a cast nobody realizes stays when the target calls
/// `builtin.unrealized_conversion_cast` legal, and the result verifies; when the target calls it
/// illegal, or has no rule for it, it is the error of a full conversion.
#[test]
fn a_cast_nobody_realizes_stays_in_a_partial_conversion_only_when_the_target_calls_it_legal() {
    let error = "failed to legalize operation 'builtin.unrealized_conversion_cast': no source \
                 materialization of the type converter realizes the cast of (i64, i64) to (i128), \
                 which is used by 'test.sink'";
    type Rules = fn(&mut ConversionTarget);
    let cases: [(&str, Rules, Option<&str>); 3] = [
        (
            "legal",
            |target| {
                target.add_legal_op::<UnrealizedConversionCast>();
            },
            None,
        ),
        (
            "illegal",
            |target| {
                target.add_illegal_op::<UnrealizedConversionCast>();
            },
            Some(error),
        ),
        ("no rule", |_| {}, Some(error)),
    ];
    for (name, rules, expected) in cases {
        let (context, function) = parse_function("()", UNREALIZED);
        let converter = wide_converter(&context);
        let mut target = constants_target(&context);
        rules(&mut target);

        let result = convert_partial(function, target, [split_constant(&context, &converter)]);

        match expected {
            None => {
                result.unwrap_or_else(|err| panic!("{name}: {}", describe(&err)));
                assert_str_eq!(
                    function.borrow().to_string(),
                    function_text(
                        "()",
                        "    %1 = test.constant 5 : i64;
    %2 = test.constant 0 : i64;
    %3 = builtin.unrealized_conversion_cast %1, %2 : (i64, i64) to i128;
    test.sink %3 : (i128);
    builtin.ret;"
                    ),
                    "{name}"
                );
            }
            Some(message) => {
                let err = result.unwrap_err();
                assert_eq!(format!("{err}"), message, "{name}");
            }
        }
    }

    // A full conversion whose target calls the cast legal still reports it.
    let (context, function) = parse_function("()", UNREALIZED);
    let converter = wide_converter(&context);
    let mut target = constants_target(&context);
    target.add_legal_op::<UnrealizedConversionCast>().add_legal_op::<Sink>();
    let err = convert(function, target, [split_constant(&context, &converter)]).unwrap_err();
    assert_eq!(format!("{err}"), error);
}
