//! Every way an attempt to convert an operation ends before its pattern mutated the IR leaves the
//! IR as it was: the casts the driver made for the operands are erased and leave the table of the
//! driver's casts.

use super::*;
use crate::SourceSpan;

/// `@f` adds two `i128`s and sinks the sum; each operand of the add becomes two `i64`s.
fn split_operands_function() -> (Rc<Context>, OperationRef) {
    parse_function(
        "(%0: i128, %1: i128)",
        "    %2 = test.add %0, %1;
    test.sink %2 : (i128);
    builtin.ret;",
    )
}

/// The `test.add` of `function`.
fn the_add(function: OperationRef) -> OperationRef {
    let mut add = None;
    function.borrow().prewalk_all(|op| {
        if op.is::<Add>() {
            add = Some(op.as_operation_ref());
        }
    });
    add.unwrap()
}

/// A pattern for the `test.add` that is given split operands and then does `outcome` without
/// changing anything.
fn add_that(
    context: &Rc<Context>,
    name: &'static str,
    outcome: fn() -> Result<bool, Report>,
) -> Rule {
    let converter = wide_converter(context);
    Rule::new::<Add>(context, name, Some(converter), [], move |_, operands, _| {
        assert_eq!(operands.get(0).len(), 2, "the operand is split");
        outcome()
    })
}

/// When inserting the cast for the second operand fails, the cast made for the first is erased
/// and leaves the table, and the error is returned: the IR prints as before.
#[test]
fn an_operand_conversion_failing_on_the_second_operand_leaves_no_cast() {
    let (context, function) = split_operands_function();
    let before = function.borrow().to_string();
    let add = the_add(function);
    let converter = Rc::new(wide_converter(&context));
    let state = ConversionState::default();
    let mut rewriter = ConversionPatternRewriter::new_for_conversion(
        context.clone(),
        Some(Rc::clone(&converter)),
        state.clone(),
    );
    rewriter.set_insertion_point_before(add);
    let mut calls = 0;
    let mut insert = |rewriter: &mut ConversionPatternRewriter,
                      value: ValueRef,
                      types: SmallVec<[Type; 2]>,
                      span: SourceSpan| {
        calls += 1;
        if calls == 2 {
            return Err(Report::msg("the second operand cannot be converted"));
        }
        rewriter
            .insert_cast(CastKind::Target, [value], types, span)
            .map(|(_, outputs)| outputs)
    };

    let err = converted_operands_with(add, Some(&converter), &mut rewriter, &mut insert)
        .map(|_| ())
        .unwrap_err();

    assert_eq!(format!("{err}"), "the second operand cannot be converted");
    assert_eq!(calls, 2);
    assert!(state.casts.borrow().is_empty(), "the first operand's cast left the table");
    assert_str_eq!(function.borrow().to_string(), before);
}

/// A pattern given converted operands that answers `Ok(false)` leaves no cast: no pattern
/// matched, and the IR prints as before.
#[test]
fn a_pattern_that_does_not_match_leaves_no_cast() {
    let (context, function) = split_operands_function();
    let before = function.borrow().to_string();
    let pattern = add_that(&context, "no-match", || Ok(false));

    let err = convert(function, wide_target(&context), [pattern]).unwrap_err();

    assert_eq!(
        format!("{err}"),
        "failed to legalize operation 'test.add': no conversion pattern matched"
    );
    assert_str_eq!(function.borrow().to_string(), before);
}

/// A pattern given converted operands that fails before changing anything leaves no cast: its
/// error is the conversion's, and the IR prints as before.
#[test]
fn a_pattern_that_fails_before_changing_anything_leaves_no_cast() {
    let (context, function) = split_operands_function();
    let before = function.borrow().to_string();
    let pattern = add_that(&context, "fails", || Err(Report::msg("the pattern gave up")));

    let err = convert(function, wide_target(&context), [pattern]).unwrap_err();

    assert_eq!(format!("{err}"), "the pattern gave up");
    assert_str_eq!(function.borrow().to_string(), before);
}

/// A pattern given converted operands that reports success without changing anything breaks the
/// pattern contract; the conversion fails, and leaves no cast: the IR prints as before.
#[test]
fn a_pattern_that_succeeds_without_changing_anything_leaves_no_cast() {
    let (context, function) = split_operands_function();
    let before = function.borrow().to_string();
    let pattern = add_that(&context, "claims-success", || Ok(true));

    let err = convert(function, wide_target(&context), [pattern]).unwrap_err();

    assert_eq!(
        format!("{err}"),
        "conversion pattern 'claims-success' reported success for operation 'test.add' without \
         mutating IR"
    );
    assert_str_eq!(function.borrow().to_string(), before);
}

/// Two patterns that do not match, the second given its own casts, leave no cast.
#[test]
fn every_pattern_that_does_not_match_leaves_no_cast() {
    let (context, function) = split_operands_function();
    let before = function.borrow().to_string();
    let first = add_that(&context, "first", || Ok(false));
    let second = add_that(&context, "second", || Ok(false));

    convert(function, wide_target(&context), [first, second]).unwrap_err();

    assert_str_eq!(function.borrow().to_string(), before);
}
