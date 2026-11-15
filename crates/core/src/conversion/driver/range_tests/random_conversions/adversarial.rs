//! Patterns and callbacks that break the contract at each point: each must be an error, never a
//! panic, a hang or silently wrong IR. `outcome` runs a conversion and says which it was.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    println,
};

use super::{super::signature_tests::ForwardBr, *};
use crate::conversion::SignatureConversion;

#[derive(Debug)]
enum Outcome {
    Error(String),
    Converted(String),
    Panicked(String),
}

impl Outcome {
    fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// The error's message, the converted IR, or the panic's message.
    fn text(&self) -> &str {
        match self {
            Self::Error(text) | Self::Converted(text) | Self::Panicked(text) => text,
        }
    }
}

const BODY: &str = "    %0 = test.constant 1 : i128;
    %1 = test.constant 2 : i128;
    %2 = test.add %0, %1;
    test.sink %2 : (i128);
    builtin.ret;";

fn outcome_of(
    name: &str,
    signature: &str,
    body: &str,
    target: impl FnOnce(&Rc<Context>) -> ConversionTarget,
    patterns: impl FnOnce(&Rc<Context>, &TypeConverter) -> Vec<Rule>,
    converter: impl FnOnce(&Rc<Context>) -> TypeConverter,
    config: ConversionConfig,
) -> Outcome {
    let (context, function) = parse_function(signature, body);
    let converter = converter(&context);
    let mut set = ConversionPatternSet::new(context.clone());
    for pattern in patterns(&context, &converter) {
        set.push(pattern);
    }
    let target = target(&context);
    let result =
        catch_unwind(AssertUnwindSafe(|| apply_full_conversion(function, target, set, config)));
    let outcome = match result {
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            Outcome::Panicked(message)
        }
        Ok(Err(err)) => Outcome::Error(format!("{err}")),
        Ok(Ok(_)) => Outcome::Converted(function.borrow().to_string()),
    };
    println!("--- {name}: {}", outcome.text());
    // The context must outlive the operation handles.
    core::mem::forget(context);
    outcome
}

fn outcome(
    name: &str,
    patterns: impl FnOnce(&Rc<Context>, &TypeConverter) -> Vec<Rule>,
) -> Outcome {
    outcome_of(
        name,
        "()",
        BODY,
        wide_target,
        patterns,
        |context| {
            let mut converter = wide_converter(context);
            split_callbacks_local(&mut converter);
            converter
        },
        ConversionConfig::default(),
    )
}

fn split_callbacks_local(converter: &mut TypeConverter) {
    converter.add_source_materialization(move |rewriter, inputs, ty, span| {
        let [low, high] = inputs else {
            return Ok(None);
        };
        if ty != rewriter.context().i128() {
            return Ok(None);
        }
        let join = rewriter.create_op::<Join, _>(span, (*low, *high))?;
        Ok(Some(join.borrow().result().as_value_ref()))
    });
    converter.add_target_materialization(move |rewriter, input, types, span| {
        let i64 = rewriter.context().i64();
        if types != [i64, i64] {
            return Ok(None);
        }
        let low = rewriter.create_op::<Low, _>(span, (input,))?;
        let high = rewriter.create_op::<High, _>(span, (input,))?;
        let low = low.borrow().result().as_value_ref();
        let high = high.borrow().result().as_value_ref();
        Ok(Some(smallvec::smallvec![low, high]))
    });
}

/// A pattern for `test.add` over `i128` whose rewrite is `f`.
fn add_rule(
    context: &Rc<Context>,
    converter: &TypeConverter,
    f: impl Fn(
        OperationRef,
        ConvertedOperands<'_>,
        &mut ConversionPatternRewriter,
    ) -> Result<bool, Report>
    + 'static,
) -> Rule {
    Rule::new::<Add>(
        context,
        "bad-add",
        Some(converter.clone()),
        [registered::<Add>(context), registered::<Constant>(context)],
        move |op, operands, rewriter| {
            if op.borrow().results()[0].borrow().ty() != op.borrow().context().i128() {
                return Ok(false);
            }
            f(op, operands, rewriter)
        },
    )
}

fn good(context: &Rc<Context>, converter: &TypeConverter) -> Vec<Rule> {
    vec![
        split_constant(context, converter),
        flatten_sink(context, converter, Rc::default()),
    ]
}

fn with_add(
    f: impl Fn(
        OperationRef,
        ConvertedOperands<'_>,
        &mut ConversionPatternRewriter,
    ) -> Result<bool, Report>
    + 'static,
) -> impl FnOnce(&Rc<Context>, &TypeConverter) -> Vec<Rule> {
    move |context, converter| {
        let mut patterns = good(context, converter);
        patterns.push(add_rule(context, converter, f));
        patterns
    }
}

fn halves_add(
    op: OperationRef,
    operands: ConvertedOperands<'_>,
    rewriter: &mut ConversionPatternRewriter,
) -> Result<(ValueRef, ValueRef), Report> {
    let ([a_low, a_high], [b_low, b_high]) = (operands.get(0), operands.get(1)) else {
        return Err(Report::msg("expected halves"));
    };
    let span = op.borrow().span();
    let low = rewriter.create_op::<Add, _>(span, (*a_low, *b_low, Overflow::Wrapping))?;
    let high = rewriter.create_op::<Add, _>(span, (*a_high, *b_high, Overflow::Wrapping))?;
    Ok((low.borrow().result().as_value_ref(), high.borrow().result().as_value_ref()))
}

#[test]
fn ok_false_after_changing_is_an_error() {
    let outcome = outcome(
        "Ok(false) after changing",
        with_add(|op, operands, rewriter| {
            halves_add(op, operands, rewriter)?;
            Ok(false)
        }),
    );
    assert!(outcome.is_error(), "{outcome:?}");
}

#[test]
fn ok_true_without_changing_is_an_error() {
    let outcome = outcome("Ok(true) without changing", with_add(|_, _, _| Ok(true)));
    assert!(outcome.is_error(), "{outcome:?}");
}

#[test]
fn err_after_changing_is_an_error() {
    let outcome = outcome(
        "Err after changing",
        with_add(|op, operands, rewriter| {
            halves_add(op, operands, rewriter)?;
            Err(Report::msg("the pattern gives up"))
        }),
    );
    assert!(outcome.is_error(), "{outcome:?}");
}

#[test]
fn replacing_with_the_wrong_types_is_an_error() {
    for (name, pick) in [
        ("one half", 0usize),
        ("three values", 1),
        ("no values", 2),
        ("the wrong number of ranges", 3),
        ("an i128 of another kind of range (two ranges of one)", 4),
    ] {
        let outcome = outcome(
            name,
            with_add(move |op, operands, rewriter| {
                let (low, high) = halves_add(op, operands, rewriter)?;
                match pick {
                    0 => rewriter.replace_op_with_multiple(op, &[&[low]])?,
                    1 => rewriter.replace_op_with_multiple(op, &[&[low, high, low]])?,
                    2 => rewriter.replace_op_with_multiple(op, &[&[]])?,
                    3 => rewriter.replace_op_with_multiple(op, &[&[low, high], &[low, high]])?,
                    _ => rewriter.replace_op_with_multiple(op, &[&[low], &[high]])?,
                }
                Ok(true)
            }),
        );
        assert!(outcome.is_error(), "{name}: {outcome:?}");
    }
}

/// Erasing the operation while its result still has users: an error from `erase_op`.
#[test]
fn erasing_an_operation_that_has_users_is_an_error() {
    let outcome = outcome(
        "erase an operation with users",
        with_add(|op, _, rewriter| {
            rewriter.erase_op(op)?;
            Ok(true)
        }),
    );
    assert!(outcome.is_error(), "{outcome:?}");
    assert!(
        outcome.text().contains("cannot be erased: its result 0 still has users"),
        "{outcome:?}"
    );
}

/// Erasing an operation the pattern does not own: the producer of an operand, which has users.
#[test]
fn erasing_the_dead_cast_that_gave_an_operand_goes_unnoticed() {
    let outcome = outcome(
        "erase the producer of an operand",
        with_add(|op, operands, rewriter| {
            let (low, high) = halves_add(op, operands, rewriter)?;
            let producer =
                op.borrow().operands()[0].borrow().as_value_ref().borrow().get_defining_op();
            rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
            if let Some(producer) = producer {
                rewriter.erase_op(producer)?;
            }
            Ok(true)
        }),
    );
    assert!(matches!(outcome, Outcome::Converted(_)), "{outcome:?}");
}

/// Erasing another operation that has no users (the sink that follows): nothing can tell.
#[test]
fn erasing_an_unused_operation_it_does_not_own_goes_unnoticed() {
    let outcome = outcome(
        "erase the following sink",
        with_add(|op, operands, rewriter| {
            let (low, high) = halves_add(op, operands, rewriter)?;
            let next = op.next().unwrap();
            rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
            rewriter.erase_op(next)?;
            Ok(true)
        }),
    );
    // The driver cannot know the sink was not the pattern's to erase.
    assert!(matches!(outcome, Outcome::Converted(_)), "{outcome:?}");
}

/// Replacing a result by a range that holds the result itself.
#[test]
fn replacing_a_result_by_itself_is_an_error() {
    let outcome = outcome(
        "replace by its own result",
        with_add(|op, _, rewriter| {
            let own = op.borrow().results()[0].borrow().as_value_ref();
            rewriter.replace_op(op, &[own])?;
            Ok(true)
        }),
    );
    assert!(outcome.is_error(), "{outcome:?}");
    assert!(
        outcome.text().contains("a result of the operation being replaced"),
        "{outcome:?}"
    );
}

/// Replacing the same operation twice.
#[test]
fn replacing_an_operation_twice_goes_unnoticed() {
    let outcome = outcome(
        "replace twice",
        with_add(|op, operands, rewriter| {
            let (low, high) = halves_add(op, operands, rewriter)?;
            rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
            rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
            Ok(true)
        }),
    );
    assert!(matches!(outcome, Outcome::Converted(_)), "{outcome:?}");
}

/// Replacing by values defined after the users (the halves are built after the sink) is an
/// error: the driver reports the use before the definition when it applies the replacement.
///
/// The conversion driver reports this before the verifier runs: the error asserted here is the
/// driver's, not the verifier's dominance check (which would also reject the IR).
#[test]
fn replacing_by_values_defined_after_their_users_is_an_error() {
    let outcome = outcome(
        "replace by later values",
        with_add(|op, operands, rewriter| {
            let last = op.parent().unwrap().borrow().body().back().as_pointer().unwrap();
            rewriter.set_insertion_point_before(last);
            let (low, high) = halves_add(op, operands, rewriter)?;
            rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
            Ok(true)
        }),
    );
    assert!(outcome.is_error(), "{outcome:?}");
    assert!(
        outcome.text().contains("but its user 'test.sink' comes before that definition"),
        "{outcome:?}"
    );
}

/// A pattern that changes nothing through `modify_op_in_place` and reports success, for an
/// operation that stays illegal: the driver offers it again until the limit, on a stack that does
/// not grow with the applications.
#[test]
fn a_pattern_that_keeps_modifying_in_place_ends_at_the_limit() {
    let outcome = outcome(
        "modify in place forever",
        with_add(|op, _, rewriter| {
            rewriter.modify_op_in_place(op, |_| ())?;
            Ok(true)
        }),
    );
    assert_eq!(
        outcome.text(),
        "dialect conversion exceeded the limit of 1024 pattern applications that follow from one \
         operation, 'test.add', while applying pattern 'bad-add' to 'test.add'",
        "{outcome:?}"
    );
}

/// The same with a limit of 20,000.
#[test]
fn a_pattern_that_keeps_modifying_in_place_ends_at_a_large_limit() {
    let mut config = ConversionConfig::default();
    config.with_max_rewrites_per_operation(20_000);
    let outcome = outcome_of(
        "modify in place forever, at a limit of 20,000",
        "()",
        BODY,
        wide_target,
        with_add(|op, _, rewriter| {
            rewriter.modify_op_in_place(op, |_| ())?;
            Ok(true)
        }),
        |context| wide_converter(context),
        config,
    );
    assert!(outcome.is_error(), "{outcome:?}");
}

/// A result retyped in place to a type the converter does not answer.
#[test]
fn retyping_a_result_to_the_wrong_type_is_an_error() {
    let outcome = outcome(
        "retype a result to i32",
        with_add(|op, _, rewriter| {
            let i32 = op.borrow().context().i32();
            rewriter.modify_op_in_place(op, |op| {
                let mut result = op.results()[0];
                result.borrow_mut().set_type(i32);
            })?;
            Ok(true)
        }),
    );
    assert!(outcome.is_error(), "{outcome:?}");
}

const TWO_BLOCKS: &str = "    test.forward_br ^bb1(%0 : i128);
^bb1(%1: i128):
    test.sink %1 : (i128);
    builtin.ret;";

fn function_rule(
    context: &Rc<Context>,
    converter: &TypeConverter,
    f: impl Fn(OperationRef, &mut ConversionPatternRewriter) -> Result<bool, Report> + 'static,
) -> Rule {
    Rule::new::<Function>(context, "bad-function", Some(converter.clone()), [], move |op, _, rw| {
        f(op, rw)
    })
}

fn block_target(context: &Rc<Context>) -> ConversionTarget {
    let mut target = wide_target(context);
    target.add_dynamically_legal_op::<Function, _>(|op: &Operation| {
        let i128 = op.context().i128();
        DynamicLegalityResult::legal_if(!op.regions().iter().any(|region| {
            region.body().iter().any(|block| block.argument_types().any(|ty| ty == i128))
        }))
    });
    target.add_dynamically_legal_op::<ForwardBr, _>(|op: &Operation| {
        DynamicLegalityResult::legal_if(has_no_wide_types(op))
    });
    target
}

fn block_outcome(
    name: &str,
    f: impl Fn(OperationRef, &mut ConversionPatternRewriter) -> Result<bool, Report> + 'static,
) -> Outcome {
    outcome_of(
        name,
        "(%0: i128)",
        TWO_BLOCKS,
        block_target,
        |context, converter| {
            vec![
                function_rule(context, converter, f),
                flatten_sink(context, converter, Rc::default()),
            ]
        },
        |context| {
            let mut converter = wide_converter(context);
            split_callbacks_local(&mut converter);
            converter
        },
        ConversionConfig::default(),
    )
}

fn second_block(op: OperationRef) -> crate::BlockRef {
    let function = op.try_downcast_op::<Function>().unwrap();
    let entry = function.borrow().entry_block();
    entry.next().unwrap()
}

#[test]
fn converting_a_block_twice_is_an_error() {
    let outcome = block_outcome("convert a block twice", |op, rewriter| {
        let block = second_block(op);
        let converter = wide_converter(&op.borrow().context_rc());
        let conversion = SignatureConversion::for_block(block, &converter)?;
        rewriter.apply_signature_conversion(block, &conversion)?;
        let again = SignatureConversion::for_block(block, &converter)?;
        rewriter.apply_signature_conversion(block, &again)?;
        Ok(true)
    });
    assert!(outcome.is_error(), "{outcome:?}");
}

/// A block converted by one pattern application and again by a later one (the function is offered
/// again, since it is still illegal).
#[test]
fn converting_a_block_again_in_a_later_application_is_an_error() {
    let outcome = block_outcome("convert a block in two applications", |op, rewriter| {
        let block = second_block(op);
        let i64 = op.borrow().context().i64();
        // Each application splits the first argument again, whatever it is.
        let mut conversion = SignatureConversion::new(block.borrow().num_arguments());
        for index in 0..block.borrow().num_arguments() {
            conversion.add_inputs(index, &[i64, i64])?;
        }
        rewriter.apply_signature_conversion(block, &conversion)?;
        Ok(true)
    });
    assert!(outcome.is_error(), "{outcome:?}");
}

#[test]
fn retyping_a_block_argument_in_place_is_an_error() {
    let outcome = block_outcome("retype a block argument in place", |op, rewriter| {
        let block = second_block(op);
        let i64 = op.borrow().context().i64();
        rewriter.modify_op_in_place(op, |_| {
            let mut argument = block.borrow().arguments()[0];
            argument.borrow_mut().set_type(i64);
        })?;
        Ok(true)
    });
    assert!(outcome.is_error(), "{outcome:?}");
}

/// The same retyping outside `modify_op_in_place`, together with a tracked no-op change: the
/// rewriter cannot see it. What stops it is verification, when anything checks the type.
#[test]
fn retyping_a_block_argument_behind_the_rewriter_is_caught_by_verification_or_not_at_all() {
    let outcome = block_outcome("retype a block argument behind the rewriter", |op, rewriter| {
        let block = second_block(op);
        let i64 = op.borrow().context().i64();
        let mut argument = block.borrow().arguments()[0];
        if argument.borrow().ty() == i64 {
            return Ok(false);
        }
        argument.borrow_mut().set_type(i64);
        rewriter.modify_op_in_place(op, |_| ())?;
        Ok(true)
    });
    println!("{outcome:?}");
}

/// An argument remapped to a value defined in the block itself: an error in every build.
#[test]
fn remapping_an_argument_to_a_value_of_the_block_is_an_error() {
    let outcome = block_outcome("remap to a value of the block", |op, rewriter| {
        let block = second_block(op);
        // The first operation of the block is the sink; build a constant in the block and remap
        // the argument to it.
        let first = block.borrow().body().front().as_pointer().unwrap();
        rewriter.set_insertion_point_before(first);
        let span = op.borrow().span();
        let constant = rewriter.create_op::<Constant, _>(span, (Int::from_bits(128, 5),))?;
        let value = constant.borrow().result().as_value_ref();
        let mut conversion = SignatureConversion::new(1);
        conversion.remap_input(0, &[value])?;
        rewriter.apply_signature_conversion(block, &conversion)?;
        Ok(true)
    });
    assert!(outcome.is_error(), "{outcome:?}");
    assert!(
        outcome
            .text()
            .contains("a value defined in the block or in a region nested in it"),
        "{outcome:?}"
    );
}

// --- Materialization callbacks that misbehave ---------------------------------------------------

fn callback_outcome(name: &str, converter: impl FnOnce(&Rc<Context>) -> TypeConverter) -> Outcome {
    // The add has no pattern and is legal: it reads the split constants through source casts.
    outcome_of(
        name,
        "()",
        BODY,
        |context| {
            let legal = |op: &Operation| DynamicLegalityResult::legal_if(has_no_wide_types(op));
            let mut target = ConversionTarget::new(context.clone());
            target
                .set_unknown_op_policy(UnknownOpPolicy::Legal)
                .add_dynamically_legal_op::<Constant, _>(legal);
            target
        },
        |context, converter| vec![split_constant(context, converter)],
        converter,
        ConversionConfig::default(),
    )
}

#[test]
fn a_source_callback_that_gives_the_wrong_type_is_an_error() {
    let outcome = callback_outcome("source callback gives a half", |context| {
        let mut converter = wide_converter(context);
        converter.add_source_materialization(|_, inputs, _, _| Ok(inputs.first().copied()));
        converter
    });
    assert!(outcome.is_error(), "{outcome:?}");
}

/// A source callback that returns the result of the cast itself.
#[test]
fn a_callback_that_answers_the_cast_result_is_an_error() {
    let outcome = callback_outcome("source callback answers the cast's own result", |context| {
        let mut converter = wide_converter(context);
        converter.add_source_materialization(|rewriter, inputs, _, span| {
            let [low, high] = inputs else {
                return Ok(None);
            };
            let join = rewriter.create_op::<Join, _>(span, (*low, *high))?;
            let cast = join.as_operation_ref().next().unwrap();
            Ok(Some(cast.borrow().results()[0].borrow().as_value_ref()))
        });
        converter
    });
    assert!(outcome.is_error(), "{outcome:?}");
    assert!(outcome.text().contains("a result of the cast itself"), "{outcome:?}");
}

/// A pattern that erases the root of the conversion: an error from the rewriter.
#[test]
fn erasing_the_root_is_an_error() {
    let outcome = outcome_of(
        "erase the root",
        "()",
        "    builtin.ret;",
        |context| {
            let mut target = ConversionTarget::new(context.clone());
            target
                .set_unknown_op_policy(UnknownOpPolicy::Legal)
                .add_illegal_op::<Function>();
            target
        },
        |context, converter| {
            vec![Rule::new::<Function>(
                context,
                "erase-root",
                Some(converter.clone()),
                [],
                |op, _, rw| {
                    rw.erase_op(op)?;
                    Ok(true)
                },
            )]
        },
        |context| wide_converter(context),
        ConversionConfig::default(),
    );
    assert!(outcome.is_error(), "{outcome:?}");
    assert_eq!(
        outcome.text(),
        "'builtin.function' is the root of the conversion, which cannot be erased"
    );
}
