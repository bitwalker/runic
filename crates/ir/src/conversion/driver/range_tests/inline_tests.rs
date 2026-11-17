//! A pattern builds a structured operation: it creates the operation and a block in its region,
//! and moves into it the operations of the blocks of the operation it replaces. The arguments of
//! a moved block are replaced by values (one, several, or none), and the driver converts what was
//! moved, once, after the operation that holds it.

use super::*;
use crate::{
    BlockRef, OpPrinter, RegionKind, RegionKindInterface, SourceSpan,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    effects::MemoryEffectOpInterface,
    traits::{NoTerminator, SingleBlock},
};

/// An operation with two regions of one block each, whose bodies a pattern moves:
/// `test.twin first { .. } second { ^bb0(%x: i32): .. };`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, NoTerminator),
    implements(RegionKindInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Twin {
    #[region]
    first: Region,
    #[region]
    second: Region,
}

impl RegionKindInterface for Twin {
    fn kind(&self) -> RegionKind {
        RegionKind::SSA
    }
}

/// An operation with one region of one block, which a pattern builds to hold what it moves:
/// `test.seq { .. };`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, NoTerminator),
    implements(RegionKindInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Seq {
    #[region]
    body: Region,
}

impl RegionKindInterface for Seq {
    fn kind(&self) -> RegionKind {
        RegionKind::SSA
    }
}

/// An operation with one region of one block, which the tests mark recursively legal:
/// `test.hold { .. };`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, NoTerminator),
    implements(RegionKindInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Hold {
    #[region]
    body: Region,
}

impl RegionKindInterface for Hold {
    fn kind(&self) -> RegionKind {
        RegionKind::SSA
    }
}

/// The entry blocks of the two regions of `twin`.
fn twin_blocks(twin: OperationRef) -> (BlockRef, BlockRef) {
    let twin = twin.try_downcast_op::<Twin>().unwrap();
    let twin = twin.borrow();
    (
        twin.first().entry_block_ref().unwrap(),
        twin.second().entry_block_ref().unwrap(),
    )
}

/// The entry block of the region of a `test.hold`.
fn hold_block(hold: OperationRef) -> BlockRef {
    hold.try_downcast_op::<Hold>()
        .unwrap()
        .borrow()
        .body()
        .entry_block_ref()
        .unwrap()
}

/// How [merge_twin] computes the values of the second block's arguments, before the twin.
type ArgumentValues =
    dyn Fn(BlockRef, &mut ConversionPatternRewriter) -> Result<Vec<Vec<ValueRef>>, Report>;

/// A `test.twin` becomes a `test.seq` whose block holds the first region's body followed by the
/// second's; the second's arguments are replaced by what `values` computes. `calls` counts the
/// pattern's applications.
fn merge_twin(
    context: &Rc<Context>,
    converter: Option<TypeConverter>,
    calls: Rc<RefCell<usize>>,
    values: impl Fn(BlockRef, &mut ConversionPatternRewriter) -> Result<Vec<Vec<ValueRef>>, Report>
    + 'static,
) -> Rule {
    let values: Box<ArgumentValues> = Box::new(values);
    Rule::new::<Twin>(
        context,
        "merge-twin",
        converter,
        [registered::<Constant>(context), registered::<Seq>(context)],
        move |op, _operands, rewriter| {
            *calls.borrow_mut() += 1;
            let (first, second) = twin_blocks(op);
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let ranges = values(second, rewriter)?;
            let ranges = ranges.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(first, block, &[])?;
            rewriter.inline_block_at_end(second, block, &ranges)?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    )
}

/// One `i32` constant per argument of `block`, numbered from 7.
fn i32_constants(
    block: BlockRef,
    rewriter: &mut ConversionPatternRewriter,
) -> Result<Vec<Vec<ValueRef>>, Report> {
    let count = block.borrow().num_arguments();
    let mut values = vec![];
    for index in 0..count {
        let value = 7 + i32::try_from(index).unwrap();
        let constant =
            rewriter.create_op::<Constant, _>(SourceSpan::default(), (Int::from(value),))?;
        values.push(vec![constant.borrow().result().as_value_ref()]);
    }
    Ok(values)
}

/// The target of these tests: [wide_target], with `test.twin` illegal.
fn twin_target(context: &Rc<Context>) -> ConversionTarget {
    let mut target = wide_target(context);
    target.add_illegal_op::<Twin>();
    target
}

fn register_ops(context: &Rc<Context>) {
    registered::<Twin>(context);
    registered::<Seq>(context);
    registered::<Hold>(context);
}

/// Parses `body` as the body of `@f` with `signature`, with this file's operations registered.
fn parse_twin_function(signature: &str, body: &str) -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    register_ops(&context);
    let function = parse::<Function>(
        ParserConfig::new(context.clone()),
        "conversion",
        &function_text(signature, body),
    )
    .into_result()
    .unwrap_or_else(|err| panic!("{}", describe(&err)));
    (context, function.as_operation_ref())
}

/// The operation of `function`'s entry block at `index`.
fn entry_op(function: OperationRef, index: usize) -> OperationRef {
    let function = function.try_downcast_op::<Function>().unwrap();
    let entry = function.borrow().entry_block();
    entry.borrow().body().iter().nth(index).unwrap().as_operation_ref()
}

/// A two-region operation becomes a one-region operation whose block holds the first region's
/// body followed by the second's; the second region's argument is replaced by a value the
/// pattern computes, one to one.
#[test]
fn a_pattern_merges_two_regions_into_the_block_of_an_operation_it_creates() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
    } second {
    ^bb0(%1: i32):
        test.sink %1, %1 : (i32, i32);
    };
    builtin.ret;",
    );
    let calls = Rc::new(RefCell::new(0));

    convert(
        function,
        twin_target(&context),
        [merge_twin(&context, None, calls.clone(), i32_constants)],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(*calls.borrow(), 1);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %2 = test.constant 7 : i32;
    test.seq {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
        test.sink %2, %2 : (i32, i32);
    };
    builtin.ret;"
        )
    );
}

/// Under the wide converter, an `i128` argument of the second region is replaced by two `i64`
/// values: a user with a pattern (`test.add`) is given them, and a legal user that stays
/// (`test.low`) reads them through the cast, which the source callback realizes as a `test.join`.
#[test]
fn an_argument_replaced_by_two_values_reaches_a_converted_user_and_a_legal_one() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
    } second {
    ^bb0(%1: i128):
        %2 = test.add %1, %1;
        test.sink %2 : (i128);
        %3 = test.low %1;
        test.sink %3 : (i64);
    };
    builtin.ret;",
    );
    let mut converter = wide_converter(&context);
    let offers = Rc::new(RefCell::new(vec![]));
    join_halves(&mut converter, offers.clone());
    let adds = Rc::new(RefCell::new(vec![]));
    let halves = |_: BlockRef, rewriter: &mut ConversionPatternRewriter| {
        let span = SourceSpan::default();
        let low = rewriter.create_op::<Constant, _>(span, (Int::from_bits(64, 5),))?;
        let high = rewriter.create_op::<Constant, _>(span, (Int::from_bits(64, 0),))?;
        let low = low.borrow().result().as_value_ref();
        let high = high.borrow().result().as_value_ref();
        Ok(vec![vec![low, high]])
    };

    convert(
        function,
        twin_target(&context),
        [
            merge_twin(&context, Some(converter.clone()), Rc::default(), halves),
            split_add(&context, &converter, adds.clone()),
            flatten_sink(&context, &converter, Rc::default()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let i64 = context.i64();
    assert_eq!(adds.borrow().len(), 1);
    for range in &adds.borrow()[0] {
        assert_eq!(types(range), [i64, i64]);
    }
    assert_eq!(*offers.borrow(), [(vec![i64, i64], vec![context.i128()])]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %4 = test.constant 5 : i64;
    %5 = test.constant 0 : i64;
    test.seq {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
        %10 = test.join %4, %5;
        %7 = test.add %4, %4;
        %8 = test.add %5, %5;
        test.sink %7, %8 : (i64, i64);
        %3 = test.low %10;
        test.sink %3 : (i64);
    };
    builtin.ret;"
        )
    );
}

/// An argument whose type converts to no type is replaced by no values: its user is given an
/// empty range, and no cast remains.
#[test]
fn an_argument_whose_type_converts_to_nothing_is_replaced_by_no_values() {
    let (context, function) = parse_twin_function(
        "(%0: ptr<array<i8; 0>>)",
        "    test.twin first {
        test.sink %0 : (ptr<array<i8; 0>>);
    } second {
    ^bb0(%1: array<i8; 0>):
        test.store %0, %1 : (ptr<array<i8; 0>>, array<i8; 0>);
    };
    builtin.ret;",
    );
    let pointer = defined_values(function)[0];
    let converter = wide_converter(&context);
    let stores = Rc::new(RefCell::new(vec![]));

    convert(
        function,
        twin_target(&context),
        [
            merge_twin(&context, Some(converter.clone()), Rc::default(), |_, _| Ok(vec![vec![]])),
            drop_store(&context, &converter, stores.clone()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(*stores.borrow(), [vec![vec![pointer], vec![]]]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "(%0: ptr<array<i8; 0>>)",
            "    test.seq {
        test.sink %0 : (ptr<array<i8; 0>>);
    };
    builtin.ret;"
        )
    );
}

/// The moved body holds an operation with its own pattern, converted once, and a `test.twin`
/// that is itself converted by moving its bodies: two levels.
#[test]
fn moved_operations_are_converted_once_at_two_levels() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        %0 = test.constant 1 : i128;
        test.twin first {
            %1 = test.add %0, %0;
            test.sink %1 : (i128);
        } second {
        ^bb0(%2: i32):
            test.sink %2 : (i32);
        };
    } second {
    ^bb0(%3: i32):
        test.sink %3 : (i32);
    };
    builtin.ret;",
    );
    let converter = wide_converter(&context);
    let twins = Rc::new(RefCell::new(0));
    let adds = Rc::new(RefCell::new(vec![]));
    let sinks = Rc::new(RefCell::new(vec![]));

    convert(
        function,
        twin_target(&context),
        [
            merge_twin(&context, None, twins.clone(), i32_constants),
            split_constant(&context, &converter),
            split_add(&context, &converter, adds.clone()),
            flatten_sink(&context, &converter, sinks.clone()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(*twins.borrow(), 2);
    assert_eq!(adds.borrow().len(), 1);
    assert_eq!(sinks.borrow().len(), 1);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %4 = test.constant 7 : i32;
    test.seq {
        %5 = test.constant 1 : i64;
        %6 = test.constant 0 : i64;
        %8 = test.constant 7 : i32;
        test.seq {
            %9 = test.add %5, %5;
            %10 = test.add %6, %6;
            test.sink %9, %10 : (i64, i64);
            test.sink %8 : (i32);
        };
        test.sink %4 : (i32);
    };
    builtin.ret;"
        )
    );
}

/// In a full conversion, an illegal operation that was moved and that no pattern converts is an
/// error that names it.
#[test]
fn a_moved_illegal_operation_without_a_pattern_is_an_error_that_names_it() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        %0 = test.constant 1 : i128;
        test.sink %0 : (i128);
    } second {
    ^bb0(%1: i32):
        test.sink %1 : (i32);
    };
    builtin.ret;",
    );
    let converter = wide_converter(&context);

    let err = convert(
        function,
        twin_target(&context),
        [
            merge_twin(&context, None, Rc::default(), i32_constants),
            split_constant(&context, &converter),
        ],
    )
    .unwrap_err();

    assert_eq!(
        format!("{err}"),
        "failed to legalize operation 'test.sink': no legalization path to target; dynamic \
         legality predicate returned illegal"
    );
}

/// In a partial conversion, every moved operation is offered once: those the walk still reaches
/// where it reaches them, and one moved out of a recursively legal operation, which the walk does
/// not enter, after it was moved.
#[test]
fn every_moved_operation_is_offered_once_and_none_is_skipped() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        test.hold {
            %0 = test.constant 1 : i32;
            test.sink %0 : (i32);
        };
        %1 = test.constant 2 : i32;
        test.sink %1 : (i32);
    } second {
    ^bb0(%2: i32):
        test.sink %2 : (i32);
    };
    builtin.ret;",
    );
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Illegal)
        .add_illegal_op::<Twin>()
        .add_legal_op::<Seq>()
        .add_legal_op::<Constant>()
        .add_legal_op::<Hold>()
        .mark_op_recursively_legal::<Hold>();
    let unhold = Rule::new::<Twin>(
        &context,
        "unhold-twin",
        None,
        [registered::<Constant>(&context), registered::<Seq>(&context)],
        |op, _operands, rewriter| {
            let (first, second) = twin_blocks(op);
            let hold = first.borrow().body().front().as_pointer().unwrap();
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let constant = rewriter.create_op::<Constant, _>(span, (Int::from(7i32),))?;
            let constant = constant.borrow().result().as_value_ref();
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(hold_block(hold), block, &[])?;
            rewriter.erase_op(hold)?;
            rewriter.inline_block_at_end(first, block, &[])?;
            rewriter.inline_block_at_end(second, block, &[&[constant]])?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    );
    let offered = Rc::new(RefCell::new(Vec::<OperationRef>::new()));
    let seen = offered.clone();
    let count_sinks = Rule::new::<Sink>(&context, "count-sinks", None, [], move |op, _, _| {
        seen.borrow_mut().push(op);
        Ok(false)
    });
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(unhold);
    set.push(count_sinks);

    apply_partial_conversion(function, target, set, ConversionConfig::default())
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let offered = offered.borrow();
    assert_eq!(offered.len(), 3, "each of the three sinks once");
    for (index, sink) in offered.iter().enumerate() {
        assert!(!offered[..index].contains(sink), "a sink offered twice");
    }
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %3 = test.constant 7 : i32;
    test.seq {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
        %1 = test.constant 2 : i32;
        test.sink %1 : (i32);
        test.sink %3 : (i32);
    };
    builtin.ret;"
        )
    );
}

/// A pattern rooted at `test.twin` whose rewrite is `rewrite`, declaring `test.seq`.
fn twin_rule(
    context: &Rc<Context>,
    name: &'static str,
    rewrite: impl Fn(OperationRef, &mut ConversionPatternRewriter) -> Result<bool, Report> + 'static,
) -> Rule {
    Rule::new::<Twin>(context, name, None, [registered::<Seq>(context)], move |op, _, rewriter| {
        rewrite(op, rewriter)
    })
}

const TWIN_AFTER_HOLD: &str = "    test.hold {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
    };
    test.twin first {
        %1 = test.constant 2 : i32;
        test.sink %1 : (i32);
    } second {
    ^bb0(%2: i32):
        test.sink %2 : (i32);
    };
    builtin.ret;";

/// A block is created only in a region of an operation the pattern created.
#[test]
fn creating_a_block_in_a_region_the_pattern_did_not_create_is_an_error() {
    let (context, function) = parse_twin_function("()", TWIN_AFTER_HOLD);
    let rule = twin_rule(&context, "block-in-twin", |op, rewriter| {
        let region = op.try_downcast_op::<Twin>().unwrap().borrow().first().as_region_ref();
        rewriter.create_block(region, &[])?;
        Ok(true)
    });

    let err = convert(function, twin_target(&context), [rule]).unwrap_err();

    assert_eq!(
        format!("{err}"),
        "a block can be created only in a region of an operation the pattern created, but this \
         region is of 'test.twin', which the pattern did not create"
    );
}

/// A pattern inlines only a block of the operation it is given (or of one nested in it).
#[test]
fn inlining_a_block_of_another_operation_is_an_error() {
    let (context, function) = parse_twin_function("()", TWIN_AFTER_HOLD);
    let hold = entry_op(function, 0);
    let rule = twin_rule(&context, "inline-hold", move |op, rewriter| {
        rewriter.inline_block_before(hold_block(hold), op, &[])?;
        Ok(true)
    });

    let err = convert(function, twin_target(&context), [rule]).unwrap_err();

    assert_eq!(
        format!("{err}"),
        "a pattern may inline only a block of the operation it is given, 'test.twin', or of an \
         operation nested in it, but the block is a block of 'test.hold'"
    );
}

/// A pattern inlines a block only into a block of an operation it created or into one that holds
/// the operation it is given.
#[test]
fn inlining_into_a_block_of_another_operation_is_an_error() {
    let (context, function) = parse_twin_function("()", TWIN_AFTER_HOLD);
    let hold = entry_op(function, 0);
    let rule = twin_rule(&context, "inline-into-hold", move |op, rewriter| {
        let (first, _) = twin_blocks(op);
        rewriter.inline_block_at_end(first, hold_block(hold), &[])?;
        Ok(true)
    });

    let err = convert(function, twin_target(&context), [rule]).unwrap_err();

    assert_eq!(
        format!("{err}"),
        "a pattern may inline a block only into a block of an operation it created or into a \
         block that holds the operation it is given, 'test.twin', but the destination is a block \
         of 'test.hold', which is neither"
    );
}

/// A pattern that inlines a block and then fails broke the pattern contract: the moves are not
/// undone, and the driver reports it.
#[test]
fn a_pattern_that_inlines_and_then_fails_is_reported() {
    let (context, function) = parse_twin_function("()", TWIN_AFTER_HOLD);
    let rule = twin_rule(&context, "inline-then-fail", |op, rewriter| {
        let (first, _) = twin_blocks(op);
        rewriter.inline_block_before(first, op, &[])?;
        Err(Report::msg("the pattern gives up"))
    });

    let err = format!("{}", convert(function, twin_target(&context), [rule]).unwrap_err());

    assert!(
        err.starts_with(
            "conversion pattern 'inline-then-fail' failed after mutating operation 'test.twin'"
        ),
        "{err}"
    );
    assert!(err.ends_with(": the pattern gives up"), "{err}");
}

/// A pattern that inlines a block and then answers that it did not match broke the pattern
/// contract, and the driver reports it.
#[test]
fn a_pattern_that_inlines_and_then_does_not_match_is_reported() {
    let (context, function) = parse_twin_function("()", TWIN_AFTER_HOLD);
    let rule = twin_rule(&context, "inline-then-no-match", |op, rewriter| {
        let (first, _) = twin_blocks(op);
        rewriter.inline_block_before(first, op, &[])?;
        Ok(false)
    });

    let err = format!("{}", convert(function, twin_target(&context), [rule]).unwrap_err());

    assert!(
        err.starts_with(
            "conversion pattern 'inline-then-no-match' mutated operation 'test.twin' but returned \
             no match"
        ),
        "{err}"
    );
}

/// The block an operation is inlined into is held directly by an operation the pattern created:
/// a block of an operation the pattern only moved into one it created is refused, since the moved
/// operation is converted after the operations moved into it would be.
#[test]
fn inlining_into_a_block_of_an_operation_the_pattern_moved_is_an_error() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
    } second {
    ^bb0(%1: i32):
        test.twin first {
            test.sink %1 : (i32);
        } second {
        ^bb1(%2: i32):
            test.sink %2 : (i32);
        };
    };
    builtin.ret;",
    );
    let rule = Rule::new::<Twin>(
        &context,
        "into-moved-twin",
        None,
        [registered::<Constant>(&context), registered::<Seq>(&context)],
        |op, _operands, rewriter| {
            let (first, second) = twin_blocks(op);
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let constant = rewriter.create_op::<Constant, _>(span, (Int::from(7i32),))?;
            let constant = constant.borrow().result().as_value_ref();
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(second, block, &[&[constant]])?;
            let inner = block.borrow().body().back().as_pointer().unwrap();
            let (inner_first, _) = twin_blocks(inner);
            rewriter.inline_block_at_end(first, inner_first, &[])?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    );

    let err = format!("{}", convert(function, twin_target(&context), [rule]).unwrap_err());

    assert!(
        err.ends_with(
            "a pattern may inline a block only into a block of an operation it created or into a \
             block that holds the operation it is given, 'test.twin', but the destination is a \
             block of 'test.twin', which is neither"
        ),
        "{err}"
    );
}

/// An operation offered before the walk reaches it (a pattern changed it in place) is not
/// offered again by the walk, nor when a pattern then moves it.
#[test]
fn an_operation_offered_early_is_not_offered_again_when_it_is_moved() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
    } second {
    ^bb0(%1: i32):
        test.sink %1 : (i32);
    };
    builtin.ret;",
    );
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Illegal)
        .add_illegal_op::<Twin>()
        .add_legal_op::<Seq>()
        .add_legal_op::<Constant>();
    let calls = Rc::new(RefCell::new(0));
    let touch = calls.clone();
    // The first application changes the first sink in place; the second merges the regions.
    let rule = Rule::new::<Twin>(
        &context,
        "touch-then-merge",
        None,
        [registered::<Constant>(&context), registered::<Seq>(&context)],
        move |op, _operands, rewriter| {
            *touch.borrow_mut() += 1;
            let (first, second) = twin_blocks(op);
            if *touch.borrow() == 1 {
                let sink = first.borrow().body().back().as_pointer().unwrap();
                rewriter.modify_op_in_place(sink, |_| ())?;
                return Ok(true);
            }
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let constant = rewriter.create_op::<Constant, _>(span, (Int::from(7i32),))?;
            let constant = constant.borrow().result().as_value_ref();
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(first, block, &[])?;
            rewriter.inline_block_at_end(second, block, &[&[constant]])?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    );
    let offered = Rc::new(RefCell::new(Vec::<OperationRef>::new()));
    let seen = offered.clone();
    let count_sinks = Rule::new::<Sink>(&context, "count-sinks", None, [], move |op, _, _| {
        seen.borrow_mut().push(op);
        Ok(false)
    });
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(rule);
    set.push(count_sinks);

    apply_partial_conversion(function, target, set, ConversionConfig::default())
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(*calls.borrow(), 2);
    let offered = offered.borrow();
    assert_eq!(offered.len(), 2, "each of the two sinks once");
    assert_ne!(offered[0], offered[1], "a sink offered twice");
}

/// An operation moved out of a recursively legal operation, which the walk does not reach, into
/// a block whose holder has not been offered yet, waits for that holder: it is offered after it.
#[test]
fn an_operation_moved_out_of_a_recursively_legal_operation_waits_for_its_new_holder() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
    } second {
    ^bb0(%0: i32):
    } attributes { poke = #builtin.bool<true> };
    test.seq {
        test.twin first {
            test.hold {
                %1 = test.constant 1 : i32;
                test.sink %1 : (i32);
            };
        } second {
        ^bb1(%2: i32):
            test.sink %2 : (i32);
        };
    };
    builtin.ret;",
    );
    let holder = entry_op(function, 1);
    let twin = holder
        .try_downcast_op::<Seq>()
        .unwrap()
        .borrow()
        .body()
        .entry()
        .body()
        .front()
        .as_pointer()
        .unwrap();
    let hold = twin_blocks(twin).0.borrow().body().front().as_pointer().unwrap();
    let held_sink = hold_block(hold).borrow().body().back().as_pointer().unwrap();
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Illegal)
        .add_illegal_op::<Twin>()
        .add_legal_op::<Constant>()
        .add_legal_op::<Hold>()
        .mark_op_recursively_legal::<Hold>();
    // The twin marked `poke` changes the other twin in place, so that it is offered before the
    // walk reaches the `test.seq` that holds it, and is erased; the other moves the body of its
    // `test.hold` before itself, into that `test.seq`, and merges its regions.
    let rule = Rule::new::<Twin>(
        &context,
        "poke-or-unhold",
        None,
        [registered::<Constant>(&context), registered::<Seq>(&context)],
        move |op, _operands, rewriter| {
            if op.borrow().get_attribute("poke").is_some() {
                rewriter.modify_op_in_place(twin, |_| ())?;
                rewriter.erase_op(op)?;
                return Ok(true);
            }
            let (first, second) = twin_blocks(op);
            let hold = first.borrow().body().front().as_pointer().unwrap();
            rewriter.inline_block_before(hold_block(hold), op, &[])?;
            rewriter.erase_op(hold)?;
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let constant = rewriter.create_op::<Constant, _>(span, (Int::from(7i32),))?;
            let constant = constant.borrow().result().as_value_ref();
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(first, block, &[])?;
            rewriter.inline_block_at_end(second, block, &[&[constant]])?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    );
    let log = Rc::new(RefCell::new(Vec::<&'static str>::new()));
    let seqs = log.clone();
    let record_seqs = Rule::new::<Seq>(&context, "record-seqs", None, [], move |op, _, _| {
        seqs.borrow_mut().push(if op == holder { "holder" } else { "seq" });
        Ok(false)
    });
    let sinks = log.clone();
    let record_sinks = Rule::new::<Sink>(&context, "record-sinks", None, [], move |op, _, _| {
        sinks.borrow_mut().push(if op == held_sink { "held sink" } else { "sink" });
        Ok(false)
    });
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(rule);
    set.push(record_seqs);
    set.push(record_sinks);

    apply_partial_conversion(function, target, set, ConversionConfig::default())
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(*log.borrow(), ["seq", "holder", "held sink", "sink"]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    test.seq {
        %1 = test.constant 1 : i32;
        test.sink %1 : (i32);
        %3 = test.constant 7 : i32;
        test.seq {
            test.sink %3 : (i32);
        };
    };
    builtin.ret;"
        )
    );
}

/// An operation moved into a recursively legal operation the pattern created is not offered, as
/// nothing under a recursively legal operation is.
#[test]
fn an_operation_moved_into_a_recursively_legal_operation_is_not_offered() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        %0 = test.constant 1 : i32;
        test.sink %0 : (i32);
    } second {
    ^bb0(%1: i32):
        test.sink %1 : (i32);
    };
    builtin.ret;",
    );
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Illegal)
        .add_illegal_op::<Twin>()
        .add_legal_op::<Seq>()
        .add_legal_op::<Constant>()
        .add_legal_op::<Hold>()
        .mark_op_recursively_legal::<Hold>();
    let rule = Rule::new::<Twin>(
        &context,
        "hold-first",
        None,
        [
            registered::<Constant>(&context),
            registered::<Seq>(&context),
            registered::<Hold>(&context),
        ],
        |op, _operands, rewriter| {
            let (first, second) = twin_blocks(op);
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let constant = rewriter.create_op::<Constant, _>(span, (Int::from(7i32),))?;
            let constant = constant.borrow().result().as_value_ref();
            let hold = rewriter.create_op::<Hold, _>(span, ())?;
            let region = hold.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(first, block, &[])?;
            rewriter.set_insertion_point_before(op);
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(second, block, &[&[constant]])?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    );
    let offered = Rc::new(RefCell::new(0));
    let seen = offered.clone();
    let count_sinks = Rule::new::<Sink>(&context, "count-sinks", None, [], move |_, _, _| {
        *seen.borrow_mut() += 1;
        Ok(false)
    });
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(rule);
    set.push(count_sinks);

    apply_partial_conversion(function, target, set, ConversionConfig::default())
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(*offered.borrow(), 1, "only the sink of the `test.seq`");
}

/// A pattern that moves the only block of a region out of an operation nested in the one it is
/// given, and leaves that operation in the IR, is an error that names the pattern.
#[test]
fn leaving_an_operation_whose_region_a_pattern_emptied_is_an_error() {
    let (context, function) = parse_twin_function(
        "()",
        "    test.twin first {
        test.twin first {
            %0 = test.constant 1 : i32;
            test.sink %0 : (i32);
        } second {
        ^bb0(%1: i32):
            test.sink %1 : (i32);
        };
    } second {
    ^bb1(%2: i32):
        test.sink %2 : (i32);
    };
    builtin.ret;",
    );
    let rule = Rule::new::<Twin>(
        &context,
        "empty-the-inner-twin",
        None,
        [registered::<Constant>(&context), registered::<Seq>(&context)],
        |op, _operands, rewriter| {
            let (first, second) = twin_blocks(op);
            let inner = first.borrow().body().front().as_pointer().unwrap();
            let (inner_first, _) = twin_blocks(inner);
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let constant = rewriter.create_op::<Constant, _>(span, (Int::from(7i32),))?;
            let constant = constant.borrow().result().as_value_ref();
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(inner_first, block, &[])?;
            rewriter.inline_block_at_end(first, block, &[])?;
            rewriter.inline_block_at_end(second, block, &[&[constant]])?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    );

    let err = convert(function, twin_target(&context), [rule]).unwrap_err();

    assert_eq!(
        format!("{err}"),
        "conversion pattern 'empty-the-inner-twin' moved every block out of a region of \
         'test.twin' and left it in the IR"
    );
}

/// Parses `body` as the body of `builtin.module @m`, with this file's operations and
/// `test.forward_br` registered.
fn parse_twin_module(body: &str) -> (Rc<Context>, OperationRef) {
    use crate::dialects::builtin::Module;
    let context = Rc::new(Context::default());
    register_ops(&context);
    registered::<super::signature_tests::ForwardBr>(&context);
    let module = parse::<Module>(
        ParserConfig::new(context.clone()),
        "conversion",
        &format!("builtin.module public @m {{\n{body}\n}};"),
    )
    .into_result()
    .unwrap_or_else(|err| panic!("{}", describe(&err)));
    (context, module.as_operation_ref())
}

/// The target of the tests that inline a function's blocks: `builtin.function` is illegal, and
/// `test.seq` and every other operation legal.
fn function_target(context: &Rc<Context>) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_illegal_op::<Function>()
        .add_legal_op::<Seq>();
    target
}

/// A pattern rooted at `builtin.function` that creates a `test.seq` before the function, moves
/// the function's entry block to the end of the seq's block, and erases the function.
fn inline_function_entry(context: &Rc<Context>) -> Rule {
    Rule::new::<Function>(
        context,
        "inline-function-entry",
        None,
        [registered::<Seq>(context)],
        |op, _operands, rewriter| {
            let entry = op.try_downcast_op::<Function>().unwrap().borrow().entry_block();
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            let block = rewriter.create_block(region, &[])?;
            rewriter.inline_block_at_end(entry, block, &[])?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    )
}

/// A block that ends in a branch is not inlined: the branch would leave the region that holds its
/// successor.
#[test]
fn inlining_a_block_that_ends_in_a_branch_is_an_error() {
    let (context, module) = parse_twin_module(
        "    builtin.function public extern(\"C\") @f() {
        test.forward_br ^bb1;
    ^bb1:
        builtin.ret;
    };",
    );
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(inline_function_entry(&context));

    let err = apply_partial_conversion(
        module,
        function_target(&context),
        set,
        ConversionConfig::default(),
    )
    .unwrap_err();

    let err = format!("{err}");
    assert!(
        err.ends_with(
            "a block of 'builtin.function' ends in a branch and cannot be inlined: the branch \
             would leave the region its successors are in"
        ),
        "{err}"
    );
}

/// Only the one block of a region is inlined: a block of a region of several is refused, even
/// when it does not branch.
#[test]
fn inlining_a_block_of_a_region_of_several_blocks_is_an_error() {
    let (context, module) = parse_twin_module(
        "    builtin.function public extern(\"C\") @f() {
        builtin.ret;
    ^bb1:
        builtin.ret;
    };",
    );
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(inline_function_entry(&context));

    let err = apply_partial_conversion(
        module,
        function_target(&context),
        set,
        ConversionConfig::default(),
    )
    .unwrap_err();

    let err = format!("{err}");
    assert!(
        err.ends_with(
            "a block of 'builtin.function' is one of the 2 blocks of its region and cannot be \
             inlined: only the one block of a region can be inlined"
        ),
        "{err}"
    );
}

/// An operation a block of which is the successor of a branch outside it is not erased: the
/// branch would be left without its successor.
#[test]
fn erasing_an_operation_whose_block_a_branch_outside_it_targets_is_an_error() {
    use super::signature_tests::ForwardBr;
    let (context, module) = parse_twin_module(
        "    builtin.function public extern(\"C\") @f() {
        builtin.ret;
    ^bb1:
        builtin.ret;
    };",
    );
    let rule = Rule::new::<Function>(
        &context,
        "branch-in-then-erase",
        None,
        [registered::<Seq>(&context), registered::<ForwardBr>(&context)],
        |op, _operands, rewriter| {
            let entry = op.try_downcast_op::<Function>().unwrap().borrow().entry_block();
            let target = entry.next().unwrap();
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            let seq = rewriter.create_op::<Seq, _>(span, ())?;
            let region = seq.borrow().body().as_region_ref();
            rewriter.create_block(region, &[])?;
            rewriter.create_op::<ForwardBr, _>(span, (target, Vec::<ValueRef>::new()))?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    );
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(rule);

    let err = format!(
        "{}",
        apply_partial_conversion(
            module,
            function_target(&context),
            set,
            ConversionConfig::default()
        )
        .unwrap_err()
    );

    assert!(
        err.ends_with(
            "'builtin.function' cannot be erased: a block in it is the successor of \
             'test.forward_br', which is not inside it and would be left without its successor"
        ),
        "{err}"
    );
}
