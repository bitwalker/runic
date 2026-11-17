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
