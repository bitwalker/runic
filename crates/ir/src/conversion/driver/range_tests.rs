//! The driver gives a pattern each operand as the values it became, and lets it replace each result
//! by a range of values.
//!
//! The conversions here split an `i128` into two `i64` halves (low, high) and drop a zero-sized
//! marker, `array<i8; 0>`. The patterns are written for the tests: a split `test.add` adds the
//! halves without a carry, which is enough to see where the values go.

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cell::RefCell;

use pretty_assertions::assert_str_eq;
use smallvec::SmallVec;

use super::*;
use crate::{
    Context, Int, OpRegistration, Operation, OperationName, Overflow, Type, Value,
    conversion::{DynamicLegalityResult, TypeConversion, UnknownOpPolicy},
    dialects::{
        builtin::Function,
        test::{Add, Constant, High, Join, Load, Low, Sink, Store, TestDialect},
    },
    parse::{ParserConfig, parse},
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind},
    testing::text::describe,
};

mod error_tests;
mod in_place_tests;
mod inline_tests;
mod partial_tests;
mod random_conversions;
mod signature_tests;

type Rewrite = dyn Fn(
    OperationRef,
    ConvertedOperands<'_>,
    &mut ConversionPatternRewriter,
) -> Result<bool, Report>;

/// One range of values per operand.
type Ranges = Vec<Vec<ValueRef>>;

/// What a pattern was given, per call.
type Seen<T> = Rc<RefCell<Vec<T>>>;

/// A conversion pattern rooted at `Root` whose rewrite is a closure.
struct Rule {
    info: PatternInfo,
    converter: Option<TypeConverter>,
    rewrite: Box<Rewrite>,
}

impl Rule {
    fn new<Root: OpRegistration>(
        context: &Rc<Context>,
        name: &'static str,
        converter: Option<TypeConverter>,
        generated: impl IntoIterator<Item = OperationName>,
        rewrite: impl Fn(
            OperationRef,
            ConvertedOperands<'_>,
            &mut ConversionPatternRewriter,
        ) -> Result<bool, Report>
        + 'static,
    ) -> Self {
        let root = registered::<Root>(context);
        let mut info = PatternInfo::new(
            context.clone(),
            name,
            PatternKind::Operation(root),
            PatternBenefit::new(1),
        );
        info.with_generated_ops(generated);
        Self {
            info,
            converter,
            rewrite: Box::new(rewrite),
        }
    }
}

impl Pattern for Rule {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for Rule {
    fn type_converter(&self) -> Option<&TypeConverter> {
        self.converter.as_ref()
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        (self.rewrite)(op, operands, rewriter)
    }
}

fn registered<T: OpRegistration>(context: &Rc<Context>) -> OperationName {
    context
        .get_or_register_dialect::<<T as OpRegistration>::Dialect>()
        .expect_registered_name::<T>()
}

/// The zero-sized marker type, `array<i8; 0>`.
fn marker(context: &Context) -> Type {
    context.array_type(context.i8(), 0)
}

/// `i128` becomes `(i64, i64)`, the marker becomes nothing, and every other type is itself.
fn wide_converter(context: &Context) -> TypeConverter {
    let (i64, i128, marker) = (context.i64(), context.i128(), marker(context));
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| {
        if ty == i128 {
            Some(TypeConversion::many([i64, i64]))
        } else if ty == marker {
            Some(TypeConversion::Drop)
        } else {
            None
        }
    });
    converter
}

/// Whether `op` has no operand or result of a type [wide_converter] changes.
fn has_no_wide_types(op: &Operation) -> bool {
    let context = op.context();
    let (i128, marker) = (context.i128(), marker(context));
    let wide = |ty: Type| ty == i128 || ty == marker;
    !op.operands().iter().any(|operand| wide(operand.borrow().ty()))
        && !op.results().iter().any(|result| wide(result.borrow().ty()))
}

/// Every operation is legal, except the `test.constant`, `test.add`, `test.sink`, `test.load`
/// and `test.store` that use a type [wide_converter] changes.
fn wide_target(context: &Rc<Context>) -> ConversionTarget {
    let legal = |op: &Operation| DynamicLegalityResult::legal_if(has_no_wide_types(op));
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Constant, _>(legal)
        .add_dynamically_legal_op::<Add, _>(legal)
        .add_dynamically_legal_op::<Sink, _>(legal)
        .add_dynamically_legal_op::<Load, _>(legal)
        .add_dynamically_legal_op::<Store, _>(legal);
    target
}

/// The values of `range`, as a vector.
fn values(range: &[ValueRef]) -> Vec<ValueRef> {
    range.to_vec()
}

/// The ranges of every operand, as vectors.
fn ranges_of(operands: &ConvertedOperands<'_>) -> Ranges {
    operands.iter().map(values).collect()
}

/// An `i128` `test.constant` becomes two `i64` constants, its low and high halves.
fn split_constant(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    let i128 = context.i128();
    Rule::new::<Constant>(
        context,
        "split-constant",
        Some(converter.clone()),
        [registered::<Constant>(context)],
        move |op, _operands, rewriter| {
            let constant = op.try_downcast_op::<Constant>().unwrap();
            if constant.borrow().result().ty() != i128 {
                return Ok(false);
            }
            let bits = constant.borrow().get_value().unsigned();
            let span = op.borrow().span();
            let low = rewriter.create_op::<Constant, _>(span, (Int::from_bits(64, bits),))?;
            let high =
                rewriter.create_op::<Constant, _>(span, (Int::from_bits(64, bits >> 64),))?;
            let low = low.borrow().result().as_value_ref();
            let high = high.borrow().result().as_value_ref();
            rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
            Ok(true)
        },
    )
}

/// An `i128` `test.add` becomes two wrapping `i64` adds of the halves; `seen` gets the ranges
/// of each call.
fn split_add(context: &Rc<Context>, converter: &TypeConverter, seen: Seen<Ranges>) -> Rule {
    let i128 = context.i128();
    Rule::new::<Add>(
        context,
        "split-add",
        Some(converter.clone()),
        [registered::<Add>(context)],
        move |op, operands, rewriter| {
            if op.borrow().results()[0].borrow().ty() != i128 {
                return Ok(false);
            }
            seen.borrow_mut().push(ranges_of(&operands));
            let ([a_low, a_high], [b_low, b_high]) = (operands.get(0), operands.get(1)) else {
                return Ok(false);
            };
            let span = op.borrow().span();
            let low = rewriter.create_op::<Add, _>(span, (*a_low, *b_low, Overflow::Wrapping))?;
            let high =
                rewriter.create_op::<Add, _>(span, (*a_high, *b_high, Overflow::Wrapping))?;
            let low = low.borrow().result().as_value_ref();
            let high = high.borrow().result().as_value_ref();
            rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
            Ok(true)
        },
    )
}

/// A `test.sink` of a type [wide_converter] changes becomes a `test.sink` of the values its
/// operands became, flattened; `seen` gets, per call, the ranges of each group.
fn flatten_sink(context: &Rc<Context>, converter: &TypeConverter, seen: Seen<Vec<Ranges>>) -> Rule {
    Rule::new::<Sink>(
        context,
        "flatten-sink",
        Some(converter.clone()),
        [registered::<Sink>(context)],
        move |op, operands, rewriter| {
            if has_no_wide_types(&op.borrow()) {
                return Ok(false);
            }
            seen.borrow_mut().push(
                (0..operands.num_groups())
                    .map(|group| operands.group(group).map(values).collect())
                    .collect(),
            );
            let mut flat = operands.group_flattened(0);
            flat.extend(operands.group_flattened(1));
            let Some((first, rest)) = flat.split_first() else {
                return Ok(false);
            };
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            rewriter.create_op::<Sink, _>(span, (*first, rest.to_vec()))?;
            rewriter.replace_op_with_multiple(op, &[])?;
            Ok(true)
        },
    )
}

/// A `test.store` of the marker stores nothing and is erased; `seen` gets the ranges of each call.
fn drop_store(context: &Rc<Context>, converter: &TypeConverter, seen: Seen<Ranges>) -> Rule {
    Rule::new::<Store>(
        context,
        "drop-store",
        Some(converter.clone()),
        [],
        move |op, operands, rewriter| {
            if has_no_wide_types(&op.borrow()) {
                return Ok(false);
            }
            seen.borrow_mut().push(ranges_of(&operands));
            rewriter.erase_op(op)?;
            Ok(true)
        },
    )
}

/// Parses `body` as the body of `@f` with `signature`, in a context with the test dialect.
fn parse_function(signature: &str, body: &str) -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let function = parse::<Function>(
        ParserConfig::new(context.clone()),
        "conversion",
        &function_text(signature, body),
    )
    .into_result()
    .unwrap_or_else(|err| panic!("{}", describe(&err)));
    (context, function.as_operation_ref())
}

fn function_text(signature: &str, body: &str) -> String {
    format!("builtin.function public extern(\"C\") @f{signature} {{\n{body}\n}};")
}

fn convert(
    function: OperationRef,
    target: ConversionTarget,
    patterns: impl IntoIterator<Item = Rule>,
) -> Result<ConversionResult, Report> {
    let mut set = ConversionPatternSet::new(target.context());
    for pattern in patterns {
        set.push(pattern);
    }
    apply_full_conversion(function, target, set, ConversionConfig::default())
}

/// The values defined in `function`'s entry block, by the order they are defined: the block
/// arguments, then each operation's results.
fn defined_values(function: OperationRef) -> Vec<ValueRef> {
    let function = function.try_downcast_op::<Function>().unwrap();
    let entry = function.borrow().entry_block();
    let entry = entry.borrow();
    let mut values = entry.arguments().iter().map(|arg| *arg as ValueRef).collect::<Vec<_>>();
    for op in entry.body() {
        values.extend(op.results().iter().map(|result| result.borrow().as_value_ref()));
    }
    values
}

fn types(values: &[ValueRef]) -> Vec<Type> {
    values.iter().map(|value| value.borrow().ty()).collect()
}

/// Each `test.add` over `i128` receives two ranges of two `i64`s and replaces its result by a
/// range of two. A user converted later reads through the cast the replacement left, so in a
/// chain of three adds, from split constants into a converted `test.sink`, no cast remains.
#[test]
fn split_operations_take_and_give_ranges_and_no_cast_remains() {
    let (context, function) = parse_function(
        "()",
        "    %0 = test.constant 1 : i128;
    %1 = test.constant 2 : i128;
    %2 = test.add %0, %1;
    %3 = test.add %2, %1;
    %4 = test.add %3, %2;
    test.sink %4 : (i128);
    builtin.ret;",
    );
    let converter = wide_converter(&context);
    let adds = Rc::new(RefCell::new(vec![]));
    let sinks = Rc::new(RefCell::new(vec![]));

    convert(
        function,
        wide_target(&context),
        [
            split_constant(&context, &converter),
            split_add(&context, &converter, adds.clone()),
            flatten_sink(&context, &converter, sinks.clone()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let i64 = context.i64();
    let adds = adds.borrow();
    assert_eq!(adds.len(), 3);
    for ranges in adds.iter() {
        assert_eq!(ranges.len(), 2);
        for range in ranges {
            assert_eq!(types(range), [i64, i64]);
        }
    }
    // The result of an add, read by a later add, is read through the cast its replacement left:
    // the later add is given the halves the earlier one was replaced by.
    let is_add = |value: &ValueRef| value.borrow().get_defining_op().unwrap().borrow().is::<Add>();
    for (add, operand) in [(1, 0), (2, 0), (2, 1)] {
        assert!(adds[add][operand].iter().all(is_add), "add {add}, operand {operand}");
    }
    assert_eq!(sinks.borrow().len(), 1);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %5 = test.constant 1 : i64;
    %6 = test.constant 0 : i64;
    %8 = test.constant 2 : i64;
    %9 = test.constant 0 : i64;
    %11 = test.add %5, %8;
    %12 = test.add %6, %9;
    %14 = test.add %11, %8;
    %15 = test.add %12, %9;
    %17 = test.add %14, %11;
    %18 = test.add %15, %12;
    test.sink %17, %18 : (i64, i64);
    builtin.ret;"
        )
    );
}

/// An operand whose type is legal is passed as the same value; one read through a cast is the
/// cast's inputs; the ranges of `test.sink`'s two operand groups are told apart.
#[test]
fn a_legal_operand_is_itself_and_operand_groups_are_told_apart() {
    let (context, function) = parse_function(
        "(%0: i32)",
        "    %1 = test.constant 7 : i128;
    test.sink %0, %1, %0 : (i32, i128, i32);
    builtin.ret;",
    );
    let argument = defined_values(function)[0];
    let converter = wide_converter(&context);
    let sinks = Rc::new(RefCell::new(vec![]));

    convert(
        function,
        wide_target(&context),
        [
            split_constant(&context, &converter),
            flatten_sink(&context, &converter, sinks.clone()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let sinks = sinks.borrow();
    let [groups] = sinks.as_slice() else {
        panic!("expected one call, got {}", sinks.len());
    };
    assert_eq!(groups.len(), 2, "two groups");
    assert_eq!(groups[0], [vec![argument]], "the first group: the legal operand, as itself");
    let [halves, again] = groups[1].as_slice() else {
        panic!("expected two ranges in the second group: {groups:?}");
    };
    assert_eq!(types(halves), [context.i64(), context.i64()]);
    assert_eq!(*again, [argument]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "(%0: i32)",
            "    %2 = test.constant 7 : i64;
    %3 = test.constant 0 : i64;
    test.sink %0, %2, %3, %0 : (i32, i64, i64, i32);
    builtin.ret;"
        )
    );
}

/// An operand whose type is dropped is an empty range, and no cast is made for it.
#[test]
fn a_dropped_operand_is_an_empty_range() {
    let (context, function) = parse_function(
        "(%0: ptr<array<i8; 0>>, %1: array<i8; 0>)",
        "    test.store %0, %1 : (ptr<array<i8; 0>>, array<i8; 0>);
    builtin.ret;",
    );
    let arguments = defined_values(function);
    let converter = wide_converter(&context);
    let stores = Rc::new(RefCell::new(vec![]));

    convert(
        function,
        wide_target(&context),
        [drop_store(&context, &converter, stores.clone())],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(*stores.borrow(), [vec![vec![arguments[0]], vec![]]]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text("(%0: ptr<array<i8; 0>>, %1: array<i8; 0>)", "    builtin.ret;")
    );
}

/// A replacement of the wrong number of ranges, or a range whose count or types are not what the
/// converter says the result becomes, is an error raised before anything changed: the IR prints
/// as it did. So is a range that is not one value of the result's type, without a converter.
#[test]
fn a_replacement_range_of_the_wrong_shape_is_an_error() {
    type Replace = fn(&[ValueRef], &[ValueRef], ValueRef) -> Vec<Vec<ValueRef>>;
    let cases: [(&str, bool, Replace, &str); 5] = [
        (
            "no range",
            true,
            |_, _, _| vec![],
            "has 1 results, but it is replaced by 0 ranges",
        ),
        (
            "two ranges",
            true,
            |a, b, _| vec![a.to_vec(), b.to_vec()],
            "replaced by 2 ranges",
        ),
        (
            "one value",
            true,
            |a, _, _| vec![vec![a[0]]],
            "result 0 of 'test.add' has type 'i128', which the type converter converts to (i64, \
             i64), but it is replaced by values of types (i64)",
        ),
        (
            "other types",
            true,
            |a, _, original| vec![vec![a[0], original]],
            "converts to (i64, i64), but it is replaced by values of types (i64, i128)",
        ),
        (
            "no converter",
            false,
            |_, _, _| vec![vec![]],
            "result 0 of 'test.add' has type 'i128', but it is replaced by values of types (), \
             and the pattern has no type converter",
        ),
    ];
    for (name, with_converter, replace, message) in cases {
        let signature = "(%0: i128, %1: i128)";
        let body = "    %2 = test.add %0, %1;
    test.sink %2 : (i128);
    builtin.ret;";
        let (context, function) = parse_function(signature, body);
        let before = function.borrow().to_string();
        let original = defined_values(function)[0];
        let converter = with_converter.then(|| wide_converter(&context));
        let pattern = Rule::new::<Add>(
            &context,
            "bad-replacement",
            converter,
            [],
            move |op, operands, rewriter| {
                let ranges = replace(operands.get(0), operands.get(1), original);
                let ranges = ranges.iter().map(Vec::as_slice).collect::<SmallVec<[_; 2]>>();
                rewriter.replace_op_with_multiple(op, &ranges)?;
                Ok(true)
            },
        );

        let err = convert(function, wide_target(&context), [pattern]).unwrap_err();

        let err = format!("{err}");
        if message.starts_with("result ") {
            assert_eq!(err, message, "{name}");
        } else {
            assert!(err.contains(message), "{name}: {err}");
        }
        assert_str_eq!(function.borrow().to_string(), before, "{name}");
    }
}

/// A `test.load` of the marker loads nothing: its result is replaced by no values.
fn drop_load(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    Rule::new::<Load>(
        context,
        "drop-load",
        Some(converter.clone()),
        [],
        move |op, _operands, rewriter| {
            if has_no_wide_types(&op.borrow()) {
                return Ok(false);
            }
            rewriter.replace_op_with_multiple(op, &[&[]])?;
            Ok(true)
        },
    )
}

/// What a materialization callback was given: the types of its inputs and the types it was asked
/// for.
type Offer = (Vec<Type>, Vec<Type>);

/// Adds to `converter` a source callback that joins two `i64` halves into an `i128`, recording
/// what it is offered in `offers`.
fn join_halves(converter: &mut TypeConverter, offers: Seen<Offer>) {
    converter.add_source_materialization(move |rewriter, inputs, ty, span| {
        offers.borrow_mut().push((types(inputs), vec![ty]));
        let [low, high] = inputs else {
            return Ok(None);
        };
        if ty != rewriter.context().i128() {
            return Ok(None);
        }
        let join = rewriter.create_op::<Join, _>(span, (*low, *high))?;
        Ok(Some(join.borrow().result().as_value_ref()))
    });
}

/// Adds to `converter` a target callback that splits an `i128` into its low and high halves,
/// recording what it is offered in `offers`.
fn split_halves(converter: &mut TypeConverter, offers: Seen<Offer>) {
    converter.add_target_materialization(move |rewriter, input, types, span| {
        offers.borrow_mut().push((vec![input.borrow().ty()], types.to_vec()));
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

/// A user no pattern converts (`builtin.ret`, legal) reads the many-to-one cast a split constant
/// left, and the source callback realizes it as a `test.join`.
#[test]
fn a_legal_user_reads_a_join_built_by_the_source_callback() {
    let (context, function) = parse_function(
        "() -> i128",
        "    %0 = test.constant 5 : i128;
    builtin.ret %0 : (i128);",
    );
    let mut converter = wide_converter(&context);
    let offers = Rc::new(RefCell::new(vec![]));
    join_halves(&mut converter, offers.clone());

    let result = convert(function, wide_target(&context), [split_constant(&context, &converter)])
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert!(result.changed());
    let (i64, i128) = (context.i64(), context.i128());
    assert_eq!(*offers.borrow(), [(vec![i64, i64], vec![i128])]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "() -> i128",
            "    %1 = test.constant 5 : i64;
    %2 = test.constant 0 : i64;
    %4 = test.join %1, %2;
    builtin.ret %4 : (i128);"
        )
    );
}

/// A legal producer (a block argument) feeding a converted user: the one-to-two casts made for
/// the operands are realized by a target callback as `test.low` and `test.high`, and the result's
/// cast for the legal `builtin.ret` by a source callback as `test.join`. Callbacks are offered a
/// cast most recently added first: the one that declines is asked before the one that answers.
#[test]
fn a_legal_producer_is_split_into_halves_by_the_target_callback() {
    let (context, function) = parse_function(
        "(%0: i128, %1: i128) -> i128",
        "    %2 = test.add %0, %1;
    builtin.ret %2 : (i128);",
    );
    let mut converter = wide_converter(&context);
    let joins = Rc::new(RefCell::new(vec![]));
    let splits = Rc::new(RefCell::new(vec![]));
    let declined = Rc::new(RefCell::new(vec![]));
    join_halves(&mut converter, joins.clone());
    split_halves(&mut converter, splits.clone());
    {
        let declined = declined.clone();
        let splits = splits.clone();
        converter.add_target_materialization(move |_, input, _, _| {
            declined.borrow_mut().push((input, splits.borrow().len()));
            Ok(None)
        });
    }
    let adds = Rc::new(RefCell::new(vec![]));

    convert(function, wide_target(&context), [split_add(&context, &converter, adds)])
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let arguments = defined_values(function);
    let (i64, i128) = (context.i64(), context.i128());
    // Asked first for each cast: before the splitting callback had seen it.
    assert_eq!(*declined.borrow(), [(arguments[0], 0), (arguments[1], 1)]);
    assert_eq!(*splits.borrow(), [(vec![i128], vec![i64, i64]), (vec![i128], vec![i64, i64])]);
    assert_eq!(*joins.borrow(), [(vec![i64, i64], vec![i128])]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "(%0: i128, %1: i128) -> i128",
            "    %10 = test.low %0;
    %11 = test.high %0;
    %12 = test.low %1;
    %13 = test.high %1;
    %7 = test.add %10, %12;
    %8 = test.add %11, %13;
    %14 = test.join %7, %8;
    builtin.ret %14 : (i128);"
        )
    );
}

/// Without a callback, the many-to-one cast a split constant left for `builtin.ret` is an error
/// that names the cast's types and its user.
#[test]
fn a_cast_without_a_callback_is_an_error() {
    let (context, function) = parse_function(
        "() -> i128",
        "    %0 = test.constant 5 : i128;
    builtin.ret %0 : (i128);",
    );
    let converter = wide_converter(&context);

    let err = convert(function, wide_target(&context), [split_constant(&context, &converter)])
        .unwrap_err();

    assert_eq!(
        format!("{err}"),
        "failed to legalize operation 'builtin.unrealized_conversion_cast': no source \
         materialization of the type converter realizes the cast of (i64, i64) to (i128), which \
         is used by 'builtin.ret'"
    );
}

/// A result whose type is dropped, and that still has a user, gets a cast with no inputs, which
/// is offered to the source callback.
#[test]
fn a_dropped_result_with_a_user_is_offered_to_the_source_callback() {
    let (context, function) = parse_function(
        "(%0: ptr<array<i8; 0>>, %1: array<i8; 0>) -> array<i8; 0>",
        "    %2 = test.load %0;
    builtin.ret %2 : (array<i8; 0>);",
    );
    let marker_value = defined_values(function)[1];
    let mut converter = wide_converter(&context);
    let offers: Seen<Offer> = Rc::new(RefCell::new(vec![]));
    {
        let offers = offers.clone();
        converter.add_source_materialization(move |_, inputs, ty, _| {
            offers.borrow_mut().push((types(inputs), vec![ty]));
            Ok(Some(marker_value))
        });
    }

    convert(function, wide_target(&context), [drop_load(&context, &converter)])
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(*offers.borrow(), [(vec![], vec![marker(&context)])]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "(%0: ptr<array<i8; 0>>, %1: array<i8; 0>) -> array<i8; 0>",
            "    builtin.ret %1 : (array<i8; 0>);"
        )
    );
}

/// With reconciliation off, nothing is folded or realized: the cast the driver inserted stays,
/// even in a target that calls it illegal, and no callback is asked.
#[test]
fn without_reconciliation_the_casts_stay_and_no_callback_is_asked() {
    let (context, function) = parse_function(
        "() -> i128",
        "    %0 = test.constant 5 : i128;
    builtin.ret %0 : (i128);",
    );
    let mut converter = wide_converter(&context);
    let offers = Rc::new(RefCell::new(vec![]));
    join_halves(&mut converter, offers.clone());
    let mut target = wide_target(&context);
    target.add_illegal_op::<UnrealizedConversionCast>();
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(split_constant(&context, &converter));
    let mut config = ConversionConfig::default();
    config.with_reconcile_unrealized_casts(false);

    apply_full_conversion(function, target, set, config)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert!(offers.borrow().is_empty());
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "() -> i128",
            "    %1 = test.constant 5 : i64;
    %2 = test.constant 0 : i64;
    %3 = builtin.unrealized_conversion_cast %1, %2 : (i64, i64) to i128;
    builtin.ret %3 : (i128);"
        )
    );
}

/// A cast in the IR that the driver did not insert is not offered to any callback; it stays
/// because the target calls it legal.
#[test]
fn a_cast_the_driver_did_not_insert_is_not_offered() {
    let (context, function) = parse_function(
        "(%0: i32) -> (i128, i64)",
        "    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %2 = test.constant 5 : i128;
    builtin.ret %2, %1 : (i128, i64);",
    );
    let mut converter = wide_converter(&context);
    let joins = Rc::new(RefCell::new(vec![]));
    let splits = Rc::new(RefCell::new(vec![]));
    join_halves(&mut converter, joins.clone());
    split_halves(&mut converter, splits.clone());

    convert(function, wide_target(&context), [split_constant(&context, &converter)])
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let (i64, i128) = (context.i64(), context.i128());
    assert_eq!(*joins.borrow(), [(vec![i64, i64], vec![i128])]);
    assert!(splits.borrow().is_empty());
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "(%0: i32) -> (i128, i64)",
            "    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %3 = test.constant 5 : i64;
    %4 = test.constant 0 : i64;
    %6 = test.join %3, %4;
    builtin.ret %6, %1 : (i128, i64);"
        )
    );
}

/// The operands of the `builtin.ret` that ends `function`'s entry block.
fn returned(function: OperationRef) -> Vec<ValueRef> {
    let function = function.try_downcast_op::<Function>().unwrap();
    let entry = function.borrow().entry_block();
    let ret = entry.borrow().terminator().unwrap();
    ret.borrow()
        .operands()
        .iter()
        .map(|operand| operand.borrow().as_value_ref())
        .collect()
}

/// Adds to `converter` a source callback that joins two `i64` halves into an `i128`; the values it
/// builds are recorded in the list returned.
fn tagging_join(converter: &mut TypeConverter) -> Seen<ValueRef> {
    let built: Seen<ValueRef> = Rc::default();
    let tags = built.clone();
    converter.add_source_materialization(move |rewriter, inputs, _, span| {
        let [low, high] = inputs else {
            return Ok(None);
        };
        let join = rewriter.create_op::<Join, _>(span, (*low, *high))?;
        let value = join.borrow().result().as_value_ref();
        tags.borrow_mut().push(value);
        Ok(Some(value))
    });
    built
}

/// Two patterns whose converters have different source callbacks: each cast that remains is
/// realized by the callback of the converter of the pattern that caused it.
#[test]
fn each_cast_is_realized_by_the_converter_of_the_pattern_that_caused_it() {
    let (context, function) = parse_function(
        "() -> (i128, i128)",
        "    %0 = test.constant 1 : i128;
    %1 = test.constant 2 : i128;
    %2 = test.add %0, %1;
    builtin.ret %0, %2 : (i128, i128);",
    );
    let mut for_constants = wide_converter(&context);
    let mut for_adds = wide_converter(&context);
    let by_constants = tagging_join(&mut for_constants);
    let by_adds = tagging_join(&mut for_adds);

    convert(
        function,
        wide_target(&context),
        [
            split_constant(&context, &for_constants),
            split_add(&context, &for_adds, Rc::default()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let returned = returned(function);
    assert_eq!(*by_constants.borrow(), [returned[0]], "the constant's cast");
    assert_eq!(*by_adds.borrow(), [returned[1]], "the add's cast");
}

/// A source callback that answers a value of another type than the cast's is an error naming the
/// cast.
#[test]
fn a_source_callback_answering_the_wrong_type_is_an_error() {
    let (context, function) = parse_function(
        "() -> i128",
        "    %0 = test.constant 5 : i128;
    builtin.ret %0 : (i128);",
    );
    let mut converter = wide_converter(&context);
    converter.add_source_materialization(|_, inputs, _, _| Ok(inputs.first().copied()));

    let err = convert(function, wide_target(&context), [split_constant(&context, &converter)])
        .unwrap_err();

    assert_eq!(
        format!("{err}"),
        "failed to legalize operation 'builtin.unrealized_conversion_cast': the source \
         materialization of the cast of (i64, i64) to (i128) gave values of types (i64)"
    );
}

/// A target callback that answers the wrong number of values is an error naming the cast.
#[test]
fn a_target_callback_answering_the_wrong_count_is_an_error() {
    let (context, function) = parse_function(
        "(%0: i128, %1: i128) -> i128",
        "    %2 = test.add %0, %1;
    builtin.ret %2 : (i128);",
    );
    let mut converter = wide_converter(&context);
    converter.add_target_materialization(|rewriter, input, _, span| {
        let low = rewriter.create_op::<Low, _>(span, (input,))?;
        Ok(Some(smallvec::smallvec![low.borrow().result().as_value_ref()]))
    });

    let err = convert(
        function,
        wide_target(&context),
        [split_add(&context, &converter, Rc::default())],
    )
    .unwrap_err();

    assert_eq!(
        format!("{err}"),
        "failed to legalize operation 'builtin.unrealized_conversion_cast': the target \
         materialization of the cast of (i128) to (i64, i64) gave values of types (i64)"
    );
}

/// A callback that fails aborts the conversion with its error.
#[test]
fn a_failing_callback_aborts_the_conversion_with_its_error() {
    let (context, function) = parse_function(
        "() -> i128",
        "    %0 = test.constant 5 : i128;
    builtin.ret %0 : (i128);",
    );
    let mut converter = wide_converter(&context);
    converter.add_source_materialization(|_, _, _, _| Err(Report::msg("the halves do not join")));

    let err = convert(function, wide_target(&context), [split_constant(&context, &converter)])
        .unwrap_err();

    assert_eq!(format!("{err}"), "the halves do not join");
}

/// After a replacement, an operation the pattern creates is built where the insertion point was:
/// where the replaced operation was, when the insertion point was at it, and elsewhere where the
/// pattern put it.
#[test]
fn after_a_replacement_the_insertion_point_is_where_it_was() {
    let cases = [
        (
            true,
            "    %0 = test.constant 1 : i32;
    %2 = test.constant 5 : i64;
    %3 = test.constant 0 : i64;
    %6 = test.join %2, %3;
    %5 = test.constant 9 : i32;
    builtin.ret %6 : (i128);",
        ),
        (
            false,
            "    %2 = test.constant 5 : i64;
    %3 = test.constant 0 : i64;
    %5 = test.constant 9 : i32;
    %0 = test.constant 1 : i32;
    %6 = test.join %2, %3;
    builtin.ret %6 : (i128);",
        ),
    ];
    for (at_op, expected) in cases {
        let (context, function) = parse_function(
            "() -> i128",
            "    %0 = test.constant 1 : i32;
    %1 = test.constant 5 : i128;
    builtin.ret %1 : (i128);",
        );
        let first = defined_values(function)[0].borrow().get_defining_op().unwrap();
        let mut converter = wide_converter(&context);
        join_halves(&mut converter, Rc::default());
        let i128 = context.i128();
        let pattern = Rule::new::<Constant>(
            &context,
            "split-then-build",
            Some(converter.clone()),
            [registered::<Constant>(&context)],
            move |op, _operands, rewriter| {
                if op.borrow().results()[0].borrow().ty() != i128 {
                    return Ok(false);
                }
                if !at_op {
                    rewriter.set_insertion_point_before(first);
                }
                let span = op.borrow().span();
                let low = rewriter.create_op::<Constant, _>(span, (Int::from(5i64),))?;
                let high = rewriter.create_op::<Constant, _>(span, (Int::from(0i64),))?;
                let low = low.borrow().result().as_value_ref();
                let high = high.borrow().result().as_value_ref();
                rewriter.replace_op_with_multiple(op, &[&[low, high]])?;
                rewriter.create_op::<Constant, _>(span, (Int::from(9i32),))?;
                Ok(true)
            },
        );

        convert(function, wide_target(&context), [pattern])
            .unwrap_or_else(|err| panic!("{}", describe(&err)));

        assert_str_eq!(
            function.borrow().to_string(),
            function_text("() -> i128", expected),
            "at the replaced operation: {at_op}"
        );
    }
}

/// A target whose rule for the test operations is that the converter finds them legal
/// ([TypeConverter::is_legal_op]) converts exactly the operations that touch an `i128`; those over
/// `i32` are left as they were.
#[test]
fn a_target_ruled_by_the_converter_converts_exactly_the_operations_that_touch_an_i128() {
    let (context, function) = parse_function(
        "(%0: i32, %1: i32)",
        "    %2 = test.constant 1 : i128;
    %3 = test.constant 2 : i128;
    %4 = test.add %2, %3;
    %5 = test.add %0, %1;
    %6 = test.constant 7 : i32;
    test.sink %4 : (i128);
    test.sink %5, %6 : (i32, i32);
    builtin.ret;",
    );
    let converter = wide_converter(&context);
    let legal = {
        let converter = converter.clone();
        move |op: &Operation| DynamicLegalityResult::legal_if(converter.is_legal_op(op))
    };
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Constant, _>(legal.clone())
        .add_dynamically_legal_op::<Add, _>(legal.clone())
        .add_dynamically_legal_op::<Sink, _>(legal);
    let adds = Rc::new(RefCell::new(vec![]));
    let sinks = Rc::new(RefCell::new(vec![]));

    let result = convert(
        function,
        target,
        [
            split_constant(&context, &converter),
            split_add(&context, &converter, adds.clone()),
            flatten_sink(&context, &converter, sinks.clone()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(result.converted_ops(), 4, "two constants, an add and a sink");
    assert_eq!(adds.borrow().len(), 1);
    assert_eq!(sinks.borrow().len(), 1);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "(%0: i32, %1: i32)",
            "    %7 = test.constant 1 : i64;
    %8 = test.constant 0 : i64;
    %10 = test.constant 2 : i64;
    %11 = test.constant 0 : i64;
    %13 = test.add %7, %10;
    %14 = test.add %8, %11;
    %5 = test.add %0, %1;
    %6 = test.constant 7 : i32;
    test.sink %13, %14 : (i64, i64);
    test.sink %5, %6 : (i32, i32);
    builtin.ret;"
        )
    );
}
