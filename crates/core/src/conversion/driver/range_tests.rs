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
        test::{Add, Constant, Sink, Store, TestDialect},
    },
    parse::{ParserConfig, parse},
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind},
    testing::text::describe,
};

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
    context.get_or_register_dialect::<TestDialect>().expect_registered_name::<T>()
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
        .add_dynamically_legal_op::<crate::dialects::test::Load, _>(legal)
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
    %21 = test.constant 2 : i64;
    %22 = test.constant 0 : i64;
    %10 = test.add %5, %21;
    %11 = test.add %6, %22;
    %15 = test.add %10, %21;
    %16 = test.add %11, %22;
    %18 = test.add %15, %10;
    %19 = test.add %16, %11;
    test.sink %18, %19 : (i64, i64);
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
            "converts to (i64, i64), but it is replaced by values of types (i64)",
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
            "and the pattern has no type converter",
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
        assert!(err.contains(message), "{name}: {err}");
        assert_str_eq!(function.borrow().to_string(), before, "{name}");
    }
}
