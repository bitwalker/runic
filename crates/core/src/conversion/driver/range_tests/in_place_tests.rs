//! A pattern changes an operation in place through the rewriter: the change is a mutation, the
//! operation is legalized again, and a result whose type changed is related to its old type by a
//! cast for the users that still expect it.

use super::*;
use crate::{
    ElementRewrite, EntityWithId, OpPrinter,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    effects::MemoryEffectOpInterface,
    replace_elements_in,
    traits::{AnyInteger, BranchOpInterface, InferTypeOpInterface, Terminator},
    types::TypeMapper,
};

/// The value of the `test.constant` `op`, as an unsigned number.
fn constant_value(op: OperationRef) -> u128 {
    op.try_downcast_op::<Constant>().unwrap().borrow().get_value().unsigned()
}

/// A pattern for a `test.constant` of value `from` that sets its value to `to` in place.
fn set_constant(context: &Rc<Context>, name: &'static str, from: u128, to: i32) -> Rule {
    Rule::new::<Constant>(context, name, None, [], move |op, _operands, rewriter| {
        if constant_value(op) != from {
            return Ok(false);
        }
        rewriter.modify_op_in_place(op, |op| {
            op.downcast_mut::<Constant>().unwrap().set_value(Int::from(to));
        })?;
        Ok(true)
    })
}

/// A pattern that changes only a property, through `modify_op_in_place`, succeeds; the operation
/// is legalized again, and a second pattern rooted on it then runs on it.
#[test]
fn a_pattern_that_changes_only_a_property_succeeds_and_the_operation_is_legalized_again() {
    let (context, function) = parse_function(
        "() -> i32",
        "    %0 = test.constant 1 : i32;
    builtin.ret %0 : (i32);",
    );
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Constant, _>(|op: &Operation| {
            DynamicLegalityResult::legal_if(constant_value(op.as_operation_ref()) == 3)
        });

    let result = convert(
        function,
        target,
        [
            set_constant(&context, "one-to-two", 1, 2),
            set_constant(&context, "two-to-three", 2, 3),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(result.converted_ops(), 2);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "() -> i32",
            "    %0 = test.constant 3 : i32;
    builtin.ret %0 : (i32);"
        )
    );
}

/// `i64` becomes `i32`.
fn narrow_converter(context: &Context) -> TypeConverter {
    let (i32, i64) = (context.i32(), context.i64());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == i64).then_some(TypeConversion::One(i32)));
    converter
}

/// Whether `op` has no operand or result of type `i64`.
fn has_no_i64(op: &Operation) -> bool {
    let i64 = op.context().i64();
    !op.operands().iter().any(|operand| operand.borrow().ty() == i64)
        && !op.results().iter().any(|result| result.borrow().ty() == i64)
}

/// A `test.constant` of type `i64` is retyped in place to `i32`, its value truncated.
fn narrow_constant_in_place(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    Rule::new::<Constant>(
        context,
        "narrow-constant-in-place",
        Some(converter.clone()),
        [],
        move |op, _operands, rewriter| {
            if has_no_i64(&op.borrow()) {
                return Ok(false);
            }
            let context = op.borrow().context_rc();
            let mapper = TypeMapper::new(&context, |context, ty| {
                if ty == context.i64() {
                    context.i32()
                } else {
                    ty
                }
            });
            let mut rewrite = ElementRewrite::types(mapper).with_int_constants(|value, _, new| {
                Ok(value.trunc(Int::width_for_type(new).expect("an integer type")))
            });
            rewriter.modify_op_in_place(op, |op| replace_elements_in(op, &mut rewrite))??;
            Ok(true)
        },
    )
}

/// A `test.add` of `i64` becomes a `test.add` of the values its operands became.
fn narrow_add(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    Rule::new::<Add>(
        context,
        "narrow-add",
        Some(converter.clone()),
        [registered::<Add>(context)],
        move |op, operands, rewriter| {
            if has_no_i64(&op.borrow()) {
                return Ok(false);
            }
            let Some([lhs, rhs]) = operands.all_one().map(|values| [values[0], values[1]]) else {
                return Ok(false);
            };
            let span = op.borrow().span();
            let add = rewriter.create_op::<Add, _>(span, (lhs, rhs, Overflow::Wrapping))?;
            let result = add.borrow().result().as_value_ref();
            rewriter.replace_op(op, &[result])?;
            Ok(true)
        },
    )
}

/// A `test.sink` of `i64` becomes a `test.sink` of the values its operands became.
fn narrow_sink(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    Rule::new::<Sink>(
        context,
        "narrow-sink",
        Some(converter.clone()),
        [registered::<Sink>(context)],
        move |op, operands, rewriter| {
            if has_no_i64(&op.borrow()) {
                return Ok(false);
            }
            let flat = operands.iter().flatten().copied().collect::<Vec<_>>();
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            rewriter.create_op::<Sink, _>(span, (flat[0], flat[1..].to_vec()))?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    )
}

/// The constants, adds and (when `sink` is set) sinks of `i64` are illegal.
fn narrow_target(context: &Rc<Context>, sink: bool) -> ConversionTarget {
    let legal = |op: &Operation| DynamicLegalityResult::legal_if(has_no_i64(op));
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Constant, _>(legal)
        .add_dynamically_legal_op::<Add, _>(legal);
    if sink {
        target.add_dynamically_legal_op::<Sink, _>(legal);
    }
    target
}

const NARROW_BODY: &str = "    %0 = test.constant 4294967297 : i64;
    %1 = test.add %0, %0;
    test.sink %0 : (i64);
    builtin.ret;";

/// A constant retyped in place from `i64` to `i32` (its value truncated) leaves the user no pattern
/// converts reading a cast of the new value back to `i64`, placed after the constant, and the user
/// that is converted reading the new value. With reconciliation off the cast stays.
#[test]
fn a_result_retyped_in_place_is_cast_back_for_the_users_that_expect_its_old_type() {
    let (context, function) = parse_function("()", NARROW_BODY);
    let converter = narrow_converter(&context);
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(narrow_constant_in_place(&context, &converter));
    set.push(narrow_add(&context, &converter));
    let mut config = ConversionConfig::default();
    config.with_reconcile_unrealized_casts(false);

    apply_full_conversion(function, narrow_target(&context, false), set, config)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %0 = test.constant 1 : i32;
    %2 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %3 = test.add %0, %0;
    test.sink %2 : (i64);
    builtin.ret;"
        )
    );
}

/// When every user of the retyped result is converted, no cast remains.
#[test]
fn a_result_retyped_in_place_leaves_no_cast_when_every_user_is_converted() {
    let (context, function) = parse_function("()", NARROW_BODY);
    let converter = narrow_converter(&context);

    convert(
        function,
        narrow_target(&context, true),
        [
            narrow_constant_in_place(&context, &converter),
            narrow_add(&context, &converter),
            narrow_sink(&context, &converter),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %0 = test.constant 1 : i32;
    %3 = test.add %0, %0;
    test.sink %0 : (i32);
    builtin.ret;"
        )
    );
}

/// A pattern for the `i64` constant whose change in place is `change`.
fn retype_constant(
    context: &Rc<Context>,
    converter: Option<TypeConverter>,
    change: fn(&mut Operation),
) -> Rule {
    Rule::new::<Constant>(context, "retype-constant", converter, [], move |op, _, rewriter| {
        rewriter.modify_op_in_place(op, change)?;
        Ok(true)
    })
}

/// Sets the type of the first result of `op` to `i16`.
fn result_to_i16(op: &mut Operation) {
    let i16 = op.context().i16();
    let mut result = op.results()[0];
    result.borrow_mut().set_type(i16);
}

/// A result retyped in place to a type the converter does not answer for its old type is an
/// error; so is any retyped result without a converter.
#[test]
fn a_result_retyped_to_a_type_the_converter_does_not_answer_is_an_error() {
    let cases = [
        (
            true,
            "result 0 of 'test.constant' changed type in place, from 'i64' to 'i16', but the type \
             converter converts 'i64' to (i32)",
        ),
        (
            false,
            "result 0 of 'test.constant' changed type in place, from 'i64' to 'i16', and the \
             pattern has no type converter",
        ),
    ];
    for (with_converter, message) in cases {
        let (context, function) = parse_function("()", NARROW_BODY);
        let converter = with_converter.then(|| narrow_converter(&context));
        let pattern = retype_constant(&context, converter, result_to_i16);

        let err = convert(function, narrow_target(&context, false), [pattern]).unwrap_err();

        let err = format!("{err}");
        assert!(err.contains(message), "{err}");
    }
}

/// Changing the type of a block argument inside `modify_op_in_place` is an error.
#[test]
fn a_block_argument_retyped_in_place_is_an_error() {
    let (context, function) = parse_function("(%0: i64)", "    builtin.ret;");
    let converter = narrow_converter(&context);
    let pattern =
        Rule::new::<Function>(&context, "retype-entry", Some(converter), [], |op, _, rewriter| {
            rewriter.modify_op_in_place(op, |op| {
                let i32 = op.context().i32();
                let entry = op.downcast_ref::<Function>().unwrap().entry_block();
                let mut argument = entry.borrow().arguments()[0];
                argument.borrow_mut().set_type(i32);
            })?;
            Ok(true)
        });
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Function, _>(|_| DynamicLegalityResult::illegal());

    let err = convert(function, target, [pattern]).unwrap_err();

    let err = format!("{err}");
    assert!(
        err.contains(
            "the arguments of block 0 in the regions of 'builtin.function' changed type in place, \
             from (i64) to (i32); a block's arguments are changed by a signature conversion"
        ),
        "{err}"
    );
}

/// The pattern contract is unchanged: `Ok(true)` without a change is an error, and so is a change
/// in place followed by `Ok(false)`.
#[test]
fn the_pattern_contract_holds_for_changes_in_place() {
    type Body = fn(OperationRef, &mut ConversionPatternRewriter) -> Result<bool, Report>;
    let cases: [(Body, &str); 2] = [
        (
            |_, _| Ok(true),
            "reported success for operation 'test.constant' without mutating IR",
        ),
        (
            |op, rewriter| {
                rewriter.modify_op_in_place(op, |_| ())?;
                Ok(false)
            },
            "mutated operation 'test.constant' but returned no match",
        ),
    ];
    for (body, message) in cases {
        let (context, function) = parse_function("()", NARROW_BODY);
        let pattern =
            Rule::new::<Constant>(&context, "contract", None, [], move |op, _, rewriter| {
                body(op, rewriter)
            });

        let err = convert(function, narrow_target(&context, false), [pattern]).unwrap_err();

        let err = format!("{err}");
        assert!(err.contains(message), "{err}");
    }
}

/// A pattern for the `i64` constant that retypes it to `i32` in place, its value truncated.
fn narrow_constant_no_cast(context: &Rc<Context>) -> Rule {
    narrow_constant_in_place(context, &narrow_converter(context))
}

/// A retyped result without users gets no cast, even with reconciliation off.
#[test]
fn a_result_retyped_in_place_without_users_gets_no_cast() {
    let (context, function) = parse_function(
        "()",
        "    %0 = test.constant 4294967297 : i64;
    builtin.ret;",
    );
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(narrow_constant_no_cast(&context));
    let mut config = ConversionConfig::default();
    config.with_reconcile_unrealized_casts(false);

    apply_full_conversion(function, narrow_target(&context, false), set, config)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %0 = test.constant 1 : i32;
    builtin.ret;"
        )
    );
}

/// Two `i64`s: `%0, %1 = test.pair;`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Pair {
    #[result]
    first: AnyInteger,
    #[result]
    second: AnyInteger,
}

impl InferTypeOpInterface for Pair {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.first_mut().set_type(context.i64());
        self.second_mut().set_type(context.i64());
        Ok(())
    }
}

/// A terminator with a result, an `i64`, and one successor: `%0 = test.value_br ^bb1;`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(Terminator),
    implements(
        BranchOpInterface,
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct ValueBr {
    #[successor]
    target: Successor,
    #[result]
    result: AnyInteger,
}

impl BranchOpInterface for ValueBr {}

impl InferTypeOpInterface for ValueBr {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i64());
        Ok(())
    }
}

/// Sets the type of every `i64` result of `op` to `i32`.
fn results_to_i32(op: &mut Operation) {
    let (i32, i64) = (op.context().i32(), op.context().i64());
    for mut result in op.results().iter().copied() {
        if result.borrow().ty() == i64 {
            result.borrow_mut().set_type(i32);
        }
    }
}

/// A pattern rooted at `Root` that retypes its `i64` results to `i32` in place.
fn narrow_results_in_place<Root: OpRegistration>(context: &Rc<Context>) -> Rule {
    Rule::new::<Root>(
        context,
        "narrow-results-in-place",
        Some(narrow_converter(context)),
        [],
        |op, _, rewriter| {
            if has_no_i64(&op.borrow()) {
                return Ok(false);
            }
            rewriter.modify_op_in_place(op, results_to_i32)?;
            Ok(true)
        },
    )
}

/// An operation is legal when no result is an `i64`.
fn no_i64_result(op: &Operation) -> DynamicLegalityResult {
    let i64 = op.context().i64();
    DynamicLegalityResult::legal_if(!op.results().iter().any(|result| result.borrow().ty() == i64))
}

/// Two results retyped in place, both used, are each cast back for their users; the casts follow
/// the operation in the order of its results.
#[test]
fn two_results_retyped_in_place_are_cast_back_in_order() {
    let (context, function) = parse_function(
        "()",
        "    %0, %1 = test.pair;
    test.sink %0, %1 : (i64, i64);
    builtin.ret;",
    );
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Pair, _>(no_i64_result);
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(narrow_results_in_place::<Pair>(&context));
    let mut config = ConversionConfig::default();
    config.with_reconcile_unrealized_casts(false);

    apply_full_conversion(function, target, set, config)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "()",
            "    %0, %1 = test.pair;
    %2 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %3 = builtin.unrealized_conversion_cast %1 : i32 to i64;
    test.sink %2, %3 : (i64, i64);
    builtin.ret;"
        )
    );
}

/// A result retyped in place on a terminator, with a user, is an error: its cast would follow the
/// terminator. No cast is inserted.
#[test]
fn a_result_retyped_in_place_on_a_terminator_is_an_error() {
    let (context, function) = parse_function(
        "()",
        "    %0 = test.value_br ^bb1;
^bb1:
    test.sink %0 : (i64);
    builtin.ret;",
    );
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<ValueBr, _>(no_i64_result);

    let err =
        convert(function, target, [narrow_results_in_place::<ValueBr>(&context)]).unwrap_err();

    let err = format!("{err}");
    assert!(
        err.contains(
            "result 0 of 'test.value_br' changed type in place, from 'i64' to 'i32', and has \
             users, but 'test.value_br' is a terminator: the cast back to 'i64' cannot be placed \
             after it"
        ),
        "{err}"
    );
    assert!(!function.borrow().to_string().contains("builtin.unrealized_conversion_cast"));
}
