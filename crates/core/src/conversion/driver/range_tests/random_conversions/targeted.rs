//! Operations under a recursively legal operation, two converters that
//! disagree, which converter realizes a cast, value callbacks, and the partial mode's reading of
//! the unknown operation policy.

use std::println;

use super::{ops::*, *};
use crate::{
    Builder, BuilderExt, OpBuilder, RegionKind, RegionKindInterface, SourceSpan,
    conversion::SignatureConversion,
    derive::{EffectOpInterface, operation},
    effects::MemoryEffectOpInterface,
    testing::Test,
    traits::{AnyType, InferTypeOpInterface, NoTerminator, SingleBlock},
};

/// An operation with one region whose operations may use the values around it.
#[derive(EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, NoTerminator),
    implements(RegionKindInterface, MemoryEffectOpInterface)
)]
pub struct Scope {
    #[region]
    body: Region,
}

impl RegionKindInterface for Scope {
    fn kind(&self) -> RegionKind {
        RegionKind::SSA
    }
}

/// An operation with one region and one `i128` result.
#[derive(EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, NoTerminator),
    implements(RegionKindInterface, MemoryEffectOpInterface, InferTypeOpInterface)
)]
pub struct Wrap {
    #[region]
    body: Region,
    #[result]
    result: AnyType,
}

impl RegionKindInterface for Wrap {
    fn kind(&self) -> RegionKind {
        RegionKind::SSA
    }
}

impl InferTypeOpInterface for Wrap {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i128());
        Ok(())
    }
}

fn join_callbacks(converter: &mut TypeConverter) {
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

/// An operation nested under a recursively legal one is not offered to a pattern, even when a
/// value it reads is replaced outside: it is left as it is, reading a realized cast.
#[test]
fn an_operation_under_a_recursively_legal_one_is_not_converted() {
    let mut test = Test::new("f", |_| ([], []));
    let context = test.context_rc();
    {
        let mut builder = test.function_builder();
        let b = builder.builder_mut();
        const S: SourceSpan = SourceSpan::UNKNOWN;
        let outer = b.create::<Constant, _>(S)(Int::from_bits(128, 7)).unwrap();
        let outer = outer.borrow().result().as_value_ref();
        let scope = b.create::<Scope, ()>(S)().unwrap();
        let region = scope.borrow().body().as_region_ref();
        let body = b.create_block(region, None, &[]);
        let after = scope.as_operation_ref();
        b.set_insertion_point_to_end(body);
        b.create::<Sink, (ValueRef, Vec<ValueRef>)>(S)(outer, vec![]).unwrap();
        b.set_insertion_point_after(after);
        b.create::<crate::dialects::builtin::Ret, (Vec<ValueRef>,)>(S)(vec![]).unwrap();
    }
    let function = test.function().as_operation_ref();
    let mut converter = wide_converter(&context);
    join_callbacks(&mut converter);
    let mut target = wide_target(&context);
    target.add_legal_op::<Scope>().mark_op_recursively_legal::<Scope>();
    convert(
        function,
        target,
        [
            split_constant(&context, &converter),
            flatten_sink(&context, &converter, Rc::default()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let output = function.borrow().to_string();
    assert!(output.contains("test.join"), "{output}");
    assert!(output.contains(": (i128);"), "{output}");
}

/// Two converters that give `i128` the same type list with different meanings: one splits into
/// (low, high), the other into (high, low). The driver reads the second pattern's operand through
/// the first pattern's cast, because the types agree: the program computes something else, and
/// nothing reports it.
#[test]
fn two_converters_that_disagree_on_the_meaning_of_a_type_list_miscompile_silently() {
    let (context, function) = parse_function(
        "() -> i64",
        "    %0 = test.constant 5 : i128;
    %1 = test.low %0;
    builtin.ret %1 : (i64);",
    );
    let low_high = wide_converter(&context);
    // The same answer, read the other way round by its patterns.
    let high_low = wide_converter(&context);
    let low_under_high_low = Rule::new::<Low>(
        &context,
        "low-of-high-low",
        Some(high_low.clone()),
        [],
        |op, operands, rewriter| {
            // Under this converter the halves are (high, low).
            let [_high, low] = operands.get(0) else {
                return Ok(false);
            };
            rewriter.replace_op(op, &[*low])?;
            Ok(true)
        },
    );
    let mut target = wide_target(&context);
    target.add_illegal_op::<Low>();

    convert(function, target, [split_constant(&context, &low_high), low_under_high_low])
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let output = function.borrow().to_string();
    println!("{output}");
    // The function returned the low half, 5; it now returns the high half, 0.
    assert!(output.contains("builtin.ret %"), "{output}");
    let returned = output.lines().find(|line| line.contains("builtin.ret")).unwrap();
    let value = returned.trim().trim_start_matches("builtin.ret ").split(' ').next().unwrap();
    let definition = output
        .lines()
        .find(|line| line.trim().starts_with(&format!("{value} =")))
        .unwrap();
    assert!(
        definition.contains("test.constant 0 : i64"),
        "pinned: the high half is returned\n{output}"
    );
}

/// A cast is realized by the callbacks of the converter of the pattern that caused it: the source
/// cast a replacement left by converter A's callbacks, the target cast made for an operand of a
/// pattern under converter B by B's.
#[test]
fn a_cast_is_realized_by_the_converter_that_caused_it() {
    let (context, function) = parse_function(
        "(%0: i64, %1: i64) -> i64",
        "    %2 = test.constant 5 : i128;
    test.sink %2 : (i128);
    %3 = test.join %0, %1;
    %4 = test.low %3;
    builtin.ret %4 : (i64);",
    );
    // A: its callbacks build test.join / test.low / test.high.
    let mut a = wide_converter(&context);
    join_callbacks(&mut a);
    // B: its callbacks build test.m_join / test.m_low / test.m_high.
    let mut b = wide_converter(&context);
    b.add_source_materialization(move |rewriter, inputs, _, span| {
        let [low, high] = inputs else {
            return Ok(None);
        };
        let join = rewriter.create_op::<MJoin, _>(span, (*low, *high))?;
        Ok(Some(join.borrow().result().as_value_ref()))
    });
    b.add_target_materialization(move |rewriter, input, _, span| {
        let low = rewriter.create_op::<MLow, _>(span, (input,))?;
        let high = rewriter.create_op::<MHigh, _>(span, (input,))?;
        let low = low.borrow().result().as_value_ref();
        let high = high.borrow().result().as_value_ref();
        Ok(Some(smallvec::smallvec![low, high]))
    });
    let low_under_b = Rule::new::<Low>(&context, "low", Some(b.clone()), [], |op, operands, rw| {
        let [low, _high] = operands.get(0) else {
            return Ok(false);
        };
        rw.replace_op(op, &[*low])?;
        Ok(true)
    });
    // Constants of i128 are illegal (converted under A); test.low is illegal (converted under B);
    // the sink and the join are legal and stay.
    let legal = |op: &Operation| DynamicLegalityResult::legal_if(has_no_wide_types(op));
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Constant, _>(legal)
        .add_illegal_op::<Low>();

    convert(function, target, [split_constant(&context, &a), low_under_b])
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let output = function.borrow().to_string();
    println!("{output}");
    // The sink reads A's join of the constant's halves.
    assert_eq!(output.matches("test.join").count(), 2, "{output}");
    assert!(!output.contains("test.m_join"), "{output}");
    // The low of the join that stayed was given B's halves of it.
    assert!(output.contains("test.m_low"), "{output}");
    assert!(!output.contains("test.low"), "{output}");
    assert!(!output.contains("unrealized_conversion_cast"), "{output}");
}

/// A value callback that answers from what defines the value: after the producer is replaced the
/// user reads a cast, the callback no longer answers for it, and the conversion that would have
/// folded away needs a materialization (here: fails, there is none).
#[test]
fn a_value_callback_that_looks_at_the_definition_stops_answering_after_a_replacement() {
    let (context, function) = parse_function(
        "()",
        "    %0 = test.constant 5 : i128;
    test.sink %0 : (i128);
    builtin.ret;",
    );
    let (i64, i128) = (context.i64(), context.i128());
    // Only the results of constants are split.
    let mut converter = TypeConverter::new();
    converter.add_value_conversion(move |_, value| {
        let value = value.borrow();
        let defined_by_constant =
            value.get_defining_op().is_some_and(|op| op.borrow().is::<Constant>());
        (value.ty() == i128 && defined_by_constant).then(|| TypeConversion::many([i64, i64]))
    });
    // The sink is given the cast's result as itself (the callback does not answer for it), so
    // its pattern builds a sink of the same i128, which is illegal again: the pattern loops until
    // the limit.
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(split_constant(&context, &converter));
    set.push(flatten_sink(&context, &converter, Rc::default()));
    let err =
        apply_full_conversion(function, wide_target(&context), set, ConversionConfig::default())
            .unwrap_err();
    assert_eq!(
        format!("{err}"),
        "dialect conversion exceeded the limit of 1024 pattern applications that follow from one \
         operation, 'test.sink', while applying pattern 'flatten-sink' to 'test.sink'"
    );
}

/// The same callback when the user is converted first (it is not here: program order converts the
/// producer first). `is_legal_op` does not ask value callbacks: an operation it calls legal is
/// given split operands.
#[test]
fn is_legal_op_does_not_ask_value_callbacks() {
    let (context, function) = parse_function(
        "()",
        "    %0 = test.constant 5 : i128;
    test.sink %0 : (i128);
    builtin.ret;",
    );
    let (i64, i128) = (context.i64(), context.i128());
    let mut converter = TypeConverter::new();
    assert!(!converter.has_value_conversions());
    converter.add_value_conversion(move |_, value| {
        (value.borrow().ty() == i128).then(|| TypeConversion::many([i64, i64]))
    });
    assert!(converter.has_value_conversions());
    let sink = function
        .borrow()
        .regions()
        .front()
        .get()
        .unwrap()
        .entry()
        .body()
        .front()
        .get()
        .unwrap()
        .as_operation_ref()
        .next()
        .unwrap();
    assert!(converter.is_legal_op(&sink.borrow()), "is_legal_op asks only type callbacks");
    assert_eq!(
        converter
            .convert_value(&context, sink.borrow().operands()[0].borrow().as_value_ref())
            .len(),
        2
    );
}

/// Partial mode and the unknown operation policy: under the default policy (illegal) an operation
/// without a rule is offered to its pattern; under the policy `Legal` it is never offered, though
/// a pattern exists for it.
#[test]
fn in_partial_mode_a_rule_free_operation_is_offered_only_when_the_policy_does_not_call_it_legal() {
    for policy_legal in [false, true] {
        let (context, function) = parse_function(
            "() -> i64",
            "    %0 = test.constant 5 : i128;
    %1 = test.low %0;
    builtin.ret %1 : (i64);",
        );
        let mut converter = wide_converter(&context);
        join_callbacks(&mut converter);
        let low =
            Rule::new::<Low>(&context, "low", Some(converter.clone()), [], |op, operands, rw| {
                let [low, _high] = operands.get(0) else {
                    return Ok(false);
                };
                rw.replace_op(op, &[*low])?;
                Ok(true)
            });
        let mut target = ConversionTarget::new(context.clone());
        if policy_legal {
            target.set_unknown_op_policy(UnknownOpPolicy::Legal);
        }
        // No rule for anything.
        let mut set = ConversionPatternSet::new(context.clone());
        set.push(split_constant(&context, &converter));
        set.push(low);
        let result = apply_partial_conversion(function, target, set, ConversionConfig::default())
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        let output = function.borrow().to_string();
        println!(
            "policy legal = {policy_legal}: {} applications\n{output}",
            result.converted_ops()
        );
        assert_eq!(result.converted_ops(), if policy_legal { 0 } else { 2 });
    }
}

/// A pattern that builds its replacement after the operation (nothing in the contract says where
/// the replacement goes, only that it is made through the rewriter).
fn split_constant_after(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    let i128 = context.i128();
    Rule::new::<Constant>(
        context,
        "split-constant-after",
        Some(converter.clone()),
        [registered::<Constant>(context)],
        move |op, _operands, rewriter| {
            let constant = op.try_downcast_op::<Constant>().unwrap();
            if constant.borrow().result().ty() != i128 {
                return Ok(false);
            }
            let bits = constant.borrow().get_value().unsigned();
            let span = op.borrow().span();
            rewriter.set_insertion_point_after(op);
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

/// The values every operation of the entry block reads are defined before it.
fn defined_before_use(function: OperationRef) -> Result<(), String> {
    let function_op = function.try_downcast_op::<Function>().unwrap();
    let entry = function_op.borrow().entry_block();
    let mut defined = entry.borrow().argument_values().collect::<Vec<_>>();
    for op in entry.borrow().body().iter() {
        for operand in op.operands().iter() {
            let value = operand.borrow().as_value_ref();
            if !defined.contains(&value) {
                return Err(format!(
                    "{} is used by '{}' before it is defined",
                    value.borrow(),
                    op.name()
                ));
            }
        }
        defined.extend(op.results().iter().map(|r| r.borrow().as_value_ref()));
    }
    Ok(())
}

/// A replacement built after the operation: the source cast for the user that stays, and so the
/// join that realizes it, is placed after the values it reads.
#[test]
fn a_replacement_built_after_the_operation_is_read_after_it() {
    let (context, function) = parse_function(
        "()",
        "    %0 = test.constant 5 : i128;
    test.sink %0 : (i128);
    builtin.ret;",
    );
    let mut converter = wide_converter(&context);
    join_callbacks(&mut converter);
    convert(
        function,
        sink_legal_target(&context),
        [split_constant_after(&context, &converter)],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));
    defined_before_use(function).unwrap_or_else(|err| panic!("{err}\n{}", function.borrow()));
}

/// A pattern for `test.sink` that counts the operations it is offered and matches none.
fn counting_sink(context: &Rc<Context>, converter: &TypeConverter, offers: Seen<()>) -> Rule {
    Rule::new::<Sink>(context, "counting-sink", Some(converter.clone()), [], move |_, _, _| {
        offers.borrow_mut().push(());
        Ok(false)
    })
}

/// In a partial conversion, an operation the target has no rule for is left alone when the
/// unknown operation policy calls it legal, by `Legal` or by a dynamic policy that answers legal:
/// it is never offered to the pattern that would match it.
#[test]
fn in_partial_mode_a_rule_free_operation_the_policy_calls_legal_is_not_offered() {
    let policies: [fn() -> UnknownOpPolicy; 2] = [
        || UnknownOpPolicy::Legal,
        || UnknownOpPolicy::Dynamic(Rc::new(|_| DynamicLegalityResult::legal_if(true))),
    ];
    for policy in policies {
        let (context, function) = parse_function(
            "(%0: i128)",
            "    test.sink %0 : (i128);
    builtin.ret;",
        );
        let before = function.borrow().to_string();
        let converter = wide_converter(&context);
        let offers: Seen<()> = Rc::default();
        let mut target = ConversionTarget::new(context.clone());
        target.set_unknown_op_policy(policy());
        let mut set = ConversionPatternSet::new(context.clone());
        set.push(counting_sink(&context, &converter, offers.clone()));

        let result = apply_partial_conversion(function, target, set, ConversionConfig::default())
            .unwrap_or_else(|err| panic!("{}", describe(&err)));

        assert_eq!(offers.borrow().len(), 0, "the sink is not offered");
        assert_eq!(result.converted_ops(), 0);
        assert_str_eq!(function.borrow().to_string(), before);
    }
}

/// In a partial conversion, the operations nested under a recursively legal operation are not
/// offered to the patterns, though a rule of the target calls them illegal, and the final check
/// does not report them.
#[test]
fn in_partial_mode_the_operations_under_a_recursively_legal_one_are_not_offered() {
    let mut test = Test::new("f", |_| ([], []));
    let context = test.context_rc();
    {
        let mut builder = test.function_builder();
        let b = builder.builder_mut();
        const S: SourceSpan = SourceSpan::UNKNOWN;
        let scope = b.create::<Scope, ()>(S)().unwrap();
        let region = scope.borrow().body().as_region_ref();
        let body = b.create_block(region, None, &[]);
        let after = scope.as_operation_ref();
        b.set_insertion_point_to_end(body);
        let inner = b.create::<Constant, _>(S)(Int::from_bits(128, 9)).unwrap();
        let inner = inner.borrow().result().as_value_ref();
        b.create::<Sink, (ValueRef, Vec<ValueRef>)>(S)(inner, vec![]).unwrap();
        b.set_insertion_point_after(after);
        b.create::<crate::dialects::builtin::Ret, (Vec<ValueRef>,)>(S)(vec![]).unwrap();
    }
    let function = test.function().as_operation_ref();
    let before = function.borrow().to_string();
    let converter = wide_converter(&context);
    let offers: Seen<()> = Rc::default();
    let mut target = ConversionTarget::new(context.clone());
    target
        .add_legal_op::<Function>()
        .add_legal_op::<crate::dialects::builtin::Ret>()
        .add_legal_op::<Scope>()
        .mark_op_recursively_legal::<Scope>()
        .add_illegal_op::<Constant>()
        .add_illegal_op::<Sink>();
    let mut set = ConversionPatternSet::new(context.clone());
    set.push(split_constant(&context, &converter));
    set.push(counting_sink(&context, &converter, offers.clone()));

    let result = apply_partial_conversion(function, target, set, ConversionConfig::default())
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    assert_eq!(offers.borrow().len(), 0, "the nested sink is not offered");
    assert_eq!(result.converted_ops(), 0, "the nested constant is not offered");
    assert_str_eq!(function.borrow().to_string(), before);
}

/// A target under which a constant without an `i128` is legal and every other operation is: the
/// sink stays, reading the old type.
fn sink_legal_target(context: &Rc<Context>) -> ConversionTarget {
    let legal = |op: &Operation| DynamicLegalityResult::legal_if(has_no_wide_types(op));
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Constant, _>(legal);
    target
}

/// The halves built after the user that stays (before `builtin.ret`): no placement of the cast
/// can satisfy that user, and the replacement is an error that names it, raised before anything
/// changes.
#[test]
fn a_replacement_built_after_a_user_is_an_error() {
    let (context, function) = parse_function(
        "()",
        "    %0 = test.constant 5 : i128;
    test.sink %0 : (i128);
    builtin.ret;",
    );
    let before = function.borrow().to_string();
    let mut converter = wide_converter(&context);
    join_callbacks(&mut converter);
    let i128 = context.i128();
    let built_late = Rule::new::<Constant>(
        &context,
        "split-constant-late",
        Some(converter.clone()),
        [registered::<Constant>(&context)],
        move |op, _operands, rewriter| {
            let constant = op.try_downcast_op::<Constant>().unwrap();
            if constant.borrow().result().ty() != i128 {
                return Ok(false);
            }
            let bits = constant.borrow().get_value().unsigned();
            let span = op.borrow().span();
            let last = op.parent().unwrap().borrow().body().back().as_pointer().unwrap();
            rewriter.set_insertion_point_before(last);
            let low = rewriter.create_op::<Constant, _>(span, (Int::from_bits(64, bits),))?;
            let high =
                rewriter.create_op::<Constant, _>(span, (Int::from_bits(64, bits >> 64),))?;
            let low = low.borrow().result().as_value_ref();
            let high = high.borrow().result().as_value_ref();
            // Nothing is changed by the failing replacement: the halves are erased by hand so that
            // the IR can be compared with what it was.
            let result = rewriter.replace_op_with_multiple(op, &[&[low, high]]);
            for half in [high, low] {
                let half = half.borrow().get_defining_op().unwrap();
                rewriter.erase_op(half)?;
            }
            result?;
            Ok(true)
        },
    );
    let err = convert(function, sink_legal_target(&context), [built_late]).unwrap_err();
    let err = describe(&err);
    assert!(
        err.contains(
            "result 0 of 'test.constant' is replaced by a range defined after it, the last value \
             by 'test.constant', but its user 'test.sink' comes before that definition"
        ),
        "{err}"
    );
    assert_str_eq!(function.borrow().to_string(), before);
}

/// Replacing an operation by a value defined inside its own region, which is erased with it: an
/// error raised before anything changes.
#[test]
fn a_replacement_from_the_operations_own_region_is_an_error() {
    let mut test = Test::new("f", |_| ([], []));
    let context = test.context_rc();
    {
        let mut builder = test.function_builder();
        let b = builder.builder_mut();
        const S: SourceSpan = SourceSpan::UNKNOWN;
        let wrap = b.create::<Wrap, ()>(S)().unwrap();
        let region = wrap.borrow().body().as_region_ref();
        let body = b.create_block(region, None, &[]);
        let after = wrap.as_operation_ref();
        let wrapped = wrap.borrow().result().as_value_ref();
        b.set_insertion_point_to_end(body);
        b.create::<Constant, _>(S)(Int::from_bits(128, 7)).unwrap();
        b.set_insertion_point_after(after);
        b.create::<Sink, (ValueRef, Vec<ValueRef>)>(S)(wrapped, vec![]).unwrap();
        b.create::<crate::dialects::builtin::Ret, (Vec<ValueRef>,)>(S)(vec![]).unwrap();
    }
    let function = test.function().as_operation_ref();
    let before = function.borrow().to_string();
    let mut target = ConversionTarget::new(context.clone());
    target.set_unknown_op_policy(UnknownOpPolicy::Legal).add_illegal_op::<Wrap>();
    let unwrap = Rule::new::<Wrap>(&context, "unwrap", None, [], |op, _, rewriter| {
        let inner = {
            let op = op.borrow();
            let region = op.regions().front().get().unwrap();
            let entry = region.entry();
            let inner = entry.body().front().as_pointer().unwrap();
            inner.borrow().results()[0].borrow().as_value_ref()
        };
        rewriter.replace_op(op, &[inner])?;
        Ok(true)
    });
    let err = convert(function, target, [unwrap]).unwrap_err();
    let err = describe(&err);
    assert!(
        err.contains(
            "result 0 of 'test.wrap' is replaced by a range that holds '%1', a value defined \
             inside the operation being replaced, which is erased with it"
        ),
        "{err}"
    );
    assert_str_eq!(function.borrow().to_string(), before);
}

/// An argument remapped to an argument of a block nested in the block being converted (in the
/// region of one of its operations): an error raised before anything changes.
#[test]
fn remapping_to_an_argument_of_a_block_nested_in_the_block_is_an_error() {
    let (context, function) = parse_function(
        "(%0: i128)",
        "    test.forward_br ^bb1(%0 : i128);
^bb1(%1: i128):
    test.sink %1 : (i128);
    builtin.ret;",
    );
    let function_op = function.try_downcast_op::<Function>().unwrap();
    let block = function_op.borrow().entry_block().next().unwrap();
    let nested_argument = {
        let mut b = OpBuilder::new(context.clone());
        b.set_insertion_point_to_start(block);
        let scope = b.create::<Scope, ()>(SourceSpan::UNKNOWN)().unwrap();
        let region = scope.borrow().body().as_region_ref();
        let inner = b.create_block(region, None, &[context.i128()]);
        inner.borrow().arguments()[0] as ValueRef
    };
    let before = function.borrow().to_string();
    let mut rewriter = ConversionPatternRewriter::new_with_type_converter(
        context.clone(),
        Some(wide_converter(&context)),
    );
    let mut conversion = SignatureConversion::new(1);
    conversion.remap_input(0, &[nested_argument]).unwrap();
    let err = rewriter.apply_signature_conversion(block, &conversion).unwrap_err();
    assert_eq!(
        format!("{err}"),
        "argument 0 of a block of 'builtin.function' is remapped to '%2: i128', a value defined \
         in the block or in a region nested in it, which does not dominate its start"
    );
    assert_str_eq!(function.borrow().to_string(), before);
    assert_eq!(rewriter.mutation_count(), 0);
}

/// A pattern for an outer constant that changes in place a sink nested under a recursively legal
/// operation: the sink is then offered to its pattern, like any operation a pattern changes.
#[test]
fn an_in_place_change_under_a_recursively_legal_operation_is_offered() {
    let mut test = Test::new("f", |_| ([], []));
    let context = test.context_rc();
    let nested_sink;
    {
        let mut builder = test.function_builder();
        let b = builder.builder_mut();
        const S: SourceSpan = SourceSpan::UNKNOWN;
        let outer = b.create::<Constant, _>(S)(Int::from_bits(128, 7)).unwrap();
        let outer = outer.borrow().result().as_value_ref();
        let scope = b.create::<Scope, ()>(S)().unwrap();
        let region = scope.borrow().body().as_region_ref();
        let body = b.create_block(region, None, &[]);
        let after = scope.as_operation_ref();
        b.set_insertion_point_to_end(body);
        let inner = b.create::<Constant, _>(S)(Int::from_bits(128, 9)).unwrap();
        let inner = inner.borrow().result().as_value_ref();
        nested_sink = b.create::<Sink, (ValueRef, Vec<ValueRef>)>(S)(outer, vec![inner])
            .unwrap()
            .as_operation_ref();
        b.set_insertion_point_after(after);
        b.create::<crate::dialects::builtin::Ret, (Vec<ValueRef>,)>(S)(vec![]).unwrap();
    }
    let function = test.function().as_operation_ref();
    let mut converter = wide_converter(&context);
    join_callbacks(&mut converter);
    let mut target = wide_target(&context);
    target.add_legal_op::<Scope>().mark_op_recursively_legal::<Scope>();
    let i128 = context.i128();
    let poke = Rule::new::<Constant>(
        &context,
        "split-and-poke",
        Some(converter.clone()),
        [registered::<Constant>(&context)],
        move |op, _, rewriter| {
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
            rewriter.modify_op_in_place(nested_sink, |_| ())?;
            Ok(true)
        },
    );
    let seen: Seen<Vec<Ranges>> = Rc::default();
    let result =
        convert(function, target, [poke, flatten_sink(&context, &converter, seen.clone())]);
    assert!(result.is_ok(), "{}", function.borrow());
    assert_eq!(seen.borrow().len(), 1, "the nested sink is offered once");
}
