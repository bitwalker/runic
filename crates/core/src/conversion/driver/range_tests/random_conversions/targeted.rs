//! Operations under a recursively legal operation, two converters that
//! disagree, which converter realizes a cast, value callbacks, and the partial mode's reading of
//! the unknown operation policy.

use std::println;

use super::{ops::*, *};
use crate::{
    Builder, BuilderExt, RegionKind, RegionKindInterface, SourceSpan,
    derive::{EffectOpInterface, operation},
    effects::MemoryEffectOpInterface,
    testing::Test,
    traits::{NoTerminator, SingleBlock},
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
    converter.add_value_conversion(move |_, value| {
        (value.borrow().ty() == i128).then(|| TypeConversion::many([i64, i64]))
    });
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
    let legal = |op: &Operation| DynamicLegalityResult::legal_if(has_no_wide_types(op));
    let mut target = ConversionTarget::new(context.clone());
    target
        .set_unknown_op_policy(UnknownOpPolicy::Legal)
        .add_dynamically_legal_op::<Constant, _>(legal);
    convert(function, target, [split_constant_after(&context, &converter)])
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    defined_before_use(function).unwrap_or_else(|err| panic!("{err}\n{}", function.borrow()));
}
