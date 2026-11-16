//! Tests of the floating-point operations: building, inference, verification, the round trip of
//! both forms, the sixteen predicates, and each fold rule.

use runic_ir::{Float, dialects::builtin::attributes::FloatAttr};

use super::*;

const F32: TypeFn = Context::f32;
const F64: TypeFn = Context::f64;

type FloatBinaryBuilder = fn(Builder<'_, '_>, ValueRef, ValueRef) -> Result<ValueRef, Report>;

fn binary_ops() -> [(&'static str, FloatBinaryBuilder); 4] {
    [
        ("arith.addf", |b, l, r| b.addf(l, r, SPAN)),
        ("arith.subf", |b, l, r| b.subf(l, r, SPAN)),
        ("arith.mulf", |b, l, r| b.mulf(l, r, SPAN)),
        ("arith.divf", |b, l, r| b.divf(l, r, SPAN)),
    ]
}

/// The message and the label messages of the error that rejects the operation `built` gives: the
/// builder's own, or the verification error of the operation it built.
fn rejection(built: Result<ValueRef, Report>) -> (String, Vec<String>) {
    let err = match built {
        Err(err) => err,
        Ok(value) => {
            let op = value.borrow().get_defining_op().unwrap();
            let err = op.borrow().verify();
            err.expect_err("an invalid operation")
        }
    };
    (err.to_string(), label_messages(err.as_diagnostic()))
}

/// The message and the label messages of the verification error of the operation defining `value`.
fn verification_error(value: ValueRef) -> (String, Vec<String>) {
    rejection(Ok(value))
}

#[test]
fn binary_float_operations_have_the_operand_type_and_round_trip() {
    for (name, build) in binary_ops() {
        for ty in [F32, F64] {
            let text = binary_round_trip(ty, |b, l, r| vec![build(b, l, r).unwrap()], same);
            assert!(text.contains(&format!("= {name} %0, %1;")), "{text}");
        }
    }
}

#[test]
fn negation_has_the_operand_type_and_round_trips() {
    for ty in [F32, F64] {
        let text = unary_round_trip(ty, |b, v| b.negf(v, SPAN).unwrap(), same_one);
        assert!(text.contains("= arith.negf %0;"), "{text}");
    }
}

#[test]
fn comparison_gives_an_i1_and_round_trips_with_each_predicate() {
    for predicate in CmpFPredicate::ALL {
        for ty in [F32, F64] {
            let text = binary_round_trip(
                ty,
                |b, l, r| vec![b.cmpf(predicate, l, r, SPAN).unwrap()],
                |cx, _| vec![cx.i1()],
            );
            assert!(text.contains(&format!("= arith.cmpf {predicate} %0, %1;")), "{text}");
        }
    }
}

#[test]
fn constants_build_print_and_parse_in_both_widths() {
    for (value, text) in [
        (Float::from_f32(1.5), "arith.constantf 1.5 : f32"),
        (Float::from_f64(-0.0), "arith.constantf -0.0 : f64"),
        (Float::from_f64(f64::NAN), "arith.constantf 0x7ff8000000000000 : f64"),
        (Float::from_f32(f32::NEG_INFINITY), "arith.constantf 0xff800000 : f32"),
        (
            Float::from_bits64(0x7ff8_0000_0000_0001),
            "arith.constantf 0x7ff8000000000001 : f64",
        ),
    ] {
        let mut test = test_with(&[]);
        let constant = test.function_builder().float(value, SPAN);
        assert_eq!(constant.borrow().ty(), value.ty(test.context()));
        let (printed, context, ops) = finish(&mut test);
        assert!(printed.contains(text), "{printed}");
        let op = ops[0].borrow();
        let parsed = op.downcast_ref::<Constantf>().unwrap();
        assert_eq!(*parsed.get_value(), value);
        assert_eq!(parsed.result().ty(), value.ty(&context));
    }
    let mut test = test_with(&[]);
    let mut b = test.function_builder();
    let (single, double) = (b.f32(2.0, SPAN), b.f64(2.0, SPAN));
    assert_eq!(single.borrow().ty(), F32(test.context()));
    assert_eq!(double.borrow().ty(), F64(test.context()));
}

#[test]
fn a_constant_reads_a_literal_before_its_type_and_keeps_an_attribute_dictionary() {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<ArithDialect>();
    let text = r#"builtin.function public extern("C") @f() -> () {
        %a = arith.constantf 1.5e0 : f32;
        %b = arith.constantf 0x3ff8000000000000 : f64 attributes { note = #builtin.i32<1> };
        builtin.ret;
    };"#;
    let (_, printed) = parse_function_fixpoint(&context, "f", text).expect("parses");
    assert!(printed.contains("%0 = arith.constantf 1.5 : f32;"), "{printed}");
    assert!(
        printed.contains("%1 = arith.constantf 1.5 : f64 attributes { note = #builtin.i32<1> };"),
        "{printed}"
    );
}

#[test]
fn a_float_operation_verifies_its_operand_types() {
    // Operands of different float types.
    for (_, build) in binary_ops() {
        let mut test = test_with(&[F32, F64]);
        let [a, b] = params(&test)[..] else {
            unreachable!()
        };
        let result = build(&mut test.function_builder(), a, b);
        let (message, _) = rejection(result);
        assert!(message.contains("operand") || message.contains("invalid"), "{message}");
    }
    // An integer operand.
    for (_, build) in binary_ops() {
        let mut test = test_with(&[Context::i32, Context::i32]);
        let [a, b] = params(&test)[..] else {
            unreachable!()
        };
        let (message, _) = rejection(build(&mut test.function_builder(), a, b));
        assert!(message.contains("expected a floating-point type, got i32"), "{message}");
    }
    let mut test = test_with(&[Context::i32]);
    let [a] = params(&test)[..] else {
        unreachable!()
    };
    let (message, _) = rejection(test.function_builder().negf(a, SPAN));
    assert!(message.contains("expected a floating-point type, got i32"), "{message}");
    // A comparison of floats of different types, and of integers.
    let mut test = test_with(&[F32, F64]);
    let [a, b] = params(&test)[..] else {
        unreachable!()
    };
    rejection(test.function_builder().cmpf(CmpFPredicate::Olt, a, b, SPAN));
    let mut test = test_with(&[Context::i32, Context::i32]);
    let [a, b] = params(&test)[..] else {
        unreachable!()
    };
    rejection(test.function_builder().cmpf(CmpFPredicate::Olt, a, b, SPAN));
}

#[test]
fn a_constant_whose_value_type_is_not_its_result_type_is_invalid() {
    let mut test = test_with(&[]);
    let constant = test.function_builder().f32(1.0, SPAN);
    let mut op = constant.borrow().get_defining_op().unwrap();
    op.borrow().verify().expect("valid");
    let f64 = test.context().f64();
    op.borrow_mut().results_mut()[0].borrow_mut().set_type(f64);
    let (message, labels) = verification_error(constant);
    assert_eq!(message, "invalid operation");
    assert_eq!(
        labels,
        ["the result of 'arith.constantf' has type `f64`, but its value has type `f32`"]
    );
}

#[test]
fn the_sixteen_predicates_are_ieee_comparisons() {
    use CmpFPredicate::*;

    let values = [0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    for a in values {
        for b in values {
            let unordered = a.is_nan() || b.is_nan();
            let expected = [
                (False, false),
                (Oeq, !unordered && a == b),
                (Ogt, !unordered && a > b),
                (Oge, !unordered && a >= b),
                (Olt, !unordered && a < b),
                (Ole, !unordered && a <= b),
                (One, !unordered && a != b),
                (Ord, !unordered),
                (Ueq, unordered || a == b),
                (Ugt, unordered || a > b),
                (Uge, unordered || a >= b),
                (Ult, unordered || a < b),
                (Ule, unordered || a <= b),
                (Une, unordered || a != b),
                (Uno, unordered),
                (True, true),
            ];
            assert_eq!(expected.map(|(p, _)| p), CmpFPredicate::ALL);
            for (predicate, holds) in expected {
                assert_eq!(
                    predicate.evaluate(Float::from_f64(a), Float::from_f64(b)),
                    holds,
                    "{a} {predicate} {b}"
                );
                assert_eq!(
                    predicate.evaluate(Float::from_f32(a as f32), Float::from_f32(b as f32)),
                    holds,
                    "{a} {predicate} {b} (f32)"
                );
            }
        }
    }
    // Zeros are equal, and a NaN is not equal to itself, whatever its payload.
    let (zero, negative) = (Float::from_f64(0.0), Float::from_f64(-0.0));
    assert!(Oeq.evaluate(zero, negative) && !Olt.evaluate(negative, zero));
    let payload = Float::from_bits64(0x7ff8_0000_0000_0001);
    assert!(!Oeq.evaluate(payload, payload) && Une.evaluate(payload, payload));
    assert!(Olt.evaluate(Float::from_bits32(1), Float::from_bits32(2)));
}

#[test]
fn predicate_text_round_trips() {
    for predicate in CmpFPredicate::ALL {
        assert_eq!(predicate.to_string().parse::<CmpFPredicate>(), Ok(predicate));
    }
    assert!("lt".parse::<CmpFPredicate>().is_err());
}

/// What a folded operation gave: a float constant or an `i1` one.
#[derive(Debug, PartialEq, Eq)]
enum Folded {
    Float(Float),
    Flag(bool),
}

/// Folds the operation defining `value`, through `fold` and through `fold_with` with the operand
/// attributes, which must agree.
fn fold_value_of(value: ValueRef) -> Option<Folded> {
    let op = value.borrow().get_defining_op().unwrap();
    let attrs = op
        .borrow()
        .operands()
        .iter()
        .map(|operand| {
            let operand = operand.borrow().as_value_ref();
            let def = operand.borrow().get_defining_op()?;
            let def = def.borrow();
            def.as_trait::<dyn runic_ir::traits::ConstantLike>()?;
            def.get_attribute("value")
        })
        .collect::<Vec<_>>();
    let read = |with: bool| {
        let op = op.borrow();
        let foldable = op.as_trait::<dyn Foldable>().unwrap();
        let mut results = SmallVec::<[OpFoldResult; 1]>::new();
        let outcome = if with {
            foldable.fold_with(&attrs, &mut results)
        } else {
            foldable.fold(&mut results)
        };
        if !outcome.is_ok() {
            assert!(results.is_empty(), "a failed fold pushed results");
            return None;
        }
        let [OpFoldResult::Attribute(attr)] = results[..] else {
            panic!("expected one folded attribute");
        };
        Some(match attr.try_downcast_attr::<FloatAttr>() {
            Ok(float) => Folded::Float(*float.borrow().as_value()),
            Err(_) => {
                let attr = attr.borrow();
                let int =
                    attr.as_attr().as_trait::<dyn runic_ir::attributes::IntegerLikeAttr>().unwrap();
                assert_eq!(int.ty(), op.context().i1());
                Folded::Flag(int.as_int() == Int::from(true))
            }
        })
    };
    let (a, b) = (read(false), read(true));
    assert_eq!(a, b, "fold and fold_with disagree");
    a
}

/// Builds `build` on float constants `values` and folds it.
fn fold_floats(
    values: &[Float],
    build: impl FnOnce(Builder<'_, '_>, &[ValueRef]) -> ValueRef,
) -> Option<Folded> {
    let mut test = test_with(&[]);
    let mut b = test.function_builder();
    let operands = values.iter().map(|value| b.float(*value, SPAN)).collect::<Vec<_>>();
    let result = build(&mut b, &operands);
    fold_value_of(result)
}

fn fold_binary(which: usize, a: Float, b: Float) -> Option<Folded> {
    fold_floats(&[a, b], |builder, v| binary_ops()[which].1(builder, v[0], v[1]).unwrap())
}

fn d(value: f64) -> Float {
    Float::from_f64(value)
}

fn s(value: f32) -> Float {
    Float::from_f32(value)
}

#[test]
fn arithmetic_folds_on_constants_to_the_host_result() {
    assert_eq!(fold_binary(0, d(1.5), d(2.25)), Some(Folded::Float(d(3.75))));
    assert_eq!(fold_binary(1, d(1.5), d(2.25)), Some(Folded::Float(d(-0.75))));
    assert_eq!(fold_binary(2, d(1.5), d(2.0)), Some(Folded::Float(d(3.0))));
    assert_eq!(fold_binary(3, d(1.0), d(4.0)), Some(Folded::Float(d(0.25))));
    // An f32 operation is computed in f32: 2^24 + 1 rounds to 2^24.
    assert_eq!(fold_binary(0, s(16777216.0), s(1.0)), Some(Folded::Float(s(16777216.0))));
    // A zero result keeps its sign: -0.0 + -0.0.
    assert_eq!(fold_binary(0, d(-0.0), d(-0.0)), Some(Folded::Float(d(-0.0))));
    // Overflow gives an infinity, and a division by zero does not trap.
    assert_eq!(fold_binary(2, d(f64::MAX), d(2.0)), Some(Folded::Float(d(f64::INFINITY))));
    assert_eq!(fold_binary(3, d(1.0), d(0.0)), Some(Folded::Float(d(f64::INFINITY))));
    assert_eq!(fold_binary(3, d(-1.0), d(0.0)), Some(Folded::Float(d(f64::NEG_INFINITY))));
    assert_eq!(fold_binary(3, s(1.0), s(-0.0)), Some(Folded::Float(s(f32::NEG_INFINITY))));
}

#[test]
fn arithmetic_that_would_give_a_nan_does_not_fold() {
    let nan = d(f64::NAN);
    for which in 0..4 {
        assert_eq!(fold_binary(which, nan, d(1.0)), None);
        assert_eq!(fold_binary(which, d(1.0), nan), None);
        assert_eq!(fold_binary(which, s(f32::NAN), s(1.0)), None);
    }
    assert_eq!(fold_binary(3, d(0.0), d(0.0)), None);
    assert_eq!(fold_binary(3, d(f64::INFINITY), d(f64::INFINITY)), None);
    assert_eq!(fold_binary(1, d(f64::INFINITY), d(f64::INFINITY)), None);
    assert_eq!(fold_binary(0, d(f64::INFINITY), d(f64::NEG_INFINITY)), None);
    assert_eq!(fold_binary(2, d(0.0), d(f64::INFINITY)), None);
}

#[test]
fn an_operation_with_one_constant_operand_does_not_fold() {
    let mut test = test_with(&[F64]);
    let [x] = params(&test)[..] else {
        unreachable!()
    };
    let mut b = test.function_builder();
    let zero = b.f64(0.0, SPAN);
    let one = b.f64(1.0, SPAN);
    for (_, build) in binary_ops() {
        assert_eq!(fold_value_of(build(&mut b, x, zero).unwrap()), None);
        assert_eq!(fold_value_of(build(&mut b, zero, x).unwrap()), None);
    }
    let product = b.mulf(one, x, SPAN).unwrap();
    assert_eq!(fold_value_of(product), None);
    let negated = b.negf(x, SPAN).unwrap();
    assert_eq!(fold_value_of(negated), None);
    let compare = b.cmpf(CmpFPredicate::Oeq, x, one, SPAN).unwrap();
    assert_eq!(fold_value_of(compare), None);
    // `false` and `true` need no constants.
    for (predicate, value) in [(CmpFPredicate::False, false), (CmpFPredicate::True, true)] {
        let compare = b.cmpf(predicate, x, x, SPAN).unwrap();
        assert_eq!(fold_value_of(compare), Some(Folded::Flag(value)));
    }
}

#[test]
fn negation_folds_by_flipping_the_sign_bit_unless_the_operand_is_a_nan() {
    let negate = |value| fold_floats(&[value], |b, v| b.negf(v[0], SPAN).unwrap());
    assert_eq!(negate(d(1.5)), Some(Folded::Float(d(-1.5))));
    assert_eq!(negate(d(0.0)), Some(Folded::Float(d(-0.0))));
    assert_eq!(negate(d(-0.0)), Some(Folded::Float(d(0.0))));
    assert_eq!(negate(s(f32::INFINITY)), Some(Folded::Float(s(f32::NEG_INFINITY))));
    assert_eq!(negate(d(f64::NAN)), None);
    assert_eq!(negate(Float::from_bits32(0xffc0_0001)), None);
}

#[test]
fn comparison_folds_on_constants_including_nans() {
    let compare = |predicate, a, b| {
        fold_floats(&[a, b], |builder, v| builder.cmpf(predicate, v[0], v[1], SPAN).unwrap())
    };
    for predicate in CmpFPredicate::ALL {
        for (a, b) in [(d(1.0), d(2.0)), (d(f64::NAN), d(1.0)), (d(0.0), d(-0.0))] {
            assert_eq!(
                compare(predicate, a, b),
                Some(Folded::Flag(predicate.evaluate(a, b))),
                "{a} {predicate} {b}"
            );
        }
    }
    assert_eq!(compare(CmpFPredicate::Une, d(f64::NAN), d(f64::NAN)), Some(Folded::Flag(true)));
    assert_eq!(compare(CmpFPredicate::Oeq, d(f64::NAN), d(f64::NAN)), Some(Folded::Flag(false)));
}

#[test]
fn a_fold_declines_on_operands_of_types_a_verified_operation_does_not_have() {
    // An f32 and an f64 constant.
    assert_eq!(fold_binary(0, s(1.0), d(1.0)), None);
    let mut test = test_with(&[]);
    let mut b = test.function_builder();
    let (a, c) = (b.f32(1.0, SPAN), b.f64(1.0, SPAN));
    let compare = b.cmpf(CmpFPredicate::Oeq, a, c, SPAN).unwrap();
    assert_eq!(fold_value_of(compare), None);
}

#[test]
fn canonicalization_replaces_a_folded_operation_and_leaves_one_that_gives_a_nan() {
    let mut test = test_fn(|_| vec![], |cx| vec![cx.f64(), cx.f64()]);
    {
        let mut b = test.function_builder();
        let one = b.f64(1.0, SPAN);
        let two = b.f64(2.0, SPAN);
        let zero = b.f64(0.0, SPAN);
        let sum = b.addf(one, two, SPAN).unwrap();
        let nan = b.divf(zero, zero, SPAN).unwrap();
        b.ret([sum, nan], SPAN).unwrap();
    }
    assert!(canonicalize(&test));
    let text = text_of(&test);
    assert!(text.contains("arith.constantf 3.0 : f64"), "{text}");
    assert!(text.contains("arith.divf"), "{text}");
    assert!(!text.contains("arith.addf"), "{text}");
}

#[test]
fn a_seeded_sample_of_bit_patterns_folds_to_exactly_the_host_bits() {
    use super::random::Rng;

    let mut rng = Rng(0x00f1_0a74);
    let mut folded = [0usize; 4];
    for round in 0..4000 {
        let pick = |rng: &mut Rng| match rng.below(6) {
            0 => 0u64,
            1 => 1 << 63,
            2 => 0x7ff0_0000_0000_0000 | (rng.next() & 0x000f_ffff_ffff_ffff),
            3 => rng.next() & 0x000f_ffff_ffff_ffff,
            _ => rng.next(),
        };
        let (x, y) = (pick(&mut rng), pick(&mut rng));
        let wide = round % 2 == 0;
        let (a, b) = if wide {
            (Float::from_bits64(x), Float::from_bits64(y))
        } else {
            (Float::from_bits32((x >> 32) as u32), Float::from_bits32((y >> 32) as u32))
        };
        for which in 0..4 {
            let host = if wide {
                let (a, b) = (f64::from_bits(a.bits()), f64::from_bits(b.bits()));
                Float::from_f64([a + b, a - b, a * b, a / b][which])
            } else {
                let (a, b) = (f32::from_bits(a.bits() as u32), f32::from_bits(b.bits() as u32));
                Float::from_f32([a + b, a - b, a * b, a / b][which])
            };
            match fold_binary(which, a, b) {
                Some(Folded::Float(got)) => {
                    assert_eq!(got.bits(), host.bits(), "op {which} on {a:?} {b:?}");
                    assert!(!host.is_nan());
                    folded[which] += 1;
                }
                None => assert!(host.is_nan(), "op {which} on {a:?} {b:?} did not fold"),
                other => panic!("{other:?}"),
            }
        }
        match fold_floats(&[a], |bld, v| bld.negf(v[0], SPAN).unwrap()) {
            Some(Folded::Float(got)) => {
                assert!(!a.is_nan());
                assert_eq!(got.bits(), a.bits() ^ (1 << (a.width() - 1)));
            }
            None => assert!(a.is_nan()),
            other => panic!("{other:?}"),
        }
    }
    assert!(folded.iter().all(|count| *count > 100), "{folded:?}");
}

#[test]
fn materialize_constant_makes_a_constantf_of_a_float_attribute() {
    let mut test = test_with(&[]);
    let context = test.context_rc();
    let dialect = context.get_or_register_dialect::<ArithDialect>();

    for value in [Float::from_f32(1.5), Float::from_f64(f64::NAN), Float::from_f64(-0.0)] {
        let ty = value.ty(&context);
        let attr = context.float_attr(value);
        let op = dialect
            .materialize_constant(test.builder_mut(), attr.as_attribute_ref(), ty, SPAN)
            .expect("materialized");
        let op = op.borrow();
        assert!(AttributeRef::ptr_eq(
            &op.get_attribute("value").unwrap(),
            &attr.as_attribute_ref()
        ));
        assert_eq!(op.downcast_ref::<Constantf>().unwrap().result().ty(), ty);
    }
    // Another type than the attribute's, and one that is no float type.
    let attr = context.float_attr(Float::from_f32(1.0)).as_attribute_ref();
    for ty in [context.f64(), context.i32()] {
        assert!(dialect.materialize_constant(test.builder_mut(), attr, ty, SPAN).is_none());
    }
    // An integer attribute makes no constant of a float type.
    let int = context.int_attr(Int::from(1i32), context.i32()).as_attribute_ref();
    assert!(
        dialect
            .materialize_constant(test.builder_mut(), int, context.f32(), SPAN)
            .is_none()
    );
}
