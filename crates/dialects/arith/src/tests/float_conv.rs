//! Tests of the float conversions and `bitcast`: building, type inference, verification, the round
//! trip of both forms, each fold rule at its boundaries, a seeded comparison with the host, and
//! range inference.

use runic_ir::Float;

use super::{
    float::{F32, F64, Folded, d, fold_value_of, rejection, s},
    *,
};

type ConvBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;

const FPTOSI: ConvBuilder = |b, v, ty| b.fptosi(v, ty, SPAN);
const FPTOUI: ConvBuilder = |b, v, ty| b.fptoui(v, ty, SPAN);
const SITOFP: ConvBuilder = |b, v, ty| b.sitofp(v, ty, SPAN);
const UITOFP: ConvBuilder = |b, v, ty| b.uitofp(v, ty, SPAN);
const FPEXT: ConvBuilder = |b, v, ty| b.fpext(v, ty, SPAN);
const FPTRUNC: ConvBuilder = |b, v, ty| b.fptrunc(v, ty, SPAN);
const BITCAST: ConvBuilder = |b, v, ty| b.bitcast(v, ty, SPAN);

/// Every operation, its name, and the legal (operand type, result type) pairs it is tested at.
fn legal_pairs() -> Vec<(&'static str, ConvBuilder, TypeFn, TypeFn)> {
    let mut pairs = Vec::new();
    for from in [F32, F64] {
        for to in [
            Context::i1,
            Context::i8,
            Context::i32,
            Context::i64,
            Context::i128,
            Context::isize,
        ] {
            pairs.push(("arith.fptosi", FPTOSI, from, to));
            pairs.push(("arith.fptoui", FPTOUI, from, to));
            pairs.push(("arith.sitofp", SITOFP, to, from));
            pairs.push(("arith.uitofp", UITOFP, to, from));
        }
    }
    pairs.push(("arith.fpext", FPEXT, F32, F64));
    pairs.push(("arith.fptrunc", FPTRUNC, F64, F32));
    pairs.push(("arith.bitcast", BITCAST, Context::i32, F32));
    pairs.push(("arith.bitcast", BITCAST, F32, Context::i32));
    pairs.push(("arith.bitcast", BITCAST, Context::i64, F64));
    pairs.push(("arith.bitcast", BITCAST, F64, Context::i64));
    pairs
}

#[test]
fn every_conversion_has_the_type_it_is_given_and_round_trips() {
    for (name, build, from, to) in legal_pairs() {
        let text = binary_round_trip(
            from,
            |b, v, _| {
                let ty = to(b.builder().context());
                vec![build(b, v, ty).unwrap()]
            },
            |cx, _| vec![to(cx)],
        );
        assert!(text.contains(&format!("= {name} %0 <{{ ty = ")), "{text}");
    }
}

#[test]
fn a_conversion_prints_its_target_type_as_a_property_like_the_integer_casts() {
    let text = unary_round_trip(
        F64,
        |b, v| {
            let ty = b.builder().context().i32();
            b.fptosi(v, ty, SPAN).unwrap()
        },
        |cx, _| cx.i32(),
    );
    assert!(text.contains("= arith.fptosi %0 <{ ty = #builtin.type<i32> }>;"), "{text}");
}

#[test]
fn a_conversion_rejects_an_operand_of_the_wrong_kind() {
    let cases: [(ConvBuilder, TypeFn, TypeFn, &str); 4] = [
        (FPTOSI, Context::i32, Context::i64, "expected a floating-point type, got i32"),
        (FPTOUI, Context::i32, Context::i64, "expected a floating-point type, got i32"),
        (SITOFP, F32, F64, "expected an integral type, got f32"),
        (UITOFP, F64, F32, "expected an integral type, got f64"),
    ];
    for (build, from, to, expected) in cases {
        let mut test = test_with(&[from]);
        let [a] = params(&test)[..] else {
            unreachable!()
        };
        let ty = to(test.context());
        let (message, _) = rejection(build(&mut test.function_builder(), a, ty));
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn a_conversion_whose_result_is_of_the_wrong_kind_is_rejected() {
    // `fptosi` to a float, `sitofp` to an integer.
    for (build, from, to) in [(FPTOSI, F64, F32), (SITOFP, Context::i32, Context::i64)] {
        let mut test = test_with(&[from]);
        let [a] = params(&test)[..] else {
            unreachable!()
        };
        let ty = to(test.context());
        rejection(build(&mut test.function_builder(), a, ty));
    }
}

#[test]
fn fpext_and_fptrunc_verify_their_exact_type_pair() {
    let cases: [(ConvBuilder, TypeFn, TypeFn, &str); 4] = [
        (
            FPEXT,
            F64,
            F64,
            "invalid 'arith.fpext' from `f64` to `f64`: it extends `f32` to `f64`, and `f64` to \
             `f64` is not that",
        ),
        (
            FPEXT,
            F32,
            F32,
            "invalid 'arith.fpext' from `f32` to `f32`: it extends `f32` to `f64`, and `f32` to \
             `f32` is not that",
        ),
        (
            FPTRUNC,
            F32,
            F32,
            "invalid 'arith.fptrunc' from `f32` to `f32`: it truncates `f64` to `f32`, and `f32` \
             to `f32` is not that",
        ),
        (
            FPTRUNC,
            F32,
            F64,
            "invalid 'arith.fptrunc' from `f32` to `f64`: it truncates `f64` to `f32`, and `f32` \
             to `f64` is not that",
        ),
    ];
    for (build, from, to, expected) in cases {
        let mut test = test_with(&[from]);
        let [a] = params(&test)[..] else {
            unreachable!()
        };
        let ty = to(test.context());
        let (message, labels) = rejection(build(&mut test.function_builder(), a, ty));
        assert_eq!(message, "invalid operation");
        assert_eq!(labels, [expected]);
    }
}

#[test]
fn bitcast_accepts_exactly_four_pairs_and_names_them_when_rejecting_another() {
    let all: [TypeFn; 8] = [
        Context::i1,
        Context::i8,
        Context::i32,
        Context::i64,
        Context::i128,
        Context::isize,
        F32,
        F64,
    ];
    let mut accepted = 0;
    for from in all {
        for to in all {
            let mut test = test_with(&[from]);
            let [a] = params(&test)[..] else {
                unreachable!()
            };
            let (from_ty, to_ty) = (from(test.context()), to(test.context()));
            let result = test.function_builder().bitcast(a, to_ty, SPAN);
            let pairs: [(TypeFn, TypeFn); 4] = [
                (Context::i32, F32),
                (F32, Context::i32),
                (Context::i64, F64),
                (F64, Context::i64),
            ];
            let allowed = pairs
                .iter()
                .any(|(f, t)| f(test.context()) == from_ty && t(test.context()) == to_ty);
            if allowed {
                accepted += 1;
                let value = result.unwrap();
                value.borrow().get_defining_op().unwrap().borrow().verify().unwrap();
            } else {
                let (message, labels) = rejection(result);
                assert_eq!(message, "invalid operation");
                assert_eq!(
                    labels,
                    [format!(
                        "invalid 'arith.bitcast' from `{from_ty}` to `{to_ty}`: `{from_ty}` to \
                         `{to_ty}` is not a bitcast the dialect has; the allowed ones are `i32` \
                         to `f32`, `f32` to `i32`, `i64` to `f64` and `f64` to `i64`"
                    )]
                );
            }
        }
    }
    assert_eq!(accepted, 4);
}

#[test]
fn a_conversion_whose_result_type_differs_from_its_ty_attribute_is_invalid() {
    let mut test = test_with(&[F64]);
    let [a] = params(&test)[..] else {
        unreachable!()
    };
    let ty = test.context().i32();
    let value = test.function_builder().fptosi(a, ty, SPAN).unwrap();
    let mut op = value.borrow().get_defining_op().unwrap();
    let i64 = test.context().i64();
    op.borrow_mut().results_mut()[0].borrow_mut().set_type(i64);
    let (message, labels) = rejection(Ok(value));
    assert_eq!(message, "invalid operation");
    assert_eq!(
        labels,
        [
            "invalid 'arith.fptosi' from `f64` to `i64`: its result type `i64` is not its `ty` \
             attribute `i32`"
        ]
    );
}

fn int_of(width: u32, signed: i128) -> Int {
    Int::from_signed(width, signed).unwrap()
}

fn uint_of(width: u32, unsigned: u128) -> Int {
    Int::from_unsigned(width, unsigned).unwrap()
}

/// Folds `build` applied to the float constant `value` and the result type `to`.
fn fold_float_conv(value: Float, build: ConvBuilder, to: TypeFn) -> Option<Folded> {
    let mut test = test_with(&[]);
    let mut b = test.function_builder();
    let operand = b.float(value, SPAN);
    let ty = to(b.builder().context());
    let result = build(&mut b, operand, ty).unwrap();
    fold_value_of(result)
}

/// Folds `build` applied to the integer constant `value` (of the type its width gives, or `isize`
/// for `isize`) and the result type `to`.
fn fold_int_conv(value: Int, isize: bool, build: ConvBuilder, to: TypeFn) -> Option<Folded> {
    let mut test = test_with(&[]);
    let mut b = test.function_builder();
    let ty = if isize {
        b.builder().context().isize()
    } else {
        b.builder().context().int_type(value.width())
    };
    let operand = b.int(value, ty, SPAN);
    let ty = to(b.builder().context());
    let result = build(&mut b, operand, ty).unwrap();
    fold_value_of(result)
}

#[test]
fn fptosi_folds_exactly_the_values_that_are_representable() {
    let to_i32 = |value| fold_float_conv(d(value), FPTOSI, Context::i32);
    assert_eq!(to_i32(2147483647.0), Some(Folded::Int(int_of(32, i32::MAX.into()))));
    assert_eq!(to_i32(2147483648.0), None);
    assert_eq!(to_i32(-2147483648.0), Some(Folded::Int(int_of(32, i32::MIN.into()))));
    assert_eq!(to_i32(-2147483649.0), None);
    assert_eq!(to_i32(-2147483648.9), Some(Folded::Int(int_of(32, i32::MIN.into()))));
    assert_eq!(to_i32(-0.9), Some(Folded::Int(int_of(32, 0))));
    assert_eq!(to_i32(0.9), Some(Folded::Int(int_of(32, 0))));
    assert_eq!(to_i32(-0.0), Some(Folded::Int(int_of(32, 0))));
    assert_eq!(to_i32(-7.99), Some(Folded::Int(int_of(32, -7))));
    assert_eq!(to_i32(f64::NAN), None);
    assert_eq!(to_i32(f64::INFINITY), None);
    assert_eq!(to_i32(f64::NEG_INFINITY), None);
    // The largest f64 below 2^63, and 2^63; and the same at 2^127.
    let below_2_63 = f64::from_bits(2f64.powi(63).to_bits() - 1);
    let to_i64 = |value| fold_float_conv(d(value), FPTOSI, Context::i64);
    assert_eq!(to_i64(below_2_63), Some(Folded::Int(int_of(64, below_2_63 as i64 as i128))));
    assert_eq!(to_i64(2f64.powi(63)), None);
    assert_eq!(to_i64(-(2f64.powi(63))), Some(Folded::Int(int_of(64, i64::MIN.into()))));
    assert_eq!(to_i64(-(2f64.powi(63)) - 2048.0), None);
    let below_2_127 = f64::from_bits(2f64.powi(127).to_bits() - 1);
    let to_i128 = |value| fold_float_conv(d(value), FPTOSI, Context::i128);
    assert_eq!(to_i128(below_2_127), Some(Folded::Int(int_of(128, below_2_127 as i128))));
    assert_eq!(to_i128(2f64.powi(127)), None);
    assert_eq!(to_i128(-(2f64.powi(127))), Some(Folded::Int(int_of(128, i128::MIN))));
    // `f32`: 2^31 does not fit an i32, the float below it does.
    let below_2_31 = f32::from_bits(2f32.powi(31).to_bits() - 1);
    let from_f32 = |value| fold_float_conv(s(value), FPTOSI, Context::i32);
    assert_eq!(from_f32(below_2_31), Some(Folded::Int(int_of(32, below_2_31 as i32 as i128))));
    assert_eq!(from_f32(2f32.powi(31)), None);
    // A narrow result: an i8 holds -128..=127, and an i1 holds -1 and 0.
    let to_i8 = |value| fold_float_conv(d(value), FPTOSI, Context::i8);
    assert_eq!(to_i8(127.9), Some(Folded::Int(int_of(8, 127))));
    assert_eq!(to_i8(128.0), None);
    assert_eq!(to_i8(-128.5), Some(Folded::Int(int_of(8, -128))));
    assert_eq!(to_i8(-129.0), None);
    assert_eq!(fold_float_conv(d(-1.0), FPTOSI, Context::i1), Some(Folded::Flag(true)));
    assert_eq!(fold_float_conv(d(1.0), FPTOSI, Context::i1), None);
}

#[test]
fn fptoui_folds_exactly_the_values_that_are_representable() {
    let to_i32 = |value| fold_float_conv(d(value), FPTOUI, Context::i32);
    assert_eq!(to_i32(4294967295.0), Some(Folded::Int(uint_of(32, u32::MAX.into()))));
    assert_eq!(to_i32(4294967295.9), Some(Folded::Int(uint_of(32, u32::MAX.into()))));
    assert_eq!(to_i32(4294967296.0), None);
    assert_eq!(to_i32(-0.9), Some(Folded::Int(uint_of(32, 0))));
    assert_eq!(to_i32(-0.0), Some(Folded::Int(uint_of(32, 0))));
    assert_eq!(to_i32(-1.0), None);
    assert_eq!(to_i32(-1.5), None);
    assert_eq!(to_i32(f64::NAN), None);
    assert_eq!(to_i32(f64::INFINITY), None);
    assert_eq!(to_i32(f64::NEG_INFINITY), None);
    let below_2_64 = f64::from_bits(2f64.powi(64).to_bits() - 1);
    let to_i64 = |value| fold_float_conv(d(value), FPTOUI, Context::i64);
    assert_eq!(to_i64(below_2_64), Some(Folded::Int(uint_of(64, below_2_64 as u64 as u128))));
    assert_eq!(to_i64(2f64.powi(64)), None);
    // The largest float below 2^63 and 2^63 itself are both fine for an unsigned i64.
    assert_eq!(to_i64(2f64.powi(63)), Some(Folded::Int(uint_of(64, 1 << 63))));
    let below_2_128 = f64::from_bits(2f64.powi(128).to_bits() - 1);
    let to_i128 = |value| fold_float_conv(d(value), FPTOUI, Context::i128);
    assert_eq!(to_i128(below_2_128), Some(Folded::Int(uint_of(128, below_2_128 as u128))));
    assert_eq!(to_i128(2f64.powi(128)), None);
    let below_2_127 = f64::from_bits(2f64.powi(127).to_bits() - 1);
    assert_eq!(to_i128(below_2_127), Some(Folded::Int(uint_of(128, below_2_127 as u128))));
    assert_eq!(to_i128(2f64.powi(127)), Some(Folded::Int(uint_of(128, 1 << 127))));
    assert_eq!(
        fold_float_conv(s(255.5), FPTOUI, Context::i8),
        Some(Folded::Int(uint_of(8, 255)))
    );
    assert_eq!(fold_float_conv(s(256.0), FPTOUI, Context::i8), None);
    assert_eq!(fold_float_conv(d(1.0), FPTOUI, Context::i1), Some(Folded::Flag(true)));
    assert_eq!(fold_float_conv(d(2.0), FPTOUI, Context::i1), None);
}

#[test]
fn a_float_to_isize_conversion_folds_only_where_both_targets_agree() {
    // 2^31 - 1 fits in 32 bits; 2^31 does not, so the 32-bit target differs.
    assert_eq!(
        fold_float_conv(d(2147483647.0), FPTOSI, Context::isize),
        Some(Folded::Int(int_of(64, i32::MAX.into())))
    );
    assert_eq!(fold_float_conv(d(2147483648.0), FPTOSI, Context::isize), None);
    assert_eq!(
        fold_float_conv(d(-2147483648.0), FPTOSI, Context::isize),
        Some(Folded::Int(int_of(64, i32::MIN.into())))
    );
    assert_eq!(fold_float_conv(d(-2147483649.0), FPTOSI, Context::isize), None);
    assert_eq!(
        fold_float_conv(d(4294967295.0), FPTOUI, Context::isize),
        Some(Folded::Int(uint_of(64, u32::MAX.into())))
    );
    assert_eq!(fold_float_conv(d(4294967296.0), FPTOUI, Context::isize), None);
    assert_eq!(fold_float_conv(d(f64::NAN), FPTOSI, Context::isize), None);
}

#[test]
fn integer_to_float_conversions_round_to_nearest_even() {
    let signed =
        |value: i128, width, to: TypeFn| fold_int_conv(int_of(width, value), false, SITOFP, to);
    let unsigned =
        |value: u128, width, to: TypeFn| fold_int_conv(uint_of(width, value), false, UITOFP, to);
    assert_eq!(signed(5, 32, F64), Some(Folded::Float(d(5.0))));
    assert_eq!(signed(-5, 8, F32), Some(Folded::Float(s(-5.0))));
    // The same bits are -1 signed and 255 unsigned.
    assert_eq!(unsigned(255, 8, F64), Some(Folded::Float(d(255.0))));
    assert_eq!(signed(-1, 8, F64), Some(Folded::Float(d(-1.0))));
    assert_eq!(
        fold_int_conv(Int::from_bits(8, 255), false, SITOFP, F64),
        Some(Folded::Float(d(-1.0)))
    );
    // 2^53 + 1 is a tie between 2^53 and 2^53 + 2; it rounds to the even one, 2^53.
    assert_eq!(signed((1 << 53) + 1, 64, F64), Some(Folded::Float(d(9007199254740992.0))));
    assert_eq!(signed((1 << 53) + 3, 64, F64), Some(Folded::Float(d(9007199254740996.0))));
    // 16777217 is a tie between 2^24 and 2^24 + 2.
    assert_eq!(signed(16777217, 32, F32), Some(Folded::Float(s(16777216.0))));
    assert_eq!(signed(16777219, 32, F32), Some(Folded::Float(s(16777220.0))));
    assert_eq!(
        unsigned(u64::MAX.into(), 64, F64),
        Some(Folded::Float(d(18446744073709551616.0)))
    );
    assert_eq!(
        unsigned(u64::MAX.into(), 64, F32),
        Some(Folded::Float(s(18446744073709551616.0)))
    );
    assert_eq!(signed(-1, 64, F32), Some(Folded::Float(s(-1.0))));
    assert_eq!(unsigned(u128::MAX, 128, F64), Some(Folded::Float(d(u128::MAX as f64))));
    assert_eq!(
        signed(i128::MIN, 128, F32),
        Some(Folded::Float(s(-170141183460469231731687303715884105728.0)))
    );
    // `u128::MAX` is closer to 2^128 than to the largest `f32`, so it rounds to infinity.
    assert_eq!(unsigned(u128::MAX, 128, F32), Some(Folded::Float(s(f32::INFINITY))));
    assert_eq!(signed(0, 1, F64), Some(Folded::Float(d(0.0))));
    assert_eq!(signed(-1, 1, F64), Some(Folded::Float(d(-1.0))));
    assert_eq!(unsigned(1, 1, F64), Some(Folded::Float(d(1.0))));
}

#[test]
fn an_isize_to_float_conversion_folds_only_where_both_targets_agree() {
    let signed = |value: i64| fold_int_conv(Int::from(value), true, SITOFP, F64);
    let unsigned = |value: u64| fold_int_conv(Int::from(value as i64), true, UITOFP, F64);
    assert_eq!(signed(5), Some(Folded::Float(d(5.0))));
    assert_eq!(signed(-5), Some(Folded::Float(d(-5.0))));
    assert_eq!(signed(i64::from(i32::MIN)), Some(Folded::Float(d(-2147483648.0))));
    assert_eq!(signed(1 << 31), None);
    assert_eq!(signed(1 << 40), None);
    assert_eq!(unsigned(u64::from(u32::MAX)), Some(Folded::Float(d(4294967295.0))));
    assert_eq!(unsigned(1 << 32), None);
    // A negative isize is a large unsigned value, which differs between the targets.
    assert_eq!(unsigned(u64::MAX), None);
}

#[test]
fn fpext_and_fptrunc_fold_unless_the_operand_is_a_nan() {
    assert_eq!(fold_float_conv(s(1.5), FPEXT, F64), Some(Folded::Float(d(1.5))));
    assert_eq!(fold_float_conv(s(-0.0), FPEXT, F64), Some(Folded::Float(d(-0.0))));
    assert_eq!(
        fold_float_conv(s(f32::NEG_INFINITY), FPEXT, F64),
        Some(Folded::Float(d(f64::NEG_INFINITY)))
    );
    // The smallest subnormal `f32` is exact in `f64`.
    assert_eq!(
        fold_float_conv(Float::from_bits32(1), FPEXT, F64),
        Some(Folded::Float(d(f64::from(f32::from_bits(1)))))
    );
    assert_eq!(fold_float_conv(s(f32::NAN), FPEXT, F64), None);
    assert_eq!(fold_float_conv(Float::from_bits32(0xffc0_0001), FPEXT, F64), None);

    assert_eq!(fold_float_conv(d(1.5), FPTRUNC, F32), Some(Folded::Float(s(1.5))));
    // 1 + 2^-24 is a tie between 1 and 1 + 2^-23: it rounds to 1.
    assert_eq!(
        fold_float_conv(d(1.0 + 2f64.powi(-24)), FPTRUNC, F32),
        Some(Folded::Float(s(1.0)))
    );
    assert_eq!(
        fold_float_conv(d(1.0 + 3.0 * 2f64.powi(-24)), FPTRUNC, F32),
        Some(Folded::Float(s(1.0 + 2.0 * 2f32.powi(-23))))
    );
    // Too large becomes an infinity, with the sign; the largest `f32` stays.
    assert_eq!(fold_float_conv(d(1e300), FPTRUNC, F32), Some(Folded::Float(s(f32::INFINITY))));
    assert_eq!(
        fold_float_conv(d(-1e300), FPTRUNC, F32),
        Some(Folded::Float(s(f32::NEG_INFINITY)))
    );
    assert_eq!(
        fold_float_conv(d(f64::from(f32::MAX)), FPTRUNC, F32),
        Some(Folded::Float(s(f32::MAX)))
    );
    assert_eq!(
        fold_float_conv(d(f64::INFINITY), FPTRUNC, F32),
        Some(Folded::Float(s(f32::INFINITY)))
    );
    // A subnormal result, an underflow to zero keeping the sign.
    let subnormal = 2f64.powi(-140);
    assert_eq!(
        fold_float_conv(d(subnormal), FPTRUNC, F32),
        Some(Folded::Float(s(subnormal as f32)))
    );
    assert!(subnormal as f32 != 0.0 && (subnormal as f32) < f32::MIN_POSITIVE);
    assert_eq!(fold_float_conv(d(-1e-300), FPTRUNC, F32), Some(Folded::Float(s(-0.0))));
    // A NaN does not fold.
    assert_eq!(fold_float_conv(d(f64::NAN), FPTRUNC, F32), None);
    assert_eq!(fold_float_conv(Float::from_bits64(0xfff8_0000_0000_0001), FPTRUNC, F32), None);
}

#[test]
fn bitcast_folds_every_constant_a_nan_included() {
    let float_bits =
        |bits: u32| fold_int_conv(Int::from_bits(32, bits.into()), false, BITCAST, F32);
    assert_eq!(float_bits(0x3fc0_0000), Some(Folded::Float(s(1.5))));
    // Bits of a NaN with a payload are the value.
    assert_eq!(float_bits(0x7fc0_0001), Some(Folded::Float(Float::from_bits32(0x7fc0_0001))));
    assert_eq!(float_bits(0xff80_0000), Some(Folded::Float(s(f32::NEG_INFINITY))));
    let wide = fold_int_conv(Int::from_bits(64, 0x7ff8_0000_0000_0001), false, BITCAST, F64);
    assert_eq!(wide, Some(Folded::Float(Float::from_bits64(0x7ff8_0000_0000_0001))));
    assert_eq!(
        fold_float_conv(s(1.5), BITCAST, Context::i32),
        Some(Folded::Int(Int::from_bits(32, 0x3fc0_0000)))
    );
    assert_eq!(
        fold_float_conv(Float::from_bits32(0xffc0_0001), BITCAST, Context::i32),
        Some(Folded::Int(Int::from_bits(32, 0xffc0_0001)))
    );
    assert_eq!(
        fold_float_conv(d(-0.0), BITCAST, Context::i64),
        Some(Folded::Int(Int::from_bits(64, 1 << 63)))
    );
    assert_eq!(
        fold_float_conv(Float::from_bits64(0xfff8_0000_0000_0001), BITCAST, Context::i64),
        Some(Folded::Int(Int::from_bits(64, 0xfff8_0000_0000_0001)))
    );
}

#[test]
fn a_bitcast_of_a_bitcast_back_to_the_original_type_is_the_original_value() {
    let mut test = test_with(&[F32, Context::i32]);
    let [x, n] = params(&test)[..] else {
        unreachable!()
    };
    let mut b = test.function_builder();
    let (i32, f32) = (b.builder().context().i32(), b.builder().context().f32());
    let as_int = b.bitcast(x, i32, SPAN).unwrap();
    let back = b.bitcast(as_int, f32, SPAN).unwrap();
    let as_float = b.bitcast(n, f32, SPAN).unwrap();
    let back_int = b.bitcast(as_float, i32, SPAN).unwrap();
    let fold_to_value = |value: ValueRef| {
        let op = value.borrow().get_defining_op().unwrap();
        let op = op.borrow();
        let mut results = SmallVec::<[OpFoldResult; 1]>::new();
        let folded = op.as_trait::<dyn Foldable>().unwrap().fold(&mut results);
        match (folded.is_ok(), &results[..]) {
            (true, [OpFoldResult::Value(v)]) => Some(*v),
            (false, []) => None,
            other => panic!("{other:?}"),
        }
    };
    assert_eq!(fold_to_value(back), Some(x));
    assert_eq!(fold_to_value(back_int), Some(n));
    // A single bitcast of a parameter does not fold.
    assert_eq!(fold_to_value(as_int), None);
}

#[test]
fn the_folds_agree_through_fold_and_fold_with_and_a_failed_fold_pushes_nothing() {
    // `fold_value_of` asserts both; run a spread of operations and operands through it.
    for value in [d(0.0), d(-1.5), d(1e30), d(f64::NAN), d(f64::INFINITY)] {
        let cases: [(ConvBuilder, TypeFn); 4] = [
            (FPTOSI, Context::i32),
            (FPTOUI, Context::i64),
            (FPTRUNC, F32),
            (BITCAST, Context::i64),
        ];
        for (build, to) in cases {
            fold_float_conv(value, build, to);
        }
    }
}

#[test]
fn canonicalization_folds_a_conversion_and_keeps_one_that_would_be_poison() {
    let mut test = test_fn(|_| vec![], |cx| vec![cx.i32(), cx.i32(), cx.f64()]);
    {
        let mut b = test.function_builder();
        let (i32, f64) = (b.builder().context().i32(), b.builder().context().f64());
        let good = b.f64(41.9, SPAN);
        let bad = b.f64(1e10, SPAN);
        let n = b.int(Int::from(7i32), i32, SPAN);
        let a = b.fptosi(good, i32, SPAN).unwrap();
        let p = b.fptosi(bad, i32, SPAN).unwrap();
        let c = b.sitofp(n, f64, SPAN).unwrap();
        b.ret([a, p, c], SPAN).unwrap();
    }
    assert!(canonicalize(&test));
    let text = text_of(&test);
    assert!(text.contains("arith.constant 41 : i32"), "{text}");
    assert!(text.contains("arith.fptosi"), "{text}");
    assert!(text.contains("arith.constantf 7.0 : f64"), "{text}");
    assert!(!text.contains("arith.sitofp"), "{text}");
}

#[test]
fn a_seeded_sample_folds_to_exactly_the_host_conversion_when_it_folds() {
    use super::random::Rng;

    let mut rng = Rng(0x0f1c_0a47);
    let mut folded = [0usize; 6];
    for round in 0..800 {
        let bits = match rng.below(5) {
            0 => rng.next(),
            1 => rng.next() & 0x800f_ffff_ffff_ffff,
            2 => 0x3ff0_0000_0000_0000 + (rng.next() & 0x07ff_ffff_ffff_ffff),
            3 => 0x41e0_0000_0000_0000 ^ (rng.next() & 0x8000_0000_0fff_ffff),
            _ => 0xc1e0_0000_0000_0000 ^ (rng.next() & 0x0000_0000_0fff_ffff),
        };
        let wide = f64::from_bits(bits);
        let narrow = f32::from_bits((bits >> 32) as u32);
        // fptosi / fptoui to i32 and i64 of an f64: the host conversion agrees whenever in range.
        for (signed, width, to) in [
            (true, 32, Context::i32 as TypeFn),
            (false, 32, Context::i32),
            (true, 64, Context::i64),
            (false, 64, Context::i64),
        ] {
            let build = if signed { FPTOSI } else { FPTOUI };
            let got = fold_float_conv(d(wide), build, to);
            let t = wide.trunc();
            let in_range = !wide.is_nan()
                && if signed {
                    t >= -(2f64.powi(width - 1)) && t < 2f64.powi(width - 1)
                } else {
                    t > -1.0 && t < 2f64.powi(width)
                };
            if in_range {
                let expected = if signed {
                    Int::from_signed(width as u32, wide as i64 as i128).unwrap()
                } else {
                    Int::from_unsigned(width as u32, wide as u64 as u128).unwrap()
                };
                assert_eq!(got, Some(Folded::Int(expected)), "{wide:?} signed={signed}");
                folded[0] += 1;
            } else {
                assert_eq!(got, None, "{wide:?}");
            }
        }
        // fptrunc and fpext.
        match fold_float_conv(d(wide), FPTRUNC, F32) {
            Some(Folded::Float(got)) => {
                assert!(!wide.is_nan());
                assert_eq!(got.bits(), u64::from((wide as f32).to_bits()));
                folded[1] += 1;
            }
            None => assert!(wide.is_nan()),
            other => panic!("{other:?}"),
        }
        match fold_float_conv(s(narrow), FPEXT, F64) {
            Some(Folded::Float(got)) => {
                assert_eq!(got.bits(), f64::from(narrow).to_bits());
                folded[2] += 1;
            }
            None => assert!(narrow.is_nan()),
            other => panic!("{other:?}"),
        }
        // sitofp / uitofp of 64 bits, to both widths.
        let int = Int::from_bits(64, u128::from(bits));
        for (to, wide_result) in [(F64, true), (F32, false)] {
            let signed = fold_int_conv(int, false, SITOFP, to);
            let unsigned = fold_int_conv(int, false, UITOFP, to);
            let (expect_s, expect_u) = if wide_result {
                ((bits as i64 as f64).to_bits(), (bits as f64).to_bits())
            } else {
                (u64::from((bits as i64 as f32).to_bits()), u64::from((bits as f32).to_bits()))
            };
            assert_eq!(signed, Some(Folded::Float(float_of(wide_result, expect_s))));
            assert_eq!(unsigned, Some(Folded::Float(float_of(wide_result, expect_u))));
            folded[3] += 1;
        }
        // bitcast, both directions: a round trip is the identity on the bits.
        let to_float = fold_int_conv(int, false, BITCAST, F64);
        assert_eq!(to_float, Some(Folded::Float(Float::from_bits64(bits))));
        let to_int = fold_float_conv(Float::from_bits64(bits), BITCAST, Context::i64);
        assert_eq!(to_int, Some(Folded::Int(int)));
        folded[4] += 1;
        let _ = round;
    }
    assert!(folded[0] > 500 && folded[1] > 500 && folded[2] > 500, "{folded:?}");
}

fn float_of(wide: bool, bits: u64) -> Float {
    if wide {
        Float::from_bits64(bits)
    } else {
        Float::from_bits32(bits as u32)
    }
}

#[test]
fn the_integer_results_have_the_full_range_of_their_type() {
    use runic_ir::{ConstantIntRange, IntegerValueRange, traits::InferIntRangeInterface};

    // No conversion implements `InferIntRangeInterface`: the range analysis gives a result of an
    // operation without it the full range of its type (`IntegerValueRange::max_for_type`), which
    // cannot be mistaken for a narrower one.
    let mut test = test_with(&[F64, Context::i64]);
    let [x, n] = params(&test)[..] else {
        unreachable!()
    };
    let mut b = test.function_builder();
    let (i8, i64, f64) = (
        b.builder().context().i8(),
        b.builder().context().i64(),
        b.builder().context().f64(),
    );
    let ops = [
        b.fptosi(x, i8, SPAN).unwrap(),
        b.fptoui(x, i8, SPAN).unwrap(),
        b.bitcast(x, i64, SPAN).unwrap(),
    ];
    let _ = (n, f64);
    for value in ops {
        let op = value.borrow().get_defining_op().unwrap();
        assert!(op.borrow().as_trait::<dyn InferIntRangeInterface>().is_none());
        let ty = value.borrow().ty();
        assert_eq!(
            IntegerValueRange::max_for_type(ty).range(),
            Some(&ConstantIntRange::max_for_type(ty))
        );
    }
}
