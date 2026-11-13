//! Tests of the constant folds: each operation, built on constants of every tested type, folds to
//! what `Int`'s methods compute on the operand values, and does not fold what traps or is
//! undefined.

use runic_core::{IsizeFold, fold_isize};

use super::*;

/// The types every fold is checked at: the fixed widths 1, 8, 32, 64 and 128, and `isize`.
const FOLD_TYPES: [TypeFn; 6] = [
    Context::i1,
    Context::i8,
    Context::i32,
    Context::i64,
    Context::i128,
    Context::isize,
];

/// What a result of a folded operation is.
#[derive(Copy, Clone)]
enum Kind {
    /// A value of the operand type.
    Operand,
    /// An `i1`: a comparison or an overflow flag.
    Flag,
}

/// A computation of one result from the operand values, with `Int`'s methods.
type Compute<'a, const N: usize> = &'a dyn Fn([Int; N]) -> Option<Int>;

/// Builds an operation on the constant operands it is given and returns its results.
type BuildOn<'a, const N: usize> = &'a dyn Fn(Builder<'_, '_>, [ValueRef; N]) -> Vec<ValueRef>;

/// A count of bits, as `Int::count_ones`.
type Count = fn(Int) -> u32;

/// A resize to a width, as `Int::trunc`.
type Resize = fn(Int, u32) -> Int;

/// `value` sign-extended or truncated to `width`.
fn resized(value: i128, width: u32) -> Int {
    Int::from(value).sext_or_trunc(width)
}

/// The operand values checked at `width`, without repeats: 0, 1, -1, the signed minimum and
/// maximum, 5, and 2^32 + 7 (which is 7 on a 32-bit target).
fn samples(width: u32) -> Vec<Int> {
    let mut values: Vec<Int> = Vec::new();
    for value in [
        Int::zero(width),
        Int::one(width),
        Int::all_ones(width),
        Int::signed_min(width),
        Int::signed_max(width),
        resized(5, width),
        resized((1 << 32) + 7, width),
    ] {
        if !values.contains(&value) {
            values.push(value);
        }
    }
    values
}

fn singles(width: u32) -> Vec<[Int; 1]> {
    samples(width).into_iter().map(|value| [value]).collect()
}

fn pairs(width: u32) -> Vec<[Int; 2]> {
    let values = samples(width);
    values.iter().flat_map(|a| values.iter().map(move |b| [*a, *b])).collect()
}

/// Builds `build` in `test` on constants `values` of type `ty`, folds it through `fold` and through
/// `fold_with` with the constants' attributes, checks that the two agree, and returns the folded
/// results.
fn fold_on<const N: usize>(
    test: &mut Test,
    ty: Type,
    values: [Int; N],
    build: BuildOn<'_, N>,
) -> Option<Vec<(Int, Type)>> {
    let results = {
        let mut b = test.function_builder();
        let operands = values.map(|value| b.int(value, ty, SPAN));
        let results = build(&mut b, operands);
        let attrs = operands.map(|operand| {
            let op = operand.borrow().get_defining_op().unwrap();
            op.borrow().get_attribute("value")
        });
        let with = fold_values(results[0], Some(&attrs));
        (fold_values(results[0], None), with)
    };
    assert_eq!(results.0, results.1, "fold and fold_with disagree");
    results.0
}

/// What `expected` gives for `values` of type `ty`: each result computed directly for a fixed
/// width, or through `fold_isize` for `isize`; `None` unless every result folds.
fn expect<const N: usize>(
    context: &Context,
    ty: Type,
    values: [Int; N],
    expected: &[(Kind, Compute<'_, N>)],
) -> Option<Vec<(Int, Type)>> {
    expected
        .iter()
        .map(|(kind, compute)| {
            let (result_ty, isize_kind) = match kind {
                Kind::Operand => (ty, IsizeFold::PointerWidth),
                Kind::Flag => (context.i1(), IsizeFold::Fixed(1)),
            };
            let value = if ty.is_isize() {
                fold_isize(values, isize_kind, compute)
            } else {
                compute(values)
            };
            value.map(|value| (value, result_ty))
        })
        .collect()
}

/// Checks the folds of `build` at every type of [FOLD_TYPES] on the operand values `operands` gives
/// for the width of a constant of the type, against `expected`. Returns how many folded.
fn check_folds<const N: usize>(
    name: &str,
    build: BuildOn<'_, N>,
    expected: &[(Kind, Compute<'_, N>)],
    operands: impl Fn(u32) -> Vec<[Int; N]>,
) -> usize {
    let mut folded = 0;
    for ty in FOLD_TYPES {
        let mut test = test_with(&[]);
        let context = test.context_rc();
        let ty = ty(&context);
        let width = Int::width_for_type(ty).unwrap();
        for values in operands(width) {
            let want = expect(&context, ty, values, expected);
            let got = fold_on(&mut test, ty, values, build);
            assert_eq!(got, want, "{name} on {values:?} of type {ty}");
            folded += usize::from(got.is_some());
        }
    }
    assert!(folded > 0, "{name} never folded");
    folded
}

/// Folds `build` on the constants `values` of the type `ty` makes.
fn fold_at<const N: usize>(
    ty: TypeFn,
    values: [Int; N],
    build: BuildOn<'_, N>,
) -> Option<Vec<(Int, Type)>> {
    let mut test = test_with(&[]);
    let ty = ty(test.context());
    fold_on(&mut test, ty, values, build)
}

/// The value of the single result `build` folds to on the constants `values` of type `ty`.
fn fold_one_at<const N: usize>(ty: TypeFn, values: [Int; N], build: BuildOn<'_, N>) -> Option<Int> {
    fold_at(ty, values, build).map(|folded| {
        assert_eq!(folded.len(), 1);
        folded[0].0
    })
}

#[test]
fn add_sub_and_mul_fold_in_each_overflow_mode() {
    use Signedness::*;

    type Case<'a> = (&'a str, BuildOn<'a, 2>, Compute<'a, 2>);
    let cases: [Case<'_>; 15] = [
        ("add", &|b, [l, r]| vec![b.add(l, r, SPAN).unwrap()], &|[a, b]| {
            Some(a.wrapping_add(b))
        }),
        (
            "add checked signed",
            &|b, [l, r]| vec![b.add_checked(l, r, Signed, SPAN).unwrap()],
            &|[a, b]| a.checked_add_signed(b),
        ),
        (
            "add checked unsigned",
            &|b, [l, r]| vec![b.add_checked(l, r, Unsigned, SPAN).unwrap()],
            &|[a, b]| a.checked_add_unsigned(b),
        ),
        (
            "add unchecked signed",
            &|b, [l, r]| vec![b.add_unchecked(l, r, Signed, SPAN).unwrap()],
            &|[a, b]| a.checked_add_signed(b),
        ),
        (
            "add unchecked unsigned",
            &|b, [l, r]| vec![b.add_unchecked(l, r, Unsigned, SPAN).unwrap()],
            &|[a, b]| a.checked_add_unsigned(b),
        ),
        ("sub", &|b, [l, r]| vec![b.sub(l, r, SPAN).unwrap()], &|[a, b]| {
            Some(a.wrapping_sub(b))
        }),
        (
            "sub checked signed",
            &|b, [l, r]| vec![b.sub_checked(l, r, Signed, SPAN).unwrap()],
            &|[a, b]| a.checked_sub_signed(b),
        ),
        (
            "sub checked unsigned",
            &|b, [l, r]| vec![b.sub_checked(l, r, Unsigned, SPAN).unwrap()],
            &|[a, b]| a.checked_sub_unsigned(b),
        ),
        (
            "sub unchecked signed",
            &|b, [l, r]| vec![b.sub_unchecked(l, r, Signed, SPAN).unwrap()],
            &|[a, b]| a.checked_sub_signed(b),
        ),
        (
            "sub unchecked unsigned",
            &|b, [l, r]| vec![b.sub_unchecked(l, r, Unsigned, SPAN).unwrap()],
            &|[a, b]| a.checked_sub_unsigned(b),
        ),
        ("mul", &|b, [l, r]| vec![b.mul(l, r, SPAN).unwrap()], &|[a, b]| {
            Some(a.wrapping_mul(b))
        }),
        (
            "mul checked signed",
            &|b, [l, r]| vec![b.mul_checked(l, r, Signed, SPAN).unwrap()],
            &|[a, b]| a.checked_mul_signed(b),
        ),
        (
            "mul checked unsigned",
            &|b, [l, r]| vec![b.mul_checked(l, r, Unsigned, SPAN).unwrap()],
            &|[a, b]| a.checked_mul_unsigned(b),
        ),
        (
            "mul unchecked signed",
            &|b, [l, r]| vec![b.mul_unchecked(l, r, Signed, SPAN).unwrap()],
            &|[a, b]| a.checked_mul_signed(b),
        ),
        (
            "mul unchecked unsigned",
            &|b, [l, r]| vec![b.mul_unchecked(l, r, Unsigned, SPAN).unwrap()],
            &|[a, b]| a.checked_mul_unsigned(b),
        ),
    ];
    for (name, build, compute) in cases {
        check_folds(name, build, &[(Kind::Operand, compute)], pairs);
    }
}

#[test]
fn an_overflow_folds_only_when_wrapping() {
    use Signedness::*;

    let i8 = Context::i8 as TypeFn;
    // Each operation overflows `i8` read as signed and as unsigned.
    type Case<'a> = (&'a str, Int, Int, BuildOn<'a, 2>, [BuildOn<'a, 2>; 3]);
    let cases: [Case<'_>; 3] = [
        (
            "add",
            Int::from(-1i8),
            Int::from(-128i8),
            &|b, [l, r]| vec![b.add(l, r, SPAN).unwrap()],
            [
                &|b, [l, r]| vec![b.add_checked(l, r, Signed, SPAN).unwrap()],
                &|b, [l, r]| vec![b.add_unchecked(l, r, Unsigned, SPAN).unwrap()],
                &|b, [l, r]| vec![b.add_checked(l, r, Unsigned, SPAN).unwrap()],
            ],
        ),
        (
            "sub",
            Int::from(0x7fi8),
            Int::from(-1i8),
            &|b, [l, r]| vec![b.sub(l, r, SPAN).unwrap()],
            [
                &|b, [l, r]| vec![b.sub_unchecked(l, r, Signed, SPAN).unwrap()],
                &|b, [l, r]| vec![b.sub_checked(l, r, Unsigned, SPAN).unwrap()],
                &|b, [l, r]| vec![b.sub_unchecked(l, r, Unsigned, SPAN).unwrap()],
            ],
        ),
        (
            "mul",
            Int::from(-128i8),
            Int::from(-1i8),
            &|b, [l, r]| vec![b.mul(l, r, SPAN).unwrap()],
            [
                &|b, [l, r]| vec![b.mul_checked(l, r, Signed, SPAN).unwrap()],
                &|b, [l, r]| vec![b.mul_unchecked(l, r, Signed, SPAN).unwrap()],
                &|b, [l, r]| vec![b.mul_checked(l, r, Unsigned, SPAN).unwrap()],
            ],
        ),
    ];
    for (name, lhs, rhs, wrapping, overflows) in cases {
        assert!(fold_one_at(i8, [lhs, rhs], wrapping).is_some(), "wrapping {name} folds");
        for build in overflows {
            assert_eq!(fold_one_at(i8, [lhs, rhs], build), None, "{name} {lhs:?} {rhs:?}");
        }
    }
    assert_eq!(
        fold_one_at(i8, [Int::from(-1i8), Int::from(-128i8)], &|b, [l, r]| vec![
            b.add(l, r, SPAN).unwrap()
        ]),
        Some(Int::from(0x7fi8))
    );
}

#[test]
fn overflowing_operations_fold_the_flag_and_the_wrapped_value() {
    type Overflowing = fn(Int, Int) -> (Int, bool);
    type PairBuilder =
        fn(Builder<'_, '_>, ValueRef, ValueRef, Signedness) -> Result<(ValueRef, ValueRef), Report>;
    let ops: [(&str, PairBuilder, Overflowing, Overflowing); 3] = [
        (
            "add_overflowing",
            |b, l, r, s| b.add_overflowing(l, r, s, SPAN),
            Int::overflowing_add_signed,
            Int::overflowing_add_unsigned,
        ),
        (
            "sub_overflowing",
            |b, l, r, s| b.sub_overflowing(l, r, s, SPAN),
            Int::overflowing_sub_signed,
            Int::overflowing_sub_unsigned,
        ),
        (
            "mul_overflowing",
            |b, l, r, s| b.mul_overflowing(l, r, s, SPAN),
            Int::overflowing_mul_signed,
            Int::overflowing_mul_unsigned,
        ),
    ];
    for (name, build, signed, unsigned) in ops {
        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            let compute = match signedness {
                Signedness::Signed => signed,
                Signedness::Unsigned => unsigned,
            };
            let build_on = |b: Builder<'_, '_>, [l, r]: [ValueRef; 2]| {
                let (flag, value) = build(b, l, r, signedness).unwrap();
                vec![flag, value]
            };
            let flag = |[a, b]: [Int; 2]| Some(Int::from(compute(a, b).1));
            let value = |[a, b]: [Int; 2]| Some(compute(a, b).0);
            check_folds(
                &format!("{name} {signedness}"),
                &build_on,
                &[(Kind::Flag, &flag), (Kind::Operand, &value)],
                pairs,
            );
        }
    }
    let i8 = Context::i8 as TypeFn;
    let add = |s| {
        move |b: Builder<'_, '_>, [l, r]: [ValueRef; 2]| {
            let (flag, value) = b.add_overflowing(l, r, s, SPAN).unwrap();
            vec![flag, value]
        }
    };
    // `-1 + 1` overflows read as unsigned, not as signed; the wrapped sum is 0 either way.
    let folded =
        fold_at(i8, [Int::from(-1i8), Int::from(1i8)], &add(Signedness::Unsigned)).unwrap();
    assert_eq!(
        folded.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
        [Int::from(true), Int::from(0i8)]
    );
    let folded = fold_at(i8, [Int::from(-1i8), Int::from(1i8)], &add(Signedness::Signed)).unwrap();
    assert_eq!(
        folded.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
        [Int::from(false), Int::from(0i8)]
    );
}

#[test]
fn the_division_family_folds_except_on_a_zero_divisor_or_min_by_minus_one() {
    type PairBuilder =
        fn(Builder<'_, '_>, ValueRef, ValueRef) -> Result<(ValueRef, ValueRef), Report>;
    let single: [(&str, BinaryBuilder, Compute<'_, 2>); 4] = [
        ("udiv", |b, l, r| b.udiv(l, r, SPAN), &|[a, b]| a.div_unsigned(b)),
        ("sdiv", |b, l, r| b.sdiv(l, r, SPAN), &|[a, b]| a.div_signed(b)),
        ("urem", |b, l, r| b.urem(l, r, SPAN), &|[a, b]| a.rem_unsigned(b)),
        // `srem` traps where `sdiv` does.
        ("srem", |b, l, r| b.srem(l, r, SPAN), &|[a, b]| {
            a.div_signed(b).and(a.rem_signed(b))
        }),
    ];
    for (name, build, compute) in single {
        check_folds(
            name,
            &|b, [l, r]| vec![build(b, l, r).unwrap()],
            &[(Kind::Operand, compute)],
            pairs,
        );
    }
    let double: [(&str, PairBuilder, Compute<'_, 2>, Compute<'_, 2>); 2] = [
        (
            "udivrem",
            |b, l, r| b.udivrem(l, r, SPAN),
            &|[a, b]| a.div_unsigned(b),
            &|[a, b]| a.rem_unsigned(b),
        ),
        (
            "sdivrem",
            |b, l, r| b.sdivrem(l, r, SPAN),
            &|[a, b]| a.div_signed(b),
            &|[a, b]| a.rem_signed(b),
        ),
    ];
    for (name, build, quotient, remainder) in double {
        check_folds(
            name,
            &|b, [l, r]| {
                let (q, r) = build(b, l, r).unwrap();
                vec![q, r]
            },
            &[(Kind::Operand, quotient), (Kind::Operand, remainder)],
            pairs,
        );
    }

    let ops: [(&str, BuildOn<'_, 2>); 6] = [
        ("udiv", &|b, [l, r]| vec![b.udiv(l, r, SPAN).unwrap()]),
        ("sdiv", &|b, [l, r]| vec![b.sdiv(l, r, SPAN).unwrap()]),
        ("urem", &|b, [l, r]| vec![b.urem(l, r, SPAN).unwrap()]),
        ("srem", &|b, [l, r]| vec![b.srem(l, r, SPAN).unwrap()]),
        ("udivrem", &|b, [l, r]| {
            let (q, r) = b.udivrem(l, r, SPAN).unwrap();
            vec![q, r]
        }),
        ("sdivrem", &|b, [l, r]| {
            let (q, r) = b.sdivrem(l, r, SPAN).unwrap();
            vec![q, r]
        }),
    ];
    for (name, build) in ops {
        assert_eq!(
            fold_at(Context::i32, [Int::from(7i32), Int::from(0i32)], build),
            None,
            "{name}"
        );
        let min_by_minus_one =
            fold_at(Context::i32, [Int::from(i32::MIN), Int::from(-1i32)], build);
        assert_eq!(min_by_minus_one.is_none(), name.starts_with('s'), "{name}");
    }
    // The quotient comes first.
    let folded = fold_at(Context::i32, [Int::from(-7i32), Int::from(2i32)], ops[5].1).unwrap();
    assert_eq!(
        folded.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
        [Int::from(-3i32), Int::from(-1i32)]
    );
}

#[test]
fn bitwise_operations_neg_min_and_max_fold() {
    let binary: [(&str, BinaryBuilder, Compute<'_, 2>); 7] = [
        ("band", |b, l, r| b.band(l, r, SPAN), &|[a, b]| Some(a & b)),
        ("bor", |b, l, r| b.bor(l, r, SPAN), &|[a, b]| Some(a | b)),
        ("bxor", |b, l, r| b.bxor(l, r, SPAN), &|[a, b]| Some(a ^ b)),
        ("min signed", |b, l, r| b.min(l, r, Signedness::Signed, SPAN), &|[a, b]| {
            Some(a.min_signed(b))
        }),
        ("min unsigned", |b, l, r| b.min(l, r, Signedness::Unsigned, SPAN), &|[a, b]| {
            Some(a.min_unsigned(b))
        }),
        ("max signed", |b, l, r| b.max(l, r, Signedness::Signed, SPAN), &|[a, b]| {
            Some(a.max_signed(b))
        }),
        ("max unsigned", |b, l, r| b.max(l, r, Signedness::Unsigned, SPAN), &|[a, b]| {
            Some(a.max_unsigned(b))
        }),
    ];
    for (name, build, compute) in binary {
        check_folds(
            name,
            &|b, [l, r]| vec![build(b, l, r).unwrap()],
            &[(Kind::Operand, compute)],
            pairs,
        );
    }
    let unary: [(&str, UnaryBuilder, Compute<'_, 1>); 2] = [
        ("bnot", |b, v| b.bnot(v, SPAN), &|[a]| Some(!a)),
        ("neg", |b, v| b.neg(v, SPAN), &|[a]| Some(a.wrapping_neg())),
    ];
    for (name, build, compute) in unary {
        check_folds(
            name,
            &|b, [v]| vec![build(b, v).unwrap()],
            &[(Kind::Operand, compute)],
            singles,
        );
    }
}

#[test]
fn counts_fold_to_a_value_of_the_operand_type() {
    let count = |count: Count| {
        move |[a]: [Int; 1]| Some(Int::from_unsigned(a.width(), u128::from(count(a))).unwrap())
    };
    let ops: [(&str, UnaryBuilder, Count); 5] = [
        ("popcnt", |b, v| b.popcnt(v, SPAN), Int::count_ones),
        ("clz", |b, v| b.clz(v, SPAN), Int::leading_zeros),
        ("ctz", |b, v| b.ctz(v, SPAN), Int::trailing_zeros),
        ("clo", |b, v| b.clo(v, SPAN), Int::leading_ones),
        ("cto", |b, v| b.cto(v, SPAN), Int::trailing_ones),
    ];
    for (name, build, method) in ops {
        check_folds(
            name,
            &|b, [v]| vec![build(b, v).unwrap()],
            &[(Kind::Operand, &count(method))],
            singles,
        );
    }
    let clz: BuildOn<'_, 1> = &|b, [v]| vec![b.clz(v, SPAN).unwrap()];
    assert_eq!(fold_one_at(Context::i128, [Int::one(128)], clz), Some(Int::from(127i128)));
    assert_eq!(fold_one_at(Context::i1, [Int::zero(1)], clz), Some(Int::one(1)));
    // The leading zeros of an `isize` depend on the target, unless there are none.
    assert_eq!(fold_one_at(Context::isize, [Int::from(5i64)], clz), None);
    assert_eq!(fold_one_at(Context::isize, [Int::from(-5i64)], clz), Some(Int::zero(64)));
}

/// The shift amounts checked at `width`: 0, 1, `width - 1`, `width`, `width + 1`, and for a
/// rotation `2 * width + 3`, plus 31 and 32 (which decide an `isize` shift).
fn amounts(width: u32) -> Vec<[Int; 2]> {
    let mut amounts: Vec<Int> = Vec::new();
    for amount in [0, 1, width - 1, width, width + 1, 2 * width + 3, 31, 32] {
        let amount = Int::from(u128::from(amount)).trunc(width);
        if !amounts.contains(&amount) {
            amounts.push(amount);
        }
    }
    samples(width)
        .into_iter()
        .flat_map(|value| amounts.clone().into_iter().map(move |amount| [value, amount]))
        .collect()
}

#[test]
fn shifts_fold_below_the_width_and_rotates_at_every_amount() {
    let shift = |op: fn(Int, u32) -> Int| {
        move |[a, b]: [Int; 2]| {
            let amount = b.shift_amount_unsigned();
            (amount < a.width()).then(|| op(a, amount))
        }
    };
    let ops: [(&str, BinaryBuilder, Compute<'_, 2>); 5] = [
        ("shl", |b, l, r| b.shl(l, r, SPAN), &shift(|a, n| a << n)),
        ("shr", |b, l, r| b.shr(l, r, SPAN), &shift(Int::lshr)),
        ("ashr", |b, l, r| b.ashr(l, r, SPAN), &shift(Int::ashr)),
        ("rotl", |b, l, r| b.rotl(l, r, SPAN), &|[a, b]| {
            Some(a.rotl(b.rotate_amount_unsigned(a.width())))
        }),
        ("rotr", |b, l, r| b.rotr(l, r, SPAN), &|[a, b]| {
            Some(a.rotr(b.rotate_amount_unsigned(a.width())))
        }),
    ];
    for (name, build, compute) in ops {
        check_folds(
            name,
            &|b, [l, r]| vec![build(b, l, r).unwrap()],
            &[(Kind::Operand, compute)],
            amounts,
        );
        for ty in [Context::i8 as TypeFn, Context::i32, Context::i64, Context::i128] {
            let width = Int::width_for_type(ty(&Context::default())).unwrap();
            let value = Int::from(0b1011u8).zext(width);
            for (amount, folds) in [
                (0, true),
                (1, true),
                (width - 1, true),
                (width, name.starts_with("rot")),
                (width + 1, name.starts_with("rot")),
                (2 * width + 3, name.starts_with("rot")),
            ] {
                let amount = Int::from(u128::from(amount)).trunc(width);
                let folded =
                    fold_one_at(ty, [value, amount], &|b, [l, r]| vec![build(b, l, r).unwrap()]);
                assert_eq!(folded.is_some(), folds, "{name} by {amount:?} at width {width}");
            }
        }
    }
    let rotl: BuildOn<'_, 2> = &|b, [l, r]| vec![b.rotl(l, r, SPAN).unwrap()];
    assert_eq!(
        fold_one_at(Context::i8, [Int::from(0x81u8), Int::from(17u8)], rotl),
        Some(Int::from(0x03u8))
    );
}

#[test]
fn cmpi_folds_every_predicate() {
    for predicate in CmpPredicate::ALL {
        let operands = |width| {
            vec![
                [Int::all_ones(width), Int::one(width)],
                [Int::one(width), Int::one(width)],
                [Int::signed_min(width), Int::signed_max(width)],
            ]
        };
        check_folds(
            &format!("cmpi {predicate}"),
            &|b, [l, r]| vec![b.cmpi(predicate, l, r, SPAN).unwrap()],
            &[(Kind::Flag, &|[a, b]| Some(Int::from(predicate.evaluate(a, b))))],
            operands,
        );
    }
    let slt: BuildOn<'_, 2> = &|b, [l, r]| vec![b.slt(l, r, SPAN).unwrap()];
    let ult: BuildOn<'_, 2> = &|b, [l, r]| vec![b.ult(l, r, SPAN).unwrap()];
    let (minus_one, one) = (Int::from(-1i32), Int::from(1i32));
    assert_eq!(fold_one_at(Context::i32, [minus_one, one], slt), Some(Int::from(true)));
    assert_eq!(fold_one_at(Context::i32, [minus_one, one], ult), Some(Int::from(false)));
}

#[test]
fn isize_folds_only_what_32_and_64_bit_targets_agree_on() {
    let isize = Context::isize as TypeFn;
    let big = Int::from(1i64 << 32);
    let add: BuildOn<'_, 2> = &|b, [l, r]| vec![b.add(l, r, SPAN).unwrap()];
    assert_eq!(
        fold_one_at(isize, [big, Int::from(1i64)], add),
        Some(Int::from((1i64 << 32) + 1))
    );
    let udiv: BuildOn<'_, 2> = &|b, [l, r]| vec![b.udiv(l, r, SPAN).unwrap()];
    assert_eq!(fold_one_at(isize, [big, Int::from(2i64)], udiv), None);
    let ult: BuildOn<'_, 2> = &|b, [l, r]| vec![b.ult(l, r, SPAN).unwrap()];
    assert_eq!(fold_one_at(isize, [big, Int::from(1i64)], ult), None);
    assert_eq!(
        fold_one_at(isize, [Int::from(5i64), Int::from(7i64)], ult),
        Some(Int::from(true))
    );
    let shr: BuildOn<'_, 2> = &|b, [l, r]| vec![b.shr(l, r, SPAN).unwrap()];
    assert_eq!(fold_one_at(isize, [Int::from(-1i64), Int::from(1i64)], shr), None);
    let shl: BuildOn<'_, 2> = &|b, [l, r]| vec![b.shl(l, r, SPAN).unwrap()];
    assert_eq!(
        fold_one_at(isize, [Int::from(1i64), Int::from(31i64)], shl),
        Some(Int::from(1i64 << 31))
    );
    assert_eq!(fold_one_at(isize, [Int::from(1i64), Int::from(32i64)], shl), None);
    // A checked addition that overflows 32 bits traps on a 32-bit target.
    let checked: BuildOn<'_, 2> =
        &|b, [l, r]| vec![b.add_checked(l, r, Signedness::Signed, SPAN).unwrap()];
    assert_eq!(
        fold_one_at(isize, [Int::from(i64::from(i32::MAX)), Int::from(1i64)], checked),
        None
    );
    assert_eq!(
        fold_one_at(isize, [Int::from(-1i64), Int::from(1i64)], checked),
        Some(Int::from(0i64))
    );
}

/// Folds the cast `build` of the constant `value` of the type `from` makes to the type `to` makes,
/// and checks that the result has the type `to`.
fn fold_cast(
    build: CastBuilder,
    from: &dyn Fn(&Context) -> Type,
    to: &dyn Fn(&Context) -> Type,
    value: Int,
) -> Option<Int> {
    let mut test = test_with(&[]);
    let context = test.context_rc();
    let (from, to) = (from(&context), to(&context));
    let mut b = test.function_builder();
    let source = b.int(value, from, SPAN);
    let result = build(&mut b, source, to).unwrap();
    let attr = source.borrow().get_defining_op().unwrap().borrow().get_attribute("value");
    let folded = fold_value(result, None);
    assert_eq!(folded, fold_value(result, Some(&[attr])), "fold and fold_with disagree");
    folded.map(|(value, ty)| {
        assert_eq!(ty, to);
        value
    })
}

type CastBuilder = fn(Builder<'_, '_>, ValueRef, Type) -> Result<ValueRef, Report>;

#[test]
fn fixed_width_casts_fold_at_every_width() {
    let widths = [1, 8, 32, 64, 128];
    let casts: [(&str, CastBuilder, Resize, bool); 3] = [
        ("trunc", |b, v, t| b.trunc(v, t, SPAN), Int::trunc, true),
        ("zext", |b, v, t| b.zext(v, t, SPAN), Int::zext, false),
        ("sext", |b, v, t| b.sext(v, t, SPAN), Int::sext, false),
    ];
    for (name, build, resize, narrows) in casts {
        for from in widths {
            for to in widths.into_iter().filter(|to| if narrows { *to < from } else { *to > from })
            {
                for value in samples(from) {
                    let folded =
                        fold_cast(build, &|cx| cx.int_type(from), &|cx| cx.int_type(to), value);
                    assert_eq!(folded, Some(resize(value, to)), "{name} {value:?} to i{to}");
                }
            }
        }
    }
}

#[test]
fn isize_conversions_fold_when_both_targets_agree() {
    let casts: [(&str, CastBuilder, Resize); 2] = [
        ("sext_or_trunc", |b, v, t| b.sext_or_trunc(v, t, SPAN), Int::sext_or_trunc),
        ("zext_or_trunc", |b, v, t| b.zext_or_trunc(v, t, SPAN), Int::zext_or_trunc),
    ];
    for (name, build, resize) in casts {
        for width in [1, 8, 32, 48, 64, 128] {
            let fixed = |cx: &Context| cx.int_type(width);
            // From `isize`: the result is `width` bits wide on both targets, which must agree.
            for value in samples(64) {
                let expected =
                    fold_isize([value], IsizeFold::Fixed(width), |[a]| Some(resize(a, width)));
                let folded = fold_cast(build, &Context::isize, &fixed, value);
                assert_eq!(folded, expected, "{name} {value:?} to i{width}");
            }
            // To `isize`: the operand resized to the 64 bits of an `isize` constant.
            for value in samples(width) {
                let folded = fold_cast(build, &fixed, &Context::isize, value);
                assert_eq!(folded, Some(resize(value, 64)), "{name} {value:?} of i{width}");
            }
        }
    }

    let (sext, zext) = (casts[0].1, casts[1].1);
    let isize = &Context::isize;
    assert_eq!(fold_cast(sext, isize, &Context::i64, Int::from(5i64)), Some(Int::from(5i64)));
    assert_eq!(fold_cast(sext, isize, &Context::i64, Int::from(1i64 << 32)), None);
    assert_eq!(
        fold_cast(zext, &Context::i32, isize, Int::from(-1i32)),
        Some(Int::from(0xffff_ffffi64))
    );
    assert_eq!(fold_cast(sext, &Context::i32, isize, Int::from(-1i32)), Some(Int::from(-1i64)));
    assert_eq!(
        fold_cast(sext, &Context::i128, isize, Int::from((1i128 << 64) + 5)),
        Some(Int::from(5i64))
    );
}

#[test]
fn only_constants_of_the_operand_type_fold() {
    // A non-constant operand.
    let mut test = test_with(&[Context::i32]);
    let [param] = params(&test)[..] else {
        unreachable!()
    };
    let context = test.context_rc();
    context.get_or_register_dialect::<runic_dialect_ub::UndefinedBehaviorDialect>();
    let mut b = test.function_builder();
    let one = b.i32(1, SPAN);
    let sum = b.add(param, one, SPAN).unwrap();
    assert_eq!(fold_value(sum, None), None);

    // A constant of another type, handed to `fold_with`.
    let one_attr = one.borrow().get_defining_op().unwrap().borrow().get_attribute("value");
    let wide = Some(context.int_attr(Int::from(1i64), context.i64()).as_attribute_ref());
    let sum = b.add(one, one, SPAN).unwrap();
    assert_eq!(fold_value(sum, Some(&[one_attr, wide])), None);
    assert_eq!(
        fold_value(sum, Some(&[one_attr, one_attr])),
        Some((Int::from(2i32), context.i32()))
    );
    assert_eq!(fold_value(sum, Some(&[one_attr, None])), None);

    // A `ub.poison` operand, through `fold` and through `fold_with`.
    let poison = runic_dialect_ub::UndefinedBehaviorOpBuilder::poison(&mut b, context.i32(), SPAN);
    let sum = b.add(one, poison, SPAN).unwrap();
    assert_eq!(fold_value(sum, None), None);
    let poison_attr = poison.borrow().get_defining_op().unwrap().borrow().get_attribute("value");
    assert!(poison_attr.is_some());
    assert_eq!(fold_value(sum, Some(&[one_attr, poison_attr])), None);
    let negated = b.neg(poison, SPAN).unwrap();
    assert_eq!(fold_value(negated, None), None);
}
