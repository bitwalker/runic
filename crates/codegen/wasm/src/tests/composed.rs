//! The differential test of the composed integer patterns (the overflow family, `min`, `max`,
//! `udivrem`, `sdivrem`, rotates, bit counts) and of `cf.assert`, declared as the integer tests
//! are ([super::int]): each operation is lowered at each width, run by the evaluator, and compared
//! with the `Int` operations `arith`'s folds call, every result (an overflow flag included)
//! checked for carried form.
//!
//! The overflow family is tried on the edge values, on a pseudo-random sample with a fixed seed,
//! and on pairs at the overflow boundary: for each width and signedness, operands whose exact sum,
//! difference or product is the largest (or the smallest) that fits, and the next one.

use alloc::{collections::BTreeSet, format, vec, vec::Vec};

use runic_ir::Int;

use super::{
    eval::{Outcome, Val, run},
    int::{Expect, Ty, amounts, check, check_tuples, edges, lower_text, product, types},
    *,
};

/// The pseudo-random numbers of the samples: SplitMix64 from a fixed seed.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// An `n`-bit value whose magnitude has a random number of bits, negated half the time, so
    /// that products of every size occur.
    fn value(&mut self, n: u32) -> Int {
        let length = (self.next() % u64::from(n + 1)) as u32;
        let bits = if length == 0 {
            0
        } else {
            u128::from(self.next()) & ((1u128 << length) - 1)
        };
        let value = Int::from_bits(n, bits);
        if self.next() & 1 == 1 {
            value.wrapping_neg()
        } else {
            value
        }
    }
}

/// The number of pseudo-random pairs per width.
const SAMPLES: usize = 300;

/// `value` as an `n`-bit integer, if it is in the range of `n` bits read as `signed` or not.
fn int(n: u32, value: i128, signed: bool) -> Option<Int> {
    if signed {
        Int::from_signed(n, value)
    } else {
        u128::try_from(value).ok().and_then(|value| Int::from_unsigned(n, value))
    }
}

/// The pairs around the overflow boundary at the width `n`: for each signedness, pairs whose
/// exact sum, difference or product is the largest that fits and the smallest that does not, at
/// the top and (signed) the bottom of the range.
fn boundary(n: u32) -> Vec<Vec<Int>> {
    let mut pairs: BTreeSet<(u128, u128)> = BTreeSet::new();
    let mut push = |a: Option<Int>, b: Option<Int>| {
        if let (Some(a), Some(b)) = (a, b) {
            pairs.insert((a.unsigned(), b.unsigned()));
            pairs.insert((b.unsigned(), a.unsigned()));
        }
    };
    let umax = (1i128 << n) - 1;
    let smax = (1i128 << (n - 1)) - 1;
    let smin = -(1i128 << (n - 1));
    let square = 1i128 << n.div_ceil(2);
    let factors = [1, 2, 3, 5, 7, 10, 255, square - 1, square, square + 1, smax / 3, smax];
    for k in [0i128, 1, 5] {
        // Sums and differences at the top of the unsigned range and at both ends of the signed one.
        push(int(n, umax - k, false), int(n, k, false));
        push(int(n, umax - k, false), int(n, k + 1, false));
        push(int(n, k, false), int(n, k + 1, false));
        push(int(n, smax - k, true), int(n, k, true));
        push(int(n, smax - k, true), int(n, k + 1, true));
        push(int(n, smin + k, true), int(n, -k, true));
        push(int(n, smin + k, true), int(n, -k - 1, true));
        push(int(n, smin + k, true), int(n, k + 1, true));
        push(int(n, smax - k, true), int(n, -k - 1, true));
    }
    for a in factors.into_iter().filter(|a| *a >= 1) {
        // Unsigned: `a * (max / a)` is the largest product that fits, `a * (max / a + 1)` does not.
        let b = umax / a;
        push(int(n, a, false), int(n, b, false));
        push(int(n, a, false), int(n, b + 1, false));
        // Signed, for `a` and `-a`: the quotients toward zero of the limits fit, and one step
        // away from zero does not.
        for a in [a, -a] {
            let toward_max = smax / a;
            let toward_min = smin / a;
            let away = |q: i128| if q < 0 { q - 1 } else { q + 1 };
            push(int(n, a, true), int(n, toward_max, true));
            push(int(n, a, true), int(n, away(toward_max), true));
            push(int(n, a, true), int(n, toward_min, true));
            push(int(n, a, true), int(n, away(toward_min), true));
        }
    }
    pairs
        .into_iter()
        .map(|(a, b)| vec![Int::from_bits(n, a), Int::from_bits(n, b)])
        .collect()
}

/// The operand pairs the overflow family is tried on at the width `n`: the edge values with each
/// other, [SAMPLES] pseudo-random pairs (a seed per width) and the [boundary] pairs.
fn overflow_pairs(n: u32) -> Vec<Vec<Int>> {
    let mut tuples = product(&[edges(n), edges(n)]);
    let mut random = Random(0x5eed_0000 + u64::from(n));
    for _ in 0..SAMPLES {
        tuples.push(vec![random.value(n), random.value(n)]);
    }
    tuples.extend(boundary(n));
    tuples
}

/// The operations of the overflow family, with the `Int` operation `arith`'s folds call for each
/// signedness: the wrapped result and whether the exact one does not fit.
type Overflowing = fn(Int, Int) -> (Int, bool);
const FAMILY: [(&str, Overflowing, Overflowing); 3] = [
    ("add", Int::overflowing_add_signed, Int::overflowing_add_unsigned),
    ("sub", Int::overflowing_sub_signed, Int::overflowing_sub_unsigned),
    ("mul", Int::overflowing_mul_signed, Int::overflowing_mul_unsigned),
];

#[test]
fn overflowing_operations_match_arith_in_flag_and_value() {
    let context = context();
    let flag = Ty::int(1);
    let mut count = 0;
    for (name, signed, unsigned) in FAMILY {
        for (sign, compute) in [("signed", signed), ("unsigned", unsigned)] {
            for ty in types() {
                let body = format!(
                    "arith.{name}_overflowing %a, %b <{{ signedness = #builtin.signedness<{sign}> \
                     }}>"
                );
                count += check_tuples(
                    &context,
                    &format!("{name}_overflowing {sign}"),
                    &[&ty, &ty],
                    &[&flag, &ty],
                    &body,
                    &overflow_pairs(ty.n),
                    |operands| {
                        let (value, overflowed) = compute(operands[0], operands[1]);
                        vec![Expect::Value(Int::from(overflowed)), Expect::Value(value)]
                    },
                );
            }
        }
    }
    assert!(count > 20_000, "{count}");
}

#[test]
fn checked_operations_trap_exactly_when_the_result_does_not_fit() {
    let context = context();
    let mut count = 0;
    let mut traps = 0;
    for (name, signed, unsigned) in FAMILY {
        for (sign, compute) in [("signed", signed), ("unsigned", unsigned)] {
            for ty in types() {
                let body = format!(
                    "arith.{name} %a, %b <{{ overflow = #builtin.overflow<checked {sign}> }}>"
                );
                let pairs = overflow_pairs(ty.n);
                traps += pairs.iter().filter(|pair| compute(pair[0], pair[1]).1).count();
                count += check_tuples(
                    &context,
                    &format!("{name} checked {sign}"),
                    &[&ty, &ty],
                    &[&ty],
                    &body,
                    &pairs,
                    |operands| match compute(operands[0], operands[1]) {
                        (_, true) => vec![Expect::Trap],
                        (value, false) => vec![Expect::Value(value)],
                    },
                );
            }
        }
    }
    assert!(count > 20_000 && traps > 5_000, "{count} evaluations, {traps} traps");
}

#[test]
fn min_and_max_match_arith() {
    let context = context();
    let mut count = 0;
    for (name, signed, unsigned) in [
        (
            "min",
            Int::min_signed as fn(Int, Int) -> Int,
            Int::min_unsigned as fn(Int, Int) -> Int,
        ),
        ("max", Int::max_signed, Int::max_unsigned),
    ] {
        for (sign, compute) in [("signed", signed), ("unsigned", unsigned)] {
            for ty in types() {
                count += check(
                    &context,
                    &format!("{name} {sign}"),
                    &[&ty, &ty],
                    &ty,
                    &format!(
                        "arith.{name} %a, %b <{{ signedness = #builtin.signedness<{sign}> }}>"
                    ),
                    &[edges(ty.n), edges(ty.n)],
                    |operands| Expect::Value(compute(operands[0], operands[1])),
                );
            }
        }
    }
    assert!(count > 2000, "{count}");
}

#[test]
fn division_with_remainder_matches_arith() {
    let context = context();
    let mut count = 0;
    for ty in types() {
        count += check_tuples(
            &context,
            "udivrem",
            &[&ty, &ty],
            &[&ty, &ty],
            "arith.udivrem %a, %b",
            &product(&[edges(ty.n), edges(ty.n)]),
            |operands| {
                let (a, b) = (operands[0], operands[1]);
                // A zero divisor traps in `arith` and in the lowering.
                match (a.div_unsigned(b), a.rem_unsigned(b)) {
                    (Some(quotient), Some(remainder)) => {
                        vec![Expect::Value(quotient), Expect::Value(remainder)]
                    }
                    _ => vec![Expect::Trap, Expect::Trap],
                }
            },
        );
        count += check_tuples(
            &context,
            "sdivrem",
            &[&ty, &ty],
            &[&ty, &ty],
            "arith.sdivrem %a, %b",
            &product(&[edges(ty.n), edges(ty.n)]),
            |operands| {
                let (a, b) = (operands[0], operands[1]);
                match (a.div_signed(b), a.rem_signed(b)) {
                    (Some(quotient), Some(remainder)) => {
                        vec![Expect::Value(quotient), Expect::Value(remainder)]
                    }
                    _ if b.is_zero() => vec![Expect::Trap, Expect::Trap],
                    // The minimum divided by -1 traps in `arith`. Below the carrier's width the
                    // sign-extended division does not trap, and gives the minimum (masked) and 0,
                    // which `arith` allows (a trap is never introduced, and may be removed); at the
                    // carrier's width Wasm's `div_s` traps as well.
                    _ if ty.n < ty.carrier() => {
                        vec![Expect::Value(a), Expect::Value(Int::zero(ty.n))]
                    }
                    _ => vec![Expect::Trap, Expect::Trap],
                }
            },
        );
    }
    assert!(count > 1000, "{count}");
}

#[test]
fn rotates_match_arith_for_every_amount() {
    let context = context();
    let mut count = 0;
    for (name, rotate) in [("rotl", Int::rotl as fn(Int, u32) -> Int), ("rotr", Int::rotr)] {
        for ty in types() {
            count += check(
                &context,
                name,
                &[&ty, &ty],
                &ty,
                &format!("arith.{name} %a, %b"),
                &[edges(ty.n), amounts(ty.n)],
                |operands| {
                    let (a, b) = (operands[0], operands[1]);
                    Expect::Value(rotate(a, b.rotate_amount_unsigned(a.width())))
                },
            );
        }
    }
    assert!(count > 5000, "{count}");
}

/// The values a bit count is tried on at the width `n`: the edge values, each power of two and
/// each power of two less 1.
fn count_values(n: u32) -> Vec<Int> {
    let mut bits: BTreeSet<u128> = edges(n).into_iter().map(Int::unsigned).collect();
    for k in 0..n {
        bits.insert(1u128 << k);
        bits.insert((1u128 << k) - 1);
        bits.insert(Int::all_ones(n).unsigned() ^ (1u128 << k));
    }
    bits.into_iter().map(|bits| Int::from_bits(n, bits)).collect()
}

#[test]
fn bit_counts_match_arith() {
    let context = context();
    let mut count = 0;
    for (name, counted) in [
        ("popcnt", Int::count_ones as fn(Int) -> u32),
        ("clz", Int::leading_zeros),
        ("ctz", Int::trailing_zeros),
        ("clo", Int::leading_ones),
        ("cto", Int::trailing_ones),
    ] {
        for ty in types() {
            count += check(
                &context,
                name,
                &[&ty],
                &ty,
                &format!("arith.{name} %a"),
                &[count_values(ty.n)],
                |operands| {
                    let a = operands[0];
                    let counted = Int::from_unsigned(a.width(), u128::from(counted(a)))
                        .expect("a count fits its width");
                    Expect::Value(counted)
                },
            );
        }
    }
    assert!(count > 3000, "{count}");
}

#[test]
fn an_assertion_traps_exactly_when_its_condition_is_zero() {
    let context = context();
    let lowered = lower_text(&context, &["i1"], &[], "cf.assert %a, \"the message\"")
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    assert_eq!(run(lowered.function, &[Val::i32(1)]), Outcome::Values(vec![]));
    assert_eq!(run(lowered.function, &[Val::i32(0)]), Outcome::Trap);
}

#[test]
fn an_assertion_is_lowered_without_its_message() {
    let (_context, module) = parse_module(
        "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, ptr = \
         32> } {\n    builtin.function public extern(\"wasm\") @f(%c: i1) {\n        cf.assert \
         %c, \"the message\";\n        builtin.ret;\n    };\n};",
    );
    crate::convert::convert_module(module).unwrap_or_else(|report| panic!("{}", describe(&report)));
    let text = text_of(module);
    assert!(!text.contains("the message"), "{text}");
    let lines = lines(&text);
    let start = lines.iter().position(|line| line.starts_with("wasm.if")).expect("a wasm.if");
    assert_eq!(
        lines[start..start + 5],
        [
            "wasm.if % then {",
            "wasm.end : ();",
            "} else {",
            "wasm.unreachable;",
            "} : (i32) -> ();"
        ]
    );
}
