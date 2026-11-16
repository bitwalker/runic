//! The ranges `arith.add` and `arith.cmpi` infer, and `IntegerValueRange::refined_by`, against
//! concrete values.
//!
//! * For random constants inside random operand ranges, what the real fold of `arith.add` (every
//!   overflow mode) and `arith.cmpi` (every predicate) computes lies inside the range the
//!   operation infers through `InferIntRangeInterface`, as the analysis reaches it.
//! * A value inside an operand's range that passes a check at the same width lies inside the
//!   refined range; across widths the refined range is the checked one, whatever the operand's
//!   range and however a cast between the widths is realized.
//!
//! See [super::random] for the seeds; each failure message names its seed.

use runic_ir::{ConstantIntRange, IntegerValueRange, Overflow, traits::InferIntRangeInterface};

use super::{
    model::mask,
    random::{Rng, seeds},
    *,
};

const MODES: [Overflow; 5] = [
    Overflow::Wrapping,
    Overflow::Checked(Signedness::Signed),
    Overflow::Checked(Signedness::Unsigned),
    Overflow::Unchecked(Signedness::Signed),
    Overflow::Unchecked(Signedness::Unsigned),
];

const PREDICATES: [CmpPredicate; 10] = [
    CmpPredicate::Eq,
    CmpPredicate::Ne,
    CmpPredicate::Slt,
    CmpPredicate::Sle,
    CmpPredicate::Sgt,
    CmpPredicate::Sge,
    CmpPredicate::Ult,
    CmpPredicate::Ule,
    CmpPredicate::Ugt,
    CmpPredicate::Uge,
];

/// The widths the tests run at: the cached types, the limits, and widths around the limb sizes.
const WIDTHS: [u32; 12] = [1, 2, 7, 8, 16, 31, 32, 33, 64, 65, 127, 128];

/// The number of random operand pairs per width and seed in the execution tests.
const CASES_PER_WIDTH: usize = 1200;

/// The number of random refinements per seed.
const REFINEMENTS: usize = 200_000;

/// A width from [WIDTHS].
fn width(rng: &mut Rng) -> u32 {
    WIDTHS[rng.below(WIDTHS.len() as u64) as usize]
}

/// The `w`-bit pattern `bits` read as a signed number.
fn signed(w: u32, bits: u128) -> i128 {
    Int::from_bits(w, bits).signed()
}

/// The `w`-bit pattern of `value`.
fn unsigned_of(w: u32, value: i128) -> u128 {
    (value as u128) & mask(w)
}

/// A value of `w` bits, weighted toward the edges, sometimes small.
fn value(rng: &mut Rng, w: u32) -> u128 {
    match rng.below(4) {
        0 => rng.below(17) as u128 & mask(w),
        1 => unsigned_of(w, -(rng.below(17) as i128)),
        _ => rng.edge(w).bits,
    }
}

/// A range of two bounds read as unsigned or as signed numbers.
fn simple_range(rng: &mut Rng, w: u32) -> ConstantIntRange {
    let (a, b) = (value(rng, w), value(rng, w));
    match rng.below(5) {
        0 => ConstantIntRange::from_unsigned(w as u16, 0, mask(w)),
        1 => ConstantIntRange::from_unsigned(w as u16, a, a),
        2 => ConstantIntRange::from_unsigned(w as u16, a.min(b), a.max(b)),
        3 => {
            let (a, b) = (signed(w, a), signed(w, b));
            ConstantIntRange::from_signed(w as u16, a.min(b), a.max(b))
        }
        _ => {
            // A short run from `a`
            let len = rng.below(9) as u128;
            ConstantIntRange::from_unsigned(w as u16, a, a.saturating_add(len).min(mask(w)))
        }
    }
}

/// A range as the lattice can hold one: a simple one, or a union, intersection or widening of
/// two, whose unsigned and signed bounds need not describe one interval.
fn range(rng: &mut Rng, w: u32) -> ConstantIntRange {
    let a = simple_range(rng, w);
    let b = simple_range(rng, w);
    match rng.below(6) {
        0 => a.union(&b),
        1 => a.intersection(&b).unwrap_or(a),
        2 => a.widen(&b),
        _ => a,
    }
}

/// True if the `w`-bit pattern `bits` satisfies both the unsigned and the signed bounds of
/// `range`, which is `w` bits wide.
fn contains(range: &ConstantIntRange, bits: u128) -> bool {
    let w = u32::from(range.width());
    range.umin() <= bits
        && bits <= range.umax()
        && range.smin() <= signed(w, bits)
        && signed(w, bits) <= range.smax()
}

/// A value inside `range`, if one is found.
fn value_in(rng: &mut Rng, range: &ConstantIntRange) -> Option<u128> {
    let w = u32::from(range.width());
    for _ in 0..12 {
        let span = range.umax() - range.umin();
        let candidate = match rng.below(8) {
            0 => range.umin(),
            1 => range.umax(),
            2 => unsigned_of(w, range.smin()),
            3 => unsigned_of(w, range.smax()),
            4 => range.umin() + (rng.below(5) as u128).min(span),
            5 => range.umax() - (rng.below(5) as u128).min(span),
            6 => value(rng, w),
            _ => {
                if span == u128::MAX {
                    rng.bits()
                } else {
                    range.umin() + rng.bits() % (span + 1)
                }
            }
        };
        if contains(range, candidate) {
            return Some(candidate);
        }
    }
    None
}

/// A function of two parameters of `iW` holding one `arith.add` per overflow mode and one
/// `arith.cmpi` per predicate, on the parameters.
struct Ops {
    test: Test,
    ty: Type,
    adds: Vec<(Overflow, ValueRef)>,
    cmps: Vec<(CmpPredicate, ValueRef)>,
}

impl Ops {
    fn new(w: u32) -> Self {
        let mut test = test_fn(move |cx| vec![cx.int_type(w), cx.int_type(w)], |_| vec![]);
        let ty = test.context().int_type(w);
        let [lhs, rhs] = params(&test)[..] else {
            unreachable!()
        };
        let mut b = test.function_builder();
        let adds = MODES
            .into_iter()
            .map(|mode| {
                let sum = match mode {
                    Overflow::Wrapping => b.add(lhs, rhs, SPAN),
                    Overflow::Checked(sign) => b.add_checked(lhs, rhs, sign, SPAN),
                    Overflow::Unchecked(sign) => b.add_unchecked(lhs, rhs, sign, SPAN),
                };
                (mode, sum.unwrap())
            })
            .collect();
        let cmps = PREDICATES
            .into_iter()
            .map(|predicate| (predicate, b.cmpi(predicate, lhs, rhs, SPAN).unwrap()))
            .collect();
        Self {
            test,
            ty,
            adds,
            cmps,
        }
    }

    /// The range the operation defining `result` infers from operand ranges.
    fn infer(
        result: ValueRef,
        lhs: &ConstantIntRange,
        rhs: &ConstantIntRange,
    ) -> IntegerValueRange {
        let op = result.borrow().get_defining_op().unwrap();
        let op = op.borrow();
        let infer = op.as_trait::<dyn InferIntRangeInterface>().expect("infers ranges");
        let mut inferred = None;
        infer.infer_result_ranges(
            &[IntegerValueRange::new(*lhs), IntegerValueRange::new(*rhs)],
            &mut |value, range| {
                assert_eq!(value, result);
                assert!(inferred.replace(range).is_none(), "one range per result");
            },
        );
        inferred.expect("a range for the result")
    }

    /// What the operation defining `result` folds to on the constants `a` and `b`.
    fn fold(&self, result: ValueRef, a: u128, b: u128) -> Option<(Int, Type)> {
        let w = Int::width_for_type(self.ty).unwrap();
        let context = self.test.context_rc();
        let attrs = [a, b].map(|bits| {
            let attr: AttributeRef =
                context.int_attr(Int::from_bits(w, bits), self.ty).as_attribute_ref();
            Some(attr)
        });
        fold_value(result, Some(&attrs))
    }
}

#[test]
fn execution_folded_sums_lie_in_the_inferred_range() {
    let mut checked = 0usize;
    let mut declined = 0usize;
    let mut narrow = 0usize;
    for seed in seeds() {
        let mut rng = Rng(seed);
        for w in WIDTHS {
            let ops = Ops::new(w);
            let top = ConstantIntRange::max_for_type(ops.ty);
            for _ in 0..CASES_PER_WIDTH {
                let lhs = range(&mut rng, w);
                let rhs = range(&mut rng, w);
                let (Some(a), Some(b)) = (value_in(&mut rng, &lhs), value_in(&mut rng, &rhs))
                else {
                    continue;
                };
                for (mode, sum) in &ops.adds {
                    let inferred = Ops::infer(*sum, &lhs, &rhs).range().copied().unwrap();
                    assert_eq!(u32::from(inferred.width()), w, "seed {seed}");
                    narrow += usize::from(inferred != top);
                    match ops.fold(*sum, a, b) {
                        Some((folded, ty)) => {
                            assert_eq!(ty, ops.ty, "seed {seed}");
                            assert!(
                                contains(&inferred, folded.unsigned()),
                                "seed {seed}: add {mode} on i{w}: {a} in {lhs} + {b} in {rhs} \
                                 folds to {folded:?}, outside {inferred}"
                            );
                            checked += 1;
                        }
                        // A trap or an undefined result: no value to be inside the range
                        None => {
                            assert!(!matches!(mode, Overflow::Wrapping), "seed {seed}");
                            declined += 1;
                        }
                    }
                }
            }
        }
    }
    let runs = seeds().len();
    assert!(
        checked > 10_000 * runs && declined > 200 * runs && narrow > 2_000 * runs,
        "{checked} folded sums, {declined} did not fold, {narrow} narrower than the type"
    );
}

#[test]
fn execution_folded_comparisons_lie_in_the_inferred_range() {
    let mut checked = 0usize;
    let mut decided = 0usize;
    for seed in seeds() {
        let mut rng = Rng(seed);
        for w in WIDTHS {
            let ops = Ops::new(w);
            for _ in 0..CASES_PER_WIDTH {
                let lhs = range(&mut rng, w);
                let rhs = range(&mut rng, w);
                let (Some(a), Some(b)) = (value_in(&mut rng, &lhs), value_in(&mut rng, &rhs))
                else {
                    continue;
                };
                for (predicate, cmp) in &ops.cmps {
                    let inferred = Ops::infer(*cmp, &lhs, &rhs).range().copied().unwrap();
                    assert_eq!(inferred.width(), 1, "seed {seed}");
                    decided += usize::from(inferred.umin() == inferred.umax());
                    let (folded, _) = ops.fold(*cmp, a, b).expect("a comparison folds");
                    assert!(
                        contains(&inferred, folded.unsigned()),
                        "seed {seed}: cmpi {predicate} on i{w}: {a} in {lhs}, {b} in {rhs} folds \
                         to {folded:?}, outside {inferred}"
                    );
                    checked += 1;
                }
            }
        }
    }
    let runs = seeds().len();
    assert!(
        checked > 20_000 * runs && decided > 2_000 * runs,
        "{checked} folded comparisons, {decided} decided by the ranges"
    );
}

/// At one width: a value inside the operand's range that passes the check is inside the refined
/// range.
#[test]
fn refinement_at_one_width_keeps_every_value_that_passes() {
    let mut checked = 0usize;
    for seed in seeds() {
        let mut rng = Rng(seed);
        for _ in 0..REFINEMENTS {
            let w = width(&mut rng);
            let current = range(&mut rng, w);
            let check = range(&mut rng, w);
            let refined = IntegerValueRange::new(current).refined_by(&check);
            let refined = refined.range().copied().unwrap();
            let Some(v) = value_in(&mut rng, &current) else {
                continue;
            };
            if contains(&check, v) {
                assert!(
                    contains(&refined, v),
                    "seed {seed}: {v}: {current} refined by {check} is {refined}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 4_000 * seeds().len(), "{checked} values passed their check");
}

/// Across widths the refined range is the checked range, whatever the operand's: an unrealized
/// cast between the widths may keep the unsigned value, sign-extend it or truncate it, and in each
/// case every value that passes the check lies in the refined range.
#[test]
fn refinement_across_widths_assumes_nothing_about_the_cast() {
    let mut passed = 0usize;
    for seed in seeds() {
        let mut rng = Rng(seed);
        for _ in 0..REFINEMENTS {
            let (from, to) = (width(&mut rng), width(&mut rng));
            if from == to {
                continue;
            }
            let current = if rng.below(2) == 0 {
                ConstantIntRange::from_unsigned(from as u16, 0, mask(from))
            } else {
                range(&mut rng, from)
            };
            let check = range(&mut rng, to);
            let refined = IntegerValueRange::new(current).refined_by(&check);
            let refined = refined.range().copied().unwrap();
            assert_eq!(refined, check, "seed {seed}: {current} refined by {check}");
            let Some(v) = value_in(&mut rng, &current) else {
                continue;
            };
            let realizations = [
                // zero extension, or a truncation of a value that fits
                (to > from || v <= mask(to)).then_some(v),
                // sign extension
                (to > from).then(|| unsigned_of(to, signed(from, v))),
                // truncation of a value that need not fit
                (to < from).then(|| v & mask(to)),
            ];
            for cast in realizations.into_iter().flatten() {
                if contains(&check, cast) {
                    assert!(
                        contains(&refined, cast),
                        "seed {seed}: {v} as {cast}: {current} refined by {check} is {refined}"
                    );
                    passed += 1;
                }
            }
        }
    }
    assert!(passed > 1_000 * seeds().len(), "{passed} cast values passed their check");
}
