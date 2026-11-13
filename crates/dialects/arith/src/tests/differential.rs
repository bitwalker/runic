//! The randomized differential test of every fold against the independent [model].
//!
//! For every foldable operation and every variant of it (overflow mode, signedness, predicate):
//! a random operand type (a width in `1..=128`, or `isize`), random operands weighted toward edge
//! values, the operation built on `arith.constant`s, folded through `Foldable::fold` and
//! `Foldable::fold_with`, and compared with the model in value, in type, and in whether it folds
//! at all. The same cases also go through the greedy driver.
//!
//! For `isize` the model computes at 64 bits and at 32 bits (the operands truncated) and expects a
//! fold exactly when neither is a trap or undefined and the 32-bit result is the 64-bit one
//! truncated (for a fixed-width result: when the two are equal).
//!
//! See [super::random] for the seeds and the number of cases.

use alloc::collections::BTreeMap;

use super::{
    model::{self, Arith, M, Sign},
    random::{Rng, cases_per_variant, seeds},
    *,
};

/// What a result of an operation on operands of type `T` is.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Res {
    /// A value of type `T`.
    Operand,
    /// An `i1`.
    Flag,
}

type Build = Rc<dyn Fn(Builder<'_, '_>, &[ValueRef]) -> Vec<ValueRef>>;
/// The results on operands of one width, the width of the operands given; `None` for a trap or an
/// undefined result.
type Model = Rc<dyn Fn(&[M]) -> Option<Vec<M>>>;

/// An operation whose operands have one type.
struct Variant {
    name: String,
    arity: usize,
    results: Vec<Res>,
    build: Build,
    model: Model,
}

/// One random case: an operation on constants, and what the model says it folds to.
struct Plan {
    /// The operation and its mode, without the types: the key of the per-variant counts.
    key: String,
    name: String,
    /// The types of the operation's results.
    result_tys: Vec<Type>,
    operand_ty: Type,
    values: Vec<M>,
    build: Build,
    /// The value and type of each result, or `None` if the operation must not fold.
    expected: Option<Vec<(M, Type)>>,
}

fn sign_of(signedness: Signedness) -> Sign {
    match signedness {
        Signedness::Signed => Sign::Signed,
        Signedness::Unsigned => Sign::Unsigned,
    }
}

fn variant(
    name: impl Into<String>,
    arity: usize,
    results: &[Res],
    build: impl Fn(Builder<'_, '_>, &[ValueRef]) -> Vec<ValueRef> + 'static,
    model: impl Fn(&[M]) -> Option<Vec<M>> + 'static,
) -> Variant {
    Variant {
        name: name.into(),
        arity,
        results: results.to_vec(),
        build: Rc::new(build),
        model: Rc::new(model),
    }
}

/// Every same-type operation, in every mode.
fn variants() -> Vec<Variant> {
    use Res::*;
    let mut all = Vec::new();
    let signs = [Signedness::Signed, Signedness::Unsigned];

    // add, sub, mul in the five overflow modes.
    for (op, name) in [(Arith::Add, "add"), (Arith::Sub, "sub"), (Arith::Mul, "mul")] {
        all.push(variant(
            format!("{name} wrapping"),
            2,
            &[Operand],
            move |b, v| {
                vec![
                    match op {
                        Arith::Add => b.add(v[0], v[1], SPAN),
                        Arith::Sub => b.sub(v[0], v[1], SPAN),
                        Arith::Mul => b.mul(v[0], v[1], SPAN),
                    }
                    .unwrap(),
                ]
            },
            move |v| Some(vec![model::wrapping(op, v[0], v[1])]),
        ));
        for signedness in signs {
            for checked in [true, false] {
                let mode = if checked { "checked" } else { "unchecked" };
                all.push(variant(
                    format!("{name} {mode} {signedness}"),
                    2,
                    &[Operand],
                    move |b, v| {
                        vec![
                            match (op, checked) {
                                (Arith::Add, true) => b.add_checked(v[0], v[1], signedness, SPAN),
                                (Arith::Add, false) => {
                                    b.add_unchecked(v[0], v[1], signedness, SPAN)
                                }
                                (Arith::Sub, true) => b.sub_checked(v[0], v[1], signedness, SPAN),
                                (Arith::Sub, false) => {
                                    b.sub_unchecked(v[0], v[1], signedness, SPAN)
                                }
                                (Arith::Mul, true) => b.mul_checked(v[0], v[1], signedness, SPAN),
                                (Arith::Mul, false) => {
                                    b.mul_unchecked(v[0], v[1], signedness, SPAN)
                                }
                            }
                            .unwrap(),
                        ]
                    },
                    move |v| {
                        let (value, overflow) =
                            model::overflowing(op, v[0], v[1], sign_of(signedness));
                        (!overflow).then(|| vec![value])
                    },
                ));
            }
            all.push(variant(
                format!("{name}_overflowing {signedness}"),
                2,
                &[Flag, Operand],
                move |b, v| {
                    let (flag, value) = match op {
                        Arith::Add => b.add_overflowing(v[0], v[1], signedness, SPAN),
                        Arith::Sub => b.sub_overflowing(v[0], v[1], signedness, SPAN),
                        Arith::Mul => b.mul_overflowing(v[0], v[1], signedness, SPAN),
                    }
                    .unwrap();
                    vec![flag, value]
                },
                move |v| {
                    let (value, overflow) = model::overflowing(op, v[0], v[1], sign_of(signedness));
                    Some(vec![M::flag(overflow), value])
                },
            ));
        }
    }

    // The division family. `sdiv MIN, -1` and `sdivrem MIN, -1` trap; `srem MIN, -1` is `0`.
    all.push(variant(
        "udiv",
        2,
        &[Operand],
        |b, v| vec![b.udiv(v[0], v[1], SPAN).unwrap()],
        |v| model::udivrem(v[0], v[1]).map(|(q, _)| vec![q]),
    ));
    all.push(variant(
        "urem",
        2,
        &[Operand],
        |b, v| vec![b.urem(v[0], v[1], SPAN).unwrap()],
        |v| model::udivrem(v[0], v[1]).map(|(_, r)| vec![r]),
    ));
    all.push(variant(
        "udivrem",
        2,
        &[Operand, Operand],
        |b, v| {
            let (q, r) = b.udivrem(v[0], v[1], SPAN).unwrap();
            vec![q, r]
        },
        |v| model::udivrem(v[0], v[1]).map(|(q, r)| vec![q, r]),
    ));
    all.push(variant(
        "sdiv",
        2,
        &[Operand],
        |b, v| vec![b.sdiv(v[0], v[1], SPAN).unwrap()],
        |v| model::sdivrem(v[0], v[1])?.0.map(|q| vec![q]),
    ));
    all.push(variant(
        "srem",
        2,
        &[Operand],
        |b, v| vec![b.srem(v[0], v[1], SPAN).unwrap()],
        |v| model::sdivrem(v[0], v[1]).map(|(_, r)| vec![r]),
    ));
    all.push(variant(
        "sdivrem",
        2,
        &[Operand, Operand],
        |b, v| {
            let (q, r) = b.sdivrem(v[0], v[1], SPAN).unwrap();
            vec![q, r]
        },
        |v| {
            let (q, r) = model::sdivrem(v[0], v[1])?;
            Some(vec![q?, r])
        },
    ));

    // Bitwise.
    all.push(variant(
        "band",
        2,
        &[Operand],
        |b, v| vec![b.band(v[0], v[1], SPAN).unwrap()],
        |v| Some(vec![model::bitwise(v[0], v[1], |x, y| x & y)]),
    ));
    all.push(variant(
        "bor",
        2,
        &[Operand],
        |b, v| vec![b.bor(v[0], v[1], SPAN).unwrap()],
        |v| Some(vec![model::bitwise(v[0], v[1], |x, y| x | y)]),
    ));
    all.push(variant(
        "bxor",
        2,
        &[Operand],
        |b, v| vec![b.bxor(v[0], v[1], SPAN).unwrap()],
        |v| Some(vec![model::bitwise(v[0], v[1], |x, y| x ^ y)]),
    ));

    // Shifts and rotates.
    all.push(variant(
        "shl",
        2,
        &[Operand],
        |b, v| vec![b.shl(v[0], v[1], SPAN).unwrap()],
        |v| model::shl(v[0], v[1]).map(|r| vec![r]),
    ));
    all.push(variant(
        "shr",
        2,
        &[Operand],
        |b, v| vec![b.shr(v[0], v[1], SPAN).unwrap()],
        |v| model::lshr(v[0], v[1]).map(|r| vec![r]),
    ));
    all.push(variant(
        "ashr",
        2,
        &[Operand],
        |b, v| vec![b.ashr(v[0], v[1], SPAN).unwrap()],
        |v| model::ashr(v[0], v[1]).map(|r| vec![r]),
    ));
    all.push(variant(
        "rotl",
        2,
        &[Operand],
        |b, v| vec![b.rotl(v[0], v[1], SPAN).unwrap()],
        |v| Some(vec![model::rotl(v[0], v[1])]),
    ));
    all.push(variant(
        "rotr",
        2,
        &[Operand],
        |b, v| vec![b.rotr(v[0], v[1], SPAN).unwrap()],
        |v| Some(vec![model::rotr(v[0], v[1])]),
    ));

    // Comparisons.
    for predicate in CmpPredicate::ALL {
        all.push(variant(
            format!("cmpi {predicate}"),
            2,
            &[Flag],
            move |b, v| vec![b.cmpi(predicate, v[0], v[1], SPAN).unwrap()],
            move |v| Some(vec![M::flag(model::compare(predicate.as_str(), v[0], v[1]))]),
        ));
    }

    // Min and max.
    for signedness in signs {
        all.push(variant(
            format!("min {signedness}"),
            2,
            &[Operand],
            move |b, v| vec![b.min(v[0], v[1], signedness, SPAN).unwrap()],
            move |v| Some(vec![model::min(v[0], v[1], sign_of(signedness))]),
        ));
        all.push(variant(
            format!("max {signedness}"),
            2,
            &[Operand],
            move |b, v| vec![b.max(v[0], v[1], signedness, SPAN).unwrap()],
            move |v| Some(vec![model::max(v[0], v[1], sign_of(signedness))]),
        ));
    }

    // Unary.
    all.push(variant(
        "neg",
        1,
        &[Operand],
        |b, v| vec![b.neg(v[0], SPAN).unwrap()],
        |v| Some(vec![model::neg(v[0])]),
    ));
    all.push(variant(
        "bnot",
        1,
        &[Operand],
        |b, v| vec![b.bnot(v[0], SPAN).unwrap()],
        |v| Some(vec![model::bnot(v[0])]),
    ));
    all.push(variant(
        "popcnt",
        1,
        &[Operand],
        |b, v| vec![b.popcnt(v[0], SPAN).unwrap()],
        |v| Some(vec![model::popcnt(v[0])]),
    ));
    all.push(variant(
        "clz",
        1,
        &[Operand],
        |b, v| vec![b.clz(v[0], SPAN).unwrap()],
        |v| Some(vec![model::run(v[0], false, true)]),
    ));
    all.push(variant(
        "ctz",
        1,
        &[Operand],
        |b, v| vec![b.ctz(v[0], SPAN).unwrap()],
        |v| Some(vec![model::run(v[0], false, false)]),
    ));
    all.push(variant(
        "clo",
        1,
        &[Operand],
        |b, v| vec![b.clo(v[0], SPAN).unwrap()],
        |v| Some(vec![model::run(v[0], true, true)]),
    ));
    all.push(variant(
        "cto",
        1,
        &[Operand],
        |b, v| vec![b.cto(v[0], SPAN).unwrap()],
        |v| Some(vec![model::run(v[0], true, false)]),
    ));
    all
}

/// What the model says `variant` folds to on `values`, of `isize` if `is_isize`.
fn expect(variant: &Variant, is_isize: bool, values: &[M]) -> Option<Vec<M>> {
    let wide = (variant.model)(values);
    if !is_isize {
        return wide;
    }
    let narrow_values: Vec<M> = values.iter().map(|value| value.trunc(32)).collect();
    let narrow = (variant.model)(&narrow_values);
    let (wide, narrow) = (wide?, narrow?);
    for ((wide, narrow), kind) in wide.iter().zip(&narrow).zip(&variant.results) {
        let agree = match kind {
            Res::Operand => wide.trunc(32) == *narrow,
            Res::Flag => wide == narrow,
        };
        if !agree {
            return None;
        }
    }
    Some(wide)
}

/// A random case of `variant`: a fixed width three times in four, `isize` otherwise.
fn plan_same_type(variant: &Variant, rng: &mut Rng, context: &Rc<Context>) -> Plan {
    let is_isize = rng.below(4) == 0;
    let (ty, width) = if is_isize {
        (context.isize(), 64)
    } else {
        let width = rng.width();
        (context.int_type(width), width)
    };
    let mut values: Vec<M> = (0..variant.arity)
        .map(|_| {
            if is_isize {
                rng.isize_edge()
            } else {
                rng.edge(width)
            }
        })
        .collect();
    // Equal operands are an edge of their own.
    if variant.arity == 2 && rng.below(10) == 0 {
        values[1] = values[0];
    }
    let i1 = context.i1();
    let result_tys: Vec<Type> = variant
        .results
        .iter()
        .map(|kind| if *kind == Res::Flag { i1 } else { ty })
        .collect();
    let expected = expect(variant, is_isize, &values)
        .map(|results| results.into_iter().zip(result_tys.iter().copied()).collect());
    Plan {
        key: variant.name.clone(),
        name: format!("{} : {ty}", variant.name),
        result_tys,
        operand_ty: ty,
        values,
        build: variant.build.clone(),
        expected,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Cast {
    Trunc,
    Zext,
    Sext,
    SextOrTrunc,
    ZextOrTrunc,
}

const CASTS: [Cast; 5] =
    [Cast::Trunc, Cast::Zext, Cast::Sext, Cast::SextOrTrunc, Cast::ZextOrTrunc];

fn build_cast(cast: Cast, to: Type) -> Build {
    Rc::new(move |b, v| {
        vec![
            match cast {
                Cast::Trunc => b.trunc(v[0], to, SPAN),
                Cast::Zext => b.zext(v[0], to, SPAN),
                Cast::Sext => b.sext(v[0], to, SPAN),
                Cast::SextOrTrunc => b.sext_or_trunc(v[0], to, SPAN),
                Cast::ZextOrTrunc => b.zext_or_trunc(v[0], to, SPAN),
            }
            .unwrap(),
        ]
    })
}

/// A random valid case of `cast`.
fn plan_cast(cast: Cast, rng: &mut Rng, context: &Rc<Context>) -> Plan {
    let resize = |value: M, width: u32| match cast {
        Cast::Trunc => value.trunc(width),
        Cast::Zext => value.zext(width),
        Cast::Sext => value.sext(width),
        Cast::SextOrTrunc => value.sext_or_trunc(width),
        Cast::ZextOrTrunc => value.zext_or_trunc(width),
    };
    match cast {
        Cast::Trunc | Cast::Zext | Cast::Sext => {
            // Two distinct widths, in the order the cast needs.
            let (a, b) = loop {
                let (a, b) = (rng.width(), rng.width());
                if a != b {
                    break (a.min(b), a.max(b));
                }
            };
            let (from, to) = if cast == Cast::Trunc { (b, a) } else { (a, b) };
            let value = rng.edge(from);
            let to_ty = context.int_type(to);
            Plan {
                key: format!("{cast:?}"),
                name: format!("{cast:?} i{from} -> i{to}"),
                result_tys: vec![to_ty],
                operand_ty: context.int_type(from),
                values: vec![value],
                build: build_cast(cast, to_ty),
                expected: Some(vec![(resize(value, to), to_ty)]),
            }
        }
        Cast::SextOrTrunc | Cast::ZextOrTrunc => {
            let fixed = rng.width();
            let fixed_ty = context.int_type(fixed);
            if rng.below(2) == 0 {
                // From `isize`: the result is `fixed` bits on both targets, and they must agree.
                let value = rng.isize_edge();
                let (wide, narrow) = (resize(value, fixed), resize(value.trunc(32), fixed));
                Plan {
                    key: format!("{cast:?} from isize"),
                    name: format!("{cast:?} isize -> i{fixed}"),
                    result_tys: vec![fixed_ty],
                    operand_ty: context.isize(),
                    values: vec![value],
                    build: build_cast(cast, fixed_ty),
                    expected: (wide == narrow).then(|| vec![(wide, fixed_ty)]),
                }
            } else {
                // To `isize`: a 64-bit target has the operand resized to 64 bits, a 32-bit one to
                // 32. The constant is the former, and it folds if its low 32 bits are the latter.
                let value = rng.edge(fixed);
                let (wide, narrow) = (resize(value, 64), resize(value, 32));
                Plan {
                    key: format!("{cast:?} to isize"),
                    name: format!("{cast:?} i{fixed} -> isize"),
                    result_tys: vec![context.isize()],
                    operand_ty: fixed_ty,
                    values: vec![value],
                    build: build_cast(cast, context.isize()),
                    expected: (wide.trunc(32) == narrow).then(|| vec![(wide, context.isize())]),
                }
            }
        }
    }
}

/// What a batch of cases found.
#[derive(Default)]
struct Tally {
    cases: usize,
    folded: usize,
    declined: usize,
    /// Per variant name: how many cases folded, and how many were declined.
    by_variant: BTreeMap<String, (usize, usize)>,
    failures: Vec<String>,
}

impl Tally {
    fn fail(&mut self, plan: &Plan, what: String) {
        if self.failures.len() < 40 {
            self.failures.push(format!("{} on {:x?}: {what}", plan.name, plan.values));
        }
    }

    fn finish(self, what: &str) {
        std::println!(
            "{what}: {} cases, {} folded, {} declined, {} variants",
            self.cases,
            self.folded,
            self.declined,
            self.by_variant.len()
        );
        assert!(
            self.failures.is_empty(),
            "{what}: {} mismatches with the model:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

fn show(results: &Option<Vec<(M, Type)>>) -> String {
    match results {
        None => "no fold".into(),
        Some(results) => results
            .iter()
            .map(|(value, ty)| format!("{:#x} : {ty} ({} bits)", value.bits, value.w))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// Builds `plan` in `test` and checks `fold` and `fold_with` against its expectation.
fn check_fold(test: &mut Test, plan: &Plan, tally: &mut Tally) {
    let (result, attrs) = {
        let mut b = test.function_builder();
        let operands: Vec<ValueRef> = plan
            .values
            .iter()
            .map(|value| b.int(value.to_int(), plan.operand_ty, SPAN))
            .collect();
        let results = (plan.build)(&mut b, &operands);
        let attrs: Vec<_> = operands
            .iter()
            .map(|operand| {
                let constant = operand.borrow().get_defining_op().unwrap();
                constant.borrow().get_attribute("value")
            })
            .collect();
        (results[0], attrs)
    };
    let op = result.borrow().get_defining_op().unwrap();
    tally.cases += 1;
    if let Err(err) = op.borrow().verify() {
        tally.fail(plan, format!("the operation does not verify: {err}"));
        return;
    }
    // The types the operation itself declares for its results.
    let declared: Vec<Type> =
        op.borrow().results().iter().map(|result| result.borrow().ty()).collect();
    if declared != plan.result_tys {
        tally.fail(plan, format!("result types {declared:?}, expected {:?}", plan.result_tys));
    }
    let entry = tally.by_variant.entry(plan.key.clone()).or_default();
    if plan.expected.is_some() {
        entry.0 += 1;
        tally.folded += 1;
    } else {
        entry.1 += 1;
        tally.declined += 1;
    }
    for (how, operands) in [("fold", None), ("fold_with", Some(&attrs[..]))] {
        let got = fold_values(result, operands).map(|folded| {
            folded.into_iter().map(|(value, ty)| (M::of_int(value), ty)).collect::<Vec<_>>()
        });
        if got != plan.expected {
            tally.fail(
                plan,
                format!("{how} gave {}, the model {}", show(&got), show(&plan.expected)),
            );
        }
    }
    // The operand constants are as they were: a fold creates attributes, it edits none.
    for (attr, value) in attrs.iter().zip(&plan.values) {
        let attr = attr.unwrap();
        let attr = attr.borrow();
        let int = attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap();
        if M::of_int(int.as_int()) != *value || int.ty() != plan.operand_ty {
            tally.fail(plan, "an operand constant changed".into());
        }
    }
}

/// Runs `cases` random plans of each variant and cast through `check`.
fn run(seed: u64, cases: usize, mut check: impl FnMut(&Rc<Context>, &mut Test, &Plan)) {
    let mut rng = Rng(seed);
    let variants = variants();
    let plans: usize = variants.len() + CASTS.len();
    for index in 0..plans {
        let mut done = 0;
        while done < cases {
            // A fresh context every so often keeps the arena small.
            let mut test = test_with(&[]);
            let context = test.context_rc();
            for _ in 0..64.min(cases - done) {
                let plan = if index < variants.len() {
                    plan_same_type(&variants[index], &mut rng, &context)
                } else {
                    plan_cast(CASTS[index - variants.len()], &mut rng, &context)
                };
                check(&context, &mut test, &plan);
                done += 1;
            }
        }
    }
}

#[test]
fn every_fold_agrees_with_the_model_on_random_operands() {
    let mut tally = Tally::default();
    for seed in seeds() {
        run(seed, cases_per_variant(), |_, test, plan| check_fold(test, plan, &mut tally));
    }
    // Every variant folded somewhere. That the cases which must not fold are reached is checked by
    // `the_generator_reaches_the_cases_that_must_not_fold`.
    let never_folded: Vec<_> = tally
        .by_variant
        .iter()
        .filter(|(_, (folded, _))| *folded == 0)
        .map(|(name, _)| name.clone())
        .collect();
    assert!(never_folded.is_empty(), "never folded: {never_folded:?}");
    tally.finish("fold and fold_with");
}

/// The same plans through the greedy driver: a function that returns the operation's results is
/// canonicalized. A plan that folds leaves only constants of the expected values and types; a plan
/// that does not fold leaves the function as it was.
#[test]
fn every_fold_agrees_with_the_model_through_the_greedy_driver() {
    let mut tally = Tally::default();
    let cases = (cases_per_variant() / 8).max(8);
    for seed in seeds() {
        let mut rng = Rng(seed ^ 0xd41e);
        let variants = variants();
        for index in 0..variants.len() + CASTS.len() {
            for _ in 0..cases {
                let mut test = Test::default();
                test.context().get_or_register_dialect::<ArithDialect>();
                let context = test.context_rc();
                let plan = if index < variants.len() {
                    plan_same_type(&variants[index], &mut rng, &context)
                } else {
                    plan_cast(CASTS[index - variants.len()], &mut rng, &context)
                };
                check_driver(&mut test, &plan, &mut tally);
            }
        }
    }
    tally.finish("greedy driver");
}

fn check_driver(test: &mut Test, plan: &Plan, tally: &mut Tally) {
    tally.cases += 1;
    test.with_function("f", &[], &plan.result_tys);
    let results = {
        let mut b = test.function_builder();
        let operands: Vec<ValueRef> = plan
            .values
            .iter()
            .map(|value| b.int(value.to_int(), plan.operand_ty, SPAN))
            .collect();
        let results = (plan.build)(&mut b, &operands);
        b.ret(results.iter().copied(), SPAN).unwrap();
        results
    };
    let name = results[0].borrow().get_defining_op().unwrap().borrow().name();
    canonicalize(test);
    if let Err(err) = test.function().borrow().as_operation().verify() {
        tally.fail(plan, format!("invalid after canonicalization: {err}"));
        return;
    }
    let after = text_of(test);
    let ops = ops_of(test.function());
    match &plan.expected {
        None => {
            tally.declined += 1;
            tally.by_variant.entry(plan.key.clone()).or_default().1 += 1;
            // The driver may merge equal constants; the operation itself must survive, on
            // constants of the same values, and still be what is returned.
            let survivors: Vec<_> =
                ops.iter().filter(|op| op.borrow().name() == name).copied().collect();
            let [survivor] = survivors[..] else {
                tally.fail(plan, format!("must not fold, but became:\n{after}"));
                return;
            };
            let survivor = survivor.borrow();
            let operands: Vec<Option<M>> = survivor
                .operands()
                .all()
                .iter()
                .map(|operand| {
                    let def = operand.borrow().as_value_ref().borrow().get_defining_op()?;
                    let def = def.borrow();
                    let constant = def.downcast_ref::<Constant>()?;
                    (constant.result().ty() == plan.operand_ty)
                        .then(|| M::of_int(*constant.get_value()))
                })
                .collect();
            let want: Vec<Option<M>> = plan.values.iter().copied().map(Some).collect();
            let ret = ops.last().unwrap().borrow();
            let returned: Vec<ValueRef> =
                ret.operands().all().iter().map(|o| o.borrow().as_value_ref()).collect();
            let results: Vec<ValueRef> =
                survivor.results().iter().map(|r| r.borrow().as_value_ref()).collect();
            if operands != want || returned != results || ops.len() > plan.values.len() + 2 {
                tally.fail(plan, format!("must not fold, but became:\n{after}"));
            }
        }
        Some(expected) => {
            tally.folded += 1;
            tally.by_variant.entry(plan.key.clone()).or_default().0 += 1;
            let Some((ret, constants)) = ops.split_last() else {
                tally.fail(plan, "empty function".into());
                return;
            };
            if constants.iter().any(|op| op.borrow().name() == name) {
                tally.fail(plan, format!("must fold, but the operation survives:\n{after}"));
                return;
            }
            let ret = ret.borrow();
            let returned: Vec<ValueRef> =
                ret.operands().all().iter().map(|o| o.borrow().as_value_ref()).collect();
            if returned.len() != expected.len() {
                tally.fail(plan, format!("returns {} values:\n{after}", returned.len()));
                return;
            }
            for (value, (want, want_ty)) in returned.iter().zip(expected) {
                let Some(def) = value.borrow().get_defining_op() else {
                    tally.fail(plan, "a returned value has no definition".into());
                    continue;
                };
                let def = def.borrow();
                let Some(constant) = def.downcast_ref::<Constant>() else {
                    tally.fail(plan, format!("returns a non-constant:\n{after}"));
                    continue;
                };
                let got = M::of_int(*constant.get_value());
                if got != *want || constant.result().ty() != *want_ty {
                    tally.fail(
                        plan,
                        format!(
                            "driver gave {:#x} : {}, the model {:#x} : {want_ty}",
                            got.bits,
                            constant.result().ty(),
                            want.bits
                        ),
                    );
                }
            }
        }
    }
}

/// The plans are not vacuous: the generator reaches every "does not fold" row of the design's
/// table, and the `isize` cases split into folded and declined for the operations where the
/// targets can disagree.
#[test]
fn the_generator_reaches_the_cases_that_must_not_fold() {
    let mut declined: BTreeMap<String, usize> = BTreeMap::new();
    let mut isize_folded: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for seed in seeds() {
        run(seed, cases_per_variant(), |_, _, plan| {
            // The operation's name: the first word of the plan's.
            let name: String = plan.name.split(' ').next().unwrap().into();
            let entry = declined.entry(name.clone()).or_default();
            if plan.expected.is_none() {
                *entry += 1;
            }
            if plan.operand_ty.is_isize() {
                let entry = isize_folded.entry(name).or_default();
                if plan.expected.is_some() {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }
        });
    }
    for name in [
        "add",
        "sub",
        "mul",
        "udiv",
        "sdiv",
        "urem",
        "srem",
        "udivrem",
        "sdivrem",
        "shl",
        "shr",
        "ashr",
        "SextOrTrunc",
        "ZextOrTrunc",
        "cmpi",
        "min",
        "max",
        "clz",
        "ctz",
        "clo",
        "cto",
        "rotl",
        "rotr",
    ] {
        assert!(declined.get(name).copied().unwrap_or(0) > 0, "{name} was never declined");
    }
    for (name, (folded, not)) in &isize_folded {
        assert!(*folded > 0, "{name} never folded on isize ({not} declined)");
    }
}
