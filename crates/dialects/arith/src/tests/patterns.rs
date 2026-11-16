//! The canonicalization patterns preserve meaning under signless types.
//!
//! Random small functions over two `i1` parameters and an `i8` one are built from the operations
//! the patterns look at (`zext`, `sext`, `trunc`, `cmpi`, `bnot`, the bitwise three, constants),
//! interpreted with [super::model] on every input, canonicalized with the patterns and the folds,
//! and interpreted again: the results must be the same, and the patterns must have fired.

use runic_ir::{Operation, dialects::builtin::Ret};

use super::{
    model::{self, M},
    random::{Rng, cases_per_variant, seeds},
    *,
};

/// The value of every result of `op` given the values of its operands, for the operations the
/// random functions use.
fn interpret(op: &Operation, operands: &[M]) -> Vec<M> {
    if let Some(constant) = op.downcast_ref::<Constant>() {
        return vec![M::of_int(*constant.get_value())];
    }
    let width_of = |index: usize| op.results()[index].borrow().ty().integer_width().unwrap();
    if let Some(cmpi) = op.downcast_ref::<Cmpi>() {
        let holds = model::compare(cmpi.get_predicate().as_str(), operands[0], operands[1]);
        return vec![M::flag(holds)];
    }
    if op.is::<Bnot>() {
        return vec![model::bnot(operands[0])];
    }
    if op.is::<Zext>() {
        return vec![operands[0].zext(width_of(0))];
    }
    if op.is::<Sext>() {
        return vec![operands[0].sext(width_of(0))];
    }
    if op.is::<Trunc>() {
        return vec![operands[0].trunc(width_of(0))];
    }
    if op.is::<Band>() {
        return vec![model::bitwise(operands[0], operands[1], |x, y| x & y)];
    }
    if op.is::<Bor>() {
        return vec![model::bitwise(operands[0], operands[1], |x, y| x | y)];
    }
    if op.is::<Bxor>() {
        return vec![model::bitwise(operands[0], operands[1], |x, y| x ^ y)];
    }
    panic!("the interpreter does not know '{}'", op.name());
}

/// What `test`'s function returns on `inputs`.
fn run(test: &Test, inputs: &[M]) -> Vec<M> {
    let mut env: Vec<(ValueRef, M)> =
        params(test).into_iter().zip(inputs.iter().copied()).collect();
    let lookup = |env: &[(ValueRef, M)], value: ValueRef| {
        env.iter()
            .find(|(known, _)| *known == value)
            .unwrap_or_else(|| panic!("use of a value with no definition yet"))
            .1
    };
    for op in ops_of(test.function()) {
        let op = op.borrow();
        let operands: Vec<M> = op
            .operands()
            .all()
            .iter()
            .map(|operand| lookup(&env, operand.borrow().as_value_ref()))
            .collect();
        if op.is::<Ret>() {
            return operands;
        }
        let results = interpret(&op, &operands);
        for (result, value) in op.results().iter().zip(results) {
            let result = result.borrow();
            assert_eq!(result.ty().integer_width(), Some(value.w), "'{}' result width", op.name());
            env.push((result.as_value_ref(), value));
        }
    }
    panic!("no builtin.ret");
}

/// One step of a random function: an operation on earlier values (by index into the values so
/// far, the parameters first), with the width of its result.
#[derive(Clone, Debug)]
enum Step {
    Zext(usize, u32),
    Sext(usize, u32),
    Trunc(usize, u32),
    /// `cmpi predicate a, b`, or with the operands swapped.
    Cmp(CmpPredicate, usize, usize, bool),
    /// `cmpi predicate a, constant`, or with the operands swapped.
    CmpConst(CmpPredicate, usize, u128, bool),
    Bnot(usize),
    Bit(u8, usize, usize),
}

const PARAM_WIDTHS: [u32; 3] = [1, 1, 8];

/// A random function of `len` steps: the steps and the width of every value.
fn random_steps(rng: &mut Rng, len: usize) -> (Vec<Step>, Vec<u32>) {
    let mut widths: Vec<u32> = PARAM_WIDTHS.to_vec();
    let mut steps = Vec::new();
    while steps.len() < len {
        let a = rng.below(widths.len() as u64) as usize;
        // Favour recent values, so that chains form.
        let a = if rng.below(2) == 0 {
            widths.len() - 1 - a.min(2).min(widths.len() - 1)
        } else {
            a
        };
        let w = widths[a];
        let wider: Vec<u32> = [2u32, 8, 16].into_iter().filter(|to| *to > w).collect();
        let narrower: Vec<u32> = [1u32, 2].into_iter().filter(|to| *to < w).collect();
        let predicate = if rng.below(3) == 0 {
            CmpPredicate::ALL[rng.below(10) as usize]
        } else if rng.below(2) == 0 {
            CmpPredicate::Eq
        } else {
            CmpPredicate::Ne
        };
        let step = match rng.below(9) {
            0 | 1 if !wider.is_empty() => {
                let to = wider[rng.below(wider.len() as u64) as usize];
                if rng.below(2) == 0 {
                    Step::Zext(a, to)
                } else {
                    Step::Sext(a, to)
                }
            }
            2 if !narrower.is_empty() => {
                Step::Trunc(a, narrower[rng.below(narrower.len() as u64) as usize])
            }
            3..=5 => {
                let constant = match rng.below(6) {
                    0 | 1 => 0,
                    2 | 3 => 1,
                    4 => model::mask(w),
                    _ => rng.bits() & model::mask(w),
                };
                Step::CmpConst(predicate, a, constant, rng.below(2) == 0)
            }
            6 => Step::Bnot(a),
            7 => {
                let same: Vec<usize> = (0..widths.len()).filter(|b| widths[*b] == w).collect();
                let b = same[rng.below(same.len() as u64) as usize];
                Step::Cmp(predicate, a, b, rng.below(2) == 0)
            }
            8 => {
                let same: Vec<usize> = (0..widths.len()).filter(|b| widths[*b] == w).collect();
                let b = same[rng.below(same.len() as u64) as usize];
                Step::Bit(rng.below(3) as u8, a, b)
            }
            _ => continue,
        };
        widths.push(match &step {
            Step::Zext(_, to) | Step::Sext(_, to) | Step::Trunc(_, to) => *to,
            Step::Cmp(..) | Step::CmpConst(..) => 1,
            Step::Bnot(a) | Step::Bit(_, a, _) => widths[*a],
        });
        steps.push(step);
    }
    (steps, widths)
}

/// Builds the function of `steps`, returning the last `returned` values.
fn build(steps: &[Step], widths: &[u32], returned: usize) -> Test {
    let result_widths: Vec<u32> = widths[widths.len() - returned..].to_vec();
    let mut test = test_fn(
        |cx| PARAM_WIDTHS.iter().map(|w| cx.int_type(*w)).collect(),
        move |cx| result_widths.iter().map(|w| cx.int_type(*w)).collect(),
    );
    let context = test.context_rc();
    let mut values = params(&test);
    {
        let mut b = test.function_builder();
        for step in steps {
            let value = match step {
                Step::Zext(a, to) => b.zext(values[*a], context.int_type(*to), SPAN),
                Step::Sext(a, to) => b.sext(values[*a], context.int_type(*to), SPAN),
                Step::Trunc(a, to) => b.trunc(values[*a], context.int_type(*to), SPAN),
                Step::Cmp(predicate, x, y, swap) => {
                    let (x, y) = if *swap { (*y, *x) } else { (*x, *y) };
                    b.cmpi(*predicate, values[x], values[y], SPAN)
                }
                Step::CmpConst(predicate, x, constant, swap) => {
                    let ty = values[*x].borrow().ty();
                    let w = ty.integer_width().unwrap();
                    let constant = b.int(Int::from_bits(w, *constant), ty, SPAN);
                    if *swap {
                        b.cmpi(*predicate, constant, values[*x], SPAN)
                    } else {
                        b.cmpi(*predicate, values[*x], constant, SPAN)
                    }
                }
                Step::Bnot(a) => b.bnot(values[*a], SPAN),
                Step::Bit(0, x, y) => b.band(values[*x], values[*y], SPAN),
                Step::Bit(1, x, y) => b.bor(values[*x], values[*y], SPAN),
                Step::Bit(_, x, y) => b.bxor(values[*x], values[*y], SPAN),
            }
            .unwrap();
            values.push(value);
        }
        let results: Vec<ValueRef> = values[values.len() - returned..].to_vec();
        b.ret(results, SPAN).unwrap();
    }
    test
}

/// Every input of the three parameters that is worth trying: both booleans, and the edges of `i8`.
fn inputs() -> Vec<[M; 3]> {
    let mut all = Vec::new();
    for a in 0..2u128 {
        for b in 0..2u128 {
            for c in [0u128, 1, 2, 3, 127, 128, 254, 255, 0x55, 0xaa] {
                all.push([M::new(1, a), M::new(1, b), M::new(8, c)]);
            }
        }
    }
    all
}

fn count_ops(test: &Test) -> usize {
    ops_of(test.function()).len()
}

#[test]
fn canonicalization_preserves_the_meaning_of_random_functions() {
    let functions = cases_per_variant().max(400);
    let (mut changed, mut shrunk) = (0, 0);
    for seed in seeds() {
        let mut rng = Rng(seed ^ 0xca11);
        for _ in 0..functions {
            let len = 3 + rng.below(8) as usize;
            let (steps, widths) = random_steps(&mut rng, len);
            let returned = 1 + rng.below(2.min(len as u64)) as usize;
            let test = build(&steps, &widths, returned);
            test.function().borrow().as_operation().verify().expect("valid before");
            let before_text = text_of(&test);
            let before: Vec<Vec<M>> = inputs().iter().map(|input| run(&test, input)).collect();
            let before_ops = count_ops(&test);
            let did_change = canonicalize(&test);
            if let Err(err) = test.function().borrow().as_operation().verify() {
                panic!(
                    "invalid after canonicalization: {err} \
                     {:?}\nbefore:\n{before_text}\nafter:\n{}",
                    label_messages(err.as_diagnostic()),
                    text_of(&test)
                );
            }
            let after: Vec<Vec<M>> = inputs().iter().map(|input| run(&test, input)).collect();
            if before != after {
                let at = before.iter().zip(&after).position(|(x, y)| x != y).unwrap();
                panic!(
                    "canonicalization changed the result on {:?}: {:?} became \
                     {:?}\nbefore:\n{before_text}\nafter:\n{}",
                    inputs()[at],
                    before[at],
                    after[at],
                    text_of(&test)
                );
            }
            // A second run finds nothing left to do.
            let again = text_of(&test);
            canonicalize(&test);
            assert_eq!(text_of(&test), again, "not a fixpoint; before:\n{before_text}");
            changed += usize::from(did_change);
            shrunk += usize::from(count_ops(&test) < before_ops);
        }
    }
    assert!(changed > 0, "canonicalization never changed anything");
    assert!(shrunk > 0, "canonicalization never simplified anything");
}

/// Each pattern fires on the shapes the design names, at widths 1, 2, 8 and 16, and the result is
/// what the model computes: the truth table of `cmpi pred (ext b), c` against its rewrite.
#[test]
fn the_bool_compare_pattern_agrees_with_the_truth_table() {
    let mut fired = 0;
    for extension in ["none", "zext", "sext", "zext-sext", "sext-zext"] {
        for width in [1u32, 2, 8, 16] {
            if (extension == "none") != (width == 1) {
                continue;
            }
            for predicate in CmpPredicate::ALL {
                for constant in [0u128, 1, model::mask(width)] {
                    for swap in [false, true] {
                        let mut steps = Vec::new();
                        let mut widths = PARAM_WIDTHS.to_vec();
                        let mut value = 0;
                        let mut push = |step: Step, w: u32, steps: &mut Vec<Step>| {
                            steps.push(step);
                            widths.push(w);
                            widths.len() - 1
                        };
                        match extension {
                            "zext" => value = push(Step::Zext(0, width), width, &mut steps),
                            "sext" => value = push(Step::Sext(0, width), width, &mut steps),
                            "zext-sext" if width > 2 => {
                                let inner = push(Step::Zext(0, 2), 2, &mut steps);
                                value = push(Step::Sext(inner, width), width, &mut steps);
                            }
                            "sext-zext" if width > 2 => {
                                let inner = push(Step::Sext(0, 2), 2, &mut steps);
                                value = push(Step::Zext(inner, width), width, &mut steps);
                            }
                            "none" => {}
                            _ => continue,
                        }
                        push(Step::CmpConst(predicate, value, constant, swap), 1, &mut steps);
                        let test = build(&steps, &widths, 1);
                        let before: Vec<Vec<M>> =
                            inputs().iter().map(|input| run(&test, input)).collect();
                        let ops = count_ops(&test);
                        canonicalize(&test);
                        test.function().borrow().as_operation().verify().expect("valid after");
                        let after: Vec<Vec<M>> =
                            inputs().iter().map(|input| run(&test, input)).collect();
                        assert_eq!(
                            before,
                            after,
                            "{extension} i{width} {predicate} {constant} swap={swap}:\n{}",
                            text_of(&test)
                        );
                        fired += usize::from(count_ops(&test) < ops);
                    }
                }
            }
        }
    }
    assert!(fired > 20, "the pattern fired only {fired} times");
}
