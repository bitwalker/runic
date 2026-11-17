//! The differential test of the integer patterns: each operation of `arith` (and `cf.select`,
//! `ub.poison`) is lowered by `convert-to-wasm` at each width, the lowered function is run by the
//! evaluator ([super::eval]) on a set of edge values, and the result is compared with `arith`'s
//! own arithmetic (the `Int` operations its folds call).
//!
//! A case is declared by [check] (one operation at one type) and the helpers below it: [binary],
//! [unary], [casts]. The reference of a case is a closure from the operands, as `Int`s of the
//! operand type's width, to an [Expect].
//!
//! Every result is also checked for the carried-form invariant: no bit at or above `N` is set.

use alloc::{boxed::Box, collections::BTreeSet, format, string::String, vec, vec::Vec};

use runic_dialect_arith::CmpPredicate;
use runic_ir::{Int, OperationRef, dialects::builtin::ModuleRef};

use super::{
    eval::{Outcome, Val, run},
    *,
};
use crate::{convert::convert_module, verify_wasm_form};

/// The widths every operation is tried at, besides `isize`.
const WIDTHS: [u32; 8] = [1, 7, 8, 16, 31, 32, 33, 64];

/// An integer type under test.
#[derive(Clone, Debug)]
struct Ty {
    /// As written: `i8`, `isize`.
    name: String,
    /// Its width in bits; 32 for `isize`.
    n: u32,
}

impl Ty {
    fn int(n: u32) -> Self {
        Self {
            name: format!("i{n}"),
            n,
        }
    }

    fn isize() -> Self {
        Self {
            name: String::from("isize"),
            n: 32,
        }
    }

    /// The width of the carrier: 32 up to `i32` and `isize`, 64 above.
    fn carrier(&self) -> u32 {
        if self.n <= 32 { 32 } else { 64 }
    }
}

/// The types an operation is tried at: each width, and `isize`.
fn types() -> Vec<Ty> {
    WIDTHS.iter().map(|n| Ty::int(*n)).chain([Ty::isize()]).collect()
}

/// What `arith` says an operation gives on some operands.
#[derive(Clone, Debug)]
enum Expect {
    /// This value.
    Value(Int),
    /// The operation traps.
    Trap,
    /// The result is undefined: any value that keeps the invariant.
    Undefined,
}

/// The edge values of the width `n`: 0, 1, 2, the maximum, the maximum less 1, the signed minimum,
/// the signed minimum plus 1, and the two alternating bit patterns.
fn edges(n: u32) -> Vec<Int> {
    let max = Int::all_ones(n);
    let min = Int::signed_min(n);
    let bits: BTreeSet<u128> = [
        0,
        1,
        2,
        max.unsigned(),
        max.unsigned() - 1,
        min.unsigned(),
        min.unsigned() + 1,
        0x5555_5555_5555_5555_u128,
        0xaaaa_aaaa_aaaa_aaaa_u128,
    ]
    .into_iter()
    .map(|bits| Int::from_bits(n, bits).unsigned())
    .collect();
    bits.into_iter().map(|bits| Int::from_bits(n, bits)).collect()
}

/// The shift amounts tried at the width `n`: every value from 0 to `n - 1`, and the edge values
/// (some of which are `n` or more, where `arith` leaves the result undefined).
fn amounts(n: u32) -> Vec<Int> {
    let mut bits: BTreeSet<u128> = (0..u128::from(n)).collect();
    bits.extend(edges(n).into_iter().map(Int::unsigned));
    bits.into_iter().map(|bits| Int::from_bits(n, bits)).collect()
}

/// The tuples that take one of each of `candidates`, in order.
fn product(candidates: &[Vec<Int>]) -> Vec<Vec<Int>> {
    let mut tuples = vec![Vec::new()];
    for list in candidates {
        let mut next = Vec::new();
        for tuple in &tuples {
            for value in list {
                let mut extended = tuple.clone();
                extended.push(*value);
                next.push(extended);
            }
        }
        tuples = next;
    }
    tuples
}

/// A function `@f` lowered by `convert-to-wasm`, with what keeps it alive.
struct Lowered {
    _module: ModuleRef,
    function: OperationRef,
}

/// The names of the parameters of a case.
const PARAMETERS: [&str; 3] = ["a", "b", "c"];

/// Builds a module with one `extern("wasm")` function `@f(params) -> result` that holds `body`
/// (the text of the operation that defines `%r`), under the 32-bit data layout, converts it with
/// `convert-to-wasm` and verifies the result with `wasm-verify-form`.
fn lower(
    context: &Rc<Context>,
    params: &[&Ty],
    result: &Ty,
    body: &str,
) -> Result<Lowered, Report> {
    let parameters = params
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("%{}: {}", PARAMETERS[index], ty.name))
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
            "builtin.module public @m attributes {{ data_layout = #builtin.data_layout<little, \
             ptr = 32> }} {{\n    builtin.function public extern(\"wasm\") @f({parameters}) -> {} \
             {{\n        %r = {body};\n        builtin.ret %r : ({});\n    }};\n}};",
            result.name, result.name
        );
    let module = parse_module_in(context, &text)?;
    convert_module(module)?;
    module.borrow().as_operation().recursively_verify()?;
    verify_wasm_form(module.as_operation_ref())?;
    let function = runic_ir::SymbolTable::symbol_manager(&*module.borrow())
        .lookup_op("f")
        .expect("the function @f");
    Ok(Lowered {
        _module: module,
        function,
    })
}

/// The operands as `type: value` pairs, for a failure message.
fn describe_operands(params: &[&Ty], operands: &[Int]) -> String {
    params
        .iter()
        .zip(operands)
        .map(|(ty, value)| format!("{}: {}", ty.name, value.unsigned()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Lowers `body`, an operation of the parameters `params` that gives a `result`, and checks the
/// lowered function against `reference` on every tuple of `candidates` (a list per parameter).
/// Returns the number of evaluations.
///
/// `what` names the case in a failure.
#[track_caller]
fn check(
    context: &Rc<Context>,
    what: &str,
    params: &[&Ty],
    result: &Ty,
    body: &str,
    candidates: &[Vec<Int>],
    reference: impl Fn(&[Int]) -> Expect,
) -> usize {
    let lowered = lower(context, params, result, body)
        .unwrap_or_else(|report| panic!("{what} `{body}`: {}", describe(&report)));
    let mut count = 0;
    for operands in product(candidates) {
        let arguments = operands
            .iter()
            .zip(params)
            .map(|(value, ty)| Val::new(value.unsigned() as u64, ty.carrier()))
            .collect::<Vec<_>>();
        let outcome = run(lowered.function, &arguments);
        let at = || format!("{what} `{body}` at ({})", describe_operands(params, &operands));
        let expected = reference(&operands);
        match (&outcome, &expected) {
            (Outcome::Trap, Expect::Trap) => {}
            (Outcome::Values(values), expected) => {
                let [value] = values.as_slice() else {
                    panic!("{}: returned {values:?}", at());
                };
                assert_eq!(value.width, result.carrier(), "{}: the carrier of the result", at());
                // The invariant of carried form: no bit at or above `N` is set.
                assert_eq!(
                    u128::from(value.bits) >> result.n,
                    0,
                    "{}: the result {:#x} has bits at or above {}",
                    at(),
                    value.bits,
                    result.n
                );
                match expected {
                    Expect::Value(expected) => assert_eq!(
                        u128::from(value.bits),
                        expected.unsigned(),
                        "{}: wrong value",
                        at()
                    ),
                    Expect::Undefined => {}
                    Expect::Trap => {
                        panic!("{}: arith traps, but the lowered code gave {value:?}", at())
                    }
                }
            }
            (Outcome::Trap, expected) => panic!("{}: trapped, but arith says {expected:?}", at()),
        }
        count += 1;
    }
    count
}

/// The `Int` operations `arith` folds an operation with: wrapping, and the checked ones that say
/// whether the exact result fits.
struct Ops {
    wrapping: fn(Int, Int) -> Int,
    signed: fn(Int, Int) -> Option<Int>,
    unsigned: fn(Int, Int) -> Option<Int>,
}

/// `arith.{name}` with the operands `%a, %b` of every type, against `reference`, which is given the
/// operands and the width.
fn binary(
    context: &Rc<Context>,
    name: &str,
    suffix: &str,
    amounts_for_b: bool,
    reference: impl Fn(Int, Int, u32) -> Expect,
) -> usize {
    let mut count = 0;
    for ty in types() {
        let b = if amounts_for_b {
            amounts(ty.n)
        } else {
            edges(ty.n)
        };
        count += check(
            context,
            name,
            &[&ty, &ty],
            &ty,
            &format!("arith.{name} %a, %b{suffix}"),
            &[edges(ty.n), b],
            |operands| reference(operands[0], operands[1], ty.n),
        );
    }
    count
}

/// `arith.{name} %a` of every type, against `reference`.
fn unary(context: &Rc<Context>, name: &str, reference: impl Fn(Int) -> Expect) -> usize {
    let mut count = 0;
    for ty in types() {
        count += check(
            context,
            name,
            &[&ty],
            &ty,
            &format!("arith.{name} %a"),
            &[edges(ty.n)],
            |operands| reference(operands[0]),
        );
    }
    count
}

/// `arith.{name} %a <{ ty = .. }>` over each pair of `from` and `to` types, against `reference`
/// (the operand, and the width of the result).
fn casts(
    context: &Rc<Context>,
    name: &str,
    pairs: &[(Ty, Ty)],
    reference: impl Fn(Int, u32) -> Int,
) -> usize {
    let mut count = 0;
    for (from, to) in pairs {
        count += check(
            context,
            name,
            &[from],
            to,
            &format!("arith.{name} %a <{{ ty = {} }}>", to.name),
            &[edges(from.n)],
            |operands| Expect::Value(reference(operands[0], to.n)),
        );
    }
    count
}

/// Every pair of distinct fixed widths, the first narrower than the second (`widening`) or wider.
fn fixed_pairs(widening: bool) -> Vec<(Ty, Ty)> {
    let mut pairs = Vec::new();
    for from in WIDTHS {
        for to in WIDTHS {
            if (widening && from < to) || (!widening && from > to) {
                pairs.push((Ty::int(from), Ty::int(to)));
            }
        }
    }
    pairs
}

/// Every pair of `isize` and a fixed width, in both directions.
fn isize_pairs() -> Vec<(Ty, Ty)> {
    let mut pairs = Vec::new();
    for n in WIDTHS {
        pairs.push((Ty::int(n), Ty::isize()));
        pairs.push((Ty::isize(), Ty::int(n)));
    }
    pairs
}

/// The text of an overflow mode, and the reference of an operation in it.
type ModeReference = (&'static str, Box<dyn Fn(Int, Int) -> Expect>);

/// The wrapping mode, which gives the wrapped value, and the unchecked modes, which are undefined
/// when the exact result does not fit.
fn overflow_modes(ops: &'static Ops) -> [ModeReference; 3] {
    [
        ("", Box::new(|a, b| Expect::Value((ops.wrapping)(a, b)))),
        (
            " <{ overflow = #builtin.overflow<unchecked signed> }>",
            Box::new(|a, b| (ops.signed)(a, b).map_or(Expect::Undefined, Expect::Value)),
        ),
        (
            " <{ overflow = #builtin.overflow<unchecked unsigned> }>",
            Box::new(|a, b| (ops.unsigned)(a, b).map_or(Expect::Undefined, Expect::Value)),
        ),
    ]
}

/// The reference of an operation whose result is `Some` or traps.
fn value_or_trap(result: Option<Int>) -> Expect {
    result.map_or(Expect::Trap, Expect::Value)
}

static ADD: Ops = Ops {
    wrapping: Int::wrapping_add,
    signed: Int::checked_add_signed,
    unsigned: Int::checked_add_unsigned,
};
static SUB: Ops = Ops {
    wrapping: Int::wrapping_sub,
    signed: Int::checked_sub_signed,
    unsigned: Int::checked_sub_unsigned,
};
static MUL: Ops = Ops {
    wrapping: Int::wrapping_mul,
    signed: Int::checked_mul_signed,
    unsigned: Int::checked_mul_unsigned,
};

#[test]
fn add_sub_and_mul_in_the_wrapping_and_unchecked_modes_match_arith() {
    let context = context();
    let mut count = 0;
    for (name, ops) in [("add", &ADD), ("sub", &SUB), ("mul", &MUL)] {
        for (suffix, reference) in overflow_modes(ops) {
            count += binary(&context, name, suffix, false, |a, b, _| reference(a, b));
        }
    }
    assert!(count > 3000, "{count}");
}

#[test]
fn a_checked_operation_has_no_pattern_here() {
    let context = context();
    for name in ["add", "sub", "mul"] {
        for sign in ["signed", "unsigned"] {
            let body =
                format!("arith.{name} %a, %b <{{ overflow = #builtin.overflow<checked {sign}> }}>");
            let ty = Ty::int(32);
            let error = lower(&context, &[&ty, &ty], &ty, &body)
                .err()
                .map(|report| describe(&report))
                .expect("a checked operation is not lowered");
            assert!(
                error.contains(&format!("failed to legalize operation 'arith.{name}'")),
                "{error}"
            );
        }
    }
}

#[test]
fn bitwise_operations_match_arith() {
    let context = context();
    let mut count = 0;
    count += binary(&context, "band", "", false, |a, b, _| Expect::Value(a & b));
    count += binary(&context, "bor", "", false, |a, b, _| Expect::Value(a | b));
    count += binary(&context, "bxor", "", false, |a, b, _| Expect::Value(a ^ b));
    count += unary(&context, "bnot", |a| Expect::Value(!a));
    count += unary(&context, "neg", |a| Expect::Value(a.wrapping_neg()));
    assert!(count > 1000, "{count}");
}

#[test]
fn unsigned_division_and_remainder_match_arith_and_trap_on_zero() {
    let context = context();
    // A zero divisor traps in `arith` and in the lowering: `None` is a trap.
    let count = binary(&context, "udiv", "", false, |a, b, _| value_or_trap(a.div_unsigned(b)))
        + binary(&context, "urem", "", false, |a, b, _| value_or_trap(a.rem_unsigned(b)));
    assert!(count > 1000, "{count}");
}

#[test]
fn signed_division_and_remainder_match_arith() {
    let context = context();
    let carrier = |n: u32| if n <= 32 { 32 } else { 64 };
    let count = binary(&context, "sdiv", "", false, |a, b, n| match a.div_signed(b) {
        Some(quotient) => Expect::Value(quotient),
        // Division by zero traps in `arith` and in the lowering.
        None if b.is_zero() => Expect::Trap,
        // The minimum value divided by -1 traps in `arith`. The lowering divides sign-extended
        // operands in the carrier: when `N` is below the carrier's width the division does not
        // trap, and the masked quotient is the minimum value (an operation that would trap may be
        // given any behaviour that does not introduce a trap); when `N` is the carrier's width
        // Wasm's `div_s` traps as well.
        None if n < carrier(n) => Expect::Value(a),
        None => Expect::Trap,
    }) + binary(&context, "srem", "", false, |a, b, _| {
        // Only a zero divisor traps; the minimum value remainder -1 is 0 in both.
        value_or_trap(a.rem_signed(b))
    });
    assert!(count > 1000, "{count}");
}

#[test]
fn division_by_zero_traps_at_every_width() {
    let context = context();
    for ty in types() {
        for op in ["udiv", "sdiv", "urem", "srem"] {
            let lowered = lower(&context, &[&ty, &ty], &ty, &format!("arith.{op} %a, %b"))
                .unwrap_or_else(|report| panic!("{}", describe(&report)));
            let arguments = [Val::new(1, ty.carrier()), Val::new(0, ty.carrier())];
            assert_eq!(run(lowered.function, &arguments), Outcome::Trap, "{op} at {}", ty.name);
        }
    }
}

#[test]
fn the_minimum_divided_by_minus_one_is_the_minimum_below_the_carrier_and_traps_at_it() {
    let context = context();
    for ty in types() {
        let lowered = lower(&context, &[&ty, &ty], &ty, "arith.sdiv %a, %b")
            .unwrap_or_else(|report| panic!("{}", describe(&report)));
        let min = Int::signed_min(ty.n).unsigned() as u64;
        let minus_one = Int::all_ones(ty.n).unsigned() as u64;
        let arguments = [Val::new(min, ty.carrier()), Val::new(minus_one, ty.carrier())];
        let outcome = run(lowered.function, &arguments);
        if ty.n < ty.carrier() {
            // `arith` traps; the lowering does not introduce a trap and gives the minimum value.
            assert_eq!(outcome, Outcome::Values(vec![Val::new(min, ty.carrier())]), "{}", ty.name);
        } else {
            assert_eq!(outcome, Outcome::Trap, "{}", ty.name);
        }
    }
}

#[test]
fn shifts_match_arith_below_the_width_and_keep_carried_form_above_it() {
    let context = context();
    // `arith` leaves an amount of `N` or more undefined: only the invariant is asserted there.
    let shift = |a: Int, b: Int, shift: fn(Int, u32) -> Int| {
        let amount = b.shift_amount_unsigned();
        if amount < a.width() {
            Expect::Value(shift(a, amount))
        } else {
            Expect::Undefined
        }
    };
    let count = binary(&context, "shl", "", true, |a, b, _| shift(a, b, |v, k| v << k))
        + binary(&context, "shr", "", true, |a, b, _| shift(a, b, Int::lshr))
        + binary(&context, "ashr", "", true, |a, b, _| shift(a, b, Int::ashr));
    assert!(count > 3000, "{count}");
}

#[test]
fn every_comparison_matches_arith() {
    let context = context();
    let mut count = 0;
    for predicate in [
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
    ] {
        for ty in types() {
            count += check(
                &context,
                predicate.as_str(),
                &[&ty, &ty],
                &Ty::int(1),
                &format!("arith.cmpi {} %a, %b", predicate.as_str()),
                &[edges(ty.n), edges(ty.n)],
                |operands| Expect::Value(Int::from(predicate.evaluate(operands[0], operands[1]))),
            );
        }
    }
    assert!(count > 3000, "{count}");
}

#[test]
fn truncation_and_extension_between_widths_match_arith() {
    let context = context();
    let mut count = 0;
    count += casts(&context, "trunc", &fixed_pairs(false), |a, m| a.trunc(m));
    count += casts(&context, "zext", &fixed_pairs(true), |a, m| a.zext(m));
    count += casts(&context, "sext", &fixed_pairs(true), |a, m| a.sext(m));
    count += casts(&context, "zext_or_trunc", &isize_pairs(), |a, m| a.zext_or_trunc(m));
    count += casts(&context, "sext_or_trunc", &isize_pairs(), |a, m| a.sext_or_trunc(m));
    assert!(count > 500, "{count}");
}

#[test]
fn select_matches_arith() {
    let context = context();
    let mut count = 0;
    let flag = Ty::int(1);
    for ty in types() {
        count += check(
            &context,
            "cf.select",
            &[&flag, &ty, &ty],
            &ty,
            "cf.select %a, %b, %c",
            &[edges(1), edges(ty.n), edges(ty.n)],
            |operands| {
                Expect::Value(if operands[0].is_zero() {
                    operands[2]
                } else {
                    operands[1]
                })
            },
        );
    }
    assert!(count > 300, "{count}");
}

#[test]
fn poison_of_an_integer_is_a_value_that_keeps_carried_form() {
    let context = context();
    for ty in types() {
        let body = format!("ub.poison <{{ value = #ub.poison<{}> }}>", ty.name);
        check(&context, "ub.poison", &[], &ty, &body, &[], |_| Expect::Undefined);
    }
}

#[test]
fn the_operations_left_to_other_patterns_are_reported() {
    let context = context();
    let ty = Ty::int(32);
    for (name, body) in [
        ("arith.rotl", "arith.rotl %a, %b"),
        ("arith.min", "arith.min %a, %b <{ signedness = #builtin.signedness<signed> }>"),
        ("arith.clz", "arith.clz %a"),
    ] {
        let error = lower(&context, &[&ty, &ty], &ty, body)
            .err()
            .map(|report| describe(&report))
            .expect("not lowered by the integer patterns");
        assert!(error.contains(&format!("failed to legalize operation '{name}'")), "{error}");
    }
}
