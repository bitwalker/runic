//! The differential test of the integer patterns: each operation of `arith` (and `cf.select`,
//! `ub.poison`) is lowered by `convert-to-wasm` at each width, the lowered function is run by the
//! evaluator ([super::eval]) on a set of edge values, and the result is compared with `arith`'s
//! own arithmetic (the `Int` operations its folds call).
//!
//! A case is declared by [check] (one operation at one type, on every tuple of candidate lists),
//! or [check_tuples] (an operation with any number of results, on explicit tuples), and the
//! helpers below them: [binary], [unary], [casts]. The reference of a case is a closure from the
//! operands, as `Int`s of the operand type's width, to an [Expect] (one per result for
//! [check_tuples]). The other test modules of the numeric patterns declare their cases with these.
//!
//! Every result is also checked for the carried-form invariant: no bit at or above `N` is set.

use alloc::{boxed::Box, collections::BTreeSet, format, string::String, vec, vec::Vec};

use runic_dialect_arith::CmpPredicate;
use runic_ir::{Int, OperationRef, dialects::builtin::ModuleRef};

use super::{
    eval::{Outcome, Val},
    exec::{Cache, run, run_cached},
    *,
};
use crate::{convert::convert_module, verify_wasm_form};

/// The widths every operation is tried at, besides `isize`.
/// They include the boundaries of the classes of the patterns: 2 (the least width with a sign
/// bit apart from its value bits), 17 (the least whose product needs an `i64` on a 32-bit
/// carrier) and 62 (the most below 63 carried in an `i64`).
pub(super) const WIDTHS: [u32; 12] = [1, 2, 7, 8, 16, 17, 31, 32, 33, 62, 63, 64];

/// An integer type under test.
#[derive(Clone, Debug)]
pub(super) struct Ty {
    /// As written: `i8`, `isize`.
    pub name: String,
    /// Its width in bits; 32 for `isize`.
    pub n: u32,
}

impl Ty {
    pub fn int(n: u32) -> Self {
        Self {
            name: format!("i{n}"),
            n,
        }
    }

    pub fn isize() -> Self {
        Self {
            name: String::from("isize"),
            n: 32,
        }
    }

    /// The width of the carrier: 32 up to `i32` and `isize`, 64 above.
    pub fn carrier(&self) -> u32 {
        if self.n <= 32 { 32 } else { 64 }
    }
}

/// The types an operation is tried at: each width, and `isize`.
pub(super) fn types() -> Vec<Ty> {
    WIDTHS.iter().map(|n| Ty::int(*n)).chain([Ty::isize()]).collect()
}

/// What `arith` says an operation gives on some operands.
#[derive(Clone, Debug)]
pub(super) enum Expect {
    /// This value.
    Value(Int),
    /// The operation traps.
    Trap,
    /// The result is undefined: any value that keeps the invariant.
    Undefined,
}

/// The edge values of the width `n`: 0, 1, 2, the maximum, the maximum less 1, the signed minimum,
/// the signed minimum plus 1, and the two alternating bit patterns.
pub(super) fn edges(n: u32) -> Vec<Int> {
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
pub(super) fn amounts(n: u32) -> Vec<Int> {
    let mut bits: BTreeSet<u128> = (0..u128::from(n)).collect();
    bits.extend(edges(n).into_iter().map(Int::unsigned));
    bits.into_iter().map(|bits| Int::from_bits(n, bits)).collect()
}

/// The tuples that take one of each of `candidates`, in order.
pub(super) fn product<T: Copy>(candidates: &[Vec<T>]) -> Vec<Vec<T>> {
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
pub(super) struct Lowered {
    _module: ModuleRef,
    /// The lowered `builtin.function`.
    pub function: OperationRef,
}

/// The names of the parameters of a case.
const PARAMETERS: [&str; 3] = ["a", "b", "c"];

/// The text of a module with one `extern("wasm")` function `@f` whose parameters `%a`, `%b`, ...
/// have the types `params`, whose results have the types `results`, and whose body is the
/// operation `body`, which defines the values `%r` (one result) or `%r0`, `%r1`, ... (several),
/// followed by a `builtin.ret` of them; under the 32-bit data layout.
pub(super) fn module_text(params: &[&str], results: &[&str], body: &str) -> String {
    let parameters = params
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("%{}: {ty}", PARAMETERS[index]))
        .collect::<Vec<_>>()
        .join(", ");
    let names = if results.len() == 1 {
        vec![String::from("%r")]
    } else {
        (0..results.len()).map(|index| format!("%r{index}")).collect()
    };
    let names = names.join(", ");
    let types = results.join(", ");
    let defined = if results.is_empty() {
        format!("{body};\n")
    } else {
        format!("{names} = {body};\n")
    };
    let signature = match results {
        [] => String::new(),
        [one] => format!(" -> {one}"),
        _ => format!(" -> ({types})"),
    };
    let ret = if results.is_empty() {
        String::from("builtin.ret;")
    } else {
        format!("builtin.ret {names} : ({types});")
    };
    format!(
        "builtin.module public @m attributes {{ data_layout = #builtin.data_layout<little, ptr = \
         32> }} {{\n    builtin.function public extern(\"wasm\") @f({parameters}){signature} \
         {{\n        {defined}        {ret}\n    }};\n}};"
    )
}

/// Builds the module of [module_text], converts it with `convert-to-wasm` and verifies the result
/// with `wasm-verify-form`.
pub(super) fn lower_text(
    context: &Rc<Context>,
    params: &[&str],
    results: &[&str],
    body: &str,
) -> Result<Lowered, Report> {
    let module = parse_module_in(context, &module_text(params, results, body))?;
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

/// [lower_text] of a function `@f(params) -> result` whose body is `body`, the text of the
/// operation that defines `%r`.
pub(super) fn lower(
    context: &Rc<Context>,
    params: &[&Ty],
    result: &Ty,
    body: &str,
) -> Result<Lowered, Report> {
    let params = params.iter().map(|ty| ty.name.as_str()).collect::<Vec<_>>();
    lower_text(context, &params, &[result.name.as_str()], body)
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
pub(super) fn check(
    context: &Rc<Context>,
    what: &str,
    params: &[&Ty],
    result: &Ty,
    body: &str,
    candidates: &[Vec<Int>],
    reference: impl Fn(&[Int]) -> Expect,
) -> usize {
    check_tuples(context, what, params, &[result], body, &product(candidates), |operands| {
        vec![reference(operands)]
    })
}

/// Lowers `body`, an operation of the parameters `params` that gives one value of each of
/// `results`, and checks the lowered function against `reference` on each of `tuples` (one value
/// per parameter). `reference` gives one [Expect] per result; the function must trap exactly when
/// one of them is [Expect::Trap]. Returns the number of evaluations.
///
/// Every result is checked for the carried-form invariant. `what` names the case in a failure.
#[track_caller]
pub(super) fn check_tuples(
    context: &Rc<Context>,
    what: &str,
    params: &[&Ty],
    results: &[&Ty],
    body: &str,
    tuples: &[Vec<Int>],
    reference: impl Fn(&[Int]) -> Vec<Expect>,
) -> usize {
    let param_names = params.iter().map(|ty| ty.name.as_str()).collect::<Vec<_>>();
    let result_names = results.iter().map(|ty| ty.name.as_str()).collect::<Vec<_>>();
    let lowered = lower_text(context, &param_names, &result_names, body)
        .unwrap_or_else(|report| panic!("{what} `{body}`: {}", describe(&report)));
    let mut count = 0;
    let mut cache = Cache::default();
    for operands in tuples {
        let arguments = operands
            .iter()
            .zip(params)
            .map(|(value, ty)| Val::new(value.unsigned() as u64, ty.carrier()))
            .collect::<Vec<_>>();
        let outcome = run_cached(&mut cache, lowered.function, &arguments);
        let at = || format!("{what} `{body}` at ({})", describe_operands(params, operands));
        let expected = reference(operands);
        assert_eq!(expected.len(), results.len(), "{}: the reference", at());
        let traps = expected.iter().any(|expect| matches!(expect, Expect::Trap));
        match &outcome {
            Outcome::Trap if traps => {}
            Outcome::Trap => panic!("{}: trapped, but arith says {expected:?}", at()),
            Outcome::Values(values) if traps => {
                panic!("{}: arith traps, but the lowered code gave {values:?}", at())
            }
            Outcome::Values(values) => {
                assert_eq!(values.len(), results.len(), "{}: returned {values:?}", at());
                for (index, ((value, ty), expected)) in
                    values.iter().zip(results).zip(&expected).enumerate()
                {
                    assert_eq!(
                        value.width,
                        ty.carrier(),
                        "{}: the carrier of result {index}",
                        at()
                    );
                    // The invariant of carried form: no bit at or above `N` is set.
                    assert_eq!(
                        u128::from(value.bits) >> ty.n,
                        0,
                        "{}: result {index}, {:#x}, has bits at or above {}",
                        at(),
                        value.bits,
                        ty.n
                    );
                    if let Expect::Value(expected) = expected {
                        assert_eq!(
                            u128::from(value.bits),
                            expected.unsigned(),
                            "{}: wrong value of result {index}",
                            at()
                        );
                    }
                }
            }
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
    // Division by zero and the minimum value divided by -1 trap in `arith` and in the lowering, at
    // every width.
    let count = binary(&context, "sdiv", "", false, |a, b, _| value_or_trap(a.div_signed(b)))
        + binary(&context, "srem", "", false, |a, b, _| {
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
fn the_minimum_divided_by_minus_one_traps_at_every_width() {
    let context = context();
    for ty in types() {
        let lowered = lower(&context, &[&ty, &ty], &ty, "arith.sdiv %a, %b")
            .unwrap_or_else(|report| panic!("{}", describe(&report)));
        let min = Int::signed_min(ty.n).unsigned() as u64;
        let minus_one = Int::all_ones(ty.n).unsigned() as u64;
        let arguments = [Val::new(min, ty.carrier()), Val::new(minus_one, ty.carrier())];
        // `arith` traps, and a trap whose result is used is kept: below the carrier's width the
        // quotient is tested, at it `wasm.div_s` traps.
        assert_eq!(run(lowered.function, &arguments), Outcome::Trap, "{}", ty.name);
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
fn constants_and_immediate_returns_of_every_width_give_their_value_zero_extended() {
    let context = context();
    for ty in types() {
        for value in edges(ty.n) {
            let expected = Outcome::Values(vec![Val::new(value.unsigned() as u64, ty.carrier())]);
            let body = format!("arith.constant {} : {}", value.unsigned(), ty.name);
            let lowered = lower(&context, &[], &ty, &body)
                .unwrap_or_else(|report| panic!("{body}: {}", describe(&report)));
            assert_eq!(run(lowered.function, &[]), expected, "{body}");
            let text = format!(
                "builtin.module public @m {LAYOUT} {{\n    builtin.function public \
                 extern(\"wasm\") @f() -> {} {{\n        builtin.ret_imm {} : {};\n    }};\n}};",
                ty.name,
                value.unsigned(),
                ty.name
            );
            let module = parse_module_in(&context, &text)
                .unwrap_or_else(|report| panic!("{text}: {}", describe(&report)));
            convert_module(module).unwrap_or_else(|report| panic!("{}", describe(&report)));
            verify_wasm_form(module.as_operation_ref())
                .unwrap_or_else(|report| panic!("{}", describe(&report)));
            let function = runic_ir::SymbolTable::symbol_manager(&*module.borrow())
                .lookup_op("f")
                .expect("the function @f");
            assert_eq!(run(function, &[]), expected, "ret_imm {} : {}", value.unsigned(), ty.name);
        }
    }
}

#[test]
fn poison_of_a_float_or_a_pointer_is_the_zero_of_its_carrier() {
    // `cf.select` takes integers only, so its float and pointer forms do not verify.
    let context = context();
    let cases = [
        ("f32", Val::f32(0.0)),
        ("f64", Val::f64(0.0)),
        ("ptr<i8>", Val::i32(0)),
        ("ptr", Val::i32(0)),
    ];
    for (ty, zero) in cases {
        let body = format!("ub.poison <{{ value = #ub.poison<{ty}> }}>");
        let lowered = lower_text(&context, &[], &[ty], &body)
            .unwrap_or_else(|report| panic!("{ty}: {}", describe(&report)));
        assert_eq!(run(lowered.function, &[]), Outcome::Values(vec![zero]), "{ty}");
    }
}
