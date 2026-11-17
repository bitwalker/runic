//! The differential test of the float patterns: each float operation of `arith` is lowered by
//! `convert-to-wasm`, the lowered function is run by the evaluator ([super::eval]) on a set of
//! values, and the result is compared with `arith`'s own: the fold of the operation on the same
//! operands as constants, and for `arith.cmpf` the predicate's evaluation.
//!
//! A fold that declines on constant operands says the result is a NaN (`addf`, `subf`, `mulf`,
//! `divf`, `fpext`, `fptrunc`), which is compared as being a NaN, or poison (`fptosi`, `fptoui`),
//! where only carried form and the absence of a trap are asserted.

use alloc::{format, string::String, vec, vec::Vec};

use runic_dialect_arith::CmpFPredicate;
use runic_ir::{
    AttributeRef, Float, Int, OpFoldResult, OperationRef, SmallVec,
    attributes::IntegerLikeAttr,
    dialects::builtin::{Function, ModuleRef, attributes::FloatAttr},
    traits::Foldable,
};

use super::{
    eval::{Outcome, Val},
    exec::{Cache, run_cached},
    int::{Lowered, Ty, edges, lower_text, module_text, product, types},
    *,
};

/// An operand of a case.
#[derive(Copy, Clone, Debug)]
enum Arg {
    F32(f32),
    F64(f64),
    /// An integer of the type the case gives it (its width is that type's, 32 for `isize`).
    Int(Int),
}

impl Arg {
    /// The value the evaluator is given.
    fn val(self) -> Val {
        match self {
            Arg::F32(value) => Val::f32(value),
            Arg::F64(value) => Val::f64(value),
            Arg::Int(value) => {
                let carrier = if value.width() <= 32 { 32 } else { 64 };
                Val::new(value.unsigned() as u64, carrier)
            }
        }
    }
}

/// What `arith` gives.
#[derive(Clone, Debug)]
enum Expect {
    /// This float; a NaN is compared as being a NaN of its width.
    Float(Float),
    /// This integer, of the result's width `N`.
    Int(Int),
    /// Poison: any value that keeps carried form, and no trap.
    Poison,
}

/// A float type under test, by width.
fn float_name(width: u32) -> &'static str {
    if width == 32 { "f32" } else { "f64" }
}

/// The values every float operation is tried on, as `f64`s (each is an `f32` too, rounded where
/// it is not one): zeros of both signs, ones, infinities, a NaN, the largest and the smallest
/// normal of each width, a subnormal of each width, and halves.
fn specials() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
        -f64::MAX,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from(f32::MAX),
        f64::from(-f32::MAX),
        f64::from(f32::MIN_POSITIVE),
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from(f32::from_bits(1)),
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
    ]
}

/// `values` as operands of the float type of `width`.
fn floats(width: u32, values: &[f64]) -> Vec<Arg> {
    values
        .iter()
        .map(|value| {
            if width == 32 {
                Arg::F32(*value as f32)
            } else {
                Arg::F64(*value)
            }
        })
        .collect()
}

/// The floats around the limits of an `n`-bit integer, as `f64`s, for a float of `width`: for
/// each of `2^(n-1)`, `2^(n-1) - 1`, `-2^(n-1)`, `-2^(n-1) - 1`, `2^n` and `2^n - 1`, the float
/// nearest it and the floats on both sides of that one; and values that truncate to 0 and -1.
fn limits(n: u32, width: u32) -> Vec<f64> {
    let half = 1i128 << (n - 1);
    let full = 1i128 << n;
    let mut values = vec![0.99, -0.99, -1.0, -1.01];
    for anchor in [half, half - 1, -half, -half - 1, full, full - 1] {
        if width == 32 {
            let nearest = anchor as f32;
            values.extend([nearest, nearest.next_up(), nearest.next_down()].map(f64::from));
        } else {
            let nearest = anchor as f64;
            values.extend([nearest, nearest.next_up(), nearest.next_down()]);
        }
    }
    values
}

/// One tuple of one operand per value of `values`.
fn singles(values: &[Arg]) -> Vec<Vec<Arg>> {
    values.iter().map(|value| vec![*value]).collect()
}

/// A case: the lowered function and the source operation, unconverted, that gives the reference.
struct Case {
    lowered: Lowered,
    _reference: ModuleRef,
    op: OperationRef,
}

/// Lowers the operation `body` of `params` giving one `result`, and parses it again unconverted.
#[track_caller]
fn case(context: &Rc<Context>, params: &[&str], result: &str, body: &str) -> Case {
    let lowered = lower_text(context, params, &[result], body)
        .unwrap_or_else(|report| panic!("`{body}`: {}", describe(&report)));
    let reference = parse_module_in(context, &module_text(params, &[result], body))
        .unwrap_or_else(|report| panic!("`{body}`: {}", describe(&report)));
    let function = runic_ir::SymbolTable::symbol_manager(&*reference.borrow())
        .lookup_op("f")
        .expect("the function @f");
    let op = {
        let function = function.borrow();
        let function = function.downcast_ref::<Function>().expect("a builtin.function");
        let entry = function.body().entry_block_ref().expect("a body");
        let entry = entry.borrow();
        entry.body().iter().next().expect("an operation").as_operation_ref()
    };
    Case {
        lowered,
        _reference: reference,
        op,
    }
}

/// The constant attribute of `arg` as an operand of type `ty` (an integer of the type `ty`; for
/// `isize`, the 64-bit value `extend` gives).
fn attribute(context: &Rc<Context>, arg: Arg, ty: &str, extend: fn(Int) -> Int) -> AttributeRef {
    match arg {
        Arg::F32(value) => context.float_attr(Float::from_f32(value)).as_attribute_ref(),
        Arg::F64(value) => context.float_attr(Float::from_f64(value)).as_attribute_ref(),
        Arg::Int(value) => {
            let (value, ty) = if ty == "isize" {
                (extend(value), context.isize())
            } else {
                (value, context.int_type(value.width()))
            };
            context.int_attr(value, ty).as_attribute_ref()
        }
    }
}

/// The fold of `op` on the constants `attrs`: its one result, or `None` when it declines.
fn fold(op: OperationRef, attrs: Vec<AttributeRef>) -> Option<AttributeRef> {
    let operands = attrs.into_iter().map(Some).collect::<Vec<_>>();
    let mut results = SmallVec::<[OpFoldResult; 1]>::new();
    let outcome = op.borrow().fold_with(&operands, &mut results);
    if !outcome.is_ok() {
        return None;
    }
    match results.as_slice() {
        [OpFoldResult::Attribute(attr)] => Some(*attr),
        _ => panic!("a fold of `{}` gave {} results", op.borrow().name(), results.len()),
    }
}

/// The value of a folded constant.
fn folded(attr: AttributeRef) -> Expect {
    if let Ok(float) = attr.try_downcast_attr::<FloatAttr>() {
        return Expect::Float(*float.borrow().as_value());
    }
    let attr = attr.borrow();
    let int = attr.as_attr().as_trait::<dyn IntegerLikeAttr>().expect("an integer constant");
    Expect::Int(int.as_int())
}

/// Runs `case` on each of `tuples` and compares with `reference`. `result_n` is the width `N` of
/// an integer result, which is checked for carried form. Returns the number of evaluations.
#[track_caller]
fn check(
    case: &Case,
    what: &str,
    result_n: Option<u32>,
    tuples: &[Vec<Arg>],
    reference: impl Fn(&[Arg]) -> Expect,
) -> usize {
    let mut cache = Cache::default();
    for operands in tuples {
        let arguments = operands.iter().map(|arg| arg.val()).collect::<Vec<_>>();
        let at = || format!("{what} at {operands:?}");
        let Outcome::Values(values) = run_cached(&mut cache, case.lowered.function, &arguments)
        else {
            panic!("{}: trapped; no float operation traps", at());
        };
        let [value] = values.as_slice() else {
            panic!("{}: returned {values:?}", at());
        };
        if let Some(n) = result_n {
            assert!(!value.float, "{}: an integer result is {value:?}", at());
            assert_eq!(
                u128::from(value.bits) >> n,
                0,
                "{}: the result {:#x} has bits at or above {n}",
                at(),
                value.bits
            );
        }
        match reference(operands) {
            Expect::Float(expected) => {
                assert!(value.float, "{}: a float result is {value:?}", at());
                assert_eq!(value.width, expected.width(), "{}: the width", at());
                if expected.is_nan() {
                    assert!(value.is_nan(), "{}: {value:?} is not a NaN", at());
                } else {
                    assert_eq!(value.bits, expected.bits(), "{}: wrong value, {value:?}", at());
                }
            }
            Expect::Int(expected) => {
                let n = result_n.expect("an integer result has a width");
                let expected = expected.unsigned() & ((1u128 << n) - 1);
                assert_eq!(u128::from(value.bits), expected, "{}: wrong value", at());
            }
            Expect::Poison => {}
        }
    }
    tuples.len()
}

/// The NaN of `width` bits that stands for "a NaN" in a reference.
fn nan(width: u32) -> Float {
    if width == 32 {
        Float::from_f32(f32::NAN)
    } else {
        Float::from_f64(f64::NAN)
    }
}

/// The reference of `case` on `operands` of the types `types`: the fold, or `on_decline` when it
/// declines.
fn by_fold<'a>(
    context: &'a Rc<Context>,
    case: &Case,
    types: &[&str],
    extend: fn(Int) -> Int,
    on_decline: Expect,
) -> impl Fn(&[Arg]) -> Expect + use<'a> {
    let types = types.iter().map(|ty| String::from(*ty)).collect::<Vec<_>>();
    let op = case.op;
    move |operands: &[Arg]| {
        let attrs = operands
            .iter()
            .zip(&types)
            .map(|(arg, ty)| attribute(context, *arg, ty, extend))
            .collect();
        fold(op, attrs).map_or_else(|| on_decline.clone(), folded)
    }
}

#[test]
fn float_arithmetic_matches_arith() {
    let context = context();
    let mut count = 0;
    for width in [32, 64] {
        let ty = float_name(width);
        let values = floats(width, &specials());
        for name in ["addf", "subf", "mulf", "divf"] {
            let body = format!("arith.{name} %a, %b");
            let case = case(&context, &[ty, ty], ty, &body);
            // The fold declines only for a NaN result.
            let reference = by_fold(&context, &case, &[ty, ty], |v| v, Expect::Float(nan(width)));
            count += check(
                &case,
                &format!("{body} : {ty}"),
                None,
                &product(&[values.clone(), values.clone()]),
                reference,
            );
        }
        let case = case(&context, &[ty], ty, "arith.negf %a");
        let reference = by_fold(&context, &case, &[ty], |v| v, Expect::Poison);
        count += check(&case, "negf", None, &singles(&values), reference);
    }
    assert!(count > 2000, "{count}");
}

#[test]
fn every_float_comparison_matches_the_evaluation_of_its_predicate() {
    let context = context();
    let mut count = 0;
    for predicate in CmpFPredicate::ALL {
        for width in [32, 64] {
            let ty = float_name(width);
            let values = floats(width, &specials());
            let body = format!("arith.cmpf {} %a, %b", predicate.as_str());
            let case = case(&context, &[ty, ty], "i1", &body);
            let float = |arg: Arg| match arg {
                Arg::F32(value) => Float::from_f32(value),
                Arg::F64(value) => Float::from_f64(value),
                Arg::Int(_) => unreachable!("float operands"),
            };
            count += check(
                &case,
                &format!("{body} : {ty}"),
                Some(1),
                &product(&[values.clone(), values]),
                |operands| {
                    Expect::Int(Int::from(
                        predicate.evaluate(float(operands[0]), float(operands[1])),
                    ))
                },
            );
        }
    }
    assert!(count > 10_000, "{count}");
}

#[test]
fn float_to_integer_conversions_match_arith_and_never_trap() {
    let context = context();
    let mut count = 0;
    for ty in types() {
        for width in [32, 64] {
            let from = float_name(width);
            let mut values = specials();
            values.extend(limits(ty.n, width));
            let values = floats(width, &values);
            for name in ["fptosi", "fptoui"] {
                let body = format!("arith.{name} %a <{{ ty = {} }}>", ty.name);
                let case = case(&context, &[from], &ty.name, &body);
                // A declined fold is poison: only carried form and the absence of a trap hold.
                let reference = by_fold(&context, &case, &[from], |v| v, Expect::Poison);
                count += check(
                    &case,
                    &format!("{body} from {from}"),
                    Some(ty.n),
                    &singles(&values),
                    reference,
                );
            }
        }
    }
    assert!(count > 1000, "{count}");
}

/// The integers tried as operands of a conversion to a float at the width `n`: the edge values,
/// and values where `f32` and `f64` round (`2^24 + 1`, `2^24 + 3`, `2^53 + 1`, `2^53 + 3`, and
/// below the top of the width).
fn rounding(n: u32) -> Vec<Int> {
    let mut values = edges(n);
    let max = Int::all_ones(n).unsigned();
    for bits in [
        (1u128 << 24) + 1,
        (1 << 24) + 3,
        (1 << 53) + 1,
        (1 << 53) + 3,
        max.saturating_sub(0x400),
        max >> 1,
    ] {
        let value = Int::from_bits(n, bits);
        if !values.contains(&value) {
            values.push(value);
        }
    }
    values
}

#[test]
fn integer_to_float_conversions_match_arith() {
    let context = context();
    let mut count = 0;
    for ty in types() {
        let values = rounding(ty.n).into_iter().map(Arg::Int).collect::<Vec<_>>();
        for width in [32, 64] {
            let to = float_name(width);
            for (name, extend) in [
                ("sitofp", (|v: Int| v.sext(64)) as fn(Int) -> Int),
                ("uitofp", |v: Int| v.zext(64)),
            ] {
                let body = format!("arith.{name} %a <{{ ty = {to} }}>");
                let case = case(&context, &[&ty.name], to, &body);
                let declined = |operands: &[Arg]| -> Expect {
                    panic!("the fold of `{body}` declined on {operands:?}")
                };
                let reference = by_fold(&context, &case, &[&ty.name], extend, Expect::Poison);
                count += check(
                    &case,
                    &format!("{body} from {}", ty.name),
                    None,
                    &singles(&values),
                    |operands| match reference(operands) {
                        Expect::Poison => declined(operands),
                        expected => expected,
                    },
                );
            }
        }
    }
    assert!(count > 300, "{count}");
}

#[test]
fn float_extension_truncation_and_bitcasts_match_arith() {
    let context = context();
    let mut count = 0;
    let mut wide = specials();
    wide.extend([0.1, 1.0e300, -1.0e-300, f64::from(f32::MAX) * 1.5, 3.4028235677973366e38]);
    let fpext = case(&context, &["f32"], "f64", "arith.fpext %a <{ ty = f64 }>");
    let reference = by_fold(&context, &fpext, &["f32"], |v| v, Expect::Float(nan(64)));
    count += check(&fpext, "fpext", None, &singles(&floats(32, &wide)), reference);
    let fptrunc = case(&context, &["f64"], "f32", "arith.fptrunc %a <{ ty = f32 }>");
    let reference = by_fold(&context, &fptrunc, &["f64"], |v| v, Expect::Float(nan(32)));
    count += check(&fptrunc, "fptrunc", None, &singles(&floats(64, &wide)), reference);
    for width in [32, 64] {
        let float = float_name(width);
        let int = Ty::int(width);
        let to_int = case(
            &context,
            &[float],
            &int.name,
            &format!("arith.bitcast %a <{{ ty = {} }}>", int.name),
        );
        let reference = by_fold(&context, &to_int, &[float], |v| v, Expect::Poison);
        count += check(
            &to_int,
            "bitcast to int",
            Some(width),
            &singles(&floats(width, &wide)),
            reference,
        );
        let to_float =
            case(&context, &[&int.name], float, &format!("arith.bitcast %a <{{ ty = {float} }}>"));
        let reference = by_fold(&context, &to_float, &[&int.name], |v| v, Expect::Poison);
        let mut bits = edges(width);
        bits.extend(
            [0x7fc0_0001u128, 0xff80_0000, 0x7ff0_0000_0000_0001].map(|b| Int::from_bits(width, b)),
        );
        let bits = bits.into_iter().map(Arg::Int).collect::<Vec<_>>();
        count += check(&to_float, "bitcast to float", None, &singles(&bits), reference);
    }
    assert!(count > 100, "{count}");
}
