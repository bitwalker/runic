//! The patterns of the integer operations of `arith` that are not one Wasm instruction and an
//! adjustment: the overflow family (`*_overflowing`, and `add`, `sub`, `mul` in a `checked` mode),
//! `min` and `max`, `udivrem` and `sdivrem`, rotates and bit counts; and the pattern of
//! `cf.assert`.
//!
//! As in [super::int], each pattern reads the original type of its operation (`iN`, or `isize`,
//! which is 32 bits), builds `wasm` operations on the carriers of the operands, and keeps carried
//! form: every integer result is zero-extended to `N` bits, the carried `i1` of an overflow flag
//! included. `W` below is the width of the carrier, 32 or 64. No pattern introduces a trap: every
//! division it adds has a divisor that cannot be zero, and the only traps are those `arith` has
//! (a `checked` overflow, a failed assertion, a division by zero, the minimum divided by -1 in a
//! signed division).
//!
//! **The overflow flag.** `o` is 1 exactly when the exact result, with the operands read as the
//! operation's signedness says, does not fit in `N` bits read the same way. It is computed per
//! class of `(operation, signedness, N, W)`:
//!
//! * a sum or a difference with `N < W` is exact in the carrier (of zero-extended operands for
//!   unsigned, of operands sign-extended from `N` for signed), and is compared with the range;
//! * a sum or a difference with `N = W` uses the sign-bit identities;
//! * a product with `2N <= W` is exact in the carrier; one with `W = 32` and `2N > 32` is exact in
//!   an `i64`;
//! * a product with `W = 64` and `N > 32` is not exact in any Wasm type, and is tested by a
//!   division of the largest magnitude that fits by the magnitude of one operand, whose divisor is
//!   made 1 when it is 0, so it cannot trap.
//!
//! `cf.assert %c, "message"` becomes a `wasm.if` on the condition that traps
//! (`wasm.unreachable`) when it is 0. The message is dropped: a failed assertion is a bare trap.

use alloc::vec::Vec;

use runic_dialect_arith as arith;
use runic_dialect_cf as cf;
use runic_dialect_wasm as wasm;
use runic_ir::{
    BuildableOp, Op, OpRegistration, OperationRef, Report, SmallVec, SourceSpan, Spanned, Type,
    ValueRef,
    conversion::{
        ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConvertedOperands,
        TypeConverter,
    },
    dialects::builtin::attributes::{Overflow, Signedness},
    patterns::{Pattern, PatternInfo},
};

use crate::{
    carried::{self, const_int, mask, sign_extend},
    convert::calls::{pattern_info, registered},
    types::{carrier_width, width},
};

/// Adds the patterns of this module.
pub fn populate(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
) -> Result<(), Report> {
    let context = patterns.context();
    let rules: [(Rule, &'static str); 18] = [
        (Rule::Overflowing(Arith::Add), "wasm-arith-add-overflowing"),
        (Rule::Overflowing(Arith::Sub), "wasm-arith-sub-overflowing"),
        (Rule::Overflowing(Arith::Mul), "wasm-arith-mul-overflowing"),
        (Rule::Checked(Arith::Add), "wasm-arith-add-checked"),
        (Rule::Checked(Arith::Sub), "wasm-arith-sub-checked"),
        (Rule::Checked(Arith::Mul), "wasm-arith-mul-checked"),
        (Rule::Min, "wasm-arith-min"),
        (Rule::Max, "wasm-arith-max"),
        (Rule::Udivrem, "wasm-arith-udivrem"),
        (Rule::Sdivrem, "wasm-arith-sdivrem"),
        (Rule::Rotl, "wasm-arith-rotl"),
        (Rule::Rotr, "wasm-arith-rotr"),
        (Rule::Popcnt, "wasm-arith-popcnt"),
        (Rule::Clz, "wasm-arith-clz"),
        (Rule::Ctz, "wasm-arith-ctz"),
        (Rule::Clo, "wasm-arith-clo"),
        (Rule::Cto, "wasm-arith-cto"),
        (Rule::Assert, "wasm-cf-assert"),
    ];
    for (rule, name) in rules {
        let info = match rule {
            Rule::Overflowing(Arith::Add) => info::<arith::AddOverflowing>(&context, name, rule),
            Rule::Overflowing(Arith::Sub) => info::<arith::SubOverflowing>(&context, name, rule),
            Rule::Overflowing(Arith::Mul) => info::<arith::MulOverflowing>(&context, name, rule),
            Rule::Checked(Arith::Add) => info::<arith::Add>(&context, name, rule),
            Rule::Checked(Arith::Sub) => info::<arith::Sub>(&context, name, rule),
            Rule::Checked(Arith::Mul) => info::<arith::Mul>(&context, name, rule),
            Rule::Min => info::<arith::Min>(&context, name, rule),
            Rule::Max => info::<arith::Max>(&context, name, rule),
            Rule::Udivrem => info::<arith::Udivrem>(&context, name, rule),
            Rule::Sdivrem => info::<arith::Sdivrem>(&context, name, rule),
            Rule::Rotl => info::<arith::Rotl>(&context, name, rule),
            Rule::Rotr => info::<arith::Rotr>(&context, name, rule),
            Rule::Popcnt => info::<arith::Popcnt>(&context, name, rule),
            Rule::Clz => info::<arith::Clz>(&context, name, rule),
            Rule::Ctz => info::<arith::Ctz>(&context, name, rule),
            Rule::Clo => info::<arith::Clo>(&context, name, rule),
            Rule::Cto => info::<arith::Cto>(&context, name, rule),
            Rule::Assert => info::<cf::Assert>(&context, name, rule),
        };
        patterns.push(ComposedPattern {
            info,
            converter: converter.clone(),
            rule,
        });
    }
    Ok(())
}

/// The information of the pattern of `rule`, rooted at `T`.
fn info<T: OpRegistration>(
    context: &alloc::rc::Rc<runic_ir::Context>,
    name: &'static str,
    rule: Rule,
) -> PatternInfo {
    pattern_info::<T>(context, name, rule.generated(context))
}

/// The arithmetic of an operation of the overflow family.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Arith {
    Add,
    Sub,
    Mul,
}

/// What each pattern of this module lowers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Rule {
    /// `arith.add_overflowing`, `sub_overflowing`, `mul_overflowing`: the flag and the wrapped
    /// result of [overflowing].
    Overflowing(Arith),
    /// `arith.add`, `sub`, `mul` in a `checked` mode: [overflowing], then a `wasm.if` on the flag
    /// whose first region is `wasm.unreachable`; the result is the wrapped one. An operation in
    /// another mode is left to [super::int].
    Checked(Arith),
    /// `arith.min`: `wasm.select` of the operands on `lt_u`, or on `lt_s` of the operands
    /// sign-extended from `N`. The result is one of the operands, so it needs no mask.
    Min,
    /// `arith.max`: as [Rule::Min], on `gt_u` or `gt_s`.
    Max,
    /// `arith.udivrem`: `wasm.div_u` and `wasm.rem_u`; a zero divisor traps in both.
    Udivrem,
    /// `arith.sdivrem`: both operands sign-extended from `N`, `wasm.div_s` and `wasm.rem_s`, each
    /// masked. The minimum divided by -1 traps, as in `arith`: at `N = W` `wasm.div_s` traps, and
    /// when `N < W` the quotient (`2^(N-1)`, which the carrier holds) is tested before the mask
    /// and traps when it does not fit in `N` bits ([trap_unless_quotient_fits]).
    Sdivrem,
    /// `arith.rotl`: `wasm.rotl` when `N = W`; the operand itself when `N = 1`; otherwise the
    /// amount modulo `N` (`k`), then `mask((x << k) | (x >> (N - k)))`. An amount of 0 shifts `x`
    /// right by `N`, which gives 0 for a zero-extended `x`.
    Rotl,
    /// `arith.rotr`: as [Rule::Rotl], mirrored: `mask((x >> k) | (x << (N - k)))`.
    Rotr,
    /// `arith.popcnt`: `wasm.popcnt`; the bits above `N` are clear.
    Popcnt,
    /// `arith.clz`: `wasm.clz`, less the `W - N` clear bits above `N`.
    Clz,
    /// `arith.ctz`: `wasm.ctz`, of `x | 2^N` when `N < W`, so that a zero gives `N`.
    Ctz,
    /// `arith.clo`: [Rule::Clz] of `x xor (2^N - 1)`.
    Clo,
    /// `arith.cto`: [Rule::Ctz] of `x xor (2^N - 1)`; when `N < W` the `| 2^N` is folded into the
    /// constant: `x xor (2^(N+1) - 1)`, as bit `N` of `x` is clear.
    Cto,
    /// `cf.assert`: `wasm.if %c then { wasm.end } else { wasm.unreachable }`. The message is
    /// dropped: a failed assertion is a bare trap.
    Assert,
}

impl Rule {
    /// The names of the operations a pattern of this rule may create.
    fn generated(self, context: &alloc::rc::Rc<runic_ir::Context>) -> Vec<runic_ir::OperationName> {
        macro_rules! names {
            ($($op:ty),*) => { alloc::vec![$(registered::<$op>(context)),*] };
        }
        let extensions = names!(
            wasm::Const,
            wasm::Shl,
            wasm::ShrS,
            wasm::Extend8S,
            wasm::Extend16S,
            wasm::Extend32S,
            wasm::And
        );
        let mut names = match self {
            Rule::Overflowing(_) | Rule::Checked(_) => names!(
                wasm::Add,
                wasm::Sub,
                wasm::Mul,
                wasm::Xor,
                wasm::Eq,
                wasm::Ne,
                wasm::Eqz,
                wasm::LtS,
                wasm::LtU,
                wasm::GtU,
                wasm::Select,
                wasm::DivU,
                wasm::ExtendS,
                wasm::ExtendU,
                wasm::Wrap,
                wasm::If,
                wasm::End,
                wasm::Unreachable
            ),
            Rule::Min | Rule::Max => {
                names!(wasm::Select, wasm::LtU, wasm::LtS, wasm::GtU, wasm::GtS)
            }
            Rule::Udivrem => names!(wasm::DivU, wasm::RemU),
            Rule::Sdivrem => {
                names!(wasm::DivS, wasm::RemS, wasm::Ne, wasm::If, wasm::End, wasm::Unreachable)
            }
            Rule::Rotl | Rule::Rotr => {
                names!(wasm::Rotl, wasm::Rotr, wasm::RemU, wasm::ShrU, wasm::Sub, wasm::Or)
            }
            Rule::Popcnt => names!(wasm::Popcnt),
            Rule::Clz | Rule::Clo => names!(wasm::Clz, wasm::Sub, wasm::Xor),
            Rule::Ctz | Rule::Cto => names!(wasm::Ctz, wasm::Or, wasm::Xor),
            Rule::Assert => names!(wasm::If, wasm::End, wasm::Unreachable),
        };
        names.extend(extensions);
        names
    }
}

/// The pattern of one [Rule].
struct ComposedPattern {
    info: PatternInfo,
    converter: TypeConverter,
    rule: Rule,
}

impl Pattern for ComposedPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

/// What a pattern reads of its operation before it rewrites it.
struct Source {
    /// The type of the first operand.
    operand_ty: Type,
    /// The signedness the operation reads its operands with, if it has one: that of an
    /// overflowing operation, of `min` or `max`, or of a `checked` mode.
    signedness: Option<Signedness>,
    /// Whether the operation is one this module lowers: for `add`, `sub`, `mul`, whether it is in
    /// a `checked` mode.
    supported: bool,
    span: SourceSpan,
}

impl Source {
    /// Reads `op`, which `rule` is rooted at.
    fn of(rule: Rule, op: OperationRef) -> Self {
        let op_ref = op.borrow();
        let span = op_ref.span();
        let operand_ty = op_ref.operands()[0].borrow().as_value_ref().borrow().ty();
        let overflow = match rule {
            Rule::Checked(Arith::Add) => {
                op_ref.downcast_ref::<arith::Add>().map(|op| *op.get_overflow())
            }
            Rule::Checked(Arith::Sub) => {
                op_ref.downcast_ref::<arith::Sub>().map(|op| *op.get_overflow())
            }
            Rule::Checked(Arith::Mul) => {
                op_ref.downcast_ref::<arith::Mul>().map(|op| *op.get_overflow())
            }
            _ => None,
        };
        let checked = match overflow {
            Some(Overflow::Checked(signedness)) => Some(signedness),
            _ => None,
        };
        let signedness = match rule {
            Rule::Checked(_) => checked,
            Rule::Overflowing(Arith::Add) => {
                op_ref.downcast_ref::<arith::AddOverflowing>().map(|op| *op.get_signedness())
            }
            Rule::Overflowing(Arith::Sub) => {
                op_ref.downcast_ref::<arith::SubOverflowing>().map(|op| *op.get_signedness())
            }
            Rule::Overflowing(Arith::Mul) => {
                op_ref.downcast_ref::<arith::MulOverflowing>().map(|op| *op.get_signedness())
            }
            Rule::Min => op_ref.downcast_ref::<arith::Min>().map(|op| *op.get_signedness()),
            Rule::Max => op_ref.downcast_ref::<arith::Max>().map(|op| *op.get_signedness()),
            _ => None,
        };
        Self {
            operand_ty,
            signedness,
            supported: !matches!(rule, Rule::Checked(_)) || checked.is_some(),
            span,
        }
    }
}

/// The value of the one result of the new operation `T` of `a` and `b`.
fn binary<T>(
    rw: &mut ConversionPatternRewriter,
    a: ValueRef,
    b: ValueRef,
    span: SourceSpan,
) -> Result<ValueRef, Report>
where
    T: Op + BuildableOp<(ValueRef, ValueRef)>,
{
    Ok(carried::result(&rw.create_op::<T, _>(span, (a, b))?))
}

/// The value of the one result of the new operation `T` of `value`.
fn unary<T>(
    rw: &mut ConversionPatternRewriter,
    value: ValueRef,
    span: SourceSpan,
) -> Result<ValueRef, Report>
where
    T: Op + BuildableOp<(ValueRef,)>,
{
    Ok(carried::result(&rw.create_op::<T, _>(span, (value,))?))
}

/// The value of the one result of the new conversion `T` of `value` to the type `ty`.
fn convert<T>(
    rw: &mut ConversionPatternRewriter,
    value: ValueRef,
    ty: Type,
    span: SourceSpan,
) -> Result<ValueRef, Report>
where
    T: Op + BuildableOp<(ValueRef, Type)>,
{
    Ok(carried::result(&rw.create_op::<T, _>(span, (value, ty))?))
}

/// `2^n - 1`, as the bits of a constant: the low `n` bits set.
fn ones(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// The carrier type of `value` and its width.
fn carrier_of(value: ValueRef) -> (Type, u32) {
    let ty = value.borrow().ty();
    (ty, carrier_width(ty).unwrap_or(32))
}

/// The overflow flag `o` (an `i32` 0 or 1) and the wrapped result `r` (carried) of `arith` on the
/// carried `N`-bit integers `a` and `b`, read as signed or unsigned (`signed`).
///
/// Each class, and why it is exact and cannot trap:
///
/// * **Unsigned sum, `N < W`.** `s = a + b` is below `2^(N+1)`, exact in the carrier;
///   `o = s > 2^N - 1`, `r = mask(s)`.
/// * **Unsigned sum, `N = W`.** `r = a + b` wraps; the exact sum overflows exactly when the
///   wrapped one is below an operand: `o = r <u a`.
/// * **Signed sum or difference, `N < W`.** The operands sign-extended from `N` are in
///   `[-2^(N-1), 2^(N-1))`, so `s` is in `[-2^N, 2^N]`, exact in `W > N` bits; it fits exactly when
///   it equals itself sign-extended from `N`: `o = s != sext(s)`, `r = mask(s)`.
/// * **Signed sum, `N = W`.** `r = a + b` wraps; it overflows exactly when both operands have the
///   sign `r` does not: `o = ((r ^ a) & (r ^ b)) <s 0`.
/// * **Unsigned difference, any `N`.** It does not fit exactly when it is negative:
///   `o = a <u b` (both zero-extended), `r = mask(a - b)`.
/// * **Signed difference, `N = W`.** It overflows exactly when the operands' signs differ and
///   `r`'s differs from `a`'s: `o = ((a ^ b) & (a ^ r)) <s 0`.
/// * **Product, `2N <= W`.** Exact in the carrier: unsigned, `p < 2^(2N)`, and
///   `o = p >u 2^N - 1`; signed (operands sign-extended), `|p| <= 2^(2N-2)`, and
///   `o = p != sext(p)`. `r = mask(p)`.
/// * **Product, `W = 32`, `2N > 32`.** Both operands extended to `i64` (zero or sign, after the
///   sign extension from `N`): the product is exact in 64 bits (`N <= 32`), tested as above in the
///   `i64`; `r = mask(wrap(p))`.
/// * **Product, `W = 64`, `N > 32`.** No Wasm type holds the exact product. With `|a|`, `|b|` the
///   magnitudes (the operands themselves for unsigned; for signed, of the operands sign-extended
///   from `N`, as unsigned 64-bit numbers, so `|-2^63| = 2^63`), and `L` the largest magnitude
///   that fits (`2^N - 1` unsigned; signed, `2^(N-1) - 1` when the signs agree and `2^(N-1)` when
///   they differ: `L = 2^(N-1) - 1 - (sa ^ sb)` with `sa`, `sb` the sign masks), the product
///   fits exactly when `|a| * |b| <= L`, which for `|a| >= 1` is `|b| <= floor(L / |a|)`. The
///   divisor is `|a|`, or 1 when `|a| = 0` (a `select` on `eqz`), so `div_u` cannot trap, and a
///   zero `|a|` gives `L`, which `|b|` never exceeds: `o = |b| >u L / max(|a|, 1)`.
///   `r = mask(a * b)`.
fn overflowing(
    rw: &mut ConversionPatternRewriter,
    arith: Arith,
    signed: bool,
    n: u32,
    a: ValueRef,
    b: ValueRef,
    span: SourceSpan,
) -> Result<(ValueRef, ValueRef), Report> {
    let (carrier, w) = carrier_of(a);
    let i32 = rw.context().i32();
    let i64 = rw.context().i64();
    // `o` of an exact value `s` in `ty`, the signed range test, and the wrapped `r`.
    let signed_fit =
        |rw: &mut ConversionPatternRewriter, s: ValueRef| -> Result<ValueRef, Report> {
            let extended = sign_extend(rw, s, n, span)?;
            binary::<wasm::Ne>(rw, s, extended, span)
        };
    let sign_test = |rw: &mut ConversionPatternRewriter, t: ValueRef| -> Result<ValueRef, Report> {
        let zero = const_int(rw, 0, carrier, span)?;
        binary::<wasm::LtS>(rw, t, zero, span)
    };
    match (arith, signed) {
        (Arith::Add, false) if n < w => {
            let s = binary::<wasm::Add>(rw, a, b, span)?;
            // The constant of the range is the mask too.
            let top = const_int(rw, ones(n), carrier, span)?;
            let o = binary::<wasm::GtU>(rw, s, top, span)?;
            Ok((o, binary::<wasm::And>(rw, s, top, span)?))
        }
        (Arith::Add, false) => {
            let r = binary::<wasm::Add>(rw, a, b, span)?;
            let o = binary::<wasm::LtU>(rw, r, a, span)?;
            Ok((o, r))
        }
        (Arith::Add | Arith::Sub, true) if n < w => {
            let sa = sign_extend(rw, a, n, span)?;
            let sb = sign_extend(rw, b, n, span)?;
            let s = if arith == Arith::Add {
                binary::<wasm::Add>(rw, sa, sb, span)?
            } else {
                binary::<wasm::Sub>(rw, sa, sb, span)?
            };
            let o = signed_fit(rw, s)?;
            Ok((o, mask(rw, s, n, span)?))
        }
        (Arith::Add, true) => {
            let r = binary::<wasm::Add>(rw, a, b, span)?;
            let ra = binary::<wasm::Xor>(rw, r, a, span)?;
            let rb = binary::<wasm::Xor>(rw, r, b, span)?;
            let t = binary::<wasm::And>(rw, ra, rb, span)?;
            Ok((sign_test(rw, t)?, r))
        }
        (Arith::Sub, false) => {
            let o = binary::<wasm::LtU>(rw, a, b, span)?;
            let difference = binary::<wasm::Sub>(rw, a, b, span)?;
            Ok((o, mask(rw, difference, n, span)?))
        }
        (Arith::Sub, true) => {
            let r = binary::<wasm::Sub>(rw, a, b, span)?;
            let ab = binary::<wasm::Xor>(rw, a, b, span)?;
            let ar = binary::<wasm::Xor>(rw, a, r, span)?;
            let t = binary::<wasm::And>(rw, ab, ar, span)?;
            Ok((sign_test(rw, t)?, r))
        }
        (Arith::Mul, _) if 2 * n <= w => {
            let (a, b) = if signed {
                (sign_extend(rw, a, n, span)?, sign_extend(rw, b, n, span)?)
            } else {
                (a, b)
            };
            let p = binary::<wasm::Mul>(rw, a, b, span)?;
            if signed {
                let o = signed_fit(rw, p)?;
                return Ok((o, mask(rw, p, n, span)?));
            }
            // `2N <= W` means `N < W`: the constant of the range is the mask too.
            let top = const_int(rw, ones(n), carrier, span)?;
            let o = binary::<wasm::GtU>(rw, p, top, span)?;
            Ok((o, binary::<wasm::And>(rw, p, top, span)?))
        }
        (Arith::Mul, _) if w == 32 => {
            let (a, b) = if signed {
                let sa = sign_extend(rw, a, n, span)?;
                let sb = sign_extend(rw, b, n, span)?;
                (
                    convert::<wasm::ExtendS>(rw, sa, i64, span)?,
                    convert::<wasm::ExtendS>(rw, sb, i64, span)?,
                )
            } else {
                (
                    convert::<wasm::ExtendU>(rw, a, i64, span)?,
                    convert::<wasm::ExtendU>(rw, b, i64, span)?,
                )
            };
            let p = binary::<wasm::Mul>(rw, a, b, span)?;
            let o = if signed {
                signed_fit(rw, p)?
            } else {
                let top = const_int(rw, ones(n), i64, span)?;
                binary::<wasm::GtU>(rw, p, top, span)?
            };
            let low = convert::<wasm::Wrap>(rw, p, i32, span)?;
            Ok((o, mask(rw, low, n, span)?))
        }
        (Arith::Mul, false) => {
            let top = const_int(rw, ones(n), carrier, span)?;
            let o = fits_by_division(rw, top, a, b, span)?;
            let p = binary::<wasm::Mul>(rw, a, b, span)?;
            Ok((o, mask(rw, p, n, span)?))
        }
        (Arith::Mul, true) => {
            let sa = sign_extend(rw, a, n, span)?;
            let sb = sign_extend(rw, b, n, span)?;
            let shift = const_int(rw, 63, carrier, span)?;
            let sign_a = binary::<wasm::ShrS>(rw, sa, shift, span)?;
            let sign_b = binary::<wasm::ShrS>(rw, sb, shift, span)?;
            let magnitude = |rw: &mut ConversionPatternRewriter,
                             value: ValueRef,
                             sign: ValueRef|
             -> Result<ValueRef, Report> {
                let flipped = binary::<wasm::Xor>(rw, value, sign, span)?;
                binary::<wasm::Sub>(rw, flipped, sign, span)
            };
            let abs_a = magnitude(rw, sa, sign_a)?;
            let abs_b = magnitude(rw, sb, sign_b)?;
            let signs = binary::<wasm::Xor>(rw, sign_a, sign_b, span)?;
            let max = const_int(rw, ones(n - 1), carrier, span)?;
            let limit = binary::<wasm::Sub>(rw, max, signs, span)?;
            let o = fits_by_division(rw, limit, abs_a, abs_b, span)?;
            let p = binary::<wasm::Mul>(rw, a, b, span)?;
            Ok((o, mask(rw, p, n, span)?))
        }
    }
}

/// `b >u limit / max(a, 1)`, all unsigned in one carrier: whether the product of `a` and `b` is
/// above `limit`. The divisor is `a`, or 1 when `a` is 0, so the division cannot trap.
fn fits_by_division(
    rw: &mut ConversionPatternRewriter,
    limit: ValueRef,
    a: ValueRef,
    b: ValueRef,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    let (carrier, _) = carrier_of(a);
    let zero = unary::<wasm::Eqz>(rw, a, span)?;
    let one = const_int(rw, 1, carrier, span)?;
    let divisor = carried::result(&rw.create_op::<wasm::Select, _>(span, (zero, one, a))?);
    let quotient = binary::<wasm::DivU>(rw, limit, divisor, span)?;
    binary::<wasm::GtU>(rw, b, quotient, span)
}

/// A `wasm.if` on `condition` with no inputs and no results, whose first region ends in a trap
/// (`trap_first`) and the second in `wasm.end`, or the other way round. The insertion point is
/// before `op` after it.
fn trap_unless(
    rw: &mut ConversionPatternRewriter,
    condition: ValueRef,
    trap_first: bool,
    op: OperationRef,
    span: SourceSpan,
) -> Result<(), Report> {
    let if_op = rw.create_op::<wasm::If, _>(span, (condition, Vec::<ValueRef>::new()))?;
    let (then_region, else_region) = {
        let if_ref = if_op.borrow();
        (if_ref.then_body().as_region_ref(), if_ref.else_body().as_region_ref())
    };
    for (region, trap) in [(then_region, trap_first), (else_region, !trap_first)] {
        rw.create_block(region, &[])?;
        if trap {
            rw.create_op::<wasm::Unreachable, _>(span, ())?;
        } else {
            rw.create_op::<wasm::End, _>(span, (Vec::<ValueRef>::new(),))?;
        }
    }
    rw.set_insertion_point_before(op);
    Ok(())
}

/// Traps when `quotient`, the quotient of a `wasm.div_s` of two operands sign-extended from `n`
/// bits in their carrier, does not fit in `n` signed bits: `quotient != sext_n(quotient)`, which
/// holds only for the minimum divided by -1 (the quotient `2^(n-1)`), where `arith.sdiv` and
/// `arith.sdivrem` trap. Nothing when `n` is the carrier's width, where `wasm.div_s` traps itself.
/// The insertion point is before `op` after it.
pub(super) fn trap_unless_quotient_fits(
    rw: &mut ConversionPatternRewriter,
    quotient: ValueRef,
    n: u32,
    op: OperationRef,
    span: SourceSpan,
) -> Result<(), Report> {
    let (_, w) = carrier_of(quotient);
    if n >= w {
        return Ok(());
    }
    let extended = sign_extend(rw, quotient, n, span)?;
    let unfit = binary::<wasm::Ne>(rw, quotient, extended, span)?;
    trap_unless(rw, unfit, true, op, span)
}

/// The `i32` 0 or 1 of comparing `a` and `b`, of width `n`: `less` or greater, `signed` or
/// unsigned (the operands sign-extended from `n` first).
fn order(
    rw: &mut ConversionPatternRewriter,
    less: bool,
    signed: bool,
    n: u32,
    a: ValueRef,
    b: ValueRef,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    if signed {
        let a = sign_extend(rw, a, n, span)?;
        let b = sign_extend(rw, b, n, span)?;
        if less {
            binary::<wasm::LtS>(rw, a, b, span)
        } else {
            binary::<wasm::GtS>(rw, a, b, span)
        }
    } else if less {
        binary::<wasm::LtU>(rw, a, b, span)
    } else {
        binary::<wasm::GtU>(rw, a, b, span)
    }
}

/// `x` rotated by `amount` modulo `n`, left or right (`left`), both carried in `W` bits.
fn rotate(
    rw: &mut ConversionPatternRewriter,
    left: bool,
    n: u32,
    x: ValueRef,
    amount: ValueRef,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    let (carrier, w) = carrier_of(x);
    if n >= w {
        return if left {
            binary::<wasm::Rotl>(rw, x, amount, span)
        } else {
            binary::<wasm::Rotr>(rw, x, amount, span)
        };
    }
    if n == 1 {
        return Ok(x);
    }
    let width = const_int(rw, u64::from(n), carrier, span)?;
    let k = if n.is_power_of_two() {
        let low = const_int(rw, u64::from(n - 1), carrier, span)?;
        binary::<wasm::And>(rw, amount, low, span)?
    } else {
        binary::<wasm::RemU>(rw, amount, width, span)?
    };
    let rest = binary::<wasm::Sub>(rw, width, k, span)?;
    let (high, low) = if left {
        (binary::<wasm::Shl>(rw, x, k, span)?, binary::<wasm::ShrU>(rw, x, rest, span)?)
    } else {
        (binary::<wasm::Shl>(rw, x, rest, span)?, binary::<wasm::ShrU>(rw, x, k, span)?)
    };
    let rotated = binary::<wasm::Or>(rw, high, low, span)?;
    mask(rw, rotated, n, span)
}

/// The bit count of `rule` (`popcnt`, `clz`, `ctz`, `clo`, `cto`) of the `n`-bit `x`.
fn count(
    rw: &mut ConversionPatternRewriter,
    rule: Rule,
    n: u32,
    x: ValueRef,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    let (carrier, w) = carrier_of(x);
    match rule {
        Rule::Popcnt => unary::<wasm::Popcnt>(rw, x, span),
        Rule::Clz | Rule::Clo => {
            let x = if rule == Rule::Clo {
                let all = const_int(rw, ones(n), carrier, span)?;
                binary::<wasm::Xor>(rw, x, all, span)?
            } else {
                x
            };
            let zeros = unary::<wasm::Clz>(rw, x, span)?;
            if n >= w {
                return Ok(zeros);
            }
            let above = const_int(rw, u64::from(w - n), carrier, span)?;
            binary::<wasm::Sub>(rw, zeros, above, span)
        }
        Rule::Ctz => {
            let x = if n < w {
                let stop = const_int(rw, 1u64 << n, carrier, span)?;
                binary::<wasm::Or>(rw, x, stop, span)?
            } else {
                x
            };
            unary::<wasm::Ctz>(rw, x, span)
        }
        _ => {
            // `cto`: the complement in `N` bits, with bit `N` set when `N < W`.
            let bits = if n < w { ones(n + 1) } else { ones(n) };
            let all = const_int(rw, bits, carrier, span)?;
            let flipped = binary::<wasm::Xor>(rw, x, all, span)?;
            unary::<wasm::Ctz>(rw, flipped, span)
        }
    }
}

impl ComposedPattern {
    /// The values an operation of `self.rule` on `values` is replaced by, or `None` where the
    /// pattern does not apply.
    fn lower(
        &self,
        rw: &mut ConversionPatternRewriter,
        op: OperationRef,
        source: &Source,
        values: &[ValueRef],
    ) -> Result<Option<SmallVec<[ValueRef; 2]>>, Report> {
        let span = source.span;
        let rule = self.rule;
        let Some(&a) = values.first() else {
            return Ok(None);
        };
        if rule == Rule::Assert {
            trap_unless(rw, a, false, op, span)?;
            return Ok(Some(SmallVec::new()));
        }
        let Some(n) = width(source.operand_ty) else {
            return Ok(None);
        };
        let signed = source.signedness == Some(Signedness::Signed);
        let b = values.get(1).copied();
        let results: SmallVec<[ValueRef; 2]> = match (rule, b) {
            (Rule::Overflowing(arith), Some(b)) => {
                let (o, r) = overflowing(rw, arith, signed, n, a, b, span)?;
                [o, r].into_iter().collect()
            }
            (Rule::Checked(arith), Some(b)) => {
                let (o, r) = overflowing(rw, arith, signed, n, a, b, span)?;
                trap_unless(rw, o, true, op, span)?;
                [r].into_iter().collect()
            }
            (Rule::Min | Rule::Max, Some(b)) => {
                let condition = order(rw, rule == Rule::Min, signed, n, a, b, span)?;
                let select = rw.create_op::<wasm::Select, _>(span, (condition, a, b))?;
                [carried::result(&select)].into_iter().collect()
            }
            (Rule::Udivrem, Some(b)) => {
                let quotient = binary::<wasm::DivU>(rw, a, b, span)?;
                let remainder = binary::<wasm::RemU>(rw, a, b, span)?;
                [quotient, remainder].into_iter().collect()
            }
            (Rule::Sdivrem, Some(b)) => {
                let sa = sign_extend(rw, a, n, span)?;
                let sb = sign_extend(rw, b, n, span)?;
                let quotient = binary::<wasm::DivS>(rw, sa, sb, span)?;
                trap_unless_quotient_fits(rw, quotient, n, op, span)?;
                let quotient = mask(rw, quotient, n, span)?;
                let remainder = binary::<wasm::RemS>(rw, sa, sb, span)?;
                let remainder = mask(rw, remainder, n, span)?;
                [quotient, remainder].into_iter().collect()
            }
            (Rule::Rotl | Rule::Rotr, Some(b)) => {
                [rotate(rw, rule == Rule::Rotl, n, a, b, span)?].into_iter().collect()
            }
            (Rule::Popcnt | Rule::Clz | Rule::Ctz | Rule::Clo | Rule::Cto, None) => {
                [count(rw, rule, n, a, span)?].into_iter().collect()
            }
            _ => return Ok(None),
        };
        Ok(Some(results))
    }
}

impl ConversionPattern for ComposedPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let source = Source::of(self.rule, op);
        if !source.supported {
            return Ok(false);
        }
        let Some(values) = operands.all_one() else {
            return Ok(false);
        };
        let values: SmallVec<[ValueRef; 4]> = values;
        rewriter.set_insertion_point_before(op);
        let Some(results) = self.lower(rewriter, op, &source, &values)? else {
            return Ok(false);
        };
        if results.is_empty() {
            rewriter.erase_op(op)?;
        } else {
            rewriter.replace_op(op, &results)?;
        }
        Ok(true)
    }
}
