//! The patterns of the integer operations of `arith`, of `cf.select` and of `ub.poison`.
//!
//! Each pattern reads the *original* type of its operation (`iN`, or `isize`, which is 32 bits),
//! builds `wasm` operations on the carriers of the operands, and keeps carried form: the result is
//! zero-extended to `N` bits. `W` below is the width of the carrier, 32 or 64.
//!
//! Where a Wasm instruction does not give what `arith` says, it refines it: `arith` leaves the
//! result undefined or the operation trapping, and any behaviour that picks one of the allowed
//! outcomes is a valid lowering (an operation that would trap may be removed, and a trap is
//! never introduced).
//!
//! * **Shifts.** An amount at or above `N` is undefined in `arith`; Wasm takes the amount modulo
//!   `W`. Any result is allowed, but the invariant must hold still, so `shl` and `ashr` are masked
//!   (`shr` of a zero-extended value cannot set a bit above `N`).
//! * **`sdiv`.** `arith.sdiv` traps on the minimum value divided by -1. When `N < W` the operands
//!   are sign-extended, `wasm.div_s` does not trap, and the masked quotient is the minimum value;
//!   not trapping is one of the outcomes `arith` allows. For `N = W` Wasm traps too. A division by
//!   zero traps in both.
//! * **Overflow.** `unchecked` overflow is undefined, so the wrapping operation refines it. A
//!   `checked` operation is lowered by the patterns of [composed](super::composed).
//! * **`srem`.** The minimum value remainder -1 is 0 in both.
//!
//! The float operations are lowered by [float](super::float); the `checked` modes, the
//! `_overflowing` family, `min`/`max`, `udivrem`/`sdivrem`, the rotates and the bit counts by
//! [composed](super::composed).

use alloc::vec::Vec;

use runic_dialect_arith as arith;
use runic_dialect_cf as cf;
use runic_dialect_ub as ub;
use runic_dialect_wasm as wasm;
use runic_ir::{
    BuildableOp, Op, OpRegistration, OperationRef, Report, SmallVec, SourceSpan, Spanned, Type,
    Value, ValueRef,
    conversion::{
        ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConvertedOperands,
        TypeConverter,
    },
    dialects::builtin::attributes::Overflow,
    patterns::{Pattern, PatternInfo},
};

use crate::{
    WasmTarget,
    carried::{self, const_int, mask, sign_extend, zero},
    convert::calls::{pattern_info, registered},
    types::{carrier, carrier_width, width},
};

/// Adds the patterns of this module.
pub fn populate(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
    _target: &WasmTarget,
) -> Result<(), Report> {
    let context = patterns.context();
    let rules: [(Rule, &'static str); 23] = [
        (Rule::Add, "wasm-arith-add"),
        (Rule::Sub, "wasm-arith-sub"),
        (Rule::Mul, "wasm-arith-mul"),
        (Rule::Band, "wasm-arith-band"),
        (Rule::Bor, "wasm-arith-bor"),
        (Rule::Bxor, "wasm-arith-bxor"),
        (Rule::Bnot, "wasm-arith-bnot"),
        (Rule::Neg, "wasm-arith-neg"),
        (Rule::Udiv, "wasm-arith-udiv"),
        (Rule::Urem, "wasm-arith-urem"),
        (Rule::Sdiv, "wasm-arith-sdiv"),
        (Rule::Srem, "wasm-arith-srem"),
        (Rule::Shl, "wasm-arith-shl"),
        (Rule::Shr, "wasm-arith-shr"),
        (Rule::Ashr, "wasm-arith-ashr"),
        (Rule::Cmpi, "wasm-arith-cmpi"),
        (Rule::Trunc, "wasm-arith-trunc"),
        (Rule::Zext, "wasm-arith-zext"),
        (Rule::Sext, "wasm-arith-sext"),
        (Rule::SextOrTrunc, "wasm-arith-sext-or-trunc"),
        (Rule::ZextOrTrunc, "wasm-arith-zext-or-trunc"),
        (Rule::Select, "wasm-cf-select"),
        (Rule::Poison, "wasm-ub-poison"),
    ];
    for (rule, name) in rules {
        let info = match rule {
            Rule::Add => info::<arith::Add>(&context, name, rule),
            Rule::Sub => info::<arith::Sub>(&context, name, rule),
            Rule::Mul => info::<arith::Mul>(&context, name, rule),
            Rule::Band => info::<arith::Band>(&context, name, rule),
            Rule::Bor => info::<arith::Bor>(&context, name, rule),
            Rule::Bxor => info::<arith::Bxor>(&context, name, rule),
            Rule::Bnot => info::<arith::Bnot>(&context, name, rule),
            Rule::Neg => info::<arith::Neg>(&context, name, rule),
            Rule::Udiv => info::<arith::Udiv>(&context, name, rule),
            Rule::Urem => info::<arith::Urem>(&context, name, rule),
            Rule::Sdiv => info::<arith::Sdiv>(&context, name, rule),
            Rule::Srem => info::<arith::Srem>(&context, name, rule),
            Rule::Shl => info::<arith::Shl>(&context, name, rule),
            Rule::Shr => info::<arith::Shr>(&context, name, rule),
            Rule::Ashr => info::<arith::Ashr>(&context, name, rule),
            Rule::Cmpi => info::<arith::Cmpi>(&context, name, rule),
            Rule::Trunc => info::<arith::Trunc>(&context, name, rule),
            Rule::Zext => info::<arith::Zext>(&context, name, rule),
            Rule::Sext => info::<arith::Sext>(&context, name, rule),
            Rule::SextOrTrunc => info::<arith::SextOrTrunc>(&context, name, rule),
            Rule::ZextOrTrunc => info::<arith::ZextOrTrunc>(&context, name, rule),
            Rule::Select => info::<cf::Select>(&context, name, rule),
            Rule::Poison => info::<ub::Poison>(&context, name, rule),
        };
        patterns.push(IntPattern {
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

/// What each pattern of this module lowers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Rule {
    /// `arith.add`, wrapping or unchecked: `wasm.add`, then the mask. Wrapping refines unchecked.
    Add,
    /// `arith.sub`: `wasm.sub`, then the mask.
    Sub,
    /// `arith.mul`: `wasm.mul`, then the mask. The low `N` bits of the product do not depend on
    /// the bits above `N` of the operands, which the invariant keeps clear anyway.
    Mul,
    /// `arith.band`: `wasm.and`; both operands have clear high bits, and so has the result.
    Band,
    /// `arith.bor`: `wasm.or`; no bit above `N` is set in either operand.
    Bor,
    /// `arith.bxor`: `wasm.xor`; no bit above `N` is set in either operand.
    Bxor,
    /// `arith.bnot`: `wasm.xor` with `2^N - 1`, which flips the low `N` bits only.
    Bnot,
    /// `arith.neg`: `wasm.sub` from zero, then the mask.
    Neg,
    /// `arith.udiv`: `wasm.div_u`; the quotient is not above the dividend. A zero divisor traps in
    /// both.
    Udiv,
    /// `arith.urem`: `wasm.rem_u`; the remainder is below the divisor.
    Urem,
    /// `arith.sdiv`: both operands sign-extended from `N`, `wasm.div_s`, then the mask. See the
    /// module documentation for the minimum value divided by -1.
    Sdiv,
    /// `arith.srem`: both operands sign-extended from `N`, `wasm.rem_s`, then the mask.
    Srem,
    /// `arith.shl`: `wasm.shl`, then the mask. An amount of `N` or more is undefined in `arith`.
    Shl,
    /// `arith.shr`: `wasm.shr_u`. The value has clear high bits, so a logical right shift by any
    /// amount keeps them clear, and the result for an amount below `N` is `arith`'s.
    Shr,
    /// `arith.ashr`: the value sign-extended from `N`, `wasm.shr_s`, then the mask. An amount
    /// below `N` copies the sign bit of bit `N - 1`, which the extension put in the high bits.
    Ashr,
    /// `arith.cmpi`: equality and the unsigned orders compare the carriers directly (both are
    /// zero-extended); the signed orders sign-extend both operands first.
    Cmpi,
    /// `arith.trunc`: `wasm.wrap` if the carrier narrows, then the mask.
    Trunc,
    /// `arith.zext`: `wasm.extend_u` if the carrier widens; the high bits are already clear.
    Zext,
    /// `arith.sext`: sign extension from `N` in the source carrier, `wasm.extend_s` if the carrier
    /// widens, then the mask to the result's width.
    Sext,
    /// `arith.sext_or_trunc` with `isize` 32 bits wide.
    SextOrTrunc,
    /// `arith.zext_or_trunc` with `isize` 32 bits wide.
    ZextOrTrunc,
    /// `cf.select`: `wasm.select`; the carried `i1` condition is 0 or 1.
    Select,
    /// `ub.poison`: a zero of the carrier. Any value refines poison.
    Poison,
}

impl Rule {
    /// The names of the operations a pattern of this rule may create.
    fn generated(self, context: &alloc::rc::Rc<runic_ir::Context>) -> Vec<runic_ir::OperationName> {
        macro_rules! names {
            ($($op:ty),*) => { alloc::vec![$(registered::<$op>(context)),*] };
        }
        match self {
            Rule::Add => names!(wasm::Add, wasm::And, wasm::Const),
            Rule::Sub => names!(wasm::Sub, wasm::And, wasm::Const),
            Rule::Mul => names!(wasm::Mul, wasm::And, wasm::Const),
            Rule::Band => names!(wasm::And),
            Rule::Bor => names!(wasm::Or),
            Rule::Bxor => names!(wasm::Xor),
            Rule::Bnot => names!(wasm::Xor, wasm::Const),
            Rule::Neg => names!(wasm::Sub, wasm::And, wasm::Const),
            Rule::Udiv => names!(wasm::DivU),
            Rule::Urem => names!(wasm::RemU),
            Rule::Sdiv => names!(
                wasm::DivS,
                wasm::And,
                wasm::Const,
                wasm::Shl,
                wasm::ShrS,
                wasm::Extend8S,
                wasm::Extend16S,
                wasm::Extend32S
            ),
            Rule::Srem => names!(
                wasm::RemS,
                wasm::And,
                wasm::Const,
                wasm::Shl,
                wasm::ShrS,
                wasm::Extend8S,
                wasm::Extend16S,
                wasm::Extend32S
            ),
            Rule::Shl => names!(wasm::Shl, wasm::And, wasm::Const),
            Rule::Shr => names!(wasm::ShrU),
            Rule::Ashr => names!(
                wasm::ShrS,
                wasm::And,
                wasm::Const,
                wasm::Shl,
                wasm::Extend8S,
                wasm::Extend16S,
                wasm::Extend32S
            ),
            Rule::Cmpi => names!(
                wasm::Eq,
                wasm::Ne,
                wasm::LtU,
                wasm::LeU,
                wasm::GtU,
                wasm::GeU,
                wasm::LtS,
                wasm::LeS,
                wasm::GtS,
                wasm::GeS,
                wasm::Const,
                wasm::Shl,
                wasm::ShrS,
                wasm::Extend8S,
                wasm::Extend16S,
                wasm::Extend32S
            ),
            Rule::Trunc | Rule::Zext | Rule::Sext | Rule::SextOrTrunc | Rule::ZextOrTrunc => {
                names!(
                    wasm::Wrap,
                    wasm::ExtendS,
                    wasm::ExtendU,
                    wasm::And,
                    wasm::Const,
                    wasm::Shl,
                    wasm::ShrS,
                    wasm::Extend8S,
                    wasm::Extend16S,
                    wasm::Extend32S
                )
            }
            Rule::Select => names!(wasm::Select),
            Rule::Poison => names!(wasm::Const, wasm::ConstF, wasm::RefNull),
        }
    }
}

/// The pattern of one [Rule].
struct IntPattern {
    info: PatternInfo,
    converter: TypeConverter,
    rule: Rule,
}

impl Pattern for IntPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

/// What a pattern reads of its operation before it rewrites it.
struct Source {
    /// The type of the first operand.
    operand_ty: Type,
    /// The type of the result.
    result_ty: Type,
    /// The predicate, for `arith.cmpi`.
    predicate: Option<arith::CmpPredicate>,
    /// Whether the operation is one this module lowers: not a `checked` operation.
    supported: bool,
    span: SourceSpan,
}

impl Source {
    /// Reads `op`, which `rule` is rooted at.
    fn of(rule: Rule, op: OperationRef) -> Self {
        let op_ref = op.borrow();
        let span = op_ref.span();
        let operand_ty = op_ref
            .operands()
            .iter()
            .next()
            .map(|operand| operand.borrow().as_value_ref().borrow().ty())
            .unwrap_or_else(|| op_ref.results()[0].borrow().ty());
        let result_ty = op_ref.results()[0].borrow().ty();
        let overflow = match rule {
            Rule::Add => op_ref.downcast_ref::<arith::Add>().map(|op| *op.get_overflow()),
            Rule::Sub => op_ref.downcast_ref::<arith::Sub>().map(|op| *op.get_overflow()),
            Rule::Mul => op_ref.downcast_ref::<arith::Mul>().map(|op| *op.get_overflow()),
            _ => None,
        };
        let predicate = op_ref.downcast_ref::<arith::Cmpi>().map(|op| *op.get_predicate());
        Self {
            operand_ty,
            result_ty,
            predicate,
            supported: !matches!(overflow, Some(Overflow::Checked(_))),
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

/// `2^n - 1`, as the bits of a constant: the low `n` bits set.
fn ones(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

impl IntPattern {
    /// The value an operation of `self.rule` on `values` becomes, or `None` where the pattern does
    /// not apply.
    fn lower(
        &self,
        rw: &mut ConversionPatternRewriter,
        source: &Source,
        values: &[ValueRef],
    ) -> Result<Option<ValueRef>, Report> {
        let span = source.span;
        let rule = self.rule;
        if rule == Rule::Poison {
            let Some(carrier) = carrier(rw.context(), source.result_ty) else {
                return Ok(None);
            };
            return zero(rw, carrier, span).map(Some);
        }
        if rule == Rule::Select {
            let [cond, a, b] = values else {
                return Ok(None);
            };
            let select = rw.create_op::<wasm::Select, _>(span, (*cond, *a, *b))?;
            return Ok(Some(carried::result(&select)));
        }
        let Some(n) = width(source.operand_ty) else {
            return Ok(None);
        };
        let Some(&first) = values.first() else {
            return Ok(None);
        };
        let wide = values.get(1).copied();
        let value = match (rule, wide) {
            (Rule::Add, Some(b)) => {
                let sum = binary::<wasm::Add>(rw, first, b, span)?;
                mask(rw, sum, n, span)?
            }
            (Rule::Sub, Some(b)) => {
                let difference = binary::<wasm::Sub>(rw, first, b, span)?;
                mask(rw, difference, n, span)?
            }
            (Rule::Mul, Some(b)) => {
                let product = binary::<wasm::Mul>(rw, first, b, span)?;
                mask(rw, product, n, span)?
            }
            (Rule::Band, Some(b)) => binary::<wasm::And>(rw, first, b, span)?,
            (Rule::Bor, Some(b)) => binary::<wasm::Or>(rw, first, b, span)?,
            (Rule::Bxor, Some(b)) => binary::<wasm::Xor>(rw, first, b, span)?,
            (Rule::Udiv, Some(b)) => binary::<wasm::DivU>(rw, first, b, span)?,
            (Rule::Urem, Some(b)) => binary::<wasm::RemU>(rw, first, b, span)?,
            (Rule::Sdiv, Some(b)) => {
                let a = sign_extend(rw, first, n, span)?;
                let b = sign_extend(rw, b, n, span)?;
                let quotient = binary::<wasm::DivS>(rw, a, b, span)?;
                mask(rw, quotient, n, span)?
            }
            (Rule::Srem, Some(b)) => {
                let a = sign_extend(rw, first, n, span)?;
                let b = sign_extend(rw, b, n, span)?;
                let remainder = binary::<wasm::RemS>(rw, a, b, span)?;
                mask(rw, remainder, n, span)?
            }
            (Rule::Shl, Some(b)) => {
                let shifted = binary::<wasm::Shl>(rw, first, b, span)?;
                mask(rw, shifted, n, span)?
            }
            (Rule::Shr, Some(b)) => binary::<wasm::ShrU>(rw, first, b, span)?,
            (Rule::Ashr, Some(b)) => {
                let a = sign_extend(rw, first, n, span)?;
                let shifted = binary::<wasm::ShrS>(rw, a, b, span)?;
                mask(rw, shifted, n, span)?
            }
            (Rule::Bnot, None) => {
                let carrier = first.borrow().ty();
                let all = const_int(rw, ones(n), carrier, span)?;
                binary::<wasm::Xor>(rw, first, all, span)?
            }
            (Rule::Neg, None) => {
                let carrier = first.borrow().ty();
                let nothing = const_int(rw, 0, carrier, span)?;
                let negated = binary::<wasm::Sub>(rw, nothing, first, span)?;
                mask(rw, negated, n, span)?
            }
            (Rule::Cmpi, Some(b)) => {
                let Some(predicate) = source.predicate else {
                    return Ok(None);
                };
                compare(rw, predicate, n, first, b, span)?
            }
            (
                Rule::Trunc | Rule::Zext | Rule::Sext | Rule::SextOrTrunc | Rule::ZextOrTrunc,
                None,
            ) => {
                let Some(m) = width(source.result_ty) else {
                    return Ok(None);
                };
                let Some(target) = carrier(rw.context(), source.result_ty) else {
                    return Ok(None);
                };
                let signed = matches!(rule, Rule::Sext | Rule::SextOrTrunc);
                resize(rw, first, n, m, target, signed, span)?
            }
            _ => return Ok(None),
        };
        Ok(Some(value))
    }
}

/// The `i32` 0 or 1 that `arith.cmpi` with `predicate` of `a` and `b`, of width `n`, gives.
fn compare(
    rw: &mut ConversionPatternRewriter,
    predicate: arith::CmpPredicate,
    n: u32,
    a: ValueRef,
    b: ValueRef,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    use arith::CmpPredicate as P;
    let (a, b) = if matches!(predicate, P::Slt | P::Sle | P::Sgt | P::Sge) {
        (sign_extend(rw, a, n, span)?, sign_extend(rw, b, n, span)?)
    } else {
        (a, b)
    };
    match predicate {
        P::Eq => binary::<wasm::Eq>(rw, a, b, span),
        P::Ne => binary::<wasm::Ne>(rw, a, b, span),
        P::Ult => binary::<wasm::LtU>(rw, a, b, span),
        P::Ule => binary::<wasm::LeU>(rw, a, b, span),
        P::Ugt => binary::<wasm::GtU>(rw, a, b, span),
        P::Uge => binary::<wasm::GeU>(rw, a, b, span),
        P::Slt => binary::<wasm::LtS>(rw, a, b, span),
        P::Sle => binary::<wasm::LeS>(rw, a, b, span),
        P::Sgt => binary::<wasm::GtS>(rw, a, b, span),
        P::Sge => binary::<wasm::GeS>(rw, a, b, span),
    }
}

/// `value`, a carried integer of width `from`, as a carried integer of width `to` in the carrier
/// `target`: sign-extended or zero-extended when it widens (`signed`), truncated when it narrows,
/// itself when the widths agree.
fn resize(
    rw: &mut ConversionPatternRewriter,
    value: ValueRef,
    from: u32,
    to: u32,
    target: Type,
    signed: bool,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    let source_carrier = value.borrow().ty();
    let widens_carrier =
        carrier_width(source_carrier) == Some(32) && carrier_width(target) == Some(64);
    let narrows_carrier =
        carrier_width(source_carrier) == Some(64) && carrier_width(target) == Some(32);
    if to == from {
        return Ok(value);
    }
    if to < from {
        let value = if narrows_carrier {
            carried::result(&rw.create_op::<wasm::Wrap, _>(span, (value, target))?)
        } else {
            value
        };
        return mask(rw, value, to, span);
    }
    if signed {
        let value = sign_extend(rw, value, from, span)?;
        let value = if widens_carrier {
            carried::result(&rw.create_op::<wasm::ExtendS, _>(span, (value, target))?)
        } else {
            value
        };
        mask(rw, value, to, span)
    } else if widens_carrier {
        Ok(carried::result(&rw.create_op::<wasm::ExtendU, _>(span, (value, target))?))
    } else {
        Ok(value)
    }
}

impl ConversionPattern for IntPattern {
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
        let Some(value) = self.lower(rewriter, &source, &values)? else {
            return Ok(false);
        };
        rewriter.replace_op(op, &[value])?;
        Ok(true)
    }
}
