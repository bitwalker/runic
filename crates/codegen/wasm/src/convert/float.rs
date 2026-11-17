//! The patterns of the float operations of `arith`: arithmetic, comparisons and conversions.
//!
//! A float is its own carrier, so the arithmetic is one Wasm instruction each: `arith.addf`,
//! `subf`, `mulf`, `divf` and `negf` become `wasm.add`, `sub`, `mul`, `div` and `neg`. Both sides
//! round to nearest, ties to even, and leave the sign and payload of a NaN result unspecified;
//! `wasm.neg` flips the sign bit only, as `arith.negf` does.
//!
//! **Comparisons.** `arith.cmpf` gives an `i1`, carried as the `i32` 0 or 1 that a Wasm comparison
//! gives. Wasm has the ordered `eq`, `lt`, `gt`, `le`, `ge` (false when an operand is a NaN) and
//! the unordered `ne` (true when one is). Each of the sixteen predicates is one of these, the
//! `eqz` of one (the unordered predicates are the negations of the ordered ones), or a
//! combination of two (`one`, `ord`, `uno`).
//!
//! **Conversions to an integer.** `arith.fptosi` and `arith.fptoui` truncate toward zero, and
//! their result is poison when the operand is a NaN or the truncated value does not fit the
//! result. They become the *saturating* `wasm.trunc_sat_s` and `wasm.trunc_sat_u` to the
//! result's carrier, on purpose: the trapping `wasm.trunc_s` and `wasm.trunc_u` would introduce a
//! trap where `arith` has none, while any value refines poison, and the saturating instructions
//! never trap. The carrier's result is then masked to `N` bits, which is the value itself when it
//! fits and keeps carried form when it does not.
//!
//! **Conversions from an integer.** `arith.sitofp` sign-extends its operand from `N` bits to the
//! carrier and uses `wasm.convert_s`; `arith.uitofp` uses `wasm.convert_u` on the carrier, which
//! is zero-extended already. Both round to nearest, ties to even, as `arith` does.
//!
//! `arith.fpext`, `arith.fptrunc` and `arith.bitcast` are `wasm.promote`, `wasm.demote` and
//! `wasm.reinterpret`.

use alloc::vec::Vec;

use runic_dialect_arith::{self as arith, CmpFPredicate};
use runic_dialect_wasm as wasm;
use runic_ir::{
    BuildableOp, Op, OpRegistration, OperationRef, Report, SmallVec, SourceSpan, Spanned, Type,
    Value, ValueRef,
    conversion::{
        ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConvertedOperands,
        TypeConverter,
    },
    patterns::{Pattern, PatternInfo},
};

use crate::{
    carried::{self, const_int, mask, sign_extend},
    convert::calls::{pattern_info, registered},
    types::{carrier, width},
};

/// Adds the patterns of this module.
pub fn populate(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
) -> Result<(), Report> {
    let context = patterns.context();
    let rules: [(Rule, &'static str); 13] = [
        (Rule::Addf, "wasm-arith-addf"),
        (Rule::Subf, "wasm-arith-subf"),
        (Rule::Mulf, "wasm-arith-mulf"),
        (Rule::Divf, "wasm-arith-divf"),
        (Rule::Negf, "wasm-arith-negf"),
        (Rule::Cmpf, "wasm-arith-cmpf"),
        (Rule::Fptosi, "wasm-arith-fptosi"),
        (Rule::Fptoui, "wasm-arith-fptoui"),
        (Rule::Sitofp, "wasm-arith-sitofp"),
        (Rule::Uitofp, "wasm-arith-uitofp"),
        (Rule::Fpext, "wasm-arith-fpext"),
        (Rule::Fptrunc, "wasm-arith-fptrunc"),
        (Rule::Bitcast, "wasm-arith-bitcast"),
    ];
    for (rule, name) in rules {
        let info = match rule {
            Rule::Addf => info::<arith::Addf>(&context, name, rule),
            Rule::Subf => info::<arith::Subf>(&context, name, rule),
            Rule::Mulf => info::<arith::Mulf>(&context, name, rule),
            Rule::Divf => info::<arith::Divf>(&context, name, rule),
            Rule::Negf => info::<arith::Negf>(&context, name, rule),
            Rule::Cmpf => info::<arith::Cmpf>(&context, name, rule),
            Rule::Fptosi => info::<arith::Fptosi>(&context, name, rule),
            Rule::Fptoui => info::<arith::Fptoui>(&context, name, rule),
            Rule::Sitofp => info::<arith::Sitofp>(&context, name, rule),
            Rule::Uitofp => info::<arith::Uitofp>(&context, name, rule),
            Rule::Fpext => info::<arith::Fpext>(&context, name, rule),
            Rule::Fptrunc => info::<arith::Fptrunc>(&context, name, rule),
            Rule::Bitcast => info::<arith::Bitcast>(&context, name, rule),
        };
        patterns.push(FloatPattern {
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
    /// `arith.addf`: `wasm.add`.
    Addf,
    /// `arith.subf`: `wasm.sub`.
    Subf,
    /// `arith.mulf`: `wasm.mul`.
    Mulf,
    /// `arith.divf`: `wasm.div`.
    Divf,
    /// `arith.negf`: `wasm.neg`, which flips the sign bit and keeps every other bit.
    Negf,
    /// `arith.cmpf`: see [compare].
    Cmpf,
    /// `arith.fptosi`: `wasm.trunc_sat_s` to the carrier, then the mask. Saturating on purpose:
    /// `arith` makes an unrepresentable result poison, which any value refines, and the
    /// saturating instruction never traps, where `wasm.trunc_s` would introduce a trap.
    Fptosi,
    /// `arith.fptoui`: `wasm.trunc_sat_u` to the carrier, then the mask; saturating for the reason
    /// [Rule::Fptosi] gives.
    Fptoui,
    /// `arith.sitofp`: the operand sign-extended from `N` bits, then `wasm.convert_s`.
    Sitofp,
    /// `arith.uitofp`: `wasm.convert_u` of the zero-extended carrier.
    Uitofp,
    /// `arith.fpext`: `wasm.promote`.
    Fpext,
    /// `arith.fptrunc`: `wasm.demote`.
    Fptrunc,
    /// `arith.bitcast`: `wasm.reinterpret`; `i32` and `f32`, `i64` and `f64` are their own
    /// carriers.
    Bitcast,
}

impl Rule {
    /// The names of the operations a pattern of this rule may create.
    fn generated(self, context: &alloc::rc::Rc<runic_ir::Context>) -> Vec<runic_ir::OperationName> {
        macro_rules! names {
            ($($op:ty),*) => { alloc::vec![$(registered::<$op>(context)),*] };
        }
        match self {
            Rule::Addf => names!(wasm::Add),
            Rule::Subf => names!(wasm::Sub),
            Rule::Mulf => names!(wasm::Mul),
            Rule::Divf => names!(wasm::Div),
            Rule::Negf => names!(wasm::Neg),
            Rule::Cmpf => names!(
                wasm::Eq,
                wasm::Ne,
                wasm::Lt,
                wasm::Gt,
                wasm::Le,
                wasm::Ge,
                wasm::Eqz,
                wasm::And,
                wasm::Or,
                wasm::Const
            ),
            Rule::Fptosi => names!(wasm::TruncSatS, wasm::And, wasm::Const),
            Rule::Fptoui => names!(wasm::TruncSatU, wasm::And, wasm::Const),
            Rule::Sitofp => names!(
                wasm::ConvertS,
                wasm::Const,
                wasm::Shl,
                wasm::ShrS,
                wasm::Extend8S,
                wasm::Extend16S,
                wasm::Extend32S
            ),
            Rule::Uitofp => names!(wasm::ConvertU),
            Rule::Fpext => names!(wasm::Promote),
            Rule::Fptrunc => names!(wasm::Demote),
            Rule::Bitcast => names!(wasm::Reinterpret),
        }
    }
}

/// The pattern of one [Rule].
struct FloatPattern {
    info: PatternInfo,
    converter: TypeConverter,
    rule: Rule,
}

impl Pattern for FloatPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
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

/// The `i32` 0 or 1 that `arith.cmpf` with `predicate` gives on the floats `a` and `b`.
///
/// * `oeq`, `ogt`, `oge`, `olt`, `ole`: `wasm.eq`, `gt`, `ge`, `lt`, `le`, which are false when
///   an operand is a NaN.
/// * `une`: `wasm.ne`, which is true when an operand is a NaN.
/// * `one`: `(a < b) or (a > b)`; both are false on a NaN, and one is true exactly when the
///   ordered operands differ.
/// * `ord`: `(a == a) and (b == b)`; a value equals itself unless it is a NaN.
/// * `uno`: `(a != a) or (b != b)`.
/// * `ueq`: the `eqz` of the lowering of `one`.
/// * `ugt`, `uge`, `ult`, `ule`: the `eqz` of `le`, `lt`, `ge`, `gt`: "unordered or greater"
///   is "not (ordered and less or equal)", and so on.
/// * `false`, `true`: `wasm.const 0` and `wasm.const 1`.
fn compare(
    rw: &mut ConversionPatternRewriter,
    predicate: CmpFPredicate,
    a: ValueRef,
    b: ValueRef,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    use CmpFPredicate as P;
    let i32 = rw.context().i32();
    let one = |rw: &mut ConversionPatternRewriter| -> Result<ValueRef, Report> {
        let less = binary::<wasm::Lt>(rw, a, b, span)?;
        let greater = binary::<wasm::Gt>(rw, a, b, span)?;
        binary::<wasm::Or>(rw, less, greater, span)
    };
    match predicate {
        P::False => const_int(rw, 0, i32, span),
        P::True => const_int(rw, 1, i32, span),
        P::Oeq => binary::<wasm::Eq>(rw, a, b, span),
        P::Ogt => binary::<wasm::Gt>(rw, a, b, span),
        P::Oge => binary::<wasm::Ge>(rw, a, b, span),
        P::Olt => binary::<wasm::Lt>(rw, a, b, span),
        P::Ole => binary::<wasm::Le>(rw, a, b, span),
        P::Une => binary::<wasm::Ne>(rw, a, b, span),
        P::One => one(rw),
        P::Ord => {
            let left = binary::<wasm::Eq>(rw, a, a, span)?;
            let right = binary::<wasm::Eq>(rw, b, b, span)?;
            binary::<wasm::And>(rw, left, right, span)
        }
        P::Uno => {
            let left = binary::<wasm::Ne>(rw, a, a, span)?;
            let right = binary::<wasm::Ne>(rw, b, b, span)?;
            binary::<wasm::Or>(rw, left, right, span)
        }
        P::Ueq => {
            let ordered = one(rw)?;
            unary::<wasm::Eqz>(rw, ordered, span)
        }
        P::Ugt => {
            let ordered = binary::<wasm::Le>(rw, a, b, span)?;
            unary::<wasm::Eqz>(rw, ordered, span)
        }
        P::Uge => {
            let ordered = binary::<wasm::Lt>(rw, a, b, span)?;
            unary::<wasm::Eqz>(rw, ordered, span)
        }
        P::Ult => {
            let ordered = binary::<wasm::Ge>(rw, a, b, span)?;
            unary::<wasm::Eqz>(rw, ordered, span)
        }
        P::Ule => {
            let ordered = binary::<wasm::Gt>(rw, a, b, span)?;
            unary::<wasm::Eqz>(rw, ordered, span)
        }
    }
}

/// What a pattern reads of its operation before it rewrites it.
struct Source {
    /// The type of the first operand.
    operand_ty: Type,
    /// The type of the result.
    result_ty: Type,
    /// The predicate, for `arith.cmpf`.
    predicate: Option<CmpFPredicate>,
    span: SourceSpan,
}

impl Source {
    /// Reads `op`.
    fn of(op: OperationRef) -> Self {
        let op_ref = op.borrow();
        Self {
            operand_ty: op_ref.operands()[0].borrow().as_value_ref().borrow().ty(),
            result_ty: op_ref.results()[0].borrow().ty(),
            predicate: op_ref.downcast_ref::<arith::Cmpf>().map(|op| *op.get_predicate()),
            span: op_ref.span(),
        }
    }
}

impl FloatPattern {
    /// The value an operation of `self.rule` on `values` becomes, or `None` where the pattern does
    /// not apply.
    fn lower(
        &self,
        rw: &mut ConversionPatternRewriter,
        source: &Source,
        values: &[ValueRef],
    ) -> Result<Option<ValueRef>, Report> {
        let span = source.span;
        let Some(&a) = values.first() else {
            return Ok(None);
        };
        let b = values.get(1).copied();
        let value = match (self.rule, b) {
            (Rule::Addf, Some(b)) => binary::<wasm::Add>(rw, a, b, span)?,
            (Rule::Subf, Some(b)) => binary::<wasm::Sub>(rw, a, b, span)?,
            (Rule::Mulf, Some(b)) => binary::<wasm::Mul>(rw, a, b, span)?,
            (Rule::Divf, Some(b)) => binary::<wasm::Div>(rw, a, b, span)?,
            (Rule::Negf, None) => unary::<wasm::Neg>(rw, a, span)?,
            (Rule::Cmpf, Some(b)) => {
                let Some(predicate) = source.predicate else {
                    return Ok(None);
                };
                compare(rw, predicate, a, b, span)?
            }
            (Rule::Fptosi | Rule::Fptoui, None) => {
                let Some(n) = width(source.result_ty) else {
                    return Ok(None);
                };
                let Some(target) = carrier(rw.context(), source.result_ty) else {
                    return Ok(None);
                };
                let truncated = if self.rule == Rule::Fptosi {
                    convert::<wasm::TruncSatS>(rw, a, target, span)?
                } else {
                    convert::<wasm::TruncSatU>(rw, a, target, span)?
                };
                mask(rw, truncated, n, span)?
            }
            (Rule::Sitofp, None) => {
                let Some(n) = width(source.operand_ty) else {
                    return Ok(None);
                };
                let extended = sign_extend(rw, a, n, span)?;
                convert::<wasm::ConvertS>(rw, extended, source.result_ty, span)?
            }
            (Rule::Uitofp, None) => convert::<wasm::ConvertU>(rw, a, source.result_ty, span)?,
            (Rule::Fpext, None) => convert::<wasm::Promote>(rw, a, source.result_ty, span)?,
            (Rule::Fptrunc, None) => convert::<wasm::Demote>(rw, a, source.result_ty, span)?,
            (Rule::Bitcast, None) => {
                let Some(target) = carrier(rw.context(), source.result_ty) else {
                    return Ok(None);
                };
                convert::<wasm::Reinterpret>(rw, a, target, span)?
            }
            _ => return Ok(None),
        };
        Ok(Some(value))
    }
}

impl ConversionPattern for FloatPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let source = Source::of(op);
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
