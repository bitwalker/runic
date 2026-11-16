//! Conversions between floating-point and integer types, between float widths, and `bitcast`.

use alloc::{format, string::String};

use runic_ir::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::TypeAttr,
    effects::MemoryEffectOpInterface,
    traits::*,
    *,
};

use crate::ArithDialect;

/// What a conversion operation requires of its operand type and its result type, beyond what the
/// type constraints of its operand and result already require.
#[derive(Copy, Clone)]
enum ConversionRule {
    /// Any float to any integer, or any integer to any float: the constraints say everything.
    Any,
    /// `f32` to `f64`.
    Extend,
    /// `f64` to `f32`.
    Truncate,
    /// `i32` and `f32` in either direction, or `i64` and `f64` in either direction.
    Bitcast,
}

impl ConversionRule {
    /// What is wrong with a conversion from `from` to `to` under this rule, or `None` if it is
    /// valid.
    fn check(self, context: &Context, from: Type, to: Type) -> Option<String> {
        let (f32, f64) = (context.f32(), context.f64());
        match self {
            Self::Any => None,
            Self::Extend => (!(from == f32 && to == f64))
                .then(|| format!("it extends `f32` to `f64`, and `{from}` to `{to}` is not that")),
            Self::Truncate => (!(from == f64 && to == f32)).then(|| {
                format!("it truncates `f64` to `f32`, and `{from}` to `{to}` is not that")
            }),
            Self::Bitcast => {
                let (i32, i64) = (context.i32(), context.i64());
                let allowed =
                    [(i32, f32), (f32, i32), (i64, f64), (f64, i64)].contains(&(from, to));
                (!allowed).then(|| {
                    format!(
                        "`{from}` to `{to}` is not a bitcast the dialect has; the allowed ones \
                         are `i32` to `f32`, `f32` to `i32`, `i64` to `f64` and `f64` to `i64`"
                    )
                })
            }
        }
    }
}

/// Checks the operand and result types of the conversion `op` against `rule`, and its result type
/// against its `ty` attribute.
fn verify_conversion(op: &Operation, rule: ConversionRule, ty: Type) -> Result<(), Report> {
    let from = op.operands()[0].borrow().as_value_ref().borrow().ty();
    let to = op.results()[0].borrow().ty();
    let problem = if to != ty {
        Some(format!("its result type `{to}` is not its `ty` attribute `{ty}`"))
    } else {
        rule.check(&op.context_rc(), from, to)
    };
    let Some(problem) = problem else {
        return Ok(());
    };
    Err(runic_ir::diagnostics::report!(
        message: "invalid operation",
        labels: [
            primary(op.span(), format!("invalid '{}' from `{from}` to `{to}`: {problem}", op.name())),
        ],
    ))
}

/// Implements [InferTypeOpInterface] for a conversion whose result has the type `ty`, and checks
/// the conversion against its [ConversionRule].
macro_rules! conversion_type_rules {
    ($Op:ty, $rule:expr) => {
        impl InferTypeOpInterface for $Op {
            fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
                let ty = *self.get_ty();
                self.result_mut().set_type(ty);
                Ok(())
            }
        }

        impl Verify<dyn InferTypeOpInterface> for $Op {
            fn verify(&self, _context: &Context) -> Result<(), Report> {
                verify_conversion(self.as_operation(), $rule, *self.get_ty())
            }
        }
    };
}

/// Implements [Foldable] for the conversion `$Op` by `fold_conversion` with `$conversion`.
macro_rules! fold_conversion {
    ($Op:ty, $rule:expr, $conversion:expr) => {
        impl Foldable for $Op {
            fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
                let operands = crate::fold::operand_constants(self.as_operation());
                self.fold_with(&operands, results)
            }

            fn fold_with(
                &self,
                operands: &[Option<AttributeRef>],
                results: &mut SmallVec<[OpFoldResult; 1]>,
            ) -> FoldResult {
                crate::fold_float::fold_conversion(
                    self.as_operation(),
                    operands,
                    results,
                    *self.get_ty(),
                    |context, from, to| $rule.check(context, from, to).is_none(),
                    $conversion,
                )
            }
        }
    };
}

use crate::fold_float::Conversion;

/// Conversion of a float to the signed integer type `ty`: the value is truncated toward zero.
///
/// The result is **poison** if the operand is a NaN or the truncated value is not representable in
/// `ty` as a signed integer (an infinity is not). This is LLVM's and MLIR's rule. A target whose
/// conversion traps or saturates (Wasm has both) refines it. The operation is free of effects and
/// never traps. It folds on a constant operand that is not a NaN and whose truncated value is
/// representable; an operation whose result would be poison is not folded. The result of a fold of
/// an `isize` result is the same on a 32-bit and a 64-bit target, or there is none.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Fptosi {
    #[operand]
    operand: AnyFloat,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyInteger,
}

conversion_type_rules!(Fptosi, ConversionRule::Any);
fold_conversion!(Fptosi, ConversionRule::Any, Conversion::FloatToInt { signed: true });

/// Conversion of a float to the unsigned integer type `ty`: the value is truncated toward zero.
///
/// The result is **poison** if the operand is a NaN or the truncated value is not representable in
/// `ty` as an unsigned integer. A negative value that truncates to zero (`-0.5`) gives `0`; any
/// other negative value is poison. This is LLVM's and MLIR's rule. A target whose conversion traps
/// or saturates (Wasm has both) refines it. The operation is free of effects and never traps. It
/// folds on a constant operand that is not a NaN and whose truncated value is representable.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Fptoui {
    #[operand]
    operand: AnyFloat,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyInteger,
}

conversion_type_rules!(Fptoui, ConversionRule::Any);
fold_conversion!(Fptoui, ConversionRule::Any, Conversion::FloatToInt { signed: false });

/// Conversion of an integer, read as signed, to the float type `ty`, rounding to nearest, ties to
/// even. A value too large for `ty` (only `i128` to `f32` can be) becomes an infinity.
///
/// The operation is free of effects and never traps. It folds on a constant operand; the result is
/// never a NaN. For an `isize` operand it folds only when the 64-bit and the 32-bit target give the
/// same float.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Sitofp {
    #[operand]
    operand: AnyInteger,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyFloat,
}

conversion_type_rules!(Sitofp, ConversionRule::Any);
fold_conversion!(Sitofp, ConversionRule::Any, Conversion::IntToFloat { signed: true });

/// Conversion of an integer, read as unsigned, to the float type `ty`, rounding to nearest, ties to
/// even. A value too large for `ty` (only `u128` to `f32` can be) becomes an infinity.
///
/// The operation is free of effects and never traps. It folds on a constant operand; the result is
/// never a NaN. For an `isize` operand it folds only when the 64-bit and the 32-bit target give the
/// same float.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Uitofp {
    #[operand]
    operand: AnyInteger,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyFloat,
}

conversion_type_rules!(Uitofp, ConversionRule::Any);
fold_conversion!(Uitofp, ConversionRule::Any, Conversion::IntToFloat { signed: false });

/// Extension of an `f32` to `f64`, which is exact.
///
/// A NaN operand gives a NaN whose sign and payload are unspecified. The operation is free of
/// effects and never traps. It folds on a constant operand that is not a NaN.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Fpext {
    #[operand]
    operand: AnyFloat,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyFloat,
}

conversion_type_rules!(Fpext, ConversionRule::Extend);
fold_conversion!(Fpext, ConversionRule::Extend, Conversion::FloatResize);

/// Truncation of an `f64` to `f32`, rounding to nearest, ties to even. A value too large for `f32`
/// becomes an infinity, and one too small becomes a subnormal or a zero of the same sign.
///
/// A NaN operand gives a NaN whose sign and payload are unspecified. The operation is free of
/// effects and never traps. It folds on a constant operand that is not a NaN.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Fptrunc {
    #[operand]
    operand: AnyFloat,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyFloat,
}

conversion_type_rules!(Fptrunc, ConversionRule::Truncate);
fold_conversion!(Fptrunc, ConversionRule::Truncate, Conversion::FloatResize);

/// Reinterpretation of the bits of a value as another type of the same width: `i32` to `f32` and
/// back, and `i64` to `f64` and back. No other pair is allowed (not a type to itself, and not
/// `isize`, whose width depends on the target).
///
/// The operation is free of effects and never traps, and no bit pattern is poison, a NaN included.
/// It folds on a constant operand, and it folds a `bitcast` of a `bitcast` whose operand has the
/// outer result's type to that operand. This is MLIR's `arith.bitcast`, restricted to the scalars
/// the dialect has.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Bitcast {
    #[operand]
    operand: Or<AnyInteger, AnyFloat>,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: Or<AnyInteger, AnyFloat>,
}

conversion_type_rules!(Bitcast, ConversionRule::Bitcast);

impl Foldable for Bitcast {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let operands = crate::fold::operand_constants(self.as_operation());
        self.fold_with(&operands, results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        let op = self.as_operation();
        let ty = *self.get_ty();
        let context = op.context_rc();
        let rule = |context: &Context, from, to| {
            ConversionRule::Bitcast.check(context, from, to).is_none()
        };
        if crate::fold_float::fold_conversion(op, operands, results, ty, rule, Conversion::Bitcast)
            .is_ok()
        {
            return FoldResult::Ok(());
        }
        // A bitcast of a bitcast back to the original type is the original value, for every bit
        // pattern. Only a verified operation folds.
        let from = crate::fold::operand_type(op);
        let to = crate::fold::result_type(op, 0);
        if to != ty || !rule(&context, from, to) {
            return FoldResult::Failed;
        }
        let operand = self.operand().as_value_ref();
        let Some(def) = operand.borrow().get_defining_op() else {
            return FoldResult::Failed;
        };
        let def = def.borrow();
        let Some(inner) = def.downcast_ref::<Bitcast>() else {
            return FoldResult::Failed;
        };
        let original = inner.operand().as_value_ref();
        if original.borrow().ty() == to {
            results.push(OpFoldResult::Value(original));
            FoldResult::Ok(())
        } else {
            FoldResult::Failed
        }
    }
}
