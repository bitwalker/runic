use alloc::rc::Rc;

use runic_core::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    effects::MemoryEffectOpInterface,
    matchers::Matcher,
    traits::*,
    *,
};

use crate::*;

/// Implements [InferTypeOpInterface] for a unary operation whose result has the operand's type.
macro_rules! infer_return_ty_for_unary_op {
    ($Op:ty) => {
        impl InferTypeOpInterface for $Op {
            fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
                let ty = self.operand().ty();
                self.result_mut().set_type(ty);
                Ok(())
            }
        }
    };
}

/// Two's complement negation: `0 - operand`, wrapping.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
pub struct Neg {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Neg);

/// Bitwise NOT. On `i1`, logical NOT.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
    )]
pub struct Bnot {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Bnot);

impl Foldable for Bnot {
    /// Folds `bnot (bnot x)` to `x`, and the negation of a constant `i1`.
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let operand = self.operand().as_value_ref();
        if let Some(inner) = operand.borrow().get_defining_op()
            && let Some(inner) = inner.borrow().downcast_ref::<Bnot>()
        {
            results.push(OpFoldResult::Value(inner.operand().as_value_ref()));
            return FoldResult::Ok(());
        }
        let operand = matchers::foldable_operand().matches(&self.operand().as_operand_ref());
        self.fold_with(&[operand], results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        let ty = self.result().ty();
        if ty.integer_width() != Some(1) {
            return FoldResult::Failed;
        }
        let Some(value) = super::coercions::constant_operand(&operands[0], ty) else {
            return FoldResult::Failed;
        };
        let result = self.as_operation().context_rc().int_attr(!value, ty);
        results.push(OpFoldResult::Attribute(result.as_attribute_ref()));
        FoldResult::Ok(())
    }
}

impl Canonicalizable for Bnot {
    fn get_canonicalization_patterns(
        rewrites: &mut patterns::RewritePatternSet,
        context: Rc<Context>,
    ) {
        rewrites.push(crate::canonicalization::SimplifyNotCompare::new(context));
    }
}

/// The number of set bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
pub struct Popcnt {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Popcnt);

/// The number of leading (most significant) zero bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
pub struct Clz {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Clz);

/// The number of trailing (least significant) zero bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
pub struct Ctz {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Ctz);

/// The number of leading (most significant) one bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
pub struct Clo {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Clo);

/// The number of trailing (least significant) one bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
pub struct Cto {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Cto);
