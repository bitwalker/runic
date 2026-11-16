use alloc::rc::Rc;

use runic_ir::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    effects::MemoryEffectOpInterface,
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

/// `count` of `value`, as a value of its width. A count is at most the width, which fits.
fn count(value: Int, count: fn(Int) -> u32) -> Option<Int> {
    Int::from_unsigned(value.width(), u128::from(count(value)))
}

/// Two's complement negation: `0 - operand`, wrapping.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
    )]
pub struct Neg {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Neg);

fold_one!(Neg, Operand, |_op| |[a]| Some(a.wrapping_neg()));

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
    /// Folds `bnot (bnot x)` to `x`, and the complement of a constant.
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let operand = self.operand().as_value_ref();
        if let Some(inner) = operand.borrow().get_defining_op()
            && let Some(inner) = inner.borrow().downcast_ref::<Bnot>()
            && inner.operand().ty() == self.result().ty()
        {
            results.push(OpFoldResult::Value(inner.operand().as_value_ref()));
            return FoldResult::Ok(());
        }
        let operands = crate::fold::operand_constants(self.as_operation());
        self.fold_with(&operands, results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        let op = self.as_operation();
        let folded = crate::fold::fold_ints(
            &op.context_rc(),
            operands,
            crate::fold::operand_type(op),
            crate::fold::result_type(op, 0),
            crate::fold::FoldsTo::Operand,
            |[a]| Some(!a),
        );
        crate::fold::push_folded(results, folded.map(|attr| [attr]))
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
        implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
    )]
pub struct Popcnt {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Popcnt);

fold_one!(Popcnt, Operand, |_op| |[a]| count(a, Int::count_ones));

/// The number of leading (most significant) zero bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
    )]
pub struct Clz {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Clz);

fold_one!(Clz, Operand, |_op| |[a]| count(a, Int::leading_zeros));

/// The number of trailing (least significant) zero bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
    )]
pub struct Ctz {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Ctz);

fold_one!(Ctz, Operand, |_op| |[a]| count(a, Int::trailing_zeros));

/// The number of leading (most significant) one bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
    )]
pub struct Clo {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Clo);

fold_one!(Clo, Operand, |_op| |[a]| count(a, Int::leading_ones));

/// The number of trailing (least significant) one bits, as a value of the operand's type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation (
        dialect = ArithDialect,
        traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
        implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
    )]
pub struct Cto {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_unary_op!(Cto);

fold_one!(Cto, Operand, |_op| |[a]| count(a, Int::trailing_ones));
