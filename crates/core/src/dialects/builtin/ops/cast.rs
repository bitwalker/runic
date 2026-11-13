use crate::{
    Context, OpPrinter, Report, Value,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::{BuiltinDialect, attributes::TypeAttr},
    effects::MemoryEffectOpInterface,
    traits::{AnyType, InferIntRangeInterface, InferTypeOpInterface, UnaryOp},
};

#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = BuiltinDialect,
    traits(UnaryOp),
    implements(
        InferTypeOpInterface,
        InferIntRangeInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct UnrealizedConversionCast {
    #[operand]
    operand: AnyType,
    #[attr]
    ty: TypeAttr,
    #[result]
    result: AnyType,
}

impl InferTypeOpInterface for UnrealizedConversionCast {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = *self.get_ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

impl InferIntRangeInterface for UnrealizedConversionCast {
    fn infer_result_ranges(
        &self,
        operands: &[crate::IntegerValueRange],
        set_result_range: &mut dyn FnMut(crate::ValueRef, crate::IntegerValueRange),
    ) {
        let Some(range) = operands.first() else {
            return;
        };
        // The cast says nothing about how a value of one type becomes a value of another (a
        // zero extension, a sign extension, a truncation or anything else), so the source's
        // range is a fact about the result only when both are integers of one width. Otherwise
        // the result has the range of its type, untracked for a type the analysis does not
        // track.
        let top = crate::ConstantIntRange::max_for_type(self.result().ty());
        let range = match range.range() {
            None => range.clone(),
            Some(source) if !top.is_untracked() && source.width() == top.width() => range.clone(),
            Some(_) => crate::IntegerValueRange::new(top),
        };
        set_result_range(self.result().as_value_ref(), range);
    }
}
