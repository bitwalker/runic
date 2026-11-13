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
        set_result_range(self.result().as_value_ref(), range.clone());
    }
}
