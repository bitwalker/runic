mod builders;
mod ops;
mod types;

pub use self::{builders::TestOpBuilder, ops::*, types::TestVectorType};
use crate::{
    AttributeRef, Builder, BuilderExt, Dialect, DialectInfo, OperationRef, SourceSpan, Type,
    attributes::IntegerLikeAttr, derive::DialectRegistration,
};

#[derive(DialectRegistration, Debug)]
pub struct TestDialect {
    #[dialect(info)]
    info: DialectInfo,
}

impl From<DialectInfo> for TestDialect {
    fn from(info: DialectInfo) -> Self {
        Self { info }
    }
}

impl Dialect for TestDialect {
    #[inline]
    fn info(&self) -> &DialectInfo {
        &self.info
    }

    fn materialize_constant(
        &self,
        builder: &mut dyn Builder,
        attr: AttributeRef,
        ty: Type,
        span: SourceSpan,
    ) -> Option<OperationRef> {
        use crate::Op;

        // Save the current insertion point
        let mut builder = crate::InsertionGuard::new(builder);

        // Only integer constants are supported for now
        if !ty.is_integer() {
            return None;
        }

        // Currently, we expect folds to produce integer-valued attributes
        let imm = attr
            .borrow()
            .as_attr()
            .as_trait::<dyn IntegerLikeAttr>()
            .map(|attr| (attr.as_immediate(), attr.ty()));
        if let Some((imm, imm_ty)) = imm {
            // If the immediate value is of the same type as the expected result type, we're ready
            // to materialize the constant
            if imm_ty == ty {
                let op_builder = builder.create::<Constant, _>(span);
                return op_builder(imm)
                    .ok()
                    .map(|op| op.borrow().as_operation().as_operation_ref());
            }

            // The immediate value has a different type than expected, but we can coerce types, so
            // long as the value fits in the target type
            if imm_ty.integer_width()? > ty.integer_width()? {
                return None;
            }

            let imm = imm.coerced_to(ty)?;

            let op_builder = builder.create::<Constant, _>(span);
            return op_builder(imm).ok().map(|op| op.borrow().as_operation().as_operation_ref());
        }

        None
    }
}
