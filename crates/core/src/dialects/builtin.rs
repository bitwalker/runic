pub mod attributes;
mod builders;
pub mod canonicalization;
mod ops;
pub mod types;

pub use self::{
    builders::{BuiltinOpBuilder, ComponentBuilder, FunctionBuilder, ModuleBuilder, WorldBuilder},
    ops::*,
};
use crate::{
    AttributeRef, Builder, BuilderExt, Dialect, DialectInfo, OperationRef, ReadyAttr, SourceSpan,
    Type, derive::DialectRegistration,
};

#[derive(DialectRegistration, Debug)]
pub struct BuiltinDialect {
    #[dialect(info)]
    info: DialectInfo,
}

impl From<DialectInfo> for BuiltinDialect {
    fn from(info: DialectInfo) -> Self {
        Self { info }
    }
}

impl Dialect for BuiltinDialect {
    #[inline]
    fn info(&self) -> &DialectInfo {
        &self.info
    }

    /// Materializes a [BytesAttr](attributes::BytesAttr) of `N` bytes as a [Bytes] constant of
    /// type `array<i8; N>`; nothing else, and no bytes as another type.
    fn materialize_constant(
        &self,
        builder: &mut dyn Builder,
        attr: AttributeRef,
        ty: Type,
        span: SourceSpan,
    ) -> Option<OperationRef> {
        let mut builder = crate::InsertionGuard::new(builder);

        let bytes = attr.try_downcast_attr::<attributes::BytesAttr>().ok()?;
        let len = bytes.borrow().len() as u64;
        let context = builder.context();
        if ty != context.array_type(context.i8(), len) {
            return None;
        }
        let op_builder = builder.create::<Bytes, _>(span);
        let op = op_builder(ReadyAttr(bytes)).ok()?;
        Some(op.as_operation_ref())
    }
}
