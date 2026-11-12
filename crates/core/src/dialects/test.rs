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

        // Folds produce integer-valued attributes of the type they want: a signless value has
        // no default extension, so a value of another type is not materialized.
        let (value, value_ty) = {
            let attr = attr.borrow();
            let attr = attr.as_attr().as_trait::<dyn IntegerLikeAttr>()?;
            (attr.as_int(), attr.ty())
        };
        if value_ty != ty {
            return None;
        }
        let op_builder = builder.create::<Constant, _>(span);
        let mut op = op_builder(value).ok()?;
        if ty.is_isize() {
            // The builder infers the type from the value's width, which for an `isize` is `i64`
            use crate::{Attribute, Value};

            let mut constant = op.borrow_mut();
            constant.value_mut().set_type(ty);
            constant.result_mut().set_type(ty);
        }
        Some(op.borrow().as_operation().as_operation_ref())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Int, SourceSpan, Value, dialects::test::TestDialect, testing::Test};

    /// A constant is materialized only at the type of its attribute: a signless value has no
    /// default extension to another width.
    #[test]
    fn materialize_constant_requires_the_attribute_type() {
        let mut test = Test::new("f", |cx| ([cx.i32()], [cx.i32()]));
        let context = test.context_rc();
        let dialect = context.get_or_register_dialect::<TestDialect>();
        let mut builder = test.function_builder();
        let builder = builder.builder_mut();

        let narrow = context.int_attr(Int::from(1i8), context.i8()).as_attribute_ref();
        assert!(
            dialect
                .materialize_constant(builder, narrow, context.i32(), SourceSpan::UNKNOWN)
                .is_none()
        );

        for (value, ty) in [
            (Int::from(-1i32), context.i32()),
            (Int::from_bits(7, 3), context.int_type(7)),
            (Int::from(5i64), context.isize()),
        ] {
            let attr = context.int_attr(value, ty).as_attribute_ref();
            let op = dialect
                .materialize_constant(builder, attr, ty, SourceSpan::UNKNOWN)
                .unwrap_or_else(|| panic!("{value:?} : {ty}"));
            let op = op.borrow();
            let attr = op.get_attribute("value").unwrap();
            assert_eq!(attr.borrow().as_int(), Some(value), "{ty}");
            assert_eq!(attr.borrow().ty(), ty, "{ty}");
            assert_eq!(op.results()[0].borrow().ty(), ty, "{ty}");
        }
    }
}
