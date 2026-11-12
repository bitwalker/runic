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
        // An `IntAttr` whose value is not as wide as a constant of its type is invalid (see
        // `Operation::verify_int_attributes`); it is not materialized either.
        if value_ty != ty || crate::Int::width_for_type(ty) != Some(value.width()) {
            return None;
        }
        let op_builder = builder.create::<Constant, _>(span);
        let mut op = op_builder(value).ok()?;
        if ty.is_isize() {
            // The generated builder takes the attribute's value, not an `IntAttr`, so it infers
            // the type from the width (`i64` for an `isize`); retype after the op is inserted.
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
    use alloc::format;

    use crate::{
        Attribute, AttributeRef, BuilderExt, Int, Op, ReadyAttr, SourceSpan, Value,
        dialects::test::{Constant, TestDialect},
        testing::Test,
    };

    /// Whether `constant`'s `value` property is the attribute `attr`.
    fn holds(constant: &Constant, attr: AttributeRef) -> bool {
        let value = constant.as_operation().get_attribute("value").unwrap();
        AttributeRef::ptr_eq(&value, &attr)
    }

    /// A builder takes a ready-made attribute as it is: an `isize` constant keeps its type, which
    /// a 64-bit value alone would make `i64`.
    #[test]
    fn a_constant_is_built_from_a_ready_attribute() {
        let mut test = Test::new("f", |cx| ([cx.i32()], [cx.i32()]));
        let context = test.context_rc();
        let attr = context.int_attr(Int::from(-1i64), context.isize());
        let (ready, value) = {
            let mut builder = test.function_builder();
            let create = builder.builder_mut().create::<Constant, _>(SourceSpan::UNKNOWN);
            let ready = create(ReadyAttr(attr)).unwrap();
            let create = builder.builder_mut().create::<Constant, _>(SourceSpan::UNKNOWN);
            (ready, create(5i32).unwrap())
        };

        let constant = ready.borrow();
        assert!(holds(&constant, attr.as_attribute_ref()));
        assert_eq!(constant.value().ty(), context.isize());
        assert_eq!(constant.result().ty(), context.isize());
        assert!(constant.as_operation().verify().is_ok());
        let text = format!("{}", constant.as_operation());
        assert!(text.contains("test.constant -1 : isize"), "{text}");

        let constant = value.borrow();
        assert_eq!(*constant.get_value(), Int::from(5i32));
        assert_eq!(constant.value().ty(), context.i32());
        assert_eq!(constant.result().ty(), context.i32());
        assert!(constant.as_operation().verify().is_ok());
        let text = format!("{}", constant.as_operation());
        assert!(text.contains("test.constant 5 : i32"), "{text}");
    }

    /// A setter takes a ready-made attribute and installs that handle; a value still works.
    #[test]
    fn a_setter_takes_a_ready_attribute_or_a_value() {
        let mut test = Test::new("f", |cx| ([cx.i32()], [cx.i32()]));
        let context = test.context_rc();
        let mut constant = {
            let mut builder = test.function_builder();
            let create = builder.builder_mut().create::<Constant, _>(SourceSpan::UNKNOWN);
            create(Int::from(1i32)).unwrap()
        };

        let attr = context.int_attr(Int::from(7i32), context.i32());
        constant.borrow_mut().set_value(ReadyAttr(attr));
        assert!(holds(&constant.borrow(), attr.as_attribute_ref()));
        assert!(constant.borrow().as_operation().verify().is_ok());

        constant.borrow_mut().set_value(Int::from(1i32));
        assert!(!holds(&constant.borrow(), attr.as_attribute_ref()));
        assert_eq!(*constant.borrow().get_value(), Int::from(1i32));
        assert_eq!(constant.borrow().value().ty(), context.i32());
    }

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

        // an invalid `IntAttr`, an `i32` value typed `i64`, is not materialized at its type
        let invalid = context
            .create_attribute_with_type::<crate::dialects::builtin::attributes::IntAttr, _>(
                Int::from(-1i32),
                context.i64(),
            )
            .as_attribute_ref();
        assert!(
            dialect
                .materialize_constant(builder, invalid, context.i64(), SourceSpan::UNKNOWN)
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
