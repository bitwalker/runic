use alloc::rc::Rc;

use super::AttributeRegistration;
use crate::{Context, UnsafeIntrusiveEntityRef};

/// What an operation's attribute parameter accepts: a value the attribute can be created from, or
/// an attribute that already exists.
///
/// A value is converted into the attribute's value type and the attribute is created with the
/// type derived from that value. An attribute whose type is not a function of its value (an
/// `isize` integer constant holds a 64-bit value, which would be typed `i64`) is created first and
/// passed as a [ReadyAttr].
pub trait IntoAttribute<A: AttributeRegistration>: Sized {
    /// Returns the attribute this argument stands for, creating it in `context` if it is a value.
    fn into_attribute(self, context: &Rc<Context>) -> UnsafeIntrusiveEntityRef<A>;
}

/// An existing attribute passed where a builder or setter would otherwise create one.
///
/// The attribute is used as it is: its type is not derived again from its value.
///
/// The attribute must be of the kind the parameter expects: `test.constant`'s `value` is an
/// `IntAttr`, so a `BoolAttr` is not accepted.
///
/// ```compile_fail
/// use runic_core::{ReadyAttr, UnsafeIntrusiveEntityRef};
/// use runic_core::dialects::{builtin::attributes::BoolAttr, test::Constant};
///
/// fn set(constant: &mut Constant, flag: UnsafeIntrusiveEntityRef<BoolAttr>) {
///     constant.set_value(ReadyAttr(flag));
/// }
/// ```
pub struct ReadyAttr<A>(pub UnsafeIntrusiveEntityRef<A>);

/// Implemented by every type except [ReadyAttr]; it is what keeps the two implementations of
/// [IntoAttribute] apart.
///
/// The negative implementation is on the wrapper rather than on the attribute handle, because an
/// auto trait is inherited through fields: an attribute value that holds a handle (a signature, a
/// list of attributes) is still a value. No value type holds a [ReadyAttr].
pub auto trait AttributeValueArg {}

impl<A> !AttributeValueArg for ReadyAttr<A> {}

/// The compiler does not see through the field of `NonNull` (a pattern type) when it implements an
/// auto trait, so without this a value holding a `Type`, a `Vec` or a `Box` would not be a value.
impl<T: ?Sized> AttributeValueArg for core::ptr::NonNull<T> {}

impl<A, V> IntoAttribute<A> for V
where
    A: AttributeRegistration,
    V: AttributeValueArg,
    <A as AttributeRegistration>::Value: From<V>,
{
    #[inline]
    fn into_attribute(self, context: &Rc<Context>) -> UnsafeIntrusiveEntityRef<A> {
        context.create_attribute::<A, V>(self)
    }
}

impl<A: AttributeRegistration> IntoAttribute<A> for ReadyAttr<A> {
    #[inline(always)]
    fn into_attribute(self, _context: &Rc<Context>) -> UnsafeIntrusiveEntityRef<A> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Attribute, Int,
        attributes::{BoolLikeAttr, IntegerLikeAttr},
        dialects::builtin::attributes::{AbiParam, BoolAttr, IntAttr, Signature, SignatureAttr},
    };

    fn into<A: AttributeRegistration>(
        context: &Rc<Context>,
        arg: impl IntoAttribute<A>,
    ) -> UnsafeIntrusiveEntityRef<A> {
        arg.into_attribute(context)
    }

    #[test]
    fn a_value_creates_an_attribute_typed_by_the_value() {
        let context = Rc::new(Context::default());

        let attr = into::<IntAttr>(&context, 5i32);
        assert_eq!(attr.borrow().as_int(), Int::from(5i32));
        assert_eq!(attr.borrow().ty(), context.i32());

        let attr = into::<IntAttr>(&context, Int::from(5u8));
        assert_eq!(attr.borrow().as_int(), Int::from(5u8));
        assert_eq!(attr.borrow().ty(), context.i8());

        let attr = into::<BoolAttr>(&context, true);
        assert!(attr.borrow().as_bool());
        assert_eq!(attr.borrow().ty(), context.i1());
    }

    #[test]
    fn a_ready_attribute_is_the_same_handle_with_its_type_untouched() {
        let context = Rc::new(Context::default());
        let ready = context.int_attr(Int::from(5i64), context.isize());

        let attr = into::<IntAttr>(&context, ReadyAttr(ready));
        assert!(UnsafeIntrusiveEntityRef::ptr_eq(&attr, &ready));
        assert_eq!(attr.borrow().ty(), context.isize());
    }

    /// The standard containers and pointers a value type is built from are value arguments.
    #[test]
    fn standard_containers_and_pointers_are_value_arguments() {
        fn is_value_arg<T: AttributeValueArg + ?Sized>() {}
        is_value_arg::<alloc::vec::Vec<crate::Type>>();
        is_value_arg::<alloc::boxed::Box<[u8]>>();
        is_value_arg::<alloc::string::String>();
        is_value_arg::<Rc<str>>();
        is_value_arg::<core::num::NonZeroU32>();
        is_value_arg::<crate::SmallVec<[crate::Type; 2]>>();
        is_value_arg::<crate::CompactString>();
        is_value_arg::<crate::interner::Symbol>();
        is_value_arg::<crate::AttributeRef>();
        is_value_arg::<*const u8>();
        is_value_arg::<&'static str>();
    }

    #[test]
    fn a_value_that_holds_an_attribute_handle_is_still_a_value() {
        let context = Rc::new(Context::default());
        let mut signature = Signature::new(&context, [context.i8()], []);
        signature.params[0] = AbiParam::zext(context.i8(), &context);
        assert!(!signature.params[0].attrs.is_empty());

        let attr = into::<SignatureAttr>(&context, signature.clone());
        assert_eq!(**attr.borrow(), signature);
    }
}
