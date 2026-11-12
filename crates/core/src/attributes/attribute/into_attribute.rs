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

/// Marks a type that is an attribute value argument: what [IntoAttribute] treats as a value to
/// create an attribute from, as opposed to a [ReadyAttr]. It is what keeps the two implementations
/// of [IntoAttribute] apart.
///
/// It is implemented for every type except [ReadyAttr] and a type that contains a [ReadyAttr] or an
/// unsized trait object inline. An auto trait is inherited through fields, and a `dyn Trait` only
/// implements it when the trait is named in the object type, so the pointer-like types below have
/// explicit positive implementations: whatever they point to, they are values.
///
/// If a type that is a value is rejected, implement the trait for it in the crate that defines it
/// (no feature flag is needed):
///
/// ```ignore
/// impl runic_core::AttributeValueArg for MyValue {}
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be used as an attribute value argument",
    label = "not an attribute value argument",
    note = "if `{Self}` is a value type, add `impl runic_core::AttributeValueArg for {Self} {{}}` \
            in the crate that defines it"
)]
pub auto trait AttributeValueArg {}

impl<A> !AttributeValueArg for ReadyAttr<A> {}

// A trait object does not implement an auto trait unless it names it, and `NonNull` is a pattern
// type that the compiler does not see through. Without these, a value that reaches either (through
// `Box`, `Rc`, `Arc`, `PhantomData`, or a reference) would not be a value argument.
impl<T: ?Sized> AttributeValueArg for core::ptr::NonNull<T> {}
impl<T: ?Sized> AttributeValueArg for core::marker::PhantomData<T> {}
impl<T: ?Sized> AttributeValueArg for *const T {}
impl<T: ?Sized> AttributeValueArg for *mut T {}
impl<T: ?Sized> AttributeValueArg for &T {}
impl<T: ?Sized> AttributeValueArg for &mut T {}
// `DynMetadata` holds a `NonNull` of the unsized-for-`MetaSized` `VTable`, which the `NonNull` impl
// above cannot name without the unstable `sized_hierarchy`.
impl<T: ?Sized> AttributeValueArg for core::ptr::DynMetadata<T> {}

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

    fn assert_value_arg<T: AttributeValueArg + ?Sized>() {}

    /// The value type of every attribute the builtin and debuginfo dialects register, and the types
    /// a value is built from, are value arguments (the check is the bound; it fails to compile).
    #[test]
    fn every_registered_value_type_is_a_value_argument() {
        use crate::dialects::{builtin::attributes::*, debuginfo::attributes::*};
        macro_rules! values {
            ($($attr:ty),* $(,)?) => {
                $(assert_value_arg::<<$attr as AttributeRegistration>::Value>();)*
            };
        }
        values!(
            LocalVariableArrayAttr,
            MemoryEffectArrayAttr,
            TypeArrayAttr,
            U32ArrayAttr,
            BoolAttr,
            BytesAttr,
            I8Attr,
            I16Attr,
            I32Attr,
            I64Attr,
            I128Attr,
            IntAttr,
            ListAttr,
            LocalVariableAttr,
            LocationAttr,
            OverflowAttr,
            SextAttr,
            SignatureAttr,
            SretAttr,
            ZextAttr,
            StringAttr,
            SymbolRefAttr,
            TypeAttr,
            UnitAttr,
            VisibilityAttr,
            crate::IdentAttr,
            CompileUnitAttr,
            ExpressionAttr,
            InlineCallChainAttr,
            SubprogramAttr,
            VariableAttr,
        );

        assert_value_arg::<crate::OperationName>();
        assert_value_arg::<crate::AttributeName>();
        assert_value_arg::<crate::Type>();
        assert_value_arg::<crate::AttributeRef>();
        assert_value_arg::<crate::OperationRef>();
        assert_value_arg::<crate::ValueRef>();
        assert_value_arg::<alloc::boxed::Box<dyn core::any::Any>>();
        assert_value_arg::<Rc<dyn core::any::Any>>();
        assert_value_arg::<&'static dyn core::any::Any>();
        assert_value_arg::<core::ptr::DynMetadata<dyn core::any::Any>>();
        assert_value_arg::<fn(u8)>();
        assert_value_arg::<core::marker::PhantomData<dyn core::any::Any>>();
        assert_value_arg::<alloc::vec::Vec<crate::Type>>();
        assert_value_arg::<alloc::boxed::Box<[u8]>>();
        assert_value_arg::<alloc::string::String>();
        assert_value_arg::<Rc<str>>();
        assert_value_arg::<core::num::NonZeroU32>();
        assert_value_arg::<crate::SmallVec<[crate::Type; 2]>>();
        assert_value_arg::<crate::CompactString>();
        assert_value_arg::<crate::interner::Symbol>();
        assert_value_arg::<*const u8>();
        assert_value_arg::<&'static str>();
    }

    #[test]
    fn a_list_value_builds_a_list_attribute() {
        use crate::dialects::builtin::attributes::{List, ListAttr};
        let context = Rc::new(Context::default());
        let _ = into::<ListAttr>(&context, List::default());
    }

    #[test]
    fn a_value_that_holds_an_attribute_handle_is_still_a_value() {
        let context = Rc::new(Context::default());
        let mut signature = Signature::new(&context, [context.i8()], []);
        signature.params_mut()[0] = AbiParam::zext(context.i8(), &context);
        assert!(!signature.params()[0].attributes().is_empty());

        let attr = into::<SignatureAttr>(&context, signature.clone());
        assert_eq!(**attr.borrow(), signature);
    }
}
