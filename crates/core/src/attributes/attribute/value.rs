use core::clone::CloneToUninit;

use crate::{Context, Type, any::AsAny};

pub trait AttributeValue: AsAny + CloneToUninit {}

impl<T: AsAny + CloneToUninit> AttributeValue for T {}

/// Checks, in debug builds, that the types an attribute value holds may enter IR with `context`.
///
/// Every attribute's constructor calls it on the new value. It does nothing, except for the
/// values that are types (`TypeAttr`) or arrays of types (`TypeArrayAttr`), whose types must have
/// been interned by `context` and must not be open.
///
/// It runs once, when the attribute is created: a value changed afterwards is not checked again,
/// and neither is a type held in any other shape (a public field such as an `AbiParam`'s type, or
/// a type inside a signature). [Operation::verify](crate::Operation::verify) checks every type an
/// operation's attributes hold, in every build.
#[doc(hidden)]
pub trait DebugAssertIrValue {
    fn debug_assert_ir_value(&self, context: &Context);
}

impl<T: ?Sized> DebugAssertIrValue for T {
    #[inline(always)]
    default fn debug_assert_ir_value(&self, _context: &Context) {}
}

impl DebugAssertIrValue for Type {
    #[track_caller]
    fn debug_assert_ir_value(&self, context: &Context) {
        context.debug_assert_ir_type(*self, "the value of a type attribute");
    }
}

impl dyn AttributeValue {
    /// Returns true if this attribute is an instance of type `T`
    pub fn is<T: AsAny>(&self) -> bool {
        self.as_any().is::<T>()
    }

    /// Attempts to downcast a `&dyn Attribute` to `&T`, if the value is an instance of type `T`.
    pub fn downcast_ref<T: AsAny>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }

    /// Attempts to downcast this attribute value reference to `&mut T`, if the value is an instance
    /// of type `T`.
    ///
    /// Returns `None` if this value is not of type `T`.
    pub fn downcast_mut<T: AsAny>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut::<T>()
    }
}
