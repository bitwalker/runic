use alloc::boxed::Box;
use core::any::TypeId;

use super::{TypeImpl, TypeName};
use crate::{DialectRegistration, interner::Symbol, traits::TraitInfo};

/// A type that a dialect registers, so that its values can be interned in a
/// [Context](crate::Context).
pub trait TypeRegistration: TypeImpl + Sized {
    /// The dialect this type belongs to.
    type Dialect: DialectRegistration;

    /// The name of this type within its dialect.
    fn name() -> Symbol;

    /// The interfaces this type implements. Always includes `dyn Any`.
    fn traits() -> Box<[TraitInfo]>;
}

inventory::collect!(DialectTypeRegistrationInfo);

/// Registers a type with its dialect when the dialect is registered with a context.
#[doc(hidden)]
#[repr(transparent)]
pub struct DialectTypeRegistrationInfo(DialectTypeRegistrationEntry);

#[repr(C)]
struct DialectTypeRegistrationEntry {
    dialect: &'static str,
    dialect_type: TypeId,
    get_name: fn() -> Symbol,
    init: fn(Symbol) -> TypeName,
}

impl DialectTypeRegistrationInfo {
    pub const fn new<T: TypeRegistration>() -> Self {
        Self(DialectTypeRegistrationEntry {
            dialect: <T::Dialect as DialectRegistration>::NAMESPACE,
            dialect_type: TypeId::of::<T::Dialect>(),
            get_name: T::name,
            init: TypeName::new::<T>,
        })
    }

    /// Returns true if this type belongs to the dialect `dialect_type`, named `dialect`.
    pub(crate) fn belongs_to(&self, dialect_type: TypeId, dialect: &str) -> bool {
        self.0.dialect_type == dialect_type && self.0.dialect == dialect
    }

    /// The name of the registered type within its dialect.
    pub(crate) fn name(&self) -> Symbol {
        (self.0.get_name)()
    }

    /// Creates the registration record of the type, given the dialect's namespace.
    pub(crate) fn init(&self) -> fn(Symbol) -> TypeName {
        self.0.init
    }
}
