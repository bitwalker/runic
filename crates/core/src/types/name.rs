use alloc::{boxed::Box, rc::Rc, vec::Vec};
use core::{
    any::TypeId,
    fmt,
    ptr::{DynMetadata, Pointee},
};

use super::{Type, TypeRegistration, TypeStorage};
use crate::{interner::Symbol, traits::TraitInfo};

/// The registration record of a type: its dialect, its name within the dialect, and the Rust type
/// and interfaces that implement it.
///
/// Two names are equal when their dialect, name and Rust type are.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeName(Rc<TypeInfo>);

struct TypeInfo {
    dialect: Symbol,
    name: Symbol,
    /// The [TypeId] of the Rust type whose values are this type's storage.
    type_id: TypeId,
    /// The interfaces this type implements, sorted by trait `TypeId` for binary search.
    traits: Box<[TraitInfo]>,
    /// `TypeStorage::<T>::VALUE_OFFSET` for the registered `T`.
    value_offset: usize,
    /// Formats a type whose name this is.
    debug: fn(Type, &mut fmt::Formatter<'_>) -> fmt::Result,
}

fn debug_value<T: TypeRegistration>(ty: Type, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let name = ty.name();
    let value = ty.downcast_ref::<T>().expect("a type is formatted by its own name");
    write!(f, "{}.{}({value:?})", name.dialect(), name.name())
}

impl TypeName {
    pub(crate) fn new<T: TypeRegistration>(dialect: Symbol) -> Self {
        let mut traits = Vec::from(T::traits());
        traits.sort_by_key(|ti| *ti.type_id());
        Self(Rc::new(TypeInfo {
            dialect,
            name: T::name(),
            type_id: TypeId::of::<T>(),
            traits: traits.into_boxed_slice(),
            value_offset: TypeStorage::<T>::VALUE_OFFSET,
            debug: debug_value::<T>,
        }))
    }

    /// The namespace of the dialect this type belongs to.
    #[inline]
    pub fn dialect(&self) -> Symbol {
        self.0.dialect
    }

    /// The name of this type within its dialect.
    #[inline]
    pub fn name(&self) -> Symbol {
        self.0.name
    }

    /// Returns true if `T` is the Rust type that implements this type.
    #[inline]
    pub fn is<T: 'static>(&self) -> bool {
        TypeId::of::<T>() == self.0.type_id
    }

    /// Returns true if this type implements `Trait`.
    pub fn implements<Trait>(&self) -> bool
    where
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        self.get::<Trait>().is_some()
    }

    /// The vtable of `Trait` for the Rust type of this type, if it implements `Trait`.
    pub(crate) fn trait_metadata<Trait>(&self) -> Option<DynMetadata<Trait>>
    where
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        // SAFETY: the entry was found under `TypeId::of::<Trait>()`, so it was created by
        // `TraitInfo::new::<_, Trait>` and its metadata is a `DynMetadata<Trait>`.
        self.get::<Trait>().map(|ti| unsafe { ti.metadata_unchecked::<Trait>() })
    }

    /// The offset of the value inside the storage of a type with this name.
    #[inline]
    pub(crate) fn value_offset(&self) -> usize {
        self.0.value_offset
    }

    #[inline]
    pub(crate) fn debug_value(&self, ty: Type, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0.debug)(ty, f)
    }

    fn get<Trait: ?Sized + 'static>(&self) -> Option<&TraitInfo> {
        let type_id = TypeId::of::<Trait>();
        let traits = self.0.traits.as_ref();
        traits
            .binary_search_by(|ti| ti.type_id().cmp(&type_id))
            .ok()
            .map(|index| &traits[index])
    }
}

impl fmt::Debug for TypeName {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.dialect(), self.name())
    }
}

impl Eq for TypeInfo {}

impl PartialEq for TypeInfo {
    fn eq(&self, other: &Self) -> bool {
        self.dialect == other.dialect && self.name == other.name && self.type_id == other.type_id
    }
}

impl PartialOrd for TypeInfo {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TypeInfo {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.dialect
            .cmp(&other.dialect)
            .then_with(|| self.name.cmp(&other.name))
            .then_with(|| self.type_id.cmp(&other.type_id))
    }
}

impl core::hash::Hash for TypeInfo {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.dialect.hash(state);
        self.name.hash(state);
        self.type_id.hash(state);
    }
}
