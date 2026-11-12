use core::{
    any::{Any, TypeId},
    marker::Unsize,
    ptr::{DynMetadata, Pointee, null},
};

#[doc(hidden)]
pub struct TraitInfo {
    /// The [TypeId] of the trait type, used as a unique key for [TraitImpl]s
    type_id: TypeId,
    /// The name of the trait type, for diagnostics
    type_name: &'static str,
    /// The [TypeId] of the concrete type whose vtable `metadata` is
    concrete_type_id: TypeId,
    /// The name of the concrete type, for diagnostics
    concrete_type_name: &'static str,
    /// Type-erased dyn metadata containing the trait vtable pointer for the concrete type
    ///
    /// This is transmuted to the correct trait type when reifying a `&dyn Trait` reference,
    /// which is safe as `DynMetadata` is always the same size for all types.
    metadata: DynMetadata<dyn Any>,
}
impl TraitInfo {
    pub fn new<T, Trait>() -> Self
    where
        T: Any + Unsize<Trait> + crate::verifier::Verifier<Trait>,
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        let type_id = TypeId::of::<Trait>();
        let ptr = null::<T>() as *const Trait;
        let (_, metadata) = ptr.to_raw_parts();
        Self {
            type_id,
            type_name: core::any::type_name::<Trait>(),
            concrete_type_id: TypeId::of::<T>(),
            concrete_type_name: core::any::type_name::<T>(),
            metadata: unsafe {
                core::mem::transmute::<DynMetadata<Trait>, DynMetadata<dyn Any>>(metadata)
            },
        }
    }

    #[inline(always)]
    pub const fn type_id(&self) -> &TypeId {
        &self.type_id
    }

    /// The name of the trait type.
    #[inline(always)]
    pub const fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The [TypeId] of the concrete type this info was built for, whose vtable it holds.
    #[inline(always)]
    pub const fn concrete_type_id(&self) -> &TypeId {
        &self.concrete_type_id
    }

    /// The name of the concrete type this info was built for.
    #[inline(always)]
    pub const fn concrete_type_name(&self) -> &'static str {
        self.concrete_type_name
    }

    /// Obtain the dyn metadata for `Trait` from this info.
    ///
    /// # Safety
    ///
    /// This is highly unsafe - you must guarantee that `Trait` is the same type as the one used
    /// to create this `TraitInfo` instance. In debug mode, errors like this will be caught, but
    /// in release builds, no checks are performed, and absolute havoc will result if you use this
    /// incorrectly.
    ///
    /// It is intended _only_ for use by generated code which has all of the type information
    /// available to it statically. It must be public so that operations can be defined in other
    /// crates.
    pub unsafe fn metadata_unchecked<Trait>(&self) -> DynMetadata<Trait>
    where
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        debug_assert!(self.type_id == TypeId::of::<Trait>());
        unsafe { core::mem::transmute(self.metadata) }
    }
}
impl Eq for TraitInfo {}
impl PartialEq for TraitInfo {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id
    }
}
impl PartialEq<TypeId> for TraitInfo {
    fn eq(&self, other: &TypeId) -> bool {
        self.type_id.eq(other)
    }
}
impl PartialOrd for TraitInfo {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialOrd<TypeId> for TraitInfo {
    fn partial_cmp(&self, other: &TypeId) -> Option<core::cmp::Ordering> {
        Some(self.type_id.cmp(other))
    }
}
impl Ord for TraitInfo {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.type_id.cmp(&other.type_id)
    }
}
