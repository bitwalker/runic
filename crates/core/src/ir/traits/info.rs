use core::{
    any::{Any, TypeId},
    marker::Unsize,
    ptr::{DynMetadata, Pointee, null},
};

#[doc(hidden)]
pub struct TraitInfo {
    /// The [TypeId] of the trait type, used as a unique key for [TraitImpl]s
    type_id: TypeId,
    /// The [TypeId] of the concrete type whose vtable `metadata` is
    concrete_type_id: TypeId,
    /// The names of the trait type and of the concrete type, for diagnostics. A function rather
    /// than the names themselves, so that they cost one pointer instead of two `&str`s.
    type_names: fn() -> (&'static str, &'static str),
    /// Type-erased dyn metadata containing the trait vtable pointer for the concrete type
    ///
    /// This is transmuted to the correct trait type when reifying a `&dyn Trait` reference,
    /// which is safe as `DynMetadata` is always the same size for all types.
    metadata: DynMetadata<dyn Any>,
}
/// Panics unless every entry of `traits`, the trait table of the `kind` (`type`, `attribute`,
/// `operation`) implemented by `T`, was built for `T`: a lookup pairs a `T` with the vtable it
/// finds in the table.
#[track_caller]
pub(crate) fn assert_trait_table_built_for<T: 'static>(traits: &[TraitInfo], kind: &str) {
    for ti in traits {
        assert!(
            *ti.concrete_type_id() == TypeId::of::<T>(),
            "the `{}` entry in the trait table of {kind} `{}` was built for another type, `{}`",
            ti.type_name(),
            core::any::type_name::<T>(),
            ti.concrete_type_name()
        );
    }
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
            concrete_type_id: TypeId::of::<T>(),
            type_names: type_names::<T, Trait>,
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
    pub fn type_name(&self) -> &'static str {
        (self.type_names)().0
    }

    /// The [TypeId] of the concrete type this info was built for, whose vtable it holds.
    #[inline(always)]
    pub const fn concrete_type_id(&self) -> &TypeId {
        &self.concrete_type_id
    }

    /// The name of the concrete type this info was built for.
    #[inline(always)]
    pub fn concrete_type_name(&self) -> &'static str {
        (self.type_names)().1
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
/// The names of `Trait` and of `T`.
fn type_names<T: ?Sized, Trait: ?Sized>() -> (&'static str, &'static str) {
    (core::any::type_name::<Trait>(), core::any::type_name::<T>())
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

#[cfg(test)]
mod tests {
    use core::{
        any::{Any, TypeId},
        mem::size_of,
        ptr::DynMetadata,
    };

    use super::TraitInfo;

    /// A trait table holds one entry per interface of every op, attribute and type, so an entry
    /// stores only what the lookup and the vtable check need: the two `TypeId`s and the vtable.
    /// The names for diagnostics cost one function pointer.
    #[test]
    fn trait_info_holds_two_type_ids_a_vtable_and_one_function_pointer() {
        assert_eq!(
            size_of::<TraitInfo>(),
            2 * size_of::<TypeId>() + size_of::<DynMetadata<dyn Any>>() + size_of::<fn()>()
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(size_of::<TraitInfo>(), 48);
    }

    trait Named {}
    struct Concrete;
    impl Named for Concrete {}

    #[test]
    fn trait_info_names_its_trait_and_concrete_type() {
        let info = TraitInfo::new::<Concrete, dyn Named>();
        assert!(info.type_name().ends_with("Named"));
        assert!(info.concrete_type_name().ends_with("Concrete"));
    }
}
