//! Uniqued, immutable types owned by a [Context](crate::Context).

mod name;
mod registration;
mod uniquer;

use core::{
    any::{Any, TypeId},
    fmt,
    hash::{Hash, Hasher},
    ptr::{DynMetadata, NonNull, Pointee},
};

pub(crate) use self::uniquer::TypeUniquer;
pub use self::{
    name::TypeName,
    registration::{
        DefaultTypeFn, DialectTypeRegistrationInfo, ParseTypeFn, TypeParser, TypePrinter,
        TypeRegistration,
    },
};
use crate::{
    DialectRegistration,
    dialects::builtin::{
        BuiltinDialect,
        types::{FloatType, IntegerType, IsizeType, NeverType, UnknownType},
    },
};

/// The value a type is uniqued by: two types are the same type exactly when they have the same
/// Rust type and compare equal.
///
/// A type's value never changes after it is interned: it must not use interior mutability at all,
/// even in parts that `Eq` and `Hash` ignore. The uniquer finds a type by the hash and equality of
/// its value, and other code relies on an interned type staying exactly as it was created (a
/// late-filled child type could, for example, form a cycle that printing never leaves).
///
/// A type's `Eq` and `Hash` must not intern types or otherwise use the context: `Eq` runs while
/// the context's uniquer is borrowed.
pub trait TypeImpl: Any + Eq + Hash + fmt::Debug + 'static {}

/// What every interned type starts with.
#[repr(C)]
pub struct TypeHeader {
    name: TypeName,
    /// The hash this type is stored under in the context's uniquer.
    hash: u64,
}

/// An interned type: its header followed by its value.
///
/// `repr(C)` with the header first is what lets a [Type] point at the header whatever `T` is:
/// a pointer to the storage is a pointer to its header. The value follows the header at
/// [TypeStorage::VALUE_OFFSET], which depends on `T`'s alignment and is recorded in `T`'s
/// [TypeName].
#[repr(C)]
pub(crate) struct TypeStorage<T> {
    header: TypeHeader,
    value: T,
}

impl<T: TypeImpl> TypeStorage<T> {
    /// The offset of the value from the start of the storage, which depends on `T`'s alignment.
    pub(crate) const VALUE_OFFSET: usize = core::mem::offset_of!(Self, value);

    /// Creates the storage of a type whose value is `value`.
    ///
    /// Panics if `name` is not the registration record of `T`: [Type::downcast_ref] and
    /// [Type::as_trait] trust the name to describe the storage behind the header.
    pub(crate) fn new(name: TypeName, hash: u64, value: T) -> Self {
        assert!(
            name.is::<T>(),
            "type name `{name}` does not describe `{}`",
            core::any::type_name::<T>()
        );
        Self {
            header: TypeHeader { name, hash },
            value,
        }
    }
}

/// A handle to a type interned in a [Context](crate::Context).
///
/// Handles are cheap to copy, and two handles are equal exactly when they denote the same type.
/// A handle must not be used after its context is dropped.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Type(NonNull<TypeHeader>);

impl Type {
    /// Creates the handle of the type stored at `storage`.
    ///
    /// # Safety
    ///
    /// `storage` must point at an initialized `TypeStorage<T>` that lives, unmoved and unmodified,
    /// for as long as the handle (or any copy of it) is used, and the pointer must carry
    /// provenance over the whole storage.
    pub(crate) unsafe fn from_storage<T: TypeImpl>(storage: NonNull<TypeStorage<T>>) -> Self {
        Self(storage.cast())
    }

    #[inline]
    fn header(&self) -> &TypeHeader {
        // SAFETY: a handle is only created by `from_storage`, whose contract keeps the storage it
        // points at alive and unmodified while the handle is used; the storage is `repr(C)` with
        // the header first, so the pointer is a valid, aligned pointer to an initialized header.
        unsafe { self.0.as_ref() }
    }

    #[inline]
    pub(crate) fn stored_hash(&self) -> u64 {
        self.header().hash
    }

    /// The registration record of this type: its dialect, name and interfaces.
    #[inline]
    pub fn name(&self) -> &TypeName {
        &self.header().name
    }

    /// The address identifying this type.
    #[inline]
    pub fn as_ptr(&self) -> *const () {
        self.0.as_ptr().cast()
    }

    /// Returns true if this type is a `T`.
    #[inline]
    pub fn is<T: TypeImpl>(&self) -> bool {
        self.name().is::<T>()
    }

    /// Returns the value of this type if it is a `T`.
    pub fn downcast_ref<T: TypeImpl>(&self) -> Option<&T> {
        if !self.is::<T>() {
            return None;
        }
        // SAFETY: `TypeStorage::new` only accepts the name registered for its own `T`, so a
        // header whose name records `T`'s `TypeId` is the first field of a live
        // `TypeStorage<T>`, and the handle's pointer has provenance over all of it. The storage
        // is never mutated, so a shared reference to its value is valid while the handle is.
        Some(unsafe { &self.0.cast::<TypeStorage<T>>().as_ref().value })
    }

    /// Returns true if this type is an integer: an `iN` or `isize`.
    pub fn is_integer(&self) -> bool {
        self.is::<IntegerType>() || self.is_isize()
    }

    /// The width in bits of this type if it is an `iN`; `None` for `isize` and non-integers.
    pub fn integer_width(&self) -> Option<u32> {
        self.downcast_ref::<IntegerType>().map(IntegerType::width)
    }

    /// Returns true if this type is the pointer-width integer `isize`.
    pub fn is_isize(&self) -> bool {
        self.is::<IsizeType>()
    }

    /// Returns true if this type is a floating-point type.
    pub fn is_float(&self) -> bool {
        self.is::<FloatType>()
    }

    /// Returns true if this type is the unknown type `?`.
    pub fn is_unknown(&self) -> bool {
        self.is::<UnknownType>()
    }

    /// Returns true if this type is the bottom type `never`.
    pub fn is_never(&self) -> bool {
        self.is::<NeverType>()
    }

    /// Returns this type as the interface `Trait`, if its registration declares it.
    pub fn as_trait<Trait>(&self) -> Option<&Trait>
    where
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        let name = self.name();
        let metadata = name.trait_metadata::<Trait>()?;
        // SAFETY: the storage behind this header is a `TypeStorage<T>` for the `T` that `name`
        // was registered for (see `TypeStorage::new`); `name.value_offset()` is
        // `TypeStorage::<T>::VALUE_OFFSET`, so adding it stays inside the storage and yields a
        // properly aligned pointer to the initialized, never-mutated `T`. `metadata` is the
        // vtable of `Trait` for that same `T`, taken from `T`'s own trait table.
        unsafe {
            let value = self.0.as_ptr().cast::<u8>().add(name.value_offset());
            Some(&*core::ptr::from_raw_parts::<Trait>(value.cast::<()>(), metadata))
        }
    }
}

impl Eq for Type {}
impl PartialEq for Type {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        core::ptr::addr_eq(self.0.as_ptr(), other.0.as_ptr())
    }
}
impl Hash for Type {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ptr().addr().hash(state);
    }
}
impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name().debug_value(*self, f)
    }
}

/// A builtin type prints its own syntax. Any other type prints `!dialect.name`, followed by its
/// own syntax in angle brackets if it has a [TypePrinter].
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.name();
        let printer = self.as_trait::<dyn TypePrinter>();
        if name.dialect().as_str() == <BuiltinDialect as DialectRegistration>::NAMESPACE {
            let Some(printer) = printer else {
                panic!("builtin type `{name}` does not implement `TypePrinter`");
            };
            return printer.print(f);
        }
        write!(f, "!{}.{}", name.dialect(), name.name())?;
        if let Some(printer) = printer {
            f.write_str("<")?;
            printer.print(f)?;
            f.write_str(">")?;
        }
        Ok(())
    }
}

/// The hash a type with value `value` is stored under in the uniquer.
pub(crate) fn hash_of<T: TypeImpl>(value: &T) -> u64 {
    let mut hasher = crate::FxHasher::default();
    TypeId::of::<T>().hash(&mut hasher);
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc};
    use core::{any::Any, cell::Cell};

    use super::*;
    use crate::{Context, dialects::test::TestDialect, interner::Symbol, traits::TraitInfo};

    /// Defines a dialect that is not submitted to `inventory`, so that only a test that registers
    /// it explicitly (with `Context::get_or_register_dialect`) initialises it and its types.
    macro_rules! isolated_dialect {
        ($dialect:ident, $namespace:literal) => {
            #[derive(Debug)]
            struct $dialect {
                info: crate::DialectInfo,
            }
            impl crate::Dialect for $dialect {
                fn info(&self) -> &crate::DialectInfo {
                    &self.info
                }
            }
            impl crate::DialectRegistration for $dialect {
                const NAMESPACE: &'static str = $namespace;

                fn init(info: crate::DialectInfo) -> Self {
                    Self { info }
                }
            }
        };
    }

    isolated_dialect!(MislabelledTraitsDialect, "mislabelled_traits");

    /// Declares `Describe` with the vtable built for `WideType` instead of itself.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct MislabelledTraits(u8);
    impl TypeImpl for MislabelledTraits {}
    impl TypeRegistration for MislabelledTraits {
        type Dialect = MislabelledTraitsDialect;

        fn name() -> Symbol {
            Symbol::intern("mislabelled")
        }

        fn traits() -> Box<[TraitInfo]> {
            Box::from([
                TraitInfo::new::<Self, dyn Any>(),
                TraitInfo::new::<WideType, dyn Describe>(),
            ])
        }
    }
    inventory::submit!(DialectTypeRegistrationInfo::new::<MislabelledTraits>());

    #[test]
    #[should_panic(expected = "was built for another type")]
    fn a_trait_table_entry_built_for_another_type_is_rejected() {
        let context = Context::default();
        context.intern_type(MislabelledTraits(1));
    }

    /// An interface a type may implement, to exercise `Type::as_trait`.
    trait Describe {
        fn describe(&self) -> u128;
    }

    /// 16-aligned on purpose: more aligned than the header, which is 8-aligned.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct WideType {
        tag: u8,
        wide: u128,
    }
    impl TypeImpl for WideType {}
    impl TypeRegistration for WideType {
        type Dialect = TestDialect;

        fn name() -> Symbol {
            Symbol::intern("wide")
        }

        fn traits() -> Box<[TraitInfo]> {
            Box::from([TraitInfo::new::<Self, dyn Any>(), TraitInfo::new::<Self, dyn Describe>()])
        }
    }
    impl Describe for WideType {
        fn describe(&self) -> u128 {
            self.wide + self.tag as u128
        }
    }
    inventory::submit!(DialectTypeRegistrationInfo::new::<WideType>());

    #[derive(Debug, PartialEq, Eq, Hash)]
    struct OtherType {
        tag: u8,
        wide: u128,
    }
    impl TypeImpl for OtherType {}
    impl TypeRegistration for OtherType {
        type Dialect = TestDialect;

        fn name() -> Symbol {
            Symbol::intern("other")
        }

        fn traits() -> Box<[TraitInfo]> {
            Box::from([TraitInfo::new::<Self, dyn Any>()])
        }
    }
    inventory::submit!(DialectTypeRegistrationInfo::new::<OtherType>());

    #[test]
    fn equal_values_intern_to_one_handle() {
        let context = Context::default();
        let a = context.intern_type(WideType { tag: 1, wide: 7 });
        let b = context.intern_type(WideType { tag: 1, wide: 7 });
        let c = context.intern_type(WideType { tag: 2, wide: 7 });
        assert_eq!(a, b);
        assert_eq!(a.as_ptr(), b.as_ptr());
        assert_ne!(a, c);
    }

    #[test]
    fn rust_types_with_equal_fields_are_distinct_types() {
        let context = Context::default();
        let wide = context.intern_type(WideType { tag: 1, wide: 7 });
        let other = context.intern_type(OtherType { tag: 1, wide: 7 });
        assert_ne!(wide, other);
        assert!(wide.is::<WideType>() && !wide.is::<OtherType>());
        assert_eq!(wide.downcast_ref::<OtherType>(), None);
        assert_eq!(wide.name().name().as_str(), "wide");
        assert_eq!(wide.name().dialect().as_str(), "test");
    }

    #[test]
    fn a_handle_reads_back_its_value_and_its_interfaces() {
        let context = Context::default();
        let ty = context.intern_type(WideType {
            tag: 3,
            wide: u128::MAX - 3,
        });
        assert_eq!(
            ty.downcast_ref::<WideType>(),
            Some(&WideType {
                tag: 3,
                wide: u128::MAX - 3
            })
        );
        let described = ty.as_trait::<dyn Describe>().expect("WideType implements Describe");
        assert_eq!(described.describe(), u128::MAX);
        assert!(ty.name().implements::<dyn Describe>());
        let other = context.intern_type(OtherType { tag: 3, wide: 0 });
        assert!(other.as_trait::<dyn Describe>().is_none());
    }

    #[test]
    fn each_context_has_its_own_types() {
        let first = Context::default();
        let second = Context::default();
        let a = first.intern_type(WideType { tag: 1, wide: 1 });
        let b = second.intern_type(WideType { tag: 1, wide: 1 });
        assert_ne!(a, b);
        assert_eq!(a.downcast_ref::<WideType>(), b.downcast_ref::<WideType>());
    }

    #[derive(Debug)]
    struct Counted(Rc<Cell<usize>>);
    impl PartialEq for Counted {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }
    impl Eq for Counted {}
    impl core::hash::Hash for Counted {
        fn hash<H: core::hash::Hasher>(&self, _: &mut H) {}
    }
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    impl TypeImpl for Counted {}
    impl TypeRegistration for Counted {
        type Dialect = TestDialect;

        fn name() -> Symbol {
            Symbol::intern("counted")
        }

        fn traits() -> Box<[TraitInfo]> {
            Box::from([TraitInfo::new::<Self, dyn Any>()])
        }
    }
    inventory::submit!(DialectTypeRegistrationInfo::new::<Counted>());

    #[test]
    fn type_values_are_dropped_with_the_context_exactly_once() {
        let drops = Rc::new(Cell::new(0));
        {
            let context = Context::default();
            context.intern_type(Counted(drops.clone()));
            // A second, equal value is rejected by the uniquer and dropped immediately.
            context.intern_type(Counted(drops.clone()));
            assert_eq!(drops.get(), 1);
        }
        assert_eq!(drops.get(), 2);
    }

    /// Interns a type into its context when dropped, while that context is alive.
    #[derive(Debug)]
    struct InternsOnDrop(alloc::rc::Weak<Context>);
    impl PartialEq for InternsOnDrop {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }
    impl Eq for InternsOnDrop {}
    impl core::hash::Hash for InternsOnDrop {
        fn hash<H: core::hash::Hasher>(&self, _: &mut H) {}
    }
    impl Drop for InternsOnDrop {
        fn drop(&mut self) {
            if let Some(context) = self.0.upgrade() {
                context.intern_type(WideType { tag: 9, wide: 9 });
            }
        }
    }
    impl TypeImpl for InternsOnDrop {}
    impl TypeRegistration for InternsOnDrop {
        type Dialect = TestDialect;

        fn name() -> Symbol {
            Symbol::intern("interns_on_drop")
        }

        fn traits() -> Box<[TraitInfo]> {
            Box::from([TraitInfo::new::<Self, dyn Any>()])
        }
    }
    inventory::submit!(DialectTypeRegistrationInfo::new::<InternsOnDrop>());

    #[test]
    fn a_rejected_duplicate_is_dropped_after_the_uniquer_is_released() {
        let context = Rc::new(Context::default());
        let first = context.intern_type(InternsOnDrop(Rc::downgrade(&context)));
        // The duplicate is rejected, and its `Drop` interns another type into the same context.
        let second = context.intern_type(InternsOnDrop(Rc::downgrade(&context)));
        assert_eq!(first, second);
        let wide = context.intern_type(WideType { tag: 9, wide: 9 });
        assert_eq!(wide.downcast_ref::<WideType>(), Some(&WideType { tag: 9, wide: 9 }));
    }

    /// Aligned beyond the header, so the value is preceded by padding.
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[repr(C, align(32))]
    struct OverAligned {
        value: u64,
    }
    impl TypeImpl for OverAligned {}
    impl TypeRegistration for OverAligned {
        type Dialect = TestDialect;

        fn name() -> Symbol {
            Symbol::intern("over_aligned")
        }

        fn traits() -> Box<[TraitInfo]> {
            Box::from([TraitInfo::new::<Self, dyn Any>(), TraitInfo::new::<Self, dyn Describe>()])
        }
    }
    impl Describe for OverAligned {
        fn describe(&self) -> u128 {
            self.value as u128
        }
    }
    inventory::submit!(DialectTypeRegistrationInfo::new::<OverAligned>());

    #[test]
    fn a_value_aligned_past_the_header_is_found_at_its_recorded_offset() {
        let offset = TypeStorage::<OverAligned>::VALUE_OFFSET;
        assert_eq!(offset, 32);
        assert!(offset > core::mem::size_of::<TypeHeader>());

        let context = Context::default();
        let ty = context.intern_type(OverAligned { value: 0xfeed_beef });
        assert_eq!(ty.name().value_offset(), offset);
        let value = ty.downcast_ref::<OverAligned>().expect("an OverAligned type");
        assert_eq!(value.value, 0xfeed_beef);
        assert!((value as *const OverAligned).is_aligned());
        let described = ty.as_trait::<dyn Describe>().expect("OverAligned implements Describe");
        assert_eq!(described.describe(), 0xfeed_beef);
        assert_eq!(
            (described as *const dyn Describe).cast::<()>(),
            (value as *const OverAligned).cast::<()>()
        );
    }

    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect, implements(Describe))]
    struct EmptyType;
    impl Describe for EmptyType {
        fn describe(&self) -> u128 {
            1
        }
    }

    /// Zero-sized, yet aligned beyond the header.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect, implements(Describe))]
    #[repr(align(64))]
    struct EmptyAlignedType;
    impl Describe for EmptyAlignedType {
        fn describe(&self) -> u128 {
            64
        }
    }

    #[test]
    fn zero_sized_values_are_interned_and_read_back() {
        let context = Context::default();
        let empty = context.intern_type(EmptyType);
        assert_eq!(empty, context.intern_type(EmptyType));
        assert_eq!(empty.downcast_ref::<EmptyType>(), Some(&EmptyType));
        assert_eq!(empty.as_trait::<dyn Describe>().map(|d| d.describe()), Some(1));

        assert_eq!(TypeStorage::<EmptyAlignedType>::VALUE_OFFSET, 64);
        let aligned = context.intern_type(EmptyAlignedType);
        assert_ne!(aligned, empty);
        assert_eq!(aligned, context.intern_type(EmptyAlignedType));
        let value = aligned.downcast_ref::<EmptyAlignedType>().expect("an EmptyAlignedType");
        assert!((value as *const EmptyAlignedType).is_aligned());
        let described = aligned.as_trait::<dyn Describe>().expect("it implements Describe");
        assert_eq!(described.describe(), 64);
        assert!((described as *const dyn Describe).cast::<EmptyAlignedType>().is_aligned());
    }

    /// A tiny type registration: the type `$ty` named `$name` in `$dialect`.
    macro_rules! named_type {
        ($ty:ident($field:ty), $dialect:ident, $name:literal) => {
            #[derive(Debug, PartialEq, Eq, Hash)]
            struct $ty($field);
            impl TypeImpl for $ty {}
            impl TypeRegistration for $ty {
                type Dialect = $dialect;

                fn name() -> Symbol {
                    Symbol::intern($name)
                }

                fn traits() -> Box<[TraitInfo]> {
                    Box::from([TraitInfo::new::<Self, dyn Any>()])
                }
            }
        };
    }

    isolated_dialect!(ClashDialect, "type_clash");

    // Two Rust types that claim the same name in the same dialect.
    named_type!(ClashSmall(u8), ClashDialect, "clash");
    inventory::submit!(DialectTypeRegistrationInfo::new::<ClashSmall>());
    named_type!(ClashWide(u128), ClashDialect, "clash");
    inventory::submit!(DialectTypeRegistrationInfo::new::<ClashWide>());

    #[test]
    #[should_panic(
        expected = "dialect 'type_clash' registers the type name 'clash' for two Rust types"
    )]
    fn a_dialect_registering_one_name_for_two_rust_types_panics_when_registered() {
        let context = Context::default();
        context.get_or_register_dialect::<ClashDialect>();
    }

    isolated_dialect!(ClaimedNameDialect, "claimed_name");

    // `Impostor` names itself after `Claimed` but is never registered.
    named_type!(Claimed(u8), ClaimedNameDialect, "claimed");
    inventory::submit!(DialectTypeRegistrationInfo::new::<Claimed>());
    named_type!(Impostor(u8), ClaimedNameDialect, "claimed");

    #[test]
    fn a_name_registered_for_another_rust_type_is_not_handed_out() {
        let context = Context::default();
        let dialect = context.get_or_register_dialect::<ClaimedNameDialect>();
        assert!(dialect.registered_type_name::<Claimed>().is_some());
        assert!(dialect.registered_type_name::<Impostor>().is_none());
    }

    #[test]
    #[should_panic(expected = "is not registered")]
    fn interning_a_type_whose_name_another_rust_type_owns_panics() {
        let context = Context::default();
        context.intern_type(Impostor(1));
    }

    #[cfg(feature = "std")]
    std::thread_local! {
        /// The context `ReentrantDialect::init` interns into, if any; taken by the first `init`.
        static REENTRANT_CONTEXT: core::cell::RefCell<alloc::rc::Weak<Context>> =
            const { core::cell::RefCell::new(alloc::rc::Weak::new()) };
        /// The type `ReentrantDialect::init` interned.
        static REENTRANT_INNER: Cell<Option<Type>> = const { Cell::new(None) };
    }

    /// A dialect, not submitted to `inventory`, whose initialisation interns one of its own types
    /// into the context it is being registered with.
    #[cfg(feature = "std")]
    #[derive(Debug)]
    struct ReentrantDialect {
        info: crate::DialectInfo,
    }
    #[cfg(feature = "std")]
    impl crate::Dialect for ReentrantDialect {
        fn info(&self) -> &crate::DialectInfo {
            &self.info
        }
    }
    #[cfg(feature = "std")]
    impl crate::DialectRegistration for ReentrantDialect {
        const NAMESPACE: &'static str = "reentrant";

        fn init(info: crate::DialectInfo) -> Self {
            let context = REENTRANT_CONTEXT.with(|context| context.take().upgrade());
            if let Some(context) = context {
                let inner = context.intern_type(ReentrantType(1));
                REENTRANT_INNER.with(|slot| slot.set(Some(inner)));
            }
            Self { info }
        }
    }
    #[cfg(feature = "std")]
    named_type!(ReentrantType(u8), ReentrantDialect, "reentrant");
    #[cfg(feature = "std")]
    inventory::submit!(DialectTypeRegistrationInfo::new::<ReentrantType>());

    #[cfg(feature = "std")]
    #[test]
    fn interning_registers_the_dialect_outside_the_uniquer() {
        let context = Rc::new(Context::default());
        REENTRANT_CONTEXT.with(|slot| *slot.borrow_mut() = Rc::downgrade(&context));
        // The first use of the dialect registers it, and its `init` interns an equal value first.
        let ty = context.intern_type(ReentrantType(1));
        let inner = REENTRANT_INNER.with(Cell::take).expect("the dialect's init interned a type");
        assert_eq!(ty, inner, "the value interned during registration is not interned twice");
        assert_eq!(ty.downcast_ref::<ReentrantType>(), Some(&ReentrantType(1)));
        assert_eq!(ty, context.intern_type(ReentrantType(1)));
    }

    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect, name = "pair", implements(TypePrinter))]
    struct PairType {
        first: Type,
        second: Type,
    }
    impl TypePrinter for PairType {
        fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}, {}", self.first, self.second)
        }
    }

    #[derive(Debug, Default, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect)]
    struct MarkerType;

    #[test]
    fn derived_types_register_under_their_dialect_and_name() {
        let context = Context::default();
        let marker = context.intern_type(MarkerType);
        assert_eq!(marker.name().dialect().as_str(), "test");
        assert_eq!(marker.name().name().as_str(), "marker");
        let pair = context.intern_type(PairType {
            first: marker,
            second: marker,
        });
        assert_eq!(pair.name().name().as_str(), "pair");
        assert!(pair.name().implements::<dyn TypePrinter>());
        assert!(marker.name().default_fn().is_some(), "MarkerType is Default");
        assert!(pair.name().default_fn().is_none());
    }

    /// A unit-like enum with a parser, to check that a parser is picked up.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect)]
    enum ParsedType {
        Only,
    }
    impl TypeParser for ParsedType {
        fn parse(parser: &mut dyn crate::parse::Parser<'_>) -> crate::parse::ParseResult<Type> {
            Ok(parser.context().intern_type(ParsedType::Only))
        }
    }

    #[test]
    fn a_parser_is_recorded_exactly_for_types_that_have_one() {
        let context = Context::default();
        let parsed = context.intern_type(ParsedType::Only);
        assert_eq!(parsed.name().name().as_str(), "parsed");
        assert!(parsed.name().parse_fn().is_some());
        assert!(parsed.name().default_fn().is_none());
        let marker = context.intern_type(MarkerType);
        assert!(marker.name().parse_fn().is_none());
    }

    #[test]
    fn the_default_fn_interns_the_default_value() {
        let context = Context::default();
        let marker = context.intern_type(MarkerType);
        let default = marker.name().default_fn().expect("MarkerType is Default");
        assert_eq!(default(&context), marker);
    }

    #[test]
    fn dialect_types_display_with_their_dialect_prefix() {
        let context = Context::default();
        let marker = context.intern_type(MarkerType);
        assert_eq!(alloc::format!("{marker}"), "!test.marker");
        let pair = context.intern_type(PairType {
            first: marker,
            second: marker,
        });
        assert_eq!(alloc::format!("{pair}"), "!test.pair<!test.marker, !test.marker>");
        let nested = context.intern_type(PairType {
            first: pair,
            second: marker,
        });
        assert_eq!(
            alloc::format!("{nested}"),
            "!test.pair<!test.pair<!test.marker, !test.marker>, !test.marker>"
        );
    }
}
