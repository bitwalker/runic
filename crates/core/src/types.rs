//! Uniqued, immutable types owned by a [Context](crate::Context).

mod alias;
mod component;
mod invariants;
mod mapper;
mod name;
pub(crate) mod rec;
mod registration;
mod uniquer;

use core::{
    any::{Any, TypeId},
    cell::RefCell,
    fmt::{self, Write},
    hash::{Hash, Hasher},
    ptr::{DynMetadata, NonNull, Pointee},
};

pub(crate) use self::{
    alias::Aliased,
    rec::{RecGroupData, RecGroupTable},
    uniquer::TypeUniquer,
};
pub use self::{
    alias::TypeAliases,
    component::TypeComponent,
    invariants::TypeInvariants,
    mapper::{TypeMapError, TypeMapper},
    name::TypeName,
    rec::{RecGroup, RecGroupError, collect_rec_groups},
    registration::{
        AliasNameFn, DefaultTypeFn, DialectTypeRegistrationInfo, ParseTypeFn, PrintTypeFn,
        TypeAliasName, TypeParser, TypePrinter, TypeRegistration,
    },
};
use crate::dialects::builtin::types::{
    ArrayType, EnumType, FloatType, FunctionType, IntegerType, IsizeType, NeverType, PointerType,
    StructType, UnknownType,
};

/// The value a type is uniqued by: two types are the same type exactly when they have the same
/// Rust type and compare equal.
///
/// A type's value is immutable once its handle is published: it must not use interior
/// mutability at all, even in parts that `Eq` and `Hash` ignore. The uniquer finds a type by the
/// hash and equality of its value, and other code relies on a type it can reach staying exactly as
/// it was when its handle was first handed out (a child type filled in later could, for example,
/// form a cycle that printing never leaves).
///
/// A type's `Eq` and `Hash` must not intern types or otherwise use the context: `Eq` runs while
/// the context's uniquer is borrowed. They must not panic either: a panic while the uniquer is
/// probing, combined with a `Drop` that interns, aborts. They compare and hash the types the value
/// holds as [Type]s, by handle and stored hash, and must not read those types' values, nor may
/// the value's `Drop`, [Self::visit_subtypes] or [Self::replace_subtypes] (including the results
/// `replace` returns): a held type may be a recursion-group member whose value is still being
/// filled in, and while a group is being built these receive the handles of members whose values
/// are not final yet.
///
/// Type storage can be cyclic: a recursion-group member holds, directly or not, the handles of
/// the members it refers to, itself included. Any method of a type, an interface method among
/// them, that walks the types the value holds beyond the ones it holds directly must therefore
/// stop at group members ([Type::rec_group]) or track the types it has visited; otherwise it may
/// never return.
///
/// A type also exposes the types its value holds directly, through [Self::visit_subtypes] and
/// [Self::replace_subtypes]. `#[derive(DialectType)]` implements both field by field, so every
/// field of a derived type must be a [TypeComponent]; a field that is not does not compile:
///
/// ```compile_fail
/// use runic_core::{derive::DialectType, dialects::test::TestDialect};
///
/// /// Not a `TypeComponent`.
/// #[derive(Debug, Default, PartialEq, Eq, Hash)]
/// struct Opaque;
///
/// #[derive(Debug, Default, PartialEq, Eq, Hash, DialectType)]
/// #[ty(dialect = TestDialect)]
/// struct HoldsOpaqueType {
///     field: Opaque,
/// }
/// ```
pub trait TypeImpl: Any + Eq + Hash + fmt::Debug + 'static {
    /// Calls `visit` with each type this value holds directly, in a fixed order.
    ///
    /// It must be pure: it must not intern types, use the context, or panic.
    fn visit_subtypes(&self, visit: &mut dyn FnMut(Type));

    /// A copy of this value in which each directly held type `t` is `replace(t)`; `replace` is
    /// called in the order of [Self::visit_subtypes], and everything else is unchanged.
    ///
    /// It must be pure: it must not intern types, use the context, or panic.
    fn replace_subtypes(&self, replace: &mut dyn FnMut(Type) -> Type) -> Self
    where
        Self: Sized;
}

/// What every interned type starts with.
///
/// Every field is present in every build profile, so the layout of a type's storage does not
/// depend on the profile.
#[repr(C)]
pub(crate) struct TypeHeader {
    name: TypeName,
    /// The hash this type is stored under: in the context's uniquer for an ordinary type, and
    /// computed from its group and index for a group member (see [Type::stored_hash]).
    hash: u64,
    /// The recursion group this type is a member of, if any.
    group: Option<NonNull<RecGroupData>>,
    /// The id of the context that interned this type; see [Type::owner_id].
    owner: u32,
    /// Properties of the type computed when it is interned; see [TypeFlags].
    flags: TypeFlags,
    /// This type's index in `group`; 0 if it is not a member.
    index: u32,
}

/// Properties of a type, computed once when it is interned and kept in its header.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct TypeFlags(u32);

impl TypeFlags {
    /// No flag set.
    pub(crate) const NONE: Self = Self(0);
    /// The type is open: it is a recursion variable, or a type it holds directly is open.
    pub(crate) const OPEN: Self = Self(1);

    #[inline]
    fn contains(self, flags: Self) -> bool {
        self.0 & flags.0 == flags.0
    }
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

    /// Creates the storage of a type whose value is `value`, interned by the context whose id is
    /// `owner`, with the properties `flags`.
    ///
    /// Panics if `name` is not the registration record of `T`: [Type::downcast_ref] and
    /// [Type::as_trait] trust the name to describe the storage behind the header.
    pub(crate) fn new(name: TypeName, hash: u64, owner: u32, flags: TypeFlags, value: T) -> Self {
        Self::assert_named(&name);
        Self {
            header: TypeHeader {
                name,
                hash,
                group: None,
                owner,
                flags,
                index: 0,
            },
            value,
        }
    }

    /// Creates the storage of member `index` of the recursion group `group`, owned by the
    /// context whose id is `owner`. A member is never open.
    ///
    /// Panics if `name` is not the registration record of `T`, as [Self::new] does.
    pub(crate) fn new_member(
        name: TypeName,
        hash: u64,
        owner: u32,
        group: NonNull<RecGroupData>,
        index: u32,
        value: T,
    ) -> Self {
        Self::assert_named(&name);
        Self {
            header: TypeHeader {
                name,
                hash,
                group: Some(group),
                owner,
                flags: TypeFlags::NONE,
                index,
            },
            value,
        }
    }

    fn assert_named(name: &TypeName) {
        assert!(
            name.is::<T>(),
            "type name `{name}` does not describe `{}`",
            core::any::type_name::<T>()
        );
    }
}

/// A handle to a type interned in a [Context](crate::Context).
///
/// Handles are cheap to copy, and two handles are equal exactly when they denote the same type.
///
/// A handle belongs to the context that interned it, and may only be used with that context: a
/// type of one context is never equal to a type of another, even one that prints the same. Code
/// that puts a type into IR or into another type checks this with
/// [Context::owns_type](crate::Context::owns_type) in debug builds, and the IR verifier checks it
/// in every build.
///
/// A handle must not be used after its context is dropped. That cannot be detected: reading the
/// owner from the handle is already a use of the dropped context's memory.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Type(NonNull<TypeHeader>);

impl Type {
    /// Creates the handle of the type stored at `storage`.
    ///
    /// # Safety
    ///
    /// `storage` must point at an initialized `TypeStorage<T>` that lives, unmoved, for as long as
    /// the handle (or any copy of it) is used, and that is immutable once the handle is
    /// published; the pointer must carry provenance over the whole storage, for writing as well
    /// as reading.
    ///
    /// A handle is published when it is first handed to code other than the one creating it. The
    /// one exception is a recursion-group member, which `Context::rec_group` publishes only when
    /// it returns it: until then the member's value may be written, and the member's handle may
    /// be stored in other types' values and its header read (by hashing, equality, flag
    /// computation and ownership checks), but no reference to its value may exist when it is
    /// written, and nothing reads its value.
    pub(crate) unsafe fn from_storage<T: TypeImpl>(storage: NonNull<TypeStorage<T>>) -> Self {
        Self(storage.cast())
    }

    #[inline]
    fn header(&self) -> &TypeHeader {
        // SAFETY: a handle is only created by `from_storage`, whose contract keeps the storage it
        // points at alive while the handle is used; the storage is `repr(C)` with the header
        // first, so the pointer is a valid, aligned pointer to an initialized header. The header
        // is never written after the storage is created (only an unpublished member's value is),
        // and the reference covers the header alone, not the value after it.
        unsafe { self.0.as_ref() }
    }

    /// A pointer to this type's storage as a `TypeStorage<T>`, with the handle's provenance.
    ///
    /// The pointer is only valid to use if this type's Rust type is `T`.
    #[inline]
    fn as_storage_ptr<T: TypeImpl>(&self) -> *mut TypeStorage<T> {
        debug_assert!(self.is::<T>(), "a type's storage is accessed as its own Rust type");
        self.0.cast::<TypeStorage<T>>().as_ptr()
    }

    /// The hash this type hashes as: computed from its value for an ordinary type, and from its
    /// group's bodies and its index for a group member.
    #[inline]
    pub(crate) fn stored_hash(&self) -> u64 {
        self.header().hash
    }

    /// Returns true if this type is open: a recursion variable, or a type that holds an open
    /// type. Open types define recursion groups (see
    /// [Context::rec_group](crate::Context::rec_group)) and may not appear in IR. A group member
    /// is never open.
    #[inline]
    pub fn is_open(&self) -> bool {
        self.header().flags.contains(TypeFlags::OPEN)
    }

    /// The recursion group this type is a member of, if it is one.
    #[inline]
    pub fn rec_group(&self) -> Option<RecGroup> {
        self.header().group.map(RecGroup)
    }

    /// This type's index in its recursion group, if it is a member of one.
    #[inline]
    pub fn rec_index(&self) -> Option<u32> {
        let header = self.header();
        header.group.map(|_| header.index)
    }

    /// The open body of this type in its recursion group, if it is a member of one: the type the
    /// group was defined with at this member's index, in which references to the group's members
    /// are recursion variables.
    #[inline]
    pub fn rec_body(&self) -> Option<Type> {
        let group = self.rec_group()?;
        Some(group.bodies()[self.rec_index()? as usize])
    }

    /// The id of the context that interned this type.
    #[inline]
    pub(crate) fn owner_id(&self) -> u32 {
        self.header().owner
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
        // is immutable once the handle is published, so a shared reference to its value is
        // valid while the handle is.
        Some(unsafe { &self.0.cast::<TypeStorage<T>>().as_ref().value })
    }

    /// This type as an integer type `iN`, if it is one; `None` for `isize`.
    ///
    /// The structural queries on builtin types go through these accessors, rather than through
    /// [Self::downcast_ref], so that what counts as each kind of type is decided in one place.
    #[inline]
    pub fn as_integer(&self) -> Option<&IntegerType> {
        self.downcast_ref::<IntegerType>()
    }

    /// This type as a floating-point type, if it is one.
    #[inline]
    pub fn as_float(&self) -> Option<&FloatType> {
        self.downcast_ref::<FloatType>()
    }

    /// This type as a pointer type, if it is one.
    #[inline]
    pub fn as_pointer(&self) -> Option<&PointerType> {
        self.downcast_ref::<PointerType>()
    }

    /// This type as an array type, if it is one.
    #[inline]
    pub fn as_array(&self) -> Option<&ArrayType> {
        self.downcast_ref::<ArrayType>()
    }

    /// This type as a struct type, if it is one.
    #[inline]
    pub fn as_struct(&self) -> Option<&StructType> {
        self.downcast_ref::<StructType>()
    }

    /// This type as an enum type, if it is one.
    #[inline]
    pub fn as_enum(&self) -> Option<&EnumType> {
        self.downcast_ref::<EnumType>()
    }

    /// This type as a function type, if it is one.
    #[inline]
    pub fn as_function(&self) -> Option<&FunctionType> {
        self.downcast_ref::<FunctionType>()
    }

    /// Returns true if this type is an integer: an `iN` or `isize`.
    pub fn is_integer(&self) -> bool {
        self.as_integer().is_some() || self.is_isize()
    }

    /// The width in bits of this type if it is an `iN`; `None` for `isize` and non-integers.
    pub fn integer_width(&self) -> Option<u32> {
        self.as_integer().map(IntegerType::width)
    }

    /// Returns true if this type is the pointer-width integer `isize`.
    pub fn is_isize(&self) -> bool {
        self.is::<IsizeType>()
    }

    /// Returns true if this type is a floating-point type.
    pub fn is_float(&self) -> bool {
        self.as_float().is_some()
    }

    /// Returns true if this type is the unknown type `?`.
    pub fn is_unknown(&self) -> bool {
        self.is::<UnknownType>()
    }

    /// Returns true if this type is the bottom type `never`.
    pub fn is_never(&self) -> bool {
        self.is::<NeverType>()
    }

    /// Returns true if this type is a pointer.
    pub fn is_pointer(&self) -> bool {
        self.as_pointer().is_some()
    }

    /// The type this pointer points to; `None` for an opaque pointer and for non-pointers.
    pub fn pointee(&self) -> Option<Type> {
        self.as_pointer().and_then(PointerType::pointee)
    }

    /// Returns true if this type is an array.
    pub fn is_array(&self) -> bool {
        self.as_array().is_some()
    }

    /// Returns true if this type is a struct.
    pub fn is_struct(&self) -> bool {
        self.as_struct().is_some()
    }

    /// Returns true if this type is an enum.
    pub fn is_enum(&self) -> bool {
        self.as_enum().is_some()
    }

    /// Returns true if this type is a function type.
    pub fn is_function(&self) -> bool {
        self.as_function().is_some()
    }

    /// Calls `visit` with each type this type's value holds directly, in the order of its
    /// [TypeImpl::visit_subtypes].
    pub fn visit_subtypes(&self, visit: &mut dyn FnMut(Type)) {
        self.name().visit_fn()(*self, visit)
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
        // properly aligned pointer to the initialized, published, immutable `T`. `metadata` is the
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
/// A type hashes as its stored hash, which is computed from its value, and so from its child
/// types' stored hashes in turn: equal structure gives an equal hash in any context, and the
/// iteration order of a map keyed by types does not depend on where the types happen to be
/// allocated. The hash is built from the structure, Rust [TypeId]s and interned
/// [Symbol](crate::interner::Symbol) indices, so it is repeatable from run to run only for the
/// same build and the same order of symbol interning (see `hash_of`). Equality is still
/// identity, which implies an equal hash.
impl Hash for Type {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.stored_hash());
    }
}
/// Writes the [Display](fmt::Display) form, which stays finite for a recursive type; the derived
/// `Debug` of a value holding a group member would follow the cycle forever.
impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Prints a type with [TypeFormatter::ty], with no alias table.
///
/// A recursion-group member is written as `<reference> = <body>` at the top level, and as its
/// reference when nested, e.g. `!list@0.0 = struct list<i32, ptr<!list@0.0>>` (see
/// [TypeFormatter::ty]). This form is for diagnostics: the IR printer never produces it and the
/// parser does not accept it.
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TypeFormatter::new(f).ty(*self)
    }
}

/// Writes types as text: what a [TypePrinter] prints its type's syntax to.
///
/// It is written to as a [fmt::Write], and writes each type it is handed with [Self::ty], which
/// is the one place that decides how a type is written, at the top level or inside another type.
pub struct TypeFormatter<'a, 'f> {
    f: &'a mut fmt::Formatter<'f>,
    /// How many types are being written around the one being written now.
    depth: u32,
    /// The aliases recursion-group members are written as, if any.
    aliases: Option<&'a RefCell<TypeAliases>>,
}

impl<'a, 'f> TypeFormatter<'a, 'f> {
    /// A printer writing to `f`.
    pub fn new(f: &'a mut fmt::Formatter<'f>) -> Self {
        Self {
            f,
            depth: 0,
            aliases: None,
        }
    }

    /// A printer writing to `f` that writes every recursion-group member as its alias in
    /// `aliases`, naming the member's group there first if it is not named yet.
    pub fn with_aliases(f: &'a mut fmt::Formatter<'f>, aliases: &'a RefCell<TypeAliases>) -> Self {
        Self {
            f,
            depth: 0,
            aliases: Some(aliases),
        }
    }

    /// Writes `ty`.
    ///
    /// A builtin type is written as its own syntax. Any other type is written as `!dialect.name`,
    /// followed by its own syntax in angle brackets if it has a [TypePrinter].
    ///
    /// With an alias table, a recursion-group member is written as `!alias`, wherever it is.
    ///
    /// With none, a member nested inside the type being formatted is written as a reference:
    /// `!name@G.I` if its body suggests a name (see [TypeAliasName]) that is a bare identifier,
    /// otherwise `!rec@G.I`, for member `I` of the group created `G`-th in its context. A member
    /// that is the type being formatted is written as its reference, ` = `, and its body:
    /// `!list@0.0 = struct list<i32, ptr<!list@0.0>>`. That keeps the text of a recursive type
    /// finite and tells a member from the ordinary type with the same value. It is meant for
    /// diagnostics: the IR printer never writes it, and the parser does not accept it.
    pub fn ty(&mut self, ty: Type) -> fmt::Result {
        if let Some(group) = ty.rec_group() {
            if let Some(aliases) = self.aliases {
                let mut aliases = aliases.borrow_mut();
                return write!(self, "!{}", aliases.alias_of(ty));
            }
            self.member_reference(ty, group)?;
            if self.depth > 0 {
                return Ok(());
            }
            self.write_str(" = ")?;
        }
        self.definition(ty)
    }

    /// Writes the body of `ty`: its own syntax, even if it is a recursion-group member, with the
    /// types it holds written by [Self::ty].
    ///
    /// This is how a declaration writes the member it declares: `type !list = <definition>;`.
    ///
    /// It is crate-internal: a dialect printer that wrote a member it holds with it would
    /// write that member's body, and so, through the cycle, recurse forever.
    pub(crate) fn definition(&mut self, ty: Type) -> fmt::Result {
        self.depth += 1;
        let result = self.body(ty);
        self.depth -= 1;
        result
    }

    /// Writes the reference to the group member `ty` of `group`.
    fn member_reference(&mut self, ty: Type, group: RecGroup) -> fmt::Result {
        let alias = ty.name().alias_name_fn().and_then(|alias_name| alias_name(ty));
        match alias {
            Some(alias) if is_bare_identifier(alias.as_str()) => write!(self, "!{alias}"),
            _ => self.write_str("!rec"),
        }?;
        write!(self, "@{}.{}", group.id(), ty.rec_index().expect("a group member has an index"))
    }

    /// Writes `ty` itself, without the reference form of a nested member.
    fn body(&mut self, ty: Type) -> fmt::Result {
        let name = ty.name();
        let print = name.print_fn();
        if name.is_builtin() {
            let print = print.expect("registration requires builtin types to have a printer");
            return print(ty, self);
        }
        write!(self, "!{}.{}", name.dialect(), name.name())?;
        if let Some(print) = print {
            self.write_str("<")?;
            print(ty, self)?;
            self.write_str(">")?;
        }
        Ok(())
    }

    /// Writes `types` with [Self::ty], separated by `, `.
    pub fn types(&mut self, types: impl IntoIterator<Item = Type>) -> fmt::Result {
        for (i, ty) in types.into_iter().enumerate() {
            if i > 0 {
                self.write_str(", ")?;
            }
            self.ty(ty)?;
        }
        Ok(())
    }
}

impl fmt::Write for TypeFormatter<'_, '_> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.f.write_str(s)
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.f.write_char(c)
    }

    #[inline]
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> fmt::Result {
        self.f.write_fmt(args)
    }
}

/// Panics, in debug builds, if `ty` is open. `role` describes where `ty` enters IR, for the panic
/// message. Where a context is at hand, `Context::debug_assert_ir_type` checks ownership too.
#[inline]
#[track_caller]
pub(crate) fn debug_assert_closed(ty: Type, role: &str) {
    debug_assert!(
        !ty.is_open(),
        "{role} is `{ty}`, an open type: a recursion variable is only meaningful inside the \
         definition of a recursion group"
    );
}

/// Returns true if `text` is a bare identifier: an ASCII letter or `_`, then ASCII letters,
/// digits and `_`.
pub(crate) fn is_bare_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The hash a type with value `value` is stored under in the uniquer.
///
/// It is the same in every run of the same build for the same value: it hashes `T`'s [TypeId],
/// which is fixed for a build, and the value, whose child types hash as their own stored hashes
/// and whose [Symbol](crate::interner::Symbol)s hash as their interned indices. Those indices
/// depend on the order in which strings are first interned in the process, so the hash is
/// reproducible given the same order of construction, but differs between builds and may
/// differ when strings are interned in another order.
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

    isolated_dialect!(PrinterlessDialect, "printerless");

    /// A type with no [TypePrinter], registered with no dialect but its own isolated one.
    #[derive(Debug, Default, PartialEq, Eq, Hash)]
    struct PrinterlessType;
    impl TypeImpl for PrinterlessType {
        fn visit_subtypes(&self, _: &mut dyn FnMut(Type)) {}

        fn replace_subtypes(&self, _: &mut dyn FnMut(Type) -> Type) -> Self {
            PrinterlessType
        }
    }
    impl TypeRegistration for PrinterlessType {
        type Dialect = PrinterlessDialect;

        fn name() -> Symbol {
            Symbol::intern("printerless")
        }

        fn traits() -> Box<[TraitInfo]> {
            Box::from([TraitInfo::new::<Self, dyn Any>()])
        }
    }

    #[test]
    #[should_panic(
        expected = "builtin type `runic_core::types::tests::PrinterlessType` must implement \
                    `TypePrinter`"
    )]
    fn naming_a_printerless_type_in_the_builtin_dialect_panics() {
        let builtin =
            <crate::dialects::builtin::BuiltinDialect as crate::DialectRegistration>::NAMESPACE;
        let _ = TypeName::new::<PrinterlessType>(Symbol::intern(builtin));
    }

    #[test]
    fn a_printerless_type_is_named_outside_the_builtin_dialect() {
        let name = TypeName::new::<PrinterlessType>(Symbol::intern("printerless"));
        assert!(name.print_fn().is_none());
    }

    isolated_dialect!(FieldsOnlyDialect, "fields_only");

    /// Has a field, but neither a printer nor a default: no text could tell its values apart.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = FieldsOnlyDialect)]
    struct FieldsOnlyType(u8);

    #[test]
    #[should_panic(expected = "type `runic_core::types::tests::FieldsOnlyType` has no \
                               `TypePrinter`, so it must implement `Default`")]
    fn a_dialect_type_with_neither_a_printer_nor_a_default_is_rejected() {
        Context::default().get_or_register_dialect::<FieldsOnlyDialect>();
    }

    isolated_dialect!(PrintOnlyDialect, "print_only");

    /// Prints its field, but cannot parse it back.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = PrintOnlyDialect)]
    struct PrintOnlyType(u8);
    impl TypePrinter for PrintOnlyType {
        fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
            write!(printer, "{}", self.0)
        }
    }

    #[test]
    #[should_panic(
        expected = "type `runic_core::types::tests::PrintOnlyType` implements `TypePrinter` but \
                    not `TypeParser`"
    )]
    fn a_dialect_type_with_a_printer_but_no_parser_is_rejected() {
        Context::default().get_or_register_dialect::<PrintOnlyDialect>();
    }

    isolated_dialect!(ParseOnlyDialect, "parse_only");

    /// Parses a body, but prints none.
    #[derive(Debug, Default, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = ParseOnlyDialect)]
    struct ParseOnlyType;
    impl TypeParser for ParseOnlyType {
        fn parse(parser: &mut dyn crate::parse::Parser<'_>) -> crate::parse::ParseResult<Type> {
            Ok(parser.context().intern_type(ParseOnlyType))
        }
    }

    #[test]
    #[should_panic(
        expected = "type `runic_core::types::tests::ParseOnlyType` implements `TypeParser` but \
                    not `TypePrinter`"
    )]
    fn a_dialect_type_with_a_parser_but_no_printer_is_rejected() {
        Context::default().get_or_register_dialect::<ParseOnlyDialect>();
    }

    isolated_dialect!(MislabelledTraitsDialect, "mislabelled_traits");

    /// Declares `Describe` with the vtable built for `WideType` instead of itself.
    #[derive(Debug, Default, PartialEq, Eq, Hash)]
    struct MislabelledTraits(u8);
    impl TypeImpl for MislabelledTraits {
        fn visit_subtypes(&self, _: &mut dyn FnMut(Type)) {}

        fn replace_subtypes(&self, _: &mut dyn FnMut(Type) -> Type) -> Self {
            MislabelledTraits(self.0)
        }
    }
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
    #[should_panic(
        expected = "trait table of type `runic_core::types::tests::MislabelledTraits` was built \
                    for another type, `runic_core::types::tests::WideType`"
    )]
    fn a_trait_table_entry_built_for_another_type_is_rejected() {
        let context = Context::default();
        context.intern_type(MislabelledTraits(1));
    }

    /// An interface a type may implement, to exercise `Type::as_trait`.
    trait Describe {
        fn describe(&self) -> u128;
    }

    /// 16-aligned on purpose: more aligned than the header, which is 8-aligned.
    #[derive(Debug, Default, PartialEq, Eq, Hash)]
    struct WideType {
        tag: u8,
        wide: u128,
    }
    impl TypeImpl for WideType {
        fn visit_subtypes(&self, _: &mut dyn FnMut(Type)) {}

        fn replace_subtypes(&self, _: &mut dyn FnMut(Type) -> Type) -> Self {
            WideType {
                tag: self.tag,
                wide: self.wide,
            }
        }
    }
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

    #[derive(Debug, Default, PartialEq, Eq, Hash)]
    struct OtherType {
        tag: u8,
        wide: u128,
    }
    impl TypeImpl for OtherType {
        fn visit_subtypes(&self, _: &mut dyn FnMut(Type)) {}

        fn replace_subtypes(&self, _: &mut dyn FnMut(Type) -> Type) -> Self {
            OtherType {
                tag: self.tag,
                wide: self.wide,
            }
        }
    }
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

    #[test]
    fn a_context_owns_exactly_the_types_it_interned() {
        let first = Context::default();
        let second = Context::default();
        let a = first.ptr_type(first.i32());
        let b = second.ptr_type(second.i32());
        assert!(first.owns_type(a) && first.owns_type(first.i32()));
        assert!(second.owns_type(b) && second.owns_type(second.unknown_type()));
        assert!(!first.owns_type(b) && !second.owns_type(a));
        assert!(!first.owns_type(second.i32()));
        assert_ne!(a.owner_id(), b.owner_id());
    }

    /// A type hashes by structure, so that maps keyed by types iterate in the same order in
    /// every run that builds the same types in the same order.
    #[test]
    fn equal_types_of_two_contexts_hash_alike() {
        use crate::{StructField, interner::Symbol};

        fn hash(ty: Type) -> u64 {
            let mut hasher = crate::FxHasher::default();
            ty.hash(&mut hasher);
            hasher.finish()
        }
        fn build(context: &Context) -> alloc::vec::Vec<Type> {
            let i32 = context.i32();
            let point = context.struct_type(StructType::named(
                Symbol::intern("point"),
                [StructField::new(i32).named(Symbol::intern("x")), StructField::new(i32)],
            ));
            let pair = context.intern_type(PairType {
                first: point,
                second: context.ptr_type(point),
            });
            alloc::vec![
                i32,
                context.int_type(24),
                context.ptr_type(i32),
                context.array_type(context.ptr_type(point), 3),
                point,
                context.function_type([point, i32], [context.never_type()]),
                pair,
            ]
        }

        let (first, second) = (Context::default(), Context::default());
        let (a, b) = (build(&first), build(&second));
        for (a, b) in a.iter().zip(&b) {
            assert_ne!(a, b);
            assert_eq!(hash(*a), hash(*b), "{a}");
        }
        assert_ne!(hash(a[0]), hash(a[1]));
    }

    #[test]
    fn a_handle_is_one_word_and_stays_on_its_thread() {
        assert_eq!(core::mem::size_of::<Type>(), 8);
        assert_eq!(core::mem::size_of::<Option<Type>>(), 8);
    }

    // `Type` is neither `Send` nor `Sync`: either impl would make `some_item` ambiguous here.
    const _: fn() = || {
        trait AmbiguousIfImpl<A> {
            fn some_item() {}
        }
        impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
        struct IfSend;
        impl<T: ?Sized + Send> AmbiguousIfImpl<IfSend> for T {}
        struct IfSync;
        impl<T: ?Sized + Sync> AmbiguousIfImpl<IfSync> for T {}
        let _ = <Type as AmbiguousIfImpl<_>>::some_item;
    };

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "the pointee of a pointer type is `i32`, a type of another context")]
    fn a_pointer_to_a_type_of_another_context_panics() {
        let (first, second) = (Context::default(), Context::default());
        first.ptr_type_in(Some(second.i32()), 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "the element of an array type is `i8`, a type of another context")]
    fn an_array_of_a_type_of_another_context_panics() {
        let (first, second) = (Context::default(), Context::default());
        first.array_type(second.i8(), 4);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a field of a struct type is `i64`, a type of another context")]
    fn a_struct_with_a_field_of_another_context_panics() {
        use crate::dialects::builtin::types::StructField;

        let (first, second) = (Context::default(), Context::default());
        first.struct_type(StructType::new([
            StructField::new(first.i32()),
            StructField::new(second.i64()),
        ]));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a parameter of a function type is `f32`, a type of another context")]
    fn a_function_with_a_parameter_of_another_context_panics() {
        let (first, second) = (Context::default(), Context::default());
        first.function_type([second.f32()], []);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a result of a function type is `never`, a type of another context")]
    fn a_function_with_a_result_of_another_context_panics() {
        let (first, second) = (Context::default(), Context::default());
        first.function_type([first.i1()], [second.never_type()]);
    }

    #[derive(Debug, Default)]
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
    impl TypeImpl for Counted {
        fn visit_subtypes(&self, _: &mut dyn FnMut(Type)) {}

        fn replace_subtypes(&self, _: &mut dyn FnMut(Type) -> Type) -> Self {
            Counted(self.0.clone())
        }
    }
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
    #[derive(Debug, Default)]
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
    impl TypeImpl for InternsOnDrop {
        fn visit_subtypes(&self, _: &mut dyn FnMut(Type)) {}

        fn replace_subtypes(&self, _: &mut dyn FnMut(Type) -> Type) -> Self {
            InternsOnDrop(self.0.clone())
        }
    }
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
    #[derive(Debug, Default, PartialEq, Eq, Hash)]
    #[repr(C, align(32))]
    struct OverAligned {
        value: u64,
    }
    impl TypeImpl for OverAligned {
        fn visit_subtypes(&self, _: &mut dyn FnMut(Type)) {}

        fn replace_subtypes(&self, _: &mut dyn FnMut(Type) -> Type) -> Self {
            OverAligned { value: self.value }
        }
    }
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
        // The 40-byte header, rounded up to the value's 32-byte alignment.
        assert_eq!(core::mem::size_of::<TypeHeader>(), 40);
        let offset = TypeStorage::<OverAligned>::VALUE_OFFSET;
        assert_eq!(offset, 64);
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

    #[derive(Debug, Default, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect, implements(Describe))]
    struct EmptyType;
    impl Describe for EmptyType {
        fn describe(&self) -> u128 {
            1
        }
    }

    /// Zero-sized, yet aligned beyond the header.
    #[derive(Debug, Default, PartialEq, Eq, Hash, crate::derive::DialectType)]
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
            #[derive(Debug, Default, PartialEq, Eq, Hash)]
            struct $ty($field);
            impl TypeImpl for $ty {
                fn visit_subtypes(&self, _: &mut dyn FnMut(Type)) {}

                fn replace_subtypes(&self, _: &mut dyn FnMut(Type) -> Type) -> Self {
                    Self(self.0)
                }
            }
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
        /// The dialect that `ReentrantDialect::init` found registered once it had interned.
        static REENTRANT_DIALECT: core::cell::RefCell<Option<Rc<dyn crate::Dialect>>> =
            const { core::cell::RefCell::new(None) };
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
                let dialect = context.get_or_register_dialect::<ReentrantDialect>();
                REENTRANT_DIALECT.with(|slot| *slot.borrow_mut() = Some(dialect));
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
        // The dialect registered while its own registration was under way is the one kept.
        let inner = REENTRANT_DIALECT.with(|slot| slot.take()).expect("init found its dialect");
        let registered = context.get_or_register_dialect::<ReentrantDialect>();
        assert!(
            core::ptr::addr_eq(Rc::as_ptr(&inner), Rc::as_ptr(&registered)),
            "the dialect has two instances"
        );
    }

    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect, name = "pair")]
    struct PairType {
        first: Type,
        second: Type,
    }
    impl TypePrinter for PairType {
        fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
            printer.types([self.first, self.second])
        }
    }
    impl TypeParser for PairType {
        fn parse(parser: &mut dyn crate::parse::Parser<'_>) -> crate::parse::ParseResult<Type> {
            let first = parser.parse_type()?.into_inner();
            parser.parse_comma()?;
            let second = parser.parse_type()?.into_inner();
            Ok(parser.context().intern_type(PairType { first, second }))
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
        assert!(pair.name().print_fn().is_some());
        assert!(marker.name().print_fn().is_none());
        assert!(marker.name().default_fn().is_some(), "MarkerType is Default");
        assert!(pair.name().default_fn().is_none());
        assert!(!marker.name().is_builtin() && !pair.name().is_builtin());
        assert!(context.i32().name().is_builtin());
    }

    /// A unit-like enum with a parser, to check that a parser is picked up.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect)]
    enum ParsedType {
        Only,
    }
    impl TypePrinter for ParsedType {
        fn print(&self, _: &mut TypeFormatter<'_, '_>) -> fmt::Result {
            Ok(())
        }
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

    /// Implements `TypePrinter` without declaring it in `#[ty(..)]`.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = TestDialect)]
    struct UndeclaredPrinterType(u8);
    impl TypePrinter for UndeclaredPrinterType {
        fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
            write!(printer, "{}", self.0)
        }
    }
    impl TypeParser for UndeclaredPrinterType {
        fn parse(parser: &mut dyn crate::parse::Parser<'_>) -> crate::parse::ParseResult<Type> {
            use crate::parse::ParserExt;

            let value = parser.parse_decimal_integer::<u8>()?.into_inner();
            Ok(parser.context().intern_type(UndeclaredPrinterType(value)))
        }
    }

    #[test]
    fn a_printer_is_found_without_being_declared() {
        let context = Context::default();
        let ty = context.intern_type(UndeclaredPrinterType(7));
        assert!(ty.name().print_fn().is_some());
        assert_eq!(alloc::format!("{ty}"), "!test.undeclared_printer<7>");
        assert!(context.intern_type(MarkerType).name().print_fn().is_none());
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

    /// The types `ty` holds directly, in visit order.
    fn subtypes(ty: Type) -> alloc::vec::Vec<Type> {
        let mut types = alloc::vec::Vec::new();
        ty.visit_subtypes(&mut |subtype| types.push(subtype));
        types
    }

    /// A struct type of three fields, `struct<i8, ptr<i16>, array<i32; 2>>`.
    fn three_fields(context: &Context) -> (Type, [Type; 3]) {
        use crate::StructField;

        let fields = [
            context.i8(),
            context.ptr_type(context.i16()),
            context.array_type(context.i32(), 2),
        ];
        let ty = context.struct_type(StructType::new(fields.map(StructField::new)));
        (ty, fields)
    }

    #[test]
    fn builtin_types_visit_what_they_hold_directly_in_order() {
        let context = Context::default();
        let (i8, i16, i32, i64) = (context.i8(), context.i16(), context.i32(), context.i64());
        assert_eq!(subtypes(context.ptr_type(i32)), [i32]);
        assert_eq!(subtypes(context.opaque_ptr_type()), []);
        assert_eq!(subtypes(context.ptr_type_in(None, 3)), []);
        assert_eq!(subtypes(context.array_type(i8, 4)), [i8]);
        let (point, fields) = three_fields(&context);
        assert_eq!(subtypes(point), fields);
        assert_eq!(subtypes(context.function_type([i8, i16], [i32, i64])), [i8, i16, i32, i64]);
        for scalar in [
            context.i1(),
            i8,
            context.int_type(24),
            context.isize(),
            context.f32(),
            context.f64(),
            context.unknown_type(),
            context.never_type(),
        ] {
            assert_eq!(subtypes(scalar), [], "{scalar}");
        }
    }

    #[test]
    fn mapping_with_the_identity_returns_the_same_type() {
        let context = Context::default();
        let i32 = context.i32();
        let (point, _) = three_fields(&context);
        for ty in [
            context.ptr_type(i32),
            context.opaque_ptr_type(),
            context.array_type(context.i8(), 4),
            point,
            context.function_type([context.i8(), context.i16()], [i32, context.i64()]),
            i32,
            context.isize(),
            context.f64(),
            context.unknown_type(),
            context.never_type(),
            context.intern_type(crate::dialects::test::TestVectorType::new(i32, 4)),
        ] {
            assert_eq!(context.map_subtypes(ty, &mut |subtype| subtype), ty, "{ty}");
        }
    }

    #[test]
    fn mapping_replaces_only_the_directly_held_types() {
        use core::num::NonZeroU16;

        use crate::{StructField, StructRepr};

        let context = Context::default();
        let (i32, i64) = (context.i32(), context.i64());
        let point = |x: Type| {
            context.struct_type(
                StructType::named(
                    Symbol::intern("point"),
                    [
                        StructField::new(x).named(Symbol::intern("x")),
                        StructField::new(context.ptr_type(i32))
                            .named(Symbol::intern("y"))
                            .aligned(NonZeroU16::new(4).unwrap()),
                    ],
                )
                .with_repr(StructRepr::Align(NonZeroU16::new(8).unwrap())),
            )
        };
        let (before, expected) = (point(i32), point(i64));
        assert_eq!(
            alloc::format!("{before}"),
            "struct point<align(8); x: i32, y: ptr<i32> align(4)>"
        );
        let mut seen = alloc::vec::Vec::new();
        let after = context.map_subtypes(before, &mut |ty| {
            seen.push(ty);
            if ty == i32 { i64 } else { ty }
        });
        assert_eq!(seen, [i32, context.ptr_type(i32)], "map is called in visit order");
        assert_eq!(after, expected, "{after}");
        assert_eq!(
            alloc::format!("{after}"),
            "struct point<align(8); x: i64, y: ptr<i32> align(4)>"
        );
    }

    #[test]
    fn a_dialect_type_maps_its_fields() {
        use crate::dialects::test::TestVectorType;

        let context = Context::default();
        let (i32, i64) = (context.i32(), context.i64());
        let vector = context.intern_type(TestVectorType::new(i32, 4));
        assert_eq!(subtypes(vector), [i32]);
        let mapped = context.map_subtypes(vector, &mut |_| i64);
        assert_eq!(mapped, context.intern_type(TestVectorType::new(i64, 4)));
        assert_eq!(alloc::format!("{mapped}"), "!test.vector<i64 x 4>");
    }

    #[test]
    fn enum_and_unit_types_hold_no_types() {
        let context = Context::default();
        let (f32, never) = (context.f32(), context.never_type());
        assert!(f32.is::<FloatType>() && never.is::<NeverType>());
        assert_eq!(subtypes(f32), []);
        assert_eq!(subtypes(never), []);
        let marker = context.intern_type(MarkerType);
        assert_eq!(subtypes(marker), []);
        let parsed = context.intern_type(ParsedType::Only);
        assert_eq!(context.map_subtypes(parsed, &mut |_| unreachable!()), parsed);
    }

    #[test]
    fn a_derived_dialect_type_walks_its_fields_in_declaration_order() {
        let context = Context::default();
        let (i8, i16) = (context.i8(), context.i16());
        let pair = context.intern_type(PairType {
            first: i16,
            second: i8,
        });
        assert_eq!(subtypes(pair), [i16, i8]);
        let mut next = [context.i32(), context.i64()].into_iter();
        let mapped = context.map_subtypes(pair, &mut |_| next.next().unwrap());
        assert_eq!(
            mapped.downcast_ref::<PairType>(),
            Some(&PairType {
                first: context.i32(),
                second: context.i64()
            })
        );
    }

    isolated_dialect!(UninhabitedDialect, "uninhabited");

    /// A dialect type with no values; its dialect is never registered.
    #[derive(Debug, PartialEq, Eq, Hash, crate::derive::DialectType)]
    #[ty(dialect = UninhabitedDialect)]
    enum UninhabitedType {}

    #[derive(Debug, PartialEq, Eq, crate::derive::TypeComponent)]
    enum UninhabitedComponent {}

    /// The derived walks of an enum without variants compile; there is no value to call them on.
    #[test]
    fn enums_without_variants_derive_their_walks() {
        let _: fn(&UninhabitedType, &mut dyn FnMut(Type)) = UninhabitedType::visit_subtypes;
        let _: fn(&UninhabitedType, &mut dyn FnMut(Type) -> Type) -> UninhabitedType =
            UninhabitedType::replace_subtypes;
        let _: fn(&UninhabitedComponent, &mut dyn FnMut(Type)) = TypeComponent::visit_types;
        let _: fn(&UninhabitedComponent, &mut dyn FnMut(Type) -> Type) -> UninhabitedComponent =
            TypeComponent::map_types;
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a mapped sub-type is `i64`, a type of another context")]
    fn mapping_to_a_type_of_another_context_panics() {
        let (first, second) = (Context::default(), Context::default());
        let ptr = first.ptr_type(first.i32());
        first.map_subtypes(ptr, &mut |_| second.i64());
    }
}
