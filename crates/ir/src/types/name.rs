use alloc::{boxed::Box, rc::Rc, vec::Vec};
use core::{
    any::TypeId,
    fmt,
    ptr::{DynMetadata, NonNull, Pointee},
};

use blink_alloc::Blink;

use super::{
    RecGroupData, Type, TypeRegistration, TypeStorage,
    registration::{
        AliasNameFn, DefaultTypeFn, MaybeDefaultType, MaybeTypeAliasName, MaybeTypeParser,
        MaybeTypePrinter, ParseTypeFn, PrintTypeFn,
    },
};
use crate::{Context, interner::Symbol, traits::TraitInfo};

/// The registration record of a type: its dialect, its name within the dialect, and the Rust type
/// and interfaces that implement it.
///
/// Two names are equal when their dialect, name and Rust type are.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeName(Rc<TypeInfo>);

struct TypeInfo {
    dialect: Symbol,
    name: Symbol,
    /// Whether `dialect` is the builtin dialect, whose types print bare.
    is_builtin: bool,
    /// The [TypeId] of the Rust type whose values are this type's storage.
    type_id: TypeId,
    /// The name of that Rust type, for diagnostics.
    rust_type_name: &'static str,
    /// The interfaces this type implements, sorted by trait `TypeId` for binary search.
    traits: Box<[TraitInfo]>,
    /// `TypeStorage::<T>::VALUE_OFFSET` for the registered `T`.
    value_offset: usize,
    /// Calls the visitor with each type held directly by a type whose name this is.
    visit: VisitSubtypesFn,
    /// Interns a type whose name this is, with each type it holds directly replaced.
    rebuild: RebuildTypeFn,
    /// Allocates a recursion-group member whose body has this name.
    alloc_member: AllocMemberFn,
    /// Fills in the value of a member allocated by `alloc_member`.
    fill_member: FillMemberFn,
    /// Suggests the name of a member whose body has this name, if it implements
    /// `TypeAliasName`.
    alias_name: Option<AliasNameFn>,
    /// Prints the syntax of this type, if it implements `TypePrinter`.
    print: Option<PrintTypeFn>,
    /// Parses the syntax of this type, if it implements `TypeParser`.
    parse: Option<ParseTypeFn>,
    /// Interns the default value of this type, if it implements `Default`.
    default: Option<DefaultTypeFn>,
}

/// Calls the visitor with each type held directly by a type of one registered Rust type.
pub(crate) type VisitSubtypesFn = fn(Type, &mut dyn FnMut(Type));

/// Interns a type of one registered Rust type with each type it holds directly replaced.
pub(crate) type RebuildTypeFn = fn(&Context, Type, &mut dyn FnMut(Type) -> Type) -> Type;

fn visit_subtypes<T: TypeRegistration>(ty: Type, visit: &mut dyn FnMut(Type)) {
    ty.downcast_ref::<T>()
        .expect("a type is visited by its own name")
        .visit_subtypes(visit);
}

fn rebuild_type<T: TypeRegistration>(
    context: &Context,
    ty: Type,
    replace: &mut dyn FnMut(Type) -> Type,
) -> Type {
    let value = ty
        .downcast_ref::<T>()
        .expect("a type is rebuilt by its own name")
        .replace_subtypes(replace);
    context.intern_type(value)
}

/// Allocates the storage of member `index` of `group`, owned by the context whose id is `owner`,
/// whose body is `body`. The member's value is a copy of the open body until it is filled in.
pub(crate) type AllocMemberFn = fn(&Blink, Type, u64, u32, NonNull<RecGroupData>, u32) -> Type;

/// Replaces the value of a member by its body's value with each type it holds directly replaced.
///
/// # Safety
///
/// See [fill_member].
pub(crate) type FillMemberFn = unsafe fn(Type, Type, &mut dyn FnMut(Type) -> Type);

/// Allocates the storage of a member whose body is `body`, holding a copy of the open body.
///
/// The member is not published: [Type::from_storage]'s contract lets `Context::rec_group` fill it
/// in with [fill_member] before returning it.
fn alloc_member<T: TypeRegistration>(
    allocator: &Blink,
    body: Type,
    hash: u64,
    owner: u32,
    group: NonNull<RecGroupData>,
    index: u32,
) -> Type {
    let value = body.downcast_ref::<T>().expect("a member is allocated by its body's name");
    let placeholder = value.replace_subtypes(&mut |ty| ty);
    let storage = allocator.put(TypeStorage::new_member(
        body.name().clone(),
        hash,
        owner,
        group,
        index,
        placeholder,
    ));
    // SAFETY: `storage` is an initialized `TypeStorage<T>` in the arena, which keeps it at this
    // address until the context's last `Rc<Blink>` is dropped and is never reset. The `&mut`
    // from `put` is converted to a pointer here and never used again, so the pointer carries
    // write provenance over the whole storage and no reference to it remains. The value is
    // written again only by `fill_member`, before the member is published.
    unsafe { Type::from_storage(NonNull::from(storage)) }
}

/// Replaces the value of `member` by `body`'s value with its held types replaced by `replace`.
///
/// # Safety
///
/// `member` was returned by `alloc_member::<T>` for a body of the same Rust type, and its handle
/// has not been published (see [Type::from_storage]). No reference to the value of any
/// unpublished member exists when this is called, and `replace` must not read the value of any
/// unpublished member (it may read their headers, and store their handles in other types).
unsafe fn fill_member<T: TypeRegistration>(
    member: Type,
    body: Type,
    replace: &mut dyn FnMut(Type) -> Type,
) {
    // Computed completely before the storage is touched: `replace` interns types, and while it
    // runs nothing may be writing the member.
    let closed = body
        .downcast_ref::<T>()
        .expect("a member is filled from a body of its own type")
        .replace_subtypes(replace);
    assert!(member.is::<T>(), "a member is filled from a body of its own type");
    let storage = member.as_storage_ptr::<T>();
    // SAFETY: `member` is a `TypeStorage<T>` (checked above) created by `alloc_member`, whose
    // pointer carries write provenance over the whole storage. It is unpublished and, by the
    // contract, no reference to its value exists, so this write aliases nothing; references to
    // its header (`Type::header`) cover only the header, which is not written. The value was
    // initialized by `alloc_member`, so replacing it is sound.
    let placeholder = unsafe { core::ptr::addr_of_mut!((*storage).value).replace(closed) };
    // Dropped once the member holds its closed value: a value's `Drop` may run arbitrary code.
    drop(placeholder);
}

/// Returns true if `dialect` is the builtin dialect.
fn is_builtin_dialect(dialect: Symbol) -> bool {
    dialect.as_str()
        == <crate::dialects::builtin::BuiltinDialect as crate::DialectRegistration>::NAMESPACE
}

/// Builtin types print bare, with no `!dialect.name` to fall back on, so one without a
/// [TypePrinter](super::TypePrinter) cannot be displayed.
///
/// Panics if `dialect` is the builtin dialect and `has_printer` is false.
fn require_printer_for_builtin(dialect: Symbol, rust_type_name: &str, has_printer: bool) {
    assert!(
        has_printer || !is_builtin_dialect(dialect),
        "builtin type `{rust_type_name}` must implement `TypePrinter`: builtin types print bare"
    );
}

/// A dialect type prints as `!dialect.name`, followed by `<..>` holding what its
/// [TypePrinter](super::TypePrinter) prints, if it has one. For that text to parse back, a type
/// with a printer must have a [TypeParser](super::TypeParser) for its body, and one without a
/// printer must be [Default], which is what `!dialect.name` alone parses as. Builtin types print
/// and parse their own syntax, and are not subject to this.
///
/// Panics, naming `rust_type_name`, if the type `name` in `dialect` breaks either rule.
fn require_round_trip(
    dialect: Symbol,
    name: Symbol,
    rust_type_name: &str,
    has_printer: bool,
    has_parser: bool,
    has_default: bool,
) {
    if is_builtin_dialect(dialect) {
        return;
    }
    assert!(
        has_parser || !has_printer,
        "type `{rust_type_name}` implements `TypePrinter` but not `TypeParser`: the body it \
         prints after `!{dialect}.{name}` must parse back"
    );
    assert!(
        has_printer || !has_parser,
        "type `{rust_type_name}` implements `TypeParser` but not `TypePrinter`: it prints as \
         `!{dialect}.{name}` without the body its parser reads"
    );
    assert!(
        has_printer || has_default,
        "type `{rust_type_name}` has no `TypePrinter`, so it must implement `Default`: it prints \
         as `!{dialect}.{name}` alone, which parses as its default value"
    );
}

impl TypeName {
    pub(crate) fn new<T: TypeRegistration>(dialect: Symbol) -> Self {
        let rust_type_name = core::any::type_name::<T>();
        let print = <T as MaybeTypePrinter>::print_fn();
        let parse = <T as MaybeTypeParser>::parse_fn();
        let default = <T as MaybeDefaultType>::default_fn();
        require_printer_for_builtin(dialect, rust_type_name, print.is_some());
        require_round_trip(
            dialect,
            T::name(),
            rust_type_name,
            print.is_some(),
            parse.is_some(),
            default.is_some(),
        );
        let mut traits = Vec::from(T::traits());
        // `Type::as_trait` pairs a `T` with the vtable in the table, so it must be `T`'s.
        crate::traits::assert_trait_table_built_for::<T>(&traits, "type");
        traits.sort_by_key(|ti| *ti.type_id());
        Self(Rc::new(TypeInfo {
            dialect,
            name: T::name(),
            is_builtin: is_builtin_dialect(dialect),
            type_id: TypeId::of::<T>(),
            rust_type_name,
            traits: traits.into_boxed_slice(),
            value_offset: TypeStorage::<T>::VALUE_OFFSET,
            visit: visit_subtypes::<T>,
            rebuild: rebuild_type::<T>,
            alloc_member: alloc_member::<T>,
            fill_member: fill_member::<T>,
            alias_name: <T as MaybeTypeAliasName>::alias_name_fn(),
            print,
            parse,
            default,
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

    /// Returns true if this type belongs to the builtin dialect, and so prints bare.
    #[inline]
    pub(crate) fn is_builtin(&self) -> bool {
        self.0.is_builtin
    }

    /// Returns true if `T` is the Rust type that implements this type.
    #[inline]
    pub fn is<T: 'static>(&self) -> bool {
        TypeId::of::<T>() == self.0.type_id
    }

    /// Returns true if `other` is implemented by the same Rust type as this name.
    #[inline]
    pub(crate) fn has_rust_type_of(&self, other: &TypeName) -> bool {
        self.0.type_id == other.0.type_id
    }

    /// The name of the Rust type that implements this type.
    #[inline]
    pub(crate) fn rust_type_name(&self) -> &'static str {
        self.0.rust_type_name
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

    /// The printer of this type's syntax, if it implements
    /// [TypePrinter](crate::types::TypePrinter).
    #[inline]
    pub fn print_fn(&self) -> Option<PrintTypeFn> {
        self.0.print
    }

    /// The parser of this type's syntax, if it implements
    /// [TypeParser](crate::types::TypeParser).
    #[inline]
    pub fn parse_fn(&self) -> Option<ParseTypeFn> {
        self.0.parse
    }

    /// A function interning this type's default value, if it implements [Default].
    #[inline]
    pub fn default_fn(&self) -> Option<DefaultTypeFn> {
        self.0.default
    }

    /// Calls the visitor with each type held directly by a type with this name.
    #[inline]
    pub(crate) fn visit_fn(&self) -> VisitSubtypesFn {
        self.0.visit
    }

    /// Interns a type with this name with each type it holds directly replaced.
    #[inline]
    pub(crate) fn rebuild_fn(&self) -> RebuildTypeFn {
        self.0.rebuild
    }

    /// Allocates a recursion-group member whose body has this name.
    #[inline]
    pub(crate) fn alloc_member_fn(&self) -> AllocMemberFn {
        self.0.alloc_member
    }

    /// Fills in the value of a member allocated with [Self::alloc_member_fn].
    #[inline]
    pub(crate) fn fill_member_fn(&self) -> FillMemberFn {
        self.0.fill_member
    }

    /// The name suggested for a recursion-group member whose body has this name, if this type
    /// implements [TypeAliasName](crate::types::TypeAliasName).
    #[inline]
    pub fn alias_name_fn(&self) -> Option<AliasNameFn> {
        self.0.alias_name
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "builtin type `Broken` must implement `TypePrinter`")]
    fn a_builtin_type_without_a_printer_is_rejected() {
        require_printer_for_builtin(Symbol::intern("builtin"), "Broken", false);
    }

    #[test]
    fn a_printer_is_optional_outside_the_builtin_dialect() {
        require_printer_for_builtin(Symbol::intern("builtin"), "Fine", true);
        require_printer_for_builtin(Symbol::intern("other"), "Fine", false);
    }

    #[test]
    fn a_dialect_type_round_trips_with_a_printer_and_parser_or_a_default() {
        let (other, name) = (Symbol::intern("other"), Symbol::intern("t"));
        require_round_trip(other, name, "Printed", true, true, false);
        require_round_trip(other, name, "PrintedDefault", true, true, true);
        require_round_trip(other, name, "Unit", false, false, true);
        // Builtin types parse their own syntax.
        require_round_trip(Symbol::intern("builtin"), name, "Builtin", true, false, false);
    }
}
