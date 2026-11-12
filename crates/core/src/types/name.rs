use alloc::{boxed::Box, rc::Rc, vec::Vec};
use core::{
    any::TypeId,
    fmt,
    ptr::{DynMetadata, Pointee},
};

use super::{
    Type, TypeRegistration, TypeStorage,
    registration::{
        DefaultTypeFn, MaybeDefaultType, MaybeTypeParser, MaybeTypePrinter, ParseTypeFn,
        PrintTypeFn,
    },
};
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
    /// The name of that Rust type, for diagnostics.
    rust_type_name: &'static str,
    /// The interfaces this type implements, sorted by trait `TypeId` for binary search.
    traits: Box<[TraitInfo]>,
    /// `TypeStorage::<T>::VALUE_OFFSET` for the registered `T`.
    value_offset: usize,
    /// Formats a type whose name this is.
    debug: fn(Type, &mut fmt::Formatter<'_>) -> fmt::Result,
    /// Prints the syntax of this type, if it implements `TypePrinter`.
    print: Option<PrintTypeFn>,
    /// Parses the syntax of this type, if it implements `TypeParser`.
    parse: Option<ParseTypeFn>,
    /// Interns the default value of this type, if it implements `Default`.
    default: Option<DefaultTypeFn>,
}

fn debug_value<T: TypeRegistration>(ty: Type, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let name = ty.name();
    let value = ty.downcast_ref::<T>().expect("a type is formatted by its own name");
    write!(f, "{}.{}({value:?})", name.dialect(), name.name())
}

/// Builtin types print bare, with no `!dialect.name` to fall back on, so one without a
/// [TypePrinter](super::TypePrinter) cannot be displayed.
///
/// Panics if `dialect` is the builtin dialect and `has_printer` is false.
fn require_printer_for_builtin(dialect: Symbol, rust_type_name: &str, has_printer: bool) {
    assert!(
        has_printer
            || dialect.as_str()
                != <crate::dialects::builtin::BuiltinDialect as crate::DialectRegistration>::NAMESPACE,
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
    if dialect.as_str()
        == <crate::dialects::builtin::BuiltinDialect as crate::DialectRegistration>::NAMESPACE
    {
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
        for ti in traits.iter() {
            assert!(
                *ti.concrete_type_id() == TypeId::of::<T>(),
                "the `{}` entry in the trait table of type `{}` was built for another type, `{}`",
                ti.type_name(),
                rust_type_name,
                ti.concrete_type_name()
            );
        }
        traits.sort_by_key(|ti| *ti.type_id());
        Self(Rc::new(TypeInfo {
            dialect,
            name: T::name(),
            type_id: TypeId::of::<T>(),
            rust_type_name,
            traits: traits.into_boxed_slice(),
            value_offset: TypeStorage::<T>::VALUE_OFFSET,
            debug: debug_value::<T>,
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
