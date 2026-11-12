use alloc::boxed::Box;
use core::{any::TypeId, fmt};

use super::{Type, TypeImpl, TypeName};
use crate::{Context, DialectRegistration, interner::Symbol, parse, traits::TraitInfo};

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

/// Prints a type's own syntax.
///
/// A type that implements this trait is printed with it; it need not be declared in the
/// `implements(..)` of `#[derive(DialectType)]`.
///
/// A builtin type prints its whole text; any other type prints the text between the `<` and `>`
/// that follow its `!dialect.name`.
pub trait TypePrinter {
    fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// Parses what a type's [TypePrinter] prints, and interns the type it describes.
pub trait TypeParser: TypeRegistration {
    fn parse(parser: &mut dyn parse::Parser<'_>) -> parse::ParseResult<Type>;
}

/// Prints the syntax of one registered type: what its [TypePrinter] prints.
pub type PrintTypeFn = fn(Type, &mut fmt::Formatter<'_>) -> fmt::Result;

/// The printer of `T`, if `T` implements [TypePrinter].
pub(super) trait MaybeTypePrinter {
    fn print_fn() -> Option<PrintTypeFn>;
}

impl<T> MaybeTypePrinter for T {
    #[inline(always)]
    default fn print_fn() -> Option<PrintTypeFn> {
        None
    }
}

impl<T: TypeRegistration + TypePrinter> MaybeTypePrinter for T {
    #[inline(always)]
    fn print_fn() -> Option<PrintTypeFn> {
        Some(print_value::<T>)
    }
}

fn print_value<T: TypeRegistration + TypePrinter>(
    ty: Type,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    ty.downcast_ref::<T>().expect("a type is printed by its own name").print(f)
}

/// Parses the syntax of one registered type.
pub type ParseTypeFn = fn(&mut dyn parse::Parser<'_>) -> parse::ParseResult<Type>;

/// Interns the default value of one registered type.
pub type DefaultTypeFn = fn(&Context) -> Type;

/// The parser of `T`, if `T` implements [TypeParser].
pub(super) trait MaybeTypeParser {
    fn parse_fn() -> Option<ParseTypeFn>;
}

impl<T> MaybeTypeParser for T {
    #[inline(always)]
    default fn parse_fn() -> Option<ParseTypeFn> {
        None
    }
}

impl<T: TypeParser> MaybeTypeParser for T {
    #[inline(always)]
    fn parse_fn() -> Option<ParseTypeFn> {
        Some(<T as TypeParser>::parse)
    }
}

/// A function interning `T::default()`, if `T` implements [Default].
pub(super) trait MaybeDefaultType {
    fn default_fn() -> Option<DefaultTypeFn>;
}

impl<T> MaybeDefaultType for T {
    #[inline(always)]
    default fn default_fn() -> Option<DefaultTypeFn> {
        None
    }
}

impl<T: TypeRegistration + Default> MaybeDefaultType for T {
    #[inline(always)]
    fn default_fn() -> Option<DefaultTypeFn> {
        Some(|context| context.intern_type(T::default()))
    }
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
