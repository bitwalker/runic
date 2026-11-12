use alloc::boxed::Box;
use core::{
    fmt::{self, Write},
    num::NonZeroU16,
};

use crate::{
    derive::{DialectType, TypeComponent},
    dialects::builtin::BuiltinDialect,
    interner::Symbol,
    types::{Type, TypeAliasName, TypeFormatter, TypePrinter},
};

/// How a struct's fields are laid out.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, TypeComponent)]
pub enum StructRepr {
    /// Laid out by the target's default rules.
    #[default]
    Default,
    /// Aligned to at least this many bytes, `align(N)`.
    Align(NonZeroU16),
    /// Fields aligned to at most this many bytes, `packed(N)`.
    Packed(NonZeroU16),
    /// Laid out as its single field, `transparent`.
    Transparent,
}

/// A field of a [StructType].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeComponent)]
pub struct StructField {
    pub name: Option<Symbol>,
    pub ty: Type,
    pub align: Option<NonZeroU16>,
}

impl StructField {
    pub fn new(ty: Type) -> Self {
        Self {
            name: None,
            ty,
            align: None,
        }
    }

    pub fn named(mut self, name: Symbol) -> Self {
        self.name = Some(name);
        self
    }

    pub fn aligned(mut self, align: NonZeroU16) -> Self {
        self.align = Some(align);
        self
    }
}

impl From<Type> for StructField {
    fn from(ty: Type) -> Self {
        Self::new(ty)
    }
}

/// An aggregate of fields, `struct name<repr; field, ...>`.
///
/// The struct's name and its field names are part of its identity.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "struct")]
pub struct StructType {
    name: Option<Symbol>,
    repr: StructRepr,
    fields: Box<[StructField]>,
}

impl StructType {
    /// An anonymous struct with the default representation.
    pub fn new(fields: impl IntoIterator<Item = StructField>) -> Self {
        Self {
            name: None,
            repr: StructRepr::Default,
            fields: fields.into_iter().collect(),
        }
    }

    /// A named struct with the default representation.
    pub fn named(name: Symbol, fields: impl IntoIterator<Item = StructField>) -> Self {
        Self {
            name: Some(name),
            ..Self::new(fields)
        }
    }

    pub fn with_repr(mut self, repr: StructRepr) -> Self {
        self.repr = repr;
        self
    }

    pub fn name(&self) -> Option<Symbol> {
        self.name
    }

    pub fn repr(&self) -> StructRepr {
        self.repr
    }

    pub fn fields(&self) -> &[StructField] {
        &self.fields
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// A recursive struct is referred to by its name.
impl TypeAliasName for StructType {
    fn alias_name(&self) -> Option<Symbol> {
        self.name
    }
}

/// Writes `name` bare if it is a bare identifier, otherwise as a quoted, escaped string.
fn print_name(printer: &mut TypeFormatter<'_, '_>, name: Symbol) -> fmt::Result {
    let text = name.as_str();
    if crate::types::is_bare_identifier(text) {
        printer.write_str(text)
    } else {
        write!(printer, "\"{}\"", text.escape_default())
    }
}

impl TypePrinter for StructType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        printer.write_str("struct")?;
        if let Some(name) = self.name {
            printer.write_str(" ")?;
            print_name(printer, name)?;
        }
        printer.write_str("<")?;
        match self.repr {
            StructRepr::Default => (),
            StructRepr::Align(n) => write!(printer, "align({n});")?,
            StructRepr::Packed(n) => write!(printer, "packed({n});")?,
            StructRepr::Transparent => printer.write_str("transparent;")?,
        }
        for (i, field) in self.fields.iter().enumerate() {
            if i > 0 {
                printer.write_str(", ")?;
            } else if self.repr != StructRepr::Default {
                printer.write_str(" ")?;
            }
            if let Some(name) = field.name {
                print_name(printer, name)?;
                printer.write_str(": ")?;
            }
            printer.ty(field.ty)?;
            if let Some(align) = field.align {
                write!(printer, " align({align})")?;
            }
        }
        printer.write_str(">")
    }
}
