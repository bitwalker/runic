use alloc::boxed::Box;
use core::{
    fmt::{self, Write},
    num::NonZeroU16,
};

use super::print_name;
use crate::{
    derive::{DialectType, TypeComponent},
    dialects::builtin::BuiltinDialect,
    interner::Symbol,
    layout::{AggregateType, TypeLayout},
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

/// Why a [StructType] cannot be built into a type, see [StructType::validate].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InvalidStructType {
    #[error("an alignment is a power of two, and {0} is not")]
    AlignmentNotPowerOfTwo(u16),
    #[error("a `transparent` struct has exactly one field, and this one has {0}")]
    TransparentFieldCount(usize),
    #[error("the field of a `transparent` struct has no alignment of its own")]
    TransparentFieldAligned,
}

/// An aggregate of fields, `struct name<repr; field, ...>`.
///
/// The struct's name and its field names are part of its identity.
///
/// The alignments a struct states (`align(N)`, `packed(N)`, a field's `align(N)`) are powers of
/// two, and a `transparent` struct has one field, without an alignment of its own. A value that
/// breaks either rule is not a type: [Context::try_struct_type](crate::Context::try_struct_type)
/// returns the reason, and `Context::struct_type` panics with it. `replace_subtypes` replaces only
/// the types of the fields, never an alignment or the field count, so it does not run the check.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "struct", implements(TypeLayout, AggregateType))]
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

    /// Checks that `align` is a power of two.
    pub fn check_alignment(align: NonZeroU16) -> Result<(), InvalidStructType> {
        if align.is_power_of_two() {
            Ok(())
        } else {
            Err(InvalidStructType::AlignmentNotPowerOfTwo(align.get()))
        }
    }

    /// Checks that every alignment this struct states is a power of two, and that a `transparent`
    /// struct has one field without an alignment.
    pub fn validate(&self) -> Result<(), InvalidStructType> {
        if let StructRepr::Align(align) | StructRepr::Packed(align) = self.repr {
            Self::check_alignment(align)?;
        }
        for align in self.fields.iter().filter_map(|field| field.align) {
            Self::check_alignment(align)?;
        }
        if self.repr == StructRepr::Transparent {
            match &*self.fields {
                [field] if field.align.is_some() => {
                    return Err(InvalidStructType::TransparentFieldAligned);
                }
                [_] => (),
                fields => return Err(InvalidStructType::TransparentFieldCount(fields.len())),
            }
        }
        Ok(())
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
