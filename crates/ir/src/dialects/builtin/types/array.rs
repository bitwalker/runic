use core::fmt::{self, Write};

use crate::{
    derive::DialectType,
    dialects::builtin::BuiltinDialect,
    layout::{AggregateType, TypeLayout},
    types::{Type, TypeFormatter, TypePrinter},
};

/// A fixed-length sequence of elements of one type, `array<T; N>`.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "array", implements(TypeLayout, AggregateType))]
pub struct ArrayType {
    element: Type,
    len: u64,
}

impl ArrayType {
    pub fn new(element: Type, len: u64) -> Self {
        Self { element, len }
    }

    pub fn element(&self) -> Type {
        self.element
    }

    /// The number of elements.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl TypePrinter for ArrayType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        printer.write_str("array<")?;
        printer.ty(self.element)?;
        write!(printer, "; {}>", self.len)
    }
}
