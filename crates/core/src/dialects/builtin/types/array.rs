use core::fmt;

use crate::{
    derive::DialectType,
    dialects::builtin::BuiltinDialect,
    types::{Type, TypePrinter},
};

/// A fixed-length sequence of elements of one type, `array<T; N>`.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "array")]
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
    fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "array<{}; {}>", self.element, self.len)
    }
}
