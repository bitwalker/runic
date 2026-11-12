use core::fmt;

use crate::{derive::DialectType, dialects::builtin::BuiltinDialect, types::TypePrinter};

/// An IEEE 754 binary floating-point type, `f32` or `f64`.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "float", implements(TypePrinter))]
pub enum FloatType {
    F32,
    F64,
}

impl FloatType {
    /// The width of this type, in bits.
    pub fn bits(&self) -> u32 {
        match self {
            Self::F32 => 32,
            Self::F64 => 64,
        }
    }
}

impl TypePrinter for FloatType {
    fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f{}", self.bits())
    }
}
