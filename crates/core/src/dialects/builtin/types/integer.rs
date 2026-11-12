use core::fmt;

use crate::{derive::DialectType, dialects::builtin::BuiltinDialect, types::TypePrinter};

/// A signless integer of a fixed width in bits, `iN`.
///
/// Whether its values are signed is up to the operations that use it.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "int", implements(TypePrinter))]
pub struct IntegerType {
    width: u32,
}

impl IntegerType {
    /// The widest integer type, in bits.
    pub const MAX_WIDTH: u32 = 16_777_215;

    /// The integer type of `width` bits, or `None` unless `1 <= width <= MAX_WIDTH`.
    pub fn new(width: u32) -> Option<Self> {
        (1..=Self::MAX_WIDTH).contains(&width).then_some(Self { width })
    }

    /// The width of this integer type, in bits.
    pub fn width(&self) -> u32 {
        self.width
    }
}

impl TypePrinter for IntegerType {
    fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i{}", self.width)
    }
}

/// A signless, pointer-width integer, `isize`. Its width is chosen by the target.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "isize", implements(TypePrinter))]
pub struct IsizeType;

impl TypePrinter for IsizeType {
    fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("isize")
    }
}
