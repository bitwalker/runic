use core::fmt::{self, Write};

use crate::{
    derive::DialectType,
    dialects::builtin::BuiltinDialect,
    types::{TypeFormatter, TypePrinter},
};

/// A recursion variable, `rec.N`: member `N` of the recursion group being defined.
///
/// A variable, and every type that holds one, is open: it may be used to write the bodies passed
/// to `Context::rec_group`, but not in IR. `Context::rec_var` interns one.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "rec_var")]
pub struct RecVarType {
    index: u32,
}

impl RecVarType {
    /// The variable denoting member `index`.
    pub fn new(index: u32) -> Self {
        Self { index }
    }

    /// The index of the group member this variable denotes.
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl TypePrinter for RecVarType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        write!(printer, "rec.{}", self.index)
    }
}
