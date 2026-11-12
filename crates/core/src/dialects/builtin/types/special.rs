use core::fmt::{self, Write};

use crate::{
    derive::DialectType,
    dialects::builtin::BuiltinDialect,
    types::{TypeFormatter, TypePrinter},
};

/// The type that is not known yet, `?`: for example a result type before inference, or the
/// absence of an expected type.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "unknown")]
pub struct UnknownType;

impl TypePrinter for UnknownType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        printer.write_str("?")
    }
}

/// The type of no value, `never`: for example the result of a function that never returns.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "never")]
pub struct NeverType;

impl TypePrinter for NeverType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        printer.write_str("never")
    }
}
