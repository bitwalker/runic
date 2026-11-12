use core::fmt;

use crate::{derive::DialectType, dialects::builtin::BuiltinDialect, types::TypePrinter};

/// The type that is not known yet, `?`: for example a result type before inference, or the
/// absence of an expected type.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "unknown", implements(TypePrinter))]
pub struct UnknownType;

impl TypePrinter for UnknownType {
    fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("?")
    }
}

/// The type of no value, `never`: for example the result of a function that never returns.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "never", implements(TypePrinter))]
pub struct NeverType;

impl TypePrinter for NeverType {
    fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("never")
    }
}
