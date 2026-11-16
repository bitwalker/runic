use alloc::boxed::Box;
use core::fmt::{self, Write};

use crate::{
    derive::DialectType,
    dialects::builtin::BuiltinDialect,
    types::{Type, TypeFormatter, TypePrinter},
};

/// The type of a function: parameter types and result types, `(T, U) -> V`.
///
/// A single result prints without parentheses unless it is itself a function type; no result or
/// several results print parenthesised. It carries no calling convention.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "function")]
pub struct FunctionType {
    params: Box<[Type]>,
    results: Box<[Type]>,
}

impl FunctionType {
    pub fn new(
        params: impl IntoIterator<Item = Type>,
        results: impl IntoIterator<Item = Type>,
    ) -> Self {
        Self {
            params: params.into_iter().collect(),
            results: results.into_iter().collect(),
        }
    }

    pub fn params(&self) -> &[Type] {
        &self.params
    }

    pub fn results(&self) -> &[Type] {
        &self.results
    }
}

impl TypePrinter for FunctionType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        printer.write_str("(")?;
        printer.types(self.params.iter().copied())?;
        printer.write_str(") -> ")?;
        match &*self.results {
            [result] if !result.is_function() => printer.ty(*result),
            results => {
                printer.write_str("(")?;
                printer.types(results.iter().copied())?;
                printer.write_str(")")
            }
        }
    }
}
