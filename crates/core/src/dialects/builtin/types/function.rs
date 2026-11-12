use alloc::boxed::Box;
use core::fmt;

use crate::{
    derive::DialectType,
    dialects::builtin::BuiltinDialect,
    types::{Type, TypePrinter},
};

/// The type of a function: parameter types and result types, `(T, U) -> V`.
///
/// A single result prints without parentheses; no result or several results print parenthesised.
/// It carries no calling convention.
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

fn print_list(f: &mut fmt::Formatter<'_>, types: &[Type]) -> fmt::Result {
    for (i, ty) in types.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{ty}")?;
    }
    Ok(())
}

impl TypePrinter for FunctionType {
    fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        print_list(f, &self.params)?;
        f.write_str(") -> ")?;
        if let [result] = &*self.results {
            write!(f, "{result}")
        } else {
            f.write_str("(")?;
            print_list(f, &self.results)?;
            f.write_str(")")
        }
    }
}
