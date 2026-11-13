use core::fmt::{self, Write};

use crate::{
    derive::DialectType,
    dialects::builtin::BuiltinDialect,
    layout::TypeLayout,
    types::{Type, TypeFormatter, TypePrinter},
};

/// A pointer, optionally to a known pointee type, in an address space.
///
/// Prints as `ptr`, `ptr<N>`, `ptr<T>` or `ptr<T, N>`; a pointer without a pointee is opaque, and
/// address space 0 is not printed.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "ptr", implements(TypeLayout))]
pub struct PointerType {
    pointee: Option<Type>,
    addrspace: u32,
}

impl PointerType {
    pub fn new(pointee: Option<Type>, addrspace: u32) -> Self {
        Self { pointee, addrspace }
    }

    /// The type this pointer points to, or `None` if it is opaque.
    pub fn pointee(&self) -> Option<Type> {
        self.pointee
    }

    pub fn addrspace(&self) -> u32 {
        self.addrspace
    }

    /// Returns true if this pointer has no pointee type.
    pub fn is_opaque(&self) -> bool {
        self.pointee.is_none()
    }
}

impl TypePrinter for PointerType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        printer.write_str("ptr")?;
        match (self.pointee, self.addrspace) {
            (None, 0) => Ok(()),
            (None, space) => write!(printer, "<{space}>"),
            (Some(pointee), space) => {
                printer.write_str("<")?;
                printer.ty(pointee)?;
                if space != 0 {
                    write!(printer, ", {space}")?;
                }
                printer.write_str(">")
            }
        }
    }
}
