//! The data layout: how values of the builtin types are laid out in byte-addressed memory.
//!
//! A [DataLayoutSpec] states the endianness, the pointer of each address space and the alignments
//! of integers and floats; a
//! [DataLayoutAttr](crate::dialects::builtin::attributes::DataLayoutAttr) holds one.
//!
//! Widths are in bits; alignments are in bytes.

mod spec;
#[cfg(test)]
mod tests;

pub use self::spec::{Align, DataLayoutSpec, Endianness, InvalidDataLayout, PointerSpec};
