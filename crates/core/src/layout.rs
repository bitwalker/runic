//! The data layout: how values of the builtin types are laid out in byte-addressed memory.
//!
//! A [DataLayoutSpec] states the endianness, the pointer of each address space and the alignments
//! of integers and floats. An operation implementing [DataLayoutOpInterface] states one in its
//! attribute [DATA_LAYOUT_ATTR]:
//!
//! ```text
//! builtin.module public @m attributes { data_layout = #builtin.data_layout<little, ptr = 32> } {
//! };
//! ```
//!
//! [DataLayout::closest] finds the specification in effect at an operation: the one stated by
//! the operation itself or by its nearest ancestor that states one. No specification in scope
//! means no layout.
//!
//! Widths are in bits; alignments are in bytes.

mod data_layout;
mod spec;
#[cfg(test)]
mod tests;

pub(crate) use self::data_layout::LayoutCache;
pub use self::{
    data_layout::{DATA_LAYOUT_ATTR, DataLayout, DataLayoutOpInterface, set_data_layout_spec},
    spec::{Align, DataLayoutSpec, Endianness, InvalidDataLayout, PointerSpec},
};
