//! The data layout: how values of the builtin types are laid out in byte-addressed memory.
//!
//! A [DataLayoutSpec] states the endianness, the pointer of each address space, the alignments of
//! integers and floats, and, where function values are bytes in memory, their size and alignment
//! (a [FunctionSpec], `fn = 32` in the text). Without that entry a function type has no layout:
//! a target whose function references are not bytes in memory (Wasm GC, say) leaves it out. An operation implementing [DataLayoutOpInterface] states one in its
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
//! A type that has a representation in memory implements [TypeLayout];
//! [DataLayout::layout_of] gives its [Layout] under a specification.
//!
//! Widths are in bits; alignments are in bytes.

mod builtin;
mod data_layout;
mod spec;
#[cfg(test)]
mod tests;
mod type_layout;

pub(crate) use self::data_layout::LayoutCache;
pub use self::{
    data_layout::{DATA_LAYOUT_ATTR, DataLayout, DataLayoutOpInterface, set_data_layout_spec},
    spec::{Align, DataLayoutSpec, Endianness, FunctionSpec, InvalidDataLayout, PointerSpec},
    type_layout::{AggregateType, FieldsShape, Layout, LayoutError, TypeLayout, element_at},
};
