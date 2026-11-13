//! The layouts of the builtin types that have one.

use super::{DataLayout, FieldsShape, Layout, LayoutError, TypeLayout};
use crate::{
    dialects::builtin::types::{ArrayType, FloatType, IntegerType, IsizeType, PointerType},
    types::Type,
};

/// The layout of an integer of `width` bits: aligned as the specification says, its bytes
/// rounded up to that alignment.
fn integer(width: u32, layout: &DataLayout) -> Layout {
    let align = layout.spec().integer_align(width);
    let size = align
        .align_up(u64::from(width.div_ceil(8)))
        .expect("an integer of at most 2^24 bits fits a u64 at any alignment");
    Layout::scalar(size, align)
}

impl TypeLayout for IntegerType {
    fn layout(&self, _ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
        Ok(integer(self.width(), layout))
    }
}

/// `isize` is laid out as the integer of the index width of address space 0.
impl TypeLayout for IsizeType {
    fn layout(&self, _ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
        Ok(integer(layout.isize_width(), layout))
    }
}

/// `f32` and `f64` are 4 and 8 bytes, rounded up to their alignment.
impl TypeLayout for FloatType {
    fn layout(&self, _ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
        let align = layout.spec().float_align(*self);
        let size = align
            .align_up(u64::from(self.bits() / 8))
            .expect("8 bytes fit a u64 at any alignment");
        Ok(Layout::scalar(size, align))
    }
}

/// A pointer is laid out as the pointer of its address space says, its bytes rounded up to its
/// alignment. The pointee is not examined, so a type may hold a pointer to itself.
impl TypeLayout for PointerType {
    fn layout(&self, _ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
        let pointer = layout.pointer(self.addrspace());
        let size = pointer
            .align
            .align_up(u64::from(pointer.size_bits.div_ceil(8)))
            .expect("8 bytes fit a u64 at any alignment");
        Ok(Layout::scalar(size, pointer.align))
    }
}

/// An array is its elements one after another, each at a multiple of the element's size, and
/// aligned as its element. An element without a layout is the array's error.
impl TypeLayout for ArrayType {
    fn layout(&self, ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
        let element = layout.layout_of(self.element())?;
        let stride = element.size();
        let size = stride.checked_mul(self.len()).ok_or(LayoutError::TooLarge(ty))?;
        Ok(Layout::new(
            size,
            element.align(),
            FieldsShape::Array {
                stride,
                count: self.len(),
            },
        ))
    }
}
