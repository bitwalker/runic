//! The layouts of the builtin types that have one, and the elements of the builtin aggregates.

use alloc::vec::Vec;

use super::{AggregateType, Align, DataLayout, FieldsShape, Layout, LayoutError, TypeLayout};
use crate::{
    dialects::builtin::types::{
        ArrayType, EnumType, FloatType, FunctionType, IntegerType, IsizeType, PointerType,
        StructRepr, StructType,
    },
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

/// A function value is laid out as the specification's [FunctionSpec](super::FunctionSpec) says,
/// its bytes rounded up to its alignment. Under a specification without one, a function type has
/// no layout ([LayoutError::NoLayout]). The parameter and result types are not examined.
impl TypeLayout for FunctionType {
    fn layout(&self, ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
        let function = layout.function().ok_or(LayoutError::NoLayout(ty))?;
        let size = function
            .align
            .align_up(u64::from(function.size_bits.div_ceil(8)))
            .expect("8 bytes fit a u64 at any alignment");
        Ok(Layout::scalar(size, function.align))
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

/// An array's elements are `0..len`, all of its element type.
impl AggregateType for ArrayType {
    fn element_count(&self) -> u64 {
        self.len()
    }

    fn element_type(&self, index: u32) -> Option<Type> {
        (u64::from(index) < self.len()).then(|| self.element())
    }
}

/// A struct's elements are its fields.
impl AggregateType for StructType {
    fn element_count(&self) -> u64 {
        self.fields().len() as u64
    }

    fn element_type(&self, index: u32) -> Option<Type> {
        let index = usize::try_from(index).ok()?;
        self.fields().get(index).map(|field| field.ty)
    }
}

/// An enum's element 0 is its discriminant, and element `1 + v` is the payload of variant `v`;
/// a variant without a payload has no element.
impl AggregateType for EnumType {
    fn element_count(&self) -> u64 {
        1 + self.variants().len() as u64
    }

    fn element_type(&self, index: u32) -> Option<Type> {
        match index.checked_sub(1) {
            None => Some(self.discriminant()),
            Some(variant) => self.variants().get(usize::try_from(variant).ok()?)?.payload(),
        }
    }
}

/// An alignment a struct states. Construction checks that it is a power of two, and a `u16`
/// power of two is at most [Align::MAX_BYTES].
fn stated_align(bytes: core::num::NonZeroU16) -> Align {
    Align::from_bytes(u64::from(bytes.get())).expect("a struct states powers of two")
}

/// A struct is laid out by C's rules: each field at the next multiple of its alignment after the
/// field before it, and the size rounded up to the struct's alignment, the largest of its
/// fields'.
///
/// A field's alignment is its type's, capped by `packed(P)` and then raised by the field's own
/// `align(F)`; `align(N)` raises the struct's. A `transparent` struct is laid out as its one
/// field.
///
/// Panics if the struct fails [StructType::validate], which only a struct interned without its
/// check can (through `Context::intern_type`).
impl TypeLayout for StructType {
    fn layout(&self, ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
        if let Err(error) = self.validate() {
            panic!("type '{ty}' is not a valid struct and has no layout: {error}");
        }
        if self.repr() == StructRepr::Transparent {
            let field = layout.layout_of(self.fields()[0].ty)?;
            return Ok(Layout::new(field.size(), field.align(), FieldsShape::Offsets([0].into())));
        }
        let cap = match self.repr() {
            StructRepr::Packed(cap) => Some(stated_align(cap)),
            _ => None,
        };
        let mut offsets = Vec::with_capacity(self.len());
        let mut end = 0u64;
        let mut align = Align::ONE;
        for field in self.fields() {
            let laid_out = layout.layout_of(field.ty)?;
            let natural = laid_out.align();
            let capped = cap.map_or(natural, |cap| natural.min(cap));
            let effective = field.align.map_or(capped, |stated| capped.max(stated_align(stated)));
            let offset = effective.align_up(end).ok_or(LayoutError::TooLarge(ty))?;
            end = offset.checked_add(laid_out.size()).ok_or(LayoutError::TooLarge(ty))?;
            offsets.push(offset);
            align = align.max(effective);
        }
        if let StructRepr::Align(raised) = self.repr() {
            align = align.max(stated_align(raised));
        }
        let size = align.align_up(end).ok_or(LayoutError::TooLarge(ty))?;
        Ok(Layout::new(size, align, FieldsShape::Offsets(offsets.into())))
    }
}

/// An enum is its discriminant followed by a union of the payloads: every payload starts at the
/// size of the discriminant rounded up to the largest payload alignment, and the size is that
/// offset plus the largest payload size, rounded up to the enum's alignment, the larger of the
/// discriminant's and the payloads'. Without payloads that is the discriminant's layout.
///
/// Element 0 lies at offset 0, and the payload element of every variant, with a payload or not,
/// at the payload offset.
///
/// An `isize` discriminant's values are held at 64 bits and stand for their low `W` bits, `W` the
/// width of the `isize` of `layout`. A value fits when it is the zero extension or the sign
/// extension of its low `W` bits, that is, an unsigned or a signed number of `W` bits
/// ([LayoutError::DiscriminantTooWide] otherwise), and the low `W` bits of the values must be
/// distinct ([LayoutError::DiscriminantsCollide] otherwise).
impl TypeLayout for EnumType {
    fn layout(&self, ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
        if self.discriminant().is_isize() {
            check_isize_discriminants(self, ty, layout.isize_width())?;
        }
        let tag = layout.layout_of(self.discriminant())?;
        let mut payload_align = Align::ONE;
        let mut payload_size = 0u64;
        for payload in self.variants().iter().filter_map(|variant| variant.payload()) {
            let payload = layout.layout_of(payload)?;
            payload_align = payload_align.max(payload.align());
            payload_size = payload_size.max(payload.size());
        }
        let offset = payload_align.align_up(tag.size()).ok_or(LayoutError::TooLarge(ty))?;
        let align = tag.align().max(payload_align);
        let size = offset
            .checked_add(payload_size)
            .and_then(|end| align.align_up(end))
            .ok_or(LayoutError::TooLarge(ty))?;
        let offsets = core::iter::once(0).chain(self.variants().iter().map(|_| offset)).collect();
        Ok(Layout::new(size, align, FieldsShape::Offsets(offsets)))
    }
}

/// Checks the values of `enum_ty`, the value of `ty`, against an `isize` of `width` bits, 1 to 64.
///
/// A value is held at 64 bits; it fits when it equals its low `width` bits zero-extended or
/// sign-extended to 64 bits. The low `width` bits of the values must then be distinct. At 64 bits
/// the low bits are the values themselves, which [EnumType::new] keeps distinct.
fn check_isize_discriminants(enum_ty: &EnumType, ty: Type, width: u32) -> Result<(), LayoutError> {
    debug_assert!((1..=64).contains(&width), "an isize of {width} bits");
    let low_mask = u128::from(u64::MAX >> (64 - width));
    let high_mask = u128::from(u64::MAX) & !low_mask;
    let mut patterns = Vec::with_capacity(enum_ty.variants().len());
    for variant in enum_ty.variants() {
        let bits = variant.value();
        let low = bits & low_mask;
        let sign_extended = if low >> (width - 1) == 1 {
            low | high_mask
        } else {
            low
        };
        if bits != low && bits != sign_extended {
            return Err(LayoutError::DiscriminantTooWide(ty));
        }
        patterns.push(low);
    }
    if width < 64 {
        patterns.sort_unstable();
        if patterns.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(LayoutError::DiscriminantsCollide(ty));
        }
    }
    Ok(())
}
