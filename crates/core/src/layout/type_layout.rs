use alloc::boxed::Box;

use super::{Align, DataLayout};
use crate::{Report, types::Type};

/// Where the parts of a value lie within it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldsShape {
    /// No parts: an integer, a float or a pointer.
    Scalar,
    /// `count` elements, element `i` at byte `i * stride`.
    Array { stride: u64, count: u64 },
    /// Element `i` at byte `offsets[i]`.
    Offsets(Box<[u64]>),
}

impl FieldsShape {
    /// The byte offset of element `index`, or `None` if there is no such element.
    pub fn offset(&self, index: u64) -> Option<u64> {
        match self {
            Self::Scalar => None,
            // `stride * count` is the size of the value, so `stride * index` fits a `u64`
            Self::Array { stride, count } => (index < *count).then(|| stride * index),
            Self::Offsets(offsets) => {
                usize::try_from(index).ok().and_then(|index| offsets.get(index).copied())
            }
        }
    }
}

/// How a value of a type is laid out in byte-addressed memory: its size and alignment in bytes,
/// and where its parts lie.
///
/// The size is a multiple of the alignment, so it is also the distance between neighbouring
/// elements of an array of the type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    size: u64,
    align: Align,
    fields: FieldsShape,
}

impl Layout {
    /// The layout of a value of `size` bytes aligned to `align` with no parts. `size` must be a
    /// multiple of `align`.
    pub fn scalar(size: u64, align: Align) -> Layout {
        Self::new(size, align, FieldsShape::Scalar)
    }

    /// The layout of a value of `size` bytes aligned to `align` whose parts lie as `fields`
    /// says. `size` must be a multiple of `align`.
    pub fn new(size: u64, align: Align, fields: FieldsShape) -> Layout {
        assert!(
            size.is_multiple_of(align.bytes()),
            "a size of {size} bytes is not a multiple of the alignment, {align}"
        );
        Self {
            size,
            align,
            fields,
        }
    }

    /// The size in bytes, a multiple of [Self::align].
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The alignment in bytes.
    pub fn align(&self) -> Align {
        self.align
    }

    pub fn fields(&self) -> &FieldsShape {
        &self.fields
    }
}

/// Why a type has no layout under a [DataLayout].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LayoutError {
    /// The type, or a type it holds by value, does not implement [TypeLayout]: a function type,
    /// `?` or `never`, for instance. It names the innermost type without a layout.
    #[error("type '{0}' has no representation in memory")]
    NoLayout(Type),
    /// The type holds itself by value, or holds by value a type that does.
    #[error("type '{0}' has infinite size: it holds itself by value, or a type that does")]
    InfiniteSize(Type),
    /// The size of the type does not fit a `u64`.
    #[error("type '{0}' is too large: its size in bytes does not fit in 64 bits")]
    TooLarge(Type),
    /// The type has no element `index`.
    #[error("type '{ty}' has no element {index}")]
    NoSuchElement { ty: Type, index: u32 },
    /// A discriminant of the enum type does not fit the `isize` of the specification: it is
    /// neither an unsigned nor a signed number of that width.
    #[error("type '{0}' has a discriminant that does not fit the 'isize' of the data layout")]
    DiscriminantTooWide(Type),
    /// Two discriminants of the enum type are one value at the width of the `isize` of the
    /// specification, such as `-1` and `4294967295` under a 32-bit `isize`.
    #[error(
        "type '{0}' has two discriminants that are one value in the 'isize' of the data layout"
    )]
    DiscriminantsCollide(Type),
}

impl From<LayoutError> for Report {
    fn from(error: LayoutError) -> Self {
        Report::msg(error)
    }
}

/// Implemented by a type that has a representation in byte-addressed memory.
///
/// [DataLayout::layout_of] asks it for its layout once per specification and keeps the answer.
pub trait TypeLayout {
    /// The layout of `ty`, the type whose value `self` is, under `layout`. A type holding other
    /// types by value asks `layout` for theirs with [DataLayout::layout_of] and returns their
    /// errors unchanged.
    fn layout(&self, ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError>;
}

/// Implemented by a type whose values have numbered elements. Independent of any layout.
///
/// An array's elements are `0..len`; a struct's are its fields; an enum's element 0 is its
/// discriminant and element `1 + v` is the payload of variant `v`. A type that also implements
/// [TypeLayout] lays out each element at the offset its [Layout::fields] gives for that index.
pub trait AggregateType {
    /// The type of element `index`, or `None` if there is no such element.
    fn element_type(&self, index: u32) -> Option<Type>;
}
