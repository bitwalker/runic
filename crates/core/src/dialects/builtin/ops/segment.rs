use alloc::sync::Arc;
use core::fmt;

use crate::{
    OpPrinter, UnsafeIntrusiveEntityRef,
    constants::ConstantData,
    derive::{OpParser, OpPrinter, operation},
    dialects::builtin::{
        BuiltinDialect,
        attributes::{BoolAttr, BytesAttr, I64Attr},
    },
    traits::*,
};

pub type SegmentRef = UnsafeIntrusiveEntityRef<Segment>;

/// Declare a data segment in the shared memory of a [super::Component].
///
/// This operation type is only permitted in the body of a [super::Module] op, it is an error to use
/// it anywhere else. At best it will be ignored.
///
/// Data segments can have a size that is larger than the initializer data it describes; in such
/// cases, the remaining memory is either assumed to be arbitrary bytes, or if `zeroed` is set,
/// it is zeroed so that the padding bytes are all zero.
///
/// A data segment can be marked `readonly`, which indicates to the optimizer that it is allowed
/// to assume that no writes will ever occur in the boundaries of the segment, i.e. a value loaded
/// from within those bounds does not need to be reloaded after side-effecting operations, and
/// can in fact be rescheduled around them. Additionally, if a write is detected that would effect
/// memory in a readonly data segment boundary, an error will be raised.
///
/// NOTE: It is not guaranteed that the optimizer will make any assumptions with regard to data
/// segments. For the moment, even if `readonly` is set, the compiler assumes that segments are
/// mutable.
#[derive(OpPrinter, OpParser)]
#[operation(
    dialect = BuiltinDialect,
    traits(
        SingleBlock,
        NoRegionArguments,
        IsolatedFromAbove,
    ),
    implements(OpPrinter)
)]
pub struct Segment {
    /// The offset from the start of linear memory where this segment starts, held as the bits
    /// of the `u64` (see [Segment::offset_in_bytes])
    #[attr]
    offset: I64Attr,
    /// The data to initialize this segment with, determines the size of the segment
    #[attr]
    data: BytesAttr,
    /// Whether or not this segment is intended to be read-only data
    #[attr]
    #[default]
    readonly: BoolAttr,
}

impl Segment {
    /// The offset, in bytes, from the start of linear memory where this segment starts.
    pub fn offset_in_bytes(&self) -> u64 {
        *self.get_offset() as u64
    }

    /// Set the offset, in bytes, storing the bits of the `u64`.
    pub fn set_offset_in_bytes(&mut self, offset: u64) {
        **self.offset_mut() = offset as i64;
    }

    /// The size, in bytes, of this data segment.
    ///
    /// By default this will be the same size as `init`, unless explicitly given.
    pub fn size_in_bytes(&self) -> usize {
        self.get_data().len()
    }

    /// Get the data, as bytes, to initialize this data segment with.
    pub fn initializer(&self) -> Arc<ConstantData> {
        self.get_data().clone()
    }
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let data = self.initializer();
        f.debug_struct("Segment")
            .field("offset", &self.offset_in_bytes())
            .field("size", &data.len())
            .field("init", &format_args!("{data}"))
            .field("readonly", &self.get_readonly())
            .finish()
    }
}
