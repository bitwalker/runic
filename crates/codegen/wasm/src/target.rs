//! The target facts the lowering reads from the data layout in scope.

use alloc::format;

use runic_ir::{DataLayout, Endianness, Operation, Report, Spanned, diagnostics::report};

/// The width in bits of a pointer in address space 0 and of `isize` on the targets the lowering
/// supports: 32-bit Wasm (memory64 is not modelled).
pub const POINTER_WIDTH: u32 = 32;

/// What the lowering knows of its target: the data layout in scope of the operation it lowers,
/// which is little-endian, with pointers in address space 0 and `isize` 32 bits wide.
#[derive(Clone, Debug)]
pub struct WasmTarget {
    layout: DataLayout,
}

impl WasmTarget {
    /// The target of `op`: the data layout in scope of it ([DataLayout::closest]).
    ///
    /// An error, with a label on `op`, when no data layout is in scope, when the one in scope is
    /// big-endian, and when it does not give pointers in address space 0 and `isize` 32 bits.
    pub fn of(op: &Operation) -> Result<Self, Report> {
        let Some(layout) = DataLayout::closest(op) else {
            return Err(report!(
                message: "`convert-to-wasm` needs a data layout: state `data_layout = \
                          #builtin.data_layout<little, ptr = 32>` on the module, or compile with \
                          `--target`",
                labels: [primary(op.span(), "no data layout is in scope of this operation")],
            ));
        };
        if layout.endianness() != Endianness::Little {
            return Err(report!(
                message: "`convert-to-wasm` needs a little-endian data layout: Wasm memory is \
                          little-endian",
                labels: [primary(op.span(), "the data layout in scope of this operation is big-endian")],
            ));
        }
        let pointer = layout.pointer(0).size_bits;
        if pointer != POINTER_WIDTH {
            return Err(report!(
                message: "`convert-to-wasm` needs 32-bit pointers: 64-bit memories (memory64) are \
                          not modelled",
                labels: [primary(
                    op.span(),
                    format!(
                        "the data layout in scope of this operation gives pointers of {pointer} \
                         bits in address space 0"
                    )
                )],
            ));
        }
        let isize = layout.isize_width();
        if isize != POINTER_WIDTH {
            return Err(report!(
                message: "`convert-to-wasm` needs a 32-bit `isize`, the width of a Wasm address",
                labels: [primary(
                    op.span(),
                    format!("the data layout in scope of this operation gives `isize` {isize} bits")
                )],
            ));
        }
        Ok(Self { layout })
    }

    /// The data layout of the target.
    #[inline]
    pub fn layout(&self) -> &DataLayout {
        &self.layout
    }
}
