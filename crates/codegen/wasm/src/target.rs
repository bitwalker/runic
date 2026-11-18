//! The target facts the lowering reads from the data layout in scope.

use alloc::format;

use runic_ir::{DataLayout, Endianness, Operation, Report, Spanned, diagnostics::report};

/// The width in bits of a pointer in address space 0 and of `isize` on the targets the lowering
/// supports: 32-bit Wasm (memory64 is not modelled).
pub const POINTER_WIDTH: u32 = 32;

/// The width in bits of a function value in memory on the targets the lowering supports: an
/// `i32`, the index of the function in the indirect function table.
pub const FUNCTION_WIDTH: u32 = 32;

/// What the lowering knows of its target: the data layout in scope of the operation it lowers,
/// which is little-endian, with pointers in address space 0 and `isize` 32 bits wide, and with
/// function values of 32 bits when it gives them a size (`fn = 32`).
#[derive(Clone, Debug)]
pub struct WasmTarget {
    layout: DataLayout,
}

impl WasmTarget {
    /// The target of `op`: the data layout in scope of it ([DataLayout::closest]).
    ///
    /// An error, with a label on `op`, when no data layout is in scope, when the one in scope is
    /// big-endian, when it does not give pointers in address space 0 and `isize` 32 bits, and
    /// when it gives function values another size than 32 bits. A layout with no `fn` entry is
    /// accepted: a function value then has no layout, and is not loaded or stored.
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
        if let Some(function) = layout.function()
            && function.size_bits != FUNCTION_WIDTH
        {
            let bits = function.size_bits;
            return Err(report!(
                message: "`convert-to-wasm` needs a function value of 32 bits, `fn = 32` in the \
                          data layout: a function value is an `i32`, its index in the function \
                          table",
                labels: [primary(
                    op.span(),
                    format!(
                        "the data layout in scope of this operation gives function values of \
                         {bits} bits"
                    )
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
