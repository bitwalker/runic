//! The text format of an emitted module, rendered with `wasmprinter`.

use alloc::{format, string::String};

use runic_ir::Report;

/// The text format of the module `bytes`, as `wasmprinter` prints it with its default
/// configuration (flat instructions, no offsets, names from the `name` section).
///
/// An error when `bytes` are not a module `wasmprinter` can read; for a module the emitter
/// delivered, that is a defect of the backend.
pub fn print(bytes: &[u8]) -> Result<String, Report> {
    wasmprinter::print_bytes(bytes).map_err(|error| {
        Report::msg(format!(
            "the emitted WebAssembly module cannot be printed as text, which is a defect of the \
             backend: {error}"
        ))
    })
}
