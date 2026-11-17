//! The feature baseline of the backend, and the check every emitted module passes before it is
//! delivered.

use alloc::format;

use runic_ir::Report;
use wasmparser::{Validator, WasmFeatures};

/// The WebAssembly features the backend may use, and so the set the validator is given: mutable
/// globals, sign extension, saturating float-to-int conversions, bulk memory, multi-value,
/// reference types and tail calls, with floats and the reference types `wasmparser` gates behind
/// `GC_TYPES` (`externref`); nothing else. Built from the empty set, not from `wasmparser`'s
/// defaults, so that a second memory, exceptions, SIMD or extended constant expressions do not
/// validate.
pub fn baseline() -> WasmFeatures {
    WasmFeatures::empty()
        | WasmFeatures::MUTABLE_GLOBAL
        | WasmFeatures::SIGN_EXTENSION
        | WasmFeatures::SATURATING_FLOAT_TO_INT
        | WasmFeatures::BULK_MEMORY
        | WasmFeatures::MULTI_VALUE
        | WasmFeatures::REFERENCE_TYPES
        | WasmFeatures::TAIL_CALL
        | WasmFeatures::FLOATS
        | WasmFeatures::GC_TYPES
}

/// Checks that `bytes` are a valid core module under [baseline].
///
/// The error says that the emitted module does not validate, that this is a defect of the backend,
/// and gives the validator's message and the byte offset it points at.
pub fn validate(bytes: &[u8]) -> Result<(), Report> {
    Validator::new_with_features(baseline())
        .validate_all(bytes)
        .map(|_| ())
        .map_err(|error| {
            Report::msg(format!(
                "the emitted WebAssembly module does not validate, which is a defect of the \
                 backend: {} (at offset 0x{:x})",
                error.message(),
                error.offset()
            ))
        })
}
