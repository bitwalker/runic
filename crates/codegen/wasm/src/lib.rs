//! The lowering of generic IR to Wasm form, the input of the WebAssembly backend.
//!
//! **Wasm form.** A function in Wasm form is a `builtin.function` whose body is made of `wasm`
//! operations (see `runic-dialect-wasm`) and of five builtin ones: `builtin.call`,
//! `builtin.tail_call`, `builtin.ret`, `builtin.load_local` and `builtin.store_local`. Every
//! value in it, every parameter and result of its signature and every local has a Wasm value type
//! (`i32`, `i64`, `f32`, `f64`, `!wasm.funcref`, `!wasm.externref`), its control flow is
//! structured (every region holds one block), and its calling convention is `"wasm"` or `"C"`.
//! The pass [wasm-verify-form](VerifyForm) checks exactly this. Module-level operations
//! (`builtin.global_variable`, `wasm.global`, `wasm.memory`, ...) are not function bodies, and are
//! left as they are.
//!
//! **Carried form.** Each generic type is carried in a Wasm value type ([carrier]):
//!
//! * `i32`, `i64`, `f32`, `f64`, `!wasm.funcref` and `!wasm.externref` are themselves;
//! * an integer `iN` with `N < 32` is carried in an `i32`, one with `32 < N < 64` in an `i64`,
//!   **zero-extended**: wherever the value exists, the bits above `N` are zero;
//! * `isize`, a pointer in address space 0 (typed or opaque) and a value of a function type are
//!   carried in an `i32`;
//! * nothing else has a carrier: an integer wider than 64 bits, an aggregate, a pointer in another
//!   address space. `convert-to-wasm` reports the operation that holds a value of one; for a wide
//!   integer the message says that integers wider than 64 bits are not lowered yet.
//!
//! Every pattern of `convert-to-wasm` keeps that invariant, through the helpers of [carried]. The
//! calling conventions say what crosses a call: under `extern("wasm")` both sides rely on carried
//! form; under `extern("C")` nothing is assumed that the signature does not say (see
//! [convert::calls]).
//!
//! A value of a function type is refused by `convert-to-wasm` for now: the builtin conversion
//! patterns need a type converter that converts function types by their parts
//! (`TypeConverter::with_function_types`), which answers for a function type before any callback,
//! so the converter cannot give such a value its carrier.
//!
//! **The passes, in order.** [add_lowering_passes] adds them to a pass manager:
//!
//! 1. `cfg-to-scf{exits=in-place}` (`runic-dialect-scf`): unstructured control flow becomes
//!    structured, and each return and tail call stays where it is.
//! 2. [`wasm-lower-frames`](LowerFrames): every `mem.alloca` gets a slot in a frame on the shadow
//!    stack (registered, and changing nothing yet).
//! 3. [`convert-to-wasm`](ConvertToWasm): generic operations and types become `wasm` operations
//!    and Wasm value types, in a full dialect conversion of each module.
//! 4. [`wasm-verify-form`](VerifyForm): checks the result; changes nothing.
//!
//! The lowering needs a data layout in scope, little-endian with 32-bit pointers and `isize`
//! ([WasmTarget]).
//!
//! **The well-known symbols.** Lowered code names the linear memory, the function table and the
//! stack pointer by symbol: `wasm.memory @__linear_memory`, `wasm.table
//! @__indirect_function_table` of `!wasm.funcref` and `wasm.global @__stack_pointer : mut i32`.
//! [symbols] finds each in a module, or declares it there as an import from `env` whose link name
//! is its own name.

#![no_std]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod carried;
pub mod convert;
mod frames;
pub mod symbols;
mod target;
pub mod types;
mod verify_form;

#[cfg(test)]
mod tests;

use alloc::format;

use runic_ir::{
    Report,
    pass::{OpPassManager, PassInfo, PassOptions, registry::RegistryEntry},
};

pub use self::{
    convert::ConvertToWasm,
    frames::LowerFrames,
    target::{POINTER_WIDTH, WasmTarget},
    types::{carrier, type_converter, width},
    verify_form::{VerifyForm, verify_wasm_form},
};

/// The passes [add_lowering_passes] adds, in order: each pass's registered name and its options.
pub const LOWERING_PASSES: [(&str, &[(&str, &str)]); 4] = [
    ("cfg-to-scf", &[("exits", "in-place")]),
    (frames::NAME, &[]),
    (convert::NAME, &[]),
    (verify_form::NAME, &[]),
];

/// Adds the passes that lower generic IR to Wasm form to `passes`, by their registered names and
/// in order: `cfg-to-scf{exits=in-place}`, `wasm-lower-frames`, `convert-to-wasm`,
/// `wasm-verify-form` (see the crate documentation).
///
/// An error when one of them is not registered (this crate links all four), or when a pass
/// rejects its options.
pub fn add_lowering_passes(passes: &mut OpPassManager) -> Result<(), Report> {
    for (name, options) in LOWERING_PASSES {
        let info = PassInfo::lookup(name)
            .ok_or_else(|| Report::msg(format!("the pass '{name}' is not registered")))?;
        let mut pass_options = PassOptions::new();
        for (key, value) in options {
            pass_options.push(*key, Some(*value));
        }
        info.add_to_pipeline(passes, &pass_options)?;
    }
    Ok(())
}
