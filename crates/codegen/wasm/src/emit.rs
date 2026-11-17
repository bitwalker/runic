//! The emitter: lowered IR in Wasm form becomes a binary core WebAssembly module.
//!
//! The emitter builds a [module description](desc::ModuleDesc), in which every reference from a
//! body, a constant expression or data to another entity is a recorded [fix-up](desc::Fixup), not
//! a number. The [complete-module writer](write::write_module) assigns the index spaces, resolves
//! the fix-ups with the addresses and table slots a layout gives, and writes the binary with
//! `wasm-encoder`, with the custom sections of [sections]; [validate::validate] checks it with
//! `wasmparser` under the backend's [feature baseline](validate::baseline) before it is delivered.

pub mod desc;
pub mod sections;
pub mod validate;
pub mod write;
