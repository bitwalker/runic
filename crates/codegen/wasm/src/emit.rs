//! The emitter: lowered IR in Wasm form becomes a binary core WebAssembly module.
//!
//! The emitter builds a [module description](desc::ModuleDesc), in which every reference from a
//! body, a constant expression or data to another entity is a recorded [fix-up](desc::Fixup), not
//! a number. The [complete-module writer](write::write_module) assigns the index spaces, resolves
//! the fix-ups with the addresses and table slots a layout gives, and writes the binary with
//! `wasm-encoder`, with the custom sections of [sections]; [validate::validate] checks it with
//! `wasmparser` under the backend's [feature baseline](validate::baseline) before it is delivered.
//!
//! A function body is encoded by [body::encode_function]: [plan] decides where each value waits
//! between its definition and its uses (on the stack, in a local, or written again), [locals]
//! gives locals to the values that need them, and [instr] writes the instruction of each
//! operation, recording a fix-up for every reference to another entity.

pub mod body;
pub mod desc;
pub mod instr;
pub mod locals;
pub mod plan;
pub mod sections;
pub mod validate;
pub mod write;
