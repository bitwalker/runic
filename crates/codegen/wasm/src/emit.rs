//! The emitter: lowered IR in Wasm form becomes a binary core WebAssembly module.
//!
//! [emit_module] is the entry point. It [collects](collect::collect) every `builtin.module` under
//! the operation it is given into one unit linked by link name, lays out its memory and table
//! ([layout]: the shadow stack, the data, the fixed segments, the table slots, and definitions of
//! the memory, stack pointer and table the lowering imports), writes the module and validates it.
//! The module needs no host: it defines its memory, its stack and its function table.
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
//! operation, recording a fix-up for every reference to another entity. A `wasm.global`'s
//! initializer is a constant expression built by [constexpr]; a `builtin.global_variable`'s
//! initializer becomes the bytes of a data entry by [data].

pub mod body;
pub mod collect;
pub mod constexpr;
pub mod data;
pub mod desc;
pub mod instr;
pub mod layout;
pub mod locals;
pub mod plan;
pub mod sections;
pub mod validate;
pub mod write;

use alloc::{string::String, vec::Vec};

use runic_ir::{OperationRef, Report};

/// How a module is emitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitOptions {
    /// The size of the shadow stack, in bytes: a multiple of 16. The layout reserves it from
    /// address 0 when it defines the stack pointer.
    pub stack_size: u32,
    /// Whether the module carries a `name` section.
    pub names: bool,
    /// The module's name in the `name` section, when there is one.
    pub module_name: Option<String>,
}

impl Default for EmitOptions {
    /// A stack of 1 MiB (what rustc asks `wasm-ld` for on `wasm32-unknown-unknown`), a `name`
    /// section, and no module name.
    fn default() -> Self {
        Self {
            stack_size: 1024 * 1024,
            names: true,
            module_name: None,
        }
    }
}

/// The binary core WebAssembly module of the unit under `root`, validated.
///
/// `root` is a `builtin.world`, a `builtin.component` or a `builtin.module`; every
/// `builtin.module` under it, `root` included, is part of the one module emitted (see
/// [collect]), and its functions must be in Wasm form. What cannot be emitted is an error naming
/// the operation; a module that does not validate is reported as a defect of the backend and is
/// not returned.
pub fn emit_module(root: OperationRef, options: &EmitOptions) -> Result<Vec<u8>, Report> {
    let mut unit = collect::collect(root, options, &body::BodyConfig::default())?;
    let resolved = layout::layout(&mut unit, options)?;
    let bytes = write::write_module(&unit.desc, &resolved)?;
    validate::validate(&bytes)?;
    Ok(bytes)
}
