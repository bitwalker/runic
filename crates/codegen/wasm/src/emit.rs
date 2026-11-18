//! The emitter: lowered IR in Wasm form becomes a binary core WebAssembly module.
//!
//! The emitter has two ends over one [collected](collect::collect) unit: every `builtin.module`
//! under the operation it is given, linked by link name. [emit_module] gives a complete module: it
//! exports the unit's start function as `_initialize` ([collect::Unit::export_start]), lays out
//! its memory and table ([layout]: the shadow stack, the data, the fixed segments, the table
//! slots, and definitions of the memory, stack pointer and table the lowering imports), writes the
//! module and validates it. [emit_object] gives a relocatable object, which leaves that layout to
//! the linker: it never runs [layout], refuses what an object cannot express
//! ([collect::Unit::check_object]), and writes the unit with [object::write_object].
//! The memory, stack pointer and function table the lowering declares (imports from `env` of
//! `__linear_memory`, `__stack_pointer` and `__indirect_function_table`) are defined by the layout
//! and never imported, so a module of lowered generic IR needs no host for them. A memory or a
//! table the unit declares under a link name of its own is an import, like any other declaration
//! with no definition.
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
pub mod object;
pub mod plan;
pub mod sections;
pub mod validate;
pub mod write;

use alloc::{string::String, vec::Vec};

use runic_ir::{OperationRef, Report};

/// How a module is emitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitOptions {
    /// The size of the shadow stack, in bytes: a positive multiple of 16, which is checked
    /// whether or not the unit uses the stack, for an object too. The layout reserves it from address 0 when it
    /// defines the stack pointer.
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
///
/// The unit's start function, when it has one, is exported as `_initialize`; the module has no
/// start section, and its embedder calls `_initialize` once after instantiating it (`wasmtime`
/// does so itself).
pub fn emit_module(root: OperationRef, options: &EmitOptions) -> Result<Vec<u8>, Report> {
    let mut unit = collect::collect(root, options, &body::BodyConfig::default())?;
    unit.export_start()?;
    let resolved = layout::layout(&mut unit, options)?;
    let bytes = write::write_module(&unit.desc, &resolved)?;
    validate::validate(&bytes)?;
    Ok(bytes)
}

/// A relocatable object for the unit under `root`, validated: the unit in the tool conventions'
/// linking format, which `wasm-ld` links with other objects ([object]).
///
/// `root` is what [emit_module] takes. The unit is collected as for a module, and what an object
/// cannot express is an error naming the operation ([collect::Unit::check_object]), as is data
/// that does not fit a 32-bit memory; the layout is the linker's and never runs, so the stack
/// size of `options` and its names do not change the object. The stack size is checked all the
/// same, first, with the error a complete module gives for it: a value is valid or not whatever
/// is asked for. The start function, when the unit has one, is
/// the object's init function, which the linker's `__wasm_call_ctors` calls; it is not exported.
/// An object that does not validate is reported as a defect of the backend and is not returned.
pub fn emit_object(root: OperationRef, options: &EmitOptions) -> Result<Vec<u8>, Report> {
    if let Some(message) = layout::stack_size_problem(options.stack_size) {
        return Err(Report::msg(message));
    }
    let unit = collect::collect(root, options, &body::BodyConfig::default())?;
    unit.check_object()?;
    object::check_data_size(&unit)?;
    let bytes = object::write_object(&unit).map_err(object::object_error)?;
    validate::validate(&bytes)?;
    Ok(bytes)
}
