//! The WebAssembly backend: the lowering of generic IR to Wasm form, the emitter that makes a
//! binary core module of it, and the backend `wasm` of the compile harness that runs both.
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
//! **Carried form.** Each generic type is carried in a Wasm value type ([carrier]), and wherever a
//! value exists the bits of its carrier above its width are zero:
//!
//! | generic type | carried in |
//! |---|---|
//! | `i1` to `i31`, `i33` to `i63` | `i32`, `i64`, zero-extended |
//! | `i32`, `i64`, `f32`, `f64`, `!wasm.funcref`, `!wasm.externref` | themselves |
//! | `isize` | `i32` |
//! | a pointer in address space 0, typed or opaque | `i32`, an address in linear memory |
//! | a value of a function type | `i32`, an index in the indirect-function table |
//! | an integer wider than 64 bits, an aggregate, a pointer in another address space | nothing |
//!
//! `convert-to-wasm` reports the operation that holds a value that has no carrier; for a wide
//! integer the message says that integers wider than 64 bits are not lowered yet. Every pattern of
//! `convert-to-wasm` keeps the invariant, through the helpers of [carried].
//!
//! **The calling conventions.** `extern("wasm")` and `extern("C")` are lowered (see
//! [convert::calls]); any other is an error, on a declaration that nothing calls too. Under
//! `extern("wasm")` both sides of a call rely on carried form and nothing is masked or extended, so
//! a public `extern("wasm")` function trusts every caller (a host or another module included) to
//! pass its narrow parameters in carried form. Under `extern("C")` nothing is assumed that the
//! signature does not say: a function masks each narrow parameter not marked `zext` at its entry
//! and sign-extends each narrow result marked `sext` at each return, and a call sign-extends each
//! argument of a narrow `sext` parameter and masks each narrow result not marked `zext`.
//!
//! A value of a function type is the index of the function in the module's indirect-function
//! table; an indirect call calls through that table. `wasm.func_index` names the function by the
//! path it was written with, an alias included; resolving it to an index is the emitter's and the
//! linker's.
//!
//! **Traps.** Where `arith` traps, the lowering traps: a division by zero, and the minimum value
//! divided by -1 in `arith.sdiv` and `arith.sdivrem` at every width (below the carrier's width the
//! quotient is tested, since the sign-extended `wasm.div_s` gives it without trapping). Where
//! `arith` leaves a result undefined, any value that keeps carried form is given.
//!
//! **The passes, in order.** [add_lowering_passes] adds them to a pass manager. The examples show
//! the operations they change:
//!
//! 1. `cfg-to-scf{exits=in-place}` (`runic-dialect-scf`): unstructured control flow becomes
//!    structured, and each return and tail call stays where it is.
//!
//!    ```text
//!    cf.cond_br %c ^yes, ^no : (i1);
//!    ^yes:
//!        builtin.ret %v : (i32);
//!    ^no:
//!        builtin.tail_call @g(%w) : extern("C") (i32) -> i32;
//!    ```
//!
//!    becomes
//!
//!    ```text
//!    scf.if %c then {
//!        builtin.ret %v : (i32);
//!    } else {
//!        builtin.tail_call @g(%w) : extern("C") (i32) -> i32;
//!    } : (i1) -> ();
//!    ub.unreachable;
//!    ```
//!
//! 2. [`wasm-lower-frames`](LowerFrames): every `mem.alloca` gets a slot in a frame on the shadow
//!    stack, addressed from the stack pointer, which each return and tail call restores.
//!
//!    ```text
//!    %p = mem.alloca i32;
//!    ...
//!    builtin.ret %v : (i32);
//!    ```
//!
//!    becomes (a frame of 16 bytes: the slot is at offset 0)
//!
//!    ```text
//!    %sp = wasm.global_get @__stack_pointer : i32;
//!    %size = wasm.const 16 : i32;
//!    %fp = wasm.sub %sp, %size;
//!    wasm.global_set @__stack_pointer, %fp;
//!    %p = builtin.unrealized_conversion_cast %fp : i32 to ptr<i32>;
//!    ...
//!    wasm.global_set @__stack_pointer, %sp;
//!    builtin.ret %v : (i32);
//!    ```
//!
//! 3. [`convert-to-wasm`](ConvertToWasm): generic operations and types become `wasm` operations
//!    and Wasm value types, in a full dialect conversion of each module. Under `extern("wasm")`,
//!    `%s = arith.add %a, %b` of two `i8` becomes
//!
//!    ```text
//!    %t = wasm.add %a, %b;
//!    %k = wasm.const 255 : i32;
//!    %s = wasm.and %t, %k;
//!    ```
//!
//! 4. [`wasm-verify-form`](VerifyForm): checks the result; changes nothing. An operation or a value
//!    that is not in Wasm form is an error that names it and the pass that should have removed it.
//!
//! The lowering needs a data layout in scope, little-endian with 32-bit pointers and `isize`, and
//! with function values of 32 bits or no `fn` entry ([WasmTarget]). Under `fn = 32`, which the
//! backend's layout states, a function value in memory is its 32-bit index in the function table:
//! `mem.load` and `mem.store` move it as an `i32`, and `mem.gep` and the size operations step
//! through a struct that holds one.
//!
//! **The well-known symbols.** Lowered code names the linear memory, the function table and the
//! stack pointer by symbol: `wasm.memory @__linear_memory`, `wasm.table
//! @__indirect_function_table` of `!wasm.funcref` and `wasm.global @__stack_pointer : mut i32`.
//! [symbols] finds each in the module that directly holds the function whose code names it, or
//! declares it there as a private import from `env`, with the link names `__linear_memory`,
//! `__indirect_function_table` and `__stack_pointer` (each is its own name, not its path in the
//! module). A nested module has its own; nothing is declared in an outer module on behalf of an
//! inner one. Only the symbols the lowering itself needs are declared: IR that already uses `wasm`
//! operations naming one declares it itself.
//!
//! **A failed conversion changes nothing.** `convert-to-wasm` checks every module before it
//! changes any, and every diagnostic it gives on IR that verifies comes from that check
//! ([ConvertToWasm]).
//!
//! **What is not lowered.** Integers wider than 64 bits, aggregates as values, address spaces other
//! than 0, a `mem.alloca` in a loop, big-endian layouts and 64-bit pointers are errors, and so are
//! the operations that need an earlier pass (`mem.alloca` without `wasm-lower-frames`, `cf.br`,
//! `cf.cond_br` and `cf.switch` without `cfg-to-scf`). A return or tail call inside a loop is
//! dispatched after the loop, and the message of a `cf.assert` is dropped. A global variable's
//! initializer is data, and is not converted, except that the function types a global holds
//! become the function types of their converted signatures, as the functions they name do. An
//! `isize` constant the 32-bit `isize` does not hold
//! and a `builtin.unrealized_conversion_cast` to a narrow integer written before the lowering are
//! errors. The README of the crate
//! lists them with their diagnostics, and the unit test `tests::coverage` fails for a source
//! operation that is neither lowered nor listed.
//!
//! **The emitter.** [emit::emit_module] turns a `builtin.world`, `builtin.component` or
//! `builtin.module` in Wasm form into the bytes of a validated core WebAssembly module. It first
//! builds a [module description](emit::desc::ModuleDesc), in which nothing is an index: every
//! reference from a body, a constant expression or data to another entity (a function, a global, a
//! table, a type, a data address, a table slot) is a placeholder with a recorded
//! [fix-up](emit::desc::Fixup). The [layout](emit::layout) places the stack, the data and the table
//! slots; the [complete-module writer](emit::write::write_module) assigns the index spaces, resolves
//! the fix-ups and writes the binary with `wasm-encoder`; and [emit::validate::validate] checks it
//! with `wasmparser` under the [feature baseline](emit::validate::baseline) (mutable globals, sign
//! extension, saturating conversions, bulk memory, multi-value, reference types, tail calls). A
//! module that does not validate is a defect of the backend, reported and never delivered; what
//! the baseline excludes (a second memory, `wasm.tag`, an extended constant expression) is an error
//! naming the operation.
//!
//! Code is written in the order of the IR ([emit::plan]): a value used once, by the next operation,
//! stays on the Wasm stack; a constant is written again at each use; any other value waits in a
//! local, reused by type ([emit::locals]). `%s = wasm.add %x, %y; %p = wasm.mul %s, %k; %d =
//! wasm.sub %p, %s` with `%k = wasm.const 3 : i32` is
//!
//! ```text
//! local.get 0
//! local.get 1
//! i32.add
//! local.set 2
//! local.get 2
//! i32.const 3
//! i32.mul
//! local.get 2
//! i32.sub
//! ```
//!
//! **One compilation, one module.** Every `builtin.module` under the root is part of the one module
//! emitted, linked by link name ([emit::collect]): a declaration is the definition of its link name
//! when the unit has one, and an import (from `env`, or the module `wasm.import_module` names, with
//! the link name as the field) otherwise. Only what `wasm.export_name` names is exported. The
//! module needs no host: the layout defines the memory, the stack pointer and the function table
//! the lowering declares as imports from `env`. With a frame in the unit, linear memory is
//!
//! ```text
//! 0 ............ stack size   the shadow stack, growing down from `__stack_pointer`
//! stack size ... data end     every builtin.global_variable, aligned by its type
//! data end ..... heap base    rounded up to 16
//! ```
//!
//! and the data starts at 1024 without one. The undefined data symbols `__data_end`,
//! `__heap_base`, `__start_<s>` and `__stop_<s>` are addresses the layout gives. Table slots start
//! at 1, so that a null function pointer traps. A global variable's initializer becomes bytes
//! through the `mem` dialect's evaluator, [runic_dialect_mem::evaluate_initializer], which every
//! backend shares ([emit::data]): constants and what folds to one, `builtin.zero`, the address of
//! a global variable plus a constant offset (`builtin.global_symbol` and `mem.gep`), the slot of a
//! function (`builtin.function_ref`), and structs and arrays of those (`builtin.aggregate`). The
//! layout writes each address and slot into the bytes. Arithmetic on an address, an operation that
//! does not fold, and an address or a slot in a passive segment are errors naming the global. A
//! dispatch table of `struct<i32, i32, (i64) -> i64>` entries in one section, walked from
//! `__start_<s>` to `__stop_<s>` and called through with `mem.load` and `builtin.call_indirect`,
//! is a program of the run suite (`tests/run/wasm/dispatch-table.runic`); the README of the crate
//! shows it.
//!
//! **The backend.** With the `std` feature, [WasmBackend] is the backend `wasm` of `runic-compile`
//! for `wasm32-unknown-unknown` ([backend]): it states the data layout of the C ABI of wasm32, adds
//! the four passes of the lowering (with `cse` and `dce` before `wasm-verify-form` above `-O0`),
//! and emits at the checkpoints `wasm.wat` (the text, through [text::print] and `wasmprinter`) and
//! `wasm.module` (the binary, the default stop point), with the option `--wasm-stack-size`:
//!
//! ```text
//! runic-opt program.runic --target wasm32-unknown-unknown -o program.core.wasm
//! runic-opt program.runic --target wasm32-unknown-unknown --stop-after wat
//! wasmtime run --invoke main program.core.wasm
//! ```
//!
//! **The start function and the object.** A function definition written with `wasm.start =
//! #builtin.unit` is the unit's start function, at most one per compilation: a complete module
//! exports it as `_initialize`, the convention of a WASI reactor, and has no start section, so its
//! embedder calls it once after instantiating the module (`wasmtime run` does so itself). The
//! emitter's second end, [emit::emit_object], is for a relocatable object that `wasm-ld` links: it
//! collects the same unit, records for each entity what its linker symbol needs (its binding from
//! its visibility, whether an import names its host or is the linker's to resolve, which data a
//! named section retains, a function's export name), and refuses with a diagnostic naming the
//! operation what an object cannot express ([emit::collect::Unit::check_object]). The backend
//! writes the object at the checkpoint `wasm.object` (alias `object`) when the request stops there
//! or asks for an output there, and never otherwise.
//!
//! **Linking an object.** An object is linked by `rustc` with a runtime written in Rust, passed as
//! a linker argument, or by `rust-lld -flavor wasm` directly; `--fatal-warnings` makes a signature
//! mismatch fail the link instead of becoming a trap:
//!
//! ```text
//! runic-opt scaled.runic --target wasm32-unknown-unknown --stop-after object -o scaled.o
//! rustc --edition 2024 --target wasm32-unknown-unknown --crate-type cdylib -O rt.rs \
//!     -C link-arg=scaled.o -C link-arg=--fatal-warnings -o scaled.wasm
//! ```
//!
//! A declaration with no definition is an undefined symbol that the linker resolves, unless it
//! is written with `wasm.import_module`, which makes it a host import. `public`, `internal` and
//! `private` become a global, a hidden and a local symbol; a `public` or `internal` alias is a
//! second symbol of its target's function. Export names must be unique across the whole link, the
//! runtime's included: the linker does not check them. Each global variable has a data segment
//! of its own, and those with one `section` are one segment of that name, which the linker retains
//! and bounds with `__start_<section>` and `__stop_<section>`; each object's contribution is
//! aligned on its own, so the entries of a section should share one alignment and have a size
//! that is a multiple of it. A start function is an init function of the object, which the linker's
//! `__wasm_call_ctors` calls: `_initialize` calls it on the WASI targets, and a runtime for
//! `wasm32-unknown-unknown` calls it once itself, or the linker wraps every export to call it on
//! every call. The README of the crate tabulates what has no object form, the features an object
//! declares and the toolchains the tests assume.
//!
//! Not emitted yet: integers wider than 64 bits, components, exported data, DWARF, and, in
//! objects, weak symbols, COMDAT groups, thread-local data, position-independent code and
//! archives. The README of the crate says more of each part.

#![no_std]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(feature = "std")]
pub mod backend;
pub mod carried;
pub mod convert;
pub mod emit;
mod frames;
pub mod symbols;
mod target;
#[cfg(feature = "std")]
pub mod text;
pub mod types;
mod verify_form;

#[cfg(test)]
mod tests;

use alloc::format;

use runic_ir::{
    OperationRef, Report, Spanned,
    diagnostics::report,
    pass::{OpPassManager, PassInfo, PassOptions, registry::RegistryEntry},
};

#[cfg(feature = "std")]
pub use self::backend::WasmBackend;
pub use self::{
    convert::ConvertToWasm,
    frames::LowerFrames,
    target::{FUNCTION_WIDTH, POINTER_WIDTH, WasmTarget},
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

/// The error of the pass `pass` run on `root` when `root` is not a `builtin.module` and holds none
/// (through worlds and components): each of the passes of the lowering works on modules.
pub(crate) fn no_module_error(pass: &str, root: OperationRef) -> Report {
    let root = root.borrow();
    let name = root.name();
    report!(
        message: (format!(
            "`{pass}` runs on a `builtin.module`, or on a world or a component that holds one, but \
             this `{name}` holds none"
        )),
        labels: [primary(root.span(), format!("this `{name}` is not a `builtin.module` and holds none"))],
    )
}

/// Adds the passes that lower generic IR to Wasm form to `passes`, by their registered names and
/// in order: `cfg-to-scf{exits=in-place}`, `wasm-lower-frames`, `convert-to-wasm`,
/// `wasm-verify-form` (see the crate documentation).
///
/// An error when one of them is not registered (this crate links all four), or when a pass
/// rejects its options.
pub fn add_lowering_passes(passes: &mut OpPassManager) -> Result<(), Report> {
    for (name, options) in LOWERING_PASSES {
        add_lowering_pass(passes, name, options)?;
    }
    Ok(())
}

/// Adds the registered pass `name` with `options` to `passes`.
///
/// An error when no pass `name` is registered, or when the pass rejects its options.
pub(crate) fn add_lowering_pass(
    passes: &mut OpPassManager,
    name: &str,
    options: &[(&str, &str)],
) -> Result<(), Report> {
    let info = PassInfo::lookup(name)
        .ok_or_else(|| Report::msg(format!("the pass '{name}' is not registered")))?;
    let mut pass_options = PassOptions::new();
    for (key, value) in options {
        pass_options.push(*key, Some(*value));
    }
    info.add_to_pipeline(passes, &pass_options)
}
