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
//! The lowering needs a data layout in scope, little-endian with 32-bit pointers and `isize`
//! ([WasmTarget]).
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
//! initializer is data, and is not converted. An `isize` constant the 32-bit `isize` does not hold
//! and a `builtin.unrealized_conversion_cast` to a narrow integer written before the lowering are
//! errors. The README of the crate
//! lists them with their diagnostics, and the unit test `tests::coverage` fails for a source
//! operation that is neither lowered nor listed.
//!
//! **The emitter.** [emit] turns a module in Wasm form into a binary core WebAssembly module. It
//! first builds a [module description](emit::desc::ModuleDesc), in which every reference between
//! entities is a recorded fix-up; the [complete-module writer](emit::write::write_module) assigns
//! the index spaces, resolves the fix-ups and writes the binary with `wasm-encoder`, and
//! [emit::validate::validate] checks it with `wasmparser` under the backend's feature baseline.
//! With the `std` feature, [text::print] renders a module in the text format with `wasmprinter`.

#![no_std]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

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
