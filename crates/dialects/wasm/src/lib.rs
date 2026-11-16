//! The `wasm` dialect: WebAssembly's instructions as SSA operations.
//!
//! A WebAssembly function is a `builtin.function`, its locals are the function's locals
//! (`builtin.local_variable`), and its data is `builtin.global_variable`. The dialect adds what a
//! Wasm module has beyond those: operations that mirror the numeric, reference, memory, table and
//! global instructions, and module-level operations for memories, tables, globals and tags. It
//! follows release 3.0 of the WebAssembly core specification. Nothing lowers to the dialect and
//! nothing emits a binary from it: it prints, parses, verifies, and is handled by the generic
//! passes.
//!
//! **Names.** One operation stands for each family of instructions that differ only by their
//! type: `i32.div_s` is `wasm.div_s` on an `i32`, and so is `i64.div_s` on an `i64`. A dot of a
//! Wasm name becomes an underscore: `memory.grow` is `wasm.memory_grow`, `global.get` is
//! `wasm.global_get`.
//!
//! **Values.** An operand or a result of a `wasm` operation has a Wasm value type: `i32`, `i64`,
//! `f32`, `f64`, [`!wasm.funcref`](FuncRefType) or [`!wasm.externref`](ExternRefType). Conditions
//! and the results of comparisons are `i32`, as in Wasm, not `i1`. The type constraints
//! [IntType], [FloatType], [NumType], [RefType] and [ValType] say which of them an operation
//! accepts, and [is_val_type], [is_num_type], [is_ref_type] and [val_type_name] serve verifiers.
//! The other builtin types (`i8`, `i1`, `isize`, pointers, structs) are not Wasm value types and
//! no `wasm` operation accepts them.
//!
//! **Attributes.** [LimitsAttr] (`#wasm.limits<1..16>`) bounds the size of a memory or a table,
//! [MemArgAttr] (`#wasm.memarg<offset = 8, align = 4>`) is the static part of a memory access,
//! and [MutabilityAttr] (`#wasm.mutability<mut>`) says whether a global may be written.
//!
//! **Traps.** A trap is not an effect (the rule `arith` states): an operation that may trap, such
//! as an integer division or a truncation of a float, has no effects and may be removed when its
//! result is unused. An operation has an effect only when it reads or writes a memory, a global
//! or a table.
//!
//! **Control flow.** Structured control flow, branches and calls are those of `builtin`, `cf` and
//! `scf`; the dialect does not repeat them.
//!
//! **What is not here.** Garbage-collected types and their instructions, exceptions, SIMD,
//! threads and atomics, stack switching, and the Component Model.

#![no_std]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod attributes;
mod types;

#[cfg(test)]
mod tests;

use runic_ir::{
    AttributeRef, Builder, Dialect, DialectInfo, OperationRef, SourceSpan, Type,
    derive::DialectRegistration,
};

pub use self::{
    attributes::{Limits, LimitsAttr, MemArg, MemArgAttr, Mutability, MutabilityAttr},
    types::{
        ExternRefType, FloatType, FuncRefType, IntType, NumType, RefType, ValType, is_float_type,
        is_int_type, is_num_type, is_ref_type, is_val_type, val_type_name,
    },
};

/// The `wasm` dialect, registered as `wasm`.
#[derive(Debug, DialectRegistration)]
#[dialect(name = "wasm")]
pub struct WasmDialect {
    info: DialectInfo,
}

impl From<DialectInfo> for WasmDialect {
    fn from(info: DialectInfo) -> Self {
        Self { info }
    }
}

impl Dialect for WasmDialect {
    #[inline]
    fn info(&self) -> &DialectInfo {
        &self.info
    }

    /// The dialect has no folds, so it materializes no constant: `None`.
    fn materialize_constant(
        &self,
        _builder: &mut dyn Builder,
        _attr: AttributeRef,
        _ty: Type,
        _span: SourceSpan,
    ) -> Option<OperationRef> {
        None
    }
}
