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
//! **Numeric operations.** Each operation is free of effects, has no fold, and prints its operands
//! only: the result type follows from them (`%r = wasm.add %a, %b;`), except for a conversion,
//! which writes its result type as the property `ty` (`%r = wasm.trunc_s %a <{ ty = i64 }>;`).
//! A comparison gives an `i32`. Where a Wasm instruction traps (an integer division, a truncation
//! of a float), the operation says so and is still free of effects.
//!
//! | Family | Operations | Operands |
//! |---|---|---|
//! | Constants | [Const] (`const`), [ConstF] (`constf`) | an integer or float literal and its type |
//! | Selection | [Select] (`select`) | an `i32` and two values of one value type |
//! | Arithmetic, any number | [Add], [Sub], [Mul] | two of one number type |
//! | Arithmetic, integers | [DivS], [DivU], [RemS], [RemU], [And], [Or], [Xor], [Shl], [ShrS], [ShrU], [Rotl], [Rotr] | two of one integer type |
//! | Counts, integers | [Clz], [Ctz], [Popcnt] | one integer |
//! | Sign extension in place | [Extend8S], [Extend16S], [Extend32S] | one integer (`extend32_s`: `i64`) |
//! | Arithmetic, floats | [Div], [Min], [Max], [Copysign] | two of one float type |
//! | Unary, floats | [Abs], [Neg], [Sqrt], [Ceil], [Floor], [Trunc], [Nearest] | one float |
//! | Comparisons | [Eqz], [struct@Eq], [Ne], [LtS], [LtU], [GtS], [GtU], [LeS], [LeU], [GeS], [GeU], [Lt], [Gt], [Le], [Ge] | one or two numbers; `i32` result |
//! | Conversions | [Wrap], [ExtendS], [ExtendU], [TruncS], [TruncU], [TruncSatS], [TruncSatU], [ConvertS], [ConvertU], [Promote], [Demote], [Reinterpret] | one number; the result type is written |
//!
//! [WasmOpBuilder] builds them. [ConstantInstruction] marks the operations a Wasm constant
//! expression may hold, and [is_constant_instruction] is the precise question to ask about one.
//!
//! **Module-level operations.** A memory, a table, a global, an element segment and a tag are
//! symbols that sit beside the functions of a `builtin.module`, in its one name space. Each is
//! defined, or, written with `declaration` (a global: with no initializer region), an import.
//! [ModuleEntity] marks them, and [referenced_entity] is the check an operation makes that a
//! symbol names the kind it needs (a load names a [Memory]).
//!
//! | Operation | Text | Is |
//! |---|---|---|
//! | [Memory] (`memory`) | `wasm.memory public @mem : 1..16;` | limits in 64 KiB pages, at most 65536 |
//! | [Table] (`table`) | `wasm.table private @tab : 0..8 x !wasm.funcref;` | limits in elements and a reference type |
//! | [Global] (`global`) | `wasm.global private @sp : mut i32 { %v = wasm.const 65536 : i32; };` | a value type, `mut`, and a region of one constant expression |
//! | [Elem] (`elem`) | `wasm.elem private @init : @tab at 0 [@f, @g];` | an active segment of functions of a `funcref` table |
//! | [Tag] (`tag`) | `wasm.tag private @exn : (i32, i64);` | parameter types |
//!
//! **Imports and exports.** Two entries of the attribute dictionary mark them, on a
//! `builtin.function` or on one of the operations above: `wasm.import_module` (a non-empty string,
//! on a declaration; `env` when absent, see [import_module]) and `wasm.export_name` (a string or
//! an array of distinct strings, on a definition, see [export_names]). The dialect checks both on
//! any operation, and rejects any other key that begins with `wasm.`. The field name of an import
//! is the symbol's effective link name. What a module as a whole must satisfy (how many memories
//! it has, the order of its index spaces, that the exported names are unique) is for a later
//! stage.
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
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(ptr_metadata)]
#![allow(incomplete_features)]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod attributes;
mod builders;
mod linking;
mod ops;
mod types;

#[cfg(test)]
mod tests;

use runic_ir::{
    AttributeRef, Builder, Dialect, DialectInfo, Operation, OperationRef, Report, SourceSpan, Type,
    derive::DialectRegistration, interner,
};

pub use self::{
    attributes::{Limits, LimitsAttr, MemArg, MemArgAttr, Mutability, MutabilityAttr},
    builders::WasmOpBuilder,
    linking::{DEFAULT_IMPORT_MODULE, EXPORT_NAME, IMPORT_MODULE, export_names, import_module},
    ops::*,
    types::{
        ExternRefType, FloatType, FuncRefType, I32, I64, IntType, NumType, RefType, ValType,
        is_float_type, is_int_type, is_num_type, is_ref_type, is_val_type, val_type_name,
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

    /// Checks the dictionary entries of any operation whose key begins with `wasm.`:
    /// `wasm.import_module` and `wasm.export_name`, which mark imports and exports (see
    /// [import_module] and [export_names]). Any other `wasm.` key is an error.
    fn verify_operation_attribute(
        &self,
        op: &Operation,
        name: interner::Symbol,
        attr: &AttributeRef,
    ) -> Result<(), Report> {
        linking::verify_entry(op, name, attr)
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
