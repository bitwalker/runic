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
//! The conversions are named for the Wasm instruction without the type it comes from, which the
//! operand carries: [Wrap] is `i32.wrap_i64`, [ExtendS] and [ExtendU] are `i64.extend_i32_s` and
//! `i64.extend_i32_u`, [TruncS] and [TruncU] are `i32.trunc_f32_s` .. `i64.trunc_f64_u` (they
//! trap), [TruncSatS] and [TruncSatU] are the `trunc_sat` forms (they do not), [ConvertS] and
//! [ConvertU] are `f32.convert_i32_s` .. `f64.convert_i64_u`, [Promote] is `f64.promote_f32`,
//! [Demote] is `f32.demote_f64`, and [Reinterpret] is the four `reinterpret` instructions
//! (`i32.reinterpret_f32`, `f32.reinterpret_i32`, `i64.reinterpret_f64`, `f64.reinterpret_i64`).
//! The comparisons are the `eq`, `ne`, `eqz`, `lt_s`, `lt_u` .. `ge_u` of the integers and the `lt`,
//! `gt`, `le`, `ge` of the floats (`eq` and `ne` take either); [Const] and [ConstF] are `i32.const`
//! .. `f64.const`, and [Select] is the typed `select`.
//!
//! [WasmOpBuilder] builds them. [ConstantInstruction] marks the operations a Wasm constant
//! expression may hold, and [is_constant_instruction] is the precise question to ask about one.
//!
//! **Module-level operations.** A memory, a table, a global, an element segment and a tag are
//! symbols that sit beside the functions of a `builtin.module`, in its one name space. Each is
//! defined, or, written with `declaration`, an import (a global too: `wasm.global private
//! declaration @g : i32;`, with no initializer region), except the element segment, which cannot
//! be an import. [ModuleEntity] marks them, and [referenced_entity] is the check an operation makes that a
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
//! **Memory.** A load, a store and the bulk operations name the [Memory] they work on by symbol,
//! before their operands (`wasm.load @mem, %p : i32`). The memory is 32-bit, so an address is an
//! `i32`. A load writes the type it gives; an access that is not of the whole value writes
//! `as i8`, `as i16` or `as i32`, and a narrow load how it extends (`signed` or `unsigned`); the
//! static part is `<{ offset = N, align = N }>`, each entry optional (the offset is 0 and the
//! alignment the width of the access when it is not written, and the printer leaves out what is
//! so). Exactly the fourteen loads and nine stores of Wasm verify: [Storage] (`full`, `i8`, `i16`,
//! `i32`) and [Extension] (`unextended`, `signed`, `unsigned`) are the properties that say which.
//!
//! | Operation | Text | Wasm |
//! |---|---|---|
//! | [Load] (`load`) | `%v = wasm.load @mem, %p as i8 signed <{ offset = 8 }> : i64;` | `i32.load` .. `i64.load32_u` |
//! | [Store] (`store`) | `wasm.store @mem, %p, %v as i16;` | `i32.store` .. `i64.store32` |
//! | [MemorySize] (`memory_size`), [MemoryGrow] (`memory_grow`) | `%n = wasm.memory_size @mem;`, `%o = wasm.memory_grow @mem, %delta;` | `memory.size`, `memory.grow` |
//! | [MemoryCopy] (`memory_copy`), [MemoryFill] (`memory_fill`) | `wasm.memory_copy @dst, @src, %d, %s, %n;`, `wasm.memory_fill @mem, %d, %v, %n;` | `memory.copy`, `memory.fill` |
//! | [MemoryInit] (`memory_init`), [DataDrop] (`data_drop`) | `wasm.memory_init @mem, @data, %d, %s, %n;`, `wasm.data_drop @data;` | `memory.init`, `data.drop` (passive data is a `builtin.global_variable`) |
//!
//! **Variables, references and calls.** A global and a table are named by symbol too.
//!
//! | Operation | Text | Wasm |
//! |---|---|---|
//! | [GlobalGet] (`global_get`), [GlobalSet] (`global_set`) | `%v = wasm.global_get @g : i32;`, `wasm.global_set @g, %v;` | `global.get`, `global.set` (the global is `mut`) |
//! | [TableGet] (`table_get`), [TableSet] (`table_set`) | `%r = wasm.table_get @t, %i : !wasm.funcref;`, `wasm.table_set @t, %i, %r;` | `table.get`, `table.set` |
//! | [TableSize] (`table_size`), [TableGrow] (`table_grow`), [TableFill] (`table_fill`) | `%n = wasm.table_size @t;`, `%o = wasm.table_grow @t, %init, %delta;`, `wasm.table_fill @t, %i, %r, %n;` | `table.size`, `table.grow`, `table.fill` |
//! | [RefNull] (`ref_null`), [RefFunc] (`ref_func`), [RefIsNull] (`ref_is_null`) | `%r = wasm.ref_null : !wasm.funcref;`, `%r = wasm.ref_func @f;`, `%b = wasm.ref_is_null %r;` | `ref.null`, `ref.func`, `ref.is_null` |
//! | [DataAddr] (`data_addr`), [FuncIndex] (`func_index`) | `%a = wasm.data_addr @data + 16;`, `%i = wasm.func_index @f;` | **not instructions**: each stands for an `i32.const` that a layout or a linker decides |
//! | [CallIndirect] (`call_indirect`), [ReturnCallIndirect] (`return_call_indirect`) | `%r = wasm.call_indirect @t, %i(%a) : extern("C") (i32) -> i32;`, `wasm.return_call_indirect @t, %i(%a) : extern("C") (i32) -> i32;` | `call_indirect`, `return_call_indirect` |
//!
//! `global_get` of an immutable global, `ref_null`, `ref_func`, `data_addr` and `func_index` may
//! appear in a constant expression ([ConstantInstruction]; ask [is_constant_instruction]).
//!
//! **Effects.** The operations that touch state declare memory effects on a resource of the
//! dialect ([LinearMemoryResource], [GlobalsResource], [TablesResource]; one for all the
//! memories, one for all the globals, one for all the tables), so that `cse`, `dce` and the other
//! passes treat them correctly. A read (`load`, `memory_size`, `global_get`, `table_get`,
//! `table_size`) is merged with an equal one unless an operation between them may write what it
//! reads, and is removed when unused; a write (`store`, `memory_grow`, `memory_copy`,
//! `memory_fill`, `memory_init`, `data_drop`, `global_set`, `table_set`, `table_grow`,
//! `table_fill`) is never removed. [LinearMemoryResource] is a child of the `mem` dialect's memory
//! resource, so a `wasm.store` and a `mem.load` of one function are ordered, as they touch the
//! same bytes; globals and tables are not memory and are disjoint from it. An indirect call has
//! unknown effects (it implements no effect interface), as `builtin.call_indirect` does. The
//! references and the address operations have no effects.
//!
//! **Imports and exports.** Two entries of the attribute dictionary mark them, on a
//! `builtin.function` or on one of the operations above: `wasm.import_module` (a string, the empty
//! one included, on a declaration; `env` when absent, see [import_module]) and `wasm.export_name`
//! (a string or an array of distinct strings, the empty one included, on a definition or on a
//! declaration: Wasm exports what it imports, see [export_names]). The dialect checks both on any
//! operation, and rejects any other key that begins with `wasm.`. The field name of an import is
//! the symbol's effective link name.
//!
//! **What is not checked.** What a module as a whole must satisfy is for a later stage, and the
//! emitter has to satisfy it:
//!
//! - how many memories and tables the module has, the order of its index spaces, and that the
//!   exported names are unique across the module;
//! - a function named by a [RefFunc] in a function body must be *declared* in the module: named by
//!   an element segment, by an export, or by a global's initializer. The dialect accepts any
//!   function; the emitter writes a declarative element segment (`(elem declare func $f)`) for
//!   each one that nothing else declares;
//! - [Select] of a reference type is Wasm's typed `select t`, and of a number type the untyped
//!   `select`: the emitter writes the form the type needs;
//! - [FuncIndex] gives an index in the module's indirect-function table and names no table: the
//!   layout (or the linker, by relocation) decides which `funcref` table that is and the position
//!   of the function in it; the dialect assumes neither;
//! - the initializer of a global is a Wasm constant expression, which is a tree. The dialect's
//!   region is a list of SSA values: it may hold an operation whose result nothing uses, and a
//!   value used twice. The value of the global is the result of the last operation; the emitter
//!   rebuilds the tree from it, dropping what is unused and writing a shared operation at each of
//!   its uses (every constant instruction is pure, so that is sound). A `global_get` in an
//!   initializer is checked to read an immutable global that is an import or is defined before it.
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
//! **Control flow.** Direct calls and returns are those of `builtin` (`builtin.call`,
//! `builtin.tail_call`, `builtin.ret`; Wasm's `br` that names the function is `builtin.ret`). The
//! dialect adds Wasm's structured control flow, with the operand stack turned into SSA: the
//! parameters of a `block`, `loop` or `if` are its operands, which are the arguments of the entry
//! block of its body, and its results are the results of the operation.
//!
//! | Operation | Text | Wasm |
//! |---|---|---|
//! | [BlockOp] (`block`) | `%r = wasm.block %a { ^body(%x: i32): .. wasm.end %y : (i32); } : (i32) -> (i32);` | `block` |
//! | [Loop] (`loop`) | `%r = wasm.loop %a { ^body(%x: i32): .. wasm.br 0 %n : (i32); } : (i32) -> (i32);` | `loop` |
//! | [If] (`if`) | `%r = wasm.if %c, %a then { ^t(%x: i32): .. } else { ^e(%x: i32): .. } : (i32, i32) -> (i32);` | `if`, `else` |
//! | [End] (`end`) | `wasm.end %y : (i32);` | falling off the `end` of a construct |
//! | [Br] (`br`) | `wasm.br 1 %v : (i32);` | `br` |
//! | [BrTable] (`br_table`) | `wasm.br_table %i, %v [{ wasm.br 0 %v : (i32); }] default { wasm.br 1 %v : (i32); } : (i32, i32);` | `br_table` |
//! | [Unreachable] (`unreachable`) | `wasm.unreachable;` | `unreachable` |
//!
//! A body ends with [End], which leaves the construct with its operands as its results (for a
//! `loop`, falling off the end leaves the loop), or with an early exit: [Br], [BrTable],
//! `builtin.ret` or a tail call, or [Unreachable]. [Br] names a label: 0 is the innermost
//! enclosing `wasm.block`, `wasm.loop` or `wasm.if`, and each further level the next one of those
//! three; no other operation counts as a level, and each operation a branch crosses must forward
//! early exits (`scf.if` does, and is no label). A branch to a block or an `if` leaves it with the
//! values of its results; a branch to a loop goes back to the top of its body with the values as
//! the new arguments of its entry block. A label that would name the function itself is an error:
//! this IR leaves a function with `builtin.ret`. A `wasm.if` with no `else` stands for one whose
//! second region only passes its operands through, which the text leaves out.
//!
//! `br_if` is not an operation. It is an `if` whose first region holds a `br` and whose second
//! passes through (a later stage's emitter writes that shape as `br_if`):
//!
//! ```text,ignore
//! wasm.block %a {
//! ^body(%v: i32):
//!     %w = wasm.if %c, %v then {
//!     ^taken(%x: i32):
//!         wasm.br 1 %x : (i32);      // label 1: the block; label 0 is the `if`
//!     } : (i32, i32) -> (i32);
//!     wasm.end %w : (i32);
//! } : (i32) -> (i32);
//! ```
//!
//! `br_table` is a terminator with one region for each entry of the table and one for the default,
//! each holding one `br` of the values of the table: a branch states one target, and the table
//! many, so each region holds the single-target branch that every analysis knows. It never falls
//! through. Because a body may end in it, in a `unreachable` or in a `builtin.ret`, which are no
//! `end`, the results of that construct are not known to the analyses that follow the values: they
//! stay correct, and are less precise.
//!
//! [BlockOp], [Loop] and [If] carry `PropagateControlFlowBreak` and are `RegionBranchOpInterface`
//! operations; each states the edge to its parent that its body falls off its end by, which is
//! also where a nested branch that names it arrives, and a [Br] states its own edge, to the
//! results of the block or the `if` it names or to the entry block of the loop. [Loop] is a
//! `LoopLikeOpInterface` whose one region is both header and body, and allows a continue from its
//! header. The effects of the three are those of what they hold. The indirect calls above go
//! through a table.
//!
//! **What is not here.** Garbage-collected types and their instructions, exceptions, SIMD,
//! threads and atomics, stack switching, and the Component Model.

#![no_std]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(ptr_metadata)]
#![feature(specialization)]
#![allow(incomplete_features)]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod attributes;
mod builders;
mod effects;
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
    effects::{GlobalsResource, LinearMemoryResource, TablesResource},
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
