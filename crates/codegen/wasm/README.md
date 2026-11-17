# runic-codegen-wasm

The WebAssembly backend of Runic: the lowering of generic Runic IR (the dialects `arith`, `cf`,
`mem`, `scf`, `ub` and `builtin`) to _Wasm form_, the emitter that makes a binary core
WebAssembly module of lowered IR, and the backend `wasm` of the compile harness that runs both.

```sh
runic-opt program.runic --target wasm32-unknown-unknown -o program.core.wasm
wasmtime run --invoke main program.core.wasm 1 2
```

The lowering alone, as passes (its result is IR, checked as text):

```rust
let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
runic_codegen_wasm::add_lowering_passes(passes.op_pass_manager_mut())?;
passes.run(module.as_operation_ref())?;
```

or, in the tool (`cargo make runic-opt` builds it with the crate):

```sh
runic-opt program.runic -p 'cfg-to-scf{exits=in-place},wasm-lower-frames,convert-to-wasm,wasm-verify-form'
```

## The backend

`WasmBackend` is the backend named `wasm` of the compile harness (`runic-compile`), for the target
`wasm32-unknown-unknown`. `runic-opt`, built with its feature `wasm`, registers it:

```sh
runic-opt program.runic --target wasm32-unknown-unknown -o program.core.wasm
runic-opt program.runic --target wasm32-unknown-unknown --stop-after wat
runic-opt program.runic --target wasm32-unknown-unknown --emit wat=program.wat --emit module=program.core.wasm
runic-opt program.runic --target wasm32-unknown-unknown --wasm-stack-size 65536 -o program.core.wasm
```

- **Data layout**: the C ABI of wasm32: little-endian, pointers and `isize` of 32 bits, `i8`,
  `i16`, `i32`, `i64` and `i128` aligned to 1, 2, 4, 8 and 16 bytes, `f32` to 4 and `f64` to 8.
  A module that states no `data_layout` takes this one.
- **Passes**: `cfg-to-scf{exits=in-place}`, `wasm-lower-frames`, `convert-to-wasm`, then `cse` and
  `dce` when the optimization level is above `0`, then `wasm-verify-form`. `cse` and `dce` remove
  repeated and dead operations; they do not remove the unused results of a `wasm.if` or a
  `wasm.loop` (the flags `cfg-to-scf` leaves), which the emitter drops.
- **Checkpoints**, in order: `wasm.wat` (alias `wat`, a `.wat` file), the text of the module, and
  `wasm.module` (alias `module`, a `.core.wasm` file), the binary, which is the default stop point.
  Both are the same bytes, emitted once and validated.
- **Option**: `--wasm-stack-size <BYTES>`, the size of the shadow stack at the start of linear
  memory and the initial value of `__stack_pointer`; a positive multiple of 16, 1 MiB by default.
  It is checked whether or not the unit uses the stack.
- When the request asks for no debug information (`DebugInfo::None`, which a driver built on
  `runic-driver` spells `-g=none`), the module has no `name` section; it is named after the unit
  otherwise, and each entity after its link name. `runic-opt` has no flag for debug information:
  its requests take the default, so its modules always have a `name` section.

`runic-opt` compiles at `-O0`; a compiler built on the driver crate passes `-O`. What the emitter
cannot emit fails the compilation with a diagnostic that names the operation. The programs of
`tests/run` are compiled this way and run with `wasmtime` (`cargo make test-run`).

## The emitter

`emit::emit_module` takes a `builtin.world`, a `builtin.component` or a `builtin.module` in Wasm
form and returns the bytes of a validated core module. It first builds a _module description_
(`emit::desc::ModuleDesc`): the types, imports, functions, tables, memories, globals, exports,
element segments and data of the module, and the encoded body of each function. Nothing in the
description is an index: every reference from a body, a constant expression or data to another
entity (a function, a global, a table, a type, a data address, a table slot) is written as a
placeholder with a recorded _fix-up_, which says what it names and how it is encoded.

The layout then places the stack, the data and the table slots, and the complete-module writer
assigns the index spaces (imports first, then definitions, in the order the IR has them),
resolves every fix-up to its index, address or slot, and writes the module with `wasm-encoder`,
with three custom sections: `name` (left out when no debug information is asked for), `producers`
(the tool and its version, as `processed-by`) and `target_features` (the features the module uses,
when it uses any). The result is checked by `wasmparser` under the
feature baseline before it is returned: a module that does not validate is a defect of the
backend, reported with the validator's message and offset, and is never delivered.

### Where values wait

Code is written in the order of the IR; nothing is moved. In each block, a value used once, by the
next operation, as the operand that operation pushes last at that point, stays on the Wasm stack
(the operations form expression trees). A constant (`wasm.const`, `wasm.constf`, `wasm.ref_null`,
`wasm.data_addr`, `wasm.func_index`) is written again at each use. Any other value is set into a
local after its instruction, or kept with `local.tee` when it also stays on the stack, and read
with `local.get` at each use; an unused result is dropped. Locals are reused by type once a value
is dead. This function:

```
builtin.function public extern("C") @f(%x: i32, %y: i32) -> i32 {
    %s = wasm.add %x, %y;
    %k = wasm.const 3 : i32;
    %p = wasm.mul %s, %k;
    %d = wasm.sub %p, %s;
    %r = wasm.add %d, %k;
    builtin.ret %r : (i32);
} attributes { wasm.export_name = "f" };
```

is emitted as (`--stop-after wat`):

```
(func $m::f (;0;) (type 0) (param i32 i32) (result i32)
  (local i32)
  local.get 0
  local.get 1
  i32.add
  local.set 2
  local.get 2
  i32.const 3
  i32.mul
  local.get 2
  i32.sub
  i32.const 3
  i32.add
)
```

`%s` has two users, so it waits in a local; `%p` and `%d` stay on the stack; `%k` is written at
both of its uses. A structured construct (`wasm.block`, `wasm.loop`, `wasm.if`) takes its inputs
from the stack, a value used inside a loop that was defined outside it keeps its local until the
loop ends, and an `if` whose region is one `br` with no values is written as `br_if`.

### One compilation, one module

Every `builtin.module` under the root, at any depth, is part of the one module emitted: a nested
module is a namespace, not a separate binary. The modules are linked by link name (a symbol's
`link_name`, or its absolute path, such as `m::f`, when it has none), as a linker would link
the objects of one program:

- two definitions that other modules may name (`public` or `internal`) with one link name are an
  error naming both;
- a declaration is the definition of its link name when the unit has one, and its kind and type
  must agree with it;
- a declaration with no definition is an import (below).

### The layout

The lowering declares the memory, the stack pointer and the function table as imports from `env`
(`__linear_memory`, `__stack_pointer`, `__indirect_function_table`), and the layout makes each a
definition, as `wasm-ld --stack-first` does for one program. These three cannot be imported: the
layout defines every import from `env` with one of those fields, so a module of generic IR needs
no host for its memory, its stack or its table. A memory or a table that the unit declares under
a link name of its own (IR written in Wasm form) is an import like any other declaration (below),
and the layout places the data in that memory, whose minimum must hold it:

```
0 ............ stack size   the shadow stack, growing down; `__stack_pointer` starts at the
                            stack size (`--wasm-stack-size`, 1 MiB by default), so that an
                            overflow traps instead of overwriting data
stack size ... data end     every builtin.global_variable, each aligned by its type: those with
                            no section first, then each section's members together
(fixed)                     each builtin.segment at its own offset, above the stack and outside
                            the data
.............. heap base    the end of the data or of the highest segment, whichever is higher,
                            rounded up to 16
```

With no frame in the unit there is no stack, and the data starts at 1024, so that no data has the
address 0. The heap base is the end of everything static, so a heap that starts there never
overwrites a segment. A segment with no bytes is left out. The memory has the pages that reach the
heap base as its minimum and no maximum, and is exported as `memory`. Data whose bytes are all zero
has an address and no segment. The table gives a slot, from 1, to every function a function value
names; slot 0 stays empty, so that a call through a null function pointer traps, and the table is
exactly as long as its slots. A `wasm.elem` into that table is an error: a unit that fills
`__indirect_function_table` itself must define it (and cannot then take slots with
`wasm.func_index`). A table the unit defines is filled by its `wasm.elem`s as written.

The linker-defined data symbols are declarations of `builtin.global_variable` with no definition:
`__data_end` (the end of the laid-out data), `__heap_base` (the heap base above), and
`__start_<s>` and `__stop_<s>` (the start of the section `s` and the end of its last member). Each
is an address with no bytes, and the declarations of one must agree in type, as a declaration of
data the unit defines must agree with its definition. Any other declared global variable with no
definition is an error: a core module cannot import data. A global variable named by
`wasm.memory_init` or `wasm.data_drop` is a passive segment, with no address.

### Imports and exports

A function, a global, a table or a memory that is declared and not defined in the unit is an
import, from the module `wasm.import_module` names (`env` when nothing does), with its link name
as the field. Only what `wasm.export_name` names is exported, under each of its names (a symbol
being `public` exports nothing); an import may be exported, and so may an alias, as its target.
Two exports of one name are an error naming both. `wasmtime run --preload env=env.core.wasm`
satisfies the imports of a module from another emitted module.

### What a data initializer may hold

A `builtin.global_variable`'s initializer is generic code that nothing lowers, ending in
`builtin.yield`. The emitter computes the yielded value without changing the IR: a constant
(`arith.constant`, `arith.constantf`, `builtin.bytes`), the address of another global variable
(`builtin.global_symbol`), the table slot of a function (`builtin.function_ref`), or any operation
whose fold gives a constant from those (an `arith.add` of two constants, say). The value becomes
bytes by the global's type under the data layout: an integer little-endian in `size_of` bytes, a
float in its 4 or 8 bytes, bytes as they are (`builtin.bytes "0a0b"` is written most significant
first, so its bytes in memory are `0b 0a`), an address or a slot in 4 bytes. Anything else is an
error naming the operation and the global. A function value in data is read in Wasm form
(`wasm.load ... : i32`): `mem.load` cannot read one, as a function type has no layout.

### The feature baseline

The emitter uses, and the validator accepts, the features `wasm32-unknown-unknown` has by default
in rustc and tail calls: mutable globals, sign extension, saturating float-to-integer conversions,
bulk memory, multi-value, reference types and tail calls. A module that would need anything else
is an error naming the operation before anything is written: a second memory (multiple
memories), a `wasm.tag` (exception handling), and a `wasm.global` initializer that computes
(`wasm.add`, `wasm.sub`, `wasm.mul`: extended constant expressions) or reads a global the module
defines (the constant expressions of GC). A global's initializer is one instruction:
`wasm.const`, `wasm.constf`, `wasm.ref_null`, `wasm.ref_func`, `wasm.global_get` of an import, or
the address or slot `wasm.data_addr` and `wasm.func_index` give, written as an `i32.const`.

### What is not emitted yet

Relocatable objects for `wasm-ld`; integers wider than 64 bits; components and `wasm32-wasip2` or
`wasip3`, and the entities a `builtin.component` or a `builtin.world` holds outside every module
(an error); a choice of the feature set; importing `__linear_memory`, `__stack_pointer` or
`__indirect_function_table` (the layout defines them); exporting data; start functions and constructors; removing functions nothing reaches; DWARF. The
unused results of a construct are dropped, not removed.

## Wasm form

A function in Wasm form is a `builtin.function` whose body is made of `wasm` operations
(`runic-dialect-wasm`, which mirror WebAssembly's instructions in SSA form) and of five builtin
operations: `builtin.call`, `builtin.tail_call`, `builtin.ret`, `builtin.load_local` and
`builtin.store_local`. In it:

- every value, every parameter and result of the signature, and every local has a Wasm value type:
  `i32`, `i64`, `f32`, `f64`, `!wasm.funcref` or `!wasm.externref`;
- control flow is structured: every region holds one block;
- the calling convention is `"wasm"` or `"C"`.

Module-level operations (`builtin.global_variable`, `wasm.global`, `wasm.memory`, ...) are not
function bodies and are left as they are. `wasm-verify-form` checks exactly this.

## Carried form

Each generic type is _carried_ in a Wasm value type, and wherever a value exists the bits of its
carrier above its width are zero:

| generic type | carried in |
|---|---|
| `i1` to `i31` | `i32`, zero-extended |
| `i32` | `i32` |
| `i33` to `i63` | `i64`, zero-extended |
| `i64` | `i64` |
| `f32`, `f64` | `f32`, `f64` |
| `isize` | `i32` (the data layout gives `isize` 32 bits) |
| a pointer in address space 0, typed or opaque | `i32`, an address in the linear memory |
| a value of a function type | `i32`, an index in the indirect-function table |
| `!wasm.funcref`, `!wasm.externref` | themselves |
| an integer wider than 64 bits, an aggregate, a pointer in another address space | nothing: reported |

Every pattern keeps that invariant. A function value is the index of the function in
`@__indirect_function_table`: `builtin.function_ref @f` is `wasm.func_index @f`, and an indirect
call goes through the table. `wasm.func_index` names the function by the path it was written with,
an alias or a function of another module included: resolving that path to an index is left to the
emitter and the linker.

Where `arith` traps, the lowering traps: a division by zero, and the minimum value divided by -1
in `arith.sdiv` and `arith.sdivrem` at every width. Below the carrier's width the sign-extended
`wasm.div_s` does not trap on it, so the quotient is tested and a `wasm.if` traps when it does not
fit in `N` bits. `arith.srem` of the minimum by -1 is 0 and does not trap. Where `arith` leaves a
result undefined (a shift by `N` or more, an unchecked overflow, poison), any value that keeps
carried form is given.

## The calling conventions

Both are lowered; any other `extern("...")` is an error naming it, on a declaration that nothing
calls too.

- `extern("wasm")`: both sides of a call rely on carried form. Nothing is masked or extended, so
  a public `extern("wasm")` function trusts every caller, a host or another module included, to
  pass its narrow parameters in carried form; a function called by code that does not keep it is
  `extern("C")`.
- `extern("C")`: nothing is assumed that the signature does not say. A function masks each narrow
  parameter that is not marked `zext` at its entry, and sign-extends each narrow result marked
  `sext` at each return. A call sign-extends each argument whose parameter is narrow and marked
  `sext`, and masks each narrow result that is not marked `zext`. A tail call and an indirect call
  follow the same rules.

## The well-known symbols

Lowered code names three symbols. Each is found by name and kind in the `builtin.module` that
directly holds the function whose code names it, or declared there as a private import from `env`
whose `link_name` is its own name. A module nested in another has its own: nothing is declared in
an outer module on behalf of an inner one, and an inner module's own definition is used for its
code.

| symbol | operation | link name | declared by |
|---|---|---|---|
| `@__linear_memory` | `wasm.memory`, limits 0 | `__linear_memory` | `convert-to-wasm`, when an operation accesses memory |
| `@__indirect_function_table` | `wasm.table` of `!wasm.funcref`, limits 0 | `__indirect_function_table` | `convert-to-wasm`, when the module makes an indirect call |
| `@__stack_pointer` | `wasm.global` of type `mut i32` | `__stack_pointer` | `wasm-lower-frames`, when a function has a `mem.alloca` |

A module that holds a symbol of one of these names of another kind is an error naming both. Only
the symbols the lowering itself needs are declared: IR that already uses `wasm` operations naming
one of them (`wasm.load @__linear_memory`, say) declares it itself.

## The passes

The lowering needs a data layout in scope that is little-endian with 32-bit pointers and `isize`:
`data_layout = #builtin.data_layout<little, ptr = 32>` on the module, or a `--target`. The passes
run in this order (`add_lowering_passes` adds them), on a `builtin.module`. The examples take this
function:

```
builtin.function public extern("C") @f(%c: i1, %x: i8) -> i32 {
    %p = mem.alloca i32;
    %w = arith.zext %x <{ ty = i32 }>;
    mem.store %w, %p;
    cf.cond_br %c ^yes, ^no : (i1);
^yes:
    %v = mem.load %p;
    builtin.ret %v : (i32);
^no:
    builtin.tail_call @g(%w) : extern("C") (i32) -> i32;
};
```

**`cfg-to-scf{exits=in-place}`** (`runic-dialect-scf`) structures control flow. Each `builtin.ret`
and `builtin.tail_call` stays where it is, in an arm of an `scf.if`, an `scf.index_switch` or after
a loop, and the function ends in `ub.unreachable`:

```
%2 = mem.alloca i32;
%3 = arith.zext %1 <{ ty = i32 }>;
mem.store %3, %2;
scf.if %0 then {
    %4 = mem.load %2;
    builtin.ret %4 : (i32);
} else {
    builtin.tail_call @g(%3) : extern("C") (i32) -> i32;
} : (i1) -> ();
ub.unreachable;
```

**`wasm-lower-frames`** gives every `mem.alloca` a slot in its function's frame on the shadow stack:
the prologue lowers `@__stack_pointer` by the size of the frame (rounded up to the 16-byte stack
alignment, or to the largest alignment of a slot), the slot's address is the new stack pointer plus the slot's offset, and every
return and tail call restores the stack pointer first. The address is a
`builtin.unrealized_conversion_cast` to the alloca's type, which `convert-to-wasm` removes:

```
%6 = wasm.global_get @__stack_pointer : i32;
%7 = wasm.const 16 : i32;
%8 = wasm.sub %6, %7;
wasm.global_set @__stack_pointer, %8;
%9 = builtin.unrealized_conversion_cast %8 : i32 to ptr<i32>;
%3 = arith.zext %1 <{ ty = i32 }>;
mem.store %3, %9;
scf.if %0 then {
    %4 = mem.load %9;
    wasm.global_set @__stack_pointer, %6;
    builtin.ret %4 : (i32);
} else {
    wasm.global_set @__stack_pointer, %6;
    builtin.tail_call @g(%3) : extern("C") (i32) -> i32;
} : (i1) -> ();
ub.unreachable;
```

**`convert-to-wasm`** checks every module under the operation it runs on before it changes any,
so a failed run leaves the IR as it was: every diagnostic the pass gives on IR that verifies comes
from that check. Then it is a full dialect conversion of each `builtin.module`: the types become
their carriers, the operations of `arith`, `cf.select`, `cf.assert`, `mem`, `scf` and `ub` become
`wasm` operations, and the calling convention is applied. The parameters are masked at the entry of
`@f` (`extern("C")`), the `scf.if` is a `wasm.if`, and the memory is declared:

```
wasm.memory private declaration @__linear_memory : 0 attributes { link_name = #builtin.string<"__linear_memory"> };
builtin.function public extern("C") @f(%0: i32, %1: i32) -> i32 {
    %2 = wasm.const 1 : i32;
    %3 = wasm.and %0, %2;
    %4 = wasm.const 255 : i32;
    %5 = wasm.and %1, %4;
    %6 = wasm.global_get @__stack_pointer : i32;
    %7 = wasm.const 16 : i32;
    %8 = wasm.sub %6, %7;
    wasm.global_set @__stack_pointer, %8;
    wasm.store @__linear_memory, %8, %5;
    wasm.if %3 then {
        %9 = wasm.load @__linear_memory, %8 : i32;
        wasm.global_set @__stack_pointer, %6;
        builtin.ret %9 : (i32);
    } else {
        wasm.global_set @__stack_pointer, %6;
        builtin.tail_call @g(%5) : extern("C") (i32) -> i32;
    } : (i32) -> ();
    wasm.unreachable;
};
```

**`wasm-verify-form`** checks that every function is in Wasm form and changes nothing. Run on the
first function above, it reports the first thing that is not:

```
error: `wasm-verify-form`: the function `@f` is not in Wasm form
  | parameter 0 of `@f` has type `i1`, which is not a Wasm value type; `convert-to-wasm` should have converted it
```

The default optimization pipeline of a compiler (`canonicalizer,cse,sccp,canonicalizer,dce`) can run
in front of the four: the whole programs of `tests/lit/codegen/wasm/programs*.runic` are lowered
both ways.

## What is not lowered, and why

An operation that `convert-to-wasm` does not lower is a diagnostic that names the operation, says
what is unsupported and names the pass; nothing a user can write is a panic. Every operation of
`arith`, `cf`, `mem`, `scf` and `ub`, and the function-level operations of `builtin`, has a pattern
except these, which are diagnostics by design (the unit test `tests::coverage` fails for an
operation that is neither lowered nor on this list, and for a row of this table that disagrees with
it):

| operation | why it is not lowered | what the diagnostic says |
|---|---|---|
| `mem.alloca` | `wasm-lower-frames` gives each alloca a slot on the shadow stack first | `` `convert-to-wasm` does not lower `mem.alloca`: run `wasm-lower-frames` first `` |
| `cf.br` | Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first | ``unstructured control flow reached `convert-to-wasm`: run `cfg-to-scf{exits=in-place}` first`` |
| `cf.cond_br` | Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first | the same |
| `cf.switch` | Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first | the same |
| `scf.pending_exit` | `cfg-to-scf` makes it while it runs and removes it: it never verifies | `` `scf.pending_exit` is internal to `cfg-to-scf` and never appears in IR `` |

`scf.yield` and `scf.condition` are lowered with the `scf.if`, `scf.while` or `scf.index_switch`
they end.

The limits that are not per operation:

- **Integers wider than 64 bits.** A value of type `i65` or wider has no carrier: the operation, or
  the signature, that holds one is reported ("integers wider than 64 bits are not lowered yet").
- **Aggregates as values.** A `struct`, `array` or other aggregate is not one Wasm value: a function
  cannot take or return one, and `mem.load` and `mem.store` move a value that fits one Wasm access,
  so an aggregate is loaded and stored field by field (`mem.gep` to each field).
- **Address spaces other than 0.** Only address space 0 is the linear memory.
- **`mem.alloca` in a loop.** Each alloca has one slot in the function's frame, so one in a loop
  would need a slot per iteration: `wasm-lower-frames` reports it.
- **Exits from inside a loop** (`cfg-to-scf{exits=in-place}`) are not left in place: the loop
  carries the exit's operands and a selector out, and the exit is dispatched after the loop.
- **Assertion messages are dropped.** `cf.assert` becomes a `wasm.if` that traps on a false
  condition; its message is not kept.
- **32-bit little-endian only.** The data layout must be little-endian with 32-bit pointers and
  `isize`: 64-bit memories (memory64) are not modelled, and a missing or different layout is an
  error.
- **A value of a function type is not stored with `mem.store`**: the `mem` verifier rejects it,
  as a function type has no layout. A function value reaches memory as an `i32` that the frontend
  converts itself, or travels in locals and arguments.
- **An `scf.break` that names a `wasm.loop` written before the lowering is an error**: leaving such
  a loop with its results has no `wasm.br` (a `wasm.br` to a loop goes back to its top).
- **A global variable's initializer is data**, and is not converted: `convert-to-wasm` and
  `wasm-verify-form` leave the regions of `builtin.global_variable` and `wasm.global` as they are.
- **A `builtin.unrealized_conversion_cast` written before the lowering to a narrow integer** is an
  error: nothing says the bits of its input above the width are clear. `arith.trunc` makes one.
- **An `isize` constant outside `-2^31 ..= 2^32 - 1`** (an `arith.constant` or a
  `builtin.ret_imm`) is an error naming it and its value: the target's `isize` has 32 bits.
- **A `wasm.br` the frontend wrote** inside an `scf` construct keeps naming the construct it named:
  the lowering gives it the label that counts the constructs it put between them.
- **The passes run on modules**: run on an operation that is not a `builtin.module` and holds none,
  each of the three passes of this crate says so.
- **Dead code is left.** Nothing in the lowering removes it: the slot arithmetic of an alloca
  whose pointer is never used stays.
- **`ub.poison` is a zero of its type and `ub.unreachable` is a trap**: both refine the undefined
  behaviour they stand for.

The unit tests of the crate evaluate the lowered numeric code, structured control and whole
programs with an evaluator written from the WebAssembly specification (release 3.0), and run the
module the emitter makes of the same code under `wasmi`, which must agree with it. The lit tests of
`tests/lit/codegen/wasm` check the lowered text, those of `tests/lit/codegen/wasm/emit` the text of
emitted modules, and the programs of `tests/run/wasm` run under `wasmtime` (and under `wasmi` in
the unit tests, `tests::emit::programs`). `tests::emit::coverage` requires every operation of the
`wasm` dialect to be in a module that is emitted and validates, except `wasm.tag`, which is an
error by design.
