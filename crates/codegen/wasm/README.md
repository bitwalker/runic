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
runic-opt program.runic --target wasm32-unknown-unknown --emit wat=program.wat -o program.core.wasm
runic-opt program.runic --target wasm32-unknown-unknown --stop-after object -o program.o
runic-opt program.runic --target wasm32-unknown-unknown --wasm-stack-size 65536 -o program.core.wasm
```

- **Data layout**: the C ABI of wasm32: little-endian, pointers and `isize` of 32 bits, `i8`,
  `i16`, `i32`, `i64` and `i128` aligned to 1, 2, 4, 8 and 16 bytes, `f32` to 4 and `f64` to 8,
  and a function value of 32 bits aligned to 4 (`fn = 32`: its index in the function table). A
  module that states no `data_layout` takes this one.
- **Passes**: `cfg-to-scf{exits=in-place}`, `wasm-lower-frames`, `convert-to-wasm`, then `cse` and
  `dce` when the optimization level is above `0`, then `wasm-verify-form`. `cse` and `dce` remove
  repeated and dead operations; they do not remove the unused results of a `wasm.if` or a
  `wasm.loop` (the flags `cfg-to-scf` leaves), which the emitter drops.
- **Checkpoints**, in order: `wasm.object` (alias `object`, an `.o` file), a relocatable object
  for `wasm-ld`, written only when the request stops there or asks for an output there
  (`--stop-after object`, `--emit object=FILE`); the object is collected from the world on its own,
  so a request for the module alone is never refused for what an object cannot express, and
  `--wasm-stack-size` does not change it (the linker sets the stack); `wasm.wat` (alias `wat`, a `.wat` file), the text of the module, and
  `wasm.module` (alias `module`, a `.core.wasm` file), the binary, which is the default stop point.
  Both are the same bytes, emitted once and validated.
- **Option**: `--wasm-stack-size <BYTES>`, the size of the shadow stack at the start of linear
  memory and the initial value of `__stack_pointer`; a positive multiple of 16, 1 MiB by default.
  It is checked whatever is asked for, whether or not the unit uses the stack: a value that is not
  valid is the same error when the request asks for an object, which the option does not change.
- When the request asks for no debug information (`DebugInfo::None`, which a driver built on
  `runic-driver` spells `-g=none`), the module has no `name` section; it is named after the unit
  otherwise, and each entity after its link name. `runic-opt` has no flag for debug information:
  its requests take the default, so its modules always have a `name` section.

`runic-opt` compiles at `-O0`; a compiler built on the driver crate passes `-O`. What the emitter
cannot emit fails the compilation with a diagnostic that names the operation. The programs of
`tests/run` are compiled this way and run with `wasmtime` (`cargo make test-run`).

## The emitter

`emit::emit_module` takes a `builtin.world`, a `builtin.component` or a `builtin.module` in Wasm
form and returns the bytes of a validated core module; `emit::emit_object` takes the same and
returns a relocatable object ([Relocatable objects](#relocatable-objects)). Both first build a
_module description_ (`emit::desc::ModuleDesc`): the types, imports, functions, tables, memories,
globals, exports, element segments and data of the module, and the encoded body of each function.
Nothing in the description is an index: every reference from a body, a constant expression or data
to another entity (a function, a global, a table, a type, a data address, a table slot) is written
as a placeholder with a recorded _fix-up_, which says what it names and how it is encoded. The
rest of this section is the complete module; the object has a section of its own.

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

A function definition written with `wasm.start = #builtin.unit` is the unit's _start function_, at
most one per compilation (a second is an error naming both). A complete module exports it as
`_initialize`, the reactor convention of WASI, and has no start section. An embedder calls
`_initialize` once, after it instantiates the module; `wasmtime` does so itself. A unit that
exports another entity under the name `_initialize` is the duplicate-export error. The program
`tests/run/wasm/start.runic` has a start function that adds 35 to a global of 7:

```sh
runic-opt tests/run/wasm/start.runic --target wasm32-unknown-unknown -o start.core.wasm
wasmtime run --invoke read start.core.wasm 0 2>/dev/null
```

prints 42. In an object the start function is an init function instead
([The start function](#the-start-function)).

### What a data initializer may hold

A `builtin.global_variable`'s initializer is generic code that nothing lowers, ending in
`builtin.yield`; it holds no operation with effects (the verifier of `builtin.global_variable`
refuses a `mem.load` there, say). The emitter gives the global the bytes that the `mem` dialect's
evaluator, `runic_dialect_mem::evaluate_initializer`, computes from it under the data layout of
the module, without changing the IR: bytes, and relocations where they hold an address or a
function value, which the layout resolves to the address of a global variable or the slot of a
function in the table. The yielded value is made of:

| part | written as | in memory |
|---|---|---|
| a constant | `arith.constant`, `arith.constantf`, `builtin.bytes`, or an operation whose fold gives a constant from constants (`arith.add` of two, `mem.sizeof`) | an integer little-endian in `size_of` its type bytes, a float in its 4 or 8, bytes as they are (`builtin.bytes "0a0b"` is written most significant first, so its bytes in memory are `0b 0a`) |
| a zero | `builtin.zero` of any type with a layout | `size_of` zero bytes; a global that is wholly zero has an address and no segment, whatever its size |
| an address | `builtin.global_symbol @g`, then `mem.gep` with a constant index and path, `mem.ptrcast`, `mem.ptrtoint` to `isize` and `mem.inttoptr` back | 4 bytes, the address of `@g` plus the offset the geps add |
| a function | `builtin.function_ref @f`, `@f` a function, a declaration or an alias | 4 bytes, the slot of `@f` in `__indirect_function_table` (`fn = 32`, below) |
| an aggregate | `builtin.aggregate` of any of these, nested, into a `struct` or an `array` | each element at the offset the layout gives it; padding is zero |

What is refused, each an error that names the global and the operation:

- **arithmetic on an address**, such as `arith.add` of `mem.ptrtoint @g` and 4 (``arithmetic on an
  address cannot be placed: write the offset with `mem.gep` ``): an address is only known as a
  relocation, which has a target and an addend, so an offset is written with `mem.gep`;
- an operation that **does not fold** to a constant (`arith.udiv` by 0: `` `arith.udiv` does not
  fold to a constant ``), or that computes with a function value;
- a value that **does not fit** its type (an `isize` constant of 2^40: `` `arith.constant` gives
  1099511627776, which does not fit `isize` of 32 bits ``), and an address cast to an integer of
  another width than the pointer's;
- a type with no layout;
- a passive segment (a global named by `wasm.memory_init` or `wasm.data_drop`) that holds an
  address or a function value: what `memory.init` copies is not relocated.

A dispatch table is an array of such structs, or, as here, structs in a section of their own that
the layout places together, which a loop walks from `__start_dispatch` to `__stop_dispatch` (the
program `tests/run/wasm/dispatch-table.runic` runs a larger one):

```
builtin.module public @table {
    builtin.function public extern("wasm") @double(%x: i64) -> i64 {
        %r = arith.add %x, %x;
        builtin.ret %r : (i64);
    };
    builtin.function public extern("wasm") @square(%x: i64) -> i64 {
        %r = arith.mul %x, %x;
        builtin.ret %r : (i64);
    };
    builtin.global_variable private @entry_double : struct<i32, i32, (i64) -> i64> {
        %id = arith.constant 20 : i32;
        %arity = arith.constant 1 : i32;
        %f = builtin.function_ref @double : (i64) -> i64;
        %e = builtin.aggregate %id, %arity, %f : struct<i32, i32, (i64) -> i64>;
        builtin.yield %e;
    } attributes { section = "dispatch" };
    builtin.global_variable private @entry_square : struct<i32, i32, (i64) -> i64> {
        %id = arith.constant 40 : i32;
        %arity = arith.constant 1 : i32;
        %f = builtin.function_ref @square : (i64) -> i64;
        %e = builtin.aggregate %id, %arity, %f : struct<i32, i32, (i64) -> i64>;
        builtin.yield %e;
    } attributes { section = "dispatch" };
    builtin.global_variable public @start : struct<i32, i32, (i64) -> i64> attributes { link_name = "__start_dispatch" };
    builtin.global_variable public @stop : struct<i32, i32, (i64) -> i64> attributes { link_name = "__stop_dispatch" };
    builtin.function public extern("C") @lookup(%id: i32, %x: i64) -> i64 {
        %s = builtin.global_symbol @start : ptr<struct<i32, i32, (i64) -> i64>>;
        %e = builtin.global_symbol @stop : ptr<struct<i32, i32, (i64) -> i64>>;
        %si = mem.ptrtoint %s;
        %ei = mem.ptrtoint %e;
        cf.br ^head(%si : isize);
    ^head(%at: isize):
        %done = arith.cmpi uge %at, %ei;
        cf.cond_br %done ^miss, ^body : (i1);
    ^body:
        %entry = mem.inttoptr %at : ptr<struct<i32, i32, (i64) -> i64>>;
        %idp = mem.gep %entry, [0];
        %entry_id = mem.load %idp;
        %found = arith.cmpi eq %entry_id, %id;
        cf.cond_br %found ^call, ^next : (i1);
    ^next:
        %size = mem.sizeof struct<i32, i32, (i64) -> i64>;
        %at2 = arith.add %at, %size;
        cf.br ^head(%at2 : isize);
    ^call:
        %fp = mem.gep %entry, [2];
        %f = mem.load %fp;
        %r = builtin.call_indirect %f(%x) : extern("wasm") (i64) -> i64;
        builtin.ret %r : (i64);
    ^miss:
        %minus_one = arith.constant -1 : i64;
        builtin.ret %minus_one : (i64);
    } attributes { wasm.export_name = "lookup" };
};
```

Each entry is 12 bytes: the id, the arity and the slot of the function. With no frame in the unit
the data starts at 1024, and the module holds

```
(table $__indirect_function_table (;0;) 3 3 funcref)
(elem (;0;) (i32.const 1) func $table::double $table::square)
(data $table::entry_double (;0;) (i32.const 1024) "\14\00\00\00\01\00\00\00\01\00\00\00")
(data $table::entry_square (;1;) (i32.const 1036) "(\00\00\00\01\00\00\00\02\00\00\00")
```

so that `wasmtime run --invoke lookup table.core.wasm 40 5` prints 25, and an id that is not in
the table gives -1.

### Function values in memory

The backend's data layout states `fn = 32`: a function value is 32 bits aligned to 4 in memory,
its index in `__indirect_function_table`, as its carrier is an `i32`. So a function type has a
layout: `mem.load` and `mem.store` move a function value as an `i32`, `mem.sizeof`,
`mem.alignof`, `mem.offsetof` and `mem.gep` measure and step through a struct that holds one, and
an initializer places one. A module that states its own data layout must state the same, or none:
the harness refuses a module whose layout is not the target's, and the lowering refuses any other
width (`` `convert-to-wasm` needs a function value of 32 bits, `fn = 32` in the data layout ``).

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

Integers wider than 64 bits; components and `wasm32-wasip2` or `wasip3`, and the entities a
`builtin.component` or a `builtin.world` holds outside every module (an error); a choice of the
feature set; importing `__linear_memory`, `__stack_pointer` or `__indirect_function_table` in a
complete module (the layout defines them); exporting data; more than one start function in a
unit; removing functions nothing reaches; DWARF. The unused results of a construct are dropped,
not removed. What a relocatable object lacks is listed with the object.

## Relocatable objects

A relocatable object is a file in the tool conventions' linking format (`Linking.md`): a core
module with a `linking` section, relocations, and no layout. The compiler does not place its data,
number its table slots or define its memory, stack pointer and function table; the linker
(`wasm-ld`, which Rust's toolchain carries as `rust-lld`) does, with the objects of a runtime
written in Rust. A frontend compiles to an object when it links the program with code it does not
compile itself: a runtime, a standard library, the host's functions. Where the complete module is
a whole program that needs a host, an object is a part that needs a linker.

The object is written only when it is asked for: at the checkpoint `wasm.object` (alias `object`,
extension `o`), which comes before `wasm.wat` and `wasm.module`. The request stops there, or asks
for an output there:

```sh
runic-opt program.runic --target wasm32-unknown-unknown --stop-after object -o program.o
runic-opt program.runic --target wasm32-unknown-unknown --emit object=program.o -o program.core.wasm
```

The second writes the object to `program.o` and the module, the default stop point, to `-o`
(without `-o`, the module's bytes go to standard output). One compilation gives one object, as it
gives one module, and the default stop point is still the module. A request that wants only the
module never writes an object, and so is never refused for what an object cannot express.
`--wasm-stack-size` does not change an object, since the linker sets the stack, but a value that
is not valid is the error it is for a module.

### Linking an object with Rust

A runtime is a Rust library whose `#[unsafe(no_mangle)] extern "C"` functions the object calls,
and which calls the object's. This unit, `scaled.runic`, defines `scaled` and calls `rust_offset`,
which it does not define:

```
builtin.module public @scaled {
    builtin.function public extern("C") @rust_offset(i32) -> i32 attributes { link_name = "rust_offset" };

    builtin.function public extern("C") @scaled(%x: i32) -> i32 {
        %k = arith.constant 3 : i32;
        %m = arith.mul %x, %k;
        %r = builtin.call @rust_offset(%m) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);
    } attributes { wasm.export_name = "scaled" };
};
```

and this runtime, `rt.rs`, defines it:

```rust
#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_offset(x: i32) -> i32 {
    x + 100
}
```

`rustc` links the object into a `cdylib` when the object is passed as a linker argument; it goes to
`rust-lld` after Rust's own objects and libraries:

```sh
runic-opt --target wasm32-unknown-unknown --stop-after object scaled.runic -o scaled.o
rustc --edition 2024 --target wasm32-unknown-unknown --crate-type cdylib -O rt.rs \
    -C link-arg=scaled.o -C link-arg=--fatal-warnings -o scaled.wasm
wasmtime run --invoke scaled scaled.wasm 5 2>/dev/null
```

The last command prints 115 (5 times 3, plus 100). The same link without `rustc`, with the linker
of the toolchain's sysroot (`rustc --print sysroot`; its `lib/rustlib/<host>/bin/rust-lld`) and the
runtime compiled to an object of its own:

```sh
rustc --target wasm32-unknown-unknown -O --crate-type cdylib --emit obj=rt.o rt.rs
rust-lld -flavor wasm --no-entry --stack-first -z stack-size=65536 --fatal-warnings \
    rt.o scaled.o -o direct.wasm
wasmtime run --invoke scaled direct.wasm 5 2>/dev/null
```

prints 115 as well. `-O` is there because it turns off Rust's overflow checks: without it, the
`x + 100` of `rt.o` calls `core`'s overflow panic, which only the `rustc` route links, and the link
stops with `undefined symbol: core::panicking::panic_const::panic_const_add_overflow`.
`-C overflow-checks=off` in place of `-O` links too. The runtime may use `std`, as `tests/run/link/rt/alloc.rs` does; then only the
`rustc` route is practical, since the standard library's objects are linked by `rustc`. Whatever
the route:

- **Pass `--fatal-warnings`.** `wasm-ld` only warns when a call has a signature that differs from
  the function's definition (a Rust `f32` where the IR has an `f64`), and replaces the call with
  one that traps. The option turns the warning into a failed link.
- **The linker exports what the object marks.** A function is exported when it has
  `wasm.export_name` (above), and a Wasm global that has one is too; the memory is exported as
  `memory`. A `public` function with no export name is exported by neither, and
  `-C link-arg=--export=<name>` exports it. **Export names must be unique across the whole link**,
  the runtime's own exports included (a Rust `cdylib` exports each `#[no_mangle]` function under
  its name): the linker does not check them, even with `--fatal-warnings`, and two exports of one
  name link into a module that does not validate (`wasmtime` refuses it: ``duplicate export name
  `shown` already defined``). The backend sees one object and cannot tell; a downstream compiler
  keeps its export names unique, and validates the linked module (`tests/run/link/duplicate-export.runic`
  pins the linker's behaviour).
- **Objects that import one function agree on naming its host.** A declaration with
  `wasm.import_module` in one object and one without it in another are, to the linker, a host
  import and an undefined symbol of one name, and the link fails with `import module mismatch`.
  Within one unit the object check refuses the same disagreement.
- **A signed narrow integer is marked** `extend = #builtin.sext` ([The calling
  conventions](#the-calling-conventions)).
- **A runtime for `wasm32-unknown-unknown`** that has constructors, or that the object gives a
  start function, calls `__wasm_call_ctors` once ([The start function](#the-start-function)).
- The object's `producers` section says `processed-by: runic`; it has no `name` section, and the
  linker names the output's functions from the symbols.

### Symbols

Every function, global, table and global variable of the unit is a symbol of the object, named by
its effective link name: the `link_name` attribute, or the absolute path of the entity in the unit
(`scaled::scaled`) when it has none. A Rust function has the name Rust gives it, so the IR that
meets Rust names it with `link_name`, as above. An export name is a different thing: it is the name
the linked module exports the entity under.

| in the IR | the symbol |
|---|---|
| a definition, `public` | global; the linker exports nothing for it unless the entity has an export name |
| a definition, `internal` | global and hidden: other objects of the link see it, the linker never exports it |
| a definition, `private` | local: two objects may each define a private `@helper` |
| a function, global or table the unit defines, with `wasm.export_name` (a function; a global or table only under its own link name) | exported; a `private` one is hidden instead of local, since the linker exports no local symbol. A declaration the unit does not define cannot be exported, nor can the memory ([What has no object form](#what-has-no-object-form)) |
| a `builtin.function_alias`, `public` or `internal` | a second function symbol of its target, under the alias's link name, global or hidden by its own visibility: another object calls the function by either name. A `private` alias has no symbol |
| a declaration with no definition in the unit, and no `wasm.import_module` | undefined: a definition in the runtime or another object resolves it, and the link fails (`undefined symbol: helper`) when nothing does |
| a declaration with `wasm.import_module` (any module, `env` included) | a host import: undefined, with an explicit name, so the linker passes it through as an import from that module with no option |
| a declaration of a global variable with no definition | an undefined data symbol: another object's data, or one the linker defines (`__heap_base`, `__data_end`, `__start_<section>`, `__stop_<section>`) |

The declarations of one import must agree on whether they name a host. The memory, the stack pointer
and the function table are not symbols of the unit: every object imports `env.__linear_memory`,
`env.__stack_pointer` and, when something needs it, `env.__indirect_function_table`, and the linker
defines them. A unit may not define any of them, nor declare another memory, nor import the memory
from a host it names, nor export it ([What has no object form](#what-has-no-object-form)).

The functions of a module `@rules` show the rules:

```
builtin.function public extern("C") @from_rust(i32) -> i32 attributes { link_name = "from_rust" };
builtin.function public extern("C") @from_host(i32) -> i32 attributes { link_name = "triple", wasm.import_module = "host" };
builtin.function public extern("C") @shown(%x: i32) -> i32 { ... } attributes { wasm.export_name = "shown" };
builtin.function internal extern("C") @shared(%x: i32) -> i32 { ... };
builtin.function private extern("C") @helper(%x: i32) -> i32 { ... };
builtin.function private extern("C") @hidden_export(%x: i32) -> i32 { ... } attributes { wasm.export_name = "hidden_export" };
```

and `llvm-readobj --symbols` of the sysroot (`lib/rustlib/<host>/bin/llvm-readobj`) prints, in the
order of the symbol table (functions, globals, tables, data, then aliases):

```
from_rust            FUNCTION  UNDEFINED                       ImportModule: env
triple               FUNCTION  EXPLICIT_NAME UNDEFINED         ImportModule: host, ImportName: triple
rules::shown         FUNCTION  EXPORTED
rules::shared        FUNCTION  VISIBILITY_HIDDEN
rules::helper        FUNCTION  BINDING_LOCAL
rules::hidden_export FUNCTION  EXPORTED VISIBILITY_HIDDEN
```

(shortened to the columns that matter). The global variables of the same module give the data
symbols `rules::plain` (global), `rules::zeros` and `rules::entry` (local).

### Data and segments

A global variable with no `section` has a segment of its own, named `.data.<link name>`, or
`.bss.<link name>` when its bytes are all zero and it holds no address; the linker leaves a `.bss`
segment out of the linked module's data. The object holds the zero bytes of a zeroed global all
the same, so that a symbol lies within its segment, and a zeroed megabyte is a megabyte of object,
as in LLVM's objects.

The global variables with one `section` are one segment of that name, in the order written, each
at a multiple of its alignment, with the **retain** flag. The linker drops every data segment that
no live symbol names, and a reference to `__start_dispatch` does not name the entries between the
bounds: without the flag the link fails with `undefined symbol: __start_dispatch`. A named section
exists to be found by name, so every one is retained. The linker puts the same-named segments of all
objects together, in the order of the command line, and defines `__start_<section>` and
`__stop_<section>` around them; `tests/run/link/dispatch.runic` and `dispatch-more.runic`
contribute entries of one dispatch table from two objects, which `rt/dispatch.rs` finds between
the bounds and calls through. The linker gives some names a meaning of its own, so a `section` may
not start with `.bss` (it writes no bytes for such a segment: its contents would link to zeros),
`.tdata` or `.tbss` (it takes them for thread-local data); and it defines `__start_<section>` and
`__stop_<section>` only for a name that is a C identifier (letters, digits and `_`, not starting
with a digit), so a unit that declares the bounds of `a.b` is refused. The segments of the global variables of that module (`@plain` with a value, `@zeros` all zero,
`@entry` in the section `entries`), from `wasm-tools dump`:

```
Segment { name: ".data.rules::plain", alignment: 2, flags: SegmentFlags(0x0) }
Segment { name: ".bss.rules::zeros", alignment: 2, flags: SegmentFlags(0x0) }
Segment { name: "entries", alignment: 2, flags: SegmentFlags(0x4) }
```

(`alignment` is the logarithm; 0x4 is the retain flag).

**The alignment caveat.** The linker aligns each object's contribution to a section on its own,
so entries of different alignments or sizes from different objects get padding between them, inside
the range from `__start_<section>` to `__stop_<section>`. The entries of a section should share one
alignment and have a size that is a multiple of it, and a reader that walks the section should step
by that size. One object that has three `i32` entries in `entries` (12 bytes, alignment 4) and one
that has an `i64` entry (8 bytes, alignment 8), linked with a runtime that returns
`&__stop_entries - &__start_entries`:

```
narrow.o then wide.o: 24
wide.o then narrow.o: 20
```

the first with four bytes of zero between the two. Both objects hold 20 bytes of entries.

### The start function

A function definition written with `wasm.start = #builtin.unit` is the unit's start function. It
takes no parameters and gives no results (the dialect checks that where the attribute is written),
and a unit has at most one. In an object it is registered with the linker as an init function at
the default priority (65535), exactly as a Rust or C constructor (`.init_array`) is. The linker
gathers every init function of the link into `__wasm_call_ctors`, in link order. A frontend that
has several things to do at start-up writes one function that does them in order; there is no
priority.

Who calls `__wasm_call_ctors` depends on the target:

| | what runs the init functions |
|---|---|
| `wasm32-unknown-unknown` | nothing, unless the runtime calls `__wasm_call_ctors` itself, once. When nothing in the link references it, the linker wraps every export so that it calls `__wasm_call_ctors` first, **on every call**, for Rust's constructors as for ours; a runtime that references the function turns the wrapping off |
| `wasm32-wasip1`, `-wasip2`, `-wasip3` (a library) | `_initialize`, which wasi-libc's `crt1-reactor.o` defines (a Rust `cdylib` is linked with it from the pinned nightly and from stable 1.100) and the embedder calls once after instantiating the module; `wasmtime` does so itself |
| a program | `_start` (or the component's run function), before `main` |

`tests/run/link/start.runic` shows the first row with a runtime that has a constructor of its own.
The object's start function adds 1 to a counter, Rust's constructor adds 1 to another, and the
host calls the runtime's exports in one instance. When the runtime's `init` export calls
`__wasm_call_ctors`, both have run once after any number of calls (the host reads 11); with a
runtime that does not, so that the exports are wrapped, the host reads 44 after four calls, and the
one `counts` call of a fresh instance reads 11. The same object linked on `wasm32-wasip1` (checked
with the pinned nightly and stable 1.100.0) has the export `_initialize`, and `wasmtime` runs it
when it instantiates the module: `counts` alone reads 11 without anyone calling `init`.

In a complete module the start function is exported as `_initialize` and the module has no start
section ([Imports and exports](#imports-and-exports)).

### What has no object form

The object check refuses, with a diagnostic that names the operation, what an object cannot say or
what the linker would not honour. It runs only when an object is asked for, and the same unit still
compiles to a complete module. Each row is a case of `tests/lit/codegen/wasm/object/refusals.runic`.

| construct | what the linker does | the diagnostic |
|---|---|---|
| `wasm.elem` | ignores the element segments of an object: it lays out the function table itself | a relocatable object cannot express `wasm.elem` @e |
| `builtin.segment` (data at a fixed address) | places every data segment of an object itself | a relocatable object cannot express a `builtin.segment` |
| passive data (a global variable named by `wasm.memory_init` or `wasm.data_drop`) | makes every data segment of an object active; there is no relocation for a segment index | a relocatable object cannot express passive data: `wasm.memory_init` names `m::bytes` as a passive segment |
| a `wasm.global` whose initializer names another entity (the address `wasm.data_addr` gives, the slot `wasm.func_index` gives) | an object has no relocation in the global section | a relocatable object cannot express `wasm.global` @g, whose initializer names another entity |
| a defined `wasm.memory` | defines the memory of the module it links; an object imports it as `env.__linear_memory` | a relocatable object cannot express a memory it defines, `wasm.memory` @mem |
| a defined `__indirect_function_table` | defines it itself and refuses an object that does | a relocatable object cannot define the function table `__indirect_function_table` |
| a defined `__stack_pointer` | defines it itself: the link fails with a duplicate symbol | a relocatable object cannot define the stack pointer `__stack_pointer` |
| `wasm.ref_func` | declares no function for `ref.func`, so the linked module does not validate | a relocatable object cannot express `wasm.ref_func` `@callee` |
| a function with two export names | marks a symbol exported under one name | a relocatable object cannot express the function `f` exported under two names, `one` and `two` |
| a global, table or memory exported under a name other than its link name | marks such a symbol exported only under its own name | a relocatable object cannot express the export `x` of the global `g` under another name |
| declarations of one import that disagree on `wasm.import_module` | cannot tell a host import from an undefined symbol | a relocatable object cannot express the import `g`: its declarations disagree on whether it names its host |
| a global variable in a `section` whose name starts with `.bss`, `.tdata` or `.tbss` | writes no bytes for a `.bss` segment, so its contents link to zeros; takes `.tdata` and `.tbss` for thread-local data | a relocatable object cannot express the section `.bss.mine` of the global variable `bss::d` |
| a declaration of `__start_<s>` or `__stop_<s>` where `s` is not a C identifier | defines the bounds only of a section named like a C identifier | a relocatable object cannot express `__start_a.b`: the linker defines no bounds for the section `a.b` |
| a declared memory other than `env.__linear_memory`, or one that names its host | provides the memory itself, imported as `env.__linear_memory`, and exports it as `memory` | a relocatable object cannot express the memory import `host.mem` |
| `wasm.export_name` on the memory | exports the memory itself, as `memory` | a relocatable object cannot express the export `__linear_memory` of the memory `__linear_memory` |
| `wasm.export_name` on a function, global or table the unit declares and does not define | exports only what an object defines | a relocatable object cannot express the export `hosted` of the function `hosted`, which it does not define |
| a `public` or `internal` `builtin.function_alias` of a function the unit does not define | has no function for the alias's symbol to name | a relocatable object cannot express the alias `al` of a function it does not define |
| data that ends beyond 4 GiB | a 32-bit memory cannot hold it | the data of the unit does not fit the 4 GiB of a 32-bit linear memory |

A frontend that needs a function table filled at fixed slots, or data at a fixed address, writes a
complete module instead, or leaves the placement to the linker with function values
(`builtin.function_ref` in data and in code gets a table slot from the linker) and global variables.
For example, this unit (`wasm.elem` into a table):

```sh
runic-opt bad.runic --target wasm32-unknown-unknown --stop-after object -o bad.o
```

fails with exit status 1:

```
error: a relocatable object cannot express `wasm.elem` @e
 --> bad.runic:6:5
  |
6 |     wasm.elem private @e : @tab at 0 [@f];
  |     ^^^^^^^^^ this fills a table at slots it fixes, and the linker ignores the element segments of an object: it lays out the function table itself
```

and `runic-opt bad.runic --target wasm32-unknown-unknown -o bad.core.wasm` compiles it.

### The features an object declares

`target_features` lists, in alphabetical order with the prefix `+`, the features the object uses.
They are the ones the complete module lists, and three more under LLVM's names: `mutable-globals`
(the stack pointer is an imported mutable global, and so is any other mutable global the object
imports or exports), `call-indirect-overlong` (a table number written padded, in an indirect call)
and `bulk-memory-opt` (`memory.copy` and `memory.fill`), and `reference-types` when the object has
more than one table. An object with no feature to declare has no section: `tests/run/link/machine.runic`
has `+multivalue` and `+tail-call`. By default the linker allows the union of the features of
all its objects, so an object that uses `return_call` links with ordinary Rust objects.

### What is not built

Weak symbols and COMDAT groups; thread-local data; position-independent code and shared libraries;
DWARF, and relocations for custom sections; archives (`.a`); more than one start function, and
priorities for them; `i128`. The harness has no linking step: the downstream compiler invokes the
linker.

### Toolchains

A downstream compiler that links with Rust depends on two behaviours of Rust's toolchains. They
were checked on 2026-11-17 on six toolchains (the nightlies 2026-04-30, 2026-09-01, which
`rust-toolchain.toml` pins, 2026-10-04 and 2026-11-17, and stable 1.98 and 1.100.0), and again on
2026-11-18 on the pinned nightly and stable 1.100.0:

- **A `#[link_section]` static lands in a data segment of that name** with the pinned nightly,
  stable 1.100.0 and nightly 2026-11-17; stable 1.98 puts it only in a custom section, outside
  the range from `__start_<section>` to `__stop_<section>`. Rust can always read and walk such a
  section, whatever its entries hold. From 1.100 on it can contribute to one, but **only entries
  with no pointer in them**: on the Wasm targets a `#[link_section]` static must be plain bytes,
  and the pinned nightly refuses one that holds a function pointer or a reference with
  ``error: statics with a custom `#[link_section]` must be a simple list of bytes on the wasm
  target with no extra levels of indirection such as references``. So an entry of a dispatch
  table `{id, arity, function}` cannot be written in Rust: a Rust function reaches a section
  through an object the frontend writes (whose entry names the Rust function), or through a table
  Rust builds at run time. `tests/run/link/section-rust.runic` shows both: a pointer-free Rust
  entry found beside the object's, and the refusal.
- **A library on the WASI targets has the export `_initialize`**, which calls `__wasm_call_ctors`
  once: it is linked with wasi-libc's `crt1-reactor.o` from nightly 2026-09-01 and stable 1.100.
  Up to stable 1.98 such a library is linked with `--no-entry` only, and the linker wraps every
  export to run the constructors on every call.

The run suite pins its toolchain, so that its results do not depend on which `rustc` is the
default (`tests/run/README.md`).

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
  follow the same rules; an indirect call takes the marks from its own signature (below).

The integers of the IR are signless, so a frontend states the signedness of a narrow integer that
crosses `extern("C")`: a C `signed char` or a Rust `i8` is written `i8 { extend = #builtin.sext }`
in a parameter and in a result, and a `u8` with `zext` or with no mark. An unmarked `i8` result
reaches Rust zero-extended, which is a `u8`: Rust expects the result of an `extern "C" fn -> i8`
sign-extended to 32 bits. The declaration and the definition of `tests/run/link/calls.runic`, whose
runtime is written in Rust:

```
builtin.function public extern("C") @rust_mix(ptr<i32>, i8 { extend = #builtin.sext }, i64, f64) -> (i8 { extend = #builtin.sext }) attributes { link_name = "rust_mix" };

builtin.function public extern("C") @compiled_mix(%cell: ptr<i32>, %b: i8 { extend = #builtin.sext }, %x: i64, %f: f64) -> (i8 { extend = #builtin.sext }) {
    ...
} attributes { link_name = "compiled_mix" };
```

With the marks removed from that file, the runtime's `round_trip 100 27 0` printed 128210 where
it prints -127790 with them.

For an indirect call the marks go on the signature of `builtin.call_indirect`: a function type,
the type of a function value, cannot carry them (`(i8 { extend = #builtin.sext }) -> i32` is a
syntax error). A function that calls a Rust callback of type `extern "C" fn(i8) -> i32`:

```
builtin.function public extern("C") @apply(%f: (i8) -> i32, %x: i8 { extend = #builtin.sext }) -> i32 {
    %r = builtin.call_indirect %f(%x) : extern("C") (i8 { extend = #builtin.sext }) -> i32;
    builtin.ret %r : (i32);
} attributes { link_name = "apply" };
```

linked with a runtime whose export `run(x)` is `apply(widen, x as i8)`, `widen` being
`x as i32`, gives -100 for -100. With the mark removed from the call's signature (the parameter
`%x` keeps its own), the argument reaches Rust zero-extended, and `run` gives 156.

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
| `builtin.aggregate` | an aggregate value is not one Wasm value: it exists in memory or in a global variable's initializer | `` `convert-to-wasm` cannot lower `builtin.aggregate`: an aggregate value in a function body is not lowered; it exists in memory or in a global variable's initializer `` (a `builtin.zero` of an aggregate type is the same error) |

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
- **Function values in memory need `fn = 32`.** The backend's data layout states it, so
  `mem.load` and `mem.store` move a function value as its 32-bit table index; a data layout with
  function values of another width is an error, and one with no `fn` entry gives a function type
  no layout, so the `mem` verifier refuses the access.
- **An `scf.break` that names a `wasm.loop` written before the lowering is an error**: leaving such
  a loop with its results has no `wasm.br` (a `wasm.br` to a loop goes back to its top).
- **A global variable's initializer is data**, and is not converted: `convert-to-wasm` and
  `wasm-verify-form` leave the regions of `builtin.global_variable` and `wasm.global` as they are,
  with one exception. The functions an initializer names are lowered, and their signatures
  convert (`(ptr<i32>) -> i32` becomes `(i32) -> i32`), so every function type a
  `builtin.global_variable` holds, in its type and in its initializer, at any depth, becomes the
  function type of its converted signature too, and nothing else in it changes: a pointer stays a
  pointer, and the data's layout is the same. A function type in data whose parameter or result
  has no carrier (an `i128`, say) is an error naming the global, given before anything changes.
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
