# runic-codegen-wasm

The lowering of generic Runic IR (the dialects `arith`, `cf`, `mem`, `scf`, `ub` and `builtin`)
to _Wasm form_, the input of the WebAssembly backend. Nothing here emits a binary: the result is
IR, checked as text.

```rust
let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
runic_codegen_wasm::add_lowering_passes(passes.op_pass_manager_mut())?;
passes.run(module.as_operation_ref())?;
```

or, in the tool (`cargo make runic-opt` builds it with the crate):

```sh
runic-opt program.runic -p 'cfg-to-scf{exits=in-place},wasm-lower-frames,convert-to-wasm,wasm-verify-form'
```

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
programs with an evaluator written from the WebAssembly specification (release 3.0); the lit tests
of `tests/lit/codegen/wasm` check the lowered text.
