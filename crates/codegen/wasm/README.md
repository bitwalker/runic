# runic-codegen-wasm

The lowering of generic Runic IR (the dialects `arith`, `cf`, `mem`, `scf`, `ub` and `builtin`)
to _Wasm form_, the input of the WebAssembly backend.

A function in Wasm form is a `builtin.function` whose body is made of `wasm` operations
(`runic-dialect-wasm`) and of `builtin.call`, `builtin.tail_call`, `builtin.ret`,
`builtin.load_local` and `builtin.store_local`; every value in it, every parameter, result and
local has a Wasm value type (`i32`, `i64`, `f32`, `f64`, `!wasm.funcref`, `!wasm.externref`);
every region holds one block; and its calling convention is `"wasm"` or `"C"`. Nothing here emits
a binary.

The passes, in the order `add_lowering_passes` adds them:

| pass | what it does |
|---|---|
| `cfg-to-scf{exits=in-place}` | (`runic-dialect-scf`) structures control flow, leaving each return and tail call in place |
| `wasm-lower-frames` | gives every `mem.alloca` a slot in a frame on the shadow stack (registered; changes nothing yet) |
| `convert-to-wasm` | a full dialect conversion of each `builtin.module` to Wasm form |
| `wasm-verify-form` | checks that every function is in Wasm form; changes nothing |

A narrow integer (`iN` with `N` below 32, or between 32 and 64) is carried in an `i32` or an
`i64` with the bits above `N` clear. Under `extern("wasm")` both sides of a call rely on that;
under `extern("C")` nothing is assumed that the signature does not say, so narrow parameters and
results are masked or sign-extended according to their `zext` and `sext` markers.

The lowering needs a data layout in scope that is little-endian with 32-bit pointers and `isize`:
`data_layout = #builtin.data_layout<little, ptr = 32>` on the module, or a `--target`.

The crate documentation (`src/lib.rs`) describes carried form, the passes and the well-known
symbols (`@__linear_memory`, `@__indirect_function_table`, `@__stack_pointer`) in full.
