# The run suite

Whole programs compiled to WebAssembly by `runic-opt` and run with `wasmtime`. Each `*.runic`
file under `wasm/` is a program in generic IR, with `// RUN:` lines that compile it with the
backend `wasm` (`crates/codegen/wasm`) and call its exports, and checks of what they print. Where
the lit suite of `tests/lit` checks the text the lowering and the emitter produce, this one checks
that the module runs and computes what it should, under a real engine. The crate's unit tests
(`crates/codegen/wasm/src/tests/emit/programs.rs`) run the same files under `wasmi`, so the
programs are checked on a machine without `wasmtime` too. The files under `link/` are compiled to
relocatable objects instead and linked with small runtimes written in Rust, by `rustc`, the way a
downstream compiler links them ([Linking with Rust](#linking-with-rust-link)).

The suite is its own (`lit.suite.toml`, the suite `run`, with `pipefail` as in `tests/lit`)
because `litcheck` cannot skip a test whose tool is missing: it needs `wasmtime` 50.0.2, the
version the checks are written against.

## Running it

```sh
cargo make test-run
```

builds `runic-opt` into `bin/`, makes sure `wasmtime-cli` 50.0.2 is what cargo has installed,
prints the `wasmtime` it found and its version, and runs the suite. When cargo has no `wasmtime-cli`
installed, or another version of it, the task runs `cargo install --force --locked --version 50.0.2
wasmtime-cli`: it builds from source, once, needs the network, and **replaces another installed
version**. The check that fails the task with one line naming the version wanted applies after
that, when the `wasmtime` first on the `PATH` is not the one cargo installed (another
installation earlier on the `PATH`, of another version) or there is none. `cargo make wasmtime`
does the install alone. `cargo make test-all` runs this suite after `tests/lit`.

The tests of `link/` also need the toolchain `rust-toolchain.toml` pins, with its target
`wasm32-unknown-unknown` (the file lists it, so rustup installs it with the toolchain). The task
prints that `rustc`'s version, and fails with one line when rustup does not have the toolchain,
when the toolchain lacks the target, or when `link/lit.local.toml` names another toolchain than
`rust-toolchain.toml` (below).

One test, or the suite, by hand, from the repository root. `litcheck` gives its tests the
directories of `--path` in front of its own `PATH`, so `wasmtime` is found where `cargo install`
put it (`~/.cargo/bin`, usually on `PATH` already; a second `--path` names it otherwise):

```sh
cargo make runic-opt
litcheck lit run --path bin tests/run/wasm/machine.runic
litcheck lit run -v --path bin --path ~/.cargo/bin tests/run
```

## Writing a test

A program is generic IR in CFG form (`arith`, `cf`, `mem`, `builtin`), whose functions are
exported with `wasm.export_name`, with a comment that says what it shows. It needs no host: the
module defines its memory, its stack and its function table, and states no `data_layout` (it
relies on the target's). The data a program reads is in its own global variables.

Each program runs twice, with one set of checks: as written, and with the usual optimization
pipeline before the backend (`runic-opt` compiles at `-O0`, so neither the level's pipeline nor
the backend's clean-up after the lowering runs in it):

```
// RUN: runic-opt --target wasm32-unknown-unknown %s -o %t.wasm
// RUN: wasmtime run --invoke sum %t.wasm 3 2>/dev/null | filecheck %s --check-prefix=THREE
// RUN: runic-opt --target wasm32-unknown-unknown %s -p 'canonicalizer,cse,sccp,canonicalizer,dce' -o %t.opt.wasm
// RUN: wasmtime run --invoke sum %t.opt.wasm 3 2>/dev/null | filecheck %s --check-prefix=THREE
```

- **One call per RUN line.** `wasmtime run --invoke NAME MODULE ARGS..` instantiates the module,
  calls the export once, and prints each result on a line of its own. Options come before the
  module; everything after it is an argument. A program that shows state kept across calls makes
  the calls itself, in one exported function.
- **Standard output only.** `wasmtime` writes two warnings that `--invoke` is experimental to
  standard error; a RUN line discards it with `2>/dev/null`. Each case has a check prefix of its
  own, and each result is matched as a whole line, `{{^}}42{{$}}`, so that `-1` does not match
  `-12`.
- **How values print.** An argument is written in decimal (`-1`, not `0xffffffff`), a float as
  Rust parses one (`1.5`, `1e10`, `NaN`). An `i32` or `i64` result prints signed: the unsigned
  largest `i32` prints as `-1`. A narrow integer is its carrier, an `i32` holding the value
  zero-extended: the `i8` -56 prints as `200`. A float prints as Rust's `{}` does: `1.25`, `-6`
  (no `.0`), `inf`, `NaN`, `0.10000000149011612`.
- **A trap** exits with a failure status (134) and writes `wasm trap: <message>` to standard error,
  after a backtrace. A RUN line checks it with `not`, sending standard error to `filecheck` and
  discarding standard output:

  ```
  // RUN: not wasmtime run --invoke sdiv_i32 %t.wasm -2147483648 -1 2>&1 >/dev/null | filecheck %s --check-prefix=OVERFLOW
  // OVERFLOW: wasm trap: integer overflow
  ```

  The messages the programs give: `integer overflow`, `integer divide by zero`, ``wasm `unreachable`
  instruction executed`` (a failed `cf.assert`, a checked operation that overflows, the minimum
  divided by -1 below 32 bits), `out of bounds memory access`, `uninitialized element` (a call
  through the empty slot 0), `undefined element: out of bounds table access`, and `call stack
  exhausted`.
- **Imports.** A second module is a test of its own in the same directory, compiled to its own file
  with `%S/<file>.runic -o %t.<name>.wasm`, and given to the importing module with
  `wasmtime run --preload env=%t.<name>.wasm`: its exports satisfy the imports from `env`
  (`imports.runic` and `imports-env.runic`).
- `%t` is a file of the test's own, as in `tests/lit`: a suffix makes another (`%t.opt.wasm`).

A new program is added to `crates/codegen/wasm/src/tests/emit/programs.rs` too, with the same calls
and results; a test there fails when a file of `wasm/` has no test.

## The programs

| file | what it shows |
|---|---|
| `sum.runic` | a loop summing an array held in a global variable, for several lengths |
| `machine.runic` | three functions that tail-call each other and return two results; a million tail calls in a row, and the same as calls, which exhausts the stack |
| `dispatch.runic` | a `cf.switch` with two tail calls and a return behind it |
| `narrow.runic` | `i1`, `i8` and `i16` arithmetic, comparisons and selects under both calling conventions: wrapping, masking at the entry, sign extension |
| `frames.runic` | a struct in a frame, a pointer to it passed to a callee, returns at two depths, and 100,000 calls in a row that leave the stack pointer where it was |
| `globals.runic` | a global variable read and written across calls, a pointer in a global, a function value in data called indirectly |
| `dispatch-table.runic` | a dispatch table of structs built with `builtin.aggregate`, each an id, an arity and a function, in a named section: walked from `__start_dispatch` to `__stop_dispatch`, the function value loaded with `mem.load` and called through; a second section stays apart |
| `float.runic` | float arithmetic, comparisons with a NaN, conversions both ways, the saturating ones included |
| `traps.runic` | each trap the lowering produces, with the message `wasmtime` prints |
| `imports.runic`, `imports-env.runic` | imports from `env` satisfied by a second emitted module |
| `start.runic` | a start function (`wasm.start`) that a complete module exports as `_initialize`, which `wasmtime` calls when it instantiates the module: `read` gives 42, not 7 |

## Linking with Rust: `link/`

Each test of `link/` compiles its IR to a relocatable object, links the object with a runtime
written in Rust (`link/rt/<name>.rs`) into a library, and calls the library's exports:

```
// RUN: runic-opt --target wasm32-unknown-unknown --stop-after object %s -o %t.o
// RUN: rustc --edition 2024 --target wasm32-unknown-unknown --crate-type cdylib -O %S/rt/calls.rs -C link-arg=%t.o -C link-arg=--fatal-warnings -o %t.wasm
// RUN: wasmtime run --invoke round_trip %t.wasm -3 10 2.5 2>/dev/null | filecheck %s --check-prefix=SMALL
```

and the same after the optimization pipeline, as in `wasm/`. `rustc` passes the object to
`rust-lld` after Rust's own objects and libraries, and exports the runtime's `#[no_mangle]`
functions; a function the object defines is exported only when it has `wasm.export_name` (or
with `-C link-arg=--export=<name>`). A runtime is `no_std` with a panic handler of its own unless
it needs the standard library (`rt/alloc.rs`). Where a test needs a second runtime, it builds the same file with a `--cfg`
(`mismatch`, `bare`, `wrapped`).

- **Every link has `-C link-arg=--fatal-warnings`.** `rust-lld` only warns when a function is
  called with a signature that differs from its definition, and replaces the call with one that
  traps; the warning makes the link fail instead (`calls.runic` checks it).
- **The toolchain is the pinned one.** `litcheck` runs RUN lines in a temporary directory, where
  `rust-toolchain.toml` is not seen, so a bare `rustc` would be the user's default toolchain.
  `link/lit.local.toml` sets `RUSTUP_TOOLCHAIN` to the channel `rust-toolchain.toml` pins (and
  `RUSTUP_AUTO_INSTALL=0`), for the suite run by `cargo make test-run` and for a test run by hand.
  It is a second statement of the pin: `test-run` fails with one line when the two differ, and
  `toolchain.runic` checks that the tests' `rustc` is the one rustup picks inside the repository.
  The pin matters: a `#[link_section]` static of Rust's lands in a data section of that name
  from Rust 1.100 on; stable 1.98 left it out of the section's bounds (`section-rust.runic`
  counts Rust's entry).
- **Rust contributes only plain bytes to a section.** It can read and walk any section, but on the
  Wasm targets `rustc` refuses a `#[link_section]` static that holds a pointer, a function's
  included: ``error: statics with a custom `#[link_section]` must be a simple list of bytes on
  the wasm target with no extra levels of indirection such as references`` (checked on the
  pinned nightly by `section-rust.runic`). An entry that names a function, as a dispatch table's
  does, comes from an object.
- **A second module** of a test is a test of its own, with RUN lines that need nothing else, as
  in `wasm/`: `dispatch-more.runic` (a second object), and `host-env.runic`, `start-host.runic`
  and `export-global-host.runic` (complete modules that the test gives to `wasmtime --preload`).
- **IR that meets Rust** names each symbol Rust names with `link_name`, and marks a narrow
  integer as Rust's C convention passes it: a Rust `i8` is sign-extended, `extend = #builtin.sext`
  on the `i8`; an unmarked `i8` is passed zero-extended, a Rust `u8`.

| file | what it shows |
|---|---|
| `toolchain.runic` | the tests' `rustc` is the pinned one |
| `calls.runic` | Rust calls compiled code and compiled code calls Rust under `extern("C")`, with a pointer, an `i8`, an `i64` and an `f64` parameter and an `i8` result; a mismatched signature fails the link |
| `stack.runic` | one stack: a slot of compiled code's frame written by Rust through its address, calls in a row, and recursion through Rust with frames of both kinds nested |
| `memory.runic` | one memory: compiled code reads and writes a Rust `static mut`, Rust a global variable of the object; the object's data holds the address of a Rust static plus 8 |
| `dispatch.runic`, `dispatch-more.runic` | entries of the section `dispatch` from two objects, found by Rust between `__start_dispatch` and `__stop_dispatch` and called through; retained with nothing naming them |
| `machine.runic` | `(i32, i64) -> (i1, i64)` functions under `extern("wasm")` that tail-call each other a million times, entered from Rust through an `extern("C")` wrapper |
| `alloc.runic` | a runtime with `std`: compiled code calls Rust to allocate a vector, fills it, and Rust sums and frees it |
| `host.runic`, `host-env.runic` | a host import (`wasm.import_module`) satisfied by `--preload`; an undefined symbol nothing defines fails the link with the linker's message |
| `heap.runic` | `__heap_base` and `__data_end` read from compiled code |
| `start.runic`, `start-host.runic` | a start function (`wasm.start`) beside a Rust constructor in `.init_array`: run once by `__wasm_call_ctors` from the runtime's `init`, or before every export when nothing names it |
| `export-global.runic`, `export-global-host.runic` | a Wasm global with an export name is exported from the linked module, and the host reads it |
| `section-rust.runic` | Rust contributes a pointer-free entry to a section the object contributes to, and both Rust and compiled code count all three entries; an entry holding a function pointer is refused by `rustc` |
| `duplicate-export.runic` | an export of the object and one of Rust's with the same name: the link succeeds, even with `--fatal-warnings`, and `wasmtime` refuses the module as not valid (the linker does not check export names) |
