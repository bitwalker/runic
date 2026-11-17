# The run suite

Whole programs compiled to WebAssembly by `runic-opt` and run with `wasmtime`. Each `*.runic`
file under `wasm/` is a program in generic IR, with `// RUN:` lines that compile it with the
backend `wasm` (`crates/codegen/wasm`) and call its exports, and checks of what they print. Where
the lit suite of `tests/lit` checks the text the lowering and the emitter produce, this one checks
that the module runs and computes what it should, under a real engine. The crate's unit tests
(`crates/codegen/wasm/src/tests/emit/programs.rs`) run the same files under `wasmi`, so the
programs are checked on a machine without `wasmtime` too.

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
| `float.runic` | float arithmetic, comparisons with a NaN, conversions both ways, the saturating ones included |
| `traps.runic` | each trap the lowering produces, with the message `wasmtime` prints |
| `imports.runic`, `imports-env.runic` | imports from `env` satisfied by a second emitted module |
