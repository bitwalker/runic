# Runic

Runic is a compiler framework in Rust, modelled on MLIR. It provides an intermediate representation
(IR) whose operations are defined by _dialects_, a verifier, a pass manager, pattern rewriting,
dialect conversion, data-flow analyses and transformation passes, and `runic-opt`, a command-line
tool that runs passes on textual IR, like MLIR's `mlir-opt`.

The IR is made of _regions_, which hold _blocks_, which hold _operations_, which may hold regions
of their own. A region is either an SSA region (a control-flow graph of blocks) or a graph region.
The IR has a textual form, written in files with the extension `.runic`, in two forms, both parsed
and printed: a custom form, defined per operation, and a generic form that every operation has.

What exists today is the core IR, its dialects and passes, the compiler harness and the command
line a compiler embeds, and `runic-opt`. No workspace member is a frontend or a backend of a real
language or target: a compiler built on Runic brings its own. The toy language and machines the
harness is tested against are part of `runic-compile`, behind its `testing` feature.

## The crates

The workspace has nineteen members:

| crate | path | what it is for |
|---|---|---|
| `runic-ir` | `crates/ir` | the IR (operations, regions, blocks, values, types, attributes), its textual form (parser and printer), the verifier, the pass manager, pattern rewriting and dialect conversion; the `builtin` and `di` (debug information) dialects, and the `test` dialect the tests use |
| `runic-dialect-arith` | `crates/dialects/arith` | constants, arithmetic, bitwise operations, comparisons and casts on signless integers; constants, arithmetic, comparisons and conversions on `f32` and `f64` |
| `runic-dialect-cf` | `crates/dialects/cf` | unstructured control flow (`cf.br`, `cf.cond_br`, `cf.switch`), `cf.select` and `cf.assert` |
| `runic-dialect-scf` | `crates/dialects/scf` | structured control flow (`scf.if`, `scf.index_switch`, `scf.while` and their terminators) |
| `runic-dialect-mem` | `crates/dialects/mem` | byte-addressed memory, typed and opaque pointers |
| `runic-dialect-ub` | `crates/dialects/ub` | undefined behaviour (`ub.poison`, `ub.unreachable`) |
| `runic-dialect-wasm` | `crates/dialects/wasm` | WebAssembly's instructions as SSA operations (numeric, memory, global, table and reference operations, indirect calls, structured control flow with `wasm.block`, `wasm.loop`, `wasm.if`, `wasm.br` and `wasm.br_table`) and module-level operations for memories, tables, globals, element segments and tags |
| `runic-codegen-wasm` | `crates/codegen/wasm` | the lowering of generic IR to Wasm form, the input of the WebAssembly backend: the passes `convert-to-wasm` (a dialect conversion to `wasm` operations and Wasm value types, with the `"wasm"` and `"C"` calling conventions), `wasm-lower-frames` and `wasm-verify-form`, and the pipeline that runs them after `cfg-to-scf` |
| `runic-analysis` | `crates/analysis` | the data-flow analysis framework and its analyses |
| `runic-transform` | `crates/transform` | transformation passes and utilities |
| `runic-compile` | `crates/compile` | the compiler harness: a request that says what one compilation is asked to do, the stages it runs through (frontends, the optimization pipeline, backends), the checkpoints and outputs, and the toy language `calc` and machines `stackvm` and `altvm` behind the `testing` feature |
| `runic-driver` | `crates/driver` | the command line of a compiler built on Runic: the driver's flags and those its frontends and backends declare, the exit codes, and the example compiler `toyc`; `runic-opt` uses its pieces |
| `runic-diagnostics` | `crates/diagnostics` | structured diagnostics: describing, collecting and rendering them with their source |
| `runic-diagnostics-macros` | `crates/diagnostics-macros` | derive macros for `runic-diagnostics` |
| `runic-macros` | `crates/macros` | derive and attribute macros for defining dialects, operations, types, attributes and operation traits |
| `runic-symbol` | `crates/symbol` | string interning |
| `runic-log` | `crates/log` | the trace logger `runic-opt` installs (see Tracing) |
| `runic-opt` | `tools/runic-opt` | the IR-to-IR command-line tool, which also runs a compilation through the harness (below) |
| `runic-expect-test` | `tools/expect-test` | snapshot testing for the Rust tests |

The directories `crates/eval` and `crates/dialects/wasm` hold an evaluator and a WebAssembly
dialect from before Runic was split out of an earlier compiler, kept, not built: they are not
workspace members, and they are not tested.

## Building

The toolchain is pinned in `rust-toolchain.toml` (a nightly channel); `rustup` installs it on the
first `cargo` command run in the repository (this needs the network). The tasks of the
repository are run with [`cargo-make`](https://github.com/sagiegurari/cargo-make)
(`cargo install cargo-make`, which needs the network).

```sh
cargo build --workspace    # or: cargo make build (also: cargo make)
```

builds every member. The tool alone:

```sh
cargo build -p runic-opt   # target/debug/runic-opt
cargo make runic-opt       # with the `toy` feature, copied to bin/runic-opt
cargo make toyc            # the example compiler, copied to bin/toyc
```

## `runic-opt`

`runic-opt` reads one top-level operation in the textual form (from a file, or standard input),
parses and verifies it, runs the pass pipeline given with `-p`, verifying after each pass, and
prints the result. `runic-opt --help` describes every option; `--list-dialects`,
`--list-passes` and `--list-operations` list what it has registered.

An example from the test suite, `tests/lit/transforms/cse/simple-constant.runic`. The file is a
lit test: besides the IR it holds `// RUN:` lines, the commands the suite runs on it, and
`// CHECK` lines, what their output must contain (see [`tests/lit/README.md`](tests/lit/README.md)).
Its IR is this function:

```
builtin.function public extern("C") @simple_constant() -> (i32, i32) {
    %a = arith.constant 1 : i32;
    %b = arith.constant 1 : i32;
    builtin.ret %a, %b : (i32, i32);
};
```

Common subexpression elimination replaces the second constant by the first:

```console
$ bin/runic-opt tests/lit/transforms/cse/simple-constant.runic -p cse
builtin.function public extern("C") @simple_constant() -> (i32, i32) {
    %0 = arith.constant 1 : i32;
    builtin.ret %0, %0 : (i32, i32);
};
```

With `--print-generic`, the same result is printed in the generic form.

An input that a registered frontend reads (a file with the extension of a frontend other than the
one for textual IR), or a command line that writes `--target`, `--backend`, `--stop-after`,
`--emit NAME=PATH`, `--language` or a flag or `-Z` entry of a stage, takes the harness path: the
input is compiled by `runic-compile` instead of being parsed as IR, and the exit code is 0 or 1
as for IR. `--list-stages` lists the registered frontends and backends. With the cargo feature
`toy` (which `cargo make runic-opt` turns on), the tool registers the toy language `calc` and the
toy machines `stackvm` and `altvm`, so that the path can be run and tested.

### Tracing

`runic-opt` installs the logger of `runic-log`, configured by four environment variables
(described in full in the crate's documentation, `crates/log/src/lib.rs`):

- `RUNIC_TRACE`: which records are written, as a list of directives, such as `trace`,
  `dataflow=debug` or `pattern-rewrite-driver=trace`; without it, only errors.
- `RUNIC_TRACE_FILTER`: keeps, of the records that carry a key, those whose value matches, such as
  `op:add` or `dialect:arith`.
- `RUNIC_TRACE_STYLE`: whether records are styled: `auto` (the default), `always` or `never`.
- `RUNIC_TRACE_TIMING`: gives each record a timestamp of the precision `s`, `ms`, `us` or `ns`.

Records are written to standard error. A malformed `RUNIC_TRACE_FILTER` or `RUNIC_TRACE_TIMING`
is an error, and the tool runs nothing.

## Building a compiler on Runic

A compiler built on Runic supplies three things: its frontends, its backends and a `main`. A
frontend reads inputs of one language and builds IR. A backend lowers IR for the targets it
supports and writes the result. The rest comes from the workspace: the IR, the passes, the
harness in `runic-compile` that routes a compilation through the stages, and the command line
in `runic-driver`. The driver gives the compiler the flags of a compiler (`--target`, `--emit`,
`--stop-after`, `-O`, `-o`, `--print`, `--help`, `--version`, `-Z`) and the flags each of its stages
declares, and turns the outcome of a compilation into an exit code: 0 for success, 1 for a failed
compilation, 2 for a command line that is not valid, and 101 for a panic in a stage.

`crates/driver/examples/toyc.rs` is a whole compiler. It compiles `calc`, a toy language, and
the textual IR, for two toy stack machines. Its text:

```rust
use std::process::ExitCode;

use runic_compile::{
    Compiler, IrFrontend,
    testing::{AltVmBackend, CalcFrontend, StackVmBackend},
};
use runic_driver::{Driver, EXIT_INTERNAL};

fn main() -> ExitCode {
    let compiler = Compiler::builder()
        .frontend(CalcFrontend)
        .frontend(IrFrontend)
        .backend(StackVmBackend)
        .backend(AltVmBackend)
        .default_target("stackvm-unknown-tiny")
        .build();
    let compiler = match compiler {
        Ok(compiler) => compiler,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(EXIT_INTERNAL);
        }
    };
    let driver = Driver::new(&compiler)
        .name("toyc")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Compiles calc, a toy language, for two toy stack machines")
        .build();
    match driver {
        Ok(driver) => runic_driver::main_with(&driver),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(EXIT_INTERNAL)
        }
    }
}
```

Three command lines:

```sh
# Compile f.calc for the default target (stackvm-unknown-tiny). The listing goes to f.svm.
toyc f.calc

# Stop after the optimized IR and print it to standard output.
toyc f.calc --stop-after ir -o -

# Both machines support stackvm-unknown-unknown, so the command line names one. The altvm
# output goes to f.alt in the directory out, which is created if it is not there.
toyc f.calc --target stackvm-unknown-unknown --backend altvm --output-dir out
```

`toyc --print targets`, `--print backends`, `--print frontends` and `--print checkpoints` list what
the compiler has. `toyc --help` groups the flags of each stage under its own heading, such as
`stackvm backend options`.

A stage declares its options with a struct that derives `clap::Args` (`clap` 4), and returns it
from `Frontend::options` or `Backend::options` as an `OptionsDecl`. Each field is a long flag. A
stage cannot declare a short flag, a positional argument, a required argument or an `env`, since
nothing in a compilation reads the process's environment except through the request, and each
stage needs a struct of its own: a request holds one value of a type. A flag only fills a field of
the struct, so everything the command line says can be said in code, and an option nobody wrote
has the value `clap` gives it (its `default_value`) whether it is set on the command line or in
code. The driver prefixes
the flags of a backend with the backend's name: the field `comments` of the backend `stackvm` is
`--stackvm-comments`. The flags of a frontend have no prefix, unless the driver is built with
`DriverBuilder::prefix_frontend`.

An option that is not stable goes in a second struct, given with `OptionsDecl::unstable`. It is
not a flag. A user sets it with `-Z NAME[=VALUE]`, where a backend's setting has the backend's name
as a prefix: the field `keep_subtractions` of `stackvm` is `-Z stackvm-keep-subtractions`.
`-Z help` lists every unstable setting of the driver and of the stages.

## Testing

There are two test suites, the Rust tests and the lit suite, and unit tests of the Python scripts
in `tools/`:

```sh
cargo test --workspace     # the Rust tests
cargo make test            # the Rust tests under cargo-nextest, and the tests of the Python scripts in tools/
cargo make test-lit        # builds runic-opt and toyc into bin/ and runs the lit suite in tests/lit
cargo make test-all        # the two above, then removes old build output from target/
```

`cargo make test` installs `cargo-nextest`, and `cargo make test-lit` installs `litcheck`, when
they are missing (this needs the network). At its end, `cargo make test-all` deletes old
build output from `target/` (`tools/sweep-target.py`): in `target/debug`, what the workspace's build
and tests no longer use and is older than a day (`--age`); in the directories it cannot check (other
profiles, such as `test-release`, and nested caches), whatever is older than 14 days
(`--cache-age`), in use or not. Set `RUNIC_SKIP_SWEEP` to skip that. The lit suite has two areas for the harness:
`tests/lit/harness/` runs `runic-opt` on `calc` programs and with `--target`, and `tests/lit/toyc/`
runs `toyc`. How the lit suite is written and run, and how to run one test or one directory, is in
[`tests/lit/README.md`](tests/lit/README.md).

How to contribute, and the lint and format commands, are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Runic is licensed under the MIT license: see [`LICENSE.md`](LICENSE.md). The `runic-log` crate
states `MIT OR Apache-2.0` in its manifest (`crates/log/Cargo.toml`).
