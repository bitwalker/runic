# Runic

Runic is a compiler framework in Rust, modelled on MLIR. It provides an intermediate representation
(IR) whose operations are defined by _dialects_, a verifier, a pass manager, pattern rewriting,
dialect conversion, data-flow analyses and transformation passes, and `runic-opt`, a command-line
tool that runs passes on textual IR, like MLIR's `mlir-opt`.

The IR is made of _regions_, which hold _blocks_, which hold _operations_, which may hold regions
of their own. A region is either an SSA region (a control-flow graph of blocks) or a graph region.
The IR has a textual form, written in files with the extension `.runic`, in two forms, both parsed
and printed: a custom form, defined per operation, and a generic form that every operation has.

What exists today is the core IR, its dialects and passes, and the test tool. No workspace member
is a frontend or a backend: a compiler built on Runic brings its own.

## The crates

The workspace has sixteen members:

| crate | path | what it is for |
|---|---|---|
| `runic-ir` | `crates/ir` | the IR (operations, regions, blocks, values, types, attributes), its textual form (parser and printer), the verifier, the pass manager, pattern rewriting and dialect conversion; the `builtin` and `di` (debug information) dialects, and the `test` dialect the tests use |
| `runic-dialect-arith` | `crates/dialects/arith` | constants, arithmetic, bitwise operations, comparisons and casts on signless integers |
| `runic-dialect-cf` | `crates/dialects/cf` | unstructured control flow (`cf.br`, `cf.cond_br`, `cf.switch`), `cf.select` and `cf.assert` |
| `runic-dialect-scf` | `crates/dialects/scf` | structured control flow (`scf.if`, `scf.index_switch`, `scf.while` and their terminators) |
| `runic-dialect-mem` | `crates/dialects/mem` | byte-addressed memory, typed and opaque pointers |
| `runic-dialect-ub` | `crates/dialects/ub` | undefined behaviour (`ub.poison`, `ub.unreachable`) |
| `runic-analysis` | `crates/analysis` | the data-flow analysis framework and its analyses |
| `runic-transform` | `crates/transform` | transformation passes and utilities |
| `runic-compile` | `crates/compile` | the compiler harness: what a compilation is asked to do and the stages it runs through (being built; no tool uses it yet) |
| `runic-driver` | `crates/driver` | the command line of a compiler built on Runic: the driver's flags and those its frontends and backends declare (being built; no tool uses it yet) |
| `runic-diagnostics` | `crates/diagnostics` | structured diagnostics: describing, collecting and rendering them with their source |
| `runic-diagnostics-macros` | `crates/diagnostics-macros` | derive macros for `runic-diagnostics` |
| `runic-macros` | `crates/macros` | derive and attribute macros for defining dialects, operations, types, attributes and operation traits |
| `runic-symbol` | `crates/symbol` | string interning |
| `runic-log` | `crates/log` | the trace logger `runic-opt` installs (see Tracing) |
| `runic-opt` | `tools/runic-opt` | the command-line tool (below) |
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
cargo make runic-opt       # the same, and copies it to bin/runic-opt
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

## Testing

There are two test suites, the Rust tests and the lit suite, and unit tests of the Python scripts
in `tools/`:

```sh
cargo test --workspace     # the Rust tests
cargo make test            # the Rust tests under cargo-nextest, and the tests of the Python scripts in tools/
cargo make test-lit        # builds runic-opt into bin/ and runs the lit suite in tests/lit
cargo make test-all        # the two above, then removes old build output from target/
```

`cargo make test` installs `cargo-nextest`, and `cargo make test-lit` installs `litcheck`, when
they are missing (this needs the network). At its end, `cargo make test-all` deletes old
build output from `target/` (`tools/sweep-target.py`): in `target/debug`, what the workspace's build
and tests no longer use and is older than a day (`--age`); in the directories it cannot check (other
profiles, such as `test-release`, and nested caches), whatever is older than 14 days
(`--cache-age`), in use or not. Set `RUNIC_SKIP_SWEEP` to skip that. How the lit suite is
written and run, and how to run one test or one directory, is in
[`tests/lit/README.md`](tests/lit/README.md).

How to contribute, and the lint and format commands, are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Runic is licensed under the MIT license: see [`LICENSE.md`](LICENSE.md). The `runic-log` crate
states `MIT OR Apache-2.0` in its manifest (`crates/log/Cargo.toml`).
