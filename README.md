# Runic Compiler Framework

This project is an attempt at building a Rust-native, MLIR-inspired compiler framework and toolchain for custom compiler projects.

At its core, it provides a rich and flexible intermediate representation (IR), capable of representing both high-level and low-level operations via _dialects_. The IR itself is comprised of _regions_, which contain _blocks_, which contain _operations_, which in turn may have nested regions. Like MLIR, this IR is a form of _Regionalized Value State Dependence Graph_. The IR has both generic and pretty printed forms, with support for parsing both. The IR supports both SSA and graph-like (unscheduled) regions.

In addition to the core IR, Runic provides a dataflow analysis framework, a pass pipeline and manager, a canonicalizer based around a greedy pattern rewrite driver (which is also used for applying transformation passes), as well as a dialect conversion framework for lowering one or more dialects to a target.

The `runic-opt` executable can be used to parse some IR, apply a pass pipeline, and print the resulting IR - primarily useful in testing and experimentation.

The `runic` executable provides an experimental harness for batch compilation use cases (i.e. take some inputs, lower them to IR, then emit code for some target backend). It supports registering custom frontend languages (e.g. Rust) and backend targets (e.g. Wasm), and handles orchestrating the compilation pipeline so that a frontend need only lower to Runic's IR, and a backend need only understand the built-in Runic dialects to be able to emit valid programs. This part is still experimental because so much of what goes into a compilation session can be context-specific, that the utility of this is still being evaluated.

## Installing

You may install from source, in which case you will need to have Rust installed on your system, and then run `cargo install --locked runic` from the repo root.

Alternatively, you can download prebuilt binaries from the [latest GitHub release](https://github.com/bitwalker/runic/releases/latest), and install them somewhere in your shell's `PATH`.

> [!TIP]
> Our prebuilt binaries have attached GitHub build attestations, which can be verified using the `gh` CLI, like so:
>
> ```
> gh attestation verify runic-aarch64-apple-darwin.tar.gz \
>    --repo bitwalker/runic \
>    --signer-workflow bitwalker/runic/.github/workflows/release.yml
> ```

## Development

You'll need to have Rust installed. This repository pins the toolchain in `rust-toolchain.toml` at the repo root (currently a nightly channel); use `rustup` to install that exact channel after cloning so local builds match CI.

Additionally, you'll need to have [`cargo-make`](https://github.com/sagiegurari/cargo-make) installed:

    $ cargo install cargo-make

From there, build the compiler with:

    $ cargo make

To build just the compiler:

    $ cargo make runic

## Testing

To run the compiler test suite:

    $ cargo make test

This runs the Rust tests, including integration and template tests. To include the `lit`/FileCheck
suite, run `cargo make test-all`.

### Emitting internal sources/artifacts

- `RUNIC_EMIT`: Environment-variable equivalent of `--emit`. Accepts the same `KIND[=PATH]` syntax
  (comma-delimited), where `PATH` is treated either as folder e.g. `RUNIC_EMIT=ir=target/emit` or file `RUNIC_EMIT=hir=my_name.hir`.

## License

This project is dual-licensed under MIT and Apache 2.0, see [LICENSE](LICENSE.md) for details.
