# diagnostics-macros

This proc-macro crate generates implementations of the
`runic_diagnostics::Diagnostic` protocol. Most users should depend on
`runic-diagnostics` with its default `derive` feature and import the macro
from there:

```rust
use runic_diagnostics::{Diagnostic, SourceSpan};

#[derive(Debug, Diagnostic)]
#[diagnostic(
    code = "example/E0001",
    summary = "unexpected token",
    message = "unexpected token `{token}`"
)]
struct UnexpectedToken {
    token: String,
    #[label(primary, "not valid here")]
    span: SourceSpan,
}
```

## What the derive generates

The derive supports diagnostic structs and enums, descriptor-backed or inline
metadata, formatted messages, labels, notes and help, suggestions, related
diagnostics, ordinary error sources, rich diagnostic sources, transparent
wrappers, and forwarding wrappers. Generated implementations visit semantic
data; source lookup, snapshot preparation, rendering, collection, and failure
policy remain runtime responsibilities.

A crate that depends on `runic-core` but not on `runic-diagnostics` gets the
runtime through its re-export, `runic_core::diagnostics`.

Dependency renames are resolved automatically. Unusual build layouts can use
`#[diagnostic(crate = path::to::runtime, ...)]` to select the runtime path
explicitly. This is useful when deriving inside an example target that belongs
to the runtime package itself.

## Crate boundary

This crate runs on the compiler host and depends on `syn`, `quote`,
`proc-macro2`, and `proc-macro-crate`. Generated code refers to the consumer's
`runic-diagnostics` crate, or to the one `runic-core` re-exports.

See the [runtime crate README](../diagnostics/README.md) for the data model.

This crate is licensed under the MIT license: see [`LICENSE.md`](../../LICENSE.md).
