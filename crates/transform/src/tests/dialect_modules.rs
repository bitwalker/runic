//! One module per dialect of the workspace that uses each of the dialect's operations, for the
//! tests of printing and parsing (`generic_form.rs`, and `runic-opt`'s tests of
//! `--verify-roundtrip`). The modules are the parse tests of the lit suite,
//! `tests/lit/parse/<dialect>.runic`: their RUN and CHECK lines are comments to the parser. An
//! operation without a text of its own is written in the generic form. Operations added after a
//! dialect's parse test was written have a file of their own, listed after it under the dialect's
//! name with what it adds.

/// The modules, each with the name of its dialect.
pub const DIALECT_MODULES: [(&str, &str); 10] = [
    ("arith", include_str!("../../../../tests/lit/parse/arith.runic")),
    (
        "arith (floats)",
        include_str!("../../../../tests/lit/dialects/arith/float.runic"),
    ),
    ("builtin", include_str!("../../../../tests/lit/parse/builtin.runic")),
    (
        "builtin (tail calls)",
        include_str!("../../../../tests/lit/dialects/builtin/tail_call.runic"),
    ),
    ("cf", include_str!("../../../../tests/lit/parse/cf.runic")),
    ("di", include_str!("../../../../tests/lit/parse/di.runic")),
    ("mem", include_str!("../../../../tests/lit/parse/mem.runic")),
    ("scf", include_str!("../../../../tests/lit/parse/scf.runic")),
    ("test", include_str!("../../../../tests/lit/parse/test.runic")),
    ("ub", include_str!("../../../../tests/lit/parse/ub.runic")),
];
