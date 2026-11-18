//! The `wasm` dialect, the passes that lower to Wasm form, and the backend `wasm` with its
//! checkpoints are registered in `runic-opt` exactly when the cargo feature `wasm` is on.

use std::process::Command;

#[test]
fn the_wasm_dialect_is_listed_exactly_when_the_wasm_feature_is_on() {
    let output = Command::new(env!("CARGO_BIN_EXE_runic-opt"))
        .arg("--list-dialects")
        .output()
        .expect("runic-opt runs");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let listed = stdout.lines().any(|line| line.split_whitespace().next() == Some("wasm"));
    assert_eq!(listed, cfg!(feature = "wasm"), "{stdout}");
}

#[test]
fn the_passes_that_lower_to_wasm_form_are_listed_exactly_when_the_wasm_feature_is_on() {
    let output = Command::new(env!("CARGO_BIN_EXE_runic-opt"))
        .arg("--list-passes")
        .output()
        .expect("runic-opt runs");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for pass in ["convert-to-wasm", "wasm-lower-frames", "wasm-verify-form"] {
        let listed = stdout.lines().any(|line| line.split_whitespace().next() == Some(pass));
        assert_eq!(listed, cfg!(feature = "wasm"), "{pass}: {stdout}");
    }
}

#[test]
fn the_wasm_backend_and_its_checkpoints_are_listed_exactly_when_the_wasm_feature_is_on() {
    let output = Command::new(env!("CARGO_BIN_EXE_runic-opt"))
        .arg("--list-stages")
        .output()
        .expect("runic-opt runs");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for line in [
        "  wasm: wasm32-unknown-unknown",
        "  wasm.object (alias object) -> .o",
        "  wasm.wat (alias wat) -> .wat",
        "  wasm.module (alias module) -> .core.wasm",
    ] {
        let listed = stdout.lines().any(|listed| listed == line);
        assert_eq!(listed, cfg!(feature = "wasm"), "{line}: {stdout}");
    }
}

#[cfg(feature = "wasm")]
#[test]
fn the_target_wasm32_unknown_unknown_selects_the_wasm_backend_without_backend_and_gives_a_binary() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("twice.runic");
    std::fs::write(
        &input,
        "builtin.module public @m {\n    builtin.function public extern(\"C\") @twice(%x: i32) -> \
         i32 {\n        %r = arith.add %x, %x;\n        builtin.ret %r : (i32);\n    };\n};\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_runic-opt"))
        .arg(&input)
        .args(["--target", "wasm32-unknown-unknown"])
        .output()
        .expect("runic-opt runs");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(&output.stdout[..8], b"\0asm\x01\0\0\0");
}
