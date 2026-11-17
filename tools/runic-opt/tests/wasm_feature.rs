//! The `wasm` dialect and the passes that lower to Wasm form are registered in `runic-opt` exactly
//! when the cargo feature `wasm` is on.

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
