//! The target read from the data layout in scope.

use super::*;
use crate::WasmTarget;

/// The target of the module `@m` with the attribute dictionary `attributes` (empty for none).
fn target_of(attributes: &str) -> Result<WasmTarget, Report> {
    let text = format!("builtin.module public @m {attributes} {{\n}};");
    let (_context, module) = parse_module(&text);
    WasmTarget::of(module.borrow().as_operation())
}

#[test]
fn a_little_endian_layout_with_32_bit_pointers_is_the_target() {
    let target = target_of("attributes { data_layout = #builtin.data_layout<little, ptr = 32> }")
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    assert_eq!(target.layout().isize_width(), 32);
    assert_eq!(target.layout().pointer(0).size_bits, 32);
}

#[test]
fn no_layout_in_scope_is_an_error_that_says_how_to_state_one() {
    let error = error_of(target_of(""));
    assert!(
        mentions(
            &error,
            "`convert-to-wasm` needs a data layout: state `data_layout = \
             #builtin.data_layout<little, ptr = 32>` on the module, or compile with `--target`"
        ),
        "{error}"
    );
    assert!(mentions(&error, "no data layout is in scope of this operation"), "{error}");
}

#[test]
fn a_big_endian_layout_is_an_error() {
    let error =
        error_of(target_of("attributes { data_layout = #builtin.data_layout<big, ptr = 32> }"));
    assert!(
        mentions(&error, "`convert-to-wasm` needs a little-endian data layout"),
        "{error}"
    );
    assert!(
        mentions(&error, "the data layout in scope of this operation is big-endian"),
        "{error}"
    );
}

#[test]
fn a_layout_with_64_bit_pointers_is_an_error() {
    let error =
        error_of(target_of("attributes { data_layout = #builtin.data_layout<little, ptr = 64> }"));
    assert!(
        mentions(
            &error,
            "`convert-to-wasm` needs 32-bit pointers: 64-bit memories (memory64) are not modelled"
        ),
        "{error}"
    );
    assert!(mentions(&error, "gives pointers of 64 bits in address space 0"), "{error}");
}

#[test]
fn a_layout_whose_isize_is_not_32_bits_is_an_error() {
    let error = error_of(target_of(
        "attributes { data_layout = #builtin.data_layout<little, ptr = 32 index 16> }",
    ));
    assert!(mentions(&error, "`convert-to-wasm` needs a 32-bit `isize`"), "{error}");
    assert!(mentions(&error, "gives `isize` 16 bits"), "{error}");
}
