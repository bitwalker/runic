//! The translation of `scf` levels to `wasm.br` labels ([crate::convert::labels::resolve]), over
//! chains of labels written by hand, innermost first: constructs the lowering created mixed with
//! constructs written before it.

use runic_ir::RegionExitKind::{Break, Continue};

use crate::convert::labels::{
    Construct, Label,
    Label::{Frontend, Lowered as L},
    Lowered::{Inner, Loop},
    Unresolved, resolve,
};

const INNER: Label = L(Inner);
const LOOP: Label = L(Loop { wrapped: false });
const WRAPPER: Label = L(Loop { wrapped: true });

#[test]
fn a_continue_of_a_loop_with_no_block_names_the_loop() {
    // continue 1 in the `wasm.if` of the condition: the if is label 0, the loop label 1.
    assert_eq!(resolve(&[INNER, LOOP], 1, Continue), Ok(1));
}

#[test]
fn a_break_names_the_block_around_its_loop_and_a_continue_the_loop_inside_it() {
    // In an `scf.if` in the body: its if, the condition's if, the loop, the block.
    let chain = [INNER, INNER, INNER, WRAPPER];
    assert_eq!(resolve(&chain, 1, Break), Ok(3));
    assert_eq!(resolve(&chain, 1, Continue), Ok(2));
}

#[test]
fn exits_that_name_an_outer_loop_count_every_construct_they_cross() {
    // An `scf.if` in an inner loop with no block, in the body of an outer loop with a block.
    let chain = [INNER, INNER, LOOP, INNER, INNER, WRAPPER];
    assert_eq!(resolve(&chain, 2, Break), Ok(5));
    assert_eq!(resolve(&chain, 2, Continue), Ok(4));
    assert_eq!(resolve(&chain, 1, Continue), Ok(2));
}

#[test]
fn constructs_written_before_the_lowering_count_as_one_level_for_both() {
    // A lowered loop in a frontend block: break 2 leaves the block.
    let chain = [INNER, LOOP, Frontend(Construct::Block)];
    assert_eq!(resolve(&chain, 2, Break), Ok(2));
    // A frontend if inside a lowered loop counts as an `scf` level of its own.
    let chain = [Frontend(Construct::If), INNER, LOOP];
    assert_eq!(resolve(&chain, 1, Break), Ok(0));
    assert_eq!(resolve(&chain, 2, Continue), Ok(2));
    // A frontend loop is a continue target.
    let chain = [INNER, Frontend(Construct::Loop)];
    assert_eq!(resolve(&chain, 1, Continue), Ok(1));
}

#[test]
fn a_break_of_a_frontend_loop_and_a_continue_of_a_frontend_block_do_not_translate() {
    let chain = [INNER, Frontend(Construct::Loop)];
    assert_eq!(resolve(&chain, 1, Break), Err(Unresolved::BreakOfLoop));
    let chain = [Frontend(Construct::Block)];
    assert_eq!(
        resolve(&chain, 1, Continue),
        Err(Unresolved::ContinueOfNonLoop(Construct::Block))
    );
}

#[test]
fn a_break_of_a_loop_with_no_block_does_not_translate() {
    assert_eq!(resolve(&[INNER, LOOP], 1, Break), Err(Unresolved::Unwrapped));
}

#[test]
fn a_level_past_the_enclosing_loops_is_not_found() {
    assert_eq!(resolve(&[INNER, LOOP, INNER], 2, Break), Err(Unresolved::NotFound { found: 1 }));
    assert_eq!(resolve(&[], 1, Continue), Err(Unresolved::NotFound { found: 0 }));
    assert_eq!(resolve(&[LOOP], 0, Continue), Err(Unresolved::NotFound { found: 0 }));
}

#[test]
fn a_continue_in_the_block_of_a_loop_but_not_in_the_loop_is_not_found() {
    assert_eq!(resolve(&[WRAPPER], 1, Continue), Err(Unresolved::NotFound { found: 1 }));
}
