//! Branches the frontend wrote: a `wasm.br` (or the branches of a `wasm.br_table`) inside an
//! `scf.if`, `scf.while` or `scf.index_switch` names the `wasm` construct it named before the
//! lowering, although the lowering puts new constructs between them.
//!
//! Each program is lowered by `convert-to-wasm` and `wasm-verify-form`, and run by the evaluator
//! on inputs whose results are stated by hand.

use alloc::vec;

use super::{
    eval::{Outcome, Val, run},
    *,
};

/// Lowers the functions `functions` by `convert-to-wasm` and `wasm-verify-form`, and runs `@f` on
/// each input, comparing with the `i32` it should return.
#[track_caller]
fn check(functions: &str, cases: &[(&[u32], u32)]) {
    let (_context, module) = lowered_by(&module_of(functions), "convert-to-wasm,wasm-verify-form");
    let function = function_named(module, "f");
    for (inputs, expected) in cases {
        let inputs = inputs.iter().map(|value| Val::i32(*value)).collect::<Vec<_>>();
        assert_eq!(
            run(function, &inputs),
            Outcome::Values(vec![Val::i32(*expected)]),
            "inputs {inputs:?}\n{}",
            text_of(module)
        );
    }
}

/// A `wasm.br 0` in an `scf.if` names the `wasm.block` around the `scf.if`, which is no label.
const BR_IN_SCF_IF: &str = r#"
builtin.function public extern("wasm") @f(%c: i1) -> i32 {
    %r = wasm.block {
        %x = scf.if %c then {
            %v = wasm.const 7 : i32;
            wasm.br 0 %v : (i32);
        } else {
            %u = arith.constant 1 : i32;
            scf.yield %u : (i32);
        } : (i1) -> (i32);
        %h = arith.constant 100 : i32;
        %y = arith.add %x, %h <{ overflow = #builtin.overflow<wrapping> }>;
        wasm.end %y : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;

#[test]
fn a_frontend_branch_in_an_scf_if_leaves_the_block_it_named() {
    // Taken, the branch leaves the block with 7; the `wasm.if` of the `scf.if` is not its target.
    check(BR_IN_SCF_IF, &[(&[1], 7), (&[0], 101)]);
}

/// A `wasm.br 1` in a frontend `wasm.if` in the `after` body of an `scf.while` names the
/// `wasm.block` around the loop: the loop's constructs come between them after the lowering.
const BR_IN_SCF_WHILE: &str = r#"
builtin.function public extern("wasm") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = wasm.block {
        %w = scf.while %zero before {
        ^b(%i: i32):
            %more = arith.cmpi ult %i, %n;
            scf.condition %more, %i : (i1, i32);
        } after {
        ^a(%j: i32):
            %three = arith.constant 3 : i32;
            %hit = arith.cmpi eq %j, %three;
            %hz = arith.zext %hit <{ ty = i32 }>;
            wasm.if %hz then {
                %k = wasm.const 999 : i32;
                wasm.br 1 %k : (i32);
            } else {
                wasm.end : ();
            } : (i32) -> ();
            %one = arith.constant 1 : i32;
            %nj = arith.add %j, %one <{ overflow = #builtin.overflow<wrapping> }>;
            scf.yield %nj : (i32);
        } : (i32) -> (i32);
        wasm.end %w : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;

#[test]
fn a_frontend_branch_in_an_scf_while_leaves_the_block_it_named() {
    // The loop counts from 0; at 3 the branch leaves the block with 999, and below 4 the loop
    // ends first.
    check(BR_IN_SCF_WHILE, &[(&[10], 999), (&[2], 2), (&[0], 0)]);
}

/// A `wasm.br_table` in an `scf.if` in a `wasm.block`: entry 0 leaves the outer block with 7 and
/// the default leaves the inner block, after which 20 is added in the `scf.if` and 100 after it.
const BR_TABLE_IN_SCF_IF: &str = r#"
builtin.function public extern("wasm") @f(%c: i1, %s: i32) -> i32 {
    %r = wasm.block {
        %x = scf.if %c then {
            %v = wasm.const 7 : i32;
            %y = wasm.block {
                wasm.br_table %s, %v [{ wasm.br 1 %v : (i32); }] default { wasm.br 0 %v : (i32); } : (i32, i32);
            } : () -> (i32);
            %twenty = wasm.const 20 : i32;
            %z = wasm.add %y, %twenty;
            scf.yield %z : (i32);
        } else {
            %u = arith.constant 1 : i32;
            scf.yield %u : (i32);
        } : (i1) -> (i32);
        %h = arith.constant 100 : i32;
        %t = arith.add %x, %h <{ overflow = #builtin.overflow<wrapping> }>;
        wasm.end %t : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;

#[test]
fn the_branches_of_a_frontend_branch_table_in_an_scf_if_keep_their_targets() {
    check(
        BR_TABLE_IN_SCF_IF,
        &[(&[1, 0], 7), (&[1, 1], 127), (&[1, 9], 127), (&[0, 0], 101)],
    );
}

/// A frontend `wasm.loop` whose body holds an `scf.while`, whose `after` body goes back to the top
/// of the frontend loop with `wasm.br 1` (the `wasm.if` around it is label 0) while `k < 10`, and
/// steps by 2 otherwise. `extra` is put after that `wasm.if` in the `after` body.
fn continue_outer_loop(extra: &str) -> String {
    format!(
        r#"
builtin.function public extern("wasm") @f(%n: i32) -> i32 {{
    %zero = wasm.const 0 : i32;
    %r = wasm.loop %zero {{
    ^top(%i: i32):
        %w = scf.while %i before {{
        ^b(%j: i32):
            %more = arith.cmpi ult %j, %n;
            scf.condition %more, %j : (i1, i32);
        }} after {{
        ^a(%k: i32):
            %one = arith.constant 1 : i32;
            %next = arith.add %k, %one <{{ overflow = #builtin.overflow<wrapping> }}>;
            %ten = arith.constant 10 : i32;
            %small = arith.cmpi ult %k, %ten;
            %sz = arith.zext %small <{{ ty = i32 }}>;
            wasm.if %sz then {{
                wasm.br 1 %next : (i32);
            }} else {{
                wasm.end : ();
            }} : (i32) -> ();
            {extra}
            %two = arith.constant 2 : i32;
            %skip = arith.add %k, %two <{{ overflow = #builtin.overflow<wrapping> }}>;
            scf.yield %skip : (i32);
        }} : (i32) -> (i32);
        wasm.end %w : (i32);
    }} : (i32) -> (i32);
    builtin.ret %r : (i32);
}};"#
    )
}

#[test]
fn a_frontend_branch_to_a_frontend_loop_from_an_scf_while_goes_back_to_its_top() {
    // By 1 to 10 through the frontend loop, by 2 after; the loop ends at the first value not
    // below `n`.
    check(&continue_outer_loop(""), &[(&[0], 0), (&[5], 5), (&[14], 14), (&[15], 16)]);
}

#[test]
fn a_frontend_branch_to_a_frontend_loop_from_an_scf_while_wrapped_in_a_block_goes_back_to_its_top()
{
    // The break at 100 makes the lowering put a `wasm.block` around the loop it leaves, one more
    // construct between the branch and its target.
    let extra = "%hundred = arith.constant 100 : i32;\n            %stop = arith.cmpi eq %k, \
                 %hundred;\n            scf.if %stop then {\n                scf.break 1 %k : \
                 (i32);\n            } else {\n                scf.yield : ();\n            } : \
                 (i1) -> ();";
    let functions = continue_outer_loop(extra);
    let (_context, module) = lowered_by(&module_of(&functions), "convert-to-wasm");
    assert!(
        lines(&text_of(module))
            .iter()
            .filter(|line| line.contains("= wasm.block {"))
            .count()
            == 1,
        "{}",
        text_of(module)
    );
    check(&functions, &[(&[0], 0), (&[14], 14), (&[15], 16), (&[200], 100)]);
}

#[test]
fn a_frontend_branch_the_lowering_puts_no_construct_around_keeps_its_label() {
    // No `scf` construct between the branch and its target: the branch is not changed.
    let functions = "builtin.function public extern(\"wasm\") @f(%c: i32) -> i32 {\n    %r = \
                     wasm.block {\n        %v = wasm.const 7 : i32;\n        wasm.if %c then {\n  \
                     wasm.br 1 %v : (i32);\n        } else {\n            wasm.end : ();\n        \
                     } : (i32) -> ();\n        wasm.end %c : (i32);\n    } : () -> (i32);\n    \
                     builtin.ret %r : (i32);\n};";
    let (_context, module) = lowered_by(&module_of(functions), "convert-to-wasm");
    assert!(text_of(module).contains("wasm.br 1 "), "{}", text_of(module));
    check(functions, &[(&[1], 7), (&[0], 0)]);
}

#[test]
fn a_frontend_branch_that_does_not_resolve_is_reported_before_anything_changes() {
    // Not verified: the branch names label 3, and only the block encloses it.
    let context = context();
    let text = module_of(
        "builtin.function public extern(\"wasm\") @f(%c: i8) -> i8 {\n    %d = arith.add %c, %c \
         <{ overflow = #builtin.overflow<wrapping> }>;\n    wasm.block {\n        wasm.br 3 : \
         ();\n    } : () -> ();\n    builtin.ret %d : (i8);\n};",
    );
    let module = parse_unverified_in(&context, &text).unwrap();
    let before = text_of(module);
    let error = error_of(crate::convert::convert_module(module));
    assert!(
        error.contains("`convert-to-wasm` cannot lower this `wasm.br`: `wasm.br` names label 3"),
        "{error}"
    );
    assert_eq!(text_of(module), before);
}
