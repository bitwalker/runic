//! Structured control in function bodies: the text of each shape, emitted, validated and printed
//! with `wasmprinter`; and programs of loops, branches and tail calls run by `wasmi` against the
//! evaluator ([agree]), each with reuse of locals on and off.

use alloc::{format, string::String, vec::Vec};

use crate::tests::{
    eval::{Outcome, Val},
    exec::{Setup, agree, text_of_module},
    function_named, module_of, parse_module,
};

/// The text of the function `$f` of the emitted module of `functions`, from its `(func` to its
/// closing parenthesis.
#[track_caller]
fn body_of(functions: &str) -> String {
    let (_context, module) = parse_module(&module_of(functions));
    let setup = Setup {
        memory: 0,
        ..Setup::default()
    };
    let text = text_of_module(module.as_operation_ref(), &setup);
    let start = text.find("  (func $f ").unwrap_or_else(|| panic!("no `$f` in:\n{text}"));
    let end = text[start..].find("\n  )\n").unwrap_or_else(|| panic!("no end in:\n{text}"));
    String::from(&text[start..start + end + 4])
}

/// `lines` joined into one text.
fn lines(lines: &[&str]) -> String {
    lines.join("\n")
}

/// A function `@f` with the signature `signature` (after the name) and the body `body`.
fn f(signature: &str, body: &str) -> String {
    format!("builtin.function public extern(\"wasm\") @f{signature} {{\n{body}\n}};")
}

/// The `i32`s `values`.
fn i32s(values: &[u32]) -> Vec<Val> {
    values.iter().map(|value| Val::i32(*value)).collect()
}

/// Runs `@f` of `functions` on each of `cases` with [agree], and compares with the `i32`s each
/// should give.
#[track_caller]
fn agree_on(functions: &str, cases: &[(&[u32], &[u32])]) {
    let (_context, module) = parse_module(&module_of(functions));
    let function = function_named(module, "f");
    for (inputs, expected) in cases {
        assert_eq!(agree(function, &i32s(inputs)), Outcome::Values(i32s(expected)), "@f{inputs:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Text

#[test]
fn a_block_with_a_result_is_a_node_in_the_middle_of_an_expression() {
    let body = body_of(&f(
        "(%x: i32, %y: i32) -> i32",
        "%a = wasm.mul %x, %y;\n%r = wasm.block {\n%k = wasm.const 3 : i32;\n%b = wasm.add %y, \
         %k;\nwasm.end %b : (i32);\n} : () -> (i32);\n%s = wasm.sub %a, %r;\nbuiltin.ret %s : \
         (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    local.get 0",
            "    local.get 1",
            "    i32.mul",
            "    block (result i32) ;; label = @1",
            "      local.get 1",
            "      i32.const 3",
            "      i32.add",
            "    end",
            "    i32.sub",
            "  )",
        ])
    );
}

#[test]
fn an_if_with_an_else_and_a_result() {
    let body = body_of(&f(
        "(%c: i32, %x: i32, %y: i32) -> i32",
        "%r = wasm.if %c then {\n%a = wasm.add %x, %y;\nwasm.end %a : (i32);\n} else {\n%b = \
         wasm.sub %x, %y;\nwasm.end %b : (i32);\n} : (i32) -> (i32);\nbuiltin.ret %r : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32 i32) (result i32)",
            "    local.get 0",
            "    if (result i32) ;; label = @1",
            "      local.get 1",
            "      local.get 2",
            "      i32.add",
            "    else",
            "      local.get 1",
            "      local.get 2",
            "      i32.sub",
            "    end",
            "  )",
        ])
    );
}

#[test]
fn an_if_without_an_else_inputs_or_results() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32 locals(i32)",
        "builtin.store_local 0, %x;\nwasm.if %c then {\n%a = wasm.add %x, \
         %x;\nbuiltin.store_local 0, %a;\nwasm.end : ();\n} : (i32) -> ();\n%l = \
         builtin.load_local 0 : i32;\nbuiltin.ret %l : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    (local i32)",
            "    local.get 1",
            "    local.set 2",
            "    local.get 0",
            "    if ;; label = @1",
            "      local.get 1",
            "      local.get 1",
            "      i32.add",
            "      local.set 2",
            "    end",
            "    local.get 2",
            "  )",
        ])
    );
}

#[test]
fn an_if_with_inputs_sets_the_arguments_of_each_arm_and_its_results_after_its_end() {
    // The arguments of each arm are set from the last to the first; the second arm's take the
    // locals of the first's, which are dead when it runs.
    let body = body_of(&f(
        "(%c: i32, %x: i32, %y: i32) -> (i32, i32)",
        "%r, %s = wasm.if %c, %x, %y then {\n^t(%a: i32, %b: i32):\nwasm.end %b, %a : (i32, \
         i32);\n} else {\n^e(%a: i32, %b: i32):\n%d = wasm.sub %a, %b;\nwasm.end %d, %b : (i32, \
         i32);\n} : (i32, i32, i32) -> (i32, i32);\nbuiltin.ret %r, %s : (i32, i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32 i32) (result i32 i32)",
            "    (local i32 i32 i32 i32)",
            "    local.get 1",
            "    local.get 2",
            "    local.get 0",
            "    if (type 1) (param i32 i32) (result i32 i32) ;; label = @1",
            "      local.set 6",
            "      local.set 5",
            "      local.get 6",
            "      local.get 5",
            "    else",
            "      local.set 6",
            "      local.set 5",
            "      local.get 5",
            "      local.get 6",
            "      i32.sub",
            "      local.get 6",
            "    end",
            "    local.set 4",
            "    local.set 3",
            "    local.get 3",
            "    local.get 4",
            "  )",
        ])
    );
}

#[test]
fn an_if_with_an_input_and_no_else_passes_the_input_through() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32",
        "%r = wasm.if %c, %x then {\n^t(%a: i32):\n%d = wasm.mul %a, %a;\nwasm.end %d : (i32);\n} \
         : (i32, i32) -> (i32);\nbuiltin.ret %r : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    (local i32)",
            "    local.get 1",
            "    local.get 0",
            "    if (type 1) (param i32) (result i32) ;; label = @1",
            "      local.set 2",
            "      local.get 2",
            "      local.get 2",
            "      i32.mul",
            "    end",
            "  )",
        ])
    );
}

#[test]
fn a_branch_out_of_two_levels_carries_its_value() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32",
        "%r = wasm.block {\n%s = wasm.block {\nwasm.if %c then {\nwasm.br 2 %x : (i32);\n} : \
         (i32) -> ();\n%k = wasm.const 1 : i32;\nwasm.end %k : (i32);\n} : () -> (i32);\n%t = \
         wasm.add %s, %x;\nwasm.end %t : (i32);\n} : () -> (i32);\nbuiltin.ret %r : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    block (result i32) ;; label = @1",
            "      block (result i32) ;; label = @2",
            "        local.get 0",
            "        if ;; label = @3",
            "          local.get 1",
            "          br 2 (;@1;)",
            "        end",
            "        i32.const 1",
            "      end",
            "      local.get 1",
            "      i32.add",
            "    end",
            "  )",
        ])
    );
}

#[test]
fn an_if_that_only_branches_further_out_is_written_br_if() {
    // The `if` was label 0: `wasm.br 1` is `br_if 0`.
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32",
        "wasm.block {\nwasm.if %c then {\nwasm.br 1 : ();\n} : (i32) -> ();\nbuiltin.ret %x : \
         (i32);\n} : () -> ();\n%k = wasm.const 0 : i32;\nbuiltin.ret %k : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    block ;; label = @1",
            "      local.get 0",
            "      br_if 0 (;@1;)",
            "      local.get 1",
            "      return",
            "    end",
            "    i32.const 0",
            "  )",
        ])
    );
}

#[test]
fn an_if_whose_branch_carries_a_value_is_not_br_if() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32",
        "%r = wasm.block {\nwasm.if %c then {\nwasm.br 1 %x : (i32);\n} : (i32) -> ();\n%k = \
         wasm.const 0 : i32;\nwasm.end %k : (i32);\n} : () -> (i32);\nbuiltin.ret %r : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    block (result i32) ;; label = @1",
            "      local.get 0",
            "      if ;; label = @2",
            "        local.get 1",
            "        br 1 (;@1;)",
            "      end",
            "      i32.const 0",
            "    end",
            "  )",
        ])
    );
}

#[test]
fn an_if_whose_branch_names_the_if_itself_is_not_br_if() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32",
        "wasm.if %c then {\nwasm.br 0 : ();\n} : (i32) -> ();\nbuiltin.ret %x : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    local.get 0",
            "    if ;; label = @1",
            "      br 0 (;@1;)",
            "    end",
            "    local.get 1",
            "  )",
        ])
    );
}

#[test]
fn an_if_whose_else_does_something_is_not_br_if() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32 locals(i32)",
        "wasm.block {\nwasm.if %c then {\nwasm.br 1 : ();\n} else {\nbuiltin.store_local 0, \
         %x;\nwasm.end : ();\n} : (i32) -> ();\nwasm.end : ();\n} : () -> ();\n%l = \
         builtin.load_local 0 : i32;\nbuiltin.ret %l : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    (local i32)",
            "    block ;; label = @1",
            "      local.get 0",
            "      if ;; label = @2",
            "        br 1 (;@1;)",
            "      else",
            "        local.get 1",
            "        local.set 2",
            "      end",
            "    end",
            "    local.get 2",
            "  )",
        ])
    );
}

#[test]
fn an_if_whose_arm_holds_more_than_the_branch_is_not_br_if() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32 locals(i32)",
        "wasm.block {\nwasm.if %c then {\nbuiltin.store_local 0, %x;\nwasm.br 1 : ();\n} : (i32) \
         -> ();\nwasm.end : ();\n} : () -> ();\n%l = builtin.load_local 0 : i32;\nbuiltin.ret %l \
         : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    (local i32)",
            "    block ;; label = @1",
            "      local.get 0",
            "      if ;; label = @2",
            "        local.get 1",
            "        local.set 2",
            "        br 1 (;@1;)",
            "      end",
            "    end",
            "    local.get 2",
            "  )",
        ])
    );
}

#[test]
fn a_branch_table_with_values_pushes_them_then_its_index_and_takes_each_level() {
    let body = body_of(&f(
        "(%i: i32, %x: i32) -> i32",
        "%r = wasm.block {\n%s = wasm.block {\n%t = wasm.block {\n%u = wasm.block \
         {\nwasm.br_table %i, %x [{ wasm.br 0 %x : (i32); }, { wasm.br 1 %x : (i32); }, { wasm.br \
         2 %x : (i32); }] default { wasm.br 3 %x : (i32); } : (i32, i32);\n} : () -> (i32);\n%a = \
         wasm.const 1 : i32;\n%b = wasm.add %u, %a;\nwasm.end %b : (i32);\n} : () -> (i32);\n%c = \
         wasm.const 2 : i32;\n%d = wasm.mul %t, %c;\nwasm.end %d : (i32);\n} : () -> (i32);\n%e = \
         wasm.const 3 : i32;\n%g = wasm.sub %s, %e;\nwasm.end %g : (i32);\n} : () -> \
         (i32);\nbuiltin.ret %r : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    block (result i32) ;; label = @1",
            "      block (result i32) ;; label = @2",
            "        block (result i32) ;; label = @3",
            "          block (result i32) ;; label = @4",
            "            local.get 1",
            "            local.get 0",
            "            br_table 0 (;@4;) 1 (;@3;) 2 (;@2;) 3 (;@1;)",
            "          end",
            "          i32.const 1",
            "          i32.add",
            "        end",
            "        i32.const 2",
            "        i32.mul",
            "      end",
            "      i32.const 3",
            "      i32.sub",
            "    end",
            "  )",
        ])
    );
}

#[test]
fn a_branch_table_without_values_takes_the_levels_of_its_entries_then_the_default() {
    let body = body_of(&f(
        "(%i: i32) -> i32",
        "wasm.block {\nwasm.block {\nwasm.block {\nwasm.block {\nwasm.br_table %i [{ wasm.br 0 : \
         (); }, { wasm.br 1 : (); }, { wasm.br 2 : (); }] default { wasm.br 3 : (); } : (i32);\n} \
         : () -> ();\n%a = wasm.const 10 : i32;\nbuiltin.ret %a : (i32);\n} : () -> ();\n%b = \
         wasm.const 20 : i32;\nbuiltin.ret %b : (i32);\n} : () -> ();\n%c = wasm.const 30 : \
         i32;\nbuiltin.ret %c : (i32);\n} : () -> ();\n%d = wasm.const 40 : i32;\nbuiltin.ret %d \
         : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32) (result i32)",
            "    block ;; label = @1",
            "      block ;; label = @2",
            "        block ;; label = @3",
            "          block ;; label = @4",
            "            local.get 0",
            "            br_table 0 (;@4;) 1 (;@3;) 2 (;@2;) 3 (;@1;)",
            "          end",
            "          i32.const 10",
            "          return",
            "        end",
            "        i32.const 20",
            "        return",
            "      end",
            "      i32.const 30",
            "      return",
            "    end",
            "    i32.const 40",
            "  )",
        ])
    );
}

#[test]
fn returns_in_both_arms_of_an_if_are_written_return_and_unreachable_follows() {
    let body = body_of(&f(
        "(%c: i32, %x: i32, %y: i32) -> i32",
        "wasm.if %c then {\nbuiltin.ret %x : (i32);\n} else {\nbuiltin.ret %y : (i32);\n} : (i32) \
         -> ();\nwasm.unreachable;",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32 i32) (result i32)",
            "    local.get 0",
            "    if ;; label = @1",
            "      local.get 1",
            "      return",
            "    else",
            "      local.get 2",
            "      return",
            "    end",
            "    unreachable",
            "  )",
        ])
    );
}

#[test]
fn a_tail_call_inside_an_if() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32",
        "wasm.if %c then {\nbuiltin.tail_call @f(%x, %x) : extern(\"wasm\") (i32, i32) -> i32;\n} \
         : (i32) -> ();\nbuiltin.ret %x : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    local.get 0",
            "    if ;; label = @1",
            "      local.get 1",
            "      local.get 1",
            "      return_call $f",
            "    end",
            "    local.get 1",
            "  )",
        ])
    );
}

#[test]
fn a_constant_used_at_three_depths_is_written_three_times_and_takes_no_local() {
    let body = body_of(&f(
        "(%c: i32, %x: i32) -> i32",
        "%k = wasm.const 5 : i32;\n%a = wasm.add %x, %k;\n%r = wasm.block {\n%b = wasm.add %a, \
         %k;\n%s = wasm.if %c then {\n%d = wasm.mul %b, %k;\nwasm.end %d : (i32);\n} else \
         {\nwasm.end %b : (i32);\n} : (i32) -> (i32);\nwasm.end %s : (i32);\n} : () -> \
         (i32);\nbuiltin.ret %r : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    (local i32 i32)",
            "    local.get 1",
            "    i32.const 5",
            "    i32.add",
            "    local.set 2",
            "    block (result i32) ;; label = @1",
            "      local.get 2",
            "      i32.const 5",
            "      i32.add",
            "      local.set 3",
            "      local.get 0",
            "      if (result i32) ;; label = @2",
            "        local.get 3",
            "        i32.const 5",
            "        i32.mul",
            "      else",
            "        local.get 3",
            "      end",
            "    end",
            "  )",
        ])
    );
}

#[test]
fn a_value_teed_into_an_if_is_read_inside_its_arm_after_the_tee() {
    // `a` is the condition and is used again inside the first arm: it is teed before the `if`
    // runs, so the arm reads it after it was set.
    let body = body_of(&f(
        "(%x: i32, %y: i32) -> i32",
        "%a = wasm.mul %x, %y;\n%r = wasm.if %a then {\n%b = wasm.add %a, %x;\nwasm.end %b : \
         (i32);\n} else {\nwasm.end %y : (i32);\n} : (i32) -> (i32);\nbuiltin.ret %r : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    (local i32)",
            "    local.get 0",
            "    local.get 1",
            "    i32.mul",
            "    local.tee 2",
            "    if (result i32) ;; label = @1",
            "      local.get 2",
            "      local.get 0",
            "      i32.add",
            "    else",
            "      local.get 1",
            "    end",
            "  )",
        ])
    );
}

// ---------------------------------------------------------------------------------------------
// Programs, against the evaluator

/// A loop that swaps two variables on each iteration: a branch back to the loop with its
/// arguments crossed, which assigning locals before the branch would get wrong without a parallel
/// copy.
pub(super) const SWAP: &str = "builtin.function public extern(\"wasm\") @f(%n: i32, %a: i32, %b: \
                               i32) -> (i32, i32) {
    %z = wasm.const 0 : i32;
    %r, %s = wasm.loop %z, %a, %b {
    ^l(%i: i32, %x: i32, %y: i32):
        %done = wasm.ge_u %i, %n;
        %o, %p = wasm.if %done then {
            wasm.end %x, %y : (i32, i32);
        } else {
            %one = wasm.const 1 : i32;
            %i1 = wasm.add %i, %one;
            wasm.br 1 %i1, %y, %x : (i32, i32, i32);
        } : (i32) -> (i32, i32);
        wasm.end %o, %p : (i32, i32);
    } : (i32, i32, i32) -> (i32, i32);
    builtin.ret %r, %s : (i32, i32);
};";

#[test]
fn a_loop_that_swaps_two_variables_on_each_iteration_agrees() {
    agree_on(
        SWAP,
        &[
            (&[0, 10, 20], &[10, 20]),
            (&[1, 10, 20], &[20, 10]),
            (&[2, 10, 20], &[10, 20]),
            (&[7, 10, 20], &[20, 10]),
        ],
    );
}

/// `%v` is defined before the loop and used at the top of its body, on every iteration; `%d`, of
/// the same type, dies just before the loop, so `%v` takes its local; `%i1` and `%w`, defined
/// later in the body, would take `%v`'s local if its live range ended at its use in the body.
pub(super) const LIVE_ACROSS_LOOP: &str = "builtin.function public extern(\"wasm\") @f(%n: i32, \
                                           %a: i32) -> i32 {
    %d = wasm.sub %n, %a;
    %d2 = wasm.mul %d, %d;
    %v = wasm.mul %a, %a;
    %z = wasm.const 0 : i32;
    %r = wasm.loop %z, %d2 {
    ^l(%i: i32, %acc: i32):
        %t = wasm.add %acc, %v;
        %one = wasm.const 1 : i32;
        %i1 = wasm.add %i, %one;
        %w = wasm.mul %i1, %i1;
        %w2 = wasm.add %w, %w;
        %acc2 = wasm.add %t, %w2;
        %more = wasm.lt_u %i1, %n;
        %res = wasm.if %more then {
            wasm.br 1 %i1, %acc2 : (i32, i32);
        } else {
            wasm.end %acc2 : (i32);
        } : (i32) -> (i32);
        wasm.end %res : (i32);
    } : (i32, i32) -> (i32);
    builtin.ret %r : (i32);
};";

#[test]
fn a_value_defined_before_a_loop_keeps_its_local_until_the_loop_s_second_iteration_reads_it() {
    let model = |n: u32, a: u32| {
        let d = n.wrapping_sub(a);
        let v = a.wrapping_mul(a);
        let mut acc = d.wrapping_mul(d);
        let mut i = 0u32;
        loop {
            let t = acc.wrapping_add(v);
            let i1 = i.wrapping_add(1);
            acc = t.wrapping_add(i1.wrapping_mul(i1).wrapping_mul(2));
            if i1 >= n {
                return acc;
            }
            i = i1;
        }
    };
    let cases = [(0, 3), (1, 3), (2, 3), (5, 7), (9, 2)]
        .map(|(n, a)| ([n, a], [model(n, a)]))
        .to_vec();
    let cases = cases.iter().map(|(i, o)| (&i[..], &o[..])).collect::<Vec<_>>();
    agree_on(LIVE_ACROSS_LOOP, &cases);
}

/// `%kk` is defined before two nested loops and used in the inner one; values defined in the
/// inner loop after its use, and in the outer loop after the inner one, need locals of the same
/// type.
const NESTED_LOOPS: &str = "builtin.function public extern(\"wasm\") @f(%n: i32, %m: i32, %k: \
                            i32) -> i32 {
    %kk = wasm.mul %k, %k;
    %z = wasm.const 0 : i32;
    %r = wasm.loop %z, %z {
    ^o(%i: i32, %acc: i32):
        %inner = wasm.loop %z, %acc {
        ^in(%j: i32, %a: i32):
            %a1 = wasm.add %a, %kk;
            %one = wasm.const 1 : i32;
            %j1 = wasm.add %j, %one;
            %w = wasm.mul %j1, %j1;
            %w2 = wasm.sub %w, %w;
            %a2 = wasm.add %a1, %w2;
            %more = wasm.lt_u %j1, %m;
            %res = wasm.if %more then {
                wasm.br 1 %j1, %a2 : (i32, i32);
            } else {
                wasm.end %a2 : (i32);
            } : (i32) -> (i32);
            wasm.end %res : (i32);
        } : (i32, i32) -> (i32);
        %uno = wasm.const 1 : i32;
        %i1 = wasm.add %i, %uno;
        %x = wasm.mul %i1, %i1;
        %x2 = wasm.sub %x, %x;
        %acc2 = wasm.add %inner, %x2;
        %again = wasm.lt_u %i1, %n;
        %out = wasm.if %again then {
            wasm.br 1 %i1, %acc2 : (i32, i32);
        } else {
            wasm.end %acc2 : (i32);
        } : (i32) -> (i32);
        wasm.end %out : (i32);
    } : (i32, i32) -> (i32);
    builtin.ret %r : (i32);
};";

#[test]
fn a_loop_nested_in_a_loop_reads_a_value_from_outside_both_on_every_iteration() {
    // Each loop runs at least once: the result is max(n, 1) * max(m, 1) * k * k.
    let model =
        |n: u32, m: u32, k: u32| n.max(1).wrapping_mul(m.max(1)).wrapping_mul(k.wrapping_mul(k));
    let cases = [(0, 0, 3), (1, 1, 3), (2, 3, 5), (4, 2, 7), (3, 5, 1)]
        .map(|(n, m, k)| ([n, m, k], [model(n, m, k)]))
        .to_vec();
    let cases = cases.iter().map(|(i, o)| (&i[..], &o[..])).collect::<Vec<_>>();
    agree_on(NESTED_LOOPS, &cases);
}

/// The first `i` below `max(n, 1)` whose square is at least `t`, found by a loop that leaves the
/// block around it with `wasm.br 2` from inside an `if`; 999 when there is none.
const BREAK_OUT_OF_LOOP: &str = "builtin.function public extern(\"wasm\") @f(%n: i32, %t: i32) -> \
                                 i32 {
    %r = wasm.block {
        %z = wasm.const 0 : i32;
        %l = wasm.loop %z {
        ^l(%i: i32):
            %sq = wasm.mul %i, %i;
            %hit = wasm.ge_u %sq, %t;
            wasm.if %hit then {
                wasm.br 2 %i : (i32);
            } : (i32) -> ();
            %one = wasm.const 1 : i32;
            %i1 = wasm.add %i, %one;
            %more = wasm.lt_u %i1, %n;
            wasm.if %more then {
                wasm.br 1 %i1 : (i32);
            } : (i32) -> ();
            %none = wasm.const 999 : i32;
            wasm.end %none : (i32);
        } : (i32) -> (i32);
        wasm.end %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};";

#[test]
fn a_branch_from_inside_an_if_carries_a_value_out_of_a_loop() {
    agree_on(
        BREAK_OUT_OF_LOOP,
        &[
            (&[10, 0], &[0]),
            (&[10, 1], &[1]),
            (&[10, 17], &[5]),
            (&[10, 81], &[9]),
            (&[10, 82], &[999]),
            (&[0, 5], &[999]),
        ],
    );
}

/// The sum of `1..=n` plus `acc`, by a tail call of itself in the second arm of an `if`.
const TAIL_RECURSION: &str = "builtin.function public extern(\"wasm\") @f(%n: i32, %acc: i32) -> \
                              i32 {
    %z = wasm.const 0 : i32;
    %done = wasm.eq %n, %z;
    wasm.if %done then {
        builtin.ret %acc : (i32);
    } else {
        %one = wasm.const 1 : i32;
        %n1 = wasm.sub %n, %one;
        %a1 = wasm.add %acc, %n;
        builtin.tail_call @f(%n1, %a1) : extern(\"wasm\") (i32, i32) -> i32;
    } : (i32) -> ();
    wasm.unreachable;
};";

#[test]
fn a_recursion_ten_thousand_deep_through_a_tail_call_in_an_if_does_not_grow_the_stack() {
    agree_on(
        TAIL_RECURSION,
        &[(&[0, 4], &[4]), (&[3, 0], &[6]), (&[10_000, 0], &[50_005_000])],
    );
}

#[test]
fn a_value_teed_into_an_if_and_read_in_its_arm_agrees() {
    let functions = f(
        "(%x: i32, %y: i32) -> i32",
        "%a = wasm.mul %x, %y;\n%r = wasm.if %a then {\n%b = wasm.add %a, %x;\nwasm.end %b : \
         (i32);\n} else {\nwasm.end %y : (i32);\n} : (i32) -> (i32);\nbuiltin.ret %r : (i32);",
    );
    agree_on(&functions, &[(&[3, 4], &[15]), (&[0, 4], &[4]), (&[5, 0], &[0])]);
}

/// The lowered `@sum` of `tests/lit/codegen/wasm/programs.runic`, as the four passes leave it: a
/// loop of three variables whose `if` gives four results.
const SUM: &str = "wasm.memory private declaration @__linear_memory : 0;
builtin.function public extern(\"C\") @f(%42: i32, %43: i32) -> i32 {
    %46 = wasm.const 1 : i32;
    %47 = wasm.const 0 : i32;
    %48 = wasm.const 0 : i32;
    %50 = wasm.const 0 : i32;
    %52 = wasm.const 1 : i32;
    %54 = wasm.const 0 : i32;
    %55 = wasm.const 0 : i32;
    %56 = wasm.const 0 : i32;
    %58, %59, %60 = wasm.loop %56, %55, %47 {
    ^block14(%61: i32, %62: i32, %63: i32):
        %68 = wasm.ge_u %61, %43;
        %70, %71, %72, %73 = wasm.if %68 then {
            wasm.end %48, %47, %46, %50 : (i32, i32, i32, i32);
        } else {
            %77 = wasm.const 4 : i32;
            %78 = wasm.mul %61, %77;
            %79 = wasm.add %42, %78;
            %81 = wasm.load @__linear_memory, %79 : i32;
            %82 = wasm.add %62, %81;
            %83 = wasm.const 1 : i32;
            %85 = wasm.add %61, %83;
            wasm.end %85, %82, %54, %52 : (i32, i32, i32, i32);
        } : (i32) -> (i32, i32, i32, i32);
        wasm.if %73 then {
            wasm.br 1 %70, %71, %62 : (i32, i32, i32);
        } : (i32) -> ();
        wasm.end %70, %71, %62 : (i32, i32, i32);
    } : (i32, i32, i32) -> (i32, i32, i32);
    builtin.ret %60 : (i32);
};";

#[test]
fn the_lowered_sum_a_loop_of_three_variables_in_full() {
    // The loop's arguments are set at its top (the unused third dropped); the comparison, the
    // address arithmetic and the load stay on the stack; the four results of the `if` (one
    // dropped), the running total and the loop's result take locals; `br 1` goes back to the loop
    // with three values.
    assert_eq!(
        body_of(SUM),
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    (local i32 i32 i32 i32 i32 i32 i32)",
            "    i32.const 0",
            "    i32.const 0",
            "    i32.const 0",
            "    loop (type 1) (param i32 i32 i32) (result i32 i32 i32) ;; label = @1",
            "      drop",
            "      local.set 4",
            "      local.set 3",
            "      local.get 3",
            "      local.get 1",
            "      i32.ge_u",
            "      if (type 2) (result i32 i32 i32 i32) ;; label = @2",
            "        i32.const 0",
            "        i32.const 0",
            "        i32.const 1",
            "        i32.const 0",
            "      else",
            "        local.get 4",
            "        local.get 0",
            "        local.get 3",
            "        i32.const 4",
            "        i32.mul",
            "        i32.add",
            "        i32.load",
            "        i32.add",
            "        local.set 8",
            "        local.get 3",
            "        i32.const 1",
            "        i32.add",
            "        local.get 8",
            "        i32.const 0",
            "        i32.const 1",
            "      end",
            "      local.set 7",
            "      drop",
            "      local.set 6",
            "      local.set 5",
            "      local.get 7",
            "      if ;; label = @2",
            "        local.get 5",
            "        local.get 6",
            "        local.get 4",
            "        br 1 (;@1;)",
            "      end",
            "      local.get 5",
            "      local.get 6",
            "      local.get 4",
            "    end",
            "    local.set 2",
            "    drop",
            "    drop",
            "    local.get 2",
            "  )",
        ])
    );
}

#[test]
fn a_br_if_and_a_branch_table_agree() {
    let br_if = f(
        "(%c: i32, %x: i32) -> i32",
        "wasm.block {\nwasm.if %c then {\nwasm.br 1 : ();\n} : (i32) -> ();\nbuiltin.ret %x : \
         (i32);\n} : () -> ();\n%k = wasm.const 0 : i32;\nbuiltin.ret %k : (i32);",
    );
    agree_on(&br_if, &[(&[0, 7], &[7]), (&[1, 7], &[0]), (&[u32::MAX, 7], &[0])]);
    let table = f(
        "(%i: i32, %x: i32) -> i32",
        "%r = wasm.block {\n%s = wasm.block {\n%t = wasm.block {\n%u = wasm.block \
         {\nwasm.br_table %i, %x [{ wasm.br 0 %x : (i32); }, { wasm.br 1 %x : (i32); }, { wasm.br \
         2 %x : (i32); }] default { wasm.br 3 %x : (i32); } : (i32, i32);\n} : () -> (i32);\n%a = \
         wasm.const 1 : i32;\n%b = wasm.add %u, %a;\nwasm.end %b : (i32);\n} : () -> (i32);\n%c = \
         wasm.const 2 : i32;\n%d = wasm.mul %t, %c;\nwasm.end %d : (i32);\n} : () -> (i32);\n%e = \
         wasm.const 3 : i32;\n%g = wasm.sub %s, %e;\nwasm.end %g : (i32);\n} : () -> \
         (i32);\nbuiltin.ret %r : (i32);",
    );
    // Entry 0: ((x + 1) * 2) - 3; entry 1: x * 2 - 3; entry 2: x - 3; the default: x.
    agree_on(
        &table,
        &[
            (&[0, 10], &[19]),
            (&[1, 10], &[17]),
            (&[2, 10], &[7]),
            (&[3, 10], &[10]),
            (&[u32::MAX, 10], &[10]),
        ],
    );
}
