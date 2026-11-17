//! The plan of a block (decision 5), on functions in Wasm form written in the tests, compared as
//! text: one line per operation with its stack operands in the order they are pushed.

use alloc::{format, string::String};

use crate::{
    emit::plan::plan_block,
    tests::{function_named, module_of, parse_module},
};

/// The plan of the body of `@f`, the function of `functions` (with whatever declarations they
/// hold), as text.
#[track_caller]
fn plan_of(functions: &str) -> String {
    let (_context, module) = parse_module(&module_of(functions));
    let function = function_named(module, "f");
    let entry = {
        let op = function.borrow();
        let function = op.downcast_ref::<runic_ir::dialects::builtin::Function>().unwrap();
        function.body().entry_block_ref().unwrap()
    };
    format!("{}", plan_block(entry))
}

/// A function `@f(%x: i32, %y: i32, %z: i32) -> i32` with the body `body` and the locals
/// `locals` (a type list, or empty).
fn f(locals: &str, body: &str) -> String {
    let locals = if locals.is_empty() {
        String::new()
    } else {
        format!(" locals({locals})")
    };
    format!(
        "builtin.function public extern(\"wasm\") @f(%x: i32, %y: i32, %z: i32) -> i32{locals} \
         {{\n{body}\n}};"
    )
}

#[test]
fn a_chain_of_single_uses_stays_on_the_stack() {
    let body = "%a = wasm.mul %x, %y;\n%b = wasm.add %a, %z;\nbuiltin.ret %b : (i32);";
    assert_eq!(
        plan_of(&f("", body)),
        "0 wasm.mul [get arg0, get arg1]: on the stack\n1 wasm.add [stack %0, get arg2]: on the \
         stack\n2 builtin.ret [stack %1]: root\n"
    );
}

#[test]
fn an_operand_pushed_before_a_stacked_one_is_fetched_before_its_tree() {
    // `z - a`: `z` is fetched, then `a`'s tree (the `mul`) runs, then the `sub`.
    let body = "%a = wasm.mul %x, %y;\n%b = wasm.sub %z, %a;\nbuiltin.ret %b : (i32);";
    assert_eq!(
        plan_of(&f("", body)),
        "0 wasm.mul [get arg0, get arg1]: on the stack\n1 wasm.sub [get arg2, stack %0]: on the \
         stack\n2 builtin.ret [stack %1]: root\n"
    );
}

#[test]
fn a_value_used_by_the_next_operation_and_later_is_teed() {
    let body = "%a = wasm.mul %x, %y;\n%b = wasm.add %a, %z;\nbuiltin.store_local 0, %b;\n%c = \
                wasm.sub %a, %z;\nbuiltin.ret %c : (i32);";
    assert_eq!(
        plan_of(&f("i32", body)),
        "0 wasm.mul [get arg0, get arg1]: on the stack, tee\n1 wasm.add [stack %0, get arg2]: on \
         the stack\n2 builtin.store_local [stack %1]: root\n3 wasm.sub [get %0, get arg2]: on the \
         stack\n4 builtin.ret [stack %3]: root\n"
    );
}

#[test]
fn a_value_used_twice_by_one_operation_is_in_a_local() {
    let body = "%a = wasm.mul %x, %y;\n%b = wasm.add %a, %a;\nbuiltin.ret %b : (i32);";
    assert_eq!(
        plan_of(&f("", body)),
        "0 wasm.mul [get arg0, get arg1]: root, set\n1 wasm.add [get %0, get %0]: on the stack\n2 \
         builtin.ret [stack %1]: root\n"
    );
}

/// The case the condition on teeing exists for: `a` is used by `add` (right after it) and by
/// `sub`, the user of `add`, as an operand pushed *before* `add`'s tree.
///
/// Without the condition, `add` would claim `mul` and tee `a`, and `sub` would fetch `a`, which
/// is emitted before `add`'s tree:
///
/// ```text
/// local.get $a        ;; read before it is set
/// local.get $x
/// local.get $y
/// i32.mul
/// local.tee $a
/// local.get $z
/// i32.add
/// i32.sub
/// ```
///
/// With it, `add` fetches `a`, and `sub` itself claims `mul` with a tee: `mul` runs first.
#[test]
fn a_value_whose_other_use_is_an_earlier_operand_of_its_users_user_is_not_teed_inside() {
    let body = "%a = wasm.mul %x, %y;\n%b = wasm.add %a, %z;\n%c = wasm.sub %a, %b;\nbuiltin.ret \
                %c : (i32);";
    assert_eq!(
        plan_of(&f("", body)),
        "0 wasm.mul [get arg0, get arg1]: on the stack, tee\n1 wasm.add [get %0, get arg2]: on \
         the stack\n2 wasm.sub [stack %0, stack %1]: on the stack\n3 builtin.ret [stack %2]: \
         root\n"
    );
}

#[test]
fn a_value_whose_other_use_is_pushed_after_the_tree_that_uses_it_is_not_teed_either() {
    // `sub %b, %a` pushes `a` after `b`'s tree, where a tee inside it would be sound; the rule
    // does not reason about the order and keeps `a` in a local set by its root.
    let body = "%a = wasm.mul %x, %y;\n%b = wasm.add %a, %z;\n%c = wasm.sub %b, %a;\nbuiltin.ret \
                %c : (i32);";
    assert_eq!(
        plan_of(&f("", body)),
        "0 wasm.mul [get arg0, get arg1]: root, set\n1 wasm.add [get %0, get arg2]: on the \
         stack\n2 wasm.sub [stack %1, get %0]: on the stack\n3 builtin.ret [stack %2]: root\n"
    );
}

#[test]
fn an_operation_between_a_definition_and_its_use_puts_the_value_in_a_local() {
    let body = "%a = wasm.mul %x, %y;\nbuiltin.store_local 0, %x;\n%b = wasm.add %a, \
                %z;\nbuiltin.ret %b : (i32);";
    assert_eq!(
        plan_of(&f("i32", body)),
        "0 wasm.mul [get arg0, get arg1]: root, set\n1 builtin.store_local [get arg0]: root\n2 \
         wasm.add [get %0, get arg2]: on the stack\n3 builtin.ret [stack %2]: root\n"
    );
}

#[test]
fn the_results_of_a_call_with_two_results_are_set_into_locals() {
    let functions = format!(
        "builtin.function public extern(\"wasm\") @g(i32) -> (i32, i32);\n{}",
        f(
            "",
            "%p, %q = builtin.call @g(%x) : extern(\"wasm\") (i32) -> (i32, i32);\n%s = wasm.add \
             %p, %q;\nbuiltin.ret %s : (i32);"
        )
    );
    assert_eq!(
        plan_of(&functions),
        "0 builtin.call [get arg0]: root, set set\n1 wasm.add [get %0.0, get %0.1]: on the \
         stack\n2 builtin.ret [stack %1]: root\n"
    );
}

#[test]
fn an_unused_result_of_an_operation_that_can_trap_is_emitted_and_dropped() {
    let body = "%d = wasm.div_s %x, %y;\n%r = wasm.const 0 : i32;\nbuiltin.ret %r : (i32);";
    assert_eq!(
        plan_of(&f("", body)),
        "0 wasm.div_s [get arg0, get arg1]: root, drop\n1 wasm.const: written again\n2 \
         builtin.ret [const %1]: root\n"
    );
}

#[test]
fn a_constant_used_three_times_is_written_three_times_and_takes_no_local() {
    let body = "%k = wasm.const 3 : i32;\n%a = wasm.add %x, %k;\n%b = wasm.mul %a, %k;\n%c = \
                wasm.sub %b, %k;\nbuiltin.ret %c : (i32);";
    assert_eq!(
        plan_of(&f("", body)),
        "0 wasm.const: written again\n1 wasm.add [get arg0, const %0]: on the stack\n2 wasm.mul \
         [stack %1, const %0]: on the stack\n3 wasm.sub [stack %2, const %0]: on the stack\n4 \
         builtin.ret [stack %3]: root\n"
    );
}

#[test]
fn a_constant_between_a_definition_and_its_use_does_not_separate_them() {
    let body = "%a = wasm.mul %x, %y;\n%k = wasm.const 3 : i32;\n%b = wasm.add %a, \
                %k;\nbuiltin.ret %b : (i32);";
    assert_eq!(
        plan_of(&f("", body)),
        "0 wasm.mul [get arg0, get arg1]: on the stack\n1 wasm.const: written again\n2 wasm.add \
         [stack %0, const %1]: on the stack\n3 builtin.ret [stack %2]: root\n"
    );
}

#[test]
fn call_indirect_pushes_its_index_last_and_select_its_condition_last() {
    let functions = format!(
        "wasm.table private declaration @t : 0 x !wasm.funcref;\n{}",
        f(
            "",
            "%i = wasm.add %x, %y;\n%r = wasm.call_indirect @t, %i(%x, %z) : extern(\"wasm\") \
             (i32, i32) -> i32;\n%c = wasm.eqz %r;\n%s = wasm.select %c, %r, %x;\nbuiltin.ret %s \
             : (i32);"
        )
    );
    // The index `%i` is the last operand pushed, so the `add` right before the call stays on the
    // stack; the condition `%c` is pushed last, so the `eqz` right before the select stays, and
    // it fetches `%r`, which the select's first operand, pushed before, leaves with a tee.
    assert_eq!(
        plan_of(&functions),
        "0 wasm.add [get arg0, get arg1]: on the stack\n1 wasm.call_indirect [get arg0, get arg2, \
         stack %0]: on the stack, tee\n2 wasm.eqz [get %1]: on the stack\n3 wasm.select [stack \
         %1, get arg0, stack %2]: on the stack\n4 builtin.ret [stack %3]: root\n"
    );
}

#[test]
fn an_operation_without_results_and_a_root_with_an_unused_result_end_their_trees() {
    // The `add` is a root whose result is dropped; it takes `%a` from the stack with a tee, and
    // the `ret`, a root of its own, fetches it.
    let body = "builtin.store_local 0, %x;\n%a = builtin.load_local 0 : i32;\n%u = wasm.add %a, \
                %y;\nbuiltin.ret %a : (i32);";
    assert_eq!(
        plan_of(&f("i32", body)),
        "0 builtin.store_local [get arg0]: root\n1 builtin.load_local: on the stack, tee\n2 \
         wasm.add [stack %1, get arg1]: root, drop\n3 builtin.ret [get %1]: root\n"
    );
}
