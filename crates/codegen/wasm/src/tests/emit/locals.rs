//! The assignment of locals: live ranges over the numbering of a function's operations, reuse by
//! type, and no reuse of parameters and declared locals.

use alloc::{rc::Rc, vec::Vec};

use runic_ir::{Context, ValueRef, dialects::builtin::ModuleRef};

use crate::{
    emit::body::{BodyConfig, FunctionPlan, plan_function},
    tests::{function_named, module_of, parse_module},
};

/// The plan of `@f`, the function of `function`, with reuse of locals `reuse`, and what keeps it
/// alive.
#[track_caller]
fn planned(function: &str, reuse: bool) -> (Rc<Context>, ModuleRef, FunctionPlan) {
    let (context, module) = parse_module(&module_of(function));
    let plan = plan_function(
        function_named(module, "f"),
        &BodyConfig {
            reuse_locals: reuse,
        },
    )
    .unwrap_or_else(|report| panic!("{report}"));
    (context, module, plan)
}

/// The result of the operation at `position` of the plan.
fn result(plan: &FunctionPlan, position: usize) -> ValueRef {
    let op = plan.plan.steps[position].op.borrow();
    op.results().iter().next().unwrap().borrow().as_value_ref()
}

/// The local of the result of each of `positions`.
fn locals_of(plan: &FunctionPlan, positions: &[usize]) -> Vec<Option<u32>> {
    positions
        .iter()
        .map(|position| plan.locals.of(result(plan, *position)))
        .collect()
}

/// `a` is used last by `b`'s definition, `b` by `c`'s; each is used twice, so each is set into a
/// local.
const CHAIN: &str = "builtin.function public extern(\"wasm\") @f(%x: i32, %y: i32) -> i32 {\n%a = \
                     wasm.mul %x, %y;\n%b = wasm.add %a, %a;\n%c = wasm.sub %b, %b;\n%d = \
                     wasm.mul %c, %c;\nbuiltin.ret %d : (i32);\n};";

#[test]
fn values_of_one_type_whose_ranges_do_not_overlap_share_a_local_and_overlapping_ones_do_not() {
    let (_context, _module, plan) = planned(CHAIN, true);
    // `a` lives over 0..=1, `b` over 1..=2, `c` over 2..=3: `c` is defined after `a`'s last use
    // and takes its local; `b` is defined at `a`'s last use, so the two overlap.
    assert_eq!(locals_of(&plan, &[0, 1, 2]), [Some(2), Some(3), Some(2)]);
    assert_eq!(plan.locals.emitter().len(), 2);
    assert_eq!(plan.locals.of(result(&plan, 3)), None, "`d` stays on the stack");
}

#[test]
fn without_reuse_every_value_has_its_own_local() {
    let (_context, _module, plan) = planned(CHAIN, false);
    assert_eq!(locals_of(&plan, &[0, 1, 2]), [Some(2), Some(3), Some(4)]);
    assert_eq!(plan.locals.emitter().len(), 3);
}

#[test]
fn values_of_different_types_never_share_a_local() {
    let function = "builtin.function public extern(\"wasm\") @f(%x: i32, %y: i32) -> i32 {\n%a = \
                    wasm.mul %x, %y;\n%b = wasm.add %a, %a;\n%w = wasm.extend_u %b <{ ty = i64 \
                    }>;\n%v = wasm.add %w, %w;\n%r = wasm.wrap %v <{ ty = i32 }>;\nbuiltin.ret %r \
                    : (i32);\n};";
    let (_context, _module, plan) = planned(function, true);
    // `w` is defined after the last use of `a`, but is an `i64`.
    assert_eq!(locals_of(&plan, &[0, 2]), [Some(2), Some(3)]);
    assert_eq!(
        plan.locals.emitter(),
        [crate::emit::desc::ValType::I32, crate::emit::desc::ValType::I64]
    );
}

#[test]
fn parameters_and_declared_locals_are_never_reused() {
    let function = "builtin.function public extern(\"wasm\") @f(%x: i32, %y: i32) -> i32 \
                    locals(i32) {\n%a = wasm.add %x, %y;\n%b = wasm.mul %a, %a;\n%c = wasm.sub \
                    %b, %b;\nbuiltin.ret %c : (i32);\n};";
    let (_context, _module, plan) = planned(function, true);
    // The parameters are dead after the first operation and the declared local is never used,
    // yet the emitter's locals start after all three.
    assert_eq!(locals_of(&plan, &[0, 1]), [Some(3), Some(4)]);
    assert_eq!(plan.locals.types().len(), 3);
    assert_eq!(plan.locals.declared(0), 2);
}

#[test]
fn the_parameters_are_the_first_locals_in_order() {
    let (_context, module, plan) = planned(CHAIN, true);
    let function = function_named(module, "f");
    let entry = {
        let op = function.borrow();
        op.downcast_ref::<runic_ir::dialects::builtin::Function>()
            .unwrap()
            .body()
            .entry_block_ref()
            .unwrap()
    };
    let params = entry.borrow().argument_values().collect::<Vec<_>>();
    let locals = params.iter().map(|param| plan.locals.of(*param)).collect::<Vec<_>>();
    assert_eq!(locals, [Some(0), Some(1)], "{}", plan.plan);
}

/// The results of the operations of the body of the construct at `position` of the plan of the
/// body's block, in order (the first result of each that has one).
fn nested_results(plan: &FunctionPlan, position: usize) -> Vec<ValueRef> {
    let construct = plan.plan.steps[position].op;
    let block = construct.borrow().regions().iter().next().unwrap().entry_block_ref().unwrap();
    let nested = plan.plan_of(block).expect("the body of the construct is planned");
    nested
        .steps
        .iter()
        .filter_map(|step| {
            step.op
                .borrow()
                .results()
                .iter()
                .next()
                .map(|result| result.borrow().as_value_ref())
        })
        .collect()
}

#[test]
fn a_value_defined_before_a_loop_and_used_inside_it_keeps_its_local_to_the_loop_s_end() {
    let (_context, _module, plan) = planned(super::control::LIVE_ACROSS_LOOP, true);
    // `d` (0) dies at `d2` (1), just before the loop: `v` (2) takes its local.
    assert_eq!(locals_of(&plan, &[0, 2]), [Some(2), Some(2)], "{}", plan.plan);
    let v = plan.locals.of(result(&plan, 2));
    // No value of the loop's body (4) shares `v`'s local, although `v`'s one use in the body is
    // its first operation.
    for value in nested_results(&plan, 4) {
        let local = plan.locals.of(value);
        assert!(local.is_none() || local != v, "a value of the loop takes `v`'s local {v:?}");
    }
}

#[test]
fn the_arguments_of_a_region_take_the_local_of_an_input_whose_last_use_is_the_construct() {
    // `a` and `b` are used last as the inputs of the `if`; its first region's arguments take
    // their locals, and its second region's arguments take those of the first's.
    let function = "builtin.function public extern(\"wasm\") @f(%c: i32, %x: i32) -> i32 {\n%a = \
                    wasm.add %x, %x;\n%b = wasm.mul %a, %a;\n%r = wasm.if %c, %b then {\n^t(%p: \
                    i32):\n%q = wasm.add %p, %p;\nwasm.end %q : (i32);\n} else {\n^e(%p: \
                    i32):\n%s = wasm.sub %p, %p;\nwasm.end %s : (i32);\n} : (i32, i32) -> \
                    (i32);\nbuiltin.ret %r : (i32);\n};";
    let (_context, _module, plan) = planned(function, true);
    let construct = plan.plan.steps[2].op;
    let arguments = construct
        .borrow()
        .regions()
        .iter()
        .map(|region| {
            let block = region.entry_block_ref().unwrap();
            let argument = block.borrow().argument_values().next().unwrap();
            plan.locals.of(argument)
        })
        .collect::<Vec<_>>();
    // `a` is in local 2, `b` (used once, by the `if`, the operation right after it) on the stack.
    assert_eq!(locals_of(&plan, &[0]), [Some(2)]);
    assert_eq!(arguments, [Some(2), Some(2)], "{}", plan.plan);
    let (_context, _module, plan) = planned(function, false);
    assert_eq!(plan.locals.emitter().len(), 3, "without reuse: `a` and the two arguments");
}
