//! The emitted modules of 6c's straight-line lowered functions, run by `wasmi` against the
//! evaluator ([agree](crate::tests::exec::agree)).
//!
//! Two sources, found by trying: the operations of 6c's differential tests of the numeric
//! patterns (`int.rs`, `float.rs`, `composed.rs`), rebuilt here at every type and lowered as
//! they lower them; and every function of 6c's whole programs and of the lit files of the
//! lowering whose functions it evaluates (`programs*.runic`, `indirect-calls.runic`,
//! `frames-pipeline.runic`, `numeric-*.runic`), lowered by each file's pipelines. A function is
//! run when it, and every function it can reach, holds no control (Task 3 emits control); it is
//! skipped and counted otherwise. The counts are asserted, so that a change in the lowering shows.

use alloc::{format, string::String, vec, vec::Vec};

use runic_dialect_arith::{CmpFPredicate, CmpPredicate};
use runic_dialect_wasm as wasm;
use runic_ir::{
    Forward, Int, OperationRef, RawWalk, Report, Symbol as _, WalkResult,
    dialects::builtin::{Call, Function, GlobalVariable, TailCall},
};

use crate::{
    emit::body::has_control,
    tests::{
        context,
        eval::Val,
        exec::{Cache, PAGE, Setup, agree_cached, ops_of},
        int::{Ty, WIDTHS, edges, lower_text, product, types},
        parse_module, run_pipeline,
    },
};

/// The operations of the `wasm` dialect and `builtin` that the evaluator interprets; a function
/// that holds another is not compared.
const EVALUATED: [&str; 71] = [
    "wasm.const",
    "wasm.constf",
    "wasm.select",
    "wasm.add",
    "wasm.sub",
    "wasm.mul",
    "wasm.div_s",
    "wasm.div_u",
    "wasm.rem_s",
    "wasm.rem_u",
    "wasm.and",
    "wasm.or",
    "wasm.xor",
    "wasm.shl",
    "wasm.shr_s",
    "wasm.shr_u",
    "wasm.rotl",
    "wasm.rotr",
    "wasm.clz",
    "wasm.ctz",
    "wasm.popcnt",
    "wasm.div",
    "wasm.neg",
    "wasm.eqz",
    "wasm.eq",
    "wasm.ne",
    "wasm.lt_s",
    "wasm.lt_u",
    "wasm.gt_s",
    "wasm.gt_u",
    "wasm.le_s",
    "wasm.le_u",
    "wasm.ge_s",
    "wasm.ge_u",
    "wasm.lt",
    "wasm.gt",
    "wasm.le",
    "wasm.ge",
    "wasm.wrap",
    "wasm.extend_s",
    "wasm.extend_u",
    "wasm.extend8_s",
    "wasm.extend16_s",
    "wasm.extend32_s",
    "wasm.trunc_sat_s",
    "wasm.trunc_sat_u",
    "wasm.convert_s",
    "wasm.convert_u",
    "wasm.promote",
    "wasm.demote",
    "wasm.reinterpret",
    "wasm.load",
    "wasm.store",
    "wasm.memory_copy",
    "wasm.memory_fill",
    "wasm.global_get",
    "wasm.global_set",
    "wasm.data_addr",
    "wasm.func_index",
    "wasm.call_indirect",
    "wasm.return_call_indirect",
    "wasm.unreachable",
    "builtin.call",
    "builtin.tail_call",
    "builtin.ret",
    "builtin.load_local",
    "builtin.store_local",
    // Control, which a function that is run does not hold.
    "wasm.block",
    "wasm.loop",
    "wasm.if",
    "wasm.end",
];

/// Why a function is not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Skip {
    /// It, or a function it can reach, holds control.
    Control,
    /// It can reach a declaration, which the evaluator cannot call.
    Declaration,
    /// It holds an operation the evaluator does not interpret.
    Unevaluated,
}

/// What a sweep ran and skipped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    /// Functions run.
    functions: usize,
    /// Cases run (a function on one tuple of arguments).
    cases: usize,
    /// Functions skipped because they hold control.
    control: usize,
    /// Functions skipped because they reach a declaration.
    declaration: usize,
    /// Functions skipped because they hold an operation the evaluator does not interpret.
    unevaluated: usize,
}

impl Counts {
    fn skip(&mut self, why: Skip) {
        match why {
            Skip::Control => self.control += 1,
            Skip::Declaration => self.declaration += 1,
            Skip::Unevaluated => self.unevaluated += 1,
        }
    }
}

/// The body block of the function `function`.
fn entry(function: OperationRef) -> runic_ir::BlockRef {
    let op = function.borrow();
    op.downcast_ref::<Function>().unwrap().body().entry_block_ref().unwrap()
}

/// The functions `function` calls, tail-calls or names by `wasm.func_index` or `wasm.ref_func`.
fn callees(function: OperationRef) -> Vec<OperationRef> {
    let mut found = Vec::new();
    let _ = function.raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult<Report> {
        let op = op.borrow();
        let resolved = if let Some(call) = op.downcast_ref::<Call>() {
            call.callee().resolve_callable().ok()
        } else if let Some(call) = op.downcast_ref::<TailCall>() {
            call.callee().resolve_callable().ok()
        } else if let Some(index) = op.downcast_ref::<wasm::FuncIndex>() {
            index.callee().resolve_callable().ok()
        } else if let Some(reference) = op.downcast_ref::<wasm::RefFunc>() {
            reference.callee().resolve_callable().ok()
        } else {
            None
        };
        if let Some(callee) = resolved.and_then(|callee| callee.target().as_function()) {
            found.push(callee.as_operation_ref());
        }
        WalkResult::Continue(())
    });
    found
}

/// Whether `function` is run, or why not; the functions of its module that it cannot reach are
/// added to `stubs` by name (their bodies are not emitted).
fn runnable(function: OperationRef) -> Result<Vec<String>, Skip> {
    let mut reached = vec![function];
    let mut next = 0;
    while next < reached.len() {
        let current = reached[next];
        next += 1;
        if current.borrow().downcast_ref::<Function>().unwrap().is_declaration() {
            return Err(Skip::Declaration);
        }
        if has_control(entry(current)) {
            return Err(Skip::Control);
        }
        let mut unknown = false;
        let _ = current.raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult<Report> {
            if op != current && !EVALUATED.contains(&format!("{}", op.borrow().name()).as_str()) {
                unknown = true;
            }
            WalkResult::Continue(())
        });
        if unknown {
            return Err(Skip::Unevaluated);
        }
        for callee in callees(current) {
            if !reached.contains(&callee) {
                reached.push(callee);
            }
        }
    }
    let module = function.borrow().parent_op().unwrap();
    Ok(ops_of(module)
        .into_iter()
        .filter(|op| {
            op.borrow().downcast_ref::<Function>().is_some_and(|f| !f.is_declaration())
                && !reached.contains(op)
        })
        .map(|op| op.borrow().downcast_ref::<Function>().unwrap().get_name().as_str().into())
        .collect())
}

/// Runs `function` on `cases` under `setup` both ways when it is runnable; counts it.
#[track_caller]
fn sweep(counts: &mut Counts, setup: &Setup, function: OperationRef, cases: &[Vec<Val>]) {
    match runnable(function) {
        Err(why) => counts.skip(why),
        Ok(stubs) => {
            let setup = Setup {
                stubs,
                ..setup.clone()
            };
            let mut cache = Cache::default();
            for case in cases {
                agree_cached(&mut cache, &setup, function, case);
            }
            counts.functions += 1;
            counts.cases += cases.len();
        }
    }
}

/// The value of the integer `value` of the type `ty`, as the evaluator takes it.
fn int_val(ty: &Ty, value: Int) -> Val {
    Val::new(value.unsigned() as u64, ty.carrier())
}

/// The edge values of `ty`.
fn int_vals(ty: &Ty) -> Vec<Val> {
    edges(ty.n).into_iter().map(|value| int_val(ty, value)).collect()
}

/// Lowers the function of `params`, `results` and `body` as 6c's numeric tests do, and sweeps it
/// on `cases`.
#[track_caller]
fn sweep_text(
    counts: &mut Counts,
    context: &alloc::rc::Rc<runic_ir::Context>,
    params: &[&str],
    results: &[&str],
    body: &str,
    cases: &[Vec<Val>],
) {
    let lowered = lower_text(context, params, results, body).unwrap_or_else(|report| {
        panic!("`{body}`: {}", runic_ir::testing::text::describe(&report))
    });
    sweep(counts, &Setup::default(), lowered.function, cases);
}

#[test]
fn the_integer_operations_of_6c_agree_where_they_are_straight_line() {
    let context = context();
    let mut counts = Counts::default();
    let predicates = [
        CmpPredicate::Eq,
        CmpPredicate::Ne,
        CmpPredicate::Slt,
        CmpPredicate::Sle,
        CmpPredicate::Sgt,
        CmpPredicate::Sge,
        CmpPredicate::Ult,
        CmpPredicate::Ule,
        CmpPredicate::Ugt,
        CmpPredicate::Uge,
    ];
    for ty in types() {
        let t = ty.name.as_str();
        let pairs = product(&[int_vals(&ty), int_vals(&ty)]);
        let singles = int_vals(&ty).into_iter().map(|v| vec![v]).collect::<Vec<_>>();
        let mut binary = Vec::new();
        for op in ["add", "sub", "mul"] {
            binary.push(format!("arith.{op} %a, %b"));
            for mode in
                ["unchecked signed", "unchecked unsigned", "checked signed", "checked unsigned"]
            {
                binary.push(format!(
                    "arith.{op} %a, %b <{{ overflow = #builtin.overflow<{mode}> }}>"
                ));
            }
        }
        for op in ["band", "bor", "bxor", "udiv", "sdiv", "urem", "srem", "shl", "shr", "ashr"] {
            binary.push(format!("arith.{op} %a, %b"));
        }
        for op in ["rotl", "rotr"] {
            binary.push(format!("arith.{op} %a, %b"));
        }
        for op in ["min", "max"] {
            for sign in ["signed", "unsigned"] {
                binary.push(format!(
                    "arith.{op} %a, %b <{{ signedness = #builtin.signedness<{sign}> }}>"
                ));
            }
        }
        for body in &binary {
            sweep_text(&mut counts, &context, &[t, t], &[t], body, &pairs);
        }
        for op in ["add", "sub", "mul"] {
            for sign in ["signed", "unsigned"] {
                let body = format!(
                    "arith.{op}_overflowing %a, %b <{{ signedness = #builtin.signedness<{sign}> \
                     }}>"
                );
                sweep_text(&mut counts, &context, &[t, t], &["i1", t], &body, &pairs);
            }
        }
        for op in ["udivrem", "sdivrem"] {
            let body = format!("arith.{op} %a, %b");
            sweep_text(&mut counts, &context, &[t, t], &[t, t], &body, &pairs);
        }
        for op in ["bnot", "neg", "clz", "ctz", "clo", "cto", "popcnt"] {
            sweep_text(&mut counts, &context, &[t], &[t], &format!("arith.{op} %a"), &singles);
        }
        for predicate in predicates {
            let body = format!("arith.cmpi {} %a, %b", predicate.as_str());
            sweep_text(&mut counts, &context, &[t, t], &["i1"], &body, &pairs);
        }
        let flag = Ty::int(1);
        let selects = product(&[int_vals(&flag), int_vals(&ty), int_vals(&ty)]);
        sweep_text(&mut counts, &context, &["i1", t, t], &[t], "cf.select %a, %b, %c", &selects);
        for value in edges(ty.n) {
            let body = format!("arith.constant {} : {t}", value.unsigned());
            sweep_text(&mut counts, &context, &[], &[t], &body, &[vec![]]);
        }
    }
    for from in WIDTHS {
        for to in WIDTHS {
            let (from, to) = (Ty::int(from), Ty::int(to));
            let ops: &[&str] = if from.n > to.n {
                &["trunc"]
            } else if from.n < to.n {
                &["zext", "sext"]
            } else {
                &[]
            };
            let singles = int_vals(&from).into_iter().map(|v| vec![v]).collect::<Vec<_>>();
            for op in ops {
                let body = format!("arith.{op} %a <{{ ty = {} }}>", to.name);
                sweep_text(&mut counts, &context, &[&from.name], &[&to.name], &body, &singles);
            }
        }
    }
    let assert = ["i1"];
    let flags = int_vals(&Ty::int(1)).into_iter().map(|v| vec![v]).collect::<Vec<_>>();
    sweep_text(&mut counts, &context, &assert, &[], "cf.assert %a, \"the message\"", &flags);
    // Recorded as found: what ran, and what waits for Task 3.
    assert_eq!(
        counts,
        Counts {
            functions: 946,
            cases: 42022,
            control: 99,
            declaration: 0,
            unevaluated: 0,
        }
    );
}

/// The float values of width `width` 6c's float tests try, a sample.
fn float_vals(width: u32) -> Vec<Val> {
    let values: [f64; 14] = [
        0.0,
        -0.0,
        1.0,
        -1.5,
        2.5,
        0.1,
        255.9,
        -2_147_483_649.0,
        4_294_967_296.0,
        9.3e18,
        f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    values
        .iter()
        .map(|value| {
            if width == 32 {
                Val::f32(*value as f32)
            } else {
                Val::f64(*value)
            }
        })
        .collect()
}

#[test]
fn the_float_operations_of_6c_agree_where_they_are_straight_line() {
    let context = context();
    let mut counts = Counts::default();
    for (t, width) in [("f32", 32), ("f64", 64)] {
        let pairs = product(&[float_vals(width), float_vals(width)]);
        let singles = float_vals(width).into_iter().map(|v| vec![v]).collect::<Vec<_>>();
        for op in ["addf", "subf", "mulf", "divf"] {
            sweep_text(&mut counts, &context, &[t, t], &[t], &format!("arith.{op} %a, %b"), &pairs);
        }
        sweep_text(&mut counts, &context, &[t], &[t], "arith.negf %a", &singles);
        for predicate in CmpFPredicate::ALL {
            let body = format!("arith.cmpf {} %a, %b", predicate.as_str());
            sweep_text(&mut counts, &context, &[t, t], &["i1"], &body, &pairs);
        }
        for ty in types() {
            for op in ["fptosi", "fptoui"] {
                let body = format!("arith.{op} %a <{{ ty = {} }}>", ty.name);
                sweep_text(&mut counts, &context, &[t], &[&ty.name], &body, &singles);
            }
            let ints = int_vals(&ty).into_iter().map(|v| vec![v]).collect::<Vec<_>>();
            for op in ["sitofp", "uitofp"] {
                let body = format!("arith.{op} %a <{{ ty = {t} }}>");
                sweep_text(&mut counts, &context, &[&ty.name], &[t], &body, &ints);
            }
        }
        let int = if width == 32 { "i32" } else { "i64" };
        let ints = int_vals(&Ty::int(width)).into_iter().map(|v| vec![v]).collect::<Vec<_>>();
        sweep_text(
            &mut counts,
            &context,
            &[t],
            &[int],
            &format!("arith.bitcast %a <{{ ty = {int} }}>"),
            &singles,
        );
        sweep_text(
            &mut counts,
            &context,
            &[int],
            &[t],
            &format!("arith.bitcast %a <{{ ty = {t} }}>"),
            &ints,
        );
    }
    let f32s = float_vals(32).into_iter().map(|v| vec![v]).collect::<Vec<_>>();
    let f64s = float_vals(64).into_iter().map(|v| vec![v]).collect::<Vec<_>>();
    sweep_text(
        &mut counts,
        &context,
        &["f32"],
        &["f64"],
        "arith.fpext %a <{ ty = f64 }>",
        &f32s,
    );
    sweep_text(
        &mut counts,
        &context,
        &["f64"],
        &["f32"],
        "arith.fptrunc %a <{ ty = f32 }>",
        &f64s,
    );
    assert_eq!(
        counts,
        Counts {
            functions: 152,
            cases: 9090,
            control: 0,
            declaration: 0,
            unevaluated: 0,
        }
    );
}

/// The lit files of 6c whose functions are swept, with the pipelines of their `RUN` lines.
const FILES: [(&str, &str, &[&str]); 8] = [
    (
        "programs.runic",
        include_str!("../../../../../../tests/lit/codegen/wasm/programs.runic"),
        &[super::super::LOWERING, OPTIMIZED],
    ),
    (
        "programs-memory.runic",
        include_str!("../../../../../../tests/lit/codegen/wasm/programs-memory.runic"),
        &[super::super::LOWERING, OPTIMIZED],
    ),
    (
        "programs-narrow.runic",
        include_str!("../../../../../../tests/lit/codegen/wasm/programs-narrow.runic"),
        &[super::super::LOWERING, OPTIMIZED],
    ),
    (
        "indirect-calls.runic",
        include_str!("../../../../../../tests/lit/codegen/wasm/indirect-calls.runic"),
        &["convert-to-wasm,wasm-verify-form"],
    ),
    (
        "frames-pipeline.runic",
        include_str!("../../../../../../tests/lit/codegen/wasm/frames-pipeline.runic"),
        &["wasm-lower-frames,convert-to-wasm,wasm-verify-form"],
    ),
    (
        "numeric-composed.runic",
        include_str!("../../../../../../tests/lit/codegen/wasm/numeric-composed.runic"),
        &["convert-to-wasm,wasm-verify-form"],
    ),
    (
        "numeric-float.runic",
        include_str!("../../../../../../tests/lit/codegen/wasm/numeric-float.runic"),
        &["convert-to-wasm,wasm-verify-form"],
    ),
    (
        "numeric-int.runic",
        include_str!("../../../../../../tests/lit/codegen/wasm/numeric-int.runic"),
        &["convert-to-wasm,wasm-verify-form"],
    ),
];

/// The optimizing pipeline of the programs' second `RUN` line.
const OPTIMIZED: &str = "canonicalizer,cse,sccp,canonicalizer,dce,cfg-to-scf{exits=in-place},\
                         wasm-lower-frames,convert-to-wasm,wasm-verify-form";

/// The arguments tried for a parameter of the Wasm type of `ty`.
fn arguments_of(ty: runic_ir::Type) -> Vec<Val> {
    match crate::emit::instr::val_type(ty) {
        Some(crate::emit::desc::ValType::I32) => {
            vec![Val::i32(0), Val::i32(1), Val::i32(16), Val::i32(0xffff_fff0)]
        }
        Some(crate::emit::desc::ValType::I64) => {
            vec![Val::i64(0), Val::i64(3), Val::i64(1 << 63), Val::i64(u64::MAX)]
        }
        Some(crate::emit::desc::ValType::F32) => {
            vec![Val::f32(0.0), Val::f32(1.5), Val::f32(-2.5), Val::f32(f32::NAN)]
        }
        Some(crate::emit::desc::ValType::F64) => {
            vec![Val::f64(0.0), Val::f64(1.5), Val::f64(-2.5), Val::f64(f64::NAN)]
        }
        _ => panic!("a parameter of type `{ty}`"),
    }
}

/// The setup of a lowered program: a patterned page, each data symbol of the module at its own
/// address, the stack pointer in the middle of the page and every other global zero.
fn program_setup(module: OperationRef) -> Setup {
    let bytes = (0..256u32).map(|i| (i * 7 + 3) as u8).collect::<Vec<_>>();
    let mut setup = Setup::default().write(0, &bytes).comparing_memory();
    let mut next = 4096;
    for op in ops_of(module) {
        let op = op.borrow();
        if let Some(variable) = op.downcast_ref::<GlobalVariable>() {
            setup = setup.data(variable.get_name().as_str(), next);
            next += 256;
        } else if let Some(global) = op.downcast_ref::<wasm::Global>() {
            let name = global.get_name().as_str();
            let value = match crate::emit::instr::val_type(*global.get_ty()) {
                Some(crate::emit::desc::ValType::I64) => Val::i64(0),
                Some(crate::emit::desc::ValType::F32) => Val::f32(0.0),
                Some(crate::emit::desc::ValType::F64) => Val::f64(0.0),
                _ if name == "__stack_pointer" => Val::i32((PAGE / 2) as u32),
                _ => Val::i32(0),
            };
            setup = setup.global(name, value);
        }
    }
    setup
}

#[test]
fn the_functions_of_6c_s_programs_agree_where_they_are_straight_line() {
    let mut counts = Counts::default();
    for (file, text, pipelines) in FILES {
        for chunk in text.split("// -----") {
            if !chunk.contains("builtin.module") {
                continue;
            }
            for pipeline in pipelines {
                let (context, module) = parse_module(chunk);
                run_pipeline(&context, module.as_operation_ref(), pipeline).unwrap_or_else(
                    |report| panic!("{file}: {}", runic_ir::testing::text::describe(&report)),
                );
                let module = module.as_operation_ref();
                let setup = program_setup(module);
                for op in ops_of(module) {
                    let is_definition = op
                        .borrow()
                        .downcast_ref::<Function>()
                        .is_some_and(|function| !function.is_declaration());
                    if !is_definition {
                        continue;
                    }
                    let params = entry(op)
                        .borrow()
                        .argument_values()
                        .map(|value| arguments_of(value.borrow().ty()))
                        .collect::<Vec<_>>();
                    let cases =
                        product(&params).into_iter().step_by(3).take(48).collect::<Vec<_>>();
                    sweep(&mut counts, &setup, op, &cases);
                }
                drop(context);
            }
        }
    }
    assert_eq!(
        counts,
        Counts {
            functions: 197,
            cases: 980,
            control: 34,
            declaration: 3,
            unevaluated: 0,
        }
    );
}
