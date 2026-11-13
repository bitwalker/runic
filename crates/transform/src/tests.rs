//! Tests across the passes of this crate (and `LiftControlFlowToSCF`): `cf.assert` in every
//! position through every pass, and `cf.switch` keys as 32-bit patterns through the lift.

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    Context, Operation, Report,
    dialects::builtin::{Function, FunctionRef},
    parse::{self, ParserConfig},
    pass::{Nesting, OperationPass, PassManager},
    testing::text::describe,
};
use runic_dialect_cf::Assert;
use runic_dialect_scf::transforms::LiftControlFlowToSCF;

use crate::{
    Canonicalizer, CommonSubexpressionElimination, ControlFlowSink, DeadCodeElimination,
    SinkOperandDefs, SparseConditionalConstantPropagation,
};

/// `f(%c: i1, %d: i1, %a: i32, %b: i32, %p: ptr<i32>) -> i32` with the body `body`.
fn make_function(context: &Rc<Context>, body: &str) -> FunctionRef {
    let text = format!(
        "builtin.function public extern(\"C\") @f(%c: i1, %d: i1, %a: i32, %b: i32, %p: ptr<i32>) \
         -> i32 {{\n{body}\n}};"
    );
    parse::parse::<Function>(ParserConfig::new(context.clone()), "passes", text)
        .into_result()
        .unwrap_or_else(|err| panic!("{}\n{body}", describe(&err)))
}

/// The message of every `cf.assert` under `op`, in pre-order.
fn asserts(op: &Operation) -> Vec<String> {
    let mut found = Vec::new();
    op.prewalk_all(|op: &Operation| {
        if let Some(assert) = op.downcast_ref::<Assert>() {
            found.push(assert.get_message().to_string());
        }
    });
    found
}

type MakePass = fn() -> Box<dyn OperationPass>;

const PASSES: [(&str, MakePass); 7] = [
    ("dce", || Box::<DeadCodeElimination>::default()),
    ("cse", || Box::<CommonSubexpressionElimination>::default()),
    ("control-flow-sink", || Box::<ControlFlowSink>::default()),
    ("sink-operand-defs", || Box::<SinkOperandDefs>::default()),
    ("sccp", || Box::<SparseConditionalConstantPropagation>::default()),
    ("canonicalize", || Box::<Canonicalizer>::default()),
    ("cfg-to-scf", || Box::<LiftControlFlowToSCF>::default()),
];

/// Runs `pass` on `function`, verifying the result.
fn run(
    context: &Rc<Context>,
    function: FunctionRef,
    pass: Box<dyn OperationPass>,
) -> Result<(), Report> {
    let mut passes = PassManager::on::<Function>(context.clone(), Nesting::Implicit);
    passes.add_pass(pass);
    passes.enable_verifier(true);
    passes.run(function.as_operation_ref())
}

/// Positions of a `cf.assert` whose condition is not a constant. Every pass must keep every one of
/// them, in the same order.
const POSITIONS: [(&str, &str); 9] = [
    (
        "unused and duplicated in one block",
        r#"
    cf.assert %c, "one";
    cf.assert %c, "one";
    %e = arith.cmpi eq %a, %b;
    cf.assert %e, "cmp";
    %e2 = arith.cmpi eq %a, %b;
    cf.assert %e2, "cmp";
    builtin.ret %a : (i32);"#,
    ),
    (
        "in both regions of an if without results",
        r#"
    %e = arith.cmpi eq %a, %b;
    scf.if %c then {
        cf.assert %e, "same";
        scf.yield : ();
    } else {
        cf.assert %e, "same";
        scf.yield : ();
    } : (i1) -> ();
    builtin.ret %a : (i32);"#,
    ),
    (
        "in a region of an if whose result is unused",
        r#"
    %e = arith.cmpi eq %a, %b;
    %x = scf.if %c then {
        cf.assert %e, "then";
        scf.yield %a : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %a : (i32);"#,
    ),
    (
        "in the body of a loop whose result is unused",
        r#"
    %zero = arith.constant 0 : i32;
    %e = arith.cmpi ne %a, %b;
    %count = scf.while %zero before {
    ^head(%i: i32):
        cf.assert %e, "head";
        %continue = arith.cmpi ult %i, %a;
        scf.condition %continue, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        cf.assert %e, "body";
        %in = arith.cmpi ult %j, %b;
        cf.assert %in, "bound";
        %one = arith.constant 1 : i32;
        %next = arith.add %j, %one;
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %a : (i32);"#,
    ),
    (
        "after a branch, in each successor and at the join",
        r#"
    %e = arith.cmpi eq %a, %b;
    cf.assert %e, "entry";
    cf.cond_br %c ^t, ^f : (i1);
^t:
    cf.assert %e, "entry";
    cf.assert %d, "then";
    cf.br ^m(%a : i32);
^f:
    cf.assert %e, "entry";
    cf.assert %d, "else";
    cf.br ^m(%b : i32);
^m(%r: i32):
    cf.assert %e, "entry";
    cf.assert %d, "join";
    builtin.ret %r : (i32);"#,
    ),
    (
        "between a definition and its only use in a region",
        r#"
    %sum = arith.add %a, %b;
    %q = arith.udiv %a, %b;
    cf.assert %c, "before";
    %x = scf.if %d then {
        cf.assert %c, "inside";
        %y = arith.add %sum, %q;
        scf.yield %y : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    cf.assert %c, "after";
    builtin.ret %x : (i32);"#,
    ),
    (
        "of the constant 0, with code after it",
        r#"
    %false = arith.constant 0 : i1;
    cf.assert %false, "never";
    %sum = arith.add %a, %b;
    cf.assert %c, "later";
    builtin.ret %sum : (i32);"#,
    ),
    (
        "between two loads and two stores",
        r#"
    %x = test.load %p;
    cf.assert %c, "between loads";
    %y = test.load %p;
    test.store %p, %a : (ptr<i32>, i32);
    cf.assert %c, "between stores";
    test.store %p, %b : (ptr<i32>, i32);
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    ),
    (
        "in two identical ifs",
        r#"
    %e = arith.cmpi eq %a, %b;
    %x = scf.if %c then {
        cf.assert %e, "twin";
        scf.yield %a : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    %y = scf.if %c then {
        cf.assert %e, "twin";
        scf.yield %a : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    ),
];

/// Every pass, on every position, keeps every assert and their order (pre-order of the
/// function). The canonicalizer and `cfg-to-scf` move blocks, so there the multiset is compared.
#[test]
fn every_pass_keeps_every_assert_in_every_position() {
    let mut failures = Vec::new();
    for (position, body) in POSITIONS {
        for (pass, make) in PASSES {
            let context = Rc::new(Context::default());
            let function = make_function(&context, body);
            let before = asserts(&function.as_operation_ref().borrow());
            assert!(!before.is_empty());
            if let Err(err) = run(&context, function, make()) {
                failures.push(format!("{pass} on '{position}' failed: {}", describe(&err)));
                continue;
            }
            let mut after = asserts(&function.as_operation_ref().borrow());
            let mut expected = before.clone();
            if pass == "cfg-to-scf" || pass == "canonicalize" {
                after.sort();
                expected.sort();
            }
            if after != expected {
                failures.push(format!(
                    "{pass} on '{position}': asserts {before:?} became {after:?}\n{}",
                    function.as_operation_ref().borrow()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n\n"));
}

/// All passes in sequence, twice, on every position, keep every assert.
#[test]
fn the_whole_pipeline_keeps_every_assert() {
    for (position, body) in POSITIONS {
        let context = Rc::new(Context::default());
        let function = make_function(&context, body);
        let mut before = asserts(&function.as_operation_ref().borrow());
        before.sort();
        for _ in 0..2 {
            for (pass, make) in PASSES {
                run(&context, function, make())
                    .unwrap_or_else(|err| panic!("{pass} on '{position}': {}", describe(&err)));
            }
        }
        let mut after = asserts(&function.as_operation_ref().borrow());
        after.sort();
        assert_eq!(after, before, "'{position}':\n{}", function.as_operation_ref().borrow());
    }
}

/// A `cf.switch` with the keys -1, 0, `i32::MIN` and `i32::MAX` lifts to an `scf.index_switch`
/// with the same case values, and each constant selector, in several spellings, reaches its
/// case: in CFG form per the dead-code analysis, and through lift-then-canonicalize and
/// canonicalize-then-lift.
#[test]
fn a_switch_with_extreme_keys_lifts_and_folds_as_32_bit_patterns() {
    let body = |selector: &str| {
        format!(
            r#"
    {selector}
    cf.switch %s [#builtin.i32<-1> -> ^m1, #builtin.i32<0> -> ^z, #builtin.i32<-2147483648> -> ^min, #builtin.i32<2147483647> -> ^max], ^other : (i32);
^m1:
    cf.assert %c, "-1";
    cf.br ^join;
^z:
    cf.assert %c, "0";
    cf.br ^join;
^min:
    cf.assert %c, "min";
    cf.br ^join;
^max:
    cf.assert %c, "max";
    cf.br ^join;
^other:
    cf.assert %c, "other";
    cf.br ^join;
^join:
    builtin.ret %a : (i32);"#
        )
    };
    // Lift with an unknown selector: the cases are kept as they are
    let context = Rc::new(Context::default());
    let function = make_function(&context, &body("%s = arith.add %a, %b;"));
    run(&context, function, Box::<LiftControlFlowToSCF>::default()).unwrap();
    let text = function.as_operation_ref().borrow().to_string();
    for case in ["case -1 ", "case 0 ", "case -2147483648 ", "case 2147483647 "] {
        assert!(text.contains(case), "missing `{case}`:\n{text}");
    }
    let mut lifted = asserts(&function.as_operation_ref().borrow());
    lifted.sort();
    assert_eq!(lifted, ["-1", "0", "max", "min", "other"]);

    // With each constant selector, written in several spellings: lift, then canonicalize
    for (literal, expected) in [
        ("-1", "-1"),
        ("0xffffffff", "-1"),
        ("4294967295", "-1"),
        ("0", "0"),
        ("-2147483648", "min"),
        ("0x80000000", "min"),
        ("2147483648", "min"),
        ("2147483647", "max"),
        ("0x7fffffff", "max"),
        ("1", "other"),
        ("-2", "other"),
        ("0x7ffffffe", "other"),
    ] {
        // In CFG form no pass folds a switch on a constant; the dead-code analysis reads the
        // selector through `get_successor_for_operands`, so ask it which successors are live
        {
            use runic_analysis::{
                DataFlowSolver,
                analyses::{DeadCodeAnalysis, SparseConstantPropagation, dce::Executable},
            };
            use runic_core::{ProgramPoint, pass::AnalysisManager};

            let context = Rc::new(Context::default());
            let function =
                make_function(&context, &body(&format!("%s = arith.constant {literal} : i32;")));
            let mut solver = DataFlowSolver::default();
            solver.load::<DeadCodeAnalysis>();
            solver.load::<SparseConstantPropagation>();
            let op = function.as_operation_ref();
            solver.initialize_and_run(&op.borrow(), AnalysisManager::new(op, None)).unwrap();
            let function = function.borrow();
            let mut live = Vec::new();
            for block in function.body().body().iter() {
                let is_live = solver
                    .get::<Executable, _>(&ProgramPoint::at_start_of(block.as_block_ref()))
                    .is_some_and(|state| state.is_live());
                if is_live {
                    for op in block.body().iter() {
                        if let Some(assert) = op.downcast_ref::<Assert>() {
                            live.push(assert.get_message().to_string());
                        }
                    }
                }
            }
            assert_eq!(live, [expected], "selector {literal}: live successors per the analysis");
        }
        for order in [["cfg-to-scf", "canonicalize"], ["canonicalize", "cfg-to-scf"]] {
            let context = Rc::new(Context::default());
            let function =
                make_function(&context, &body(&format!("%s = arith.constant {literal} : i32;")));
            for name in order {
                let (_, make) = PASSES.iter().find(|(pass, _)| *pass == name).unwrap();
                run(&context, function, make()).unwrap();
            }
            // one more canonicalization so that folds enabled by the lift apply
            run(&context, function, Box::<Canonicalizer>::default()).unwrap();
            let remaining = asserts(&function.as_operation_ref().borrow());
            assert_eq!(
                remaining,
                [expected],
                "selector {literal} through {order:?}:\n{}",
                function.as_operation_ref().borrow()
            );
        }
    }
}

mod calls;
