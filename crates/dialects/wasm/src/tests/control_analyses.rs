//! The analyses of `runic-analysis` and the passes of `runic-transform` over the structured control
//! flow operations.
//!
//! Each analysis test has a control: the same program with the branch or the end that matters
//! changed, which shows that the answer comes from the edge under test. The edge of a `wasm.br`
//! leaves its own construct: to the results of the block or the `if` it names, or to the top of the
//! loop it names; an analysis that did not see it would, silently, call the code after the
//! construct unreachable, or a value that the branch changes constant.

use runic_analysis::{
    DataFlowConfig, DataFlowSolver, Lattice,
    analyses::{
        CallableUseSnapshot, DeadCodeAnalysis, IntegerRangeAnalysis, LivenessAnalysis,
        SparseConstantPropagation,
        constant_propagation::ConstantValue,
        dce::{Executable, PredecessorState},
        liveness::{LOOP_EXIT_DISTANCE, Liveness},
    },
};
use runic_ir::{
    BlockRef, OperationRef, ProgramPoint, ValueRef,
    dialects::builtin::Function,
    pass::{AnalysisManager, PassPipeline},
    testing::text::normalize,
};

use super::{control::rich_context, *};

/// A function parsed with the dialects of the tests, and the context that keeps it alive.
struct Program {
    #[allow(unused)]
    context: Rc<Context>,
    function: OperationRef,
}

impl Program {
    fn parse(source: &str) -> Self {
        let context = rich_context();
        let function = parse::parse::<Function>(ParserConfig::new(context.clone()), "f", source)
            .into_result()
            .unwrap_or_else(|report| panic!("{}\n{source}", describe(&report)));
        Self {
            context,
            function: function.as_operation_ref(),
        }
    }

    /// The operations named `name`, in pre-order.
    fn ops(&self, name: &str) -> Vec<OperationRef> {
        let mut found = Vec::new();
        self.function.borrow().prewalk_all(|op: &Operation| {
            if op.name().to_string() == name {
                found.push(op.as_operation_ref());
            }
        });
        found
    }

    /// The first operation named `name`, in pre-order.
    fn op(&self, name: &str) -> OperationRef {
        *self.ops(name).first().unwrap_or_else(|| panic!("no {name}"))
    }

    fn result(&self, name: &str, index: usize) -> ValueRef {
        self.op(name).borrow().results()[index].borrow().as_value_ref()
    }

    fn argument(&self, index: usize) -> ValueRef {
        let function = self.function.borrow();
        let entry = function.downcast_ref::<Function>().unwrap().entry_block();
        entry.borrow().arguments()[index].borrow().as_value_ref()
    }

    /// The entry block of the `region`th region of the first operation named `name`.
    fn entry_of(&self, name: &str, region: usize) -> BlockRef {
        self.op(name).borrow().region(region).entry_block_ref().unwrap()
    }

    fn solve(&self) -> DataFlowSolver {
        let mut solver = DataFlowSolver::new(DataFlowConfig::default());
        solver.load::<DeadCodeAnalysis>();
        solver.load::<SparseConstantPropagation>();
        solver
            .initialize_and_run(&self.function.borrow(), AnalysisManager::new(self.function, None))
            .expect("the analyses converge");
        solver
    }

    fn liveness(&self) -> Rc<LivenessAnalysis> {
        AnalysisManager::new(self.function, None)
            .get_analysis_for::<LivenessAnalysis, Operation>()
            .expect("liveness runs")
    }
}

/// Whether the block is executable, according to `solver`.
fn executable(solver: &DataFlowSolver, block: BlockRef) -> bool {
    solver
        .get::<Executable, _>(&ProgramPoint::at_start_of(block))
        .is_some_and(|state| state.is_live())
}

/// The names of the known predecessors of the point.
fn predecessors(solver: &DataFlowSolver, point: ProgramPoint) -> Vec<String> {
    solver
        .get::<PredecessorState, _>(&point)
        .map(|state| {
            state
                .known_predecessors()
                .iter()
                .map(|op| op.borrow().name().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// What constant propagation knows of a value: `None` when it is not one constant. A value the
/// analysis never reached is an error.
fn constant(solver: &DataFlowSolver, value: ValueRef) -> Option<u64> {
    let lattice = solver
        .get::<Lattice<ConstantValue>, _>(&value)
        .expect("the value has a lattice");
    assert!(!lattice.value().is_uninitialized(), "the analysis never reached the value");
    let attr = lattice.value().constant_value()?;
    let value = attr
        .borrow()
        .as_attr()
        .as_trait::<dyn runic_ir::attributes::IntegerLikeAttr>()
        .expect("an integer constant")
        .as_int()
        .try_unsigned::<u64>()
        .expect("fits");
    Some(value)
}

/// Every analysis of the analysis crate over `root`: the names of those that returned an error.
fn analyze(root: OperationRef) -> Vec<String> {
    let mut errors = Vec::new();
    let top = root.parent_op().unwrap_or(root);
    let snapshot = CallableUseSnapshot::new(&top.borrow());
    for (_, info) in snapshot.iter() {
        let _ = (info.known_callers().len(), info.is_address_taken(), info.has_unknown_callers());
    }
    for interprocedural in [true, false] {
        let mut config = DataFlowConfig::default();
        config.set_interprocedural(interprocedural);
        let mut solver = DataFlowSolver::new(config);
        solver.load::<DeadCodeAnalysis>();
        solver.load::<SparseConstantPropagation>();
        if let Err(err) = solver.initialize_and_run(&top.borrow(), AnalysisManager::new(top, None))
        {
            errors.push(format!("dce+sccp ({interprocedural}): {}", describe(&err)));
        }
        solver.load::<IntegerRangeAnalysis>();
        if let Err(err) = solver.initialize_and_run(&top.borrow(), AnalysisManager::new(top, None))
        {
            errors.push(format!("ranges ({interprocedural}): {}", describe(&err)));
        }
    }
    let mut solver = DataFlowSolver::new(DataFlowConfig::default());
    solver.load::<DeadCodeAnalysis>();
    solver.load::<SparseConstantPropagation>();
    solver.load::<Liveness>();
    if let Err(err) = solver.initialize_and_run(&root.borrow(), AnalysisManager::new(root, None)) {
        errors.push(format!("liveness: {}", describe(&err)));
    }
    errors
}

// ---------------------------------------------------------------------------------------------
// Dead code

/// A block that an `if` in it may leave with another value than the one it falls off its end with.
const BLOCK: &str = r#"
builtin.function public extern("C") @f(%c: i32) -> i32 {
    %one = wasm.const 1 : i32;
    %two = wasm.const 2 : i32;
    %r = wasm.block {
        wasm.if %c then {
            wasm.br 1 %two : (i32);
        } : (i32) -> ();
        wasm.end %one : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;

/// A block whose body never falls off its end: it is left by a branch, and only by that.
const BLOCK_ONLY_BRANCH: &str = r#"
builtin.function public extern("C") @f(%c: i32) -> i32 {
    %two = wasm.const 2 : i32;
    %r = wasm.block {
        wasm.if %c then {
            wasm.br 1 %two : (i32);
        } : (i32) -> ();
        wasm.br 0 %two : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;

#[test]
fn control_that_a_branch_gives_to_a_block_arrives_after_the_block() {
    let program = Program::parse(BLOCK);
    let solver = program.solve();
    let block = program.op("wasm.block");
    let names = predecessors(&solver, ProgramPoint::after(&*block.borrow()));
    assert!(names.contains(&String::from("wasm.br")), "{names:?}");
    assert!(names.contains(&String::from("wasm.end")), "{names:?}");
    assert!(executable(&solver, program.entry_of("wasm.block", 0)));
    assert!(executable(&solver, program.entry_of("wasm.if", 0)));

    // control: nothing branches, so only the end of the body arrives
    let control = Program::parse(&BLOCK.replace("wasm.br 1 %two : (i32);", "wasm.end : ();"));
    let solver = control.solve();
    let block = control.op("wasm.block");
    let names = predecessors(&solver, ProgramPoint::after(&*block.borrow()));
    assert!(!names.contains(&String::from("wasm.br")), "{names:?}");
    assert!(names.contains(&String::from("wasm.end")), "{names:?}");
}

#[test]
fn a_block_that_only_a_branch_leaves_is_left_through_the_branch() {
    let program = Program::parse(BLOCK_ONLY_BRANCH);
    let solver = program.solve();
    let block = program.op("wasm.block");
    let names = predecessors(&solver, ProgramPoint::after(&*block.borrow()));
    assert!(names.contains(&String::from("wasm.br")), "{names:?}");
    // the result is the value the two branches carry
    assert_eq!(constant(&solver, program.result("wasm.block", 0)), Some(2));
    assert!(executable(&solver, program.entry_of("wasm.if", 0)));
}

/// A body that ends in a terminator that is no `wasm.end` (a `wasm.unreachable`, a `wasm.br_table`)
/// is not a region-branch terminator of the block, so the analyses cannot say what it forwards to
/// the results of the block: they are not known. That is sound and imprecise, and it is what
/// `ub.unreachable` under an `scf.if` gets.
#[test]
fn a_block_that_ends_in_a_trap_or_a_table_has_results_that_are_not_known() {
    let trap = BLOCK_ONLY_BRANCH.replace("wasm.br 0 %two : (i32);", "wasm.unreachable;");
    let program = Program::parse(&trap);
    let solver = program.solve();
    assert_eq!(constant(&solver, program.result("wasm.block", 0)), None);
    // but the branch is still an edge to after the block
    let block = program.op("wasm.block");
    let names = predecessors(&solver, ProgramPoint::after(&*block.borrow()));
    assert!(names.contains(&String::from("wasm.br")), "{names:?}");
}

#[test]
fn a_loop_that_only_a_branch_to_an_outer_block_leaves_is_executable_and_its_exit_arrives() {
    let source = r#"
builtin.function public extern("C") @f(%c: i32) -> i32 {
    %zero = wasm.const 0 : i32;
    %r = wasm.block {
        %l = wasm.loop %zero {
        ^h(%i: i32):
            wasm.if %c then {
                wasm.br 2 %i : (i32);
            } : (i32) -> ();
            wasm.br 0 %i : (i32);
        } : (i32) -> (i32);
        wasm.end %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;
    let program = Program::parse(source);
    let solver = program.solve();
    let block = program.op("wasm.block");
    let names = predecessors(&solver, ProgramPoint::after(&*block.borrow()));
    assert!(names.contains(&String::from("wasm.br")), "{names:?}");
    assert!(executable(&solver, program.entry_of("wasm.loop", 0)));
    assert!(executable(&solver, program.entry_of("wasm.if", 0)));
    // the branch to the loop is a predecessor of the top of the loop, as the loop is
    let names = predecessors(&solver, ProgramPoint::at_start_of(program.entry_of("wasm.loop", 0)));
    assert!(names.contains(&String::from("wasm.loop")), "{names:?}");
    assert!(names.contains(&String::from("wasm.br")), "{names:?}");
    // the loop carries the same value round: the block result is the constant 0
    assert_eq!(constant(&solver, program.result("wasm.block", 0)), Some(0));
}

#[test]
fn the_regions_of_an_if_are_executable_as_its_condition_says() {
    let source = |cond: &str| {
        format!(
            r#"
builtin.function public extern("C") @f(%c: i32) {{
    %zero = wasm.const 0 : i32;
    %one = wasm.const 1 : i32;
    wasm.if {cond} then {{
        wasm.end : ();
    }} else {{
        wasm.end : ();
    }} : (i32) -> ();
    builtin.ret;
}};"#
        )
    };
    for (cond, then_live, else_live) in
        [("%one", true, false), ("%zero", false, true), ("%c", true, true)]
    {
        let program = Program::parse(&source(cond));
        let solver = program.solve();
        assert_eq!(executable(&solver, program.entry_of("wasm.if", 0)), then_live, "{cond}");
        assert_eq!(executable(&solver, program.entry_of("wasm.if", 1)), else_live, "{cond}");
    }
}

#[test]
fn the_regions_of_a_branch_table_are_executable_as_its_index_says() {
    let source = |index: &str| {
        format!(
            r#"
builtin.function public extern("C") @f(%c: i32) {{
    %zero = wasm.const 0 : i32;
    %one = wasm.const 1 : i32;
    %seven = wasm.const 7 : i32;
    wasm.block {{
        wasm.br_table {index} [{{
            wasm.br 0 : ();
        }}, {{
            wasm.br 0 : ();
        }}] default {{
            wasm.br 0 : ();
        }} : (i32);
    }} : () -> ();
    builtin.ret;
}};"#
        )
    };
    // the regions are the default first, then the entries
    for (index, live) in [
        ("%zero", [false, true, false]),
        ("%one", [false, false, true]),
        ("%seven", [true, false, false]),
        ("%c", [true, true, true]),
    ] {
        let program = Program::parse(&source(index));
        let solver = program.solve();
        let table = program.op("wasm.br_table");
        for (region, expected) in live.into_iter().enumerate() {
            let entry = table.borrow().region(region).entry_block_ref().unwrap();
            assert_eq!(executable(&solver, entry), expected, "{index}, region {region}");
        }
        // some branch of the table always arrives after the block
        let block = program.op("wasm.block");
        let names = predecessors(&solver, ProgramPoint::after(&*block.borrow()));
        assert!(names.contains(&String::from("wasm.br")), "{index}: {names:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Constants

#[test]
fn constant_propagation_does_not_fold_a_block_result_to_the_value_it_falls_off_the_end_with() {
    let program = Program::parse(BLOCK);
    let solver = program.solve();
    assert_eq!(
        constant(&solver, program.result("wasm.block", 0)),
        None,
        "1 by the end of the body, and 2 by the branch"
    );

    // control: the branch carries the value the end does
    let control = Program::parse(&BLOCK.replace("wasm.br 1 %two", "wasm.br 1 %one"));
    let solver = control.solve();
    assert_eq!(constant(&solver, control.result("wasm.block", 0)), Some(1));
}

/// A loop whose body argument a branch replaces with another constant.
const LOOP: &str = r#"
builtin.function public extern("C") @f(%c: i32) -> i32 {
    %zero = wasm.const 0 : i32;
    %two = wasm.const 2 : i32;
    %r = wasm.block {
        %l = wasm.loop %zero {
        ^h(%i: i32):
            wasm.if %c then {
                wasm.br 2 %i : (i32);
            } : (i32) -> ();
            wasm.br 0 %two : (i32);
        } : (i32) -> (i32);
        wasm.end %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;

#[test]
fn constant_propagation_does_not_fold_a_loop_argument_to_its_initial_operand_when_a_branch_passes_another()
 {
    let program = Program::parse(LOOP);
    let solver = program.solve();
    let argument = program.entry_of("wasm.loop", 0).borrow().arguments()[0].borrow().as_value_ref();
    assert_eq!(constant(&solver, argument), None, "0 on entry, and 2 after a branch");
    assert_eq!(constant(&solver, program.result("wasm.block", 0)), None);

    // control: the branch passes the initial operand again
    let control = Program::parse(&LOOP.replace("wasm.br 0 %two", "wasm.br 0 %zero"));
    let solver = control.solve();
    let argument = control.entry_of("wasm.loop", 0).borrow().arguments()[0].borrow().as_value_ref();
    assert_eq!(constant(&solver, argument), Some(0));
    assert_eq!(constant(&solver, control.result("wasm.block", 0)), Some(0));
}

#[test]
fn constant_propagation_does_not_fold_the_result_of_a_block_a_branch_table_leaves_with_another_value()
 {
    let source = r#"
builtin.function public extern("C") @f(%c: i32) -> i32 {
    %one = wasm.const 1 : i32;
    %two = wasm.const 2 : i32;
    %r = wasm.block {
        wasm.if %c then {
            wasm.br_table %c, %two [{
                wasm.br 1 %two : (i32);
            }] default {
                wasm.br 1 %two : (i32);
            } : (i32, i32);
        } : (i32) -> ();
        wasm.end %one : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;
    let program = Program::parse(source);
    let solver = program.solve();
    assert_eq!(
        constant(&solver, program.result("wasm.block", 0)),
        None,
        "1 by the end, 2 by the table"
    );

    // control: the table carries what the end does
    let control = Program::parse(
        &source.replace("%two [{", "%one [{").replace("wasm.br 1 %two", "wasm.br 1 %one"),
    );
    let solver = control.solve();
    assert_eq!(constant(&solver, control.result("wasm.block", 0)), Some(1));
}

/// A loop that passes its own arguments back, swapped (`br 0 %y, %x`): the lattice of each is held
/// while the other is read.
const SWAP_LOOP: &str = r#"
builtin.function public extern("C") @f(%c: i32) -> i32 {
    %zero = wasm.const 0 : i32;
    %one = wasm.const 1 : i32;
    %r = wasm.block {
        %l = wasm.loop %zero, %one {
        ^h(%x: i32, %y: i32):
            wasm.if %c then {
                wasm.br 2 %x : (i32);
            } : (i32) -> ();
            wasm.br 0 BACK : (i32, i32);
        } : (i32, i32) -> (i32);
        wasm.end %l : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;

#[test]
fn constant_propagation_passes_the_arguments_of_a_loop_back_to_each_other() {
    // swapped: each argument is 0 or 1
    let program = Program::parse(&SWAP_LOOP.replace("BACK", "%y, %x"));
    let solver = program.solve();
    let entry = program.entry_of("wasm.loop", 0);
    for index in 0..2 {
        let argument = entry.borrow().arguments()[index].borrow().as_value_ref();
        assert_eq!(constant(&solver, argument), None, "argument {index}");
    }
    assert_eq!(constant(&solver, program.result("wasm.block", 0)), None);
    assert!(analyze(program.function).is_empty());

    // control: each argument goes to itself, and stays what it starts as
    let control = Program::parse(&SWAP_LOOP.replace("BACK", "%x, %y"));
    let solver = control.solve();
    let entry = control.entry_of("wasm.loop", 0);
    for (index, expected) in [0, 1].into_iter().enumerate() {
        let argument = entry.borrow().arguments()[index].borrow().as_value_ref();
        assert_eq!(constant(&solver, argument), Some(expected), "argument {index}");
    }
    assert_eq!(constant(&solver, control.result("wasm.block", 0)), Some(0));

    // the one that is passed to the other, and one that is new
    let mixed = Program::parse(&SWAP_LOOP.replace("BACK", "%y, %y"));
    let solver = mixed.solve();
    let entry = mixed.entry_of("wasm.loop", 0);
    let first = entry.borrow().arguments()[0].borrow().as_value_ref();
    let second = entry.borrow().arguments()[1].borrow().as_value_ref();
    assert_eq!(constant(&solver, first), None, "0, then what the second is");
    assert_eq!(constant(&solver, second), Some(1), "1, and the second again");
}

// ---------------------------------------------------------------------------------------------
// Liveness

const LIVE_LOOP: &str = r#"
builtin.function public extern("C") @f(%c: i32, %k: i32) -> i32 {
    %zero = wasm.const 0 : i32;
    %r = wasm.loop %zero {
    ^h(%i: i32):
        %m = wasm.add %i, %k;
        wasm.if %c then {
            INNER
        } : (i32) -> ();
        TAIL
    } : (i32) -> (i32);
    builtin.ret %r : (i32);
};"#;

/// `LIVE_LOOP` with the end of the `if` and the end of the body.
fn live_loop(inner: &str, tail: &str) -> String {
    LIVE_LOOP.replace("INNER", inner).replace("TAIL", tail)
}

const REPEAT_IF: &str = "wasm.br 1 %m : (i32);";
const REPEAT_BODY: &str = "wasm.br 0 %m : (i32);";
const END_IF: &str = "wasm.end : ();";
const END_BODY: &str = "wasm.end %m : (i32);";

#[test]
fn a_value_used_at_the_top_of_a_loop_is_live_across_the_back_edge() {
    let program = Program::parse(&live_loop(REPEAT_IF, REPEAT_BODY));
    let liveness = program.liveness();
    let k = program.argument(1);
    let back = program.ops("wasm.br").pop().unwrap();
    assert!(
        liveness.is_live_before(k, &back.borrow()),
        "`%k` is used when the loop starts again"
    );
    assert!(liveness.is_live_after(k, &back.borrow()));
    // it is live before the `if` in the body, which is before the branch
    assert!(liveness.is_live_before(k, &program.op("wasm.if").borrow()));

    // control: the body ends, and nothing goes back to the use
    let control = Program::parse(&live_loop(END_IF, END_BODY));
    let liveness = control.liveness();
    let k = control.argument(1);
    let end = control.ops("wasm.end").pop().unwrap();
    assert!(!liveness.is_live_before(k, &end.borrow()));
}

#[test]
fn a_value_used_after_a_block_is_live_where_a_branch_leaves_the_block() {
    let source = r#"
builtin.function public extern("C") @f(%c: i32, %k: i32) -> i32 {
    %zero = wasm.const 0 : i32;
    %r = wasm.block {
        wasm.if %c then {
            wasm.br 1 %zero : (i32);
        } : (i32) -> ();
        wasm.end %zero : (i32);
    } : () -> (i32);
    %z = wasm.add %r, USE;
    builtin.ret %z : (i32);
};"#;
    let program = Program::parse(&source.replace("USE", "%k"));
    let liveness = program.liveness();
    let k = program.argument(1);
    let branch = program.op("wasm.br");
    assert!(liveness.is_live_before(k, &branch.borrow()), "`%k` is used after the block");
    assert!(liveness.is_live_before(k, &program.op("wasm.if").borrow()));

    // control: `%k` is not used after the block
    let control = Program::parse(&source.replace("USE", "%r"));
    let liveness = control.liveness();
    let k = control.argument(1);
    assert!(!liveness.is_live_before(k, &control.op("wasm.br").borrow()));
}

#[test]
fn liveness_prices_leaving_a_loop_that_a_branch_repeats_as_a_loop_exit() {
    let after_loop = |inner: &str| {
        let source = live_loop(inner, END_BODY).replace(
            "builtin.ret %r : (i32);",
            "%z = wasm.add %r, %k;\n    builtin.ret %z : (i32);",
        );
        let program = Program::parse(&source);
        let liveness = program.liveness();
        let k = program.argument(1);
        // where the loop is left: the end of its body
        let end = program.ops("wasm.end").pop().unwrap();
        liveness
            .next_uses_at(&ProgramPoint::before(&*end.borrow()))
            .map(|uses| uses.distance(k))
            .expect("the point has next uses")
    };
    assert!(after_loop(REPEAT_IF) >= LOOP_EXIT_DISTANCE);
    assert!(after_loop(END_IF) < LOOP_EXIT_DISTANCE);
}

// ---------------------------------------------------------------------------------------------
// All the analyses

/// Programs of the kinds above, with the early exits of every shape.
fn programs() -> Vec<String> {
    let mut all = alloc::vec![
        String::from(BLOCK),
        String::from(BLOCK_ONLY_BRANCH),
        String::from(LOOP),
        live_loop(REPEAT_IF, REPEAT_BODY),
        live_loop(REPEAT_IF, END_BODY),
        SWAP_LOOP.replace("BACK", "%y, %x"),
    ];
    all.push(String::from(
        r#"
builtin.function public extern("C") @f(%c: i32, %k: i32) -> i32 {
    %zero = wasm.const 0 : i32;
    %r = wasm.block %k {
    ^b(%v: i32):
        %q = wasm.loop %v {
        ^h(%w: i32):
            wasm.br_table %c, %w [{
                wasm.br 0 %w : (i32);
            }, {
                wasm.br 1 %w : (i32);
            }] default {
                wasm.br 1 %w : (i32);
            } : (i32, i32);
        } : (i32) -> (i32);
        wasm.end %q : (i32);
    } : (i32) -> (i32);
    wasm.block {
        wasm.unreachable;
    } : () -> ();
    builtin.ret %r : (i32);
};"#,
    ));
    all.push(String::from(
        r#"
builtin.function public extern("C") @f(%c: i32, %f: i1, %k: i32) -> i32 {
    %r = wasm.block {
        scf.if %f then {
            wasm.br 0 %k : (i32);
        } else {
            scf.yield : ();
        } : (i1) -> ();
        wasm.if %c then {
            builtin.ret %k : (i32);
        } : (i32) -> ();
        wasm.end %k : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#,
    ));
    all
}

#[test]
fn every_analysis_runs_over_the_control_flow_programs() {
    for source in programs() {
        let program = Program::parse(&source);
        let errors = analyze(program.function);
        assert!(errors.is_empty(), "{errors:#?}\n{source}");
    }
}

// ---------------------------------------------------------------------------------------------
// Passes

/// `source`, the text of a function, after `pipeline` ran on it (in a module).
pub(super) fn run(pipeline: &str, source: &str) -> Result<String, Report> {
    let context = rich_context();
    let text = format!("builtin.module public @m {{\n{source}\n}};");
    let module =
        parse::parse_any(ParserConfig::new(context.clone()), "control", text).into_result()?;
    let pipeline = PassPipeline::parse_elements(&module.borrow().name(), pipeline)?;
    let mut passes = pipeline.load(context.clone(), Nesting::Implicit)?;
    passes.enable_verifier(true);
    passes.run(module)?;
    let text = module.borrow().to_string();
    drop(passes);
    Ok(text)
}

/// The text of the module holding `source`, as it is printed with no pass run.
fn printed(source: &str) -> String {
    let context = rich_context();
    let text = format!("builtin.module public @m {{\n{source}\n}};");
    let module = parse::parse_any(ParserConfig::new(context.clone()), "control", text)
        .into_result()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    module.borrow().to_string()
}

const PIPELINES: [&str; 5] = [
    "dce",
    "cse",
    "sccp",
    "canonicalizer",
    "canonicalizer,cse,sccp,canonicalizer,dce",
];

/// Every pass leaves programs that have nothing to remove as they are.
#[test]
fn no_pass_changes_the_meaning_of_the_programs() {
    for source in programs() {
        let before = normalize(&printed(&source));
        for pipeline in PIPELINES {
            let after = run(pipeline, &source)
                .unwrap_or_else(|report| panic!("{pipeline}: {}\n{source}", describe(&report)));
            // a constant nothing uses may go, and nothing else
            let after = normalize(&after);
            for name in [
                "wasm.block",
                "wasm.loop",
                "wasm.if",
                "wasm.br ",
                "wasm.br_table",
                "wasm.end",
                "wasm.unreachable",
                "builtin.ret",
            ] {
                assert_eq!(
                    before.matches(name).count(),
                    after.matches(name).count(),
                    "{pipeline}: `{name}` changed\n--- before\n{before}\n--- after\n{after}"
                );
            }
        }
    }
}

/// The result of a block that a nested branch gives another value is not replaced by the value the
/// body ends with.
#[test]
fn sccp_keeps_the_result_of_a_block_that_a_branch_leaves_with_another_value() {
    for pipeline in PIPELINES {
        let after = run(pipeline, BLOCK).unwrap();
        assert!(after.contains("builtin.ret %"), "{pipeline}: {after}");
        let ret = after.lines().find(|line| line.contains("builtin.ret")).unwrap();
        let block = after.lines().find(|line| line.contains("= wasm.block")).unwrap();
        let block_result = block.trim().split(' ').next().unwrap();
        assert!(
            ret.contains(block_result),
            "{pipeline}: `{ret}` does not return `{block_result}`"
        );
    }
}

/// The argument of a loop that a branch passes another value to is not folded to the initial one.
#[test]
fn sccp_keeps_the_body_argument_of_a_loop_that_a_branch_passes_another_value_to() {
    for pipeline in PIPELINES {
        let after = run(pipeline, LOOP).unwrap();
        // the loop still returns through the argument
        assert!(after.contains("wasm.br 2 %"), "{pipeline}: {after}");
        let arg = after.lines().find(|line| line.contains("^block")).unwrap();
        let arg = arg.split('(').nth(1).unwrap().split(':').next().unwrap();
        assert!(after.contains(&format!("wasm.br 2 {arg} : (i32);")), "{pipeline}: {after}");
    }
}

/// No pass removes a loop, a block or an `if` that holds a branch that leaves a construct around
/// it, whatever its results.
#[test]
fn no_pass_removes_a_construct_whose_branch_leaves_one_around_it() {
    let source = r#"
builtin.function public extern("C") @f(%c: i32, %k: i32) -> i32 {
    %r = wasm.block {
        %unused = wasm.loop %k {
        ^h(%i: i32):
            wasm.if %c then {
                wasm.br 2 %i : (i32);
            } : (i32) -> ();
            wasm.br 0 %i : (i32);
        } : (i32) -> (i32);
        wasm.end %k : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};"#;
    for pipeline in PIPELINES {
        let after = run(pipeline, source).unwrap();
        for name in ["wasm.block", "wasm.loop", "wasm.if", "wasm.br 2", "wasm.br 0"] {
            assert!(after.contains(name), "{pipeline}: `{name}` is not in\n{after}");
        }
    }
}

/// A block that holds nothing and gives nothing is dead code, and is removed; one that holds a
/// trap is not.
#[test]
fn dce_removes_an_empty_block_and_keeps_one_that_traps() {
    let empty = r#"
builtin.function public extern("C") @f(%a: i32) -> i32 {
    wasm.block {
        wasm.end : ();
    } : () -> ();
    builtin.ret %a : (i32);
};"#;
    let after = run("dce", empty).unwrap();
    assert!(!after.contains("wasm.block"), "{after}");
    let traps = empty.replace("wasm.end : ();", "wasm.unreachable;");
    let after = run("dce", &traps).unwrap();
    assert!(after.contains("wasm.block") && after.contains("wasm.unreachable"), "{after}");
}

/// `cse` merges two equal constants across a block, and does not merge across a construct what the
/// construct changes.
#[test]
fn cse_merges_what_is_equal_and_leaves_the_constructs() {
    let source = r#"
builtin.function public extern("C") @f(%c: i32, %a: i32) -> i32 {
    %x = wasm.add %a, %a;
    %r = wasm.block {
        %y = wasm.add %a, %a;
        wasm.end %y : (i32);
    } : () -> (i32);
    %z = wasm.add %x, %r;
    builtin.ret %z : (i32);
};"#;
    let after = run("cse", source).unwrap();
    // the inner `add` is the outer one
    assert_eq!(after.matches("wasm.add %").count(), 2, "{after}");
    assert!(after.contains("wasm.block"), "{after}");
}
