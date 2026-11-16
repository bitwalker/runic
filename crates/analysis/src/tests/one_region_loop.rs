//! The analyses of this crate over a loop with one region (`test.loop`), whose back edge is a
//! `test.continue` and not an edge of its region graph: the region is reached from the continue,
//! the arguments of its entry block receive what the continue carries, and a value used at the top
//! of the region is live at the continue.
//!
//! Each test has a control: the same program whose `test.continue` is replaced by what leaves the
//! loop to its parent, which shows the analysis learns something from the program, and that what
//! it reports for the first is because of the continue.

use alloc::{rc::Rc, string::ToString, vec::Vec};

use runic_ir::{
    BlockRef, Context, Operation, OperationRef, ProgramPoint, ValueRef,
    dialects::{builtin::Function, test::TestDialect},
    parse::{self, ParserConfig},
    pass::AnalysisManager,
    testing::text::describe,
};

use crate::{
    DataFlowConfig, DataFlowSolver, Lattice,
    analyses::{
        DeadCodeAnalysis, LivenessAnalysis, SparseConstantPropagation,
        constant_propagation::ConstantValue,
        dce::{Executable, PredecessorState},
        liveness::LOOP_EXIT_DISTANCE,
    },
};

/// `%i` starts as 0 and is replaced by `%n`, which is `%i + 1`, when `CONTINUE` runs. `%k` is used
/// at the top of the body only. The guard's condition is not known.
const LOOP: &str = r#"
builtin.function public extern("C") @f(%c: i1, %k: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    %r = test.loop %zero {
    ^head(%i: i32):
        %n = arith.add %i, %one;
        %m = arith.add %n, %k;
        test.guard %c {
            CONTINUE
        } : (i1);
        test.yield %i : (i32);
    } : (i32) -> (i32);
    builtin.ret %r : (i32);
};"#;

fn with_continue() -> alloc::string::String {
    LOOP.replace("CONTINUE", "test.continue %n : (i32);")
}

/// The control: the guard's region ends, and the loop goes on to its `test.yield`.
fn without_continue() -> alloc::string::String {
    LOOP.replace("CONTINUE", "test.yield : ();")
}

struct Program {
    #[allow(unused)]
    context: Rc<Context>,
    function: OperationRef,
}

impl Program {
    fn parse(source: &str) -> Self {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<runic_dialect_arith::ArithDialect>();
        context.get_or_register_dialect::<TestDialect>();
        let config = ParserConfig::new(context.clone());
        let function = parse::parse::<Function>(config, "one_region_loop", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        Self {
            context,
            function: function.as_operation_ref(),
        }
    }

    /// The first operation named `name`, in pre-order.
    fn op(&self, name: &str) -> OperationRef {
        let mut found = None;
        self.function.borrow().prewalk_all(|op: &Operation| {
            if found.is_none() && op.name().to_string() == name {
                found = Some(op.as_operation_ref());
            }
        });
        found.unwrap_or_else(|| panic!("no {name}"))
    }

    /// The `index`th result of the first operation named `name`.
    fn result(&self, name: &str, index: usize) -> ValueRef {
        self.op(name).borrow().results()[index].borrow().as_value_ref()
    }

    fn argument(&self, index: usize) -> ValueRef {
        let function = self.function.borrow();
        let entry = function.downcast_ref::<Function>().unwrap().entry_block();
        entry.borrow().arguments()[index].borrow().as_value_ref()
    }

    /// The entry block of the region of the first operation named `name`.
    fn entry_of(&self, name: &str) -> BlockRef {
        self.op(name).borrow().region(0).entry_block_ref().unwrap()
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

/// What constant propagation knows of a value: `None` when it is not one constant.
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

/// The region of a one-region loop is executable, and so is the region nested in it that holds
/// the continue; the continue is a known predecessor of the entry of the loop's region.
#[test]
fn the_region_of_a_loop_is_executable_and_its_continue_is_a_predecessor_of_its_entry() {
    let program = Program::parse(&with_continue());
    let solver = program.solve();
    let head = program.entry_of("test.loop");
    let guarded = program.entry_of("test.guard");
    for block in [head, guarded] {
        assert!(
            solver
                .get::<Executable, _>(&ProgramPoint::at_start_of(block))
                .is_some_and(|state| state.is_live()),
            "a block is not executable"
        );
    }
    let predecessors = solver
        .get::<PredecessorState, _>(&ProgramPoint::at_start_of(head))
        .expect("the entry of the region has predecessors");
    let names = predecessors
        .known_predecessors()
        .iter()
        .map(|op| op.borrow().name().to_string())
        .collect::<Vec<_>>();
    assert!(names.iter().any(|name| name == "test.continue"), "{names:?}");
    assert!(names.iter().any(|name| name == "test.loop"), "{names:?}");

    // The control has no continue, so the entry of the region has the loop as its one
    // predecessor.
    let control = Program::parse(&without_continue());
    let solver = control.solve();
    let predecessors = solver
        .get::<PredecessorState, _>(&ProgramPoint::at_start_of(control.entry_of("test.loop")))
        .expect("the entry of the region has predecessors");
    let names = predecessors
        .known_predecessors()
        .iter()
        .map(|op| op.borrow().name().to_string())
        .collect::<Vec<_>>();
    assert_eq!(names, ["test.loop"]);
}

/// The values a continue carries reach the arguments of the entry block: a loop-carried value that
/// a continue changes is not a constant, and is one when nothing changes it.
#[test]
fn constant_propagation_joins_what_a_continue_carries_into_the_entry_arguments() {
    let program = Program::parse(&with_continue());
    let solver = program.solve();
    let head = program.entry_of("test.loop");
    let argument = head.borrow().arguments()[0].borrow().as_value_ref();
    assert_eq!(constant(&solver, argument), None, "0 on entry, and 1 after a continue");
    assert_eq!(constant(&solver, program.result("arith.add", 0)), None);
    assert_eq!(constant(&solver, program.result("test.loop", 0)), None);

    let control = Program::parse(&without_continue());
    let solver = control.solve();
    let head = control.entry_of("test.loop");
    let argument = head.borrow().arguments()[0].borrow().as_value_ref();
    assert_eq!(constant(&solver, argument), Some(0));
    assert_eq!(constant(&solver, control.result("test.loop", 0)), Some(0));
}

/// A value used at the top of a one-region loop is live at the continue that goes back to it, and
/// across the whole region; with no continue it is dead there.
#[test]
fn liveness_carries_a_use_at_the_top_of_the_region_back_across_the_continue() {
    let program = Program::parse(&with_continue());
    let liveness = program.liveness();
    let k = program.argument(1);
    let the_continue = program.op("test.continue");
    assert!(
        liveness.is_live_before(k, &the_continue.borrow()),
        "`%k` is used when the loop goes back to its top"
    );
    assert!(liveness.is_live_after(k, &the_continue.borrow()));
    // It is live before the guard that holds the continue, and it is dead where the loop is left,
    // at the yield after the guard: nothing after it uses `%k`.
    let guard = program.op("test.guard");
    assert!(liveness.is_live_before(k, &guard.borrow()));
    let yield_op = program.op("test.yield");
    assert!(!liveness.is_live_before(k, &yield_op.borrow()));

    let control = Program::parse(&without_continue());
    let liveness = control.liveness();
    let k = control.argument(1);
    let guard = control.op("test.guard");
    assert!(
        !liveness.is_live_before(k, &guard.borrow()),
        "nothing goes back to the use of `%k`"
    );
}

/// Leaving a one-region loop that a continue repeats is a loop exit, which liveness prices as one
/// (the next use of a value used after the loop is farther by [LOOP_EXIT_DISTANCE]); leaving a
/// loop nothing repeats is not. The answer comes from `is_repetitive_region` of the loop.
#[test]
fn liveness_prices_leaving_a_loop_that_a_continue_repeats_as_a_loop_exit() {
    let after_loop = |source: &str| {
        let source = source.replace(
            "builtin.ret %r : (i32);",
            "%z = arith.add %r, %k;\n    builtin.ret %z : (i32);",
        );
        let program = Program::parse(&source);
        let liveness = program.liveness();
        let k = program.argument(1);
        liveness
            .next_uses_at(&ProgramPoint::before(program.op("test.yield")))
            .map(|uses| uses.distance(k))
            .expect("the point has next uses")
    };
    assert!(after_loop(&with_continue()) >= LOOP_EXIT_DISTANCE);
    assert!(after_loop(&without_continue()) < LOOP_EXIT_DISTANCE);
}

/// A continue may carry an argument of the entry block to itself, which a loop whose two regions
/// differ never does: every analysis runs over it, and a loop-carried value nothing changes is
/// still the value the loop starts with.
#[test]
fn a_continue_that_carries_an_entry_argument_to_itself_is_analyzed() {
    let source = r#"
builtin.function public extern("C") @f(%c: i1, %k: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = test.loop %zero {
    ^head(%i: i32):
        %m = arith.add %i, %k;
        test.guard %c {
            test.continue %i : (i32);
        } : (i1);
        test.yield %i : (i32);
    } : (i32) -> (i32);
    builtin.ret %r : (i32);
};"#;
    let program = Program::parse(source);
    let errors = super::analyze(program.function);
    assert!(errors.is_empty(), "{errors:#?}");

    let solver = program.solve();
    let head = program.entry_of("test.loop");
    let argument = head.borrow().arguments()[0].borrow().as_value_ref();
    assert_eq!(constant(&solver, argument), Some(0), "nothing changes the carried value");
    assert_eq!(constant(&solver, program.result("test.loop", 0)), Some(0));
    program.liveness();
}
