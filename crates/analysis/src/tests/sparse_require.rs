//! A transfer function may ask the solver for a lattice the framework holds a shared guard on: a
//! sparse backward transfer function for one of its operation's results, a sparse forward one for
//! one of its operation's operands. [DataFlowSolver::require] records the dependency beside the
//! shared guard, and the operation is visited again when that lattice changes.
//!
//! Each test reads the lattice only through [DataFlowSolver::require], and its program is written
//! so that the lattice changes after the operation's first visit: the answer is the fixed point
//! only if the operation is visited again.

use alloc::{collections::BTreeMap, string::String};

use runic_ir::{
    Backward, Forward, OpOperandImpl, Operation, ProgramPoint, Report, pass::AnalysisManager,
};

use super::sparse_backward::{DEMANDED, Demand, Demanded, Program, set};
use crate::{
    AnalysisStateGuard, AnalysisStateGuardMut, BuildableDataFlowAnalysis, DataFlowConfig,
    DataFlowSolver, Lattice, LatticeLike, SparseBackwardDataFlowAnalysis,
    SparseForwardDataFlowAnalysis, SparseLattice,
    analyses::{DeadCodeAnalysis, SparseConstantPropagation},
    sparse::{OperandLattices, SparseDataFlowAnalysis},
};

/// The test analysis of `sparse_backward`, whose transfer function reads whether a result is
/// demanded by requiring the result's lattice from the solver, not from the guards it is given.
#[derive(Default)]
struct DemandedThroughRequire;

impl BuildableDataFlowAnalysis for DemandedThroughRequire {
    type Strategy = SparseDataFlowAnalysis<Self, Backward>;

    fn new(_solver: &mut DataFlowSolver) -> Self {
        Self
    }
}

impl SparseBackwardDataFlowAnalysis for DemandedThroughRequire {
    type Lattice = Lattice<Demand>;

    fn visit_operation(
        &self,
        op: &Operation,
        operands: &mut OperandLattices<'_, Self::Lattice>,
        _results: &[AnalysisStateGuard<'_, Self::Lattice>],
        solver: &mut DataFlowSolver,
    ) -> Result<(), Report> {
        let mut result_demanded = false;
        for result in op.results().iter() {
            let result = result.borrow().as_value_ref();
            let lattice = solver.require::<Lattice<Demand>, _>(result, ProgramPoint::after(op));
            result_demanded |= lattice.value().0;
        }
        if !op.would_be_trivially_dead() || result_demanded {
            for index in 0..operands.len() {
                SparseLattice::meet(&mut operands[index], &DEMANDED);
            }
        }
        Ok(())
    }

    fn visit_branch_operand(&self, operand: &OpOperandImpl, solver: &mut DataFlowSolver) {
        Demanded::demand(operand, solver);
    }

    fn visit_call_operand(&self, operand: &OpOperandImpl, solver: &mut DataFlowSolver) {
        Demanded::demand(operand, solver);
    }

    fn set_to_exit_state(&self, lattice: &mut AnalysisStateGuardMut<'_, Self::Lattice>) {
        SparseLattice::meet(lattice, &DEMANDED);
    }
}

/// A loop whose header computes `%d` from `%k`; `%d` is demanded only through the back edge, which
/// passes it to `%y`, and `%y` becomes demanded only after `%d`'s definition is first visited (the
/// use that demands it is in the exit block, through `%x`). The definition, which requires its own
/// result `%d`, is visited again when `%d` becomes demanded, and demands `%k`. The control passes
/// `%y` back to itself, and neither `%d` nor `%k` is demanded.
#[test]
fn a_backward_transfer_function_may_require_its_own_result() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32, %b: i32, %k: i32) {
    cf.br ^head(%a, %b : i32, i32);
^head(%x: i32, %y: i32):
    %d = arith.add %k, %k;
    cf.cond_br %c ^head(%y, BACK : i32, i32), ^exit : (i1);
^exit:
    test.store %p, %x : (ptr<i32>, i32);
    builtin.ret;
};"#;
    let program = Program::parse(&source.replace("BACK", "%d"));
    assert_eq!(
        program.demanded_by::<DemandedThroughRequire>(),
        set(&["%p", "%c", "%a", "%b", "%x", "%y", "%d", "%k"])
    );
    let control = Program::parse(&source.replace("BACK", "%y"));
    assert_eq!(
        control.demanded_by::<DemandedThroughRequire>(),
        set(&["%p", "%c", "%a", "%b", "%x", "%y"])
    );
}

/// Whether a value may be computed from an argument of a public function: `true` is the top of
/// the lattice, and `join` moves towards it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Taint(bool);

impl LatticeLike for Taint {
    fn join(&self, other: &Self) -> Self {
        Self(self.0 || other.0)
    }

    fn meet(&self, other: &Self) -> Self {
        Self(self.0 && other.0)
    }
}

/// A sparse forward analysis whose transfer function reads its operands' lattices by requiring
/// them from the solver, not from the guards it is given: a result is tainted when an operand is.
#[derive(Default)]
struct TaintThroughRequire;

impl BuildableDataFlowAnalysis for TaintThroughRequire {
    type Strategy = SparseDataFlowAnalysis<Self, Forward>;

    fn new(_solver: &mut DataFlowSolver) -> Self {
        Self
    }
}

impl SparseForwardDataFlowAnalysis for TaintThroughRequire {
    type Lattice = Lattice<Taint>;

    fn visit_operation(
        &self,
        op: &Operation,
        _operands: &[AnalysisStateGuard<'_, Self::Lattice>],
        results: &mut [AnalysisStateGuardMut<'_, Self::Lattice>],
        solver: &mut DataFlowSolver,
    ) -> Result<(), Report> {
        let mut tainted = false;
        for operand in op.operands().iter() {
            let operand = operand.borrow().as_value_ref();
            let lattice = solver.require::<Lattice<Taint>, _>(operand, ProgramPoint::after(op));
            tainted |= lattice.value().0;
        }
        for result in results {
            SparseLattice::join(result, &Taint(tainted));
        }
        Ok(())
    }

    fn set_to_entry_state(&self, lattice: &mut AnalysisStateGuardMut<'_, Self::Lattice>) {
        SparseLattice::join(lattice, &Taint(true));
    }
}

/// Which of `names` [TaintThroughRequire] finds tainted in `source`.
fn tainted(source: &str, names: &[&str]) -> BTreeMap<String, bool> {
    let program = Program::parse(source);
    let mut config = DataFlowConfig::default();
    config.set_interprocedural(true);
    let mut solver = DataFlowSolver::new(config);
    solver.load::<DeadCodeAnalysis>();
    solver.load::<SparseConstantPropagation>();
    solver.load::<TaintThroughRequire>();
    solver
        .initialize_and_run(&program.root.borrow(), AnalysisManager::new(program.root, None))
        .expect("the analyses converge");
    program
        .values()
        .into_iter()
        .map(|value| (program.name_of(value), value))
        .filter(|(name, _)| names.contains(&name.as_str()))
        .map(|(name, value)| {
            let tainted =
                solver.get::<Lattice<Taint>, _>(&value).is_some_and(|lattice| lattice.value().0);
            (name, tainted)
        })
        .collect()
}

/// A loop whose header computes `%n` from its argument `%x`; `%x` is tainted only through the
/// back edge, after `%n`'s definition is first visited. The definition, which requires its own
/// operand `%x`, is visited again when `%x` becomes tainted; the control carries `%x` back
/// unchanged, and only `%y` is tainted.
#[test]
fn a_forward_transfer_function_may_require_its_own_operand() {
    let source = r#"builtin.function public extern("C") @f(%c: i1, %p: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    cf.br ^head(%zero, %p : i32, i32);
^head(%x: i32, %y: i32):
    %n = arith.add %x, %one;
    cf.cond_br %c ^head(BACK : i32, i32), ^exit : (i1);
^exit:
    builtin.ret %n : (i32);
};"#;
    let names = ["%x", "%y", "%n"];
    let expect = |states: [bool; 3]| {
        names
            .iter()
            .map(|name| String::from(*name))
            .zip(states)
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(tainted(&source.replace("BACK", "%y, %n"), &names), expect([true, true, true]));
    assert_eq!(tainted(&source.replace("BACK", "%x, %n"), &names), expect([false, true, false]));
}
