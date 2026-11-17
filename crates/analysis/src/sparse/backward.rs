use bitvec::bitvec;
use runic_ir::{
    AttributeRef, Backward, CallOpInterface, CallableOpInterface, EntityWithId, OpOperandImpl,
    OpOperandRange, OpResultRange, Operation, OperationRef, ProgramPoint, RegionBranchOpInterface,
    RegionBranchTerminatorOpInterface, RegionSuccessorIter, Report, SmallVec, StorableEntity,
    SuccessorOperands, ValueRef,
    traits::{BranchOpInterface, ReturnLike},
};

use super::{SparseDataFlowAnalysis, SparseLattice};
use crate::{
    AnalysisState, AnalysisStateGuard, AnalysisStateGuardMut, BuildableAnalysisState,
    DataFlowSolver,
    analyses::dce::{Executable, PredecessorState},
};

/// A sparse (backward) data-flow analysis for propagating SSA value lattices backwards across the
/// IR by implementing transfer functions for operations.
///
/// Visiting a program point in sparse backward data-flow analysis will invoke the transfer function
/// of the operation preceding the program point. Visiting a program point at the begining of block
/// will visit the block itself.
///
/// The framework holds no lattice guard when it calls [Self::visit_branch_operand],
/// [Self::visit_call_operand] or [Self::visit_external_call], so these may take the guards they
/// need from the solver.
#[allow(unused_variables)]
pub trait SparseBackwardDataFlowAnalysis: 'static {
    type Lattice: BuildableAnalysisState + SparseLattice;

    fn debug_name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }

    /// The operation transfer function.
    ///
    /// Given the result lattices, this function is expected to set the operand lattices. An
    /// operation that uses one value more than once has one lattice for all those operands (see
    /// [OperandLattices]).
    fn visit_operation(
        &self,
        op: &Operation,
        operands: &mut OperandLattices<'_, Self::Lattice>,
        results: &[AnalysisStateGuard<'_, Self::Lattice>],
        solver: &mut DataFlowSolver,
    ) -> Result<(), Report>;

    /// The transfer function for calls to external functions.
    ///
    /// This function is expected to set lattice values of the call operands, which it takes from
    /// `solver`: no lattice guard is held when it is called. The framework has required the
    /// lattices of the call's results at the point after the call, so the call is visited again
    /// when they change. By default, this calls [Self::visit_call_operand] for all operands.
    fn visit_external_call(&self, call: &dyn CallOpInterface, solver: &mut DataFlowSolver) {
        for operand in call.arguments() {
            let operand = operand.borrow();
            self.visit_call_operand(&operand, solver);
        }
    }

    /// Visit operands on branch instructions that are not forwarded.
    fn visit_branch_operand(&self, operand: &OpOperandImpl, solver: &mut DataFlowSolver);

    /// Visit operands on call instructions that are not forwarded.
    fn visit_call_operand(&self, operand: &OpOperandImpl, solver: &mut DataFlowSolver);

    /// Set the given lattice element(s) at control flow exit point(s).
    fn set_to_exit_state(&self, lattice: &mut AnalysisStateGuardMut<'_, Self::Lattice>);
}

/// The lattices of the operands of one operation, as a sparse backward analysis writes them.
///
/// It is indexed by operand position, as a slice of guards would be, and it holds one guard per
/// distinct value: an operation that uses one value at several positions (`arith.add %a, %a`) has
/// one lattice at all of them, as upstream MLIR's array of lattice pointers does, so what is
/// written through one position is read through the others.
pub struct OperandLattices<'guard, T: AnalysisState + 'static> {
    /// One guard per distinct operand value, in the order of first use
    lattices: SmallVec<[AnalysisStateGuardMut<'guard, T>; 2]>,
    /// The values the guards of `lattices` are for, in the same order
    values: SmallVec<[ValueRef; 2]>,
    /// For each operand position, the index in `lattices` of its value's guard
    positions: SmallVec<[usize; 4]>,
}

impl<'guard, T: BuildableAnalysisState> OperandLattices<'guard, T> {
    /// Take the guard of the lattice of each distinct value of `operands` from `solver`.
    fn new(operands: OpOperandRange<'_>, solver: &mut DataFlowSolver) -> Self {
        let mut lattices = SmallVec::new();
        let mut values = SmallVec::<[ValueRef; 2]>::new();
        let mut positions = SmallVec::with_capacity(operands.len());
        for operand in operands.iter() {
            let value = operand.borrow().as_value_ref();
            let index = match values.iter().position(|held| *held == value) {
                Some(index) => index,
                None => {
                    values.push(value);
                    lattices.push(solver.get_or_create_mut::<T, _>(value));
                    values.len() - 1
                }
            };
            positions.push(index);
        }
        Self {
            lattices,
            values,
            positions,
        }
    }
}

impl<'guard, T: AnalysisState + 'static> OperandLattices<'guard, T> {
    /// The number of operands, which is the number of positions this can be indexed by.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the operation has no operands.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The guards of the distinct lattices, each once: what a function that changes every operand
    /// lattice the same way (as [set_all_to_exit_states] does) is given.
    pub fn distinct(&mut self) -> &mut [AnalysisStateGuardMut<'guard, T>] {
        &mut self.lattices
    }

    /// The guard held for the lattice of `value`, if `value` is one of the operands.
    fn held_mut(&mut self, value: ValueRef) -> Option<&mut AnalysisStateGuardMut<'guard, T>> {
        let index = self.values.iter().position(|held| *held == value)?;
        Some(&mut self.lattices[index])
    }
}

impl<'guard, T: AnalysisState + 'static> core::ops::Index<usize> for OperandLattices<'guard, T> {
    type Output = AnalysisStateGuardMut<'guard, T>;

    /// The guard of the lattice of the operand at `position`.
    ///
    /// # Panics
    ///
    /// Panics if `position` is not less than [OperandLattices::len], as indexing a slice does.
    fn index(&self, position: usize) -> &Self::Output {
        &self.lattices[self.positions[position]]
    }
}

impl<T: AnalysisState + 'static> core::ops::IndexMut<usize> for OperandLattices<'_, T> {
    /// The guard of the lattice of the operand at `position`.
    ///
    /// # Panics
    ///
    /// Panics if `position` is not less than [OperandLattices::len], as indexing a slice does.
    fn index_mut(&mut self, position: usize) -> &mut Self::Output {
        &mut self.lattices[self.positions[position]]
    }
}

/// Set each of `lattices` to the exit state of `analysis`.
pub fn set_all_to_exit_states<A>(
    analysis: &A,
    lattices: &mut [AnalysisStateGuardMut<'_, <A as SparseBackwardDataFlowAnalysis>::Lattice>],
) where
    A: SparseBackwardDataFlowAnalysis,
{
    for lattice in lattices {
        analysis.set_to_exit_state(lattice);
    }
}

/// Recursively initialize the analysis on nested operations and blocks.
pub(super) fn initialize_recursively<A>(
    analysis: &SparseDataFlowAnalysis<A, Backward>,
    op: &Operation,
    solver: &mut DataFlowSolver,
) -> Result<(), Report>
where
    A: SparseBackwardDataFlowAnalysis,
{
    log::trace!(target: analysis.analysis.debug_name(), "initializing op recursively");
    visit_operation(analysis, op, solver)?;

    for region in op.regions() {
        for block in region.body() {
            log::trace!(target: analysis.analysis.debug_name(), "initializing analysis for block {}", block.id());
            {
                let state =
                    solver.get_or_create::<Executable, _>(ProgramPoint::at_start_of(&*block));
                AnalysisStateGuard::subscribe(&state, analysis);
            }

            // Initialize ops in reverse order, so we can do as much initial propagation as possible
            // without having to go through the solver queue.
            let mut ops = block.body().back();
            log::trace!(target: analysis.analysis.debug_name(), "initializing ops of {} bottom-up", block.id());
            while let Some(op) = ops.as_pointer() {
                ops.move_prev();

                let op = op.borrow();
                initialize_recursively(analysis, &op, solver)?;
            }
            log::trace!(target: analysis.analysis.debug_name(), "all ops of {} have been initialized", block.id());
        }
    }

    Ok(())
}

/// Visit an operation. If this is a call operation or an operation with
/// region control-flow, then its operand lattices are set accordingly.
/// Otherwise, the operation transfer function is invoked.
///
/// The lattices are guarded at run time, and one lattice may be reached by more than one path of
/// a visit: an operation may use one value twice, a branch may pass a block's own argument back to
/// it, a return may return the results of a recursive call of its function. No path holds a guard
/// while another path may take one on the same lattice: the paths that forward values along an
/// edge (branches, region branches and their terminators, calls) hold one operand lattice at a
/// time, and the hooks they call ([SparseBackwardDataFlowAnalysis::visit_branch_operand] and the
/// others) are called with none held. Where a lattice read is one already held, it is read through
/// the guard held, and the dependency is recorded through that guard.
///
/// Every visit ends: the lattices have finite height, and `meet` only moves a lattice down, so a
/// lattice changes a finite number of times, and a point is visited again only when a lattice it
/// read changes.
pub(super) fn visit_operation<A>(
    analysis: &SparseDataFlowAnalysis<A, Backward>,
    op: &Operation,
    solver: &mut DataFlowSolver,
) -> Result<(), Report>
where
    A: SparseBackwardDataFlowAnalysis,
{
    // If we're in a dead block, bail out.
    let in_dead_block = op.parent().is_some_and(|block| {
        !solver
            .get_or_create_mut::<Executable, _>(ProgramPoint::at_start_of(block))
            .is_live()
    });
    if in_dead_block {
        log::trace!(target: analysis.analysis.debug_name(), "skipping analysis for op in dead/non-executable block: {op}");
        return Ok(());
    }

    // Whatever the path below, the operation is visited again when the lattice of one of its
    // results changes. The guards are let go at once, and each path takes the guards it needs: a
    // region branch whose regions may be skipped forwards its operands to its own results, which
    // `visit_region_successors` reads, and the transfer function's path takes them again below.
    let current_point = ProgramPoint::after(op);
    drop(get_lattice_elements_for::<A>(
        analysis,
        current_point,
        op.results().all(),
        solver,
    ));

    // Block arguments of region branch operations flow back into the operands of the parent op
    if let Some(branch) = op.as_trait::<dyn RegionBranchOpInterface>() {
        log::trace!(target: analysis.analysis.debug_name(), "op implements RegionBranchOpInterface - handling as special case");
        visit_region_successors(analysis, branch, solver);
        return Ok(());
    }

    // Block arguments of successor blocks flow back into our operands.
    if let Some(branch) = op.as_trait::<dyn BranchOpInterface>() {
        log::trace!(target: analysis.analysis.debug_name(), "op implements BranchOpInterface - handling as special case");

        // We remember all operands not forwarded to any block in a bitvector.
        // We can't just cut out a range here, since the non-forwarded ops might be non-contiguous
        // (if there's more than one successor).
        let mut unaccounted = bitvec![1; op.num_operands()];

        for successor_index in 0..branch.num_successors() {
            let successor_operands = branch.get_successor_operands(successor_index);
            let num_produced = successor_operands.num_produced();
            let block = op.successors()[successor_index].successor();
            let block = block.borrow();
            // The forwarded operand at `position` is passed to the argument after the produced
            // ones, as upstream's `getBranchSuccessorArgument` computes it.
            for (position, operand) in successor_operands.forwarded().iter().enumerate() {
                let operand = operand.borrow();
                unaccounted.set(operand.index(), false);
                if let Some(argument) = block.arguments().get(position + num_produced) {
                    meet_operand_with_input(
                        analysis,
                        operand.as_value_ref(),
                        argument.borrow().as_value_ref(),
                        current_point,
                        solver,
                    );
                }
            }
        }

        // Operands not forwarded to successor blocks are typically parameters of the branch
        // operation itself (for example the boolean for if/else).
        for index in unaccounted.iter_ones() {
            let operand = op.operands().all()[index].borrow();
            analysis.visit_branch_operand(&operand, solver);
        }

        return Ok(());
    }

    // For function calls, connect the arguments of the entry blocks to the operands of the call op
    // that are forwarded to these arguments.
    if let Some(call) = op.as_trait::<dyn CallOpInterface>() {
        log::trace!(target: analysis.analysis.debug_name(), "op implements CallOpInterface - handling as special case");

        // TODO: resolve_in_symbol_table
        if let Some(target) = call.resolve() {
            let callable = target.borrow();
            log::trace!(target: analysis.analysis.debug_name(), "resolved callee as {}", target.as_symbol_ref());
            // Not all operands of a call op forward to arguments. Such operands are stored in
            // `unaccounted`.
            let mut unaccounted = bitvec![1; op.num_operands()];

            // If the call invokes an external function (or a function treated as external due to
            // config), defer to the corresponding extension hook. By default, it just does
            // `visit_call_operand` for all operands.
            let arg_operands = call.arguments();
            let region = callable.get_callable_region();
            let Some(region) = region.filter(|region| {
                !region.borrow().is_empty() && solver.config().is_interprocedural()
            }) else {
                log::trace!(target: analysis.analysis.debug_name(), "{} is an external callee", target.as_symbol_ref());
                analysis.visit_external_call(call, solver);
                return Ok(());
            };

            // Otherwise, propagate information from the entry point of the function back to
            // operands whenever possible. A recursive call may pass an entry argument of its
            // callee to that argument (see `meet_operand_with_input`).
            log::trace!(target: analysis.analysis.debug_name(), "propagating value lattices from callee entry to call operands");
            let region = region.borrow();
            let block = region.entry();
            for (block_arg, arg_operand) in block.arguments().iter().zip(arg_operands.iter()) {
                let arg_operand = arg_operand.borrow();
                meet_operand_with_input(
                    analysis,
                    arg_operand.as_value_ref(),
                    block_arg.borrow().as_value_ref(),
                    current_point,
                    solver,
                );
                unaccounted.set(arg_operand.index as usize, false);
            }

            // Handle the operands of the call op that aren't forwarded to any arguments.
            for index in unaccounted.iter_ones() {
                let operand = op.operands().all()[index].borrow();
                analysis.visit_call_operand(&operand, solver);
            }

            return Ok(());
        }
    }

    // When the region of an op implementing `RegionBranchOpInterface` has a terminator implementing
    // `RegionBranchTerminatorOpInterface` or a return-like terminator, the region's successors'
    // arguments flow back into the "successor operands" of this terminator.
    //
    // A successor operand with respect to an op implementing `RegionBranchOpInterface` is an
    // operand that is forwarded to a region successor's input. There are two types of successor
    // operands: the operands of this op itself and the operands of the terminators of the regions
    // of this op.
    if let Some(terminator) = op.as_trait::<dyn RegionBranchTerminatorOpInterface>() {
        log::trace!(target: analysis.analysis.debug_name(), "op implements RegionBranchTerminatorOpInterface");
        let parent_op = op.parent_op().unwrap();
        let parent_op = parent_op.borrow();
        if let Some(branch) = parent_op.as_trait::<dyn RegionBranchOpInterface>() {
            log::trace!(
                target: analysis.analysis.debug_name(),
                "op's parent implements RegionBranchOpInterface - handling as special case"
            );
            visit_region_successors_from_terminator(analysis, terminator, branch, solver);
            return Ok(());
        }
    }

    // The remaining paths hold the guards of all operand lattices at once, and those of the
    // result lattices. An operand is never one of the operation's own results (the verifier
    // refuses an operation that reads its own result, in a graph region too), so the two never
    // name one lattice; operands that are one value share one guard (see [OperandLattices]).
    let results =
        get_lattice_elements_for::<A>(analysis, current_point, op.results().all(), solver);
    let mut operands = OperandLattices::<A::Lattice>::new(op.operands().all(), solver);

    if op.implements::<dyn ReturnLike>() {
        log::trace!(target: analysis.analysis.debug_name(), "op implements ReturnLike");
        // Going backwards, the operands of the return are derived from the results of all CallOps
        // calling this CallableOp.
        let parent_op = op.parent_op().unwrap();
        let parent_op = parent_op.borrow();
        if let Some(callable) = parent_op.as_trait::<dyn CallableOpInterface>() {
            log::trace!(target: analysis.analysis.debug_name(), "op's parent implements CallableOpInterface - visiting call sites");
            let callsites = solver.require::<PredecessorState, _>(
                ProgramPoint::after(callable.as_operation()),
                current_point,
            );
            if callsites.all_predecessors_known() {
                log::trace!(
                    target: analysis.analysis.debug_name(),
                    "found all {} call sites of the current callable op",
                    callsites.known_predecessors().len()
                );
                log::trace!(target: analysis.analysis.debug_name(), "meeting lattices of return values and call site results");
                for call in callsites.known_predecessors() {
                    let call = call.borrow();
                    for (index, result) in call.results().iter().enumerate().take(operands.len()) {
                        let result = result.borrow().as_value_ref();
                        // A recursive call of this function may be the definition of what this
                        // return returns (`%r = call @f(..); ret %r`, or two results returned
                        // swapped). Then the call's result lattice is one of the operand lattices
                        // held, and the solver cannot hand it out again: it is read from the
                        // guard held, and the dependency is recorded through that guard
                        // (`AnalysisStateGuardMut::require` is `DataFlowSolver::require` for a
                        // state already borrowed). A later change of that lattice, by this visit
                        // or another, visits the return again, and the meet is made with the new
                        // value. When the result is the operand at `index` itself, the meet
                        // changes nothing.
                        if let Some(held) = operands.held_mut(result) {
                            held.require(analysis, current_point);
                            let value = held.lattice().clone();
                            operands[index].meet(&value);
                        } else {
                            let lattice = get_lattice_element_for::<A>(
                                analysis,
                                current_point,
                                result,
                                solver,
                            );
                            operands[index].meet(lattice.lattice());
                        }
                    }
                }
            } else {
                // If we don't know all the callers, we can't know where the returned values go.
                // Note that, in particular, this will trigger for the return ops of any public
                // functions.
                log::trace!(
                    target: analysis.analysis.debug_name(),
                    "not all call sites are known - setting return value lattices to exit state"
                );
                set_all_to_exit_states(analysis, operands.distinct());
            }
            return Ok(());
        }
    }

    log::trace!(target: analysis.analysis.debug_name(), "invoking {}::visit_operation", core::any::type_name::<A>());
    analysis.visit_operation(op, &mut operands, &results, solver)
}

/// Meet the lattice of `operand` with the lattice of `input`, the value it is passed to along an
/// edge of control flow, and visit `point` again when the lattice of `input` changes.
///
/// The two are one lattice when the edge passes a value to itself: a branch that passes a block's
/// own argument back to it, a continue that carries an entry argument of its loop back to that
/// argument, a recursive call that passes an entry argument of its callee to that argument. The
/// solver cannot hand out a second guard on a lattice already guarded, so one guard is taken, and
/// the dependency is recorded through it (`AnalysisStateGuardMut::require` is
/// `DataFlowSolver::require` for a state already borrowed). The meet itself is left out: the meet
/// of a lattice with itself changes nothing, now or after any later change of the lattice, so the
/// answer is the one two lattices would give, and with the dependency recorded nothing else is
/// lost. Every other edge passes a value to another value, and the two guards are on two
/// lattices.
fn meet_operand_with_input<A>(
    analysis: &SparseDataFlowAnalysis<A, Backward>,
    operand: ValueRef,
    input: ValueRef,
    point: ProgramPoint,
    solver: &mut DataFlowSolver,
) where
    A: SparseBackwardDataFlowAnalysis,
{
    if operand == input {
        let mut lattice = get_lattice_element_mut::<A>(analysis, operand, solver);
        lattice.require(analysis, point);
        return;
    }
    let input_lattice = get_lattice_element_for::<A>(analysis, point, input, solver);
    let mut operand_lattice = get_lattice_element_mut::<A>(analysis, operand, solver);
    operand_lattice.meet(input_lattice.lattice());
}

/// Visit an op with regions (like e.g. `scf.while`)
///
/// The operands of the op are defined outside it, and the inputs of its entry successors are the
/// arguments of its regions or its own results, so an operand is never an input. One operand may
/// be forwarded to more than one input (`scf.while %x, %x`, or one value to two regions): each
/// edge takes its guards and lets them go before the next.
fn visit_region_successors<A>(
    analysis: &SparseDataFlowAnalysis<A, Backward>,
    branch: &dyn RegionBranchOpInterface,
    solver: &mut DataFlowSolver,
) where
    A: SparseBackwardDataFlowAnalysis,
{
    let op = branch.as_operation();
    let point = ProgramPoint::after(op);

    let mut const_operands =
        SmallVec::<[Option<AttributeRef>; 2]>::with_capacity(op.num_operands());
    const_operands.resize_with(op.num_operands(), || None);
    let successors = branch.get_entry_successor_regions(&const_operands);

    // All operands not forwarded to any successor. This set can be non-contiguous in the presence
    // of multiple successors.
    let mut unaccounted = bitvec![1; op.num_operands()];

    for successor in successors {
        let operands = branch.get_entry_successor_operands(*successor.branch_point());
        let inputs = successor.successor_inputs();
        for (operand, input) in operands.forwarded().iter().zip(inputs.iter()) {
            let operand = operand.borrow();
            meet_operand_with_input(analysis, operand.as_value_ref(), input, point, solver);
            unaccounted.set(operand.index as usize, false);
        }
    }

    // All operands not forwarded to regions are typically parameters of the branch operation itself
    // (for example the boolean for if/else).
    for index in unaccounted.iter_ones() {
        analysis.visit_branch_operand(&op.operands().all()[index].borrow(), solver);
    }
}

/// Visit a `RegionBranchTerminatorOpInterface` to compute the lattice values
/// of its operands, given its parent op `branch`. The lattice value of an
/// operand is determined based on the corresponding arguments in
/// `terminator`'s region successor(s).
///
/// An operand may be one of the inputs: a continue that carries an entry argument of its loop back
/// to that argument (see `meet_operand_with_input`), or to another argument, which is another
/// lattice.
fn visit_region_successors_from_terminator<A>(
    analysis: &SparseDataFlowAnalysis<A, Backward>,
    terminator: &dyn RegionBranchTerminatorOpInterface,
    branch: &dyn RegionBranchOpInterface,
    solver: &mut DataFlowSolver,
) where
    A: SparseBackwardDataFlowAnalysis,
{
    assert!(
        OperationRef::ptr_eq(
            &terminator.as_operation().parent_op().unwrap(),
            &branch.as_operation().as_operation_ref()
        ),
        "expected `branch` to be the parent op of `terminator`"
    );

    let num_operands = terminator.num_operands();
    let mut const_operands = SmallVec::<[Option<AttributeRef>; 2]>::with_capacity(num_operands);
    const_operands.resize_with(num_operands, || None);

    let terminator_op = terminator.as_operation();
    let point = ProgramPoint::after(terminator_op);
    let successors = terminator.get_successor_regions(&const_operands);
    let successors = RegionSuccessorIter::new(terminator_op, successors);

    // All operands not forwarded to any successor. This set can be non-contiguous in the presence
    // of multiple successors.
    let mut unaccounted = bitvec![1; num_operands];

    for successor in successors {
        let inputs = successor.successor_inputs();
        let operands = terminator.get_successor_operands(*successor.branch_point());
        for (operand, input) in operands.forwarded().iter().zip(inputs.iter()) {
            let operand = operand.borrow();
            meet_operand_with_input(analysis, operand.as_value_ref(), input, point, solver);
            unaccounted.set(operand.index(), false);
        }
    }

    // Visit operands of the branch op not forwarded to the next region. (Like e.g. the boolean of
    // `scf.conditional`)
    for index in unaccounted.iter_ones() {
        analysis.visit_branch_operand(&terminator_op.operands()[index].borrow(), solver);
    }
}

#[inline]
fn get_lattice_element_mut<'guard, A>(
    analysis: &SparseDataFlowAnalysis<A, Backward>,
    value: ValueRef,
    solver: &mut DataFlowSolver,
) -> AnalysisStateGuardMut<'guard, <A as SparseBackwardDataFlowAnalysis>::Lattice>
where
    A: SparseBackwardDataFlowAnalysis,
{
    log::trace!(target: analysis.debug_name(), "getting lattice for {value}");
    solver.get_or_create_mut::<_, _>(value)
}

#[inline]
fn get_lattice_element_for<'guard, A>(
    analysis: &SparseDataFlowAnalysis<A, Backward>,
    point: ProgramPoint,
    value: ValueRef,
    solver: &mut DataFlowSolver,
) -> AnalysisStateGuard<'guard, <A as SparseBackwardDataFlowAnalysis>::Lattice>
where
    A: SparseBackwardDataFlowAnalysis,
{
    log::trace!(target: analysis.debug_name(), "getting lattice for {value} at {point}");
    solver.require::<_, _>(value, point)
}

/// Get the lattice elements for a range of values, and also set up dependencies so that the
/// analysis on the given ProgramPoint is re-invoked if any of the values change.
fn get_lattice_elements_for<'guard, A>(
    analysis: &SparseDataFlowAnalysis<A, Backward>,
    point: ProgramPoint,
    values: OpResultRange<'_>,
    solver: &mut DataFlowSolver,
) -> SmallVec<[AnalysisStateGuard<'guard, <A as SparseBackwardDataFlowAnalysis>::Lattice>; 2]>
where
    A: SparseBackwardDataFlowAnalysis,
{
    log::trace!(target: analysis.debug_name(), "getting lattices for {:#?}", values.as_slice());
    let mut results = SmallVec::with_capacity(values.len());
    for value in values.iter() {
        let lattice = solver.require(value.borrow().as_value_ref(), point);
        results.push(lattice);
    }
    results
}
