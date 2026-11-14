use alloc::vec::Vec;

use runic_core::{
    Backward, Builder, EntityMut, Forward, FxHashSet, OpBuilder, Operation, OperationName,
    OperationRef, ProgramPoint, RawWalk, Region, RegionBranchOpInterface,
    RegionExitTerminatorOpInterface, RegionRef, Report, SmallVec, Usable, Value, ValueRef,
    adt::SmallDenseMap,
    dominance::DominanceInfo,
    matchers::{self, Matcher},
    pass::{Pass, PassExecutionState, PostPassStatus},
    traits::{BinaryOp, ConstantLike, ReturnLike, Transparent},
};

/// Check whether `operation` is the sole _non-transparent_ user of `value`.
///
/// Transparent users are excluded, because they are purely informational and their uses are not
/// considered for purposes of computing liveness.
fn is_sole_non_transparent_user(value: &dyn Value, operation: OperationRef) -> bool {
    value
        .iter_uses()
        .all(|user| user.owner == operation || user.owner.borrow().implements::<dyn Transparent>())
}

/// Whether any region of `op` (transitively) contains an operation that LEAVES the region it
/// terminates — a return, an `scf.break`, an `scf.continue`.
fn regions_contain_an_exit(op: &Operation) -> bool {
    let root = op.as_operation_ref();
    let mut found = false;
    op.prewalk_all(|nested: &Operation| {
        if OperationRef::ptr_eq(&nested.as_operation_ref(), &root) {
            return;
        }
        found |= nested.implements::<dyn ReturnLike>()
            || nested.implements::<dyn RegionExitTerminatorOpInterface>();
    });
    found
}

/// Whether sinking `def` to a use in `user_region` would carry an early exit into a region other
/// than the one the exit was written in.
///
/// An operation whose regions contain an exit is lowered by rewriting the region that holds it,
/// and each kind of region answers a different exit protocol — a loop's condition region has no
/// channel to carry an exit at all, so an exit sunk into one is refused outright by codegen. A
/// `?` bound before a loop and read by that loop's condition is exactly that shape: the sole use
/// of the `scf.if` the `?` lowers to sits in the loop's `before` region.
///
/// Sinking is a scheduling convenience, so it gives way: an operation carrying an exit stays in
/// the region it was written in, however few uses its result has.
fn would_sink_an_exit_between_regions(def: &Operation, user_region: Option<RegionRef>) -> bool {
    if !regions_contain_an_exit(def) {
        return false;
    }
    match (def.parent_region(), user_region) {
        (Some(from), Some(to)) => !RegionRef::ptr_eq(&from, &to),
        _ => true,
    }
}

/// Whether moving `def` to a use in `user_region` would move it into a region other than its own
/// that may be executed more than once each time the region holding `def` is.
///
/// Every region between the two must belong to a region operation that executes it at most once
/// (the test [ControlFlowSink] uses for the regions it sinks into), so that the moved operation is
/// executed at most as often as before. A region operation that is not a
/// [RegionBranchOpInterface], or a use outside the region holding `def`, counts as "may be
/// executed more than once".
fn would_move_into_a_region_that_may_repeat(
    def: &Operation,
    user_region: Option<RegionRef>,
) -> bool {
    let (Some(from), Some(mut region)) = (def.parent_region(), user_region) else {
        return true;
    };
    while !RegionRef::ptr_eq(&region, &from) {
        let Some(owner) = region.borrow().parent() else {
            return true;
        };
        let executed_at_most_once = {
            let owner = owner.borrow();
            let Some(branch) = owner.as_trait::<dyn RegionBranchOpInterface>() else {
                return true;
            };
            let mut regions = SmallVec::<[RegionRef; 4]>::default();
            get_singly_executed_regions_to_sink(branch, &mut regions);
            regions.iter().any(|singly| RegionRef::ptr_eq(singly, &region))
        };
        if !executed_at_most_once {
            return true;
        }
        let Some(outer) = owner.borrow().parent_region() else {
            return true;
        };
        region = outer;
    }
    false
}

/// Erase all transparent operations that reference the given value.
///
/// This is used before erasing a defining op whose result is only kept alive by
/// transparent uses.
fn erase_transparent_users(value: ValueRef) {
    // Handle debug value users first: erasing them must leave a `di.debug_kill` marker behind,
    // so debuggers report the affected variables as optimized out instead of showing a stale
    // earlier location.
    runic_core::dialects::debuginfo::transform::erase_debug_info(&value);
    let transparent_ops: SmallVec<[OperationRef; 2]> = {
        let v = value.borrow();
        v.iter_uses()
            .filter_map(|user| {
                if user.owner.borrow().implements::<dyn Transparent>() {
                    Some(user.owner)
                } else {
                    None
                }
            })
            .collect()
    };
    for mut op in transparent_ops {
        op.borrow_mut().erase();
    }
}

/// This transformation moves an effect-free operation into a region of a region operation when
/// every user of its results is inside that region, and the region is executed at most once, so
/// that the operation is not executed on paths that do not need its results.
///
/// Operations with memory effects are never moved. An operation other than a constant is not moved
/// into the region unless every region between the two, not only the target, is executed at most
/// once each time the region holding the operation is: an operation that may trap is never moved to
/// where it would execute more often, so a definition before a loop stays out of an `scf.if` in the
/// loop's body.
///
/// # Implementation
///
/// Given a list of regions, perform control flow sinking on them. For each region, control-flow
/// sinking moves operations that dominate the region but whose only users are in the region into
/// the regions so that they aren't executed on paths where their results are not needed.
///
/// TODO: For the moment, this is a *simple* control-flow sink, i.e., no duplicating of ops. It
/// should be made to accept a cost model to determine whether duplicating a particular op is
/// profitable.
///
/// Example:
///
/// ```mlir
/// %0 = arith.addi %arg0, %arg1
/// scf.if %cond {
///   scf.yield %0
/// } else {
///   scf.yield %arg2
/// }
/// ```
///
/// After control-flow sink:
///
/// ```mlir
/// scf.if %cond {
///   %0 = arith.addi %arg0, %arg1
///   scf.yield %0
/// } else {
///   scf.yield %arg2
/// }
/// ```
///
/// If using the `control_flow_sink` function, callers can supply a callback
/// `should_move_into_region` that determines whether the given operation that only has users in the
/// given operation should be moved into that region. If this returns true, `move_into_region` is
/// called on the same operation and region.
///
/// `move_into_region` must move the operation into the region such that dominance of the operation
/// is preserved; for example, by moving the operation to the start of the entry block. This ensures
/// the preservation of SSA dominance of the operation's results.
#[derive(Default)]
pub struct ControlFlowSink;

runic_core::inventory::submit!(::runic_core::pass::registry::PassInfo::new::<ControlFlowSink>(
    "control-flow-sink",
    "control flow sinking"
));

impl Pass for ControlFlowSink {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "control-flow-sink"
    }

    fn argument(&self) -> &'static str {
        "control-flow-sink"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let op = op.into_entity_ref();
        log::debug!(target: "control-flow-sink", "sinking operations in {op}");

        let operation = op.as_operation_ref();
        drop(op);

        let dominfo = state.analysis_manager().get_analysis::<DominanceInfo>()?;

        let mut sunk = PostPassStatus::Unchanged;
        operation.raw_prewalk_all::<Forward, _>(|op: OperationRef| {
            let regions_to_sink = {
                let op = op.borrow();
                let Some(branch) = op.as_trait::<dyn RegionBranchOpInterface>() else {
                    return;
                };
                let mut regions = SmallVec::<[_; 4]>::default();
                // Get the regions are that known to be executed at most once.
                get_singly_executed_regions_to_sink(branch, &mut regions);
                regions
            };

            // Sink side-effect free operations.
            sunk = control_flow_sink(
                &regions_to_sink,
                &dominfo,
                |op: &Operation, region: &Region| {
                    op.is_memory_effect_free()
                        && (op.implements::<dyn ConstantLike>()
                            || !would_move_into_a_region_that_may_repeat(
                                op,
                                Some(region.as_region_ref()),
                            ))
                },
                |mut op: OperationRef, region: RegionRef| {
                    // Move the operation to the beginning of the region's entry block.
                    // This guarantees the preservation of SSA dominance of all of the
                    // operation's uses are in the region.
                    let entry_block = region.borrow().entry_block_ref().unwrap();
                    op.borrow_mut().move_to(ProgramPoint::at_start_of(entry_block));
                },
            );
        });

        state.set_post_pass_status(sunk);

        Ok(())
    }
}

/// This transformation moves the definitions of an operation's operands next to their use:
///
/// 1. A constant used by the operation is moved directly before it if this is its only use;
///    otherwise a fresh copy of the constant is materialized there, and the original stays for its
///    other uses.
///
/// 2. An effect-free operation with a single result, whose only use is this operation, is moved
///    directly before it, and the definitions of its own operands are then moved the same way.
///    Such an operation is not moved into a region other than its own unless every region between
///    the two is executed at most once each time the enclosing one is (as [ControlFlowSink]
///    requires of the regions it sinks into): an operation that may trap is never moved to where
///    it would execute more often, so a definition before a loop stays out of the loop's body.
///
/// The definitions are placed in a fixed operand order (see `OpOperandSink::operand_index`): for a
/// binary operation the definition of the right-hand operand ends up closest to the operation and
/// that of the left-hand operand before it; for any other operation the definition of the first
/// operand ends up closest. Values that cannot be moved (block arguments, operations with memory
/// effects, values with other uses) stay where they are.
///
/// An operation left without uses by this rewrite is erased when it is trivially dead.
///
/// The purpose of this rewrite is to shorten the live ranges of values that are cheap to
/// materialize where they are needed.
#[derive(Default)]
pub struct SinkOperandDefs;

impl SinkOperandDefs {
    const NAME: &str = "sink-operand-defs";
}

runic_core::inventory::submit!(::runic_core::pass::registry::PassInfo::new::<SinkOperandDefs>(
    "sink-operand-defs",
    "operand definition sinking"
));

impl Pass for SinkOperandDefs {
    type Target = Operation;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn argument(&self) -> &'static str {
        Self::NAME
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let operation = op.as_operation_ref();
        drop(op);

        log::debug!(target: Self::NAME, "sinking operand defs for regions of {}", operation.borrow());

        // For each operation, we enqueue it in this worklist, we then recurse on each of it's
        // dependency operations until all dependencies have been visited. We move up blocks from
        // the bottom, and skip any operations we've already visited. Once the queue is built, we
        // then process the worklist, moving everything into position.
        let mut worklist = alloc::collections::VecDeque::default();

        let mut changed = PostPassStatus::Unchanged;
        // Visit ops in "true" post-order (i.e. block bodies are visited bottom-up).
        operation.raw_postwalk_all::<Backward, _>(|operation: OperationRef| {
            // Determine if any of this operation's operands represent one of the following:
            //
            // 1. A constant value
            // 2. The sole use of the defining op's single result, and that op has no side-effects
            //
            // If 1, then we either materialize a fresh copy of the constant, or move the original
            // if there are no more uses.
            //
            // In both cases, to the extent possible, we order operand definitions so that their
            // values are produced in the order the using operation consumes them. This means that
            // we visit operands in the order given by `OpOperandSink::operand_index`, and move
            // defining ops directly before `op` when possible. Some values may be block
            // arguments, or be defined by operations we are unable to move, and those stay out
            // of order.
            let op = operation.borrow();

            log::trace!(target: Self::NAME, "visiting {op}");

            for operand in op.operands().iter().rev() {
                let value = operand.borrow();
                let value = value.value();
                // Exclude transparent uses when determining whether this is the sole user.
                let is_sole_user = is_sole_non_transparent_user(&*value, operation);

                let Some(defining_op) = value.get_defining_op() else {
                    // Skip block arguments, nothing to move in that situation
                    //
                    // NOTE: In theory, we could move effect-free operations _up_ the block to place
                    // them closer to the block arguments they use, but that's unlikely to be all
                    // that profitable of a rewrite in practice.
                    log::trace!(target: Self::NAME, "  ignoring block argument operand '{value}'");
                    continue;
                };

                log::trace!(target: Self::NAME, "  evaluating operand '{value}'");

                let def = defining_op.borrow();
                if def.implements::<dyn ConstantLike>() {
                    log::trace!(target: Self::NAME, "    defining '{}' is constant-like", def.name());
                    worklist.push_back(OpOperandSink::new(operation));
                    break;
                }

                let incorrect_result_count = def.num_results() != 1;
                let has_effects = !def.is_memory_effect_free();
                if !is_sole_user || incorrect_result_count || has_effects {
                    // Skip this operand if the defining op cannot be safely moved
                    //
                    // NOTE: For now we do not move ops that produce more than a single result, but
                    // if the other results are unused, or the users would still be dominated by
                    // the new location, then we could still move those ops.
                    log::trace!(target: Self::NAME, "    defining '{}' cannot be moved:", def.name());
                    log::trace!(target: Self::NAME, "      * op has multiple uses");
                    if incorrect_result_count {
                        log::trace!(target: Self::NAME, "      * op has incorrect number of results ({})", def.num_results());
                    }
                    if has_effects {
                        log::trace!(target: Self::NAME, "      * op has memory effects");
                    }
                } else {
                    log::trace!(target: Self::NAME, "    defining '{}' is moveable, but is non-constant", def.name());
                    worklist.push_back(OpOperandSink::new(operation));
                    break;
                }
            }
        });

        for sinker in worklist.iter() {
            log::debug!(target: Self::NAME, "sink scheduled for {}", sinker.operation.borrow());
        }

        let mut visited = FxHashSet::default();
        let mut erased = FxHashSet::default();
        'next_operation: while let Some(mut sink_state) = worklist.pop_front() {
            let mut operation = sink_state.operation;
            let op = operation.borrow();

            // Ignore transparent ops - we do not sink them directly
            if op.implements::<dyn Transparent>() {
                continue;
            }

            // If this operation is unused, remove it now if it has no side effects.
            //
            // NOTE: We explicitly DO NOT remove transparent ops here, unless we're removing the
            // defining op of the transparent operand
            let has_real_uses = op.results().iter().any(|result| result.borrow().has_real_uses());
            if !has_real_uses && op.would_be_trivially_dead() && erased.insert(operation) {
                log::debug!(target: Self::NAME, "erasing unused, effect-free, non-terminator op {op}");
                drop(op);
                // Erase any remaining transparent uses before erasing the defining op.
                //
                // NOTE: The result values are collected first, so that no borrow of the operation
                // or its results is held while the transparent users are erased — erasing a user
                // unlinks its operands, which requires mutable access to the referenced values.
                let results = SmallVec::<[ValueRef; 2]>::from_iter(
                    operation
                        .borrow()
                        .results()
                        .iter()
                        .map(|result| result.borrow().as_value_ref()),
                );
                for value in results {
                    erase_transparent_users(value);
                }
                operation.borrow_mut().erase();
                continue;
            }

            // If we've already worked this operation, skip it
            if !visited.insert(operation) && sink_state.visited_operands == 0 {
                log::trace!(target: Self::NAME, "already visited {}", operation.borrow());
                continue;
            } else {
                log::trace!(target: Self::NAME, "visiting {}", operation.borrow());
            }

            let mut builder = OpBuilder::new(op.context_rc());
            builder.set_insertion_point(sink_state.ip);
            // Each definition is placed directly before `op`, after the ones placed earlier, so
            // the operand visited last is materialized last, immediately before `op`. Operands
            // are visited in the order given by `OpOperandSink::operand_index`.
            'next_operand: while sink_state.visited_operands < op.num_operands() {
                let next_operand_index = sink_state.operand_index(&op);
                log::debug!(target: Self::NAME, "  sinking next operand def for {op} at index {next_operand_index}");

                let mut operand = op.operands()[next_operand_index];
                sink_state.visited_operands += 1;
                let operand_value = operand.borrow().as_value_ref();
                log::trace!(target: Self::NAME, "  visiting operand {operand_value}");

                // Reuse moved/materialized replacements when the same operand is used multiple times
                if let Some(replacement) = sink_state.replacements.get(&operand_value).copied() {
                    if replacement != operand_value {
                        log::trace!(target: Self::NAME, "    rewriting operand {operand_value} as {replacement}");
                        operand.borrow_mut().set(replacement);

                        changed = PostPassStatus::Changed;
                        // If no real uses of this value remain, then remove the original
                        // operation, as it is now dead.
                        if !operand_value.borrow().has_real_uses() {
                            log::trace!(target: Self::NAME, "    {operand_value} is no longer used, erasing definition");
                            erase_transparent_users(operand_value);
                            // Replacements are only ever for op results
                            let mut defining_op = operand_value.borrow().get_defining_op().unwrap();
                            defining_op.borrow_mut().erase();
                        }
                    }
                    continue 'next_operand;
                }

                let value = operand_value.borrow();
                // Exclude transparent uses when determining sole-user status.
                let is_sole_user = is_sole_non_transparent_user(&*value, operation);

                let Some(mut defining_op) = value.get_defining_op() else {
                    // Skip block arguments, nothing to move in that situation
                    //
                    // NOTE: In theory, we could move effect-free operations _up_ the block to place
                    // them closer to the block arguments they use, but that's unlikely to be all
                    // that profitable of a rewrite in practice.
                    log::trace!(target: Self::NAME, "    {value} is a block argument, ignoring..");
                    continue 'next_operand;
                };

                log::trace!(target: Self::NAME, "    is sole user of {value}? {is_sole_user}");

                let def = defining_op.borrow();
                if let Some(attr) = matchers::constant().matches(&*def) {
                    if !is_sole_user {
                        log::trace!(target: Self::NAME, "    defining op is a constant with multiple uses, materializing fresh copy");
                        // Materialize a fresh copy of the original constant
                        let span = value.span();
                        let ty = value.ty();
                        let Some(new_def) =
                            def.dialect().materialize_constant(&mut builder, attr, ty, span)
                        else {
                            log::trace!(target: Self::NAME, "    unable to materialize copy, skipping rewrite of this operand");
                            continue 'next_operand;
                        };
                        drop(def);
                        drop(value);
                        let replacement = new_def.borrow().results()[0] as ValueRef;
                        log::trace!(target: Self::NAME, "    rewriting operand {operand_value} as {replacement}");
                        sink_state.replacements.insert(operand_value, replacement);
                        operand.borrow_mut().set(replacement);
                        changed = PostPassStatus::Changed;
                    } else {
                        log::trace!(target: Self::NAME, "    defining op is a constant with no other uses, moving into place");
                        // The original op can be moved
                        drop(def);
                        drop(value);
                        let mut def_op = defining_op.borrow_mut();
                        def_op.move_to(*builder.insertion_point());
                        sink_state.replacements.insert(operand_value, operand_value);

                        // Move any transparent users of `defining_op` after it
                        move_transparent_users_to(&def_op, &[operation]);
                    }
                } else if !is_sole_user
                    || def.num_results() != 1
                    || !def.is_memory_effect_free()
                    || would_sink_an_exit_between_regions(&def, op.parent_region())
                    || would_move_into_a_region_that_may_repeat(&def, op.parent_region())
                {
                    // Skip this operand if the defining op cannot be safely moved
                    //
                    // NOTE: For now we do not move ops that produce more than a single result, but
                    // if the other results are unused, or the users would still be dominated by
                    // the new location, then we could still move those ops.
                    log::trace!(target: Self::NAME, "    defining op is unsuitable for sinking, ignoring this operand");
                } else {
                    // The original op can be moved: it is not moved into a region that may
                    // execute more often than its own (checked above)
                    drop(def);
                    drop(value);
                    log::trace!(target: Self::NAME, "    defining op can be moved and has no other uses, moving into place");
                    let mut def_op = defining_op.borrow_mut();
                    def_op.move_to(*builder.insertion_point());
                    sink_state.replacements.insert(operand_value, operand_value);

                    // Move any transparent users of `defining_op` after it
                    move_transparent_users_to(&def_op, &[operation]);
                    drop(def_op);

                    // Enqueue the defining op to be visited before continuing with this op's operands
                    log::trace!(target: Self::NAME, "    enqueing defining op for immediate processing");
                    sink_state.ip = ProgramPoint::before(operation);
                    worklist.push_front(sink_state);
                    worklist.push_front(OpOperandSink::new(defining_op));
                    continue 'next_operation;
                }
            }
        }

        state.set_post_pass_status(changed);
        Ok(())
    }
}

fn move_transparent_users_to(op: &Operation, exclude: &[OperationRef]) {
    use runic_core::adt::SmallSet;

    let ip = ProgramPoint::after(op.as_operation_ref());
    let mut visited = SmallSet::<_, 4>::from_iter(exclude.iter().copied());
    for result in op.results().iter() {
        let result = result.borrow();
        for user in result.iter_uses() {
            if !visited.insert(user.owner) {
                continue;
            }
            let mut user = user.owner;
            let mut user = user.borrow_mut();
            if user.implements::<dyn Transparent>() {
                user.move_to(ip);
            }
        }
    }
}

/// The state of sinking the operand definitions of one operation.
struct OpOperandSink {
    operation: OperationRef,
    ip: ProgramPoint,
    replacements: SmallDenseMap<ValueRef, ValueRef, 4>,
    /// The number of operands visited so far; see [`Self::operand_index`].
    visited_operands: usize,
}

impl OpOperandSink {
    pub fn new(operation: OperationRef) -> Self {
        Self {
            operation,
            ip: ProgramPoint::before(operation),
            replacements: SmallDenseMap::new(),
            visited_operands: 0,
        }
    }

    /// Returns the index of the operand to visit next.
    ///
    /// Each definition is moved directly before the operation, after the definitions moved
    /// earlier, so the operand visited last is materialized last, immediately before the
    /// operation. Operands of a binary operation are visited in operand order, so the definition
    /// of the right-hand operand is closest to the operation; the operands of any other operation
    /// are visited in reverse order, so the definition of the first operand is closest.
    fn operand_index(&self, op: &Operation) -> usize {
        if op.implements::<dyn BinaryOp>() {
            self.visited_operands
        } else {
            op.num_operands() - 1 - self.visited_operands
        }
    }
}

/// A helper struct for control-flow sinking.
struct Sinker<'a, P, F> {
    /// Dominance info to determine op user dominance with respect to regions.
    dominfo: &'a DominanceInfo,
    /// The callback to determine whether an op should be moved in to a region.
    should_move_into_region: P,
    /// The calback to move an operation into the region.
    move_into_region: F,
    /// The number of operations sunk
    num_sunk: usize,
}
impl<'a, P, F> Sinker<'a, P, F>
where
    P: Fn(&Operation, &Region) -> bool,
    F: Fn(OperationRef, RegionRef),
{
    /// Create an operation sinker with given dominance info.
    pub fn new(
        dominfo: &'a DominanceInfo,
        should_move_into_region: P,
        move_into_region: F,
    ) -> Self {
        Self {
            dominfo,
            should_move_into_region,
            move_into_region,
            num_sunk: 0,
        }
    }

    /// Given a list of regions, find operations to sink and sink them.
    ///
    /// Returns the number of operations sunk.
    pub fn sink_regions(mut self, regions: &[RegionRef]) -> usize {
        for region in regions.iter().copied() {
            if !region.borrow().is_empty() {
                self.sink_region(region);
            }
        }

        self.num_sunk
    }

    /// Given a region and an op which dominates the region, returns true if all _non-transparent_
    /// users of the given op are dominated by the entry block of the region, and thus the operation
    /// can be sunk into the region.
    ///
    /// Transparent uses are excluded because they are observational and should not prevent
    /// control-flow sinking.
    fn all_users_dominated_by(&self, op: &Operation, region: &Region) -> bool {
        assert!(
            region.find_ancestor_op(op.as_operation_ref()).is_none(),
            "expected op to be defined outside the region"
        );
        let region_entry = region.entry_block_ref().unwrap();
        op.results().iter().all(|result| {
            let result = result.borrow();
            result.iter_uses().all(|user| {
                if user.owner.borrow().implements::<dyn Transparent>() {
                    return true;
                }
                // The user is dominated by the region if its containing block is dominated
                // by the region's entry block.
                self.dominfo.dominates(&region_entry, &user.owner.parent().unwrap())
            })
        })
    }

    /// Given a region and a top-level op (an op whose parent region is the given
    /// region), determine whether the defining ops of the op's operands can be
    /// sunk into the region.
    ///
    /// Add moved ops to the work queue.
    fn try_to_sink_predecessors(
        &mut self,
        user: OperationRef,
        region: RegionRef,
        stack: &mut Vec<OperationRef>,
    ) {
        log::trace!(target: "control-flow-sink", "contained op: {}", user.borrow());
        let user = user.borrow();
        for operand in user.operands().iter() {
            let op = operand.borrow().value().get_defining_op();
            // Ignore block arguments and ops that are already inside the region.
            if op.is_none_or(|op| op.grandparent().is_some_and(|r| r == region)) {
                continue;
            }

            let op = unsafe { op.unwrap_unchecked() };

            log::trace!(target: "control-flow-sink", "try to sink op: {}", op.borrow());

            // If the op's users are all in the region and it can be moved, then do so.
            let (all_users_dominated_by, should_move_into_region) = {
                let op = op.borrow();
                let region = region.borrow();
                let all_users_dominated_by = self.all_users_dominated_by(&op, &region);
                let should_move_into_region = (self.should_move_into_region)(&op, &region);
                (all_users_dominated_by, should_move_into_region)
            };
            if all_users_dominated_by && should_move_into_region {
                (self.move_into_region)(op, region);

                // Move all transparent users of `op` into the region after it
                move_transparent_users_to(&op.borrow(), &[user.as_operation_ref()]);

                self.num_sunk += 1;

                // Add the op to the work queue
                stack.push(op);
            }
        }
    }

    /// Iterate over all the ops in a region and try to sink their predecessors.
    /// Recurse on subgraphs using a work queue.
    fn sink_region(&mut self, region: RegionRef) {
        // Initialize the work queue with all the ops in the region.
        let mut stack = Vec::new();
        for block in region.borrow().body() {
            for op in block.body() {
                stack.push(op.as_operation_ref());
            }
        }

        // Process all the ops depth-first. This ensures that nodes of subgraphs are sunk in the
        // correct order.
        while let Some(op) = stack.pop() {
            self.try_to_sink_predecessors(op, region, &mut stack);
        }
    }
}

pub fn control_flow_sink<P, F>(
    regions: &[RegionRef],
    dominfo: &DominanceInfo,
    should_move_into_region: P,
    move_into_region: F,
) -> PostPassStatus
where
    P: Fn(&Operation, &Region) -> bool,
    F: Fn(OperationRef, RegionRef),
{
    let sinker = Sinker::new(dominfo, should_move_into_region, move_into_region);
    let sunk_regions = sinker.sink_regions(regions);
    (sunk_regions > 0).into()
}

/// Populates `regions` with regions of the provided region branch op that are executed at most once
/// at that are reachable given the current operands of the op. These regions can be passed to
/// `control_flow_sink` to perform sinking on the regions of the operation.
fn get_singly_executed_regions_to_sink(
    branch: &dyn RegionBranchOpInterface,
    regions: &mut SmallVec<[RegionRef; 4]>,
) {
    use runic_core::matchers::Matcher;

    // Collect constant operands.
    let mut operands = SmallVec::<[_; 4]>::with_capacity(branch.num_operands());

    for operand in branch.operands().iter() {
        let matcher = matchers::foldable_operand();
        operands.push(matcher.matches(operand));
    }

    // Get the invocation bounds.
    let bounds = branch.get_region_invocation_bounds(&operands);

    // For a simple control-flow sink, only consider regions that are executed at most once.
    for (region, bound) in branch.regions().iter().zip(bounds) {
        use core::ops::Bound;
        match bound.max() {
            Bound::Unbounded => continue,
            Bound::Excluded(bound) if *bound > 2 => continue,
            Bound::Excluded(0) => continue,
            Bound::Included(bound) if *bound > 1 => continue,
            _ => {
                regions.push(region.as_region_ref());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::ToString};

    use runic_core::{
        Context, Report,
        dialects::builtin::Function,
        parse::{self, ParserConfig},
        pass::{Nesting, PassManager},
    };

    use super::{ControlFlowSink, SinkOperandDefs};

    /// `f` with the body `body`, after `pass` ran on it, as text.
    fn run(
        pass: Box<dyn runic_core::pass::OperationPass>,
        body: &str,
    ) -> Result<alloc::string::String, Report> {
        let context = Rc::new(Context::default());
        let text = alloc::format!(
            "builtin.function public extern(\"C\") @f(%c: i1, %a: i32, %b: i32, %p: ptr<i32>) -> \
             i32 {{\n{body}\n}};"
        );
        let function = parse::parse::<Function>(ParserConfig::new(context.clone()), "sink", text)
            .into_result()?;
        let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
        passes.add_pass(pass).unwrap();
        passes.enable_verifier(true);
        passes.run(function.as_operation_ref())?;
        Ok(function.as_operation_ref().borrow().to_string())
    }

    /// `run`, for a function `@f` (with the same parameters) in a module that also holds
    /// `@pure(i32) -> i32 effects()`, `@unknown(i32) -> i32` and
    /// `@writes(ptr<i32>, i32) -> i32 effects(write(arg 0))`; the text is that of `@f`.
    fn run_with_callees(
        pass: Box<dyn runic_core::pass::OperationPass>,
        body: &str,
    ) -> Result<alloc::string::String, Report> {
        use runic_core::{SymbolTable, dialects::builtin::Module};

        let context = Rc::new(Context::default());
        let text = alloc::format!(
            "builtin.module public @m {{
    builtin.function public extern(\"C\") @pure(i32) -> i32 effects();
    builtin.function public extern(\"C\") @unknown(i32) -> i32;
    builtin.function public extern(\"C\") @writes(ptr<i32>, i32) -> i32 effects(write(arg 0));
    builtin.function public extern(\"C\") @f(%c: i1, %a: i32, %b: i32, %p: ptr<i32>) -> i32 {{
{body}
    }};
}};"
        );
        let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "sink", text)
            .into_result()?;
        let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
        let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
        passes.add_pass(pass).unwrap();
        passes.enable_verifier(true);
        passes.run(function)?;
        Ok(function.borrow().to_string())
    }

    /// A call whose only use is in the then-region of an `scf.if` is moved there by both passes
    /// when its callee declares no effects, and stays where it is when its callee states no
    /// effects or writes.
    #[test]
    fn sinking_moves_a_call_only_when_its_callee_declares_no_effects() -> Result<(), Report> {
        let calls = [
            (r#"builtin.call @pure(%a) : extern("C") (i32) -> i32"#, true),
            (r#"builtin.call @unknown(%a) : extern("C") (i32) -> i32"#, false),
            (r#"builtin.call @writes(%p, %a) : extern("C") (ptr<i32>, i32) -> i32"#, false),
        ];
        for (call, moves) in calls {
            let body = alloc::format!(
                r#"
    %s = {call};
    %x = scf.if %c then {{
        %y = arith.add %b, %s;
        scf.yield %y : (i32);
    }} else {{
        scf.yield %b : (i32);
    }} : (i1) -> (i32);
    builtin.ret %x : (i32);"#
            );
            for pass in [
                Box::<SinkOperandDefs>::default() as Box<dyn runic_core::pass::OperationPass>,
                Box::<ControlFlowSink>::default(),
            ] {
                let name = pass.name();
                let text = run_with_callees(pass, &body)?;
                let moved = text.find("builtin.call").unwrap() > text.find("scf.if").unwrap();
                assert_eq!(moved, moves, "{name}: {text}");
            }
        }
        Ok(())
    }

    /// An indirect call whose only use is in the then-region of an `scf.if` has unknown effects,
    /// and neither pass moves it, even when it calls a function that declares no effects.
    #[test]
    fn sinking_never_moves_an_indirect_call() -> Result<(), Report> {
        let body = r#"
    %f = builtin.function_ref @pure : (i32) -> i32;
    %s = builtin.call_indirect %f(%a) : extern("C") (i32) -> i32;
    %x = scf.if %c then {
        %y = arith.add %b, %s;
        scf.yield %y : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %x : (i32);"#;
        for pass in [
            Box::<SinkOperandDefs>::default() as Box<dyn runic_core::pass::OperationPass>,
            Box::<ControlFlowSink>::default(),
        ] {
            let name = pass.name();
            let text = run_with_callees(pass, body)?;
            let call = text.find("builtin.call_indirect").unwrap();
            assert!(call < text.find("scf.if").unwrap(), "{name}: {text}");
        }
        Ok(())
    }

    /// A division defined before a loop and used once in its body: neither sinking pass moves it
    /// into the body, where it would execute once per iteration (and trap after the store).
    #[test]
    fn operand_sinking_does_not_move_a_division_into_a_loop() -> Result<(), Report> {
        let body = r#"
    %q = arith.udiv %a, %b;
    %zero = arith.constant 0 : i32;
    %count = scf.while %zero before {
    ^head(%i: i32):
        %continue = arith.cmpi ult %i, %a;
        scf.condition %continue, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        test.store %p, %j : (ptr<i32>, i32);
        %next = arith.add %j, %q;
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %count : (i32);"#;
        for pass in [
            Box::<SinkOperandDefs>::default() as Box<dyn runic_core::pass::OperationPass>,
            Box::<ControlFlowSink>::default(),
        ] {
            let text = run(pass, body)?;
            assert!(text.find("arith.udiv").unwrap() < text.find("scf.while").unwrap(), "{text}");
        }
        Ok(())
    }

    /// The body of a function with a division `%q` before a loop whose body holds an `scf.if`
    /// whose then-region is the only user of `%q`.
    fn division_before_a_loop_used_in_an_if_in_its_body() -> &'static str {
        r#"
    %q = arith.udiv %a, %b;
    %zero = arith.constant 0 : i32;
    %count = scf.while %zero before {
    ^head(%i: i32):
        %continue = arith.cmpi ult %i, %a;
        scf.condition %continue, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        test.store %p, %j : (ptr<i32>, i32);
        %next = scf.if %c then {
            %y = arith.add %b, %q;
            scf.yield %y : (i32);
        } else {
            scf.yield %j : (i32);
        } : (i1) -> (i32);
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %count : (i32);"#
    }

    /// A division defined before a loop whose only use is in the then-region of an `scf.if` in
    /// the loop's body: the `if` runs its region at most once, but the loop runs the `if` once per
    /// iteration, so neither sinking pass moves the division into it.
    #[test]
    fn sinking_does_not_move_a_division_into_an_if_inside_a_loop() -> Result<(), Report> {
        for pass in [
            Box::<SinkOperandDefs>::default() as Box<dyn runic_core::pass::OperationPass>,
            Box::<ControlFlowSink>::default(),
        ] {
            let name = pass.name();
            let text = run(pass, division_before_a_loop_used_in_an_if_in_its_body())?;
            let (udiv, start) = (text.find("arith.udiv").unwrap(), text.find("scf.while").unwrap());
            assert!(udiv < start, "{name}: {text}");
        }
        Ok(())
    }

    /// Both sinking passes move effect-free definitions (an addition, and the division that reads
    /// it) into the region of an `scf.if`, which is executed at most once, when that region holds
    /// their only uses.
    #[test]
    fn sinking_moves_a_definition_into_a_region_executed_at_most_once() -> Result<(), Report> {
        let body = r#"
    %s = arith.add %a, %b;
    %q = arith.udiv %s, %b;
    %x = scf.if %c then {
        %y = arith.add %b, %q;
        scf.yield %y : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %x : (i32);"#;
        for pass in [
            Box::<SinkOperandDefs>::default() as Box<dyn runic_core::pass::OperationPass>,
            Box::<ControlFlowSink>::default(),
        ] {
            let name = pass.name();
            let text = run(pass, body)?;
            let start = text.find("scf.if").unwrap();
            assert!(text.find("arith.add").unwrap() > start, "{name}: {text}");
            assert!(text.find("arith.udiv").unwrap() > start, "{name}: {text}");
        }
        Ok(())
    }

    /// A division defined inside a loop's body and used only in the then-region of an `scf.if` in
    /// the same body is moved into that region by both passes: it runs at most once per iteration
    /// there, as it did before.
    #[test]
    fn sinking_moves_a_definition_in_a_loop_body_into_an_if_in_that_body() -> Result<(), Report> {
        let body = r#"
    %zero = arith.constant 0 : i32;
    %count = scf.while %zero before {
    ^head(%i: i32):
        %continue = arith.cmpi ult %i, %a;
        scf.condition %continue, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %q = arith.udiv %a, %j;
        test.store %p, %j : (ptr<i32>, i32);
        %next = scf.if %c then {
            %y = arith.add %b, %q;
            scf.yield %y : (i32);
        } else {
            scf.yield %j : (i32);
        } : (i1) -> (i32);
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %count : (i32);"#;
        for pass in [
            Box::<SinkOperandDefs>::default() as Box<dyn runic_core::pass::OperationPass>,
            Box::<ControlFlowSink>::default(),
        ] {
            let name = pass.name();
            let text = run(pass, body)?;
            let udiv = text.find("arith.udiv").unwrap();
            assert!(udiv > text.find("scf.if").unwrap(), "{name}: {text}");
            assert!(udiv < text.find("} else {").unwrap(), "{name}: {text}");
        }
        Ok(())
    }
}
