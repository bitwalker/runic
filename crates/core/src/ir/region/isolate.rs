use alloc::collections::VecDeque;

use smallvec::SmallVec;

use crate::{
    FxHashMap, FxHashSet, InsertionGuard, NamedAttribute, OpOperandImpl, Operation, OperationRef,
    OperationState, Region, RegionRef, Rewriter, RewriterExt, SourceSpan, Spanned, Symbol, Type,
    Value, ValueRef, WalkResult, dialects::builtin::attributes::SymbolRefAttr,
};

impl Region {
    /// Make `region` isolated from above: afterwards no operation in it, or in a region nested in
    /// it, uses a value defined above it.
    ///
    /// * The values defined above `region` and used in it are captured.
    /// * `clone_into_region` is called with the defining operation of a captured value. If it
    ///   returns true, the operation is cloned into the start of the entry block and its clone's
    ///   results replace the uses of its results in `region`; its operands are then captured in
    ///   turn, unless their own defining operations are cloned too. An operation that holds
    ///   regions or successors, more than one group of results, or a symbol reference, or that
    ///   is a symbol, is never cloned, whatever `clone_into_region` says: its results are
    ///   captured instead. Nor is an operation of a cycle of operations to clone that read each
    ///   other (possible in a graph region above), where it cannot be placed after what it reads.
    /// * The entry block gets one argument for each remaining captured value, after the arguments
    ///   it has, and the uses of the value in `region` are replaced with that argument.
    ///
    /// Returns the captured values that became entry block arguments, in the order of the new
    /// arguments. The caller passes them to `region` the way its operation expects.
    ///
    /// Corresponds to upstream `makeRegionIsolatedFromAbove`.
    pub fn make_isolated_from_above<R, F>(
        region: RegionRef,
        rewriter: &mut R,
        mut clone_into_region: F,
    ) -> SmallVec<[ValueRef; 1]>
    where
        R: ?Sized + Rewriter,
        F: FnMut(&Operation) -> bool,
    {
        let Some(entry) = region.borrow().entry_block_ref() else {
            // An empty region uses nothing.
            return SmallVec::new();
        };

        // The values used in the region but defined above it.
        let initial = Self::get_used_values_defined_above_any(&[region]);
        let mut worklist = VecDeque::from_iter(initial);
        let mut visited = FxHashSet::<ValueRef>::default();
        let mut visited_ops = FxHashSet::<OperationRef>::default();

        let mut captured = SmallVec::<[ValueRef; 1]>::default();
        let mut captured_set = FxHashSet::<ValueRef>::default();
        let mut capture = |value: ValueRef| {
            if captured_set.insert(value) {
                captured.push(value);
            }
        };
        let mut cloned = SmallVec::<[OperationRef; 4]>::default();
        while let Some(value) = worklist.pop_front() {
            if !visited.insert(value) {
                continue;
            }

            let defining_op = value.borrow().get_defining_op();
            let Some(defining_op) = defining_op.filter(|op| !visited_ops.contains(op)) else {
                capture(value);
                continue;
            };
            visited_ops.insert(defining_op);

            let clone = {
                let op = defining_op.borrow();
                can_clone(&op) && clone_into_region(&op)
            };
            if !clone {
                // The defining operation stays where it is, so the value itself is captured.
                capture(value);
                continue;
            }

            // The operands of the cloned operation are captured in its place.
            for operand in defining_op.borrow().operands().iter() {
                let operand = operand.borrow().as_value_ref();
                if !visited.contains(&operand) {
                    worklist.push_back(operand);
                }
            }
            cloned.push(defining_op);
        }

        // A clone is inserted after the clones of the operations defining its operands. Operations
        // that read each other in a cycle (in a graph region above) cannot all be ordered so: the
        // ones that could not be are not cloned, and their results are captured instead.
        let unordered = compute_topological_sorting(&mut cloned);
        cloned.retain(|op| !unordered.contains(op));
        for op in unordered {
            for result in op.borrow().results().iter() {
                let result = *result as ValueRef;
                let used = result.borrow().iter_uses().any(|user| {
                    Self::contains_user(&region, &user) || cloned.contains(&user.owner)
                });
                if used && captured_set.insert(result) {
                    captured.push(result);
                }
            }
        }

        let mut guard = InsertionGuard::new(rewriter);
        let rewriter = &mut *guard;

        // The new entry block takes the arguments of the current one, then one per captured value.
        let (mut arg_types, mut arg_spans) = {
            let entry = entry.borrow();
            let types = entry.argument_types().collect::<SmallVec<[Type; 4]>>();
            let spans = entry
                .arguments()
                .iter()
                .map(|arg| arg.borrow().span())
                .collect::<SmallVec<[SourceSpan; 4]>>();
            (types, spans)
        };
        let num_old_arguments = arg_types.len();
        for value in captured.iter() {
            let value = value.borrow();
            arg_types.push(value.ty());
            arg_spans.push(value.span());
        }
        let new_entry = rewriter.create_block_before(entry, &arg_types);
        let new_arguments = new_entry
            .borrow()
            .arguments()
            .iter()
            .zip(arg_spans)
            .map(|(arg, span)| {
                let mut arg = *arg;
                arg.borrow_mut().set_span(span);
                arg as ValueRef
            })
            .collect::<SmallVec<[ValueRef; 4]>>();

        let in_region = |operand: &OpOperandImpl| Self::contains_user(&region, operand);

        // Each captured value is replaced in the region with its argument.
        let mut mapping = FxHashMap::<ValueRef, ValueRef>::default();
        for (captured_value, arg) in captured.iter().zip(&new_arguments[num_old_arguments..]) {
            mapping.insert(*captured_value, *arg);
            rewriter.maybe_replace_uses_of_value_with(*captured_value, *arg, in_region);
        }

        // Each cloned operation is cloned into the new entry block, and its results are replaced in
        // the region with the clone's.
        rewriter.set_insertion_point_to_end(new_entry);
        for op in cloned {
            let clone = clone_operation(rewriter, &op.borrow(), &mapping);
            let results = clone
                .borrow()
                .results()
                .iter()
                .map(|result| *result as ValueRef)
                .collect::<SmallVec<[ValueRef; 2]>>();
            for (result, clone_result) in op.borrow().results().iter().zip(results.iter()) {
                mapping.insert(*result as ValueRef, *clone_result);
            }
            rewriter.maybe_replace_op_uses_with(op, &results, in_region);
        }

        // The operations of the old entry block follow the clones, using the new block's leading
        // arguments in place of its own.
        let old_arguments = new_arguments[..num_old_arguments]
            .iter()
            .map(|arg| Some(*arg))
            .collect::<SmallVec<[Option<ValueRef>; 4]>>();
        rewriter.merge_blocks(entry, new_entry, &old_arguments);

        captured
    }
}

/// True if [clone_operation] reproduces `op` exactly: it has no regions, no successors, at most
/// one group of results, is no symbol (a copy would be a second definition of its name), and
/// holds no symbol reference (whose use the clone would have to link to its symbol).
fn can_clone(op: &Operation) -> bool {
    !op.has_regions()
        && !op.implements::<dyn Symbol>()
        && !op.has_successors()
        && op.results().num_groups() <= 1
        && !op
            .walk_attributes(&mut |_, attr| {
                if attr.try_downcast_attr::<SymbolRefAttr>().is_ok() {
                    WalkResult::Break(crate::Report::msg("a symbol reference"))
                } else {
                    WalkResult::Continue(())
                }
            })
            .was_interrupted()
}

/// Build a copy of `op`, which [can_clone], at the insertion point of `rewriter`: the same name,
/// span, properties, attributes (each a copy) and result types, with each operand replaced by its
/// value in `mapping`, if it has one.
fn clone_operation<R>(
    rewriter: &mut R,
    op: &Operation,
    mapping: &FxHashMap<ValueRef, ValueRef>,
) -> OperationRef
where
    R: ?Sized + Rewriter,
{
    let mut state = OperationState::new(op.span(), op.name());
    for property in op.properties() {
        state
            .attrs
            .push(NamedAttribute::new(property.name, property.value.borrow().dyn_clone()));
    }
    for attribute in op.attributes().iter() {
        let attribute = attribute.as_named_attribute();
        state
            .attrs
            .push(NamedAttribute::new(attribute.name, attribute.value.borrow().dyn_clone()));
    }
    for group in op.operands().groups() {
        let operands = group
            .iter()
            .map(|operand| {
                let value = operand.borrow().as_value_ref();
                mapping.get(&value).copied().unwrap_or(value)
            })
            .collect();
        state.add_operands(operands);
    }
    state.results.extend(op.results().iter().map(|result| result.borrow().ty()));
    rewriter.create_operation(&mut state).expect(
        "a copy of a valid operation that holds no regions, successors or symbol references, with \
         operands of the same types, passes the checks the builder makes",
    )
}

/// Order `ops` so that each operation comes after the operations among them that define its
/// operands. Where the operations form a cycle, the first one not placed yet is placed next,
/// although it reads an operation placed after it.
///
/// Returns the operations placed that way, which are not ordered after what they read; it is
/// empty if every operation is.
///
/// Corresponds to upstream `computeTopologicalSorting`, which returns whether that list is empty.
fn compute_topological_sorting(ops: &mut [OperationRef]) -> SmallVec<[OperationRef; 2]> {
    let mut unordered = SmallVec::new();
    let mut unscheduled = FxHashSet::from_iter(ops.iter().copied());
    let mut next_scheduled = 0;
    while !unscheduled.is_empty() {
        let mut scheduled_one = false;
        // The operations not scheduled when this pass over them starts.
        let first_unscheduled = next_scheduled;
        for index in first_unscheduled..ops.len() {
            if !is_op_ready(ops[index], &unscheduled) {
                continue;
            }
            unscheduled.remove(&ops[index]);
            ops.swap(index, next_scheduled);
            scheduled_one = true;
            next_scheduled += 1;
        }

        // A cycle: schedule the first operation not scheduled yet, and carry on.
        if !scheduled_one {
            unscheduled.remove(&ops[next_scheduled]);
            unordered.push(ops[next_scheduled]);
            next_scheduled += 1;
        }
    }
    unordered
}

/// True if no operand of `op` is defined by an operation of `unscheduled`, or by an operation
/// nested in one.
fn is_op_ready(op: OperationRef, unscheduled: &FxHashSet<OperationRef>) -> bool {
    op.borrow().operands().iter().all(|operand| {
        let mut parent = operand.borrow().value().get_defining_op();
        while let Some(defining_op) = parent {
            if defining_op == op {
                return true;
            }
            if unscheduled.contains(&defining_op) {
                return false;
            }
            parent = defining_op.borrow().parent_op();
        }
        true
    })
}
