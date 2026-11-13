use alloc::{rc::Rc, vec::Vec};

use runic_core::{
    AsValueRange, BlockRef, EntityMut, FxHashMap, FxHashSet, Operation, OperationName,
    OperationRef, RegionRef, Report, Rewriter, RewriterExt, SmallVec, ValueRef,
    adt::SmallDenseMap,
    cfg::Graph,
    dominance::{DomTreeNode, DominanceInfo, PostDominanceInfo},
    effects::MemoryEffect,
    pass::{Pass, PassExecutionState, PostPassStatus},
    patterns::{RewriterImpl, RewriterListener, TracingRewriterListener},
    traits::{IsolatedFromAbove, Terminator, Transparent},
};

/// This transformation pass performs a simple common sub-expression elimination algorithm on
/// operations within a region.
#[derive(Default)]
pub struct CommonSubexpressionElimination;

runic_core::inventory::submit!(::runic_core::pass::registry::PassInfo::new::<
    CommonSubexpressionElimination,
>("cse", "common subexpression elimintation"));

impl Pass for CommonSubexpressionElimination {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "cse"
    }

    fn argument(&self) -> &'static str {
        "cse"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        // Run sparse constant propagation + dead code analysis
        let op = op.into_entity_ref();

        // Rewrite based on results of analysis
        let context = op.context_rc();
        let op = {
            let op_ref = op.as_operation_ref();
            drop(op);
            op_ref
        };

        let dominfo = state.analysis_manager().get_analysis::<DominanceInfo>()?;
        let mut rewriter = RewriterImpl::<TracingRewriterListener>::new(context)
            .with_listener(TracingRewriterListener);

        let mut driver = CSEDriver {
            rewriter: &mut rewriter,
            domtree: dominfo,
            ops_to_erase: Default::default(),
            mem_effects_cache: Default::default(),
        };

        let status = driver.simplify(op);

        if status.ir_changed() {
            // CSE only replaces uses and erases operations; it does not change the CFG of
            // remaining regions, so dominance/post-dominance for remaining blocks is preserved.
            state.preserved_analyses_mut().preserve::<DominanceInfo>();
            state.preserved_analyses_mut().preserve::<PostDominanceInfo>();
        } else {
            // If there was no change to the IR, we mark all analyses as preserved.
            state.preserved_analyses_mut().preserve_all();
        }

        Ok(())
    }
}

/// Simple common sub-expression elimination.
struct CSEDriver<'a> {
    rewriter: &'a mut RewriterImpl<TracingRewriterListener>,
    /// Operations marked as dead and to be erased.
    ops_to_erase: Vec<OperationRef>,
    /// Dominance info provided by the current analysis state
    domtree: Rc<DominanceInfo>,
    /// Cache holding MemoryEffect information between two operations.
    ///
    /// The first operation is the key, the second operation is paired with whatever effect exists
    /// between the two operations. If the effect is None, then we assume there is no operation with
    /// `MemoryEffect::Write` between the two operations.
    mem_effects_cache: SmallDenseMap<OperationRef, (OperationRef, Option<MemoryEffect>)>,
}

/// Tracks common subexpression candidates visible in the current dominance scope.
#[derive(Clone, Default)]
struct ScopedCseCandidates {
    /// Hash index used for O(1) lookups when operation keys are stable.
    index: FxHashMap<OpKey, OperationRef>,
    /// Traversal-order candidates used for graph-region fallback lookups.
    ///
    /// This is populated for SSA regions too, so a nested graph region can see candidates from
    /// its enclosing SSA scopes without depending on mutable `OpKey` hashes.
    ops: Vec<OperationRef>,
}

impl ScopedCseCandidates {
    /// Looks up an equivalent candidate for `op`.
    fn get(&self, op: OperationRef, has_ssa_dominance: bool) -> Option<OperationRef> {
        if has_ssa_dominance {
            self.index.get(&OpKey(op)).copied()
        } else {
            // Graph regions permit use-before-def, so candidate keys may only be compared against
            // current IR state. Avoid the hash index because OpKey is derived from mutable operands.
            self.ops.iter().copied().find(|existing| OpKey(*existing) == OpKey(op))
        }
    }

    /// Returns equivalent candidates for memory-read CSE.
    ///
    /// Graph regions scan candidates in reverse visitation order. SSA regions use the hash index
    /// and return the current canonical candidate, if one exists.
    ///
    /// Graph memory-read CSE is intentionally disabled today, but the graph branch keeps this
    /// helper correct if that conservative restriction is relaxed later.
    fn equivalent_ops_rev(
        &self,
        op: OperationRef,
        has_ssa_dominance: bool,
    ) -> SmallVec<[OperationRef; 2]> {
        if has_ssa_dominance {
            self.get(op, has_ssa_dominance).into_iter().collect()
        } else {
            self.ops
                .iter()
                .rev()
                .copied()
                .filter(|existing| OpKey(*existing) == OpKey(op))
                .collect()
        }
    }

    /// Inserts `op` into the current scope.
    fn insert(&mut self, op: OperationRef, has_ssa_dominance: bool) {
        if has_ssa_dominance {
            self.index.insert(OpKey(op), op);
        }
        self.ops.push(op);
    }
}

impl CSEDriver<'_> {
    pub fn simplify(&mut self, op: OperationRef) -> PostPassStatus {
        // Simplify all regions.
        let mut known_values = ScopedCseCandidates::default();
        let mut next_region = op.borrow().regions().front().as_pointer();
        let mut status = PostPassStatus::Unchanged;
        while let Some(region) = next_region.take() {
            next_region = region.next();

            status |= self.simplify_region(&mut known_values, region);
        }

        // Erase any operations that were marked as dead during simplification.
        status |= PostPassStatus::from(!self.ops_to_erase.is_empty());
        for op in self.ops_to_erase.drain(..) {
            self.rewriter.erase_op(op);
        }

        status
    }

    /// Attempt to eliminate a redundant operation.
    ///
    /// Returns [PostPassStatus::Changed] when the operation was marked for removal or at least one
    /// use was replaced, and [PostPassStatus::Unchanged] otherwise.
    fn simplify_operation(
        &mut self,
        known_values: &mut ScopedCseCandidates,
        visited_ops: &FxHashSet<OperationRef>,
        op: OperationRef,
        has_ssa_dominance: bool,
    ) -> PostPassStatus {
        // Don't simplify terminator operations.
        let operation = op.borrow();
        if operation.implements::<dyn Terminator>() {
            return PostPassStatus::Unchanged;
        }

        if operation.implements::<dyn Transparent>() {
            return PostPassStatus::Unchanged;
        }

        // If the operation is already trivially dead just add it to the erase list.
        if operation.is_trivially_dead() {
            self.ops_to_erase.push(op);
            return PostPassStatus::Changed;
        }

        // Don't simplify operations with regions that have multiple blocks.
        // TODO: We need additional tests to verify that we handle such IR correctly.
        if !operation.regions().iter().all(|r| r.is_empty() || r.has_one_block()) {
            return PostPassStatus::Unchanged;
        }

        // Some simple use case of operation with memory side-effect are dealt with here.
        // Operations with no side-effect are done after.
        if !operation.is_memory_effect_free() {
            if !has_ssa_dominance {
                return PostPassStatus::Unchanged;
            }

            // TODO: Only basic use case for operations with MemoryEffects::Read can be
            // eleminated now. More work needs to be done for more complicated patterns
            // and other side-effects.
            if !operation.has_single_memory_effect(MemoryEffect::Read) {
                return PostPassStatus::Unchanged;
            }

            // Look for an existing definition for the operation.
            let candidates = known_values.equivalent_ops_rev(op, has_ssa_dominance);
            if let Some(existing) = candidates.into_iter().find(|existing| {
                existing.parent() == op.parent()
                    && !self.has_other_side_effecting_op_in_between(*existing, op)
            }) {
                // The operation that can be deleted has been reach with no
                // side-effecting operations in between the existing operation and
                // this one so we can remove the duplicate.
                return self.replace_uses_and_delete(visited_ops, op, existing, has_ssa_dominance);
            }
            known_values.insert(op, has_ssa_dominance);
            return PostPassStatus::Unchanged;
        }

        // Look for an existing definition for the operation.
        if let Some(existing) = known_values.get(op, has_ssa_dominance) {
            self.replace_uses_and_delete(visited_ops, op, existing, has_ssa_dominance)
        } else {
            // Otherwise, we add this operation to the known values map.
            known_values.insert(op, has_ssa_dominance);
            PostPassStatus::Unchanged
        }
    }

    fn simplify_block(
        &mut self,
        known_values: &mut ScopedCseCandidates,
        block: BlockRef,
        has_ssa_dominance: bool,
    ) -> PostPassStatus {
        let mut changed = PostPassStatus::Unchanged;
        let mut visited_ops = FxHashSet::default();
        let mut next_op = block.borrow().body().front().as_pointer();
        while let Some(op) = next_op.take() {
            next_op = op.next();

            // Most operations don't have regions, so fast path that case.
            let operation = op.borrow();
            if operation.has_regions() {
                // The graph-region visit guard below is block-local. Nested regions are simplified
                // as separate scopes, relying on nested values not escaping upward and parent op
                // results not being used inside their own nested regions.
                // If this operation is isolated above, we can't process nested regions with the
                // given 'known_values' map. This would cause the insertion of implicit captures in
                // explicit capture only regions.
                if operation.implements::<dyn IsolatedFromAbove>() {
                    let mut nested_known_values = ScopedCseCandidates::default();
                    let mut next_region = operation.regions().front().as_pointer();
                    while let Some(region) = next_region.take() {
                        next_region = region.next();

                        changed |= self.simplify_region(&mut nested_known_values, region);
                    }
                } else {
                    // Otherwise, process nested regions normally.
                    let mut next_region = operation.regions().front().as_pointer();
                    while let Some(region) = next_region.take() {
                        next_region = region.next();

                        changed |= self.simplify_region(known_values, region);
                    }
                }
            }

            changed |= self.simplify_operation(known_values, &visited_ops, op, has_ssa_dominance);
            if !has_ssa_dominance {
                visited_ops.insert(op);
            }
        }

        // Clear the MemoryEffect cache since its usage is by block only.
        self.mem_effects_cache.clear();

        changed
    }

    fn simplify_region(
        &mut self,
        known_values: &mut ScopedCseCandidates,
        region: RegionRef,
    ) -> PostPassStatus {
        // If the region is empty there is nothing to do.
        let region = region.borrow();
        if region.is_empty() {
            return PostPassStatus::Unchanged;
        }

        let has_ssa_dominance = self.domtree.has_ssa_dominance(region.as_region_ref());

        // If the region only contains one block, then simplify it directly.
        if region.has_one_block() {
            let mut scope = known_values.clone();
            let block = region.entry_block_ref().unwrap();
            drop(region);
            return self.simplify_block(&mut scope, block, has_ssa_dominance);
        }

        // If the region does not have dominanceInfo, then skip it.
        // TODO: Regions without SSA dominance should define a different traversal order which is
        // appropriate and can be used here.
        if !has_ssa_dominance {
            return PostPassStatus::Unchanged;
        }

        // Process the nodes of the dom tree for this region.
        let mut stack = Vec::<CfgStackNode>::with_capacity(16);
        let dominfo = self.domtree.dominance(region.as_region_ref());
        stack.push(CfgStackNode::new(known_values.clone(), dominfo.root_node().unwrap()));

        let mut changed = PostPassStatus::Unchanged;
        while let Some(current_node) = stack.last_mut() {
            // Check to see if we need to process this node.
            if !current_node.processed {
                current_node.processed = true;
                changed |= self.simplify_block(
                    &mut current_node.scope,
                    current_node.node.block().unwrap(),
                    has_ssa_dominance,
                );
            }

            // Otherwise, check to see if we need to process a child node.
            if let Some(next_child) = current_node.children.next() {
                let scope = current_node.scope.clone();
                stack.push(CfgStackNode::new(scope, next_child));
            } else {
                // Finally, if the node and all of its children have been processed then we delete
                // the node.
                stack.pop();
            }
        }
        changed
    }

    fn replace_uses_and_delete(
        &mut self,
        visited_ops: &FxHashSet<OperationRef>,
        op: OperationRef,
        mut existing: OperationRef,
        has_ssa_dominance: bool,
    ) -> PostPassStatus {
        // If we find one then replace all uses of the current operation with the existing one and
        // mark it for deletion. We can only replace an operand in an operation if it has not been
        // visited yet.
        if has_ssa_dominance {
            // If the region has SSA dominance, then we are guaranteed to have not visited any use
            // of the current operation.
            self.rewriter.notify_operation_replaced(op, existing);
            // Replace all uses, but do not remove the operation yet. This does not notify the
            // listener because the original op is not erased.
            let operation = op.borrow();
            let existing = existing.borrow();
            let op_results = operation.results().as_value_range().into_smallvec();
            let existing_results = existing
                .results()
                .iter()
                .copied()
                .map(|result| Some(result as ValueRef))
                .collect::<SmallVec<[_; 2]>>();
            self.rewriter.replace_all_uses_with(&op_results, &existing_results);
            self.ops_to_erase.push(op);
        } else {
            // When the region does not have SSA dominance, we need to check if we have visited a
            // use before replacing any use.
            let can_replace_use = |operand: &runic_core::OpOperandImpl| {
                !Self::has_visited_owner_or_ancestor(visited_ops, operand.owner)
            };

            let op_results = op.borrow().results().as_value_range().into_smallvec();
            let has_replaceable_use = op_results.iter().any(|v| {
                let v = v.borrow();
                v.iter_uses().any(|user| can_replace_use(&user))
            });
            if !has_replaceable_use {
                return PostPassStatus::Unchanged;
            }

            let should_replace_op = op_results.iter().all(|v| {
                let v = v.borrow();
                v.iter_uses().all(|user| can_replace_use(&user))
            });
            if should_replace_op {
                self.rewriter.notify_operation_replaced(op, existing);
            }

            // Replace all uses, but do not remove the operation yet. This does not notify the
            // listener because the original op is not erased.
            let existing_results = existing.borrow().results().as_value_range().into_smallvec();
            self.rewriter
                .maybe_replace_uses_with(&op_results, &existing_results, can_replace_use);

            // There may be some remaining uses of the operation.
            if !op.borrow().is_used() {
                self.ops_to_erase.push(op);
            }
        }

        runic_core::dialects::debuginfo::transform::merge_inline_call_locations(existing, op);

        // If the existing operation has an unknown location and the current operation doesn't,
        // then set the existing op's location to that of the current op.
        let mut existing = existing.borrow_mut();
        let op_span = op.borrow().span;
        if existing.span.is_unknown() && !op_span.is_unknown() {
            existing.set_span(op_span);
        }

        PostPassStatus::Changed
    }

    /// Returns true if `owner` or one of its enclosing operations was already visited.
    fn has_visited_owner_or_ancestor(
        visited_ops: &FxHashSet<OperationRef>,
        owner: OperationRef,
    ) -> bool {
        let mut current = Some(owner);
        while let Some(op) = current.take() {
            if visited_ops.contains(&op) {
                return true;
            }
            current = op.borrow().parent_op();
        }
        false
    }

    /// Check if there is side-effecting operations other than the given effect between the two
    /// operations.
    fn has_other_side_effecting_op_in_between(
        &mut self,
        from: OperationRef,
        to: OperationRef,
    ) -> bool {
        assert_eq!(from.parent(), to.parent(), "expected operations to be in the same block");
        let from_op = from.borrow();
        assert!(
            from_op.has_memory_effect(MemoryEffect::Read),
            "expected read effect on `from` op"
        );
        assert!(
            to.borrow().has_memory_effect(MemoryEffect::Read),
            "expected read effect on `to` op"
        );

        let result = self.mem_effects_cache.entry(from).or_insert((from, None));
        let mut next_op = if result.1.is_none() {
            // No `MemoryEffect::Write` has been detected until the cached operation, continue
            // looking from the cached operation to `to`.
            Some(result.0)
        } else {
            // MemoryEffects::Write has been detected before so there is no need to check
            // further.
            return true;
        };

        while let Some(next) = next_op.take()
            && next != to
        {
            next_op = next.next();

            let effects = next.borrow().get_effects_recursively::<MemoryEffect>();
            if let Some(effects) = effects.as_deref() {
                for effect in effects {
                    if effect.effect() == MemoryEffect::Write {
                        *result = (next, Some(MemoryEffect::Write));
                        return true;
                    }
                }
            } else {
                // TODO: Do we need to handle other effects generically?
                // If the operation does not implement the MemoryEffectOpInterface we conservatively
                // assume it writes.
                *result = (next, Some(MemoryEffect::Write));
                return true;
            }
        }

        *result = (to, None);
        false
    }
}

/// Represents a single entry in the depth first traversal of a CFG.
struct CfgStackNode {
    /// Scope for the known values.
    scope: ScopedCseCandidates,
    node: Rc<DomTreeNode>,
    children: <Rc<DomTreeNode> as Graph>::ChildIter,
    /// If this node has been fully processed yet or not.
    processed: bool,
}

impl CfgStackNode {
    pub fn new(scope: ScopedCseCandidates, node: Rc<DomTreeNode>) -> Self {
        let children = <Rc<DomTreeNode> as Graph>::children(node.clone());
        Self {
            scope,
            node,
            children,
            processed: false,
        }
    }
}

/// A wrapper type for [OperationRef] which hashes/compares using operation equivalence flags that
/// ignore locations and result values, considering only operands and properties of the operation
/// itself
#[derive(Copy, Clone)]
struct OpKey(OperationRef);

impl core::hash::Hash for OpKey {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        use runic_core::equivalence::{
            DefaultValueHasher, IgnoreValueHasher, OperationEquivalenceFlags,
        };
        self.0.borrow().hash_with_options(
            OperationEquivalenceFlags::IGNORE_LOCATIONS,
            DefaultValueHasher,
            IgnoreValueHasher,
            state,
        );
    }
}

impl Eq for OpKey {}

impl PartialEq for OpKey {
    fn eq(&self, other: &Self) -> bool {
        use runic_core::equivalence::{DefaultValueEquivalence, OperationEquivalenceFlags};

        if self.0 == other.0 {
            return true;
        }
        let lhs = self.0.borrow();
        let rhs = other.0.borrow();

        // Operands are compared by identity, mirroring the `DefaultValueHasher` used by `Hash`.
        lhs.is_equivalent_with_options(
            &rhs,
            OperationEquivalenceFlags::IGNORE_LOCATIONS,
            DefaultValueEquivalence,
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, format, string::ToString};

    use litcheck_filecheck::{filecheck, litcheck};
    use runic_core::{
        Builder, SourceSpan,
        dialects::builtin::{BuiltinOpBuilder, Module},
        pass::{Nesting, PassManager},
        print::AsmPrinter,
        testing::Test,
    };
    use runic_dialect_arith::ArithOpBuilder;
    use runic_dialect_cf::ControlFlowOpBuilder;

    use super::*;

    pub(super) fn inline_chain(
        context: &Rc<runic_core::Context>,
        call_line: u32,
    ) -> runic_core::AttributeRef {
        use runic_core::dialects::debuginfo::attributes::{
            InlineCallChain, InlineCallChainAttr, InlineCallFrame,
        };

        context
            .create_attribute::<InlineCallChainAttr, _>(InlineCallChain::new(alloc::vec![
                InlineCallFrame {
                    name: "callee".into(),
                    linkage_name: None,
                    file: "callee.rs".into(),
                    line: 1,
                    column: 1,
                    call_file: "caller.rs".into(),
                    call_line,
                    call_column: 1,
                }
            ]))
            .as_attribute_ref()
    }

    #[test]
    fn operation_equivalence_ignores_only_inline_locations_when_requested() {
        use core::hash::Hash;

        use runic_core::{
            dialects::debuginfo::attributes::INLINE_CALL_CHAIN_ATTR_NAME,
            equivalence::{IgnoreValueEquivalence, OperationEquivalenceFlags},
        };

        let mut test = Test::new("inline_equivalence", |cx| ([], [cx.i32(), cx.i32()]));
        let (mut first, mut second) = {
            let mut builder = test.function_builder();
            let first = builder.i32(1, SourceSpan::UNKNOWN);
            let second = builder.i32(1, SourceSpan::UNKNOWN);
            builder.ret([first, second], SourceSpan::UNKNOWN).unwrap();
            (
                first.borrow().get_defining_op().unwrap(),
                second.borrow().get_defining_op().unwrap(),
            )
        };
        let first_chain = inline_chain(&test.context_rc(), 10);
        let second_chain = inline_chain(&test.context_rc(), 20);
        first.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, first_chain);
        second.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, second_chain);

        let hash = |op| {
            let mut hasher = runic_core::FxHasher::default();
            OpKey(op).hash(&mut hasher);
            core::hash::Hasher::finish(&hasher)
        };
        assert!(OpKey(first) == OpKey(second));
        assert_eq!(hash(first), hash(second));
        assert!(!first.borrow().is_equivalent_with_options(
            &second.borrow(),
            OperationEquivalenceFlags::NONE,
            IgnoreValueEquivalence,
        ));

        second.borrow_mut().remove_attribute(INLINE_CALL_CHAIN_ATTR_NAME);
        assert!(OpKey(first) == OpKey(second));
        assert_eq!(hash(first), hash(second));

        first.borrow_mut().set_attribute("semantic", first_chain);
        second.borrow_mut().set_attribute("semantic", second_chain);
        assert!(OpKey(first) != OpKey(second));
        assert_ne!(hash(first), hash(second));
    }

    #[test]
    fn cse_merges_inline_sites_and_retains_only_unambiguous_chains() {
        use runic_core::dialects::debuginfo::attributes::INLINE_CALL_CHAIN_ATTR_NAME;

        for second_line in [None, Some(10), Some(20)] {
            let mut test = Test::new("inline_constants", |cx| ([], [cx.i32(), cx.i32()]));
            let (mut first, mut second) = {
                let mut builder = test.function_builder();
                let first = builder.i32(1, SourceSpan::UNKNOWN);
                let second = builder.i32(1, SourceSpan::UNKNOWN);
                builder.ret([first, second], SourceSpan::UNKNOWN).unwrap();
                (
                    first.borrow().get_defining_op().unwrap(),
                    second.borrow().get_defining_op().unwrap(),
                )
            };
            let chain = inline_chain(&test.context_rc(), 10);
            first.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, chain);
            if let Some(line) = second_line {
                let chain = inline_chain(&test.context_rc(), line);
                second.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, chain);
            }
            test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");
            let entry = test.entry_block();
            let entry = entry.borrow();
            assert_eq!(entry.body().len(), 2);
            let ret = entry.terminator().unwrap();
            let ret = ret.borrow();
            let operands = ret.operands().all();
            assert_eq!(operands[0].borrow().as_value_ref(), operands[1].borrow().as_value_ref());
            assert_eq!(
                first.borrow().get_attribute(INLINE_CALL_CHAIN_ATTR_NAME),
                (second_line == Some(10)).then_some(chain),
            );
        }
    }

    #[test]
    fn simple_constant() {
        let mut test = Test::new("simple_constant", |cx| ([], [cx.i32(), cx.i32()]));
        {
            let mut builder = test.function_builder();
            let v0 = builder.i32(1, SourceSpan::UNKNOWN);
            let v1 = builder.i32(1, SourceSpan::UNKNOWN);
            builder.ret([v0, v1], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @simple_constant() -> (i32, i32) {
    // CHECK: [[V0:%\d+]] = arith.constant 1 : i32;
    %0 = arith.constant 1 : i32;

    // CHECK-NEXT: builtin.ret [[V0]], [[V0]] : (i32, i32);
    %1 = arith.constant 1 : i32;
    builtin.ret %0, %1 : (i32, i32);
};
            "#
        );
    }

    #[test]
    fn cse_eliminates_redundant_memory_reads_in_ssa_region() {
        use runic_core::dialects::test::TestOpBuilder;

        let mut test =
            Test::new("ssa_memory_reads", |cx| ([cx.ptr_type(cx.i8())], [cx.i8(), cx.i8()]));
        {
            let mut builder = test.function_builder();
            let ptr = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let first = builder.load(ptr, SourceSpan::UNKNOWN).unwrap();
            let second = builder.load(ptr, SourceSpan::UNKNOWN).unwrap();
            builder.ret([first, second], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @ssa_memory_reads(%0: ptr<i8>) -> (i8, i8) {
    // CHECK: [[LOAD:%\d+]] = test.load %0;
    %1 = test.load %0;
    %2 = test.load %0;

    // CHECK-NEXT: builtin.ret [[LOAD]], [[LOAD]] : (i8, i8);
    builtin.ret %1, %2 : (i8, i8);
};
            "#
        );
    }

    #[test]
    fn cse_does_not_eliminate_memory_reads_in_graph_region() {
        use runic_core::dialects::test::TestOpBuilder;

        let mut test = Test::named("graph_memory_reads").in_module("graph_memory_reads");
        {
            let module_body = test.module().borrow().body().entry_block_ref().unwrap();
            let ptr_ty = test.context().ptr_type(test.context().i8());
            let mut builder = test.builder_mut();
            builder.set_insertion_point_to_end(module_body);

            let addr = builder.i32(0, SourceSpan::UNKNOWN);
            let ptr =
                builder.unrealized_conversion_cast(addr, ptr_ty, SourceSpan::UNKNOWN).unwrap();
            let first = TestOpBuilder::load(&mut builder, ptr, SourceSpan::UNKNOWN).unwrap();
            let second = TestOpBuilder::load(&mut builder, ptr, SourceSpan::UNKNOWN).unwrap();
            let same =
                ArithOpBuilder::eq(&mut builder, first, second, SourceSpan::UNKNOWN).unwrap();
            builder.assert(same, "loads differ", SourceSpan::UNKNOWN).unwrap();
        }

        let module = test.module().as_operation_ref();
        module.borrow().recursively_verify().expect("valid ir before CSE");

        let mut pm = PassManager::on::<Module>(test.context_rc(), Nesting::Implicit);
        pm.add_pass(Box::<CommonSubexpressionElimination>::default());
        pm.run(module).expect("valid ir");
        module.borrow().recursively_verify().expect("valid ir after CSE");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.module().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.module private @graph_memory_reads
    // CHECK: [[ADDR:%\d+]] = arith.constant 0 : i32;
    // CHECK-NEXT: [[PTR:%\d+]] = builtin.unrealized_conversion_cast [[ADDR]] <{ ty = #builtin.type<ptr<i8>> }>;
    // CHECK-NEXT: [[FIRST:%\d+]] = test.load [[PTR]];
    // CHECK-NEXT: [[SECOND:%\d+]] = test.load [[PTR]];
    // CHECK-NEXT: [[SAME:%\d+]] = arith.cmpi eq [[FIRST]], [[SECOND]];
    // CHECK-NEXT: cf.assert [[SAME]], "loads differ";
            "#
        );
    }

    /// Commutative operations are equal under [OpKey] regardless of operand order, and the key
    /// hashes operands in the same canonical order equality compares them in, so both the
    /// identical and the operand-swapped duplicates are always deduplicated. A pair that compared
    /// equal but hashed differently would be deduplicated or not depending on hash-bucket
    /// coincidences, that is, on allocation addresses.
    #[test]
    fn cse_commutative_operand_order() {
        let mut test =
            Test::new("commutative", |cx| ([cx.i32(), cx.i32()], [cx.i32(), cx.i32(), cx.i32()]));
        {
            let mut builder = test.function_builder();
            let a = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let b = builder.entry_block().borrow().arguments()[1] as ValueRef;
            let v0 = builder.add(a, b, SourceSpan::UNKNOWN).unwrap();
            let v1 = builder.add(b, a, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.add(a, b, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v0, v1, v2], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @commutative(%0: i32, %1: i32) -> (i32, i32, i32) {
    // CHECK: [[SUM:%\d+]] = arith.add %0, %1
    %2 = arith.add %0, %1;

    // Both the operand-swapped and the identical duplicate are deduplicated.
    %3 = arith.add %1, %0;
    %4 = arith.add %0, %1;

    // CHECK-NEXT: builtin.ret [[SUM]], [[SUM]], [[SUM]] : (i32, i32, i32);
    builtin.ret %2, %3, %4 : (i32, i32, i32);
};
            "#
        );
    }

    /// A [core::hash::Hasher] that records the exact byte stream written to it, so two keys can be
    /// compared on the input they feed a hasher rather than on a lossy `finish()` value.
    #[derive(Default)]
    struct RecordingHasher(alloc::vec::Vec<u8>);

    impl core::hash::Hasher for RecordingHasher {
        fn finish(&self) -> u64 {
            0
        }

        fn write(&mut self, bytes: &[u8]) {
            self.0.extend_from_slice(bytes);
        }
    }

    fn opkey_hash_stream(key: OpKey) -> alloc::vec::Vec<u8> {
        use core::hash::Hash;
        let mut hasher = RecordingHasher::default();
        key.hash(&mut hasher);
        hasher.0
    }

    /// Directly pins the [OpKey] Hash/Eq contract: whenever two
    /// operations are equal under [OpKey], they must feed an identical byte stream to the hasher.
    /// A violation makes CSE deduplication depend on hash-bucket coincidence, i.e. on allocation
    /// addresses, which is non-deterministic. Recording the hash input (rather than comparing
    /// `finish()` values) keeps the check exact and independent of the concrete hasher.
    #[test]
    fn opkey_hash_is_consistent_with_eq() {
        let mut test = Test::new("opkey_contract", |cx| {
            ([cx.i32(), cx.i32()], [cx.i32(), cx.i32(), cx.i32(), cx.i32()])
        });
        let (ab, ba, ab_again, aa) = {
            let mut builder = test.function_builder();
            let a = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let b = builder.entry_block().borrow().arguments()[1] as ValueRef;
            let ab = builder.add(a, b, SourceSpan::UNKNOWN).unwrap();
            let ba = builder.add(b, a, SourceSpan::UNKNOWN).unwrap();
            let ab_again = builder.add(a, b, SourceSpan::UNKNOWN).unwrap();
            let aa = builder.add(a, a, SourceSpan::UNKNOWN).unwrap();
            let ops = (
                ab.borrow().get_defining_op().unwrap(),
                ba.borrow().get_defining_op().unwrap(),
                ab_again.borrow().get_defining_op().unwrap(),
                aa.borrow().get_defining_op().unwrap(),
            );
            builder.ret([ab, ba, ab_again, aa], SourceSpan::UNKNOWN).unwrap();
            ops
        };

        // Commutative operands are canonicalized, so a swap is equal and must hash identically.
        assert!(OpKey(ab) == OpKey(ba), "commutative operand swap must be equal");
        assert_eq!(opkey_hash_stream(OpKey(ab)), opkey_hash_stream(OpKey(ba)));

        // Result identity is ignored, so two distinct `add(a, b)` ops (with distinct result
        // values) are equal and must hash identically. Guards against the hash observing more than
        // equality does.
        assert!(OpKey(ab) == OpKey(ab_again), "distinct result identity must be ignored");
        assert_eq!(opkey_hash_stream(OpKey(ab)), opkey_hash_stream(OpKey(ab_again)));

        // Sanity that the check is not vacuous: different operands are not equal, and the hash
        // streams differ accordingly.
        assert!(OpKey(ab) != OpKey(aa), "different operands must not be equal");
        assert_ne!(opkey_hash_stream(OpKey(ab)), opkey_hash_stream(OpKey(aa)));
    }

    /// [DefaultValueHasher] hashes a value by its thin address only. The fat pointer (data address
    /// plus the vtable pointer, which can differ for the same value across codegen units) would
    /// make equal CSE keys hash differently, and deduplication depend on where values live.
    #[test]
    fn default_value_hasher_hashes_thin_address_only() {
        use core::hash::Hash;

        use runic_core::equivalence::{DefaultValueHasher, ValueHasher};

        let mut test = Test::new("value_hasher", |cx| ([cx.i32()], [cx.i32()]));
        let value = {
            let mut builder = test.function_builder();
            let a = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.ret([a], SourceSpan::UNKNOWN).unwrap();
            a
        };

        let mut via_hasher = RecordingHasher::default();
        DefaultValueHasher.hash_value(value, &mut via_hasher);

        let mut via_thin_address = RecordingHasher::default();
        ValueRef::as_ptr(&value).addr().hash(&mut via_thin_address);

        assert_eq!(
            via_hasher.0, via_thin_address.0,
            "DefaultValueHasher must hash only the thin value address; hashing the fat pointer \
             makes CSE deduplication non-deterministic"
        );
    }

    #[test]
    fn basic() {
        let mut test = Test::new("basic", |cx| ([], [cx.i32(), cx.i32()]));
        {
            let mut builder = test.function_builder();
            let v0 = builder.i32(0, SourceSpan::UNKNOWN);
            let v1 = builder.i32(0, SourceSpan::UNKNOWN);
            let v2 = builder.i32(1, SourceSpan::UNKNOWN);
            let v3 = builder.mul(v0, v2, SourceSpan::UNKNOWN).unwrap();
            let v4 = builder.mul(v1, v2, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v3, v4], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @basic() -> (i32, i32) {
    // CHECK: [[V0:%\d+]] = arith.constant 0 : i32;
    %0 = arith.constant 0 : i32;
    %1 = arith.constant 0 : i32;

    // CHECK-NEXT: [[V2:%\d+]] = arith.constant 1 : i32;
    %2 = arith.constant 1 : i32;

    // CHECK-NEXT: [[V3:%\d+]] = arith.mul [[V0]], [[V2]];
    %3 = arith.mul %0, %2;
    %4 = arith.mul %1, %2;

    // CHECK-NEXT: builtin.ret [[V3]], [[V3]] : (i32, i32);
    builtin.ret %3, %3 : (i32, i32);
};
            "#
        );
    }

    #[test]
    fn many() {
        let mut test = Test::new("many", |cx| ([cx.i32(), cx.i32()], [cx.i32()]));
        {
            let mut builder = test.function_builder();
            let [v0, v1] = *builder.entry_block().borrow().arguments()[0..2].as_array().unwrap();
            let v0 = v0 as ValueRef;
            let v1 = v1 as ValueRef;
            let v2 = builder.add(v0, v1, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.add(v0, v1, SourceSpan::UNKNOWN).unwrap();
            let v4 = builder.add(v0, v1, SourceSpan::UNKNOWN).unwrap();
            let v5 = builder.add(v0, v1, SourceSpan::UNKNOWN).unwrap();
            let v6 = builder.add(v2, v3, SourceSpan::UNKNOWN).unwrap();
            let v7 = builder.add(v4, v5, SourceSpan::UNKNOWN).unwrap();
            let v8 = builder.add(v2, v4, SourceSpan::UNKNOWN).unwrap();
            let v9 = builder.add(v6, v7, SourceSpan::UNKNOWN).unwrap();
            let v10 = builder.add(v7, v8, SourceSpan::UNKNOWN).unwrap();
            let v11 = builder.add(v9, v10, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v11], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @many(%0: i32, %1: i32) -> i32 {
    // CHECK: [[V2:%\d+]] = arith.add %{{\d+}}, %{{\d+}};
    %2 = arith.add %0, %1;
    %3 = arith.add %0, %1;
    %4 = arith.add %0, %1;
    %5 = arith.add %0, %1;

    // CHECK-NEXT: [[V6:%\d+]] = arith.add [[V2]], [[V2]];
    %6 = arith.add %2, %3;
    %7 = arith.add %4, %5;
    %8 = arith.add %2, %4;

    // CHECK-NEXT: [[V9:%\d+]] = arith.add [[V6]], [[V6]];
    %9 = arith.add %6, %7;
    %10 = arith.add %7, %8;

    // CHECK-NEXT: [[V11:%\d+]] = arith.add [[V9]], [[V9]];
    %11 = arith.add %9, %10;

    // CHECK-NEXT: builtin.ret [[V11]] : (i32);
    builtin.ret %11 : (i32);
};
            "#
        );
    }

    /// Check that operations are not eliminated if they have different operands.
    #[test]
    fn ops_with_different_operands_are_not_elimited() {
        let mut test = Test::new("different_operands", |cx| ([], [cx.i32(), cx.i32()]));
        {
            let mut builder = test.function_builder();
            let v0 = builder.i32(0, SourceSpan::UNKNOWN);
            let v1 = builder.i32(1, SourceSpan::UNKNOWN);
            builder.ret([v0, v1], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @different_operands() -> (i32, i32) {
    // CHECK: [[V0:%\d+]] = arith.constant 0 : i32;
    // CHECK-NEXT: [[V1:%\d+]] = arith.constant 1 : i32;
    %0 = arith.constant 0 : i32;
    %1 = arith.constant 1 : i32;

    // CHECK-NEXT: builtin.ret [[V0]], [[V1]] : (i32, i32);
    builtin.ret %0, %1 : (i32, i32);
};
            "#
        );
    }

    /// Check that operations are not eliminated if they have different result types.
    #[test]
    fn ops_with_different_result_types_are_not_elimited() {
        let mut test = Test::new("different_results", |cx| ([cx.i32()], [cx.i64(), cx.i128()]));
        let (i64_ty, i128_ty) = (test.context().i64(), test.context().i128());
        {
            let mut builder = test.function_builder();
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let v1 = builder.sext(v0, i64_ty, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.sext(v0, i128_ty, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v1, v2], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @different_results(%0: i32) -> (i64, i128) {
    // CHECK: [[V1:%\d+]] = arith.sext %0 <{ ty = #builtin.type<i64> }>;
    // CHECK-NEXT: [[V2:%\d+]] = arith.sext %0 <{ ty = #builtin.type<i128> }>;
    v1 = arith.sext %0 <{ ty = #builtin.type<i64> }>;
    v2 = arith.sext %0 <{ ty = #builtin.type<i128> }>;

    // CHECK-NEXT: builtin.ret [[V1]], [[V2]] : (i64, i128);
    builtin.ret %1, %2 : (i64, i128);
};
            "#
        );
    }

    /// Check that operations are not eliminated if they have different attributes.
    #[test]
    fn ops_with_different_attributes_are_not_elimited() {
        use runic_core::Signedness;

        let mut test = Test::new("different_attributes", |cx| ([cx.i32()], [cx.i32(), cx.i32()]));
        {
            let mut builder = test.function_builder();
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let v1 = builder.i32(1, SourceSpan::UNKNOWN);
            let v2 =
                builder.add_checked(v0, v1, Signedness::Unsigned, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder
                .add_unchecked(v0, v1, Signedness::Unsigned, SourceSpan::UNKNOWN)
                .unwrap();
            builder.ret([v2, v3], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @different_attributes(%0: i32) -> (i32, i32) {
    // CHECK: [[V1:%\d+]] = arith.constant 1 : i32;
    v1 = arith.constant 1 : i32;

    // CHECK-NEXT: [[V2:%\d+]] = arith.add %0, %1 <{ overflow = #builtin.overflow<checked unsigned> }>;
    v2 = arith.add %0, %1 <{ overflow = #builtin.overflow<checked unsigned> }>;

    // CHECK-NEXT: [[V3:%\d+]] = arith.add %0, %1 <{ overflow = #builtin.overflow<unchecked unsigned> }>;
    v3 = arith.add %0, %1 <{ overflow = #builtin.overflow<unchecked unsigned> }>;

    // CHECK-NEXT: builtin.ret [[V2]], [[V3]] : (i32, i32);
    builtin.ret %2, %3 : (i32, i32);
};
            "#
        );
    }

    /// Identical effectful operations (asserts and stores) are neither erased, merged, nor
    /// reordered by dead-code elimination, CSE, control-flow sinking or operand sinking.
    #[test]
    fn generic_effect_order_survives_dce_cse_and_sinking() {
        use runic_core::dialects::test::TestOpBuilder;

        let mut test =
            Test::new("generic_barriers", |cx| ([cx.i1(), cx.ptr_type(cx.i32()), cx.i32()], []));
        {
            let mut b = test.function_builder();
            let [cond, ptr, value] =
                *b.entry_block().borrow().arguments()[0..3].as_array().unwrap();
            let (cond, ptr, value) = (cond as ValueRef, ptr as ValueRef, value as ValueRef);
            b.assert(cond, "barrier", SourceSpan::UNKNOWN).unwrap();
            b.store(ptr, value, SourceSpan::UNKNOWN).unwrap();
            b.assert(cond, "barrier", SourceSpan::UNKNOWN).unwrap();
            b.store(ptr, value, SourceSpan::UNKNOWN).unwrap();
            b.ret([], SourceSpan::UNKNOWN).unwrap();
        }
        let expected = test
            .function()
            .borrow()
            .entry_block()
            .borrow()
            .body()
            .iter()
            .map(|op| op.as_operation_ref())
            .collect::<Vec<_>>();
        test.apply_pass::<crate::DeadCodeElimination>(true).unwrap();
        test.apply_pass::<CommonSubexpressionElimination>(true).unwrap();
        test.apply_pass::<crate::ControlFlowSink>(true).unwrap();
        test.apply_pass::<crate::SinkOperandDefs>(true).unwrap();
        let actual = test
            .function()
            .borrow()
            .entry_block()
            .borrow()
            .body()
            .iter()
            .map(|op| op.as_operation_ref())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "effectful operations must retain identity and order");
    }

    /// Check that operations with side effects are not eliminated.
    #[test]
    fn ops_with_side_effects_are_not_elimited() {
        use runic_core::dialects::test::TestOpBuilder;

        let mut test = Test::new("side_effect", |cx| ([cx.ptr_type(cx.i8())], [cx.i8(), cx.i8()]));
        {
            let mut builder = test.function_builder();
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let v1 = builder.i8(1, SourceSpan::UNKNOWN);
            builder.store(v0, v1, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.load(v0, SourceSpan::UNKNOWN).unwrap();
            builder.store(v0, v1, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.load(v0, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v2, v3], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        std::println!("{output}");
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @side_effect(%0: ptr<i8>) -> (i8, i8) {
    // CHECK: [[V1:%\d+]] = arith.constant 1 : i8;
    %1 = arith.constant 1 : i8;

    // CHECK-NEXT: test.store %0, %1 : (ptr<i8>, i8);
    // CHECK-NEXT: [[V2:%\d+]] = test.load %0;
    test.store %0, %1 : (ptr<i8>, i8);
    %2 = test.load %0;

    // CHECK-NEXT: test.store %0, %1 : (ptr<i8>, i8);
    // CHECK-NEXT: [[V3:%\d+]] = test.load %0;
    test.store %0, %1 : (ptr<i8>, i8);
    %3 = test.load %0;

    // CHECK-NEXT: builtin.ret [[V2]], [[V3]] : (i8, i8);
    builtin.ret %2, %3 : (i8, i8);
};
            "#
        );
    }

    /// Counts the `cf.assert` operations in the body of the test's function.
    fn count_asserts(test: &Test) -> usize {
        let function = test.function();
        let function = function.borrow();
        function
            .body()
            .body()
            .iter()
            .map(|block| {
                block.body().iter().filter(|op| op.is::<runic_dialect_cf::Assert>()).count()
            })
            .sum()
    }

    /// Two identical `cf.assert` operations in one block are both kept: an assert is never merged.
    #[test]
    fn identical_asserts_in_one_block_are_not_merged() {
        let mut test = Test::new("identical_asserts", |cx| ([cx.i1()], []));
        {
            let mut builder = test.function_builder();
            let cond = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.assert(cond, "m", SourceSpan::UNKNOWN).unwrap();
            builder.assert(cond, "m", SourceSpan::UNKNOWN).unwrap();
            builder.ret(None, SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        assert_eq!(count_asserts(&test), 2);
        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        filecheck!(
            output,
            r#"
// CHECK: cf.assert %0, "m";
// CHECK-NEXT: cf.assert %0, "m";
// CHECK-NEXT: builtin.ret;
            "#
        );
    }

    /// An assert in a dominated block is not replaced by an identical one in its dominator.
    #[test]
    fn dominated_assert_is_not_replaced_by_dominating_one() {
        let mut test = Test::new("dominated_assert", |cx| ([cx.i1()], []));
        let next = {
            let mut builder = test.function_builder();
            let cond = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let next = builder.create_block();
            builder.assert(cond, "m", SourceSpan::UNKNOWN).unwrap();
            builder.br(next, [], SourceSpan::UNKNOWN).unwrap();
            builder.switch_to_block(next);
            builder.assert(cond, "m", SourceSpan::UNKNOWN).unwrap();
            builder.ret(None, SourceSpan::UNKNOWN).unwrap();
            next
        };

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        assert_eq!(count_asserts(&test), 2);
        assert!(
            next.borrow().body().iter().any(|op| op.is::<runic_dialect_cf::Assert>()),
            "the dominated assert must stay in its block"
        );
    }

    /// Check that operation definitions are properly propagated down the dominance tree.
    #[test]
    fn proper_propagation_of_ops_down_dominance_tree() {
        use runic_dialect_scf::StructuredControlFlowOpBuilder;

        let mut test = Test::new("down_propagate_while", |_| ([], []));
        let i32_ty = test.context().i32();
        {
            let mut builder = test.function_builder();
            let v0 = builder.i32(0, SourceSpan::UNKNOWN);
            let v1 = builder.i32(1, SourceSpan::UNKNOWN);
            let v2 = builder.i32(4, SourceSpan::UNKNOWN);
            let while_op =
                builder.r#while([v0, v1], &[i32_ty, i32_ty], SourceSpan::UNKNOWN).unwrap();
            builder.ret(None, SourceSpan::UNKNOWN).unwrap();
            {
                let before_block = while_op.borrow().before().entry().as_block_ref();
                let after_block = while_op.borrow().after().entry().as_block_ref();
                builder.switch_to_block(before_block);
                let [v3, v4] = *before_block.borrow().arguments()[0..2].as_array().unwrap();
                let v3 = v3 as ValueRef;
                let v4 = v4 as ValueRef;
                let v5 = builder.i32(1, SourceSpan::UNKNOWN);
                let v6 = builder.add(v3, v5, SourceSpan::UNKNOWN).unwrap();
                let v7 = builder.slt(v6, v2, SourceSpan::UNKNOWN).unwrap();
                builder.condition(v7, [v6, v4], SourceSpan::UNKNOWN).unwrap();

                builder.switch_to_block(after_block);
                let [v8, v9] = *after_block.borrow().arguments()[0..2].as_array().unwrap();
                builder.r#yield([v8 as ValueRef, v9 as ValueRef], SourceSpan::UNKNOWN).unwrap();
            }
        }

        test.apply_pass::<CommonSubexpressionElimination>(true).expect("invalid ir");

        let flags = Default::default();
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
        printer.print_operation(test.function().borrow());
        let output = format!("{}", printer.finish());
        std::println!("output: {output}");
        filecheck!(
            output,
            r#"
builtin.function public extern("C") @down_propagate_while() {
    // CHECK: [[V0:%\d+]] = arith.constant 0 : i32;
    %0 = arith.constant 0 : i32;
    // CHECK: [[V1:%\d+]] = arith.constant 1 : i32;
    %1 = arith.constant 1 : i32;
    // CHECK: [[V2:%\d+]] = arith.constant 4 : i32;
    %2 = arith.constant 4 : i32;

    // CHECK-NEXT: scf.while %0, %1 before {
    // CHECK-NEXT: ^block{{\d}}([[V3:%\d+]]: i32, [[V4:%\d+]]: i32):
    %10, %11 = scf.while %0, %1 before {
    ^block1(%3: i32, %4: i32):
        // CHECK-NEXT: [[V6:%\d+]] = arith.add [[V3]], [[V1]];
        %5 = arith.constant 1 : i32;
        %6 = arith.add %3, %5;
        // CHECK-NEXT: [[V7:%\d+]] = arith.cmpi slt [[V6]], [[V2]];
        %7 = arith.cmpi slt %6, %2;
        // CHECK-NEXT: scf.condition [[V7]], [[V6]], [[V4]] : (i1, i32, i32);
        scf.condition %7, %6, %4 : (i1, i32, i32);
    } after {
    ^block2(%8: i32, %9: i32):
        // CHECK-NEXT: } after {
        // CHECK-NEXT: ^block{{\d}}([[V8:%\d+]]: i32, [[V9:%\d+]]: i32):
        // CHECK-NEXT: scf.yield [[V8]], [[V9]] : (i32, i32);
        scf.yield %8, %9 : (i32, i32);
    } : (i32, i32) -> (i32, i32);

    // CHECK: builtin.ret;
    builtin.ret;
};
            "#
        );
    }

    /// Callees for the call tests: `@reads` and `@reads_too` declare a read of their argument,
    /// `@unknown` states no effects.
    const CALLEES: &str = r#"
    builtin.function public extern("C") @reads(ptr<i32>) -> i32 effects(read(arg 0));
    builtin.function public extern("C") @reads_too(ptr<i32>) -> i32 effects(read(arg 0));
    builtin.function public extern("C") @unknown(ptr<i32>) -> i32;"#;

    /// The function `@f(%p: ptr<i32>, %v: i32) -> (i32, i32)` with the body `body`, in a module
    /// that also holds [CALLEES], after CSE ran on it, as text.
    fn cse_in_module(body: &str) -> Result<alloc::string::String, runic_core::Report> {
        use runic_core::{
            SymbolTable,
            dialects::builtin::Function,
            parse::{self, ParserConfig},
        };

        let context = Rc::new(runic_core::Context::default());
        let text = format!(
            "builtin.module public @m {{\n{CALLEES}\nbuiltin.function public extern(\"C\") @f(%p: \
             ptr<i32>, %v: i32) -> (i32, i32) {{\n{body}\n}};\n}};"
        );
        let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "cse", text)
            .into_result()?;
        let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
        let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
        passes.add_pass(Box::<CommonSubexpressionElimination>::default());
        passes.enable_verifier(true);
        passes.run(function)?;
        Ok(function.borrow().to_string())
    }

    /// Two equal calls of a callee that only reads merge into the first.
    #[test]
    fn equal_calls_that_only_read_are_merged() -> Result<(), runic_core::Report> {
        let text = cse_in_module(
            r#"
    %1 = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32;
    %2 = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32;
    builtin.ret %1, %2 : (i32, i32);"#,
        )?;
        filecheck!(
            text,
            r#"
// CHECK: [[R:%\d+]] = builtin.call @reads(%0) : extern("C") (ptr<i32>) -> i32;
// CHECK-NEXT: builtin.ret [[R]], [[R]] : (i32, i32);
            "#
        );
        Ok(())
    }

    /// Calls that read are not merged across a write, across a call with unknown effects, or when
    /// their callees differ; calls with unknown effects are never merged.
    #[test]
    fn calls_are_not_merged_across_writes_unknown_effects_or_other_callees()
    -> Result<(), runic_core::Report> {
        let read = r#"builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32"#;
        let unknown = r#"builtin.call @unknown(%p) : extern("C") (ptr<i32>) -> i32"#;
        let read_too = r#"builtin.call @reads_too(%p) : extern("C") (ptr<i32>) -> i32"#;
        let cases = [
            (read, "test.store %p, %v : (ptr<i32>, i32);", read),
            (read, "%3 = builtin.call @unknown(%p) : extern(\"C\") (ptr<i32>) -> i32;", read),
            (unknown, "", unknown),
            (read, "", read_too),
        ];
        for (first, between, second) in cases {
            let body = format!(
                "%1 = {first};\n{between}\n%2 = {second};\nbuiltin.ret %1, %2 : (i32, i32);"
            );
            let text = cse_in_module(&body)?;
            assert_eq!(
                text.matches("builtin.call").count(),
                body.matches("builtin.call").count(),
                "{text}"
            );
            assert!(
                !text.contains("builtin.ret %1, %1") && !text.contains("builtin.ret %2, %2"),
                "{text}"
            );
        }
        Ok(())
    }

    /// Two equal indirect calls have unknown effects and are never merged, even through one
    /// reference to a function that only reads; two equal references are merged.
    #[test]
    fn indirect_calls_are_never_merged() -> Result<(), runic_core::Report> {
        let text = cse_in_module(
            r#"
    %f = builtin.function_ref @reads : (ptr<i32>) -> i32;
    %g = builtin.function_ref @reads : (ptr<i32>) -> i32;
    %1 = builtin.call_indirect %f(%p) : extern("C") (ptr<i32>) -> i32;
    %2 = builtin.call_indirect %g(%p) : extern("C") (ptr<i32>) -> i32;
    builtin.ret %1, %2 : (i32, i32);"#,
        )?;
        filecheck!(
            text,
            r#"
// CHECK: [[F:%\d+]] = builtin.function_ref @reads : (ptr<i32>) -> i32;
// CHECK-NEXT: [[R1:%\d+]] = builtin.call_indirect [[F]](%0) : extern("C") (ptr<i32>) -> i32;
// CHECK-NEXT: [[R2:%\d+]] = builtin.call_indirect [[F]](%0) : extern("C") (ptr<i32>) -> i32;
// CHECK-NEXT: builtin.ret [[R1]], [[R2]] : (i32, i32);
            "#
        );
        Ok(())
    }
}

//------------ TESTS TO FINISH TRANSLATING
/*

// CHECK-LABEL: @down_propagate
func.func @down_propagate() -> i32 {
  // CHECK-NEXT: %[[VAR_c1_i32:[0-9a-zA-Z_]+]] = arith.constant 1 : i32
  %0 = arith.constant 1 : i32

  // CHECK-NEXT: %[[VAR_true:[0-9a-zA-Z_]+]] = arith.constant true
  %cond = arith.constant true

  // CHECK-NEXT: cf.cond_br %[[VAR_true]], ^bb1, ^bb2(%[[VAR_c1_i32]] : i32)
  cf.cond_br %cond, ^bb1, ^bb2(%0 : i32)

^bb1: // CHECK: ^bb1:
  // CHECK-NEXT: cf.br ^bb2(%[[VAR_c1_i32]] : i32)
  %1 = arith.constant 1 : i32
  cf.br ^bb2(%1 : i32)

^bb2(%arg : i32):
  return %arg : i32
}

// -----

/// Check that operation definitions are NOT propagated up the dominance tree.
// CHECK-LABEL: @up_propagate_for
func.func @up_propagate_for() -> i32 {
  // CHECK: affine.for {{.*}} = 0 to 4 {
  affine.for %i = 0 to 4 {
    // CHECK-NEXT: %[[VAR_c1_i32_0:[0-9a-zA-Z_]+]] = arith.constant 1 : i32
    // CHECK-NEXT: "foo"(%[[VAR_c1_i32_0]]) : (i32) -> ()
    %0 = arith.constant 1 : i32
    "foo"(%0) : (i32) -> ()
  }

  // CHECK: %[[VAR_c1_i32:[0-9a-zA-Z_]+]] = arith.constant 1 : i32
  // CHECK-NEXT: return %[[VAR_c1_i32]] : i32
  %1 = arith.constant 1 : i32
  return %1 : i32
}

// -----

// CHECK-LABEL: func @up_propagate
func.func @up_propagate() -> i32 {
  // CHECK-NEXT:  %[[VAR_c0_i32:[0-9a-zA-Z_]+]] = arith.constant 0 : i32
  %0 = arith.constant 0 : i32

  // CHECK-NEXT: %[[VAR_true:[0-9a-zA-Z_]+]] = arith.constant true
  %cond = arith.constant true

  // CHECK-NEXT: cf.cond_br %[[VAR_true]], ^bb1, ^bb2(%[[VAR_c0_i32]] : i32)
  cf.cond_br %cond, ^bb1, ^bb2(%0 : i32)

^bb1: // CHECK: ^bb1:
  // CHECK-NEXT: %[[VAR_c1_i32:[0-9a-zA-Z_]+]] = arith.constant 1 : i32
  %1 = arith.constant 1 : i32

  // CHECK-NEXT: cf.br ^bb2(%[[VAR_c1_i32]] : i32)
  cf.br ^bb2(%1 : i32)

^bb2(%arg : i32): // CHECK: ^bb2
  // CHECK-NEXT: %[[VAR_c1_i32_0:[0-9a-zA-Z_]+]] = arith.constant 1 : i32
  %2 = arith.constant 1 : i32

  // CHECK-NEXT: %[[VAR_1:[0-9a-zA-Z_]+]] = arith.addi %{{.*}}, %[[VAR_c1_i32_0]] : i32
  %add = arith.addi %arg, %2 : i32

  // CHECK-NEXT: return %[[VAR_1]] : i32
  return %add : i32
}

// -----

/// The same test as above except that we are testing on a cfg embedded within
/// an operation region.
// CHECK-LABEL: func @up_propagate_region
func.func @up_propagate_region() -> i32 {
  // CHECK-NEXT: {{.*}} "foo.region"
  %0 = "foo.region"() ({
    // CHECK-NEXT:  %[[VAR_c0_i32:[0-9a-zA-Z_]+]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[VAR_true:[0-9a-zA-Z_]+]] = arith.constant true
    // CHECK-NEXT: cf.cond_br

    %1 = arith.constant 0 : i32
    %true = arith.constant true
    cf.cond_br %true, ^bb1, ^bb2(%1 : i32)

  ^bb1: // CHECK: ^bb1:
    // CHECK-NEXT: %[[VAR_c1_i32:[0-9a-zA-Z_]+]] = arith.constant 1 : i32
    // CHECK-NEXT: cf.br

    %c1_i32 = arith.constant 1 : i32
    cf.br ^bb2(%c1_i32 : i32)

  ^bb2(%arg : i32): // CHECK: ^bb2(%[[VAR_1:.*]]: i32):
    // CHECK-NEXT: %[[VAR_c1_i32_0:[0-9a-zA-Z_]+]] = arith.constant 1 : i32
    // CHECK-NEXT: %[[VAR_2:[0-9a-zA-Z_]+]] = arith.addi %[[VAR_1]], %[[VAR_c1_i32_0]] : i32
    // CHECK-NEXT: "foo.yield"(%[[VAR_2]]) : (i32) -> ()

    %c1_i32_0 = arith.constant 1 : i32
    %2 = arith.addi %arg, %c1_i32_0 : i32
    "foo.yield" (%2) : (i32) -> ()
  }) : () -> (i32)
  return %0 : i32
}

// -----

/// This test checks that nested regions that are isolated from above are
/// properly handled.
// CHECK-LABEL: @nested_isolated
func.func @nested_isolated() -> i32 {
  // CHECK-NEXT: arith.constant 1
  %0 = arith.constant 1 : i32

  // CHECK-NEXT: builtin.module
  // CHECK-NEXT: @nested_func
  builtin.module {
    func.func @nested_func() {
      // CHECK-NEXT: arith.constant 1
      %foo = arith.constant 1 : i32
      "foo.yield"(%foo) : (i32) -> ()
    }
  }

  // CHECK: "foo.region"
  "foo.region"() ({
    // CHECK-NEXT: arith.constant 1
    %foo = arith.constant 1 : i32
    "foo.yield"(%foo) : (i32) -> ()
  }) : () -> ()

  return %0 : i32
}

// -----

/// This test is checking that CSE gracefully handles values in graph regions
/// where the use occurs before the def, and one of the defs could be CSE'd with
/// the other.
// CHECK-LABEL: @use_before_def
func.func @use_before_def() {
  // CHECK-NEXT: test.graph_region
  test.graph_region {
    // CHECK-NEXT: arith.addi
    %0 = arith.addi %1, %2 : i32

    // CHECK-NEXT: arith.constant 1
    // CHECK-NEXT: arith.constant 1
    %1 = arith.constant 1 : i32
    %2 = arith.constant 1 : i32

    // CHECK-NEXT: "foo.yield"(%{{.*}}) : (i32) -> ()
    "foo.yield"(%0) : (i32) -> ()
  }
  return
}

// -----

/// This test is checking that CSE is removing duplicated read op that follow
/// other.
// CHECK-LABEL: @remove_direct_duplicated_read_op
func.func @remove_direct_duplicated_read_op() -> i32 {
  // CHECK-NEXT: %[[READ_VALUE:.*]] = "test.op_with_memread"() : () -> i32
  %0 = "test.op_with_memread"() : () -> (i32)
  %1 = "test.op_with_memread"() : () -> (i32)
  // CHECK-NEXT: %{{.*}} = arith.addi %[[READ_VALUE]], %[[READ_VALUE]] : i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}

// -----

/// This test is checking that CSE is removing duplicated read op that follow
/// other.
// CHECK-LABEL: @remove_multiple_duplicated_read_op
func.func @remove_multiple_duplicated_read_op() -> i64 {
  // CHECK: %[[READ_VALUE:.*]] = "test.op_with_memread"() : () -> i64
  %0 = "test.op_with_memread"() : () -> (i64)
  %1 = "test.op_with_memread"() : () -> (i64)
  // CHECK-NEXT: %{{.*}} = arith.addi %{{.*}}, %[[READ_VALUE]] : i64
  %2 = arith.addi %0, %1 : i64
  %3 = "test.op_with_memread"() : () -> (i64)
  // CHECK-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i64
  %4 = arith.addi %2, %3 : i64
  %5 = "test.op_with_memread"() : () -> (i64)
  // CHECK-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i64
  %6 = arith.addi %4, %5 : i64
  // CHECK-NEXT: return %{{.*}} : i64
  return %6 : i64
}

// -----

/// This test is checking that CSE is not removing duplicated read op that
/// have write op in between.
// CHECK-LABEL: @dont_remove_duplicated_read_op_with_sideeffecting
func.func @dont_remove_duplicated_read_op_with_sideeffecting() -> i32 {
  // CHECK-NEXT: %[[READ_VALUE0:.*]] = "test.op_with_memread"() : () -> i32
  %0 = "test.op_with_memread"() : () -> (i32)
  "test.op_with_memwrite"() : () -> ()
  // CHECK: %[[READ_VALUE1:.*]] = "test.op_with_memread"() : () -> i32
  %1 = "test.op_with_memread"() : () -> (i32)
  // CHECK-NEXT: %{{.*}} = arith.addi %[[READ_VALUE0]], %[[READ_VALUE1]] : i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}

// -----

// Check that an operation with a single region can CSE.
func.func @cse_single_block_ops(%a : tensor<?x?xf32>, %b : tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32):
    test.region_yield %arg0 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32):
    test.region_yield %arg0 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @cse_single_block_ops
//       CHECK:   %[[OP:.+]] = test.cse_of_single_block_op
//   CHECK-NOT:   test.cse_of_single_block_op
//       CHECK:   return %[[OP]], %[[OP]]

// -----

// Operations with different number of bbArgs dont CSE.
func.func @no_cse_varied_bbargs(%a : tensor<?x?xf32>, %b : tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    test.region_yield %arg0 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32):
    test.region_yield %arg0 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @no_cse_varied_bbargs
//       CHECK:   %[[OP0:.+]] = test.cse_of_single_block_op
//       CHECK:   %[[OP1:.+]] = test.cse_of_single_block_op
//       CHECK:   return %[[OP0]], %[[OP1]]

// -----

// Operations with different regions dont CSE
func.func @no_cse_region_difference_simple(%a : tensor<?x?xf32>, %b : tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    test.region_yield %arg0 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    test.region_yield %arg1 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @no_cse_region_difference_simple
//       CHECK:   %[[OP0:.+]] = test.cse_of_single_block_op
//       CHECK:   %[[OP1:.+]] = test.cse_of_single_block_op
//       CHECK:   return %[[OP0]], %[[OP1]]

// -----

// Operation with identical region with multiple statements CSE.
func.func @cse_single_block_ops_identical_bodies(%a : tensor<?x?xf32>, %b : tensor<?x?xf32>, %c : f32, %d : i1)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %1 = arith.divf %arg0, %arg1 : f32
    %2 = arith.remf %arg0, %c : f32
    %3 = arith.select %d, %1, %2 : f32
    test.region_yield %3 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %1 = arith.divf %arg0, %arg1 : f32
    %2 = arith.remf %arg0, %c : f32
    %3 = arith.select %d, %1, %2 : f32
    test.region_yield %3 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @cse_single_block_ops_identical_bodies
//       CHECK:   %[[OP:.+]] = test.cse_of_single_block_op
//   CHECK-NOT:   test.cse_of_single_block_op
//       CHECK:   return %[[OP]], %[[OP]]

// -----

// Operation with non-identical regions dont CSE.
func.func @no_cse_single_block_ops_different_bodies(%a : tensor<?x?xf32>, %b : tensor<?x?xf32>, %c : f32, %d : i1)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %1 = arith.divf %arg0, %arg1 : f32
    %2 = arith.remf %arg0, %c : f32
    %3 = arith.select %d, %1, %2 : f32
    test.region_yield %3 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = test.cse_of_single_block_op inputs(%a, %b) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %1 = arith.divf %arg0, %arg1 : f32
    %2 = arith.remf %arg0, %c : f32
    %3 = arith.select %d, %2, %1 : f32
    test.region_yield %3 : f32
  } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @no_cse_single_block_ops_different_bodies
//       CHECK:   %[[OP0:.+]] = test.cse_of_single_block_op
//       CHECK:   %[[OP1:.+]] = test.cse_of_single_block_op
//       CHECK:   return %[[OP0]], %[[OP1]]

// -----

func.func @failing_issue_59135(%arg0: tensor<2x2xi1>, %arg1: f32, %arg2 : tensor<2xi1>) -> (tensor<2xi1>, tensor<2xi1>) {
  %false_2 = arith.constant false
  %true_5 = arith.constant true
  %9 = test.cse_of_single_block_op inputs(%arg2) {
  ^bb0(%out: i1):
    %true_144 = arith.constant true
    test.region_yield %true_144 : i1
  } : tensor<2xi1> -> tensor<2xi1>
  %15 = test.cse_of_single_block_op inputs(%arg2) {
  ^bb0(%out: i1):
    %true_144 = arith.constant true
    test.region_yield %true_144 : i1
  } : tensor<2xi1> -> tensor<2xi1>
  %93 = arith.maxsi %false_2, %true_5 : i1
  return %9, %15 : tensor<2xi1>, tensor<2xi1>
}
// CHECK-LABEL: func @failing_issue_59135
//       CHECK:   %[[TRUE:.+]] = arith.constant true
//       CHECK:   %[[OP:.+]] = test.cse_of_single_block_op
//       CHECK:     test.region_yield %[[TRUE]]
//       CHECK:   return %[[OP]], %[[OP]]

// -----

func.func @cse_multiple_regions(%c: i1, %t: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  %r1 = scf.if %c -> (tensor<5xf32>) {
    %0 = tensor.empty() : tensor<5xf32>
    scf.yield %0 : tensor<5xf32>
  } else {
    scf.yield %t : tensor<5xf32>
  }
  %r2 = scf.if %c -> (tensor<5xf32>) {
    %0 = tensor.empty() : tensor<5xf32>
    scf.yield %0 : tensor<5xf32>
  } else {
    scf.yield %t : tensor<5xf32>
  }
  return %r1, %r2 : tensor<5xf32>, tensor<5xf32>
}
// CHECK-LABEL: func @cse_multiple_regions
//       CHECK:   %[[if:.*]] = scf.if {{.*}} {
//       CHECK:     tensor.empty
//       CHECK:     scf.yield
//       CHECK:   } else {
//       CHECK:     scf.yield
//       CHECK:   }
//   CHECK-NOT:   scf.if
//       CHECK:   return %[[if]], %[[if]]

// -----

// CHECK-LABEL: @cse_recursive_effects_success
func.func @cse_recursive_effects_success() -> (i32, i32, i32) {
  // CHECK-NEXT: %[[READ_VALUE:.*]] = "test.op_with_memread"() : () -> i32
  %0 = "test.op_with_memread"() : () -> (i32)

  // do something with recursive effects, containing no side effects
  %true = arith.constant true
  // CHECK-NEXT: %[[TRUE:.+]] = arith.constant true
  // CHECK-NEXT: %[[IF:.+]] = scf.if %[[TRUE]] -> (i32) {
  %1 = scf.if %true -> (i32) {
    %c42 = arith.constant 42 : i32
    scf.yield %c42 : i32
    // CHECK-NEXT: %[[C42:.+]] = arith.constant 42 : i32
    // CHECK-NEXT: scf.yield %[[C42]]
    // CHECK-NEXT: } else {
  } else {
    %c24 = arith.constant 24 : i32
    scf.yield %c24 : i32
    // CHECK-NEXT: %[[C24:.+]] = arith.constant 24 : i32
    // CHECK-NEXT: scf.yield %[[C24]]
    // CHECK-NEXT: }
  }

  // %2 can be removed
  // CHECK-NEXT: return %[[READ_VALUE]], %[[READ_VALUE]], %[[IF]] : i32, i32, i32
  %2 = "test.op_with_memread"() : () -> (i32)
  return %0, %2, %1 : i32, i32, i32
}

// -----

// CHECK-LABEL: @cse_recursive_effects_failure
func.func @cse_recursive_effects_failure() -> (i32, i32, i32) {
  // CHECK-NEXT: %[[READ_VALUE:.*]] = "test.op_with_memread"() : () -> i32
  %0 = "test.op_with_memread"() : () -> (i32)

  // do something with recursive effects, containing a write effect
  %true = arith.constant true
  // CHECK-NEXT: %[[TRUE:.+]] = arith.constant true
  // CHECK-NEXT: %[[IF:.+]] = scf.if %[[TRUE]] -> (i32) {
  %1 = scf.if %true -> (i32) {
    "test.op_with_memwrite"() : () -> ()
    // CHECK-NEXT: "test.op_with_memwrite"() : () -> ()
    %c42 = arith.constant 42 : i32
    scf.yield %c42 : i32
    // CHECK-NEXT: %[[C42:.+]] = arith.constant 42 : i32
    // CHECK-NEXT: scf.yield %[[C42]]
    // CHECK-NEXT: } else {
  } else {
    %c24 = arith.constant 24 : i32
    scf.yield %c24 : i32
    // CHECK-NEXT: %[[C24:.+]] = arith.constant 24 : i32
    // CHECK-NEXT: scf.yield %[[C24]]
    // CHECK-NEXT: }
  }

  // %2 can not be be removed because of the write
  // CHECK-NEXT: %[[READ_VALUE2:.*]] = "test.op_with_memread"() : () -> i32
  // CHECK-NEXT: return %[[READ_VALUE]], %[[READ_VALUE2]], %[[IF]] : i32, i32, i32
  %2 = "test.op_with_memread"() : () -> (i32)
  return %0, %2, %1 : i32, i32, i32
}
*/

#[cfg(test)]
#[path = "cse/region_equivalence_tests.rs"]
mod region_equivalence_tests;
