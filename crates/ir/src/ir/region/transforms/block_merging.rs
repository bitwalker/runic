use alloc::{collections::BTreeSet, vec::Vec};
use core::hash::Hasher;

use smallvec::SmallVec;

use super::RegionTransformFailed;
use crate::{
    BlockArgument, BlockRef, FxHashMap, FxHashSet, FxHasher, OpResult, Operation, OperationRef,
    Region, RegionRef, Rewriter, SuccessorOperands, Value, ValueRef,
    equivalence::{IgnoreValueEquivalence, IgnoreValueHasher, OperationEquivalenceFlags},
    traits::{BranchOpInterface, Transparent},
};

/// What is needed to compare two blocks for merging. Blocks are equivalent if they hold the same
/// operations in the same order. Only the operands that come from outside of their block may
/// differ: the uses of values defined in the block must correspond.
///
/// Equivalent:
///
/// ```text,ignore
/// ^bb1(%arg0: i32):
///     test.ret %arg0, %foo : (i32, i32);
/// ^bb2(%arg1: i32):
///     test.ret %arg1, %bar : (i32, i32);
/// ```
///
/// Not equivalent:
///
/// ```text,ignore
/// ^bb1(%arg0: i32):
///     test.ret %foo, %arg0 : (i32, i32);
/// ^bb2(%arg1: i32):
///     test.ret %arg1, %bar : (i32, i32);
/// ```
///
/// Corresponds to upstream `BlockEquivalenceData`.
struct BlockEquivalenceData {
    /// The block this data refers to
    block: BlockRef,
    /// A hash of the operations of the block, their operands and results left out
    hash: u64,
    /// The relative order of each operation of the block that has results: the number of values
    /// defined in the block (its arguments included) before the operation.
    op_order_index: FxHashMap<OperationRef, usize>,
}

impl BlockEquivalenceData {
    fn new(block: BlockRef) -> Self {
        let mut op_order_index = FxHashMap::default();
        let mut hasher = FxHasher::default();

        let b = block.borrow();
        let mut order = b.num_arguments();
        for op in b.body() {
            let num_results = op.num_results();
            if num_results > 0 {
                op_order_index.insert(op.as_operation_ref(), order);
                order += num_results;
            }
            op.hash_with_options(
                OperationEquivalenceFlags::IGNORE_LOCATIONS,
                IgnoreValueHasher,
                IgnoreValueHasher,
                &mut hasher,
            );
        }

        Self {
            block,
            hash: hasher.finish(),
            op_order_index,
        }
    }

    /// The order of `value`, which is defined in the block of this data: an argument's index, or
    /// the order of the operation defining it plus the index of the result.
    fn get_order_of(&self, value: &dyn Value) -> usize {
        assert!(value.parent_block() == Some(self.block), "expected value of this block");

        if let Some(block_arg) = value.downcast_ref::<BlockArgument>() {
            return block_arg.index();
        }

        let result = value.downcast_ref::<OpResult>().unwrap();
        let order = self.op_order_index.get(&result.owner()).expect("expected op to have an order");
        order + result.index()
    }
}

/// A cluster of blocks to be merged into its leader.
///
/// Corresponds to upstream `BlockMergeCluster`.
struct BlockMergeCluster {
    /// The equivalence data of the leader of the cluster
    leader: BlockEquivalenceData,
    /// The blocks that can be merged into the leader
    blocks_to_merge: SmallVec<[BlockRef; 1]>,
    /// The operands, as (operation index in the block, operand index) pairs, that differ between
    /// the blocks of the cluster and become arguments of the leader when the cluster is merged
    operands_to_merge: BTreeSet<(usize, usize)>,
}

impl BlockMergeCluster {
    fn new(leader: BlockEquivalenceData) -> Self {
        Self {
            leader,
            blocks_to_merge: SmallVec::new(),
            operands_to_merge: BTreeSet::new(),
        }
    }

    /// Add the block of `block_data` to this cluster, if it is equivalent to the leader.
    ///
    /// Returns false, and leaves the cluster unchanged, if it is not.
    ///
    /// Corresponds to upstream `BlockMergeCluster::addToCluster`.
    fn add_to_cluster(&mut self, block_data: &BlockEquivalenceData) -> bool {
        if self.leader.hash != block_data.hash {
            return false;
        }
        let leader_block = self.leader.block;
        let merge_block = block_data.block;
        if !leader_block.borrow().argument_types().eq(merge_block.borrow().argument_types()) {
            return false;
        }

        let lhs_ops = block_ops(leader_block);
        let rhs_ops = block_ops(merge_block);
        // Make sure that the block sizes are equivalent.
        if lhs_ops.len() != rhs_ops.len() {
            return false;
        }

        // The operands that differ between the leader and the new block.
        let mut mismatched_operands = SmallVec::<[(usize, usize); 8]>::new();
        for (op_index, (lhs, rhs)) in lhs_ops.iter().zip(rhs_ops.iter()).enumerate() {
            let lhs = lhs.borrow();
            let rhs = rhs.borrow();
            // Check that the operations are equivalent.
            if !lhs.is_equivalent_with_options(
                &rhs,
                OperationEquivalenceFlags::IGNORE_LOCATIONS,
                IgnoreValueEquivalence,
            ) {
                return false;
            }

            // Compare the operands of the two operations. An operand defined in the block must
            // refer to the same value, relative to the block, in both.
            for (operand_index, (lhs_operand, rhs_operand)) in
                lhs.operands().iter().zip(rhs.operands().iter()).enumerate()
            {
                let lhs_value = lhs_operand.borrow().as_value_ref();
                let rhs_value = rhs_operand.borrow().as_value_ref();
                if lhs_value == rhs_value {
                    continue;
                }
                let lhs_value = lhs_value.borrow();
                let rhs_value = rhs_value.borrow();
                // Check that the types of the operands match.
                if lhs_value.ty() != rhs_value.ty() {
                    return false;
                }

                // Check that these uses are both external, or both internal.
                let lhs_is_in_block = lhs_value.parent_block() == Some(leader_block);
                let rhs_is_in_block = rhs_value.parent_block() == Some(merge_block);
                if lhs_is_in_block != rhs_is_in_block {
                    return false;
                }

                // Operands defined in another block may differ: they become new arguments of the
                // merged block, passed by its predecessors, which a result of a predecessor's
                // terminator cannot be.
                if !lhs_is_in_block {
                    if !is_valid_successor_arg(leader_block, &*lhs_value)
                        || !is_valid_successor_arg(merge_block, &*rhs_value)
                    {
                        return false;
                    }
                    mismatched_operands.push((op_index, operand_index));
                    continue;
                }

                // Otherwise, these operands must have the same order within their block.
                if self.leader.get_order_of(&*lhs_value) != block_data.get_order_of(&*rhs_value) {
                    return false;
                }
            }

            // If either operation has a use outside of its block, the blocks cannot be merged:
            // the merged operation would be neither of them, but one or the other depending on
            // the predecessor the merged block was entered from.
            if rhs.is_used_outside_of_block(&merge_block)
                || lhs.is_used_outside_of_block(&leader_block)
            {
                return false;
            }
        }

        // If we get here, the blocks are equivalent and can be merged.
        self.operands_to_merge.extend(mismatched_operands);
        self.blocks_to_merge.push(merge_block);
        true
    }

    /// Merge the blocks of this cluster into its leader.
    ///
    /// Returns `Err` if the cluster has no blocks to merge, or if the merge needs new arguments
    /// and a predecessor of one of its blocks cannot pass them.
    ///
    /// Corresponds to upstream `BlockMergeCluster::merge`.
    fn merge(self, rewriter: &mut dyn Rewriter) -> Result<(), RegionTransformFailed> {
        // Don't consider clusters that don't have blocks to merge.
        if self.blocks_to_merge.is_empty() {
            return Err(RegionTransformFailed);
        }

        let leader_block = self.leader.block;
        if !self.operands_to_merge.is_empty() {
            // The predecessors of each of the blocks must be able to pass the new arguments.
            if !able_to_update_pred_operands(leader_block)
                || !self.blocks_to_merge.iter().copied().all(able_to_update_pred_operands)
            {
                return Err(RegionTransformFailed);
            }

            // The operations of each block, the leader first.
            let cluster_ops = core::iter::once(leader_block)
                .chain(self.blocks_to_merge.iter().copied())
                .map(block_ops)
                .collect::<SmallVec<[_; 2]>>();

            // The value each block uses for each operand to merge, which its predecessors will
            // pass to the merged block.
            let new_arguments = cluster_ops
                .iter()
                .map(|ops| {
                    self.operands_to_merge
                        .iter()
                        .map(|&(op_index, operand_index)| {
                            ops[op_index].borrow().operands()[operand_index].borrow().as_value_ref()
                        })
                        .collect::<SmallVec<[ValueRef; 8]>>()
                })
                .collect::<SmallVec<[_; 2]>>();

            // Every block's predecessors must be able to pass its values. `add_to_cluster` checks
            // this only where a block differs from the leader, but a block that uses the same
            // value as the leader where another block differs passes it too: a result of the
            // terminator of one of its predecessors would become an operand of that terminator.
            let blocks = core::iter::once(leader_block).chain(self.blocks_to_merge.iter().copied());
            for (block, values) in blocks.zip(new_arguments.iter()) {
                if !values.iter().all(|value| is_valid_successor_arg(block, &*value.borrow())) {
                    return Err(RegionTransformFailed);
                }
            }

            // Each operand to merge of the leader uses a new argument instead.
            let num_old_arguments = leader_block.borrow().num_arguments();
            for &(op_index, operand_index) in self.operands_to_merge.iter() {
                let op = cluster_ops[0][op_index];
                let mut operand = op.borrow().operands()[operand_index];
                let (ty, span) = {
                    let value = operand.borrow().as_value_ref();
                    let value = value.borrow();
                    (value.ty(), value.span())
                };
                let arg = rewriter.context().append_block_argument(leader_block, ty, span);
                rewriter.notify_operation_modification_started(&op);
                operand.borrow_mut().set(arg);
                rewriter.notify_operation_modified(op);
            }

            // Prune redundant arguments and update the leader block argument list
            let new_arguments =
                prune_redundant_arguments(new_arguments, rewriter, num_old_arguments, leader_block);

            // Update the predecessors for each of the blocks.
            let blocks = core::iter::once(leader_block).chain(self.blocks_to_merge.iter().copied());
            for (block, arguments) in blocks.zip(new_arguments.iter()) {
                append_to_predecessors(block, arguments, rewriter);
            }
        }

        // Replace all uses of the merged blocks with the leader and erase them.
        for block in self.blocks_to_merge {
            rewriter.replace_all_uses_of_block_with(block, leader_block);
            rewriter.erase_block(block);
        }
        Ok(())
    }
}

/// The operations of `block`, in order.
fn block_ops(block: BlockRef) -> SmallVec<[OperationRef; 8]> {
    block.borrow().body().iter().map(|op| op.as_operation_ref()).collect()
}

/// True unless `value` is a result of the terminator of a predecessor of `block`: such a value
/// does not dominate that terminator, so it cannot be passed by it as a successor operand.
fn is_valid_successor_arg(block: BlockRef, value: &dyn Value) -> bool {
    let (Some(defining_op), Some(defining_block)) = (value.get_defining_op(), value.parent_block())
    else {
        return true;
    };
    if defining_block.borrow().terminator() != Some(defining_op) {
        return true;
    }
    !block.borrow().predecessors().any(|pred| pred.predecessor() == defining_block)
}

/// True if the terminator of every predecessor of `block` can take more operands for its edge to
/// `block`.
///
/// Corresponds to upstream `ableToUpdatePredOperands`.
fn able_to_update_pred_operands(block: BlockRef) -> bool {
    block
        .borrow()
        .predecessors()
        .all(|pred| can_append_successor_operands(&pred.owner.borrow(), pred.index as usize))
}

/// True if operands can be appended to the forwarded operands of successor `succ_index` of `op`:
/// `op` implements [BranchOpInterface], the successor's forwarded operands are its whole operand
/// group, and no other successor of `op` forwards the same group.
fn can_append_successor_operands(op: &Operation, succ_index: usize) -> bool {
    let Some(branch) = op.as_trait::<dyn BranchOpInterface>() else {
        return false;
    };
    let group = op.successors()[succ_index].operand_group;
    let group_range = op.operands().group(group as usize).range().clone();
    let successor_operands = branch.get_successor_operands(succ_index);
    successor_operands.forwarded().range() == &group_range
        && op.successors().iter().filter(|succ| succ.operand_group == group).count() == 1
}

/// Append `arguments` to the forwarded operands of each edge into `block`.
fn append_to_predecessors(block: BlockRef, arguments: &[ValueRef], rewriter: &mut dyn Rewriter) {
    let edges = block
        .borrow()
        .predecessors()
        .map(|pred| (pred.owner, pred.index as usize))
        .collect::<SmallVec<[_; 4]>>();
    for (mut terminator, succ_index) in edges {
        rewriter.notify_operation_modification_started(&terminator);
        {
            let context = rewriter.context();
            let mut op = terminator.borrow_mut();
            let branch = op
                .as_trait_mut::<dyn BranchOpInterface>()
                .expect("expected the predecessor to be a branch");
            let operands = arguments
                .iter()
                .map(|value| context.make_operand(*value, terminator, 0))
                .collect::<SmallVec<[_; 8]>>();
            branch.get_successor_operands_mut(succ_index).forwarded_mut().extend(operands);
        }
        rewriter.notify_operation_modified(terminator);
    }
}

/// Prune the redundant new arguments of `block`: given the lists of values each block of a
/// cluster passes, a new argument for which every list passes the same value as for an earlier
/// one is replaced by that earlier argument and erased. E.g. with one list `[x, y, z, x]`, the
/// fourth argument is replaced by the first.
///
/// The new arguments are the arguments of `block` after the first `num_old_arguments`.
///
/// Returns the lists without the values passed for the erased arguments.
///
/// Corresponds to upstream `pruneRedundantArguments`.
fn prune_redundant_arguments(
    new_arguments: SmallVec<[SmallVec<[ValueRef; 8]>; 2]>,
    rewriter: &mut dyn Rewriter,
    num_old_arguments: usize,
    mut block: BlockRef,
) -> SmallVec<[SmallVec<[ValueRef; 8]>; 2]> {
    let Some(first_list) = new_arguments.first() else {
        return new_arguments;
    };
    let num_args = first_list.len();

    // The index of the first occurrence of each value of the first list.
    let mut first_value_to_index = FxHashMap::<ValueRef, usize>::default();
    for (index, value) in first_list.iter().enumerate() {
        first_value_to_index.entry(*value).or_insert(index);
    }

    // For each argument that can be replaced by an earlier one, the index of that one. Only the
    // first occurrence in the first list is considered, so that this is not quadratic in the
    // number of arguments: with `[a, a, a]` and `[c, b, b]`, the third argument is not replaced
    // by the second.
    let mut index_to_replacement = FxHashMap::<usize, usize>::default();
    for (j, value) in first_list.iter().enumerate() {
        let k = first_value_to_index[value];
        if k == j {
            continue;
        }
        // It can be pruned only if it is redundant in every list.
        if new_arguments[1..].iter().all(|list| list[k] == list[j]) {
            index_to_replacement.insert(j, k);
        }
    }

    // Populate the pruned argument lists.
    let pruned = new_arguments
        .iter()
        .map(|list| {
            list.iter()
                .enumerate()
                .filter(|(j, _)| !index_to_replacement.contains_key(j))
                .map(|(_, value)| *value)
                .collect()
        })
        .collect();

    // Replace the block's redundant arguments, then erase them.
    let mut to_erase = (0..num_args)
        .filter_map(|j| index_to_replacement.get(&j).map(|k| (j, *k)))
        .collect::<SmallVec<[_; 4]>>();
    for &(j, k) in to_erase.iter() {
        let (old_arg, new_arg) = {
            let block = block.borrow();
            (
                block.get_argument(num_old_arguments + j) as ValueRef,
                block.get_argument(num_old_arguments + k) as ValueRef,
            )
        };
        rewriter.replace_all_uses_of_value_with(old_arg, new_arg);
    }
    to_erase.reverse();
    let mut block = block.borrow_mut();
    for (j, _) in to_erase {
        block.erase_argument(num_old_arguments + j);
    }
    pruned
}

impl Region {
    /// Identify identical blocks within the regions in `regions`, and in the regions nested in
    /// them, and merge them, adding block arguments where they differ in values defined
    /// elsewhere.
    ///
    /// Returns `Ok` if any blocks were merged, `Err` otherwise.
    ///
    /// Corresponds to upstream `mergeIdenticalBlocks(rewriter, regions)`.
    pub(in crate::ir::region) fn merge_identical_blocks(
        regions: &[RegionRef],
        rewriter: &mut dyn Rewriter,
    ) -> Result<(), RegionTransformFailed> {
        let mut worklist = RegionWorklist::default();
        for region in regions {
            worklist.insert(*region);
        }

        let mut any_changed = false;
        while let Some(region) = worklist.pop() {
            if Self::merge_identical_blocks_in(region, rewriter).is_ok() {
                worklist.insert(region);
                any_changed = true;
            }

            // Add any nested regions to the worklist.
            for block in region.borrow().body() {
                for op in block.body() {
                    for nested in op.regions() {
                        worklist.insert(nested.as_region_ref());
                    }
                }
            }
        }

        if any_changed {
            Ok(())
        } else {
            Err(RegionTransformFailed)
        }
    }

    /// Identify identical blocks within `region` and merge them, adding block arguments where
    /// they differ in values defined elsewhere.
    ///
    /// The entry block is never merged; nor is a block that holds an operation with a non-empty
    /// region, a block whose arguments are used outside of it, or a block that holds a
    /// [Transparent] operation (debug information, whose use of a value must not become a new
    /// block argument and successor operands).
    ///
    /// Returns `Ok` if any blocks were merged, `Err` otherwise.
    ///
    /// Corresponds to upstream `mergeIdenticalBlocks(rewriter, region)`.
    fn merge_identical_blocks_in(
        region: RegionRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<(), RegionTransformFailed> {
        let blocks = {
            let region = region.borrow();
            if region.is_empty() || region.has_one_block() {
                return Err(RegionTransformFailed);
            }
            region
                .body()
                .iter()
                .skip(1)
                .map(|block| block.as_block_ref())
                .collect::<Vec<_>>()
        };

        // Group the blocks other than the entry by their successors, the groups in the order of
        // their first block, so that the result does not depend on hashing. Each group holds the
        // candidates for merging with each other.
        let mut groups = Vec::<SmallVec<[BlockRef; 2]>>::new();
        let mut group_of_successors = FxHashMap::<SmallVec<[BlockRef; 2]>, usize>::default();
        for block in blocks {
            let successors = {
                let b = block.borrow();
                (0..b.num_successors()).map(|index| b.get_successor(index)).collect()
            };
            let group = *group_of_successors.entry(successors).or_insert_with(|| {
                groups.push(SmallVec::new());
                groups.len() - 1
            });
            groups[group].push(block);
        }

        let mut merged_any_blocks = false;
        for blocks in groups {
            if blocks.len() == 1 {
                continue;
            }

            let mut clusters = SmallVec::<[BlockMergeCluster; 1]>::new();
            for block in blocks {
                if !is_merge_candidate(block) {
                    continue;
                }

                // Try to add this block to an existing cluster.
                let data = BlockEquivalenceData::new(block);
                if !clusters.iter_mut().any(|cluster| cluster.add_to_cluster(&data)) {
                    clusters.push(BlockMergeCluster::new(data));
                }
            }
            for cluster in clusters {
                merged_any_blocks |= cluster.merge(rewriter).is_ok();
            }
        }

        if merged_any_blocks {
            Ok(())
        } else {
            Err(RegionTransformFailed)
        }
    }
}

/// True if `block` may be merged with an equivalent block: no operation in it holds a non-empty
/// region or is [Transparent], and its arguments are used only in it.
fn is_merge_candidate(block: BlockRef) -> bool {
    let b = block.borrow();
    let ops_allow_merging = b.body().iter().all(|op| {
        !op.implements::<dyn Transparent>() && op.regions().iter().all(|region| region.is_empty())
    });
    ops_allow_merging
        && !b.arguments().iter().any(|arg| arg.borrow().is_used_outside_of_block(&block))
}

/// A stack of regions, each at most once.
#[derive(Default)]
struct RegionWorklist {
    regions: SmallVec<[RegionRef; 1]>,
    members: FxHashSet<RegionRef>,
}

impl RegionWorklist {
    fn insert(&mut self, region: RegionRef) {
        if self.members.insert(region) {
            self.regions.push(region);
        }
    }

    fn pop(&mut self) -> Option<RegionRef> {
        let region = self.regions.pop()?;
        self.members.remove(&region);
        Some(region)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, rc::Rc, string::String};

    use super::*;
    use crate::{
        Context, EntityWithId, Op, OpPrinter,
        derive::{EffectOpInterface, OpParser, OpPrinter, operation},
        dialects::{builtin::Function, test::TestDialect},
        effects::MemoryEffectOpInterface,
        parse::{self, ParserConfig},
        patterns::{NoopRewriterListener, RewriterImpl},
        print::same_program,
        testing::text::describe,
        traits::{AnyType, NoTerminator, SingleBlock, Terminator},
    };

    /// A branch to one block: `test.bm_br ^bb1(%0 : i32);`.
    #[derive(EffectOpInterface, OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
    pub struct BmBr {
        #[successor]
        target: Successor,
    }

    impl BranchOpInterface for BmBr {}

    /// A branch to one of two blocks: `test.bm_cond_br %c ^bb1(%0 : i32), ^bb2 : (i1);`.
    #[derive(EffectOpInterface, OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
    pub struct BmCondBr {
        #[operand]
        condition: AnyType,
        #[successor]
        then_dest: Successor,
        #[successor]
        else_dest: Successor,
    }

    impl BranchOpInterface for BmCondBr {}

    /// A terminator with two successors that is no [BranchOpInterface]: nothing can be said about
    /// how its operands reach its successors. `test.bm_opaque_br ^bb1, ^bb2;`.
    #[derive(EffectOpInterface, OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(MemoryEffectOpInterface, OpPrinter)
    )]
    pub struct BmOpaqueBr {
        #[successor]
        first: Successor,
        #[successor]
        second: Successor,
    }

    /// A branch to one block whose first operand is its own, the rest being forwarded to the
    /// block: `test.bm_tagged_br ^bb1(%tag, %0 : i32, i32);` passes `%0`. Operands appended to
    /// its operand group would not be forwarded last.
    #[derive(EffectOpInterface, OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
    pub struct BmTaggedBr {
        #[successor]
        target: Successor,
    }

    impl BranchOpInterface for BmTaggedBr {
        fn get_successor_operands(&self, index: usize) -> crate::SuccessorOperandRange<'_> {
            let op = self.as_operation();
            let group = op.operands().group(op.successors()[index].operand_group as usize);
            let forwarded = (group.range().start + 1)..group.range().end;
            let operands = crate::EntityRange::new(forwarded, op.operands().all().as_slice());
            crate::SuccessorOperandRange::forward(operands)
        }
    }

    /// A branch to one block that produces the first argument of the block itself, and forwards
    /// its operands to the rest: `test.bm_produced_br ^bb1(%0 : i32);` into `^bb1(%e, %x)`.
    #[derive(EffectOpInterface, OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
    )]
    pub struct BmProducedBr {
        #[successor]
        target: Successor,
    }

    impl BranchOpInterface for BmProducedBr {
        fn get_successor_operands(&self, index: usize) -> crate::SuccessorOperandRange<'_> {
            let op = self.as_operation();
            let group = op.successors()[index].operand_group as usize;
            crate::SuccessorOperandRange::new(1, op.operands().group(group))
        }
    }

    /// A branch to one of two blocks that defines a result:
    /// `%0 = "test.bm_producing_br"() [ ^bb1, ^bb2 ] : () -> i32;`.
    #[derive(EffectOpInterface)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(BranchOpInterface, MemoryEffectOpInterface)
    )]
    pub struct BmProducingBr {
        #[successor]
        first: Successor,
        #[successor]
        second: Successor,
        #[result]
        value: AnyType,
    }

    impl BranchOpInterface for BmProducingBr {}

    /// Debug information about a value, which is no use of it: `test.bm_note %0 : (i32);`.
    #[derive(EffectOpInterface, OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(Transparent),
        implements(MemoryEffectOpInterface, OpPrinter)
    )]
    pub struct BmNote {
        #[operand]
        value: AnyType,
    }

    /// An operation holding one block: `"test.bm_holder"() ({ .. }) : () -> ();`.
    #[derive(EffectOpInterface)]
    #[operation(
        dialect = TestDialect,
        traits(SingleBlock, NoTerminator),
        implements(MemoryEffectOpInterface)
    )]
    pub struct BmHolder {
        #[region]
        body: Region,
    }

    /// An operation holding a region of any number of blocks:
    /// `"test.bm_cfg_holder"() ({ .. }) : () -> ();`.
    #[derive(EffectOpInterface)]
    #[operation(dialect = TestDialect, implements(MemoryEffectOpInterface))]
    pub struct BmCfgHolder {
        #[region]
        body: Region,
    }

    /// Ends a block of a [BmCfgHolder]: `"test.bm_yield"() : () -> ();`.
    #[derive(EffectOpInterface)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(MemoryEffectOpInterface)
    )]
    pub struct BmYield {}

    /// `body` as the function `@f` with the parameters `params`.
    fn function(params: &str, body: &str) -> String {
        format!("builtin.function public extern(\"C\") @f({params}) {{\n{body}\n}};")
    }

    fn parse_function(context: &Rc<Context>, source: &str) -> OperationRef {
        parse::parse::<Function>(ParserConfig::new(context.clone()), "f", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}\n{source}", describe(&err)))
            .as_operation_ref()
    }

    /// Merge the identical blocks of the function `input` and print it.
    ///
    /// Returns whether any blocks were merged, and the function as printed.
    fn merge(input: &str) -> (bool, String) {
        let context = Rc::new(Context::default());
        let function = parse_function(&context, input);
        let body = function.borrow().regions().front().as_pointer().unwrap();
        let mut rewriter = RewriterImpl::<NoopRewriterListener>::new(context.clone());
        let merged = Region::merge_identical_blocks(&[body], &mut rewriter).is_ok();
        function
            .borrow()
            .recursively_verify()
            .unwrap_or_else(|err| panic!("{}\n{}", describe(&err), function.borrow()));
        (merged, format!("{}", function.borrow()))
    }

    /// `function` as printed once parsed.
    fn printed(function: &str) -> String {
        let context = Rc::new(Context::default());
        format!("{}", parse_function(&context, function).borrow())
    }

    /// Block merging turns the function `input` into the function `expected`.
    #[track_caller]
    fn assert_merges(input: &str, expected: &str) {
        let (merged, output) = merge(input);
        assert!(merged, "no blocks were merged in\n{output}");
        let expected = printed(expected);
        assert!(same_program(&output, &expected), "expected\n{expected}\nfound\n{output}");
    }

    /// Block merging leaves the function `input` as it is.
    #[track_caller]
    fn assert_unchanged(input: &str) {
        let (merged, output) = merge(input);
        assert!(!merged, "blocks were merged in\n{output}");
        let expected = printed(input);
        assert!(same_program(&output, &expected), "expected\n{expected}\nfound\n{output}");
    }

    #[test]
    fn identical_blocks_are_merged() {
        let input = function(
            "%c: i1",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    builtin.ret;
^b:
    builtin.ret;",
        );
        let expected = function(
            "%c: i1",
            "    test.bm_cond_br %c ^a, ^a : (i1);
^a:
    builtin.ret;",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn a_predecessor_that_is_no_branch_does_not_prevent_a_merge_without_new_arguments() {
        let input = function(
            "",
            "    test.bm_opaque_br ^a, ^b;
^a:
    builtin.ret;
^b:
    builtin.ret;",
        );
        let expected = function(
            "",
            "    test.bm_opaque_br ^a, ^a;
^a:
    builtin.ret;",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn identical_blocks_with_matching_arguments_are_merged() {
        let input = function(
            "%c: i1, %p: i32",
            "    test.bm_cond_br %c ^a(%p : i32), ^b(%p : i32) : (i1);
^a(%x: i32):
    %y = test.add %x, %x;
    test.ret %y : (i32);
^b(%z: i32):
    %w = test.add %z, %z;
    test.ret %w : (i32);",
        );
        let expected = function(
            "%c: i1, %p: i32",
            "    test.bm_cond_br %c ^a(%p : i32), ^a(%p : i32) : (i1);
^a(%x: i32):
    %y = test.add %x, %x;
    test.ret %y : (i32);",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn blocks_that_differ_in_an_external_operand_are_merged_with_a_new_argument() {
        let input = function(
            "%c: i1, %p: i32, %q: i32",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    test.ret %p : (i32);
^b:
    test.ret %q : (i32);",
        );
        let expected = function(
            "%c: i1, %p: i32, %q: i32",
            "    test.bm_cond_br %c ^a(%p : i32), ^a(%q : i32) : (i1);
^a(%x: i32):
    test.ret %x : (i32);",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn new_arguments_follow_the_existing_ones() {
        let input = function(
            "%c: i1, %p: i32, %q: i32",
            "    test.bm_cond_br %c ^a(%q : i32), ^b(%p : i32) : (i1);
^a(%x: i32):
    test.ret %p, %x : (i32, i32);
^b(%y: i32):
    test.ret %q, %y : (i32, i32);",
        );
        let expected = function(
            "%c: i1, %p: i32, %q: i32",
            "    test.bm_cond_br %c ^a(%q, %p : i32, i32), ^a(%p, %q : i32, i32) : (i1);
^a(%x: i32, %e: i32):
    test.ret %e, %x : (i32, i32);",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn a_new_argument_that_repeats_an_earlier_one_in_every_predecessor_is_pruned() {
        let input = function(
            "%c: i1, %p: i32, %q: i32",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    %x = test.add %p, %p;
    test.ret %x : (i32);
^b:
    %y = test.add %q, %q;
    test.ret %y : (i32);",
        );
        let expected = function(
            "%c: i1, %p: i32, %q: i32",
            "    test.bm_cond_br %c ^a(%p : i32), ^a(%q : i32) : (i1);
^a(%e: i32):
    %x = test.add %e, %e;
    test.ret %x : (i32);",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn a_new_argument_that_repeats_an_earlier_one_in_only_some_predecessors_is_kept() {
        let input = function(
            "%c: i1, %p: i32, %q: i32, %r: i32",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    %x = test.add %p, %p;
    test.ret %x : (i32);
^b:
    %y = test.add %q, %r;
    test.ret %y : (i32);",
        );
        let expected = function(
            "%c: i1, %p: i32, %q: i32, %r: i32",
            "    test.bm_cond_br %c ^a(%p, %p : i32, i32), ^a(%q, %r : i32, i32) : (i1);
^a(%e: i32, %f: i32):
    %x = test.add %e, %f;
    test.ret %x : (i32);",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn blocks_whose_arguments_are_used_differently_are_not_merged() {
        assert_unchanged(&function(
            "%c: i1, %p: i32, %q: i32",
            "    test.bm_cond_br %c ^a(%q : i32), ^b(%p : i32) : (i1);
^a(%x: i32):
    test.ret %p, %x : (i32, i32);
^b(%y: i32):
    test.ret %y, %q : (i32, i32);",
        ));
    }

    #[test]
    fn blocks_whose_arguments_differ_in_type_or_number_are_not_merged() {
        assert_unchanged(&function(
            "%c: i1, %p: i32, %q: i16",
            "    test.bm_cond_br %c ^a(%p : i32), ^b(%q : i16) : (i1);
^a(%x: i32):
    test.sink %x : (i32);
    builtin.ret;
^b(%y: i16):
    test.sink %y : (i16);
    builtin.ret;",
        ));
        assert_unchanged(&function(
            "%c: i1, %p: i32",
            "    test.bm_cond_br %c ^a(%p : i32), ^b : (i1);
^a(%x: i32):
    builtin.ret;
^b:
    builtin.ret;",
        ));
    }

    #[test]
    fn blocks_whose_operations_differ_in_kind_or_number_are_not_merged() {
        assert_unchanged(&function(
            "%c: i1, %p: i32",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    %x = test.add %p, %p;
    builtin.ret;
^b:
    %y = test.mul %p, %p;
    builtin.ret;",
        ));
        assert_unchanged(&function(
            "%c: i1, %p: i32",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    test.sink %p : (i32);
    builtin.ret;
^b:
    builtin.ret;",
        ));
    }

    #[test]
    fn blocks_whose_operands_differ_in_type_are_not_merged() {
        assert_unchanged(&function(
            "%c: i1, %p: i32, %q: i16",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    test.sink %p : (i32);
    builtin.ret;
^b:
    test.sink %q : (i16);
    builtin.ret;",
        ));
    }

    #[test]
    fn blocks_holding_regions_are_not_merged() {
        assert_unchanged(&function(
            "%c: i1",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    \"test.bm_holder\"() ({
        %z = test.constant 1 : i32;
    }) : () -> ();
    builtin.ret;
^b:
    \"test.bm_holder\"() ({
        %z = test.constant 1 : i32;
    }) : () -> ();
    builtin.ret;",
        ));
    }

    #[test]
    fn blocks_holding_debug_information_are_not_merged() {
        assert_unchanged(&function(
            "%c: i1, %p: i32",
            "    test.bm_cond_br %c ^a, ^b : (i1);
^a:
    test.bm_note %p : (i32);
    builtin.ret;
^b:
    test.bm_note %p : (i32);
    builtin.ret;",
        ));
    }

    #[test]
    fn blocks_are_merged_in_nested_regions() {
        let input = function(
            "%c: i1",
            "    \"test.bm_cfg_holder\"() ({
        test.bm_cond_br %c ^a, ^b : (i1);
    ^a:
        \"test.bm_yield\"() : () -> ();
    ^b:
        \"test.bm_yield\"() : () -> ();
    }) : () -> ();
    builtin.ret;",
        );
        let expected = function(
            "%c: i1",
            "    \"test.bm_cfg_holder\"() ({
        test.bm_cond_br %c ^a, ^a : (i1);
    ^a:
        \"test.bm_yield\"() : () -> ();
    }) : () -> ();
    builtin.ret;",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn a_new_argument_needs_predecessors_that_can_pass_it() {
        assert_unchanged(&function(
            "%p: i32, %q: i32",
            "    test.bm_opaque_br ^a, ^b;
^a:
    test.ret %p : (i32);
^b:
    test.ret %q : (i32);",
        ));
    }

    #[test]
    fn a_new_argument_needs_predecessors_that_forward_their_whole_operand_group() {
        assert_unchanged(&function(
            "%c: i1, %p: i32, %q: i32",
            "    test.bm_cond_br %c ^ta, ^tb : (i1);
^ta:
    test.bm_tagged_br ^a(%c, %p : i1, i32);
^tb:
    test.bm_tagged_br ^b(%c, %q : i1, i32);
^a(%x: i32):
    test.ret %x, %p : (i32, i32);
^b(%y: i32):
    test.ret %y, %q : (i32, i32);",
        ));
    }

    /// `^y` joins `^l`'s cluster using the same value, `%r`, and `^x` then makes that operand a new
    /// argument: `^y`'s predecessor would pass `%r`, its own result, so nothing is merged.
    #[test]
    fn a_block_that_would_pass_a_result_of_its_predecessor_terminator_prevents_the_merge() {
        assert_unchanged(&function(
            "%c: i1, %s: i32",
            "    %r = \"test.bm_producing_br\"() [ ^z, ^y ] : () -> i32;
^l:
    test.ret %r : (i32);
^y:
    test.ret %r : (i32);
^x:
    test.ret %s : (i32);
^z:
    test.bm_cond_br %c ^l, ^x : (i1);",
        ));
    }

    /// A new argument is forwarded after the operands a branch produces itself. Once `^a` and `^b`
    /// are merged, `^pa` and `^pb` differ only in what they forward, and are merged in turn.
    #[test]
    fn a_new_argument_follows_the_produced_and_forwarded_operands() {
        let input = function(
            "%c: i1, %p: i32, %q: i32, %s: i32, %t: i32",
            "    test.bm_cond_br %c ^pa, ^pb : (i1);
^pa:
    test.bm_produced_br ^a(%p : i32);
^pb:
    test.bm_produced_br ^b(%q : i32);
^a(%e: i32, %x: i32):
    test.ret %e, %x, %s : (i32, i32, i32);
^b(%f: i32, %y: i32):
    test.ret %f, %y, %t : (i32, i32, i32);",
        );
        let expected = function(
            "%c: i1, %p: i32, %q: i32, %s: i32, %t: i32",
            "    test.bm_cond_br %c ^p(%p, %s : i32, i32), ^p(%q, %t : i32, i32) : (i1);
^p(%u: i32, %v: i32):
    test.bm_produced_br ^a(%u, %v : i32, i32);
^a(%e: i32, %x: i32, %n: i32):
    test.ret %e, %x, %n : (i32, i32, i32);",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn a_back_edge_passes_the_new_argument_too() {
        let input = function(
            "%c: i1, %d: i1, %p: i32",
            "    %e = test.eq %p, %p;
    test.bm_cond_br %c ^b, ^exit : (i1);
^a:
    test.bm_cond_br %e ^a, ^exit : (i1);
^b:
    test.bm_cond_br %d ^a, ^exit : (i1);
^exit:
    builtin.ret;",
        );
        let expected = function(
            "%c: i1, %d: i1, %p: i32",
            "    %e = test.eq %p, %p;
    test.bm_cond_br %c ^a(%d : i1), ^exit : (i1);
^a(%x: i1):
    test.bm_cond_br %x ^a(%e : i1), ^exit : (i1);
^exit:
    builtin.ret;",
        );
        assert_merges(&input, &expected);
    }

    #[test]
    fn blocks_whose_values_are_used_outside_of_them_are_not_merged() {
        // `^b` and `^next` both branch to `^loop`, the same way, but `^loop` uses what `^b`
        // computes: merged, it would see what `^next` computes on the back edge.
        assert_unchanged(&function(
            "%p: i32, %i: i32",
            "    %one = test.constant 1 : i32;
    %lt = test.eq %i, %p;
    test.bm_cond_br %lt ^b, ^exit : (i1);
^b:
    %ip1 = test.add %i, %one;
    test.bm_br ^loop(%ip1 : i32);
^loop(%j: i32):
    %jlt = test.eq %j, %p;
    test.sink %j, %ip1 : (i32, i32);
    test.bm_cond_br %jlt ^next, ^exit : (i1);
^next:
    %jp1 = test.add %j, %one;
    test.bm_br ^loop(%jp1 : i32);
^exit:
    builtin.ret;",
        ));
    }

    #[test]
    fn a_result_of_a_predecessor_terminator_cannot_become_a_new_argument() {
        // `%r` does not dominate the terminator that defines it, so it cannot be passed by it.
        assert_unchanged(&function(
            "",
            "    %r = \"test.bm_producing_br\"() [ ^a, ^b ] : () -> i32;
^a:
    test.bm_br ^join(%r : i32);
^b:
    %s = test.constant 1 : i32;
    test.bm_br ^c;
^c:
    test.bm_br ^join(%s : i32);
^join(%x: i32):
    test.sink %x : (i32);
    builtin.ret;",
        ));
    }

    /// Values defined in the blocks are compared by their order in their block: the blocks hash
    /// alike, as the hash leaves operands out, but `%z` uses the values of its block in another
    /// order than `%w` does.
    #[test]
    fn values_defined_in_the_blocks_must_be_used_in_the_same_order() {
        let context = Rc::new(Context::default());
        let source = function(
            "%c: i1, %p: i32",
            "    test.bm_cond_br %c ^a(%p : i32), ^b(%p : i32) : (i1);
^a(%x: i32):
    %y = test.add %x, %p;
    %z = test.add %y, %x;
    test.ret %z : (i32);
^b(%u: i32):
    %v = test.add %u, %p;
    %w = test.add %u, %v;
    test.ret %w : (i32);",
        );
        let function = parse_function(&context, &source);
        let body = function.borrow().regions().front().as_pointer().unwrap();
        let blocks = body.borrow().body().iter().map(|b| b.as_block_ref()).collect::<Vec<_>>();
        let leader = BlockEquivalenceData::new(blocks[1]);
        let other = BlockEquivalenceData::new(blocks[2]);
        assert_eq!(leader.hash, other.hash);
        let mut cluster = BlockMergeCluster::new(leader);
        assert!(!cluster.add_to_cluster(&other));
        assert_unchanged(&source);
    }
}
