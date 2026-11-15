//! The checks [Operation::recursively_verify] makes beyond each operation's own: the shape of
//! blocks and regions ([verify_structure]), and that every value is defined before it is read
//! and no value crosses an operation isolated from above ([verify_dominance]).
//!
//! # Why the walk has its own dominance query
//!
//! The walk answers "does the definition dominate this use" itself, beside the shared query
//! (`impl Dominates<Operation> for dyn Value` over a [crate::dominance::DominanceInfo]), and does
//! not call it. [crate::dominance::DominanceInfo] finds the information of a region by a linear
//! search over every region it has seen, so asking it once per operand makes verification
//! quadratic in the size of a module (measured at 4.9x the cost of verification without the walk
//! on a module of 200 functions). The walk instead keeps the kind and the dominator tree of each
//! region on a stack of the regions that hold the operation being checked, and it has to walk a
//! use out to its definition's region anyway, for the isolation rule and for the reachability
//! rule. Its answer, for a use it checks, is the shared query's
//! `properly_dominates` with enclosing operations not counted; the randomized test
//! `the_walk_agrees_with_the_shared_query` (in `tests`) ties the two together.

#[cfg(test)]
mod model;
#[cfg(test)]
mod tests;

use alloc::{format, string::String, vec::Vec};
use core::cell::OnceCell;

use crate::{
    BlockRef, Operation, OperationRef, Region, RegionKindInterface, RegionRef, Report, Value,
    diagnostics::Spanned,
    dominance::DomTreeBase,
    traits::{IsolatedFromAbove, NoTerminator, Terminator},
};

/// The structural rules of blocks and regions that hold at `op`, checked once `op` has verified:
///
/// * an operation with successors is the last operation of its block;
/// * every successor of an operation is a block of the region that holds the operation;
/// * an operation with the [Terminator] trait is the last operation of its block;
/// * unless `op` has the [NoTerminator] trait, every block of each of its regions is non-empty
///   and ends with an operation that has the [Terminator] trait (a graph region included);
/// * the entry block of each region of `op` has no predecessors;
/// * a graph region of `op` (see [RegionKindInterface]) has at most one block.
///
/// The dominance walk ([verify_dominance]) and the dominator trees it builds rely on these.
pub(crate) fn verify_structure(op: &Operation) -> Result<(), Report> {
    if op.has_successors()
        && let Some(block) = op.parent()
    {
        let is_last = block.borrow().back().is_some_and(|last| last == op.as_operation_ref());
        if !is_last {
            return Err(crate::diagnostics::report!(
                message: (format!(
                    "'{}' has successors, but is not the last operation of its block",
                    op.name()
                )),
                labels: [
                    primary(
                        op.span(),
                        "an operation that transfers control must end its block"
                    ),
                ],
            ));
        }
        let region = block.parent();
        for (index, successor) in op.successor_iter().enumerate() {
            let dest = successor.dest.borrow().successor();
            if dest.parent() != region {
                let other = if dest.parent().is_none() {
                    "a block in no region"
                } else {
                    "a block of another region"
                };
                return Err(crate::diagnostics::report!(
                    message: (format!(
                        "successor #{index} of '{}' is not a block of its region",
                        op.name()
                    )),
                    labels: [
                        primary(
                            op.span(),
                            format!("this branches to {other}")
                        ),
                    ],
                ));
            }
        }
    }

    if op.implements::<dyn Terminator>()
        && let Some(block) = op.parent()
        && let Some(next) = op.as_operation_ref().next()
    {
        debug_assert!(next.parent() == Some(block));
        let next = next.borrow();
        return Err(crate::diagnostics::report!(
            message: (format!(
                "'{}' is a terminator, but is not the last operation of its block",
                op.name()
            )),
            labels: [
                primary(op.span(), "a terminator must end its block"),
                context(next.span(), format!("'{}' follows it here", next.name())),
            ],
        ));
    }

    if !op.implements::<dyn NoTerminator>() {
        for (index, region) in op.regions().iter().enumerate() {
            for (position, block) in region.body().iter().enumerate() {
                let Some(last) = block.body().back().as_pointer() else {
                    return Err(crate::diagnostics::report!(
                        message: (format!(
                            "block #{position} of region #{index} of '{}' is empty, and does not \
                             end with a terminator",
                            op.name()
                        )),
                        labels: [
                            primary(
                                op.span(),
                                "every block of this operation's regions must end with a \
                                 terminator"
                            ),
                        ],
                    ));
                };
                let last = last.borrow();
                if !last.implements::<dyn Terminator>() {
                    return Err(crate::diagnostics::report!(
                        message: (format!(
                            "block #{position} of region #{index} of '{}' does not end with a \
                             terminator",
                            op.name()
                        )),
                        labels: [
                            primary(
                                last.span(),
                                format!(
                                    "the block ends with '{}', which is not a terminator",
                                    last.name()
                                )
                            ),
                            context(
                                op.span(),
                                "every block of this operation's regions must end with a \
                                 terminator"
                            ),
                        ],
                    ));
                }
            }
        }
    }

    let graph = op
        .as_trait::<dyn RegionKindInterface>()
        .is_some_and(|kind| kind.has_graph_regions());
    for (index, region) in op.regions().iter().enumerate() {
        let Some(entry) = region.entry_block_ref() else {
            continue;
        };
        if let Some(predecessor) = entry.borrow().predecessors().next() {
            let branch = predecessor.owner;
            return Err(crate::diagnostics::report!(
                message: (format!(
                    "the entry block of region #{index} of '{}' has predecessors",
                    op.name()
                )),
                labels: [
                    primary(op.span(), "the entry block of a region may not be branched to"),
                    context(
                        branch.borrow().span(),
                        format!("'{}' branches to it here", branch.borrow().name())
                    ),
                ],
            ));
        }
        if graph {
            let blocks = region.body().iter().count();
            if blocks > 1 {
                return Err(crate::diagnostics::report!(
                    message: (format!(
                        "region #{index} of '{}' is a graph region with {blocks} blocks",
                        op.name()
                    )),
                    labels: [
                        primary(op.span(), "a graph region has at most one block"),
                    ],
                ));
            }
        }
    }
    Ok(())
}

/// Check every operand of every operation in the regions of `root`, at any depth: no operand is
/// defined outside an operation isolated from above that holds its user, and the definition of
/// every operand dominates its use. The first failure is returned.
///
/// The isolation rule holds for every use. The dominance rule holds for each use that can be
/// reached: one whose block is reachable from the entry of its region and which, walked out
/// through the operations that hold it to the region of the definition, sits in a block that is
/// reachable from the entry of that region. Such a use is dominated when, in that region (where
/// the use stands for the operation of that region that holds it):
///
/// * the region is a graph region, in which every definition dominates every use; or
/// * the definition is an argument of the use's block, or an operation before the use in it; or
/// * the definition's block properly dominates the use's block.
///
/// A use in a region that is not nested in the definition's region is never dominated, nor is a
/// use inside the operation that defines the value, nor a use of a value defined in no region.
/// The operands of `root` itself are not checked, nor is a use of a value defined outside `root`
/// (when `root` is verified alone): the walk does not look at regions it was not asked to verify.
/// Isolation is still checked up to `root`, `root` included.
///
/// The kind and the dominator tree of each region are computed once per call: the kind when the
/// walk enters the region, the tree (for a region with more than one block) the first time it is
/// needed. They are kept on a stack of the regions that hold the operation being checked, which a
/// use walked out to its definition's region climbs.
pub(crate) fn verify_dominance(root: &Operation) -> Result<(), Report> {
    let mut walk = DominanceWalk {
        root: root.as_operation_ref(),
        frames: Vec::new(),
        #[cfg(test)]
        verdicts: None,
    };
    walk.regions_of(root)
}

/// The walk's verdict on every operand of every operation in the regions of `root`, in walk order,
/// instead of the first failure.
#[cfg(test)]
pub(crate) fn dominance_verdicts(root: &Operation) -> Vec<(OperationRef, usize, Verdict)> {
    let mut walk = DominanceWalk {
        root: root.as_operation_ref(),
        frames: Vec::new(),
        verdicts: Some(Vec::new()),
    };
    walk.regions_of(root)
        .expect("recording the verdicts does not stop at a failure");
    walk.verdicts.unwrap()
}

/// What the walk concluded about one use.
#[cfg(test)]
pub(crate) type Verdict = Result<Checked, Report>;

/// A use the walk accepts: dominated by its definition, or not checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Checked {
    /// The definition dominates the use
    Dominated,
    /// The use, walked out to the definition's region, is in a block that cannot be reached
    Unreachable,
    /// The definition is outside the root being verified
    OutsideRoot,
}

/// What the walk knows of one region.
struct Frame {
    region: RegionRef,
    /// Whether the region has SSA dominance (it is not a graph region)
    ssa: bool,
    /// The dominator tree of a region with more than one block, built when first needed
    tree: OnceCell<Option<DomTreeBase<false>>>,
}

impl Frame {
    fn new(region: RegionRef) -> Self {
        let ssa = region.borrow().has_ssa_dominance();
        Self {
            region,
            ssa,
            tree: OnceCell::new(),
        }
    }

    fn tree(&self) -> Option<&DomTreeBase<false>> {
        self.tree
            .get_or_init(|| {
                let region = self.region.borrow();
                if region.is_empty() || region.has_one_block() {
                    None
                } else {
                    drop(region);
                    DomTreeBase::new(self.region).ok()
                }
            })
            .as_ref()
    }

    /// Whether `block`, a block of this region, is reachable from its entry
    fn is_reachable(&self, block: BlockRef) -> bool {
        self.tree().is_none_or(|tree| tree.is_reachable_from_entry(block))
    }

    /// Whether `a` properly dominates `b`, both blocks of this region
    fn properly_dominates(&self, a: BlockRef, b: BlockRef) -> bool {
        a != b && self.tree().is_some_and(|tree| tree.properly_dominates(Some(a), Some(b)))
    }
}

struct DominanceWalk {
    /// The operation being verified
    root: OperationRef,
    /// The regions that hold the operation being checked, outermost first: the first is a region
    /// of `root`
    frames: Vec<Frame>,
    /// When set, every verdict is recorded here and the walk does not stop at a failure
    #[cfg(test)]
    verdicts: Option<Vec<(OperationRef, usize, Verdict)>>,
}

/// Where a definition that does not dominate its use is, relative to the use.
enum Placement {
    /// Defined by the operation that reads it
    ItsOwnUser,
    /// Defined by an operation that holds the use in one of its regions
    Encloses(OperationRef),
    /// Defined later in the block that holds the use (or the operation holding it)
    LaterInBlock,
    /// Defined in a block that does not dominate the block that holds the use
    NonDominatingBlock,
    /// Defined inside a region of an operation of the use's region
    NestedRegion,
    /// Defined in a region that is not an ancestor of the use's region
    UnrelatedRegion,
    /// Defined by an operation that is in no block, or in a block that is in no region
    Detached,
}

impl Placement {
    fn describe(&self) -> String {
        match self {
            Self::ItsOwnUser => "it is defined by the operation that reads it".into(),
            Self::Encloses(op) => format!(
                "it is defined here by '{}', which holds the use in one of its regions",
                op.borrow().name()
            ),
            Self::LaterInBlock => "it is defined here, later in the same block".into(),
            Self::NonDominatingBlock => {
                "it is defined here, in a block that does not dominate the use".into()
            }
            Self::NestedRegion => "it is defined here, inside a region nested in an operation of \
                                   the use's region"
                .into(),
            Self::UnrelatedRegion => {
                "it is defined here, in a region that is not an ancestor of the use's region".into()
            }
            Self::Detached => "it is defined here, outside of every region".into(),
        }
    }
}

impl DominanceWalk {
    /// Check the uses in the regions of `op`, in order, descending into each operation's regions
    /// after its own operands.
    fn regions_of(&mut self, op: &Operation) -> Result<(), Report> {
        for region in op.regions().iter() {
            self.frames.push(Frame::new(region.as_region_ref()));
            let result = self.blocks_of(&region);
            self.frames.pop();
            result?;
        }
        Ok(())
    }

    fn blocks_of(&mut self, region: &Region) -> Result<(), Report> {
        for block in region.body().iter() {
            let reachable = self.frames.last().unwrap().is_reachable(block.as_block_ref());
            for child in block.body().iter() {
                for (index, operand) in child.operands().iter().enumerate() {
                    let operand = operand.borrow();
                    let verdict = self.check_use(&child, index, &*operand.value(), reachable);
                    #[cfg(test)]
                    if let Some(verdicts) = self.verdicts.as_mut() {
                        verdicts.push((child.as_operation_ref(), index, verdict));
                        continue;
                    }
                    verdict?;
                }
                if child.has_regions() {
                    self.regions_of(&child)?;
                }
            }
        }
        Ok(())
    }

    /// Check operand `index` of `user`, which reads `value`; `user` is in the innermost region of
    /// the stack, and `reachable` says whether its block is reachable from the region's entry.
    fn check_use(
        &self,
        user: &Operation,
        index: usize,
        value: &dyn Value,
        reachable: bool,
    ) -> Result<Checked, Report> {
        let def_block = value.parent_block();
        let Some(def_region) = def_block.and_then(|block| block.parent()) else {
            // A value defined in no region dominates nothing; it is not "outside" an isolated
            // operation either, so this is checked before isolation
            if !reachable {
                return Ok(Checked::Unreachable);
            }
            return Err(dominance_error(user, index, value, Placement::Detached));
        };

        // Walk the use out, through the operations that hold it, to the definition's region,
        // checking that it crosses no operation isolated from above on the way. `level` is the
        // frame of the region the walk is in; the walk stops at the regions of the root.
        let mut at = user.as_operation_ref();
        let mut at_block = user.parent().expect("a checked operation is in a block");
        let mut level = self.frames.len() - 1;
        loop {
            let region = self.frames[level].region;
            debug_assert!(at_block.parent() == Some(region));
            if region == def_region {
                break;
            }
            let holder = region.parent().expect("a region under the root is held by an operation");
            // A definition outside the region of the use, but inside the isolated operation,
            // is not in an ancestor region: the walk goes on, and fails to find it.
            if holder.borrow().implements::<dyn IsolatedFromAbove>()
                && !is_inside(def_region, holder)
            {
                return Err(isolation_error(user, index, value, holder));
            }
            if level == 0 {
                // `holder` is the root: the definition is outside it, or in a region under it
                // that does not hold the use
                if !is_inside(def_region, self.root) {
                    return Ok(Checked::OutsideRoot);
                }
                if !reachable {
                    return Ok(Checked::Unreachable);
                }
                let use_region = self.frames.last().unwrap().region;
                let placement = if is_nested_in(def_region, use_region) {
                    Placement::NestedRegion
                } else {
                    Placement::UnrelatedRegion
                };
                return Err(dominance_error(user, index, value, placement));
            }
            at = holder;
            at_block = holder.parent().expect("an operation under the root is in a block");
            level -= 1;
        }

        if !reachable {
            return Ok(Checked::Unreachable);
        }
        let (ancestor, block) = (at, at_block);
        let frame = &self.frames[level];
        if !frame.ssa {
            return Ok(Checked::Dominated);
        }
        // A use in an unreachable block of the definition's region is in dead code
        if ancestor != user.as_operation_ref() && !frame.is_reachable(block) {
            return Ok(Checked::Unreachable);
        }
        let def_block = def_block.expect("the definition is in the region the walk found");
        let placement = if def_block == block {
            let Some(def_op) = value.get_defining_op() else {
                // A block argument dominates every operation of its block
                return Ok(Checked::Dominated);
            };
            if def_op == ancestor {
                if ancestor == user.as_operation_ref() {
                    Placement::ItsOwnUser
                } else {
                    Placement::Encloses(def_op)
                }
            } else if def_op.borrow().is_before_in_block(&ancestor) {
                return Ok(Checked::Dominated);
            } else {
                Placement::LaterInBlock
            }
        } else if frame.properly_dominates(def_block, block) {
            return Ok(Checked::Dominated);
        } else {
            Placement::NonDominatingBlock
        };
        Err(dominance_error(user, index, value, placement))
    }
}

/// Returns true if `region` is a region of `op`, or nested in one at any depth.
fn is_inside(region: RegionRef, op: OperationRef) -> bool {
    let mut current = Some(region);
    while let Some(region) = current {
        let Some(holder) = region.parent() else {
            return false;
        };
        if holder == op {
            return true;
        }
        current = holder.parent_region();
    }
    false
}

/// Returns true if `inner` is a region nested, at any depth, in an operation of `outer`.
fn is_nested_in(inner: RegionRef, outer: RegionRef) -> bool {
    let mut current = inner.borrow().parent_region();
    while let Some(region) = current {
        if region == outer {
            return true;
        }
        current = region.borrow().parent_region();
    }
    false
}

/// Where `value` is defined, for a label.
fn definition_span(value: &dyn Value) -> crate::SourceSpan {
    match value.get_defining_op() {
        Some(op) => op.borrow().span(),
        None => value.span(),
    }
}

fn dominance_error(
    user: &Operation,
    index: usize,
    value: &dyn Value,
    placement: Placement,
) -> Report {
    let name = user.name();
    if let Placement::ItsOwnUser = placement {
        return crate::diagnostics::report!(
            message: (format!("operand #{index} of '{name}' is not dominated by its definition")),
            labels: [
                primary(user.span(), placement.describe()),
            ],
        );
    }
    crate::diagnostics::report!(
        message: (format!("operand #{index} of '{name}' is not dominated by its definition")),
        labels: [
            primary(user.span(), format!("operand #{index} is read here")),
            context(definition_span(value), placement.describe()),
        ],
    )
}

fn isolation_error(
    user: &Operation,
    index: usize,
    value: &dyn Value,
    isolated: OperationRef,
) -> Report {
    let isolated = isolated.borrow();
    let isolated_name = isolated.name();
    crate::diagnostics::report!(
        message: (format!(
            "operand #{index} of '{}' is defined outside '{isolated_name}', which is isolated \
             from above",
            user.name()
        )),
        labels: [
            primary(user.span(), format!("operand #{index} is read here")),
            context(
                isolated.span(),
                format!("inside this '{isolated_name}'")
            ),
            context(definition_span(value), "it is defined here"),
        ],
    )
}
