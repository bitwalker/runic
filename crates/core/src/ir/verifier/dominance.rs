//! The checks [Operation::recursively_verify] makes beyond each operation's own: the shape of
//! blocks and regions ([verify_structure]), and that every value is defined before it is read
//! and no value crosses an operation isolated from above ([verify_dominance]).

#[cfg(test)]
mod tests;

use alloc::{format, rc::Rc, string::String, vec::Vec};
use core::cell::{OnceCell, RefCell};

use crate::{
    BlockRef, FxHashMap, Operation, OperationRef, Region, RegionKindInterface, RegionRef, Report,
    Value, diagnostics::Spanned, dominance::DomTreeBase, traits::IsolatedFromAbove,
};

/// The structural rules of blocks and regions that hold at `op`, checked once `op` has verified:
///
/// * an operation with successors is the last operation of its block;
/// * every successor of an operation is a block of the region that holds the operation;
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
/// use inside the operation that defines the value. The operands of `root` itself are not
/// checked.
///
/// The kind and the dominator tree of each region are computed once per call: the kind when the
/// walk enters the region, the tree (for a region with more than one block) the first time it is
/// needed. They are kept on a stack of the regions that hold the operation being checked, which a
/// use walked out to its definition's region climbs; a region outside `root` is looked up in a
/// map.
pub(crate) fn verify_dominance(root: &Operation) -> Result<(), Report> {
    let mut walk = DominanceWalk {
        frames: Vec::new(),
        outside: RefCell::new(FxHashMap::default()),
    };
    walk.regions_of(root)
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
    /// The regions that hold the operation being checked, outermost first
    frames: Vec<Frame>,
    /// The regions outside the root that a use was walked out to
    outside: RefCell<FxHashMap<RegionRef, Rc<Frame>>>,
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
                    self.check_use(&child, index, &*operand.value(), reachable)?;
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
    ) -> Result<(), Report> {
        let def_block = value.parent_block();
        let def_region = def_block.and_then(|block| block.parent());

        // Walk the use out, through the operations that hold it, to the definition's region,
        // checking that it crosses no operation isolated from above on the way. `level` is the
        // frame of the region the walk is in, while it is under the root.
        let mut at = user.as_operation_ref();
        let mut at_block = user.parent().expect("a checked operation is in a block");
        let mut level = Some(self.frames.len() - 1);
        let found = loop {
            let Some(region) = at_block.parent() else {
                break None;
            };
            if Some(region) == def_region {
                break Some((at, at_block, level));
            }
            let Some(holder) = region.parent() else {
                break None;
            };
            // A definition outside the region of the use, but inside the isolated operation,
            // is not in an ancestor region: the walk goes on, and fails to find it.
            if holder.borrow().implements::<dyn IsolatedFromAbove>()
                && !def_region.is_some_and(|def_region| is_inside(def_region, holder))
            {
                return Err(isolation_error(user, index, value, holder));
            }
            at = holder;
            level = level.and_then(|level| level.checked_sub(1));
            match holder.parent() {
                Some(block) => at_block = block,
                None => break None,
            }
        };

        if !reachable {
            return Ok(());
        }
        let Some((ancestor, block, level)) = found else {
            let placement = match def_region {
                None => Placement::Detached,
                Some(def_region) => {
                    let use_region = self.frames.last().unwrap().region;
                    if is_nested_in(def_region, use_region) {
                        Placement::NestedRegion
                    } else {
                        Placement::UnrelatedRegion
                    }
                }
            };
            return Err(dominance_error(user, index, value, placement));
        };
        let outside;
        let frame = match level {
            Some(level) => &self.frames[level],
            None => {
                let region = block.parent().expect("the walk found a block in a region");
                outside = self
                    .outside
                    .borrow_mut()
                    .entry(region)
                    .or_insert_with(|| Rc::new(Frame::new(region)))
                    .clone();
                &*outside
            }
        };
        debug_assert!(Some(frame.region) == def_region);
        if !frame.ssa {
            return Ok(());
        }
        // A use in an unreachable block of the definition's region is in dead code
        if ancestor != user.as_operation_ref() && !frame.is_reachable(block) {
            return Ok(());
        }
        let def_block = def_block.expect("the definition is in the region the walk found");
        let placement = if def_block == block {
            let Some(def_op) = value.get_defining_op() else {
                // A block argument dominates every operation of its block
                return Ok(());
            };
            if def_op == ancestor {
                if ancestor == user.as_operation_ref() {
                    Placement::ItsOwnUser
                } else {
                    Placement::Encloses(def_op)
                }
            } else if def_op.borrow().is_before_in_block(&ancestor) {
                return Ok(());
            } else {
                Placement::LaterInBlock
            }
        } else if frame.properly_dominates(def_block, block) {
            return Ok(());
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
