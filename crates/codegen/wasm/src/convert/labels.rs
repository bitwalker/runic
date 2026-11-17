//! The translation of the levels of `scf.break` and `scf.continue` to the labels of `wasm.br`.
//!
//! The two count differently. `scf.break N` and `scf.continue N` count, from 1, the enclosing
//! operations that receive breaking control flow: `scf.while`, and `wasm.block`, `wasm.loop` and
//! `wasm.if` too (`scf.if` and `scf.index_switch` are crossed without counting). `wasm.br L`
//! counts, from 0, only `wasm.block`, `wasm.loop` and `wasm.if`. The lowering replaces each `scf`
//! construct by `wasm` constructs, outside in: when an exit is converted, every `scf` construct
//! around it already has been, so its count can no longer be read off the IR.
//!
//! So the patterns of [super::control] record, in one [Labels] table per conversion, what each
//! `wasm` construct they create stands for ([Lowered]):
//!
//! * the construct of a lowered `scf.while` ([Lowered::Loop]) counts as one `scf` level: the
//!   `wasm.block` around the loop when some `scf.break` leaves the loop (a break leaves that
//!   block, and a continue goes back to the `wasm.loop` directly inside it), or the `wasm.loop`
//!   itself when none does (a continue goes back to it);
//! * every other construct they create ([Lowered::Inner]: the `wasm.if` of a loop's condition,
//!   the `wasm.if` of an `scf.if`, the blocks and ifs of a switch) counts as none.
//!
//! A `wasm.block`, `wasm.loop` or `wasm.if` that is not in the table was written before the
//! lowering, and counts as one level for both, as the verifiers of `scf` and `wasm` have it. What
//! an exit that names one means is what the `scf` verifier says: a break leaves it with its
//! results, which `wasm.br` does for a `wasm.block` or a `wasm.if` but not for a `wasm.loop` (a
//! branch to a loop goes back to its top), so a break that names a `wasm.loop` written before the
//! lowering is an error; and a continue goes back to the top of a `wasm.loop`, which is a
//! `wasm.br` to it (the `scf` verifier rejects a continue that names a block or an if).
//!
//! [resolve] is the computation, over the labels on the way out from the exit ([Label]), and
//! [level_of] reads those labels off the IR. [check_exit] makes the same computation before the
//! conversion starts, over the labels the lowering will make, so that an exit that does not
//! translate is reported before anything changes.
//!
//! **Branches the frontend wrote.** A `wasm.br` in the input (the branches inside a
//! `wasm.br_table` included) names a `wasm` construct by counting the constructs around it, and
//! the lowering puts new ones (those of an `scf.if`, `scf.while` or `scf.index_switch` it crosses)
//! between it and its target. So before the conversion starts each such branch is resolved, with
//! the dialect's own resolver ([wasm::Br::target]), and the table records the construct it names
//! ([record_branches]); the target is a `wasm` construct, which the conversion keeps. During the
//! conversion a recorded branch whose label no longer names its target is illegal
//! ([names_its_target]), and a pattern gives it the label that does ([level_to_target]),
//! counting every enclosing `wasm.block`, `wasm.loop` and `wasm.if`, those the lowering made
//! included. A branch is offered after every construct around it has been converted, as an exit
//! is.

use alloc::{format, rc::Rc, string::String, vec::Vec};
use core::cell::RefCell;

use runic_dialect_scf as scf;
use runic_dialect_wasm as wasm;
use runic_ir::{
    CallableOpInterface, ExitResolution, Forward, FxHashMap, Operation, OperationRef,
    PropagateControlFlowBreak, RawWalk, RegionExitKind, RegionExitTerminatorOpInterface, Report,
    Spanned, WalkResult, diagnostics::report,
};

/// What a `wasm.block`, `wasm.loop` or `wasm.if` that the lowering created stands for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Lowered {
    /// A construct that implements part of an `scf` construct and counts as no `scf` level: the
    /// `wasm.if` of an `scf.if` or of a loop's condition, the constructs of a switch, the
    /// `wasm.loop` inside the block of a lowered loop.
    Inner,
    /// The construct of a lowered `scf.while`, which counts as one `scf` level. With `wrapped`, it
    /// is the `wasm.block` around the loop: a break leaves it, and a continue goes back to the
    /// `wasm.loop` directly inside it. Without, it is the `wasm.loop` itself, which no break
    /// names.
    Loop {
        /// Whether the construct is the `wasm.block` around the loop.
        wrapped: bool,
    },
}

/// What the conversion of one module knows of its labels: the constructs the lowering created,
/// and the construct each branch the frontend wrote names.
#[derive(Default)]
pub struct LabelTable {
    /// What each construct the lowering created stands for.
    pub constructs: FxHashMap<OperationRef, Lowered>,
    /// The construct each `wasm.br` of the input names, resolved before the conversion started.
    pub branches: FxHashMap<OperationRef, OperationRef>,
}

/// The table of labels, shared by the conversion target and the patterns of the conversion of
/// one module.
pub type Labels = Rc<RefCell<LabelTable>>;

/// A new, empty table.
pub fn new_labels() -> Labels {
    Rc::new(RefCell::new(LabelTable::default()))
}

/// A `wasm` construct that introduces a label.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Construct {
    /// `wasm.block`.
    Block,
    /// `wasm.loop`.
    Loop,
    /// `wasm.if`.
    If,
}

impl Construct {
    /// The construct `op` is, if it is one.
    fn of(op: &Operation) -> Option<Self> {
        if op.is::<wasm::BlockOp>() {
            Some(Self::Block)
        } else if op.is::<wasm::Loop>() {
            Some(Self::Loop)
        } else if op.is::<wasm::If>() {
            Some(Self::If)
        } else {
            None
        }
    }

    /// The name of the operation.
    fn name(self) -> &'static str {
        match self {
            Self::Block => "wasm.block",
            Self::Loop => "wasm.loop",
            Self::If => "wasm.if",
        }
    }
}

/// A label on the way out from an exit: a `wasm` construct that encloses it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Label {
    /// A construct written before the lowering: one level for both counts.
    Frontend(Construct),
    /// A construct the lowering created.
    Lowered(Lowered),
}

/// Why an exit's level does not translate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unresolved {
    /// Fewer than the exit's level of enclosing constructs count as `scf` levels: `found` do.
    NotFound {
        /// The number of constructs that count.
        found: u32,
    },
    /// A break names a `wasm.loop` written before the lowering, which a `wasm.br` cannot leave.
    BreakOfLoop,
    /// A continue names a construct that is not a loop.
    ContinueOfNonLoop(Construct),
    /// A break names a lowered loop that the lowering gave no block to leave.
    Unwrapped,
}

/// The `wasm.br` label of an exit of `kind` that names enclosing loop `levels` (from 1), given
/// the labels that enclose it, innermost first.
///
/// Each label is one `wasm` level; a [Label::Frontend] and a [Lowered::Loop] are also one `scf`
/// level each. The label at which the `scf` levels are used up is the target: for a break, that
/// label; for a continue, the loop of it (the label just inside a wrapped [Lowered::Loop]).
pub fn resolve(chain: &[Label], levels: u32, kind: RegionExitKind) -> Result<u32, Unresolved> {
    if levels == 0 {
        return Err(Unresolved::NotFound { found: 0 });
    }
    let mut remaining = levels;
    for (level, label) in (0u32..).zip(chain.iter()) {
        if *label == Label::Lowered(Lowered::Inner) {
            continue;
        }
        remaining -= 1;
        if remaining != 0 {
            continue;
        }
        return match (label, kind) {
            (Label::Frontend(Construct::Loop), RegionExitKind::Break) => {
                Err(Unresolved::BreakOfLoop)
            }
            (Label::Frontend(Construct::Loop), RegionExitKind::Continue) => Ok(level),
            (Label::Frontend(_), RegionExitKind::Break) => Ok(level),
            (Label::Frontend(construct), RegionExitKind::Continue) => {
                Err(Unresolved::ContinueOfNonLoop(*construct))
            }
            (Label::Lowered(Lowered::Loop { wrapped: true }), RegionExitKind::Break) => Ok(level),
            (Label::Lowered(Lowered::Loop { wrapped: false }), RegionExitKind::Break) => {
                Err(Unresolved::Unwrapped)
            }
            (Label::Lowered(Lowered::Loop { wrapped: true }), RegionExitKind::Continue) => {
                // The loop is directly inside its block, so it is the label just inside it; an
                // exit with no label there is in the block and not in the loop.
                match level.checked_sub(1).and_then(|inner| chain.get(inner as usize)) {
                    Some(Label::Lowered(Lowered::Inner)) => Ok(level - 1),
                    _ => Err(Unresolved::NotFound {
                        found: levels - remaining,
                    }),
                }
            }
            (Label::Lowered(Lowered::Loop { wrapped: false }), RegionExitKind::Continue) => {
                Ok(level)
            }
            // Skipped above: an inner label is never the target.
            (Label::Lowered(Lowered::Inner), _) => Err(Unresolved::NotFound {
                found: levels - remaining,
            }),
        };
    }
    Err(Unresolved::NotFound {
        found: levels - remaining,
    })
}

/// The `wasm.br` label of the exit `op`, an `scf.break` or `scf.continue` of `kind` that names
/// enclosing loop `levels`, from the labels around it in the IR and the table `labels`.
///
/// An error, with a label on `op`, when the level does not translate ([Unresolved]), when the exit
/// would cross an operation that does not forward it, or when an `scf` construct around it is not
/// converted yet.
pub fn level_of(
    op: &Operation,
    levels: i32,
    kind: RegionExitKind,
    labels: &Labels,
) -> Result<i32, Report> {
    let name = op.name();
    let error = |problem: String, label: String| {
        report!(
            message: (format!("`convert-to-wasm` cannot lower this `{name}`: {problem}")),
            labels: [primary(op.span(), label)],
        )
    };
    let Some(levels) = u32::try_from(levels).ok().filter(|levels| *levels >= 1) else {
        return Err(error(
            format!("its level {levels} is not at least 1"),
            format!("this `{name}` names no enclosing loop"),
        ));
    };
    let chain = chain_of(op, labels).map_err(|(problem, label)| error(problem, label))?;
    match resolve(&chain, levels, kind) {
        Ok(level) => i32::try_from(level).map_err(|_| {
            error(String::from("its label is too deep"), format!("this `{name}` is too deep"))
        }),
        Err(unresolved) => Err(unresolved_error(op, levels, unresolved)),
    }
}

/// The error of the exit `op`, which names enclosing loop `levels`, whose level does not
/// translate for the reason `unresolved`.
fn unresolved_error(op: &Operation, levels: u32, unresolved: Unresolved) -> Report {
    let name = op.name();
    let (problem, label) = match unresolved {
        Unresolved::NotFound { found } => (
            format!("it names enclosing loop {levels}, but only {found} enclose it"),
            format!("this `{name}` names no enclosing loop"),
        ),
        Unresolved::BreakOfLoop => (
            String::from(
                "it leaves a `wasm.loop` that was written before the lowering, and a `wasm.br` to \
                 a loop goes back to its top instead",
            ),
            format!("this `{name}` names a `wasm.loop`"),
        ),
        Unresolved::ContinueOfNonLoop(construct) => (
            format!("it names a `{}`, which has no top to go back to", construct.name()),
            format!("this `{name}` names a `{}`", construct.name()),
        ),
        Unresolved::Unwrapped => (
            String::from("the lowering made no block to leave around the loop it names"),
            format!("this `{name}` leaves a loop with no block around it"),
        ),
    };
    report!(
        message: (format!("`convert-to-wasm` cannot lower this `{name}`: {problem}")),
        labels: [primary(op.span(), label)],
    )
}

/// Checks, before the conversion starts, that the exit `op` (an `scf.break` or `scf.continue`
/// of `kind` that names enclosing loop `levels`) translates: [resolve] over the labels the
/// lowering will make around it, with the error [level_of] would give.
///
/// An `scf.while` will be a [Lowered::Loop], wrapped in a block when a break leaves it
/// (`control::is_left_by_a_break`), with its `wasm.loop` just inside; an `scf.if` and an
/// `scf.index_switch` will be inner constructs; a `wasm` construct is written by the frontend.
pub fn check_exit(op: &Operation, levels: i32, kind: RegionExitKind) -> Result<(), Report> {
    let name = op.name();
    let Some(levels) = u32::try_from(levels).ok().filter(|levels| *levels >= 1) else {
        return Err(report!(
            message: (format!(
                "`convert-to-wasm` cannot lower this `{name}`: its level {levels} is not at least 1"
            )),
            labels: [primary(op.span(), format!("this `{name}` names no enclosing loop"))],
        ));
    };
    let mut chain = Vec::new();
    let mut parent = op.parent_op();
    while let Some(parent_ref) = parent {
        let parent_op = parent_ref.borrow();
        if parent_op.implements::<dyn CallableOpInterface>() {
            break;
        }
        if let Some(construct) = Construct::of(&parent_op) {
            chain.push(Label::Frontend(construct));
        } else if parent_op.is::<scf::While>() {
            if super::control::is_left_by_a_break(parent_ref) {
                chain.push(Label::Lowered(Lowered::Inner));
                chain.push(Label::Lowered(Lowered::Loop { wrapped: true }));
            } else {
                chain.push(Label::Lowered(Lowered::Loop { wrapped: false }));
            }
        } else if parent_op.is::<scf::If>() || parent_op.is::<scf::IndexSwitch>() {
            chain.push(Label::Lowered(Lowered::Inner));
        }
        parent = parent_op.parent_op();
    }
    match resolve(&chain, levels, kind) {
        Ok(_) => Ok(()),
        Err(unresolved) => Err(unresolved_error(op, levels, unresolved)),
    }
}

/// Resolves every `wasm.br` under `root` (those inside a `wasm.br_table` included), as the
/// frontend wrote it, and records in `labels` the construct each names.
///
/// An error naming the branch when one does not resolve, with what the dialect's resolver says.
pub fn record_branches(root: OperationRef, labels: &Labels) -> Result<(), Report> {
    let mut table = labels.borrow_mut();
    root.raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult<Report> {
        let op_ref = op.borrow();
        let Some(branch) = op_ref.downcast_ref::<wasm::Br>() else {
            return WalkResult::Continue(());
        };
        match branch.resolve_exit() {
            ExitResolution::Resolved(resolved) => {
                table.branches.insert(op, resolved.target);
                WalkResult::Continue(())
            }
            ExitResolution::Unresolved(defect) => WalkResult::Break(report!(
                message: (format!(
                    "`convert-to-wasm` cannot lower this `wasm.br`: {}",
                    defect.detail
                )),
                labels: [primary(op_ref.span(), "this `wasm.br` names no construct")],
            )),
        }
    })
    .into_result()
}

/// Whether the `wasm.br` `op` names what it should: the construct recorded for it before the
/// conversion ([record_branches]), or anything when it was not in the input.
pub fn names_its_target(op: &Operation, labels: &Labels) -> bool {
    let Some(branch) = op.downcast_ref::<wasm::Br>() else {
        return true;
    };
    let table = labels.borrow();
    match table.branches.get(&op.as_operation_ref()) {
        Some(target) => branch.target() == Some(*target),
        None => true,
    }
}

/// The label that names, from the `wasm.br` `op`, the construct recorded for it before the
/// conversion: the number of `wasm.block`, `wasm.loop` and `wasm.if` between them, those the
/// lowering made included. `None` when nothing was recorded for `op`; an error naming `op` when
/// the construct does not enclose it any more, or an `scf` construct around it is not converted
/// yet.
pub fn level_to_target(op: &Operation, labels: &Labels) -> Result<Option<i32>, Report> {
    let Some(target) = labels.borrow().branches.get(&op.as_operation_ref()).copied() else {
        return Ok(None);
    };
    let error = |label: String| {
        report!(
            message: "`convert-to-wasm` cannot give this `wasm.br` the label of the construct it \
                      named before the lowering",
            labels: [primary(op.span(), label)],
        )
    };
    let mut level = 0i32;
    let mut parent = op.parent_op();
    while let Some(parent_ref) = parent {
        if parent_ref == target {
            return Ok(Some(level));
        }
        let parent_op = parent_ref.borrow();
        if parent_op.implements::<dyn CallableOpInterface>() {
            break;
        }
        if Construct::of(&parent_op).is_some() {
            level += 1;
        } else if parent_op.is::<scf::While>()
            || parent_op.is::<scf::If>()
            || parent_op.is::<scf::IndexSwitch>()
        {
            return Err(error(format!(
                "the `{}` around this `wasm.br` is not converted yet",
                parent_op.name()
            )));
        }
        parent = parent_op.parent_op();
    }
    Err(error(String::from(
        "the construct this `wasm.br` named does not enclose it any more",
    )))
}

/// The labels around `op`, innermost first, up to the function: an error (the problem and the
/// text of the label) when the walk crosses an operation that does not forward breaking control
/// flow, or an `scf` construct (which the lowering replaces before what it holds).
fn chain_of(op: &Operation, labels: &Labels) -> Result<Vec<Label>, (String, String)> {
    let table = labels.borrow();
    let name = op.name();
    let mut chain = Vec::new();
    let mut parent = op.parent_op();
    while let Some(parent_ref) = parent {
        let parent_op = parent_ref.borrow();
        if parent_op.implements::<dyn CallableOpInterface>() {
            break;
        }
        if let Some(construct) = Construct::of(&parent_op) {
            chain.push(match table.constructs.get(&parent_ref) {
                Some(lowered) => Label::Lowered(*lowered),
                None => Label::Frontend(construct),
            });
        } else if parent_op.is::<scf::While>()
            || parent_op.is::<scf::If>()
            || parent_op.is::<scf::IndexSwitch>()
        {
            return Err((
                format!("the `{}` around it is not converted yet", parent_op.name()),
                format!("this `{name}` is converted before what encloses it"),
            ));
        }
        if !parent_op.implements::<dyn PropagateControlFlowBreak>() {
            return Err((
                format!(
                    "it would have to cross `{}`, which does not forward breaking control flow",
                    parent_op.name()
                ),
                format!("this `{name}` crosses `{}`", parent_op.name()),
            ));
        }
        parent = parent_op.parent_op();
    }
    Ok(chain)
}
