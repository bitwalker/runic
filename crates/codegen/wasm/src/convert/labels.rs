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
//! [level_of] reads those labels off the IR.

use alloc::{format, rc::Rc, string::String, vec::Vec};
use core::cell::RefCell;

use runic_dialect_scf as scf;
use runic_dialect_wasm as wasm;
use runic_ir::{
    CallableOpInterface, FxHashMap, Operation, OperationRef, PropagateControlFlowBreak,
    RegionExitKind, Report, Spanned, diagnostics::report,
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

/// The table of the constructs the lowering created, shared by the patterns of the conversion of
/// one module.
pub type Labels = Rc<RefCell<FxHashMap<OperationRef, Lowered>>>;

/// A new, empty table.
pub fn new_labels() -> Labels {
    Rc::new(RefCell::new(FxHashMap::default()))
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
        Err(Unresolved::NotFound { found }) => Err(error(
            format!("it names enclosing loop {levels}, but only {found} enclose it"),
            format!("this `{name}` names no enclosing loop"),
        )),
        Err(Unresolved::BreakOfLoop) => Err(error(
            String::from(
                "it leaves a `wasm.loop` that was written before the lowering, and a `wasm.br` to \
                 a loop goes back to its top instead",
            ),
            format!("this `{name}` names a `wasm.loop`"),
        )),
        Err(Unresolved::ContinueOfNonLoop(construct)) => Err(error(
            format!("it names a `{}`, which has no top to go back to", construct.name()),
            format!("this `{name}` names a `{}`", construct.name()),
        )),
        Err(Unresolved::Unwrapped) => Err(error(
            String::from("the lowering made no block to leave around the loop it names"),
            format!("this `{name}` leaves a loop with no block around it"),
        )),
    }
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
            chain.push(match table.get(&parent_ref) {
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
