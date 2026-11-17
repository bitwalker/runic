//! Where each value of a block waits between its definition and its uses: on the Wasm stack, in
//! a local, or written again at each use (decision 5 of the plan).
//!
//! Code is emitted in the order of the IR; nothing is moved, so no reasoning about effects is
//! needed. The plan of a block is a function of the IR alone:
//!
//! - The operations of [REMATERIALIZED](super::instr::REMATERIALIZED) (`wasm.const`,
//!   `wasm.constf`, `wasm.ref_null`, `wasm.data_addr`, `wasm.func_index`) are written again at
//!   each use and emit nothing at their own position; they never take a local. Below, "the block's
//!   operations" excludes them.
//! - The block's operations form a forest of expression trees, built backwards from the last
//!   operation. Each operation states the operands it takes from the stack in the order they are
//!   pushed ([stack_operands]); they are visited from the last to the first. An operand whose
//!   value is the single result of the operation right before the current tree, that occurs once
//!   among the operands of its user, and whose other uses (if any) are not operands of an
//!   operation on the chain of trees being built, is left **on the stack** by that operation,
//!   which is claimed and whose own tree is built in turn; when the value has other uses it is
//!   also saved with `local.tee`. Any other operand is **fetched** from its value's local.
//! - An operation no tree claims is a **root**: after its instruction its results are set into
//!   locals (from the last to the first) or dropped when unused.
//!
//! The invariants the emission relies on:
//!
//! 1. A tree covers a contiguous run of operations ending at its root, and its post-order is their
//!    order in the IR; so emitting the roots in order emits every operation in the IR's order.
//! 2. Between two roots the stack is empty (the body encoder checks it in debug builds).
//! 3. A fetch is emitted at its operand's position, before the subtrees of the operands pushed
//!    after it. A value read that way is never defined in such a later subtree: a value that is
//!    an operand of an operation on the chain being built is not teed inside the chain, so its
//!    local is set before its first fetch.

use alloc::{format, string::String, vec, vec::Vec};
use core::fmt;

use runic_ir::{BlockRef, FxHashMap, OperationRef, Usable, ValueRef};

use super::instr::{is_rematerialized, name_of, stack_operands};

/// How a stack operand of an operation is obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Left on the stack by the claimed operation at this position of the plan, whose tree is
    /// emitted here.
    Stack(usize),
    /// Fetched from the value's local with `local.get`.
    Local,
    /// Written here: the instruction of the rematerialized operation at this position of the
    /// plan, which defines the value. A value of a rematerialized operation of another block has
    /// the position `None`.
    Constant(Option<usize>),
}

/// One stack operand of an operation: its value and how it is obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    /// The value.
    pub value: ValueRef,
    /// How it is obtained.
    pub source: Source,
}

/// What an operation is in the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A root of a tree: emitted when the walk reaches it, with the stack empty before it; its
    /// results are then set into locals or dropped.
    Root,
    /// Claimed by the tree of the operation after it: its one result is left on the stack, and
    /// with `tee`, also saved in its local with `local.tee`.
    Claimed {
        /// Whether the result is also saved in its local.
        tee: bool,
    },
    /// Written again at each use; nothing is emitted at its position.
    Rematerialized,
}

/// The plan of one operation.
#[derive(Clone, Debug)]
pub struct Step {
    /// The operation.
    pub op: OperationRef,
    /// What it is.
    pub role: Role,
    /// Its stack operands, in the order they are pushed. Empty for a rematerialized operation.
    pub operands: Vec<Operand>,
}

/// The plan of a block: one [Step] per operation, in the order of the IR.
#[derive(Clone, Debug, Default)]
pub struct BlockPlan {
    /// The steps, in the order of the IR.
    pub steps: Vec<Step>,
    /// The block's arguments, in order; for a value's name in [fmt::Display].
    arguments: Vec<ValueRef>,
}

impl BlockPlan {
    /// The positions of the roots, in the order of the IR.
    pub fn roots(&self) -> impl Iterator<Item = usize> + '_ {
        self.steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| matches!(step.role, Role::Root).then_some(index))
    }

    /// Whether the value `value`, a result of the step `index`, needs a local: a used result of
    /// a root, or a teed value.
    pub fn needs_local(&self, index: usize, value: ValueRef) -> bool {
        match self.steps[index].role {
            Role::Root => value.borrow().is_used(),
            Role::Claimed { tee } => tee,
            Role::Rematerialized => false,
        }
    }

    /// The name of `value` in the text of the plan: `arg0` for an argument of the block, `%3` for
    /// the result of the operation at position 3 (`%3.1` for its second result), `outer` for any
    /// other value.
    fn value_name(&self, value: ValueRef) -> String {
        if let Some(index) = self.arguments.iter().position(|argument| *argument == value) {
            return format!("arg{index}");
        }
        for (position, step) in self.steps.iter().enumerate() {
            let op = step.op.borrow();
            let results = op.results();
            if let Some(index) =
                results.iter().position(|result| result.borrow().as_value_ref() == value)
            {
                return if results.len() == 1 {
                    format!("%{position}")
                } else {
                    format!("%{position}.{index}")
                };
            }
        }
        String::from("outer")
    }
}

/// The text of a plan: one line per operation, its position, its name, its stack operands in the
/// order they are pushed (`stack %2`, `get arg0`, `const %1`), and what it is: `written again`,
/// `on the stack` (`on the stack, tee`), or `root` followed by what happens to its results
/// (`set`, `drop`, one per result from the first; nothing for none).
impl fmt::Display for BlockPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, step) in self.steps.iter().enumerate() {
            write!(f, "{position} {}", name_of(&step.op.borrow()))?;
            if !step.operands.is_empty() {
                let operands = step
                    .operands
                    .iter()
                    .map(|operand| {
                        let name = self.value_name(operand.value);
                        match operand.source {
                            Source::Stack(_) => format!("stack {name}"),
                            Source::Local => format!("get {name}"),
                            Source::Constant(_) => format!("const {name}"),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, " [{operands}]")?;
            }
            match step.role {
                Role::Rematerialized => f.write_str(": written again")?,
                Role::Claimed { tee: false } => f.write_str(": on the stack")?,
                Role::Claimed { tee: true } => f.write_str(": on the stack, tee")?,
                Role::Root => {
                    f.write_str(": root")?;
                    let op = step.op.borrow();
                    let outcomes = op
                        .results()
                        .iter()
                        .map(|result| {
                            if result.borrow().is_used() {
                                "set"
                            } else {
                                "drop"
                            }
                        })
                        .collect::<Vec<_>>();
                    if !outcomes.is_empty() {
                        write!(f, ", {}", outcomes.join(" "))?;
                    }
                }
            }
            f.write_str("\n")?;
        }
        Ok(())
    }
}

/// A tree being built: the position of its operation among the block's operations, its stack
/// operands, how many of them are still to be visited, and the cursor (the position among the
/// block's operations right before the tree's first operation, `None` before the first).
struct Frame {
    position: usize,
    operands: Vec<ValueRef>,
    remaining: usize,
    cursor: Option<usize>,
}

/// The plan of `block` (see the [module docs](self)).
pub fn plan_block(block: BlockRef) -> BlockPlan {
    let (ops, arguments) = {
        let block = block.borrow();
        let ops = block.body().iter().map(|op| op.as_operation_ref()).collect::<Vec<_>>();
        (ops, block.argument_values().collect::<Vec<_>>())
    };
    let mut steps = ops
        .iter()
        .map(|op| Step {
            op: *op,
            role: if is_rematerialized(&op.borrow()) {
                Role::Rematerialized
            } else {
                Role::Root
            },
            operands: Vec::new(),
        })
        .collect::<Vec<_>>();
    // The rematerialized operations by the value they define, and the block's operations (the
    // others) by position among them.
    let mut constants = FxHashMap::default();
    let mut order = Vec::new();
    for (index, step) in steps.iter().enumerate() {
        if matches!(step.role, Role::Rematerialized) {
            for result in step.op.borrow().results().iter() {
                constants.insert(result.borrow().as_value_ref(), index);
            }
        } else {
            order.push(index);
        }
    }
    let mut claimed = vec![false; order.len()];
    for root in (0..order.len()).rev() {
        if !claimed[root] {
            build(root, &order, &mut steps, &constants, &mut claimed);
        }
    }
    BlockPlan { steps, arguments }
}

/// Whether `value` is defined by a rematerialized operation, here or in another block.
fn defined_by_constant(value: ValueRef) -> bool {
    value
        .borrow()
        .get_defining_op()
        .is_some_and(|op| is_rematerialized(&op.borrow()))
}

/// Builds the tree of the block's operation at `root` (a position in `order`), claiming the
/// operations it covers. Iterative, so that a long chain does not exhaust the native stack.
fn build(
    root: usize,
    order: &[usize],
    steps: &mut [Step],
    constants: &FxHashMap<ValueRef, usize>,
    claimed: &mut [bool],
) {
    let new_frame = |position: usize, steps: &[Step]| {
        let operands = stack_operands(&steps[order[position]].op.borrow());
        Frame {
            position,
            remaining: operands.len(),
            operands,
            cursor: position.checked_sub(1),
        }
    };
    let mut chain = vec![new_frame(root, steps)];
    while let Some(top) = chain.len().checked_sub(1) {
        if chain[top].remaining == 0 {
            // The tree is complete: its first operation is right after the cursor, which the
            // tree it is an operand of continues from.
            let done = chain.pop().expect("a frame");
            let index = order[done.position];
            // The operands were recorded from the last to the first.
            steps[index].operands.reverse();
            debug_assert_eq!(steps[index].operands.len(), done.operands.len());
            if let Some(parent) = chain.last_mut() {
                parent.cursor = done.cursor;
            }
            continue;
        }
        chain[top].remaining -= 1;
        let value = chain[top].operands[chain[top].remaining];
        let user = order[chain[top].position];
        let cursor = chain[top].cursor;
        let source = if let Some(constant) = constants.get(&value) {
            Source::Constant(Some(*constant))
        } else if defined_by_constant(value) {
            Source::Constant(None)
        } else if let Some(candidate) = cursor
            && !claimed[candidate]
            && let Some(tee) =
                stays_on_stack(value, &steps[order[candidate]], user, steps, &chain, order)
        {
            claimed[candidate] = true;
            let index = order[candidate];
            steps[index].role = Role::Claimed { tee };
            steps[user].operands.push(Operand {
                value,
                source: Source::Stack(index),
            });
            let child = new_frame(candidate, steps);
            chain.push(child);
            continue;
        } else {
            Source::Local
        };
        steps[user].operands.push(Operand { value, source });
    }
}

/// Whether the operand `value` of the operation at `user` (a position of the plan) can be left on
/// the stack by `candidate`, the operation right before the tree being built, and if so whether
/// it is also teed: `Some(tee)` when it can.
///
/// It can when `candidate` has exactly one result, which is `value`, `value` occurs once among
/// the operands of `user`, and no other use of `value` is an operand of an operation on the chain
/// of trees being built (the user included). It is teed when it has other uses.
fn stays_on_stack(
    value: ValueRef,
    candidate: &Step,
    user: usize,
    steps: &[Step],
    chain: &[Frame],
    order: &[usize],
) -> Option<bool> {
    {
        let op = candidate.op.borrow();
        let results = op.results();
        if results.len() != 1 || results.iter().next()?.borrow().as_value_ref() != value {
            return None;
        }
    }
    let user_op = steps[user].op;
    let occurrences = user_op
        .borrow()
        .operands()
        .iter()
        .filter(|operand| operand.borrow().as_value_ref() == value)
        .count();
    if occurrences != 1 {
        return None;
    }
    let value = value.borrow();
    let mut others = 0;
    for use_ in value.iter_uses() {
        if use_.owner == user_op {
            continue;
        }
        others += 1;
        if chain.iter().any(|frame| steps[order[frame.position]].op == use_.owner) {
            return None;
        }
    }
    Some(others > 0)
}
