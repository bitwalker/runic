//! The locals of a function: the numbering of its operations, the live range of each value that
//! needs a local, and the assignment of locals to those values.
//!
//! Wasm numbers a function's locals from its parameters: the parameters (the arguments of the
//! body's entry block) are `0..n`, the function's declared locals (`locals(..)`, which
//! `builtin.load_local N` and `builtin.store_local N` name) are `n..n+k`, and the locals of the
//! emitter follow. A value needs a local of the emitter when it is a used result of a root of a
//! plan, or a teed value ([BlockPlan::needs_local](super::plan::BlockPlan::needs_local)).
//!
//! The function's operations are numbered in emission order ([numbering]): a pre-order walk, so a
//! construct comes before the operations nested in it, which follow it contiguously. A value's
//! live range runs from the number of the operation that defines it (for an argument of a block of
//! a construct, the region's entry: the number of the block's first operation, as the arguments are
//! set right before it) to the number of its last use, with one rule for loops:
//!
//! **A value defined outside a `wasm.loop` and used inside it is live to the loop's last
//! position** (the number of the last operation nested in it, at any depth); with loops nested in
//! one another, to the last position of the outermost loop that holds the use but not the
//! definition. A branch back to the loop runs the use again, after every position of the loop:
//! without the rule, a value defined later in the loop could take the local while the first value
//! is still to be read on the next iteration.
//!
//! That is enough. A value defined *inside* a loop (an argument of the loop's own body included)
//! needs no extension: its definition dominates its uses, so on every iteration it is written
//! again before any of them reads it, and what another value held in its local meanwhile is never
//! read. The numbering is otherwise conservative where the emission writes a local later than the
//! number says (the results of a construct are set after its `end`): a later write cannot
//! overwrite a value that is still to be read, and an earlier start only keeps two values apart.
//! Reads are never later than the number of their use: an operand is fetched while its user's tree
//! is written, before the user's instruction. An argument's local may be one whose value was last
//! used by the construct itself (an input, pushed before the construct's instruction) or, for the
//! second region of an `if`, in the first region, which does not run when the second does.
//!
//! With reuse ([BodyConfig::reuse_locals](super::body::BodyConfig::reuse_locals)), a local of the
//! emitter is given to a later value of the same type whose definition comes strictly after the
//! last use of the value that holds it; parameters and declared locals are never reused. Without
//! reuse, each value has its own local.

use alloc::vec::Vec;

use runic_ir::{FxHashMap, OperationRef, RegionRef, ValueRef};

use super::desc::ValType;

/// The numbering of a function body's operations in emission order: a pre-order walk, so an
/// operation with regions comes before the operations nested in it, which take the numbers right
/// after it.
#[derive(Clone, Debug, Default)]
pub struct Numbering {
    /// The number of each operation.
    numbers: FxHashMap<OperationRef, u32>,
    /// The number of the last operation nested in each operation, at any depth (its own number
    /// when it holds none).
    last: FxHashMap<OperationRef, u32>,
}

impl Numbering {
    /// The number of `op`, if it is an operation of the body.
    pub fn of(&self, op: OperationRef) -> Option<u32> {
        self.numbers.get(&op).copied()
    }

    /// The number of the last operation nested in `op` at any depth, or `op`'s own number when it
    /// holds none; `None` when `op` is not an operation of the body.
    pub fn last_nested(&self, op: OperationRef) -> Option<u32> {
        self.last.get(&op).copied()
    }
}

/// The numbering of the operations of `body`, a function's body (see [Numbering]).
pub fn numbering(body: RegionRef) -> Numbering {
    let mut numbering = Numbering::default();
    // A stack of the operations still to number: for each region entered, its operations in
    // reverse, so that the next is the last; and the operations whose nested operations are being
    // numbered, with the depth of `pending` at which theirs end.
    let mut pending: Vec<Vec<OperationRef>> = Vec::new();
    let mut open: Vec<(OperationRef, usize)> = Vec::new();
    let push_region = |region: RegionRef, pending: &mut Vec<Vec<OperationRef>>| {
        let mut ops = Vec::new();
        for block in region.borrow().body().iter() {
            ops.extend(block.body().iter().map(|op| op.as_operation_ref()));
        }
        ops.reverse();
        pending.push(ops);
    };
    push_region(body, &mut pending);
    let mut next = 0u32;
    loop {
        // Close the operations all of whose nested operations are numbered.
        while let Some((op, depth)) = open.last().copied()
            && pending.len() <= depth
        {
            numbering.last.insert(op, next - 1);
            open.pop();
        }
        let Some(ops) = pending.last_mut() else {
            break;
        };
        let Some(op) = ops.pop() else {
            pending.pop();
            continue;
        };
        numbering.numbers.insert(op, next);
        next += 1;
        let nested = op.borrow().regions().iter().map(|r| r.as_region_ref()).collect::<Vec<_>>();
        if nested.is_empty() {
            numbering.last.insert(op, next - 1);
            continue;
        }
        open.push((op, pending.len()));
        for region in nested.into_iter().rev() {
            push_region(region, &mut pending);
        }
    }
    numbering
}

/// The live range of a value that needs a local.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRange {
    /// The value.
    pub value: ValueRef,
    /// Its type.
    pub ty: ValType,
    /// The number of the operation that defines it (for an argument of a block of a construct,
    /// the number of the block's first operation).
    pub def: u32,
    /// The number of its last use (the definition's, when it has no use), extended by the rule
    /// for loops (see the [module docs](self)).
    pub last_use: u32,
}

/// The point at which the use of a value defined in `region` by `user` ends its live range: the
/// number of `user`, or, when `user` is nested in `wasm.loop`s that `region` is not in, the last
/// position of the outermost of them.
fn use_point(numbering: &Numbering, region: Option<RegionRef>, user: OperationRef) -> u32 {
    let mut point = numbering.of(user).unwrap_or(0);
    let mut op = user;
    while let Some(parent_region) = op.parent_region()
        && Some(parent_region) != region
        && let Some(parent) = parent_region.borrow().parent()
    {
        if parent.borrow().is::<runic_dialect_wasm::Loop>()
            && let Some(last) = numbering.last_nested(parent)
        {
            point = point.max(last);
        }
        op = parent;
    }
    point
}

/// The live range of `value` of type `ty`, defined at the number `def`.
pub fn live_range(numbering: &Numbering, value: ValueRef, ty: ValType, def: u32) -> LiveRange {
    let value_ref = value.borrow();
    let region = value_ref.parent_region();
    let last_use = value_ref
        .iter_uses()
        .map(|use_| use_point(numbering, region, use_.owner))
        .fold(def, u32::max);
    LiveRange {
        value,
        ty,
        def,
        last_use,
    }
}

/// The locals of a function: where each value that has one lives.
#[derive(Clone, Debug, Default)]
pub struct Locals {
    /// The number of parameters.
    params: u32,
    /// The types of the declared locals.
    declared: Vec<ValType>,
    /// The types of the locals of the emitter, in order.
    emitter: Vec<ValType>,
    /// The local of each value that has one: the parameters and the values given a local.
    by_value: FxHashMap<ValueRef, u32>,
}

impl Locals {
    /// The local index of `value`, if it has a local.
    pub fn of(&self, value: ValueRef) -> Option<u32> {
        self.by_value.get(&value).copied()
    }

    /// The number of parameters, which is the index of the first declared local.
    pub fn params(&self) -> u32 {
        self.params
    }

    /// The local index of declared local `index`.
    pub fn declared(&self, index: u32) -> u32 {
        self.params + index
    }

    /// The types of the locals of the emitter, in order; the first is the local after the last
    /// declared one.
    pub fn emitter(&self) -> &[ValType] {
        &self.emitter
    }

    /// The types of every local beyond the parameters: the declared ones, then the emitter's.
    pub fn types(&self) -> Vec<ValType> {
        self.declared.iter().chain(&self.emitter).copied().collect()
    }
}

/// Gives each of `ranges` a local, after the parameters `params` and the declared locals of types
/// `declared`; with `reuse`, a local of the emitter is shared by values of one type whose ranges
/// do not overlap (one's definition strictly after the other's last use).
///
/// The ranges are taken in the order of their definitions (ties in the order given), and each
/// gets the lowest-numbered free local of its type, or a new one.
pub fn assign(
    params: &[ValueRef],
    declared: &[ValType],
    ranges: &[LiveRange],
    reuse: bool,
) -> Locals {
    let mut locals = Locals {
        params: params.len() as u32,
        declared: declared.to_vec(),
        ..Locals::default()
    };
    for (index, param) in params.iter().enumerate() {
        locals.by_value.insert(*param, index as u32);
    }
    let first = locals.params + declared.len() as u32;
    let mut ordered = ranges.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|range| range.def);
    // For each local of the emitter, the last use of the value that holds it.
    let mut free_after: Vec<u32> = Vec::new();
    for range in ordered {
        let shared = reuse
            .then(|| {
                (0..locals.emitter.len()).find(|local| {
                    locals.emitter[*local] == range.ty && free_after[*local] < range.def
                })
            })
            .flatten();
        let local = match shared {
            Some(local) => {
                free_after[local] = range.last_use;
                local
            }
            None => {
                locals.emitter.push(range.ty);
                free_after.push(range.last_use);
                locals.emitter.len() - 1
            }
        };
        locals.by_value.insert(range.value, first + local as u32);
    }
    locals
}
