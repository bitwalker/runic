//! The locals of a function: the numbering of its operations, the live range of each value that
//! needs a local, and the assignment of locals to those values.
//!
//! Wasm numbers a function's locals from its parameters: the parameters (the arguments of the
//! body's entry block) are `0..n`, the function's declared locals (`locals(..)`, which
//! `builtin.load_local N` and `builtin.store_local N` name) are `n..n+k`, and the locals of the
//! emitter follow. A value needs a local of the emitter when it is a used result of a root of a
//! plan, or a teed value ([BlockPlan::needs_local](super::plan::BlockPlan::needs_local)).
//!
//! The function's operations are numbered in emission order ([numbering]). A value's live range
//! runs from the number of the operation that defines it to the number of its last use. With
//! reuse ([BodyConfig::reuse_locals](super::body::BodyConfig::reuse_locals)), a local of the
//! emitter is given to a later value of the same type whose definition comes strictly after the
//! last use of the value that holds it; parameters and declared locals are never reused. Without
//! reuse, each value has its own local.

use alloc::vec::Vec;

use runic_ir::{FxHashMap, OperationRef, RegionRef, ValueRef};

use super::desc::ValType;

/// The number of each operation of a function body in emission order: a pre-order walk, so an
/// operation with regions comes before the operations nested in it.
pub fn numbering(body: RegionRef) -> FxHashMap<OperationRef, u32> {
    let mut numbers = FxHashMap::default();
    // A stack of the operations still to number: for each region entered, its operations in
    // reverse, so that the next is the last.
    let mut pending: Vec<Vec<OperationRef>> = Vec::new();
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
    while let Some(ops) = pending.last_mut() {
        let Some(op) = ops.pop() else {
            pending.pop();
            continue;
        };
        numbers.insert(op, next);
        next += 1;
        let nested = op.borrow().regions().iter().map(|r| r.as_region_ref()).collect::<Vec<_>>();
        for region in nested.into_iter().rev() {
            push_region(region, &mut pending);
        }
    }
    numbers
}

/// The live range of a value that needs a local.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRange {
    /// The value.
    pub value: ValueRef,
    /// Its type.
    pub ty: ValType,
    /// The number of the operation that defines it.
    pub def: u32,
    /// The number of its last use (the definition's, when it has no use).
    pub last_use: u32,
}

/// The point at which the use of a value by `user` ends its live range.
fn use_point(numbers: &FxHashMap<OperationRef, u32>, user: OperationRef) -> u32 {
    // TASK 3: a value defined outside a `wasm.loop` and used inside it lives to the end of the
    // loop. The rule goes here: when `user` is nested in a loop that does not hold the value's
    // definition, the use ends at the number of the loop's last operation instead.
    numbers.get(&user).copied().unwrap_or(0)
}

/// The live range of `value` of type `ty`, defined by the operation numbered `def`.
pub fn live_range(
    numbers: &FxHashMap<OperationRef, u32>,
    value: ValueRef,
    ty: ValType,
    def: u32,
) -> LiveRange {
    let last_use = value
        .borrow()
        .iter_uses()
        .map(|use_| use_point(numbers, use_.owner))
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
