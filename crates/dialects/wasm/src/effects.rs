//! The resources the state-touching operations declare their effects on.
//!
//! An effect names a resource, and two effects can interfere only when their resources are not
//! disjoint: one is the other or one of its ancestors (see [Resource]). The generic passes
//! (`cse`, `dce`, `sink`, ..) use exactly that question, so the resources decide what they may
//! merge, move or remove.
//!
//! There is one resource for each kind of state a Wasm instruction touches: linear memory,
//! globals and tables. They are types, not instances: every memory of a module is
//! [LinearMemoryResource], every global [GlobalsResource] and every table [TablesResource], so a
//! write to one table is ordered against a read of another (the dialect does not tell the
//! symbols apart; a pass that wants to would need a resource for each).

use runic_dialect_mem::MemoryResource;
use runic_ir::effects::Resource;

/// The linear memory that `wasm.load` reads and `wasm.store` and the bulk operations write.
///
/// It is a *child of the `mem` dialect's [MemoryResource]*, not a sibling of it: a `wasm.store` and
/// a `mem.load` in one function (a pointer made with `builtin.global_symbol`, say) address the
/// same bytes, and the effect system orders two effects whose resources are one and the other's
/// ancestor. So a `mem.store` is ordered against a `wasm.load` and the reverse, and so is an effect
/// on the default resource (a call with unknown effects); an effect on [GlobalsResource] or
/// [TablesResource] is not. The size of the memory (`wasm.memory_size`, `wasm.memory_grow`) is part
/// of this resource: growing the memory changes which addresses are valid.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LinearMemoryResource;

impl Resource for LinearMemoryResource {
    fn name(&self) -> &'static str {
        "wasm.memory"
    }

    fn parent(&self) -> Option<&'static dyn Resource> {
        Some(&MemoryResource)
    }
}

/// The globals that `wasm.global_get` reads and `wasm.global_set` writes: a child of the default
/// resource, so it is ordered against every effect on the default resource, and against no effect
/// on another of its children (memory, tables).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct GlobalsResource;

impl Resource for GlobalsResource {
    fn name(&self) -> &'static str {
        "wasm.globals"
    }
}

/// The tables that the `wasm.table_*` operations read and write, and the size of each: a child of
/// the default resource, as [GlobalsResource] is.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TablesResource;

impl Resource for TablesResource {
    fn name(&self) -> &'static str {
        "wasm.tables"
    }
}
