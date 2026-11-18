//! A `builtin.global_variable`'s initializer as the bytes of a data entry, with fix-ups (decision
//! 4 of the plan).
//!
//! The initializer is generic code that nothing lowers, ending in `builtin.yield %v`. [evaluate]
//! gives it to the `mem` dialect's evaluator ([evaluate_initializer]) under the data layout in
//! scope of the global ([DataLayout::closest]), which places `%v` (a constant, a zero, the address
//! of a global variable plus a constant, a function, or a struct or an array of those) in the bytes
//! of the global's type. Every error about what an initializer holds is the evaluator's.
//!
//! What the evaluator gives becomes a data entry:
//!
//! - its bytes, size and alignment are the entry's. Bytes that are all zero, with no relocation,
//!   are the empty bytes of the evaluator ([Initializer::is_zero]): the entry has no bytes of its
//!   own and gets no segment (linear memory starts zeroed), however large it is;
//! - each relocation is a fix-up at its offset: the address of a global variable is a
//!   [FixupKind::MemoryAddr] of the data entry of that global with the relocation's addend, and a
//!   function is a [FixupKind::TableSlot] of that function. Each target is found through the
//!   unit's linking, so a declaration that links to a definition in another module names the
//!   definition's entry, and the function an alias leads to is the function. A fix-up in data is
//!   four bytes: a relocation of any other size is an error naming the global. An addend is
//!   added modulo 2^32, as the arithmetic of `mem.gep` on a 32-bit address wraps.

use alloc::{format, string::String, vec::Vec};

use runic_dialect_mem::{Initializer, Relocation, Target, evaluate_initializer};
use runic_ir::{
    DataLayout, Operation, OperationRef, Report, Spanned, diagnostics::report,
    dialects::builtin::GlobalVariable,
};

use super::desc::{DATA_PLACEHOLDER, DataId, Fixup, FixupKind, FuncId};

/// The bytes of a global variable, their size, how they are aligned, and the fix-ups in them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DataInit {
    /// The size of the global, in bytes: `size_of` its type.
    pub size: u64,
    /// The bytes: empty when every byte is zero and there is no fix-up, otherwise exactly `size`
    /// bytes, zero under each fix-up.
    pub bytes: Vec<u8>,
    /// The alignment of the global's address, in bytes: `align_of` its type.
    pub align: u32,
    /// The fix-ups in `bytes`, in ascending order of offset, each over four zero bytes.
    pub fixups: Vec<Fixup>,
}

/// The entities of the unit that the relocations of an initializer name.
pub trait DataTargets {
    /// The data entry of the global variable `target` (a definition, or a declaration that links
    /// to one or is an undefined data symbol), whose address the initializer of `global` takes.
    fn data_entry(&mut self, global: &Operation, target: OperationRef) -> Result<DataId, Report>;
    /// The function `target` (what a `builtin.function_ref` resolves to, aliases followed), whose
    /// table slot the initializer of `global` holds.
    fn function(&mut self, global: &Operation, target: OperationRef) -> Result<FuncId, Report>;
}

/// The bytes of `global`, a `builtin.global_variable` definition, with the entities its
/// relocations name found through `targets` (see the module docs).
pub fn evaluate(global: &Operation, targets: &mut dyn DataTargets) -> Result<DataInit, Report> {
    let Some(variable) = global.downcast_ref::<GlobalVariable>() else {
        return Err(cannot_lay_out(global, String::from("it is not a `builtin.global_variable`")));
    };
    let Some(layout) = DataLayout::closest(global) else {
        return Err(cannot_lay_out(global, String::from("no data layout is in scope of it")));
    };
    let initializer = evaluate_initializer(variable, &layout)?;
    let Initializer {
        size,
        align,
        bytes,
        relocations,
    } = initializer;
    let Ok(align) = u32::try_from(align) else {
        return Err(cannot_lay_out(
            global,
            format!("its alignment of {align} bytes is beyond a 32-bit memory"),
        ));
    };
    let mut fixups = Vec::with_capacity(relocations.len());
    for relocation in &relocations {
        fixups.push(fixup(global, relocation, targets)?);
    }
    Ok(DataInit {
        size,
        bytes,
        align,
        fixups,
    })
}

/// The fix-up of `relocation`, a relocation in the initializer of `global`.
fn fixup(
    global: &Operation,
    relocation: &Relocation,
    targets: &mut dyn DataTargets,
) -> Result<Fixup, Report> {
    let what = match &relocation.target {
        Target::Data(_) => "an address",
        Target::Function(_) => "a function value",
    };
    if usize::from(relocation.size) != DATA_PLACEHOLDER.len() {
        return Err(report!(
            message: (format!(
                "the global variable `{}` holds {what} of {} bytes, which the backend cannot place",
                name(global),
                relocation.size
            )),
            labels: [primary(
                global.span(),
                format!(
                    "its initializer places {what} of {} bytes at offset {}; the backend writes \
                     an address or a table slot in data as 4 bytes, as the data layout of wasm32 \
                     gives them",
                    relocation.size, relocation.offset
                )
            )],
        ));
    }
    let Ok(offset) = u32::try_from(relocation.offset) else {
        return Err(cannot_lay_out(
            global,
            format!(
                "its initializer places {what} at offset {}, beyond a 32-bit memory",
                relocation.offset
            ),
        ));
    };
    let (kind, addend) = match &relocation.target {
        Target::Data(target) => {
            let id = targets.data_entry(global, target.as_operation_ref())?;
            // Addresses are 32 bits: the addend wraps as `mem.gep`'s arithmetic does.
            (FixupKind::MemoryAddr(id), relocation.addend as i32)
        }
        Target::Function(target) => (FixupKind::TableSlot(targets.function(global, *target)?), 0),
    };
    Ok(Fixup {
        offset,
        kind,
        addend,
    })
}

/// The name of the global variable `global` for a message: `@g`.
fn name(global: &Operation) -> String {
    match global.downcast_ref::<GlobalVariable>() {
        Some(variable) => format!("@{}", variable.get_name().as_str()),
        None => format!("`{}`", global.name()),
    }
}

/// The error of a global variable `global` that cannot be laid out, with `label`.
fn cannot_lay_out(global: &Operation, label: String) -> Report {
    report!(
        message: (format!("the global variable `{}` cannot be laid out in linear memory", name(global))),
        labels: [primary(global.span(), label)],
    )
}
