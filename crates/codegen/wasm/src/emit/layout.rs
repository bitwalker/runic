//! The layout of linear memory and of the indirect-function table (decision 2 of the plan).
//!
//! THE SEAM OF TASK 5. The layout places the shadow stack and the data, defines the well-known
//! symbols, gives table slots and resolves the linker-defined data symbols; none of that is built
//! yet. What is built: an undefined data symbol is the error that a core module cannot import
//! data; and a unit that needs no layout (no data entry, no `builtin.segment`, no table slot) is
//! given an empty [Resolved]. Any other unit is the error that the layout is not built yet,
//! naming the operation that needs it.

use alloc::format;

use runic_ir::{Operation, Report, Spanned, diagnostics::report};

use super::{
    EmitOptions,
    collect::{Unit, cannot_import_data},
    desc::{Fixup, FixupKind},
    write::Resolved,
};

/// Lays out the unit: the addresses of its data and the table slots of its functions, which the
/// writer needs. See the module docs for what is built.
pub fn layout(unit: &mut Unit, options: &EmitOptions) -> Result<Resolved, Report> {
    let _ = options.stack_size;
    if let Some(data) = unit.undefined_data().first()
        && let Some(op) = unit.data_op(*data)
    {
        let name = unit.desc.data()[data.index() as usize].name.clone();
        return Err(cannot_import_data(&op.borrow(), &name));
    }
    if let Some(segment) = unit.segments().first() {
        return Err(not_built(
            &segment.borrow(),
            "this is data at a fixed address in linear memory, which the layout places",
        ));
    }
    if let Some(op) = unit.data_op(super::desc::DataId::new(0)) {
        return Err(not_built(
            &op.borrow(),
            "this is data in linear memory, which the layout places",
        ));
    }
    let slot = |fixups: &[Fixup]| {
        fixups.iter().find_map(|fixup| match fixup.kind {
            FixupKind::TableSlot(function) => Some(function),
            _ => None,
        })
    };
    for (index, function) in unit.desc.functions().iter().enumerate() {
        if let Some(body) = function.entity.definition()
            && let Some(target) = slot(&body.fixups)
            && let Some(op) = unit.function_op(super::desc::FuncId::new(index as u32))
        {
            let target = unit.desc.function(target).name.clone();
            return Err(not_built(
                &op.borrow(),
                &format!(
                    "this function takes the slot of `{target}` in the function table, which the \
                     layout gives"
                ),
            ));
        }
    }
    for (index, global) in unit.desc.globals().iter().enumerate() {
        if let Some(expr) = global.entity.definition()
            && let Some(target) = slot(&expr.fixups)
            && let Some(op) = unit.global_op(super::desc::GlobalId::new(index as u32))
        {
            let target = unit.desc.function(target).name.clone();
            return Err(not_built(
                &op.borrow(),
                &format!(
                    "this global takes the slot of `{target}` in the function table, which the \
                     layout gives"
                ),
            ));
        }
    }
    Ok(Resolved::new())
}

/// The error of a unit that needs the layout, at `op`, with `label` saying what needs it.
fn not_built(op: &Operation, label: &str) -> Report {
    report!(
        message: "the module needs the layout of its linear memory or function table, which this \
                  backend does not build yet",
        labels: [primary(op.span(), alloc::string::String::from(label))],
    )
}
