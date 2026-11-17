//! The layout of linear memory and of the indirect function table (decision 2 of the plan): what
//! a linker does for one unit, so that the module the writer makes needs no host.
//!
//! [layout] turns the unit's description into its final form and gives the [Resolved] the writer
//! needs: the address of every data entry that has one and the slot of every function in the
//! table. With a stack, linear memory is:
//!
//! ```text
//! 0 ............ stack size   the shadow stack, growing down (an overflow traps instead of
//!                             overwriting data); `__stack_pointer` starts at the stack size
//! stack size ... data end     every builtin.global_variable laid out, aligned by its type
//! data end ..... heap base    rounded up to 16
//! ```
//!
//! **The well-known symbols.** A memory, a table or a global that is an import from `env` with the
//! field `__linear_memory`, `__indirect_function_table` or `__stack_pointer` (what the lowering
//! declares) becomes a definition. One the unit defines itself is left as it is.
//!
//! **The stack.** It is reserved only when the layout defines `__stack_pointer`: the
//! [stack size](EmitOptions::stack_size) bytes from address 0, which must be a multiple of 16, and
//! the stack pointer, a `mut i32`, starts at that size. Otherwise no stack is reserved and data
//! starts at 1024, so that no data has the address 0.
//!
//! **Data.** The entries with the placement [DataPlacement::Layout] are laid out in this order:
//! those with no `section`, as written; then each section in order of first appearance, its
//! members as written. Each entry is at a multiple of its alignment; a section starts at a
//! multiple of the largest alignment among its members. The data ends where the last entry ends
//! (at the start of the data when there is none). An entry whose bytes are all zero and that has no
//! fix-up gets no segment: linear memory starts zeroed.
//!
//! **Linker-defined data symbols.** An undefined data symbol named `__data_end` is the end of the
//! data; `__heap_base` is that rounded up to 16; `__start_<s>` and `__stop_<s>` are the start of
//! the section `s` and the end of its last member, and a section with no member is an error naming
//! the declaration. Each is a [DataPlacement::Symbol]: an address and no segment. Any other
//! undefined data symbol is an error: a core module cannot import data.
//!
//! **Passive data.** A global variable named by a `wasm.memory_init` or a `wasm.data_drop` is a
//! passive segment, with no address. Naming it also by `wasm.data_addr`, or taking its address with
//! `builtin.global_symbol` in another global variable's initializer, is an error naming both.
//!
//! **Fixed segments.** A `builtin.segment` is an active segment at its offset with its bytes. One
//! that overlaps the stack, the laid-out data, or another fixed segment is an error naming it (and
//! the other segment).
//!
//! **The memory.** The pages needed are the end of everything above (the heap base, or the end of
//! the highest fixed segment when that is higher) rounded up to 64 KiB pages; more than 4 GiB is
//! an error. When the layout defines the memory, that many pages are its minimum (or the
//! minimum the declaration states, if it is larger), it has no maximum, and it is exported as
//! `memory` unless the unit already has an export of that name. A memory the unit defines or
//! imports otherwise keeps its limits; when something is placed in it (a stack, laid-out data or a
//! fixed segment), a minimum below the pages needed is an error naming the memory and the number.
//! A unit with data and no memory at all gets `__linear_memory` defined as above.
//!
//! **The table.** Every function that is the target of a [FixupKind::TableSlot] fix-up gets a
//! slot, from 1, in order of first appearance: function bodies by id, then the initializers of
//! globals, then data, the fix-ups of each by offset. Slot 0 stays empty, so that a null function
//! pointer traps. When the layout defines the table, it is a `funcref` table whose minimum and
//! maximum are both one more than the number of slots, with one active element segment at offset
//! 1 (none when there are no slots). When there are slots to give and the unit has no such table,
//! the layout defines one. A table named `__indirect_function_table` that the unit lays out itself
//! cannot take slots: when there are slots to give, it is an error naming it.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{OperationRef, Report, Spanned, diagnostics::report, dialects::builtin::Segment};
use wasm_encoder::Instruction;

use super::{
    EmitOptions,
    collect::{Unit, cannot_import_data},
    desc::{
        ConstExpr, DataEntry, DataId, DataPlacement, ElemSegment, Entity, Export, ExportTarget,
        Feature, FeatureSet, FixupKind, FuncId, Global, GlobalId, Import, Limits, Memory, MemoryId,
        RefType, Table, TableId, ValType,
    },
    write::Resolved,
};

/// The size of a page of linear memory, in bytes.
pub const PAGE_SIZE: u64 = 65_536;

/// Where data starts when no stack is reserved: no data has the address 0.
pub const DATA_BASE_WITHOUT_STACK: u64 = 1024;

/// The alignment of the stack's size and of the heap base, in bytes.
pub const STACK_ALIGNMENT: u64 = 16;

/// The size of a 32-bit linear memory, in bytes.
const MEMORY_LIMIT: u64 = 1 << 32;

/// The import module of the well-known symbols.
const ENV: &str = "env";

/// The well-known symbols, as the lowering names them.
const LINEAR_MEMORY: &str = crate::symbols::LINEAR_MEMORY;
const INDIRECT_FUNCTION_TABLE: &str = crate::symbols::INDIRECT_FUNCTION_TABLE;
const STACK_POINTER: &str = crate::symbols::STACK_POINTER;

/// The name of the memory export of a memory the layout defines.
const MEMORY_EXPORT: &str = "memory";

/// Lays out the unit: defines the well-known symbols it imports, places the stack, the data and
/// the fixed segments, gives table slots, and returns the addresses and slots the writer needs
/// (see the module docs for the rules). An error names the operation the rule it breaks is about.
pub fn layout(unit: &mut Unit, options: &EmitOptions) -> Result<Resolved, Report> {
    let mut resolved = Resolved::new();
    let linker = linker_symbols(unit)?;
    passive(unit)?;

    // The stack, when the layout defines the stack pointer.
    let stack_pointer = well_known_global(unit);
    let stack = match stack_pointer {
        Some(id) => Some(stack(unit, id, options.stack_size)?),
        None => None,
    };
    let data_start = stack.map_or(DATA_BASE_WITHOUT_STACK, u64::from);

    // Data.
    let placed = place_data(unit, data_start, &mut resolved)?;
    let heap_base = align_up(placed.end, STACK_ALIGNMENT);
    if heap_base > u64::from(u32::MAX) {
        return Err(too_large(heap_base));
    }
    for (id, symbol) in &linker {
        let address = match symbol {
            Linker::DataEnd => placed.end,
            Linker::HeapBase => heap_base,
            Linker::Start(section) | Linker::Stop(section) => {
                let Some((start, stop)) = placed.section(section) else {
                    return Err(empty_section(unit, *id, symbol));
                };
                if matches!(symbol, Linker::Start(_)) {
                    start
                } else {
                    stop
                }
            }
        };
        resolved.set_data_address(*id, address as u32);
    }

    // Fixed segments.
    let fixed_end = fixed_segments(unit, stack, data_start, placed.end)?;

    // The memory.
    let end = heap_base.max(fixed_end);
    if end > MEMORY_LIMIT {
        return Err(too_large(end));
    }
    let pages = end.div_ceil(PAGE_SIZE) as u32;
    let occupied = stack.is_some()
        || unit.desc.data().iter().any(|entry| {
            matches!(entry.placement, DataPlacement::Layout | DataPlacement::Fixed(_))
                && !entry.bytes.is_empty()
        });
    let has_data = unit.desc.data().iter().any(|entry| entry.placement != DataPlacement::Symbol);
    memory(unit, pages, occupied, has_data || stack.is_some())?;

    // The table.
    table(unit, &mut resolved)?;

    // The stack pointer.
    if let (Some(id), Some(size)) = (stack_pointer, stack) {
        let mut init = ConstExpr::default();
        init.writer().instruction(&Instruction::I32Const(size as i32));
        unit.desc.global_mut(id).entity = Entity::Definition(init);
    }

    features(unit);
    Ok(resolved)
}

/// The error of a unit whose stack and data need `end` bytes, beyond what a 32-bit memory
/// addresses.
fn too_large(end: u64) -> Report {
    Report::msg(format!(
        "the unit's stack and data need {end} bytes of linear memory, beyond the 4 GiB a 32-bit \
         memory addresses"
    ))
}

/// A linker-defined data symbol.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Linker {
    /// `__data_end`.
    DataEnd,
    /// `__heap_base`.
    HeapBase,
    /// `__start_<section>`.
    Start(String),
    /// `__stop_<section>`.
    Stop(String),
}

impl Linker {
    /// The linker-defined symbol `name` stands for, if any.
    fn of(name: &str) -> Option<Self> {
        match name {
            "__data_end" => Some(Self::DataEnd),
            "__heap_base" => Some(Self::HeapBase),
            _ => {
                if let Some(section) = name.strip_prefix("__start_") {
                    Some(Self::Start(section.to_string()))
                } else {
                    name.strip_prefix("__stop_").map(|section| Self::Stop(section.to_string()))
                }
            }
        }
    }
}

/// The undefined data symbols that are linker-defined, each made a [DataPlacement::Symbol], or
/// the error of the first that is not.
fn linker_symbols(unit: &mut Unit) -> Result<Vec<(DataId, Linker)>, Report> {
    let mut symbols = Vec::new();
    for id in unit.undefined_data().to_vec() {
        let name = unit.desc.data()[id.index() as usize].name.clone();
        match Linker::of(&name) {
            Some(symbol) => {
                unit.desc.data_mut(id).placement = DataPlacement::Symbol;
                symbols.push((id, symbol));
            }
            None => {
                let op = unit.data_op(id).expect("an undefined data symbol has a declaration");
                return Err(cannot_import_data(&op.borrow(), &name));
            }
        }
    }
    Ok(symbols)
}

/// The error of `__start_<s>` or `__stop_<s>` (the data symbol `id`) for a section `s` with no
/// member.
fn empty_section(unit: &Unit, id: DataId, symbol: &Linker) -> Report {
    let (Linker::Start(section) | Linker::Stop(section)) = symbol else {
        unreachable!("only a section bound names a section");
    };
    let (name, bound) = match symbol {
        Linker::Start(_) => (format!("__start_{section}"), "start"),
        _ => (format!("__stop_{section}"), "end"),
    };
    let op = unit.data_op(id).expect("an undefined data symbol has a declaration");
    let op = op.borrow();
    report!(
        message: (format!("the section `{section}` has no member, so `{name}` has no address")),
        labels: [primary(
            op.span(),
            format!(
                "this declares `{name}`, the {bound} of the section `{section}`, and no global \
                 variable of the unit is placed in it"
            )
        )],
    )
}

/// Makes every global variable named by `wasm.memory_init` or `wasm.data_drop` a passive entry;
/// one whose address is also taken, or that is a linker-defined symbol, is an error.
fn passive(unit: &mut Unit) -> Result<(), Report> {
    for index in 0..unit.desc.data().len() {
        let id = DataId::new(index as u32);
        let Some(passive) = unit.passive_use(id) else {
            continue;
        };
        let name = unit.desc.data()[index].name.clone();
        if unit.undefined_data().contains(&id) {
            let passive = passive.borrow();
            return Err(report!(
                message: (format!(
                    "`{}` names `{name}`, an address the layout gives, which has no bytes to copy",
                    passive.name()
                )),
                labels: [primary(
                    passive.span(),
                    format!("this names `{name}` as a passive data segment")
                )],
            ));
        }
        if let Some(address) = unit.address_use(id) {
            let (address, passive) = (address.borrow(), passive.borrow());
            return Err(report!(
                message: (format!(
                    "the global variable `{name}` is both a passive data segment and data with an \
                     address"
                )),
                labels: [
                    primary(
                        address.span(),
                        format!(
                            "this takes the address of `{name}`, which a passive segment does not \
                             have"
                        )
                    ),
                    context(
                        passive.span(),
                        format!("this makes `{name}` a passive segment, copied by `memory.init`")
                    ),
                ],
            ));
        }
        unit.desc.data_mut(id).placement = DataPlacement::Passive;
    }
    Ok(())
}

/// The global that is the import of `__stack_pointer` from `env`, which the layout defines.
fn well_known_global(unit: &Unit) -> Option<GlobalId> {
    unit.desc
        .globals()
        .iter()
        .position(|global| is_well_known(global.entity.import(), STACK_POINTER))
        .map(|index| GlobalId::new(index as u32))
}

/// Whether `import` is the import of the well-known symbol `field` from `env`.
fn is_well_known(import: Option<&Import>, field: &str) -> bool {
    import.is_some_and(|import| import.module == ENV && import.field == field)
}

/// The size of the stack the layout reserves for the stack pointer `id`, or the error that the
/// stack pointer is not a `mut i32` or that `size` is not a multiple of 16.
fn stack(unit: &Unit, id: GlobalId, size: u32) -> Result<u32, Report> {
    let global = &unit.desc.globals()[id.index() as usize];
    let op = unit.global_op(id).expect("an import has a declaration");
    let op = op.borrow();
    if global.ty != ValType::I32 || !global.mutable {
        return Err(report!(
            message: "the stack pointer `__stack_pointer` is a `mut i32`",
            labels: [primary(
                op.span(),
                "this declares `__stack_pointer`, which the layout defines, with another type"
            )],
        ));
    }
    if u64::from(size) % STACK_ALIGNMENT != 0 {
        return Err(report!(
            message: (format!(
                "the stack size {size} (`--wasm-stack-size`) is not a multiple of 16, the \
                 alignment of the stack"
            )),
            labels: [primary(
                op.span(),
                "the stack this stack pointer points into is laid out with that size"
            )],
        ));
    }
    Ok(size)
}

/// `value` rounded up to a multiple of `align` (a power of two).
fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

/// Where the laid-out data is.
struct Placed {
    /// The end of the data.
    end: u64,
    /// Each section with members: its name, its start and the end of its last member.
    sections: Vec<(String, u64, u64)>,
}

impl Placed {
    /// The start and end of the section `name`, when it has members.
    fn section(&self, name: &str) -> Option<(u64, u64)> {
        self.sections
            .iter()
            .find(|(section, ..)| section == name)
            .map(|(_, start, end)| (*start, *end))
    }
}

/// Lays out the entries with the placement [DataPlacement::Layout] from `start`, giving each its
/// address in `resolved`.
fn place_data(unit: &Unit, start: u64, resolved: &mut Resolved) -> Result<Placed, Report> {
    let entries = unit.desc.data();
    let laid_out = |index: &usize| entries[*index].placement == DataPlacement::Layout;
    let mut sections: Vec<(String, Vec<usize>)> = Vec::new();
    let mut plain = Vec::new();
    for index in (0..entries.len()).filter(laid_out) {
        match &entries[index].section {
            None => plain.push(index),
            Some(section) => match sections.iter_mut().find(|(name, _)| name == section) {
                Some((_, members)) => members.push(index),
                None => sections.push((section.clone(), Vec::from([index]))),
            },
        }
    }
    let mut address = start;
    let mut place = |index: usize, address: &mut u64| -> Result<(), Report> {
        let entry = &entries[index];
        *address = align_up(*address, u64::from(entry.align.max(1)));
        let end = *address + entry.bytes.len() as u64;
        if end > MEMORY_LIMIT {
            let op = unit.data_op(DataId::new(index as u32)).expect("a definition has an origin");
            let op = op.borrow();
            return Err(report!(
                message: "the data of the unit does not fit the 4 GiB of a 32-bit linear memory",
                labels: [primary(op.span(), format!("this global variable would end at {end}"))],
            ));
        }
        resolved.set_data_address(DataId::new(index as u32), *address as u32);
        *address = end;
        Ok(())
    };
    for index in plain {
        place(index, &mut address)?;
    }
    let mut bounds = Vec::new();
    for (name, members) in sections {
        let align = members.iter().map(|index| entries[*index].align.max(1)).max().unwrap_or(1);
        address = align_up(address, u64::from(align));
        let section_start = address;
        for index in members {
            place(index, &mut address)?;
        }
        bounds.push((name, section_start, address));
    }
    Ok(Placed {
        end: address,
        sections: bounds,
    })
}

/// Adds each `builtin.segment` to the description as an entry at its fixed offset, or the error
/// of one that overlaps the stack (`0..stack`), the laid-out data (`data_start..data_end`) or
/// another segment. Gives the end of the highest segment (0 when there is none).
fn fixed_segments(
    unit: &mut Unit,
    stack: Option<u32>,
    data_start: u64,
    data_end: u64,
) -> Result<u64, Report> {
    let mut placed: Vec<(OperationRef, u64, u64)> = Vec::new();
    let mut highest = 0;
    for op in unit.segments().to_vec() {
        let borrowed = op.borrow();
        let segment = borrowed.downcast_ref::<Segment>().expect("a builtin.segment");
        let start = segment.offset_in_bytes();
        let bytes = segment.initializer().as_slice().to_vec();
        let end = start.saturating_add(bytes.len() as u64);
        let overlap = |label: String| {
            report!(
                message: "a `builtin.segment` overlaps memory the layout uses",
                labels: [primary(borrowed.span(), label)],
            )
        };
        if end > MEMORY_LIMIT || start > u64::from(u32::MAX) {
            return Err(overlap(format!(
                "this segment holds the bytes {start}..{end}, beyond the 4 GiB of a 32-bit linear \
                 memory"
            )));
        }
        if !bytes.is_empty() {
            if let Some(stack) = stack
                && start < u64::from(stack)
            {
                return Err(overlap(format!(
                    "this segment holds the bytes {start}..{end}, and the shadow stack is at \
                     0..{stack}"
                )));
            }
            if data_end > data_start && start < data_end && end > data_start {
                return Err(overlap(format!(
                    "this segment holds the bytes {start}..{end}, and the global variables are \
                     laid out at {data_start}..{data_end}"
                )));
            }
            if let Some((other, other_start, other_end)) = placed
                .iter()
                .find(|(_, other_start, other_end)| start < *other_end && end > *other_start)
            {
                let other = other.borrow();
                return Err(report!(
                    message: "two `builtin.segment`s overlap",
                    labels: [
                        primary(
                            borrowed.span(),
                            format!("this segment holds the bytes {start}..{end}")
                        ),
                        context(
                            other.span(),
                            format!("this one holds the bytes {other_start}..{other_end}")
                        ),
                    ],
                ));
            }
            placed.push((op, start, end));
            highest = highest.max(end);
        }
        drop(borrowed);
        unit.desc.add_data(DataEntry {
            name: String::new(),
            bytes,
            align: 1,
            section: None,
            placement: DataPlacement::Fixed(start as u32),
            fixups: Vec::new(),
        });
    }
    Ok(highest)
}

/// Defines the memory the unit imports as `__linear_memory` from `env`, of `pages` pages, or
/// checks that the memory the unit has holds `pages` pages when something is placed in it
/// (`occupied`), or defines `__linear_memory` when the unit `needs` one and has none.
fn memory(unit: &mut Unit, pages: u32, occupied: bool, needs: bool) -> Result<(), Report> {
    let well_known = unit
        .desc
        .memories()
        .iter()
        .position(|memory| is_well_known(memory.entity.import(), LINEAR_MEMORY));
    if let Some(index) = well_known {
        let id = MemoryId::new(index as u32);
        let memory = unit.desc.memory_mut(id);
        memory.limits = Limits::at_least(pages.max(memory.limits.min));
        memory.entity = Entity::Definition(());
        export_memory(unit, id);
        return Ok(());
    }
    if let Some((index, memory)) = unit.desc.memories().iter().enumerate().next() {
        if occupied && memory.limits.min < pages {
            let op = unit.memory_op(MemoryId::new(index as u32)).expect("a memory has an origin");
            let op = op.borrow();
            let name = op
                .as_symbol()
                .map_or_else(|| String::from("the memory"), |symbol| format!("@{}", symbol.name()));
            return Err(report!(
                message: (format!(
                    "the memory {name} is too small for what the layout places in it: it needs \
                     {pages} pages of 64 KiB, and its minimum is {}",
                    memory.limits.min
                )),
                labels: [primary(
                    op.span(),
                    format!(
                        "this memory has {} pages at first, and the stack and data need {pages}",
                        memory.limits.min
                    )
                )],
            ));
        }
        return Ok(());
    }
    if needs {
        let id = unit.desc.add_memory(Memory {
            limits: Limits::at_least(pages),
            name: String::from(LINEAR_MEMORY),
            entity: Entity::Definition(()),
        });
        export_memory(unit, id);
    }
    Ok(())
}

/// Exports the memory `id` as `memory`, unless the unit already has an export of that name.
fn export_memory(unit: &mut Unit, id: MemoryId) {
    if !unit.desc.exports().iter().any(|export| export.name == MEMORY_EXPORT) {
        unit.desc.add_export(Export {
            name: String::from(MEMORY_EXPORT),
            target: ExportTarget::Memory(id),
        });
    }
}

/// Gives a table slot to every function a [FixupKind::TableSlot] fix-up names, and defines the
/// table that holds them (see the module docs).
fn table(unit: &mut Unit, resolved: &mut Resolved) -> Result<(), Report> {
    let (slots, first) = slots(unit);
    for (slot, function) in slots.iter().enumerate() {
        resolved.set_table_slot(*function, slot as u32 + 1);
    }
    let size = slots.len() as u32 + 1;
    let well_known = unit
        .desc
        .tables()
        .iter()
        .position(|table| is_well_known(table.entity.import(), INDIRECT_FUNCTION_TABLE));
    let id = match well_known {
        Some(index) => {
            let id = TableId::new(index as u32);
            if unit.desc.tables()[index].element != RefType::FuncRef {
                let op = unit.table_op(id).expect("an import has a declaration");
                let op = op.borrow();
                return Err(report!(
                    message: "the function table `__indirect_function_table` is a table of `funcref`",
                    labels: [primary(
                        op.span(),
                        "this declares `__indirect_function_table`, which the layout defines, \
                         with another element type"
                    )],
                ));
            }
            let table = unit.desc.table_mut(id);
            table.limits = Limits {
                min: size,
                max: Some(size),
            };
            table.entity = Entity::Definition(());
            id
        }
        None if slots.is_empty() => return Ok(()),
        None => {
            // The unit's own table is the one the lowering's code names: a table whose symbol
            // name or link name is `__indirect_function_table`.
            let own = (0..unit.desc.tables().len()).find(|index| {
                let id = TableId::new(*index as u32);
                unit.desc.tables()[*index].name == INDIRECT_FUNCTION_TABLE
                    || unit.table_op(id).is_some_and(|op| {
                        op.borrow()
                            .as_symbol()
                            .is_some_and(|symbol| symbol.name().as_str() == INDIRECT_FUNCTION_TABLE)
                    })
            });
            if let Some(index) = own {
                let op = unit.table_op(TableId::new(index as u32)).expect("a table has an origin");
                let op = op.borrow();
                let (site, function) = first.expect("there are slots");
                let site = site.borrow();
                let function = unit.desc.function(function).name.clone();
                return Err(report!(
                    message: "the backend cannot place functions in a function table the module \
                              lays out itself",
                    labels: [
                        primary(
                            op.span(),
                            "this is `__indirect_function_table`, which the module lays out \
                             itself, so the layout cannot give its slots"
                        ),
                        context(site.span(), format!("this takes the slot of `{function}`")),
                    ],
                ));
            }
            unit.desc.add_table(Table {
                element: RefType::FuncRef,
                limits: Limits {
                    min: size,
                    max: Some(size),
                },
                name: String::from(INDIRECT_FUNCTION_TABLE),
                entity: Entity::Definition(()),
            })
        }
    };
    if !slots.is_empty() {
        unit.desc.add_element(ElemSegment::Active {
            table: id,
            offset: 1,
            functions: slots,
        });
    }
    Ok(())
}

/// The functions [FixupKind::TableSlot] fix-ups name, each once, in order of first appearance:
/// function bodies by id, then globals' initializers, then data, each's fix-ups by offset. With
/// them, the operation that holds the first fix-up and the function it names.
fn slots(unit: &Unit) -> (Vec<FuncId>, Option<(OperationRef, FuncId)>) {
    let mut slots = Vec::new();
    let mut first = None;
    let mut visit = |fixups: &[super::desc::Fixup], site: Option<OperationRef>| {
        let mut fixups = fixups.to_vec();
        fixups.sort_by_key(|fixup| fixup.offset);
        for fixup in fixups {
            if let FixupKind::TableSlot(function) = fixup.kind {
                if first.is_none()
                    && let Some(site) = site
                {
                    first = Some((site, function));
                }
                if !slots.contains(&function) {
                    slots.push(function);
                }
            }
        }
    };
    for (index, function) in unit.desc.functions().iter().enumerate() {
        if let Some(body) = function.entity.definition() {
            visit(&body.fixups, unit.function_op(FuncId::new(index as u32)));
        }
    }
    for (index, global) in unit.desc.globals().iter().enumerate() {
        if let Some(expr) = global.entity.definition() {
            visit(&expr.fixups, unit.global_op(GlobalId::new(index as u32)));
        }
    }
    for (index, entry) in unit.desc.data().iter().enumerate() {
        visit(&entry.fixups, unit.data_op(DataId::new(index as u32)));
    }
    (slots, first)
}

/// The features of the laid-out module: mutable-globals only for a mutable global that is still
/// imported or is exported (the stack pointer the layout defines is neither), and reference-types
/// for more than one table.
fn features(unit: &mut Unit) {
    let desc = &unit.desc;
    let exported = |id: GlobalId| {
        desc.exports().iter().any(|export| export.target == ExportTarget::Global(id))
    };
    let mutable_globals =
        desc.globals().iter().enumerate().any(|(index, global): (usize, &Global)| {
            global.mutable && (global.entity.is_import() || exported(GlobalId::new(index as u32)))
        });
    let mut features = FeatureSet::new();
    for feature in desc.features.iter() {
        if feature != Feature::MutableGlobals || mutable_globals {
            features.insert(feature);
        }
    }
    if desc.tables().len() > 1 {
        features.insert(Feature::ReferenceTypes);
    }
    unit.desc.features = features;
}
