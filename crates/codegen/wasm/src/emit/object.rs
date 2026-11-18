//! The object writer: a collected [Unit] becomes a relocatable object in the tool conventions'
//! linking format (`Linking.md`), which `wasm-ld` links with other objects.
//!
//! Where the complete-module writer ([super::write]) resolves every fix-up to its final value,
//! the object writer keeps each placeholder at its width, fills it with a provisional value, and
//! records a relocation that tells the linker what the placeholder stands for. Nothing is laid
//! out: the linker places the data, gives the table slots and defines the memory, the stack
//! pointer and the function table. What follows is the object as [write_object] writes it; LLVM's
//! `WasmObjectWriter.cpp` is the model wherever these rules do not say otherwise.
//!
//! **Sections**, in this order, each left out when empty: type, import, function, table (the
//! tables the unit defines), global, export, code, data, then the custom sections `linking`,
//! `reloc.CODE`, `reloc.DATA`, `producers` and `target_features`. There is no memory, element,
//! data-count or `name` section: the linker builds the output's names from the symbols.
//!
//! **Imports.** The memory is always the import `env.__linear_memory`, with as many pages as the
//! object's data needs (its provisional size rounded up), and has no symbol; a memory the unit
//! declares is not imported under its own name. Then every function, table and global the unit
//! declares and does not define, from its module and field. The function table
//! `env.__indirect_function_table` is imported only when a fix-up needs it: a table number that
//! names it, or a table slot of a function. It is added when a table slot needs it and the unit
//! does not declare it, and left out when the unit declares it and nothing needs it.
//!
//! **Index spaces.** Functions, tables and globals are numbered as a complete module numbers
//! them: imports first, then definitions, each in description order. Types keep their interned
//! order.
//!
//! **Placeholders.** Each keeps its width (five bytes in code, a padded LEB128; four in data) and
//! holds a provisional value that validates: the object's own index of a function, type, global
//! or table (a block type is a padded unsigned LEB128, as LLVM writes it); for an address, the
//! provisional address of the data, its addend included (0 for an undefined data symbol); for a
//! table slot, 0.
//!
//! **Data segments.** Each global variable the unit defines without a `section` has a segment of
//! its own, named `.data.<link name>`, or `.bss.<link name>` when the evaluator gives it no bytes
//! (all zero, with no fix-up). Its zero bytes are written all the same: a data symbol must lie
//! within the bytes of its segment, and the linker leaves `.bss` segments out of its output. The
//! global variables of one `section` share one segment named exactly that, in the order written,
//! each at a multiple of its alignment, with the retain flag, so that the linker keeps it although
//! nothing names it. Segments are in order of their first member in the description. A segment's
//! alignment is the largest of its members'; it is active at a provisional offset, the running
//! size of the data before it rounded up to its alignment, which the linker ignores.
//!
//! **The symbol table** has one symbol for every function, every global, every table of the
//! object and every data entry, defined or not, in this order: the functions in description
//! order, then the globals, then the tables in the object's order (an added function table last),
//! then the data entries in description order. A symbol's binding gives its flags: a global
//! symbol none, a hidden one `VISIBILITY_HIDDEN`, a local one `BINDING_LOCAL`, an undefined one
//! `UNDEFINED`, with `EXPLICIT_NAME` when it names its host (so that the linker imports it from
//! that module). A function with an export name is `EXPORTED`, and the export section holds that
//! name for it; a global or a table exported under its own link name is `EXPORTED`. An exported
//! entity that is private is hidden rather than local: the linker exports no local symbol. The function
//! table is `NO_STRIP` when a table slot is taken, as LLVM marks it. A function, global or table
//! symbol carries its name when it is defined or names its host (otherwise the linker takes the
//! import's field); a data symbol always does, and a defined one gives its segment, its offset in
//! it and its size.
//!
//! **The rest of `linking`** (version 2): the segment info (each segment's name, the logarithm of
//! its alignment, and the retain flag), and, when the unit has a start function, one init
//! function at priority 65535 that names its symbol.
//!
//! **Relocations.** `reloc.CODE` and `reloc.DATA` name their section by its index among all the
//! sections written, then list one entry per fix-up in ascending offset: its type, its offset
//! from the start of the section's payload (for code, past the function count, every earlier body
//! with its size, this body's size and its locals; for data, past the segment count, every
//! earlier segment and this segment's header), the symbol (the type index, for a type) and, for an
//! address, the addend:
//!
//! | fix-up | in code | in data |
//! |---|---|---|
//! | [FixupKind::FunctionIndex] | `R_WASM_FUNCTION_INDEX_LEB` | |
//! | [FixupKind::TypeIndex], [FixupKind::BlockType] | `R_WASM_TYPE_INDEX_LEB` | |
//! | [FixupKind::GlobalIndex] | `R_WASM_GLOBAL_INDEX_LEB` | |
//! | [FixupKind::TableNumber] | `R_WASM_TABLE_NUMBER_LEB` | |
//! | [FixupKind::MemoryAddr] | `R_WASM_MEMORY_ADDR_SLEB` | `R_WASM_MEMORY_ADDR_I32` |
//! | [FixupKind::TableSlot] | `R_WASM_TABLE_INDEX_SLEB` | `R_WASM_TABLE_INDEX_I32` |
//!
//! A [FixupKind::DataIndex] has no relocation (the object check refuses passive data), and
//! neither does a fix-up in a global's initializer (the object check refuses those too).
//!
//! **Features.** `target_features` lists, with the prefix `+` and in alphabetical order, the
//! features the description records, and those LLVM's names add: `mutable-globals` when a
//! mutable global is imported or exported (the stack pointer is one), `call-indirect-overlong`
//! when a table number is written padded, `bulk-memory-opt` beside `bulk-memory` (in an object
//! only `memory.copy` and `memory.fill` use bulk memory), and `reference-types` when the object
//! has more than one table.

use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{Report, Spanned, diagnostics::report};
use wasm_encoder::{
    CustomSection, DataSymbolDefinition, Encode, EntityType, ExportKind, ExportSection,
    FunctionSection, GlobalSection, GlobalType, ImportSection, MemoryType, Module, RawSection,
    SectionId, SymbolTable, TableSection, TypeSection,
};

use super::{
    collect::Unit,
    desc::{
        Binding, CODE_PLACEHOLDER, DATA_PLACEHOLDER, DataId, Entity, ExportTarget, Feature, Fixup,
        FixupKind, FuncId, GlobalId, Import, Limits, ModuleDesc, RefType, TableId, ValType,
    },
    layout::{MEMORY_LIMIT, PAGE_SIZE},
    sections,
    write::{Site, WriteError, index_space, table_type},
};
use crate::symbols::INDIRECT_FUNCTION_TABLE;

/// The import module of the memory and the function table.
const ENV: &str = "env";

/// The field of the memory every object imports.
const LINEAR_MEMORY: &str = crate::symbols::LINEAR_MEMORY;

/// The priority of the start function among the init functions: the default of a constructor.
pub const START_PRIORITY: u32 = 65_535;

/// The version of the `linking` section.
const LINKING_VERSION: u32 = 2;

/// The `linking` subsection of the segment info.
const WASM_SEGMENT_INFO: u8 = 5;

/// The `linking` subsection of the init functions.
const WASM_INIT_FUNCS: u8 = 6;

/// The flag of a segment the linker keeps although no live symbol names it.
pub const WASM_SEG_FLAG_RETAIN: u32 = 0x4;

/// The type of a relocation, by its number in the tool conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum RelocType {
    FunctionIndexLeb = 0,
    TableIndexSleb = 1,
    TableIndexI32 = 2,
    MemoryAddrSleb = 4,
    MemoryAddrI32 = 5,
    TypeIndexLeb = 6,
    GlobalIndexLeb = 7,
    TableNumberLeb = 20,
}

impl RelocType {
    /// Whether the entry of a relocation of this type carries an addend.
    const fn has_addend(self) -> bool {
        matches!(self, Self::MemoryAddrSleb | Self::MemoryAddrI32)
    }

    /// Whether the placeholder is four bytes (in data) rather than a five-byte padded LEB128.
    const fn is_i32(self) -> bool {
        matches!(self, Self::TableIndexI32 | Self::MemoryAddrI32)
    }

    /// Whether the placeholder is a padded signed LEB128 (an `i32.const` operand).
    const fn is_signed(self) -> bool {
        matches!(self, Self::TableIndexSleb | Self::MemoryAddrSleb)
    }
}

/// One relocation: its type, the offset of its placeholder from the start of the section's
/// payload, the symbol (or type) it names, and its addend.
#[derive(Clone, Copy, Debug)]
struct Reloc {
    ty: RelocType,
    offset: u32,
    index: u32,
    addend: i32,
}

/// `value` as a five-byte padded unsigned LEB128.
fn padded_u32(value: u32) -> [u8; 5] {
    [
        (value & 0x7f) as u8 | 0x80,
        ((value >> 7) & 0x7f) as u8 | 0x80,
        ((value >> 14) & 0x7f) as u8 | 0x80,
        ((value >> 21) & 0x7f) as u8 | 0x80,
        ((value >> 28) & 0x0f) as u8,
    ]
}

/// `value` as a five-byte padded signed LEB128.
fn padded_i32(value: i32) -> [u8; 5] {
    let bits = value as u32;
    [
        (bits & 0x7f) as u8 | 0x80,
        ((bits >> 7) & 0x7f) as u8 | 0x80,
        ((bits >> 14) & 0x7f) as u8 | 0x80,
        ((bits >> 21) & 0x7f) as u8 | 0x80,
        ((value >> 28) & 0x7f) as u8,
    ]
}

/// `value` rounded up to a multiple of `align` (a power of two).
fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

/// A table of the object: one of the unit, or the function table the object adds.
struct ObjTable {
    /// The table of the unit, or `None` for the added function table.
    id: Option<TableId>,
    element: RefType,
    limits: Limits,
    name: String,
    import: Option<Import>,
    binding: Binding,
    no_strip: bool,
}

/// A data segment of the object.
struct Segment {
    name: String,
    align: u32,
    retain: bool,
    /// Each member and its offset in the segment.
    members: Vec<(DataId, u64)>,
    size: u64,
    /// The provisional offset of the segment in memory.
    offset: u64,
}

/// Where each entity is in the object: indices, symbols, tables and segments.
struct Plan<'a> {
    desc: &'a ModuleDesc,
    functions: Vec<u32>,
    globals: Vec<u32>,
    tables: Vec<ObjTable>,
    /// The position in `tables` of each table of the unit, when the object has it.
    table_of: Vec<Option<usize>>,
    /// The index of each table of `tables`.
    table_indices: Vec<u32>,
    segments: Vec<Segment>,
    /// The segment and offset of each data entry the unit defines.
    data_of: Vec<Option<(usize, u64)>>,
    /// The provisional size of all the data.
    data_size: u64,
}

/// Whether `import` is the import of the function table from `env`.
fn is_function_table(import: Option<&Import>) -> bool {
    import.is_some_and(|import| import.module == ENV && import.field == INDIRECT_FUNCTION_TABLE)
}

/// Every fix-up of the unit's function bodies and data.
fn fixups(desc: &ModuleDesc) -> impl Iterator<Item = &Fixup> {
    let bodies = desc
        .functions()
        .iter()
        .filter_map(|function| function.entity.definition())
        .flat_map(|body| body.fixups.iter());
    bodies.chain(desc.data().iter().flat_map(|entry| entry.fixups.iter()))
}

impl<'a> Plan<'a> {
    fn new(unit: &'a Unit) -> Result<Self, WriteError> {
        let desc = &unit.desc;
        let functions = index_space(desc.functions().iter().map(|f| f.entity.is_import()));
        let globals = index_space(desc.globals().iter().map(|g| g.entity.is_import()));

        // Tables.
        let slots = fixups(desc).any(|fixup| matches!(fixup.kind, FixupKind::TableSlot(_)));
        let numbered =
            |id: TableId| fixups(desc).any(|fixup| fixup.kind == FixupKind::TableNumber(id));
        let mut tables = Vec::new();
        let mut table_of = Vec::new();
        let mut function_table = false;
        for (index, table) in desc.tables().iter().enumerate() {
            let id = TableId::new(index as u32);
            let is_function_table = is_function_table(table.entity.import());
            if is_function_table && !slots && !numbered(id) {
                table_of.push(None);
                continue;
            }
            function_table |= is_function_table;
            table_of.push(Some(tables.len()));
            tables.push(ObjTable {
                id: Some(id),
                element: table.element,
                limits: table.limits,
                name: table.name.clone(),
                import: table.entity.import().cloned(),
                // Every table of the unit has a binding: none is added by a layout.
                binding: unit.table_binding(id).unwrap_or(Binding::Undefined { names_host: false }),
                no_strip: is_function_table && slots,
            });
        }
        if slots && !function_table {
            tables.push(ObjTable {
                id: None,
                element: RefType::FuncRef,
                limits: Limits::at_least(0),
                name: String::from(INDIRECT_FUNCTION_TABLE),
                import: Some(Import::new(ENV, INDIRECT_FUNCTION_TABLE)),
                binding: Binding::Undefined { names_host: false },
                no_strip: true,
            });
        }
        let table_indices = index_space(tables.iter().map(|table| table.import.is_some()));

        // Data.
        let mut segments: Vec<Segment> = Vec::new();
        let mut data_of = Vec::new();
        for (index, entry) in desc.data().iter().enumerate() {
            let id = DataId::new(index as u32);
            let site = || Site::Data {
                id,
                name: entry.name.clone(),
            };
            let binding = unit.data_binding(id).ok_or_else(|| WriteError::UnknownEntity {
                site: site(),
                entity: id.to_string(),
            })?;
            if binding.is_undefined() {
                data_of.push(None);
                continue;
            }
            let position = match &entry.section {
                Some(section) => match segments
                    .iter()
                    .position(|segment| segment.retain && segment.name == *section)
                {
                    Some(position) => position,
                    None => {
                        segments.push(Segment {
                            name: section.clone(),
                            align: 1,
                            retain: true,
                            members: Vec::new(),
                            size: 0,
                            offset: 0,
                        });
                        segments.len() - 1
                    }
                },
                None => {
                    let zero = entry.bytes.is_empty() && entry.fixups.is_empty();
                    let prefix = if zero { ".bss." } else { ".data." };
                    segments.push(Segment {
                        name: format!("{prefix}{}", entry.name),
                        align: 1,
                        retain: false,
                        members: Vec::new(),
                        size: 0,
                        offset: 0,
                    });
                    segments.len() - 1
                }
            };
            let segment = &mut segments[position];
            let align = entry.align.max(1);
            segment.align = segment.align.max(align);
            let offset = align_up(segment.size, u64::from(align));
            segment.size = offset.saturating_add(entry.size);
            segment.members.push((id, offset));
            data_of.push(Some((position, offset)));
        }
        let mut data_size = 0u64;
        for segment in &mut segments {
            segment.offset = align_up(data_size, u64::from(segment.align));
            data_size = segment.offset.saturating_add(segment.size);
        }

        Ok(Self {
            desc,
            functions,
            globals,
            tables,
            table_of,
            table_indices,
            segments,
            data_of,
            data_size,
        })
    }

    fn function_symbol(&self, id: FuncId) -> u32 {
        id.index()
    }

    fn global_symbol(&self, id: GlobalId) -> u32 {
        self.functions.len() as u32 + id.index()
    }

    fn table_symbol(&self, position: usize) -> u32 {
        (self.functions.len() + self.globals.len() + position) as u32
    }

    fn data_symbol(&self, id: DataId) -> u32 {
        (self.functions.len() + self.globals.len() + self.tables.len()) as u32 + id.index()
    }

    /// The provisional address of the data entry `id`: 0 when it is undefined.
    fn address(&self, id: DataId) -> u64 {
        match self.data_of.get(id.index() as usize).copied().flatten() {
            Some((segment, offset)) => self.segments[segment].offset + offset,
            None => 0,
        }
    }

    /// The relocation of `fixup`, in code or in data, with the provisional value of its
    /// placeholder; its offset is the fix-up's, which the caller rebases.
    fn relocation(
        &self,
        fixup: &Fixup,
        in_data: bool,
        site: &Site,
    ) -> Result<(Reloc, u32), WriteError> {
        let unknown = |entity: String| WriteError::UnknownEntity {
            site: site.clone(),
            entity,
        };
        let unsupported = || WriteError::UnsupportedFixup {
            site: site.clone(),
            fixup: *fixup,
        };
        let is_index = !matches!(fixup.kind, FixupKind::MemoryAddr(_));
        if (in_data && !fixup.kind.is_value()) || (is_index && fixup.addend != 0) {
            return Err(unsupported());
        }
        let (ty, index, value) = match fixup.kind {
            FixupKind::FunctionIndex(id) => {
                let index = self.functions.get(id.index() as usize).copied();
                let index = index.ok_or_else(|| unknown(id.to_string()))?;
                (RelocType::FunctionIndexLeb, self.function_symbol(id), index)
            }
            FixupKind::TypeIndex(id) | FixupKind::BlockType(id) => {
                if id.index() as usize >= self.desc.types().len() {
                    return Err(unknown(id.to_string()));
                }
                (RelocType::TypeIndexLeb, id.index(), id.index())
            }
            FixupKind::GlobalIndex(id) => {
                let index = self.globals.get(id.index() as usize).copied();
                let index = index.ok_or_else(|| unknown(id.to_string()))?;
                (RelocType::GlobalIndexLeb, self.global_symbol(id), index)
            }
            FixupKind::TableNumber(id) => {
                let position = self.table_of.get(id.index() as usize).copied().flatten();
                let position = position.ok_or_else(|| unknown(id.to_string()))?;
                (
                    RelocType::TableNumberLeb,
                    self.table_symbol(position),
                    self.table_indices[position],
                )
            }
            FixupKind::MemoryAddr(id) => {
                if id.index() as usize >= self.desc.data().len() {
                    return Err(unknown(id.to_string()));
                }
                let ty = if in_data {
                    RelocType::MemoryAddrI32
                } else {
                    RelocType::MemoryAddrSleb
                };
                let value = match self.data_of[id.index() as usize] {
                    Some(_) => (self.address(id) as u32).wrapping_add(fixup.addend as u32),
                    None => 0,
                };
                (ty, self.data_symbol(id), value)
            }
            FixupKind::TableSlot(id) => {
                if id.index() as usize >= self.functions.len() {
                    return Err(unknown(id.to_string()));
                }
                if fixup.addend != 0 {
                    return Err(unsupported());
                }
                let ty = if in_data {
                    RelocType::TableIndexI32
                } else {
                    RelocType::TableIndexSleb
                };
                (ty, self.function_symbol(id), 0)
            }
            FixupKind::DataIndex(_) => return Err(unsupported()),
        };
        let reloc = Reloc {
            ty,
            offset: fixup.offset,
            index,
            addend: if ty.has_addend() { fixup.addend } else { 0 },
        };
        Ok((reloc, value))
    }

    /// `code` with each placeholder of `fixups` holding its provisional value, and the
    /// relocations, at offsets from the start of `code` plus `base`.
    fn code(
        &self,
        code: &[u8],
        fixups: &[Fixup],
        base: u32,
        site: &Site,
        relocs: &mut Vec<Reloc>,
    ) -> Result<Vec<u8>, WriteError> {
        let mut out = code.to_vec();
        let mut fixups = fixups.to_vec();
        fixups.sort_by_key(|fixup| fixup.offset);
        let mut end = 0usize;
        for fixup in &fixups {
            let start = fixup.offset as usize;
            let stop = start + CODE_PLACEHOLDER.len();
            if start < end || code.get(start..stop) != Some(&CODE_PLACEHOLDER[..]) {
                return Err(WriteError::MisplacedPlaceholder {
                    site: site.clone(),
                    fixup: *fixup,
                });
            }
            let (mut reloc, value) = self.relocation(fixup, false, site)?;
            let bytes = if reloc.ty.is_signed() {
                padded_i32(value as i32)
            } else {
                padded_u32(value)
            };
            out[start..stop].copy_from_slice(&bytes);
            reloc.offset += base;
            relocs.push(reloc);
            end = stop;
        }
        Ok(out)
    }

    /// The bytes of the segment `segment`: each member at its offset (a member with no bytes is
    /// its size in zeros), each placeholder holding its provisional value, and the relocations at
    /// offsets from the start of the segment's bytes plus `base`.
    fn segment(
        &self,
        segment: &Segment,
        base: u32,
        relocs: &mut Vec<Reloc>,
        out: &mut Vec<u8>,
    ) -> Result<(), WriteError> {
        let start = out.len();
        for (id, offset) in &segment.members {
            let entry = &self.desc.data()[id.index() as usize];
            let site = Site::Data {
                id: *id,
                name: entry.name.clone(),
            };
            out.resize(start + *offset as usize, 0);
            let at = out.len();
            if entry.bytes.is_empty() {
                out.resize(at + entry.size as usize, 0);
            } else {
                out.extend_from_slice(&entry.bytes);
            }
            let mut fixups = entry.fixups.clone();
            fixups.sort_by_key(|fixup| fixup.offset);
            let mut end = 0usize;
            for fixup in &fixups {
                let first = fixup.offset as usize;
                let stop = first + DATA_PLACEHOLDER.len();
                if first < end || entry.bytes.get(first..stop) != Some(&DATA_PLACEHOLDER[..]) {
                    return Err(WriteError::MisplacedPlaceholder {
                        site,
                        fixup: *fixup,
                    });
                }
                let (mut reloc, value) = self.relocation(fixup, true, &site)?;
                out[at + first..at + stop].copy_from_slice(&value.to_le_bytes());
                reloc.offset = base + (at - start) as u32 + fixup.offset;
                relocs.push(reloc);
                end = stop;
            }
        }
        out.resize(start + segment.size as usize, 0);
        Ok(())
    }
}

/// The binding of the symbol of an entity of `binding` that is exported when `exported`: a local
/// symbol that is exported is hidden instead, since the linker exports only the symbols other
/// objects may name.
fn exported_binding(binding: Binding, exported: bool) -> Binding {
    match binding {
        Binding::Local if exported => Binding::Hidden,
        binding => binding,
    }
}

/// The flags of a function, global or table symbol of `binding`.
fn binding_flags(binding: Binding) -> u32 {
    match binding {
        Binding::Global => 0,
        Binding::Hidden => SymbolTable::WASM_SYM_VISIBILITY_HIDDEN,
        Binding::Local => SymbolTable::WASM_SYM_BINDING_LOCAL,
        Binding::Undefined { names_host: false } => SymbolTable::WASM_SYM_UNDEFINED,
        Binding::Undefined { names_host: true } => {
            SymbolTable::WASM_SYM_UNDEFINED | SymbolTable::WASM_SYM_EXPLICIT_NAME
        }
    }
}

/// The name a symbol of `binding` carries: its own when it is defined or names its host.
fn symbol_name(binding: Binding, name: &str) -> Option<&str> {
    match binding {
        Binding::Undefined { names_host: false } => None,
        _ => Some(name),
    }
}

/// The local declarations of `locals`: runs of one type, each its count and its type.
fn locals_bytes(locals: &[ValType]) -> Vec<u8> {
    let mut runs: Vec<(u32, ValType)> = Vec::new();
    for ty in locals {
        match runs.last_mut() {
            Some((count, last)) if last == ty => *count += 1,
            _ => runs.push((1, *ty)),
        }
    }
    let mut out = Vec::new();
    (runs.len() as u32).encode(&mut out);
    for (count, ty) in runs {
        count.encode(&mut out);
        ty.to_encoder().encode(&mut out);
    }
    out
}

/// Checks every relocation of `relocs` against the payload `payload` it applies to: the
/// placeholder at its offset has its width, five bytes of padded LEB128 or four bytes.
fn check_relocations(payload: &[u8], relocs: &[Reloc], site: &str) -> Result<(), WriteError> {
    for reloc in relocs {
        let start = reloc.offset as usize;
        let fits = if reloc.ty.is_i32() {
            payload.len() >= start + 4
        } else {
            payload.get(start..start + 5).is_some_and(|bytes| {
                bytes[..4].iter().all(|byte| byte & 0x80 != 0) && bytes[4] & 0x80 == 0
            })
        };
        if !fits {
            return Err(WriteError::MisplacedRelocation {
                section: String::from(site),
                offset: reloc.offset,
            });
        }
    }
    Ok(())
}

/// Appends the subsection `id` with the contents `data` to the `linking` section `sink`.
fn subsection(sink: &mut Vec<u8>, id: u8, data: &[u8]) {
    sink.push(id);
    data.encode(sink);
}

/// The relocation section `name` for the section of index `section`, with `relocs`.
fn reloc_section(name: &'static str, section: u32, relocs: &[Reloc]) -> CustomSection<'static> {
    let mut data = Vec::new();
    section.encode(&mut data);
    (relocs.len() as u32).encode(&mut data);
    for reloc in relocs {
        data.push(reloc.ty as u8);
        reloc.offset.encode(&mut data);
        reloc.index.encode(&mut data);
        if reloc.ty.has_addend() {
            reloc.addend.encode(&mut data);
        }
    }
    CustomSection {
        name: Cow::Borrowed(name),
        data: Cow::Owned(data),
    }
}

/// The `target_features` section of an object: `names`, each with the prefix `+`.
fn features_section(names: &[&str]) -> Option<CustomSection<'static>> {
    if names.is_empty() {
        return None;
    }
    let mut data = Vec::new();
    (names.len() as u32).encode(&mut data);
    for name in names {
        data.push(b'+');
        name.encode(&mut data);
    }
    Some(CustomSection {
        name: Cow::Borrowed("target_features"),
        data: Cow::Owned(data),
    })
}

/// A module being written, which counts its sections.
struct Sections {
    module: Module,
    count: u32,
}

impl Sections {
    /// Appends `section` and gives its index among the sections written.
    fn push(&mut self, section: &impl wasm_encoder::Section) -> u32 {
        self.module.section(section);
        self.count += 1;
        self.count - 1
    }
}

/// Checks that the data of `unit` fits a 32-bit linear memory when an object lays it out
/// provisionally (as [write_object] does), or gives the error naming the last global variable of
/// the first segment that would end beyond it.
pub fn check_data_size(unit: &Unit) -> Result<(), Report> {
    let plan = Plan::new(unit).map_err(object_error)?;
    for segment in &plan.segments {
        let end = segment.offset.saturating_add(segment.size);
        // A segment's size is written as 32 bits, so one of 4 GiB does not fit either.
        if end <= MEMORY_LIMIT && segment.size <= u64::from(u32::MAX) {
            continue;
        }
        let (id, _) = *segment.members.last().expect("a segment has a member");
        let op = unit.data_op(id).expect("a definition has an origin");
        let op = op.borrow();
        return Err(report!(
            message: "the data of the unit does not fit the 4 GiB of a 32-bit linear memory",
            labels: [primary(
                op.span(),
                format!(
                    "this global variable is in the segment `{}`, which would end at {end} in the \
                     object's data",
                    segment.name
                )
            )],
        ));
    }
    Ok(())
}

/// The error of a unit the object writer cannot write: a defect of the backend.
pub fn object_error(error: WriteError) -> Report {
    Report::msg(format!(
        "the relocatable object could not be written, which is a defect of the backend: {error}"
    ))
}

/// The unit `unit` written as a relocatable object (see the module docs).
///
/// The bytes are not validated (see [super::validate::validate]). The unit must have passed
/// [Unit::check_object]; an error is a defect of the code that built the unit: a fix-up that an
/// object has no relocation for, a placeholder that is not where its fix-up says, or an id that is
/// not of the unit.
pub fn write_object(unit: &Unit) -> Result<Vec<u8>, WriteError> {
    let desc = &unit.desc;
    let plan = Plan::new(unit)?;
    let function_site = |id: FuncId| Site::Function {
        id,
        name: desc.function(id).name.clone(),
    };
    let type_index = |id: FuncId| {
        let ty = desc.function(id).ty;
        if (ty.index() as usize) < desc.types().len() {
            Ok(ty.index())
        } else {
            Err(WriteError::UnknownEntity {
                site: function_site(id),
                entity: ty.to_string(),
            })
        }
    };
    let mut out = Sections {
        module: Module::new(),
        count: 0,
    };

    // Types.
    let mut types = TypeSection::new();
    for ty in desc.types() {
        types.ty().function(
            ty.params.iter().map(|ty| ty.to_encoder()),
            ty.results.iter().map(|ty| ty.to_encoder()),
        );
    }
    if !types.is_empty() {
        out.push(&types);
    }

    // Imports: the memory, then functions, tables and globals, each kind in description order.
    let mut imports = ImportSection::new();
    let pages = plan.data_size.div_ceil(PAGE_SIZE);
    imports.import(
        ENV,
        LINEAR_MEMORY,
        MemoryType {
            minimum: pages,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        },
    );
    for (i, function) in desc.functions().iter().enumerate() {
        if let Entity::Import(import) = &function.entity {
            let ty = type_index(FuncId::new(i as u32))?;
            imports.import(&import.module, &import.field, EntityType::Function(ty));
        }
    }
    for table in &plan.tables {
        if let Some(import) = &table.import {
            imports.import(&import.module, &import.field, table_type(table.element, table.limits));
        }
    }
    for global in desc.globals() {
        if let Entity::Import(import) = &global.entity {
            let ty = GlobalType {
                val_type: global.ty.to_encoder(),
                mutable: global.mutable,
                shared: false,
            };
            imports.import(&import.module, &import.field, ty);
        }
    }
    out.push(&imports);

    // Functions.
    let mut functions = FunctionSection::new();
    for (i, function) in desc.functions().iter().enumerate() {
        if !function.entity.is_import() {
            functions.function(type_index(FuncId::new(i as u32))?);
        }
    }
    if !functions.is_empty() {
        out.push(&functions);
    }

    // Tables the unit defines.
    let mut tables = TableSection::new();
    for table in plan.tables.iter().filter(|table| table.import.is_none()) {
        tables.table(table_type(table.element, table.limits));
    }
    if !tables.is_empty() {
        out.push(&tables);
    }

    // Globals: an initializer names no other entity.
    let mut globals = GlobalSection::new();
    for (i, global) in desc.globals().iter().enumerate() {
        if let Entity::Definition(init) = &global.entity {
            if let Some(fixup) = init.fixups.first() {
                return Err(WriteError::UnsupportedFixup {
                    site: Site::Global {
                        id: GlobalId::new(i as u32),
                        name: global.name.clone(),
                    },
                    fixup: *fixup,
                });
            }
            let ty = GlobalType {
                val_type: global.ty.to_encoder(),
                mutable: global.mutable,
                shared: false,
            };
            globals.global(ty, &wasm_encoder::ConstExpr::raw(init.code.iter().copied()));
        }
    }
    if !globals.is_empty() {
        out.push(&globals);
    }

    // Exports: each function with an export name, under it.
    let mut exports = ExportSection::new();
    for (i, index) in plan.functions.iter().enumerate() {
        if let Some(name) = unit.function_export(FuncId::new(i as u32)) {
            exports.export(name, ExportKind::Func, *index);
        }
    }
    if !exports.is_empty() {
        out.push(&exports);
    }

    // Code.
    let mut code_relocs = Vec::new();
    let mut code = Vec::new();
    let bodies = desc.functions().iter().filter(|function| !function.entity.is_import()).count();
    (bodies as u32).encode(&mut code);
    for (i, function) in desc.functions().iter().enumerate() {
        let Entity::Definition(body) = &function.entity else {
            continue;
        };
        let site = function_site(FuncId::new(i as u32));
        let locals = locals_bytes(&body.locals);
        let size = (locals.len() + body.code.len()) as u32;
        size.encode(&mut code);
        let base = (code.len() + locals.len()) as u32;
        let bytes = plan.code(&body.code, &body.fixups, base, &site, &mut code_relocs)?;
        code.extend_from_slice(&locals);
        code.extend_from_slice(&bytes);
    }
    let code_section = if bodies > 0 {
        check_relocations(&code, &code_relocs, "CODE")?;
        Some(out.push(&RawSection {
            id: SectionId::Code.into(),
            data: &code,
        }))
    } else {
        None
    };

    // Data.
    let mut data_relocs = Vec::new();
    let mut data = Vec::new();
    (plan.segments.len() as u32).encode(&mut data);
    for segment in &plan.segments {
        // Active in memory 0 at the provisional offset: `i32.const <offset> end`.
        data.push(0x00);
        data.push(0x41);
        (segment.offset as u32 as i32).encode(&mut data);
        data.push(0x0b);
        (segment.size as u32).encode(&mut data);
        let base = data.len() as u32;
        plan.segment(segment, base, &mut data_relocs, &mut data)?;
    }
    let data_section = if plan.segments.is_empty() {
        None
    } else {
        check_relocations(&data, &data_relocs, "DATA")?;
        Some(out.push(&RawSection {
            id: SectionId::Data.into(),
            data: &data,
        }))
    };

    // The symbol table.
    let mut symbols = SymbolTable::new();
    for (i, function) in desc.functions().iter().enumerate() {
        let id = FuncId::new(i as u32);
        let binding = unit.function_binding(id).ok_or_else(|| WriteError::UnknownEntity {
            site: function_site(id),
            entity: id.to_string(),
        })?;
        let exported = unit.function_export(id).is_some();
        let mut flags = binding_flags(exported_binding(binding, exported));
        if exported {
            flags |= SymbolTable::WASM_SYM_EXPORTED;
        }
        symbols.function(flags, plan.functions[i], symbol_name(binding, &function.name));
    }
    let exported = |target: ExportTarget| desc.export_names(target).next().is_some();
    for (i, global) in desc.globals().iter().enumerate() {
        let id = GlobalId::new(i as u32);
        let binding = unit.global_binding(id).ok_or_else(|| WriteError::UnknownEntity {
            site: Site::Global {
                id,
                name: global.name.clone(),
            },
            entity: id.to_string(),
        })?;
        let is_exported = exported(ExportTarget::Global(id));
        let mut flags = binding_flags(exported_binding(binding, is_exported));
        if is_exported {
            flags |= SymbolTable::WASM_SYM_EXPORTED;
        }
        symbols.global(flags, plan.globals[i], symbol_name(binding, &global.name));
    }
    for (position, table) in plan.tables.iter().enumerate() {
        let is_exported = table.id.is_some_and(|id| exported(ExportTarget::Table(id)));
        let mut flags = binding_flags(exported_binding(table.binding, is_exported));
        if table.no_strip {
            flags |= SymbolTable::WASM_SYM_NO_STRIP;
        }
        if is_exported {
            flags |= SymbolTable::WASM_SYM_EXPORTED;
        }
        symbols.table(flags, plan.table_indices[position], symbol_name(table.binding, &table.name));
    }
    for (i, entry) in desc.data().iter().enumerate() {
        let id = DataId::new(i as u32);
        let definition = plan.data_of[i].map(|(segment, offset)| DataSymbolDefinition {
            index: segment as u32,
            offset: offset as u32,
            size: entry.size as u32,
        });
        let flags = match unit.data_binding(id) {
            Some(Binding::Undefined { .. }) | None => SymbolTable::WASM_SYM_UNDEFINED,
            Some(binding) => binding_flags(binding),
        };
        symbols.data(flags, &entry.name, definition);
    }

    // The `linking` section: the symbol table, the segment info, the init functions.
    let mut linking = Vec::new();
    LINKING_VERSION.encode(&mut linking);
    symbols.encode(&mut linking);
    if !plan.segments.is_empty() {
        let mut info = Vec::new();
        (plan.segments.len() as u32).encode(&mut info);
        for segment in &plan.segments {
            segment.name.encode(&mut info);
            segment.align.trailing_zeros().encode(&mut info);
            let flags = if segment.retain {
                WASM_SEG_FLAG_RETAIN
            } else {
                0
            };
            flags.encode(&mut info);
        }
        subsection(&mut linking, WASM_SEGMENT_INFO, &info);
    }
    if let Some(start) = unit.start() {
        let mut init = Vec::new();
        1u32.encode(&mut init);
        START_PRIORITY.encode(&mut init);
        plan.function_symbol(start).encode(&mut init);
        subsection(&mut linking, WASM_INIT_FUNCS, &init);
    }
    out.push(&CustomSection {
        name: Cow::Borrowed("linking"),
        data: Cow::Owned(linking),
    });

    // The relocations.
    if let Some(section) = code_section
        && !code_relocs.is_empty()
    {
        out.push(&reloc_section("reloc.CODE", section, &code_relocs));
    }
    if let Some(section) = data_section
        && !data_relocs.is_empty()
    {
        out.push(&reloc_section("reloc.DATA", section, &data_relocs));
    }

    out.push(&sections::producers_section());

    // The features.
    let mutable_globals = desc.globals().iter().enumerate().any(|(i, global)| {
        global.mutable
            && (global.entity.is_import()
                || exported(ExportTarget::Global(GlobalId::new(i as u32))))
    });
    let overlong = fixups(desc).any(|fixup| matches!(fixup.kind, FixupKind::TableNumber(_)));
    let mut names: Vec<&str> = desc
        .features
        .iter()
        .filter(|feature| *feature != Feature::MutableGlobals || mutable_globals)
        .map(Feature::name)
        .collect();
    if mutable_globals {
        names.push(Feature::MutableGlobals.name());
    }
    if desc.features.contains(Feature::BulkMemory) {
        names.push("bulk-memory-opt");
    }
    if overlong {
        names.push("call-indirect-overlong");
    }
    if plan.tables.len() > 1 {
        names.push(Feature::ReferenceTypes.name());
    }
    names.sort_unstable();
    names.dedup();
    if let Some(features) = features_section(&names) {
        out.push(&features);
    }

    Ok(out.module.finish())
}
