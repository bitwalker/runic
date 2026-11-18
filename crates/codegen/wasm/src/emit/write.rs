//! The complete-module writer: a [ModuleDesc] becomes the bytes of a core WebAssembly module.
//!
//! The writer assigns each index space (imports first, in description order, then definitions in
//! description order), replaces every placeholder with the shortest encoding of its resolved value,
//! and writes the sections in the order WebAssembly requires, then the custom sections of
//! [super::sections]. An empty section is not written, and neither is a data entry that has no
//! segment ([DataEntry::has_segment](super::desc::DataEntry::has_segment): zeroed data, or an
//! address with no bytes); the data-count section counts the segments written. It does not
//! validate the result; see [super::validate].

use alloc::{
    borrow::Cow,
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use runic_ir::Report;
use wasm_encoder::{
    CodeSection, DataCountSection, DataSection, ElementSection, Elements, Encode, EntityType,
    ExportKind, ExportSection, FunctionSection, GlobalSection, GlobalType, ImportSection,
    MemorySection, MemoryType, Module, TableSection, TableType, TypeSection,
};

use super::{
    desc::{
        CODE_PLACEHOLDER, DATA_PLACEHOLDER, DataId, DataPlacement, ElemId, ElemSegment, Entity,
        ExportTarget, Fixup, FixupKind, FuncId, GlobalId, Limits, MemoryId, ModuleDesc, TableId,
        TypeId,
    },
    sections,
};

/// What only a layout knows, which the writer needs: the address of each data entry that has one
/// and the slot in the indirect function table of each function that has one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resolved {
    addresses: BTreeMap<DataId, u32>,
    slots: BTreeMap<FuncId, u32>,
}

impl Resolved {
    /// Nothing resolved.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gives the data entry `data` the address `address`.
    pub fn set_data_address(&mut self, data: DataId, address: u32) {
        self.addresses.insert(data, address);
    }

    /// Gives the function `function` the slot `slot` in the indirect function table.
    pub fn set_table_slot(&mut self, function: FuncId, slot: u32) {
        self.slots.insert(function, slot);
    }

    /// The address of the data entry `data`, when it has one.
    pub fn data_address(&self, data: DataId) -> Option<u32> {
        self.addresses.get(&data).copied()
    }

    /// The slot of the function `function` in the indirect function table, when it has one.
    pub fn table_slot(&self, function: FuncId) -> Option<u32> {
        self.slots.get(&function).copied()
    }
}

/// The final index of each entity of a description: for each kind, imports first in description
/// order, then definitions in description order. Types keep their interned order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexSpaces {
    functions: Vec<u32>,
    tables: Vec<u32>,
    memories: Vec<u32>,
    globals: Vec<u32>,
    types: u32,
    data: Vec<Option<u32>>,
    segments: u32,
}

/// The index of each entity of one kind, given whether each is an import: imports first, then
/// definitions, each in the order given. The object writer numbers its entities the same way.
pub(super) fn index_space(imports: impl Iterator<Item = bool> + Clone) -> Vec<u32> {
    let mut next_import = 0u32;
    let mut next_definition = imports.clone().filter(|import| *import).count() as u32;
    imports
        .map(|import| {
            let next = if import {
                &mut next_import
            } else {
                &mut next_definition
            };
            let index = *next;
            *next += 1;
            index
        })
        .collect()
}

impl IndexSpaces {
    /// The index spaces of `desc`.
    pub fn new(desc: &ModuleDesc) -> Self {
        Self {
            functions: index_space(desc.functions().iter().map(|f| f.entity.is_import())),
            tables: index_space(desc.tables().iter().map(|t| t.entity.is_import())),
            memories: index_space(desc.memories().iter().map(|m| m.entity.is_import())),
            globals: index_space(desc.globals().iter().map(|g| g.entity.is_import())),
            types: desc.types().len() as u32,
            data: desc
                .data()
                .iter()
                .scan(0u32, |next, entry| {
                    Some(entry.has_segment().then(|| {
                        *next += 1;
                        *next - 1
                    }))
                })
                .collect(),
            segments: desc.data().iter().filter(|entry| entry.has_segment()).count() as u32,
        }
    }

    /// The index of the function `id`, when it is a function of the description.
    pub fn function(&self, id: FuncId) -> Option<u32> {
        self.functions.get(id.index() as usize).copied()
    }

    /// The index of the table `id`, when it is a table of the description.
    pub fn table(&self, id: TableId) -> Option<u32> {
        self.tables.get(id.index() as usize).copied()
    }

    /// The index of the memory `id`, when it is a memory of the description.
    pub fn memory(&self, id: MemoryId) -> Option<u32> {
        self.memories.get(id.index() as usize).copied()
    }

    /// The index of the global `id`, when it is a global of the description.
    pub fn global(&self, id: GlobalId) -> Option<u32> {
        self.globals.get(id.index() as usize).copied()
    }

    /// The index of the type `id`, when it is a type of the description.
    pub fn ty(&self, id: TypeId) -> Option<u32> {
        (id.index() < self.types).then_some(id.index())
    }

    /// The index of the data segment of the entry `id`, when it is an entry of the description
    /// that [has a segment](super::desc::DataEntry::has_segment): the entries that have one are
    /// numbered in description order.
    pub fn data(&self, id: DataId) -> Option<u32> {
        self.data.get(id.index() as usize).copied().flatten()
    }

    /// The number of data segments: the entries of the description that have one.
    pub fn data_segments(&self) -> u32 {
        self.segments
    }
}

/// Where in a description the writer found what it reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Site {
    /// The body of a function, or its declaration.
    Function {
        /// The function.
        id: FuncId,
        /// Its name.
        name: String,
    },
    /// The initializer of a global, or its declaration.
    Global {
        /// The global.
        id: GlobalId,
        /// Its name.
        name: String,
    },
    /// A data entry.
    Data {
        /// The entry.
        id: DataId,
        /// Its name.
        name: String,
    },
    /// An element segment.
    Element(ElemId),
    /// An export, by its name.
    Export(String),
}

impl fmt::Display for Site {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (what, name) = match self {
            Self::Function { id, name } => (id.to_string(), name),
            Self::Global { id, name } => (id.to_string(), name),
            Self::Data { id, name } => (id.to_string(), name),
            Self::Element(id) => return write!(f, "{id}"),
            Self::Export(name) => return write!(f, "the export `{name}`"),
        };
        if name.is_empty() {
            f.write_str(&what)
        } else {
            write!(f, "{what} (`{name}`)")
        }
    }
}

/// A description the writer cannot write. Each is a defect of the code that built the
/// description, not of the IR it was built from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteError {
    /// A value that only the layout knows is missing from [Resolved]: the address of a data entry
    /// or the table slot of a function.
    Unresolved {
        /// Where the value is needed.
        site: Site,
        /// The value: a [FixupKind::MemoryAddr] or a [FixupKind::TableSlot].
        target: FixupKind,
    },
    /// The bytes at a fix-up's offset are not a placeholder of the form its place requires, or
    /// overlap the placeholder of an earlier fix-up.
    MisplacedPlaceholder {
        /// Where the fix-up is.
        site: Site,
        /// The fix-up.
        fixup: Fixup,
    },
    /// A fix-up that cannot be where it is: an index in data, or an index with an addend.
    UnsupportedFixup {
        /// Where the fix-up is.
        site: Site,
        /// The fix-up.
        fixup: Fixup,
    },
    /// An id that names no entity of the description.
    UnknownEntity {
        /// Where the id is.
        site: Site,
        /// The id, as text.
        entity: String,
    },
    /// An active data segment, but the module has no memory.
    NoMemory {
        /// The data entry.
        site: Site,
    },
    /// A relocation of an object whose offset is not at a placeholder of its width in the
    /// section it applies to.
    MisplacedRelocation {
        /// The section, as the relocation section names it (`CODE` or `DATA`).
        section: String,
        /// The offset, from the start of the section's payload.
        offset: u32,
    },
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unresolved { site, target } => {
                write!(f, "{site} needs {target}, which the layout did not resolve")
            }
            Self::MisplacedPlaceholder { site, fixup } => write!(
                f,
                "{site} has a fix-up for {} at offset {} where there is no placeholder for it",
                fixup.kind, fixup.offset
            ),
            Self::UnsupportedFixup { site, fixup } => write!(
                f,
                "{site} has a fix-up for {} with addend {} at offset {}, which cannot be there",
                fixup.kind, fixup.addend, fixup.offset
            ),
            Self::UnknownEntity { site, entity } => {
                write!(f, "{site} names {entity}, which the module does not have")
            }
            Self::NoMemory { site } => {
                write!(f, "{site} is an active data segment, but the module has no memory")
            }
            Self::MisplacedRelocation { section, offset } => write!(
                f,
                "a relocation of the section {section} is at offset {offset}, where there is no \
                 placeholder of its width"
            ),
        }
    }
}

impl core::error::Error for WriteError {}

impl From<WriteError> for Report {
    fn from(error: WriteError) -> Self {
        Report::msg(format!(
            "the WebAssembly module could not be written, which is a defect of the backend: \
             {error}"
        ))
    }
}

/// Resolves fix-ups against a description, its index spaces and a layout.
struct Resolver<'a> {
    desc: &'a ModuleDesc,
    spaces: &'a IndexSpaces,
    resolved: &'a Resolved,
}

impl Resolver<'_> {
    /// The address of the data entry `id`: the offset of a fixed segment, or the layout's.
    fn address(&self, id: DataId, site: &Site) -> Result<u32, WriteError> {
        let entry =
            self.desc
                .data()
                .get(id.index() as usize)
                .ok_or_else(|| WriteError::UnknownEntity {
                    site: site.clone(),
                    entity: id.to_string(),
                })?;
        match entry.placement {
            DataPlacement::Fixed(offset) => Ok(offset),
            DataPlacement::Layout | DataPlacement::Passive | DataPlacement::Symbol => {
                self.resolved.data_address(id).ok_or_else(|| WriteError::Unresolved {
                    site: site.clone(),
                    target: FixupKind::MemoryAddr(id),
                })
            }
        }
    }

    /// The value of a fix-up that is a value (an address or a table slot), its addend included.
    fn value(&self, fixup: &Fixup, site: &Site) -> Result<i32, WriteError> {
        let base = match fixup.kind {
            FixupKind::MemoryAddr(id) => self.address(id, site)?,
            FixupKind::TableSlot(id) => {
                self.spaces.function(id).ok_or_else(|| unknown(site, id))?;
                self.resolved.table_slot(id).ok_or_else(|| WriteError::Unresolved {
                    site: site.clone(),
                    target: fixup.kind,
                })?
            }
            _ => {
                return Err(WriteError::UnsupportedFixup {
                    site: site.clone(),
                    fixup: *fixup,
                });
            }
        };
        Ok(base.wrapping_add(fixup.addend as u32) as i32)
    }

    /// The index a fix-up for an index stands for.
    fn index(&self, fixup: &Fixup, site: &Site) -> Result<u32, WriteError> {
        if fixup.addend != 0 {
            return Err(WriteError::UnsupportedFixup {
                site: site.clone(),
                fixup: *fixup,
            });
        }
        let spaces = self.spaces;
        match fixup.kind {
            FixupKind::FunctionIndex(id) => spaces.function(id).ok_or_else(|| unknown(site, id)),
            FixupKind::TypeIndex(id) | FixupKind::BlockType(id) => {
                spaces.ty(id).ok_or_else(|| unknown(site, id))
            }
            FixupKind::GlobalIndex(id) => spaces.global(id).ok_or_else(|| unknown(site, id)),
            FixupKind::TableNumber(id) => spaces.table(id).ok_or_else(|| unknown(site, id)),
            FixupKind::DataIndex(id) => spaces.data(id).ok_or_else(|| unknown(site, id)),
            FixupKind::MemoryAddr(_) | FixupKind::TableSlot(_) => {
                Err(WriteError::UnsupportedFixup {
                    site: site.clone(),
                    fixup: *fixup,
                })
            }
        }
    }

    /// `code` with each placeholder of `fixups` replaced by the shortest encoding of its value.
    fn code(&self, code: &[u8], fixups: &[Fixup], site: &Site) -> Result<Vec<u8>, WriteError> {
        let mut fixups = fixups.to_vec();
        fixups.sort_by_key(|fixup| fixup.offset);
        let mut out = Vec::with_capacity(code.len());
        let mut copied = 0usize;
        for fixup in &fixups {
            let start = fixup.offset as usize;
            let end = start + CODE_PLACEHOLDER.len();
            if start < copied || code.get(start..end) != Some(&CODE_PLACEHOLDER[..]) {
                return Err(WriteError::MisplacedPlaceholder {
                    site: site.clone(),
                    fixup: *fixup,
                });
            }
            out.extend_from_slice(&code[copied..start]);
            match fixup.kind {
                FixupKind::MemoryAddr(_) | FixupKind::TableSlot(_) => {
                    self.value(fixup, site)?.encode(&mut out)
                }
                FixupKind::BlockType(_) => i64::from(self.index(fixup, site)?).encode(&mut out),
                _ => self.index(fixup, site)?.encode(&mut out),
            }
            copied = end;
        }
        out.extend_from_slice(&code[copied..]);
        Ok(out)
    }

    /// `bytes` with each placeholder of `fixups` replaced by its value, four bytes little-endian.
    fn data(&self, bytes: &[u8], fixups: &[Fixup], site: &Site) -> Result<Vec<u8>, WriteError> {
        let mut out = bytes.to_vec();
        let mut fixups = fixups.to_vec();
        fixups.sort_by_key(|fixup| fixup.offset);
        let mut written = 0usize;
        for fixup in &fixups {
            let start = fixup.offset as usize;
            let end = start + DATA_PLACEHOLDER.len();
            if start < written || bytes.get(start..end) != Some(&DATA_PLACEHOLDER[..]) {
                return Err(WriteError::MisplacedPlaceholder {
                    site: site.clone(),
                    fixup: *fixup,
                });
            }
            let value = self.value(fixup, site)?;
            out[start..end].copy_from_slice(&value.to_le_bytes());
            written = end;
        }
        Ok(out)
    }
}

/// The error for an id that names no entity.
fn unknown(site: &Site, id: impl fmt::Display) -> WriteError {
    WriteError::UnknownEntity {
        site: site.clone(),
        entity: id.to_string(),
    }
}

/// The table type of a table of `element`s with `limits`.
pub(super) fn table_type(element: super::desc::RefType, limits: Limits) -> TableType {
    TableType {
        element_type: element.to_encoder(),
        table64: false,
        minimum: limits.min.into(),
        maximum: limits.max.map(Into::into),
        shared: false,
    }
}

fn memory_type(limits: Limits) -> MemoryType {
    MemoryType {
        minimum: limits.min.into(),
        maximum: limits.max.map(Into::into),
        memory64: false,
        shared: false,
        page_size_log2: None,
    }
}

/// `desc` written as a complete core WebAssembly module, with the addresses and table slots of
/// `resolved`.
///
/// The bytes are not validated (see [super::validate::validate]). An error is a defect of the code
/// that built `desc` or `resolved`: a fix-up whose value `resolved` does not give, a placeholder
/// that is not where its fix-up says, or an id that is not of `desc`.
pub fn write_module(desc: &ModuleDesc, resolved: &Resolved) -> Result<Vec<u8>, WriteError> {
    let spaces = IndexSpaces::new(desc);
    let resolver = Resolver {
        desc,
        spaces: &spaces,
        resolved,
    };
    let function_site = |id: FuncId| Site::Function {
        id,
        name: desc.function(id).name.clone(),
    };
    let type_index =
        |id: TypeId, site: &dyn Fn() -> Site| spaces.ty(id).ok_or_else(|| unknown(&site(), id));

    let mut module = Module::new();

    // Types.
    let mut types = TypeSection::new();
    for ty in desc.types() {
        types.ty().function(
            ty.params.iter().map(|ty| ty.to_encoder()),
            ty.results.iter().map(|ty| ty.to_encoder()),
        );
    }
    if !types.is_empty() {
        module.section(&types);
    }

    // Imports: functions, tables, memories, globals; each kind in description order.
    let mut imports = ImportSection::new();
    for (i, function) in desc.functions().iter().enumerate() {
        if let Entity::Import(import) = &function.entity {
            let id = FuncId::new(i as u32);
            let ty = type_index(function.ty, &|| function_site(id))?;
            imports.import(&import.module, &import.field, EntityType::Function(ty));
        }
    }
    for table in desc.tables() {
        if let Entity::Import(import) = &table.entity {
            imports.import(&import.module, &import.field, table_type(table.element, table.limits));
        }
    }
    for memory in desc.memories() {
        if let Entity::Import(import) = &memory.entity {
            imports.import(&import.module, &import.field, memory_type(memory.limits));
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
    if !imports.is_empty() {
        module.section(&imports);
    }

    // Functions.
    let mut functions = FunctionSection::new();
    for (i, function) in desc.functions().iter().enumerate() {
        if !function.entity.is_import() {
            let id = FuncId::new(i as u32);
            functions.function(type_index(function.ty, &|| function_site(id))?);
        }
    }
    if !functions.is_empty() {
        module.section(&functions);
    }

    // Tables.
    let mut tables = TableSection::new();
    for table in desc.tables().iter().filter(|table| !table.entity.is_import()) {
        tables.table(table_type(table.element, table.limits));
    }
    if !tables.is_empty() {
        module.section(&tables);
    }

    // Memories.
    let mut memories = MemorySection::new();
    for memory in desc.memories().iter().filter(|memory| !memory.entity.is_import()) {
        memories.memory(memory_type(memory.limits));
    }
    if !memories.is_empty() {
        module.section(&memories);
    }

    // Globals.
    let mut globals = GlobalSection::new();
    for (i, global) in desc.globals().iter().enumerate() {
        if let Entity::Definition(init) = &global.entity {
            let site = Site::Global {
                id: GlobalId::new(i as u32),
                name: global.name.clone(),
            };
            let code = resolver.code(&init.code, &init.fixups, &site)?;
            let ty = GlobalType {
                val_type: global.ty.to_encoder(),
                mutable: global.mutable,
                shared: false,
            };
            globals.global(ty, &wasm_encoder::ConstExpr::raw(code));
        }
    }
    if !globals.is_empty() {
        module.section(&globals);
    }

    // Exports.
    let mut exports = ExportSection::new();
    for export in desc.exports() {
        let site = || Site::Export(export.name.clone());
        let (kind, index) = match export.target {
            ExportTarget::Function(id) => {
                (ExportKind::Func, spaces.function(id).ok_or_else(|| unknown(&site(), id))?)
            }
            ExportTarget::Table(id) => {
                (ExportKind::Table, spaces.table(id).ok_or_else(|| unknown(&site(), id))?)
            }
            ExportTarget::Memory(id) => {
                (ExportKind::Memory, spaces.memory(id).ok_or_else(|| unknown(&site(), id))?)
            }
            ExportTarget::Global(id) => {
                (ExportKind::Global, spaces.global(id).ok_or_else(|| unknown(&site(), id))?)
            }
        };
        exports.export(&export.name, kind, index);
    }
    if !exports.is_empty() {
        module.section(&exports);
    }

    // Elements.
    let mut elements = ElementSection::new();
    for (i, segment) in desc.elements().iter().enumerate() {
        let site = Site::Element(ElemId::new(i as u32));
        let indices = |functions: &[FuncId]| {
            functions
                .iter()
                .map(|id| spaces.function(*id).ok_or_else(|| unknown(&site, *id)))
                .collect::<Result<Vec<_>, _>>()
        };
        match segment {
            ElemSegment::Active {
                table,
                offset,
                functions,
            } => {
                let table = spaces.table(*table).ok_or_else(|| unknown(&site, *table))?;
                let functions = indices(functions)?;
                // Table 0 takes the MVP encoding, which needs no feature.
                let table = (table != 0).then_some(table);
                let offset = wasm_encoder::ConstExpr::i32_const(*offset as i32);
                elements.active(table, &offset, Elements::Functions(Cow::Owned(functions)));
            }
            ElemSegment::Declarative(functions) => {
                let functions = indices(functions)?;
                elements.declared(Elements::Functions(Cow::Owned(functions)));
            }
        }
    }
    if !elements.is_empty() {
        module.section(&elements);
    }

    // The data count.
    if desc.uses_data_count {
        module.section(&DataCountSection {
            count: spaces.data_segments(),
        });
    }

    // Code.
    let mut code = CodeSection::new();
    for (i, function) in desc.functions().iter().enumerate() {
        if let Entity::Definition(body) = &function.entity {
            let site = function_site(FuncId::new(i as u32));
            let bytes = resolver.code(&body.code, &body.fixups, &site)?;
            let mut encoded = wasm_encoder::Function::new_with_locals_types(
                body.locals.iter().map(|ty| ty.to_encoder()),
            );
            encoded.raw(bytes);
            code.function(&encoded);
        }
    }
    if !code.is_empty() {
        module.section(&code);
    }

    // Data.
    let mut data = DataSection::new();
    for (i, entry) in desc.data().iter().enumerate() {
        let id = DataId::new(i as u32);
        let site = Site::Data {
            id,
            name: entry.name.clone(),
        };
        if !entry.has_segment() {
            continue;
        }
        let bytes = resolver.data(&entry.bytes, &entry.fixups, &site)?;
        match entry.placement {
            DataPlacement::Passive => {
                data.passive(bytes);
            }
            DataPlacement::Symbol => {}
            DataPlacement::Layout | DataPlacement::Fixed(_) => {
                if desc.memories().is_empty() {
                    return Err(WriteError::NoMemory { site });
                }
                let address = resolver.address(id, &site)?;
                let offset = wasm_encoder::ConstExpr::i32_const(address as i32);
                data.active(0, &offset, bytes);
            }
        }
    }
    if !data.is_empty() {
        module.section(&data);
    }

    sections::write_custom_sections(&mut module, desc, &spaces);
    Ok(module.finish())
}
