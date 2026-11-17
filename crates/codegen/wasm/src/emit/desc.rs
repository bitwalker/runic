//! The module description: what the emitter knows about the module it builds, before any entity
//! has its final index.
//!
//! A description holds types, functions with their encoded bodies, tables, memories, globals, data
//! entries, element segments, exports, names and the features used. Each entity is named by a typed
//! id ([FuncId], [GlobalId], ...) that indexes its own list in the order of creation; it is not the
//! entity's index in the Wasm module. Wherever a body, a constant expression or a data entry refers
//! to another entity, it holds a placeholder and a [Fixup] that says what the placeholder stands
//! for. The writer ([super::write]) assigns the index spaces and replaces each placeholder; the
//! object writer of part 6e keeps the placeholders and turns the fix-ups into relocations.

use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::fmt;

use wasm_encoder::{Encode, Instruction, InstructionSink};

macro_rules! define_id {
    ($(#[$doc:meta])* $name:ident, $what:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            /// The id with position `index` in its list.
            pub(crate) const fn new(index: u32) -> Self {
                Self(index)
            }

            /// The position of the entity in its list of the description (not its index in the
            /// Wasm module).
            pub const fn index(self) -> u32 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($what, " #{}"), self.0)
            }
        }
    };
}

define_id!(
    /// A function type of a description: its position in the list of interned types.
    TypeId,
    "type"
);
define_id!(
    /// A function of a description, imported or defined.
    FuncId,
    "function"
);
define_id!(
    /// A table of a description, imported or defined.
    TableId,
    "table"
);
define_id!(
    /// A memory of a description, imported or defined.
    MemoryId,
    "memory"
);
define_id!(
    /// A global of a description, imported or defined.
    GlobalId,
    "global"
);
define_id!(
    /// A data entry of a description.
    DataId,
    "data entry"
);
define_id!(
    /// An element segment of a description.
    ElemId,
    "element segment"
);

/// A WebAssembly value type, as far as this backend uses them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValType {
    /// `i32`.
    I32,
    /// `i64`.
    I64,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
    /// `funcref`, a nullable reference to a function.
    FuncRef,
    /// `externref`, a nullable reference to a host value.
    ExternRef,
}

impl ValType {
    /// The value type as `wasm-encoder` spells it.
    pub fn to_encoder(self) -> wasm_encoder::ValType {
        match self {
            Self::I32 => wasm_encoder::ValType::I32,
            Self::I64 => wasm_encoder::ValType::I64,
            Self::F32 => wasm_encoder::ValType::F32,
            Self::F64 => wasm_encoder::ValType::F64,
            Self::FuncRef => wasm_encoder::ValType::FUNCREF,
            Self::ExternRef => wasm_encoder::ValType::EXTERNREF,
        }
    }
}

impl From<RefType> for ValType {
    fn from(ty: RefType) -> Self {
        match ty {
            RefType::FuncRef => Self::FuncRef,
            RefType::ExternRef => Self::ExternRef,
        }
    }
}

/// The element type of a table: a reference type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RefType {
    /// `funcref`.
    FuncRef,
    /// `externref`.
    ExternRef,
}

impl RefType {
    /// The reference type as `wasm-encoder` spells it.
    pub fn to_encoder(self) -> wasm_encoder::RefType {
        match self {
            Self::FuncRef => wasm_encoder::RefType::FUNCREF,
            Self::ExternRef => wasm_encoder::RefType::EXTERNREF,
        }
    }
}

/// A function type: its parameters and its results, each list in order.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FuncType {
    /// The types of the parameters.
    pub params: Vec<ValType>,
    /// The types of the results.
    pub results: Vec<ValType>,
}

impl FuncType {
    /// The function type `(params) -> (results)`.
    pub fn new(
        params: impl IntoIterator<Item = ValType>,
        results: impl IntoIterator<Item = ValType>,
    ) -> Self {
        Self {
            params: params.into_iter().collect(),
            results: results.into_iter().collect(),
        }
    }
}

/// Where an imported entity comes from: the module and field names of the import.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Import {
    /// The module name of the import (`env`, for example).
    pub module: String,
    /// The field name of the import.
    pub field: String,
}

impl Import {
    /// The import of `field` from `module`.
    pub fn new(module: impl Into<String>, field: impl Into<String>) -> Self {
        Self {
            module: module.into(),
            field: field.into(),
        }
    }
}

/// An entity is either imported or defined by the module; `D` is what a definition holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entity<D> {
    /// The entity is imported. Imports take the lower indices of their index space.
    Import(Import),
    /// The entity is defined by the module, with `D`.
    Definition(D),
}

impl<D> Entity<D> {
    /// The import, when the entity is imported.
    pub fn import(&self) -> Option<&Import> {
        match self {
            Self::Import(import) => Some(import),
            Self::Definition(_) => None,
        }
    }

    /// The definition, when the entity is defined.
    pub fn definition(&self) -> Option<&D> {
        match self {
            Self::Import(_) => None,
            Self::Definition(definition) => Some(definition),
        }
    }

    /// The definition, mutably, when the entity is defined.
    pub fn definition_mut(&mut self) -> Option<&mut D> {
        match self {
            Self::Import(_) => None,
            Self::Definition(definition) => Some(definition),
        }
    }

    /// Whether the entity is imported.
    pub fn is_import(&self) -> bool {
        matches!(self, Self::Import(_))
    }
}

/// The limits of a table (in elements) or a memory (in 64 KiB pages).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Limits {
    /// The initial size.
    pub min: u32,
    /// The maximum size, if there is one.
    pub max: Option<u32>,
}

impl Limits {
    /// The limits `min` with no maximum.
    pub const fn at_least(min: u32) -> Self {
        Self { min, max: None }
    }
}

/// A function: its type, its name, and either an import or a body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    /// The function's type.
    pub ty: TypeId,
    /// The function's name in the `name` section; an empty name is not written.
    pub name: String,
    /// The import, or the body.
    pub entity: Entity<FunctionBody>,
}

/// The body of a defined function.
///
/// `code` is the instruction bytes of the body including the final `end`, without the
/// declaration of the locals and without a size prefix. Every reference in it to another entity is
/// a placeholder recorded in `fixups`, written through [FunctionBody::writer].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionBody {
    /// The types of the locals beyond the parameters, in order: the first is the local after the
    /// last parameter.
    pub locals: Vec<ValType>,
    /// The encoded instructions, ending with `end`.
    pub code: Vec<u8>,
    /// The fix-ups in `code`, each at the offset of its placeholder from the start of `code`.
    pub fixups: Vec<Fixup>,
    /// Names of locals for the `name` section: the local's index (parameters first) and its name.
    pub local_names: Vec<(u32, String)>,
}

impl FunctionBody {
    /// A writer that appends to `code` and records fix-ups in `fixups`.
    pub fn writer(&mut self) -> CodeWriter<'_> {
        CodeWriter {
            code: &mut self.code,
            fixups: &mut self.fixups,
        }
    }
}

/// A table: its element type, its limits and its name; imported or defined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    /// The type of the table's elements.
    pub element: RefType,
    /// The table's limits, in elements.
    pub limits: Limits,
    /// The table's name in the `name` section; an empty name is not written.
    pub name: String,
    /// The import, or the definition (which holds nothing more).
    pub entity: Entity<()>,
}

/// A memory: its limits in 64 KiB pages and its name; imported or defined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    /// The memory's limits, in pages.
    pub limits: Limits,
    /// The memory's name in the `name` section; an empty name is not written.
    pub name: String,
    /// The import, or the definition (which holds nothing more).
    pub entity: Entity<()>,
}

/// A global: its value type, its mutability and its name; imported, or defined with a constant
/// expression that initializes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Global {
    /// The global's value type.
    pub ty: ValType,
    /// Whether the global is mutable.
    pub mutable: bool,
    /// The global's name in the `name` section; an empty name is not written.
    pub name: String,
    /// The import, or the initializer.
    pub entity: Entity<ConstExpr>,
}

/// A constant expression, such as a global's initializer.
///
/// `code` is the encoded instructions **without** the final `end` (the writer adds it). Every
/// reference in it to another entity is a placeholder recorded in `fixups`, in the same forms as in
/// a function body (see [Fixup]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstExpr {
    /// The encoded instructions, without the final `end`.
    pub code: Vec<u8>,
    /// The fix-ups in `code`, each at the offset of its placeholder from the start of `code`.
    pub fixups: Vec<Fixup>,
}

impl ConstExpr {
    /// A writer that appends to `code` and records fix-ups in `fixups`.
    pub fn writer(&mut self) -> CodeWriter<'_> {
        CodeWriter {
            code: &mut self.code,
            fixups: &mut self.fixups,
        }
    }
}

/// Where a data entry goes in the module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataPlacement {
    /// An active segment in memory 0 at the address the layout gives the entry
    /// ([super::write::Resolved::data_address]).
    Layout,
    /// A passive segment, copied into memory by `memory.init`.
    Passive,
    /// An active segment in memory 0 at a fixed offset (a `builtin.segment`).
    Fixed(u32),
    /// An address the layout gives ([super::write::Resolved::data_address]) and no bytes: a
    /// linker-defined data symbol such as `__heap_base`. It has no segment.
    Symbol,
}

/// A data entry: bytes for linear memory, with where they go, how they are aligned, the section
/// they belong to and the fix-ups in them.
///
/// The writer gives each data entry that [has a segment](DataEntry::has_segment) one data
/// segment, in description order; the data index of an entry is the number of entries before it
/// that have one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataEntry {
    /// The entry's name in the `name` section; an empty name is not written.
    pub name: String,
    /// The bytes.
    pub bytes: Vec<u8>,
    /// The alignment of the entry's address, in bytes (a power of two).
    pub align: u32,
    /// The section the entry belongs to, when it has one: entries of one section are laid out
    /// together.
    pub section: Option<String>,
    /// Where the entry goes.
    pub placement: DataPlacement,
    /// The fix-ups in `bytes`, each at the offset of its four placeholder bytes from the start of
    /// `bytes`. Only [FixupKind::MemoryAddr] and [FixupKind::TableSlot] can be in data.
    pub fixups: Vec<Fixup>,
}

impl DataEntry {
    /// Whether the entry is a passive segment.
    pub fn is_passive(&self) -> bool {
        self.placement == DataPlacement::Passive
    }

    /// Whether a complete module has a data segment for the entry: every passive or fixed entry
    /// does; a [DataPlacement::Layout] entry does unless its bytes are all zero and it has no
    /// fix-up (linear memory starts zeroed); a [DataPlacement::Symbol] never does.
    pub fn has_segment(&self) -> bool {
        match self.placement {
            DataPlacement::Passive | DataPlacement::Fixed(_) => true,
            DataPlacement::Layout => {
                !self.fixups.is_empty() || self.bytes.iter().any(|byte| *byte != 0)
            }
            DataPlacement::Symbol => false,
        }
    }
}

/// An element segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElemSegment {
    /// An active segment: `functions` are written into `table` from slot `offset` at
    /// instantiation.
    Active {
        /// The table the segment initializes.
        table: TableId,
        /// The first slot the segment writes.
        offset: u32,
        /// The functions, in slot order.
        functions: Vec<FuncId>,
    },
    /// A declarative segment: it declares `functions` so that `ref.func` may name them.
    Declarative(Vec<FuncId>),
}

/// What an export exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExportTarget {
    /// A function.
    Function(FuncId),
    /// A table.
    Table(TableId),
    /// A memory.
    Memory(MemoryId),
    /// A global.
    Global(GlobalId),
}

/// An export: the name it has outside the module and what it exports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Export {
    /// The export's name.
    pub name: String,
    /// What is exported.
    pub target: ExportTarget,
}

/// A feature of WebAssembly beyond the MVP that a module may use, by its name in the tool
/// conventions' `target_features` section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Feature {
    /// Imported and exported mutable globals: `mutable-globals`.
    MutableGlobals,
    /// The sign-extension operators: `sign-ext`.
    SignExt,
    /// The saturating float-to-int conversions: `nontrapping-fptoint`.
    NontrappingFptoint,
    /// Bulk memory operations and passive segments: `bulk-memory`.
    BulkMemory,
    /// Multiple results and block parameters: `multivalue`.
    Multivalue,
    /// Reference types and several tables: `reference-types`.
    ReferenceTypes,
    /// Tail calls: `tail-call`.
    TailCall,
}

impl Feature {
    /// Every feature, in the order `target_features` lists them.
    pub const ALL: [Feature; 7] = [
        Self::MutableGlobals,
        Self::SignExt,
        Self::NontrappingFptoint,
        Self::BulkMemory,
        Self::Multivalue,
        Self::ReferenceTypes,
        Self::TailCall,
    ];

    /// The feature's name in the tool conventions (`Linking.md`, "Target Features Section").
    pub const fn name(self) -> &'static str {
        match self {
            Self::MutableGlobals => "mutable-globals",
            Self::SignExt => "sign-ext",
            Self::NontrappingFptoint => "nontrapping-fptoint",
            Self::BulkMemory => "bulk-memory",
            Self::Multivalue => "multivalue",
            Self::ReferenceTypes => "reference-types",
            Self::TailCall => "tail-call",
        }
    }
}

/// A set of [Feature]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FeatureSet(u8);

impl FeatureSet {
    /// The empty set.
    pub const fn new() -> Self {
        Self(0)
    }

    /// Adds `feature` to the set.
    pub fn insert(&mut self, feature: Feature) {
        self.0 |= 1 << feature as u8;
    }

    /// Whether `feature` is in the set.
    pub const fn contains(self, feature: Feature) -> bool {
        self.0 & (1 << feature as u8) != 0
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The features in the set, in the order of [Feature::ALL].
    pub fn iter(self) -> impl Iterator<Item = Feature> {
        Feature::ALL.into_iter().filter(move |feature| self.contains(*feature))
    }
}

impl FromIterator<Feature> for FeatureSet {
    fn from_iter<I: IntoIterator<Item = Feature>>(features: I) -> Self {
        let mut set = Self::new();
        for feature in features {
            set.insert(feature);
        }
        set
    }
}

/// The five bytes of a placeholder in code: zero as a padded LEB128, which reads the same as an
/// unsigned and as a signed number.
pub const CODE_PLACEHOLDER: [u8; 5] = [0x80, 0x80, 0x80, 0x80, 0x00];

/// The four bytes of a placeholder in data: zero, little-endian.
pub const DATA_PLACEHOLDER: [u8; 4] = [0; 4];

/// What a placeholder stands for, with its target.
///
/// The form of the placeholder follows from where the fix-up is. In code (a function body or a
/// constant expression) it is five bytes ([CODE_PLACEHOLDER]): for an index, the immediate is a
/// padded unsigned LEB128 (a padded signed one for [FixupKind::BlockType]); for
/// [FixupKind::MemoryAddr] and [FixupKind::TableSlot] it is the padded signed LEB128 operand of an
/// `i32.const`. In data it is four little-endian bytes ([DATA_PLACEHOLDER]), and only
/// [FixupKind::MemoryAddr] and [FixupKind::TableSlot] can be there. The relocation type of the tool
/// conventions' `Linking.md` that each corresponds to is given with each kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FixupKind {
    /// The index of a function (`call`, `return_call`, `ref.func`): `R_WASM_FUNCTION_INDEX_LEB`.
    FunctionIndex(FuncId),
    /// The index of a type (`call_indirect`, `return_call_indirect`): `R_WASM_TYPE_INDEX_LEB`.
    TypeIndex(TypeId),
    /// The index of a type as the block type of `block`, `loop` or `if`, which is a signed
    /// (`s33`) number: `R_WASM_TYPE_INDEX_LEB`, whose padded form reads the same.
    BlockType(TypeId),
    /// The index of a global (`global.get`, `global.set`): `R_WASM_GLOBAL_INDEX_LEB`.
    GlobalIndex(GlobalId),
    /// The index of a table (the table operand of `call_indirect` and the `table.*`
    /// instructions): `R_WASM_TABLE_NUMBER_LEB`.
    TableNumber(TableId),
    /// The index of a data segment (`memory.init`, `data.drop`). The tool conventions have no
    /// relocation for it.
    DataIndex(DataId),
    /// The address of a data entry plus the addend: `R_WASM_MEMORY_ADDR_SLEB` in code,
    /// `R_WASM_MEMORY_ADDR_I32` in data.
    MemoryAddr(DataId),
    /// The slot of a function in the indirect function table, plus the addend:
    /// `R_WASM_TABLE_INDEX_SLEB` in code, `R_WASM_TABLE_INDEX_I32` in data.
    TableSlot(FuncId),
}

impl FixupKind {
    /// Whether the placeholder of this kind in code is the operand of an `i32.const`, a signed
    /// value that an addend may adjust, rather than an index.
    pub const fn is_value(self) -> bool {
        matches!(self, Self::MemoryAddr(_) | Self::TableSlot(_))
    }
}

impl fmt::Display for FixupKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FunctionIndex(id) => write!(f, "the index of {id}"),
            Self::TypeIndex(id) => write!(f, "the index of {id}"),
            Self::BlockType(id) => write!(f, "the block type {id}"),
            Self::GlobalIndex(id) => write!(f, "the index of {id}"),
            Self::TableNumber(id) => write!(f, "the index of {id}"),
            Self::DataIndex(id) => write!(f, "the segment index of {id}"),
            Self::MemoryAddr(id) => write!(f, "the address of {id}"),
            Self::TableSlot(id) => write!(f, "the table slot of {id}"),
        }
    }
}

/// A placeholder at `offset` that stands for `kind`'s target, plus `addend` (which is zero for an
/// index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fixup {
    /// The offset of the placeholder from the start of the bytes that hold it.
    pub offset: u32,
    /// What the placeholder stands for.
    pub kind: FixupKind,
    /// What is added to the target's value; zero for an index.
    pub addend: i32,
}

/// Appends instructions to encoded code and records the fix-ups of the references in it.
///
/// [CodeWriter::fixup_index] and [CodeWriter::fixup_i32_const] are the only ways a reference to
/// another entity gets into code; the other methods write instructions whose immediates name no
/// entity.
pub struct CodeWriter<'a> {
    code: &'a mut Vec<u8>,
    fixups: &'a mut Vec<Fixup>,
}

impl CodeWriter<'_> {
    /// The number of bytes written so far.
    pub fn len(&self) -> usize {
        self.code.len()
    }

    /// Whether nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Appends `bytes` as they are.
    pub fn raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.code.extend_from_slice(bytes);
        self
    }

    /// Appends `instruction`, encoded by `wasm-encoder`.
    pub fn instruction(&mut self, instruction: &Instruction<'_>) -> &mut Self {
        instruction.encode(self.code);
        self
    }

    /// A `wasm-encoder` instruction sink that appends to the code.
    pub fn sink(&mut self) -> InstructionSink<'_> {
        InstructionSink::new(self.code)
    }

    /// Appends `opcode` (its bytes, which may be empty for a second immediate of one instruction),
    /// then a placeholder for `kind`, and records the fix-up at the placeholder with addend zero.
    pub fn fixup_index(&mut self, opcode: &[u8], kind: FixupKind) -> &mut Self {
        self.code.extend_from_slice(opcode);
        self.placeholder(kind, 0)
    }

    /// Appends an `i32.const` whose operand is a placeholder for `kind` plus `addend`, and records
    /// the fix-up.
    pub fn fixup_i32_const(&mut self, kind: FixupKind, addend: i32) -> &mut Self {
        self.code.push(0x41);
        self.placeholder(kind, addend)
    }

    fn placeholder(&mut self, kind: FixupKind, addend: i32) -> &mut Self {
        let offset = u32::try_from(self.code.len()).expect("code larger than 4 GiB");
        self.fixups.push(Fixup {
            offset,
            kind,
            addend,
        });
        self.code.extend_from_slice(&CODE_PLACEHOLDER);
        self
    }
}

/// The description of a module: everything the writer needs to write it, with every reference
/// between entities recorded as a fix-up.
///
/// Each list is in order of creation, and each id returned by an `add_*` method indexes its list.
/// Function types are interned: [ModuleDesc::intern_type] returns one id for equal types.
#[derive(Clone, Debug, Default)]
pub struct ModuleDesc {
    /// The module's name for the `name` section.
    pub name: Option<String>,
    /// Whether the writer writes a `name` section.
    pub names: bool,
    /// The features the module uses, listed in the `target_features` section.
    pub features: FeatureSet,
    /// Whether some body uses `memory.init` or `data.drop`, which requires the data count
    /// section.
    pub uses_data_count: bool,
    types: Vec<FuncType>,
    interned: BTreeMap<FuncType, TypeId>,
    functions: Vec<Function>,
    tables: Vec<Table>,
    memories: Vec<Memory>,
    globals: Vec<Global>,
    data: Vec<DataEntry>,
    elements: Vec<ElemSegment>,
    exports: Vec<Export>,
}

/// The position of the next element of a list of `len` elements.
fn next_id(len: usize) -> u32 {
    u32::try_from(len).expect("more than 2^32 entities of one kind")
}

impl ModuleDesc {
    /// An empty description.
    pub fn new() -> Self {
        Self::default()
    }

    /// The id of `ty`, interning it if no equal type has been interned.
    pub fn intern_type(&mut self, ty: FuncType) -> TypeId {
        if let Some(id) = self.interned.get(&ty) {
            return *id;
        }
        let id = TypeId::new(next_id(self.types.len()));
        self.types.push(ty.clone());
        self.interned.insert(ty, id);
        id
    }

    /// The interned types, in order of interning.
    pub fn types(&self) -> &[FuncType] {
        &self.types
    }

    /// The type `id`.
    ///
    /// # Panics
    ///
    /// When `id` is not a type of this description.
    pub fn ty(&self, id: TypeId) -> &FuncType {
        &self.types[id.0 as usize]
    }

    /// Adds `function` and returns its id.
    pub fn add_function(&mut self, function: Function) -> FuncId {
        let id = FuncId::new(next_id(self.functions.len()));
        self.functions.push(function);
        id
    }

    /// The functions, in order of creation.
    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    /// The function `id`.
    ///
    /// # Panics
    ///
    /// When `id` is not a function of this description.
    pub fn function(&self, id: FuncId) -> &Function {
        &self.functions[id.0 as usize]
    }

    /// The function `id`, mutably (to give it its body once its callees have ids, for example).
    ///
    /// # Panics
    ///
    /// When `id` is not a function of this description.
    pub fn function_mut(&mut self, id: FuncId) -> &mut Function {
        &mut self.functions[id.0 as usize]
    }

    /// Adds `table` and returns its id.
    pub fn add_table(&mut self, table: Table) -> TableId {
        let id = TableId::new(next_id(self.tables.len()));
        self.tables.push(table);
        id
    }

    /// The tables, in order of creation.
    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    /// The table `id`, mutably.
    ///
    /// # Panics
    ///
    /// When `id` is not a table of this description.
    pub fn table_mut(&mut self, id: TableId) -> &mut Table {
        &mut self.tables[id.0 as usize]
    }

    /// Adds `memory` and returns its id.
    pub fn add_memory(&mut self, memory: Memory) -> MemoryId {
        let id = MemoryId::new(next_id(self.memories.len()));
        self.memories.push(memory);
        id
    }

    /// The memories, in order of creation.
    pub fn memories(&self) -> &[Memory] {
        &self.memories
    }

    /// The memory `id`, mutably.
    ///
    /// # Panics
    ///
    /// When `id` is not a memory of this description.
    pub fn memory_mut(&mut self, id: MemoryId) -> &mut Memory {
        &mut self.memories[id.0 as usize]
    }

    /// Adds `global` and returns its id.
    pub fn add_global(&mut self, global: Global) -> GlobalId {
        let id = GlobalId::new(next_id(self.globals.len()));
        self.globals.push(global);
        id
    }

    /// The globals, in order of creation.
    pub fn globals(&self) -> &[Global] {
        &self.globals
    }

    /// The global `id`, mutably.
    ///
    /// # Panics
    ///
    /// When `id` is not a global of this description.
    pub fn global_mut(&mut self, id: GlobalId) -> &mut Global {
        &mut self.globals[id.0 as usize]
    }

    /// Adds `entry` and returns its id.
    pub fn add_data(&mut self, entry: DataEntry) -> DataId {
        let id = DataId::new(next_id(self.data.len()));
        self.data.push(entry);
        id
    }

    /// The data entries, in order of creation.
    pub fn data(&self) -> &[DataEntry] {
        &self.data
    }

    /// The data entry `id`, mutably.
    ///
    /// # Panics
    ///
    /// When `id` is not a data entry of this description.
    pub fn data_mut(&mut self, id: DataId) -> &mut DataEntry {
        &mut self.data[id.0 as usize]
    }

    /// Adds `segment` and returns its id.
    pub fn add_element(&mut self, segment: ElemSegment) -> ElemId {
        let id = ElemId::new(next_id(self.elements.len()));
        self.elements.push(segment);
        id
    }

    /// The element segments, in order of creation.
    pub fn elements(&self) -> &[ElemSegment] {
        &self.elements
    }

    /// The element segment `id`, mutably.
    ///
    /// # Panics
    ///
    /// When `id` is not an element segment of this description.
    pub fn element_mut(&mut self, id: ElemId) -> &mut ElemSegment {
        &mut self.elements[id.0 as usize]
    }

    /// Adds `export`. Exports are written in order of addition; two with one name are not
    /// diagnosed here.
    pub fn add_export(&mut self, export: Export) {
        self.exports.push(export);
    }

    /// The exports, in order of addition.
    pub fn exports(&self) -> &[Export] {
        &self.exports
    }
}
