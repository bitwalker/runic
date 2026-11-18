//! The unit: every `builtin.module` under a root, linked as one WebAssembly module (decisions 1
//! and 3 of the plan).
//!
//! **What is in it.** [collect] walks the root (a `builtin.world`, a `builtin.component` or a
//! `builtin.module`) and takes every `builtin.module` under it at any depth, the root included: a
//! nested module is a namespace, not a separate binary. More than one `builtin.component` under
//! the root is an error, since each component becomes its own binary, which this backend does not
//! produce yet. So is an entity that a `builtin.component` or a `builtin.world` holds directly, not
//! inside a module (a function, an alias, a global variable, a `builtin.segment` or a `wasm`
//! entity): component-level entities are not emitted yet. The operations a module directly holds become entries of the [ModuleDesc], in the
//! order the modules and their symbols are written:
//!
//! - `builtin.function`: a function, its type from its signature (every parameter and result a
//!   Wasm value type, or the error that the function is not in Wasm form); a definition's body is
//!   encoded by [encode_function];
//! - `wasm.global` (its initializer by [constexpr]), `wasm.table`, `wasm.memory`;
//! - `wasm.elem`: an active element segment;
//! - `builtin.global_variable`: a data entry with its name, its `section` and the placement
//!   [DataPlacement::Layout]; a definition's bytes, alignment and fix-ups are its initializer's
//!   ([data::evaluate]), and the layout gives it its place. A declaration that links to no
//!   definition is an *undefined data symbol* ([Unit::undefined_data]), a stub with no bytes;
//! - `builtin.function_alias`: no entity of its own: a use of it is a use of its target;
//! - `builtin.segment`: remembered for the layout ([Unit::segments]);
//! - `wasm.tag`: an error (exceptions are not emitted).
//!
//! **Linking.** What links is the effective link name ([effective_link_name]). A definition that
//! *other modules may name* is one whose visibility is `public` or `internal` (the docs of
//! [Visibility]: a `private` symbol may only be referenced from its own symbol table, an
//! `internal` one from any symbol table of the compilation). A `builtin.function_alias` with such
//! a visibility is a definition of a function under its link name, standing for its target.
//!
//! - Two definitions that other modules may name with one link name are an error naming both,
//!   whatever their kinds: a linker has one namespace.
//! - A declaration of link name `L` is the same entity as the definition of `L`, when there is one
//!   that other modules may name. Its kind must be the declaration's, and its type must satisfy
//!   the declaration's: for a function, the same Wasm function type; for a global, the same value
//!   type and mutability; for a global variable, the same type; for a table, the same element type; for a table or a memory, limits
//!   within the declaration's (a minimum at least the declaration's, and a maximum at most the
//!   declaration's when it has one). Otherwise it is an error naming both.
//! - A declaration of a function, global, table or memory with no such definition is an import:
//!   from the module [import_module] gives (`env` by default), with the field `L`. All
//!   declarations with one module, field and kind are one import, and their types must be equal,
//!   or it is an error naming two of them. One module and field declared as two kinds is an error
//!   naming both declarations: a host module exports one entity under a name, so no host could
//!   provide both. A declared global variable with no definition is one undefined data symbol per
//!   link name, and its declarations must agree in type the same way.
//! - A reference by path resolves through the IR ([SymbolRefAttr::resolve], or
//!   [SymbolRefAttr::resolve_callable] for a function, which follows aliases), whichever module
//!   holds the target, then through a declaration to the entity it stands for.
//!
//! **Exports.** Only what `wasm.export_name` ([export_names]) names is exported: a function, a
//! global, a table or a memory, defined or declared (an import may be exported), or an alias (its
//! target is exported). Two exports with one name are an error naming both operations.
//!
//! **Features.** A mutable global that is imported or exported records mutable-globals. More than
//! one table, a table of `externref`, or a global of a reference type records reference-types.
//!
//! **Names.** Each entity's name is its effective link name (the symbol's name when it has none).
//! A function that carries a `di.subprogram` with parameter names gives them to its parameters.
//!
//! **Declared references.** Every function named by a `wasm.ref_func` in a function body is listed
//! in one declarative element segment, as Wasm requires of a function used by `ref.func`
//! (declaring one that an export or an active segment also declares is valid). There is none when
//! no body has a `wasm.ref_func`.
//!
//! A module may hold at most one memory, definitions and imports together: a second is an error
//! naming both, as multiple memories are outside the backend's feature set.
//!
//! **Symbols.** The unit records what a linker symbol of each entity needs and the description
//! does not hold, for the object writer; the complete-module writer reads none of it. The
//! [binding](Binding) of each function, global, table and data entry follows from the visibility
//! of the operation it was made from (`public` a global symbol, `internal` a hidden one, `private` a
//! local one), and is that of an undefined symbol for an import or an undefined data symbol. An
//! import *names its host* when its first declaration is written with `wasm.import_module`; one
//! written without it is the linker's to resolve. A data entry with a `section` is *retained*
//! ([Unit::is_retained]): a named section exists to be found by name. [Unit::function_export]
//! gives the export name of a function.
//!
//! **The start function.** A function definition written with `wasm.start` ([wasm::is_start]) is
//! the unit's start function ([Unit::start]); a second one is an error naming both. A complete
//! module exports it as `_initialize` ([Unit::export_start]), the convention of a WASI reactor,
//! and has no start section.
//!
//! **What an object cannot express.** [Unit::check_object] refuses, with a diagnostic naming the
//! operation, what a complete module allows and a relocatable object has no form for (see its
//! docs). Only the object path runs it.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use runic_dialect_wasm::{self as wasm, IMPORT_MODULE, export_names, import_module};
use runic_ir::{
    Forward, FxHashMap, Operation, OperationRef, RawWalk, Report, Spanned, SymbolRef, Visibility,
    WalkResult,
    diagnostics::report,
    dialects::{
        builtin::{
            Component, Function as FunctionOp, FunctionAlias, GlobalVariable, Module, Segment,
            World, attributes::SymbolRefAttr,
        },
        debuginfo::attributes::SubprogramAttr,
    },
    effective_link_name, section,
};

use super::{
    EmitOptions,
    body::{BodyConfig, encode_function},
    constexpr, data,
    desc::{
        Binding, ConstExpr, DataEntry, DataId, DataPlacement, ElemSegment, Entity, Export,
        ExportTarget, Feature, FixupKind, FuncId, FuncType, Function, FunctionBody, Global,
        GlobalId, Import, Limits, Memory, MemoryId, ModuleDesc, RefType, Table, TableId, ValType,
    },
    instr::{Entities, record_type_features, unsupported, val_type},
};
use crate::LOWERING_PASSES;

/// The unit as collected: the module description, and for each entity the operation it was made
/// from, the undefined data symbols and the `builtin.segment`s, which the layout reads.
///
/// The well-known symbols the lowering declares (`__linear_memory`, `__indirect_function_table`,
/// `__stack_pointer`) are, when nothing in the unit defines them, imports of the description from
/// `env` with those fields, like any other declaration.
#[derive(Debug)]
pub struct Unit {
    /// The module description: every entity, with function bodies and constant expressions
    /// encoded, exports, element segments, names and features. Data entries are stubs.
    pub desc: ModuleDesc,
    origins: Origins,
    undefined_data: Vec<DataId>,
    segments: Vec<OperationRef>,
    elements: Vec<(OperationRef, TableId)>,
    data_uses: FxHashMap<DataId, DataUses>,
    bindings: Bindings,
    start: Option<FuncId>,
    host_conflict: Option<(OperationRef, OperationRef)>,
    exported: FxHashMap<String, OperationRef>,
    ref_func: Option<OperationRef>,
}

/// The name a complete module exports its start function under: the convention of a WASI
/// reactor, whose embedder calls it once after instantiating the module.
pub const START_EXPORT: &str = "_initialize";

/// The name of the function table the linker defines.
const INDIRECT_FUNCTION_TABLE: &str = crate::symbols::INDIRECT_FUNCTION_TABLE;

/// The name of the stack pointer the linker defines.
const STACK_POINTER: &str = crate::symbols::STACK_POINTER;

/// The binding of each entity of the description that has a symbol, by id.
#[derive(Debug, Default)]
struct Bindings {
    functions: Vec<Binding>,
    globals: Vec<Binding>,
    tables: Vec<Binding>,
    data: Vec<Binding>,
}

/// The first operations of the unit that name a data entry, by how they name it.
#[derive(Clone, Copy, Debug, Default)]
struct DataUses {
    /// The first `wasm.memory_init` or `wasm.data_drop`, which make the entry passive.
    passive: Option<OperationRef>,
    /// The first `wasm.data_addr` or `builtin.global_symbol`, which take its address.
    address: Option<OperationRef>,
}

/// The operation each entity of the description was made from, by id: the definition, or the
/// first declaration of an import or of an undefined data symbol.
#[derive(Debug, Default)]
struct Origins {
    functions: Vec<OperationRef>,
    globals: Vec<OperationRef>,
    tables: Vec<OperationRef>,
    memories: Vec<OperationRef>,
    data: Vec<OperationRef>,
}

impl Unit {
    /// The operation the function `id` was made from: its definition, or the first declaration of
    /// the import. `None` for a function the unit did not collect.
    pub fn function_op(&self, id: FuncId) -> Option<OperationRef> {
        self.origins.functions.get(id.index() as usize).copied()
    }

    /// The operation the global `id` was made from, as [Self::function_op] says.
    pub fn global_op(&self, id: GlobalId) -> Option<OperationRef> {
        self.origins.globals.get(id.index() as usize).copied()
    }

    /// The operation the table `id` was made from, as [Self::function_op] says.
    pub fn table_op(&self, id: TableId) -> Option<OperationRef> {
        self.origins.tables.get(id.index() as usize).copied()
    }

    /// The operation the memory `id` was made from, as [Self::function_op] says.
    pub fn memory_op(&self, id: MemoryId) -> Option<OperationRef> {
        self.origins.memories.get(id.index() as usize).copied()
    }

    /// The `builtin.global_variable` the data entry `id` was made from: its definition, or the
    /// first declaration of an undefined data symbol.
    pub fn data_op(&self, id: DataId) -> Option<OperationRef> {
        self.origins.data.get(id.index() as usize).copied()
    }

    /// The undefined data symbols: the data entries made from declared global variables that link
    /// to no definition, one per link name, in the order they are first declared. Each is a stub
    /// whose name is the link name; [Self::data_op] gives its first declaration.
    pub fn undefined_data(&self) -> &[DataId] {
        &self.undefined_data
    }

    /// The `builtin.segment`s of the unit, in the order they are written. None of them is in the
    /// description.
    pub fn segments(&self) -> &[OperationRef] {
        &self.segments
    }

    /// The `wasm.elem`s of the unit, in the order they are written, each with the table it
    /// fills. Each is an active element segment of the description.
    pub fn elements(&self) -> &[(OperationRef, TableId)] {
        &self.elements
    }

    /// The first `wasm.memory_init` or `wasm.data_drop` of the unit (in a function body) that
    /// names the data entry `id`, which makes it a passive segment.
    pub fn passive_use(&self, id: DataId) -> Option<OperationRef> {
        self.data_uses.get(&id).and_then(|uses| uses.passive)
    }

    /// The first operation of the unit that takes the address of the data entry `id`: a
    /// `wasm.data_addr` (in a body or a `wasm.global`'s initializer), or a `builtin.global_symbol`
    /// in a global variable's initializer.
    pub fn address_use(&self, id: DataId) -> Option<OperationRef> {
        self.data_uses.get(&id).and_then(|uses| uses.address)
    }

    /// The binding of the symbol of the function `id`: from the visibility of its definition, or
    /// [Binding::Undefined] for an import. `None` for a function the unit did not collect.
    pub fn function_binding(&self, id: FuncId) -> Option<Binding> {
        self.bindings.functions.get(id.index() as usize).copied()
    }

    /// The binding of the symbol of the global `id`, as [Self::function_binding] says.
    pub fn global_binding(&self, id: GlobalId) -> Option<Binding> {
        self.bindings.globals.get(id.index() as usize).copied()
    }

    /// The binding of the symbol of the table `id`, as [Self::function_binding] says.
    pub fn table_binding(&self, id: TableId) -> Option<Binding> {
        self.bindings.tables.get(id.index() as usize).copied()
    }

    /// The binding of the symbol of the data entry `id`: from the visibility of its definition, or
    /// [Binding::Undefined] (which never names a host) for an undefined data symbol. `None` for an
    /// entry the unit did not collect (one the layout adds).
    pub fn data_binding(&self, id: DataId) -> Option<Binding> {
        self.bindings.data.get(id.index() as usize).copied()
    }

    /// Whether the data entry `id` is retained: it is defined by the unit and has a `section`.
    /// The linker keeps a retained entry's segment although no live symbol names it, as a section
    /// whose members are found between `__start_<s>` and `__stop_<s>` needs.
    pub fn is_retained(&self, id: DataId) -> bool {
        self.data_binding(id).is_some_and(|binding| !binding.is_undefined())
            && self
                .desc
                .data()
                .get(id.index() as usize)
                .is_some_and(|entry| entry.section.is_some())
    }

    /// The name the function `id` is exported under, when it is exported: the first of its export
    /// names. [Self::check_object] refuses a function with more than one.
    pub fn function_export(&self, id: FuncId) -> Option<&str> {
        self.desc.export_names(ExportTarget::Function(id)).next()
    }

    /// The operation whose `wasm.export_name` gives the export `name`, or the start function for
    /// [START_EXPORT] once [Self::export_start] has exported it. `None` for a name the unit does
    /// not export (or one the layout adds, such as `memory`).
    pub fn export_op(&self, name: &str) -> Option<OperationRef> {
        self.exported.get(name).copied()
    }

    /// The start function of the unit: the function definition written with `wasm.start`, when
    /// there is one. A unit has at most one.
    pub fn start(&self) -> Option<FuncId> {
        self.start
    }

    /// Exports the start function, when the unit has one, as [START_EXPORT] (`_initialize`), as
    /// a complete module does; a unit that already exports that name is the error that two
    /// exports have one name, naming both. A complete module has no start section: its embedder
    /// calls `_initialize`.
    pub fn export_start(&mut self) -> Result<(), Report> {
        let Some(start) = self.start else {
            return Ok(());
        };
        let op = self.function_op(start).expect("the start function has a definition");
        if let Some(first) = self.exported.get(START_EXPORT) {
            let (op, first) = (op.borrow(), first.borrow());
            return Err(two_exports(
                START_EXPORT,
                &op,
                format!(
                    "this is the start function, {}, which a complete module exports as \
                     `{START_EXPORT}`",
                    symbol_name(&op)
                ),
                &first,
            ));
        }
        self.exported.insert(String::from(START_EXPORT), op);
        self.desc.add_export(Export {
            name: String::from(START_EXPORT),
            target: ExportTarget::Function(start),
        });
        Ok(())
    }
}

impl Unit {
    /// Checks that a relocatable object can express the unit, or gives the error of the first
    /// construct it cannot, naming the operation. A complete module expresses each of them; an
    /// object leaves the function table, the data's addresses and the memory to the linker, which
    /// does not do what they need:
    ///
    /// - a `wasm.elem` (the linker ignores the element segments of its inputs);
    /// - a `builtin.segment` (the linker places every data segment; there is no fixed address);
    /// - a passive data entry, labelled on the `wasm.memory_init` or `wasm.data_drop` that names
    ///   it (the linker makes every segment active, and has no relocation for a segment index);
    /// - a `wasm.global` whose initializer names another entity (there is no relocation in the
    ///   global section);
    /// - a memory the unit defines (the linker defines the memory);
    /// - a table the unit defines whose symbol name or link name is `__indirect_function_table`
    ///   (the linker defines that table, and refuses an object that defines it);
    /// - a global the unit defines, not private, whose link name is `__stack_pointer` (the linker
    ///   defines the stack pointer, and the link fails with a duplicate symbol);
    /// - a `wasm.ref_func` in a body (Wasm requires a module to declare each function `ref.func`
    ///   names, and the linker declares none: the linked module would not validate);
    /// - a function exported under more than one name (an object exports a function under one);
    /// - a global, a table or a memory exported under a name other than its link name (an object
    ///   marks such a symbol exported under its own name only);
    /// - an import whose declarations disagree on whether it names its host (it would be both a
    ///   host import and the linker's to resolve).
    ///
    /// A table slot with an addend has no relocation either, and none arises: the code writes a
    /// slot only for `wasm.func_index`, which has no offset, and the evaluator gives a function in
    /// data no addend (a debug assertion holds each to it).
    pub fn check_object(&self) -> Result<(), Report> {
        if let Some((elem, _)) = self.elements.first() {
            let elem = elem.borrow();
            return Err(cannot_express(
                format!("a relocatable object cannot express {}", described(&elem)),
                &elem,
                "this fills a table at slots it fixes, and the linker ignores the element \
                 segments of an object: it lays out the function table itself",
            ));
        }
        if let Some(segment) = self.segments.first() {
            let segment = segment.borrow();
            let offset =
                segment.downcast_ref::<Segment>().map_or(0, |segment| segment.offset_in_bytes());
            return Err(cannot_express(
                String::from("a relocatable object cannot express a `builtin.segment`"),
                &segment,
                format!(
                    "this places bytes at the address {offset}, and the linker places every data \
                     segment of an object itself: there is no data at a fixed address"
                ),
            ));
        }
        for index in 0..self.desc.data().len() {
            let id = DataId::new(index as u32);
            if let Some(passive) = self.passive_use(id) {
                let passive = passive.borrow();
                let name = &self.desc.data()[index].name;
                return Err(cannot_express(
                    format!(
                        "a relocatable object cannot express passive data: `{}` names `{name}` as \
                         a passive segment",
                        passive.name()
                    ),
                    &passive,
                    format!(
                        "this names `{name}` as a passive data segment, and the linker makes \
                         every data segment of an object active, with no relocation for a segment \
                         index"
                    ),
                ));
            }
        }
        for (index, global) in self.desc.globals().iter().enumerate() {
            let Some(fixup) = global.entity.definition().and_then(|expr| expr.fixups.first())
            else {
                continue;
            };
            let op = self.global_op(GlobalId::new(index as u32)).expect("a global has an origin");
            let op = op.borrow();
            return Err(cannot_express(
                format!(
                    "a relocatable object cannot express {}, whose initializer names another \
                     entity",
                    described(&op)
                ),
                &op,
                format!(
                    "the initializer of this global holds {}, and an object has no relocation in \
                     the global section",
                    self.fixup_target(fixup.kind)
                ),
            ));
        }
        for (index, memory) in self.desc.memories().iter().enumerate() {
            if memory.entity.is_import() {
                continue;
            }
            let op = self.memory_op(MemoryId::new(index as u32)).expect("a memory has an origin");
            let op = op.borrow();
            return Err(cannot_express(
                format!(
                    "a relocatable object cannot express a memory it defines, {}",
                    described(&op)
                ),
                &op,
                "this defines a memory, and the linker defines the memory of the module it links: \
                 an object imports it as `env.__linear_memory`",
            ));
        }
        for (index, table) in self.desc.tables().iter().enumerate() {
            if table.entity.is_import() {
                continue;
            }
            let op = self.table_op(TableId::new(index as u32)).expect("a table has an origin");
            let op = op.borrow();
            let symbol = op.as_symbol().map(|symbol| symbol.name().to_string());
            if table.name != INDIRECT_FUNCTION_TABLE
                && symbol.as_deref() != Some(INDIRECT_FUNCTION_TABLE)
            {
                continue;
            }
            return Err(cannot_express(
                format!(
                    "a relocatable object cannot define the function table \
                     `{INDIRECT_FUNCTION_TABLE}`"
                ),
                &op,
                format!(
                    "this defines `{INDIRECT_FUNCTION_TABLE}`, which the linker defines itself: \
                     it refuses an object that defines it"
                ),
            ));
        }
        for (index, global) in self.desc.globals().iter().enumerate() {
            let id = GlobalId::new(index as u32);
            if global.entity.is_import()
                || global.name != STACK_POINTER
                || self.global_binding(id) == Some(Binding::Local)
            {
                continue;
            }
            let op = self.global_op(id).expect("a global has an origin");
            let op = op.borrow();
            return Err(cannot_express(
                format!("a relocatable object cannot define the stack pointer `{STACK_POINTER}`"),
                &op,
                format!(
                    "this defines `{STACK_POINTER}`, which the linker defines itself: the link \
                     fails with a duplicate symbol"
                ),
            ));
        }
        if let Some(op) = self.ref_func {
            let op = op.borrow();
            let callee = op
                .downcast_ref::<wasm::RefFunc>()
                .map_or_else(String::new, |ref_func| format!(" `@{}`", ref_func.callee().path()));
            return Err(cannot_express(
                format!("a relocatable object cannot express `wasm.ref_func`{callee}"),
                &op,
                "this takes a reference to a function, which Wasm requires the module to declare, \
                 and the linker declares no function for `ref.func`: the linked module would not \
                 validate",
            ));
        }
        for index in 0..self.desc.functions().len() {
            let id = FuncId::new(index as u32);
            let mut names = self.desc.export_names(ExportTarget::Function(id));
            let (Some(first), Some(second)) = (names.next(), names.next()) else {
                continue;
            };
            let op = self
                .export_op(second)
                .or_else(|| self.function_op(id))
                .expect("an export of the unit has an origin");
            let op = op.borrow();
            let function = &self.desc.function(id).name;
            return Err(cannot_express(
                format!(
                    "a relocatable object cannot express the function `{function}` exported under \
                     two names, `{first}` and `{second}`"
                ),
                &op,
                format!(
                    "this exports `{function}` as `{second}`, and it is exported as `{first}`: an \
                     object exports a function under one name"
                ),
            ));
        }
        for export in self.desc.exports() {
            let (what, link_name) = match export.target {
                ExportTarget::Function(_) => continue,
                ExportTarget::Global(id) => {
                    ("global", &self.desc.globals()[id.index() as usize].name)
                }
                ExportTarget::Table(id) => ("table", &self.desc.tables()[id.index() as usize].name),
                ExportTarget::Memory(id) => {
                    ("memory", &self.desc.memories()[id.index() as usize].name)
                }
            };
            if export.name == *link_name {
                continue;
            }
            let Some(op) = self.export_op(&export.name) else {
                continue;
            };
            let op = op.borrow();
            return Err(cannot_express(
                format!(
                    "a relocatable object cannot express the export `{}` of the {what} \
                     `{link_name}` under another name",
                    export.name
                ),
                &op,
                format!(
                    "this exports the {what} `{link_name}` as `{}`, and an object marks a {what} \
                     exported only under its own link name",
                    export.name
                ),
            ));
        }
        if let Some((declaration, first)) = self.host_conflict {
            let (declaration, first) = (declaration.borrow(), first.borrow());
            let link_name = name_of_entity(&declaration);
            let says = |op: &Operation| {
                if names_host(op) {
                    format!(
                        "names the host module `{}` with `{IMPORT_MODULE}`, so the linker imports \
                         `{link_name}` from it",
                        import_module(op).map_or_else(String::new, |module| module.to_string())
                    )
                } else {
                    format!(
                        "has no `{IMPORT_MODULE}`, so the linker resolves `{link_name}` to a \
                         definition of another object"
                    )
                }
            };
            return Err(report!(
                message: (format!(
                    "a relocatable object cannot express the import `{link_name}`: its \
                     declarations disagree on whether it names its host"
                )),
                labels: [
                    primary(declaration.span(), format!("this declaration {}", says(&declaration))),
                    context(first.span(), format!("the first declaration {}", says(&first))),
                ],
            ));
        }
        Ok(())
    }

    /// What a fix-up of the kind `kind` stands for, by the name of its target, for a message.
    fn fixup_target(&self, kind: FixupKind) -> String {
        let function = |id: FuncId| &self.desc.function(id).name;
        match kind {
            FixupKind::FunctionIndex(id) => {
                format!("a reference to the function `{}`", function(id))
            }
            FixupKind::TableSlot(id) => {
                format!("the table slot of the function `{}`", function(id))
            }
            FixupKind::GlobalIndex(id) => {
                format!("the global `{}`", self.desc.globals()[id.index() as usize].name)
            }
            FixupKind::TableNumber(id) => {
                format!("the table `{}`", self.desc.tables()[id.index() as usize].name)
            }
            FixupKind::MemoryAddr(id) | FixupKind::DataIndex(id) => {
                format!("the address of `{}`", self.desc.data()[id.index() as usize].name)
            }
            FixupKind::TypeIndex(_) | FixupKind::BlockType(_) => String::from("a type index"),
        }
    }
}

/// The error that a relocatable object cannot express `op`: the message `message`, and the
/// primary label `label` on `op`.
fn cannot_express(message: String, op: &Operation, label: impl Into<String>) -> Report {
    let label: String = label.into();
    report!(
        message: (message),
        labels: [primary(op.span(), label)],
    )
}

/// The operation `op` for a message: `` `wasm.elem` @e ``.
fn described(op: &Operation) -> String {
    format!("`{}` {}", op.name(), symbol_name(op))
}

/// The error of two exports named `name`: `op` exports it, as `label` says, and `first` exported
/// it first.
fn two_exports(name: &str, op: &Operation, label: String, first: &Operation) -> Report {
    report!(
        message: (format!("two exports are named `{name}`")),
        labels: [
            primary(op.span(), label),
            context(
                first.span(),
                format!("this exports {} as `{name}` first", symbol_name(first))
            ),
        ],
    )
}

/// The error of a declared global variable `op` that links to no definition: a core module cannot
/// import data.
pub fn cannot_import_data(op: &Operation, link_name: &str) -> Report {
    report!(
        message: (format!(
            "the global variable `{link_name}` is declared and not defined: a core WebAssembly \
             module cannot import data"
        )),
        labels: [primary(
            op.span(),
            format!("nothing in the unit defines `{link_name}`, and Wasm does not import data")
        )],
    )
}

/// Collects the unit under `root`, a `builtin.world`, a `builtin.component` or a
/// `builtin.module`: every module under it, the root included, linked as one module (see the
/// module docs). Function bodies are encoded with `config`; the names come from `options`.
pub fn collect(
    root: OperationRef,
    options: &EmitOptions,
    config: &BodyConfig,
) -> Result<Unit, Report> {
    let ops = entities_of(root)?;
    let mut collector = Collector::default();
    collector.desc.names = options.names;
    collector.desc.name = options.module_name.clone();
    collector.namespace(&ops)?;
    for op in &ops {
        collector.entry(*op)?;
    }
    collector.link()?;
    collector.check_memories()?;
    for op in &ops {
        collector.contents(*op, config)?;
    }
    collector.exports(&ops)?;
    collector.finish();
    Ok(Unit {
        desc: collector.desc,
        origins: collector.origins,
        undefined_data: collector.undefined_data,
        segments: collector.segments,
        elements: collector.elements,
        data_uses: collector.resolver.data_uses,
        bindings: collector.bindings,
        start: collector.start.map(|(id, _)| id),
        host_conflict: collector.host_conflict,
        exported: collector.exported,
        ref_func: collector.resolver.first_ref_func,
    })
}

/// The operations the modules under `root` directly hold, in the order they are written (a nested
/// module's at the place of the module), or the error of a second component.
fn entities_of(root: OperationRef) -> Result<Vec<OperationRef>, Report> {
    let mut ops = Vec::new();
    let mut component: Option<OperationRef> = None;
    let result = root.raw_prewalk::<Forward, _, _>(|op: OperationRef| {
        let borrowed = op.borrow();
        if borrowed.is::<Component>() {
            if let Some(first) = component {
                return WalkResult::Break(several_components(&first.borrow(), &borrowed));
            }
            component = Some(op);
            return WalkResult::Continue(());
        }
        if borrowed.is::<World>() || borrowed.is::<Module>() {
            return WalkResult::Continue(());
        }
        let Some(parent) = borrowed.parent_op() else {
            return WalkResult::Skip;
        };
        let parent = parent.borrow();
        if parent.is::<Module>() {
            ops.push(op);
        } else if (parent.is::<Component>() || parent.is::<World>())
            && what(&borrowed) != What::Other
        {
            return WalkResult::Break(outside_a_module(&borrowed, &parent));
        }
        WalkResult::Skip
    });
    match result {
        WalkResult::Break(report) => Err(report),
        _ => Ok(ops),
    }
}

/// The error of an entity `op` (a function, an alias, a global variable, a `builtin.segment` or
/// a `wasm` entity) that `parent`, a `builtin.component` or a `builtin.world`, holds directly.
fn outside_a_module(op: &Operation, parent: &Operation) -> Report {
    let what = match op.as_symbol() {
        Some(symbol) => format!("`{}` @{}", op.name(), symbol.name()),
        None => format!("a `{}`", op.name()),
    };
    report!(
        message: (format!(
            "{what} is held by a `{}`, not by a `builtin.module`: component-level entities are \
             not emitted yet",
            parent.name()
        )),
        labels: [
            primary(
                op.span(),
                "this is outside every `builtin.module`, and only the entities of modules are \
                 emitted"
            ),
            context(parent.span(), format!("this `{}` holds it", parent.name())),
        ],
    )
}

/// The error of a second component `second` under the root, `first` being the first.
fn several_components(first: &Operation, second: &Operation) -> Report {
    report!(
        message: "a world with several components gives several binaries, which this backend does \
                  not produce yet",
        labels: [
            primary(second.span(), "this is a second `builtin.component`"),
            context(first.span(), "this is the first"),
        ],
    )
}

/// What an operation a module holds is, for the unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum What {
    Function,
    Alias,
    Global,
    Table,
    Memory,
    Elem,
    Tag,
    Data,
    Segment,
    Other,
}

fn what(op: &Operation) -> What {
    if op.is::<FunctionOp>() {
        What::Function
    } else if op.is::<FunctionAlias>() {
        What::Alias
    } else if op.is::<wasm::Global>() {
        What::Global
    } else if op.is::<wasm::Table>() {
        What::Table
    } else if op.is::<wasm::Memory>() {
        What::Memory
    } else if op.is::<wasm::Elem>() {
        What::Elem
    } else if op.is::<wasm::Tag>() {
        What::Tag
    } else if op.is::<GlobalVariable>() {
        What::Data
    } else if op.is::<Segment>() {
        What::Segment
    } else {
        What::Other
    }
}

/// The kinds of entity a link name can stand for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Function,
    Global,
    Table,
    Memory,
    Data,
}

impl Kind {
    /// The kind with its article, for a message: `a function`.
    fn noun(self) -> &'static str {
        match self {
            Self::Function => "a function",
            Self::Global => "a global",
            Self::Table => "a table",
            Self::Memory => "a memory",
            Self::Data => "a global variable",
        }
    }

    /// The kind a symbol of `what` links as, if it links at all.
    fn of(what: What) -> Option<Self> {
        match what {
            What::Function | What::Alias => Some(Self::Function),
            What::Global => Some(Self::Global),
            What::Table => Some(Self::Table),
            What::Memory => Some(Self::Memory),
            What::Data => Some(Self::Data),
            What::Elem | What::Tag | What::Segment | What::Other => None,
        }
    }
}

/// An entity of the description, of any kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Id {
    Function(FuncId),
    Global(GlobalId),
    Table(TableId),
    Memory(MemoryId),
    Data(DataId),
}

/// The type of an entity, as linking compares it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Shape {
    Function(FuncType),
    Global(ValType, bool),
    Table(RefType, Limits),
    Memory(Limits),
    /// A global variable, of its type.
    Data(runic_ir::Type),
}

impl Shape {
    fn kind(&self) -> Kind {
        match self {
            Self::Function(_) => Kind::Function,
            Self::Global(..) => Kind::Global,
            Self::Table(..) => Kind::Table,
            Self::Memory(_) => Kind::Memory,
            Self::Data(_) => Kind::Data,
        }
    }

    /// The type for a message: `(i32) -> i64`, `mut i32`, `funcref 1..8`, `1..16 pages`.
    fn text(&self) -> String {
        match self {
            Self::Function(ty) => func_type_text(ty),
            Self::Global(ty, mutable) => {
                format!("{}{}", if *mutable { "mut " } else { "" }, val_type_text(*ty))
            }
            Self::Table(element, limits) => {
                format!("{} {}", val_type_text(ValType::from(*element)), limits_text(*limits))
            }
            Self::Memory(limits) => format!("{} pages", limits_text(*limits)),
            Self::Data(ty) => ty.to_string(),
        }
    }

    /// Whether a definition of this type satisfies a declaration of the type `declared`.
    fn satisfies(&self, declared: &Self) -> bool {
        match (self, declared) {
            (Self::Table(element, limits), Self::Table(declared_element, declared_limits)) => {
                element == declared_element && within(*limits, *declared_limits)
            }
            (Self::Memory(limits), Self::Memory(declared_limits)) => {
                within(*limits, *declared_limits)
            }
            _ => self == declared,
        }
    }
}

/// Whether the limits `limits` of a definition are within the limits `declared` of a declaration:
/// a minimum at least the declared one, and a maximum at most the declared one when it has one.
fn within(limits: Limits, declared: Limits) -> bool {
    limits.min >= declared.min
        && match declared.max {
            None => true,
            Some(max) => limits.max.is_some_and(|own| own <= max),
        }
}

fn val_type_text(ty: ValType) -> &'static str {
    match ty {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "f32",
        ValType::F64 => "f64",
        ValType::FuncRef => "funcref",
        ValType::ExternRef => "externref",
    }
}

fn func_type_text(ty: &FuncType) -> String {
    let list = |types: &[ValType]| {
        types.iter().map(|ty| val_type_text(*ty)).collect::<Vec<_>>().join(", ")
    };
    format!("({}) -> ({})", list(&ty.params), list(&ty.results))
}

fn limits_text(limits: Limits) -> String {
    match limits.max {
        None => format!("{}", limits.min),
        Some(max) => format!("{}..{max}", limits.min),
    }
}

/// The symbol name of `op` for a message: `@f`.
fn symbol_name(op: &Operation) -> String {
    match op.as_symbol() {
        Some(symbol) => format!("@{}", symbol.name()),
        None => format!("`{}`", op.name()),
    }
}

/// Whether `op` is a declaration.
fn is_declaration(op: &Operation) -> bool {
    op.as_symbol().is_some_and(|symbol| symbol.is_declaration())
}

/// The binding of the symbol of `op`, a definition, from its visibility.
fn binding_of(op: &Operation) -> Binding {
    match op.as_symbol().map(|symbol| symbol.visibility()) {
        Some(Visibility::Public) => Binding::Global,
        Some(Visibility::Internal) => Binding::Hidden,
        Some(Visibility::Private) | None => Binding::Local,
    }
}

/// Whether the declaration `op` names its host: it is written with `wasm.import_module`.
fn names_host(op: &Operation) -> bool {
    op.get_attribute(runic_ir::interner::Symbol::intern(IMPORT_MODULE)).is_some()
}

/// Whether other modules may name `op`: its visibility is `public` or `internal`.
fn other_modules_may_name(op: &Operation) -> bool {
    op.as_symbol().is_some_and(|symbol| symbol.visibility() != Visibility::Private)
}

/// The effective link name of `op`, or the error that it has none.
fn link_name_of(op: &Operation) -> Result<String, Report> {
    effective_link_name(op).map(|name| name.to_string()).ok_or_else(|| {
        report!(
            message: (format!("`{}` {} has no link name", op.name(), symbol_name(op))),
            labels: [primary(
                op.span(),
                "its name cannot be spelled as a path, so it needs a `link_name` entry"
            )],
        )
    })
}

/// The name of `op` in the `name` section: its effective link name, or its symbol name.
fn name_of_entity(op: &Operation) -> String {
    match effective_link_name(op) {
        Some(name) => name.to_string(),
        None => op.as_symbol().map(|symbol| symbol.name().to_string()).unwrap_or_default(),
    }
}

/// The lowering pipeline, as a textual pipeline, for messages.
fn lowering_pipeline() -> String {
    LOWERING_PASSES
        .iter()
        .map(|(name, options)| {
            if options.is_empty() {
                String::from(*name)
            } else {
                let options =
                    options.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",");
                format!("{name}{{{options}}}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// The Wasm function type of the `builtin.function` `op`, or the error that it is not in Wasm
/// form.
fn function_type(op: &Operation) -> Result<FuncType, Report> {
    let function = op.downcast_ref::<FunctionOp>().expect("a builtin.function");
    let signature = function.get_signature();
    let mut types = [Vec::new(), Vec::new()];
    let lists = [
        ("parameter", signature.params().iter().map(|param| param.ty).collect::<Vec<_>>()),
        ("result", signature.results().iter().map(|result| result.ty).collect::<Vec<_>>()),
    ];
    for (slot, (what, list)) in types.iter_mut().zip(lists) {
        for (index, ty) in list.into_iter().enumerate() {
            let Some(val) = val_type(ty) else {
                return Err(report!(
                    message: (format!(
                        "the function {} is not in Wasm form, so it cannot be emitted as \
                         WebAssembly",
                        symbol_name(op)
                    )),
                    labels: [primary(
                        op.span(),
                        format!(
                            "its {what} {index} has type `{}`, which is not a WebAssembly value \
                             type: the lowering (`{}`) gives every parameter and result one",
                            wasm::val_type_name(ty),
                            lowering_pipeline()
                        )
                    )],
                ));
            };
            slot.push(val);
        }
    }
    let [params, results] = types;
    Ok(FuncType::new(params, results))
}

/// The Wasm value type of the type `ty` of `op`'s `what`.
fn value_type(op: &Operation, what: &str, ty: runic_ir::Type) -> Result<ValType, Report> {
    val_type(ty).ok_or_else(|| {
        unsupported(
            op,
            format!(
                "{what} has type `{}`, which is not a WebAssembly value type",
                wasm::val_type_name(ty)
            ),
        )
    })
}

/// The limits of a table or memory as the description holds them.
fn limits(op: &Operation, limits: wasm::Limits) -> Result<Limits, Report> {
    let bound = |value: u64| {
        u32::try_from(value)
            .map_err(|_| unsupported(op, format!("its limit {value} does not fit 32 bits")))
    };
    Ok(Limits {
        min: bound(limits.min())?,
        max: limits.max().map(bound).transpose()?,
    })
}

/// The type of `op`, a symbol of one of the linked kinds.
fn shape_of(op: &Operation) -> Result<Shape, Report> {
    match what(op) {
        What::Function => function_type(op).map(Shape::Function),
        What::Global => {
            let global = op.downcast_ref::<wasm::Global>().expect("a wasm.global");
            let ty = value_type(op, "the global", *global.get_ty())?;
            Ok(Shape::Global(ty, global.get_mutability().is_mut()))
        }
        What::Table => {
            let table = op.downcast_ref::<wasm::Table>().expect("a wasm.table");
            let element = match value_type(op, "an element", *table.get_element_type())? {
                ValType::FuncRef => RefType::FuncRef,
                ValType::ExternRef => RefType::ExternRef,
                _ => {
                    return Err(unsupported(op, "its element type is not a reference type"));
                }
            };
            Ok(Shape::Table(element, limits(op, *table.get_limits())?))
        }
        What::Memory => {
            let memory = op.downcast_ref::<wasm::Memory>().expect("a wasm.memory");
            Ok(Shape::Memory(limits(op, *memory.get_limits())?))
        }
        What::Data => {
            let variable = op.downcast_ref::<GlobalVariable>().expect("a builtin.global_variable");
            Ok(Shape::Data(*variable.get_ty()))
        }
        _ => Err(unsupported(op, "it is not an entity a declaration can name")),
    }
}

/// A declaration that links to a definition of the unit, resolved once every definition has an
/// id.
struct Pending {
    declaration: OperationRef,
    shape: Shape,
    link_name: String,
    definition: OperationRef,
}

/// The entities of the unit: the id each operation stands for, and the functions named by a
/// `wasm.ref_func` in a body.
#[derive(Default)]
struct Resolver {
    ids: FxHashMap<OperationRef, Id>,
    /// Whether a `wasm.ref_func` is recorded in `ref_funcs` (in function bodies only).
    recording: bool,
    ref_funcs: Vec<FuncId>,
    /// The first `wasm.ref_func` in a body.
    first_ref_func: Option<OperationRef>,
    /// How each data entry is first named.
    data_uses: FxHashMap<DataId, DataUses>,
}

impl Resolver {
    /// The id of `target`, the operation `reference` (held by `user`) resolves to, or the error
    /// naming `user` that it resolves to nothing or to no entity of the kind `kind`.
    fn entity(
        &self,
        user: &Operation,
        reference: &SymbolRefAttr,
        kind: Kind,
        target: Option<OperationRef>,
    ) -> Result<Id, Report> {
        let path = reference.path();
        let Some(target) = target else {
            return Err(unsupported(user, format!("`@{path}` resolves to no symbol")));
        };
        match self.ids.get(&target) {
            Some(id) if Kind::of_id(*id) == kind => Ok(*id),
            _ => Err(unsupported(
                user,
                format!(
                    "`@{path}` names `{}`, which is not {} of the unit",
                    target.borrow().name(),
                    kind.noun()
                ),
            )),
        }
    }
}

impl Kind {
    fn of_id(id: Id) -> Self {
        match id {
            Id::Function(_) => Self::Function,
            Id::Global(_) => Self::Global,
            Id::Table(_) => Self::Table,
            Id::Memory(_) => Self::Memory,
            Id::Data(_) => Self::Data,
        }
    }
}

/// The operation `symbol` is.
fn op_of(symbol: SymbolRef) -> OperationRef {
    symbol.borrow().as_symbol_operation().as_operation_ref()
}

/// The canonical target of the alias `op` (through every alias), or `op` itself when it is not
/// an alias; `None` when the alias does not resolve.
fn canonical(op: OperationRef) -> Option<OperationRef> {
    if !op.borrow().is::<FunctionAlias>() {
        return Some(op);
    }
    let symbol = op.borrow().as_symbol_ref()?;
    symbol.resolve_canonical().ok().map(op_of)
}

impl Entities for Resolver {
    fn function(&mut self, user: &Operation, callee: &SymbolRefAttr) -> Result<FuncId, Report> {
        let target = callee
            .resolve_callable()
            .map_err(|error| {
                unsupported(user, format!("`@{}` does not resolve: {error}", callee.path()))
            })?
            .target()
            .as_operation_ref();
        let Id::Function(id) = self.entity(user, callee, Kind::Function, Some(target))? else {
            unreachable!("the kind is checked");
        };
        if self.recording && user.is::<wasm::RefFunc>() {
            self.first_ref_func.get_or_insert(user.as_operation_ref());
            if !self.ref_funcs.contains(&id) {
                self.ref_funcs.push(id);
            }
        }
        Ok(id)
    }

    fn global(&mut self, user: &Operation, global: &SymbolRefAttr) -> Result<GlobalId, Report> {
        match self.entity(user, global, Kind::Global, global.resolve().map(op_of))? {
            Id::Global(id) => Ok(id),
            _ => unreachable!("the kind is checked"),
        }
    }

    fn table(&mut self, user: &Operation, table: &SymbolRefAttr) -> Result<TableId, Report> {
        match self.entity(user, table, Kind::Table, table.resolve().map(op_of))? {
            Id::Table(id) => Ok(id),
            _ => unreachable!("the kind is checked"),
        }
    }

    fn data(&mut self, user: &Operation, data: &SymbolRefAttr) -> Result<DataId, Report> {
        let Id::Data(id) = self.entity(user, data, Kind::Data, data.resolve().map(op_of))? else {
            unreachable!("the kind is checked");
        };
        let uses = self.data_uses.entry(id).or_default();
        let first = if user.is::<wasm::MemoryInit>() || user.is::<wasm::DataDrop>() {
            &mut uses.passive
        } else {
            &mut uses.address
        };
        first.get_or_insert(user.as_operation_ref());
        Ok(id)
    }
}

impl data::DataTargets for Resolver {
    fn data_entry(&mut self, global: &Operation, target: OperationRef) -> Result<DataId, Report> {
        let Some(Id::Data(id)) = self.ids.get(&target).copied() else {
            return Err(not_of_the_unit(global, target, Kind::Data));
        };
        self.data_uses
            .entry(id)
            .or_default()
            .address
            .get_or_insert(global.as_operation_ref());
        Ok(id)
    }

    fn function(&mut self, global: &Operation, target: OperationRef) -> Result<FuncId, Report> {
        match self.ids.get(&target).copied() {
            Some(Id::Function(id)) => Ok(id),
            _ => Err(not_of_the_unit(global, target, Kind::Function)),
        }
    }
}

/// The error, naming the global variable `global`, that its initializer names `target`, which is
/// not an entity of the unit of the kind `kind`.
fn not_of_the_unit(global: &Operation, target: OperationRef, kind: Kind) -> Report {
    let target = target.borrow();
    unsupported(
        global,
        format!(
            "its initializer names {}, which is not {} of the unit",
            symbol_name(&target),
            kind.noun()
        ),
    )
}

/// The state of a collection.
#[derive(Default)]
struct Collector {
    desc: ModuleDesc,
    origins: Origins,
    resolver: Resolver,
    /// The definitions that other modules may name, by link name: the kind and the operation.
    names: FxHashMap<String, (Kind, OperationRef)>,
    /// The first declaration of each import, by kind, module and field.
    imports: FxHashMap<(Kind, String, String), OperationRef>,
    /// The kind and first declaration of each import field, by module and field.
    fields: FxHashMap<(String, String), (Kind, OperationRef)>,
    /// The first declaration of each undefined data symbol, by link name.
    undefined: FxHashMap<String, OperationRef>,
    undefined_data: Vec<DataId>,
    segments: Vec<OperationRef>,
    elements: Vec<(OperationRef, TableId)>,
    pending: Vec<Pending>,
    bindings: Bindings,
    /// The start function and its definition.
    start: Option<(FuncId, OperationRef)>,
    /// The first declaration of an import that disagrees with the import's first declaration on
    /// whether it names its host, and that first declaration.
    host_conflict: Option<(OperationRef, OperationRef)>,
    /// The operation that gives each export, by name.
    exported: FxHashMap<String, OperationRef>,
}

impl Collector {
    /// Records every definition that other modules may name by its link name; two with one name
    /// are an error naming both.
    fn namespace(&mut self, ops: &[OperationRef]) -> Result<(), Report> {
        for op in ops {
            let borrowed = op.borrow();
            let Some(kind) = Kind::of(what(&borrowed)) else {
                continue;
            };
            if is_declaration(&borrowed) || !other_modules_may_name(&borrowed) {
                continue;
            }
            let link_name = link_name_of(&borrowed)?;
            if let Some((_, first)) = self.names.get(&link_name) {
                let first = first.borrow();
                return Err(report!(
                    message: (format!("`{link_name}` is defined twice in the unit")),
                    labels: [
                        primary(
                            borrowed.span(),
                            format!("this defines `{link_name}` as {}", kind.noun())
                        ),
                        context(
                            first.span(),
                            format!(
                                "this defines it as {} first",
                                Kind::of(what(&first)).map_or("a symbol", Kind::noun)
                            )
                        ),
                    ],
                ));
            }
            self.names.insert(link_name, (kind, *op));
        }
        Ok(())
    }

    /// Makes the entry of `op`, or links it, or remembers it: the second pass, in order.
    fn entry(&mut self, op: OperationRef) -> Result<(), Report> {
        let borrowed = op.borrow();
        match what(&borrowed) {
            What::Tag => Err(unsupported(
                &borrowed,
                "exceptions are not emitted in this version: a `wasm.tag` needs the exception \
                 handling feature, which is outside the backend's feature set",
            )),
            What::Segment => {
                self.segments.push(op);
                Ok(())
            }
            What::Alias | What::Elem | What::Other => Ok(()),
            What::Function | What::Global | What::Table | What::Memory | What::Data => {
                let shape = shape_of(&borrowed)?;
                if is_declaration(&borrowed) {
                    self.declare(op, &borrowed, shape)
                } else {
                    let id = self.define(op, &borrowed, shape, None, binding_of(&borrowed))?;
                    self.resolver.ids.insert(op, id);
                    if let Id::Function(id) = id
                        && wasm::is_start(&borrowed)
                    {
                        self.start_function(id, op, &borrowed)?;
                    }
                    Ok(())
                }
            }
        }
    }

    /// Records the function definition `op`, of id `id`, as the start function; a second one is an
    /// error naming both.
    fn start_function(
        &mut self,
        id: FuncId,
        op: OperationRef,
        borrowed: &Operation,
    ) -> Result<(), Report> {
        if let Some((_, first)) = self.start {
            let first = first.borrow();
            return Err(report!(
                message: "the unit has two start functions: a compilation has at most one",
                labels: [
                    primary(
                        borrowed.span(),
                        format!(
                            "this marks {} as the start function with `{}`",
                            symbol_name(borrowed),
                            wasm::START
                        )
                    ),
                    context(
                        first.span(),
                        format!("{} is the start function already", symbol_name(&first))
                    ),
                ],
            ));
        }
        self.start = Some((id, op));
        Ok(())
    }

    /// Adds the entity `op` of type `shape` to the description, with the binding `binding`: a
    /// definition when `import` is `None`, else that import (or, for data, an undefined data
    /// symbol).
    fn define(
        &mut self,
        op: OperationRef,
        borrowed: &Operation,
        shape: Shape,
        import: Option<Import>,
        binding: Binding,
    ) -> Result<Id, Report> {
        let name = match &import {
            Some(import) => import.field.clone(),
            None => name_of_entity(borrowed),
        };
        let id = match shape {
            Shape::Function(ty) => {
                record_type_features(&mut self.desc, &ty);
                let ty = self.desc.intern_type(ty);
                let entity = match import {
                    Some(import) => Entity::Import(import),
                    None => Entity::Definition(FunctionBody::default()),
                };
                self.origins.functions.push(op);
                self.bindings.functions.push(binding);
                Id::Function(self.desc.add_function(Function { ty, name, entity }))
            }
            Shape::Global(ty, mutable) => {
                let entity = match import {
                    Some(import) => Entity::Import(import),
                    None => Entity::Definition(ConstExpr::default()),
                };
                self.origins.globals.push(op);
                self.bindings.globals.push(binding);
                Id::Global(self.desc.add_global(Global {
                    ty,
                    mutable,
                    name,
                    entity,
                }))
            }
            Shape::Table(element, limits) => {
                let entity = import.map_or(Entity::Definition(()), Entity::Import);
                self.origins.tables.push(op);
                self.bindings.tables.push(binding);
                Id::Table(self.desc.add_table(Table {
                    element,
                    limits,
                    name,
                    entity,
                }))
            }
            Shape::Memory(limits) => {
                let entity = import.map_or(Entity::Definition(()), Entity::Import);
                self.origins.memories.push(op);
                Id::Memory(self.desc.add_memory(Memory {
                    limits,
                    name,
                    entity,
                }))
            }
            Shape::Data(_) => {
                self.origins.data.push(op);
                self.bindings.data.push(binding);
                Id::Data(self.desc.add_data(DataEntry {
                    name,
                    size: 0,
                    bytes: Vec::new(),
                    align: 1,
                    section: section(borrowed).map(|section| section.to_string()),
                    placement: DataPlacement::Layout,
                    fixups: Vec::new(),
                }))
            }
        };
        Ok(id)
    }

    /// The declaration `op` of type `shape`: linked to a definition that other modules may name,
    /// or one import (or undefined data symbol) with the other declarations of its name.
    fn declare(
        &mut self,
        op: OperationRef,
        borrowed: &Operation,
        shape: Shape,
    ) -> Result<(), Report> {
        let link_name = link_name_of(borrowed)?;
        if let Some((kind, definition)) = self.names.get(&link_name).copied() {
            if kind != shape.kind() {
                return Err(report!(
                    message: (format!(
                        "the declaration of `{link_name}` does not agree with its definition"
                    )),
                    labels: [
                        primary(
                            borrowed.span(),
                            format!("this declares `{link_name}` as {}", shape.kind().noun())
                        ),
                        context(
                            definition.borrow().span(),
                            format!("this defines it as {}", kind.noun())
                        ),
                    ],
                ));
            }
            self.pending.push(Pending {
                declaration: op,
                shape,
                link_name,
                definition,
            });
            return Ok(());
        }
        let first = if shape.kind() == Kind::Data {
            self.undefined.get(&link_name).copied()
        } else {
            let module =
                import_module(borrowed).map_or_else(|| String::from("env"), |m| m.to_string());
            let key = (shape.kind(), module.clone(), link_name.clone());
            match self.imports.get(&key).copied() {
                Some(first) => Some(first),
                None => {
                    let field = (module.clone(), link_name.clone());
                    if let Some((kind, other)) = self.fields.get(&field).copied() {
                        let other = other.borrow();
                        return Err(report!(
                            message: (format!(
                                "the import `{module}.{link_name}` is declared as {} and as {}: \
                                 no host module can provide both under one name",
                                kind.noun(),
                                shape.kind().noun()
                            )),
                            labels: [
                                primary(
                                    borrowed.span(),
                                    format!(
                                        "this declares `{module}.{link_name}` as {}",
                                        shape.kind().noun()
                                    )
                                ),
                                context(
                                    other.span(),
                                    format!("this declares it as {}", kind.noun())
                                ),
                            ],
                        ));
                    }
                    self.fields.insert(field, (shape.kind(), op));
                    let binding = Binding::Undefined {
                        names_host: names_host(borrowed),
                    };
                    let id = self.define(
                        op,
                        borrowed,
                        shape,
                        Some(Import::new(module, link_name)),
                        binding,
                    )?;
                    self.imports.insert(key, op);
                    self.resolver.ids.insert(op, id);
                    return Ok(());
                }
            }
        };
        match first {
            Some(first) => {
                let first_shape = shape_of(&first.borrow())?;
                if first_shape != shape {
                    let first = first.borrow();
                    return Err(report!(
                        message: (format!("two declarations of `{link_name}` disagree in type")),
                        labels: [
                            primary(
                                borrowed.span(),
                                format!("this declares `{link_name}` as `{}`", shape.text())
                            ),
                            context(
                                first.span(),
                                format!("this declares it as `{}`", first_shape.text())
                            ),
                        ],
                    ));
                }
                if self.host_conflict.is_none()
                    && names_host(borrowed) != names_host(&first.borrow())
                {
                    self.host_conflict = Some((op, first));
                }
                let id = self.resolver.ids[&first];
                self.resolver.ids.insert(op, id);
            }
            None => {
                // A stub named by the link name (an effective link name exists: it was found).
                let binding = Binding::Undefined { names_host: false };
                let id = self.define(op, borrowed, shape, None, binding)?;
                let Id::Data(data) = id else {
                    unreachable!("a data shape gives a data entry");
                };
                self.undefined.insert(link_name, op);
                self.undefined_data.push(data);
                self.resolver.ids.insert(op, id);
            }
        }
        Ok(())
    }

    /// Gives each declaration that links to a definition the definition's id, through aliases and
    /// declarations, and checks their types agree.
    fn link(&mut self) -> Result<(), Report> {
        let mut pending = core::mem::take(&mut self.pending);
        while !pending.is_empty() {
            let before = pending.len();
            let mut waiting = Vec::new();
            for link in pending {
                let target = canonical(link.definition);
                match target.and_then(|target| self.resolver.ids.get(&target).copied()) {
                    Some(id) => {
                        let shape = self.shape_of_id(id);
                        if !shape.satisfies(&link.shape) {
                            let declaration = link.declaration.borrow();
                            let definition = link.definition.borrow();
                            return Err(report!(
                                message: (format!(
                                    "the declaration of `{}` does not agree with its definition",
                                    link.link_name
                                )),
                                labels: [
                                    primary(
                                        declaration.span(),
                                        format!(
                                            "this declares `{}` as `{}`",
                                            link.link_name,
                                            link.shape.text()
                                        )
                                    ),
                                    context(
                                        definition.span(),
                                        format!("this defines it as `{}`", shape.text())
                                    ),
                                ],
                            ));
                        }
                        self.resolver.ids.insert(link.declaration, id);
                    }
                    None => waiting.push(link),
                }
            }
            if waiting.len() == before {
                let link = &waiting[0];
                let declaration = link.declaration.borrow();
                return Err(report!(
                    message: (format!("the declaration of `{}` links to no entity", link.link_name)),
                    labels: [
                        primary(
                            declaration.span(),
                            format!(
                                "this declares `{}`, whose definition is an alias that does not \
                                 resolve to a function of the unit",
                                link.link_name
                            )
                        ),
                        context(link.definition.borrow().span(), "this is the definition"),
                    ],
                ));
            }
            pending = waiting;
        }
        Ok(())
    }

    /// The type of the entity `id` of the description.
    fn shape_of_id(&self, id: Id) -> Shape {
        match id {
            Id::Function(id) => Shape::Function(self.desc.ty(self.desc.function(id).ty).clone()),
            Id::Global(id) => {
                let global = &self.desc.globals()[id.index() as usize];
                Shape::Global(global.ty, global.mutable)
            }
            Id::Table(id) => {
                let table = &self.desc.tables()[id.index() as usize];
                Shape::Table(table.element, table.limits)
            }
            Id::Memory(id) => Shape::Memory(self.desc.memories()[id.index() as usize].limits),
            Id::Data(id) => {
                let op = self.origins.data[id.index() as usize].borrow();
                let variable =
                    op.downcast_ref::<GlobalVariable>().expect("a builtin.global_variable");
                Shape::Data(*variable.get_ty())
            }
        }
    }

    /// A second memory, defined or imported, is an error naming both.
    fn check_memories(&self) -> Result<(), Report> {
        if let [first, second, ..] = self.origins.memories.as_slice() {
            let (first, second) = (first.borrow(), second.borrow());
            return Err(report!(
                message: "a module with several memories cannot be emitted: multiple memories \
                          are outside the backend's feature set",
                labels: [
                    primary(
                        second.span(),
                        format!("this is a second memory, {}", symbol_name(&second))
                    ),
                    context(
                        first.span(),
                        format!("the module already has {}", symbol_name(&first))
                    ),
                ],
            ));
        }
        Ok(())
    }

    /// Encodes what a definition holds: a function's body and parameter names, a global's
    /// initializer, a global variable's bytes, an element segment.
    fn contents(&mut self, op: OperationRef, config: &BodyConfig) -> Result<(), Report> {
        let what = what(&op.borrow());
        match what {
            What::Function if !is_declaration(&op.borrow()) => {
                let Some(Id::Function(id)) = self.resolver.ids.get(&op).copied() else {
                    unreachable!("every definition has an id");
                };
                self.resolver.recording = true;
                let body = encode_function(op, &mut self.desc, &mut self.resolver, config);
                self.resolver.recording = false;
                let mut body = body?;
                body.local_names = parameter_names(&op.borrow());
                self.desc.function_mut(id).entity = Entity::Definition(body);
            }
            What::Global if !is_declaration(&op.borrow()) => {
                let Some(Id::Global(id)) = self.resolver.ids.get(&op).copied() else {
                    unreachable!("every definition has an id");
                };
                let expr =
                    constexpr::initializer(&op.borrow(), &mut self.desc, &mut self.resolver)?;
                self.desc.global_mut(id).entity = Entity::Definition(expr);
            }
            What::Data if !is_declaration(&op.borrow()) => {
                let Some(Id::Data(id)) = self.resolver.ids.get(&op).copied() else {
                    unreachable!("every definition has an id");
                };
                let init = data::evaluate(&op.borrow(), &mut self.resolver)?;
                // A table slot has no addend in the tool conventions; the evaluator gives a
                // function's slot none.
                debug_assert!(
                    init.fixups.iter().all(|fixup| {
                        !matches!(fixup.kind, FixupKind::TableSlot(_)) || fixup.addend == 0
                    }),
                    "a table slot with an addend in data"
                );
                let entry = self.desc.data_mut(id);
                entry.size = init.size;
                entry.bytes = init.bytes;
                entry.align = init.align;
                entry.fixups = init.fixups;
            }
            What::Elem => {
                let borrowed = op.borrow();
                let elem = borrowed.downcast_ref::<wasm::Elem>().expect("a wasm.elem");
                let table = self.resolver.table(&borrowed, &elem.table())?;
                let functions = elem
                    .function_refs()
                    .into_iter()
                    .map(|reference| self.resolver.function(&borrowed, &reference.borrow()))
                    .collect::<Result<Vec<_>, _>>()?;
                self.desc.add_element(ElemSegment::Active {
                    table,
                    offset: elem.offset_value(),
                    functions,
                });
                self.elements.push((op, table));
            }
            _ => {}
        }
        Ok(())
    }

    /// The exports `wasm.export_name` gives, in order; two of one name are an error naming both.
    fn exports(&mut self, ops: &[OperationRef]) -> Result<(), Report> {
        let exported = &mut self.exported;
        for op in ops {
            let borrowed = op.borrow();
            let names = export_names(&borrowed);
            if names.is_empty() {
                continue;
            }
            let target = canonical(*op).and_then(|target| self.resolver.ids.get(&target).copied());
            let target = match target {
                Some(Id::Function(id)) => ExportTarget::Function(id),
                Some(Id::Global(id)) => ExportTarget::Global(id),
                Some(Id::Table(id)) => ExportTarget::Table(id),
                Some(Id::Memory(id)) => ExportTarget::Memory(id),
                Some(Id::Data(_)) | None => {
                    return Err(unsupported(
                        &borrowed,
                        "it is exported, but it is not a function, a global, a table or a memory \
                         of the unit",
                    ));
                }
            };
            for name in names {
                let name = name.to_string();
                if let Some(first) = exported.get(&name) {
                    let first = first.borrow();
                    let label = format!("this exports {} as `{name}`", symbol_name(&borrowed));
                    return Err(two_exports(&name, &borrowed, label, &first));
                }
                exported.insert(name.clone(), *op);
                self.desc.add_export(Export { name, target });
            }
        }
        Ok(())
    }

    /// The declarative segment and the features that follow from the whole module.
    fn finish(&mut self) {
        if !self.resolver.ref_funcs.is_empty() {
            let functions = core::mem::take(&mut self.resolver.ref_funcs);
            self.desc.add_element(ElemSegment::Declarative(functions));
        }
        let exported_globals = self
            .desc
            .exports()
            .iter()
            .filter_map(|export| match export.target {
                ExportTarget::Global(id) => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut features = Vec::new();
        for (index, global) in self.desc.globals().iter().enumerate() {
            let exported = exported_globals.iter().any(|id| id.index() as usize == index);
            if global.mutable && (global.entity.is_import() || exported) {
                features.push(Feature::MutableGlobals);
            }
            if crate::emit::instr::is_reference(global.ty) {
                features.push(Feature::ReferenceTypes);
            }
        }
        let tables = self.desc.tables();
        if tables.len() > 1 || tables.iter().any(|table| table.element == RefType::ExternRef) {
            features.push(Feature::ReferenceTypes);
        }
        for feature in features {
            self.desc.features.insert(feature);
        }
    }
}

/// The names of the parameters of the function `op` from its `di.subprogram`, as local names:
/// the parameter's index and its name (an empty name is left out).
fn parameter_names(op: &Operation) -> Vec<(u32, String)> {
    let Some(attr) = op.get_attribute("di.subprogram") else {
        return Vec::new();
    };
    let attr = attr.borrow();
    let Some(subprogram) = attr.downcast_ref::<SubprogramAttr>() else {
        return Vec::new();
    };
    let params = op
        .downcast_ref::<FunctionOp>()
        .map_or(0, |function| function.get_signature().params().len());
    subprogram
        .as_value()
        .param_names
        .iter()
        .take(params)
        .enumerate()
        .filter(|(_, name)| !name.as_str().is_empty())
        .map(|(index, name)| (index as u32, name.as_str().to_string()))
        .collect()
}
