//! The dictionary entries that mark imports and exports: `wasm.import_module` and
//! `wasm.export_name`.
//!
//! They are discardable attributes of the operations a Wasm module imports and exports: a
//! `builtin.function`, a [Global], a [Memory], a [Table] or a [Tag]. An import is a declaration:
//! the operation has no body, and its field name is the symbol's effective link name
//! ([effective_link_name](runic_ir::effective_link_name)); `wasm.import_module` names the module,
//! `env` when it is absent (a C toolchain's convention), so the entry is needed only for another
//! module. An export is a definition, and `wasm.export_name` is the name, or an array of names
//! when one entity is exported under several. A `builtin.global_variable` is data in linear
//! memory, and Wasm neither imports nor exports it.
//!
//! The dialect checks the entries on any operation (see
//! [WasmDialect](crate::WasmDialect)'s `verify_operation_attribute`).

use alloc::{format, string::String, vec::Vec};

use runic_ir::{
    AttributeRef, CompactString, Operation, Report, SmallVec, Spanned,
    diagnostics::report,
    dialects::builtin::{
        Function, GlobalVariable,
        attributes::{ListAttr, StringAttr},
    },
    interner,
};

use crate::{Global, Memory, Table, Tag};

/// The name of the entry that gives the module an imported entity comes from: a non-empty string.
pub const IMPORT_MODULE: &str = "wasm.import_module";

/// The name of the entry that gives the name an entity is exported under: a string, or an array of
/// distinct non-empty strings.
pub const EXPORT_NAME: &str = "wasm.export_name";

/// The module an import comes from when the declaration has no [IMPORT_MODULE] entry.
pub const DEFAULT_IMPORT_MODULE: &str = "env";

/// Whether `op` is a kind of operation Wasm imports and exports: a `builtin.function`, a
/// `wasm.global`, a `wasm.memory`, a `wasm.table` or a `wasm.tag`.
fn is_linkable(op: &Operation) -> bool {
    op.is::<Function>()
        || op.is::<Global>()
        || op.is::<Memory>()
        || op.is::<Table>()
        || op.is::<Tag>()
}

/// Whether `op` is a declaration: a symbol without a body or a definition.
fn is_declaration(op: &Operation) -> bool {
    op.as_symbol().is_some_and(|symbol| symbol.is_declaration())
}

/// The text of `attr` as an attribute value is written.
fn print_attribute(op: &Operation, attr: &AttributeRef) -> String {
    let flags = Default::default();
    let mut printer = runic_ir::print::AsmPrinter::root(op.context_rc(), &flags);
    printer.print_attribute_value(&*attr.borrow());
    alloc::string::ToString::to_string(&printer.finish())
}

/// The string value of `attr`, if it is a string attribute.
fn string_value(attr: &AttributeRef) -> Option<CompactString> {
    let value = attr.borrow();
    value
        .downcast_ref::<StringAttr>()
        .map(|string| CompactString::from(string.as_str()))
}

/// The module `op` is imported from: the value of its `wasm.import_module` entry, or `env` when
/// `op` is a declaration of an operation Wasm imports and has none. `None` when `op` is not such a
/// declaration, or the entry is not a string.
///
/// The imported entity's field name is its [effective link
/// name](runic_ir::effective_link_name).
pub fn import_module(op: &Operation) -> Option<CompactString> {
    if !is_linkable(op) || !is_declaration(op) {
        return None;
    }
    match op.get_attribute(interner::Symbol::intern(IMPORT_MODULE)) {
        Some(attr) => string_value(&attr),
        None => Some(CompactString::from(DEFAULT_IMPORT_MODULE)),
    }
}

/// The names `op` is exported under: the value of its `wasm.export_name` entry, one string or an
/// array of them. Empty when `op` has no such entry or the entry is malformed (the dialect's
/// verifier reports it).
pub fn export_names(op: &Operation) -> SmallVec<[CompactString; 1]> {
    let mut names = SmallVec::new();
    let Some(attr) = op.get_attribute(interner::Symbol::intern(EXPORT_NAME)) else {
        return names;
    };
    if let Some(name) = string_value(&attr) {
        names.push(name);
    } else if let Some(entries) = string_entries(&attr) {
        names.extend(entries);
    }
    names
}

/// The strings of `attr`, if it is an array of strings.
fn string_entries(attr: &AttributeRef) -> Option<Vec<CompactString>> {
    let list = attr.borrow();
    let list = list.downcast_ref::<ListAttr>()?;
    let entries: &runic_ir::attributes::AttrList = list.as_value().as_ref();
    entries.iter().map(|entry| string_value(&entry.as_attribute_ref())).collect()
}

/// The report of `op` being invalid because of `problem`.
fn invalid(op: &Operation, problem: String) -> Report {
    report!(
        message: "invalid operation",
        labels: [primary(op.span(), problem)],
    )
}

/// The names of the kinds of operation Wasm imports and exports, for the messages.
const LINKABLE: &str =
    "`builtin.function`, `wasm.global`, `wasm.memory`, `wasm.table` and `wasm.tag`";

/// The name of `op` for a message: `wasm.memory @mem`, or just the operation's name when it has no
/// symbol name.
fn describe(op: &Operation) -> String {
    match op.as_symbol() {
        Some(symbol) => format!("`{} @{}`", op.name(), symbol.name()),
        None => format!("`{}`", op.name()),
    }
}

/// Checks the entry `name`, whose value is `attr`, of the dictionary of `op`: a `wasm.import_module`
/// or a `wasm.export_name` entry, or an unknown key of the dialect.
pub(crate) fn verify_entry(
    op: &Operation,
    name: interner::Symbol,
    attr: &AttributeRef,
) -> Result<(), Report> {
    match name.as_str() {
        IMPORT_MODULE => verify_import_module(op, attr),
        EXPORT_NAME => verify_export_name(op, attr),
        other => Err(invalid(
            op,
            format!(
                "`{other}` is not an attribute of the `wasm` dialect: it knows `{IMPORT_MODULE}` \
                 and `{EXPORT_NAME}`"
            ),
        )),
    }
}

/// `wasm.import_module` is a non-empty string, on a declaration of an entity Wasm imports.
fn verify_import_module(op: &Operation, attr: &AttributeRef) -> Result<(), Report> {
    match string_value(attr) {
        Some(module) if !module.is_empty() => {}
        _ => {
            return Err(invalid(
                op,
                format!(
                    "`{IMPORT_MODULE}` must be a non-empty string, the name of the module an \
                     import comes from, but this is {}",
                    print_attribute(op, attr)
                ),
            ));
        }
    }
    if !is_linkable(op) {
        let reason = if op.is::<GlobalVariable>() {
            "data in linear memory is not something Wasm imports"
        } else {
            "Wasm imports functions, globals, memories, tables and tags"
        };
        return Err(invalid(
            op,
            format!(
                "`{IMPORT_MODULE}` marks an import, but {} is not one: {reason} ({LINKABLE})",
                describe(op)
            ),
        ));
    }
    if !is_declaration(op) {
        return Err(invalid(
            op,
            format!(
                "`{IMPORT_MODULE}` marks an import, and an import is a declaration, but {} is a \
                 definition",
                describe(op)
            ),
        ));
    }
    Ok(())
}

/// `wasm.export_name` is a string or an array of distinct non-empty strings, on a definition of an
/// entity Wasm exports.
fn verify_export_name(op: &Operation, attr: &AttributeRef) -> Result<(), Report> {
    let shape = |problem: &str| {
        invalid(
            op,
            format!(
                "`{EXPORT_NAME}` must be a non-empty string or an array of distinct non-empty \
                 strings, but {problem}"
            ),
        )
    };
    if let Some(name) = string_value(attr) {
        if name.is_empty() {
            return Err(shape("this is an empty string"));
        }
    } else if let Some(names) = string_entries(attr) {
        for (index, name) in names.iter().enumerate() {
            if name.is_empty() {
                return Err(shape(&format!("entry {index} is an empty string")));
            }
            if names[..index].contains(name) {
                return Err(shape(&format!("`{name}` is exported twice")));
            }
        }
    } else {
        return Err(shape(&format!("this is {}", print_attribute(op, attr))));
    }
    if !is_linkable(op) {
        return Err(invalid(
            op,
            format!(
                "`{EXPORT_NAME}` marks an export, but {} is not one: Wasm exports functions, \
                 globals, memories, tables and tags ({LINKABLE})",
                describe(op)
            ),
        ));
    }
    if is_declaration(op) {
        return Err(invalid(
            op,
            format!(
                "`{EXPORT_NAME}` marks an export, and an export is a definition, but {} is a \
                 declaration (an import)",
                describe(op)
            ),
        ));
    }
    Ok(())
}
