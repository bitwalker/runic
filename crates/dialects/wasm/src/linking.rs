//! The dictionary entries that mark imports and exports, `wasm.import_module` and
//! `wasm.export_name`, and the one that marks the start function, `wasm.start`.
//!
//! They are discardable attributes of the operations a Wasm module imports and exports: a
//! `builtin.function`, a [Global], a [Memory], a [Table] or a [Tag]. An import is a declaration:
//! the operation has no body, and its field name is the symbol's effective link name
//! ([effective_link_name](runic_ir::effective_link_name)); `wasm.import_module` names the module,
//! `env` when it is absent (a C toolchain's convention), so the entry is needed only for another
//! module. An export is a definition or a declaration (Wasm exports what it imports: a module that
//! forwards an imported memory or table writes both entries), and `wasm.export_name` is the name,
//! or an array of names when one entity is exported under several. A name is any string, the empty
//! one included, as in Wasm. A `builtin.global_variable` is data in linear
//! memory, and Wasm neither imports nor exports it.
//!
//! `wasm.start` marks the function a module runs once before anything else: its value is the unit
//! attribute (`attributes { wasm.start = #builtin.unit }`), and it is written on a
//! `builtin.function` definition that takes no parameters and gives no results. How the start
//! function is run is the emitter's: a complete module exports it as `_initialize`, and a
//! relocatable object registers it with the linker as an init function.
//!
//! The dialect checks the entries on any operation (see
//! [WasmDialect](crate::WasmDialect)'s `verify_operation_attribute`).

use alloc::{format, string::String, vec::Vec};

use runic_ir::{
    AttributeRef, CompactString, Operation, Report, SmallVec, Spanned,
    diagnostics::report,
    dialects::builtin::{
        Function, GlobalVariable,
        attributes::{ListAttr, StringAttr, UnitAttr},
    },
    interner,
};

use crate::{Global, Memory, Table, Tag};

/// The name of the entry that gives the module an imported entity comes from: a string, which may
/// be empty.
pub const IMPORT_MODULE: &str = "wasm.import_module";

/// The name of the entry that gives the name an entity is exported under: a string, or an array of
/// distinct strings, each of which may be empty.
pub const EXPORT_NAME: &str = "wasm.export_name";

/// The name of the entry that marks the start function of a module: the unit attribute
/// (`#builtin.unit`), on a `builtin.function` definition with no parameters and no results.
pub const START: &str = "wasm.start";

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

/// Whether `op` carries the entry [START], whatever its value: whether it is marked as the start
/// function. The dialect's verifier checks that only a `builtin.function` definition with no
/// parameters and no results carries it, with the unit attribute as its value.
pub fn is_start(op: &Operation) -> bool {
    op.get_attribute(interner::Symbol::intern(START)).is_some()
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

/// Checks the entry `name`, whose value is `attr`, of the dictionary of `op`: a
/// `wasm.import_module`, `wasm.export_name` or `wasm.start` entry, or an unknown key of the
/// dialect.
pub(crate) fn verify_entry(
    op: &Operation,
    name: interner::Symbol,
    attr: &AttributeRef,
) -> Result<(), Report> {
    match name.as_str() {
        IMPORT_MODULE => verify_import_module(op, attr),
        EXPORT_NAME => verify_export_name(op, attr),
        START => verify_start(op, attr),
        other => Err(invalid(
            op,
            format!(
                "`{other}` is not an attribute of the `wasm` dialect: it knows `{IMPORT_MODULE}` \
                 and `{EXPORT_NAME}`, which mark imports and exports, and `{START}`, which marks \
                 the start function"
            ),
        )),
    }
}

/// `wasm.import_module` is a string (the empty one is a module name in Wasm), on a declaration of
/// an entity Wasm imports.
fn verify_import_module(op: &Operation, attr: &AttributeRef) -> Result<(), Report> {
    match string_value(attr) {
        Some(_) => {}
        None => {
            return Err(invalid(
                op,
                format!(
                    "`{IMPORT_MODULE}` must be a string, the name of the module an import comes \
                     from, but this is {}",
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

/// `wasm.export_name` is a string or an array of distinct strings (the empty one is a name in
/// Wasm), on a definition or a declaration of an entity Wasm exports.
fn verify_export_name(op: &Operation, attr: &AttributeRef) -> Result<(), Report> {
    let shape = |problem: &str| {
        invalid(
            op,
            format!(
                "`{EXPORT_NAME}` must be a string or an array of distinct strings, but {problem}"
            ),
        )
    };
    if string_value(attr).is_some() {
        // one name, whatever it is
    } else if let Some(names) = string_entries(attr) {
        for (index, name) in names.iter().enumerate() {
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
    Ok(())
}

/// `wasm.start` is the unit attribute, on a `builtin.function` definition whose signature has no
/// parameters and no results.
fn verify_start(op: &Operation, attr: &AttributeRef) -> Result<(), Report> {
    if attr.borrow().downcast_ref::<UnitAttr>().is_none() {
        return Err(invalid(
            op,
            format!(
                "`{START}` marks the start function and takes no value: it is written `{START} = \
                 #builtin.unit`, but this is {}",
                print_attribute(op, attr)
            ),
        ));
    }
    let Some(function) = op.downcast_ref::<Function>() else {
        return Err(invalid(
            op,
            format!(
                "`{START}` marks the start function, but {} is not a function: only a \
                 `builtin.function` is one",
                describe(op)
            ),
        ));
    };
    if is_declaration(op) {
        return Err(invalid(
            op,
            format!(
                "`{START}` marks the start function, which the module runs, but {} is a \
                 declaration: the start function is a definition",
                describe(op)
            ),
        ));
    }
    let signature = function.get_signature();
    let (params, results) = (signature.params().len(), signature.results().len());
    if params != 0 || results != 0 {
        return Err(invalid(
            op,
            format!(
                "`{START}` marks the start function, which takes no parameters and gives no \
                 results, but {} takes {params} and gives {results}",
                describe(op)
            ),
        ));
    }
    Ok(())
}
