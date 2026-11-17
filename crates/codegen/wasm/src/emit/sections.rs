//! The custom sections of an emitted module, in this order: `name` (when the description asks for
//! names), `producers`, `target_features` (when the module uses a feature).

use alloc::{borrow::Cow, vec::Vec};

use wasm_encoder::{
    CustomSection, Encode, IndirectNameMap, Module, NameMap, NameSection, ProducersField,
    ProducersSection,
};

use super::{
    desc::{DataId, Entity, FeatureSet, FuncId, GlobalId, MemoryId, ModuleDesc, TableId},
    write::IndexSpaces,
};

/// The tool named in the `processed-by` field of the `producers` section.
pub const PRODUCER: &str = "runic";

/// The version named with [PRODUCER]: this crate's.
pub const PRODUCER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A name map of `names`, which are `(index, name)` pairs in any order; empty names are left out.
fn name_map<'a>(names: impl IntoIterator<Item = (u32, &'a str)>) -> NameMap {
    let mut names: Vec<_> = names.into_iter().filter(|(_, name)| !name.is_empty()).collect();
    names.sort_by_key(|(index, _)| *index);
    let mut map = NameMap::new();
    for (index, name) in names {
        map.append(index, name);
    }
    map
}

/// The `name` section of `desc`, with the final indices of `spaces`: the module's name, then the
/// names of functions, locals, tables, memories, globals and data segments, each in ascending
/// order of index. `None` when there is nothing to name.
pub fn name_section(desc: &ModuleDesc, spaces: &IndexSpaces) -> Option<NameSection> {
    let mut section = NameSection::new();
    let mut named = false;
    if let Some(name) = desc.name.as_deref().filter(|name| !name.is_empty()) {
        section.module(name);
        named = true;
    }

    let functions = name_map(desc.functions().iter().enumerate().filter_map(|(i, function)| {
        Some((spaces.function(FuncId::new(i as u32))?, function.name.as_str()))
    }));
    if !functions.is_empty() {
        section.functions(&functions);
        named = true;
    }

    let mut locals: Vec<(u32, NameMap)> = desc
        .functions()
        .iter()
        .enumerate()
        .filter_map(|(i, function)| {
            let Entity::Definition(body) = &function.entity else {
                return None;
            };
            let names =
                name_map(body.local_names.iter().map(|(index, name)| (*index, name.as_str())));
            let index = spaces.function(FuncId::new(i as u32))?;
            (!names.is_empty()).then_some((index, names))
        })
        .collect();
    locals.sort_by_key(|(index, _)| *index);
    if !locals.is_empty() {
        let mut map = IndirectNameMap::new();
        for (index, names) in &locals {
            map.append(*index, names);
        }
        section.locals(&map);
        named = true;
    }

    let tables = name_map(desc.tables().iter().enumerate().filter_map(|(i, table)| {
        Some((spaces.table(TableId::new(i as u32))?, table.name.as_str()))
    }));
    if !tables.is_empty() {
        section.tables(&tables);
        named = true;
    }

    let memories = name_map(desc.memories().iter().enumerate().filter_map(|(i, memory)| {
        Some((spaces.memory(MemoryId::new(i as u32))?, memory.name.as_str()))
    }));
    if !memories.is_empty() {
        section.memories(&memories);
        named = true;
    }

    let globals = name_map(desc.globals().iter().enumerate().filter_map(|(i, global)| {
        Some((spaces.global(GlobalId::new(i as u32))?, global.name.as_str()))
    }));
    if !globals.is_empty() {
        section.globals(&globals);
        named = true;
    }

    let data =
        name_map(desc.data().iter().enumerate().filter_map(|(i, entry)| {
            Some((spaces.data(DataId::new(i as u32))?, entry.name.as_str()))
        }));
    if !data.is_empty() {
        section.data(&data);
        named = true;
    }

    named.then_some(section)
}

/// The `producers` section: one field, `processed-by`, holding ([PRODUCER], [PRODUCER_VERSION]).
pub fn producers_section() -> ProducersSection {
    let mut field = ProducersField::new();
    field.value(PRODUCER, PRODUCER_VERSION);
    let mut section = ProducersSection::new();
    section.field("processed-by", &field);
    section
}

/// The `target_features` section of the tool conventions for `features`: their count, then for
/// each the prefix `+` (used) and its name. `None` when `features` is empty.
pub fn target_features_section(features: FeatureSet) -> Option<CustomSection<'static>> {
    if features.is_empty() {
        return None;
    }
    let mut data = Vec::new();
    (features.iter().count() as u32).encode(&mut data);
    for feature in features.iter() {
        data.push(b'+');
        feature.name().encode(&mut data);
    }
    Some(CustomSection {
        name: Cow::Borrowed("target_features"),
        data: Cow::Owned(data),
    })
}

/// Appends the custom sections of `desc` to `module`: `name` when `desc.names` is set and there is
/// something to name, `producers`, and `target_features` when the module uses a feature.
pub(crate) fn write_custom_sections(module: &mut Module, desc: &ModuleDesc, spaces: &IndexSpaces) {
    if desc.names
        && let Some(names) = name_section(desc, spaces)
    {
        module.section(&names);
    }
    module.section(&producers_section());
    if let Some(features) = target_features_section(desc.features) {
        module.section(&features);
    }
}
