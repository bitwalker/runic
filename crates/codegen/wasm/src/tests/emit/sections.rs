//! The custom sections: `name`, `producers` and `target_features`.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use wasmparser::{BinaryReader, KnownCustom, Name, Payload};

use super::{
    module::every_section,
    support::{payloads, section_ids, written},
};
use crate::emit::{
    desc::{Feature, FeatureSet, ModuleDesc},
    sections::{PRODUCER, PRODUCER_VERSION},
    write::Resolved,
};

/// The custom section `name` of `bytes`: its contents and where it starts.
fn custom<'a>(
    payloads: &'a [Payload<'a>],
    name: &str,
) -> Option<&'a wasmparser::CustomSectionReader<'a>> {
    payloads.iter().find_map(|payload| match payload {
        Payload::CustomSection(custom) if custom.name() == name => Some(custom),
        _ => None,
    })
}

#[test]
fn the_name_section_names_every_kind_of_entity_by_its_final_index() {
    let (desc, resolved) = every_section();
    let bytes = written(&desc, &resolved);
    let payloads = payloads(&bytes);
    let KnownCustom::Name(reader) = custom(&payloads, "name").unwrap().as_known() else {
        panic!("not a name section");
    };
    let mut names: Vec<String> = Vec::new();
    for name in reader {
        let name = name.unwrap();
        let (kind, map) = match name {
            Name::Module { name, .. } => {
                names.push(alloc::format!("module {name}"));
                continue;
            }
            Name::Local(map) => {
                for function in map {
                    let function = function.unwrap();
                    for local in function.names {
                        let local = local.unwrap();
                        names.push(alloc::format!(
                            "local {} {} {}",
                            function.index,
                            local.index,
                            local.name
                        ));
                    }
                }
                continue;
            }
            Name::Function(map) => ("function", map),
            Name::Table(map) => ("table", map),
            Name::Memory(map) => ("memory", map),
            Name::Global(map) => ("global", map),
            Name::Data(map) => ("data", map),
            _ => panic!("an unexpected name subsection"),
        };
        for naming in map {
            let naming = naming.unwrap();
            names.push(alloc::format!("{kind} {} {}", naming.index, naming.name));
        }
    }
    assert_eq!(
        names,
        [
            "module every",
            "function 0 imported",
            "function 1 main",
            "local 1 0 saved",
            "table 0 table",
            "memory 0 memory",
            "global 0 stack_pointer",
            "data 0 entry",
        ]
    );
}

#[test]
fn without_names_there_is_no_name_section() {
    let (mut desc, resolved) = every_section();
    desc.names = false;
    let bytes = written(&desc, &resolved);
    let custom: Vec<_> = section_ids(&bytes)
        .into_iter()
        .filter(|(id, _)| *id == 0)
        .map(|(_, name)| name)
        .collect();
    assert_eq!(custom, ["producers", "target_features"]);
}

#[test]
fn the_producers_section_says_runic_processed_the_module() {
    let bytes = written(&ModuleDesc::new(), &Resolved::new());
    let payloads = payloads(&bytes);
    let KnownCustom::Producers(reader) = custom(&payloads, "producers").unwrap().as_known() else {
        panic!("not a producers section");
    };
    let fields: Vec<(String, Vec<(String, String)>)> = reader
        .into_iter()
        .map(|field| {
            let field = field.unwrap();
            let values = field
                .values
                .into_iter()
                .map(|value| {
                    let value = value.unwrap();
                    (value.name.to_string(), value.version.to_string())
                })
                .collect();
            (field.name.to_string(), values)
        })
        .collect();
    assert_eq!(
        fields,
        [(
            "processed-by".to_string(),
            alloc::vec![(PRODUCER.to_string(), PRODUCER_VERSION.to_string())]
        )]
    );
    assert_eq!(PRODUCER, "runic");
    assert_eq!(PRODUCER_VERSION, env!("CARGO_PKG_VERSION"));
}

#[test]
fn target_features_lists_the_features_used_after_producers() {
    let mut desc = ModuleDesc::new();
    desc.features =
        FeatureSet::from_iter([Feature::TailCall, Feature::SignExt, Feature::Multivalue]);
    let bytes = written(&desc, &Resolved::new());
    let ids: Vec<_> = section_ids(&bytes).into_iter().map(|(_, name)| name).collect();
    assert_eq!(ids, ["producers", "target_features"]);

    let payloads = payloads(&bytes);
    let section = custom(&payloads, "target_features").unwrap();
    let mut reader = BinaryReader::new(section.data(), 0);
    let count = reader.read_var_u32().unwrap();
    let features: Vec<(u8, String)> = (0..count)
        .map(|_| (reader.read_u8().unwrap(), reader.read_string().unwrap().to_string()))
        .collect();
    assert!(reader.eof());
    assert_eq!(
        features,
        [
            (b'+', "sign-ext".to_string()),
            (b'+', "multivalue".to_string()),
            (b'+', "tail-call".to_string()),
        ]
    );
}

#[test]
fn a_module_that_uses_no_feature_has_no_target_features_section() {
    let bytes = written(&ModuleDesc::new(), &Resolved::new());
    let ids: Vec<_> = section_ids(&bytes).into_iter().map(|(_, name)| name).collect();
    assert_eq!(ids, ["producers"]);
}

#[test]
fn every_feature_has_its_tool_conventions_name() {
    let names: Vec<_> = Feature::ALL.iter().map(|feature| feature.name()).collect();
    assert_eq!(
        names,
        [
            "mutable-globals",
            "sign-ext",
            "nontrapping-fptoint",
            "bulk-memory",
            "multivalue",
            "reference-types",
            "tail-call",
        ]
    );
    let all = FeatureSet::from_iter(Feature::ALL);
    assert_eq!(all.iter().collect::<Vec<_>>(), Feature::ALL);
    assert!(FeatureSet::new().is_empty());
}
