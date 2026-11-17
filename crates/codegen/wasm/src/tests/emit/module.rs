//! The complete-module writer: index spaces, entities, segments, exports and the order of
//! sections.

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use wasmparser::{DataKind, Payload};

use super::support::{
    declare, define, import, payloads, producers_text, section_ids, text, written,
};
use crate::emit::{
    desc::{
        ConstExpr, DataEntry, DataPlacement, ElemSegment, Entity, Export, ExportTarget, Feature,
        FeatureSet, FixupKind, FuncType, Global, Import, Limits, Memory, ModuleDesc, RefType,
        Table, ValType,
    },
    validate::validate,
    write::{Resolved, WriteError, write_module},
};

const I32: ValType = ValType::I32;
const I64: ValType = ValType::I64;
const F32: ValType = ValType::F32;
const F64: ValType = ValType::F64;

fn memory(desc: &mut ModuleDesc, name: &str, pages: u32) -> crate::emit::desc::MemoryId {
    desc.add_memory(Memory {
        limits: Limits::at_least(pages),
        name: name.to_string(),
        entity: Entity::Definition(()),
    })
}

fn data(name: &str, bytes: &[u8], placement: DataPlacement) -> DataEntry {
    DataEntry {
        name: name.to_string(),
        bytes: bytes.to_vec(),
        align: 1,
        section: None,
        placement,
        fixups: Vec::new(),
    }
}

#[test]
fn the_empty_description_is_a_module_with_only_the_producers_section() {
    let desc = ModuleDesc::new();
    assert_eq!(text(&desc, &Resolved::new()), format!("(module\n{})\n", producers_text()));
}

#[test]
fn a_function_that_adds_two_parameters_is_written_with_its_type() {
    let mut desc = ModuleDesc::new();
    desc.names = true;
    desc.name = Some("m".to_string());
    define(&mut desc, "add", (&[I32, I32], &[I32]), &[], |code| {
        code.sink().local_get(0).local_get(1).i32_add();
    });
    let expected = format!(
        "(module $m\n  (type (;0;) (func (param i32 i32) (result i32)))\n  (func $add (;0;) (type \
         0) (param i32 i32) (result i32)\n    local.get 0\n    local.get 1\n    i32.add\n  \
         )\n{})\n",
        producers_text()
    );
    assert_eq!(text(&desc, &Resolved::new()), expected);
}

#[test]
fn imports_take_the_lower_indices_whatever_the_order_of_creation() {
    let mut desc = ModuleDesc::new();
    desc.names = true;
    // Definitions first, then imports, of each kind.
    let defined_function = define(&mut desc, "defined_function", (&[], &[]), &[], |_| {});
    let imported_function = import(&mut desc, "imported_function", (&[], &[]));
    let defined_table = desc.add_table(Table {
        element: RefType::FuncRef,
        limits: Limits::at_least(1),
        name: "defined_table".to_string(),
        entity: Entity::Definition(()),
    });
    let imported_table = desc.add_table(Table {
        element: RefType::ExternRef,
        limits: Limits {
            min: 2,
            max: Some(3),
        },
        name: "imported_table".to_string(),
        entity: Entity::Import(Import::new("env", "table")),
    });
    memory(&mut desc, "defined_memory", 1);
    let defined_global = desc.add_global(Global {
        ty: I64,
        mutable: true,
        name: "defined_global".to_string(),
        entity: Entity::Definition(ConstExpr {
            code: vec![0x42, 0x07],
            fixups: Vec::new(),
        }),
    });
    let imported_global = desc.add_global(Global {
        ty: I32,
        mutable: false,
        name: "imported_global".to_string(),
        entity: Entity::Import(Import::new("env", "global")),
    });
    // A body that names each entity, so the indices show in the text.
    super::support::set_body(&mut desc, defined_function, &[], |code| {
        code.fixup_index(&[0x10], FixupKind::FunctionIndex(imported_function));
        code.fixup_index(&[0x10], FixupKind::FunctionIndex(defined_function));
        code.fixup_index(&[0x23], FixupKind::GlobalIndex(imported_global)).sink().drop();
        code.fixup_index(&[0x23], FixupKind::GlobalIndex(defined_global)).sink().drop();
        code.sink().i32_const(0);
        code.fixup_index(&[0x25], FixupKind::TableNumber(defined_table)).sink().drop();
        code.sink().i32_const(0);
        code.fixup_index(&[0x25], FixupKind::TableNumber(imported_table)).sink().drop();
    });
    // The imported memory comes last but is still memory 0.
    desc.add_memory(Memory {
        limits: Limits::at_least(1),
        name: "imported_memory".to_string(),
        entity: Entity::Import(Import::new("env", "memory")),
    });

    let text = text_without_multi_memory(&desc);
    let expected = "(module
  (type (;0;) (func))
  (import \"env\" \"imported_function\" (func $imported_function (;0;) (type 0)))
  (import \"env\" \"table\" (table $imported_table (;0;) 2 3 externref))
  (import \"env\" \"memory\" (memory $imported_memory (;0;) 1))
  (import \"env\" \"global\" (global $imported_global (;0;) i32))
  (table $defined_table (;1;) 1 funcref)
  (memory $defined_memory (;1;) 1)
  (global $defined_global (;1;) (mut i64) i64.const 7)
  (func $defined_function (;1;) (type 0)
    call $imported_function
    call $defined_function
    global.get $imported_global
    drop
    global.get $defined_global
    drop
    i32.const 0
    table.get $defined_table
    drop
    i32.const 0
    table.get $imported_table
    drop
  )
";
    assert!(text.starts_with(expected), "{text}");
}

/// The text of `desc`, which has two memories and so is printed without the baseline's check.
fn text_without_multi_memory(desc: &ModuleDesc) -> alloc::string::String {
    let bytes = write_module(desc, &Resolved::new()).unwrap();
    let error = validate(&bytes).unwrap_err().to_string();
    assert!(error.contains("multiple memories"), "{error}");
    crate::text::print(&bytes).unwrap()
}

#[test]
fn equal_function_types_intern_to_one_and_a_type_may_have_several_results() {
    let mut desc = ModuleDesc::new();
    let a = desc.intern_type(FuncType::new([I32, I64], [F32, F64]));
    let b = desc.intern_type(FuncType::new([I32], []));
    let c = desc.intern_type(FuncType::new([I32, I64], [F32, F64]));
    assert_eq!(a, c);
    assert_ne!(a, b);
    assert_eq!(desc.types().len(), 2);
    define(&mut desc, "pair", (&[I32, I64], &[F32, F64]), &[], |code| {
        code.sink().f32_const(1.0.into()).f64_const(2.0.into());
    });
    let text = text(&desc, &Resolved::new());
    assert!(
        text.starts_with(
            "(module\n  (type (;0;) (func (param i32 i64) (result f32 f64)))\n  (type (;1;) (func \
             (param i32)))\n  (func (;0;) (type 0) (param i32 i64) (result f32 f64)\n"
        ),
        "{text}"
    );
}

#[test]
fn an_active_element_segment_fills_the_table_and_a_declarative_one_allows_ref_func() {
    let mut desc = ModuleDesc::new();
    desc.names = true;
    let table = desc.add_table(Table {
        element: RefType::FuncRef,
        limits: Limits::at_least(3),
        name: "table".to_string(),
        entity: Entity::Definition(()),
    });
    let f = define(&mut desc, "f", (&[], &[]), &[], |_| {});
    let g = define(&mut desc, "g", (&[], &[]), &[], |_| {});
    let h = define(&mut desc, "h", (&[], &[]), &[], |_| {});
    define(&mut desc, "user", (&[], &[]), &[], |code| {
        code.fixup_index(&[0xd2], FixupKind::FunctionIndex(h)).sink().drop();
    });
    desc.add_element(ElemSegment::Active {
        table,
        offset: 1,
        functions: vec![f, g],
    });
    desc.add_element(ElemSegment::Declarative(vec![h]));
    let text = text(&desc, &Resolved::new());
    assert!(
        text.contains("  (elem (;0;) (i32.const 1) func $f $g)\n  (elem (;1;) declare func $h)\n"),
        "{text}"
    );
    assert!(text.contains("    ref.func $h\n"), "{text}");
}

#[test]
fn data_entries_are_active_at_their_address_passive_or_at_a_fixed_offset() {
    let mut desc = ModuleDesc::new();
    desc.names = true;
    memory(&mut desc, "memory", 1);
    let laid_out = desc.add_data(data("laid_out", b"abc", DataPlacement::Layout));
    let passive = desc.add_data(data("passive", b"xyz", DataPlacement::Passive));
    desc.add_data(data("fixed", b"\x01\x02", DataPlacement::Fixed(4096)));
    desc.uses_data_count = true;
    define(&mut desc, "init", (&[], &[]), &[], |code| {
        code.sink().i32_const(0).i32_const(0).i32_const(3);
        code.fixup_index(&[0xfc, 0x08], FixupKind::DataIndex(passive)).raw(&[0x00]);
        code.fixup_index(&[0xfc, 0x09], FixupKind::DataIndex(passive));
    });
    let mut resolved = Resolved::new();
    resolved.set_data_address(laid_out, 16);
    let bytes = written(&desc, &resolved);
    assert_eq!(
        section_ids(&bytes).iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        [1, 3, 5, 12, 10, 11, 0, 0]
    );
    let text = crate::text::print(&bytes).unwrap();
    assert!(text.contains("    memory.init $passive\n    data.drop $passive\n"), "{text}");
    assert!(
        text.contains(
            "  (data $laid_out (;0;) (i32.const 16) \"abc\")\n  (data $passive (;1;) \"xyz\")\n  \
             (data $fixed (;2;) (i32.const 4096) \"\\01\\02\")\n"
        ),
        "{text}"
    );
}

#[test]
fn zeroed_and_address_only_entries_have_no_segment_and_the_data_indices_count_those_written() {
    let mut desc = ModuleDesc::new();
    desc.names = true;
    memory(&mut desc, "memory", 1);
    let zero = desc.add_data(data("zero", &[0; 8], DataPlacement::Layout));
    let symbol = desc.add_data(data("__heap_base", &[], DataPlacement::Symbol));
    desc.add_data(data("first", b"ab", DataPlacement::Passive));
    let second = desc.add_data(data("second", b"cd", DataPlacement::Passive));
    desc.uses_data_count = true;
    define(&mut desc, "init", (&[], &[I32]), &[], |code| {
        code.sink().i32_const(0).i32_const(0).i32_const(2);
        code.fixup_index(&[0xfc, 0x08], FixupKind::DataIndex(second)).raw(&[0x00]);
        code.fixup_i32_const(FixupKind::MemoryAddr(symbol), 0);
    });
    let mut resolved = Resolved::new();
    resolved.set_data_address(zero, 1024);
    resolved.set_data_address(symbol, 1040);
    let bytes = written(&desc, &resolved);
    let text = crate::text::print(&bytes).unwrap();
    // `memory.init 1` names the second passive segment; the address of the symbol is written.
    assert!(text.contains("    memory.init $second\n    i32.const 1040\n"), "{text}");
    assert!(
        text.contains("  (data $first (;0;) \"ab\")\n  (data $second (;1;) \"cd\")\n"),
        "{text}"
    );
    assert!(!text.contains("$zero") && !text.contains("$__heap_base"), "{text}");
    let count = payloads(&bytes).iter().find_map(|payload| match payload {
        Payload::DataCountSection { count, .. } => Some(*count),
        _ => None,
    });
    assert_eq!(count, Some(2));
}

#[test]
fn a_laid_out_entry_with_a_fix_up_has_a_segment_though_its_bytes_are_zero() {
    let mut desc = ModuleDesc::new();
    memory(&mut desc, "memory", 1);
    let target = desc.add_data(data("target", b"x", DataPlacement::Layout));
    let mut pointer = data("pointer", &[0; 4], DataPlacement::Layout);
    pointer.fixups.push(crate::emit::desc::Fixup {
        offset: 0,
        kind: FixupKind::MemoryAddr(target),
        addend: 0,
    });
    let pointer = desc.add_data(pointer);
    desc.add_data(data("zero", &[0; 4], DataPlacement::Layout));
    let mut resolved = Resolved::new();
    resolved.set_data_address(target, 1024);
    resolved.set_data_address(pointer, 1028);
    let text = text(&desc, &resolved);
    assert!(text.contains("(data (;1;) (i32.const 1028) \"\\00\\04\\00\\00\")"), "{text}");
    assert_eq!(text.matches("(data ").count(), 2, "{text}");
}

#[test]
fn without_memory_init_or_data_drop_there_is_no_data_count_section() {
    let mut desc = ModuleDesc::new();
    memory(&mut desc, "memory", 1);
    desc.add_data(data("passive", b"xyz", DataPlacement::Passive));
    let ids = section_ids(&written(&desc, &Resolved::new()));
    assert_eq!(ids.iter().map(|(id, _)| *id).collect::<Vec<_>>(), [5, 11, 0]);
}

#[test]
fn each_kind_of_entity_is_exported_by_its_final_index() {
    let mut desc = ModuleDesc::new();
    import(&mut desc, "imported", (&[], &[]));
    let f = define(&mut desc, "f", (&[], &[]), &[], |_| {});
    let table = desc.add_table(Table {
        element: RefType::FuncRef,
        limits: Limits::at_least(0),
        name: String::new(),
        entity: Entity::Definition(()),
    });
    let memory = memory(&mut desc, "", 1);
    let global = desc.add_global(Global {
        ty: F32,
        mutable: false,
        name: String::new(),
        entity: Entity::Definition(ConstExpr {
            code: vec![0x43, 0, 0, 0x80, 0x3f],
            fixups: Vec::new(),
        }),
    });
    desc.add_export(Export {
        name: "f".to_string(),
        target: ExportTarget::Function(f),
    });
    desc.add_export(Export {
        name: "table".to_string(),
        target: ExportTarget::Table(table),
    });
    desc.add_export(Export {
        name: "memory".to_string(),
        target: ExportTarget::Memory(memory),
    });
    desc.add_export(Export {
        name: "global".to_string(),
        target: ExportTarget::Global(global),
    });
    let text = text(&desc, &Resolved::new());
    assert!(
        text.contains(
            "  (export \"f\" (func 1))\n  (export \"table\" (table 0))\n  (export \"memory\" \
             (memory 0))\n  (export \"global\" (global 0))\n"
        ),
        "{text}"
    );
}

/// A description with every section: types, an import, functions, a table, a memory, a global,
/// exports, elements, a data count, code, data, and the three custom sections.
pub(super) fn every_section() -> (ModuleDesc, Resolved) {
    let mut desc = ModuleDesc::new();
    desc.names = true;
    desc.name = Some("every".to_string());
    desc.features = FeatureSet::from_iter([Feature::BulkMemory, Feature::MutableGlobals]);
    let imported = import(&mut desc, "imported", (&[I32], &[]));
    let table = desc.add_table(Table {
        element: RefType::FuncRef,
        limits: Limits::at_least(2),
        name: "table".to_string(),
        entity: Entity::Definition(()),
    });
    let memory = memory(&mut desc, "memory", 1);
    let entry = desc.add_data(data("entry", b"hello", DataPlacement::Layout));
    let global = desc.add_global(Global {
        ty: I32,
        mutable: true,
        name: "stack_pointer".to_string(),
        entity: Entity::Definition(ConstExpr {
            code: vec![0x41, 0x00],
            fixups: Vec::new(),
        }),
    });
    let main = declare(&mut desc, "main", (&[], &[]));
    super::support::set_body(&mut desc, main, &[I32], |code| {
        code.fixup_i32_const(FixupKind::MemoryAddr(entry), 0);
        code.fixup_index(&[0x10], FixupKind::FunctionIndex(imported));
        code.sink().i32_const(0).i32_const(0).i32_const(0);
        code.fixup_index(&[0xfc, 0x08], FixupKind::DataIndex(entry)).raw(&[0x00]);
        code.fixup_index(&[0x23], FixupKind::GlobalIndex(global)).sink().local_set(0);
    });
    desc.function_mut(main).entity.definition_mut().unwrap().local_names =
        vec![(0, "saved".to_string())];
    desc.uses_data_count = true;
    desc.add_element(ElemSegment::Active {
        table,
        offset: 1,
        functions: vec![main],
    });
    desc.add_export(Export {
        name: "main".to_string(),
        target: ExportTarget::Function(main),
    });
    desc.add_export(Export {
        name: "memory".to_string(),
        target: ExportTarget::Memory(memory),
    });
    let mut resolved = Resolved::new();
    resolved.set_data_address(entry, 1024);
    (desc, resolved)
}

#[test]
fn every_section_is_written_once_and_in_the_order_webassembly_requires() {
    let (desc, resolved) = every_section();
    let ids = section_ids(&written(&desc, &resolved));
    let expected: Vec<(u8, String)> = [
        (1, ""),
        (2, ""),
        (3, ""),
        (4, ""),
        (5, ""),
        (6, ""),
        (7, ""),
        (9, ""),
        (12, ""),
        (10, ""),
        (11, ""),
        (0, "name"),
        (0, "producers"),
        (0, "target_features"),
    ]
    .into_iter()
    .map(|(id, name)| (id, name.to_string()))
    .collect();
    assert_eq!(ids, expected);
}

#[test]
fn a_body_that_returns_the_wrong_type_is_reported_with_the_validators_offset() {
    let mut desc = ModuleDesc::new();
    define(&mut desc, "wrong", (&[], &[I32]), &[], |code| {
        code.sink().i64_const(0);
    });
    let bytes = write_module(&desc, &Resolved::new()).unwrap();
    // The `end` of the only body: the validator reports the mismatch where the body ends.
    let end = payloads(&bytes)
        .iter()
        .find_map(|payload| match payload {
            Payload::CodeSectionEntry(body) => Some(body.range().end - 1),
            _ => None,
        })
        .unwrap();
    assert_eq!(bytes[end as usize], 0x0b);
    let report = validate(&bytes).unwrap_err().to_string();
    assert_eq!(
        report,
        format!(
            "the emitted WebAssembly module does not validate, which is a defect of the backend: \
             type mismatch: expected i32, found i64 (at offset 0x{end:x})"
        )
    );
}

#[test]
fn a_second_memory_is_outside_the_baseline_and_does_not_validate() {
    let mut desc = ModuleDesc::new();
    memory(&mut desc, "a", 1);
    memory(&mut desc, "b", 1);
    let bytes = write_module(&desc, &Resolved::new()).unwrap();
    let report = validate(&bytes).unwrap_err().to_string();
    assert!(report.contains("multiple memories"), "{report}");
    assert!(report.contains("defect of the backend"), "{report}");
}

#[test]
fn externref_values_are_in_the_baseline() {
    let mut desc = ModuleDesc::new();
    define(&mut desc, "id", (&[ValType::ExternRef], &[ValType::ExternRef]), &[], |code| {
        code.sink().local_get(0);
    });
    written(&desc, &Resolved::new());
}

#[test]
fn an_active_data_entry_in_a_module_without_memory_is_a_write_error() {
    let mut desc = ModuleDesc::new();
    let entry = desc.add_data(data("orphan", b"a", DataPlacement::Fixed(0)));
    let error = write_module(&desc, &Resolved::new()).unwrap_err();
    assert_eq!(
        error,
        WriteError::NoMemory {
            site: crate::emit::write::Site::Data {
                id: entry,
                name: "orphan".to_string()
            }
        }
    );
    assert_eq!(
        error.to_string(),
        "data entry #0 (`orphan`) is an active data segment, but the module has no memory"
    );
}

#[test]
fn a_passive_segment_reads_back_as_passive_and_a_laid_out_one_at_its_address() {
    let mut desc = ModuleDesc::new();
    memory(&mut desc, "memory", 1);
    let laid_out = desc.add_data(data("laid_out", b"abc", DataPlacement::Layout));
    desc.add_data(data("passive", b"xyz", DataPlacement::Passive));
    let mut resolved = Resolved::new();
    resolved.set_data_address(laid_out, 64);
    let bytes = written(&desc, &resolved);
    let mut kinds = Vec::new();
    for payload in payloads(&bytes) {
        if let Payload::DataSection(reader) = payload {
            for data in reader {
                let data = data.unwrap();
                kinds.push(match data.kind {
                    DataKind::Passive => (None, data.data.to_vec()),
                    DataKind::Active {
                        memory_index,
                        offset_expr,
                    } => {
                        assert_eq!(memory_index, 0);
                        let mut ops = offset_expr.get_operators_reader();
                        let wasmparser::Operator::I32Const { value } = ops.read().unwrap() else {
                            panic!("not an i32.const");
                        };
                        (Some(value), data.data.to_vec())
                    }
                });
            }
        }
    }
    assert_eq!(kinds, [(Some(64), b"abc".to_vec()), (None, b"xyz".to_vec())]);
}
