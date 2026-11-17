//! Fix-ups: placeholders in bodies, constant expressions and data, and what the writer makes of
//! them.

use alloc::{string::ToString, vec, vec::Vec};

use runic_ir::Report;
use wasmparser::{Operator, Payload};

use super::support::{declare, define, payloads, set_body, text, written};
use crate::emit::{
    desc::{
        CODE_PLACEHOLDER, ConstExpr, DataEntry, DataPlacement, Entity, Fixup, FixupKind, Global,
        Import, Limits, Memory, ModuleDesc, RefType, Table, ValType,
    },
    write::{Resolved, Site, WriteError, write_module},
};

/// The functions called by each body of the module `bytes`, and the size of each body (its locals
/// declaration included, its size prefix not).
fn calls_and_sizes(bytes: &[u8]) -> Vec<(Vec<u32>, u64)> {
    payloads(bytes)
        .iter()
        .filter_map(|payload| match payload {
            Payload::CodeSectionEntry(body) => {
                let mut calls = Vec::new();
                let mut reader = body.get_operators_reader().unwrap();
                while !reader.eof() {
                    if let Operator::Call { function_index } = reader.read().unwrap() {
                        calls.push(function_index);
                    }
                }
                Some((calls, body.range().end - body.range().start))
            }
            _ => None,
        })
        .collect()
}

/// A module of `count` functions `() -> ()`, the last of which calls `first` and then `second`.
fn caller_of(count: u32, first: u32, second: u32) -> ModuleDesc {
    let mut desc = ModuleDesc::new();
    let ids: Vec<_> = (0..count).map(|_| declare(&mut desc, "", (&[], &[]))).collect();
    let caller = *ids.last().unwrap();
    set_body(&mut desc, caller, &[], |code| {
        code.fixup_index(&[0x10], FixupKind::FunctionIndex(ids[first as usize]));
        code.fixup_index(&[0x10], FixupKind::FunctionIndex(ids[second as usize]));
    });
    desc
}

#[test]
fn a_call_is_written_with_the_shortest_index_and_a_second_call_lands_after_it() {
    // In code: two five-byte placeholders, each after its one-byte opcode, then `end`.
    let desc = caller_of(130, 5, 129);
    let body = desc.functions()[129].entity.definition().unwrap();
    assert_eq!(body.code.len(), 1 + 5 + 1 + 5 + 1);
    assert_eq!(body.fixups.iter().map(|fixup| fixup.offset).collect::<Vec<_>>(), [1, 7]);

    let bytes = written(&desc, &Resolved::new());
    let (calls, size) = calls_and_sizes(&bytes).pop().unwrap();
    assert_eq!(calls, [5, 129]);
    // No locals (1), `call 5` (1 + 1), `call 129` (1 + 2), `end` (1).
    assert_eq!(size, 1 + 2 + 3 + 1);
}

#[test]
fn an_index_that_needs_three_bytes_is_written_in_three() {
    let desc = caller_of(16_400, 16_399, 127);
    let bytes = written(&desc, &Resolved::new());
    let (calls, size) = calls_and_sizes(&bytes).pop().unwrap();
    assert_eq!(calls, [16_399, 127]);
    // No locals (1), `call 16399` (1 + 3), `call 127` (1 + 1), `end` (1).
    assert_eq!(size, 1 + 4 + 2 + 1);
}

/// A module with a memory, a data entry at `address` and a table holding `@target` in slot 1.
fn with_data_and_table(
    address: u32,
) -> (ModuleDesc, Resolved, crate::emit::desc::DataId, crate::emit::desc::FuncId) {
    let mut desc = ModuleDesc::new();
    desc.names = true;
    desc.add_memory(Memory {
        limits: Limits::at_least(1),
        name: "memory".to_string(),
        entity: Entity::Definition(()),
    });
    desc.add_table(Table {
        element: RefType::FuncRef,
        limits: Limits::at_least(2),
        name: "table".to_string(),
        entity: Entity::Definition(()),
    });
    let entry = desc.add_data(DataEntry {
        name: "entry".to_string(),
        bytes: vec![1, 2, 3, 4],
        align: 4,
        section: None,
        placement: DataPlacement::Layout,
        fixups: Vec::new(),
    });
    let target = define(&mut desc, "target", (&[], &[]), &[], |_| {});
    let mut resolved = Resolved::new();
    resolved.set_data_address(entry, address);
    resolved.set_table_slot(target, 1);
    (desc, resolved, entry, target)
}

#[test]
fn an_address_and_a_table_slot_in_code_are_i32_constants_with_their_addends() {
    let (mut desc, resolved, entry, target) = with_data_and_table(2);
    define(&mut desc, "user", (&[], &[]), &[], |code| {
        code.fixup_i32_const(FixupKind::MemoryAddr(entry), 100).sink().drop();
        code.fixup_i32_const(FixupKind::MemoryAddr(entry), -4).sink().drop();
        code.fixup_i32_const(FixupKind::TableSlot(target), 0).sink().drop();
    });
    let text = text(&desc, &resolved);
    assert!(
        text.contains(
            "  (func $user (;1;) (type 0)\n    i32.const 102\n    drop\n    i32.const -2\n    \
             drop\n    i32.const 1\n    drop\n  )\n"
        ),
        "{text}"
    );
}

#[test]
fn an_address_and_a_table_slot_in_data_are_four_little_endian_bytes() {
    let (mut desc, resolved, entry, target) = with_data_and_table(0x0102_0304);
    let pointers = desc.add_data(DataEntry {
        name: "pointers".to_string(),
        bytes: vec![0xaa, 0, 0, 0, 0, 0, 0, 0, 0, 0xbb],
        align: 1,
        section: None,
        placement: DataPlacement::Fixed(0),
        fixups: vec![
            Fixup {
                offset: 1,
                kind: FixupKind::MemoryAddr(entry),
                addend: -0x0304,
            },
            Fixup {
                offset: 5,
                kind: FixupKind::TableSlot(target),
                addend: 0,
            },
        ],
    });
    let bytes = written(&desc, &resolved);
    let segments: Vec<Vec<u8>> = payloads(&bytes)
        .into_iter()
        .filter_map(|payload| match payload {
            Payload::DataSection(reader) => Some(reader),
            _ => None,
        })
        .flatten()
        .map(|data| data.unwrap().data.to_vec())
        .collect();
    assert_eq!(segments[pointers.index() as usize], [0xaa, 0, 0, 2, 1, 1, 0, 0, 0, 0xbb]);
}

#[test]
fn a_globals_constant_expression_holds_fix_ups() {
    let (mut desc, resolved, entry, target) = with_data_and_table(1024);
    let imported = desc.add_global(Global {
        ty: ValType::I32,
        mutable: false,
        name: "base".to_string(),
        entity: Entity::Import(Import::new("env", "base")),
    });
    let mut address = ConstExpr::default();
    address.writer().fixup_i32_const(FixupKind::MemoryAddr(entry), 16);
    desc.add_global(Global {
        ty: ValType::I32,
        mutable: false,
        name: "address".to_string(),
        entity: Entity::Definition(address),
    });
    let mut copy = ConstExpr::default();
    copy.writer().fixup_index(&[0x23], FixupKind::GlobalIndex(imported));
    desc.add_global(Global {
        ty: ValType::I32,
        mutable: false,
        name: "copy".to_string(),
        entity: Entity::Definition(copy),
    });
    let mut function = ConstExpr::default();
    function.writer().fixup_index(&[0xd2], FixupKind::FunctionIndex(target));
    desc.add_global(Global {
        ty: ValType::FuncRef,
        mutable: false,
        name: "function".to_string(),
        entity: Entity::Definition(function),
    });
    let text = text(&desc, &resolved);
    assert!(
        text.contains(
            "  (global $address (;1;) i32 i32.const 1040)\n  (global $copy (;2;) i32 global.get \
             $base)\n  (global $function (;3;) funcref ref.func $target)\n"
        ),
        "{text}"
    );
}

#[test]
fn a_fix_up_with_no_resolved_value_is_a_write_error() {
    let (mut desc, _, entry, target) = with_data_and_table(0);
    let user = define(&mut desc, "user", (&[], &[]), &[], |code| {
        code.fixup_i32_const(FixupKind::TableSlot(target), 0).sink().drop();
    });
    let mut resolved = Resolved::new();
    resolved.set_data_address(entry, 0);
    let error = write_module(&desc, &resolved).unwrap_err();
    assert_eq!(
        error,
        WriteError::Unresolved {
            site: Site::Function {
                id: user,
                name: "user".to_string()
            },
            target: FixupKind::TableSlot(target),
        }
    );
    assert_eq!(
        Report::from(error).to_string(),
        "the WebAssembly module could not be written, which is a defect of the backend: function \
         #1 (`user`) needs the table slot of function #0, which the layout did not resolve"
    );
}

#[test]
fn a_data_entry_without_an_address_is_a_write_error() {
    let (desc, _, _, target) = with_data_and_table(0);
    let mut resolved = Resolved::new();
    resolved.set_table_slot(target, 1);
    let error = write_module(&desc, &resolved).unwrap_err();
    assert_eq!(
        error.to_string(),
        "data entry #0 (`entry`) needs the address of data entry #0, which the layout did not \
         resolve"
    );
}

#[test]
fn a_fix_up_that_is_not_on_a_placeholder_is_a_write_error() {
    let mut desc = ModuleDesc::new();
    let callee = define(&mut desc, "callee", (&[], &[]), &[], |_| {});
    let caller = define(&mut desc, "caller", (&[], &[]), &[], |code| {
        code.fixup_index(&[0x10], FixupKind::FunctionIndex(callee));
    });
    let fixup = {
        let body = desc.function_mut(caller).entity.definition_mut().unwrap();
        body.fixups[0].offset = 0;
        body.fixups[0]
    };
    let error = write_module(&desc, &Resolved::new()).unwrap_err();
    assert_eq!(
        error,
        WriteError::MisplacedPlaceholder {
            site: Site::Function {
                id: caller,
                name: "caller".to_string()
            },
            fixup,
        }
    );
    assert_eq!(
        error.to_string(),
        "function #1 (`caller`) has a fix-up for the index of function #0 at offset 0 where there \
         is no placeholder for it"
    );
}

#[test]
fn an_index_in_data_is_a_write_error() {
    let (mut desc, resolved, _, target) = with_data_and_table(0);
    desc.add_data(DataEntry {
        name: "bad".to_string(),
        bytes: vec![0; 4],
        align: 4,
        section: None,
        placement: DataPlacement::Fixed(64),
        fixups: vec![Fixup {
            offset: 0,
            kind: FixupKind::FunctionIndex(target),
            addend: 0,
        }],
    });
    let error = write_module(&desc, &resolved).unwrap_err();
    assert_eq!(
        error.to_string(),
        "data entry #1 (`bad`) has a fix-up for the index of function #0 with addend 0 at offset \
         0, which cannot be there"
    );
}

#[test]
fn the_code_writer_records_each_placeholder_where_it_writes_it() {
    let mut desc = ModuleDesc::new();
    let ty = desc.intern_type(crate::emit::desc::FuncType::new([], []));
    let mut body = crate::emit::desc::FunctionBody::default();
    let mut writer = body.writer();
    writer.sink().i32_const(0);
    writer.fixup_index(&[0x11], FixupKind::TypeIndex(ty));
    writer.raw(&[0x00]);
    assert_eq!(body.code, [&[0x41, 0x00, 0x11][..], &CODE_PLACEHOLDER, &[0x00]].concat());
    assert_eq!(
        body.fixups,
        [Fixup {
            offset: 3,
            kind: FixupKind::TypeIndex(ty),
            addend: 0
        }]
    );
}

#[test]
fn a_block_type_index_is_written_as_a_signed_number() {
    // 65 types: the last, index 64, is one byte unsigned (0x40) but two bytes signed (0xc0 0x00).
    let mut desc = ModuleDesc::new();
    for i in 0..64 {
        desc.intern_type(crate::emit::desc::FuncType::new(vec![ValType::I32; i], []));
    }
    let block =
        desc.intern_type(crate::emit::desc::FuncType::new([], [ValType::I32, ValType::I64]));
    assert_eq!(block.index(), 64);
    define(&mut desc, "f", (&[], &[]), &[], |code| {
        code.fixup_index(&[0x02], FixupKind::BlockType(block));
        code.sink().i32_const(1).i64_const(2).end().drop().drop();
    });
    let text = text(&desc, &Resolved::new());
    assert!(text.contains("    block (type 64) (result i32 i64) ;; label = @1\n"), "{text}");
}
