//! Relocatable objects read back with `wasmparser`: the sections in order, every symbol, segment,
//! init function and relocation compared with what each test states, and the placeholder at each
//! relocation holding its provisional value.

use alloc::{format, rc::Rc, string::String, vec::Vec};

use runic_ir::{
    Context, Op, OperationRef,
    dialects::builtin::World,
    parse::{self, ParserConfig},
    testing::text::describe,
};
use wasmparser::{KnownCustom, Linking, Parser, Payload, RelocationType, SymbolInfo, TypeRef};

use crate::{
    emit::{EmitOptions, emit_object},
    tests::{LOWERING, context, run_pipeline},
};

// ---- Helpers ------------------------------------------------------------------------------------

/// The attributes of a test module: the data layout of the lowering, with function values of 32
/// bits as the backend's data layout has them.
const LAYOUT: &str = "attributes { data_layout = #builtin.data_layout<little, ptr = 32, fn = 32> }";

/// A module `@name` of the items `items`, with [LAYOUT].
pub(super) fn module(name: &str, items: &[&str]) -> String {
    format!("builtin.module public @{name} {LAYOUT} {{\n{}\n}};", items.join("\n"))
}

/// The world of `modules`, parsed, verified and lowered by the crate's pipeline, in its context.
#[track_caller]
pub(super) fn lowered_world(modules: &[String]) -> (Rc<Context>, OperationRef) {
    let text = format!("builtin.world {{\n{}\n}};", modules.join("\n"));
    let context = context();
    let world = parse::parse::<World>(ParserConfig::new(context.clone()), "w", &text)
        .into_result()
        .and_then(|world| {
            world.borrow().as_operation().recursively_verify()?;
            Ok(world.as_operation_ref())
        })
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    run_pipeline(&context, world, LOWERING)
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, world)
}

/// The world of `modules`, lowered ([lowered_world]) and emitted as an object, or the error.
pub(super) fn try_object_of(modules: &[String]) -> Result<Vec<u8>, String> {
    let (_context, world) = lowered_world(modules);
    emit_object(world, &EmitOptions::default()).map_err(|report| describe(&report))
}

/// The object of the world of `modules` (see [try_object_of]), which must be emitted.
#[track_caller]
pub(super) fn object_of(modules: &[String]) -> Vec<u8> {
    try_object_of(modules).unwrap_or_else(|message| panic!("{message}"))
}

/// The flag of a local symbol.
const LOCAL: u32 = 0x2;
/// The flag of a hidden symbol.
const HIDDEN: u32 = 0x4;
/// The flag of an undefined symbol.
const UNDEFINED: u32 = 0x10;
/// The flag of an exported symbol.
const EXPORTED: u32 = 0x20;
/// The flag of a symbol with an explicit name.
const EXPLICIT_NAME: u32 = 0x40;
/// The flag of a symbol the linker keeps.
const NO_STRIP: u32 = 0x80;
/// The flag of a segment the linker keeps.
const RETAIN: u32 = 0x4;

/// A symbol as read back: its kind, flags, index (or for data, its definition) and name.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Sym {
    Func(u32, u32, Option<&'static str>),
    Global(u32, u32, Option<&'static str>),
    Table(u32, u32, Option<&'static str>),
    /// Flags, name, and for a defined one its segment, offset and size.
    Data(u32, &'static str, Option<(u32, u32, u32)>),
}

/// A relocation as read back: the section it applies to (`CODE` or `DATA`), its type, its offset,
/// the symbol (or type) it names, its addend, and the provisional value its placeholder holds.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Rel(&'static str, RelocationType, u32, u32, i64, i64);

/// An object read back with `wasmparser`.
#[derive(Debug, Default)]
struct Read {
    /// Each section by its name: the standard ones lower case, a custom one by its name.
    sections: Vec<String>,
    /// Each import: its module, field and kind.
    imports: Vec<(String, String, &'static str)>,
    /// The minimum, in pages, of each memory import.
    memory_pages: Vec<u64>,
    /// Each export: its name and the function's index.
    exports: Vec<(String, u32)>,
    symbols: Vec<Sym>,
    /// Each segment: its name, the logarithm of its alignment and its flags.
    segments: Vec<(String, u32, u32)>,
    /// Each data segment: its provisional offset and its bytes.
    data: Vec<(i32, Vec<u8>)>,
    /// Each init function: its priority and its symbol.
    inits: Vec<(u32, u32)>,
    relocs: Vec<Rel>,
    features: Vec<String>,
}

/// The string `name` with a static lifetime, for comparing with what a test states.
fn leak(name: &str) -> &'static str {
    String::leak(String::from(name))
}

/// The value of the five-byte padded LEB128 at `bytes`, signed or unsigned; it panics when the
/// bytes are not a padded LEB128 of five bytes.
#[track_caller]
fn padded(bytes: &[u8], signed: bool) -> i64 {
    assert!(
        bytes.len() == 5 && bytes[..4].iter().all(|b| b & 0x80 != 0) && bytes[4] & 0x80 == 0,
        "no padded LEB128 at a relocation: {bytes:02x?}"
    );
    let mut value = 0i64;
    for (k, byte) in bytes.iter().enumerate() {
        value |= i64::from(byte & 0x7f) << (7 * k);
    }
    if signed && bytes[4] & 0x40 != 0 {
        value -= 1 << 35;
    }
    value
}

/// The object `bytes` read back (see [Read]).
#[track_caller]
fn read(bytes: &[u8]) -> Read {
    let mut read = Read::default();
    // Each section by its index among all sections, with the range of its payload.
    let mut payloads: Vec<core::ops::Range<u64>> = Vec::new();
    let mut relocs = Vec::new();
    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload.expect("the object parses");
        let (name, range) = match &payload {
            Payload::TypeSection(r) => ("type", r.range()),
            Payload::ImportSection(r) => ("import", r.range()),
            Payload::FunctionSection(r) => ("function", r.range()),
            Payload::TableSection(r) => ("table", r.range()),
            Payload::MemorySection(r) => ("memory", r.range()),
            Payload::GlobalSection(r) => ("global", r.range()),
            Payload::ExportSection(r) => ("export", r.range()),
            Payload::StartSection { range, .. } => ("start", range.clone()),
            Payload::ElementSection(r) => ("element", r.range()),
            Payload::DataCountSection { range, .. } => ("datacount", range.clone()),
            Payload::CodeSectionStart {
                unchecked_range, ..
            } => ("code", unchecked_range.clone()),
            Payload::DataSection(r) => ("data", r.range()),
            Payload::CustomSection(r) => (leak(r.name()), r.range()),
            _ => continue,
        };
        read.sections.push(String::from(name));
        payloads.push(range);
        match payload {
            Payload::ImportSection(reader) => {
                for import in reader.into_imports() {
                    let import = import.expect("an import");
                    let kind = match import.ty {
                        TypeRef::Func(_) => "func",
                        TypeRef::Table(_) => "table",
                        TypeRef::Memory(memory) => {
                            read.memory_pages.push(memory.initial);
                            "memory"
                        }
                        TypeRef::Global(_) => "global",
                        _ => "other",
                    };
                    read.imports.push((import.module.into(), import.name.into(), kind));
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.expect("an export");
                    read.exports.push((export.name.into(), export.index));
                }
            }
            Payload::DataSection(reader) => {
                for data in reader {
                    let data = data.expect("a segment");
                    let wasmparser::DataKind::Active { offset_expr, .. } = data.kind else {
                        panic!("a passive segment in an object");
                    };
                    let offset = match offset_expr.get_operators_reader().read() {
                        Ok(wasmparser::Operator::I32Const { value }) => value,
                        other => panic!("a segment at {other:?}"),
                    };
                    read.data.push((offset, data.data.to_vec()));
                }
            }
            Payload::CustomSection(reader) => match reader.as_known() {
                KnownCustom::Linking(linking) => {
                    for subsection in linking.subsections() {
                        match subsection.expect("a subsection") {
                            Linking::SymbolTable(symbols) => {
                                for symbol in symbols {
                                    let name = |name: Option<&str>| name.map(leak);
                                    read.symbols.push(match symbol.expect("a symbol") {
                                        SymbolInfo::Func {
                                            flags,
                                            index,
                                            name: n,
                                        } => Sym::Func(flags.bits(), index, name(n)),
                                        SymbolInfo::Global {
                                            flags,
                                            index,
                                            name: n,
                                        } => Sym::Global(flags.bits(), index, name(n)),
                                        SymbolInfo::Table {
                                            flags,
                                            index,
                                            name: n,
                                        } => Sym::Table(flags.bits(), index, name(n)),
                                        SymbolInfo::Data {
                                            flags,
                                            name: n,
                                            symbol,
                                        } => Sym::Data(
                                            flags.bits(),
                                            leak(n),
                                            symbol.map(|d| (d.index, d.offset, d.size)),
                                        ),
                                        other => panic!("a symbol of another kind: {other:?}"),
                                    });
                                }
                            }
                            Linking::SegmentInfo(segments) => {
                                for segment in segments {
                                    let segment = segment.expect("a segment");
                                    read.segments.push((
                                        segment.name.into(),
                                        segment.alignment,
                                        segment.flags.bits(),
                                    ));
                                }
                            }
                            Linking::InitFuncs(inits) => {
                                for init in inits {
                                    let init = init.expect("an init function");
                                    read.inits.push((init.priority, init.symbol_index));
                                }
                            }
                            other => panic!("another subsection: {other:?}"),
                        }
                    }
                }
                KnownCustom::Reloc(reloc) => {
                    let section = reloc.section_index() as usize;
                    let which = if reader.name() == "reloc.CODE" {
                        "CODE"
                    } else {
                        "DATA"
                    };
                    for entry in reloc.entries() {
                        relocs.push((which, section, entry.expect("a relocation")));
                    }
                }
                _ => {
                    if reader.name() == "target_features" {
                        let mut data = wasmparser::BinaryReader::new(reader.data(), 0);
                        for _ in 0..data.read_var_u32().unwrap() {
                            let prefix = data.read_u8().unwrap();
                            let name = data.read_string().unwrap();
                            read.features.push(format!("{}{name}", prefix as char));
                        }
                    }
                }
            },
            _ => {}
        }
    }
    let mut last = (String::new(), 0u32);
    for (which, section, entry) in relocs {
        let range = payloads[section].clone();
        let name = &read.sections[section];
        assert_eq!(name.to_uppercase(), which, "reloc.{which} names the section {name}");
        let payload = &bytes[range.start as usize..range.end as usize];
        let start = entry.offset as usize;
        assert!(
            last.0 != which || entry.offset > last.1,
            "relocations of {which} out of order at {}",
            entry.offset
        );
        last = (String::from(which), entry.offset);
        let value = match entry.ty {
            RelocationType::TableIndexI32 | RelocationType::MemoryAddrI32 => {
                i64::from(i32::from_le_bytes(payload[start..start + 4].try_into().unwrap()))
            }
            RelocationType::TableIndexSleb | RelocationType::MemoryAddrSleb => {
                padded(&payload[start..start + 5], true)
            }
            _ => padded(&payload[start..start + 5], false),
        };
        read.relocs
            .push(Rel(which, entry.ty, entry.offset, entry.index, entry.addend, value));
    }
    read
}

/// The object of the world of `modules`, read back, after checking that the linker accepts it
/// (with `--allow-undefined`, so that undefined symbols do not stop the link).
#[track_caller]
fn read_object(modules: &[String]) -> Read {
    let object = object_of(modules);
    super::link::link(&[&object], &["--allow-undefined"])
        .unwrap_or_else(|message| panic!("the linker rejects the object:\n{message}"));
    read(&object)
}

/// The strings `names`, owned.
fn strings(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| String::from(*name)).collect()
}

/// An import of `module` and `field` of the kind `kind`.
fn import(module: &str, field: &str, kind: &'static str) -> (String, String, &'static str) {
    (String::from(module), String::from(field), kind)
}

/// The import of the memory every object has.
fn memory() -> (String, String, &'static str) {
    import("env", "__linear_memory", "memory")
}

/// The declaration of the function table, as the lowering writes it.
const TABLE: &str = "wasm.table private declaration @__indirect_function_table : 0 x \
                     !wasm.funcref attributes { link_name = \"__indirect_function_table\" };";

/// The declaration of the memory, as the lowering writes it.
const MEMORY: &str = "wasm.memory private declaration @__linear_memory : 0 attributes { link_name \
                      = \"__linear_memory\" };";

// ---- Sections -----------------------------------------------------------------------------------

#[test]
fn an_object_of_one_function_imports_only_the_memory_and_has_one_symbol_and_no_relocation() {
    // The function table is declared and not used: the object does not import it.
    let read = read_object(&[module(
        "m",
        &[
            TABLE,
            MEMORY,
            "builtin.function public extern(\"wasm\") @f(%x: i32) -> i32 { builtin.ret %x : \
             (i32); } attributes { wasm.export_name = \"f\" };",
        ],
    )]);
    assert_eq!(
        read.sections,
        strings(&["type", "import", "function", "export", "code", "linking", "producers"])
    );
    assert_eq!(read.imports, [memory()]);
    assert_eq!(read.exports, [(String::from("f"), 0)]);
    assert_eq!(read.symbols, [Sym::Func(EXPORTED, 0, Some("m::f"))]);
    assert!(read.segments.is_empty() && read.data.is_empty() && read.inits.is_empty());
    assert!(read.relocs.is_empty() && read.features.is_empty());
}

#[test]
fn the_sections_of_an_object_are_in_the_order_the_linker_reads_them() {
    let read = read_object(&[module(
        "m",
        &[
            "wasm.table public @t : 1 x !wasm.funcref;",
            "wasm.global public @g : i32 { %v = wasm.const 3 : i32; builtin.yield %v; };",
            "builtin.function public extern(\"C\") @callee(%x: i32) -> i32 { builtin.ret %x : \
             (i32); };",
            "builtin.global_variable public @stored : (i32) -> i32 { %f = builtin.function_ref \
             @callee : (i32) -> i32; builtin.yield %f; };",
            "builtin.function public extern(\"C\") @f(%x: i32) -> i32 {
            %r = builtin.call @callee(%x) : extern(\"C\") (i32) -> i32;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"f\" };",
        ],
    )]);
    assert_eq!(
        read.sections,
        strings(&[
            "type",
            "import",
            "function",
            "table",
            "global",
            "export",
            "code",
            "data",
            "linking",
            "reloc.CODE",
            "reloc.DATA",
            "producers",
            "target_features",
        ])
    );
    // The defined table is table 1, after the imported function table.
    assert_eq!(read.imports, [memory(), import("env", "__indirect_function_table", "table")]);
    assert_eq!(
        read.symbols[3..],
        [
            Sym::Table(0, 1, Some("m::t")),
            Sym::Table(UNDEFINED | NO_STRIP, 0, None),
            Sym::Data(0, "m::stored", Some((0, 0, 4))),
        ]
    );
    assert_eq!(read.features, strings(&["+reference-types"]));
}

// ---- Relocations --------------------------------------------------------------------------------

#[test]
fn each_fixup_in_code_is_a_relocation_at_its_placeholder_holding_the_provisional_value() {
    let read = read_object(&[module(
        "m",
        &[
            TABLE,
            MEMORY,
            "wasm.global public @g : mut i32 { %v = wasm.const 0 : i32; builtin.yield %v; };",
            "builtin.global_variable public @d : array<i32; 2> { %z = builtin.zero : array<i32; \
             2>; builtin.yield %z; };",
            "builtin.function public extern(\"wasm\") @callee(%x: i32) -> i32 { builtin.ret %x : \
             (i32); };",
            "builtin.function public extern(\"wasm\") @all(%x: i32) -> (i32, i32) {
            %c = builtin.call @callee(%x) : extern(\"wasm\") (i32) -> i32;
            %v = wasm.global_get @g : i32;
            %a = wasm.data_addr @d + 4;
            %l = wasm.load @__linear_memory, %a : i32;
            %s = wasm.func_index @callee;
            %r = wasm.call_indirect @__indirect_function_table, %s(%c) : extern(\"wasm\") (i32) -> \
             i32;
            %p, %q = wasm.block %r {
            ^body(%y: i32):
                wasm.end %y, %v : (i32, i32);
            } : (i32) -> (i32, i32);
            %t = wasm.add %p, %l;
            builtin.ret %t, %q : (i32, i32);
        };",
        ],
    )]);
    assert_eq!(read.imports, [memory(), import("env", "__indirect_function_table", "table")]);
    assert_eq!(
        read.symbols,
        [
            Sym::Func(0, 0, Some("m::callee")),
            Sym::Func(0, 1, Some("m::all")),
            Sym::Global(0, 0, Some("m::g")),
            Sym::Table(UNDEFINED | NO_STRIP, 0, None),
            Sym::Data(0, "m::d", Some((0, 0, 8))),
        ]
    );
    // The code section: the count (1), `@callee` (its size, no locals, `local.get 0 end`), then
    // `@all`, its size and five locals (`01 05 7f`) from offset 7: `local.get 0`, `call`.
    assert_eq!(
        read.relocs,
        [
            Rel("CODE", RelocationType::FunctionIndexLeb, 13, 0, 0, 0),
            Rel("CODE", RelocationType::GlobalIndexLeb, 21, 2, 0, 0),
            Rel("CODE", RelocationType::MemoryAddrSleb, 29, 4, 4, 4),
            Rel("CODE", RelocationType::TableIndexSleb, 42, 0, 0, 0),
            Rel("CODE", RelocationType::TypeIndexLeb, 48, 0, 0, 0),
            Rel("CODE", RelocationType::TableNumberLeb, 53, 3, 0, 0),
            // The block type `(i32) -> (i32, i32)` is the type 1, a padded unsigned LEB128.
            Rel("CODE", RelocationType::TypeIndexLeb, 59, 1, 0, 1),
        ]
    );
    assert_eq!(read.features, strings(&["+call-indirect-overlong", "+multivalue"]));
}

#[test]
fn data_of_each_kind_has_its_segment_and_its_fixups_are_relocations_of_the_data_section() {
    let read = read_object(&[module(
        "m",
        &[
            "builtin.global_variable public @plain : i32 { %v = arith.constant 7 : i32; \
             builtin.yield %v; };",
            "builtin.global_variable public @small : i32 { %v = arith.constant 11 : i32; \
             builtin.yield %v; } attributes { section = \"dispatch\" };",
            "builtin.global_variable public @zero : array<i8; 3> { %z = builtin.zero : array<i8; \
             3>; builtin.yield %z; };",
            "builtin.global_variable public @wide : i64 { %v = arith.constant 22 : i64; \
             builtin.yield %v; } attributes { section = \"dispatch\" };",
            "builtin.global_variable public @array : array<i32; 4> { %z = builtin.zero : \
             array<i32; 4>; builtin.yield %z; };",
            "builtin.global_variable public @pointer : ptr<i32> { %t = builtin.global_symbol \
             @array : ptr<array<i32; 4>>; %q = mem.gep %t, [2]; builtin.yield %q; };",
            "builtin.function public extern(\"C\") @double(%x: i32) -> i32 { %r = arith.add %x, \
             %x; builtin.ret %r : (i32); };",
            "builtin.global_variable public @function : (i32) -> i32 { %f = builtin.function_ref \
             @double : (i32) -> i32; builtin.yield %f; };",
        ],
    )]);
    // A table slot needs the function table, which the object imports and keeps.
    assert_eq!(read.imports, [memory(), import("env", "__indirect_function_table", "table")]);
    assert_eq!(
        read.symbols,
        [
            Sym::Func(0, 0, Some("m::double")),
            Sym::Table(UNDEFINED | NO_STRIP, 0, None),
            Sym::Data(0, "m::plain", Some((0, 0, 4))),
            Sym::Data(0, "m::small", Some((1, 0, 4))),
            Sym::Data(0, "m::zero", Some((2, 0, 3))),
            // After the 4-aligned `@small`, at a multiple of its own alignment.
            Sym::Data(0, "m::wide", Some((1, 8, 8))),
            Sym::Data(0, "m::array", Some((3, 0, 16))),
            Sym::Data(0, "m::pointer", Some((4, 0, 4))),
            Sym::Data(0, "m::function", Some((5, 0, 4))),
        ]
    );
    // The section's segment is 8-aligned (the largest of its members) and retained; zero data is
    // `.bss`.
    assert_eq!(
        read.segments,
        [
            (String::from(".data.m::plain"), 2, 0),
            (String::from("dispatch"), 3, RETAIN),
            (String::from(".bss.m::zero"), 0, 0),
            (String::from(".bss.m::array"), 2, 0),
            (String::from(".data.m::pointer"), 2, 0),
            (String::from(".data.m::function"), 2, 0),
        ]
    );
    // Each segment at its provisional offset, the zeros written out; the pointer holds the
    // provisional address of `@array` (28) plus 8.
    assert_eq!(
        read.data,
        [
            (0, Vec::from([7, 0, 0, 0])),
            (8, Vec::from([11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0])),
            (24, Vec::from([0; 3])),
            (28, Vec::from([0; 16])),
            (44, Vec::from([36, 0, 0, 0])),
            (48, Vec::from([0; 4])),
        ]
    );
    // The data section: the count (1), then each segment's header (`00 41 <offset> 0b <size>`,
    // five bytes here) and its bytes: the pointer's bytes are at 65, the function's at 74.
    assert_eq!(
        read.relocs,
        [
            Rel("DATA", RelocationType::MemoryAddrI32, 65, 6, 8, 36),
            Rel("DATA", RelocationType::TableIndexI32, 74, 0, 0, 0),
        ]
    );
    assert!(read.features.is_empty());
}

// ---- Symbols ------------------------------------------------------------------------------------

#[test]
fn the_binding_of_each_symbol_follows_the_visibility_of_its_definition() {
    let mut items = Vec::new();
    for visibility in ["public", "internal", "private"] {
        items.push(format!(
            "builtin.function {visibility} extern(\"wasm\") @f_{visibility}() {{ builtin.ret; }};"
        ));
        items.push(format!(
            "wasm.global {visibility} @g_{visibility} : i32 {{ %v = wasm.const 0 : i32; \
             builtin.yield %v; }};"
        ));
        items.push(format!("wasm.table {visibility} @t_{visibility} : 1 x !wasm.funcref;"));
        items.push(format!(
            "builtin.global_variable {visibility} @d_{visibility} : i32 {{ %v = arith.constant 1 \
             : i32; builtin.yield %v; }};"
        ));
    }
    let items = items.iter().map(String::as_str).collect::<Vec<_>>();
    let read = read_object(&[module("m", &items)]);
    assert_eq!(
        read.symbols,
        [
            Sym::Func(0, 0, Some("m::f_public")),
            Sym::Func(HIDDEN, 1, Some("m::f_internal")),
            Sym::Func(LOCAL, 2, Some("m::f_private")),
            Sym::Global(0, 0, Some("m::g_public")),
            Sym::Global(HIDDEN, 1, Some("m::g_internal")),
            Sym::Global(LOCAL, 2, Some("m::g_private")),
            Sym::Table(0, 0, Some("m::t_public")),
            Sym::Table(HIDDEN, 1, Some("m::t_internal")),
            Sym::Table(LOCAL, 2, Some("m::t_private")),
            Sym::Data(0, "m::d_public", Some((0, 0, 4))),
            Sym::Data(HIDDEN, "m::d_internal", Some((1, 0, 4))),
            Sym::Data(LOCAL, "m::d_private", Some((2, 0, 4))),
        ]
    );
    assert_eq!(
        read.segments,
        [
            (String::from(".data.m::d_public"), 2, 0),
            (String::from(".data.m::d_internal"), 2, 0),
            (String::from(".data.m::d_private"), 2, 0),
        ]
    );
    assert_eq!(read.features, strings(&["+reference-types"]));
}

#[test]
fn an_undefined_symbol_is_named_and_imported_from_its_host_only_when_it_names_one() {
    let read = read_object(&[module(
        "m",
        &[
            "builtin.function public extern(\"wasm\") @f_linked(i32) -> i32;",
            "builtin.function public extern(\"wasm\") @f_hosted(i32) -> i32 attributes { \
             wasm.import_module = \"host\" };",
            "wasm.global public declaration @g_linked : mut i32;",
            "wasm.global public declaration @g_hosted : i32 attributes { wasm.import_module = \
             \"host\" };",
            "wasm.table public declaration @t_linked : 1 x !wasm.funcref;",
            "wasm.table public declaration @t_hosted : 1 x !wasm.externref attributes { \
             wasm.import_module = \"host\" };",
            "builtin.global_variable public @d_linked : i32;",
            "builtin.function public extern(\"wasm\") @use(%x: i32) -> i32 {
            %a = builtin.call @f_linked(%x) : extern(\"wasm\") (i32) -> i32;
            %b = builtin.call @f_hosted(%a) : extern(\"wasm\") (i32) -> i32;
            %g = wasm.global_get @g_linked : i32;
            %h = wasm.global_get @g_hosted : i32;
            %n = wasm.table_size @t_linked;
            %m = wasm.table_size @t_hosted;
            %d = wasm.data_addr @d_linked + 8;
            %s1 = wasm.add %b, %g;
            %s2 = wasm.add %s1, %h;
            %s3 = wasm.add %s2, %n;
            %s4 = wasm.add %s3, %m;
            %s5 = wasm.add %s4, %d;
            builtin.ret %s5 : (i32);
        };",
        ],
    )]);
    assert_eq!(
        read.imports,
        [
            memory(),
            import("env", "m::f_linked", "func"),
            import("host", "m::f_hosted", "func"),
            import("env", "m::t_linked", "table"),
            import("host", "m::t_hosted", "table"),
            import("env", "m::g_linked", "global"),
            import("host", "m::g_hosted", "global"),
        ]
    );
    assert_eq!(
        read.symbols,
        [
            Sym::Func(UNDEFINED, 0, None),
            Sym::Func(UNDEFINED | EXPLICIT_NAME, 1, Some("m::f_hosted")),
            Sym::Func(0, 2, Some("m::use")),
            Sym::Global(UNDEFINED, 0, None),
            Sym::Global(UNDEFINED | EXPLICIT_NAME, 1, Some("m::g_hosted")),
            Sym::Table(UNDEFINED, 0, None),
            Sym::Table(UNDEFINED | EXPLICIT_NAME, 1, Some("m::t_hosted")),
            Sym::Data(UNDEFINED, "m::d_linked", None),
        ]
    );
    // Each placeholder holds the import's index; an undefined address is 0, its addend kept.
    assert_eq!(
        read.relocs,
        [
            Rel("CODE", RelocationType::FunctionIndexLeb, 8, 0, 0, 0),
            Rel("CODE", RelocationType::FunctionIndexLeb, 14, 1, 0, 1),
            Rel("CODE", RelocationType::GlobalIndexLeb, 22, 3, 0, 0),
            Rel("CODE", RelocationType::GlobalIndexLeb, 30, 4, 0, 1),
            Rel("CODE", RelocationType::TableNumberLeb, 39, 5, 0, 0),
            Rel("CODE", RelocationType::TableNumberLeb, 48, 6, 0, 1),
            Rel("CODE", RelocationType::MemoryAddrSleb, 70, 7, 8, 0),
        ]
    );
    assert!(read.segments.is_empty() && read.data.is_empty());
    assert_eq!(
        read.features,
        strings(&["+call-indirect-overlong", "+mutable-globals", "+reference-types"])
    );
}

#[test]
fn the_start_function_is_the_one_init_function_and_an_export_name_is_exported() {
    let read = read_object(&[module(
        "m",
        &[
            "builtin.function private extern(\"wasm\") @init() { builtin.ret; } attributes { \
             wasm.start = #builtin.unit };",
            "builtin.function public extern(\"wasm\") @f(%x: i32) -> i32 { builtin.ret %x : \
             (i32); } attributes { wasm.export_name = \"renamed\" };",
        ],
    )]);
    assert_eq!(
        read.symbols,
        [Sym::Func(LOCAL, 0, Some("m::init")), Sym::Func(EXPORTED, 1, Some("m::f")),]
    );
    assert_eq!(read.inits, [(65_535, 0)]);
    // The start function is not exported as `_initialize`, as a complete module exports it.
    assert_eq!(read.exports, [(String::from("renamed"), 1)]);
}

#[test]
fn the_stack_pointer_is_an_undefined_mutable_global_of_env() {
    let read = read_object(&[module(
        "m",
        &[
            "builtin.function public extern(\"C\") @f(%x: i32) -> i32 { %p = mem.alloca i32; \
             mem.store %x, %p; %v = mem.load %p; builtin.ret %v : (i32); };",
        ],
    )]);
    assert_eq!(read.imports, [memory(), import("env", "__stack_pointer", "global")]);
    assert_eq!(read.symbols, [Sym::Func(0, 0, Some("m::f")), Sym::Global(UNDEFINED, 0, None)]);
    assert_eq!(
        read.relocs,
        [
            Rel("CODE", RelocationType::GlobalIndexLeb, 6, 1, 0, 0),
            Rel("CODE", RelocationType::GlobalIndexLeb, 19, 1, 0, 0),
            Rel("CODE", RelocationType::GlobalIndexLeb, 41, 1, 0, 0),
        ]
    );
    assert_eq!(read.features, strings(&["+mutable-globals"]));
}

// ---- Features -----------------------------------------------------------------------------------

#[test]
fn the_features_of_an_object_are_those_it_uses_under_llvms_names() {
    let cases: [(&[&str], &[&str]); 9] =
        [
            (
                &["builtin.function public extern(\"wasm\") @f(%x: i32) -> i32 { %e = \
                   wasm.extend8_s %x; builtin.ret %e : (i32); };"],
                &["+sign-ext"],
            ),
            (
                &["builtin.function public extern(\"wasm\") @f(%x: f64) -> i32 { %t = \
                   wasm.trunc_sat_s %x <{ ty = i32 }>; builtin.ret %t : (i32); };"],
                &["+nontrapping-fptoint"],
            ),
            (
                &[
                    MEMORY,
                    "builtin.function public extern(\"wasm\") @f(%d: i32, %s: i32, %n: i32) { \
                     wasm.memory_copy @__linear_memory, @__linear_memory, %d, %s, %n; \
                     builtin.ret; };",
                ],
                &["+bulk-memory", "+bulk-memory-opt"],
            ),
            (
                &["builtin.function public extern(\"wasm\") @f(%x: i32) -> (i32, i32) { \
                   builtin.ret %x, %x : (i32, i32); };"],
                &["+multivalue"],
            ),
            (
                &["builtin.function public extern(\"wasm\") @f() -> i32 { %n = wasm.ref_null : \
                   !wasm.funcref; %z = wasm.ref_is_null %n; builtin.ret %z : (i32); };"],
                &["+reference-types"],
            ),
            (
                &["builtin.function public extern(\"wasm\") @f(%x: i32) -> i32 { \
                   builtin.tail_call @f(%x) : extern(\"wasm\") (i32) -> i32; };"],
                &["+tail-call"],
            ),
            (
                &["builtin.function public extern(\"C\") @f(%x: i32) -> i32 { %p = mem.alloca \
                   i32; mem.store %x, %p; %v = mem.load %p; builtin.ret %v : (i32); };"],
                &["+mutable-globals"],
            ),
            (
                &["wasm.global public @g : mut i32 { %v = wasm.const 7 : i32; builtin.yield %v; \
                   } attributes { wasm.export_name = \"m::g\" };"],
                &["+mutable-globals"],
            ),
            (
                &[
                    TABLE,
                    "builtin.function public extern(\"wasm\") @f(%i: i32) { wasm.call_indirect \
                     @__indirect_function_table, %i() : extern(\"wasm\") () -> (); builtin.ret; };",
                ],
                &["+call-indirect-overlong"],
            ),
        ];
    for (items, features) in cases {
        let read = read_object(&[module("m", items)]);
        assert_eq!(read.features, strings(features), "{items:?}");
    }
}

#[test]
fn a_global_exported_under_its_link_name_is_an_exported_symbol_with_no_export_entry() {
    let read = read_object(&[module(
        "m",
        &[
            "wasm.global public @g : mut i32 { %v = wasm.const 7 : i32; builtin.yield %v; } \
             attributes { wasm.export_name = \"m::g\" };",
        ],
    )]);
    assert_eq!(read.symbols, [Sym::Global(EXPORTED, 0, Some("m::g"))]);
    assert!(read.exports.is_empty());
}

#[test]
fn data_beyond_a_32_bit_memory_is_an_error_naming_the_global_variable() {
    let error = try_object_of(&[module(
        "m",
        &[
            "builtin.global_variable public @small : i32 { %v = arith.constant 1 : i32; \
             builtin.yield %v; };",
            "builtin.global_variable public @huge : array<i8; 4294967296> { %z = builtin.zero : \
             array<i8; 4294967296>; builtin.yield %z; };",
        ],
    )])
    .expect_err("the data does not fit");
    assert!(
        error.contains("the data of the unit does not fit the 4 GiB of a 32-bit linear memory"),
        "{error}"
    );
    assert!(
        error.contains(
            "this global variable is in the segment `.bss.m::huge`, which would end at 4294967300 \
             in the object's data"
        ),
        "{error}"
    );
}

#[test]
fn the_memory_import_has_as_many_pages_as_the_objects_data_needs() {
    let function = "builtin.function public extern(\"wasm\") @f(%x: i32) -> i32 { builtin.ret %x \
                    : (i32); } attributes { wasm.export_name = \"f\" };";
    assert_eq!(read_object(&[module("m", &[function])]).memory_pages, [0]);
    // A page and one byte of data, zero or not, and four bytes more: two pages.
    let read = read_object(&[module(
        "m",
        &[
            "builtin.global_variable private @zeros : array<i8; 65537> { %z = builtin.zero : \
             array<i8; 65537>; builtin.yield %z; };",
            "builtin.global_variable private @word : i32 { %v = arith.constant 9 : i32; \
             builtin.yield %v; };",
        ],
    )]);
    assert_eq!(read.imports, [memory()]);
    assert_eq!(read.memory_pages, [2]);
    let read = read_object(&[module(
        "m",
        &[
            "builtin.global_variable private @page : array<i8; 65536> { %z = builtin.zero : \
             array<i8; 65536>; builtin.yield %z; };",
        ],
    )]);
    assert_eq!(read.memory_pages, [1]);
}

#[test]
fn an_alias_other_modules_may_name_is_a_second_symbol_of_its_targets_function() {
    let read = read_object(&[module(
        "m",
        &[
            "builtin.function private extern(\"wasm\") @first(%x: i32) -> i32 { builtin.ret %x : \
             (i32); };",
            "builtin.function public extern(\"wasm\") @target(%x: i32) -> i32 { builtin.ret %x : \
             (i32); } attributes { link_name = \"target\" };",
            "builtin.function_alias public @al -> @target attributes { link_name = \"al\" };",
            "builtin.function_alias internal @inner -> @al attributes { link_name = \"inner\" };",
            "builtin.function_alias private @mine -> @first;",
            "builtin.global_variable private @d : i32 { %v = arith.constant 1 : i32; \
             builtin.yield %v; };",
        ],
    )]);
    // After the data, in the order written: the target's index, the alias's own name and binding;
    // a private alias has none.
    assert_eq!(
        read.symbols,
        [
            Sym::Func(LOCAL, 0, Some("m::first")),
            Sym::Func(0, 1, Some("target")),
            Sym::Data(LOCAL, "m::d", Some((0, 0, 4))),
            Sym::Func(0, 1, Some("al")),
            Sym::Func(HIDDEN, 1, Some("inner")),
        ]
    );
    assert!(read.exports.is_empty());
}
