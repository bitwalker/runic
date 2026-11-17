//! The unit: every `builtin.module` under a root linked as one module by link name, its imports,
//! exports, globals, tables, memories, element segments and names, and the diagnostics of the
//! whole module, through [emit_module].

use alloc::{format, rc::Rc, string::String, vec::Vec};

use runic_ir::{
    Context, Op, OperationRef, Report,
    dialects::builtin::World,
    parse::{self, ParserConfig},
    testing::text::describe,
};
use wasmparser::{KnownCustom, Name, Payload};

use super::support::{payloads, producers_text};
use crate::{
    emit::{
        EmitOptions,
        body::BodyConfig,
        collect::collect,
        desc::{Entity, FixupKind},
        emit_module,
    },
    tests::{LAYOUT, LOWERING, context, error_of, parse_module, run_pipeline},
    text::print,
};

/// `text`, a world, parsed and verified in `context`.
fn parse_world_in(context: &Rc<Context>, text: &str) -> Result<OperationRef, Report> {
    let world =
        parse::parse::<World>(ParserConfig::new(context.clone()), "w", text).into_result()?;
    world.borrow().as_operation().recursively_verify()?;
    Ok(world.as_operation_ref())
}

/// `text`, a world, parsed and verified in a new context.
#[track_caller]
fn parse_world(text: &str) -> (Rc<Context>, OperationRef) {
    let context = context();
    let world =
        parse_world_in(&context, text).unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, world)
}

/// `text`, a world, parsed without being verified.
#[track_caller]
fn parse_world_unverified(text: &str) -> (Rc<Context>, OperationRef) {
    let context = context();
    let mut config = ParserConfig::new(context.clone());
    config.verify = false;
    let world = parse::parse::<World>(config, "w", text)
        .into_result()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, world.as_operation_ref())
}

/// A world of the modules `modules`.
fn world_of(modules: &[&str]) -> String {
    format!("builtin.world {{\n{}\n}};", modules.join("\n"))
}

/// A module `@name` of `body`.
fn module(name: &str, body: &str) -> String {
    format!("builtin.module public @{name} {{\n{body}\n}};")
}

/// The module emitted from `root` with `options`, validated, as text; an error panics.
#[track_caller]
fn wat_with(root: OperationRef, options: &EmitOptions) -> String {
    let bytes = emit_module(root, options).unwrap_or_else(|report| panic!("{}", describe(&report)));
    print(&bytes).unwrap_or_else(|report| panic!("{report}"))
}

/// The module emitted from `root` with the default options, as text.
#[track_caller]
fn wat(root: OperationRef) -> String {
    wat_with(root, &EmitOptions::default())
}

/// The text of the world `text` emitted.
#[track_caller]
fn wat_of_world(text: &str) -> String {
    let (_context, world) = parse_world(text);
    wat(world)
}

/// The error of emitting the world `text`, with its labels.
#[track_caller]
fn error_of_world(text: &str) -> String {
    let (_context, world) = parse_world(text);
    error_of(emit_module(world, &EmitOptions::default()))
}

/// The whole text of a module whose items are `items`, as `wasmprinter` prints it.
fn whole(items: &[&str]) -> String {
    let mut text = String::from("(module\n");
    for item in items {
        text.push_str("  ");
        text.push_str(item);
        text.push('\n');
    }
    text.push_str(&producers_text());
    text.push_str(")\n");
    text
}

/// Asserts that `haystack` holds every one of `needles`.
#[track_caller]
fn assert_mentions(haystack: &str, needles: &[&str]) {
    for needle in needles {
        assert!(crate::tests::mentions(haystack, needle), "no `{needle}` in:\n{haystack}");
    }
}

/// A public `extern("wasm")` function `@name` of `(i32) -> i32` that adds `k` to its parameter.
fn adder(name: &str, k: i32, attributes: &str) -> String {
    format!(
        "builtin.function public extern(\"wasm\") @{name}(%x: i32) -> i32 {{\n    %k = wasm.const \
         {k} : i32;\n    %y = wasm.add %x, %k;\n    builtin.ret %y : (i32);\n}}{attributes};"
    )
}

/// A function `@name` of `(i32) -> i32` that calls each callee of `callees` in turn on its
/// parameter, by the path given.
fn caller(name: &str, callees: &[&str], attributes: &str) -> String {
    let mut body = String::new();
    let mut value = String::from("%x");
    for (index, callee) in callees.iter().enumerate() {
        body.push_str(&format!(
            "    %r{index} = builtin.call {callee}({value}) : extern(\"wasm\") (i32) -> i32;\n"
        ));
        value = format!("%r{index}");
    }
    format!(
        "builtin.function public extern(\"wasm\") @{name}(%x: i32) -> i32 {{\n{body}    \
         builtin.ret {value} : (i32);\n}}{attributes};"
    )
}

/// A declaration `@name` of `(i32) -> i32` with the attributes `attributes`.
fn declaration(name: &str, attributes: &str) -> String {
    format!("builtin.function private extern(\"wasm\") @{name}(i32) -> i32{attributes};")
}

// ---- One module ------------------------------------------------------------------------------

#[test]
fn a_module_of_two_functions_one_calling_the_other_is_one_wasm_module() {
    let text = world_of(&[&module(
        "m",
        &[adder("inc", 1, ""), caller("twice", &["@inc", "@inc"], "")].join("\n"),
    )]);
    assert_eq!(
        wat_of_world(&text),
        whole(&[
            "(type (;0;) (func (param i32) (result i32)))",
            "(func $m::inc (;0;) (type 0) (param i32) (result i32)",
            "  local.get 0",
            "  i32.const 1",
            "  i32.add",
            ")",
            "(func $m::twice (;1;) (type 0) (param i32) (result i32)",
            "  local.get 0",
            "  call $m::inc",
            "  call $m::inc",
            ")",
        ])
    );
}

#[test]
fn a_declaration_is_an_import_whose_field_is_its_link_name_from_env_or_its_import_module() {
    let text = world_of(&[&module(
        "m",
        &[
            declaration("plain", ""),
            declaration("named", " attributes { link_name = \"ext\" }"),
            declaration(
                "hosted",
                " attributes { link_name = \"get\", wasm.import_module = \"host\" }",
            ),
            caller("f", &["@plain", "@named", "@hosted"], ""),
        ]
        .join("\n"),
    )]);
    assert_eq!(
        wat_of_world(&text),
        whole(&[
            "(type (;0;) (func (param i32) (result i32)))",
            "(import \"env\" \"m::plain\" (func $m::plain (;0;) (type 0) (param i32) (result \
             i32)))",
            "(import \"env\" \"ext\" (func $ext (;1;) (type 0) (param i32) (result i32)))",
            "(import \"host\" \"get\" (func $get (;2;) (type 0) (param i32) (result i32)))",
            "(func $m::f (;3;) (type 0) (param i32) (result i32)",
            "  local.get 0",
            "  call $m::plain",
            "  call $ext",
            "  call $get",
            ")",
        ])
    );
}

#[test]
fn a_bare_module_is_a_unit_of_its_own() {
    let (_context, module) = parse_module(&format!(
        "builtin.module public @m {LAYOUT} {{\n{}\n}};",
        adder("inc", 1, " attributes { wasm.export_name = \"inc\" }")
    ));
    let text = wat(module.as_operation_ref());
    assert_mentions(&text, &["(func $m::inc (;0;)", "(export \"inc\" (func $m::inc))"]);
}

// ---- Several modules ---------------------------------------------------------------------------

#[test]
fn modules_are_linked_by_path_by_declaration_and_through_a_nested_module() {
    let a = module(
        "a",
        &[adder("inc", 1, ""), adder("hidden", 2, "").replace("public", "private")].join("\n"),
    );
    let b = module(
        "b",
        &[
            declaration("inc_decl", " attributes { link_name = \"a::inc\" }"),
            declaration("hidden_decl", " attributes { link_name = \"a::hidden\" }"),
            module("inner", &adder("deep", 3, "")),
            caller(
                "f",
                &["::@a::@inc", "@inc_decl", "@hidden_decl", "::@b::@inner::@deep"],
                " attributes { wasm.export_name = \"f\" }",
            ),
        ]
        .join("\n"),
    );
    assert_eq!(
        wat_of_world(&world_of(&[&a, &b])),
        whole(&[
            "(type (;0;) (func (param i32) (result i32)))",
            "(import \"env\" \"a::hidden\" (func $a::hidden (;0;) (type 0) (param i32) (result \
             i32)))",
            "(export \"f\" (func $b::f))",
            "(func $a::inc (;1;) (type 0) (param i32) (result i32)",
            "  local.get 0",
            "  i32.const 1",
            "  i32.add",
            ")",
            // The private definition shares the import's name; the printer tells them apart.
            "(func $\"#func2 a::hidden\" (@name \"a::hidden\") (;2;) (type 0) (param i32) (result \
             i32)",
            "  local.get 0",
            "  i32.const 2",
            "  i32.add",
            ")",
            "(func $b::inner::deep (;3;) (type 0) (param i32) (result i32)",
            "  local.get 0",
            "  i32.const 3",
            "  i32.add",
            ")",
            "(func $b::f (;4;) (type 0) (param i32) (result i32)",
            "  local.get 0",
            "  call $a::inc",
            "  call $a::inc",
            "  call $a::hidden",
            "  call $b::inner::deep",
            ")",
        ])
    );
}

#[test]
fn an_import_declared_in_two_modules_is_one_import() {
    let decl = declaration("ext", " attributes { link_name = \"ext\" }");
    let a = module("a", &[decl.clone(), caller("f", &["@ext"], "")].join("\n"));
    let b = module("b", &[decl, caller("g", &["@ext"], "")].join("\n"));
    let text = wat_of_world(&world_of(&[&a, &b]));
    assert_eq!(text.matches("(import ").count(), 1, "{text}");
    assert_mentions(
        &text,
        &["(import \"env\" \"ext\" (func $ext (;0;) (type 0) (param i32) (result i32)))"],
    );
}

#[test]
fn the_well_known_symbols_the_lowering_declares_in_two_modules_are_three_imports() {
    let function = |name: &str| {
        format!(
            "builtin.function public extern(\"C\") @{name}(%p: ptr<i32>, %fr: (i32) -> i32) -> \
             i32 {{\n    %v = mem.load %p : i32;\n    %q = mem.alloca i32;\n    mem.store %v, \
             %q;\n    %w = mem.load %q : i32;\n    %r = builtin.call_indirect %fr(%w) : \
             extern(\"C\") (i32) -> i32;\n    builtin.ret %r : (i32);\n}};"
        )
    };
    let text = format!(
        "builtin.world {{\nbuiltin.module public @a {LAYOUT} {{\n{}\n}};\nbuiltin.module public \
         @b {LAYOUT} {{\n{}\n}};\n}};",
        function("f"),
        function("g"),
    );
    let (context, world) = parse_world(&text);
    run_pipeline(&context, world, LOWERING)
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let text = wat(world);
    assert_eq!(text.matches("(import ").count(), 3, "{text}");
    assert_mentions(
        &text,
        &[
            "(import \"env\" \"__indirect_function_table\" (table $__indirect_function_table \
             (;0;) 0 funcref))",
            "(import \"env\" \"__linear_memory\" (memory $__linear_memory (;0;) 0))",
            "(import \"env\" \"__stack_pointer\" (global $__stack_pointer (;0;) (mut i32)))",
            "(@custom \"target_features\"",
        ],
    );
}

// ---- Globals, tables, memories, exports ---------------------------------------------------------

#[test]
fn a_global_a_table_and_a_memory_are_defined_and_exported() {
    let text = world_of(&[&module(
        "m",
        "wasm.memory public @mem : 1..2 attributes { wasm.export_name = \"mem\" };\nwasm.table \
         public @tab : 2 x !wasm.funcref attributes { wasm.export_name = \"tab\" };\nwasm.global \
         public @g : mut i32 { %v = wasm.const 7 : i32; builtin.yield %v; } attributes { \
         wasm.export_name = \"g\" };",
    )]);
    assert_eq!(
        wat_of_world(&text),
        whole(&[
            "(table $m::tab (;0;) 2 funcref)",
            "(memory $m::mem (;0;) 1 2)",
            "(global $m::g (;0;) (mut i32) i32.const 7)",
            "(export \"mem\" (memory $m::mem))",
            "(export \"tab\" (table $m::tab))",
            "(export \"g\" (global $m::g))",
        ])
        .replace(
            &producers_text(),
            &format!(
                "{}  (@custom \"target_features\" (after export) \"\\01+\\0fmutable-globals\")\n",
                producers_text()
            ),
        )
    );
}

#[test]
fn a_global_a_table_and_a_memory_are_imported_and_exported_again() {
    let text = world_of(&[&module(
        "m",
        "wasm.memory private declaration @mem : 1 attributes { wasm.export_name = \"mem\" \
         };\nwasm.table private declaration @tab : 1..4 x !wasm.externref attributes { link_name \
         = \"refs\", wasm.import_module = \"host\", wasm.export_name = \"tab\" };\nwasm.global \
         private declaration @g : i64 attributes { wasm.export_name = \"g\" };",
    )]);
    let text = wat_of_world(&text);
    assert_mentions(
        &text,
        &[
            "(import \"env\" \"m::mem\" (memory $m::mem (;0;) 1))",
            "(import \"host\" \"refs\" (table $refs (;0;) 1 4 externref))",
            "(import \"env\" \"m::g\" (global $m::g (;0;) i64))",
            "(export \"mem\" (memory $m::mem))",
            "(export \"tab\" (table $refs))",
            "(export \"g\" (global $m::g))",
        ],
    );
    // An `externref` table records reference types; an immutable global, nothing.
    assert_mentions(&text, &["reference-types"]);
    assert!(!text.contains("mutable-globals"), "{text}");
}

#[test]
fn a_function_with_two_export_names_and_an_alias_called_and_exported() {
    // The dialect's verifier accepts `wasm.export_name` only on what Wasm exports, which a
    // `builtin.function_alias` is not; the emitter exports an alias's target all the same, so the
    // IR is not verified here.
    let text = world_of(&[&module(
        "m",
        &[
            adder("target", 1, ""),
            String::from(
                "builtin.function_alias public @alias -> @target attributes { wasm.export_name = \
                 \"alias\" };",
            ),
            caller("f", &["@alias"], " attributes { wasm.export_name = [\"one\", \"two\"] }"),
        ]
        .join("\n"),
    )]);
    let (_context, world) = parse_world_unverified(&text);
    let text = wat(world);
    assert_mentions(
        &text,
        &[
            "  local.get 0\n    call $m::target\n  )",
            "(export \"alias\" (func $m::target))",
            "(export \"one\" (func $m::f))",
            "(export \"two\" (func $m::f))",
        ],
    );
}

// ---- Constant expressions, element segments ------------------------------------------------------

#[test]
fn a_global_s_initializer_is_each_kind_of_constant_expression() {
    let text = world_of(&[&module(
        "m",
        "wasm.global private declaration @limit : i32 attributes { link_name = \"limit\" \
         };\nwasm.global private @i : i64 { %v = wasm.const -5 : i64; builtin.yield %v; \
         };\nwasm.global private @f : f64 { %v = wasm.constf 2.5 : f64; builtin.yield %v; \
         };\nwasm.global private @n : !wasm.externref { %v = wasm.ref_null : !wasm.externref; \
         builtin.yield %v; };\nwasm.global private @r : !wasm.funcref { %v = wasm.ref_func \
         @callee; builtin.yield %v; };\nwasm.global private @again : i32 { %v = wasm.global_get \
         @limit : i32; builtin.yield %v; };\nbuiltin.function public extern(\"wasm\") @callee() { \
         builtin.ret; };",
    )]);
    let text = wat_of_world(&text);
    assert_mentions(
        &text,
        &[
            "(import \"env\" \"limit\" (global $limit (;0;) i32))",
            "(global $m::i (;1;) i64 i64.const -5)",
            "(global $m::f (;2;) f64 f64.const 0x1.4p+1 (;=2.5;))",
            "(global $m::n (;3;) externref ref.null extern)",
            "(global $m::r (;4;) funcref ref.func $m::callee)",
            "(global $m::again (;5;) i32 global.get $limit)",
        ],
    );
    // No declarative segment: a global's initializer declares the function it names.
    assert!(!text.contains("declare"), "{text}");
}

#[test]
fn an_address_and_a_table_slot_in_an_initializer_are_value_fix_ups() {
    let text = world_of(&[&module(
        "m",
        "builtin.global_variable public @data : i32 { %v = arith.constant 1 : i32; builtin.yield \
         %v; };\nwasm.global private @a : i32 { %v = wasm.data_addr @data + 8; builtin.yield %v; \
         };\nwasm.global private @s : i32 { %v = wasm.func_index @callee; builtin.yield %v; \
         };\nbuiltin.function public extern(\"wasm\") @callee() { builtin.ret; };",
    )]);
    let (_context, world) = parse_world(&text);
    let unit = collect(world, &EmitOptions::default(), &BodyConfig::default())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let kinds = unit
        .desc
        .globals()
        .iter()
        .map(|global| match &global.entity {
            Entity::Definition(expr) => {
                expr.fixups.iter().map(|fixup| (fixup.kind, fixup.addend)).collect::<Vec<_>>()
            }
            Entity::Import(_) => Vec::new(),
        })
        .collect::<Vec<_>>();
    let data = unit.desc.data()[0].name.clone();
    assert_eq!(data, "m::data");
    let callee = unit.desc.functions()[0].name.clone();
    assert_eq!(callee, "m::callee");
    assert!(
        matches!(kinds[0][..], [(FixupKind::MemoryAddr(id), 8)] if id.index() == 0),
        "{kinds:?}"
    );
    assert!(
        matches!(kinds[1][..], [(FixupKind::TableSlot(id), 0)] if id.index() == 0),
        "{kinds:?}"
    );
    // Both need the layout, which is not built yet.
    let error = error_of(emit_module(world, &EmitOptions::default()));
    assert_mentions(&error, &["does not build yet"]);
}

#[test]
fn arithmetic_in_an_initializer_needs_extended_const_and_is_an_error_naming_it_and_the_global() {
    let error = error_of_world(&world_of(&[&module(
        "m",
        "wasm.global private @g : i32 { %a = wasm.const 1 : i32; %b = wasm.const 2 : i32; %c = \
         wasm.add %a, %b; builtin.yield %c; };",
    )]));
    assert_mentions(
        &error,
        &[
            "the operation `wasm.add` cannot be emitted as WebAssembly",
            "`wasm.add` in the initializer of `@g` needs the extended-const feature, which is \
             outside the backend's feature set",
            "the global it initializes",
        ],
    );
}

#[test]
fn a_global_read_in_an_initializer_must_be_imported() {
    let error = error_of_world(&world_of(&[&module(
        "m",
        "wasm.global private @base : i32 { %v = wasm.const 1 : i32; builtin.yield %v; \
         };\nwasm.global private @g : i32 { %v = wasm.global_get @base : i32; builtin.yield %v; };",
    )]));
    assert_mentions(
        &error,
        &["`wasm.global_get` reads a global the module defines", "only an imported global"],
    );
}

#[test]
fn a_ref_func_in_a_body_declares_its_function_once_and_an_elem_is_an_active_segment() {
    let text = world_of(&[&module(
        "m",
        "wasm.table private @tab : 2 x !wasm.funcref;\nwasm.elem private @init : @tab at 1 \
         [@callee];\nbuiltin.function public extern(\"wasm\") @callee() { builtin.ret; \
         };\nbuiltin.function public extern(\"wasm\") @f() -> !wasm.funcref {\n    %a = \
         wasm.ref_func @callee;\n    %b = wasm.ref_func @f;\n    %c = wasm.ref_func @callee;\n    \
         %t = wasm.ref_is_null %a;\n    %u = wasm.ref_is_null %b;\n    %s = wasm.select %t, %b, \
         %c;\n    builtin.ret %s : (!wasm.funcref);\n};",
    )]);
    // `@callee` is named twice and declared once; the active segment declares it too, which is
    // valid.
    assert_eq!(
        wat_of_world(&text),
        whole(&[
            "(type (;0;) (func))",
            "(type (;1;) (func (result funcref)))",
            "(table $m::tab (;0;) 2 funcref)",
            "(elem (;0;) (i32.const 1) func $m::callee)",
            "(elem (;1;) declare func $m::callee $m::f)",
            "(func $m::callee (;0;) (type 0))",
            "(func $m::f (;1;) (type 1) (result funcref)",
            "  (local funcref funcref funcref i32)",
            "  ref.func $m::callee",
            "  local.set 0",
            "  ref.func $m::f",
            "  local.set 1",
            "  ref.func $m::callee",
            "  local.set 2",
            "  local.get 0",
            "  ref.is_null",
            "  local.set 3",
            "  local.get 1",
            "  ref.is_null",
            "  drop",
            "  local.get 1",
            "  local.get 2",
            "  local.get 3",
            "  select (result funcref)",
            ")",
        ])
        .replace(
            &producers_text(),
            &format!(
                "{}  (@custom \"target_features\" (after code) \"\\01+\\0freference-types\")\n",
                producers_text()
            ),
        )
    );
}

// ---- Names ----------------------------------------------------------------------------------

/// The local names of the `name` section of `bytes`: (function, local, name).
fn local_names(bytes: &[u8]) -> Vec<(u32, u32, String)> {
    let mut names = Vec::new();
    for payload in payloads(bytes) {
        let Payload::CustomSection(custom) = payload else {
            continue;
        };
        let KnownCustom::Name(reader) = custom.as_known() else {
            continue;
        };
        for name in reader {
            if let Name::Local(map) = name.unwrap() {
                for function in map {
                    let function = function.unwrap();
                    for local in function.names {
                        let local = local.unwrap();
                        names.push((function.index, local.index, String::from(local.name)));
                    }
                }
            }
        }
    }
    names
}

#[test]
fn a_subprogram_s_parameter_names_are_the_names_of_the_parameters() {
    let text = world_of(&[&module(
        "m",
        "builtin.function public extern(\"wasm\") @f(%a: i32, %b: i64) -> i64 {\n    builtin.ret \
         %b : (i64);\n} attributes { di.subprogram = #di.subprogram<{ name = \"f\", file = \
         \"t.rs\", line = 1, ty = (i32, i64) -> i64, params = [\"count\", \"total\"] }> };",
    )]);
    let (_context, world) = parse_world(&text);
    let options = EmitOptions {
        module_name: Some(String::from("unit")),
        ..EmitOptions::default()
    };
    let bytes =
        emit_module(world, &options).unwrap_or_else(|report| panic!("{}", describe(&report)));
    assert_eq!(
        local_names(&bytes),
        [(0, 0, String::from("count")), (0, 1, String::from("total"))]
    );
    let text = print(&bytes).unwrap();
    assert_mentions(
        &text,
        &[
            "(module $unit",
            "(func $m::f (;0;) (type 0) (param $count i32) (param $total i64)",
        ],
    );

    let bytes = emit_module(
        world,
        &EmitOptions {
            names: false,
            ..options
        },
    )
    .unwrap_or_else(|report| panic!("{}", describe(&report)));
    assert!(local_names(&bytes).is_empty());
    let text = print(&bytes).unwrap();
    assert!(!text.contains('$'), "{text}");
}

// ---- Diagnostics ------------------------------------------------------------------------------

#[test]
fn two_definitions_with_one_link_name_are_an_error_naming_both_whatever_their_kinds() {
    let error = error_of_world(&world_of(&[
        &module("a", &adder("f", 1, " attributes { link_name = \"same\" }")),
        &module(
            "b",
            "wasm.global public @g : i32 { %v = wasm.const 1 : i32; builtin.yield %v; } \
             attributes { link_name = \"same\" };",
        ),
    ]));
    assert_mentions(
        &error,
        &[
            "`same` is defined twice in the unit",
            "this defines `same` as a global",
            "this defines it as a function first",
        ],
    );
}

#[test]
fn private_definitions_may_share_a_link_name() {
    let text = wat_of_world(&world_of(&[
        &module(
            "a",
            &adder("f", 1, " attributes { link_name = \"same\" }").replace("public", "private"),
        ),
        &module(
            "b",
            &adder("f", 2, " attributes { link_name = \"same\" }").replace("public", "private"),
        ),
    ]));
    assert_mentions(&text, &["(func $same (;0;)", "(func $\"#func1 same\" (@name \"same\") (;1;)"]);
}

#[test]
fn a_declaration_that_disagrees_with_its_definition_is_an_error_naming_both() {
    // A function.
    let error = error_of_world(&world_of(&[
        &module("a", &adder("f", 1, "")),
        &module(
            "b",
            "builtin.function private extern(\"wasm\") @f(i64) -> i32 attributes { link_name = \
             \"a::f\" };",
        ),
    ]));
    assert_mentions(
        &error,
        &[
            "the declaration of `a::f` does not agree with its definition",
            "this declares `a::f` as `(i64) -> (i32)`",
            "this defines it as `(i32) -> (i32)`",
        ],
    );
    // A global: the mutability.
    let error = error_of_world(&world_of(&[
        &module(
            "a",
            "wasm.global public @g : mut i32 { %v = wasm.const 1 : i32; builtin.yield %v; };",
        ),
        &module(
            "b",
            "wasm.global private declaration @g : i32 attributes { link_name = \"a::g\" };",
        ),
    ]));
    assert_mentions(&error, &["this declares `a::g` as `i32`", "this defines it as `mut i32`"]);
    // A table: a maximum beyond the declaration's.
    let error = error_of_world(&world_of(&[
        &module("a", "wasm.table public @t : 1..8 x !wasm.funcref;"),
        &module(
            "b",
            "wasm.table private declaration @t : 1..4 x !wasm.funcref attributes { link_name = \
             \"a::t\" };",
        ),
    ]));
    assert_mentions(
        &error,
        &["this declares `a::t` as `funcref 1..4`", "this defines it as `funcref 1..8`"],
    );
    // A table within the declaration's limits links.
    let text = wat_of_world(&world_of(&[
        &module(
            "a",
            "wasm.table public @t : 2..4 x !wasm.funcref attributes { wasm.export_name = \"t\" };",
        ),
        &module(
            "b",
            "wasm.table private declaration @t : 1..8 x !wasm.funcref attributes { link_name = \
             \"a::t\" };",
        ),
    ]));
    assert!(!text.contains("(import"), "{text}");
    // Another kind.
    let error = error_of_world(&world_of(&[
        &module("a", &adder("f", 1, "")),
        &module(
            "b",
            "wasm.global private declaration @f : i32 attributes { link_name = \"a::f\" };",
        ),
    ]));
    assert_mentions(&error, &["this declares `a::f` as a global", "this defines it as a function"]);
}

#[test]
fn two_declarations_of_one_import_that_disagree_are_an_error_naming_both() {
    let error = error_of_world(&world_of(&[
        &module("a", &declaration("x", " attributes { link_name = \"ext\" }")),
        &module(
            "b",
            "builtin.function private extern(\"wasm\") @x(i32) -> i64 attributes { link_name = \
             \"ext\" };",
        ),
    ]));
    assert_mentions(
        &error,
        &[
            "two declarations of `ext` disagree in type",
            "this declares `ext` as `(i32) -> (i64)`",
            "this declares it as `(i32) -> (i32)`",
        ],
    );
    let error = error_of_world(&world_of(&[
        &module(
            "a",
            "wasm.global private declaration @g : i32 attributes { link_name = \"g\" };",
        ),
        &module(
            "b",
            "wasm.global private declaration @g : mut i32 attributes { link_name = \"g\" };",
        ),
    ]));
    assert_mentions(&error, &["this declares `g` as `mut i32`", "this declares it as `i32`"]);
    let error = error_of_world(&world_of(&[
        &module(
            "a",
            "wasm.table private declaration @t : 1 x !wasm.funcref attributes { link_name = \"t\" \
             };",
        ),
        &module(
            "b",
            "wasm.table private declaration @t : 1 x !wasm.externref attributes { link_name = \
             \"t\" };",
        ),
    ]));
    assert_mentions(
        &error,
        &["this declares `t` as `externref 1`", "this declares it as `funcref 1`"],
    );
}

#[test]
fn two_exports_of_one_name_are_an_error_naming_both() {
    let error = error_of_world(&world_of(&[&module(
        "m",
        &[
            adder("f", 1, " attributes { wasm.export_name = \"run\" }"),
            adder("g", 2, " attributes { wasm.export_name = [\"go\", \"run\"] }"),
        ]
        .join("\n"),
    )]));
    assert_mentions(
        &error,
        &[
            "two exports are named `run`",
            "this exports @g as `run`",
            "this exports @f as `run` first",
        ],
    );
}

#[test]
fn a_second_memory_is_an_error_naming_both() {
    let error = error_of_world(&world_of(&[
        &module("a", "wasm.memory public @mem : 1;"),
        &module("b", "wasm.memory private declaration @imported : 1;"),
    ]));
    assert_mentions(
        &error,
        &[
            "a module with several memories cannot be emitted",
            "this is a second memory, @imported",
            "the module already has @mem",
        ],
    );
}

#[test]
fn a_tag_is_an_error_naming_it() {
    let error = error_of_world(&world_of(&[&module("m", "wasm.tag private @exn : (i32);")]));
    assert_mentions(
        &error,
        &[
            "the operation `wasm.tag` cannot be emitted as WebAssembly",
            "exceptions are not emitted",
        ],
    );
}

#[test]
fn two_components_are_an_error_naming_the_second() {
    let error = error_of_world(
        "builtin.world {\n    builtin.component public @acme::@one { };\n    builtin.component \
         public @acme::@two { };\n};",
    );
    assert_mentions(
        &error,
        &[
            "a world with several components gives several binaries, which this backend does not \
             produce yet",
            "this is a second `builtin.component`",
            "this is the first",
        ],
    );
}

#[test]
fn the_modules_of_one_component_are_the_unit() {
    let text = wat_of_world(&format!(
        "builtin.world {{\n    builtin.component public @acme::@app {{\n{}\n{}\n}};\n}};",
        module("a", &adder("inc", 1, "")),
        module("b", &caller("f", &["::@acme::@app::@a::@inc"], "")),
    ));
    assert_mentions(&text, &["call $acme::app::a::inc"]);
}

#[test]
fn a_function_not_in_wasm_form_is_an_error_naming_the_lowering() {
    let error = error_of_world(&world_of(&[&module(
        "m",
        "builtin.function public extern(\"C\") @f(%x: i8) -> i8 {\n    builtin.ret %x : (i8);\n};",
    )]));
    assert_mentions(
        &error,
        &[
            "the function @f is not in Wasm form",
            "its parameter 0 has type `i8`, which is not a WebAssembly value type: the lowering \
             (`cfg-to-scf{exits=in-place},wasm-lower-frames,convert-to-wasm,wasm-verify-form`)",
        ],
    );
}

#[test]
fn an_undefined_data_symbol_is_an_error_naming_the_declaration() {
    let error = error_of_world(&world_of(&[&module(
        "m",
        "builtin.global_variable public @g : i32 attributes { link_name = \"elsewhere\" \
         };\nbuiltin.function public extern(\"wasm\") @f() -> i32 {\n    %a = wasm.data_addr \
         @g;\n    builtin.ret %a : (i32);\n};",
    )]));
    assert_mentions(
        &error,
        &[
            "the global variable `elsewhere` is declared and not defined: a core WebAssembly \
             module cannot import data",
            "nothing in the unit defines `elsewhere`",
        ],
    );
}

#[test]
fn a_module_that_needs_the_layout_is_the_error_of_the_seam() {
    let seam = "the module needs the layout of its linear memory or function table, which this \
                backend does not build yet";
    let error = error_of_world(&world_of(&[&module(
        "m",
        "builtin.global_variable public @g : i32 { %v = arith.constant 1 : i32; builtin.yield %v; \
         };",
    )]));
    assert_mentions(&error, &[seam, "this is data in linear memory"]);
    let error = error_of_world(&world_of(&[&module(
        "m",
        &[
            adder("inc", 1, ""),
            String::from(
                "builtin.function public extern(\"wasm\") @f() -> i32 {\n    %i = wasm.func_index \
                 @inc;\n    builtin.ret %i : (i32);\n};",
            ),
        ]
        .join("\n"),
    )]));
    assert_mentions(
        &error,
        &[seam, "this function takes the slot of `m::inc` in the function table"],
    );
    let error = error_of_world(&world_of(&[&module(
        "m",
        "builtin.segment <{ offset = #builtin.i64<16>, data = #builtin.bytes<\"cc3f\"> }>;",
    )]));
    assert_mentions(&error, &[seam, "this is data at a fixed address"]);
}
