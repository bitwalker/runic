//! What the collected unit records for a linker symbol, and what an object cannot express: the
//! binding of each function, global, table and data entry; whether an import names its host;
//! which data is retained; the export name of a function; the start function, and how a complete
//! module exports it as `_initialize`; and each refusal of [Unit::check_object], with the same
//! unit compiled to a complete module.

use alloc::{format, rc::Rc, string::String, vec::Vec};

use runic_ir::{
    Context, Op, OperationRef, Report,
    dialects::builtin::World,
    parse::{self, ParserConfig},
    testing::text::describe,
};

use crate::{
    emit::{
        EmitOptions,
        body::BodyConfig,
        collect::{START_EXPORT, Unit, collect},
        desc::{Binding, DataId, ExportTarget, FuncId, GlobalId, TableId},
        emit_module, emit_object,
    },
    tests::{
        LAYOUT, LOWERING, context, error_of,
        eval::{Outcome, Val},
        exec::Instance,
        mentions, run_pipeline,
    },
    text::print,
};

// ---- Helpers ------------------------------------------------------------------------------------

/// The declaration of the memory the lowering names, as it writes it.
const MEMORY: &str = "wasm.memory private declaration @__linear_memory : 0 attributes { link_name \
                      = \"__linear_memory\" };";

/// A module `@name` of the items `items`, with the data layout of the lowering.
fn module(name: &str, items: &[&str]) -> String {
    format!("builtin.module public @{name} {LAYOUT} {{\n{}\n}};", items.join("\n"))
}

/// The world of `modules`, parsed and verified.
#[track_caller]
fn world(modules: &[String]) -> (Rc<Context>, OperationRef) {
    let text = format!("builtin.world {{\n{}\n}};", modules.join("\n"));
    let context = context();
    let world = parse::parse::<World>(ParserConfig::new(context.clone()), "w", &text)
        .into_result()
        .and_then(|world| {
            world.borrow().as_operation().recursively_verify()?;
            Ok(world.as_operation_ref())
        })
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, world)
}

/// The unit of `root`, collected as both ends collect it.
fn collected(root: OperationRef) -> Result<Unit, Report> {
    collect(root, &EmitOptions::default(), &BodyConfig::default())
}

/// The unit of the world of `modules`, which must collect.
#[track_caller]
fn unit_of(modules: &[String]) -> (Rc<Context>, Unit) {
    let (context, world) = world(modules);
    let unit = collected(world).unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, unit)
}

/// The symbol name of the operation `op`, if it has one.
fn symbol_of(op: OperationRef) -> Option<String> {
    op.borrow().as_symbol().map(|symbol| String::from(symbol.name().as_str()))
}

/// The id of the function the operation of symbol name `name` was made from.
#[track_caller]
fn function(unit: &Unit, name: &str) -> FuncId {
    (0..unit.desc.functions().len() as u32)
        .map(FuncId::new)
        .find(|id| unit.function_op(*id).and_then(symbol_of).as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no function @{name}"))
}

/// The id of the global the operation of symbol name `name` was made from.
#[track_caller]
fn global(unit: &Unit, name: &str) -> GlobalId {
    (0..unit.desc.globals().len() as u32)
        .map(GlobalId::new)
        .find(|id| unit.global_op(*id).and_then(symbol_of).as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no global @{name}"))
}

/// The id of the table the operation of symbol name `name` was made from.
#[track_caller]
fn table(unit: &Unit, name: &str) -> TableId {
    (0..unit.desc.tables().len() as u32)
        .map(TableId::new)
        .find(|id| unit.table_op(*id).and_then(symbol_of).as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no table @{name}"))
}

/// The id of the data entry the operation of symbol name `name` was made from.
#[track_caller]
fn data(unit: &Unit, name: &str) -> DataId {
    (0..unit.desc.data().len() as u32)
        .map(DataId::new)
        .find(|id| unit.data_op(*id).and_then(symbol_of).as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no data entry @{name}"))
}

/// A Wasm-form function `@name` of `visibility` that returns nothing, with `attributes` after it.
fn empty(visibility: &str, name: &str, attributes: &str) -> String {
    format!(
        "builtin.function {visibility} extern(\"wasm\") @{name}() {{\n    \
         builtin.ret;\n}}{attributes};"
    )
}

/// A Wasm-form function `@name` of `(i32) -> i32` that returns its parameter, with `attributes`.
fn identity(visibility: &str, name: &str, attributes: &str) -> String {
    format!(
        "builtin.function {visibility} extern(\"wasm\") @{name}(%x: i32) -> i32 {{\n    \
         builtin.ret %x : (i32);\n}}{attributes};"
    )
}

/// An immutable `wasm.global` `@name` of `visibility` that holds the `i32` 0, with `attributes`.
fn zero_global(visibility: &str, name: &str, attributes: &str) -> String {
    format!(
        "wasm.global {visibility} @{name} : i32 {{ %v = wasm.const 0 : i32; builtin.yield %v; \
         }}{attributes};"
    )
}

/// A `builtin.global_variable` `@name` of `visibility` that holds the `i32` 1, with `attributes`.
fn word(visibility: &str, name: &str, attributes: &str) -> String {
    format!(
        "builtin.global_variable {visibility} @{name} : i32 {{ %v = arith.constant 1 : i32; \
         builtin.yield %v; }}{attributes};"
    )
}

/// The error of the object check of the world of `modules`, through [emit_object], with its
/// labels; the same world must compile to a complete module.
#[track_caller]
fn refused(modules: &[String]) -> String {
    let (_context, root) = world(modules);
    emit_module(root, &EmitOptions::default())
        .unwrap_or_else(|report| panic!("a complete module is refused: {}", describe(&report)));
    error_of(emit_object(root, &EmitOptions::default()))
}

/// Asserts that `haystack` holds every one of `needles`.
#[track_caller]
fn assert_mentions(haystack: &str, needles: &[&str]) {
    for needle in needles {
        assert!(mentions(haystack, needle), "no `{needle}` in:\n{haystack}");
    }
}

/// The world of `modules`, lowered by the crate's pipeline and emitted as a complete module.
#[track_caller]
fn lowered_module(modules: &[String]) -> Result<Vec<u8>, Report> {
    let (context, root) = world(modules);
    run_pipeline(&context, root, LOWERING).unwrap_or_else(|report| panic!("{}", describe(&report)));
    emit_module(root, &EmitOptions::default())
}

// ---- Bindings -----------------------------------------------------------------------------------

#[test]
fn the_binding_of_each_kind_of_entity_follows_the_visibility_of_its_definition() {
    let mut items = Vec::new();
    for visibility in ["public", "internal", "private"] {
        items.push(empty(visibility, &format!("f_{visibility}"), ""));
        items.push(zero_global(visibility, &format!("g_{visibility}"), ""));
        items.push(format!("wasm.table {visibility} @t_{visibility} : 1 x !wasm.funcref;"));
        items.push(word(visibility, &format!("d_{visibility}"), ""));
    }
    let items = items.iter().map(String::as_str).collect::<Vec<_>>();
    let (_context, unit) = unit_of(&[module("m", &items)]);
    for (visibility, binding) in [
        ("public", Binding::Global),
        ("internal", Binding::Hidden),
        ("private", Binding::Local),
    ] {
        let name = |kind: &str| format!("{kind}_{visibility}");
        assert_eq!(unit.function_binding(function(&unit, &name("f"))), Some(binding));
        assert_eq!(unit.global_binding(global(&unit, &name("g"))), Some(binding));
        assert_eq!(unit.table_binding(table(&unit, &name("t"))), Some(binding));
        assert_eq!(unit.data_binding(data(&unit, &name("d"))), Some(binding));
    }
}

#[test]
fn a_declaration_names_its_host_only_when_it_is_written_with_an_import_module() {
    let (_context, unit) = unit_of(&[module(
        "m",
        &[
            "builtin.function public extern(\"wasm\") @hosted() attributes { wasm.import_module = \
             \"host\" };",
            "builtin.function public extern(\"wasm\") @linked();",
            // `env` written out names the host as much as another module does
            "builtin.function internal extern(\"wasm\") @env_named() attributes { \
             wasm.import_module = \"env\" };",
            "wasm.global private declaration @g_hosted : i32 attributes { wasm.import_module = \
             \"host\" };",
            "wasm.global private declaration @g_linked : i32;",
            "wasm.table private declaration @t_linked : 1 x !wasm.funcref;",
            "builtin.global_variable public @d_linked : i32;",
        ],
    )]);
    let host = Some(Binding::Undefined { names_host: true });
    let linker = Some(Binding::Undefined { names_host: false });
    assert_eq!(unit.function_binding(function(&unit, "hosted")), host);
    assert_eq!(unit.function_binding(function(&unit, "linked")), linker);
    assert_eq!(unit.function_binding(function(&unit, "env_named")), host);
    assert_eq!(unit.global_binding(global(&unit, "g_hosted")), host);
    assert_eq!(unit.global_binding(global(&unit, "g_linked")), linker);
    assert_eq!(unit.table_binding(table(&unit, "t_linked")), linker);
    // An undefined data symbol is the linker's: Wasm does not import data
    let d = data(&unit, "d_linked");
    assert!(unit.undefined_data().contains(&d));
    assert_eq!(unit.data_binding(d), linker);
    // The imports are as they were: from the module named, `env` by default
    let import = |id: FuncId| unit.desc.function(id).entity.import().cloned().unwrap();
    assert_eq!(import(function(&unit, "hosted")).module, "host");
    assert_eq!(import(function(&unit, "linked")).module, "env");
}

#[test]
fn a_declaration_that_links_to_a_definition_has_the_definitions_binding() {
    let (_context, unit) = unit_of(&[
        module("a", &[&identity("internal", "f", " attributes { link_name = \"shared\" }")]),
        module(
            "b",
            &[
                "builtin.function public extern(\"wasm\") @f(i32) -> i32 attributes { link_name = \
                 \"shared\" };",
                &identity("public", "caller", ""),
            ],
        ),
    ]);
    assert_eq!(unit.desc.functions().len(), 2, "the declaration is the definition");
    assert_eq!(unit.function_binding(function(&unit, "f")), Some(Binding::Hidden));
}

#[test]
fn declarations_that_disagree_on_naming_their_host_collect_and_an_object_refuses_them() {
    let modules = [
        module(
            "a",
            &["builtin.function public extern(\"wasm\") @f() attributes { link_name = \"g\" };"],
        ),
        module(
            "b",
            &["builtin.function public extern(\"wasm\") @f() attributes { link_name = \"g\", \
               wasm.import_module = \"env\" };"],
        ),
    ];
    // One import, from `env`, whose first declaration does not name the host
    let (_context, unit) = unit_of(&modules);
    assert_eq!(unit.desc.functions().len(), 1);
    assert_eq!(
        unit.function_binding(FuncId::new(0)),
        Some(Binding::Undefined { names_host: false })
    );
    let error = refused(&modules);
    assert_mentions(
        &error,
        &[
            "a relocatable object cannot express the import `g`: its declarations disagree on \
             whether it names its host",
            "this declaration names the host module `env` with `wasm.import_module`, so the \
             linker imports `g` from it",
            "the first declaration has no `wasm.import_module`, so the linker resolves `g` to a \
             definition of another object",
        ],
    );
}

// ---- Data, exports ------------------------------------------------------------------------------

#[test]
fn a_data_entry_is_retained_when_it_has_a_section() {
    let (_context, unit) = unit_of(&[module(
        "m",
        &[
            &word("public", "in_section", " attributes { section = \"dispatch\" }"),
            &word("public", "plain", ""),
            &word("private", "private_in_section", " attributes { section = \"dispatch\" }"),
            "builtin.global_variable public @start : i8 attributes { link_name = \
             \"__start_dispatch\" };",
        ],
    )]);
    assert!(unit.is_retained(data(&unit, "in_section")));
    assert!(unit.is_retained(data(&unit, "private_in_section")));
    assert!(!unit.is_retained(data(&unit, "plain")));
    // An undefined data symbol has no segment to retain
    assert!(!unit.is_retained(data(&unit, "start")));
    assert_eq!(unit.desc.data()[data(&unit, "in_section").index() as usize].size, 4);
}

#[test]
fn the_export_name_of_a_function_is_found_by_its_id() {
    let (_context, unit) = unit_of(&[module(
        "m",
        &[
            &identity("public", "exported", " attributes { wasm.export_name = \"outside\" }"),
            &identity("public", "twice", " attributes { wasm.export_name = [\"one\", \"two\"] }"),
            &identity("public", "hidden", ""),
            &zero_global("private", "g", " attributes { wasm.export_name = \"g\" }"),
        ],
    )]);
    assert_eq!(unit.function_export(function(&unit, "exported")), Some("outside"));
    assert_eq!(unit.function_export(function(&unit, "hidden")), None);
    // A complete module exports a function under both names; the lookup gives each in order
    let twice = ExportTarget::Function(function(&unit, "twice"));
    assert_eq!(unit.desc.export_names(twice).collect::<Vec<_>>(), ["one", "two"]);
    let g = ExportTarget::Global(global(&unit, "g"));
    assert_eq!(unit.desc.export_names(g).collect::<Vec<_>>(), ["g"]);
}

// ---- The start function -------------------------------------------------------------------------

#[test]
fn the_start_function_is_the_one_marked_with_wasm_start() {
    let (_context, unit) = unit_of(&[module(
        "m",
        &[
            &empty("private", "init", " attributes { wasm.start = #builtin.unit }"),
            &empty("public", "other", ""),
        ],
    )]);
    assert_eq!(unit.start(), Some(function(&unit, "init")));
    let (_context, unit) = unit_of(&[module("m", &[&empty("public", "other", "")])]);
    assert_eq!(unit.start(), None);
}

#[test]
fn two_start_functions_in_one_unit_are_an_error_naming_both() {
    let (_context, root) = world(&[
        module("a", &[&empty("public", "first", " attributes { wasm.start = #builtin.unit }")]),
        module("b", &[&empty("public", "second", " attributes { wasm.start = #builtin.unit }")]),
    ]);
    let error = error_of(collected(root));
    assert_mentions(
        &error,
        &[
            "the unit has two start functions: a compilation has at most one",
            "this marks @second as the start function with `wasm.start`",
            "@first is the start function already",
        ],
    );
    // The second is the primary label
    let primary = error.find("this marks @second").unwrap();
    let secondary = error.find("@first is the start function already").unwrap();
    assert!(primary < secondary, "{error}");
}

/// A start function that adds 35 to `@state` (7 in data), and `@read`, exported, which reads it
/// and adds its `i8` argument: generic IR, lowered by the crate's pipeline.
fn starting(attributes: &str) -> String {
    module(
        "start",
        &[
            "builtin.global_variable public @state : i32 { %seven = arith.constant 7 : i32; \
             builtin.yield %seven; };",
            &format!(
                "builtin.function public extern(\"C\") @init() {{
    %p = builtin.global_symbol @state : ptr<i32>;
    %old = mem.load %p;
    %step = arith.constant 35 : i32;
    %new = arith.add %old, %step;
    mem.store %new, %p;
    builtin.ret;
}} attributes {{ wasm.start = #builtin.unit{attributes} }};"
            ),
            "builtin.function public extern(\"C\") @read(%k: i8) -> i32 {
    %p = builtin.global_symbol @state : ptr<i32>;
    %v = mem.load %p;
    %w = arith.zext %k <{ ty = i32 }>;
    %r = arith.add %v, %w;
    builtin.ret %r : (i32);
} attributes { wasm.export_name = \"read\" };",
        ],
    )
}

#[test]
fn a_complete_module_exports_its_start_function_as_initialize_and_has_no_start_section() {
    let bytes =
        lowered_module(&[starting("")]).unwrap_or_else(|report| panic!("{}", describe(&report)));
    let text = print(&bytes).unwrap_or_else(|report| panic!("{report}"));
    assert!(text.contains("(export \"_initialize\" (func $start::init))"), "{text}");
    assert!(!text.contains("(start"), "{text}");
    assert!(
        !wasmparser::Parser::new(0)
            .parse_all(&bytes)
            .any(|payload| matches!(payload, Ok(wasmparser::Payload::StartSection { .. }))),
        "the module has a start section"
    );
    // Instantiating runs nothing: `@state` holds its initial value until `_initialize` runs
    let mut instance = Instance::new(&bytes);
    assert_eq!(instance.call("read", &[Val::i32(1)]), Outcome::Values(Vec::from([Val::i32(8)])));
    let mut instance = Instance::new(&bytes);
    assert_eq!(instance.call(START_EXPORT, &[]), Outcome::Values(Vec::new()));
    assert_eq!(
        instance.call("read", &[Val::i32(1)]),
        Outcome::Values(Vec::from([Val::i32(43)]))
    );
}

#[test]
fn a_unit_that_exports_initialize_and_has_a_start_function_has_two_exports_of_that_name() {
    let error =
        error_of(lowered_module(&[starting("")
            .replace("wasm.export_name = \"read\"", "wasm.export_name = \"_initialize\"")]));
    assert_mentions(
        &error,
        &[
            "two exports are named `_initialize`",
            "this is the start function, @init, which a complete module exports as `_initialize`",
            "this exports @read as `_initialize` first",
        ],
    );
    // The start function itself exported as `_initialize` is the same error: two exports
    let error = error_of(lowered_module(&[starting(", wasm.export_name = \"_initialize\"")]));
    assert_mentions(&error, &["two exports are named `_initialize`"]);
}

// ---- What an object cannot express ----------------------------------------------------------------

#[test]
fn a_unit_an_object_can_express_passes_the_check_and_the_writer_is_not_built_yet() {
    let modules = [module(
        "m",
        &[
            MEMORY,
            &identity("public", "f", " attributes { wasm.export_name = \"f\" }"),
            &empty("public", "init", " attributes { wasm.start = #builtin.unit }"),
            &word("public", "d", " attributes { section = \"dispatch\" }"),
            "wasm.table private declaration @__indirect_function_table : 0 x !wasm.funcref \
             attributes { link_name = \"__indirect_function_table\" };",
            "wasm.global private declaration @__stack_pointer : mut i32 attributes { link_name = \
             \"__stack_pointer\" };",
            &zero_global(
                "public",
                "g",
                " attributes { link_name = \"g\", wasm.export_name = \"g\" }",
            ),
        ],
    )];
    let (_context, unit) = unit_of(&modules);
    unit.check_object().unwrap_or_else(|report| panic!("{}", describe(&report)));
    // The start function is not exported as `_initialize` on the object's path
    assert_eq!(unit.export_op(START_EXPORT), None);
    let error = refused(&modules);
    assert_mentions(&error, &["the relocatable object writer is not built yet"]);
}

#[test]
fn an_object_cannot_express_an_element_segment() {
    let error = refused(&[module(
        "m",
        &[
            "wasm.table private @tab : 1 x !wasm.funcref;",
            &empty("public", "f", ""),
            "wasm.elem private @e : @tab at 0 [@f];",
        ],
    )]);
    assert_mentions(
        &error,
        &[
            "a relocatable object cannot express `wasm.elem` @e",
            "this fills a table at slots it fixes, and the linker ignores the element segments of \
             an object: it lays out the function table itself",
        ],
    );
}

#[test]
fn an_object_cannot_express_a_segment_at_a_fixed_address() {
    let error = refused(&[module(
        "m",
        &[
            MEMORY,
            "builtin.segment <{ offset = #builtin.i64<2048>, data = #builtin.bytes<\"2a2a2a2a\"> \
             }>;",
        ],
    )]);
    assert_mentions(
        &error,
        &[
            "a relocatable object cannot express a `builtin.segment`",
            "this places bytes at the address 2048, and the linker places every data segment of \
             an object itself: there is no data at a fixed address",
        ],
    );
}

#[test]
fn an_object_cannot_express_passive_data_and_names_the_operation_that_makes_it_passive() {
    let copy = "builtin.function public extern(\"wasm\") @copy(%d: i32) {
    %s = wasm.const 0 : i32;
    %n = wasm.const 4 : i32;
    wasm.memory_init @__linear_memory, @bytes, %d, %s, %n;
    builtin.ret;
};";
    let error = refused(&[module("m", &[MEMORY, &word("private", "bytes", ""), copy])]);
    assert_mentions(
        &error,
        &[
            "a relocatable object cannot express passive data: `wasm.memory_init` names \
             `m::bytes` as a passive segment",
            "this names `m::bytes` as a passive data segment, and the linker makes every data \
             segment of an object active, with no relocation for a segment index",
        ],
    );
    let drop = "builtin.function public extern(\"wasm\") @drop() {
    wasm.data_drop @bytes;
    builtin.ret;
};";
    let error = refused(&[module("m", &[MEMORY, &word("private", "bytes", ""), drop])]);
    assert_mentions(&error, &["`wasm.data_drop` names `m::bytes` as a passive segment"]);
}

#[test]
fn an_object_cannot_express_a_global_whose_initializer_names_another_entity() {
    for (initializer, holds) in [
        ("%v = wasm.data_addr @d; builtin.yield %v;", "the address of `m::d`"),
        (
            "%v = wasm.func_index @f; builtin.yield %v;",
            "the table slot of the function `m::f`",
        ),
        (
            "%v = wasm.global_get @imported : i32; builtin.yield %v;",
            "the global `m::imported`",
        ),
    ] {
        let global = format!("wasm.global private @g : i32 {{ {initializer} }};");
        let error = refused(&[module(
            "m",
            &[
                MEMORY,
                &word("public", "d", ""),
                &empty("public", "f", ""),
                "wasm.global private declaration @imported : i32;",
                &global,
            ],
        )]);
        assert_mentions(
            &error,
            &[
                "a relocatable object cannot express `wasm.global` @g, whose initializer names \
                 another entity",
                &format!(
                    "the initializer of this global holds {holds}, and an object has no \
                     relocation in the global section"
                ),
            ],
        );
    }
    let error = refused(&[module(
        "m",
        &[
            &empty("public", "f", ""),
            "wasm.global private @r : !wasm.funcref { %v = wasm.ref_func @f; builtin.yield %v; };",
        ],
    )]);
    assert_mentions(&error, &["holds a reference to the function `m::f`"]);
}

#[test]
fn an_object_cannot_define_its_memory() {
    let error = refused(&[module("m", &["wasm.memory public @mem : 1;"])]);
    assert_mentions(
        &error,
        &[
            "a relocatable object cannot express a memory it defines, `wasm.memory` @mem",
            "this defines a memory, and the linker defines the memory of the module it links: an \
             object imports it as `env.__linear_memory`",
        ],
    );
}

#[test]
fn an_object_cannot_define_the_indirect_function_table_by_symbol_or_by_link_name() {
    for table in [
        "wasm.table private @__indirect_function_table : 1 x !wasm.funcref;",
        "wasm.table private @functions : 1 x !wasm.funcref attributes { link_name = \
         \"__indirect_function_table\" };",
    ] {
        let error = refused(&[module("m", &[table])]);
        assert_mentions(
            &error,
            &[
                "a relocatable object cannot define the function table `__indirect_function_table`",
                "this defines `__indirect_function_table`, which the linker defines itself: it \
                 refuses an object that defines it",
            ],
        );
    }
    // A table of another name is the object's own
    let (_context, unit) =
        unit_of(&[module("m", &["wasm.table private @own : 1 x !wasm.funcref;"])]);
    unit.check_object().unwrap_or_else(|report| panic!("{}", describe(&report)));
}

#[test]
fn an_object_cannot_export_a_function_under_two_names() {
    let error = refused(&[module(
        "m",
        &[&identity(
            "public",
            "f",
            " attributes { wasm.export_name = [\"one\", \"two\"] }",
        )],
    )]);
    assert_mentions(
        &error,
        &[
            "a relocatable object cannot express the function `m::f` exported under two names, \
             `one` and `two`",
            "this exports `m::f` as `two`, and it is exported as `one`: an object exports a \
             function under one name",
        ],
    );
    // A declaration that links to the definition exports the definition: two names
    let error =
        refused(&[
            module(
                "a",
                &[&identity(
                    "public",
                    "f",
                    " attributes { link_name = \"f\", wasm.export_name = \"one\" }",
                )],
            ),
            module(
                "b",
                &["builtin.function public extern(\"wasm\") @f(i32) -> i32 attributes { \
                   link_name = \"f\", wasm.export_name = \"two\" };"],
            ),
        ]);
    assert_mentions(&error, &["exported under two names, `one` and `two`"]);
}

#[test]
fn an_object_exports_a_global_a_table_or_a_memory_only_under_its_link_name() {
    for (item, what, link_name) in [
        (
            zero_global(
                "private",
                "g",
                " attributes { link_name = \"g\", wasm.export_name = \"x\" }",
            ),
            "global",
            "g",
        ),
        (
            String::from(
                "wasm.table private declaration @t : 1 x !wasm.funcref attributes { link_name = \
                 \"t\", wasm.export_name = \"x\" };",
            ),
            "table",
            "t",
        ),
        (
            String::from(
                "wasm.memory private declaration @__linear_memory : 0 attributes { link_name = \
                 \"__linear_memory\", wasm.export_name = \"x\" };",
            ),
            "memory",
            "__linear_memory",
        ),
    ] {
        let error = refused(&[module("m", &[&item])]);
        assert_mentions(
            &error,
            &[
                &format!(
                    "a relocatable object cannot express the export `x` of the {what} \
                     `{link_name}` under another name"
                ),
                &format!(
                    "this exports the {what} `{link_name}` as `x`, and an object marks a {what} \
                     exported only under its own link name"
                ),
            ],
        );
    }
    // Under its own link name, a global is exported
    let (_context, unit) = unit_of(&[module(
        "m",
        &[&zero_global(
            "private",
            "g",
            " attributes { link_name = \"g\", wasm.export_name = \"g\" }",
        )],
    )]);
    unit.check_object().unwrap_or_else(|report| panic!("{}", describe(&report)));
}
