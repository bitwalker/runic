//! Nested modules: a function's code names the well-known symbols found or declared in the
//! `builtin.module` that directly holds the function, for both passes that name them, and nothing
//! is declared in an outer module on behalf of an inner one.

use alloc::vec;

use runic_ir::{OperationRef, RawWalk, dialects::builtin::World};

use super::{
    eval::{Machine, Outcome, Val},
    exec::{PAGE, run_in},
    *,
};

/// A function `@name` that loads from its parameter, allocates a slot and stores the value in it,
/// and calls `@twice` (a function of the same module) through a function value on it; and
/// `@twice`, prefixed with `prefix` so that each module has its own.
fn functions_using_every_symbol(name: &str, prefix: &str) -> String {
    format!(
        "builtin.function public extern(\"C\") @{prefix}twice(%x: i32) -> i32 {{\n    %r = \
         arith.add %x, %x;\n    builtin.ret %r : (i32);\n}};\nbuiltin.function public \
         extern(\"C\") @{name}(%p: ptr<i32>) -> i32 {{\n    %v = mem.load %p : i32;\n    %q = \
         mem.alloca i32;\n    mem.store %v, %q;\n    %w = mem.load %q : i32;\n    %f = \
         builtin.function_ref @{prefix}twice : (i32) -> i32;\n    %r = builtin.call_indirect \
         %f(%w) : extern(\"C\") (i32) -> i32;\n    builtin.ret %r : (i32);\n}};"
    )
}

/// The lines of the text of the operations `module` directly holds that are not modules.
fn own_lines(module: OperationRef) -> Vec<String> {
    let module = module.borrow();
    let mut own = Vec::new();
    for region in module.regions().iter() {
        for block in region.body().iter() {
            for op in block.body().iter().filter(|op| !op.is::<Module>()) {
                own.extend(lines(&op.to_string()));
            }
        }
    }
    own
}

/// How many declarations of the well-known symbol `name` the lines `lines` hold.
fn declarations_of(lines: &[String], name: &str) -> usize {
    lines
        .iter()
        .filter(|line| line.contains(&format!("declaration @{name} ")))
        .count()
}

/// The module `@name` nested in `module`.
fn nested(module: ModuleRef, name: &str) -> ModuleRef {
    function_named(module, name).try_downcast_op::<Module>().expect("a module")
}

/// Runs `@function` of `module` on the address 16, which holds 21, with the stack pointer at 256.
fn run_on_21(module: ModuleRef, function: &str) -> Outcome {
    let mut machine = Machine::new(PAGE);
    machine.globals.insert(String::from("__stack_pointer"), Val::i32(256));
    machine.write_words(16, &[21]);
    let outcome = run_in(&mut machine, function_named(module, function), &[Val::i32(16)]);
    assert_eq!(machine.globals["__stack_pointer"], Val::i32(256));
    outcome
}

#[test]
fn each_module_of_a_nested_pair_declares_the_symbols_its_own_functions_name() {
    let text = format!(
        "builtin.module public @outer {LAYOUT} {{\n    builtin.module public @inner {{\n{}\n    \
         }};\n{}\n}};",
        functions_using_every_symbol("f", "inner_"),
        functions_using_every_symbol("g", "outer_")
    );
    let (_context, outer) = lowered_by(&text, LOWERING);
    let inner = nested(outer, "inner");
    for module in [outer, inner] {
        let own = own_lines(module.as_operation_ref());
        for name in ["__linear_memory", "__indirect_function_table", "__stack_pointer"] {
            assert_eq!(declarations_of(&own, name), 1, "@{name}\n{}", text_of(outer));
        }
    }
    // Every use names the symbol of its own module, by its bare name.
    assert!(!text_of(outer).contains("::@outer::@__"), "{}", text_of(outer));
    assert_eq!(run_on_21(inner, "f"), Outcome::Values(vec![Val::i32(42)]));
    assert_eq!(run_on_21(outer, "g"), Outcome::Values(vec![Val::i32(42)]));
}

#[test]
fn an_inner_module_that_defines_its_own_memory_uses_it_and_gets_no_declaration() {
    let text = format!(
        "builtin.module public @outer {LAYOUT} {{\n    builtin.module public @inner {{\n    \
         wasm.memory public @__linear_memory : 1..16;\n{}\n    }};\n{}\n}};",
        functions_using_every_symbol("f", "inner_"),
        functions_using_every_symbol("g", "outer_")
    );
    let (_context, outer) = lowered_by(&text, LOWERING);
    let inner = nested(outer, "inner");
    let inner_lines = own_lines(inner.as_operation_ref());
    assert_eq!(declarations_of(&inner_lines, "__linear_memory"), 0, "{}", text_of(outer));
    assert!(
        inner_lines
            .iter()
            .any(|line| line.contains("wasm.memory public @__linear_memory"))
    );
    let outer_lines = own_lines(outer.as_operation_ref());
    assert_eq!(declarations_of(&outer_lines, "__linear_memory"), 1, "{}", text_of(outer));
    // The inner function's loads name the inner memory.
    let load = inner_lines.iter().find(|line| line.contains("wasm.load")).expect("a load");
    assert!(load.contains("wasm.load @__linear_memory,"), "{load}");
    assert_eq!(run_on_21(inner, "f"), Outcome::Values(vec![Val::i32(42)]));
}

#[test]
fn an_outer_module_whose_functions_name_no_symbol_gets_no_declaration() {
    let text = format!(
        "builtin.module public @outer {LAYOUT} {{\n    builtin.module public @inner {{\n{}\n    \
         }};\n}};",
        functions_using_every_symbol("f", "inner_")
    );
    let (_context, outer) = lowered_by(&text, LOWERING);
    let own = own_lines(outer.as_operation_ref());
    assert!(!own.iter().any(|line| line.contains("declaration")), "{}", text_of(outer));
}

#[test]
fn each_module_of_a_world_and_of_a_component_declares_its_own_symbols() {
    let module = |name: &str| {
        format!(
            "builtin.module public @{name} {LAYOUT} {{\n{}\n}};",
            functions_using_every_symbol("f", &format!("{name}_"))
        )
    };
    let text = format!(
        "builtin.world {{\n{}\n{}\nbuiltin.component public @acme::@comp {{\n{}\n{}\n}};\n}};",
        module("a"),
        module("b"),
        module("c"),
        module("d")
    );
    let context = context();
    let world = parse::parse::<World>(ParserConfig::new(context.clone()), "w", &text)
        .into_result()
        .unwrap();
    run_pipeline(&context, world.as_operation_ref(), LOWERING)
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let mut modules = Vec::new();
    let _ = world
        .as_operation_ref()
        .raw_prewalk::<runic_ir::Forward, _, _>(|op: OperationRef| {
            if let Ok(module) = op.try_downcast_op::<Module>() {
                modules.push(module);
            }
            runic_ir::WalkResult::<()>::Continue(())
        });
    assert_eq!(modules.len(), 4);
    for module in modules {
        let own = own_lines(module.as_operation_ref());
        for name in ["__linear_memory", "__indirect_function_table", "__stack_pointer"] {
            assert_eq!(declarations_of(&own, name), 1, "@{name}\n{}", text_of(module));
        }
        assert_eq!(run_on_21(module, "f"), Outcome::Values(vec![Val::i32(42)]));
    }
}
