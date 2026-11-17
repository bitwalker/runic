//! The well-known symbols: found in a module, or declared there.

use runic_dialect_wasm::{Global, Memory, Table};
use runic_ir::{Op, OperationRef, UnsafeIntrusiveEntityRef};

use super::*;
use crate::symbols::{indirect_function_table, linear_memory, stack_pointer};

/// The module `@m` holding `body`.
fn module_with(body: &str) -> (Rc<Context>, ModuleRef) {
    parse_module(&format!("builtin.module public @m {{\n{body}\n}};"))
}

/// The operation of `entity`.
fn op_of<T: Op>(entity: UnsafeIntrusiveEntityRef<T>) -> OperationRef {
    entity.borrow().as_operation().as_operation_ref()
}

/// `module` verifies, and its printed text parses back to the same text.
#[track_caller]
fn verifies_and_round_trips(module: ModuleRef) -> String {
    module
        .borrow()
        .as_operation()
        .recursively_verify()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let text = text_of(module);
    let (_context, reparsed) = parse_module(&text);
    assert_eq!(text_of(reparsed), text);
    text
}

#[test]
fn each_symbol_is_declared_private_at_the_start_of_the_module_with_its_own_link_name() {
    let (_context, module) = module_with(
        "    builtin.function public extern(\"C\") @f() {\n        builtin.ret;\n    };",
    );
    stack_pointer(module).unwrap();
    indirect_function_table(module).unwrap();
    linear_memory(module).unwrap();
    let text = verifies_and_round_trips(module);
    assert_eq!(
        lines(&text),
        [
            "builtin.module public @m {",
            "wasm.memory private declaration @__linear_memory : 0 attributes { link_name = \
             #builtin.string<\"__linear_memory\"> };",
            "wasm.table private declaration @__indirect_function_table : 0 x !wasm.funcref \
             attributes { link_name = #builtin.string<\"__indirect_function_table\"> };",
            "wasm.global private declaration @__stack_pointer : mut i32 attributes { link_name = \
             #builtin.string<\"__stack_pointer\"> };",
            "builtin.function public extern(\"C\") @f() {",
            "builtin.ret;",
            "};",
            "};",
        ]
    );
}

#[test]
fn each_symbol_is_an_import_from_env() {
    let (_context, module) = module_with("");
    let memory = op_of(linear_memory(module).unwrap());
    let table = op_of(indirect_function_table(module).unwrap());
    let global = op_of(stack_pointer(module).unwrap());
    for op in [memory, table, global] {
        assert_eq!(runic_dialect_wasm::import_module(&op.borrow()).as_deref(), Some("env"));
        assert_eq!(
            runic_ir::effective_link_name(&op.borrow()).as_deref(),
            Some(op.borrow().as_symbol().unwrap().name().as_str())
        );
    }
}

#[test]
fn a_definition_of_the_name_and_kind_is_used_as_it_is() {
    let body = "    wasm.memory public @__linear_memory : 1..16;\n    wasm.table public \
                @__indirect_function_table : 4 x !wasm.funcref;\n    wasm.global public \
                @__stack_pointer : mut i32 {\n        %v = wasm.const 65536 : i32;\n        \
                builtin.yield %v;\n    };";
    let (_context, module) = module_with(body);
    let before = text_of(module);
    let lookup = |name: &'static str| {
        runic_ir::SymbolTable::symbol_manager(&*module.borrow())
            .lookup_op(runic_ir::interner::Symbol::intern(name))
            .unwrap()
    };
    assert!(op_of(linear_memory(module).unwrap()) == lookup("__linear_memory"));
    assert!(op_of(indirect_function_table(module).unwrap()) == lookup("__indirect_function_table"));
    assert!(op_of(stack_pointer(module).unwrap()) == lookup("__stack_pointer"));
    assert_eq!(text_of(module), before);
}

#[test]
fn a_declaration_of_the_name_and_kind_is_used_as_it_is() {
    let body = "    wasm.memory public declaration @__linear_memory : 2;\n    wasm.global private \
                declaration @__stack_pointer : mut i32;";
    let (_context, module) = module_with(body);
    let before = text_of(module);
    let memory = op_of(linear_memory(module).unwrap());
    let global = op_of(stack_pointer(module).unwrap());
    assert!(memory.borrow().is::<Memory>());
    assert!(global.borrow().is::<Global>());
    assert_eq!(text_of(module), before);
}

#[test]
fn called_twice_each_gives_the_same_symbol() {
    let (_context, module) = module_with("");
    let first = [
        op_of(linear_memory(module).unwrap()),
        op_of(indirect_function_table(module).unwrap()),
        op_of(stack_pointer(module).unwrap()),
    ];
    let second = [
        op_of(linear_memory(module).unwrap()),
        op_of(indirect_function_table(module).unwrap()),
        op_of(stack_pointer(module).unwrap()),
    ];
    assert!(first == second);
    assert!(first[1].borrow().is::<Table>());
    assert_eq!(text_of(module).matches("declaration").count(), 3);
    verifies_and_round_trips(module);
}

#[test]
fn a_symbol_of_the_name_and_another_kind_is_an_error_naming_both() {
    let (_context, module) =
        module_with("    builtin.function public extern(\"C\") @__linear_memory();");
    let error = error_of(linear_memory(module));
    assert!(
        mentions(
            &error,
            "the module names `@__linear_memory` the wasm.memory that lowered code uses"
        ),
        "{error}"
    );
    assert!(
        mentions(&error, "`@__linear_memory` is a `builtin.function`, not a `wasm.memory`"),
        "{error}"
    );
}

#[test]
fn a_table_of_another_element_type_is_an_error() {
    let (_context, module) =
        module_with("    wasm.table public @__indirect_function_table : 1 x !wasm.externref;");
    let error = error_of(indirect_function_table(module));
    assert!(
        mentions(
            &error,
            "`@__indirect_function_table` is a `wasm.table` of `!wasm.externref`, not of \
             `!wasm.funcref`"
        ),
        "{error}"
    );
}

#[test]
fn a_stack_pointer_of_another_type_or_constant_is_an_error() {
    for (global, found) in [
        ("wasm.global public declaration @__stack_pointer : mut i64;", "mut i64"),
        ("wasm.global public declaration @__stack_pointer : i32;", "i32"),
    ] {
        let (_context, module) = module_with(&format!("    {global}"));
        let error = error_of(stack_pointer(module));
        assert!(
            mentions(
                &error,
                &format!(
                    "`@__stack_pointer` is a `wasm.global` of type `{found}`, not of type `mut \
                     i32`"
                )
            ),
            "{error}"
        );
    }
}
