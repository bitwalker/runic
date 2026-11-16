//! The operations on globals and tables, the reference operations, and the two address
//! operations.

use runic_ir::{
    Builder, Ident, SourceSpan, ValueRef, Visibility,
    dialects::builtin::BuiltinOpBuilder,
    testing::{Test, text::round_trip},
};

use super::*;

/// One of each operation of this file, over the entities of [STATE].
const EVERY_OPERATION: &str = "
    %g = wasm.global_get @g : i32;
    %w = wasm.global_get @wide : i64;
    %k = wasm.global_get @k : i32;
    wasm.global_set @g, %a;
    wasm.global_set @wide, %l;
    %t = wasm.table_get @tab, %a : !wasm.funcref;
    %e = wasm.table_get @refs, %a : !wasm.externref;
    wasm.table_set @tab, %a, %r;
    wasm.table_set @refs, %a, %x;
    %n = wasm.table_size @tab;
    %old = wasm.table_grow @refs, %x, %n;
    wasm.table_fill @tab, %a, %r, %n;
    %null = wasm.ref_null : !wasm.funcref;
    %enull = wasm.ref_null : !wasm.externref;
    %fn = wasm.ref_func @callee;
    %z = wasm.ref_is_null %r;
    %ez = wasm.ref_is_null %x;
    %d0 = wasm.data_addr @data;
    %d1 = wasm.data_addr @data + 16;
    %d2 = wasm.data_addr @data - 4;
    %i = wasm.func_index @callee;";

#[test]
fn every_variable_and_reference_operation_round_trips_in_the_custom_and_the_generic_form() {
    let (_context, module) = accepted(EVERY_OPERATION);
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

#[test]
fn the_operations_print_the_text_forms_the_documentation_gives() {
    let (_context, module) = accepted(EVERY_OPERATION);
    let text = module.borrow().as_operation().to_string();
    for part in [
        "= wasm.global_get @g : i32;",
        "wasm.global_set @g, %",
        "= wasm.table_get @tab, %5 : !wasm.funcref;",
        "wasm.table_set @tab, %5, %8;",
        "= wasm.table_size @tab;",
        "= wasm.table_grow @refs, %9, %",
        "wasm.table_fill @tab, %5, %8, %",
        "= wasm.ref_null : !wasm.funcref;",
        "= wasm.ref_func @callee;",
        "= wasm.ref_is_null %8;",
        "= wasm.data_addr @data;",
        "= wasm.data_addr @data + 16;",
        "= wasm.data_addr @data - 4;",
        "= wasm.func_index @callee;",
    ] {
        assert!(text.contains(part), "`{part}` is not in\n{text}");
    }
}

#[test]
fn a_global_is_read_with_its_type_and_written_when_it_is_mutable() {
    rejected(
        "%v = wasm.global_get @g : i64;",
        "`wasm.global_get` takes the name of a global and gives a value of its type, but its \
         result has type `i64`, but the global `g` has type `i32`",
    );
    rejected(
        "%v = wasm.global_get @mem : i32;",
        "`wasm.global_get` names `mem`, which is a `wasm.memory`, not a `wasm.global`",
    );
    rejected(
        "%v = wasm.global_get @callee : i32;",
        "`wasm.global_get` names `callee`, which is a `builtin.function`, not a `wasm.global`",
    );
    rejected(
        "wasm.global_set @k, %a;",
        "`wasm.global_set` takes the name of a mutable global and a value of its type, but the \
         global `k` is immutable, and `global.set` writes only a `mut` global",
    );
    rejected(
        "wasm.global_set @g, %l;",
        "`wasm.global_set` takes the name of a mutable global and a value of its type, but its \
         value has type `i64`, but the global `g` has type `i32`",
    );
    rejected(
        "wasm.global_set @tab, %a;",
        "`wasm.global_set` names `tab`, which is a `wasm.table`, not a `wasm.global`",
    );
    rejected("wasm.global_set @nothing, %a;", "nothing");
}

#[test]
fn a_table_is_read_and_written_with_an_i32_index_and_its_element_type() {
    rejected(
        "%v = wasm.table_get @tab, %l : !wasm.funcref;",
        "`wasm.table_get` takes a table and an `i32` index and gives an element, a reference of \
         the table's type, but its index has type `i64`",
    );
    rejected(
        "%v = wasm.table_get @tab, %a : !wasm.externref;",
        "but its result has type `externref`, but the table `tab` holds `funcref`",
    );
    rejected(
        "%v = wasm.table_get @mem, %a : !wasm.funcref;",
        "`wasm.table_get` names `mem`, which is a `wasm.memory`, not a `wasm.table`",
    );
    rejected(
        "wasm.table_set @tab, %a, %x;",
        "`wasm.table_set` takes a table, an `i32` index and a reference of the table's element \
         type, but its value has type `externref`, but the table `tab` holds `funcref`",
    );
    rejected(
        "wasm.table_set @tab, %l, %r;",
        "`wasm.table_set` takes a table, an `i32` index and a reference of the table's element \
         type, but its index has type `i64`",
    );
    rejected(
        "wasm.table_set @refs, %a, %a;",
        "its value has type `i32`, but the table `refs` holds `externref`",
    );
    rejected(
        "%n = wasm.table_size @g;",
        "`wasm.table_size` names `g`, which is a `wasm.global`, not a `wasm.table`",
    );
    rejected(
        "%o = wasm.table_grow @tab, %x, %a;",
        "`wasm.table_grow` takes a table, a reference of the table's element type and an `i32` \
         number of elements, and gives an `i32`, but its initial value has type `externref`, but \
         the table `tab` holds `funcref`",
    );
    rejected("%o = wasm.table_grow @tab, %r, %l;", "but its delta has type `i64`");
    rejected(
        "wasm.table_fill @tab, %a, %x, %a;",
        "`wasm.table_fill` takes a table, an `i32` index, a reference of the table's element type \
         and an `i32` number of elements, but its value has type `externref`, but the table `tab` \
         holds `funcref`",
    );
    rejected("wasm.table_fill @tab, %l, %r, %a;", "but its index has type `i64`");
    rejected("wasm.table_fill @tab, %a, %r, %l;", "but its length has type `i64`");
}

#[test]
fn a_null_reference_has_a_reference_type() {
    accepted("%n = wasm.ref_null : !wasm.funcref;\n %m = wasm.ref_null : !wasm.externref;");
    for ty in ["i32", "f64", "i8", "ptr<i32>"] {
        let shown = ty;
        rejected(
            &format!("%n = wasm.ref_null : {ty};"),
            &format!(
                "`wasm.ref_null` gives a null reference of a reference type, `funcref` or \
                 `externref`, but its result has type `{shown}`, not `funcref` or `externref`"
            ),
        );
    }
}

#[test]
fn a_function_reference_names_a_function_and_gives_a_funcref() {
    accepted("%v = wasm.ref_func @callee;\n %o = wasm.ref_func @opaque;");
    let error = |body: &str| match parse_state(&state_module(
        STATE_PARAMS,
        "",
        &format!("{body}\nbuiltin.ret;"),
    )) {
        Ok(_) => panic!("accepted: {body}"),
        Err(error) => error,
    };
    for name in ["mem", "tab", "g", "data"] {
        let error = error(&format!("%v = wasm.ref_func @{name};"));
        assert!(
            error.contains(&format!(
                "`wasm.ref_func` takes the name of a function and gives a `funcref`, but `{name}` \
                 does not name a function"
            )),
            "{error}"
        );
    }
    let error = error("%v = wasm.ref_func @nothing;");
    assert!(error.contains("nothing"), "{error}");
}

#[test]
fn a_reference_is_tested_for_null_and_the_result_is_an_i32() {
    accepted("%z = wasm.ref_is_null %r;\n %y = wasm.ref_is_null %x;");
    for operand in ["%a", "%l", "%f", "%p"] {
        let ty = match operand {
            "%a" => "i32",
            "%l" => "i64",
            "%f" => "f32",
            _ => "ptr<i32>",
        };
        rejected(
            &format!("%z = wasm.ref_is_null {operand};"),
            &format!(
                "`wasm.ref_is_null` takes a reference, `funcref` or `externref`, and gives an \
                 `i32`, but its operand has type `{ty}`"
            ),
        );
    }
}

#[test]
fn a_data_address_names_data_and_has_a_signed_32_bit_offset() {
    let (_context, module) = accepted(
        "%v0 = wasm.data_addr @data;
         %v1 = wasm.data_addr @data + 0;
         %v2 = wasm.data_addr @data + 2147483647;
         %v3 = wasm.data_addr @data - 2147483648;",
    );
    let text = module.borrow().as_operation().to_string();
    assert_eq!(count(&text, "wasm.data_addr @data;"), 2, "{text}");
    assert!(text.contains("wasm.data_addr @data + 2147483647;"), "{text}");
    assert!(text.contains("wasm.data_addr @data - 2147483648;"), "{text}");
    rejected(
        "%v = wasm.data_addr @data + 2147483648;",
        "2147483648 is not a 32-bit signed offset",
    );
    rejected(
        "%v = wasm.data_addr @data - 2147483649;",
        "-2147483649 is not a 32-bit signed offset",
    );
    rejected(
        "%v = wasm.data_addr @mem;",
        "`wasm.data_addr` names `mem`, which is a `wasm.memory`, not a `builtin.global_variable`",
    );
    rejected(
        "%v = wasm.data_addr @callee + 4;",
        "`wasm.data_addr` names `callee`, which is a `builtin.function`, not a \
         `builtin.global_variable`",
    );
    rejected("%v = wasm.data_addr @nothing;", "nothing");
}

#[test]
fn a_function_index_names_a_function_and_has_the_type_i32() {
    accepted("%i = wasm.func_index @callee;\n %j = wasm.func_index @opaque;");
    for name in ["mem", "tab", "g", "data"] {
        let result = parse_state(&state_module(
            STATE_PARAMS,
            "",
            &format!("%i = wasm.func_index @{name};\nbuiltin.ret;"),
        ));
        let error = result.err().expect("not a function");
        assert!(
            error.contains(&format!(
                "`wasm.func_index` takes the name of a function and gives its index in the \
                 indirect-call table, an `i32`, but `{name}` does not name a function"
            )),
            "{error}"
        );
    }
}

#[test]
fn an_alias_of_a_function_is_a_function_to_name() {
    let result = parse_state(
        "builtin.module public @m {
            builtin.function public extern(\"C\") @real();
            builtin.function_alias public @alias -> @real;
            builtin.function public extern(\"C\") @f() {
                %r = wasm.ref_func @alias;
                %i = wasm.func_index @alias;
                builtin.ret;
            };
        };",
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn the_constant_instructions_include_the_references_the_addresses_and_the_reads_of_constants() {
    let (_context, module) = accepted(EVERY_OPERATION);
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let mut seen = Vec::new();
    let _ = function.borrow().prewalk(|op: &runic_ir::Operation| {
        let name = op.name().to_string();
        if name.starts_with("wasm.") {
            let global = op
                .downcast_ref::<GlobalGet>()
                .map(|get| get.global_op().unwrap().borrow().get_mutability().is_mut());
            seen.push((name, global, is_constant_instruction(op)));
        }
        runic_ir::WalkResult::<()>::Continue(())
    });
    for (name, mutable, constant) in seen {
        let expected = match name.as_str() {
            "wasm.global_get" => mutable == Some(false),
            "wasm.ref_null" | "wasm.ref_func" | "wasm.data_addr" | "wasm.func_index" => true,
            _ => false,
        };
        assert_eq!(constant, expected, "{name} (mutable: {mutable:?})");
    }
    // The marker alone over-approximates for a mutable global, as it does for a float `add`
    for name in ["global_get", "ref_null", "ref_func", "data_addr", "func_index"] {
        let (_context, module) = accepted(match name {
            "global_get" => "%v = wasm.global_get @k : i32;",
            "ref_null" => "%v = wasm.ref_null : !wasm.funcref;",
            "ref_func" => "%v = wasm.ref_func @callee;",
            "data_addr" => "%v = wasm.data_addr @data;",
            _ => "%v = wasm.func_index @callee;",
        });
        let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
        let mut marked = false;
        let _ = function.borrow().prewalk(|op: &runic_ir::Operation| {
            if op.name().to_string() == format!("wasm.{name}") {
                marked = op.implements::<dyn ConstantInstruction>();
            }
            runic_ir::WalkResult::<()>::Continue(())
        });
        assert!(marked, "{name}");
    }
}

#[test]
fn the_operations_that_read_or_write_state_are_not_constant_instructions() {
    for body in [
        "%v = wasm.load @mem, %a : i32;",
        "%n = wasm.memory_size @mem;",
        "%v = wasm.table_get @tab, %a : !wasm.funcref;",
        "%z = wasm.ref_is_null %r;",
        "%n = wasm.table_size @tab;",
    ] {
        let (_context, module) = accepted(body);
        let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
        let _ = function.borrow().prewalk(|op: &runic_ir::Operation| {
            if op.name().to_string().starts_with("wasm.") {
                assert!(!is_constant_instruction(op), "{body}");
            }
            runic_ir::WalkResult::<()>::Continue(())
        });
    }
}

/// A module with `globals` after the state of [STATE], parsed and verified.
fn module_with_globals(globals: &str) -> Result<(), String> {
    parse_state(&format!("builtin.module public @m {{\n{STATE}\n{globals}\n}};")).map(|_| ())
}

#[test]
fn a_global_initializer_may_hold_references_addresses_and_the_value_of_an_immutable_global() {
    for (ty, expression) in [
        ("!wasm.funcref", "%v = wasm.ref_null : !wasm.funcref;"),
        ("!wasm.externref", "%v = wasm.ref_null : !wasm.externref;"),
        ("!wasm.funcref", "%v = wasm.ref_func @callee;"),
        ("i32", "%v = wasm.global_get @k : i32;"),
        ("i32", "%v = wasm.data_addr @data + 16;"),
        ("i32", "%v = wasm.func_index @callee;"),
        (
            "i32",
            "%a = wasm.data_addr @data;
             %b = wasm.const 8 : i32;
             %v = wasm.add %a, %b;",
        ),
        (
            "i32",
            "%a = wasm.global_get @k : i32;
             %v = wasm.mul %a, %a;",
        ),
    ] {
        let result =
            module_with_globals(&format!("wasm.global private @x : {ty} {{ {expression} }};"));
        assert!(result.is_ok(), "{expression}: {:?}", result.err());
    }
}

#[test]
fn a_global_initializer_rejects_the_value_of_a_mutable_global_and_the_state_operations() {
    let message = |op: &str| {
        format!(
            "the initializer of `wasm.global` `@x` holds `{op}`, but a Wasm constant expression \
             holds only constant instructions"
        )
    };
    for (ty, expression, op) in [
        ("i32", "%v = wasm.global_get @g : i32;", "wasm.global_get"),
        ("i64", "%v = wasm.global_get @wide : i64;", "wasm.global_get"),
        ("i32", "%a = wasm.const 0 : i32;\n %v = wasm.load @mem, %a : i32;", "wasm.load"),
        ("i32", "%v = wasm.memory_size @mem;", "wasm.memory_size"),
        (
            "i32",
            "%n = wasm.ref_null : !wasm.funcref;\n %v = wasm.ref_is_null %n;",
            "wasm.ref_is_null",
        ),
        ("i32", "%v = wasm.table_size @tab;", "wasm.table_size"),
    ] {
        let error =
            module_with_globals(&format!("wasm.global private @x : {ty} {{ {expression} }};"))
                .err()
                .unwrap_or_else(|| panic!("accepted: {expression}"));
        assert!(error.contains(&message(op)), "{error}");
    }
}

#[test]
fn each_variable_and_reference_operation_declares_its_effects() {
    let cases: [(&str, &str, &[&str]); 12] = [
        ("wasm.global_get", "%v = wasm.global_get @g : i32;", &["Read wasm.globals"]),
        ("wasm.global_set", "wasm.global_set @g, %a;", &["Write wasm.globals"]),
        (
            "wasm.table_get",
            "%v = wasm.table_get @tab, %a : !wasm.funcref;",
            &["Read wasm.tables"],
        ),
        ("wasm.table_set", "wasm.table_set @tab, %a, %r;", &["Write wasm.tables"]),
        ("wasm.table_size", "%v = wasm.table_size @tab;", &["Read wasm.tables"]),
        ("wasm.table_grow", "%v = wasm.table_grow @tab, %r, %a;", &["Write wasm.tables"]),
        ("wasm.table_fill", "wasm.table_fill @tab, %a, %r, %a;", &["Write wasm.tables"]),
        ("wasm.ref_null", "%v = wasm.ref_null : !wasm.funcref;", &[]),
        ("wasm.ref_func", "%v = wasm.ref_func @callee;", &[]),
        ("wasm.ref_is_null", "%v = wasm.ref_is_null %r;", &[]),
        ("wasm.data_addr", "%v = wasm.data_addr @data + 4;", &[]),
        ("wasm.func_index", "%v = wasm.func_index @callee;", &[]),
    ];
    for (name, body, expected) in cases {
        let found = effects_of(name, body);
        assert_eq!(found, expected, "{name}");
    }
}

/// Two reads, `first` and `second` (`%l1` and `%l2` of `i32`), with `middle` between them: how many
/// of the operation `op` are left after `cse`.
fn reads(first: &str, middle: &str, second: &str, op: &str) -> usize {
    after_cse(first, middle, second, op)
}

const GLOBAL_A: &str = "%l1 = wasm.global_get @g : i32;";
const GLOBAL_B: &str = "%l2 = wasm.global_get @g : i32;";

#[test]
fn two_reads_of_one_global_merge_and_do_not_merge_across_a_write_of_a_global_or_a_call() {
    assert_eq!(reads(GLOBAL_A, "", GLOBAL_B, "wasm.global_get"), 1);
    // Two globals are two reads
    assert_eq!(reads(GLOBAL_A, "", "%l2 = wasm.global_get @k : i32;", "wasm.global_get"), 2);
    for middle in [
        "wasm.global_set @g, %a;",
        "wasm.global_set @wide, %l;",
        "builtin.call @opaque() : extern(\"C\") () -> ();",
        "wasm.call_indirect @tab, %a() : extern(\"C\") () -> ();",
    ] {
        assert_eq!(reads(GLOBAL_A, middle, GLOBAL_B, "wasm.global_get"), 2, "{middle}");
    }
    for middle in [
        "wasm.store @mem, %a, %b;",
        "%o = wasm.memory_grow @mem, %b;",
        "wasm.table_set @tab, %a, %r;",
        "%n = wasm.table_grow @tab, %r, %a;",
        "mem.store %b, %p;",
    ] {
        assert_eq!(reads(GLOBAL_A, middle, GLOBAL_B, "wasm.global_get"), 1, "{middle}");
    }
}

#[test]
fn two_reads_of_a_table_merge_only_when_nothing_wrote_a_table() {
    let get_a = "%l1 = wasm.table_size @tab;";
    let get_b = "%l2 = wasm.table_size @tab;";
    assert_eq!(reads(get_a, "", get_b, "wasm.table_size"), 1);
    for middle in [
        "%o = wasm.table_grow @tab, %r, %a;",
        "%o = wasm.table_grow @refs, %x, %a;",
        "wasm.table_set @tab, %a, %r;",
        "wasm.table_fill @refs, %a, %x, %a;",
        "builtin.call @opaque() : extern(\"C\") () -> ();",
    ] {
        assert_eq!(reads(get_a, middle, get_b, "wasm.table_size"), 2, "{middle}");
    }
    for middle in ["wasm.global_set @g, %a;", "wasm.store @mem, %a, %b;", "mem.store %b, %p;"] {
        assert_eq!(reads(get_a, middle, get_b, "wasm.table_size"), 1, "{middle}");
    }
    let join = "%u = wasm.select %a, %l1, %l2;";
    let first = "%l1 = wasm.table_get @tab, %a : !wasm.funcref;";
    let second = "%l2 = wasm.table_get @tab, %a : !wasm.funcref;";
    let merged = after_cse_of(first, "", second, join, "!wasm.funcref", "wasm.table_get");
    assert_eq!(merged, 1);
    let blocked = after_cse_of(
        first,
        "wasm.table_set @tab, %a, %r;",
        second,
        join,
        "!wasm.funcref",
        "wasm.table_get",
    );
    assert_eq!(blocked, 2);
    let across = after_cse_of(
        first,
        "wasm.global_set @g, %a;",
        second,
        join,
        "!wasm.funcref",
        "wasm.table_get",
    );
    assert_eq!(across, 1);
}

#[test]
fn an_unused_read_of_a_global_or_table_is_removed_and_a_write_is_never() {
    let kept = |body: &str, op: &str| count(&state_dce(body), op);
    for (body, op) in [
        ("%v = wasm.global_get @g : i32;", "wasm.global_get"),
        ("%v = wasm.table_get @tab, %a : !wasm.funcref;", "wasm.table_get"),
        ("%v = wasm.table_size @tab;", "wasm.table_size"),
        ("%v = wasm.ref_null : !wasm.funcref;", "wasm.ref_null"),
        ("%v = wasm.ref_func @callee;", "wasm.ref_func"),
        ("%v = wasm.ref_is_null %r;", "wasm.ref_is_null"),
        ("%v = wasm.data_addr @data + 4;", "wasm.data_addr"),
        ("%v = wasm.func_index @callee;", "wasm.func_index"),
    ] {
        assert_eq!(kept(body, op), 0, "{body}");
    }
    for (body, op) in [
        ("wasm.global_set @g, %a;", "wasm.global_set"),
        ("wasm.table_set @tab, %a, %r;", "wasm.table_set"),
        ("%v = wasm.table_grow @tab, %r, %a;", "wasm.table_grow"),
        ("wasm.table_fill @tab, %a, %r, %a;", "wasm.table_fill"),
    ] {
        assert_eq!(kept(body, op), 1, "{body}");
    }
}

#[test]
fn equal_references_and_addresses_merge() {
    let merged = |body: &str, op: &str| {
        let text =
            state_cse("i32", &format!("{body}\n%u = wasm.add %l1, %l2;\nbuiltin.ret %u : (i32);"));
        count(&text, op)
    };
    assert_eq!(
        merged(
            "%l1 = wasm.data_addr @data + 8;\n%l2 = wasm.data_addr @data + 8;",
            "wasm.data_addr"
        ),
        1
    );
    assert_eq!(
        merged(
            "%l1 = wasm.data_addr @data + 8;\n%l2 = wasm.data_addr @data + 9;",
            "wasm.data_addr"
        ),
        2
    );
    assert_eq!(
        merged(
            "%l1 = wasm.func_index @callee;\n%l2 = wasm.func_index @callee;",
            "wasm.func_index"
        ),
        1
    );
    assert_eq!(
        merged(
            "%l1 = wasm.func_index @callee;\n%l2 = wasm.func_index @opaque;",
            "wasm.func_index"
        ),
        2
    );
}

#[test]
fn the_builders_build_every_variable_and_reference_operation() {
    let mut test = Test::named("f").in_module("m");
    let context = test.context_rc();
    context.get_or_register_dialect::<WasmDialect>();
    let (i32, i64) = (context.i32(), context.i64());
    let funcref = FuncRefType::get(&context);
    let module = test.module();
    let body = module.borrow().body().entry_block_ref().unwrap();
    let ident = |name: &str| Ident::with_empty_span(name.to_string().into());
    let (table, global, constant, data, callee) = {
        let builder = test.builder_mut();
        builder.set_insertion_point_to_end(body);
        let table = builder
            .create_table(ident("tab"), Visibility::Private, Limits::at_least(2), funcref)
            .unwrap();
        let global = builder
            .create_global_declaration(ident("g"), Visibility::Private, i64, Mutability::Mut)
            .unwrap();
        let constant = builder
            .create_global_declaration(ident("k"), Visibility::Private, i32, Mutability::Const)
            .unwrap();
        let data = builder.create_global_variable(ident("data"), Visibility::Private, i32).unwrap();
        let callee = test.define_function("callee", &[], &[]);
        (table, global, constant, data, callee)
    };
    test.with_function("f", &[i32, i64], &[]);
    let (a, l) = {
        let entry = test.entry_block();
        let entry = entry.borrow();
        (entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef)
    };
    let builder = test.builder_mut();
    let span = SourceSpan::UNKNOWN;

    let value = builder.global_get(global, span).unwrap();
    assert_eq!(value.borrow().ty(), i64);
    builder.global_set(global, l, span).unwrap();
    let read = builder.global_get(constant, span).unwrap();
    assert_eq!(read.borrow().ty(), i32);
    let element = builder.table_get(table, a, span).unwrap();
    assert_eq!(element.borrow().ty(), funcref);
    builder.table_set(table, a, element, span).unwrap();
    let size = builder.table_size(table, span).unwrap();
    assert_eq!(size.borrow().ty(), i32);
    let old = builder.table_grow(table, element, size, span).unwrap();
    assert_eq!(old.borrow().ty(), i32);
    builder.table_fill(table, a, element, size, span).unwrap();
    let null = builder.ref_null(funcref, span).unwrap();
    assert_eq!(null.borrow().ty(), funcref);
    let function = builder.ref_func(callee, span).unwrap();
    assert_eq!(function.borrow().ty(), funcref);
    let is_null = builder.ref_is_null(null, span).unwrap();
    assert_eq!(is_null.borrow().ty(), i32);
    let address = builder.data_addr(data, -8, span).unwrap();
    assert_eq!(address.borrow().ty(), i32);
    let index = builder.func_index(callee, span).unwrap();
    assert_eq!(index.borrow().ty(), i32);
    builder.ret(None, span).unwrap();

    module.borrow().as_operation().recursively_verify().unwrap_or_else(|report| {
        panic!("{}\n{}", describe(&report), module.borrow().as_operation())
    });
    let text = module.borrow().as_operation().to_string();
    for part in [
        "wasm.global_get ::@m::@g : i64;",
        "wasm.global_set ::@m::@g, %",
        "wasm.table_get ::@m::@tab, %",
        "wasm.table_set ::@m::@tab",
        "wasm.table_size ::@m::@tab",
        "wasm.table_grow ::@m::@tab",
        "wasm.table_fill ::@m::@tab",
        "wasm.ref_null : !wasm.funcref;",
        "wasm.ref_func ::@m::@callee;",
        "wasm.ref_is_null",
        "wasm.data_addr ::@m::@data - 8;",
        "wasm.func_index ::@m::@callee;",
    ] {
        assert!(text.contains(part), "`{part}` is not in\n{text}");
    }
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
