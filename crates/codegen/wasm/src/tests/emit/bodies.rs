//! Function bodies as text: one module per family of operations, emitted, validated and printed
//! with `wasmprinter`, the function `@f` compared in full.

use alloc::{format, string::String};

use crate::tests::{
    exec::{Setup, text_of_module},
    module_of, parse_module,
};

/// The text of the function `$f` of the emitted module of `functions`, from its `(func` to its
/// closing parenthesis.
#[track_caller]
fn body_of(functions: &str) -> String {
    let (_context, module) = parse_module(&module_of(functions));
    // No memory unless the module has one; the data symbols of the tests at fixed addresses.
    let setup = Setup {
        memory: 0,
        ..Setup::default()
    }
    .data("d", 1024)
    .passive("seg", &[1, 2, 3]);
    let text = text_of_module(module.as_operation_ref(), &setup);
    let start = text.find("  (func $f ").unwrap_or_else(|| panic!("no `$f` in:\n{text}"));
    let end = text[start..].find("\n  )\n").unwrap_or_else(|| panic!("no end in:\n{text}"));
    String::from(&text[start..start + end + 4])
}

/// `lines` joined into one text.
fn lines(lines: &[&str]) -> String {
    lines.join("\n")
}

/// A function `@f` with the signature `signature` (after the name), the locals `locals` and the
/// body `body`.
fn f(signature: &str, body: &str) -> String {
    format!("builtin.function public extern(\"wasm\") @f{signature} {{\n{body}\n}};")
}

#[test]
fn integer_arithmetic_keeps_single_uses_on_the_stack_and_fetches_an_earlier_operand_first() {
    let body = body_of(&f(
        "(%x: i32, %y: i32, %z: i32) -> i32",
        "%a = wasm.mul %x, %y;\n%b = wasm.sub %z, %a;\n%k = wasm.const -7 : i32;\n%c = wasm.shr_s \
         %b, %k;\nbuiltin.ret %c : (i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32 i32) (result i32)",
            "    local.get 2",
            "    local.get 0",
            "    local.get 1",
            "    i32.mul",
            "    i32.sub",
            "    i32.const -7",
            "    i32.shr_s",
            "  )",
        ])
    );
}

#[test]
fn a_teed_value_a_dropped_result_and_a_value_in_a_local() {
    let body = body_of(&f(
        "(%x: i64, %y: i64) -> i64 locals(i64)",
        "%a = wasm.add %x, %y;\n%b = wasm.mul %a, %x;\nbuiltin.store_local 0, %b;\n%q = \
         wasm.div_u %a, %y;\n%r = wasm.rem_s %a, %a;\n%l = builtin.load_local 0 : \
         i64;\nbuiltin.ret %l : (i64);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i64 i64) (result i64)",
            "    (local i64 i64)",
            "    local.get 0",
            "    local.get 1",
            "    i64.add",
            "    local.tee 3",
            "    local.get 0",
            "    i64.mul",
            "    local.set 2",
            "    local.get 3",
            "    local.get 1",
            "    i64.div_u",
            "    drop",
            "    local.get 3",
            "    local.get 3",
            "    i64.rem_s",
            "    drop",
            "    local.get 2",
            "  )",
        ])
    );
}

#[test]
fn floats_and_conversions() {
    let body = body_of(&f(
        "(%a: f64, %b: f32) -> (i64, f32, i32)",
        "%h = wasm.constf 0.5 : f64;\n%m = wasm.mul %a, %h;\n%t = wasm.trunc_sat_s %m <{ ty = i64 \
         }>;\n%p = wasm.promote %b <{ ty = f64 }>;\n%s = wasm.sqrt %p;\n%d = wasm.demote %s <{ ty \
         = f32 }>;\n%c = wasm.copysign %d, %b;\n%l = wasm.lt %a, %h;\n%e = wasm.extend8_s \
         %l;\nbuiltin.ret %t, %c, %e : (i64, f32, i32);",
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param f64 f32) (result i64 f32 i32)",
            "    local.get 0",
            "    f64.const 0x1p-1 (;=0.5;)",
            "    f64.mul",
            "    i64.trunc_sat_f64_s",
            "    local.get 1",
            "    f64.promote_f32",
            "    f64.sqrt",
            "    f32.demote_f64",
            "    local.get 1",
            "    f32.copysign",
            "    local.get 0",
            "    f64.const 0x1p-1 (;=0.5;)",
            "    f64.lt",
            "    i32.extend8_s",
            "  )",
        ])
    );
}

#[test]
fn loads_and_stores_with_offsets_alignments_and_narrow_storage() {
    let body = body_of(&format!(
        "wasm.memory public @mem : 1;\n{}",
        f(
            "(%p: i32, %v: i64) -> i64",
            "%a = wasm.load @mem, %p as i16 signed <{ offset = 6, align = 1 }> : i64;\n%b = \
             wasm.add %a, %v;\nwasm.store @mem, %p, %b as i32 <{ offset = 16 }>;\n%f = wasm.load \
             @mem, %p : f64;\nwasm.store @mem, %p, %f <{ align = 4 }>;\n%u = wasm.load @mem, %p \
             as i8 unsigned : i32;\n%w = wasm.extend_u %u <{ ty = i64 }>;\nbuiltin.ret %w : (i64);"
        )
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i64) (result i64)",
            "    local.get 0",
            "    local.get 0",
            "    i64.load16_s offset=6 align=1",
            "    local.get 1",
            "    i64.add",
            "    i64.store32 offset=16",
            "    local.get 0",
            "    local.get 0",
            "    f64.load",
            "    f64.store align=4",
            "    local.get 0",
            "    i32.load8_u",
            "    i64.extend_i32_u",
            "  )",
        ])
    );
}

#[test]
fn globals_data_addresses_and_function_indices() {
    let body = body_of(&format!(
        "wasm.memory public @mem : 1;\nwasm.global private declaration @sp : mut \
         i32;\nbuiltin.global_variable public @d : i32;\n{}",
        f(
            "() -> i32",
            "%sp = wasm.global_get @sp : i32;\n%k = wasm.const 16 : i32;\n%n = wasm.sub %sp, \
             %k;\nwasm.global_set @sp, %n;\n%a = wasm.data_addr @d + 8;\n%v = wasm.load @mem, %a \
             : i32;\n%i = wasm.func_index @f;\n%s = wasm.add %v, %i;\nbuiltin.ret %s : (i32);"
        )
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (result i32)",
            "    global.get $sp",
            "    i32.const 16",
            "    i32.sub",
            "    global.set $sp",
            "    i32.const 1032",
            "    i32.load",
            "    i32.const 0",
            "    i32.add",
            "  )",
        ])
    );
}

#[test]
fn calls_with_two_results_indirect_calls_and_a_tail_call() {
    let body = body_of(&format!(
        "wasm.table private declaration @t : 0 x !wasm.funcref;\nbuiltin.function public \
         extern(\"wasm\") @g(i32) -> (i32, i32);\n{}",
        f(
            "(%x: i32, %i: i32) -> i32",
            "%p, %q = builtin.call @g(%x) : extern(\"wasm\") (i32) -> (i32, i32);\n%r = \
             wasm.call_indirect @t, %i(%p, %q) : extern(\"wasm\") (i32, i32) -> i32;\n%s = \
             wasm.select %r, %p, %q;\nbuiltin.tail_call @f(%s, %i) : extern(\"wasm\") (i32, i32) \
             -> i32;"
        )
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;1;) (type 1) (param i32 i32) (result i32)",
            "    (local i32 i32)",
            "    local.get 0",
            "    call $g",
            "    local.set 3",
            "    local.set 2",
            "    local.get 2",
            "    local.get 3",
            "    local.get 2",
            "    local.get 3",
            "    local.get 1",
            "    call_indirect (type 1)",
            "    select",
            "    local.get 1",
            "    return_call $f",
            "  )",
        ])
    );
}

#[test]
fn bulk_memory_and_passive_data() {
    let body = body_of(&format!(
        "wasm.memory public @mem : 1;\nbuiltin.global_variable public @seg : i32;\n{}",
        f(
            "(%d: i32, %s: i32, %n: i32)",
            "wasm.memory_copy @mem, @mem, %d, %s, %n;\n%z = wasm.const 0 : i32;\nwasm.memory_fill \
             @mem, %d, %z, %n;\nwasm.memory_init @mem, @seg, %d, %z, %n;\nwasm.data_drop \
             @seg;\n%g = wasm.memory_grow @mem, %n;\n%m = wasm.memory_size @mem;\nbuiltin.ret;"
        )
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32 i32)",
            "    local.get 0",
            "    local.get 1",
            "    local.get 2",
            "    memory.copy",
            "    local.get 0",
            "    i32.const 0",
            "    local.get 2",
            "    memory.fill",
            "    local.get 0",
            "    i32.const 0",
            "    local.get 2",
            "    memory.init $seg",
            "    data.drop $seg",
            "    local.get 2",
            "    memory.grow",
            "    drop",
            "    memory.size",
            "    drop",
            "  )",
        ])
    );
}

#[test]
fn tables_references_and_a_typed_select() {
    let body = body_of(&format!(
        "wasm.table public @tab : 2 x !wasm.funcref;\n{}",
        f(
            "(%i: i32, %c: i32) -> i32",
            "%f = wasm.ref_func @f;\n%n = wasm.ref_null : !wasm.funcref;\n%s = wasm.select %c, \
             %f, %n;\nwasm.table_set @tab, %i, %s;\n%e = wasm.table_get @tab, %i : \
             !wasm.funcref;\n%old = wasm.table_grow @tab, %e, %i;\nwasm.table_fill @tab, %i, %n, \
             %c;\n%size = wasm.table_size @tab;\n%z = wasm.ref_is_null %e;\n%r = wasm.add %size, \
             %z;\nbuiltin.ret %r : (i32);"
        )
    ));
    assert_eq!(
        body,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32) (result i32)",
            "    (local funcref)",
            "    local.get 0",
            "    ref.func $f",
            "    ref.null func",
            "    local.get 1",
            "    select (result funcref)",
            "    table.set $tab",
            "    local.get 0",
            "    table.get $tab",
            "    local.tee 2",
            "    local.get 0",
            "    table.grow $tab",
            "    drop",
            "    local.get 0",
            "    ref.null func",
            "    local.get 1",
            "    table.fill $tab",
            "    table.size $tab",
            "    local.get 2",
            "    ref.is_null",
            "    i32.add",
            "  )",
        ])
    );
}

#[test]
fn a_function_that_ends_in_unreachable_and_one_that_ends_in_return_call_indirect() {
    let trap = body_of(&f("(%x: i32, %y: i32)", "%q = wasm.div_s %x, %y;\nwasm.unreachable;"));
    assert_eq!(
        trap,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32 i32)",
            "    local.get 0",
            "    local.get 1",
            "    i32.div_s",
            "    drop",
            "    unreachable",
            "  )",
        ])
    );
    let tail = body_of(&format!(
        "wasm.table private declaration @t : 0 x !wasm.funcref;\n{}",
        f(
            "(%i: i32) -> i64",
            "%k = wasm.const 2 : i64;\nwasm.return_call_indirect @t, %i(%k) : extern(\"wasm\") \
             (i64) -> i64;"
        )
    ));
    assert_eq!(
        tail,
        lines(&[
            "  (func $f (;0;) (type 0) (param i32) (result i64)",
            "    i64.const 2",
            "    local.get 0",
            "    return_call_indirect (type 1)",
            "  )",
        ])
    );
}
