//! The operations on a linear memory: loads, stores, the bulk operations, and their effects.

use runic_dialect_mem::MemoryResource;
use runic_ir::{
    Builder, Ident, SourceSpan, ValueRef, Visibility,
    dialects::builtin::BuiltinOpBuilder,
    effects::{DefaultResource, Resource},
    testing::{Test, text::round_trip},
};

use super::*;

/// The fourteen loads of Wasm, as the text of the access (`as ..`), and the type loaded.
const LOADS: [(&str, &str); 14] = [
    ("", "i32"),
    ("", "i64"),
    ("", "f32"),
    ("", "f64"),
    (" as i8 signed", "i32"),
    (" as i8 unsigned", "i32"),
    (" as i16 signed", "i32"),
    (" as i16 unsigned", "i32"),
    (" as i8 signed", "i64"),
    (" as i8 unsigned", "i64"),
    (" as i16 signed", "i64"),
    (" as i16 unsigned", "i64"),
    (" as i32 signed", "i64"),
    (" as i32 unsigned", "i64"),
];

/// The nine stores of Wasm: the text of the access, and the parameter stored, of the type the
/// store takes.
const STORES: [(&str, &str); 9] = [
    ("", "%a"),
    ("", "%l"),
    ("", "%f"),
    ("", "%d"),
    (" as i8", "%a"),
    (" as i16", "%a"),
    (" as i8", "%l"),
    (" as i16", "%l"),
    (" as i32", "%l"),
];

/// A load in the generic form, of the type `ty` through `storage` and `extension`.
fn generic_load(ty: &str, storage: &str, extension: &str) -> String {
    format!(
        "%v = \"wasm.load\"(%a) <{{ memory = #builtin.symbol<@mem>, memarg = #wasm.memarg<offset \
         = 0, align = 1>, storage = #wasm.storage<{storage}>, extension = \
         #wasm.extension<{extension}> }}> : i32 -> {ty};"
    )
}

/// A store in the generic form of `value`, of the type `ty`, through `storage`.
fn generic_store(value: &str, ty: &str, storage: &str) -> String {
    format!(
        "\"wasm.store\"(%a, {value}) <{{ memory = #builtin.symbol<@mem>, memarg = \
         #wasm.memarg<offset = 0, align = 1>, storage = #wasm.storage<{storage}> }}> : (i32, \
         {ty}) -> ();"
    )
}

#[test]
fn the_fourteen_loads_of_wasm_are_accepted() {
    for (access, ty) in LOADS {
        accepted(&format!("%v = wasm.load @mem, %a{access} : {ty};"));
    }
}

#[test]
fn exactly_the_fourteen_loads_of_wasm_verify() {
    let mut accepted_loads = 0;
    for ty in ["i32", "i64", "f32", "f64"] {
        for storage in ["full", "i8", "i16", "i32"] {
            for extension in ["unextended", "signed", "unsigned"] {
                let text = generic_load(ty, storage, extension);
                let result =
                    parse_state(&state_module(STATE_PARAMS, "", &format!("{text}\nbuiltin.ret;")));
                let access = match (storage, extension) {
                    ("full", "unextended") => String::new(),
                    (storage, extension) => format!(" as {storage} {extension}"),
                };
                let is_wasm = LOADS.contains(&(access.as_str(), ty));
                assert_eq!(result.is_ok(), is_wasm, "{ty}{access}: {:?}", result.as_ref().err());
                accepted_loads += usize::from(result.is_ok());
            }
        }
    }
    assert_eq!(accepted_loads, 14);
}

#[test]
fn the_nine_stores_of_wasm_are_accepted() {
    for (access, value) in STORES {
        accepted(&format!("wasm.store @mem, %a, {value}{access};"));
    }
}

#[test]
fn exactly_the_nine_stores_of_wasm_verify() {
    let mut accepted_stores = 0;
    for (value, ty) in [("%a", "i32"), ("%l", "i64"), ("%f", "f32"), ("%d", "f64")] {
        for storage in ["full", "i8", "i16", "i32"] {
            let text = generic_store(value, ty, storage);
            let result =
                parse_state(&state_module(STATE_PARAMS, "", &format!("{text}\nbuiltin.ret;")));
            let access = if storage == "full" {
                String::new()
            } else {
                format!(" as {storage}")
            };
            let is_wasm = STORES.contains(&(access.as_str(), value));
            assert_eq!(result.is_ok(), is_wasm, "{ty}{access}: {:?}", result.as_ref().err());
            accepted_stores += usize::from(result.is_ok());
        }
    }
    assert_eq!(accepted_stores, 9);
}

#[test]
fn the_forms_wasm_does_not_have_are_rejected_with_the_rule() {
    // No narrow load or store of a float
    rejected(
        "%v = wasm.load @mem, %a as i8 signed : f32;",
        "`wasm.load` takes an `i32` address in a memory and gives a number, `i32`, `i64`, `f32` \
         or `f64`, read from it, but its result has type `f32` and its storage is `i8`, and only \
         an integer is read through narrower storage (Wasm has no narrow float access)",
    );
    rejected(
        "wasm.store @mem, %a, %d as i16;",
        "its value has type `f64` and its storage is `i16`, and only an integer is written \
         through narrower storage (Wasm has no narrow float access)",
    );
    // Storage as wide as the value, or wider
    rejected(
        "%v = wasm.load @mem, %a as i32 signed : i32;",
        "its result has type `i32` and its storage is `i32`, which is not narrower than the value \
         (narrow storage is `i8` or `i16` for an `i32`; `i8`, `i16` or `i32` for an `i64`), and \
         an access of the whole value names no storage",
    );
    rejected("wasm.store @mem, %a, %a as i32;", "which is not narrower than the value");
    // A narrow load names how it extends, and a load of the whole value does not
    rejected(
        "%v = wasm.load @mem, %a as i8 : i32;",
        "its storage is `i8`, so a narrow load says whether it extends the value read with \
         `signed` or `unsigned`, and this one says neither",
    );
    let error = parse_state(&state_module(
        STATE_PARAMS,
        "",
        &format!("{}\nbuiltin.ret;", generic_load("i32", "full", "signed")),
    ))
    .err()
    .expect("a load of the whole value that extends does not verify");
    assert!(
        error.contains(
            "it is `signed`, and a load of the whole value extends nothing (only a narrow load is \
             `signed` or `unsigned`)"
        ),
        "{error}"
    );
    // A store has no signedness
    rejected(
        "wasm.store @mem, %a, %b as i8 signed;",
        "`signed` is not part of a store: it keeps the low bits of its value and has no signedness",
    );
    rejected("wasm.store @mem, %a, %b as i8 unsigned;", "`unsigned` is not part of a store");
}

#[test]
fn the_alignment_is_at_most_the_width_of_the_access() {
    for (access, ty) in LOADS {
        let width = if access.contains("i8") {
            1
        } else if access.contains("i16") {
            2
        } else if access.contains("i32") || ty == "f32" || ty == "i32" {
            4
        } else {
            8
        };
        accepted(&format!("%v = wasm.load @mem, %a{access} <{{ align = {width} }}> : {ty};"));
        if width < 8 {
            let error = format!(
                "its alignment is {} bytes and the access is {width} {} wide, and the alignment \
                 of a Wasm access is at most its width",
                width * 2,
                if width == 1 { "byte" } else { "bytes" }
            );
            rejected(
                &format!("%v = wasm.load @mem, %a{access} <{{ align = {} }}> : {ty};", width * 2),
                &error,
            );
        }
    }
    rejected(
        "wasm.store @mem, %a, %l as i16 <{ align = 4 }>;",
        "its alignment is 4 bytes and the access is 2 bytes wide",
    );
    accepted("wasm.store @mem, %a, %l as i16 <{ align = 2 }>;");
    accepted("wasm.store @mem, %a, %l <{ align = 1 }>;");
}

#[test]
fn the_alignment_is_a_power_of_two() {
    for align in [0, 3, 5, 6] {
        rejected(
            &format!("%v = wasm.load @mem, %a <{{ align = {align} }}> : i64;"),
            &format!(
                "an alignment of {align} bytes, but a Wasm memarg alignment is a power of two, at \
                 least 1"
            ),
        );
    }
}

#[test]
fn the_offset_is_an_unsigned_32_bit_integer() {
    accepted("%v = wasm.load @mem, %a <{ offset = 4294967295 }> : i32;");
    rejected(
        "%v = wasm.load @mem, %a <{ offset = 4294967296 }> : i32;",
        "its offset is 4294967296, and the offset of a Wasm access is an unsigned 32-bit integer",
    );
    rejected(
        "wasm.store @mem, %a, %b <{ offset = 4294967296 }>;",
        "its offset is 4294967296, and the offset of a Wasm access is an unsigned 32-bit integer",
    );
}

#[test]
fn the_text_of_an_access_is_read_in_any_order_and_without_duplicates() {
    accepted("%v = wasm.load @mem, %a <{ align = 2, offset = 8 }> : i32;");
    rejected(
        "%v = wasm.load @mem, %a <{ offset = 1, offset = 2 }> : i32;",
        "`offset` is given twice",
    );
    rejected("wasm.store @mem, %a, %b <{ align = 1, align = 1 }>;", "`align` is given twice");
}

#[test]
fn a_memory_access_prints_the_offset_and_the_alignment_only_when_they_differ_from_the_default() {
    let (_context, module) = accepted(
        "%v1 = wasm.load @mem, %a <{ offset = 0, align = 4 }> : i32;
         %v2 = wasm.load @mem, %a <{ offset = 8 }> : i32;
         %v3 = wasm.load @mem, %a <{ align = 2 }> : i32;
         %v4 = wasm.load @mem, %a as i8 signed <{ offset = 3, align = 1 }> : i64;
         wasm.store @mem, %a, %l as i32 <{ offset = 1, align = 2 }>;
         wasm.store @mem, %a, %l <{ offset = 0, align = 8 }>;",
    );
    let text = module.borrow().as_operation().to_string();
    for part in [
        "wasm.load @mem, %5 : i32;",
        "wasm.load @mem, %5 <{ offset = 8 }> : i32;",
        "wasm.load @mem, %5 <{ align = 2 }> : i32;",
        "wasm.load @mem, %5 as i8 signed <{ offset = 3 }> : i64;",
        "wasm.store @mem, %5, %10 as i32 <{ offset = 1, align = 2 }>;",
        "wasm.store @mem, %5, %10;",
    ] {
        assert!(text.contains(part), "`{part}` is not in\n{text}");
    }
}

#[test]
fn a_load_and_a_store_name_a_memory_and_take_an_i32_address() {
    rejected(
        "%v = wasm.load @tab, %a : i32;",
        "`wasm.load` names `@tab`, which is a `wasm.table`, not a `wasm.memory`",
    );
    rejected(
        "wasm.store @g, %a, %b;",
        "`wasm.store` names `@g`, which is a `wasm.global`, not a `wasm.memory`",
    );
    rejected("%v = wasm.load @nothing, %a : i32;", "nothing");
    rejected(
        "%v = wasm.load @mem, %l : i32;",
        "`wasm.load` takes an `i32` address in a memory and gives a number, `i32`, `i64`, `f32` \
         or `f64`, read from it, but its address has type `i64`",
    );
    rejected(
        "wasm.store @mem, %l, %b;",
        "`wasm.store` takes an `i32` address in a memory and a number, `i32`, `i64`, `f32` or \
         `f64`, to write there, but its address has type `i64`",
    );
    rejected("%v = wasm.load @mem, %r : i32;", "its address has type `funcref`");
}

#[test]
fn a_load_gives_and_a_store_takes_a_number() {
    for ty in ["!wasm.funcref", "!wasm.externref", "i8", "i1", "isize", "ptr<i32>"] {
        let shown = match ty {
            "!wasm.funcref" => "funcref",
            "!wasm.externref" => "externref",
            other => other,
        };
        rejected(
            &format!("%v = wasm.load @mem, %a : {ty};"),
            &format!("but its result has type `{shown}`"),
        );
    }
    rejected("wasm.store @mem, %a, %r;", "but its value has type `funcref`");
    rejected("wasm.store @mem, %a, %x;", "but its value has type `externref`");
    rejected("wasm.store @mem, %a, %p;", "but its value has type `ptr<i32>`");
}

#[test]
fn a_store_has_no_extension_property() {
    // The generic form of a store with the property a load has: the property is not the store's
    let text = "\"wasm.store\"(%a, %b) <{ memory = #builtin.symbol<@mem>, memarg = \
                #wasm.memarg<offset = 0, align = 1>, storage = #wasm.storage<i8>, extension = \
                #wasm.extension<signed> }> : (i32, i32) -> ();";
    let result = parse_state(&state_module(STATE_PARAMS, "", &format!("{text}\nbuiltin.ret;")));
    // Whatever the framework does with a property an operation does not have, the extension is
    // not kept by the store: it prints with its own properties only
    if let Ok((_, module)) = result {
        let printed = module.borrow().as_operation().to_string();
        assert!(!printed.contains("signed"), "{printed}");
    }
}

#[test]
fn the_memory_size_and_grow_operations_take_i32s_and_give_an_i32() {
    accepted("%n = wasm.memory_size @mem;\n %o = wasm.memory_grow @mem, %n;");
    rejected(
        "%n = wasm.memory_size @tab;",
        "`wasm.memory_size` names `@tab`, which is a `wasm.table`, not a `wasm.memory`",
    );
    rejected(
        "%o = wasm.memory_grow @mem, %l;",
        "`wasm.memory_grow` takes a memory and an `i32` number of pages and gives an `i32`, but \
         its delta has type `i64`",
    );
    rejected(
        "%o = wasm.memory_grow @g, %a;",
        "`wasm.memory_grow` names `@g`, which is a `wasm.global`, not a `wasm.memory`",
    );
}

#[test]
fn the_bulk_operations_take_their_memories_and_three_i32s() {
    accepted(
        "wasm.memory_copy @mem, @other, %a, %b, %a;
         wasm.memory_fill @mem, %a, %b, %a;
         wasm.memory_init @mem, @data, %a, %b, %a;
         wasm.data_drop @data;",
    );
    rejected(
        "wasm.memory_copy @mem, @other, %a, %l, %a;",
        "`wasm.memory_copy` takes two memories and three `i32`s: a destination address, a source \
         address and a number of bytes, but its source address has type `i64`",
    );
    rejected(
        "wasm.memory_copy @mem, @tab, %a, %a, %a;",
        "`wasm.memory_copy` names `@tab`, which is a `wasm.table`, not a `wasm.memory`",
    );
    rejected(
        "wasm.memory_copy @tab, @mem, %a, %a, %a;",
        "`wasm.memory_copy` names `@tab`, which is a `wasm.table`, not a `wasm.memory`",
    );
    rejected(
        "wasm.memory_fill @mem, %a, %a, %l;",
        "`wasm.memory_fill` takes a memory and three `i32`s: a destination address, a byte value \
         and a number of bytes, but its length has type `i64`",
    );
    rejected("wasm.memory_init @mem, @data, %a, %a, %l;", "but its length has type `i64`");
    rejected(
        "wasm.memory_init @mem, @k, %a, %a, %a;",
        "`wasm.memory_init` names `@k`, which is a `wasm.global`, not a `builtin.global_variable`",
    );
    rejected(
        "wasm.memory_init @data, @data, %a, %a, %a;",
        "`wasm.memory_init` names `@data`, which is a `builtin.global_variable`, not a \
         `wasm.memory`",
    );
    rejected(
        "wasm.data_drop @mem;",
        "`wasm.data_drop` names `@mem`, which is a `wasm.memory`, not a `builtin.global_variable`",
    );
}

#[test]
fn every_memory_operation_round_trips_in_the_custom_and_the_generic_form() {
    let mut body = String::new();
    for (index, (access, ty)) in LOADS.iter().enumerate() {
        body +=
            &format!("%l{index} = wasm.load @mem, %a{access} <{{ offset = {index} }}> : {ty};\n");
    }
    for (access, value) in STORES {
        body += &format!("wasm.store @mem, %a, {value}{access} <{{ offset = 4 }}>;\n");
    }
    body += "%n = wasm.memory_size @mem;
             %o = wasm.memory_grow @mem, %n;
             wasm.memory_copy @mem, @other, %a, %b, %n;
             wasm.memory_fill @other, %a, %b, %n;
             wasm.memory_init @mem, @data, %a, %b, %n;
             wasm.data_drop @data;";
    let (_context, module) = accepted(&body);
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

#[test]
fn the_operation_names_are_the_wasm_instructions_with_underscores() {
    let (_context, module) = accepted(
        "%v = wasm.load @mem, %a : i32;
         wasm.store @mem, %a, %v;
         %n = wasm.memory_size @mem;
         %o = wasm.memory_grow @mem, %n;
         wasm.memory_copy @mem, @other, %a, %b, %n;
         wasm.memory_fill @mem, %a, %b, %n;
         wasm.memory_init @mem, @data, %a, %b, %n;
         wasm.data_drop @data;",
    );
    let text = module.borrow().as_operation().to_string();
    for name in [
        "wasm.load",
        "wasm.store",
        "wasm.memory_size",
        "wasm.memory_grow",
        "wasm.memory_copy",
        "wasm.memory_fill",
        "wasm.memory_init",
        "wasm.data_drop",
    ] {
        assert!(text.contains(name), "{name} is not in\n{text}");
    }
}

#[test]
fn the_builders_build_every_memory_operation() {
    let mut test = Test::named("f").in_module("m");
    let context = test.context_rc();
    context.get_or_register_dialect::<WasmDialect>();
    let (i32, i64) = (context.i32(), context.i64());
    let module = test.module();
    let body = module.borrow().body().entry_block_ref().unwrap();
    let ident = |name: &str| Ident::with_empty_span(name.to_string().into());
    let (mem, other, data) = {
        let builder = test.builder_mut();
        builder.set_insertion_point_to_end(body);
        let mem = builder
            .create_memory(ident("mem"), Visibility::Public, Limits::between(1, 16))
            .unwrap();
        let other = builder
            .create_memory(ident("other"), Visibility::Public, Limits::at_least(1))
            .unwrap();
        let data = builder
            .create_global_variable(ident("data"), Visibility::Private, context.i32())
            .unwrap();
        (mem, other, data)
    };
    test.with_function("f", &[i32, i32], &[]);
    let (a, b) = {
        let entry = test.entry_block();
        let entry = entry.borrow();
        (entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef)
    };
    let builder = test.builder_mut();
    let span = SourceSpan::UNKNOWN;

    let v = builder.load(mem, a, i32, MemArg::new(4, 4), span).unwrap();
    assert_eq!(v.borrow().ty(), i32);
    let w = builder
        .load_narrow(mem, a, i64, Storage::I8, Extension::Signed, MemArg::new(0, 1), span)
        .unwrap();
    assert_eq!(w.borrow().ty(), i64);
    builder.store(mem, a, v, MemArg::new(0, 4), span).unwrap();
    builder.store_narrow(mem, a, w, Storage::I16, MemArg::new(0, 2), span).unwrap();
    let pages = builder.memory_size(mem, span).unwrap();
    assert_eq!(pages.borrow().ty(), i32);
    let old = builder.memory_grow(mem, pages, span).unwrap();
    assert_eq!(old.borrow().ty(), i32);
    builder.memory_copy(mem, other, a, b, pages, span).unwrap();
    builder.memory_fill(mem, a, b, pages, span).unwrap();
    builder.memory_init(mem, data, a, b, pages, span).unwrap();
    builder.data_drop(data, span).unwrap();
    builder.ret(None, span).unwrap();

    module.borrow().as_operation().recursively_verify().unwrap_or_else(|report| {
        panic!("{}\n{}", describe(&report), module.borrow().as_operation())
    });
    let text = module.borrow().as_operation().to_string();
    for line in [
        "wasm.load @::@m::@mem, %",
        "<{ offset = 4 }>",
        "as i8 signed",
        "wasm.store @::@m::@mem",
        "as i16",
        "wasm.memory_size",
        "wasm.memory_grow",
        "wasm.memory_copy",
        "wasm.memory_fill",
        "wasm.memory_init",
        "wasm.data_drop",
    ] {
        let line = line.replace("@::@m::@mem", "::@m::@mem");
        assert!(text.contains(&line), "`{line}` is not in\n{text}");
    }
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn the_resource_of_the_linear_memory_is_ordered_against_the_mem_dialects_memory() {
    let memory: &dyn Resource = &LinearMemoryResource;
    assert!(!memory.is_disjoint_from(&MemoryResource));
    assert!(!(&MemoryResource as &dyn Resource).is_disjoint_from(memory));
    assert!(!memory.is_disjoint_from(&DefaultResource));
    assert!(memory.is_disjoint_from(&GlobalsResource));
    assert!(memory.is_disjoint_from(&TablesResource));
    assert!((&GlobalsResource as &dyn Resource).is_disjoint_from(&TablesResource));
    assert!((&GlobalsResource as &dyn Resource).is_disjoint_from(&MemoryResource));
    assert!(!(&GlobalsResource as &dyn Resource).is_disjoint_from(&DefaultResource));
}

#[test]
fn each_memory_operation_declares_its_effects() {
    let cases: [(&str, &str, &[&str]); 8] = [
        ("wasm.load", "%v = wasm.load @mem, %a : i32;", &["Read wasm.memory"]),
        ("wasm.store", "wasm.store @mem, %a, %b;", &["Write wasm.memory"]),
        ("wasm.memory_size", "%v = wasm.memory_size @mem;", &["Read wasm.memory"]),
        ("wasm.memory_grow", "%v = wasm.memory_grow @mem, %a;", &["Write wasm.memory"]),
        (
            "wasm.memory_copy",
            "wasm.memory_copy @mem, @other, %a, %a, %a;",
            &["Write wasm.memory", "Read wasm.memory"],
        ),
        ("wasm.memory_fill", "wasm.memory_fill @mem, %a, %a, %a;", &["Write wasm.memory"]),
        (
            "wasm.memory_init",
            "wasm.memory_init @mem, @data, %a, %a, %a;",
            &["Read wasm.memory", "Write wasm.memory"],
        ),
        ("wasm.data_drop", "wasm.data_drop @data;", &["Write wasm.memory"]),
    ];
    for (name, body, expected) in cases {
        let mut found = effects_of(name, body);
        let mut expected = expected.iter().map(|effect| effect.to_string()).collect::<Vec<_>>();
        found.sort();
        expected.sort();
        assert_eq!(found, expected, "{name}");
    }
}

const LOAD_A: &str = "%l1 = wasm.load @mem, %a : i32;";
const LOAD_B: &str = "%l2 = wasm.load @mem, %a : i32;";

#[test]
fn two_loads_of_one_address_merge() {
    assert_eq!(after_cse(LOAD_A, "", LOAD_B, "wasm.load"), 1);
    // The memory, the address, the offset and the storage all tell loads apart
    assert_eq!(after_cse(LOAD_A, "", "%l2 = wasm.load @other, %a : i32;", "wasm.load"), 2);
    assert_eq!(after_cse(LOAD_A, "", "%l2 = wasm.load @mem, %b : i32;", "wasm.load"), 2);
    assert_eq!(
        after_cse(LOAD_A, "", "%l2 = wasm.load @mem, %a <{ offset = 1 }> : i32;", "wasm.load"),
        2
    );
    assert_eq!(
        after_cse(LOAD_A, "", "%l2 = wasm.load @mem, %a as i8 signed : i32;", "wasm.load"),
        2
    );
    assert_eq!(
        after_cse(LOAD_A, "", "%l2 = wasm.load @mem, %a as i16 signed : i32;", "wasm.load"),
        2
    );
}

#[test]
fn two_loads_do_not_merge_across_an_operation_that_may_write_memory() {
    for middle in [
        "wasm.store @mem, %a, %b;",
        "wasm.store @other, %a, %b;",
        "wasm.store @mem, %a, %b as i8;",
        "%g = wasm.memory_grow @mem, %b;",
        "wasm.memory_fill @mem, %a, %b, %b;",
        "wasm.memory_copy @mem, @other, %a, %a, %b;",
        "wasm.memory_init @mem, @data, %a, %a, %b;",
        "wasm.data_drop @data;",
        "builtin.call @opaque() : extern(\"C\") () -> ();",
        "%c = builtin.call @callee(%a) : extern(\"C\") (i32) -> i32;",
        "wasm.call_indirect @tab, %a() : extern(\"C\") () -> ();",
        "mem.store %b, %p;",
    ] {
        assert_eq!(after_cse(LOAD_A, middle, LOAD_B, "wasm.load"), 2, "{middle}");
    }
}

#[test]
fn two_loads_merge_across_an_operation_that_cannot_write_memory() {
    for middle in [
        "wasm.global_set @g, %a;",
        "wasm.table_set @tab, %a, %r;",
        "%n = wasm.table_grow @tab, %r, %a;",
        "wasm.table_fill @tab, %a, %r, %a;",
        "%s = wasm.memory_size @mem;",
        "%z = wasm.load @mem, %b : i32;",
        "%e = wasm.global_get @g : i32;",
        "%t = wasm.table_get @tab, %a : !wasm.funcref;",
        "%null = wasm.ref_null : !wasm.funcref;",
    ] {
        assert_eq!(after_cse(LOAD_A, middle, LOAD_B, "wasm.load"), 1, "{middle}");
    }
}

#[test]
fn a_wasm_store_and_a_mem_load_touch_the_same_memory() {
    // A `mem.load` is not merged across a `wasm.store`, and a `wasm.load` is not across a `mem.store`
    assert_eq!(
        after_cse(
            "%l1 = mem.load %p;",
            "wasm.store @mem, %a, %b;",
            "%l2 = mem.load %p;",
            "mem.load"
        ),
        2
    );
    assert_eq!(
        after_cse(
            "%l1 = mem.load %p;",
            "wasm.memory_fill @mem, %a, %b, %b;",
            "%l2 = mem.load %p;",
            "mem.load"
        ),
        2
    );
    assert_eq!(
        after_cse(
            "%l1 = mem.load %p;",
            "%g = wasm.memory_grow @mem, %b;",
            "%l2 = mem.load %p;",
            "mem.load"
        ),
        2
    );
    assert_eq!(after_cse("%l1 = mem.load %p;", "", "%l2 = mem.load %p;", "mem.load"), 1);
    assert_eq!(
        after_cse(
            "%l1 = mem.load %p;",
            "wasm.global_set @g, %a;",
            "%l2 = mem.load %p;",
            "mem.load"
        ),
        1
    );
    assert_eq!(after_cse(LOAD_A, "mem.store %b, %p;", LOAD_B, "wasm.load"), 2);
}

#[test]
fn two_sizes_of_a_memory_merge_only_when_it_has_not_grown() {
    assert_eq!(
        after_cse(
            "%l1 = wasm.memory_size @mem;",
            "",
            "%l2 = wasm.memory_size @mem;",
            "wasm.memory_size"
        ),
        1
    );
    assert_eq!(
        after_cse(
            "%l1 = wasm.memory_size @mem;",
            "%g = wasm.memory_grow @mem, %a;",
            "%l2 = wasm.memory_size @mem;",
            "wasm.memory_size"
        ),
        2
    );
}

#[test]
fn an_unused_load_is_removed_and_a_store_is_never() {
    let kept = |body: &str, op: &str| count(&state_dce(body), op);
    assert_eq!(kept("%v = wasm.load @mem, %a : i32;", "wasm.load"), 0);
    assert_eq!(kept("%v = wasm.load @mem, %a as i8 signed : i64;", "wasm.load"), 0);
    assert_eq!(kept("%v = wasm.memory_size @mem;", "wasm.memory_size"), 0);
    for (body, op) in [
        ("wasm.store @mem, %a, %b;", "wasm.store"),
        ("wasm.store @mem, %a, %l as i32;", "wasm.store"),
        ("%v = wasm.memory_grow @mem, %a;", "wasm.memory_grow"),
        ("wasm.memory_copy @mem, @other, %a, %a, %a;", "wasm.memory_copy"),
        ("wasm.memory_fill @mem, %a, %a, %a;", "wasm.memory_fill"),
        ("wasm.memory_init @mem, @data, %a, %a, %a;", "wasm.memory_init"),
        ("wasm.data_drop @data;", "wasm.data_drop"),
    ] {
        assert_eq!(kept(body, op), 1, "{body}");
    }
}

#[test]
fn a_used_load_is_kept() {
    let text = state_dce("%v = wasm.load @mem, %a : i32;\n wasm.store @mem, %a, %v;");
    assert_eq!(count(&text, "wasm.load"), 1);
}

#[test]
fn a_trap_is_not_an_effect_so_an_unused_load_out_of_bounds_is_still_removed() {
    // The rule `arith` states: an operation that may trap is removed with its trap when unused
    let text = state_dce("%v = wasm.load @mem, %a <{ offset = 4294967295 }> : i64;");
    assert_eq!(count(&text, "wasm.load"), 0);
}
