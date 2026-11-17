//! The layout of a complete module through [emit_module]: global variables as data, the shadow
//! stack, sections and the linker-defined data symbols, zeroed and passive data, fixed segments,
//! the memory and the function table. Modules are written in generic IR or in Wasm form, lowered
//! with the crate's pipeline and emitted; what is written is checked as text, and what it does is
//! run under `wasmi` with no imports.

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{
    Context, Op, OperationRef,
    dialects::builtin::World,
    parse::{self, ParserConfig},
    testing::text::describe,
};

use crate::{
    emit::{EmitOptions, emit_module},
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

/// The declaration of the function table the lowering names, as it writes it.
const TABLE: &str = "wasm.table private declaration @__indirect_function_table : 0 x \
                     !wasm.funcref attributes { link_name = \"__indirect_function_table\" };";

/// A module `@name` of the items `items`, with the data layout of the lowering.
fn module(name: &str, items: &[&str]) -> String {
    format!("builtin.module public @{name} {LAYOUT} {{\n{}\n}};", items.join("\n"))
}

/// The world of `modules`, parsed, verified and lowered by the crate's pipeline.
#[track_caller]
fn lowered(modules: &[String]) -> (Rc<Context>, OperationRef) {
    let text = format!("builtin.world {{\n{}\n}};", modules.join("\n"));
    let context = context();
    let world = parse::parse::<World>(ParserConfig::new(context.clone()), "w", &text)
        .into_result()
        .and_then(|world| {
            world.borrow().as_operation().recursively_verify()?;
            Ok(world.as_operation_ref())
        })
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    run_pipeline(&context, world, LOWERING).unwrap_or_else(|report| {
        panic!("{}\n--- the IR after the failure:\n{}", describe(&report), world.borrow())
    });
    (context, world)
}

/// An emitted module.
struct Emitted {
    bytes: Vec<u8>,
}

impl Emitted {
    /// The module as text.
    fn wat(&self) -> String {
        print(&self.bytes).unwrap_or_else(|report| panic!("{report}"))
    }

    /// The module instantiated with no imports.
    fn run(&self) -> Instance {
        Instance::new(&self.bytes)
    }
}

/// The modules `modules` lowered and emitted with `options` (which validates them).
#[track_caller]
fn emitted_with(modules: &[String], options: &EmitOptions) -> Emitted {
    let (_context, world) = lowered(modules);
    let bytes =
        emit_module(world, options).unwrap_or_else(|report| panic!("{}", describe(&report)));
    Emitted { bytes }
}

/// The modules `modules` lowered and emitted with the default options.
#[track_caller]
fn emitted(modules: &[String]) -> Emitted {
    emitted_with(modules, &EmitOptions::default())
}

/// The error of emitting the modules `modules` with `options`, with its labels.
#[track_caller]
fn failure_with(modules: &[String], options: &EmitOptions) -> String {
    let (_context, world) = lowered(modules);
    error_of(emit_module(world, options))
}

/// The error of emitting the modules `modules` with the default options.
#[track_caller]
fn failure(modules: &[String]) -> String {
    failure_with(modules, &EmitOptions::default())
}

/// Emit options with a stack of `size` bytes.
fn stack_of(size: u32) -> EmitOptions {
    EmitOptions {
        stack_size: size,
        ..EmitOptions::default()
    }
}

/// Asserts that `haystack` holds every one of `needles`.
#[track_caller]
fn assert_mentions(haystack: &str, needles: &[&str]) {
    for needle in needles {
        assert!(mentions(haystack, needle), "no `{needle}` in:\n{haystack}");
    }
}

/// The one value of `outcome`, as bits.
#[track_caller]
fn one(outcome: Outcome) -> u64 {
    match outcome {
        Outcome::Values(values) if values.len() == 1 => values[0].bits,
        other => panic!("expected one value, got {other:?}"),
    }
}

/// A Wasm-form function `@name`, exported as `name`, that returns the address of `@symbol`.
fn address_of(name: &str, symbol: &str) -> String {
    format!(
        "builtin.function public extern(\"wasm\") @{name}() -> i32 {{\n    %a = wasm.data_addr \
         @{symbol};\n    builtin.ret %a : (i32);\n}} attributes {{ wasm.export_name = \"{name}\" \
         }};"
    )
}

/// A Wasm-form function `@name`, exported as `name`, that loads `@symbol` with `access` (what
/// follows the address in `wasm.load`) and returns the value, of type `ty`.
fn loader(name: &str, symbol: &str, access: &str, ty: &str) -> String {
    format!(
        "builtin.function public extern(\"wasm\") @{name}() -> {ty} {{\n    %a = wasm.data_addr \
         @{symbol};\n    %v = wasm.load @__linear_memory, %a{access};\n    builtin.ret %v : \
         ({ty});\n}} attributes {{ wasm.export_name = \"{name}\" }};"
    )
}

/// A Wasm-form function `@name`, exported as `name`, that loads the `i32` at the address it is
/// given.
fn peek(name: &str) -> String {
    format!(
        "builtin.function public extern(\"wasm\") @{name}(%a: i32) -> i32 {{\n    %v = wasm.load \
         @__linear_memory, %a : i32;\n    builtin.ret %v : (i32);\n}} attributes {{ \
         wasm.export_name = \"{name}\" }};"
    )
}

/// A global variable `@name` of type `ty` whose initializer is the constant `value` of type `ty`.
fn constant(name: &str, ty: &str, value: &str, attributes: &str) -> String {
    format!(
        "builtin.global_variable private @{name} : {ty} {{ %v = arith.constant {value} : {ty}; \
         builtin.yield %v; }}{attributes};"
    )
}

/// A generic function `@name`, exported as `name`, with a frame: it stores `%x` in a stack slot
/// and returns what it loads back.
fn framed(name: &str) -> String {
    format!(
        "builtin.function public extern(\"C\") @{name}(%x: i32) -> i32 {{\n    %slot = mem.alloca \
         i32;\n    mem.store %x, %slot;\n    %v = mem.load %slot;\n    builtin.ret %v : \
         (i32);\n}} attributes {{ wasm.export_name = \"{name}\" }};"
    )
}

// ---- Data ---------------------------------------------------------------------------------------

#[test]
fn a_byte_array_is_laid_out_and_read_back_by_a_loop() {
    let number = r#"builtin.function public extern("C") @number() -> i32 {
    %base = builtin.global_symbol @digits : ptr<array<i8; 4>>;
    %p = mem.ptrcast %base : ptr<i8>;
    %zero = arith.constant 0 : i32;
    %first = arith.constant 0 : isize;
    %n = arith.constant 4 : isize;
    %ten = arith.constant 10 : i32;
    cf.br ^head(%first, %zero : isize, i32);
^head(%i: isize, %acc: i32):
    %done = arith.cmpi uge %i, %n;
    cf.cond_br %done ^exit, ^body : (i1);
^body:
    %q = mem.gep %p, %i;
    %b = mem.load %q;
    %w = arith.zext %b <{ ty = i32 }>;
    %s = arith.mul %acc, %ten;
    %acc2 = arith.add %s, %w;
    %one = arith.constant 1 : isize;
    %i2 = arith.add %i, %one;
    cf.br ^head(%i2, %acc2 : isize, i32);
^exit:
    builtin.ret %acc : (i32);
} attributes { wasm.export_name = "number" };"#;
    // The first byte of `builtin.bytes` is the last pair of digits: memory holds 1, 2, 3, 4.
    let digits = "builtin.global_variable private @digits : array<i8; 4> { %b = builtin.bytes \
                  \"04030201\" : array<i8; 4>; builtin.yield %b; };";
    let emitted = emitted(&[module("m", &[digits, number])]);
    let text = emitted.wat();
    // No stack (nothing has a frame): the data starts at 1024.
    assert_mentions(
        &text,
        &[
            "(memory $__linear_memory (;0;) 1)",
            "(export \"memory\" (memory $__linear_memory))",
            "(data $m::digits (;0;) (i32.const 1024) \"\\01\\02\\03\\04\")",
        ],
    );
    assert_eq!(one(emitted.run().call("number", &[])), 1234);
}

#[test]
fn integer_and_float_globals_of_each_width_are_read_back() {
    let globals = [
        constant("b", "i8", "-3", ""),
        constant("h", "i16", "-300", ""),
        constant("w", "i32", "123456", ""),
        constant("d", "i64", "-5000000000", ""),
        String::from(
            "builtin.global_variable private @f : f32 { %v = arith.constantf 1.5 : f32; \
             builtin.yield %v; };",
        ),
        String::from(
            "builtin.global_variable private @g : f64 { %v = arith.constantf -2.25 : f64; \
             builtin.yield %v; };",
        ),
    ];
    let loaders = [
        loader("get_b", "b", " as i8 signed : i32", "i32"),
        loader("get_h", "h", " as i16 signed : i32", "i32"),
        loader("get_w", "w", " : i32", "i32"),
        loader("get_d", "d", " : i64", "i64"),
        loader("get_f", "f", " : f32", "f32"),
        loader("get_g", "g", " : f64", "f64"),
    ];
    let mut items = alloc::vec![MEMORY];
    items.extend(globals.iter().map(String::as_str));
    items.extend(loaders.iter().map(String::as_str));
    let emitted = emitted(&[module("m", &items)]);
    let text = emitted.wat();
    // Each at a multiple of its alignment, from 1024.
    assert_mentions(
        &text,
        &[
            "(data $m::b (;0;) (i32.const 1024) \"\\fd\")",
            "(data $m::h (;1;) (i32.const 1026) \"\\d4\\fe\")",
            "(data $m::w (;2;) (i32.const 1028) \"@\\e2\\01\\00\")",
            "(data $m::d (;3;) (i32.const 1032) \"\\00\\0e\\fa\\d5\\fe\\ff\\ff\\ff\")",
            "(data $m::f (;4;) (i32.const 1040) \"\\00\\00\\c0?\")",
            "(data $m::g (;5;) (i32.const 1048) \"\\00\\00\\00\\00\\00\\00\\02\\c0\")",
        ],
    );
    let mut instance = emitted.run();
    assert_eq!(one(instance.call("get_b", &[])), u64::from(-3i32 as u32));
    assert_eq!(one(instance.call("get_h", &[])), u64::from(-300i32 as u32));
    assert_eq!(one(instance.call("get_w", &[])), 123_456);
    assert_eq!(one(instance.call("get_d", &[])), -5_000_000_000i64 as u64);
    assert_eq!(one(instance.call("get_f", &[])), u64::from(1.5f32.to_bits()));
    assert_eq!(one(instance.call("get_g", &[])), (-2.25f64).to_bits());
}

#[test]
fn an_unfolded_addition_in_an_initializer_is_folded_by_the_emitter() {
    let sum = "builtin.global_variable private @sum : i32 { %a = arith.constant 40 : i32; %b = \
               arith.constant 2 : i32; %c = arith.add %a, %b; builtin.yield %c; };";
    let get = loader("get", "sum", " : i32", "i32");
    let emitted = emitted(&[module("m", &[MEMORY, sum, &get])]);
    assert_eq!(one(emitted.run().call("get", &[])), 42);
}

#[test]
fn a_pointer_global_holds_the_address_of_another_global_and_is_followed() {
    let target = constant("target", "i32", "7", "");
    let pointer = "builtin.global_variable private @pointer : ptr<i32> { %p = \
                   builtin.global_symbol @target : ptr<i32>; builtin.yield %p; };";
    let follow = r#"builtin.function public extern("C") @follow() -> i32 {
    %pp = builtin.global_symbol @pointer : ptr<ptr<i32>>;
    %p = mem.load %pp;
    %v = mem.load %p;
    builtin.ret %v : (i32);
} attributes { wasm.export_name = "follow" };"#;
    let emitted = emitted(&[module("m", &[&target, pointer, follow])]);
    // The address 1024 of `@target`, little-endian, where the fix-up was.
    assert_mentions(
        &emitted.wat(),
        &["(data $m::pointer (;1;) (i32.const 1028) \"\\00\\04\\00\\00\")"],
    );
    assert_eq!(one(emitted.run().call("follow", &[])), 7);
}

#[test]
fn a_function_pointer_in_data_is_loaded_and_called_indirectly() {
    let double = "builtin.function public extern(\"wasm\") @double(%x: i32) -> i32 {\n    %y = \
                  wasm.add %x, %x;\n    builtin.ret %y : (i32);\n};";
    let pointer = "builtin.global_variable private @fp : (i32) -> i32 { %f = builtin.function_ref \
                   @double : (i32) -> i32; builtin.yield %f; };";
    let call = r#"builtin.function public extern("wasm") @call(%x: i32) -> i32 {
    %a = wasm.data_addr @fp;
    %i = wasm.load @__linear_memory, %a : i32;
    %r = wasm.call_indirect @__indirect_function_table, %i(%x) : extern("wasm") (i32) -> i32;
    builtin.ret %r : (i32);
} attributes { wasm.export_name = "call" };"#;
    let emitted = emitted(&[module("m", &[MEMORY, TABLE, double, pointer, call])]);
    let text = emitted.wat();
    // The slot of `@double` is 1, in a table of two.
    assert_mentions(
        &text,
        &[
            "(table $__indirect_function_table (;0;) 2 2 funcref)",
            "(elem (;0;) (i32.const 1) func $m::double)",
            "(data $m::fp (;0;) (i32.const 1024) \"\\01\\00\\00\\00\")",
        ],
    );
    assert_eq!(one(emitted.run().call("call", &[Val::i32(21)])), 42);
}

#[test]
fn a_global_after_a_narrower_one_is_at_a_multiple_of_its_alignment() {
    let byte = constant("byte", "i8", "1", "");
    let wide = constant("wide", "i64", "2", "");
    let emitted = emitted(&[module(
        "m",
        &[
            MEMORY,
            &byte,
            &wide,
            &address_of("byte_at", "byte"),
            &address_of("wide_at", "wide"),
        ],
    )]);
    let mut instance = emitted.run();
    assert_eq!(one(instance.call("byte_at", &[])), 1024);
    assert_eq!(one(instance.call("wide_at", &[])), 1032);
}

#[test]
fn a_wasm_global_initialized_to_a_data_address_and_one_to_a_table_slot_are_read_back() {
    let data = constant("data", "i32", "5", "");
    let callee = "builtin.function public extern(\"wasm\") @callee() { builtin.ret; };";
    let address = "wasm.global private @address : i32 { %v = wasm.data_addr @data + 8; \
                   builtin.yield %v; } attributes { wasm.export_name = \"address\" };";
    let slot = "wasm.global private @slot : i32 { %v = wasm.func_index @callee; builtin.yield %v; \
                } attributes { wasm.export_name = \"slot\" };";
    let emitted = emitted(&[module("m", &[MEMORY, &data, callee, address, slot])]);
    let text = emitted.wat();
    assert_mentions(
        &text,
        &[
            "(global $m::address (;0;) i32 i32.const 1032)",
            "(global $m::slot (;1;) i32 i32.const 1)",
            // The table the layout defines for the slot: no unit declared one.
            "(table $__indirect_function_table (;0;) 2 2 funcref)",
        ],
    );
    let instance = emitted.run();
    assert_eq!(instance.global("address").bits, 1032);
    assert_eq!(instance.global("slot").bits, 1);
}

// ---- The stack ----------------------------------------------------------------------------------

#[test]
fn the_stack_pointer_starts_at_the_stack_size_and_a_function_with_a_frame_runs() {
    let data = constant("after", "i32", "9", "");
    for (options, size) in [(EmitOptions::default(), 1_048_576u64), (stack_of(65_536), 65_536)] {
        let emitted = emitted_with(
            &[module("m", &[&framed("framed"), &data, &address_of("after_at", "after")])],
            &options,
        );
        let text = emitted.wat();
        assert_mentions(
            &text,
            &[&format!("(global $__stack_pointer (;0;) (mut i32) i32.const {size})")],
        );
        // The stack pointer is defined, not imported or exported: no mutable-globals.
        assert!(!text.contains("(import"), "{text}");
        assert!(!text.contains("mutable-globals"), "{text}");
        let mut instance = emitted.run();
        assert_eq!(one(instance.call("framed", &[Val::i32(77)])), 77);
        // The data starts at the end of the stack.
        assert_eq!(one(instance.call("after_at", &[])), size);
    }
}

#[test]
fn a_recursion_that_overflows_the_stack_traps_and_the_data_after_the_stack_is_intact() {
    let deep = r#"builtin.function public extern("C") @deep(%n: i32) -> i32 {
    %slot = mem.alloca array<i32; 1024>;
    %p = mem.ptrcast %slot : ptr<i32>;
    mem.store %n, %p;
    %zero = arith.constant 0 : i32;
    %done = arith.cmpi eq %n, %zero;
    cf.cond_br %done ^exit, ^more : (i1);
^more:
    %one = arith.constant 1 : i32;
    %m = arith.sub %n, %one;
    %r = builtin.call @deep(%m) : extern("C") (i32) -> i32;
    %v = mem.load %p;
    %s = arith.add %r, %v;
    builtin.ret %s : (i32);
^exit:
    builtin.ret %zero : (i32);
} attributes { wasm.export_name = "deep" };"#;
    let data = constant("after", "i32", "1234567", "");
    let emitted = emitted_with(
        &[module(
            "m",
            &[MEMORY, deep, &data, &loader("read_after", "after", " : i32", "i32")],
        )],
        &stack_of(65_536),
    );
    let mut instance = emitted.run();
    // Ten frames of 4 KiB fit in 64 KiB; a hundred do not.
    assert_eq!(one(instance.call("deep", &[Val::i32(10)])), 55);
    assert_eq!(instance.call("deep", &[Val::i32(100)]), Outcome::Trap);
    assert_eq!(one(instance.call("read_after", &[])), 1_234_567);
}

#[test]
fn a_stack_size_that_is_not_a_multiple_of_16_is_an_error_naming_the_option() {
    let error = failure_with(&[module("m", &[&framed("framed")])], &stack_of(1000));
    assert_mentions(
        &error,
        &[
            "the stack size 1000 (`--wasm-stack-size`) is not a multiple of 16, the alignment of \
             the stack",
            "the stack this stack pointer points into is laid out with that size",
        ],
    );
}

#[test]
fn a_stack_size_that_is_not_a_multiple_of_16_is_an_error_when_nothing_uses_the_stack() {
    // No function has a frame, so the layout reserves no stack; the option is checked all the
    // same.
    let error = failure_with(&[module("m", &[&constant("word", "i32", "1", "")])], &stack_of(8));
    assert_mentions(
        &error,
        &[
            "the stack size 8 (`--wasm-stack-size`) is not a multiple of 16, the alignment of the \
             stack",
        ],
    );
}

#[test]
fn a_stack_size_of_0_is_an_error_since_data_would_start_at_address_0() {
    let message = "the stack size 0 (`--wasm-stack-size`) is not positive: data would start at \
                   the address 0, which a null pointer has";
    let error = failure_with(
        &[module("m", &[&framed("framed"), &constant("word", "i32", "1", "")])],
        &stack_of(0),
    );
    assert_mentions(
        &error,
        &[message, "the stack this stack pointer points into is laid out with that size"],
    );
    let error = failure_with(&[module("m", &[&constant("word", "i32", "1", "")])], &stack_of(0));
    assert_mentions(&error, &[message]);
}

// ---- Sections and linker-defined symbols --------------------------------------------------------

#[test]
fn two_sections_whose_members_are_interleaved_are_each_contiguous() {
    let a1 = constant("a1", "i32", "1", " attributes { section = \"a\" }");
    let b1 = constant("b1", "i32", "2", " attributes { section = \"b\" }");
    let plain = constant("plain", "i32", "3", "");
    let a2 = constant("a2", "i32", "4", " attributes { section = \"a\" }");
    let b2 = constant("b2", "i64", "5", " attributes { section = \"b\" }");
    let mut items = alloc::vec![MEMORY.to_string(), a1, b1, plain, a2, b2];
    for name in ["a1", "b1", "plain", "a2", "b2"] {
        items.push(address_of(&format!("{name}_at"), name));
    }
    let items = items.iter().map(String::as_str).collect::<Vec<_>>();
    let mut instance = emitted(&[module("m", &items)]).run();
    let at = |instance: &mut Instance, name: &str| one(instance.call(&format!("{name}_at"), &[]));
    // Unsectioned data first; then `a`; then `b`, which starts at a multiple of 8 (its `i64`).
    assert_eq!(at(&mut instance, "plain"), 1024);
    assert_eq!(at(&mut instance, "a1"), 1028);
    assert_eq!(at(&mut instance, "a2"), 1032);
    assert_eq!(at(&mut instance, "b1"), 1040);
    assert_eq!(at(&mut instance, "b2"), 1048);
}

#[test]
fn the_bounds_of_a_section_sum_its_entries() {
    let total = r#"builtin.function public extern("C") @total() -> i32 {
    %s = builtin.global_symbol @start : ptr<i32>;
    %e = builtin.global_symbol @stop : ptr<i32>;
    %si = mem.ptrtoint %s;
    %ei = mem.ptrtoint %e;
    %zero = arith.constant 0 : i32;
    cf.br ^head(%si, %zero : isize, i32);
^head(%i: isize, %acc: i32):
    %done = arith.cmpi uge %i, %ei;
    cf.cond_br %done ^exit, ^body : (i1);
^body:
    %q = mem.inttoptr %i : ptr<i32>;
    %v = mem.load %q;
    %acc2 = arith.add %acc, %v;
    %four = arith.constant 4 : isize;
    %i2 = arith.add %i, %four;
    cf.br ^head(%i2, %acc2 : isize, i32);
^exit:
    builtin.ret %acc : (i32);
} attributes { wasm.export_name = "total" };"#;
    let items = [
        constant("n1", "i32", "1", " attributes { section = \"nums\" }"),
        constant("other", "i32", "100", " attributes { section = \"other\" }"),
        constant("n2", "i32", "2", " attributes { section = \"nums\" }"),
        constant("n3", "i32", "4", " attributes { section = \"nums\" }"),
        String::from(
            "builtin.global_variable public @start : i32 attributes { link_name = \
             \"__start_nums\" };",
        ),
        String::from(
            "builtin.global_variable public @stop : i32 attributes { link_name = \"__stop_nums\" \
             };",
        ),
        String::from(total),
    ];
    let items = items.iter().map(String::as_str).collect::<Vec<_>>();
    let emitted = emitted(&[module("m", &items)]);
    // The bounds are addresses, not data: no segment for them.
    assert_eq!(emitted.wat().matches("(data ").count(), 4, "{}", emitted.wat());
    assert_eq!(one(emitted.run().call("total", &[])), 7);
}

#[test]
fn the_start_of_a_section_with_no_member_is_an_error_naming_the_declaration() {
    let error = failure(&[module(
        "m",
        &[
            MEMORY,
            "builtin.global_variable public @start : i32 attributes { link_name = \
             \"__start_empty\" };",
            &address_of("start_at", "start"),
        ],
    )]);
    assert_mentions(
        &error,
        &[
            "the section `empty` has no member, so `__start_empty` has no address",
            "this declares `__start_empty`, the start of the section `empty`, and no global \
             variable of the unit is placed in it",
        ],
    );
}

#[test]
fn the_data_end_and_the_heap_base_follow_the_data() {
    let word = constant("word", "i32", "1", "");
    let byte = constant("byte", "i8", "2", "");
    let mut instance = emitted(&[module(
        "m",
        &[
            MEMORY,
            &word,
            &byte,
            "builtin.global_variable public @end : i8 attributes { link_name = \"__data_end\" };",
            "builtin.global_variable public @heap : i8 attributes { link_name = \"__heap_base\" };",
            &address_of("end_at", "end"),
            &address_of("heap_at", "heap"),
        ],
    )])
    .run();
    assert_eq!(one(instance.call("end_at", &[])), 1029);
    assert_eq!(one(instance.call("heap_at", &[])), 1040);
}

#[test]
fn a_bump_allocator_over_the_heap_base_grows_the_memory_and_writes_at_its_top() {
    let alloc = r#"builtin.function public extern("wasm") @alloc(%n: i32) -> i32 {
    %p = wasm.global_get @next : i32;
    %e = wasm.add %p, %n;
    wasm.global_set @next, %e;
    %k = wasm.const 65535 : i32;
    %r = wasm.add %e, %k;
    %s = wasm.const 16 : i32;
    %need = wasm.shr_u %r, %s;
    %have = wasm.memory_size @__linear_memory;
    %d = wasm.sub %need, %have;
    %old = wasm.memory_grow @__linear_memory, %d;
    builtin.ret %p : (i32);
} attributes { wasm.export_name = "alloc" };"#;
    let poke = r#"builtin.function public extern("wasm") @poke(%a: i32, %v: i32) {
    wasm.store @__linear_memory, %a, %v;
    builtin.ret;
} attributes { wasm.export_name = "poke" };"#;
    let emitted = emitted(&[module(
        "m",
        &[
            MEMORY,
            &constant("word", "i32", "1", ""),
            "builtin.global_variable public @heap : i8 attributes { link_name = \"__heap_base\" };",
            "wasm.global private @next : mut i32 { %v = wasm.data_addr @heap; builtin.yield %v; };",
            alloc,
            poke,
            &peek("peek"),
        ],
    )]);
    let mut instance = emitted.run();
    assert_eq!(instance.memory().len(), 65_536);
    let base = one(instance.call("alloc", &[Val::i32(100_000)]));
    assert_eq!(base, 1040);
    let top = base as u32 + 100_000 - 4;
    assert_eq!(instance.memory().len(), 2 * 65_536);
    assert_eq!(
        instance.call("poke", &[Val::i32(top), Val::i32(42)]),
        Outcome::Values(Vec::new())
    );
    assert_eq!(one(instance.call("peek", &[Val::i32(top)])), 42);
    assert_eq!(one(instance.call("alloc", &[Val::i32(16)])), u64::from(base as u32 + 100_000));
}

#[test]
fn the_heap_base_is_above_a_fixed_segment_and_a_bump_allocation_leaves_the_segment_intact() {
    let alloc = r#"builtin.function public extern("wasm") @alloc(%n: i32) -> i32 {
    %p = wasm.global_get @next : i32;
    %e = wasm.add %p, %n;
    wasm.global_set @next, %e;
    builtin.ret %p : (i32);
} attributes { wasm.export_name = "alloc" };"#;
    let fill = r#"builtin.function public extern("wasm") @fill(%a: i32, %n: i32) {
    %v = wasm.const 0 : i32;
    wasm.memory_fill @__linear_memory, %a, %v, %n;
    builtin.ret;
} attributes { wasm.export_name = "fill" };"#;
    let emitted = emitted(&[module(
        "m",
        &[
            MEMORY,
            "builtin.segment <{ offset = #builtin.i64<2048>, data = #builtin.bytes<\"2a2a2a2a\"> \
             }>;",
            &constant("x", "i32", "7", ""),
            "builtin.global_variable public @end : i8 attributes { link_name = \"__data_end\" };",
            "builtin.global_variable public @heap : i8 attributes { link_name = \"__heap_base\" };",
            "wasm.global private @next : mut i32 { %v = wasm.data_addr @heap; builtin.yield %v; };",
            &address_of("end_at", "end"),
            &address_of("heap_at", "heap"),
            alloc,
            fill,
            &peek("peek"),
        ],
    )]);
    let mut instance = emitted.run();
    // The data ends after `@x`; the heap starts after the segment, at 2052 rounded up to 16.
    assert_eq!(one(instance.call("end_at", &[])), 1028);
    assert_eq!(one(instance.call("heap_at", &[])), 2064);
    // Zero the 2 KiB the allocator gives first (from 1040, they would cover the segment): the
    // segment is below them and keeps its bytes.
    let base = one(instance.call("alloc", &[Val::i32(2048)]));
    assert_eq!(base, 2064);
    assert_eq!(
        instance.call("fill", &[Val::i32(base as u32), Val::i32(2048)]),
        Outcome::Values(Vec::new())
    );
    assert_eq!(one(instance.call("peek", &[Val::i32(2048)])), 0x2a2a_2a2a);
}

// ---- Zero, passive and fixed data ---------------------------------------------------------------

#[test]
fn zeroed_data_has_no_segment_reads_zero_and_the_passive_segments_after_it_keep_their_indices() {
    let copy = r#"builtin.function public extern("wasm") @copy(%d: i32) -> i32 {
    %s = wasm.const 0 : i32;
    %n = wasm.const 4 : i32;
    wasm.memory_init @__linear_memory, @second, %d, %s, %n;
    wasm.data_drop @first;
    %v = wasm.load @__linear_memory, %d : i32;
    builtin.ret %v : (i32);
} attributes { wasm.export_name = "copy" };"#;
    let items = [
        String::from(MEMORY),
        String::from(
            "builtin.global_variable private @zeros : array<i8; 8> { %b = builtin.bytes \
             \"0000000000000000\" : array<i8; 8>; builtin.yield %b; };",
        ),
        constant("zero", "i32", "0", ""),
        constant("first", "i32", "17", ""),
        constant("second", "i32", "33", ""),
        loader("get_zero", "zero", " : i32", "i32"),
        address_of("zeros_at", "zeros"),
        String::from(copy),
    ];
    let items = items.iter().map(String::as_str).collect::<Vec<_>>();
    let emitted = emitted(&[module("m", &items)]);
    let text = emitted.wat();
    assert!(!text.contains("(data $m::zero"), "{text}");
    assert_mentions(
        &text,
        &[
            "(data $m::first (;0;) \"\\11\\00\\00\\00\")",
            "(data $m::second (;1;) \"!\\00\\00\\00\")",
            "memory.init $m::second",
            "data.drop $m::first",
        ],
    );
    let mut instance = emitted.run();
    assert_eq!(one(instance.call("get_zero", &[])), 0);
    assert_eq!(one(instance.call("zeros_at", &[])), 1024);
    assert_eq!(one(instance.call("copy", &[Val::i32(2048)])), 33);
}

#[test]
fn passive_data_is_copied_by_memory_init_and_emptied_by_data_drop() {
    let copy = r#"builtin.function public extern("wasm") @copy(%d: i32, %n: i32) -> i32 {
    %s = wasm.const 0 : i32;
    wasm.memory_init @__linear_memory, @bytes, %d, %s, %n;
    %v = wasm.load @__linear_memory, %d : i32;
    builtin.ret %v : (i32);
} attributes { wasm.export_name = "copy" };"#;
    let drop = r#"builtin.function public extern("wasm") @drop() {
    wasm.data_drop @bytes;
    builtin.ret;
} attributes { wasm.export_name = "drop" };"#;
    let bytes = "builtin.global_variable private @bytes : array<i8; 4> { %b = builtin.bytes \
                 \"0a0b0c0d\" : array<i8; 4>; builtin.yield %b; };";
    let mut instance = emitted(&[module("m", &[MEMORY, bytes, copy, drop])]).run();
    assert_eq!(one(instance.call("copy", &[Val::i32(4096), Val::i32(4)])), 0x0a0b_0c0d);
    assert_eq!(instance.call("drop", &[]), Outcome::Values(Vec::new()));
    // A dropped segment is empty: copying from it traps.
    assert_eq!(instance.call("copy", &[Val::i32(4096), Val::i32(4)]), Outcome::Trap);
}

#[test]
fn passive_data_whose_address_is_taken_is_an_error_naming_both_operations() {
    let both = r#"builtin.function public extern("wasm") @both(%d: i32) -> i32 {
    %s = wasm.const 0 : i32;
    %n = wasm.const 4 : i32;
    wasm.memory_init @__linear_memory, @bytes, %d, %s, %n;
    %a = wasm.data_addr @bytes;
    builtin.ret %a : (i32);
};"#;
    let error = failure(&[module("m", &[MEMORY, &constant("bytes", "i32", "1", ""), both])]);
    assert_mentions(
        &error,
        &[
            "the global variable `m::bytes` is both a passive data segment and data with an \
             address",
            "this takes the address of `m::bytes`, which a passive segment does not have",
            "this makes `m::bytes` a passive segment, copied by `memory.init`",
        ],
    );
    // An address taken in another global variable's initializer is the same error.
    let pointer = "builtin.global_variable private @pointer : ptr<i32> { %p = \
                   builtin.global_symbol @bytes : ptr<i32>; builtin.yield %p; };";
    let init = r#"builtin.function public extern("wasm") @init(%d: i32) {
    %s = wasm.const 0 : i32;
    %n = wasm.const 4 : i32;
    wasm.memory_init @__linear_memory, @bytes, %d, %s, %n;
    builtin.ret;
};"#;
    let error =
        failure(&[module("m", &[MEMORY, &constant("bytes", "i32", "1", ""), pointer, init])]);
    assert_mentions(&error, &["is both a passive data segment and data with an address"]);
}

#[test]
fn a_fixed_segment_high_in_memory_is_placed_and_the_memory_holds_it() {
    let segment =
        "builtin.segment <{ offset = #builtin.i64<200000>, data = #builtin.bytes<\"0000002a\"> }>;";
    let emitted = emitted(&[module("m", &[MEMORY, segment, &peek("peek")])]);
    let text = emitted.wat();
    assert_mentions(
        &text,
        &[
            "(memory $__linear_memory (;0;) 4)",
            "(data (;0;) (i32.const 200000) \"*\\00\\00\\00\")",
        ],
    );
    assert_eq!(one(emitted.run().call("peek", &[Val::i32(200_000)])), 42);
}

#[test]
fn an_empty_fixed_segment_has_no_data_segment_and_the_module_instantiates() {
    // Far beyond the memory the module needs: as an active segment it would trap at
    // instantiation.
    let segment =
        "builtin.segment <{ offset = #builtin.i64<1048576000>, data = #builtin.bytes<\"\"> }>;";
    let f = "builtin.function public extern(\"C\") @f() -> i32 {\n    %x = arith.constant 1 : \
             i32;\n    builtin.ret %x : (i32);\n} attributes { wasm.export_name = \"f\" };";
    let module_of_f = emitted(&[module("m", &[segment, f])]);
    let text = module_of_f.wat();
    assert!(!text.contains("(data"), "{text}");
    assert_eq!(one(module_of_f.run().call("f", &[])), 1);
    // Nor does it overlap anything: one at the address of the data is no error.
    let at_data = "builtin.segment <{ offset = #builtin.i64<1024>, data = #builtin.bytes<\"\"> }>;";
    let text = emitted(&[module("m", &[MEMORY, at_data, &constant("word", "i32", "1", "")])]).wat();
    assert_eq!(text.matches("(data ").count(), 1, "{text}");
}

#[test]
fn a_fixed_segment_that_overlaps_the_data_or_another_is_an_error_naming_it() {
    let data = constant("word", "i32", "1", "");
    let error = failure(&[module(
        "m",
        &[
            MEMORY,
            &data,
            "builtin.segment <{ offset = #builtin.i64<1026>, data = #builtin.bytes<\"ffff\"> }>;",
        ],
    )]);
    assert_mentions(
        &error,
        &[
            "a `builtin.segment` overlaps memory the layout uses",
            "this segment holds the bytes 1026..1028, and the global variables are laid out at \
             1024..1028",
        ],
    );
    let error = failure(&[module(
        "m",
        &[
            MEMORY,
            "builtin.segment <{ offset = #builtin.i64<16>, data = #builtin.bytes<\"ffffffff\"> }>;",
            "builtin.segment <{ offset = #builtin.i64<18>, data = #builtin.bytes<\"ffff\"> }>;",
        ],
    )]);
    assert_mentions(
        &error,
        &[
            "two `builtin.segment`s overlap",
            "this segment holds the bytes 18..20",
            "this one holds the bytes 16..20",
        ],
    );
    let error = failure_with(
        &[module(
            "m",
            &[
                &framed("framed"),
                "builtin.segment <{ offset = #builtin.i64<16>, data = #builtin.bytes<\"ff\"> }>;",
            ],
        )],
        &stack_of(65_536),
    );
    assert_mentions(
        &error,
        &["this segment holds the bytes 16..17, and the shadow stack is at 0..65536"],
    );
}

// ---- The memory ---------------------------------------------------------------------------------

#[test]
fn the_memory_has_the_pages_its_stack_and_data_need_and_is_exported() {
    // No stack: one page holds the data from 1024.
    let small = emitted(&[module("m", &[MEMORY, &constant("word", "i32", "1", "")])]).wat();
    assert_mentions(
        &small,
        &[
            "(memory $__linear_memory (;0;) 1)",
            "(export \"memory\" (memory $__linear_memory))",
        ],
    );
    // 100 KiB of data from 1024 needs two pages.
    let hex = "01".repeat(102_400);
    let big = format!(
        "builtin.global_variable private @big : array<i8; 102400> {{ %b = builtin.bytes \"{hex}\" \
         : array<i8; 102400>; builtin.yield %b; }};"
    );
    let text = emitted(&[module("m", &[MEMORY, &big])]).wat();
    assert_mentions(&text, &["(memory $__linear_memory (;0;) 2)"]);
    // The default stack of 1 MiB and a little data need 17 pages.
    let text =
        emitted(&[module("m", &[&framed("framed"), &constant("word", "i32", "1", "")])]).wat();
    assert_mentions(&text, &["(memory $__linear_memory (;0;) 17)"]);
    // An export named `memory` already: the memory is not exported under that name again.
    let taken = "builtin.function public extern(\"wasm\") @f() { builtin.ret; } attributes { \
                 wasm.export_name = \"memory\" };";
    let text = emitted(&[module("m", &[MEMORY, &constant("word", "i32", "1", ""), taken])]).wat();
    assert_mentions(&text, &["(export \"memory\" (func $m::f))"]);
    assert!(!text.contains("(export \"memory\" (memory"), "{text}");
}

#[test]
fn a_memory_the_unit_defines_keeps_its_limits_and_one_too_small_is_an_error_with_the_pages() {
    let memory = "wasm.memory public @__linear_memory : 1;";
    let text =
        emitted(&[module("m", &[memory, &constant("word", "i32", "5", ""), &peek("peek")])]).wat();
    assert_mentions(
        &text,
        &[
            "(memory $m::__linear_memory (;0;) 1)",
            "(data $m::word (;0;) (i32.const 1024) \"\\05\\00\\00\\00\")",
        ],
    );
    assert!(!text.contains("(export \"memory\""), "{text}");
    let hex = "01".repeat(70_000);
    let big = format!(
        "builtin.global_variable private @big : array<i8; 70000> {{ %b = builtin.bytes \"{hex}\" \
         : array<i8; 70000>; builtin.yield %b; }};"
    );
    let error = failure(&[module("m", &[memory, &big])]);
    assert_mentions(
        &error,
        &[
            "the memory @__linear_memory is too small for what the layout places in it: it needs \
             2 pages of 64 KiB, and its minimum is 1",
            "this memory has 1 pages at first, and the stack and data need 2",
        ],
    );
}

#[test]
fn a_unit_that_defines_its_own_stack_pointer_reserves_no_stack_and_places_data_from_1024() {
    let pointer = "wasm.global public @__stack_pointer : mut i32 { %v = wasm.const 4096 : i32; \
                   builtin.yield %v; };";
    let emitted = emitted(&[module(
        "m",
        &[
            pointer,
            &framed("framed"),
            &constant("word", "i32", "1", ""),
            &address_of("word_at", "word"),
        ],
    )]);
    let text = emitted.wat();
    assert_mentions(
        &text,
        &[
            "(global $m::__stack_pointer (;0;) (mut i32) i32.const 4096)",
            "(memory $__linear_memory (;0;) 1)",
        ],
    );
    let mut instance = emitted.run();
    assert_eq!(one(instance.call("word_at", &[])), 1024);
    assert_eq!(one(instance.call("framed", &[Val::i32(3)])), 3);
}

// ---- The table ----------------------------------------------------------------------------------

/// A Wasm-form function `@name` of `(i32) -> i32` that adds `k`.
fn adder(name: &str, k: i32) -> String {
    format!(
        "builtin.function public extern(\"wasm\") @{name}(%x: i32) -> i32 {{\n    %k = wasm.const \
         {k} : i32;\n    %y = wasm.add %x, %k;\n    builtin.ret %y : (i32);\n}};"
    )
}

/// A Wasm-form function `@name`, exported, that calls slot `%i` of the table on `%x`.
fn dispatch(name: &str) -> String {
    format!(
        "builtin.function public extern(\"wasm\") @{name}(%i: i32, %x: i32) -> i32 {{\n    %r = \
         wasm.call_indirect @__indirect_function_table, %i(%x) : extern(\"wasm\") (i32) -> \
         i32;\n    builtin.ret %r : (i32);\n}} attributes {{ wasm.export_name = \"{name}\" }};"
    )
}

#[test]
fn functions_named_by_func_index_get_slots_from_1_once_each_and_slot_0_traps() {
    let slots = r#"builtin.function public extern("wasm") @slots() -> (i32, i32, i32, i32) {
    %a = wasm.func_index @one;
    %b = wasm.func_index @two;
    %c = wasm.func_index @three;
    %d = wasm.func_index @two;
    builtin.ret %a, %b, %c, %d : (i32, i32, i32, i32);
} attributes { wasm.export_name = "slots" };"#;
    let emitted = emitted(&[module(
        "m",
        &[
            TABLE,
            &adder("one", 1),
            &adder("two", 2),
            &adder("three", 3),
            slots,
            &dispatch("call"),
        ],
    )]);
    let text = emitted.wat();
    assert_mentions(
        &text,
        &[
            "(table $__indirect_function_table (;0;) 4 4 funcref)",
            "(elem (;0;) (i32.const 1) func $m::one $m::two $m::three)",
        ],
    );
    let mut instance = emitted.run();
    assert_eq!(
        instance.call("slots", &[]),
        Outcome::Values(alloc::vec![Val::i32(1), Val::i32(2), Val::i32(3), Val::i32(2)])
    );
    assert_eq!(one(instance.call("call", &[Val::i32(3), Val::i32(10)])), 13);
    assert_eq!(instance.call("call", &[Val::i32(0), Val::i32(10)]), Outcome::Trap);
    assert_eq!(instance.call("call", &[Val::i32(4), Val::i32(10)]), Outcome::Trap);
}

#[test]
fn a_func_index_with_a_function_table_the_unit_defines_is_an_error_naming_the_table() {
    let table = "wasm.table public @__indirect_function_table : 2 x !wasm.funcref;";
    let index = "builtin.function public extern(\"wasm\") @f() -> i32 {\n    %i = wasm.func_index \
                 @one;\n    builtin.ret %i : (i32);\n};";
    let error = failure(&[module("m", &[table, &adder("one", 1), index])]);
    assert_mentions(
        &error,
        &[
            "the backend cannot place functions in a function table the module lays out itself",
            "this is `__indirect_function_table`, which the module lays out itself, so the layout \
             cannot give its slots",
            "this takes the slot of `m::one`",
        ],
    );
}

/// A module whose `wasm.elem` puts `@two` at `slot` of the function table the lowering declares,
/// with a function that takes the slot of `@one`.
fn elem_into_the_lowered_table(slot: u32) -> String {
    module(
        "m",
        &[
            TABLE,
            &adder("one", 1),
            &adder("two", 2),
            &format!("wasm.elem private @init : @__indirect_function_table at {slot} [@two];"),
            "builtin.function public extern(\"wasm\") @call1(%x: i32) -> i32 {\n    %k = \
             wasm.func_index @one;\n    %r = wasm.call_indirect @__indirect_function_table, \
             %k(%x) : extern(\"wasm\") (i32) -> i32;\n    builtin.ret %r : (i32);\n} attributes { \
             wasm.export_name = \"call1\" };",
        ],
    )
}

#[test]
fn a_wasm_elem_into_the_function_table_the_layout_defines_is_an_error_naming_it() {
    // Inside the slots the layout gives (where its own segment would overwrite the entry) and
    // past them (where the segment would fail at instantiation).
    for slot in [1, 5] {
        let error = failure(&[elem_into_the_lowered_table(slot)]);
        assert_mentions(
            &error,
            &[
                "a `wasm.elem` cannot fill the function table the backend lays out",
                "this fills `__indirect_function_table`, whose size and slots the backend lays \
                 out itself; a unit that fills that table must define it",
                "this declares `__indirect_function_table`, which the layout defines",
            ],
        );
    }
}

#[test]
fn a_table_the_unit_defines_is_filled_by_its_own_wasm_elem_and_called_through() {
    let call = "builtin.function public extern(\"wasm\") @call(%i: i32, %x: i32) -> i32 {\n    %r \
                = wasm.call_indirect @t, %i(%x) : extern(\"wasm\") (i32) -> i32;\n    builtin.ret \
                %r : (i32);\n} attributes { wasm.export_name = \"call\" };";
    let emitted = emitted(&[module(
        "m",
        &[
            "wasm.table public @t : 4 x !wasm.funcref;",
            &adder("one", 1),
            &adder("two", 2),
            "wasm.elem private @init : @t at 2 [@two, @one];",
            call,
        ],
    )]);
    assert_mentions(
        &emitted.wat(),
        &[
            "(table $m::t (;0;) 4 funcref)",
            "(elem (;0;) (i32.const 2) func $m::two $m::one)",
        ],
    );
    let mut instance = emitted.run();
    assert_eq!(one(instance.call("call", &[Val::i32(2), Val::i32(10)])), 12);
    assert_eq!(one(instance.call("call", &[Val::i32(3), Val::i32(10)])), 11);
    assert_eq!(instance.call("call", &[Val::i32(1), Val::i32(10)]), Outcome::Trap);
}

// ---- Two modules --------------------------------------------------------------------------------

#[test]
fn two_modules_with_data_and_frames_share_one_memory_one_stack_pointer_and_one_table() {
    let module_with = |name: &str| {
        let get = format!(
            "builtin.function public extern(\"C\") @get_{name}(%x: i32) -> i32 {{\n    %slot = \
             mem.alloca i32;\n    mem.store %x, %slot;\n    %p = builtin.global_symbol @value : \
             ptr<i32>;\n    %v = mem.load %p;\n    %w = mem.load %slot;\n    %f = \
             builtin.function_ref @add : (i32, i32) -> i32;\n    %r = builtin.call_indirect \
             %f(%v, %w) : extern(\"C\") (i32, i32) -> i32;\n    builtin.ret %r : (i32);\n}} \
             attributes {{ wasm.export_name = \"get_{name}\" }};"
        );
        let add = "builtin.function private extern(\"C\") @add(%a: i32, %b: i32) -> i32 {\n    %s \
                   = arith.add %a, %b;\n    builtin.ret %s : (i32);\n};";
        let value = constant("value", "i32", if name == "a" { "100" } else { "200" }, "");
        module(name, &[&value, add, &get])
    };
    let emitted = emitted(&[module_with("a"), module_with("b")]);
    let text = emitted.wat();
    assert_eq!(text.matches("\n  (memory ").count(), 1, "{text}");
    assert_eq!(text.matches("\n  (global ").count(), 1, "{text}");
    assert_eq!(text.matches("\n  (table ").count(), 1, "{text}");
    assert!(!text.contains("(import"), "{text}");
    assert_mentions(&text, &["(elem (;0;) (i32.const 1) func $a::add $b::add)"]);
    let mut instance = emitted.run();
    assert_eq!(one(instance.call("get_a", &[Val::i32(1)])), 101);
    assert_eq!(one(instance.call("get_b", &[Val::i32(2)])), 202);
}

// ---- Errors -------------------------------------------------------------------------------------

#[test]
fn an_initializer_that_does_not_fold_is_an_error_naming_the_operation_and_the_global() {
    let target = constant("target", "i32", "1", "");
    let integer = "builtin.global_variable private @integer : isize { %p = builtin.global_symbol \
                   @target : ptr<i32>; %i = mem.ptrtoint %p; builtin.yield %i; };";
    let error = failure(&[module("m", &[&target, integer])]);
    assert_mentions(
        &error,
        &[
            "the initializer of the global variable `@integer` is not a constant the backend can \
             place",
            "`mem.ptrtoint` does not fold to a constant",
            "the global variable it initializes",
        ],
    );
}

#[test]
fn a_value_that_does_not_fit_its_global_is_an_error_naming_the_global() {
    // An `isize` constant is 64 bits wide; the global is 4 bytes.
    let error = failure(&[module("m", &[&constant("big", "isize", "1099511627776", "")])]);
    assert_mentions(
        &error,
        &[
            "the value of the global variable `@big` does not fit its type",
            "its initializer gives 1099511627776, and its type `isize` is 4 bytes",
        ],
    );
    // One that fits is written as its low bytes.
    let text = emitted(&[module("m", &[&constant("small", "isize", "-2", "")])]).wat();
    assert_mentions(&text, &["\"\\fe\\ff\\ff\\ff\""]);
}

#[test]
fn a_stack_pointer_or_a_function_table_declared_with_another_type_is_an_error() {
    let pointer =
        "wasm.global private declaration @sp : i64 attributes { link_name = \"__stack_pointer\" };";
    let error = failure(&[module("m", &[pointer])]);
    assert_mentions(
        &error,
        &[
            "the stack pointer `__stack_pointer` is a `mut i32`",
            "this declares `__stack_pointer`, which the layout defines, with another type",
        ],
    );
    let table = "wasm.table private declaration @t : 0 x !wasm.externref attributes { link_name = \
                 \"__indirect_function_table\" };";
    let error = failure(&[module("m", &[table])]);
    assert_mentions(
        &error,
        &[
            "the function table `__indirect_function_table` is a table of `funcref`",
            "with another element type",
        ],
    );
}

#[test]
fn data_that_fits_4_gib_with_no_address_left_for_the_heap_base_is_an_error_naming_the_last_global()
{
    // The `i64` ends at 4294967288 and the `i128` exactly at 4 GiB: the data fits either way, and
    // the heap base, rounded up to 16, is 4294967296, which no 32-bit address holds.
    for (ty, end) in [("i64", 4_294_967_288u64), ("i128", 4_294_967_296)] {
        let data = constant("wide", ty, "1", "");
        let error =
            failure_with(&[module("m", &[&framed("framed"), &data])], &stack_of(0xffff_fff0));
        assert_mentions(
            &error,
            &[
                &format!(
                    "the heap base does not fit a 32-bit linear memory: the unit's static data \
                     ends at {end}, and the heap base, rounded up to 16, would be 4294967296, \
                     beyond the last address 4294967295"
                ),
                &format!(
                    "this global variable is placed last and ends at {end}, leaving no address \
                     for the heap base"
                ),
            ],
        );
    }
}

#[test]
fn data_beyond_4_gib_is_an_error_naming_the_global_variable_that_does_not_fit() {
    let wide = constant("wide", "i128", "1", "");
    let word = constant("word", "i32", "1", "");
    let error =
        failure_with(&[module("m", &[&framed("framed"), &wide, &word])], &stack_of(0xffff_fff0));
    assert_mentions(
        &error,
        &[
            "the data of the unit does not fit the 4 GiB of a 32-bit linear memory",
            "this global variable would end at 4294967300",
        ],
    );
}
