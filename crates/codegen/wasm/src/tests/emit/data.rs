//! Global variables whose initializers are structs and arrays of constants, addresses and
//! functions, as the `mem` dialect's evaluator places them: written in generic IR, compiled
//! through the compile harness with the backend `wasm` (whose data layout states `fn = 32`), and
//! run under `wasmi` with fuel. What the emitted module holds is checked as text and as the bytes
//! of linear memory.

use alloc::{format, string::String, vec, vec::Vec};

use runic_compile::{
    Compilation, CompilationRequest, Compiler, EmitRequest, EmitTo, Emitted, Input, IrFrontend,
    OptLevel,
};
use runic_ir::{
    dialects::builtin::Module,
    parse::{self, ParserConfig},
    testing::text::describe,
};

use crate::{
    backend::{TRIPLE, WasmBackend},
    emit::{EmitOptions, emit_module},
    tests::{
        context, error_of,
        eval::{Outcome, Val},
        exec::Instance,
        mentions,
    },
    text::print,
};

// ---- Helpers ------------------------------------------------------------------------------------

/// The program `text` compiled for `wasm32-unknown-unknown` at the optimization level none: the
/// binary module, or the error of the compilation.
fn compile(text: &str) -> Result<Vec<u8>, String> {
    let compiler = Compiler::builder()
        .frontend(IrFrontend)
        .backend(WasmBackend)
        .build()
        .expect("the IR frontend and the Wasm backend form a valid compiler");
    let mut request = CompilationRequest::new();
    request.name = Some("program".into());
    request.inputs = vec![Input::memory("program.runic", text)];
    request.target = Some(TRIPLE.parse().unwrap());
    request.opt_level = OptLevel::None;
    request.emit.push(EmitRequest {
        checkpoint: "wasm.module".into(),
        to: EmitTo::Memory,
    });
    let compilation: Compilation = compiler.compile(request, &mut ());
    if !compilation.succeeded() {
        return Err(compilation.error().map(describe).unwrap_or_default());
    }
    Ok(compilation
        .emitted()
        .iter()
        .find(|emitted| matches!(emitted, Emitted::Memory { .. }))
        .and_then(|emitted| emitted.bytes())
        .expect("the module is delivered to memory")
        .to_vec())
}

/// A compiled program.
struct Program {
    bytes: Vec<u8>,
}

impl Program {
    /// The module as text.
    fn wat(&self) -> String {
        print(&self.bytes).unwrap_or_else(|report| panic!("{report}"))
    }

    /// The module instantiated with no imports.
    fn run(&self) -> Instance {
        Instance::new(&self.bytes)
    }
}

/// The program `text`, compiled.
#[track_caller]
fn compiled(text: &str) -> Program {
    let bytes = compile(text).unwrap_or_else(|error| panic!("{error}"));
    Program { bytes }
}

/// The error of compiling the program `text`.
#[track_caller]
fn failure(text: &str) -> String {
    match compile(text) {
        Ok(_) => panic!("the program compiled:\n{text}"),
        Err(error) => error,
    }
}

/// Asserts that `haystack` holds every one of `needles`.
#[track_caller]
fn assert_mentions(haystack: &str, needles: &[&str]) {
    for needle in needles {
        assert!(mentions(haystack, needle), "no `{needle}` in:\n{haystack}");
    }
}

/// The values of `outcome`, as bits.
#[track_caller]
fn values(outcome: Outcome) -> Vec<u64> {
    match outcome {
        Outcome::Values(values) => values.iter().map(|value| value.bits).collect(),
        other => panic!("expected values, got {other:?}"),
    }
}

/// The one value of `outcome`, as bits.
#[track_caller]
fn one(outcome: Outcome) -> u64 {
    match values(outcome).as_slice() {
        [value] => *value,
        other => panic!("expected one value, got {other:?}"),
    }
}

/// The `len` bytes of the memory of `instance` at `address`.
fn bytes_at(instance: &Instance, address: u64, len: usize) -> Vec<u8> {
    let start = address as usize;
    instance.memory()[start..start + len].to_vec()
}

/// A generic function `@name`, exported as `name`, that returns the address of `@symbol` of type
/// `ty` as an `isize`.
fn address_of(name: &str, symbol: &str, ty: &str) -> String {
    format!(
        "builtin.function public extern(\"C\") @{name}() -> isize {{\n    %p = \
         builtin.global_symbol @{symbol} : ptr<{ty}>;\n    %i = mem.ptrtoint %p;\n    builtin.ret \
         %i : (isize);\n}} attributes {{ wasm.export_name = \"{name}\" }};"
    )
}

/// A module `@m` of `items`, relying on the data layout of the target.
fn module(items: &[&str]) -> String {
    format!("builtin.module public @m {{\n{}\n}};", items.join("\n"))
}

// ---- Structs and arrays -------------------------------------------------------------------------

#[test]
fn a_struct_global_is_read_back_field_by_field() {
    let global = "builtin.global_variable private @s : struct<i8, i32, i64, f64> {
    %a = arith.constant -5 : i8;
    %b = arith.constant 100000 : i32;
    %c = arith.constant -7000000000 : i64;
    %d = arith.constantf 2.5 : f64;
    %s = builtin.aggregate %a, %b, %c, %d : struct<i8, i32, i64, f64>;
    builtin.yield %s;
};";
    let fields = r#"builtin.function public extern("wasm") @fields() -> (i32, i32, i64, f64) {
    %p = builtin.global_symbol @s : ptr<struct<i8, i32, i64, f64>>;
    %p0 = mem.gep %p, [0];
    %a = mem.load %p0;
    %a32 = arith.sext %a <{ ty = i32 }>;
    %p1 = mem.gep %p, [1];
    %b = mem.load %p1;
    %p2 = mem.gep %p, [2];
    %c = mem.load %p2;
    %p3 = mem.gep %p, [3];
    %d = mem.load %p3;
    builtin.ret %a32, %b, %c, %d : (i32, i32, i64, f64);
} attributes { wasm.export_name = "fields" };"#;
    let program = compiled(&module(&[
        global,
        fields,
        &address_of("s_at", "s", "struct<i8, i32, i64, f64>"),
    ]));
    let mut instance = program.run();
    assert_eq!(
        values(instance.call("fields", &[])),
        vec![u64::from(-5i32 as u32), 100_000, -7_000_000_000i64 as u64, 2.5f64.to_bits()]
    );
    // The fields at 0, 4, 8 and 16, the padding after the `i8` zero.
    let at = one(instance.call("s_at", &[]));
    let mut expected = vec![0xfb, 0, 0, 0];
    expected.extend_from_slice(&100_000i32.to_le_bytes());
    expected.extend_from_slice(&(-7_000_000_000i64).to_le_bytes());
    expected.extend_from_slice(&2.5f64.to_le_bytes());
    assert_eq!(bytes_at(&instance, at, 24), expected);
}

#[test]
fn a_nested_struct_and_an_array_of_structs_are_summed_by_a_loop() {
    let mut items = String::new();
    for k in 1..=3 {
        items.push_str(&format!(
            "    %x{k} = arith.constant {k} : i32;\n    %y{k} = arith.constant {} : i16;\n    \
             %z{k} = arith.constant {} : i64;\n    %i{k} = builtin.aggregate %y{k}, %z{k} : \
             struct<i16, i64>;\n    %e{k} = builtin.aggregate %x{k}, %i{k} : struct<i32, \
             struct<i16, i64>>;\n",
            10 * k,
            100 * k
        ));
    }
    let global = format!(
        "builtin.global_variable private @pairs : array<struct<i32, struct<i16, i64>>; 3> \
         {{\n{items}    %a = builtin.aggregate %e1, %e2, %e3 : array<struct<i32, struct<i16, \
         i64>>; 3>;\n    builtin.yield %a;\n}};"
    );
    let sum = r#"builtin.function public extern("C") @sum() -> i64 {
    %base = builtin.global_symbol @pairs : ptr<array<struct<i32, struct<i16, i64>>; 3>>;
    %p = mem.ptrcast %base : ptr<struct<i32, struct<i16, i64>>>;
    %first = arith.constant 0 : isize;
    %n = arith.constant 3 : isize;
    %zero = arith.constant 0 : i64;
    cf.br ^head(%first, %zero : isize, i64);
^head(%i: isize, %acc: i64):
    %done = arith.cmpi uge %i, %n;
    cf.cond_br %done ^exit, ^body : (i1);
^body:
    %px = mem.gep %p, %i, [0];
    %x = mem.load %px;
    %py = mem.gep %p, %i, [1, 0];
    %y = mem.load %py;
    %pz = mem.gep %p, %i, [1, 1];
    %z = mem.load %pz;
    %x64 = arith.sext %x <{ ty = i64 }>;
    %y64 = arith.sext %y <{ ty = i64 }>;
    %s1 = arith.add %acc, %x64;
    %s2 = arith.add %s1, %y64;
    %s3 = arith.add %s2, %z;
    %one = arith.constant 1 : isize;
    %i2 = arith.add %i, %one;
    cf.br ^head(%i2, %s3 : isize, i64);
^exit:
    builtin.ret %acc : (i64);
} attributes { wasm.export_name = "sum" };"#;
    let program = compiled(&module(&[
        &global,
        sum,
        &address_of("pairs_at", "pairs", "array<struct<i32, struct<i16, i64>>; 3>"),
    ]));
    let mut instance = program.run();
    assert_eq!(one(instance.call("sum", &[])), 666);
    // Each element is 24 bytes: the `i32` at 0, the inner struct at 8 (its `i16` at 8, its `i64`
    // at 16).
    let at = one(instance.call("pairs_at", &[]));
    let element = |k: u8| {
        let mut bytes = vec![k, 0, 0, 0, 0, 0, 0, 0, 10 * k, 0, 0, 0, 0, 0, 0, 0];
        bytes.extend_from_slice(&(100 * u64::from(k)).to_le_bytes());
        bytes
    };
    assert_eq!(bytes_at(&instance, at, 72), [element(1), element(2), element(3)].concat());
}

/// The four functions of the table of `an_array_of_names_numbers_and_functions_is_walked_and_called`
/// and what each does to its argument.
const OPERATIONS: [(&str, &str); 4] = [
    ("twice", "%r = arith.add %x, %x;"),
    ("square", "%r = arith.mul %x, %x;"),
    ("negate", "%z = arith.constant 0 : i64;\n    %r = arith.sub %z, %x;"),
    ("next", "%one = arith.constant 1 : i64;\n    %r = arith.add %x, %one;"),
];

#[test]
fn an_array_of_names_numbers_and_functions_is_walked_and_called() {
    let entry = "struct<ptr<i8>, i32, (i64) -> i64>";
    let mut items: Vec<String> = Vec::new();
    let mut table = String::new();
    for (k, (name, body)) in OPERATIONS.iter().enumerate() {
        items.push(format!(
            "builtin.function public extern(\"C\") @{name}(%x: i64) -> i64 {{\n    {body}\n    \
             builtin.ret %r : (i64);\n}};"
        ));
        // Each name is two bytes, its first letter and a 0; `builtin.bytes` is written most
        // significant byte first.
        items.push(format!(
            "builtin.global_variable private @name_{name} : array<i8; 2> {{ %b = builtin.bytes \
             \"00{:02x}\" : array<i8; 2>; builtin.yield %b; }};",
            name.as_bytes()[0]
        ));
        table.push_str(&format!(
            "    %g{k} = builtin.global_symbol @name_{name} : ptr<array<i8; 2>>;\n    %n{k} = \
             mem.ptrcast %g{k} : ptr<i8>;\n    %c{k} = arith.constant {} : i32;\n    %f{k} = \
             builtin.function_ref @{name} : (i64) -> i64;\n    %e{k} = builtin.aggregate %n{k}, \
             %c{k}, %f{k} : {entry};\n",
            100 + k
        ));
    }
    items.push(format!(
        "builtin.global_variable private @table : array<{entry}; 4> {{\n{table}    %t = \
         builtin.aggregate %e0, %e1, %e2, %e3 : array<{entry}; 4>;\n    builtin.yield %t;\n}};"
    ));
    items.push(format!(
        r#"builtin.function public extern("wasm") @call(%i: isize, %x: i64) -> (i32, i32, i64) {{
    %base = builtin.global_symbol @table : ptr<array<{entry}; 4>>;
    %p = mem.ptrcast %base : ptr<{entry}>;
    %pn = mem.gep %p, %i, [0];
    %n = mem.load %pn;
    %letter = mem.load %n;
    %l = arith.zext %letter <{{ ty = i32 }}>;
    %pc = mem.gep %p, %i, [1];
    %c = mem.load %pc;
    %pf = mem.gep %p, %i, [2];
    %f = mem.load %pf;
    %r = builtin.call_indirect %f(%x) : extern("C") (i64) -> i64;
    builtin.ret %l, %c, %r : (i32, i32, i64);
}} attributes {{ wasm.export_name = "call" }};"#
    ));
    let items = items.iter().map(String::as_str).collect::<Vec<_>>();
    let program = compiled(&module(&items));
    let text = program.wat();
    // A slot for each function, from 1, in the order the initializer names them.
    assert_mentions(
        &text,
        &[
            "(table $__indirect_function_table (;0;) 5 5 funcref)",
            "(elem (;0;) (i32.const 1) func $m::twice $m::square $m::negate $m::next)",
        ],
    );
    let mut instance = program.run();
    let expected = [(b't', 7i64, 14i64), (b's', 7, 49), (b'n', 7, -7), (b'n', -1, 0)];
    for (k, (letter, x, r)) in expected.into_iter().enumerate() {
        assert_eq!(
            values(instance.call("call", &[Val::i32(k as u32), Val::i64(x as u64)])),
            vec![u64::from(letter), 100 + k as u64, r as u64],
            "entry {k}"
        );
    }
}

// ---- Addresses ----------------------------------------------------------------------------------

#[test]
fn a_tagged_pointer_in_an_isize_field_is_untagged_and_followed() {
    let items = [
        "builtin.global_variable private @target : i32 { %v = arith.constant 77 : i32; \
         builtin.yield %v; };",
        "builtin.global_variable private @tagged : struct<i8, isize> {
    %k = arith.constant 1 : i8;
    %p = builtin.global_symbol @target : ptr<i32>;
    %b = mem.ptrcast %p : ptr<i8>;
    %three = arith.constant 3 : isize;
    %q = mem.gep %b, %three;
    %i = mem.ptrtoint %q;
    %s = builtin.aggregate %k, %i : struct<i8, isize>;
    builtin.yield %s;
};",
        r#"builtin.function public extern("wasm") @follow() -> (isize, i32) {
    %g = builtin.global_symbol @tagged : ptr<struct<i8, isize>>;
    %f = mem.gep %g, [1];
    %i = mem.load %f;
    %three = arith.constant 3 : isize;
    %tag = arith.band %i, %three;
    %mask = arith.constant -4 : isize;
    %u = arith.band %i, %mask;
    %p = mem.inttoptr %u : ptr<i32>;
    %v = mem.load %p;
    builtin.ret %tag, %v : (isize, i32);
} attributes { wasm.export_name = "follow" };"#,
        &address_of("target_at", "target", "i32"),
        &address_of("tagged_at", "tagged", "struct<i8, isize>"),
    ];
    let items = items.iter().map(|item| item.as_ref()).collect::<Vec<&str>>();
    let mut instance = compiled(&module(&items)).run();
    assert_eq!(values(instance.call("follow", &[])), vec![3, 77]);
    let target = one(instance.call("target_at", &[]));
    let tagged = one(instance.call("tagged_at", &[]));
    let mut expected = vec![1, 0, 0, 0];
    expected.extend_from_slice(&(target as u32 + 3).to_le_bytes());
    assert_eq!(bytes_at(&instance, tagged, 8), expected);
}

#[test]
fn a_pointer_into_the_middle_of_another_global_is_followed() {
    let mut elements = String::new();
    for k in 0..4 {
        elements.push_str(&format!(
            "    %a{k} = arith.constant {k} : i32;\n    %b{k} = arith.constant {} : i64;\n    \
             %e{k} = builtin.aggregate %a{k}, %b{k} : struct<i32, i64>;\n",
            1000 + k
        ));
    }
    let table = format!(
        "builtin.global_variable private @table : array<struct<i32, i64>; 4> {{\n{elements}    %t \
         = builtin.aggregate %e0, %e1, %e2, %e3 : array<struct<i32, i64>; 4>;\n    builtin.yield \
         %t;\n}};"
    );
    let middle = "builtin.global_variable private @middle : ptr<i64> {
    %g = builtin.global_symbol @table : ptr<array<struct<i32, i64>; 4>>;
    %p = mem.gep %g, [2, 1];
    builtin.yield %p;
};";
    let follow = r#"builtin.function public extern("C") @follow() -> i64 {
    %m = builtin.global_symbol @middle : ptr<ptr<i64>>;
    %p = mem.load %m;
    %v = mem.load %p;
    builtin.ret %v : (i64);
} attributes { wasm.export_name = "follow" };"#;
    let program = compiled(&module(&[
        &table,
        middle,
        follow,
        &address_of("table_at", "table", "array<struct<i32, i64>; 4>"),
        &address_of("middle_at", "middle", "ptr<i64>"),
    ]));
    let mut instance = program.run();
    assert_eq!(one(instance.call("follow", &[])), 1002);
    // Element 2 is at 32, its `i64` at 8 more.
    let table = one(instance.call("table_at", &[]));
    let middle = one(instance.call("middle_at", &[]));
    assert_eq!(bytes_at(&instance, middle, 4), (table as u32 + 40).to_le_bytes());
}

// ---- Zero ---------------------------------------------------------------------------------------

#[test]
fn a_zeroed_megabyte_has_no_segment_and_the_memory_covers_it() {
    let items = [
        "builtin.global_variable private @big : array<i64; 131072> { %z = builtin.zero : \
         array<i64; 131072>; builtin.yield %z; };",
        "builtin.global_variable private @after : i32 { %v = arith.constant 9 : i32; \
         builtin.yield %v; };",
        r#"builtin.function public extern("wasm") @ends() -> (i64, i64) {
    %g = builtin.global_symbol @big : ptr<array<i64; 131072>>;
    %p = mem.ptrcast %g : ptr<i64>;
    %first = mem.load %p;
    %n = arith.constant 131071 : isize;
    %q = mem.gep %p, %n;
    %last = mem.load %q;
    builtin.ret %first, %last : (i64, i64);
} attributes { wasm.export_name = "ends" };"#,
        r#"builtin.function public extern("C") @store_last(%v: i64) -> i64 {
    %g = builtin.global_symbol @big : ptr<array<i64; 131072>>;
    %q = mem.gep %g, [131071];
    mem.store %v, %q;
    %r = mem.load %q;
    builtin.ret %r : (i64);
} attributes { wasm.export_name = "store_last" };"#,
        &address_of("big_at", "big", "array<i64; 131072>"),
        &address_of("after_at", "after", "i32"),
    ];
    let items = items.iter().map(|item| item.as_ref()).collect::<Vec<&str>>();
    let program = compiled(&module(&items));
    let text = program.wat();
    // No stack (no frame): the data starts at 1024, `@big` takes 1 MiB, `@after` follows it, and
    // the heap base at 1049616 needs 17 pages.
    assert!(!text.contains("(data $m::big"), "{text}");
    assert_eq!(text.matches("(data ").count(), 1, "{text}");
    assert_mentions(
        &text,
        &[
            "(memory $__linear_memory (;0;) 17)",
            "(data $m::after (;0;) (i32.const 1049600) \"\\09\\00\\00\\00\")",
        ],
    );
    let mut instance = program.run();
    assert_eq!(one(instance.call("big_at", &[])), 1024);
    assert_eq!(one(instance.call("after_at", &[])), 1_049_600);
    assert_eq!(values(instance.call("ends", &[])), vec![0, 0]);
    assert_eq!(one(instance.call("store_last", &[Val::i64(42)])), 42);
    assert_eq!(values(instance.call("ends", &[])), vec![0, 42]);
}

// ---- Relocations --------------------------------------------------------------------------------

#[test]
fn a_global_of_only_relocations_holds_addresses_and_a_table_slot() {
    let items = [
        "builtin.global_variable private @x : i32 { %v = arith.constant 5 : i32; builtin.yield \
         %v; };",
        "builtin.global_variable private @y : i32 { %v = arith.constant 6 : i32; builtin.yield \
         %v; };",
        "builtin.function public extern(\"C\") @twice(%x: i64) -> i64 {\n    %r = arith.add %x, \
         %x;\n    builtin.ret %r : (i64);\n};",
        "builtin.global_variable private @refs : struct<ptr<i32>, (i64) -> i64, ptr<i32>> {
    %x = builtin.global_symbol @x : ptr<i32>;
    %f = builtin.function_ref @twice : (i64) -> i64;
    %y = builtin.global_symbol @y : ptr<i32>;
    %s = builtin.aggregate %x, %f, %y : struct<ptr<i32>, (i64) -> i64, ptr<i32>>;
    builtin.yield %s;
};",
        r#"builtin.function public extern("wasm") @use(%a: i64) -> (i32, i64, i32) {
    %g = builtin.global_symbol @refs : ptr<struct<ptr<i32>, (i64) -> i64, ptr<i32>>>;
    %p0 = mem.gep %g, [0];
    %x = mem.load %p0;
    %vx = mem.load %x;
    %p1 = mem.gep %g, [1];
    %f = mem.load %p1;
    %r = builtin.call_indirect %f(%a) : extern("C") (i64) -> i64;
    %p2 = mem.gep %g, [2];
    %y = mem.load %p2;
    %vy = mem.load %y;
    builtin.ret %vx, %r, %vy : (i32, i64, i32);
} attributes { wasm.export_name = "use" };"#,
    ];
    let program = compiled(&module(&items));
    let text = program.wat();
    // `@x` at 1024, `@y` at 1028, `@refs` at 1032: the two addresses around the slot 1.
    assert_mentions(
        &text,
        &[
            "(data $m::refs (;2;) (i32.const 1032) \
             \"\\00\\04\\00\\00\\01\\00\\00\\00\\04\\04\\00\\00\")",
            "(elem (;0;) (i32.const 1) func $m::twice)",
        ],
    );
    let mut instance = program.run();
    assert_eq!(values(instance.call("use", &[Val::i64(21)])), vec![5, 42, 6]);
}

#[test]
fn globals_of_two_modules_point_at_each_other() {
    // Each module declares the other's global by its link name; each declaration links to the
    // definition, whose address the initializer takes.
    let world = r#"builtin.world {
    builtin.module public @a {
        builtin.global_variable public @pb : ptr attributes { link_name = "b::pb" };
        builtin.global_variable public @pa : ptr {
            %g = builtin.global_symbol @pb : ptr<ptr>;
            %p = mem.ptrcast %g : ptr;
            builtin.yield %p;
        };
        builtin.function public extern("wasm") @check() -> (isize, isize, isize, isize) {
            %ga = builtin.global_symbol @pa : ptr<ptr>;
            %gb = builtin.global_symbol @pb : ptr<ptr>;
            %va = mem.load %ga;
            %vb = mem.load %gb;
            %ia = mem.ptrtoint %ga;
            %ib = mem.ptrtoint %gb;
            %xa = mem.ptrtoint %va;
            %xb = mem.ptrtoint %vb;
            builtin.ret %xa, %ib, %xb, %ia : (isize, isize, isize, isize);
        } attributes { wasm.export_name = "check" };
    };
    builtin.module public @b {
        builtin.global_variable public @pa : ptr attributes { link_name = "a::pa" };
        builtin.global_variable public @pb : ptr {
            %g = builtin.global_symbol @pa : ptr<ptr>;
            %p = mem.ptrcast %g : ptr;
            builtin.yield %p;
        };
    };
};"#;
    let mut instance = compiled(world).run();
    // `@a::pa` holds the address of `@b::pb`, and `@b::pb` that of `@a::pa`.
    let [held_by_a, b, held_by_b, a] = values(instance.call("check", &[]))[..] else {
        panic!("four values");
    };
    assert_eq!(held_by_a, b);
    assert_eq!(held_by_b, a);
    assert_ne!(a, b);
}

#[test]
fn a_function_of_another_module_and_one_through_an_alias_are_called_from_data() {
    // `@app` declares `@lib::triple` and `@lib::squared`, an alias of `@lib::square`, by their
    // link names, and has an alias of its own, `@thrice`, of its declaration of `@triple`.
    let world = r#"builtin.world {
    builtin.module public @lib {
        builtin.function public extern("C") @triple(%x: i64) -> i64 {
            %three = arith.constant 3 : i64;
            %r = arith.mul %x, %three;
            builtin.ret %r : (i64);
        };
        builtin.function public extern("C") @square(%x: i64) -> i64 {
            %r = arith.mul %x, %x;
            builtin.ret %r : (i64);
        };
        builtin.function_alias public @squared -> @square;
    };
    builtin.module public @app {
        builtin.function public extern("C") @triple(i64) -> i64 attributes { link_name = "lib::triple" };
        builtin.function public extern("C") @squared(i64) -> i64 attributes { link_name = "lib::squared" };
        builtin.function_alias private @thrice -> @triple;
        builtin.global_variable private @ops : array<(i64) -> i64; 3> {
            %t = builtin.function_ref @thrice : (i64) -> i64;
            %s = builtin.function_ref @squared : (i64) -> i64;
            %u = builtin.function_ref @triple : (i64) -> i64;
            %a = builtin.aggregate %t, %s, %u : array<(i64) -> i64; 3>;
            builtin.yield %a;
        };
        builtin.function public extern("C") @apply(%i: isize, %x: i64) -> i64 {
            %g = builtin.global_symbol @ops : ptr<array<(i64) -> i64; 3>>;
            %p = mem.ptrcast %g : ptr<(i64) -> i64>;
            %q = mem.gep %p, %i;
            %f = mem.load %q;
            %r = builtin.call_indirect %f(%x) : extern("C") (i64) -> i64;
            builtin.ret %r : (i64);
        } attributes { wasm.export_name = "apply" };
    };
};"#;
    let program = compiled(world);
    // Each function has one slot, however it is named.
    assert_mentions(&program.wat(), &["(elem (;0;) (i32.const 1) func $lib::triple $lib::square)"]);
    let mut instance = program.run();
    assert_eq!(one(instance.call("apply", &[Val::i32(0), Val::i64(5)])), 15);
    assert_eq!(one(instance.call("apply", &[Val::i32(1), Val::i64(5)])), 25);
    assert_eq!(one(instance.call("apply", &[Val::i32(2), Val::i64(6)])), 18);
}

// ---- Functions whose signatures the lowering converts -------------------------------------------

#[test]
fn a_function_value_in_data_whose_callee_takes_a_pointer_is_called_through() {
    let program = compiled(&module(&[
        r#"builtin.function public extern("C") @deref(%p: ptr<i32>) -> i32 {
    %v = mem.load %p;
    builtin.ret %v : (i32);
};"#,
        "builtin.global_variable private @k : i32 { %v = arith.constant 77 : i32; builtin.yield \
         %v; };",
        "builtin.global_variable public @f : (ptr<i32>) -> i32 {
    %b = builtin.function_ref @deref : (ptr<i32>) -> i32;
    builtin.yield %b;
};",
        r#"builtin.function public extern("C") @run() -> i32 {
    %fp = builtin.global_symbol @f : ptr<(ptr<i32>) -> i32>;
    %f = mem.load %fp;
    %k = builtin.global_symbol @k : ptr<i32>;
    %r = builtin.call_indirect %f(%k) : extern("C") (ptr<i32>) -> i32;
    builtin.ret %r : (i32);
} attributes { wasm.export_name = "run" };"#,
    ]));
    assert_eq!(one(program.run().call("run", &[])), 77);
}

#[test]
fn functions_taking_a_narrow_integer_and_a_pointer_are_called_from_a_struct_in_data() {
    let ty = "struct<(i8) -> i8, (ptr<i32>) -> i32, ptr<i32>>";
    let program = compiled(&module(&[
        r#"builtin.function public extern("C") @inc8(%x: i8) -> i8 {
    %one = arith.constant 1 : i8;
    %r = arith.add %x, %one;
    builtin.ret %r : (i8);
};"#,
        r#"builtin.function public extern("C") @deref(%p: ptr<i32>) -> i32 {
    %v = mem.load %p;
    builtin.ret %v : (i32);
};"#,
        "builtin.global_variable private @k : i32 { %v = arith.constant 77 : i32; builtin.yield \
         %v; };",
        &format!(
            "builtin.global_variable public @t : {ty} {{
    %a = builtin.function_ref @inc8 : (i8) -> i8;
    %b = builtin.function_ref @deref : (ptr<i32>) -> i32;
    %c = builtin.global_symbol @k : ptr<i32>;
    %v = builtin.aggregate %a, %b, %c : {ty};
    builtin.yield %v;
}};"
        ),
        &format!(
            r#"builtin.function public extern("C") @narrow(%x: i8) -> i8 {{
    %t = builtin.global_symbol @t : ptr<{ty}>;
    %fp = mem.gep %t, [0];
    %f = mem.load %fp;
    %y = builtin.call_indirect %f(%x) : extern("C") (i8) -> i8;
    builtin.ret %y : (i8);
}} attributes {{ wasm.export_name = "narrow" }};"#
        ),
        &format!(
            r#"builtin.function public extern("C") @follow() -> i32 {{
    %t = builtin.global_symbol @t : ptr<{ty}>;
    %gp = mem.gep %t, [1];
    %g = mem.load %gp;
    %pp = mem.gep %t, [2];
    %p = mem.load %pp;
    %z = builtin.call_indirect %g(%p) : extern("C") (ptr<i32>) -> i32;
    builtin.ret %z : (i32);
}} attributes {{ wasm.export_name = "follow" }};"#
        ),
    ]));
    let mut instance = program.run();
    assert_eq!(one(instance.call("narrow", &[Val::i32(41)])), 42);
    // `i8` wraps: 255 + 1 is 0.
    assert_eq!(one(instance.call("narrow", &[Val::i32(255)])), 0);
    assert_eq!(one(instance.call("follow", &[])), 77);
}

#[test]
fn a_global_of_a_function_type_declared_in_one_module_and_defined_in_another_agree() {
    // Both modules hold the type `(ptr<i64>) -> i64`; each converts it as a signature.
    let world = r#"builtin.world {
    builtin.module public @lib {
        builtin.function public extern("C") @twice(%p: ptr<i64>) -> i64 {
            %v = mem.load %p;
            %r = arith.add %v, %v;
            builtin.ret %r : (i64);
        };
        builtin.global_variable public @op : (ptr<i64>) -> i64 {
            %f = builtin.function_ref @twice : (ptr<i64>) -> i64;
            builtin.yield %f;
        };
        builtin.global_variable public @n : i64 {
            %v = arith.constant 21 : i64;
            builtin.yield %v;
        };
    };
    builtin.module public @app {
        builtin.global_variable public @op : (ptr<i64>) -> i64 attributes { link_name = "lib::op" };
        builtin.global_variable public @n : i64 attributes { link_name = "lib::n" };
        builtin.function public extern("C") @apply() -> i64 {
            %g = builtin.global_symbol @op : ptr<(ptr<i64>) -> i64>;
            %f = mem.load %g;
            %n = builtin.global_symbol @n : ptr<i64>;
            %r = builtin.call_indirect %f(%n) : extern("C") (ptr<i64>) -> i64;
            builtin.ret %r : (i64);
        } attributes { wasm.export_name = "apply" };
    };
};"#;
    assert_eq!(one(compiled(world).run().call("apply", &[])), 42);
}

#[test]
fn a_function_type_in_data_whose_parameter_has_no_carrier_is_an_error_naming_the_global() {
    let error = failure(&module(&[
        "builtin.global_variable public @wide : struct<i32, ptr<(i128) -> i32>>;",
        r#"builtin.function public extern("C") @ok(%x: i8) -> i8 { builtin.ret %x : (i8); };"#,
    ]));
    assert_mentions(
        &error,
        &[
            "`convert-to-wasm` cannot lower a value of type `i128`: integers wider than 64 bits \
             are not lowered yet",
            "parameter 0 of the function type `(i128) -> i32`, which the global variable `@wide` \
             holds, has type `i128`",
        ],
    );
}

// ---- Errors -------------------------------------------------------------------------------------

#[test]
fn arithmetic_on_an_address_in_an_initializer_is_the_evaluators_error() {
    let error = failure(&module(&[
        "builtin.global_variable private @x : i32 { %v = arith.constant 5 : i32; builtin.yield \
         %v; };",
        "builtin.global_variable private @p : isize {
    %g = builtin.global_symbol @x : ptr<i32>;
    %i = mem.ptrtoint %g;
    %one = arith.constant 1 : isize;
    %j = arith.add %i, %one;
    builtin.yield %j;
};",
    ]));
    assert_mentions(
        &error,
        &[
            "the initializer of `@p` is not a constant that can be placed",
            "arithmetic on an address cannot be placed: write the offset with `mem.gep`",
            "the global variable it initializes",
        ],
    );
}

#[test]
fn a_global_larger_than_the_memory_is_refused_by_its_type_before_its_bytes_are_built() {
    // 8 GiB and one byte: refused from its type, so no byte of it is built.
    let ty = "struct<array<i8; 8589934592>, i8>";
    let error = failure(&module(&[&format!(
        "builtin.global_variable private @big : {ty} {{
    %z = builtin.zero : array<i8; 8589934592>;
    %o = arith.constant 1 : i8;
    %v = builtin.aggregate %z, %o : {ty};
    builtin.yield %v;
}};"
    )]));
    assert_mentions(
        &error,
        &[
            "the global variable `@big` cannot be laid out in linear memory",
            "its type `struct<array<i8; 8589934592>, i8>` is 8589934593 bytes, more than the \
             4294967296 bytes of a 32-bit linear memory",
        ],
    );
    // A zero of the same size is refused the same way.
    let error = failure(&module(&["builtin.global_variable private @zeros : array<i8; \
                                   8589934592> { %z = builtin.zero : array<i8; 8589934592>; \
                                   builtin.yield %z; };"]));
    assert_mentions(
        &error,
        &["its type `array<i8; 8589934592>` is 8589934592 bytes, more than the 4294967296 bytes"],
    );
}

#[test]
fn a_function_value_in_data_under_a_layout_without_fn_is_the_evaluators_error() {
    // The harness refuses a module whose layout is not the target's, so the module is lowered and
    // emitted directly.
    let error = emit_failure(
        "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, ptr = \
         32> } {
    builtin.function public extern(\"wasm\") @f() { builtin.ret; };
    builtin.global_variable private @fp : () -> () {
        %f = builtin.function_ref @f : () -> ();
        builtin.yield %f;
    };
};",
    );
    assert_mentions(
        &error,
        &[
            "the initializer of `@fp` is not a constant that can be placed",
            "a function value has a layout only where the data layout states one, as in `fn = 32`",
        ],
    );
}

#[test]
fn a_passive_segment_that_holds_an_address_is_an_error_naming_the_global() {
    let error = failure(&module(&[
        "builtin.global_variable private @x : i32 { %v = arith.constant 5 : i32; builtin.yield \
         %v; };",
        "builtin.global_variable private @p : ptr<i32> { %g = builtin.global_symbol @x : \
         ptr<i32>; builtin.yield %g; };",
        "wasm.memory private declaration @__linear_memory : 0 attributes { link_name = \
         \"__linear_memory\" };",
        r#"builtin.function public extern("wasm") @copy(%d: i32) {
    %s = wasm.const 0 : i32;
    %n = wasm.const 4 : i32;
    wasm.memory_init @__linear_memory, @p, %d, %s, %n;
    builtin.ret;
};"#,
    ]));
    assert_mentions(
        &error,
        &[
            "the global variable `m::p` is a passive data segment and holds the address of \
             `m::x`, which the layout cannot write into a passive segment",
            "the initializer of this global variable places the address of `m::x` at offset 0",
            "this makes `m::p` a passive segment, copied by `memory.init`: the layout's addresses \
             and table slots are not applied to what it copies",
        ],
    );
}

#[test]
fn a_zeroed_passive_segment_is_written_out_and_copied() {
    let program = compiled(&module(&[
        "builtin.global_variable private @zeros : array<i32; 2> { %z = builtin.zero : array<i32; \
         2>; builtin.yield %z; };",
        "wasm.memory private declaration @__linear_memory : 0 attributes { link_name = \
         \"__linear_memory\" };",
        r#"builtin.function public extern("wasm") @copy(%d: i32) -> i32 {
    %v = wasm.const 7 : i32;
    wasm.store @__linear_memory, %d, %v;
    %s = wasm.const 0 : i32;
    %n = wasm.const 8 : i32;
    wasm.memory_init @__linear_memory, @zeros, %d, %s, %n;
    %r = wasm.load @__linear_memory, %d : i32;
    builtin.ret %r : (i32);
} attributes { wasm.export_name = "copy" };"#,
    ]));
    assert_mentions(
        &program.wat(),
        &["(data $m::zeros (;0;) \"\\00\\00\\00\\00\\00\\00\\00\\00\")"],
    );
    assert_eq!(one(program.run().call("copy", &[Val::i32(4096)])), 0);
}

/// The error of emitting `text`, a module in Wasm form, with [emit_module]: the harness, which
/// would give it the target's data layout, is not involved.
fn emit_failure(text: &str) -> String {
    let context = context();
    // The context outlives the module parsed in it.
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "m", text)
        .into_result()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    error_of(emit_module(module.as_operation_ref(), &EmitOptions::default()))
}

#[test]
fn an_address_or_a_function_value_of_8_bytes_is_an_error_naming_the_global() {
    let address = emit_failure(
        "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, ptr = \
         64> } {
    builtin.global_variable private @x : i32 { %v = arith.constant 5 : i32; builtin.yield %v; };
    builtin.global_variable private @p : ptr<i32> {
        %g = builtin.global_symbol @x : ptr<i32>;
        builtin.yield %g;
    };
};",
    );
    assert_mentions(
        &address,
        &[
            "the global variable `@p` holds an address of 8 bytes, which the backend cannot place",
            "its initializer places an address of 8 bytes at offset 0; the backend writes an \
             address or a table slot in data as 4 bytes, as the data layout of wasm32 gives them",
        ],
    );
    let function = emit_failure(
        "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, ptr = \
         32, fn = 64> } {
    builtin.function public extern(\"wasm\") @f() { builtin.ret; };
    builtin.global_variable private @fp : () -> () {
        %f = builtin.function_ref @f : () -> ();
        builtin.yield %f;
    };
};",
    );
    assert_mentions(
        &function,
        &[
            "the global variable `@fp` holds a function value of 8 bytes, which the backend \
             cannot place",
        ],
    );
}
