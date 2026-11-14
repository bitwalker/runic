//! Malformed text of every `mem` operation (and the local, global and bytes operations of
//! this part) is a diagnostic, never a panic; what still parses survives the passes.

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    Context, SymbolTable,
    dialects::builtin::{Function, Module},
    parse::{self, ParserConfig},
    pass::{Nesting, OperationPass, PassManager},
    testing::text::Rng,
};
use runic_transform::{Canonicalizer, CommonSubexpressionElimination, DeadCodeElimination};

/// Valid operations, one per line, in the function of [module].
const LINES: &[&str] = &[
    "%x = mem.load %p;",
    "%x = mem.load %q : i32;",
    "%x = mem.load %p, align 1;",
    "%x = mem.load %q, align 8 : i64;",
    "%x = mem.load %p attributes { note = #builtin.i32<1> };",
    "mem.store %v, %p;",
    "mem.store %v, %p, align 1;",
    "mem.store %v, %q, align 2;",
    "%x = mem.alloca i32;",
    "%x = mem.alloca struct<i32, i8>, align 16;",
    "%x = mem.gep %s, %n, [1, 2];",
    "%x = mem.gep %s, [1];",
    "%x = mem.gep %s, %n;",
    "%x = mem.gep %s;",
    "%x = mem.gep %e, [3, 1];",
    "%x = mem.sizeof struct<i32, i8>;",
    "%x = mem.alignof i64;",
    "%x = mem.offsetof struct<i32, i8>, [1];",
    "%x = mem.offsetof enum<i8; a: i32, b, c: struct<i8, i16>>, [3, 1];",
    "mem.memcpy %p, %q, %n;",
    "mem.memmove %q, %p, %n;",
    "mem.memset %p, %b, %n;",
    "%x = mem.ptrtoint %p;",
    "%x = mem.inttoptr %n : ptr<i32>;",
    "%x = mem.ptrcast %p : ptr<i8>;",
    "%x = mem.ptrcast %r : ptr<i64, 1>;",
    "%x = builtin.load_local 0 : i32;",
    "builtin.store_local 0, %v;",
    "%x = builtin.global_symbol @g : ptr<struct<i32, i8>>;",
    "%x = builtin.bytes \"0201\" : array<i8; 2>;",
    "%x = \"mem.load\"(%p) <{ ty = #builtin.type<?>, align = #mem.align<natural> }> : ptr<i32> -> \
     i32;",
    "%x = \"mem.load\"(%q) <{ ty = #builtin.type<i64>, align = #mem.align<8> }> : ptr -> i64;",
    "\"mem.store\"(%v, %p) <{ align = #mem.align<1> }> : (i32, ptr<i32>) -> ();",
    "%x = \"mem.alloca\"() <{ ty = #builtin.type<i32>, align = #mem.align<natural> }> : () -> \
     ptr<i32>;",
    "%x = \"mem.sizeof\"() <{ ty = #builtin.type<struct<i32, i8>> }> : () -> isize;",
    "%x = \"mem.offsetof\"() <{ ty = #builtin.type<struct<i32, i8>>, path = #builtin.array_i32<1> \
     }> : () -> isize;",
    "\"mem.memcpy\"(%p, %q, %n) : (ptr<i32>, ptr, isize) -> ();",
    "\"mem.memset\"(%p, %b, %n) : (ptr<i32>, i8, isize) -> ();",
    "%x = \"mem.ptrtoint\"(%p) : ptr<i32> -> isize;",
    "%x = \"mem.inttoptr\"(%n) <{ ty = #builtin.type<ptr<i32>> }> : isize -> ptr<i32>;",
    "%x = \"mem.ptrcast\"(%p) <{ ty = #builtin.type<ptr<i8>> }> : ptr<i32> -> ptr<i8>;",
    "%x = \"mem.gep\"(%s) <{ path = #builtin.array_i32<1> }> : ptr<struct<i32, array<i8; 4>>> -> \
     ptr<array<i8; 4>>;",
    "%x = \"builtin.load_local\"() <{ local = #builtin.local_variable<0, i32> }> : () -> i32;",
];

/// Tokens that mutations may put in place of another.
const POOL: &[&str] = &[
    "%p",
    "%q",
    "%v",
    "%n",
    "%b",
    "%s",
    "%e",
    "%r",
    "%x",
    "%undefined",
    "i32",
    "i8",
    "isize",
    "ptr",
    "ptr<i8>",
    "ptr<i32, 1>",
    "?",
    "never",
    "(i32) -> i32",
    "struct<>",
    "array<i8; 0>",
    "0",
    "1",
    "-1",
    "3",
    "4294967295",
    "4294967296",
    "2147483648",
    "18446744073709551615",
    "18446744073709551616",
    "340282366920938463463374607431768211456",
    "align",
    "natural",
    ",",
    ":",
    ";",
    "[",
    "]",
    "<",
    ">",
    "<{",
    "}>",
    "(",
    ")",
    "{",
    "}",
    "=",
    "->",
    "@g",
    "@f",
    "@nope",
    "attributes",
    "#mem.align<3>",
    "#builtin.type<?>",
    "#builtin.array_i32<-1>",
    "#builtin.array_i32<4294967296, 1>",
    "#builtin.array_i32<>",
    "#builtin.local_variable<7, i64>",
    "\"text\"",
    "mem.load",
    "mem.gep",
    "^bb",
];

fn tokenize(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars = line.chars().collect::<Vec<_>>();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '"' {
            let start = i;
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            i += 1;
            tokens.push(chars[start..i.min(chars.len())].iter().collect());
        } else if c.is_alphanumeric() || "%@#_.^".contains(c) {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || "%@#_.^".contains(chars[i])) {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
        } else {
            tokens.push(c.to_string());
            i += 1;
        }
    }
    tokens
}

fn mutate(rng: &mut Rng, tokens: &mut Vec<String>) {
    if tokens.is_empty() {
        tokens.push(rng.pick(POOL).to_string());
        return;
    }
    let at = rng.below(tokens.len() as u64) as usize;
    match rng.below(6) {
        0 => {
            tokens.remove(at);
        }
        1 => {
            let token = tokens[at].clone();
            tokens.insert(at, token);
        }
        2 => {
            let other = rng.below(tokens.len() as u64) as usize;
            tokens.swap(at, other);
        }
        3 => {
            if at + 1 < tokens.len() {
                tokens.swap(at, at + 1);
            }
        }
        4 => tokens[at] = rng.pick(POOL).to_string(),
        _ => tokens.insert(at, rng.pick(POOL).to_string()),
    }
}

fn module(layout: bool, line: &str) -> String {
    let attributes = if layout {
        " attributes { data_layout = #builtin.data_layout<little, ptr = 32> }"
    } else {
        ""
    };
    format!(
        "builtin.module public @m{attributes} {{
    builtin.global_variable public @g : struct<i32, i8>;
    builtin.function public extern(\"C\") @f(%p: ptr<i32>, %q: ptr, %r: ptr<i8, 1>, %v: i32, %n: \
         isize, %b: i8, %s: ptr<struct<i32, array<i8; 4>>>, %e: ptr<enum<i8; a: i32, b, c: \
         struct<i8, i16>>>) locals(i32) {{
        {line}
        builtin.ret;
    }};
}};"
    )
}

type MakePass = fn() -> Box<dyn OperationPass>;

/// Parses and verifies `text`; when it is valid, prints it and runs the passes on it.
fn exercise(text: &str) -> bool {
    let context = Rc::new(Context::default());
    let Ok(module) =
        parse::parse::<Module>(ParserConfig::new(context.clone()), "fuzz", text.to_string())
            .into_result()
    else {
        return false;
    };
    let _ = module.as_operation_ref().borrow().to_string();
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let passes: [MakePass; 3] = [
        || Box::<Canonicalizer>::default(),
        || Box::<CommonSubexpressionElimination>::default(),
        || Box::<DeadCodeElimination>::default(),
    ];
    for make in passes {
        let mut manager = PassManager::on::<Function>(context.clone(), Nesting::Implicit);
        manager.add_pass(make()).unwrap();
        manager.enable_verifier(true);
        manager.run(function).expect("a pass leaves valid IR valid");
    }
    let _ = module.as_operation_ref().borrow().to_string();
    true
}

/// Every valid line parses and survives the passes, with and without a layout (the generic form
/// of a gep is the known exception: it does not parse back).
#[test]
fn every_seed_line_is_valid() {
    for line in LINES.iter().filter(|line| !line.contains("\"mem.gep\"")) {
        for layout in [false, true] {
            assert!(exercise(&module(layout, line)), "{line} (layout: {layout})");
        }
    }
}

/// 15 000 mutations of valid operations: none panics the parser, a verifier, the printer or a
/// pass.
#[test]
fn mutated_text_never_panics() {
    let mut rng = Rng(super::seed(0x0f02_23d3));
    // Silence the panics this test catches, and only those: other tests run on other threads
    let fuzzing = std::thread::current().id();
    let previous: alloc::sync::Arc<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync> =
        std::panic::take_hook().into();
    let chained = previous.clone();
    std::panic::set_hook(Box::new(move |info| {
        if std::thread::current().id() != fuzzing {
            chained(info);
        }
    }));
    let mut panics = Vec::new();
    let mut valid = 0;
    let total = 15_000;
    for _ in 0..total {
        let line = rng.pick(LINES);
        let mut tokens = tokenize(line);
        for _ in 0..1 + rng.below(3) {
            mutate(&mut rng, &mut tokens);
        }
        let mutated = tokens.join(" ");
        let text = module(rng.below(2) == 0, &mutated);
        match std::panic::catch_unwind(|| exercise(&text)) {
            Ok(true) => valid += 1,
            Ok(false) => (),
            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                panics.push(format!("`{mutated}` (from `{line}`): {message}"));
            }
        }
    }
    let _ = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| previous(info)));
    std::println!("{valid} of {total} mutations were valid; {} panicked", panics.len());
    panics.sort();
    panics.dedup();
    assert!(
        panics.is_empty(),
        "{} distinct panics:\n{}",
        panics.len(),
        panics.iter().take(40).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(valid > 120, "{valid}");
}
