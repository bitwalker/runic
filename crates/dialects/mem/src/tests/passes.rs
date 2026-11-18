//! What an initializer denotes through every registered pass: [evaluate_initializer] gives the
//! same [Initializer] before and after each, and after the default pipeline, for initializers that
//! hold addresses (a `mem.gep` chain, a `mem.ptrcast`, a `mem.ptrtoint` and `mem.inttoptr` pair),
//! functions (through an alias) and a nested aggregate that shares them, with operations the
//! passes can merge, fold and remove beside them.

use alloc::{
    collections::BTreeMap,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{
    Context, DataLayout, Op, Operation, OperationRef, Report,
    dialects::builtin::GlobalVariable,
    parse::{ParserConfig, parse_any},
    pass::{
        Nesting, OpPassManager, PassOptions, PassPipeline,
        registry::{PassInfo, RegistryEntry},
    },
    testing::text::describe,
};

use crate::evaluate_initializer;

/// The globals: the targets, then the shapes.
const SHAPES: &str = r#"builtin.module public @shapes attributes { data_layout = #builtin.data_layout<little, ptr = 32, fn = 32> } {
    builtin.global_variable public @target : i32 {
        %v = arith.constant 7 : i32;
        builtin.yield %v;
    };
    builtin.global_variable public @table : struct<i8, array<struct<i16, i32>; 4>> {
        %v = builtin.zero : struct<i8, array<struct<i16, i32>; 4>>;
        builtin.yield %v;
    };
    builtin.function public extern("C") @callee(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function_alias public @alias -> @callee;

    // A gep chain: an index computed by a fold, a gep of a gep of a gep, and an unused gep.
    builtin.global_variable public @chain : ptr<i32> {
        %p = builtin.global_symbol @table : ptr<struct<i8, array<struct<i16, i32>; 4>>>;
        %one = arith.constant 1 : isize;
        %i = arith.add %one, %one;
        %q = mem.gep %p, %i, [1];
        %r = mem.gep %q, [3];
        %unused = mem.gep %q, [2];
        %v = mem.gep %r, [1];
        builtin.yield %v;
    };

    // A ptrcast, then a gep over bytes by a constant, twice over the same base.
    builtin.global_variable public @cast : ptr<i8> {
        %p = builtin.global_symbol @table : ptr<struct<i8, array<struct<i16, i32>; 4>>>;
        %c = mem.ptrcast %p : ptr<i8>;
        %d = mem.ptrcast %p : ptr<i8>;
        %k = arith.constant 3 : isize;
        %e = mem.gep %d, %k;
        %v = mem.gep %c, %k;
        builtin.yield %v;
    };

    // A ptrtoint and inttoptr pair, which a fold may remove, and one to another pointer type.
    builtin.global_variable public @pair : ptr<i32> {
        %p = builtin.global_symbol @target : ptr<i32>;
        %i = mem.ptrtoint %p;
        %v = mem.inttoptr %i : ptr<i32>;
        builtin.yield %v;
    };
    builtin.global_variable public @retyped : ptr<i16> {
        %p = builtin.global_symbol @target : ptr<i32>;
        %i = mem.ptrtoint %p;
        %v = mem.inttoptr %i : ptr<i16>;
        builtin.yield %v;
    };
    builtin.global_variable public @tagged : isize {
        %p = builtin.global_symbol @target : ptr<i32>;
        %b = mem.ptrcast %p : ptr<i8>;
        %t = arith.constant 1 : isize;
        %q = mem.gep %b, %t;
        %v = mem.ptrtoint %q;
        builtin.yield %v;
    };

    // A function through an alias, and the same function named twice.
    builtin.global_variable public @function : (i32) -> i32 {
        %a = builtin.function_ref @callee : (i32) -> i32;
        %v = builtin.function_ref @alias : (i32) -> i32;
        builtin.yield %v;
    };

    // An absolute address and a null pointer.
    builtin.global_variable public @absolute : struct<ptr, ptr> {
        %a = arith.constant 4096 : isize;
        %z = arith.constant 0 : isize;
        %p = mem.inttoptr %a : ptr;
        %n = mem.inttoptr %z : ptr;
        %v = builtin.aggregate %p, %n : struct<ptr, ptr>;
        builtin.yield %v;
    };

    // A nested aggregate sharing an address, a function and an inner aggregate, with an equal
    // inner aggregate and an unused one beside it.
    builtin.global_variable public @nested : struct<ptr<i32>, (i32) -> i32, array<struct<ptr<i32>, (i32) -> i32, i32>; 2>> {
        %p = builtin.global_symbol @target : ptr<i32>;
        %b = mem.ptrcast %p : ptr<i8>;
        %four = arith.constant 4 : isize;
        %q = mem.gep %b, %four;
        %r = mem.ptrcast %q : ptr<i32>;
        %f = builtin.function_ref @alias : (i32) -> i32;
        %two = arith.constant 2 : i32;
        %three = arith.constant 3 : i32;
        %five = arith.add %two, %three;
        %x = builtin.aggregate %p, %f, %five : struct<ptr<i32>, (i32) -> i32, i32>;
        %y = builtin.aggregate %p, %f, %five : struct<ptr<i32>, (i32) -> i32, i32>;
        %unused = builtin.aggregate %r, %f, %two : struct<ptr<i32>, (i32) -> i32, i32>;
        %arr = builtin.aggregate %x, %y : array<struct<ptr<i32>, (i32) -> i32, i32>; 2>;
        %v = builtin.aggregate %r, %f, %arr : struct<ptr<i32>, (i32) -> i32, array<struct<ptr<i32>, (i32) -> i32, i32>; 2>>;
        builtin.yield %v;
    };
};"#;

/// What [evaluate_initializer] gives for each global of a module, printed, by name.
type Initializers = BTreeMap<String, String>;

/// What [evaluate_initializer] gives for each global with an initializer of `module`, by name,
/// printed (a relocation's target prints its symbol).
fn initializers(module: OperationRef) -> Initializers {
    let mut found = BTreeMap::new();
    module.borrow().prewalk_all(|op: &Operation| {
        let Some(global) = op.downcast_ref::<GlobalVariable>() else {
            return;
        };
        if global.initializer().is_empty() {
            return;
        }
        let layout = DataLayout::closest(global.as_operation()).expect("a layout");
        let name = global.get_name().as_str().to_string();
        let init = evaluate_initializer(global, &layout)
            .map(|init| format!("{init:?}"))
            .unwrap_or_else(|report| format!("error: {}", describe(&report)));
        found.insert(name, init);
    });
    found
}

/// The shapes, parsed in a context of their own.
fn shapes() -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    let module = parse_any(ParserConfig::new(context.clone()), "shapes", SHAPES.to_string())
        .into_result()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, module)
}

/// Runs `pipeline` on the shapes with the verifier on: what each global denotes before and after.
fn run(pipeline: &str) -> Result<(Initializers, Initializers), Report> {
    let (context, module) = shapes();
    let before = initializers(module);
    let pipeline = PassPipeline::parse_elements(&module.borrow().name(), pipeline)?;
    let mut passes = pipeline.load(context, Nesting::Implicit)?;
    passes.enable_verifier(true);
    passes.run(module)?;
    let after = initializers(module);
    drop(passes);
    Ok((before, after))
}

/// Whether `pass` can be scheduled on a `builtin.module`, which holds the globals.
fn on_the_module(pass: &PassInfo) -> bool {
    let (context, module) = shapes();
    let name = module.borrow().name();
    let mut manager = OpPassManager::any(Nesting::Implicit, context);
    pass.add_to_pipeline(&mut manager, &PassOptions::new())
        .unwrap_or_else(|error| panic!("`{}` cannot be added: {error}", pass.argument()));
    manager.passes().iter().all(|pass| pass.can_schedule_on(&name))
}

/// The registered passes this test does not run, with the reason.
const NOT_RUN: [(&str, &str); 1] = [("test-fail", "fails at the operation it runs on, by design")];

/// The default pipeline of the compiler.
const DEFAULT: &str = "canonicalizer,cse,sccp,canonicalizer,dce";

/// The passes that must run on the module, so that the test compares what they do to the globals.
const ON_THE_MODULE: [&str; 6] =
    ["canonicalizer", "cse", "dce", "sccp", "sink-operand-defs", "control-flow-sink"];

#[test]
fn every_shape_evaluates_before_any_pass_runs() {
    let (_context, module) = shapes();
    let found = initializers(module);
    assert_eq!(found.len(), 10, "{found:?}");
    for (name, init) in &found {
        assert!(!init.starts_with("error"), "@{name}: {init}");
    }
    // The shapes hold what they say: an address in the middle of `@table` (2 * 36 + 4 + 3 * 8 +
    // 4), a function through the alias, and an absolute address with no relocation
    assert!(
        found["chain"].contains("target: Data(@table), addend: 104"),
        "{}",
        found["chain"]
    );
    assert!(found["function"].contains("Function(@callee)"), "{}", found["function"]);
    assert!(found["absolute"].contains("bytes: [0, 16, 0, 0, 0, 0, 0, 0], relocations: []"));
}

#[test]
fn no_registered_pass_changes_what_an_initializer_with_addresses_and_functions_denotes() {
    let registered = PassInfo::registered();
    let mut pipelines = registered
        .iter()
        .filter(|pass| !NOT_RUN.iter().any(|(name, _)| *name == pass.argument()))
        .map(|pass| {
            let pipeline = pass.argument().to_string();
            if on_the_module(pass) {
                (pipeline, true)
            } else {
                (format!("recursive(builtin.function({pipeline}))"), false)
            }
        })
        .collect::<Vec<_>>();
    pipelines.push((DEFAULT.to_string(), true));
    let mut failures = Vec::new();
    let mut ran = Vec::new();
    for (pipeline, module) in pipelines {
        match run(&pipeline) {
            Err(error) => failures.push(format!("`{pipeline}` could not run: {error}")),
            Ok((before, after)) if before != after => {
                for (name, init) in &before {
                    if after.get(name) != Some(init) {
                        failures.push(format!(
                            "`{pipeline}` changed `@{name}`:\nbefore: {init}\nafter: {:?}",
                            after.get(name)
                        ));
                    }
                }
            }
            Ok(_) => ran.push((pipeline, module)),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // The passes changed the initializers: the default pipeline removes, merges and folds
    let (context, module) = shapes();
    let count = |module: OperationRef| {
        let mut count = 0usize;
        module.borrow().prewalk_all(|_: &Operation| count += 1);
        count
    };
    let before = count(module);
    let mut passes = PassPipeline::parse_elements(&module.borrow().name(), DEFAULT)
        .and_then(|pipeline| pipeline.load(context, Nesting::Implicit))
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    passes.run(module).unwrap_or_else(|report| panic!("{}", describe(&report)));
    assert!(count(module) < before, "the default pipeline changed nothing");
    drop(passes);
    for pass in ON_THE_MODULE.iter().chain([&DEFAULT]) {
        assert!(
            ran.iter().any(|(name, module)| name == pass && *module),
            "`{pass}` did not run on the module: {ran:?}"
        );
    }
}
