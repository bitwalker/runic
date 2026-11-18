//! The initializer of a global through the passes of this crate: it is ordinary code that ends
//! with a `builtin.yield` of the value of the global. A pass may remove an operation nothing uses,
//! merge equal operations and fold constants there, and may not change the value that is yielded.

extern crate std;

use alloc::{
    collections::BTreeMap,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{
    Context, Operation, OperationRef, Report, ValueRef,
    attributes::IntegerLikeAttr,
    dialects::builtin::{Aggregate, Bytes, GlobalVariable, Zero},
    matchers::{self, Matcher},
    parse::{ParserConfig, parse_any},
    pass::{
        Nesting, OpPassManager, PassOptions, PassPipeline,
        registry::{PassInfo, RegistryEntry},
    },
};

/// Initializers of both kinds of global, in the shapes a pass is most likely to change: a single
/// constant; a duplicate constant (the yielded one is the later of two equal operations, so
/// merging `2; 1; 2` into `2; 1` must keep the yield on the first); an unused operation; two
/// constants and an operation on them; the same operation in two globals and in a function beside
/// them; equal operations that are used; a nested aggregate whose element value is used by two
/// aggregates, beside an unused aggregate.
const SHAPES: &str = r#"builtin.module public @shapes {
    builtin.global_variable public @data : array<i8; 2> {
        %b = builtin.bytes "0a0b" : array<i8; 2>;
        builtin.yield %b;
    };
    builtin.global_variable public @k_single : i32 {
        %a = arith.constant 7 : i32;
        builtin.yield %a;
    };
    builtin.global_variable public @k_duplicate : i32 {
        %a = arith.constant 2 : i32;
        %b = arith.constant 1 : i32;
        %c = arith.constant 2 : i32;
        builtin.yield %c;
    };
    builtin.global_variable public @k_unused : i32 {
        %a = arith.constant 4 : i32;
        %b = arith.constant 5 : i32;
        %c = arith.constant 6 : i32;
        builtin.yield %b;
    };
    builtin.global_variable public @k_operation : i32 {
        %a = arith.constant 6 : i32;
        %b = arith.constant 7 : i32;
        %c = arith.add %a, %b;
        builtin.yield %c;
    };
    builtin.global_variable public @k_equal_sums : i32 {
        %a = arith.constant 2 : i32;
        %b = arith.constant 1 : i32;
        %c = arith.constant 2 : i32;
        %d = arith.add %a, %b;
        %e = arith.add %c, %b;
        %f = arith.add %d, %e;
        builtin.yield %f;
    };
    builtin.global_variable public @k_shared_one : i32 {
        %a = arith.constant 2 : i32;
        %b = arith.add %a, %a;
        builtin.yield %b;
    };
    builtin.global_variable public @k_shared_two : i32 {
        %a = arith.constant 2 : i32;
        %b = arith.add %a, %a;
        builtin.yield %b;
    };
    builtin.global_variable public @aggregate : array<struct<i32, i64>; 2> {
        %a = arith.constant 7 : i32;
        %b = arith.constant 8 : i32;
        %z = builtin.zero : i64;
        %unused = builtin.aggregate %b, %z : struct<i32, i64>;
        %first = builtin.aggregate %a, %z : struct<i32, i64>;
        %second = builtin.aggregate %b, %z : struct<i32, i64>;
        %both = builtin.aggregate %first, %second : array<struct<i32, i64>; 2>;
        builtin.yield %both;
    };
    wasm.global public @w_single : i32 {
        %a = wasm.const 7 : i32;
        builtin.yield %a;
    };
    wasm.global public @w_duplicate : i32 {
        %a = wasm.const 2 : i32;
        %b = wasm.const 1 : i32;
        %c = wasm.const 2 : i32;
        builtin.yield %c;
    };
    wasm.global public @w_unused : i32 {
        %a = wasm.const 4 : i32;
        %b = wasm.const 5 : i32;
        %c = wasm.const 6 : i32;
        builtin.yield %a;
    };
    wasm.global public @w_operation : i32 {
        %a = wasm.const 6 : i32;
        %b = wasm.const 7 : i32;
        %c = wasm.add %a, %b;
        builtin.yield %c;
    };
    wasm.global public @w_product : i32 {
        %a = wasm.const 6 : i32;
        %b = wasm.const 7 : i32;
        %c = wasm.mul %a, %b;
        builtin.yield %c;
    };
    wasm.global public @w_equal_sums : i32 {
        %a = wasm.const 2 : i32;
        %b = wasm.const 2 : i32;
        %c = wasm.add %a, %b;
        %d = wasm.add %b, %a;
        %e = wasm.add %c, %d;
        builtin.yield %e;
    };
    wasm.global public @w_shared_one : i32 {
        %a = wasm.const 2 : i32;
        %b = wasm.add %a, %a;
        builtin.yield %b;
    };
    wasm.global public @w_shared_two : i32 {
        %a = wasm.const 2 : i32;
        %b = wasm.add %a, %a;
        builtin.yield %b;
    };
    builtin.function public extern("C") @beside() -> i32 {
        %a = arith.constant 2 : i32;
        %b = arith.constant 2 : i32;
        %c = arith.add %a, %b;
        %d = wasm.const 2 : i32;
        %e = wasm.add %d, %d;
        %f = arith.constant 4 : i32;
        %g = arith.add %c, %f;
        builtin.ret %g : (i32);
    };
}; "#;

/// What a global holds, computed from its initializer.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Int(i64),
    Bytes(Vec<u8>),
    /// The zero of a type
    Zero,
    /// A struct or an array: the values of its elements, in order
    Aggregate(Vec<Value>),
}

impl Value {
    fn int(&self) -> i64 {
        match self {
            Self::Int(value) => *value,
            Self::Bytes(_) | Self::Zero | Self::Aggregate(_) => {
                panic!("{self:?} is not an integer")
            }
        }
    }
}

/// The value of `value`, computed from the operations that define it: the constants, the zeros,
/// the aggregates of their elements, and the additions and multiplications of 32-bit integers the
/// shapes use.
fn evaluate(value: ValueRef) -> Value {
    let definition = value
        .borrow()
        .get_defining_op()
        .unwrap_or_else(|| panic!("`{}` has no defining operation", value.borrow()));
    let op = definition.borrow();
    if let Some(bytes) = op.downcast_ref::<Bytes>() {
        return Value::Bytes(bytes.data().as_slice().to_vec());
    }
    if op.is::<Zero>() {
        return Value::Zero;
    }
    if op.is::<Aggregate>() {
        let elements =
            op.operands().iter().map(|operand| evaluate(operand.borrow().as_value_ref()));
        return Value::Aggregate(elements.collect());
    }
    if let Some(attribute) = matchers::constant().matches(&op) {
        let integer = attribute
            .borrow()
            .as_attr()
            .as_trait::<dyn IntegerLikeAttr>()
            .expect("an integer constant")
            .as_int();
        return Value::Int(integer.try_signed::<i64>().expect("a small integer"));
    }
    let operands = op
        .operands()
        .iter()
        .map(|operand| evaluate(operand.borrow().as_value_ref()).int())
        .collect::<Vec<_>>();
    let combined = match op.name().to_string().as_str() {
        "arith.add" | "wasm.add" => operands[0].wrapping_add(operands[1]),
        "wasm.mul" => operands[0].wrapping_mul(operands[1]),
        other => panic!("the evaluator does not know `{other}`"),
    };
    Value::Int(i64::from(combined as i32))
}

/// What a global is: its declared type, whether it is mutable, its visibility, and the value its
/// initializer computes. (A builtin global has no mutability, so it is `"-"`.)
#[derive(Debug, Clone, PartialEq, Eq)]
struct Shape {
    ty: String,
    mutability: &'static str,
    visibility: String,
    value: Value,
}

/// What each global of a module is by name, and the size of its initializer.
type Globals = BTreeMap<String, (Shape, usize)>;

/// What each global of `module` is by name, and the number of operations in its initializer (the
/// yield is not counted).
fn globals(module: OperationRef) -> Globals {
    let mut found = BTreeMap::new();
    module.borrow().prewalk_all(|op: &Operation| {
        let (name, ty, mutability, visibility, value) =
            if let Some(global) = op.downcast_ref::<GlobalVariable>() {
                (
                    global.get_name().as_str().to_string(),
                    global.get_ty().to_string(),
                    "-",
                    global.get_visibility().as_str().to_string(),
                    global.initializer_value(),
                )
            } else if let Some(global) = op.downcast_ref::<runic_dialect_wasm::Global>() {
                (
                    global.get_name().as_str().to_string(),
                    global.get_ty().to_string(),
                    if global.get_mutability().is_mut() {
                        "mut"
                    } else {
                        "const"
                    },
                    global.get_visibility().as_str().to_string(),
                    global.initializer_value(),
                )
            } else {
                return;
            };
        let value = value.unwrap_or_else(|| panic!("`@{name}` yields no value:\n{op}"));
        let mut count = 0;
        op.prewalk_all(|_: &Operation| count += 1);
        // The global itself and the yield are not operations of the computation
        let shape = Shape {
            ty,
            mutability,
            visibility,
            value: evaluate(value),
        };
        found.insert(name, (shape, count - 2));
    });
    found
}

/// Parses `SHAPES`, runs `pipeline` on it with the verifier on, and returns the globals before and
/// after.
fn run_shapes(pipeline: &str) -> Result<(Globals, Globals), Report> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_wasm::WasmDialect>();
    let module = parse_any(ParserConfig::new(context.clone()), "shapes", SHAPES.to_string())
        .into_result()?;
    let before = globals(module);
    let pipeline = PassPipeline::parse_elements(&module.borrow().name(), pipeline)?;
    let mut passes = pipeline.load(context.clone(), Nesting::Implicit)?;
    passes.enable_verifier(true);
    passes.run(module)?;
    let after = globals(module);
    drop(passes);
    Ok((before, after))
}

/// Where a registered pass is scheduled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Schedule {
    /// On the module, so that it reaches every global
    OnTheModule,
    /// On the functions only (`recursive(builtin.function(..))`): the pass cannot be scheduled on
    /// a module, and no global is reached
    OnFunctions,
}

/// Decides from the pass itself, not from a failed run, where it can be scheduled: whether it can
/// be scheduled on a `builtin.module`.
fn schedule_of(pass: &PassInfo) -> Schedule {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_wasm::WasmDialect>();
    let module = parse_any(ParserConfig::new(context.clone()), "shapes", SHAPES.to_string())
        .into_result()
        .expect("the shapes parse");
    let name = module.borrow().name();
    let mut manager = OpPassManager::any(Nesting::Implicit, context);
    pass.add_to_pipeline(&mut manager, &PassOptions::new())
        .unwrap_or_else(|error| panic!("`{}` cannot be added: {error}", pass.argument()));
    if manager.passes().iter().all(|pass| pass.can_schedule_on(&name)) {
        Schedule::OnTheModule
    } else {
        Schedule::OnFunctions
    }
}

/// Runs `pipeline` where `schedule` says: on the module as it is, or nested on the functions. A
/// failure is a failure: it is not retried in the other form.
fn run_scheduled(pipeline: &str, schedule: Schedule) -> Result<(Globals, Globals), Report> {
    match schedule {
        Schedule::OnTheModule => run_shapes(pipeline),
        Schedule::OnFunctions => run_shapes(&format!("recursive(builtin.function({pipeline}))")),
    }
}

/// The registered passes this test cannot run on `SHAPES` as they are, with the reason. Every
/// other registered pass, and so every pass added to the build later, is run. (`test-fail` is
/// registered by `runic-opt`'s own tests, which this crate's test binary does not link: it is
/// listed so that the list says what a build that links it does.)
const NOT_RUN: [(&str, &str); 1] = [("test-fail", "fails at the operation it runs on, by design")];

/// The default pipeline of the compiler.
const DEFAULT: &str = "canonicalizer,cse,sccp,canonicalizer,dce";

/// The passes that are run on the module, and so reach every global: each must be, or the test
/// compares nothing for it.
const ON_THE_MODULE: [&str; 7] = [
    "canonicalizer",
    "cse",
    "dce",
    "sccp",
    "sink-operand-defs",
    "control-flow-sink",
    "cfg-to-scf",
];

/// No pass in the build changes what a global is: every registered pass runs over the shapes
/// above (and the default pipeline too), on the module when it can be scheduled there and on the
/// functions when it cannot, the module verifies after each, and each global is what it was.
#[test]
fn no_registered_pass_changes_what_a_global_is() {
    let registered = PassInfo::registered();
    let mut ran = Vec::new();
    let mut failures = Vec::new();
    let mut pipelines = registered
        .iter()
        .filter(|pass| !NOT_RUN.iter().any(|(name, _)| *name == pass.argument()))
        .map(|pass| (pass.argument().to_string(), schedule_of(pass)))
        .collect::<Vec<_>>();
    pipelines.push((DEFAULT.to_string(), Schedule::OnTheModule));
    for (pipeline, schedule) in pipelines {
        let (before, after) = match run_scheduled(&pipeline, schedule) {
            Ok(result) => result,
            Err(error) => {
                failures.push(format!("`{pipeline}` ({schedule:?}) could not run: {error}"));
                continue;
            }
        };
        let shapes = |globals: &Globals| {
            globals
                .iter()
                .map(|(name, (shape, _))| (name.clone(), shape.clone()))
                .collect::<Vec<_>>()
        };
        if shapes(&before) != shapes(&after) {
            failures.push(format!(
                "`{pipeline}` changed a global:\nbefore: {:?}\nafter: {:?}",
                shapes(&before),
                shapes(&after)
            ));
        }
        ran.push((pipeline, schedule));
    }
    assert!(failures.is_empty(), "{}\n(ran: {ran:?})", failures.join("\n"));
    for pass in ON_THE_MODULE {
        assert!(
            ran.iter()
                .any(|(name, schedule)| name == pass && *schedule == Schedule::OnTheModule),
            "{pass} did not run on the module: {ran:?}"
        );
    }
    assert!(
        ran.iter()
            .any(|(name, schedule)| name == DEFAULT && *schedule == Schedule::OnTheModule),
        "the default pipeline did not run on the module: {ran:?}"
    );
}

/// The values the shapes hold, as computed from their initializers, are the ones the text states:
/// the evaluator is right, so the test above compares real values.
#[test]
fn the_evaluator_computes_the_values_the_shapes_hold() {
    let (before, _) = run_shapes("").expect("the shapes parse and verify");
    let values = before
        .iter()
        .map(|(name, (shape, _))| (name.as_str(), shape.value.clone()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(values["data"], Value::Bytes(alloc::vec![0x0b, 0x0a]));
    assert_eq!(values["k_single"], Value::Int(7));
    assert_eq!(values["k_duplicate"], Value::Int(2));
    assert_eq!(values["k_unused"], Value::Int(5));
    assert_eq!(values["k_operation"], Value::Int(13));
    assert_eq!(values["k_equal_sums"], Value::Int(6));
    assert_eq!(values["w_duplicate"], Value::Int(2));
    assert_eq!(values["w_unused"], Value::Int(4));
    assert_eq!(values["w_product"], Value::Int(42));
    assert_eq!(values["w_equal_sums"], Value::Int(8));
    let element = |a| Value::Aggregate(alloc::vec![Value::Int(a), Value::Zero]);
    assert_eq!(values["aggregate"], Value::Aggregate(alloc::vec![element(7), element(8)]));
}

/// An initializer is optimized: `dce` removes the operations nothing uses, `cse` merges the equal
/// ones, `sccp` and the canonicalizer fold constants, and each global yields what it did.
#[test]
fn the_pipeline_removes_unused_operations_and_merges_and_folds_the_rest() {
    let size = |globals: &Globals, name: &str| globals[name].1;

    let (before, after) = run_shapes("dce").unwrap();
    assert_eq!(size(&before, "k_unused"), 3);
    assert_eq!(size(&after, "k_unused"), 1, "dce removes the two operations nothing uses");
    assert_eq!(size(&after, "w_unused"), 1);
    assert_eq!(size(&after, "k_duplicate"), 1, "dce removes the constant `1` and the first `2`");
    assert_eq!(size(&before, "aggregate"), 7);
    assert_eq!(size(&after, "aggregate"), 6, "dce removes the aggregate nothing uses");

    let (before, after) = run_shapes("cse").unwrap();
    assert_eq!(size(&before, "k_equal_sums"), 6);
    assert!(
        size(&after, "k_equal_sums") < size(&before, "k_equal_sums"),
        "cse merges the equal constants and the equal sums"
    );
    assert!(size(&after, "w_equal_sums") < size(&before, "w_equal_sums"));
    assert_eq!(size(&after, "w_duplicate"), 1, "the first `2` is the `2` that is yielded");

    let (before, after) = run_shapes(DEFAULT).unwrap();
    assert_eq!(size(&before, "k_operation"), 3);
    assert_eq!(size(&after, "k_operation"), 1, "6 + 7 is the constant 13");
    assert_eq!(size(&after, "k_equal_sums"), 1, "(2 + 1) + (2 + 1) is the constant 6");
    assert_eq!(size(&after, "w_unused"), 1);
    for (name, (shape, _)) in &before {
        assert_eq!(&after[name].0, shape, "`@{name}`");
    }
}
