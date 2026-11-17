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
    dialects::builtin::{Bytes, GlobalVariable},
    matchers::{self, Matcher},
    parse::{ParserConfig, parse_any},
    pass::{Nesting, PassPipeline, registry::PassInfo},
};

/// Initializers of both kinds of global, in the shapes a pass is most likely to change: a single
/// constant; a duplicate constant (the yielded one is the later of two equal operations, so
/// merging `2; 1; 2` into `2; 1` must keep the yield on the first); an unused operation; two
/// constants and an operation on them; the same operation in two globals and in a function beside
/// them; equal operations that are used.
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
}

impl Value {
    fn int(&self) -> i64 {
        match self {
            Self::Int(value) => *value,
            Self::Bytes(_) => panic!("{self:?} is not an integer"),
        }
    }
}

/// The value of `value`, computed from the operations that define it: the constants, and the
/// additions and multiplications of 32-bit integers the shapes use.
fn evaluate(value: ValueRef) -> Value {
    let definition = value
        .borrow()
        .get_defining_op()
        .unwrap_or_else(|| panic!("`{}` has no defining operation", value.borrow()));
    let op = definition.borrow();
    if let Some(bytes) = op.downcast_ref::<Bytes>() {
        return Value::Bytes(bytes.data().as_slice().to_vec());
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

/// The value of each global of a module by name, and the size of its initializer.
type Globals = BTreeMap<String, (Value, usize)>;

/// The value of each global of `module` by name, and the number of operations in its initializer
/// (the yield is not counted).
fn globals(module: OperationRef) -> Globals {
    let mut found = BTreeMap::new();
    module.borrow().prewalk_all(|op: &Operation| {
        let (name, value) = if let Some(global) = op.downcast_ref::<GlobalVariable>() {
            (global.get_name().as_str().to_string(), global.initializer_value())
        } else if let Some(global) = op.downcast_ref::<runic_dialect_wasm::Global>() {
            (global.get_name().as_str().to_string(), global.initializer_value())
        } else {
            return;
        };
        let value = value.unwrap_or_else(|| panic!("`@{name}` yields no value:\n{op}"));
        let mut count = 0;
        op.prewalk_all(|_: &Operation| count += 1);
        // The global itself and the yield are not operations of the computation
        found.insert(name, (evaluate(value), count - 2));
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

/// Runs `pipeline` as it is, or, for a pass that runs on functions, nested on them.
fn run_nested(pipeline: &str) -> Result<(Globals, Globals), Report> {
    run_shapes(pipeline)
        .or_else(|_| run_shapes(&format!("recursive(builtin.function({pipeline}))")))
}

/// The registered passes this test cannot run on `SHAPES` as they are, with the reason. Every
/// other registered pass, and so every pass added to the build later, is run. (`test-fail` is
/// registered by `runic-opt`'s own tests, which this crate's test binary does not link: it is
/// listed so that the list says what a build that links it does.)
const NOT_RUN: [(&str, &str); 1] = [("test-fail", "fails at the operation it runs on, by design")];

/// The default pipeline of the compiler.
const DEFAULT: &str = "canonicalizer,cse,sccp,canonicalizer,dce";

/// No pass in the build changes the value of a global: every registered pass runs over the shapes
/// above (and the default pipeline too), the module verifies after each, and each global yields
/// the value it did.
#[test]
fn no_registered_pass_changes_the_value_of_a_global() {
    let registered = PassInfo::registered();
    let mut ran = Vec::new();
    let mut failures = Vec::new();
    let mut pipelines = registered
        .iter()
        .map(|pass| pass.argument())
        .filter(|argument| !NOT_RUN.iter().any(|(name, _)| name == argument))
        .map(String::from)
        .collect::<Vec<_>>();
    pipelines.push(DEFAULT.to_string());
    for pipeline in pipelines {
        let (before, after) = match run_nested(&pipeline) {
            Ok(result) => result,
            Err(error) => {
                failures.push(format!("`{pipeline}` could not run: {error}"));
                continue;
            }
        };
        let values = |globals: &Globals| {
            globals
                .iter()
                .map(|(name, (value, _))| (name.clone(), value.clone()))
                .collect::<Vec<_>>()
        };
        if values(&before) != values(&after) {
            failures.push(format!(
                "`{pipeline}` changed the value of a global:\nbefore: {:?}\nafter: {:?}",
                values(&before),
                values(&after)
            ));
        }
        ran.push(pipeline);
    }
    assert!(failures.is_empty(), "{}\n(ran: {ran:?})", failures.join("\n"));
    for pass in ["canonicalizer", "cse", "dce", "sccp", "sink-operand-defs", "control-flow-sink"] {
        assert!(ran.iter().any(|ran| ran == pass), "{pass} did not run: {ran:?}");
    }
}

/// The values the shapes hold, as computed from their initializers, are the ones the text states:
/// the evaluator is right, so the test above compares real values.
#[test]
fn the_evaluator_computes_the_values_the_shapes_hold() {
    let (before, _) = run_shapes("").expect("the shapes parse and verify");
    let values = before
        .iter()
        .map(|(name, (value, _))| (name.as_str(), value.clone()))
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
    for (name, (value, _)) in &before {
        assert_eq!(&after[name].0, value, "`@{name}`");
    }
}
