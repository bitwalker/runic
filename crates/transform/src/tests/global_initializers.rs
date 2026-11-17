//! The initializer of a global through the passes of this crate: its operations are what the
//! global holds, so no pass removes, merges, folds or moves one, and a definition stays one.

extern crate std;

use alloc::{rc::Rc, string::ToString};

use runic_ir::{
    Context, Report,
    parse::{ParserConfig, parse_any},
    pass::{Nesting, PassPipeline},
};

const MODULE: &str = r#"builtin.module public @m {
    builtin.global_variable public @data : array<i8; 2> {
        %b = builtin.bytes "0a0b" : array<i8; 2>;
    };
    wasm.global public @g : i32 {
        %a = wasm.const 6 : i32;
        %b = wasm.const 7 : i32;
        %p = wasm.mul %a, %b;
    };
    builtin.global_variable public @k : i32 {
        %a = arith.constant 2 : i32;
        %b = arith.constant 1 : i32;
        %c = arith.add %b, %b;
    };
    builtin.global_variable public @l : i32 {
        %b = arith.constant 1 : i32;
        %c = arith.add %b, %b;
        %a = arith.constant 2 : i32;
    };
};"#;

fn run(pipeline: &str) -> Result<alloc::string::String, Report> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_wasm::WasmDialect>();
    let module =
        parse_any(ParserConfig::new(context.clone()), "global_initializers", MODULE.to_string())
            .into_result()?;
    let pipeline = PassPipeline::parse_elements(&module.borrow().name(), pipeline)?;
    let mut passes = pipeline.load(context.clone(), Nesting::Implicit)?;
    passes.enable_verifier(true);
    passes.run(module)?;
    let text = module.borrow().to_string();
    drop(passes);
    Ok(text)
}

/// The text of the module with the numbers of the values taken out, which the passes may assign
/// in another order.
fn shape(text: &str) -> alloc::string::String {
    let mut out = alloc::string::String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                chars.next();
            }
            out.push('%');
        } else {
            out.push(c);
        }
    }
    out
}

/// Each pass that reaches a global leaves every initializer as it was written.
#[test]
fn no_pass_changes_the_initializer_of_a_global() -> Result<(), Report> {
    let unchanged = shape(&run("")?);
    for pipeline in [
        "dce",
        "cse",
        "sccp",
        "canonicalizer",
        "control-flow-sink",
        "sink-operand-defs",
        "recursive(builtin.function(local2reg))",
        "cfg-to-scf",
        "canonicalizer,cse,sccp,canonicalizer,dce",
    ] {
        let text = run(pipeline)?;
        assert_eq!(shape(&text), unchanged, "{pipeline}");
    }
    Ok(())
}

/// A definition does not become a declaration.
#[test]
fn a_global_with_an_initializer_stays_a_definition_after_dce() -> Result<(), Report> {
    let text = run("dce")?;
    assert!(text.contains("builtin.bytes"), "{text}");
    assert!(text.contains("wasm.mul"), "{text}");
    Ok(())
}

// ---- every registered pass ---------------------------------------------------------------------

use alloc::{string::String, vec::Vec};

use runic_ir::{Operation, pass::registry::PassInfo};

/// Initializers of both kinds of global, in the shapes a pass is most likely to change: a single
/// constant; a duplicate constant before the last operation (the value of the initializer is the
/// last operation, so merging `2; 1; 2` into `2; 1` changes it); an unused intermediate; two
/// constants and an operation on them; equal constants in two globals and in a function beside
/// them.
const SHAPES: &str = r#"builtin.module public @shapes {
    builtin.global_variable public @k_single : i32 {
        %a = arith.constant 7 : i32;
    };
    builtin.global_variable public @k_duplicate : i32 {
        %a = arith.constant 2 : i32;
        %b = arith.constant 1 : i32;
        %c = arith.constant 2 : i32;
    };
    builtin.global_variable public @k_unused : i32 {
        %a = arith.constant 4 : i32;
        %b = arith.constant 5 : i32;
        %c = arith.constant 6 : i32;
    };
    builtin.global_variable public @k_operation : i32 {
        %a = arith.constant 6 : i32;
        %b = arith.constant 7 : i32;
        %c = arith.add %a, %b;
    };
    builtin.global_variable public @k_shared_one : i32 {
        %a = arith.constant 2 : i32;
        %b = arith.add %a, %a;
    };
    builtin.global_variable public @k_shared_two : i32 {
        %a = arith.constant 2 : i32;
        %b = arith.add %a, %a;
    };
    wasm.global public @w_single : i32 {
        %a = wasm.const 7 : i32;
    };
    wasm.global public @w_duplicate : i32 {
        %a = wasm.const 2 : i32;
        %b = wasm.const 1 : i32;
        %c = wasm.const 2 : i32;
    };
    wasm.global public @w_unused : i32 {
        %a = wasm.const 4 : i32;
        %b = wasm.const 5 : i32;
        %c = wasm.const 6 : i32;
    };
    wasm.global public @w_operation : i32 {
        %a = wasm.const 6 : i32;
        %b = wasm.const 7 : i32;
        %c = wasm.add %a, %b;
    };
    wasm.global public @w_shared_one : i32 {
        %a = wasm.const 2 : i32;
        %b = wasm.add %a, %a;
    };
    wasm.global public @w_shared_two : i32 {
        %a = wasm.const 2 : i32;
        %b = wasm.add %a, %a;
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

/// The text of every global of `module`, each printed alone.
fn globals(module: runic_ir::OperationRef) -> Vec<String> {
    let mut found = Vec::new();
    module.borrow().prewalk_all(|op: &Operation| {
        let name = op.name().to_string();
        if name == "builtin.global_variable" || name == "wasm.global" {
            found.push(op.to_string());
        }
    });
    found
}

/// Parses `SHAPES`, runs `pipeline` on it and returns the globals before and after.
fn run_shapes(pipeline: &str) -> Result<(Vec<String>, Vec<String>), Report> {
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

/// The registered passes this test cannot run on `SHAPES` as they are, with the reason. Every
/// other registered pass, and so every pass added to the build later, is run. (`test-fail` is
/// registered by `runic-opt`'s own tests, which this crate's test binary does not link: it is
/// listed so that the list says what a build that links it does.)
const NOT_RUN: [(&str, &str); 1] = [("test-fail", "fails at the operation it runs on, by design")];

/// No pass in the build changes an initializer: every registered pass runs over the shapes of
/// initializers above (and the default pipeline too), and each global prints as it was written.
#[test]
fn no_registered_pass_changes_an_initializer() {
    let registered = PassInfo::registered();
    let mut ran = Vec::new();
    let mut failures = Vec::new();
    let mut pipelines = registered
        .iter()
        .map(|pass| pass.argument())
        .filter(|argument| !NOT_RUN.iter().any(|(name, _)| name == argument))
        .map(String::from)
        .collect::<Vec<_>>();
    pipelines.push("canonicalizer,cse,sccp,canonicalizer,dce".to_string());
    for pipeline in pipelines {
        // A pass that runs on functions is nested on them, as `local2reg` is in the test above.
        let (before, after) = match run_shapes(&pipeline) {
            Ok(result) => result,
            Err(_) => {
                match run_shapes(&alloc::format!("recursive(builtin.function({pipeline}))")) {
                    Ok(result) => result,
                    Err(error) => {
                        failures.push(alloc::format!("`{pipeline}` could not run: {error}"));
                        continue;
                    }
                }
            }
        };
        if before != after {
            failures.push(alloc::format!(
                "`{pipeline}` changed an initializer:\nbefore:\n{}\nafter:\n{}",
                before.join("\n"),
                after.join("\n")
            ));
        }
        ran.push(pipeline);
    }
    assert!(failures.is_empty(), "{}\n(ran: {ran:?})", failures.join("\n"));
    for pass in ["canonicalizer", "cse", "dce", "sccp", "sink-operand-defs", "control-flow-sink"] {
        assert!(ran.iter().any(|ran| ran == pass), "{pass} did not run: {ran:?}");
    }
}
