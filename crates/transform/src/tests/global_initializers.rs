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
