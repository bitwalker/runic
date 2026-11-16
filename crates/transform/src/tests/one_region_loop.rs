//! A loop with one region (`test.loop`), whose back edge is a `test.continue`, through the passes
//! of this crate: no pass removes it or its continue, or folds a value the continue changes. A
//! test with a control runs the same program with a `test.yield` where the first has a continue.

use alloc::{format, rc::Rc, string::ToString};

use runic_ir::{
    Context, Report,
    dialects::test::TestDialect,
    parse::{ParserConfig, parse_any},
    pass::{Nesting, PassPipeline},
};

/// `source`, the text of a function, after `pipeline` ran on it.
fn run(pipeline: &str, source: &str) -> Result<alloc::string::String, Report> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_arith::ArithDialect>();
    context.get_or_register_dialect::<TestDialect>();
    let text = format!("builtin.module public @m {{\n{source}\n}};");
    let module =
        parse_any(ParserConfig::new(context.clone()), "one_region_loop", text).into_result()?;
    let pipeline = PassPipeline::parse_elements(&module.borrow().name(), pipeline)?;
    let mut passes = pipeline.load(context.clone(), Nesting::Implicit)?;
    passes.enable_verifier(true);
    passes.run(module)?;
    let text = module.borrow().to_string();
    drop(passes);
    Ok(text)
}

/// The result of the loop is unused, and the loop may never end: it is not dead code, however
/// little it computes.
const UNUSED_LOOP: &str = r#"
    builtin.function public extern("C") @f(%c: i1, %a: i32) {
        %unused = test.loop %a {
        ^head(%i: i32):
            test.guard %c {
                test.continue %i : (i32);
            } : (i1);
            test.yield %i : (i32);
        } : (i32) -> (i32);
        builtin.ret;
    };"#;

const PIPELINES: [&str; 4] =
    ["dce", "sccp", "canonicalizer", "canonicalizer,cse,sccp,canonicalizer,dce"];

/// No pass erases a loop whose result is unused, or the continue in it.
#[test]
fn no_pass_removes_a_loop_that_a_continue_repeats() -> Result<(), Report> {
    for pipeline in PIPELINES {
        let text = run(pipeline, UNUSED_LOOP)?;
        assert!(text.contains("test.loop"), "{pipeline}: {text}");
        assert!(text.contains("test.continue"), "{pipeline}: {text}");
    }
    Ok(())
}

/// A loop with one region is a loop whether or not a continue goes back to it, and a loop may not
/// end: `dce` keeps one that nothing repeats too (soundness over precision).
#[test]
fn dce_keeps_a_one_region_loop_that_nothing_repeats() -> Result<(), Report> {
    let source = UNUSED_LOOP.replace("test.continue %i : (i32);", "test.yield : ();");
    let text = run("dce", &source)?;
    assert!(text.contains("test.loop"), "{text}");
    Ok(())
}

/// The value a continue changes is not folded to the constant the loop starts with, and is in the
/// control, where nothing changes it.
#[test]
fn sccp_does_not_fold_a_value_a_continue_changes() -> Result<(), Report> {
    let source = r#"
    builtin.function public extern("C") @f(%c: i1) -> i32 {
        %zero = arith.constant 0 : i32;
        %one = arith.constant 1 : i32;
        %r = test.loop %zero {
        ^head(%i: i32):
            %n = arith.add %i, %one;
            test.guard %c {
                test.continue %n : (i32);
            } : (i1);
            test.yield %i : (i32);
        } : (i32) -> (i32);
        builtin.ret %r : (i32);
    };"#;
    // The function returns the loop's result, which is not the constant the loop starts with
    let text = run("sccp", source)?;
    assert!(
        text.contains("%5 = test.loop %6") && text.contains("builtin.ret %5 : (i32)"),
        "{text}"
    );

    // The control returns that constant
    let control = source.replace("test.continue %n : (i32);", "test.yield : ();");
    let text = run("sccp", &control)?;
    assert!(
        text.contains("%5 = test.loop %6") && text.contains("builtin.ret %6 : (i32)"),
        "{text}"
    );
    Ok(())
}
