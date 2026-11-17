//! The unit tests of the crate.

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{
    Context, Op, OperationRef, Report,
    dialects::builtin::{Module, ModuleRef},
    parse::{self, ParserConfig},
    pass::{Nesting, PassPipeline},
    testing::text::describe,
};

mod atomic;
#[cfg(feature = "std")]
mod backend;
mod branches;
mod carried;
mod composed;
mod control;
mod conventions;
mod coverage;
#[cfg(feature = "std")]
mod emit {
    //! The emitter: its module description and writers, on descriptions built by hand; the plan,
    //! the locals and the bodies of functions in Wasm form; and emitted modules run with `wasmi`
    //! against the evaluator.

    mod bodies;
    mod control;
    mod errors;
    mod fixups;
    mod instructions;
    mod layout;
    mod locals;
    mod lowered;
    mod module;
    mod plan;
    mod sections;
    mod support;
    mod unit;
}
mod eval;
mod evaluated;
#[cfg(feature = "std")]
mod exec;
mod float;
mod form;
mod frames;
mod int;
mod labels;
mod modules;
mod pipeline;
mod programs;
mod symbols;
mod target;
mod types;

/// The attributes of a test module: the data layout the lowering needs.
const LAYOUT: &str = "attributes { data_layout = #builtin.data_layout<little, ptr = 32> }";

/// The four passes of the lowering, as a textual pipeline.
const LOWERING: &str =
    "cfg-to-scf{exits=in-place},wasm-lower-frames,convert-to-wasm,wasm-verify-form";

/// A module `@m` of the functions `functions`, with the data layout of the lowering.
fn module_of(functions: &str) -> String {
    format!("builtin.module public @m {LAYOUT} {{\n{functions}\n}};")
}

/// `text`, a module, parsed in `context` without being verified.
fn parse_unverified_in(context: &Rc<Context>, text: &str) -> Result<ModuleRef, Report> {
    let mut config = ParserConfig::new(context.clone());
    config.verify = false;
    parse::parse::<Module>(config, "m", text).into_result()
}

/// Runs the textual pipeline `pipeline` on `root`, with the verifier on.
fn run_pipeline(context: &Rc<Context>, root: OperationRef, pipeline: &str) -> Result<(), Report> {
    let anchor = root.borrow().name();
    let mut passes = PassPipeline::parse_elements(&anchor, pipeline)
        .and_then(|pipeline| pipeline.load(context.clone(), Nesting::Implicit))?;
    passes.enable_verifier(true);
    passes.run(root)
}

/// The module `text`, parsed, verified and lowered by the textual pipeline `pipeline`.
#[track_caller]
fn lowered_by(text: &str, pipeline: &str) -> (Rc<Context>, ModuleRef) {
    let (context, module) = parse_module(text);
    run_pipeline(&context, module.as_operation_ref(), pipeline).unwrap_or_else(|report| {
        panic!("{}\n--- the IR after the failure:\n{}", describe(&report), text_of(module))
    });
    (context, module)
}

/// The function `@name` that `module` directly holds.
#[track_caller]
fn function_named(module: ModuleRef, name: &str) -> OperationRef {
    runic_ir::SymbolTable::symbol_manager(&*module.borrow())
        .lookup_op(runic_ir::interner::Symbol::intern(name))
        .unwrap_or_else(|| panic!("no function @{name}"))
}

/// A context with the dialects the lowering reads and writes registered.
fn context() -> Rc<Context> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_wasm::WasmDialect>();
    context.get_or_register_dialect::<runic_dialect_arith::ArithDialect>();
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    context.get_or_register_dialect::<runic_dialect_mem::MemDialect>();
    context.get_or_register_dialect::<runic_dialect_scf::ScfDialect>();
    context.get_or_register_dialect::<runic_dialect_ub::UndefinedBehaviorDialect>();
    context
}

/// `text`, a module, parsed and verified in `context`.
fn parse_module_in(context: &Rc<Context>, text: &str) -> Result<ModuleRef, Report> {
    let module =
        parse::parse::<Module>(ParserConfig::new(context.clone()), "m", text).into_result()?;
    module.borrow().as_operation().recursively_verify()?;
    Ok(module)
}

/// `text`, a module, parsed and verified in a new context.
#[track_caller]
fn parse_module(text: &str) -> (Rc<Context>, ModuleRef) {
    let context = context();
    let module =
        parse_module_in(&context, text).unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, module)
}

/// The text of `module`.
fn text_of(module: ModuleRef) -> String {
    module.borrow().as_operation().to_string()
}

/// `text` with every value number (`%12`) written `%`, so that a test does not depend on the
/// numbering.
fn without_numbers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '%' {
            while chars.peek().is_some_and(char::is_ascii_digit) {
                chars.next();
            }
        }
    }
    out
}

/// The lines of `text` without their indentation, and without value numbers.
fn lines(text: &str) -> Vec<String> {
    text.lines().map(|line| without_numbers(line.trim())).collect()
}

/// The error of `result`, which must be one, described with its labels.
#[track_caller]
fn error_of<T>(result: Result<T, Report>) -> String {
    match result {
        Ok(_) => panic!("succeeded"),
        Err(report) => describe(&report),
    }
}

/// Whether `haystack` holds `needle`, as written or as a description escapes it.
fn mentions(haystack: &str, needle: &str) -> bool {
    haystack.contains(needle) || haystack.contains(&needle.replace('"', "\\\""))
}

#[test]
fn without_numbers_drops_the_numbers_of_values_only() {
    assert_eq!(without_numbers("%12 = wasm.and %3, %45 : i32;"), "% = wasm.and %, % : i32;");
    assert_eq!(without_numbers("wasm.const 255 : i32;"), "wasm.const 255 : i32;");
}
