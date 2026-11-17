//! The unit tests of the crate.

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{
    Context, Op, Report,
    dialects::builtin::{Module, ModuleRef},
    parse::{self, ParserConfig},
    testing::text::describe,
};

mod carried;
mod pipeline;
mod symbols;
mod target;
mod types;

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
