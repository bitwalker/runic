use alloc::{format, rc::Rc, string::String};

use runic_ir::{
    Context, Op,
    testing::{parse_function_fixpoint, text::describe},
};

use crate::*;

mod attributes;
mod types;

/// A context with the `wasm` dialect registered.
fn context() -> Rc<Context> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<WasmDialect>();
    context
}

/// Parses `value` as the value of an attribute `a` of a function, verifies the function, and
/// gives the text of the attribute when the function is printed, or the error.
fn round_trip_attribute(value: &str) -> Result<String, String> {
    let context = context();
    let text = format!(
        "builtin.function public extern(\"C\") @f() {{\n    builtin.ret;\n}} attributes {{ a = \
         {value} }};"
    );
    let (function, printed) =
        parse_function_fixpoint(&context, "f", &text).map_err(|report| describe(&report))?;
    function
        .borrow()
        .as_operation()
        .verify()
        .map_err(|report| format!("does not verify: {}", describe(&report)))?;
    let (_, rest) = printed.split_once("attributes { a = ").expect("the attribute is printed");
    let (attr, _) = rest.split_once(" }").expect("the dictionary ends");
    Ok(String::from(attr))
}
