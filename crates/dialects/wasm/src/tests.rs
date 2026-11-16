use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{
    Context, Op, OperationRef, Report, SymbolTable,
    dialects::builtin::{FunctionRef, Module},
    parse::{self, ParserConfig},
    pass::{Nesting, PassManager},
    testing::{parse_function_fixpoint, text::describe},
};
use runic_transform::{CommonSubexpressionElimination, DeadCodeElimination};

use crate::*;

mod attributes;
mod compare;
mod convert;
mod numeric;
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

/// The ten types the numeric operations are tested against: the six Wasm value types, and the
/// four builtin types that are not (`i8`, `i1`, `isize`, a pointer).
const TYPES: [&str; 10] = [
    "i32",
    "i64",
    "f32",
    "f64",
    "!wasm.funcref",
    "!wasm.externref",
    "i8",
    "i1",
    "isize",
    "ptr<i32>",
];

/// A function `@f` with the parameters `params` (as written in text) and the body `body`, which
/// ends in `builtin.ret;`.
fn function_text(params: &str, body: &str) -> String {
    format!("builtin.function public extern(\"C\") @f({params}) {{\n{body}\n    builtin.ret;\n}};")
}

/// `text`, a function, parsed and verified: its printed text, or the error with its labels.
fn check(text: &str) -> Result<String, String> {
    let context = context();
    let (function, printed) =
        parse_function_fixpoint(&context, "f", text).map_err(|report| describe(&report))?;
    function
        .borrow()
        .as_operation()
        .verify()
        .map_err(|report| format!("does not verify: {}", describe(&report)))?;
    Ok(printed)
}

/// The error of the function with the parameters `params` and the body `body`; it must not verify.
fn rejection(params: &str, body: &str) -> String {
    match check(&function_text(params, body)) {
        Ok(printed) => panic!("accepted:\n{printed}"),
        Err(error) => error,
    }
}

/// The integer types of Wasm.
const INT: &[&str] = &["i32", "i64"];
/// The floating-point types of Wasm.
const FLOAT: &[&str] = &["f32", "f64"];
/// The number types of Wasm.
const NUM: &[&str] = &["i32", "i64", "f32", "f64"];

/// The operations of two operands of one type and a result of that type, each with the types it
/// accepts.
const BINARY: &[(&str, &[&str])] = &[
    ("add", NUM),
    ("sub", NUM),
    ("mul", NUM),
    ("div_s", INT),
    ("div_u", INT),
    ("rem_s", INT),
    ("rem_u", INT),
    ("and", INT),
    ("or", INT),
    ("xor", INT),
    ("shl", INT),
    ("shr_s", INT),
    ("shr_u", INT),
    ("rotl", INT),
    ("rotr", INT),
    ("div", FLOAT),
    ("min", FLOAT),
    ("max", FLOAT),
    ("copysign", FLOAT),
];

/// The operations of one operand and a result of its type, each with the types it accepts.
const UNARY: &[(&str, &[&str])] = &[
    ("clz", INT),
    ("ctz", INT),
    ("popcnt", INT),
    ("abs", FLOAT),
    ("neg", FLOAT),
    ("sqrt", FLOAT),
    ("ceil", FLOAT),
    ("floor", FLOAT),
    ("trunc", FLOAT),
    ("nearest", FLOAT),
    ("extend8_s", INT),
    ("extend16_s", INT),
    ("extend32_s", &["i64"]),
];

/// The comparisons of two operands of one type and an `i32` result, each with the types it
/// accepts.
const COMPARE: &[(&str, &[&str])] = &[
    ("eq", NUM),
    ("ne", NUM),
    ("lt_s", INT),
    ("lt_u", INT),
    ("gt_s", INT),
    ("gt_u", INT),
    ("le_s", INT),
    ("le_u", INT),
    ("ge_s", INT),
    ("ge_u", INT),
    ("lt", FLOAT),
    ("gt", FLOAT),
    ("le", FLOAT),
    ("ge", FLOAT),
];

/// The conversions, each with the (operand type, result type) pairs it has.
const CONVERSIONS: &[(&str, &[(&str, &str)])] = &[
    ("wrap", &[("i64", "i32")]),
    ("extend_s", &[("i32", "i64")]),
    ("extend_u", &[("i32", "i64")]),
    ("trunc_s", &[("f32", "i32"), ("f32", "i64"), ("f64", "i32"), ("f64", "i64")]),
    ("trunc_u", &[("f32", "i32"), ("f32", "i64"), ("f64", "i32"), ("f64", "i64")]),
    ("trunc_sat_s", &[("f32", "i32"), ("f32", "i64"), ("f64", "i32"), ("f64", "i64")]),
    ("trunc_sat_u", &[("f32", "i32"), ("f32", "i64"), ("f64", "i32"), ("f64", "i64")]),
    ("convert_s", &[("i32", "f32"), ("i32", "f64"), ("i64", "f32"), ("i64", "f64")]),
    ("convert_u", &[("i32", "f32"), ("i32", "f64"), ("i64", "f32"), ("i64", "f64")]),
    ("promote", &[("f32", "f64")]),
    ("demote", &[("f64", "f32")]),
    ("reinterpret", &[("i32", "f32"), ("f32", "i32"), ("i64", "f64"), ("f64", "i64")]),
];

/// A function `@f` with every numeric operation at every type it accepts, each applied to
/// parameters of its operand types; the results are unused. Constants are included.
fn every_operation() -> String {
    let mut params = Vec::new();
    for (index, ty) in NUM.iter().enumerate() {
        params.push(format!("%{ty}1: {ty}"));
        params.push(format!("%{ty}2: {ty}"));
        let _ = index;
    }
    params.push(String::from("%f: !wasm.funcref"));
    params.push(String::from("%g: !wasm.funcref"));
    params.push(String::from("%x: !wasm.externref"));
    let mut body = String::new();
    for ty in INT {
        body += &format!("    %c{ty} = wasm.const 7 : {ty};\n");
    }
    for ty in FLOAT {
        body += &format!("    %c{ty} = wasm.constf 1.5 : {ty};\n");
    }
    for (op, types) in BINARY.iter().chain(COMPARE) {
        for ty in *types {
            body += &format!("    %{op}_{ty} = wasm.{op} %{ty}1, %{ty}2;\n");
        }
    }
    for (op, types) in UNARY {
        for ty in *types {
            body += &format!("    %{op}_{ty} = wasm.{op} %{ty}1;\n");
        }
    }
    for ty in INT {
        body += &format!("    %eqz_{ty} = wasm.eqz %{ty}1;\n");
    }
    for (op, pairs) in CONVERSIONS {
        for (from, to) in *pairs {
            body += &format!("    %{op}_{from}_{to} = wasm.{op} %{from}1 <{{ ty = {to} }}>;\n");
        }
    }
    for (ty, a, b) in [
        ("i32", "%i321", "%i322"),
        ("i64", "%i641", "%i642"),
        ("f32", "%f321", "%f322"),
        ("f64", "%f641", "%f642"),
        ("funcref", "%f", "%g"),
        ("externref", "%x", "%x"),
    ] {
        body += &format!("    %select_{ty} = wasm.select %i321, {a}, {b};\n");
    }
    function_text(&params.join(", "), &body)
}

/// The operations of the entry block of `function`, without the `builtin` ones.
fn wasm_ops(function: FunctionRef) -> Vec<OperationRef> {
    let function = function.borrow();
    let body = function.body();
    let entry = body.entry();
    entry
        .body()
        .iter()
        .map(|op| op.as_operation_ref())
        .filter(|op| op.borrow().name().to_string().starts_with("wasm."))
        .collect()
}

/// The function with the parameters `params` and the body `body`, parsed.
fn parse_function(params: &str, body: &str) -> (Rc<Context>, FunctionRef) {
    let context = context();
    let config = ParserConfig::new(context.clone());
    let function = parse::parse::<runic_ir::dialects::builtin::Function>(
        config,
        "f",
        function_text(params, body),
    )
    .into_result()
    .unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, function)
}

/// The text of the result types of the first `wasm` operation of the function with the
/// parameters `params` and the body `body`.
fn result_types(params: &str, body: &str) -> Vec<String> {
    let (_context, function) = parse_function(params, body);
    let op = wasm_ops(function)[0];
    let op = op.borrow();
    op.results()
        .iter()
        .map(|result| runic_ir::Value::ty(&*result.borrow()).to_string())
        .collect()
}

/// The function `@f(params) -> (results)` with the body `body`, in a module, after `pass` ran on
/// it with verification, as text.
fn run_pass(
    pass: Box<dyn runic_ir::pass::OperationPass>,
    params: &str,
    results: &str,
    body: &str,
) -> Result<String, Report> {
    let context = context();
    let text = format!(
        "builtin.module public @m {{\nbuiltin.function public extern(\"C\") @f({params}) -> \
         ({results}) {{\n{body}\n}};\n}};"
    );
    let module =
        parse::parse::<Module>(ParserConfig::new(context.clone()), "wasm", text).into_result()?;
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let mut passes =
        PassManager::on::<runic_ir::dialects::builtin::Function>(context, Nesting::Implicit);
    passes.add_pass(pass).unwrap();
    passes.enable_verifier(true);
    passes.run(function)?;
    Ok(function.borrow().to_string())
}

/// `body` after dead-code elimination.
fn dce(params: &str, results: &str, body: &str) -> Result<String, Report> {
    run_pass(Box::<DeadCodeElimination>::default(), params, results, body)
}

/// `body` after common-subexpression elimination.
fn cse(params: &str, results: &str, body: &str) -> Result<String, Report> {
    run_pass(Box::<CommonSubexpressionElimination>::default(), params, results, body)
}
