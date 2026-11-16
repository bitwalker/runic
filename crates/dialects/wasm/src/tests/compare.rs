//! The comparisons: operations on numbers whose result is an `i32`.

use alloc::{format, vec::Vec};

use runic_ir::{
    Op, SourceSpan, Value, ValueRef, dialects::builtin::BuiltinOpBuilder, testing::Test,
};

use super::*;

const SPAN: SourceSpan = SourceSpan::UNKNOWN;

#[test]
fn a_comparison_accepts_exactly_the_types_its_wasm_instructions_have() {
    for (op, allowed) in COMPARE {
        for ty in TYPES {
            let result = check(&function_text(
                &format!("%a: {ty}, %b: {ty}"),
                &format!("%r = wasm.{op} %a, %b;"),
            ));
            assert_eq!(result.is_ok(), allowed.contains(&ty), "wasm.{op} on {ty}: {result:?}");
        }
    }
    for ty in TYPES {
        let result = check(&function_text(&format!("%a: {ty}"), "%r = wasm.eqz %a;"));
        assert_eq!(result.is_ok(), INT.contains(&ty), "wasm.eqz on {ty}: {result:?}");
    }
}

#[test]
fn the_result_of_a_comparison_is_an_i32_whatever_the_operands() {
    for (op, allowed) in COMPARE {
        for ty in *allowed {
            let types =
                result_types(&format!("%a: {ty}, %b: {ty}"), &format!("%r = wasm.{op} %a, %b;"));
            assert_eq!(types, ["i32"], "wasm.{op} on {ty}");
        }
    }
    for ty in INT {
        assert_eq!(result_types(&format!("%a: {ty}"), "%r = wasm.eqz %a;"), ["i32"], "eqz on {ty}");
    }
}

#[test]
fn a_comparison_rejects_operands_of_different_types() {
    for (op, _) in COMPARE {
        for (lhs, rhs) in [("i32", "i64"), ("f32", "f64"), ("i32", "f32"), ("i64", "i8")] {
            let error =
                rejection(&format!("%a: {lhs}, %b: {rhs}"), &format!("%r = wasm.{op} %a, %b;"));
            assert!(
                error.contains(&format!("`wasm.{op}` takes two operands of one type")),
                "{error}"
            );
            assert!(error.contains("and gives an `i32`"), "{error}");
        }
    }
}

#[test]
fn the_messages_of_a_comparison_name_its_rule() {
    let error = rejection("%a: f32, %b: f32", "%r = wasm.lt_s %a, %b;");
    assert!(
        error.contains(
            "`wasm.lt_s` takes two operands of one type, `i32` or `i64`, and gives an `i32`, but \
             its first operand has type `f32`"
        ),
        "{error}"
    );
    let error = rejection("%a: i32, %b: i64", "%r = wasm.eq %a, %b;");
    assert!(error.contains("but its operands have the types `i32` and `i64`"), "{error}");
    let error = rejection("%a: f32", "%r = wasm.eqz %a;");
    assert!(
        error.contains(
            "`wasm.eqz` takes one operand, `i32` or `i64`, and gives an `i32`, but its operand \
             has type `f32`"
        ),
        "{error}"
    );
}

#[test]
fn a_comparison_whose_result_is_not_an_i32_does_not_verify() {
    // The parser infers the result and rejects another type.
    let error = rejection("%a: i64, %b: i64", "%r = \"wasm.eq\"(%a, %b) : (i64, i64) -> i64;");
    assert!(
        error.contains("this result is written as 'i64', but 'wasm.eq' gives it 'i32'"),
        "{error}"
    );
    // A result changed after parsing is caught by the verifier.
    for (params, body, op) in [
        ("%a: i64, %b: i64", "%r = wasm.eq %a, %b;", "eq"),
        ("%a: f64, %b: f64", "%r = wasm.ge %a, %b;", "ge"),
        ("%a: i64", "%r = wasm.eqz %a;", "eqz"),
    ] {
        let (context, function) = parse_function(params, body);
        let mut operation = wasm_ops(function)[0];
        operation.borrow_mut().results_mut()[0].borrow_mut().set_type(context.i64());
        let error = describe(&operation.borrow().verify().expect_err(op));
        assert!(error.contains("but its result has type `i64`, not `i32`"), "{op}: {error}");
    }
}

#[test]
fn the_comparisons_that_wasm_has_for_both_integers_and_floats_are_eq_and_ne() {
    for ty in NUM {
        for op in ["eq", "ne"] {
            assert!(
                check(&function_text(
                    &format!("%a: {ty}, %b: {ty}"),
                    &format!("%r = wasm.{op} %a, %b;")
                ))
                .is_ok()
            );
        }
    }
    // `lt` has no integer form and `lt_s` no float form, as in Wasm.
    assert!(check(&function_text("%a: i32, %b: i32", "%r = wasm.lt %a, %b;")).is_err());
    assert!(check(&function_text("%a: f32, %b: f32", "%r = wasm.lt_s %a, %b;")).is_err());
}

#[test]
fn the_builder_builds_each_comparison_with_an_i32_result() {
    let mut test = Test::new("f", |cx| {
        (
            [cx.i32(), cx.i32(), cx.i64(), cx.i64(), cx.f32(), cx.f32(), cx.f64(), cx.f64()],
            [],
        )
    });
    test.context().get_or_register_dialect::<WasmDialect>();
    let params = test
        .entry_block()
        .borrow()
        .arguments()
        .iter()
        .map(|arg| arg.borrow().as_value_ref())
        .collect::<Vec<ValueRef>>();
    let mut b = test.function_builder();
    let mut built: Vec<(&str, ValueRef)> = Vec::new();
    macro_rules! compare {
        ($lhs:expr, $rhs:expr; $($method:ident),*) => {
            $(built.push((stringify!($method), b.$method($lhs, $rhs, SPAN).unwrap()));)*
        };
    }
    for (lhs, rhs) in [(params[0], params[1]), (params[2], params[3])] {
        compare!(lhs, rhs; eq, ne, lt_s, lt_u, gt_s, gt_u, le_s, le_u, ge_s, ge_u);
        built.push(("eqz", b.eqz(lhs, SPAN).unwrap()));
    }
    for (lhs, rhs) in [(params[4], params[5]), (params[6], params[7])] {
        compare!(lhs, rhs; eq, ne, lt, gt, le, ge);
    }
    let i32 = test.context().i32();
    let text = format!("{}", test.function().borrow().as_operation());
    for (method, value) in &built {
        assert_eq!(value.borrow().ty(), i32, "{method}");
        assert!(text.contains(&format!("wasm.{method} ")), "{method}: {text}");
    }
    assert_eq!(built.len(), 2 * 11 + 2 * 6);
    test.function_builder().ret([], SPAN).unwrap();
    test.function().borrow().as_operation().verify().expect("valid function");
    let printed = format!("{}", test.function().borrow().as_operation());
    let (_, reprinted) = parse_function_fixpoint(&context(), "f", &printed).unwrap();
    assert_eq!(reprinted, printed);
}
