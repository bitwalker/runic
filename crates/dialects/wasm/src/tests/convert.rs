//! The conversions, and the sign extensions in place.

use alloc::{format, vec, vec::Vec};

use runic_ir::{
    Context, Op, SourceSpan, Type, Value, ValueRef, dialects::builtin::BuiltinOpBuilder,
    testing::Test,
};

use super::*;

const SPAN: SourceSpan = SourceSpan::UNKNOWN;

#[test]
fn a_conversion_accepts_exactly_the_pairs_of_types_its_wasm_instructions_have() {
    for (op, pairs) in CONVERSIONS {
        for from in NUM {
            for to in NUM {
                let result = check(&function_text(
                    &format!("%a: {from}"),
                    &format!("%r = wasm.{op} %a <{{ ty = {to} }}>;"),
                ));
                assert_eq!(
                    result.is_ok(),
                    pairs.contains(&(*from, *to)),
                    "wasm.{op} from {from} to {to}: {result:?}"
                );
            }
        }
    }
}

#[test]
fn a_conversion_rejects_an_operand_or_a_result_that_is_not_a_number() {
    for (op, _) in CONVERSIONS {
        for ty in TYPES.iter().filter(|ty| !NUM.contains(ty)) {
            for good in NUM {
                let operand = check(&function_text(
                    &format!("%a: {ty}"),
                    &format!("%r = wasm.{op} %a <{{ ty = {good} }}>;"),
                ));
                assert!(operand.is_err(), "wasm.{op} of {ty}");
                let result = check(&function_text(
                    &format!("%a: {good}"),
                    &format!("%r = wasm.{op} %a <{{ ty = {ty} }}>;"),
                ));
                assert!(result.is_err(), "wasm.{op} to {ty}");
            }
        }
    }
}

#[test]
fn the_result_of_a_conversion_has_the_type_the_text_writes() {
    for (op, pairs) in CONVERSIONS {
        for (from, to) in *pairs {
            let types = result_types(
                &format!("%a: {from}"),
                &format!("%r = wasm.{op} %a <{{ ty = {to} }}>;"),
            );
            assert_eq!(types, [*to], "wasm.{op} from {from}");
        }
    }
}

#[test]
fn a_conversion_prints_its_result_type_as_a_property_and_reads_it_back() {
    let printed = check(&function_text(
        "%a: f32",
        "%r = wasm.trunc_sat_u %a <{ ty = i64 }>;\n    %s = wasm.trunc_sat_u %a <{ ty = \
         #builtin.type<i32> }>;",
    ))
    .unwrap();
    assert!(
        printed.contains("wasm.trunc_sat_u %0 <{ ty = #builtin.type<i64> }>;"),
        "{printed}"
    );
    assert!(
        printed.contains("wasm.trunc_sat_u %0 <{ ty = #builtin.type<i32> }>;"),
        "{printed}"
    );
}

#[test]
fn the_messages_of_a_conversion_name_the_pairs_it_has() {
    let error = rejection("%a: i32", "%r = wasm.wrap %a <{ ty = i32 }>;");
    assert!(
        error.contains("`wasm.wrap` converts an `i64` to an `i32`, but its operand has type `i32`"),
        "{error}"
    );
    let error = rejection("%a: i64", "%r = wasm.wrap %a <{ ty = i64 }>;");
    assert!(
        error.contains(
            "`wasm.wrap` converts an `i64` to an `i32`, but `i64` to `i64` is not one of them"
        ),
        "{error}"
    );
    let error = rejection("%a: i32", "%r = wasm.extend_s %a <{ ty = i8 }>;");
    assert!(error.contains("`wasm.extend_s` converts an `i32` to an `i64`, but"), "{error}");
    let error = rejection("%a: i32", "%r = wasm.trunc_s %a <{ ty = i32 }>;");
    assert!(
        error.contains(
            "`wasm.trunc_s` converts a float (`f32` or `f64`) to an integer (`i32` or `i64`), but \
             its operand has type `i32`"
        ),
        "{error}"
    );
    let error = rejection("%a: i32", "%r = wasm.convert_u %a <{ ty = i64 }>;");
    assert!(
        error.contains(
            "`wasm.convert_u` converts an integer (`i32` or `i64`) to a float (`f32` or `f64`), \
             but `i32` to `i64` is not one of them"
        ),
        "{error}"
    );
    let error = rejection("%a: f32", "%r = wasm.reinterpret %a <{ ty = f64 }>;");
    assert!(
        error.contains(
            "`wasm.reinterpret` converts an `i32` to an `f32`, an `f32` to an `i32`, an `i64` to \
             an `f64` or an `f64` to an `i64`, but `f32` to `f64` is not one of them"
        ),
        "{error}"
    );
    let error = rejection("%a: f32", "%r = wasm.promote %a <{ ty = f32 }>;");
    assert!(
        error.contains(
            "`wasm.promote` converts an `f32` to an `f64`, but `f32` to `f32` is not one of them"
        ),
        "{error}"
    );
}

#[test]
fn a_conversion_whose_result_is_not_its_ty_property_does_not_verify() {
    let (context, function) = parse_function("%a: f64", "%r = wasm.trunc_s %a <{ ty = i32 }>;");
    let mut op = wasm_ops(function)[0];
    op.borrow_mut().results_mut()[0].borrow_mut().set_type(context.i64());
    let error = describe(&op.borrow().verify().expect_err("the result type is wrong"));
    assert!(
        error.contains("its result has type `i64`, not its `ty` property `i32`"),
        "{error}"
    );
}

#[test]
fn the_sign_extensions_in_place_have_the_widths_wasm_gives_them() {
    for op in ["extend8_s", "extend16_s"] {
        for ty in TYPES {
            let result =
                check(&function_text(&format!("%a: {ty}"), &format!("%r = wasm.{op} %a;")));
            assert_eq!(result.is_ok(), INT.contains(&ty), "wasm.{op} on {ty}");
        }
    }
    for ty in TYPES {
        let result = check(&function_text(&format!("%a: {ty}"), "%r = wasm.extend32_s %a;"));
        assert_eq!(result.is_ok(), ty == "i64", "wasm.extend32_s on {ty}: {result:?}");
    }
    let error = rejection("%a: i32", "%r = wasm.extend32_s %a;");
    assert!(error.contains("`wasm.extend32_s` takes one operand, `i64`"), "{error}");
}

#[test]
fn the_two_truncations_are_different_operations() {
    // `wasm.trunc` rounds a float to a float; `wasm.trunc_s` converts a float to an integer.
    assert_eq!(result_types("%a: f32", "%r = wasm.trunc %a;"), ["f32"]);
    assert_eq!(result_types("%a: f32", "%r = wasm.trunc_s %a <{ ty = i32 }>;"), ["i32"]);
    assert!(check(&function_text("%a: f32", "%r = wasm.trunc %a <{ ty = i32 }>;")).is_err());
    assert!(check(&function_text("%a: f32", "%r = wasm.trunc_s %a;")).is_err());
}

#[test]
fn the_builder_builds_each_conversion_with_its_result_type() {
    let mut test = Test::new("f", |cx| ([cx.i32(), cx.i64(), cx.f32(), cx.f64()], []));
    test.context().get_or_register_dialect::<WasmDialect>();
    let params = test
        .entry_block()
        .borrow()
        .arguments()
        .iter()
        .map(|arg| arg.borrow().as_value_ref())
        .collect::<Vec<ValueRef>>();
    let [i32v, i64v, f32v, f64v] = params[..] else {
        unreachable!()
    };
    let cx = test.context_rc();
    let mut b = test.function_builder();
    let mut built: Vec<(&str, ValueRef, Type)> = vec![
        ("wrap", b.wrap(i64v, SPAN).unwrap(), cx.i32()),
        ("extend_s", b.extend_s(i32v, SPAN).unwrap(), cx.i64()),
        ("extend_u", b.extend_u(i32v, SPAN).unwrap(), cx.i64()),
        ("promote", b.promote(f32v, SPAN).unwrap(), cx.f64()),
        ("demote", b.demote(f64v, SPAN).unwrap(), cx.f32()),
    ];
    for float in [f32v, f64v] {
        for ty in [cx.i32(), cx.i64()] {
            built.push(("trunc_s", b.trunc_s(float, ty, SPAN).unwrap(), ty));
            built.push(("trunc_u", b.trunc_u(float, ty, SPAN).unwrap(), ty));
            built.push(("trunc_sat_s", b.trunc_sat_s(float, ty, SPAN).unwrap(), ty));
            built.push(("trunc_sat_u", b.trunc_sat_u(float, ty, SPAN).unwrap(), ty));
        }
    }
    for int in [i32v, i64v] {
        for ty in [cx.f32(), cx.f64()] {
            built.push(("convert_s", b.convert_s(int, ty, SPAN).unwrap(), ty));
            built.push(("convert_u", b.convert_u(int, ty, SPAN).unwrap(), ty));
        }
    }
    built.push(("reinterpret", b.reinterpret(i32v, cx.f32(), SPAN).unwrap(), cx.f32()));
    built.push(("reinterpret", b.reinterpret(f32v, cx.i32(), SPAN).unwrap(), cx.i32()));
    built.push(("reinterpret", b.reinterpret(i64v, cx.f64(), SPAN).unwrap(), cx.f64()));
    built.push(("reinterpret", b.reinterpret(f64v, cx.i64(), SPAN).unwrap(), cx.i64()));
    let text = format!("{}", test.function().borrow().as_operation());
    for (method, value, ty) in &built {
        assert_eq!(value.borrow().ty(), *ty, "{method}");
        assert!(text.contains(&format!("wasm.{method} ")), "{method}: {text}");
    }
    assert_eq!(built.len(), 5 + 16 + 8 + 4);
    test.function_builder().ret([], SPAN).unwrap();
    test.function().borrow().as_operation().verify().expect("valid function");
    let printed = format!("{}", test.function().borrow().as_operation());
    let (_, reprinted) = parse_function_fixpoint(&context(), "f", &printed).unwrap();
    assert_eq!(reprinted, printed);
}

#[test]
fn a_built_conversion_between_types_it_does_not_have_does_not_verify() {
    let mut test = Test::new("f", |cx| ([cx.i32()], []));
    test.context().get_or_register_dialect::<WasmDialect>();
    let param = test.entry_block().borrow().arguments()[0].borrow().as_value_ref();
    let f64 = test.context().f64();
    let mut b = test.function_builder();
    // `i32` to `f64` is a `convert_s`, not a `reinterpret`.
    let value = b.reinterpret(param, f64, SPAN).unwrap();
    let op = value.borrow().get_defining_op().unwrap();
    let error = describe(&op.borrow().verify().expect_err("not a pair reinterpret has"));
    assert!(error.contains("`i32` to `f64` is not one of them"), "{error}");
    let _: &Context = test.context();
}
