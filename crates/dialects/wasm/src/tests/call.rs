//! The indirect calls: `call_indirect` and `return_call_indirect`.

use runic_ir::{
    Builder, Ident, SourceSpan, Value, ValueRef, Visibility,
    dialects::builtin::attributes::Signature,
    effects::MemoryEffectOpInterface,
    testing::{Test, text::round_trip},
};

use super::*;

/// The function of [STATE_PARAMS] with the body `body` and then `builtin.ret;`, parsed: the error,
/// which must contain `message`.
#[track_caller]
fn rejected_call(body: &str, message: &str) {
    rejected(body, message);
}

/// The text of a module with the function `@f(params) -> (results)` whose body is `body`, after
/// the entities of [STATE], parsed and verified.
fn function(params: &str, results: &str, body: &str) -> Result<(), String> {
    parse_state(&state_module(params, results, body)).map(|_| ())
}

#[test]
fn an_indirect_call_calls_through_a_table_with_a_signature_and_gives_its_results() {
    let (_context, module) = accepted(
        "wasm.call_indirect @tab, %a() : extern(\"C\") () -> ();
         wasm.call_indirect @tab, %a(%b) : extern(\"C\") (i32);
         %one = wasm.call_indirect @tab, %a(%b, %l) : extern(\"C\") (i32, i64) -> i32;
         %p1, %p2 = wasm.call_indirect @tab, %a(%f, %d) : extern(\"C\") (f32, f64) -> (i64, \
         !wasm.funcref);
         %e = wasm.call_indirect @tab, %a(%r, %x) : extern(\"C\") (!wasm.funcref, !wasm.externref) \
         -> !wasm.externref;",
    );
    let text = module.borrow().as_operation().to_string();
    for part in [
        "wasm.call_indirect @tab, %5() : extern(\"C\") ();",
        "wasm.call_indirect @tab, %5(%6) : extern(\"C\") (i32);",
        "= wasm.call_indirect @tab, %5(%6, %10) : extern(\"C\") (i32, i64) -> i32;",
        "= wasm.call_indirect @tab, %5(%11, %12) : extern(\"C\") (f32, f64) -> (i64, \
         !wasm.funcref);",
        "-> !wasm.externref;",
    ] {
        assert!(text.contains(part), "`{part}` is not in\n{text}");
    }
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

#[test]
fn an_indirect_tail_call_is_a_terminator_with_no_results_that_round_trips() {
    let result = parse_state(&state_module(
        "%i: i32, %v: i32",
        "i32",
        "wasm.return_call_indirect @tab, %i(%v) : extern(\"C\") (i32) -> i32;",
    ));
    let (_context, module) = result.unwrap_or_else(|error| panic!("{error}"));
    let text = module.borrow().as_operation().to_string();
    assert!(text.contains("wasm.return_call_indirect @tab, %"), "{text}");
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

#[test]
fn an_indirect_call_goes_through_a_funcref_table_by_an_i32_index() {
    rejected_call(
        "wasm.call_indirect @mem, %a() : extern(\"C\") ();",
        "`wasm.call_indirect` names `mem`, which is a `wasm.memory`, not a `wasm.table`",
    );
    rejected_call(
        "wasm.call_indirect @refs, %a() : extern(\"C\") ();",
        "`wasm.call_indirect` calls a function through a `funcref` table and an `i32` index, \
         passes the arguments of its signature and gives its results, all of Wasm value types, \
         but the table `refs` holds `externref`, and an indirect call goes through a table of \
         `funcref`",
    );
    let error = parse_state(&state_module(
        STATE_PARAMS,
        "",
        "wasm.call_indirect @tab, %l() : extern(\"C\") ();\nbuiltin.ret;",
    ))
    .err()
    .expect("an i64 index");
    assert!(
        error.contains("but its index has type `i64`") || error.contains("i64"),
        "{error}"
    );
    rejected_call("wasm.return_call_indirect @refs, %a() : extern(\"C\") ();", "holds `externref`");
}

#[test]
fn a_signature_of_an_indirect_call_has_wasm_value_types() {
    for ty in ["i8", "i1", "isize", "ptr<i32>"] {
        let kind = |text: &str| {
            parse_state(&state_module(
                &format!("%i: i32, %v: {ty}"),
                "",
                &format!("{text}\nbuiltin.ret;"),
            ))
            .err()
            .unwrap_or_else(|| panic!("accepted {text}"))
        };
        let error = kind(&format!("wasm.call_indirect @tab, %i(%v) : extern(\"C\") ({ty});"));
        assert!(
            error.contains(&format!(
                "parameter 0 of its signature has type `{ty}`, but a Wasm value type is `i32`, \
                 `i64`, `f32`, `f64`, `funcref` or `externref`"
            )),
            "{error}"
        );
        let error =
            kind(&format!("%r = wasm.call_indirect @tab, %i(%v) : extern(\"C\") ({ty}) -> i32;"));
        assert!(error.contains("parameter 0 of its signature"), "{error}");
    }
    let error = parse_state(&state_module(
        "%i: i32",
        "",
        "%r = wasm.call_indirect @tab, %i() : extern(\"C\") () -> i8;\nbuiltin.ret;",
    ))
    .err()
    .expect("an i8 result");
    assert!(
        error.contains(
            "result 0 of its signature has type `i8`, but a Wasm value type is `i32`, `i64`, \
             `f32`, `f64`, `funcref` or `externref`"
        ),
        "{error}"
    );
    let error = parse_state(&state_module(
        "%i: i32",
        "",
        "%r1, %r2 = wasm.call_indirect @tab, %i() : extern(\"C\") () -> (i32, i8);\nbuiltin.ret;",
    ))
    .err()
    .expect("an i8 result");
    assert!(error.contains("result 1 of its signature has type `i8`"), "{error}");
}

#[test]
fn the_arguments_and_results_of_an_indirect_call_agree_with_its_signature() {
    // The custom form types the arguments by the signature: a wrong argument is a parse error
    let error = parse_state(&state_module(
        STATE_PARAMS,
        "",
        "wasm.call_indirect @tab, %a(%l) : extern(\"C\") (i32);\nbuiltin.ret;",
    ))
    .err()
    .expect("an argument of the wrong type");
    assert!(error.contains("i64") && error.contains("i32"), "{error}");
    // The generic form reaches the verifier
    let signature = "#builtin.signature<extern(\"C\") (i32, i64) -> i32>";
    let call = |arguments: &str, types: &str, results: &str, result_types: &str| {
        parse_state(&state_module(
            STATE_PARAMS,
            "",
            &format!(
                "{results}\"wasm.call_indirect\"(%a{arguments}) <{{ table = \
                 #builtin.symbol<@tab>, signature = {signature} }}> : (i32{types}) -> \
                 {result_types};\nbuiltin.ret;"
            ),
        ))
        .err()
        .unwrap_or_else(|| panic!("accepted"))
    };
    let error = call(", %b", " | i32", "%v = ", "i32");
    assert!(
        error.contains("it passes 1 argument(s), but its signature has 2 parameter(s)"),
        "{error}"
    );
    let error = call(", %b, %b", " | i32, i32", "%v = ", "i32");
    assert!(
        error
            .contains("argument 1 has type `i32`, but parameter 1 of its signature has type `i64`"),
        "{error}"
    );
    let error = call(", %b, %l", " | i32, i64", "", "()");
    assert!(
        // The parser makes the results from the signature, so it is the one to notice
        error.contains("this gives 0 result type(s), but 'wasm.call_indirect' has 1 result(s)"),
        "{error}"
    );
    let error = call(", %b, %l", " | i32, i64", "%v = ", "i64");
    assert!(
        error.contains("result 0 has type `i64`, but result 0 of its signature has type `i32`"),
        "{error}"
    );
}

/// `body`, the body of a function `@f(%i: i32, %v: i32) -> (results)` that ends in an indirect tail
/// call: the error. `signature` is the signature of the call.
fn tail_call_error(results: &str, signature: &str) -> String {
    parse_state(&state_module(
        "%i: i32, %v: i32",
        results,
        &format!("wasm.return_call_indirect @tab, %i(%v) : {signature};"),
    ))
    .err()
    .unwrap_or_else(|| panic!("accepted {results} {signature}"))
}

#[test]
fn an_indirect_tail_call_declares_the_results_of_its_function() {
    let error = tail_call_error("i64", "extern(\"C\") (i32) -> i32");
    assert!(
        error.contains("this tail call returns (i32), but the function returns (i64)"),
        "{error}"
    );
    assert!(error.contains("invalid operation builtin.function"), "{error}");
    assert!(error.contains("in this function"), "{error}");
    let error = tail_call_error("", "extern(\"C\") (i32) -> i32");
    assert!(
        error.contains("this tail call returns (i32), but the function returns ()"),
        "{error}"
    );
    let error = tail_call_error("i32, i32", "extern(\"C\") (i32) -> i32");
    assert!(
        error.contains("this tail call returns (i32), but the function returns (i32, i32)"),
        "{error}"
    );
    let error = tail_call_error("i32", "extern(\"C\") (i32) -> (i32, i32)");
    assert!(
        error.contains("this tail call returns (i32, i32), but the function returns (i32)"),
        "{error}"
    );
    function(
        "%i: i32, %v: i32",
        "i32, i32",
        "wasm.return_call_indirect @tab, %i(%v) : extern(\"C\") (i32) -> (i32, i32);",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    function(
        "%i: i32, %v: i32",
        "",
        "wasm.return_call_indirect @tab, %i(%v) : extern(\"C\") (i32);",
    )
    .unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn an_indirect_tail_call_uses_the_calling_convention_of_its_function() {
    let error = tail_call_error("i32", "extern(\"wasm\") (i32) -> i32");
    assert!(
        error.contains(
            "this tail call uses the calling convention \\\"wasm\\\", but the function uses \
             \\\"C\\\""
        ),
        "{error}"
    );
}

#[test]
fn an_indirect_tail_call_is_a_terminator_the_function_verifier_checks_for_exits() {
    // Not the last operation of its block
    let error = parse_state(&state_module(
        "%i: i32, %v: i32",
        "i32",
        "wasm.return_call_indirect @tab, %i(%v) : extern(\"C\") (i32) -> i32;\n builtin.ret %v : \
         (i32);",
    ))
    .err()
    .expect("a terminator in the middle of a block");
    assert!(!error.is_empty());
    // An operation with a result in the generic form
    let error = parse_state(&state_module(
        "%i: i32, %v: i32",
        "i32",
        "%x = \"wasm.return_call_indirect\"(%i, %v) <{ table = #builtin.symbol<@tab>, signature = \
         #builtin.signature<extern(\"C\") (i32) -> i32> }> : (i32, i32) -> i32;\n builtin.ret %x \
         : (i32);",
    ))
    .err()
    .expect("a result");
    assert!(!error.is_empty());
    // The function verifier applies the rule about the operations an exit crosses to it as well:
    // the operation is `TailCallLike`
    let (_context, module) = parse_state(&state_module(
        "%i: i32, %v: i32",
        "i32",
        "wasm.return_call_indirect @tab, %i(%v) : extern(\"C\") (i32) -> i32;",
    ))
    .unwrap_or_else(|error| panic!("{error}"));
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let mut found = false;
    let _ = function.borrow().prewalk(|op: &runic_ir::Operation| {
        if op.name().to_string() == "wasm.return_call_indirect" {
            found = true;
            assert!(op.implements::<dyn runic_ir::traits::TailCallLike>());
            assert!(op.implements::<dyn runic_ir::TailCallOpInterface>());
            assert!(op.implements::<dyn runic_ir::traits::Terminator>());
            assert!(op.leaves_callable());
        }
        runic_ir::WalkResult::<()>::Continue(())
    });
    assert!(found);
}

#[test]
fn an_indirect_call_has_unknown_effects_so_nothing_merges_moves_or_removes_it() {
    let (_context, module) = accepted(
        "%v = wasm.call_indirect @tab, %a(%b) : extern(\"C\") (i32) -> i32;
         wasm.call_indirect @tab, %a() : extern(\"C\") ();",
    );
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let mut calls = 0;
    let _ = function.borrow().prewalk(|op: &runic_ir::Operation| {
        if op.name().to_string() == "wasm.call_indirect" {
            calls += 1;
            assert!(op.as_trait::<dyn MemoryEffectOpInterface>().is_none());
            assert!(!op.would_be_trivially_dead());
        }
        runic_ir::WalkResult::<()>::Continue(())
    });
    assert_eq!(calls, 2);
    // Two equal calls are two calls, and an unused one stays
    let text = state_cse(
        "i32",
        "%l1 = wasm.call_indirect @tab, %a(%b) : extern(\"C\") (i32) -> i32;
         %l2 = wasm.call_indirect @tab, %a(%b) : extern(\"C\") (i32) -> i32;
         %u = wasm.add %l1, %l2;
         builtin.ret %u : (i32);",
    );
    assert_eq!(count(&text, "wasm.call_indirect"), 2);
    let text = state_dce("%l1 = wasm.call_indirect @tab, %a(%b) : extern(\"C\") (i32) -> i32;");
    assert_eq!(count(&text, "wasm.call_indirect"), 1);
}

#[test]
fn the_builders_build_both_indirect_calls() {
    let mut test = Test::named("f").in_module("m");
    let context = test.context_rc();
    context.get_or_register_dialect::<WasmDialect>();
    let (i32, i64) = (context.i32(), context.i64());
    let module = test.module();
    let body = module.borrow().body().entry_block_ref().unwrap();
    let ident = |name: &str| Ident::with_empty_span(name.to_string().into());
    let table = {
        let builder = test.builder_mut();
        builder.set_insertion_point_to_end(body);
        builder
            .create_table(
                ident("tab"),
                Visibility::Private,
                Limits::at_least(2),
                FuncRefType::get(&context),
            )
            .unwrap()
    };
    test.with_function("f", &[i32, i64], &[i64]);
    let (index, value) = {
        let entry = test.entry_block();
        let entry = entry.borrow();
        (entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef)
    };
    let builder = test.builder_mut();
    let span = SourceSpan::UNKNOWN;
    let signature = Signature::new(&context, [i64], [i64, i32]);
    let call = builder.call_through_table(table, index, signature, [value], span).unwrap();
    let results = call
        .borrow()
        .as_operation()
        .results()
        .iter()
        .map(|result| result.borrow().ty())
        .collect::<Vec<_>>();
    assert_eq!(results, [i64, i32]);
    let signature = Signature::new(&context, [i64], [i64]);
    builder
        .return_call_through_table(table, index, signature, [value], span)
        .unwrap();

    module.borrow().as_operation().recursively_verify().unwrap_or_else(|report| {
        panic!("{}\n{}", describe(&report), module.borrow().as_operation())
    });
    let text = module.borrow().as_operation().to_string();
    assert!(text.contains("wasm.call_indirect ::@m::@tab, %"), "{text}");
    assert!(text.contains("wasm.return_call_indirect ::@m::@tab, %"), "{text}");
    let mut skipped = Vec::new();
    let failures = round_trip::<Module>(module.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
