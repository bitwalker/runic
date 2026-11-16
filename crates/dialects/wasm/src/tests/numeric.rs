//! The constants, `select` and the arithmetic of the numeric operations.

use alloc::{format, string::ToString, vec::Vec};

use runic_ir::{
    Context, Op, SourceSpan, Value, ValueRef,
    dialects::builtin::BuiltinOpBuilder,
    testing::{
        Test,
        text::{describe, round_trip},
    },
};

use super::*;

const SPAN: SourceSpan = SourceSpan::UNKNOWN;

/// The text of a binary operation `op` on two parameters of type `ty`.
fn binary(op: &str, ty: &str) -> Result<alloc::string::String, alloc::string::String> {
    check(&function_text(
        &format!("%a: {ty}, %b: {ty}"),
        &format!("%r = wasm.{op} %a, %b;"),
    ))
}

#[test]
fn every_numeric_operation_has_the_name_of_its_wasm_instruction() {
    let text = every_operation();
    let printed = check(&text).unwrap_or_else(|error| panic!("{error}\n{text}"));
    for (op, types) in BINARY.iter().chain(COMPARE).chain(UNARY) {
        for ty in *types {
            assert!(printed.contains(&format!(" = wasm.{op} %")), "{op} on {ty}");
        }
    }
    for op in ["const", "constf", "select", "eqz"] {
        assert!(printed.contains(&format!(" = wasm.{op} ")), "{op}");
    }
    for (op, _) in CONVERSIONS {
        assert!(printed.contains(&format!(" = wasm.{op} %")), "{op}");
    }
}

#[test]
fn an_operation_of_two_operands_accepts_exactly_the_types_its_wasm_instructions_have() {
    for (op, allowed) in BINARY {
        for ty in TYPES {
            let result = binary(op, ty);
            assert_eq!(result.is_ok(), allowed.contains(&ty), "wasm.{op} on {ty}: {result:?}");
        }
    }
}

#[test]
fn an_operation_of_one_operand_accepts_exactly_the_types_its_wasm_instructions_have() {
    for (op, allowed) in UNARY {
        for ty in TYPES {
            let result =
                check(&function_text(&format!("%a: {ty}"), &format!("%r = wasm.{op} %a;")));
            assert_eq!(result.is_ok(), allowed.contains(&ty), "wasm.{op} on {ty}: {result:?}");
        }
    }
}

#[test]
fn an_operation_of_two_operands_rejects_operands_of_different_types() {
    for (op, allowed) in BINARY {
        for lhs in TYPES {
            for rhs in TYPES {
                if lhs == rhs {
                    continue;
                }
                let error =
                    rejection(&format!("%a: {lhs}, %b: {rhs}"), &format!("%r = wasm.{op} %a, %b;"));
                assert!(
                    error.contains(&format!("`wasm.{op}` takes two operands of one type")),
                    "{error}"
                );
                if allowed.contains(&lhs) && allowed.contains(&rhs) {
                    assert!(
                        error.contains(&format!(
                            "its operands have the types `{}` and `{}`",
                            val_type_name_of(lhs),
                            val_type_name_of(rhs)
                        )),
                        "{error}"
                    );
                }
            }
        }
    }
}

/// How the messages spell the type written as `text`.
fn val_type_name_of(text: &str) -> &str {
    text.trim_start_matches("!wasm.")
}

#[test]
fn the_message_for_an_operand_of_the_wrong_type_names_the_rule_and_the_type_written() {
    let error = rejection("%a: f32, %b: f32", "%r = wasm.shl %a, %b;");
    assert!(
        error.contains(
            "`wasm.shl` takes two operands of one type, `i32` or `i64`, and gives a result of \
             that type, but its first operand has type `f32`"
        ),
        "{error}"
    );
    let error = rejection("%a: i32, %b: f32", "%r = wasm.shl %a, %b;");
    assert!(
        error.contains("but its operands have the types `i32` and `f32`")
            || error.contains("but its second operand has type `f32`"),
        "{error}"
    );
    let error = rejection("%a: i8", "%r = wasm.clz %a;");
    assert!(
        error.contains(
            "`wasm.clz` takes one operand, `i32` or `i64`, and gives a result of its type, but \
             its operand has type `i8`"
        ),
        "{error}"
    );
    let error = rejection("%a: !wasm.funcref, %b: !wasm.funcref", "%r = wasm.add %a, %b;");
    assert!(error.contains("but its first operand has type `funcref`"), "{error}");
    let error = rejection("%a: isize, %b: isize", "%r = wasm.add %a, %b;");
    assert!(error.contains("but its first operand has type `isize`"), "{error}");
}

#[test]
fn an_operation_whose_result_is_not_of_its_operands_type_does_not_verify() {
    for (op, allowed) in BINARY {
        let ty = allowed[0];
        let other = if ty == "i32" { "i64" } else { "i32" };
        let text = function_text(
            &format!("%a: {ty}, %b: {ty}"),
            &format!("%r = \"wasm.{op}\"(%a, %b) : ({ty}, {ty}) -> {other};"),
        );
        let error = check(&text).expect_err(op);
        assert!(
            error.contains(&format!(
                "this result is written as '{other}', but 'wasm.{op}' gives it '{ty}'"
            )),
            "{op}: {error}"
        );
        // The generic form with the right result type is the same operation.
        let text = function_text(
            &format!("%a: {ty}, %b: {ty}"),
            &format!("%r = \"wasm.{op}\"(%a, %b) : ({ty}, {ty}) -> {ty};"),
        );
        let printed = check(&text).unwrap_or_else(|error| panic!("{op}: {error}"));
        assert!(printed.contains(&format!("%2 = wasm.{op} %0, %1;")), "{printed}");
    }
    let error = rejection("%a: i32", "%r = \"wasm.clz\"(%a) : (i32) -> f32;");
    assert!(
        error.contains("this result is written as 'f32', but 'wasm.clz' gives it 'i32'"),
        "{error}"
    );
}

/// The result type of the first operation of the function with the parameters and body given,
/// set to `ty` after parsing, and the error of verifying the function.
fn verify_with_result_type(params: &str, body: &str, ty: fn(&Context) -> runic_ir::Type) -> String {
    let (context, function) = parse_function(params, body);
    let mut op = wasm_ops(function)[0];
    op.borrow_mut().results_mut()[0].borrow_mut().set_type(ty(&context));
    describe(&op.borrow().verify().expect_err("the result type is wrong"))
}

#[test]
fn the_verifier_rejects_a_result_whose_type_was_changed_after_parsing() {
    let error = verify_with_result_type("%a: i32, %b: i32", "%r = wasm.add %a, %b;", Context::i64);
    assert!(
        error.contains(
            "`wasm.add` takes two operands of one type, `i32`, `i64`, `f32` or `f64`, and gives a \
             result of that type, but its result has type `i64`, not `i32`"
        ),
        "{error}"
    );
    let error = verify_with_result_type("%a: f64", "%r = wasm.neg %a;", Context::f32);
    assert!(error.contains("but its result has type `f32`, not `f64`"), "{error}");
    let error = verify_with_result_type(
        "%c: i32, %a: f32, %b: f32",
        "%r = wasm.select %c, %a, %b;",
        Context::i32,
    );
    assert!(error.contains("its result has type `i32`, not `f32`"), "{error}");
    let error = verify_with_result_type("", "%r = wasm.const 5 : i32;", Context::i64);
    assert!(
        error.contains(
            "`wasm.const` takes a value of type `i32` or `i64` and gives a result of that type, \
             but its result has type `i64`, not `i32`"
        ),
        "{error}"
    );
    let error = verify_with_result_type("", "%r = wasm.constf 5.0 : f32;", Context::f64);
    assert!(error.contains("but its result has type `f64`, not `f32`"), "{error}");
}

#[test]
fn the_result_of_an_operation_has_the_type_of_its_operands() {
    for (op, allowed) in BINARY {
        for ty in *allowed {
            let types =
                result_types(&format!("%a: {ty}, %b: {ty}"), &format!("%r = wasm.{op} %a, %b;"));
            assert_eq!(types, [*ty], "wasm.{op}");
        }
    }
    for (op, allowed) in UNARY {
        for ty in *allowed {
            let types = result_types(&format!("%a: {ty}"), &format!("%r = wasm.{op} %a;"));
            assert_eq!(types, [*ty], "wasm.{op}");
        }
    }
}

#[test]
fn constants_are_accepted_at_the_types_wasm_has_and_no_other() {
    for ty in TYPES {
        // The integer constant: the parser reads an integer type, the verifier says which.
        let int = check(&function_text("", &format!("%r = wasm.const 5 : {ty};")));
        assert_eq!(int.is_ok(), INT.contains(&ty), "wasm.const at {ty}: {int:?}");
        let float = check(&function_text("", &format!("%r = wasm.constf 1.5 : {ty};")));
        assert_eq!(float.is_ok(), FLOAT.contains(&ty), "wasm.constf at {ty}: {float:?}");
    }
}

#[test]
fn a_constant_prints_its_value_and_type_and_reads_hexadecimal_and_negative_values() {
    let printed = check(&function_text(
        "",
        "%a = wasm.const -1 : i32;\n    %b = wasm.const 0xff : i64;\n    %c = wasm.constf -0.5 : \
         f32;\n    %d = wasm.constf 0x7ff8000000000001 : f64;\n    %e = wasm.const 0xffffffff : \
         i32;",
    ))
    .unwrap();
    assert!(printed.contains("wasm.const -1 : i32;"), "{printed}");
    assert!(printed.contains("wasm.const 255 : i64;"), "{printed}");
    assert!(printed.contains("wasm.constf -0.5 : f32;"), "{printed}");
    assert!(printed.contains("wasm.constf 0x7ff8000000000001 : f64;"), "{printed}");
}

#[test]
fn a_constant_whose_result_is_not_the_type_of_its_value_does_not_verify() {
    let error = rejection("", "%r = wasm.const 5 : i8;");
    assert!(error.contains("expected `i32` or `i64`, got 'i8'"), "{error}");
    let error = rejection("", "%r = wasm.constf 5.0 : i32;");
    assert!(error.contains("expected `f32` or `f64`, got 'i32'"), "{error}");
}

#[test]
fn select_accepts_an_i32_condition_and_two_values_of_one_value_type() {
    let value_types = ["i32", "i64", "f32", "f64", "!wasm.funcref", "!wasm.externref"];
    for cond in TYPES {
        for a in TYPES {
            for b in TYPES {
                let ok = cond == "i32" && a == b && value_types.contains(&a);
                let result = check(&function_text(
                    &format!("%c: {cond}, %a: {a}, %b: {b}"),
                    "%r = wasm.select %c, %a, %b;",
                ));
                assert_eq!(result.is_ok(), ok, "wasm.select {cond}, {a}, {b}: {result:?}");
            }
        }
    }
}

#[test]
fn the_result_of_select_has_the_type_of_its_values() {
    for ty in ["i32", "i64", "f32", "f64", "!wasm.funcref", "!wasm.externref"] {
        let types =
            result_types(&format!("%c: i32, %a: {ty}, %b: {ty}"), "%r = wasm.select %c, %a, %b;");
        assert_eq!(types, [ty], "select of {ty}");
    }
}

#[test]
fn the_messages_of_select_name_its_rule() {
    let error = rejection("%c: i64, %a: i32, %b: i32", "%r = wasm.select %c, %a, %b;");
    assert!(
        error.contains(
            "`wasm.select` takes an `i32` condition and two values of one type, `i32`, `i64`, \
             `f32`, `f64`, `funcref` or `externref`, and gives a result of that type, but its \
             condition has type `i64`"
        ),
        "{error}"
    );
    let error = rejection("%c: i32, %a: i32, %b: f32", "%r = wasm.select %c, %a, %b;");
    assert!(error.contains("but its values have the types `i32` and `f32`"), "{error}");
    let error = rejection("%c: i32, %a: i8, %b: i8", "%r = wasm.select %c, %a, %b;");
    assert!(error.contains("but its first value has type `i8`"), "{error}");
    let error = rejection("%c: i32, %a: i32, %b: i8", "%r = wasm.select %c, %a, %b;");
    assert!(error.contains("but its second value has type `i8`"), "{error}");
    let error = rejection(
        "%c: i32, %a: i32, %b: i32",
        "%r = \"wasm.select\"(%c, %a, %b) : (i32, i32, i32) -> i64;",
    );
    assert!(
        error.contains("this result is written as 'i64', but 'wasm.select' gives it 'i32'"),
        "{error}"
    );
}

#[test]
fn every_operation_round_trips_in_the_custom_and_the_generic_form() {
    let text = every_operation();
    let (_context, function) = parse_function_from(&text);
    let mut skipped = Vec::new();
    let failures = round_trip::<runic_ir::dialects::builtin::Function>(
        function.as_operation_ref(),
        true,
        &mut skipped,
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

/// The function `text` defines, parsed and verified.
fn parse_function_from(
    text: &str,
) -> (alloc::rc::Rc<Context>, runic_ir::dialects::builtin::FunctionRef) {
    let context = context();
    let config = runic_ir::parse::ParserConfig::new(context.clone());
    let function =
        runic_ir::parse::parse::<runic_ir::dialects::builtin::Function>(config, "f", text)
            .into_result()
            .unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, function)
}

/// Every operation name the dialect registers is one of the tested names or a module-level
/// operation, and each tested name is registered.
#[test]
fn the_dialect_has_exactly_the_numeric_operations_the_tables_list() {
    let context = context();
    let dialect = context.get_or_register_dialect::<WasmDialect>();
    let mut registered = dialect
        .registered_ops()
        .iter()
        .map(|name| name.to_string())
        .filter(|name| {
            !["memory", "table", "global", "elem", "tag"]
                .map(|op| format!("wasm.{op}"))
                .contains(name)
        })
        .collect::<Vec<_>>();
    registered.sort();
    let mut expected = ["const", "constf", "select", "eqz"]
        .iter()
        .map(|op| format!("wasm.{op}"))
        .chain(BINARY.iter().chain(COMPARE).chain(UNARY).map(|(op, _)| format!("wasm.{op}")))
        .chain(CONVERSIONS.iter().map(|(op, _)| format!("wasm.{op}")))
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(registered, expected);
    assert_eq!(registered.len(), 62);
}

#[test]
fn is_constant_instruction_holds_for_constants_and_integer_add_sub_and_mul_only() {
    let text = every_operation();
    let (_context, function) = parse_function_from(&text);
    let mut seen = 0;
    for op in wasm_ops(function) {
        let op = op.borrow();
        let name = op.name().to_string();
        let ty = op.results()[0].borrow().ty();
        let expected = match name.as_str() {
            "wasm.const" | "wasm.constf" => true,
            "wasm.add" | "wasm.sub" | "wasm.mul" => ty.to_string().starts_with('i'),
            _ => false,
        };
        assert_eq!(is_constant_instruction(&op), expected, "{name} at {ty}");
        seen += usize::from(expected);
    }
    // const at two integer types, constf at two float types, and three operations at two types.
    assert_eq!(seen, 4 + 6);
}

#[test]
fn the_marker_alone_over_approximates_and_is_constant_instruction_does_not() {
    let (_context, function) = parse_function("%a: f32, %b: f32", "%r = wasm.add %a, %b;");
    let op = wasm_ops(function)[0];
    let op = op.borrow();
    assert!(op.implements::<dyn ConstantInstruction>());
    assert!(!is_constant_instruction(&op));
}

#[test]
fn the_builder_builds_each_operation_with_the_type_its_operands_have() {
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

    macro_rules! binary {
        ($lhs:expr, $rhs:expr; $($method:ident),*) => {
            $(built.push((stringify!($method), b.$method($lhs, $rhs, SPAN).unwrap()));)*
        };
    }
    macro_rules! unary {
        ($operand:expr; $($method:ident),*) => {
            $(built.push((stringify!($method), b.$method($operand, SPAN).unwrap()));)*
        };
    }
    for (lhs, rhs) in [(params[0], params[1]), (params[2], params[3])] {
        binary!(lhs, rhs; add, sub, mul, div_s, div_u, rem_s, rem_u, and, or, xor, shl, shr_s, shr_u, rotl, rotr);
        unary!(lhs; clz, ctz, popcnt, extend8_s, extend16_s);
    }
    unary!(params[2]; extend32_s);
    for (lhs, rhs) in [(params[4], params[5]), (params[6], params[7])] {
        binary!(lhs, rhs; add, sub, mul, div, min, max, copysign);
        unary!(lhs; abs, neg, sqrt, ceil, floor, trunc, nearest);
    }
    let one = b.i32(1, SPAN);
    let two = b.i64(2, SPAN);
    let three = b.f32(3.0, SPAN);
    let four = b.f64(4.0, SPAN);
    let chosen = b.select(one, params[0], params[1], SPAN).unwrap();
    let text = text_of(&test);
    for (method, value) in &built {
        let ty = value.borrow().ty();
        let operand = params.iter().find(|p| p.borrow().ty() == ty).unwrap().borrow().ty();
        assert_eq!(ty, operand, "{method}");
        assert!(text.contains(&format!("wasm.{method} ")), "{method}: {text}");
    }
    assert_eq!(one.borrow().ty(), test.context().i32());
    assert_eq!(two.borrow().ty(), test.context().i64());
    assert_eq!(three.borrow().ty(), test.context().f32());
    assert_eq!(four.borrow().ty(), test.context().f64());
    assert_eq!(chosen.borrow().ty(), test.context().i32());
    assert!(text.contains("wasm.const 1 : i32;"), "{text}");
    assert!(
        text.contains("wasm.constf 3 : f32;") || text.contains("wasm.constf 3.0 : f32;"),
        "{text}"
    );
    test.function_builder().ret([], SPAN).unwrap();
    test.function().borrow().as_operation().verify().expect("valid function");
    let printed = text_of(&test);
    let context = context();
    let (_, reprinted) = parse_function_fixpoint(&context, "f", &printed).unwrap();
    assert_eq!(reprinted, printed);
}

/// The text of `test`'s function.
fn text_of(test: &Test) -> alloc::string::String {
    format!("{}", test.function().borrow().as_operation())
}

#[test]
fn dead_code_elimination_removes_every_unused_operation_including_the_ones_that_may_trap() {
    let text = every_operation();
    // The parameters are the function's own; the function has no results.
    let context = context();
    let module_text = format!("builtin.module public @m {{\n{text}\n}};");
    let module = runic_ir::parse::parse::<runic_ir::dialects::builtin::Module>(
        runic_ir::parse::ParserConfig::new(context.clone()),
        "wasm",
        module_text,
    )
    .into_result()
    .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let mut passes = runic_ir::pass::PassManager::on::<runic_ir::dialects::builtin::Function>(
        context,
        runic_ir::pass::Nesting::Implicit,
    );
    passes.add_pass(alloc::boxed::Box::<DeadCodeElimination>::default()).unwrap();
    passes.enable_verifier(true);
    passes.run(function).unwrap();
    let after = function.borrow().to_string();
    assert!(!after.contains(" = wasm."), "{after}");
    assert!(after.contains("builtin.ret"), "{after}");
}

#[test]
fn dead_code_elimination_removes_a_division_by_zero_whose_result_is_unused() {
    let after = dce(
        "%a: i32",
        "",
        "%z = wasm.const 0 : i32;\n%q = wasm.div_s %a, %z;\n%r = wasm.rem_u %a, %z;\nbuiltin.ret;",
    )
    .unwrap();
    assert!(!after.contains(" = wasm."), "{after}");
}

#[test]
fn dead_code_elimination_keeps_an_operation_whose_result_is_used() {
    let after =
        dce("%a: i32, %b: i32", "i32", "%q = wasm.div_s %a, %b;\nbuiltin.ret %q : (i32);").unwrap();
    assert!(after.contains("wasm.div_s"), "{after}");
}

#[test]
fn common_subexpression_elimination_merges_equal_operations() {
    for (op, types) in BINARY.iter().chain(COMPARE) {
        let ty = types[0];
        let result = if COMPARE.iter().any(|(name, _)| name == op) {
            "i32"
        } else {
            ty
        };
        let after = cse(
            &format!("%a: {ty}, %b: {ty}"),
            &format!("{result}, {result}"),
            &format!(
                "%x = wasm.{op} %a, %b;\n%y = wasm.{op} %a, %b;\nbuiltin.ret %x, %y : ({result}, \
                 {result});"
            ),
        )
        .unwrap();
        assert_eq!(after.matches(&format!("wasm.{op} ")).count(), 1, "{op}: {after}");
    }
    for (op, types) in UNARY {
        let ty = types[0];
        let after = cse(
            &format!("%a: {ty}"),
            &format!("{ty}, {ty}"),
            &format!("%x = wasm.{op} %a;\n%y = wasm.{op} %a;\nbuiltin.ret %x, %y : ({ty}, {ty});"),
        )
        .unwrap();
        assert_eq!(after.matches(&format!("wasm.{op} ")).count(), 1, "{op}: {after}");
    }
}

#[test]
fn common_subexpression_elimination_merges_constants_and_equal_conversions_and_selects() {
    let after = cse(
        "",
        "i32, i32",
        "%x = wasm.const 4 : i32;\n%y = wasm.const 4 : i32;\nbuiltin.ret %x, %y : (i32, i32);",
    )
    .unwrap();
    assert_eq!(after.matches("wasm.const").count(), 1, "{after}");
    let after = cse(
        "%a: i64",
        "i32, i32",
        "%x = wasm.wrap %a <{ ty = i32 }>;\n%y = wasm.wrap %a <{ ty = i32 }>;\nbuiltin.ret %x, %y \
         : (i32, i32);",
    )
    .unwrap();
    assert_eq!(after.matches("wasm.wrap").count(), 1, "{after}");
    let after = cse(
        "%a: f32",
        "i32, i64",
        "%x = wasm.trunc_s %a <{ ty = i32 }>;\n%y = wasm.trunc_s %a <{ ty = i64 }>;\nbuiltin.ret \
         %x, %y : (i32, i64);",
    )
    .unwrap();
    assert_eq!(after.matches("wasm.trunc_s").count(), 2, "{after}");
    let after = cse(
        "%c: i32, %a: i32, %b: i32",
        "i32, i32",
        "%x = wasm.select %c, %a, %b;\n%y = wasm.select %c, %a, %b;\nbuiltin.ret %x, %y : (i32, \
         i32);",
    )
    .unwrap();
    assert_eq!(after.matches("wasm.select").count(), 1, "{after}");
}

#[test]
fn common_subexpression_elimination_merges_commutative_operations_with_swapped_operands() {
    let commutative = ["add", "mul", "and", "or", "xor", "eq", "ne"];
    for (op, types) in BINARY.iter().chain(COMPARE) {
        let ty = types[0];
        let result = if COMPARE.iter().any(|(name, _)| name == op) {
            "i32"
        } else {
            ty
        };
        let after = cse(
            &format!("%a: {ty}, %b: {ty}"),
            &format!("{result}, {result}"),
            &format!(
                "%x = wasm.{op} %a, %b;\n%y = wasm.{op} %b, %a;\nbuiltin.ret %x, %y : ({result}, \
                 {result});"
            ),
        )
        .unwrap();
        let expected = if commutative.contains(op) { 1 } else { 2 };
        assert_eq!(after.matches(&format!("wasm.{op} ")).count(), expected, "{op}: {after}");
    }
}
