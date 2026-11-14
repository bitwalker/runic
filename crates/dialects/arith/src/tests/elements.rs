//! Rewriting the types `arith` operations hold: a constant's value and the `ty` a cast keeps beside
//! its result change together with the result.

use alloc::{format, string::String, vec::Vec};

use runic_core::{
    Attribute, Context, ElementRewrite, Int, Operation, OperationRef, Report, Type, TypeMapper,
    Value,
    dialects::builtin::{Function, attributes::TypeAttr},
    interner::Symbol,
    recursively_replace_elements_in,
    testing::text::parse,
};

use super::label_messages;

const FUNCTION: &str = r#"builtin.function public extern("C") @f(%a: i64, %b: i16) -> i64 {
    %c = "arith.constant"() <{ value = #builtin.int<4294967297 : i64> }> : () -> i64;
    %t = "arith.trunc"(%a) <{ ty = #builtin.type<i16> }> : (i64) -> i16;
    %z = "arith.zext"(%b) <{ ty = #builtin.type<i64> }> : (i16) -> i64;
    %s = "arith.sext"(%b) <{ ty = #builtin.type<i64> }> : (i16) -> i64;
    builtin.ret %c : (i64);
};"#;

/// `from` to `to`, everything else as it is.
fn replacing(context: &Context, from: Type, to: Type) -> TypeMapper<'_> {
    TypeMapper::new(context, move |_, ty| if ty == from { to } else { ty })
}

/// Truncates a constant to the width of its new type.
fn truncate(value: &Int, _old: Type, new: Type) -> Result<Int, Report> {
    Ok(value.trunc(Int::width_for_type(new).expect("an integer type")))
}

/// The operations in the body of `function`.
fn body(function: OperationRef) -> Vec<OperationRef> {
    let mut ops = Vec::new();
    function.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
    ops.remove(0);
    ops
}

fn name(op: OperationRef) -> String {
    format!("{}", op.borrow().name())
}

fn result_type(op: OperationRef) -> Type {
    op.borrow().results()[0].borrow().ty()
}

#[test]
fn a_constant_and_the_casts_keep_their_types_in_step() {
    let (context, function) = parse::<Function>(FUNCTION).expect("the function parses");
    let root = function.as_operation_ref();
    root.borrow().recursively_verify().expect("the function verifies");

    let mut rewrite = ElementRewrite::types(replacing(&context, context.i64(), context.i32()))
        .with_int_constants(truncate);
    assert!(recursively_replace_elements_in(root, &mut rewrite).expect("the rewrite"));
    drop(rewrite);
    if let Err(err) = root.borrow().recursively_verify() {
        panic!("the rewritten function does not verify: {err:?}");
    }

    for op in body(root) {
        let op_name = name(op);
        match op_name.as_str() {
            "arith.constant" => {
                let op = op.borrow();
                let constant = op.downcast_ref::<crate::Constant>().unwrap();
                assert_eq!(*constant.get_value(), Int::from(1i32));
                assert_eq!(constant.value().ty(), context.i32());
                assert_eq!(constant.result().ty(), context.i32());
            }
            "arith.trunc" | "arith.zext" | "arith.sext" => {
                let ty = *op
                    .borrow()
                    .get_property(Symbol::intern("ty"))
                    .downcast_ref::<TypeAttr>()
                    .unwrap()
                    .as_value();
                assert_eq!(ty, result_type(op), "{op_name}");
            }
            _ => (),
        }
    }
    let zext = body(root).into_iter().find(|op| name(*op) == "arith.zext").unwrap();
    assert_eq!(result_type(zext), context.i32());
    let trunc = body(root).into_iter().find(|op| name(*op) == "arith.trunc").unwrap();
    assert_eq!(result_type(trunc), context.i16());
    assert_eq!(
        trunc.borrow().operands()[0].borrow().as_value_ref().borrow().ty(),
        context.i32()
    );
}

/// Rewriting types does not make a conversion pattern unnecessary: an `arith.zext` whose two types
/// became equal is rewritten as asked, and its verifier rejects it.
#[test]
fn a_cast_whose_types_became_equal_is_rejected_by_its_verifier() {
    let (context, function) = parse::<Function>(FUNCTION).expect("the function parses");
    let root = function.as_operation_ref();
    let mut rewrite = ElementRewrite::types(replacing(&context, context.i64(), context.i16()))
        .with_int_constants(truncate);
    assert!(recursively_replace_elements_in(root, &mut rewrite).expect("the rewrite"));
    drop(rewrite);

    let zext = body(root).into_iter().find(|op| name(*op) == "arith.zext").unwrap();
    assert_eq!(result_type(zext), context.i16());
    let err = zext.borrow().verify().expect_err("a zext from i16 to i16 is invalid");
    let message = label_messages(err.as_diagnostic()).join(" | ");
    assert_eq!(
        message,
        "invalid 'arith.zext' from `i16` to `i16`: `i16` is not wider than `i16`"
    );
}
