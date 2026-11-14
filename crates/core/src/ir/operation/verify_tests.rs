//! The generic checks of [Operation::verify]: every attribute an operation holds verifies itself,
//! at any depth, and every type it holds belongs to its context and is closed.

use alloc::{format, rc::Rc, string::String, vec};

use crate::{
    AttributeRef, CallConv, Context, Int, Op, Operation,
    attributes::AttrList,
    dialects::{
        builtin::{
            FunctionRef,
            attributes::{
                AbiParam, BoolAttr, DeclaredEffects, DeclaredEffectsAttr, IntAttr, List, ListAttr,
                MemoryEffectDescriptor, Signature, SignatureAttr, TypeArrayAttr, TypeAttr,
            },
        },
        debuginfo::attributes::{Variable, VariableAttr},
    },
    effects::MemoryEffect,
    interner::Symbol,
    testing::{Test, label_messages},
};

/// A list attribute holding `entries`.
fn list(context: &Rc<Context>, entries: impl IntoIterator<Item = AttributeRef>) -> AttributeRef {
    let mut list = AttrList::new();
    for entry in entries {
        list.push_back(entry.borrow().as_attr().as_attr_ref());
    }
    context.create_attribute::<ListAttr, _>(List::from(list)).as_attribute_ref()
}

/// A signature `(i32)` whose parameter holds `attr` as its attribute `note`.
fn signature_holding(context: &Rc<Context>, attr: AttributeRef) -> AttributeRef {
    let mut param = AbiParam::new(context.i32());
    param.set_attribute(Symbol::intern("note"), attr);
    let signature = Signature::from_parts(CallConv::default(), vec![param], vec![]);
    context.create_attribute::<SignatureAttr, _>(signature).as_attribute_ref()
}

/// The labels of the error `op` fails to verify with, on one line.
fn verify_error(op: &Operation) -> String {
    let err = op.verify().expect_err("the operation must not verify");
    label_messages(err.as_diagnostic()).join(" | ")
}

/// A test function `@f`, a definition taking an `i32`.
fn function() -> (Test, FunctionRef) {
    let test = Test::new("f", |cx| ([cx.i32()], []));
    let function = test.function();
    (test, function)
}

/// `function` with the discardable attribute `name` set to `attr`: the error it fails to verify
/// with.
fn error_with(mut function: FunctionRef, name: &'static str, attr: AttributeRef) -> String {
    function.borrow_mut().as_operation_mut().set_attribute(name, attr);
    let message = verify_error(function.borrow().as_operation());
    function.borrow_mut().as_operation_mut().remove_attribute(name);
    assert!(function.borrow().as_operation().verify().is_ok());
    message
}

/// An effect that names both an argument and a result.
fn arg_and_result() -> DeclaredEffects {
    DeclaredEffects::from([MemoryEffectDescriptor {
        effect: MemoryEffect::Read,
        argument: Some(0),
        result: Some(0),
    }])
}

/// An integer constant edited in place to a width its type does not have is caught by the next
/// verification, wherever it is held.
#[test]
fn an_int_attr_edited_in_place_is_caught() {
    let (test, mut function) = function();
    let context = test.context_rc();
    let mut constant = context.int_attr(Int::from(1i32), context.i32()).as_attribute_ref();
    function.borrow_mut().as_operation_mut().set_attribute("value", constant);
    assert!(function.borrow().as_operation().verify().is_ok());

    *constant.borrow_mut().value_mut().as_any_mut().downcast_mut::<Int>().unwrap() = Int::from(1i8);
    let message = verify_error(function.borrow().as_operation());
    assert!(
        message.contains(
            "attribute 'value' of 'builtin.function' is an integer constant of type `i32` holding \
             a 8-bit value, but a constant of that type is 32 bits wide"
        ),
        "{message}"
    );
}

/// The integer rule fires for a constant held by a signature's parameter, itself held by a list.
#[test]
fn an_int_attr_in_a_signature_parameter_is_verified() {
    let (test, function) = function();
    let context = test.context_rc();
    let bad = || {
        context
            .create_attribute_with_type::<IntAttr, _>(Int::from(-1i32), context.i8())
            .as_attribute_ref()
    };
    let expected = "an attribute held by attribute 'sig' of 'builtin.function' is an integer \
                    constant of type `i8` holding a 32-bit value, but a constant of that type is \
                    8 bits wide";
    let message = error_with(function, "sig", signature_holding(&context, bad()));
    assert!(message.contains(expected), "{message}");

    let message = error_with(function, "sig", list(&context, [signature_holding(&context, bad())]));
    assert!(message.contains(expected), "{message}");
}

/// The signature rule fires for a signature held by another signature's parameter.
#[test]
fn a_signature_in_a_signature_parameter_is_verified() {
    let (test, function) = function();
    let context = test.context_rc();
    let invalid = Signature::from_parts(
        CallConv::default(),
        vec![AbiParam::new(context.never_type())],
        vec![],
    );
    let invalid = context.create_attribute::<SignatureAttr, _>(invalid).as_attribute_ref();
    let message = error_with(function, "sig", signature_holding(&context, invalid));
    assert!(
        message.contains(
            "an attribute held by attribute 'sig' of 'builtin.function' is an invalid signature: \
             parameter 0 has type `never`"
        ),
        "{message}"
    );
}

/// An effect naming both an argument and a result is rejected as the function's property, in a
/// list, and in a signature's parameter.
#[test]
fn a_declared_effect_naming_an_argument_and_a_result_is_rejected() {
    let (test, mut function) = function();
    let context = test.context_rc();
    let reason = "is an invalid effect declaration: the effect `read(arg 0)(result 0)` names both \
                  an argument and a result";

    function.borrow_mut().set_memory_effects(arg_and_result());
    let message = verify_error(function.borrow().as_operation());
    assert!(
        message.contains(&format!("property 'memory_effects' of 'builtin.function' {reason}")),
        "{message}"
    );
    function.borrow_mut().set_memory_effects(DeclaredEffects::Unknown);
    assert!(function.borrow().as_operation().verify().is_ok());

    let effects = || {
        context
            .create_attribute::<DeclaredEffectsAttr, _>(arg_and_result())
            .as_attribute_ref()
    };
    let nested = format!("an attribute held by attribute 'extra' of 'builtin.function' {reason}");
    let message = error_with(function, "extra", list(&context, [effects()]));
    assert!(message.contains(&nested), "{message}");
    let message = error_with(function, "extra", signature_holding(&context, effects()));
    assert!(message.contains(&nested), "{message}");
}

/// The type held by a type attribute is checked: one edited to an open type, or to a type of
/// another context, is rejected, directly or in a list.
#[test]
fn a_type_attribute_holding_an_invalid_type_is_rejected() {
    let (test, function) = function();
    let context = test.context_rc();
    let other = Rc::new(Context::default());
    let open = context.ptr_type(context.rec_var(0));
    let holding = |ty| {
        let mut attr = context.create_attribute::<TypeAttr, _>(context.i32());
        *attr.borrow_mut().as_value_mut() = ty;
        attr.as_attribute_ref()
    };

    let message = error_with(function, "ty", holding(open));
    assert!(
        message.contains(
            "attribute 'ty' of 'builtin.function' holds type `ptr<rec.0>`, which is open"
        ),
        "{message}"
    );
    let message = error_with(function, "ty", list(&context, [holding(open)]));
    assert!(
        message.contains(
            "attribute 'ty' of 'builtin.function' holds type `ptr<rec.0>`, which is open"
        ),
        "{message}"
    );
    let message = error_with(function, "ty", holding(other.i64()));
    assert!(
        message.contains(
            "attribute 'ty' of 'builtin.function' holds type `i64`, which belongs to another \
             context"
        ),
        "{message}"
    );
}

/// An element of a type array is checked.
#[test]
fn a_type_array_element_that_is_invalid_is_rejected() {
    let (test, function) = function();
    let context = test.context_rc();
    let other = Rc::new(Context::default());
    for (ty, problem) in [
        (context.ptr_type(context.rec_var(0)), "which is open"),
        (other.i64(), "which belongs to another context"),
    ] {
        let mut attr = context.create_attribute::<TypeArrayAttr, _>([context.i32()]);
        attr.borrow_mut().as_value_mut().push(ty);
        let message = error_with(function, "types", attr.as_attribute_ref());
        assert!(
            message.contains(&format!(
                "attribute 'types' of 'builtin.function' holds type `{ty}`, {problem}"
            )),
            "{message}"
        );
    }
}

/// The type of a debug-info variable is checked.
#[test]
fn a_debug_info_variable_of_an_invalid_type_is_rejected() {
    let (test, function) = function();
    let context = test.context_rc();
    let other = Rc::new(Context::default());
    for (ty, problem) in [
        (context.ptr_type(context.rec_var(0)), "which is open"),
        (other.i64(), "which belongs to another context"),
    ] {
        let mut attr = context.create_attribute::<VariableAttr, _>(Variable::default());
        attr.borrow_mut().as_value_mut().ty = Some(ty);
        let message = error_with(function, "var", attr.as_attribute_ref());
        assert!(
            message.contains(&format!(
                "attribute 'var' of 'builtin.function' holds type `{ty}`, {problem}"
            )),
            "{message}"
        );
    }
}

/// A kind that takes its type from its value does not list that type among its elements, and the
/// type is still checked.
#[test]
fn a_fixed_own_type_of_another_context_is_rejected() {
    let (test, function) = function();
    let context = test.context_rc();
    let other = Rc::new(Context::default());
    let mut flag = context.create_attribute::<BoolAttr, _>(true).as_attribute_ref();
    flag.borrow_mut().as_attr_mut().set_type_unchecked(other.i1());
    let message = error_with(function, "flag", list(&context, [flag]));
    assert!(
        message.contains(
            "an attribute held by attribute 'flag' of 'builtin.function' has type `i1`, which \
             belongs to another context"
        ),
        "{message}"
    );
}
