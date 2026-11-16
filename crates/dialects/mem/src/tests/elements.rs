//! Rewriting the types `mem` operations hold: the `ty` an operation keeps beside its result (or as
//! the only statement of the type it measures or reads) changes together with the result.

use alloc::{format, string::String};

use runic_ir::{
    Context, ElementRewrite, Operation, OperationRef, Type, TypeMapper, Value,
    dialects::builtin::attributes::TypeAttr, interner::Symbol, recursively_replace_elements_in,
};

use super::parse_function;

const FUNCTION: &str = r#"builtin.function public extern("C") @f(%p: ptr<i64>, %q: ptr, %x: isize) {
    %0 = mem.alloca i64;
    %1 = mem.sizeof i64;
    %2 = mem.inttoptr %x : ptr<i64>;
    %3 = mem.ptrcast %p : ptr<struct<i64, i8>>;
    %4 = mem.load %q : i64;
    builtin.ret;
};"#;

/// `i64` to `i32`, everything else as it is.
fn i64_to_i32(context: &Context) -> TypeMapper<'_> {
    TypeMapper::new(context, |context, ty| {
        if ty == context.i64() {
            context.i32()
        } else {
            ty
        }
    })
}

/// The operation named `name` in `function`, with its `ty` and its result type.
fn find(function: OperationRef, name: &str) -> (Type, Type) {
    let mut found = None;
    function.borrow().prewalk_all(|op: &Operation| {
        if format!("{}", op.name()) == name {
            found = Some(op.as_operation_ref());
        }
    });
    let op = found.unwrap_or_else(|| panic!("no '{name}'"));
    let op = op.borrow();
    let ty = *op
        .get_property(Symbol::intern("ty"))
        .downcast_ref::<TypeAttr>()
        .unwrap()
        .as_value();
    (ty, op.results()[0].borrow().ty())
}

#[test]
fn every_stated_type_changes_with_the_result() {
    let (context, function) = parse_function(FUNCTION).expect("the function parses");
    let root = function.as_operation_ref();
    root.borrow().recursively_verify().expect("the function verifies");

    let mut rewrite = ElementRewrite::types(i64_to_i32(&context));
    assert!(recursively_replace_elements_in(root, &mut rewrite).expect("the rewrite"));
    drop(rewrite);
    if let Err(err) = root.borrow().recursively_verify() {
        panic!("the rewritten function does not verify: {err:?}");
    }

    let (i32, isize) = (context.i32(), context.isize());
    let ptr_i32 = context.ptr_type(i32);
    assert_eq!(find(root, "mem.alloca"), (i32, ptr_i32));
    assert_eq!(find(root, "mem.sizeof"), (i32, isize));
    assert_eq!(find(root, "mem.inttoptr"), (ptr_i32, ptr_i32));
    let (ty, result) = find(root, "mem.ptrcast");
    assert_eq!((format!("{ty}"), ty), (String::from("ptr<struct<i32, i8>>"), result));
    // a load through an opaque pointer states the type it reads
    assert_eq!(find(root, "mem.load"), (i32, i32));

    let printed = format!("{}", *root.borrow());
    let (_expected_context, expected) =
        parse_function(&FUNCTION.replace("i64", "i32")).expect("the expected function parses");
    assert_eq!(printed, format!("{}", *expected.as_operation_ref().borrow()));
}
