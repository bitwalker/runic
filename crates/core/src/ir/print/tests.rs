use alloc::{string::ToString, vec::Vec};
use core::cell::Cell;

use pretty_assertions::assert_eq;

use crate::{
    Context, Op, OperationRef, SourceSpan, StructType, Type, ValueRef,
    dialects::builtin::BuiltinOpBuilder, interner::Symbol, testing::Test,
};

/// `struct <name><ptr<rec.0>, held>`, as a group of its own.
fn singleton(context: &Context, name: &str, held: Type) -> Type {
    let body = context.struct_type(StructType::named(
        Symbol::intern(name),
        [held.into(), context.ptr_type(context.rec_var(0)).into()],
    ));
    context.rec_group(&[body]).expect("a valid group")[0]
}

/// A function `f` taking one argument of each type in `params`, whose body casts its first
/// argument to each type in `casts` in turn, then returns. Returns the test and the casts.
fn function_with_casts(
    params: impl FnOnce(&Context) -> Vec<Type>,
    casts: impl FnOnce(&Context) -> Vec<Type>,
) -> (Test, Vec<OperationRef>) {
    let mut test = Test::new("f", |cx| (params(cx), []));
    let casts = casts(test.context());
    let mut ops = Vec::new();
    {
        let mut f = test.function_builder();
        let arg = f.entry_block().borrow().arguments()[0] as ValueRef;
        for ty in casts {
            let result = f
                .builder_mut()
                .unrealized_conversion_cast(arg, ty, SourceSpan::UNKNOWN)
                .unwrap();
            ops.push(result.borrow().get_defining_op().unwrap());
        }
        f.builder_mut().ret([], SourceSpan::UNKNOWN).unwrap();
    }
    (test, ops)
}

fn print_function(test: &Test) -> alloc::string::String {
    test.function().borrow().as_operation().to_string()
}

#[test]
fn an_operation_without_recursive_types_prints_as_before() {
    let (test, _) = function_with_casts(|cx| [cx.i32()].into(), |cx| [cx.i64()].into());
    assert_eq!(
        print_function(&test),
        "\
builtin.function public extern(\"C\") @f(%0: i32) {
    %1 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<i64> }>;
    builtin.ret;
};"
    );
}

#[test]
fn a_root_operation_declares_the_groups_it_refers_to() {
    let list = Cell::new(None);
    let (test, _) = function_with_casts(
        |cx| {
            let member = singleton(cx, "list", cx.i32());
            list.set(Some(member));
            [cx.ptr_type(member)].into()
        },
        |_| [list.get().unwrap()].into(),
    );
    assert_eq!(
        print_function(&test),
        "\
rec {
  type !list = struct list<i32, ptr<!list>>;
}

builtin.function public extern(\"C\") @f(%0: ptr<!list>) {
    %1 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<!list> }>;
    builtin.ret;
};"
    );
}

#[test]
fn a_group_is_declared_after_the_groups_it_refers_to() {
    let (test, _) = function_with_casts(
        |cx| {
            let a = singleton(cx, "a", cx.i32());
            let b = singleton(cx, "b", cx.ptr_type(a));
            [b].into()
        },
        |_| Vec::new(),
    );
    assert_eq!(
        print_function(&test),
        "\
rec {
  type !a = struct a<i32, ptr<!a>>;
}
rec {
  type !b = struct b<ptr<!a>, ptr<!b>>;
}

builtin.function public extern(\"C\") @f(%0: !b) {
    builtin.ret;
};"
    );
}

#[test]
fn a_mutually_recursive_group_is_declared_once() {
    let node = Cell::new(None);
    let (test, _) = function_with_casts(
        |cx| {
            let (list, node_var) = (cx.rec_var(0), cx.rec_var(1));
            let members = cx
                .rec_group(&[
                    cx.struct_type(StructType::named(
                        Symbol::intern("list"),
                        [cx.i32().into(), cx.ptr_type(node_var).into()],
                    )),
                    cx.struct_type(StructType::named(
                        Symbol::intern("node"),
                        [cx.ptr_type(list).into(), cx.ptr_type(node_var).into()],
                    )),
                ])
                .unwrap();
            node.set(Some(members[1]));
            [members[0]].into()
        },
        |_| [node.get().unwrap(), node.get().unwrap()].into(),
    );
    let printed = print_function(&test);
    assert_eq!(
        printed,
        "\
rec {
  type !list = struct list<i32, ptr<!node>>;
  type !node = struct node<ptr<!list>, ptr<!node>>;
}

builtin.function public extern(\"C\") @f(%0: !list) {
    %1 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<!node> }>;
    %2 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<!node> }>;
    builtin.ret;
};"
    );
    assert_eq!(print_function(&test), printed);
}

#[test]
fn a_nested_operation_prints_no_declarations() {
    let list = Cell::new(None);
    let (_test, ops) = function_with_casts(
        |cx| {
            let member = singleton(cx, "list", cx.i32());
            list.set(Some(member));
            [member].into()
        },
        |cx| [cx.ptr_type(list.get().unwrap())].into(),
    );
    assert_eq!(
        ops[0].borrow().to_string(),
        "%1 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<ptr<!list>> }>;"
    );
}

#[test]
fn a_group_reached_only_as_a_dependency_is_named_after_the_printed_ones() {
    let (test, _) = function_with_casts(
        |cx| {
            let a = singleton(cx, "list", cx.i32());
            let b = singleton(cx, "list", cx.ptr_type(a));
            [b].into()
        },
        |_| Vec::new(),
    );
    assert_eq!(
        print_function(&test),
        "\
rec {
  type !list_0 = struct list<i32, ptr<!list_0>>;
}
rec {
  type !list = struct list<ptr<!list_0>, ptr<!list>>;
}

builtin.function public extern(\"C\") @f(%0: !list) {
    builtin.ret;
};"
    );
}

#[test]
fn op_print_declares_the_groups_of_a_root_only() {
    use crate::OpPrintingFlags;

    let list = Cell::new(None);
    let (test, ops) = function_with_casts(
        |cx| {
            let member = singleton(cx, "list", cx.i32());
            list.set(Some(member));
            [member].into()
        },
        |cx| [cx.ptr_type(list.get().unwrap())].into(),
    );
    let flags = OpPrintingFlags::default();
    let root = test.function().borrow().print(&flags).to_string();
    assert_eq!(root, print_function(&test));
    assert!(root.starts_with("rec {\n  type !list = struct list<i32, ptr<!list>>;\n}\n\n"));
    let cast = ops[0]
        .try_downcast_op::<crate::dialects::builtin::UnrealizedConversionCast>()
        .unwrap();
    assert_eq!(
        cast.borrow().print(&flags).to_string(),
        "%1 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<ptr<!list>> }>;"
    );
}

#[test]
fn a_signature_attribute_shares_the_aliases_of_its_operation() {
    use crate::dialects::builtin::attributes::{Signature, SignatureAttr};

    let list = Cell::new(None);
    let (test, mut ops) = function_with_casts(
        |cx| {
            let member = singleton(cx, "list", cx.i32());
            list.set(Some(member));
            [member].into()
        },
        |cx| [cx.ptr_type(list.get().unwrap())].into(),
    );
    let context = test.context_rc();
    let list = list.get().unwrap();
    let signature = Signature::new(&context, [context.ptr_type(list)], [list]);
    let attr = context.create_attribute::<SignatureAttr, _>(signature);
    ops[0].borrow_mut().set_attribute("sig", attr.as_attribute_ref());
    let printed = print_function(&test);
    assert_eq!(
        printed,
        "\
rec {
  type !list = struct list<i32, ptr<!list>>;
}

builtin.function public extern(\"C\") @f(%0: !list) {
    %1 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<ptr<!list>> }> attributes { \
         sig = #builtin.signature<extern(\"C\") (ptr<!list>) -> !list> };
    builtin.ret;
};"
    );
}

#[test]
fn a_region_or_block_printed_on_its_own_declares_its_groups() {
    use crate::{OpPrintingFlags, formatter::PrettyPrint};

    let list = Cell::new(None);
    let (test, _) = function_with_casts(
        |cx| {
            let member = singleton(cx, "list", cx.i32());
            list.set(Some(member));
            [cx.i32()].into()
        },
        |_| [list.get().unwrap()].into(),
    );
    let declaration = "rec {\n  type !list = struct list<i32, ptr<!list>>;\n}\n\n";
    let function = test.function();
    let function = function.borrow();
    let region = function.as_operation().regions().front().get().unwrap();
    let flags = OpPrintingFlags::default();
    let region_text = region.print(test.context_rc(), &flags).to_string();
    assert!(region_text.starts_with(declaration), "unexpected region text:\n{region_text}");
    assert_eq!(region_text.matches("rec {").count(), 1);

    let block = region.entry_block_ref().unwrap();
    let block_text = block.borrow().render().to_string();
    assert!(block_text.starts_with(declaration), "unexpected block text:\n{block_text}");
    assert_eq!(block_text.matches("rec {").count(), 1);
}
