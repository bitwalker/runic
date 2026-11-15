use alloc::{string::ToString, vec, vec::Vec};
use core::cell::Cell;

use pretty_assertions::assert_eq;

use crate::{
    Builder, Context, Op, OperationRef, SourceSpan, StructType, Type, ValueRef,
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
            let cast = f
                .builder_mut()
                .unrealized_conversion_cast([arg], [ty], SourceSpan::UNKNOWN)
                .unwrap();
            ops.push(cast.as_operation_ref());
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
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64;
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
    %1 = builtin.unrealized_conversion_cast %0 : ptr<!list> to !list;
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
    %1 = builtin.unrealized_conversion_cast %0 : !list to !node;
    %2 = builtin.unrealized_conversion_cast %0 : !list to !node;
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
        "%1 = builtin.unrealized_conversion_cast %0 : !list to ptr<!list>;"
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
        "%1 = builtin.unrealized_conversion_cast %0 : !list to ptr<!list>;"
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
    %1 = builtin.unrealized_conversion_cast %0 : !list to ptr<!list> attributes { sig = \
         #builtin.signature<extern(\"C\") (ptr<!list>) -> !list> };
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

/// A function `f(%0: i32)` whose body has `blocks` blocks. Each holds `casts` casts of its first
/// argument (the entry block's is `%0`; each other block takes an `i32` of its own), every third
/// of them casting two values to two results, then a return.
fn long_function(blocks: usize, casts: usize) -> Test {
    let mut test = Test::new("f", |cx| ([cx.i32()], []));
    let region = test.function().borrow().body().as_region_ref();
    let (i32, i64) = (test.context().i32(), test.context().i64());
    for i in 0..blocks {
        let block = if i == 0 {
            test.entry_block()
        } else {
            test.builder_mut().create_block(region, None, &[i32])
        };
        let arg = block.borrow().arguments()[0] as ValueRef;
        let builder = test.builder_mut();
        builder.set_insertion_point_to_end(block);
        for j in 0..casts {
            if j % 3 == 2 {
                builder
                    .unrealized_conversion_cast([arg, arg], [i64, i64], SourceSpan::UNKNOWN)
                    .unwrap();
            } else {
                builder.unrealized_conversion_cast([arg], [i64], SourceSpan::UNKNOWN).unwrap();
            }
        }
        builder.ret([], SourceSpan::UNKNOWN).unwrap();
    }
    test
}

/// The printed text of a block of 50 operations: one operation a line, in order.
#[test]
fn the_printed_text_of_a_block_of_fifty_operations() {
    let test = long_function(1, 49);
    assert_eq!(
        print_function(&test),
        r#"builtin.function public extern("C") @f(%0: i32) {
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %2 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %3, %4 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %5 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %6 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %7, %8 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %9 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %10 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %11, %12 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %13 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %14 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %15, %16 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %17 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %18 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %19, %20 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %21 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %22 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %23, %24 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %25 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %26 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %27, %28 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %29 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %30 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %31, %32 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %33 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %34 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %35, %36 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %37 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %38 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %39, %40 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %41 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %42 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %43, %44 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %45 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %46 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %47, %48 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %49 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %50 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %51, %52 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %53 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %54 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %55, %56 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %57 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %58 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %59, %60 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %61 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %62 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %63, %64 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to (i64, i64);
    %65 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    builtin.ret;
};"#
    );
}

/// The printed text of a region of 5 blocks: each block under its label, in order.
#[test]
fn the_printed_text_of_a_region_of_five_blocks() {
    let test = long_function(5, 2);
    assert_eq!(
        print_function(&test),
        r#"builtin.function public extern("C") @f(%0: i32) {
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    %2 = builtin.unrealized_conversion_cast %0 : i32 to i64;
    builtin.ret;
^block1(%3: i32):
    %4 = builtin.unrealized_conversion_cast %3 : i32 to i64;
    %5 = builtin.unrealized_conversion_cast %3 : i32 to i64;
    builtin.ret;
^block2(%6: i32):
    %7 = builtin.unrealized_conversion_cast %6 : i32 to i64;
    %8 = builtin.unrealized_conversion_cast %6 : i32 to i64;
    builtin.ret;
^block3(%9: i32):
    %10 = builtin.unrealized_conversion_cast %9 : i32 to i64;
    %11 = builtin.unrealized_conversion_cast %9 : i32 to i64;
    builtin.ret;
^block4(%12: i32):
    %13 = builtin.unrealized_conversion_cast %12 : i32 to i64;
    %14 = builtin.unrealized_conversion_cast %12 : i32 to i64;
    builtin.ret;
};"#
    );
}

/// Runs `f` on a thread with a 256 KiB stack, so that a recursion whose depth grows with the size
/// of the IR overflows it.
#[cfg(feature = "std")]
fn on_a_small_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(f)
        .expect("spawn a thread")
        .join()
        .expect("the thread finishes");
}

/// A block of 20,000 operations is built, printed and dropped without deep recursion.
#[cfg(feature = "std")]
#[test]
fn a_long_block_prints_on_a_small_stack() {
    on_a_small_stack(|| {
        let test = long_function(1, 20_000);
        let text = print_function(&test);
        assert_eq!(text.lines().count(), 20_003);
        assert!(text.ends_with("    builtin.ret;\n};"), "{}", &text[text.len() - 100..]);
        drop(text);
        drop(test);
    });
}

/// A region of 20,000 blocks is built, printed and dropped without deep recursion.
#[cfg(feature = "std")]
#[test]
fn a_long_region_prints_on_a_small_stack() {
    on_a_small_stack(|| {
        let test = long_function(20_000, 1);
        let text = print_function(&test);
        assert_eq!(text.lines().count(), 1 + 20_000 * 2 + 19_999 + 1);
        assert!(text.contains("\n^block19999(%"), "{}", &text[text.len() - 200..]);
        drop(text);
        drop(test);
    });
}

/// An operation with 20,000 operands and results, in a block with 20,000 arguments, is printed
/// and dropped without deep recursion.
#[cfg(feature = "std")]
#[test]
fn long_lists_print_on_a_small_stack() {
    on_a_small_stack(|| {
        const N: usize = 20_000;
        let mut test = Test::new("f", |cx| ([cx.i32()], []));
        let region = test.function().borrow().body().as_region_ref();
        let (i32, i64) = (test.context().i32(), test.context().i64());
        let entry = test.entry_block();
        let builder = test.builder_mut();
        builder.set_insertion_point_to_end(entry);
        builder.ret([], SourceSpan::UNKNOWN).unwrap();
        // The lists are on the heap: an array of 20,000 types would not fit on the stack.
        let block = builder.create_block(region, None, &vec![i32; N]);
        let args = block
            .borrow()
            .arguments()
            .iter()
            .map(|arg| *arg as ValueRef)
            .collect::<Vec<_>>();
        builder
            .unrealized_conversion_cast(args, vec![i64; N], SourceSpan::UNKNOWN)
            .unwrap();
        builder.ret([], SourceSpan::UNKNOWN).unwrap();
        let text = print_function(&test);
        assert_eq!(text.lines().count(), 6);
        assert_eq!(text.matches(": i32").count(), N + 1);
        drop(text);
        drop(test);
    });
}

/// A module with two functions, a nested module and operations with properties, attributes and
/// results, as the default printer prints it.
const NESTED_MODULE: &str = "\
builtin.module public @m {
    builtin.function public extern(\"C\") @f(%0: i32, %1: i32) -> i32 {
        %2 = test.add %0, %1 <{ overflow = #builtin.overflow<checked unsigned> }>;
        %3 = test.constant 7 : i32;
        builtin.ret %2 : (i32);
    };
    builtin.module public @inner {
        builtin.function public extern(\"C\") @g() {
            builtin.ret;
        };
    };
};";

/// `op` printed from a root printer with `flags`.
fn print_with(op: &crate::Operation, flags: &crate::OpPrintingFlags) -> alloc::string::String {
    let mut printer = crate::print::AsmPrinter::root(op.context_rc(), flags);
    crate::print::OpPrinter::print(op, &mut printer);
    printer.finish().to_string()
}

/// With `generic`, every operation at every depth prints in the generic form; the default print
/// is the custom form.
#[test]
fn the_generic_flag_prints_every_operation_generically() {
    use alloc::rc::Rc;

    use crate::{
        OpPrintingFlags,
        dialects::test::TestDialect,
        parse::{ParserConfig, parse_any},
    };

    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let module = parse_any(ParserConfig::new(context.clone()), "nested.runic", NESTED_MODULE)
        .into_result()
        .unwrap();
    let module = module.borrow();
    assert_eq!(print_with(&module, &OpPrintingFlags::default()), NESTED_MODULE);
    assert_eq!(module.to_string(), NESTED_MODULE);

    let flags = OpPrintingFlags {
        generic: true,
        ..Default::default()
    };
    let generic = print_with(&module, &flags);
    // An operation line is a quoted name, or a result list and a quoted name; every other line
    // closes the regions of an operation or labels a block.
    let mut operations = 0;
    for line in generic.lines().map(str::trim_start) {
        let name = match line.split_once(" = \"") {
            Some((results, rest))
                if results.split(", ").all(|result| {
                    result.strip_prefix('%').is_some_and(|id| id.chars().all(char::is_alphanumeric))
                }) =>
            {
                rest
            }
            _ if line.starts_with(['}', '^']) => continue,
            _ => line
                .strip_prefix('"')
                .unwrap_or_else(|| panic!("not generic: {line}\n{generic}")),
        };
        let name = name.split_once('"').map(|(name, _)| name);
        assert!(
            name.is_some_and(|name| name.contains('.')),
            "not an operation name: {line}\n{generic}"
        );
        operations += 1;
    }
    assert_eq!(operations, 8, "{generic}");
    assert!(
        generic.starts_with("\"builtin.module\"() <{ name = #builtin.ident<@m>"),
        "{generic}"
    );
    assert!(generic.contains("%2 = \"test.add\"(%0, %1) <{ overflow"), "{generic}");
    assert!(
        generic.contains("\"test.constant\"() <{ value = #builtin.int<7 : i32> }>"),
        "{generic}"
    );

    // A part of the module printed generically is generic too.
    let inner = module.regions().front().get().unwrap().entry().back().unwrap();
    let inner = print_with(&inner.borrow(), &flags);
    assert!(
        inner.starts_with("\"builtin.module\"() <{ name = #builtin.ident<@inner>"),
        "{inner}"
    );
    assert!(generic.contains(&inner.replace('\n', "\n    ")), "{inner}");

    // The flags have not changed the default print.
    assert_eq!(module.to_string(), NESTED_MODULE);
}

/// `text` parsed and verified, with the test dialect registered, and the context that owns it.
fn parse_with_test_dialect(
    text: &str,
) -> Result<(alloc::rc::Rc<Context>, OperationRef), alloc::string::String> {
    use alloc::rc::Rc;

    use crate::{
        dialects::test::TestDialect,
        parse::{ParserConfig, parse_any},
        testing::text::describe,
    };

    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    parse_any(ParserConfig::new(context.clone()), "text.runic", text)
        .into_result()
        .map(|op| (context, op))
        .map_err(|err| describe(&err))
}

/// `text` printed in the generic form, the generic text parsed again, and the second parse
/// printed in both forms: the same programs as the first parse printed in each form.
fn generic_reads_back(text: &str) -> alloc::string::String {
    use crate::print::same_program;

    let generic_flags = crate::OpPrintingFlags {
        generic: true,
        ..Default::default()
    };
    let (_context, first) = parse_with_test_dialect(text).unwrap_or_else(|err| panic!("{err}"));
    let custom = first.borrow().to_string();
    let generic = print_with(&first.borrow(), &generic_flags);
    let (_context, second) =
        parse_with_test_dialect(&generic).unwrap_or_else(|err| panic!("{err}\n{generic}"));
    assert!(same_program(&second.borrow().to_string(), &custom), "{custom}\n{generic}");
    assert!(
        same_program(&print_with(&second.borrow(), &generic_flags), &generic),
        "{generic}"
    );
    generic
}

/// The entry block of a callable is printed with its label and arguments in the generic form, as
/// nothing else in the text gives them, and reads back.
#[test]
fn the_generic_form_labels_the_entry_block_of_a_callable() {
    let generic = generic_reads_back(
        "builtin.function public extern(\"C\") @f(%a: i32, %b: i64) -> i32 {\n    builtin.ret %a \
         : (i32);\n};",
    );
    assert!(generic.contains("({\n^block1(%0: i32, %1: i64):\n"), "{generic}");
}

/// The regions of an operation in the generic form are separated by commas.
#[test]
fn the_generic_form_separates_regions_with_commas() {
    let generic =
        generic_reads_back("builtin.module public @m {\n    builtin.module public @a { };\n};");
    // One region each here; the separator is checked on the text of two.
    assert!(generic.contains("({ })"), "{generic}");
    let (_context, module) = parse_with_test_dialect("builtin.module public @m { };")
        .unwrap_or_else(|err| panic!("{err}"));
    let flags = crate::OpPrintingFlags {
        generic: true,
        ..Default::default()
    };
    let mut module = module;
    let region = module.borrow().context_rc().create_region();
    module.borrow_mut().regions_mut().push_back(region);
    let two = print_with(&module.borrow(), &flags);
    assert!(two.contains("({ }, { })"), "{two}");
}

/// An operation that reads a value defined later in its block, which only a graph region allows,
/// is printed in the generic form, which states the operand's type: its custom form, which takes
/// the type from the operand, could not be read back. The operations around it keep their custom
/// forms.
#[test]
fn an_operation_with_a_forward_operand_prints_in_the_generic_form() {
    let text = "\
builtin.module public @m {
    %0 = \"test.add\"(%1, %1) <{ overflow = #builtin.overflow<wrapping> }> : (i32, i32) -> i32;
    %1 = test.constant 1 : i32;
    %2 = test.add %0, %1;
};";
    let (_context, module) = parse_with_test_dialect(text).unwrap_or_else(|err| panic!("{err}"));
    let printed = module.borrow().to_string();
    assert_eq!(
        printed,
        "\
builtin.module public @m {
    %1 = \"test.add\"(%2, %2) <{ overflow = #builtin.overflow<wrapping> }> : (i32, i32) -> i32;
    %2 = test.constant 1 : i32;
    %3 = test.add %1, %2;
};"
    );
    let (_context, again) = parse_with_test_dialect(&printed).unwrap_or_else(|err| panic!("{err}"));
    assert!(crate::print::same_program(&again.borrow().to_string(), &printed));
}

/// A region whose one block a pass emptied prints as the region parsed from `{ }` does.
#[test]
fn an_emptied_region_prints_as_an_empty_region() {
    let (_context, module) =
        parse_with_test_dialect("builtin.module public @m {\n    %0 = test.constant 1 : i32;\n};")
            .unwrap_or_else(|err| panic!("{err}"));
    let mut constant = module.borrow().regions().front().get().unwrap().entry().front().unwrap();
    constant.borrow_mut().erase();
    assert_eq!(module.borrow().to_string(), "builtin.module public @m { };");
    let (_context, parsed) = parse_with_test_dialect("builtin.module public @m { };")
        .unwrap_or_else(|err| panic!("{err}"));
    assert_eq!(parsed.borrow().to_string(), module.borrow().to_string());
}
