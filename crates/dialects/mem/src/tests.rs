use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    Align, Context, Int, Op, OperationRef, Report, SourceSpan, SymbolTable, ValueRef,
    dialects::builtin::{BuiltinOpBuilder, Function, Module, attributes::StringAttr},
    effects::{
        AutomaticAllocationScopeResource, DefaultResource, MemoryEffect, MemoryEffectOpInterface,
        Resource,
    },
    parse::{self, ParserConfig},
    pass::{Nesting, PassManager},
    testing::{
        Test,
        text::{parse, round_trip},
    },
    traits::{OperandRangeRequirement, operation_operand_range_requirement},
};
use runic_dialect_arith::Constant;
use runic_expect_test::expect;
use runic_transform::{CommonSubexpressionElimination, DeadCodeElimination};

use crate::*;

/// The data layout the tests that need one state on their module.
const LAYOUT: &str = "#builtin.data_layout<little, ptr = 32>";

/// A function `@f` with the parameters `params` (as written in text, e.g. `%p: ptr<i32>`) and the
/// body `body`, ending in `builtin.ret;`.
fn function(params: &str, body: &str) -> String {
    format!("builtin.function public extern(\"C\") @f({params}) {{\n{body}\n    builtin.ret;\n}};")
}

/// `text`, a function, parsed and verified; the error is described with its labels.
fn parse_function(
    text: &str,
) -> Result<(Rc<Context>, runic_core::dialects::builtin::FunctionRef), String> {
    parse::<Function>(text)
}

/// The operations of the entry block of `function`, without its terminator.
fn body_ops(function: runic_core::dialects::builtin::FunctionRef) -> Vec<OperationRef> {
    let entry = function.borrow().entry_block();
    let entry = entry.borrow();
    entry
        .body()
        .iter()
        .map(|op| op.as_operation_ref())
        .filter(|op| !op.borrow().name().to_string().starts_with("builtin.ret"))
        .collect()
}

/// Every form of the three operations in text.
const EVERY_FORM: &str = "\
builtin.function public extern(\"C\") @f(%p: ptr<i32>, %q: ptr, %v: i32) {
    %0 = mem.load %p;
    %1 = mem.load %q : i32;
    %2 = mem.load %p, align 1;
    %3 = mem.load %q, align 8 : i64;
    mem.store %v, %p;
    mem.store %v, %p, align 1;
    mem.store %0, %q, align 2;
    %4 = mem.alloca i32;
    %5 = mem.alloca struct<i32, i8>, align 16;
    %6 = mem.load %5;
    builtin.ret;
};";

/// Every form prints as it was written, and round trips through text, with each operation also
/// written in the generic form.
#[test]
fn every_form_round_trips() {
    let (_context, function) = parse_function(EVERY_FORM).expect("valid ir");
    let printed = format!("{}", function.borrow().as_operation());
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32>, %1: ptr, %2: i32) {
            %3 = mem.load %0;
            %4 = mem.load %1 : i32;
            %5 = mem.load %0, align 1;
            %6 = mem.load %1, align 8 : i64;
            mem.store %2, %0;
            mem.store %2, %0, align 1;
            mem.store %3, %1, align 2;
            %7 = mem.alloca i32;
            %8 = mem.alloca struct<i32, i8>, align 16;
            %9 = mem.load %8;
            builtin.ret;
        };"#]]
    .assert_eq(&printed);

    let mut skipped = Vec::new();
    let failures = round_trip::<Function>(function.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

/// The generic forms: the properties `ty` and `align`, with the alignment as `#mem.align<..>`.
#[test]
fn the_generic_forms_carry_the_properties() {
    let (_context, function) = parse_function(EVERY_FORM).expect("valid ir");
    let generic = body_ops(function)
        .into_iter()
        .map(|op| runic_core::testing::text::print_generic(&op.borrow()))
        .collect::<Vec<_>>()
        .join("\n");
    expect![[r#"
        %3 = "mem.load"(%0) <{ ty = #builtin.type<?>, align = #mem.align<natural> }> : ptr<i32> -> i32;
        %4 = "mem.load"(%1) <{ ty = #builtin.type<i32>, align = #mem.align<natural> }> : ptr -> i32;
        %5 = "mem.load"(%0) <{ ty = #builtin.type<?>, align = #mem.align<1> }> : ptr<i32> -> i32;
        %6 = "mem.load"(%1) <{ ty = #builtin.type<i64>, align = #mem.align<8> }> : ptr -> i64;
        "mem.store"(%2, %0) <{ align = #mem.align<natural> }> : (i32, ptr<i32>) -> ();
        "mem.store"(%2, %0) <{ align = #mem.align<1> }> : (i32, ptr<i32>) -> ();
        "mem.store"(%3, %1) <{ align = #mem.align<2> }> : (i32, ptr) -> ();
        %7 = "mem.alloca"() <{ ty = #builtin.type<i32>, align = #mem.align<natural> }> : () -> ptr<i32>;
        %8 = "mem.alloca"() <{ ty = #builtin.type<struct<i32, i8>>, align = #mem.align<16> }> : () -> ptr<struct<i32, i8>>;
        %9 = "mem.load"(%8) <{ ty = #builtin.type<?>, align = #mem.align<natural> }> : ptr<struct<i32, i8>> -> struct<i32, i8>;"#]]
    .assert_eq(&generic);
}

/// The generic form may leave out a property at its default, and infers the result of a load or
/// an alloca from its operand and properties.
#[test]
fn the_generic_form_may_leave_out_defaults() {
    let text = function(
        "%p: ptr<i32>, %q: ptr, %v: i32",
        r#"    %1 = "mem.load"(%p) : ptr<i32> -> i32;
    %2 = "mem.load"(%q) <{ ty = #builtin.type<i8>, align = #mem.align<4> }> : ptr -> i8;
    "mem.store"(%v, %p) : (i32, ptr<i32>) -> ();
    %3 = "mem.alloca"() <{ ty = #builtin.type<i64> }> : () -> ptr<i64>;"#,
    );
    let (_context, function) = parse_function(&text).expect("valid ir");
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32>, %1: ptr, %2: i32) {
            %3 = mem.load %0;
            %4 = mem.load %1, align 4 : i8;
            mem.store %2, %0;
            %5 = mem.alloca i64;
            builtin.ret;
        };"#]]
    .assert_eq(&format!("{}", function.borrow().as_operation()));
}

/// A typed pointer may be followed by its pointee as the written type; the printer writes the
/// type only for an opaque pointer.
#[test]
fn a_typed_pointer_may_state_its_pointee() {
    let text = function("%p: ptr<i32>", "    %1 = mem.load %p, align 2 : i32;");
    let (context, function) = parse_function(&text).expect("valid ir");
    let load = body_ops(function)[0];
    assert_eq!(*load.borrow().downcast_ref::<Load>().unwrap().get_ty(), context.i32());
    let printed = format!("{}", function.borrow().as_operation());
    assert!(printed.contains("%1 = mem.load %0, align 2;"), "{printed}");
}

/// The builder makes every form, with the text the parser reads.
#[test]
fn the_builder_makes_every_form() -> Result<(), Report> {
    let mut test =
        Test::new("f", |cx| ([cx.ptr_type(cx.i32()), cx.opaque_ptr_type(), cx.i32()], []));
    let context = test.context_rc();
    let [p, q, v] = {
        let entry = test.entry_block();
        let entry = entry.borrow();
        [0, 1, 2].map(|i| entry.arguments()[i] as ValueRef)
    };
    let four = Alignment::Bytes(Align::from_bytes(4).unwrap());
    {
        let mut b = test.function_builder();
        let loaded = b.load(p, SourceSpan::UNKNOWN)?;
        assert_eq!(loaded.borrow().ty(), context.i32());
        let wide = b.load_as(q, context.i64(), SourceSpan::UNKNOWN)?;
        assert_eq!(wide.borrow().ty(), context.i64());
        b.load_aligned(p, four, SourceSpan::UNKNOWN)?;
        b.load_as_aligned(q, context.i8(), four, SourceSpan::UNKNOWN)?;
        let store = b.store(v, p, SourceSpan::UNKNOWN)?;
        assert_eq!(*store.borrow().get_align(), Alignment::Natural);
        b.store_aligned(wide, q, four, SourceSpan::UNKNOWN)?;
        let slot = b.alloca(context.i8(), SourceSpan::UNKNOWN)?;
        assert_eq!(slot.borrow().ty(), context.ptr_type(context.i8()));
        b.alloca_aligned(context.i64(), four, SourceSpan::UNKNOWN)?;
        b.ret([], SourceSpan::UNKNOWN)?;
    }
    let function = test.function();
    function.borrow().as_operation().recursively_verify()?;
    let printed = format!("{}", function.borrow().as_operation());
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32>, %1: ptr, %2: i32) {
            %3 = mem.load %0;
            %4 = mem.load %1 : i64;
            %5 = mem.load %0, align 4;
            %6 = mem.load %1, align 4 : i8;
            mem.store %2, %0;
            mem.store %4, %1, align 4;
            %7 = mem.alloca i8;
            %8 = mem.alloca i64, align 4;
            builtin.ret;
        };"#]]
    .assert_eq(&printed);

    let mut skipped = Vec::new();
    let failures = round_trip::<Function>(function.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

/// The alignment attribute prints and parses `natural` and every power of two from 1 to 32768;
/// any other number is a parse diagnostic, in the attribute and in an operation's text.
#[test]
fn alignments_are_powers_of_two_up_to_32768() {
    for log in 0..=15u32 {
        let bytes = 1u64 << log;
        let text = function("%p: ptr<i32>", &format!("    %1 = mem.load %p, align {bytes};"));
        let (_context, function) = parse_function(&text).expect("valid ir");
        let load = body_ops(function)[0];
        let load = load.borrow();
        let load = load.downcast_ref::<Load>().unwrap();
        assert_eq!(load.get_align().bytes().map(Align::bytes), Some(bytes));
    }
    for bad in ["0", "3", "6", "65536", "4294967296"] {
        for body in [
            format!("    %1 = mem.load %p, align {bad};"),
            format!("    mem.store %v, %p, align {bad};"),
            format!("    %1 = mem.alloca i32, align {bad};"),
            format!(
                "    %1 = \"mem.load\"(%p) <{{ align = #mem.align<{bad}> }}> : ptr<i32> -> i32;"
            ),
        ] {
            let text = function("%p: ptr<i32>, %v: i32", &body);
            let err = parse_function(&text).err().unwrap_or_else(|| panic!("`{body}` parsed"));
            assert!(
                err.contains("an alignment is a power of two from 1 to 32768"),
                "{body}: {err}"
            );
        }
    }
    for body in ["    %1 = mem.load %p, align;", "    %1 = mem.load %p, natural;"] {
        let text = function("%p: ptr<i32>", body);
        assert!(parse_function(&text).is_err(), "`{body}` parsed");
    }
    let text = function(
        "%p: ptr<i32>",
        "    %1 = \"mem.load\"(%p) <{ align = #mem.align }> : ptr<i32> -> i32;",
    );
    let (_context, function) = parse_function(&text).expect("a body-less alignment is natural");
    let printed = format!("{}", function.borrow().as_operation());
    assert!(printed.contains("mem.load %0;"), "{printed}");
}

/// The verifier's diagnostics: a store of a value that is not the pointee, a load through an
/// opaque pointer that does not state its type, a typed pointer with another stated type, a
/// result that is not the type read or allocated (in the generic form), and an alloca of an
/// unknown type.
#[test]
fn the_verifier_rejects_mismatched_types() {
    let cases = [
        (
            "%p: ptr<i32>, %v: i8",
            "    mem.store %v, %p;",
            "the value has type `i8`, but the pointer points to `i32`",
        ),
        (
            "%q: ptr",
            "    %1 = mem.load %q;",
            "a load through the opaque pointer type `ptr` must state the type it reads",
        ),
        (
            "%p: ptr<i32>",
            "    %1 = mem.load %p : i64;",
            "the pointer points to `i32`, but the load states that it reads `i64`",
        ),
        (
            "%p: ptr<i32>",
            "    %1 = \"mem.load\"(%p) : ptr<i32> -> i64;",
            "the result has type `i64`, but the load reads `i32`",
        ),
        (
            "%q: ptr",
            "    %1 = \"mem.load\"(%q) <{ ty = #builtin.type<i32> }> : ptr -> i8;",
            "the result has type `i8`, but the load reads `i32`",
        ),
        (
            "%p: ptr<i32>",
            "    %1 = \"mem.load\"(%p) <{ ty = #builtin.type<i8> }> : ptr<i32> -> i8;",
            "the pointer points to `i32`, but the load states that it reads `i8`",
        ),
        (
            "",
            "    %1 = \"mem.alloca\"() <{ ty = #builtin.type<i32> }> : () -> ptr<i64>;",
            "the result has type `ptr<i64>`, but an alloca of `i32` gives `ptr<i32>`",
        ),
        (
            "",
            "    %1 = \"mem.alloca\"() <{ ty = #builtin.type<i32> }> : () -> ptr<i32, 1>;",
            "the result has type `ptr<i32, 1>`, but an alloca of `i32` gives `ptr<i32>`",
        ),
        (
            "",
            "    %1 = \"mem.alloca\"() <{ ty = #builtin.type<?> }> : () -> ptr<?>;",
            "the allocated type is not known",
        ),
    ];
    for (params, body, expected) in cases {
        let err = parse_function(&function(params, body))
            .err()
            .unwrap_or_else(|| panic!("`{body}` verified"));
        assert!(err.contains(expected), "{body}: {err}");
        assert!(err.contains("invalid operation mem."), "{body}: {err}");
    }
    let err = parse_function(&function("", "    %1 = \"mem.alloca\"() : () -> ptr<i32>;"))
        .err()
        .expect("an alloca states its type");
    assert!(err.contains("missing required property 'ty'"), "{err}");
}

/// An address that is not a pointer is rejected when the operation is made, by the constraint
/// of its operand.
#[test]
fn an_address_must_be_a_pointer() {
    for body in [
        "    %1 = mem.load %v : i32;",
        "    mem.store %v, %v;",
        "    %1 = \"mem.load\"(%v) <{ ty = #builtin.type<i32> }> : i32 -> i32;",
    ] {
        let err = parse_function(&function("%v: i32", body))
            .err()
            .unwrap_or_else(|| panic!("`{body}` parsed"));
        assert!(err.contains("invalid operand"), "{body}: {err}");
        assert!(err.contains("pointer"), "{body}: {err}");
    }
}

/// A module stating [LAYOUT] that holds `declarations` and a function `@f(params)` with `body`.
fn module_with_layout(layout: bool, declarations: &str, params: &str, body: &str) -> String {
    let attributes = if layout {
        format!(" attributes {{ data_layout = {LAYOUT} }}")
    } else {
        String::new()
    };
    format!(
        "{declarations}\nbuiltin.module public @m{attributes} {{\n{}\n}};",
        function(params, body)
    )
}

/// Under a data layout, a load of a value of function type and an alloca of a struct that holds
/// itself by value are diagnostics that name the type; with no layout in scope, they verify.
#[test]
fn the_accessed_type_needs_a_layout_when_one_is_in_scope() {
    let rec = "rec { type !s = struct s<i32, !s>; }";
    let cases = [
        (
            "",
            "%p: ptr<(i32) -> i32>",
            "    %1 = mem.load %p;",
            "the type read, `(i32) -> i32`, has no layout under the data layout in scope: type \
             '(i32) -> i32' has no representation in memory",
        ),
        (
            "",
            "%q: ptr, %v: (i32) -> i32",
            "    mem.store %v, %q;",
            "the type written, `(i32) -> i32`, has no layout",
        ),
        (rec, "", "    %1 = mem.alloca !s;", "has infinite size"),
        (
            "",
            "",
            "    %1 = mem.alloca (i32) -> i32;",
            "the allocated type, `(i32) -> i32`",
        ),
    ];
    for (declarations, params, body, expected) in cases {
        let with = module_with_layout(true, declarations, params, body);
        let err = parse::<Module>(&with).err().unwrap_or_else(|| panic!("`{body}` verified"));
        assert!(err.contains(expected), "{body}: {err}");
        assert!(err.contains("invalid operation mem."), "{body}: {err}");

        let without = module_with_layout(false, declarations, params, body);
        parse::<Module>(&without).unwrap_or_else(|err| panic!("`{body}`: {err}"));
    }
    // A layout in scope with types that have one
    let fine = module_with_layout(
        true,
        "",
        "%p: ptr<i64>, %v: i64",
        "    %1 = mem.load %p;\n    mem.store %v, %p;\n    %2 = mem.alloca struct<i32, i8>;",
    );
    parse::<Module>(&fine).unwrap();
}

/// The effects: a load reads and a store writes [MemoryResource] at the address, a child of the
/// default resource; an alloca allocates its result in the automatic allocation scope.
#[test]
fn the_effects_name_their_resources() {
    let (_context, function) = parse_function(&function(
        "%p: ptr<i32>, %v: i32",
        "    %1 = mem.load %p;\n    mem.store %v, %p;\n    %2 = mem.alloca i32;",
    ))
    .unwrap();
    let ops = body_ops(function);
    let p = function.borrow().entry_block().borrow().arguments()[0] as ValueRef;
    let effects_of = |op: OperationRef| {
        let op = op.borrow();
        assert!(op.implements::<dyn MemoryEffectOpInterface>());
        op.get_effects_recursively::<MemoryEffect>().unwrap()
    };

    let load = effects_of(ops[0]);
    assert_eq!(load.len(), 1);
    assert_eq!(*load[0].effect(), MemoryEffect::Read);
    assert_eq!(load[0].value(), Some(p));
    assert!(load[0].resource().dyn_eq(&MemoryResource));

    let store = effects_of(ops[1]);
    assert_eq!(store.len(), 1);
    assert_eq!(*store[0].effect(), MemoryEffect::Write);
    assert_eq!(store[0].value(), Some(p));
    assert!(store[0].resource().dyn_eq(&MemoryResource));

    let alloca = effects_of(ops[2]);
    assert_eq!(alloca.len(), 1);
    assert_eq!(*alloca[0].effect(), MemoryEffect::Allocate);
    let result = ops[2].borrow().results()[0] as ValueRef;
    assert_eq!(alloca[0].value(), Some(result));
    assert!(alloca[0].resource().dyn_eq(&AutomaticAllocationScopeResource));

    assert_eq!(MemoryResource.parent().unwrap().name(), DefaultResource.name());
    let memory: &dyn Resource = &MemoryResource;
    assert!(!memory.is_disjoint_from(&DefaultResource));
    assert!(memory.is_disjoint_from(&runic_core::dialects::test::TestMemoryResource));

    assert!(ops[0].borrow().would_be_trivially_dead());
    assert!(!ops[1].borrow().would_be_trivially_dead());
    assert!(ops[2].borrow().would_be_trivially_dead());
}

/// A dereferenced address is `Unexpressed`; the stored value has no requirement.
#[test]
fn a_dereferenced_address_has_an_unexpressed_requirement() {
    let (_context, function) = parse_function(&function(
        "%p: ptr<i32>, %v: i32",
        "    %1 = mem.load %p;\n    mem.store %v, %p;",
    ))
    .unwrap();
    let ops = body_ops(function);
    let load = ops[0].borrow();
    assert_eq!(
        operation_operand_range_requirement(&load, 0),
        OperandRangeRequirement::Unexpressed
    );
    let store = ops[1].borrow();
    assert_eq!(operation_operand_range_requirement(&store, 0), OperandRangeRequirement::None);
    assert_eq!(
        operation_operand_range_requirement(&store, 1),
        OperandRangeRequirement::Unexpressed
    );
}

/// The dialect materializes an integer through `arith`, and nothing else.
#[test]
fn materialize_constant_delegates_integers_to_arith() {
    let mut test = Test::new("f", |_| ([], []));
    let context = test.context_rc();
    let dialect = context.get_or_register_dialect::<MemDialect>();

    let int = context.int_attr(Int::from(5i32), context.i32()).as_attribute_ref();
    let op = dialect
        .materialize_constant(test.builder_mut(), int, context.i32(), SourceSpan::UNKNOWN)
        .expect("an integer materializes");
    assert!(op.borrow().is::<Constant>());

    let string = context.create_attribute::<StringAttr, _>("no").as_attribute_ref();
    assert!(
        dialect
            .materialize_constant(test.builder_mut(), string, context.i32(), SourceSpan::UNKNOWN)
            .is_none()
    );
}

/// Callees for the pass tests: `@unknown` states no effects.
const CALLEES: &str = r#"
    builtin.function public extern("C") @unknown(ptr<i32>) -> i32;"#;

/// The function `@f(%p: ptr<i32>, %v: i32) -> (i32, i32)` with the body `body`, in a module that
/// also holds [CALLEES], after `pass` ran on it with verification, as text.
fn run_pass(pass: Box<dyn runic_core::pass::OperationPass>, body: &str) -> Result<String, Report> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    let text = format!(
        "builtin.module public @m {{\n{CALLEES}\nbuiltin.function public extern(\"C\") @f(%p: \
         ptr<i32>, %v: i32) -> (i32, i32) {{\n{body}\n}};\n}};"
    );
    let module =
        parse::parse::<Module>(ParserConfig::new(context.clone()), "mem", text).into_result()?;
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
    passes.add_pass(pass);
    passes.enable_verifier(true);
    passes.run(function)?;
    Ok(function.borrow().to_string())
}

fn dce(body: &str) -> Result<String, Report> {
    run_pass(Box::<DeadCodeElimination>::default(), body)
}

fn cse(body: &str) -> Result<String, Report> {
    run_pass(Box::<CommonSubexpressionElimination>::default(), body)
}

/// Dead-code elimination erases an unused load and an unused alloca, and keeps a store.
#[test]
fn dce_erases_unused_loads_and_allocas_and_keeps_stores() -> Result<(), Report> {
    let text = dce(r#"
    %1 = mem.load %p;
    %2 = mem.alloca i64;
    mem.store %v, %p;
    builtin.ret %v, %v : (i32, i32);"#)?;
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32>, %1: i32) -> (i32, i32) {
            mem.store %1, %0;
            builtin.ret %1, %1 : (i32, i32);
        };"#]]
    .assert_eq(&text);
    Ok(())
}

/// Two equal `mem.load`s of `%p` with `between` between them, after CSE: the number of loads left.
/// Asserts that "one load left" and "both results are one value" agree.
fn loads_left_with_between(between: &str) -> Result<usize, Report> {
    let body = format!(
        "%1 = mem.load %p;\n{between}\n%2 = mem.load %p;\nbuiltin.ret %1, %2 : (i32, i32);"
    );
    let text = cse(&body)?;
    let ret = text.lines().find(|line| line.contains("builtin.ret")).unwrap();
    let operands = ret.trim().trim_start_matches("builtin.ret").split(':').next().unwrap();
    let operands = operands.split(',').map(str::trim).collect::<Vec<_>>();
    let merged = operands[0] == operands[1];
    let left = text.matches("mem.load").count();
    assert_eq!(merged, left == 1, "{text}");
    Ok(left)
}

/// Two equal loads merge; across a `cf.assert` and across a `test.store` (another resource)
/// too.
#[test]
fn equal_loads_merge_across_other_resources() -> Result<(), Report> {
    assert_eq!(loads_left_with_between("")?, 1);
    let assert = "%c = arith.cmpi eq %v, %v;\ncf.assert %c, \"between\";";
    assert_eq!(loads_left_with_between(assert)?, 1);
    assert_eq!(loads_left_with_between("test.store %p, %v : (ptr<i32>, i32);")?, 1);

    let text = cse(&format!(
        "%1 = mem.load %p;\n{assert}\n%2 = mem.load %p;\nbuiltin.ret %1, %2 : (i32, i32);"
    ))?;
    assert!(text.contains("cf.assert"), "{text}");
    Ok(())
}

/// Two loads do not merge across a `mem.store`, or across a call with unknown effects; loads of
/// two types through one opaque pointer, or with two alignments, do not merge either.
#[test]
fn loads_do_not_merge_across_stores_or_unknown_calls() -> Result<(), Report> {
    for between in [
        "mem.store %v, %p;",
        r#"%3 = builtin.call @unknown(%p) : extern("C") (ptr<i32>) -> i32;"#,
    ] {
        assert_eq!(loads_left_with_between(between)?, 2, "{between}");
    }
    let text = cse(r#"
    %1 = mem.load %p;
    %2 = mem.load %p, align 1;
    builtin.ret %1, %2 : (i32, i32);"#)?;
    assert_eq!(text.matches("mem.load").count(), 2, "{text}");
    Ok(())
}

/// Two equal stores are both kept, and two allocas are two storages.
#[test]
fn stores_and_allocas_are_never_merged() -> Result<(), Report> {
    let text = cse(r#"
    %1 = mem.alloca i32;
    %2 = mem.alloca i32;
    mem.store %v, %1;
    mem.store %v, %2;
    mem.store %v, %p;
    mem.store %v, %p;
    builtin.ret %v, %v : (i32, i32);"#)?;
    assert_eq!(text.matches("mem.alloca").count(), 2, "{text}");
    assert_eq!(text.matches("mem.store").count(), 4, "{text}");
    Ok(())
}
