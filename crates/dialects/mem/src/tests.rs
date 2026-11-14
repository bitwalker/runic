use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    Align, Context, Int, Op, OperationRef, Report, SmallVec, SourceSpan, SymbolTable, ValueRef,
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
use runic_transform::{Canonicalizer, CommonSubexpressionElimination, DeadCodeElimination};

use crate::*;

mod elements;
mod folds;
mod fuzz;
mod probes;

/// The seed of a randomized test: `MEM_TEST_SEED` if set, else `default`. It is printed, so a
/// failing test shows the seed that reproduces it.
fn seed(default: u64) -> u64 {
    let seed = match std::env::var("MEM_TEST_SEED") {
        Ok(seed) => seed.parse().expect("MEM_TEST_SEED is a u64"),
        Err(_) => default,
    };
    std::eprintln!("seed {seed:#x} (MEM_TEST_SEED={seed})");
    seed
}

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

/// A typed pointer may be followed by its pointee as the written type; the parser drops it, so
/// the load is the one written without it, and the printer writes the type only for an opaque
/// pointer.
#[test]
fn a_typed_pointer_may_state_its_pointee() {
    let text = function("%p: ptr<i32>", "    %1 = mem.load %p, align 2 : i32;");
    let (_context, function) = parse_function(&text).expect("valid ir");
    let load = body_ops(function)[0];
    assert!(load.borrow().downcast_ref::<Load>().unwrap().get_ty().is_unknown());
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
    passes.add_pass(pass).unwrap();
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

// --- gep, sizeof, alignof, offsetof --------------------------------------------------------------

/// Every form of `gep`, `sizeof`, `alignof` and `offsetof` in text.
const ADDRESS_FORMS: &str = "\
builtin.function public extern(\"C\") @f(%p: ptr<struct<i32, array<i8; 4>>>, %i: isize, %q: \
                             ptr<i64, 1>) {
    %0 = mem.gep %p, %i, [1, 2];
    %1 = mem.gep %p, [1];
    %2 = mem.gep %p, %i;
    %3 = mem.gep %p;
    %4 = mem.gep %q, %i;
    %5 = mem.gep %p, %i attributes { note = \"kept\" };
    %6 = mem.gep %p, [];
    %7 = mem.sizeof struct<i32, i8>;
    %8 = mem.alignof i64;
    %9 = mem.offsetof struct<i32, i8>, [1];
    %10 = mem.offsetof struct<i32, i8>;
    %11 = mem.offsetof struct<i32, struct<i8, i16>>, [1, 1] attributes { note = \"kept\" };
    builtin.ret;
};";

/// Every form prints as it was written (an empty path is left out) and round trips through text.
/// The size operations also round trip with their generic form substituted; a gep's does not
/// parse back (see [the_generic_form_of_a_gep_does_not_parse_back]).
#[test]
fn address_and_size_forms_round_trip() {
    let (_context, function) = parse_function(ADDRESS_FORMS).expect("valid ir");
    let printed = format!("{}", function.borrow().as_operation());
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<struct<i32, array<i8; 4>>>, %1: isize, %2: ptr<i64, 1>) {
            %3 = mem.gep %0, %1, [1, 2];
            %4 = mem.gep %0, [1];
            %5 = mem.gep %0, %1;
            %6 = mem.gep %0;
            %7 = mem.gep %2, %1;
            %8 = mem.gep %0, %1 attributes { note = #builtin.string<"kept"> };
            %9 = mem.gep %0;
            %10 = mem.sizeof struct<i32, i8>;
            %11 = mem.alignof i64;
            %12 = mem.offsetof struct<i32, i8>, [1];
            %13 = mem.offsetof struct<i32, i8>;
            %14 = mem.offsetof struct<i32, struct<i8, i16>>, [1, 1] attributes { note = #builtin.string<"kept"> };
            builtin.ret;
        };"#]]
    .assert_eq(&printed);

    let mut skipped = Vec::new();
    let failures = round_trip::<Function>(function.as_operation_ref(), false, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    let sizes = ADDRESS_FORMS
        .lines()
        .filter(|line| !line.contains("mem.gep"))
        .collect::<Vec<_>>()
        .join("\n");
    let (_context, function) = parse_function(&sizes).expect("valid ir");
    let failures = round_trip::<Function>(function.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

/// A gep has two operand groups, the base and the optional index, and the generic parser puts
/// every operand in one group, so the generic form of a gep, with or without an index, does not
/// parse back. The custom form is the one that round trips.
#[test]
fn the_generic_form_of_a_gep_does_not_parse_back() {
    for (body, expected) in [
        (
            r#"    %1 = "mem.gep"(%p) <{ path = #builtin.array_i32<1> }> : ptr<struct<i32, i8>> -> ptr<i8>;"#,
            "requires operand group 1",
        ),
        (
            r#"    %1 = "mem.gep"(%p, %i) : (ptr<struct<i32, i8>>, isize) -> ptr<struct<i32, i8>>;"#,
            "operand group 0 of 'mem.gep' expects 1 operands, but has 2",
        ),
    ] {
        let err = parse_function(&function("%p: ptr<struct<i32, i8>>, %i: isize", body))
            .err()
            .unwrap_or_else(|| panic!("`{body}` parsed"));
        assert!(err.contains(expected), "{body}: {err}");
    }
}

/// The generic forms: a gep's path and the measured type are properties.
#[test]
fn the_generic_forms_of_address_and_size_operations() {
    let (_context, parsed) = parse_function(ADDRESS_FORMS).expect("valid ir");
    let generic = body_ops(parsed)
        .into_iter()
        .map(|op| runic_core::testing::text::print_generic(&op.borrow()))
        .collect::<Vec<_>>()
        .join("\n");
    expect![[r#"
        %3 = "mem.gep"(%0, %1) <{ path = #builtin.array_i32<1, 2> }> : (ptr<struct<i32, array<i8; 4>>>, isize) -> ptr<i8>;
        %4 = "mem.gep"(%0) <{ path = #builtin.array_i32<1> }> : ptr<struct<i32, array<i8; 4>>> -> ptr<array<i8; 4>>;
        %5 = "mem.gep"(%0, %1) <{ path = #builtin.array_i32<> }> : (ptr<struct<i32, array<i8; 4>>>, isize) -> ptr<struct<i32, array<i8; 4>>>;
        %6 = "mem.gep"(%0) <{ path = #builtin.array_i32<> }> : ptr<struct<i32, array<i8; 4>>> -> ptr<struct<i32, array<i8; 4>>>;
        %7 = "mem.gep"(%2, %1) <{ path = #builtin.array_i32<> }> : (ptr<i64, 1>, isize) -> ptr<i64, 1>;
        %8 = "mem.gep"(%0, %1) <{ path = #builtin.array_i32<> }> { note = #builtin.string<"kept"> } : (ptr<struct<i32, array<i8; 4>>>, isize) -> ptr<struct<i32, array<i8; 4>>>;
        %9 = "mem.gep"(%0) <{ path = #builtin.array_i32<> }> : ptr<struct<i32, array<i8; 4>>> -> ptr<struct<i32, array<i8; 4>>>;
        %10 = "mem.sizeof"() <{ ty = #builtin.type<struct<i32, i8>> }> : () -> isize;
        %11 = "mem.alignof"() <{ ty = #builtin.type<i64> }> : () -> isize;
        %12 = "mem.offsetof"() <{ ty = #builtin.type<struct<i32, i8>>, path = #builtin.array_i32<1> }> : () -> isize;
        %13 = "mem.offsetof"() <{ ty = #builtin.type<struct<i32, i8>>, path = #builtin.array_i32<> }> : () -> isize;
        %14 = "mem.offsetof"() <{ ty = #builtin.type<struct<i32, struct<i8, i16>>>, path = #builtin.array_i32<1, 1> }> { note = #builtin.string<"kept"> } : () -> isize;"#]]
    .assert_eq(&generic);

    // The generic form of a size operation parses, and may leave out an empty path
    let text = function(
        "",
        r#"    %1 = "mem.sizeof"() <{ ty = #builtin.type<i8> }> : () -> isize;
    %2 = "mem.alignof"() <{ ty = #builtin.type<i16> }> : () -> isize;
    %3 = "mem.offsetof"() <{ ty = #builtin.type<struct<i32, i8>> }> : () -> isize;
    %4 = "mem.offsetof"() <{ ty = #builtin.type<struct<i32, i8>>, path = #builtin.array_i32<1> }> : () -> isize;"#,
    );
    let (_context, parsed) = parse_function(&text).expect("valid ir");
    expect![[r#"
        builtin.function public extern("C") @f() {
            %0 = mem.sizeof i8;
            %1 = mem.alignof i16;
            %2 = mem.offsetof struct<i32, i8>;
            %3 = mem.offsetof struct<i32, i8>, [1];
            builtin.ret;
        };"#]]
    .assert_eq(&format!("{}", parsed.borrow().as_operation()));
}

/// The parameters the gep tests use: a struct holding an array of structs, an enum in address
/// space 2, an index and an opaque pointer.
const GEP_PARAMS: &str = "%s: ptr<struct<i8, array<struct<i16, i32>; 4>>>, %e: ptr<enum<i8; a: \
                          i32, b, c: struct<i8, i16>>, 2>, %i: isize, %o: ptr, %v: i32";

/// The result of a gep is a pointer to the element its path selects, in the address space of the
/// base; the index does not change it.
#[test]
fn a_gep_points_to_the_element_its_path_selects() {
    let cases = [
        ("%s, [0]", "ptr<i8>"),
        ("%s, %i, [1]", "ptr<array<struct<i16, i32>; 4>>"),
        ("%s, [1, 3]", "ptr<struct<i16, i32>>"),
        ("%s, %i, [1, 2, 1]", "ptr<i32>"),
        ("%s, %i", "ptr<struct<i8, array<struct<i16, i32>; 4>>>"),
        ("%s", "ptr<struct<i8, array<struct<i16, i32>; 4>>>"),
        ("%e, [0]", "ptr<i8, 2>"),
        ("%e, %i, [1]", "ptr<i32, 2>"),
        ("%e, [3, 1]", "ptr<i16, 2>"),
        ("%e, %i", "ptr<enum<i8; a: i32, b, c: struct<i8, i16>>, 2>"),
    ];
    let body = cases
        .iter()
        .enumerate()
        .map(|(n, (operands, _))| format!("    %r{n} = mem.gep {operands};"))
        .collect::<Vec<_>>()
        .join("\n");
    let (_context, parsed) = parse_function(&function(GEP_PARAMS, &body)).expect("valid ir");
    for (op, (operands, expected)) in body_ops(parsed).into_iter().zip(cases) {
        let ty = runic_core::Value::ty(&*op.borrow().results()[0].borrow());
        assert_eq!(ty.to_string(), expected, "mem.gep {operands}");
    }
}

/// The address of a struct global is a pointer to the struct; a gep of it reaches a field, and a
/// load of that field reads the field's type.
#[test]
fn a_gep_of_a_global_reaches_its_fields() {
    let source = "\
builtin.module public @m {
    builtin.global_variable public @g : struct<i32, i8>;
    builtin.function public extern(\"C\") @f() -> i8 {
        %p = builtin.global_symbol @g : ptr<struct<i32, i8>>;
        %q = mem.gep %p, [1];
        %v = mem.load %q;
        builtin.ret %v : (i8);
    };
};";
    let (_context, module) = parse::<Module>(source).unwrap_or_else(|err| panic!("{err}"));
    let mut types = Vec::new();
    module.borrow().as_operation().prewalk_all(|op: &runic_core::Operation| {
        if op.num_results() == 1 {
            types.push((op.name().to_string(), runic_core::Value::ty(&*op.results()[0].borrow())));
        }
    });
    let types = types.into_iter().map(|(name, ty)| format!("{name}: {ty}")).collect::<Vec<_>>();
    assert_eq!(
        types,
        [
            "builtin.global_symbol: ptr<struct<i32, i8>>",
            "mem.gep: ptr<i8>",
            "mem.load: i8"
        ]
    );
}

/// A path into a scalar, past the end, into a variant without a payload, or with a negative step,
/// and a gep through an opaque pointer, are diagnostics; so is an index that is not an `isize`.
#[test]
fn the_verifier_rejects_invalid_geps() {
    let cases = [
        (
            "    %1 = mem.gep %s, [0, 0];",
            "the path [0, 0] selects no element of `struct<i8, array<struct<i16, i32>; 4>>`: type \
             'i8' has no element 0",
        ),
        (
            "    %1 = mem.gep %s, %i, [2];",
            "the path [2] selects no element of `struct<i8, array<struct<i16, i32>; 4>>`: type \
             'struct<i8, array<struct<i16, i32>; 4>>' has no element 2",
        ),
        ("    %1 = mem.gep %s, [1, 4];", "has no element 4"),
        (
            "    %1 = mem.gep %e, [2];",
            "type 'enum<i8; a: i32, b, c: struct<i8, i16>>' has no element 2",
        ),
        ("    %1 = mem.gep %e, [4];", "has no element 4"),
        ("    %1 = mem.gep %s, [1, -1];", "the path [1, -1] has the negative step -1"),
        (
            "    %1 = mem.gep %o, %i;",
            "the base has the opaque pointer type `ptr`, but a gep steps through a typed pointer: \
             cast the base to one first",
        ),
    ];
    for (body, expected) in cases {
        let err = parse_function(&function(GEP_PARAMS, body))
            .err()
            .unwrap_or_else(|| panic!("`{body}` verified"));
        assert!(err.contains(expected), "{body}: {err}");
        assert!(err.contains("invalid operation mem.gep"), "{body}: {err}");
    }
    for body in ["    %1 = mem.gep %s, %v;", "    %1 = mem.gep %v;"] {
        let err = parse_function(&function(GEP_PARAMS, body))
            .err()
            .unwrap_or_else(|| panic!("`{body}` parsed"));
        assert!(err.contains("invalid operand"), "{body}: {err}");
    }
}

/// An `offsetof` path is checked without a data layout; an unknown measured type is a
/// diagnostic.
#[test]
fn the_verifier_rejects_invalid_offsetof_paths() {
    let cases = [
        ("    %1 = mem.offsetof struct<i8, i32>, [2];", "the path [2] selects no element"),
        ("    %1 = mem.offsetof i32, [0];", "type 'i32' has no element 0"),
        ("    %1 = mem.offsetof enum<i8; a, b: i8>, [1];", "has no element 1"),
        (
            "    %1 = mem.offsetof struct<i8, i32>, [-1];",
            "the path [-1] has the negative step -1",
        ),
        (
            "    %1 = \"mem.sizeof\"() <{ ty = #builtin.type<?> }> : () -> isize;",
            "the type measured is not known",
        ),
        (
            "    %1 = \"mem.alignof\"() <{ ty = #builtin.type<?> }> : () -> isize;",
            "the type measured is not known",
        ),
    ];
    for (body, expected) in cases {
        for layout in [false, true] {
            let text = module_with_layout(layout, "", "", body);
            let err = parse::<Module>(&text).err().unwrap_or_else(|| panic!("`{body}` verified"));
            assert!(err.contains(expected), "{body}: {err}");
            assert!(err.contains("invalid operation mem."), "{body}: {err}");
        }
    }
    // The result is an `isize`
    let body = "    %1 = \"mem.sizeof\"() <{ ty = #builtin.type<i8> }> : () -> i64;";
    assert!(parse_function(&function("", body)).is_err(), "{body}");
}

/// Under a data layout, the measured type must have one; without one in scope, it need not.
#[test]
fn the_measured_type_needs_a_layout_when_one_is_in_scope() {
    let cases = [
        (
            "    %1 = mem.sizeof (i32) -> i32;",
            "the type measured, `(i32) -> i32`, has no layout under the data layout in scope",
        ),
        (
            "    %1 = mem.alignof (i32) -> i32;",
            "the type measured, `(i32) -> i32`, has no layout",
        ),
        (
            "    %1 = mem.offsetof struct<i32, (i32) -> i32>, [0];",
            "the type the element is selected in, `struct<i32, (i32) -> i32>`, has no layout",
        ),
    ];
    for (body, expected) in cases {
        let with = module_with_layout(true, "", "", body);
        let err = parse::<Module>(&with).err().unwrap_or_else(|| panic!("`{body}` verified"));
        assert!(err.contains(expected), "{body}: {err}");
        let without = module_with_layout(false, "", "", body);
        parse::<Module>(&without).unwrap_or_else(|err| panic!("`{body}`: {err}"));
    }
}

/// The address and size operations have no effects: an unused one is removed, and two equal ones
/// are one value.
#[test]
fn address_and_size_operations_are_pure() -> Result<(), Report> {
    let (_context, parsed) = parse_function(ADDRESS_FORMS).expect("valid ir");
    for op in body_ops(parsed) {
        let op = op.borrow();
        assert!(op.is_memory_effect_free(), "{}", op.name());
        assert!(op.would_be_trivially_dead(), "{}", op.name());
    }
    let text = cse(r#"
    %k = mem.sizeof i32;
    %l = mem.sizeof i32;
    %1 = mem.gep %p, %k;
    %2 = mem.gep %p, %l;
    %3 = mem.load %1;
    %4 = mem.load %2;
    builtin.ret %3, %4 : (i32, i32);"#)?;
    assert_eq!(text.matches("mem.sizeof").count(), 1, "{text}");
    assert_eq!(text.matches("mem.gep").count(), 1, "{text}");
    assert_eq!(text.matches("mem.load").count(), 1, "{text}");
    Ok(())
}

/// The function `@f(params) -> (results)` with `body`, in a module stating the data layout
/// `#builtin.data_layout<layout>` when `layout` is given, after the canonicalizer ran on it with
/// verification, as text.
fn canonicalize(
    layout: Option<&str>,
    params: &str,
    results: &str,
    body: &str,
) -> Result<String, Report> {
    let context = Rc::new(Context::default());
    let attributes = layout
        .map(|layout| format!(" attributes {{ data_layout = #builtin.data_layout<{layout}> }}"))
        .unwrap_or_default();
    let text = format!(
        "builtin.module public @m{attributes} {{\nbuiltin.function public extern(\"C\") \
         @f({params}) -> ({results}) {{\n{body}\n}};\n}};"
    );
    let module =
        parse::parse::<Module>(ParserConfig::new(context.clone()), "mem", text).into_result()?;
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
    passes.add_pass(Box::<Canonicalizer>::default()).unwrap();
    passes.enable_verifier(true);
    passes.run(function)?;
    Ok(function.borrow().to_string())
}

/// The sizes the brief names, under `layout`, after canonicalization.
fn folded_sizes(layout: Option<&str>) -> Result<String, Report> {
    canonicalize(
        layout,
        "",
        "isize, isize, isize, isize",
        r#"    %1 = mem.sizeof struct<i32, i8>;
    %2 = mem.alignof i64;
    %3 = mem.offsetof struct<i8, i32>, [1];
    %4 = mem.sizeof ptr<i8>;
    builtin.ret %1, %2, %3, %4 : (isize, isize, isize, isize);"#,
    )
}

/// Under a data layout, a size, an alignment and an offset fold to `isize` constants, which the
/// dialect materializes as `arith.constant`.
#[test]
fn sizes_fold_under_a_data_layout() -> Result<(), Report> {
    expect![[r#"
        builtin.function public extern("C") @f() -> (isize, isize, isize, isize) {
            %4 = arith.constant 8 : isize;
            %6 = arith.constant 4 : isize;
            builtin.ret %4, %4, %6, %6 : (isize, isize, isize, isize);
        };"#]]
    .assert_eq(&folded_sizes(Some("little, ptr = 32"))?);
    expect![[r#"
        builtin.function public extern("C") @f() -> (isize, isize, isize, isize) {
            %4 = arith.constant 8 : isize;
            %6 = arith.constant 4 : isize;
            builtin.ret %4, %4, %6, %4 : (isize, isize, isize, isize);
        };"#]]
    .assert_eq(&folded_sizes(Some("little, ptr = 64"))?);
    Ok(())
}

/// With no data layout in scope nothing folds.
#[test]
fn sizes_do_not_fold_without_a_data_layout() -> Result<(), Report> {
    expect![[r#"
        builtin.function public extern("C") @f() -> (isize, isize, isize, isize) {
            %0 = mem.sizeof struct<i32, i8>;
            %1 = mem.alignof i64;
            %2 = mem.offsetof struct<i8, i32>, [1];
            %3 = mem.sizeof ptr<i8>;
            builtin.ret %0, %1, %2, %3 : (isize, isize, isize, isize);
        };"#]]
    .assert_eq(&folded_sizes(None)?);
    Ok(())
}

/// A size that does not fit in the `isize` of the layout does not fold.
#[test]
fn a_size_folds_only_when_it_fits_in_an_isize() -> Result<(), Report> {
    let huge = |layout| {
        canonicalize(
            Some(layout),
            "",
            "isize",
            "    %1 = mem.sizeof array<i8; 1099511627776>;\n    builtin.ret %1 : (isize);",
        )
    };
    let narrow = huge("little, ptr = 32")?;
    assert!(narrow.contains("mem.sizeof array<i8; 1099511627776>"), "{narrow}");
    assert!(!narrow.contains("arith.constant"), "{narrow}");
    let wide = huge("little, ptr = 64")?;
    assert!(wide.contains("arith.constant 1099511627776 : isize"), "{wide}");
    assert!(!wide.contains("mem.sizeof"), "{wide}");
    Ok(())
}

/// A measured type without a layout, or an invalid path, does not fold and does not fail the
/// fold: the verifier reports it.
#[test]
fn an_unmeasurable_type_does_not_fold() {
    use runic_core::{OpFoldResult, traits::Foldable};

    let context = Rc::new(Context::default());
    let body = "    %1 = mem.sizeof (i32) -> i32;
    %2 = mem.alignof !s;
    %3 = mem.offsetof struct<i32, (i32) -> i32>, [1];
    %4 = mem.offsetof struct<i32, i8>, [2];
    %5 = mem.offsetof struct<i32, i8>, [-1];";
    let text = module_with_layout(true, "rec { type !s = struct s<i32, !s>; }", "", body);
    let module = parse::parse::<Module>(
        ParserConfig::new(context.clone()).verify_after_parse(false),
        "mem",
        text,
    )
    .into_result()
    .expect("parses");
    assert!(module.borrow().as_operation().recursively_verify().is_err());
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let function = function.borrow();
    let function = function.downcast_ref::<Function>().unwrap();
    let ops = body_ops(function.as_function_ref());
    assert_eq!(ops.len(), 5);
    for op in ops {
        let mut results = SmallVec::<[OpFoldResult; 1]>::default();
        let op = op.borrow();
        assert!(op.fold(&mut results).is_failed(), "{}", *op);
        assert!(results.is_empty());
    }
}

/// A gep with no index and an empty path is its base; a gep with no index whose base is another
/// gep becomes one gep on the inner base, with the inner index and both paths; a gep with an
/// index is not merged into the gep that defines its base.
#[test]
fn geps_canonicalize() -> Result<(), Report> {
    let text = canonicalize(
        None,
        "%s: ptr<struct<i8, array<struct<i16, i32>; 4>>>, %i: isize",
        "ptr<i32>, ptr<struct<i16, i32>>, ptr<struct<i8, array<struct<i16, i32>; 4>>>, ptr<i16>",
        r#"    %1 = mem.gep %s, %i, [1];
    %2 = mem.gep %1, [2];
    %3 = mem.gep %2, [1];
    %4 = mem.gep %s, [1];
    %5 = mem.gep %4, %i, [0];
    %6 = mem.gep %3;
    %7 = mem.gep %s;
    %8 = mem.gep %s, [1];
    %9 = mem.gep %8, [0, 0];
    builtin.ret %6, %5, %7, %9 : (ptr<i32>, ptr<struct<i16, i32>>, ptr<struct<i8, array<struct<i16, i32>; 4>>>, ptr<i16>);"#,
    )?;
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<struct<i8, array<struct<i16, i32>; 4>>>, %1: isize) -> (ptr<i32>, ptr<struct<i16, i32>>, ptr<struct<i8, array<struct<i16, i32>; 4>>>, ptr<i16>) {
            %4 = mem.gep %0, %1, [1, 2, 1];
            %5 = mem.gep %0, [1];
            %6 = mem.gep %5, %1, [0];
            %10 = mem.gep %0, [1, 0, 0];
            builtin.ret %4, %6, %0, %10 : (ptr<i32>, ptr<struct<i16, i32>>, ptr<struct<i8, array<struct<i16, i32>; 4>>>, ptr<i16>);
        };"#]]
    .assert_eq(&text);
    Ok(())
}

/// The builder makes every form of the address and size operations.
#[test]
fn the_builder_makes_address_and_size_operations() -> Result<(), Report> {
    let mut test = Test::new("f", |cx| {
        let inner = cx.struct_type(runic_core::dialects::builtin::types::StructType::new([
            cx.i16().into(),
            cx.i32().into(),
        ]));
        ([cx.ptr_type_in(Some(cx.array_type(inner, 4)), 3), cx.isize()], [])
    });
    let context = test.context_rc();
    let [p, i] = {
        let entry = test.entry_block();
        let entry = entry.borrow();
        [0, 1].map(|n| entry.arguments()[n] as ValueRef)
    };
    {
        let mut b = test.function_builder();
        let q = b.gep(p, Some(i), &[1, 1], SourceSpan::UNKNOWN)?;
        assert_eq!(q.borrow().ty(), context.ptr_type_in(Some(context.i32()), 3));
        b.gep(p, None, &[2], SourceSpan::UNKNOWN)?;
        b.gep(p, Some(i), &[], SourceSpan::UNKNOWN)?;
        b.gep(p, None, &[], SourceSpan::UNKNOWN)?;
        let n = b.sizeof(context.i64(), SourceSpan::UNKNOWN)?;
        assert_eq!(n.borrow().ty(), context.isize());
        b.alignof(context.i64(), SourceSpan::UNKNOWN)?;
        b.offsetof(context.i64(), &[], SourceSpan::UNKNOWN)?;
        let element = context.array_type(context.i8(), 2);
        b.offsetof(element, &[1], SourceSpan::UNKNOWN)?;
        assert!(b.gep(p, None, &[u32::MAX], SourceSpan::UNKNOWN).is_err());
        b.ret([], SourceSpan::UNKNOWN)?;
    }
    let function = test.function();
    function.borrow().as_operation().recursively_verify()?;
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<array<struct<i16, i32>; 4>, 3>, %1: isize) {
            %2 = mem.gep %0, %1, [1, 1];
            %3 = mem.gep %0, [2];
            %4 = mem.gep %0, %1;
            %5 = mem.gep %0;
            %6 = mem.sizeof i64;
            %7 = mem.alignof i64;
            %8 = mem.offsetof i64;
            %9 = mem.offsetof array<i8; 2>, [1];
            builtin.ret;
        };"#]]
    .assert_eq(&format!("{}", function.borrow().as_operation()));
    let mut skipped = Vec::new();
    let failures = round_trip::<Function>(function.as_operation_ref(), false, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

/// A gep built with two indices is a diagnostic: the index group holds at most one.
#[test]
fn a_gep_has_at_most_one_index() {
    let mut test = Test::new("f", |cx| ([cx.ptr_type(cx.i32()), cx.isize()], []));
    let [p, i] = {
        let entry = test.entry_block();
        let entry = entry.borrow();
        [0, 1].map(|n| entry.arguments()[n] as ValueRef)
    };
    let built = {
        let op_builder = runic_core::BuilderExt::create::<Gep, (_, [ValueRef; 2], _)>(
            test.builder_mut(),
            SourceSpan::UNKNOWN,
        );
        op_builder(p, [i, i], runic_core::dialects::builtin::attributes::Array::default())
    };
    let err = match built {
        Ok(gep) => gep.borrow().as_operation().verify().expect_err("two indices verified"),
        Err(err) => err,
    };
    let err = runic_core::testing::text::describe(&err);
    assert!(err.contains("a gep has at most one index, but this one has 2"), "{err}");
}

// --- memcpy, memmove, memset and the pointer casts ----------------------------------------------

/// Every form of the bulk operations and the casts in text.
const BULK_AND_CAST_FORMS: &str = "\
builtin.function public extern(\"C\") @f(%p: ptr<i32>, %q: ptr, %r: ptr<i8, 1>, %n: isize, %b: i8) \
                                   {
    mem.memcpy %p, %q, %n;
    mem.memmove %q, %r, %n;
    mem.memset %r, %b, %n;
    mem.memcpy %p, %p, %n attributes { note = \"kept\" };
    %0 = mem.ptrtoint %p;
    %1 = mem.ptrtoint %r;
    %2 = mem.inttoptr %n : ptr<i32>;
    %3 = mem.inttoptr %0 : ptr<i64, 2>;
    %4 = mem.inttoptr %0 : ptr;
    %5 = mem.ptrcast %p : ptr<i8>;
    %6 = mem.ptrcast %p : ptr;
    %7 = mem.ptrcast %q : ptr<i32>;
    %8 = mem.ptrcast %r : ptr<1>;
    %9 = mem.ptrcast %r : ptr<i16, 1> attributes { note = \"kept\" };
    builtin.ret;
};";

/// Every form prints as it was written and round trips through text, with each operation also
/// written in the generic form: each has one operand group.
#[test]
fn bulk_and_cast_forms_round_trip() {
    let (_context, function) = parse_function(BULK_AND_CAST_FORMS).expect("valid ir");
    let printed = format!("{}", function.borrow().as_operation());
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32>, %1: ptr, %2: ptr<i8, 1>, %3: isize, %4: i8) {
            mem.memcpy %0, %1, %3;
            mem.memmove %1, %2, %3;
            mem.memset %2, %4, %3;
            mem.memcpy %0, %0, %3 attributes { note = #builtin.string<"kept"> };
            %5 = mem.ptrtoint %0;
            %6 = mem.ptrtoint %2;
            %7 = mem.inttoptr %3 : ptr<i32>;
            %8 = mem.inttoptr %5 : ptr<i64, 2>;
            %9 = mem.inttoptr %5 : ptr;
            %10 = mem.ptrcast %0 : ptr<i8>;
            %11 = mem.ptrcast %0 : ptr;
            %12 = mem.ptrcast %1 : ptr<i32>;
            %13 = mem.ptrcast %2 : ptr<1>;
            %14 = mem.ptrcast %2 : ptr<i16, 1> attributes { note = #builtin.string<"kept"> };
            builtin.ret;
        };"#]]
    .assert_eq(&printed);

    let mut skipped = Vec::new();
    let failures = round_trip::<Function>(function.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(skipped.is_empty(), "{skipped:?}");
}

/// The generic forms: a cast's type is the property `ty`.
#[test]
fn the_generic_forms_of_bulk_and_cast_operations() {
    let (_context, parsed) = parse_function(BULK_AND_CAST_FORMS).expect("valid ir");
    let generic = body_ops(parsed)
        .into_iter()
        .map(|op| runic_core::testing::text::print_generic(&op.borrow()))
        .collect::<Vec<_>>()
        .join("\n");
    expect![[r#"
        "mem.memcpy"(%0, %1, %3) : (ptr<i32>, ptr, isize) -> ();
        "mem.memmove"(%1, %2, %3) : (ptr, ptr<i8, 1>, isize) -> ();
        "mem.memset"(%2, %4, %3) : (ptr<i8, 1>, i8, isize) -> ();
        "mem.memcpy"(%0, %0, %3) { note = #builtin.string<"kept"> } : (ptr<i32>, ptr<i32>, isize) -> ();
        %5 = "mem.ptrtoint"(%0) : ptr<i32> -> isize;
        %6 = "mem.ptrtoint"(%2) : ptr<i8, 1> -> isize;
        %7 = "mem.inttoptr"(%3) <{ ty = #builtin.type<ptr<i32>> }> : isize -> ptr<i32>;
        %8 = "mem.inttoptr"(%5) <{ ty = #builtin.type<ptr<i64, 2>> }> : isize -> ptr<i64, 2>;
        %9 = "mem.inttoptr"(%5) <{ ty = #builtin.type<ptr> }> : isize -> ptr;
        %10 = "mem.ptrcast"(%0) <{ ty = #builtin.type<ptr<i8>> }> : ptr<i32> -> ptr<i8>;
        %11 = "mem.ptrcast"(%0) <{ ty = #builtin.type<ptr> }> : ptr<i32> -> ptr;
        %12 = "mem.ptrcast"(%1) <{ ty = #builtin.type<ptr<i32>> }> : ptr -> ptr<i32>;
        %13 = "mem.ptrcast"(%2) <{ ty = #builtin.type<ptr<1>> }> : ptr<i8, 1> -> ptr<1>;
        %14 = "mem.ptrcast"(%2) <{ ty = #builtin.type<ptr<i16, 1>> }> { note = #builtin.string<"kept"> } : ptr<i8, 1> -> ptr<i16, 1>;"#]]
    .assert_eq(&generic);
}

/// The parameters the verifier tests use.
const CAST_PARAMS: &str = "%p: ptr<i32>, %q: ptr, %r: ptr<i8, 1>, %n: isize, %b: i8, %v: i32";

/// A cast to a type that is not a pointer, a `ptrcast` to another address space, and a result
/// that is not the stated type (in the generic form) are diagnostics.
#[test]
fn the_verifier_rejects_invalid_casts() {
    let cases = [
        (
            "    %1 = mem.inttoptr %n : i32;",
            "the result type of a cast to a pointer is a pointer type, not `i32`",
        ),
        (
            "    %1 = mem.ptrcast %p : isize;",
            "the result type of a cast to a pointer is a pointer type, not `isize`",
        ),
        (
            "    %1 = mem.ptrcast %p : ptr<i8, 1>;",
            "the operand `ptr<i32>` is in address space 0, but the cast to `ptr<i8, 1>` is in \
             address space 1: a cast keeps the address space",
        ),
        ("    %1 = mem.ptrcast %r : ptr;", "is in address space 1, but the cast to `ptr`"),
        ("    %1 = mem.ptrcast %q : ptr<i8, 3>;", "a cast keeps the address space"),
        (
            "    %1 = \"mem.inttoptr\"(%n) <{ ty = #builtin.type<ptr<i32>> }> : isize -> ptr<i8>;",
            "the result has type `ptr<i8>`, but the cast is to `ptr<i32>`",
        ),
        (
            "    %1 = \"mem.ptrcast\"(%p) <{ ty = #builtin.type<ptr> }> : ptr<i32> -> ptr<i8>;",
            "the result has type `ptr<i8>`, but the cast is to `ptr`",
        ),
    ];
    for (body, expected) in cases {
        let err = parse_function(&function(CAST_PARAMS, body))
            .err()
            .unwrap_or_else(|| panic!("`{body}` verified"));
        assert!(err.contains(expected), "{body}: {err}");
        assert!(err.contains("invalid operation mem."), "{body}: {err}");
    }
    let err =
        parse_function(&function(CAST_PARAMS, "    %1 = \"mem.ptrcast\"(%p) : ptr<i32> -> ptr;"))
            .err()
            .expect("a cast states its type");
    assert!(err.contains("missing required property 'ty'"), "{err}");
}

/// A length that is not an `isize`, a fill byte that is not an `i8`, a pointer operand that is not
/// a pointer, a `ptrtoint` result or `inttoptr` operand that is not an `isize`: the constraints of
/// the operands and results reject them.
#[test]
fn the_operand_constraints_reject_mistyped_operands() {
    for body in [
        "    mem.memcpy %p, %q, %v;",
        "    mem.memmove %p, %q, %b;",
        "    mem.memset %p, %b, %v;",
        "    mem.memset %p, %v, %n;",
        "    mem.memset %p, %n, %n;",
        "    mem.memcpy %n, %q, %n;",
        "    mem.memmove %p, %v, %n;",
        "    %1 = mem.ptrtoint %n;",
        "    %1 = mem.inttoptr %v : ptr<i32>;",
        "    %1 = mem.inttoptr %p : ptr<i32>;",
        "    %1 = mem.ptrcast %n : ptr<i32>;",
        "    %1 = \"mem.ptrtoint\"(%p) : ptr<i32> -> i64;",
    ] {
        let err = parse_function(&function(CAST_PARAMS, body))
            .err()
            .unwrap_or_else(|| panic!("`{body}` parsed"));
        assert!(
            err.contains("invalid operand") || err.contains("invalid result"),
            "{body}: {err}"
        );
    }
}

/// `memcpy` and `memmove` write [MemoryResource] at `dst` and read it at `src`; `memset` writes it
/// at `dst`; the casts have no effects. The pointers are `Unexpressed`, the length and byte have no
/// requirement.
#[test]
fn bulk_operations_write_memory_and_casts_are_pure() {
    let (_context, parsed) = parse_function(BULK_AND_CAST_FORMS).expect("valid ir");
    let [p, q, r] = {
        let entry = parsed.borrow().entry_block();
        let entry = entry.borrow();
        [0, 1, 2].map(|i| entry.arguments()[i] as ValueRef)
    };
    let ops = body_ops(parsed);
    let effects_of = |op: OperationRef| {
        let op = op.borrow();
        assert!(op.implements::<dyn MemoryEffectOpInterface>());
        op.get_effects_recursively::<MemoryEffect>()
            .unwrap()
            .into_iter()
            .map(|effect| {
                assert!(effect.resource().dyn_eq(&MemoryResource), "{}", op.name());
                (*effect.effect(), effect.value())
            })
            .collect::<Vec<_>>()
    };
    let mut copy = effects_of(ops[0]);
    copy.sort_by_key(|(effect, _)| *effect == MemoryEffect::Read);
    assert_eq!(copy, [(MemoryEffect::Write, Some(p)), (MemoryEffect::Read, Some(q))]);
    let mut moved = effects_of(ops[1]);
    moved.sort_by_key(|(effect, _)| *effect == MemoryEffect::Read);
    assert_eq!(moved, [(MemoryEffect::Write, Some(q)), (MemoryEffect::Read, Some(r))]);
    assert_eq!(effects_of(ops[2]), [(MemoryEffect::Write, Some(r))]);

    for op in &ops[..4] {
        let op = op.borrow();
        assert!(!op.would_be_trivially_dead(), "{}", op.name());
        let pointers = if op.is::<MemSet>() { 1 } else { 2 };
        for index in 0..3 {
            let expected = if index < pointers {
                OperandRangeRequirement::Unexpressed
            } else {
                OperandRangeRequirement::None
            };
            assert_eq!(
                operation_operand_range_requirement(&op, index),
                expected,
                "{} {index}",
                op.name()
            );
        }
    }
    for op in &ops[4..] {
        let op = op.borrow();
        assert!(op.is_memory_effect_free(), "{}", op.name());
        assert!(op.would_be_trivially_dead(), "{}", op.name());
    }
}

/// The function `@f(params) -> (results)` with `body`, after `pass` ran on it with verification,
/// as text.
fn run_pass_on(
    pass: Box<dyn runic_core::pass::OperationPass>,
    params: &str,
    results: &str,
    body: &str,
) -> Result<String, Report> {
    let context = Rc::new(Context::default());
    let text = format!(
        "builtin.module public @m {{\nbuiltin.function public extern(\"C\") @f({params}) -> \
         ({results}) {{\n{body}\n}};\n}};"
    );
    let module =
        parse::parse::<Module>(ParserConfig::new(context.clone()), "mem", text).into_result()?;
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
    passes.add_pass(pass).unwrap();
    passes.enable_verifier(true);
    passes.run(function)?;
    Ok(function.borrow().to_string())
}

/// Bulk operations of an unknown length and of a non-zero constant length, which no pass removes.
const KEPT_BULK_OPERATIONS: &str = r#"    %k = arith.constant 4 : isize;
    %b = arith.constant 7 : i8;
    mem.memcpy %p, %q, %n;
    mem.memmove %q, %p, %n;
    mem.memset %p, %b, %n;
    mem.memcpy %p, %q, %k;
    mem.memmove %q, %p, %k;
    mem.memset %q, %b, %k;
    builtin.ret;"#;

/// Neither dead-code elimination nor the canonicalizer removes a `memcpy`, `memmove` or `memset`
/// whose length is unknown or a non-zero constant.
#[test]
fn bulk_operations_are_kept_unless_their_length_is_zero() -> Result<(), Report> {
    let params = "%p: ptr<i32>, %q: ptr, %n: isize";
    for pass in [
        Box::<DeadCodeElimination>::default() as Box<dyn runic_core::pass::OperationPass>,
        Box::<Canonicalizer>::default(),
    ] {
        let text = run_pass_on(pass, params, "", KEPT_BULK_OPERATIONS)?;
        for op in ["mem.memcpy", "mem.memmove", "mem.memset"] {
            assert_eq!(text.matches(op).count(), 2, "{op}: {text}");
        }
    }
    Ok(())
}

/// A `memcpy`, `memmove` or `memset` of the constant length 0 is erased by the canonicalizer.
#[test]
fn bulk_operations_of_length_zero_are_erased() -> Result<(), Report> {
    let text = run_pass_on(
        Box::<Canonicalizer>::default(),
        "%p: ptr<i32>, %q: ptr, %n: isize",
        "",
        r#"    %z = arith.constant 0 : isize;
    %k = arith.constant 4 : isize;
    %b = arith.constant 7 : i8;
    mem.memcpy %p, %q, %z;
    mem.memmove %q, %p, %z;
    mem.memset %p, %b, %z;
    mem.memcpy %p, %q, %k;
    mem.memmove %q, %p, %n;
    mem.memset %q, %b, %k;
    builtin.ret;"#,
    )?;
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32>, %1: ptr, %2: isize) {
            %4 = arith.constant 4 : isize;
            %5 = arith.constant 7 : i8;
            mem.memcpy %0, %1, %4;
            mem.memmove %1, %0, %2;
            mem.memset %1, %5, %4;
            builtin.ret;
        };"#]]
    .assert_eq(&text);
    Ok(())
}

/// `memcpy`, `memmove` and `memset` write memory, so each blocks the merging of two loads, whatever
/// address it writes: a write to memory blocks the merging of reads of memory.
#[test]
fn bulk_operations_block_the_merging_of_loads() -> Result<(), Report> {
    let n = "%n = mem.ptrtoint %p;";
    let b = "%b = arith.constant 0 : i8;";
    let other = "%a = mem.alloca i64;";
    for between in [
        format!("{n}\nmem.memcpy %p, %p, %n;"),
        format!("{n}\nmem.memmove %p, %p, %n;"),
        format!("{n}\n{b}\nmem.memset %p, %b, %n;"),
        format!("{n}\n{other}\nmem.memcpy %a, %p, %n;"),
        format!("{n}\n{other}\nmem.memmove %a, %p, %n;"),
        format!("{n}\n{b}\n{other}\nmem.memset %a, %b, %n;"),
    ] {
        assert_eq!(loads_left_with_between(&between)?, 2, "{between}");
    }
    Ok(())
}

/// The casts are pure: two loads merge across them. A load through a cast of `%p` and a load
/// through `%p` are loads of two operands, and are not merged.
#[test]
fn loads_merge_across_casts_but_not_through_them() -> Result<(), Report> {
    for between in [
        "%c = mem.ptrcast %p : ptr<i8>;",
        "%a = mem.ptrtoint %p;",
        "%a = mem.ptrtoint %p;\n%c = mem.inttoptr %a : ptr<i64>;",
    ] {
        assert_eq!(loads_left_with_between(between)?, 1, "{between}");
    }
    let text = cse(r#"
    %c = mem.ptrcast %p : ptr;
    %1 = mem.load %p;
    %2 = mem.load %c : i32;
    builtin.ret %1, %2 : (i32, i32);"#)?;
    assert_eq!(text.matches("mem.load").count(), 2, "{text}");
    Ok(())
}

/// Under a data layout whose pointers are as wide as its `isize`: `ptrtoint (inttoptr %a)` is
/// `%a`; `inttoptr (ptrtoint %p)` is `%p` when the result has `%p`'s type and is left alone
/// otherwise; a `ptrcast` to its operand's type is the operand; a cast of a cast becomes one cast,
/// and then folds away when it casts back to the original type.
#[test]
fn casts_fold_and_canonicalize() -> Result<(), Report> {
    let text = canonicalize(
        Some("little, ptr = 32"),
        "%p: ptr<i32>, %a: isize, %r: ptr<i8, 1>",
        "isize, ptr<i32>, ptr<i8>, ptr<i32>, ptr<i16>, ptr<i32>, isize, ptr<i16, 1>",
        r#"    %1 = mem.inttoptr %a : ptr<i64, 2>;
    %2 = mem.ptrtoint %1;
    %3 = mem.ptrtoint %p;
    %4 = mem.inttoptr %3 : ptr<i32>;
    %5 = mem.ptrtoint %p;
    %6 = mem.inttoptr %5 : ptr<i8>;
    %7 = mem.ptrcast %p : ptr<i32>;
    %8 = mem.ptrcast %p : ptr;
    %9 = mem.ptrcast %8 : ptr<i16>;
    %10 = mem.ptrcast %p : ptr<i8>;
    %11 = mem.ptrcast %10 : ptr;
    %12 = mem.ptrcast %11 : ptr<i32>;
    %13 = mem.inttoptr %a : ptr<i32>;
    %14 = mem.ptrtoint %13;
    %15 = mem.ptrcast %r : ptr<1>;
    %16 = mem.ptrcast %15 : ptr<i16, 1>;
    builtin.ret %2, %4, %6, %7, %9, %12, %14, %16 : (isize, ptr<i32>, ptr<i8>, ptr<i32>, ptr<i16>, ptr<i32>, isize, ptr<i16, 1>);"#,
    )?;
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32>, %1: isize, %2: ptr<i8, 1>) -> (isize, ptr<i32>, ptr<i8>, ptr<i32>, ptr<i16>, ptr<i32>, isize, ptr<i16, 1>) {
            %7 = mem.ptrtoint %0;
            %8 = mem.inttoptr %7 : ptr<i8>;
            %11 = mem.ptrcast %0 : ptr<i16>;
            %18 = mem.ptrcast %2 : ptr<i16, 1>;
            builtin.ret %1, %0, %8, %0, %11, %0, %1, %18 : (isize, ptr<i32>, ptr<i8>, ptr<i32>, ptr<i16>, ptr<i32>, isize, ptr<i16, 1>);
        };"#]]
    .assert_eq(&text);
    Ok(())
}

const ROUND_TRIP_INT: &str = "    %q = mem.inttoptr %n : ptr<i8, 1>;
    %m = mem.ptrtoint %q;
    builtin.ret %m : (isize);";

const ROUND_TRIP_PTR: &str = "    %n = mem.ptrtoint %p;
    %q = mem.inttoptr %n : ptr<i8, 1>;
    builtin.ret %q : (ptr<i8, 1>);";

/// `ptrtoint (inttoptr x)` is `x` only when the pointer holds every bit of an `isize`: under
/// `ptr = 32, ptr<1> = 16` the pointer keeps 16 of the 32 bits, and the pair stays.
#[test]
fn ptrtoint_of_inttoptr_is_kept_when_the_pointer_is_narrower_than_isize() -> Result<(), Report> {
    let text =
        canonicalize(Some("little, ptr = 32, ptr<1> = 16"), "%n: isize", "isize", ROUND_TRIP_INT)?;
    assert!(text.contains("mem.inttoptr") && text.contains("mem.ptrtoint"), "{text}");
    // and it folds when the pointer is at least as wide as `isize`
    for layout in ["little, ptr = 32", "little, ptr = 32, ptr<1> = 64", "little, ptr = 64"] {
        let text = canonicalize(Some(layout), "%n: isize", "isize", ROUND_TRIP_INT)?;
        assert!(!text.contains("mem."), "{layout}\n{text}");
    }
    Ok(())
}

/// `inttoptr (ptrtoint p)` is `p` only when an `isize` holds every bit of the pointer: under
/// `ptr = 32, ptr<1> = 64`, and under `ptr = 64 index 32`, it holds 32 of the 64 bits.
#[test]
fn inttoptr_of_ptrtoint_is_kept_when_isize_is_narrower_than_the_pointer() -> Result<(), Report> {
    for layout in ["little, ptr = 32, ptr<1> = 64", "little, ptr = 64 align 8 index 32"] {
        let text = canonicalize(Some(layout), "%p: ptr<i8, 1>", "ptr<i8, 1>", ROUND_TRIP_PTR)?;
        assert!(
            text.contains("mem.inttoptr") && text.contains("mem.ptrtoint"),
            "{layout}\n{text}"
        );
    }
    // and it folds when `isize` is at least as wide as the pointer
    for layout in ["little, ptr = 32", "little, ptr = 32, ptr<1> = 16", "little, ptr = 64"] {
        let text = canonicalize(Some(layout), "%p: ptr<i8, 1>", "ptr<i8, 1>", ROUND_TRIP_PTR)?;
        assert!(!text.contains("mem."), "{layout}\n{text}");
    }
    Ok(())
}

/// With no data layout in scope nothing is known about the widths, and neither round trip folds.
#[test]
fn the_cast_round_trips_do_not_fold_without_a_data_layout() -> Result<(), Report> {
    let text = canonicalize(None, "%n: isize", "isize", ROUND_TRIP_INT)?;
    assert!(text.contains("mem.inttoptr") && text.contains("mem.ptrtoint"), "{text}");
    let text = canonicalize(None, "%p: ptr<i8, 1>", "ptr<i8, 1>", ROUND_TRIP_PTR)?;
    assert!(text.contains("mem.inttoptr") && text.contains("mem.ptrtoint"), "{text}");
    Ok(())
}

/// The folds that do not depend on a width, with no data layout in scope. A cast to the operand's
/// type is the operand; nested casts merge and then fold; `inttoptr (ptrtoint p)` to another
/// pointer type stays. A cast across address spaces, direct or as the outer of two, is rejected.
#[test]
fn the_width_free_cast_folds() -> Result<(), Report> {
    let text = canonicalize(
        None,
        "%p: ptr<i32, 1>, %o: ptr<1>",
        "ptr<i32, 1>, ptr<i32, 1>, ptr<i16, 1>, ptr<i8, 1>, ptr<i32, 1>",
        "    %a = mem.ptrcast %p : ptr<i32, 1>;
    %b = mem.ptrcast %p : ptr<i8, 1>;
    %c = mem.ptrcast %b : ptr<i32, 1>;
    %d = mem.ptrcast %b : ptr<i16, 1>;
    %n = mem.ptrtoint %p;
    %e = mem.inttoptr %n : ptr<i8, 1>;
    %f = mem.ptrcast %o : ptr<i32, 1>;
    builtin.ret %a, %c, %d, %e, %f : (ptr<i32, 1>, ptr<i32, 1>, ptr<i16, 1>, ptr<i8, 1>, ptr<i32, \
         1>);",
    )?;
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32, 1>, %1: ptr<1>) -> (ptr<i32, 1>, ptr<i32, 1>, ptr<i16, 1>, ptr<i8, 1>, ptr<i32, 1>) {
            %5 = mem.ptrcast %0 : ptr<i16, 1>;
            %6 = mem.ptrtoint %0;
            %7 = mem.inttoptr %6 : ptr<i8, 1>;
            %8 = mem.ptrcast %1 : ptr<i32, 1>;
            builtin.ret %0, %0, %5, %7, %8 : (ptr<i32, 1>, ptr<i32, 1>, ptr<i16, 1>, ptr<i8, 1>, ptr<i32, 1>);
        };"#]]
    .assert_eq(&text);
    for body in [
        "    %a = mem.ptrcast %p : ptr<i32>;",
        "    %b = mem.ptrcast %p : ptr<i8, 1>;\n    %a = mem.ptrcast %b : ptr<i32, 2>;",
    ] {
        let err = canonicalize(None, "%p: ptr<i32, 1>", "", &format!("{body}\n    builtin.ret;"))
            .unwrap_err();
        let err = runic_core::testing::text::describe(&err);
        assert!(err.contains("a cast keeps the address space"), "{err}");
    }
    Ok(())
}

/// The builder makes every bulk operation and cast, with the text the parser reads.
#[test]
fn the_builder_makes_bulk_and_cast_operations() -> Result<(), Report> {
    let mut test = Test::new("f", |cx| {
        ([cx.ptr_type(cx.i32()), cx.opaque_ptr_type(), cx.isize(), cx.i8()], [])
    });
    let context = test.context_rc();
    let [p, q, n, b] = {
        let entry = test.entry_block();
        let entry = entry.borrow();
        [0, 1, 2, 3].map(|i| entry.arguments()[i] as ValueRef)
    };
    {
        let mut builder = test.function_builder();
        builder.memcpy(p, q, n, SourceSpan::UNKNOWN)?;
        builder.memmove(q, p, n, SourceSpan::UNKNOWN)?;
        builder.memset(p, b, n, SourceSpan::UNKNOWN)?;
        let a = builder.ptrtoint(p, SourceSpan::UNKNOWN)?;
        assert_eq!(a.borrow().ty(), context.isize());
        let r =
            builder.inttoptr(a, context.ptr_type_in(Some(context.i8()), 1), SourceSpan::UNKNOWN)?;
        assert_eq!(r.borrow().ty(), context.ptr_type_in(Some(context.i8()), 1));
        let c = builder.ptrcast(q, context.ptr_type(context.i64()), SourceSpan::UNKNOWN)?;
        assert_eq!(c.borrow().ty(), context.ptr_type(context.i64()));
        builder.ret([], SourceSpan::UNKNOWN)?;
    }
    let function = test.function();
    function.borrow().as_operation().recursively_verify()?;
    expect![[r#"
        builtin.function public extern("C") @f(%0: ptr<i32>, %1: ptr, %2: isize, %3: i8) {
            mem.memcpy %0, %1, %2;
            mem.memmove %1, %0, %2;
            mem.memset %0, %3, %2;
            %4 = mem.ptrtoint %0;
            %5 = mem.inttoptr %4 : ptr<i8, 1>;
            %6 = mem.ptrcast %1 : ptr<i64>;
            builtin.ret;
        };"#]]
    .assert_eq(&format!("{}", function.borrow().as_operation()));
    let mut skipped = Vec::new();
    let failures = round_trip::<Function>(function.as_operation_ref(), true, &mut skipped);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
