use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::ops::{Deref, DerefMut};

use litcheck_filecheck::{filecheck, litcheck};
use pretty_assertions::assert_eq;

use crate::{
    BuilderExt, CallConv, Context, FunctionType, Immediate, OpParser, OpRegistration, OperationRef,
    Symbol, SymbolTable, Type, UnsafeIntrusiveEntityRef, ValueRef, Visibility,
    attributes::IntegerLikeAttr,
    diagnostics::{Report, SourceProvider, SourceSpan},
    dialects::{
        builtin::{
            BuiltinOpBuilder, Function, Module, Ret, RetImm, UnrealizedConversionCast, World,
            WorldRef,
            attributes::{AbiParam, Signature},
        },
        test::TestDialect,
    },
    parse::{self, ParseResult, ParserConfig},
    print::AsmPrinter,
    testing::Test,
    traits::Terminator,
};

type TestResult<T = ()> = Result<T, Report>;

#[crate::derive::operation(
    dialect = TestDialect,
    traits(Terminator)
)]
pub struct GroupedSuccessors {
    #[successors()]
    destinations: crate::Successor,
    #[successor]
    fallback: crate::Successor,
}

#[crate::derive::operation(dialect = TestDialect, traits(Terminator))]
pub struct VariadicSuccessors {
    #[successors()]
    destinations: crate::Successor,
}

#[crate::derive::operation(dialect = TestDialect, traits(Terminator))]
pub struct FixedSuccessors {
    #[successor]
    first: crate::Successor,
    #[successor]
    second: crate::Successor,
}

#[crate::derive::operation(dialect = TestDialect)]
pub struct GroupedOperands {
    #[operand]
    first: crate::traits::AnyType,
    #[operands]
    middle: crate::traits::Bool,
    #[operand]
    last: crate::traits::AnyType,
}

#[test]
fn declared_operand_cardinality_is_checked_before_building_and_verifying() -> TestResult {
    use crate::{GenericOperationBuilder, Op, OpVerifier};

    let mut test = Test::named("operand_groups");
    let i1 = test.context().i1();
    test.with_function("operand_groups", &[i1], &[]);
    let context = test.context_rc();
    context.get_or_register_dialect::<TestDialect>();
    let value = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
    let name = context.get_registered_name::<GroupedOperands>();

    // Fixed groups must have exactly one operand; the intervening variadic group may be empty.
    for (sizes, valid) in [
        (&[1, 0, 1][..], true),
        (&[1, 3, 1][..], true),
        (&[0, 0, 1][..], false),
        (&[2, 0, 1][..], false),
        (&[1, 0][..], false),
        (&[1, 0, 1, 1][..], false),
    ] {
        let mut builder = context.clone().builder();
        let op = name.alloc_default(context.clone());
        let mut op_builder = GenericOperationBuilder::new(&mut builder, op);
        for (group, count) in sizes.iter().copied().enumerate() {
            op_builder.with_operands_in_group(group, core::iter::repeat_n(value, count));
        }
        assert_eq!(op_builder.build().is_ok(), valid, "operand groups: {sizes:?}");
    }

    // Verification must catch corruption after construction, even for ops without trait verifiers.
    let mut builder = context.clone().builder();
    let op = name.alloc_default(context.clone());
    let mut op_builder = GenericOperationBuilder::new(&mut builder, op);
    op_builder.with_operands_in_group(0, [value]);
    op_builder.with_operands_in_group(1, []);
    op_builder.with_operands_in_group(2, [value]);
    let mut op = op_builder.build()?;
    op.borrow_mut().operands_mut().group_mut(0).clear();
    assert!(op.borrow().downcast_ref::<GroupedOperands>().unwrap().verify(&context).is_err());
    Ok(())
}

#[test]
fn variadic_operand_types_are_checked_when_building_and_verifying() -> TestResult {
    use crate::{GenericOperationBuilder, Op, OpVerifier};

    let mut test = Test::named("operand_types");
    let i1 = test.context().i1();
    let i32 = test.context().i32();
    test.with_function("operand_types", &[i1, i32], &[]);
    let context = test.context_rc();
    context.get_or_register_dialect::<TestDialect>();
    let (boolean, integer) = {
        let function = test.function();
        let function = function.borrow();
        let entry = function.entry_block();
        let entry = entry.borrow();
        (entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef)
    };
    let name = context.get_registered_name::<GroupedOperands>();

    let mut builder = context.clone().builder();
    let op = name.alloc_default(context.clone());
    let mut op_builder = GenericOperationBuilder::new(&mut builder, op);
    op_builder.with_operands_in_group(0, [boolean]);
    op_builder.with_operands_in_group(1, [integer]);
    op_builder.with_operands_in_group(2, [boolean]);
    assert!(op_builder.build().is_err(), "i32 must not satisfy the variadic Bool constraint");

    let op = name.alloc_default(context.clone());
    let mut op_builder = GenericOperationBuilder::new(&mut builder, op);
    op_builder.with_operands_in_group(0, [boolean]);
    op_builder.with_operands_in_group(1, []);
    op_builder.with_operands_in_group(2, [boolean]);
    let mut op = op_builder.build()?;
    let invalid_operand = context.make_operand(integer, op, 0);
    op.borrow_mut().operands_mut().push_to_group(1, invalid_operand);
    assert!(op.borrow().downcast_ref::<GroupedOperands>().unwrap().verify(&context).is_err());
    Ok(())
}

#[test]
fn generic_operand_cardinality_is_checked_before_type_inference() {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    // Shl infers its result type from lhs. Missing operands must be rejected before that
    // accessor runs, even when recursive verification is disabled.
    let source = r#"builtin.function public extern("C") @missing() -> i32 {
    %result = "test.shl"() : () -> i32;
    builtin.ret %result : (i32);
};"#;
    let error = test
        .parse_any_unverified("missing_inference_operand.hir", source)
        .expect_err("missing operands must not reach type inference");
    assert!(error.to_string().contains("invalid operand count"), "{error}");
}

#[test]
fn generic_parser_rejects_incorrect_provided_successor_counts() -> TestResult {
    use super::{
        DefaultParser, OperationState, ParserState, Scanner, TokenStream,
        operation::OperationParser,
    };

    for count in [0, 1, 3] {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<TestDialect>();
        let world = context.clone().builder().create::<World, ()>(SourceSpan::UNKNOWN)()?;
        let tokens = TokenStream::new(super::SourceId::UNKNOWN, Scanner::new(": () -> ()"));
        let parser =
            DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), tokens));
        let mut parser = OperationParser::new(parser, world);
        let mut state = OperationState::new(
            SourceSpan::UNKNOWN,
            context.get_registered_name::<FixedSuccessors>(),
        );
        let successors = alloc::vec![context.create_block(); count];
        let error = parser
            .parse_generic_operation_after_name(
                &mut state,
                Some(&[]),
                Some(&successors),
                None,
                None,
                None,
            )
            .expect_err("provided successors must satisfy the declared count");
        assert!(error.to_string().contains("invalid successor count"), "{error}");
    }
    Ok(())
}

#[test]
fn generic_parser_rejects_provided_successors_with_multiple_groups() -> TestResult {
    use super::{
        DefaultParser, OperationState, ParserState, Scanner, TokenStream,
        operation::OperationParser,
    };

    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<crate::dialects::test::TestDialect>();
    let world = context.clone().builder().create::<World, ()>(SourceSpan::UNKNOWN)()?;
    let tokens = TokenStream::new(super::SourceId::UNKNOWN, Scanner::new(": () -> ()"));
    let parser = DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), tokens));
    let mut parser = OperationParser::new(parser, world);
    let mut state = OperationState::new(
        SourceSpan::UNKNOWN,
        context.get_registered_name::<GroupedSuccessors>(),
    );
    let error = parser
        .parse_generic_operation_after_name(
            &mut state,
            Some(&[]),
            Some(&[context.create_block()]),
            None,
            None,
            None,
        )
        .expect_err("generic parsing must not flatten distinct successor groups");
    assert!(
        error
            .to_string()
            .contains("generic assembly cannot represent the successor layout"),
        "{error}"
    );
    Ok(())
}

#[test]
fn generic_parser_reserves_operand_groups_for_provided_successors() -> TestResult {
    use super::{
        DefaultParser, OperationState, ParserState, Scanner, TokenStream,
        operation::OperationParser,
    };

    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let world = context.clone().builder().create::<World, ()>(SourceSpan::UNKNOWN)()?;
    for (name, count) in [
        (context.get_registered_name::<VariadicSuccessors>(), 0),
        (context.get_registered_name::<VariadicSuccessors>(), 3),
        (context.get_registered_name::<FixedSuccessors>(), 2),
        (context.get_registered_name::<Ret>(), 0),
    ] {
        let tokens = TokenStream::new(super::SourceId::UNKNOWN, Scanner::new(": () -> ()"));
        let parser =
            DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), tokens));
        let mut parser = OperationParser::new(parser, world);
        let mut state = OperationState::new(SourceSpan::UNKNOWN, name);
        let successors = alloc::vec![context.create_block(); count];
        parser.parse_generic_operation_after_name(
            &mut state,
            Some(&[]),
            Some(&successors),
            None,
            None,
            None,
        )?;

        assert_eq!(state.operands.len(), count + 1);
        assert!(state.operands.iter().all(|group| group.is_empty()));
        assert_eq!(state.successors.len(), count);
        for (i, successor) in state.successors.iter().enumerate() {
            assert_eq!(successor.block, successors[i]);
            assert_eq!(successor.operand_group as usize, i + 1);
        }
    }
    Ok(())
}

#[test]
fn parse_simple_function() -> TestResult {
    let mut test = ParserTest::default();

    let source = "\
builtin.function public extern(\"C\") @entrypoint(%a: i32) -> i32 {
    ret %a : (i32);
};";

    let entrypoint = test.parse::<Function>("parse_simple_function.hir", source)?;
    let entrypoint = entrypoint.borrow();

    assert_eq!(entrypoint.name().as_str(), "entrypoint");
    assert_eq!(
        &*entrypoint.get_signature(),
        &Signature::new(&test.context_rc(), [test.context().i32()], [test.context().i32()])
    );
    assert_eq!(entrypoint.num_locals(), 0);
    assert_eq!(entrypoint.body().entry().body().len(), 1);

    Ok(())
}

#[test]
#[ignore]
fn parse_simple_function_generic() -> TestResult {
    let mut test = ParserTest::default();

    let source = r#""builtin.function"() <{
        name = @entrypoint,
        signature: #builtin.signature<"public extern(\"C\") (i32) -> i32">,
    }> ({
^entry(%a: i32):
    "builtin.ret" %a : (i32) -> ();
}) : () -> ();"#;

    let world = test.parse_generic("parse_simple_function_generic.hir", source)?;
    let entrypoint = world.borrow().body().entry().front().unwrap();
    let entrypoint = entrypoint.borrow();
    let entrypoint = entrypoint.downcast_ref::<Function>().expect("expected to parse a function");

    assert_eq!(entrypoint.name().as_str(), "entrypoint");
    assert_eq!(
        &*entrypoint.get_signature(),
        &Signature::new(&test.context_rc(), [test.context().i32()], [test.context().i32()])
    );
    assert_eq!(entrypoint.num_locals(), 0);
    assert_eq!(entrypoint.body().entry().body().len(), 1);

    Ok(())
}

#[test]
fn parse_module_with_intra_function_symbol_references() -> TestResult {
    let mut test = ParserTest::default();

    let source = "\
    builtin.module public @test {
        builtin.global_variable public @var : i32;

        builtin.function public extern(\"C\") @entrypoint(%a: i32) -> ptr<i8> {
            %ptr = builtin.global_symbol ::@test::@var+8 : ptr<i8>;
            builtin.ret %ptr : (ptr<i8>);
        };
    };";

    let parsed = test.parse_any("parse_module_with_intra_function_symbol_refs.hir", source)?;
    let parsed = parsed.borrow();
    let module = parsed.downcast_ref::<Module>().unwrap();

    assert_eq!(module.get_name().as_str(), "test");
    let symbol_manager = module.symbol_manager();
    assert_eq!(symbol_manager.symbols().symbols().count(), 2);
    let var = symbol_manager
        .lookup_op("var")
        .expect("'var' was not registered in symbol table after parsing");
    let entrypoint = symbol_manager
        .lookup_op("entrypoint")
        .expect("'entrypoint' was not registered in symbol table after parsing");
    let var = var.borrow();
    let var_uses = var.as_symbol().unwrap().iter_uses().count();
    assert_eq!(var_uses, 1);

    Ok(())
}

#[test]
fn module_symbol_attribute_resolves_from_enclosing_table() -> TestResult {
    use crate::dialects::builtin::attributes::SymbolRefAttr;

    let test = ParserTest::default();
    let source = "\
builtin.world {
    builtin.module public @reader attributes { reference = #builtin.symbol<@target> } {
        builtin.module public @target {};
    };
    builtin.module public @target {};
};";

    let parsed = test.parse_any("module_symbol_attribute.hir", source)?;
    let (reader, target) = {
        let world = parsed.borrow();
        let symbols = world.as_symbol_table().unwrap().symbol_manager();
        (symbols.lookup_op("reader").unwrap(), symbols.lookup_op("target").unwrap())
    };
    let (reference, inner_target) = {
        let reader = reader.borrow();
        (
            reader.get_typed_attribute::<SymbolRefAttr>("reference").unwrap(),
            reader.as_symbol_table().unwrap().symbol_manager().lookup_op("target").unwrap(),
        )
    };

    // The attribute uses the forward-declared sibling, even though the reader contains a
    // symbol with the same name. Resolution and the tracked use list must agree.
    let target = target.borrow();
    let mut uses = target.as_symbol().unwrap().iter_uses();
    let symbol_use = uses.next().expect("the sibling target must have a tracked use");
    assert!(uses.next().is_none(), "the sibling target must have exactly one use");
    assert_eq!(symbol_use.owner, reader);
    assert_eq!(symbol_use.attr, reference);
    assert_eq!(reference.borrow().user().borrow().owner, reader);
    assert_eq!(
        reference.borrow().resolve().unwrap().borrow().as_operation_ref(),
        target.as_operation_ref()
    );
    assert_eq!(inner_target.borrow().as_symbol().unwrap().iter_uses().count(), 0);

    Ok(())
}

#[test]
fn absolute_forward_symbol_uses_preserve_each_owner() -> TestResult {
    use crate::dialects::builtin::attributes::SymbolRefAttr;

    let source = r#"
builtin.module public @outer {
    builtin.module public @first attributes { reference = #builtin.symbol<::@outer::@target> } {};
    builtin.module public @second attributes { reference = #builtin.symbol<::@outer::@target> } {};
    builtin.module public @target {};
};
"#;
    for mode in ["module", "world"] {
        let test = ParserTest::default();
        let parsed = match mode {
            "module" => test.parse_any("forward_uses.hir", source)?,
            "world" => {
                test.parse_any("forward_uses.hir", &format!("builtin.world {{ {source} }};"))?
            }
            _ => unreachable!(),
        };
        let module = if parsed.borrow().is::<Module>() {
            parsed
        } else {
            parsed
                .borrow()
                .as_symbol_table()
                .unwrap()
                .symbol_manager()
                .lookup_op("outer")
                .unwrap()
        };
        let module = module.borrow();
        let symbols = module.as_symbol_table().unwrap().symbol_manager();
        let target = symbols.lookup_op("target").unwrap();
        let target = target.borrow();
        let uses = target.as_symbol().unwrap().iter_uses().collect::<alloc::vec::Vec<_>>();
        assert_eq!(uses.len(), 2, "{mode}");
        for name in ["first", "second"] {
            let owner = symbols.lookup_op(name).unwrap();
            let reference =
                owner.borrow().get_typed_attribute::<SymbolRefAttr>("reference").unwrap();
            assert_eq!(
                uses.iter()
                    .filter(|usage| usage.owner == owner && usage.attr == reference)
                    .count(),
                1
            );
            let reference = reference.borrow();
            assert_eq!(reference.user().borrow().owner, owner);
            assert_eq!(
                reference.resolve().unwrap().borrow().as_operation_ref(),
                target.as_operation_ref()
            );
        }
    }
    Ok(())
}

/// The `reserved_memory` module attribute is an input to the linker's memory layout, so it must
/// survive a print/parse round-trip of the textual HIR.
#[test]
fn module_reserved_memory_roundtrip() -> TestResult {
    use crate::{Op, dialects::builtin::attributes::U64Attr};

    let mut test = ParserTest::default();

    let source = "\
    builtin.module public @test {
        builtin.function internal extern(\"C\") @callee(%a: i32, %b: i32) -> i32 {
            builtin.ret %a : (i32);
        };
    };";

    let parsed = test.parse_any("reserved_memory_roundtrip.hir", source)?;

    // Attach the linker-facing memory reservation, then round-trip through print/parse and
    // verify nothing is lost
    {
        let attr = test.context_rc().create_attribute::<U64Attr, _>(0x110000u64);
        let mut op = parsed;
        let mut op = op.borrow_mut();
        op.set_attribute(Module::RESERVED_MEMORY_ATTR, attr);
    }

    let flags = Default::default();
    let mut printer = AsmPrinter::new(test.context_rc(), &flags);
    printer.print_operation(parsed.borrow());
    let printed = printer.finish().to_string();
    std::println!("{printed}");

    let reparsed = test.parse_any("reserved_memory_roundtrip2.hir", &printed)?;
    let reparsed = reparsed.borrow();
    let module = reparsed.downcast_ref::<Module>().unwrap();

    let reserved = module
        .as_operation()
        .get_typed_attribute::<U64Attr>(Module::RESERVED_MEMORY_ATTR)
        .expect("reserved_memory attribute lost in print/parse round-trip");
    assert_eq!(**reserved.borrow(), 0x110000);

    Ok(())
}

#[test]
fn derive_roundtrip_test() -> TestResult {
    let mut test = Test::named("derive_roundtrip_test");
    let (i32, i64) = (test.context().i32(), test.context().i64());
    test.with_function("derive_roundtrip_test", &[i32], &[i64]);
    let mut test = ParserTest { test };

    {
        let mut f = test.function_builder();
        let v0 = f.entry_block().borrow().arguments()[0] as ValueRef;
        let v1 = f.builder_mut().unrealized_conversion_cast(v0, i64, SourceSpan::UNKNOWN)?;
        f.builder_mut().ret([v1], SourceSpan::UNKNOWN);
    }

    let flags = Default::default();
    let mut printer = AsmPrinter::new(test.context_rc(), &flags);
    printer.print_operation(test.function().borrow());
    let source = printer.render().to_string();

    let parsed = test.parse::<Function>("derive_roundtrip.hir", &source)?;
    let parsed = parsed.borrow();

    printer.print_operation(&parsed);
    let roundtripped = printer.finish().to_string();

    std::println!("{source}");
    std::println!("{roundtripped}");
    //assert_eq!(&source, &roundtripped);
    filecheck!(
        &roundtripped,
        r#"
    // CHECK: builtin.function public extern("C") @derive_roundtrip_test([[V0:%\d+]]: i32) -> i64 {
    // CHECK-NEXT: [[V1:%\d+]] = builtin.unrealized_conversion_cast [[V0]] <{ ty = #builtin.type<i64> }>;
    // CHECK-NEXT: builtin.ret [[V1]] : (i64);
    // CHECK-NEXT: };
    "#
    );

    Ok(())
}

#[test]
fn parse_ret_imm_coerces_literal_to_declared_type() -> TestResult {
    let test = ParserTest::default();

    let source = "\
builtin.function public extern(\"C\") @retconst() -> i32 {
    builtin.ret_imm 42 : i32;
};";

    let function = test.parse::<Function>("parse_ret_imm.hir", source)?;
    let printed = format!("{}", function.as_operation_ref().borrow());
    assert!(
        printed.contains("builtin.ret_imm 42 : i32"),
        "expected the declared type to survive the round trip, got:\n{printed}"
    );

    let function = function.borrow();
    let ret_imm = function
        .body()
        .entry()
        .terminator()
        .unwrap()
        .try_downcast_op::<RetImm>()
        .expect("expected the function terminator to be builtin.ret_imm");
    let imm = ret_imm.borrow().value().as_ref().as_immediate();
    assert!(
        matches!(imm, Immediate::I32(42)),
        "expected the literal to be coerced to the declared type, got {imm:?}"
    );

    // A literal that cannot be represented in the declared type must be rejected.
    let result = test.parse::<Function>(
        "parse_ret_imm_invalid.hir",
        "\
builtin.function public extern(\"C\") @retconst() -> i8 {
    builtin.ret_imm 300 : i8;
};",
    );
    assert!(result.is_err(), "expected an out-of-range immediate to be rejected");

    Ok(())
}

/// A whole `builtin.world` — the shape `runic --emit=hir` writes, and the shape that must
/// survive being fed back in.
const WORLD_SOURCE: &str = "\
builtin.world {
    builtin.module public @lib {
        builtin.function public extern(\"C\") @main() {
            builtin.ret;
        };
    };
};";

/// The same content without the enclosing world, which the parser does have to wrap.
const MODULE_SOURCE: &str = "\
builtin.module public @lib {
    builtin.function public extern(\"C\") @main() {
        builtin.ret;
    };
};";

#[test]
fn parsing_a_world_yields_that_world_rather_than_nesting_it() -> TestResult {
    let test = ParserTest::default();

    let parsed = test.parse_any("parse_world.hir", WORLD_SOURCE)?;
    let world = parsed
        .try_downcast_op::<World>()
        .expect("expected the parsed operation to be a 'builtin.world'");

    // The world the file declares *is* the root. Nothing was created around it.
    assert!(
        parsed.parent_op().is_none(),
        "a parsed 'builtin.world' must not be nested inside another operation, but it hangs from \
         a '{}'",
        parsed.parent_op().unwrap().borrow().name()
    );

    // And because it is the root, symbols inside it have well-formed paths. A world is an
    // anonymous symbol table, and `Symbol::path` asserts that an anonymous symbol table has no
    // parent — so a world nested in a world does not merely produce a wrong path here, it
    // panics.
    let module_op = world.borrow().body().entry().front().expect("the world declares a module");
    {
        let module = module_op.borrow();
        let module = module.downcast_ref::<Module>().expect("expected 'lib' to be a module");
        let path = module.path();
        assert!(path.is_absolute(), "a symbol path is rooted at the world");
        assert_eq!(path.to_string(), "::lib");
    }

    // The function's path crosses two symbol tables, so it also pins that the world contributes
    // no component of its own — it is the root, not a named ancestor.
    let function = {
        let module = module_op.borrow();
        let module = module.downcast_ref::<Module>().unwrap();
        module.body().entry().front().expect("the module defines a function")
    };
    let function = function.borrow();
    let function = function.downcast_ref::<Function>().expect("expected 'main' to be a function");
    assert_eq!(function.path().to_string(), "::lib::main");

    Ok(())
}

#[test]
fn parsing_a_module_still_wraps_it_in_a_world() -> TestResult {
    let test = ParserTest::default();

    let parsed = test.parse_any("parse_module_wrapped.hir", MODULE_SOURCE)?;
    let module = parsed
        .try_downcast_op::<Module>()
        .expect("expected the parsed operation to be a 'builtin.module'");

    // The discriminating half: an operation that is not a world still gets the world it needs to
    // resolve symbols against.
    let parent = parsed
        .parent_op()
        .expect("a top-level operation that is not a world must be given a world to live in");
    assert!(
        parent.borrow().is::<World>(),
        "expected the wrapper to be a 'builtin.world', got a '{}'",
        parent.borrow().name()
    );
    assert!(parent.parent_op().is_none(), "the wrapper world must itself be the root");

    let path = module.borrow().path();
    assert!(path.is_absolute(), "a symbol path is rooted at the wrapper world");
    assert_eq!(path.to_string(), "::lib");

    Ok(())
}

/// Well-formed as text, ill-formed as IR: `builtin.function` carries the `SingleRegion` trait, so
/// a function with no region at all parses (it is how a declaration is written) and then fails
/// verification with "requires exactly one region, but got 0".
const UNVERIFIABLE_SOURCE: &str = "\
builtin.module public @lib {
    builtin.function public extern(\"C\") @main();
};";

#[test]
fn parsing_verifies_what_it_parsed() {
    let test = ParserTest::default();

    // Verification used to run inside `OperationParser::finalize`, and moving it out to
    // `parse_anchored_source` left nothing pinning that it still runs at all: the tests that
    // exercise invalid IR reach the verifier by calling `recursively_verify` themselves, and the
    // one fixture that is deliberately malformed is parsed with verification turned off. Deleting
    // the call would have kept every one of them green. This is the test that would not be.
    let result = test.parse_any("parse_unverifiable.hir", UNVERIFIABLE_SOURCE);

    let err = result.err().map(|err| err.to_string()).unwrap_or_else(|| {
        panic!(
            "the default parser configuration verifies, so ill-formed IR must not parse \
             successfully"
        )
    });

    // And the failure has to be the *verifier's*: the same source parses when verification is
    // turned off, so nothing about the text itself is what rejected it.
    test.parse_any_unverified("parse_unverifiable_unverified.hir", UNVERIFIABLE_SOURCE)
        .expect("the source is well-formed as text — only verification rejects it");

    assert!(
        err.contains("invalid operation builtin.function"),
        "expected the verifier's rejection of the region-less function, got: {err}"
    );
}

#[test]
fn parse_preserves_full_width_frame_base_local_indices() -> TestResult {
    use crate::dialects::debuginfo::DebugInfoDialect;

    let mut test = ParserTest::default();
    test.context().get_or_register_dialect::<DebugInfoDialect>();
    let source = r#"builtin.function public extern("C") @frame_base(%0: i32) -> i32 {
    di.debug_declare <{ variable = #di.variable<{ name = "x", file = "test.rs", line = 1 }>, expression = #di.expression<[DI_OP_frame_base(local_slot, 2147483648+0)]> }>;
    builtin.ret %0 : (i32);
};"#;

    let function = test.parse::<Function>("frame_base_full_width_index.hir", source)?;
    let printed = format!("{}", function.as_operation_ref().borrow());
    assert!(
        printed.contains("DI_OP_frame_base(local_slot, 2147483648+0)"),
        "frame-base index was not preserved: {printed}"
    );

    Ok(())
}

#[derive(Default)]
struct ParserTest {
    test: Test,
}

impl Deref for ParserTest {
    type Target = Test;

    fn deref(&self) -> &Self::Target {
        &self.test
    }
}

impl DerefMut for ParserTest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.test
    }
}

impl ParserTest {
    #[allow(unused)]
    pub fn parse_generic(&self, name: &str, source: &str) -> TestResult<WorldRef> {
        let config = ParserConfig::new(self.test.context_rc());
        parse::parse_generic(config, name, source).into_result()
    }

    pub fn parse<T: OpParser + OpRegistration>(
        &self,
        name: &str,
        source: &str,
    ) -> TestResult<UnsafeIntrusiveEntityRef<T>> {
        let config = ParserConfig::new(self.test.context_rc());
        parse::parse::<T>(config, name, source).into_result()
    }

    pub fn parse_any(&self, name: &str, source: &str) -> TestResult<OperationRef> {
        let config = ParserConfig::new(self.test.context_rc());
        parse::parse_any(config, name, source).into_result()
    }

    pub fn parse_any_unverified(&self, name: &str, source: &str) -> TestResult<OperationRef> {
        let config = ParserConfig::new(self.test.context_rc()).verify_after_parse(false);
        parse::parse_any(config, name, source).into_result()
    }
}

/// An unregistered dialect is a different failure from an unregistered operation inside a dialect
/// that exists, and the diagnostic has to say which one it is.
#[test]
fn an_unregistered_dialect_is_named_in_the_diagnostic() {
    let test = ParserTest::default();
    let Err(err) =
        test.parse_any("unknown_dialect.hir", "builtin.module public @t { nosuchdialect.op; };")
    else {
        panic!("an unregistered dialect must not parse");
    };
    let rendered = alloc::string::ToString::to_string(&err);
    assert!(
        rendered.contains("unknown dialect") && rendered.contains("nosuchdialect"),
        "the diagnostic must name the unregistered dialect, got: {rendered}"
    );
}

/// The control: an unknown operation inside a dialect that is registered still reports as an
/// invalid operation, not as an unknown dialect.
#[test]
fn an_unknown_operation_in_a_known_dialect_still_reports_separately() {
    let test = ParserTest::default();
    let Err(err) = test.parse_any("unknown_op.hir", "builtin.module public @t { builtin.nope; };")
    else {
        panic!("an unregistered operation must not parse");
    };
    let rendered = alloc::string::ToString::to_string(&err);
    assert!(
        !rendered.contains("unknown dialect"),
        "a known dialect must not be reported as unknown, got: {rendered}"
    );
}

/// A module with an empty body defines no symbols; building its symbol table must not unwrap the
/// missing entry block.
#[test]
fn an_empty_module_body_parses() {
    let test = ParserTest::default();
    test.parse_any("empty.hir", "builtin.module public @t {};")
        .expect("an empty module body should parse");
}

/// Two symbols with one name is something the author wrote, so it must be reported rather than
/// tripping the uniqueness assertion inside the symbol table.
#[test]
fn a_duplicate_symbol_is_reported() {
    let test = ParserTest::default();
    let source = "\
builtin.module public @t {
    builtin.function internal extern(\"C\") @f(%a: i32) -> i32 {
        builtin.ret %a : (i32);
    };
    builtin.function internal extern(\"C\") @f(%b: i32) -> i32 {
        builtin.ret %b : (i32);
    };
};";
    assert!(test.parse_any("dup.hir", source).is_err(), "a duplicate symbol must not parse");
}

/// A parsed world is held to the rule `WorldBuilder` enforces: a component name must not shadow
/// a module tree.
#[test]
fn a_component_shadowing_a_module_tree_is_reported() {
    let test = ParserTest::default();
    let source = "\
builtin.world {
    builtin.module public @acme {
        builtin.module public @a {
        };
    };
    builtin.component private @acme::@a {
    };
};";
    let err = test
        .parse_any("shadowing.hir", source)
        .expect_err("a component shadowing a module tree must not parse")
        .to_string();
    assert!(
        err.contains(
            "component `acme::a` and `builtin.module` `::acme::a` have the same path; a component \
             must not share its path with another symbol"
        ),
        "unexpected diagnostic: {err}"
    );
}

/// A parsed module is held to the rule the builders enforce: a module name has no `::`.
#[test]
fn a_module_named_by_a_path_is_reported() {
    let test = ParserTest::default();
    let source = "\
builtin.world {
    builtin.module public @\"a::b\" {
    };
};";
    let err = test
        .parse_any("module_path.hir", source)
        .expect_err("a module named by a path must not parse")
        .to_string();
    assert!(
        err.contains(
            "module `a::b`: a module name cannot contain `::` (only components are named by \
             `::`-joined paths)"
        ),
        "unexpected diagnostic: {err}"
    );
}

#[test]
fn nested_component_names_are_reported() {
    let test = ParserTest::default();
    let source = "\
builtin.world {
    builtin.component private @acme::@app {
    };
    builtin.component private @acme::@app::@main {
    };
};";
    let err = test
        .parse_any("nesting.hir", source)
        .expect_err("components with nesting names must not parse")
        .to_string();
    assert!(
        err.contains(
            "component `acme::app` and component `acme::app::main` nest (`acme::app` is a \
             segment-prefix of `acme::app::main`); component namespaces must not nest"
        ),
        "unexpected diagnostic: {err}"
    );
}

/// A component prints its name as a symbol path, one raw segment per segment of the name (see
/// `SymbolPath::to_symbol_name`), quoting a segment that is not a bare HIR identifier, and that
/// text parses back to the same name: the canonical spelling of the segments.
#[test]
fn component_name_roundtrips_through_the_printer() -> TestResult {
    use crate::{
        Op,
        dialects::builtin::{Component, WorldBuilder},
    };

    let test = ParserTest::default();
    for (raw, name, header) in [
        ("acme::a::b", "acme::a::b", "builtin.component private @acme::@a::@b {"),
        (
            "hir_ns:test@1.0.0",
            "\"hir_ns:test@1.0.0\"",
            "builtin.component private @\"hir_ns:test@1.0.0\" {",
        ),
        ("$kernel", "\"$kernel\"", "builtin.component private @\"$kernel\" {"),
        ("1st", "\"1st\"", "builtin.component private @\"1st\" {"),
        ("a-b", "\"a-b\"", "builtin.component private @\"a-b\" {"),
    ] {
        let world = test.context_rc().builder().create::<World, ()>(SourceSpan::UNKNOWN)()?;
        let component = WorldBuilder::new(world).define_component(raw.into())?;
        assert_eq!(Symbol::name(&*component.borrow()).as_str(), name);

        let flags = Default::default();
        let mut printer = AsmPrinter::new(test.context_rc(), &flags);
        printer.print_operation(component.as_operation_ref().borrow());
        let printed = printer.finish().to_string();
        assert!(printed.starts_with(header), "unexpected printed form:\n{printed}");

        let reparsed = test.parse::<Component>("component_roundtrip.hir", &printed)?;
        assert_eq!(Symbol::name(&*reparsed.borrow()).as_str(), name);
    }

    Ok(())
}

/// A component named by a path whose segment contains `::` prints that segment quoted, and the
/// text parses back to a component with the same segments.
#[test]
fn component_name_with_a_quoted_segment_roundtrips_through_the_printer() -> TestResult {
    use crate::{
        Op, SymbolPath,
        dialects::builtin::{Component, WorldBuilder},
    };

    let test = ParserTest::default();
    let path = SymbolPath::from_module_name("::\"acme::pkg\"::app");
    let world = test.context_rc().builder().create::<World, ()>(SourceSpan::UNKNOWN)()?;
    let component = WorldBuilder::new(world).define_component(path.to_symbol_name()?.into())?;

    let flags = Default::default();
    let mut printer = AsmPrinter::new(test.context_rc(), &flags);
    printer.print_operation(component.as_operation_ref().borrow());
    let printed = printer.finish().to_string();
    assert!(
        printed.starts_with(r#"builtin.component private @"acme::pkg"::@app {"#),
        "unexpected printed form:\n{printed}"
    );

    let reparsed = test.parse::<Component>("component_roundtrip.hir", &printed)?;
    assert_eq!(Symbol::name(&*reparsed.borrow()).as_str(), r#""acme::pkg"::app"#);
    let namespace = reparsed.borrow().namespace_path();
    assert_eq!(namespace, path);
    assert_eq!(namespace.components().len(), 3, "the root and two segments: {namespace:?}");

    Ok(())
}

/// A symbol-table op whose raw segment the spelling rules reject does not parse, a raw segment
/// wrapped in quotes included: it is not read as the segment it quotes.
#[test]
fn an_unspellable_symbol_table_name_is_reported() {
    let test = ParserTest::default();
    for (source, kind, segment) in [
        ("builtin.component private @\"q\\\"u\" {};", "component", "q\"u"),
        ("builtin.component private @\"\\\"x-y\\\"\" {};", "component", "\"x-y\""),
        ("builtin.component private @app::@\"a b\" {};", "component", "a b"),
        ("builtin.module public @\"a\\\\b\" {};", "module", "a\\b"),
        ("builtin.module public @\"\\\"m\\\"\" {};", "module", "\"m\""),
        ("builtin.module public @\"a#b\" {};", "module", "a#b"),
    ] {
        let err = test
            .parse_any("unspellable.hir", source)
            .expect_err("an unspellable name must not parse")
            .to_string();
        assert_eq!(
            err,
            format!("invalid {kind} name: the segment `{segment}` is not a valid path segment"),
            "{source}"
        );
    }
}

/// The diagnostic for a component name underlines the whole symbol path: from its root `::`, or
/// from its first identifier (after the `@` sigil) when it has no root, to the end of its last
/// identifier.
#[test]
fn a_symbol_path_diagnostic_spans_the_whole_path() {
    let test = ParserTest::default();
    // The path, and where its span starts
    for (path, first) in [("@app::@\"a b\"", "app"), ("::@app::@\"a b\"", "::")] {
        let source = format!("builtin.component private {path} {{}};");
        let err = test
            .parse_any("span.hir", &source)
            .expect_err("an unspellable name must not parse");
        let labels = crate::testing::label_ranges(err.as_diagnostic())
            .into_iter()
            .map(|range| (range.start, range.end))
            .collect::<alloc::vec::Vec<_>>();
        let path_start = source.find(path).unwrap();
        let start = path_start + path.find(first).unwrap();
        assert_eq!(labels, [(start, path_start + path.len())], "{source}");
    }
}

/// A lone leaf naming a symbol table is its raw last segment, which `Symbol::path` gives the
/// table and which resolves to it, from its parent and as a standalone root.
#[test]
fn a_lone_leaf_naming_a_symbol_table_resolves() -> TestResult {
    use crate::{
        Op, SymbolName, SymbolNameComponent, SymbolPath,
        dialects::builtin::{Component, WorldBuilder},
    };

    let test = ParserTest::default();
    let leaf = |segments: &[&str]| {
        let (last, rest) = segments.split_last().unwrap();
        SymbolPath::from_iter(
            core::iter::once(SymbolNameComponent::Root)
                .chain(rest.iter().map(|s| SymbolNameComponent::Component(SymbolName::intern(s))))
                .chain([SymbolNameComponent::Leaf(SymbolName::intern(last))]),
        )
    };

    // A module whose name needs quoting, in a world
    let source = "\
builtin.world {
    builtin.module public @\"my-mod\" {
    };
};";
    let world = test.parse_any("leaf_module.hir", source)?;
    let world = world.try_downcast_op::<World>().expect("expected a world");
    let module = world.borrow().resolve(&leaf(&["my-mod"])).expect("`::\"my-mod\"` resolves");
    assert!(module.borrow().as_symbol_operation().is::<Module>());
    let path = module.borrow().path();
    assert_eq!(path, leaf(&["my-mod"]));
    let flags = Default::default();
    let mut printer = AsmPrinter::new(test.context_rc(), &flags);
    printer.print_symbol_path(&path);
    assert_eq!(printer.finish().to_string(), "::@\"my-mod\"");

    // A component named by several segments, in a world
    let world = test.context_rc().builder().create::<World, ()>(SourceSpan::UNKNOWN)()?;
    let component = WorldBuilder::new(world).define_component("acme::a::b".into())?;
    let path = component.borrow().path();
    assert_eq!(path, leaf(&["acme", "a", "b"]));
    let resolved = world.borrow().resolve(&path).expect("`::acme::a::b` resolves");
    assert_eq!(
        resolved.borrow().as_symbol_operation().as_operation_ref(),
        component.as_operation_ref()
    );

    // A standalone component, the root itself, named by its raw segment only
    let source = "builtin.component private @\"hir_ns:test@1.0.0\" {\n};";
    let component = test.parse::<Component>("leaf_root.hir", source)?;
    let root = component.as_operation_ref();
    let resolve = |path: &SymbolPath| {
        component
            .borrow()
            .resolve(path)
            .map(|symbol| symbol.borrow().as_symbol_operation().as_operation_ref())
    };
    assert_eq!(resolve(&leaf(&["hir_ns:test@1.0.0"])), Some(root));
    assert_eq!(resolve(&leaf(&["\"hir_ns:test@1.0.0\""])), None);
    assert_eq!(resolve(&component.borrow().path()), Some(root));
    Ok(())
}

/// A component whose path is the path of a function in a module tree is rejected, while a
/// component at another path is accepted: paths identify symbols uniquely.
#[test]
fn a_component_at_the_path_of_a_module_tree_function_is_reported() -> TestResult {
    let source = |component: &str| {
        format!(
            "\
builtin.world {{
    builtin.module public @acme {{
        builtin.module public @a {{
            builtin.function public extern(\"C\") @f() {{
                builtin.ret;
            }};
        }};
    }};
    builtin.component private {component} {{
    }};
}};"
        )
    };
    let test = ParserTest::default();
    let err = test
        .parse_any("shadow.hir", &source("@acme::@a::@f"))
        .expect_err("the component shadows the function `f`")
        .to_string();
    assert_eq!(
        err,
        "component `acme::a::f` and `builtin.function` `::acme::a::f` have the same path; a \
         component must not share its path with another symbol"
    );
    test.parse_any("shadow_ctrl.hir", &source("@other"))?;
    Ok(())
}

/// A module and a function whose raw names are equal have the same path, so they cannot both be
/// defined in one symbol table, whatever their definition order.
#[test]
fn a_module_and_a_function_with_one_raw_name_are_reported() {
    let module = "\
        builtin.module private @\"my-mod\" {
            builtin.function public extern(\"C\") @g() {
                builtin.ret;
            };
        };";
    let function = "\
        builtin.function public extern(\"C\") @\"my-mod\"() {
            builtin.ret;
        };";
    let test = ParserTest::default();
    let (module_kind, function_kind) = ("module \"my-mod\"", "function my-mod");
    for (first, second, expected) in [
        (module, function, (function_kind, module_kind)),
        (function, module, (module_kind, function_kind)),
    ] {
        let source = format!(
            "builtin.world {{\n    builtin.component private @app {{\n{first}\n{second}\n    \
             }};\n}};"
        );
        let err = test
            .parse_any("crosskind.hir", &source)
            .expect_err("the module and the function share the path `::app::my-mod`")
            .to_string();
        let (new, existing) = expected;
        assert_eq!(
            err,
            format!("`{new}` and `{existing}` have the same path `::app::\"my-mod\"`"),
            "{source}"
        );
    }
}

/// A reference spells only raw segments: a segment wrapped in quotes is not read as the segment
/// it quotes.
#[test]
fn a_quoted_raw_segment_in_a_reference_does_not_resolve() -> TestResult {
    use crate::{SymbolName, SymbolNameComponent, SymbolPath};

    let source = "\
builtin.world {
    builtin.component private @app {
        builtin.module private @\"my-mod\" {
            builtin.global_variable public @v : i32;
        };
    };
};";
    let test = ParserTest::default();
    let world = test.parse_any("modq.hir", source)?;
    let world = world.try_downcast_op::<World>().expect("expected a world");
    let resolves = |module: &str| {
        let path = SymbolPath::from_iter([
            SymbolNameComponent::Root,
            SymbolNameComponent::Component("app".into()),
            SymbolNameComponent::Component(SymbolName::intern(module)),
            SymbolNameComponent::Leaf("v".into()),
        ]);
        world.borrow().resolve(&path).is_some()
    };
    assert!(resolves("my-mod"));
    assert!(!resolves("\"my-mod\""));
    Ok(())
}

/// A generic-form operation with more successors than the 16-bit operand-group index can address
/// is rejected by the parser instead of wrapping a successor onto an earlier operand group.
#[test]
fn parse_generic_operation_with_too_many_successors_is_rejected() {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    let name = test.context().get_registered_name::<VariadicSuccessors>();
    // The op's own operands take group 0, so the successors fill the remaining 65,535 groups
    // and the last one needs group 65,536
    let successors = alloc::vec!["^dest"; u16::MAX as usize + 1].join(", ");
    let source = format!(
        "\
builtin.function public extern(\"C\") @entrypoint(%a: i32) -> i32 {{
    \"{name}\"()[{successors}] : () -> ();
^dest:
    builtin.ret %a : (i32);
}};"
    );

    let error = test
        .parse::<Function>("parse_generic_too_many_successors.hir", &source)
        .err()
        .expect("expected the operation to be rejected");
    let message = format!("{error}");
    assert!(message.contains("too many operand groups"), "{message}");
}

#[test]
fn array_types_parse_print_and_report_invalid_lengths() {
    use crate::Op;
    type Expected = fn(&Context) -> Type;
    let cases: [(&str, Expected); 3] = [
        ("[i32; 4]", |context| context.array_type(context.i32(), 4)),
        ("array<i32; 4>", |context| context.array_type(context.i32(), 4)),
        ("ptr<[[i32; 4]; 2]>", |context| {
            context.ptr_type(context.array_type(context.array_type(context.i32(), 4), 2))
        }),
    ];
    for (source, expected) in cases {
        let context = Rc::new(Context::default());
        let expected = expected(&context);
        let text = format!(
            "builtin.function private extern(\"C\") @array(%x: {source}) {{ builtin.ret; }};"
        );
        let parsed =
            parse::parse_any(ParserConfig::new(context.clone()), "array.hir", &text).unwrap();
        let function = parsed.try_downcast_op::<Function>().unwrap();
        assert_eq!(function.borrow().signature().params()[0].ty, expected);
        let printed = format!("{}", function.borrow().as_operation());
        let fresh = Rc::new(Context::default());
        parse::parse_any(ParserConfig::new(fresh.clone()), "array-roundtrip.hir", &printed)
            .unwrap();
    }
    for source in ["[i32; -1]", "[i32; 18446744073709551616]", "[i32 4]", "[i32; 4>"] {
        let context = Rc::new(Context::default());
        let text = format!(
            "builtin.function private extern(\"C\") @array(%x: {source}) {{ builtin.ret; }};"
        );
        assert!(
            parse::parse_any(ParserConfig::new(context.clone()), "invalid-array.hir", &text)
                .is_err()
        );
    }
}

#[test]
fn a_clean_parse_is_ok_with_no_diagnostics() {
    let context = Rc::new(Context::default());
    let outcome =
        parse::parse_any(ParserConfig::new(context), "clean.runic", "builtin.module public @t {};");
    assert!(outcome.is_ok());
    assert!(outcome.diagnostics.is_empty());
}

#[test]
fn a_syntax_error_is_one_error_diagnostic_with_sources_attached() {
    let context = Rc::new(Context::default());
    let outcome =
        parse::parse_any(ParserConfig::new(context), "bad.runic", "builtin.module public @m {");
    assert!(outcome.result.is_err());
    assert_eq!(outcome.diagnostics.len(), 1);
    let report = outcome.into_result().unwrap_err();
    let rendered = alloc::format!("{}", report.display());
    assert!(rendered.contains("bad.runic"), "rendered without its source: {rendered}");
}

/// Resolving `loc("file":l:c)` loads `file` into the context while parsing, so the sources
/// attached to the outcome must be taken after parsing, or a diagnostic labelled with that span
/// renders without its source.
#[cfg(feature = "std")]
#[test]
fn an_outcome_renders_sources_loaded_while_parsing() {
    let context = Rc::new(Context::default());
    let origin = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    assert!(context.sources().find_by_name(origin).is_none());
    let source = format!(
        "builtin.module public @lib {{\n    builtin.function public extern(\"C\") @main() \
         loc(\"{origin}\":1:1);\n}};"
    );
    let outcome = parse::parse_any(ParserConfig::new(context.clone()), "origin.runic", source);
    assert!(context.sources().find_by_name(origin).is_some(), "the location loaded its file");

    let report = outcome.into_result().expect_err("a region-less function fails verification");
    let rendered = alloc::format!("{}", report.display());
    assert!(!rendered.contains("missing source"), "rendered without its source: {rendered}");
    assert!(
        rendered.contains("[package]"),
        "rendered without the located snippet: {rendered}"
    );
}

/// Printed source locations follow the documented `file-line-col-range` grammar, so printing IR
/// with its locations and parsing the result back yields the same locations.
#[test]
fn printed_locations_parse_back() {
    use crate::{dialects::builtin::attributes::Location, print::OpPrintingFlags};

    let context = Rc::new(Context::default());
    let location_of = |op: OperationRef| Location::from_span(op.borrow().span, &context);
    let print_with_locations = |op: OperationRef| {
        let flags = OpPrintingFlags {
            print_source_locations: true,
            ..Default::default()
        };
        let mut printer = AsmPrinter::new(context.clone(), &flags);
        printer.print_operation(op.borrow());
        printer.finish().to_string()
    };

    let original = parse::parse_any(
        ParserConfig::new(context.clone()),
        "in.runic",
        "builtin.module public @t {};",
    )
    .into_result()
    .expect("the original source parses");
    let printed = print_with_locations(original);
    assert!(
        printed.contains("loc(\"in.runic\":1:1 to "),
        "unexpected location syntax: {printed}"
    );

    let reparsed = parse::parse_any(ParserConfig::new(context.clone()), "printed.runic", printed)
        .into_result()
        .expect("printed locations parse back");
    assert_eq!(location_of(reparsed), location_of(original));

    // The end line of a range is optional, and defaults to the start line.
    for (text, end_line) in [
        ("builtin.module public @t {} loc(\"in.runic\":1:1 to 1:5);", 1),
        ("builtin.module public @t {} loc(\"in.runic\":1:1 to :5);", 1),
    ] {
        let op = parse::parse_any(ParserConfig::new(context.clone()), "range.runic", text)
            .into_result()
            .unwrap_or_else(|err| panic!("`{text}` must parse: {err}"));
        let Location::FileLineColRange {
            end_line: parsed_end_line,
            end_column,
            ..
        } = location_of(op)
        else {
            panic!("`{text}` must give a range location");
        };
        assert_eq!((parsed_end_line.to_u32(), end_column.to_u32()), (end_line, 5), "`{text}`");
    }
}

fn diagnostic_messages(diagnostics: &crate::diagnostics::DiagnosticSet) -> Vec<String> {
    diagnostics.iter().map(|entry| entry.diagnostic.to_string()).collect()
}

/// Diagnostics emitted through the parser and through its context during a parse are collected
/// together, in order, by the parse's scope, and are not left pending on the context.
#[test]
fn diagnostics_emitted_during_a_parse_belong_to_its_scope() {
    use super::{DefaultParser, Parser, ParserState, Scanner, TokenStream};
    use crate::diagnostics::{OwnedDiagnostic, Severity, diagnostic};

    let context = Rc::new(Context::default());
    let id = context.add_source("warn.runic", "").unwrap();
    let sources = context.sources();
    let scope = context.diagnostic_scope();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));

    let _ =
        parser.emit(OwnedDiagnostic::new(diagnostic!(severity: Warning, message: "from parser")));
    let _ = context.emit(diagnostic!(severity: Warning, message: "from context"));
    drop(parser);

    let diagnostics = scope.finish().finish();
    assert_eq!(diagnostic_messages(&diagnostics), ["from parser", "from context"]);
    assert!(diagnostics.iter().all(|entry| entry.metadata().severity == Severity::Warning));
    assert!(
        context.take_diagnostics().is_empty(),
        "the scope's diagnostics leaked to the context"
    );
}

/// A diagnostic pending on the context before a parse is not attributed to that parse.
#[test]
fn diagnostics_pending_before_a_parse_stay_on_the_context() {
    use crate::diagnostics::diagnostic;

    let context = Rc::new(Context::default());
    let _ = context.emit(diagnostic!(severity: Warning, message: "pending"));

    let outcome = parse::parse_any(
        ParserConfig::new(context.clone()),
        "clean.runic",
        "builtin.module public @t {};",
    );
    assert!(outcome.is_ok());
    assert!(
        outcome.diagnostics.is_empty(),
        "the pending warning was attributed to the parse"
    );

    assert_eq!(diagnostic_messages(&context.take_diagnostics()), ["pending"]);
}

/// A failed parse reports exactly its own error, and leaves earlier diagnostics pending.
#[test]
fn a_failed_parse_leaves_pending_diagnostics_on_the_context() {
    use crate::diagnostics::{Severity, diagnostic};

    let context = Rc::new(Context::default());
    let _ = context.emit(diagnostic!(severity: Warning, message: "pending"));

    let outcome = parse::parse_any(
        ParserConfig::new(context.clone()),
        "bad.runic",
        "builtin.module public @m {",
    );
    assert!(outcome.result.is_err());
    assert_eq!(outcome.diagnostics.len(), 1);
    assert_eq!(outcome.diagnostics.iter().next().unwrap().metadata().severity, Severity::Error);

    assert_eq!(diagnostic_messages(&context.take_diagnostics()), ["pending"]);
}

/// A type of the test dialect that has no parameters, and so no parser.
#[derive(Debug, Default, PartialEq, Eq, Hash, crate::derive::DialectType)]
#[ty(dialect = TestDialect, name = "unit")]
struct UnitTestType;

/// Parses `text` as exactly one type, in a parser over `text` alone.
fn parse_type_text(context: &Rc<Context>, text: &str) -> Result<Type, Report> {
    use super::{DefaultParser, Parser, ParserError, ParserState, Scanner, Token, TokenStream};

    context.get_or_register_dialect::<TestDialect>();
    let id = context.add_source("type.runic", text).unwrap();
    let sources = context.sources();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
    let result = parser.parse_type().and_then(|ty| match parser.token_stream_mut().next()? {
        None => Ok(ty.into_inner()),
        Some(token) if matches!(token.inner(), Token::Eof) => Ok(ty.into_inner()),
        Some(token) => Err(ParserError::UnexpectedToken {
            span: token.span(),
            token: token.to_string(),
            expected: Some("end of input".into()),
        }),
    });
    result.map_err(|err| Report::from(err).attach_session_sources(context.sources()))
}

/// The rendered text of a type parse error, on one line.
fn type_error_text(context: &Rc<Context>, text: &str) -> String {
    let err = parse_type_text(context, text).expect_err(text);
    format!("{}", err.display()).split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn every_builtin_type_round_trips_through_the_printer_and_parser() {
    let context = Rc::new(Context::default());
    for text in [
        "?",
        "never",
        "i1",
        "i24",
        "i32",
        "i256",
        "isize",
        "f32",
        "f64",
        "ptr",
        "ptr<1>",
        "ptr<i32>",
        "ptr<i32, 3>",
        "ptr<ptr<i8>>",
        "array<i8; 16>",
        "array<i8; 0>",
        "array<array<i32; 2>; 3>",
        "struct<>",
        "struct<i32, i8>",
        "struct point<x: i32, y: i32>",
        "struct<packed(1); i8, i32 align(2)>",
        "struct<transparent; i32>",
        "struct<align(8); i64>",
        "struct<transparent;>",
        "struct<align(2);>",
        "struct<x: i32 align(2)>",
        "struct \"not an identifier\"<\"field one\": i32>",
        "(i32, i8) -> i32",
        "() -> (i32, i8)",
        "(i32) -> ()",
        "(ptr<(i32) -> i32>) -> ()",
        "() -> (() -> i32)",
        "(i32) -> ((i8) -> (i16, i32))",
        "!test.vector<f32 x 4>",
        "ptr<!test.vector<i8 x 16>>",
        "!test.unit",
    ] {
        let ty = parse_type_text(&context, text).unwrap_or_else(|err| panic!("{text}: {err:?}"));
        assert_eq!(format!("{ty}"), text);
        assert_eq!(parse_type_text(&context, text).unwrap(), ty, "{text} is not uniqued");
    }
    assert_eq!(
        parse_type_text(&context, "[i8; 16]").unwrap(),
        context.array_type(context.i8(), 16)
    );
}

#[test]
fn types_built_in_code_print_as_the_text_that_parses_to_them() {
    use crate::{StructField, StructType};

    let context = Rc::new(Context::default());
    let empty = context.array_type(context.i8(), 0);
    assert_eq!(format!("{empty}"), "array<i8; 0>");
    assert_eq!(parse_type_text(&context, "array<i8; 0>").unwrap(), empty);

    let field = StructField::new(context.i32())
        .named(crate::interner::Symbol::intern("x"))
        .aligned(core::num::NonZeroU16::new(2).unwrap());
    let aligned = context.struct_type(StructType::new([field]));
    assert_eq!(format!("{aligned}"), "struct<x: i32 align(2)>");
    assert_eq!(parse_type_text(&context, "struct<x: i32 align(2)>").unwrap(), aligned);
}

#[test]
fn pointers_in_the_default_address_space_are_the_plain_pointer_types() {
    let context = Rc::new(Context::default());
    assert_eq!(context.ptr_type_in(None, 0), context.opaque_ptr_type());
    assert_eq!(parse_type_text(&context, "ptr<0>").unwrap(), context.opaque_ptr_type());
    assert_eq!(
        parse_type_text(&context, "ptr<i32, 0>").unwrap(),
        context.ptr_type(context.i32())
    );
    assert_eq!(context.ptr_type(context.i32()).pointee(), Some(context.i32()));
    assert_eq!(context.opaque_ptr_type().pointee(), None);
    assert_eq!(context.i32().pointee(), None);
    assert_eq!(context.array_type(context.i8(), 4).pointee(), None);
}

#[test]
fn malformed_and_unknown_types_are_errors_not_panics() {
    let context = Rc::new(Context::default());
    for text in [
        "i0",
        "u32",
        "!nosuchdialect.thing",
        "!test.nosuchtype",
        "ptr<",
        "struct<packed(0); i8>",
        "(i32) ->",
        "(i32, i8)",
        "!test.vector<f32 4>",
        "!test.vector<f32 x>",
    ] {
        assert!(parse_type_text(&context, text).is_err(), "`{text}` should not parse");
    }
}

#[test]
fn dialect_type_errors_name_the_problem() {
    let context = Rc::new(Context::default());
    for (text, expected) in [
        ("!nosuchdialect.thing", "unknown dialect 'nosuchdialect'"),
        ("!test.nosuchtype", "unknown type 'nosuchtype' in dialect 'test'"),
        ("!test.unit<i32>", "type 'test.unit' takes no parameters, but '<' follows it"),
        ("!test.vector", "type 'test.vector' requires parameters in '<..>'"),
    ] {
        let message = type_error_text(&context, text);
        assert!(message.contains(expected), "`{text}`: {message}");
    }
}

/// An op whose signature and body use non-trivial builtin types and a dialect type prints as text
/// that parses back to the same op.
#[test]
fn an_op_using_builtin_and_dialect_types_round_trips() -> TestResult {
    let source = "\
builtin.module public @types {
    builtin.function public extern(\"C\") @f(%0: i64, %1: isize, %2: ptr<struct point<x: i32, y: \
                  i32>>, %3: !test.vector<f32 x 4>) -> !test.vector<f32 x 4> {
        %4 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<isize> }>;
        %5 = builtin.unrealized_conversion_cast %2 <{ ty = #builtin.type<ptr<!test.vector<i8 x \
                  16>>> }>;
        builtin.ret %3 : (!test.vector<f32 x 4>);
    };
};";
    let print = |text: &str| -> TestResult<String> {
        let test = ParserTest::default();
        test.context().get_or_register_dialect::<TestDialect>();
        let op = test.parse_any("types.hir", text)?;
        Ok(format!("{}", op.borrow()))
    };
    let printed = print(source)?;
    assert!(printed.contains("!test.vector<f32 x 4>"), "{printed}");
    assert!(printed.contains("ptr<struct point<x: i32, y: i32>>"), "{printed}");
    assert_eq!(print(&printed)?, printed);
    Ok(())
}

/// A field may be named like a struct representation; only a `:` after the name tells them apart.
#[test]
fn a_field_named_like_a_representation_round_trips() {
    let context = Rc::new(Context::default());
    for text in [
        "struct<align: i32>",
        "struct<packed: i8>",
        "struct<transparent: i8>",
        "struct<align(2); align: i32>",
        "struct<packed(1); packed: i8, transparent: i8>",
        "struct<transparent; transparent: i8>",
        "struct point<align: i32, y: i32>",
    ] {
        let ty = parse_type_text(&context, text).unwrap_or_else(|err| panic!("{text}: {err:?}"));
        assert_eq!(format!("{ty}"), text);
    }
}

/// The integer literals that a value of an `N`-bit integer type accepts, as text, with the `N`-bit
/// pattern each stands for: every value in `[-2^(N-1), 2^N - 1]`.
fn integer_literals_in_range(width: u32) -> Vec<(String, u128)> {
    let mask = if width == 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    let half = 1u128 << (width - 1);
    alloc::vec![
        ("0".into(), 0),
        ("1".into(), 1),
        ("-1".into(), mask),
        (format!("{}", half - 1), half - 1),
        (format!("-{half}"), half),
        (format!("{half}"), half),
        (format!("{mask}"), mask),
    ]
}

/// The integer literals just outside the range of an `N`-bit integer type.
fn integer_literals_out_of_range(width: u32) -> Vec<String> {
    let half = 1u128 << (width - 1);
    let above = if width == 128 {
        "340282366920938463463374607431768211456".into()
    } else {
        format!("{}", 1u128 << width)
    };
    alloc::vec![above, format!("-{}", half + 1)]
}

/// The immediate holding the `N`-bit pattern `bits`, as the parser stores an `N`-bit literal.
fn immediate_of_width(width: u32, bits: u128) -> Immediate {
    match width {
        1 => Immediate::I1(bits & 1 == 1),
        8 => Immediate::I8(bits as u8 as i8),
        16 => Immediate::I16(bits as u16 as i16),
        32 => Immediate::I32(bits as u32 as i32),
        64 => Immediate::I64(bits as u64 as i64),
        128 => Immediate::I128(bits as i128),
        _ => unreachable!(),
    }
}

/// A function returning `literal : iN` with `builtin.ret_imm`, after `test.constant literal : iN`.
fn function_with_integer_literal(width: u32, literal: &str) -> String {
    format!(
        "builtin.function public extern(\"C\") @f() -> i{width} {{\n    %0 = test.constant \
         {literal} : i{width};\n    builtin.ret_imm {literal} : i{width};\n}};"
    )
}

/// The values of the `test.constant` and the `builtin.ret_imm` in a parsed function.
fn integer_literal_values(function: &crate::UnsafeIntrusiveEntityRef<Function>) -> [Immediate; 2] {
    let function = function.borrow();
    let body = function.body();
    let entry = body.entry();
    let constant = entry.front().unwrap();
    let constant = constant.borrow();
    let constant = constant
        .get_typed_attribute::<crate::ImmediateAttr>("value")
        .expect("test.constant has an immediate value");
    let ret = entry.terminator().unwrap().try_downcast_op::<RetImm>().unwrap();
    let ret = ret.borrow().value().as_ref().as_immediate();
    let constant = constant.borrow().as_immediate();
    [constant, ret]
}

#[test]
fn integer_literals_of_every_width_round_trip_in_signed_storage() -> TestResult {
    for width in [8, 16, 32, 64, 128] {
        for (literal, bits) in integer_literals_in_range(width) {
            let expected = immediate_of_width(width, bits);
            let test = ParserTest::default();
            test.context().get_or_register_dialect::<TestDialect>();
            let source = function_with_integer_literal(width, &literal);
            let parsed = test.parse::<Function>("literal.hir", &source)?;
            assert_eq!(integer_literal_values(&parsed), [expected; 2], "{literal} : i{width}");

            let printed = format!("{}", parsed.as_operation_ref().borrow());
            let fresh = ParserTest::default();
            fresh.context().get_or_register_dialect::<TestDialect>();
            let reparsed = fresh.parse::<Function>("literal-printed.hir", &printed)?;
            assert_eq!(integer_literal_values(&reparsed), [expected; 2], "{printed}");
            assert_eq!(format!("{}", reparsed.as_operation_ref().borrow()), printed);
        }
        for literal in integer_literals_out_of_range(width) {
            let test = ParserTest::default();
            test.context().get_or_register_dialect::<TestDialect>();
            for source in [
                format!(
                    "builtin.function public extern(\"C\") @f() -> i{width} {{\n    \
                     builtin.ret_imm {literal} : i{width};\n}};"
                ),
                function_with_integer_literal(width, "0").replacen(
                    "test.constant 0",
                    &format!("test.constant {literal}"),
                    1,
                ),
            ] {
                assert!(
                    test.parse::<Function>("out-of-range.hir", &source).is_err(),
                    "`{literal} : i{width}` should be rejected"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn an_upper_half_literal_and_its_negative_spelling_are_the_same_constant() -> TestResult {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    let unsigned = test.parse::<Function>("255.hir", &function_with_integer_literal(8, "255"))?;
    let signed = test.parse::<Function>("-1.hir", &function_with_integer_literal(8, "-1"))?;
    assert_eq!(integer_literal_values(&unsigned), integer_literal_values(&signed));
    let printed = format!("{}", unsigned.as_operation_ref().borrow());
    assert!(printed.contains("builtin.ret_imm -1 : i8"), "{printed}");
    Ok(())
}

#[test]
fn a_boolean_literal_is_minus_one_zero_or_one() -> TestResult {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    for (literal, bit) in [("0", false), ("1", true), ("-1", true)] {
        let parsed =
            test.parse::<Function>("i1.hir", &function_with_integer_literal(1, literal))?;
        assert_eq!(integer_literal_values(&parsed), [Immediate::I1(bit); 2], "{literal} : i1");
        let printed = format!("{}", parsed.as_operation_ref().borrow());
        // A set bit prints as `1`, however it was written.
        let expected = format!("builtin.ret_imm {} : i1", u8::from(bit));
        assert!(printed.contains(&expected), "{printed}");
        let reparsed = test.parse::<Function>("i1-printed.hir", &printed)?;
        assert_eq!(integer_literal_values(&reparsed), [Immediate::I1(bit); 2], "{printed}");
    }
    for literal in ["2", "-2"] {
        let source = function_with_integer_literal(1, literal);
        assert!(test.parse::<Function>("i1-invalid.hir", &source).is_err(), "{literal} : i1");
    }
    Ok(())
}

#[test]
fn a_negated_hex_or_binary_literal_is_a_spanned_error() {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    for literal in ["-0x1", "-0b1"] {
        let source = function_with_integer_literal(8, "0").replacen(
            "builtin.ret_imm 0",
            &format!("builtin.ret_imm {literal}"),
            1,
        );
        let err = test.parse::<Function>("negated.hir", &source).err().expect(literal);
        let message = format!("{}", err.display());
        assert!(message.contains("only a decimal literal can be negative"), "{message}");
    }
}

/// An integer literal in attribute position, where the expected type is known, follows the same
/// rules as one in an op's own syntax.
#[test]
fn integer_literals_in_attribute_position_accept_a_sign() {
    use super::{DefaultParser, Parser, ParserState, Scanner, TokenStream};

    let context = Rc::new(Context::default());
    let parse = |text: &str, ty: Type| {
        let id = context.add_source("attr.runic", text).unwrap();
        let sources = context.sources();
        let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
        let token_stream = TokenStream::new(id, scanner);
        let mut parser =
            DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
        parser.parse_attribute(ty).map(|attr| {
            let attr = attr.into_inner();
            let attr = attr.borrow();
            attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap().as_immediate()
        })
    };
    assert_eq!(parse("-1", context.i8()).unwrap(), Immediate::I8(-1));
    assert_eq!(parse("255", context.i8()).unwrap(), Immediate::I8(-1));
    assert_eq!(parse("-128", context.i8()).unwrap(), Immediate::I8(-128));
    assert!(parse("-129", context.i8()).is_err());
    assert!(parse("256", context.i8()).is_err());
    assert!(parse("-0x1", context.i8()).is_err());
    assert_eq!(parse("-1", context.i1()).unwrap(), Immediate::I1(true));
}

/// The test dialect's builders make a constant in the representation its parser does, so equal
/// constants are equal however they were made.
#[test]
fn a_built_constant_equals_the_same_constant_parsed_from_text() -> TestResult {
    use crate::{AttributeRef, dialects::test::TestOpBuilder};

    let constant_value = |value: ValueRef| -> AttributeRef {
        let op = value.borrow().get_defining_op().unwrap();
        op.borrow().get_attribute("value").unwrap()
    };
    let immediate_and_type = |attr: AttributeRef| {
        let attr = attr.borrow();
        let attr = attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap();
        (attr.as_immediate(), attr.ty())
    };
    for (value, text) in [(7u32, "7"), (u32::MAX, "-1"), (u32::MAX, "4294967295")] {
        let mut test = ParserTest::default();
        test.context().get_or_register_dialect::<TestDialect>();
        let parsed =
            test.parse::<Function>("constant.hir", &function_with_integer_literal(32, text))?;
        let parsed = {
            let function = parsed.borrow();
            let body = function.body();
            let op = body.entry().front().unwrap();
            op.borrow().get_attribute("value").unwrap()
        };
        let i32 = test.context().i32();
        test.with_function("built", &[], &[i32]);
        let built = {
            let mut builder = test.function_builder();
            builder.u32(value, SourceSpan::UNKNOWN)?
        };
        let built = constant_value(built);
        assert_eq!(immediate_and_type(built), immediate_and_type(parsed), "{text}");
    }
    Ok(())
}
