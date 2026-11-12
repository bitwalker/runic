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
    BuilderExt, CallConv, Context, FunctionType, Int, OpParser, OpRegistration, OperationRef,
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
    use crate::{Op, dialects::builtin::attributes::I64Attr};

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
        let attr = test.context_rc().create_attribute::<I64Attr, _>(0x110000i64);
        let mut op = parsed;
        let mut op = op.borrow_mut();
        op.set_attribute(Module::RESERVED_MEMORY_ATTR, attr);
    }

    let flags = Default::default();
    let mut printer = AsmPrinter::root(test.context_rc(), &flags);
    printer.print_operation(parsed.borrow());
    let printed = printer.finish().to_string();
    std::println!("{printed}");

    let reparsed = test.parse_any("reserved_memory_roundtrip2.hir", &printed)?;
    let reparsed = reparsed.borrow();
    let module = reparsed.downcast_ref::<Module>().unwrap();

    let reserved = module
        .as_operation()
        .get_typed_attribute::<I64Attr>(Module::RESERVED_MEMORY_ATTR)
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
    let mut printer = AsmPrinter::root(test.context_rc(), &flags);
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
    let imm = ret_imm.borrow().value().as_int();
    assert_eq!(imm, Int::from(42i32), "expected the literal to be coerced to the declared type");

    // A literal that cannot be represented in the declared type must be rejected.
    let result = test.parse::<Function>(
        "parse_ret_imm_invalid.hir",
        "\
builtin.function public extern(\"C\") @retconst() -> i8 {
    builtin.ret_imm 300 : i8;
};",
    );
    assert!(result.is_err(), "expected an out-of-range literal to be rejected");

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

/// Well-formed as text, ill-formed as IR: `test.add` carries the `SameTypeOperands` trait, so an
/// addition of an `i32` and an `i64` parses (the text gives each operand its type) and then fails
/// verification.
const UNVERIFIABLE_SOURCE: &str = "\
builtin.module public @lib {
    builtin.function public extern(\"C\") @main(%0: i32, %1: i64) {
        %2 = test.add %0, %1 <{ overflow = #builtin.overflow<checked> }>;
        builtin.ret;
    };
};";

#[test]
fn parsing_verifies_what_it_parsed() {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();

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
        err.contains("invalid operation test.add"),
        "expected the verifier's rejection of the addition of mismatched types, got: {err}"
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
        let mut printer = AsmPrinter::root(test.context_rc(), &flags);
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
    let mut printer = AsmPrinter::root(test.context_rc(), &flags);
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
    let mut printer = AsmPrinter::root(test.context_rc(), &flags);
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
    let cases: [(&str, Expected); 2] = [
        ("array<i32; 4>", |context| context.array_type(context.i32(), 4)),
        ("ptr<array<array<i32; 4>; 2>>", |context| {
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
        let printed = format!("{}", crate::Op::as_operation(&*function.borrow()));
        let fresh = Rc::new(Context::default());
        parse::parse_any(ParserConfig::new(fresh.clone()), "array-roundtrip.hir", &printed)
            .unwrap();
    }
    for source in [
        "array<i32; -1>",
        "array<i32; 18446744073709551616>",
        "array<i32 4>",
        "array<i32; 4]",
    ] {
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
    context.get_or_register_dialect::<TestDialect>();
    let origin = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    assert!(context.sources().find_by_name(origin).is_none());
    // The addition fails verification (see `UNVERIFIABLE_SOURCE`), at its location
    let source = format!(
        "builtin.module public @lib {{\n    builtin.function public extern(\"C\") @main(%0: i32, \
         %1: i64) {{\n        %2 = test.add %0, %1 <{{ overflow = #builtin.overflow<checked> }}> \
         loc(\"{origin}\":1:1);\n        builtin.ret;\n    }};\n}};"
    );
    let outcome = parse::parse_any(ParserConfig::new(context.clone()), "origin.runic", source);
    assert!(context.sources().find_by_name(origin).is_some(), "the location loaded its file");

    let report = outcome.into_result().expect_err("an ill-typed addition fails verification");
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
        let mut printer = AsmPrinter::root(context.clone(), &flags);
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

/// Parses `text` as exactly one type, with the test dialect registered.
fn parse_type_text(context: &Rc<Context>, text: &str) -> Result<Type, Report> {
    context.get_or_register_dialect::<TestDialect>();
    parse::parse_type(context, "type.runic", text).into_result()
}

/// A dialect, not submitted to `inventory`, whose type's parser returns a type of another kind.
#[derive(Debug)]
struct LiarDialect {
    info: crate::DialectInfo,
}
impl crate::Dialect for LiarDialect {
    fn info(&self) -> &crate::DialectInfo {
        &self.info
    }
}
impl crate::DialectRegistration for LiarDialect {
    const NAMESPACE: &'static str = "liar";

    fn init(info: crate::DialectInfo) -> Self {
        Self { info }
    }
}

/// Prints an empty body, which its parser reads as `i32`.
#[derive(Debug, Default, PartialEq, Eq, Hash, crate::derive::DialectType)]
#[ty(dialect = LiarDialect, name = "liar")]
struct LiarType;
impl crate::TypePrinter for LiarType {
    fn print(&self, _: &mut crate::TypeFormatter<'_, '_>) -> core::fmt::Result {
        Ok(())
    }
}
impl crate::TypeParser for LiarType {
    fn parse(parser: &mut dyn super::Parser<'_>) -> ParseResult<Type> {
        Ok(parser.context().i32())
    }
}

#[test]
fn a_dialect_type_parser_must_return_its_own_type() {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<LiarDialect>();
    let liar = context.intern_type(LiarType);
    assert_eq!(format!("{liar}"), "!liar.liar<>");
    let message = type_error_text(&context, "!liar.liar<>");
    assert!(
        message.contains("the parser of type 'liar.liar' returned `i32`, a type of another kind"),
        "{message}"
    );
    assert_eq!(parse_type_text(&context, "!liar.liar").unwrap(), liar);
}

/// `parse::parse_type` parses exactly one type, and reports errors with the source attached.
#[test]
fn a_type_parses_from_text_on_its_own() {
    let context = Rc::new(Context::default());
    let outcome = parse::parse_type(&context, "ptr.runic", "ptr<i32>");
    assert!(outcome.diagnostics.is_empty());
    assert_eq!(outcome.into_result().unwrap(), context.ptr_type(context.i32()));

    for (text, expected) in [
        ("i32 i32", "unexpected token"),
        ("i32 i32", "help: end of input"),
        ("ptr<i32", "unexpected token 'end of input'"),
        ("%0", "help: type"),
    ] {
        let err = parse::parse_type(&context, "bad.runic", text).into_result().expect_err(text);
        let message = format!("{}", err.display());
        assert!(message.contains(expected), "`{text}`: {message}");
    }
    let err = parse::parse_type(&context, "rendered.runic", "i32 i32")
        .into_result()
        .unwrap_err();
    assert!(
        format!("{}", err.display()).contains("rendered.runic"),
        "the source is attached"
    );
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
}

/// Parses `text`, prints it, and checks that the text is `printed` and that it parses to the same
/// type again.
fn assert_type_prints_as(context: &Rc<Context>, text: &str, printed: &str) -> Type {
    let ty = parse_type_text(context, text).unwrap_or_else(|err| panic!("{text}: {err:?}"));
    assert_eq!(format!("{ty}"), printed, "{text}");
    assert_eq!(parse_type_text(context, printed).unwrap(), ty, "{printed} is not uniqued");
    ty
}

#[test]
fn enum_types_round_trip_through_the_printer_and_parser() {
    let context = Rc::new(Context::default());
    for text in [
        "enum<i8>",
        "enum<i8; a>",
        "enum option<i8; none, some: i32>",
        "enum color<i8; red = 1, green = 2, blue = 4>",
        "enum<i8; a = -1>",
        "enum<i8; a, b = 5, c = 6>",
        "enum<i1; a, b>",
        "enum<isize; a, b: ptr<i8>>",
        "enum<i128; a = -170141183460469231731687303715884105728>",
        "enum<i3; a = -4, b = -3>",
        "enum \"my enum\"<i32; struct, enum: i8, \"a b\">",
        "enum<i8; a: enum<i8; b: i8>>",
        "struct<enum<i8; a>>",
        "struct<enum: enum<i8; enum>>",
        "ptr<enum<i8; a, b>>",
    ] {
        assert_type_prints_as(&context, text, text);
    }
}

#[test]
fn enum_values_are_resolved_and_printed_as_signed_decimals() {
    let context = Rc::new(Context::default());
    let minus_one = assert_type_prints_as(&context, "enum<i8; a = -1>", "enum<i8; a = -1>");
    assert_eq!(parse_type_text(&context, "enum<i8; a = 255>").unwrap(), minus_one);
    assert_eq!(minus_one.as_enum().unwrap().variants()[0].value(), 255);
    assert_eq!(parse_type_text(&context, "enum<i8; a = 0xff>").unwrap(), minus_one);
    let implicit = assert_type_prints_as(&context, "enum<i8; a = 0, b = 1>", "enum<i8; a, b>");
    assert_eq!(parse_type_text(&context, "enum<i8; a, b>").unwrap(), implicit);
    let widest = assert_type_prints_as(
        &context,
        "enum<i128; a = 340282366920938463463374607431768211455>",
        "enum<i128; a = -1>",
    );
    assert_eq!(widest.as_enum().unwrap().variants()[0].value(), u128::MAX);
    assert_eq!(parse_type_text(&context, "enum<i128; a = -1>").unwrap(), widest);
}

#[test]
fn enum_types_nest_in_aggregates_and_pointers() {
    let context = Rc::new(Context::default());
    let color = parse_type_text(&context, "enum color<i8; red, green>").unwrap();
    let ty = parse_type_text(
        &context,
        "struct<x: enum color<i8; red, green>, y: ptr<enum color<i8; red, green>>>",
    )
    .unwrap();
    let fields = ty.as_struct().unwrap().fields();
    assert_eq!(fields[0].ty, color);
    assert_eq!(fields[1].ty.pointee(), Some(color));
}

#[test]
fn invalid_enum_types_are_reported_with_a_reason() {
    let context = Rc::new(Context::default());
    for (text, reason) in [
        ("enum<i8; a, a>", "more than one variant named `a`"),
        ("enum<i8; a = 1, b = 1>", "have the same value"),
        ("enum<i8; a = 1, b = 0, c>", "have the same value"),
        ("enum<i8; a = 256>", "out of range for a discriminant of 8 bits"),
        ("enum<i8; a = -129>", "out of range for a discriminant of 8 bits"),
        ("enum<i8; a = 0xfff>", "out of range for a discriminant of 8 bits"),
        ("enum<i1; a = 2>", "out of range for a discriminant of 1 bit"),
        ("enum<i128; a = 340282366920938463463374607431768211456>", "number too large"),
        ("enum<i128; a = -170141183460469231731687303715884105729>", "out of range"),
        ("enum<i8; a = 255, b>", "the implicit value of variant `b` does not fit"),
        ("enum<f32; a>", "the discriminant of an enum is an integer type"),
        ("enum<ptr; a>", "the discriminant of an enum is an integer type"),
        ("enum<i129; a>", "the discriminant of an enum is an integer type"),
        ("enum<struct<i8>; a>", "the discriminant of an enum is an integer type"),
        ("enum<i8; a = 0xff.>", ""),
        ("enum<i8; a = -0xff>", "only a decimal literal can be negative"),
        ("enum<i8; a", ""),
        ("enum<i8; a,>", ""),
        ("enum<i8;>", ""),
        ("enum<i8 a>", ""),
        ("enum<>", ""),
        ("enum", ""),
    ] {
        let message = type_error_text(&context, text);
        assert!(message.contains(reason), "{text}: {message}");
    }
}

#[cfg(feature = "std")]
#[test]
fn enum_payload_nesting_is_limited() {
    on_a_1_mb_stack(|| {
        let context = Rc::new(Context::default());
        let deepest = nested_type_text("enum<i8; a: ", "i32", ">", 64);
        if let Err(err) = parse_type_text(&context, &deepest) {
            panic!("64 levels should parse: {err:?}");
        }
        let message = type_error_text(&context, &nested_type_text("enum<i8; a: ", "i32", ">", 65));
        assert!(message.contains("type nesting is too deep (limit 64)"), "{message}");
        let message = type_error_text(&context, &nested_type_text("enum<i8; a: ", "", "", 10_000));
        assert!(message.contains("type nesting is too deep (limit 64)"), "{message}");
    });
}

#[test]
fn a_builtin_enum_named_as_a_dialect_type_gives_its_bare_spelling() {
    let context = Rc::new(Context::default());
    let message = type_error_text(&context, "!builtin.enum");
    assert!(message.contains("`enum<..>`"), "{message}");
}

/// A builtin type named as if it were a dialect type is an error that gives its bare spelling.
#[test]
fn builtin_types_are_written_without_a_dialect_prefix() {
    let context = Rc::new(Context::default());
    for (text, spelling) in [
        ("!builtin.int<32>", "`iN`, such as `i32`"),
        ("!builtin.isize", "`isize`"),
        ("!builtin.ptr", "`ptr`"),
        ("!builtin.unknown", "`?`"),
        ("!builtin.nosuch", ""),
    ] {
        let message = type_error_text(&context, text);
        assert!(
            message.contains("builtin types are written without the '!builtin.' prefix"),
            "`{text}`: {message}"
        );
        assert!(message.contains(spelling), "`{text}`: {message}");
    }
    let builtin = context.get_registered_dialect("builtin");
    for ty in builtin.registered_types() {
        assert!(super::types::bare_builtin_spelling(ty.name().as_str()).is_some(), "{ty}");
    }
}

/// An array type is written `array<T; N>` only: `[`..`]` is left to list attributes.
#[test]
fn the_bracket_spelling_of_an_array_type_is_not_a_type() {
    let context = Rc::new(Context::default());
    for text in ["[i8; 16]", "ptr<[i8; 16]>"] {
        let message = type_error_text(&context, text);
        assert!(message.contains("unexpected token '['"), "`{text}`: {message}");
        assert!(message.contains("help: type"), "`{text}`: {message}");
    }
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
        ("!test.vector<f32 x 0>", "a vector has at least one lane"),
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

/// A function returning `literal : iN` with `builtin.ret_imm`, after `test.constant literal : iN`.
fn function_with_integer_literal(width: u32, literal: &str) -> String {
    format!(
        "builtin.function public extern(\"C\") @f() -> i{width} {{\n    %0 = test.constant \
         {literal} : i{width};\n    builtin.ret_imm {literal} : i{width};\n}};"
    )
}

/// The values of the `test.constant` and the `builtin.ret_imm` in a parsed function.
fn integer_literal_values(function: &crate::UnsafeIntrusiveEntityRef<Function>) -> [Int; 2] {
    let function = function.borrow();
    let body = function.body();
    let entry = body.entry();
    let constant = entry.front().unwrap();
    let constant = constant.borrow();
    let constant = constant
        .get_typed_attribute::<crate::dialects::builtin::attributes::IntAttr>("value")
        .expect("test.constant has an integer value");
    let ret = entry.terminator().unwrap().try_downcast_op::<RetImm>().unwrap();
    let ret = ret.borrow().value().as_int();
    let constant = constant.borrow().as_int();
    [constant, ret]
}

#[test]
fn integer_literals_of_every_width_round_trip() -> TestResult {
    for width in [8, 16, 32, 64, 128] {
        for (literal, bits) in integer_literals_in_range(width) {
            let expected = Int::from_bits(width, bits);
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
        assert_eq!(integer_literal_values(&parsed), [Int::from(bit); 2], "{literal} : i1");
        let printed = format!("{}", parsed.as_operation_ref().borrow());
        // A set bit prints as `1`, however it was written.
        let expected = format!("builtin.ret_imm {} : i1", u8::from(bit));
        assert!(printed.contains(&expected), "{printed}");
        let reparsed = test.parse::<Function>("i1-printed.hir", &printed)?;
        assert_eq!(integer_literal_values(&reparsed), [Int::from(bit); 2], "{printed}");
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
            attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap().as_int()
        })
    };
    assert_eq!(parse("-1", context.i8()).unwrap(), Int::from(-1i8));
    assert_eq!(parse("255", context.i8()).unwrap(), Int::from(-1i8));
    assert_eq!(parse("-128", context.i8()).unwrap(), Int::from(-128i8));
    assert!(parse("-129", context.i8()).is_err());
    assert!(parse("256", context.i8()).is_err());
    assert!(parse("-0x1", context.i8()).is_err());
    assert_eq!(parse("-1", context.i1()).unwrap(), Int::from(true));
}

/// The builtin integer attributes read the same literals the typed literal readers do, so what
/// they print reads back, and `255` and `-1` are one `i8`.
#[test]
fn builtin_integer_attributes_accept_a_sign_and_round_trip() -> TestResult {
    use super::{DefaultParser, Parser, ParserState, Scanner, TokenStream};

    let context = Rc::new(Context::default());
    let parse = |text: &str| {
        let id = context.add_source("attr.runic", text).unwrap();
        let sources = context.sources();
        let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
        let token_stream = TokenStream::new(id, scanner);
        let mut parser =
            DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
        parser.parse_attribute(context.unknown_type()).map(|attr| attr.into_inner())
    };
    let value_of = |attr: &crate::AttributeRef| {
        let attr = attr.borrow();
        attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap().as_int()
    };
    let printed = |attr: &crate::AttributeRef| {
        let flags = Default::default();
        let mut printer = AsmPrinter::root(context.clone(), &flags);
        printer.print_attribute_value(&*attr.borrow());
        printer.finish().to_string()
    };

    let u128_max = "340282366920938463463374607431768211455";
    for (text, expected, spelled) in [
        ("#builtin.i8<-1>", Int::from(-1i8), "#builtin.i8<-1>"),
        ("#builtin.i8<255>", Int::from(-1i8), "#builtin.i8<-1>"),
        ("#builtin.i8<-128>", Int::from(-128i8), "#builtin.i8<-128>"),
        ("#builtin.i16<-1>", Int::from(-1i16), "#builtin.i16<-1>"),
        ("#builtin.i32<-1>", Int::from(-1i32), "#builtin.i32<-1>"),
        ("#builtin.i32<4294967295>", Int::from(-1i32), "#builtin.i32<-1>"),
        ("#builtin.i64<-1>", Int::from(-1i64), "#builtin.i64<-1>"),
        ("#builtin.i128<-1>", Int::from(-1i128), "#builtin.i128<-1>"),
        (&format!("#builtin.i128<{u128_max}>"), Int::from(-1i128), "#builtin.i128<-1>"),
    ] {
        let attr = parse(text)?;
        assert_eq!(value_of(&attr), expected, "{text}");
        assert_eq!(printed(&attr), spelled, "{text}");
        let reparsed = parse(&printed(&attr))?;
        assert_eq!(value_of(&reparsed), expected, "{text}");
        assert_eq!(printed(&reparsed), spelled, "{text}");
    }
    // The unsigned kinds are gone
    for text in ["#builtin.u8<255>", "#builtin.u8<-1>"] {
        let err = parse(text).err().unwrap_or_else(|| panic!("{text} should not parse"));
        assert!(
            matches!(err, crate::parse::ParserError::UnknownAttribute { .. }),
            "{text}: {err:?}"
        );
    }

    for text in ["#builtin.i8<256>", "#builtin.i8<-129>", "#builtin.i8<-0x1>"] {
        let err = parse(text).err().unwrap_or_else(|| panic!("{text} should not parse"));
        assert!(
            matches!(err, crate::parse::ParserError::InvalidIntegerLiteral { .. }),
            "{text}: {err:?}"
        );
    }

    // An operation carrying such an attribute round-trips through print and parse
    let test = ParserTest::default();
    let module = test.parse_any(
        "signed_attr.hir",
        "builtin.module public @m attributes { a = #builtin.i8<255>, b = #builtin.i128<-1> } {};",
    )?;
    let printed = format!("{}", module.borrow());
    assert!(
        printed.contains("#builtin.i8<-1>") && printed.contains("#builtin.i128<-1>"),
        "{printed}"
    );
    let reparsed = test.parse_any("signed_attr2.hir", &printed)?;
    assert_eq!(format!("{}", reparsed.borrow()), printed);
    Ok(())
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
    let int_and_type = |attr: AttributeRef| {
        let attr = attr.borrow();
        let attr = attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap();
        (attr.as_int(), attr.ty())
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
        assert_eq!(int_and_type(built), int_and_type(parsed), "{text}");
    }
    Ok(())
}

/// A word spelled like an integer type too wide to name one is an identifier, keeping its
/// spelling, rather than a type token with a clamped width.
#[test]
fn an_over_wide_integer_type_spelling_is_an_identifier() {
    use super::{DefaultParser, Parser, ParserState, Scanner, TokenStream};

    let context = Rc::new(Context::default());
    let message = type_error_text(&context, "i99999999999");
    assert!(message.contains("unexpected token 'bare identifier'"), "{message}");
    assert!(message.contains("help: type"), "{message}");

    let id = context.add_source("key.runic", "i99999999999").unwrap();
    let sources = context.sources();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
    let key = parser.parse_optional_keyword_or_string().unwrap().unwrap().into_inner();
    assert_eq!(key.as_str(), "i99999999999");

    // The widest integer type parses as itself and prints as it was written.
    let widest = parse_type_text(&context, "i16777215").unwrap();
    assert_eq!(widest.integer_width(), Some(16_777_215));
    assert_eq!(widest, context.int_type(16_777_215));
    assert_eq!(format!("{widest}"), "i16777215");
    assert_eq!(parse_type_text(&context, &format!("{widest}")).unwrap(), widest);

    // A width that fits is still an integer type, and one beyond the limit is a type error.
    let message = type_error_text(&context, "i16777216");
    assert!(message.contains("an integer type is 1 to 16777215 bits wide"), "{message}");
}

/// `depth` levels of `open` around `i32`, each closed by `close`; unterminated if `close` is empty.
fn nested_type_text(open: &str, inner: &str, close: &str, depth: usize) -> String {
    format!("{}{inner}{}", open.repeat(depth), close.repeat(depth))
}

/// Runs `body` on a thread with a stack of exactly 1 MB, the smallest stack a parser is expected
/// to run on, so that the result does not depend on the test harness's thread size.
#[cfg(feature = "std")]
fn on_a_1_mb_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(body)
        .expect("spawn a thread")
        .join()
        .expect("type nesting is limited");
}

/// Type nesting is limited to 64 levels, so that deeply nested input is an error instead of a
/// stack overflow, on a 1 MB stack and in an unoptimized build.
#[cfg(feature = "std")]
#[test]
fn type_nesting_is_limited_to_64_levels() {
    on_a_1_mb_stack(|| {
        let context = Rc::new(Context::default());
        for (open, inner, close) in [
            ("ptr<", "i32", ">"),
            ("struct<", "i32", ">"),
            ("array<", "i32", "; 1>"),
            ("(", "i32", ") -> ()"),
            ("!test.vector<", "i32", " x 1>"),
        ] {
            let deepest = nested_type_text(open, inner, close, 64);
            if let Err(err) = parse_type_text(&context, &deepest) {
                panic!("{open}: 64 levels should parse: {err:?}");
            }

            let message = type_error_text(&context, &nested_type_text(open, inner, close, 65));
            assert!(message.contains("type nesting is too deep (limit 64)"), "{open}: {message}");

            let message = type_error_text(&context, &nested_type_text(open, "", "", 10_000));
            assert!(message.contains("type nesting is too deep (limit 64)"), "{open}: {message}");
        }
    });
}

/// A type nested to the limit in a function signature is parsed by the operation parser, whose
/// own frames are live below it, and still fits a 1 MB stack.
#[cfg(feature = "std")]
#[test]
fn a_type_nested_to_the_limit_parses_in_a_function_signature() {
    on_a_1_mb_stack(|| {
        for (open, inner, close) in [
            ("ptr<", "i32", ">"),
            ("struct<", "i32", ">"),
            ("array<", "i32", "; 1>"),
            ("(", "i32", ") -> ()"),
        ] {
            let test = ParserTest::default();
            let deepest = nested_type_text(open, inner, close, 64);
            let source = format!(
                "builtin.function public extern(\"C\") @deep(%a: {deepest}) {{\n    ret;\n}};"
            );
            if let Err(err) = test.parse::<Function>("deep.hir", &source) {
                panic!("{open}: 64 levels in a signature should parse: {err:?}");
            }
        }
    });
}

/// Parses `text` as an attribute dictionary, in a parser over `text` alone, and returns the
/// rendered error on one line.
fn attribute_dict_error_text(context: &Rc<Context>, text: &str) -> String {
    use super::{DefaultParser, ParsedAttrs, Parser, ParserState, Scanner, TokenStream};

    let id = context.add_source("attrs.runic", text).unwrap();
    let sources = context.sources();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
    let mut attrs = ParsedAttrs::default();
    let err = parser.parse_attribute_dict(&mut attrs).expect_err(text);
    let err = Report::from(err).attach_session_sources(context.sources());
    format!("{}", err.display()).split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Attribute text the parser does not support is reported as an error, not a panic.
#[test]
fn unsupported_attribute_text_is_an_error_not_a_panic() {
    let context = Rc::new(Context::default());
    for (text, expected) in [
        ("{ a = {} }", "a dictionary is not supported as an attribute value"),
        ("{ a = #nosuch.attr<1> }", "unknown dialect 'nosuch'"),
        ("{ a = #nosuch.attr }", "unknown dialect 'nosuch'"),
    ] {
        let message = attribute_dict_error_text(&context, text);
        assert!(message.contains(expected), "`{text}`: {message}");
    }
}

/// A small deterministic generator (xorshift64*), so that generated tests need no dependency
/// and fail the same way every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A number below `bound`.
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

/// A random builtin (or test dialect) type nested at most `depth` levels deep.
fn random_type(rng: &mut Rng, context: &Context, depth: u32) -> Type {
    use core::num::NonZeroU16;

    use crate::{StructField, StructRepr, StructType, dialects::test::TestVectorType};

    const NAMES: &[&str] = &[
        "x", "point", "_1", "ptr", "align", "packed", "never", "my point", "a\"b", "é", "a\\b",
    ];
    let name = |rng: &mut Rng| crate::interner::Symbol::intern(rng.pick(NAMES));
    let align = |rng: &mut Rng| NonZeroU16::new(1 << rng.below(4)).unwrap();

    let leaf = depth == 0 || rng.below(3) == 0;
    if leaf {
        return match rng.below(9) {
            0 => context.int_type(1 + rng.below(300) as u32),
            1 => *rng.pick(&[context.i1(), context.i8(), context.i32(), context.i128()]),
            2 => context.isize(),
            3 => context.f32(),
            4 => context.f64(),
            5 => context.unknown_type(),
            6 => context.never_type(),
            7 => context.opaque_ptr_type(),
            _ => context.ptr_type_in(None, rng.below(4) as u32),
        };
    }
    let mut child = |rng: &mut Rng| random_type(rng, context, depth - 1);
    match rng.below(6) {
        0 => {
            let pointee = child(rng);
            context.ptr_type_in(Some(pointee), *rng.pick(&[0, 0, 1, u32::MAX]))
        }
        1 => {
            let element = child(rng);
            context.array_type(element, *rng.pick(&[0, 1, 16, u64::MAX]))
        }
        2 | 3 => {
            let fields = (0..rng.below(4))
                .map(|_| {
                    let mut field = StructField::new(child(rng));
                    if rng.below(2) == 0 {
                        field = field.named(name(rng));
                    }
                    if rng.below(4) == 0 {
                        field = field.aligned(align(rng));
                    }
                    field
                })
                .collect::<Vec<_>>();
            let ty = if rng.below(2) == 0 {
                StructType::named(name(rng), fields)
            } else {
                StructType::new(fields)
            };
            let repr = match rng.below(5) {
                0 => StructRepr::Align(align(rng)),
                1 => StructRepr::Packed(align(rng)),
                2 => StructRepr::Transparent,
                _ => StructRepr::Default,
            };
            context.struct_type(ty.with_repr(repr))
        }
        4 => {
            let params = (0..rng.below(3)).map(|_| child(rng)).collect::<Vec<_>>();
            let results = (0..rng.below(3)).map(|_| child(rng)).collect::<Vec<_>>();
            context.function_type(params, results)
        }
        _ => {
            let element = child(rng);
            context.intern_type(TestVectorType::new(element, 1 + rng.below(8) as u32))
        }
    }
}

/// Generated nested types print as text that parses back to the same type.
#[test]
fn generated_types_round_trip_through_their_text() {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    for case in 0..6000 {
        let ty = random_type(&mut rng, &context, 6);
        let text = format!("{ty}");
        let parsed = parse_type_text(&context, &text)
            .unwrap_or_else(|err| panic!("case {case}: `{text}` does not parse: {err:?}"));
        assert_eq!(parsed, ty, "case {case}: `{text}` parses as `{parsed}`");
    }
}

/// Many distinct types intern to distinct handles, and interning them again finds the same ones,
/// across the uniquer's growth and the arena's chunks.
#[test]
fn interning_many_types_finds_each_again() {
    use crate::{StructField, StructType, interner::Symbol};

    const COUNT: u32 = 100_000;
    let context = Context::default();
    let build = |i: u32| {
        let base = context.int_type(1 + i % 4096);
        match i % 4 {
            0 => context.ptr_type_in(Some(base), i),
            1 => context.array_type(base, u64::from(i)),
            2 => context.function_type([base], [context.ptr_type_in(None, i)]),
            _ => context.struct_type(StructType::named(
                Symbol::intern(format!("s{i}")),
                [StructField::new(base), StructField::new(context.ptr_type(base))],
            )),
        }
    };
    let handles = (0..COUNT).map(build).collect::<Vec<_>>();
    let distinct = handles.iter().map(|ty| ty.as_ptr()).collect::<crate::FxHashSet<_>>();
    assert_eq!(distinct.len(), COUNT as usize);
    for (i, ty) in handles.iter().enumerate() {
        assert_eq!(build(i as u32), *ty, "type {i} was interned twice");
    }
    assert_eq!(format!("{}", handles[3]), "struct s3<i4, ptr<i4>>");
}

/// The rendered text of the error that parsing `source` gives, on one line.
fn parse_error_text(source: &str) -> String {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    let err = test.parse_any("declarations.hir", source).expect_err(source);
    format!("{}", err.display()).split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The types of the arguments of the function `f` that `source` defines.
fn argument_types(test: &ParserTest, source: &str) -> Vec<Type> {
    let function = test
        .parse::<Function>("declarations.hir", source)
        .unwrap_or_else(|err| panic!("{source}\n{err:?}"));
    let function = function.borrow();
    let entry = function.entry_block();
    let entry = entry.borrow();
    entry.arguments().iter().map(|arg| (*arg as ValueRef).borrow().ty()).collect()
}

#[test]
fn a_declared_type_alias_is_usable_in_later_declarations_and_operations() {
    let test = ParserTest::default();
    let source = "\
type !a = i32;
type !b = ptr<!a>;
builtin.function public extern(\"C\") @f(%0: !a, %1: !b) {
    builtin.ret;
};";
    let context = test.context();
    assert_eq!(argument_types(&test, source), [context.i32(), context.ptr_type(context.i32())]);
}

#[test]
fn type_alias_declarations_keep_their_checks() {
    let message = parse_error_text("type !a = i32;\ntype !a = i64;\nbuiltin.world {};");
    assert!(message.contains("alias 'a' is already defined"), "{message}");

    let message = parse_error_text("type !a.b = i32;\nbuiltin.world {};");
    assert!(message.contains("reserved for dialect-defined names"), "{message}");

    let message = parse_error_text("type !a = i32\nbuiltin.world {};");
    assert!(message.contains("invalid syntax"), "a missing ';': {message}");
}

#[test]
fn the_bare_type_alias_form_is_an_error() {
    let message = parse_error_text("!a = i32;\nbuiltin.world {};");
    assert!(
        message.contains("a type alias is declared with `type !name = ..;`"),
        "{message}"
    );
}

#[test]
fn type_and_rec_remain_attribute_names() {
    let test = ParserTest::default();
    let source = "\
builtin.function public extern(\"C\") @f(%0: i32) {
    %1 = builtin.unrealized_conversion_cast %0 <{ ty = #builtin.type<i64> }> attributes { type = \
                  true, rec = false };
    builtin.ret;
};";
    let function = test.parse::<Function>("names.hir", source).unwrap();
    let printed = format!("{}", crate::Op::as_operation(&*function.borrow()));
    assert!(
        printed.contains("attributes { rec = #builtin.bool<false>, type = #builtin.bool<true> }"),
        "{printed}"
    );
}

/// The types of the arguments of a function `f` whose argument types are `arguments`, parsed after
/// the declarations `declarations`.
fn declared_types(test: &ParserTest, declarations: &str, arguments: &[&str]) -> Vec<Type> {
    let arguments = arguments
        .iter()
        .enumerate()
        .map(|(i, ty)| format!("%{i}: {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "{declarations}\nbuiltin.function public extern(\"C\") @f({arguments}) {{\n    \
         builtin.ret;\n}};"
    );
    argument_types(test, &source)
}

/// The type of field `index` of the struct `ty`.
fn field(ty: Type, index: usize) -> Type {
    ty.as_struct().unwrap_or_else(|| panic!("`{ty}` is a struct")).fields()[index].ty
}

const LIST_AND_NODE: &str = "\
rec {
  type !list = struct list<i32, ptr<!node>>;
  type !node = struct node<ptr<!list>, ptr<!node>>;
}";

#[test]
fn a_rec_block_defines_members_that_refer_to_each_other() {
    let test = ParserTest::default();
    let [list, node] = declared_types(&test, LIST_AND_NODE, &["!list", "!node"])[..] else {
        unreachable!()
    };
    assert_eq!(list.rec_group(), node.rec_group());
    assert!(list.rec_group().is_some());
    assert_eq!((list.rec_index(), node.rec_index()), (Some(0), Some(1)));
    assert_eq!(field(list, 0), test.context().i32());
    assert_eq!(field(list, 1).pointee(), Some(node));
    assert_eq!(field(node, 0).pointee(), Some(list));
    assert_eq!(field(node, 1).pointee(), Some(node));
    assert!(!list.is_open() && !node.is_open());
}

#[test]
fn the_members_of_a_rec_block_are_in_declaration_order() {
    let test = ParserTest::default();
    let types = declared_types(
        &test,
        "rec {\n  type !b = struct b<ptr<!a>>;\n  type !a = struct a<ptr<!b>>;\n}",
        &["!a", "!b"],
    );
    let [a, b] = types[..] else { unreachable!() };
    assert_eq!((b.rec_index(), a.rec_index()), (Some(0), Some(1)));
    assert_eq!(field(b, 0).pointee(), Some(a));
    assert_eq!(field(a, 0).pointee(), Some(b));

    // `z` is mentioned before `y` is declared, so the variables are renumbered.
    let types = declared_types(
        &test,
        "rec {
  type !x = struct x<ptr<!z>, ptr<!y>>;
  type !y = struct y<ptr<!x>>;
  type !z = struct z<ptr<!y>>;
}",
        &["!x", "!y", "!z"],
    );
    let [x, y, z] = types[..] else { unreachable!() };
    assert_eq!([x, y, z].map(|ty| ty.rec_index()), [Some(0), Some(1), Some(2)]);
    assert_eq!(field(x, 0).pointee(), Some(z));
    assert_eq!(field(x, 1).pointee(), Some(y));
    assert_eq!(field(y, 0).pointee(), Some(x));
    assert_eq!(field(z, 0).pointee(), Some(y));
}

#[test]
fn a_rec_block_of_one_member() {
    let test = ParserTest::default();
    let context = test.context();

    let [list] = declared_types(&test, "rec { type !l = struct l<i32, ptr<!l>>; }", &["!l"])[..]
    else {
        unreachable!()
    };
    assert_eq!(list.rec_index(), Some(0));
    assert_eq!(field(list, 1).pointee(), Some(list));

    // A closed body is the plain type itself.
    let [closed] = declared_types(&test, "rec { type !c = struct c<i32>; }", &["!c"])[..] else {
        unreachable!()
    };
    assert_eq!(closed.rec_group(), None);
    assert_eq!(
        closed,
        context.struct_type(crate::StructType::named(
            crate::interner::Symbol::intern("c"),
            [context.i32().into()]
        ))
    );
}

#[test]
fn an_alias_defined_before_a_rec_block_is_not_a_member() {
    let test = ParserTest::default();
    let types = declared_types(
        &test,
        "type !o = struct o<i32>;\nrec { type !l = struct l<ptr<!o>, ptr<!l>>; }",
        &["!o", "!l"],
    );
    let [outer, list] = types[..] else {
        unreachable!()
    };
    assert_eq!(outer.rec_group(), None);
    assert_eq!(field(list, 0).pointee(), Some(outer));
    assert_eq!(field(list, 1).pointee(), Some(list));
}

#[test]
fn a_rec_block_parsed_twice_under_other_names_gives_the_same_members() {
    let test = ParserTest::default();
    let declarations = format!(
        "{LIST_AND_NODE}
rec {{
  type !l2 = struct list<i32, ptr<!n2>>;
  type !n2 = struct node<ptr<!l2>, ptr<!n2>>;
}}"
    );
    let types = declared_types(&test, &declarations, &["!list", "!node", "!l2", "!n2"]);
    assert_eq!(types[0], types[2]);
    assert_eq!(types[1], types[3]);
}

#[test]
fn the_alias_spellings_the_printer_chooses_parse() {
    let test = ParserTest::default();
    let types = declared_types(
        &test,
        "rec {
  type !list_0 = struct<ptr<!t0>>;
  type !t0 = struct<ptr<!a$b>>;
  type !a$b = struct a$b<ptr<!list_0>>;
}",
        &["!list_0", "!t0", "!a$b"],
    );
    assert_eq!(
        types.iter().map(|ty| ty.rec_index()).collect::<Vec<_>>(),
        [Some(0), Some(1), Some(2)]
    );
    assert_eq!(field(types[2], 0).pointee(), Some(types[0]));
}

#[test]
fn invalid_rec_blocks_are_reported() {
    let world = "\nbuiltin.world {};";
    for (declarations, expected) in [
        ("rec { type !a = struct<ptr<!b>>; }", "use of undefined type alias '!b'"),
        (
            "rec { type !a = struct<ptr<!a>>; type !a = struct<i32>; }",
            "alias 'a' is already defined",
        ),
        (
            "type !a = i32;\nrec { type !a = struct<ptr<!a>>; }",
            "alias 'a' is already defined",
        ),
        ("rec { }", "a `rec` block declares at least one type"),
        ("rec { rec { type !a = struct<ptr<!a>>; } }", "`rec` blocks do not nest"),
        (
            "rec { type !a = !a; }",
            "member 0 of a recursion group is a bare recursion variable",
        ),
        (
            "rec { type !a = struct<ptr<!b>>; type !b = i32; }",
            "member 1 of a recursion group is a scalar type",
        ),
        ("rec { type !a = struct<ptr<!a>> }", "invalid syntax"),
        ("rec { builtin.world {}; }", "a type declaration, 'type !name = ..;', or '}'"),
    ] {
        let message = parse_error_text(&format!("{declarations}{world}"));
        assert!(message.contains(expected), "{declarations}: {message}");
    }

    let message = parse_error_text("rec { type !a = struct<ptr<!a>>;");
    assert!(message.contains("unexpected end of input"), "{message}");

    // The primary label of each error: the first occurrence of `labelled` in the source, without
    // its first `skip` bytes (an alias's span leaves out the `!`).
    for (declarations, labelled, skip) in [
        ("rec { type !a = struct<ptr<!b>>; type !c = struct<ptr<!b>>; }", "!b", 1),
        ("rec { type !a = struct<ptr<!b>>; type !b = i32; }", "type !b = i32;", 0),
        ("rec { }", "rec", 0),
    ] {
        let source = format!("{declarations}{world}");
        let (message, labels) = parse_error_labels(&source);
        let (start, end) = range_of(&source, labelled);
        assert_eq!(labels.first(), Some(&(start + skip, end)), "{source}: {message}");
    }
}

#[test]
fn an_alias_mentioned_twice_before_its_declaration() {
    let test = ParserTest::default();
    let types = declared_types(
        &test,
        "rec { type !a = struct<ptr<!c>>; type !b = struct<ptr<!c>>; type !c = struct<ptr<!a>>; }",
        &["!a", "!b", "!c"],
    );
    let [a, b, c] = types[..] else { unreachable!() };
    assert_eq!([a, b, c].map(|ty| ty.rec_index()), [Some(0), Some(1), Some(2)]);
    assert_eq!(field(a, 0).pointee(), Some(c));
    assert_eq!(field(b, 0).pointee(), Some(c));
    assert_eq!(field(c, 0).pointee(), Some(a));
}

/// The message of the error that parsing `source` gives, on one line, and the ranges of its labels.
fn parse_error_labels(source: &str) -> (String, Vec<(usize, usize)>) {
    let test = ParserTest::default();
    let err = test.parse_any("declarations.hir", source).expect_err(source);
    let labels = crate::testing::label_ranges(err.as_diagnostic())
        .into_iter()
        .map(|range| (range.start, range.end))
        .collect();
    let message = format!("{}", err.display()).split_whitespace().collect::<Vec<_>>().join(" ");
    (message, labels)
}

/// The byte range of the first occurrence of `text` in `source`.
fn range_of(source: &str, text: &str) -> (usize, usize) {
    let start = source.find(text).unwrap_or_else(|| panic!("`{text}` in `{source}`"));
    (start, start + text.len())
}

#[test]
fn a_member_of_another_group_is_not_a_body_in_a_rec_block() {
    let source = "\
rec { type !list = struct list<ptr<!list>, i32>; }
rec { type !x = !list; type !y = struct<ptr<!x>>; }
builtin.world {};";
    let (message, labels) = parse_error_labels(source);
    assert!(
        message.contains(
            "member 0 is a member of another recursion group: define it with its own body"
        ),
        "{message}"
    );
    assert_eq!(labels, [range_of(source, "type !x = !list;")], "{message}");

    // Alone, a member is a closed body, and the block makes an alias of it.
    let test = ParserTest::default();
    let [list, x] = declared_types(
        &test,
        "rec { type !list = struct list<ptr<!list>, i32>; }\nrec { type !x = !list; }",
        &["!list", "!x"],
    )[..] else {
        unreachable!()
    };
    assert_eq!(x, list);
    assert_eq!(list.rec_index(), Some(0));
}

#[test]
fn the_diagnostic_form_of_a_member_does_not_parse() {
    let context = Rc::new(Context::default());
    let list = self_recursive(&context, Some("list"), context.i32());
    let text = format!("{}", context.ptr_type(list));
    assert_eq!(text, "ptr<!list@0.0>");
    assert!(parse::parse_type(&context, "member.runic", &text).into_result().is_err());

    // Not even where an alias of that name is declared.
    let message = parse_error_text(&format!(
        "rec {{ type !list = struct list<ptr<!list>, i32>; }}\nbuiltin.function public \
         extern(\"C\") @f(%0: {text}) {{\n    builtin.ret;\n}};"
    ));
    assert!(!message.is_empty());
}

#[test]
fn an_undefined_alias_outside_a_rec_block_is_still_undefined() {
    let message = parse_error_text(
        "builtin.function public extern(\"C\") @f(%0: !nope) {\n    builtin.ret;\n};",
    );
    assert!(message.contains("use of undefined type alias '!nope'"), "{message}");

    let context = Rc::new(Context::default());
    let message = type_error_text(&context, "!nope");
    assert!(message.contains("use of undefined type alias '!nope'"), "{message}");
    let message = type_error_text(&context, "struct<ptr<!nope>>");
    assert!(message.contains("use of undefined type alias '!nope'"), "{message}");
}

/// The bodies in a `rec` block are types like any other, and their nesting is limited the same way.
#[cfg(feature = "std")]
#[test]
fn type_nesting_is_limited_in_a_rec_block() {
    on_a_1_mb_stack(|| {
        let body = nested_type_text("struct<", "ptr<!a>", ">", 63);
        let test = ParserTest::default();
        declared_types(&test, &format!("rec {{ type !a = {body}; }}"), &["!a"]);

        let body = nested_type_text("struct<", "ptr<!a>", ">", 64);
        let message =
            parse_error_text(&format!("rec {{ type !a = {body}; }}\nbuiltin.world {{}};"));
        assert!(message.contains("type nesting is too deep (limit 64)"), "{message}");
    });
}

/// 20,000 type declarations, each referring to the one before it, then an operation using the last,
/// parse on a 1 MB stack.
#[cfg(feature = "std")]
#[test]
fn many_type_declarations_parse() {
    on_a_1_mb_stack(|| {
        const N: usize = 20_000;
        let mut declarations = String::from("type !a0 = i32;\n");
        for i in 1..N {
            declarations.push_str(&format!("type !a{i} = ptr<!a{}>;\n", i - 1));
        }
        let test = ParserTest::default();
        let [last] = declared_types(&test, &declarations, &[&format!("!a{}", N - 1)])[..] else {
            unreachable!()
        };
        let mut ty = last;
        for _ in 1..N {
            ty = ty.pointee().expect("a pointer");
        }
        assert_eq!(ty, test.context().i32());
    });
}

/// A function `f`, built in a test of its own with the test dialect registered, with an argument of
/// each type `params` returns and, in its body, a cast of its first argument to each type `casts`
/// returns. The casts' types are written inside an attribute, `#builtin.type<..>`.
fn function_over(
    params: impl FnOnce(&Context) -> Vec<Type>,
    casts: impl FnOnce(&Context, &[Type]) -> Vec<Type>,
) -> Test {
    let mut test = Test::named("f");
    test.context_rc().get_or_register_dialect::<TestDialect>();
    let params = params(test.context());
    let casts = casts(test.context(), &params);
    test.with_function("f", &params, &[]);
    {
        let mut f = test.function_builder();
        let arg = f.entry_block().borrow().arguments()[0] as ValueRef;
        for ty in casts {
            f.builder_mut()
                .unrealized_conversion_cast(arg, ty, SourceSpan::UNKNOWN)
                .unwrap();
        }
        f.builder_mut().ret([], SourceSpan::UNKNOWN).unwrap();
    }
    test
}

/// `function` printed as the root of a print, with the declarations of its recursion groups.
fn print_root(function: UnsafeIntrusiveEntityRef<Function>) -> String {
    let flags = Default::default();
    let function = function.as_operation_ref();
    let mut printer = AsmPrinter::root(function.borrow().context_rc(), &flags);
    printer.print_operation(function.borrow());
    printer.finish().to_string()
}

/// The argument types of `function`, then the result types of the operations in its entry block.
fn types_of(function: UnsafeIntrusiveEntityRef<Function>) -> Vec<Type> {
    let function = function.borrow();
    let entry = function.entry_block();
    let entry = entry.borrow();
    let mut types = entry
        .arguments()
        .iter()
        .map(|arg| (*arg as ValueRef).borrow().ty())
        .collect::<Vec<_>>();
    for op in entry.body() {
        types.extend(op.results().all().iter().map(|result| (*result as ValueRef).borrow().ty()));
    }
    types
}

/// Prints the function of `test`, parses the text in a fresh context and prints that again, and
/// checks that the two texts are equal; then parses the text in the context of `test`, and checks
/// that it gives the very types `test` was built with. Returns the text.
fn assert_round_trips(test: &Test) -> String {
    let printed = print_root(test.function());

    let fresh = ParserTest::default();
    fresh.context().get_or_register_dialect::<TestDialect>();
    let reparsed = fresh
        .parse::<Function>("round_trip.hir", &printed)
        .unwrap_or_else(|err| panic!("{printed}\n{err:?}"));
    assert_eq!(print_root(reparsed), printed);

    let config = ParserConfig::new(test.context_rc());
    let parsed = parse::parse::<Function>(config, "round_trip.hir", printed.as_str())
        .into_result()
        .unwrap_or_else(|err| panic!("{printed}\n{err:?}"));
    assert_eq!(types_of(parsed), types_of(test.function()));
    printed
}

/// `struct <name><ptr<rec.0>, held>` as a group of one, or unnamed if `name` is `None`.
fn self_recursive(context: &Context, name: Option<&str>, held: Type) -> Type {
    let fields = [context.ptr_type(context.rec_var(0)).into(), held.into()];
    let body = match name {
        Some(name) => context
            .struct_type(crate::StructType::named(crate::interner::Symbol::intern(name), fields)),
        None => context.struct_type(crate::StructType::new(fields)),
    };
    context.rec_group(&[body]).unwrap()[0]
}

/// The `list`/`node` pair of the design.
fn list_and_node(context: &Context) -> [Type; 2] {
    let (list, node) = (context.rec_var(0), context.rec_var(1));
    let named = |name: &str, fields: [Type; 2]| {
        context.struct_type(crate::StructType::named(
            crate::interner::Symbol::intern(name),
            fields.map(Into::into),
        ))
    };
    let members = context
        .rec_group(&[
            named("list", [context.i32(), context.ptr_type(node)]),
            named("node", [context.ptr_type(list), context.ptr_type(node)]),
        ])
        .unwrap();
    [members[0], members[1]]
}

#[test]
fn a_self_recursive_struct_round_trips() {
    let test = function_over(
        |cx| [self_recursive(cx, Some("list"), cx.i32())].into(),
        |cx, params| [cx.ptr_type(params[0])].into(),
    );
    let printed = assert_round_trips(&test);
    assert!(printed.starts_with("rec {\n  type !list = struct list<ptr<!list>, i32>;\n}\n\n"));
}

#[test]
fn a_mutually_recursive_pair_round_trips() {
    let test = function_over(|cx| list_and_node(cx).into(), |_, params| [params[1]].into());
    assert!(assert_round_trips(&test).starts_with(
        "rec {\n  type !list = struct list<i32, ptr<!node>>;\n  type !node = struct \
         node<ptr<!list>, ptr<!node>>;\n}\n\n"
    ));
}

#[test]
fn an_enum_member_round_trips() {
    use crate::{EnumType, EnumVariant, interner::Symbol};

    let test = function_over(
        |cx| {
            let tree = EnumType::new(
                Some(Symbol::intern("tree")),
                cx.i8(),
                [
                    EnumVariant::new(Symbol::intern("leaf")).with_payload(cx.i32()),
                    EnumVariant::new(Symbol::intern("node"))
                        .with_payload(cx.ptr_type(cx.rec_var(0))),
                ],
            )
            .unwrap();
            let values = EnumType::new(
                None,
                cx.i8(),
                [
                    EnumVariant::new(Symbol::intern("a")).with_payload(cx.ptr_type(cx.rec_var(0))),
                    EnumVariant::new(Symbol::intern("b")).with_value(5),
                    EnumVariant::new(Symbol::intern("c")),
                ],
            )
            .unwrap();
            [cx.enum_type(tree), cx.enum_type(values)]
                .map(|body| cx.rec_group(&[body]).unwrap()[0])
                .into()
        },
        |_, _| Vec::new(),
    );
    let printed = assert_round_trips(&test);
    assert!(
        printed.contains("type !tree = enum tree<i8; leaf: i32, node: ptr<!tree>>;"),
        "{printed}"
    );
    assert!(printed.contains("type !t0 = enum<i8; a: ptr<!t0>, b = 5, c = 6>;"), "{printed}");
}

#[test]
fn a_function_type_member_round_trips() {
    let test = function_over(
        |cx| {
            let body = cx.function_type([cx.ptr_type(cx.rec_var(0))], [cx.i32()]);
            [cx.rec_group(&[body]).unwrap()[0]].into()
        },
        |_, _| Vec::new(),
    );
    let printed = assert_round_trips(&test);
    assert!(
        printed.starts_with("rec {\n  type !t0 = (ptr<!t0>) -> i32;\n}\n\n"),
        "{printed}"
    );
}

#[test]
fn a_dialect_type_member_round_trips() {
    use crate::dialects::test::TestVectorType;

    let test = function_over(
        |cx| {
            let body = cx.intern_type(TestVectorType::new(cx.ptr_type(cx.rec_var(0)), 4));
            [cx.rec_group(&[body]).unwrap()[0]].into()
        },
        |_, _| Vec::new(),
    );
    let printed = assert_round_trips(&test);
    assert!(
        printed.starts_with("rec {\n  type !t0 = !test.vector<ptr<!t0> x 4>;\n}\n\n"),
        "{printed}"
    );
}

#[test]
fn a_group_referring_to_another_round_trips() {
    let test = function_over(
        |cx| {
            let a = self_recursive(cx, Some("a"), cx.i32());
            [self_recursive(cx, Some("b"), cx.ptr_type(a))].into()
        },
        |_, _| Vec::new(),
    );
    let printed = assert_round_trips(&test);
    assert!(
        printed.starts_with(
            "rec {\n  type !a = struct a<ptr<!a>, i32>;\n}\nrec {\n  type !b = struct b<ptr<!b>, \
             ptr<!a>>;\n}\n\n"
        ),
        "{printed}"
    );
}

#[test]
fn groups_whose_structs_share_a_name_round_trip() {
    let test = function_over(
        |cx| {
            [
                self_recursive(cx, Some("list"), cx.i32()),
                self_recursive(cx, Some("list"), cx.i64()),
            ]
            .into()
        },
        |_, _| Vec::new(),
    );
    let printed = assert_round_trips(&test);
    assert!(printed.contains("type !list = struct list<ptr<!list>, i32>;"), "{printed}");
    assert!(printed.contains("type !list_0 = struct list<ptr<!list_0>, i64>;"), "{printed}");
}

#[test]
fn unnamed_and_unusually_named_members_round_trip() {
    let test = function_over(
        |cx| {
            let (first, second) = (cx.rec_var(0), cx.rec_var(1));
            let pair = cx
                .rec_group(&[
                    cx.struct_type(crate::StructType::new([cx.ptr_type(second).into()])),
                    cx.struct_type(crate::StructType::new([cx.ptr_type(first).into()])),
                ])
                .unwrap();
            [
                pair[0],
                pair[1],
                self_recursive(cx, Some("my type"), cx.i32()),
                self_recursive(cx, Some("a$b"), cx.i32()),
                self_recursive(cx, None, cx.i8()),
            ]
            .into()
        },
        |_, _| Vec::new(),
    );
    let printed = assert_round_trips(&test);
    for declaration in [
        "type !t0 = struct<ptr<!t1>>;",
        "type !t1 = struct<ptr<!t0>>;",
        "type !t2 = struct \"my type\"<ptr<!t2>, i32>;",
        "type !a$b = struct \"a$b\"<ptr<!a$b>, i32>;",
        "type !t3 = struct<ptr<!t3>, i8>;",
    ] {
        assert!(printed.contains(declaration), "{declaration}\n{printed}");
    }
}

#[test]
fn a_member_used_only_inside_an_attribute_round_trips() {
    let test = function_over(
        |cx| [cx.i32()].into(),
        |cx, _| [cx.ptr_type(self_recursive(cx, Some("list"), cx.i32()))].into(),
    );
    let printed = assert_round_trips(&test);
    assert!(printed.starts_with("rec {\n  type !list = struct list<ptr<!list>, i32>;\n}\n\n"));
    assert!(printed.contains("#builtin.type<ptr<!list>>"), "{printed}");
}

#[test]
fn the_text_of_a_value_holding_a_member_has_no_declarations() {
    use crate::{OpFoldResult, dialects::builtin::attributes::TypeAttr};

    let context = Rc::new(Context::default());
    let list = self_recursive(&context, Some("list"), context.i32());
    let attr = context.create_attribute::<TypeAttr, _>(context.ptr_type(list));
    let folded = OpFoldResult::Attribute(attr.as_attribute_ref());
    assert_eq!(format!("{folded}"), "#builtin.type<ptr<!list>>");

    let param = AbiParam::new_with_attrs(
        context.i32(),
        [crate::NamedAttribute::new("note", attr.as_attribute_ref())],
    );
    assert_eq!(format!("{param}"), "i32 { note = #builtin.type<ptr<!list>> }");
}

#[test]
fn the_declarations_of_a_print_are_lines_of_a_document() {
    let test = function_over(|cx| list_and_node(cx).into(), |_, _| Vec::new());
    let flags = Default::default();
    let function = test.function().as_operation_ref();
    let mut printer = AsmPrinter::root(function.borrow().context_rc(), &flags);
    printer.print_operation(function.borrow());
    let declarations = printer.rec_group_declarations();
    assert_eq!(
        declarations.to_string(),
        "rec {\n  type !list = struct list<i32, ptr<!node>>;\n  type !node = struct \
         node<ptr<!list>, ptr<!node>>;\n}"
    );
    // Nested in an indented document, every line is indented with it.
    let nested = crate::formatter::indent(4, crate::formatter::nl() + declarations);
    assert_eq!(
        nested.to_string(),
        "\n    rec {\n      type !list = struct list<i32, ptr<!node>>;\n      type !node = struct \
         node<ptr<!list>, ptr<!node>>;\n    }"
    );
}

/// A group of 2,000 members, member `i` being `struct<ptr<member (i + 1) % 2000>>`, prints and
/// parses on a 1 MB stack.
#[cfg(feature = "std")]
#[test]
fn a_group_of_2000_members_round_trips() {
    on_a_1_mb_stack(|| {
        const N: u32 = 2000;
        let test = function_over(
            |cx| {
                let bodies = (0..N)
                    .map(|i| {
                        let next = cx.ptr_type(cx.rec_var((i + 1) % N));
                        cx.struct_type(crate::StructType::new([next.into()]))
                    })
                    .collect::<Vec<_>>();
                [cx.rec_group(&bodies).unwrap()[0]].into()
            },
            |_, _| Vec::new(),
        );
        let printed = print_root(test.function());
        assert!(printed.contains("type !t1999 = struct<ptr<!t0>>;"));

        let fresh = ParserTest::default();
        let reparsed = fresh.parse::<Function>("large.hir", &printed).unwrap();
        assert_eq!(print_root(reparsed), printed);

        let config = ParserConfig::new(test.context_rc());
        let parsed = parse::parse::<Function>(config, "large.hir", printed.as_str())
            .into_result()
            .unwrap();
        assert_eq!(types_of(parsed), types_of(test.function()));
    });
}

#[test]
fn hand_written_aliases_print_with_the_printers_names() {
    let test = ParserTest::default();
    let source = "\
rec {
  type !n = struct node<ptr<!l>, ptr<!n>>;
  type !l = struct list<i32, ptr<!n>>;
}

rec {
  type !anon = struct<ptr<!anon>, ptr<!l>>;
}

builtin.function public extern(\"C\") @f(%0: !l, %1: !anon) {
    builtin.ret;
};";
    let function = test.parse::<Function>("hand_written.hir", source).unwrap();
    let printed = print_root(function);
    let (declarations, operation) = printed.split_once("\n\n").unwrap();
    assert_eq!(
        declarations,
        "\
rec {
  type !node = struct node<ptr<!list>, ptr<!node>>;
  type !list = struct list<i32, ptr<!node>>;
}
rec {
  type !t0 = struct<ptr<!t0>, ptr<!list>>;
}"
    );
    assert!(operation.contains(": !list, %") && operation.contains(": !t0)"), "{operation}");

    let test = ParserTest::default();
    let reparsed = test.parse::<Function>("fixed_point.hir", &printed).unwrap();
    let reprinted = print_root(reparsed);
    let test = ParserTest::default();
    let reparsed = test.parse::<Function>("fixed_point.hir", &reprinted).unwrap();
    assert_eq!(print_root(reparsed), reprinted);
    assert_eq!(reprinted.split_once("\n\n").unwrap().0, declarations);
}

fn lit(negative: bool, magnitude: u128) -> super::IntegerLiteral {
    super::IntegerLiteral {
        negative,
        magnitude,
    }
}

#[test]
fn integer_literals_convert_to_int() {
    use crate::Int;

    let context = Context::default();
    let to_int = |literal, ty| super::IntegerLiteral::to_int(literal, ty, SourceSpan::UNKNOWN);
    let i8 = context.i8();
    assert_eq!(to_int(lit(false, 255), i8).unwrap(), Int::from(-1i8));
    assert_eq!(to_int(lit(true, 1), i8).unwrap(), Int::from(255u8));
    assert_eq!(to_int(lit(true, 128), i8).unwrap(), Int::from(-128i8));
    assert!(to_int(lit(false, 256), i8).is_err());
    assert!(to_int(lit(true, 129), i8).is_err());

    let i1 = context.i1();
    assert_eq!(to_int(lit(true, 1), i1).unwrap(), Int::from(true));
    assert_eq!(to_int(lit(false, 0), i1).unwrap(), Int::from(false));
    assert_eq!(to_int(lit(false, 1), i1).unwrap(), Int::from(true));
    assert!(to_int(lit(false, 2), i1).is_err());
    assert!(to_int(lit(true, 2), i1).is_err());

    let i7 = context.int_type(7);
    assert_eq!(to_int(lit(false, 127), i7).unwrap(), Int::from_bits(7, 127));
    assert_eq!(to_int(lit(true, 64), i7).unwrap(), Int::from_bits(7, 64));
    assert!(to_int(lit(false, 128), i7).is_err());
    assert!(to_int(lit(true, 65), i7).is_err());

    let isize = context.isize();
    assert_eq!(to_int(lit(true, 1), isize).unwrap(), Int::from(-1i64));
    assert_eq!(to_int(lit(false, 1 << 63), isize).unwrap().width(), 64);
    assert!(to_int(lit(false, 1 << 64), isize).is_err());

    assert_eq!(to_int(lit(false, u128::MAX), context.i128()).unwrap(), Int::all_ones(128));

    let reason = |result: ParseResult<Int>| match result.unwrap_err() {
        crate::parse::ParserError::InvalidIntegerLiteral { reason, .. } => reason,
        other => panic!("unexpected error {other:?}"),
    };
    assert_eq!(
        reason(to_int(lit(false, 1), context.int_type(256))),
        "integer literals wider than 128 bits are not supported"
    );
    let ptr = context.opaque_ptr_type();
    assert_eq!(
        reason(to_int(lit(false, 1), ptr)),
        format!("expected an integer type, got {ptr}")
    );
    assert_eq!(reason(to_int(lit(false, 256), i8)), format!("value is out of range for {i8}"));
}

#[test]
fn int_text_round_trips_through_literals() {
    use crate::Int;

    for w in 1..=9u32 {
        for b in 0..(1u128 << w) {
            let x = Int::from_bits(w, b);
            let text = format!("{x}");
            let (negative, digits) = match text.strip_prefix('-') {
                Some(digits) => (true, digits),
                None => (false, text.as_str()),
            };
            let literal = lit(negative, digits.parse::<u128>().unwrap());
            assert_eq!(literal.to_int_of_width(w), Some(x), "{text} : i{w}");
        }
    }
    assert_eq!(lit(false, 1).to_int_of_width(0), None);
    assert_eq!(lit(false, 1).to_int_of_width(129), None);
}

#[test]
fn integer_literals_at_127_and_128_bits() {
    assert_eq!(lit(true, 1 << 127).to_int_of_width(128).map(|x| x.unsigned()), Some(1 << 127));
    assert_eq!(lit(true, (1 << 127) + 1).to_int_of_width(128), None);

    assert_eq!(
        lit(false, (1 << 127) - 1).to_int_of_width(127).map(|x| x.unsigned()),
        Some((1 << 127) - 1)
    );
    assert_eq!(lit(true, 1 << 126).to_int_of_width(127).map(|x| x.signed()), Some(-(1 << 126)));
    assert_eq!(lit(false, 1 << 127).to_int_of_width(127), None);
    assert_eq!(lit(true, (1 << 126) + 1).to_int_of_width(127), None);
}

/// Parses `text` as an attribute value expected to have type `ty`.
fn parse_attribute_text(
    context: &Rc<Context>,
    text: &str,
    ty: Type,
) -> ParseResult<crate::AttributeRef> {
    use super::{DefaultParser, Parser, ParserState, Scanner, TokenStream};

    let id = context.add_source("attr.runic", text).unwrap();
    let sources = context.sources();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
    parser.parse_attribute(ty).map(|attr| attr.into_inner())
}

fn print_attribute_text(context: &Rc<Context>, attr: &crate::AttributeRef) -> String {
    let flags = Default::default();
    let mut printer = AsmPrinter::root(context.clone(), &flags);
    printer.print_attribute_value(&*attr.borrow());
    printer.finish().to_string()
}

/// The value and type of an integer attribute.
fn int_and_type_of(attr: &crate::AttributeRef) -> (Int, Type) {
    let attr = attr.borrow();
    let attr = attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap();
    (attr.as_int(), attr.ty())
}

#[test]
fn int_attributes_round_trip_through_text() -> TestResult {
    use crate::dialects::builtin::attributes::IntAttr;

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    for (text, value, ty, printed) in [
        (
            "#builtin.int<-1 : i7>",
            Int::from_bits(7, 0x7f),
            context.int_type(7),
            "#builtin.int<-1 : i7>",
        ),
        ("#builtin.int<255 : i8>", Int::from(-1i8), context.i8(), "#builtin.int<-1 : i8>"),
        ("#builtin.int<1 : i1>", Int::from(true), context.i1(), "#builtin.int<1 : i1>"),
        (
            "#builtin.int<5 : isize>",
            Int::from(5i64),
            context.isize(),
            "#builtin.int<5 : isize>",
        ),
        (
            "#builtin.int<0x10 : i64>",
            Int::from(16i64),
            context.i64(),
            "#builtin.int<16 : i64>",
        ),
    ] {
        let attr = parse_attribute_text(&context, text, unknown)?;
        assert!(attr.borrow().is::<IntAttr>(), "{text}");
        assert_eq!(int_and_type_of(&attr), (value, ty), "{text}");
        assert_eq!(print_attribute_text(&context, &attr), printed, "{text}");
        let reparsed = parse_attribute_text(&context, printed, unknown)?;
        assert_eq!(int_and_type_of(&reparsed), (value, ty), "{printed}");
        assert_eq!(&*reparsed.borrow(), &*attr.borrow(), "{printed}");
    }

    for text in [
        "#builtin.int<128 : i7>",
        "#builtin.int<1 : i256>",
        "#builtin.int<1 : ptr>",
        "#builtin.int<1>",
    ] {
        assert!(
            parse_attribute_text(&context, text, unknown).is_err(),
            "{text} should not parse"
        );
    }
    Ok(())
}

/// A body-less attribute with a type of its own keeps it: it is read only where that type, or an
/// unknown one, is expected.
#[test]
fn a_body_less_attribute_is_read_only_with_the_type_of_its_value() -> TestResult {
    let context = Rc::new(Context::default());
    let (unknown, i8, i32, isize, ptr, i1) = (
        context.unknown_type(),
        context.i8(),
        context.i32(),
        context.isize(),
        context.opaque_ptr_type(),
        context.i1(),
    );
    for (text, ty, value, expected) in [
        ("#builtin.int", unknown, Int::zero(32), i32),
        ("#builtin.int", i32, Int::zero(32), i32),
        ("#builtin.i32", i32, Int::zero(32), i32),
        ("#builtin.bool", i1, Int::from(false), i1),
    ] {
        let attr = parse_attribute_text(&context, text, ty)?;
        assert_eq!(int_and_type_of(&attr), (value, expected), "{text} : {ty}");
        let width = expected.integer_width().unwrap();
        assert_eq!(attr.borrow().as_int().unwrap().width(), width, "{text} : {ty}");
    }
    assert_eq!(
        print_attribute_text(&context, &parse_attribute_text(&context, "#builtin.int", i32)?),
        "#builtin.int<0 : i32>"
    );

    for (text, ty, attr_ty) in [
        ("#builtin.int", i8, "i32"),
        ("#builtin.int", isize, "i32"),
        ("#builtin.int", ptr, "i32"),
        ("#builtin.i32", i8, "i32"),
        ("#builtin.bool", i32, "i1"),
        ("#builtin.i64", isize, "i64"),
    ] {
        let err = parse_attribute_text(&context, text, ty).unwrap_err();
        let crate::parse::ParserError::InvalidAttributeValue { reason, .. } = err else {
            panic!("{text} : {ty}: {err:?}");
        };
        assert_eq!(
            reason,
            format!(
                "expected a value of type {ty}, got '{}' with no value, which has type {attr_ty}",
                text.trim_start_matches('#')
            ),
        );
    }
    Ok(())
}

/// The value of an attribute, as text that is equal exactly when the values (and types) are.
fn value_text(attr: &crate::AttributeRef) -> String {
    use crate::dialects::builtin::attributes::StringAttr;

    if let Some(string) = attr.borrow().downcast_ref::<StringAttr>() {
        return format!("string {:?}", string.as_str());
    }
    let (value, ty) = int_and_type_of(attr);
    format!("{value} : {ty}")
}

/// Parses `text` as an attribute dictionary of one entry, `a`, and returns its value.
fn parse_dictionary_entry_text(
    context: &Rc<Context>,
    text: &str,
) -> ParseResult<crate::AttributeRef> {
    use super::{DefaultParser, ParsedAttrs, Parser, ParserState, Scanner, TokenStream};

    let id = context.add_source("attrs.runic", text).unwrap();
    let sources = context.sources();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
    let mut attrs = ParsedAttrs::default();
    parser.parse_attribute_dict(&mut attrs)?;
    assert_eq!(attrs.len(), 1, "{text}");
    assert_eq!(attrs[0].name.as_str(), "a", "{text}");
    Ok(attrs[0].value)
}

#[test]
fn bool_attributes_parse() -> TestResult {
    use crate::dialects::builtin::attributes::BoolAttr;

    let context = Rc::new(Context::default());
    for value in [true, false] {
        let text = format!("#builtin.bool<{value}>");
        let attr = parse_attribute_text(&context, &text, context.unknown_type())?;
        assert_eq!(**attr.borrow().downcast_ref::<BoolAttr>().unwrap(), value);
        assert_eq!(print_attribute_text(&context, &attr), text);
        let entry = parse_dictionary_entry_text(&context, &format!("{{ a = {text} }}"))?;
        assert_eq!(**entry.borrow().downcast_ref::<BoolAttr>().unwrap(), value);
    }
    for text in ["#builtin.bool<1>", "#builtin.bool<x>", "#builtin.bool<>", "#builtin.bool<true"] {
        assert!(parse_attribute_text(&context, text, context.unknown_type()).is_err(), "{text}");
    }
    Ok(())
}

#[test]
fn string_attributes_parse() -> TestResult {
    use crate::dialects::builtin::attributes::StringAttr;

    let context = Rc::new(Context::default());
    // every character `escape_default` escapes, and some it does not
    for value in [
        "a\"b\n",
        "back\\slash",
        "tab\tcr\rnl\n",
        "it's",
        "caf\u{e9} \u{1f600} \u{0}",
        "",
        "\\\"\\n",
    ] {
        let attr = context.create_attribute::<StringAttr, _>(crate::CompactString::from(value));
        let text = print_attribute_text(&context, &attr.as_attribute_ref());
        let parsed = parse_attribute_text(&context, &text, context.unknown_type())?;
        assert_eq!(parsed.borrow().downcast_ref::<StringAttr>().unwrap().as_str(), value, "{text}");
        assert_eq!(print_attribute_text(&context, &parsed), text);
    }
    let attr =
        parse_attribute_text(&context, "#builtin.string<\"a\\\"b\\n\">", context.unknown_type())?;
    assert_eq!(attr.borrow().downcast_ref::<StringAttr>().unwrap().as_str(), "a\"b\n");
    let entry = parse_dictionary_entry_text(&context, "{ a = #builtin.string<\"str\"> }")?;
    assert_eq!(entry.borrow().downcast_ref::<StringAttr>().unwrap().as_str(), "str");
    for text in ["#builtin.string<x>", "#builtin.string<1>", "#builtin.string<\"\\q\">"] {
        assert!(parse_attribute_text(&context, text, context.unknown_type()).is_err(), "{text}");
    }
    Ok(())
}

/// Every builtin integer, bool and string attribute prints, parses, and prints the same again.
#[test]
fn builtin_integer_bool_and_string_attributes_round_trip() -> TestResult {
    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    for text in [
        "#builtin.int<-1 : i7>",
        "#builtin.int<0 : i1>",
        "#builtin.int<-1 : i128>",
        "#builtin.int<5 : isize>",
        "#builtin.bool<true>",
        "#builtin.bool<false>",
        "#builtin.i8<-128>",
        "#builtin.i16<-1>",
        "#builtin.i32<7>",
        "#builtin.i64<-9223372036854775808>",
        "#builtin.i128<-170141183460469231731687303715884105728>",
        "#builtin.string<\"\">",
        "#builtin.string<\"a\\\"b\\n\">",
        "#builtin.string<\"caf\\u{e9}\">",
    ] {
        let first = parse_attribute_text(&context, text, unknown)?;
        let printed = print_attribute_text(&context, &first);
        assert_eq!(printed, text);
        let second = parse_attribute_text(&context, &printed, unknown)?;
        assert_eq!(print_attribute_text(&context, &second), printed);
        assert_eq!(value_text(&first), value_text(&second), "{text}");
    }
    Ok(())
}

/// An integer literal read where an integer type is expected is an `IntAttr` of that type.
#[test]
fn an_integer_literal_with_an_expected_type_is_an_int_attribute() -> TestResult {
    use crate::dialects::builtin::attributes::IntAttr;

    let context = Rc::new(Context::default());
    for (ty, value) in [
        (context.i1(), Int::from(true)),
        (context.i8(), Int::from(1i8)),
        (context.int_type(7), Int::from_bits(7, 1)),
        (context.i128(), Int::from(1i128)),
        (context.isize(), Int::from(1i64)),
    ] {
        let attr = parse_attribute_text(&context, "1", ty)?;
        assert!(attr.borrow().is::<IntAttr>(), "{ty}");
        assert_eq!(int_and_type_of(&attr), (value, ty), "{ty}");
    }

    let err = parse_attribute_text(&context, "1", context.int_type(256)).unwrap_err();
    let crate::parse::ParserError::InvalidIntegerLiteral { reason, .. } = err else {
        panic!("unexpected error {err:?}");
    };
    assert_eq!(reason, "integer literals wider than 128 bits are not supported");

    // `true` and `false` are still a `BoolAttr`
    let attr = parse_attribute_text(&context, "true", context.i1())?;
    assert!(attr.borrow().is::<crate::dialects::builtin::attributes::BoolAttr>());
    Ok(())
}

/// `test.constant` holds an `IntAttr` of any integer type, and its result has that type.
#[test]
fn test_constants_of_any_integer_type_round_trip() -> TestResult {
    for (literal, ty, value, printed) in [
        ("-1", "i7", Int::from_bits(7, 0x7f), "-1"),
        ("1", "i1", Int::from(true), "1"),
        ("255", "i8", Int::from(-1i8), "-1"),
        ("5", "i128", Int::from(5i128), "5"),
        ("-1", "isize", Int::from(-1i64), "-1"),
    ] {
        let test = ParserTest::default();
        test.context().get_or_register_dialect::<TestDialect>();
        let ty_text = ty;
        let ty = parse_type_text(&test.context_rc(), ty_text)?;
        let source = format!(
            "builtin.function public extern(\"C\") @f() -> {ty} {{\n    %0 = test.constant \
             {literal} : {ty};\n    builtin.ret %0 : ({ty});\n}};"
        );
        let check = |function: &UnsafeIntrusiveEntityRef<Function>| {
            let function = function.borrow();
            let body = function.body();
            let op = body.entry().front().unwrap();
            let op = op.borrow();
            let attr = op.get_attribute("value").unwrap();
            assert_eq!(int_and_type_of(&attr), (value, ty), "{literal} : {ty}");
            assert_eq!(crate::Value::ty(&*op.results()[0].borrow()), ty, "{literal} : {ty}");
        };
        let parsed = test.parse::<Function>("constant.hir", &source)?;
        check(&parsed);
        let text = format!("{}", parsed.as_operation_ref().borrow());
        assert!(text.contains(&format!("test.constant {printed} : {ty}")), "{text}");
        let fresh = ParserTest::default();
        fresh.context().get_or_register_dialect::<TestDialect>();
        let reparsed = fresh.parse::<Function>("constant-printed.hir", &text)?;
        let fresh_ty = parse_type_text(&fresh.context_rc(), ty_text)?;
        {
            let function = reparsed.borrow();
            let body = function.body();
            let op = body.entry().front().unwrap();
            let attr = op.borrow().get_attribute("value").unwrap();
            assert_eq!(int_and_type_of(&attr), (value, fresh_ty), "{text}");
        }
        assert_eq!(format!("{}", reparsed.as_operation_ref().borrow()), text);
    }
    Ok(())
}

/// A segment's offset is a `u32` held in an `I32Attr`: an offset in the upper half of the address
/// space builds, prints, and its printed attribute parses back to the same `u32`.
///
/// The whole operation does not round-trip through text yet: `#builtin.bytes<..>` has no parser.
#[test]
fn a_segment_offset_in_the_upper_half_round_trips() -> TestResult {
    use crate::dialects::builtin::ModuleBuilder;

    let test = ParserTest::default();
    let module = test.parse_any("segment.hir", "builtin.module public @m {};")?;
    let module = module.try_downcast_op::<Module>().unwrap();
    let mut builder = ModuleBuilder::new(module);
    let mut segment = builder.define_data_segment(
        0x8000_0000,
        [1u8, 2, 3, 4].as_slice(),
        true,
        SourceSpan::UNKNOWN,
    )?;
    assert_eq!(segment.borrow().offset_in_bytes(), 0x8000_0000);
    segment.borrow_mut().set_offset_in_bytes(0xffff_ffff);
    assert_eq!(segment.borrow().offset_in_bytes(), 0xffff_ffff);
    segment.borrow_mut().set_offset_in_bytes(0x8000_0000);

    let printed = format!("{}", module.as_operation_ref().borrow());
    let offset = "#builtin.i32<-2147483648>";
    assert!(printed.contains(&format!("offset = {offset}")), "{printed}");

    let context = test.context_rc();
    for text in [offset, "#builtin.i32<2147483648>"] {
        let attr = parse_attribute_text(&context, text, context.unknown_type())?;
        assert_eq!(attr.borrow().as_u32(), Some(0x8000_0000), "{text}");
        assert_eq!(print_attribute_text(&context, &attr), offset, "{text}");
    }
    Ok(())
}

/// A property given an attribute of another kind is a diagnostic, not a panic: in the generic
/// form, in an operation's custom form, and through [crate::Operation::set_property].
#[test]
fn a_property_given_an_attribute_of_another_kind_is_a_diagnostic() {
    use crate::{
        Op,
        dialects::{builtin::attributes::I32Attr, test::Constant},
    };

    let generic = "builtin.function public extern(\"C\") @f() {\n    %0 = \"test.constant\"() { \
                   value = #builtin.i32<5> } : () -> i32;\n    builtin.ret;\n};";
    let custom = "builtin.function public extern(\"C\") @f(%0: i32) -> i64 {\n    %1 = \
                  builtin.unrealized_conversion_cast %0 <{ ty = #builtin.i32<5> }>;\n    \
                  builtin.ret %1 : (i64);\n};";
    for (source, property) in [(generic, "value"), (custom, "ty")] {
        let test = ParserTest::default();
        test.context().get_or_register_dialect::<TestDialect>();
        let err = test.parse::<Function>("g.hir", source).map(|_| ()).unwrap_err().to_string();
        assert!(
            err.contains("could not convert attribute of type 'builtin.i32' to '")
                && err.contains(&format!("for property '{property}'")),
            "{err}"
        );
    }

    let mut test = Test::new("f", |cx| ([cx.i32()], [cx.i32()]));
    let context = test.context_rc();
    let mut constant = {
        let mut builder = test.function_builder();
        let create = builder.builder_mut().create::<Constant, _>(SourceSpan::UNKNOWN);
        create(Int::from(1i32)).unwrap()
    };
    let attr = context.create_attribute::<I32Attr, _>(5i32).as_attribute_ref();
    let err = constant
        .borrow_mut()
        .as_operation_mut()
        .set_property("value", attr)
        .unwrap_err()
        .to_string();
    assert!(err.contains("could not convert attribute of type 'builtin.i32'"), "{err}");
    assert_eq!(constant.borrow().value().as_int(), Int::from(1i32));
}

/// 0, 1, 2, the signed minimum and its neighbours, all-ones and its neighbour, at `w` bits.
fn edge_bit_patterns(w: u32) -> Vec<u128> {
    let mask = Int::all_ones(w).unsigned();
    let min = 1u128 << (w - 1);
    let mut patterns = alloc::vec![
        0,
        1 & mask,
        2 & mask,
        min,
        (min + 1) & mask,
        min.wrapping_sub(1) & mask,
        mask,
        mask.wrapping_sub(1) & mask,
    ];
    patterns.sort();
    patterns.dedup();
    patterns
}

/// Every pattern of `w` bits up to 10 bits, and the edge patterns of wider integers.
fn bit_patterns(w: u32) -> Vec<u128> {
    if w <= 10 {
        (0..(1u128 << w)).collect()
    } else {
        edge_bit_patterns(w)
    }
}

/// `#builtin.int<..>`, a bare literal read with an expected type, and its hex spelling all
/// round-trip every pattern of the narrow widths and the edge patterns of the wide ones.
#[test]
fn int_attributes_round_trip_every_pattern() -> TestResult {
    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    for w in [
        1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 63, 64, 65, 126, 127, 128,
    ] {
        let ty = context.int_type(w);
        for bits in bit_patterns(w) {
            let x = Int::from_bits(w, bits);
            let attr = context.int_attr(x, ty).as_attribute_ref();
            let text = print_attribute_text(&context, &attr);
            let back = parse_attribute_text(&context, &text, unknown)?;
            assert_eq!(int_and_type_of(&back), (x, ty), "{text}");
            let bare = format!("{x}");
            let back = parse_attribute_text(&context, &bare, ty)?;
            assert_eq!(int_and_type_of(&back), (x, ty), "{bare} : {ty}");
            let hex = format!("{x:#x}");
            let back = parse_attribute_text(&context, &hex, ty)?;
            assert_eq!(int_and_type_of(&back), (x, ty), "{hex} : {ty}");
        }
    }
    for bits in edge_bit_patterns(64) {
        let x = Int::from_bits(64, bits);
        let attr = context.int_attr(x, context.isize()).as_attribute_ref();
        let text = print_attribute_text(&context, &attr);
        let back = parse_attribute_text(&context, &text, unknown)?;
        assert_eq!(int_and_type_of(&back), (x, context.isize()), "{text}");
    }
    Ok(())
}

/// The per-width integer attributes round-trip: `I8Attr` every value, the others at the edges.
#[test]
fn per_width_integer_attributes_round_trip() -> TestResult {
    use crate::dialects::builtin::attributes::{I8Attr, I32Attr, I64Attr, I128Attr};

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    let mut attrs = Vec::new();
    attrs.extend(
        (i8::MIN..=i8::MAX).map(|v| context.create_attribute::<I8Attr, _>(v).as_attribute_ref()),
    );
    attrs.extend(
        [i32::MIN, -1, 0, 1, i32::MAX]
            .map(|v| context.create_attribute::<I32Attr, _>(v).as_attribute_ref()),
    );
    attrs.extend(
        [i64::MIN, -1, 0, 1, i64::MAX]
            .map(|v| context.create_attribute::<I64Attr, _>(v).as_attribute_ref()),
    );
    attrs.extend(
        [i128::MIN, -1, 0, 1, i128::MAX]
            .map(|v| context.create_attribute::<I128Attr, _>(v).as_attribute_ref()),
    );
    for attr in attrs {
        let text = print_attribute_text(&context, &attr);
        let back = parse_attribute_text(&context, &text, unknown)?;
        assert_eq!(&*back.borrow(), &*attr.borrow(), "{text}");
    }
    Ok(())
}

/// `test.constant` and `builtin.ret_imm` round-trip every pattern of widths up to 4 and the edge
/// patterns of wider ones through the text of a whole function.
#[test]
fn constant_and_ret_imm_ops_round_trip_every_pattern() -> TestResult {
    for w in [1u32, 2, 3, 4, 7, 8, 64, 127, 128] {
        let patterns = if w <= 4 {
            bit_patterns(w)
        } else {
            edge_bit_patterns(w)
        };
        for bits in patterns {
            let x = Int::from_bits(w, bits);
            let source = format!(
                "builtin.function public extern(\"C\") @f() -> i{w} {{\n    %0 = test.constant \
                 {x} : i{w};\n    builtin.ret_imm {x} : i{w};\n}};"
            );
            let test = ParserTest::default();
            test.context().get_or_register_dialect::<TestDialect>();
            let parsed = test.parse::<Function>("c.hir", &source)?;
            let values = |f: &UnsafeIntrusiveEntityRef<Function>| {
                let f = f.borrow();
                let body = f.body();
                let entry = body.entry();
                let c = entry.front().unwrap();
                let c = c.borrow().get_attribute("value").unwrap();
                let r = entry.terminator().unwrap();
                let r = r.borrow().get_attribute("value").unwrap();
                (c.borrow().as_int().unwrap(), r.borrow().as_int().unwrap())
            };
            assert_eq!(values(&parsed), (x, x), "{source}");
            let text = format!("{}", parsed.as_operation_ref().borrow());
            let fresh = ParserTest::default();
            fresh.context().get_or_register_dialect::<TestDialect>();
            let reparsed = fresh.parse::<Function>("c2.hir", &text)?;
            assert_eq!(values(&reparsed), (x, x), "{text}");
            assert_eq!(format!("{}", reparsed.as_operation_ref().borrow()), text);
        }
    }
    Ok(())
}

/// Enum discriminants round-trip at the edge patterns of their width, `isize` included.
#[test]
fn enum_discriminants_round_trip_at_the_edges() -> TestResult {
    let context = Rc::new(Context::default());
    let cases = [1u32, 2, 7, 8, 64, 127, 128]
        .into_iter()
        .flat_map(|w| edge_bit_patterns(w).into_iter().map(move |bits| (format!("i{w}"), w, bits)))
        .chain(edge_bit_patterns(64).into_iter().map(|bits| ("isize".to_string(), 64, bits)));
    for (ty, w, bits) in cases {
        let x = Int::from_bits(w, bits);
        let text = format!("enum<{ty}; z = {x}>");
        let parsed = parse_type_text(&context, &text)?;
        let printed = format!("{parsed}");
        assert_eq!(parsed, parse_type_text(&context, &printed)?, "{text} -> {printed}");
    }
    Ok(())
}

/// A data segment layout orders and checks segments as unsigned offsets on both sides of 2^31.
#[test]
fn segment_layout_treats_offsets_as_unsigned() -> TestResult {
    use crate::dialects::builtin::{DataSegmentError, DataSegmentLayout, ModuleBuilder};

    let test = ParserTest::default();
    let module = test.parse_any("segment.hir", "builtin.module public @m {};")?;
    let module = module.try_downcast_op::<Module>().unwrap();
    let mut builder = ModuleBuilder::new(module);
    let mut segment = |offset: u32, len: usize| {
        builder
            .define_data_segment(
                offset,
                alloc::vec![1u8; len].as_slice(),
                true,
                SourceSpan::UNKNOWN,
            )
            .unwrap()
    };
    let (hi, lo, top) = (segment(0x8000_0000, 4), segment(0x10, 4), segment(0xffff_fffc, 4));
    let (overlap, past) = (segment(0x8000_0002, 4), segment(0xffff_fffe, 4));
    for order in [[hi, lo, top], [top, hi, lo], [lo, top, hi]] {
        let mut layout = DataSegmentLayout::default();
        for s in order {
            layout.insert(s).unwrap();
        }
        let offsets: Vec<u32> = layout.iter().map(|s| s.borrow().offset_in_bytes()).collect();
        assert_eq!(offsets, alloc::vec![0x10, 0x8000_0000, 0xffff_fffc]);
        assert!(matches!(
            layout.insert(overlap),
            Err(DataSegmentError::OverlappingSegments { .. })
        ));
        assert!(matches!(layout.insert(past), Err(DataSegmentError::OutOfBounds { .. })));
        assert_eq!(layout.next_available_offset(), None);
    }
    Ok(())
}

/// The verifier rejects an integer constant whose value is not as wide as its type, and a
/// `test.constant` whose result type is not its value's type, however they were made.
#[test]
fn the_verifier_rejects_an_inconsistent_int_attribute() {
    use crate::{Op, dialects::test::Constant, testing::label_messages};

    let message = |err: Report| label_messages(err.as_diagnostic()).join("\n");

    let mut test = Test::new("f", |cx| ([cx.i32()], [cx.i32()]));
    let context = test.context_rc();
    let mut constant = {
        let mut builder = test.function_builder();
        let create = builder.builder_mut().create::<Constant, _>(SourceSpan::UNKNOWN);
        create(Int::from(1i32)).unwrap()
    };
    assert!(constant.borrow().as_operation().verify().is_ok());

    // the generated setter re-infers the attribute's type and leaves the result alone
    constant.borrow_mut().set_value(Int::from(1i64 << 40));
    let err = message(constant.borrow().as_operation().verify().unwrap_err());
    assert!(err.contains("the result of 'test.constant' has type `i32`"), "{err}");
    constant.borrow_mut().set_value(Int::from(1i32));
    assert!(constant.borrow().as_operation().verify().is_ok());

    // the generated value accessor changes the width under the attribute's type
    *constant.borrow_mut().get_value_mut() = Int::from(-1i8);
    let err = message(constant.borrow().as_operation().verify().unwrap_err());
    assert!(
        err.contains(
            "property 'value' of 'test.constant' is an integer constant of type `i32` holding a \
             8-bit value, but a constant of that type is 32 bits wide"
        ),
        "{err}"
    );
    *constant.borrow_mut().get_value_mut() = Int::from(1i32);
    assert!(constant.borrow().as_operation().verify().is_ok());

    // an unchecked constructor, as a discardable attribute
    let bad = context
        .create_attribute_with_type::<crate::dialects::builtin::attributes::IntAttr, _>(
            Int::from(-1i32),
            context.i64(),
        )
        .as_attribute_ref();
    constant.borrow_mut().as_operation_mut().set_attribute("extra", bad);
    let err = message(constant.borrow().as_operation().verify().unwrap_err());
    assert!(
        err.contains(
            "attribute 'extra' of 'test.constant' is an integer constant of type `i64` holding a \
             32-bit value"
        ),
        "{err}"
    );

    // a type that no integer constant has
    let bad = context
        .create_attribute_with_type::<crate::dialects::builtin::attributes::IntAttr, _>(
            Int::from(1i32),
            context.ptr_type(context.i32()),
        )
        .as_attribute_ref();
    constant.borrow_mut().as_operation_mut().set_attribute("extra", bad);
    let err = message(constant.borrow().as_operation().verify().unwrap_err());
    assert!(err.contains("but no integer constant has that type"), "{err}");
}

/// A well-formed `isize` constant verifies: its attribute holds a 64-bit value.
#[test]
fn a_well_formed_isize_constant_verifies() -> TestResult {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    let source = "builtin.function public extern(\"C\") @f() -> isize {\n    %0 = test.constant \
                  -1 : isize;\n    builtin.ret %0 : (isize);\n};";
    let function = test.parse::<Function>("isize.hir", source)?;
    let function = function.borrow();
    let body = function.body();
    let op = body.entry().front().unwrap();
    op.borrow().verify()?;
    Ok(())
}

/// `as_bool` reads any 1-bit integer attribute, an `i1` constant as well as a `BoolAttr`, and
/// nothing wider.
#[test]
fn as_bool_reads_a_one_bit_integer_attribute() {
    use crate::dialects::builtin::attributes::BoolAttr;

    let context = Rc::new(Context::default());
    let one = context.int_attr(Int::from(true), context.i1()).as_attribute_ref();
    assert_eq!(one.borrow().as_bool(), Some(true));
    let parsed = parse_attribute_text(&context, "0", context.i1()).unwrap();
    assert_eq!(parsed.borrow().as_bool(), Some(false));
    let tru = context.create_attribute::<BoolAttr, _>(true).as_attribute_ref();
    assert_eq!(tru.borrow().as_bool(), Some(true));
    let wide = context.int_attr(Int::from(1i32), context.i32()).as_attribute_ref();
    assert_eq!(wide.borrow().as_bool(), None);
    let narrow = context.int_attr(Int::from(1i8), context.i8()).as_attribute_ref();
    assert_eq!(narrow.borrow().as_bool(), None);
}

/// An attribute read where a type is expected has that type whatever its form: written with a
/// body, bare, or without a value (whose more specific message is kept).
#[test]
fn an_attribute_with_a_body_is_held_to_the_expected_type() {
    let context = Rc::new(Context::default());
    for (text, ty, attr_ty) in [
        ("#builtin.i32<5>", context.i8(), "i32"),
        ("#builtin.int<5 : i32>", context.i8(), "i32"),
        ("#builtin.int<5 : i32>", context.opaque_ptr_type(), "i32"),
        ("#builtin.i64<0>", context.isize(), "i64"),
        ("#builtin.bool<true>", context.i32(), "i1"),
        ("true", context.i32(), "i1"),
        ("true", context.opaque_ptr_type(), "i1"),
    ] {
        let err = parse_attribute_text(&context, text, ty).unwrap_err();
        let crate::parse::ParserError::InvalidAttributeValue { reason, .. } = err else {
            panic!("{text} : {ty}: {err:?}");
        };
        assert_eq!(reason, format!("expected a value of type {ty}, got one of type {attr_ty}"));
    }
    // and the forms that keep working
    for (text, ty) in [
        ("#builtin.i32<5>", context.i32()),
        ("#builtin.int<5 : isize>", context.isize()),
        ("true", context.i1()),
        ("#builtin.bool<true>", context.i1()),
        ("#builtin.bool<true>", context.unknown_type()),
        ("#builtin.string", context.i32()),
        ("\"str\"", context.i32()),
    ] {
        assert!(parse_attribute_text(&context, text, ty).is_ok(), "{text} read as {ty}");
    }
}

/// The strings a quoted position must carry through print, parse, print: a quote, a backslash,
/// a newline, a non-ASCII character, an astral character, and the escapes themselves written out.
const ROUND_TRIP_STRINGS: &[&str] = &["a\"b", "a\\b", "a\nb", "\u{e9}", "\u{1f600}", "\\\"\\\\n"];

/// A calling-convention name survives print, parse, print.
#[test]
fn a_calling_convention_name_round_trips() -> TestResult {
    for &name in ROUND_TRIP_STRINGS {
        let test = ParserTest::default();
        let context = test.context_rc();
        let signature = Signature::with_convention(
            &context,
            CallConv::new(name),
            [context.i32()],
            [context.i32()],
        );
        let mut source = String::from("builtin.function public extern(");
        source.push_str(&format!("\"{}\"", name.escape_default()));
        source.push_str(") @f(%a: i32) -> i32 {\n    ret %a : (i32);\n};");

        let first = test.parse::<Function>("cc_first.hir", &source)?;
        assert_eq!(first.borrow().get_signature().calling_convention().as_str(), name, "{name:?}");
        assert_eq!(&*first.borrow().get_signature(), &signature, "{name:?}");
        let printed = format!("{}", first.as_operation_ref().borrow());

        let second = test.parse::<Function>("cc_second.hir", &printed)?;
        assert_eq!(&*second.borrow().get_signature(), &signature, "{name:?}");
        // value ids count up across a context, so compare the text without them
        let without_ids = |text: &str| {
            let mut out = String::new();
            let mut chars = text.chars().peekable();
            while let Some(c) = chars.next() {
                out.push(c);
                if c == '%' {
                    while chars.next_if(char::is_ascii_digit).is_some() {}
                }
            }
            out
        };
        let printed_again = format!("{}", second.as_operation_ref().borrow());
        assert_eq!(without_ids(&printed_again), without_ids(&printed), "{name:?}");
        let third = test.parse::<Function>("cc_third.hir", &printed_again)?;
        assert_eq!(&*third.borrow().get_signature(), &signature, "{name:?}");
    }
    Ok(())
}

/// A calling-convention name in `#builtin.signature<..>` survives too.
#[test]
fn a_signature_attribute_calling_convention_round_trips() -> TestResult {
    use crate::dialects::builtin::attributes::SignatureAttr;

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    for &name in ROUND_TRIP_STRINGS {
        let text =
            format!("#builtin.signature<extern(\"{}\") (i32) -> i32>", name.escape_default());
        let first = parse_attribute_text(&context, &text, unknown)?;
        let printed = print_attribute_text(&context, &first);
        let second = parse_attribute_text(&context, &printed, unknown)?;
        for attr in [&first, &second] {
            let attr = attr.borrow();
            let signature = attr.downcast_ref::<SignatureAttr>().unwrap();
            assert_eq!(signature.calling_convention().as_str(), name, "{printed}");
        }
        assert_eq!(print_attribute_text(&context, &second), printed, "{name:?}");
        assert!(printed.contains(&format!("\"{}\"", name.escape_default())), "{printed}");
    }
    Ok(())
}

/// A bare string attribute is unescaped like `#builtin.string<..>`.
#[test]
fn a_bare_string_attribute_is_unescaped() -> TestResult {
    use crate::dialects::builtin::attributes::StringAttr;

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    let bare = parse_attribute_text(&context, "\"a\\\"b\\n\"", unknown)?;
    let full = parse_attribute_text(&context, "#builtin.string<\"a\\\"b\\n\">", unknown)?;
    assert_eq!(bare.borrow().downcast_ref::<StringAttr>().unwrap().as_str(), "a\"b\n");
    assert_eq!(&*bare.borrow(), &*full.borrow());

    for &value in ROUND_TRIP_STRINGS {
        let attr = context.create_attribute::<StringAttr, _>(crate::CompactString::from(value));
        let printed = print_attribute_text(&context, &attr.as_attribute_ref());
        let bare = format!("\"{}\"", value.escape_default());
        let parsed = parse_attribute_text(&context, &bare, unknown)?;
        assert_eq!(parsed.borrow().downcast_ref::<StringAttr>().unwrap().as_str(), value);
        assert_eq!(print_attribute_text(&context, &parsed), printed, "{value:?}");
    }
    Ok(())
}

/// A string field of a debug-info attribute is stable under print and parse.
#[test]
fn a_debug_info_string_field_round_trips() -> TestResult {
    use crate::dialects::debuginfo::attributes::{Variable, VariableAttr};

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    for &value in ROUND_TRIP_STRINGS {
        let variable = Variable::new(value.into(), value.into(), 3, None);
        let attr = context.create_attribute::<VariableAttr, _>(variable.clone());
        let first = print_attribute_text(&context, &attr.as_attribute_ref());
        let parsed = parse_attribute_text(&context, &first, unknown)?;
        assert_eq!(
            **parsed.borrow().downcast_ref::<VariableAttr>().unwrap(),
            variable,
            "{value:?}"
        );
        let second = print_attribute_text(&context, &parsed);
        let reparsed = parse_attribute_text(&context, &second, unknown)?;
        let third = print_attribute_text(&context, &reparsed);
        assert_eq!(first, second, "{value:?}");
        assert_eq!(second, third, "{value:?}");
    }
    Ok(())
}

/// A file name in a location survives print and parse.
#[test]
fn a_location_file_name_round_trips() -> TestResult {
    use crate::dialects::builtin::attributes::LocationAttr;

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    for &file in ROUND_TRIP_STRINGS {
        let location = crate::Location::FileLineCol {
            file: file.into(),
            line: crate::parse::LineNumber::new(1).unwrap(),
            column: crate::parse::ColumnNumber::new(2).unwrap(),
        };
        let text = format!("loc({location})");
        let parsed = parse_attribute_text(&context, &text, unknown)?;
        let parsed = parsed.borrow();
        let crate::Location::FileLineCol {
            file: parsed_file, ..
        } = **parsed.downcast_ref::<LocationAttr>().unwrap()
        else {
            panic!("{text}");
        };
        assert_eq!(parsed_file.as_str(), file, "{text}");
    }
    Ok(())
}

/// Debug-info strings in the positions not tested above read back what the printer escapes: a
/// subprogram's parameter names, the names in an expression, and a compile unit (whose printer
/// always writes `optimized`, so the comma its parser wants after `file` is always there).
#[test]
fn debug_info_strings_that_need_escaping_round_trip() {
    use crate::dialects::debuginfo::attributes::*;

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    for &value in ROUND_TRIP_STRINGS {
        let subprogram = Subprogram::new(value.into(), value.into(), 1, None)
            .with_param_names([value.into(), "plain".into()]);
        let attr = context.create_attribute::<SubprogramAttr, _>(subprogram.clone());
        let first = print_attribute_text(&context, &attr.as_attribute_ref());
        let parsed = parse_attribute_text(&context, &first, unknown).unwrap();
        assert_eq!(**parsed.borrow().downcast_ref::<SubprogramAttr>().unwrap(), subprogram);
        assert_eq!(print_attribute_text(&context, &parsed), first);

        let unit = CompileUnit::new(value.into(), value.into());
        let attr = context.create_attribute::<CompileUnitAttr, _>(unit.clone());
        let first = print_attribute_text(&context, &attr.as_attribute_ref());
        let parsed = parse_attribute_text(&context, &first, unknown).unwrap();
        assert_eq!(**parsed.borrow().downcast_ref::<CompileUnitAttr>().unwrap(), unit);

        let expression = Expression::with_ops(alloc::vec![
            ExpressionOp::GlobalSlot(value.into()),
            ExpressionOp::FrameBase {
                base: FrameBase::GlobalSlot(value.into()),
                byte_offset: -8,
            },
        ]);
        let attr = context.create_attribute::<ExpressionAttr, _>(expression.clone());
        let first = print_attribute_text(&context, &attr.as_attribute_ref());
        let parsed = parse_attribute_text(&context, &first, unknown)
            .unwrap_or_else(|err| panic!("{first}: {err:?}"));
        assert_eq!(**parsed.borrow().downcast_ref::<ExpressionAttr>().unwrap(), expression);
        assert_eq!(print_attribute_text(&context, &parsed), first);
    }

    // the minimal compile unit the printer writes, and the hand-written one it never writes
    let unit = CompileUnit::new("rust".into(), "a.rs".into());
    let attr = context.create_attribute::<CompileUnitAttr, _>(unit);
    assert_eq!(
        print_attribute_text(&context, &attr.as_attribute_ref()),
        "#di.compile_unit<{ language = \"rust\", file = \"a.rs\", optimized = false }>"
    );
    assert!(
        parse_attribute_text(
            &context,
            "#di.compile_unit<{ language = \"rust\", file = \"a.rs\" }>",
            unknown
        )
        .is_err()
    );
}

/// An invalid escape in a quoted string is an `InvalidString` error, wherever the string is read.
#[test]
fn an_invalid_escape_in_a_string_is_an_error() {
    use super::{DefaultParser, Parser, ParserState, Scanner, TokenStream};

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    let err = parse_attribute_text(&context, "\"\\q\"", unknown).unwrap_err();
    assert!(matches!(err, crate::parse::ParserError::InvalidString { .. }), "{err:?}");

    let id = context.add_source("string.runic", "\"\\q\"").unwrap();
    let sources = context.sources();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
    let err = parser.parse_string().unwrap_err();
    assert!(matches!(err, crate::parse::ParserError::InvalidString { .. }), "{err:?}");

    let id = context.add_source("string2.runic", "\"\\q\"").unwrap();
    let sources = context.sources();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
    let err = parser.parse_optional_string().unwrap_err();
    assert!(matches!(err, crate::parse::ParserError::InvalidString { .. }), "{err:?}");

    let err = parse_attribute_text(&context, "loc(\"\\q\":1:1)", unknown).unwrap_err();
    assert!(matches!(err, crate::parse::ParserError::InvalidString { .. }), "{err:?}");
}

/// Parses `text` as an attribute dictionary and returns its entries.
fn parse_dictionary_text(
    context: &Rc<Context>,
    text: &str,
) -> ParseResult<Vec<crate::NamedAttribute>> {
    use super::{DefaultParser, ParsedAttrs, Parser, ParserState, Scanner, TokenStream};

    let id = context.add_source("attrs.runic", text).unwrap();
    let sources = context.sources();
    let scanner = Scanner::new(sources.get(id).unwrap().text.unwrap());
    let token_stream = TokenStream::new(id, scanner);
    let mut parser =
        DefaultParser::new(ParserState::new(ParserConfig::new(context.clone()), token_stream));
    let mut attrs = ParsedAttrs::default();
    parser.parse_attribute_dict(&mut attrs)?;
    Ok(attrs.iter().cloned().collect())
}

fn print_dictionary_text(context: &Rc<Context>, attrs: &[crate::NamedAttribute]) -> String {
    let flags = Default::default();
    let mut printer = AsmPrinter::root(context.clone(), &flags);
    printer.print_attribute_dictionary(attrs.iter().cloned());
    printer.finish().to_string()
}

/// An attribute dictionary key that is not a bare identifier is written quoted and escaped, and
/// read back as the same symbol.
#[test]
fn a_dictionary_key_that_is_not_bare_round_trips() -> TestResult {
    let context = Rc::new(Context::default());
    let keys = ["a b", "a\"b", "\u{e9}", "true", "loc", "a\\b", "a\nb", ""];
    let text = format!(
        "{{ {} }}",
        keys.iter()
            .map(|key| format!("\"{}\" = true", key.escape_default()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let parsed = parse_dictionary_text(&context, &text)?;
    assert_eq!(parsed.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), keys);

    let printed = print_dictionary_text(&context, &parsed);
    let reparsed = parse_dictionary_text(&context, &printed)?;
    assert_eq!(reparsed.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), keys, "{printed}");
    assert_eq!(print_dictionary_text(&context, &reparsed), printed);

    // A key that is bare stays bare, and a key that is not is quoted
    let bare = parse_dictionary_text(&context, "{ plain = true, \"quoted\" = false }")?;
    assert_eq!(
        print_dictionary_text(&context, &bare),
        "{ plain = #builtin.bool<true>, quoted = #builtin.bool<false> }"
    );
    assert!(printed.contains("\"a b\" = "), "{printed}");
    assert!(printed.contains("\"a\\\"b\" = "), "{printed}");

    // A type keyword is a bare key
    let keywords = parse_dictionary_text(&context, "{ struct = true, i32 = true, ptr = true }")?;
    let printed = print_dictionary_text(&context, &keywords);
    for key in ["struct", "i32", "ptr"] {
        assert!(printed.contains(&format!("{key} = ")), "{printed}");
    }
    Ok(())
}

#[test]
fn an_invalid_escape_in_a_dictionary_key_is_an_error() {
    let context = Rc::new(Context::default());
    let err = parse_dictionary_text(&context, "{ \"a\\qb\" = true }").unwrap_err();
    assert!(matches!(err, crate::parse::ParserError::InvalidString { .. }), "{err:?}");
}

/// `text` with the numbers of its value ids taken out: value ids count up across a context.
fn without_value_ids(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '%' {
            while chars.next_if(char::is_ascii_digit).is_some() {}
        }
    }
    out
}

/// Parses `source` as a function in the context of `test`, checks that it prints, as the root of
/// a print, as `expected` (value ids aside), then parses the print and checks that it has an equal
/// signature and prints the same again. Returns the function parsed from `source`.
fn assert_function_round_trips(
    test: &ParserTest,
    source: &str,
    expected: &str,
) -> UnsafeIntrusiveEntityRef<Function> {
    let first = test
        .parse::<Function>("signature_first.hir", source)
        .unwrap_or_else(|err| panic!("{source}\n{err:?}"));
    let printed = print_root(first);
    assert_eq!(without_value_ids(&printed), without_value_ids(expected));

    let second = test
        .parse::<Function>("signature_second.hir", &printed)
        .unwrap_or_else(|err| panic!("{printed}\n{err:?}"));
    assert_eq!(*second.borrow().get_signature(), *first.borrow().get_signature(), "{printed}");
    assert_eq!(without_value_ids(&print_root(second)), without_value_ids(&printed));
    first
}

/// The builtin markers print without a body, and parse back through their default value.
#[test]
fn marker_attributes_print_without_a_body_and_parse_back() -> TestResult {
    use crate::dialects::builtin::attributes::{SextAttr, SretAttr, ZextAttr};

    let context = Rc::new(Context::default());
    let text = "{ extend = #builtin.zext, other = #builtin.sext, sret = #builtin.sret }";
    let parsed = parse_dictionary_text(&context, text)?;
    assert!(parsed[0].value.borrow().is::<ZextAttr>());
    assert!(parsed[1].value.borrow().is::<SextAttr>());
    assert!(parsed[2].value.borrow().is::<SretAttr>());
    assert_eq!(print_dictionary_text(&context, &parsed), text);
    Ok(())
}

/// Attributes on a parameter and on a result of a definition survive print and parse.
#[test]
fn a_definition_with_parameter_and_result_attributes_round_trips() {
    use crate::dialects::builtin::attributes::ArgumentExtension;

    let test = ParserTest::default();
    let source = "\
builtin.function public extern(\"C\") @f(%0: i8 { extend = #builtin.zext }, %1: ptr<i32> { sret = \
                  #builtin.sret }) -> (i32 { extend = #builtin.sext }) {
    builtin.ret_imm 0 : i32;
};";
    let function = assert_function_round_trips(&test, source, source);
    let function = function.borrow();
    let signature = function.get_signature();
    assert_eq!(signature.params()[0].extension(), ArgumentExtension::Zext);
    assert!(!signature.params()[0].is_sret_param());
    assert!(signature.params()[1].is_sret_param());
    assert_eq!(signature.params()[1].extension(), ArgumentExtension::None);
    assert_eq!(signature.results()[0].extension(), ArgumentExtension::Sext);
}

/// A definition without attributes prints as it always has: no dictionaries, a bare result.
#[test]
fn a_definition_without_attributes_prints_as_before() {
    let test = ParserTest::default();
    let source = "\
builtin.function public extern(\"C\") @f(%0: i32, %1: i64) -> i32 {
    builtin.ret %0 : (i32);
};";
    let function = assert_function_round_trips(&test, source, source);
    let function = function.borrow();
    let signature = function.get_signature();
    assert!(signature.params().iter().chain(signature.results()).all(|p| p.attrs.is_empty()));
}

/// A parameter attribute that is not a builtin marker is kept like any other.
#[test]
fn a_parameter_attribute_that_is_not_a_marker_round_trips() {
    let test = ParserTest::default();
    let source = "\
builtin.function public extern(\"C\") @f(%0: i32 { align = #builtin.i32<8> }) {
    builtin.ret;
};";
    let function = assert_function_round_trips(&test, source, source);
    let function = function.borrow();
    let signature = function.get_signature();
    let align = signature.params()[0].attrs.find(&crate::interner::Symbol::intern("align"));
    let align = align.get().expect("an `align` attribute");
    assert_eq!(align.value.borrow().as_int(), Some(Int::from(8i32)));
}

/// A function declared with a list of types has no body, one empty region, and verifies.
#[test]
fn a_declaration_round_trips() {
    use crate::{Op, dialects::builtin::attributes::ArgumentExtension};

    let test = ParserTest::default();
    let source =
        "builtin.function public extern(\"C\") @f(ptr<i32>, i8 { extend = #builtin.zext }) -> i32;";
    let function = assert_function_round_trips(&test, source, source);
    let function = function.borrow();
    assert!(function.is_declaration());
    assert_eq!(function.as_operation().num_regions(), 1);
    assert!(function.body().is_empty());
    function.as_operation().verify().expect("a declaration verifies");
    let signature = function.get_signature();
    assert_eq!(signature.params().len(), 2);
    assert_eq!(signature.params()[0].ty, test.context().ptr_type(test.context().i32()));
    assert_eq!(signature.params()[1].extension(), ArgumentExtension::Zext);
    assert_eq!(signature.results()[0].ty, test.context().i32());
}

/// A declaration without parameters, and one whose parameter and result are function types.
#[test]
fn declarations_without_parameters_or_of_function_types_round_trip() {
    let test = ParserTest::default();
    for source in [
        "builtin.function public extern(\"C\") @f();",
        "builtin.function public extern(\"C\") @f() -> i32;",
        "builtin.function public extern(\"C\") @f((i32) -> i32) -> ((i32) -> i32);",
    ] {
        let function = assert_function_round_trips(&test, source, source);
        assert!(function.borrow().is_declaration(), "{source}");
    }
    let function = test
        .parse::<Function>("fn_type.hir", "builtin.function public extern(\"C\") @f((i32) -> i32);")
        .unwrap();
    let function = function.borrow();
    let signature = function.get_signature();
    assert_eq!(signature.params().len(), 1);
    assert!(signature.params()[0].ty.is_function());
}

/// A declaration whose signature uses a recursion-group member writes the member's alias, and
/// declares its group in front of the print.
#[test]
fn a_declaration_using_a_recursion_group_member_round_trips() {
    let test = ParserTest::default();
    let source = "\
rec { type !list = struct list<i32, ptr<!list>>; }
builtin.function public extern(\"C\") @f(ptr<!list>) -> ptr<!list>;";
    let expected = "\
rec {
  type !list = struct list<i32, ptr<!list>>;
}

builtin.function public extern(\"C\") @f(ptr<!list>) -> ptr<!list>;";
    let function = assert_function_round_trips(&test, source, expected);
    assert!(function.borrow().is_declaration());
}

/// A lone `never` result is declared, printed and read back, on a definition and on a declaration.
#[test]
fn a_never_result_round_trips() {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    for source in [
        "builtin.function public extern(\"C\") @f(%0: i32) -> never {\n    test.ret : ();\n};",
        "builtin.function public extern(\"C\") @f(i32) -> never;",
    ] {
        let function = assert_function_round_trips(&test, source, source);
        let function = function.borrow();
        let signature = function.get_signature();
        assert!(signature.never_returns(), "{source}");
        assert!(signature.results().is_empty(), "{source}");
        assert_eq!(signature.declared_results().len(), 1, "{source}");
    }
}

/// `#builtin.signature<..>` reads and writes parameter and result attributes and a `never` result.
#[test]
fn signature_attributes_round_trip() -> TestResult {
    use crate::dialects::builtin::attributes::{ArgumentExtension, SignatureAttr};

    let context = Rc::new(Context::default());
    let unknown = context.unknown_type();
    let texts = [
        "#builtin.signature<extern(\"C\") (i8 { extend = #builtin.zext }) -> (i32 { extend = \
         #builtin.sext })>",
        "#builtin.signature<extern(\"fast\") () -> never>",
        "#builtin.signature<extern(\"C\") (i32, i64) -> i32>",
    ];
    let mut signatures = Vec::new();
    for text in texts {
        let attr = parse_attribute_text(&context, text, unknown)?;
        let printed = print_attribute_text(&context, &attr);
        assert_eq!(printed, text);
        let reparsed = parse_attribute_text(&context, &printed, unknown)?;
        assert_eq!(&*reparsed.borrow(), &*attr.borrow(), "{printed}");
        let attr = attr.borrow();
        signatures.push(Signature::clone(attr.downcast_ref::<SignatureAttr>().unwrap()));
    }

    let [extended, never, plain] = &signatures[..] else {
        unreachable!()
    };
    assert_eq!(extended.params()[0].extension(), ArgumentExtension::Zext);
    assert_eq!(extended.results()[0].extension(), ArgumentExtension::Sext);
    assert_eq!(never.calling_convention().as_str(), "fast");
    assert!(never.never_returns() && never.params().is_empty());
    assert_eq!(
        plain,
        &Signature::new(&context, [context.i32(), context.i64()], [context.i32()])
    );
    Ok(())
}

/// `Signature`'s `Display` and the IR printer write through two writers; for each signature, the
/// `Display` text is the text the printer writes inside `#builtin.signature<..>`.
#[test]
fn a_signatures_display_is_the_text_the_printer_writes() {
    use crate::dialects::builtin::attributes::SignatureAttr;

    let context = Rc::new(Context::default());
    let fields = [context.i32().into(), context.ptr_type(context.rec_var(0)).into()];
    let body = context
        .struct_type(crate::StructType::named(crate::interner::Symbol::intern("list"), fields));
    let list = context.rec_group(&[body]).unwrap()[0];
    let function = context.function_type([context.i32()], [context.i64()]);

    let signatures = [
        Signature::with_convention(&context, CallConv::new("a\"b"), [context.i32()], []),
        Signature::with_convention(&context, CallConv::new("\u{e9}"), [], [context.i8()]),
        Signature::new(&context, [list, context.ptr_type(list)], [context.ptr_type(list)]),
        Signature::new(&context, [context.i32()], [function]),
        Signature::new(&context, [function], [context.never_type()]),
        Signature {
            params: alloc::vec![
                AbiParam::zext(context.i8(), &context),
                AbiParam::sret(context.ptr_type(list), &context),
            ],
            results: alloc::vec![AbiParam::sext(context.i16(), &context)],
            cc: CallConv::new("a\"b"),
        },
    ];
    for signature in signatures {
        let displayed = format!("{signature}");
        let attr = context.create_attribute::<SignatureAttr, _>(signature);
        let flags = Default::default();
        let mut printer = AsmPrinter::fragment(context.clone(), &flags);
        printer.print_attribute_value(&*attr.borrow());
        let printed = printer.finish().to_string();
        let inner = printed
            .strip_prefix("#builtin.signature<")
            .and_then(|text| text.strip_suffix('>'))
            .unwrap_or_else(|| panic!("{printed}"));
        assert_eq!(displayed, inner);
    }
}

/// A calling-convention name with a quote and a non-ASCII character survives a declaration and a
/// signature attribute.
#[test]
fn a_quoted_calling_convention_name_round_trips_in_a_declaration() -> TestResult {
    use crate::dialects::builtin::attributes::SignatureAttr;

    let name = "q\"\u{e9}";
    let escaped = name.escape_default().to_string();
    let test = ParserTest::default();
    let source = format!("builtin.function public extern(\"{escaped}\") @f(i32) -> i32;");
    let function = assert_function_round_trips(&test, &source, &source);
    assert_eq!(function.borrow().get_signature().calling_convention().as_str(), name);

    let context = test.context_rc();
    let text = format!("#builtin.signature<extern(\"{escaped}\") (i32)>");
    let attr = parse_attribute_text(&context, &text, context.unknown_type())?;
    assert_eq!(print_attribute_text(&context, &attr), text);
    let attr = attr.borrow();
    let signature = attr.downcast_ref::<SignatureAttr>().unwrap();
    assert_eq!(signature.calling_convention().as_str(), name);
    Ok(())
}

/// A parameter list names all of its parameters or none: the first element of the other form is
/// the error.
#[test]
fn a_parameter_list_mixing_named_and_unnamed_parameters_is_an_error() {
    for (source, labelled, reason) in [
        (
            "builtin.function public extern(\"C\") @f(%0: i32, i64) {\n    builtin.ret;\n};",
            "i64",
            "expected a named parameter, `%name: type`, as the first parameter of this list is \
             named",
        ),
        (
            "builtin.function public extern(\"C\") @f(i32, %1: i64);",
            "%1",
            "expected a type: the parameters of this list are not named",
        ),
    ] {
        let (message, labels) = parse_error_labels(source);
        assert!(message.contains("invalid parameter list"), "{message}");
        assert!(message.contains(reason), "{message}");
        assert_eq!(labels.first(), Some(&range_of(source, labelled)), "{message}");
    }
}

/// A function declared with a list of types has no body.
#[test]
fn a_declaration_followed_by_a_body_is_an_error() {
    let source = "builtin.function public extern(\"C\") @f(i32) {\n    builtin.ret;\n};";
    let (message, labels) = parse_error_labels(source);
    assert!(message.contains("invalid function declaration"), "{message}");
    assert!(
        message.contains("a function declared without parameter names cannot have a body"),
        "{message}"
    );
    assert_eq!(labels.first(), Some(&range_of(source, "{")), "{message}");
}

/// A dictionary after an unparenthesised result of a declaration: a declaration has no body, so the
/// `{` is reported as one, with the note that a result's attributes need the parentheses.
#[test]
fn a_dictionary_on_a_bare_result_of_a_declaration_is_an_error() {
    let source = "builtin.function public extern(\"C\") @f(i32) -> i32 { extend = #builtin.sext };";
    let (message, labels) = parse_error_labels(source);
    assert!(message.contains("invalid function declaration"), "{message}");
    assert!(
        message.contains(
            "a function declared without parameter names cannot have a body; the attributes of a \
             result are written in parentheses, `-> (type { .. })`"
        ),
        "{message}"
    );
    assert_eq!(labels.first(), Some(&range_of(source, "{")), "{message}");
}

/// A dictionary after an unparenthesised result of a definition is read as the start of the body:
/// the `{` after `-> i32` is always the body, so the dictionary's entry is read as an operation and
/// fails there. That is acceptable because the printer never writes a dictionary on an
/// unparenthesised result, so only hand-written text can reach it, and the error points at the
/// entry that is not an operation.
#[test]
fn a_dictionary_on_a_bare_result_of_a_definition_is_read_as_the_body() {
    let source = "builtin.function public extern(\"C\") @f(%0: i32) -> i32 { extend = \
                  #builtin.sext } {\n    builtin.ret %0 : (i32);\n};";
    let (message, labels) = parse_error_labels(source);
    assert!(message.contains("invalid operation"), "{message}");
    assert!(message.contains("unknown/unregistered operation"), "{message}");
    assert_eq!(labels.first(), Some(&range_of(source, "extend")), "{message}");
}

/// A function whose parameters are named is a definition, and has a body.
#[test]
fn a_definition_without_a_body_is_an_error() {
    let source = "builtin.function public extern(\"C\") @f(%0: i32) -> i32;";
    let (message, labels) = parse_error_labels(source);
    assert!(message.contains("invalid function definition"), "{message}");
    assert!(
        message.contains("expected the body of a function whose parameters are named"),
        "{message}"
    );
    assert_eq!(labels.first(), Some(&range_of(source, ";")), "{message}");
}

/// Hostile but well-formed function text. Each source parses with verification on,
/// prints as given, and the print parses back to an equal signature.
#[test]
fn hostile_but_well_formed_function_text_round_trips() {
    let f = "builtin.function public extern(\"C\") @f";
    for (source, printed) in [
        // no spaces anywhere they are optional
        (format!("{f}(i32)->i32;"), format!("{f}(i32) -> i32;")),
        (
            format!("{f}(i8{{extend=#builtin.zext}})->(i8{{extend=#builtin.sext}});"),
            format!("{f}(i8 {{ extend = #builtin.zext }}) -> (i8 {{ extend = #builtin.sext }});"),
        ),
        (
            format!(
                "{f}(%0: i8{{extend=#builtin.zext}})->(i8{{extend=#builtin.sext}}){{builtin.ret \
                 %0 : (i8);}};"
            ),
            format!(
                "{f}(%0: i8 {{ extend = #builtin.zext }}) -> (i8 {{ extend = #builtin.sext }}) \
                 {{\n    builtin.ret %0 : (i8);\n}};"
            ),
        ),
        (
            format!("{f}()->i32{{builtin.ret_imm 1 : i32;}};"),
            format!("{f}() -> i32 {{\n    builtin.ret_imm 1 : i32;\n}};"),
        ),
        // function types as parameters and results, with and without dictionaries
        (
            format!("{f}((i32) -> i32 {{ a = true }}) -> ((i32) -> i32 {{ b = true }});"),
            format!(
                "{f}((i32) -> i32 {{ a = #builtin.bool<true> }}) -> ((i32) -> i32 {{ b = \
                 #builtin.bool<true> }});"
            ),
        ),
        (format!("{f}(() -> ()) -> (() -> ());"), format!("{f}(() -> ()) -> (() -> ());")),
        (format!("{f}(() -> ());"), format!("{f}(() -> ());")),
        (
            format!("{f}() -> ((i32) -> i32, i64);"),
            format!("{f}() -> ((i32) -> i32, i64);"),
        ),
        // parentheses the printer does not write
        (format!("{f}(i32) -> (never);"), format!("{f}(i32) -> never;")),
        (
            format!("{f}(i32) -> (never {{ a = true }});"),
            format!("{f}(i32) -> (never {{ a = #builtin.bool<true> }});"),
        ),
        (format!("{f}(i32) -> ();"), format!("{f}(i32);")),
        (format!("{f}(i32 {{}}) -> (i32 {{}});"), format!("{f}(i32) -> i32;")),
        (
            format!("{f}() -> () {{\n    builtin.ret;\n}};"),
            format!("{f}() {{\n    builtin.ret;\n}};"),
        ),
        (
            format!("{f}(%0: i32 {{}}) {{\n    builtin.ret;\n}};"),
            format!("{f}(%0: i32) {{\n    builtin.ret;\n}};"),
        ),
        // what follows a declaration
        (
            format!("{f}(i32) attributes {{ a = true }};"),
            format!("{f}(i32) attributes {{ a = #builtin.bool<true> }};"),
        ),
        (
            format!("{f}() -> i32 attributes {{ a = true }};"),
            format!("{f}() -> i32 attributes {{ a = #builtin.bool<true> }};"),
        ),
        (format!("{f}(i32) -> i32 loc(\"a\":1:1);"), format!("{f}(i32) -> i32;")),
        (format!("{f}(i32) loc(\"a\":1:1);"), format!("{f}(i32);")),
        // an argument with a dictionary and a location
        (
            format!(
                "{f}(%0: i32 {{ a = true }} loc(\"a\":1:1), %1: i64) {{\n    builtin.ret;\n}};"
            ),
            format!(
                "{f}(%0: i32 {{ a = #builtin.bool<true> }}, %1: i64) {{\n    builtin.ret;\n}};"
            ),
        ),
        // types of one token
        (format!("{f}(?) -> ?;"), format!("{f}(?) -> ?;")),
        (format!("{f}(ptr) -> ptr;"), format!("{f}(ptr) -> ptr;")),
        // a quoted key that needs no quotes, and a signature inside a dictionary
        (
            format!("{f}(i8 {{ \"extend\" = #builtin.zext }});"),
            format!("{f}(i8 {{ extend = #builtin.zext }});"),
        ),
        (
            format!(
                "{f}(i32 {{ sig = #builtin.signature<extern(\"a>b\") (i8 {{ extend = \
                 #builtin.zext }}) -> never> }});"
            ),
            format!(
                "{f}(i32 {{ sig = #builtin.signature<extern(\"a>b\") (i8 {{ extend = \
                 #builtin.zext }}) -> never> }});"
            ),
        ),
        // convention names
        (
            "builtin.function public extern(\"\") @f();".to_string(),
            "builtin.function public extern(\"\") @f();".to_string(),
        ),
        (
            "builtin.function public extern(\"a\\u{0}b\") @f();".to_string(),
            "builtin.function public extern(\"a\\u{0}b\") @f();".to_string(),
        ),
        // an empty region is no body: this is a declaration
        (format!("{f}() {{}};"), format!("{f}();")),
    ] {
        let test = ParserTest::default();
        let first = test
            .parse::<Function>("hostile.hir", &source)
            .unwrap_or_else(|err| panic!("{source}\n{}", one_line(&err)));
        let text = print_root(first);
        assert_eq!(without_value_ids(&text), without_value_ids(&printed), "{source}");
        let second = test
            .parse::<Function>("hostile2.hir", &text)
            .unwrap_or_else(|err| panic!("{text}\n{}", one_line(&err)));
        assert_eq!(signature_of(second), signature_of(first), "{text}");
    }
}

/// Text that is close to a function but is not one, with the error each gives.
#[test]
fn hostile_function_text_that_is_an_error() {
    let f = "builtin.function public extern(\"C\") @f";
    for (source, reason) in [
        // a bare function-type result. The printer writes `-> ((i32) -> i32)`.
        (format!("{f}() -> (i32) -> i32;"), "unexpected token '->'"),
        (
            format!("{f}(i32) -> () {{\n    builtin.ret;\n}};"),
            "a function declared without parameter names cannot have a body",
        ),
        (format!("{f}() -> i32 {{ a = true }};"), "unknown/unregistered operation"),
        (format!("{f}(%0: i32,) {{\n    builtin.ret;\n}};"), "expected a named parameter"),
        (format!("{f}(i32,);"), "unexpected token ')'"),
        (format!("{f}(i8 {{ extend = #builtin.zext<> }});"), "has no parser"),
        // an open type cannot be written outside a `rec` block, in any build
        (format!("{f}(ptr<rec.0>);"), "unexpected token"),
        (format!("{f}(%0: ptr<rec.0>) {{\n    builtin.ret;\n}};"), "unexpected token"),
        // variadics are not in the grammar yet: `...` is a token, and it is an error here
        (format!("{f}(i32, ...);"), "unexpected token '...'"),
        (format!("{f}(...);"), "unexpected token '...'"),
        // a named list's region may not label its entry block
        (
            format!("{f}(%0: i32) {{\n^entry:\n    builtin.ret;\n}};"),
            "invalid block name in region with named arguments",
        ),
    ] {
        let test = ParserTest::default();
        let err = test.parse::<Function>("hostile.hir", &source).map(|_| ()).expect_err(&source);
        let message = one_line(&err);
        assert!(message.contains(reason), "{source}\n{message}");
    }
}

/// A key given twice in the dictionary of a parameter or a result is an error at its second
/// occurrence, as in an operation's attribute dictionary.
#[test]
fn a_repeated_key_in_a_parameter_dictionary_is_an_error() {
    let dict = "{ extend = #builtin.zext, extend = #builtin.sext }";
    let second = |source: &str| {
        let start = source.rfind("extend").unwrap();
        (start, start + "extend".len())
    };
    for source in [
        format!("builtin.function public extern(\"C\") @f(i8 {dict});"),
        format!(
            "builtin.function public extern(\"C\") @f(%0: i8 {dict}) {{\n    builtin.ret;\n}};"
        ),
        format!("builtin.function public extern(\"C\") @f() -> (i8 {dict});"),
    ] {
        let (message, labels) = parse_error_labels(&source);
        assert!(
            message.contains("attribute 'extend' occurs more than once in the attribute list"),
            "{message}"
        );
        assert_eq!(labels.first(), Some(&second(&source)), "{message}");
    }

    let context = Rc::new(Context::default());
    let text = format!("#builtin.signature<extern(\"C\") (i8 {dict})>");
    let err = parse_attribute_text(&context, &text, context.unknown_type()).unwrap_err();
    let crate::parse::ParserError::DuplicateAttribute { span, name } = err else {
        panic!("{err:?}");
    };
    assert_eq!(name.as_str(), "extend");
    let range = span.range();
    assert_eq!((range.start() as usize, range.end() as usize), second(&text));
}

/// The signature of `function`, cloned.
fn signature_of(function: UnsafeIntrusiveEntityRef<Function>) -> Signature {
    function.borrow().get_signature().clone()
}

/// The display of `err` on one line.
fn one_line(err: &Report) -> String {
    format!("{}", err.display()).split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A public function `f` built in code with `signature` and no body: a declaration.
fn declared_function(
    context: &Rc<Context>,
    signature: Signature,
) -> UnsafeIntrusiveEntityRef<Function> {
    let mut builder = crate::OpBuilder::new(context.clone());
    builder
        .create_function(crate::Ident::with_empty_span("f".into()), Visibility::Public, signature)
        .unwrap()
}

/// Marking a parameter `sret` twice leaves one `sret` entry, and the function prints text that
/// parses back to the same signature.
#[test]
fn marking_a_parameter_sret_twice_keeps_one_entry() {
    let test = ParserTest::default();
    let context = test.context_rc();
    let mut param = AbiParam::sret(context.ptr_type(context.i32()), &context);
    param.mark_sret(&context);
    assert_eq!(param.attrs.iter().count(), 1, "{param}");

    let signature = Signature {
        params: alloc::vec![param],
        results: alloc::vec![],
        cc: CallConv::default(),
    };
    let printed = print_root(declared_function(&context, signature.clone()));
    let parsed = test
        .parse::<Function>("sret.hir", &printed)
        .unwrap_or_else(|err| panic!("{printed}\n{}", one_line(&err)));
    assert_eq!(signature_of(parsed), signature);
}

/// A key given twice to `AbiParam::new_with_attrs` keeps one entry, the last, as
/// `Operation::set_attribute` does.
#[test]
fn a_parameter_built_with_a_repeated_key_keeps_the_last() {
    use crate::{
        NamedAttribute,
        dialects::builtin::attributes::{ArgumentExtension, SextAttr, ZextAttr},
    };

    let context = Rc::new(Context::default());
    let zext = context.create_attribute::<ZextAttr, _>(()).as_attribute_ref();
    let sext = context.create_attribute::<SextAttr, _>(()).as_attribute_ref();
    let key = crate::interner::symbols::Extend;
    let param = AbiParam::new_with_attrs(
        context.i8(),
        [NamedAttribute::new(key, zext), NamedAttribute::new(key, sext)],
    );
    assert_eq!(param.attrs.iter().count(), 1, "{param}");
    assert_eq!(param.extension(), ArgumentExtension::Sext);
}

/// The text the printer writes for a parameter marked `sret` twice parses back to an equal
/// parameter.
#[test]
fn a_parameter_marked_sret_twice_prints_text_that_parses_back() {
    use crate::dialects::builtin::attributes::SignatureAttr;

    let context = Rc::new(Context::default());
    let mut param = AbiParam::sret(context.ptr_type(context.i32()), &context);
    param.mark_sret(&context);
    assert_eq!(format!("{param}"), "ptr<i32> { sret = #builtin.sret }");
    let signature = Signature {
        params: alloc::vec![param],
        results: alloc::vec![],
        cc: CallConv::default(),
    };
    let attr = context.create_attribute::<SignatureAttr, _>(signature.clone());
    let printed = print_attribute_text(&context, &attr.as_attribute_ref());
    let parsed = parse_attribute_text(&context, &printed, context.unknown_type())
        .unwrap_or_else(|err| panic!("{printed}: {err:?}"));
    let parsed = parsed.borrow();
    assert_eq!(**parsed.downcast_ref::<SignatureAttr>().unwrap(), signature);
}

/// An operation of the test dialect with a region of its own, which is not a callable.
#[crate::derive::operation(dialect = TestDialect)]
pub struct RegionHolder {
    #[region]
    body: crate::RegionRef,
}

/// Parses `source`, which verifies, and checks that it fails verification of `builtin.function`
/// with `reason`, the first label starting where `labelled` does.
#[track_caller]
fn assert_verification_error(source: &str, labelled: &str, reason: &str) {
    assert_verification_error_headed(source, "invalid operation builtin.function", labelled, reason)
}

/// As [assert_verification_error], with the message headed `heading`.
#[track_caller]
fn assert_verification_error_headed(source: &str, heading: &str, labelled: &str, reason: &str) {
    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    test.parse_any_unverified("verify.hir", source)
        .unwrap_or_else(|err| panic!("{source} is well-formed as text: {err:?}"));
    let (message, labels) = parse_error_labels(source);
    assert!(message.contains(heading), "{message}");
    assert!(message.contains(reason), "{message}");
    let start = source.find(labelled).unwrap_or_else(|| panic!("`{labelled}` in `{source}`"));
    assert_eq!(labels.first().map(|label| label.0), Some(start), "{message}");
}

/// The message of the error that verifying `function` alone gives, then the messages of its
/// labels (a function built in code has no spans, so its labels are not rendered), on one line.
fn verify_error_text(function: UnsafeIntrusiveEntityRef<Function>) -> String {
    use crate::Op;

    let function = function.borrow();
    let err = function.as_operation().verify().expect_err("the function does not verify");
    let labels = crate::testing::label_messages(err.as_diagnostic()).join(" | ");
    let text = format!("{} {labels}", err.display());
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `never` is the single result of a function that never returns, and nothing else. The
/// signature is checked by `Operation::verify_signature_attributes`, which runs before the
/// function's own verifier: the message is the generic one, naming the property.
#[test]
fn never_as_a_parameter_or_among_results_does_not_verify() {
    let declaration = "builtin.function public extern(\"C\") @f";
    let invalid = "property 'signature' of 'builtin.function' is an invalid signature";
    for (source, reason) in [
        (
            format!("{declaration}(i32, never);"),
            "parameter 1 has type `never`, which is only valid as the single result of a function \
             that never returns",
        ),
        (
            format!("{declaration}() -> (never, i32);"),
            "result 0 has type `never`, which is only valid as the single result of a function \
             that never returns",
        ),
        (
            format!("{declaration}(%0: i32) -> (i32, never) {{\n    builtin.ret %0 : (i32);\n}};"),
            "result 1 has type `never`, which is only valid as the single result of a function \
             that never returns",
        ),
    ] {
        let reason = format!("{invalid}: {reason}");
        assert_verification_error_headed(&source, "invalid operation", "builtin.function", &reason);
        let (message, _) = parse_error_labels(&source);
        assert!(!message.contains("invalid operation builtin.function"), "{message}");
    }
}

/// A signature type that belongs to another context does not verify, in a declaration, where no
/// entry block or return stands in for it.
#[test]
fn a_signature_type_of_another_context_does_not_verify() {
    let test = ParserTest::default();
    let other = Rc::new(Context::default());
    let foreign = other.ptr_type(other.i64());
    let context = test.context_rc();
    let signature = Signature {
        params: alloc::vec![AbiParam::new(foreign)],
        results: alloc::vec![AbiParam::new(foreign)],
        cc: CallConv::default(),
    };
    let message = verify_error_text(declared_function(&context, signature));
    assert!(
        message.contains(
            "property 'signature' of 'builtin.function' is an invalid signature: parameter 0 has \
             type `ptr<i64>`, which belongs to another context"
        ),
        "{message}"
    );

    let signature = Signature {
        params: alloc::vec![],
        results: alloc::vec![AbiParam::new(foreign)],
        cc: CallConv::default(),
    };
    let message = verify_error_text(declared_function(&context, signature));
    assert!(
        message.contains("result 0 has type `ptr<i64>`, which belongs to another context"),
        "{message}"
    );
}

/// An open type in a declaration's signature does not verify. `AbiParam::new` panics on it in
/// debug builds, so it is built as a literal, which no build checks.
#[test]
fn an_open_signature_type_does_not_verify() {
    let test = ParserTest::default();
    let context = test.context_rc();
    let open = AbiParam {
        ty: context.ptr_type(context.rec_var(0)),
        attrs: Default::default(),
    };
    let signature = Signature {
        params: alloc::vec![AbiParam::new(context.i32()), open],
        results: alloc::vec![],
        cc: CallConv::default(),
    };
    let message = verify_error_text(declared_function(&context, signature));
    assert!(
        message.contains(
            "property 'signature' of 'builtin.function' is an invalid signature: parameter 1 has \
             type `ptr<rec.0>`, which is open"
        ),
        "{message}"
    );
}

/// A signature held as a discardable attribute is checked like the signature of a function.
#[test]
fn an_invalid_signature_in_a_discardable_attribute_does_not_verify() {
    let source = "builtin.function public extern(\"C\") @f(i32) attributes { sig = \
                  #builtin.signature<extern(\"C\") (never) -> (never, i32)> };";
    assert_verification_error_headed(
        source,
        "invalid operation",
        "builtin.function",
        "attribute 'sig' of 'builtin.function' is an invalid signature: parameter 0 has type \
         `never`",
    );
}

/// `InvalidSignature` says which parameter or result is the problem.
#[test]
fn a_signature_validates_where_never_appears() {
    use crate::dialects::builtin::attributes::InvalidSignature;

    let context = Rc::new(Context::default());
    let never = context.never_type();
    let i32 = context.i32();
    assert_eq!(Signature::new(&context, [i32], [never]).validate(), Ok(()));
    assert_eq!(Signature::new(&context, [i32], [i32, i32]).validate(), Ok(()));
    assert_eq!(
        Signature::new(&context, [i32, never], []).validate(),
        Err(InvalidSignature::NeverParameter { index: 1 })
    );
    assert_eq!(
        Signature::new(&context, [], [i32, never]).validate(),
        Err(InvalidSignature::NeverAmongResults { index: 1 })
    );
    let err = Signature::new(&context, [never], []).validate().unwrap_err();
    assert_eq!(
        err.to_string(),
        "parameter 0 has type `never`, which is only valid as the single result of a function \
         that never returns"
    );
}

/// A test function `f` built in code with the signature `signature` gives, whose entry block
/// ends in a `builtin.ret` of nothing.
fn built_function(signature: impl FnOnce(&Context) -> (Vec<Type>, Vec<Type>)) -> Test {
    let mut test = Test::new("f", signature);
    test.function_builder().builder_mut().ret([], SourceSpan::UNKNOWN).unwrap();
    test
}

/// The entry block of a definition has one argument per parameter, of the parameter's type. The
/// parser cannot produce a function that breaks this, so these are built in code.
#[test]
fn an_entry_block_that_disagrees_with_the_parameters_does_not_verify() {
    let test = built_function(|cx| (alloc::vec![cx.i32()], alloc::vec![]));
    let mut function = test.function();
    let i64 = test.context().i64();
    function.borrow_mut().get_signature_mut().params.push(AbiParam::new(i64));
    let message = verify_error_text(function);
    assert!(message.contains("invalid operation builtin.function"), "{message}");
    assert!(
        message.contains(
            "expected one entry block argument per parameter; parameter count: 2, entry block \
             argument count: 1"
        ),
        "{message}"
    );

    let test = built_function(|cx| (alloc::vec![cx.i32(), cx.i32()], alloc::vec![]));
    let mut function = test.function();
    let i64 = test.context().i64();
    function.borrow_mut().get_signature_mut().params[1] = AbiParam::new(i64);
    let message = verify_error_text(function);
    assert!(
        message.contains(
            "argument 1 of the entry block has type `i32`, but parameter 1 has type `i64`"
        ),
        "{message}"
    );
}

/// Every `builtin.ret` returns exactly the results of the signature.
#[test]
fn a_return_that_disagrees_with_the_results_does_not_verify() {
    let function = "builtin.function public extern(\"C\") @f";
    for (source, labelled, reason) in [
        (
            format!("{function}(%0: i64) -> i32 {{\n    builtin.ret %0 : (i64);\n}};"),
            "builtin.ret",
            "this returns (i64), but the function returns (i32)",
        ),
        (
            format!("{function}(%0: i32) -> i32 {{\n    builtin.ret;\n}};"),
            "builtin.ret",
            "this returns (), but the function returns (i32)",
        ),
        (
            format!("{function}(%0: i32) {{\n    builtin.ret %0 : (i32);\n}};"),
            "builtin.ret",
            "this returns (i32), but the function returns ()",
        ),
        (
            format!("{function}(%0: i32) -> ptr<i32> {{\n    builtin.ret_imm 5 : i32;\n}};"),
            "builtin.ret_imm",
            "this returns (i32), but the function returns (ptr<i32>)",
        ),
        (
            format!("{function}(%0: i32) -> (i32, i32) {{\n    builtin.ret_imm 5 : i32;\n}};"),
            "builtin.ret_imm",
            "this returns (i32), but the function returns (i32, i32)",
        ),
        (
            format!("{function}(%0: i32) -> never {{\n    builtin.ret;\n}};"),
            "builtin.ret",
            "this returns from a function that never returns",
        ),
        (
            format!("{function}(%0: i32) -> never {{\n    builtin.ret %0 : (i32);\n}};"),
            "builtin.ret",
            "this returns from a function that never returns",
        ),
        (
            format!("{function}(%0: i32) -> never {{\n    builtin.ret_imm 5 : i32;\n}};"),
            "builtin.ret_imm",
            "this returns from a function that never returns",
        ),
    ] {
        assert_verification_error(&source, labelled, reason);
    }
}

/// A `builtin.ret` in a region of an operation in the body is checked like one in the body.
#[test]
fn a_return_in_a_nested_region_is_checked() {
    use crate::Builder;

    let mut test = Test::named("f");
    test.context_rc().get_or_register_dialect::<TestDialect>();
    let (i64, i32) = (test.context().i64(), test.context().i32());
    test.with_function("f", &[i64], &[i32]);
    {
        let mut f = test.function_builder();
        let entry = f.entry_block();
        let arg = entry.borrow().arguments()[0] as ValueRef;
        let builder = f.builder_mut();
        let holder = builder.create::<RegionHolder, ()>(SourceSpan::UNKNOWN)().unwrap();
        let body = holder.borrow().body().as_region_ref();
        let block = builder.create_block(body, None, &[]);
        builder.set_insertion_point_to_end(block);
        builder.ret([arg], SourceSpan::UNKNOWN).unwrap();
        builder.set_insertion_point_to_end(entry);
        builder.ret_imm(Int::from(0i32), SourceSpan::UNKNOWN).unwrap();
    }
    let message = verify_error_text(test.function());
    assert!(
        message.contains("this returns (i64), but the function returns (i32)"),
        "{message}"
    );
}

/// The walk does not enter a function nested in the body: its returns are its own.
#[test]
fn a_return_in_a_nested_function_is_not_checked_against_the_outer_one() {
    use crate::{Builder, Ident, Op};

    let mut test = Test::named("f");
    let (i64, i32) = (test.context().i64(), test.context().i32());
    test.with_function("f", &[], &[i32]);
    let context = test.context_rc();
    {
        let mut f = test.function_builder();
        let entry = f.entry_block();
        let builder = f.builder_mut();
        let signature = Signature::new(&context, [], [i64]);
        let mut nested = builder.create::<Function, (_, _, _)>(SourceSpan::UNKNOWN)(
            Ident::with_empty_span("g".into()),
            Visibility::Public,
            signature,
        )
        .unwrap();
        let block = nested.borrow_mut().create_entry_block();
        builder.set_insertion_point_to_end(block);
        builder.ret_imm(Int::from(0i64), SourceSpan::UNKNOWN).unwrap();
        builder.set_insertion_point_to_end(entry);
        builder.ret_imm(Int::from(0i32), SourceSpan::UNKNOWN).unwrap();
    }
    test.function()
        .borrow()
        .as_operation()
        .verify()
        .expect("the outer function verifies");
}

/// Functions whose bodies agree with their signatures verify.
#[test]
fn functions_that_agree_with_their_signatures_verify() {
    use crate::Op;

    let test = ParserTest::default();
    test.context().get_or_register_dialect::<TestDialect>();
    let function = "builtin.function public extern(\"C\") @f";
    for source in [
        format!("{function}(%0: i32) {{\n    builtin.ret;\n}};"),
        format!("{function}(%0: i32) -> i32 {{\n    builtin.ret %0 : (i32);\n}};"),
        format!("{function}(%0: i32) -> i32 {{\n    builtin.ret_imm 5 : i32;\n}};"),
        format!(
            "{function}(%0: i32, %1: i64) -> (i64, i32) {{\n    builtin.ret %1, %0 : (i64, \
             i32);\n}};"
        ),
        format!("{function}(i32, i8 {{ extend = #builtin.zext }}) -> i32;"),
        format!("{function}(i32) -> never;"),
        format!("{function}(%0: i32) -> never {{\n    test.ret : ();\n}};"),
    ] {
        let parsed = test
            .parse::<Function>("valid.hir", &source)
            .unwrap_or_else(|err| panic!("{source}\n{err:?}"));
        parsed
            .borrow()
            .as_operation()
            .verify()
            .unwrap_or_else(|err| panic!("{source}\n{err:?}"));
    }
}
