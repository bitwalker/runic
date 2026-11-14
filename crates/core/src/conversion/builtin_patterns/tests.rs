//! The function pattern and the local patterns over the builtin and test dialects.
//!
//! The converter splits an `i128` into two `i64` halves (low, high) and drops the zero-sized
//! marker `array<i8; 0>`, as the driver's own tests do; `test.join`, `test.low` and `test.high`
//! realize the casts that remain.

use alloc::{format, rc::Rc, string::ToString};

use pretty_assertions::assert_str_eq;

use super::*;
use crate::{
    AttributeRef, EntityWithId, OpPrinter, RegionKind, RegionKindInterface,
    attributes::AttrList,
    conversion::{
        ConversionConfig, ConversionResult, TypeConversion, UnknownOpPolicy, apply_full_conversion,
        apply_partial_conversion,
    },
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::{
        builtin::{
            Module,
            attributes::{List, ListAttr, SymbolRefAttr, TypeAttr},
        },
        debuginfo::{
            DebugInfoDialect,
            attributes::{Subprogram, SubprogramAttr},
        },
        test::{High, Join, Low, TestDialect},
    },
    effects::MemoryEffectOpInterface,
    parse::{ParserConfig, parse},
    testing::text::describe,
    traits::{NoTerminator, SingleBlock, Terminator},
};

/// A branch that forwards its operands to its one successor: `test.pass_br ^bb1(%0 : i32);`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(Terminator),
    implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct PassBr {
    #[successor]
    target: Successor,
}

impl BranchOpInterface for PassBr {}

/// The zero-sized marker type, `array<i8; 0>`.
fn marker(context: &Context) -> Type {
    context.array_type(context.i8(), 0)
}

/// `i128` becomes `(i64, i64)`, the marker becomes nothing, every other type is itself, and
/// function types convert by their parts; the casts that remain are realized with `test.join`,
/// `test.low` and `test.high`.
fn wide_converter(context: &Context) -> TypeConverter {
    let (i64, i128, marker) = (context.i64(), context.i128(), marker(context));
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| {
        if ty == i128 {
            Some(TypeConversion::many([i64, i64]))
        } else if ty == marker {
            Some(TypeConversion::Drop)
        } else {
            None
        }
    });
    converter.add_source_materialization(move |rewriter, inputs, ty, span| {
        let [low, high] = inputs else {
            return Ok(None);
        };
        if ty != rewriter.context().i128() {
            return Ok(None);
        }
        let join = rewriter.create_op::<Join, _>(span, (*low, *high))?;
        Ok(Some(join.borrow().result().as_value_ref()))
    });
    converter.add_target_materialization(move |rewriter, input, types, span| {
        let i64 = rewriter.context().i64();
        if types != [i64, i64] {
            return Ok(None);
        }
        let low = rewriter.create_op::<Low, _>(span, (input,))?;
        let high = rewriter.create_op::<High, _>(span, (input,))?;
        let low = low.borrow().result().as_value_ref();
        let high = high.borrow().result().as_value_ref();
        Ok(Some(smallvec::smallvec![low, high]))
    });
    converter.with_function_types()
}

/// Every operation is legal, except the builtin operations [add_builtin_legality_by_type] judges
/// by type under `converter`.
fn by_type_target(context: &Rc<Context>, converter: &TypeConverter) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target.set_unknown_op_policy(UnknownOpPolicy::Legal);
    add_builtin_legality_by_type(&mut target, converter);
    target
}

fn builtin_patterns(context: &Rc<Context>, converter: &TypeConverter) -> ConversionPatternSet {
    let mut patterns = ConversionPatternSet::new(context.clone());
    populate_builtin_conversion_patterns(&mut patterns, converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    patterns
}

/// Parses `body` as the body of `builtin.module @m`, in a context with the test and debug-info
/// dialects, and verifies it.
fn parse_module(body: &str) -> (Rc<Context>, OperationRef) {
    parse_module_verified(body, true)
}

/// [parse_module], verifying the module after parsing only when `verify` is true.
fn parse_module_verified(body: &str, verify: bool) -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    context.get_or_register_dialect::<DebugInfoDialect>();
    let source = format!("builtin.module public @m {{\n{body}\n}};");
    let config = ParserConfig::new(context.clone()).verify_after_parse(verify);
    let module = parse::<Module>(config, "builtin-patterns", &source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    (context, module.as_operation_ref())
}

fn module_text(body: &str) -> alloc::string::String {
    format!("builtin.module public @m {{\n{body}\n}};")
}

/// The one function in `module`.
fn function_of(module: OperationRef) -> OperationRef {
    let mut found = None;
    module.borrow().prewalk_all(|op: &Operation| {
        if op.is::<Function>() {
            found = Some(op.as_operation_ref());
        }
    });
    found.expect("a function")
}

fn convert_full(
    module: OperationRef,
    target: ConversionTarget,
    converter: &TypeConverter,
) -> Result<ConversionResult, Report> {
    let patterns = builtin_patterns(&target.context(), converter);
    apply_full_conversion(module, target, patterns, ConversionConfig::default())
}

/// The populate function refuses a converter that does not convert function types, and one with
/// a value callback, before adding anything.
#[test]
fn the_patterns_refuse_a_converter_they_cannot_be_driven_by() {
    let context = Rc::new(Context::default());
    let mut patterns = ConversionPatternSet::new(context.clone());

    let flat = TypeConverter::new();
    let err = populate_builtin_conversion_patterns(&mut patterns, &flat).unwrap_err();
    assert!(err.to_string().contains("with_function_types"), "{err}");
    assert!(patterns.patterns().is_empty());

    let mut valued = wide_converter(&context);
    valued.add_value_conversion(|_, _| None);
    let err = populate_builtin_conversion_patterns(&mut patterns, &valued).unwrap_err();
    assert!(err.to_string().contains("value callbacks"), "{err}");
    assert!(patterns.patterns().is_empty());

    populate_builtin_conversion_patterns(&mut patterns, &wide_converter(&context)).unwrap();
    assert_eq!(patterns.patterns().len(), 8);
}

/// A declaration's signature and declared effects are converted: an effect on parameter 1 follows
/// it past the split of parameter 0, and an effect on result 0 is repeated on both halves.
#[test]
fn a_declaration_is_converted_with_its_effects() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(i128 { extend = #builtin.zext }, array<i8; \
         0>, i32 { extend = #builtin.zext }) -> i128 effects(read(arg 2), write(result 0));",
    );
    let converter = wide_converter(&context);
    let result = convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert!(result.changed());
    assert_str_eq!(
        module.borrow().to_string(),
        module_text(
            "    builtin.function public extern(\"C\") @f(i64, i64, i32 { extend = #builtin.zext \
             }) -> (i64, i64) effects(read(arg 2), write(result 0), write(result 1));"
        )
    );
    assert!(converter.is_legal_op(&function_of(module).borrow()));
}

/// A one-block definition whose body is legal test operations reading the parameters: the entry
/// block takes the converted parameters, and each parameter that became two values is read
/// through a `test.join` that realizes its cast. A parameter that became nothing is gone.
#[test]
fn a_definition_converts_its_entry_block_from_its_signature() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(%0: i128, %1: array<i8; 0>, %2: i32) {
        test.sink %0, %2 : (i128, i32);
        builtin.ret;
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_str_eq!(
        module.borrow().to_string(),
        module_text(
            "    builtin.function public extern(\"C\") @f(%3: i64, %4: i64, %5: i32) {
        %7 = test.join %3, %4;
        test.sink %7, %5 : (i128, i32);
        builtin.ret;
    };"
        )
    );
}

/// The locals `(i128, i32, marker, i64)` become `(i64, i64, i32, i64)`: local 0 is `[0, 2)`, local
/// 1 is 2, local 2 is no local and local 3 is 3. Every load and store is converted, the indices a
/// test operation holds in a property and in a list follow, a debug declaration of local 1 names
/// slot 2, one of local 3 names slot 3, and one of local 2 is erased. The function verifies, and a
/// second conversion changes nothing.
#[test]
fn the_locals_are_converted_with_every_reference_to_them() {
    let declare = |name: &str, slot: u32| {
        format!(
            "        di.debug_declare <{{ variable = #di.variable<{{ name = \"{name}\", file = \
             \"a.rs\", line = 1 }}>, expression = #di.expression<[DI_OP_local_slot({slot})]> }}>;"
        )
    };
    let (context, module) = parse_module(&format!(
        "    builtin.function public extern(\"C\") @f(%0: i128, %1: i32, %2: array<i8; 0>, %3: \
         i64) locals(i128, i32, array<i8; 0>, i64) {{
        builtin.store_local 0, %0;
        builtin.store_local 1, %1;
        builtin.store_local 2, %2;
        builtin.store_local 3, %3;
        %4 = builtin.load_local 0 : i128;
        %5 = builtin.load_local 1 : i32;
        %6 = builtin.load_local 2 : array<i8; 0>;
        %7 = builtin.load_local 3 : i64;
        test.sink %4, %5, %7 : (i128, i32, i64);
        test.touch_local <{{ local = #builtin.local_variable<1, i32> }}>;
        test.touch_local <{{ local = #builtin.local_variable<3, i64> }}>;
{}
{}
{}
        builtin.ret;
    }};",
        declare("a", 1),
        declare("b", 3),
        declare("c", 2),
    ));
    // The second `test.touch_local` also names local 1 inside a list in its dictionary.
    let mut listed: Option<OperationRef> = None;
    module.borrow().prewalk_all(|op: &Operation| {
        if op.name().to_string() == "test.touch_local" {
            listed = Some(op.as_operation_ref());
        }
    });
    let local = context
        .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(1, context.i32()))
        .as_attribute_ref();
    listed
        .unwrap()
        .borrow_mut()
        .set_attribute("listed", list_of(&context, &[local]));
    module
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    // A list does not print: its entry is checked here, and it is removed before printing.
    let mut listed = listed.unwrap();
    {
        let held = listed.borrow().get_attribute("listed").unwrap();
        let held = held.borrow();
        let entries = held.downcast_ref::<ListAttr>().unwrap();
        let entries = entries.as_value().as_ref().iter().collect::<alloc::vec::Vec<_>>();
        assert_eq!(entries.len(), 1);
        let entry = entries[0].as_attribute_ref();
        let entry = entry.borrow();
        let local = entry.downcast_ref::<LocalVariableAttr>().unwrap();
        assert_eq!(*local.as_value(), LocalVariable::new(2, context.i32()));
    }
    function_of(module)
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    listed.borrow_mut().remove_attribute("listed");

    let declared = |name: &str, slot: u32| {
        format!(
            "        di.debug_declare <{{ variable = #di.variable<{{ name = \"{name}\", file = \
             \"a.rs\", line = 1 }}>, expression = #di.expression<[DI_OP_local_slot({slot})]> }}>;"
        )
    };
    assert_str_eq!(
        module.borrow().to_string(),
        module_text(&format!(
            "    builtin.function public extern(\"C\") @f(%8: i64, %9: i64, %10: i32, %11: i64) \
             locals(i64, i64, i32, i64) {{
        builtin.store_local 0, %8;
        builtin.store_local 1, %9;
        builtin.store_local 2, %10;
        builtin.store_local 3, %11;
        %14 = builtin.load_local 0 : i64;
        %15 = builtin.load_local 1 : i64;
        %17 = test.join %14, %15;
        %5 = builtin.load_local 2 : i32;
        %7 = builtin.load_local 3 : i64;
        test.sink %17, %5, %7 : (i128, i32, i64);
        test.touch_local <{{ local = #builtin.local_variable<2, i32> }}>;
        test.touch_local <{{ local = #builtin.local_variable<3, i64> }}>;
{}
{}
        builtin.ret;
    }};",
            declared("a", 2),
            declared("b", 3),
        ))
    );
    let function = function_of(module);
    function
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert!(converter.is_legal_op(&function.borrow()));

    let again = convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert!(!again.changed(), "a second conversion changes nothing");
}

fn list_of(context: &Rc<Context>, entries: &[AttributeRef]) -> AttributeRef {
    let mut list = AttrList::new();
    for entry in entries {
        list.push_back(entry.borrow().as_attr().as_attr_ref());
    }
    context.create_attribute::<ListAttr, _>(List::from(list)).as_attribute_ref()
}

/// A debug-info subprogram the function carries holds its function type: the type is replaced by
/// the converted function type, the function is converted once, and it is legal afterwards. A
/// type held by an attribute that converts to several types cannot be replaced, and the
/// conversion reports the function.
#[test]
fn a_type_another_attribute_of_the_function_holds_is_converted() {
    let source = "    builtin.function public extern(\"C\") @f(i128);";
    let (context, module) = parse_module(source);
    let converter = wide_converter(&context);
    let mut function = function_of(module);
    let ty = context.function_type([context.i128()], []);
    let subprogram = Subprogram::new(
        crate::interner::Symbol::intern("f"),
        crate::interner::Symbol::intern("a.rs"),
        1,
        None,
    )
    .with_function_type(ty);
    let subprogram = context.create_attribute::<SubprogramAttr, _>(subprogram).as_attribute_ref();
    function.borrow_mut().set_attribute("di.subprogram", subprogram);
    assert!(!converter.is_legal_op(&function.borrow()));

    let result = convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_eq!(result.converted_ops(), 1, "the function is converted once");
    assert!(converter.is_legal_op(&function.borrow()));
    let held = function.borrow().get_attribute("di.subprogram").unwrap();
    let held = held.borrow();
    let held = held.downcast_ref::<SubprogramAttr>().unwrap();
    assert_eq!(
        held.as_value().ty,
        Some(context.function_type([context.i64(), context.i64()], []))
    );

    // A `TypeAttr` of `i128` converts to two types: the function stays illegal.
    let (context, module) = parse_module(source);
    let converter = wide_converter(&context);
    let wide = context.create_attribute::<TypeAttr, _>(context.i128()).as_attribute_ref();
    function_of(module).borrow_mut().set_attribute("wide", wide);
    let err = convert_full(module, by_type_target(&context, &converter), &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("failed to legalize operation 'builtin.function'"), "{err}");
    assert!(
        err.contains(
            "the function still holds a type that is not legal after its signature, locals and \
             attributes were converted: `i128`, held by attribute 'wide'"
        ),
        "{err}"
    );
}

/// An integer constant in the function's dictionary is not given the type its type converts to,
/// which would give it a value of the wrong width: it is left as it is, the function stays
/// illegal, and the conversion reports the constant's type and where it is held. The function
/// still verifies.
#[test]
fn an_integer_constant_the_function_holds_is_left_as_it_is() {
    let (context, module) = parse_module("    builtin.function public extern(\"C\") @f(i16);");
    let (i16, i32) = (context.i16(), context.i32());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == i16).then(|| TypeConversion::one(i32)));
    let converter = converter.with_function_types();
    let mut function = function_of(module);
    let constant = context.int_attr(crate::Int::from(1i16), i16).as_attribute_ref();
    function.borrow_mut().set_attribute("c", constant);

    let err = convert_full(module, by_type_target(&context, &converter), &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("failed to legalize operation 'builtin.function'"), "{err}");
    assert!(
        err.contains(
            "is not legal after its signature, locals and attributes were converted: `i16`, held \
             by attribute 'c'"
        ),
        "{err}"
    );
    let held = function.borrow().get_attribute("c").unwrap();
    let held = held.borrow();
    let held = held.downcast_ref::<IntAttr>().unwrap();
    assert_eq!(crate::Attribute::ty(held), i16, "the constant keeps its type");
    function.borrow().verify().unwrap_or_else(|err| panic!("{}", describe(&err)));
}

/// A function that stays illegal after its first application, because a dictionary entry holds a
/// type that converts to two, is renumbered once: its reference to local 1 names local 2 after
/// the first application, and the second offer, which finds nothing to change, reports it.
#[test]
fn a_function_that_stays_illegal_is_renumbered_once() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f() locals(i128, i32) {
        test.touch_local <{ local = #builtin.local_variable<1, i32> }>;
        builtin.ret;
    };",
    );
    let converter = wide_converter(&context);
    let wide = context.create_attribute::<TypeAttr, _>(context.i128()).as_attribute_ref();
    function_of(module).borrow_mut().set_attribute("wide", wide);
    let err = convert_full(module, by_type_target(&context, &converter), &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("failed to legalize operation 'builtin.function'"), "{err}");
    assert!(err.contains("`i128`, held by attribute 'wide'"), "{err}");
    let text = module.borrow().to_string();
    assert!(text.contains("locals(i64, i64, i32)"), "{text}");
    assert!(
        text.contains("test.touch_local <{ local = #builtin.local_variable<2, i32> }>;"),
        "{text}"
    );
}

/// The `local` property of the one `test.touch_local` in `region`'s first block.
fn touched_local(region: crate::RegionRef) -> LocalVariable {
    let block = region.borrow().entry_block_ref().unwrap();
    let block = block.borrow();
    let op = block
        .body()
        .iter()
        .find(|op| op.is::<crate::dialects::test::TouchLocal>())
        .expect("a touch_local")
        .as_operation_ref();
    let op = op.borrow();
    *op.downcast_ref::<crate::dialects::test::TouchLocal>().unwrap().get_local()
}

/// An operation with one region that is not a callable.
#[derive(EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, NoTerminator),
    implements(RegionKindInterface, MemoryEffectOpInterface)
)]
pub struct Holder {
    #[region]
    body: Region,
}

impl RegionKindInterface for Holder {
    fn kind(&self) -> RegionKind {
        RegionKind::SSA
    }
}

/// A callable with one region that is not a symbol (a function cannot be nested in a function
/// in a module: a symbol must be a child of its symbol table).
#[derive(EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, NoTerminator),
    implements(RegionKindInterface, MemoryEffectOpInterface, CallableOpInterface)
)]
pub struct Lambda {
    #[region]
    body: Region,
}

impl RegionKindInterface for Lambda {
    fn kind(&self) -> RegionKind {
        RegionKind::SSA
    }
}

impl CallableOpInterface for Lambda {
    fn get_callable_region(&self) -> Option<crate::RegionRef> {
        Some(self.body().as_region_ref())
    }

    fn signature(&self) -> crate::dialects::builtin::attributes::Signature {
        crate::dialects::builtin::attributes::Signature::new(
            &self.as_operation().context_rc(),
            [],
            [],
        )
    }
}

/// A reference to a local in a region of an operation that is not a callable is renumbered with
/// the function's; one in a callable nested in the body is the callable's own, and is left alone.
#[test]
fn references_in_nested_regions_are_renumbered_unless_the_region_is_a_callable() {
    use crate::{Builder, BuilderExt, OpBuilder, dialects::test::TouchLocal};

    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f() locals(i128, i32) {
        builtin.ret;
    };",
    );
    let function = function_of(module);
    let ret = function.borrow().regions().front().get().unwrap().entry_block_ref().unwrap();
    let ret = ret.borrow().body().back().as_pointer().unwrap();
    let span = crate::SourceSpan::UNKNOWN;
    let i32 = context.i32();
    let local = LocalVariable::new(1, i32);
    let mut builder = OpBuilder::new(context.clone());
    builder.set_insertion_point_before(ret);
    let holder = builder.create::<Holder, ()>(span)().unwrap();
    let holder_body = holder.borrow().body().as_region_ref();
    builder.set_insertion_point_before(ret);
    let lambda = builder.create::<Lambda, ()>(span)().unwrap();
    let lambda_body = lambda.borrow().body().as_region_ref();
    for body in [holder_body, lambda_body] {
        let block = builder.create_block(body, None, &[]);
        builder.set_insertion_point_to_end(block);
        builder.create::<TouchLocal, _>(span)(local).unwrap();
    }
    module
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_eq!(touched_local(holder_body), LocalVariable::new(2, i32));
    assert_eq!(touched_local(lambda_body), LocalVariable::new(1, i32));
    function
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
}

/// The declared effects of a definition whose parameter and result split are remapped with its
/// signature, and its body returns the halves.
#[test]
fn the_effects_of_a_definition_are_remapped() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(%0: i128, %1: i32) -> i128 effects(read(arg \
         1), write(result 0)) {
        builtin.ret %0 : (i128);
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_str_eq!(
        converted_text(module),
        module_text(
            "    builtin.function public extern(\"C\") @f(%2: i64, %3: i64, %4: i32) -> (i64, \
             i64) effects(read(arg 2), write(result 0), write(result 1)) {
        builtin.ret %2, %3 : (i64, i64);
    };"
        )
    );
}

/// In a partial conversion whose target calls the function legal, the function is not converted,
/// and a load of an `i128` local is left as it is: the load pattern does not write loads against
/// unconverted locals, and the conversion reports what remains.
#[test]
fn a_load_of_a_local_of_an_unconverted_function_is_left_alone() {
    let source = "    builtin.function public extern(\"C\") @f() locals(i128) {
        %0 = builtin.load_local 0 : i128;
        test.sink %0 : (i128);
        builtin.ret;
    };";
    let (context, module) = parse_module(source);
    let converter = wide_converter(&context);
    let mut target = by_type_target(&context, &converter);
    target.add_legal_op::<Function>();
    let patterns = builtin_patterns(&context, &converter);
    let err = apply_partial_conversion(module, target, patterns, ConversionConfig::default())
        .unwrap_err();
    let err = describe(&err);
    assert!(err.contains("failed to legalize operation 'builtin.load_local'"), "{err}");
    assert!(err.contains("not converted in its function yet"), "{err}");
    assert_str_eq!(module.borrow().to_string(), module_text(source));
}

/// A function with two blocks, under a target that calls the branch legal outright: the second
/// block's arguments are converted with the function, the branch is not, and the conversion fails
/// verification with the branch's error. (The branch pattern converts a branch the target judges
/// by type; see [branches_forward_the_values_their_operands_became].)
#[test]
fn a_branch_to_a_converted_block_that_is_not_converted_fails_verification() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(%0: i128) {
        test.pass_br ^bb1(%0 : i128);
    ^bb1(%1: i128):
        test.sink %1 : (i128);
        builtin.ret;
    };",
    );
    let converter = wide_converter(&context);
    let mut target = by_type_target(&context, &converter);
    target.add_legal_op::<PassBr>();
    let err = convert_full(module, target, &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("invalid operation test.pass_br"), "{err}");
    assert!(err.contains("takes 2 block argument(s)"), "{err}");
}

/// A debug record of `name` whose operation is `op` (`di.debug_declare` or `di.debug_value %v`)
/// and whose expression is `expression`.
fn debug_record(op: &str, name: &str, expression: &str) -> alloc::string::String {
    format!(
        "        {op} <{{ variable = #di.variable<{{ name = \"{name}\", file = \"a.rs\", line = 1 \
         }}>, expression = #di.expression<[{expression}]> }}>{};",
        if op.starts_with("di.debug_value") {
            " : (i32)"
        } else {
            ""
        }
    )
}

/// The local slots of debug expressions follow their locals: a frame base in local 1 names slot
/// 2 after local 0 split. Of a local that became no local, a declaration that is exactly its slot
/// is erased, and a value record that is exactly its slot loses the slot while its other
/// attributes are renumbered.
#[test]
fn debug_expressions_follow_their_locals() {
    let (context, module) = parse_module(&format!(
        "    builtin.function public extern(\"C\") @f(%0: i32) locals(i128, i32, array<i8; 0>, \
         i32) {{
{}
{}
{}
        builtin.ret;
    }};",
        debug_record("di.debug_declare", "a", "DI_OP_frame_base(local_slot, 1+8)"),
        debug_record("di.debug_declare", "b", "DI_OP_local_slot(2)"),
        debug_record("di.debug_value %0", "c", "DI_OP_local_slot(2)"),
    ));
    // The value record of the vanished local also names local 1 in its dictionary.
    let mut value_record = None;
    module.borrow().prewalk_all(|op: &Operation| {
        if op.is::<DebugValue>() {
            value_record = Some(op.as_operation_ref());
        }
    });
    let mut value_record = value_record.unwrap();
    let local = context
        .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(1, context.i32()))
        .as_attribute_ref();
    value_record.borrow_mut().set_attribute("home", local);

    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_str_eq!(
        module.borrow().to_string(),
        module_text(&format!(
            "    builtin.function public extern(\"C\") @f(%0: i32) locals(i64, i64, i32, i32) {{
{}
{}
        builtin.ret;
    }};",
            debug_record("di.debug_declare", "a", "DI_OP_frame_base(local_slot, 2+8)"),
            debug_record("di.debug_value %0", "c", "").replace(
                " : (i32);",
                " attributes { home = #builtin.local_variable<2, i32> } : (i32);"
            ),
        ))
    );
}

/// A debug expression that names a local which became no local, other than exactly its slot,
/// is refused wherever the local is among the function's locals: the conversion fails, naming the
/// operation and the local, before the function is changed.
#[test]
fn a_debug_expression_of_a_vanished_local_is_an_error() {
    for (locals, expression, index) in [
        // The vanished local is the last.
        ("i32, array<i8; 0>", "DI_OP_local_slot(1), DW_OP_deref", 1),
        // A local follows the vanished one.
        ("i128, i32, array<i8; 0>, i32", "DI_OP_local_slot(2), DW_OP_deref", 2),
        // A frame base in the vanished local.
        ("array<i8; 0>, i128", "DI_OP_frame_base(local_slot, 0+8)", 0),
    ] {
        let source = format!(
            "    builtin.function public extern(\"C\") @f() locals({locals}) {{
{}
        builtin.ret;
    }};",
            debug_record("di.debug_declare", "a", expression),
        );
        let (context, module) = parse_module(&source);
        let converter = wide_converter(&context);
        let err =
            convert_full(module, by_type_target(&context, &converter), &converter).unwrap_err();
        let err = describe(&err);
        assert!(
            err.contains(&format!(
                "a debug expression of 'di.debug_declare' names local {index}, which becomes no \
                 local"
            )),
            "{err}"
        );
        assert_str_eq!(module.borrow().to_string(), module_text(&source));
    }
}

/// A store of a local in no function does not match: there is no function whose locals it could
/// name, and the conversion reports it.
#[test]
fn a_store_of_a_local_outside_a_function_is_left_alone() {
    let (context, module) = parse_module(
        "    builtin.global_variable public @g : i32 {
        %0 = test.constant 0 : i128;
        \"builtin.store_local\"(%0) <{ local = #builtin.local_variable<0, i128> }> : (i128) -> ();
    };",
    );
    let converter = wide_converter(&context);
    let err = convert_full(module, by_type_target(&context, &converter), &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("failed to legalize operation 'builtin.store_local'"), "{err}");
    assert!(err.contains("the local is in no function"), "{err}");
}

/// A function made without verification can declare an effect on a parameter or result its
/// signature does not have: the function pattern reports it, naming the function and the effect,
/// instead of remapping it.
#[test]
fn an_effect_outside_the_signature_is_an_error() {
    for (effect, names) in [
        ("read(arg 3)", "names argument 3, but its signature has 1 parameter(s)"),
        ("write(result 1)", "names result 1, but its signature has 1 result(s)"),
    ] {
        let source =
            format!("    builtin.function public extern(\"C\") @f(i128) -> i32 effects({effect});");
        let (context, module) = parse_module_verified(&source, false);
        let converter = wide_converter(&context);
        let err =
            convert_full(module, by_type_target(&context, &converter), &converter).unwrap_err();
        let err = describe(&err);
        assert!(err.contains(&format!("the declared effect `{effect}` of function @f")), "{err}");
        assert!(err.contains(names), "{err}");
        assert_str_eq!(module.borrow().to_string(), module_text(&source));
    }
}

/// A terminator with two successors: the first takes its forwarded operands, the second takes one
/// argument the branch produces itself, then its forwarded operands:
/// `test.two_br ^bb1(%0 : i128), ^bb2(%1 : i32);`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(Terminator),
    implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct TwoBr {
    #[successor]
    first: Successor,
    #[successor]
    second: Successor,
}

impl BranchOpInterface for TwoBr {
    fn get_successor_operands(&self, index: usize) -> crate::SuccessorOperandRange<'_> {
        let op = self.as_operation();
        let group = op.successors()[index].operand_group as usize;
        crate::SuccessorOperandRange::new(index, op.operands().group(group))
    }

    fn get_successor_operands_mut(&mut self, index: usize) -> crate::SuccessorOperandRangeMut<'_> {
        let op = self.as_operation_mut();
        let group = op.successors()[index].operand_group as usize;
        crate::SuccessorOperandRangeMut::new(index, op.operands_mut().group_mut(group))
    }
}

/// The text of `module`, after checking that it verifies and holds neither an `i128` nor a cast.
fn converted_text(module: OperationRef) -> alloc::string::String {
    module
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let text = module.borrow().to_string();
    assert!(!text.contains("i128"), "{text}");
    assert!(!text.contains("unrealized_conversion_cast"), "{text}");
    text
}

/// A whole module in one run: a definition, an alias of it, a declaration, and a caller that calls
/// the definition directly, through the alias, through a function reference, and the declaration.
/// The result verifies (each call's signature equals its callee's), and holds no `i128` and no
/// cast.
#[test]
fn a_module_is_converted_in_one_run() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(%0: i128, %1: i32) -> i128 {
        builtin.ret %0 : (i128);
    };
    builtin.function_alias public @g -> @f;
    builtin.function public extern(\"C\") @decl(i128) -> i128;
    builtin.function public extern(\"C\") @caller(%0: i128, %1: i32) -> i128 {
        %2 = builtin.call @f(%0, %1) : extern(\"C\") (i128, i32) -> i128;
        %3 = builtin.call @g(%2, %1) : extern(\"C\") (i128, i32) -> i128;
        %4 = builtin.function_ref @f : (i128, i32) -> i128;
        %5 = builtin.call_indirect %4(%3, %1) : extern(\"C\") (i128, i32) -> i128;
        %6 = builtin.call @decl(%5) : extern(\"C\") (i128) -> i128;
        builtin.ret %6 : (i128);
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let text = converted_text(module);
    assert_str_eq!(
        text,
        module_text(
            "    builtin.function public extern(\"C\") @f(%9: i64, %10: i64, %11: i32) -> (i64, \
             i64) {
        builtin.ret %9, %10 : (i64, i64);
    };
    builtin.function_alias public @g -> @f;
    builtin.function public extern(\"C\") @decl(i64, i64) -> (i64, i64);
    builtin.function public extern(\"C\") @caller(%13: i64, %14: i64, %15: i32) -> (i64, i64) {
        %17, %18 = builtin.call @f(%13, %14, %15) : extern(\"C\") (i64, i64, i32) -> (i64, i64);
        %20, %21 = builtin.call @g(%17, %18, %15) : extern(\"C\") (i64, i64, i32) -> (i64, i64);
        %23 = builtin.function_ref @f : (i64, i64, i32) -> (i64, i64);
        %25, %26 = builtin.call_indirect %23(%20, %21, %15) : extern(\"C\") (i64, i64, i32) -> \
             (i64, i64);
        %28, %29 = builtin.call @decl(%25, %26) : extern(\"C\") (i64, i64) -> (i64, i64);
        builtin.ret %28, %29 : (i64, i64);
    };"
        )
    );
    let again = convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert!(!again.changed(), "a second conversion changes nothing");
}

/// The results of a call are grouped as the signature's results: of `(i32, i128, i32)`, the
/// call's results 1 and 2 stand for the old result 1, and a return of the old results in another
/// order returns their parts in that order.
#[test]
fn the_results_of_a_call_are_grouped_by_the_signature() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @h() -> (i32, i128, i32);
    builtin.function public extern(\"C\") @caller() -> (i128, i32, i32) {
        %0, %1, %2 = builtin.call @h() : extern(\"C\") () -> (i32, i128, i32);
        builtin.ret %1, %2, %0 : (i128, i32, i32);
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_str_eq!(
        converted_text(module),
        module_text(
            "    builtin.function public extern(\"C\") @h() -> (i32, i64, i64, i32);
    builtin.function public extern(\"C\") @caller() -> (i64, i64, i32, i32) {
        %3, %4, %5, %6 = builtin.call @h() : extern(\"C\") () -> (i32, i64, i64, i32);
        builtin.ret %4, %5, %6, %3 : (i64, i64, i32, i32);
    };"
        )
    );
}

/// A call is converted with its callee, by the same function: under a target that leaves the
/// callee alone, the call's converted signature differs from its callee's, and verification
/// fails with the call's error.
#[test]
fn a_call_converted_without_its_callee_does_not_verify() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(i128, i32) -> i128;
    builtin.function public extern(\"C\") @caller(%0: i128, %1: i32) -> i128 {
        %2 = builtin.call @f(%0, %1) : extern(\"C\") (i128, i32) -> i128;
        builtin.ret %2 : (i128);
    };",
    );
    let converter = wide_converter(&context);
    let mut target = by_type_target(&context, &converter);
    let only_the_caller = converter.clone();
    target.add_dynamically_legal_op::<Function, _>(move |op: &Operation| {
        let callee = op.as_trait::<dyn Symbol>().is_some_and(|f| f.name().as_str() == "f");
        DynamicLegalityResult::legal_if(callee || only_the_caller.is_legal_op(op))
    });
    let err = convert_full(module, target, &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("invalid operation builtin.call"), "{err}");
    assert!(err.contains("the call's signature differs from its callee's"), "{err}");
}

/// A marker-typed argument and result vanish from the call and its callee alike.
#[test]
fn a_marker_argument_and_result_vanish_from_call_and_callee() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @m(%0: array<i8; 0>, %1: i32) -> (array<i8; \
         0>, i32) {
        builtin.ret %0, %1 : (array<i8; 0>, i32);
    };
    builtin.function public extern(\"C\") @caller(%0: array<i8; 0>, %1: i32) -> i32 {
        %2, %3 = builtin.call @m(%0, %1) : extern(\"C\") (array<i8; 0>, i32) -> (array<i8; 0>, \
         i32);
        builtin.ret %3 : (i32);
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let text = converted_text(module);
    assert!(!text.contains("array"), "{text}");
    assert_str_eq!(
        text,
        module_text(
            "    builtin.function public extern(\"C\") @m(%6: i32) -> i32 {
        builtin.ret %6 : (i32);
    };
    builtin.function public extern(\"C\") @caller(%8: i32) -> i32 {
        %10 = builtin.call @m(%8) : extern(\"C\") (i32) -> i32;
        builtin.ret %10 : (i32);
    };"
        )
    );
}

/// A return of `(i128, i32)` returns the three values they became.
#[test]
fn a_return_returns_the_flattened_ranges() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @r(%0: i128, %1: i32) -> (i128, i32) {
        builtin.ret %0, %1 : (i128, i32);
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_str_eq!(
        converted_text(module),
        module_text(
            "    builtin.function public extern(\"C\") @r(%2: i64, %3: i64, %4: i32) -> (i64, \
             i64, i32) {
        builtin.ret %2, %3, %4 : (i64, i64, i32);
    };"
        )
    );
}

/// The branch pattern flattens the forwarded `i128` values of each successor in place, leaves the
/// argument the branch produces for its second successor alone, and leaves that successor's
/// forwarded operands, which do not change, as they are; the block arguments the function
/// pattern converted agree with them.
#[test]
fn branches_forward_the_values_their_operands_became() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @b(%0: i128, %1: i32) -> i128 {
        test.two_br ^bb1(%0, %1 : i128, i32), ^bb2(%1 : i32);
    ^bb1(%2: i128, %3: i32):
        builtin.ret %2 : (i128);
    ^bb2(%4: i32, %5: i32):
        test.sink %4, %5 : (i32, i32);
        builtin.ret %0 : (i128);
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_str_eq!(
        converted_text(module),
        module_text(
            "    builtin.function public extern(\"C\") @b(%6: i64, %7: i64, %8: i32) -> (i64, \
             i64) {
        test.two_br ^block3(%6, %7, %8 : i64, i64, i32), ^block4(%8 : i32);
    ^block3(%10: i64, %11: i64, %12: i32):
        builtin.ret %10, %11 : (i64, i64);
    ^block4(%4: i32, %5: i32):
        test.sink %4, %5 : (i32, i32);
        builtin.ret %6, %7 : (i64, i64);
    };"
        )
    );
}

/// A branch none of whose operands changes is not matched: under a target that calls it illegal
/// outright, the pattern reports no match, and the conversion reports the branch instead of
/// trying it again until the rewrite limit.
#[test]
fn a_branch_whose_operands_do_not_change_is_not_matched() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @b(%0: i32) {
        test.two_br ^bb1(%0 : i32), ^bb2;
    ^bb1(%1: i32):
        builtin.ret;
    ^bb2(%2: i32):
        builtin.ret;
    };",
    );
    let converter = wide_converter(&context);
    let mut target = by_type_target(&context, &converter);
    target.add_illegal_op::<TwoBr>();
    let err = convert_full(module, target, &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("failed to legalize operation 'test.two_br'"), "{err}");
    assert!(!err.contains("limit"), "{err}");
}

/// A converted call keeps its attribute dictionary, and a symbol reference in it gets a use of its
/// own, linked to the same symbol.
#[test]
fn a_converted_call_keeps_its_attributes() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(i128);
    builtin.function public extern(\"C\") @caller(%0: i128) {
        builtin.call @f(%0) : extern(\"C\") (i128) attributes { note = #builtin.i32<7>, target = \
         #builtin.symbol<@f> };
        builtin.ret;
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let text = converted_text(module);
    assert!(
        text.contains(
            "builtin.call @f(%1, %2) : extern(\"C\") (i64, i64) attributes { note = \
             #builtin.i32<7>, target = #builtin.symbol<@f> };"
        ),
        "{text}"
    );
    let mut call = None;
    module.borrow().prewalk_all(|op: &Operation| {
        if op.is::<Call>() {
            call = Some(op.as_operation_ref());
        }
    });
    let call = call.unwrap();
    let target = call.borrow().get_attribute("target").unwrap();
    let target = target.try_downcast_attr::<SymbolRefAttr>().ok().unwrap();
    let linked = target.borrow().resolve().expect("the reference is linked");
    assert_eq!(linked.borrow().name().as_str(), "f");
    assert_eq!(target.borrow().try_user().unwrap().borrow().owner, call);
}
