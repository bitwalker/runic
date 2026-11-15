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

/// The populate function refuses a converter that does not convert function types before adding
/// anything.
#[test]
fn the_patterns_refuse_a_converter_they_cannot_be_driven_by() {
    let context = Rc::new(Context::default());
    let mut patterns = ConversionPatternSet::new(context.clone());

    let flat = TypeConverter::new();
    let err = populate_builtin_conversion_patterns(&mut patterns, &flat).unwrap_err();
    assert!(err.to_string().contains("with_function_types"), "{err}");
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

/// A function with thousands of loads of a local whose index moves converts under the default
/// rewrite limit: the renumbered loads are each legalized in their own turn, not as changes the
/// function's pattern made.
#[test]
fn a_function_with_many_references_to_a_renumbered_local_converts_under_the_default_limit() {
    for count in [1_100, 5_000] {
        let mut body = alloc::string::String::from(
            "    builtin.function public extern(\"C\") @big(%a: i128) -> i128 locals(i128, i128) {
        builtin.store_local 0, %a;
        builtin.store_local 1, %a;\n",
        );
        for index in 0..count {
            body.push_str(&format!("        %l{index} = builtin.load_local 1 : i128;\n"));
        }
        body.push_str("        builtin.ret %l0 : (i128);\n    };");
        let (context, module) = parse_module(&body);
        let converter = wide_converter(&context);
        convert_full(module, by_type_target(&context, &converter), &converter)
            .unwrap_or_else(|err| panic!("{count} loads: {}", describe(&err)));
        // Counted by a walk: printing a block of this length overflows a test thread's stack.
        let mut loads = alloc::collections::BTreeMap::<(u32, alloc::string::String), usize>::new();
        module.borrow().prewalk_all(|op: &Operation| {
            if let Some(load) = op.downcast_ref::<LoadLocal>() {
                let local = *load.get_local();
                *loads.entry((local.index(), local.ty().to_string())).or_default() += 1;
            }
        });
        let expected = [((2, "i64".to_string()), count), ((3, "i64".to_string()), count)];
        assert_eq!(loads, expected.into_iter().collect(), "{count} loads");
    }
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

/// A terminator with an operand of its own ahead of its two successors, a condition, whose
/// successors take only forwarded operands.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(Terminator),
    implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct CondTwoBr {
    #[operand]
    condition: crate::traits::AnyType,
    #[successor]
    first: Successor,
    #[successor]
    second: Successor,
}

impl BranchOpInterface for CondTwoBr {}

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

/// A converter answers for types only, so every path of the driver and of the builtin patterns
/// (operands, results, block arguments, signatures of definitions, declarations and calls, function
/// types, locals, forwarded branch operands) asks the type callbacks: a callback that counts the
/// types it is asked about is asked about each type of a value of the module, and, the answers
/// being remembered, once. A function type is answered by the converter from its parts
/// ([TypeConverter::with_function_types]), whose types are asked about.
#[test]
fn every_path_of_a_conversion_asks_the_type_callbacks() {
    use core::cell::RefCell;

    use crate::{FxHashMap, Value as _};

    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(%0: i128, %1: i32) -> i128 locals(i128) {
        builtin.store_local 0, %0;
        %2 = builtin.load_local 0 : i128;
        test.pass_br ^bb1(%2 : i128);
    ^bb1(%3: i128):
        builtin.ret %3 : (i128);
    };
    builtin.function public extern(\"C\") @decl(i128) -> i128;
    builtin.function public extern(\"C\") @caller(%0: i128, %1: i32) -> i128 {
        %2 = builtin.call @f(%0, %1) : extern(\"C\") (i128, i32) -> i128;
        %3 = builtin.function_ref @f : (i128, i32) -> i128;
        %4 = builtin.call_indirect %3(%2, %1) : extern(\"C\") (i128, i32) -> i128;
        %5 = builtin.call @decl(%4) : extern(\"C\") (i128) -> i128;
        builtin.ret %5 : (i128);
    };",
    );
    // The types of every value the module holds, before the conversion, a function type by its
    // parameters and results.
    let mut value_types = alloc::collections::BTreeSet::new();
    let mut insert = |ty: Type| match ty.as_function() {
        Some(function) => {
            for part in function.params().iter().chain(function.results()) {
                value_types.insert(part.to_string());
            }
        }
        None => {
            value_types.insert(ty.to_string());
        }
    };
    module.borrow().prewalk_all(|op: &Operation| {
        for operand in op.operands().iter() {
            insert(operand.borrow().ty());
        }
        for result in op.results().iter() {
            insert(result.borrow().ty());
        }
        for region in op.regions() {
            for block in region.body() {
                for argument in block.argument_values() {
                    insert(argument.borrow().ty());
                }
            }
        }
    });
    assert_eq!(value_types.len(), 2, "{value_types:?}");

    let asked = Rc::new(RefCell::new(FxHashMap::<Type, usize>::default()));
    let mut converter = wide_converter(&context);
    {
        let asked = Rc::clone(&asked);
        // Tried first, and declines: the callbacks of `wide_converter` answer.
        converter.add_conversion(move |_, ty| {
            *asked.borrow_mut().entry(ty).or_default() += 1;
            None
        });
    }
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    converted_text(module);

    let asked = asked.borrow();
    assert!(
        asked.values().all(|count| *count == 1),
        "each type is asked about once: {asked:?}"
    );
    let asked_types = asked
        .keys()
        .map(|ty| ty.to_string())
        .collect::<alloc::collections::BTreeSet<_>>();
    assert!(
        value_types.is_subset(&asked_types),
        "every type of a value is asked about: {value_types:?} of {asked_types:?}"
    );
    for ty in [context.i128(), context.i32()] {
        assert!(asked.contains_key(&ty), "{ty} is asked about");
    }
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

/// A branch with a condition ahead of its successors keeps the condition, and its forwarded
/// operands become the values they became: an `i128` its two halves and a marker nothing, so a
/// successor that forwarded only a marker forwards nothing.
#[test]
fn a_branch_keeps_its_condition_and_forwards_what_its_operands_became() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @b(%c: i1, %0: i128, %m: array<i8; 0>, %1: \
         i32) -> i128 {
        test.cond_two_br %c ^bb1(%0, %m, %1 : i128, array<i8; 0>, i32), ^bb2(%m : array<i8; 0>) : \
         (i1);
    ^bb1(%2: i128, %3: array<i8; 0>, %4: i32):
        builtin.ret %2 : (i128);
    ^bb2(%5: array<i8; 0>):
        builtin.ret %0 : (i128);
    };",
    );
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_str_eq!(
        converted_text(module),
        module_text(
            "    builtin.function public extern(\"C\") @b(%8: i1, %9: i64, %10: i64, %11: i32) -> \
             (i64, i64) {
        test.cond_two_br %8 ^block3(%9, %10, %11 : i64, i64, i32), ^block4 : (i1);
    ^block3(%14: i64, %15: i64, %16: i32):
        builtin.ret %14, %15 : (i64, i64);
    ^block4:
        builtin.ret %9, %10 : (i64, i64);
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

/// A world of two modules: `@one` defines `@id`, an alias of it and an alias of that alias, and
/// `@two` calls `@id` through `call`.
fn two_modules(call: &str) -> alloc::string::String {
    format!(
        "builtin.world {{
    builtin.module public @one {{
        builtin.function public extern(\"C\") @id(%a: i128) -> i128 {{
            builtin.ret %a : (i128);
        }};
        builtin.function_alias public @first -> @id;
        builtin.function_alias public @second -> @first;
    }};
    builtin.module public @two {{
        builtin.function public extern(\"C\") @user(%a: i128) -> i128 {{
            %r = builtin.call {call}(%a) : extern(\"C\") (i128) -> i128;
            builtin.ret %r : (i128);
        }};
    }};
}};"
    )
}

/// Parses `source` as a `builtin.world` and verifies it.
fn parse_world(source: &str) -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    let world = parse::<crate::dialects::builtin::World>(
        ParserConfig::new(context.clone()),
        "world",
        source,
    )
    .into_result()
    .unwrap_or_else(|err| panic!("{}", describe(&err)));
    (context, world.as_operation_ref())
}

/// The operation `@name` under `root`.
fn symbol_named(root: OperationRef, name: &str) -> OperationRef {
    let mut found = None;
    root.borrow().prewalk_all(|op: &Operation| {
        if op.as_trait::<dyn Symbol>().is_some_and(|symbol| symbol.name().as_str() == name) {
            found = Some(op.as_operation_ref());
        }
    });
    found.unwrap_or_else(|| panic!("no @{name}"))
}

/// A conversion rooted at the callee's module, whose caller is in another module, is refused
/// before anything changes: directly, and through an alias of an alias. The error names the
/// function and the user outside the root.
#[test]
fn a_function_with_a_user_outside_the_root_is_refused() {
    for (call, user) in [
        ("::@one::@id", "'builtin.call' in ::two::user"),
        ("::@one::@second", "'builtin.call' in ::two::user"),
    ] {
        let (context, world) = parse_world(&two_modules(call));
        let before = world.borrow().to_string();
        let converter = wide_converter(&context);
        let one = symbol_named(world, "one");
        let err = convert_full(one, by_type_target(&context, &converter), &converter).unwrap_err();
        let err = describe(&err);
        assert!(err.contains("function 'builtin.function' ::one::id"), "{err}");
        assert!(err.contains(&format!("named by {user} outside the root")), "{err}");
        assert!(err.contains("'builtin.module' ::one"), "{err}");
        assert_str_eq!(world.borrow().to_string(), before);
        world.borrow().recursively_verify().expect("the world verifies");
    }
}

/// An alias outside the root that names the converted function is a user outside the root.
#[test]
fn an_alias_outside_the_root_is_a_user_outside_the_root() {
    let source = "builtin.world {
    builtin.module public @one {
        builtin.function public extern(\"C\") @id(i128) -> i128;
    };
    builtin.module public @two {
        builtin.function_alias public @far -> ::@one::@id;
    };
};";
    let (context, world) = parse_world(source);
    let before = world.borrow().to_string();
    let converter = wide_converter(&context);
    let one = symbol_named(world, "one");
    let err = convert_full(one, by_type_target(&context, &converter), &converter).unwrap_err();
    let err = describe(&err);
    assert!(
        err.contains("named by 'builtin.function_alias' ::two::far outside the root"),
        "{err}"
    );
    assert_str_eq!(world.borrow().to_string(), before);
}

/// A conversion rooted at one function, whose caller is beside it in the module, is refused.
#[test]
fn a_conversion_rooted_at_a_function_with_a_caller_beside_it_is_refused() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @id(%a: i128, %n: i64) -> i128 locals(i128) {
        builtin.store_local 0, %a;
        %b = builtin.load_local 0 : i128;
        builtin.ret %b : (i128);
    };
    builtin.function public extern(\"C\") @user(%a: i128, %n: i64) -> i128 {
        %r = builtin.call @id(%a, %n) : extern(\"C\") (i128, i64) -> i128;
        builtin.ret %r : (i128);
    };",
    );
    let before = module.borrow().to_string();
    let converter = wide_converter(&context);
    let id = symbol_named(module, "id");
    let err = convert_full(id, by_type_target(&context, &converter), &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("named by 'builtin.call' in ::m::user outside the root"), "{err}");
    assert_str_eq!(module.borrow().to_string(), before);
}

/// The same world converted from its root: both modules, directly and through the aliases.
#[test]
fn a_conversion_rooted_at_the_world_converts_every_module() {
    for call in ["::@one::@id", "::@one::@second"] {
        let (context, world) = parse_world(&two_modules(call));
        let converter = wide_converter(&context);
        convert_full(world, by_type_target(&context, &converter), &converter)
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        let text = world.borrow().to_string();
        assert!(text.contains(": i64) -> (i64, i64) {"), "{text}");
        assert!(text.contains(&format!("{call}(")), "{text}");
        assert!(text.contains(": extern(\"C\") (i64, i64) -> (i64, i64);"), "{text}");
        world.borrow().recursively_verify().expect("the world verifies");
    }
}

/// `i128` becomes `(i128, i64)`: an answer that holds a type that is not itself legal.
fn answer_that_is_not_legal(context: &Context) -> TypeConverter {
    let (i64, i128) = (context.i64(), context.i128());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == i128).then(|| TypeConversion::many([i128, i64])));
    converter.with_function_types()
}

/// A converter whose answer for a type is not legal is an error that names the type and what it
/// was the answer for: a parameter of a definition, a result of a declaration, a local, and a
/// parameter of a call whose function is converted after it.
#[test]
fn an_answer_that_is_not_legal_is_an_error_that_says_so() {
    let cases = [
        (
            "    builtin.function public extern(\"C\") @id(%a: i128) -> i128 {
        builtin.ret %a : (i128);
    };",
            "parameter 0 of the signature of 'builtin.function' ::m::id",
        ),
        (
            "    builtin.function public extern(\"C\") @far(i64) -> i128;",
            "result 0 of the signature of 'builtin.function' ::m::far",
        ),
        (
            "    builtin.function public extern(\"C\") @f() locals(i64, i128) {
        builtin.ret;
    };",
            "local 1 of 'builtin.function' ::m::f",
        ),
        (
            "    builtin.function public extern(\"C\") @user(%a: i64) {
        %b = builtin.unrealized_conversion_cast %a : i64 to i128;
        builtin.call @far(%b) : extern(\"C\") (i128);
        builtin.ret;
    };
    builtin.function public extern(\"C\") @far(i128);",
            "parameter 0 of the signature of 'builtin.call' in ::m::user",
        ),
    ];
    for (body, what) in cases {
        let (context, module) = parse_module(body);
        let converter = answer_that_is_not_legal(&context);
        let err =
            convert_full(module, by_type_target(&context, &converter), &converter).expect_err(what);
        let err = describe(&err);
        assert!(
            err.contains(&format!(
                "the type converter answers (i128, i64) for `i128`, {what}, but `i128` is not \
                 legal under the converter"
            )),
            "{err}"
        );
    }
}

/// A `builtin.function_ref` whose function type has an answer that is not legal is an error that
/// names the type, not a conversion that runs to the rewrite limit, even ahead of its callee.
#[test]
fn a_function_reference_under_an_answer_that_is_not_legal_is_an_error() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @g() {
        %0 = builtin.function_ref @f : (i128) -> i128;
        builtin.ret;
    };
    builtin.function public extern(\"C\") @f(i128) -> i128;",
    );
    let converter = answer_that_is_not_legal(&context);
    let err = convert_full(module, by_type_target(&context, &converter), &converter)
        .expect_err("the answer is not legal");
    let err = describe(&err);
    assert!(err.contains("the type of builtin.function_ref"), "{err}");
    assert!(err.contains("is not legal under the converter"), "{err}");
    assert!(!err.contains("exceeded the limit"), "{err}");
}

/// A caller inside the root that comes before the refused function in its module is converted
/// before the refusal and keeps its change, as after any error: the conversion reports the
/// refusal, and the IR is not to be used afterwards.
#[test]
fn a_refusal_after_earlier_changes_in_the_root_is_reported() {
    let source = "builtin.world {
    builtin.module public @one {
        builtin.function public extern(\"C\") @a(%a: i128) -> i128 {
            %r = builtin.call @id(%a) : extern(\"C\") (i128) -> i128;
            builtin.ret %r : (i128);
        };
        builtin.function public extern(\"C\") @id(%a: i128) -> i128 {
            builtin.ret %a : (i128);
        };
    };
    builtin.module public @two {
        builtin.function public extern(\"C\") @user(%a: i128) -> i128 {
            %r = builtin.call ::@one::@id(%a) : extern(\"C\") (i128) -> i128;
            builtin.ret %r : (i128);
        };
    };
};";
    let (context, world) = parse_world(source);
    let converter = wide_converter(&context);
    let one = symbol_named(world, "one");
    let err = convert_full(one, by_type_target(&context, &converter), &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("function 'builtin.function' ::one::id"), "{err}");
    assert!(err.contains("named by 'builtin.call' in ::two::user outside the root"), "{err}");
}

/// Each operation a pattern replaces (a `builtin.function_ref`, a `builtin.call_indirect`, a
/// `builtin.call` and a `builtin.ret`) gives its replacement its attribute dictionary: a symbol
/// reference held directly, and one held in a list, each get a use of their own, owned by the
/// replacement and linked to the same symbol. The symbol has exactly one use per reference that
/// names it afterwards: none is left with an erased operation.
#[test]
fn replaced_operations_keep_their_references_with_uses_of_their_own() {
    let (context, module) = parse_module(
        "    builtin.function public extern(\"C\") @f(i128) -> i128;
    builtin.function public extern(\"C\") @g(%0: i128) -> i128 {
        %1 = builtin.function_ref @f : (i128) -> i128;
        %2 = builtin.call_indirect %1(%0) : extern(\"C\") (i128) -> i128;
        %3 = builtin.call @f(%2) : extern(\"C\") (i128) -> i128;
        builtin.ret %3 : (i128);
    };",
    );
    let f = symbol_named(module, "f");
    let symbol = f.borrow().as_symbol_ref().expect("a symbol");
    let held = |op: &Operation| {
        op.is::<FunctionReference>() || op.is::<CallIndirect>() || op.is::<Call>() || op.is::<Ret>()
    };
    let mut ops = Vec::new();
    module.borrow().prewalk_all(|op: &Operation| {
        if held(op) {
            ops.push(op.as_operation_ref());
        }
    });
    assert_eq!(ops.len(), 4);
    for mut op in ops {
        let mut op = op.borrow_mut();
        op.set_symbol_attribute("direct", symbol);
        let direct = op.get_attribute("direct").unwrap();
        let nested = list_of(&context, &[direct.borrow().dyn_clone()]);
        crate::ir::give_copied_references_their_own_uses(&mut op, &nested);
        op.set_attribute("nested", nested);
    }
    let uses = |f: OperationRef| f.borrow().as_trait::<dyn Symbol>().unwrap().iter_uses().count();
    // The callees of the reference and the call, and two references on each of four operations.
    assert_eq!(uses(f), 10);

    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    // Not printed: a list has no text form.
    module
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let mut references = 0;
    module.borrow().prewalk_all(|op: &Operation| {
        if !held(op) {
            return;
        }
        let direct = op.get_attribute("direct").expect("the direct reference is kept");
        let nested = op.get_attribute("nested").expect("the list is kept");
        let nested = nested.borrow();
        let nested = nested.downcast_ref::<ListAttr>().expect("a list");
        let entries = nested.as_value().as_ref().iter().map(|entry| entry.as_attribute_ref());
        for reference in core::iter::once(direct).chain(entries) {
            let reference = reference.try_downcast_attr::<SymbolRefAttr>().ok().unwrap();
            let user = reference.borrow().try_user().expect("the reference has a use");
            assert_eq!(user.borrow().owner, op.as_operation_ref(), "{}", op.name());
            assert_eq!(user.borrow().attr, reference, "{}", op.name());
            let linked = reference.borrow().resolve().expect("the reference is linked");
            assert_eq!(linked.borrow().name().as_str(), "f", "{}", op.name());
            references += 1;
        }
    });
    assert_eq!(references, 8);
    assert_eq!(uses(f), 10);
}
