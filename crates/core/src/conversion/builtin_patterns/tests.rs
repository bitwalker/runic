//! The function pattern and the local patterns over the builtin and test dialects.
//!
//! The converter splits an `i128` into two `i64` halves (low, high) and drops the zero-sized
//! marker `array<i8; 0>`, as the driver's own tests do; `test.join`, `test.low` and `test.high`
//! realize the casts that remain.

use alloc::{format, rc::Rc, string::ToString};

use pretty_assertions::assert_str_eq;

use super::*;
use crate::{
    AttributeRef, EntityWithId, OpPrinter,
    attributes::AttrList,
    conversion::{
        ConversionConfig, ConversionResult, TypeConversion, UnknownOpPolicy, apply_full_conversion,
        apply_partial_conversion,
    },
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::{
        builtin::{
            Module,
            attributes::{List, ListAttr, TypeAttr},
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
    traits::Terminator,
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
    assert_eq!(patterns.patterns().len(), 3);
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
/// verification with the branch's error. (A branch pattern lifts this.)
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
/// is erased, a value record that is exactly its slot loses the slot, and a longer expression
/// names the local that follows it.
#[test]
fn debug_expressions_follow_their_locals() {
    let (context, module) = parse_module(&format!(
        "    builtin.function public extern(\"C\") @f(%0: i32) locals(i128, i32, array<i8; 0>, \
         i32) {{
{}
{}
{}
{}
        builtin.ret;
    }};",
        debug_record("di.debug_declare", "a", "DI_OP_frame_base(local_slot, 1+8)"),
        debug_record("di.debug_declare", "b", "DI_OP_local_slot(2)"),
        debug_record("di.debug_value %0", "c", "DI_OP_local_slot(2)"),
        debug_record("di.debug_declare", "d", "DI_OP_local_slot(2), DW_OP_deref"),
    ));
    let converter = wide_converter(&context);
    convert_full(module, by_type_target(&context, &converter), &converter)
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    assert_str_eq!(
        module.borrow().to_string(),
        module_text(&format!(
            "    builtin.function public extern(\"C\") @f(%0: i32) locals(i64, i64, i32, i32) {{
{}
{}
{}
        builtin.ret;
    }};",
            debug_record("di.debug_declare", "a", "DI_OP_frame_base(local_slot, 2+8)"),
            debug_record("di.debug_value %0", "c", ""),
            debug_record("di.debug_declare", "d", "DI_OP_local_slot(3), DW_OP_deref"),
        ))
    );
}

/// A longer debug expression that names a local which became no local, with no local after it,
/// cannot be renumbered: the conversion fails before the function is changed.
#[test]
fn a_debug_expression_of_a_vanished_last_local_is_an_error() {
    let source = format!(
        "    builtin.function public extern(\"C\") @f() locals(i32, array<i8; 0>) {{
{}
        builtin.ret;
    }};",
        debug_record("di.debug_declare", "a", "DI_OP_local_slot(1), DW_OP_deref"),
    );
    let (context, module) = parse_module(&source);
    let converter = wide_converter(&context);
    let err = convert_full(module, by_type_target(&context, &converter), &converter).unwrap_err();
    let err = describe(&err);
    assert!(err.contains("names local 1, which becomes no local"), "{err}");
    assert_str_eq!(module.borrow().to_string(), module_text(&source));
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
