use alloc::{string::ToString, vec::Vec};

use runic_ir::{
    DataLayout, DataLayoutSpec, Endianness, PointerSpec, Type, layout::LayoutError,
    parse::parse_type, traits::TypeConstraint,
};

use super::*;

/// The six Wasm value types.
fn value_types(context: &Context) -> [Type; 6] {
    [
        context.i32(),
        context.i64(),
        context.f32(),
        context.f64(),
        FuncRefType::get(context),
        ExternRefType::get(context),
    ]
}

/// Types that are not Wasm value types: narrow and wide integers, `isize`, a pointer, a struct, a
/// function type.
fn other_types(context: &Rc<Context>) -> Vec<Type> {
    let parse = |text| parse_type(context, "t.runic", text).into_result().expect(text);
    alloc::vec![
        context.int_type(8),
        context.i1(),
        context.int_type(16),
        context.i128(),
        context.isize(),
        parse("ptr"),
        parse("ptr<i32>"),
        parse("struct<i32, i32>"),
        context.function_type([context.i32()], [context.i32()]),
    ]
}

fn assert_accepts<C: TypeConstraint>(accepted: &[Type], all: &[Type]) {
    let constraint = C::get();
    for ty in all {
        assert_eq!(
            constraint.matches(*ty),
            accepted.contains(ty),
            "{ty} against {}",
            constraint.description()
        );
    }
}

#[test]
fn the_reference_types_print_and_parse_as_wasms_names() {
    let context = context();
    let funcref = FuncRefType::get(&context);
    let externref = ExternRefType::get(&context);
    assert_eq!(funcref.to_string(), "!wasm.funcref");
    assert_eq!(externref.to_string(), "!wasm.externref");
    let parse = |text| parse_type(&context, "t.runic", text).into_result().expect(text);
    assert_eq!(parse("!wasm.funcref"), funcref);
    assert_eq!(parse("!wasm.externref"), externref);
}

#[test]
fn the_reference_types_are_uniqued_and_distinct() {
    let context = context();
    assert_eq!(FuncRefType::get(&context), FuncRefType::get(&context));
    assert_eq!(ExternRefType::get(&context), ExternRefType::get(&context));
    assert_ne!(FuncRefType::get(&context), ExternRefType::get(&context));
    assert!(FuncRefType::get(&context).is::<FuncRefType>());
    assert!(!FuncRefType::get(&context).is::<ExternRefType>());
}

#[test]
fn a_reference_type_with_a_body_does_not_parse() {
    let context = context();
    let err = parse_type(&context, "t.runic", "!wasm.funcref<i32>").into_result();
    assert!(err.is_err());
}

#[test]
fn a_reference_type_has_no_layout() {
    let context = context();
    let layout = DataLayout::new(
        &context,
        &DataLayoutSpec::new(Endianness::Little, PointerSpec::sized(32).unwrap()).unwrap(),
    );
    for ty in [FuncRefType::get(&context), ExternRefType::get(&context)] {
        assert!(matches!(layout.layout_of(ty), Err(LayoutError::NoLayout(_))), "{ty}");
    }
}

#[test]
fn the_constraints_accept_exactly_their_wasm_types() {
    let context = context();
    let [i32, i64, f32, f64, funcref, externref] = value_types(&context);
    let mut all: Vec<Type> = value_types(&context).to_vec();
    all.extend(other_types(&context));
    assert_accepts::<IntType>(&[i32, i64], &all);
    assert_accepts::<FloatType>(&[f32, f64], &all);
    assert_accepts::<NumType>(&[i32, i64, f32, f64], &all);
    assert_accepts::<RefType>(&[funcref, externref], &all);
    assert_accepts::<ValType>(&[i32, i64, f32, f64, funcref, externref], &all);
}

#[test]
fn the_predicates_agree_with_the_constraints() {
    let context = context();
    let mut all: Vec<Type> = value_types(&context).to_vec();
    all.extend(other_types(&context));
    for ty in all {
        assert_eq!(is_int_type(ty), IntType.matches(ty), "{ty}");
        assert_eq!(is_float_type(ty), FloatType.matches(ty), "{ty}");
        assert_eq!(is_num_type(ty), NumType.matches(ty), "{ty}");
        assert_eq!(is_ref_type(ty), RefType.matches(ty), "{ty}");
        assert_eq!(is_val_type(ty), ValType.matches(ty), "{ty}");
    }
}

#[test]
fn a_value_type_is_named_as_wasm_spells_it() {
    let context = context();
    let names: Vec<_> = value_types(&context).into_iter().map(val_type_name).collect();
    assert_eq!(names, ["i32", "i64", "f32", "f64", "funcref", "externref"]);
    assert_eq!(val_type_name(context.isize()), "isize");
}

#[test]
fn the_constraints_describe_themselves_in_wasm_terms() {
    assert_eq!(IntType.description(), "`i32` or `i64`");
    assert_eq!(FloatType.description(), "`f32` or `f64`");
    assert_eq!(NumType.description(), "`i32`, `i64`, `f32` or `f64`");
    assert_eq!(RefType.description(), "`funcref` or `externref`");
    assert_eq!(ValType.description(), "`i32`, `i64`, `f32`, `f64`, `funcref` or `externref`");
}

#[test]
fn the_dialect_materializes_no_constant() {
    use runic_ir::{Builder, OpBuilder, SourceSpan};

    let context = context();
    let dialect = context.get_or_register_dialect::<WasmDialect>();
    let mut builder = OpBuilder::new(context.clone());
    let attr = context
        .create_attribute::<MutabilityAttr, _>(Mutability::Mut)
        .as_attribute_ref();
    assert!(
        dialect
            .materialize_constant(
                &mut builder as &mut dyn Builder,
                attr,
                context.i32(),
                SourceSpan::UNKNOWN
            )
            .is_none()
    );
}
