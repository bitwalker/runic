//! The carriers of the types, their widths, and the type converter.

use runic_dialect_wasm::{ExternRefType, FuncRefType};
use runic_ir::{
    Type,
    dialects::builtin::types::{StructField, StructType},
};

use super::*;
use crate::{carrier, type_converter, width};

/// The carrier of each type, as text: `None` for a type without one.
fn carriers(context: &Context, types: &[Type]) -> Vec<Option<String>> {
    types
        .iter()
        .map(|ty| carrier(context, *ty).map(|carrier| carrier.to_string()))
        .collect()
}

#[test]
fn an_integer_of_up_to_32_bits_is_carried_in_an_i32_and_one_of_up_to_64_in_an_i64() {
    let context = context();
    let widths = [1, 7, 31, 32, 33, 63, 64];
    let types = widths.map(|width| context.int_type(width));
    let expected = ["i32", "i32", "i32", "i32", "i64", "i64", "i64"];
    assert_eq!(carriers(&context, &types), expected.map(|ty| Some(String::from(ty))));
}

#[test]
fn an_integer_wider_than_64_bits_has_no_carrier() {
    let context = context();
    let types = [context.int_type(65), context.i128()];
    assert_eq!(carriers(&context, &types), [None, None]);
}

#[test]
fn isize_pointers_in_address_space_0_and_function_types_are_carried_in_an_i32() {
    let context = context();
    let i8 = context.i8();
    let types = [
        context.isize(),
        context.ptr_type(i8),
        context.opaque_ptr_type(),
        context.function_type([context.i8()], [context.i64()]),
    ];
    assert_eq!(carriers(&context, &types), alloc::vec![Some(String::from("i32")); 4]);
}

#[test]
fn a_pointer_in_another_address_space_and_an_aggregate_have_no_carrier() {
    let context = context();
    let i8 = context.i8();
    let structure = context
        .struct_type(StructType::new([StructField::new(context.i32()), StructField::new(i8)]));
    let types = [context.ptr_type_in(Some(i8), 1), structure, context.array_type(i8, 4)];
    assert_eq!(carriers(&context, &types), [None, None, None]);
}

#[test]
fn the_wasm_value_types_are_their_own_carriers() {
    let context = context();
    let types = [
        context.i32(),
        context.i64(),
        context.f32(),
        context.f64(),
        FuncRefType::get(&context),
        ExternRefType::get(&context),
    ];
    let expected = types.map(|ty| Some(ty.to_string()));
    assert_eq!(carriers(&context, &types), expected);
}

#[test]
fn the_width_of_a_carried_integer_is_its_own_and_that_of_isize_is_32() {
    let context = context();
    for n in [1, 7, 8, 16, 31, 32, 33, 63, 64] {
        assert_eq!(width(context.int_type(n)), Some(n), "i{n}");
    }
    assert_eq!(width(context.isize()), Some(32));
    assert_eq!(width(context.int_type(65)), None);
    assert_eq!(width(context.i128()), None);
    assert_eq!(width(context.opaque_ptr_type()), None);
    assert_eq!(width(context.f32()), None);
}

#[test]
fn the_converter_gives_each_type_its_carrier_and_leaves_one_without_alone() {
    let context = context();
    let converter = type_converter();
    let convert = |ty: Type| -> Vec<String> {
        converter.convert_type(&context, ty).iter().map(ToString::to_string).collect()
    };
    let i8 = context.i8();
    assert_eq!(convert(context.i1()), ["i32"]);
    assert_eq!(convert(context.int_type(7)), ["i32"]);
    assert_eq!(convert(context.int_type(31)), ["i32"]);
    assert_eq!(convert(context.i32()), ["i32"]);
    assert_eq!(convert(context.int_type(33)), ["i64"]);
    assert_eq!(convert(context.int_type(63)), ["i64"]);
    assert_eq!(convert(context.i64()), ["i64"]);
    assert_eq!(convert(context.isize()), ["i32"]);
    assert_eq!(convert(context.ptr_type(i8)), ["i32"]);
    assert_eq!(convert(context.f64()), ["f64"]);
    assert_eq!(convert(FuncRefType::get(&context)), ["!wasm.funcref"]);
    // No carrier: the type converts to itself.
    assert_eq!(convert(context.int_type(65)), ["i65"]);
    assert_eq!(convert(context.i128()), ["i128"]);
    assert_eq!(convert(context.ptr_type_in(Some(i8), 1)), ["ptr<i8, 1>"]);
    assert!(converter.is_legal_type(&context, context.i128()));
    assert!(!converter.is_legal_type(&context, i8));
}

/// The converter answers for function types by their parts, as the builtin patterns require, so a
/// function type converts to the function type of its carriers, not to its own carrier `i32`.
#[test]
fn the_converter_converts_a_function_type_by_its_parts() {
    let context = context();
    let converter = type_converter();
    assert!(converter.converts_function_types());
    let function = context.function_type([context.i8(), context.int_type(33)], [context.i1()]);
    let converted = converter.convert_type(&context, function);
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].to_string(), "(i32, i64) -> i32");
}
