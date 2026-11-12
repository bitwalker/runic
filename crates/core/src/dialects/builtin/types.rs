//! The builtin dialect's types, which print without a `!builtin.` prefix.

mod array;
mod enumeration;
mod float;
mod function;
mod integer;
mod pointer;
mod rec_var;
mod special;
mod structure;

use core::fmt::{self, Write};

pub use self::{
    array::ArrayType,
    enumeration::{EnumType, EnumVariant, InvalidEnumType},
    float::FloatType,
    function::FunctionType,
    integer::{IntegerType, IsizeType},
    pointer::PointerType,
    rec_var::RecVarType,
    special::{NeverType, UnknownType},
    structure::{StructField, StructRepr, StructType},
};
use crate::types::TypeFormatter;

/// Writes `name` bare if it is a bare identifier, otherwise as a quoted, escaped string.
fn print_name(printer: &mut TypeFormatter<'_, '_>, name: crate::interner::Symbol) -> fmt::Result {
    let text = name.as_str();
    if crate::types::is_bare_identifier(text) {
        printer.write_str(text)
    } else {
        write!(printer, "\"{}\"", text.escape_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn scalar_types_print_bare_and_are_cached() {
        let context = Context::default();
        for (ty, text) in [
            (context.i1(), "i1"),
            (context.i32(), "i32"),
            (context.int_type(256), "i256"),
            (context.int_type(24), "i24"),
            (context.isize(), "isize"),
            (context.f32(), "f32"),
            (context.f64(), "f64"),
            (context.unknown_type(), "?"),
            (context.never_type(), "never"),
        ] {
            assert_eq!(alloc::format!("{ty}"), text);
        }
        assert_eq!(context.i32(), context.int_type(32));
        assert_eq!(context.i32(), context.intern_type(IntegerType::new(32).unwrap()));
        assert_ne!(context.i32(), context.i64());
    }

    #[test]
    fn integer_queries_are_signless_and_isize_has_no_fixed_width() {
        let context = Context::default();
        assert!(context.i64().is_integer());
        assert_eq!(context.i64().integer_width(), Some(64));
        assert!(context.isize().is_integer() && context.isize().is_isize());
        assert_eq!(context.isize().integer_width(), None);
        assert!(!context.f64().is_integer() && context.f64().is_float());
        assert!(context.unknown_type().is_unknown() && context.never_type().is_never());
        assert_eq!(IntegerType::new(0), None);
        assert_eq!(IntegerType::new(IntegerType::MAX_WIDTH + 1), None);
        assert_eq!(context.try_int_type(0), None);
    }

    #[test]
    fn composite_types_print_in_the_documented_syntax() {
        let context = Context::default();
        let i32 = context.i32();
        let i8 = context.i8();
        let sym = crate::interner::Symbol::intern;
        let two = core::num::NonZeroU16::new(2).unwrap();
        let cases = [
            (context.opaque_ptr_type(), "ptr"),
            (context.ptr_type_in(None, 1), "ptr<1>"),
            (context.ptr_type(i32), "ptr<i32>"),
            (context.ptr_type_in(Some(i32), 3), "ptr<i32, 3>"),
            (context.array_type(i8, 16), "array<i8; 16>"),
            (context.struct_type(StructType::new([])), "struct<>"),
            (context.struct_type(StructType::new([i32.into(), i8.into()])), "struct<i32, i8>"),
            (
                context.struct_type(StructType::named(
                    sym("point"),
                    [StructField::new(i32).named(sym("x")), StructField::new(i32).named(sym("y"))],
                )),
                "struct point<x: i32, y: i32>",
            ),
            (
                context.struct_type(
                    StructType::new([StructField::new(i8), StructField::new(i32).aligned(two)])
                        .with_repr(StructRepr::Packed(core::num::NonZeroU16::MIN)),
                ),
                "struct<packed(1); i8, i32 align(2)>",
            ),
            (
                context
                    .struct_type(StructType::new([i32.into()]).with_repr(StructRepr::Transparent)),
                "struct<transparent; i32>",
            ),
            (
                context
                    .struct_type(StructType::new([i32.into()]).with_repr(StructRepr::Align(two))),
                "struct<align(2); i32>",
            ),
            (context.function_type([i32, i8], [i32]), "(i32, i8) -> i32"),
            (context.function_type([], [i32, i8]), "() -> (i32, i8)"),
            (context.function_type([i32], []), "(i32) -> ()"),
            (
                context.function_type([], [context.function_type([i32], [i32])]),
                "() -> ((i32) -> i32)",
            ),
            (
                context.struct_type(StructType::new([]).with_repr(StructRepr::Transparent)),
                "struct<transparent;>",
            ),
            (
                context.struct_type(StructType::new([]).with_repr(StructRepr::Align(two))),
                "struct<align(2);>",
            ),
            (
                context.struct_type(StructType::named(
                    sym("my point"),
                    [StructField::new(i32).named(sym("x\"y"))],
                )),
                "struct \"my point\"<\"x\\\"y\": i32>",
            ),
        ];
        for (ty, text) in cases {
            assert_eq!(alloc::format!("{ty}"), text);
        }
    }

    #[test]
    fn names_are_part_of_struct_identity_and_structure_is_uniqued() {
        let context = Context::default();
        let i32 = context.i32();
        let sym = crate::interner::Symbol::intern;
        let anonymous = context.struct_type(StructType::new([i32.into(), i32.into()]));
        assert_eq!(anonymous, context.struct_type(StructType::new([i32.into(), i32.into()])));
        let point = context.struct_type(StructType::named(sym("point"), [i32.into(), i32.into()]));
        let size = context.struct_type(StructType::named(sym("size"), [i32.into(), i32.into()]));
        assert_ne!(anonymous, point);
        assert_ne!(point, size);
        let labelled = context.struct_type(StructType::new([
            StructField::new(i32).named(sym("x")),
            StructField::new(i32).named(sym("y")),
        ]));
        assert_ne!(anonymous, labelled);
        assert_eq!(context.ptr_type(point), context.ptr_type(point));
        assert_ne!(context.ptr_type(point), context.opaque_ptr_type());
        assert_eq!(context.ptr_type(point).pointee(), Some(point));
        assert_eq!(context.opaque_ptr_type().pointee(), None);
        assert!(context.opaque_ptr_type().is_pointer());
    }

    #[test]
    fn each_structural_accessor_answers_for_its_own_kind_only() {
        let context = Context::default();
        let i32 = context.i32();
        let ptr = context.ptr_type(i32);
        let array = context.array_type(i32, 2);
        let structure = context.struct_type(StructType::new([i32.into()]));
        let function = context.function_type([i32], []);
        let float = context.f64();
        assert_eq!(i32.as_integer().map(IntegerType::width), Some(32));
        assert_eq!(float.as_float().map(FloatType::bits), Some(64));
        assert_eq!(ptr.as_pointer().and_then(PointerType::pointee), Some(i32));
        assert_eq!(array.as_array().map(ArrayType::len), Some(2));
        assert_eq!(structure.as_struct().map(StructType::len), Some(1));
        assert_eq!(function.as_function().map(|f| f.params().len()), Some(1));
        for ty in [i32, ptr, array, structure, function, float, context.isize()] {
            let kinds = [
                ty.as_integer().is_some(),
                ty.as_float().is_some(),
                ty.as_pointer().is_some(),
                ty.as_array().is_some(),
                ty.as_struct().is_some(),
                ty.as_function().is_some(),
            ];
            let expected = usize::from(ty != context.isize());
            assert_eq!(kinds.iter().filter(|kind| **kind).count(), expected, "{ty}");
        }
    }

    #[test]
    #[should_panic(expected = "1 to 16777215")]
    fn int_type_rejects_zero_width() {
        Context::default().int_type(0);
    }

    #[test]
    #[should_panic(expected = "1 to 16777215")]
    fn int_type_rejects_widths_above_the_maximum() {
        Context::default().int_type(IntegerType::MAX_WIDTH + 1);
    }

    #[test]
    fn the_maximum_integer_width_is_accepted() {
        let context = Context::default();
        assert!(IntegerType::new(IntegerType::MAX_WIDTH).is_some());
        assert!(context.try_int_type(IntegerType::MAX_WIDTH).is_some());
    }
}
