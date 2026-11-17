//! What each generic type is carried in: the type converter of `convert-to-wasm`.
//!
//! The carrier of a type is the Wasm value type a value of it is held in ([carrier]); a type
//! without one cannot be lowered. An integer narrower than its carrier is held zero-extended
//! (see the crate documentation, "Carried form").

use runic_dialect_wasm::{ExternRefType, FuncRefType};
use runic_ir::{
    Context, Type,
    conversion::{TypeConversion, TypeConverter},
};

/// The Wasm value type a value of type `ty` is carried in, or `None` when it has none.
///
/// `i32`, `i64`, `f32`, `f64`, `!wasm.funcref` and `!wasm.externref` are their own carriers. An
/// integer `iN` with `N <= 32` is carried in an `i32`, and one with `32 < N <= 64` in an `i64`;
/// `isize`, a pointer in address space 0 (typed or opaque) and a value of a function type are
/// carried in an `i32`. An integer wider than 64 bits, an aggregate, a pointer in another address
/// space and every other type have no carrier.
pub fn carrier(context: &Context, ty: Type) -> Option<Type> {
    if let Some(width) = ty.integer_width() {
        return match width {
            0..=32 => Some(context.i32()),
            33..=64 => Some(context.i64()),
            _ => None,
        };
    }
    if ty.is_isize() || ty.is_function() {
        return Some(context.i32());
    }
    if let Some(pointer) = ty.as_pointer() {
        return (pointer.addrspace() == 0).then(|| context.i32());
    }
    if ty.is_float() || ty.is::<FuncRefType>() || ty.is::<ExternRefType>() {
        return Some(ty);
    }
    None
}

/// The width `N` of a carried integer: `N` for `iN` with `N <= 64`, and 32 for `isize` (the
/// lowering requires a 32-bit `isize`). `None` for every other type, a pointer included.
///
/// A value of a carried integer whose width is below its carrier's has the bits above `N` clear.
pub fn width(ty: Type) -> Option<u32> {
    match ty.integer_width() {
        Some(width) if width <= 64 => Some(width),
        Some(_) => None,
        None => ty.is_isize().then_some(crate::POINTER_WIDTH),
    }
}

/// The width in bits of the Wasm integer type `ty` (`i32` or `i64`), or `None` for any other type.
pub(crate) fn carrier_width(ty: Type) -> Option<u32> {
    match ty.integer_width() {
        Some(width @ (32 | 64)) => Some(width),
        _ => None,
    }
}

/// The type converter of `convert-to-wasm`: every type with a [carrier] other than itself
/// converts to it, and every other type converts to itself.
///
/// The converter converts builtin function types by their parts
/// ([TypeConverter::with_function_types]), as the builtin conversion patterns require: a function
/// type converts to the function type of its parameters' and results' carriers. So under this
/// converter a value of a function type keeps a function type, not its carrier `i32`: the framework
/// answers for function types before any callback, and `convert-to-wasm` refuses a value of a
/// function type before it converts anything (see [crate::ConvertToWasm]).
///
/// A type with no carrier converts to itself, which the converter calls legal: `convert-to-wasm`
/// reports the operation that holds one before the conversion.
pub fn type_converter() -> TypeConverter {
    let mut converter = TypeConverter::new();
    converter.add_conversion(|context, ty| match carrier(context, ty) {
        Some(carrier) if carrier != ty => Some(TypeConversion::one(carrier)),
        _ => None,
    });
    converter.with_function_types()
}
