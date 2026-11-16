//! The reference types of the dialect and the constraints on Wasm value types.

use alloc::string::{String, ToString};

use runic_ir::{CompactString, Context, Type, derive::DialectType, traits::TypeConstraint};

use crate::WasmDialect;

/// WebAssembly's `funcref`: a nullable reference to a function.
///
/// Text: `!wasm.funcref`. It has no parameters and no layout: a reference is not stored in linear
/// memory, so asking for its size is an error.
#[derive(Debug, Default, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = WasmDialect, name = "funcref")]
pub struct FuncRefType;

impl FuncRefType {
    /// The type `!wasm.funcref` in `context`.
    pub fn get(context: &Context) -> Type {
        context.intern_type(Self)
    }
}

/// WebAssembly's `externref`: a nullable reference to a host object.
///
/// Text: `!wasm.externref`. It has no parameters and no layout.
#[derive(Debug, Default, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = WasmDialect, name = "externref")]
pub struct ExternRefType;

impl ExternRefType {
    /// The type `!wasm.externref` in `context`.
    pub fn get(context: &Context) -> Type {
        context.intern_type(Self)
    }
}

/// Whether `ty` is `i32` or `i64`: Wasm's `inn`.
pub fn is_int_type(ty: Type) -> bool {
    matches!(ty.integer_width(), Some(32 | 64))
}

/// Whether `ty` is `f32` or `f64`: Wasm's `fnn`.
pub fn is_float_type(ty: Type) -> bool {
    ty.is_float()
}

/// Whether `ty` is `i32`, `i64`, `f32` or `f64`: a Wasm number type.
pub fn is_num_type(ty: Type) -> bool {
    is_int_type(ty) || is_float_type(ty)
}

/// Whether `ty` is `!wasm.funcref` or `!wasm.externref`: a Wasm reference type.
pub fn is_ref_type(ty: Type) -> bool {
    ty.is::<FuncRefType>() || ty.is::<ExternRefType>()
}

/// Whether `ty` is a Wasm value type: a number type or a reference type.
pub fn is_val_type(ty: Type) -> bool {
    is_num_type(ty) || is_ref_type(ty)
}

/// Wasm's spelling of `ty` for a message: `i32`, `i64`, `f32`, `f64`, `funcref`, `externref`. Any
/// other type is spelled as the framework prints it.
pub fn val_type_name(ty: Type) -> String {
    if ty.is::<FuncRefType>() {
        String::from("funcref")
    } else if ty.is::<ExternRefType>() {
        String::from("externref")
    } else {
        ty.to_string()
    }
}

macro_rules! constraint {
    ($(#[$doc:meta])* $Constraint:ident, $description:literal, $matcher:expr) => {
        $(#[$doc])*
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub struct $Constraint;

        impl TypeConstraint for $Constraint {
            #[inline(always)]
            fn get() -> Self {
                Self
            }

            fn description(&self) -> CompactString {
                CompactString::const_new($description)
            }

            #[inline]
            fn matches(&self, ty: Type) -> bool {
                let matcher: fn(Type) -> bool = $matcher;
                matcher(ty)
            }
        }
    };
}

constraint!(
    /// `i32` or `i64`: Wasm's `inn`, the integer number types. Not `i8`, `i1` or `isize`.
    IntType,
    "`i32` or `i64`",
    is_int_type
);

constraint!(
    /// `f32` or `f64`: Wasm's `fnn`, the floating-point number types.
    FloatType,
    "`f32` or `f64`",
    is_float_type
);

constraint!(
    /// `i32`, `i64`, `f32` or `f64`: Wasm's `numtype`.
    NumType,
    "`i32`, `i64`, `f32` or `f64`",
    is_num_type
);

constraint!(
    /// `!wasm.funcref` or `!wasm.externref`: Wasm's `reftype`.
    RefType,
    "`funcref` or `externref`",
    is_ref_type
);

constraint!(
    /// `i32`, `i64`, `f32`, `f64`, `!wasm.funcref` or `!wasm.externref`: Wasm's `valtype`.
    ValType,
    "`i32`, `i64`, `f32`, `f64`, `funcref` or `externref`",
    is_val_type
);
