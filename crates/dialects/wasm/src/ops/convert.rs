//! Conversions between the number types, and the sign extensions of an integer in place.

use runic_ir::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::TypeAttr,
    effects::MemoryEffectOpInterface,
    traits::*,
    *,
};

use crate::{FloatType, I32, I64, IntType, NumType, *};

conversion! {
    /// Wrap of an `i64` to an `i32`: `i32.wrap_i64`.
    ///
    /// The operand is an `i64`, the result an `i32` (the property `ty`, written in the text) with
    /// the low 32 bits of the operand. It never traps.
    Wrap, "wrap", I64 => I32, |context, from, to| from == context.i64() && to == context.i32(),
    "an `i64` to an `i32`"
}

conversion! {
    /// Sign extension of an `i32` to an `i64`: `i64.extend_i32_s`.
    ///
    /// The operand is an `i32`, the result an `i64` that is the operand read as a signed number.
    /// It never traps.
    ExtendS, "extend_s", I32 => I64, |context, from, to| from == context.i32() && to == context.i64(),
    "an `i32` to an `i64`"
}

conversion! {
    /// Zero extension of an `i32` to an `i64`: `i64.extend_i32_u`.
    ///
    /// The operand is an `i32`, the result an `i64` that is the operand read as an unsigned number.
    /// It never traps.
    ExtendU, "extend_u", I32 => I64, |context, from, to| from == context.i32() && to == context.i64(),
    "an `i32` to an `i64`"
}

same_type_unary! {
    /// Sign extension of the low 8 bits: `i32.extend8_s`, `i64.extend8_s`.
    ///
    /// One operand of an integer type and a result of that type: the low 8 bits of the operand,
    /// read as a signed number. It never traps.
    Extend8S, "extend8_s", IntType, []
}

same_type_unary! {
    /// Sign extension of the low 16 bits: `i32.extend16_s`, `i64.extend16_s`.
    ///
    /// One operand of an integer type and a result of that type: the low 16 bits of the operand,
    /// read as a signed number. It never traps.
    Extend16S, "extend16_s", IntType, []
}

same_type_unary! {
    /// Sign extension of the low 32 bits of an `i64`: `i64.extend32_s`.
    ///
    /// One `i64` operand and an `i64` result: the low 32 bits of the operand, read as a signed
    /// number. There is no such instruction on an `i32`. It never traps.
    Extend32S, "extend32_s", I64, []
}

conversion! {
    /// Conversion of a float to a signed integer, trapping: `i32.trunc_f32_s`, `i32.trunc_f64_s`,
    /// `i64.trunc_f32_s`, `i64.trunc_f64_s`.
    ///
    /// The operand is a float (`f32` or `f64`), the result an integer type (`i32` or `i64`) that
    /// the property `ty` names, written in the text: the operand truncated toward zero. It traps
    /// when the operand is a NaN, and when the truncated value does not fit the signed result. A
    /// trap is not an effect: an unused result removes the operation. [TruncSatS] saturates
    /// instead; [Trunc] rounds a float to a float.
    TruncS, "trunc_s", FloatType => IntType, |_, _, _| true, "a float (`f32` or `f64`) to an integer (`i32` or `i64`)"
}

conversion! {
    /// Conversion of a float to an unsigned integer, trapping: `i32.trunc_f32_u`,
    /// `i32.trunc_f64_u`, `i64.trunc_f32_u`, `i64.trunc_f64_u`.
    ///
    /// The operand is a float, the result an integer type that the property `ty` names: the
    /// operand truncated toward zero. It traps when the operand is a NaN, and when the truncated
    /// value does not fit the unsigned result (a value in `(-1, 0)` truncates to zero and does
    /// not trap). A trap is not an effect: an unused result removes the operation.
    TruncU, "trunc_u", FloatType => IntType, |_, _, _| true, "a float (`f32` or `f64`) to an integer (`i32` or `i64`)"
}

conversion! {
    /// Conversion of a float to a signed integer, saturating: `i32.trunc_sat_f32_s`,
    /// `i32.trunc_sat_f64_s`, `i64.trunc_sat_f32_s`, `i64.trunc_sat_f64_s`.
    ///
    /// The operand is a float, the result an integer type that the property `ty` names: the
    /// operand truncated toward zero, the nearest bound of the signed result when it does not fit,
    /// and 0 for a NaN. It never traps.
    TruncSatS, "trunc_sat_s", FloatType => IntType, |_, _, _| true, "a float (`f32` or `f64`) to an integer (`i32` or `i64`)"
}

conversion! {
    /// Conversion of a float to an unsigned integer, saturating: `i32.trunc_sat_f32_u`,
    /// `i32.trunc_sat_f64_u`, `i64.trunc_sat_f32_u`, `i64.trunc_sat_f64_u`.
    ///
    /// The operand is a float, the result an integer type that the property `ty` names: the
    /// operand truncated toward zero, the nearest bound of the unsigned result when it does not
    /// fit, and 0 for a NaN. It never traps.
    TruncSatU, "trunc_sat_u", FloatType => IntType, |_, _, _| true, "a float (`f32` or `f64`) to an integer (`i32` or `i64`)"
}

conversion! {
    /// Conversion of a signed integer to a float: `f32.convert_i32_s`, `f32.convert_i64_s`,
    /// `f64.convert_i32_s`, `f64.convert_i64_s`.
    ///
    /// The operand is an integer (`i32` or `i64`) read as a signed number, the result a float type
    /// that the property `ty` names, rounded to nearest, ties to even. It never traps.
    ConvertS, "convert_s", IntType => FloatType, |_, _, _| true, "an integer (`i32` or `i64`) to a float (`f32` or `f64`)"
}

conversion! {
    /// Conversion of an unsigned integer to a float: `f32.convert_i32_u`, `f32.convert_i64_u`,
    /// `f64.convert_i32_u`, `f64.convert_i64_u`.
    ///
    /// The operand is an integer read as an unsigned number, the result a float type that the
    /// property `ty` names, rounded to nearest, ties to even. It never traps.
    ConvertU, "convert_u", IntType => FloatType, |_, _, _| true, "an integer (`i32` or `i64`) to a float (`f32` or `f64`)"
}

conversion! {
    /// Promotion of an `f32` to an `f64`: `f64.promote_f32`.
    ///
    /// The operand is an `f32`, the result an `f64` of the same value, which is exact. A NaN gives
    /// a NaN with an unspecified payload. It never traps.
    Promote, "promote", FloatType => FloatType, |context, from, to| from == context.f32() && to == context.f64(),
    "an `f32` to an `f64`"
}

conversion! {
    /// Demotion of an `f64` to an `f32`: `f32.demote_f64`.
    ///
    /// The operand is an `f64`, the result an `f32`, rounded to nearest, ties to even; a value too
    /// large becomes an infinity. A NaN gives a NaN with an unspecified payload. It never traps.
    Demote, "demote", FloatType => FloatType, |context, from, to| from == context.f64() && to == context.f32(),
    "an `f64` to an `f32`"
}

conversion! {
    /// Reinterpretation of the bits of a number as a number of the other kind and the same width:
    /// `i32.reinterpret_f32`, `i64.reinterpret_f64`, `f32.reinterpret_i32`, `f64.reinterpret_i64`.
    ///
    /// The operand is an `i32` and the result an `f32`, or the other way round, or an `i64` and an
    /// `f64`, or the other way round; the result is the type the property `ty` names. The bits do
    /// not change, a NaN's payload included. It never traps.
    Reinterpret, "reinterpret", NumType => NumType, |context, from, to| {
        let (i32, i64, f32, f64) = (context.i32(), context.i64(), context.f32(), context.f64());
        [(i32, f32), (f32, i32), (i64, f64), (f64, i64)].contains(&(from, to))
    },
    "an `i32` to an `f32`, an `f32` to an `i32`, an `i64` to an `f64` or an `f64` to an `i64`"
}
