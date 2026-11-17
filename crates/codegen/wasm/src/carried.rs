//! The helpers the patterns of `convert-to-wasm` share to keep carried form: constants, the mask
//! that zero-extends a narrow integer in its carrier, sign extension, and zero.
//!
//! Each builds its operations through a [ConversionPatternRewriter] at its insertion point, and
//! creates as few operations as the case needs: none where the value is already what is asked
//! for. Each returns an error only when building an operation fails.

use runic_dialect_wasm as wasm;
use runic_ir::{
    Float, Int, Op, Report, SourceSpan, Type, UnsafeIntrusiveEntityRef, Value, ValueRef,
    conversion::ConversionPatternRewriter,
};

/// The one result of a new operation.
pub(crate) fn result<T: Op>(op: &UnsafeIntrusiveEntityRef<T>) -> ValueRef {
    op.borrow().as_operation().results()[0].borrow().as_value_ref()
}

/// The width of `value`'s type, an `i32` or an `i64`.
fn carrier_width_of(value: ValueRef) -> Result<u32, Report> {
    let ty = value.borrow().ty();
    crate::types::carrier_width(ty).ok_or_else(|| {
        Report::msg(alloc::format!(
            "the value `{}` has type `{ty}`, but a carried integer is an `i32` or an `i64`",
            value.borrow()
        ))
    })
}

/// A `wasm.const` of type `carrier` (`i32` or `i64`) holding the low bits of `bits`: the bits
/// above the carrier's width are dropped.
pub fn const_int(
    rw: &mut ConversionPatternRewriter,
    bits: u64,
    carrier: Type,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    let width = crate::types::carrier_width(carrier).ok_or_else(|| {
        Report::msg(alloc::format!("`{carrier}` is not a Wasm integer type, `i32` or `i64`"))
    })?;
    let value = Int::from_bits(width, u128::from(bits));
    Ok(result(&rw.create_op::<wasm::Const, _>(span, (value,))?))
}

/// `value`, an `i32` or an `i64`, with every bit above its low `n` cleared: a `wasm.and` with
/// `2^n - 1`, or `value` itself when `n` is the carrier's width.
///
/// `n` is from 1 to the carrier's width.
pub fn mask(
    rw: &mut ConversionPatternRewriter,
    value: ValueRef,
    n: u32,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    let width = carrier_width_of(value)?;
    if n >= width {
        return Ok(value);
    }
    let carrier = value.borrow().ty();
    let bits = (1u64 << n) - 1;
    let ones = const_int(rw, bits, carrier, span)?;
    Ok(result(&rw.create_op::<wasm::And, _>(span, (value, ones))?))
}

/// `value`, an `i32` or an `i64`, with its low `n` bits sign-extended to the whole carrier:
/// `value` itself when `n` is the carrier's width; `wasm.extend8_s`, `wasm.extend16_s` or
/// `wasm.extend32_s` (in an `i64`) where `n` is 8, 16 or 32; otherwise a `wasm.shl` and a
/// `wasm.shr_s` by the carrier's width less `n`.
///
/// `n` is from 1 to the carrier's width.
pub fn sign_extend(
    rw: &mut ConversionPatternRewriter,
    value: ValueRef,
    n: u32,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    let width = carrier_width_of(value)?;
    match n {
        n if n >= width => Ok(value),
        8 => Ok(result(&rw.create_op::<wasm::Extend8S, _>(span, (value,))?)),
        16 => Ok(result(&rw.create_op::<wasm::Extend16S, _>(span, (value,))?)),
        32 => Ok(result(&rw.create_op::<wasm::Extend32S, _>(span, (value,))?)),
        n => {
            let carrier = value.borrow().ty();
            let shift = const_int(rw, u64::from(width - n), carrier, span)?;
            let high = result(&rw.create_op::<wasm::Shl, _>(span, (value, shift))?);
            Ok(result(&rw.create_op::<wasm::ShrS, _>(span, (high, shift))?))
        }
    }
}

/// The zero of the Wasm value type `ty`: a `wasm.const 0` of an integer type, a `wasm.constf`
/// positive zero of a float type, a `wasm.ref_null` of a reference type. An error for any other
/// type.
pub fn zero(
    rw: &mut ConversionPatternRewriter,
    ty: Type,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    if crate::types::carrier_width(ty).is_some() {
        return const_int(rw, 0, ty, span);
    }
    match ty.as_float().map(|float| float.bits()) {
        Some(32) => {
            return Ok(result(&rw.create_op::<wasm::ConstF, _>(span, (Float::from_f32(0.0),))?));
        }
        Some(64) => {
            return Ok(result(&rw.create_op::<wasm::ConstF, _>(span, (Float::from_f64(0.0),))?));
        }
        _ => {}
    }
    if wasm::is_ref_type(ty) {
        let mut null = rw.create_op::<wasm::RefNull, _>(span, ())?;
        null.borrow_mut().result_mut().set_type(ty);
        return Ok(result(&null));
    }
    Err(Report::msg(alloc::format!("`{ty}` is not a Wasm value type, and has no zero")))
}
