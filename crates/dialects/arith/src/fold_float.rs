//! Constant folding of floating-point operations.
//!
//! Every fold of a float operation goes through [fold_floats] (one float result) or
//! [fold_float_compare] (an `i1`). They decline unless the operation's types are what a verified
//! operation has, read the operand constants, compute the result with a closure over
//! [Float]s, and create a new attribute for it.
//!
//! An operation that computes a NaN does not fold: which NaN it gives (its sign and payload) differs
//! between hosts and engines, so the closures of `addf`, `subf`, `mulf`, `divf`, `fpext` and
//! `fptrunc` return `None` for a result that is one, and the operation is left to run. An operation
//! that is exact on bits folds for every bit pattern: `bitcast`, and `negf`, which flips the sign
//! bit (see [negated]). A comparison is deterministic, NaN operands included, and does fold.

use alloc::rc::Rc;

use runic_ir::{
    AttributeRef, Context, Float, FoldResult, Int, OpFoldResult, Operation, SmallVec, Type,
    dialects::builtin::attributes::FloatAttr,
};

/// The value of the constant `operand`, if it is a [FloatAttr] of type `ty` holding a value as
/// wide as a constant of `ty`.
///
/// Anything else is not a constant this dialect folds: a missing operand (not a constant), a
/// constant of another type, or an attribute that is not a float constant.
pub(crate) fn constant_float_operand(operand: &Option<AttributeRef>, ty: Type) -> Option<Float> {
    let attr = (*operand.as_ref()?).try_downcast_attr::<FloatAttr>().ok()?;
    let attr = attr.borrow();
    let value = *attr.as_value();
    (runic_ir::Attribute::ty(&*attr) == ty && Float::width_for_type(ty) == Some(value.width()))
        .then_some(value)
}

/// The values of `operands`, if there are `N` of them and each is a constant of type `ty`.
fn constant_float_operands<const N: usize>(
    operands: &[Option<AttributeRef>],
    ty: Type,
) -> Option<[Float; N]> {
    let operands: &[Option<AttributeRef>; N] = operands.try_into().ok()?;
    let mut values = [Float::from_bits32(0); N];
    for (value, operand) in values.iter_mut().zip(operands) {
        *value = constant_float_operand(operand, ty)?;
    }
    Some(values)
}

/// Folds an operation whose operands are constants of the float type `operand_ty` to a float
/// constant of type `result_ty`: the value `f` computes on the operand values.
///
/// Returns `None`, and creates nothing, unless `operand_ty` is a float type, every operand is a
/// [FloatAttr] of that type, `f` folds, and its value is as wide as a constant of `result_ty`. `f`
/// declines (returns `None`) for a NaN result.
pub(crate) fn fold_floats<const N: usize>(
    context: &Rc<Context>,
    operands: &[Option<AttributeRef>],
    operand_ty: Type,
    result_ty: Type,
    f: impl Fn([Float; N]) -> Option<Float>,
) -> Option<AttributeRef> {
    let values = constant_float_operands(operands, operand_ty)?;
    let value = f(values)?;
    (Float::width_for_type(result_ty) == Some(value.width()))
        .then(|| context.float_attr(value).as_attribute_ref())
}

/// Folds an operation whose operands are constants of the float type `operand_ty` to an `i1`
/// constant: the value `f` computes on the operand values.
///
/// Returns `None`, and creates nothing, unless `result_ty` is `i1`, `operand_ty` is a float type,
/// every operand is a [FloatAttr] of that type, and `f` folds.
pub(crate) fn fold_float_compare<const N: usize>(
    context: &Rc<Context>,
    operands: &[Option<AttributeRef>],
    operand_ty: Type,
    result_ty: Type,
    f: impl Fn([Float; N]) -> Option<bool>,
) -> Option<AttributeRef> {
    if result_ty != context.i1() || Float::width_for_type(operand_ty).is_none() {
        return None;
    }
    let values = constant_float_operands(operands, operand_ty)?;
    let value = f(values)?;
    Some(context.int_attr(Int::from(value), result_ty).as_attribute_ref())
}

/// A binary float operation, computed in the width of its operands.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum FloatBinary {
    Add,
    Sub,
    Mul,
    Div,
}

impl FloatBinary {
    /// The result of the operation on `a` and `b`, computed in `f32` for 32-bit operands and in
    /// `f64` for 64-bit ones, round to nearest ties to even; `None` if it is a NaN or the operands
    /// have different widths.
    pub(crate) fn apply(self, a: Float, b: Float) -> Option<Float> {
        let value = match (a.to_f32().zip(b.to_f32()), a.to_f64().zip(b.to_f64())) {
            (Some((a, b)), _) => {
                let value = match self {
                    Self::Add => a + b,
                    Self::Sub => a - b,
                    Self::Mul => a * b,
                    Self::Div => a / b,
                };
                Float::from_f32(value)
            }
            (_, Some((a, b))) => {
                let value = match self {
                    Self::Add => a + b,
                    Self::Sub => a - b,
                    Self::Mul => a * b,
                    Self::Div => a / b,
                };
                Float::from_f64(value)
            }
            _ => return None,
        };
        (!value.is_nan()).then_some(value)
    }
}

/// `value` with its sign bit flipped and every other bit kept, for every bit pattern, NaNs
/// included.
pub(crate) fn negated(value: Float) -> Float {
    let sign = 1u64 << (value.width() - 1);
    let bits = value.bits() ^ sign;
    if value.width() == 32 {
        Float::from_bits32(bits as u32)
    } else {
        Float::from_bits64(bits)
    }
}

/// 2 to the power `n`, exactly, for `n` up to 1023.
fn pow2(n: u32) -> f64 {
    f64::from_bits((1023 + u64::from(n)) << 52)
}

/// `value` truncated toward zero, as a `width`-bit signed or unsigned integer; `None` if `value`
/// is a NaN or the truncated value is not representable (the cases where the conversion is
/// poison). The range is checked explicitly: Rust's `as` casts saturate.
pub(crate) fn float_to_int(value: Float, width: u32, signed: bool) -> Option<Int> {
    // An `f32` widens to `f64` exactly.
    let value = match (value.to_f64(), value.to_f32()) {
        (Some(value), _) => value,
        (_, Some(value)) => f64::from(value),
        _ => return None,
    };
    if value.is_nan() {
        return None;
    }
    let truncated = value.trunc();
    if signed {
        let limit = pow2(width - 1);
        (truncated >= -limit && truncated < limit)
            .then(|| Int::from_signed(width, truncated as i128))
            .flatten()
    } else {
        // `-0.5` truncates to `-0.0`, which is `0`: only a value of magnitude 1 or more is
        // negative.
        (truncated > -1.0 && truncated < pow2(width))
            .then(|| Int::from_unsigned(width, truncated as u128))
            .flatten()
    }
}

/// `value`, read as signed or unsigned, rounded to nearest, ties to even, as a float of `width`
/// bits (32 or 64).
pub(crate) fn int_to_float(value: Int, width: u32, signed: bool) -> Float {
    match (width, signed) {
        (32, true) => Float::from_f32(value.signed() as f32),
        (32, false) => Float::from_f32(value.unsigned() as f32),
        (_, true) => Float::from_f64(value.signed() as f64),
        (_, false) => Float::from_f64(value.unsigned() as f64),
    }
}

/// The float with the low `width` bits (32 or 64) of `bits`.
fn float_from_bits(width: u32, bits: u128) -> Float {
    if width == 32 {
        Float::from_bits32(bits as u32)
    } else {
        Float::from_bits64(bits as u64)
    }
}

/// A conversion between types, folded on a constant operand by [fold_conversion].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Conversion {
    /// A float to an integer, truncating toward zero; no fold where the result would be poison.
    FloatToInt { signed: bool },
    /// An integer to a float, rounding to nearest, ties to even.
    IntToFloat { signed: bool },
    /// A float to the other float width; no fold for a NaN.
    FloatResize,
    /// The same bits as the other type.
    Bitcast,
}

impl Conversion {
    /// The constant that `self` gives for the constant `operand` of type `from`, at the result
    /// type `to`; `None` if it does not fold.
    ///
    /// An `isize` operand or result folds only where the 64-bit and the 32-bit target agree
    /// (as `fold_isize` decides for integers): the value is computed for both widths, and the
    /// results must be the same value (for an `isize` result, the 64-bit one truncated to 32 bits
    /// must be the 32-bit one).
    fn apply(
        self,
        context: &Rc<Context>,
        operand: Option<&Option<AttributeRef>>,
        from: Type,
        to: Type,
    ) -> Option<AttributeRef> {
        let operand = operand?;
        match self {
            Self::FloatToInt { signed } => {
                let value = constant_float_operand(operand, from)?;
                let result = if to.is_isize() {
                    let wide = float_to_int(value, 64, signed)?;
                    let narrow = float_to_int(value, 32, signed)?;
                    (wide.trunc(32) == narrow).then_some(wide)?
                } else {
                    float_to_int(value, Int::width_for_type(to)?, signed)?
                };
                Some(context.int_attr(result, to).as_attribute_ref())
            }
            Self::IntToFloat { signed } => {
                let value = crate::fold::constant_operand(operand, from)?;
                let width = Float::width_for_type(to)?;
                let result = int_to_float(value, width, signed);
                if from.is_isize() && int_to_float(value.trunc(32), width, signed) != result {
                    return None;
                }
                Some(context.float_attr(result).as_attribute_ref())
            }
            Self::FloatResize => {
                let value = constant_float_operand(operand, from)?;
                let result = match (value.to_f32(), value.to_f64()) {
                    (Some(value), _) if Float::width_for_type(to) == Some(64) => {
                        Float::from_f64(f64::from(value))
                    }
                    (_, Some(value)) if Float::width_for_type(to) == Some(32) => {
                        Float::from_f32(value as f32)
                    }
                    _ => return None,
                };
                (!value.is_nan()).then(|| context.float_attr(result).as_attribute_ref())
            }
            Self::Bitcast => {
                if from.is_float() {
                    let value = constant_float_operand(operand, from)?;
                    let width = Int::width_for_type(to)?;
                    (width == value.width()).then(|| {
                        let int = Int::from_bits(width, u128::from(value.bits()));
                        context.int_attr(int, to).as_attribute_ref()
                    })
                } else {
                    let value = crate::fold::constant_operand(operand, from)?;
                    let width = Float::width_for_type(to)?;
                    (width == value.width()).then(|| {
                        context
                            .float_attr(float_from_bits(width, value.unsigned()))
                            .as_attribute_ref()
                    })
                }
            }
        }
    }
}

/// Folds the conversion `op` to the type `ty` on its constant operand (`operands`) with
/// `conversion`, pushing the constant on `results`.
///
/// It declines, pushing nothing, unless the result type is `ty` and `valid(context, from, to)`
/// says the type pair is one the operation verifies: an operation that does not verify does not
/// fold.
pub(crate) fn fold_conversion(
    op: &Operation,
    operands: &[Option<AttributeRef>],
    results: &mut SmallVec<[OpFoldResult; 1]>,
    ty: Type,
    valid: impl Fn(&Context, Type, Type) -> bool,
    conversion: Conversion,
) -> FoldResult {
    let context = op.context_rc();
    let from = crate::fold::operand_type(op);
    let to = crate::fold::result_type(op, 0);
    let folded = if to == ty && valid(&context, from, to) {
        conversion.apply(&context, operands.first(), from, to)
    } else {
        None
    };
    crate::fold::push_folded(results, folded.map(|attr| [attr]))
}

/// Implements [Foldable](runic_ir::traits::Foldable) for an operation of one float result, of the
/// operand type: `$f` is an expression, which may read the operation through `$op`, giving the
/// closure [fold_floats] computes the result with.
macro_rules! fold_float_one {
    ($Op:ty, |$op:ident| $f:expr) => {
        impl runic_ir::traits::Foldable for $Op {
            fn fold(
                &self,
                results: &mut runic_ir::SmallVec<[runic_ir::OpFoldResult; 1]>,
            ) -> runic_ir::FoldResult {
                let operands = $crate::fold::operand_constants(self.as_operation());
                self.fold_with(&operands, results)
            }

            fn fold_with(
                &self,
                operands: &[Option<runic_ir::AttributeRef>],
                results: &mut runic_ir::SmallVec<[runic_ir::OpFoldResult; 1]>,
            ) -> runic_ir::FoldResult {
                let $op = self;
                let op = self.as_operation();
                let operand_ty = $crate::fold::operand_type(op);
                let result_ty = $crate::fold::result_type(op, 0);
                // A verified operation has the type of its operands.
                let folded = if operand_ty == result_ty {
                    $crate::fold_float::fold_floats(
                        &op.context_rc(),
                        operands,
                        operand_ty,
                        result_ty,
                        $f,
                    )
                } else {
                    None
                };
                $crate::fold::push_folded(results, folded.map(|attr| [attr]))
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sum_is_computed_in_the_width_of_its_operands() {
        // `2^24 + 1` is not representable in `f32`: it rounds to `2^24`, as it would not in `f64`.
        let sum = FloatBinary::Add.apply(Float::from_f32(16777216.0), Float::from_f32(1.0));
        assert_eq!(sum, Some(Float::from_f32(16777216.0)));
        let sum = FloatBinary::Add.apply(Float::from_f64(16777216.0), Float::from_f64(1.0));
        assert_eq!(sum, Some(Float::from_f64(16777217.0)));
    }

    #[test]
    fn a_nan_result_is_no_value() {
        let inf = Float::from_f64(f64::INFINITY);
        assert_eq!(FloatBinary::Sub.apply(inf, inf), None);
        assert_eq!(FloatBinary::Div.apply(Float::from_f64(0.0), Float::from_f64(0.0)), None);
        assert_eq!(FloatBinary::Div.apply(Float::from_f64(1.0), Float::from_f64(0.0)), Some(inf));
        // Operands of different widths are not a verified operation.
        assert_eq!(FloatBinary::Add.apply(Float::from_f32(1.0), Float::from_f64(1.0)), None);
    }

    #[test]
    fn negation_flips_the_sign_bit_of_every_pattern() {
        for bits in [0u32, 0x8000_0000, 0x7fc0_0001, 0xffc0_0001, 0x7f80_0000, 1] {
            assert_eq!(negated(Float::from_bits32(bits)), Float::from_bits32(bits ^ 0x8000_0000));
        }
        for bits in [0u64, 1 << 63, 0x7ff8_0000_0000_0001, 0x7ff0_0000_0000_0000] {
            assert_eq!(negated(Float::from_bits64(bits)), Float::from_bits64(bits ^ (1 << 63)));
        }
    }
}
