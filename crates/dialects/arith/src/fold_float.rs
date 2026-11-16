//! Constant folding of floating-point operations.
//!
//! Every fold of a float operation goes through [fold_floats] (one float result) or
//! [fold_float_compare] (an `i1`). They decline unless the operation's types are what a verified
//! operation has, read the operand constants, compute the result with a closure over
//! [Float]s, and create a new attribute for it.
//!
//! A fold never produces a NaN: which NaN an operation gives (its sign and payload) differs between
//! hosts and engines, so the closures return `None` for a result that is one, and the operation is
//! left to run. A comparison is deterministic, NaN operands included, and does fold.

use alloc::rc::Rc;

use runic_ir::{AttributeRef, Context, Float, Int, Type, dialects::builtin::attributes::FloatAttr};

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
