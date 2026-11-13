use runic_core::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::{OverflowAttr, SignednessAttr},
    effects::*,
    traits::*,
    *,
};

use crate::ArithDialect;

/// Implements [InferTypeOpInterface] for a binary operation whose results have the type of `lhs`,
/// except the results named after `,`, which are `i1`.
macro_rules! infer_return_ty_for_binary_op {
    ($Op:ty) => {
        impl InferTypeOpInterface for $Op {
            fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
                let lhs = self.lhs().ty();
                self.result_mut().set_type(lhs);
                Ok(())
            }
        }
    };

    ($Op:ty, $($flag:ident),+) => {
        paste::paste! {
            impl InferTypeOpInterface for $Op {
                fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
                    let lhs = self.lhs().ty();
                    self.result_mut().set_type(lhs);
                    $(
                        self.[<$flag _mut>]().set_type(context.i1());
                    )*
                    Ok(())
                }
            }
        }
    };
}

/// Two's complement sum.
///
/// `overflow` says what happens when the exact sum does not fit: the result wraps (the default),
/// the operation traps, or the result is undefined, with the operands read as signed or as unsigned
/// numbers for the latter two.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, InferIntRangeInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Add {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    overflow: OverflowAttr,
}

infer_return_ty_for_binary_op!(Add);

impl InferIntRangeInterface for Add {
    fn infer_result_ranges(
        &self,
        operands: &[IntegerValueRange],
        set_result_range: &mut dyn FnMut(ValueRef, IntegerValueRange),
    ) {
        let [lhs, rhs] = operands else {
            return;
        };
        let Some(confidence) = lhs.join_confidence(rhs) else {
            return;
        };
        let (Some(lhs), Some(rhs)) = (lhs.range(), rhs.range()) else {
            return;
        };
        let range = infer_add_range(self.result().ty(), lhs, rhs, *self.get_overflow(), confidence);
        set_result_range(self.result().as_value_ref(), range);
    }
}

/// The range of the sum of `lhs` and `rhs`, of type `ty`, under the overflow mode `overflow`.
fn infer_add_range(
    ty: Type,
    lhs: &ConstantIntRange,
    rhs: &ConstantIntRange,
    overflow: Overflow,
    confidence: IntegerRangeConfidence,
) -> IntegerValueRange {
    let result_top = ConstantIntRange::max_for_type(ty);
    // Checked and wrapping operations establish the result domain independently of their
    // operands. Assumptions cannot prove a narrower bound, but must not hide that guarantee.
    if confidence == IntegerRangeConfidence::Assumed && !overflow.is_unchecked() {
        return IntegerValueRange::proven(result_top);
    }
    let precise = if lhs.width() != result_top.width() || rhs.width() != result_top.width() {
        None
    } else {
        match overflow {
            Overflow::Wrapping => lhs.add_wrapping(rhs),
            Overflow::Checked(Signedness::Signed) | Overflow::Unchecked(Signedness::Signed) => {
                lhs.add_signed_without_overflow(rhs)
            }
            Overflow::Checked(Signedness::Unsigned) | Overflow::Unchecked(Signedness::Unsigned) => {
                lhs.add_unsigned_without_overflow(rhs)
            }
        }
    };
    precise
        .filter(|range| result_top.union(range) == result_top)
        .map(|range| IntegerValueRange::with_confidence(range, confidence))
        .unwrap_or_else(|| {
            if overflow.is_unchecked() {
                IntegerValueRange::assumed(result_top)
            } else {
                IntegerValueRange::proven(result_top)
            }
        })
}

/// Two's complement sum with an overflow flag: the first result says whether the exact sum does not
/// fit, with the operands read as `signedness` says; the second is the wrapped sum.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct AddOverflowing {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    overflowed: Bool,
    #[result]
    result: AnyInteger,
    #[attr]
    signedness: SignednessAttr,
}

infer_return_ty_for_binary_op!(AddOverflowing, overflowed);

/// Two's complement difference, with an overflow mode as for [Add].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Sub {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    overflow: OverflowAttr,
}

infer_return_ty_for_binary_op!(Sub);

/// Two's complement difference with an overflow flag, as for [AddOverflowing].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct SubOverflowing {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    overflowed: Bool,
    #[result]
    result: AnyInteger,
    #[attr]
    signedness: SignednessAttr,
}

infer_return_ty_for_binary_op!(SubOverflowing, overflowed);

/// Two's complement product, with an overflow mode as for [Add].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Mul {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    overflow: OverflowAttr,
}

infer_return_ty_for_binary_op!(Mul);

/// Two's complement product with an overflow flag, as for [AddOverflowing].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct MulOverflowing {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    overflowed: Bool,
    #[result]
    result: AnyInteger,
    #[attr]
    signedness: SignednessAttr,
}

infer_return_ty_for_binary_op!(MulOverflowing, overflowed);

/// Unsigned integer division, rounding toward zero. Traps on a zero divisor.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Udiv {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Udiv);

/// Signed integer division, rounding toward zero. Traps on a zero divisor and on dividing the
/// minimum signed value by -1.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Sdiv {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Sdiv);

/// Unsigned integer remainder. Traps on a zero divisor.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Urem {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Urem);

/// Signed integer remainder of the division rounding toward zero: the result has the sign of the
/// dividend (`lhs`). Traps on a zero divisor and on dividing the minimum signed value by -1.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Srem {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Srem);

/// Unsigned integer division and remainder, as [Udiv] and [Urem]: the first result is the
/// remainder, the second the quotient. Traps on a zero divisor.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Udivrem {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    remainder: AnyInteger,
    #[result]
    quotient: AnyInteger,
}

impl InferTypeOpInterface for Udivrem {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let lhs = self.lhs().ty();
        self.remainder_mut().set_type(lhs);
        self.quotient_mut().set_type(lhs);
        Ok(())
    }
}

/// Signed integer division and remainder, as [Sdiv] and [Srem]: the first result is the remainder,
/// which has the sign of the dividend (`lhs`), the second the quotient. Traps on a zero divisor and
/// on dividing the minimum signed value by -1.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Sdivrem {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    remainder: AnyInteger,
    #[result]
    quotient: AnyInteger,
}

impl InferTypeOpInterface for Sdivrem {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let lhs = self.lhs().ty();
        self.remainder_mut().set_type(lhs);
        self.quotient_mut().set_type(lhs);
        Ok(())
    }
}

/// Bitwise AND. On `i1`, logical AND.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Band {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Band);

/// Bitwise OR. On `i1`, logical OR.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Bor {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Bor);

/// Bitwise XOR. On `i1`, logical XOR.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Bxor {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Bxor);

/// Shift left, filling with zeros.
///
/// The amount has the type of the value and is read as an unsigned number; the result is
/// undefined if it is the width of the type or more.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Shl {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    shift: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Shl);

/// Logical shift right, filling with zeros.
///
/// The amount is as for [Shl].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Shr {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    shift: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Shr);

/// Arithmetic shift right, filling with copies of the sign bit.
///
/// The amount is as for [Shl].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Ashr {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    shift: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Ashr);

/// Rotate left.
///
/// The amount has the type of the value, is read as an unsigned number, and is taken modulo the
/// width of the type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Rotl {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    shift: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Rotl);

/// Rotate right.
///
/// The amount is as for [Rotl].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Rotr {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    shift: AnyInteger,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Rotr);

/// The smaller operand, with the operands read as `signedness` says.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Min {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    signedness: SignednessAttr,
}

infer_return_ty_for_binary_op!(Min);

/// The larger operand, with the operands read as `signedness` says.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Max {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    signedness: SignednessAttr,
}

infer_return_ty_for_binary_op!(Max);

#[cfg(test)]
mod integer_range_tests {
    use super::*;

    const ALL_MODES: [Overflow; 5] = [
        Overflow::Wrapping,
        Overflow::Checked(Signedness::Signed),
        Overflow::Checked(Signedness::Unsigned),
        Overflow::Unchecked(Signedness::Signed),
        Overflow::Unchecked(Signedness::Unsigned),
    ];

    #[test]
    fn checked_and_wrapping_add_prove_the_domain_without_proving_assumptions() {
        let context = Context::default();
        let i32 = context.i32();
        for lhs in [ConstantIntRange::max_for_type(i32), ConstantIntRange::from_unsigned(32, 0, 10)]
        {
            for rhs in [0i32, 1] {
                for overflow in ALL_MODES.into_iter().filter(|mode| !mode.is_unchecked()) {
                    let result = infer_add_range(
                        i32,
                        &lhs,
                        &ConstantIntRange::constant(Int::from(rhs)),
                        overflow,
                        IntegerRangeConfidence::Assumed,
                    );
                    assert_eq!(
                        result,
                        IntegerValueRange::proven(ConstantIntRange::max_for_type(i32))
                    );
                }
            }
        }
    }

    #[test]
    fn addition_distinguishes_checked_wrapping_and_unchecked_modes() {
        let context = Context::default();
        let i32 = context.i32();
        // Each sum fits when read with the signedness of its checked mode.
        let cases = [
            (Signedness::Unsigned, Int::from(0x7fff_ffffi32), Int::from(1i32), 0x8000_0000),
            (Signedness::Signed, Int::from(-1i32), Int::from(1i32), 0),
        ];
        for (signedness, lhs, rhs, expected) in cases {
            for overflow in [Overflow::Checked(signedness), Overflow::Wrapping] {
                let result = infer_add_range(
                    i32,
                    &ConstantIntRange::constant(lhs),
                    &ConstantIntRange::constant(rhs),
                    overflow,
                    IntegerRangeConfidence::Proven,
                );
                assert_eq!(
                    result.range().map(|r| (r.umin(), r.umax())),
                    Some((expected, expected)),
                    "{overflow}"
                );
                assert!(result.is_proven());
            }
        }
        // Read as unsigned, both sums overflow; an unchecked addition must not acquire the
        // wrapped result as a proof.
        for lhs in [Int::from(u32::MAX), Int::from(-1i32)] {
            let result = infer_add_range(
                i32,
                &ConstantIntRange::constant(lhs),
                &ConstantIntRange::constant(Int::from(1i32)),
                Overflow::Unchecked(Signedness::Unsigned),
                IntegerRangeConfidence::Proven,
            );
            assert!(!result.is_proven(), "unchecked overflow must not acquire a wrapped proof");
        }
        // Read as signed, `-1 + 1` fits, so an unchecked signed addition has the exact sum.
        let result = infer_add_range(
            i32,
            &ConstantIntRange::constant(Int::from(-1i32)),
            &ConstantIntRange::constant(Int::from(1i32)),
            Overflow::Unchecked(Signedness::Signed),
            IntegerRangeConfidence::Proven,
        );
        assert_eq!(result, IntegerValueRange::proven(ConstantIntRange::constant(Int::from(0i32))));
        // `true + true` overflows an `i1` read either way.
        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            let boolean_overflow = infer_add_range(
                context.i1(),
                &ConstantIntRange::constant(Int::from(true)),
                &ConstantIntRange::constant(Int::from(true)),
                Overflow::Unchecked(signedness),
                IntegerRangeConfidence::Proven,
            );
            assert!(!boolean_overflow.is_proven());
        }
    }

    #[test]
    fn isize_addition_is_untracked() {
        let context = Context::default();
        let isize = context.isize();
        for overflow in ALL_MODES {
            let result = infer_add_range(
                isize,
                &ConstantIntRange::constant(Int::from(1i64)),
                &ConstantIntRange::constant(Int::from(2i64)),
                overflow,
                IntegerRangeConfidence::Proven,
            );
            assert!(result.range().is_some_and(ConstantIntRange::is_untracked), "{overflow}");
        }
    }
}
