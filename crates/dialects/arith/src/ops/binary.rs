use alloc::format;

use runic_ir::{
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

            // The value result has the type of the operands. Each flag's `Bool` constraint is
            // checked by the generated verifier.
            impl Verify<dyn InferTypeOpInterface> for $Op {
                fn verify(&self, _context: &Context) -> Result<(), Report> {
                    let expected = self.lhs().ty();
                    let actual = self.result().ty();
                    if actual != expected {
                        return Err(runic_ir::diagnostics::report!(
                            message: "invalid operation",
                            labels: [
                                primary(
                                    self.as_operation().span(),
                                    format!(
                                        "the value result of '{}' has type `{actual}`, but its \
                                         operands have type `{expected}`",
                                        self.as_operation().name()
                                    )
                                ),
                            ],
                        ));
                    }
                    Ok(())
                }
            }
        }
    };
}

/// The unsigned range `0 ..= width - 1` of a shift amount of type `ty`. For a type whose range is
/// not tracked (`isize`, an integer wider than 128 bits) the requirement exists but cannot be
/// written as a range, and is [OperandRangeRequirement::Unexpressed].
fn shift_amount_requirement(ty: Type) -> OperandRangeRequirement {
    let width = ConstantIntRange::max_for_type(ty).width();
    if width == 0 {
        OperandRangeRequirement::Unexpressed
    } else {
        OperandRangeRequirement::Range(ConstantIntRange::from_unsigned(
            width,
            0,
            u128::from(width) - 1,
        ))
    }
}

/// The unsigned range `1 ..= 2^width - 1` of a divisor of type `ty` (every value but zero). For a
/// type whose range is not tracked (`isize`, an integer wider than 128 bits) the requirement exists
/// but cannot be written as a range, and is [OperandRangeRequirement::Unexpressed].
fn divisor_requirement(ty: Type) -> OperandRangeRequirement {
    let top = ConstantIntRange::max_for_type(ty);
    if top.is_untracked() {
        OperandRangeRequirement::Unexpressed
    } else {
        OperandRangeRequirement::Range(ConstantIntRange::from_unsigned(top.width(), 1, top.umax()))
    }
}

/// The requirement of a signed division on its dividend: `MIN / -1` traps, a condition on both
/// operands that no range of the dividend expresses.
fn signed_dividend_requirement(_ty: Type) -> OperandRangeRequirement {
    OperandRangeRequirement::Unexpressed
}

/// The requirement of nothing: every value is accepted.
fn no_requirement(_ty: Type) -> OperandRangeRequirement {
    OperandRangeRequirement::None
}

/// Implements [OperandRangeRequirementOpInterface] for `$Op`, whose operand 1 (`$operand`) must
/// lie in the range `$requirement` gives for its type, and whose operand 0 (`$lhs`) has the
/// requirement `$lhs_requirement` gives (none, unless given).
macro_rules! operand_one_requirement {
    ($Op:ty, $operand:ident, $requirement:ident) => {
        operand_one_requirement!($Op, $operand, $requirement, lhs, no_requirement);
    };
    ($Op:ty, $operand:ident, $requirement:ident, $lhs:ident, $lhs_requirement:ident) => {
        impl OperandRangeRequirementOpInterface for $Op {
            fn operand_range_requirement(&self, index: usize) -> OperandRangeRequirement {
                match index {
                    0 => $lhs_requirement(self.$lhs().ty()),
                    1 => $requirement(self.$operand().ty()),
                    _ => OperandRangeRequirement::None,
                }
            }
        }
    };
}

/// Implements [OperandRangeRequirementOpInterface] for `$Op`, an operation with an `overflow`
/// property: in the wrapping mode every value is accepted; in a checked or unchecked mode the
/// operation traps or is undefined unless the exact result fits, a condition on both operands that
/// no range of one expresses, so both are [OperandRangeRequirement::Unexpressed].
macro_rules! overflow_requirement {
    ($Op:ty) => {
        impl OperandRangeRequirementOpInterface for $Op {
            fn operand_range_requirement(&self, index: usize) -> OperandRangeRequirement {
                match (index, *self.get_overflow()) {
                    (0 | 1, Overflow::Checked(_) | Overflow::Unchecked(_)) => {
                        OperandRangeRequirement::Unexpressed
                    }
                    _ => OperandRangeRequirement::None,
                }
            }
        }
    };
}

/// Two's complement sum.
///
/// `overflow` says what happens when the exact sum does not fit: the result wraps (the default),
/// the operation traps, or the result is undefined, with the operands read as signed or as unsigned
/// numbers for the latter two. The text leaves the default out: `arith.add %0, %1` wraps, and
/// `arith.add %0, %1 <{ overflow = #builtin.overflow<checked signed> }>` traps.
///
/// Whether the exact sum fits depends on both operands, so in a checked or unchecked mode the
/// requirement on each operand is unexpressed ([OperandRangeRequirementOpInterface]); in the
/// wrapping mode there is none.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, InferIntRangeInterface, MemoryEffectOpInterface, OperandRangeRequirementOpInterface, Foldable, OpPrinter)
)]
pub struct Add {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    #[default]
    overflow: OverflowAttr,
}

infer_return_ty_for_binary_op!(Add);
overflow_requirement!(Add);

impl InferIntRangeInterface for Add {
    fn infer_result_ranges(
        &self,
        operands: &[IntegerValueRange],
        set_result_range: &mut dyn FnMut(ValueRef, IntegerValueRange),
    ) {
        let [lhs, rhs] = operands else {
            return;
        };
        let (Some(lhs), Some(rhs)) = (lhs.range(), rhs.range()) else {
            return;
        };
        let range = infer_add_range(self.result().ty(), lhs, rhs, *self.get_overflow());
        set_result_range(self.result().as_value_ref(), range);
    }
}

/// The range of the sum of `lhs` and `rhs`, of type `ty`, under the overflow mode `overflow`.
fn infer_add_range(
    ty: Type,
    lhs: &ConstantIntRange,
    rhs: &ConstantIntRange,
    overflow: Overflow,
) -> IntegerValueRange {
    let result_top = ConstantIntRange::max_for_type(ty);
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
        .map(IntegerValueRange::new)
        .unwrap_or_else(|| IntegerValueRange::new(result_top))
}

/// Two's complement sum with an overflow flag: the first result says whether the exact sum does not
/// fit, with the operands read as `signedness` says; the second is the wrapped sum.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OperandRangeRequirementOpInterface, Foldable, OpPrinter)
)]
pub struct Sub {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    #[default]
    overflow: OverflowAttr,
}

infer_return_ty_for_binary_op!(Sub);
overflow_requirement!(Sub);

/// Two's complement difference with an overflow flag, as for [AddOverflowing].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OperandRangeRequirementOpInterface, Foldable, OpPrinter)
)]
pub struct Mul {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    #[default]
    overflow: OverflowAttr,
}

infer_return_ty_for_binary_op!(Mul);
overflow_requirement!(Mul);

/// Two's complement product with an overflow flag, as for [AddOverflowing].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
///
/// A trap is not an effect: an unused division may be removed with its trap, and a division is
/// never moved to where it would execute more often (see the crate documentation). The divisor's
/// requirement is stated through [OperandRangeRequirementOpInterface].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
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
operand_one_requirement!(Udiv, rhs, divisor_requirement);

/// Signed integer division, rounding toward zero. Traps on a zero divisor and on dividing the
/// minimum signed value by -1.
///
/// A trap is not an effect: an unused division may be removed with its trap, and a division is
/// never moved to where it would execute more often (see the crate documentation). The divisor's
/// requirement is stated through [OperandRangeRequirementOpInterface]; the dividend's is
/// unexpressed, since `MIN / -1` relates the two operands.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
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
operand_one_requirement!(Sdiv, rhs, divisor_requirement, lhs, signed_dividend_requirement);

/// Unsigned integer remainder. Traps on a zero divisor.
///
/// A trap is not an effect: an unused division may be removed with its trap, and a division is
/// never moved to where it would execute more often (see the crate documentation). The divisor's
/// requirement is stated through [OperandRangeRequirementOpInterface].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
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
operand_one_requirement!(Urem, rhs, divisor_requirement);

/// Signed integer remainder of the division rounding toward zero: the result has the sign of the
/// dividend (`lhs`). Traps on a zero divisor only: the remainder of the minimum signed value by -1
/// is `0`.
///
/// A trap is not an effect: an unused division may be removed with its trap, and a division is
/// never moved to where it would execute more often (see the crate documentation). The divisor's
/// requirement is stated through [OperandRangeRequirementOpInterface].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
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
operand_one_requirement!(Srem, rhs, divisor_requirement);

/// Unsigned integer division and remainder, as [Udiv] and [Urem]: the first result is the
/// quotient, the second the remainder. Traps on a zero divisor.
///
/// A trap is not an effect: an unused division may be removed with its trap, and a division is
/// never moved to where it would execute more often (see the crate documentation). The divisor's
/// requirement is stated through [OperandRangeRequirementOpInterface].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
)]
pub struct Udivrem {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    quotient: AnyInteger,
    #[result]
    remainder: AnyInteger,
}

operand_one_requirement!(Udivrem, rhs, divisor_requirement);

impl InferTypeOpInterface for Udivrem {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let lhs = self.lhs().ty();
        self.quotient_mut().set_type(lhs);
        self.remainder_mut().set_type(lhs);
        Ok(())
    }
}

/// Signed integer division and remainder, as [Sdiv] and [Srem]: the first result is the quotient,
/// the second the remainder, which has the sign of the dividend (`lhs`). Traps on a zero divisor,
/// and on dividing the minimum signed value by -1, because the quotient does not fit (the remainder
/// alone would be `0`).
///
/// A trap is not an effect: an unused division may be removed with its trap, and a division is
/// never moved to where it would execute more often (see the crate documentation). The divisor's
/// requirement is stated through [OperandRangeRequirementOpInterface]; the dividend's is
/// unexpressed, since `MIN / -1` relates the two operands.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
)]
pub struct Sdivrem {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    quotient: AnyInteger,
    #[result]
    remainder: AnyInteger,
}

operand_one_requirement!(Sdivrem, rhs, divisor_requirement, lhs, signed_dividend_requirement);

impl InferTypeOpInterface for Sdivrem {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let lhs = self.lhs().ty();
        self.quotient_mut().set_type(lhs);
        self.remainder_mut().set_type(lhs);
        Ok(())
    }
}

/// Bitwise AND. On `i1`, logical AND.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
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
operand_one_requirement!(Shl, shift, shift_amount_requirement);

/// Logical shift right, filling with zeros.
///
/// The amount is as for [Shl].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
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
operand_one_requirement!(Shr, shift, shift_amount_requirement);

/// Arithmetic shift right, filling with copies of the sign bit.
///
/// The amount is as for [Shl].
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        Foldable,
        OpPrinter
    )
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
operand_one_requirement!(Ashr, shift, shift_amount_requirement);

/// Rotate left.
///
/// The amount has the type of the value, is read as an unsigned number, and is taken modulo the
/// width of the type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
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

// Constant folds. Each closure is written against the width of the operands it is given; see
// `crate::fold` for how `isize` operands are folded.

/// The exact result of an operation in the overflow mode `overflow`: the wrapped one for
/// [Overflow::Wrapping], and for a checked or unchecked mode the result if it fits read with the
/// mode's signedness, `None` (no fold: a trap or an undefined result) if it does not.
fn with_overflow(
    overflow: Overflow,
    lhs: Int,
    rhs: Int,
    wrapping: fn(Int, Int) -> Int,
    signed: fn(Int, Int) -> Option<Int>,
    unsigned: fn(Int, Int) -> Option<Int>,
) -> Option<Int> {
    match overflow.signedness() {
        None => Some(wrapping(lhs, rhs)),
        Some(Signedness::Signed) => signed(lhs, rhs),
        Some(Signedness::Unsigned) => unsigned(lhs, rhs),
    }
}

fold_one!(Add, Operand, |op| {
    let overflow = *op.get_overflow();
    move |[a, b]| {
        with_overflow(
            overflow,
            a,
            b,
            Int::wrapping_add,
            Int::checked_add_signed,
            Int::checked_add_unsigned,
        )
    }
});

fold_one!(Sub, Operand, |op| {
    let overflow = *op.get_overflow();
    move |[a, b]| {
        with_overflow(
            overflow,
            a,
            b,
            Int::wrapping_sub,
            Int::checked_sub_signed,
            Int::checked_sub_unsigned,
        )
    }
});

fold_one!(Mul, Operand, |op| {
    let overflow = *op.get_overflow();
    move |[a, b]| {
        with_overflow(
            overflow,
            a,
            b,
            Int::wrapping_mul,
            Int::checked_mul_signed,
            Int::checked_mul_unsigned,
        )
    }
});

/// Implements [Foldable] for the overflowing operation `$Op`: the flag, as an `i1`, and the
/// wrapped value, from `$signed` or `$unsigned` as the operation's `signedness` says.
macro_rules! fold_overflowing {
    ($Op:ty, $signed:expr, $unsigned:expr) => {
        fold_two!($Op, [Bool, Operand], |op| {
            let compute: fn(Int, Int) -> (Int, bool) = match *op.get_signedness() {
                Signedness::Signed => $signed,
                Signedness::Unsigned => $unsigned,
            };
            (
                move |[a, b]: [Int; 2]| Some(Int::from(compute(a, b).1)),
                move |[a, b]: [Int; 2]| Some(compute(a, b).0),
            )
        });
    };
}

fold_overflowing!(AddOverflowing, Int::overflowing_add_signed, Int::overflowing_add_unsigned);
fold_overflowing!(SubOverflowing, Int::overflowing_sub_signed, Int::overflowing_sub_unsigned);
fold_overflowing!(MulOverflowing, Int::overflowing_mul_signed, Int::overflowing_mul_unsigned);

// `div_*` and `rem_*` are `None` for a zero divisor, and `div_signed` for `MIN / -1`.
// `rem_signed` gives 0 for `MIN % -1`, which is what `srem` gives. `sdivrem` traps there because of
// its quotient, so its remainder closure declines when the division does: each closure is right on
// its own, whichever runs first.
fold_one!(Udiv, Operand, |_op| |[a, b]| a.div_unsigned(b));
fold_one!(Sdiv, Operand, |_op| |[a, b]| a.div_signed(b));
fold_one!(Urem, Operand, |_op| |[a, b]| a.rem_unsigned(b));
fold_one!(Srem, Operand, |_op| |[a, b]| a.rem_signed(b));
fold_two!(Udivrem, [Operand, Operand], |_op| (
    |[a, b]: [Int; 2]| a.div_unsigned(b),
    |[a, b]: [Int; 2]| a.rem_unsigned(b)
));
fold_two!(Sdivrem, [Operand, Operand], |_op| (
    |[a, b]: [Int; 2]| a.div_signed(b),
    |[a, b]: [Int; 2]| a.div_signed(b).and(a.rem_signed(b))
));

fold_one!(Band, Operand, |_op| |[a, b]| Some(a & b));
fold_one!(Bor, Operand, |_op| |[a, b]| Some(a | b));
fold_one!(Bxor, Operand, |_op| |[a, b]| Some(a ^ b));

/// `shift` on `value` by the unsigned value of `amount`, or `None` (no fold: the result is
/// undefined) if the amount is the width or more.
fn shift(value: Int, amount: Int, shift: fn(Int, u32) -> Int) -> Option<Int> {
    let amount = amount.shift_amount_unsigned();
    (amount < value.width()).then(|| shift(value, amount))
}

fold_one!(Shl, Operand, |_op| |[a, b]| shift(a, b, |value, amount| value << amount));
fold_one!(Shr, Operand, |_op| |[a, b]| shift(a, b, Int::lshr));
fold_one!(Ashr, Operand, |_op| |[a, b]| shift(a, b, Int::ashr));
fold_one!(Rotl, Operand, |_op| |[a, b]| Some(a.rotl(b.rotate_amount_unsigned(a.width()))));
fold_one!(Rotr, Operand, |_op| |[a, b]| Some(a.rotr(b.rotate_amount_unsigned(a.width()))));

fold_one!(Min, Operand, |op| {
    let min = match *op.get_signedness() {
        Signedness::Signed => Int::min_signed,
        Signedness::Unsigned => Int::min_unsigned,
    };
    move |[a, b]| Some(min(a, b))
});

fold_one!(Max, Operand, |op| {
    let max = match *op.get_signedness() {
        Signedness::Signed => Int::max_signed,
        Signedness::Unsigned => Int::max_unsigned,
    };
    move |[a, b]| Some(max(a, b))
});

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
    fn adding_zero_or_one_to_a_full_width_or_small_range() {
        let context = Context::default();
        let i32 = context.i32();
        let top = ConstantIntRange::max_for_type(i32);
        let small = ConstantIntRange::from_unsigned(32, 0, 10);
        let add = |lhs: &ConstantIntRange, rhs: i32, overflow| {
            infer_add_range(i32, lhs, &ConstantIntRange::constant(Int::from(rhs)), overflow)
        };
        let range = |range| IntegerValueRange::new(range);

        // Adding zero changes nothing in any mode.
        for overflow in ALL_MODES {
            assert_eq!(add(&top, 0, overflow), range(top), "top + 0, {overflow}");
        }
        for overflow in ALL_MODES {
            assert_eq!(add(&small, 0, overflow), range(small), "[0, 10] + 0, {overflow}");
        }
        // A small range plus one cannot overflow, so every mode gives [1, 11].
        for overflow in ALL_MODES {
            assert_eq!(
                add(&small, 1, overflow),
                range(ConstantIntRange::from_unsigned(32, 1, 11)),
                "[0, 10] + 1, {overflow}"
            );
        }
        // A wrapping sum may wrap to any value, so the result is the full range.
        assert_eq!(add(&top, 1, Overflow::Wrapping), range(top), "top + 1, wrapping");
        // A checked sum of the full range plus one overflows for one operand value, which traps,
        // so the exact result range would exclude one end. The interval sum only narrows when its
        // exact bounds fit the type, and here the upper bound does not, so the result is the full
        // range: sound, though not the tightest.
        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            assert_eq!(
                add(&top, 1, Overflow::Checked(signedness)),
                range(top),
                "top + 1, checked {signedness:?}"
            );
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
                );
                assert_eq!(
                    result.range().map(|r| (r.umin(), r.umax())),
                    Some((expected, expected)),
                    "{overflow}"
                );
            }
        }
        // Read as unsigned, both sums overflow; an unchecked addition must not acquire the
        // wrapped result as a proof.
        for (lhs, rhs) in [(u32::MAX, 1u32), (u32::MAX - 1, 2)] {
            let result = infer_add_range(
                i32,
                &ConstantIntRange::constant(Int::from(lhs)),
                &ConstantIntRange::constant(Int::from(rhs)),
                Overflow::Unchecked(Signedness::Unsigned),
            );
            assert_eq!(
                result,
                IntegerValueRange::max_for_type(i32),
                "unchecked overflow must not acquire a wrapped result"
            );
        }
        // Read as signed, `-1 + 1` fits, so an unchecked signed addition has the exact sum.
        let result = infer_add_range(
            i32,
            &ConstantIntRange::constant(Int::from(-1i32)),
            &ConstantIntRange::constant(Int::from(1i32)),
            Overflow::Unchecked(Signedness::Signed),
        );
        assert_eq!(result, IntegerValueRange::new(ConstantIntRange::constant(Int::from(0i32))));
        // `true + true` overflows an `i1` read either way.
        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            let boolean_overflow = infer_add_range(
                context.i1(),
                &ConstantIntRange::constant(Int::from(true)),
                &ConstantIntRange::constant(Int::from(true)),
                Overflow::Unchecked(signedness),
            );
            assert_eq!(boolean_overflow, IntegerValueRange::max_for_type(context.i1()));
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
            );
            assert!(result.range().is_some_and(ConstantIntRange::is_untracked), "{overflow}");
        }
    }
}
