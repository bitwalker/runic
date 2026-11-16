use crate::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::{builtin::attributes::OverflowAttr, test::TestDialect},
    effects::*,
    traits::*,
    *,
};

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

    ($Op:ty as $manually_specified_ty:ident) => {
        impl InferTypeOpInterface for $Op {
            fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
                self.result_mut().set_type(context.$manually_specified_ty());
                Ok(())
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

/// Two's complement sum. `overflow` says what happens when the exact sum does not fit: the result
/// wraps (the default), the operation traps (`checked`), or the result is undefined (`unchecked`).
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        OpPrinter
    )
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

/// Two's complement product, with an overflow mode as for [Add].
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(BinaryOp, Commutative, SameTypeOperands),
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        OpPrinter
    )
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

/// Bitwise shift-left
///
/// Shifts larger than the bitwidth of the value will be wrapped to zero.
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(BinaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Shl {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    shift: Int32,
    #[result]
    result: AnyInteger,
}

infer_return_ty_for_binary_op!(Shl);

/// Equality comparison
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(BinaryOp, Commutative, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Eq {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: Bool,
}

infer_return_ty_for_binary_op!(Eq as i1);

/// Inequality comparison
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(BinaryOp, Commutative, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Neq {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: Bool,
}

infer_return_ty_for_binary_op!(Neq as i1);

#[cfg(test)]
mod tests {
    use crate::{
        BuilderExt, Overflow, Signedness, SourceSpan, ValueRef,
        dialects::test::{Add, Mul, TestOpBuilder},
        testing::Test,
        traits::{OperandRangeRequirement, operation_operand_range_requirement},
    };

    /// What the operation defining `result` requires of its two operands.
    fn requirements(result: ValueRef) -> [OperandRangeRequirement; 2] {
        let op = result.borrow().get_defining_op().unwrap();
        let op = op.borrow();
        [0, 1].map(|index| operation_operand_range_requirement(&op, index))
    }

    /// `test.add` and `test.mul` accept every value when they wrap; in a checked or unchecked mode
    /// whether the result fits depends on both operands, so both requirements are unexpressed.
    #[test]
    fn checked_and_unchecked_add_and_mul_state_an_unexpressed_requirement() {
        use OperandRangeRequirement::{None as Any, Unexpressed};

        const SPAN: SourceSpan = SourceSpan::UNKNOWN;
        let mut test = Test::new("f", |cx| ([cx.i32(), cx.i32()], []));
        let (a, b) = {
            let entry = test.entry_block();
            let entry = entry.borrow();
            (entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef)
        };
        let mut builder = test.function_builder();
        let built = [
            (Any, builder.add_wrapping(a, b, SPAN).unwrap()),
            (Unexpressed, builder.add(a, b, SPAN).unwrap()),
            (Unexpressed, builder.add_unchecked(a, b, SPAN).unwrap()),
            (Any, builder.mul_wrapping(a, b, SPAN).unwrap()),
            (Unexpressed, builder.mul(a, b, SPAN).unwrap()),
            (Unexpressed, builder.mul_unchecked(a, b, SPAN).unwrap()),
        ];
        let signed = Overflow::Checked(Signedness::Signed);
        let add = builder.builder_mut().create::<Add, (_, _, _)>(SPAN)(a, b, signed).unwrap();
        let mul = builder.builder_mut().create::<Mul, (_, _, _)>(SPAN)(a, b, signed).unwrap();
        for (expected, result) in built.into_iter().chain([
            (Unexpressed, add.borrow().result().as_value_ref()),
            (Unexpressed, mul.borrow().result().as_value_ref()),
        ]) {
            assert_eq!(requirements(result), [expected, expected], "{}", result.borrow());
        }
    }
}
