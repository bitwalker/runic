use alloc::format;

use crate::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::{builtin::attributes::IntAttr, test::TestDialect},
    effects::*,
    traits::*,
    *,
};

/// Its operand, having trapped unless the operand, read as an unsigned number, lies in
/// `umin ..= umax`: `%r = test.check_range %x <{ umin = .., umax = .. }>;`.
///
/// The result reports its range through [InferIntRangeInterface] as the operand's range refined
/// by the checked one ([IntegerValueRange::refined_by]). The operation has no effects: a trap is
/// not an effect, so an unused check is dead code like any other operation that may trap.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, InferIntRangeInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct CheckRange {
    #[operand]
    operand: AnyInteger,
    #[result]
    result: AnyInteger,
    #[attr]
    umin: IntAttr,
    #[attr]
    umax: IntAttr,
}

impl CheckRange {
    /// The checked range, at the width of the operand's type: untracked for an `isize` operand.
    pub fn checked_range(&self) -> ConstantIntRange {
        let width = ConstantIntRange::max_for_type(self.operand().ty()).width();
        ConstantIntRange::from_unsigned(
            width,
            self.get_umin().unsigned(),
            self.get_umax().unsigned(),
        )
    }
}

impl InferTypeOpInterface for CheckRange {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.operand().ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// Both bounds have the operand's type, and `umin <= umax` read as unsigned numbers.
impl Verify<dyn InferTypeOpInterface> for CheckRange {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let ty = self.operand().ty();
        let span = self.as_operation().span();
        let name = self.as_operation().name();
        for (bound, attr_ty) in [("umin", self.umin().ty()), ("umax", self.umax().ty())] {
            if attr_ty != ty {
                return Err(crate::diagnostics::report!(
                    message: "invalid operation",
                    labels: [
                        primary(
                            span,
                            format!(
                                "the {bound} of '{name}' has type `{attr_ty}`, but its operand \
                                 has type `{ty}`"
                            )
                        ),
                    ],
                ));
            }
        }
        let (umin, umax) = (*self.get_umin(), *self.get_umax());
        if umin.width() != umax.width() || umin.unsigned() > umax.unsigned() {
            return Err(crate::diagnostics::report!(
                message: "invalid operation",
                labels: [
                    primary(
                        span,
                        format!(
                            "the umin of '{name}' ({}) is greater than its umax ({})",
                            umin.unsigned(),
                            umax.unsigned()
                        )
                    ),
                ],
            ));
        }
        Ok(())
    }
}

impl InferIntRangeInterface for CheckRange {
    fn infer_result_ranges(
        &self,
        operands: &[IntegerValueRange],
        set_result_range: &mut dyn FnMut(ValueRef, IntegerValueRange),
    ) {
        let Some(operand) = operands.first() else {
            return;
        };
        // An uninitialized operand gives an uninitialized result: joining it changes nothing, and
        // the analysis visits this operation again when the operand's range arrives.
        set_result_range(self.result().as_value_ref(), operand.refined_by(&self.checked_range()));
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use crate::{
        ConstantIntRange, Int, IntegerValueRange, SourceSpan, Type, ValueRef,
        dialects::test::{CheckRange, TestOpBuilder},
        testing::Test,
        traits::InferIntRangeInterface,
    };

    /// Builds `test.check_range` of the parameter of type `ty` and returns the operation's result.
    fn check_range(ty: fn(&crate::Context) -> Type, umin: u128, umax: u128) -> (Test, ValueRef) {
        let mut test = Test::new("f", move |cx| ([ty(cx)], [ty(cx)]));
        let param = test.entry_block().borrow().arguments()[0].borrow().as_value_ref();
        let width = Int::width_for_type(param.borrow().ty()).unwrap();
        let result = test
            .function_builder()
            .check_range(
                param,
                Int::from_unsigned(width, umin).unwrap(),
                Int::from_unsigned(width, umax).unwrap(),
                SourceSpan::UNKNOWN,
            )
            .unwrap();
        (test, result)
    }

    /// The range `test.check_range` reports for its result, given the operand range `operand`.
    fn infer(result: ValueRef, operand: IntegerValueRange) -> IntegerValueRange {
        let op = result.borrow().get_defining_op().unwrap();
        let op = op.borrow();
        let infer = op.as_trait::<dyn InferIntRangeInterface>().expect("infers ranges");
        let mut range = None;
        infer.infer_result_ranges(&[operand], &mut |value, inferred| {
            assert_eq!(value, result);
            assert!(range.is_none(), "one result range");
            range = Some(inferred);
        });
        range.expect("the result range is always set")
    }

    #[test]
    fn a_proven_operand_range_is_intersected_with_the_checked_range() {
        let (_test, result) = check_range(crate::Context::i32, 10, 100);
        let operand = IntegerValueRange::proven(ConstantIntRange::from_unsigned(32, 0, 50));
        assert_eq!(
            infer(result, operand),
            IntegerValueRange::proven(ConstantIntRange::from_unsigned(32, 10, 50))
        );
    }

    #[test]
    fn an_assumed_operand_range_is_replaced_by_the_checked_range() {
        let (test, result) = check_range(crate::Context::i32, 10, 100);
        let operand = IntegerValueRange::max_for_type(test.context().i32());
        assert_eq!(
            infer(result, operand),
            IntegerValueRange::proven(ConstantIntRange::from_unsigned(32, 10, 100))
        );
    }

    #[test]
    fn an_uninitialized_operand_gives_an_uninitialized_result() {
        let (_test, result) = check_range(crate::Context::i32, 10, 100);
        assert!(infer(result, IntegerValueRange::uninitialized()).is_uninitialized());
    }

    #[test]
    fn a_disjoint_proven_operand_range_gives_the_checked_range() {
        let (_test, result) = check_range(crate::Context::i8, 10, 20);
        let operand = IntegerValueRange::proven(ConstantIntRange::from_unsigned(8, 30, 40));
        assert_eq!(
            infer(result, operand),
            IntegerValueRange::proven(ConstantIntRange::from_unsigned(8, 10, 20))
        );
    }

    #[test]
    fn a_check_of_one_value_gives_an_exact_result() {
        let (test, result) = check_range(crate::Context::i64, 7, 7);
        let exact = IntegerValueRange::proven(ConstantIntRange::constant(Int::from(7i64)));
        assert_eq!(infer(result, IntegerValueRange::max_for_type(test.context().i64())), exact);
        let operand = IntegerValueRange::proven(ConstantIntRange::from_unsigned(64, 0, 100));
        assert_eq!(infer(result, operand), exact);
    }

    /// The range of an `isize` value is not tracked, so the checked range is untracked too: an
    /// assumed (untracked) operand gives a proven untracked result, which says nothing about its
    /// values; an uninitialized one stays uninitialized.
    #[test]
    fn an_isize_check_refines_by_an_untracked_range() {
        let (test, result) = check_range(crate::Context::isize, 0, 10);
        let op = result.borrow().get_defining_op().unwrap();
        let checked = op.borrow().downcast_ref::<CheckRange>().unwrap().checked_range();
        assert!(checked.is_untracked());
        let operand = IntegerValueRange::max_for_type(test.context().isize());
        assert_eq!(
            infer(result, operand),
            IntegerValueRange::proven(ConstantIntRange::untracked())
        );
        assert!(infer(result, IntegerValueRange::uninitialized()).is_uninitialized());
    }

    #[test]
    fn a_check_prints_and_verifies() {
        let (_test, result) = check_range(crate::Context::i8, 1, 200);
        let op = result.borrow().get_defining_op().unwrap();
        let op = op.borrow();
        assert!(op.verify().is_ok());
        let text = format!("{op}");
        assert!(text.contains("test.check_range"), "{text}");
        assert!(text.contains("umin = #builtin.int<1 : i8>"), "{text}");
        assert!(text.contains("umax = #builtin.int<-56 : i8>"), "{text}");
    }

    #[test]
    fn a_check_whose_bounds_are_reversed_does_not_verify() {
        let (_test, result) = check_range(crate::Context::i8, 200, 1);
        let op = result.borrow().get_defining_op().unwrap();
        assert!(op.borrow().verify().is_err());
    }

    #[test]
    fn a_check_whose_bounds_have_another_type_does_not_verify() {
        let (test, result) = check_range(crate::Context::i8, 1, 2);
        let context = test.context_rc();
        let mut op = result.borrow().get_defining_op().unwrap();
        let wide = context.int_attr(Int::from(1i32), context.i32());
        op.borrow_mut()
            .downcast_mut::<CheckRange>()
            .unwrap()
            .set_umin(crate::ReadyAttr(wide));
        assert!(op.borrow().verify().is_err());
    }

    /// The text form round trips, and both bounds are required.
    #[test]
    fn a_check_parses_from_text_and_both_bounds_are_required() {
        use alloc::rc::Rc;

        use crate::{Context, dialects::test::TestDialect};

        let function = |body: &str| {
            format!(
                "builtin.function public extern(\"C\") @f(%0: i8) -> i8 {{\n    {body}\n    \
                 builtin.ret %1 : (i8);\n}};"
            )
        };
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<TestDialect>();
        let source = function(
            "%1 = test.check_range %0 <{ umax = #builtin.int<-56 : i8>, umin = #builtin.int<1 : \
             i8> }>;",
        );
        let (parsed, printed) =
            crate::testing::parse_function_fixpoint(&context, "check_range.hir", &source).unwrap();
        assert!(printed.contains("test.check_range %0"), "{printed}");
        let entry = parsed.borrow().entry_block();
        let op = entry.borrow().front().unwrap();
        let checked = op.borrow().downcast_ref::<CheckRange>().unwrap().checked_range();
        assert_eq!(checked, ConstantIntRange::from_unsigned(8, 1, 200));

        let source = function("%1 = test.check_range %0 <{ umin = #builtin.int<1 : i8> }>;");
        assert!(
            crate::testing::parse_function_fixpoint(&context, "check_range.hir", &source).is_err()
        );
    }
}
