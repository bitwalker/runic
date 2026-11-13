use alloc::rc::Rc;

use runic_core::{
    attributes::IntegerLikeAttr,
    matchers::Matcher,
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind, RewritePattern},
    *,
};
use runic_dialect_arith::ArithOpBuilder;

use crate::*;

/// Simplifies a `cf.select` between the boolean constants into the condition itself:
///
/// * `select(c, true, false)` becomes `c`
/// * `select(c, false, true)` becomes `bnot c`
///
/// Such selects are produced when control flow that merely forwards a flag is converted to
/// selects, and each one costs more than the condition, which is already at hand.
pub struct SimplifySelectOfBools {
    info: PatternInfo,
}

impl SimplifySelectOfBools {
    /// Creates the pattern, matching `cf.select`.
    pub fn new(context: Rc<Context>) -> Self {
        let select_op = context
            .get_or_register_dialect::<ControlFlowDialect>()
            .expect_registered_name::<Select>();
        Self {
            info: PatternInfo::new(
                context,
                "simplify-select-of-bools",
                PatternKind::Operation(select_op),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for SimplifySelectOfBools {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for SimplifySelectOfBools {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let (span, cond, first, second) = {
            let op = operation.borrow();
            let Some(select) = op.downcast_ref::<Select>() else {
                return Ok(false);
            };
            if select.result().ty().integer_width() != Some(1) {
                return Ok(false);
            }
            let constant_bool = |value: ValueRef| -> Option<bool> {
                let defining_op = value.borrow().get_defining_op()?;
                let attr = matchers::constant().matches(&*defining_op.borrow())?;
                let attr = attr.borrow();
                let value = attr.as_attr().as_trait::<dyn IntegerLikeAttr>()?.as_int();
                Some(value.is_one())
            };
            let (Some(first), Some(second)) = (
                constant_bool(select.first().as_value_ref()),
                constant_bool(select.second().as_value_ref()),
            ) else {
                return Ok(false);
            };
            (op.span(), select.cond().as_value_ref(), first, second)
        };

        let replacement = match (first, second) {
            (true, false) => cond,
            (false, true) => {
                rewriter.set_insertion_point_before(operation);
                rewriter.bnot(cond, span)?
            }
            // Arms with equal constants are left alone: `Select::fold` folds them only when both
            // arms are the same value.
            _ => return Ok(false),
        };
        rewriter.replace_op_with_values(operation, &[Some(replacement)]);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, format};

    use runic_core::testing::parse_function_fixpoint;
    use runic_expect_test::expect;

    use super::*;
    use crate::canonicalization::testing::apply_with_borrowing_listener;

    /// `select(c, 0, 1)` of `i1` constants becomes `bnot c`, and `select(c, 1, 0)` becomes `c`.
    #[test]
    fn select_of_bools_becomes_the_condition_or_its_negation() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @f(%c: i1) -> (i1, i1) {
    %t = arith.constant 1 : i1;
    %f = arith.constant 0 : i1;
    %same = cf.select %c, %t, %f;
    %not = cf.select %c, %f, %t;
    builtin.ret %same, %not : (i1, i1);
};";
        let (function, _) = parse_function_fixpoint(&context, "simplify_select.hir", source)?;
        let pattern: Box<dyn RewritePattern> =
            Box::new(SimplifySelectOfBools::new(context.clone()));
        assert!(apply_with_borrowing_listener(&context, function, pattern)?);
        let printed = format!("{}", function.as_operation_ref().borrow());
        expect![[r#"
            builtin.function public extern("C") @f(%0: i1) -> (i1, i1) {
                %5 = arith.bnot %0;
                builtin.ret %0, %5 : (i1, i1);
            };"#]]
        .assert_eq(&printed);
        Ok(())
    }
}
