use alloc::rc::Rc;

use runic_core::{
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind, RewritePattern},
    *,
};
use runic_dialect_arith::ArithOpBuilder;

use crate::*;

/// Simplify a 'cf.switch' that is being used like a 'cf.cond_br', by converting the former into
/// the latter predicated on a single equality check.
pub struct SimplifyCondBrLikeSwitch {
    info: PatternInfo,
}

impl SimplifyCondBrLikeSwitch {
    pub fn new(context: Rc<Context>) -> Self {
        let cf_dialect = context.get_or_register_dialect::<ControlFlowDialect>();
        let switch_op =
            cf_dialect.registered_name::<Switch>().expect("cf.switch is not registered");
        Self {
            info: PatternInfo::new(
                context,
                "simplify-cond-br-like-switch",
                PatternKind::Operation(switch_op),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for SimplifyCondBrLikeSwitch {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for SimplifyCondBrLikeSwitch {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let op = operation.borrow();
        let Some(switch_op) = op.downcast_ref::<Switch>() else {
            return Ok(false);
        };

        // Check if the switch is cond_br like
        if switch_op.num_successors() != 2 {
            return Ok(false);
        }

        // Get the conditional value we need to compare for equality
        let cases = switch_op.cases();
        let if_true_case = cases.get(0).unwrap();
        let else_case = switch_op.fallback();

        // Materialize comparison
        let selector = switch_op.selector().as_value_ref();
        let expected_value = rewriter.i32(*if_true_case.key(), switch_op.span());
        let is_true = rewriter.eq(selector, expected_value, switch_op.span())?;

        // Rewrite as cf.cond_br
        let new_op = rewriter.cond_br(
            is_true,
            if_true_case.block(),
            ValueRange::<2>::from(if_true_case.arguments().as_slice()),
            else_case.successor(),
            ValueRange::<2>::from(else_case.arguments),
            switch_op.span(),
        )?;

        drop(op);

        rewriter.replace_op(operation, new_op.as_operation_ref());

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

    /// A switch with one case becomes a `cf.cond_br` on an `i32` equality with the case's key.
    #[test]
    fn switch_with_one_case_becomes_cond_br() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @f(%s: i32, %a: i32, %b: i32) -> i32 {
    cf.switch %s [#builtin.i32<-1> -> ^one], ^two : (i32);
^one:
    builtin.ret %a : (i32);
^two:
    builtin.ret %b : (i32);
};";
        let (function, _) =
            parse_function_fixpoint(&context, "simplify_cond_br_like_switch.runic", source)?;
        let pattern: Box<dyn RewritePattern> =
            Box::new(SimplifyCondBrLikeSwitch::new(context.clone()));
        assert!(apply_with_borrowing_listener(&context, function, pattern)?);
        let printed = format!("{}", function.as_operation_ref().borrow());
        expect![[r#"
            builtin.function public extern("C") @f(%0: i32, %1: i32, %2: i32) -> i32 {
                %3 = arith.constant -1 : i32;
                %4 = arith.cmpi eq %0, %3;
                cf.cond_br %4 ^block2, ^block3 : (i1);
            ^block2:
                builtin.ret %1 : (i32);
            ^block3:
                builtin.ret %2 : (i32);
            };"#]]
        .assert_eq(&printed);
        Ok(())
    }
}
