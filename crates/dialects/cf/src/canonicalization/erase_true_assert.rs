use alloc::rc::Rc;

use runic_core::{
    attributes::IntegerLikeAttr,
    matchers::Matcher,
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind, RewritePattern},
    *,
};

use crate::*;

/// Erases a `cf.assert` whose condition is the constant `1` of type `i1`, which cannot trap.
///
/// The constant may be any integer-like attribute (an `IntAttr` or a `BoolAttr`). An assert of the
/// constant `0` is the trap, and an assert of any other value may trap: both are left alone.
pub struct EraseTrueAssert {
    info: PatternInfo,
}

impl EraseTrueAssert {
    /// Creates the pattern, matching `cf.assert`.
    pub fn new(context: Rc<Context>) -> Self {
        let assert_op = context
            .get_or_register_dialect::<ControlFlowDialect>()
            .expect_registered_name::<Assert>();
        Self {
            info: PatternInfo::new(
                context,
                "erase-true-assert",
                PatternKind::Operation(assert_op),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for EraseTrueAssert {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for EraseTrueAssert {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let is_true = {
            let op = operation.borrow();
            let Some(assert) = op.downcast_ref::<Assert>() else {
                return Ok(false);
            };
            let Some(defining_op) = assert.cond().as_value_ref().borrow().get_defining_op() else {
                return Ok(false);
            };
            let Some(attr) = matchers::constant().matches(&*defining_op.borrow()) else {
                return Ok(false);
            };
            let attr = attr.borrow();
            attr.as_attr()
                .as_trait::<dyn IntegerLikeAttr>()
                .map(|value| value.as_int())
                .is_some_and(|value| value.width() == 1 && value.is_one())
        };
        if !is_true {
            return Ok(false);
        }
        rewriter.erase_op(operation);
        Ok(true)
    }
}
