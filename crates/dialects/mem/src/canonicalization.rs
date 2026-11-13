//! The canonicalization patterns of the `mem` dialect.

use alloc::rc::Rc;

use runic_core::{
    dialects::builtin::attributes::Array,
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind, RewritePattern},
    *,
};

use crate::*;

/// Makes a `mem.gep` with no index whose base is another `mem.gep` one gep: the inner gep's base
/// and index, and the inner path followed by the outer one.
///
/// `mem.gep (mem.gep %p, %i, [1]), [2]` is `mem.gep %p, %i, [1, 2]`: the outer path selects inside
/// the element the inner gep selected. A gep with an index is left alone: its index steps over
/// values of the inner gep's element type, which one gep on the inner base cannot say in general.
pub struct MergeGepOfGep {
    info: PatternInfo,
}

impl MergeGepOfGep {
    /// Creates the pattern, matching `mem.gep`.
    pub fn new(context: Rc<Context>) -> Self {
        let gep = context.get_or_register_dialect::<MemDialect>().expect_registered_name::<Gep>();
        Self {
            info: PatternInfo::new(
                context,
                "merge-gep-of-gep",
                PatternKind::Operation(gep),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for MergeGepOfGep {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for MergeGepOfGep {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let (base, index, path) = {
            let op = operation.borrow();
            let Some(outer) = op.downcast_ref::<Gep>() else {
                return Ok(false);
            };
            if outer.dynamic_index().is_some() {
                return Ok(false);
            }
            let Some(defining_op) = outer.base().as_value_ref().borrow().get_defining_op() else {
                return Ok(false);
            };
            let defining_op = defining_op.borrow();
            let Some(inner) = defining_op.downcast_ref::<Gep>() else {
                return Ok(false);
            };
            let index = inner.operands().group(1);
            if index.len() > 1 {
                return Ok(false);
            }
            let index = index
                .iter()
                .map(|operand| operand.borrow().as_value_ref())
                .collect::<SmallVec<[ValueRef; 1]>>();
            let mut path: Array<i32> = inner.get_path().clone();
            for step in outer.get_path().iter() {
                path.push(*step);
            }
            (inner.base().as_value_ref(), index, path)
        };

        let _guard = rewriter.modify_op_in_place(operation);
        let context = operation.borrow().context_rc();
        let mut target = operation;
        let mut op = target.borrow_mut();
        let gep = op.downcast_mut::<Gep>().expect("matched a gep");
        gep.base_mut().set(base);
        gep.index_mut().set_operands(index, operation, &context);
        gep.set_path(path);
        Ok(true)
    }
}
