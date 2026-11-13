use alloc::{rc::Rc, vec::Vec};

use runic_core::{
    Backward, Context, EntityMut, Operation, OperationName, OperationRef, RawWalk, Report,
    Rewriter, Value,
    pass::{Pass, PassExecutionState, PostPassStatus},
    patterns::{RewriterImpl, TracingRewriterListener},
};
/*
use runic_analysis::{
    analyses::{DeadCodeAnalysis, SparseConstantPropagation, liveness::Liveness},
};
 */

/// This transformation pass uses liveness analysis to remove any instructions which consist solely
/// of dead values, and which have no side effects (i.e. `MemoryEffect::Write` or
/// `MemoryEffect::Free`).
#[derive(Default)]
pub struct DeadCodeElimination;

runic_core::inventory::submit!(::runic_core::pass::registry::PassInfo::new::<DeadCodeElimination>(
    "dce",
    "dead code elimination"
));

impl Pass for DeadCodeElimination {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "dead-code-elimination"
    }

    fn argument(&self) -> &'static str {
        "dce"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        // Run sccp + dead code + liveness analysis so we can remove relevant dead ops
        let op = op.into_entity_ref();
        //let mut solver = DataFlowSolver::default();
        //solver.load::<DeadCodeAnalysis>();
        //solver.load::<SparseConstantPropagation>();
        //solver.load::<Liveness>();
        //solver.initialize_and_run(&op, state.analysis_manager().clone())?;

        // Rewrite based on results of analysis
        let context = op.context_rc();
        let op = {
            let op_ref = op.as_operation_ref();
            drop(op);
            op_ref
        };
        self.rewrite(op, context, state)
    }
}

impl DeadCodeElimination {
    fn rewrite(
        &mut self,
        op: OperationRef,
        context: Rc<Context>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let mut rewriter = RewriterImpl::<TracingRewriterListener>::new(context)
            .with_listener(TracingRewriterListener);
        let mut changed = PostPassStatus::Unchanged;
        op.raw_postwalk_all::<Backward, _>(|op: OperationRef| {
            // Transparent uses are informational and must not keep runtime computations alive.
            // Preserve standalone transparent ops, but when a producer has no real users, replace
            // its debug-value users with kills before erasing it.
            let dead_results = {
                let op = op.borrow();
                if op.implements::<dyn runic_core::traits::Transparent>()
                    || !op.would_be_trivially_dead()
                    || op.results().iter().any(|result| result.borrow().has_real_uses())
                {
                    None
                } else {
                    Some(
                        op.results()
                            .iter()
                            .map(|result| result.borrow().as_value_ref())
                            .collect::<Vec<_>>(),
                    )
                }
            };
            if let Some(results) = dead_results {
                changed = PostPassStatus::Changed;
                for result in results {
                    runic_core::dialects::debuginfo::transform::erase_debug_info(&result);
                }
                rewriter.erase_op(op);
            }
        });

        state.set_post_pass_status(changed);

        if !changed.ir_changed() {
            state.preserved_analyses_mut().preserve_all();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{string::ToString, vec::Vec};

    use runic_core::{SourceSpan, ValueRef, dialects::builtin::BuiltinOpBuilder, testing::Test};
    use runic_dialect_arith::ArithOpBuilder;
    use runic_dialect_cf::{Assert, ControlFlowOpBuilder};

    use super::DeadCodeElimination;

    /// A `cf.assert` has no results and a write effect, so dead-code elimination keeps it, and
    /// with it the definition of its condition; an unused, effect-free operation is erased.
    #[test]
    fn assert_survives_dead_code_elimination() {
        let mut test = Test::new("assert_survives_dce", |cx| ([cx.i32(), cx.i32()], []));
        {
            let mut builder = test.function_builder();
            let [a, b] = *builder.entry_block().borrow().arguments()[0..2].as_array().unwrap();
            let (a, b) = (a as ValueRef, b as ValueRef);
            builder.add(a, b, SourceSpan::UNKNOWN).unwrap();
            let cond = builder.ult(a, b, SourceSpan::UNKNOWN).unwrap();
            builder.assert(cond, "m", SourceSpan::UNKNOWN).unwrap();
            builder.ret(None, SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<DeadCodeElimination>(true).expect("invalid ir");

        let entry = test.entry_block();
        let entry = entry.borrow();
        let names = entry.body().iter().map(|op| op.name().to_string()).collect::<Vec<_>>();
        assert_eq!(names, ["arith.cmpi", "cf.assert", "builtin.ret"]);
        let assert = entry.body().iter().find(|op| op.is::<Assert>()).unwrap();
        let assert = assert.downcast_ref::<Assert>().unwrap();
        assert_eq!(assert.message().as_str(), "m");
    }
}
