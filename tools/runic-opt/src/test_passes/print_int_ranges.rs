//! `test-print-int-ranges`: the integer range analysis, as hints.

use runic_analysis::{
    DataFlowConfig, DataFlowSolver,
    analyses::{DeadCodeAnalysis, IntegerRangeAnalysis, integer_range::value_range},
};
use runic_ir::{
    EntityMut, Op, Operation, OperationName, Report,
    diagnostics::diagnostic,
    dialects::builtin::Function,
    pass::{Pass, PassExecutionState, PassInfo, PostPassStatus},
};

/// Runs the integer range analysis over the function it runs on and emits, on the context, for
/// each result of an integer type of each operation in it, a hint at the operation:
/// `range of result N: <range>`, the range as the analysis displays it, or `unknown` when the
/// analysis has none. It changes nothing.
#[derive(Default)]
pub struct PrintIntRanges;

const NAME: &str = "test-print-int-ranges";
const DESCRIPTION: &str =
    "Emits a hint with the range the integer range analysis gives each integer result";

runic_ir::inventory::submit!(PassInfo::new::<PrintIntRanges>(NAME, DESCRIPTION));

impl Pass for PrintIntRanges {
    type Target = Function;

    fn name(&self) -> &'static str {
        NAME
    }

    fn argument(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        DESCRIPTION
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Function>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let function = op.as_operation_ref();
        let context = op.as_operation().context_rc();
        // The analysis borrows the function itself.
        drop(op);

        let mut config = DataFlowConfig::new();
        config.set_interprocedural(true);
        let mut solver = DataFlowSolver::new(config);
        solver.load::<DeadCodeAnalysis>();
        solver.load::<IntegerRangeAnalysis>();
        solver.initialize_and_run(&function.borrow(), state.analysis_manager().clone())?;

        let mut hints = Vec::new();
        function.borrow().prewalk_all(|op: &Operation| {
            for (index, result) in op.results().iter().enumerate() {
                let value = result.borrow().as_value_ref();
                if !value.borrow().ty().is_integer() {
                    continue;
                }
                let range = match value_range(&solver, value) {
                    Some(range) => range.to_string(),
                    None => String::from("unknown"),
                };
                hints.push((op.span, format!("range of result {index}: {range}")));
            }
        });
        for (span, message) in hints {
            let _ = context.emit(diagnostic!(
                severity: Hint,
                message: (message),
                labels: [primary(span)],
            ));
        }
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}
