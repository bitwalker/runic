mod analysis;
#[cfg(feature = "std")]
mod duration;
mod instrumentation;
mod manager;
#[allow(clippy::module_inception)]
mod pass;
mod pipeline;
pub mod registry;
mod specialization;
pub mod statistics;

use alloc::{borrow::Cow, boxed::Box, format, string::String};

#[cfg(feature = "std")]
pub use self::duration::HumanDuration;
pub use self::{
    analysis::{Analysis, AnalysisManager, OperationAnalysis, PreservedAnalyses},
    instrumentation::{PassInstrumentation, PassInstrumentor, PipelineParentInfo},
    manager::{IRPrintingConfig, Nesting, OpPassManager, PassManager, PassSelection},
    pass::{OperationPass, Pass, PassExecutionState, PostPassStatus},
    pipeline::{Anchor, PassPipeline, PipelineElement, SelectedPass},
    registry::{PassInfo, PassPipelineInfo},
    specialization::PassTarget,
    statistics::{PassStatistic, Statistic, StatisticValue},
};
use crate::{Forward, Operation, OperationRef, RawWalk, Report, WalkResult};

/// The stream the pass manager writes printed IR and statistics to.
///
/// It is [`std::io::Write`] with the `std` feature, so that standard output or a file can be given
/// as it is, and [`core::fmt::Write`] without it.
#[cfg(feature = "std")]
pub type PassOutput = dyn std::io::Write;
/// The stream the pass manager writes printed IR and statistics to.
///
/// It is [`std::io::Write`] with the `std` feature, so that standard output or a file can be given
/// as it is, and [`core::fmt::Write`] without it.
#[cfg(not(feature = "std"))]
pub type PassOutput = dyn core::fmt::Write;

/// Writes `text` to `out`, and flushes it when it is a [`std::io::Write`].
pub(crate) fn write_output(out: &mut PassOutput, text: &str) -> Result<(), Report> {
    #[cfg(feature = "std")]
    let result = out.write_all(text.as_bytes()).and_then(|_| out.flush());
    #[cfg(not(feature = "std"))]
    let result = out.write_str(text);
    result.map_err(|error| Report::msg(format!("could not write pass output: {error}")))
}

/// Which operations under the operation a pass ran on are printed.
///
/// An operation is printed when one of the filters of [`IRPrintingConfig::filters`] matches it;
/// the operations nested in a printed operation are not tried again.
#[derive(Default, Debug, Clone)]
pub enum OpFilter {
    /// The operation the pass ran on
    #[default]
    All,
    /// Any `Symbol` operation, or, with a string, any whose symbol name contains it
    Symbol(Option<Cow<'static, str>>),
    /// Operations of the given dialect and opcode
    Type {
        dialect: crate::interner::Symbol,
        op: crate::interner::Symbol,
    },
}

impl OpFilter {
    /// Whether this filter selects `op`, the operation a pass ran on (`root`) or one under it.
    fn matches(&self, op: &Operation, root: bool) -> bool {
        match self {
            Self::All => root,
            Self::Symbol(None) => op.as_symbol().is_some(),
            Self::Symbol(Some(part)) => {
                op.as_symbol().is_some_and(|sym| sym.name().as_str().contains(part.as_ref()))
            }
            Self::Type {
                dialect,
                op: opcode,
            } => {
                let name = op.name();
                name.dialect() == *dialect && name.name() == *opcode
            }
        }
    }
}

/// Prints the operation a pass runs on before and after the passes an [`IRPrintingConfig`]
/// selects, to a stream; installed by [`PassManager::enable_ir_printing`].
///
/// Each print is a header line, `// ----- IR before <pass> on <operation name> -----` (or
/// `after`), where `<pass>` is the command-line name of the pass, followed by the operation
/// printed with the configured [`crate::OpPrintingFlags`] and a blank line. With filters, each
/// operation they select is printed under its own header instead. A pass manager adaptor (a
/// nested pipeline) is never printed for: the passes it runs are.
struct Print {
    config: IRPrintingConfig,
    out: Box<PassOutput>,
}

impl Print {
    fn new(config: IRPrintingConfig, out: Box<PassOutput>) -> Self {
        Self { config, out }
    }

    fn print(&mut self, phase: &str, pass: &dyn OperationPass, op: &OperationRef) {
        let argument = pass.argument();
        let mut text = String::new();
        if self.config.filters.is_empty() {
            self.render(&mut text, phase, argument, &op.borrow());
        } else {
            let root = *op;
            let _ = op.raw_prewalk::<Forward, _, ()>(|nested| {
                let nested_op = nested.borrow();
                let is_root = nested == root;
                if self.config.filters.iter().any(|filter| filter.matches(&nested_op, is_root)) {
                    self.render(&mut text, phase, argument, &nested_op);
                    WalkResult::Skip
                } else {
                    WalkResult::Continue(())
                }
            });
        }
        if !text.is_empty()
            && let Err(error) = write_output(&mut *self.out, &text)
        {
            log::warn!(target: "pass", "{error}");
        }
    }

    fn render(&self, text: &mut String, phase: &str, argument: &str, op: &Operation) {
        use core::fmt::Write;

        use crate::{OpPrinter, print::AsmPrinter};

        let flags = &self.config.flags;
        let context = op.context_rc();
        let mut printer = if op.parent_op().is_none() {
            AsmPrinter::root(context, flags)
        } else {
            AsmPrinter::fragment(context, flags)
        };
        op.print(&mut printer);
        let document = printer.finish();
        let _ = writeln!(text, "// ----- IR {phase} {argument} on {} -----", op.name());
        let _ = writeln!(text, "{document}\n");
    }

    fn is_adaptor(pass: &dyn OperationPass) -> bool {
        pass.as_any().is::<manager::OpToOpPassAdaptor>()
    }
}

impl PassInstrumentation for Print {
    fn run_before_pass(&mut self, pass: &dyn OperationPass, op: &OperationRef) {
        if !Self::is_adaptor(pass) && self.config.before.selects(pass.argument()) {
            self.print("before", pass, op);
        }
    }

    fn run_after_pass(
        &mut self,
        pass: &dyn OperationPass,
        op: &OperationRef,
        post_execution_state: &PassExecutionState,
    ) {
        if Self::is_adaptor(pass) || !self.config.after.selects(pass.argument()) {
            return;
        }
        if self.config.only_when_modified
            && matches!(post_execution_state.post_pass_status(), PostPassStatus::Unchanged)
        {
            return;
        }
        self.print("after", pass, op);
    }
}
