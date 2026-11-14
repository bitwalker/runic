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

use alloc::{borrow::Cow, boxed::Box, rc::Rc, string::String};
use core::cell::RefCell;

#[cfg(feature = "std")]
pub use self::duration::HumanDuration;
pub use self::{
    analysis::{Analysis, AnalysisManager, OperationAnalysis, PreservedAnalyses},
    instrumentation::{PassInstrumentation, PassInstrumentor, PipelineParentInfo},
    manager::{IRPrintingConfig, Nesting, OpPassManager, PassManager, PassSelection},
    pass::{OperationPass, Pass, PassExecutionState, PostPassStatus},
    pipeline::{Anchor, PassPipeline, PipelineElement, SelectedPass, write_pass_option},
    registry::{PassInfo, PassPipelineInfo},
    specialization::PassTarget,
    statistics::{PassStatistic, Statistic, StatisticValue},
};
use crate::{Forward, Operation, OperationRef, RawWalk, Report, WalkResult};

/// The stream the pass manager writes printed IR and statistics to.
///
/// It is a [`core::fmt::Write`] in every build. With the `std` feature, [`IoOutput`] makes one of a
/// [`std::io::Write`], such as standard error or a file.
///
/// Each print, and the statistics of a run, are written with one call to
/// [`core::fmt::Write::write_str`]. After the first call that fails, nothing more is written to
/// the stream, and [`PassManager::run`] returns the failure when the passes themselves succeeded.
pub type PassOutput = dyn core::fmt::Write;

/// A [`PassOutput`] that writes to a [`std::io::Write`], and flushes it after each write.
///
/// A [`core::fmt::Error`] carries no message, so the first [`std::io::Error`] is kept here, and
/// the [`core::fmt::Write::write_str`] that met it and every one after it fail without writing.
#[cfg(feature = "std")]
pub struct IoOutput<W> {
    writer: W,
    error: Option<std::io::Error>,
}

#[cfg(feature = "std")]
impl<W: std::io::Write> IoOutput<W> {
    /// An output that writes to `writer`.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            error: None,
        }
    }

    /// An output that writes to `writer`, boxed as the pass manager takes it.
    pub fn boxed(writer: W) -> Box<PassOutput>
    where
        W: 'static,
    {
        Box::new(Self::new(writer))
    }

    /// The first error the writer returned, if it returned one.
    pub fn error(&self) -> Option<&std::io::Error> {
        self.error.as_ref()
    }

    /// The writer, and the first error it returned, if it returned one.
    pub fn into_inner(self) -> (W, Option<std::io::Error>) {
        (self.writer, self.error)
    }
}

#[cfg(feature = "std")]
impl<W: std::io::Write> core::fmt::Write for IoOutput<W> {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        if self.error.is_some() {
            return Err(core::fmt::Error);
        }
        match self.writer.write_all(text.as_bytes()).and_then(|()| self.writer.flush()) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.error = Some(error);
                Err(core::fmt::Error)
            }
        }
    }
}

/// A [`PassOutput`] the pass manager writes to, which writes nothing after its first failure.
pub(crate) struct OutputStream {
    out: Box<PassOutput>,
    failed: bool,
}

impl OutputStream {
    pub(crate) fn new(out: Box<PassOutput>) -> Self {
        Self { out, failed: false }
    }

    /// Writes `text`, unless an earlier write failed; the error is returned only for the write that
    /// failed.
    pub(crate) fn write(&mut self, text: &str) -> Result<(), Report> {
        if self.failed {
            return Ok(());
        }
        self.out.write_str(text).map_err(|_| {
            self.failed = true;
            Report::msg("could not write pass output: the stream returned an error")
        })
    }
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
///
/// The first write that fails is recorded in `error`, where [`PassManager::run`] finds it; nothing
/// is written after it.
struct Print {
    config: IRPrintingConfig,
    out: OutputStream,
    error: Rc<RefCell<Option<Report>>>,
}

impl Print {
    fn new(
        config: IRPrintingConfig,
        out: Box<PassOutput>,
        error: Rc<RefCell<Option<Report>>>,
    ) -> Self {
        Self {
            config,
            out: OutputStream::new(out),
            error,
        }
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
            && let Err(error) = self.out.write(&text)
        {
            let mut slot = self.error.borrow_mut();
            if slot.is_none() {
                *slot = Some(error);
            }
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
