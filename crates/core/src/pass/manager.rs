use alloc::{
    boxed::Box,
    collections::BTreeMap,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::cell::RefCell;

use compact_str::{CompactString, ToCompactString};
use smallvec::{SmallVec, smallvec};

use super::{
    AnalysisManager, OpFilter, OperationPass, OutputStream, Pass, PassExecutionState,
    PassInstrumentation, PassInstrumentor, PassOutput, PipelineParentInfo, Statistic,
};
use crate::{
    Context, EntityMut, Forward, OpPrintingFlags, OpRegistration, Operation, OperationName,
    OperationRef, RawWalk, Report, SourceSpan, WalkResult,
    pass::{PostPassStatus, Print},
    traits::IsolatedFromAbove,
};

/// What a pass manager does with a pass that targets one operation type (see
/// [`super::PassTarget::target_name`]) when the pass is added to a manager anchored on another.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub enum Nesting {
    /// The pass is nested one level: under a manager for its type, which runs it on the
    /// operations of that type directly inside the regions of the anchor's operation, not on ones
    /// nested deeper. To reach deeper, a pipeline says `recursive(..)` (the outermost operations
    /// of the type at any depth) or names the anchors in between, as in
    /// `builtin.module(builtin.function(..))`.
    Implicit,
    /// The add is an error that names the pass, its target and the anchor.
    #[default]
    Explicit,
}

/// The passes IR is printed for, selected by their command-line names ([`Pass::argument`]).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum PassSelection {
    /// No pass
    #[default]
    None,
    /// Every pass
    All,
    /// The passes with these command-line names
    Named(Vec<String>),
}

impl PassSelection {
    /// Whether the pass whose command-line name is `argument` is selected.
    pub fn selects(&self, argument: &str) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Named(names) => names.iter().any(|name| name == argument),
        }
    }
}

/// When and how [`PassManager::enable_ir_printing`] prints IR.
#[derive(Default, Debug)]
pub struct IRPrintingConfig {
    /// The passes before which the operation they run on is printed
    pub before: PassSelection,
    /// The passes after which the operation they ran on is printed
    pub after: PassSelection,
    /// The passes after whose failure the operation they ran on is printed, as the failed pass
    /// left it (a failure of the verifier after a pass is a failure of the pass)
    pub after_failure: PassSelection,
    /// Print after a pass only when the pass reported that it changed the IR (see
    /// [`PassExecutionState::set_post_pass_status`]); printing before a pass is not affected
    pub only_when_modified: bool,
    /// When not empty, print only the operations these filters select, each on its own, instead of
    /// the operation the pass ran on
    pub filters: Vec<OpFilter>,
    /// How operations are printed
    pub flags: OpPrintingFlags,
}

/// The main pass manager and pipeline builder
pub struct PassManager {
    context: Rc<Context>,
    /// The underlying pass manager
    pm: OpPassManager,
    /// A manager for pass instrumentation
    instrumentor: Rc<PassInstrumentor>,
    // An optional crash reproducer generator, if this pass manager is setup to
    // generate reproducers.
    //crash_reproducer_generator: Rc<PassCrashReproducerGenerator>,
    /// Where pass statistics are written after a run, if they are
    statistics: Option<OutputStream>,
    /// The first failure to write printed IR (see [`Self::enable_ir_printing`]), returned by the
    /// next [`Self::run`] whose passes succeed
    output_error: Rc<RefCell<Option<Report>>>,
    /// Indicates whether or not to run verification between passes
    verification: bool,
}

impl PassManager {
    /// Create a new pass manager under the given context with a specific nesting style. The created
    /// pass manager can schedule operations that match `name`: `any`, or the name of an operation
    /// (`<dialect>.<opcode>`) registered in `context`.
    ///
    /// Returns an error when `name` is neither.
    pub fn new(
        context: Rc<Context>,
        name: impl AsRef<str>,
        nesting: Nesting,
    ) -> Result<Self, Report> {
        let pm = OpPassManager::new(name.as_ref(), nesting, context.clone())?;
        Ok(Self::with_op_pass_manager(context, pm))
    }

    /// Create a new pass manager under the given context with a specific nesting style.
    ///
    /// The created pass manager can schedule operations that match type `T`.
    pub fn on<T: OpRegistration>(context: Rc<Context>, nesting: Nesting) -> Self {
        let pm = OpPassManager::on::<T>(nesting, context.clone());
        Self::with_op_pass_manager(context, pm)
    }

    fn with_op_pass_manager(context: Rc<Context>, pm: OpPassManager) -> Self {
        Self {
            context,
            pm,
            instrumentor: Default::default(),
            statistics: None,
            output_error: Rc::default(),
            verification: true,
        }
    }

    /// Run the passes within this manager on the provided operation. The
    /// specified operation must have the same name as the one provided the pass
    /// manager on construction.
    pub fn run(&mut self, op: OperationRef) -> Result<(), Report> {
        use crate::Spanned;

        let op_name = op.borrow().name();
        let anchor = self.pm.name();
        if let Some(anchor) = anchor
            && anchor != &op_name
        {
            return Err(crate::diagnostics::report!(
                message: "failed to construct pass manager",
                labels: [
                    primary(
                        op.borrow().span(),
                        format!("can't run '{anchor}' pass manager on '{op_name}'")
                    ),
                ],
            ));
        }

        // Register all dialects for the current pipeline.
        /*
        let dependent_dialects = self.get_dependent_dialects();
        self.context.append_dialect_registry(dependent_dialects);
        for dialect_name in dependent_dialects.names() {
            self.context.get_or_register_dialect(dialect_name);
        }
        */

        // Before running, make sure to finalize the pipeline pass list.
        self.pm.finalize_pass_list()?;

        // Run pass initialization
        self.pm.initialize()?;

        // Construct a top level analysis manager for the pipeline.
        let analysis_manager = AnalysisManager::new(op, Some(self.instrumentor.clone()));

        // If reproducer generation is enabled, run the pass manager with crash handling enabled.
        /*
        let result = if self.crash_reproducer_generator.is_some() {
            self.run_with_crash_recovery(op, analysis_manager);
        } else {
            self.run_passes(op, analysis_manager);
        }
        */
        let result = self.run_passes(op, analysis_manager);
        let printed = self.output_error.borrow_mut().take().map_or(Ok(()), Err);

        // Write the pass statistics if they were asked for, whether or not the passes succeeded.
        let mut written = Ok(());
        if let Some(out) = self.statistics.as_mut() {
            let mut text = String::new();
            written = self
                .pm
                .print_statistics(&mut text)
                .map_err(Report::msg)
                .and_then(|()| out.write(&text));
        }

        result.and(printed).and(written)
    }

    fn run_passes(
        &mut self,
        op: OperationRef,
        analysis_manager: AnalysisManager,
    ) -> Result<(), Report> {
        OpToOpPassAdaptor::run_pipeline(
            &mut self.pm,
            op,
            analysis_manager,
            self.verification,
            Some(self.instrumentor.clone()),
            Some(&PipelineParentInfo { pass: None }),
        )
    }

    #[inline]
    pub fn context(&self) -> Rc<Context> {
        self.context.clone()
    }

    /// Runs the verifier after each individual pass.
    pub fn enable_verifier(&mut self, yes: bool) -> &mut Self {
        self.verification = yes;
        self
    }

    pub fn add_instrumentation(&mut self, pi: Box<dyn PassInstrumentation>) -> &mut Self {
        self.instrumentor.add_instrumentation(pi);
        self
    }

    /// Prints the operation a pass ran on, before and after the selected passes, to `out`.
    ///
    /// A pass is selected by its command-line name; see [`IRPrintingConfig`] for what is printed.
    /// Nothing is installed, and nothing written, when no pass is selected.
    ///
    /// After a write to `out` fails, nothing more is written to it. The run that met the failure
    /// returns it when its passes succeed; every later run of this manager writes nothing to `out`
    /// and does not report it again. The failure of `out` does not stop the stream of
    /// [`Self::enable_statistics`], which is separate even when both write to the same device. The
    /// reported error does not say why the write failed; an [`IoOutput`](crate::pass::IoOutput)
    /// keeps the reason in the handle it returns.
    pub fn enable_ir_printing(
        &mut self,
        config: IRPrintingConfig,
        out: Box<PassOutput>,
    ) -> &mut Self {
        if config.before != PassSelection::None
            || config.after != PassSelection::None
            || config.after_failure != PassSelection::None
        {
            let error = self.output_error.clone();
            self.add_instrumentation(Box::new(Print::new(config, out, error)));
        }
        self
    }

    /// Writes the statistics of the passes to `out` at the end of each run, in the form of
    /// [`OpPassManager::print_statistics`].
    ///
    /// A run whose passes succeed returns the failure to write them. After a write fails, nothing
    /// more is written to `out`: every later run of this manager writes no statistics and does not
    /// report it again. The failure of `out` does not stop the stream of
    /// [`Self::enable_ir_printing`], which is separate even when both write to the same device. The
    /// reported error does not say why the write failed; an [`IoOutput`](crate::pass::IoOutput)
    /// keeps the reason in the handle it returns.
    pub fn enable_statistics(&mut self, out: Box<PassOutput>) -> &mut Self {
        self.statistics = Some(OutputStream::new(out));
        self
    }

    pub fn print_as_textual_pipeline(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.pm.print_as_textual_pipeline(f)
    }

    pub fn nest<T: OpRegistration>(&mut self) -> NestedOpPassManager<'_> {
        self.pm.nest::<T>()
    }

    /// Schedule a pipeline on outermost proper descendants of type `T`.
    ///
    /// See [`OpPassManager::nest_recursive`] for traversal and ordering semantics.
    pub fn nest_recursive<T: OpRegistration>(&mut self) -> NestedOpPassManager<'_> {
        self.pm.nest_recursive::<T>()
    }

    pub fn nest_pass_manager(&mut self, nested: OpPassManager) -> NestedOpPassManager<'_> {
        self.pm.nest_pass_manager(nested)
    }

    /// Nest a new op-specific pass manager (for the op with the given name), under this pass manager.
    ///
    /// Returns an error when `nested_name` is neither `any` nor the name of a registered operation.
    pub fn nest_with_type(&mut self, nested_name: &str) -> Result<NestedOpPassManager<'_>, Report> {
        self.pm.nest_with_type(nested_name)
    }

    /// Nest a new op-agnostic ("any") pass manager under this pass manager.
    pub fn nest_any(&mut self) -> NestedOpPassManager<'_> {
        self.pm.nest_any()
    }

    /// Add `pass` to this pass manager; see [`OpPassManager::add_pass`].
    pub fn add_pass(&mut self, pass: Box<dyn OperationPass>) -> Result<(), Report> {
        self.pm.add_pass(pass)
    }

    /// Add `pass` to a pass manager for operations of type `T` nested under this one.
    pub fn add_nested_pass<T: OpRegistration>(
        &mut self,
        pass: Box<dyn OperationPass>,
    ) -> Result<(), Report> {
        self.pm.add_nested_pass::<T>(pass)
    }

    #[inline(always)]
    pub fn op_pass_manager_mut(&mut self) -> &mut OpPassManager {
        &mut self.pm
    }
}

/// This class represents a pass manager that runs passes on either a specific
/// operation type, or any isolated operation. This pass manager can not be run
/// on an operation directly, but must be run either as part of a top-level
/// `PassManager`(e.g. when constructed via `nest` calls), or dynamically within
/// a pass by using the `Pass::runPipeline` API.
pub struct OpPassManager {
    /// The current context
    context: Rc<Context>,
    /// The name of the operation that passes of this pass manager operate on
    name: Option<OperationName>,
    /// The set of passes to run as part of this pass manager
    passes: SmallVec<[Box<dyn OperationPass>; 8]>,
    /// Control the implicit nesting of passes that mismatch the name set for this manager
    nesting: Nesting,
}

impl OpPassManager {
    pub const ANY: &str = "any";

    /// Construct a new op-agnostic ("any") pass manager with the given operation
    /// type and nesting behavior. This is the same as invoking:
    /// `OpPassManager(OpPassManager::ANY, nesting)`.
    pub fn any(nesting: Nesting, context: Rc<Context>) -> Self {
        Self {
            context,
            name: None,
            passes: Default::default(),
            nesting,
        }
    }

    /// Construct a pass manager for the operations named `name`: `any` (see [`Self::any`]), or the
    /// name of an operation (`<dialect>.<opcode>`) registered in `context`.
    ///
    /// Returns an error when `name` is neither: it has no dialect, names a dialect that is not
    /// registered, or names an operation its dialect does not have.
    pub fn new(name: &str, nesting: Nesting, context: Rc<Context>) -> Result<Self, Report> {
        if name == Self::ANY {
            return Ok(Self::any(nesting, context));
        }

        let Some((dialect_name, opcode)) = name.split_once('.') else {
            return Err(Report::msg(format!(
                "invalid pass manager anchor '{name}': expected 'any' or an operation name of the \
                 form '<dialect>.<opcode>'"
            )));
        };
        let Some(dialect) =
            context.find_registered_dialect(crate::interner::Symbol::intern(dialect_name))
        else {
            return Err(Report::msg(format!(
                "invalid pass manager anchor '{name}': no dialect '{dialect_name}' is registered"
            )));
        };
        let Some(name) =
            dialect.registered_ops().iter().find(|op| op.name().as_str() == opcode).cloned()
        else {
            return Err(Report::msg(format!(
                "invalid pass manager anchor '{name}': dialect '{dialect_name}' has no operation \
                 '{opcode}'"
            )));
        };
        Ok(Self::for_operation(name, nesting, context))
    }

    /// Construct a pass manager for the operations of type `T`, registering the dialect of `T` in
    /// `context` if that has not been done.
    pub fn on<T: OpRegistration>(nesting: Nesting, context: Rc<Context>) -> Self {
        let name = <T as super::PassTarget>::target_name(&context).unwrap_or_else(|| {
            panic!(
                "operation `{}` is not registered with its dialect '{}'",
                core::any::type_name::<T>(),
                <T as OpRegistration>::dialect_name()
            )
        });
        Self::for_operation(name, nesting, context)
    }

    pub fn for_operation(name: OperationName, nesting: Nesting, context: Rc<Context>) -> Self {
        Self {
            context,
            name: Some(name),
            passes: Default::default(),
            nesting,
        }
    }

    pub fn context(&self) -> Rc<Context> {
        self.context.clone()
    }

    pub fn passes(&self) -> &[Box<dyn OperationPass>] {
        &self.passes
    }

    pub fn passes_mut(&mut self) -> &mut [Box<dyn OperationPass>] {
        &mut self.passes
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_op_agnostic(&self) -> bool {
        self.name.is_none()
    }

    pub fn clear(&mut self) {
        self.passes.clear();
    }

    /// Nest a new op-specific pass manager (for the op with the given name), under this pass manager.
    ///
    /// Returns an error when `nested_name` is neither `any` nor the name of a registered operation.
    pub fn nest_with_type(&mut self, nested_name: &str) -> Result<NestedOpPassManager<'_>, Report> {
        let nested = Self::new(nested_name, self.nesting, self.context.clone())?;
        Ok(self.nest_pass_manager(nested))
    }

    pub fn nest<T: OpRegistration>(&mut self) -> NestedOpPassManager<'_> {
        self.nest_pass_manager(Self::on::<T>(self.nesting, self.context.clone()))
    }

    /// Schedule a pipeline on outermost proper descendants of type `T`, searching through
    /// unmatched operations in region, block, and operation order.
    ///
    /// The current operation is excluded. A matching operation owns its subtree: only the
    /// nested pipeline may schedule further passes within it. In contrast, [`Self::nest`]
    /// considers only immediate children. [`Nesting::Implicit`] does not enable recursion.
    ///
    /// Recursive pipelines retain their stage boundaries: each completes its traversal before
    /// the next pipeline starts. They are not coalesced with adjacent nested pipelines.
    pub fn nest_recursive<T: OpRegistration>(&mut self) -> NestedOpPassManager<'_> {
        let nested = Self::on::<T>(self.nesting, self.context.clone());
        self.nest_recursive_pass_manager(nested)
    }

    /// Like [`Self::nest_recursive`], for the operations named `nested_name`: `any`, or the name
    /// of a registered operation.
    ///
    /// Returns an error when `nested_name` is neither.
    pub fn nest_recursive_with_type(
        &mut self,
        nested_name: &str,
    ) -> Result<NestedOpPassManager<'_>, Report> {
        let nested = Self::new(nested_name, self.nesting, self.context.clone())?;
        Ok(self.nest_recursive_pass_manager(nested))
    }

    fn nest_recursive_pass_manager(&mut self, nested: Self) -> NestedOpPassManager<'_> {
        let mut nested = OpToOpPassAdaptor::new(nested);
        nested.recursive = true;
        NestedOpPassManager {
            parent: self,
            nested: Some(Box::new(nested)),
        }
    }

    /// Nest a new op-agnostic ("any") pass manager under this pass manager.
    pub fn nest_any(&mut self) -> NestedOpPassManager<'_> {
        self.nest_pass_manager(Self::any(self.nesting, self.context.clone()))
    }

    fn nest_for(&mut self, nested_name: OperationName) -> NestedOpPassManager<'_> {
        self.nest_pass_manager(Self::for_operation(nested_name, self.nesting, self.context.clone()))
    }

    /// Add `pass` to this pass manager.
    ///
    /// A pass that targets one operation type (see [`super::PassTarget::target_name`]) added to a
    /// manager anchored on another is, with [`Nesting::Implicit`], nested one level under a
    /// manager for its type: it runs on the operations of its type directly inside the regions of
    /// the anchor's operation, and not on ones nested deeper (a pipeline reaches those with
    /// `recursive(..)` or by naming the anchors in between). With [`Nesting::Explicit`] it is an
    /// error that names the pass, its target and the anchor. A manager for `any` takes every pass
    /// as it is: it schedules only on operations every one of its passes can be scheduled on,
    /// and a pass that targets a type or an interface can be scheduled only on operations of that
    /// target, whatever its own [`Pass::can_schedule_on`] says.
    pub fn add_pass(&mut self, pass: Box<dyn OperationPass>) -> Result<(), Report> {
        let pass_op_name = pass.target_name(&self.context);
        if let Some(pass_op_name) = pass_op_name
            && let Some(anchor) = self.name.as_ref()
            && anchor != &pass_op_name
        {
            if matches!(self.nesting, Nesting::Implicit) {
                let mut nested = self.nest_for(pass_op_name);
                return nested.add_pass(pass);
            }
            let pass_name = match pass.argument() {
                "" => pass.name(),
                argument => argument,
            };
            return Err(Report::msg(format!(
                "cannot add pass '{pass_name}', which runs on '{pass_op_name}', to a pass manager \
                 anchored on '{anchor}': nest it under '{pass_op_name}', or use implicit nesting"
            )));
        }

        self.passes.push(pass);
        Ok(())
    }

    /// Add `pass` to a pass manager for operations of type `T` nested under this one.
    pub fn add_nested_pass<T: OpRegistration>(
        &mut self,
        pass: Box<dyn OperationPass>,
    ) -> Result<(), Report> {
        let mut nested = self.nest::<T>();
        nested.add_pass(pass)
    }

    pub fn finalize_pass_list(&mut self) -> Result<(), Report> {
        let finalize_adaptor = |adaptor: &mut OpToOpPassAdaptor| -> Result<(), Report> {
            for pm in adaptor.pass_managers_mut() {
                pm.finalize_pass_list()?;
            }

            Ok(())
        };

        // Walk the pass list and merge adjacent adaptors.
        let num_passes = self.passes.len();
        let passes = core::mem::replace(&mut self.passes, SmallVec::with_capacity(num_passes));
        let prev_adaptor = None::<Box<OpToOpPassAdaptor>>;
        let (_, prev_adaptor) = passes.into_iter().try_fold(
            (&mut self.passes, prev_adaptor),
            |(passes, prev), mut pass| {
                // Is this pass an adaptor?
                match pass.as_any_mut().downcast_mut::<OpToOpPassAdaptor>() {
                    // Yes, merge it into the previous one if present, otherwise use this as the
                    // first adaptor in a potential chain of them
                    Some(adaptor) => match prev {
                        Some(mut prev_adaptor) => {
                            if adaptor.try_merge_into(&mut prev_adaptor) {
                                Ok::<_, Report>((passes, Some(prev_adaptor)))
                            } else {
                                finalize_adaptor(&mut prev_adaptor)?;
                                passes.push(prev_adaptor as Box<dyn OperationPass>);
                                let current =
                                    pass.into_any().downcast::<OpToOpPassAdaptor>().unwrap();
                                Ok((passes, Some(current)))
                            }
                        }
                        None => {
                            let current = pass.into_any().downcast::<OpToOpPassAdaptor>().unwrap();
                            Ok((passes, Some(current)))
                        }
                    },
                    // This pass isn't an adaptor, but if we have one, we need to finalize it
                    None => {
                        match prev {
                            Some(mut prev_adaptor) => {
                                finalize_adaptor(&mut prev_adaptor)?;
                                passes.push(prev_adaptor as Box<dyn OperationPass>);
                                passes.push(pass);
                            }
                            None => {
                                passes.push(pass);
                            }
                        }
                        Ok((passes, None))
                    }
                }
            },
        )?;

        if let Some(mut prev_adaptor) = prev_adaptor {
            finalize_adaptor(&mut prev_adaptor)?;
            self.passes.push(prev_adaptor);
        }

        // If this is a op-agnostic pass manager, there is nothing left to do.
        match self.name.as_ref() {
            None => Ok(()),
            // Otherwise, verify that all of the passes are valid for the current operation anchor.
            Some(name) => {
                for pass in self.passes.iter() {
                    if !pass.can_schedule_on(name) {
                        return Err(crate::diagnostics::report!(
                            message: (format!(
                                "unable to schedule pass '{}' on pass manager intended for \
                                 '{name}'",
                                pass.name()
                            )),
                        ));
                    }
                }

                Ok(())
            }
        }
    }

    pub fn name(&self) -> Option<&OperationName> {
        self.name.as_ref()
    }

    pub fn set_nesting(&mut self, nesting: Nesting) {
        self.nesting = nesting;
    }

    pub fn nesting(&self) -> Nesting {
        self.nesting
    }

    /// Indicate if this pass manager can be scheduled on the given operation
    pub fn can_schedule_on(&self, name: &OperationName) -> bool {
        // If this pass manager is op-specific, we simply check if the provided operation name
        // is the same as this one.
        if let Some(op_name) = self.name() {
            return op_name == name;
        }

        // Otherwise, this is an op-agnostic pass manager. Check that the operation can be
        // scheduled on all passes within the manager.
        if !name.implements::<dyn IsolatedFromAbove>() {
            return false;
        }
        self.passes.iter().all(|pass| pass.can_schedule_on(name))
    }

    fn initialize(&mut self) -> Result<(), Report> {
        for pass in self.passes.iter_mut() {
            // If this pass isn't an adaptor, directly initialize it
            if let Some(adaptor) = pass.as_any_mut().downcast_mut::<OpToOpPassAdaptor>() {
                for pm in adaptor.pass_managers_mut() {
                    pm.initialize()?;
                }
            } else {
                pass.initialize(self.context.clone())?;
            }
        }

        Ok(())
    }

    #[allow(unused)]
    fn merge_into(&mut self, rhs: &mut Self) {
        assert_eq!(self.name, rhs.name, "merging unrelated pass managers");
        for pass in self.passes.drain(..) {
            rhs.passes.push(pass);
        }
    }

    pub fn nest_pass_manager(&mut self, nested: Self) -> NestedOpPassManager<'_> {
        let adaptor = Box::new(OpToOpPassAdaptor::new(nested));
        NestedOpPassManager {
            parent: self,
            nested: Some(adaptor),
        }
    }

    /// Prints the passes of this pass manager as pipeline text: `anchor(pass,pass)`, in the
    /// grammar that [`super::PassPipeline::parse`] reads.
    pub fn print_as_textual_pipeline(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if let Some(anchor) = self.name() {
            write!(f, "{anchor}(")?;
        } else {
            f.write_str("any(")?;
        }
        for (i, pass) in self.passes().iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            pass.print_as_textual_pipeline(f)?;
        }
        f.write_str(")")
    }

    /// Prints the statistics of the passes of this pass manager and of the pass managers nested in
    /// it: a header, then each pass that has statistics, by name, with its statistics sorted by
    /// name. The statistics of passes of one name are merged.
    pub fn print_statistics(&self, out: &mut dyn core::fmt::Write) -> core::fmt::Result {
        const PASS_STATS_DESCRIPTION: &str = "... Pass statistics report ...";

        // Print the stats header, with the description centered in 80 columns.
        writeln!(out, "=={:-<76}==", "")?;
        let padding = 80usize.saturating_sub(PASS_STATS_DESCRIPTION.len()) / 2;
        writeln!(out, "{:padding$}{PASS_STATS_DESCRIPTION}", "")?;
        writeln!(out, "=={:-<76}==", "")?;

        self.print_statistics_as_list(out)
    }

    fn add_stats(
        pass: &dyn OperationPass,
        merged_stats: &mut BTreeMap<&str, SmallVec<[Box<dyn Statistic>; 4]>>,
    ) {
        use alloc::collections::btree_map::Entry;

        if let Some(adaptor) = pass.as_any().downcast_ref::<OpToOpPassAdaptor>() {
            // Recursively add each of the children.
            for pass_manager in adaptor.pass_managers() {
                for pass in pass_manager.passes() {
                    Self::add_stats(&**pass, merged_stats);
                }
            }
        } else {
            // If this is not an adaptor, add the stats to the list if there are any.
            if !pass.has_statistics() {
                return;
            }
            let statistics = SmallVec::<[Box<dyn Statistic>; 4]>::from_iter(
                pass.statistics().iter().map(|stat| Statistic::clone(&**stat)),
            );
            match merged_stats.entry(pass.name()) {
                Entry::Vacant(entry) => {
                    entry.insert(statistics);
                }
                Entry::Occupied(mut entry) => {
                    let prev_stats = entry.get_mut();
                    assert_eq!(prev_stats.len(), statistics.len());
                    for (index, mut stat) in statistics.into_iter().enumerate() {
                        let _ = prev_stats[index].try_merge(stat.value_as_any_mut());
                    }
                }
            }
        }
    }

    /// Prints the statistics of each pass, in the order of the names of the passes.
    fn print_statistics_as_list(&self, out: &mut dyn core::fmt::Write) -> core::fmt::Result {
        let mut merged_stats = BTreeMap::<&str, SmallVec<[Box<dyn Statistic>; 4]>>::default();
        for pass in self.passes.iter() {
            Self::add_stats(&**pass, &mut merged_stats);
        }

        for (pass, stats) in merged_stats.iter() {
            self.print_pass_entry(out, 2, pass, stats)?;
        }

        Ok(())
    }

    fn print_pass_entry(
        &self,
        out: &mut dyn core::fmt::Write,
        indent: usize,
        pass: &str,
        stats: &[Box<dyn Statistic>],
    ) -> core::fmt::Result {
        use core::fmt::Write;

        writeln!(out, "{pass}")?;
        if stats.is_empty() {
            return Ok(());
        }

        // Collect the largest name and value length from each of the statistics.

        struct Rendered<'a> {
            name: &'a str,
            description: &'a str,
            value: compact_str::CompactString,
        }

        let mut largest_name = 0usize;
        let mut largest_value = 0usize;
        let mut rendered_stats = SmallVec::<[Rendered; 4]>::default();
        for stat in stats {
            let mut value = compact_str::CompactString::default();
            let doc = stat.pretty_print();
            write!(&mut value, "{doc}")?;
            let name = stat.name();
            largest_name = core::cmp::max(largest_name, name.len());
            largest_value = core::cmp::max(largest_value, value.len());
            rendered_stats.push(Rendered {
                name,
                description: stat.description(),
                value,
            });
        }

        // Sort the statistics by name.
        rendered_stats.sort_by(|a, b| a.name.cmp(b.name));

        // Print statistics
        for stat in rendered_stats {
            write!(out, "{:indent$}(S) ", "")?;
            write!(out, "{: <1$} ", stat.value, largest_value)?;
            if stat.description.is_empty() {
                writeln!(out, "{}", stat.name)?;
            } else {
                writeln!(out, "{: <1$} - {2}", stat.name, largest_name, stat.description)?;
            }
        }

        Ok(())
    }
}

pub struct NestedOpPassManager<'parent> {
    parent: &'parent mut OpPassManager,
    nested: Option<Box<OpToOpPassAdaptor>>,
}

impl core::ops::Deref for NestedOpPassManager<'_> {
    type Target = OpPassManager;

    fn deref(&self) -> &Self::Target {
        &self.nested.as_deref().unwrap().pass_managers()[0]
    }
}
impl core::ops::DerefMut for NestedOpPassManager<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.nested.as_deref_mut().unwrap().pass_managers_mut()[0]
    }
}

impl Drop for NestedOpPassManager<'_> {
    fn drop(&mut self) {
        // An adaptor targets no operation type, so it is added as it is.
        self.parent.passes.push(self.nested.take().unwrap() as Box<dyn OperationPass>);
    }
}

pub struct OpToOpPassAdaptor {
    pms: SmallVec<[OpPassManager; 1]>,
    recursive: bool,
}
impl OpToOpPassAdaptor {
    pub fn new(pm: OpPassManager) -> Self {
        Self {
            pms: smallvec![pm],
            recursive: false,
        }
    }

    pub fn name(&self) -> CompactString {
        use core::fmt::Write;

        let mut name = CompactString::default();
        let names =
            crate::formatter::DisplayValues::new(self.pms.iter().map(|pm| match pm.name() {
                None => alloc::borrow::Cow::Borrowed(OpPassManager::ANY),
                Some(name) => alloc::borrow::Cow::Owned(name.to_string()),
            }));
        let kind = if self.recursive {
            "Recursive Pipeline"
        } else {
            "Pipeline Collection"
        };
        write!(&mut name, "{kind}: [{names}]").unwrap();
        name
    }

    /// Try to merge the current pass adaptor into 'rhs'.
    ///
    /// This will try to append the pass managers of this adaptor into those within `rhs`, or return
    /// failure if merging isn't possible. The main situation in which merging is not possible is if
    /// one of the adaptors has an `any` pipeline that is not compatible with a pass manager in the
    /// other adaptor. For example, if this adaptor has a `builtin.function` pipeline and `rhs` has an
    /// `any` pipeline that operates on a FunctionOpInterface. In this situation the pipelines have
    /// a conflict (they both want to run on the same operations), so we can't merge.
    pub fn try_merge_into(&mut self, rhs: &mut Self) -> bool {
        // Coalescing recursive anchors can change which subtrees are visited, and even equal
        // anchors must retain the ordering of whole-tree stages rather than per-operation passes.
        if self.recursive || rhs.recursive {
            return false;
        }
        // Functor used to detect if the given generic pass manager will have a potential schedule
        // conflict with the given `pms`.
        let has_schedule_conflict_with = |generic_pm: &OpPassManager, pms: &[OpPassManager]| {
            pms.iter().any(|pm| {
                // If this is a non-generic pass manager, a conflict will arise if a non-generic
                // pass manager's operation name can be scheduled on the generic passmanager.
                if let Some(name) = pm.name() {
                    generic_pm.can_schedule_on(name)
                } else {
                    // Otherwise, this is a generic pass manager. We current can't determine when
                    // generic pass managers can be merged, so conservatively assume they conflict.
                    true
                }
            })
        };

        // Check that if either adaptor has a generic pass manager, that pm is compatible within any
        // non-generic pass managers.
        //
        // Check the current adaptor.
        let lhs_generic = self.pass_managers().iter().find(|pm| pm.is_op_agnostic());
        if lhs_generic.is_some_and(|pm| has_schedule_conflict_with(pm, rhs.pass_managers())) {
            return false;
        }

        // Check the rhs adaptor.
        let rhs_generic = rhs.pass_managers().iter().find(|pm| pm.is_op_agnostic());
        if rhs_generic.is_some_and(|pm| has_schedule_conflict_with(pm, self.pass_managers())) {
            return false;
        }

        for mut pm in self.pms.drain(..) {
            // If an existing pass manager exists, then merge the given pass manager into it.
            if let Some(existing) =
                rhs.pass_managers_mut().iter_mut().find(|rpm| pm.name() == rpm.name())
            {
                pm.merge_into(existing);
            } else {
                // Otherwise, add the given pass manager to the list.
                rhs.pms.push(pm);
            }
        }

        // After coalescing, sort the pass managers within rhs by name.
        rhs.pms.sort_by(|lhs, rhs| {
            use core::cmp::Ordering;
            // Order op-specific pass managers first and op-agnostic pass managers last.
            match (lhs.name(), rhs.name()) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(lhs), Some(rhs)) => lhs.cmp(rhs),
            }
        });

        true
    }

    pub fn pass_managers(&self) -> &[OpPassManager] {
        &self.pms
    }

    pub fn pass_managers_mut(&mut self) -> &mut [OpPassManager] {
        &mut self.pms
    }

    /// Run the given operation and analysis manager on a provided op pass manager.
    fn run_pipeline(
        pm: &mut OpPassManager,
        op: OperationRef,
        analysis_manager: AnalysisManager,
        verify: bool,
        instrumentor: Option<Rc<PassInstrumentor>>,
        parent_info: Option<&PipelineParentInfo>,
    ) -> Result<(), Report> {
        if verify {
            // We run an initial recursive verification, since this is the first verification done
            // to the operations
            Self::verify(&op, true)?;
        }

        assert!(
            instrumentor.is_none() || parent_info.is_some(),
            "expected parent info if instrumentor is provided"
        );

        // Clear out any computed operation analyses on exit.
        //
        // These analyses won't be used anymore in this pipeline, and this helps reduce the
        // current working set of memory. If preserving these analyses becomes important in the
        // future, we can re-evaluate.
        let _clear = analysis_manager.defer_clear();

        // Run the pipeline over the provided operation.
        let mut op_name = None;
        if let Some(instrumentor) = instrumentor.as_deref() {
            op_name = pm.name().cloned();
            instrumentor.run_before_pipeline(op_name.as_ref(), parent_info.as_ref().unwrap(), op);
        }

        for pass in pm.passes_mut() {
            Self::run(&mut **pass, op, analysis_manager.clone(), verify)?;
        }

        if let Some(instrumentor) = instrumentor.as_deref() {
            instrumentor.run_after_pipeline(op_name.as_ref(), parent_info.as_ref().unwrap());
        }

        Ok(())
    }

    /// The error for `pass`, which cannot run on an operation named `op_name` at `span`. The pass is
    /// named by its command-line name, or its display name when it has none.
    fn cannot_run_on(
        pass: &dyn OperationPass,
        op_name: &OperationName,
        span: SourceSpan,
    ) -> Report {
        let pass_name = match pass.argument() {
            "" => pass.name(),
            argument => argument,
        };
        crate::diagnostics::report!(
            message: (format!("pass '{pass_name}' cannot run on '{op_name}'")),
            labels: [
                primary(span, format!("pass '{pass_name}' does not run on this operation")),
            ],
        )
    }

    /// The error when a dynamic `pipeline` cannot be scheduled on `root`, if it cannot: its anchor
    /// is another operation, `root` is not `IsolatedFromAbove`, or one of its passes does not run
    /// on `root`.
    fn cannot_schedule_pipeline(pipeline: &OpPassManager, root: &Operation) -> Option<Report> {
        use crate::Spanned;

        let root_name = root.name();
        if pipeline.can_schedule_on(&root_name) {
            return None;
        }
        let span = root.span();
        if let Some(anchor) = pipeline.name() {
            return Some(crate::diagnostics::report!(
                message: (format!("a pipeline on '{anchor}' cannot run on '{root_name}'")),
                labels: [
                    primary(span, format!("the pipeline runs only on '{anchor}'")),
                ],
            ));
        }
        if !root_name.implements::<dyn IsolatedFromAbove>() {
            return Some(crate::diagnostics::report!(
                message: (format!("a pipeline cannot run on '{root_name}'")),
                labels: [
                    primary(span, "this operation does not implement `IsolatedFromAbove`"),
                ],
            ));
        }
        pipeline
            .passes()
            .iter()
            .find(|pass| !pass.can_schedule_on(&root_name))
            .map(|pass| Self::cannot_run_on(&**pass, &root_name, span))
    }

    /// Run the given operation and analysis manager on a single pass.
    fn run(
        pass: &mut dyn OperationPass,
        op: OperationRef,
        analysis_manager: AnalysisManager,
        verify: bool,
    ) -> Result<(), Report> {
        use crate::Spanned;

        let (op_name, span, context) = {
            let op = op.borrow();
            (op.name(), op.span(), op.context_rc())
        };
        if !op_name.implements::<dyn IsolatedFromAbove>() {
            return Err(crate::diagnostics::report!(
                message: "failed to execute pass",
                labels: [
                    primary(
                        span,
                        "trying to schedule a pass on an operation which does not implement \
                         `IsolatedFromAbove`"
                    ),
                ],
            ));
        }
        if !pass.can_schedule_on(&op_name) {
            return Err(Self::cannot_run_on(pass, &op_name, span));
        }

        // Initialize the pass state with a callback for the pass to dynamically execute a pipeline
        // on the currently visited operation.
        let pi = analysis_manager.pass_instrumentor();
        let parent_info = PipelineParentInfo {
            pass: Some(pass.name().to_compact_string()),
        };
        let callback_op = op;
        let callback_analysis_manager = analysis_manager.clone();
        let pipeline_callback: Box<super::pass::DynamicPipelineExecutor> = Box::new(
            move |pipeline: &mut OpPassManager, root: OperationRef| -> Result<(), Report> {
                let pi = callback_analysis_manager.pass_instrumentor();
                let op = callback_op.borrow();
                let root_op = root.borrow();
                if !root_op.is_ancestor_of(&op) {
                    return Err(crate::diagnostics::report!(
                        message: "failed to execute pass",
                        labels: [
                            primary(
                                root_op.span(),
                                "trying to schedule a dynamic pass pipeline on an operation that \
                                 isn't nested under the current operation the pass is processing"
                            ),
                        ],
                    ));
                }
                if let Some(error) = Self::cannot_schedule_pipeline(pipeline, &root_op) {
                    return Err(error);
                }
                // Before running, finalize the passes held by the pipeline
                pipeline.finalize_pass_list()?;

                // Initialize the user-provided pipeline and execute the pipeline
                pipeline.initialize()?;

                let nested_am = if root == callback_op {
                    callback_analysis_manager.clone()
                } else {
                    callback_analysis_manager.nest(root)
                };
                Self::run_pipeline(pipeline, root, nested_am, verify, pi, Some(&parent_info))
            },
        );

        let mut execution_state = PassExecutionState::new(
            op,
            context.clone(),
            analysis_manager.clone(),
            Some(pipeline_callback),
        );

        // Instrument before the pass has run
        if let Some(instrumentor) = pi.as_deref() {
            instrumentor.run_before_pass(pass, &op);
        }

        // A failure of a pass, or of the verifier after it, says which pass it was and what it ran
        // on. An adaptor adds nothing: the failure comes from a pass it ran, which said so.
        let is_adaptor = pass.as_any().is::<OpToOpPassAdaptor>();
        let mut result =
            if let Some(adaptor) = pass.as_any_mut().downcast_mut::<OpToOpPassAdaptor>() {
                adaptor.run_on_operation(op, &mut execution_state, verify)
            } else {
                pass.run_on_operation(op, &mut execution_state).map_err(|error| {
                    error.context(format!(
                        "pass '{}' failed on {}",
                        Self::command_line_name(pass),
                        Self::describe(&op)
                    ))
                })
            };

        // Invalidate any non-preserved analyses
        analysis_manager.invalidate(execution_state.preserved_analyses_mut());

        // When `verify == true`, we run the verifier (unless the pass failed)
        if result.is_ok() && verify {
            // Immediate adaptors already verified their children. Recursive adaptors also
            // traverse unmatched containers whose invariants may depend on changed descendants;
            // reverify the subtree before allowing the next stage to run.
            let run_verifier_recursively = pass
                .as_any()
                .downcast_ref::<OpToOpPassAdaptor>()
                .is_none_or(|adaptor| adaptor.recursive);

            // Reduce compile time by avoiding running the verifier if the pass didn't change the
            // IR since the last time the verifier was run:
            //
            // * If the pass said that it preserved all analyses then it can't have permuted the IR
            let run_verifier_now = !execution_state.preserved_analyses().is_all();

            if run_verifier_now
                && let Err(verification_result) = Self::verify(&op, run_verifier_recursively)
            {
                result = Err(if is_adaptor {
                    verification_result
                } else {
                    verification_result.context(format!(
                        "the IR that pass '{}' left on {} is invalid",
                        Self::command_line_name(pass),
                        Self::describe(&op)
                    ))
                });
            }
        }

        if let Some(instrumentor) = pi.as_deref() {
            if result.is_err() {
                instrumentor.run_after_pass_failed(pass, &op);
            } else {
                instrumentor.run_after_pass(pass, &op, &execution_state);
            }
        }

        // Return the pass result
        result.map(|_| ())
    }

    /// The command-line name of `pass`, or its display name when it has none.
    fn command_line_name(pass: &dyn OperationPass) -> &'static str {
        match pass.argument() {
            "" => pass.name(),
            argument => argument,
        }
    }

    /// The name of `op` in quotes, followed by its symbol name when it is a symbol:
    /// `'builtin.function' @f`.
    fn describe(op: &OperationRef) -> String {
        let op = op.borrow();
        match op.as_symbol() {
            Some(symbol) => format!("'{}' @{}", op.name(), symbol.name().as_str()),
            None => format!("'{}'", op.name()),
        }
    }

    fn verify(op: &OperationRef, verify_recursively: bool) -> Result<(), Report> {
        let op = op.borrow();
        if verify_recursively {
            op.recursively_verify()
        } else {
            op.verify()
        }
    }

    fn run_on_operation(
        &mut self,
        op: OperationRef,
        state: &mut PassExecutionState,
        verify: bool,
    ) -> Result<(), Report> {
        let analysis_manager = state.analysis_manager().clone();
        let instrumentor = analysis_manager.pass_instrumentor();
        let parent_info = PipelineParentInfo {
            pass: Some(self.name()),
        };

        let result = op.raw_prewalk::<Forward, _, _>(|nested| {
            // Scheduling always starts below the manager's anchor.
            if nested == op {
                return WalkResult::Continue(());
            }
            let name = nested.borrow().name();
            if let Some(manager) = self.pms.iter_mut().find(|pm| pm.can_schedule_on(&name)) {
                let am = analysis_manager.nest(nested);
                match Self::run_pipeline(
                    manager,
                    nested,
                    am,
                    verify,
                    instrumentor.clone(),
                    Some(&parent_info),
                ) {
                    // A matching pipeline owns its subtree, including further nested scheduling.
                    Ok(()) => WalkResult::Skip,
                    Err(err) => WalkResult::Break(err),
                }
            } else if self.recursive {
                WalkResult::Continue(())
            } else {
                WalkResult::Skip
            }
        });
        if let WalkResult::Break(err) = result {
            return Err(err);
        }
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}

impl Pass for OpToOpPassAdaptor {
    type Target = Operation;

    fn name(&self) -> &'static str {
        crate::interner::Symbol::intern(self.name()).as_str()
    }

    #[inline(always)]
    fn target_name(&self, _context: &Context) -> Option<OperationName> {
        None
    }

    fn print_as_textual_pipeline(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.recursive {
            f.write_str("recursive(")?;
        }
        let pms = self.pass_managers();
        for (i, pm) in pms.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            pm.print_as_textual_pipeline(f)?;
        }

        if self.recursive {
            f.write_str(")")?;
        }

        Ok(())
    }

    #[inline(always)]
    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Operation>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        unreachable!("unexpected call to `Pass::run_on_operation` for OpToOpPassAdaptor")
    }
}

#[cfg(test)]
#[path = "manager/tests.rs"]
mod scheduling_tests;

#[cfg(test)]
#[path = "manager/printing_tests.rs"]
mod printing_tests;

#[cfg(all(test, feature = "std"))]
#[path = "manager/scheduling_model.rs"]
mod scheduling_model;

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;

    use crate::{
        EntityMut, Operation, OperationName, Report, SourceSpan,
        dialects::{builtin::BuiltinOpBuilder, test::TestOpBuilder},
        pass::{Pass, PassExecutionState},
        testing::Test,
    };

    /// A pass that corrupts BinaryOp operations by clearing their operands.
    struct CorruptBinaryOp;

    impl Pass for CorruptBinaryOp {
        type Target = Operation;

        fn name(&self) -> &'static str {
            "corrupt-binary-op"
        }

        fn can_schedule_on(&self, _name: &OperationName) -> bool {
            true
        }

        fn run_on_operation(
            &mut self,
            op: EntityMut<'_, Operation>,
            _state: &mut PassExecutionState,
        ) -> Result<(), Report> {
            // Navigate to the first operation with 2 operands (a BinaryOp)
            // and remove its operands, breaking the BinaryOp invariant
            let op_ref = op.as_operation_ref();
            drop(op);
            let borrowed = op_ref.borrow();
            if let Some(region) = borrowed.regions().front().as_pointer()
                && let Some(block) = region.borrow().body().front().as_pointer()
            {
                let mut next_op = block.borrow().front();
                while let Some(mut inner) = next_op.take() {
                    if inner.borrow().num_operands() == 2 {
                        inner.borrow_mut().set_operands(core::iter::empty());
                        return Ok(());
                    }
                    next_op = inner.next();
                }
            }
            Ok(())
        }
    }

    #[test]
    fn pass_manager_detects_broken_invariant() {
        let mut test = Test::named("test_verify");
        let i32 = test.context().i32();
        test.with_function("test_verify", &[i32, i32], &[i32]);

        {
            let entry = test.entry_block();
            let lhs = entry.borrow().get_argument(0).borrow().as_value_ref();
            let rhs = entry.borrow().get_argument(1).borrow().as_value_ref();
            let mut fb = test.function_builder();
            let sum = fb.add(lhs, rhs, SourceSpan::UNKNOWN).unwrap();
            fb.ret([sum], SourceSpan::UNKNOWN).unwrap();
        }

        // Run the corrupting pass with verification enabled
        let result = test.apply_boxed_pass(Box::new(CorruptBinaryOp), true);
        assert!(result.is_err(), "PassManager should detect broken BinaryOp invariant");
    }
}
