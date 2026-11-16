//! The optimization step: the pass pipeline is built, run on the world, and `ir.optimized` is
//! reached.

use core::{cell::RefCell, fmt, ops::ControlFlow};
use std::rc::Rc;

use runic_ir::{
    OpPrintingFlags, Report,
    pass::{
        Anchor, IRPrintingConfig, Nesting, OpPassManager, PassInfo, PassManager, PassPipeline,
        PassSelection,
    },
};
use runic_transform::{
    Canonicalizer, CommonSubexpressionElimination, DeadCodeElimination,
    SparseConditionalConstantPropagation,
};

use super::{
    Session,
    checkpoints::{Checkpoints, Progress, Stage},
};
use crate::{CheckpointId, Flow, Observer, OptLevel, Verification};

/// The name of the operation the optimization pipeline runs on.
const WORLD: &str = "builtin.world";

/// Adds the default optimization passes for the optimization level of `session`'s request: none at
/// `-O0`, and otherwise `canonicalizer`, `cse`, `sccp`, `canonicalizer` and `dce`, in that order.
pub(crate) fn default_optimization(
    session: &Session<'_>,
    passes: &mut OpPassManager,
) -> Result<(), Report> {
    if session.request().opt_level == OptLevel::None {
        return Ok(());
    }
    passes.add_pass(Box::<Canonicalizer>::default())?;
    passes.add_pass(Box::<CommonSubexpressionElimination>::default())?;
    passes.add_pass(Box::<SparseConditionalConstantPropagation>::default())?;
    passes.add_pass(Box::<Canonicalizer>::default())?;
    passes.add_pass(Box::<DeadCodeElimination>::default())?;
    Ok(())
}

/// What the pass manager prints, shared with the step that keeps it.
#[derive(Clone, Default)]
struct Buffer(Rc<RefCell<String>>);

impl fmt::Write for Buffer {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0.borrow_mut().push_str(text);
        Ok(())
    }
}

/// The text of a loaded pass manager's pipeline.
struct Textual<'a>(&'a PassManager);

impl fmt::Display for Textual<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.print_as_textual_pipeline(f)
    }
}

impl Session<'_> {
    /// What the pass pipeline printed: the pipeline itself, when the request asked to dump it,
    /// and the IR before and after the passes the request selected, in the order they were
    /// printed, for the optimization pipeline and then for the lowering pipeline. It is empty until
    /// the optimize step has run, and holds what was printed up to the failure when a step failed.
    pub fn pass_output(&self) -> &str {
        &self.progress.pass_output
    }

    /// Runs the optimize step: the optimization pipeline is built, run on the world, and the
    /// harness reaches `ir.optimized` with the world.
    ///
    /// The pipeline is the request's own text when it has one (it must be anchored on
    /// `builtin.world` or `any`, or on nothing). Otherwise it is the passes each frontend that has
    /// inputs adds, in the order the frontends ran, followed by the passes of the compiler's
    /// optimization (the default for the optimization level unless the compiler's builder gave
    /// one). The request's IR options apply: the pipeline is dumped, the IR is printed before and
    /// after the passes it names, and the IR is verified after each pass, once after the last, or
    /// never. Everything printed is kept in [Session::pass_output].
    ///
    /// `observer` is told of the checkpoint reached. The answer is `Break` when the compilation
    /// was asked to stop at `ir.optimized`, and then [Session::stopped] holds its artifact. It is
    /// an error to run the step before the translate step reached `ir.translated`, after the
    /// compilation stopped, or twice, to name a pass that is not registered to print IR for, to
    /// give a pipeline that does not read, load or anchor on the world, and for a pass or the
    /// verification to fail; the error is returned as it is.
    pub fn optimize(&mut self, observer: &mut dyn Observer) -> Result<Flow<()>, Report> {
        self.step("optimize", |session, progress| session.run_optimize(progress, observer))
    }

    fn run_optimize(
        &self,
        progress: &mut Progress,
        observer: &mut dyn Observer,
    ) -> Result<Flow<()>, Report> {
        if progress.optimized {
            return Err(Report::msg("the optimize step was run twice: a session optimizes once"));
        }
        if progress.stopped.is_some() {
            return Err(Report::msg(format!(
                "the optimize step cannot run: the compilation stopped at '{}'",
                self.goal.stop_after()
            )));
        }
        if !progress.reached.contains(&CheckpointId::IR_TRANSLATED) {
            return Err(Report::msg(
                "the optimize step cannot run before the translate step reached 'ir.translated'",
            ));
        }
        progress.optimized = true;

        self.run_pipeline(progress, || self.build_pipeline(), "optimization")?;

        let mut checkpoints =
            Checkpoints::new(progress, observer, &self.route, &self.goal, self, Stage::Harness);
        match checkpoints.reach(CheckpointId::IR_OPTIMIZED, self.world)? {
            ControlFlow::Break(stopped) => Ok(ControlFlow::Break(stopped)),
            ControlFlow::Continue(_) => Ok(ControlFlow::Continue(())),
        }
    }

    /// Runs the lowering step: the backend's passes are built, run on the world, and the harness
    /// reaches `ir.lowered` with the world.
    ///
    /// The pipeline is the passes the backend of the target adds, anchored on the world; a
    /// request's pipeline applies to the optimization step only and does not change it. The
    /// request's IR options apply as they do to [Session::optimize]: the pipeline is dumped on a
    /// line of its own after the optimization pipeline's, the IR is printed before and after the
    /// passes the request names, and the IR is verified after each pass, once after the last, or
    /// never. Everything printed is added to [Session::pass_output].
    ///
    /// `observer` is told of the checkpoint reached. The answer is `Break` when the compilation
    /// was asked to stop at `ir.lowered`, and then [Session::stopped] holds its artifact. It is an
    /// error to run the step when the compilation has no target and so no backend, before the
    /// optimize step reached `ir.optimized`, after the compilation stopped, or twice, and for the
    /// backend to fail to add its passes (the error says which backend), for a pass to fail, or
    /// for the verification to fail; the error is returned as it is.
    pub fn lower(&mut self, observer: &mut dyn Observer) -> Result<Flow<()>, Report> {
        self.step("lower", |session, progress| session.run_lower(progress, observer))
    }

    fn run_lower(
        &self,
        progress: &mut Progress,
        observer: &mut dyn Observer,
    ) -> Result<Flow<()>, Report> {
        let Some(target) = &self.target else {
            return Err(Report::msg(
                "the lower step cannot run: the route of this compilation has no backend, because \
                 no target was given",
            ));
        };
        if progress.lowered {
            return Err(Report::msg("the lower step was run twice: a session lowers once"));
        }
        if progress.stopped.is_some() {
            return Err(Report::msg(format!(
                "the lower step cannot run: the compilation stopped at '{}'",
                self.goal.stop_after()
            )));
        }
        if !progress.reached.contains(&CheckpointId::IR_OPTIMIZED) {
            return Err(Report::msg(
                "the lower step cannot run before the optimize step reached 'ir.optimized'",
            ));
        }
        progress.lowered = true;

        let backend = target.backend;
        self.run_pipeline(
            progress,
            || {
                let mut pm = PassManager::new(self.context.clone(), WORLD, Nesting::Implicit)?;
                backend.add_passes(self, pm.op_pass_manager_mut()).map_err(|error| {
                    error.context(format!(
                        "backend '{}' could not add its lowering passes",
                        backend.name()
                    ))
                })?;
                Ok(pm)
            },
            "lowering",
        )?;

        let mut checkpoints =
            Checkpoints::new(progress, observer, &self.route, &self.goal, self, Stage::Harness);
        match checkpoints.reach(CheckpointId::IR_LOWERED, self.world)? {
            ControlFlow::Break(stopped) => Ok(ControlFlow::Break(stopped)),
            ControlFlow::Continue(_) => Ok(ControlFlow::Continue(())),
        }
    }

    /// Runs the pipeline `build` makes on the world with the request's IR options, for the step
    /// that `what` names (`optimization` or `lowering`): the IR printing selections are checked
    /// before anything is built, the pipeline is dumped when the request asked, and the IR is
    /// verified after each pass, once after the last, or never. What was printed is added to the
    /// session's pass output whether or not the pipeline succeeded.
    fn run_pipeline(
        &self,
        progress: &mut Progress,
        build: impl FnOnce() -> Result<PassManager, Report>,
        what: &str,
    ) -> Result<(), Report> {
        let ir = &self.request.ir;
        let printing = IRPrintingConfig {
            before: selection("before", &ir.print_before)?,
            after: selection("after", &ir.print_after)?,
            flags: OpPrintingFlags {
                print_entry_block_headers: false,
                print_source_locations: ir.print_locations,
                generic: ir.print_generic,
            },
            ..Default::default()
        };

        let mut pm = build()?;
        pm.enable_verifier(ir.verification == Verification::EachPass);
        let output = Buffer::default();
        pm.enable_ir_printing(printing, Box::new(output.clone()));
        if ir.dump_pass_pipeline {
            output.0.borrow_mut().push_str(&format!("{}\n", Textual(&pm)));
        }

        let result = pm.run(self.world.as_operation_ref());
        drop(pm);
        progress.pass_output.push_str(&output.0.borrow());
        result?;

        if ir.verification == Verification::Once {
            self.world.as_operation_ref().borrow().recursively_verify().map_err(|report| {
                report.context(format!("the IR the {what} pipeline produced does not verify"))
            })?;
        }
        Ok(())
    }

    /// The pass manager of the optimization pipeline, anchored on the world: loaded from the
    /// request's text, or built from the frontends' passes and the compiler's.
    fn build_pipeline(&self) -> Result<PassManager, Report> {
        let context = self.context.clone();
        if let Some(text) = &self.request.pipeline {
            let pipeline = self.read_pipeline(text)?;
            return pipeline.load(context, Nesting::Implicit).map_err(|error| {
                error.context(format!("cannot load the optimization pipeline '{text}'"))
            });
        }
        let mut pm = PassManager::new(context, WORLD, Nesting::Implicit)?;
        for (frontend, _) in &self.frontends {
            frontend.add_passes(self, pm.op_pass_manager_mut()).map_err(|error| {
                error.context(format!(
                    "frontend '{}' could not add its optimization passes",
                    frontend.name()
                ))
            })?;
        }
        self.compiler
            .add_optimization_passes(self, pm.op_pass_manager_mut())
            .map_err(|error| error.context("could not add the optimization passes"))?;
        Ok(pm)
    }

    /// Reads `text`: text that parses as a whole pipeline must be anchored on the world or on
    /// `any`; any other text is a list of passes anchored on the world.
    fn read_pipeline(&self, text: &str) -> Result<PassPipeline, Report> {
        let name = self.world.as_operation_ref().borrow().name();
        match PassPipeline::parse(text) {
            Ok(pipeline) => match pipeline.anchor() {
                Anchor::Any => Ok(pipeline),
                Anchor::Operation { dialect, opcode }
                    if *dialect == name.dialect() && *opcode == name.name() =>
                {
                    Ok(pipeline)
                }
                anchor => Err(Report::msg(format!(
                    "the optimization pipeline '{text}' is anchored on '{anchor}', but the \
                     optimization pipeline runs on the world: anchor it on '{WORLD}' or 'any', or \
                     leave the anchor out"
                ))),
            },
            Err(_) => PassPipeline::parse_elements(&name, text).map_err(|error| {
                error.context(format!("cannot read the optimization pipeline '{text}'"))
            }),
        }
    }
}

/// The passes `names` select to print IR `when` (`before` or `after`): none, `all`, or registered passes.
fn selection(when: &str, names: &[String]) -> Result<PassSelection, Report> {
    if names.is_empty() {
        return Ok(PassSelection::None);
    }
    if names.iter().any(|name| name == "all") {
        return Ok(PassSelection::All);
    }
    for name in names {
        if PassInfo::lookup(name).is_none() {
            let hint = match PassInfo::similar_to(name).as_slice() {
                [] => String::new(),
                [one] => format!("; did you mean '{one}'?"),
                many => format!(
                    "; did you mean one of {}?",
                    many.iter().map(|name| format!("'{name}'")).collect::<Vec<_>>().join(", ")
                ),
            };
            return Err(Report::msg(format!(
                "unknown pass '{name}' named to print the IR {when} it{hint}"
            )));
        }
    }
    Ok(PassSelection::Named(names.to_vec()))
}

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;
    use std::sync::LazyLock;

    use runic_ir::{
        OpPrinter, OpPrintingFlags, Report,
        dialects::builtin::{Module, WorldRef},
        pass::{Nesting, PassInfo, PassOptions, PassPipeline, registry::RegistryEntry},
        print::AsmPrinter,
    };

    use crate::{
        CheckpointDecl, CheckpointId, CompilationRequest, Compiler, EmitRequest, EmitTo, Emitted,
        Input, IrFrontend, OptLevel, RecordingObserver, Session, Verification,
        testing::{TestBackend, TestFrontend, add_mark, decl},
    };

    const ALPHA: &[CheckpointDecl] = &[decl("alpha.ast", &[], "ast")];
    const ALPHA_FRONTEND: TestFrontend = TestFrontend::new("alpha", &["a"])
        .with_checkpoints(ALPHA)
        .with_add_passes(|_, passes| add_mark(passes, "from_alpha"));
    const BETA_FRONTEND: TestFrontend = TestFrontend::new("beta", &["b"])
        .with_add_passes(|_, passes| add_mark(passes, "from_beta"));

    fn compiler() -> &'static Compiler {
        static COMPILER: LazyLock<Compiler> = LazyLock::new(|| {
            Compiler::builder()
                .frontend(ALPHA_FRONTEND)
                .frontend(BETA_FRONTEND)
                .build()
                .unwrap()
        });
        &COMPILER
    }

    fn request(inputs: &[&str]) -> CompilationRequest {
        let mut request = CompilationRequest::new();
        request.inputs = inputs.iter().map(|name| Input::memory(*name, "")).collect();
        // The one default output is kept in memory, not written to the working directory.
        request.output_file = Some("-".parse().unwrap());
        request
    }

    /// A session that has translated, with the world holding what its frontends made.
    fn translated(compiler: &Compiler, request: CompilationRequest) -> Session<'_> {
        let mut session = compiler.session(request).unwrap();
        let _ = session.translate(&mut ()).unwrap();
        session
    }

    fn optimized(compiler: &Compiler, request: CompilationRequest) -> Session<'_> {
        let mut session = translated(compiler, request);
        let _ = session.optimize(&mut ()).unwrap();
        session
    }

    /// The error of the optimize step: its message, then each context it was given.
    fn failure(compiler: &Compiler, request: CompilationRequest) -> String {
        let report = translated(compiler, request).optimize(&mut ()).unwrap_err();
        let mut text = report.to_string();
        for context in report.contexts() {
            text.push('\n');
            text.push_str(context.message());
        }
        text
    }

    fn print(world: WorldRef) -> String {
        let operation = world.as_operation_ref();
        let operation = operation.borrow();
        let flags = OpPrintingFlags::default();
        let mut printer = AsmPrinter::root(operation.context_rc(), &flags);
        operation.print(&mut printer);
        printer.finish().to_string()
    }

    /// The names of the modules in the world, in order: the marks the passes left.
    fn marks(session: &Session<'_>) -> Vec<String> {
        let mut names = Vec::new();
        session.world().as_operation_ref().borrow().postwalk_all(|op| {
            if let Some(module) = op.downcast_ref::<Module>() {
                names.push(module.name().as_str().to_owned());
            }
        });
        names
    }

    fn pipeline_text(compiler: &Compiler, request: CompilationRequest) -> String {
        let mut request = request;
        request.ir.dump_pass_pipeline = true;
        let session = optimized(compiler, request);
        session.pass_output().lines().next().unwrap().to_owned()
    }

    const DEFAULT: &str = "canonicalizer,cse,sccp,canonicalizer,dce";

    // The pipeline that is built

    #[test]
    fn the_passes_of_the_frontends_come_first_in_the_order_they_ran_then_the_default() {
        let text = pipeline_text(compiler(), request(&["x.b", "y.a"]));
        assert_eq!(
            text,
            format!(
                "builtin.world(test-mark{{name=\"from_beta\"}},test-mark{{name=\"from_alpha\"}},\
                 {DEFAULT})"
            )
        );
    }

    #[test]
    fn the_frontends_passes_ran_in_the_order_of_the_frontends() {
        let session = optimized(compiler(), request(&["x.a", "y.b"]));
        assert_eq!(marks(&session), ["from_alpha", "from_beta"]);
        let session = optimized(compiler(), request(&["x.b", "y.a"]));
        assert_eq!(marks(&session), ["from_beta", "from_alpha"]);
    }

    #[test]
    fn a_frontend_with_no_inputs_adds_no_passes() {
        let text = pipeline_text(compiler(), request(&["x.a"]));
        assert!(text.contains("from_alpha") && !text.contains("from_beta"), "{text}");
    }

    #[test]
    fn the_default_runs_at_every_level_but_none_and_adds_nothing_at_none() {
        for (level, expected) in [
            (OptLevel::None, ""),
            (OptLevel::Basic, DEFAULT),
            (OptLevel::Balanced, DEFAULT),
            (OptLevel::Max, DEFAULT),
            (OptLevel::Size, DEFAULT),
            (OptLevel::SizeMin, DEFAULT),
        ] {
            let mut request = request(&["x.b"]);
            request.opt_level = level;
            let text = pipeline_text(compiler(), request);
            let defaults =
                text.strip_prefix("builtin.world(test-mark{name=\"from_beta\"}").unwrap();
            let defaults = defaults.strip_suffix(')').unwrap().trim_start_matches(',');
            assert_eq!(defaults, expected, "at level {level}");
        }
    }

    #[test]
    fn at_the_first_level_the_frontends_passes_still_run() {
        let mut request = request(&["x.a", "y.b"]);
        request.opt_level = OptLevel::None;
        let session = optimized(compiler(), request);
        assert_eq!(marks(&session), ["from_alpha", "from_beta"]);
    }

    #[test]
    fn a_builder_override_replaces_the_default_and_follows_the_frontends() {
        let compiler = Compiler::builder()
            .frontend(ALPHA_FRONTEND)
            .optimization(|session, passes| {
                assert_eq!(session.name(), "x");
                add_mark(passes, "from_compiler")
            })
            .build()
            .unwrap();
        let mut request = request(&["x.a"]);
        request.ir.dump_pass_pipeline = true;
        let session = optimized(&compiler, request);
        assert_eq!(marks(&session), ["from_alpha", "from_compiler"]);
        let text = session.pass_output().lines().next().unwrap();
        assert_eq!(
            text,
            "builtin.world(test-mark{name=\"from_alpha\"},test-mark{name=\"from_compiler\"})"
        );
    }

    #[test]
    fn a_builder_override_runs_at_the_first_level_too() {
        let compiler = Compiler::builder()
            .frontend(BETA_FRONTEND)
            .optimization(|_, passes| add_mark(passes, "from_compiler"))
            .build()
            .unwrap();
        let mut request = request(&["x.b"]);
        request.opt_level = OptLevel::None;
        assert_eq!(marks(&optimized(&compiler, request)), ["from_beta", "from_compiler"]);
    }

    #[test]
    fn a_failing_builder_override_is_an_error_that_says_where_it_came_from() {
        let compiler = Compiler::builder()
            .frontend(BETA_FRONTEND)
            .optimization(|_, _| Err(Report::msg("no passes today")))
            .build()
            .unwrap();
        let message = failure(&compiler, request(&["x.b"]));
        assert!(message.contains("no passes today"), "{message}");
        assert!(message.contains("optimization passes"), "{message}");
    }

    #[test]
    fn a_failing_frontend_names_itself() {
        let frontend = TestFrontend::new("gamma", &["g"])
            .with_add_passes(|_, _| Err(Report::msg("gamma has no passes")));
        let compiler = Compiler::builder().frontend(frontend).build().unwrap();
        let message = failure(&compiler, request(&["x.g"]));
        assert!(
            message.contains("'gamma'") && message.contains("gamma has no passes"),
            "{message}"
        );
    }

    // The request's pipeline

    #[test]
    fn the_requests_pipeline_replaces_the_frontends_and_the_default() {
        let mut request = request(&["x.a", "y.b"]);
        request.pipeline = Some("test-mark{name=p1},test-mark{name=p2}".into());
        request.ir.dump_pass_pipeline = true;
        let session = optimized(compiler(), request);
        assert_eq!(marks(&session), ["p1", "p2"]);
        assert_eq!(
            session.pass_output().lines().next().unwrap(),
            "builtin.world(test-mark{name=\"p1\"},test-mark{name=\"p2\"})"
        );
    }

    #[test]
    fn the_requests_pipeline_replaces_a_builder_override_too() {
        let compiler = Compiler::builder()
            .frontend(ALPHA_FRONTEND)
            .optimization(|_, passes| add_mark(passes, "from_compiler"))
            .build()
            .unwrap();
        let mut request = request(&["x.a"]);
        request.pipeline = Some("test-mark{name=p1}".into());
        assert_eq!(marks(&optimized(&compiler, request)), ["p1"]);
    }

    #[test]
    fn the_requests_pipeline_may_be_anchored_on_the_world_or_on_any() {
        for text in [
            "builtin.world(test-mark{name=p1})",
            "any(test-mark{name=p1})",
            "test-mark{name=p1}",
        ] {
            let mut request = request(&["x.a"]);
            request.pipeline = Some(text.into());
            assert_eq!(marks(&optimized(compiler(), request)), ["p1"], "{text}");
        }
    }

    #[test]
    fn an_empty_requests_pipeline_runs_nothing() {
        let mut request = request(&["x.a"]);
        request.pipeline = Some(String::new());
        assert_eq!(marks(&optimized(compiler(), request)), Vec::<String>::new());
    }

    #[test]
    fn a_requests_pipeline_with_a_syntax_error_is_an_error_that_names_the_text() {
        let mut request = request(&["x.a"]);
        request.pipeline = Some("test-mark{name=".into());
        let message = failure(compiler(), request);
        assert!(message.contains("'test-mark{name='"), "{message}");
    }

    #[test]
    fn a_requests_pipeline_with_an_unknown_pass_is_an_error_that_names_the_text() {
        let mut request = request(&["x.a"]);
        request.pipeline = Some("test-mark,no-such-pass".into());
        let message = failure(compiler(), request);
        assert!(message.contains("'test-mark,no-such-pass'"), "{message}");
        assert!(message.contains("no-such-pass"), "{message}");
    }

    #[test]
    fn a_requests_pipeline_with_an_unknown_option_is_an_error_that_names_the_text() {
        let mut request = request(&["x.a"]);
        request.pipeline = Some("test-mark{colour=red}".into());
        let message = failure(compiler(), request);
        assert!(message.contains("'test-mark{colour=red}'"), "{message}");
        assert!(message.contains("colour"), "{message}");
    }

    #[test]
    fn a_requests_pipeline_anchored_off_the_world_is_an_error_that_names_the_text() {
        let mut request = request(&["x.a"]);
        request.pipeline = Some("builtin.module(test-mark)".into());
        let message = failure(compiler(), request);
        assert!(message.contains("'builtin.module(test-mark)'"), "{message}");
        assert!(message.contains("runs on the world"), "{message}");
        assert!(message.contains("'builtin.module'"), "{message}");
    }

    // The dump

    #[test]
    fn the_dumped_pipeline_reads_back_as_the_same_pipeline() {
        for (level, pipeline) in [
            (OptLevel::Balanced, None),
            (OptLevel::None, None),
            (OptLevel::Balanced, Some("test-mark{name=p1},cse")),
        ] {
            let mut request = request(&["x.b", "y.a"]);
            request.opt_level = level;
            request.pipeline = pipeline.map(Into::into);
            let text = pipeline_text(compiler(), request);

            let compiler = compiler();
            let session = compiler.session(self::request(&["x.a"])).unwrap();
            let mut loaded = PassPipeline::parse(&text)
                .unwrap()
                .load(session.context().clone(), Nesting::Implicit)
                .unwrap();
            let again = format!("{}", super::Textual(&loaded));
            assert_eq!(again, text);
            // And it runs.
            loaded.run(session.world().as_operation_ref()).unwrap();
        }
    }

    #[test]
    fn the_pipeline_is_dumped_only_when_asked_and_before_anything_else_is_printed() {
        let session = optimized(compiler(), request(&["x.a"]));
        assert_eq!(session.pass_output(), "");

        let mut request = request(&["x.a"]);
        request.ir.dump_pass_pipeline = true;
        request.ir.print_after = vec!["test-mark".into()];
        let session = optimized(compiler(), request);
        let output = session.pass_output();
        assert!(output.starts_with("builtin.world("), "{output}");
        assert!(output.lines().next().unwrap().ends_with(")"), "{output}");
        assert!(output.lines().count() > 1, "{output}");
    }

    // Verification

    fn breaking(verification: Verification) -> CompilationRequest {
        let mut request = request(&["x.a"]);
        request.pipeline = Some("test-mark{name=before},test-break,test-mark{name=after}".into());
        request.ir.verification = verification;
        request
    }

    #[test]
    fn verifying_each_pass_stops_at_the_pass_that_broke_the_ir() {
        let compiler = compiler();
        let mut session = translated(compiler, breaking(Verification::EachPass));
        let message = session.optimize(&mut ()).unwrap_err().to_string();
        assert!(message.contains("does not end with a terminator"), "{message}");
        assert_eq!(marks(&session), ["before", "broken"]);
        assert_eq!(session.reached(), [CheckpointId::IR_TRANSLATED]);
    }

    #[test]
    fn verifying_once_runs_the_whole_pipeline_and_then_fails() {
        let compiler = compiler();
        let mut session = translated(compiler, breaking(Verification::Once));
        let error = session.optimize(&mut ()).unwrap_err();
        let frames: Vec<_> =
            error.contexts().iter().map(|frame| frame.message().to_string()).collect();
        assert!(
            frames.iter().any(|frame| frame.contains("the optimization pipeline produced")),
            "{frames:?} / {error}"
        );
        assert_eq!(marks(&session), ["before", "broken", "after"]);
        assert_eq!(session.reached(), [CheckpointId::IR_TRANSLATED]);
    }

    #[test]
    fn with_verification_off_a_broken_world_is_optimized() {
        let compiler = compiler();
        let session = optimized(compiler, breaking(Verification::None));
        assert_eq!(marks(&session), ["before", "broken", "after"]);
        assert_eq!(session.reached().last(), Some(&CheckpointId::IR_OPTIMIZED));
    }

    #[test]
    fn a_world_that_verifies_passes_in_every_mode() {
        for verification in [Verification::EachPass, Verification::Once, Verification::None] {
            let mut request = request(&["x.a"]);
            request.ir.verification = verification;
            optimized(compiler(), request);
        }
    }

    // Printing

    fn printing(before: &[&str], after: &[&str]) -> CompilationRequest {
        let mut request = request(&["x.a"]);
        request.pipeline = Some("test-mark{name=one},test-mark{name=two}".into());
        request.ir.print_before = before.iter().map(|name| name.to_string()).collect();
        request.ir.print_after = after.iter().map(|name| name.to_string()).collect();
        request
    }

    #[test]
    fn the_ir_is_printed_before_and_after_a_named_pass() {
        let session = optimized(compiler(), printing(&["test-mark"], &["test-mark"]));
        let output = session.pass_output();
        assert_eq!(output.matches("IR before").count(), 2, "{output}");
        assert_eq!(output.matches("IR after").count(), 2, "{output}");
        let before = output.find("IR before").unwrap();
        let after = output.find("IR after").unwrap();
        assert!(before < after, "{output}");
        assert!(output.contains("@one") && output.contains("@two"), "{output}");
    }

    #[test]
    fn only_the_selected_side_is_printed() {
        let session = optimized(compiler(), printing(&[], &["test-mark"]));
        let output = session.pass_output();
        assert!(!output.contains("IR before") && output.contains("IR after"), "{output}");
        let session = optimized(compiler(), printing(&["test-mark"], &[]));
        let output = session.pass_output();
        assert!(output.contains("IR before") && !output.contains("IR after"), "{output}");
    }

    #[test]
    fn printing_all_selects_every_pass() {
        let mut request = request(&["x.a"]);
        request.pipeline = Some("test-mark{name=one},cse,test-mark{name=two}".into());
        request.ir.print_after = vec!["all".into()];
        let session = optimized(compiler(), request);
        assert_eq!(session.pass_output().matches("IR after").count(), 3);
    }

    #[test]
    fn printing_after_a_pass_that_does_not_run_prints_nothing() {
        let session = optimized(compiler(), printing(&[], &["cse"]));
        assert_eq!(session.pass_output(), "");
    }

    #[test]
    fn an_unknown_pass_to_print_is_an_error_before_anything_runs() {
        let compiler = compiler();
        for request in [printing(&["test-mrak"], &[]), printing(&[], &["test-mrak"])] {
            let mut session = translated(compiler, request);
            let message = session.optimize(&mut ()).unwrap_err().to_string();
            assert!(message.contains("'test-mrak'"), "{message}");
            assert!(
                message.contains("did you mean") && message.contains("'test-mark'"),
                "{message}"
            );
            assert!(marks(&session).is_empty());
            assert_eq!(session.pass_output(), "");
        }
    }

    #[test]
    fn what_was_printed_stays_in_the_pass_output_when_a_pass_fails() {
        let mut request = printing(&["test-mark"], &[]);
        request.pipeline = Some("test-mark{name=one},test-break".into());
        request.ir.print_before = vec!["all".into()];
        let compiler = compiler();
        let mut session = translated(compiler, request);
        session.optimize(&mut ()).unwrap_err();
        assert_eq!(session.pass_output().matches("IR before").count(), 2);
    }

    #[test]
    fn the_printing_flags_apply_to_the_ir_printed_between_passes() {
        let mut generic = printing(&[], &["test-mark"]);
        generic.ir.print_generic = true;
        let generic = optimized(compiler(), generic);
        assert!(
            generic.pass_output().contains("\"builtin.module\""),
            "{}",
            generic.pass_output()
        );
        let plain = optimized(compiler(), printing(&[], &["test-mark"]));
        assert!(!plain.pass_output().contains("\"builtin.module\""), "{}", plain.pass_output());
    }

    // Order and errors of the step

    #[test]
    fn optimizing_before_translating_is_an_error() {
        let compiler = compiler();
        let mut session = compiler.session(request(&["x.a"])).unwrap();
        let message = session.optimize(&mut ()).unwrap_err().to_string();
        assert!(message.contains("before the translate step"), "{message}");
        assert!(session.reached().is_empty());
        // Nothing was used up: the step can still run in its place.
        let _ = session.translate(&mut ()).unwrap();
        let _ = session.optimize(&mut ()).unwrap();
    }

    #[test]
    fn optimizing_twice_is_an_error() {
        let compiler = compiler();
        let mut session = optimized(compiler, request(&["x.a"]));
        let message = session.optimize(&mut ()).unwrap_err().to_string();
        assert!(message.contains("run twice"), "{message}");
        assert_eq!(marks(&session), ["from_alpha"]);
    }

    #[test]
    fn optimizing_after_a_stop_at_translated_is_an_error() {
        let compiler = compiler();
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.translated".into());
        let mut session = translated(compiler, request);
        assert!(session.stopped().is_some());
        let message = session.optimize(&mut ()).unwrap_err().to_string();
        assert!(message.contains("the compilation stopped at 'ir.translated'"), "{message}");
        assert!(marks(&session).is_empty());
    }

    #[test]
    fn optimizing_after_a_stop_in_a_frontend_is_an_error() {
        let frontend = TestFrontend::new("alpha", &["a"]).with_checkpoints(ALPHA).with_translate(
            |_, _, _, checkpoints| {
                Ok(checkpoints.reach(CheckpointId::new("alpha.ast"), 0u8)?.to_unit())
            },
        );
        let compiler = Compiler::builder().frontend(frontend).build().unwrap();
        let mut request = request(&["x.a"]);
        request.stop_after = Some("alpha.ast".into());
        let mut session = translated(&compiler, request);
        let message = session.optimize(&mut ()).unwrap_err().to_string();
        assert!(message.contains("the compilation stopped at 'alpha.ast'"), "{message}");
    }

    trait ToUnit {
        fn to_unit(self) -> crate::Flow<()>;
    }

    impl<T> ToUnit for crate::Flow<T> {
        fn to_unit(self) -> crate::Flow<()> {
            match self {
                ControlFlow::Continue(_) => ControlFlow::Continue(()),
                ControlFlow::Break(stopped) => ControlFlow::Break(stopped),
            }
        }
    }

    // Reaching ir.optimized

    #[test]
    fn the_step_reaches_the_optimized_ir_and_tells_the_observer() {
        let compiler = compiler();
        let mut session = translated(compiler, request(&["x.a"]));
        let mut observer = RecordingObserver::default();
        // With no backend the compilation stops at the optimized IR.
        let flow = session.optimize(&mut observer).unwrap();
        assert!(matches!(flow, ControlFlow::Break(_)));
        assert_eq!(session.reached(), [CheckpointId::IR_TRANSLATED, CheckpointId::IR_OPTIMIZED]);
        assert_eq!(observer.checkpoints(), [CheckpointId::IR_OPTIMIZED]);
    }

    #[test]
    fn a_stop_at_the_optimized_ir_is_honoured() {
        let compiler = compiler();
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.optimized".into());
        let mut session = translated(compiler, request);
        let flow = session.optimize(&mut ()).unwrap();
        match flow {
            ControlFlow::Break(stopped) => {
                assert_eq!(stopped.checkpoint(), CheckpointId::IR_OPTIMIZED)
            }
            ControlFlow::Continue(()) => panic!("the compilation should have stopped"),
        }
        let artifact = session.stopped().unwrap();
        assert_eq!(artifact.checkpoint(), CheckpointId::IR_OPTIMIZED);
        assert!(artifact.downcast_ref::<WorldRef>().is_some());
    }

    #[test]
    fn an_output_at_each_ir_checkpoint_holds_the_ir_as_it_was_there() {
        let compiler = compiler();
        let mut request = request(&["x.a"]);
        request.output_file = None;
        request.emit.push(EmitRequest {
            checkpoint: "ir.translated".into(),
            to: EmitTo::Memory,
        });
        request.emit.push(EmitRequest {
            checkpoint: "ir.optimized".into(),
            to: EmitTo::Memory,
        });
        let session = optimized(compiler, request);
        let texts: Vec<(CheckpointId, String)> = session
            .emitted()
            .iter()
            .map(|emitted| match emitted {
                Emitted::Memory { checkpoint, bytes } => {
                    (*checkpoint, String::from_utf8(bytes.clone()).unwrap())
                }
                other => panic!("unexpected output {other:?}"),
            })
            .collect();
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0].0, CheckpointId::IR_TRANSLATED);
        assert!(!texts[0].1.contains("from_alpha"), "{}", texts[0].1);
        assert_eq!(texts[1].0, CheckpointId::IR_OPTIMIZED);
        assert!(texts[1].1.contains("from_alpha"), "{}", texts[1].1);
    }

    #[test]
    fn nothing_is_written_to_a_stream_and_no_output_is_delivered_when_a_pass_fails() {
        let mut request = request(&["x.a"]);
        request.output_file = None;
        request.emit.push(EmitRequest {
            checkpoint: "ir.optimized".into(),
            to: EmitTo::Memory,
        });
        request.pipeline = Some("test-mark{name=one},test-fail".into());
        let compiler = compiler();
        let mut session = translated(compiler, request);
        assert!(session.optimize(&mut ()).is_err());
        assert!(session.emitted().is_empty());
    }

    #[test]
    fn no_step_runs_after_a_step_that_failed() {
        let mut request = request(&["x.a"]);
        request.pipeline = Some("test-fail".into());
        let compiler = compiler();
        let mut session = translated(compiler, request);
        assert!(session.optimize(&mut ()).is_err());
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert_eq!(message, "the lower step cannot run: an earlier step of the compilation failed");
    }

    // The default pipeline on real IR

    const SOURCE: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        %x = arith.constant 1 : i32;
        %y = arith.constant 2 : i32;
        %z = arith.add %x, %y;
        %dead = arith.add %a, %a;
        builtin.ret %z : (i32);
    };
};
"#;

    #[test]
    fn the_default_pipeline_over_a_module_leaves_valid_ir() {
        use runic_dialect_arith as _;

        let compiler = Compiler::builder().frontend(IrFrontend).build().unwrap();
        let mut request = CompilationRequest::new();
        request.inputs = vec![Input::memory("m.runic", SOURCE)];
        request.output_file = Some("-".parse().unwrap());
        request.ir.dump_pass_pipeline = true;
        let mut session = compiler.session(request).unwrap();
        let _ = session.translate(&mut ()).unwrap();
        let before = print(session.world());
        let _ = session.optimize(&mut ()).unwrap();
        let after = print(session.world());
        session.world().as_operation_ref().borrow().recursively_verify().unwrap();
        assert!(session.pass_output().contains(DEFAULT));
        assert_ne!(before, after, "the default pipeline changed nothing:\n{after}");
        assert!(!after.contains("%dead"), "{after}");
    }

    // The lowering step

    const OUT: &[CheckpointDecl] = &[decl("one.out", &["out"], "o")];
    const ONE: TestBackend = TestBackend::new("one", &["wasm32-wasip2"], OUT)
        .with_add_passes(|_, passes| add_mark(passes, "from_backend"));

    fn with_backend() -> &'static Compiler {
        static COMPILER: LazyLock<Compiler> = LazyLock::new(|| {
            Compiler::builder()
                .frontend(ALPHA_FRONTEND)
                .frontend(BETA_FRONTEND)
                .backend(ONE)
                .build()
                .unwrap()
        });
        &COMPILER
    }

    fn targeted(inputs: &[&str]) -> CompilationRequest {
        let mut request = request(inputs);
        request.target = Some("wasm32-wasip2".parse().unwrap());
        request
    }

    fn lowered(compiler: &Compiler, request: CompilationRequest) -> Session<'_> {
        let mut session = optimized(compiler, request);
        let _ = session.lower(&mut ()).unwrap();
        session
    }

    fn add_named(passes: &mut runic_ir::pass::OpPassManager, name: &str) -> Result<(), Report> {
        PassInfo::lookup(name)
            .expect("the test pass is registered")
            .add_to_pipeline(passes, &PassOptions::new())
    }

    fn backend_with(add_passes: crate::testing::BackendAddPassesFn) -> Compiler {
        Compiler::builder()
            .frontend(ALPHA_FRONTEND)
            .backend(ONE.with_add_passes(add_passes))
            .build()
            .unwrap()
    }

    fn error_text(report: Report) -> String {
        let mut text = report.to_string();
        for context in report.contexts() {
            text.push('\n');
            text.push_str(context.message());
        }
        text
    }

    #[test]
    fn the_backends_passes_run_after_the_optimization_pipeline() {
        let session = lowered(with_backend(), targeted(&["x.a", "y.b"]));
        assert_eq!(marks(&session), ["from_alpha", "from_beta", "from_backend"]);
    }

    #[test]
    fn the_requests_pipeline_replaces_the_optimization_pipeline_and_leaves_the_backends_passes() {
        let mut request = targeted(&["x.a"]);
        request.pipeline = Some("test-mark{name=p1}".into());
        let session = lowered(with_backend(), request);
        assert_eq!(marks(&session), ["p1", "from_backend"]);
    }

    #[test]
    fn the_lowering_pipeline_is_dumped_on_its_own_line_after_the_optimization_pipeline() {
        let mut request = targeted(&["x.a"]);
        request.opt_level = OptLevel::None;
        request.ir.dump_pass_pipeline = true;
        let session = lowered(with_backend(), request);
        let lines: Vec<_> = session.pass_output().lines().collect();
        assert_eq!(
            lines,
            [
                "builtin.world(test-mark{name=\"from_alpha\"})",
                "builtin.world(test-mark{name=\"from_backend\"})"
            ]
        );
    }

    #[test]
    fn the_dumped_lowering_pipeline_is_the_backends_even_when_the_request_gives_a_pipeline() {
        let mut request = targeted(&["x.a"]);
        request.pipeline = Some("test-mark{name=p1}".into());
        request.ir.dump_pass_pipeline = true;
        let session = lowered(with_backend(), request);
        let lines: Vec<_> = session.pass_output().lines().collect();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].contains("\"p1\""), "{lines:?}");
        assert!(lines[1].contains("\"from_backend\"") && !lines[1].contains("p1"), "{lines:?}");
    }

    #[test]
    fn the_ir_is_printed_around_the_backends_passes_too() {
        let mut request = targeted(&["x.a"]);
        request.ir.print_before = vec!["test-mark".into()];
        request.ir.print_after = vec!["test-mark".into()];
        let session = lowered(with_backend(), request);
        let output = session.pass_output();
        // One mark of the frontend, one of the backend.
        assert_eq!(output.matches("IR before").count(), 2, "{output}");
        assert_eq!(output.matches("IR after").count(), 2, "{output}");
        assert!(output.contains("@from_backend"), "{output}");
    }

    fn breaking_backend() -> Compiler {
        backend_with(|_, passes| {
            add_mark(passes, "before")?;
            add_named(passes, "test-break")?;
            add_mark(passes, "after")
        })
    }

    #[test]
    fn verifying_each_pass_stops_at_the_backends_pass_that_broke_the_ir() {
        let compiler = breaking_backend();
        let mut request = targeted(&["x.a"]);
        request.ir.verification = Verification::EachPass;
        let mut session = optimized(&compiler, request);
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert!(message.contains("does not end with a terminator"), "{message}");
        assert_eq!(marks(&session), ["from_alpha", "before", "broken"]);
        assert_eq!(session.reached().last(), Some(&CheckpointId::IR_OPTIMIZED));
    }

    #[test]
    fn verifying_once_runs_all_the_backends_passes_and_then_fails() {
        let compiler = breaking_backend();
        let mut request = targeted(&["x.a"]);
        request.ir.verification = Verification::Once;
        let mut session = optimized(&compiler, request);
        let text = error_text(session.lower(&mut ()).unwrap_err());
        assert!(text.contains("the lowering pipeline produced"), "{text}");
        assert_eq!(marks(&session), ["from_alpha", "before", "broken", "after"]);
        assert_eq!(session.reached().last(), Some(&CheckpointId::IR_OPTIMIZED));
    }

    #[test]
    fn with_verification_off_a_world_the_backend_broke_is_lowered() {
        let compiler = breaking_backend();
        let mut request = targeted(&["x.a"]);
        request.ir.verification = Verification::None;
        let session = lowered(&compiler, request);
        assert_eq!(session.reached().last(), Some(&CheckpointId::IR_LOWERED));
    }

    #[test]
    fn what_was_printed_stays_in_the_pass_output_when_a_backend_pass_fails() {
        let compiler = backend_with(|_, passes| {
            add_mark(passes, "one")?;
            add_named(passes, "test-fail")
        });
        let mut request = targeted(&["x.a"]);
        request.ir.print_before = vec!["all".into()];
        let mut session = optimized(&compiler, request);
        let before = session.pass_output().matches("IR before").count();
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert_eq!(message, "test-fail");
        assert_eq!(session.pass_output().matches("IR before").count() - before, 2);
    }

    #[test]
    fn a_backend_that_cannot_add_its_passes_is_named_in_the_error() {
        let compiler = backend_with(|_, _| Err(Report::msg("one has no passes")));
        let mut session = optimized(&compiler, targeted(&["x.a"]));
        let text = error_text(session.lower(&mut ()).unwrap_err());
        assert!(text.contains("one has no passes"), "{text}");
        assert!(text.contains("backend 'one'"), "{text}");
        assert_eq!(session.reached().last(), Some(&CheckpointId::IR_OPTIMIZED));
    }

    #[test]
    fn lowering_with_no_target_says_the_route_has_no_backend() {
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.optimized".into());
        let mut session = optimized(with_backend(), request);
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert!(message.contains("no backend") && message.contains("no target"), "{message}");
    }

    #[test]
    fn lowering_for_a_compiler_with_no_backend_says_the_route_has_no_backend() {
        let mut session = optimized(compiler(), request(&["x.a"]));
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert!(message.contains("no backend"), "{message}");
    }

    #[test]
    fn lowering_before_optimizing_is_an_error_and_uses_nothing_up() {
        let compiler = with_backend();
        let mut session = translated(compiler, targeted(&["x.a"]));
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert!(message.contains("before the optimize step reached 'ir.optimized'"), "{message}");
        assert!(marks(&session).is_empty());
        let _ = session.optimize(&mut ()).unwrap();
        let _ = session.lower(&mut ()).unwrap();
    }

    #[test]
    fn lowering_before_translating_is_an_error() {
        let compiler = with_backend();
        let mut session = compiler.session(targeted(&["x.a"])).unwrap();
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert!(message.contains("'ir.optimized'"), "{message}");
    }

    #[test]
    fn lowering_twice_is_an_error() {
        let mut session = lowered(with_backend(), targeted(&["x.a"]));
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert!(message.contains("run twice"), "{message}");
        assert_eq!(marks(&session), ["from_alpha", "from_backend"]);
    }

    #[test]
    fn lowering_after_a_stop_at_the_optimized_ir_is_an_error() {
        let mut request = targeted(&["x.a"]);
        request.stop_after = Some("ir.optimized".into());
        let mut session = optimized(with_backend(), request);
        let message = session.lower(&mut ()).unwrap_err().to_string();
        assert!(message.contains("the compilation stopped at 'ir.optimized'"), "{message}");
        assert_eq!(marks(&session), ["from_alpha"]);
    }

    #[test]
    fn the_step_reaches_the_lowered_ir_and_tells_the_observer() {
        let mut session = optimized(with_backend(), targeted(&["x.a"]));
        let mut observer = RecordingObserver::default();
        let flow = session.lower(&mut observer).unwrap();
        assert!(matches!(flow, ControlFlow::Continue(())));
        assert_eq!(session.reached().last(), Some(&CheckpointId::IR_LOWERED));
        assert_eq!(observer.checkpoints(), [CheckpointId::IR_LOWERED]);
    }

    #[test]
    fn a_stop_at_the_lowered_ir_is_honoured_and_holds_the_world() {
        let mut request = targeted(&["x.a"]);
        request.stop_after = Some("ir.lowered".into());
        let mut session = optimized(with_backend(), request);
        match session.lower(&mut ()).unwrap() {
            ControlFlow::Break(stopped) => {
                assert_eq!(stopped.checkpoint(), CheckpointId::IR_LOWERED)
            }
            ControlFlow::Continue(()) => panic!("the compilation should have stopped"),
        }
        let artifact = session.stopped().unwrap();
        assert_eq!(artifact.checkpoint(), CheckpointId::IR_LOWERED);
        assert!(artifact.downcast_ref::<WorldRef>().is_some());
    }

    #[test]
    fn an_output_at_the_lowered_ir_holds_the_ir_the_backends_passes_left() {
        let mut request = targeted(&["x.a"]);
        request.output_file = None;
        for checkpoint in ["ir.optimized", "ir.lowered"] {
            request.emit.push(EmitRequest {
                checkpoint: checkpoint.into(),
                to: EmitTo::Memory,
            });
        }
        let session = lowered(with_backend(), request);
        let texts: Vec<String> = session
            .emitted()
            .iter()
            .map(|emitted| String::from_utf8(emitted.bytes().unwrap().to_vec()).unwrap())
            .collect();
        assert!(!texts[0].contains("from_backend"), "{}", texts[0]);
        assert!(texts[1].contains("from_backend"), "{}", texts[1]);
    }

    #[test]
    fn no_output_is_delivered_when_a_backend_pass_fails() {
        let compiler = backend_with(|_, passes| add_named(passes, "test-fail"));
        let mut request = targeted(&["x.a"]);
        request.output_file = None;
        request.emit.push(EmitRequest {
            checkpoint: "ir.lowered".into(),
            to: EmitTo::Memory,
        });
        let mut session = optimized(&compiler, request);
        assert!(session.lower(&mut ()).is_err());
        assert!(session.emitted().is_empty());
    }
}
