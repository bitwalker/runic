//! A compilation in one call, and what it returns.

use core::{fmt, ops::ControlFlow};
use std::{path::PathBuf, rc::Rc};

use runic_core::{Context, Report, diagnostics::DiagnosticSet};

use crate::{Artifact, CheckpointId, CompilationRequest, Compiler, Emitted, Observer};

/// What a session did, as it hands it over when it ends.
pub(crate) struct Finished {
    pub(crate) reached: Vec<CheckpointId>,
    pub(crate) stopped: Option<Artifact>,
    pub(crate) emitted: Vec<Emitted>,
    pub(crate) pass_output: String,
    pub(crate) files_read: Vec<PathBuf>,
}

/// What a compilation produced, whether it succeeded or failed.
///
/// The caller that ran the compilation reads it: whether it succeeded and, if not, the error; the
/// checkpoints reached and the artifact at the stop point; the outputs delivered, which include
/// files written before a failure; the files read; the text the passes printed; and the
/// diagnostics that were reported without being the error. The error and the diagnostics point
/// into the source map of [Compilation::context].
pub struct Compilation {
    error: Option<Report>,
    reached: Vec<CheckpointId>,
    stopped: Option<Artifact>,
    emitted: Vec<Emitted>,
    files_read: Vec<PathBuf>,
    pass_output: String,
    diagnostics: DiagnosticSet,
    // Declared last, so that it is dropped last: after the artifact, which may hold IR.
    context: Rc<Context>,
}

impl Compilation {
    pub(crate) fn new(
        error: Option<Report>,
        finished: Finished,
        diagnostics: DiagnosticSet,
        context: Rc<Context>,
    ) -> Self {
        Self {
            error,
            reached: finished.reached,
            stopped: finished.stopped,
            emitted: finished.emitted,
            files_read: finished.files_read,
            pass_output: finished.pass_output,
            diagnostics,
            context,
        }
    }

    /// A compilation that failed before it started: nothing was reached, and the context is a
    /// new empty one.
    fn failed_to_start(error: Report) -> Self {
        Self::new(
            Some(error),
            Finished {
                reached: Vec::new(),
                stopped: None,
                emitted: Vec::new(),
                pass_output: String::new(),
                files_read: Vec::new(),
            },
            DiagnosticSet::default(),
            Rc::new(Context::default()),
        )
    }

    /// Why the compilation failed, or `None` when it succeeded.
    pub fn error(&self) -> Option<&Report> {
        self.error.as_ref()
    }

    /// Whether the compilation succeeded: it has no error.
    pub fn succeeded(&self) -> bool {
        self.error.is_none()
    }

    /// Every checkpoint the compilation reached, in the order it reached them.
    pub fn reached(&self) -> &[CheckpointId] {
        &self.reached
    }

    /// The artifact at the stop point, when the compilation reached it.
    pub fn artifact(&self) -> Option<&Artifact> {
        self.stopped.as_ref()
    }

    /// The outputs delivered, in the order their checkpoints were reached: the files that were
    /// written, and the bytes meant for standard output or for memory. A failed compilation lists
    /// what it delivered before it failed.
    pub fn emitted(&self) -> &[Emitted] {
        &self.emitted
    }

    /// The files the compilation read, in the order each was first read, once each.
    pub fn files_read(&self) -> &[PathBuf] {
        &self.files_read
    }

    /// What the pass pipelines printed: the pipeline dump and the IR printed between passes.
    pub fn pass_output(&self) -> &str {
        &self.pass_output
    }

    /// The diagnostics reported on the context during the compilation that are not its error:
    /// warnings and notes, and errors a stage reported without failing.
    pub fn diagnostics(&self) -> &DiagnosticSet {
        &self.diagnostics
    }

    /// The context of the compilation. The error and the diagnostics point into its source map.
    pub fn context(&self) -> &Rc<Context> {
        &self.context
    }
}

impl fmt::Debug for Compilation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Compilation")
            .field("error", &self.error)
            .field("reached", &self.reached)
            .field("artifact", &self.stopped.as_ref().map(Artifact::checkpoint))
            .field("emitted", &self.emitted.len())
            .field("files_read", &self.files_read.len())
            .field("pass_output_bytes", &self.pass_output.len())
            .field("diagnostics", &self.diagnostics.len())
            .finish_non_exhaustive()
    }
}

impl Compiler {
    /// Runs `request` to the end: resolves it, then translates, optimizes, and, when the request
    /// has a target, lowers and emits, stopping at the first step that stops the compilation or
    /// fails. `observer` is told of every checkpoint reached.
    ///
    /// This is the standard composition of the session's steps; a caller that needs a different
    /// one calls [Compiler::session] and the steps itself. It answers on every path: a request
    /// that cannot be resolved gives a failed [Compilation] with a new empty context. It installs
    /// nothing process-wide, prints nothing, never exits, and does not catch a panic in a stage.
    pub fn compile(&self, request: CompilationRequest, observer: &mut dyn Observer) -> Compilation {
        let mut session = match self.session(request) {
            Ok(session) => session,
            Err(error) => return Compilation::failed_to_start(error),
        };
        let mut outcome = session.translate(observer);
        if matches!(outcome, Ok(ControlFlow::Continue(()))) {
            outcome = session.optimize(observer);
        }
        if matches!(outcome, Ok(ControlFlow::Continue(()))) && session.target().is_some() {
            outcome = session.lower(observer);
        }
        if matches!(outcome, Ok(ControlFlow::Continue(()))) && session.target().is_some() {
            outcome = session.emit_code(observer);
        }
        if matches!(outcome, Ok(ControlFlow::Continue(()))) {
            outcome = Err(Report::msg(format!(
                "the compilation ended without reaching its stop point '{}'",
                session.goal().stop_after()
            )));
        }
        session.finish(outcome)
    }
}

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;
    use std::fs;

    use runic_core::{Report, diagnostics::diagnostic};

    use crate::{
        CheckpointDecl, CheckpointId, Compilation, CompilationRequest, Compiler, EmitRequest,
        EmitTo, Emitted, Input, IrFrontend, RecordingObserver,
        testing::{TestBackend, TestFrontend, add_fail, decl},
    };

    const OUT: &[CheckpointDecl] =
        &[decl("one.first", &["first"], "o1"), decl("one.second", &["second"], "o2")];
    const FIRST: CheckpointId = CheckpointId::new("one.first");
    const SECOND: CheckpointId = CheckpointId::new("one.second");
    const ONE: TestBackend = TestBackend::new("one", &["wasm32-wasip2"], OUT);

    const WHOLE_ROUTE: [CheckpointId; 5] = [
        CheckpointId::IR_TRANSLATED,
        CheckpointId::IR_OPTIMIZED,
        CheckpointId::IR_LOWERED,
        FIRST,
        SECOND,
    ];

    fn compiler(backend: TestBackend, alpha: TestFrontend) -> Compiler {
        Compiler::builder()
            .frontend(IrFrontend)
            .frontend(alpha)
            .backend(backend)
            .build()
            .unwrap()
    }

    fn plain() -> Compiler {
        compiler(ONE, TestFrontend::new("alpha", &["a"]))
    }

    fn ir() -> Input {
        Input::memory(
            "m.runic",
            "builtin.module public @m { builtin.function public extern(\"C\") @f(%x: i32) -> i32 \
             { builtin.ret %x : (i32); }; };",
        )
    }

    fn request() -> CompilationRequest {
        let mut request = CompilationRequest::new();
        request.inputs = vec![ir()];
        request.target = Some("wasm32-wasip2".parse().unwrap());
        request.output_file = Some("-".parse().unwrap());
        request
    }

    fn stop_after(checkpoint: &str) -> CompilationRequest {
        let mut request = request();
        request.stop_after = Some(checkpoint.into());
        request
    }

    fn run(compiler: &Compiler, request: CompilationRequest) -> (Compilation, RecordingObserver) {
        let mut observer = RecordingObserver::default();
        let compilation = compiler.compile(request, &mut observer);
        (compilation, observer)
    }

    fn message(compilation: &Compilation) -> String {
        compilation.error().expect("the compilation should have failed").to_string()
    }

    #[test]
    fn a_full_run_with_a_backend_succeeds_and_writes_the_default_output() {
        let dir = tempfile::tempdir().unwrap();
        let mut request = request();
        request.output_file = None;
        request.output_dir = Some(dir.path().to_path_buf());
        let (compilation, observer) = run(&plain(), request);

        assert!(compilation.succeeded(), "{:?}", compilation.error());
        assert!(compilation.error().is_none());
        assert_eq!(compilation.reached(), WHOLE_ROUTE);
        assert_eq!(compilation.reached(), observer.checkpoints());
        assert_eq!(compilation.artifact().unwrap().checkpoint(), SECOND);
        let path = dir.path().join("m.o2");
        assert_eq!(
            compilation.emitted(),
            [Emitted::File {
                checkpoint: SECOND,
                path: path.clone()
            }]
        );
        assert_eq!(fs::read_to_string(path).unwrap(), "one.second");
        assert!(compilation.diagnostics().is_empty());
    }

    #[test]
    fn a_full_run_can_deliver_its_output_in_memory() {
        let dir = tempfile::tempdir().unwrap();
        let mut request = request();
        request.output_file = None;
        request.output_dir = Some(dir.path().to_path_buf());
        request.emit.push(EmitRequest {
            checkpoint: "one.second".into(),
            to: EmitTo::Memory,
        });
        let (compilation, _) = run(&plain(), request);

        assert!(compilation.succeeded(), "{:?}", compilation.error());
        assert_eq!(
            compilation.emitted(),
            [Emitted::Memory {
                checkpoint: SECOND,
                bytes: b"one.second".to_vec()
            }]
        );
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_run_with_no_backend_ends_at_the_optimized_ir_and_succeeds() {
        let mut request = request();
        request.target = None;
        let compiler = Compiler::builder().frontend(IrFrontend).build().unwrap();
        let (compilation, observer) = run(&compiler, request);
        assert!(compilation.succeeded(), "{:?}", compilation.error());
        assert_eq!(
            compilation.reached(),
            [CheckpointId::IR_TRANSLATED, CheckpointId::IR_OPTIMIZED]
        );
        assert_eq!(compilation.reached(), observer.checkpoints());
        assert_eq!(compilation.artifact().unwrap().checkpoint(), CheckpointId::IR_OPTIMIZED);
        assert_eq!(compilation.emitted().len(), 1);
    }

    #[test]
    fn a_stop_point_ends_the_compilation_there_and_keeps_its_artifact() {
        for (name, stop, count) in [
            ("ir.translated", CheckpointId::IR_TRANSLATED, 1),
            ("ir.optimized", CheckpointId::IR_OPTIMIZED, 2),
            ("ir.lowered", CheckpointId::IR_LOWERED, 3),
            ("one.first", FIRST, 4),
        ] {
            let (compilation, observer) = run(&plain(), stop_after(name));
            assert!(compilation.succeeded(), "{name}: {:?}", compilation.error());
            assert_eq!(compilation.reached(), &WHOLE_ROUTE[..count], "{name}");
            assert_eq!(compilation.reached(), observer.checkpoints(), "{name}");
            assert_eq!(compilation.artifact().unwrap().checkpoint(), stop, "{name}");
            assert_eq!(compilation.emitted().len(), 1, "{name}");
            assert_eq!(compilation.emitted()[0].checkpoint(), stop, "{name}");
        }
    }

    #[test]
    fn a_request_that_does_not_resolve_fails_with_an_empty_usable_context() {
        let mut request = request();
        request.inputs = vec![Input::memory("x.zzz", "")];
        let (compilation, observer) = run(&plain(), request);

        assert!(message(&compilation).contains("'x.zzz'"));
        assert!(!compilation.succeeded());
        assert!(compilation.reached().is_empty());
        assert!(observer.checkpoints().is_empty());
        assert!(compilation.artifact().is_none());
        assert!(compilation.emitted().is_empty());
        assert!(compilation.files_read().is_empty());
        assert!(compilation.pass_output().is_empty());
        assert!(compilation.diagnostics().is_empty());
        let context = compilation.context();
        assert!(context.take_diagnostics().is_empty());
        assert!(context.add_source("later.runic", "text").is_ok());
    }

    #[test]
    fn a_frontend_error_fails_the_compilation_and_keeps_what_was_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.runic");
        fs::write(&path, "this is not IR").unwrap();
        let mut request = request();
        request.inputs = vec![Input::path(&path)];
        let (compilation, _) = run(&plain(), request);

        assert!(!compilation.succeeded());
        assert!(compilation.reached().is_empty());
        assert!(compilation.artifact().is_none());
        assert_eq!(compilation.files_read(), [path]);
    }

    #[test]
    fn a_failing_optimization_pass_fails_with_what_the_passes_printed() {
        let mut request = request();
        request.pipeline = Some("builtin.world(test-fail)".into());
        request.ir.dump_pass_pipeline = true;
        let (compilation, observer) = run(&plain(), request);

        assert!(message(&compilation).contains("test-fail"));
        assert_eq!(compilation.reached(), [CheckpointId::IR_TRANSLATED]);
        assert_eq!(compilation.reached(), observer.checkpoints());
        assert!(compilation.artifact().is_none());
        assert!(compilation.pass_output().contains("test-fail"), "{}", compilation.pass_output());
        assert!(compilation.emitted().is_empty());
    }

    #[test]
    fn a_failing_backend_pass_fails_after_the_optimized_ir() {
        let backend = ONE.with_add_passes(|_, passes| add_fail(passes));
        let (compilation, _) =
            run(&compiler(backend, TestFrontend::new("alpha", &["a"])), request());

        assert!(message(&compilation).contains("test-fail"));
        assert_eq!(
            compilation.reached(),
            [CheckpointId::IR_TRANSLATED, CheckpointId::IR_OPTIMIZED]
        );
        assert!(compilation.artifact().is_none());
    }

    #[test]
    fn a_failing_emission_fails_and_still_lists_the_output_delivered_before_it() {
        let backend = ONE.with_emit(|_, _, checkpoints| {
            let _ = checkpoints.reach(FIRST, "first".to_owned())?;
            Err(Report::msg("the backend gave up"))
        });
        let mut request = request();
        request.output_file = None;
        request.emit.push(EmitRequest {
            checkpoint: "one.first".into(),
            to: EmitTo::Memory,
        });
        request.stop_after = None;
        let (compilation, observer) =
            run(&compiler(backend, TestFrontend::new("alpha", &["a"])), request);

        assert_eq!(message(&compilation), "the backend gave up");
        assert_eq!(compilation.reached(), &WHOLE_ROUTE[..4]);
        assert_eq!(compilation.reached(), observer.checkpoints());
        assert!(compilation.artifact().is_none());
        assert_eq!(
            compilation.emitted(),
            [Emitted::Memory {
                checkpoint: FIRST,
                bytes: b"first".to_vec()
            }]
        );
    }

    fn warn(session: &crate::Session<'_>) {
        let _ = session.emit(diagnostic!(severity: Warning, message: "take care"));
    }

    fn report_error(session: &crate::Session<'_>) {
        let _ = session.emit(diagnostic!(severity: Error, message: "reported, not returned"));
    }

    fn alpha_reporting(report: crate::testing::TranslateFn) -> Compiler {
        compiler(ONE, TestFrontend::new("alpha", &["a"]).with_translate(report))
    }

    fn alpha_request() -> CompilationRequest {
        let mut request = stop_after("ir.optimized");
        request.inputs = vec![Input::memory("x.a", "")];
        request
    }

    #[test]
    fn a_warning_a_frontend_reported_is_in_the_result_and_no_longer_pending() {
        let compiler = alpha_reporting(|session, _, _, _| {
            warn(session);
            Ok(ControlFlow::Continue(()))
        });
        let (compilation, _) = run(&compiler, alpha_request());

        assert!(compilation.succeeded(), "{:?}", compilation.error());
        let messages: Vec<String> = compilation
            .diagnostics()
            .iter()
            .map(|entry| entry.diagnostic.to_string())
            .collect();
        assert_eq!(messages, ["take care"]);
        assert!(!compilation.diagnostics().has_errors());
        assert!(compilation.context().take_diagnostics().is_empty());
    }

    #[test]
    fn an_error_a_frontend_reported_and_did_not_return_fails_the_compilation() {
        let compiler = alpha_reporting(|session, _, _, _| {
            report_error(session);
            Ok(ControlFlow::Continue(()))
        });
        let (compilation, _) = run(&compiler, alpha_request());

        assert!(!compilation.succeeded());
        assert!(message(&compilation).contains("reported 1 error"), "{}", message(&compilation));
        assert!(compilation.diagnostics().has_errors());
        assert_eq!(compilation.diagnostics().len(), 1);
        assert!(compilation.context().take_diagnostics().is_empty());
    }

    #[test]
    fn a_compilation_debug_prints_counts_and_not_contents() {
        let (compilation, _) = run(&plain(), stop_after("ir.translated"));
        let text = format!("{compilation:?}");
        assert!(text.contains("Compilation") && text.contains("reached"), "{text}");
    }
}
