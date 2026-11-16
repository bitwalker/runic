//! The hand a stage uses to reach checkpoints.

use core::any::Any;

use runic_ir::Report;

use super::Session;
use crate::{
    Artifact, Backend, CheckpointDecl, CheckpointId, Destination, Emitted, Flow, Frontend, Goal,
    IR_CHECKPOINTS, Observer, Route, Stopped, write_file,
};

/// What a session has done so far: the checkpoints reached, the artifact kept at the stop point,
/// the outputs delivered, and whether each step has run, and what the pass pipeline printed.
#[derive(Default)]
pub(super) struct Progress {
    pub(super) reached: Vec<CheckpointId>,
    pub(super) stopped: Option<Artifact>,
    pub(super) emitted: Vec<Emitted>,
    pub(super) translated: bool,
    pub(super) optimized: bool,
    pub(super) lowered: bool,
    pub(super) code_emitted: bool,
    pub(super) pass_output: String,
    /// A step returned an error: no later step runs.
    pub(super) failed: bool,
}

/// The stage that is running, and so which checkpoints it may reach.
#[derive(Clone, Copy)]
pub(super) enum Stage<'a> {
    Frontend(&'a dyn Frontend),
    Backend(&'a dyn Backend),
    Harness,
}

impl Stage<'_> {
    fn declared(&self) -> &'static [CheckpointDecl] {
        match self {
            Self::Frontend(frontend) => frontend.checkpoints(),
            Self::Backend(backend) => backend.checkpoints(),
            Self::Harness => &IR_CHECKPOINTS,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Frontend(frontend) => format!("frontend '{}'", frontend.name()),
            Self::Backend(backend) => format!("backend '{}'", backend.name()),
            Self::Harness => "the harness".to_string(),
        }
    }
}

/// How the stage that is running reaches the checkpoints it declared.
///
/// A frontend is given one by the translate step, and a backend by the emission step; each calls [Checkpoints::reach] each time it holds
/// an artifact it declared a checkpoint for. The harness checks that the checkpoint is the stage's
/// own and comes after the last one reached, tells the observer, delivers the output requested at
/// the checkpoint, if any, and, at the checkpoint the compilation is to stop after, keeps the
/// artifact and says so.
pub struct Checkpoints<'a> {
    progress: &'a mut Progress,
    observer: &'a mut dyn Observer,
    route: &'a Route,
    goal: &'a Goal,
    session: &'a Session<'a>,
    stage: Stage<'a>,
}

impl<'a> Checkpoints<'a> {
    pub(super) fn new(
        progress: &'a mut Progress,
        observer: &'a mut dyn Observer,
        route: &'a Route,
        goal: &'a Goal,
        session: &'a Session<'a>,
        stage: Stage<'a>,
    ) -> Self {
        Self {
            progress,
            observer,
            route,
            goal,
            session,
            stage,
        }
    }

    /// Reaches `checkpoint` holding `value`.
    ///
    /// It is an error to reach a checkpoint after the stop point was reached, when an error has
    /// been reported on the context (so no output is delivered after one), one the running
    /// stage did not declare, or one that does not come later on the route than the last one
    /// reached (so a checkpoint cannot be reached twice). Otherwise the observer is told and, when
    /// an output was requested at `checkpoint`, the artifact is rendered as it is now with the
    /// checkpoint's renderer and delivered to the output's destination: a file is written, and
    /// the bytes for standard output or memory are kept (see [Session::emitted]). A renderer that
    /// fails, or a file that cannot be written, is an error that names the checkpoint. The answer
    /// is `Continue(value)` to go on, or, when `checkpoint` is where the compilation stops,
    /// `Break` after the artifact has been kept: the stage returns that `Break` at once.
    pub fn reach<T: Any>(&mut self, checkpoint: CheckpointId, value: T) -> Result<Flow<T>, Report> {
        let stage = self.stage.describe();
        let stop = self.goal.stop_after();
        if self.progress.stopped.is_some() {
            return Err(Report::msg(format!(
                "{stage} reached checkpoint '{checkpoint}' after the compilation had stopped at \
                 '{stop}'"
            )));
        }
        // A stage may report an error on the context and carry on: nothing is reached, and so
        // nothing is delivered, once the compilation has an error.
        let errors = self.session.context().diagnostic_counts().errors();
        if errors > 0 {
            return Err(Report::msg(format!(
                "the compilation reported {errors} error(s) before checkpoint '{checkpoint}' and \
                 goes no further"
            )));
        }
        let declared = self.stage.declared();
        if !declared.iter().any(|decl| decl.id == checkpoint) {
            let ids: Vec<String> = declared.iter().map(|decl| format!("'{}'", decl.id)).collect();
            let known = if ids.is_empty() {
                "it declares none".to_string()
            } else {
                format!("it declares {}", ids.join(", "))
            };
            return Err(Report::msg(format!(
                "{stage} reached checkpoint '{checkpoint}', which it did not declare ({known})"
            )));
        }
        let Some(position) = self.route.position(checkpoint) else {
            return Err(Report::msg(format!(
                "{stage} reached checkpoint '{checkpoint}', which is not on the route of this \
                 compilation"
            )));
        };
        if let Some(last) = self.progress.reached.last() {
            let after = self.route.position(*last).is_some_and(|last| position > last);
            if !after {
                return Err(Report::msg(format!(
                    "{stage} reached checkpoint '{checkpoint}' after '{last}': checkpoints are \
                     reached in the order of the route, each once"
                )));
            }
        }

        let artifact = Artifact::new(checkpoint, value);
        self.observer.on_checkpoint(checkpoint, &artifact);
        self.progress.reached.push(checkpoint);
        if let Some(output) = self.goal.output_at(checkpoint) {
            let mut bytes = Vec::new();
            (self.route.get(position).render)(&artifact, self.session, &mut bytes).map_err(
                |report| report.context(format!("could not render checkpoint '{checkpoint}'")),
            )?;
            let emitted = match output.destination() {
                Destination::File(path) => {
                    // The output directory is created when a file goes directly into it. The
                    // directory of any other file is not made.
                    if let Some(dir) = self.session.request().output_dir.as_deref()
                        && path.parent() == Some(dir)
                    {
                        if dir.exists() && !dir.is_dir() {
                            return Err(Report::msg(format!(
                                "the output directory '{}' is not a directory",
                                dir.display()
                            )));
                        }
                        std::fs::create_dir_all(dir).map_err(|error| {
                            Report::msg(format!(
                                "could not create the output directory '{}': {error}",
                                dir.display()
                            ))
                        })?;
                    }
                    write_file(path, &bytes).map_err(|error| {
                        Report::msg(format!(
                            "could not write the output of checkpoint '{checkpoint}' to '{}': \
                             {error}",
                            path.display()
                        ))
                    })?;
                    Emitted::File {
                        checkpoint,
                        path: path.clone(),
                    }
                }
                Destination::Stdout => Emitted::Stdout { checkpoint, bytes },
                Destination::Memory => Emitted::Memory { checkpoint, bytes },
            };
            self.progress.emitted.push(emitted);
        }
        if checkpoint == stop {
            self.progress.stopped = Some(artifact);
            return Ok(Flow::Break(Stopped::new(checkpoint)));
        }
        match artifact.downcast::<T>() {
            Ok(value) => Ok(Flow::Continue(value)),
            Err(_) => unreachable!("the artifact was made from a value of this type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;
    use std::{fs, path::PathBuf};

    use runic_ir::{
        Ident, Op, Report, Visibility,
        dialects::builtin::{ModuleBuilder, WorldBuilder, attributes::Signature},
    };

    use crate::{
        CheckpointDecl, CheckpointId, Checkpoints, CompilationRequest, Compiler, EmitRequest,
        EmitTo, Emitted, Flow, Input, IrFrontend, Session,
        testing::{TestBackend, TestFrontend, decl, render_text},
    };

    const MY_TEXT: CheckpointId = CheckpointId::new("alpha.text");
    const MY_LATE: CheckpointId = CheckpointId::new("alpha.late");
    const MY_BAD: CheckpointId = CheckpointId::new("alpha.bad");
    const ALPHA: &[CheckpointDecl] = &[
        decl("alpha.text", &["text"], "txt"),
        decl("alpha.late", &["late"], "late"),
        CheckpointDecl {
            render: |_, _, _| Err(Report::msg("the renderer gave up")),
            ..decl("alpha.bad", &["bad"], "bad")
        },
    ];
    const OUT: &[CheckpointDecl] = &[decl("one.out", &[], "o")];
    const ONE: TestBackend = TestBackend::new("one", &["wasm32-wasip2"], OUT);

    fn reach_text(
        c: &mut Checkpoints<'_>,
        id: CheckpointId,
        text: &str,
    ) -> Result<Flow<()>, Report> {
        Ok(match c.reach(id, text.to_string())? {
            ControlFlow::Continue(_) => ControlFlow::Continue(()),
            ControlFlow::Break(stopped) => ControlFlow::Break(stopped),
        })
    }

    /// Reaches `alpha.text` with a string, then `alpha.late` (unless stopped).
    fn translate_two(
        _: &Session<'_>,
        _: &[Input],
        _: runic_ir::dialects::builtin::WorldRef,
        c: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        if let ControlFlow::Break(stopped) = reach_text(c, MY_TEXT, "first")? {
            return Ok(ControlFlow::Break(stopped));
        }
        reach_text(c, MY_LATE, "second")
    }

    fn alpha() -> TestFrontend {
        TestFrontend::new("alpha", &["a"])
            .with_checkpoints(ALPHA)
            .with_translate(translate_two)
    }

    fn compiler() -> Compiler {
        Compiler::builder()
            .frontend(alpha())
            .frontend(IrFrontend)
            .backend(ONE)
            .build()
            .unwrap()
    }

    fn request(
        inputs: Vec<Input>,
        stop_after: &str,
        emit: &[(&str, EmitTo)],
    ) -> CompilationRequest {
        let mut request = CompilationRequest::new();
        request.inputs = inputs;
        request.stop_after = Some(stop_after.into());
        for (checkpoint, to) in emit {
            request.emit.push(EmitRequest {
                checkpoint: (*checkpoint).into(),
                to: to.clone(),
            });
        }
        request
    }

    fn a() -> Input {
        Input::memory("x.a", "")
    }

    fn ir() -> Input {
        Input::memory(
            "m.runic",
            "builtin.module public @m { builtin.function public extern(\"C\") @f(%x: i32) -> i32 \
             { builtin.ret %x : (i32); }; };",
        )
    }

    fn translate(session: &mut Session<'_>) -> Result<Flow<()>, Report> {
        session.translate(&mut ())
    }

    #[test]
    fn an_output_to_memory_is_kept_as_the_rendered_bytes() {
        let compiler = compiler();
        let request = request(vec![a()], "alpha.late", &[("alpha.text", EmitTo::Memory)]);
        let mut session = compiler.session(request).unwrap();
        let _ = translate(&mut session).unwrap();
        assert_eq!(
            session.emitted(),
            [Emitted::Memory {
                checkpoint: MY_TEXT,
                bytes: b"first".to_vec()
            }]
        );
    }

    #[test]
    fn an_output_to_standard_output_is_kept_and_not_printed() {
        let compiler = compiler();
        let request = request(vec![a()], "alpha.late", &[("alpha.late", EmitTo::Stdout)]);
        let mut session = compiler.session(request).unwrap();
        let _ = translate(&mut session).unwrap();
        assert_eq!(
            session.emitted(),
            [Emitted::Stdout {
                checkpoint: MY_LATE,
                bytes: b"second".to_vec()
            }]
        );
    }

    #[test]
    fn an_output_to_a_file_is_written_and_listed_by_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("late.txt");
        let compiler = compiler();
        let request =
            request(vec![a()], "alpha.late", &[("alpha.late", EmitTo::File(path.clone()))]);
        let mut session = compiler.session(request).unwrap();
        let _ = translate(&mut session).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        assert_eq!(
            session.emitted(),
            [Emitted::File {
                checkpoint: MY_LATE,
                path
            }]
        );
    }

    #[test]
    fn outputs_are_listed_in_the_order_their_checkpoints_were_reached() {
        let compiler = compiler();
        let request = request(
            vec![a()],
            "alpha.late",
            &[("alpha.late", EmitTo::Memory), ("alpha.text", EmitTo::Stdout)],
        );
        let mut session = compiler.session(request).unwrap();
        let _ = translate(&mut session).unwrap();
        let order: Vec<_> = session.emitted().iter().map(Emitted::checkpoint).collect();
        assert_eq!(order, [MY_TEXT, MY_LATE]);
    }

    #[test]
    fn an_output_is_rendered_as_the_artifact_is_when_its_checkpoint_is_reached() {
        // The frontend goes on to reach another checkpoint after the first one is emitted.
        let compiler = compiler();
        let request = request(vec![a()], "alpha.late", &[("alpha.text", EmitTo::Memory)]);
        let mut session = compiler.session(request).unwrap();
        let _ = translate(&mut session).unwrap();
        assert_eq!(session.emitted()[0].bytes(), Some(&b"first"[..]));
    }

    #[test]
    fn the_ir_is_rendered_as_it_is_when_translated_is_reached_not_as_it_is_later() {
        let compiler = compiler();
        let request = request(vec![ir()], "ir.translated", &[("ir.translated", EmitTo::Memory)]);
        let mut session = compiler.session(request).unwrap();
        let _ = translate(&mut session).unwrap();
        let before = String::from_utf8(session.emitted()[0].bytes().unwrap().to_vec()).unwrap();
        assert!(before.contains("builtin.module public @m"), "{before}");
        assert!(before.ends_with('\n'), "{before:?}");

        let module = WorldBuilder::new(session.world())
            .declare_module(Ident::with_empty_span("added_later".into()))
            .unwrap();
        let mut builder = ModuleBuilder::new(module);
        let signature = Signature::new(session.context(), [], []);
        builder
            .define_function(Ident::with_empty_span("g".into()), Visibility::Public, signature)
            .unwrap();
        let after = String::from_utf8(session.emitted()[0].bytes().unwrap().to_vec()).unwrap();
        assert_eq!(before, after);
        assert!(!after.contains("added_later"), "{after}");
        let now = session.world().borrow().as_operation().to_string();
        assert!(now.contains("added_later"), "the world did change: {now}");
    }

    #[test]
    fn the_ir_goes_to_a_file_and_to_standard_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.runic");
        let compiler = compiler();
        let mut file = compiler
            .session(request(
                vec![ir()],
                "ir.translated",
                &[("ir.translated", EmitTo::File(path.clone()))],
            ))
            .unwrap();
        let _ = translate(&mut file).unwrap();
        let mut stdout = compiler
            .session(request(vec![ir()], "ir.translated", &[("ir.translated", EmitTo::Stdout)]))
            .unwrap();
        let _ = translate(&mut stdout).unwrap();
        let written = fs::read(&path).unwrap();
        assert_eq!(Some(&written[..]), stdout.emitted()[0].bytes());
        assert!(matches!(stdout.emitted()[0], Emitted::Stdout { .. }));
    }

    #[test]
    fn print_generic_changes_the_emitted_ir() {
        let compiler = compiler();
        let text = |generic: bool| {
            let mut request =
                request(vec![ir()], "ir.translated", &[("ir.translated", EmitTo::Memory)]);
            request.ir.print_generic = generic;
            let mut session = compiler.session(request).unwrap();
            let _ = translate(&mut session).unwrap();
            String::from_utf8(session.emitted()[0].bytes().unwrap().to_vec()).unwrap()
        };
        let (custom, generic) = (text(false), text(true));
        assert_ne!(custom, generic);
        assert!(generic.contains("\"builtin.module\""), "{generic}");
    }

    #[test]
    fn a_renderer_that_fails_gives_an_error_naming_the_checkpoint_and_the_cause() {
        fn reach_bad(
            _: &Session<'_>,
            _: &[Input],
            _: runic_ir::dialects::builtin::WorldRef,
            c: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            reach_text(c, MY_BAD, "x")
        }
        let compiler = Compiler::builder()
            .frontend(alpha().with_translate(reach_bad))
            .backend(ONE)
            .build()
            .unwrap();
        let mut session = compiler
            .session(request(vec![a()], "alpha.bad", &[("alpha.bad", EmitTo::Memory)]))
            .unwrap();
        let message = format!("{:?}", translate(&mut session).unwrap_err());
        assert!(
            message.contains("'alpha.bad'") && message.contains("the renderer gave up"),
            "{message}"
        );
        assert!(session.emitted().is_empty());
    }

    #[test]
    fn a_file_that_cannot_be_written_gives_an_error_naming_the_path_and_the_cause() {
        let dir = tempfile::tempdir().unwrap();
        let path: PathBuf = dir.path().join("missing").join("late.txt");
        let compiler = compiler();
        let mut session = compiler
            .session(request(
                vec![a()],
                "alpha.late",
                &[("alpha.late", EmitTo::File(path.clone()))],
            ))
            .unwrap();
        let message = translate(&mut session).unwrap_err().to_string();
        assert!(message.contains(&path.display().to_string()), "{message}");
        assert!(message.contains("alpha.late"), "{message}");
        assert!(session.emitted().is_empty());
    }

    #[test]
    fn nothing_is_written_for_a_checkpoint_that_is_never_reached() {
        fn fail(
            _: &Session<'_>,
            _: &[Input],
            _: runic_ir::dialects::builtin::WorldRef,
            _: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            Err(Report::msg("frontend failed"))
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("late.txt");
        let compiler = Compiler::builder()
            .frontend(alpha().with_translate(fail))
            .backend(ONE)
            .build()
            .unwrap();
        let mut session = compiler
            .session(request(
                vec![a()],
                "alpha.late",
                &[("alpha.late", EmitTo::File(path.clone()))],
            ))
            .unwrap();
        assert!(translate(&mut session).is_err());
        assert!(!path.exists());
        assert!(session.emitted().is_empty());
    }

    #[test]
    fn an_output_at_the_stop_point_is_delivered_before_the_stop() {
        let compiler = compiler();
        let mut session = compiler
            .session(request(vec![a()], "alpha.text", &[("alpha.text", EmitTo::Memory)]))
            .unwrap();
        let flow = translate(&mut session).unwrap();
        assert!(flow.is_break());
        assert_eq!(session.emitted()[0].bytes(), Some(&b"first"[..]));
        assert_eq!(session.stopped().unwrap().checkpoint(), MY_TEXT);
    }

    #[test]
    fn a_checkpoint_with_no_output_is_not_rendered() {
        // The renderer of `alpha.bad` always fails, so rendering it would be an error.
        fn reach_all(
            _: &Session<'_>,
            _: &[Input],
            _: runic_ir::dialects::builtin::WorldRef,
            c: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            let _ = reach_text(c, MY_TEXT, "t")?;
            let _ = reach_text(c, MY_LATE, "l")?;
            reach_text(c, MY_BAD, "b")
        }
        let compiler = Compiler::builder()
            .frontend(alpha().with_translate(reach_all))
            .backend(ONE)
            .build()
            .unwrap();
        let mut session = compiler
            .session(request(vec![a()], "alpha.bad", &[("alpha.late", EmitTo::Memory)]))
            .unwrap();
        let _ = translate(&mut session).unwrap();
        assert_eq!(session.emitted().len(), 1);
    }

    #[test]
    fn the_test_renderer_writes_the_checkpoints_id_for_a_value_that_is_not_text() {
        fn reach_number(
            _: &Session<'_>,
            _: &[Input],
            _: runic_ir::dialects::builtin::WorldRef,
            c: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            Ok(match c.reach(MY_TEXT, 7u32)? {
                ControlFlow::Continue(_) => ControlFlow::Continue(()),
                ControlFlow::Break(stopped) => ControlFlow::Break(stopped),
            })
        }
        let _: crate::checkpoint::Render = render_text;
        let compiler = Compiler::builder()
            .frontend(alpha().with_translate(reach_number))
            .backend(ONE)
            .build()
            .unwrap();
        let mut session = compiler
            .session(request(vec![a()], "alpha.text", &[("alpha.text", EmitTo::Memory)]))
            .unwrap();
        let _ = translate(&mut session).unwrap();
        assert_eq!(session.emitted()[0].bytes(), Some(&b"alpha.text"[..]));
    }
}
