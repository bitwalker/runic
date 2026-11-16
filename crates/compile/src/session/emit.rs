//! The emission step: the backend produces its artifacts from the lowered world.

use core::ops::ControlFlow;

use runic_core::Report;

use super::{
    Session,
    checkpoints::{Checkpoints, Progress, Stage},
};
use crate::{CheckpointId, Flow, Observer};

impl Session<'_> {
    /// Runs the emission step: the backend of the target produces its artifacts from the world its
    /// passes lowered, reaching its checkpoints in order.
    ///
    /// The outputs requested at the backend's checkpoints are delivered as each is reached, like
    /// those of every checkpoint, and stay listed in [Session::emitted] when the backend fails
    /// later. `observer` is told of every checkpoint reached. The answer is `Break` when the
    /// compilation was asked to stop at a checkpoint of the backend, and then [Session::stopped]
    /// holds its artifact and the backend reached no later one.
    ///
    /// It is an error to run the step when the compilation has no target and so no backend,
    /// before the lower step reached `ir.lowered`, after the compilation stopped, or twice. It is
    /// an error for the backend to fail (its error is returned as it is), to return a `Break` for
    /// a stop it did not reach, to go on after the checkpoint the compilation stops at, or to
    /// finish without reaching it.
    pub fn emit_code(&mut self, observer: &mut dyn Observer) -> Result<Flow<()>, Report> {
        self.step("emit", |session, progress| session.run_emit_code(progress, observer))
    }

    fn run_emit_code(
        &self,
        progress: &mut Progress,
        observer: &mut dyn Observer,
    ) -> Result<Flow<()>, Report> {
        let Some(target) = &self.target else {
            return Err(Report::msg(
                "the emit step cannot run: the route of this compilation has no backend, because \
                 no target was given",
            ));
        };
        if progress.code_emitted {
            return Err(Report::msg("the emit step was run twice: a session emits once"));
        }
        let stop = self.goal.stop_after();
        if progress.stopped.is_some() {
            return Err(Report::msg(format!(
                "the emit step cannot run: the compilation stopped at '{stop}'"
            )));
        }
        if !progress.reached.contains(&CheckpointId::IR_LOWERED) {
            return Err(Report::msg(
                "the emit step cannot run before the lower step reached 'ir.lowered'",
            ));
        }
        progress.code_emitted = true;

        let backend = target.backend;
        let name = backend.name();
        let flow = {
            let mut checkpoints = Checkpoints::new(
                progress,
                observer,
                &self.route,
                &self.goal,
                self,
                Stage::Backend(backend),
            );
            backend.emit(self, self.world, &mut checkpoints)?
        };
        match flow {
            ControlFlow::Break(stopped) => {
                if progress.stopped.is_none() || stopped.checkpoint() != stop {
                    return Err(Report::msg(format!(
                        "backend '{name}' reported that the compilation stopped at '{}', but it \
                         never reached that checkpoint as the stop point",
                        stopped.checkpoint()
                    )));
                }
                Ok(ControlFlow::Break(stopped))
            }
            ControlFlow::Continue(()) => {
                if progress.stopped.is_some() {
                    Err(Report::msg(format!(
                        "backend '{name}' reached '{stop}', where the compilation stops, and went \
                         on instead of returning the stop"
                    )))
                } else {
                    Err(Report::msg(format!(
                        "backend '{name}' finished without reaching '{stop}', where the \
                         compilation stops"
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;
    use std::fs;

    use runic_core::{Report, dialects::builtin::WorldRef};

    use crate::{
        CheckpointDecl, CheckpointId, Checkpoints, CompilationRequest, Compiler, EmitRequest,
        EmitTo, Emitted, Flow, Input, RecordingObserver, Session, Stopped,
        testing::{TestBackend, TestFrontend, decl},
    };

    const ALPHA: &[CheckpointDecl] = &[decl("alpha.ast", &[], "ast")];
    const OUT: &[CheckpointDecl] =
        &[decl("one.first", &["first"], "o1"), decl("one.second", &["second"], "o2")];
    const FIRST: CheckpointId = CheckpointId::new("one.first");
    const SECOND: CheckpointId = CheckpointId::new("one.second");

    const ONE: TestBackend = TestBackend::new("one", &["wasm32-wasip2"], OUT);

    type Step = Result<Flow<()>, Report>;

    fn compiler(backend: TestBackend) -> Compiler {
        let alpha = TestFrontend::new("alpha", &["a"]).with_checkpoints(ALPHA);
        Compiler::builder().frontend(alpha).backend(backend).build().unwrap()
    }

    fn request() -> CompilationRequest {
        let mut request = CompilationRequest::new();
        request.inputs = vec![Input::memory("x.a", "")];
        request.target = Some("wasm32-wasip2".parse().unwrap());
        request.output_file = Some("-".parse().unwrap());
        request
    }

    fn to_memory(request: &mut CompilationRequest, checkpoint: &str) {
        request.emit.push(EmitRequest {
            checkpoint: checkpoint.into(),
            to: EmitTo::Memory,
        });
    }

    /// Runs the steps in order, up to the first that stops the compilation, and answers with the
    /// last answer.
    fn drive(session: &mut Session<'_>) -> Step {
        for step in [Session::translate, Session::optimize, Session::lower, Session::emit_code] {
            if let ControlFlow::Break(stopped) = step(session, &mut ())? {
                return Ok(ControlFlow::Break(stopped));
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    fn lowered(session: &mut Session<'_>) {
        for step in [Session::translate, Session::optimize, Session::lower] {
            let _ = step(session, &mut ()).unwrap();
        }
    }

    fn stopped_at(flow: Step) -> CheckpointId {
        match flow.unwrap() {
            ControlFlow::Break(stopped) => stopped.checkpoint(),
            ControlFlow::Continue(()) => panic!("the compilation should have stopped"),
        }
    }

    fn message(flow: Step) -> String {
        flow.unwrap_err().to_string()
    }

    // A compilation that runs to the end

    #[test]
    fn a_full_run_ends_with_a_stop_at_the_backends_last_checkpoint() {
        let compiler = compiler(ONE);
        let mut session = compiler.session(request()).unwrap();
        let mut observer = RecordingObserver::default();
        let _ = session.translate(&mut observer).unwrap();
        let _ = session.optimize(&mut observer).unwrap();
        let _ = session.lower(&mut observer).unwrap();
        assert_eq!(stopped_at(session.emit_code(&mut observer)), SECOND);
        assert_eq!(session.stopped().unwrap().checkpoint(), SECOND);
        assert_eq!(
            session.reached(),
            [
                CheckpointId::IR_TRANSLATED,
                CheckpointId::IR_OPTIMIZED,
                CheckpointId::IR_LOWERED,
                FIRST,
                SECOND
            ]
        );
        assert_eq!(observer.checkpoints(), session.reached());
    }

    #[test]
    fn the_default_output_of_a_full_run_is_written_to_the_output_directory() {
        let dir = tempfile::tempdir().unwrap();
        let mut request = request();
        request.output_file = None;
        request.output_dir = Some(dir.path().to_owned());
        let compiler = compiler(ONE);
        let mut session = compiler.session(request).unwrap();
        assert_eq!(stopped_at(drive(&mut session)), SECOND);

        let [Emitted::File { checkpoint, path }] = session.emitted() else {
            panic!("one file should have been written: {:?}", session.emitted());
        };
        assert_eq!(*checkpoint, SECOND);
        assert_eq!(path.parent(), Some(dir.path()));
        assert_eq!(fs::read(path).unwrap(), b"one.second");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn the_default_output_of_a_full_run_can_be_kept_in_memory() {
        let mut request = request();
        request.output_file = None;
        to_memory(&mut request, "one.second");
        let compiler = compiler(ONE);
        let mut session = compiler.session(request).unwrap();
        assert_eq!(stopped_at(drive(&mut session)), SECOND);
        assert_eq!(
            session.emitted(),
            [Emitted::Memory {
                checkpoint: SECOND,
                bytes: b"one.second".to_vec()
            }]
        );
    }

    #[test]
    fn the_backend_is_given_the_lowered_world() {
        fn check(
            session: &Session<'_>,
            world: WorldRef,
            checkpoints: &mut Checkpoints<'_>,
        ) -> Step {
            assert!(world == session.world());
            Ok(checkpoints.reach(FIRST, 0u8)?.map_break_only())
        }
        let compiler = compiler(ONE.with_emit(check));
        let mut request = request();
        request.stop_after = Some("one.first".into());
        let mut session = compiler.session(request).unwrap();
        assert_eq!(stopped_at(drive(&mut session)), FIRST);
    }

    // A stop before the end

    #[test]
    fn a_stop_at_a_checkpoint_that_is_not_the_last_reaches_no_later_one_and_writes_no_later_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut request = request();
        request.output_file = None;
        request.output_dir = Some(dir.path().to_owned());
        request.stop_after = Some("one.first".into());
        let compiler = compiler(ONE);
        let mut session = compiler.session(request).unwrap();
        assert_eq!(stopped_at(drive(&mut session)), FIRST);
        assert_eq!(session.reached().last(), Some(&FIRST));
        assert!(!session.reached().contains(&SECOND));
        let [Emitted::File { checkpoint, .. }] = session.emitted() else {
            panic!("one file should have been written: {:?}", session.emitted());
        };
        assert_eq!(*checkpoint, FIRST);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        assert_eq!(session.stopped().unwrap().checkpoint(), FIRST);
    }

    #[test]
    fn a_stop_at_the_lowered_ir_leaves_nothing_for_the_backend() {
        let mut request = request();
        request.stop_after = Some("ir.lowered".into());
        let compiler = compiler(ONE);
        let mut session = compiler.session(request).unwrap();
        assert_eq!(stopped_at(drive(&mut session)), CheckpointId::IR_LOWERED);
        let message = message(session.emit_code(&mut ()));
        assert!(message.contains("the compilation stopped at 'ir.lowered'"), "{message}");
        assert!(!session.reached().contains(&FIRST));
    }

    // Errors of the step

    #[test]
    fn emitting_with_no_target_is_an_error_that_says_the_route_has_no_backend() {
        let compiler = compiler(ONE);
        let mut request = request();
        request.target = None;
        request.stop_after = Some("ir.optimized".into());
        let mut session = compiler.session(request).unwrap();
        let message = message(session.emit_code(&mut ()));
        assert!(message.contains("no backend") && message.contains("no target"), "{message}");
    }

    #[test]
    fn emitting_before_lowering_is_an_error_and_uses_nothing_up() {
        let compiler = compiler(ONE);
        let mut session = compiler.session(request()).unwrap();
        let _ = session.translate(&mut ()).unwrap();
        let _ = session.optimize(&mut ()).unwrap();
        let message = message(session.emit_code(&mut ()));
        assert!(message.contains("before the lower step reached 'ir.lowered'"), "{message}");
        assert!(session.reached().iter().all(|id| id.as_str().starts_with("ir.")));
        // The step can still run in its place.
        let _ = session.lower(&mut ()).unwrap();
        assert_eq!(stopped_at(session.emit_code(&mut ())), SECOND);
    }

    #[test]
    fn emitting_before_translating_is_an_error() {
        let compiler = compiler(ONE);
        let mut session = compiler.session(request()).unwrap();
        let message = message(session.emit_code(&mut ()));
        assert!(message.contains("'ir.lowered'"), "{message}");
    }

    #[test]
    fn emitting_twice_is_an_error() {
        let compiler = compiler(ONE);
        let mut request = request();
        // Stopping after the step keeps the second call from reporting the stop instead.
        request.stop_after = Some("one.second".into());
        let mut session = compiler.session(request).unwrap();
        lowered(&mut session);
        let _ = session.emit_code(&mut ()).unwrap();
        let message = message(session.emit_code(&mut ()));
        assert!(message.contains("run twice"), "{message}");
    }

    // What the backend answers

    #[test]
    fn a_backend_that_fails_returns_its_error_as_it_is() {
        let compiler = compiler(ONE.with_emit(|_, _, _| Err(Report::msg("the backend gave up"))));
        let mut session = compiler.session(request()).unwrap();
        lowered(&mut session);
        assert_eq!(message(session.emit_code(&mut ())), "the backend gave up");
    }

    #[test]
    fn a_backend_that_fails_after_an_output_was_written_leaves_it_listed() {
        fn first_then_fail(
            _: &Session<'_>,
            _: WorldRef,
            checkpoints: &mut Checkpoints<'_>,
        ) -> Step {
            let _ = checkpoints.reach(FIRST, "partial".to_owned())?;
            Err(Report::msg("the disk caught fire"))
        }
        let dir = tempfile::tempdir().unwrap();
        let mut request = request();
        request.output_file = None;
        request.output_dir = Some(dir.path().to_owned());
        request.emit.push(EmitRequest {
            checkpoint: "one.first".into(),
            to: EmitTo::Default,
        });
        let compiler = compiler(ONE.with_emit(first_then_fail));
        let mut session = compiler.session(request).unwrap();
        let error = drive(&mut session).unwrap_err();
        assert_eq!(error.to_string(), "the disk caught fire");
        let [Emitted::File { checkpoint, path }] = session.emitted() else {
            panic!("the file written should be listed: {:?}", session.emitted());
        };
        assert_eq!(*checkpoint, FIRST);
        assert_eq!(fs::read(path).unwrap(), b"partial");
        assert!(session.stopped().is_none());
    }

    #[test]
    fn a_backend_reaching_a_checkpoint_of_a_frontend_is_an_error_naming_both() {
        fn reach_alpha(_: &Session<'_>, _: WorldRef, checkpoints: &mut Checkpoints<'_>) -> Step {
            Ok(checkpoints.reach(CheckpointId::new("alpha.ast"), 0u8)?.map_break_only())
        }
        let compiler = compiler(ONE.with_emit(reach_alpha));
        let mut session = compiler.session(request()).unwrap();
        lowered(&mut session);
        let message = message(session.emit_code(&mut ()));
        assert!(
            message.contains("backend 'one'") && message.contains("'alpha.ast'"),
            "{message}"
        );
        assert!(message.contains("'one.first', 'one.second'"), "{message}");
    }

    #[test]
    fn a_backend_reaching_a_checkpoint_of_the_harness_is_an_error() {
        fn reach_ir(_: &Session<'_>, _: WorldRef, checkpoints: &mut Checkpoints<'_>) -> Step {
            Ok(checkpoints.reach(CheckpointId::IR_LOWERED, 0u8)?.map_break_only())
        }
        let compiler = compiler(ONE.with_emit(reach_ir));
        let mut session = compiler.session(request()).unwrap();
        lowered(&mut session);
        let message = message(session.emit_code(&mut ()));
        assert!(
            message.contains("backend 'one'") && message.contains("'ir.lowered'"),
            "{message}"
        );
    }

    #[test]
    fn a_backend_that_skips_its_last_checkpoint_is_an_error_naming_it() {
        fn only_first(_: &Session<'_>, _: WorldRef, checkpoints: &mut Checkpoints<'_>) -> Step {
            Ok(checkpoints.reach(FIRST, 0u8)?.map_break_only())
        }
        let compiler = compiler(ONE.with_emit(only_first));
        let mut session = compiler.session(request()).unwrap();
        lowered(&mut session);
        let message = message(session.emit_code(&mut ()));
        assert!(
            message.contains("backend 'one'") && message.contains("without reaching 'one.second'"),
            "{message}"
        );
        assert!(session.stopped().is_none());
    }

    #[test]
    fn a_backend_that_returns_a_stop_it_did_not_reach_is_an_error() {
        fn forged(_: &Session<'_>, _: WorldRef, _: &mut Checkpoints<'_>) -> Step {
            Ok(ControlFlow::Break(Stopped::new(SECOND)))
        }
        let compiler = compiler(ONE.with_emit(forged));
        let mut session = compiler.session(request()).unwrap();
        lowered(&mut session);
        let message = message(session.emit_code(&mut ()));
        assert!(
            message.contains("backend 'one'") && message.contains("never reached that checkpoint"),
            "{message}"
        );
        assert!(session.stopped().is_none());
    }

    #[test]
    fn a_backend_that_returns_a_stop_other_than_the_one_it_reached_is_an_error() {
        fn wrong_stop(_: &Session<'_>, _: WorldRef, c: &mut Checkpoints<'_>) -> Step {
            let _ = c.reach(FIRST, 0u8)?;
            let _ = c.reach(SECOND, 0u8)?;
            Ok(ControlFlow::Break(Stopped::new(FIRST)))
        }
        let compiler = compiler(ONE.with_emit(wrong_stop));
        let mut session = compiler.session(request()).unwrap();
        lowered(&mut session);
        let message = message(session.emit_code(&mut ()));
        assert!(message.contains("'one.first'"), "{message}");
        assert!(message.contains("never reached that checkpoint"), "{message}");
    }

    #[test]
    fn a_backend_that_goes_on_after_the_stop_is_an_error() {
        fn goes_on(_: &Session<'_>, _: WorldRef, checkpoints: &mut Checkpoints<'_>) -> Step {
            let _ = checkpoints.reach(FIRST, 0u8)?;
            Ok(ControlFlow::Continue(()))
        }
        let mut request = request();
        request.stop_after = Some("one.first".into());
        let compiler = compiler(ONE.with_emit(goes_on));
        let mut session = compiler.session(request).unwrap();
        lowered(&mut session);
        let message = message(session.emit_code(&mut ()));
        assert!(
            message.contains("backend 'one'")
                && message.contains("'one.first'")
                && message.contains("went on instead of returning the stop"),
            "{message}"
        );
    }

    #[test]
    fn a_backend_reaching_a_checkpoint_after_the_stop_is_an_error() {
        fn too_far(_: &Session<'_>, _: WorldRef, checkpoints: &mut Checkpoints<'_>) -> Step {
            let _ = checkpoints.reach(FIRST, 0u8)?;
            Ok(checkpoints.reach(SECOND, 0u8)?.map_break_only())
        }
        let mut request = request();
        request.stop_after = Some("one.first".into());
        let compiler = compiler(ONE.with_emit(too_far));
        let mut session = compiler.session(request).unwrap();
        lowered(&mut session);
        let message = message(session.emit_code(&mut ()));
        assert!(message.contains("after the compilation had stopped"), "{message}");
    }

    trait MapBreakOnly {
        fn map_break_only(self) -> Flow<()>;
    }

    impl<T> MapBreakOnly for Flow<T> {
        fn map_break_only(self) -> Flow<()> {
            match self {
                ControlFlow::Continue(_) => ControlFlow::Continue(()),
                ControlFlow::Break(stopped) => ControlFlow::Break(stopped),
            }
        }
    }
}
