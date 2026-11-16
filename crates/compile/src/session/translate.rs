//! The translate step: the frontends run, and the world is verified.

use core::ops::ControlFlow;

use runic_core::{Report, link_symbol_refs};

use super::{
    Session,
    checkpoints::{Checkpoints, Progress, Stage},
};
use crate::{CheckpointId, Flow, Observer, Verification};

impl Session<'_> {
    /// Runs the translate step: each frontend that has inputs translates them in turn into the
    /// world, the symbol references left unlinked are linked to the symbols their paths name in
    /// the whole world, the world is verified (unless the request's verification is
    /// [Verification::None]), and the harness reaches `ir.translated` with the world.
    ///
    /// `observer` is told of every checkpoint reached, in order; a caller that has nobody to tell
    /// passes `&mut ()`. The answer is `Break` when the compilation was asked to stop at a
    /// checkpoint reached during the step, and then [Session::stopped] holds its artifact and no
    /// later frontend was called. It is an error to run the step twice, for a frontend to return
    /// an error or an answer that does not match what it reached, to leave the compilation's stop
    /// point among its own checkpoints unreached, or to leave IR that does not verify. The
    /// session keeps what was reached whatever the outcome.
    pub fn translate(&mut self, observer: &mut dyn Observer) -> Result<Flow<()>, Report> {
        self.step("translate", |session, progress| session.run_translate(progress, observer))
    }

    fn run_translate(
        &self,
        progress: &mut Progress,
        observer: &mut dyn Observer,
    ) -> Result<Flow<()>, Report> {
        if progress.translated {
            return Err(Report::msg("the translate step was run twice: a session translates once"));
        }
        progress.translated = true;

        let stop = self.goal.stop_after();
        for (frontend, inputs) in &self.frontends {
            let name = frontend.name();
            let flow = {
                let mut checkpoints = Checkpoints::new(
                    progress,
                    observer,
                    &self.route,
                    &self.goal,
                    self,
                    Stage::Frontend(*frontend),
                );
                frontend.translate(self, inputs, self.world, &mut checkpoints)?
            };
            match flow {
                ControlFlow::Break(stopped) => {
                    if progress.stopped.is_none() || stopped.checkpoint() != stop {
                        return Err(Report::msg(format!(
                            "frontend '{name}' reported that the compilation stopped at '{}', but \
                             it never reached that checkpoint as the stop point",
                            stopped.checkpoint()
                        )));
                    }
                    return Ok(ControlFlow::Break(stopped));
                }
                ControlFlow::Continue(()) => {
                    if progress.stopped.is_some() {
                        return Err(Report::msg(format!(
                            "frontend '{name}' reached '{stop}', where the compilation stops, and \
                             went on instead of returning the stop"
                        )));
                    }
                    let declares_stop = frontend.checkpoints().iter().any(|decl| decl.id == stop);
                    if declares_stop && !progress.reached.contains(&stop) {
                        return Err(Report::msg(format!(
                            "frontend '{name}' declares checkpoint '{stop}' and never reached it, \
                             but the compilation was asked to stop there"
                        )));
                    }
                }
            }
        }

        // Each frontend links the references it can see; a reference to a symbol another
        // frontend, or another input, defined is linked now that the world is whole.
        link_symbol_refs(self.world.as_operation_ref());

        if self.request.ir.verification != Verification::None {
            self.world.as_operation_ref().borrow().recursively_verify().map_err(|report| {
                report.context("the IR the frontends produced does not verify")
            })?;
        }

        let mut checkpoints =
            Checkpoints::new(progress, observer, &self.route, &self.goal, self, Stage::Harness);
        match checkpoints.reach(CheckpointId::IR_TRANSLATED, self.world)? {
            ControlFlow::Break(stopped) => Ok(ControlFlow::Break(stopped)),
            ControlFlow::Continue(_) => Ok(ControlFlow::Continue(())),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;

    use runic_core::{
        DataLayout, Ident, Report, Visibility,
        dialects::builtin::{
            FunctionBuilder, ModuleBuilder, WorldBuilder, WorldRef, attributes::Signature,
        },
    };

    use crate::{
        CheckpointDecl, CheckpointId, Checkpoints, CompilationRequest, Compiler, Flow, Input,
        RecordingObserver, Session, Stopped, Verification,
        testing::{TestBackend, TestFrontend, decl},
    };

    const ALPHA: &[CheckpointDecl] =
        &[decl("alpha.ast", &[], "ast"), decl("alpha.typed", &[], "tast")];
    const BETA: &[CheckpointDecl] = &[decl("beta.ast", &[], "bast")];
    const OUT: &[CheckpointDecl] = &[decl("one.out", &[], "o")];
    const ONE: TestBackend = TestBackend::new("one", &["wasm32-wasip2"], OUT);

    const ALPHA_AST: CheckpointId = CheckpointId::new("alpha.ast");
    const ALPHA_TYPED: CheckpointId = CheckpointId::new("alpha.typed");
    const BETA_AST: CheckpointId = CheckpointId::new("beta.ast");

    type Step = Result<Flow<()>, Report>;

    fn go() -> Step {
        Ok(Flow::Continue(()))
    }

    /// Reaches `id` and passes the answer on as the frontend's own.
    fn reach(checkpoints: &mut Checkpoints<'_>, id: CheckpointId) -> Step {
        Ok(checkpoints.reach(id, 0u8)?.map_continue_unit())
    }

    trait FlowExt {
        fn map_continue_unit(self) -> Flow<()>;
    }

    impl<T> FlowExt for Flow<T> {
        fn map_continue_unit(self) -> Flow<()> {
            match self {
                ControlFlow::Continue(_) => ControlFlow::Continue(()),
                ControlFlow::Break(stopped) => ControlFlow::Break(stopped),
            }
        }
    }

    fn compiler(alpha: TestFrontend, beta: TestFrontend) -> Compiler {
        Compiler::builder().frontend(alpha).frontend(beta).backend(ONE).build().unwrap()
    }

    fn alpha() -> TestFrontend {
        TestFrontend::new("alpha", &["a"]).with_checkpoints(ALPHA)
    }

    fn beta() -> TestFrontend {
        TestFrontend::new("beta", &["b"]).with_checkpoints(BETA)
    }

    fn request(inputs: &[&str], stop_after: &str) -> CompilationRequest {
        let mut request = CompilationRequest::new();
        request.inputs = inputs.iter().map(|name| Input::memory(*name, "")).collect();
        request.stop_after = Some(stop_after.into());
        // The one default output is kept, not written to the working directory.
        request.output_file = Some("-".parse().unwrap());
        request
    }

    fn session<'c>(compiler: &'c Compiler, inputs: &[&str], stop_after: &str) -> Session<'c> {
        let mut request = request(inputs, stop_after);
        request.target = Some("wasm32-wasip2".parse().unwrap());
        compiler.session(request).unwrap()
    }

    fn ids(checkpoints: &[CheckpointId]) -> Vec<&'static str> {
        checkpoints.iter().map(|id| id.as_str()).collect()
    }

    fn failure(session: &mut Session<'_>) -> String {
        session.translate(&mut ()).unwrap_err().to_string()
    }

    // Checkpoints

    #[test]
    fn a_frontend_reaching_a_checkpoint_it_did_not_declare_is_an_error_naming_both() {
        let alpha = alpha().with_translate(|_, _, _, c| reach(c, BETA_AST));
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a", "y.b"], "one.out");
        let message = failure(&mut session);
        assert!(
            message.contains("frontend 'alpha'") && message.contains("'beta.ast'"),
            "{message}"
        );
        assert!(message.contains("'alpha.ast', 'alpha.typed'"), "{message}");
    }

    #[test]
    fn a_frontend_that_declares_nothing_cannot_reach_anything() {
        let beta =
            TestFrontend::new("beta", &["b"]).with_translate(|_, _, _, c| reach(c, ALPHA_AST));
        let compiler = compiler(alpha(), beta);
        let mut session = session(&compiler, &["y.b"], "one.out");
        let message = failure(&mut session);
        assert!(
            message.contains("frontend 'beta'") && message.contains("declares none"),
            "{message}"
        );
    }

    #[test]
    fn a_frontend_reaching_a_checkpoint_of_the_harness_is_an_error() {
        let alpha = alpha().with_translate(|_, _, _, c| reach(c, CheckpointId::IR_TRANSLATED));
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        let message = failure(&mut session);
        assert!(
            message.contains("frontend 'alpha'") && message.contains("'ir.translated'"),
            "{message}"
        );
    }

    #[test]
    fn a_checkpoint_reached_twice_is_an_error_naming_the_stage_and_the_checkpoint() {
        let alpha = alpha().with_translate(|_, _, _, c| {
            let _ = reach(c, ALPHA_AST)?;
            reach(c, ALPHA_AST)
        });
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        let message = failure(&mut session);
        assert!(message.contains("frontend 'alpha'"), "{message}");
        assert!(message.contains("'alpha.ast' after 'alpha.ast'"), "{message}");
    }

    #[test]
    fn a_checkpoint_reached_out_of_order_is_an_error_naming_both() {
        let alpha = alpha().with_translate(|_, _, _, c| {
            let _ = reach(c, ALPHA_TYPED)?;
            reach(c, ALPHA_AST)
        });
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        let message = failure(&mut session);
        assert!(message.contains("'alpha.ast' after 'alpha.typed'"), "{message}");
        assert_eq!(ids(session.reached()), ["alpha.typed"]);
    }

    #[test]
    fn an_observer_is_told_of_each_checkpoint_in_order_with_its_artifact() {
        struct Collect(Vec<(CheckpointId, Option<u8>)>);
        impl crate::Observer for Collect {
            fn on_checkpoint(&mut self, checkpoint: CheckpointId, artifact: &crate::Artifact) {
                assert_eq!(artifact.checkpoint(), checkpoint);
                self.0.push((checkpoint, artifact.downcast_ref::<u8>().copied()));
            }
        }
        let alpha = alpha().with_translate(|_, _, _, c| {
            assert_eq!(c.reach(ALPHA_AST, 1u8)?, ControlFlow::Continue(1));
            assert_eq!(c.reach(ALPHA_TYPED, 2u8)?, ControlFlow::Continue(2));
            go()
        });
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        let mut observer = Collect(Vec::new());
        let _ = session.translate(&mut observer).unwrap();
        assert_eq!(observer.0[0], (ALPHA_AST, Some(1)));
        assert_eq!(observer.0[1], (ALPHA_TYPED, Some(2)));
        assert_eq!(observer.0[2].0, CheckpointId::IR_TRANSLATED);
        assert_eq!(observer.0[2].1, None, "the world is not a u8");
        assert_eq!(ids(session.reached()), ["alpha.ast", "alpha.typed", "ir.translated"]);
    }

    #[test]
    fn the_stop_point_keeps_the_artifact_and_breaks() {
        let alpha = alpha().with_translate(|_, _, _, c| {
            let flow = c.reach(ALPHA_AST, 7u8)?;
            assert_eq!(flow, ControlFlow::Break(Stopped::new(ALPHA_AST)));
            Ok(flow.map_continue_unit())
        });
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "alpha.ast");
        let mut observer = RecordingObserver::default();
        let flow = session.translate(&mut observer).unwrap();
        assert_eq!(flow, ControlFlow::Break(Stopped::new(ALPHA_AST)));
        assert_eq!(observer.checkpoints(), &[ALPHA_AST]);
        let kept = session.stopped().expect("the artifact is kept");
        assert_eq!(kept.checkpoint(), ALPHA_AST);
        assert_eq!(kept.downcast_ref::<u8>(), Some(&7));
    }

    #[test]
    fn nothing_is_reached_after_the_stop_point() {
        let alpha = alpha().with_translate(|_, _, _, c| {
            let _ = c.reach(ALPHA_AST, 0u8)?;
            reach(c, ALPHA_TYPED)
        });
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "alpha.ast");
        let message = failure(&mut session);
        assert!(
            message.contains("frontend 'alpha'")
                && message.contains("'alpha.typed'")
                && message.contains("stopped at 'alpha.ast'"),
            "{message}"
        );
    }

    #[test]
    fn a_declared_checkpoint_that_is_never_reached_is_not_an_error() {
        let alpha = alpha().with_translate(|_, _, _, c| reach(c, ALPHA_AST));
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        assert_eq!(session.translate(&mut ()).unwrap(), ControlFlow::Continue(()));
        assert_eq!(ids(session.reached()), ["alpha.ast", "ir.translated"]);
    }

    // The translate step

    #[test]
    fn frontends_run_in_the_order_of_their_first_input_each_with_its_inputs() {
        let alpha = alpha().with_translate(|_, inputs, _, c| {
            let names: Vec<_> = inputs.iter().map(|i| i.name().into_owned()).collect();
            assert_eq!(names, ["one.a", "three.a"]);
            reach(c, ALPHA_AST)
        });
        let beta = beta().with_translate(|_, inputs, _, c| {
            assert_eq!(inputs.len(), 1);
            reach(c, BETA_AST)
        });
        let compiler = compiler(alpha, beta);
        let mut session = session(&compiler, &["two.b", "one.a", "three.a"], "one.out");
        let mut observer = RecordingObserver::default();
        assert_eq!(session.translate(&mut observer).unwrap(), ControlFlow::Continue(()));
        assert_eq!(ids(observer.checkpoints()), ["beta.ast", "alpha.ast", "ir.translated"]);
    }

    #[test]
    fn a_stop_at_the_first_frontends_checkpoint_calls_no_later_frontend() {
        let beta = beta().with_translate(|_, _, _, c| reach(c, BETA_AST));
        let alpha = alpha().with_translate(|_, _, _, _| Err(Report::msg("alpha must not run")));
        let compiler = compiler(alpha, beta);
        let mut session = session(&compiler, &["two.b", "one.a"], "beta.ast");
        let flow = session.translate(&mut ()).unwrap();
        assert_eq!(flow, ControlFlow::Break(Stopped::new(BETA_AST)));
        assert_eq!(ids(session.reached()), ["beta.ast"]);
    }

    #[test]
    fn stopping_at_translated_keeps_the_world() {
        let compiler = compiler(alpha(), beta());
        let mut session = session(&compiler, &["x.a"], "ir.translated");
        let flow = session.translate(&mut ()).unwrap();
        assert_eq!(flow, ControlFlow::Break(Stopped::new(CheckpointId::IR_TRANSLATED)));
        let kept = session.stopped().unwrap();
        assert!(kept.downcast_ref::<WorldRef>() == Some(&session.world()));
    }

    #[test]
    fn a_frontend_error_is_returned_as_it_is() {
        let alpha = alpha().with_translate(|_, _, _, _| Err(Report::msg("alpha failed on x.a")));
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        assert_eq!(failure(&mut session), "alpha failed on x.a");
        assert!(session.reached().is_empty());
    }

    #[test]
    fn a_break_the_frontend_did_not_reach_is_an_error() {
        let alpha =
            alpha().with_translate(|_, _, _, _| Ok(ControlFlow::Break(Stopped::new(ALPHA_AST))));
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "alpha.ast");
        let message = failure(&mut session);
        assert!(
            message.contains("frontend 'alpha'")
                && message.contains("'alpha.ast'")
                && message.contains("never reached"),
            "{message}"
        );
    }

    #[test]
    fn a_frontend_that_never_reaches_the_stop_point_it_declared_is_an_error() {
        let compiler = compiler(alpha(), beta());
        let mut session = session(&compiler, &["x.a"], "alpha.typed");
        let message = failure(&mut session);
        assert!(
            message.contains("frontend 'alpha'")
                && message.contains("'alpha.typed'")
                && message.contains("asked to stop there"),
            "{message}"
        );
    }

    #[test]
    fn a_frontend_that_goes_on_after_the_stop_is_an_error() {
        let alpha = alpha().with_translate(|_, _, _, c| {
            let _ = c.reach(ALPHA_AST, 0u8)?;
            go()
        });
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "alpha.ast");
        let message = failure(&mut session);
        assert!(
            message.contains("frontend 'alpha'") && message.contains("'alpha.ast'"),
            "{message}"
        );
    }

    #[test]
    fn translating_twice_is_an_error() {
        let compiler = compiler(alpha(), beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        let _ = session.translate(&mut ()).unwrap();
        let message = failure(&mut session);
        assert!(message.contains("twice"), "{message}");
        assert_eq!(ids(session.reached()), ["ir.translated"], "the first run is kept");
    }

    /// Leaves a function whose entry block has no terminator.
    fn leave_invalid_ir(
        session: &Session<'_>,
        _inputs: &[Input],
        world: WorldRef,
        c: &mut Checkpoints<'_>,
    ) -> Step {
        let module = WorldBuilder::new(world).declare_module(Ident::with_empty_span("m".into()))?;
        let mut builder = ModuleBuilder::new(module);
        let signature = Signature::new(session.context(), [], []);
        let function = builder.define_function(
            Ident::with_empty_span("f".into()),
            Visibility::Public,
            signature,
        )?;
        let _ = FunctionBuilder::new(function, builder.builder());
        reach(c, ALPHA_AST)
    }

    #[test]
    fn invalid_ir_left_by_a_frontend_is_an_error_saying_the_ir_does_not_verify() {
        let compiler = compiler(alpha().with_translate(leave_invalid_ir), beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        let error = session.translate(&mut ()).unwrap_err();
        let frames: Vec<_> =
            error.contexts().iter().map(|frame| frame.message().to_string()).collect();
        assert!(
            frames
                .iter()
                .any(|frame| frame.contains("the IR the frontends produced does not verify")),
            "{frames:?} / {error}"
        );
        assert_eq!(ids(session.reached()), ["alpha.ast"], "ir.translated is not reached");
    }

    #[test]
    fn invalid_ir_is_accepted_when_verification_is_off() {
        let compiler = compiler(alpha().with_translate(leave_invalid_ir), beta());
        let mut request = request(&["x.a"], "one.out");
        request.target = Some("wasm32-wasip2".parse().unwrap());
        request.ir.verification = Verification::None;
        let mut session = compiler.session(request).unwrap();
        assert_eq!(session.translate(&mut ()).unwrap(), ControlFlow::Continue(()));
        assert_eq!(ids(session.reached()), ["alpha.ast", "ir.translated"]);
    }

    #[test]
    fn the_data_layout_is_visible_to_a_frontend_inside_translate() {
        let alpha = alpha().with_translate(|session, _, world, c| {
            let layout = DataLayout::closest(&world.as_operation_ref().borrow())
                .ok_or_else(|| Report::msg("the world states no data layout"))?;
            assert_eq!(layout.spec(), session.target().unwrap().data_layout());
            reach(c, ALPHA_AST)
        });
        let compiler = compiler(alpha, beta());
        let mut session = session(&compiler, &["x.a"], "one.out");
        assert_eq!(session.translate(&mut ()).unwrap(), ControlFlow::Continue(()));
    }
}
