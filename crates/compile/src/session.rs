//! A session: one request resolved against a compiler, and the steps that run it.

use core::{any::Any, cell::RefCell};
use std::{path::PathBuf, rc::Rc};

use runic_core::{
    BuilderExt, Context, DataLayoutSpec, Report, SourceSpan,
    dialects::builtin::{World, WorldRef},
    set_data_layout_spec,
};

use crate::{
    Backend, Compilation, CompilationRequest, Compiler, Destination, Emitted, Flow, Frontend, Goal,
    Input, InputSource, OutputNaming, Route, Triple, compilation::Finished,
};

mod checkpoints;
mod emit;
mod inputs;
mod pipeline;
mod resolve;
mod translate;

pub use self::checkpoints::Checkpoints;
use self::checkpoints::Progress;
pub(crate) use self::pipeline::default_optimization;

/// A frontend that takes part in a session, with the inputs it reads.
type Participant<'c> = (&'c dyn Frontend, Vec<Input>);

/// What a compilation is for: the triple, the backend that lowers for it, and the data layout the
/// backend states for the triple.
///
/// A session has one when the compilation reaches a backend. Frontends read the data layout
/// through the session's world, and the backend reads the rest.
pub struct Target<'c> {
    triple: Triple,
    backend: &'c dyn Backend,
    data_layout: DataLayoutSpec,
}

impl Target<'_> {
    /// The triple the compilation is for.
    pub fn triple(&self) -> &Triple {
        &self.triple
    }

    /// The backend that lowers for the triple.
    pub fn backend(&self) -> &dyn Backend {
        self.backend
    }

    /// The data layout the backend states for the triple.
    pub fn data_layout(&self) -> &DataLayoutSpec {
        &self.data_layout
    }
}

/// One request resolved against a [Compiler]: which frontend reads each input, which target and
/// backend the compilation is for, the route and the goal, and the context and world the stages
/// build in.
///
/// The session owns the one [Context] of the compilation, and the world in it is empty except for
/// the data layout of the target. The stages that run read the session and nothing else decides
/// for them; the session borrows its compiler, so it cannot outlive it.
pub struct Session<'c> {
    compiler: &'c Compiler,
    request: CompilationRequest,
    name: String,
    frontends: Vec<Participant<'c>>,
    target: Option<Target<'c>>,
    route: Route,
    goal: Goal,
    world: WorldRef,
    progress: Progress,
    files_read: RefCell<Vec<PathBuf>>,
    // Declared last, so that it is dropped last: the artifact the progress keeps may hold IR.
    context: Rc<Context>,
}

impl Compiler {
    /// Resolves `request` against this compiler.
    ///
    /// In order: the unit's name; the frontend of each input; the target and backend; the route of
    /// the frontends that have inputs and the backend; the goal; then a new context with an empty
    /// world that states the target's data layout. Fails at the first step that cannot be
    /// resolved, with a message that names what the request wrote and lists what is known, and
    /// then nothing has been created.
    pub fn session(&self, request: CompilationRequest) -> Result<Session<'_>, Report> {
        let name = request.unit_name()?;
        let frontends = resolve::assign_inputs(self, &request.inputs)?;
        let target = resolve::resolve_target(
            self,
            request.target.as_ref(),
            request.backend.as_deref(),
            request.stop_after.is_some(),
        )?;

        let declared: Vec<_> = frontends
            .iter()
            .map(|(frontend, _)| (frontend.name(), frontend.checkpoints()))
            .collect();
        let backend = target.as_ref().map(|t| (t.backend.name(), t.backend.checkpoints()));
        let route = Route::new(&declared, backend)?;

        if target.is_none() {
            // A name that only a backend would put on the route is missing for want of a target;
            // any other unknown name gets the goal's own error.
            let wanted = request
                .stop_after
                .iter()
                .map(String::as_str)
                .chain(request.emit.iter().map(|emit| emit.checkpoint.as_str()));
            for wanted in wanted {
                if route.resolve(wanted).is_some() {
                    continue;
                }
                let with_backend = self.backends().any(|backend| {
                    Route::new(&declared, Some((backend.name(), backend.checkpoints())))
                        .is_ok_and(|route| route.resolve(wanted).is_some())
                });
                if with_backend {
                    let known: Vec<String> =
                        self.targets().iter().map(|triple| format!("'{triple}'")).collect();
                    return Err(Report::msg(format!(
                        "the checkpoint '{wanted}' is reached only with a backend, and no target \
                         was given: give one with --target (the targets are {})",
                        known.join(", ")
                    )));
                }
            }
        }

        let goal = crate::resolve_goal(
            &route,
            request.stop_after.as_deref(),
            &request.emit,
            &OutputNaming {
                unit: &name,
                output_dir: request.output_dir.as_deref(),
                output_file: request.output_file.as_ref(),
            },
        )?;

        // An output written over an input would destroy what is being compiled.
        for output in goal.outputs() {
            let Destination::File(path) = output.destination() else {
                continue;
            };
            let overwritten = request
                .inputs
                .iter()
                .any(|input| matches!(&input.source, InputSource::Path(source) if source == path));
            if overwritten {
                return Err(Report::msg(format!(
                    "the output of checkpoint '{}' would be written to '{}', which is an input",
                    output.checkpoint(),
                    path.display()
                )));
            }
        }

        let context = Rc::new(Context::default());
        let world = context.clone().builder().create::<World, ()>(SourceSpan::UNKNOWN)()?;
        if let Some(target) = &target {
            set_data_layout_spec(
                &mut world.as_operation_ref().borrow_mut(),
                target.data_layout.clone(),
            );
        }
        Ok(Session {
            compiler: self,
            request,
            name,
            frontends,
            target,
            route,
            goal,
            context,
            world,
            progress: Progress::default(),
            files_read: RefCell::default(),
        })
    }
}

impl<'c> Session<'c> {
    /// Ends the session: moves what it did into a [Compilation].
    ///
    /// `outcome` is what the last step that ran answered. A compilation succeeded when that is the
    /// `Break` of its stop point. The error of the compilation is otherwise: the outcome's error;
    /// when the diagnostics pending on the context hold errors that a stage reported without
    /// failing, an error that says so (the diagnostics stay in the result); and when the outcome
    /// is not the stop at the stop point (the session is finished part of the way), an error that
    /// names the stop point. The pending diagnostics are taken from the context.
    pub fn finish(self, outcome: Result<Flow<()>, Report>) -> Compilation {
        let diagnostics = self.context.take_diagnostics();
        let stop = self.goal.stop_after();
        let error = match outcome {
            Err(report) => Some(report),
            Ok(_) if diagnostics.has_errors() => Some(Report::msg(format!(
                "the compilation reported {} error(s)",
                diagnostics.counts().errors()
            ))),
            Ok(Flow::Break(stopped))
                if stopped.checkpoint() == stop && self.progress.stopped.is_some() =>
            {
                None
            }
            Ok(_) => Some(Report::msg(format!(
                "the compilation was ended before it reached its stop point '{stop}'"
            ))),
        };
        let Session {
            progress,
            files_read,
            context,
            ..
        } = self;
        Compilation::new(
            error,
            Finished {
                reached: progress.reached,
                stopped: progress.stopped,
                emitted: progress.emitted,
                pass_output: progress.pass_output,
                files_read: files_read.into_inner(),
            },
            diagnostics,
            context,
        )
    }

    /// The compiler this session was resolved against.
    pub fn compiler(&self) -> &'c Compiler {
        self.compiler
    }

    /// The context every IR object of the compilation lives in.
    pub fn context(&self) -> &Rc<Context> {
        &self.context
    }

    /// The request this session resolved.
    pub fn request(&self) -> &CompilationRequest {
        &self.request
    }

    /// The name of the unit being compiled.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The target, when the compilation reaches a backend.
    pub fn target(&self) -> Option<&Target<'c>> {
        self.target.as_ref()
    }

    /// The checkpoints the compilation passes through.
    pub fn route(&self) -> &Route {
        &self.route
    }

    /// Where the compilation stops and what it outputs.
    pub fn goal(&self) -> &Goal {
        &self.goal
    }

    /// The world the frontends translate into. It states the data layout of the target, if any.
    pub fn world(&self) -> WorldRef {
        self.world
    }

    /// The options of type `T` the request carries, if any.
    pub fn options<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.request.options::<T>()
    }

    /// The outputs delivered so far, in the order their checkpoints were reached: files that were
    /// written, and the bytes meant for standard output or for memory.
    pub fn emitted(&self) -> &[Emitted] {
        &self.progress.emitted
    }

    /// The frontends that have inputs, in the order they run, each with its inputs.
    pub fn frontends(&self) -> impl Iterator<Item = (&'c dyn Frontend, &[Input])> {
        self.frontends.iter().map(|(frontend, inputs)| (*frontend, inputs.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use runic_core::{DataLayout, Endianness, PointerSpec};

    use super::*;
    use crate::{
        CheckpointDecl, EmitRequest,
        testing::{TestBackend, TestFrontend, decl},
    };

    const ALPHA: &[CheckpointDecl] = &[decl("alpha.ast", &["ast"], "ast")];
    const BETA: &[CheckpointDecl] = &[decl("beta.ast", &[], "bast")];
    const OUT: &[CheckpointDecl] = &[decl("one.out", &["out"], "o")];
    const A: TestFrontend = TestFrontend::new("alpha", &["a"]).with_checkpoints(ALPHA);
    const B: TestFrontend = TestFrontend::new("beta", &["b"]).with_checkpoints(BETA);
    const ONE: TestBackend = TestBackend::new("one", &["wasm32-wasip2"], OUT);

    fn compiler() -> Compiler {
        Compiler::builder().frontend(A).frontend(B).backend(ONE).build().unwrap()
    }

    fn request(inputs: &[&str]) -> CompilationRequest {
        let mut request = CompilationRequest::new();
        request.inputs = inputs.iter().map(|name| Input::memory(*name, "")).collect();
        request
    }

    fn error(compiler: &Compiler, request: CompilationRequest) -> String {
        match compiler.session(request) {
            Ok(_) => panic!("the request should not resolve"),
            Err(error) => error.to_string(),
        }
    }

    fn route_ids(session: &Session<'_>) -> Vec<&'static str> {
        session.route().checkpoints().iter().map(|decl| decl.id.as_str()).collect()
    }

    #[test]
    fn a_request_with_no_name_and_no_input_fails_on_the_name() {
        let message = error(&compiler(), CompilationRequest::new());
        assert!(message.contains("--name"), "{message}");
    }

    #[test]
    fn a_request_with_a_name_and_no_input_fails_saying_so() {
        let mut request = CompilationRequest::new();
        request.name = Some("unit".into());
        assert!(error(&compiler(), request).contains("no input was given"));
    }

    #[test]
    fn an_input_no_frontend_claims_fails_on_the_inputs() {
        let message = error(&compiler(), request(&["x.zzz"]));
        assert!(message.contains("'x.zzz'") && message.contains("'a', 'b'"), "{message}");
    }

    #[test]
    fn a_target_no_backend_supports_fails_on_the_target() {
        let mut request = request(&["x.a"]);
        request.target = Some("mips-unknown-none".parse().unwrap());
        assert!(error(&compiler(), request).contains("no backend supports the target"));
    }

    #[test]
    fn no_target_with_a_backend_and_no_stop_point_fails_on_the_target() {
        let message = error(&compiler(), request(&["x.a"]));
        assert!(
            message.contains("no target was given") && message.contains("'wasm32-wasip2'"),
            "{message}"
        );
    }

    #[test]
    fn a_stop_point_of_the_backend_with_no_target_names_it_and_asks_for_a_target() {
        let mut request = request(&["x.a"]);
        request.stop_after = Some("one.out".into());
        let message = error(&compiler(), request);
        assert!(message.contains("'one.out'"), "{message}");
        assert!(
            message.contains("no target was given") && message.contains("--target"),
            "{message}"
        );
        assert!(message.contains("'wasm32-wasip2'"), "{message}");
    }

    #[test]
    fn an_emit_of_the_backend_with_no_target_names_it_and_asks_for_a_target() {
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.optimized".into());
        request.emit.push("out".parse::<EmitRequest>().unwrap());
        let message = error(&compiler(), request);
        assert!(message.contains("'out'") && message.contains("--target"), "{message}");
    }

    #[test]
    fn a_misspelt_checkpoint_with_no_target_is_not_blamed_on_the_target() {
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.optimised".into());
        let message = error(&compiler(), request);
        assert!(message.contains("unknown checkpoint 'ir.optimised'"), "{message}");
        assert!(!message.contains("--target"), "{message}");
        assert!(message.contains("'ir.optimized'"), "{message}");
    }

    #[test]
    fn the_lowered_ir_with_no_target_asks_for_a_target() {
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.lowered".into());
        let message = error(&compiler(), request);
        assert!(message.contains("'ir.lowered'") && message.contains("--target"), "{message}");
    }

    #[test]
    fn a_checkpoint_of_a_frontend_with_no_inputs_is_not_on_the_route() {
        let mut request = request(&["x.a"]);
        request.target = Some("wasm32-wasip2".parse().unwrap());
        request.stop_after = Some("beta.ast".into());
        let message = error(&compiler(), request);
        assert!(message.contains("unknown checkpoint 'beta.ast'"), "{message}");
    }

    #[test]
    fn an_unknown_stop_point_fails_on_the_goal_when_the_compiler_has_no_backend() {
        let compiler = Compiler::builder().frontend(A).build().unwrap();
        let mut request = request(&["x.a"]);
        request.stop_after = Some("nope".into());
        let message = error(&compiler, request);
        assert!(
            message.contains("unknown checkpoint 'nope'") && message.contains("alpha.ast"),
            "{message}"
        );
    }

    #[test]
    fn an_output_file_for_two_outputs_fails_on_the_goal() {
        let mut request = request(&["x.a"]);
        request.target = Some("wasm32-wasip2".parse().unwrap());
        request.output_file = Some("-".parse().unwrap());
        request.emit.push("ast".parse().unwrap());
        request.emit.push("out".parse().unwrap());
        assert!(compiler().session(request).is_err());
    }

    #[test]
    fn a_resolved_session_reports_what_it_resolved() {
        let compiler = compiler();
        let mut request = request(&["app.b", "other.a", "more.b"]);
        request.target = Some("wasm32-wasip2".parse().unwrap());
        let session = compiler.session(request).unwrap();

        assert_eq!(session.name(), "app");
        let target = session.target().unwrap();
        assert_eq!(target.triple().as_str(), "wasm32-wasip2");
        assert_eq!(target.backend().name(), "one");
        assert_eq!(
            route_ids(&session),
            [
                "beta.ast",
                "alpha.ast",
                "ir.translated",
                "ir.optimized",
                "ir.lowered",
                "one.out"
            ]
        );
        assert_eq!(session.goal().stop_after().as_str(), "one.out");
        let frontends: Vec<(&str, Vec<String>)> = session
            .frontends()
            .map(|(f, inputs)| (f.name(), inputs.iter().map(|i| i.name().into_owned()).collect()))
            .collect();
        assert_eq!(
            frontends,
            [
                ("beta", vec!["app.b".to_owned(), "more.b".to_owned()]),
                ("alpha", vec!["other.a".to_owned()]),
            ]
        );
        assert_eq!(session.request().inputs.len(), 3);
        assert!(core::ptr::eq(session.compiler(), &compiler));
    }

    #[test]
    fn a_given_name_names_the_unit_and_a_default_target_resolves_the_session() {
        let compiler = Compiler::builder()
            .frontend(A)
            .backend(ONE)
            .default_target("wasm32-wasip2")
            .build()
            .unwrap();
        let mut request = request(&["x.a"]);
        request.name = Some("given".into());
        let session = compiler.session(request).unwrap();
        assert_eq!(session.name(), "given");
        assert_eq!(session.target().unwrap().triple().as_str(), "wasm32-wasip2");
    }

    #[test]
    fn a_session_with_no_target_stopping_early_has_no_backend_on_its_route() {
        let compiler = compiler();
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.optimized".into());
        let session = compiler.session(request).unwrap();
        assert!(session.target().is_none());
        assert_eq!(route_ids(&session), ["alpha.ast", "ir.translated", "ir.optimized"]);
        assert_eq!(session.goal().stop_after().as_str(), "ir.optimized");
    }

    #[test]
    fn an_output_that_would_overwrite_an_input_is_an_error() {
        let mut request = CompilationRequest::new();
        request.inputs = vec![Input::path("unit.a")];
        request.target = Some("wasm32-wasip2".parse().unwrap());
        request.emit.push("ir.translated=unit.a".parse::<EmitRequest>().unwrap());
        let message = error(&compiler(), request);
        assert_eq!(
            message,
            "the output of checkpoint 'ir.translated' would be written to 'unit.a', which is an \
             input"
        );
    }

    #[test]
    fn a_session_of_a_compiler_with_no_backend_needs_no_target() {
        let compiler = Compiler::builder().frontend(A).build().unwrap();
        let session = compiler.session(request(&["x.a"])).unwrap();
        assert!(session.target().is_none());
        assert_eq!(session.goal().stop_after().as_str(), "ir.optimized");
    }

    #[test]
    fn the_world_states_the_backends_data_layout() {
        let compiler = compiler();
        let mut request = request(&["x.a"]);
        request.target = Some("wasm32-wasip2".parse().unwrap());
        let session = compiler.session(request).unwrap();
        let world = session.world().as_operation_ref();
        let layout = DataLayout::closest(&world.borrow()).expect("the world states a layout");
        let expected = ONE.data_layout(session.target().unwrap().triple()).unwrap();
        assert_eq!(layout.spec(), &expected);
        assert_eq!(session.target().unwrap().data_layout(), &expected);
        let spec =
            DataLayoutSpec::new(Endianness::Little, PointerSpec::sized(32).unwrap()).unwrap();
        assert_eq!(layout.spec(), &spec);
    }

    #[test]
    fn the_world_states_no_data_layout_when_there_is_no_target() {
        let compiler = compiler();
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.translated".into());
        let session = compiler.session(request).unwrap();
        let world = session.world().as_operation_ref();
        assert!(DataLayout::closest(&world.borrow()).is_none());
    }

    #[test]
    fn options_in_the_request_are_read_through_the_session() {
        #[derive(Debug, PartialEq)]
        struct Options(u8);
        let compiler = compiler();
        let mut request = request(&["x.a"]);
        request.stop_after = Some("ir.translated".into());
        request.set_options(Options(7));
        let session = compiler.session(request).unwrap();
        assert_eq!(session.options::<Options>(), Some(&Options(7)));
        assert_eq!(session.options::<u32>(), None);
    }
}
