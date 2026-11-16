use runic_core::{Report, pass::OpPassManager};

use crate::{Backend, Declarer, Frontend, Route, Session, Triple, session::default_optimization};

/// What adds the optimization passes of a compilation to a pass manager anchored on the world.
type Optimization =
    Box<dyn Fn(&Session<'_>, &mut OpPassManager) -> Result<(), Report> + Send + Sync>;

/// A compiler: the frontends and backends it offers and the target it assumes, checked once.
///
/// A compiler is built with [Compiler::builder], whose `build` rejects every configuration that
/// could fail later for a reason the compiler's author controls. Frontends and backends are kept
/// in the order they were registered, and every listing and lookup reads them in that order.
/// One registered stage serves every session of the compiler.
pub struct Compiler {
    frontends: Vec<Box<dyn Frontend>>,
    backends: Vec<Box<dyn Backend>>,
    default_target: Option<Triple>,
    optimization: Option<Optimization>,
}

impl Compiler {
    /// A builder with no frontend, no backend and no default target.
    pub fn builder() -> CompilerBuilder {
        CompilerBuilder {
            frontends: Vec::new(),
            backends: Vec::new(),
            default_target: None,
            optimization: None,
        }
    }

    /// The frontends, in the order they were registered.
    pub fn frontends(&self) -> impl Iterator<Item = &dyn Frontend> {
        self.frontends.iter().map(|frontend| &**frontend)
    }

    /// The backends, in the order they were registered.
    pub fn backends(&self) -> impl Iterator<Item = &dyn Backend> {
        self.backends.iter().map(|backend| &**backend)
    }

    /// The frontend called `name`, the value of `--language`.
    pub fn frontend(&self, name: &str) -> Option<&dyn Frontend> {
        self.frontends().find(|frontend| frontend.name() == name)
    }

    /// The frontend that claims `extension`, written without the dot. The match is exact: case
    /// is significant.
    pub fn frontend_for_extension(&self, extension: &str) -> Option<&dyn Frontend> {
        self.frontends().find(|frontend| frontend.extensions().contains(&extension))
    }

    /// The backend called `name`, the value of `--backend`.
    pub fn backend(&self, name: &str) -> Option<&dyn Backend> {
        self.backends().find(|backend| backend.name() == name)
    }

    /// Every target a backend lists, in the order the backends were registered, each once.
    pub fn targets(&self) -> Vec<Triple> {
        unique_targets(&self.backends)
    }

    /// The target a compilation is for when its request names none.
    pub fn default_target(&self) -> Option<&Triple> {
        self.default_target.as_ref()
    }

    /// Adds the optimization passes of this compiler to `passes`: those its builder gave, or the
    /// default for the session's optimization level.
    pub(crate) fn add_optimization_passes(
        &self,
        session: &Session<'_>,
        passes: &mut OpPassManager,
    ) -> Result<(), Report> {
        match &self.optimization {
            Some(optimization) => optimization(session, passes),
            None => default_optimization(session, passes),
        }
    }
}

/// Collects the stages and the default target of a [Compiler], and checks them in [build].
///
/// [build]: CompilerBuilder::build
pub struct CompilerBuilder {
    frontends: Vec<Box<dyn Frontend>>,
    backends: Vec<Box<dyn Backend>>,
    default_target: Option<String>,
    optimization: Option<Optimization>,
}

impl CompilerBuilder {
    /// Registers a frontend, after those registered before it.
    pub fn frontend(mut self, frontend: impl Frontend + 'static) -> Self {
        self.frontends.push(Box::new(frontend));
        self
    }

    /// Registers a backend, after those registered before it.
    pub fn backend(mut self, backend: impl Backend + 'static) -> Self {
        self.backends.push(Box::new(backend));
        self
    }

    /// Sets the target used when a request names none, as the text of a triple. The text is
    /// checked by `build`.
    pub fn default_target(mut self, triple: &str) -> Self {
        self.default_target = Some(triple.to_owned());
        self
    }

    /// Replaces the default optimization passes of the compiler with `optimization`.
    ///
    /// The optimization pipeline calls it after the passes of the frontends, with the session and
    /// a pass manager anchored on the world, and it adds the passes to run. Without one, the
    /// compiler runs no pass at `-O0` and, at every other level, `canonicalizer`, `cse`, `sccp`,
    /// `canonicalizer` and `dce`, in that order. A request's own pipeline replaces all of it.
    pub fn optimization(
        mut self,
        optimization: impl Fn(&Session<'_>, &mut OpPassManager) -> Result<(), Report>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.optimization = Some(Box::new(optimization));
        self
    }

    /// Checks the configuration and builds the compiler.
    ///
    /// Fails, with a message that names the stage or the text at fault, when:
    ///
    /// * two stages have one name (a name `Route` rejects, such as an empty one, is reported as
    ///   `Route` words it);
    /// * a frontend claims an extension that is empty or holds a `.` (an input's extension is the
    ///   text after its last `.`), or claims one twice, or two frontends claim the same one;
    /// * a backend lists a target it does not support;
    /// * the frontends with no backend, or with any one backend, make a route that [Route::new]
    ///   rejects;
    /// * the default target is not a triple, the compiler has no backend to support it, no backend
    ///   supports it, or a backend that supports it cannot state its data layout.
    pub fn build(self) -> Result<Compiler, Report> {
        let Self {
            frontends,
            backends,
            default_target,
            optimization,
        } = self;

        check_names(&frontends, &backends)?;
        check_extensions(&frontends)?;
        for backend in &backends {
            if let Some(unsupported) = backend.targets().iter().find(|t| !backend.supports(t)) {
                return Err(Report::msg(format!(
                    "the backend '{}' lists the target '{unsupported}', which it does not support",
                    backend.name()
                )));
            }
        }

        let declared: Vec<Declarer<'_>> =
            frontends.iter().map(|f| (f.name(), f.checkpoints())).collect();
        Route::new(&declared, None)?;
        for backend in &backends {
            Route::new(&declared, Some((backend.name(), backend.checkpoints())))?;
        }

        let default_target = match default_target {
            None => None,
            Some(text) => {
                let triple: Triple = text.parse().map_err(|error| {
                    Report::msg(format!("the default target is not usable: {error}"))
                })?;
                if backends.is_empty() {
                    return Err(Report::msg(format!(
                        "the default target '{triple}' was set, but the compiler has no backend: \
                         a target is what a backend generates code for"
                    )));
                }
                let mut supported = false;
                for backend in backends.iter().filter(|b| b.supports(&triple)) {
                    supported = true;
                    backend.data_layout(&triple).map_err(|report| {
                        report.context(format!(
                            "the backend '{}' cannot state the data layout of the default target \
                             '{triple}'",
                            backend.name()
                        ))
                    })?;
                }
                if !supported {
                    let known = known_targets(&backends);
                    return Err(Report::msg(format!(
                        "no backend supports the default target '{triple}'; the known targets are \
                         {known}"
                    )));
                }
                Some(triple)
            }
        };

        Ok(Compiler {
            frontends,
            backends,
            default_target,
            optimization,
        })
    }
}

/// Every target the backends list, in registration order, each once.
fn unique_targets(backends: &[Box<dyn Backend>]) -> Vec<Triple> {
    let mut targets: Vec<Triple> = Vec::new();
    for target in backends.iter().flat_map(|backend| backend.targets()) {
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    targets
}

/// The targets of `backends` as a list for a message, each once, in registration order.
fn known_targets(backends: &[Box<dyn Backend>]) -> String {
    let targets = unique_targets(backends);
    let list: Vec<&str> = targets.iter().map(Triple::as_str).collect();
    if list.is_empty() {
        "none".to_owned()
    } else {
        list.join(", ")
    }
}

/// Checks that no two stages share a name.
fn check_names(
    frontends: &[Box<dyn Frontend>],
    backends: &[Box<dyn Backend>],
) -> Result<(), Report> {
    let stages: Vec<(&str, &'static str)> = frontends
        .iter()
        .map(|f| ("frontend", f.name()))
        .chain(backends.iter().map(|b| ("backend", b.name())))
        .collect();
    for (index, (kind, name)) in stages.iter().enumerate() {
        if let Some((first_kind, _)) = stages[..index].iter().find(|(_, other)| other == name) {
            let what = if first_kind == kind {
                format!("two {kind}s are both called '{name}'")
            } else {
                format!("a {first_kind} and a {kind} are both called '{name}'")
            };
            return Err(Report::msg(format!(
                "{what}: the name of a stage must be unique, because a user chooses a frontend \
                 with --language and a backend with --backend by name"
            )));
        }
    }
    Ok(())
}

/// Checks that every extension is non-empty and is claimed once.
fn check_extensions(frontends: &[Box<dyn Frontend>]) -> Result<(), Report> {
    for (index, frontend) in frontends.iter().enumerate() {
        let name = frontend.name();
        for (position, extension) in frontend.extensions().iter().enumerate() {
            if extension.is_empty() {
                return Err(Report::msg(format!(
                    "the frontend '{name}' claims an empty file extension"
                )));
            }
            if extension.contains('.') {
                return Err(Report::msg(format!(
                    "the frontend '{name}' claims the file extension '{extension}', which holds a \
                     '.': the extension of an input is the text after its last '.'"
                )));
            }
            if frontend.extensions()[..position].contains(extension) {
                return Err(Report::msg(format!(
                    "the frontend '{name}' claims the file extension '{extension}' twice"
                )));
            }
            if let Some(other) =
                frontends[..index].iter().find(|o| o.extensions().contains(extension))
            {
                return Err(Report::msg(format!(
                    "the file extension '{extension}' is claimed by both the frontend '{}' and \
                     the frontend '{name}': an input is sent to a frontend by its extension",
                    other.name()
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CheckpointDecl,
        testing::{TestBackend, TestFrontend, decl},
    };

    const MYLANG: &[CheckpointDecl] = &[decl("mylang.ast", &["ast"], "ast")];
    const WASM: &[CheckpointDecl] = &[decl("wasm.module", &["wat"], "wat")];
    const OTHER: &[CheckpointDecl] = &[decl("other.module", &[], "omod")];
    const MY: TestFrontend =
        TestFrontend::new("mylang", &["my", "mylang"]).with_checkpoints(MYLANG);
    const SECOND: TestFrontend = TestFrontend::new("second", &["two"]);
    const WASM_BACKEND: TestBackend =
        TestBackend::new("wasm", &["wasm32-unknown-unknown", "wasm32-wasip2"], WASM);
    const OTHER_BACKEND: TestBackend =
        TestBackend::new("other", &["wasm32-wasip2", "x86_64-unknown-linux-gnu"], OTHER);

    fn error(builder: CompilerBuilder) -> String {
        match builder.build() {
            Ok(_) => panic!("the configuration should be rejected"),
            Err(error) => error.to_string(),
        }
    }

    fn names<'a>(stages: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
        stages.collect()
    }

    #[test]
    fn a_compiler_without_stages_builds() {
        let compiler = Compiler::builder().build().unwrap();
        assert_eq!(compiler.frontends().count(), 0);
        assert_eq!(compiler.backends().count(), 0);
        assert!(compiler.targets().is_empty());
        assert!(compiler.default_target().is_none());
    }

    #[test]
    fn two_frontends_build() {
        let compiler = Compiler::builder().frontend(MY).frontend(SECOND).build().unwrap();
        assert_eq!(names(compiler.frontends().map(|f| f.name())), ["mylang", "second"]);
    }

    #[test]
    fn two_backends_that_share_a_triple_build() {
        let compiler = Compiler::builder().backend(WASM_BACKEND).backend(OTHER_BACKEND).build();
        assert!(compiler.is_ok());
    }

    #[test]
    fn two_frontends_with_one_name_are_rejected_saying_so() {
        let message = error(Compiler::builder().frontend(MY).frontend(MY));
        assert!(message.contains("two frontends are both called 'mylang'"), "{message}");
    }

    #[test]
    fn two_backends_with_one_name_are_rejected_saying_so() {
        let message = error(Compiler::builder().backend(WASM_BACKEND).backend(WASM_BACKEND));
        assert!(message.contains("two backends are both called 'wasm'"), "{message}");
    }

    #[test]
    fn a_frontend_and_a_backend_with_one_name_are_rejected_saying_so() {
        let same = TestFrontend::new("wasm", &["w"]);
        let message = error(Compiler::builder().frontend(same).backend(WASM_BACKEND));
        assert!(message.contains("a frontend and a backend are both called 'wasm'"), "{message}");
    }

    #[test]
    fn a_frontend_name_that_route_rejects_is_rejected_with_its_reason() {
        let message = error(Compiler::builder().frontend(TestFrontend::new("ir", &["x"])));
        assert!(message.contains("frontend 'ir' cannot be called 'ir'"), "{message}");
        let message = error(Compiler::builder().frontend(TestFrontend::new("", &["x"])));
        assert!(message.contains("the name of frontend '' is empty"), "{message}");
    }

    #[test]
    fn an_empty_extension_is_rejected() {
        let message = error(Compiler::builder().frontend(TestFrontend::new("mylang", &["my", ""])));
        assert_eq!(message, "the frontend 'mylang' claims an empty file extension");
    }

    #[test]
    fn an_extension_claimed_by_two_frontends_is_rejected_naming_both() {
        let clash = TestFrontend::new("second", &["two", "mylang"]);
        let message = error(Compiler::builder().frontend(MY).frontend(clash));
        assert!(
            message.contains(
                "'mylang' is claimed by both the frontend 'mylang' and the frontend 'second'"
            ),
            "{message}"
        );
    }

    #[test]
    fn an_extension_claimed_twice_by_one_frontend_is_rejected() {
        let twice = TestFrontend::new("mylang", &["my", "my"]);
        let message = error(Compiler::builder().frontend(twice));
        assert_eq!(message, "the frontend 'mylang' claims the file extension 'my' twice");
    }

    #[test]
    fn a_route_that_the_frontends_alone_cannot_make_is_rejected_naming_the_stage() {
        const BAD: &[CheckpointDecl] = &[decl("elsewhere.ast", &[], "ast")];
        let bad = TestFrontend::new("mylang", &["my"]).with_checkpoints(BAD);
        let message = error(Compiler::builder().frontend(bad));
        assert!(
            message.contains("frontend 'mylang' declares the checkpoint 'elsewhere.ast'"),
            "{message}"
        );
    }

    #[test]
    fn a_route_that_one_backend_spoils_is_rejected_naming_the_backend() {
        let silent = TestBackend::new("silent", &["wasm32-unknown-unknown"], &[]);
        let message = error(Compiler::builder().frontend(MY).backend(WASM_BACKEND).backend(silent));
        assert!(message.contains("backend 'silent' declares no checkpoint"), "{message}");
    }

    #[test]
    fn a_checkpoint_clash_between_a_frontend_and_a_backend_is_rejected() {
        const CLASH: &[CheckpointDecl] = &[decl("wasm.module", &[], "ast")];
        let backend = TestBackend::new("wasm", &["wasm32-unknown-unknown"], CLASH);
        let message = error(Compiler::builder().frontend(MY).backend(backend));
        assert!(message.contains("'ast'"), "{message}");
    }

    #[test]
    fn a_malformed_default_target_is_rejected_naming_it() {
        let message = error(Compiler::builder().default_target("not a triple"));
        assert!(message.starts_with("the default target is not usable: "), "{message}");
        assert!(message.contains("'not a triple'"), "{message}");
    }

    #[test]
    fn a_default_target_no_backend_supports_is_rejected_listing_the_known_targets() {
        let message = error(
            Compiler::builder()
                .backend(WASM_BACKEND)
                .backend(OTHER_BACKEND)
                .default_target("riscv64-unknown-none"),
        );
        assert!(
            message.contains("no backend supports the default target 'riscv64-unknown-none'"),
            "{message}"
        );
        assert!(
            message.contains("wasm32-unknown-unknown, wasm32-wasip2, x86_64-unknown-linux-gnu"),
            "{message}"
        );
    }

    #[test]
    fn a_default_target_one_or_several_backends_support_is_kept() {
        let compiler = Compiler::builder()
            .backend(WASM_BACKEND)
            .backend(OTHER_BACKEND)
            .default_target("wasm32-wasip2")
            .build()
            .unwrap();
        assert_eq!(compiler.default_target().map(Triple::as_str), Some("wasm32-wasip2"));
        let compiler = Compiler::builder()
            .backend(WASM_BACKEND)
            .default_target("wasm32-unknown-unknown")
            .build()
            .unwrap();
        assert_eq!(compiler.default_target().map(Triple::as_str), Some("wasm32-unknown-unknown"));
    }

    #[test]
    fn a_default_target_is_rejected_when_the_compiler_has_no_backend() {
        let message = error(Compiler::builder().default_target("riscv64-unknown-none"));
        assert_eq!(
            message,
            "the default target 'riscv64-unknown-none' was set, but the compiler has no backend: \
             a target is what a backend generates code for"
        );
    }

    #[test]
    fn an_extension_with_a_dot_is_rejected() {
        let message = error(Compiler::builder().frontend(TestFrontend::new("tar", &["tar.gz"])));
        assert!(
            message.contains("claims the file extension 'tar.gz', which holds a '.'"),
            "{message}"
        );
    }

    #[test]
    fn a_compiler_can_be_shared_between_threads() {
        fn shareable<T: Send + Sync>() {}
        shareable::<Compiler>();
    }

    #[test]
    fn a_frontend_is_found_by_name() {
        let compiler = Compiler::builder().frontend(MY).frontend(SECOND).build().unwrap();
        assert_eq!(compiler.frontend("second").map(|f| f.name()), Some("second"));
        assert!(compiler.frontend("third").is_none());
        assert!(compiler.frontend("").is_none());
    }

    #[test]
    fn a_frontend_is_found_by_the_exact_extension() {
        let compiler = Compiler::builder().frontend(MY).frontend(SECOND).build().unwrap();
        assert_eq!(compiler.frontend_for_extension("mylang").map(|f| f.name()), Some("mylang"));
        assert_eq!(compiler.frontend_for_extension("two").map(|f| f.name()), Some("second"));
        assert!(compiler.frontend_for_extension("MY").is_none());
        assert!(compiler.frontend_for_extension(".my").is_none());
        assert!(compiler.frontend_for_extension("").is_none());
    }

    #[test]
    fn a_backend_is_found_by_name() {
        let compiler = Compiler::builder()
            .backend(WASM_BACKEND)
            .backend(OTHER_BACKEND)
            .build()
            .unwrap();
        assert_eq!(compiler.backend("other").map(|b| b.name()), Some("other"));
        assert!(compiler.backend("llvm").is_none());
        assert_eq!(names(compiler.backends().map(|b| b.name())), ["wasm", "other"]);
    }

    #[test]
    fn the_targets_of_a_compiler_are_each_backends_in_order_without_repeats() {
        let compiler = Compiler::builder()
            .backend(WASM_BACKEND)
            .backend(OTHER_BACKEND)
            .build()
            .unwrap();
        let targets: Vec<String> = compiler.targets().iter().map(ToString::to_string).collect();
        assert_eq!(
            targets,
            ["wasm32-unknown-unknown", "wasm32-wasip2", "x86_64-unknown-linux-gnu"]
        );
    }
}
