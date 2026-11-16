//! The command line of a compiler built on Runic.
//!
//! A [Driver] is the `clap` command of a [Compiler]: the driver's own flags, the flags
//! each of the compiler's frontends and backends declares through an
//! [OptionsDecl](runic_compile::OptionsDecl), and `-Z`, with which a user sets the unstable
//! settings of the driver and of the stages.
//!
//! The rules the command holds its stages to:
//!
//! * a backend's flags are prefixed with the backend's name, so that `allow_undefined` of the
//!   backend `wasm-cm` is `--wasm-cm-allow-undefined`; a frontend's are not, unless the builder is
//!   asked to prefix it ([DriverBuilder::prefix_frontend]);
//! * a stage's options are long flags that take at most one value each: no short flag, no
//!   positional argument, no required argument (nor a required group, nor a subcommand), no alias,
//!   and no `env`: nothing in a compilation reads the process's environment except through the
//!   request;
//! * a stage's options are a `clap` 4 `Args` type of its own: a request holds one value of a
//!   type, so two declarers cannot share one;
//! * no two flags have one spelling, and no two unstable settings do;
//! * a stage's unstable settings are never flags of the main command: they are reachable only as
//!   `-Z NAME[=VALUE]`.
//!
//! The command is checked when it is built, and each failure names what was declared and by whom.
//!
//! [Driver::run] runs a command line: it starts the compilation on a thread with a large stack,
//! renders what it printed and the diagnostics it reported to the streams it is given, and answers
//! with an exit code. [main] is a compiler's whole `main`.
//!
//! [Driver::request] reads a command line and answers with an [Invocation] (a compilation to run,
//! or text to print) or an [Exit] (help, or a usage error). The driver's own flags, the `-Z`
//! settings and the stages' flags are all read there, and each fills a field of the request or of
//! a stage's options and nothing else.
//!
//! The pieces the driver is built on are public, for a command line of another shape that runs a
//! compilation (a developer tool with its own flags): [StageFlags] adds the stages' flags and `-Z`
//! to any `clap` command and reads them back into a request, [worker] runs work on a thread with a
//! large stack and records its panic, and [render] renders a diagnostic as the driver does.
//!
//! `examples/toyc.rs` is a whole compiler built on the driver: a few lines that name its stages and
//! hand the compiler to [main_with]. The tests in `tests/toyc.rs` run it through [Driver::run].
#![feature(panic_can_unwind)]
#![deny(warnings)]

mod command;
pub mod render;
mod request;
mod run;
mod stages;
mod unstable;
pub mod worker;

use clap::{Args as _, Command};
use runic_compile::{Compiler, RUNIC_REVISION, RUNIC_VERSION};
use runic_ir::Report;

pub use self::{
    command::ColorChoice,
    request::{Exit, Invocation, RenderOptions},
    run::{EXIT_FAILURE, EXIT_INTERNAL, EXIT_USAGE, RunOptions, main, main_with},
    stages::{StageFlags, StageFlagsBuilder},
};
use self::{
    command::{Flags, PrintKind},
    stages::{check_prefixes, scratch},
    unstable::Unstable,
};

/// The name of a driver that is given none.
const DEFAULT_NAME: &str = "runic";

/// The command line of a compiler: the driver's flags, and those its frontends and backends
/// declare.
///
/// Built by [Driver::new] and [DriverBuilder::build].
#[derive(Clone)]
pub struct Driver<'c> {
    compiler: &'c Compiler,
    command: Command,
    stage_flags: StageFlags<'c>,
}

impl<'c> Driver<'c> {
    /// Starts the command line of `compiler`.
    #[allow(clippy::new_ret_no_self, reason = "a driver is made by its builder")]
    pub fn new(compiler: &'c Compiler) -> DriverBuilder<'c> {
        DriverBuilder {
            compiler,
            name: None,
            version: None,
            about: None,
            unstable: true,
            prefixed: Vec::new(),
        }
    }

    /// The compiler the command line is for.
    pub fn compiler(&self) -> &'c Compiler {
        self.compiler
    }

    /// The whole command, as `clap` holds it: for help, completions and tests.
    pub fn command(&self) -> &Command {
        &self.command
    }

    /// The flags of the compiler's stages the command carries.
    pub(crate) fn stage_flags(&self) -> &StageFlags<'c> {
        &self.stage_flags
    }

    /// The text `-Z help` prints: every unstable setting with its help, grouped by declarer. Empty
    /// when the driver does not allow unstable flags.
    pub fn unstable_help(&self) -> String {
        self.stage_flags.unstable_help()
    }
}

/// Builds a [Driver]; made by [Driver::new].
#[derive(Clone)]
pub struct DriverBuilder<'c> {
    compiler: &'c Compiler,
    name: Option<String>,
    version: Option<String>,
    about: Option<String>,
    unstable: bool,
    prefixed: Vec<String>,
}

impl<'c> DriverBuilder<'c> {
    /// The name of the program, as help and `--version` show it. The default is `runic`.
    pub fn name(self, name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..self
        }
    }

    /// The version of the program, as `--version` shows it. The default is
    /// [RUNIC_VERSION].
    pub fn version(self, version: impl Into<String>) -> Self {
        Self {
            version: Some(version.into()),
            ..self
        }
    }

    /// What the program is, in one sentence, as help shows it.
    pub fn about(self, about: impl Into<String>) -> Self {
        Self {
            about: Some(about.into()),
            ..self
        }
    }

    /// Whether the command accepts `-Z`. The default is `true`. When it is `false`, the stages'
    /// unstable declarations are still checked, so that allowing them later cannot fail.
    pub fn unstable(self, allowed: bool) -> Self {
        Self {
            unstable: allowed,
            ..self
        }
    }

    /// The named frontend's flags get its name as a prefix, as a backend's always do.
    pub fn prefix_frontend(mut self, frontend: &str) -> Self {
        self.prefixed.push(frontend.to_owned());
        self
    }

    /// Checks the stages' declarations against the rules of the command line and against each
    /// other, and builds the command.
    ///
    /// Fails, naming the stage and the field or the flag, when:
    ///
    /// * [prefix_frontend](Self::prefix_frontend) names a frontend the compiler does not have;
    /// * a stage declares an option, stable or unstable, that has a short flag, is positional or
    ///   required, has an action other than setting a value, appending one, a switch or a count,
    ///   takes more than one value per occurrence, has an alias or reads the environment, or when
    ///   its options have a required group or subcommands;
    /// * two declarers use one options type, or one stage uses a type as both its stable and its
    ///   unstable options;
    /// * two declarers give one spelling to a flag, or to an unstable setting. The driver is a
    ///   declarer, and `help` and `version` are its flags.
    pub fn build(self) -> Result<Driver<'c>, Report> {
        let Self {
            compiler,
            name,
            version,
            about,
            unstable,
            prefixed,
        } = self;

        check_prefixes(compiler, &prefixed, "the driver was")?;

        let program = name.clone().unwrap_or_else(|| DEFAULT_NAME.to_owned());
        let customized = name.is_some_and(|name| name != DEFAULT_NAME)
            || version.as_ref().is_some_and(|version| version != RUNIC_VERSION);
        let version = version.unwrap_or_else(|| RUNIC_VERSION.to_owned());
        let runic_line = format!("{RUNIC_VERSION} ({RUNIC_REVISION})");
        let long_version = if customized {
            format!("{version}\nrunic {runic_line}")
        } else {
            runic_line
        };

        let mut command = Flags::augment_args(
            Command::new(program.clone())
                .bin_name(program)
                .version(version)
                .long_version(long_version)
                .disable_help_subcommand(true),
        );
        if let Some(about) = about {
            command = command.about(about);
        }

        let mut stage_flags = StageFlags::new(compiler)
            .unstable(unstable)
            .own_unstable("driver", scratch(Unstable::augment_args));
        for frontend in &prefixed {
            stage_flags = stage_flags.prefix_frontend(frontend);
        }
        let stage_flags = stage_flags.build()?;
        let mut command = stage_flags.augment(command, "the driver")?;

        command.build();
        Ok(Driver {
            compiler,
            command,
            stage_flags,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use clap::{Arg, Args, error::ErrorKind};
    use runic_compile::{
        Artifact, Backend, CheckpointDecl, CheckpointId, Checkpoints, Flow, Frontend, Input,
        OptionsDecl, Session, Triple,
        testing::{AltVmBackend, CalcFrontend, StackVmBackend},
    };
    use runic_ir::{
        DataLayoutSpec, Endianness, PointerSpec, dialects::builtin::WorldRef, pass::OpPassManager,
    };

    use super::*;
    use crate::command::UNSTABLE_ARG;

    #[derive(Args, Default, Debug, Clone)]
    struct Loud {
        /// Be loud.
        #[arg(long)]
        loud: bool,
    }

    /// Another type of one flag, `--loud`.
    #[derive(Args, Default, Debug, Clone)]
    struct LoudToo {
        /// Be loud, too.
        #[arg(long)]
        loud: bool,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct YZ {
        #[arg(long)]
        y_z: bool,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct Target {
        #[arg(long)]
        target: bool,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct Help {
        #[arg(long)]
        help: bool,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct Short {
        #[arg(short = 'x', long)]
        extra: bool,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct Positional {
        #[arg()]
        source: String,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct Required {
        #[arg(long, required = true)]
        needed: String,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct Pair {
        #[arg(long, num_args = 2)]
        pair: Vec<String>,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct Aliased {
        #[arg(long, alias = "other")]
        aliased: bool,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct FromEnvironment {
        #[arg(long, env = "RUNIC_DRIVER_TEST_UNSET")]
        home: Option<String>,
    }

    #[derive(Args, Default, Debug, Clone)]
    #[group(required = true)]
    struct Grouped {
        #[arg(long)]
        left: bool,
        #[arg(long)]
        right: bool,
    }

    #[derive(clap::Subcommand, Debug, Clone)]
    enum Go {
        Now,
    }

    #[derive(Args, Default, Debug, Clone)]
    struct WithSubcommand {
        #[command(subcommand)]
        go: Option<Go>,
    }

    type Declare = fn() -> Option<OptionsDecl>;

    /// A frontend that declares what a test says and does nothing else.
    pub(crate) struct Fe(pub(crate) &'static str, pub(crate) fn() -> Option<OptionsDecl>);

    impl Frontend for Fe {
        fn name(&self) -> &'static str {
            self.0
        }

        fn extensions(&self) -> &'static [&'static str] {
            &[]
        }

        fn translate(
            &self,
            _session: &Session<'_>,
            _inputs: &[Input],
            _world: WorldRef,
            _checkpoints: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            Ok(Flow::Continue(()))
        }

        fn options(&self) -> Option<OptionsDecl> {
            (self.1)()
        }
    }

    /// A backend that declares what a test says and does nothing else.
    pub(crate) struct Be(&'static str, fn() -> Option<OptionsDecl>, &'static [CheckpointDecl]);

    fn render_nothing(
        _artifact: &Artifact,
        _session: &Session<'_>,
        _out: &mut dyn std::io::Write,
    ) -> Result<(), Report> {
        Ok(())
    }

    pub(crate) fn backend(name: &'static str, options: fn() -> Option<OptionsDecl>) -> Be {
        let id: &'static str = Box::leak(format!("{name}.out").into_boxed_str());
        let checkpoints = Box::leak(Box::new([CheckpointDecl {
            id: CheckpointId::new(id),
            aliases: &[],
            extension: "out",
            render: render_nothing,
        }]));
        Be(name, options, checkpoints)
    }

    impl Backend for Be {
        fn name(&self) -> &'static str {
            self.0
        }

        fn supports(&self, _triple: &Triple) -> bool {
            false
        }

        fn targets(&self) -> Vec<Triple> {
            Vec::new()
        }

        fn data_layout(&self, _triple: &Triple) -> Result<DataLayoutSpec, Report> {
            Ok(DataLayoutSpec::new(Endianness::Little, PointerSpec::sized(32).unwrap()).unwrap())
        }

        fn checkpoints(&self) -> &'static [CheckpointDecl] {
            self.2
        }

        fn add_passes(
            &self,
            _session: &Session<'_>,
            _passes: &mut OpPassManager,
        ) -> Result<(), Report> {
            Ok(())
        }

        fn emit(
            &self,
            _session: &Session<'_>,
            _world: WorldRef,
            _checkpoints: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            Ok(Flow::Continue(()))
        }

        fn options(&self) -> Option<OptionsDecl> {
            (self.1)()
        }
    }

    pub(crate) fn stable<T: Args + Default + Send + Sync + 'static>() -> Option<OptionsDecl> {
        Some(OptionsDecl::of::<T>())
    }

    fn unstable_only<T: Args + Default + Send + Sync + 'static>() -> Option<OptionsDecl> {
        Some(OptionsDecl::unstable_only::<T>())
    }

    pub(crate) fn toy() -> Compiler {
        Compiler::builder()
            .frontend(CalcFrontend)
            .frontend(runic_compile::IrFrontend)
            .backend(StackVmBackend)
            .backend(AltVmBackend)
            .build()
            .unwrap()
    }

    fn with(frontends: Vec<Fe>, backends: Vec<Be>) -> Compiler {
        let mut builder = Compiler::builder();
        for frontend in frontends {
            builder = builder.frontend(frontend);
        }
        for backend in backends {
            builder = builder.backend(backend);
        }
        builder.build().unwrap()
    }

    fn built(driver: &Driver<'_>) -> Command {
        let mut command = driver.command().clone();
        command.build();
        command
    }

    fn longs(command: &Command) -> Vec<String> {
        let mut longs: Vec<String> = command
            .get_arguments()
            .filter_map(|a| a.get_long().map(str::to_owned))
            .collect();
        longs.sort();
        longs
    }

    fn error_of(driver: Result<Driver<'_>, Report>) -> String {
        match driver {
            Ok(_) => panic!("the driver was built"),
            Err(report) => report.to_string(),
        }
    }

    #[test]
    fn the_main_command_has_exactly_the_documented_flags_and_short_flags() {
        let compiler = toy();
        let driver = Driver::new(&compiler).build().unwrap();
        let command = built(&driver);
        let mut expected: Vec<&str> = vec![
            "backend",
            "color",
            "debug",
            "emit",
            "extern",
            "help",
            "language",
            "name",
            "opt-level",
            "output-dir",
            "output-file",
            "print",
            "remap-path-prefix",
            "search-path",
            "stackvm-comments",
            "stop-after",
            "target",
            "version",
            "wrap-on-overflow",
        ];
        expected.sort();
        assert_eq!(longs(&command), expected);
        let mut shorts: Vec<char> = command.get_arguments().filter_map(Arg::get_short).collect();
        shorts.sort();
        assert_eq!(shorts, ['L', 'O', 'V', 'Z', 'g', 'h', 'o']);
    }

    #[test]
    fn a_backends_flag_is_prefixed_with_its_name_and_listed_under_its_heading() {
        let compiler = toy();
        let driver = Driver::new(&compiler).build().unwrap();
        let command = built(&driver);
        let comments = command.get_arguments().find(|a| a.get_long() == Some("stackvm-comments"));
        let comments = comments.expect("--stackvm-comments exists");
        assert_eq!(comments.get_help_heading(), Some("stackvm backend options"));
        assert_eq!(comments.get_id(), "backend:stackvm:comments");
        assert!(!command.get_arguments().any(|a| a.get_long() == Some("comments")));
        assert!(
            !command
                .get_arguments()
                .any(|a| a.get_long() == Some("stackvm-keep-subtractions"))
        );
    }

    #[test]
    fn a_frontends_flag_has_no_prefix_unless_the_builder_asks_for_one() {
        let compiler = toy();
        let plain = Driver::new(&compiler).build().unwrap();
        let command = built(&plain);
        let flag = command.get_arguments().find(|a| a.get_long() == Some("wrap-on-overflow"));
        assert_eq!(flag.unwrap().get_help_heading(), Some("calc frontend options"));

        let prefixed = Driver::new(&compiler).prefix_frontend("calc").build().unwrap();
        let command = built(&prefixed);
        let flag = command.get_arguments().find(|a| a.get_long() == Some("calc-wrap-on-overflow"));
        assert_eq!(flag.unwrap().get_help_heading(), Some("calc frontend options"));
        assert!(!command.get_arguments().any(|a| a.get_long() == Some("wrap-on-overflow")));
    }

    #[test]
    fn two_backends_that_declare_one_field_get_two_flags() {
        let compiler = with(
            Vec::new(),
            vec![backend("one", stable::<Loud>), backend("two", stable::<LoudToo>)],
        );
        let driver = Driver::new(&compiler).build().unwrap();
        let command = built(&driver);
        assert!(command.get_arguments().any(|a| a.get_long() == Some("one-loud")));
        assert!(command.get_arguments().any(|a| a.get_long() == Some("two-loud")));
    }

    #[test]
    fn a_flag_that_two_unprefixed_frontends_declare_is_an_error_naming_both() {
        let compiler = with(vec![Fe("a", stable::<Loud>), Fe("b", stable::<Loud>)], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the flag '--loud' is declared by the frontend 'a' and by the frontend 'b'"
        );
    }

    #[test]
    fn a_frontend_flag_named_like_a_driver_flag_is_an_error_naming_the_driver() {
        let compiler = with(vec![Fe("a", stable::<Target>)], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the flag '--target' is declared by the driver and by the frontend 'a'"
        );
        let compiler = with(vec![Fe("a", stable::<Help>)], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the flag '--help' is declared by the driver and by the frontend 'a'"
        );
    }

    #[test]
    fn a_backend_flag_that_spells_a_frontend_flag_is_an_error_naming_both() {
        #[derive(Args, Default, Debug, Clone)]
        struct Xyz {
            #[arg(long)]
            x_y_z: bool,
        }
        let compiler = with(vec![Fe("f", stable::<Xyz>)], vec![backend("x", stable::<YZ>)]);
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the flag '--x-y-z' is declared by the frontend 'f' and by the backend 'x'"
        );
    }

    #[test]
    fn the_same_three_collisions_in_the_unstable_namespace_are_errors() {
        let compiler =
            with(vec![Fe("a", unstable_only::<Loud>), Fe("b", unstable_only::<Loud>)], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the unstable setting '-Z loud' is declared by the frontend 'a' and by the frontend \
             'b'"
        );

        #[derive(Args, Default, Debug, Clone)]
        struct Generic {
            #[arg(long)]
            print_generic: bool,
        }
        let compiler = with(vec![Fe("a", unstable_only::<Generic>)], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the unstable setting '-Z print-generic' is declared by the driver and by the \
             frontend 'a'"
        );
        let compiler = with(vec![Fe("a", unstable_only::<Help>)], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the unstable setting '-Z help' is declared by the driver and by the frontend 'a'"
        );

        #[derive(Args, Default, Debug, Clone)]
        struct Xyz {
            #[arg(long)]
            x_y_z: bool,
        }
        let compiler =
            with(vec![Fe("f", unstable_only::<Xyz>)], vec![backend("x", unstable_only::<YZ>)]);
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the unstable setting '-Z x-y-z' is declared by the frontend 'f' and by the backend \
             'x'"
        );
    }

    #[test]
    fn a_stable_and_an_unstable_setting_may_share_a_spelling() {
        let compiler = with(
            vec![Fe("a", || Some(OptionsDecl::of::<Loud>().unstable::<LoudToo>()))],
            Vec::new(),
        );
        Driver::new(&compiler).build().unwrap();
    }

    #[test]
    fn two_backends_that_declare_one_options_type_are_an_error_naming_the_type_and_both() {
        let compiler =
            with(Vec::new(), vec![backend("one", stable::<Loud>), backend("two", stable::<Loud>)]);
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the options type 'runic_driver::tests::Loud' is declared by the backend 'one' and by \
             the backend 'two': each declarer needs a type of its own, since a request holds one \
             value of a type"
        );
    }

    #[test]
    fn a_frontend_and_a_backend_that_declare_one_options_type_are_an_error_naming_both() {
        let compiler = with(vec![Fe("f", stable::<Loud>)], vec![backend("b", stable::<Loud>)]);
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the options type 'runic_driver::tests::Loud' is declared by the frontend 'f' and by \
             the backend 'b': each declarer needs a type of its own, since a request holds one \
             value of a type"
        );
    }

    #[test]
    fn a_stage_that_declares_one_type_as_stable_and_unstable_is_an_error_naming_the_type() {
        let compiler =
            with(vec![Fe("a", || Some(OptionsDecl::of::<Loud>().unstable::<Loud>()))], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the options type 'runic_driver::tests::Loud' is declared by the frontend 'a' as both \
             its stable and its unstable options: each needs a type of its own, since a request \
             holds one value of a type"
        );
    }

    #[test]
    fn a_stage_option_that_reads_the_environment_is_an_error_naming_the_stage_and_the_field() {
        let compiler = with(vec![Fe("a", stable::<FromEnvironment>)], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the frontend 'a' declares the option 'home', which reads the environment variable \
             'RUNIC_DRIVER_TEST_UNSET'; options of a stage never read the environment, since \
             nothing in a compilation reads it except through the request"
        );
    }

    #[test]
    fn a_stage_whose_options_have_a_required_group_or_subcommands_is_an_error_naming_the_stage() {
        let compiler = with(vec![Fe("a", stable::<Grouped>)], Vec::new());
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the frontend 'a' declares options with the required group 'Grouped'; options of a \
             stage are never required"
        );
        let compiler = with(Vec::new(), vec![backend("b", stable::<WithSubcommand>)]);
        assert_eq!(
            error_of(Driver::new(&compiler).build()),
            "the backend 'b' declares options with subcommands; options of a stage are flags"
        );
    }

    #[test]
    fn a_stage_option_that_breaks_a_rule_is_an_error_naming_the_stage_and_the_field() {
        let cases: [(Declare, &str); 5] = [
            (
                stable::<Short>,
                "the frontend 'a' declares the option 'extra', which has the short flag '-x'; \
                 options of a stage have long flags only",
            ),
            (
                stable::<Positional>,
                "the frontend 'a' declares the option 'source', which is positional; options of a \
                 stage are flags",
            ),
            (
                stable::<Required>,
                "the frontend 'a' declares the option 'needed', which is required; options of a \
                 stage are never required",
            ),
            (
                stable::<Pair>,
                "the frontend 'a' declares the option 'pair', which takes more than one value per \
                 occurrence; it takes at most one",
            ),
            (
                stable::<Aliased>,
                "the frontend 'a' declares the option 'aliased', which has an alias, which a \
                 stage's prefix cannot be applied to",
            ),
        ];
        for (options, message) in cases {
            let compiler = with(vec![Fe("a", options)], Vec::new());
            assert_eq!(error_of(Driver::new(&compiler).build()), message);
        }
    }

    #[test]
    fn a_rule_is_checked_for_unstable_options_and_for_backends_too() {
        let compiler = with(Vec::new(), vec![backend("b", unstable_only::<Short>)]);
        assert_eq!(
            error_of(Driver::new(&compiler).unstable(false).build()),
            "the backend 'b' declares the unstable option 'extra', which has the short flag '-x'; \
             options of a stage have long flags only"
        );
    }

    #[test]
    fn prefixing_a_frontend_the_compiler_lacks_is_an_error_listing_the_frontends() {
        let compiler = toy();
        assert_eq!(
            error_of(Driver::new(&compiler).prefix_frontend("nope").build()),
            "the driver was asked to prefix the flags of the frontend 'nope', but the compiler \
             has no such frontend; its frontends are calc, runic"
        );
    }

    #[test]
    fn the_command_accepts_dash_z_only_when_unstable_flags_are_allowed() {
        let compiler = toy();
        let allowed = Driver::new(&compiler).build().unwrap();
        assert!(allowed.stage_flags().unstable_allowed());
        assert!(built(&allowed).get_arguments().any(|a| a.get_short() == Some('Z')));
        let refused = Driver::new(&compiler).unstable(false).build().unwrap();
        assert!(!refused.stage_flags().unstable_allowed());
        assert!(!built(&refused).get_arguments().any(|a| a.get_short() == Some('Z')));
        assert_eq!(refused.unstable_help(), "");
    }

    #[test]
    fn the_unstable_help_lists_the_drivers_settings_and_each_stages_under_its_heading() {
        let compiler = toy();
        let driver = Driver::new(&compiler).build().unwrap();
        let help = driver.unstable_help();
        assert!(help.starts_with("Unstable options (-Z NAME[=VALUE]):\n"), "{help}");
        for name in [
            "pass-pipeline=PIPELINE",
            "dump-pass-pipeline",
            "print-ir-before=PASSES",
            "print-ir-after=PASSES",
            "print-generic",
            "print-locations",
            "no-verify",
            "verify-each[=BOOL]",
        ] {
            assert!(help.contains(&format!("  {name}")), "{name} is missing from\n{help}");
        }
        let driver_at = help.find("\ndriver\n").unwrap();
        let stackvm_at = help.find("\nstackvm backend\n").unwrap();
        let keep_at = help.find("stackvm-keep-subtractions").unwrap();
        assert!(driver_at < stackvm_at && stackvm_at < keep_at, "{help}");
        assert!(!help.contains("calc frontend"), "{help}");
    }

    fn rendered(driver: &Driver<'_>, flag: &str) -> String {
        let error = driver.command().clone().try_get_matches_from(["x", flag]).unwrap_err();
        error.to_string()
    }

    #[test]
    fn help_shows_the_programs_name_and_the_stage_headings() {
        let compiler = toy();
        let driver = Driver::new(&compiler).name("mycc").about("Compiles things").build().unwrap();
        let help = rendered(&driver, "--help");
        assert!(help.contains("Usage: mycc"), "{help}");
        assert!(help.contains("Compiles things"), "{help}");
        assert!(help.contains("calc frontend options:"), "{help}");
        assert!(help.contains("stackvm backend options:"), "{help}");
        assert!(help.contains("--stackvm-comments"), "{help}");
    }

    #[test]
    fn version_has_two_lines_for_a_named_program_and_one_for_the_default() {
        let compiler = toy();
        let named = Driver::new(&compiler).name("mycc").version("1.2.3").build().unwrap();
        assert_eq!(
            rendered(&named, "--version"),
            format!("mycc 1.2.3\nrunic {RUNIC_VERSION} ({RUNIC_REVISION})\n")
        );
        let default = Driver::new(&compiler).build().unwrap();
        assert_eq!(
            rendered(&default, "--version"),
            format!("runic {RUNIC_VERSION} ({RUNIC_REVISION})\n")
        );
    }

    #[test]
    fn the_command_passes_clap_s_own_consistency_check() {
        let compiler = toy();
        Driver::new(&compiler).build().unwrap().command().clone().debug_assert();
        Driver::new(&compiler)
            .unstable(false)
            .build()
            .unwrap()
            .command()
            .clone()
            .debug_assert();
    }

    #[test]
    fn a_bad_value_of_a_typed_flag_is_a_usage_error_that_names_the_flag() {
        let compiler = toy();
        let driver = Driver::new(&compiler).build().unwrap();
        let error = driver
            .command()
            .clone()
            .try_get_matches_from(["x", "--opt-level=9"])
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
        assert!(error.to_string().contains("--opt-level"), "{error}");
        let matches = driver
            .command()
            .clone()
            .try_get_matches_from([
                "x",
                "-O2",
                "-g",
                "--stackvm-comments",
                "-Zfoo=1",
                "-",
                "a.calc",
            ])
            .unwrap();
        let flags = <Flags as clap::FromArgMatches>::from_arg_matches(&matches).unwrap();
        assert_eq!(
            flags.input,
            [std::path::PathBuf::from("-"), std::path::PathBuf::from("a.calc")]
        );
        assert_eq!(flags.debug, Some(runic_compile::DebugInfo::Full));
        assert_eq!(flags.opt_level, Some(runic_compile::OptLevel::Balanced));
        assert_eq!(
            matches.get_many::<String>(UNSTABLE_ARG).unwrap().collect::<Vec<_>>(),
            ["foo=1"]
        );
    }
}
