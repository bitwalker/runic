#[cfg(feature = "std")]
use alloc::sync::Arc;
#[cfg(feature = "std")]
use alloc::{borrow::ToOwned, format, string::ToString, vec};
use alloc::{boxed::Box, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::ffi::OsString;

#[cfg(feature = "std")]
use clap::{Parser, builder::ArgPredicate};
use runic_session::{
    ColorChoice, DebugInfo, IrFilter, LinkLibrary, OptLevel, Options, OutputFile, OutputTypeSpec,
    OutputTypes, PanicStrategy, PathBuf, RemapPathPrefix, Verbosity, Warnings,
    add_target_link_libraries,
};
#[cfg(feature = "std")]
use runic_session::{InputFile, Session, diagnostics::Emitter};

/// Compile a program from source to target
#[derive(Debug, Clone)]
#[cfg_attr(feature = "std", derive(Parser))]
#[cfg_attr(feature = "std", command(name = "runic"))]
pub struct Compiler {
    /// Write all intermediate compiler artifacts to `<target_dir>/<profile>`
    ///
    /// Defaults to a directory named `target/runic` in the current working directory
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            value_name = "DIR",
            env = "RUNIC_TARGET_DIR",
            default_value = "target/runic",
            help_heading = "Output"
        )
    )]
    pub target_dir: PathBuf,
    /// The working directory for the compiler
    ///
    /// By default this will be the working directory the compiler is executed from
    #[cfg_attr(
        feature = "std",
        arg(long, value_name = "DIR", help_heading = "Output")
    )]
    pub working_dir: Option<PathBuf>,
    /// The path to the root directory of the current target toolchain.
    ///
    /// The path given should be a directory whose layout matches a typical compiler toolchain, i.e.
    /// it has `bin` and `lib` directories containing toolchain components and libraries.
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            value_name = "DIR",
            env = "RUNIC_SYSROOT",
            help_heading = "Compiler"
        )
    )]
    pub sysroot: Option<PathBuf>,
    /// Write compiled output to compiler-chosen filename in `<dir>`
    ///
    /// If not specified, the output directory is determined from `--target-dir`
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            short = 'O',
            value_name = "DIR",
            env = "RUNIC_OUT_DIR",
            help_heading = "Output"
        )
    )]
    pub output_dir: Option<PathBuf>,
    /// Write compiled output to `<filename>`
    ///
    /// This overrides the default output location given by `--target-dir`.
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            short = 'o',
            value_name = "FILENAME",
            conflicts_with("output_dir"),
            help_heading = "Output"
        )
    )]
    pub output_file: Option<PathBuf>,
    /// Write output to stdout
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            conflicts_with("output_file"),
            conflicts_with("output_dir"),
            help_heading = "Output"
        )
    )]
    pub stdout: bool,
    /// Specify the target triple you wish to compile for, e.g. `wasm32-unknown-unknown`
    ///
    /// By default, if this is not specified, then the target is inferred from the host environment.
    #[cfg_attr(feature = "std", arg(
        long,
        value_name = "TARGET",
        default_value = None,
        help_heading = "Compiler"
    ))]
    pub target: Option<String>,
    /// Specify what type and level of informational output to emit
    #[cfg_attr(feature = "std", arg(
        long = "verbose",
        short = 'v',
        value_enum,
        value_name = "LEVEL",
        default_value_t = Verbosity::Info,
        default_missing_value = "debug",
        num_args(0..=1),
        help_heading = "Diagnostics"
    ))]
    pub verbosity: Verbosity,
    /// Specify how warnings should be treated by the compiler.
    #[cfg_attr(feature = "std", arg(
        long,
        short = 'W',
        value_enum,
        value_name = "LEVEL",
        default_value_t = Warnings::All,
        default_missing_value = "all",
        num_args(0..=1),
        help_heading = "Diagnostics"
    ))]
    pub warn: Warnings,
    /// Whether, and how, to color terminal output
    #[cfg_attr(feature = "std", arg(
        long,
        value_enum,
        default_value_t = ColorChoice::Auto,
        default_missing_value = "auto",
        num_args(0..=1),
        help_heading = "Diagnostics"
    ))]
    pub color: ColorChoice,
    /// Specify the function to call as the entrypoint for the program.
    ///
    /// Defaults to `main` if not otherwise specified, and an executable artifact is requested.
    #[cfg_attr(feature = "std", arg(long, help_heading = "Compiler", hide(true)))]
    pub entrypoint: Option<String>,
    /// Tells the compiler to produce an executable artifact
    ///
    /// Implied by `--entrypoint`, defaults to false.
    #[cfg_attr(feature = "std", arg(
        long = "exe",
        default_value_t = false,
        default_value_ifs([
            // Setting the entrypoint implies building an executable in all other cases
            ("entrypoint", ArgPredicate::IsPresent, Some("true")),
        ]),
        help_heading = "Linker"
    ))]
    pub is_program: bool,
    /// Tells the compiler to produce a library artifact
    ///
    /// Defaults to true, unless `--executable` or `--entrypoint` are specified.
    #[cfg_attr(feature = "std", arg(
        long = "lib",
        conflicts_with("is_program"),
        conflicts_with("entrypoint"),
        default_value_t = true,
        default_value_ifs([
            // When an entrypoint is specified, always set the default to false
            ("is_program", ArgPredicate::IsPresent, Some("false")),
            // When an entrypoint is specified, always set the default to false
            ("entrypoint", ArgPredicate::IsPresent, Some("false")),
        ]),
        help_heading = "Linker"
    ))]
    pub is_library: bool,
    /// Specify one or more search paths for link libraries requested via `-l`
    #[cfg_attr(
        feature = "std",
        arg(
            long = "search-path",
            short = 'L',
            value_name = "PATH",
            help_heading = "Linker"
        )
    )]
    pub search_path: Vec<PathBuf>,
    /// Link compiled artifacts to the specified library NAME.
    ///
    /// The optional KIND can be provided to indicate what type of library it is.
    ///
    /// NAME must either be an absolute path (with extension when applicable), or
    /// a library namespace (no extension). The former will be used as the path
    /// to load the library, without looking for it in the library search paths,
    /// while the latter will be located in the search path based on its KIND.
    ///
    /// See below for valid KINDs:
    #[cfg_attr(
        feature = "std",
        arg(
            long = "link-library",
            short = 'l',
            value_name = "[KIND=]NAME",
            value_delimiter = ',',
            next_line_help(true),
            help_heading = "Linker"
        )
    )]
    pub link_libraries: Vec<LinkLibrary>,
    /// Specify one or more output types for the compiler to emit
    ///
    /// The format for SPEC is `KIND[=PATH]`. You can specify multiple items at
    /// once by separating each SPEC with a comma, you can also pass this flag
    /// multiple times.
    ///
    /// PATH must be a directory in which to place the outputs, or `-` for stdout.
    #[cfg_attr(
        feature = "std",
        arg(
            long = "emit",
            value_name = "SPEC",
            value_delimiter = ',',
            env = "RUNIC_EMIT",
            next_line_help(true),
            help_heading = "Output"
        )
    )]
    pub output_types: Vec<OutputTypeSpec>,
    /// Stop compilation after CHECKPOINT, rather than building an artifact
    ///
    /// CHECKPOINT is either an alias — `parse`, `analyze`, `transform`, `lower`, `assemble` —
    /// or a fully-qualified checkpoint id such as `hir.initial`.
    ///
    /// Which names are accepted depends on the input: each frontend declares its own route,
    /// and a name that route does not reach is reported along with the ones it does. A
    /// Miden Assembly input, for instance, has no `transform` phase to stop after.
    ///
    /// This is the general form of the `-C` stop flags (`-Cparse-only`, `-Canalyze-only`,
    /// `-Clink-only`); giving both is an error rather than one silently winning.
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            value_name = "CHECKPOINT",
            env = "RUNIC_STOP_AFTER",
            next_line_help(true),
            help_heading = "Output"
        )
    )]
    pub stop_after: Option<String>,
    /// Specify what level of debug information to emit in compilation artifacts
    #[cfg_attr(feature = "std", arg(
        long,
        value_enum,
        value_name = "LEVEL",
        next_line_help(true),
        default_value_t = DebugInfo::Full,
        default_missing_value = "full",
        num_args(0..=1),
        help_heading = "Output"
    ))]
    pub debug: DebugInfo,
    /// Specify what type, and to what degree, of optimizations to apply to code during
    /// compilation.
    #[cfg_attr(feature = "std", arg(
        long = "optimize",
        value_enum,
        value_name = "LEVEL",
        next_line_help(true),
        default_value_t = OptLevel::None,
        default_missing_value = "balanced",
        num_args(0..=1),
        help_heading = "Output"
    ))]
    pub opt_level: OptLevel,
    /// Set a codegen option
    ///
    /// Use `-C help` to print available options
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            short = 'C',
            value_name = "OPT[=VALUE]",
            help_heading = "Compiler"
        )
    )]
    pub codegen: Vec<String>,
    /// Set an unstable compiler option
    ///
    /// Use `-Z help` to print available options
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            short = 'Z',
            value_name = "OPT[=VALUE]",
            help_heading = "Compiler"
        )
    )]
    pub unstable: Vec<String>,
    /// Specify path prefixes to remap for any file paths encoded in debug info
    #[cfg_attr(
        feature = "std",
        arg(
            long = "remap-path-prefix",
            action = clap::ArgAction::Append,
            value_name = "PATH",
            value_delimiter = ',',
            help_heading = "Debugging",
            value_parser(runic_session::RemapPathPrefixParser)
        )
    )]
    pub remap_path_prefixes: Vec<RemapPathPrefix>,
}

#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "std", derive(Parser))]
#[cfg_attr(feature = "std", command(name = "-C"))]
pub struct CodegenOptions {
    /// Tell the compiler to exit after it has parsed the inputs
    #[cfg_attr(feature = "std", arg(
        long,
        conflicts_with_all(["analyze_only", "link_only"]),
        default_value_t = false,
    ))]
    pub parse_only: bool,
    /// Tell the compiler to exit after it has performed semantic analysis on the inputs
    #[cfg_attr(feature = "std", arg(
        long,
        conflicts_with_all(["parse_only", "link_only"]),
        default_value_t = false,
    ))]
    pub analyze_only: bool,
    /// Tell the compiler to exit after linking the inputs, without generating Miden Assembly
    #[cfg_attr(feature = "std", arg(
        long,
        conflicts_with_all(["no_link"]),
        default_value_t = false,
    ))]
    pub link_only: bool,
    /// Tell the compiler to generate Miden Assembly from the inputs without linking them
    #[cfg_attr(feature = "std", arg(long, default_value_t = false))]
    pub no_link: bool,
    /// The Rust panic strategy: trap immediately or invoke the crate's panic handler
    ///
    /// Direct rustc builds ignore `immediate-abort` with a warning because their prebuilt
    /// standard libraries are incompatible with that strategy.
    #[cfg_attr(
        feature = "std",
        arg(
            long = "panic",
            value_enum,
            value_name = "STRATEGY",
            default_value = "immediate-abort",
        )
    )]
    pub panic_strategy: PanicStrategy,
}

#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "std", derive(Parser))]
#[cfg_attr(feature = "std", command(name = "-Z"))]
pub struct UnstableOptions {
    /// Run the experimental linter prior to artifact emission
    #[cfg_attr(
        feature = "std",
        arg(long, default_value_t = false, help_heading = "Analysis")
    )]
    pub lint: bool,
    /// Print the CFG after each IR pass is applied
    #[cfg_attr(
        feature = "std",
        arg(long, default_value_t = false, help_heading = "Passes")
    )]
    pub print_cfg_after_all: bool,
    /// Print the CFG after running a specific IR pass
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            value_name = "PASS",
            value_delimiter = ',',
            help_heading = "Passes"
        )
    )]
    pub print_cfg_after_pass: Vec<String>,
    /// Print the IR before each compiler stage.
    ///
    /// The available stages are:
    ///
    /// * `link`     - performs initial translation to IR, and links dependencies into the graph
    ///
    /// * `rewrite`  - performs rewrites/optimizations on the initial unmodified IR
    ///
    /// * `codegen`  - performs lowering the IR to the backend representation
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            value_name = "STAGE",
            value_delimiter = ',',
            next_line_help(true),
            help_heading = "Passes"
        )
    )]
    pub print_ir_before_stage: Vec<String>,
    /// Print the IR after each pass is applied
    #[cfg_attr(
        feature = "std",
        arg(long, default_value_t = false, help_heading = "Passes")
    )]
    pub print_ir_after_all: bool,
    /// Print the IR after running a specific pass
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            value_name = "PASS",
            value_delimiter = ',',
            help_heading = "Passes"
        )
    )]
    pub print_ir_after_pass: Vec<String>,
    /// Only print the IR if the pass modified the IR structure. If this flag is set, and no IR
    /// filter flag is; then the default behavior is to print the IR after every pass.
    #[cfg_attr(
        feature = "std",
        arg(long, default_value_t = false, help_heading = "Passes")
    )]
    pub print_ir_after_modified: bool,
    /// Only print IR that matches the given filter.
    ///
    /// The syntax for filters are as follows:
    ///
    /// * `any` (default)         - matches any operation
    ///
    /// * `symbol:*`              - matches any symbol
    ///
    /// * `symbol:<pattern>`      - matches any symbol whose name contains `<pattern>`
    ///
    /// * `op:<dialect>.<opcode>` - matches any instance of the given operation name
    #[cfg_attr(
        feature = "std",
        arg(
            long,
            action = clap::ArgAction::Append,
            value_name = "FILTER",
            value_delimiter = ',',
            next_line_help(true),
            help_heading = "Passes"
        )
    )]
    pub print_ir_filter: Vec<IrFilter>,
    /// Print source location information in IR output
    ///
    /// When enabled, IR output will include #loc() annotations showing the source file, line, and
    /// column for each operation.
    #[cfg_attr(
        feature = "std",
        arg(long, default_value_t = false, help_heading = "Printers")
    )]
    pub print_source_locations: bool,
}

impl CodegenOptions {
    #[cfg(feature = "std")]
    fn parse_argv(argv: Vec<String>) -> Self {
        let command = <CodegenOptions as clap::CommandFactory>::command()
            .no_binary_name(true)
            .arg_required_else_help(false)
            .help_template(
                "\
Available codegen options:

Usage: runic -C <opt>

{all-args}{after-help}

NOTE: When specifying these options, strip the leading '--'",
            );

        let argv = if argv.iter().any(|arg| matches!(arg.as_str(), "--help" | "-h" | "help")) {
            vec!["--help".to_string()]
        } else {
            argv.into_iter()
                .flat_map(|arg| match arg.split_once('=') {
                    None => vec![format!("--{arg}")],
                    Some((opt, value)) => {
                        vec![format!("--{opt}"), value.to_string()]
                    }
                })
                .collect::<Vec<_>>()
        };

        let mut matches = command.try_get_matches_from(argv).unwrap_or_else(|err| err.exit());
        <CodegenOptions as clap::FromArgMatches>::from_arg_matches_mut(&mut matches)
            .map_err(format_error::<CodegenOptions>)
            .unwrap_or_else(|err| err.exit())
    }

    #[cfg(not(feature = "std"))]
    fn parse_argv(_argv: Vec<String>) -> Self {
        Self::default()
    }
}

impl UnstableOptions {
    #[cfg(feature = "std")]
    fn parse_argv(argv: Vec<String>) -> Self {
        let command = <UnstableOptions as clap::CommandFactory>::command()
            .no_binary_name(true)
            .arg_required_else_help(false)
            .help_template(
                "\
Available unstable options:

Usage: runic -Z <opt>

{all-args}{after-help}

NOTE: When specifying these options, strip the leading '--'",
            );

        let argv = if argv.iter().any(|arg| matches!(arg.as_str(), "--help" | "-h" | "help")) {
            vec!["--help".to_string()]
        } else {
            argv.into_iter()
                .flat_map(|arg| match arg.split_once('=') {
                    None => vec![format!("--{arg}")],
                    Some((opt, value)) => {
                        vec![format!("--{opt}"), value.to_string()]
                    }
                })
                .collect::<Vec<_>>()
        };

        let mut matches = command.try_get_matches_from(argv).unwrap_or_else(|err| err.exit());
        <UnstableOptions as clap::FromArgMatches>::from_arg_matches_mut(&mut matches)
            .map_err(format_error::<UnstableOptions>)
            .unwrap_or_else(|err| err.exit())
    }

    #[cfg(not(feature = "std"))]
    fn parse_argv(_argv: Vec<String>) -> Self {
        Self::default()
    }
}

impl Compiler {
    /// Parse command-line arguments into compiler [Options].
    ///
    /// Returns the parsed options or an error if parsing failed.
    ///
    /// Used by embedders and tests that build a session from an argument vector without going
    /// through the `midenc` command line.
    #[cfg(feature = "std")]
    pub fn try_parse_from<I, T>(cwd: PathBuf, iter: I) -> Result<Box<Options>, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        let argv = [OsString::from("midenc")]
            .into_iter()
            .chain(iter.into_iter().map(|arg| arg.into()));
        let command = <Self as clap::CommandFactory>::command();
        let command = runic_session::flags::register_flags(command);
        let mut matches = command.try_get_matches_from(argv)?;
        let compile_matches = matches.clone();

        let opts = <Self as clap::FromArgMatches>::from_arg_matches_mut(&mut matches)
            .map_err(format_error::<Self>)?;

        let mut opts = opts.into_options(cwd);
        opts.set_extra_flags(compile_matches.into());
        Ok(opts)
    }

    /// Construct a [Compiler] programatically
    #[cfg(feature = "std")]
    pub fn new_session<A, S>(
        cwd: PathBuf,
        input: Option<InputFile>,
        emitter: Option<Arc<dyn Emitter>>,
        argv: A,
    ) -> Session
    where
        A: IntoIterator<Item = S>,
        S: Into<std::ffi::OsString> + Clone,
    {
        let opts = Self::try_parse_from(cwd, argv).unwrap_or_else(|err| err.exit());
        Self::into_session(opts, input, emitter)
    }

    pub fn into_options(self, cwd: PathBuf) -> Box<Options> {
        let Self {
            target_dir,
            working_dir,
            sysroot,
            output_dir,
            output_file,
            stdout,
            target,
            verbosity,
            warn,
            color,
            entrypoint,
            is_program,
            is_library,
            search_path,
            mut link_libraries,
            output_types,
            stop_after,
            debug,
            opt_level,
            codegen,
            unstable,
            remap_path_prefixes,
        } = self;
        let CodegenOptions {
            parse_only,
            analyze_only,
            link_only,
            no_link,
            panic_strategy,
        } = CodegenOptions::parse_argv(codegen);
        let UnstableOptions {
            lint,
            print_cfg_after_all,
            print_cfg_after_pass,
            print_ir_before_stage,
            print_ir_after_all,
            print_ir_after_pass,
            print_ir_after_modified,
            print_ir_filter,
            print_source_locations,
        } = UnstableOptions::parse_argv(unstable);

        // Determine if a specific output file has been requested
        let output_file = match output_file {
            Some(path) => Some(OutputFile::Real(path)),
            None if stdout => Some(OutputFile::Stdout),
            None => None,
        };

        let cwd = working_dir.unwrap_or(cwd);

        // Establish --target-dir
        let target_dir = if target_dir.is_absolute() {
            target_dir
        } else {
            cwd.join(&target_dir)
        };

        // Initialize output types
        let output_types = OutputTypes::new(output_types).unwrap_or_else(|err| err.exit());

        // Consolidate all compiler options
        let mut options = Box::new(Options::new(
            target.clone(),
            target_type,
            cwd,
            target_dir,
            output_dir,
            sysroot,
        ))
        .with_color(color)
        .with_verbosity(verbosity)
        .with_warnings(warn)
        .with_debug_info(debug)
        .with_optimization(opt_level)
        .with_output_types(output_types, output_file);
        options.target = target;
        options.profile = profile;
        options.manifest_path = manifest_path;
        options.midenup_home = midenup_home;
        options.toolchain = toolchain;
        options.search_paths.extend(search_path);
        add_target_link_libraries(&mut link_libraries, options.target_requires_protocol());
        options.link_libraries = link_libraries;
        options.entrypoint = entrypoint;
        options.workspace = workspace;
        options.packages = package;
        options.cargo_locked = locked;
        options.cargo_offline = offline;
        options.stop_after = stop_after;
        options.parse_only = parse_only;
        options.analyze_only = analyze_only;
        options.link_only = link_only;
        options.no_link = no_link;
        options.panic_strategy = panic_strategy;
        options.lint = lint;
        options.cargo_frontmatter = cargo_frontmatter;
        options.print_cfg_after_all = print_cfg_after_all;
        options.print_cfg_after_pass = print_cfg_after_pass;
        options.print_ir_before_stage = print_ir_before_stage;
        options.print_ir_after_all = print_ir_after_all;
        options.print_ir_after_pass = print_ir_after_pass;
        options.print_ir_after_modified = print_ir_after_modified;
        options.print_ir_filters = print_ir_filter;
        options.print_hir_source_locations = print_hir_source_locations;
        options.remap_path_prefixes = remap_path_prefixes;

        #[cfg(feature = "std")]
        if options.remap_path_prefixes.is_empty() {
            options.remap_path_prefixes.push(RemapPathPrefix {
                from: options.current_dir.clone().into_boxed_path(),
                to: None,
            });
        }

        options
    }

    /// Use this configuration to obtain a [Session] used for compilation
    #[cfg(feature = "std")]
    fn into_session(
        options: Box<Options>,
        input: Option<InputFile>,
        emitter: Option<Arc<dyn Emitter>>,
    ) -> Session {
        let input = options.resolve_input(input).unwrap_or_else(|err| {
            let cmd = <Compiler as clap::CommandFactory>::command();
            let mut err =
                clap::Error::raw(clap::error::ErrorKind::ValueValidation, err).with_cmd(&cmd);
            err.insert(
                clap::error::ContextKind::InvalidArg,
                clap::error::ContextValue::String("INPUT".to_string()),
            );
            err.exit();
        });

        log::trace!(target: "driver", "current working directory = {}", options.current_dir.display());

        match options.into_session(input, emitter, None) {
            Ok(session) => session,
            Err(err) => {
                let cmd = <Compiler as clap::CommandFactory>::command();
                let err =
                    clap::Error::raw(clap::error::ErrorKind::ValueValidation, err).with_cmd(&cmd);
                err.exit();
            }
        }
    }
}

#[cfg(feature = "std")]
fn format_error<I: clap::CommandFactory>(err: clap::Error) -> clap::Error {
    let mut cmd = I::command();
    err.format(&mut cmd)
}

#[cfg(feature = "std")]
#[derive(Clone)]
struct TargetTypeValueParser;

#[cfg(feature = "std")]
impl clap::builder::TypedValueParser for TargetTypeValueParser {
    type Value = TargetType;

    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let value = value.to_string_lossy();
        value.parse::<TargetType>().map_err(|err| {
            let mut err = clap::Error::raw(clap::error::ErrorKind::ValueValidation, err);
            if let Some(arg) = arg {
                err.insert(
                    clap::error::ContextKind::InvalidArg,
                    clap::error::ContextValue::String(arg.to_string()),
                );
            }
            err
        })
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Parse `args` as a command line, as [`Compiler::try_parse_from`] does.
    fn options(args: &[&str]) -> Box<Options> {
        Compiler::try_parse_from(PathBuf::from("/tmp"), args)
            .unwrap_or_else(|err| panic!("`midenc {}` should parse: {err}", args.join(" ")))
    }

    #[test]
    fn panic_strategy_defaults_to_immediate_abort() {
        assert_eq!(options(&[]).panic_strategy, PanicStrategy::ImmediateAbort);
        assert_eq!(Options::default().panic_strategy, PanicStrategy::ImmediateAbort);
        assert_eq!(CodegenOptions::default().panic_strategy, PanicStrategy::ImmediateAbort);
    }

    #[test]
    fn panic_strategy_reaches_the_options() {
        assert_eq!(options(&["-C", "panic=abort"]).panic_strategy, PanicStrategy::Abort);
        assert_eq!(
            options(&["-Cpanic=immediate-abort"]).panic_strategy,
            PanicStrategy::ImmediateAbort
        );
    }

    #[test]
    fn panic_strategy_rejects_unsupported_values() {
        let err = CodegenOptions::try_parse_from(["-C", "--panic", "unwind"])
            .expect_err("unwinding is not a supported panic strategy");
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidValue);
    }

    /// `--stop-after` reaches [`Options`], which is the only way the pipeline can see it.
    ///
    /// The value is carried through uninterpreted: which names are valid depends on the route
    /// the input selects, so `hir.initial` is accepted here and rejected later by a frontend
    /// whose route lacks it. Parsing must not pre-empt that.
    #[test]
    fn stop_after_reaches_the_options() {
        for value in ["parse", "analyze", "transform", "lower", "assemble", "hir.initial"] {
            assert_eq!(
                options(&["--stop-after", value]).stop_after.as_deref(),
                Some(value),
                "`--stop-after={value}` must reach the options unchanged"
            );
        }
        assert_eq!(
            options(&["--stop-after=parse"]).stop_after.as_deref(),
            Some("parse"),
            "the `=` form must parse too"
        );
    }

    /// Absent the flag there is no stop point, which is what makes a full build the default.
    #[test]
    fn no_stop_after_means_no_cap() {
        assert_eq!(options(&[]).stop_after, None);
    }

    /// `--manifest-path` reaches [`Options`] as given — relative to the directory the compiler is
    /// run from, never joined onto `--working-dir` — so it means what it does in a shell, and what
    /// `cargo` makes of the same flag.
    #[test]
    fn manifest_path_reaches_the_options_as_given() {
        let manifest = PathBuf::from("../contract/Cargo.toml");
        assert_eq!(
            options(&["--manifest-path", "../contract/Cargo.toml"]).manifest_path,
            Some(manifest.clone())
        );
        assert_eq!(
            options(&["--working-dir", "/elsewhere", "--manifest-path", "../contract/Cargo.toml"])
                .manifest_path,
            Some(manifest),
            "`--working-dir` must not move it"
        );
    }

    /// Cargo resolution policy flags reach the Rust project build through the session options.
    #[test]
    fn cargo_resolution_policy_reaches_the_options() {
        let options = options(&["--locked", "--offline"]);
        assert!(options.cargo_locked);
        assert!(options.cargo_offline);
    }
}
