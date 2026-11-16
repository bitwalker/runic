//! The flags of a compiler's stages, on any command.
//!
//! [StageFlags] is what a command line needs of the frontends and backends of a [Compiler]: it
//! checks what they declare against the rules of a command line, adds their stable flags to a
//! `clap` command (and `-Z`, with which the unstable settings are set), and reads what a user
//! wrote back into a [CompilationRequest]. The driver builds its command on it, and so does any
//! other command line that hands a compiler's stages their options.

use std::{
    any::TypeId,
    collections::HashMap,
    ffi::{OsStr, OsString},
};

use clap::{Arg, ArgAction, ArgMatches, Command, parser::ValueSource};
use runic_compile::{CompilationRequest, Compiler, OptionsDecl};
use runic_ir::Report;

use crate::{
    command::UNSTABLE_ARG,
    request::{Exit, scratch_text, usage},
};

/// Whether a stage is a frontend or a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageKind {
    /// A stage that reads inputs.
    Frontend,
    /// A stage that generates code.
    Backend,
}

impl core::fmt::Display for StageKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Frontend => "frontend",
            Self::Backend => "backend",
        })
    }
}

/// A flag of the main command that is a stable option of a stage.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StageFlag {
    id: String,
    long: String,
    field: String,
}

/// A frontend or backend that declares options, and what was built from them.
///
/// A reader of the main command's matches hands the stage its values through the scratch
/// commands: it rebuilds a command line for them, parses it with `stable` or `unstable`, and
/// stores the result with the stage's [OptionsDecl]. The ids and long names of the scratch
/// commands are the ones in `flags`, without the id prefix.
#[derive(Debug, Clone)]
struct StageCommands {
    kind: StageKind,
    name: &'static str,
    /// How an error names the stage: `the frontend 'calc'`.
    declarer: String,
    /// The help heading of the stage's flags on the main command.
    heading: String,
    decl: OptionsDecl,
    /// A command of the stage's stable options only, with the long names the main command has
    /// (prefixed) and the ids the stage's own options declare. It has no binary name.
    stable: Command,
    /// A command of the stage's unstable options only, with prefixed long names and the ids the
    /// stage declares. It has no binary name.
    unstable: Command,
    /// The stable flags of the stage that are on the main command, in declaration order.
    flags: Vec<StageFlag>,
}

/// The unstable settings of the command that owns the flags.
#[derive(Debug, Clone)]
struct OwnUnstable {
    /// Their group in `-Z help`.
    heading: String,
    /// A built command of them only, with no binary name.
    command: Command,
}

impl OwnUnstable {
    /// How an error names the declarer of the settings.
    fn declarer(&self) -> String {
        format!("the {}", self.heading)
    }
}

/// The flags of a [Compiler]'s frontends and backends, checked against the rules of a command
/// line and ready to be added to a command.
///
/// The rules the flags are held to:
///
/// * a backend's flags are prefixed with the backend's name, so that `allow_undefined` of the
///   backend `wasm-cm` is `--wasm-cm-allow-undefined`; a frontend's are not, unless the builder is
///   asked to prefix it ([StageFlagsBuilder::prefix_frontend]);
/// * a stage's options are long flags that take at most one value each: no short flag, no
///   positional argument, no required argument (nor a required group, nor a subcommand), no alias,
///   and no `env`: nothing in a compilation reads the process's environment except through the
///   request;
/// * a stage's options are a `clap` 4 `Args` type of its own: a request holds one value of a
///   type, so two declarers cannot share one;
/// * no two flags have one spelling, and no two unstable settings do;
/// * a stage's unstable settings are never flags of the command: they are reachable only as
///   `-Z NAME[=VALUE]`.
///
/// Built by [StageFlags::new] and [StageFlagsBuilder::build]; the command is given the flags by
/// [augment](Self::augment), and its matches are read by [store](Self::store).
#[derive(Clone)]
pub struct StageFlags<'c> {
    compiler: &'c Compiler,
    unstable_allowed: bool,
    own: Option<OwnUnstable>,
    stages: Vec<StageCommands>,
}

/// Builds [StageFlags]; made by [StageFlags::new].
#[derive(Clone)]
pub struct StageFlagsBuilder<'c> {
    compiler: &'c Compiler,
    unstable: bool,
    prefixed: Vec<String>,
    own: Option<OwnUnstable>,
}

impl<'c> StageFlags<'c> {
    /// Starts the flags of `compiler`'s frontends and backends.
    #[allow(
        clippy::new_ret_no_self,
        reason = "the flags are made by their builder"
    )]
    pub fn new(compiler: &'c Compiler) -> StageFlagsBuilder<'c> {
        StageFlagsBuilder {
            compiler,
            unstable: true,
            prefixed: Vec::new(),
            own: None,
        }
    }

    /// The compiler whose stages the flags are of.
    pub fn compiler(&self) -> &'c Compiler {
        self.compiler
    }

    /// `command` with every stage's stable flags added under the stage's help heading, and `-Z`
    /// when unstable settings are allowed. `owner` is how an error names the declarer of the flags
    /// `command` already has (`the driver`, `runic-opt`): a stage flag that spells one of them, or
    /// `help` or `version`, is an error naming both.
    ///
    /// The command is not built: the caller may add to it first.
    pub fn augment(&self, mut command: Command, owner: &str) -> Result<Command, Report> {
        let mut names = Namespace::new("flag", "--");
        for name in command.get_arguments().filter_map(Arg::get_long) {
            names.claim(name, owner)?;
        }
        for name in ["help", "version"] {
            if !names.contains(name) {
                names.claim(name, owner)?;
            }
        }
        if self.unstable_allowed {
            let taken = command
                .get_arguments()
                .any(|arg| arg.get_id() == UNSTABLE_ARG || arg.get_short() == Some('Z'));
            if taken {
                return Err(Report::msg(format!(
                    "'-Z' is the flag of the unstable settings, and {owner} declares an argument \
                     with its id or its short spelling"
                )));
            }
            command = command.arg(
                Arg::new(UNSTABLE_ARG)
                    .short('Z')
                    .value_name("NAME[=VALUE]")
                    .action(ArgAction::Append)
                    .help("Set an unstable option; `-Z help` lists them"),
            );
        }
        for stage in &self.stages {
            for flag in &stage.flags {
                names.claim(&flag.long, &stage.declarer)?;
            }
            let from = stage.stable.get_arguments();
            for (flag, arg) in stage.flags.iter().zip(from) {
                command = command.arg(carried(arg, &flag.id, &flag.long, &stage.heading));
            }
        }
        Ok(command)
    }

    /// Reads the stages' values from the matches of a command [augment](Self::augment) built and
    /// stores them in `request`: each stage's stable flags, in the order written, and the `-Z`
    /// entries. Answers with the matches of the owner's own unstable settings, when it declared
    /// some and at least one was written.
    ///
    /// Fails with [Exit::Help] for `-Z help`, and with an [Exit::Usage] for an unknown name, a
    /// value where none is taken, a missing value, a bad value (worded with `-Z NAME`), and a
    /// failed parse of a stage's stable flags.
    pub fn store(
        &self,
        matches: &ArgMatches,
        request: &mut CompilationRequest,
    ) -> Result<Option<ArgMatches>, Exit> {
        let settings = self.resolve(matches)?;
        self.store_resolved(matches, &settings, request)
    }

    /// Whether the command line the matches were read from wrote a flag of a stage or a `-Z`
    /// entry, as a command line that decides by it whether the stages take part in a compilation
    /// asks. Matches of a command that was not [augment](Self::augment)ed answer `false`.
    pub fn written(&self, matches: &ArgMatches) -> bool {
        let command_line = |id: &str| {
            matches.try_contains_id(id).unwrap_or(false)
                && matches.value_source(id) == Some(ValueSource::CommandLine)
        };
        let unstable = self.unstable_allowed
            && matches
                .try_get_many::<String>(UNSTABLE_ARG)
                .ok()
                .flatten()
                .is_some_and(|mut entries| entries.next().is_some());
        unstable
            || self
                .stages
                .iter()
                .any(|stage| stage.flags.iter().any(|flag| command_line(&flag.id)))
    }

    /// The text `-Z help` prints: the owner's settings first, then each stage's. Empty when
    /// unstable settings are not allowed.
    pub fn unstable_help(&self) -> String {
        if !self.unstable_allowed {
            return String::new();
        }
        let mut groups: Vec<(String, Vec<(String, String)>)> = Vec::new();
        if let Some(own) = &self.own {
            groups.push((own.heading.clone(), settings(&own.command)));
        }
        for stage in self.stages.iter().filter(|stage| stage.decl.has_unstable()) {
            groups.push((format!("{} {}", stage.name, stage.kind), settings(&stage.unstable)));
        }
        let width = groups
            .iter()
            .flat_map(|(_, settings)| settings.iter().map(|(spec, _)| spec.len()))
            .max()
            .unwrap_or(0);
        let mut text = String::from("Unstable options (-Z NAME[=VALUE]):\n");
        for (heading, settings) in &groups {
            text.push('\n');
            text.push_str(heading);
            text.push('\n');
            for (spec, help) in settings {
                text.push_str(&format!("  {spec:width$}  {help}\n").replace("  \n", "\n"));
            }
        }
        text
    }

    /// Whether the command accepts `-Z`.
    pub fn unstable_allowed(&self) -> bool {
        self.unstable_allowed
    }

    /// The commands of unstable settings: the owner's, then each stage's.
    #[cfg(test)]
    pub(crate) fn unstable_commands(&self) -> impl Iterator<Item = &Command> {
        self.own
            .iter()
            .map(|own| &own.command)
            .chain(self.stages.iter().map(|stage| &stage.unstable))
    }

    /// The `-Z` entries of the command line, each resolved to the argument it becomes. `-Z help`
    /// is an [Exit::Help].
    pub(crate) fn resolve(&self, matches: &ArgMatches) -> Result<Resolved, Exit> {
        if !self.unstable_allowed {
            return Ok(Resolved(Vec::new()));
        }
        let Ok(Some(entries)) = matches.try_get_many::<String>(UNSTABLE_ARG) else {
            return Ok(Resolved(Vec::new()));
        };
        let mut settings = Vec::new();
        for entry in entries {
            let (name, value) = match entry.split_once('=') {
                Some((name, value)) => (name, Some(value)),
                None => (entry.as_str(), None),
            };
            if name.is_empty() {
                return Err(usage(
                    "'-Z' needs the name of an unstable option; '-Z help' lists them",
                ));
            }
            if name == "help" {
                return match value {
                    None => Err(Exit::Help(self.unstable_help())),
                    Some(_) => Err(usage("'-Z help' takes no value")),
                };
            }
            let Some((owner, arg)) = self.setting(name) else {
                return Err(usage(format!(
                    "unknown unstable option '-Z {name}': the unstable options are {} ('-Z help' \
                     describes them)",
                    self.setting_names()
                        .iter()
                        .map(|name| format!("'{name}'"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            };
            match (Arity::of(arg), value) {
                (Arity::None, Some(_)) => {
                    return Err(usage(format!(
                        "'-Z {entry}' gives a value to '-Z {name}', which takes none"
                    )));
                }
                (Arity::Required, None) => {
                    return Err(usage(format!(
                        "'-Z {name}' needs a value: write '-Z {name}=VALUE'"
                    )));
                }
                _ => {}
            }
            settings.push(Setting {
                name: name.to_owned(),
                arg: OsString::from(format!("--{entry}")),
                owner,
            });
        }
        Ok(Resolved(settings))
    }

    /// [store](Self::store) for the `-Z` entries [resolve](Self::resolve) already checked.
    pub(crate) fn store_resolved(
        &self,
        matches: &ArgMatches,
        resolved: &Resolved,
        request: &mut CompilationRequest,
    ) -> Result<Option<ArgMatches>, Exit> {
        for stage in &self.stages {
            store_stable(stage, matches, request)?;
        }
        let owned = |owner: Owner| -> Vec<&Setting> {
            resolved.0.iter().filter(|setting| setting.owner == owner).collect()
        };
        // An error of a parse speaks of `-Z NAME`: the scratch commands know only `--NAME`.
        let parse = |command: &Command, owned: &[&Setting]| -> Result<ArgMatches, Exit> {
            command
                .clone()
                .try_get_matches_from(owned.iter().map(|setting| setting.arg.clone()))
                .map_err(|error| in_z_words(&error, owned))
        };

        let mut own_matches = None;
        let own = owned(Owner::Own);
        if let Some(settings) = &self.own
            && !own.is_empty()
        {
            own_matches = Some(parse(&settings.command, &own)?);
        }
        for (index, stage) in self.stages.iter().enumerate() {
            let owned = owned(Owner::Stage(index));
            if owned.is_empty() {
                continue;
            }
            let matches = parse(&stage.unstable, &owned)?;
            stage
                .decl
                .store_unstable(&matches, request)
                .map_err(|error| in_z_words(&error, &owned))?;
        }
        Ok(own_matches)
    }

    /// The unstable setting called `name`, and who owns it.
    fn setting(&self, name: &str) -> Option<(Owner, &Arg)> {
        let by_name = |arg: &&Arg| arg.get_long() == Some(name);
        if let Some(own) = &self.own
            && let Some(arg) = own.command.get_arguments().find(by_name)
        {
            return Some((Owner::Own, arg));
        }
        self.stages.iter().enumerate().find_map(|(index, stage)| {
            let arg = stage.unstable.get_arguments().find(by_name)?;
            Some((Owner::Stage(index), arg))
        })
    }

    /// The names of every unstable setting that is not hidden, the owner's first.
    fn setting_names(&self) -> Vec<&str> {
        let own = self.own.iter().flat_map(|own| own.command.get_arguments());
        let stages = self.stages.iter().flat_map(|stage| stage.unstable.get_arguments());
        own.chain(stages)
            .filter(|arg| !arg.is_hide_set())
            .filter_map(Arg::get_long)
            .collect()
    }
}

impl<'c> StageFlagsBuilder<'c> {
    /// The named frontend's flags get its name as a prefix, as a backend's always do.
    pub fn prefix_frontend(mut self, frontend: &str) -> Self {
        self.prefixed.push(frontend.to_owned());
        self
    }

    /// Whether `-Z` is accepted. The default is `true`. When it is `false`, the stages' unstable
    /// declarations are still checked, so that allowing them later cannot fail.
    pub fn unstable(self, allowed: bool) -> Self {
        Self {
            unstable: allowed,
            ..self
        }
    }

    /// The owner's own unstable settings: a command whose arguments are long flags (as
    /// `T::augment_args` of a `clap::Args` type gives), reachable as `-Z NAME[=VALUE]` beside the
    /// stages'. `heading` is their group in `-Z help` (`driver`), and an error that names the
    /// declarer of a setting says `the` and the heading.
    pub fn own_unstable(self, heading: &str, mut settings: Command) -> Self {
        settings.build();
        Self {
            own: Some(OwnUnstable {
                heading: heading.to_owned(),
                command: settings,
            }),
            ..self
        }
    }

    /// Checks every stage's declarations against the rules and against each other.
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
    /// * two declarers give one spelling to a flag, or to an unstable setting. The owner's own
    ///   settings are a declarer, and `help` is one of them.
    ///
    /// The spellings of the flags of the command are checked against the stages' when the command
    /// is given them, by [StageFlags::augment].
    pub fn build(self) -> Result<StageFlags<'c>, Report> {
        let Self {
            compiler,
            unstable,
            prefixed,
            own,
        } = self;
        check_prefixes(compiler, &prefixed, "the stage flags were")?;

        let mut stable_names = Namespace::new("flag", "--");
        let mut unstable_names = Namespace::new("unstable setting", "-Z ");
        let own_declarer = own.as_ref().map(OwnUnstable::declarer);
        if let (Some(own), Some(declarer)) = (&own, &own_declarer) {
            for name in own.command.get_arguments().filter_map(Arg::get_long) {
                unstable_names.claim(name, declarer)?;
            }
        }
        if !unstable_names.contains("help") {
            unstable_names.claim("help", own_declarer.as_deref().unwrap_or("'-Z' itself"))?;
        }

        let mut types: HashMap<TypeId, String> = HashMap::new();
        let mut stages = Vec::new();
        let stage_decls = compiler
            .frontends()
            .map(|f| (StageKind::Frontend, f.name(), f.options()))
            .chain(compiler.backends().map(|b| (StageKind::Backend, b.name(), b.options())));
        for (kind, stage_name, decl) in stage_decls {
            let Some(decl) = decl else { continue };
            let prefix = match kind {
                StageKind::Backend => Some(stage_name),
                StageKind::Frontend => {
                    prefixed.iter().any(|p| p == stage_name).then_some(stage_name)
                }
            };
            let declarer = format!("the {kind} '{stage_name}'");

            let stable = declared(&decl, kind, stage_name, prefix, false)?;
            let unstable_scratch = declared(&decl, kind, stage_name, prefix, true)?;
            for name in stable.get_arguments().filter_map(Arg::get_long) {
                stable_names.claim(name, &declarer)?;
            }
            for name in unstable_scratch.get_arguments().filter_map(Arg::get_long) {
                unstable_names.claim(name, &declarer)?;
            }
            for (type_id, type_name) in
                [decl.stable_type(), decl.unstable_type()].into_iter().flatten()
            {
                match types.get(&type_id) {
                    Some(first) if *first == declarer => {
                        return Err(Report::msg(format!(
                            "the options type '{type_name}' is declared by {declarer} as both its \
                             stable and its unstable options: each needs a type of its own, since \
                             a request holds one value of a type"
                        )));
                    }
                    Some(first) => {
                        return Err(Report::msg(format!(
                            "the options type '{type_name}' is declared by {first} and by \
                             {declarer}: each declarer needs a type of its own, since a request \
                             holds one value of a type"
                        )));
                    }
                    None => {
                        types.insert(type_id, declarer.clone());
                    }
                }
            }

            let flags = stable
                .get_arguments()
                .map(|arg| {
                    let field = arg.get_id().as_str().to_owned();
                    StageFlag {
                        id: format!("{kind}:{stage_name}:{field}"),
                        long: arg.get_long().unwrap_or_default().to_owned(),
                        field,
                    }
                })
                .collect();
            stages.push(StageCommands {
                kind,
                name: stage_name,
                heading: format!("{stage_name} {kind} options"),
                declarer,
                decl,
                stable,
                unstable: unstable_scratch,
                flags,
            });
        }

        Ok(StageFlags {
            compiler,
            unstable_allowed: unstable,
            own,
            stages,
        })
    }
}

/// Fails, in the words of `who` (`the driver was`), when a frontend that is to be prefixed is not one of the compiler's.
pub(crate) fn check_prefixes(
    compiler: &Compiler,
    prefixed: &[String],
    who: &str,
) -> Result<(), Report> {
    let frontends: Vec<&str> = compiler.frontends().map(|f| f.name()).collect();
    for wanted in prefixed {
        if !frontends.contains(&wanted.as_str()) {
            let known = if frontends.is_empty() {
                "none".to_owned()
            } else {
                frontends.join(", ")
            };
            return Err(Report::msg(format!(
                "{who} asked to prefix the flags of the frontend '{wanted}', but the compiler has \
                 no such frontend; its frontends are {known}"
            )));
        }
    }
    Ok(())
}

/// The settings of a built command of unstable arguments, each as the name a user writes (with
/// `=VALUE`, or `[=VALUE]` when the value is optional) and its help on one line.
fn settings(command: &Command) -> Vec<(String, String)> {
    command
        .get_arguments()
        .filter(|arg| !arg.is_hide_set())
        .filter_map(|arg| {
            let long = arg.get_long()?;
            let takes_value = matches!(arg.get_action(), ArgAction::Set | ArgAction::Append)
                && arg.get_num_args().is_some_and(|range| range.max_values() > 0);
            let spec = if takes_value {
                let value = arg
                    .get_value_names()
                    .and_then(|names| names.first())
                    .map(|name| name.as_str().to_owned())
                    .unwrap_or_else(|| "VALUE".to_owned());
                if arg.get_num_args().is_some_and(|range| range.min_values() == 0) {
                    format!("{long}[={value}]")
                } else {
                    format!("{long}={value}")
                }
            } else {
                long.to_owned()
            };
            let help = arg
                .get_help()
                .map(|help| help.to_string().split_whitespace().collect::<Vec<_>>().join(" "))
                .unwrap_or_default();
            Some((spec, help))
        })
        .collect()
}

/// A command with no binary name and no help or version flag, to hold declared arguments.
pub(crate) fn scratch(add: impl FnOnce(Command) -> Command) -> Command {
    add(Command::new("options")
        .no_binary_name(true)
        .disable_help_flag(true)
        .disable_version_flag(true)
        .disable_help_subcommand(true))
}

/// The scratch command of one stage's stable or unstable options, with long names prefixed, built
/// and checked.
fn declared(
    decl: &OptionsDecl,
    kind: StageKind,
    name: &str,
    prefix: Option<&str>,
    unstable: bool,
) -> Result<Command, Report> {
    let mut command = scratch(|command| {
        if unstable {
            decl.augment_unstable(command)
        } else {
            decl.augment(command)
        }
    });
    if let Some(prefix) = prefix {
        command = command.mut_args(|arg| match arg.get_long().map(str::to_owned) {
            Some(long) => arg.long(format!("{prefix}-{long}")),
            None => arg,
        });
    }
    command.build();
    let what = if unstable {
        "unstable option"
    } else {
        "option"
    };
    for arg in command.get_arguments() {
        check(arg, kind, name, what)?;
    }
    let by = format!("the {kind} '{name}' declares {what}s");
    if let Some(group) = command.get_groups().find(|group| group.is_required_set()) {
        return Err(Report::msg(format!(
            "{by} with the required group '{}'; options of a stage are never required",
            group.get_id()
        )));
    }
    if command.has_subcommands() {
        return Err(Report::msg(format!("{by} with subcommands; options of a stage are flags")));
    }
    Ok(command)
}

/// The argument of the main command that stands for `arg`, a stable option of a stage: its
/// spelling, what it takes and how that is parsed, and its help, under `heading`.
///
/// It carries the option's default of a value-taking option, so that help shows it, but not its
/// relations to the stage's other options (conflicts, requirements): those name ids of the stage's
/// own command, where they are applied when the stage is handed its values. A default is not a
/// value the command line wrote (its source is `DefaultValue`), so the stage is handed only what
/// was written, and its own command applies the defaults.
fn carried(arg: &Arg, id: &str, long: &str, heading: &str) -> Arg {
    let mut carried = Arg::new(id.to_owned())
        .long(long.to_owned())
        .action(arg.get_action().clone())
        .value_parser(arg.get_value_parser().clone())
        .require_equals(arg.is_require_equals_set())
        .allow_hyphen_values(arg.is_allow_hyphen_values_set())
        .allow_negative_numbers(arg.is_allow_negative_numbers_set())
        .ignore_case(arg.is_ignore_case_set())
        .hide(arg.is_hide_set())
        .help_heading(heading.to_owned());
    if let Some(range) = arg.get_num_args() {
        carried = carried.num_args(range);
    }
    if let Some(names) = arg.get_value_names() {
        carried = carried.value_names(names.iter().map(|name| name.as_str().to_owned()));
    }
    if let Some(delimiter) = arg.get_value_delimiter() {
        carried = carried.value_delimiter(delimiter);
    }
    if matches!(arg.get_action(), ArgAction::Set | ArgAction::Append)
        && !arg.get_default_values().is_empty()
    {
        carried = carried.default_values(arg.get_default_values().iter().cloned());
    }
    if let Some(help) = arg.get_help() {
        carried = carried.help(help.clone());
    }
    if let Some(help) = arg.get_long_help() {
        carried = carried.long_help(help.clone());
    }
    carried
}

/// Checks one declared argument of a stage against the rules of the command line.
fn check(arg: &Arg, kind: StageKind, name: &str, what: &str) -> Result<(), Report> {
    let field = arg.get_id();
    let by = format!("the {kind} '{name}' declares the {what} '{field}'");
    let fail = |problem: &str| Err(Report::msg(format!("{by}, which {problem}")));
    if let Some(short) = arg.get_short() {
        return fail(&format!(
            "has the short flag '-{short}'; options of a stage have long flags only"
        ));
    }
    if arg.is_positional() {
        return fail("is positional; options of a stage are flags");
    }
    if arg.is_required_set() {
        return fail("is required; options of a stage are never required");
    }
    if !matches!(
        arg.get_action(),
        ArgAction::Set
            | ArgAction::Append
            | ArgAction::SetTrue
            | ArgAction::SetFalse
            | ArgAction::Count
    ) {
        return fail(
            "has an action the driver cannot carry; it must set a value, append one, be a switch \
             or count",
        );
    }
    if let Some(variable) = arg.get_env() {
        return fail(&format!(
            "reads the environment variable '{}'; options of a stage never read the environment, \
             since nothing in a compilation reads it except through the request",
            variable.to_string_lossy()
        ));
    }
    if arg.get_num_args().is_some_and(|range| range.max_values() > 1) {
        return fail("takes more than one value per occurrence; it takes at most one");
    }
    let has_alias = arg.get_all_aliases().is_some_and(|aliases| !aliases.is_empty())
        || arg.get_all_short_aliases().is_some_and(|aliases| !aliases.is_empty());
    if has_alias {
        return fail("has an alias, which a stage's prefix cannot be applied to");
    }
    Ok(())
}

/// The spellings in use in one namespace of the command line, and who declared each.
struct Namespace {
    what: &'static str,
    dashes: &'static str,
    seen: HashMap<String, String>,
}

impl Namespace {
    fn new(what: &'static str, dashes: &'static str) -> Self {
        Self {
            what,
            dashes,
            seen: HashMap::new(),
        }
    }

    /// Whether some declarer has given `name` to a flag.
    fn contains(&self, name: &str) -> bool {
        self.seen.contains_key(name)
    }

    /// Records that `declarer` gives `name` to a flag, or fails naming the declarer that did first.
    fn claim(&mut self, name: &str, declarer: &str) -> Result<(), Report> {
        match self.seen.get(name) {
            Some(first) => Err(Report::msg(format!(
                "the {} '{}{name}' is declared by {first} and by {declarer}",
                self.what, self.dashes
            ))),
            None => {
                self.seen.insert(name.to_owned(), declarer.to_owned());
                Ok(())
            }
        }
    }
}

/// How many values one occurrence of an unstable setting takes.
enum Arity {
    /// A switch: none.
    None,
    /// `-Z NAME` and `-Z NAME=VALUE` are both allowed.
    Optional,
    /// Only `-Z NAME=VALUE` is allowed.
    Required,
}

impl Arity {
    fn of(arg: &Arg) -> Self {
        let takes_value = matches!(arg.get_action(), ArgAction::Set | ArgAction::Append)
            && arg.get_num_args().is_some_and(|range| range.max_values() > 0);
        match arg.get_num_args() {
            Some(range) if takes_value && range.min_values() == 0 => Self::Optional,
            _ if takes_value => Self::Required,
            _ => Self::None,
        }
    }
}

/// The owner of an unstable setting.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Owner {
    /// The command's own unstable settings.
    Own,
    /// The stage at this index.
    Stage(usize),
}

/// One `-Z` entry, resolved: its name, the argument it becomes and who owns it.
struct Setting {
    name: String,
    arg: OsString,
    owner: Owner,
}

/// The `-Z` entries of a command line, resolved by [StageFlags::resolve].
pub(crate) struct Resolved(Vec<Setting>);

/// The usage error for a failed parse of unstable settings, which speaks of `-Z NAME` where the
/// scratch command says `--NAME`.
fn in_z_words(error: &clap::Error, settings: &[&Setting]) -> Exit {
    let mut text = scratch_text(error);
    for setting in settings {
        text = text.replace(&format!("--{}", setting.name), &format!("-Z {}", setting.name));
    }
    Exit::Usage(text)
}

/// Hands a stage the stable flags the command line gave it: rebuilds them as the arguments of the
/// stage's own command, which applies the stage's defaults and relations, and stores the result in
/// `request`. A stage none of whose flags was written stores nothing.
///
/// The arguments are rebuilt in the order the command line wrote them, so that a stage's own
/// relations that depend on order (`overrides_with`: `--color --no-color`) resolve as the user
/// wrote them. `clap` reports one index for each value of a flag (and one for a switch, and one,
/// the last, for a count), so:
///
/// * a switch is placed at its index, and a count's arguments all at the count's index;
/// * an occurrence of a flag that takes a value is one argument, `--LONG=VALUE` (a value-delimited
///   occurrence is rejoined with the delimiter), placed at the index of its first value;
/// * when the indices of a flag cannot be matched to its occurrences (an occurrence with no value
///   at all: an optional value left out where the argument has no `default_missing_value`), all of
///   the flag's arguments are placed at the flag's first index, or at the start when it has none.
fn store_stable(
    stage: &StageCommands,
    matches: &ArgMatches,
    request: &mut CompilationRequest,
) -> Result<(), Exit> {
    let mut args: Vec<(usize, OsString)> = Vec::new();
    let mut given = false;
    for flag in &stage.flags {
        if matches.value_source(&flag.id) != Some(ValueSource::CommandLine) {
            continue;
        }
        given = true;
        let Some(arg) =
            stage.stable.get_arguments().find(|arg| arg.get_id() == flag.field.as_str())
        else {
            continue;
        };
        let long = format!("--{}", flag.long);
        let indices: Vec<usize> =
            matches.indices_of(&flag.id).map(Iterator::collect).unwrap_or_default();
        let first = indices.first().copied().unwrap_or(0);
        match arg.get_action() {
            ArgAction::SetTrue | ArgAction::SetFalse => args.push((first, long.into())),
            ArgAction::Count => {
                for _ in 0..matches.get_count(&flag.id) {
                    args.push((first, long.clone().into()));
                }
            }
            _ => {
                let occurrences: Vec<Vec<&OsStr>> = matches
                    .get_raw_occurrences(&flag.id)
                    .into_iter()
                    .flatten()
                    .map(Iterator::collect)
                    .collect();
                let values: usize = occurrences.iter().map(Vec::len).sum();
                let placed = values == indices.len() && occurrences.iter().all(|o| !o.is_empty());
                let mut next = 0;
                for occurrence in occurrences {
                    let at = if placed { indices[next] } else { first };
                    next += occurrence.len();
                    match (occurrence.as_slice(), arg.get_value_delimiter()) {
                        ([], _) => args.push((at, long.clone().into())),
                        (values, Some(delimiter)) => {
                            let mut joined = OsString::new();
                            for (position, value) in values.iter().enumerate() {
                                if position > 0 {
                                    joined.push(delimiter.to_string());
                                }
                                joined.push(value);
                            }
                            args.push((at, with_value(&long, &joined)));
                        }
                        (values, None) => {
                            for value in values {
                                args.push((at, with_value(&long, value)));
                            }
                        }
                    }
                }
            }
        }
    }
    if !given {
        return Ok(());
    }
    args.sort_by_key(|(at, _)| *at);
    let scratch = stage
        .stable
        .clone()
        .try_get_matches_from(args.into_iter().map(|(_, arg)| arg))
        .map_err(|error| Exit::Usage(scratch_text(&error)))?;
    stage
        .decl
        .store(&scratch, request)
        .map_err(|error| Exit::Usage(scratch_text(&error)))
}

/// The argument `--LONG=VALUE`, built without going through a string.
fn with_value(long: &str, value: &OsStr) -> OsString {
    let mut arg = OsString::from(long);
    arg.push("=");
    arg.push(value);
    arg
}

#[cfg(test)]
mod tests {
    use clap::{Args, error::ErrorKind};
    use runic_compile::{
        OptionsDecl,
        testing::{CalcOptions, StackVmOptions, StackVmUnstable},
    };

    use super::*;
    use crate::{
        Driver,
        tests::{backend, stable, toy},
    };

    /// A command of a program's own: a flag and a positional.
    fn own_command() -> Command {
        Command::new("mytool")
            .arg(Arg::new("pipeline").long("pass-pipeline").value_name("PIPELINE"))
            .arg(Arg::new("input").value_name("INPUT"))
    }

    fn flags(compiler: &Compiler) -> StageFlags<'_> {
        StageFlags::new(compiler).build().unwrap()
    }

    fn matches_of(command: &Command, args: &[&str]) -> ArgMatches {
        command
            .clone()
            .try_get_matches_from(core::iter::once("mytool").chain(args.iter().copied()))
            .unwrap()
    }

    fn longs(command: &Command) -> Vec<&str> {
        command.get_arguments().filter_map(Arg::get_long).collect()
    }

    #[test]
    fn a_stage_flag_is_recorded_with_its_main_id_and_long_name() {
        let compiler = toy();
        let flags = flags(&compiler);
        let names: Vec<(StageKind, &str)> = flags.stages.iter().map(|s| (s.kind, s.name)).collect();
        assert_eq!(names, [(StageKind::Frontend, "calc"), (StageKind::Backend, "stackvm")]);
        let stackvm = &flags.stages[1];
        let flag = &stackvm.flags[0];
        assert_eq!(
            (flag.id.as_str(), flag.long.as_str(), flag.field.as_str()),
            ("backend:stackvm:comments", "stackvm-comments", "comments")
        );
        assert!(stackvm.stable.get_arguments().any(|a| a.get_id() == "comments"));
    }

    #[test]
    fn the_toy_stages_flags_appear_on_a_command_of_the_programs_own_under_their_headings() {
        let compiler = toy();
        let command = flags(&compiler).augment(own_command(), "mytool").unwrap();
        let wrap = command.get_arguments().find(|a| a.get_long() == Some("wrap-on-overflow"));
        assert_eq!(wrap.unwrap().get_help_heading(), Some("calc frontend options"));
        let comments = command.get_arguments().find(|a| a.get_long() == Some("stackvm-comments"));
        assert_eq!(comments.unwrap().get_help_heading(), Some("stackvm backend options"));
        assert_eq!(longs(&command)[0], "pass-pipeline");
        assert!(command.get_arguments().any(|a| a.get_short() == Some('Z')));
        let mut built = command;
        built.build();
        built.debug_assert();
    }

    #[test]
    fn a_frontend_flag_spelled_like_one_of_the_commands_is_an_error_naming_the_owner() {
        #[derive(Args, Default, Debug, Clone)]
        struct Pipeline {
            #[arg(long)]
            pass_pipeline: bool,
        }
        let compiler = Compiler::builder()
            .frontend(crate::tests::Fe("a", stable::<Pipeline>))
            .build()
            .unwrap();
        let error = flags(&compiler).augment(own_command(), "mytool").unwrap_err();
        assert_eq!(
            error.to_string(),
            "the flag '--pass-pipeline' is declared by mytool and by the frontend 'a'"
        );
    }

    #[test]
    fn a_stage_flag_spelled_help_or_version_is_an_error_naming_the_owner() {
        #[derive(Args, Default, Debug, Clone)]
        struct Version {
            #[arg(long)]
            version: bool,
        }
        let compiler =
            Compiler::builder().backend(backend("v", stable::<Version>)).build().unwrap();
        // The backend's flag is `--v-version`: only an unprefixed frontend can collide.
        flags(&compiler).augment(own_command(), "mytool").unwrap();
        let compiler = Compiler::builder()
            .frontend(crate::tests::Fe("v", stable::<Version>))
            .build()
            .unwrap();
        let error = flags(&compiler).augment(own_command(), "mytool").unwrap_err();
        assert_eq!(
            error.to_string(),
            "the flag '--version' is declared by mytool and by the frontend 'v'"
        );
    }

    #[test]
    fn a_stage_flag_or_an_unstable_entry_written_is_told_from_the_matches() {
        let compiler = toy();
        let flags = flags(&compiler);
        let command = flags.augment(own_command(), "mytool").unwrap();
        assert!(!flags.written(&matches_of(&command, &["--pass-pipeline", "cse", "x"])));
        assert!(flags.written(&matches_of(&command, &["--wrap-on-overflow"])));
        assert!(flags.written(&matches_of(&command, &["--stackvm-comments"])));
        assert!(flags.written(&matches_of(&command, &["-Z", "stackvm-keep-subtractions"])));
        assert!(!flags.written(&matches_of(&own_command(), &["x"])));
    }

    #[test]
    fn values_written_reach_a_request_through_store() {
        let compiler = toy();
        let command = flags(&compiler).augment(own_command(), "mytool").unwrap();
        let matches =
            matches_of(&command, &["--stackvm-comments", "-Z", "stackvm-keep-subtractions", "in"]);
        let mut request = CompilationRequest::new();
        let own = flags(&compiler).store(&matches, &mut request).unwrap();
        assert!(own.is_none());
        assert!(request.options::<StackVmOptions>().unwrap().comments);
        assert!(request.options::<StackVmUnstable>().unwrap().keep_subtractions);
        assert!(request.options::<CalcOptions>().is_none());
    }

    #[test]
    fn the_owners_own_unstable_settings_are_answered_when_one_is_written() {
        let compiler = toy();
        let flags = StageFlags::new(&compiler)
            .own_unstable("mytool", scratch(crate::unstable::Unstable::augment_args))
            .build()
            .unwrap();
        let command = flags.augment(own_command(), "mytool").unwrap();
        let mut request = CompilationRequest::new();
        let none = flags.store(&matches_of(&command, &["in"]), &mut request).unwrap();
        assert!(none.is_none());
        let some = flags
            .store(&matches_of(&command, &["-Z", "print-generic"]), &mut request)
            .unwrap()
            .expect("a setting of the owner was written");
        assert!(some.get_flag("print_generic"));
        let help = flags.unstable_help();
        assert!(help.find("\nmytool\n").unwrap() < help.find("\nstackvm backend\n").unwrap());
    }

    #[test]
    fn without_own_settings_a_name_that_only_the_owner_would_have_is_unknown_and_the_stages_are_listed()
     {
        let compiler = toy();
        let flags = flags(&compiler);
        let command = flags.augment(own_command(), "mytool").unwrap();
        let mut request = CompilationRequest::new();
        let Err(Exit::Usage(text)) =
            flags.store(&matches_of(&command, &["-Z", "pass-pipeline=x"]), &mut request)
        else {
            panic!("an unknown name is a usage error");
        };
        assert!(text.contains("unknown unstable option '-Z pass-pipeline'"), "{text}");
        assert!(text.contains("'stackvm-keep-subtractions'"), "{text}");
        assert!(!text.contains("'print-generic'"), "{text}");
    }

    #[test]
    fn with_unstable_settings_not_allowed_there_is_no_dash_z() {
        let compiler = toy();
        let flags = StageFlags::new(&compiler).unstable(false).build().unwrap();
        assert!(!flags.unstable_allowed());
        assert_eq!(flags.unstable_help(), "");
        let command = flags.augment(own_command(), "mytool").unwrap();
        assert!(!command.get_arguments().any(|a| a.get_short() == Some('Z')));
        let error = command.try_get_matches_from(["mytool", "-Z", "x"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnknownArgument);
    }

    #[test]
    fn a_command_that_has_dash_z_already_is_an_error_naming_the_owner() {
        let compiler = toy();
        let command = own_command().arg(Arg::new("zed").short('Z'));
        let error = flags(&compiler).augment(command, "mytool").unwrap_err();
        assert_eq!(
            error.to_string(),
            "'-Z' is the flag of the unstable settings, and mytool declares an argument with its \
             id or its short spelling"
        );
    }

    #[test]
    fn prefixing_a_frontend_the_compiler_lacks_is_an_error_listing_the_frontends() {
        let compiler = toy();
        let error = StageFlags::new(&compiler).prefix_frontend("nope").build().err().unwrap();
        assert_eq!(
            error.to_string(),
            "the stage flags were asked to prefix the flags of the frontend 'nope', but the \
             compiler has no such frontend; its frontends are calc, runic"
        );
    }

    #[derive(Args, Default, Debug, Clone)]
    struct Related {
        /// How many.
        #[arg(long, value_name = "N", default_value_t = 3)]
        level: u8,
        #[arg(long, conflicts_with = "level")]
        plain: bool,
        #[arg(long, value_name = "WHEN", num_args = 0..=1, require_equals = true,
              default_missing_value = "always")]
        trace: Option<String>,
    }

    #[test]
    fn a_stage_flag_on_the_main_command_carries_its_default_and_no_relation() {
        let compiler = Compiler::builder()
            .backend(backend("rel", || Some(OptionsDecl::of::<Related>())))
            .build()
            .unwrap();
        let driver = Driver::new(&compiler).build().unwrap();
        // The relation names an id of the stage's own command; the main command stays consistent.
        driver.command().clone().debug_assert();

        // A default is not a value the command line wrote: its source says so, and the driver
        // hands the stage only what was written.
        let matches = driver.command().clone().try_get_matches_from(["x"]).unwrap();
        assert_eq!(matches.value_source("backend:rel:level"), Some(ValueSource::DefaultValue));
        assert_eq!(matches.get_one::<u8>("backend:rel:level"), Some(&3));
        assert_eq!(matches.value_source("backend:rel:trace"), None);
        let mut command = driver.command().clone();
        let help = command.render_long_help().to_string();
        assert!(help.contains("[default: 3]"), "{help}");

        // What the flag takes and how it is parsed is carried.
        let matches = driver
            .command()
            .clone()
            .try_get_matches_from(["x", "--rel-level", "7", "--rel-trace", "in.x"])
            .unwrap();
        assert_eq!(matches.get_one::<u8>("backend:rel:level"), Some(&7));
        assert_eq!(matches.value_source("backend:rel:level"), Some(ValueSource::CommandLine));
        // `--rel-trace` takes its value with `=` only, so `in.x` is an input.
        assert_eq!(matches.get_raw("backend:rel:trace").map(|values| values.count()), Some(0));
        let error = driver
            .command()
            .clone()
            .try_get_matches_from(["x", "--rel-level", "many"])
            .unwrap_err();
        assert!(error.to_string().contains("--rel-level <N>"), "{error}");

        // The stage's own command applies the default and the relation.
        let stage = &driver.stage_flags().stages[0];
        let matches = stage.stable.clone().try_get_matches_from(["--rel-trace"]).unwrap();
        assert_eq!(matches.get_one::<u8>("level"), Some(&3));
        assert_eq!(matches.get_one::<String>("trace").map(String::as_str), Some("always"));
        let error = stage
            .stable
            .clone()
            .try_get_matches_from(["--rel-level=1", "--rel-plain"])
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
    }
}
