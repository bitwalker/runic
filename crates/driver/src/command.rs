//! The driver's own stable flags.

use std::{path::PathBuf, str::FromStr};

use clap::{Args, ValueEnum};
use runic_compile::{
    DebugInfo, EmitRequest, Extern, OptLevel, OutputFile, RemapPathPrefix, Triple,
};
use runic_ir::Report;

/// The id of the `-Z` argument of the main command, which is on the command only when the driver
/// allows unstable flags. Its values are the texts `NAME[=VALUE]`, in the order written.
pub const UNSTABLE_ARG: &str = "unstable";

/// When the driver colors what it prints.
#[derive(ValueEnum, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColorChoice {
    /// Color when the output is a terminal.
    #[default]
    Auto,
    /// Always color.
    Always,
    /// Never color.
    Never,
}

/// What `--print` can ask the driver to list instead of compiling.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintKind {
    /// The targets the compiler's backends generate code for.
    Targets,
    /// The names of the compiler's backends.
    Backends,
    /// The names of the compiler's frontends.
    Frontends,
    /// The checkpoints a compilation can stop at.
    Checkpoints,
}

/// Parses `text` with the `FromStr` of `T` and gives `clap` the report's message, so that its
/// error names the flag and the text the user wrote.
fn from_str_report<T: FromStr<Err = Report>>(text: &str) -> Result<T, String> {
    text.parse().map_err(|report: Report| report.to_string())
}

/// The stable flags of the driver itself: what a command line says about the inputs, the target
/// and the outputs of a compilation.
///
/// The driver adds them to its command, and a parser of its matches reads them back with
/// `FromArgMatches`. The `-Z` argument and the flags of the stages are not fields: the first is
/// the argument [UNSTABLE_ARG], and the second are described by the driver's stages.
#[derive(Args, Debug, Clone)]
#[command(about = None, long_about = None)]
pub struct Flags {
    /// The files to compile, or `-` for standard input.
    #[arg(value_name = "INPUT")]
    pub input: Vec<String>,

    /// The name of the unit compiled, which also names its output files. The default is the name
    /// of the first input without its extension.
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,

    /// The language of the inputs written after this flag, by the name of its frontend.
    #[arg(long, value_name = "NAME")]
    pub language: Vec<String>,

    /// The target to generate code for, as a triple.
    #[arg(long, value_name = "TRIPLE", value_parser = from_str_report::<Triple>)]
    pub target: Option<Triple>,

    /// The backend to generate code with, by name.
    #[arg(long, value_name = "NAME")]
    pub backend: Option<String>,

    /// How much to optimize: `0`, `1`, `2`, `3`, `s` or `z`.
    #[arg(
        short = 'O',
        long,
        value_name = "LEVEL",
        value_parser = from_str_report::<OptLevel>
    )]
    pub opt_level: Option<OptLevel>,

    /// How much debug information to produce: `none`, `line` or `full` (the default with no
    /// value). The value is given with `=`.
    #[arg(
        short = 'g',
        long,
        value_name = "LEVEL",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "full",
        value_parser = from_str_report::<DebugInfo>
    )]
    pub debug: Option<DebugInfo>,

    /// A library the inputs use, as `NAME` or `NAME=PATH`.
    #[arg(
        long = "extern",
        value_name = "NAME[=PATH]",
        value_parser = from_str_report::<Extern>
    )]
    pub externs: Vec<Extern>,

    /// A directory to search for the libraries named by `--extern`.
    #[arg(short = 'L', long, value_name = "PATH")]
    pub search_path: Vec<PathBuf>,

    /// An output to write, as `NAME` or `NAME=PATH`, where NAME is a checkpoint or one of its
    /// aliases (`--print checkpoints` lists them) and a PATH of `-` is standard output.
    #[arg(long, value_name = "NAME[=PATH]", value_parser = from_str_report::<EmitRequest>)]
    pub emit: Vec<EmitRequest>,

    /// The checkpoint after which the compilation stops.
    #[arg(long, value_name = "NAME")]
    pub stop_after: Option<String>,

    /// The file to write the output to, or `-` for standard output.
    #[arg(
        short = 'o',
        long,
        value_name = "PATH",
        value_parser = from_str_report::<OutputFile>
    )]
    pub output_file: Option<OutputFile>,

    /// The directory to write the outputs to.
    #[arg(long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Replace the prefix FROM of the paths recorded in debug information with TO (`.` when TO is
    /// not given).
    #[arg(
        long,
        value_name = "FROM[=TO]",
        value_parser = from_str_report::<RemapPathPrefix>
    )]
    pub remap_path_prefix: Vec<RemapPathPrefix>,

    /// When to color what the driver prints.
    #[arg(long, value_name = "WHEN", value_enum, default_value_t = ColorChoice::Auto)]
    pub color: ColorChoice,

    /// Print what the compiler offers instead of compiling.
    #[arg(long, value_name = "WHAT", value_enum)]
    pub print: Vec<PrintKind>,
}
