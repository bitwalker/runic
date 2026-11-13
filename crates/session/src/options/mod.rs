mod printing;

use alloc::{boxed::Box, string::String, vec, vec::Vec};

pub use self::printing::IrFilter;
use crate::{
    ColorChoice, CompileFlags, InputFile, LinkLibrary, OutputFile, OutputTypes, PathBuf,
    diagnostics::Report,
};

/// This struct contains all of the configuration options for the compiler
#[derive(Debug, Clone)]
pub struct Options {
    /// The name of the program being compiled
    pub name: Option<String>,
    /// The name of the current project target being compiled
    pub target: Option<String>,
    /// The optimization level for the current program
    pub optimize: OptLevel,
    /// The level of debugging info for the current program
    pub debug: DebugInfo,
    /// The type of outputs to emit
    pub output_types: OutputTypes,
    /// The paths in which to search for libraries to link against
    pub search_paths: Vec<PathBuf>,
    /// The set of libraries to link against
    pub link_libraries: Vec<LinkLibrary>,
    /// Whether, and how, to color terminal output
    pub color: ColorChoice,
    /// The current working directory of the compiler
    pub current_dir: PathBuf,
    /// The target directory of the compiler
    pub target_dir: PathBuf,
    /// The artifact output directory of the compiler
    pub output_dir: Option<PathBuf>,
    /// The output file requested by the user, if requested
    pub output_file: Option<OutputFile>,
    /// Path prefixes to remap for any file paths encoded in debug info
    pub remap_path_prefixes: Vec<RemapPathPrefix>,
    /// Print source location information in HIR output
    pub print_hir_source_locations: bool,
    /// Stop compilation after the named checkpoint, as `--stop-after` asked.
    ///
    /// An alias declared by the route being compiled — `parse`, `analyze`, `transform`,
    /// `lower`, `assemble` — or a fully-qualified checkpoint id, `<dialect>.<name>`. Which
    /// names are valid depends on the frontend the input selects, so the value is carried
    /// uninterpreted and resolved against that route once it is known; an unrecognized one is
    /// reported there, listing the names that route does accept.
    ///
    /// This is the general form of the `-C` stop flags below, and naming both is a usage error
    /// rather than a precedence rule.
    pub stop_after: Option<String>,
    /// Only parse inputs
    pub parse_only: bool,
    /// Only perform semantic analysis on the input
    pub analyze_only: bool,
    /// Print CFG to stdout after each pass
    pub print_cfg_after_all: bool,
    /// Print CFG to stdout each time the named passes are applied
    pub print_cfg_after_pass: Vec<String>,
    /// Print IR to stdout at the start of each stage
    pub print_ir_before_stage: Vec<String>,
    /// Print IR to stdout after each pass
    pub print_ir_after_all: bool,
    /// Print IR to stdout each time the named passes are applied
    pub print_ir_after_pass: Vec<String>,
    /// Only print the IR if the pass modified the IR structure.
    pub print_ir_after_modified: bool,
    /// Apply filters to what IR is printed, when printing is enabled
    pub print_ir_filters: Vec<IrFilter>,
    /// Save intermediate artifacts in memory during compilation
    pub save_temps: bool,
    /// We store any leftover argument matches in the session options for use
    /// by any downstream crates that register custom flags
    pub flags: CompileFlags,
}

impl Default for Options {
    fn default() -> Self {
        let current_dir = current_dir();
        let target_dir = current_dir.join("target");

        Self::new(None, current_dir, target_dir, None)
    }
}

impl Options {
    pub fn new(
        name: Option<String>,
        current_dir: PathBuf,
        target_dir: PathBuf,
        output_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            name,
            target: None,
            optimize: OptLevel::None,
            debug: DebugInfo::None,
            output_types: Default::default(),
            search_paths: vec![],
            link_libraries: vec![],
            color: Default::default(),
            current_dir,
            target_dir,
            output_dir,
            output_file: None,
            print_hir_source_locations: false,
            stop_after: None,
            parse_only: false,
            analyze_only: false,
            save_temps: false,
            print_cfg_after_all: false,
            print_cfg_after_pass: vec![],
            print_ir_before_stage: vec![],
            print_ir_after_all: false,
            print_ir_after_pass: vec![],
            print_ir_after_modified: false,
            print_ir_filters: vec![],
            remap_path_prefixes: vec![],
            flags: CompileFlags::default(),
        }
    }

    #[inline(always)]
    pub fn with_color(mut self: Box<Self>, color: ColorChoice) -> Box<Self> {
        self.color = color;
        self
    }

    #[inline(always)]
    pub fn with_debug_info(mut self: Box<Self>, debug: DebugInfo) -> Box<Self> {
        self.debug = debug;
        self
    }

    #[inline(always)]
    pub fn with_optimization(mut self: Box<Self>, level: OptLevel) -> Box<Self> {
        self.optimize = level;
        self
    }

    pub fn with_output_types(
        mut self: Box<Self>,
        mut output_types: OutputTypes,
        output_file: Option<OutputFile>,
    ) -> Box<Self> {
        use crate::OutputType;
        let has_final_output = output_types.keys().any(|ty| matches!(ty, OutputType::Hir));
        if !has_final_output {
            // By default, we always produce a final artifact; `--emit` selects additional outputs.
            output_types.insert(OutputType::Hir, output_file);
        } else if output_file.is_some() && output_types.get(&OutputType::Hir).is_some() {
            // The -o flag overrides --emit
            output_types.insert(OutputType::Hir, output_file);
        }
        self.output_types = output_types;
        self
    }

    #[doc(hidden)]
    pub fn with_extra_flags(mut self: Box<Self>, flags: CompileFlags) -> Box<Self> {
        self.flags = flags;
        self
    }

    #[doc(hidden)]
    pub fn set_extra_flags(&mut self, flags: CompileFlags) {
        self.flags = flags;
    }

    /// Use this configuration to obtain a [crate::Session] used for compilation
    pub fn into_session(self: Box<Self>, input: InputFile) -> Result<crate::Session, Report> {
        crate::Session::new(input, self)
    }

    /// Returns true if source location information should be emitted by the compiler
    #[inline(always)]
    pub fn emit_source_locations(&self) -> bool {
        matches!(self.debug, DebugInfo::Line | DebugInfo::Full)
    }

    /// Returns true if rich debugging information should be emitted by the compiler.
    /// This enables AssemblyOp decorators which carry source location info for runtime errors.
    #[inline(always)]
    pub fn emit_debug_decorators(&self) -> bool {
        matches!(self.debug, DebugInfo::Line | DebugInfo::Full)
    }

    /// Returns true if debug assertions are enabled
    #[inline(always)]
    pub fn emit_debug_assertions(&self) -> bool {
        self.debug != DebugInfo::None && matches!(self.optimize, OptLevel::None | OptLevel::Basic)
    }

    /// The IR printer flags these options select.
    pub fn op_printing_flags(&self) -> runic_core::print::OpPrintingFlags {
        runic_core::print::OpPrintingFlags {
            print_source_locations: self.print_hir_source_locations,
            ..Default::default()
        }
    }

    /// The IR printing configuration these options select.
    pub fn ir_printing_config(&self) -> Result<runic_core::pass::IRPrintingConfig, Report> {
        use runic_core::pass::OpFilter;

        let pass_filters = self.print_ir_after_pass.clone();

        if self.print_ir_after_all && !pass_filters.is_empty() {
            return Err(Report::msg(
                "Flags `print_ir_after_all` and `print_ir_after_pass` are mutually exclusive. \
                 Please select only one.",
            ));
        };

        let print_ir_filters = self
            .print_ir_filters
            .iter()
            .cloned()
            .map(OpFilter::from)
            .filter(|filter| !matches!(filter, OpFilter::All))
            .collect();

        Ok(runic_core::pass::IRPrintingConfig {
            print_ir_after_all: self.print_ir_after_all,
            print_ir_after_pass: pass_filters.into(),
            print_ir_after_modified: self.print_ir_after_modified,
            print_ir_filters,
            ..Default::default()
        })
    }
}

impl From<IrFilter> for runic_core::pass::OpFilter {
    fn from(value: IrFilter) -> Self {
        match value {
            IrFilter::Any => Self::All,
            IrFilter::Symbol(pattern) => Self::Symbol(pattern),
            IrFilter::Op { dialect, op } => Self::Type { dialect, op },
        }
    }
}

/// This enum describes the degree to which compiled programs will be optimized
#[derive(Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "std", derive(clap::ValueEnum))]
pub enum OptLevel {
    /// No optimizations at all
    None,
    /// Only basic optimizations are applied, e.g. constant propagation
    Basic,
    /// Most optimizations are applied, except when the cost is particularly high.
    #[default]
    Balanced,
    /// All optimizations are applied, with all tradeoffs in favor of runtime performance
    Max,
    /// Most optimizations are applied, but tuned to trade runtime performance for code size
    Size,
    /// Only optimizations which reduce code size are applied
    SizeMin,
}

/// This enum describes what type of debugging information to emit in compiled programs
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(clap::ValueEnum))]
pub enum DebugInfo {
    /// Do not emit debug info in the final output
    None,
    /// Emit source location information in the final output
    #[default]
    Line,
    /// Emit all available debug information in the final output
    Full,
}

/// Represents the `--remap-path-prefix` flag, which rewrites source paths encoded in debug info
/// with a different path, typically to avoid encoding machine-specific details in artifacts.
#[derive(Debug, Clone)]
pub struct RemapPathPrefix {
    /// The path prefix to remap
    pub from: Box<crate::Path>,
    /// The remapped path prefix
    ///
    /// If `None`, the value `.` is used, representing the current working directory
    pub to: Option<Box<crate::Path>>,
}

impl RemapPathPrefix {
    pub fn source_prefix(&self) -> &crate::Path {
        &self.from
    }

    pub fn target_prefix(&self) -> &crate::Path {
        self.to.as_deref().unwrap_or(crate::Path::new(""))
    }
}

/// Parses `--remap-path-prefix=<from>`, `--remap-path-prefix=<from>=<to>`
#[doc(hidden)]
#[derive(Clone)]
#[cfg(feature = "std")]
pub struct RemapPathPrefixParser;

#[cfg(feature = "std")]
impl clap::builder::TypedValueParser for RemapPathPrefixParser {
    type Value = RemapPathPrefix;

    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        _arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::error::Error> {
        use clap::error::{Error, ErrorKind};

        let input = value.to_str().ok_or_else(|| Error::new(ErrorKind::InvalidUtf8))?;

        Ok(match input.split_once('=') {
            Some((from, to)) => RemapPathPrefix {
                from: PathBuf::from(from.trim()).into_boxed_path(),
                to: Some(PathBuf::from(to.trim()).into_boxed_path()),
            },
            None => RemapPathPrefix {
                from: PathBuf::from(input.trim()).into_boxed_path(),
                to: None,
            },
        })
    }
}

#[cfg(feature = "std")]
fn current_dir() -> PathBuf {
    std::env::current_dir().expect("could not get working directory")
}

#[cfg(not(feature = "std"))]
fn current_dir() -> PathBuf {
    PathBuf::from(".")
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn ir_filters_map_onto_op_filters() {
        use runic_core::pass::OpFilter;

        assert!(matches!(OpFilter::from(IrFilter::Any), OpFilter::All));
        assert!(matches!(OpFilter::from(IrFilter::Symbol(None)), OpFilter::Symbol(None)));
    }

    #[test]
    fn print_after_all_and_after_pass_are_mutually_exclusive() {
        let options = Options {
            print_ir_after_all: true,
            print_ir_after_pass: vec!["canonicalize".to_string()],
            ..Options::default()
        };
        assert!(options.ir_printing_config().is_err());
    }

    #[test]
    fn the_any_filter_is_not_a_printing_filter() {
        let options = Options {
            print_ir_filters: vec![IrFilter::Any, IrFilter::Symbol(None)],
            ..Options::default()
        };
        let config = options.ir_printing_config().unwrap();
        assert_eq!(config.print_ir_filters.len(), 1);
    }

    #[test]
    fn source_locations_flag_reaches_the_printer_flags() {
        let options = Options {
            print_hir_source_locations: true,
            ..Options::default()
        };
        assert!(options.op_printing_flags().print_source_locations);
    }
}
