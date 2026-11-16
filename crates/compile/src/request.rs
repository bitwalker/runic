use core::any::Any;
use std::path::PathBuf;

use runic_core::Report;

use crate::Triple;

mod input;
mod levels;
mod options;

pub use self::{
    input::{EmitRequest, EmitTo, Extern, Input, InputSource, OutputFile, RemapPathPrefix},
    levels::{DebugInfo, IrOptions, OptLevel, Verification},
    options::ExtensionOptions,
};

/// Everything a caller can ask of one compilation.
///
/// A request is a plain value: building it reads no file, no environment variable and no argument,
/// and nothing in it is checked until the harness runs it. Paths are kept as written. A command
/// line fills it flag by flag; a program that embeds the compiler fills it directly. The doc of
/// each field names the flag that sets it.
#[derive(Debug, Clone, Default)]
pub struct CompilationRequest {
    /// The name of the unit being compiled (`--name`). When unset, the first input's file stem
    /// names it; see [CompilationRequest::unit_name]. Output file names derive from it.
    pub name: Option<String>,
    /// The inputs (positional arguments), in the order given. Frontends translate them.
    pub inputs: Vec<Input>,
    /// The target (`--target`). The backend that supports it lowers the IR for it.
    pub target: Option<Triple>,
    /// The backend, by name (`--backend`), when the target does not settle it.
    pub backend: Option<String>,
    /// The optimization level (`--opt-level`, `-O`). The optimization pipeline and the backend
    /// read it.
    pub opt_level: OptLevel,
    /// The debug information level (`--debug`, `-g`). Frontends and the backend read it.
    pub debug: DebugInfo,
    /// The named dependencies (`--extern`). The harness does not open them: frontends and the
    /// backend read them.
    pub externs: Vec<Extern>,
    /// The directories searched for what an extern or a source names without a path
    /// (`--search-path`, `-L`). Frontends and the backend read them.
    pub search_paths: Vec<PathBuf>,
    /// The checkpoints whose artifacts to write (`--emit`). They never change how far the
    /// compilation runs.
    pub emit: Vec<EmitRequest>,
    /// The checkpoint after which to stop (`--stop-after`), by id or alias. It alone decides how
    /// far the compilation runs.
    pub stop_after: Option<String>,
    /// Where the output goes when exactly one output has no destination of its own
    /// (`--output-file`, `-o`).
    pub output_file: Option<OutputFile>,
    /// The directory that outputs without a destination of their own go in (`--output-dir`).
    pub output_dir: Option<PathBuf>,
    /// The optimization pipeline, in textual form (`-Z pass-pipeline`), replacing the one the
    /// optimization level would choose.
    pub pipeline: Option<String>,
    /// What is printed and checked while passes run (`-Z print-ir-before` and the like).
    pub ir: IrOptions,
    /// Prefix rewrites for paths recorded in debug information (`--remap-path-prefix`). Frontends
    /// and the backend read them.
    pub remap_path_prefixes: Vec<RemapPathPrefix>,
    options: ExtensionOptions,
}

impl CompilationRequest {
    /// A request that asks for nothing yet: no inputs, no target, default levels.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores the options of a frontend or backend, replacing earlier ones of the same type.
    pub fn set_options<T: Any>(&mut self, value: T) {
        self.options.set(value);
    }

    /// The options of type `T`, if some were set.
    pub fn options<T: Any>(&self) -> Option<&T> {
        self.options.get::<T>()
    }

    /// The name of the unit: [CompilationRequest::name] if set, else the first input's file stem.
    ///
    /// It is an error for the name to be empty, and for there to be neither a name nor an input
    /// with a stem to derive one from.
    pub fn unit_name(&self) -> Result<String, Report> {
        if let Some(name) = &self.name {
            if name.is_empty() {
                return Err(Report::msg("invalid unit name '': it is empty"));
            }
            return Ok(name.clone());
        }
        match self.inputs.first().and_then(Input::stem) {
            Some(stem) => Ok(stem.to_string()),
            None => Err(Report::msg(
                "no name was given for the compilation and none can be derived from its inputs; \
                 give one with --name",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_request_asks_for_nothing_and_has_the_default_levels() {
        let request = CompilationRequest::new();
        assert!(request.inputs.is_empty() && request.emit.is_empty());
        assert_eq!(request.target, None);
        assert_eq!(request.opt_level, OptLevel::Balanced);
        assert_eq!(request.debug, DebugInfo::Line);
        assert_eq!(request.stop_after, None);
    }

    #[test]
    fn a_given_name_is_the_unit_name() {
        let mut request = CompilationRequest::new();
        request.name = Some("app".into());
        request.inputs.push(Input::path("other.wasm"));
        assert_eq!(request.unit_name().unwrap(), "app");
    }

    #[test]
    fn the_unit_name_comes_from_the_first_path_input() {
        let mut request = CompilationRequest::new();
        request.inputs.push(Input::path("src/main.wasm"));
        request.inputs.push(Input::path("src/second.wasm"));
        assert_eq!(request.unit_name().unwrap(), "main");
    }

    #[test]
    fn the_unit_name_comes_from_the_name_of_a_memory_input() {
        let mut request = CompilationRequest::new();
        request.inputs.push(Input::memory("snippet.wat", b"".to_vec()));
        assert_eq!(request.unit_name().unwrap(), "snippet");
    }

    #[test]
    fn with_no_name_and_no_input_the_error_says_how_to_give_one() {
        let message = CompilationRequest::new().unit_name().unwrap_err().to_string();
        assert!(message.contains("no name was given") && message.contains("--name"), "{message}");
    }

    #[test]
    fn an_input_with_no_stem_cannot_name_the_unit() {
        let mut request = CompilationRequest::new();
        request.inputs.push(Input::memory("", Vec::new()));
        assert!(request.unit_name().unwrap_err().to_string().contains("--name"));
    }

    #[test]
    fn an_empty_name_is_an_error() {
        let mut request = CompilationRequest::new();
        request.name = Some(String::new());
        request.inputs.push(Input::path("a.wasm"));
        assert!(request.unit_name().unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn options_set_on_a_request_are_read_back_and_survive_a_clone() {
        #[derive(Debug, PartialEq)]
        struct Frontend(&'static str);
        let mut request = CompilationRequest::new();
        assert_eq!(request.options::<Frontend>(), None);
        request.set_options(Frontend("wasm"));
        let copy = request.clone();
        assert_eq!(copy.options::<Frontend>(), Some(&Frontend("wasm")));
        assert!(format!("{copy:?}").contains("ExtensionOptions(1 entries)"));
    }
}
