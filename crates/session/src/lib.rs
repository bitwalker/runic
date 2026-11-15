#![no_std]
#![feature(debug_closure_helpers)]
#![feature(specialization)]
#![cfg_attr(not(feature = "std"), feature(slice_split_once))]
// Specialization
#![allow(incomplete_features)]
#![deny(warnings)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use alloc::format;
use alloc::{
    borrow::ToOwned,
    string::{String, ToString},
};

mod color;
mod emit;
pub mod flags;
mod inputs;
mod libs;
mod options;
mod outputs;
pub mod path;
#[cfg(feature = "std")]
mod statistics;

use alloc::{boxed::Box, fmt};

/// The version associated with the current compiler toolchain
pub const RUNIC_BUILD_VERSION: &str = env!("RUNIC_BUILD_VERSION");

/// The git revision associated with the current compiler toolchain
pub const RUNIC_BUILD_REV: &str = env!("RUNIC_BUILD_REV");

pub use runic_core::diagnostics;
use runic_core::diagnostics::Report;
#[cfg(feature = "std")]
pub use runic_core::pass::HumanDuration;
use runic_symbol::Symbol;

pub use self::{
    color::ColorChoice,
    emit::{Emit, Writer},
    flags::{ArgMatches, CompileFlag, CompileFlags, FlagAction},
    inputs::{FileName, InputFile, InputType},
    libs::LinkLibrary,
    options::*,
    outputs::{OutputFile, OutputFiles, OutputMode, OutputType, OutputTypeSpec, OutputTypes},
    path::{Path, PathBuf},
};
#[cfg(feature = "std")]
pub use self::{emit::EmitExt, statistics::Statistics};

/// This struct provides access to all of the metadata and configuration
/// needed during a single compilation session.
#[derive(Clone)]
pub struct Session {
    /// The name of this session
    pub name: String,
    /// Configuration for the current compiler session
    pub options: Box<Options>,
    /// The inputs being compiled
    pub input: Option<InputFile>,
    /// The outputs to be produced by the compiler during compilation
    pub output_files: OutputFiles,
    /// Statistics gathered from the current compiler session
    #[cfg(feature = "std")]
    pub statistics: Statistics,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("name", &self.name)
            .field("options", &self.options)
            .field("inputs", &self.input)
            .field("output_files", &self.output_files)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Open a session compiling `input` under `options`.
    ///
    /// The session is named by `options.name` if set; otherwise by the stem of the requested
    /// output file; otherwise by the input.
    pub fn new(input: InputFile, options: Box<Options>) -> Result<Self, Report> {
        let name = options
            .name
            .clone()
            .or_else(|| {
                log::debug!(target: "driver", "no name specified, attempting to derive from output file");
                options.output_file.as_ref().and_then(|of| of.filestem().map(|stem| stem.to_string()))
            })
            .unwrap_or_else(|| {
                log::debug!(target: "driver", "unable to derive name from output file, deriving from input");
                match &input {
                    InputFile {
                        file: InputType::Real(path),
                    } => path
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .or_else(|| path.extension().and_then(|stem| stem.to_str()))
                        .unwrap_or_else(|| {
                            panic!(
                                "invalid input path: '{}' has no file stem or extension",
                                path.display()
                            )
                        })
                        .to_string(),
                    input @ InputFile {
                        file: InputType::Stdin { name, .. },
                    } => {
                        let name = name.as_str();
                        if matches!(name, "empty" | "stdin") {
                            log::debug!(target: "driver", "no good input file name to use, using current directory base name");
                            options
                                .current_dir
                                .file_stem()
                                .and_then(|stem| stem.to_str())
                                .unwrap_or(name)
                                .to_string()
                        } else {
                            input.filestem().to_owned()
                        }
                    }
                }
            });
        log::debug!(target: "driver", "artifact name set to '{name}'");

        log::debug!(target: "driver", "creating session {name}");
        if log::log_enabled!(target: "driver", log::Level::Debug) {
            log::debug!(target: "driver", " | input = {}", input.file_name());
            log::debug!(
                target: "driver",
                " | outputs_dir = {}",
                options.output_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or("<unset>".to_string())
            );
            log::debug!(
                target: "driver",
                " | output_file = {}",
                options.output_file.as_ref().map(|of| of.to_string()).unwrap_or("<unset>".to_string())
            );
            log::debug!(target: "driver", " | target_dir = {}", options.target_dir.display());
        }

        create_target_dir(&options.target_dir);

        let output_dir = options
            .output_dir
            .as_deref()
            .or_else(|| options.output_file.as_ref().and_then(|of| of.parent()))
            .map(|path| path.to_path_buf())
            .unwrap_or_else(|| options.target_dir.clone());
        create_target_dir(&output_dir);

        log::debug!(target: "driver", " | output dir = {}", output_dir.display());
        if log::log_enabled!(target: "driver", log::Level::Debug) {
            for lib in options.link_libraries.iter() {
                if let Some(path) = lib.path.as_deref() {
                    log::debug!(target: "driver", " | linking library '{}' from {}", lib.name, path.display());
                } else {
                    log::debug!(target: "driver", " | linking library '{}'", lib.name);
                }
            }
        }

        let output_files = OutputFiles::new(
            name.clone(),
            options.current_dir.clone(),
            output_dir,
            options.output_file.clone(),
            options.target_dir.clone(),
            options.output_types.clone(),
        );

        Ok(Self {
            name,
            options,
            input: Some(input),
            output_files,
            #[cfg(feature = "std")]
            statistics: Default::default(),
        })
    }

    #[doc(hidden)]
    pub fn with_output_type(mut self, ty: OutputType, path: Option<OutputFile>) -> Self {
        self.output_files.outputs.insert(ty, path.clone());
        self.options.output_types.insert(ty, path.clone());
        self
    }

    #[doc(hidden)]
    pub fn with_extra_flags(mut self, flags: CompileFlags) -> Self {
        self.options.set_extra_flags(flags);
        self
    }

    /// Get the value of a custom flag with action `FlagAction::SetTrue` or `FlagAction::SetFalse`
    #[inline]
    pub fn get_flag(&self, name: &str) -> bool {
        self.options.flags.get_flag(name)
    }

    /// Get the count of a specific custom flag with action `FlagAction::Count`
    #[inline]
    pub fn get_flag_count(&self, name: &str) -> usize {
        self.options.flags.get_flag_count(name)
    }

    /// Get the remaining [ArgMatches] left after parsing the base session configuration
    #[inline]
    pub fn matches(&self) -> &ArgMatches {
        self.options.flags.matches()
    }

    /// The name of this session (used as the name of the project, output file, etc.)
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the [OutputFile] to write the final output to
    pub fn out_file(&self) -> OutputFile {
        let out_file = self.output_files.output_file(OutputType::Ir, None);

        if let OutputFile::Real(ref path) = out_file {
            self.check_file_is_writeable(path);
        }

        out_file
    }

    #[cfg(not(feature = "std"))]
    fn check_file_is_writeable(&self, file: &Path) {
        panic!(
            "Compiler exited with a fatal error: cannot write '{}' - compiler was built without \
             standard library",
            file.display()
        );
    }

    #[cfg(feature = "std")]
    fn check_file_is_writeable(&self, file: &Path) {
        if let Ok(m) = file.metadata()
            && m.permissions().readonly()
        {
            panic!("Compiler exited with a fatal error: file is not writeable: {}", file.display());
        }
    }

    /// Returns true if the compiler should exit after parsing the input
    pub fn parse_only(&self) -> bool {
        self.options.parse_only
    }

    /// Returns true if the compiler should exit after performing semantic analysis
    pub fn analyze_only(&self) -> bool {
        self.options.analyze_only
    }

    /// Returns true if the compiler should exit after applying rewrites to the IR
    pub fn rewrite_only(&self) -> bool {
        !self.options.parse_only && !self.options.analyze_only
    }

    /// Returns true if the given [OutputType] should be emitted as an output
    pub fn should_emit(&self, ty: OutputType) -> bool {
        self.options.output_types.contains_key(&ty)
    }

    /// Returns true if IR should be printed to stdout after the pass whose command-line name is
    /// `pass`: after every pass with `print_ir_after_all`, else after the passes named in
    /// `print_ir_after_pass`
    pub fn should_print_ir(&self, pass: &str) -> bool {
        self.options.print_ir_after_all
            || self.options.print_ir_after_pass.iter().any(|p| p == pass)
    }

    /// Returns true if IR should be printed to stdout, at the start of `stage`
    pub fn should_print_ir_before_stage(&self, stage: &str) -> bool {
        self.options.print_ir_before_stage.iter().any(|s| s == stage)
    }

    /// Returns true if CFG should be printed to stdout, after executing a pass named `pass`
    pub fn should_print_cfg(&self, pass: &str) -> bool {
        self.options.print_cfg_after_all
            || self.options.print_cfg_after_pass.iter().any(|p| p == pass)
    }

    /// Print the given emittable IR to stdout, as produced by a pass with name `pass`
    #[cfg(feature = "std")]
    pub fn print(&self, ir: impl Emit, pass: &str) -> Result<(), Report> {
        if self.should_print_ir(pass) {
            ir.write_to_stdout(self)?;
        }
        Ok(())
    }

    /// Get the path to emit the given [OutputType] to
    pub fn emit_to(&self, ty: OutputType, name: Option<Symbol>) -> Option<PathBuf> {
        if self.should_emit(ty) {
            match self.output_files.output_file(ty, name.map(|n| n.as_str())) {
                OutputFile::Real(path) => Some(path),
                OutputFile::Directory(_) => {
                    unreachable!("OutputFiles::output_file never returns OutputFile::Directory")
                }
                OutputFile::Stdout => None,
            }
        } else {
            None
        }
    }

    /// Emit an item to stdout/file system depending on the current configuration
    #[cfg(feature = "std")]
    pub fn emit<E: Emit>(&self, mode: OutputMode, item: &E) -> Result<(), Report> {
        let output_type = item.output_type(mode);
        let name = item.name().map(|n| n.as_str());
        match self.output_path_for(output_type, name) {
            Some(OutputFile::Real(path)) => {
                item.write_to_file(&path, mode, self)?;
            }
            Some(OutputFile::Directory(_)) => {
                unreachable!("OutputFiles::output_file never returns OutputFile::Directory")
            }
            Some(OutputFile::Stdout) => {
                let stdout = std::io::stdout().lock();
                item.write_to(stdout, mode, self)?;
            }
            None => (),
        }

        Ok(())
    }

    /// Given an [OutputType] and an optional name, return the output file path that would be
    /// written to.
    ///
    /// Returns `Some` if the specified output type should be emitted, and `None` if it should not.
    #[cfg(feature = "std")]
    pub fn output_path_for(
        &self,
        output_type: OutputType,
        name: Option<&str>,
    ) -> Option<OutputFile> {
        if self.should_emit(output_type) {
            Some(self.output_files.output_file(output_type, name))
        } else {
            None
        }
    }

    #[cfg(not(feature = "std"))]
    pub fn emit<E: Emit>(&self, _mode: OutputMode, _item: &E) -> Result<(), Report> {
        Ok(())
    }
}

/// The system temporary directory, resolved to its canonical form.
///
/// Callers want the resolved form because on macOS [`std::env::temp_dir`] is not it: `TMPDIR` is
/// a symlink into `/private/var`, so the temporary directory and the paths `rustc` reports in
/// debug information disagree until one of them is canonicalized.
///
/// Resolving it reads the filesystem, and fails when the directory does not exist or cannot be
/// traversed. [`std::env::temp_dir`] hands back whatever `TMPDIR` (or `TEMP`/`TMP` on Windows)
/// names without checking that anything is there, so an unusable setting used to reach a panic
/// on this path. Both call sites return a `Result` and report every other filesystem failure
/// they hit, so this one reports too.
#[cfg(feature = "std")]
pub fn canonical_temp_dir() -> Result<PathBuf, Report> {
    canonicalize_directory(&std::env::temp_dir())
}

/// The absolute, symlink-resolved form of `dir`, or why it could not be resolved.
#[cfg(feature = "std")]
fn canonicalize_directory(dir: &Path) -> Result<PathBuf, Report> {
    dir.canonicalize().map_err(|err| {
        Report::msg(format!("unable to canonicalize directory '{}': {err}", dir.display()))
    })
}

#[cfg(feature = "std")]
fn create_target_dir(path: &Path) {
    if !path.exists() {
        std::fs::create_dir_all(path).unwrap_or_else(|err| {
            panic!("unable to create --target-dir '{}': {err}", path.display())
        });
    }
}

#[cfg(not(feature = "std"))]
fn create_target_dir(_path: &Path) {}

#[cfg(all(test, feature = "std"))]
mod tests {
    use tempfile::TempDir;

    use super::*;

    // `canonical_temp_dir` reads `TMPDIR` from the process environment, which no single test can
    // own; `canonicalize_directory` is the whole filesystem decision it makes, and is what a
    // missing temporary directory exercises.
    #[test]
    fn canonicalizing_a_directory_distinguishes_a_real_one_from_a_missing_one() {
        let temp = TempDir::new().unwrap();
        let missing = temp.path().join("no-such-child");

        assert_eq!(
            canonicalize_directory(temp.path()).unwrap(),
            temp.path().canonicalize().unwrap(),
            "a directory that exists resolves to its canonical form"
        );

        let error = canonicalize_directory(&missing).unwrap_err().to_string();
        assert!(
            error.starts_with("unable to canonicalize directory"),
            "a missing directory is a diagnostic naming the failed operation, got: {error}"
        );
    }

    #[test]
    fn the_process_temporary_directory_canonicalizes() {
        let temp_dir = canonical_temp_dir().unwrap();
        assert!(
            temp_dir.is_absolute(),
            "canonical form is absolute, got: {}",
            temp_dir.display()
        );
    }

    #[test]
    fn a_session_is_named_after_its_input_and_creates_its_target_directory() {
        let temp = TempDir::new().unwrap();
        let options = Options {
            current_dir: temp.path().to_path_buf(),
            target_dir: temp.path().join("target"),
            ..Options::default()
        };
        let input = InputFile::new(InputType::Real("some/foo.runic".into()));
        let session = Session::new(input, Box::new(options)).unwrap();

        assert_eq!(session.name(), "foo");
        assert!(temp.path().join("target").is_dir());
    }

    #[test]
    fn an_explicit_name_wins_over_the_input() {
        let temp = TempDir::new().unwrap();
        let options = Options {
            name: Some("explicit".into()),
            current_dir: temp.path().to_path_buf(),
            target_dir: temp.path().join("target"),
            ..Options::default()
        };
        let input = InputFile::new(InputType::Real("foo.runic".into()));
        let session = Session::new(input, Box::new(options)).unwrap();

        assert_eq!(session.name(), "explicit");
    }
}

/// The diagnostic derives resolve their runtime through `runic_core::diagnostics` for a crate,
/// like this one, that depends on `runic-core` but not on `runic-diagnostics`.
#[cfg(test)]
mod diagnostic_derive_tests {
    use runic_core::diagnostics::{
        Diagnostic, LabelStyle, NoteKind, Report, SourceId, SourceNamespace, SourceSpan, Spanned,
        TextRange, prepare_ref,
    };

    #[derive(Debug, Spanned)]
    struct Named {
        #[span]
        span: SourceSpan,
    }

    #[derive(Debug, Diagnostic)]
    #[diagnostic(message = "unknown name", help = "declare it before use")]
    struct UnknownName {
        #[label(primary, "not found")]
        name: Named,
    }

    #[test]
    fn derived_diagnostics_resolve_through_runic_core() {
        let span = SourceSpan::session(
            SourceId::new(SourceNamespace::new_unchecked(1), 0),
            TextRange::new(2, 5).unwrap(),
        );
        let name = Named { span };
        assert_eq!(name.span(), span);

        let report = Report::from(UnknownName { name });
        let snapshot = prepare_ref(report.as_diagnostic()).unwrap();
        assert_eq!(snapshot.message, "unknown name");
        assert_eq!(snapshot.labels.len(), 1);
        assert_eq!(snapshot.labels[0].span, span);
        assert_eq!(snapshot.labels[0].style, LabelStyle::Primary);
        assert_eq!(snapshot.labels[0].message.as_deref(), Some("not found"));
        assert!(
            snapshot
                .notes
                .iter()
                .any(|note| note.kind == NoteKind::Help && note.message == "declare it before use")
        );
    }
}
