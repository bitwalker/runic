//! The harness path: an input that a registered frontend reads, or that a flag of the compiler
//! asks for, goes through the compiler harness instead of being parsed as IR.
//!
//! [compiler] is the compiler the tool registers its stages in. [Harness] is what a command line
//! asks of one compilation, read once: every case of the input is compiled with a request made
//! from it.

use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    str::FromStr,
};

#[cfg(feature = "toy")]
use runic_compile::testing::{AltVmBackend, CalcFrontend, StackVmBackend};
use runic_compile::{
    CompilationRequest, Compiler, EmitRequest, EmitTo, Emitted, Input, IrFrontend, IrOptions,
    OptLevel, RecordingObserver, Verification,
};
use runic_driver::{Exit, StageFlags};
use runic_ir::Report;

use crate::{
    Diagnostics,
    options::{Options, Verification as OptVerification},
};

/// The stop point of a compilation that has no target and was given none.
const DEFAULT_STOP: &str = "ir.optimized";

/// The compiler the tool compiles with: the frontend for textual IR, and the frontends and
/// backends registered behind the features of this crate.
///
/// This is where the frontends and backends of the tree are registered, each behind a cargo
/// feature of its own. The `toy` feature registers the toy language `calc` and the toy machines
/// `stackvm` and `altvm`, for the tests and the lit suite. The compiler has no default target.
pub fn compiler() -> Compiler {
    let builder = Compiler::builder().frontend(IrFrontend);
    #[cfg(feature = "toy")]
    let builder = builder.frontend(CalcFrontend).backend(StackVmBackend).backend(AltVmBackend);
    builder.build().expect("the stages the tool registers form a valid compiler")
}

/// The passes the registered stages add to the registry, by name: `--list-passes` leaves them
/// out, since they belong to the stages and not to the tool.
#[cfg(feature = "toy")]
pub(crate) const STAGE_PASSES: &[&str] = &["stackvm-legalize"];
/// The passes the registered stages add to the registry, by name: none.
#[cfg(not(feature = "toy"))]
pub(crate) const STAGE_PASSES: &[&str] = &[];

/// The name of the frontend that reads textual IR.
fn ir_language() -> &'static str {
    runic_compile::Frontend::name(&IrFrontend)
}

/// The flag or input that sends the input through the compiler, in words, or `None` when the
/// input is read as IR.
///
/// The input goes through the compiler when `--target`, `--backend`, `--stop-after`, `--emit` or
/// `--language` is written, when a flag of a stage or a `-Z` entry is (`stage_flag`), or when the
/// input is a file whose extension belongs to a registered frontend other than the one that reads
/// textual IR.
pub(crate) fn reason(compiler: &Compiler, options: &Options, stage_flag: bool) -> Option<String> {
    let written = [
        (options.target.is_some(), "--target"),
        (options.backend.is_some(), "--backend"),
        (options.stop_after.is_some(), "--stop-after"),
        (!options.emit.is_empty(), "--emit"),
        (options.language.is_some(), "--language"),
        (stage_flag, "a flag or a '-Z' entry of a frontend or a backend"),
    ];
    if let Some((_, flag)) = written.iter().find(|(written, _)| *written) {
        return Some((*flag).to_owned());
    }
    let input = options.input.as_deref().filter(|input| *input != "-")?;
    let extension = Path::new(input).extension()?.to_str()?;
    let frontend = compiler.frontend_for_extension(extension)?;
    (frontend.name() != ir_language())
        .then(|| format!("'{input}' is read by the frontend '{}'", frontend.name()))
}

/// What a command line asks of the compilation of each case of an input.
pub(crate) struct Harness<'c> {
    compiler: &'c Compiler,
    /// What the flags say, with no input yet
    request: CompilationRequest,
    /// The language of the input, when the flags or the input's name do not settle it
    language: Option<String>,
    /// The name of the request's unit: the stem of the input file, or `stdin`
    unit: String,
    /// The files `--emit` writes, as the user wrote each entry, the checkpoint and the path,
    /// checked by [Harness::check]
    emit: Vec<(String, String, PathBuf)>,
    /// The outputs every case's request asks for, set by [Harness::resolve]
    outputs: Vec<EmitRequest>,
    /// The files written from the result of a case: those `--emit` names for the stop point, set
    /// by [Harness::resolve]
    deferred: Vec<PathBuf>,
    /// What brought the input here, for the errors that say a flag does not apply to it
    reason: String,
}

impl<'c> Harness<'c> {
    /// The request the flags of `options` and the stages' flags in `matches` make, for the input
    /// named `input` (`None` for standard input). `reason` is what sent the input here.
    ///
    /// Fails with the help `-Z help` prints or the usage error of a stage's flags.
    pub(crate) fn new(
        compiler: &'c Compiler,
        options: &Options,
        flags: &StageFlags<'_>,
        matches: &clap::ArgMatches,
        reason: String,
    ) -> Result<Self, Exit> {
        let mut request = CompilationRequest::new();
        request.target = options.target.clone();
        request.backend = options.backend.clone();
        // No pass runs unless the pipeline names one: the empty pipeline, not the level's.
        request.opt_level = OptLevel::None;
        request.pipeline = Some(options.pass_pipeline.clone().unwrap_or_default());
        request.ir = IrOptions {
            print_before: options.print_ir_before.clone(),
            print_after: options.print_ir_after.clone(),
            print_generic: options.print_generic,
            print_locations: options.print_locations,
            verification: match options.verification() {
                OptVerification::EachPass => Verification::EachPass,
                OptVerification::Once => Verification::Once,
                OptVerification::None => Verification::None,
            },
            dump_pass_pipeline: options.dump_pass_pipeline,
        };
        request.stop_after = match (&options.stop_after, &options.target) {
            (Some(name), _) => Some(name.clone()),
            (None, None) => Some(DEFAULT_STOP.to_owned()),
            (None, Some(_)) => None,
        };
        flags.store(matches, &mut request)?;

        let stdin = matches!(options.input.as_deref(), None | Some("-"));
        let language = match &options.language {
            Some(language) => Some(language.clone()),
            None if stdin => Some(ir_language().to_owned()),
            None => None,
        };
        let unit = match options.input.as_deref() {
            Some(path) if !stdin => Path::new(path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("input")
                .to_owned(),
            _ => "stdin".to_owned(),
        };
        Ok(Self {
            compiler,
            request,
            language,
            unit,
            emit: Vec::new(),
            outputs: Vec::new(),
            deferred: Vec::new(),
            reason,
        })
    }

    /// Checks what the harness path does not do, and reads the `--emit` entries.
    ///
    /// Each is an error naming what the user wrote: a flag the compiler's settings have no place
    /// for, `--emit` with `--split-input-file`, and an `--emit` entry that is not `NAME=PATH`.
    pub(crate) fn check(&mut self, options: &Options) -> Result<(), Report> {
        for (written, flag) in [
            (options.verify_roundtrip, "--verify-roundtrip"),
            (options.print_ir_after_change, "--print-ir-after-change"),
            (options.print_ir_after_failure, "--print-ir-after-failure"),
        ] {
            if written {
                return Err(Report::msg(format!(
                    "{flag} is not available when the input goes through a frontend or a backend, \
                     as it does here: {}",
                    self.reason
                )));
            }
        }
        if !options.emit.is_empty() && options.split_input_file.is_some() {
            return Err(Report::msg(
                "--emit is not available with --split-input-file: every case would write the same \
                 file",
            ));
        }
        for text in &options.emit {
            let needs_path = || {
                Report::msg(format!(
                    "--emit {text}: the checkpoint needs a file to be written to: write '--emit \
                     NAME=PATH'"
                ))
            };
            let request = EmitRequest::from_str(text)
                .map_err(|error| Report::msg(format!("--emit {text}: {error}")))?;
            match request.to {
                EmitTo::File(path) => self.emit.push((text.clone(), request.checkpoint, path)),
                _ => return Err(needs_path()),
            }
        }
        Ok(())
    }

    /// Compiles `text`, the case of the input named `name`, and gives what it reports to
    /// `diagnostics`. The result, when the compilation succeeded, is put in `stdout`. Returns
    /// whether the case succeeded.
    pub(crate) fn process_into(
        &self,
        name: &str,
        text: String,
        stdout: &mut Vec<u8>,
        diagnostics: &mut Diagnostics,
    ) -> bool {
        let mut request = self.request.clone();
        request.name = Some(self.unit.clone());
        let mut input = Input::memory(name, text);
        input.language = self.language.clone();
        request.inputs.push(input);

        request.emit = self.outputs.clone();

        let sink = Box::new(diagnostics.stderr.clone());
        let compilation =
            self.compiler.compile_to(request, &mut RecordingObserver::default(), sink);
        diagnostics.compilation(&compilation);
        if !compilation.succeeded() || compilation.diagnostics().has_errors() {
            return false;
        }

        let result = compilation.emitted().iter().find_map(|emitted| match emitted {
            Emitted::Memory { bytes, .. } => Some(bytes.as_slice()),
            _ => None,
        });
        let bytes = result.unwrap_or_default();
        // A file for the checkpoint the result is of is written from the result.
        for path in &self.deferred {
            if let Err(error) = runic_compile::write_file(path, bytes) {
                diagnostics.failure(&Report::msg(format!(
                    "could not write '{}': {error}",
                    path.display()
                )));
                return false;
            }
        }
        stdout.extend_from_slice(bytes);
        true
    }

    /// Resolves the request the cases will use, once, before the input is read: the flags with a
    /// placeholder memory input named `name`, through [Compiler::session]. A failure is what every
    /// case would have reported: an unknown target, backend, checkpoint or language, a pipeline
    /// that is not anchored on the world, an `--emit` of a checkpoint the compilation does not
    /// reach.
    ///
    /// Keeps what a case needs of it: the outputs its request asks for, and the files written from
    /// its result. The result is the artifact at the stop point, delivered to memory. A request has
    /// one output for a checkpoint, so an `--emit` entry for the stop point is not given to the
    /// compilation: its file is written from the result.
    ///
    /// Two `--emit` entries that name one checkpoint, by id or by alias, are an error naming both.
    pub(crate) fn resolve(&mut self, name: &str) -> Result<(), Report> {
        let mut request = self.request.clone();
        request.name = Some(self.unit.clone());
        let mut input = Input::memory(name, String::new());
        input.language = self.language.clone();
        request.inputs.push(input);

        let session = self.compiler.session(request.clone())?;
        let stop = session.goal().stop_after();
        let route = session.route();
        let mut emit = vec![EmitRequest {
            checkpoint: stop.as_str().to_owned(),
            to: EmitTo::Memory,
        }];
        let mut deferred = Vec::new();
        // The entries seen so far: the position of the checkpoint on the route, when it has one,
        // the entry as written and its checkpoint as written.
        let mut seen: Vec<(Option<usize>, &str, &str)> = Vec::new();
        for (text, checkpoint, path) in &self.emit {
            let at = route.resolve(checkpoint);
            let first = seen.iter().find(|(other, _, written)| match (at, other) {
                (Some(at), Some(other)) => at == *other,
                _ => written == checkpoint,
            });
            if let Some((_, first, _)) = first {
                return Err(Report::msg(format!(
                    "--emit {first} and --emit {text} name the same checkpoint: a checkpoint is \
                     written to one file"
                )));
            }
            seen.push((at, text, checkpoint));
            if at.is_some_and(|at| route.get(at).id == stop) {
                deferred.push(path.clone());
            } else {
                emit.push(EmitRequest {
                    checkpoint: checkpoint.clone(),
                    to: EmitTo::File(path.clone()),
                });
            }
        }
        drop(session);

        // The outputs are resolved too, so that one the compilation cannot deliver is found here.
        request.emit = emit.clone();
        self.compiler.session(request)?;
        self.outputs = emit;
        self.deferred = deferred;
        Ok(())
    }
}

/// The stages the compiler registers, as `--list-stages` prints them: the frontends with their
/// extensions, the backends with their targets, and the checkpoints in the order a compilation
/// reaches them.
pub(crate) fn list_stages(compiler: &Compiler, out: &mut String) {
    let _ = writeln!(out, "frontends:");
    for frontend in compiler.frontends() {
        let extensions: Vec<_> =
            frontend.extensions().iter().map(|ext| format!(".{ext}")).collect();
        let _ = writeln!(out, "  {}: {}", frontend.name(), extensions.join(", "));
    }
    let _ = writeln!(out, "backends:");
    for backend in compiler.backends() {
        let targets: Vec<_> = backend.targets().iter().map(ToString::to_string).collect();
        let _ = writeln!(out, "  {}: {}", backend.name(), targets.join(", "));
    }
    let _ = writeln!(out, "checkpoints:");
    let frontends = compiler.frontends().flat_map(|frontend| frontend.checkpoints());
    let backends = compiler.backends().flat_map(|backend| backend.checkpoints());
    let harness = runic_compile::IR_CHECKPOINTS.iter();
    let (first, last): (Vec<_>, Vec<_>) = (frontends.collect(), backends.collect());
    for decl in first.into_iter().chain(harness).chain(last) {
        let aliases = match decl.aliases {
            [] => String::new(),
            [one] => format!(" (alias {one})"),
            many => format!(" (aliases {})", many.join(", ")),
        };
        let _ = writeln!(out, "  {}{aliases} -> .{}", decl.id, decl.extension);
    }
}
