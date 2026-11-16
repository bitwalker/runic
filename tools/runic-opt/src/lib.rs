//! `runic-opt` reads textual IR, runs a pass pipeline named on the command line on it, verifies
//! and prints the result.
//!
//! The input is one top-level operation, or with `--split-input-file` several, one per case. Each
//! is parsed in a context of its own, on a thread with a stack large enough for regions nested
//! [`MAX_REGION_DEPTH`] deep (and lists nested [`MAX_LIST_NESTING`] deep); the pass manager is
//! anchored on the operation with implicit nesting. Standard output holds only what was asked for:
//! the printed IR, or a listing. Every diagnostic, and every print of IR between passes, goes to
//! standard error. The exit code is 0 when no case produced an error (a pass that emits one fails
//! its case like a pass that fails) and 1 otherwise. The pipeline's text is checked once, before
//! the input is read.
//!
//! With `--verify-diagnostics` the diagnostics of each case are not rendered: they are checked
//! against the expectations written in the case (see the module `expect`), and what does not match
//! is rendered instead. The exit code is then 1 exactly when something did not match, and a case's
//! result is printed when the case was processed without error, whether or not its diagnostics
//! met the expectations.
//!
//! Besides the passes of the crates it links, the tool registers passes for testing from text
//! what has no pass of its own (in `test_passes`): `test-convert-wide-ints`, `test-fail` and
//! `test-print-int-ranges`.
//!
//! With `--verify-roundtrip`, after the input is parsed and before any pass, both forms of the
//! operation are checked to read back (see the module `roundtrip`); a form that does not is an
//! error of the case.
//!
//! The tool has a second path, through the compiler harness (see the module `harness`). An input
//! that a registered frontend other than the one for textual IR reads by its file extension, or a
//! command line that writes `--language` (`runic` included), `--target`, `--backend`,
//! `--stop-after`, `--emit NAME=PATH`, or a flag of a stage or a `-Z` entry, is compiled by
//! `runic-compile` instead of being parsed as IR: each case becomes a request, and what it emits
//! is written as `--emit` says. The request is resolved once, before the input is read, so what
//! is wrong with it is reported once. The pipeline then runs on the world (`builtin.world`): a
//! whole pipeline is anchored on `builtin.world` or `any`, where on the IR path it is anchored on
//! the input's top-level operation or `any`. Every case goes to the frontend, which decides what
//! an input with no operation is; with `--split-input-file` only a case of whitespace is skipped.
//! `-Z` is accepted only when a registered stage declares unstable options. `--list-stages` lists the registered frontends and backends. With the cargo
//! feature `toy`, the toy language `calc` and the toy machines `stackvm` and `altvm` are
//! registered.
//!
//! [`run`] is the whole tool, given its arguments and streams; the binary calls it with the
//! process's.
// The operations the tests declare are built by the operation derive, which needs these.
#![cfg_attr(test, feature(unboxed_closures, fn_traits))]
#![deny(warnings)]

#[cfg(test)]
extern crate alloc;

mod expect;
mod harness;
mod options;
mod roundtrip;
mod split;
mod test_passes;

use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::OsString,
    fmt, fs,
    io::{Read, Write},
    process::ExitCode,
    rc::Rc,
};

use clap::{CommandFactory, FromArgMatches};
use runic_compile::{Compilation, Compiler};
// The dialects and passes the tool links: naming each crate keeps its registrations.
use runic_dialect_arith as _;
use runic_dialect_cf as _;
use runic_dialect_mem as _;
use runic_dialect_scf as _;
use runic_dialect_ub as _;
use runic_driver::{
    StageFlags,
    worker::{self, SharedBuffer, WorkerError, WorkerPanic},
};
use runic_ir::{
    Context, OpPrinter, OpPrintingFlags, OperationRef, Report,
    diagnostics::{
        DiagnosticDisplay, DiagnosticSet, NoteKind, OwnedDiagnostic, SourceId, SourceSpan,
        TextRange, diagnostic,
    },
    parse::{ParserConfig, parse_any},
    pass::{
        IRPrintingConfig, IoOutput, Nesting, OpPassManager, OperationPass, PassInfo,
        PassInstrumentation, PassManager, PassOptions, PassPipeline, PipelineElement,
        registry::RegistryEntry,
    },
    print::AsmPrinter,
};
use runic_transform as _;

pub use self::harness::compiler;
use self::{
    harness::Harness,
    options::{Options, Verification},
};

/// How deeply regions may nest in the input: text with a region inside more enclosing regions
/// than this is the parser's diagnostic.
pub const MAX_REGION_DEPTH: u32 = 1_000;

/// How deeply list attributes may nest in the input: text with a list inside more enclosing lists
/// than this is the parser's diagnostic.
pub const MAX_LIST_NESTING: u32 = 1_000;

/// The name of the thread an input is processed on. A panic on it is reported by the tool, as an
/// internal error on the standard error it was given (see [`run`]).
const CHUNK_THREAD: &str = "runic-opt chunk";

/// Runs the tool. `args` excludes the program name. Returns the process's exit code.
///
/// The first input processed installs a panic hook for the whole process, which stays installed. It acts
/// only on the threads `run` starts to process the input: it records a panic's message, location
/// and (when `RUST_BACKTRACE` asks for one) backtrace, which the tool then reports as an internal
/// error on `stderr`; a panic that cannot unwind is written, after what the input's thread wrote,
/// to the process's standard error before the process aborts. A panic on any other thread goes to
/// the hook installed before. A host that replaces the hook after the first call loses those
/// reports: a panic while processing an input is then an internal error with its message alone,
/// and the hook it installed also sees that panic.
///
/// When logging is enabled (`RUNIC_TRACE`), the logger writes to the process's standard error from
/// the thread that processes the input, while `run` waits for that thread: `stdout` and `stderr`
/// must then not be locked standard streams (such as `io::stderr().lock()`), or the tool
/// deadlocks.
pub fn run(
    args: impl IntoIterator<Item = OsString>,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> ExitCode {
    run_with_setup(args, stdin, stdout, stderr, || Ok(()))
}

/// [`run`], with `setup` called once the arguments are read and before the tool does anything
/// else: the place for what the process sets up for a run, such as its logger.
///
/// `setup` is not called when the arguments ask for help or the version, or are invalid: asking
/// for help does not depend on it. An error it returns is written to `stderr` as `error: ...`,
/// and the tool fails.
pub fn run_with_setup(
    args: impl IntoIterator<Item = OsString>,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    setup: impl FnOnce() -> Result<(), String>,
) -> ExitCode {
    run_on(&compiler(), args, stdin, stdout, stderr, setup)
}

/// [`run_with_setup`] over `compiler`'s stages instead of the ones the tool registers: for a tool
/// that registers more, and for tests.
///
/// The frontends and backends of `compiler` are the ones that read an input, and generate code
/// for it, on the harness path (see [`compiler`]). `compiler` must register the frontend for
/// textual IR ([`IrFrontend`](runic_compile::IrFrontend)): standard input on the harness path
/// without `--language` is read by it. `--list-passes` leaves out only the passes of the toy
/// stages, whatever stages `compiler` registers.
pub fn run_on(
    compiler: &Compiler,
    args: impl IntoIterator<Item = OsString>,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    setup: impl FnOnce() -> Result<(), String>,
) -> ExitCode {
    let mut failed = false;
    let mut out = Vec::new();
    let mut err = String::new();
    match run_to_buffers(compiler, args, stdin, &mut out, &mut err, setup) {
        Ok(true) => {}
        Ok(false) => failed = true,
        Err(Exit::Help(text)) => out = text.into_bytes(),
        Err(Exit::Usage(text)) => {
            err = text;
            failed = true;
        }
    }
    // Nothing is left to report a failure to write standard error to.
    let _ = stderr.write_all(err.as_bytes()).and_then(|()| stderr.flush());
    if let Err(error) = stdout.write_all(&out).and_then(|()| stdout.flush()) {
        let error = Report::msg(format!("could not write standard output: {error}"));
        let _ = stderr.write_all(render(error.display()).as_bytes());
        failed = true;
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// The internal error for the panics on a chunk's thread, rendered: the first panic's message,
/// where it happened, a note for each later panic, and the first's backtrace.
fn report_panics(panics: &[WorkerPanic]) -> String {
    let Some((first, later)) = panics.split_first() else {
        return String::new();
    };
    let message = format!("internal error: {}", first.message);
    let mut error = match &first.location {
        Some(location) => diagnostic!(
            severity: Error,
            message: (message),
            notes: [note((format!("the panic happened at {location}")))],
        ),
        None => diagnostic!(severity: Error, message: (message)),
    };
    for panic in later {
        match &panic.location {
            Some(location) => error.push_note(
                NoteKind::Note,
                format_args!("then a panic at {location}: {}", panic.message),
            ),
            None => {
                error.push_note(NoteKind::Note, format_args!("then a panic: {}", panic.message))
            }
        }
    }
    let mut text = render(OwnedDiagnostic::new(error).display());
    if let Some(backtrace) = &first.backtrace {
        text += "stack backtrace:\n";
        text += backtrace;
        if !text.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

/// What ends the tool before it reads its input.
enum Exit {
    /// Help or the version was asked for: the text goes to standard output and the tool succeeds
    Help(String),
    /// The arguments are invalid, or the setup of the run failed: the text goes to standard error
    /// and the tool fails
    Usage(String),
}

/// What the driver's pieces say ends the command line, as this tool's exit.
fn exit_of(exit: runic_driver::Exit) -> Exit {
    match exit {
        runic_driver::Exit::Help(text) => Exit::Help(text),
        runic_driver::Exit::Usage(text) => Exit::Usage(text),
        // A way to end that the driver adds later is a usage error here.
        other => Exit::Usage(format!("error: the command line cannot be used: {other:?}\n")),
    }
}

/// Runs the tool, writing what goes to standard output to `out` and what goes to standard error
/// to `err`; returns whether it succeeded.
fn run_to_buffers(
    compiler: &Compiler,
    args: impl IntoIterator<Item = OsString>,
    stdin: &mut dyn Read,
    out: &mut Vec<u8>,
    err: &mut String,
    setup: impl FnOnce() -> Result<(), String>,
) -> Result<bool, Exit> {
    let args = core::iter::once(OsString::from("runic-opt")).chain(args);
    // A flag of a stage that collides with one of the tool's is a defect of the build.
    let flags = StageFlags::new(compiler)
        .unstable(StageFlags::declares_unstable(compiler))
        .build()
        .and_then(|flags| {
            let command = flags.augment(Options::command(), "runic-opt")?;
            Ok((flags, command))
        })
        .map_err(|error| Exit::Usage(render(error.display())))?;
    let (flags, command) = flags;
    let from_clap = |error: clap::Error| {
        let text = error.render().to_string();
        match error.kind() {
            clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
                Exit::Help(text)
            }
            _ => Exit::Usage(text),
        }
    };
    let matches = command.try_get_matches_from(args).map_err(from_clap)?;
    let options = Options::from_arg_matches(&matches).map_err(from_clap)?;

    // Whether the input goes through the compiler is known from the arguments alone.
    let mut harness = match harness::reason(compiler, &options, flags.written(&matches)) {
        Some(reason) => {
            Some(Harness::new(compiler, &options, &flags, &matches, reason).map_err(exit_of)?)
        }
        None => None,
    };
    setup().map_err(|message| Exit::Usage(format!("error: {message}\n")))?;

    if options.list_stages {
        let mut text = String::new();
        harness::list_stages(compiler, &mut text);
        out.extend_from_slice(text.as_bytes());
        return Ok(true);
    }

    if options.list_passes {
        let mut text = String::new();
        list_passes(&mut text);
        out.extend_from_slice(text.as_bytes());
        return Ok(true);
    }
    if options.list_dialects {
        let mut text = String::new();
        list_dialects(&mut text);
        out.extend_from_slice(text.as_bytes());
        return Ok(true);
    }
    if options.list_operations {
        let mut text = String::new();
        list_operations(&mut text);
        out.extend_from_slice(text.as_bytes());
        return Ok(true);
    }

    if let Some(harness) = &mut harness
        && let Err(error) = harness.check(&options)
    {
        *err += &render(error.display());
        return Ok(false);
    }

    // A pass named wrongly, or an empty marker, is an error before anything is read.
    let printing = match options.ir_printing() {
        Ok(printing) => printing,
        Err(error) => {
            *err += &render(error.display());
            return Ok(false);
        }
    };
    let marker = options.split_input_file.as_deref().map(str::trim);
    if marker == Some("") {
        *err += &render(Report::msg("--split-input-file: the marker is only whitespace").display());
        return Ok(false);
    }

    // The pipeline's text, its pass names and their options are checked once, whatever the
    // input holds; whether its anchor fits is known only once an operation is parsed.
    if let Some(pipeline) = &options.pass_pipeline
        && let Err(error) = check_pipeline(pipeline)
    {
        *err += &render(error.display());
        return Ok(false);
    }

    // The request is resolved once, whatever the input holds: what is wrong with it is wrong
    // for every case.
    if let Some(harness) = &mut harness
        && let Err(error) = harness.resolve(&input_name(&options))
    {
        *err += &render(error.display());
        return Ok(false);
    }

    let (name, text) = match read_input(&options, stdin) {
        Ok(input) => input,
        Err(error) => {
            *err += &render(error.display());
            return Ok(false);
        }
    };

    let mut cases = match marker {
        Some(marker) => split::split(&text, marker),
        None => vec![split::whole(text)],
    };
    // On the harness path the frontend decides what an input with no operation is: only a case
    // of whitespace under a marker (what a marker at the end of a file leaves) is skipped.
    if harness.is_some() {
        for case in &mut cases {
            case.empty = marker.is_some() && case.blank;
        }
    } else if cases.iter().all(|case| case.empty) {
        *err += &render(Report::msg(format!("no operation found in {name}")).display());
        return Ok(false);
    }
    let mut written = Vec::new();
    let mut succeeded = true;
    // What the cases wrote to standard error, also where the panic hook can reach it.
    let finished = SharedBuffer::default();
    for case in cases {
        let chunk = Chunk {
            harness: harness.as_ref(),
            name: name.clone(),
            text: case.text,
            first_line: case.first_line,
            empty: case.empty,
            pipeline: options.pass_pipeline.clone(),
            verification: options.verification(),
            dump_pass_pipeline: options.dump_pass_pipeline,
            printing: Some(printing.clone()),
            flags: options.printing_flags(),
            verify_diagnostics: options.verify_diagnostics,
            verify_roundtrip: options.verify_roundtrip,
        };
        match chunk.process_on_its_own_thread(&finished) {
            Ok(output) => {
                if output.succeeded && !output.skipped {
                    written.push(output.stdout);
                }
                // When diagnostics are checked, a case fails only by not matching them.
                let failed = if options.verify_diagnostics {
                    output.mismatched
                } else {
                    !output.succeeded
                };
                if failed {
                    succeeded = false;
                }
            }
            Err(error) => {
                finished.push(&render(error.display()));
                succeeded = false;
            }
        }
    }
    *err += &String::from_utf8_lossy(&finished.take());

    let separator = match marker {
        Some(marker) => format!("{marker}\n"),
        None => String::new(),
    };
    let result = written.join(separator.as_bytes());
    match &options.output {
        // The file is written only when everything succeeded: otherwise it is left as it was.
        Some(_) if !succeeded => {}
        Some(path) => {
            if let Err(error) = runic_compile::write_file(path, &result) {
                *err += &render(
                    Report::msg(format!("could not write '{}': {error}", path.display())).display(),
                );
                return Ok(false);
            }
        }
        None => out.extend_from_slice(&result),
    }
    Ok(succeeded)
}

/// Checks the pipeline's `text` as [`read_pipeline`] reads it, apart from its anchor: its syntax,
/// that each pass is registered, and that each takes the options given.
fn check_pipeline(text: &str) -> Result<(), Report> {
    let context = Rc::new(Context::default());
    let elements = match PassPipeline::parse(text) {
        Ok(pipeline) => pipeline.elements().to_vec(),
        Err(_) => {
            // The elements do not depend on what they are anchored on: any operation will do.
            let dialects = context.registered_dialects();
            let name = dialects
                .values()
                .find_map(|dialect| dialect.registered_ops().first().cloned())
                .expect("an operation is registered");
            PassPipeline::parse_elements(&name, text)?.elements().to_vec()
        }
    };
    // Loaded under `any`, with implicit nesting, every pass is instantiated with its options.
    let any = elements.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
    PassPipeline::parse(&format!("any({any})"))?.load(context, Nesting::Implicit)?;
    Ok(())
}

/// `name  description`, a line for each registered pass, sorted by name.
fn list_passes(out: &mut String) {
    use fmt::Write;

    // `test-panic` exists for the tool's own tests and is not offered, nor are the passes of the
    // toy stages: they are the passes of a backend, which a pipeline can still name.
    let passes = PassInfo::registered()
        .into_iter()
        .filter(|pass| {
            pass.argument() != test_passes::panic::NAME
                && !harness::STAGE_PASSES.contains(&pass.argument())
        })
        .collect::<Vec<_>>();
    let width = passes.iter().map(|pass| pass.argument().len()).max().unwrap_or(0);
    for pass in passes {
        let _ = match pass.description() {
            "" => writeln!(out, "{}", pass.argument()),
            description => writeln!(out, "{:width$}  {description}", pass.argument()),
        };
    }
}

/// `name  N operations`, a line for each registered dialect, sorted by name.
fn list_dialects(out: &mut String) {
    use fmt::Write;

    let context = Context::default();
    let dialects = context.registered_dialects();
    let mut dialects = dialects
        .values()
        .map(|dialect| (dialect.name().as_str(), dialect.registered_ops().len()))
        .collect::<Vec<_>>();
    dialects.sort();
    let width = dialects.iter().map(|(name, _)| name.len()).max().unwrap_or(0);
    for (name, ops) in dialects {
        let noun = if ops == 1 { "operation" } else { "operations" };
        let _ = writeln!(out, "{name:width$}  {ops} {noun}");
    }
}

/// The name of each registered operation, one per line, sorted.
fn list_operations(out: &mut String) {
    use fmt::Write;

    let context = Context::default();
    let dialects = context.registered_dialects();
    let mut names = dialects
        .values()
        .flat_map(|dialect| dialect.registered_ops().iter().map(|name| name.to_string()))
        .collect::<Vec<_>>();
    names.sort();
    for name in names {
        let _ = writeln!(out, "{name}");
    }
}

/// The name the input is known by in diagnostics.
fn input_name(options: &Options) -> String {
    match options.input.as_deref() {
        None | Some("-") => String::from("<stdin>"),
        Some(path) => String::from(path),
    }
}

/// The name the input is known by in diagnostics, and its text.
fn read_input(options: &Options, stdin: &mut dyn Read) -> Result<(String, String), Report> {
    match options.input.as_deref() {
        None | Some("-") => {
            let mut text = String::new();
            stdin
                .read_to_string(&mut text)
                .map_err(|error| Report::msg(format!("could not read standard input: {error}")))?;
            Ok((String::from("<stdin>"), text))
        }
        Some(path) => {
            let text = fs::read_to_string(path)
                .map_err(|error| Report::msg(format!("could not read '{path}': {error}")))?;
            Ok((String::from(path), text))
        }
    }
}

/// Renders a diagnostic, ending with a newline.
fn render(display: DiagnosticDisplay<'_>) -> String {
    runic_driver::render::diagnostic(display, false)
}

/// Where the diagnostics of a chunk go: rendered to its standard error as they come, or, when
/// they are checked, collected.
struct Diagnostics {
    context: Rc<Context>,
    /// The name the chunk's source is registered under
    name: String,
    stderr: SharedBuffer,
    /// What was collected, when diagnostics are checked
    found: Option<Vec<expect::Found>>,
}

impl Diagnostics {
    fn set(&mut self, diagnostics: &DiagnosticSet) {
        match &mut self.found {
            None => {
                for entry in diagnostics.iter() {
                    self.stderr.push(&render(entry.diagnostic.display()));
                }
            }
            Some(found) => {
                let sources = self.context.sources();
                // The chunk's own source, once it was parsed
                let own = sources.find_by_name(self.name.as_str());
                let mut collector = expect::Collector::new(&sources, own);
                collector.set(diagnostics);
                found.extend(collector.finish());
            }
        }
    }

    fn report(&mut self, report: Report) {
        match &mut self.found {
            None => {
                let report = report.attach_session_sources(self.context.sources());
                self.stderr.push(&render(report.display()));
            }
            Some(found) => {
                let sources = self.context.sources();
                // The chunk's own source, once it was parsed
                let own = sources.find_by_name(self.name.as_str());
                let mut collector = expect::Collector::new(&sources, own);
                collector.report(&report);
                found.extend(collector.finish());
            }
        }
    }

    /// An error of a case that is not a diagnostic of an operation: it points into the sources of
    /// the chunk's context, which is the compilation's when there was one.
    fn failure(&mut self, error: &Report) {
        match &mut self.found {
            None => {
                let sources = self.context.sources();
                self.stderr.push(&render(error.display_with_sources(&*sources)));
            }
            Some(found) => {
                let sources = self.context.sources();
                let own = sources.find_by_name(self.name.as_str());
                let mut collector = expect::Collector::new(&sources, own);
                collector.report(error);
                found.extend(collector.finish());
            }
        }
    }

    /// What a compilation reported: its diagnostics that are not its error, then its error. They
    /// point into the compilation's context, which the chunk's sources are looked up in from now
    /// on.
    fn compilation(&mut self, compilation: &Compilation) {
        self.context = compilation.context().clone();
        match &mut self.found {
            None => {
                let sources = self.context.sources();
                for entry in compilation.diagnostics().iter() {
                    self.stderr.push(&render(entry.diagnostic.display_with_sources(&*sources)));
                }
            }
            Some(_) => self.set(compilation.diagnostics()),
        }
        if let Some(error) = compilation.error() {
            self.failure(error);
        }
    }

    fn owned(&mut self, diagnostic: OwnedDiagnostic) {
        match &mut self.found {
            None => self.stderr.push(&render(diagnostic.display())),
            Some(found) => {
                let sources = self.context.sources();
                // The chunk's own source, once it was parsed
                let own = sources.find_by_name(self.name.as_str());
                let mut collector = expect::Collector::new(&sources, own);
                collector.owned(&diagnostic, diagnostic.severity());
                found.extend(collector.finish());
            }
        }
    }
}

/// One top-level operation's text and what to do with it.
struct Chunk<'c> {
    /// How the case is compiled, when it goes through a frontend or a backend instead of being
    /// read as IR
    harness: Option<&'c Harness<'c>>,
    /// The name the text is registered under
    name: String,
    /// The case's text, padded with a newline for each line of the input before it
    text: String,
    /// The line (counted from 0) at which the case begins in the input
    first_line: usize,
    /// Whether the case holds no operation (only whitespace and comments): it is skipped, and
    /// when diagnostics are checked an expectation in it is an error
    empty: bool,
    /// The pipeline text given on the command line
    pipeline: Option<String>,
    verification: Verification,
    dump_pass_pipeline: bool,
    /// How IR is printed between passes; taken by the pass manager
    printing: Option<IRPrintingConfig>,
    flags: OpPrintingFlags,
    /// Whether diagnostics are checked against the expectations in the text
    verify_diagnostics: bool,
    /// Whether both forms of the parsed operation are checked to read back
    verify_roundtrip: bool,
}

/// What processing a chunk wrote, whether it succeeded, and whether its diagnostics did not
/// match its expectations.
struct ChunkOutput {
    stdout: Vec<u8>,
    succeeded: bool,
    mismatched: bool,
    /// The chunk held no operation and was not processed
    skipped: bool,
}

/// How processing a chunk ended.
#[derive(Copy, Clone)]
struct Outcome {
    succeeded: bool,
    mismatched: bool,
    skipped: bool,
}

impl Chunk<'_> {
    /// Processes the chunk on a thread of its own, with a large stack. Everything made from it,
    /// the context first, lives and dies on that thread. A panic there is an internal error.
    ///
    /// What the chunk writes to its standard error goes to `stderr`, after what the cases before it
    /// wrote, and survives a panic: it is followed by the internal error, and the chunk failed
    /// (and, when diagnostics are checked, mismatched). A panic that cannot unwind writes `stderr`
    /// to the process's standard error before the process aborts.
    fn process_on_its_own_thread(self, stderr: &SharedBuffer) -> Result<ChunkOutput, Report> {
        let worker = worker::run(CHUNK_THREAD, stderr, report_panics, || {
            let mut stdout = Vec::new();
            let outcome = self.check_into(&mut stdout, stderr);
            (stdout, outcome)
        });
        match worker {
            Ok((stdout, outcome)) => Ok(ChunkOutput {
                stdout,
                succeeded: outcome.succeeded,
                mismatched: outcome.mismatched,
                skipped: outcome.skipped,
            }),
            Err(WorkerError::NotStarted(error)) => {
                Err(Report::msg(format!("could not start a thread: {error}")))
            }
            Err(WorkerError::Panicked(panics)) => {
                stderr.push(&report_panics(&panics));
                Ok(ChunkOutput {
                    stdout: Vec::new(),
                    succeeded: false,
                    mismatched: true,
                    skipped: false,
                })
            }
            // A way for a worker to end that the driver adds later is still a failure to run.
            Err(error) => Err(Report::msg(format!("could not run a thread: {error}"))),
        }
    }

    /// Processes the chunk and, when diagnostics are checked, checks them.
    ///
    /// An expectation that cannot be read is a mismatch, and the chunk is then not processed. A
    /// chunk with no operation is skipped; an expectation in it is a mismatch.
    fn check_into(mut self, stdout: &mut Vec<u8>, stderr: &SharedBuffer) -> Outcome {
        let context = Rc::new(Context::default());
        let text = core::mem::take(&mut self.text);
        let mismatch = |context: &Context, error: OwnedDiagnostic| {
            stderr.push(&render(error.attach_session_sources(context.sources()).display()));
        };
        let at = |id: Option<SourceId>, range: core::ops::Range<usize>, message: String| {
            OwnedDiagnostic::new(match id.and_then(|id| span_of(id, range)) {
                Some(span) => {
                    diagnostic!(severity: Error, message: (message), labels: [primary(span)],)
                }
                None => diagnostic!(severity: Error, message: (message)),
            })
        };
        let expectations = if self.verify_diagnostics {
            match expect::scan(&text, self.first_line) {
                Ok(expectations) if self.empty && !expectations.is_empty() => {
                    let id = context.add_source(self.name.as_str(), text).ok();
                    for expectation in expectations {
                        let message = String::from(
                            "an expectation in a case with no operation: nothing is checked \
                             against it",
                        );
                        mismatch(&context, at(id, expectation.range, message));
                    }
                    return Outcome {
                        succeeded: false,
                        mismatched: true,
                        skipped: true,
                    };
                }
                Ok(expectations) => Some(expectations),
                Err(malformed) => {
                    let id = context.add_source(self.name.as_str(), text).ok();
                    for malformed in malformed {
                        mismatch(&context, at(id, malformed.range, malformed.message));
                    }
                    return Outcome {
                        succeeded: false,
                        mismatched: true,
                        skipped: self.empty,
                    };
                }
            }
        } else {
            None
        };
        if self.empty {
            return Outcome {
                succeeded: true,
                mismatched: false,
                skipped: true,
            };
        }

        let mut diagnostics = Diagnostics {
            context: context.clone(),
            name: self.name.clone(),
            stderr: stderr.clone(),
            found: expectations.as_ref().map(|_| Vec::new()),
        };
        let succeeded = match self.harness {
            Some(harness) => {
                harness.process_into(self.name.as_str(), text, stdout, &mut diagnostics)
            }
            None => self.process_into(&context, text, stdout, &mut diagnostics),
        };
        // The diagnostics point into the context of the compilation, when there was one.
        let context = diagnostics.context.clone();
        let Some(expectations) = expectations else {
            return Outcome {
                succeeded,
                mismatched: false,
                skipped: false,
            };
        };

        let found = diagnostics.found.unwrap_or_default();
        let id = context.sources().find_by_name(self.name.as_str());
        let mismatches = expect::mismatches(&expectations, &found, id);
        let mismatched = !mismatches.is_empty();
        for error in mismatches {
            mismatch(&context, error);
        }
        Outcome {
            succeeded,
            mismatched,
            skipped: false,
        }
    }

    /// Parses, runs the pipeline on, verifies and prints the chunk's `text`; returns whether it
    /// succeeded.
    fn process_into(
        &mut self,
        context: &Rc<Context>,
        text: String,
        stdout: &mut Vec<u8>,
        diagnostics: &mut Diagnostics,
    ) -> bool {
        let config = ParserConfig::new(context.clone())
            .with_max_region_depth(MAX_REGION_DEPTH)
            .with_max_list_nesting(MAX_LIST_NESTING)
            .verify_after_parse(self.verification != Verification::None);
        let parsed = parse_any(config, self.name.as_str(), text);
        diagnostics.set(&parsed.diagnostics);
        let op = match parsed.result {
            Ok(op) if !parsed.diagnostics.has_errors() => op,
            _ => return false,
        };

        if self.verify_roundtrip {
            let errors = roundtrip::check(op, self.verification != Verification::None);
            let failed = !errors.is_empty();
            for error in errors {
                diagnostics.owned(error);
            }
            if failed {
                return false;
            }
        }

        match self.pipeline.take() {
            Some(pipeline) => {
                if !self.run_pipeline(context, op, &pipeline, diagnostics) {
                    return false;
                }
            }
            // Without a pipeline no pass runs: the pipeline is the empty one.
            None if self.dump_pass_pipeline => {
                diagnostics.stderr.push(&format!("{}()\n", op.borrow().name()));
            }
            None => {}
        }

        if self.verification == Verification::Once
            && let Err(error) = op.borrow().recursively_verify()
        {
            diagnostics.report(error);
            return false;
        }

        let mut printed = print_operation(op, &self.flags);
        printed.push('\n');
        *stdout = printed.into_bytes();
        true
    }

    /// Loads `pipeline` for `op`, runs it, and gives what it reports to `diagnostics`; returns
    /// whether it succeeded.
    fn run_pipeline(
        &mut self,
        context: &Rc<Context>,
        op: OperationRef,
        pipeline: &str,
        diagnostics: &mut Diagnostics,
    ) -> bool {
        let pipeline = match read_pipeline(op, pipeline) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                diagnostics.report(error);
                return false;
            }
        };
        let mut pm = match pipeline.load(context.clone(), Nesting::Implicit) {
            Ok(pm) => pm,
            Err(error) => {
                diagnostics.report(error);
                return false;
            }
        };
        let stderr = diagnostics.stderr.clone();
        if self.dump_pass_pipeline {
            stderr.push(&format!("{}\n", Textual(&pm)));
        }
        pm.enable_verifier(self.verification == Verification::EachPass);
        let printing = IRPrintingConfig {
            flags: self.flags,
            ..self.printing.take().unwrap_or_default()
        };
        // The buffer never fails a write, so the handle of the first error is not kept.
        let (output, _) = IoOutput::boxed(stderr.clone());
        pm.enable_ir_printing(printing, output);
        let runs = Rc::<RefCell<BTreeMap<&'static str, usize>>>::default();
        pm.add_instrumentation(Box::new(CountRuns(runs.clone())));

        // What the passes emit on the context is collected apart from anything emitted before.
        let scope = context.diagnostic_scope();
        let result = pm.run(op);
        drop(pm);
        let emitted = scope.finish().finish().attach_session_sources(context.sources());
        diagnostics.set(&emitted);
        if let Err(error) = result {
            diagnostics.report(error);
            return false;
        }
        // An error a pass emitted fails the case, as the parser's do.
        if emitted.has_errors() {
            return false;
        }

        // Implicit nesting reaches one level deep: say which passes of the pipeline ran on nothing.
        let anchor = op.borrow().name();
        let runs = runs.borrow();
        let mut named = Vec::new();
        pass_names(pipeline.elements(), &mut named);
        let mut warned = Vec::new();
        for name in named {
            if runs.get(name).copied().unwrap_or(0) > 0 || warned.contains(&name) {
                continue;
            }
            warned.push(name);
            let help = match target_of(context, name) {
                Some(target) => format!(
                    "implicit nesting runs a pass on the operations of its type directly inside \
                     '{anchor}', not deeper; to reach deeper, write 'recursive({target}({name}))' \
                     or name the operations in between"
                ),
                None => String::from(
                    "it runs on any operation, but the pipeline it is in is nested under \
                     operations the input does not hold where the pipeline looks for them",
                ),
            };
            let warning = diagnostic!(
                severity: Warning,
                message: (format!("pass '{name}' ran on no operation")),
                notes: [help(help)],
            );
            diagnostics.owned(OwnedDiagnostic::new(warning));
        }
        true
    }
}

/// The span of `range` in the source `id`.
fn span_of(id: SourceId, range: core::ops::Range<usize>) -> Option<SourceSpan> {
    TextRange::try_from_usize(range.start, range.end)
        .ok()
        .map(|range| SourceSpan::session(id, range))
}

/// Reads `text` for `op`: text that parses as one whole pipeline is anchored, and its anchor must
/// be `any` or the name of `op`; any other text is a list of elements anchored on `op`.
fn read_pipeline(op: OperationRef, text: &str) -> Result<PassPipeline, Report> {
    let name = op.borrow().name();
    match PassPipeline::parse(text) {
        Ok(pipeline) => match pipeline.anchor() {
            runic_ir::pass::Anchor::Any => Ok(pipeline),
            runic_ir::pass::Anchor::Operation { dialect, opcode }
                if *dialect == name.dialect() && *opcode == name.name() =>
            {
                Ok(pipeline)
            }
            anchor => Err(Report::msg(format!(
                "the pass pipeline is anchored on '{anchor}', but the input's top-level operation \
                 is '{name}': anchor it on '{name}' or 'any', or leave the anchor out"
            ))),
        },
        Err(_) => PassPipeline::parse_elements(&name, text),
    }
}

/// The names of the passes of `elements`, at every depth, in order.
fn pass_names<'a>(elements: &'a [PipelineElement], names: &mut Vec<&'a str>) {
    for element in elements {
        match element {
            PipelineElement::Pass(pass) => names.push(pass.name()),
            PipelineElement::Nested(pipeline) | PipelineElement::Recursive(pipeline) => {
                pass_names(pipeline.elements(), names)
            }
        }
    }
}

/// The operation the registered pass `name` runs on, or `None` when it runs on any.
fn target_of(context: &Rc<Context>, name: &str) -> Option<runic_ir::OperationName> {
    // A manager for `any` takes a pass as it is, so the pass it holds is the one asked about.
    let mut pm = OpPassManager::any(Nesting::Explicit, context.clone());
    PassInfo::lookup(name)?.add_to_pipeline(&mut pm, &PassOptions::new()).ok()?;
    pm.passes().first()?.target_name(context)
}

/// Prints `op` with `flags`.
fn print_operation(op: OperationRef, flags: &OpPrintingFlags) -> String {
    let op = op.borrow();
    let mut printer = AsmPrinter::root(op.context_rc(), flags);
    op.print(&mut printer);
    printer.finish().to_string()
}

/// The text of a loaded pass manager's pipeline.
struct Textual<'a>(&'a PassManager);

impl fmt::Display for Textual<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.print_as_textual_pipeline(f)
    }
}

/// Counts the runs of each pass, by command-line name.
struct CountRuns(Rc<RefCell<BTreeMap<&'static str, usize>>>);

impl PassInstrumentation for CountRuns {
    fn run_before_pass(&mut self, pass: &dyn OperationPass, _op: &OperationRef) {
        let argument = pass.argument();
        if !argument.is_empty() {
            *self.0.borrow_mut().entry(argument).or_default() += 1;
        }
    }
}

#[cfg(test)]
mod tests;
