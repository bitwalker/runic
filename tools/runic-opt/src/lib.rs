//! `runic-opt` reads textual IR, runs a pass pipeline named on the command line on it, verifies
//! and prints the result.
//!
//! The input is one top-level operation, or with `--split-input-file` several, one per case. Each
//! is parsed in a context of its own, on a thread with a stack large enough for regions nested
//! [`MAX_REGION_DEPTH`] deep; the pass manager is anchored on the operation with implicit nesting. Standard output holds only what was asked for: the
//! printed IR, or a listing. Every diagnostic, and every print of IR between passes, goes to
//! standard error. The exit code is 0 when everything succeeded and 1 otherwise.
//!
//! With `--verify-diagnostics` the diagnostics of each case are not rendered: they are checked
//! against the expectations written in the case (see [`expect`]), and what does not match is
//! rendered instead. The exit code is then 1 exactly when something did not match, and a case
//! is printed only when it succeeded.
//!
//! [`run`] is the whole tool, given its arguments and streams; the binary calls it with the
//! process's.
#![deny(warnings)]

mod expect;
mod options;
mod split;

use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::OsString,
    fmt, fs,
    io::{self, Read, Write},
    process::ExitCode,
    rc::Rc,
    sync::{Arc, Mutex, Once, PoisonError},
};

use clap::Parser;
use runic_core::{
    Context, OpPrinter, OpPrintingFlags, OperationRef, Report,
    diagnostics::{
        DiagnosticDisplay, DiagnosticSet, OwnedDiagnostic, RenderConfig, SourceSpan, TextRange,
        diagnostic,
    },
    parse::{ParserConfig, parse_any},
    pass::{
        IRPrintingConfig, IoOutput, Nesting, OpPassManager, OperationPass, PassInfo,
        PassInstrumentation, PassManager, PassPipeline, PipelineElement, registry::RegistryEntry,
    },
    print::AsmPrinter,
};
// The dialects and passes the tool links: naming each crate keeps its registrations.
use runic_dialect_arith as _;
use runic_dialect_cf as _;
use runic_dialect_mem as _;
use runic_dialect_scf as _;
use runic_dialect_ub as _;
use runic_transform as _;

use self::options::{Options, Verification};

/// How deeply regions may nest in the input: text with a region inside more enclosing regions
/// than this is the parser's diagnostic.
pub const MAX_REGION_DEPTH: u32 = 1_000;

/// The stack of the thread an input is processed on, which regions nested
/// [`MAX_REGION_DEPTH`] deep need in every build.
const CHUNK_STACK_SIZE: usize = 256 << 20;

/// The name of the thread an input is processed on. A panic on it is reported by the tool, as an
/// internal error on the standard error it was given, and the panic hook says nothing of it.
const CHUNK_THREAD: &str = "runic-opt chunk";

/// How diagnostics are rendered: plain text of a fixed width, with the source lines each one is
/// about rather than the whole source, so that the output does not depend on the terminal.
const RENDER_CONFIG: RenderConfig = RenderConfig {
    fold: true,
    ..RenderConfig::DEFAULT
};

/// Runs the tool. `args` excludes the program name. Returns the process's exit code.
pub fn run(
    args: impl IntoIterator<Item = OsString>,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> ExitCode {
    silence_chunk_panics();
    let mut failed = false;
    let mut out = String::new();
    let mut err = String::new();
    match run_to_buffers(args, stdin, &mut out, &mut err) {
        Ok(true) => {}
        Ok(false) => failed = true,
        Err(Exit::Help(text)) => out = text,
        Err(Exit::Usage(text)) => {
            err = text;
            failed = true;
        }
    }
    // Nothing is left to report a failure to write standard error to.
    let _ = stderr.write_all(err.as_bytes()).and_then(|()| stderr.flush());
    if let Err(error) = stdout.write_all(out.as_bytes()).and_then(|()| stdout.flush()) {
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

/// Installs, once in the process, a panic hook that says nothing of a panic on a chunk's thread
/// (the tool reports it) and hands any other to the hook installed before it.
fn silence_chunk_panics() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().name() != Some(CHUNK_THREAD) {
                previous(info);
            }
        }));
    });
}

/// What ends the tool before it reads its input.
enum Exit {
    /// Help or the version was asked for: the text goes to standard output and the tool succeeds
    Help(String),
    /// The arguments are invalid: the text goes to standard error and the tool fails
    Usage(String),
}

/// Runs the tool, writing what goes to standard output to `out` and what goes to standard error
/// to `err`; returns whether it succeeded.
fn run_to_buffers(
    args: impl IntoIterator<Item = OsString>,
    stdin: &mut dyn Read,
    out: &mut String,
    err: &mut String,
) -> Result<bool, Exit> {
    let args = core::iter::once(OsString::from("runic-opt")).chain(args);
    let options = Options::try_parse_from(args).map_err(|error| {
        let text = error.render().to_string();
        match error.kind() {
            clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
                Exit::Help(text)
            }
            _ => Exit::Usage(text),
        }
    })?;

    if options.list_passes {
        list_passes(out);
        return Ok(true);
    }
    if options.list_dialects {
        list_dialects(out);
        return Ok(true);
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

    let (name, text) = match read_input(&options, stdin) {
        Ok(input) => input,
        Err(error) => {
            *err += &render(error.display());
            return Ok(false);
        }
    };

    let pieces = match marker {
        Some(marker) => split::split(&text, marker),
        None => vec![text],
    };
    if pieces.is_empty() {
        *err += &render(Report::msg(format!("no operation found in {name}")).display());
        return Ok(false);
    }
    let mut written = Vec::new();
    let mut succeeded = true;
    for text in pieces {
        let chunk = Chunk {
            name: name.clone(),
            text,
            pipeline: options.pass_pipeline.clone(),
            verification: options.verification(),
            dump_pass_pipeline: options.dump_pass_pipeline,
            printing: Some(printing.clone()),
            flags: options.printing_flags(),
            verify_diagnostics: options.verify_diagnostics,
        };
        match chunk.process_on_its_own_thread() {
            Ok(output) => {
                *err += &output.stderr;
                if output.succeeded {
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
                *err += &render(error.display());
                succeeded = false;
            }
        }
    }

    let separator = match marker {
        Some(marker) => format!("{marker}\n"),
        None => String::new(),
    };
    let result = written.join(separator.as_str());
    match &options.output {
        // The file is written only when everything succeeded: otherwise it is left as it was.
        Some(_) if !succeeded => {}
        Some(path) => {
            if let Err(error) = fs::write(path, result.as_bytes()) {
                *err += &render(
                    Report::msg(format!("could not write '{}': {error}", path.display())).display(),
                );
                return Ok(false);
            }
        }
        None => *out += &result,
    }
    Ok(succeeded)
}

/// `name  description`, a line for each registered pass, sorted by name.
fn list_passes(out: &mut String) {
    use fmt::Write;

    let passes = PassInfo::registered();
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
    let mut text = display.with_config(RENDER_CONFIG).to_string();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// Where the diagnostics of a chunk go: rendered to its standard error as they come, or, when
/// they are checked, collected.
struct Diagnostics {
    context: Rc<Context>,
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
                let mut collector = expect::Collector::new(&sources);
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
                let mut collector = expect::Collector::new(&sources);
                collector.report(&report);
                found.extend(collector.finish());
            }
        }
    }

    fn owned(&mut self, diagnostic: OwnedDiagnostic) {
        match &mut self.found {
            None => self.stderr.push(&render(diagnostic.display())),
            Some(found) => {
                let sources = self.context.sources();
                let mut collector = expect::Collector::new(&sources);
                collector.owned(&diagnostic, diagnostic.severity());
                found.extend(collector.finish());
            }
        }
    }
}

/// One top-level operation's text and what to do with it.
struct Chunk {
    /// The name the text is registered under
    name: String,
    text: String,
    /// The pipeline text given on the command line
    pipeline: Option<String>,
    verification: Verification,
    dump_pass_pipeline: bool,
    /// How IR is printed between passes; taken by the pass manager
    printing: Option<IRPrintingConfig>,
    flags: OpPrintingFlags,
    /// Whether diagnostics are checked against the expectations in the text
    verify_diagnostics: bool,
}

/// What processing a chunk wrote, whether it succeeded, and whether its diagnostics did not
/// match its expectations.
struct ChunkOutput {
    stdout: String,
    stderr: String,
    succeeded: bool,
    mismatched: bool,
}

impl Chunk {
    /// Processes the chunk on a thread with a stack of [`CHUNK_STACK_SIZE`]. Everything made from
    /// it, the context first, lives and dies on that thread. A panic there is an internal error.
    ///
    /// What the chunk wrote to its standard error survives a panic: it is kept, and followed by
    /// the internal error, and the chunk failed (and, when diagnostics are checked, mismatched).
    fn process_on_its_own_thread(self) -> Result<ChunkOutput, Report> {
        let stderr = SharedBuffer::default();
        let chunk_stderr = stderr.clone();
        let thread = std::thread::Builder::new()
            .name(String::from(CHUNK_THREAD))
            .stack_size(CHUNK_STACK_SIZE)
            .spawn(move || {
                let mut stdout = String::new();
                let (succeeded, mismatched) = self.check_into(&mut stdout, &chunk_stderr);
                (stdout, succeeded, mismatched)
            })
            .map_err(|error| Report::msg(format!("could not start a thread: {error}")))?;
        match thread.join() {
            Ok((stdout, succeeded, mismatched)) => Ok(ChunkOutput {
                stdout,
                stderr: stderr.take(),
                succeeded,
                mismatched,
            }),
            Err(payload) => {
                let message = panic_message(payload.as_ref());
                let mut written = stderr.take();
                written += &render(Report::msg(format!("internal error: {message}")).display());
                Ok(ChunkOutput {
                    stdout: String::new(),
                    stderr: written,
                    succeeded: false,
                    mismatched: true,
                })
            }
        }
    }

    /// Processes the chunk and, when diagnostics are checked, checks them; returns whether it
    /// succeeded and whether its diagnostics did not match its expectations.
    ///
    /// An expectation that cannot be read is a mismatch, and the chunk is then not processed.
    fn check_into(mut self, stdout: &mut String, stderr: &SharedBuffer) -> (bool, bool) {
        let context = Rc::new(Context::default());
        let text = core::mem::take(&mut self.text);
        let expectations = if self.verify_diagnostics {
            match expect::scan(&text) {
                Ok(expectations) => Some(expectations),
                Err(malformed) => {
                    let id = context.add_source(self.name.as_str(), text).ok();
                    for malformed in malformed {
                        let span = id.and_then(|id| span_of(id, malformed.range));
                        let error = match span {
                            Some(span) => diagnostic!(
                                severity: Error,
                                message: (malformed.message),
                                labels: [primary(span)],
                            ),
                            None => diagnostic!(severity: Error, message: (malformed.message)),
                        };
                        stderr.push(&render(
                            OwnedDiagnostic::new(error)
                                .attach_session_sources(context.sources())
                                .display(),
                        ));
                    }
                    return (false, true);
                }
            }
        } else {
            None
        };

        let mut diagnostics = Diagnostics {
            context: context.clone(),
            stderr: stderr.clone(),
            found: expectations.as_ref().map(|_| Vec::new()),
        };
        let succeeded = self.process_into(&context, text, stdout, &mut diagnostics);
        let Some(expectations) = expectations else {
            return (succeeded, false);
        };

        let found = diagnostics.found.unwrap_or_default();
        let (unexpected, missing) = expect::unmatched(&expectations, &found);
        let id = context.sources().find_by_name(self.name.as_str());
        let mut mismatches = Vec::new();
        for found in &unexpected {
            let message =
                format!("unexpected {}: {}", expect::kind_name(found.kind), found.message);
            mismatches.push(match &found.primary {
                Some((span, Some(label))) => diagnostic!(
                    severity: Error,
                    message: (message),
                    labels: [primary(*span, (label.as_str()))],
                ),
                Some((span, None)) => {
                    diagnostic!(severity: Error, message: (message), labels: [primary(*span)],)
                }
                None => diagnostic!(severity: Error, message: (message)),
            });
        }
        for expectation in &missing {
            let message = format!(
                "expected {} \"{}\" was not produced",
                expect::kind_name(expectation.kind),
                expectation.text
            );
            mismatches.push(match id.and_then(|id| span_of(id, expectation.range.clone())) {
                Some(span) => {
                    diagnostic!(severity: Error, message: (message), labels: [primary(span)],)
                }
                None => diagnostic!(severity: Error, message: (message)),
            });
        }
        let mismatched = !mismatches.is_empty();
        for mismatch in mismatches {
            stderr.push(&render(
                OwnedDiagnostic::new(mismatch)
                    .attach_session_sources(context.sources())
                    .display(),
            ));
        }
        (succeeded, mismatched)
    }

    /// Parses, runs the pipeline on, verifies and prints the chunk's `text`; returns whether it
    /// succeeded.
    fn process_into(
        &mut self,
        context: &Rc<Context>,
        text: String,
        stdout: &mut String,
        diagnostics: &mut Diagnostics,
    ) -> bool {
        let config = ParserConfig::new(context.clone())
            .with_max_region_depth(MAX_REGION_DEPTH)
            .verify_after_parse(self.verification != Verification::None);
        let parsed = parse_any(config, self.name.as_str(), text);
        diagnostics.set(&parsed.diagnostics);
        let op = match parsed.result {
            Ok(op) if !parsed.diagnostics.has_errors() => op,
            _ => return false,
        };

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

        *stdout = print_operation(op, &self.flags);
        stdout.push('\n');
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
fn span_of(
    id: runic_core::diagnostics::SourceId,
    range: core::ops::Range<usize>,
) -> Option<SourceSpan> {
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
            runic_core::pass::Anchor::Any => Ok(pipeline),
            runic_core::pass::Anchor::Operation { dialect, opcode }
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
fn target_of(context: &Rc<Context>, name: &str) -> Option<runic_core::OperationName> {
    // A manager for `any` takes a pass as it is, so the pass it holds is the one asked about.
    let mut pm = OpPassManager::any(Nesting::Explicit, context.clone());
    PassInfo::lookup(name)?.add_to_pipeline(&mut pm, "").ok()?;
    pm.passes().first()?.target_name(context)
}

/// The message of a panic: its text, or the rendered report it carries.
fn panic_message(payload: &(dyn core::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        String::from(*text)
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(report) = payload.downcast_ref::<Report>() {
        render(report.display()).trim_end().to_owned()
    } else {
        String::from("a panic without a message")
    }
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

/// What the chunk writes to standard error, shared with the pass manager's IR printing so that
/// everything stays in the order it was written, and with the thread that started the chunk so
/// that it survives a panic.
#[derive(Clone, Default)]
struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl SharedBuffer {
    fn push(&self, text: &str) {
        self.bytes().extend_from_slice(text.as_bytes());
    }

    fn take(&self) -> String {
        String::from_utf8_lossy(&core::mem::take(&mut *self.bytes())).into_owned()
    }

    /// The bytes, also after a panic while they were held.
    fn bytes(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
