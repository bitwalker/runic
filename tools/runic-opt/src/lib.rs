//! `runic-opt` reads textual IR, runs a pass pipeline named on the command line on it, verifies
//! and prints the result.
//!
//! The input is one top-level operation. It is parsed in a context of its own, on a thread with
//! a stack large enough for regions nested [`MAX_REGION_DEPTH`] deep; the pass manager is anchored
//! on the operation with implicit nesting. Standard output holds only what was asked for: the
//! printed IR, or a listing. Every diagnostic, and every print of IR between passes, goes to
//! standard error. The exit code is 0 when everything succeeded and 1 otherwise.
//!
//! [`run`] is the whole tool, given its arguments and streams; the binary calls it with the
//! process's.
#![deny(warnings)]

mod options;

use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::OsString,
    fmt, fs,
    io::{self, Read, Write},
    process::ExitCode,
    rc::Rc,
};

use clap::Parser;
use runic_core::{
    Context, OpPrinter, OpPrintingFlags, OperationRef, Report,
    diagnostics::{DiagnosticDisplay, DiagnosticSet, OwnedDiagnostic, RenderConfig, diagnostic},
    parse::{ParserConfig, parse_any},
    pass::{
        IRPrintingConfig, IoOutput, Nesting, OperationPass, PassInfo, PassInstrumentation,
        PassManager, PassPipeline, PipelineElement,
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

    let printing = match options.ir_printing() {
        Ok(printing) => printing,
        Err(error) => {
            *err += &render(error.display());
            return Ok(false);
        }
    };

    let (name, text) = match read_input(&options, stdin) {
        Ok(input) => input,
        Err(error) => {
            *err += &render(error.display());
            return Ok(false);
        }
    };

    let chunk = Chunk {
        name,
        text,
        pipeline: options.pass_pipeline.clone(),
        verification: options.verification(),
        dump_pass_pipeline: options.dump_pass_pipeline,
        printing: Some(printing),
        flags: options.printing_flags(),
    };
    let output = match chunk.process_on_its_own_thread() {
        Ok(output) => output,
        Err(error) => {
            *err += &render(error.display());
            return Ok(false);
        }
    };
    *err += &output.stderr;
    if !output.succeeded {
        return Ok(false);
    }

    match &options.output {
        Some(path) => {
            if let Err(error) = fs::write(path, output.stdout.as_bytes()) {
                *err += &render(
                    Report::msg(format!("could not write '{}': {error}", path.display())).display(),
                );
                return Ok(false);
            }
        }
        None => *out += &output.stdout,
    }
    Ok(true)
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

/// Renders each diagnostic of `diagnostics`.
fn render_set(err: &mut String, diagnostics: &DiagnosticSet) {
    for entry in diagnostics.iter() {
        *err += &render(entry.diagnostic.display());
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
}

/// What processing a chunk wrote, and whether it succeeded.
struct ChunkOutput {
    stdout: String,
    stderr: String,
    succeeded: bool,
}

impl Chunk {
    /// Processes the chunk on a thread with a stack of [`CHUNK_STACK_SIZE`]. Everything made from
    /// it, the context first, lives and dies on that thread. A panic there is an internal error.
    fn process_on_its_own_thread(self) -> Result<ChunkOutput, Report> {
        let thread = std::thread::Builder::new()
            .name(String::from("runic-opt"))
            .stack_size(CHUNK_STACK_SIZE)
            .spawn(move || self.process())
            .map_err(|error| Report::msg(format!("could not start a thread: {error}")))?;
        thread.join().map_err(|payload| {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("a panic without a message");
            Report::msg(format!("internal error: {message}"))
        })
    }

    fn process(self) -> ChunkOutput {
        let stderr = SharedBuffer::default();
        let mut stdout = String::new();
        let succeeded = self.process_into(&mut stdout, &stderr);
        ChunkOutput {
            stdout,
            stderr: stderr.take(),
            succeeded,
        }
    }

    /// Parses, runs the pipeline on, verifies and prints the chunk; returns whether it succeeded.
    fn process_into(mut self, stdout: &mut String, stderr: &SharedBuffer) -> bool {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context.clone())
            .with_max_region_depth(MAX_REGION_DEPTH)
            .verify_after_parse(self.verification != Verification::None);
        let text = core::mem::take(&mut self.text);
        let parsed = parse_any(config, self.name.as_str(), text);
        stderr.push(&{
            let mut text = String::new();
            render_set(&mut text, &parsed.diagnostics);
            text
        });
        let op = match parsed.result {
            Ok(op) if !parsed.diagnostics.has_errors() => op,
            _ => return false,
        };

        match self.pipeline.take() {
            Some(pipeline) => {
                if !self.run_pipeline(&context, op, &pipeline, stderr) {
                    return false;
                }
            }
            // Without a pipeline no pass runs: the pipeline is the empty one.
            None if self.dump_pass_pipeline => {
                stderr.push(&format!("{}()\n", op.borrow().name()));
            }
            None => {}
        }

        if self.verification == Verification::Once
            && let Err(error) = op.borrow().recursively_verify()
        {
            stderr.push(&render(error.attach_session_sources(context.sources()).display()));
            return false;
        }

        *stdout = print_operation(op, &self.flags);
        stdout.push('\n');
        true
    }

    /// Loads `pipeline` for `op`, runs it, and writes what it reports to `stderr`; returns whether
    /// it succeeded.
    fn run_pipeline(
        &mut self,
        context: &Rc<Context>,
        op: OperationRef,
        pipeline: &str,
        stderr: &SharedBuffer,
    ) -> bool {
        let report =
            |error: Report| render(error.attach_session_sources(context.sources()).display());
        let pipeline = match read_pipeline(op, pipeline) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                stderr.push(&report(error));
                return false;
            }
        };
        let mut pm = match pipeline.load(context.clone(), Nesting::Implicit) {
            Ok(pm) => pm,
            Err(error) => {
                stderr.push(&report(error));
                return false;
            }
        };
        if self.dump_pass_pipeline {
            stderr.push(&format!("{}\n", Textual(&pm)));
        }
        pm.enable_verifier(self.verification == Verification::EachPass);
        let printing = IRPrintingConfig {
            flags: copy_flags(&self.flags),
            ..self.printing.take().unwrap_or_default()
        };
        // The buffer never fails a write, so the handle of the first error is not kept.
        let (output, _) = IoOutput::boxed(stderr.clone());
        pm.enable_ir_printing(printing, output);
        let runs = Rc::<RefCell<BTreeMap<&'static str, usize>>>::default();
        pm.add_instrumentation(Box::new(CountRuns(runs.clone())));

        let result = pm.run(op);
        drop(pm);
        let mut text = String::new();
        render_set(&mut text, &context.take_diagnostics());
        stderr.push(&text);
        if let Err(error) = result {
            stderr.push(&report(error));
            return false;
        }

        // Implicit nesting reaches one level deep: say which passes of the pipeline ran on nothing.
        let anchor = op.borrow().name();
        let runs = runs.borrow();
        let mut named = Vec::new();
        pass_names(pipeline.elements(), &mut named);
        named.dedup();
        let mut warned = Vec::new();
        for name in named {
            if runs.get(name).copied().unwrap_or(0) == 0 && !warned.contains(&name) {
                warned.push(name);
                let warning = diagnostic!(
                    severity: Warning,
                    message: (format!("pass '{name}' ran on no operation")),
                    notes: [
                        help(format!(
                            "implicit nesting runs a pass on the operations of its type directly \
                             inside '{anchor}', not deeper; to reach deeper, write \
                             'recursive(<operation>({name}))' or name the operations in between"
                        )),
                    ],
                );
                stderr.push(&render(OwnedDiagnostic::new(warning).display()));
            }
        }
        true
    }
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

fn copy_flags(flags: &OpPrintingFlags) -> OpPrintingFlags {
    OpPrintingFlags {
        print_entry_block_headers: flags.print_entry_block_headers,
        print_source_locations: flags.print_source_locations,
        generic: flags.generic,
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
/// everything stays in the order it was written.
#[derive(Clone, Default)]
struct SharedBuffer(Rc<RefCell<Vec<u8>>>);

impl SharedBuffer {
    fn push(&self, text: &str) {
        self.0.borrow_mut().extend_from_slice(text.as_bytes());
    }

    fn take(&self) -> String {
        String::from_utf8_lossy(&self.0.take()).into_owned()
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
