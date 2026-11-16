//! Running a command line: the thread the compilation runs on, what is written where, and the
//! exit code.

use std::{
    ffi::OsString,
    io::{self, IsTerminal, Read, Write},
    process::ExitCode,
};

use runic_compile::Compiler;

use crate::{
    ColorChoice, Driver, Exit, Invocation,
    render::render_compilation,
    worker::{self, SharedBuffer, WorkerError, WorkerPanic},
};

/// The exit code of a compilation that failed, of a `setup` that failed, and of standard output
/// that could not be written.
pub const EXIT_FAILURE: u8 = 1;

/// The exit code of a command line that is not valid, and of a request that does not resolve.
pub const EXIT_USAGE: u8 = 2;

/// The exit code of a defect of the compiler: a panic while it compiled.
pub const EXIT_INTERNAL: u8 = 101;

/// The name of the thread a compilation runs on.
const WORKER_THREAD: &str = "runic-driver compile";

/// The text that reports `panics`, the first of which is what went wrong.
fn report_panics(panics: &[WorkerPanic], name: &str) -> String {
    let mut text = String::new();
    let Some(first) = panics.first() else {
        return text;
    };
    text += &format!("error: internal compiler error: {}\n", first.message);
    if let Some(location) = &first.location {
        text += &format!("  at {location}\n");
    }
    for later in &panics[1..] {
        match &later.location {
            Some(location) => {
                text += &format!("note: then a panic at {location}: {}\n", later.message)
            }
            None => text += &format!("note: then a panic: {}\n", later.message),
        }
    }
    text += &format!("note: this is a defect of {name}, not of its input\n");
    if let Some(backtrace) = &first.backtrace {
        text += "stack backtrace:\n";
        text += backtrace;
        if !text.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

/// Writes `bytes` to `stream` and flushes it.
fn write_all(stream: &mut dyn Write, bytes: &[u8]) -> io::Result<()> {
    stream.write_all(bytes).and_then(|()| stream.flush())
}

/// What a worker thread hands back: the bytes for standard output and the exit code.
struct WorkerResult {
    stdout: Vec<u8>,
    code: u8,
}

/// What a caller of [Driver::run_with] says about the run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RunOptions {
    /// Whether a color of `auto` is on: a caller that writes to a terminal sets it.
    pub auto_color: bool,
}

impl Driver<'_> {
    /// Runs one command line and returns the exit code (`ExitCode::from` turns it into a
    /// process's). `args` excludes the program's name.
    ///
    /// The run touches nothing of the process but what it is given: the streams, and, once in the
    /// process, the panic hook. The hook acts only on the thread a compilation runs on (it records
    /// the panic for the driver to report) and hands every other panic to the hook installed
    /// before it. A host that replaces the hook after the first run loses the location and
    /// backtrace of a panic in a stage, and the hook it installed sees that panic too. The first
    /// panic recorded is what the report says went wrong, and it may be one a stage caught itself.
    /// When the process cannot unwind (it is built with `panic = "abort"`), the process aborts
    /// after the panic, and the report, with what had been rendered, goes to the process's
    /// standard error and not to the stream given. When logging is enabled, the logger writes to
    /// the process's standard error from the thread of the compilation while `run` waits for it:
    /// `stdout` and `stderr` must then not be locked standard streams, or the run deadlocks.
    ///
    /// The exit code is 0 when the compilation succeeded, and for help, the version and `--print`;
    /// [EXIT_FAILURE] when it failed, or when standard output could not be written; [EXIT_USAGE]
    /// for a command line that is not valid and for a request that does not resolve; and
    /// [EXIT_INTERNAL] when a stage panicked. After a panic the files the compilation had already
    /// written stay written, and standard error holds what was rendered before it, including what
    /// the passes printed (it is written to the buffer as it is printed), then the report of the
    /// panic. A color of `auto` is off here, as [RunOptions::default] has it: [run_with]
    /// turns it on, and the process's [main] turns it on when standard error is a terminal.
    ///
    /// Everything standard output holds is written when the compilation has ended, after what
    /// standard error holds.
    ///
    /// [run_with]: Driver::run_with
    pub fn run(
        &self,
        args: impl IntoIterator<Item = OsString>,
        stdin: &mut dyn Read,
        stdout: &mut dyn Write,
        stderr: &mut dyn Write,
    ) -> u8 {
        self.run_with(args, stdin, stdout, stderr, RunOptions::default(), || Ok(()))
    }

    /// [run](Driver::run), with `options` and with `setup` called once the arguments are read and
    /// a compilation is asked for, and before it starts: the place for what a process sets up for
    /// a run, such as its logger.
    ///
    /// `setup` is not called for help, the version, `--print` or a usage error: asking for help
    /// does not depend on it. An error it returns is written to `stderr` as `error: ..`, and the
    /// exit code is [EXIT_FAILURE].
    pub fn run_with(
        &self,
        args: impl IntoIterator<Item = OsString>,
        stdin: &mut dyn Read,
        stdout: &mut dyn Write,
        stderr: &mut dyn Write,
        options: RunOptions,
        setup: impl FnOnce() -> Result<(), String>,
    ) -> u8 {
        let (request, render) = match self.request(args, stdin) {
            Ok(Invocation::Compile { request, render }) => (request, render),
            Ok(Invocation::Print(text)) | Err(Exit::Help(text)) => {
                return finish(stdout, stderr, text.as_bytes(), &[], 0);
            }
            Err(Exit::Usage(text)) => {
                return finish(stdout, stderr, &[], text.as_bytes(), EXIT_USAGE);
            }
        };
        if let Err(message) = setup() {
            return finish(
                stdout,
                stderr,
                &[],
                format!("error: {message}\n").as_bytes(),
                EXIT_FAILURE,
            );
        }
        let styled = match render.color {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => options.auto_color,
        };

        let name = self.command().get_name();
        let stderr_buffer = SharedBuffer::default();
        let compiler: &Compiler = self.compiler();
        let report_name = name.to_owned();
        let joined = worker::run(
            WORKER_THREAD,
            &stderr_buffer,
            move |panics| report_panics(panics, &report_name),
            || {
                let compilation =
                    compiler.compile_to(*request, &mut (), Box::new(stderr_buffer.clone()));
                let rendered = render_compilation(&compilation, styled, name, &stderr_buffer);
                let code = if rendered.unresolved {
                    EXIT_USAGE
                } else if rendered.failed {
                    EXIT_FAILURE
                } else {
                    0
                };
                WorkerResult {
                    stdout: rendered.stdout,
                    code,
                }
            },
        );
        match joined {
            Err(WorkerError::NotStarted(error)) => finish(
                stdout,
                stderr,
                &[],
                format!("error: could not start a thread: {error}\n").as_bytes(),
                EXIT_INTERNAL,
            ),
            Ok(result) => {
                finish(stdout, stderr, &result.stdout, &stderr_buffer.take(), result.code)
            }
            Err(WorkerError::Panicked(panics)) => {
                let mut text = stderr_buffer.take();
                text.extend_from_slice(report_panics(&panics, name).as_bytes());
                finish(stdout, stderr, &[], &text, EXIT_INTERNAL)
            }
        }
    }
}

/// Writes `err` to `stderr` and then `out` to `stdout`, and answers `code`, or [EXIT_FAILURE] when
/// `out` could not be written (which is then said on `stderr`) and `code` is not
/// [EXIT_INTERNAL]. Standard output is not touched when there is nothing for it, so that a run
/// that writes nothing there does not depend on it.
fn finish(stdout: &mut dyn Write, stderr: &mut dyn Write, out: &[u8], err: &[u8], code: u8) -> u8 {
    // Nothing is left to report a failure to write standard error to.
    let _ = write_all(stderr, err);
    if out.is_empty() {
        return code;
    }
    if let Err(error) = write_all(stdout, out) {
        let _ = write_all(
            stderr,
            format!("error: could not write standard output: {error}\n").as_bytes(),
        );
        return if code == EXIT_INTERNAL {
            code
        } else {
            EXIT_FAILURE
        };
    }
    code
}

/// A compiler's whole `main`: a default [Driver] over `compiler`, the process's arguments and
/// standard streams, and the logger `RUNIC_TRACE` describes.
///
/// A color of `auto` is on exactly when standard error is a terminal. A driver that cannot be
/// built (two of the compiler's stages declare one flag) is a defect of the compiler, not of its
/// user: the error is written to standard error and the exit code is [EXIT_INTERNAL].
pub fn main(compiler: &Compiler) -> ExitCode {
    match Driver::new(compiler).build() {
        Ok(driver) => main_with(&driver),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(EXIT_INTERNAL)
        }
    }
}

/// [main], for a driver the caller built: its own name, version and prefixes.
pub fn main_with(driver: &Driver<'_>) -> ExitCode {
    // The streams are not locked for the run: the logger writes to standard error from the
    // worker thread, and would wait for ever on a lock this thread held while it waits for that
    // thread.
    let mut stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut stderr = io::stderr();
    let auto_color = stderr.is_terminal();
    ExitCode::from(driver.run_with(
        std::env::args_os().skip(1),
        &mut stdin,
        &mut stdout,
        &mut stderr,
        RunOptions { auto_color },
        install_logger,
    ))
}

/// Installs the logger `RUNIC_TRACE` and its companions describe. The driver calls this once it
/// has read its arguments and a compilation is asked for, so that `--help` answers whatever those
/// variables hold.
fn install_logger() -> Result<(), String> {
    let (logger, filter) = runic_log::runic_logger()?;
    if log::set_boxed_logger(logger).is_ok() {
        log::set_max_level(filter);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, fs, path::Path};

    use runic_compile::{
        Artifact, Backend, CheckpointDecl, CheckpointId, Checkpoints, Flow, Frontend, Input,
        IrFrontend, OptionsDecl, Render, Session, Triple,
        testing::{AltVmBackend, CalcFrontend, StackVmBackend},
    };
    use runic_ir::{
        DataLayoutSpec, Endianness, PointerSpec, Report, SourceSpan,
        diagnostics::{SourceKey, diagnostic, report},
        dialects::builtin::WorldRef,
        pass::OpPassManager,
    };

    use super::*;

    const PROBE_X: CheckpointId = CheckpointId::new("probe.x");
    const BOOM_OUT: CheckpointId = CheckpointId::new("boom.out");

    const PROBE_CHECKPOINTS: &[CheckpointDecl] = &[CheckpointDecl {
        id: PROBE_X,
        aliases: &[],
        extension: "x",
        render: render_text as Render,
    }];

    const BOOM_CHECKPOINTS: &[CheckpointDecl] = &[CheckpointDecl {
        id: BOOM_OUT,
        aliases: &[],
        extension: "boom",
        render: render_text as Render,
    }];

    fn render_text(
        artifact: &Artifact,
        _session: &Session<'_>,
        out: &mut dyn Write,
    ) -> Result<(), Report> {
        let text = artifact.downcast_ref::<String>().cloned().unwrap_or_default();
        out.write_all(text.as_bytes()).map_err(|error| Report::msg(error.to_string()))
    }

    /// A frontend whose first line says what it does.
    struct Probe;

    /// Recurses `depth` times with a frame of about 64 KiB each.
    fn deep(depth: usize) {
        let frame = [0u8; 64 * 1024];
        if depth > 0 {
            deep(depth - 1);
        }
        std::hint::black_box(&frame);
    }

    impl Frontend for Probe {
        fn name(&self) -> &'static str {
            "probe"
        }

        fn extensions(&self) -> &'static [&'static str] {
            &["probe"]
        }

        fn checkpoints(&self) -> &'static [CheckpointDecl] {
            PROBE_CHECKPOINTS
        }

        fn translate(
            &self,
            session: &Session<'_>,
            inputs: &[Input],
            _world: WorldRef,
            checkpoints: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            let bytes = session.read(&inputs[0])?;
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let first = text.lines().next().unwrap_or_default();
            match first.split_once(' ').unwrap_or((first, "")) {
                ("warn", _) => {
                    let _ = session.emit(diagnostic!(severity: Warning, message: "take care"));
                }
                ("fail", _) => {
                    let id = session.load_source(&inputs[0])?;
                    let span = SourceSpan::try_from_range(SourceKey::Session(id), None, 5..11)
                        .unwrap_or_default();
                    return Err(report!(
                        message: "the probe failed on purpose",
                        labels: [primary(span, "here")],
                    ));
                }
                ("panic", _) => panic!("the probe panics on purpose"),
                ("stray", _) => {
                    // The thread's panic goes to the previous hook (the test process's standard
                    // error), not to the run.
                    let _ = std::thread::spawn(|| panic!("a stray thread panics")).join();
                }
                ("deep", depth) => deep(depth.parse().unwrap()),
                ("emit-panic", _) => {
                    let _ = checkpoints.reach(PROBE_X, String::from("reached probe.x\n"))?;
                    panic!("the probe panics after its checkpoint");
                }
                _ => {}
            }
            Ok(Flow::Continue(()))
        }
    }

    /// A backend that panics when it emits, for the target `boom-unknown-unknown`.
    struct Boom;

    impl Backend for Boom {
        fn name(&self) -> &'static str {
            "boom"
        }

        fn supports(&self, triple: &Triple) -> bool {
            triple.as_str() == "boom-unknown-unknown"
        }

        fn targets(&self) -> Vec<Triple> {
            vec!["boom-unknown-unknown".parse().unwrap()]
        }

        fn data_layout(&self, _triple: &Triple) -> Result<DataLayoutSpec, Report> {
            Ok(DataLayoutSpec::new(Endianness::Little, PointerSpec::sized(32).unwrap()).unwrap())
        }

        fn checkpoints(&self) -> &'static [CheckpointDecl] {
            BOOM_CHECKPOINTS
        }

        fn add_passes(
            &self,
            _session: &Session<'_>,
            _passes: &mut OpPassManager,
        ) -> Result<(), Report> {
            Ok(())
        }

        fn emit(
            &self,
            _session: &Session<'_>,
            _world: WorldRef,
            _checkpoints: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            panic!("the boom backend explodes")
        }

        fn options(&self) -> Option<OptionsDecl> {
            None
        }
    }

    fn compiler() -> Compiler {
        Compiler::builder()
            .frontend(CalcFrontend)
            .frontend(IrFrontend)
            .frontend(Probe)
            .backend(StackVmBackend)
            .backend(AltVmBackend)
            .backend(Boom)
            .build()
            .unwrap()
    }

    /// What a run left behind.
    #[derive(Debug)]
    struct Run {
        code: u8,
        stdout: String,
        stderr: String,
    }

    fn drive_with(args: &[&str], stdin: &str, auto_color: bool) -> Run {
        let compiler = compiler();
        let driver = Driver::new(&compiler).name("probe-compiler").build().unwrap();
        // A request names a target, and a backend that is one of two that support it.
        let mut args = args.to_vec();
        if !args.contains(&"--target") {
            args.extend(["--target", "stackvm-unknown-unknown", "--backend", "stackvm"]);
        }
        // Files the compilation names after its input go to a directory of their own, not to the
        // directory the tests run in.
        let outputs = tempfile::tempdir().unwrap();
        if !args.contains(&"-o") {
            args.extend(["--output-dir", outputs.path().to_str().unwrap()]);
        }
        let mut stdin = stdin.as_bytes();
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let code = driver.run_with(
            args.iter().map(OsString::from),
            &mut stdin,
            &mut stdout,
            &mut stderr,
            RunOptions { auto_color },
            || Ok(()),
        );
        Run {
            code,
            stdout: String::from_utf8(stdout).unwrap(),
            stderr: String::from_utf8(stderr).unwrap(),
        }
    }

    fn drive(args: &[&str]) -> Run {
        drive_with(args, "", false)
    }

    /// Writes a file and gives its path.
    fn file(dir: &Path, name: &str, text: &str) -> String {
        let path = dir.join(name);
        fs::write(&path, text).unwrap();
        path.to_str().unwrap().to_owned()
    }

    const INC: &str = "fn inc(x) = x + 1;\n";

    #[test]
    fn a_compilation_that_succeeds_exits_with_zero() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        let out = dir.path().join("out.svm");
        let run = drive(&[&input, "-o", out.to_str().unwrap()]);
        assert_eq!(run.code, 0, "{run:?}");
    }

    #[test]
    fn a_compilation_that_fails_exits_with_one() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "bad.probe", "fail\nsecond line\n");
        assert_eq!(drive(&[&input]).code, EXIT_FAILURE);
    }

    #[test]
    fn an_unknown_flag_exits_with_two() {
        assert_eq!(drive(&["--no-such-flag"]).code, EXIT_USAGE);
    }

    #[test]
    fn an_unknown_target_exits_with_two() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        assert_eq!(drive(&[&input, "--target", "nowhere-unknown-unknown"]).code, EXIT_USAGE);
    }

    #[test]
    fn no_inputs_exits_with_two() {
        assert_eq!(drive(&[]).code, EXIT_USAGE);
    }

    #[test]
    fn a_panic_exits_with_one_hundred_and_one() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "boom.probe", "panic\n");
        assert_eq!(drive(&[&input]).code, EXIT_INTERNAL);
    }

    #[test]
    fn help_the_version_and_listings_exit_with_zero() {
        for args in [&["--help"][..], &["--version"], &["--print", "targets"], &["-Z", "help"]] {
            let run = drive(args);
            assert_eq!(run.code, 0, "{args:?}: {run:?}");
            assert!(!run.stdout.is_empty(), "{args:?}");
            assert!(run.stderr.is_empty(), "{args:?}: {}", run.stderr);
        }
    }

    #[test]
    fn the_exit_codes_are_the_documented_ones() {
        assert_eq!((EXIT_FAILURE, EXIT_USAGE, EXIT_INTERNAL), (1, 2, 101));
    }

    #[test]
    fn run_answers_the_exit_code_as_a_number() {
        let compiler = compiler();
        let driver = Driver::new(&compiler).build().unwrap();
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let code = driver.run(
            [OsString::from("--no-such-flag")],
            &mut io::empty(),
            &mut stdout,
            &mut stderr,
        );
        assert_eq!(code, EXIT_USAGE);
    }

    #[test]
    fn a_compilation_that_writes_a_file_leaves_both_streams_empty() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        let out = dir.path().join("out.svm");
        let run = drive(&[&input, "-o", out.to_str().unwrap()]);
        assert_eq!((run.stdout.as_str(), run.stderr.as_str()), ("", ""), "{run:?}");
        assert!(fs::read_to_string(out).unwrap().contains("inc"));
    }

    #[test]
    fn output_to_a_dash_goes_to_standard_output_and_nothing_to_standard_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        let run = drive(&[&input, "-o", "-"]);
        assert_eq!(run.code, 0, "{run:?}");
        assert!(run.stdout.contains("inc") && run.stdout.contains("ret"), "{}", run.stdout);
        assert_eq!(run.stderr, "");
    }

    #[test]
    fn an_emitted_checkpoint_goes_to_standard_output_and_the_listing_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        let out = dir.path().join("out.svm");
        let run = drive(&[&input, "--emit", "ast=-", "--emit", "asm", "-o", out.to_str().unwrap()]);
        assert_eq!(run.code, 0, "{run:?}");
        assert!(run.stdout.contains("fn inc(x)"), "{}", run.stdout);
        assert!(!run.stdout.contains("ret"), "{}", run.stdout);
        assert_eq!(run.stderr, "");
        assert!(fs::read_to_string(out).unwrap().contains("ret"));
    }

    #[test]
    fn two_outputs_to_standard_output_are_a_usage_error_the_session_reports() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        let run = drive(&[&input, "--emit", "ast=-", "--emit", "asm=-"]);
        assert_eq!(run.code, EXIT_USAGE, "{run:?}");
        assert_eq!(run.stdout, "");
        assert!(run.stderr.contains("error"), "{}", run.stderr);
        assert!(run.stderr.contains("standard output"), "{}", run.stderr);
    }

    #[test]
    fn a_warning_is_rendered_to_standard_error_and_the_compilation_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "careful.probe", "warn\n");
        let run = drive(&[&input]);
        assert_eq!(run.code, 0, "{run:?}");
        assert!(run.stderr.contains("warning"), "{}", run.stderr);
        assert!(run.stderr.contains("take care"), "{}", run.stderr);
        assert_eq!(run.stdout, "");
    }

    #[test]
    fn an_error_with_a_label_names_the_file_and_the_line_of_the_label() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "bad.probe", "fail\nsecond line\n");
        let run = drive(&[&input]);
        assert_eq!(run.code, EXIT_FAILURE, "{run:?}");
        assert!(run.stderr.contains("error"), "{}", run.stderr);
        assert!(run.stderr.contains("the probe failed on purpose"), "{}", run.stderr);
        assert!(run.stderr.contains("bad.probe:2"), "{}", run.stderr);
        assert_eq!(run.stdout, "");
    }

    #[test]
    fn an_invalid_flag_is_a_usage_error_on_standard_error_only() {
        let run = drive(&["--no-such-flag"]);
        assert_eq!(run.code, EXIT_USAGE);
        assert!(run.stderr.contains("--no-such-flag"), "{}", run.stderr);
        assert!(run.stderr.contains("--help"), "{}", run.stderr);
        assert_eq!(run.stdout, "");
    }

    #[test]
    fn a_request_that_does_not_resolve_is_an_error_then_the_usage_lines() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        let run = drive(&[&input, "--target", "nowhere-unknown-unknown"]);
        assert_eq!(run.code, EXIT_USAGE, "{run:?}");
        assert!(run.stderr.contains("nowhere-unknown-unknown"), "{}", run.stderr);
        assert!(
            run.stderr.ends_with(
                "usage: probe-compiler [OPTIONS] <INPUT>...\nFor more information, try '--help'.\n"
            ),
            "{}",
            run.stderr
        );
        assert_eq!(run.stdout, "");
    }

    #[test]
    fn a_panic_is_reported_as_an_internal_error_with_its_message_and_location() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "boom.probe", "panic\n");
        let run = drive(&[&input]);
        assert_eq!(run.code, EXIT_INTERNAL, "{run:?}");
        assert_eq!(run.stdout, "");
        assert!(
            run.stderr
                .starts_with("error: internal compiler error: the probe panics on purpose\n"),
            "{}",
            run.stderr
        );
        assert!(run.stderr.contains("  at crates/driver/src/run.rs:"), "{}", run.stderr);
        assert!(
            run.stderr
                .ends_with("note: this is a defect of probe-compiler, not of its input\n")
                || run.stderr.contains("stack backtrace:"),
            "{}",
            run.stderr
        );
    }

    #[test]
    fn a_file_a_panicking_stage_had_written_stays_written() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.probe", "emit-panic\n");
        let out = dir.path().join("reached.x");
        let run = drive(&[&input, "--emit", &format!("probe.x={}", out.to_str().unwrap())]);
        assert_eq!(run.code, EXIT_INTERNAL, "{run:?}");
        assert_eq!(fs::read_to_string(out).unwrap(), "reached probe.x\n");
        assert!(run.stderr.contains("the probe panics after its checkpoint"), "{}", run.stderr);
    }

    #[test]
    fn what_the_passes_printed_precedes_the_internal_error_when_a_backend_panics_after_them() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        let run = drive(&[
            input.as_str(),
            "--target",
            "boom-unknown-unknown",
            "-Z",
            "print-ir-after=all",
            "-Z",
            "dump-pass-pipeline",
        ]);
        assert_eq!(run.code, EXIT_INTERNAL, "{run:?}");
        let printed = run.stderr.find("IR after").expect("the passes' output is kept");
        let error = run.stderr.find("error: internal compiler error: the boom backend").unwrap();
        assert!(run.stderr.starts_with("builtin.world("), "{}", run.stderr);
        assert!(printed < error, "{}", run.stderr);
    }

    #[test]
    fn a_pipeline_or_a_pass_name_that_does_not_read_is_a_usage_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "demo.calc", INC);
        for (flag, text) in [
            ("pass-pipeline=nonesuch", "nonesuch"),
            ("pass-pipeline=cse(", "cse("),
            ("print-ir-after=nonesuch", "unknown pass 'nonesuch'"),
        ] {
            let run = drive(&[&input, "-Z", flag]);
            assert_eq!(run.code, EXIT_USAGE, "{flag}: {run:?}");
            assert!(run.stderr.contains(text), "{flag}: {}", run.stderr);
            assert!(
                run.stderr.ends_with(
                    "usage: probe-compiler [OPTIONS] <INPUT>...\nFor more information, try \
                     '--help'.\n"
                ),
                "{flag}: {}",
                run.stderr
            );
            assert_eq!(run.stdout, "", "{flag}");
        }
    }

    #[test]
    fn pass_output_comes_before_diagnostics_on_standard_error_for_a_compilation_that_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "careful.probe", "warn\n");
        let run = drive(&[&input, "-Z", "print-ir-after=all"]);
        assert_eq!(run.code, 0, "{run:?}");
        let printed = run.stderr.find("IR after").expect("the passes printed");
        let warning = run.stderr.find("take care").expect("the warning is rendered");
        assert!(printed < warning, "{}", run.stderr);
    }

    #[test]
    fn color_always_puts_escapes_in_a_rendered_error_and_never_and_auto_do_not() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "bad.probe", "fail\nsecond line\n");
        let always = drive(&[&input, "--color", "always"]);
        assert!(always.stderr.contains("\x1b["), "{:?}", always.stderr);
        let never = drive(&[&input, "--color", "never"]);
        assert!(!never.stderr.contains("\x1b["), "{:?}", never.stderr);
        let auto = drive(&[&input]);
        assert!(!auto.stderr.contains("\x1b["), "{:?}", auto.stderr);
        let auto_on_a_terminal = drive_with(&[&input], "", true);
        assert!(auto_on_a_terminal.stderr.contains("\x1b["), "{:?}", auto_on_a_terminal.stderr);
        let never_on_a_terminal = drive_with(&[&input, "--color", "never"], "", true);
        assert!(!never_on_a_terminal.stderr.contains("\x1b["));
    }

    #[test]
    fn a_run_after_a_run_that_panicked_is_unaffected() {
        let dir = tempfile::tempdir().unwrap();
        let boom = file(dir.path(), "boom.probe", "panic\n");
        let calm = file(dir.path(), "calm.probe", "warn\n");
        assert_eq!(drive(&[&boom]).code, EXIT_INTERNAL);
        let second = drive(&[&calm, "--color", "never"]);
        assert_eq!(second.code, 0, "{second:?}");
        assert!(!second.stderr.contains("internal compiler error"), "{}", second.stderr);
    }

    #[test]
    fn a_panic_on_another_thread_while_a_run_is_going_does_not_reach_its_report() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "stray.probe", "stray\n");
        let run = drive(&[&input]);
        assert_eq!(run.code, 0, "{run:?}");
        assert!(!run.stderr.contains("internal compiler error"), "{}", run.stderr);
        assert!(!run.stderr.contains("a stray thread"), "{}", run.stderr);
    }

    #[test]
    fn a_deep_recursion_succeeds_on_the_workers_large_stack() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "deep.probe", "deep 1000\n");
        let run = drive(&[&input]);
        assert_eq!(run.code, 0, "{run:?}");
        assert_eq!(run.stderr, "");
    }

    #[test]
    fn setup_runs_for_a_compilation_and_not_for_help_a_listing_or_a_usage_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = file(dir.path(), "calm.probe", "warn\n");
        let compiler = compiler();
        let driver = Driver::new(&compiler).build().unwrap();
        let calls = Cell::new(0);
        let run = |args: &[&str]| {
            let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
            driver.run_with(
                args.iter().map(OsString::from),
                &mut io::empty(),
                &mut stdout,
                &mut stderr,
                RunOptions::default(),
                || {
                    calls.set(calls.get() + 1);
                    Ok(())
                },
            )
        };
        for args in [&["--help"][..], &["--version"], &["--print", "targets"], &["--bogus"]] {
            run(args);
        }
        assert_eq!(calls.get(), 0);
        let out = dir.path().to_str().unwrap();
        let target = [
            "--target",
            "stackvm-unknown-unknown",
            "--backend",
            "stackvm",
            "--output-dir",
            out,
        ];
        assert_eq!(run(&[&[input.as_str()][..], &target].concat()), 0);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn a_setup_that_fails_is_reported_and_the_compilation_does_not_run() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.svm");
        let input = file(dir.path(), "demo.calc", INC);
        let compiler = compiler();
        let driver = Driver::new(&compiler).build().unwrap();
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let args = [&input[..], "-o", out.to_str().unwrap()];
        let code = driver.run_with(
            args.iter().map(OsString::from),
            &mut io::empty(),
            &mut stdout,
            &mut stderr,
            RunOptions::default(),
            || Err(String::from("the logger is not configured")),
        );
        assert_eq!(code, EXIT_FAILURE);
        assert_eq!(String::from_utf8(stderr).unwrap(), "error: the logger is not configured\n");
        assert!(!out.exists());
    }

    #[test]
    fn standard_input_is_an_input() {
        let run = drive_with(&["--language", "calc", "--name", "demo", "-o", "-", "-"], INC, false);
        assert_eq!(run.code, 0, "{run:?}");
        assert!(run.stdout.contains("inc"), "{}", run.stdout);
    }

    #[test]
    fn a_standard_output_that_cannot_be_written_fails_the_run_and_says_why() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("the disk is full"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let compiler = compiler();
        let driver = Driver::new(&compiler).build().unwrap();
        let mut stderr = Vec::new();
        let code = driver.run_with(
            ["--print", "targets"].map(OsString::from),
            &mut io::empty(),
            &mut Broken,
            &mut stderr,
            RunOptions::default(),
            || Ok(()),
        );
        assert_eq!(code, EXIT_FAILURE);
        let stderr = String::from_utf8(stderr).unwrap();
        assert!(stderr.contains("could not write standard output: the disk is full"), "{stderr}");
    }
}
