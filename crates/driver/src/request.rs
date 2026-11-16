//! From a command line to a [CompilationRequest].

use std::{ffi::OsString, io::Read};

use clap::{ArgMatches, FromArgMatches, error::ErrorKind, parser::ValueSource};
use runic_compile::{CompilationRequest, Compiler, IR_CHECKPOINTS, Input, Verification};

use crate::{ColorChoice, Driver, Flags, PrintKind, Unstable};

/// What a command line asks for.
#[derive(Debug)]
#[non_exhaustive]
pub enum Invocation {
    /// A compilation, and how its diagnostics are rendered.
    Compile {
        /// What the command line asked the compiler to do.
        request: Box<CompilationRequest>,
        /// How the compilation's diagnostics are rendered.
        render: RenderOptions,
    },
    /// Text to write to standard output (`--print`).
    Print(String),
}

/// How a compilation's diagnostics are rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct RenderOptions {
    /// When to color them (`--color`).
    pub color: ColorChoice,
}

/// A command line that asks for no compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Exit {
    /// Help, the version, or `-Z help`: text for standard output, and success.
    Help(String),
    /// A usage error: text for standard error, ending with a newline, and exit code 2.
    Usage(String),
}

/// The text of a usage error that `message` says, as the driver words its own errors.
pub(crate) fn usage(message: impl core::fmt::Display) -> Exit {
    Exit::Usage(format!("error: {message}\n"))
}

/// `text` with exactly one newline at its end.
fn with_newline(mut text: String) -> String {
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// What `clap` says about a command line it rejects or answers: help and the version are output,
/// the rest are usage errors.
fn from_clap(error: clap::Error) -> Exit {
    let text = with_newline(error.render().to_string());
    match error.kind() {
        ErrorKind::DisplayHelp
        | ErrorKind::DisplayVersion
        | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => Exit::Help(text),
        _ => Exit::Usage(text),
    }
}

/// What `clap` says about the arguments of a scratch command, which has no program name or help
/// to point to: the message without the usage and the hint that follow it.
pub(crate) fn scratch_text(error: &clap::Error) -> String {
    let text = error.render().to_string();
    let end = ["\n\nUsage:", "\n\nFor more information"]
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()
        .unwrap_or(text.len());
    with_newline(text[..end].trim_end().to_owned())
}

impl Driver<'_> {
    /// What `args` ask the compiler to do. `args` excludes the program's name, and `stdin` is read
    /// only when an input is `-`.
    ///
    /// Everything a user can get wrong is an [Exit::Usage]; help, the version and `-Z help` are
    /// [Exit::Help]. Every `-Z` entry is checked, and `-Z help` answered, before `--print`
    /// is read: a bad `-Z` entry is a usage error and `-Z help` is help even beside `--print`.
    /// A command line that passes those checks and has `--print` is [Invocation::Print], whatever
    /// else was written: `--print` wins over a compilation. The request that
    /// comes back holds what the flags said and nothing else: what a flag leaves unsaid is left to
    /// the session, which stores the defaults of the stages that take part.
    pub fn request(
        &self,
        args: impl IntoIterator<Item = OsString>,
        stdin: &mut dyn Read,
    ) -> Result<Invocation, Exit> {
        let program = OsString::from(self.command().get_name());
        let matches = self
            .command()
            .clone()
            .try_get_matches_from(core::iter::once(program).chain(args))
            .map_err(from_clap)?;
        let flags = Flags::from_arg_matches(&matches).map_err(from_clap)?;

        let settings = self.stage_flags().resolve(&matches)?;

        if !flags.print.is_empty() {
            let listings: String =
                flags.print.iter().map(|kind| listing(self.compiler(), *kind)).collect();
            return Ok(Invocation::Print(listings));
        }

        let inputs = self.inputs(&matches, &flags, stdin)?;
        let mut request = CompilationRequest::default();
        request.name = flags.name;
        request.inputs = inputs;
        request.target = flags.target;
        request.backend = flags.backend;
        request.opt_level = flags.opt_level.unwrap_or_default();
        request.debug = flags.debug.unwrap_or_default();
        request.externs = flags.externs;
        request.search_paths = flags.search_path;
        request.emit = flags.emit;
        request.stop_after = flags.stop_after;
        request.output_file = flags.output_file;
        request.output_dir = flags.output_dir;
        request.remap_path_prefixes = flags.remap_path_prefix;

        if let Some(own) = self.stage_flags().store_resolved(&matches, &settings, &mut request)? {
            store_unstable(&own, &mut request)?;
        }

        Ok(Invocation::Compile {
            request: Box::new(request),
            render: RenderOptions { color: flags.color },
        })
    }

    /// The inputs of the command line, each with the nearest `--language` written before it, and
    /// standard input read where an input is `-`.
    fn inputs(
        &self,
        matches: &ArgMatches,
        flags: &Flags,
        stdin: &mut dyn Read,
    ) -> Result<Vec<Input>, Exit> {
        let indices = |id: &str| -> Vec<usize> {
            matches.indices_of(id).map(Iterator::collect).unwrap_or_default()
        };
        let input_at = indices("input");
        let language_at = indices("language");

        for (index, (at, language)) in language_at.iter().zip(&flags.language).enumerate() {
            let Some(next_input) = input_at.iter().find(|input| *input > at) else {
                return Err(usage(format!(
                    "'--language {language}' applies to the inputs after it, and none follows"
                )));
            };
            if let Some((_, next)) = language_at
                .iter()
                .zip(&flags.language)
                .nth(index + 1)
                .filter(|(next_at, _)| *next_at < next_input)
            {
                return Err(usage(format!(
                    "'--language {language}' applies to the inputs after it, and none follows \
                     before '--language {next}'"
                )));
            }
        }

        if flags.input.iter().filter(|input| input.as_os_str() == "-").count() > 1 {
            return Err(usage(
                "standard input ('-') is given as an input more than once; it can be read once",
            ));
        }

        let mut inputs = Vec::with_capacity(flags.input.len());
        for (input, at) in flags.input.iter().zip(&input_at) {
            let language = language_at
                .iter()
                .zip(&flags.language)
                .take_while(|(language_at, _)| *language_at < at)
                .last()
                .map(|(_, language)| language.as_str());
            let input = if input.as_os_str() == "-" {
                let Some(language) = language else {
                    return Err(usage(
                        "the input '-' is standard input, which has no file name to tell its \
                         language by; give the language with '--language NAME' before it",
                    ));
                };
                let mut bytes = Vec::new();
                stdin.read_to_end(&mut bytes).map_err(|error| {
                    usage(format!("could not read standard input, the input '-': {error}"))
                })?;
                Input::memory("<stdin>", bytes).with_language(language)
            } else {
                let input = Input::path(input.clone());
                match language {
                    Some(language) => input.with_language(language),
                    None => input,
                }
            };
            inputs.push(input);
        }
        Ok(inputs)
    }
}

/// Stores the settings of the driver itself that `matches`, the parse of the `-Z` entries it owns,
/// holds in `request`.
fn store_unstable(matches: &ArgMatches, request: &mut CompilationRequest) -> Result<(), Exit> {
    let unstable = Unstable::from_arg_matches(matches).map_err(from_clap)?;
    let explicit = matches.value_source("verify_each") == Some(ValueSource::CommandLine);
    if unstable.no_verify && explicit {
        return Err(usage(
            "'-Z no-verify' and '-Z verify-each' contradict each other: the first verifies \
             nothing and the second says when to verify",
        ));
    }
    request.pipeline = unstable.pass_pipeline;
    request.ir.print_before = unstable.print_ir_before;
    request.ir.print_after = unstable.print_ir_after;
    request.ir.print_generic = unstable.print_generic;
    request.ir.print_locations = unstable.print_locations;
    request.ir.dump_pass_pipeline = unstable.dump_pass_pipeline;
    request.ir.verification = if unstable.no_verify {
        Verification::None
    } else if unstable.verify_each {
        Verification::EachPass
    } else {
        Verification::Once
    };
    Ok(())
}

/// The text `--print KIND` writes.
fn listing(compiler: &Compiler, kind: PrintKind) -> String {
    let mut text = String::new();
    match kind {
        PrintKind::Targets => {
            for triple in compiler.targets() {
                text.push_str(triple.as_str());
                if compiler.default_target() == Some(&triple) {
                    text.push_str(" (default)");
                }
                text.push('\n');
            }
        }
        PrintKind::Backends => {
            for backend in compiler.backends() {
                let targets: Vec<String> =
                    backend.targets().iter().map(|triple| triple.to_string()).collect();
                text.push_str(&format!("{}: {}\n", backend.name(), targets.join(", ")));
            }
        }
        PrintKind::Frontends => {
            for frontend in compiler.frontends() {
                let extensions: Vec<String> =
                    frontend.extensions().iter().map(|ext| format!(".{ext}")).collect();
                text.push_str(&format!("{}: {}\n", frontend.name(), extensions.join(", ")));
            }
        }
        PrintKind::Checkpoints => {
            // In the order a compilation reaches them: the frontends', the harness's, the
            // backends'.
            let declared = compiler
                .frontends()
                .flat_map(|frontend| frontend.checkpoints())
                .chain(IR_CHECKPOINTS.iter())
                .chain(compiler.backends().flat_map(|backend| backend.checkpoints()));
            for decl in declared {
                text.push_str(decl.id.as_str());
                if !decl.aliases.is_empty() {
                    text.push_str(&format!(" (alias {})", decl.aliases.join(", ")));
                }
                text.push_str(&format!(" -> .{}\n", decl.extension));
            }
        }
    }
    text.replace(": \n", ":\n")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::{Arg, ArgAction, Args};
    use runic_compile::{
        Artifact, Backend, CheckpointDecl, CheckpointId, Checkpoints, DebugInfo, EmitRequest,
        EmitTo, Extern, Flow, IrOptions, OptLevel, OptionsDecl, OutputFile, RemapPathPrefix,
        Session, Triple,
        testing::{CalcFrontend, CalcOptions, StackVmBackend, StackVmOptions, StackVmUnstable},
    };
    use runic_ir::{DataLayoutSpec, Report, dialects::builtin::WorldRef, pass::OpPassManager};

    use super::*;
    use crate::tests::{backend, stable, toy};

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    fn run_on(compiler: &Compiler, args: &[&str], stdin: &[u8]) -> Result<Invocation, Exit> {
        let driver = Driver::new(compiler).build().unwrap();
        let mut stdin = stdin;
        driver.request(os(args), &mut stdin)
    }

    fn compile_on(compiler: &Compiler, args: &[&str]) -> (CompilationRequest, RenderOptions) {
        match run_on(compiler, args, b"") {
            Ok(Invocation::Compile { request, render }) => (*request, render),
            other => panic!("{args:?} is not a compilation: {other:?}"),
        }
    }

    fn request(args: &[&str]) -> CompilationRequest {
        compile_on(&toy(), args).0
    }

    fn exit(args: &[&str]) -> Exit {
        match run_on(&toy(), args, b"") {
            Err(exit) => exit,
            Ok(invocation) => panic!("{args:?} is accepted: {invocation:?}"),
        }
    }

    fn usage_text(args: &[&str]) -> String {
        match exit(args) {
            Exit::Usage(text) => {
                assert!(text.ends_with('\n'), "{text:?}");
                text
            }
            Exit::Help(text) => panic!("{args:?} is help: {text}"),
        }
    }

    fn printed(args: &[&str]) -> String {
        match run_on(&toy(), args, b"") {
            Ok(Invocation::Print(text)) => text,
            other => panic!("{args:?} does not print: {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn an_input_path_that_is_not_unicode_reaches_the_request_as_it_was_written() {
        use std::os::unix::ffi::OsStringExt;

        let name = OsString::from_vec(b"caf\xe9.calc".to_vec());
        let driver_compiler = toy();
        let driver = Driver::new(&driver_compiler).build().unwrap();
        let Ok(Invocation::Compile { request, .. }) =
            driver.request([name.clone()], &mut std::io::empty())
        else {
            panic!("the input is not accepted");
        };
        assert_eq!(request.inputs, [Input::path(PathBuf::from(name))]);
    }

    /// Compares every public field of two requests, and the options of the types the toy stages
    /// declare.
    #[track_caller]
    fn assert_same(actual: &CompilationRequest, expected: &CompilationRequest) {
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.inputs, expected.inputs);
        assert_eq!(actual.target, expected.target);
        assert_eq!(actual.backend, expected.backend);
        assert_eq!(actual.opt_level, expected.opt_level);
        assert_eq!(actual.debug, expected.debug);
        assert_eq!(actual.externs, expected.externs);
        assert_eq!(actual.search_paths, expected.search_paths);
        assert_eq!(actual.emit, expected.emit);
        assert_eq!(actual.stop_after, expected.stop_after);
        assert_eq!(actual.output_file, expected.output_file);
        assert_eq!(actual.output_dir, expected.output_dir);
        assert_eq!(actual.pipeline, expected.pipeline);
        assert_eq!(actual.ir, expected.ir);
        assert_eq!(actual.remap_path_prefixes, expected.remap_path_prefixes);
        assert_eq!(actual.options::<CalcOptions>(), expected.options::<CalcOptions>());
        assert_eq!(actual.options::<StackVmOptions>(), expected.options::<StackVmOptions>());
        assert_eq!(actual.options::<StackVmUnstable>(), expected.options::<StackVmUnstable>());
    }

    fn built(change: impl FnOnce(&mut CompilationRequest)) -> CompilationRequest {
        let mut request = CompilationRequest::new();
        change(&mut request);
        request
    }

    #[test]
    fn no_arguments_is_the_default_request_and_automatic_color() {
        let (actual, render) = compile_on(&toy(), &[]);
        assert_same(&actual, &CompilationRequest::new());
        assert_eq!(
            render,
            RenderOptions {
                color: ColorChoice::Auto
            }
        );
    }

    #[test]
    fn each_stable_flag_alone_fills_its_field_in_each_spelling() {
        let cases: Vec<(Vec<&str>, CompilationRequest)> = vec![
            (vec!["--name", "app"], built(|r| r.name = Some("app".into()))),
            (vec!["--name=app"], built(|r| r.name = Some("app".into()))),
            (
                vec!["--target", "stackvm-unknown-tiny"],
                built(|r| r.target = Some("stackvm-unknown-tiny".parse::<Triple>().unwrap())),
            ),
            (vec!["--backend=altvm"], built(|r| r.backend = Some("altvm".into()))),
            (vec!["-O2"], built(|r| r.opt_level = OptLevel::Balanced)),
            (vec!["-O", "3"], built(|r| r.opt_level = OptLevel::Max)),
            (vec!["-Os"], built(|r| r.opt_level = OptLevel::Size)),
            (vec!["--opt-level=s"], built(|r| r.opt_level = OptLevel::Size)),
            (vec!["--opt-level", "z"], built(|r| r.opt_level = OptLevel::SizeMin)),
            (vec!["-g"], built(|r| r.debug = DebugInfo::Full)),
            (vec!["--debug"], built(|r| r.debug = DebugInfo::Full)),
            (vec!["--debug=line"], built(|r| r.debug = DebugInfo::Line)),
            (vec!["-g=none"], built(|r| r.debug = DebugInfo::None)),
            (vec!["-L", "p"], built(|r| r.search_paths = vec!["p".into()])),
            (vec!["-Lp"], built(|r| r.search_paths = vec!["p".into()])),
            (vec!["--search-path=p"], built(|r| r.search_paths = vec!["p".into()])),
            (vec!["--stop-after", "ir"], built(|r| r.stop_after = Some("ir".into()))),
            (vec!["-o", "x"], built(|r| r.output_file = Some(OutputFile::Path("x".into())))),
            (
                vec!["--output-file", "x"],
                built(|r| r.output_file = Some(OutputFile::Path("x".into()))),
            ),
            (
                vec!["--output-file=x"],
                built(|r| r.output_file = Some(OutputFile::Path("x".into()))),
            ),
            (vec!["-o", "-"], built(|r| r.output_file = Some(OutputFile::Stdout))),
            (vec!["--output-dir", "out"], built(|r| r.output_dir = Some("out".into()))),
            (
                vec!["--remap-path-prefix=/a=/b"],
                built(|r| {
                    r.remap_path_prefixes = vec![RemapPathPrefix {
                        from: "/a".into(),
                        to: Some("/b".into()),
                    }]
                }),
            ),
        ];
        for (args, expected) in cases {
            assert_same(&request(&args), &expected);
        }
    }

    #[test]
    fn repeated_flags_keep_the_order_they_were_written_in() {
        let actual = request(&[
            "--extern",
            "b=lib/b",
            "--extern=a",
            "-L",
            "two",
            "--search-path=one",
            "--emit",
            "ir",
            "--emit=asm=out.svm",
            "--emit",
            "ir.translated=-",
            "--remap-path-prefix=/x",
            "--remap-path-prefix",
            "/y=/z",
            "c.calc",
            "a.calc",
            "b.runic",
        ]);
        let expected = built(|r| {
            r.externs = vec![
                Extern {
                    name: "b".into(),
                    path: Some("lib/b".into()),
                },
                Extern {
                    name: "a".into(),
                    path: None,
                },
            ];
            r.search_paths = vec!["two".into(), "one".into()];
            r.emit = vec![
                EmitRequest {
                    checkpoint: "ir".into(),
                    to: EmitTo::Default,
                },
                EmitRequest {
                    checkpoint: "asm".into(),
                    to: EmitTo::File("out.svm".into()),
                },
                EmitRequest {
                    checkpoint: "ir.translated".into(),
                    to: EmitTo::Stdout,
                },
            ];
            r.remap_path_prefixes = vec![
                RemapPathPrefix {
                    from: "/x".into(),
                    to: None,
                },
                RemapPathPrefix {
                    from: "/y".into(),
                    to: Some("/z".into()),
                },
            ];
            r.inputs = vec![Input::path("c.calc"), Input::path("a.calc"), Input::path("b.runic")];
        });
        assert_same(&actual, &expected);
    }

    #[test]
    fn a_debug_flag_before_an_input_does_not_take_the_input_as_its_value() {
        let actual = request(&["-g", "a.calc"]);
        assert_same(
            &actual,
            &built(|r| {
                r.debug = DebugInfo::Full;
                r.inputs = vec![Input::path("a.calc")];
            }),
        );
        let actual = request(&["--debug", "a.calc"]);
        assert_eq!(actual.inputs, [Input::path("a.calc")]);
    }

    #[test]
    fn a_language_applies_to_the_inputs_written_after_it_until_the_next_language() {
        let actual = request(&["a.x", "--language", "calc", "b", "c", "--language=runic", "d"]);
        let expected = built(|r| {
            r.inputs = vec![
                Input::path("a.x"),
                Input::path("b").with_language("calc"),
                Input::path("c").with_language("calc"),
                Input::path("d").with_language("runic"),
            ];
        });
        assert_same(&actual, &expected);
    }

    #[test]
    fn a_language_after_the_last_input_is_a_usage_error() {
        let text = usage_text(&["a.x", "--language", "calc"]);
        assert!(
            text.contains("'--language calc' applies to the inputs after it, and none follows"),
            "{text}"
        );
        let text = usage_text(&["--language=runic"]);
        assert!(text.contains("'--language runic' applies to the inputs"), "{text}");
        let text = usage_text(&["--language", "calc", "a", "--language", "runic"]);
        assert!(text.contains("'--language runic'"), "{text}");
    }

    #[test]
    fn a_language_with_no_input_before_the_next_language_is_a_usage_error_naming_both() {
        let text = usage_text(&["--language", "calc", "--language", "runic", "a.calc"]);
        assert!(
            text.contains(
                "'--language calc' applies to the inputs after it, and none follows before \
                 '--language runic'"
            ),
            "{text}"
        );
    }

    #[test]
    fn a_dash_reads_standard_input_into_a_memory_input_in_the_language_before_it() {
        let (actual, _) = {
            let invocation =
                run_on(&toy(), &["--language", "calc", "-", "b.calc"], b"def f(x) = x\n");
            match invocation {
                Ok(Invocation::Compile { request, render }) => (*request, render),
                other => panic!("{other:?}"),
            }
        };
        let expected = built(|r| {
            r.inputs = vec![
                Input::memory("<stdin>", b"def f(x) = x\n".to_vec()).with_language("calc"),
                Input::path("b.calc").with_language("calc"),
            ];
        });
        assert_same(&actual, &expected);
    }

    #[test]
    fn a_dash_without_a_language_is_a_usage_error_that_says_to_give_one() {
        let text = usage_text(&["-"]);
        assert!(text.contains("'-'") && text.contains("--language"), "{text}");
        // A language written after it does not apply to it.
        let text = usage_text(&["-", "--language", "calc", "a"]);
        assert!(text.contains("--language"), "{text}");
    }

    #[test]
    fn standard_input_given_twice_is_a_usage_error() {
        let text = usage_text(&["--language", "calc", "-", "-"]);
        assert!(text.contains("more than once"), "{text}");
    }

    #[test]
    fn a_standard_input_that_cannot_be_read_is_a_usage_error_naming_it() {
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("the pipe is gone"))
            }
        }
        let compiler = toy();
        let driver = Driver::new(&compiler).build().unwrap();
        let exit = driver.request(os(&["--language", "calc", "-"]), &mut Broken).unwrap_err();
        let Exit::Usage(text) = exit else {
            panic!("{exit:?}")
        };
        assert!(text.contains("standard input") && text.contains("the pipe is gone"), "{text}");
    }

    #[test]
    fn color_reaches_the_render_options_and_leaves_the_request_alone() {
        for (text, color) in [
            ("always", ColorChoice::Always),
            ("never", ColorChoice::Never),
            ("auto", ColorChoice::Auto),
        ] {
            let (request, render) = compile_on(&toy(), &["--color", text]);
            assert_eq!(render.color, color);
            assert_same(&request, &CompilationRequest::new());
        }
    }

    #[test]
    fn print_targets_lists_one_triple_per_line() {
        assert_eq!(
            printed(&["--print", "targets"]),
            "stackvm-unknown-unknown\nstackvm-unknown-tiny\n"
        );
    }

    #[test]
    fn print_targets_marks_the_default() {
        let compiler = Compiler::builder()
            .backend(runic_compile::testing::StackVmBackend)
            .default_target("stackvm-unknown-tiny")
            .build()
            .unwrap();
        let Ok(Invocation::Print(text)) = run_on(&compiler, &["--print=targets"], b"") else {
            panic!()
        };
        assert_eq!(text, "stackvm-unknown-unknown\nstackvm-unknown-tiny (default)\n");
    }

    #[test]
    fn print_backends_and_frontends_list_each_with_what_it_supports() {
        assert_eq!(
            printed(&["--print", "backends"]),
            "stackvm: stackvm-unknown-unknown, stackvm-unknown-tiny\naltvm: \
             stackvm-unknown-unknown\n"
        );
        assert_eq!(printed(&["--print", "frontends"]), "calc: .calc\nrunic: .runic\n");
    }

    #[test]
    fn print_checkpoints_lists_each_with_its_aliases_and_extension() {
        assert_eq!(
            printed(&["--print", "checkpoints"]),
            "calc.ast (alias ast) -> .ast\nir.translated -> .translated.runic\nir.optimized \
             (alias ir) -> .runic\nir.lowered -> .lowered.runic\nstackvm.asm (alias asm) -> \
             .svm\naltvm.out -> .alt\n"
        );
    }

    #[test]
    fn two_prints_are_written_in_the_order_given_and_inputs_do_not_stop_them() {
        let text =
            printed(&["--print", "frontends", "a.calc", "--print", "targets", "--print=frontends"]);
        assert_eq!(
            text,
            "calc: .calc\nrunic: .runic\nstackvm-unknown-unknown\nstackvm-unknown-tiny\ncalc: \
             .calc\nrunic: .runic\n"
        );
    }

    #[test]
    fn a_print_does_not_read_standard_input() {
        let text = printed(&["--print", "targets", "-"]);
        assert!(text.starts_with("stackvm"));
    }

    #[test]
    fn each_unstable_setting_of_the_driver_fills_its_field() {
        let cases: Vec<(Vec<&str>, CompilationRequest)> = vec![
            (
                vec!["-Z", "pass-pipeline=builtin.module(cse)"],
                built(|r| r.pipeline = Some("builtin.module(cse)".into())),
            ),
            (
                vec!["-Zdump-pass-pipeline"],
                built(|r| {
                    r.ir = IrOptions {
                        dump_pass_pipeline: true,
                        ..IrOptions::default()
                    }
                }),
            ),
            (
                vec!["-Z", "print-ir-before=a,b", "-Z", "print-ir-before=c"],
                built(|r| {
                    r.ir.print_before = vec!["a".into(), "b".into(), "c".into()];
                }),
            ),
            (vec!["-Z", "print-ir-after=x"], built(|r| r.ir.print_after = vec!["x".into()])),
            (vec!["-Z", "print-generic"], built(|r| r.ir.print_generic = true)),
            (vec!["-Z", "print-locations"], built(|r| r.ir.print_locations = true)),
            (vec!["-Z", "no-verify"], built(|r| r.ir.verification = Verification::None)),
            (
                vec!["-Z", "verify-each=false"],
                built(|r| r.ir.verification = Verification::Once),
            ),
            (
                vec!["-Z", "verify-each=true"],
                built(|r| r.ir.verification = Verification::EachPass),
            ),
            (vec!["-Z", "verify-each"], built(|r| r.ir.verification = Verification::EachPass)),
        ];
        for (args, expected) in cases {
            assert_same(&request(&args), &expected);
        }
    }

    #[test]
    fn no_verify_with_an_explicit_verify_each_is_a_usage_error() {
        for args in [
            vec!["-Z", "no-verify", "-Z", "verify-each"],
            vec!["-Z", "verify-each=true", "-Z", "no-verify"],
        ] {
            let text = usage_text(&args);
            assert!(text.contains("-Z no-verify") && text.contains("-Z verify-each"), "{text}");
        }
    }

    #[test]
    fn an_unknown_unstable_name_lists_the_known_ones() {
        let text = usage_text(&["-Z", "nonesuch"]);
        assert!(text.contains("'-Z nonesuch'"), "{text}");
        for name in ["pass-pipeline", "verify-each", "stackvm-keep-subtractions"] {
            assert!(text.contains(&format!("'{name}'")), "{name} is missing from {text}");
        }
    }

    #[test]
    fn the_unknown_name_error_lists_only_settings_that_are_not_hidden() {
        let text = usage_text(&["-Z", "nonesuch"]);
        let compiler = toy();
        let driver = Driver::new(&compiler).build().unwrap();
        let hidden: Vec<&str> = driver
            .stage_flags()
            .unstable_commands()
            .flat_map(|command| command.get_arguments())
            .filter(|arg| arg.is_hide_set())
            .filter_map(Arg::get_long)
            .collect();
        for name in &hidden {
            assert!(!text.contains(&format!("'{name}'")), "{name} is listed in {text}");
        }
        assert!(text.contains("'pass-pipeline'"), "{text}");
    }

    #[test]
    fn an_empty_unstable_name_is_its_own_usage_error() {
        for args in [vec!["-Z", "=x"], vec!["-Z", ""], vec!["-Z="]] {
            let text = usage_text(&args);
            assert_eq!(
                text, "error: '-Z' needs the name of an unstable option; '-Z help' lists them\n",
                "{args:?}"
            );
        }
    }

    #[test]
    fn a_value_for_a_switch_and_no_value_for_a_required_one_are_usage_errors_naming_the_setting() {
        let text = usage_text(&["-Z", "print-generic=1"]);
        assert!(text.contains("-Z print-generic") && text.contains("takes none"), "{text}");
        let text = usage_text(&["-Z", "pass-pipeline"]);
        assert!(text.contains("-Z pass-pipeline") && text.contains("needs a value"), "{text}");
        let text = usage_text(&["-Z", "stackvm-keep-subtractions=yes"]);
        assert!(text.contains("-Z stackvm-keep-subtractions"), "{text}");
    }

    #[test]
    fn a_bad_value_of_an_unstable_setting_speaks_of_the_z_flag_and_not_of_a_long_one() {
        let text = usage_text(&["-Z", "verify-each=maybe"]);
        assert!(text.contains("-Z verify-each"), "{text}");
        assert!(!text.contains("--verify-each"), "{text}");
        assert!(!text.contains("Usage:"), "{text}");
    }

    #[test]
    fn z_help_is_help_text() {
        let compiler = toy();
        let driver = Driver::new(&compiler).build().unwrap();
        let Exit::Help(text) = exit(&["-Z", "help"]) else {
            panic!()
        };
        assert_eq!(text, driver.unstable_help());
        assert!(matches!(exit(&["-Zhelp=1"]), Exit::Usage(_)));
    }

    #[test]
    fn a_stackvm_flag_sets_its_options_and_without_one_the_request_holds_none() {
        let actual = request(&["--stackvm-comments"]);
        assert_eq!(actual.options::<StackVmOptions>(), Some(&StackVmOptions { comments: true }));
        assert_eq!(actual.options::<StackVmUnstable>(), None);
        assert_eq!(actual.options::<CalcOptions>(), None);

        let actual = request(&["-Z", "stackvm-keep-subtractions"]);
        assert_eq!(
            actual.options::<StackVmUnstable>(),
            Some(&StackVmUnstable {
                keep_subtractions: true
            })
        );
        assert_eq!(actual.options::<StackVmOptions>(), None);

        let actual = request(&["a.calc", "-O1"]);
        assert_eq!(actual.options::<StackVmOptions>(), None);
        assert_eq!(actual.options::<StackVmUnstable>(), None);
        assert_eq!(actual.options::<CalcOptions>(), None);

        let actual = request(&["--wrap-on-overflow"]);
        assert_eq!(
            actual.options::<CalcOptions>(),
            Some(&CalcOptions {
                wrap_on_overflow: true
            })
        );
    }

    #[derive(Args, Default, Debug, Clone, PartialEq)]
    struct Kinds {
        #[arg(long)]
        flag: bool,
        #[arg(long)]
        text: Option<String>,
        #[arg(long, value_delimiter = ',')]
        list: Vec<String>,
        #[arg(long, action = ArgAction::Count)]
        verbose: u8,
        #[arg(long, default_value = "none")]
        path: PathBuf,
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "bare")]
        maybe: Option<String>,
        #[arg(long, default_value_t = 3)]
        level: u8,
        #[arg(long)]
        left: bool,
        #[arg(long, conflicts_with = "left")]
        right: bool,
    }

    fn kinds(args: &[&str]) -> Result<Option<Kinds>, Exit> {
        let compiler = Compiler::builder().backend(backend("x", stable::<Kinds>)).build().unwrap();
        match run_on(&compiler, args, b"")? {
            Invocation::Compile { request, .. } => Ok(request.options::<Kinds>().cloned()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_stage_option_of_each_action_kind_round_trips_through_the_stage_command() {
        assert_eq!(kinds(&["a.x"]).unwrap(), None);

        let all = kinds(&[
            "--x-flag",
            "--x-text",
            "hello world",
            "--x-list=a,b",
            "--x-list",
            "c",
            "--x-verbose",
            "--x-verbose",
            "--x-verbose",
            "--x-path=-weird",
            "--x-maybe=given",
            "--x-level",
            "9",
            "--x-left",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            all,
            Kinds {
                flag: true,
                text: Some("hello world".into()),
                list: vec!["a".into(), "b".into(), "c".into()],
                verbose: 3,
                path: "-weird".into(),
                maybe: Some("given".into()),
                level: 9,
                left: true,
                right: false,
            }
        );
    }

    #[test]
    fn an_optional_value_left_out_gets_the_stage_default_and_a_flag_left_unwritten_its_default() {
        let bare = kinds(&["--x-maybe"]).unwrap().unwrap();
        assert_eq!(bare.maybe.as_deref(), Some("bare"));
        assert_eq!(bare.level, 3);
        assert_eq!(bare.path, PathBuf::from("none"));
        let none = kinds(&["--x-flag"]).unwrap().unwrap();
        assert_eq!(none.maybe, None);
        assert!(none.flag);
    }

    #[test]
    fn a_stage_option_value_may_hold_commas_and_equals_signs_in_a_scalar() {
        let kinds = kinds(&["--x-text=a,b=c"]).unwrap().unwrap();
        assert_eq!(kinds.text.as_deref(), Some("a,b=c"));
    }

    #[test]
    fn two_stage_options_that_conflict_are_a_usage_error_naming_the_prefixed_flags() {
        let Err(Exit::Usage(text)) = kinds(&["--x-left", "--x-right"]) else {
            panic!()
        };
        assert!(text.contains("--x-left") && text.contains("--x-right"), "{text}");
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn malformed_values_are_usage_errors_naming_the_flag() {
        for (args, flag) in [
            (vec!["--target", "a"], "--target"),
            (vec!["--emit", "=x"], "--emit"),
            (vec!["-O", "9"], "--opt-level"),
            (vec!["--debug=lots"], "--debug"),
            (vec!["--color", "sometimes"], "--color"),
            (vec!["--print", "nothing"], "--print"),
        ] {
            let text = usage_text(&args);
            assert!(text.contains(flag), "{args:?}: {text}");
        }
    }

    #[test]
    fn help_and_version_are_help_and_an_unknown_flag_is_a_usage_error() {
        let Exit::Help(help) = exit(&["--help"]) else {
            panic!()
        };
        assert!(help.contains("Usage:") && help.contains("--stackvm-comments"), "{help}");
        let Exit::Help(version) = exit(&["--version"]) else {
            panic!()
        };
        assert!(version.starts_with("runic "), "{version}");
        assert!(matches!(exit(&["-V"]), Exit::Help(_)));
        let text = usage_text(&["--nonesuch"]);
        assert!(text.contains("--nonesuch"), "{text}");
    }

    #[test]
    fn with_unstable_flags_not_allowed_z_is_a_usage_error() {
        let compiler = toy();
        let driver = Driver::new(&compiler).unstable(false).build().unwrap();
        let mut stdin: &[u8] = b"";
        let Err(Exit::Usage(text)) = driver.request(os(&["-Z", "no-verify"]), &mut stdin) else {
            panic!()
        };
        assert!(text.contains("-Z"), "{text}");
    }

    #[derive(Args, Default, Debug, Clone, PartialEq)]
    struct Ordered {
        #[arg(long, overrides_with = "no_color")]
        color: bool,
        #[arg(long, overrides_with = "color")]
        no_color: bool,
        #[arg(long)]
        other: bool,
        #[arg(long, value_delimiter = ',')]
        list: Vec<String>,
        #[arg(long, value_delimiter = ',', action = ArgAction::Set)]
        pair: Option<Vec<String>>,
        #[arg(long, action = ArgAction::Count)]
        verbose: u8,
        #[arg(long, num_args = 0..=1, require_equals = true)]
        maybe: Option<String>,
    }

    fn ordered(args: &[&str]) -> Ordered {
        let compiler =
            Compiler::builder().backend(backend("x", stable::<Ordered>)).build().unwrap();
        match run_on(&compiler, args, b"") {
            Ok(Invocation::Compile { request, .. }) => {
                request.options::<Ordered>().cloned().expect("a flag of the stage was written")
            }
            other => panic!("{args:?}: {other:?}"),
        }
    }

    #[test]
    fn a_stage_is_handed_its_flags_in_the_order_they_were_written() {
        let color = |o: &Ordered| (o.color, o.no_color);
        assert_eq!(color(&ordered(&["--x-no-color", "--x-color"])), (true, false));
        assert_eq!(color(&ordered(&["--x-color", "--x-no-color"])), (false, true));
        assert_eq!(
            color(&ordered(&["--x-no-color", "--x-other", "-O2", "--x-color", "-g", "a.calc"])),
            (true, false)
        );
        assert_eq!(
            color(&ordered(&["--x-color", "--x-other", "-O2", "--x-no-color", "-g", "a.calc"])),
            (false, true)
        );
        // A flag of the driver between them changes nothing.
        let both = ordered(&["--x-color", "--name", "u", "--x-no-color", "--x-other"]);
        assert_eq!((color(&both), both.other), ((false, true), true));
    }

    #[test]
    fn a_delimited_occurrence_reaches_the_stage_as_one_argument() {
        let kinds = ordered(&["--x-list=a,b", "--x-list=c", "--x-pair=d,e"]);
        assert_eq!(kinds.list, ["a", "b", "c"]);
        assert_eq!(kinds.pair, Some(vec!["d".to_owned(), "e".to_owned()]));
        let kinds = ordered(&["--x-list", "a,b", "--x-pair", "d,e"]);
        assert_eq!(kinds.list, ["a", "b"]);
        assert_eq!(kinds.pair, Some(vec!["d".to_owned(), "e".to_owned()]));
    }

    #[test]
    fn a_count_and_an_optional_value_left_out_keep_their_meaning_beside_other_flags() {
        let kinds =
            ordered(&["--x-verbose", "--x-other", "--x-verbose", "--x-maybe", "--x-verbose"]);
        assert_eq!(kinds.verbose, 3);
        assert!(kinds.other);
        assert_eq!(kinds.maybe, None);
    }

    #[derive(Args, Debug, Clone, PartialEq, Default)]
    struct Leveled {
        #[arg(long, default_value_t = 3)]
        level: u8,
        #[arg(long)]
        loud: bool,
    }

    fn render_nothing(
        _artifact: &Artifact,
        _session: &Session<'_>,
        _out: &mut dyn std::io::Write,
    ) -> Result<(), Report> {
        Ok(())
    }

    const LEVELED_CHECKPOINTS: &[CheckpointDecl] = &[CheckpointDecl {
        id: CheckpointId::new("lvl.out"),
        aliases: &[],
        extension: "out",
        render: render_nothing,
    }];

    /// A backend for the `stackvm` target that declares `Leveled` and does nothing else.
    struct LeveledBackend;

    impl Backend for LeveledBackend {
        fn name(&self) -> &'static str {
            "lvl"
        }

        fn supports(&self, triple: &Triple) -> bool {
            StackVmBackend.supports(triple)
        }

        fn targets(&self) -> Vec<Triple> {
            StackVmBackend.targets()
        }

        fn data_layout(&self, triple: &Triple) -> Result<DataLayoutSpec, Report> {
            StackVmBackend.data_layout(triple)
        }

        fn checkpoints(&self) -> &'static [CheckpointDecl] {
            LEVELED_CHECKPOINTS
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
            Ok(Flow::Continue(()))
        }

        fn options(&self) -> Option<OptionsDecl> {
            Some(OptionsDecl::of::<Leveled>())
        }
    }

    /// What a stage reads of `Leveled` for the command line `args`, through a session.
    fn level_read_by_the_stage(args: &[&str]) -> Leveled {
        let compiler = Compiler::builder()
            .frontend(CalcFrontend)
            .backend(LeveledBackend)
            .build()
            .unwrap();
        let mut line = vec!["--target", "stackvm-unknown-unknown", "a.calc"];
        line.extend(args);
        let (request, _) = compile_on(&compiler, &line);
        let session = compiler.session(request).unwrap();
        session.options::<Leveled>().cloned().expect("the session stores a value")
    }

    #[test]
    fn a_stage_reads_the_clap_default_whether_or_not_another_of_its_flags_was_written() {
        let nothing = level_read_by_the_stage(&[]);
        assert_eq!(
            nothing,
            Leveled {
                level: 3,
                loud: false
            }
        );
        assert_ne!(Leveled::default().level, 3);
        assert_eq!(
            level_read_by_the_stage(&["--lvl-loud"]),
            Leveled {
                level: 3,
                loud: true
            }
        );
        assert_eq!(level_read_by_the_stage(&["--lvl-level", "5"]).level, 5);
    }

    #[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Fast,
        #[allow(dead_code, reason = "the second value of the enum")]
        Slow,
    }

    #[derive(Args, Default, Debug, Clone, PartialEq)]
    struct Parsing {
        #[arg(long, allow_negative_numbers = true)]
        offset: Option<i32>,
        #[arg(long, value_enum, ignore_case = true)]
        mode: Option<Mode>,
    }

    #[test]
    fn a_stage_flag_that_allows_negative_numbers_or_any_case_is_parsed_so_by_the_main_command() {
        let compiler =
            Compiler::builder().backend(backend("x", stable::<Parsing>)).build().unwrap();
        let Ok(Invocation::Compile { request, .. }) =
            run_on(&compiler, &["--x-offset", "-1", "--x-mode", "FAST"], b"")
        else {
            panic!("the flags are not accepted")
        };
        assert_eq!(
            request.options::<Parsing>(),
            Some(&Parsing {
                offset: Some(-1),
                mode: Some(Mode::Fast)
            })
        );
    }
}
