//! The harness path: inputs that a frontend reads, and the flags that send an input through the
//! compiler.

use runic_compile::{
    Artifact, Backend, CheckpointDecl, CheckpointId, Checkpoints, Compiler, Flow, Frontend, Input,
    IrFrontend, Session, Triple,
};
use runic_ir::{DataLayoutSpec, Report, dialects::builtin::WorldRef, pass::OpPassManager};

use super::{Output, REPEATED_ADD, opt, opt_on};
use crate::compiler;

/// The compiler with the toy stages.
fn toy() -> Compiler {
    compiler()
}

/// The compiler with the frontend for IR alone.
fn ir_only() -> Compiler {
    Compiler::builder().frontend(IrFrontend).build().unwrap()
}

fn toy_opt(args: &[&str], stdin: &str) -> Output {
    opt_on(&toy(), args, stdin)
}

/// A calc file whose body has a constant expression to fold.
const FOLDABLE: &str = "fn k() = 1 + 2;\n";

/// What `FOLDABLE` is at `ir.optimized` with no pass run.
const FOLDABLE_IR: &str = r#"builtin.world {
    builtin.module public @m {
        builtin.function public extern("C") @k() -> i32 {
            %0 = arith.constant 1 : i32;
            %1 = arith.constant 2 : i32;
            %2 = arith.add %0, %1 <{ overflow = #builtin.overflow<checked signed> }>;
            builtin.ret %2 : (i32);
        };
    };
};
"#;

/// What `FOLDABLE` is once the canonicalizer has run.
const FOLDED_IR: &str = r#"builtin.world {
    builtin.module public @m {
        builtin.function public extern("C") @k() -> i32 {
            %3 = arith.constant 3 : i32;
            builtin.ret %3 : (i32);
        };
    };
};
"#;

/// A function with a subtraction of a constant, and one that adds.
const SUBTRACTS: &str = "fn f(x) = x - 3;\n";

/// A directory holding the files a test writes, removed when the test ends.
struct Files(tempfile::TempDir);

impl Files {
    fn new() -> Self {
        Self(tempfile::tempdir().unwrap())
    }

    /// Writes `text` to `name` and gives the path.
    fn write(&self, name: &str, text: &str) -> String {
        std::fs::write(self.0.path().join(name), text).unwrap();
        self.path(name)
    }

    fn path(&self, name: &str) -> String {
        self.0.path().join(name).to_str().unwrap().to_owned()
    }

    fn read(&self, name: &str) -> Option<String> {
        std::fs::read_to_string(self.0.path().join(name)).ok()
    }
}

// Which path an input takes

#[test]
fn a_runic_file_with_no_flag_of_the_compiler_is_read_as_ir_and_a_bare_function_prints_back() {
    let files = Files::new();
    let text = "builtin.function public extern(\"C\") @f() -> i32 {\n    %0 = arith.constant 1 : \
                i32;\n    builtin.ret %0 : (i32);\n};\n";
    let input = files.write("f.runic", text);
    let output = toy_opt(&[&input], "");
    output.succeeded();
    assert_eq!(output.stdout, text);
}

#[test]
fn the_same_file_with_a_stop_point_goes_through_the_frontend_which_refuses_a_bare_function() {
    let files = Files::new();
    let text = "builtin.function public extern(\"C\") @f() -> i32 {\n    %0 = arith.constant 1 : \
                i32;\n    builtin.ret %0 : (i32);\n};\n";
    let input = files.write("f.runic", text);
    let output = toy_opt(&[&input, "--stop-after", "ir.translated"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.contains("holds a 'builtin.function' at its top level"),
        "{}",
        output.stderr
    );
}

#[test]
fn a_calc_file_goes_through_the_frontend_of_a_compiler_that_has_it_and_is_read_as_ir_otherwise() {
    let files = Files::new();
    let input = files.write("m.calc", FOLDABLE);
    let output = toy_opt(&[&input], "");
    output.succeeded();
    assert_eq!(output.stdout, FOLDABLE_IR);

    let output = opt_on(&ir_only(), &[&input], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.starts_with("error: invalid operation"), "{}", output.stderr);
}

#[test]
fn standard_input_is_calc_with_a_language_and_ir_otherwise() {
    let output = toy_opt(&["--language", "calc", "--stop-after", "ast"], "fn q() = 4;\n");
    output.succeeded();
    assert_eq!(output.stdout, "file <stdin>\n  fn q()\n    int 4\n");

    // On the harness path, standard input with no language is read by the frontend of IR.
    let output = toy_opt(&["--stop-after", "ir.translated"], REPEATED_ADD);
    output.succeeded();
    assert!(output.stdout.starts_with("builtin.world {"), "{}", output.stdout);
    let output = toy_opt(&["--stop-after", "ir.translated"], "fn q() = 4;\n");
    output.failed();
}

#[test]
fn a_language_that_is_not_registered_is_an_error_of_the_case() {
    let output = opt_on(&ir_only(), &["--language", "calc"], "fn q() = 4;\n");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.contains("'calc'") && output.stderr.contains("runic"),
        "{}",
        output.stderr
    );
}

#[test]
fn the_ir_path_is_unchanged_for_input_the_harness_would_not_take() {
    let output = toy_opt(&["-p", "cse"], REPEATED_ADD);
    output.succeeded();
    assert_eq!(output.stdout, opt(&["-p", "cse"], REPEATED_ADD).stdout);
    assert!(output.stdout.starts_with("builtin.module public @m {"), "{}", output.stdout);
}

// What a calc input is

#[test]
fn a_calc_input_prints_the_ir_at_optimized_and_a_foldable_expression_is_not_folded() {
    let files = Files::new();
    let input = files.write("m.calc", FOLDABLE);
    let output = toy_opt(&[&input], "");
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(output.stdout, FOLDABLE_IR);
}

#[test]
fn with_a_pass_pipeline_it_runs_on_the_world_and_the_expression_is_folded() {
    let files = Files::new();
    let input = files.write("m.calc", FOLDABLE);
    let output = toy_opt(&[&input, "--pass-pipeline", "canonicalizer"], "");
    output.succeeded();
    assert_eq!(output.stdout, FOLDED_IR);
}

#[test]
fn a_pipeline_that_does_not_load_is_an_error_before_anything_is_read() {
    let output = toy_opt(&["x.calc", "-p", "nonesuch"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.contains("unknown pass 'nonesuch'"), "{}", output.stderr);
}

#[test]
fn a_target_with_one_backend_prints_the_listing() {
    let files = Files::new();
    let input = files.write("m.calc", "fn inc(x) = x + 1;\n");
    let output = toy_opt(&[&input, "--target", "stackvm-unknown-tiny"], "");
    output.succeeded();
    assert_eq!(output.stdout, "m::inc:\n  load 0\n  push 1\n  add\n  ret\n");
}

#[test]
fn a_target_that_two_backends_support_needs_one_to_be_named() {
    let files = Files::new();
    let input = files.write("m.calc", SUBTRACTS);
    let output = toy_opt(&[&input, "--target", "stackvm-unknown-unknown"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert_eq!(
        output.stderr,
        "error: more than one backend supports the target 'stackvm-unknown-unknown': choose one \
         with --backend; the candidates are 'stackvm', 'altvm'\n"
    );

    let output =
        toy_opt(&[&input, "--target", "stackvm-unknown-unknown", "--backend", "altvm"], "");
    output.succeeded();
    assert_eq!(output.stdout, "m::f/1\n");
}

#[test]
fn an_unknown_target_or_stop_point_is_an_error_of_the_case_naming_what_was_written() {
    let files = Files::new();
    let input = files.write("m.calc", SUBTRACTS);
    let output = toy_opt(&[&input, "--stop-after", "nope"], "");
    output.failed();
    assert!(output.stderr.contains("unknown checkpoint 'nope'"), "{}", output.stderr);
    let output = toy_opt(&[&input, "--target", "stackvm-unknown-nowhere"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.contains("stackvm-unknown-nowhere"), "{}", output.stderr);
}

#[test]
fn stopping_after_the_ast_prints_it_and_after_lowering_prints_the_lowered_ir() {
    let files = Files::new();
    let input = files.write("m.calc", SUBTRACTS);
    let output = toy_opt(&[&input, "--stop-after", "ast"], "");
    output.succeeded();
    assert_eq!(output.stdout, "file m\n  fn f(x)\n    sub\n      param x\n      int 3\n");

    let output =
        toy_opt(&[&input, "--stop-after", "ir.lowered", "--target", "stackvm-unknown-tiny"], "");
    output.succeeded();
    assert!(output.stdout.contains("arith.constant -3 : i32"), "{}", output.stdout);
    assert!(output.stdout.contains("data_layout"), "{}", output.stdout);
}

#[test]
fn a_runic_module_with_a_target_gives_a_listing() {
    let files = Files::new();
    let input = files.write("m.runic", REPEATED_ADD);
    let output = toy_opt(&[&input, "--target", "stackvm-unknown-tiny"], "");
    output.succeeded();
    assert!(
        output.stdout.starts_with("m::f:\n  load 0\n  load 1\n  add\n"),
        "{}",
        output.stdout
    );
}

// Emitting

#[test]
fn emit_writes_the_checkpoint_to_the_file_and_the_stop_point_is_still_printed() {
    let files = Files::new();
    let input = files.write("m.calc", SUBTRACTS);
    let ast = files.path("out.ast");
    let emit = format!("ast={ast}");
    let output = toy_opt(&[&input, "--emit", &emit], "");
    output.succeeded();
    assert_eq!(output.stdout.lines().next(), Some("builtin.world {"));
    assert_eq!(
        files.read("out.ast").unwrap(),
        "file m\n  fn f(x)\n    sub\n      param x\n      int 3\n"
    );

    // The file of the stop point itself is written from the result.
    let both = format!("ast={}", files.path("again.ast"));
    let output = toy_opt(&[&input, "--stop-after", "ast", "--emit", &both], "");
    output.succeeded();
    assert_eq!(files.read("again.ast").unwrap(), output.stdout);
    assert!(output.stdout.starts_with("file m\n"), "{}", output.stdout);
}

#[test]
fn several_emits_write_several_files() {
    let files = Files::new();
    let input = files.write("m.calc", SUBTRACTS);
    let first = format!("ast={}", files.path("a.ast"));
    let second = format!("ir.translated={}", files.path("t.runic"));
    let output = toy_opt(
        &[&input, "--target", "stackvm-unknown-tiny", "--emit", &first, "--emit", &second],
        "",
    );
    output.succeeded();
    assert!(output.stdout.starts_with("m::f:\n"), "{}", output.stdout);
    assert!(files.read("a.ast").unwrap().starts_with("file m\n"));
    assert!(files.read("t.runic").unwrap().starts_with("builtin.world {"));
}

#[test]
fn an_emit_without_a_path_or_with_split_input_file_is_refused_before_anything_is_read() {
    let output = toy_opt(&["--emit", "ast", "no-such-file.calc"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert_eq!(
        output.stderr,
        "error: --emit ast: the checkpoint needs a file to be written to: write '--emit \
         NAME=PATH'\n"
    );
    let output = toy_opt(&["--emit", "ast=-", "no-such-file.calc"], "");
    output.failed();
    assert!(
        output.stderr.contains("--emit ast=-: the checkpoint needs a file"),
        "{}",
        output.stderr
    );
    let output = toy_opt(&["--emit", "ast=", "no-such-file.calc"], "");
    output.failed();
    assert!(output.stderr.starts_with("error: --emit ast=: "), "{}", output.stderr);

    let output = toy_opt(&["--emit", "ast=x.ast", "--split-input-file", "no-such-file.calc"], "");
    output.failed();
    assert_eq!(
        output.stderr,
        "error: --emit is not available with --split-input-file: every case would write the same \
         file\n"
    );
}

// Where the result goes

#[test]
fn the_result_goes_to_the_file_of_o_and_a_failed_case_leaves_it_alone() {
    let files = Files::new();
    let input = files.write("m.calc", FOLDABLE);
    let out = files.path("out.runic");
    let output = toy_opt(&[&input, "-o", &out], "");
    output.succeeded();
    assert_eq!(output.stdout, "");
    assert_eq!(files.read("out.runic").unwrap(), FOLDABLE_IR);

    let broken = files.write("b.calc", "fn k() = y;\n");
    let other = files.path("other.runic");
    let output = toy_opt(&[&broken, "-o", &other], "");
    output.failed();
    assert_eq!(files.read("other.runic"), None);
}

#[test]
fn split_cases_print_in_order_and_an_error_names_a_line_of_the_whole_file() {
    let files = Files::new();
    let input = files.write("s.calc", "fn a(x) = x;\n// -----\nfn b(x) = y;\n");
    let output = toy_opt(&[&input, "--split-input-file"], "");
    output.failed();
    assert!(output.stdout.starts_with("builtin.world {"), "{}", output.stdout);
    assert!(!output.stdout.contains("@b"), "{}", output.stdout);
    assert!(output.stdout.contains("@a"), "{}", output.stdout);
    assert!(output.stderr.contains("s.calc:3:11"), "{}", output.stderr);
    assert!(output.stderr.contains("'y' is not a parameter"), "{}", output.stderr);
}

#[test]
fn two_good_cases_are_separated_by_the_marker() {
    let output = toy_opt(
        &["--language", "calc", "--stop-after", "ast", "--split-input-file"],
        "fn a(x) = x;\n// -----\n  \n// -----\nfn b() = 1;\n",
    );
    output.succeeded();
    assert_eq!(
        output.stdout,
        "file <stdin>\n  fn a(x)\n    param x\n// -----\nfile <stdin>\n  fn b()\n    int 1\n"
    );
}

// Diagnostics

#[test]
fn an_error_of_the_input_is_rendered_to_standard_error_and_fails_the_run() {
    let files = Files::new();
    let input = files.write("e.calc", "fn k() = 1 +;\n");
    let output = toy_opt(&[&input], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.starts_with("error: "), "{}", output.stderr);
    assert!(output.stderr.contains("e.calc:1:"), "{}", output.stderr);
}

#[test]
fn a_matching_expectation_succeeds_silently_and_a_wrong_one_is_a_mismatch() {
    let files = Files::new();
    let good =
        files.write("g.calc", "fn k() = 1 +; // expected-error {{expected an expression}}\n");
    let output = toy_opt(&[&good, "--verify-diagnostics"], "");
    output.succeeded();
    assert_eq!(output.stdout, "");
    assert_eq!(output.stderr, "");

    let wrong = files.write("w.calc", "fn k() = 1 +; // expected-error {{nonsense}}\n");
    let output = toy_opt(&[&wrong, "--verify-diagnostics"], "");
    output.failed();
    assert!(
        output.stderr.contains("expected error \"nonsense\" was not produced"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("unexpected error"), "{}", output.stderr);
}

// The stages' flags

#[test]
fn flags_of_the_stages_change_the_listing() {
    let files = Files::new();
    let input = files.write("m.calc", SUBTRACTS);
    let tiny = ["--target", "stackvm-unknown-tiny"];
    let plain = toy_opt(&[&input, tiny[0], tiny[1]], "");
    plain.succeeded();
    assert_eq!(plain.stdout, "m::f:\n  load 0\n  push -3\n  add\n  ret\n");

    let commented = toy_opt(&[&input, tiny[0], tiny[1], "--stackvm-comments"], "");
    commented.succeeded();
    assert!(
        commented.stdout.starts_with("; m::f: 1 parameter\nm::f:\n"),
        "{}",
        commented.stdout
    );

    let kept = toy_opt(&[&input, tiny[0], tiny[1], "-Z", "stackvm-keep-subtractions"], "");
    kept.succeeded();
    assert_eq!(kept.stdout, "m::f:\n  load 0\n  push 3\n  sub\n  ret\n");
}

#[test]
fn a_flag_of_a_stage_alone_sends_the_input_through_the_compiler() {
    let output = toy_opt(&["--wrap-on-overflow", "--language", "calc"], "fn f(x) = x + 1;\n");
    output.succeeded();
    assert!(output.stdout.contains("arith.add %0, %1;"), "{}", output.stdout);
    assert!(!output.stdout.contains("overflow = #builtin"), "{}", output.stdout);
}

#[test]
fn dash_z_help_lists_the_unstable_options_of_the_stages_and_an_unknown_name_is_an_error() {
    let output = toy_opt(&["-Z", "help"], "");
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert!(
        output.stdout.starts_with("Unstable options (-Z NAME[=VALUE]):\n"),
        "{}",
        output.stdout
    );
    assert!(output.stdout.contains("stackvm-keep-subtractions"), "{}", output.stdout);

    let output = toy_opt(&["-Z", "nonesuch", "x.runic"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.starts_with("error: unknown unstable option '-Z nonesuch'"),
        "{}",
        output.stderr
    );
}

#[test]
fn list_stages_lists_the_frontends_the_backends_and_the_checkpoints() {
    let output = toy_opt(&["--list-stages"], "");
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert_eq!(
        output.stdout,
        "frontends:\n  runic: .runic\n  calc: .calc\nbackends:\n  stackvm: \
         stackvm-unknown-unknown, stackvm-unknown-tiny\n  altvm: stackvm-unknown-unknown\n  wasm: \
         wasm32-unknown-unknown\ncheckpoints:\n  calc.ast (alias ast) -> .ast\n  ir.translated -> \
         .translated.runic\n  ir.optimized (alias ir) -> .runic\n  ir.lowered -> .lowered.runic\n  \
         stackvm.asm (alias asm) -> .svm\n  altvm.out -> .alt\n  wasm.wat (alias wat) -> .wat\n  \
         wasm.module (alias module) -> .core.wasm\n"
    );

    let output = opt_on(&ir_only(), &["--list-stages"], "");
    output.succeeded();
    assert_eq!(
        output.stdout,
        "frontends:\n  runic: .runic\nbackends:\ncheckpoints:\n  ir.translated -> \
         .translated.runic\n  ir.optimized (alias ir) -> .runic\n  ir.lowered -> .lowered.runic\n"
    );
}

// The flags of the IR options

#[test]
fn printing_ir_after_passes_goes_to_standard_error_and_the_pipeline_can_be_dumped() {
    let files = Files::new();
    let input = files.write("m.calc", FOLDABLE);
    let output = toy_opt(&[&input, "--print-ir-after=all", "--pass-pipeline", "cse"], "");
    output.succeeded();
    assert!(
        output.stderr.starts_with("// ----- IR after cse on builtin.world -----\n"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("arith.constant 1 : i32"), "{}", output.stderr);
    assert!(output.stdout.starts_with("builtin.world {"), "{}", output.stdout);

    let output = toy_opt(&[&input, "--dump-pass-pipeline", "--pass-pipeline", "cse"], "");
    output.succeeded();
    assert_eq!(output.stderr, "builtin.world(cse)\n");
    let output = toy_opt(&[&input, "--dump-pass-pipeline"], "");
    output.succeeded();
    assert_eq!(output.stderr, "builtin.world()\n");
}

#[test]
fn print_generic_and_print_locations_reach_the_printed_ir() {
    let files = Files::new();
    let input = files.write("m.calc", FOLDABLE);
    let output = toy_opt(&[&input, "--print-generic"], "");
    output.succeeded();
    assert!(output.stdout.contains("\"arith.add\""), "{}", output.stdout);
    let output = toy_opt(&[&input, "--print-locations"], "");
    output.succeeded();
    assert!(output.stdout.contains("m.calc"), "{}", output.stdout);
}

#[test]
fn flags_the_harness_has_no_setting_for_are_refused_naming_the_flag_and_what_sent_the_input() {
    for (flag, why) in [
        ("--verify-roundtrip", "--target"),
        ("--print-ir-after-change", "--target"),
        ("--print-ir-after-failure", "--target"),
    ] {
        let output = toy_opt(&[flag, "--target", "stackvm-unknown-tiny", "no-such-file.calc"], "");
        output.failed();
        assert_eq!(output.stdout, "");
        assert_eq!(
            output.stderr,
            format!(
                "error: {flag} is not available when the input goes through a frontend or a \
                 backend, as it does here: {why}\n"
            )
        );
    }
    let output = toy_opt(&["--verify-roundtrip", "no-such-file.calc"], "");
    output.failed();
    assert!(
        output
            .stderr
            .ends_with("here: 'no-such-file.calc' is read by the frontend 'calc'\n"),
        "{}",
        output.stderr
    );
    // On the path of IR they are not refused.
    toy_opt(&["--verify-roundtrip"], REPEATED_ADD).succeeded();
}

#[test]
fn a_flag_of_a_stage_that_collides_with_one_of_the_tool_is_an_error_before_anything_is_read() {
    #[derive(clap::Args, Default, Debug, Clone)]
    struct Colliding {
        #[arg(long)]
        emit: bool,
    }
    struct Collides;
    impl Frontend for Collides {
        fn name(&self) -> &'static str {
            "collides"
        }

        fn extensions(&self) -> &'static [&'static str] {
            &["collides"]
        }

        fn translate(
            &self,
            _: &Session<'_>,
            _: &[Input],
            _: WorldRef,
            _: &mut Checkpoints<'_>,
        ) -> Result<Flow<()>, Report> {
            Ok(Flow::Continue(()))
        }

        fn options(&self) -> Option<runic_compile::OptionsDecl> {
            Some(runic_compile::OptionsDecl::of::<Colliding>())
        }
    }
    let compiler = Compiler::builder().frontend(Collides).build().unwrap();
    let output = opt_on(&compiler, &["no-such-file.runic"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.starts_with("error: "), "{}", output.stderr);
    assert!(
        output
            .stderr
            .contains("'--emit' is declared by runic-opt and by the frontend 'collides'"),
        "{}",
        output.stderr
    );
}

// Faults

/// A frontend that panics when it translates.
struct Panics;

impl Frontend for Panics {
    fn name(&self) -> &'static str {
        "panics"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["boom"]
    }

    fn translate(
        &self,
        _: &Session<'_>,
        _: &[Input],
        _: WorldRef,
        _: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        panic!("the frontend panics on purpose")
    }
}

#[test]
fn a_frontend_that_panics_is_an_internal_error_of_the_tool() {
    let files = Files::new();
    let input = files.write("x.boom", "anything\n");
    let compiler = Compiler::builder().frontend(IrFrontend).frontend(Panics).build().unwrap();
    let output = opt_on(&compiler, &[&input], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output
            .stderr
            .starts_with("error: internal error: the frontend panics on purpose\n"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("the panic happened at"), "{}", output.stderr);
}

// The command

#[test]
fn help_begins_as_before_and_shows_the_headings_of_the_stages_of_the_compiler() {
    let output = toy_opt(&["--help"], "");
    output.succeeded();
    let usage = output.stdout.find("Usage: runic-opt [OPTIONS] [FILE]").expect("usage");
    assert!(output.stdout[..usage].starts_with("Reads textual IR"), "{}", output.stdout);
    assert!(output.stdout.contains("calc frontend options:"), "{}", output.stdout);
    assert!(output.stdout.contains("stackvm backend options:"), "{}", output.stdout);
    assert!(output.stdout.contains("--stackvm-comments"), "{}", output.stdout);
    for flag in ["--target", "--backend", "--stop-after", "--emit", "--language", "--list-stages"] {
        assert!(output.stdout.contains(flag), "{flag}: {}", output.stdout);
    }

    let output = opt_on(&ir_only(), &["--help"], "");
    output.succeeded();
    assert!(output.stdout.contains("Usage: runic-opt [OPTIONS] [FILE]"), "{}", output.stdout);
    assert!(!output.stdout.contains("frontend options:"), "{}", output.stdout);
    assert!(!output.stdout.contains("backend options:"), "{}", output.stdout);
}

#[test]
fn the_passes_of_the_stages_are_not_listed_but_a_pipeline_can_name_them() {
    let output = toy_opt(&["--list-passes"], "");
    output.succeeded();
    assert!(!output.stdout.contains("stackvm-legalize"), "{}", output.stdout);
    let output = toy_opt(&["-p", "stackvm-legalize"], REPEATED_ADD);
    assert!(!output.stderr.contains("unknown pass"), "{}", output.stderr);
}

// What an input with no operation is

/// What a unit named `name` that has no function is at `ir.optimized`.
fn empty_module(name: &str) -> String {
    format!("builtin.world {{\n    builtin.module public @{name} {{ }};\n}};\n")
}

#[test]
fn an_empty_calc_file_or_one_of_comments_alone_is_the_empty_module() {
    let files = Files::new();
    for (name, text) in [("e", ""), ("c", "// only a comment\n"), ("h", "# only a comment\n")] {
        let input = files.write(&format!("{name}.calc"), text);
        let output = toy_opt(&[&input], "");
        output.succeeded();
        assert_eq!(output.stderr, "", "{name}");
        assert_eq!(output.stdout, empty_module(name), "{name}");
    }
}

#[test]
fn empty_standard_input_with_a_language_is_the_empty_module() {
    let output = toy_opt(&["--language", "calc"], "");
    output.succeeded();
    assert_eq!(output.stdout, empty_module("\"<stdin>\""));
}

#[test]
fn a_blank_last_case_is_skipped_and_the_first_is_printed_alone() {
    let output =
        toy_opt(&["--language", "calc", "--split-input-file"], "fn a(x) = x;\n// -----\n\n");
    output.succeeded();
    assert_eq!(output.stderr, "");
    assert!(output.stdout.starts_with("builtin.world {"), "{}", output.stdout);
    assert_eq!(output.stdout.matches("builtin.world {").count(), 1, "{}", output.stdout);
    assert!(!output.stdout.contains("// -----"), "{}", output.stdout);
}

#[test]
fn the_ir_path_still_refuses_an_input_with_no_operation() {
    let output = toy_opt(&[], "// nothing\n");
    output.failed();
    assert_eq!(output.stderr, "error: no operation found in <stdin>\n");
    opt(&["--split-input-file"], "// a\n// -----\n\n").failed();
}

// What is resolved before the input is read

#[test]
fn a_pipeline_anchored_on_a_module_is_reported_before_the_input_is_read() {
    let output = toy_opt(&["no-such-file.calc", "-p", "builtin.module(cse)"], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(output.stderr.contains("builtin.module"), "{}", output.stderr);
    assert!(!output.stderr.contains("could not read"), "{}", output.stderr);
}

#[test]
fn an_error_of_the_request_is_printed_once_however_many_cases_there_are() {
    let files = Files::new();
    let input = files.write("s.calc", "fn a() = 1;\n// -----\nfn b() = 2;\n");
    let output = toy_opt(&[&input, "--split-input-file", "-p", "builtin.module(cse)"], "");
    output.failed();
    assert!(output.stderr.contains("builtin.module"), "{}", output.stderr);
    assert_eq!(output.stderr.matches("error:").count(), 1, "{}", output.stderr);
}

#[test]
fn a_stop_point_or_target_that_does_not_resolve_is_reported_once_before_the_input_is_read() {
    let output = toy_opt(&["no-such-file.calc", "--stop-after", "nope"], "");
    output.failed();
    assert_eq!(output.stderr.matches("error:").count(), 1, "{}", output.stderr);
    assert!(output.stderr.contains("unknown checkpoint 'nope'"), "{}", output.stderr);

    let files = Files::new();
    let input = files.write("s.calc", "fn a() = 1;\n// -----\nfn b() = 2;\n");
    let output = toy_opt(&[&input, "--split-input-file", "--target", "nowhere-x-y"], "");
    output.failed();
    assert_eq!(output.stderr.matches("error:").count(), 1, "{}", output.stderr);
    assert!(output.stderr.contains("nowhere-x-y"), "{}", output.stderr);
    assert!(!output.stderr.contains("could not read"), "{}", output.stderr);
}

#[test]
fn two_emits_of_one_checkpoint_by_id_or_alias_are_refused_with_both_texts() {
    let files = Files::new();
    let input = files.write("m.calc", SUBTRACTS);
    let (a, b) = (files.path("a.out"), files.path("b.out"));
    let (first, second) = (format!("ast={a}"), format!("calc.ast={b}"));
    let output = toy_opt(&[&input, "--emit", &first, "--emit", &second], "");
    output.failed();
    assert_eq!(output.stdout, "");
    assert_eq!(
        output.stderr,
        format!(
            "error: --emit {first} and --emit {second} name the same checkpoint: a checkpoint is \
             written to one file\n"
        )
    );
    assert!(files.read("a.out").is_none() && files.read("b.out").is_none());

    // The stop point is a checkpoint like any other.
    let (first, second) = (format!("ir={a}"), format!("ir.optimized={b}"));
    let output = toy_opt(&[&input, "--emit", &first, "--emit", &second], "");
    output.failed();
    assert!(output.stderr.contains("name the same checkpoint"), "{}", output.stderr);
    assert!(
        output.stderr.contains(&first) && output.stderr.contains(&second),
        "{}",
        output.stderr
    );
}

#[test]
fn one_emit_of_the_stop_point_still_writes_its_file_from_the_result() {
    let files = Files::new();
    let input = files.write("m.calc", SUBTRACTS);
    let emit = format!("ir={}", files.path("o.runic"));
    let output = toy_opt(&[&input, "--emit", &emit], "");
    output.succeeded();
    assert_eq!(files.read("o.runic").unwrap(), output.stdout);
}

// What the result is

/// A backend whose artifact is the three bytes `[0xff, 0x00, 0xfe]`, which are not UTF-8.
struct Bytes;

const BYTES_OUT: CheckpointId = CheckpointId::new("bytes.out");

const BYTES_CHECKPOINTS: &[CheckpointDecl] = &[CheckpointDecl {
    id: BYTES_OUT,
    aliases: &[],
    extension: "bin",
    render: render_bytes,
}];

fn render_bytes(_: &Artifact, _: &Session<'_>, out: &mut dyn std::io::Write) -> Result<(), Report> {
    out.write_all(&[0xff, 0x00, 0xfe])
        .map_err(|error| Report::msg(error.to_string()))
}

impl Backend for Bytes {
    fn name(&self) -> &'static str {
        "bytes"
    }

    fn supports(&self, triple: &Triple) -> bool {
        triple.as_str() == "bytes-unknown-unknown"
    }

    fn targets(&self) -> Vec<Triple> {
        vec!["bytes-unknown-unknown".parse().unwrap()]
    }

    fn data_layout(&self, _: &Triple) -> Result<DataLayoutSpec, Report> {
        runic_compile::testing::StackVmBackend.data_layout(&"stackvm-unknown-tiny".parse().unwrap())
    }

    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        BYTES_CHECKPOINTS
    }

    fn add_passes(&self, _: &Session<'_>, _: &mut OpPassManager) -> Result<(), Report> {
        Ok(())
    }

    fn emit(
        &self,
        _: &Session<'_>,
        _: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        Ok(match checkpoints.reach(BYTES_OUT, String::new())? {
            Flow::Continue(_) => Flow::Continue(()),
            Flow::Break(stopped) => Flow::Break(stopped),
        })
    }
}

#[test]
fn an_artifact_that_is_not_utf_8_reaches_standard_output_and_the_file_of_o_unchanged() {
    let compiler = Compiler::builder().frontend(IrFrontend).backend(Bytes).build().unwrap();
    let files = Files::new();
    let input = files.write("m.runic", "builtin.world {\n    builtin.module public @m { };\n};\n");

    let mut stdin: &[u8] = b"";
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let code = crate::run_on(
        &compiler,
        [input.as_str(), "--target", "bytes-unknown-unknown"].map(std::ffi::OsString::from),
        &mut stdin,
        &mut stdout,
        &mut stderr,
        || Ok(()),
    );
    assert_eq!(code, std::process::ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, [0xff, 0x00, 0xfe]);

    let out = files.path("out.bin");
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let code = crate::run_on(
        &compiler,
        [input.as_str(), "--target", "bytes-unknown-unknown", "-o", out.as_str()]
            .map(std::ffi::OsString::from),
        &mut stdin,
        &mut stdout,
        &mut stderr,
        || Ok(()),
    );
    assert_eq!(code, std::process::ExitCode::SUCCESS, "{}", String::from_utf8_lossy(&stderr));
    assert!(stdout.is_empty());
    assert_eq!(std::fs::read(&out).unwrap(), [0xff, 0x00, 0xfe]);
}

// `-Z`

#[test]
fn dash_z_is_on_the_command_only_when_a_stage_declares_unstable_options() {
    let output = opt_on(&ir_only(), &["--help"], "");
    output.succeeded();
    assert!(!output.stdout.contains("unstable option"), "{}", output.stdout);
    let output = opt_on(&ir_only(), &["-Z", "x"], REPEATED_ADD);
    output.failed();
    assert!(output.stderr.contains("unexpected argument '-Z'"), "{}", output.stderr);

    let output = toy_opt(&["--help"], "");
    assert!(output.stdout.contains("Set an unstable option"), "{}", output.stdout);
    let output = toy_opt(&["-Z", "help"], "");
    output.succeeded();
    assert!(output.stdout.contains("stackvm-keep-subtractions"), "{}", output.stdout);
}
