//! `toyc`, the toy compiler of `examples/toyc.rs`, driven through [Driver::run] the way its
//! `main` drives it. The compiler built here must match the example's.

use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use runic_compile::{
    Compiler, IrFrontend,
    testing::{AltVmBackend, CalcFrontend, StackVmBackend},
};
use runic_driver::{Driver, EXIT_FAILURE, EXIT_USAGE};

/// The compiler of `examples/toyc.rs`: the example must match it.
fn compiler() -> Compiler {
    Compiler::builder()
        .frontend(CalcFrontend)
        .frontend(IrFrontend)
        .backend(StackVmBackend)
        .backend(AltVmBackend)
        .default_target("stackvm-unknown-tiny")
        .build()
        .expect("the toy stages form a valid compiler")
}

/// What a run of `toyc` did.
struct Run {
    code: ExitCode,
    stdout: String,
    stderr: String,
}

impl Run {
    fn exited_with(&self, code: u8) -> bool {
        format!("{:?}", self.code) == format!("{:?}", ExitCode::from(code))
    }

    /// Asserts that the run succeeded and wrote nothing to either stream.
    fn assert_silent_success(&self) {
        assert!(self.exited_with(0), "{:?}: {}", self.code, self.stderr);
        assert_eq!((self.stdout.as_str(), self.stderr.as_str()), ("", ""));
    }
}

fn toyc_with_input(args: &[&str], stdin: &str) -> Run {
    let compiler = compiler();
    let driver = Driver::new(&compiler)
        .name("toyc")
        .version("1.2.3")
        .about("Compiles calc, a toy language, for two toy stack machines")
        .build()
        .expect("the toy driver builds");
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let code = driver.run(
        args.iter().map(OsString::from),
        &mut stdin.as_bytes(),
        &mut stdout,
        &mut stderr,
    );
    Run {
        code,
        stdout: String::from_utf8(stdout).unwrap(),
        stderr: String::from_utf8(stderr).unwrap(),
    }
}

fn toyc(args: &[&str]) -> Run {
    toyc_with_input(args, "")
}

/// Writes `text` to `name` in `dir` and gives the path as text.
fn write(dir: &Path, name: &str, text: &str) -> String {
    let path = dir.join(name);
    fs::write(&path, text).unwrap();
    path.to_str().unwrap().to_owned()
}

fn path(dir: &Path, name: &str) -> String {
    dir.join(name).to_str().unwrap().to_owned()
}

fn read(dir: &Path, name: &str) -> String {
    fs::read_to_string(dir.join(name)).unwrap_or_else(|e| panic!("{name} is not there: {e}"))
}

fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// A program with constants to fold and a subtraction of a constant.
const PROGRAM: &str =
    "# adds and subtracts\nfn add(a, b) = a + b;\nfn f(x) = add(x - 3, 2 + 3 * 4);\n";

const LISTING: &str = "\
f::add:
  load 0
  load 1
  add
  ret
f::f:
  load 0
  push -3
  add
  push 14
  call f::add
  ret
";

/// A temporary directory holding `f.calc`, and a directory beside it for outputs.
fn workspace() -> (tempfile::TempDir, String, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let input = write(dir.path(), "f.calc", PROGRAM);
    let out = dir.path().join("out");
    fs::create_dir(&out).unwrap();
    (dir, input, out)
}

fn out_str(out: &Path) -> &str {
    out.to_str().unwrap()
}

#[test]
fn a_calc_file_compiles_into_the_output_directory_as_a_svm_file_named_after_the_input() {
    let (_dir, input, out) = workspace();

    toyc(&[&input, "--output-dir", out_str(&out)]).assert_silent_success();

    assert_eq!(names(&out), ["f.svm"]);
    assert_eq!(read(&out, "f.svm"), LISTING);
}

#[test]
fn an_output_directory_that_is_not_there_is_created_and_the_directory_of_an_output_file_is_not() {
    let (dir, input, _out) = workspace();
    let nested = dir.path().join("made").join("here");

    toyc(&[&input, "--output-dir", out_str(&nested)]).assert_silent_success();
    assert_eq!(names(&nested), ["f.svm"]);

    let file = path(&dir.path().join("absent"), "f.svm");
    let run = toyc(&[&input, "-o", &file]);
    assert!(run.exited_with(1), "{}", run.stderr);
    assert!(run.stderr.contains("could not write the output"), "{}", run.stderr);
    assert!(!dir.path().join("absent").exists());
}

#[test]
fn the_output_file_flag_names_the_listing() {
    let (dir, input, _out) = workspace();
    let file = path(dir.path(), "out.svm");

    toyc(&[&input, "-o", &file]).assert_silent_success();

    assert_eq!(read(dir.path(), "out.svm"), LISTING);
}

#[test]
fn each_checkpoint_of_the_route_is_emitted_to_a_file_with_the_extension_it_declares() {
    let route = [
        ("calc.ast", "f.ast"),
        ("ast", "f.ast"),
        ("ir.translated", "f.translated.runic"),
        ("ir.optimized", "f.runic"),
        ("ir", "f.runic"),
        ("ir.lowered", "f.lowered.runic"),
        ("stackvm.asm", "f.svm"),
        ("asm", "f.svm"),
    ];
    for (name, file) in route {
        let (_dir, input, out) = workspace();

        toyc(&[&input, "--emit", name, "--stop-after", name, "--output-dir", out_str(&out)])
            .assert_silent_success();

        assert_eq!(names(&out), [file], "--emit {name}");
        assert!(!read(&out, file).is_empty(), "--emit {name}");
    }
}

#[test]
fn stopping_after_the_optimized_ir_writes_no_listing() {
    let (_dir, input, out) = workspace();

    toyc(&[&input, "--stop-after", "ir", "--output-dir", out_str(&out)]).assert_silent_success();

    // The output of the stop point is written, and the stack machine's is not.
    assert_eq!(names(&out), ["f.runic"]);
}

#[test]
fn every_checkpoint_emitted_in_one_run_gives_one_file_each() {
    let (_dir, input, out) = workspace();
    let mut args = vec![input.as_str(), "--output-dir", out_str(&out)];
    for name in ["ast", "ir.translated", "ir", "ir.lowered", "asm"] {
        args.extend(["--emit", name]);
    }

    toyc(&args).assert_silent_success();

    assert_eq!(
        names(&out),
        ["f.ast", "f.lowered.runic", "f.runic", "f.svm", "f.translated.runic"]
    );
    assert_eq!(read(&out, "f.svm"), LISTING);
    assert!(read(&out, "f.ast").starts_with("file f\n"));
}

#[test]
fn the_comments_flag_puts_a_comment_line_before_each_function() {
    let (_dir, input, out) = workspace();

    toyc(&[&input, "--stackvm-comments", "--output-dir", out_str(&out)]).assert_silent_success();

    let listing = read(&out, "f.svm");
    assert!(listing.starts_with("; f::add: 2 parameters\nf::add:\n"), "{listing}");
    assert!(listing.contains("; f::f: 1 parameter\nf::f:\n"), "{listing}");
}

#[test]
fn the_unstable_keep_subtractions_setting_keeps_a_subtraction() {
    let (_dir, input, out) = workspace();

    toyc(&[&input, "-Z", "stackvm-keep-subtractions", "--output-dir", out_str(&out)])
        .assert_silent_success();

    let listing = read(&out, "f.svm");
    assert!(listing.contains("  push 3\n  sub\n"), "{listing}");
}

#[test]
fn the_unstable_keep_subtractions_flag_is_not_a_stable_flag() {
    let (_dir, input, _out) = workspace();

    let run = toyc(&[&input, "--stackvm-keep-subtractions"]);

    assert!(run.exited_with(EXIT_USAGE), "{}", run.stderr);
}

#[test]
fn wrapping_on_overflow_changes_the_translated_ir() {
    let translate = |extra: &[&str]| {
        let (_dir, input, out) = workspace();
        let mut args = vec![input.as_str(), "--stop-after", "ir.translated"];
        args.extend(["--output-dir", out_str(&out)]);
        args.extend(extra);
        toyc(&args).assert_silent_success();
        read(&out, "f.translated.runic")
    };

    let trapping = translate(&[]);
    let wrapping = translate(&["--wrap-on-overflow"]);

    assert!(trapping.contains("checked signed"), "{trapping}");
    assert!(!wrapping.contains("checked signed"), "{wrapping}");
}

#[test]
fn a_triple_two_backends_support_is_a_usage_error_that_names_both() {
    let (_dir, input, out) = workspace();

    let run = toyc(&[&input, "--target", "stackvm-unknown-unknown", "--output-dir", out_str(&out)]);

    assert!(run.exited_with(EXIT_USAGE), "{}", run.stderr);
    assert!(
        run.stderr.contains("'stackvm'") && run.stderr.contains("'altvm'"),
        "{}",
        run.stderr
    );
    assert!(run.stderr.contains("--backend"), "{}", run.stderr);
    assert!(names(&out).is_empty());
}

#[test]
fn a_named_backend_settles_a_triple_two_backends_support() {
    let (_dir, input, out) = workspace();

    toyc(&[
        &input,
        "--target",
        "stackvm-unknown-unknown",
        "--backend",
        "altvm",
        "--output-dir",
        out_str(&out),
    ])
    .assert_silent_success();

    assert_eq!(names(&out), ["f.alt"]);
    assert_eq!(read(&out, "f.alt"), "f::add/2\nf::f/1\n");
}

#[test]
fn print_lists_what_the_compiler_offers() {
    let print = |what: &str| {
        let run = toyc(&["--print", what]);
        assert!(run.exited_with(0), "{what}: {}", run.stderr);
        assert_eq!(run.stderr, "");
        run.stdout
    };

    assert_eq!(print("targets"), "stackvm-unknown-unknown\nstackvm-unknown-tiny (default)\n");
    assert_eq!(
        print("backends"),
        "stackvm: stackvm-unknown-unknown, stackvm-unknown-tiny\naltvm: stackvm-unknown-unknown\n"
    );
    assert_eq!(print("frontends"), "calc: .calc\nrunic: .runic\n");
    assert_eq!(
        print("checkpoints"),
        "\
calc.ast (alias ast) -> .ast
ir.translated -> .translated.runic
ir.optimized (alias ir) -> .runic
ir.lowered -> .lowered.runic
stackvm.asm (alias asm) -> .svm
altvm.out -> .alt
"
    );
}

#[test]
fn help_names_the_compiler_and_groups_the_flags_of_each_stage() {
    let run = toyc(&["--help"]);

    assert!(run.exited_with(0), "{}", run.stderr);
    assert!(run.stdout.contains("Compiles calc, a toy language, for two toy stack machines"));
    assert!(run.stdout.contains("Usage: toyc "), "{}", run.stdout);
    assert!(run.stdout.contains("calc frontend options:"), "{}", run.stdout);
    assert!(run.stdout.contains("stackvm backend options:"), "{}", run.stdout);
    assert!(run.stdout.contains("--wrap-on-overflow"));
    assert!(run.stdout.contains("--stackvm-comments"));
}

#[test]
fn the_version_flags_name_the_compiler_and_runic() {
    let long = toyc(&["--version"]);
    let short = toyc(&["-V"]);

    assert!(long.exited_with(0) && short.exited_with(0));
    assert_eq!(short.stdout, "toyc 1.2.3\n");
    let lines: Vec<_> = long.stdout.lines().collect();
    assert_eq!(lines.len(), 2, "{}", long.stdout);
    assert_eq!(lines[0], "toyc 1.2.3");
    assert!(lines[1].starts_with("runic "), "{}", lines[1]);
}

#[test]
fn a_program_from_standard_input_is_listed_on_standard_output() {
    let run = toyc_with_input(&["--language", "calc", "-", "-o", "-"], "fn k(x) = x;\n");

    assert!(run.exited_with(0), "{}", run.stderr);
    assert_eq!(run.stderr, "");
    assert!(run.stdout.ends_with("::k:\n  load 0\n  ret\n"), "{}", run.stdout);
}

#[test]
fn the_name_flag_names_the_output_file_of_a_program_from_standard_input() {
    let dir = tempfile::tempdir().unwrap();

    let run = toyc_with_input(
        &[
            "--language",
            "calc",
            "-",
            "--name",
            "unit",
            "--output-dir",
            dir.path().to_str().unwrap(),
        ],
        "fn k(x) = x;\n",
    );

    assert!(run.exited_with(0), "{}", run.stderr);
    assert_eq!(names(dir.path()), ["unit.svm"]);
    assert!(read(dir.path(), "unit.svm").ends_with("::k:\n  load 0\n  ret\n"));
}

#[test]
fn a_calc_file_and_a_runic_file_are_one_compilation() {
    let dir = tempfile::tempdir().unwrap();
    let calc = write(dir.path(), "lib.calc", "fn add(a, b) = a + b;\n");
    let runic = write(
        dir.path(),
        "user.runic",
        r#"builtin.module public @user {
    builtin.function public extern("C") @twice(%x: i32) -> i32 {
        %r = builtin.call ::@lib::@add(%x, %x) : extern("C") (i32, i32) -> i32;
        builtin.ret %r : (i32);
    };
};
"#,
    );
    let file = path(dir.path(), "both.svm");

    toyc(&[&calc, &runic, "-o", &file]).assert_silent_success();

    assert_eq!(
        read(dir.path(), "both.svm"),
        "\
lib::add:
  load 0
  load 1
  add
  ret
user::twice:
  load 0
  load 0
  call lib::add
  ret
"
    );
}

#[test]
fn a_syntax_error_in_a_calc_file_fails_and_names_the_file_and_the_line() {
    let dir = tempfile::tempdir().unwrap();
    let bad = write(dir.path(), "bad.calc", "fn ok() = 1;\nfn g() = 1 +* 2;\n");

    let run = toyc(&[&bad, "--output-dir", dir.path().to_str().unwrap()]);

    assert!(run.exited_with(EXIT_FAILURE), "{}", run.stderr);
    assert!(run.stderr.contains("expected an expression"), "{}", run.stderr);
    assert!(run.stderr.contains("bad.calc:2:"), "{}", run.stderr);
    assert_eq!(run.stdout, "");
    assert_eq!(names(dir.path()), ["bad.calc"]);
}

#[test]
fn opt_level_zero_leaves_a_foldable_expression_and_the_default_folds_it() {
    let optimized = |extra: &[&str]| {
        let (_dir, input, out) = workspace();
        let mut args = vec![input.as_str(), "--stop-after", "ir", "--output-dir", out_str(&out)];
        args.extend(extra);
        toyc(&args).assert_silent_success();
        read(&out, "f.runic")
    };

    let folded = optimized(&[]);
    let unfolded = optimized(&["-O0"]);

    assert!(
        folded.contains("arith.constant 14 : i32") && !folded.contains("arith.mul"),
        "{folded}"
    );
    assert!(unfolded.contains("arith.mul"), "{unfolded}");
}
