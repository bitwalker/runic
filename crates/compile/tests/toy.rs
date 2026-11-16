//! End-to-end tests of the harness with the toy language and the toy machines. They use only the
//! public API of the crate.

use std::{fs, path::Path};

use runic_compile::{
    Compilation, CompilationRequest, Compiler, EmitRequest, EmitTo, Emitted, Input, IrFrontend,
    OptLevel, OutputFile,
    testing::{
        AltVmBackend, CalcFrontend, CalcOptions, StackVmBackend, StackVmOptions, StackVmUnstable,
    },
};
use runic_ir::diagnostics::{Label, LayeredSourceProvider, SourceKey, VisitDiagnostic};

const TINY: &str = "stackvm-unknown-tiny";
const UNKNOWN: &str = "stackvm-unknown-unknown";

/// A program with constants to fold and a subtraction of a constant.
const PROGRAM: &str =
    "# adds and subtracts\nfn add(a, b) = a + b;\nfn f(x) = add(x - 3, 2 + 3 * 4);\n";

const PROGRAM_LISTING: &str = "\
demo::add:
  load 0
  load 1
  add
  ret
demo::f:
  load 0
  push -3
  add
  push 14
  call demo::add
  ret
";

/// The checkpoints of the route of a `.calc` file compiled for a stack machine, in order.
const ROUTE: [&str; 5] = ["calc.ast", "ir.translated", "ir.optimized", "ir.lowered", "stackvm.asm"];

fn compiler() -> Compiler {
    Compiler::builder()
        .frontend(CalcFrontend)
        .frontend(IrFrontend)
        .backend(StackVmBackend)
        .backend(AltVmBackend)
        .build()
        .expect("the toy stages form a valid compiler")
}

fn calc(name: &str, text: &str) -> Input {
    Input::memory(name, text)
}

/// A request to compile `inputs` for `target`, with a name.
fn request(inputs: Vec<Input>, target: Option<&str>) -> CompilationRequest {
    let mut request = CompilationRequest::new();
    request.name = Some("demo".into());
    request.inputs = inputs;
    request.target = target.map(|target| target.parse().unwrap());
    request
}

fn emit_to_memory(request: &mut CompilationRequest, checkpoints: &[&str]) {
    for checkpoint in checkpoints {
        request.emit.push(EmitRequest {
            checkpoint: (*checkpoint).into(),
            to: EmitTo::Memory,
        });
    }
}

fn compile(request: CompilationRequest) -> Compilation {
    compiler().compile(request, &mut ())
}

/// The text delivered to memory for `checkpoint`.
fn memory(compilation: &Compilation, checkpoint: &str) -> String {
    let found = compilation.emitted().iter().find(|emitted| {
        matches!(emitted, Emitted::Memory { .. }) && emitted.checkpoint().as_str() == checkpoint
    });
    let bytes = found
        .unwrap_or_else(|| panic!("nothing was delivered to memory for '{checkpoint}'"))
        .bytes()
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn error_of(compilation: &Compilation) -> String {
    let error = compilation.error().expect("the compilation failed");
    let mut message = error.to_string();
    for frame in error.contexts() {
        message.push_str(&format!(" / {}", frame.message()));
    }
    message
}

fn succeeded(compilation: &Compilation) {
    assert!(compilation.succeeded(), "{:?}", compilation.error().map(|e| e.to_string()));
}

fn ids(compilation: &Compilation) -> Vec<&'static str> {
    compilation.reached().iter().map(|id| id.as_str()).collect()
}

#[test]
fn a_calc_file_compiles_to_a_svm_file_named_after_the_unit() {
    let sources = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let path = sources.path().join("prog.calc");
    fs::write(&path, PROGRAM).unwrap();
    let mut request = request(vec![Input::path(&path)], Some(TINY));
    request.output_dir = Some(out.path().to_owned());

    let compilation = compile(request);

    succeeded(&compilation);
    let written = out.path().join("demo.svm");
    // The functions are in the module of the input; the file is named after the unit.
    assert_eq!(
        fs::read_to_string(&written).unwrap(),
        PROGRAM_LISTING.replace("demo::", "prog::")
    );
    assert_eq!(fs::read_dir(out.path()).unwrap().count(), 1, "nothing else is written");
    assert!(matches!(
        compilation.emitted(),
        [Emitted::File { checkpoint, path }]
            if checkpoint.as_str() == "stackvm.asm" && path == &written
    ));
    assert_eq!(ids(&compilation), ROUTE);
}

#[test]
fn each_checkpoint_of_the_route_is_a_stop_point_whose_output_is_the_default() {
    for (at, stop) in ROUTE.iter().enumerate() {
        let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(TINY));
        request.stop_after = Some((*stop).into());
        emit_to_memory(&mut request, &[stop]);
        let compilation = compile(request);

        succeeded(&compilation);
        assert_eq!(ids(&compilation), ROUTE[..=at], "stopping after '{stop}'");
        assert_eq!(compilation.artifact().unwrap().checkpoint().as_str(), *stop);
        assert_eq!(compilation.emitted().len(), 1, "stopping after '{stop}'");
        assert!(!memory(&compilation, stop).is_empty(), "stopping after '{stop}'");

        // With no request for an output, the output of the stop point goes where
        // `--output-file` says.
        let mut request = request_for_default(stop);
        request.output_file = Some(OutputFile::Stdout);
        let compilation = compile(request);
        succeeded(&compilation);
        let [Emitted::Stdout { checkpoint, bytes }] = compilation.emitted() else {
            panic!("expected one output on standard output: {:?}", compilation.emitted());
        };
        assert_eq!(checkpoint.as_str(), *stop);
        assert!(!bytes.is_empty());
    }
}

fn request_for_default(stop: &str) -> CompilationRequest {
    let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(TINY));
    request.stop_after = Some(stop.into());
    request
}

#[test]
fn every_checkpoint_emitted_to_memory_in_one_run_shows_what_each_stage_changed() {
    let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(TINY));
    emit_to_memory(&mut request, &ROUTE);
    let compilation = compile(request);

    succeeded(&compilation);
    let delivered: Vec<_> = compilation
        .emitted()
        .iter()
        .map(|emitted| emitted.checkpoint().as_str())
        .collect();
    assert_eq!(delivered, ROUTE);
    let texts: Vec<_> = ROUTE.iter().map(|checkpoint| memory(&compilation, checkpoint)).collect();
    for (one, text) in texts.iter().enumerate() {
        for other in &texts[one + 1..] {
            assert_ne!(text, other);
        }
    }

    let translated = memory(&compilation, "ir.translated");
    let optimized = memory(&compilation, "ir.optimized");
    let lowered = memory(&compilation, "ir.lowered");
    // The optimization folded `2 + 3 * 4`.
    assert!(
        translated.contains("arith.mul") && !optimized.contains("arith.mul"),
        "{optimized}"
    );
    assert!(optimized.contains("arith.constant 14 : i32"), "{optimized}");
    // The lowering turned the subtraction of a constant into an addition.
    assert!(optimized.contains("arith.sub") && !lowered.contains("arith.sub"), "{lowered}");
    assert!(lowered.contains("arith.constant -3 : i32"), "{lowered}");
    assert_eq!(memory(&compilation, "stackvm.asm"), PROGRAM_LISTING);
}

#[test]
fn the_translated_ir_of_a_program_verifies_and_prints_as_expected() {
    let mut request =
        request(vec![calc("demo.calc", "fn add(a, b) = a + b;\nfn five() = add(2, 3);\n")], None);
    request.stop_after = Some("ir.translated".into());
    emit_to_memory(&mut request, &["ir.translated"]);
    request.set_options(CalcOptions {
        wrap_on_overflow: true,
    });
    let compilation = compile(request);

    succeeded(&compilation);
    let expected = r#"builtin.world {
    builtin.module public @demo {
        builtin.function public extern("C") @add(%0: i32, %1: i32) -> i32 {
            %2 = arith.add %0, %1;
            builtin.ret %2 : (i32);
        };
        builtin.function public extern("C") @five() -> i32 {
            %3 = arith.constant 2 : i32;
            %4 = arith.constant 3 : i32;
            %5 = builtin.call ::@demo::@add(%3, %4) : extern("C") (i32, i32) -> i32;
            builtin.ret %5 : (i32);
        };
    };
};
"#;
    assert_eq!(memory(&compilation, "ir.translated"), expected);
}

#[test]
fn opt_level_zero_leaves_the_optimized_ir_equal_to_the_translated_ir() {
    let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(TINY));
    request.opt_level = OptLevel::None;
    emit_to_memory(&mut request, &["ir.translated", "ir.optimized"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(memory(&compilation, "ir.translated"), memory(&compilation, "ir.optimized"));
}

#[test]
fn a_pipeline_replaces_the_optimization_and_the_machine_still_legalizes() {
    let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(TINY));
    request.pipeline = Some("builtin.world(dce)".into());
    request.ir.dump_pass_pipeline = true;
    emit_to_memory(&mut request, &["ir.optimized", "ir.lowered"]);
    let compilation = compile(request);

    succeeded(&compilation);
    let dump = compilation.pass_output();
    assert!(dump.contains("builtin.world(dce)"), "{dump}");
    assert!(dump.contains("builtin.world(stackvm-legalize)"), "{dump}");
    assert!(!dump.contains("canonicalizer"), "{dump}");
    // `dce` folds nothing, so the constants of the program are still there, and the machine's
    // pass still rewrote the subtraction.
    assert!(memory(&compilation, "ir.optimized").contains("arith.mul"));
    assert!(memory(&compilation, "ir.lowered").contains("arith.constant -3 : i32"));
}

const LIBRARY: &str = "fn add(a, b) = a + b;\n";

/// A module in the textual IR that calls `add` of the module `lib` by its path.
const CALLER: &str = r#"builtin.module public @user {
    builtin.function public extern("C") @twice(%x: i32) -> i32 {
        %r = builtin.call ::@lib::@add(%x, %x) : extern("C") (i32, i32) -> i32;
        builtin.ret %r : (i32);
    };
};
"#;

const LINKED_LISTING: &str = "\
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
";

#[test]
fn a_runic_module_calls_a_function_a_calc_file_defines_when_calc_comes_first() {
    let request =
        request(vec![calc("lib.calc", LIBRARY), Input::memory("user.runic", CALLER)], Some(TINY));
    let mut request = request;
    emit_to_memory(&mut request, &["stackvm.asm"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(memory(&compilation, "stackvm.asm"), LINKED_LISTING);
}

#[test]
fn a_runic_module_calls_a_function_a_calc_file_defines_when_runic_comes_first() {
    let mut request =
        request(vec![Input::memory("user.runic", CALLER), calc("lib.calc", LIBRARY)], Some(TINY));
    emit_to_memory(&mut request, &["stackvm.asm"]);
    let compilation = compile(request);

    succeeded(&compilation);
    // The frontends run in the order of their first input: the module of the `.runic` file comes
    // first in the world.
    let listing = memory(&compilation, "stackvm.asm");
    let (user, lib) = (listing.find("user::twice:").unwrap(), listing.find("lib::add:").unwrap());
    assert!(user < lib, "{listing}");
    assert!(listing.contains("  call lib::add\n"), "{listing}");
}

#[test]
fn two_calc_files_in_one_unit_give_two_modules_in_input_order() {
    let mut request = request(
        vec![
            calc("second.calc", "fn two() = 2;\n"),
            calc("first.calc", "fn one() = 1;\nfn also_one() = one();\n"),
        ],
        Some(TINY),
    );
    emit_to_memory(&mut request, &["ir.translated", "stackvm.asm"]);
    let compilation = compile(request);

    succeeded(&compilation);
    let translated = memory(&compilation, "ir.translated");
    assert_eq!(translated.matches("builtin.module public").count(), 2, "{translated}");
    assert_eq!(
        memory(&compilation, "stackvm.asm"),
        "\
second::two:
  push 2
  ret
first::one:
  push 1
  ret
first::also_one:
  call first::one
  ret
"
    );
}

#[test]
fn the_ast_is_an_indented_tree_one_file_after_another() {
    let mut request = request_with_ast_files();
    emit_to_memory(&mut request, &["calc.ast"]);
    let compilation = compile(request);
    succeeded(&compilation);
    assert_eq!(
        memory(&compilation, "calc.ast"),
        "\
file first
  fn f(a, b)
    sub
      sub
        param a
        param b
      mul
        int 2
        add
          param b
          int 1
file second
  fn g()
    int 7
  fn h(x)
    add
      call g
      param x
"
    );
    assert_eq!(ids(&compilation), ["calc.ast"], "the IR is built after the checkpoint");
    assert_eq!(compilation.emitted().len(), 1);
}

fn request_with_ast_files() -> CompilationRequest {
    let mut request = request(
        vec![
            calc("first.calc", "fn f(a, b) = a - b - 2 * (b + 1);\n"),
            calc("second.calc", "fn g() = 7;\nfn h(x) = g() + x;\n"),
        ],
        None,
    );
    request.stop_after = Some("ast".into());
    request
}

#[test]
fn a_triple_with_two_backends_is_an_error_that_lists_them() {
    let compilation = compile(request(vec![calc("demo.calc", PROGRAM)], Some(UNKNOWN)));

    assert!(!compilation.succeeded());
    let message = error_of(&compilation);
    assert!(message.contains("'stackvm'") && message.contains("'altvm'"), "{message}");
    assert!(message.contains("--backend"), "{message}");
}

#[test]
fn a_named_backend_settles_a_triple_with_two_backends() {
    let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(UNKNOWN));
    request.backend = Some("altvm".into());
    emit_to_memory(&mut request, &["altvm.out"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(memory(&compilation, "altvm.out"), "demo::add/2\ndemo::f/1\n");
    assert_eq!(ids(&compilation).last(), Some(&"altvm.out"));
}

#[test]
fn the_other_backend_of_the_triple_is_chosen_by_name_too() {
    let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(UNKNOWN));
    request.backend = Some("stackvm".into());
    emit_to_memory(&mut request, &["stackvm.asm"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(memory(&compilation, "stackvm.asm"), PROGRAM_LISTING);
}

#[test]
fn a_triple_with_one_backend_needs_no_backend_name() {
    let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(TINY));
    emit_to_memory(&mut request, &["stackvm.asm"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(memory(&compilation, "stackvm.asm"), PROGRAM_LISTING);
}

#[test]
fn no_target_and_no_stop_point_is_an_error_that_asks_for_a_target() {
    let compilation = compile(request(vec![calc("demo.calc", PROGRAM)], None));

    assert!(!compilation.succeeded());
    let message = error_of(&compilation);
    assert!(message.contains("--target"), "{message}");
}

/// The labels of `report`, each with the name of the source and the text it covers.
fn labelled(compilation: &Compilation) -> Vec<(String, String, String)> {
    struct Collect(Vec<(runic_ir::diagnostics::SourceSpan, String)>);
    impl VisitDiagnostic for Collect {
        fn label(&mut self, label: Label<'_>) {
            self.0
                .push((label.span, label.message.map(|m| m.to_string()).unwrap_or_default()));
        }
    }
    let mut collect = Collect(Vec::new());
    compilation.error().unwrap().as_diagnostic().visit(&mut collect);
    let sources = compilation.context().sources();
    let sources = LayeredSourceProvider::session_only(&*sources);
    collect
        .0
        .into_iter()
        .map(|(span, message)| {
            let SourceKey::Session(_) = span.source() else {
                panic!("the label is not in a source of the session");
            };
            let resolved = sources.resolve(span.source()).expect("the source of the label");
            let range = span.range();
            let text = resolved.source.text.unwrap();
            (
                resolved.source.display_name.to_owned(),
                text[range.start() as usize..range.end() as usize].to_owned(),
                message,
            )
        })
        .collect()
}

#[test]
fn a_syntax_error_in_a_calc_file_fails_with_a_label_in_that_file() {
    let mut request = request(
        vec![calc("fine.calc", "fn ok() = 1;\n"), calc("bad.calc", "fn f(x) = x +* 2;\n")],
        None,
    );
    request.stop_after = Some("ir.translated".into());
    let compilation = compile(request);

    assert!(!compilation.succeeded());
    assert!(
        error_of(&compilation).contains("expected an expression"),
        "{}",
        error_of(&compilation)
    );
    assert_eq!(
        labelled(&compilation),
        [("bad.calc".to_owned(), "*".to_owned(), "expected an expression here".to_owned())]
    );
    assert!(compilation.reached().is_empty(), "no checkpoint is reached after a parse error");
}

#[test]
fn each_kind_of_mistake_in_a_calc_file_is_labelled_at_what_the_user_wrote() {
    let cases = [
        ("fn f(x) = y;", "the name 'y' is not a parameter", "y"),
        ("fn f(x) = g(x);", "'g' is called but not defined", "g"),
        ("fn f(x) = f(x);", "'f' is called but not defined", "f"),
        (
            "fn g(a, b) = a;\nfn f(x) = g(x);",
            "'g' takes 2 arguments, but the call gives 1",
            "g(x)",
        ),
        ("fn g() = 1;\nfn g() = 2;", "'g' is defined twice", "g"),
        ("fn g(a, a) = a;", "the parameter 'a' of 'g' is named twice", "a"),
        ("fn g() = 2147483648;", "'2147483648' is not a 32-bit integer", "2147483648"),
        ("fn g() = 1 $ 2;", "unexpected character '$'", "$"),
        ("fn g() = (1;", "expected ')'", ";"),
        ("fn g() = 1", "expected ';', found the end of the file", "1"),
        ("g() = 1;", "expected 'fn'", "g"),
    ];
    for (text, message, covered) in cases {
        let mut request = request(vec![calc("bad.calc", text)], None);
        request.stop_after = Some("calc.ast".into());
        let compilation = compile(request);

        assert!(!compilation.succeeded(), "{text}");
        let error = error_of(&compilation);
        assert!(error.contains(message), "{text}: {error}");
        let labels = labelled(&compilation);
        assert_eq!(labels.len(), 1, "{text}: {labels:?}");
        assert_eq!(labels[0].0, "bad.calc", "{text}");
        assert_eq!(labels[0].1, covered, "{text}");
    }
}

#[test]
fn the_largest_literal_and_an_empty_file_are_fine() {
    let mut request = request(vec![calc("big.calc", "fn m() = 2147483647;\n# nothing\n")], None);
    request.stop_after = Some("ir.translated".into());
    request.output_file = Some(OutputFile::Stdout);
    succeeded(&compile(request));
    let mut request = request_for_default("ir.translated");
    request.output_file = Some(OutputFile::Stdout);
    request.inputs = vec![calc("empty.calc", "")];
    succeeded(&compile(request));
}

#[test]
fn files_read_lists_a_calc_path_input() {
    let sources = tempfile::tempdir().unwrap();
    let path = sources.path().join("demo.calc");
    fs::write(&path, PROGRAM).unwrap();
    let mut request = request(vec![Input::path(&path)], None);
    request.stop_after = Some("ir.optimized".into());
    request.output_file = Some(OutputFile::Stdout);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(compilation.files_read(), [path]);
}

#[test]
fn memory_inputs_are_not_files_read() {
    let mut request = request(vec![calc("demo.calc", PROGRAM)], None);
    request.stop_after = Some("ir.optimized".into());
    request.output_file = Some(OutputFile::Stdout);
    let compilation = compile(request);

    succeeded(&compilation);
    assert!(compilation.files_read().is_empty());
}

#[test]
fn the_name_of_the_unit_is_the_name_of_the_output_file_not_the_name_of_the_input() {
    let out = tempfile::tempdir().unwrap();
    let mut request = request(vec![calc("input_stem.calc", "fn one() = 1;\n")], Some(TINY));
    request.name = Some("chosen".into());
    request.output_dir = Some(out.path().to_owned());
    let compilation = compile(request);

    succeeded(&compilation);
    assert!(Path::new(&out.path().join("chosen.svm")).is_file());
    assert_eq!(
        fs::read_to_string(out.path().join("chosen.svm")).unwrap(),
        "input_stem::one:\n  push 1\n  ret\n"
    );
}

#[test]
fn without_a_name_the_stem_of_the_first_input_names_the_unit() {
    let out = tempfile::tempdir().unwrap();
    let mut request = request(vec![calc("first_input.calc", "fn one() = 1;\n")], Some(TINY));
    request.name = None;
    request.output_dir = Some(out.path().to_owned());
    succeeded(&compile(request));

    assert!(out.path().join("first_input.svm").is_file());
}

#[test]
fn wrapping_on_overflow_changes_the_translated_ir() {
    let translated = |options: Option<CalcOptions>| {
        let mut request = request(vec![calc("demo.calc", "fn f(a, b) = a * b - a + b;\n")], None);
        request.stop_after = Some("ir.translated".into());
        emit_to_memory(&mut request, &["ir.translated"]);
        if let Some(options) = options {
            request.set_options(options);
        }
        let compilation = compile(request);
        succeeded(&compilation);
        memory(&compilation, "ir.translated")
    };
    let trapping = translated(None);
    let wrapping = translated(Some(CalcOptions {
        wrap_on_overflow: true,
    }));

    assert_eq!(trapping.matches("checked signed").count(), 3, "{trapping}");
    assert_eq!(wrapping.matches("checked signed").count(), 0, "{wrapping}");
    assert_ne!(trapping, wrapping);
    assert_eq!(translated(Some(CalcOptions::default())), trapping);
}

#[test]
fn the_stack_machine_rejects_an_operation_it_cannot_run_and_names_it() {
    let divides = r#"builtin.module public @user {
    builtin.function public extern("C") @flip(%x: i32) -> i32 {
        %y = arith.bnot %x;
        builtin.ret %y : (i32);
    };
};
"#;
    let compilation = compile(request(vec![Input::memory("user.runic", divides)], Some(TINY)));

    assert!(!compilation.succeeded());
    let message = error_of(&compilation);
    assert!(message.contains("arith.bnot") && message.contains("user::flip"), "{message}");
    assert_eq!(ids(&compilation), ["ir.translated", "ir.optimized"], "it fails while lowering");
}

#[test]
fn the_machine_that_lowers_nothing_accepts_what_the_stack_machine_rejects() {
    let flips = r#"builtin.module public @user {
    builtin.function public extern("C") @flip(%x: i32) -> i32 {
        %y = arith.bnot %x;
        builtin.ret %y : (i32);
    };
};
"#;
    let mut request = request(vec![Input::memory("user.runic", flips)], Some(UNKNOWN));
    request.backend = Some("altvm".into());
    emit_to_memory(&mut request, &["altvm.out"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(memory(&compilation, "altvm.out"), "user::flip/1\n");
}

#[test]
fn the_largest_literal_is_subtracted_by_adding_its_negation() {
    let mut request = request(vec![calc("demo.calc", "fn f(x) = x - 2147483647;\n")], Some(TINY));
    emit_to_memory(&mut request, &["ir.lowered", "stackvm.asm"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(
        memory(&compilation, "stackvm.asm"),
        "demo::f:\n  load 0\n  push -2147483647\n  add\n  ret\n"
    );
}

#[test]
fn a_subtraction_of_the_smallest_integer_is_left_alone_because_its_negation_does_not_fit() {
    let text = r#"builtin.module public @user {
    builtin.function public extern("C") @f(%x: i32) -> i32 {
        %c = arith.constant -2147483648 : i32;
        %y = arith.sub %x, %c;
        builtin.ret %y : (i32);
    };
};
"#;
    let mut request = request(vec![Input::memory("user.runic", text)], Some(TINY));
    emit_to_memory(&mut request, &["stackvm.asm"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(
        memory(&compilation, "stackvm.asm"),
        "user::f:\n  load 0\n  push -2147483648\n  sub\n  ret\n"
    );
}

#[test]
fn a_subtraction_of_a_value_that_is_not_a_constant_stays_a_subtraction() {
    let mut request = request(vec![calc("demo.calc", "fn f(a, b) = a - b;\n")], Some(TINY));
    emit_to_memory(&mut request, &["stackvm.asm"]);
    let compilation = compile(request);

    succeeded(&compilation);
    assert_eq!(
        memory(&compilation, "stackvm.asm"),
        "demo::f:\n  load 0\n  load 1\n  sub\n  ret\n"
    );
}

#[test]
fn a_subtraction_read_as_unsigned_or_of_another_width_is_not_rewritten() {
    let text = r#"builtin.module public @m {
        builtin.function public extern("C") @unsigned(%x: i32) -> i32 {
            %c = arith.constant 3 : i32;
            %r = arith.sub %x, %c <{ overflow = #builtin.overflow<checked unsigned> }>;
            builtin.ret %r : (i32);
        };
        builtin.function public extern("C") @wide(%x: i64) -> i64 {
            %c = arith.constant 3 : i64;
            %r = arith.sub %x, %c;
            builtin.ret %r : (i64);
        };
        builtin.function public extern("C") @signed(%x: i32) -> i32 {
            %c = arith.constant 3 : i32;
            %r = arith.sub %x, %c <{ overflow = #builtin.overflow<checked signed> }>;
            builtin.ret %r : (i32);
        };
    };"#;
    let mut request = request(vec![Input::memory("m.runic", text)], Some(TINY));
    request.opt_level = runic_compile::OptLevel::None;
    request.stop_after = Some("ir.lowered".into());
    emit_to_memory(&mut request, &["ir.lowered"]);
    let compilation = compile(request);

    succeeded(&compilation);
    let lowered = memory(&compilation, "ir.lowered");
    assert_eq!(lowered.matches("arith.sub").count(), 2, "{lowered}");
    assert!(lowered.contains("checked unsigned"), "{lowered}");
    assert!(lowered.contains("arith.constant 3 : i64"), "{lowered}");
    assert!(lowered.contains("arith.constant -3 : i32"), "{lowered}");
}

#[test]
fn a_function_body_that_is_too_long_is_an_error_and_not_a_stack_overflow() {
    let deep = format!("fn f(a) = {}a{};\n", "(".repeat(100_000), ")".repeat(100_000));
    let long = format!("fn f(a) = a{};\n", " + a".repeat(100_000));
    for text in [deep, long] {
        let mut request = request(vec![calc("demo.calc", &text)], Some(TINY));
        request.stop_after = Some("calc.ast".into());
        emit_to_memory(&mut request, &["calc.ast"]);
        let compilation = compile(request);
        let error = compilation.error().expect("the body is too long").to_string();
        assert!(error.contains("too long"), "{error}");
    }
}

#[test]
fn the_comments_option_puts_a_line_before_each_function_in_the_listing() {
    let listing = |options: Option<StackVmOptions>| {
        let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(TINY));
        emit_to_memory(&mut request, &["stackvm.asm"]);
        if let Some(options) = options {
            request.set_options(options);
        }
        let compilation = compile(request);
        succeeded(&compilation);
        memory(&compilation, "stackvm.asm")
    };
    let commented = listing(Some(StackVmOptions { comments: true }));
    assert!(commented.starts_with("; demo::add: 2 parameters\ndemo::add:\n"), "{commented}");
    assert!(commented.contains("; demo::f: 1 parameter\ndemo::f:\n"), "{commented}");
    assert_eq!(listing(None), PROGRAM_LISTING);
    assert_eq!(listing(Some(StackVmOptions::default())), PROGRAM_LISTING);
}

#[test]
fn keeping_subtractions_leaves_one_in_the_lowered_ir_and_in_the_listing() {
    let compiled = |options: Option<StackVmUnstable>| {
        let mut request = request(vec![calc("demo.calc", PROGRAM)], Some(TINY));
        emit_to_memory(&mut request, &["ir.lowered", "stackvm.asm"]);
        if let Some(options) = options {
            request.set_options(options);
        }
        let compilation = compile(request);
        succeeded(&compilation);
        (memory(&compilation, "ir.lowered"), memory(&compilation, "stackvm.asm"))
    };
    let (lowered, listing) = compiled(Some(StackVmUnstable {
        keep_subtractions: true,
    }));
    assert!(lowered.contains("arith.sub"), "{lowered}");
    assert!(listing.contains("  push 3\n  sub\n"), "{listing}");

    let (lowered, listing) = compiled(None);
    assert!(!lowered.contains("arith.sub"), "{lowered}");
    assert_eq!(listing, PROGRAM_LISTING);
}
