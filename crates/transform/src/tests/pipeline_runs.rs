//! What running a pipeline does, stated as it holds: where a typed pass reaches with implicit
//! nesting, what each registered pass changes at each level of a module, that the pass manager
//! verifies its input before the first pass, what a verifier failure after a pass reports, and
//! how much of the source a rendered diagnostic shows.

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::cell::Cell;

use runic_ir::{
    Context, EntityMut, Op, Operation, OperationName, OperationRef, Report,
    diagnostics::{Diagnostic, Label, LabelStyle, RenderConfig, SourceProvider, VisitDiagnostic},
    dialects::builtin::Function,
    parse::{ParserConfig, parse_any},
    pass::{
        Nesting, OperationPass, Pass, PassExecutionState, PassInstrumentation, PassPipeline,
        PostPassStatus,
    },
};

fn new_context() -> Rc<Context> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    context
}

/// The line of the first primary label of `diagnostic`, if it has one with a known span.
fn primary_line(diagnostic: &dyn Diagnostic, context: &Context) -> Option<usize> {
    struct Primary(Option<runic_ir::SourceSpan>);
    impl VisitDiagnostic for Primary {
        fn label(&mut self, label: Label<'_>) {
            if self.0.is_none() && label.style == LabelStyle::Primary {
                self.0 = Some(label.span);
            }
        }
    }
    let mut primary = Primary(None);
    diagnostic.visit(&mut primary);
    let span = primary.0.filter(|span| !span.is_unknown())?;
    let sources = context.sources();
    let sources: &dyn SourceProvider = &*sources;
    let position = sources.line_column(span.source().id(), span.range().start())?;
    Some(position.line().to_usize())
}

/// Implicit nesting nests a typed pass one level: `local2reg` named alone for a `builtin.world`
/// whose function is inside a module runs on nothing, and says nothing; under
/// `recursive(builtin.function(..))`, or with the anchors in between named, it runs.
#[test]
fn a_typed_pass_on_a_world_reaches_only_the_functions_directly_inside() {
    const WORLD: &str = "\
builtin.world {
    builtin.module public @m {
        builtin.function public extern(\"C\") @f(%a: i32) -> i32 locals(i32) {
            builtin.store_local 0, %a;
            %v = builtin.load_local 0 : i32;
            builtin.ret %v : (i32);
        };
    };
};";
    let run = |pipeline: &str| {
        let context = new_context();
        let world = parse_any(ParserConfig::new(context.clone()), "world.runic", WORLD)
            .into_result()
            .unwrap();
        let before = world.borrow().to_string();
        let pipeline = PassPipeline::parse_elements(&world.borrow().name(), pipeline).unwrap();
        let mut pm = pipeline.load(context.clone(), Nesting::Implicit).unwrap();
        pm.run(world).unwrap();
        let changed = world.borrow().to_string() != before;
        drop(pm);
        changed
    };
    assert!(!run("local2reg"), "one level: the world holds no function directly");
    assert!(run("recursive(builtin.function(local2reg))"));
    assert!(run("builtin.module(local2reg)"));
    assert!(run("builtin.module(builtin.function(local2reg))"));
}

const PASSES: [&str; 8] = [
    "canonicalizer",
    "cfg-to-scf",
    "control-flow-sink",
    "cse",
    "dce",
    "local2reg",
    "sccp",
    "sink-operand-defs",
];

/// A module with something for most passes to change in `@f` and `@g`, a local to promote in
/// `@k`, and a module inside it with constants to fold in `@h` and a local to promote in `@l`.
const MODULE: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32, %b: i32) -> i32 {
        %x = arith.add %a, %b;
        %y = arith.add %a, %b;
        %z = arith.add %x, %y;
        builtin.ret %z : (i32);
    };
    builtin.function public extern("C") @g(%c: i1, %a: i32) -> i32 locals(i32) {
        builtin.store_local 0, %a;
        %dead = arith.add %a, %a;
        cf.cond_br %c ^t, ^e : (i1);
    ^t:
        %v = builtin.load_local 0 : i32;
        builtin.ret %v : (i32);
    ^e:
        builtin.ret %a : (i32);
    };
    builtin.function public extern("C") @k(%a: i32) -> i32 locals(i32) {
        builtin.store_local 0, %a;
        %v = builtin.load_local 0 : i32;
        builtin.ret %v : (i32);
    };
    builtin.module public @inner {
        builtin.function public extern("C") @h(%a: i32) -> i32 {
            %one = arith.constant 1 : i32;
            %two = arith.add %one, %one;
            %r = arith.add %a, %two;
            builtin.ret %r : (i32);
        };
        builtin.function public extern("C") @l(%a: i32) -> i32 locals(i32) {
            builtin.store_local 0, %a;
            %v = builtin.load_local 0 : i32;
            builtin.ret %v : (i32);
        };
    };
};"#;

/// What each registered pass changes, by the functions it changes, when it is written at the
/// top of the pipeline, under `builtin.function`, under `any`, and under `recursive(..)`: the
/// typed pass (`local2reg`) reaches the functions directly inside the module unless the pipeline
/// says `recursive(..)`, and a pass on any operation that runs on the module walks into the
/// nested module by itself.
#[test]
fn what_each_pass_does_at_each_level_of_a_module() {
    fn functions(module: &Operation) -> Vec<(String, String)> {
        let mut found = Vec::new();
        module.prewalk_all(|op: &Operation| {
            if let Some(function) = op.downcast_ref::<Function>() {
                found.push((runic_ir::Symbol::name(function).as_str().into(), op.to_string()));
            }
        });
        found
    }

    let mut table = String::new();
    for pass in PASSES {
        for form in [
            "builtin.module(PASS)",
            "builtin.module(builtin.function(PASS))",
            "builtin.module(any(PASS))",
            "builtin.module(recursive(builtin.function(PASS)))",
            "builtin.module(recursive(any(PASS)))",
        ] {
            let text = form.replace("PASS", pass);
            let context = new_context();
            let module = parse_any(ParserConfig::new(context.clone()), "levels.runic", MODULE)
                .into_result()
                .expect("a valid module");
            let before = functions(&module.borrow());
            let mut pm = PassPipeline::parse(&text)
                .unwrap()
                .load(context.clone(), Nesting::Implicit)
                .unwrap();
            let result = pm.run(module);
            let after = functions(&module.borrow());
            let changed = before
                .iter()
                .zip(&after)
                .filter(|(b, a)| b.1 != a.1)
                .map(|(b, _)| b.0.as_str())
                .collect::<Vec<_>>()
                .join(",");
            let outcome = match &result {
                Ok(()) => "ok".to_string(),
                Err(err) => format!("error: {err}"),
            };
            table += &format!("{text} [{changed}] {outcome}\n");
            drop(pm);
        }
    }
    // Each line: the pipeline, the functions it changed, the outcome.
    let expected = "\
builtin.module(canonicalizer) [g,h] ok
builtin.module(builtin.function(canonicalizer)) [g] ok
builtin.module(any(canonicalizer)) [g,h] ok
builtin.module(recursive(builtin.function(canonicalizer))) [g,h] ok
builtin.module(recursive(any(canonicalizer))) [g,h] ok
builtin.module(cfg-to-scf) [g] ok
builtin.module(builtin.function(cfg-to-scf)) [g] ok
builtin.module(any(cfg-to-scf)) [g] ok
builtin.module(recursive(builtin.function(cfg-to-scf))) [g] ok
builtin.module(recursive(any(cfg-to-scf))) [g] ok
builtin.module(control-flow-sink) [] ok
builtin.module(builtin.function(control-flow-sink)) [] ok
builtin.module(any(control-flow-sink)) [] ok
builtin.module(recursive(builtin.function(control-flow-sink))) [] ok
builtin.module(recursive(any(control-flow-sink))) [] ok
builtin.module(cse) [f,g] ok
builtin.module(builtin.function(cse)) [f,g] ok
builtin.module(any(cse)) [f,g] ok
builtin.module(recursive(builtin.function(cse))) [f,g] ok
builtin.module(recursive(any(cse))) [f,g] ok
builtin.module(dce) [g] ok
builtin.module(builtin.function(dce)) [g] ok
builtin.module(any(dce)) [g] ok
builtin.module(recursive(builtin.function(dce))) [g] ok
builtin.module(recursive(any(dce))) [g] ok
builtin.module(local2reg) [k] ok
builtin.module(builtin.function(local2reg)) [k] ok
builtin.module(any(local2reg)) [k] ok
builtin.module(recursive(builtin.function(local2reg))) [k,l] ok
builtin.module(recursive(any(local2reg))) [k,l] ok
builtin.module(sccp) [h] ok
builtin.module(builtin.function(sccp)) [] ok
builtin.module(any(sccp)) [h] ok
builtin.module(recursive(builtin.function(sccp))) [h] ok
builtin.module(recursive(any(sccp))) [h] ok
builtin.module(sink-operand-defs) [] ok
builtin.module(builtin.function(sink-operand-defs)) [] ok
builtin.module(any(sink-operand-defs)) [] ok
builtin.module(recursive(builtin.function(sink-operand-defs))) [] ok
builtin.module(recursive(any(sink-operand-defs))) [] ok
";
    assert_eq!(table, expected);
}

/// IR parsed without verification is verified by the pass manager before its first pass when
/// the verifier is on, and the failure names no pass; with the verifier off the pass runs.
#[test]
fn the_pass_manager_verifies_its_input_before_the_first_pass() {
    struct Count(Rc<Cell<usize>>);
    impl PassInstrumentation for Count {
        fn run_before_pass(&mut self, pass: &dyn OperationPass, _op: &OperationRef) {
            if !pass.argument().is_empty() {
                self.0.set(self.0.get() + 1);
            }
        }
    }

    // The function returns nothing where it says it returns an `i32`.
    let invalid = "builtin.function public extern(\"C\") @f() -> i32 {\n    builtin.ret;\n};";
    for verify_each in [true, false] {
        let context = new_context();
        let config = ParserConfig::new(context.clone()).verify_after_parse(false);
        let function = parse_any(config, "invalid.runic", invalid).into_result().unwrap();
        let pipeline = PassPipeline::parse_elements(&function.borrow().name(), "dce").unwrap();
        let mut pm = pipeline.load(context.clone(), Nesting::Implicit).unwrap();
        pm.enable_verifier(verify_each);
        let runs = Rc::new(Cell::new(0));
        pm.add_instrumentation(Box::new(Count(runs.clone())));
        let result = pm.run(function);
        if verify_each {
            let message = result.unwrap_err().to_string();
            assert_eq!(message, "invalid operation builtin.function");
            assert_eq!(runs.get(), 0, "the input is rejected before any pass runs");
        } else {
            result.unwrap();
            assert_eq!(runs.get(), 1);
        }
        drop(pm);
    }
}

/// A pass that leaves invalid IR: it removes the operands of every `builtin.ret`.
#[derive(Default)]
struct Break;

impl Pass for Break {
    type Target = Function;

    fn name(&self) -> &'static str {
        "break-returns"
    }

    fn argument(&self) -> &'static str {
        "break-returns"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Function>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let function = op.as_operation_ref();
        drop(op);
        let mut rets = Vec::new();
        function.borrow().prewalk_all(|op: &Operation| {
            if op.name().to_string() == "builtin.ret" {
                rets.push(op.as_operation_ref());
            }
        });
        for mut ret in rets {
            ret.borrow_mut().set_operands(core::iter::empty());
        }
        state.set_post_pass_status(PostPassStatus::Changed);
        Ok(())
    }
}

/// A verifier failure after a pass is one report, labelled at the operation that is invalid,
/// whose message does not name the pass and whose context frame does; nothing is emitted on the
/// context.
#[test]
fn a_verifier_failure_after_a_pass_as_data() {
    const MODULE: &str = "\
builtin.module public @m {
    builtin.function public extern(\"C\") @f(%a: i32) -> i32 {
        builtin.ret %a : (i32);
    };
};";
    let context = new_context();
    let module = parse_any(ParserConfig::new(context.clone()), "break.runic", MODULE)
        .into_result()
        .unwrap();
    let mut pm = runic_ir::pass::PassManager::on::<runic_ir::dialects::builtin::Module>(
        context.clone(),
        Nesting::Implicit,
    );
    pm.add_pass(Box::<Break>::default()).unwrap();
    let scope = context.diagnostic_scope();
    let error = pm.run(module).unwrap_err();
    let emitted = scope.finish().finish();
    let mut message = String::new();
    error.as_diagnostic().message(&mut message).unwrap();
    assert_eq!(message, "invalid operation builtin.function");
    assert!(!message.contains("break-returns"), "the message does not name the pass");
    let frames = error.contexts().iter().map(|frame| frame.message()).collect::<Vec<_>>();
    assert_eq!(
        frames,
        ["the IR that pass 'break-returns' left on 'builtin.function' @f is invalid"]
    );
    // Labelled at the `builtin.ret` the pass broke.
    assert_eq!(primary_line(error.as_diagnostic(), &context), Some(3));
    assert_eq!(emitted.iter().count(), 0);
    drop(pm);
}

/// How much of the source a rendered diagnostic shows for an error in the middle of a long file:
/// the default configuration renders the whole file, the short one a single line.
#[test]
fn how_much_source_a_rendered_diagnostic_shows() {
    let mut text = String::from("builtin.module public @m {\n");
    for i in 0..300 {
        text += &format!(
            "    // CHECK: something about function {i}\n    builtin.function public \
             extern(\"C\") @f{i}() {{\n        builtin.ret;\n    }};\n"
        );
        if i == 150 {
            text += "    builtin.function public extern(\"C\") @bad() -> i32 {\n        \
                     builtin.ret;\n    };\n";
        }
    }
    text += "};\n";
    let context = new_context();
    let outcome = parse_any(ParserConfig::new(context.clone()), "long.runic", text.as_str());
    assert!(outcome.is_err());
    let entries = outcome.diagnostics.iter().collect::<Vec<_>>();
    assert_eq!(entries.len(), 1);
    let diagnostic = &entries[0].diagnostic;
    let default = format!("{}", diagnostic.display());
    assert_eq!(text.lines().count(), 1205);
    // The whole file, every `// CHECK:` comment of it among the lines.
    assert_eq!(default.lines().count(), 1211, "{default}");
    assert_eq!(default.lines().filter(|line| line.contains("CHECK")).count(), 300);
    let short = format!(
        "{}",
        diagnostic.display().with_config(RenderConfig {
            short: true,
            ..RenderConfig::DEFAULT
        })
    );
    assert_eq!(
        short.trim_end(),
        "long.runic:607:9: error: invalid operation builtin.function: this returns (), but the \
         function returns (i32)"
    );
}
