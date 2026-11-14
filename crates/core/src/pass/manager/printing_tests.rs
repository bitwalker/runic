//! IR printed between passes, and statistics, written to a stream the caller gives.

use alloc::{boxed::Box, format, rc::Rc, string::String, vec, vec::Vec};
use core::cell::RefCell;

use crate::{
    Context, EntityMut, Forward, Operation, OperationName, OperationRef, RawWalk, Report,
    WalkResult,
    dialects::builtin::{Function, Module},
    interner::Symbol,
    parse::{ParserConfig, parse_any},
    pass::{
        IRPrintingConfig, Nesting, OpFilter, OperationPass, Pass, PassExecutionState, PassManager,
        PassOutput, PassSelection, PassStatistic, PostPassStatus, Statistic,
    },
};

/// A stream whose text a test reads after the pass manager has written to it.
#[derive(Clone, Default)]
struct Buffer(Rc<RefCell<String>>);

impl Buffer {
    fn output(&self) -> Box<PassOutput> {
        Box::new(self.clone())
    }

    fn text(&self) -> String {
        self.0.borrow().clone()
    }
}

#[cfg(feature = "std")]
impl std::io::Write for Buffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().push_str(core::str::from_utf8(bytes).expect("utf-8 output"));
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(not(feature = "std"))]
impl core::fmt::Write for Buffer {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        self.0.borrow_mut().push_str(text);
        Ok(())
    }
}

/// A module of two functions, `@a` and `@b`.
const MODULE: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @a() { builtin.ret; };
    builtin.function public extern("C") @b() { builtin.ret; };
};"#;

fn module() -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    let module = parse_any(ParserConfig::new(context.clone()), "module.runic", MODULE)
        .expect("valid module");
    (context, module)
}

/// A pass on `builtin.function` with the command-line name `visit`.
struct Visit;

impl Pass for Visit {
    type Target = Function;

    fn name(&self) -> &'static str {
        "visit functions"
    }

    fn argument(&self) -> &'static str {
        "visit"
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Function>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}

/// A pass on any operation that reports whether it changed the IR (it changes nothing).
struct Reporting {
    argument: &'static str,
    changed: bool,
}

impl Pass for Reporting {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "reporting pass"
    }

    fn argument(&self) -> &'static str {
        self.argument
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        state.set_post_pass_status(PostPassStatus::from(self.changed));
        Ok(())
    }
}

fn reporting(argument: &'static str, changed: bool) -> Box<dyn OperationPass> {
    Box::new(Reporting { argument, changed })
}

/// Runs the passes `(argument, changed)` on the module of `@a` and `@b` with `config`, and
/// returns what was written and the module, with the context that keeps it alive.
fn run(
    config: IRPrintingConfig,
    passes: &[(&'static str, bool)],
) -> (String, (Rc<Context>, OperationRef)) {
    let (context, module) = module();
    let mut pm = PassManager::on::<Module>(context.clone(), Nesting::Explicit);
    for &(argument, changed) in passes {
        pm.add_pass(reporting(argument, changed)).unwrap();
    }
    let buffer = Buffer::default();
    pm.enable_ir_printing(config, buffer.output());
    pm.run(module).unwrap();
    (buffer.text(), (context, module))
}

fn print(phase: &str, pass: &str, op: &OperationRef) -> String {
    let op = op.borrow();
    format!("// ----- IR {phase} {pass} on {} -----\n{op}\n\n", op.name())
}

const MODULE_TEXT: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @a() {
        builtin.ret;
    };
    builtin.function public extern("C") @b() {
        builtin.ret;
    };
};"#;

#[test]
fn ir_is_printed_after_the_named_passes_only() {
    let config = IRPrintingConfig {
        after: PassSelection::Named(vec!["first".into()]),
        ..Default::default()
    };
    let (text, _) = run(config, &[("first", true), ("second", true)]);
    assert_eq!(
        text,
        format!("// ----- IR after first on builtin.module -----\n{MODULE_TEXT}\n\n")
    );
}

#[test]
fn ir_is_printed_before_every_pass() {
    let config = IRPrintingConfig {
        before: PassSelection::All,
        ..Default::default()
    };
    let (text, (_context, module)) = run(config, &[("first", false), ("second", false)]);
    assert_eq!(
        text,
        format!("{}{}", print("before", "first", &module), print("before", "second", &module))
    );
}

#[test]
fn only_when_modified_skips_a_pass_that_changed_nothing() {
    let config = IRPrintingConfig {
        after: PassSelection::All,
        only_when_modified: true,
        ..Default::default()
    };
    let (text, (_context, module)) =
        run(config, &[("first", false), ("second", true), ("third", false)]);
    assert_eq!(text, print("after", "second", &module));

    // Without it, every selected pass prints.
    let config = IRPrintingConfig {
        after: PassSelection::All,
        ..Default::default()
    };
    let (text, _) = run(config, &[("first", false), ("second", true), ("third", false)]);
    assert_eq!(text.matches("// ----- IR after").count(), 3, "{text}");
}

#[test]
fn filters_print_each_operation_they_select() {
    let function = OpFilter::Type {
        dialect: Symbol::intern("builtin"),
        op: Symbol::intern("function"),
    };
    let config = IRPrintingConfig {
        after: PassSelection::All,
        filters: vec![function],
        ..Default::default()
    };
    let (text, (_context, module)) = run(config, &[("first", true)]);
    let all = functions(&module);
    assert_eq!(all.len(), 2);
    assert_eq!(
        text,
        format!("{}{}", print("after", "first", &all[0]), print("after", "first", &all[1]))
    );
    assert!(text.contains("@a()") && text.contains("@b()"), "{text}");
    assert!(!text.contains("builtin.module"), "{text}");

    // A symbol filter by part of a name selects `@b` only.
    let config = IRPrintingConfig {
        after: PassSelection::All,
        filters: vec![OpFilter::Symbol(Some("b".into()))],
        ..Default::default()
    };
    let (text, (_context, module)) = run(config, &[("first", true)]);
    assert_eq!(text, print("after", "first", &functions(&module)[1]));

    // An operation that is printed is not printed again for an operation under it.
    let config = IRPrintingConfig {
        after: PassSelection::All,
        filters: vec![OpFilter::Symbol(None)],
        ..Default::default()
    };
    let (text, (_context, module)) = run(config, &[("first", true)]);
    assert_eq!(text, print("after", "first", &module));
}

/// The functions under `module`, in order.
fn functions(module: &OperationRef) -> Vec<OperationRef> {
    let mut functions = Vec::new();
    let _ = module.raw_prewalk::<Forward, _, ()>(|op: OperationRef| {
        if op.borrow().name().is::<Function>() {
            functions.push(op);
        }
        WalkResult::Continue(())
    });
    functions
}

#[test]
fn nothing_is_written_when_nothing_is_selected() {
    let (text, _) = run(IRPrintingConfig::default(), &[("first", true), ("second", true)]);
    assert_eq!(text, "");

    let config = IRPrintingConfig {
        after: PassSelection::Named(vec!["elsewhere".into()]),
        ..Default::default()
    };
    let (text, _) = run(config, &[("first", true)]);
    assert_eq!(text, "");
}

#[test]
fn a_nested_pass_prints_the_operation_it_ran_on() {
    let (context, module) = module();
    let mut pm = PassManager::on::<Module>(context, Nesting::Implicit);
    // A pass on `builtin.function`, nested implicitly; the adaptor that runs it prints nothing.
    pm.add_pass(Box::new(Visit)).unwrap();
    let buffer = Buffer::default();
    let config = IRPrintingConfig {
        before: PassSelection::All,
        after: PassSelection::All,
        ..Default::default()
    };
    pm.enable_ir_printing(config, buffer.output());
    pm.run(module).unwrap();
    let text = buffer.text();
    let headers = text.lines().filter(|line| line.starts_with("// -----")).collect::<Vec<_>>();
    assert_eq!(
        headers,
        [
            "// ----- IR before visit on builtin.function -----",
            "// ----- IR after visit on builtin.function -----",
            "// ----- IR before visit on builtin.function -----",
            "// ----- IR after visit on builtin.function -----",
        ]
    );
}

/// A pass that counts its runs in a statistic.
#[derive(Default)]
struct Count {
    runs: u64,
    statistics: Vec<Box<dyn Statistic>>,
}

impl Pass for Count {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "count"
    }

    fn argument(&self) -> &'static str {
        "count"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn statistics(&self) -> &[Box<dyn Statistic>] {
        &self.statistics
    }

    fn statistics_mut(&mut self) -> &mut [Box<dyn Statistic>] {
        &mut self.statistics
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        self.runs += 1;
        self.statistics = vec![
            Box::new(PassStatistic::new("runs".into(), "times the pass ran".into(), self.runs)),
            Box::new(PassStatistic::new("ones".into(), "".into(), 1u64)),
        ];
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}

#[test]
fn statistics_are_written_to_the_given_stream_as_a_list() {
    let (context, module) = module();
    let mut pm = PassManager::on::<Module>(context, Nesting::Explicit);
    // Two passes of one name: their statistics are merged.
    pm.add_pass(Box::<Count>::default()).unwrap();
    pm.add_pass(Box::<Count>::default()).unwrap();
    pm.add_pass(reporting("first", false)).unwrap();
    let buffer = Buffer::default();
    pm.enable_statistics(buffer.output());
    pm.run(module).unwrap();
    let rule = format!("=={}==", "-".repeat(76));
    assert_eq!(
        buffer.text(),
        format!(
            "{rule}\n{}... Pass statistics report ...\n{rule}\ncount\n  (S) 2 ones\n  (S) 2 runs \
             - times the pass ran\n",
            " ".repeat(25)
        )
    );
}
