//! IR printed between passes, and statistics, written to a stream the caller gives.

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
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

/// A pass without a command-line name is selected by `All` and named by its display name.
#[test]
fn a_pass_without_a_command_line_name_is_named_by_its_display_name() {
    let config = IRPrintingConfig {
        after: PassSelection::All,
        ..Default::default()
    };
    let (text, _) = run(config, &[("", true)]);
    assert_eq!(
        text,
        format!("// ----- IR after reporting pass on builtin.module -----\n{MODULE_TEXT}\n\n")
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

/// A stream that accepts `accepted` writes, fails the next, and counts the writes it is handed.
#[derive(Clone)]
struct Failing {
    accepted: usize,
    writes: Rc<RefCell<usize>>,
}

impl core::fmt::Write for Failing {
    fn write_str(&mut self, _text: &str) -> core::fmt::Result {
        let mut writes = self.writes.borrow_mut();
        *writes += 1;
        if *writes > self.accepted {
            Err(core::fmt::Error)
        } else {
            Ok(())
        }
    }
}

#[test]
fn after_a_failed_write_of_printed_ir_nothing_more_is_written_and_the_run_fails() {
    let (context, module) = module();
    let mut pm = PassManager::on::<Module>(context, Nesting::Explicit);
    for argument in ["first", "second", "third"] {
        pm.add_pass(reporting(argument, true)).unwrap();
    }
    let writes = Rc::new(RefCell::new(0));
    let out = Failing {
        accepted: 1,
        writes: writes.clone(),
    };
    let config = IRPrintingConfig {
        after: PassSelection::All,
        ..Default::default()
    };
    pm.enable_ir_printing(config, Box::new(out));
    let error = pm.run(module).expect_err("the second write fails");
    assert!(error.to_string().contains("could not write pass output"), "{error}");
    // The first print is written, the second fails, the third is not attempted.
    assert_eq!(*writes.borrow(), 2);
    // The stream stays failed: a second run writes nothing, and its passes succeed.
    pm.run(module).expect("the failure was returned by the run that met it");
    assert_eq!(*writes.borrow(), 2);
}

#[test]
fn a_failed_write_of_statistics_fails_the_run() {
    let (context, module) = module();
    let mut pm = PassManager::on::<Module>(context, Nesting::Explicit);
    pm.add_pass(Box::<Count>::default()).unwrap();
    let writes = Rc::new(RefCell::new(0));
    pm.enable_statistics(Box::new(Failing {
        accepted: 0,
        writes: writes.clone(),
    }));
    let error = pm.run(module).expect_err("the statistics are not written");
    assert!(error.to_string().contains("could not write pass output"), "{error}");
    assert_eq!(*writes.borrow(), 1);
    pm.run(module).expect("nothing more is written to a failed stream");
    assert_eq!(*writes.borrow(), 1);
}

#[cfg(feature = "std")]
mod io_output {
    use alloc::{boxed::Box, rc::Rc, string::ToString, vec::Vec};
    use core::{cell::RefCell, fmt::Write};

    use crate::pass::IoOutput;

    /// An `io::Write` that records what it is given and whether it was flushed, and fails every
    /// write after `accepted` of them.
    struct Device {
        accepted: usize,
        log: Rc<RefCell<Vec<&'static str>>>,
        bytes: Vec<u8>,
    }

    impl std::io::Write for Device {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.accepted == 0 {
                self.log.borrow_mut().push("fail");
                return Err(std::io::Error::other("the device is full"));
            }
            self.accepted -= 1;
            self.log.borrow_mut().push("write");
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.log.borrow_mut().push("flush");
            Ok(())
        }
    }

    #[test]
    fn an_io_output_flushes_each_write_and_keeps_the_first_error() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let (mut out, error) = IoOutput::new(Device {
            accepted: 1,
            log: log.clone(),
            bytes: Vec::new(),
        });
        out.write_str("one").unwrap();
        assert!(out.write_str("two").is_err());
        assert!(out.write_str("three").is_err());
        assert_eq!(
            error.error().map(|error| error.to_string()).as_deref(),
            Some("the device is full")
        );
        // Taking the error does not let the output write again.
        assert!(error.take().is_some());
        assert!(error.error().is_none());
        assert!(out.write_str("four").is_err());
        // The third and fourth writes never reach the device.
        assert_eq!(*log.borrow(), ["write", "flush", "fail"]);
        assert_eq!(out.into_inner().bytes, b"one");
    }

    /// A caller that gave the boxed output to the pass manager reads why its write failed from
    /// the handle, and each stream fails on its own.
    #[test]
    fn the_reason_a_boxed_io_output_failed_is_read_from_its_handle() {
        use crate::{
            dialects::builtin::Module,
            pass::{IRPrintingConfig, Nesting, PassManager, PassSelection},
        };

        let (context, module) = super::module();
        let mut pm = PassManager::on::<Module>(context, Nesting::Explicit);
        pm.add_pass(super::reporting("first", true)).unwrap();
        pm.add_pass(Box::<super::Count>::default()).unwrap();
        let log = Rc::new(RefCell::new(Vec::new()));
        let (ir, ir_error) = IoOutput::boxed(Device {
            accepted: 0,
            log: log.clone(),
            bytes: Vec::new(),
        });
        let config = IRPrintingConfig {
            after: PassSelection::All,
            ..Default::default()
        };
        pm.enable_ir_printing(config, ir);
        let statistics = Rc::new(RefCell::new(Vec::new()));
        let (stats, stats_error) = IoOutput::boxed(Device {
            accepted: usize::MAX,
            log: statistics.clone(),
            bytes: Vec::new(),
        });
        pm.enable_statistics(stats);
        let report = pm.run(module).expect_err("the printed IR is not written");
        assert!(report.to_string().contains("could not write pass output"), "{report}");
        assert_eq!(
            ir_error.error().map(|error| error.to_string()).as_deref(),
            Some("the device is full")
        );
        // The statistics are written to their own stream after the IR stream failed.
        assert!(stats_error.error().is_none());
        assert_eq!(*statistics.borrow(), ["write", "flush"]);
        // A later run writes nothing to the failed stream and does not report it again.
        pm.run(module).expect("the failure was returned by the run that met it");
        assert_eq!(*log.borrow(), ["fail"]);
    }
}

/// A pass on any operation, with the command-line name `fail`, that fails.
struct Fail;

impl Pass for Fail {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "failing pass"
    }

    fn argument(&self) -> &'static str {
        "fail"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Operation>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        Err(Report::msg("this pass always fails"))
    }
}

/// Runs `reporting("first", true)` then `fail` on the module, with `config`, and returns what
/// was written, the error, and the module, with the context that keeps it alive.
fn run_failing(config: IRPrintingConfig) -> (String, Report, (Rc<Context>, OperationRef)) {
    let (context, module) = module();
    let mut pm = PassManager::on::<Module>(context.clone(), Nesting::Explicit);
    pm.add_pass(reporting("first", true)).unwrap();
    pm.add_pass(Box::new(Fail)).unwrap();
    let buffer = Buffer::default();
    pm.enable_ir_printing(config, buffer.output());
    let error = pm.run(module).unwrap_err();
    drop(pm);
    (buffer.text(), error, (context, module))
}

#[test]
fn ir_is_printed_after_a_selected_pass_that_failed() {
    let config = IRPrintingConfig {
        after_failure: PassSelection::All,
        ..Default::default()
    };
    let (text, _, (_context, module)) = run_failing(config);
    assert_eq!(text, print("after failed", "fail", &module));
}

#[test]
fn ir_is_not_printed_after_a_failure_unless_the_failed_pass_is_selected() {
    let config = IRPrintingConfig {
        after_failure: PassSelection::Named(vec!["first".into()]),
        after: PassSelection::All,
        ..Default::default()
    };
    let (text, _, (_context, module)) = run_failing(config);
    // `first` succeeded and is printed after; `fail` is neither printed after (it failed) nor
    // after its failure (it is not selected).
    assert_eq!(text, print("after", "first", &module));
}

#[test]
fn a_failed_pass_is_named_with_the_operation_it_ran_on() {
    let (_, error, _module) = run_failing(IRPrintingConfig::default());
    let contexts = error.contexts().iter().map(|frame| frame.message()).collect::<Vec<_>>();
    assert_eq!(contexts, ["pass 'fail' failed on 'builtin.module' @m"]);
}
