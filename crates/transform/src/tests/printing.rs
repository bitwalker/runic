//! IR printed between the passes of a pipeline loaded from text, as `runic-opt` prints it.

use alloc::{boxed::Box, rc::Rc, string::String, vec, vec::Vec};
use core::cell::RefCell;

use runic_core::{
    Context,
    parse::{ParserConfig, parse_any},
    pass::{IRPrintingConfig, Nesting, PassPipeline, PassSelection},
};

/// A stream whose text the test reads after the pass manager has written to it.
#[derive(Clone, Default)]
struct Buffer(Rc<RefCell<String>>);

impl core::fmt::Write for Buffer {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        self.0.borrow_mut().push_str(text);
        Ok(())
    }
}

const MODULE: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32, %b: i32) -> i32 {
        %x = arith.add %a, %b;
        %y = arith.add %a, %b;
        %z = arith.add %x, %y;
        builtin.ret %z : (i32);
    };
};"#;

/// Runs `cse,dce` on the module with `config` and returns what was written.
fn run(config: IRPrintingConfig) -> String {
    let context = Rc::new(Context::default());
    let module = parse_any(ParserConfig::new(context.clone()), "printing", MODULE)
        .into_result()
        .expect("a valid module");
    let pipeline = PassPipeline::parse("builtin.module(cse,dce)").unwrap();
    let mut pm = pipeline.load(context.clone(), Nesting::Implicit).unwrap();
    let buffer = Buffer::default();
    pm.enable_ir_printing(config, Box::new(buffer.clone()));
    pm.run(module).unwrap();
    buffer.0.borrow().clone()
}

fn headers(text: &str) -> Vec<&str> {
    text.lines().filter(|line| line.starts_with("// -----")).collect()
}

#[test]
fn ir_is_printed_after_cse_and_not_after_dce() {
    let text = run(IRPrintingConfig {
        after: PassSelection::Named(vec!["cse".into()]),
        ..Default::default()
    });
    assert_eq!(headers(&text), ["// ----- IR after cse on builtin.module -----"], "{text}");
    // After `cse`, `%y` is gone: one `arith.add %a, %b` remains.
    assert_eq!(text.matches("arith.add").count(), 2, "{text}");
    assert!(text.ends_with("};\n\n"), "{text:?}");
}

#[test]
fn ir_is_printed_before_each_pass() {
    let text = run(IRPrintingConfig {
        before: PassSelection::All,
        ..Default::default()
    });
    assert_eq!(
        headers(&text),
        [
            "// ----- IR before cse on builtin.module -----",
            "// ----- IR before dce on builtin.module -----",
        ],
        "{text}"
    );
}

/// The registered passes of the workspace, by command-line name.
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

/// A module with something for each registered pass to change: a common subexpression and a dead
/// operation (`@f`, `@g`), a conditional branch (`@g`), a local stored and loaded in one block
/// (`@l`), constants to fold (`@h`), a definition used in one successor only (`@k`) and one used in
/// one region of an `scf.if` only (`@s`).
const CHANGED: &str = r#"builtin.module public @m {
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
    builtin.function public extern("C") @h(%a: i32) -> i32 {
        %one = arith.constant 1 : i32;
        %two = arith.add %one, %one;
        %r = arith.add %a, %two;
        builtin.ret %r : (i32);
    };
    builtin.function public extern("C") @k(%c: i1, %a: i32) -> i32 {
        %s = arith.add %a, %a;
        cf.cond_br %c ^t, ^e : (i1);
    ^t:
        builtin.ret %s : (i32);
    ^e:
        builtin.ret %a : (i32);
    };
    builtin.function public extern("C") @l(%a: i32) -> i32 locals(i32) {
        builtin.store_local 0, %a;
        %v = builtin.load_local 0 : i32;
        builtin.ret %v : (i32);
    };
    builtin.function public extern("C") @s(%c: i1, %a: i32, %b: i32) -> i32 {
        %t = arith.add %a, %b;
        %x = scf.if %c then {
            scf.yield %t : (i32);
        } else {
            scf.yield %a : (i32);
        } : (i1) -> (i32);
        builtin.ret %x : (i32);
    };
};"#;

/// A module no registered pass changes.
const UNCHANGED: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        builtin.ret %a : (i32);
    };
};"#;

/// Runs `pass` on each function of `module` with IR printed after it only when it reports a
/// change, and returns whether the module changed and whether anything was printed.
fn changed_and_printed(pass: &str, module: &str) -> (bool, bool) {
    use alloc::{format, string::ToString};

    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    let module = parse_any(ParserConfig::new(context.clone()), "changed", module)
        .into_result()
        .expect("a valid module");
    let before = module.borrow().to_string();
    let mut pm = PassPipeline::parse(&format!("builtin.module(builtin.function({pass}))"))
        .unwrap()
        .load(context.clone(), Nesting::Implicit)
        .unwrap();
    let buffer = Buffer::default();
    pm.enable_ir_printing(
        IRPrintingConfig {
            after: PassSelection::All,
            only_when_modified: true,
            ..Default::default()
        },
        Box::new(buffer.clone()),
    );
    pm.run(module).unwrap();
    let changed = module.borrow().to_string() != before;
    let printed = !buffer.0.borrow().is_empty();
    drop(pm);
    (changed, printed)
}

/// Two `scf.if`s in one function: the first has a definition to sink, the second none.
const TWO_IFS_SINK_FIRST: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @s(%c: i1, %a: i32, %b: i32) -> i32 {
        %t = arith.add %a, %b;
        %x = scf.if %c then {
            scf.yield %t : (i32);
        } else {
            scf.yield %a : (i32);
        } : (i1) -> (i32);
        %y = scf.if %c then {
            scf.yield %x : (i32);
        } else {
            scf.yield %a : (i32);
        } : (i1) -> (i32);
        builtin.ret %y : (i32);
    };
};"#;

/// Two `scf.if`s in one function: the last has a definition to sink. After the sinking, the
/// `scf.if` that moved into it is visited and sinks nothing.
const TWO_IFS_SINK_LAST: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @s(%c: i1, %a: i32, %b: i32) -> i32 {
        %x = scf.if %c then {
            scf.yield %a : (i32);
        } else {
            scf.yield %b : (i32);
        } : (i1) -> (i32);
        %t = arith.add %a, %x;
        %y = scf.if %c then {
            scf.yield %t : (i32);
        } else {
            scf.yield %a : (i32);
        } : (i1) -> (i32);
        builtin.ret %y : (i32);
    };
};"#;

/// Each registered pass reports a change on a module it changes, and none on a module it leaves
/// alone, so printing only when the IR changed prints exactly after the passes that changed it.
/// `control-flow-sink` reports a change when any of the operations it visits sank something, not
/// only the last.
#[test]
fn every_pass_that_changes_the_ir_reports_it() {
    for pass in PASSES {
        assert_eq!(
            changed_and_printed(pass, CHANGED),
            (true, true),
            "{pass} on a module it changes"
        );
        assert_eq!(
            changed_and_printed(pass, UNCHANGED),
            (false, false),
            "{pass} on a module it leaves alone"
        );
    }
    assert_eq!(
        changed_and_printed("control-flow-sink", TWO_IFS_SINK_FIRST),
        (true, true),
        "control-flow-sink, sinking under the first of two scf.if"
    );
    assert_eq!(
        changed_and_printed("control-flow-sink", TWO_IFS_SINK_LAST),
        (true, true),
        "control-flow-sink, sinking under the last of two scf.if"
    );
}
