//! The sparse forward analyses (`sccp`, the integer range analysis) over a loop with one region
//! whose back edge carries operands that are themselves arguments of the loop's entry block: the
//! value an argument holds on a later trip round the loop is computed from the values the other
//! arguments held on the trip before, so the analyses must propagate a change of one argument to
//! every join that reads it.
//!
//! Each program runs for `test.loop` (`test.continue` is the back edge) and for `wasm.loop`
//! (`wasm.br 0`). Every argument is tested in a condition that guards an early exit, so that a
//! wrong "this argument is always zero" shows as a folded condition, and the integer ranges of the
//! arguments are checked directly. The values come from `arith.constant`, which folds, so that the
//! analyses' answers are visible (the `wasm` dialect folds nothing but its own constants).

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use runic_analysis::{
    DataFlowConfig, DataFlowSolver,
    analyses::{IntegerRangeAnalysis, integer_range::value_range},
};
use runic_ir::{
    BlockArgument, Context, Operation, OperationRef, Report,
    dialects::test::TestDialect,
    parse::{ParserConfig, parse_any},
    pass::{AnalysisManager, Nesting, PassPipeline},
};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Test,
    Wasm,
}

/// A loop whose entry block takes `inits.len()` arguments `%a`, `%b`, `%c`, ..., started from the
/// constants `inits` (`zero` or `five`) and sent back with `next` (the names of arguments, or of
/// `zero`, `five`). Each argument named in `checked` guards an early exit.
struct Loop {
    inits: &'static [&'static str],
    next: &'static [&'static str],
    checked: &'static [&'static str],
}

impl Loop {
    fn text(&self, kind: Kind) -> String {
        let names = ["a", "b", "c", "d"];
        let arity = self.inits.len();
        let params =
            (0..arity).map(|i| format!("%{}: i32", names[i])).collect::<Vec<_>>().join(", ");
        let types = vec!["i32"; arity].join(", ");
        let inits = self.inits.iter().map(|i| format!("%{i}")).collect::<Vec<_>>().join(", ");
        let next = self.next.iter().map(|i| format!("%{i}")).collect::<Vec<_>>().join(", ");
        let mut body = String::new();
        for x in self.checked {
            match kind {
                Kind::Wasm => body.push_str(&format!(
                    "        wasm.if %{x} then {{ builtin.ret %{x} : (i32); }} : (i32) -> ();\n"
                )),
                Kind::Test => body.push_str(&format!(
                    "        %t{x} = arith.cmpi ne %{x}, %zero;\n        test.guard %t{x} {{ \
                     test.continue {next} : ({types}); }} : (i1);\n"
                )),
            }
        }
        let mut text = String::from(
            "builtin.function public extern(\"C\") @f() -> i32 {\n    %zero = arith.constant 0 : \
             i32;\n    %five = arith.constant 5 : i32;\n",
        );
        match kind {
            Kind::Wasm => text.push_str(&format!(
                "    wasm.loop {inits} {{\n    ^b({params}):\n{body}        wasm.br 0 {next} : \
                 ({types});\n    }} : ({types}) -> ();\n    wasm.unreachable;\n}};\n"
            )),
            Kind::Test => {
                let results = (0..arity).map(|i| format!("%r{i}")).collect::<Vec<_>>().join(", ");
                text.push_str(&format!(
                    "    {results} = test.loop {inits} {{\n    ^b({params}):\n{body}        \
                     test.continue {next} : ({types});\n    }} : ({types}) -> ({types});\n    \
                     builtin.ret %zero : (i32);\n}};\n"
                ));
            }
        }
        text
    }

    fn parse(&self, kind: Kind) -> (Rc<Context>, OperationRef) {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<runic_dialect_arith::ArithDialect>();
        context.get_or_register_dialect::<runic_dialect_wasm::WasmDialect>();
        context.get_or_register_dialect::<TestDialect>();
        let text = format!("builtin.module public @m {{\n{}\n}};", self.text(kind));
        let module = parse_any(ParserConfig::new(context.clone()), "loop_carried", text)
            .into_result()
            .unwrap_or_else(|e| panic!("{e:?}"));
        (context, module)
    }
}

fn first(module: OperationRef, name: &str) -> OperationRef {
    let mut found = None;
    module.borrow().prewalk_all(|op: &Operation| {
        if found.is_none() && op.name().to_string() == name {
            found = Some(op.as_operation_ref());
        }
    });
    found.unwrap_or_else(|| panic!("no {name}"))
}

fn run(context: &Rc<Context>, module: OperationRef, pipeline: &str) -> Result<(), Report> {
    let pipeline = PassPipeline::parse_elements(&module.borrow().name(), pipeline)?;
    let mut passes = pipeline.load(context.clone(), Nesting::Implicit)?;
    passes.enable_verifier(true);
    passes.run(module)?;
    Ok(())
}

/// Every condition of an early exit is still computed from the loop's arguments (a block argument
/// itself for `wasm.if`, a comparison for `test.guard`), and not folded to a constant.
fn assert_conditions_are_not_folded(kind: Kind, module: OperationRef, count: usize, what: &str) {
    let name = match kind {
        Kind::Wasm => "wasm.if",
        Kind::Test => "test.guard",
    };
    let mut seen = 0;
    module.borrow().prewalk_all(|op: &Operation| {
        if op.name().to_string() != name {
            return;
        }
        seen += 1;
        let condition = op.operands()[0].borrow().as_value_ref();
        let folded = match kind {
            Kind::Wasm => !condition.borrow().is::<BlockArgument>(),
            Kind::Test => condition
                .borrow()
                .get_defining_op()
                .is_none_or(|def| def.borrow().name().to_string() != "arith.cmpi"),
        };
        assert!(!folded, "{what}: a condition of {name} was folded:\n{}", module.borrow());
    });
    assert_eq!(seen, count, "{what}: the number of `{name}` left");
}

/// The unsigned range of each argument of the loop's entry block after the integer range analysis.
fn argument_ranges(module: OperationRef, kind: Kind) -> Vec<(u128, u128)> {
    let function = first(module, "builtin.function");
    let mut solver = DataFlowSolver::new(DataFlowConfig::default());
    solver.load::<IntegerRangeAnalysis>();
    solver
        .initialize_and_run(&function.borrow(), AnalysisManager::new(function, None))
        .expect("the analyses converge");
    let looped = first(
        module,
        match kind {
            Kind::Wasm => "wasm.loop",
            Kind::Test => "test.loop",
        },
    );
    let entry = looped.borrow().region(0).entry_block_ref().unwrap();
    entry
        .borrow()
        .arguments()
        .iter()
        .map(|argument| {
            let range = value_range(&solver, argument.borrow().as_value_ref())
                .expect("the argument has a range");
            let range = *range.range().expect("a tracked range");
            (range.umin(), range.umax())
        })
        .collect()
}

/// `loop(a=0, b=0) { if b then ret b; br 0 (5, a) }`: the pair goes (0,0), (5,0), (5,5).
const SHIFT: Loop = Loop {
    inits: &["zero", "zero"],
    next: &["five", "a"],
    checked: &["b"],
};

/// `loop(a=0, b=0, c=5) { ..; br 0 (c, a, b) }`: three values rotate; each is 0 or 5 at some trip.
const ROTATE: Loop = Loop {
    inits: &["zero", "zero", "five"],
    next: &["c", "a", "b"],
    checked: &["a", "b", "c"],
};

/// `loop(a=0, b=0) { ..; br 0 (b, 5) }`: `a` copies `b`, which is 0 on the first trip and 5 on the
/// second, so `a` changes only on the second trip round the loop.
const SECOND_TRIP: Loop = Loop {
    inits: &["zero", "zero"],
    next: &["b", "five"],
    checked: &["a"],
};

/// A program, and the unsigned range of each argument of its loop.
type Case = (&'static str, Loop, &'static [(u128, u128)]);

const CASES: [Case; 3] = [
    ("shift", SHIFT, &[(0, 5), (0, 5)]),
    ("rotate", ROTATE, &[(0, 5), (0, 5), (0, 5)]),
    ("second_trip", SECOND_TRIP, &[(0, 5), (0, 5)]),
];

#[test]
fn sccp_does_not_fold_a_loop_carried_value_that_changes_on_a_later_trip() -> Result<(), Report> {
    for kind in [Kind::Test, Kind::Wasm] {
        for (name, case, _) in CASES {
            let (context, module) = case.parse(kind);
            run(&context, module, "sccp")?;
            assert_conditions_are_not_folded(
                kind,
                module,
                case.checked.len(),
                &format!("{kind:?} {name}"),
            );
        }
    }
    Ok(())
}

#[test]
fn the_standard_pipeline_does_not_fold_a_loop_carried_value_that_changes_on_a_later_trip()
-> Result<(), Report> {
    for kind in [Kind::Test, Kind::Wasm] {
        for (name, case, _) in CASES {
            let (context, module) = case.parse(kind);
            run(&context, module, "canonicalizer,cse,sccp,canonicalizer,dce")?;
            assert_conditions_are_not_folded(
                kind,
                module,
                case.checked.len(),
                &format!("{kind:?} {name}"),
            );
        }
    }
    Ok(())
}

#[test]
fn the_integer_range_of_a_loop_carried_value_covers_every_trip() {
    for kind in [Kind::Test, Kind::Wasm] {
        for (name, case, expected) in CASES {
            let (_context, module) = case.parse(kind);
            assert_eq!(argument_ranges(module, kind), expected, "{kind:?} {name}");
        }
    }
}

/// A control: when nothing changes the arguments (`br 0 (a, b)`), they stay 0 and the conditions
/// do fold, so the tests above fail because of the back edge and not because nothing is learned.
#[test]
fn a_loop_that_sends_its_arguments_back_unchanged_keeps_them_constant() -> Result<(), Report> {
    let case = Loop {
        inits: &["zero", "zero"],
        next: &["a", "b"],
        checked: &["b"],
    };
    for kind in [Kind::Test, Kind::Wasm] {
        let (_context, module) = case.parse(kind);
        assert_eq!(argument_ranges(module, kind), [(0, 0), (0, 0)], "{kind:?}");
    }
    Ok(())
}
