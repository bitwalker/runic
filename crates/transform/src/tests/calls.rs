//! Calls through the passes of this crate (and `LiftControlFlowToSCF`): operations with unknown
//! effects in every position, calls in each effect state, the canonicalization of an indirect
//! call of a function reference, symbol references that do not resolve, the symbol uses of erased
//! operations, and calls of a function that never returns.

extern crate std;

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    BlockRef, Context, Operation, OperationRef, Report, Symbol, SymbolTable,
    dialects::builtin::{Function, Module},
    effects::MemoryEffect,
    parse::{self, ParserConfig},
    pass::{Nesting, OperationPass, PassManager},
    testing::{
        calls::{CORPUS, mutations},
        text::describe,
    },
    traits::Terminator,
};
use runic_dialect_scf::transforms::LiftControlFlowToSCF;

use crate::{
    Canonicalizer, CommonSubexpressionElimination, ControlFlowSink, DeadCodeElimination,
    SinkOperandDefs, SparseConditionalConstantPropagation,
};

/// The seed of a randomized test: `CALLS_TEST_SEED` if set, else `default`.
fn seed(default: u64) -> u64 {
    match std::env::var("CALLS_TEST_SEED") {
        Ok(seed) => seed.parse().expect("CALLS_TEST_SEED is a u64"),
        Err(_) => default,
    }
}

const CALLEES: &str = r#"
    builtin.function public extern("C") @unknown(i32) -> i32;
    builtin.function public extern("C") @pure(i32) -> i32 effects();
    builtin.function public extern("C") @reads(ptr<i32>) -> i32 effects(read(arg 0));
    builtin.function public extern("C") @reads_any(i32) -> i32 effects(read);
    builtin.function public extern("C") @writes(ptr<i32>, i32) -> i32 effects(write(arg 0));
    builtin.function public extern("C") @writes_any(i32) -> i32 effects(write);
    builtin.function public extern("C") @frees(ptr<i32>) effects(free(arg 0));
    builtin.function public extern("C") @allocates(i32) -> ptr<i32> effects(allocate(result 0));
    builtin.function_alias public @alias_one -> @reads;
    builtin.function_alias public @alias_two -> @reads;
    builtin.function_alias public @alias_pure -> @pure;
    builtin.function_alias public @alias_alias_pure -> @alias_pure;
    builtin.function_alias public @alias_unknown -> @unknown;
"#;

/// A module holding [CALLEES] and
/// `f(%c: i1, %d: i1, %a: i32, %b: i32, %p: ptr<i32>, %g: (i32) -> i32) -> i32` with `body`.
fn make(context: &Rc<Context>, body: &str) -> Result<(OperationRef, OperationRef), Report> {
    let text = format!(
        "builtin.module public @m {{\n{CALLEES}\nbuiltin.function public extern(\"C\") @f(%c: i1, \
         %d: i1, %a: i32, %b: i32, %p: ptr<i32>, %g: (i32) -> i32) -> i32 {{\n{body}\n}};\n}};"
    );
    let module =
        parse::parse::<Module>(ParserConfig::new(context.clone()), "calls", text).into_result()?;
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    Ok((module.as_operation_ref(), function))
}

type MakePass = fn() -> Box<dyn OperationPass>;

const PASSES: [(&str, MakePass); 7] = [
    ("dce", || Box::<DeadCodeElimination>::default()),
    ("cse", || Box::<CommonSubexpressionElimination>::default()),
    ("control-flow-sink", || Box::<ControlFlowSink>::default()),
    ("sink-operand-defs", || Box::<SinkOperandDefs>::default()),
    ("sccp", || Box::<SparseConditionalConstantPropagation>::default()),
    ("canonicalize", || Box::<Canonicalizer>::default()),
    ("cfg-to-scf", || Box::<LiftControlFlowToSCF>::default()),
];

/// Runs `pass` on `function`, verifying after it.
fn run(
    context: &Rc<Context>,
    function: OperationRef,
    pass: Box<dyn OperationPass>,
) -> Result<(), Report> {
    let mut passes = PassManager::on::<Function>(context.clone(), Nesting::Implicit);
    passes.add_pass(pass).unwrap();
    passes.enable_verifier(true);
    passes.run(function)
}

/// An operation without regions, not a terminator or a symbol, whose effects are unknown.
fn has_unknown_effects(op: &Operation) -> bool {
    !op.has_regions()
        && !op.implements::<dyn Terminator>()
        && !op.implements::<dyn Symbol>()
        && op.get_effects_recursively::<MemoryEffect>().is_none()
}

/// An operation that writes, frees, or has unknown effects: one no pass may remove or move.
fn is_pinned(op: &Operation) -> bool {
    if op.has_regions() || op.implements::<dyn Terminator>() || op.implements::<dyn Symbol>() {
        return false;
    }
    match op.get_effects_recursively::<MemoryEffect>() {
        None => true,
        Some(effects) => effects
            .iter()
            .any(|e| matches!(e.effect(), MemoryEffect::Write | MemoryEffect::Free)),
    }
}

/// The operations under `op` that satisfy `pick`, in pre-order, each with its block.
fn collect(
    op: &Operation,
    pick: fn(&Operation) -> bool,
) -> Vec<(OperationRef, Option<BlockRef>, String)> {
    let mut found = Vec::new();
    op.prewalk_all(|op: &Operation| {
        if pick(op) {
            found.push((op.as_operation_ref(), op.parent(), op.name().to_string()));
        }
    });
    found
}

/// The operations with unknown effects, each defining one `i32`.
const UNKNOWN: [(&str, &str); 4] = [
    ("test.effects", "test.effects <{ known = #builtin.bool<false> }>"),
    ("builtin.call", r#"builtin.call @unknown(%a) : extern("C") (i32) -> i32"#),
    (
        "call through alias",
        r#"builtin.call @alias_unknown(%a) : extern("C") (i32) -> i32"#,
    ),
    (
        "builtin.call_indirect",
        r#"builtin.call_indirect %g(%a) : extern("C") (i32) -> i32"#,
    ),
];

/// Positions of an operation with unknown effects, written `{U}`.
const POSITIONS: [(&str, &str); 14] = [
    (
        "unused and duplicated in one block",
        r#"
    %u1 = {U};
    %u2 = {U};
    builtin.ret %a : (i32);"#,
    ),
    (
        "used and duplicated in one block",
        r#"
    %u1 = {U};
    %u2 = {U};
    %s = arith.add %u1, %u2;
    builtin.ret %s : (i32);"#,
    ),
    (
        "in both regions of an if without results",
        r#"
    scf.if %c then {
        %u1 = {U};
        scf.yield : ();
    } else {
        %u2 = {U};
        scf.yield : ();
    } : (i1) -> ();
    builtin.ret %a : (i32);"#,
    ),
    (
        "in a region of an if whose result is unused",
        r#"
    %x = scf.if %c then {
        %u1 = {U};
        scf.yield %a : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %a : (i32);"#,
    ),
    (
        "in both regions of an if that yield the same value",
        r#"
    %x = scf.if %c then {
        %u1 = {U};
        scf.yield %a : (i32);
    } else {
        %u2 = {U};
        scf.yield %a : (i32);
    } : (i1) -> (i32);
    builtin.ret %x : (i32);"#,
    ),
    (
        "in the head and body of a loop whose result is unused",
        r#"
    %zero = arith.constant 0 : i32;
    %count = scf.while %zero before {
    ^head(%i: i32):
        %u1 = {U};
        %continue = arith.cmpi ult %i, %a;
        scf.condition %continue, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %u2 = {U};
        %one = arith.constant 1 : i32;
        %next = arith.add %j, %one;
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %a : (i32);"#,
    ),
    (
        "after a branch, in each successor and at the join",
        r#"
    %u0 = {U};
    cf.cond_br %c ^t, ^f : (i1);
^t:
    %u1 = {U};
    cf.br ^m(%a : i32);
^f:
    %u2 = {U};
    cf.br ^m(%b : i32);
^m(%r: i32):
    %u3 = {U};
    builtin.ret %r : (i32);"#,
    ),
    (
        "defined before an if, its only use in one region",
        r#"
    %u1 = {U};
    %x = scf.if %d then {
        %y = arith.add %u1, %b;
        scf.yield %y : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %x : (i32);"#,
    ),
    (
        "defined before a loop, its only use in the body",
        r#"
    %u1 = {U};
    %zero = arith.constant 0 : i32;
    %count = scf.while %zero before {
    ^head(%i: i32):
        %continue = arith.cmpi ult %i, %a;
        scf.condition %continue, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %next = arith.add %j, %u1;
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %count : (i32);"#,
    ),
    (
        "between two loads and two stores",
        r#"
    %x = test.load %p;
    %u1 = {U};
    %y = test.load %p;
    test.store %p, %a : (ptr<i32>, i32);
    %u2 = {U};
    test.store %p, %b : (ptr<i32>, i32);
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    ),
    (
        "between two calls that read",
        r#"
    %x = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32;
    %u1 = {U};
    %y = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    ),
    (
        "in two identical ifs",
        r#"
    %x = scf.if %c then {
        %u1 = {U};
        scf.yield %u1 : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    %y = scf.if %c then {
        %u2 = {U};
        scf.yield %u2 : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    ),
    (
        "before a branch, its only use in one successor",
        r#"
    %u1 = {U};
    cf.cond_br %c ^t, ^f : (i1);
^t:
    %y = arith.add %u1, %b;
    builtin.ret %y : (i32);
^f:
    builtin.ret %b : (i32);"#,
    ),
    (
        "between a pure definition and its only use in a region",
        r#"
    %sum = arith.add %a, %b;
    %k = builtin.call @pure(%a) : extern("C") (i32) -> i32;
    %u1 = {U};
    %x = scf.if %d then {
        %u2 = {U};
        %y = arith.add %sum, %k;
        scf.yield %y : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    %u3 = {U};
    builtin.ret %x : (i32);"#,
    ),
];

/// What `pass` did to the pinned operations of `function`, as an error text, or `None` when every
/// one of them is still there (and, for a pass that moves no blocks, where it was and in order).
fn check(
    pass: &str,
    function: OperationRef,
    before: &[(OperationRef, Option<BlockRef>, String)],
) -> Option<String> {
    let after = collect(&function.borrow(), is_pinned);
    let moves_blocks = pass == "cfg-to-scf" || pass == "canonicalize";
    if moves_blocks {
        let mut a = after.iter().map(|(_, _, name)| name.clone()).collect::<Vec<_>>();
        let mut b = before.iter().map(|(_, _, name)| name.clone()).collect::<Vec<_>>();
        a.sort();
        b.sort();
        (a != b).then(|| format!("pinned operations {b:?} became {a:?}"))
    } else {
        let same = after.len() == before.len()
            && after.iter().zip(before).all(|(a, b)| a.0 == b.0 && a.1 == b.1);
        (!same).then(|| {
            format!(
                "pinned operations changed: {:?} became {:?}",
                before.iter().map(|b| b.2.as_str()).collect::<Vec<_>>(),
                after.iter().map(|a| a.2.as_str()).collect::<Vec<_>>()
            )
        })
    }
}

/// Every pass keeps every operation with unknown effects, and every write, in every position:
/// not removed, not merged, and (except by the two passes that restructure blocks) not moved.
#[test]
fn every_pass_keeps_every_operation_with_unknown_effects_in_every_position() {
    let mut failures = Vec::new();
    let mut ran = 0;
    for (position, body) in POSITIONS {
        for (kind, unknown) in UNKNOWN {
            for (pass, make_pass) in PASSES {
                let context = Rc::new(Context::default());
                let body = body.replace("{U}", unknown);
                let (_module, function) = match make(&context, &body) {
                    Ok(made) => made,
                    Err(err) => panic!("{position} / {kind}: {}\n{body}", describe(&err)),
                };
                let before = collect(&function.borrow(), is_pinned);
                let unknown_before = collect(&function.borrow(), has_unknown_effects).len();
                assert!(unknown_before > 0, "{position} / {kind}");
                let reads_before = function.borrow().to_string().matches("@reads(").count();
                if let Err(err) = run(&context, function, make_pass()) {
                    failures.push(format!("{pass} on '{position}' / {kind}: {}", describe(&err)));
                    continue;
                }
                ran += 1;
                if let Some(problem) = check(pass, function, &before) {
                    failures.push(format!(
                        "{pass} on '{position}' / {kind}: {problem}\n{}",
                        function.borrow()
                    ));
                }
                // Reads on either side of an operation with unknown effects are not merged
                let text = function.borrow().to_string();
                if position == "between two loads and two stores"
                    && text.matches("test.load").count() != 2
                {
                    failures.push(format!("{pass} / {kind}: loads merged\n{text}"));
                }
                if text.matches("@reads(").count() != reads_before {
                    failures.push(format!("{pass} / {kind}: reading calls merged\n{text}"));
                }
            }
        }
    }
    assert_eq!(ran, POSITIONS.len() * UNKNOWN.len() * PASSES.len());
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n\n"));
}

/// `f` after `pass` ran on it with `body`.
fn after(pass: usize, body: &str) -> String {
    let context = Rc::new(Context::default());
    let (module, function) =
        make(&context, body).unwrap_or_else(|err| panic!("{}\n{body}", describe(&err)));
    run(&context, function, PASSES[pass].1())
        .unwrap_or_else(|err| panic!("{}\n{body}", describe(&err)));
    module.borrow().recursively_verify().unwrap();
    function.borrow().to_string()
}

const DCE: usize = 0;
const CSE: usize = 1;

/// Calls that differ only in an argument, or that name two aliases of one
/// function, are not merged; a call through an alias takes its target's effects.
#[test]
fn cse_of_calls_compares_arguments_and_the_path_of_the_callee() {
    // differing only in an argument
    let text = after(
        CSE,
        r#"
    %x = builtin.call @reads_any(%a) : extern("C") (i32) -> i32;
    %y = builtin.call @reads_any(%b) : extern("C") (i32) -> i32;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    );
    assert_eq!(text.matches("builtin.call").count(), 2, "{text}");
    // two aliases of one function: the same callee, but not merged
    let text = after(
        CSE,
        r#"
    %x = builtin.call @alias_one(%p) : extern("C") (ptr<i32>) -> i32;
    %y = builtin.call @alias_two(%p) : extern("C") (ptr<i32>) -> i32;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    );
    assert_eq!(text.matches("builtin.call").count(), 2, "{text}");
    // an alias and its target: not merged either
    let text = after(
        CSE,
        r#"
    %x = builtin.call @alias_one(%p) : extern("C") (ptr<i32>) -> i32;
    %y = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    );
    assert_eq!(text.matches("builtin.call").count(), 2, "{text}");
    // the same alias twice: merged, with the target's effects (a read)
    let text = after(
        CSE,
        r#"
    %x = builtin.call @alias_one(%p) : extern("C") (ptr<i32>) -> i32;
    %y = builtin.call @alias_one(%p) : extern("C") (ptr<i32>) -> i32;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    );
    assert_eq!(text.matches("builtin.call").count(), 1, "{text}");
    // the same alias twice across a store: not merged
    let text = after(
        CSE,
        r#"
    %x = builtin.call @alias_one(%p) : extern("C") (ptr<i32>) -> i32;
    test.store %p, %a : (ptr<i32>, i32);
    %y = builtin.call @alias_one(%p) : extern("C") (ptr<i32>) -> i32;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    );
    assert_eq!(text.matches("builtin.call").count(), 2, "{text}");
    // a relative and an absolute path of one function: not merged (conservative)
    let text = after(
        CSE,
        r#"
    %x = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32;
    %y = builtin.call ::@m::@reads(%p) : extern("C") (ptr<i32>) -> i32;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    );
    assert_eq!(text.matches("builtin.call").count(), 2, "{text}");
}

/// Calls that read are merged only when nothing between them writes, frees or has unknown
/// effects, in the same block; calls that allocate, write or free are never merged.
#[test]
fn cse_of_calls_respects_what_is_between_them() {
    let merged = |between: &str| {
        let text = after(
            CSE,
            &format!(
                r#"
    %x = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32;
    {between}
    %y = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#
            ),
        );
        text.matches("@reads(").count() == 1
    };
    assert!(merged(""));
    assert!(merged(r#"%k = builtin.call @pure(%a) : extern("C") (i32) -> i32;"#));
    assert!(merged(r#"%k = builtin.call @reads_any(%a) : extern("C") (i32) -> i32;"#));
    assert!(merged(r#"%k = builtin.call @allocates(%a) : extern("C") (i32) -> ptr<i32>;"#));
    assert!(!merged(
        r#"%k = builtin.call @writes(%p, %a) : extern("C") (ptr<i32>, i32) -> i32;"#
    ));
    assert!(!merged(r#"%k = builtin.call @writes_any(%a) : extern("C") (i32) -> i32;"#));
    assert!(!merged(r#"%k = builtin.call @unknown(%a) : extern("C") (i32) -> i32;"#));
    assert!(!merged(r#"%k = builtin.call @alias_unknown(%a) : extern("C") (i32) -> i32;"#));
    assert!(!merged(r#"%k = builtin.call_indirect %g(%a) : extern("C") (i32) -> i32;"#));
    assert!(!merged("%k = test.effects <{ known = #builtin.bool<false> }>;"));
    // a write inside a region between them
    assert!(!merged(
        r#"scf.if %c then {
        %k = builtin.call @writes_any(%a) : extern("C") (i32) -> i32;
        scf.yield : ();
    } else {
        scf.yield : ();
    } : (i1) -> ();"#
    ));
    // an unknown effect inside a region between them
    assert!(!merged(
        r#"scf.if %c then {
        %k = builtin.call @unknown(%a) : extern("C") (i32) -> i32;
        scf.yield : ();
    } else {
        scf.yield : ();
    } : (i1) -> ();"#
    ));

    for (callee, signature) in [
        ("@writes_any", r#"extern("C") (i32) -> i32"#),
        ("@unknown", r#"extern("C") (i32) -> i32"#),
    ] {
        let text = after(
            CSE,
            &format!(
                r#"
    %x = builtin.call {callee}(%a) : {signature};
    %y = builtin.call {callee}(%a) : {signature};
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#
            ),
        );
        assert_eq!(text.matches("builtin.call").count(), 2, "{callee}: {text}");
    }
    // two equal allocations are two allocations
    let text = after(
        CSE,
        r#"
    %x = builtin.call @allocates(%a) : extern("C") (i32) -> ptr<i32>;
    %y = builtin.call @allocates(%a) : extern("C") (i32) -> ptr<i32>;
    test.store %x, %a : (ptr<i32>, i32);
    test.store %y, %b : (ptr<i32>, i32);
    builtin.ret %a : (i32);"#,
    );
    assert_eq!(text.matches("builtin.call").count(), 2, "{text}");
}

/// Unused calls through dead-code elimination in each effect state.
#[test]
fn dce_of_unused_calls_in_each_effect_state() {
    let kept = |call: &str| {
        let text = after(DCE, &format!("{call};\nbuiltin.ret %a : (i32);"));
        text.contains("builtin.call")
    };
    assert!(!kept(r#"%k = builtin.call @pure(%a) : extern("C") (i32) -> i32"#));
    assert!(!kept(r#"%k = builtin.call @alias_pure(%a) : extern("C") (i32) -> i32"#));
    assert!(!kept(r#"%k = builtin.call @alias_alias_pure(%a) : extern("C") (i32) -> i32"#));
    assert!(!kept(r#"%k = builtin.call @reads(%p) : extern("C") (ptr<i32>) -> i32"#));
    assert!(!kept(r#"%k = builtin.call @reads_any(%a) : extern("C") (i32) -> i32"#));
    assert!(!kept(r#"%k = builtin.call @allocates(%a) : extern("C") (i32) -> ptr<i32>"#));
    assert!(kept(
        r#"%k = builtin.call @writes(%p, %a) : extern("C") (ptr<i32>, i32) -> i32"#
    ));
    assert!(kept(r#"%k = builtin.call @writes_any(%a) : extern("C") (i32) -> i32"#));
    assert!(kept(r#"builtin.call @frees(%p) : extern("C") (ptr<i32>)"#));
    assert!(kept(r#"%k = builtin.call @unknown(%a) : extern("C") (i32) -> i32"#));
    assert!(kept(r#"%k = builtin.call @alias_unknown(%a) : extern("C") (i32) -> i32"#));
}

/// Two modules of one world, each with its own `@callee` (one declares no effects, one states
/// none) and a caller of it with the same text: CSE and DCE run on the world treat each call by
/// the callee of its own module.
#[test]
fn equal_calls_in_two_modules_are_not_confused() {
    use runic_core::dialects::builtin::World;

    let source = r#"
builtin.world {
    builtin.component public @c {
        builtin.module public @one {
            builtin.function public extern("C") @callee() -> i32 effects();
            builtin.function public extern("C") @user() -> i32 {
                %x = builtin.call @callee() : extern("C") () -> i32;
                %y = builtin.call @callee() : extern("C") () -> i32;
                %s = arith.add %x, %y;
                builtin.ret %s : (i32);
            };
        };
        builtin.module public @two {
            builtin.function public extern("C") @callee() -> i32;
            builtin.function public extern("C") @user() -> i32 {
                %x = builtin.call @callee() : extern("C") () -> i32;
                %y = builtin.call @callee() : extern("C") () -> i32;
                %s = arith.add %x, %y;
                builtin.ret %s : (i32);
            };
        };
    };
};"#;
    for pass in [CSE, DCE] {
        let context = Rc::new(Context::default());
        let world = parse::parse::<World>(ParserConfig::new(context.clone()), "worlds", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        let mut passes = PassManager::on::<World>(context.clone(), Nesting::Implicit);
        passes.add_pass(PASSES[pass].1()).unwrap();
        passes.enable_verifier(true);
        passes.run(world.as_operation_ref()).unwrap();
        let text = world.as_operation_ref().borrow().to_string();
        let one = &text[text.find("@one").unwrap()..text.find("@two").unwrap()];
        let two = &text[text.find("@two").unwrap()..];
        if pass == CSE {
            assert_eq!(one.matches("builtin.call").count(), 1, "{text}");
        }
        assert_eq!(two.matches("builtin.call").count(), 2, "{text}");
    }
}

// ---------------------------------------------------------------------------------------------
// The canonicalization of `builtin.call_indirect` of a `builtin.function_ref`
// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*, as `runic_core::testing::text::Rng`
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Sig {
    cc: &'static str,
    /// type and whether it carries `{ extend = #builtin.zext }`
    params: Vec<(&'static str, bool)>,
    result: Option<&'static str>,
}

impl Sig {
    fn random(rng: &mut Rng) -> Self {
        const TYPES: [&str; 3] = ["i8", "i32", "i64"];
        let params = (0..rng.below(3))
            .map(|_| {
                let ty = TYPES[rng.below(3)];
                (ty, ty == "i8" && rng.chance(30))
            })
            .collect();
        Sig {
            cc: if rng.chance(75) { "C" } else { "fast" },
            params,
            result: rng.chance(70).then(|| TYPES[1 + rng.below(2)]),
        }
    }

    fn params_text(&self) -> String {
        self.params
            .iter()
            .map(|(ty, zext)| {
                if *zext {
                    format!("{ty} {{ extend = #builtin.zext }}")
                } else {
                    ty.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// As a declaration's and a call's signature
    fn text(&self) -> String {
        let result = self.result.map(|ty| format!(" -> {ty}")).unwrap_or_default();
        format!("extern(\"{}\") ({}){result}", self.cc, self.params_text())
    }

    fn function_type(&self) -> String {
        let params = self.params.iter().map(|(ty, _)| *ty).collect::<Vec<_>>().join(", ");
        format!("({params}) -> {}", self.result.unwrap_or("()"))
    }

    fn arguments(&self) -> String {
        self.params
            .iter()
            .map(|(ty, _)| format!("%x{}", &ty[1..]))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The same types, with the calling convention or an attribute changed
    fn perturbed(&self, rng: &mut Rng) -> Self {
        let mut sig = self.clone();
        let narrow = sig.params.iter().position(|(ty, _)| *ty == "i8");
        match narrow {
            Some(index) if rng.chance(50) => sig.params[index].1 = !sig.params[index].1,
            _ => sig.cc = if sig.cc == "C" { "fast" } else { "C" },
        }
        sig
    }
}

/// One random module: functions, aliases (chains included), and callers that call through
/// references with signatures that match their callee's or differ from it in the calling
/// convention or a parameter attribute. Returns the text, and per caller how many direct calls
/// and how many indirect calls the canonicalizer must leave.
fn random_module(rng: &mut Rng) -> (String, Vec<(usize, usize)>) {
    let mut text = String::from("builtin.module public @m {\n");
    // name -> signature of its canonical target, and whether an unused call of it is dead
    let mut symbols: Vec<(String, Sig, bool)> = Vec::new();
    for index in 0..1 + rng.below(4) {
        let sig = Sig::random(rng);
        let visibility = if rng.chance(50) { "public" } else { "private" };
        let effects = ["", " effects()", " effects(read)", " effects(write)"][rng.below(4)];
        let result = sig.result.map(|ty| format!(" -> {ty}")).unwrap_or_default();
        // a declaration with no parameters is written with an empty parameter list
        text += &format!(
            "    builtin.function {visibility} extern(\"{}\") @f{index}({}){result}{effects};\n",
            sig.cc,
            sig.params_text()
        );
        symbols.push((
            format!("@f{index}"),
            sig,
            effects == " effects()" || effects == " effects(read)",
        ));
    }
    for index in 0..rng.below(4) {
        let (target, sig, dead) = symbols[rng.below(symbols.len())].clone();
        text += &format!("    builtin.function_alias private @a{index} -> {target};\n");
        symbols.push((format!("@a{index}"), sig, dead));
    }
    let mut expected = Vec::new();
    for caller in 0..1 + rng.below(2) {
        text += &format!(
            "    builtin.function public extern(\"C\") @caller{caller}(%x8: i8, %x32: i32, %x64: \
             i64, %g: (i32) -> i32) -> i32 {{\n"
        );
        let (mut direct, mut indirect) = (0, 0);
        let mut acc = String::from("%x32");
        for statement in 0..1 + rng.below(5) {
            if rng.chance(15) {
                // a value that is not a reference: stays indirect
                text += &format!(
                    "        %r{statement} = builtin.call_indirect %g(%x32) : extern(\"C\") (i32) \
                     -> i32;\n"
                );
                indirect += 1;
                continue;
            }
            let (name, sig, dead_when_unused) = symbols[rng.below(symbols.len())].clone();
            let call_sig = if rng.chance(65) {
                sig.clone()
            } else {
                sig.perturbed(rng)
            };
            let used = call_sig.result == Some("i32") && rng.chance(70);
            if call_sig != sig {
                indirect += 1;
            } else if used || !dead_when_unused {
                // the direct call of a callee that declares no effects, or only reads, is erased
                // by the driver when its result is unused
                direct += 1;
            }
            text += &format!(
                "        %f{statement} = builtin.function_ref {name} : {};\n",
                sig.function_type()
            );
            let define = if call_sig.result.is_some() {
                format!("%r{statement} = ")
            } else {
                String::new()
            };
            text += &format!(
                "        {define}builtin.call_indirect %f{statement}({}) : {};\n",
                call_sig.arguments(),
                call_sig.text()
            );
            if used {
                text += &format!("        %s{statement} = arith.add {acc}, %r{statement};\n");
                acc = format!("%s{statement}");
            }
        }
        text += &format!("        builtin.ret {acc} : (i32);\n    }};\n");
        expected.push((direct, indirect));
    }
    text += "};\n";
    (text, expected)
}

/// Random modules through the canonicalizer: the result verifies; an indirect call of a reference
/// becomes a direct call exactly when the signatures are equal; a reference left without uses is
/// erased; a second run changes nothing.
#[test]
fn fuzz_call_indirect_of_function_ref() {
    const ROUNDS: usize = 400;
    let seed = seed(0x9e37_79b9_7f4a_7c15);
    let mut rng = Rng(seed);
    let mut rewritten = 0;
    let mut kept = 0;
    for round in 0..ROUNDS {
        let round = format!("seed {seed:#x} (CALLS_TEST_SEED={seed}), round {round}");
        let (source, expected) = random_module(&mut rng);
        let context = Rc::new(Context::default());
        let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "fuzz", &*source)
            .into_result()
            .unwrap_or_else(|err| panic!("round {round}: {}\n{source}", describe(&err)));
        for (caller, (direct, indirect)) in expected.iter().enumerate() {
            let function =
                module.borrow().symbol_manager().lookup_op(format!("caller{caller}")).unwrap();
            run(&context, function, Box::<Canonicalizer>::default()).unwrap_or_else(|err| {
                panic!("round {round}: {}\n{source}\n{}", describe(&err), function.borrow())
            });
            module.as_operation_ref().borrow().recursively_verify().unwrap_or_else(|err| {
                panic!("round {round}: {}\n{source}\n{}", describe(&err), function.borrow())
            });
            let text = function.borrow().to_string();
            assert_eq!(
                (
                    text.matches("builtin.call ").count(),
                    text.matches("builtin.call_indirect").count()
                ),
                (*direct, *indirect),
                "round {round}:\n{source}\n{text}"
            );
            // every reference that remains is the callee of an indirect call that remains
            assert!(
                text.matches("builtin.function_ref").count() <= *indirect,
                "round {round}:\n{source}\n{text}"
            );
            run(&context, function, Box::<Canonicalizer>::default()).unwrap();
            assert_eq!(function.borrow().to_string(), text, "round {round}: not a fixed point");
            rewritten += direct;
            kept += indirect;
            // every direct call names the symbol its reference named, and verifies (above)
            assert!(!text.contains("::@m"), "round {round}:\n{text}");
        }
        // The text of the module parses back and verifies
        let printed = module.as_operation_ref().borrow().to_string();
        let context = Rc::new(Context::default());
        parse::parse::<Module>(ParserConfig::new(context.clone()), "fuzz-again", &*printed)
            .into_result()
            .unwrap_or_else(|err| panic!("round {round}: {}\n{printed}", describe(&err)));
    }
    assert!(
        rewritten > ROUNDS / 2 && kept > ROUNDS / 2,
        "seed {seed:#x}: {rewritten} rewritten, {kept} kept"
    );
}

/// The canonicalizer's direct call keeps the attribute dictionary of the indirect call.
#[test]
fn the_canonicalizer_keeps_the_attribute_dictionary_of_an_indirect_call() {
    let context = Rc::new(Context::default());
    let (_module, function) = make(
        &context,
        r#"
    %f = builtin.function_ref @unknown : (i32) -> i32;
    %r = builtin.call_indirect %f(%a) : extern("C") (i32) -> i32 attributes { note = #builtin.i32<7> };
    builtin.ret %r : (i32);"#,
    )
    .unwrap();
    run(&context, function, Box::<Canonicalizer>::default()).unwrap();
    let text = function.borrow().to_string();
    assert!(text.contains("builtin.call @unknown(%2)"), "{text}");
    assert!(text.contains("note"), "{text}");
}

// ---------------------------------------------------------------------------------------------
// Symbol uses of erased operations
// ---------------------------------------------------------------------------------------------

/// What [uses_of_callee_after] reports.
#[derive(Debug, PartialEq, Eq)]
struct CalleeUses {
    /// The number of uses in `@callee`'s use list
    uses: usize,
    /// The number of known callers `CallableUseSnapshot` reports for `@callee`
    known_callers: usize,
    address_taken: bool,
    unknown_callers: bool,
}

/// A module with a private `@callee`, and `@caller` with `body`; after `pass` ran on the caller:
/// what the use list of `@callee` and `CallableUseSnapshot` of the module say of it, and the text
/// of the caller.
fn uses_of_callee_after(pass: MakePass, body: &str) -> (CalleeUses, String) {
    use runic_analysis::analyses::callable_uses::CallableUseSnapshot;
    use runic_core::{Usable, dialects::builtin::ModuleBuilder};

    let context = Rc::new(Context::default());
    let source = format!(
        "builtin.module public @m {{
    builtin.function private extern(\"C\") @callee(%x: i32) -> i32 effects() {{
        builtin.ret %x : (i32);
    }};
    builtin.function public extern(\"C\") @caller(%a: i32) -> i32 {{
{body}
    }};
}};"
    );
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "uses", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let caller = module.borrow().symbol_manager().lookup_op("caller").unwrap();
    run(&context, caller, pass()).unwrap();
    module.as_operation_ref().borrow().recursively_verify().unwrap();
    let target = ModuleBuilder::new(module).resolve_callable("callee").unwrap().target();
    let callee = ModuleBuilder::new(module).get_function("callee").unwrap();
    let uses = callee.borrow().iter_uses().count();
    let snapshot = CallableUseSnapshot::new(&module.as_operation_ref().borrow());
    let info = snapshot.get(target).unwrap();
    let report = CalleeUses {
        uses,
        known_callers: info.known_callers().len(),
        address_taken: info.is_address_taken(),
        unknown_callers: info.has_unknown_callers(),
    };
    (report, caller.borrow().to_string())
}

const INDIRECT: &str = r#"
        %f = builtin.function_ref @callee : (i32) -> i32;
        %r = builtin.call_indirect %f(%a) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);"#;

const TWO_CALLS: &str = r#"
        %r = builtin.call @callee(%a) : extern("C") (i32) -> i32;
        %dead = builtin.call @callee(%a) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);"#;

/// The use list of a symbol holds exactly the operations that remain: what a pass erased is not
/// a use, so the callee of a canonicalized indirect call is not address-taken, and a callee one of
/// whose calls was erased (`dce`) or merged (`cse`) is not used from outside the module.
#[test]
fn an_erased_operation_leaves_the_use_list_of_its_symbol() {
    let one_known_caller = CalleeUses {
        uses: 1,
        known_callers: 1,
        address_taken: false,
        unknown_callers: false,
    };
    let (report, _) = uses_of_callee_after(|| Box::<Canonicalizer>::default(), INDIRECT);
    assert_eq!(report, one_known_caller);
    for pass in [(|| Box::<DeadCodeElimination>::default()) as MakePass, || {
        Box::<CommonSubexpressionElimination>::default()
    }] {
        let (report, text) = uses_of_callee_after(pass, TWO_CALLS);
        assert_eq!(text.matches("builtin.call").count(), 1, "{text}");
        assert_eq!(report, one_known_caller, "{text}");
    }
}

/// After the canonicalizer rewrites an indirect call of a reference to a private function into a
/// direct call, the function is not address-taken and has exactly that one known caller.
#[test]
fn a_canonicalized_indirect_call_leaves_its_callee_with_one_known_caller() {
    use runic_analysis::analyses::callable_uses::CallableUseSnapshot;
    use runic_core::{CallOpInterface, dialects::builtin::ModuleBuilder};

    let context = Rc::new(Context::default());
    let source = format!(
        "builtin.module public @m {{
    builtin.function private extern(\"C\") @callee(%x: i32) -> i32 {{
        builtin.ret %x : (i32);
    }};
    builtin.function public extern(\"C\") @caller(%a: i32) -> i32 {{
{INDIRECT}
    }};
}};"
    );
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "indirect", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let caller = module.borrow().symbol_manager().lookup_op("caller").unwrap();
    run(&context, caller, Box::<Canonicalizer>::default()).unwrap();
    let text = caller.borrow().to_string();
    assert!(
        !text.contains("function_ref") && text.contains("builtin.call @callee"),
        "{text}"
    );

    let mut call = None;
    caller.borrow().prewalk_all(|op: &runic_core::Operation| {
        if op.implements::<dyn CallOpInterface>() {
            call = Some(op.as_operation_ref());
        }
    });
    let call = call.expect("the direct call");
    let target = ModuleBuilder::new(module).resolve_callable("callee").unwrap().target();
    let snapshot = CallableUseSnapshot::new(&module.as_operation_ref().borrow());
    let info = snapshot.get(target).unwrap();
    assert!(!info.is_address_taken(), "{text}");
    assert!(!info.has_unknown_callers(), "{text}");
    assert_eq!(info.known_callers().len(), 1, "{text}");
    assert!(info.known_callers().contains(&call), "{text}");
}

// ---------------------------------------------------------------------------------------------
// The passes over IR that holds symbol references that do not resolve
// ---------------------------------------------------------------------------------------------

/// Each source of the corpus with one declaration deleted or renamed, or one reference renamed,
/// parsed with verification off: no pass panics on any function of it.
#[test]
fn fuzz_no_pass_panics_on_a_dangling_symbol() {
    let mut cases = 0;
    let mut problems = Vec::new();
    for source in CORPUS {
        for (what, text, _fails) in mutations(source) {
            for (pass, make_pass) in PASSES {
                cases += 1;
                let outcome = std::panic::catch_unwind(|| {
                    let context = Rc::new(Context::default());
                    let config = ParserConfig::new(context.clone()).verify_after_parse(false);
                    let root = parse::parse_any(config, "dangling", &*text)
                        .into_result()
                        .map_err(|err| describe(&err))?;
                    let mut functions = Vec::new();
                    root.borrow().prewalk_all(|op: &Operation| {
                        if op.downcast_ref::<Function>().is_some_and(|f| !f.is_declaration()) {
                            functions.push(op.as_operation_ref());
                        }
                    });
                    for function in functions {
                        let mut passes =
                            PassManager::on::<Function>(context.clone(), Nesting::Implicit);
                        passes.add_pass(make_pass()).unwrap();
                        passes.enable_verifier(false);
                        // an error is a diagnostic, which is fine; a panic is not
                        let _ = passes.run(function);
                    }
                    let _ = root.borrow().to_string();
                    Ok::<_, String>(())
                });
                match outcome {
                    Err(_) => problems.push(format!("PANIC {pass}, {what}:\n{text}")),
                    Ok(Err(err)) => problems.push(format!("PARSE FAILED {what}: {err}")),
                    Ok(Ok(())) => (),
                }
            }
        }
    }
    assert!(cases > 400, "{cases}");
    assert!(
        problems.is_empty(),
        "{} problems of {cases} cases:\n{}",
        problems.len(),
        problems.join("\n\n")
    );
}

// ---------------------------------------------------------------------------------------------
// Declared effects on a definition are trusted
// ---------------------------------------------------------------------------------------------

/// `@caller` after `pass` ran on the module holding `callee` and a caller that calls it and
/// ignores the result.
fn caller_after(pass: MakePass, callee: &str) -> String {
    let context = Rc::new(Context::default());
    let source = format!(
        "builtin.module public @m {{
    {callee}
    builtin.function public extern(\"C\") @caller(%p: ptr<i32>) -> i32 {{
        %r = builtin.call @callee(%p) : extern(\"C\") (ptr<i32>) -> i32;
        %zero = arith.constant 0 : i32;
        builtin.ret %zero : (i32);
    }};
}};"
    );
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "trust", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
    passes.add_pass(pass()).unwrap();
    passes.enable_verifier(true);
    passes.run(module.as_operation_ref()).unwrap();
    let caller = module.borrow().symbol_manager().lookup_op("caller").unwrap();
    caller.borrow().to_string()
}

/// As designed: a definition that declares `effects()` is believed, whatever its body does. A
/// call of a function that stores through its argument, or never returns, is removed when its
/// result is unused; nothing checks the declaration against the body.
#[test]
fn a_definition_that_declares_no_effects_is_believed() {
    let stores = r#"builtin.function private extern("C") @callee(%p: ptr<i32>) -> i32 effects() {
        %one = arith.constant 1 : i32;
        test.store %p, %one : (ptr<i32>, i32);
        builtin.ret %one : (i32);
    };"#;
    let text = caller_after(|| Box::<DeadCodeElimination>::default(), stores);
    assert!(!text.contains("builtin.call"), "{text}");

    let calls_unknown = r#"builtin.function private extern("C") @other(ptr<i32>) -> i32;
    builtin.function private extern("C") @callee(%p: ptr<i32>) -> i32 effects() {
        %r = builtin.call @other(%p) : extern("C") (ptr<i32>) -> i32;
        builtin.ret %r : (i32);
    };"#;
    let text = caller_after(|| Box::<DeadCodeElimination>::default(), calls_unknown);
    assert!(!text.contains("builtin.call"), "{text}");

    // the same definitions without the declaration: the call stays
    let text =
        caller_after(|| Box::<DeadCodeElimination>::default(), &stores.replace(" effects()", ""));
    assert!(text.contains("builtin.call"), "{text}");
}

/// An unused call of a function declaring `effects()` inside that function is erased by DCE and
/// by the canonicalizer, run on the module and on the function itself, without a borrow panic:
/// erasing the call takes its use out of the use list of the function the pass is running on.
#[test]
fn a_pass_erases_a_dead_recursive_call() {
    let source = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 effects() {
        %r = builtin.call @f(%a) : extern("C") (i32) -> i32;
        builtin.ret %a : (i32);
    };
};"#;
    let passes: [(&str, MakePass); 2] = [
        ("dce", || Box::<DeadCodeElimination>::default()),
        ("canonicalize", || Box::<Canonicalizer>::default()),
    ];
    for (name, pass) in passes {
        for on_module in [true, false] {
            let context = Rc::new(Context::default());
            let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "rec", source)
                .into_result()
                .unwrap_or_else(|err| panic!("{}", describe(&err)));
            let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
            let uses = || function.borrow().as_symbol_ref().unwrap().borrow().iter_uses().count();
            assert_eq!(uses(), 1);
            if on_module {
                let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
                passes.add_pass(pass()).unwrap();
                passes.enable_verifier(true);
                passes.run(module.as_operation_ref()).unwrap();
            } else {
                run(&context, function, pass()).unwrap();
            }
            let text = function.borrow().to_string();
            assert!(!text.contains("builtin.call"), "{name}, on the module: {on_module}\n{text}");
            assert_eq!(uses(), 0, "{name}, on the module: {on_module}");
            module.as_operation_ref().borrow().recursively_verify().unwrap();
        }
    }
}

/// The SCCP pass does not follow calls (its solver is not interprocedural): the result of a call
/// of a private function that returns a constant is not replaced, so the pass never erases a
/// call, whatever its callee declares.
#[test]
fn sccp_leaves_calls_alone() {
    for effects in ["", " effects()", " effects(write)"] {
        let context = Rc::new(Context::default());
        let source = format!(
            "builtin.module public @m {{
    builtin.function private extern(\"C\") @answer() -> i32{effects} {{
        %c = arith.constant 42 : i32;
        builtin.ret %c : (i32);
    }};
    builtin.function public extern(\"C\") @caller() -> i32 {{
        %r = builtin.call @answer() : extern(\"C\") () -> i32;
        builtin.ret %r : (i32);
    }};
}};"
        );
        let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "sccp", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
        let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
        passes.add_pass(Box::<SparseConditionalConstantPropagation>::default()).unwrap();
        passes.enable_verifier(true);
        passes.run(module.as_operation_ref()).unwrap();
        let caller = module.borrow().symbol_manager().lookup_op("caller").unwrap();
        let text = caller.borrow().to_string();
        assert!(text.contains("builtin.call"), "effects `{effects}`:\n{text}");
    }
}

// ---------------------------------------------------------------------------------------------
// A call of a function that never returns
// ---------------------------------------------------------------------------------------------

const NEVER: &str = r#"
builtin.module public @m {
    builtin.function public extern("C") @abort(i32) -> never EFFECTS;
    builtin.function public extern("C") @f(%c: i1, %a: i32, %g: (i32) -> never) -> i32 {
        scf.if %c then {
            CALL
            scf.yield : ();
        } else {
            scf.yield : ();
        } : (i1) -> ();
        builtin.ret %a : (i32);
    };
};"#;

/// `@f` of [NEVER], with `@abort` declaring `effects` and `call` in the `scf.if`, after `pass`.
fn never_after(pass: MakePass, effects: &str, call: &str) -> String {
    let context = Rc::new(Context::default());
    let source = NEVER.replace("EFFECTS", effects).replace("CALL", call);
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "never", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    run(&context, function, pass()).unwrap();
    module.as_operation_ref().borrow().recursively_verify().unwrap();
    function.borrow().to_string()
}

/// A call whose signature says it never returns is not dead, whatever memory effects its callee
/// declares: no effects, only reads, a write, or nothing at all.
#[test]
fn a_call_that_never_returns_is_kept() {
    let call = r#"builtin.call @abort(%a) : extern("C") (i32) -> never;"#;
    for effects in ["effects()", "effects(read)", "", "effects(write)"] {
        for pass in [(|| Box::<DeadCodeElimination>::default()) as MakePass, || {
            Box::<Canonicalizer>::default()
        }] {
            let text = never_after(pass, effects, call);
            assert!(text.contains("builtin.call @abort"), "{effects}: {text}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Constant conditions (the region or block that runs keeps its operation)
// ---------------------------------------------------------------------------------------------

/// An operation with unknown effects in the one region or block that runs, under a constant
/// condition: whatever a pass folds away, the operation is still there, once.
#[test]
fn folding_a_constant_condition_keeps_the_operation_that_runs() {
    const BODIES: [(&str, &str); 5] = [
        (
            "the then-region of an if on true",
            r#"
    %t = arith.constant 1 : i1;
    %x = scf.if %t then {
        %u1 = {U};
        scf.yield %u1 : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %x : (i32);"#,
        ),
        (
            "the else-region of an if on false, result unused",
            r#"
    %t = arith.constant 0 : i1;
    %x = scf.if %t then {
        scf.yield %b : (i32);
    } else {
        %u1 = {U};
        scf.yield %a : (i32);
    } : (i1) -> (i32);
    builtin.ret %a : (i32);"#,
        ),
        (
            "the head of a loop whose condition is false",
            r#"
    %zero = arith.constant 0 : i32;
    %no = arith.constant 0 : i1;
    %count = scf.while %zero before {
    ^head(%i: i32):
        %u1 = {U};
        scf.condition %no, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        scf.yield %j : (i32);
    } : (i32) -> i32;
    builtin.ret %a : (i32);"#,
        ),
        (
            "the taken successor of a branch on true",
            r#"
    %t = arith.constant 1 : i1;
    cf.cond_br %t ^t, ^f : (i1);
^t:
    %u1 = {U};
    builtin.ret %a : (i32);
^f:
    builtin.ret %b : (i32);"#,
        ),
        (
            "the selected case of a switch on a constant",
            r#"
    %s = arith.constant 1 : i32;
    %x = scf.index_switch %s
    case 0 {
        scf.yield %a : (i32);
    }
    case 1 {
        %u1 = {U};
        scf.yield %b : (i32);
    }
    default {
        scf.yield %a : (i32);
    } : (i32);
    builtin.ret %a : (i32);"#,
        ),
    ];
    let mut failures = Vec::new();
    for (position, body) in BODIES {
        for (kind, unknown) in UNKNOWN {
            let body = body.replace("{U}", unknown);
            let context = Rc::new(Context::default());
            let (module, function) = match make(&context, &body) {
                Ok(made) => made,
                Err(err) => panic!("'{position}': {}", describe(&err)),
            };
            for _ in 0..2 {
                for (pass, make_pass) in PASSES {
                    if let Err(err) = run(&context, function, make_pass()) {
                        failures
                            .push(format!("{pass} on '{position}' / {kind}: {}", describe(&err)));
                    }
                    let unknown = collect(&function.borrow(), has_unknown_effects).len();
                    if unknown != 1 {
                        failures.push(format!(
                            "{pass} on '{position}' / {kind}: {unknown} operations with unknown \
                             effects\n{}",
                            function.borrow()
                        ));
                    }
                }
            }
            module.borrow().recursively_verify().unwrap();
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n\n"));
}
