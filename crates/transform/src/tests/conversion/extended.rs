//! Whole modules of a richer shape converted, checked against a model, evaluated, and run
//! through every pass of the crate afterwards.
//!
//! - `xgen`: the generator (see its documentation) and the profiles of its modules.
//! - `xeval`: the evaluator, with memory and external functions.
//! - `model`: what each converted function must be, from what the generator knows of it, and the
//!   check of the module's symbol uses.
//! - `passes`: the pass pipeline over a converted module.
//! - `probes`: fixed cases, each pinning what holds for one question: converters that misbehave,
//!   conversions rooted at a world, a module or a function, `scf` over a converted type, large
//!   functions, and what the passes do on a module.
//!
//! The conversion: `i128` becomes `(i64, i64)`, `i8` (the marker) becomes nothing, function types
//! convert by their parts. After a conversion that succeeds: the module verifies; no cast
//! remains; in a full conversion no converted type is held; every function's signature, entry
//! block, declared effects and locals are what the model says; every symbol use belongs to a
//! call, a reference or an alias of the module; every function computes what it computed
//! (results and memory); a second conversion changes nothing; the printed module parses, verifies,
//! prints the same up to value numbers and computes the same; then the passes of this crate run
//! over the converted module, in a random order with `cfg-to-scf` last, and it is evaluated again
//! after each.
//!
//! The tests run fixed seeds under the crate's convention (`TRANSFORM_TEST_SEED`,
//! `TRANSFORM_TEST_CASES`); `TRANSFORM_TEST_SHOW=1` adds the modules to a failure's report.

extern crate std;

use alloc::{
    collections::BTreeMap,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use std::println;

use runic_dialect_arith::{self as arith, ArithDialect};
use runic_dialect_mem::MemDialect;
use runic_ir::{
    Context, Int, OperationRef, Report, Type, Value,
    conversion::{
        ConversionConfig, ConversionPatternSet, ConversionResult, ConversionTarget,
        DynamicLegalityResult, TypeConversion, TypeConverter, add_builtin_legality_by_type,
        apply_full_conversion, apply_partial_conversion, populate_builtin_conversion_patterns,
    },
    dialects::builtin::{FunctionAlias, Module, UnrealizedConversionCast},
    patterns::PatternKind,
    testing::text::describe,
};

use self::{
    model::{check_model, check_symbol_uses},
    passes::run_pass,
    xeval::{Evaluator, TOO_DEEP, TOO_LONG, Val},
    xgen::{Arg, Generated, Profile, Sig, Ty},
};
use super::{
    Family, Mode, all_operations, converted_type_held, mask, parse_module,
    patterns::{self, Kind, Rule},
    random::Rng,
};
use crate::tests::{cases, seeds};

mod model;
mod passes;
mod probes;
mod xeval;
mod xgen;

// ---------------------------------------------------------------------------------------------
// The conversion
// ---------------------------------------------------------------------------------------------

/// One conversion: its mode and the `arith` kinds that have a pattern and a rule.
#[derive(Debug, Clone)]
struct Opts {
    mode: Mode,
    kinds: Vec<Kind>,
    /// Run the passes over the converted module.
    passes: bool,
}

impl Opts {
    fn full() -> Self {
        Self {
            mode: Mode::Full,
            kinds: Kind::of(Family::Wide).to_vec(),
            passes: true,
        }
    }

    fn partial(rng: &mut Rng) -> Self {
        Self {
            mode: Mode::Partial,
            kinds: Kind::of(Family::Wide)
                .iter()
                .copied()
                .filter(|kind| *kind != Kind::Or)
                .filter(|_| !rng.chance(40))
                .collect(),
            passes: true,
        }
    }
}

/// `i128` becomes `(i64, i64)`, `i8` becomes nothing, and function types convert by their parts.
/// With `materialize`: the halves are joined and split as `patterns::wide_converter` does, and a
/// marker a user still expects is the truncation of an `i64` zero.
fn mix_converter(context: &Context, materialize: bool) -> TypeConverter {
    let (i8, i64, i128) = (context.i8(), context.i64(), context.i128());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| {
        if ty == i128 {
            Some(TypeConversion::many([i64, i64]))
        } else if ty == i8 {
            Some(TypeConversion::drop())
        } else {
            None
        }
    });
    if materialize {
        converter.add_source_materialization(move |rewriter, inputs, ty, span| {
            if ty == i8 && inputs.is_empty() {
                let zero = rewriter.create_op::<arith::Constant, _>(span, (Int::from(0i64),))?;
                let zero = patterns::result(&zero);
                let marker = rewriter.create_op::<arith::Trunc, _>(span, (zero, i8))?;
                return Ok(Some(patterns::result(&marker)));
            }
            let [low, high] = inputs else {
                return Ok(None);
            };
            if ty != i128 {
                return Ok(None);
            }
            let low = patterns::result(&rewriter.create_op::<arith::Zext, _>(span, (*low, i128))?);
            let high =
                patterns::result(&rewriter.create_op::<arith::Zext, _>(span, (*high, i128))?);
            let amount = patterns::sixty_four(rewriter, span, i128)?;
            let high =
                patterns::result(&rewriter.create_op::<arith::Shl, _>(span, (high, amount))?);
            let joined = patterns::result(&rewriter.create_op::<arith::Bor, _>(span, (high, low))?);
            Ok(Some(joined))
        });
        converter.add_target_materialization(move |rewriter, input, types, span| {
            if types != [i64, i64] || input.borrow().ty() != i128 {
                return Ok(None);
            }
            let low = patterns::result(&rewriter.create_op::<arith::Trunc, _>(span, (input, i64))?);
            let amount = patterns::sixty_four(rewriter, span, i128)?;
            let high =
                patterns::result(&rewriter.create_op::<arith::Shr, _>(span, (input, amount))?);
            let high = patterns::result(&rewriter.create_op::<arith::Trunc, _>(span, (high, i64))?);
            Ok(Some(runic_ir::smallvec![low, high]))
        });
    }
    converter.with_function_types()
}

fn mix_target(context: &Rc<Context>, opts: &Opts, converter: &TypeConverter) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target.add_legal_op::<Module>().add_legal_op::<FunctionAlias>();
    add_builtin_legality_by_type(&mut target, converter);
    {
        let converter = converter.clone();
        target.add_dynamically_legal_dialect::<MemDialect, _>(move |op| {
            DynamicLegalityResult::legal_if(converter.is_legal_op(op))
        });
    }
    match opts.mode {
        Mode::Full => {
            let converter = converter.clone();
            target.add_dynamically_legal_dialect::<ArithDialect, _>(move |op| {
                DynamicLegalityResult::legal_if(converter.is_legal_op(op))
            });
        }
        Mode::Partial => {
            for kind in &opts.kinds {
                kind.add_rule(&mut target, converter);
            }
        }
    }
    target
}

/// The patterns that erase an `arith` operation whose result is a marker.
fn marker_rules(context: &Rc<Context>, converter: &TypeConverter, kinds: &[Kind]) -> Vec<Rule> {
    let mut rules = Vec::new();
    for (kind, name) in [
        (Kind::Constant, "marker-constant"),
        (Kind::Add, "marker-add"),
        (Kind::Xor, "marker-xor"),
    ] {
        if !kinds.contains(&kind) {
            continue;
        }
        rules.push(Rule::new(
            context,
            name,
            PatternKind::Operation(kind.name(context)),
            converter,
            [],
            |op, _, rw| {
                let marker = {
                    let op = op.borrow();
                    op.results()[0].borrow().ty() == op.context().i8()
                };
                if !marker {
                    return Ok(false);
                }
                rw.replace_op_with_multiple(op, &[&[]])?;
                Ok(true)
            },
        ));
    }
    rules
}

fn mix_patterns(
    context: &Rc<Context>,
    opts: &Opts,
    converter: &TypeConverter,
) -> Result<ConversionPatternSet, Report> {
    let mut set = ConversionPatternSet::new(context.clone());
    populate_builtin_conversion_patterns(&mut set, converter)?;
    for rule in patterns::wide_patterns(context, converter, &opts.kinds) {
        set.push(rule);
    }
    for rule in marker_rules(context, converter, &opts.kinds) {
        set.push(rule);
    }
    Ok(set)
}

fn mix_convert(
    context: &Rc<Context>,
    root: OperationRef,
    opts: &Opts,
) -> Result<ConversionResult, Report> {
    let converter = mix_converter(context, opts.mode == Mode::Partial);
    let target = mix_target(context, opts, &converter);
    let set = mix_patterns(context, opts, &converter)?;
    match opts.mode {
        Mode::Full => apply_full_conversion(root, target, set, ConversionConfig::default()),
        Mode::Partial => apply_partial_conversion(root, target, set, ConversionConfig::default()),
    }
}

// ---------------------------------------------------------------------------------------------
// Evaluation and the checks
// ---------------------------------------------------------------------------------------------

/// What one call gave: its results and the cells its pointers named, afterwards.
type Observed = (Vec<Val>, Vec<u128>);

struct Bounds {
    depth: usize,
    steps: usize,
}

/// Evaluates `function` of the original module on `args`. `Ok(None)`: a bound was reached.
fn evaluate_original(
    evaluator: &mut Evaluator,
    function: &str,
    args: &[Arg],
) -> Result<Option<Observed>, String> {
    evaluator.memory.clear();
    let mut values = Vec::new();
    for arg in args {
        values.push(match arg {
            Arg::Int(value) => Val::Int(*value),
            Arg::Ptr(initial) => {
                evaluator.memory.push(*initial);
                Val::Ptr(evaluator.memory.len() - 1)
            }
            Arg::Func(name) => Val::Func(name.clone()),
        });
    }
    match evaluator.call(function, values) {
        Ok(results) => Ok(Some((results, evaluator.memory.clone()))),
        Err(err) if err == TOO_DEEP || err == TOO_LONG => Ok(None),
        Err(err) => Err(format!("the source does not evaluate @{function}: {err}")),
    }
}

/// Evaluates `function` of the converted module and compares with `expected`.
fn evaluate_converted(
    evaluator: &mut Evaluator,
    sig: &Sig,
    function: &str,
    args: &[Arg],
    expected: &Observed,
) -> Result<(), String> {
    evaluator.memory.clear();
    let mut values = Vec::new();
    for (ty, arg) in sig.params.iter().zip(args) {
        match (ty, arg) {
            (Ty::X, Arg::Int(value)) => {
                values.push(Val::Int(mask(*value, 64)));
                values.push(Val::Int(*value >> 64));
            }
            (Ty::M, _) => {}
            (_, Arg::Int(value)) => values.push(Val::Int(*value)),
            (_, Arg::Ptr(initial)) => {
                evaluator.memory.push(*initial);
                values.push(Val::Ptr(evaluator.memory.len() - 1));
            }
            (_, Arg::Func(name)) => values.push(Val::Func(name.clone())),
        }
    }
    let results = evaluator
        .call(function, values)
        .map_err(|err| format!("the result does not evaluate @{function}: {err}"))?;
    let mut results = results.iter();
    let mut next = || results.next().ok_or_else(|| String::from("too few results"));
    let int = |value: &Val| match value {
        Val::Int(value) => Ok(*value),
        other => Err(format!("{other:?} where an integer is returned")),
    };
    let mut actual = Vec::new();
    let mut wanted = Vec::new();
    for (ty, value) in sig.results.iter().zip(&expected.0) {
        match ty {
            Ty::X => {
                let low = int(next()?)?;
                let high = int(next()?)?;
                actual.push(Val::Int(low | high << 64));
                wanted.push(value.clone());
            }
            Ty::M => {}
            _ => {
                actual.push(next()?.clone());
                wanted.push(value.clone());
            }
        }
    }
    if next().is_ok() {
        return Err(String::from("too many results"));
    }
    if actual != wanted {
        return Err(format!("MISCOMPILE @{function}: gave {actual:x?}, expected {wanted:x?}"));
    }
    if evaluator.memory != expected.1 {
        return Err(format!(
            "MISCOMPILE @{function}: memory is {:x?}, expected {:x?}",
            evaluator.memory, expected.1
        ));
    }
    Ok(())
}

/// Evaluates every input the original evaluated on the converted `module`.
fn compare_all(
    module: OperationRef,
    case: &Generated,
    externs: &BTreeMap<String, (Vec<Type>, Vec<Type>)>,
    expected: &[Option<Observed>],
    bounds: &Bounds,
) -> Result<(), String> {
    let mut evaluator = Evaluator::new(module);
    evaluator.externs = externs.clone();
    evaluator.max_depth = bounds.depth * 2;
    evaluator.max_steps = bounds.steps * 100;
    for ((function, args), expected) in case.inputs.iter().zip(expected) {
        let Some(expected) = expected else {
            continue;
        };
        let index: usize = function[1..].parse().unwrap();
        evaluate_converted(&mut evaluator, &case.model.sigs[index], function, args, expected)
            .map_err(|err| format!("@{function}({args:x?}): {err}"))?;
    }
    Ok(())
}

/// `text` with every `%N` value name replaced by `%`.
fn strip_numbers(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '%' {
            while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                chars.next();
            }
        }
    }
    out
}

#[derive(Debug, Default, Clone, Copy)]
struct Stats {
    evaluated: usize,
    skipped: usize,
    operations: usize,
    calls: usize,
}

/// Converts `case` as `opts` says and checks everything the module documentation lists.
fn xcheck(case: &Generated, opts: &Opts, bounds: &Bounds, rng: &mut Rng) -> Result<Stats, String> {
    xcheck_with(case, opts, bounds, rng, &mut |context, root| mix_convert(context, root, opts))
}

/// A conversion of a root, in its context.
type Convert<'a> = dyn FnMut(&Rc<Context>, OperationRef) -> Result<ConversionResult, Report> + 'a;

/// [xcheck] with the conversion given.
fn xcheck_with(
    case: &Generated,
    opts: &Opts,
    bounds: &Bounds,
    rng: &mut Rng,
    convert: &mut Convert<'_>,
) -> Result<Stats, String> {
    let (context, module) = parse_module(&case.source)?;
    module
        .borrow()
        .recursively_verify()
        .map_err(|err| format!("the source does not verify: {}", describe(&err)))?;
    let mut stats = Stats::default();
    let mut expected = Vec::new();
    let externs = {
        let mut evaluator = Evaluator::new(module);
        evaluator.externs = evaluator.declarations();
        evaluator.max_depth = bounds.depth;
        evaluator.max_steps = bounds.steps;
        for (function, args) in &case.inputs {
            let outcome = evaluate_original(&mut evaluator, function, args)?;
            match &outcome {
                Some(_) => stats.evaluated += 1,
                None => stats.skipped += 1,
            }
            stats.calls += evaluator.calls;
            expected.push(outcome);
        }
        evaluator.externs.clone()
    };
    stats.operations = all_operations(module).len();

    let result = convert(&context, module)
        .map_err(|err| format!("the conversion failed: {}", describe(&err)))?;
    let converted = module.borrow().to_string();
    let show = std::env::var("TRANSFORM_TEST_SHOW").is_ok();
    let fail = |what: String| {
        if show {
            format!("{what}\n--- converted\n{converted}")
        } else {
            what
        }
    };
    if !result.changed() {
        return Err(fail(String::from("the conversion changed nothing")));
    }
    module
        .borrow()
        .recursively_verify()
        .map_err(|err| fail(format!("the result does not verify: {}", describe(&err))))?;
    if all_operations(module)
        .iter()
        .any(|op| op.borrow().is::<UnrealizedConversionCast>())
    {
        return Err(fail(String::from("a cast remains")));
    }
    if opts.mode == Mode::Full {
        let converter = mix_converter(&context, false);
        if let Some(held) = converted_type_held(module, &converter) {
            return Err(fail(format!("a converted type remains: {held}")));
        }
    }
    check_model(&context, module, &case.model).map_err(fail)?;
    check_symbol_uses(module).map_err(fail)?;
    compare_all(module, case, &externs, &expected, bounds).map_err(fail)?;

    let again = convert(&context, module)
        .map_err(|err| fail(format!("the second conversion failed: {}", describe(&err))))?;
    let second = module.borrow().to_string();
    if again.changed() || second != converted {
        return Err(fail(format!(
            "the second conversion changed the module (changed() = {})",
            again.changed()
        )));
    }

    // The printed module parses to itself.
    {
        let (_context, reparsed) = parse_module(&converted)
            .map_err(|err| fail(format!("the converted text does not parse: {err}")))?;
        reparsed.borrow().recursively_verify().map_err(|err| {
            fail(format!("the reparsed result does not verify: {}", describe(&err)))
        })?;
        // Value names are numbered anew by the parser: compare with the numbers taken out.
        let text = reparsed.borrow().to_string();
        if strip_numbers(&text) != strip_numbers(&converted) {
            if show {
                println!("--- reparsed\n{text}");
            }
            return Err(fail(String::from("the converted text does not round-trip")));
        }
        compare_all(reparsed, case, &BTreeMap::new(), &expected, bounds)
            .map_err(|err| fail(format!("REPARSED: {err}")))
            .or_else(|err| if externs.is_empty() { Err(err) } else { Ok(()) })?;
    }

    if opts.passes {
        let mut ran = Vec::new();
        for (name, make) in passes::shuffled(rng) {
            ran.push(name);
            run_pass(&context, module, make()).map_err(|err| {
                fail(format!("PASS: after the conversion, {ran:?} failed: {}", describe(&err)))
            })?;
            compare_all(module, case, &externs, &expected, bounds).map_err(|err| {
                let tail = if show {
                    format!("\n--- after the passes\n{}", module.borrow())
                } else {
                    String::new()
                };
                fail(format!("PASS: after the conversion and {ran:?}: {err}{tail}"))
            })?;
        }
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------------------------

/// Runs `body` on a thread with a large stack: the evaluator recurses once per call.
fn on_a_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap_or_else(|err| std::panic::resume_unwind(err));
}

fn run(profile: fn() -> Profile, mode: Mode, default_seeds: &'static [u64], default_cases: usize) {
    on_a_large_stack(move || {
        let profile = profile();
        let bounds = Bounds {
            depth: if profile.chain { 2000 } else { 64 },
            steps: 400_000,
        };
        let mut findings = Vec::new();
        let mut total = Stats::default();
        let mut ran = 0;
        for seed in seeds(default_seeds) {
            let mut rng = Rng(seed);
            for case in 0..cases(default_cases) {
                let case_seed = if case == 0 { seed } else { rng.next() };
                let mut case_rng = Rng(case_seed);
                let opts = match mode {
                    Mode::Full => Opts::full(),
                    Mode::Partial => Opts::partial(&mut case_rng),
                };
                let generated = xgen::module(&mut case_rng, &profile);
                ran += 1;
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    xcheck(&generated, &opts, &bounds, &mut case_rng)
                }));
                match outcome {
                    Ok(Ok(stats)) => {
                        total.evaluated += stats.evaluated;
                        total.skipped += stats.skipped;
                        total.operations += stats.operations;
                        total.calls += stats.calls;
                    }
                    Ok(Err(finding)) => {
                        let source = if std::env::var("TRANSFORM_TEST_SHOW").is_ok() {
                            format!("\n--- source\n{}", generated.source)
                        } else {
                            String::new()
                        };
                        findings.push((case_seed, format!("{opts:?}\n{finding}{source}")));
                    }
                    Err(panic) => {
                        let message = panic
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default();
                        findings.push((case_seed, format!("{opts:?}\nPANIC: {message}")));
                    }
                }
            }
        }
        println!(
            "{ran} modules under {mode:?}: {} operations, {} calls evaluated, {} inputs compared, \
             {} skipped at a bound; {} failed",
            total.operations,
            total.calls,
            total.evaluated,
            total.skipped,
            findings.len()
        );
        let mut kinds = BTreeMap::<String, (usize, u64)>::new();
        for (seed, finding) in &findings {
            let line = finding.lines().nth(1).unwrap_or("").chars().take(160).collect::<String>();
            let entry = kinds.entry(line).or_insert((0, *seed));
            entry.0 += 1;
        }
        for (line, (count, seed)) in &kinds {
            println!("  {count} x (TRANSFORM_TEST_SEED={seed}): {line}");
        }
        if let Some((case_seed, finding)) = findings.first() {
            panic!(
                "{} of {ran} cases failed; the first (TRANSFORM_TEST_SEED={case_seed} \
                 TRANSFORM_TEST_CASES=1 TRANSFORM_TEST_SHOW=1 runs it alone):\n{finding}",
                findings.len()
            );
        }
    });
}

/// The number of cases per seed of a test: `dev` in a profile with debug assertions, where a
/// module of the large profiles takes seconds to check, else `optimized`. `TRANSFORM_TEST_CASES`
/// overrides both, so a large profile runs in the dev profile too when it is set.
const fn dev_or(dev: usize, optimized: usize) -> usize {
    if cfg!(debug_assertions) {
        dev
    } else {
        optimized
    }
}

#[test]
fn extended_small_modules_convert_fully() {
    run(Profile::small, Mode::Full, &[0x5e3_0001, 0x5e3_0002], dev_or(8, 150));
}

#[test]
fn extended_small_modules_convert_partially() {
    run(Profile::small, Mode::Partial, &[0x5e3_0011, 0x5e3_0012], dev_or(8, 150));
}

#[test]
fn extended_wide_signatures_convert_fully() {
    run(Profile::wide_signatures, Mode::Full, &[0x5e3_0021], dev_or(8, 150));
}

#[test]
fn extended_wide_signatures_convert_partially() {
    run(Profile::wide_signatures, Mode::Partial, &[0x5e3_0031], dev_or(8, 150));
}

#[test]
fn extended_many_functions_convert_fully() {
    run(Profile::many_functions, Mode::Full, &[0x5e3_0041], dev_or(0, 4));
}

#[test]
fn extended_many_functions_convert_partially() {
    run(Profile::many_functions, Mode::Partial, &[0x5e3_0051], dev_or(0, 4));
}

#[test]
fn extended_long_bodies_convert_fully() {
    run(Profile::long_bodies, Mode::Full, &[0x5e3_0061], dev_or(0, 4));
}

#[test]
fn extended_long_bodies_convert_partially() {
    run(Profile::long_bodies, Mode::Partial, &[0x5e3_0071], dev_or(0, 4));
}

#[test]
fn extended_chains_of_calls_convert_fully() {
    run(Profile::chain, Mode::Full, &[0x5e3_0081], dev_or(0, 4));
}

/// The generator gives what the tests claim.
#[test]
fn extended_generator_covers_every_feature() {
    let mut rng = Rng(0x5e3_0091);
    let mut text = String::new();
    for _ in 0..60 {
        text.push_str(&xgen::module(&mut rng, &Profile::small()).source);
    }
    for feature in [
        "builtin.call @f",
        "builtin.call @a",
        "builtin.call @aa",
        "builtin.call_indirect",
        "builtin.function_ref @a",
        "builtin.function_alias",
        "locals(",
        "builtin.load_local",
        "effects(",
        "effects()",
        "allocate(result",
        "write(arg",
        "cf.cond_br",
        "cf.switch",
        "#builtin.i32<2> -> ",
        "mem.store",
        "mem.load",
        ": i8",
        "ptr<i64>",
        "arith.sub %v0, ",
        "builtin.ret_imm",
    ] {
        assert!(text.contains(feature), "no `{feature}` in 60 modules");
    }
}
