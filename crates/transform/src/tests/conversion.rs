//! Whole modules converted by type, with the builtin conversion patterns and the patterns a
//! dialect would own, checked by an evaluator before and after.
//!
//! Two families of conversion:
//!
//! - `i128` becomes `(i64, i64)`, low half first. The test owns the patterns `arith` would own for
//!   it: `arith.constant`, `arith.add` (with a carry), `arith.band`, `arith.bor`, `arith.bxor`,
//!   `arith.cmpi eq`, and `builtin.ret_imm` (a constant and a return).
//! - `isize` becomes `i32`. One pattern for any operation retypes `arith.constant`, `add`, `sub`,
//!   `mul`, `band`, `bor` and `bxor` in place; truncation commutes with each of them, so the
//!   converted program computes the low 32 bits of what the original computed.
//!
//! Functions, calls, function references, returns, branches and locals are converted by
//! [populate_builtin_conversion_patterns] in every run, under [add_builtin_legality_by_type].
//!
//! - In a full conversion the `arith` dialect is judged by type, so every operation over a
//!   converted type must be converted, and the converter has no materializations: a cast that
//!   remains is an error.
//! - In a partial conversion some of the test's `arith` kinds get neither a rule nor a pattern,
//!   under the default unknown-operation policy: they are offered, stay as they are, and read
//!   values the materializations build with real `arith` operations (for `i128`: `zext`, `shl`,
//!   `bor` to join, `trunc` and `shr` to split; for `isize`: `sext_or_trunc` either way). The kinds
//!   that keep a pattern are judged by type with a rule of their own.
//!
//! After every conversion that succeeds: the module verifies; no cast remains; in a full
//! conversion no operation holds a type the converter changes (its own types and its operands');
//! the declared effects of every function are on the new indices of the parts of what they were
//! on; every function computes, on the same inputs, what the original computed (an `i128` read
//! from its halves; an `isize` compared in its low 32 bits); and converting the result again
//! changes nothing. The random partial conversions must also leave, in a large share of their
//! cases, an operation over the converted type that reads values a materialization built.
//!
//! - `eval`: the evaluator of a module.
//! - `patterns`: the converters, the targets and the patterns the test owns.
//! - `cases`: modules written by hand, one per feature, whose converted text is pinned.
//! - `random`: random modules, in both families and both modes.
//! - `extended`: random modules of a richer shape (a type that vanishes, memory, external
//!   functions, recursion, `cf.switch`, large modules), checked against a model of what each
//!   function becomes and run through every pass of the crate after the conversion; and fixed
//!   probes of what holds.
//!
//! [populate_builtin_conversion_patterns]: runic_core::conversion::populate_builtin_conversion_patterns
//! [add_builtin_legality_by_type]: runic_core::conversion::add_builtin_legality_by_type

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    Context, Operation, OperationRef, Report, Type, WalkResult,
    conversion::{ConversionResult, TypeConverter},
    dialects::builtin::{
        Function, Module, UnrealizedConversionCast, attributes::MemoryEffectDescriptor,
    },
    parse::{self, ParserConfig},
    testing::text::describe,
};

use self::{
    eval::{Evaluator, Val},
    patterns::Kind,
};

mod cases;
mod eval;
mod extended;
mod patterns;
mod random;

/// What a type becomes.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Family {
    /// `i128` becomes `(i64, i64)`.
    Wide,
    /// `isize` becomes `i32`.
    Size,
}

impl Family {
    /// The type the converter changes.
    fn converted(self, context: &Context) -> Type {
        match self {
            Self::Wide => context.i128(),
            Self::Size => context.isize(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Mode {
    Full,
    Partial,
}

/// One conversion: its family, its mode, and the `arith` kinds that have a pattern and a rule
/// (in a full conversion, every kind of the family).
#[derive(Debug, Clone)]
struct Setup {
    family: Family,
    mode: Mode,
    kinds: Vec<Kind>,
}

impl Setup {
    fn full(family: Family) -> Self {
        Self {
            family,
            mode: Mode::Full,
            kinds: Kind::of(family).to_vec(),
        }
    }
}

/// Parses `text`, a `builtin.module`, in a fresh context, and verifies it.
fn parse_module(text: &str) -> Result<(Rc<Context>, OperationRef), String> {
    let context = Rc::new(Context::default());
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "conversion", text)
        .into_result()
        .map_err(|err| format!("the module does not parse: {}", describe(&err)))?;
    Ok((context, module.as_operation_ref()))
}

/// Converts `module` as `setup` says.
fn convert(
    context: &Rc<Context>,
    module: OperationRef,
    setup: &Setup,
) -> Result<ConversionResult, Report> {
    patterns::convert(context, module, setup)
}

/// A call to evaluate: the function's name and one value per parameter of its original
/// signature.
type Input = (String, Vec<u128>);

/// The low `width` bits of `value`.
fn mask(value: u128, width: u32) -> u128 {
    if width >= 128 {
        value
    } else {
        value & ((1u128 << width) - 1)
    }
}

/// The arguments of the converted function for the original `args`, of the original types
/// `types`: an `i128` as its halves, an `isize` as its low 32 bits.
fn converted_args(context: &Context, family: Family, types: &[Type], args: &[u128]) -> Vec<Val> {
    let converted = family.converted(context);
    let mut flat = Vec::new();
    for (ty, value) in types.iter().zip(args) {
        match family {
            Family::Wide if *ty == converted => {
                flat.push(Val::Int(mask(*value, 64)));
                flat.push(Val::Int(*value >> 64));
            }
            Family::Size if *ty == converted => flat.push(Val::Int(mask(*value, 32))),
            _ => flat.push(Val::Int(*value)),
        }
    }
    flat
}

/// Whether the converted function's `results` are what the original gave (`original`, of the
/// original result types `types`): an `i128` is read from its two halves, and an `isize` is
/// compared in its low 32 bits.
fn compare_results(
    context: &Context,
    family: Family,
    types: &[Type],
    original: &[Val],
    results: &[Val],
) -> Result<(), String> {
    let converted = family.converted(context);
    let int = |value: &Val| match value {
        Val::Int(value) => Ok(*value),
        Val::Func(name) => Err(format!("a function value @{name} where an integer is returned")),
    };
    let mut results = results.iter();
    let mut next = || results.next().ok_or_else(|| String::from("too few results"));
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    for (ty, value) in types.iter().zip(original) {
        match family {
            Family::Wide if *ty == converted => {
                let low = int(next()?)?;
                let high = int(next()?)?;
                expected.push(int(value)?);
                actual.push(low | high << 64);
            }
            Family::Size if *ty == converted => {
                expected.push(mask(int(value)?, 32));
                actual.push(int(next()?)?);
            }
            _ => {
                expected.push(int(value)?);
                actual.push(int(next()?)?);
            }
        }
    }
    if next().is_ok() {
        return Err(String::from("too many results"));
    }
    if expected != actual {
        return Err(format!("MISCOMPILE: gave {actual:#x?}, expected {expected:#x?}"));
    }
    Ok(())
}

/// Every operation under `root`, `root` included.
fn all_operations(root: OperationRef) -> Vec<OperationRef> {
    let mut ops = Vec::new();
    root.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
    ops
}

/// The first place in `module` that holds a type `converter` changes: an operation's own types
/// (results, block arguments, attributes, at any depth) or its operands' types.
fn converted_type_held(module: OperationRef, converter: &TypeConverter) -> Option<String> {
    for op in all_operations(module) {
        let op = op.borrow();
        let context = op.context_rc();
        let mut found = None;
        let _ = op.walk_types(&mut |site, ty| {
            if converter.is_legal_type(&context, ty) {
                WalkResult::Continue(())
            } else {
                found = Some(format!("'{}' holds `{ty}` at {site:?}", op.name()));
                WalkResult::Break(Report::msg("a converted type"))
            }
        });
        if found.is_some() {
            return found;
        }
        for (index, operand) in op.operands().iter().enumerate() {
            let ty = operand.borrow().ty();
            if !converter.is_legal_type(&context, ty) {
                return Some(format!("operand {index} of '{}' has type `{ty}`", op.name()));
            }
        }
    }
    None
}

/// The declared effects of each function of `module`, by name, with its parameter and result
/// types: `None` for effects that are not known.
type Effects = Vec<(String, Vec<Type>, Vec<Type>, Option<Vec<MemoryEffectDescriptor>>)>;

fn effects_of(module: OperationRef) -> Effects {
    let mut effects = Vec::new();
    for op in all_operations(module) {
        let op = op.borrow();
        let Some(function) = op.downcast_ref::<Function>() else {
            continue;
        };
        let signature = function.get_signature();
        effects.push((
            function.name().as_str().to_string(),
            signature.params().iter().map(|param| param.ty).collect(),
            signature.results().iter().map(|result| result.ty).collect(),
            function.get_memory_effects().known().map(<[_]>::to_vec),
        ));
    }
    effects
}

/// Checks the effects of the converted `module` against `before`, the effects of the original: an
/// effect on old parameter or result `k` is on each new index of `k`'s parts, in order (an `i128`
/// has two parts, every other type one), and an effect on no value is kept.
fn check_effects(
    context: &Context,
    family: Family,
    before: &Effects,
    module: OperationRef,
) -> Result<(), String> {
    let parts = |ty: &Type| {
        if family == Family::Wide && *ty == context.i128() {
            2
        } else {
            1
        }
    };
    let starts = |types: &[Type]| {
        let mut starts = Vec::new();
        let mut next = 0u32;
        for ty in types {
            starts.push((next, parts(ty)));
            next += parts(ty);
        }
        starts
    };
    let after = effects_of(module);
    for ((name, params, results, effects), (_, _, _, converted)) in before.iter().zip(&after) {
        let expected = effects.as_ref().map(|effects| {
            let (params, results) = (starts(params), starts(results));
            let mut expected = Vec::<MemoryEffectDescriptor>::new();
            for effect in effects {
                let (on, starts) = match (effect.argument, effect.result) {
                    (Some(index), _) => (index, &params),
                    (None, Some(index)) => (index, &results),
                    (None, None) => {
                        expected.push(effect.clone());
                        continue;
                    }
                };
                let (start, count) = starts[on as usize];
                for new in start..start + count {
                    let mut part = effect.clone();
                    if effect.argument.is_some() {
                        part.argument = Some(new);
                    } else {
                        part.result = Some(new);
                    }
                    if !expected.contains(&part) {
                        expected.push(part);
                    }
                }
            }
            expected
        });
        if expected != *converted {
            return Err(format!(
                "the effects of @{name} are {converted:?}, expected {expected:?} (before: \
                 {effects:?})"
            ));
        }
    }
    Ok(())
}

/// What a successful [check] saw besides the converted text.
struct Checked {
    text: String,
    /// In a partial conversion: some operation still holds the converted type.
    unconverted: bool,
    /// A materialization was built: the module holds an operation that only a materialization
    /// builds (`arith.shl` or `arith.trunc` for `i128`, `arith.sext_or_trunc` for `isize`).
    materialized: bool,
}

/// Converts the module `source` as `setup` says and checks the result (see the module
/// documentation), evaluating each of `inputs` before and after. Returns the converted text, or
/// what went wrong. A conversion that fails is reported as `Err` too: every module the tests make
/// must convert.
fn check(source: &str, setup: &Setup, inputs: &[Input]) -> Result<String, String> {
    check_counting(source, setup, inputs).map(|checked| checked.text)
}

/// [check], with what it saw.
fn check_counting(source: &str, setup: &Setup, inputs: &[Input]) -> Result<Checked, String> {
    let (context, module) = parse_module(source)?;
    module
        .borrow()
        .recursively_verify()
        .map_err(|err| format!("the source does not verify: {}", describe(&err)))?;
    let effects = effects_of(module);

    // What the original computes, and the original signature of each function evaluated.
    let mut expected = Vec::new();
    {
        let mut evaluator = Evaluator::new(module);
        for (function, args) in inputs {
            let (params, results) = evaluator.signature(function)?;
            let args = args.iter().map(|value| Val::Int(*value)).collect::<Vec<_>>();
            let outcome = evaluator
                .call(function, args)
                .map_err(|err| format!("the source does not evaluate @{function}: {err}"))?;
            expected.push((params, results, outcome));
        }
    }

    let result = convert(&context, module, setup)
        .map_err(|err| format!("the conversion failed: {}", describe(&err)))?;
    let converted = module.borrow().to_string();
    let fail = |what: String| format!("{what}\n--- converted\n{converted}");
    if inputs.is_empty() || !result.changed() {
        // Every module the tests make holds a converted type.
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
    let held = converted_type_held(module, &patterns::converter(&context, setup));
    if setup.mode == Mode::Full
        && let Some(held) = held.as_ref()
    {
        return Err(fail(format!("a converted type remains: {held}")));
    }
    check_effects(&context, setup.family, &effects, module).map_err(fail)?;
    {
        let mut evaluator = Evaluator::new(module);
        for ((function, args), (params, results, outcome)) in inputs.iter().zip(&expected) {
            let flat = converted_args(&context, setup.family, params, args);
            let actual = evaluator
                .call(function, flat)
                .map_err(|err| fail(format!("the result does not evaluate @{function}: {err}")))?;
            compare_results(&context, setup.family, results, outcome, &actual).map_err(|err| {
                fail(format!("@{function}({args:#x?}): {err}\n--- source\n{source}"))
            })?;
        }
    }

    let again = convert(&context, module, setup)
        .map_err(|err| fail(format!("the second conversion failed: {}", describe(&err))))?;
    let second = module.borrow().to_string();
    if again.changed() || second != converted {
        return Err(fail(format!(
            "the second conversion changed the module (changed() = {})\n--- after it\n{second}",
            again.changed()
        )));
    }
    let materialized = match setup.family {
        Family::Wide => converted.contains("arith.shl") || converted.contains("arith.trunc"),
        Family::Size => converted.contains("arith.sext_or_trunc"),
    };
    Ok(Checked {
        unconverted: held.is_some(),
        materialized,
        text: converted,
    })
}
