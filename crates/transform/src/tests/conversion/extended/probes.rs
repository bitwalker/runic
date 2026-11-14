//! Fixed cases, one per question, each asserting what holds. Each prints `PROBE <name>:
//! <outcome>`. Two measure instead of asserting, and are ignored (run them with `--ignored` in an
//! optimized profile).

extern crate std;

use alloc::{
    collections::BTreeMap,
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cell::Cell;
use std::println;

use runic_core::{
    Context, Operation, OperationRef, Report, Symbol,
    conversion::{
        ConversionConfig, ConversionPatternSet, ConversionResult, ConversionTarget,
        DynamicLegalityResult, TypeConversion, TypeConverter, add_builtin_legality_by_type,
        apply_full_conversion, apply_partial_conversion, populate_builtin_conversion_patterns,
    },
    dialects::builtin::{Component, Function, FunctionAlias, Module, World},
    parse::{self, ParserConfig},
    testing::text::describe,
};
use runic_dialect_arith::ArithDialect;
use runic_dialect_scf::ScfDialect;

use super::{
    Bounds, Opts, Rng, mix_convert, mix_converter, mix_patterns, mix_target,
    xeval::{Evaluator, Val},
    xgen::{self, Profile},
};
use crate::tests::conversion::{Mode, all_operations, parse_module, patterns};

fn module(body: &str) -> (Rc<Context>, OperationRef) {
    let source = format!("builtin.module public @m {{\n{body}\n}};");
    let (context, module) = parse_module(&source).unwrap_or_else(|err| panic!("{err}\n{source}"));
    module
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("the source does not verify: {}\n{source}", describe(&err)));
    (context, module)
}

fn outcome(name: &str, result: &Result<ConversionResult, Report>, root: OperationRef) -> String {
    let text = match result {
        Ok(result) => format!("Ok(changed = {})", result.changed()),
        Err(err) => format!("Err: {}", describe(err)),
    };
    let verifies = match root.borrow().recursively_verify() {
        Ok(()) => String::from("verifies"),
        Err(err) => format!("does not verify: {}", describe(&err)),
    };
    let text = format!("{text}; afterwards the root {verifies}");
    println!("PROBE {name}: {text}");
    text
}

fn by_type_target(context: &Rc<Context>, converter: &TypeConverter) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target.add_legal_op::<Module>().add_legal_op::<FunctionAlias>();
    add_builtin_legality_by_type(&mut target, converter);
    let converter = converter.clone();
    target.add_dynamically_legal_dialect::<ArithDialect, _>(move |op| {
        DynamicLegalityResult::legal_if(converter.is_legal_op(op))
    });
    target
}

fn builtin_only(context: &Rc<Context>, converter: &TypeConverter) -> ConversionPatternSet {
    let mut set = ConversionPatternSet::new(context.clone());
    populate_builtin_conversion_patterns(&mut set, converter).unwrap();
    set
}

const IDENTITY: &str = r#"
    builtin.function public extern("C") @id(%a: i128, %n: i64) -> i128 locals(i128) {
        builtin.store_local 0, %a;
        %b = builtin.load_local 0 : i128;
        builtin.ret %b : (i128);
    };
    builtin.function public extern("C") @user(%a: i128, %n: i64) -> i128 {
        %r = builtin.call @id(%a, %n) : extern("C") (i128, i64) -> i128;
        builtin.ret %r : (i128);
    };"#;

// ---------------------------------------------------------------------------------------------
// Converters that misbehave
// ---------------------------------------------------------------------------------------------

/// A converter whose answer for a type is itself not legal: `i128 -> (i128, i64)`.
#[test]
fn an_answer_that_is_not_legal_is_an_error() {
    let (context, module) = module(IDENTITY);
    let (i64, i128) = (context.i64(), context.i128());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == i128).then(|| TypeConversion::many([i128, i64])));
    let converter = converter.with_function_types();
    let started = std::time::Instant::now();
    let result = apply_full_conversion(
        module,
        by_type_target(&context, &converter),
        builtin_only(&context, &converter),
        ConversionConfig::default(),
    );
    let text = outcome("an answer that is not legal", &result, module);
    println!("PROBE an answer that is not legal: took {:?}", started.elapsed());
    assert!(text.contains("is not legal under the converter"), "{text}");
}

/// A converter that changes `never`.
#[test]
fn a_converter_that_changes_never() {
    let (context, module) = module(
        r#"
    builtin.function public extern("C") @stop(i128) -> never;
    builtin.function public extern("C") @other(%a: i128) -> i128 {
        builtin.ret %a : (i128);
    };"#,
    );
    let (i64, i128, never) = (context.i64(), context.i128(), context.never_type());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| {
        if ty == i128 {
            Some(TypeConversion::many([i64, i64]))
        } else if ty == never {
            Some(TypeConversion::one(i64))
        } else {
            None
        }
    });
    let converter = converter.with_function_types();
    let result = apply_full_conversion(
        module,
        by_type_target(&context, &converter),
        builtin_only(&context, &converter),
        ConversionConfig::default(),
    );
    let text = outcome("a converter that changes never", &result, module);
    println!("{}", module.borrow());
    // Either an error, or `never` kept.
    if result.is_ok() {
        assert!(module.borrow().to_string().contains("-> never"), "{text}");
    }
}

/// A callback that answers for function types in a way that disagrees with
/// `with_function_types`: function values become `i64`.
#[test]
fn a_callback_for_function_types_is_not_asked() {
    let (context, module) = module(
        r#"
    builtin.function public extern("C") @id(%a: i128) -> i128 {
        builtin.ret %a : (i128);
    };
    builtin.function public extern("C") @user(%a: i128) -> i128 locals((i128) -> i128) {
        %f = builtin.function_ref @id : (i128) -> i128;
        builtin.store_local 0, %f;
        %g = builtin.load_local 0 : (i128) -> i128;
        %r = builtin.call_indirect %g(%a) : extern("C") (i128) -> i128;
        builtin.ret %r : (i128);
    };"#,
    );
    let (i64, i128) = (context.i64(), context.i128());
    let asked = Rc::new(Cell::new(0));
    let mut converter = TypeConverter::new();
    {
        let asked = asked.clone();
        converter.add_conversion(move |_, ty| {
            if ty == i128 {
                Some(TypeConversion::many([i64, i64]))
            } else if ty.as_function().is_some() {
                asked.set(asked.get() + 1);
                Some(TypeConversion::one(i64))
            } else {
                None
            }
        });
    }
    let converter = converter.with_function_types();
    let result = apply_full_conversion(
        module,
        by_type_target(&context, &converter),
        builtin_only(&context, &converter),
        ConversionConfig::default(),
    );
    let text = outcome("a callback for function types", &result, module);
    println!(
        "PROBE a callback for function types: asked {} times\n{}",
        asked.get(),
        module.borrow()
    );
    assert!(result.is_ok(), "{text}");
    assert_eq!(asked.get(), 0);
}

/// Two converters built from one stateful callback: the patterns' converter splits an `i128`
/// into two, the target's and the dialect patterns' into four.
#[test]
fn two_converters_that_disagree_are_an_error() {
    let (context, module) = module(
        r#"
    builtin.function public extern("C") @sum(%a: i128, %b: i128) -> i128 {
        %s = arith.bxor %a, %b;
        builtin.ret %s : (i128);
    };"#,
    );
    let (i32, i64, i128) = (context.i32(), context.i64(), context.i128());
    let asked = Rc::new(Cell::new(0));
    let make = || {
        let asked = asked.clone();
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| {
            (ty == i128).then(|| {
                asked.set(asked.get() + 1);
                if asked.get() == 1 {
                    TypeConversion::many([i64, i64])
                } else {
                    TypeConversion::many([i32, i32, i32, i32])
                }
            })
        });
        converter.with_function_types()
    };
    let first = make();
    // Ask the first now, so that it is the one that answers `(i64, i64)`.
    assert_eq!(first.convert_type(&context, i128).len(), 2);
    let second = make();
    let opts = Opts::full();
    let mut set = builtin_only(&context, &first);
    for rule in patterns::wide_patterns(&context, &second, &opts.kinds) {
        set.push(rule);
    }
    let result = apply_full_conversion(
        module,
        by_type_target(&context, &second),
        set,
        ConversionConfig::default(),
    );
    let text = outcome("two converters that disagree", &result, module);
    assert!(result.is_err(), "{text}");
}

/// Two types that convert to the same list.
#[test]
fn two_types_with_one_answer_are_kept_apart() {
    let (context, module) = module(
        r#"
    builtin.function public extern("C") @swap(%a: i128, %b: i96) -> (i96, i128) locals(i96, i128) {
        builtin.store_local 0, %b;
        builtin.store_local 1, %a;
        cf.br ^next(%b, %a : i96, i128);
    ^next(%x: i96, %y: i128):
        %p = builtin.load_local 0 : i96;
        %q = builtin.load_local 1 : i128;
        builtin.ret %p, %y : (i96, i128);
    };
    builtin.function public extern("C") @user(%a: i128, %b: i96) -> (i96, i128) {
        %r, %s = builtin.call @swap(%a, %b) : extern("C") (i128, i96) -> (i96, i128);
        builtin.ret %r, %s : (i96, i128);
    };"#,
    );
    let (i64, i128) = (context.i64(), context.i128());
    let i96 = context.int_type(96);
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| {
        (ty == i128 || ty == i96).then(|| TypeConversion::many([i64, i64]))
    });
    let converter = converter.with_function_types();
    let result = apply_full_conversion(
        module,
        by_type_target(&context, &converter),
        builtin_only(&context, &converter),
        ConversionConfig::default(),
    );
    let text = outcome("two types with one answer", &result, module);
    assert!(result.is_ok(), "{text}");
    // (a_lo, a_hi, b_lo, b_hi) -> (b_lo, b_hi, a_lo, a_hi)
    let mut evaluator = Evaluator::new(module);
    let values = [1u128, 2, 3, 4].map(Val::Int).to_vec();
    assert_eq!(evaluator.call("user", values).unwrap(), [3u128, 4, 1, 2].map(Val::Int).to_vec());
}

/// The builtin patterns added twice, under two converters: one splits `i128`, the other drops
/// the marker. The target judges by both.
#[test]
fn two_pattern_sets_with_two_converters() {
    super::on_a_large_stack(|| {
        let mut rng = Rng(0x5e3_0b96);
        let mut failures = Vec::new();
        let mut converted = 0;
        let count = super::dev_or(30, 150);
        for case in 0..count {
            let seed = rng.next();
            let mut case_rng = Rng(seed);
            let generated = xgen::module(&mut case_rng, &Profile::small());
            let opts = Opts::full();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::xcheck_with(
                    &generated,
                    &opts,
                    &Bounds {
                        depth: 64,
                        steps: 400_000,
                    },
                    &mut case_rng,
                    &mut |context, root| {
                        let both = mix_converter(context, false);
                        let (i8, i64, i128) = (context.i8(), context.i64(), context.i128());
                        let mut wide = TypeConverter::new();
                        wide.add_conversion(move |_, ty| {
                            (ty == i128).then(|| TypeConversion::many([i64, i64]))
                        });
                        let wide = wide.with_function_types();
                        let mut marker = TypeConverter::new();
                        marker.add_conversion(move |_, ty| (ty == i8).then(TypeConversion::drop));
                        let marker = marker.with_function_types();
                        let mut set = ConversionPatternSet::new(context.clone());
                        if case % 2 == 0 {
                            populate_builtin_conversion_patterns(&mut set, &wide)?;
                            populate_builtin_conversion_patterns(&mut set, &marker)?;
                        } else {
                            populate_builtin_conversion_patterns(&mut set, &marker)?;
                            populate_builtin_conversion_patterns(&mut set, &wide)?;
                        }
                        for rule in patterns::wide_patterns(context, &wide, &opts.kinds) {
                            set.push(rule);
                        }
                        for rule in super::marker_rules(context, &marker, &opts.kinds) {
                            set.push(rule);
                        }
                        apply_full_conversion(
                            root,
                            mix_target(context, &opts, &both),
                            set,
                            ConversionConfig::default(),
                        )
                    },
                )
            }));
            match outcome {
                Ok(Ok(_)) => converted += 1,
                Ok(Err(err)) => failures.push((seed, err)),
                Err(_) => failures.push((seed, String::from("PANIC"))),
            }
        }
        let mut kinds = BTreeMap::<String, (usize, u64)>::new();
        for (seed, finding) in &failures {
            let line = finding.lines().next().unwrap_or("").chars().take(200).collect::<String>();
            let entry = kinds.entry(line).or_insert((0, *seed));
            entry.0 += 1;
        }
        println!(
            "PROBE two pattern sets with two converters: {converted} of {count} converted and \
             checked; {} failed",
            failures.len()
        );
        for (line, (count, seed)) in &kinds {
            println!("  {count} x (seed {seed}): {line}");
        }
        // Each must be an error or a correct result: never a module that verifies and computes
        // something else, and never a panic.
        for (seed, finding) in &failures {
            assert!(
                !finding.contains("MISCOMPILE")
                    && !finding.contains("PANIC")
                    && !finding.contains("MODEL"),
                "seed {seed}: {finding}"
            );
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Partial conversions of real shapes
// ---------------------------------------------------------------------------------------------

/// A target that judges builtin operations by type, except that the function `legal` is legal.
fn target_with_a_legal_function(
    context: &Rc<Context>,
    converter: &TypeConverter,
    legal: &'static str,
) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target.add_legal_op::<Module>().add_legal_op::<FunctionAlias>();
    add_builtin_legality_by_type(&mut target, converter);
    let converter = converter.clone();
    target.add_dynamically_legal_op::<Function, _>(move |op: &Operation| {
        let name = op.downcast_ref::<Function>().unwrap().name();
        DynamicLegalityResult::legal_if(name.as_str() == legal || converter.is_legal_op(op))
    });
    target
}

fn partial_with_a_legal_function(legal: &'static str, body: &str) -> (String, String) {
    let (context, module) = module(body);
    let converter = mix_converter(&context, true);
    let opts = Opts {
        mode: Mode::Partial,
        kinds: Vec::new(),
        passes: false,
    };
    let set = mix_patterns(&context, &opts, &converter).unwrap();
    let result = apply_partial_conversion(
        module,
        target_with_a_legal_function(&context, &converter, legal),
        set,
        ConversionConfig::default(),
    );
    let text = outcome(&format!("partial, @{legal} called legal"), &result, module);
    (text, module.borrow().to_string())
}

#[test]
fn a_callee_called_legal_and_its_caller_not() {
    let (text, _) = partial_with_a_legal_function("id", IDENTITY);
    assert!(text.starts_with("Err"), "{text}");
}

#[test]
fn a_caller_called_legal_and_its_callee_not() {
    let (text, _) = partial_with_a_legal_function("user", IDENTITY);
    assert!(text.starts_with("Err"), "{text}");
}

/// A function left alone whose body reads a local of a converted type, and calls nothing.
#[test]
fn a_function_left_alone_that_reads_a_converted_local() {
    let (text, after) = partial_with_a_legal_function(
        "alone",
        r#"
    builtin.function public extern("C") @alone(%a: i128) -> i128 locals(i128) {
        builtin.store_local 0, %a;
        %b = builtin.load_local 0 : i128;
        %c = arith.add %b, %a;
        builtin.ret %c : (i128);
    };"#,
    );
    println!("{after}");
    // An error that names the load, or a function that is left exactly as it was.
    if text.starts_with("Ok") {
        assert!(text.contains("verifies") && !text.contains("does not verify"), "{text}");
        assert!(after.contains("builtin.load_local 0 : i128"), "{after}");
    }
}

const TWO_MODULES: &str = r#"builtin.world {
    builtin.module public @one {
        builtin.function public extern("C") @id(%a: i128) -> i128 {
            builtin.ret %a : (i128);
        };
    };
    builtin.module public @two {
        builtin.function public extern("C") @user(%a: i128) -> i128 {
            %r = builtin.call ::@one::@id(%a) : extern("C") (i128) -> i128;
            builtin.ret %r : (i128);
        };
    };
};"#;

fn parse_world(source: &str) -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    let world = parse::parse::<World>(ParserConfig::new(context.clone()), "world", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let world = world.as_operation_ref();
    world
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("the source does not verify: {}", describe(&err)));
    (context, world)
}

fn child_named(root: OperationRef, name: &str) -> OperationRef {
    all_operations(root)
        .into_iter()
        .find(|op| {
            let op = op.borrow();
            op.as_trait::<dyn Symbol>().is_some_and(|symbol| symbol.name().as_str() == name)
        })
        .unwrap_or_else(|| panic!("no @{name}"))
}

fn world_target(context: &Rc<Context>, converter: &TypeConverter) -> ConversionTarget {
    let mut target = by_type_target(context, converter);
    target.add_legal_op::<World>().add_legal_op::<Component>();
    target
}

/// A conversion rooted at a world converts the modules of its components together.
#[test]
fn a_conversion_rooted_at_a_world() {
    let (context, world) = parse_world(TWO_MODULES);
    let converter = mix_converter(&context, false);
    let result = apply_full_conversion(
        world,
        world_target(&context, &converter),
        builtin_only(&context, &converter),
        ConversionConfig::default(),
    );
    let text = outcome("rooted at a world", &result, world);
    println!("{}", world.borrow());
    assert!(text.starts_with("Ok") && !text.contains("does not verify"), "{text}");
    let mut evaluator = Evaluator::new(world);
    assert_eq!(
        evaluator.call("user", vec![Val::Int(7), Val::Int(9)]).unwrap(),
        vec![Val::Int(7), Val::Int(9)]
    );
}

/// A conversion whose root does not hold a user of a function whose signature it changes is an
/// error, and the IR prints as before: rooted at the callee's module in a world (the caller is in
/// the other module), and rooted at the callee itself (the caller is beside it).
#[test]
fn wanted_a_conversion_that_breaks_a_user_outside_its_root_is_an_error() {
    let (context, world) = parse_world(TWO_MODULES);
    let (root, outer) = (child_named(world, "one"), world);
    let (context_of_module, module) = module(IDENTITY);
    let id = child_named(module, "id");
    for (context, root, outer) in [(&context, root, outer), (&context_of_module, id, module)] {
        let before = outer.borrow().to_string();
        let converter = mix_converter(context, false);
        let result = apply_full_conversion(
            root,
            by_type_target(context, &converter),
            builtin_only(context, &converter),
            ConversionConfig::default(),
        );
        let text = outcome("a user outside the root", &result, root);
        assert!(text.contains("outside the root of the conversion"), "{text}");
        assert_eq!(outer.borrow().to_string(), before);
        outer
            .borrow()
            .recursively_verify()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
    }
}

/// A conversion rooted at the module of the caller: its callee is not under the root.
#[test]
fn a_conversion_rooted_at_the_callers_module() {
    let (context, world) = parse_world(TWO_MODULES);
    let two = child_named(world, "two");
    let converter = mix_converter(&context, false);
    let result = apply_full_conversion(
        two,
        by_type_target(&context, &converter),
        builtin_only(&context, &converter),
        ConversionConfig::default(),
    );
    let text = outcome("rooted at the caller's module", &result, two);
    assert!(text.starts_with("Err"), "{text}");
}

// ---------------------------------------------------------------------------------------------
// `scf` operations that hold a converted type
// ---------------------------------------------------------------------------------------------

const STRUCTURED: &str = r#"
    builtin.function public extern("C") @pick(%c: i1, %a: i128, %b: i128) -> i128 {
        %x = scf.if %c then {
            %s = arith.bxor %a, %b;
            scf.yield %s : (i128);
        } else {
            scf.yield %b : (i128);
        } : (i1) -> (i128);
        %y = arith.bxor %x, %a;
        builtin.ret %y : (i128);
    };"#;

fn structured(name: &str, mode: Mode, scf: &str) -> String {
    let (context, module) = module(STRUCTURED);
    let mut expected = Vec::new();
    let inputs = [(0u128, 5u128, 9u128 << 70), (1, u128::MAX, 3), (1, 1 << 64, (1 << 64) + 1)];
    {
        let mut evaluator = Evaluator::new(module);
        for (c, a, b) in inputs {
            let args = vec![Val::Int(c), Val::Int(a), Val::Int(b)];
            expected.push(evaluator.call("pick", args).unwrap());
        }
    }
    let opts = match mode {
        Mode::Full => Opts::full(),
        Mode::Partial => Opts {
            mode,
            kinds: vec![patterns::Kind::Xor, patterns::Kind::Constant],
            passes: false,
        },
    };
    let converter = mix_converter(&context, mode == Mode::Partial);
    let mut target = mix_target(&context, &opts, &converter);
    match scf {
        "no rule" => {}
        "legal" => {
            target.add_legal_dialect::<ScfDialect>();
        }
        "by type" => {
            let converter = converter.clone();
            target.add_dynamically_legal_dialect::<ScfDialect, _>(move |op| {
                DynamicLegalityResult::legal_if(converter.is_legal_op(op))
            });
        }
        _ => unreachable!(),
    }
    let set = mix_patterns(&context, &opts, &converter).unwrap();
    let result = match mode {
        Mode::Full => apply_full_conversion(module, target, set, ConversionConfig::default()),
        Mode::Partial => apply_partial_conversion(module, target, set, ConversionConfig::default()),
    };
    let text = outcome(&format!("scf.if of i128, {mode:?}, scf {scf} ({name})"), &result, module);
    if result.is_ok() {
        println!("{}", module.borrow());
        let mut evaluator = Evaluator::new(module);
        for ((c, a, b), expected) in inputs.iter().zip(&expected) {
            let args = vec![
                Val::Int(*c),
                Val::Int(*a & u128::from(u64::MAX)),
                Val::Int(*a >> 64),
                Val::Int(*b & u128::from(u64::MAX)),
                Val::Int(*b >> 64),
            ];
            let actual = evaluator.call("pick", args).unwrap_or_else(|err| panic!("{err}"));
            let [Val::Int(low), Val::Int(high)] = actual.as_slice() else {
                panic!("odd results {actual:?}");
            };
            assert_eq!(vec![Val::Int(low | high << 64)], *expected, "MISCOMPILE of scf.if");
        }
    }
    text
}

#[test]
fn scf_holding_a_converted_type_is_never_silently_wrong() {
    let mut outcomes = Vec::new();
    for mode in [Mode::Full, Mode::Partial] {
        for scf in ["no rule", "legal", "by type"] {
            outcomes.push(structured("pick", mode, scf));
        }
    }
    // Full: an error in every case.
    for text in &outcomes[..3] {
        assert!(text.starts_with("Err"), "{text}");
    }
}

// ---------------------------------------------------------------------------------------------
// What the passes do, with no conversion at all
// ---------------------------------------------------------------------------------------------

const RECURSIVE: &str = r#"
    builtin.function public extern("C") @down(%n: i64, %c: i1) -> i64 effects() {
        %r = builtin.call @down(%n, %c) : extern("C") (i64, i1) -> i64;
        cf.cond_br %c ^a(%r : i64), ^b(%n : i64) : (i1);
    ^a(%x: i64):
        cf.br ^b(%n : i64);
    ^b(%y: i64):
        %z = arith.add %y, %n;
        builtin.ret %z : (i64);
    };"#;

/// The canonicalizer, run on a module, finds the block argument dead and with it the call of the
/// enclosing function (which declares no effects), and erases the call while the clean-up of the
/// module's region is inside that function: the call's use of the function is unlinked.
#[test]
fn wanted_canonicalizing_a_module_with_a_dead_call_of_the_enclosing_function() {
    let (context, module) = module(RECURSIVE);
    super::run_pass(&context, module, alloc::boxed::Box::<crate::Canonicalizer>::default())
        .unwrap();
    assert!(!module.borrow().to_string().contains("builtin.call @down"));
}

/// A pass whose target is `builtin.function` added to a pass manager anchored on a module with
/// implicit nesting is run on the module itself, and panics (`PassTarget::target_name` names no
/// concrete operation type, so `add_pass` never nests).
#[test]
fn a_function_pass_on_a_module_is_not_nested_implicitly() {
    use runic_core::pass::{Nesting, PassManager};
    let (context, module) = module(RECURSIVE);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut manager = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
        manager.add_pass(alloc::boxed::Box::<crate::Local2Reg>::default());
        manager.enable_verifier(true);
        manager.run(module)
    }));
    println!(
        "PROBE local2reg on a module, implicit nesting: {}",
        if outcome.is_err() { "PANIC" } else { "ran" }
    );
    assert!(outcome.is_err());
}

/// What a fix should give. Left ignored: implicit nesting is a matter of the pass manager, which
/// stage 4 takes up (`runic-opt` adds passes named on a command line to a module's manager).
#[test]
#[ignore = "wanted: a pass that targets builtin.function is nested under a module implicitly"]
fn wanted_a_function_pass_on_a_module_is_nested_implicitly() {
    use runic_core::pass::{Nesting, PassManager};
    let (context, module) = module(RECURSIVE);
    let mut manager = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
    manager.add_pass(alloc::boxed::Box::<crate::Local2Reg>::default());
    manager.enable_verifier(true);
    manager.run(module).unwrap();
}

/// A declaration under a converter whose answer is not legal: an error that names the type, at
/// once (not at the rewrite limit, with a signature grown each application).
#[test]
fn an_answer_that_is_not_legal_is_an_error_for_a_declaration() {
    let (context, module) = module(
        r#"
    builtin.function public extern("C") @far(i128) -> i128;"#,
    );
    let (i64, i128) = (context.i64(), context.i128());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == i128).then(|| TypeConversion::many([i128, i64])));
    let converter = converter.with_function_types();
    let started = std::time::Instant::now();
    let result = apply_full_conversion(
        module,
        by_type_target(&context, &converter),
        builtin_only(&context, &converter),
        ConversionConfig::default(),
    );
    let elapsed = started.elapsed();
    let text = match &result {
        Ok(_) => String::from("Ok"),
        Err(err) => format!("Err: {}", describe(err).chars().take(300).collect::<String>()),
    };
    println!("PROBE an answer that is not legal, a declaration: {text}; took {elapsed:?}");
    assert!(text.contains("is not legal under the converter"), "{text}");
    assert!(!text.contains("limit"), "{text}");
}

/// What a conversion and the verifier cost on a large module (optimized profile for numbers that
/// mean something).
#[test]
#[ignore = "a measurement: run with --ignored in an optimized profile"]
fn costs_of_conversion_and_verification() {
    super::on_a_large_stack(|| {
        for (name, profile) in [
            ("many functions", Profile::many_functions()),
            ("long bodies", Profile::long_bodies()),
        ] {
            let mut rng = Rng(0x5e3_c057);
            let generated = xgen::module(&mut rng, &profile);
            let (context, module) = parse_module(&generated.source).unwrap();
            let operations = all_operations(module).len();
            let time = |f: &mut dyn FnMut()| {
                let started = std::time::Instant::now();
                f();
                started.elapsed()
            };
            let verify_before = time(&mut || module.borrow().recursively_verify().unwrap());
            let verify_again = time(&mut || module.borrow().recursively_verify().unwrap());
            let opts = Opts::full();
            let converter = mix_converter(&context, false);
            let mut unverified = ConversionConfig::default();
            unverified.with_verify_after_conversion(false);
            let convert = time(&mut || {
                apply_full_conversion(
                    module,
                    mix_target(&context, &opts, &converter),
                    mix_patterns(&context, &opts, &converter).unwrap(),
                    unverified,
                )
                .unwrap();
            });
            let after = all_operations(module).len();
            let verify_after = time(&mut || module.borrow().recursively_verify().unwrap());
            let again = time(&mut || {
                mix_convert(&context, module, &opts).unwrap();
            });
            println!(
                "PROBE costs, {name}: {operations} operations ({after} after); verify \
                 {verify_before:?} (again {verify_again:?}); convert without verification \
                 {convert:?}; verify after {verify_after:?}; a second conversion (nothing to do, \
                 verified) {again:?}"
            );
            for (pass, make) in crate::tests::drive::PASSES {
                if pass == "cfg-to-scf" {
                    continue;
                }
                let started = std::time::Instant::now();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    use runic_core::pass::{Nesting, PassManager};
                    let mut manager = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
                    if pass == "local2reg" {
                        manager.add_nested_pass::<Function>(make());
                    } else {
                        manager.add_pass(make());
                    }
                    manager.enable_verifier(false);
                    manager.run(module)
                }));
                let unverified = started.elapsed();
                let verify = time(&mut || {
                    let _ = module.borrow().recursively_verify();
                });
                println!(
                    "PROBE costs, {name}: {pass} {unverified:?} ({}), verification after it \
                     {verify:?}",
                    match result {
                        Ok(Ok(())) => "ok",
                        Ok(Err(_)) => "error",
                        Err(_) => "panic",
                    }
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------
// The rewrite limit against a large function
// ---------------------------------------------------------------------------------------------

/// `count` loads of local 1 (an `i128`, after another `i128` local, so that its index changes),
/// summed with `arith.bxor`.
fn many_loads(count: usize) -> String {
    let mut body =
        String::from(
            "    builtin.function public extern(\"C\") @big(%a: i128) -> i128 locals(i128, i128) \
             {\n        builtin.store_local 0, %a;\n        builtin.store_local 1, %a;\n",
        );
    for index in 0..count {
        body.push_str(&format!("        %l{index} = builtin.load_local 1 : i128;\n"));
    }
    body.push_str("        builtin.ret %l0 : (i128);\n    };");
    body
}

fn convert_many_loads(count: usize) -> String {
    let (context, module) = module(&many_loads(count));
    let converter = mix_converter(&context, false);
    let result = apply_full_conversion(
        module,
        by_type_target(&context, &converter),
        builtin_only(&context, &converter),
        ConversionConfig::default(),
    );
    let text = match &result {
        Ok(result) => format!("Ok(changed = {})", result.changed()),
        Err(err) => format!("Err: {}", describe(err).chars().take(400).collect::<String>()),
    };
    println!("PROBE a function with {count} loads of a renumbered local: {text}");
    text
}

/// The references to a renumbered local are each legalized in their own turn, so the size of a
/// function does not reach the rewrite limit.
#[test]
fn wanted_a_large_function_converts_under_the_default_limit() {
    // Five thousand in an optimized profile; core's own test converts as many in any profile.
    assert!(convert_many_loads(1100).starts_with("Ok"));
    if !cfg!(debug_assertions) {
        assert!(convert_many_loads(5000).starts_with("Ok"));
    }
}

/// How the time of a conversion grows with the size of one function (optimized profile); the
/// sizes are `TRANSFORM_TEST_SIZES`, separated by commas, if set.
#[test]
#[ignore = "a measurement: run with --ignored in an optimized profile"]
fn conversion_time_against_function_size() {
    super::on_a_large_stack(|| {
        let sizes: Vec<usize> = match std::env::var("TRANSFORM_TEST_SIZES") {
            Ok(sizes) => sizes.split(',').map(|size| size.trim().parse().unwrap()).collect(),
            Err(_) => vec![500, 1000, 2000, 4000],
        };
        for steps in sizes {
            let profile = Profile {
                front: (1, 2),
                fuel: (0, 1),
                back: (0, 1),
                shapes: (0, 1),
                externs: (0, 1),
                steps: (steps, steps + 1),
                constructs: steps / 60,
                ..Profile::long_bodies()
            };
            let mut rng = Rng(0x5e3_51ce);
            let generated = xgen::module(&mut rng, &profile);
            let (context, module) = parse_module(&generated.source).unwrap();
            let operations = all_operations(module).len();
            let blocks = generated.source.matches("\n    ^b").count();
            let opts = Opts::full();
            let converter = mix_converter(&context, false);
            let mut unverified = ConversionConfig::default();
            unverified.with_verify_after_conversion(false);
            let started = std::time::Instant::now();
            apply_full_conversion(
                module,
                mix_target(&context, &opts, &converter),
                mix_patterns(&context, &opts, &converter).unwrap(),
                unverified,
            )
            .unwrap();
            let elapsed = started.elapsed();
            println!(
                "PROBE size: {operations} operations in {blocks} blocks convert in {elapsed:?} \
                 ({:.1} us per operation)",
                elapsed.as_secs_f64() * 1e6 / operations as f64
            );
        }
    });
}
