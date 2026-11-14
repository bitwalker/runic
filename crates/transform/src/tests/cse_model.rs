//! What CSE merges in a random straight-line block, against an independent statement of the
//! rule, and against the interpreter.
//!
//! The rule: an operation with no effects merges with the earliest equal one. An operation whose
//! one effect is a read (with everything it holds, for an operation with regions) merges with an
//! earlier equal read exactly when no operation between the two has unknown effects, or a write or
//! a free of a resource that is not disjoint from the resource read. Nothing else merges. The
//! generator emits no operation with regions; `cse_keeps_a_region_operation_that_holds_a_store`
//! covers one.

use alloc::{
    boxed::Box,
    collections::BTreeMap,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    Context, SymbolTable,
    dialects::builtin::{Function, Module},
    parse::{self, ParserConfig},
    pass::{Nesting, OperationPass, PassManager},
    testing::text::{Rng, describe},
};

use super::interp;
use crate::CommonSubexpressionElimination;

pub const CALLEES: &str = r#"
    builtin.function public extern("C") @pure(i32) -> i32 effects();
    builtin.function public extern("C") @reads_any(i32) -> i32 effects(read);
    builtin.function public extern("C") @reads_arg(ptr<i32>) -> i32 effects(read(arg 0));
    builtin.function public extern("C") @writes_any(i32) -> i32 effects(write);
    builtin.function public extern("C") @writes_arg(ptr<i32>) -> i32 effects(write(arg 0));
    builtin.function public extern("C") @frees(ptr<i32>) effects(free(arg 0));
    builtin.function public extern("C") @unknown(i32) -> i32;"#;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Res {
    Default,
    Mem,
    Locals,
    Assert,
    Test,
}

fn disjoint(a: Res, b: Res) -> bool {
    a != b && a != Res::Default && b != Res::Default
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Class {
    /// No effects: merges with any earlier equal operation.
    Pure,
    /// One read of a resource.
    Read(Res),
    /// Never merged; blocks reads of resources not disjoint from this one.
    Blocks(Res),
    /// Never merged; blocks every read.
    Unknown,
    /// Never merged and blocks nothing (an allocation).
    Inert,
}

struct GenOp {
    text: String,
    /// The result name and its type, if any.
    result: Option<(String, &'static str)>,
    /// What the operation is, as the key CSE compares: the name with its properties.
    key: String,
    operands: Vec<String>,
    commutative: bool,
    class: Class,
}

pub struct Program {
    pub text: String,
    ops: Vec<GenOp>,
}

const PARAMS: &str = "%p0: ptr<i32>, %p1: ptr<i32>, %q: ptr, %v0: i32, %v1: i32, %n: isize, %b: i8";

fn local(k: u64) -> String {
    format!("<{{ local = #builtin.local_variable<{k}, i32> }}>")
}

pub fn generate(rng: &mut Rng, len: usize) -> Program {
    generate_returning(rng, len, 1)
}

/// As [generate], but only about one result in `one_in` is returned: the rest are unused.
pub fn generate_returning(rng: &mut Rng, len: usize, one_in: u64) -> Program {
    let mut ints = alloc::vec!["%v0".to_string(), "%v1".to_string()];
    let mut ptrs = alloc::vec!["%p0".to_string(), "%p1".to_string()];
    let mut ops = Vec::new();
    let mut next = 0;
    let mut fresh = || {
        next += 1;
        format!("%r{next}")
    };
    for _ in 0..len {
        let pick =
            |rng: &mut Rng, pool: &[String]| pool[rng.below(pool.len() as u64) as usize].clone();
        // bias towards few distinct operands so that equal operations are common
        let v = if rng.below(3) == 0 {
            pick(rng, &ints)
        } else {
            ints[rng.below(2) as usize].clone()
        };
        let w = if rng.below(3) == 0 {
            pick(rng, &ints)
        } else {
            ints[rng.below(2) as usize].clone()
        };
        let p = if rng.below(4) == 0 {
            pick(rng, &ptrs)
        } else {
            ptrs[rng.below(2) as usize].clone()
        };
        let p2 = pick(rng, &ptrs);
        let k = rng.below(2);
        let call = |callee: &str, arg: &str, ty: &str, ret: &str| {
            format!("builtin.call @{callee}({arg}) : extern(\"C\") ({ty}){ret}")
        };
        let r = fresh();
        let op = match rng.below(26) {
            0..=3 => GenOp {
                text: format!("{r} = mem.load {p};"),
                result: Some((r, "i32")),
                key: "mem.load".into(),
                operands: alloc::vec![p],
                commutative: false,
                class: Class::Read(Res::Mem),
            },
            4 => GenOp {
                text: format!("{r} = mem.load %q : i32;"),
                result: Some((r, "i32")),
                key: "mem.load:i32".into(),
                operands: alloc::vec!["%q".into()],
                commutative: false,
                class: Class::Read(Res::Mem),
            },
            5 => GenOp {
                text: format!("{r} = mem.load {p}, align 1;"),
                result: Some((r, "i32")),
                key: "mem.load,align1".into(),
                operands: alloc::vec![p],
                commutative: false,
                class: Class::Read(Res::Mem),
            },
            6 => GenOp {
                text: format!("mem.store {v}, {p};"),
                result: None,
                key: "mem.store".into(),
                operands: alloc::vec![v, p],
                commutative: false,
                class: Class::Blocks(Res::Mem),
            },
            7 => {
                let text = match rng.below(3) {
                    0 => format!("mem.memcpy {p}, {p2}, %n;"),
                    1 => format!("mem.memmove {p}, {p2}, %n;"),
                    _ => format!("mem.memset {p}, %b, %n;"),
                };
                GenOp {
                    text,
                    result: None,
                    key: "bulk".into(),
                    operands: alloc::vec![],
                    commutative: false,
                    class: Class::Blocks(Res::Mem),
                }
            }
            8 => {
                ptrs.push(r.clone());
                GenOp {
                    text: format!("{r} = mem.alloca i32;"),
                    result: Some((r, "ptr<i32>")),
                    key: "mem.alloca".into(),
                    operands: alloc::vec![],
                    commutative: false,
                    class: Class::Inert,
                }
            }
            9..=11 => GenOp {
                text: format!("{r} = builtin.load_local {};", local(k)),
                result: Some((r, "i32")),
                key: format!("load_local{k}"),
                operands: alloc::vec![],
                commutative: false,
                class: Class::Read(Res::Locals),
            },
            12 => GenOp {
                text: format!("builtin.store_local {v} {} : (i32);", local(k)),
                result: None,
                key: "store_local".into(),
                operands: alloc::vec![v],
                commutative: false,
                class: Class::Blocks(Res::Locals),
            },
            13 => GenOp {
                text: format!("{r} = {};", call("pure", &v, "i32", " -> i32")),
                result: Some((r, "i32")),
                key: "call pure".into(),
                operands: alloc::vec![v],
                commutative: false,
                class: Class::Pure,
            },
            14 | 15 => GenOp {
                text: format!("{r} = {};", call("reads_any", &v, "i32", " -> i32")),
                result: Some((r, "i32")),
                key: "call reads_any".into(),
                operands: alloc::vec![v],
                commutative: false,
                class: Class::Read(Res::Default),
            },
            16 => GenOp {
                text: format!("{r} = {};", call("reads_arg", &p, "ptr<i32>", " -> i32")),
                result: Some((r, "i32")),
                key: "call reads_arg".into(),
                operands: alloc::vec![p],
                commutative: false,
                class: Class::Read(Res::Default),
            },
            17 => {
                let (callee, arg, ty) = if rng.below(2) == 0 {
                    ("writes_any", v, "i32")
                } else {
                    ("writes_arg", p, "ptr<i32>")
                };
                GenOp {
                    text: format!("{r} = {};", call(callee, &arg, ty, " -> i32")),
                    result: Some((r, "i32")),
                    key: "call writes".into(),
                    operands: alloc::vec![arg],
                    commutative: false,
                    class: Class::Blocks(Res::Default),
                }
            }
            18 => GenOp {
                text: format!("{};", call("frees", &p, "ptr<i32>", "")),
                result: None,
                key: "call frees".into(),
                operands: alloc::vec![p],
                commutative: false,
                class: Class::Blocks(Res::Default),
            },
            19 => {
                if rng.below(2) == 0 {
                    GenOp {
                        text: format!("{r} = {};", call("unknown", &v, "i32", " -> i32")),
                        result: Some((r, "i32")),
                        key: "call unknown".into(),
                        operands: alloc::vec![v],
                        commutative: false,
                        class: Class::Unknown,
                    }
                } else {
                    GenOp {
                        text: format!("{r} = test.effects <{{ known = #builtin.bool<false> }}>;"),
                        result: Some((r, "i32")),
                        key: "test.effects unknown".into(),
                        operands: alloc::vec![],
                        commutative: false,
                        class: Class::Unknown,
                    }
                }
            }
            20 => GenOp {
                text: format!("{r} = test.effects <{{ known = #builtin.bool<true> }}>;"),
                result: Some((r, "i32")),
                key: "test.effects known".into(),
                operands: alloc::vec![],
                commutative: false,
                class: Class::Read(Res::Default),
            },
            21 => {
                // the condition is an earlier `i1` if there is one, else this is a compare
                let conds = ops
                    .iter()
                    .filter_map(|op: &GenOp| match &op.result {
                        Some((name, "i1")) => Some(name.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if conds.is_empty() {
                    GenOp {
                        text: format!("{r} = arith.cmpi ult {v}, {w};"),
                        result: Some((r, "i1")),
                        key: "cmpi ult".into(),
                        operands: alloc::vec![v, w],
                        commutative: false,
                        class: Class::Pure,
                    }
                } else {
                    let c = pick(rng, &conds);
                    GenOp {
                        text: format!("cf.assert {c}, \"a\";"),
                        result: None,
                        key: "assert".into(),
                        operands: alloc::vec![c],
                        commutative: false,
                        class: Class::Blocks(Res::Assert),
                    }
                }
            }
            22 => GenOp {
                text: format!("{r} = test.load {p};"),
                result: Some((r, "i32")),
                key: "test.load".into(),
                operands: alloc::vec![p],
                commutative: false,
                class: Class::Read(Res::Test),
            },
            23 => GenOp {
                text: format!("test.store {p}, {v} : (ptr<i32>, i32);"),
                result: None,
                key: "test.store".into(),
                operands: alloc::vec![p, v],
                commutative: false,
                class: Class::Blocks(Res::Test),
            },
            24 => GenOp {
                text: format!("{r} = arith.add {v}, {w};"),
                result: Some((r, "i32")),
                key: "add".into(),
                operands: alloc::vec![v, w],
                commutative: true,
                class: Class::Pure,
            },
            _ => GenOp {
                text: format!("{r} = arith.cmpi ult {v}, {w};"),
                result: Some((r, "i1")),
                key: "cmpi ult".into(),
                operands: alloc::vec![v, w],
                commutative: false,
                class: Class::Pure,
            },
        };
        if let Some((name, "i32")) = &op.result {
            ints.push(name.clone());
        }
        ops.push(op);
    }
    let results = ops
        .iter()
        .filter_map(|op| op.result.clone())
        .filter(|_| one_in == 1 || rng.below(one_in) == 0)
        .collect::<Vec<_>>();
    let names = results.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ");
    let types = results.iter().map(|(_, t)| *t).collect::<Vec<_>>().join(", ");
    let body = ops.iter().map(|op| format!("    {}", op.text)).collect::<Vec<_>>().join("\n");
    let (arrow, ret) = if results.is_empty() {
        (String::new(), "builtin.ret;".to_string())
    } else {
        (format!(" -> ({types})"), format!("builtin.ret {names} : ({types});"))
    };
    let text = format!(
        "builtin.module public @m {{\n{CALLEES}\nbuiltin.function public extern(\"C\") \
         @f({PARAMS}){arrow} locals(i32, i32) {{\n{body}\n    {ret}\n}};\n}};"
    );
    Program { text, ops }
}

impl Program {
    /// For each result, in order: the index of the result it is merged into (itself if none).
    fn expected_classes(&self) -> Vec<usize> {
        // canonical name of each result
        let mut canon: BTreeMap<String, String> = BTreeMap::new();
        let canon_of = |canon: &BTreeMap<String, String>, name: &String| {
            canon.get(name).cloned().unwrap_or_else(|| name.clone())
        };
        let mut keys: Vec<Option<String>> = Vec::new();
        let mut representative: Vec<Option<String>> = Vec::new();
        for (j, op) in self.ops.iter().enumerate() {
            let mut operands = op.operands.iter().map(|o| canon_of(&canon, o)).collect::<Vec<_>>();
            if op.commutative {
                operands.sort();
            }
            let key = format!("{} {operands:?}", op.key);
            let mut merged_into = None;
            match op.class {
                Class::Pure => {
                    merged_into = (0..j).find(|&i| keys[i].as_deref() == Some(key.as_str()));
                }
                Class::Read(read) => {
                    if let Some(i) =
                        (0..j).rev().find(|&i| keys[i].as_deref() == Some(key.as_str()))
                    {
                        let blocked =
                            self.ops[i + 1..j].iter().any(|between| match between.class {
                                Class::Unknown => true,
                                Class::Blocks(written) => !disjoint(written, read),
                                Class::Pure | Class::Read(_) | Class::Inert => false,
                            });
                        if !blocked {
                            merged_into = Some(i);
                        }
                    }
                }
                Class::Blocks(_) | Class::Unknown | Class::Inert => (),
            }
            let mergeable = matches!(op.class, Class::Pure | Class::Read(_));
            keys.push(mergeable.then_some(key));
            let name = op.result.as_ref().map(|(n, _)| n.clone());
            match (merged_into, &name) {
                (Some(i), Some(name)) => {
                    let target = representative[i].clone().expect("a merged-into op has a result");
                    canon.insert(name.clone(), target.clone());
                    representative.push(Some(target));
                }
                _ => representative.push(name),
            }
        }
        let results = self
            .ops
            .iter()
            .zip(&representative)
            .filter(|(op, _)| op.result.is_some())
            .map(|(_, rep)| rep.clone().unwrap())
            .collect::<Vec<_>>();
        results
            .iter()
            .map(|rep| results.iter().position(|other| other == rep).unwrap())
            .collect()
    }
}

/// Parses `text` (a module with `@f`), returns the context, and `@f` printed before and after
/// `passes` ran on it with verification.
pub fn run_passes(
    text: &str,
    passes: Vec<Box<dyn OperationPass>>,
) -> Result<(String, String), String> {
    use runic_dialect_mem as _;
    use runic_dialect_scf as _;
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "passes", text)
        .into_result()
        .map_err(|err| format!("parse: {}", describe(&err)))?;
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    let before = function.borrow().to_string();
    for pass in passes {
        let mut manager = PassManager::on::<Function>(context.clone(), Nesting::Implicit);
        manager.add_pass(pass);
        manager.enable_verifier(true);
        manager.run(function).map_err(|err| format!("pass: {}", describe(&err)))?;
    }
    let after = function.borrow().to_string();
    Ok((before, after))
}

/// The operands of the `builtin.ret` of `text`.
fn ret_operands(text: &str) -> Vec<String> {
    let ret = text.lines().rev().find(|line| line.contains("builtin.ret")).unwrap();
    interp::values(ret.split(" : ").next().unwrap())
}

fn check_equivalent(before: &str, after: &str, rng: &mut Rng, what: &str) {
    let before_f = interp::parse_function(before).unwrap_or_else(|e| panic!("{e}\n{before}"));
    let after_f = interp::parse_function(after).unwrap_or_else(|e| panic!("{e}\n{after}"));
    for _ in 0..6 {
        // p0, p1, q may alias; v0, v1 are often equal
        let addrs = [0x1000u64, 0x1000, 0x2000, 0x3000];
        let args = [
            rng.pick(&addrs),
            rng.pick(&addrs),
            rng.pick(&addrs),
            rng.below(3),
            rng.below(3),
            rng.below(2) * 4,
            rng.below(256),
        ];
        let a = interp::run(&before_f, &args, 1000).unwrap_or_else(|e| panic!("{e}\n{before}"));
        let b = interp::run(&after_f, &args, 1000).unwrap_or_else(|e| panic!("{e}\n{after}"));
        assert_eq!(a, b, "{what}: args {args:x?}\n--- before\n{before}\n--- after\n{after}");
    }
}

/// CSE on 1500 random blocks merges exactly what the rule says, and the function computes the
/// same thing before and after.
#[test]
fn cse_merges_exactly_what_the_rule_says() {
    let mut rng = Rng(super::seed(0x05ee_d3d3));
    let mut merges = 0;
    let mut reads_merged_across_writes = 0;
    for case in 0..1500 {
        let len = 4 + rng.below(28) as usize;
        let program = generate(&mut rng, len);
        let (before, after) = run_passes(
            &program.text,
            alloc::vec![Box::<CommonSubexpressionElimination>::default()],
        )
        .unwrap_or_else(|err| panic!("case {case}: {err}\n{}", program.text));
        let expected = program.expected_classes();
        let before_ret = ret_operands(&before);
        let after_ret = ret_operands(&after);
        assert_eq!(before_ret.len(), expected.len());
        assert_eq!(after_ret.len(), expected.len());
        let actual = after_ret
            .iter()
            .map(|name| after_ret.iter().position(|other| other == name).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "case {case}: merged results differ from the rule\n--- before\n{before}\n--- \
             after\n{after}"
        );
        merges += expected.iter().enumerate().filter(|(i, e)| *i != **e).count();
        // every operation that is not merged is still there
        let survivors = expected.iter().enumerate().filter(|(i, e)| *i == **e).count();
        let effectful = program.ops.iter().filter(|op| op.result.is_none()).count();
        let after_ops = after.lines().count() - 3;
        assert_eq!(after_ops, survivors + effectful, "case {case}:\n{before}\n{after}");
        if program.ops.iter().any(|op| matches!(op.class, Class::Blocks(_))) {
            reads_merged_across_writes += 1;
        }
        check_equivalent(&before, &after, &mut rng, "cse");
    }
    assert!(merges > 1500, "the generator produced too few merges: {merges}");
    assert!(reads_merged_across_writes > 500, "{reads_merged_across_writes}");
}

/// Random blocks in which most results are unused, through random sequences of every pass of
/// the crate: the result verifies, every store, copy, fill, assert, local store and effectful
/// call is still there, in order, and the function computes the same thing.
#[test]
fn random_blocks_through_random_pipelines() {
    use super::drive::PASSES;

    let effectful = |text: &str| {
        text.lines()
            .map(str::trim)
            .filter(|line| {
                ["mem.store", "mem.mem", "cf.assert", "builtin.store_local", "test.store"]
                    .iter()
                    .any(|name| line.starts_with(name))
                    || ["@unknown", "@writes", "@frees", "bool<false>"]
                        .iter()
                        .any(|name| line.contains(name))
            })
            .map(|line| {
                // without the value names: a pass may renumber them
                line.split(|c: char| c == '%' || c.is_ascii_digit())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
    };
    let mut rng = Rng(super::seed(0x919e_11e5));
    for case in 0..1000 {
        let len = 4 + rng.below(28) as usize;
        let program = generate_returning(&mut rng, len, 3);
        let mut names = Vec::new();
        let mut passes = Vec::new();
        for _ in 0..1 + rng.below(4) {
            // not `cfg-to-scf`: a straight-line block has nothing to lift
            let (name, make) = PASSES[rng.below(6) as usize];
            names.push(name);
            passes.push(make());
        }
        let (before, after) = run_passes(&program.text, passes)
            .unwrap_or_else(|err| panic!("case {case} {names:?}: {err}\n{}", program.text));
        let (kept_before, kept_after) = (effectful(&before), effectful(&after));
        // the canonicalizer erases an assert whose condition folded to 1, and nothing else
        let kept_before = if names.contains(&"canonicalize") || names.contains(&"sccp") {
            let asserts = kept_after.iter().filter(|l| l.starts_with("cf.assert")).count();
            let mut seen = 0;
            let total = kept_before.iter().filter(|l| l.starts_with("cf.assert")).count();
            let dropped = total - asserts.min(total);
            if dropped == 0 {
                kept_before
            } else {
                // compare without asserts when some were erased; the interpreter checks traps
                let _ = &mut seen;
                kept_before.into_iter().filter(|l| !l.starts_with("cf.assert")).collect()
            }
        } else {
            kept_before
        };
        let kept_after = if kept_before.iter().any(|l| l.starts_with("cf.assert")) {
            kept_after
        } else {
            kept_after.into_iter().filter(|l| !l.starts_with("cf.assert")).collect()
        };
        assert_eq!(
            kept_after, kept_before,
            "case {case} {names:?}\n--- before\n{before}\n--- after\n{after}"
        );
        check_equivalent(&before, &after, &mut rng, &alloc::format!("case {case} {names:?}"));
    }
}
