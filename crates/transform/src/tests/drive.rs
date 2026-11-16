//! Every `mem` operation, the local operations, asserts and calls through every pass
//! of this crate (and `LiftControlFlowToSCF`), in the positions the earlier parts' tests use.
//!
//! Checked for each (fixture, pass): the result verifies; no store, copy, fill, assert, local
//! store or effectful call is removed, duplicated or (for the passes that do not restructure
//! control flow) reordered; and, where the interpreter understands both texts, the function
//! computes the same thing before and after.

use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};

use runic_dialect_scf::transforms::LiftControlFlowToSCF;
use runic_ir::{pass::OperationPass, testing::text::Rng};

use super::{cse_model::run_passes, interp};
use crate::{
    Canonicalizer, CommonSubexpressionElimination, ControlFlowSink, DeadCodeElimination, Local2Reg,
    SinkOperandDefs, SparseConditionalConstantPropagation,
};

type MakePass = fn() -> Box<dyn OperationPass>;

pub const PASSES: [(&str, MakePass); 8] = [
    ("dce", || Box::<DeadCodeElimination>::default()),
    ("cse", || Box::<CommonSubexpressionElimination>::default()),
    ("control-flow-sink", || Box::<ControlFlowSink>::default()),
    ("sink-operand-defs", || Box::<SinkOperandDefs>::default()),
    ("sccp", || Box::<SparseConditionalConstantPropagation>::default()),
    ("canonicalize", || Box::<Canonicalizer>::default()),
    ("cfg-to-scf", || Box::<LiftControlFlowToSCF>::default()),
    ("local2reg", || Box::<Local2Reg>::default()),
];

const CALLEES: &str = r#"
    builtin.function public extern("C") @pure(i32) -> i32 effects();
    builtin.function public extern("C") @reads_any(i32) -> i32 effects(read);
    builtin.function public extern("C") @writes_any(i32) -> i32 effects(write);
    builtin.function public extern("C") @unknown(i32) -> i32;"#;

pub fn module(body: &str) -> String {
    format!(
        "builtin.module public @m {{\n{CALLEES}\nbuiltin.function public extern(\"C\") @f(%c: i1, \
         %d: i1, %a: i32, %b: i32, %p: ptr<i32>, %q: ptr<i32>, %n: isize, %z: i8) -> i32 \
         locals(i32, i32) {{\n{body}\n}};\n}};"
    )
}

/// The fixtures: (name, body).
pub const FIXTURES: [(&str, &str); 12] = [
    (
        "every operation, unused",
        r#"
    %x1 = mem.load %p;
    %x2 = mem.alloca i64;
    %x3 = mem.gep %p, %n;
    %x4 = mem.sizeof struct<i32, i8>;
    %x5 = mem.alignof i64;
    %x6 = mem.offsetof struct<i8, i32>, [1];
    %x7 = mem.ptrtoint %p;
    %x8 = mem.inttoptr %n : ptr<i32>;
    %x9 = mem.ptrcast %p : ptr<i8>;
    %x10 = builtin.load_local 0 : i32;
    mem.store %a, %p;
    mem.memcpy %p, %q, %n;
    mem.memmove %p, %q, %n;
    mem.memset %p, %z, %n;
    builtin.store_local 1, %a;
    cf.assert %c, "one";
    builtin.ret %a : (i32);"#,
    ),
    (
        "every operation, duplicated",
        r#"
    %x1 = mem.load %p;
    %y1 = mem.load %p;
    %x2 = mem.alloca i32;
    %y2 = mem.alloca i32;
    mem.store %a, %x2;
    mem.store %a, %y2;
    mem.store %a, %p;
    mem.store %a, %p;
    mem.memcpy %p, %q, %n;
    mem.memcpy %p, %q, %n;
    mem.memmove %p, %q, %n;
    mem.memmove %p, %q, %n;
    mem.memset %p, %z, %n;
    mem.memset %p, %z, %n;
    %x3 = builtin.load_local 0 : i32;
    %y3 = builtin.load_local 0 : i32;
    builtin.store_local 0, %x3;
    builtin.store_local 0, %y3;
    cf.assert %c, "one";
    cf.assert %c, "one";
    %s1 = arith.add %x1, %y1;
    %s2 = arith.add %x3, %y3;
    %s = arith.add %s1, %s2;
    builtin.ret %s : (i32);"#,
    ),
    (
        "a load before a store, used after it",
        r#"
    %x = mem.load %p;
    mem.store %a, %p;
    %y = mem.load %p;
    mem.store %b, %q;
    %w = mem.load %p;
    %s = arith.add %x, %y;
    %t = arith.add %s, %w;
    builtin.ret %t : (i32);"#,
    ),
    (
        "a load defined before a store and used only in a region after it",
        r#"
    %x = mem.load %p;
    %l = builtin.load_local 0 : i32;
    mem.store %a, %p;
    builtin.store_local 0, %b;
    %r = scf.if %c then {
        %s = arith.add %x, %l;
        scf.yield %s : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %r : (i32);"#,
    ),
    (
        "in both regions of an if",
        r#"
    %r = scf.if %c then {
        %x = mem.load %p;
        mem.store %a, %p;
        %y = mem.load %p;
        mem.memcpy %q, %p, %n;
        cf.assert %d, "then";
        %s = arith.add %x, %y;
        scf.yield %s : (i32);
    } else {
        %x2 = mem.load %p;
        mem.store %a, %p;
        %y2 = mem.load %p;
        mem.memset %q, %z, %n;
        cf.assert %d, "else";
        %s2 = arith.add %x2, %y2;
        scf.yield %s2 : (i32);
    } : (i1) -> (i32);
    %after = mem.load %p;
    %t = arith.add %r, %after;
    builtin.ret %t : (i32);"#,
    ),
    (
        "loads around an if that stores",
        r#"
    %x = mem.load %p;
    %l = builtin.load_local 0 : i32;
    scf.if %c then {
        mem.store %a, %q;
        builtin.store_local 0, %a;
        scf.yield : ();
    } else {
        scf.yield : ();
    } : (i1) -> ();
    %y = mem.load %p;
    %m = builtin.load_local 0 : i32;
    %s = arith.add %x, %y;
    %t = arith.add %l, %m;
    %u = arith.add %s, %t;
    builtin.ret %u : (i32);"#,
    ),
    (
        "loads around an if that only asserts",
        r#"
    %x = mem.load %p;
    scf.if %c then {
        cf.assert %d, "inside";
        scf.yield : ();
    } else {
        scf.yield : ();
    } : (i1) -> ();
    %y = mem.load %p;
    %s = arith.add %x, %y;
    builtin.ret %s : (i32);"#,
    ),
    (
        "in the body of a loop",
        r#"
    %zero = arith.constant 0 : i32;
    %before = mem.load %p;
    %count = scf.while %zero before {
    ^head(%i: i32):
        %h = mem.load %p;
        %continue = arith.cmpi ult %i, %h;
        scf.condition %continue, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %x = mem.load %p;
        %inv = mem.load %q;
        mem.store %j, %p;
        %y = mem.load %p;
        mem.memmove %q, %p, %n;
        cf.assert %c, "body";
        %l = builtin.load_local 0 : i32;
        builtin.store_local 0, %y;
        %one = arith.constant 1 : i32;
        %next = arith.add %j, %one;
        scf.yield %next : (i32);
    } : (i32) -> i32;
    %after = mem.load %p;
    %s = arith.add %before, %after;
    builtin.ret %s : (i32);"#,
    ),
    (
        "after a branch, in each successor and at the join",
        r#"
    %x = mem.load %p;
    mem.store %a, %p;
    cf.cond_br %c ^t, ^f : (i1);
^t:
    %xt = mem.load %p;
    mem.store %b, %p;
    cf.assert %d, "then";
    builtin.store_local 0, %xt;
    cf.br ^m(%xt : i32);
^f:
    %xf = mem.load %p;
    mem.memset %p, %z, %n;
    builtin.store_local 0, %xf;
    cf.br ^m(%xf : i32);
^m(%r: i32):
    %xm = mem.load %p;
    %l = builtin.load_local 0 : i32;
    %s = arith.add %r, %xm;
    %t = arith.add %s, %x;
    %u = arith.add %t, %l;
    builtin.ret %u : (i32);"#,
    ),
    (
        "bulk operations of constant lengths",
        r#"
    %zero = arith.constant 0 : isize;
    %four = arith.constant 4 : isize;
    %wrap = arith.constant 4294967296 : isize;
    mem.memcpy %p, %q, %four;
    mem.memmove %p, %q, %wrap;
    mem.memset %p, %z, %wrap;
    mem.memset %q, %z, %four;
    mem.memcpy %p, %q, %zero;
    mem.memmove %p, %q, %zero;
    mem.memset %p, %z, %zero;
    %x = mem.load %p;
    builtin.ret %x : (i32);"#,
    ),
    (
        "calls between loads",
        r#"
    %x = mem.load %p;
    %l = builtin.load_local 0 : i32;
    %k = builtin.call @pure(%a) : extern("C") (i32) -> i32;
    %y = mem.load %p;
    %m = builtin.load_local 0 : i32;
    %u = builtin.call @unknown(%a) : extern("C") (i32) -> i32;
    %w = mem.load %p;
    %o = builtin.load_local 0 : i32;
    %ra = builtin.call @reads_any(%a) : extern("C") (i32) -> i32;
    mem.store %a, %q;
    %rb = builtin.call @reads_any(%a) : extern("C") (i32) -> i32;
    %s1 = arith.add %x, %y;
    %s2 = arith.add %s1, %w;
    %s3 = arith.add %l, %m;
    %s4 = arith.add %s3, %o;
    %s5 = arith.add %ra, %rb;
    %s6 = arith.add %s2, %s4;
    %s7 = arith.add %s6, %s5;
    %s8 = arith.add %s7, %k;
    %s9 = arith.add %s8, %u;
    builtin.ret %s9 : (i32);"#,
    ),
    (
        "casts and a local round trip",
        r#"
    %i = mem.ptrtoint %p;
    %p2 = mem.inttoptr %i : ptr<i32>;
    %c1 = mem.ptrcast %p : ptr<i8>;
    %c2 = mem.ptrcast %c1 : ptr<i32>;
    %x = mem.load %p2;
    mem.store %a, %c2;
    %y = mem.load %p;
    builtin.store_local 0, %y;
    %l = builtin.load_local 0 : i32;
    builtin.store_local 1, %x;
    %s = arith.add %x, %l;
    builtin.ret %s : (i32);"#,
    ),
];

/// The lines of `text` that are operations no pass may remove, duplicate or reorder, reduced to
/// what identifies them.
fn kept_operations(text: &str, pass: &str) -> Vec<String> {
    let mut kept = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let name = ["mem.store", "mem.memcpy", "mem.memmove", "mem.memset", "cf.assert"]
            .into_iter()
            .find(|name| line.starts_with(name));
        if let Some(name) = name {
            let message = line.find('"').map(|at| &line[at..]).unwrap_or("");
            kept.push(format!("{name} {message}"));
        } else if line.starts_with("builtin.store_local") && pass != "local2reg" {
            kept.push("builtin.store_local".to_string());
        } else if line.contains("builtin.call @unknown") || line.contains("builtin.call @writes") {
            kept.push("call".to_string());
        }
    }
    kept
}

#[test]
fn every_pass_keeps_every_effectful_operation_in_every_position() {
    let mut rng = Rng(super::seed(0x000d_417e));
    let mut failures = Vec::new();
    let mut interpreted = 0;
    for (fixture, body) in FIXTURES {
        for (pass, make) in PASSES {
            let (before, after) = match run_passes(&module(body), alloc::vec![make()]) {
                Ok(texts) => texts,
                Err(err) => {
                    failures.push(format!("{pass} on '{fixture}' failed: {err}"));
                    continue;
                }
            };
            let mut expected = kept_operations(&before, pass);
            let mut actual = kept_operations(&after, pass);
            if fixture == "bulk operations of constant lengths" && pass == "canonicalize" {
                // the three of length 0 go, and only those
                assert_eq!(expected.len(), 7);
                assert_eq!(actual.len(), 4, "{after}");
                assert!(after.contains("arith.constant 4294967296 : isize"), "{after}");
                expected.truncate(4);
            }
            if pass == "cfg-to-scf" || pass == "canonicalize" {
                expected.sort();
                actual.sort();
            }
            if expected != actual {
                failures.push(format!(
                    "{pass} on '{fixture}': {expected:?} became {actual:?}\n--- \
                     before\n{before}\n--- after\n{after}"
                ));
            }
            if let (Ok(f), Ok(g)) =
                (interp::parse_function(&before), interp::parse_function(&after))
            {
                for _ in 0..8 {
                    let addrs = [0x1000u64, 0x1000, 0x2000];
                    let args = [
                        rng.below(2),
                        rng.below(2),
                        rng.below(4),
                        rng.below(4),
                        rng.pick(&addrs),
                        rng.pick(&addrs),
                        rng.below(2) * 4,
                        rng.below(256),
                    ];
                    match (interp::run(&f, &args, 100), interp::run(&g, &args, 100)) {
                        (Ok(x), Ok(y)) => {
                            interpreted += 1;
                            if x != y {
                                failures.push(format!(
                                    "{pass} on '{fixture}' with {args:x?}: {x:?} became \
                                     {y:?}\n--- before\n{before}\n--- after\n{after}"
                                ));
                                break;
                            }
                        }
                        _ => break,
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n\n"));
    assert!(interpreted > 300, "{interpreted}");
}

/// All passes in sequence, twice, on every fixture: every effectful operation is kept, and the
/// function computes the same thing where the interpreter understands it.
#[test]
fn the_whole_pipeline_keeps_every_effectful_operation() {
    let mut rng = Rng(super::seed(0x919e));
    for (fixture, body) in FIXTURES {
        let mut passes = Vec::new();
        for _ in 0..2 {
            for (pass, make) in PASSES {
                if pass != "local2reg" {
                    passes.push(make());
                }
            }
        }
        let (before, after) =
            run_passes(&module(body), passes).unwrap_or_else(|err| panic!("'{fixture}': {err}"));
        let mut expected = kept_operations(&before, "");
        let mut actual = kept_operations(&after, "");
        if fixture == "bulk operations of constant lengths" {
            expected.truncate(4);
        }
        expected.sort();
        actual.sort();
        assert_eq!(actual, expected, "'{fixture}':\n--- before\n{before}\n--- after\n{after}");
        if let (Ok(f), Ok(g)) = (interp::parse_function(&before), interp::parse_function(&after)) {
            for _ in 0..8 {
                let addrs = [0x1000u64, 0x1000, 0x2000];
                let args = [
                    rng.below(2),
                    rng.below(2),
                    rng.below(4),
                    rng.below(4),
                    rng.pick(&addrs),
                    rng.pick(&addrs),
                    rng.below(2) * 4,
                    rng.below(256),
                ];
                if let (Ok(x), Ok(y)) = (interp::run(&f, &args, 100), interp::run(&g, &args, 100)) {
                    assert_eq!(x, y, "'{fixture}' {args:x?}:\n{before}\n{after}");
                }
            }
        }
    }
}

/// What CSE does with the fixtures that pin a merging decision.
#[test]
fn cse_merges_loads_across_an_if_that_asserts_but_not_one_that_stores() {
    let cse = || -> Vec<Box<dyn OperationPass>> {
        alloc::vec![Box::<CommonSubexpressionElimination>::default()]
    };
    let body = |name: &str| {
        let (_, body) = FIXTURES.iter().find(|(fixture, _)| *fixture == name).unwrap();
        module(body)
    };
    let (_, after) = run_passes(&body("loads around an if that stores"), cse()).unwrap();
    assert_eq!(after.matches("mem.load").count(), 2, "{after}");
    assert_eq!(after.matches("builtin.load_local").count(), 2, "{after}");
    let (_, after) = run_passes(&body("loads around an if that only asserts"), cse()).unwrap();
    assert_eq!(after.matches("mem.load").count(), 1, "{after}");
    assert_eq!(after.matches("cf.assert").count(), 1, "{after}");
    // in a loop body: the unused loads go; the load after the store stays, as do the load in
    // the head and the loads before and after the loop, which are not merged
    let (_, after) = run_passes(&body("in the body of a loop"), cse()).unwrap();
    assert_eq!(after.matches("mem.load").count(), 4, "{after}");
    let (_, after) = run_passes(&body("calls between loads"), cse()).unwrap();
    // %x and %y merge across the pure call; %w does not merge across the unknown call
    assert_eq!(after.matches("mem.load").count(), 2, "{after}");
    assert_eq!(after.matches("builtin.load_local").count(), 2, "{after}");
    assert_eq!(after.matches("@reads_any").count(), 2, "{after}");
}
