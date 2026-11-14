//! `Local2Reg` on random functions, against an independent statement of its rule and against
//! the interpreter.
//!
//! The rule: a local that an operation other than its loads and stores names is left alone.
//! Otherwise, a local with exactly one load and exactly one store, in one block, the store first,
//! with no call and no region operation between them, loses both (the load's uses take the stored
//! value); a local with stores and no load loses its stores; every other local is left alone.

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    format,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::testing::text::Rng;

use super::{cse_model::run_passes, interp};
use crate::Local2Reg;

const CALLEES: &str = r#"
    builtin.function public extern("C") @pure(i32) -> i32 effects();
    builtin.function public extern("C") @unknown(i32) -> i32;"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Plain,
    Call,
    Region,
}

#[derive(Default)]
struct Uses {
    /// (block path, position) of each store and load of a local
    stores: BTreeMap<u64, Vec<(String, usize)>>,
    loads: BTreeMap<u64, Vec<(String, usize)>>,
    pinned: BTreeSet<u64>,
    /// the kind of each item of each block, by block path
    kinds: BTreeMap<String, Vec<Kind>>,
}

struct Gen<'a> {
    rng: &'a mut Rng,
    locals: u64,
    next: u32,
    uses: Uses,
}

fn local(k: u64) -> String {
    format!("<{{ local = #builtin.local_variable<{k}, i32> }}>")
}

impl Gen<'_> {
    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("%r{}", self.next)
    }

    /// Emits the items of one block, at `indent`, into `out`; `ints` are the values in scope.
    fn items(
        &mut self,
        path: &str,
        depth: usize,
        count: usize,
        ints: &mut Vec<String>,
        out: &mut Vec<String>,
    ) {
        let pad = "    ".repeat(depth + 1);
        let mut kinds = Vec::new();
        for _ in 0..count {
            let pos = kinds.len();
            let v = ints[self.rng.below(ints.len() as u64) as usize].clone();
            let w = ints[self.rng.below(ints.len() as u64) as usize].clone();
            let k = self.rng.below(self.locals);
            let choice = self.rng.below(100);
            let kind = match choice {
                0..=29 => {
                    out.push(format!("{pad}builtin.store_local {v} {} : (i32);", local(k)));
                    self.uses.stores.entry(k).or_default().push((path.to_string(), pos));
                    Kind::Plain
                }
                30..=59 => {
                    let r = self.fresh();
                    out.push(format!("{pad}{r} = builtin.load_local {};", local(k)));
                    self.uses.loads.entry(k).or_default().push((path.to_string(), pos));
                    ints.push(r);
                    Kind::Plain
                }
                60..=69 => {
                    let r = self.fresh();
                    out.push(format!("{pad}{r} = arith.add {v}, {w};"));
                    ints.push(r);
                    Kind::Plain
                }
                70..=77 => {
                    let r = self.fresh();
                    let callee = if self.rng.below(2) == 0 {
                        "pure"
                    } else {
                        "unknown"
                    };
                    out.push(format!(
                        "{pad}{r} = builtin.call @{callee}({v}) : extern(\"C\") (i32) -> i32;"
                    ));
                    ints.push(r);
                    Kind::Call
                }
                78..=80 => {
                    out.push(format!("{pad}test.touch_local {};", local(k)));
                    self.uses.pinned.insert(k);
                    Kind::Plain
                }
                81..=83 => {
                    out.push(format!("{pad}mem.store {v}, %p;"));
                    Kind::Plain
                }
                84..=86 => {
                    // a store, sometimes something between, then a load of the same local
                    out.push(format!("{pad}builtin.store_local {v} {} : (i32);", local(k)));
                    self.uses.stores.entry(k).or_default().push((path.to_string(), pos));
                    kinds.push(Kind::Plain);
                    match self.rng.below(4) {
                        0 => {
                            let r = self.fresh();
                            out.push(format!(
                                "{pad}{r} = builtin.call @pure({v}) : extern(\"C\") (i32) -> i32;"
                            ));
                            kinds.push(Kind::Call);
                        }
                        1 => {
                            out.push(format!("{pad}mem.store {w}, %p;"));
                            kinds.push(Kind::Plain);
                        }
                        _ => (),
                    }
                    let r = self.fresh();
                    out.push(format!("{pad}{r} = builtin.load_local {};", local(k)));
                    self.uses.loads.entry(k).or_default().push((path.to_string(), kinds.len()));
                    ints.push(r);
                    Kind::Plain
                }
                87..=90 => {
                    // observe a value: an unknown call records its argument in the trace
                    let r = self.fresh();
                    out.push(format!(
                        "{pad}{r} = builtin.call @unknown({v}) : extern(\"C\") (i32) -> i32;"
                    ));
                    Kind::Call
                }
                _ if depth < 2 => {
                    let r = self.fresh();
                    out.push(format!("{pad}{r} = scf.if %c then {{"));
                    for (arm, tag) in [("t", "} else {"), ("e", "")] {
                        let mut inner = ints.clone();
                        let n = self.rng.below(4) as usize;
                        let inner_path = format!("{path}/{pos}{arm}");
                        self.items(&inner_path, depth + 1, n, &mut inner, out);
                        let y = inner[self.rng.below(inner.len() as u64) as usize].clone();
                        out.push(format!("{pad}    scf.yield {y} : (i32);"));
                        if !tag.is_empty() {
                            out.push(format!("{pad}{tag}"));
                        }
                    }
                    out.push(format!("{pad}}} : (i1) -> (i32);"));
                    ints.push(r);
                    Kind::Region
                }
                _ => {
                    let r = self.fresh();
                    out.push(format!("{pad}{r} = arith.add {v}, {v};"));
                    ints.push(r);
                    Kind::Plain
                }
            };
            kinds.push(kind);
        }
        self.uses.kinds.insert(path.to_string(), kinds);
    }
}

struct Program {
    text: String,
    locals: u64,
    uses: Uses,
}

fn generate(rng: &mut Rng) -> Program {
    let locals = 1 + rng.below(12);
    let blocks = 1 + rng.below(4) as usize;
    let mut generator = Gen {
        rng,
        locals,
        next: 0,
        uses: Uses::default(),
    };
    let mut out = Vec::new();
    for block in 0..blocks {
        if block > 0 {
            out.push(format!("^b{block}:"));
        }
        let mut ints = alloc::vec!["%a".to_string(), "%b".to_string()];
        let count = 1 + generator.rng.below(8) as usize;
        generator.items(&format!("b{block}"), 0, count, &mut ints, &mut out);
        let v = ints[generator.rng.below(ints.len() as u64) as usize].clone();
        if block + 1 == blocks {
            // observe, then return
            out.push(format!("    builtin.ret {v} : (i32);"));
        } else {
            let later = |rng: &mut Rng| block as u64 + 1 + rng.below((blocks - block - 1) as u64);
            match generator.rng.below(10) {
                0..=4 => {
                    let t = later(generator.rng);
                    out.push(format!("    cf.br ^b{t};"));
                }
                5..=7 => {
                    let (t, e) = (later(generator.rng), later(generator.rng));
                    out.push(format!("    cf.cond_br %c ^b{t}, ^b{e} : (i1);"));
                }
                _ => {
                    // a back edge to a block that is not the entry, or a self loop
                    let back = 1 + generator.rng.below(block.max(1) as u64);
                    let back = if block == 0 {
                        later(generator.rng)
                    } else {
                        back
                    };
                    let t = later(generator.rng);
                    out.push(format!("    cf.cond_br %d ^b{back}, ^b{t} : (i1);"));
                }
            }
        }
    }
    let local_types = (0..locals).map(|_| "i32").collect::<Vec<_>>().join(", ");
    let text = format!(
        "builtin.module public @m {{\n{CALLEES}\nbuiltin.function public extern(\"C\") @f(%a: \
         i32, %b: i32, %c: i1, %d: i1, %p: ptr<i32>) -> i32 locals({local_types}) {{\n{}\n}};\n}};",
        out.join("\n")
    );
    Program {
        text,
        locals,
        uses: generator.uses,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fate {
    Kept,
    Promoted,
    DeadStoresErased,
}

impl Program {
    fn expected(&self, k: u64) -> Fate {
        let none = Vec::new();
        let stores = self.uses.stores.get(&k).unwrap_or(&none);
        let loads = self.uses.loads.get(&k).unwrap_or(&none);
        if self.uses.pinned.contains(&k) {
            return Fate::Kept;
        }
        if loads.is_empty() {
            return if stores.is_empty() {
                Fate::Kept
            } else {
                Fate::DeadStoresErased
            };
        }
        if loads.len() != 1 || stores.len() != 1 {
            return Fate::Kept;
        }
        let (store_block, store_at) = &stores[0];
        let (load_block, load_at) = &loads[0];
        if store_block != load_block || store_at >= load_at {
            return Fate::Kept;
        }
        let kinds = &self.uses.kinds[store_block];
        if kinds[store_at + 1..*load_at].iter().any(|kind| *kind != Kind::Plain) {
            return Fate::Kept;
        }
        Fate::Promoted
    }
}

fn count(text: &str, op: &str, k: u64) -> usize {
    let needle = format!("local_variable<{k}, i32>");
    text.lines().filter(|line| line.contains(op) && line.contains(&needle)).count()
}

/// `Local2Reg` on 2000 random functions: each local meets the fate the rule gives it, the result
/// verifies, and the function computes the same thing before and after.
#[test]
fn local2reg_promotes_exactly_what_the_rule_says() {
    let mut rng = Rng(super::seed(0x10ca_124e));
    let mut fates = BTreeMap::<&str, usize>::new();
    for case in 0..2000 {
        let program = generate(&mut rng);
        let (before, after) = run_passes(&program.text, alloc::vec![Box::<Local2Reg>::default()])
            .unwrap_or_else(|err| panic!("case {case}: {err}\n{}", program.text));
        for k in 0..program.locals {
            let fate = program.expected(k);
            let (stores_before, loads_before) = (
                count(&before, "builtin.store_local", k),
                count(&before, "builtin.load_local", k),
            );
            let (stores_after, loads_after) =
                (count(&after, "builtin.store_local", k), count(&after, "builtin.load_local", k));
            let expected = match fate {
                Fate::Kept => (stores_before, loads_before),
                Fate::Promoted => (0, 0),
                Fate::DeadStoresErased => (0, 0),
            };
            assert_eq!(
                (stores_after, loads_after),
                expected,
                "case {case}: local {k} should be {fate:?}\n--- before\n{before}\n--- \
                 after\n{after}"
            );
            *fates
                .entry(match fate {
                    Fate::Kept => "kept",
                    Fate::Promoted => "promoted",
                    Fate::DeadStoresErased => "dead stores erased",
                })
                .or_default() += 1;
        }
        // nothing else changed: the other operations are all still there
        for op in ["builtin.call", "mem.store", "test.touch_local", "arith.add", "scf.if", "cf."] {
            assert_eq!(
                before.matches(op).count(),
                after.matches(op).count(),
                "case {case}: {op}\n--- before\n{before}\n--- after\n{after}"
            );
        }
        let before_f = interp::parse_function(&before).unwrap_or_else(|e| panic!("{e}\n{before}"));
        let after_f = interp::parse_function(&after).unwrap_or_else(|e| panic!("{e}\n{after}"));
        for _ in 0..6 {
            let args = [rng.below(5), rng.below(5), rng.below(2), rng.below(2), 0x1000];
            let x = interp::run(&before_f, &args, 40).unwrap_or_else(|e| panic!("{e}\n{before}"));
            let y = interp::run(&after_f, &args, 40).unwrap_or_else(|e| panic!("{e}\n{after}"));
            assert_eq!(
                x, y,
                "case {case}: args {args:?}\n--- before\n{before}\n--- after\n{after}"
            );
        }
    }
    std::println!("fates: {fates:?}");
    assert!(fates["promoted"] > 100, "{fates:?}");
    assert!(fates["dead stores erased"] > 100, "{fates:?}");
    assert!(fates["kept"] > 100, "{fates:?}");
}
