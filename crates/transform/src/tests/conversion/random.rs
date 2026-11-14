//! Random modules, in both families and both modes.
//!
//! A module has two to four functions; function `k` calls only functions after it (directly, by
//! an alias, or indirectly through a function reference), so no call recurses. A body is
//! straight-line arithmetic over the converted type and `i64`, comparisons that give an `i1`,
//! calls, loads and stores of locals (each local is stored at entry), `cf.cond_br` diamonds that
//! forward values to their arms and from them to the join (an arm may return early), and loops
//! of at most three trips whose counter and accumulators are block arguments. Every function is
//! evaluated on random inputs before and after the conversion.
//!
//! In the `isize` family only operations for which truncation commutes touch an `isize`: the
//! comparisons and loop counters are over `i64`.
//!
//! The tests run fixed seeds; `CONVERSION_TEST_SEED` (decimal, or hex with `0x`) runs one seed
//! instead, and `CONVERSION_TEST_CASES` sets the number of cases per seed. Each case has a seed of
//! its own: a failing case prints the values of the two variables that run it alone.

extern crate std;

use alloc::{format, string::String, vec, vec::Vec};
use std::println;

use super::{Family, Input, Mode, Setup, check, patterns::Kind};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// True with probability `percent` in 100.
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    /// A value of `width` bits, often one that reaches a carry or a boundary.
    fn value(&mut self, width: u32) -> u128 {
        let bits = match self.below(8) {
            0 => 0,
            1 => 1,
            2 => u128::MAX,
            3 => u128::from(u64::MAX),
            4 => 1 << self.below(width as usize),
            5 => u128::MAX << self.below(width as usize),
            _ => u128::from(self.next()) << 64 | u128::from(self.next()),
        };
        super::mask(bits, width)
    }
}

/// The seeds a test runs: `CONVERSION_TEST_SEED` if set, else `default`.
fn seeds(default: &[u64]) -> Vec<u64> {
    let Ok(seed) = std::env::var("CONVERSION_TEST_SEED") else {
        return default.to_vec();
    };
    let seed = seed.trim();
    let seed = match seed.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => seed.parse(),
    };
    vec![seed.expect("CONVERSION_TEST_SEED is a u64")]
}

/// The number of cases per seed: `CONVERSION_TEST_CASES` if set, else `default`.
fn cases(default: usize) -> usize {
    match std::env::var("CONVERSION_TEST_CASES") {
        Ok(n) => n.trim().parse().expect("CONVERSION_TEST_CASES is a number"),
        Err(_) => default,
    }
}

// ---------------------------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Ty {
    /// The type the converter changes: `i128` or `isize`.
    X,
    /// `i64`.
    N,
    /// `i1`.
    B,
}

#[derive(Debug, Clone)]
struct Sig {
    params: Vec<Ty>,
    results: Vec<Ty>,
}

/// The generator of one module.
struct Gen<'r> {
    rng: &'r mut Rng,
    family: Family,
    sigs: Vec<Sig>,
    /// The aliases: their name and the index of the function they name.
    aliases: Vec<(String, usize)>,
    /// The function being generated.
    me: usize,
    lines: Vec<String>,
    /// The values that dominate the point being generated, with their types.
    scope: Vec<(String, Ty)>,
    values: usize,
    blocks: usize,
    locals: Vec<Ty>,
    /// The structured constructs (diamonds and loops) the function may still have.
    budget: usize,
}

impl Gen<'_> {
    fn name(&self, ty: Ty) -> &'static str {
        match (ty, self.family) {
            (Ty::X, Family::Wide) => "i128",
            (Ty::X, Family::Size) => "isize",
            (Ty::N, _) => "i64",
            (Ty::B, _) => "i1",
        }
    }

    fn width(&self, ty: Ty) -> u32 {
        match (ty, self.family) {
            (Ty::X, Family::Wide) => 128,
            (Ty::X, Family::Size) | (Ty::N, _) => 64,
            (Ty::B, _) => 1,
        }
    }

    fn type_list(&self, types: &[Ty]) -> String {
        types.iter().map(|ty| self.name(*ty)).collect::<Vec<_>>().join(", ")
    }

    /// `-> T`, `-> (T, U)`, or nothing.
    fn arrow(&self, results: &[Ty]) -> String {
        match results.len() {
            0 => String::new(),
            1 => format!(" -> {}", self.name(results[0])),
            _ => format!(" -> ({})", self.type_list(results)),
        }
    }

    fn emit(&mut self, line: String) {
        self.lines.push(format!("        {line};"));
    }

    fn label(&mut self, block: usize, params: &[(String, Ty)]) {
        let params = params
            .iter()
            .map(|(name, ty)| format!("{name}: {}", self.name(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        if params.is_empty() {
            self.lines.push(format!("    ^b{block}:"));
        } else {
            self.lines.push(format!("    ^b{block}({params}):"));
        }
    }

    fn fresh(&mut self, ty: Ty) -> String {
        let name = format!("%v{}", self.values);
        self.values += 1;
        self.scope.push((name.clone(), ty));
        name
    }

    fn block(&mut self) -> usize {
        self.blocks += 1;
        self.blocks
    }

    /// `^bN` or `^bN(%a, %b : T, U)`.
    fn successor(&self, block: usize, values: &[(String, Ty)]) -> String {
        if values.is_empty() {
            return format!("^b{block}");
        }
        let names = values.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
        let types = values.iter().map(|(_, ty)| *ty).collect::<Vec<_>>();
        format!("^b{block}({names} : {})", self.type_list(&types))
    }

    /// A value of `ty` in scope, preferring recent ones.
    fn pick(&mut self, ty: Ty) -> Option<String> {
        let candidates = self
            .scope
            .iter()
            .filter(|(_, t)| *t == ty)
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        let index = if self.rng.chance(60) {
            candidates.len() - 1 - self.rng.below(candidates.len().min(3))
        } else {
            self.rng.below(candidates.len())
        };
        Some(candidates[index].clone())
    }

    /// A value of `ty`: one in scope, or a new constant.
    fn operand(&mut self, ty: Ty) -> String {
        if self.rng.chance(85)
            && let Some(value) = self.pick(ty)
        {
            return value;
        }
        if ty == Ty::B {
            return self.condition();
        }
        self.constant(ty)
    }

    fn constant(&mut self, ty: Ty) -> String {
        let value = self.rng.value(self.width(ty));
        let name = self.fresh(ty);
        self.emit(format!("{name} = arith.constant {value} : {}", self.name(ty)));
        name
    }

    /// An `i1`: a comparison over `i64`, or over `i128` (`eq`, the one the patterns convert).
    fn condition(&mut self) -> String {
        if self.family == Family::Wide && self.rng.chance(50) {
            let (a, b) = (self.operand(Ty::X), self.operand(Ty::X));
            let name = self.fresh(Ty::B);
            self.emit(format!("{name} = arith.cmpi eq {a}, {b}"));
            return name;
        }
        let predicates = ["eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sge"];
        let predicate = predicates[self.rng.below(predicates.len())];
        let (a, b) = (self.operand(Ty::N), self.operand(Ty::N));
        let name = self.fresh(Ty::B);
        self.emit(format!("{name} = arith.cmpi {predicate} {a}, {b}"));
        name
    }

    /// The name of a random binary operation over `ty`.
    fn binary_op(&mut self, ty: Ty) -> &'static str {
        let ops: &[&'static str] = match (ty, self.family) {
            (Ty::X, Family::Wide) => &["add", "band", "bor", "bxor"],
            _ => &["add", "sub", "mul", "band", "bor", "bxor"],
        };
        ops[self.rng.below(ops.len())]
    }

    fn binary(&mut self, ty: Ty) {
        let op = self.binary_op(ty);
        let (a, b) = (self.operand(ty), self.operand(ty));
        let name = self.fresh(ty);
        self.emit(format!("{name} = arith.{op} {a}, {b}"));
    }

    /// A value of `ty` that observes what was computed: usually up to three of the most recent
    /// values of `ty` in scope folded together by `add` or `bxor`, else [Self::operand].
    fn observed(&mut self, ty: Ty) -> String {
        let recent = self
            .scope
            .iter()
            .rev()
            .filter(|(_, t)| *t == ty)
            .take(3)
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        if recent.len() < 2 || self.rng.chance(30) {
            return self.operand(ty);
        }
        let mut folded = recent[0].clone();
        for value in &recent[1..] {
            let op = if self.rng.chance(50) { "add" } else { "bxor" };
            let name = self.fresh(ty);
            self.emit(format!("{name} = arith.{op} {folded}, {value}"));
            folded = name;
        }
        folded
    }

    fn call(&mut self) {
        let count = self.sigs.len();
        if self.me + 1 >= count {
            return;
        }
        let callee = self.me + 1 + self.rng.below(count - self.me - 1);
        let sig = self.sigs[callee].clone();
        let aliases = self
            .aliases
            .iter()
            .filter(|(_, target)| *target == callee)
            .map(|(name, _)| name.clone());
        let aliases = aliases.collect::<Vec<_>>();
        let path = if !aliases.is_empty() && self.rng.chance(50) {
            aliases[self.rng.below(aliases.len())].clone()
        } else {
            format!("f{callee}")
        };
        let args = sig.params.iter().map(|ty| self.operand(*ty)).collect::<Vec<_>>().join(", ");
        let signature = format!("({}){}", self.type_list(&sig.params), self.arrow(&sig.results));
        let callee_text = if self.rng.chance(35) {
            let function = format!("%v{}", self.values);
            self.values += 1;
            self.emit(format!("{function} = builtin.function_ref @{path} : {signature}"));
            format!("builtin.call_indirect {function}({args})")
        } else {
            format!("builtin.call @{path}({args})")
        };
        let results = sig.results.iter().map(|ty| self.fresh(*ty)).collect::<Vec<_>>().join(", ");
        self.emit(format!("{results} = {callee_text} : extern(\"C\") {signature}"));
    }

    fn store(&mut self) {
        if self.locals.is_empty() {
            return;
        }
        let index = self.rng.below(self.locals.len());
        let value = self.operand(self.locals[index]);
        self.emit(format!("builtin.store_local {index}, {value}"));
    }

    fn load(&mut self) {
        if self.locals.is_empty() {
            return;
        }
        let index = self.rng.below(self.locals.len());
        let ty = self.locals[index];
        let name = self.fresh(ty);
        self.emit(format!("{name} = builtin.load_local {index} : {}", self.name(ty)));
    }

    /// One to three values of the converted type or `i64`, from scope.
    fn some_values(&mut self, max: usize) -> Vec<(String, Ty)> {
        let count = self.rng.below(max + 1);
        (0..count)
            .map(|_| {
                let ty = if self.rng.chance(65) { Ty::X } else { Ty::N };
                (self.operand(ty), ty)
            })
            .collect()
    }

    /// A return of the function's results: `builtin.ret`, or `builtin.ret_imm` of a single
    /// `i128` or `i64` result.
    fn ret(&mut self) {
        let results = self.sigs[self.me].results.clone();
        let immediate = results.len() == 1
            && (results[0] == Ty::N || self.family == Family::Wide)
            && self.rng.chance(20);
        if immediate {
            let value = self.rng.value(self.width(results[0]));
            self.emit(format!("builtin.ret_imm {value} : {}", self.name(results[0])));
            return;
        }
        let values = results.iter().map(|ty| self.observed(*ty)).collect::<Vec<_>>().join(", ");
        self.emit(format!("builtin.ret {values} : ({})", self.type_list(&results)));
    }

    /// `cf.cond_br` to two arms, each given its own forwarded values; each arm branches to the
    /// join with values of the join's argument types, or returns.
    fn diamond(&mut self, depth: usize) {
        let condition = self.operand(Ty::B);
        let (then_block, else_block, join) = (self.block(), self.block(), self.block());
        let to_then = self.some_values(2);
        let to_else = self.some_values(2);
        let joined = (0..self.rng.below(3))
            .map(|_| if self.rng.chance(65) { Ty::X } else { Ty::N })
            .collect::<Vec<_>>();
        let branch = format!(
            "cf.cond_br {condition} {}, {} : (i1)",
            self.successor(then_block, &to_then),
            self.successor(else_block, &to_else)
        );
        self.emit(branch);
        let mark = self.scope.len();
        let mut returned = false;
        for (block, forwarded) in [(then_block, to_then), (else_block, to_else)] {
            let params = forwarded.iter().map(|(_, ty)| (self.fresh(*ty), *ty)).collect::<Vec<_>>();
            self.label(block, &params);
            let count = 1 + self.rng.below(3);
            self.steps(depth + 1, count);
            if !returned && self.rng.chance(15) {
                returned = true;
                self.ret();
            } else {
                let values = joined.iter().map(|ty| (self.observed(*ty), *ty)).collect::<Vec<_>>();
                let branch = format!("cf.br {}", self.successor(join, &values));
                self.emit(branch);
            }
            self.scope.truncate(mark);
        }
        let params = joined.iter().map(|ty| (self.fresh(*ty), *ty)).collect::<Vec<_>>();
        self.label(join, &params);
    }

    /// A loop of at most three trips: a counter (`i128` in the wide family, `i64` in the other)
    /// and accumulators, block arguments of the head, which exits when the counter equals its
    /// start plus the trips.
    fn repeat(&mut self, depth: usize) {
        let counter = if self.family == Family::Wide {
            Ty::X
        } else {
            Ty::N
        };
        let width = self.width(counter);
        let start = match self.rng.below(3) {
            0 => 0,
            1 => super::mask(u128::from(u64::MAX) - 1, width),
            _ => self.rng.value(width),
        };
        let trips = self.rng.below(4) as u128;
        let end = super::mask(start.wrapping_add(trips), width);
        let start_value = self.fresh(counter);
        self.emit(format!("{start_value} = arith.constant {start} : {}", self.name(counter)));
        let carried = {
            let mut carried = self.some_values(2);
            if carried.is_empty() {
                carried.push((self.operand(Ty::X), Ty::X));
            }
            carried
        };
        let (head, body, exit) = (self.block(), self.block(), self.block());
        let mut entry = vec![(start_value, counter)];
        entry.extend(carried.iter().cloned());
        let branch = format!("cf.br {}", self.successor(head, &entry));
        self.emit(branch);

        let index = self.fresh(counter);
        let mut params = vec![(index.clone(), counter)];
        params.extend(carried.iter().map(|(_, ty)| (self.fresh(*ty), *ty)));
        self.label(head, &params);
        let bound = format!("%v{}", self.values);
        self.values += 1;
        self.emit(format!("{bound} = arith.constant {end} : {}", self.name(counter)));
        let done = self.fresh(Ty::B);
        self.emit(format!("{done} = arith.cmpi eq {index}, {bound}"));
        self.emit(format!("cf.cond_br {done} ^b{exit}, ^b{body} : (i1)"));
        let head_scope = self.scope.len();

        self.label(body, &[]);
        let count = 1 + self.rng.below(3);
        self.steps(depth + 1, count);
        let one = format!("%v{}", self.values);
        let next = format!("%v{}", self.values + 1);
        self.values += 2;
        self.emit(format!("{one} = arith.constant 1 : {}", self.name(counter)));
        self.emit(format!("{next} = arith.add {index}, {one}"));
        let mut back = vec![(next, counter)];
        back.extend(carried.iter().map(|(_, ty)| (self.observed(*ty), *ty)));
        let branch = format!("cf.br {}", self.successor(head, &back));
        self.emit(branch);
        self.scope.truncate(head_scope);

        self.label(exit, &[]);
    }

    fn steps(&mut self, depth: usize, count: usize) {
        for _ in 0..count {
            match self.rng.below(12) {
                0..=2 => self.binary(Ty::X),
                3 => self.binary(Ty::N),
                4 => {
                    let ty = if self.rng.chance(60) { Ty::X } else { Ty::N };
                    self.constant(ty);
                }
                5 => {
                    self.condition();
                }
                6 => self.call(),
                7 => self.store(),
                8 => self.load(),
                9 if depth < 2 && self.budget > 0 => {
                    self.budget -= 1;
                    self.diamond(depth);
                }
                10 if depth < 2 && self.budget > 0 => {
                    self.budget -= 1;
                    self.repeat(depth);
                }
                _ => self.binary(Ty::X),
            }
        }
    }

    fn function(&mut self, me: usize) -> String {
        self.me = me;
        self.lines.clear();
        self.scope.clear();
        self.values = 0;
        self.blocks = 0;
        self.budget = 3;
        let sig = self.sigs[me].clone();
        let params = sig
            .params
            .iter()
            .map(|ty| {
                let name = self.fresh(*ty);
                format!("{name}: {}", self.name(*ty))
            })
            .collect::<Vec<_>>()
            .join(", ");
        self.locals = (0..self.rng.below(4))
            .map(|_| if self.rng.chance(60) { Ty::X } else { Ty::N })
            .collect();
        for index in 0..self.locals.len() {
            let value = self.operand(self.locals[index]);
            self.emit(format!("builtin.store_local {index}, {value}"));
        }
        let count = 2 + self.rng.below(6);
        self.steps(0, count);
        self.ret();

        let effects = if self.rng.chance(25) {
            let arg = self.rng.below(sig.params.len());
            format!(" effects(read(arg {arg}), write(result 0))")
        } else {
            String::new()
        };
        let locals = if self.locals.is_empty() {
            String::new()
        } else {
            format!(" locals({})", self.type_list(&self.locals))
        };
        format!(
            "    builtin.function public extern(\"C\") @f{me}({params}){}{effects}{locals} \
             {{\n{}\n    }};",
            self.arrow(&sig.results),
            self.lines.join("\n")
        )
    }
}

/// A random module of `family`, and the calls to evaluate: three of each function.
fn module(rng: &mut Rng, family: Family) -> (String, Vec<Input>) {
    let count = 2 + rng.below(3);
    let ty = |rng: &mut Rng| if rng.chance(60) { Ty::X } else { Ty::N };
    let mut sigs = (0..count)
        .map(|_| Sig {
            params: (0..1 + rng.below(3)).map(|_| ty(rng)).collect(),
            results: (0..1 + rng.below(2)).map(|_| ty(rng)).collect(),
        })
        .collect::<Vec<_>>();
    // The first parameter of the first function has the converted type, so that every module
    // holds one.
    sigs[0].params[0] = Ty::X;
    let mut aliases = Vec::new();
    for target in 1..count {
        if rng.chance(30) {
            aliases.push((format!("a{target}"), target));
        }
    }
    let mut inputs = Vec::new();
    for (index, sig) in sigs.iter().enumerate() {
        for _ in 0..3 {
            let args = sig
                .params
                .iter()
                .map(|ty| match (ty, family) {
                    (Ty::X, Family::Wide) => rng.value(128),
                    _ => rng.value(64),
                })
                .collect();
            inputs.push((format!("f{index}"), args));
        }
    }
    let mut generator = Gen {
        rng,
        family,
        sigs,
        aliases: aliases.clone(),
        me: 0,
        lines: Vec::new(),
        scope: Vec::new(),
        values: 0,
        blocks: 0,
        locals: Vec::new(),
        budget: 0,
    };
    let mut text = String::from("builtin.module public @m {\n");
    for me in 0..count {
        text.push_str(&generator.function(me));
        text.push('\n');
    }
    for (alias, target) in &aliases {
        text.push_str(&format!("    builtin.function_alias public @{alias} -> @f{target};\n"));
    }
    text.push_str("};");
    (text, inputs)
}

/// The setup of a case: in a partial conversion each kind of the family is left without a rule
/// and a pattern with probability 2 in 5, and `arith.bor` always in the wide family, since the
/// join materialization builds an `i128` one, which a rule judging it by type would call illegal.
fn setup(rng: &mut Rng, family: Family, mode: Mode) -> Setup {
    let kinds = match mode {
        Mode::Full => Kind::of(family).to_vec(),
        Mode::Partial => Kind::of(family)
            .iter()
            .copied()
            .filter(|kind| !(family == Family::Wide && *kind == Kind::Or))
            .filter(|_| !rng.chance(40))
            .collect(),
    };
    Setup {
        family,
        mode,
        kinds,
    }
}

/// Runs `default_cases` random cases per seed of `family` under `mode`; every one must convert and
/// pass the checks.
fn run(family: Family, mode: Mode, default_seeds: &[u64], default_cases: usize) {
    let mut findings = Vec::new();
    let mut ran = 0;
    for seed in seeds(default_seeds) {
        let mut rng = Rng(seed);
        for case in 0..cases(default_cases) {
            // Each case draws from a stream of its own, the first from the seed itself, so that a
            // case runs alone with its seed and a count of one.
            let case_seed = if case == 0 { seed } else { rng.next() };
            let mut case_rng = Rng(case_seed);
            let setup = setup(&mut case_rng, family, mode);
            let (source, inputs) = module(&mut case_rng, family);
            ran += 1;
            if let Err(finding) = check(&source, &setup, &inputs) {
                findings.push((case_seed, format!("{setup:?}\n{finding}\n--- source\n{source}")));
            }
        }
    }
    println!("{ran} modules of {family:?} converted under {mode:?} and checked");
    if let Some((case_seed, finding)) = findings.first() {
        panic!(
            "{} of {ran} cases failed; the first (CONVERSION_TEST_SEED={case_seed} \
             CONVERSION_TEST_CASES=1 runs it alone):\n{finding}",
            findings.len()
        );
    }
}

/// The default number of cases per seed of each test.
const CASES: usize = 100;

#[test]
fn random_wide_modules_convert_fully() {
    run(Family::Wide, Mode::Full, &[0x3e3_0001, 0x3e3_0002], CASES);
}

#[test]
fn random_wide_modules_convert_partially() {
    run(Family::Wide, Mode::Partial, &[0x3e3_0011, 0x3e3_0012], CASES);
}

#[test]
fn random_size_modules_convert_fully() {
    run(Family::Size, Mode::Full, &[0x3e3_0021, 0x3e3_0022], CASES);
}

#[test]
fn random_size_modules_convert_partially() {
    run(Family::Size, Mode::Partial, &[0x3e3_0031, 0x3e3_0032], CASES);
}

/// The generator gives what the tests claim: every module has several functions, and over the
/// cases of one seed calls of each form, aliases, locals, diamonds, loops, early returns and
/// immediates appear.
#[test]
fn the_generator_covers_every_feature() {
    let mut rng = Rng(0x3e3_0041);
    let mut text = String::new();
    for _ in 0..40 {
        let (source, _) = module(&mut rng, Family::Wide);
        text.push_str(&source);
    }
    for feature in [
        "builtin.call @f",
        "builtin.call @a",
        "builtin.call_indirect",
        "builtin.function_ref @a",
        "builtin.function_alias",
        "locals(",
        "builtin.load_local",
        "effects(",
        "cf.cond_br",
        "arith.cmpi eq",
        "builtin.ret_imm",
        "arith.band",
        "arith.bor",
        "arith.bxor",
    ] {
        assert!(text.contains(feature), "no `{feature}` in 40 modules");
    }
}
