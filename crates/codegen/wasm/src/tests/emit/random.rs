//! Random functions in Wasm form, run against the evaluator ([agree_cached]) with reuse of locals
//! on and off.
//!
//! A fixed seed gives each program, so a run is the same every time and a failure names the seed
//! and prints the program. A program is a function `@f` of three `i32` parameters and one
//! declared local that nests, up to five levels deep, blocks and ifs with inputs and results
//! (including an else that passes its inputs through), counted loops, `br`s to enclosing labels
//! with values, `br_if`-shaped exits, `br_table`s with values, early returns, calls with two
//! results, and reads and writes of the declared local; values are picked mostly from the most
//! recent ones, so that the planner keeps some on the stack and puts others in locals. Every value
//! is an `i32`.

use alloc::{format, string::String, vec, vec::Vec};

use crate::tests::{
    eval::Val,
    exec::{Cache, Setup, agree_cached},
    function_named, module_of, parse_module,
};

/// How many programs the test runs: a few hundred with debug assertions (the dev profile), a few
/// thousand without (the `test-release` profile).
const PROGRAMS: u64 = if cfg!(debug_assertions) { 500 } else { 3000 };

/// The arguments each program is run on.
const INPUTS: [[u32; 3]; 4] = [[0, 0, 0], [1, 2, 3], [0xffff_ffff, 7, 0x8000_0000], [5, 0, 9]];

/// A xorshift generator of 64 bits.
struct Rng(u64);

impl Rng {
    /// The next number.
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A number below `n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// True `percent` times in a hundred.
    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

/// An enclosing label: the number of values a branch to it carries, or `None` for a loop, which
/// no random branch targets (a branch back to a loop is only the loop's own counted one).
#[derive(Clone, Copy)]
struct Label {
    arity: Option<usize>,
}

/// The generator of one program.
struct Generator {
    rng: Rng,
    /// The number of the last fresh name.
    next: usize,
    /// The text so far.
    out: String,
    /// How many more statements may be written.
    budget: usize,
}

impl Generator {
    /// A fresh value name.
    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("%v{}", self.next)
    }

    /// A value of `pool`: the latest most often, then the one before it, else any.
    fn pick(&mut self, pool: &[String]) -> String {
        if self.rng.chance(55) {
            pool[pool.len() - 1].clone()
        } else if self.rng.chance(40) && pool.len() >= 2 {
            pool[pool.len() - 2].clone()
        } else {
            pool[self.rng.below(pool.len())].clone()
        }
    }

    /// Writes `text` as a line.
    fn line(&mut self, text: &str) {
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// The type list of `n` `i32`s.
    fn types(n: usize) -> String {
        format!("({})", vec!["i32"; n].join(", "))
    }

    /// One to five statements, each value defined pushed to `pool`.
    fn sequence(&mut self, pool: &mut Vec<String>, labels: &mut Vec<Label>, depth: usize) {
        let n = 1 + self.rng.below(5);
        for _ in 0..n {
            if self.budget == 0 {
                break;
            }
            self.budget -= 1;
            self.statement(pool, labels, depth);
        }
    }

    /// One statement: arithmetic, a constant, a select, the declared local, a construct (below
    /// depth 4) or a call.
    fn statement(&mut self, pool: &mut Vec<String>, labels: &mut Vec<Label>, depth: usize) {
        let choice = if depth >= 4 {
            self.rng.below(6)
        } else {
            self.rng.below(13)
        };
        match choice {
            0..=2 => {
                let ops = ["add", "sub", "mul", "xor", "and", "or", "rotl", "shr_u"];
                let op = ops[self.rng.below(ops.len())];
                let b = self.pick(pool);
                let a = self.pick(pool);
                let v = self.fresh();
                self.line(&format!("{v} = wasm.{op} {a}, {b};"));
                pool.push(v);
            }
            3 => {
                let v = self.fresh();
                let k = self.rng.below(7) as i64 - 2;
                self.line(&format!("{v} = wasm.const {k} : i32;"));
                pool.push(v);
            }
            4 => {
                let c = self.pick(pool);
                let b = self.pick(pool);
                let a = self.pick(pool);
                let v = self.fresh();
                self.line(&format!("{v} = wasm.select {c}, {a}, {b};"));
                pool.push(v);
            }
            5 => {
                if self.rng.chance(50) {
                    let a = self.pick(pool);
                    self.line(&format!("builtin.store_local 0, {a};"));
                } else {
                    let v = self.fresh();
                    self.line(&format!("{v} = builtin.load_local 0 : i32;"));
                    pool.push(v);
                }
            }
            6 | 7 => self.block(pool, labels, depth),
            8 | 9 => self.if_(pool, labels, depth),
            10 => self.counted_loop(pool, labels, depth),
            11 => self.exit_block(pool, labels, depth),
            _ => {
                let b = self.pick(pool);
                let a = self.pick(pool);
                let p = self.fresh();
                let q = self.fresh();
                self.line(&format!(
                    "{p}, {q} = builtin.call @g({a}, {b}) : extern(\"wasm\") (i32, i32) -> (i32, \
                     i32);"
                ));
                pool.push(p);
                if self.rng.chance(60) {
                    pool.push(q);
                }
            }
        }
    }

    /// The names of `count` block arguments and the header of a block that has them.
    fn arguments(&mut self, count: usize) -> (Vec<String>, String) {
        let names = (0..count).map(|_| self.fresh()).collect::<Vec<_>>();
        let header = if count == 0 {
            String::new()
        } else {
            let list = names.iter().map(|n| format!("{n}: i32")).collect::<Vec<_>>().join(", ");
            let block = self.fresh();
            format!("^b{}({list}):\n", &block[2..])
        };
        (names, header)
    }

    /// The end of a region of `results` values, sometimes after a conditional exit.
    fn finish(&mut self, pool: &mut [String], labels: &[Label], results: usize) {
        if self.rng.chance(35) {
            self.maybe_exit(pool, labels);
        }
        let values = (0..results).map(|_| self.pick(pool)).collect::<Vec<_>>();
        self.line(&format!("wasm.end {} : {};", values.join(", "), Self::types(results)));
    }

    /// An `if` that returns, branches to an enclosing label with values, or branches through a
    /// `br_table` to labels of one arity.
    fn maybe_exit(&mut self, pool: &mut [String], labels: &[Label]) {
        let targets = labels
            .iter()
            .rev()
            .enumerate()
            .filter_map(|(level, label)| label.arity.map(|arity| (level, arity)))
            .collect::<Vec<_>>();
        let c = self.pick(pool);
        let kind = self.rng.below(4);
        if kind == 0 || targets.is_empty() {
            let v = self.pick(pool);
            self.line(&format!("wasm.if {c} then {{\nbuiltin.ret {v} : (i32);\n}} : (i32) -> ();"));
            return;
        }
        let (level, arity) = targets[self.rng.below(targets.len())];
        if kind == 3 {
            // Inside the `if`, every level is one more.
            let same = targets
                .iter()
                .filter(|(_, other)| *other == arity)
                .map(|(level, _)| *level + 1)
                .collect::<Vec<_>>();
            let values = (0..arity).map(|_| self.pick(pool)).collect::<Vec<_>>();
            let index = self.pick(pool);
            let entries = (0..1 + self.rng.below(3))
                .map(|_| {
                    let level = same[self.rng.below(same.len())];
                    format!("{{ wasm.br {level} {} : {}; }}", values.join(", "), Self::types(arity))
                })
                .collect::<Vec<_>>()
                .join(", ");
            let default = same[self.rng.below(same.len())];
            let separator = if arity == 0 { "" } else { ", " };
            self.line(&format!(
                "wasm.if {c} then {{\nwasm.br_table {index}{separator}{} [{entries}] default {{ \
                 wasm.br {default} {} : {}; }} : {};\n}} : (i32) -> ();",
                values.join(", "),
                values.join(", "),
                Self::types(arity),
                Self::types(arity + 1),
            ));
            return;
        }
        let values = (0..arity).map(|_| self.pick(pool)).collect::<Vec<_>>();
        self.line(&format!(
            "wasm.if {c} then {{\nwasm.br {} {} : {};\n}} : (i32) -> ();",
            level + 1,
            values.join(", "),
            Self::types(arity)
        ));
    }

    /// A `wasm.block` of up to two inputs and up to two results.
    fn block(&mut self, pool: &mut Vec<String>, labels: &mut Vec<Label>, depth: usize) {
        let inputs = self.rng.below(3);
        let results = self.rng.below(3);
        let ins = (0..inputs).map(|_| self.pick(pool)).collect::<Vec<_>>();
        let res = (0..results).map(|_| self.fresh()).collect::<Vec<_>>();
        let (arguments, header) = self.arguments(inputs);
        let lhs = if results == 0 {
            String::new()
        } else {
            format!("{} = ", res.join(", "))
        };
        let ins_text = if inputs == 0 {
            String::new()
        } else {
            format!(" {}", ins.join(", "))
        };
        self.line(&format!("{lhs}wasm.block{ins_text} {{"));
        self.out.push_str(&header);
        let mut inner = pool.clone();
        inner.extend(arguments);
        labels.push(Label {
            arity: Some(results),
        });
        self.sequence(&mut inner, labels, depth + 1);
        self.finish(&mut inner, labels, results);
        labels.pop();
        self.line(&format!("}} : {} -> {};", Self::types(inputs), Self::types(results)));
        pool.extend(res);
    }

    /// A block of no results that ends in a `br_if`-shaped exit and a write of the declared
    /// local.
    fn exit_block(&mut self, pool: &mut [String], labels: &mut Vec<Label>, depth: usize) {
        self.line("wasm.block {");
        labels.push(Label { arity: Some(0) });
        let mut inner = pool.to_vec();
        self.sequence(&mut inner, labels, depth + 1);
        let c = self.pick(&inner);
        self.line(&format!("wasm.if {c} then {{\nwasm.br 1 : ();\n}} : (i32) -> ();"));
        let a = self.pick(&inner);
        self.line(&format!("builtin.store_local 0, {a};"));
        self.line("wasm.end : ();");
        labels.pop();
        self.line("} : () -> ();");
    }

    /// A `wasm.if` of up to two inputs and up to two results, whose arms may return.
    fn if_(&mut self, pool: &mut Vec<String>, labels: &mut Vec<Label>, depth: usize) {
        let inputs = self.rng.below(3);
        let results = if self.rng.chance(30) {
            inputs
        } else {
            self.rng.below(3)
        };
        let c = self.pick(pool);
        let ins = (0..inputs).map(|_| self.pick(pool)).collect::<Vec<_>>();
        let res = (0..results).map(|_| self.fresh()).collect::<Vec<_>>();
        let lhs = if results == 0 {
            String::new()
        } else {
            format!("{} = ", res.join(", "))
        };
        let ins_text = ins.iter().map(|i| format!(", {i}")).collect::<String>();
        self.line(&format!("{lhs}wasm.if {c}{ins_text} then {{"));
        labels.push(Label {
            arity: Some(results),
        });
        for arm in 0..2 {
            let (arguments, header) = self.arguments(inputs);
            if arm == 1 {
                self.line("} else {");
                if results == inputs && self.rng.chance(50) {
                    // An else that passes its inputs through.
                    self.out.push_str(&header);
                    self.line(&format!(
                        "wasm.end {} : {};",
                        arguments.join(", "),
                        Self::types(inputs)
                    ));
                    continue;
                }
            }
            self.out.push_str(&header);
            let mut inner = pool.clone();
            inner.extend(arguments);
            self.sequence(&mut inner, labels, depth + 1);
            if self.rng.chance(10) {
                let v = self.pick(&inner);
                self.line(&format!("builtin.ret {v} : (i32);"));
            } else {
                self.finish(&mut inner, labels, results);
            }
        }
        labels.pop();
        self.line(&format!("}} : {} -> {};", Self::types(inputs + 1), Self::types(results)));
        pool.extend(res);
    }

    /// A `wasm.loop` of a counter and an accumulator that runs one to three times.
    fn counted_loop(&mut self, pool: &mut Vec<String>, labels: &mut Vec<Label>, depth: usize) {
        let zero = self.fresh();
        self.line(&format!("{zero} = wasm.const 0 : i32;"));
        let init = self.pick(pool);
        let r = self.fresh();
        let i = self.fresh();
        let acc = self.fresh();
        self.line(&format!("{r} = wasm.loop {zero}, {init} {{"));
        self.line(&format!("^l{}({i}: i32, {acc}: i32):", self.next));
        labels.push(Label { arity: None });
        let mut inner = pool.clone();
        inner.push(i.clone());
        inner.push(acc.clone());
        self.sequence(&mut inner, labels, depth + 1);
        let new = self.pick(&inner);
        let one = self.fresh();
        let next = self.fresh();
        let k = self.fresh();
        let more = self.fresh();
        let result = self.fresh();
        let trips = 1 + self.rng.below(3);
        self.line(&format!("{one} = wasm.const 1 : i32;"));
        self.line(&format!("{next} = wasm.add {i}, {one};"));
        self.line(&format!("{k} = wasm.const {trips} : i32;"));
        self.line(&format!("{more} = wasm.lt_u {next}, {k};"));
        self.line(&format!(
            "{result} = wasm.if {more} then {{\nwasm.br 1 {next}, {new} : (i32, i32);\n}} else \
             {{\nwasm.end {new} : (i32);\n}} : (i32) -> (i32);"
        ));
        self.line(&format!("wasm.end {result} : (i32);"));
        labels.pop();
        self.line("} : (i32, i32) -> (i32);");
        pool.push(r);
    }
}

/// The program of `seed`: `@g`, a callee of two results, and `@f`, the random function.
fn program(seed: u64) -> String {
    let mut generator = Generator {
        rng: Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1),
        next: 0,
        out: String::new(),
        budget: 40,
    };
    let mut pool = vec![String::from("%p0"), String::from("%p1"), String::from("%p2")];
    let mut labels = Vec::new();
    generator.line("builtin.store_local 0, %p2;");
    generator.sequence(&mut pool, &mut labels, 0);
    generator.sequence(&mut pool, &mut labels, 0);
    // The result depends on the last value picked and on the declared local.
    let v = generator.pick(&pool);
    let local = generator.fresh();
    generator.line(&format!("{local} = builtin.load_local 0 : i32;"));
    let result = generator.fresh();
    generator.line(&format!("{result} = wasm.xor {v}, {local};"));
    generator.line(&format!("builtin.ret {result} : (i32);"));
    format!(
        "builtin.function public extern(\"wasm\") @g(%a: i32, %b: i32) -> (i32, i32) {{\n%s = \
         wasm.sub %a, %b;\n%m = wasm.mul %a, %s;\nbuiltin.ret %m, %s : (i32, \
         i32);\n}};\nbuiltin.function public extern(\"wasm\") @f(%p0: i32, %p1: i32, %p2: i32) -> \
         i32 locals(i32) {{\n{}}};",
        generator.out
    )
}

#[test]
fn random_functions_of_nested_constructs_agree_with_the_evaluator_with_and_without_reuse() {
    for seed in 1..=PROGRAMS {
        let text = program(seed);
        let result = std::panic::catch_unwind(|| {
            let (_context, module) = parse_module(&module_of(&text));
            let function = function_named(module, "f");
            let mut cache = Cache::default();
            for input in &INPUTS {
                let arguments = input.iter().map(|value| Val::i32(*value)).collect::<Vec<_>>();
                agree_cached(&mut cache, &Setup::default(), function, &arguments);
            }
        });
        if result.is_err() {
            panic!("the random program of seed {seed} does not agree (above):\n{text}");
        }
    }
}
