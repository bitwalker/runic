//! A randomized test of the structured control flow operations: nested constructs with parameters
//! and results, ends, branches at random levels, branch tables and traps, some with a deliberate
//! defect. The verifier accepts exactly the programs the generator knows to be well formed; each
//! accepted program round trips and survives `dce` and `cse` still verifying.
//!
//! The seed is `WASM_CONTROL_TEST_SEED` (a `u64`), or a fixed one; a failure names the seed and
//! the round.

use runic_ir::testing::text::Rng;

use super::{
    control::{assert_round_trips, check_in, rich_context},
    control_analyses::run,
    *,
};

const ROUNDS: usize = 300;
const TYPES: [&str; 2] = ["i32", "i64"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Block,
    Loop,
    If,
}

/// A label a branch may name: the construct, and the values a branch to it carries.
#[derive(Clone)]
struct Label {
    kind: Kind,
    params: Vec<&'static str>,
    results: Vec<&'static str>,
}

impl Label {
    /// The types a branch to the label carries.
    fn expects(&self) -> &[&'static str] {
        match self.kind {
            Kind::Loop => &self.params,
            Kind::Block | Kind::If => &self.results,
        }
    }
}

/// What the generator has seen, so that the test can say it was not vacuous.
#[derive(Default, Debug)]
struct Counts {
    accepted: usize,
    rejected: usize,
    blocks: usize,
    loops: usize,
    ifs: usize,
    ifs_without_else: usize,
    ends: usize,
    branches: usize,
    branches_to_loops: usize,
    tables: usize,
    traps: usize,
    returns: usize,
    deepest: usize,
}

struct Generator {
    rng: Rng,
    /// A defect was put in the program being generated.
    defective: bool,
    next: usize,
    counts: Counts,
}

/// The values in scope: the name and the type of each.
type Values = Vec<(String, &'static str)>;

impl Generator {
    fn fresh(&mut self, prefix: &str) -> String {
        self.next += 1;
        format!("%{prefix}{}", self.next)
    }

    fn pick_type(&mut self) -> &'static str {
        self.rng.pick(&TYPES)
    }

    /// A value of type `ty` in scope; one is made with a constant when there is none (or at random).
    fn value(
        &mut self,
        ty: &'static str,
        values: &mut Values,
        out: &mut String,
        ind: &str,
    ) -> String {
        let candidates = values
            .iter()
            .filter(|(_, candidate)| *candidate == ty)
            .cloned()
            .collect::<Vec<_>>();
        if !candidates.is_empty() && self.rng.below(4) != 0 {
            return candidates[self.rng.below(candidates.len() as u64) as usize].0.clone();
        }
        let name = self.fresh("k");
        let literal = self.rng.below(100);
        out.push_str(&format!("{ind}{name} = wasm.const {literal} : {ty};\n"));
        values.push((name.clone(), ty));
        name
    }

    /// Types that differ from `types`: a defect.
    fn corrupt(&mut self, types: &[&'static str]) -> Vec<&'static str> {
        let mut wrong = types.to_vec();
        if wrong.is_empty() {
            return alloc::vec!["i32"];
        }
        match self.rng.below(3) {
            0 => {
                wrong.pop();
            }
            1 => wrong[0] = if wrong[0] == "i32" { "i64" } else { "i32" },
            _ => wrong.push("i32"),
        }
        wrong
    }

    /// Whether to put a defect here (at most one in a program).
    fn defect(&mut self) -> bool {
        if !self.defective && self.rng.below(30) == 0 {
            self.defective = true;
            true
        } else {
            false
        }
    }

    fn type_list(types: &[&str]) -> String {
        types.join(", ")
    }

    /// `values` of `types`, as the operand list text (names separated by commas).
    fn operands(
        &mut self,
        types: &[&'static str],
        values: &mut Values,
        out: &mut String,
        ind: &str,
    ) -> Vec<String> {
        types.iter().map(|ty| self.value(ty, values, out, ind)).collect()
    }

    /// The body of a construct: statements, then a terminator, written to `out`.
    fn body(
        &mut self,
        depth: usize,
        labels: &mut Vec<Label>,
        values: &mut Values,
        results: &[&'static str],
        out: &mut String,
        ind: &str,
    ) {
        self.counts.deepest = self.counts.deepest.max(depth);
        for _ in 0..self.rng.below(3) {
            match self.rng.below(4) {
                0 | 1 if depth < 4 => self.construct(depth + 1, labels, values, out, ind),
                2 => {
                    let ty = self.pick_type();
                    let a = self.value(ty, values, out, ind);
                    let b = self.value(ty, values, out, ind);
                    let name = self.fresh("v");
                    out.push_str(&format!("{ind}{name} = wasm.add {a}, {b};\n"));
                    values.push((name, ty));
                }
                _ => {}
            }
        }
        self.terminator(labels, values, results, out, ind);
    }

    fn terminator(
        &mut self,
        labels: &mut [Label],
        values: &mut Values,
        results: &[&'static str],
        out: &mut String,
        ind: &str,
    ) {
        let choice = self.rng.below(100);
        if choice < 40 || labels.is_empty() && choice < 80 {
            // fall off the end
            self.counts.ends += 1;
            let types = if self.defect() {
                self.corrupt(results)
            } else {
                results.to_vec()
            };
            let carried = self.operands(&types, values, out, ind);
            out.push_str(&format!(
                "{ind}wasm.end {}{}: ({});\n",
                carried.join(", "),
                if carried.is_empty() { "" } else { " " },
                Self::type_list(&types)
            ));
        } else if choice < 75 && !labels.is_empty() {
            self.counts.branches += 1;
            let level = self.rng.below(labels.len() as u64) as usize;
            let label = labels[labels.len() - 1 - level].clone();
            if label.kind == Kind::Loop {
                self.counts.branches_to_loops += 1;
            }
            let (level, types) = if self.defect() {
                if self.rng.below(2) == 0 {
                    // a label that is the function or beyond it
                    (labels.len() + self.rng.below(2) as usize, label.expects().to_vec())
                } else {
                    (level, self.corrupt(label.expects()))
                }
            } else {
                (level, label.expects().to_vec())
            };
            let carried = self.operands(&types, values, out, ind);
            out.push_str(&format!(
                "{ind}wasm.br {level} {}{}: ({});\n",
                carried.join(", "),
                if carried.is_empty() { "" } else { " " },
                Self::type_list(&types)
            ));
        } else if choice < 90 && !labels.is_empty() {
            self.table(labels, values, out, ind);
        } else if choice < 95 {
            self.counts.traps += 1;
            out.push_str(&format!("{ind}wasm.unreachable;\n"));
        } else {
            self.counts.returns += 1;
            out.push_str(&format!("{ind}builtin.ret;\n"));
        }
    }

    fn table(&mut self, labels: &[Label], values: &mut Values, out: &mut String, ind: &str) {
        self.counts.tables += 1;
        // the values the table carries are those of one label, and the labels it names take them
        let model = labels[self.rng.below(labels.len() as u64) as usize].clone();
        let types = model.expects().to_vec();
        let candidates = (0..labels.len())
            .filter(|level| labels[labels.len() - 1 - level].expects() == types.as_slice())
            .collect::<Vec<_>>();
        let carried = self.operands(&types, values, out, ind);
        let index = self.value("i32", values, out, ind);
        let entries = self.rng.below(4) as usize;
        let spelling = |level: usize| {
            format!(
                "wasm.br {level} {}{}: ({});",
                carried.join(", "),
                if carried.is_empty() { "" } else { " " },
                Self::type_list(&types)
            )
        };
        let defective_entry = if self.defect() {
            Some(self.rng.below(entries as u64 + 1))
        } else {
            None
        };
        let mut regions = Vec::new();
        for entry in 0..=entries {
            let mut level = candidates[self.rng.below(candidates.len() as u64) as usize];
            if defective_entry == Some(entry as u64) {
                level = labels.len() + self.rng.below(2) as usize;
            }
            regions.push(spelling(level));
        }
        let default = regions.pop().expect("a default");
        let listed = regions
            .iter()
            .map(|region| format!("{{\n{ind}    {region}\n{ind}}}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut operand_types = alloc::vec!["i32"];
        operand_types.extend(types.iter().copied());
        let mut operand_names = alloc::vec![index];
        operand_names.extend(carried.iter().cloned());
        out.push_str(&format!(
            "{ind}wasm.br_table {} [{listed}] default {{\n{ind}    {default}\n{ind}}} : ({});\n",
            operand_names.join(", "),
            Self::type_list(&operand_types)
        ));
    }

    /// A block, a loop or an `if`, as a statement.
    fn construct(
        &mut self,
        depth: usize,
        labels: &mut Vec<Label>,
        values: &mut Values,
        out: &mut String,
        ind: &str,
    ) {
        let kind = match self.rng.below(3) {
            0 => Kind::Block,
            1 => Kind::Loop,
            _ => Kind::If,
        };
        let param_count = self.rng.below(3) as usize;
        let params = (0..param_count).map(|_| self.pick_type()).collect::<Vec<_>>();
        let results = if kind == Kind::If && self.rng.below(3) == 0 {
            // the same as the parameters: an `if` that may leave out its `else`
            params.clone()
        } else {
            (0..self.rng.below(3)).map(|_| self.pick_type()).collect::<Vec<_>>()
        };
        let carried = self.operands(&params, values, out, ind);
        let condition = if kind == Kind::If {
            let ty = if self.defect() { "i64" } else { "i32" };
            Some((self.value(ty, values, out, ind), ty))
        } else {
            None
        };
        let names = results.iter().map(|_| self.fresh("r")).collect::<Vec<_>>();
        let lhs = if names.is_empty() {
            String::new()
        } else {
            format!("{} = ", names.join(", "))
        };
        let inner = format!("{ind}    ");
        let label = Label {
            kind,
            params: params.clone(),
            results: results.clone(),
        };
        let mut operand_names = Vec::new();
        let mut operand_types = Vec::new();
        if let Some((name, ty)) = &condition {
            operand_names.push(name.clone());
            operand_types.push(*ty);
        }
        operand_names.extend(carried.iter().cloned());
        operand_types.extend(params.iter().copied());
        let name = match kind {
            Kind::Block => "block",
            Kind::Loop => "loop",
            Kind::If => "if",
        };
        match kind {
            Kind::Block => self.counts.blocks += 1,
            Kind::Loop => self.counts.loops += 1,
            Kind::If => self.counts.ifs += 1,
        }
        let region = |this: &mut Self,
                      label_name: &str,
                      labels: &mut Vec<Label>,
                      values: &mut Values|
         -> String {
            let arguments = params.iter().map(|ty| (this.fresh("a"), *ty)).collect::<Vec<_>>();
            let mut inside = values.clone();
            inside.extend(arguments.iter().cloned());
            let header = arguments
                .iter()
                .map(|(name, ty)| format!("{name}: {ty}"))
                .collect::<Vec<_>>()
                .join(", ");
            let mut text = format!("{ind}^{label_name}{}({header}):\n", this.next);
            labels.push(label.clone());
            this.body(depth, labels, &mut inside, &results, &mut text, &inner);
            labels.pop();
            text
        };
        let mut text = String::new();
        let operand_text = operand_names.join(", ");
        let first = region(self, "b", labels, values);
        match kind {
            Kind::Block | Kind::Loop => {
                text.push_str(&format!("{ind}{lhs}wasm.{name} {operand_text} {{\n{first}"));
                text.push_str(&format!(
                    "{ind}}} : ({}) -> ({});\n",
                    Self::type_list(&operand_types),
                    Self::type_list(&results)
                ));
            }
            Kind::If => {
                text.push_str(&format!("{ind}{lhs}wasm.if {operand_text} then {{\n{first}"));
                if params == results && self.rng.below(2) == 0 {
                    self.counts.ifs_without_else += 1;
                } else {
                    let second = region(self, "e", labels, values);
                    text.push_str(&format!("{ind}}} else {{\n{second}"));
                }
                text.push_str(&format!(
                    "{ind}}} : ({}) -> ({});\n",
                    Self::type_list(&operand_types),
                    Self::type_list(&results)
                ));
            }
        }
        out.push_str(&text);
        values.extend(names.into_iter().zip(results.iter().copied()));
    }

    /// A function of three `i32`s and an `i64`, and its text.
    fn function(&mut self) -> String {
        self.defective = false;
        self.next = 0;
        let mut values: Values = alloc::vec![
            (String::from("%p"), "i32"),
            (String::from("%q"), "i32"),
            (String::from("%w"), "i64"),
        ];
        let mut out = String::new();
        let mut labels = Vec::new();
        for _ in 0..1 + self.rng.below(3) {
            self.construct(1, &mut labels, &mut values, &mut out, "    ");
        }
        out.push_str("    builtin.ret;\n");
        format!("builtin.function public extern(\"C\") @f(%p: i32, %q: i32, %w: i64) {{\n{out}}};")
    }
}

#[test]
fn the_verifier_accepts_exactly_the_well_formed_nests_of_constructs_and_the_passes_keep_them() {
    let seed = match std::env::var("WASM_CONTROL_TEST_SEED") {
        Ok(seed) => seed.parse().expect("WASM_CONTROL_TEST_SEED is a u64"),
        Err(_) => 0x7a5b_0001u64,
    };
    let mut generator = Generator {
        rng: Rng(seed),
        defective: false,
        next: 0,
        counts: Counts::default(),
    };
    for round in 0..ROUNDS {
        let context = rich_context();
        let text = generator.function();
        let round = format!("seed {seed:#x} (WASM_CONTROL_TEST_SEED={seed}), round {round}");
        let verdict = check_in(&context, &text);
        match (verdict, generator.defective) {
            (Ok(_), false) => {
                generator.counts.accepted += 1;
                assert_round_trips(&text);
                for pipeline in [
                    "dce",
                    "cse",
                    "dce,cse",
                    "sccp",
                    "canonicalizer",
                    "control-flow-sink",
                    "sink-operand-defs",
                    "local2reg",
                    "canonicalizer,cse,sccp,canonicalizer,dce",
                ] {
                    let after = run(pipeline, &text).unwrap_or_else(|report| {
                        panic!("{round}: {pipeline}: {}\n{text}", describe(&report))
                    });
                    // the result is a program that verifies
                    let reparsed = parse::parse_any(
                        ParserConfig::new(context.clone()),
                        "random",
                        after.clone(),
                    )
                    .into_result();
                    assert!(reparsed.is_ok(), "{round}: {pipeline}\n{text}\n--- after\n{after}");
                }
            }
            (Err(_), true) => generator.counts.rejected += 1,
            (Ok(printed), true) => {
                panic!("{round}: accepted a program with a defect:\n{text}\n--- printed\n{printed}")
            }
            (Err(error), false) => {
                panic!("{round}: rejected a well formed program: {error}\n{text}")
            }
        }
    }
    let counts = &generator.counts;
    assert!(counts.accepted > 100 && counts.rejected > 20, "{counts:?}");
    for (name, count) in [
        ("blocks", counts.blocks),
        ("loops", counts.loops),
        ("ifs", counts.ifs),
        ("ifs without else", counts.ifs_without_else),
        ("ends", counts.ends),
        ("branches", counts.branches),
        ("branches to loops", counts.branches_to_loops),
        ("tables", counts.tables),
        ("traps", counts.traps),
        ("returns", counts.returns),
    ] {
        assert!(count > 10, "too few {name}: {counts:?}");
    }
    assert!(counts.deepest >= 3, "{counts:?}");
}
