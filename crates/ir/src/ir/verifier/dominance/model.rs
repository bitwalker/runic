//! The dominance walk against a naive model, over randomly generated modules.
//!
//! The model shares nothing with the walk or with `crate::dominance`: reachability is a search
//! over the successors of each block's last operation, and block `a` dominates block `b` when
//! `b` cannot be reached from the entry once `a` is removed.
//!
//! A generated module has two functions, each of one to five blocks with random branches and block
//! arguments, holding operations with regions (`test.dom_region`, `test.dom_holder`,
//! `test.dom_graph`, and `test.dom_isolated`, isolated from above) nested up to five deep, whose
//! regions have one to five blocks too. Every block ends with a terminator (a branch, or
//! `builtin.ret` in a function), except the last block of a nested region, which may end with any
//! operation: the operations with regions have `NoTerminator`. Once a module holds
//! [MAX_NAMES] names it opens no more regions.
//!
//! The tests run the seeds of [crate::testing::test_seeds], or the one in `CORE_TEST_SEED`.

use alloc::{
    collections::BTreeMap,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    BlockRef, Context, OpPrinter, Operation, OperationRef, RegionRef, ValueRef,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::test::TestDialect,
    effects::MemoryEffectOpInterface,
    parse::{ParserConfig, parse_any},
    testing::{
        test_seeds,
        text::{Rng, describe},
    },
    traits::{IsolatedFromAbove, NoTerminator},
    verifier::dominance::{Checked, dominance_verdicts},
};

/// An operation isolated from above holding one region with SSA dominance.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(NoTerminator, IsolatedFromAbove),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct DomIsolated {
    #[region]
    body: RegionRef,
}

/// The module `source`, parsed without verification.
fn parse(source: &str) -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let config = ParserConfig::new(context.clone()).verify_after_parse(false);
    let module = parse_any(config, "model.runic", source.to_string())
        .into_result()
        .unwrap_or_else(|err| panic!("the input parses: {}\n{source}", describe(&err)));
    (context, module)
}

fn nested_ops(root: OperationRef) -> Vec<OperationRef> {
    let mut found = Vec::new();
    root.borrow().prewalk_all(|op: &Operation| found.push(op.as_operation_ref()));
    found.retain(|op| *op != root);
    found
}

fn set_operand(mut op: OperationRef, index: usize, value: ValueRef) {
    op.borrow_mut().operands_mut()[index].borrow_mut().set(value);
}

fn operand(op: OperationRef, index: usize) -> ValueRef {
    op.borrow().operands()[index].borrow().as_value_ref()
}

// ----- The naive model -----

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Model {
    Dominated,
    NotChecked,
    NotDominated,
    Isolation,
}

fn is_graph(region: RegionRef) -> bool {
    let holder = region.parent().unwrap();
    let name = holder.borrow().name().to_string();
    matches!(name.as_str(), "test.dom_graph" | "builtin.module" | "builtin.world")
}

fn blocks_of(region: RegionRef) -> Vec<BlockRef> {
    region.borrow().body().iter().map(|block| block.as_block_ref()).collect()
}

fn successors(block: BlockRef) -> Vec<BlockRef> {
    let block = block.borrow();
    match block.body().back().get() {
        Some(last) => last.successor_iter().map(|s| s.dest.borrow().successor()).collect(),
        None => Vec::new(),
    }
}

/// The blocks of `region` reachable from its entry without passing through `removed`.
fn reachable_without(region: RegionRef, removed: Option<BlockRef>) -> Vec<BlockRef> {
    let blocks = blocks_of(region);
    let Some(&entry) = blocks.first() else {
        return Vec::new();
    };
    if Some(entry) == removed {
        return Vec::new();
    }
    let mut seen = alloc::vec![entry];
    let mut work = alloc::vec![entry];
    while let Some(block) = work.pop() {
        for next in successors(block) {
            if Some(next) != removed && !seen.contains(&next) && blocks.contains(&next) {
                seen.push(next);
                work.push(next);
            }
        }
    }
    seen
}

fn position(op: OperationRef) -> usize {
    let block = op.borrow().parent().unwrap();
    let block = block.borrow();
    block.body().iter().position(|other| other.as_operation_ref() == op).unwrap()
}

/// The chain of (operation, block, region) from `user` outward: the user, the operation that
/// holds it, and so on up to an operation directly in a region of `root`.
fn chain(user: OperationRef, root: OperationRef) -> Vec<(OperationRef, BlockRef, RegionRef)> {
    let mut links = Vec::new();
    let mut at = user;
    loop {
        let block = at.borrow().parent().unwrap();
        let region = block.parent().unwrap();
        links.push((at, block, region));
        let holder = region.parent().unwrap();
        if holder == root {
            return links;
        }
        at = holder;
    }
}

fn model(user: OperationRef, value: ValueRef, root: OperationRef) -> Model {
    let value = value.borrow();
    let def_block = value.parent_block().expect("the model has no detached definitions");
    let def_region = def_block.parent().unwrap();
    let def_op = value.get_defining_op();
    // An operation that reads its own result is refused in a region of any kind, whether or not
    // it can be reached
    if def_op == Some(user) {
        return Model::NotDominated;
    }
    let links = chain(user, root);

    // Isolation: an operation isolated from above that holds the user (the root included), and
    // does not hold the definition.
    let mut holders = links.iter().skip(1).map(|link| link.0).collect::<Vec<_>>();
    holders.push(root);
    for holder in holders {
        if !holder.borrow().implements::<dyn IsolatedFromAbove>() {
            continue;
        }
        // Is the definition inside `holder`?
        let mut inside = false;
        let mut region = Some(def_region);
        while let Some(current) = region {
            let parent = current.parent().unwrap();
            if parent == holder {
                inside = true;
                break;
            }
            region = parent.borrow().parent().and_then(|block| block.parent());
        }
        if !inside {
            return Model::Isolation;
        }
    }

    // Can the use be reached?
    let (_, use_block, use_region) = links[0];
    if !reachable_without(use_region, None).contains(&use_block) {
        return Model::NotChecked;
    }
    let Some(&(ancestor, block, _)) = links.iter().find(|link| link.2 == def_region) else {
        return Model::NotDominated;
    };
    if !reachable_without(def_region, None).contains(&block) {
        return Model::NotChecked;
    }
    if is_graph(def_region) {
        return Model::Dominated;
    }
    if def_block == block {
        return match def_op {
            None => Model::Dominated,
            Some(def_op) if def_op != ancestor && position(def_op) < position(ancestor) => {
                Model::Dominated
            }
            Some(_) => Model::NotDominated,
        };
    }
    if reachable_without(def_region, Some(def_block)).contains(&block) {
        Model::NotDominated
    } else {
        Model::Dominated
    }
}

// ----- The generator -----

/// The number of names past which a generated module opens no more regions.
const MAX_NAMES: usize = 150;

struct Gen {
    rng: Rng,
    next: usize,
    text: String,
}

impl Gen {
    fn fresh(&mut self) -> usize {
        self.next += 1;
        self.next
    }

    /// The body of a region at nesting `depth` (1 is a function body). `function` says whether
    /// the last block ends with `builtin.ret`; `graph` that the region has one block.
    fn region(&mut self, depth: usize, function: bool, graph: bool, scope: Option<String>) {
        let indent = "    ".repeat(depth);
        let blocks = if graph {
            1
        } else {
            1 + self.rng.below(5) as usize
        };
        let labels = (0..blocks).map(|_| self.fresh()).collect::<Vec<_>>();
        let with_argument =
            (0..blocks).map(|block| block > 0 && self.rng.below(2) == 0).collect::<Vec<_>>();
        let constant = match scope {
            Some(name) => name,
            None => {
                let name = self.fresh();
                self.text.push_str(&format!("{indent}%c{name} = test.constant {name} : i32;\n"));
                format!("%c{name}")
            }
        };
        for block in 0..blocks {
            if block > 0 {
                let label = labels[block];
                if with_argument[block] {
                    let name = self.fresh();
                    self.text.push_str(&format!("{indent}^b{label}(%p{name}: i32):\n"));
                } else {
                    self.text.push_str(&format!("{indent}^b{label}:\n"));
                }
            }
            for _ in 0..self.rng.below(4) {
                self.op(depth, &constant);
            }
            let last = block + 1 == blocks;
            let branch = blocks > 1 && (!last || self.rng.below(3) == 0);
            // Every block of a function ends with a terminator, and so does every block of a
            // nested region but its last, which may end with any operation
            if branch && (function || !last || self.rng.below(5) != 0) {
                let target = |this: &mut Self| {
                    let to = 1 + this.rng.below(blocks as u64 - 1) as usize;
                    if with_argument[to] {
                        format!("^b{}({constant} : i32)", labels[to])
                    } else {
                        format!("^b{}", labels[to])
                    }
                };
                if self.rng.below(2) == 0 {
                    let to = target(self);
                    self.text.push_str(&format!("{indent}test.dom_br {to};\n"));
                } else {
                    let (first, second) = (target(self), target(self));
                    self.text.push_str(&format!("{indent}test.dom_either {first}, {second};\n"));
                }
            } else if function {
                self.text.push_str(&format!("{indent}builtin.ret;\n"));
            }
        }
    }

    fn op(&mut self, depth: usize, constant: &str) {
        let indent = "    ".repeat(depth);
        // Past five levels, or once the module has grown large, no more regions: the cost of the
        // model grows with the size of the module times its number of uses
        let kind = if depth < 5 && self.next < MAX_NAMES {
            self.rng.below(9)
        } else {
            self.rng.below(3)
        };
        let name = self.fresh();
        match kind {
            0 => self.text.push_str(&format!("{indent}%v{name} = test.constant {name} : i32;\n")),
            1 => self
                .text
                .push_str(&format!("{indent}%v{name} = test.add {constant}, {constant};\n")),
            2 => self
                .text
                .push_str(&format!("{indent}test.sink {constant}, {constant} : (i32, i32);\n")),
            3 | 4 => {
                self.text.push_str(&format!("{indent}test.dom_region {{\n"));
                self.region(depth + 1, false, false, None);
                self.text.push_str(&format!("{indent}}};\n"));
            }
            5 => {
                self.text.push_str(&format!("{indent}%v{name} = test.dom_holder {{\n"));
                self.region(depth + 1, false, false, None);
                self.text.push_str(&format!("{indent}}};\n"));
            }
            6 | 7 => {
                self.text.push_str(&format!("{indent}test.dom_graph {{\n"));
                self.region(depth + 1, false, true, None);
                self.text.push_str(&format!("{indent}}};\n"));
            }
            _ => {
                self.text.push_str(&format!("{indent}test.dom_isolated {{\n"));
                self.region(depth + 1, false, false, None);
                self.text.push_str(&format!("{indent}}};\n"));
            }
        }
    }

    fn module(seed: u64) -> String {
        let mut this = Gen {
            rng: Rng(seed | 1),
            next: 0,
            text: String::new(),
        };
        this.text.push_str("builtin.module public @m {\n");
        for function in 0..2 {
            this.text.push_str(&format!(
                "builtin.function public extern(\"C\") @f{function}(%a{function}: i32) {{\n"
            ));
            this.region(1, true, false, Some(format!("%a{function}")));
            this.text.push_str("};\n");
        }
        this.text.push_str("};\n");
        this.text
    }
}

/// Every value under `root`: block arguments and operation results, at any depth.
fn values(root: OperationRef) -> Vec<ValueRef> {
    let mut values = Vec::new();
    let mut ops = nested_ops(root);
    ops.insert(0, root);
    for op in ops {
        for result in op.borrow().results().iter() {
            values.push(result.borrow().as_value_ref());
        }
        for region in op.borrow().regions().iter() {
            for block in region.body().iter() {
                for argument in block.arguments() {
                    values.push(argument.borrow().as_value_ref());
                }
            }
        }
    }
    values
}

fn uses(root: OperationRef) -> Vec<(OperationRef, usize)> {
    let mut uses = Vec::new();
    for op in nested_ops(root) {
        for index in 0..op.borrow().operands().len() {
            uses.push((op, index));
        }
    }
    uses
}

fn walk_class(verdict: &Result<Checked, crate::Report>) -> Model {
    match verdict {
        Ok(Checked::Dominated) => Model::Dominated,
        Ok(Checked::Unreachable) => Model::NotChecked,
        Ok(Checked::OutsideRoot) => panic!("nothing is outside the module"),
        Err(report) => {
            let text = describe(report);
            if text.contains("isolated from above") {
                Model::Isolation
            } else {
                assert!(text.contains("is not dominated by its definition"), "{text}");
                Model::NotDominated
            }
        }
    }
}

/// How many modules a test generates and checks for each seed.
const MODULES_PER_SEED: u64 = 30;

/// The seed of each generated module, with the test seed it is derived from.
fn modules() -> impl Iterator<Item = (u64, u64)> {
    test_seeds(3).into_iter().flat_map(|seed| {
        (0..MODULES_PER_SEED)
            .map(move |case| (seed, seed.wrapping_add(case).wrapping_mul(0x9e37_79b9_7f4a_7c15)))
    })
}

/// Every operand set to a random value of the module that its operation does not define (the
/// verifier refuses that before the walk); the walk's verdict on each operand equals the model's.
/// (Per operand, so independent of what the walk reports first.)
#[test]
fn every_verdict_of_the_walk_is_the_models() {
    let mut counts = BTreeMap::<Model, usize>::new();
    for (seed, module_seed) in modules() {
        let seed = format!("seed {seed:#x} (set CORE_TEST_SEED to rerun), module {module_seed:#x}");
        let source = Gen::module(module_seed);
        let (_context, module) = parse(&source);
        if let Err(err) = module.borrow().recursively_verify() {
            panic!("{seed}: the generated module verifies: {}\n{source}", describe(&err));
        }
        let mut rng = Rng(module_seed ^ 0xabcdef);
        let pool = values(module);
        for (op, index) in uses(module) {
            let value =
                pick_where(&mut rng, &pool, |value| value.borrow().get_defining_op() != Some(op));
            set_operand(op, index, value.unwrap_or_else(|| panic!("{seed}: another value")));
        }
        let verdicts = dominance_verdicts(&module.borrow());
        assert_eq!(verdicts.len(), uses(module).len());
        for (user, index, verdict) in verdicts {
            let expected = model(user, operand(user, index), module);
            let walk = walk_class(&verdict);
            // Isolation is reported by the walk before it decides whether the use is checked,
            // as the model does.
            assert_eq!(
                walk,
                expected,
                "{seed}: operand #{index} of '{}' reading {}: {verdict:?}\n{}",
                user.borrow().name(),
                operand(user, index),
                module.borrow()
            );
            *counts.entry(expected).or_default() += 1;
        }
    }
    assert_eq!(counts.len(), 4, "{counts:?}");
}

/// A value of `pool` drawn at random among those for which `wanted` holds, or `None` if it holds
/// for none. A few draws from the whole pool are tried first, which is the same distribution as
/// a draw among the values found by asking `wanted` of each, at a fraction of the cost.
fn pick_where(
    rng: &mut Rng,
    pool: &[ValueRef],
    mut wanted: impl FnMut(ValueRef) -> bool,
) -> Option<ValueRef> {
    for _ in 0..32 {
        let value = rng.pick(pool);
        if wanted(value) {
            return Some(value);
        }
    }
    let found = pool.iter().copied().filter(|value| wanted(*value)).collect::<Vec<_>>();
    (!found.is_empty()).then(|| rng.pick(&found))
}

/// Every operand set to a random value the model accepts: the module verifies. Then one operand
/// set to a value the model rejects: the module is rejected, for that operand.
#[test]
fn the_verifier_accepts_exactly_what_the_model_accepts() {
    let (mut accepted, mut rejected) = (0, 0);
    for (seed, module_seed) in modules() {
        let seed = format!("seed {seed:#x} (set CORE_TEST_SEED to rerun), module {module_seed:#x}");
        let source = Gen::module(module_seed);
        let (_context, module) = parse(&source);
        let mut rng = Rng(module_seed ^ 0x1234567);
        let pool = values(module);
        let all = uses(module);
        for &(op, index) in &all {
            let good =
                pick_where(&mut rng, &pool, |value| model(op, value, module) <= Model::NotChecked);
            set_operand(
                op,
                index,
                good.unwrap_or_else(|| panic!("{seed}: a value the model accepts")),
            );
        }
        if let Err(err) = module.borrow().recursively_verify() {
            panic!("{seed}: the model accepts every use: {}\n{}", describe(&err), module.borrow());
        }
        accepted += 1;
        if all.is_empty() {
            continue;
        }
        for _ in 0..8 {
            let (op, index) = rng.pick(&all);
            let before = operand(op, index);
            let Some(value) = pick_where(&mut rng, &pool, |value| {
                model(op, value, module) >= Model::NotDominated
            }) else {
                continue;
            };
            let expected = model(op, value, module);
            set_operand(op, index, value);
            let Err(err) = module.borrow().recursively_verify() else {
                panic!(
                    "{seed}: the model rejects operand #{index} of '{}' reading {value} \
                     ({expected:?}), the verifier accepts it\n{}",
                    op.borrow().name(),
                    module.borrow()
                );
            };
            let text = describe(&err);
            assert!(
                text.starts_with(&format!("operand #{index} of '{}'", op.borrow().name())),
                "{seed}: {text}"
            );
            assert_eq!(
                text.contains("isolated from above"),
                expected == Model::Isolation,
                "{seed}: {text}"
            );
            rejected += 1;
            set_operand(op, index, before);
            assert!(module.borrow().recursively_verify().is_ok(), "{seed}");
        }
    }
    assert!(accepted > 0 && rejected > 0);
}
