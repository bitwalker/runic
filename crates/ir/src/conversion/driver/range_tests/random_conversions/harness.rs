//! The random conversion harness.
//!
//! Random straight-line and multi-block (a chain of blocks joined by `test.forward_br`) functions
//! over `i128` (split into two `i64`), `isize` (becomes `i64`, one for one), `i64` (legal) and the
//! marker `array<i8; 0>` (becomes nothing); a random set of patterns (each kind has one or not;
//! replacing 1:1 and 1:N, modifying in place, converting block arguments); a random target (each
//! kind has a dynamic rule, is legal, is illegal or has no rule); converters with and without
//! materialization callbacks; both entry points; both values of `reconcile_unrealized_casts`.
//!
//! For every run: either an error, or the result verifies, evaluates to what the original did
//! (casts read as split/join where they remain), holds no cast when the mode promises none and no
//! operation the target calls illegal; and a second run changes nothing.

use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
    println,
};

use super::{super::signature_tests::ForwardBr, ops::*, *};
use crate::{
    ElementRewrite, Int, Op,
    dialects::builtin::{Ret, attributes::AbiParam},
    replace_elements_in,
    types::TypeMapper,
};

pub(super) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }

    /// True with probability `percent` in 100.
    pub fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn value(&mut self, width: u32) -> u128 {
        let bits = match self.below(6) {
            0 => 0,
            1 => 1,
            2 => u128::MAX,
            3 => 1 << self.below(width as usize),
            _ => (self.next() as u128) << 64 | self.next() as u128,
        };
        bits & mask(width)
    }
}

fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

// ---------------------------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Ty {
    /// `i128`: becomes two `i64`.
    W,
    /// `i64`: legal.
    N,
    /// `isize`: becomes `i64`.
    P,
    /// The marker: becomes nothing.
    M,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Self::W => "i128",
            Self::N => "i64",
            Self::P => "isize",
            Self::M => "array<i8; 0>",
        }
    }

    fn of(context: &Context, ty: Type) -> Option<Self> {
        if ty == context.i128() {
            Some(Self::W)
        } else if ty == context.i64() {
            Some(Self::N)
        } else if ty == context.isize() {
            Some(Self::P)
        } else if ty == marker(context) {
            Some(Self::M)
        } else {
            None
        }
    }

    fn converted(self) -> bool {
        self != Self::N
    }
}

#[derive(Debug, Clone)]
enum Step {
    Const(Ty, u128),
    Mark,
    Add(usize, usize),
    AddXor(usize, usize),
    Join(usize, usize),
    Low(usize),
    High(usize),
    Tag(usize, usize),
    Sink(usize, Vec<usize>),
}

#[derive(Debug, Clone)]
enum Term {
    Br(Vec<usize>),
    Ret(Vec<usize>),
}

#[derive(Debug, Clone)]
struct BlockP {
    params: Vec<usize>,
    steps: Vec<(Step, Vec<usize>)>,
    term: Term,
}

#[derive(Debug, Clone)]
pub(super) struct Program {
    blocks: Vec<BlockP>,
    /// The type of each value, by its number.
    types: Vec<Ty>,
}

/// The kinds of operation a target has a rule for and a pattern set a pattern.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kind {
    Function,
    Constant,
    Mark,
    Add,
    AddXor,
    Join,
    Low,
    High,
    Tag,
    Sink,
    Br,
    Ret,
    // Operations only patterns and callbacks build.
    Carry,
    Mat,
    Cast,
}

const PROGRAM_KINDS: [Kind; 12] = [
    Kind::Function,
    Kind::Constant,
    Kind::Mark,
    Kind::Add,
    Kind::AddXor,
    Kind::Join,
    Kind::Low,
    Kind::High,
    Kind::Tag,
    Kind::Sink,
    Kind::Br,
    Kind::Ret,
];

impl Program {
    pub(super) fn random(rng: &mut Rng, markers_in_signature: bool) -> Self {
        let mut program = Program {
            blocks: vec![],
            types: vec![],
        };
        let block_count = if rng.chance(50) { 1 } else { 2 + rng.below(2) };
        let mut params = vec![];
        for _ in 0..rng.below(4) {
            let ty = match rng.below(if markers_in_signature { 8 } else { 7 }) {
                0..=2 => Ty::W,
                3..=4 => Ty::N,
                5..=6 => Ty::P,
                _ => Ty::M,
            };
            params.push(program.value(ty));
        }
        for index in 0..block_count {
            let mut block = BlockP {
                params: core::mem::take(&mut params),
                steps: vec![],
                term: Term::Ret(vec![]),
            };
            let count = rng.below(9);
            let mut attempts = 0;
            while block.steps.len() < count && attempts < 100 {
                attempts += 1;
                if let Some((step, results)) = program.random_step(rng) {
                    let results = results.into_iter().map(|ty| program.value(ty)).collect();
                    block.steps.push((step, results));
                }
            }
            let available = program.types.len();
            let pick_values = |rng: &mut Rng, program: &Program, max: usize, markers: bool| {
                let mut values = vec![];
                if available == 0 {
                    return values;
                }
                for _ in 0..rng.below(max + 1) {
                    let value = rng.below(available);
                    if markers || program.types[value] != Ty::M {
                        values.push(value);
                    }
                }
                values
            };
            if index + 1 == block_count {
                block.term = Term::Ret(pick_values(rng, &program, 3, markers_in_signature));
            } else {
                let forwarded = pick_values(rng, &program, 3, true);
                params = forwarded.iter().map(|v| program.value(program.types[*v])).collect();
                block.term = Term::Br(forwarded);
            }
            program.blocks.push(block);
        }
        program
    }

    fn value(&mut self, ty: Ty) -> usize {
        self.types.push(ty);
        self.types.len() - 1
    }

    fn random_step(&self, rng: &mut Rng) -> Option<(Step, Vec<Ty>)> {
        let of = |ty: Ty| {
            self.types
                .iter()
                .enumerate()
                .filter(|(_, t)| **t == ty)
                .map(|(index, _)| index)
                .collect::<Vec<_>>()
        };
        // Recent values are preferred, so block arguments and chains are used.
        let pick = |rng: &mut Rng, values: &[usize]| -> Option<usize> {
            if values.is_empty() {
                return None;
            }
            if rng.chance(60) {
                let recent = values.len().min(3);
                Some(values[values.len() - 1 - rng.below(recent)])
            } else {
                Some(values[rng.below(values.len())])
            }
        };
        let int_ty = |rng: &mut Rng| match rng.below(6) {
            0..=2 => Ty::W,
            3 => Ty::N,
            _ => Ty::P,
        };
        Some(match rng.below(12) {
            0 | 1 => {
                let ty = int_ty(rng);
                let width = if ty == Ty::W { 128 } else { 64 };
                (Step::Const(ty, rng.value(width)), vec![ty])
            }
            2 => (Step::Mark, vec![Ty::M]),
            3 | 4 => {
                let ty = int_ty(rng);
                let values = of(ty);
                (Step::Add(pick(rng, &values)?, pick(rng, &values)?), vec![ty])
            }
            5 => {
                let ty = int_ty(rng);
                let values = of(ty);
                (Step::AddXor(pick(rng, &values)?, pick(rng, &values)?), vec![ty, ty])
            }
            6 => {
                let values = of(Ty::N);
                (Step::Join(pick(rng, &values)?, pick(rng, &values)?), vec![Ty::W])
            }
            7 => (Step::Low(pick(rng, &of(Ty::W))?), vec![Ty::N]),
            8 => (Step::High(pick(rng, &of(Ty::W))?), vec![Ty::N]),
            9 => {
                let ty = int_ty(rng);
                (Step::Tag(pick(rng, &of(Ty::M))?, pick(rng, &of(ty))?), vec![ty])
            }
            _ => {
                if self.types.is_empty() {
                    return None;
                }
                let first = rng.below(self.types.len());
                let rest = (0..rng.below(4)).map(|_| rng.below(self.types.len())).collect();
                (Step::Sink(first, rest), vec![])
            }
        })
    }

    fn type_list(&self, values: &[usize]) -> String {
        values.iter().map(|v| self.types[*v].name()).collect::<Vec<_>>().join(", ")
    }

    fn value_list(values: &[usize]) -> String {
        values.iter().map(|v| format!("%{v}")).collect::<Vec<_>>().join(", ")
    }

    fn returned(&self) -> &[usize] {
        match &self.blocks.last().unwrap().term {
            Term::Ret(values) => values,
            Term::Br(_) => unreachable!(),
        }
    }

    pub(super) fn signature(&self) -> String {
        let params = self.blocks[0]
            .params
            .iter()
            .map(|v| format!("%{v}: {}", self.types[*v].name()))
            .collect::<Vec<_>>()
            .join(", ");
        let returned = self.returned();
        match returned.len() {
            0 => format!("({params})"),
            1 => format!("({params}) -> {}", self.type_list(returned)),
            _ => format!("({params}) -> ({})", self.type_list(returned)),
        }
    }

    pub(super) fn body(&self) -> String {
        let mut body = String::new();
        for (index, block) in self.blocks.iter().enumerate() {
            if index > 0 {
                let params = block
                    .params
                    .iter()
                    .map(|v| format!("%{v}: {}", self.types[*v].name()))
                    .collect::<Vec<_>>()
                    .join(", ");
                if params.is_empty() {
                    body.push_str(&format!("^bb{index}:\n"));
                } else {
                    body.push_str(&format!("^bb{index}({params}):\n"));
                }
            }
            for (step, results) in &block.steps {
                let line = match step {
                    Step::Const(ty, value) => format!("test.constant {value} : {}", ty.name()),
                    Step::Mark => String::from("test.mark"),
                    Step::Add(a, b) => format!("test.add %{a}, %{b}"),
                    Step::AddXor(a, b) => format!("test.add_xor %{a}, %{b}"),
                    Step::Join(a, b) => format!("test.join %{a}, %{b}"),
                    Step::Low(a) => format!("test.low %{a}"),
                    Step::High(a) => format!("test.high %{a}"),
                    Step::Tag(m, x) => format!("test.tag %{m}, %{x}"),
                    Step::Sink(first, rest) => {
                        let mut all = vec![*first];
                        all.extend(rest);
                        format!("test.sink {} : ({})", Self::value_list(&all), self.type_list(&all))
                    }
                };
                if results.is_empty() {
                    body.push_str(&format!("    {line};\n"));
                } else {
                    body.push_str(&format!("    {} = {line};\n", Self::value_list(results)));
                }
            }
            match &block.term {
                Term::Br(values) if values.is_empty() => {
                    body.push_str(&format!("    test.forward_br ^bb{};\n", index + 1));
                }
                Term::Br(values) => body.push_str(&format!(
                    "    test.forward_br ^bb{}({} : {});\n",
                    index + 1,
                    Self::value_list(values),
                    self.type_list(values)
                )),
                Term::Ret(values) if values.is_empty() => body.push_str("    builtin.ret;"),
                Term::Ret(values) => body.push_str(&format!(
                    "    builtin.ret {} : ({});",
                    Self::value_list(values),
                    self.type_list(values)
                )),
            }
        }
        body
    }

    fn has(&self, kind: Kind) -> bool {
        match kind {
            Kind::Function | Kind::Ret => true,
            Kind::Br => self.blocks.len() > 1,
            _ => self.blocks.iter().flat_map(|block| &block.steps).any(|(step, _)| {
                matches!(
                    (kind, step),
                    (Kind::Constant, Step::Const(..))
                        | (Kind::Mark, Step::Mark)
                        | (Kind::Add, Step::Add(..))
                        | (Kind::AddXor, Step::AddXor(..))
                        | (Kind::Join, Step::Join(..))
                        | (Kind::Low, Step::Low(..))
                        | (Kind::High, Step::High(..))
                        | (Kind::Tag, Step::Tag(..))
                        | (Kind::Sink, Step::Sink(..))
                )
            }),
        }
    }

    /// Random inputs: one `u64` per flat parameter.
    fn inputs(&self, rng: &mut Rng) -> Vec<u64> {
        let mut inputs = vec![];
        for param in &self.blocks[0].params {
            match self.types[*param] {
                Ty::W => inputs.extend([rng.value(64) as u64, rng.value(64) as u64]),
                Ty::N | Ty::P => inputs.push(rng.value(64) as u64),
                Ty::M => {}
            }
        }
        inputs
    }
}

// ---------------------------------------------------------------------------------------------
// The evaluator
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Val {
    Int(u128),
    Unit,
}

/// What a function did: the values it gave its sinks and the values it returned, each flattened
/// (an `i128` as its low and high halves, a marker as nothing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Outcome {
    sinks: Vec<Vec<u64>>,
    returned: Vec<u64>,
}

fn flatten(context: &Context, values: &[(Type, Val)]) -> Result<Vec<u64>, String> {
    let mut flat = vec![];
    for (ty, value) in values {
        match (Ty::of(context, *ty), value) {
            (Some(Ty::W), Val::Int(v)) => flat.extend([*v as u64, (*v >> 64) as u64]),
            (Some(Ty::N | Ty::P), Val::Int(v)) => flat.push(*v as u64),
            (Some(Ty::M), Val::Unit) => {}
            other => return Err(format!("a value of type {ty} holds {other:?}")),
        }
    }
    Ok(flat)
}

/// Runs `function` on `inputs` (one `u64` per flat parameter), reading a cast that remains as a
/// split, a join, a marker out of nothing, or the same bits.
pub(super) fn evaluate(function: OperationRef, inputs: &[u64]) -> Result<Outcome, String> {
    let context = function.borrow().context_rc();
    let function = function.try_downcast_op::<Function>().unwrap();
    let mut env = crate::FxHashMap::<ValueRef, Val>::default();
    let mut outcome = Outcome {
        sinks: vec![],
        returned: vec![],
    };
    let mut block = function.borrow().entry_block();
    // The entry arguments consume the inputs by their types.
    let mut inputs = inputs.iter().copied();
    for argument in block.borrow().argument_values().collect::<Vec<_>>() {
        let ty = argument.borrow().ty();
        let mut next =
            || inputs.next().ok_or_else(|| String::from("too few inputs")).map(u128::from);
        let value = match Ty::of(&context, ty) {
            Some(Ty::W) => {
                let low = next()?;
                Val::Int(low | next()? << 64)
            }
            Some(Ty::N | Ty::P) => Val::Int(next()?),
            Some(Ty::M) => Val::Unit,
            None => return Err(format!("an entry argument of type {ty}")),
        };
        env.insert(argument, value);
    }
    if inputs.next().is_some() {
        return Err(String::from("too many inputs for the entry block"));
    }

    for _ in 0..64 {
        let ops = block.borrow().body().iter().map(|op| op.as_operation_ref()).collect::<Vec<_>>();
        let mut next_block = None;
        for op_ref in ops {
            let op = op_ref.borrow();
            let mut operands = vec![];
            for operand in op.operands().iter() {
                let value = operand.borrow().as_value_ref();
                let Some(number) = env.get(&value) else {
                    return Err(format!(
                        "{} is used by '{}' before it is defined",
                        value.borrow(),
                        op.name()
                    ));
                };
                operands.push((value.borrow().ty(), *number));
            }
            let int = |index: usize| match operands[index].1 {
                Val::Int(v) => Ok(v),
                Val::Unit => Err(format!("operand {index} of '{}' is a marker", op.name())),
            };
            let result_ty = |index: usize| op.results()[index].borrow().ty();
            let width = |index: usize| match Ty::of(&context, result_ty(index)) {
                Some(Ty::W) => Ok(128),
                Some(Ty::N | Ty::P) => Ok(64),
                _ => {
                    Err(format!("result {index} of '{}' has type {}", op.name(), result_ty(index)))
                }
            };
            let mut results: Vec<Val> = vec![];
            if op.is::<Ret>() {
                outcome.returned = flatten(&context, &operands)?;
                return Ok(outcome);
            } else if op.is::<ForwardBr>() {
                let dest = op.successor(0).successor();
                let arguments = dest.borrow().argument_values().collect::<Vec<_>>();
                if arguments.len() != operands.len() {
                    return Err(String::from("a branch forwards the wrong number of values"));
                }
                for (argument, (ty, value)) in arguments.into_iter().zip(&operands) {
                    if argument.borrow().ty() != *ty {
                        return Err(String::from("a branch forwards a value of the wrong type"));
                    }
                    env.insert(argument, *value);
                }
                next_block = Some(dest);
            } else if op.is::<Sink>() {
                // A sink of markers alone is erased by its pattern: it observes nothing.
                let seen = flatten(&context, &operands)?;
                if !seen.is_empty() {
                    outcome.sinks.push(seen);
                }
            } else if let Some(constant) = op.downcast_ref::<Constant>() {
                results.push(Val::Int(constant.get_value().unsigned() & mask(width(0)?)));
            } else if op.is::<Add>() {
                results.push(Val::Int(int(0)?.wrapping_add(int(1)?) & mask(width(0)?)));
            } else if op.is::<Carry>() {
                results.push(Val::Int(((int(0)? & mask(64)) + (int(1)? & mask(64))) >> 64));
            } else if op.is::<AddXor>() {
                results.push(Val::Int(int(0)?.wrapping_add(int(1)?) & mask(width(0)?)));
                results.push(Val::Int((int(0)? ^ int(1)?) & mask(width(1)?)));
            } else if op.is::<Join>() || op.is::<MJoin>() {
                results.push(Val::Int((int(0)? & mask(64)) | int(1)? << 64));
            } else if op.is::<Low>() || op.is::<MLow>() {
                results.push(Val::Int(int(0)? & mask(64)));
            } else if op.is::<High>() || op.is::<MHigh>() {
                results.push(Val::Int(int(0)? >> 64));
            } else if op.is::<Mark>() || op.is::<MMark>() {
                results.push(Val::Unit);
            } else if op.is::<Tag>() {
                if operands[0].1 != Val::Unit {
                    return Err(String::from("a tag whose marker is not a marker"));
                }
                results.push(Val::Int(int(1)?));
            } else if op.is::<ToWord>() || op.is::<ToSize>() {
                results.push(Val::Int(int(0)?));
            } else if op.is::<UnrealizedConversionCast>() {
                let from = operands
                    .iter()
                    .map(|(ty, _)| Ty::of(&context, *ty))
                    .collect::<Option<Vec<_>>>();
                let to = (0..op.num_results())
                    .map(|index| Ty::of(&context, result_ty(index)))
                    .collect::<Option<Vec<_>>>();
                match (from.as_deref(), to.as_deref()) {
                    (Some([Ty::W]), Some([Ty::N, Ty::N])) => {
                        results.push(Val::Int(int(0)? & mask(64)));
                        results.push(Val::Int(int(0)? >> 64));
                    }
                    (Some([Ty::N, Ty::N]), Some([Ty::W])) => {
                        results.push(Val::Int((int(0)? & mask(64)) | int(1)? << 64));
                    }
                    (Some([]), Some([Ty::M])) => results.push(Val::Unit),
                    (Some([Ty::N]), Some([Ty::P])) | (Some([Ty::P]), Some([Ty::N])) => {
                        results.push(Val::Int(int(0)?));
                    }
                    _ => return Err(format!("the evaluator cannot read the cast '{}'", *op)),
                }
            } else {
                return Err(format!("the evaluator does not know '{}'", op.name()));
            }
            if results.len() != op.num_results() {
                return Err(format!("'{}' has {} results", op.name(), op.num_results()));
            }
            for (index, value) in results.into_iter().enumerate() {
                env.insert(op.results()[index].borrow().as_value_ref(), value);
            }
        }
        match next_block {
            Some(next) => block = next,
            None => return Err(String::from("a block does not end in a branch or a return")),
        }
    }
    Err(String::from("the evaluator ran too long"))
}

// ---------------------------------------------------------------------------------------------
// The configuration of one run
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum RuleKind {
    /// Legal when the operation uses no type the converter changes.
    Dyn,
    Legal,
    Illegal,
    NoRule,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum Variant {
    Replace,
    InPlace,
}

#[derive(Debug, Clone)]
pub(super) struct Config {
    pub full: bool,
    pub reconcile: bool,
    pub policy_legal: bool,
    /// Whether converter 0 and converter 1 have materialization callbacks.
    pub callbacks: [bool; 2],
    pub rules: BTreeMap<Kind, RuleKind>,
    /// The kinds that have a pattern: which converter it uses and how it converts 1:1.
    pub patterns: BTreeMap<Kind, (usize, Variant)>,
    /// Whether the function pattern converts the declared results (so returns must be flattened).
    pub convert_results: bool,
    /// Kinds that also have a pattern that never matches (it is given its operands, so casts are
    /// made for it, and answers `Ok(false)`), under the other converter and of a higher benefit
    /// or a lower one.
    pub decoys: BTreeMap<Kind, bool>,
}

impl Config {
    pub(super) fn random(rng: &mut Rng) -> Self {
        let mut config = Config {
            full: rng.chance(50),
            reconcile: rng.chance(85),
            policy_legal: rng.chance(75),
            callbacks: [rng.chance(85), rng.chance(85)],
            rules: BTreeMap::new(),
            patterns: BTreeMap::new(),
            convert_results: rng.chance(60),
            decoys: BTreeMap::new(),
        };
        for kind in [Kind::Add, Kind::AddXor, Kind::Low, Kind::High, Kind::Tag, Kind::Sink] {
            if rng.chance(30) {
                config.decoys.insert(kind, rng.coin());
            }
        }
        // Most runs convert everything; some leave kinds out.
        let coverage = [100, 100, 90, 70, 50][rng.below(5)];
        let function = rng.chance(60);
        for kind in PROGRAM_KINDS {
            let has = match kind {
                Kind::Function => function,
                Kind::Br => function == rng.chance(92),
                Kind::Ret => (function && config.convert_results) == rng.chance(92),
                _ => rng.chance(coverage),
            };
            if has {
                let converter = usize::from(rng.chance(25));
                let variant = if rng.coin() {
                    Variant::Replace
                } else {
                    Variant::InPlace
                };
                config.patterns.insert(kind, (converter, variant));
            }
            let roll = rng.below(100);
            let rule = if has {
                match roll {
                    0..=64 => RuleKind::Dyn,
                    65..=86 => RuleKind::NoRule,
                    87..=97 => RuleKind::Legal,
                    _ => RuleKind::Illegal,
                }
            } else {
                match roll {
                    0..=44 => RuleKind::Legal,
                    45..=91 => RuleKind::NoRule,
                    92..=97 => RuleKind::Dyn,
                    _ => RuleKind::Illegal,
                }
            };
            config.rules.insert(kind, rule);
        }
        for kind in [Kind::Carry, Kind::Mat] {
            let rule = if rng.chance(88) {
                RuleKind::Legal
            } else {
                RuleKind::NoRule
            };
            config.rules.insert(kind, rule);
        }
        config.rules.insert(
            Kind::Cast,
            match rng.below(3) {
                0 => RuleKind::Legal,
                1 => RuleKind::Illegal,
                _ => RuleKind::NoRule,
            },
        );
        config
    }

    /// A configuration that converts everything, with callbacks.
    pub(super) fn complete(rng: &mut Rng) -> Self {
        let mut config = Self::random(rng);
        config.callbacks = [true, true];
        config.policy_legal = true;
        config.convert_results = true;
        for kind in PROGRAM_KINDS {
            let converter = usize::from(rng.chance(25));
            let variant = if rng.coin() {
                Variant::Replace
            } else {
                Variant::InPlace
            };
            config.patterns.insert(kind, (converter, variant));
            config.rules.insert(kind, RuleKind::Dyn);
        }
        config.rules.insert(Kind::Carry, RuleKind::Legal);
        config.rules.insert(Kind::Mat, RuleKind::Legal);
        config
    }

    fn rule(&self, kind: Kind) -> RuleKind {
        self.rules[&kind]
    }

    /// Whether operations of `kind` that use a converted type are offered to patterns.
    fn offered(&self, kind: Kind) -> bool {
        match self.rule(kind) {
            RuleKind::Dyn | RuleKind::Illegal => true,
            RuleKind::Legal => false,
            RuleKind::NoRule => !self.policy_legal,
        }
    }

    fn converts(&self, kind: Kind) -> bool {
        self.patterns.contains_key(&kind) && self.offered(kind)
    }

    /// Whether the run must succeed. One-directional: a run this does not predict may still
    /// succeed.
    fn must_succeed(&self, program: &Program) -> bool {
        let all = PROGRAM_KINDS.iter().chain(&[Kind::Carry, Kind::Mat]);
        for kind in all.copied() {
            let present = program.has(kind) || matches!(kind, Kind::Carry | Kind::Mat);
            match self.rule(kind) {
                // A kind a pattern may build counts as present.
                RuleKind::Illegal => return false,
                RuleKind::Dyn if present && !self.patterns.contains_key(&kind) => return false,
                // A full conversion under a policy that calls rule-free operations illegal needs
                // every such operation converted, the legal-typed ones included.
                RuleKind::NoRule if self.full && !self.policy_legal => return false,
                _ => {}
            }
        }
        if !self.callbacks[0] || !self.callbacks[1] {
            return false;
        }
        let function = self.converts(Kind::Function);
        if program.has(Kind::Br) && self.converts(Kind::Br) != function {
            return false;
        }
        if self.converts(Kind::Ret) != (function && self.convert_results) {
            return false;
        }
        // The constants, adds and add-xors a pattern builds must be acceptable.
        true
    }
}

/// Whether `op` has an operand or a result of a type the converter changes.
fn uses_converted_types(op: &Operation) -> bool {
    let context = op.context();
    let converted = |ty: Type| Ty::of(context, ty).is_none_or(Ty::converted);
    op.operands().iter().any(|operand| converted(operand.borrow().ty()))
        || op.results().iter().any(|result| converted(result.borrow().ty()))
}

/// Whether the function `op` still has a converted type in its parameters, in its results (when
/// `results`), or in the arguments of a block.
fn function_uses_converted_types(op: &Operation, results: bool) -> bool {
    let context = op.context();
    let converted = |ty: Type| Ty::of(context, ty).is_none_or(Ty::converted);
    let function = op.downcast_ref::<Function>().unwrap();
    let signature = function.get_signature();
    signature.params().iter().any(|param| converted(param.ty))
        || (results && signature.results().iter().any(|result| converted(result.ty)))
        || op
            .regions()
            .iter()
            .any(|region| region.body().iter().any(|block| block.argument_types().any(converted)))
}

fn add_rule<T: OpRegistration>(
    target: &mut ConversionTarget,
    rule: RuleKind,
    legal: impl Fn(&Operation) -> bool + 'static,
) {
    match rule {
        RuleKind::Dyn => {
            target.add_dynamically_legal_op::<T, _>(move |op: &Operation| {
                DynamicLegalityResult::legal_if(legal(op))
            });
        }
        RuleKind::Legal => {
            target.add_legal_op::<T>();
        }
        RuleKind::Illegal => {
            target.add_illegal_op::<T>();
        }
        RuleKind::NoRule => {}
    }
}

pub(super) fn make_target(context: &Rc<Context>, config: &Config) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target.set_unknown_op_policy(if config.policy_legal {
        UnknownOpPolicy::Legal
    } else {
        UnknownOpPolicy::Illegal
    });
    let plain = |op: &Operation| !uses_converted_types(op);
    let results = config.convert_results;
    add_rule::<Function>(&mut target, config.rule(Kind::Function), move |op| {
        !function_uses_converted_types(op, results)
    });
    add_rule::<Constant>(&mut target, config.rule(Kind::Constant), plain);
    add_rule::<Mark>(&mut target, config.rule(Kind::Mark), plain);
    add_rule::<Add>(&mut target, config.rule(Kind::Add), plain);
    add_rule::<AddXor>(&mut target, config.rule(Kind::AddXor), plain);
    add_rule::<Join>(&mut target, config.rule(Kind::Join), plain);
    add_rule::<Low>(&mut target, config.rule(Kind::Low), plain);
    add_rule::<High>(&mut target, config.rule(Kind::High), plain);
    add_rule::<Tag>(&mut target, config.rule(Kind::Tag), plain);
    add_rule::<Sink>(&mut target, config.rule(Kind::Sink), plain);
    add_rule::<ForwardBr>(&mut target, config.rule(Kind::Br), plain);
    add_rule::<Ret>(&mut target, config.rule(Kind::Ret), plain);
    add_rule::<Carry>(&mut target, config.rule(Kind::Carry), plain);
    add_rule::<MJoin>(&mut target, config.rule(Kind::Mat), plain);
    add_rule::<MLow>(&mut target, config.rule(Kind::Mat), plain);
    add_rule::<MHigh>(&mut target, config.rule(Kind::Mat), plain);
    add_rule::<MMark>(&mut target, config.rule(Kind::Mat), plain);
    add_rule::<ToWord>(&mut target, config.rule(Kind::Mat), plain);
    add_rule::<ToSize>(&mut target, config.rule(Kind::Mat), plain);
    add_rule::<UnrealizedConversionCast>(&mut target, config.rule(Kind::Cast), plain);
    target
}

/// `i128` becomes `(i64, i64)`, `isize` becomes `i64`, the marker becomes nothing. With
/// `callbacks`, source and target materializations that build `test.m_join`, `test.m_low`,
/// `test.m_high`, `test.to_word`, `test.to_size` and `test.m_mark`; `calls` counts them.
pub(super) fn make_converter(
    context: &Context,
    callbacks: bool,
    calls: Rc<core::cell::Cell<usize>>,
) -> TypeConverter {
    let (i64, i128, isize, marker) =
        (context.i64(), context.i128(), context.isize(), marker(context));
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| {
        if ty == i128 {
            Some(TypeConversion::many([i64, i64]))
        } else if ty == isize {
            Some(TypeConversion::One(i64))
        } else if ty == marker {
            Some(TypeConversion::Drop)
        } else {
            None
        }
    });
    if !callbacks {
        return converter;
    }
    let source_calls = calls.clone();
    converter.add_source_materialization(move |rewriter, inputs, ty, span| {
        source_calls.set(source_calls.get() + 1);
        let value = if ty == i128 {
            let [low, high] = inputs else {
                return Ok(None);
            };
            let op = rewriter.create_op::<MJoin, _>(span, (*low, *high))?;
            op.borrow().result().as_value_ref()
        } else if ty == isize {
            let [value] = inputs else {
                return Ok(None);
            };
            let op = rewriter.create_op::<ToSize, _>(span, (*value,))?;
            op.borrow().result().as_value_ref()
        } else if ty == marker {
            if !inputs.is_empty() {
                return Ok(None);
            }
            let op = rewriter.create_op::<MMark, _>(span, ())?;
            op.borrow().result().as_value_ref()
        } else {
            return Ok(None);
        };
        Ok(Some(value))
    });
    converter.add_target_materialization(move |rewriter, input, types, span| {
        calls.set(calls.get() + 1);
        let ty = input.borrow().ty();
        if ty == i128 && types == [i64, i64] {
            let low = rewriter.create_op::<MLow, _>(span, (input,))?;
            let high = rewriter.create_op::<MHigh, _>(span, (input,))?;
            let low = low.borrow().result().as_value_ref();
            let high = high.borrow().result().as_value_ref();
            Ok(Some(smallvec::smallvec![low, high]))
        } else if ty == isize && types == [i64] {
            let op = rewriter.create_op::<ToWord, _>(span, (input,))?;
            Ok(Some(smallvec::smallvec![op.borrow().result().as_value_ref()]))
        } else {
            Ok(None)
        }
    });
    converter
}

// ---------------------------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------------------------

/// A pattern rooted at `Root` under `converter`. Before `rewrite` runs, the ranges it is given are
/// checked against the converter: the types of each range are what the converter
/// answers for that operand, and an operand the converter leaves alone is given as itself.
fn rule<Root: OpRegistration>(
    context: &Rc<Context>,
    name: &'static str,
    converter: &TypeConverter,
    generated: impl IntoIterator<Item = OperationName>,
    violations: &Seen<String>,
    rewrite: impl Fn(
        OperationRef,
        ConvertedOperands<'_>,
        &mut ConversionPatternRewriter,
    ) -> Result<bool, Report>
    + 'static,
) -> Rule {
    let check = converter.clone();
    let violations = violations.clone();
    Rule::new::<Root>(context, name, Some(converter.clone()), generated, move |op, operands, rw| {
        let context = op.borrow().context_rc();
        let values = op
            .borrow()
            .operands()
            .iter()
            .map(|operand| operand.borrow().as_value_ref())
            .collect::<Vec<_>>();
        if values.len() != operands.len() {
            violations.borrow_mut().push(format!("{name}: {} ranges", operands.len()));
        }
        for (index, value) in values.iter().enumerate() {
            let expected = check.convert_value(&context, *value);
            let range = operands.get(index);
            if types(range) != expected.to_vec() {
                violations.borrow_mut().push(format!(
                    "{name}: operand {index} of type {} was given as {:?}",
                    value.borrow().ty(),
                    types(range)
                ));
            }
            if expected.as_slice() == [value.borrow().ty()] && range != [*value] {
                violations
                    .borrow_mut()
                    .push(format!("{name}: legal operand {index} was not given as itself"));
            }
        }
        let groups = op.borrow().operands().groups().map(|group| group.len()).collect::<Vec<_>>();
        let seen = (0..operands.num_groups())
            .map(|g| operands.group(g).count())
            .collect::<Vec<_>>();
        if groups != seen {
            violations
                .borrow_mut()
                .push(format!("{name}: groups {seen:?}, expected {groups:?}"));
        }
        rewrite(op, operands, rw)
    })
}

fn ty_of(op: OperationRef, value: ValueRef) -> Option<Ty> {
    Ty::of(op.borrow().context(), value.borrow().ty())
}

fn result_ty(op: OperationRef, index: usize) -> Option<Ty> {
    let value = op.borrow().results()[index].borrow().as_value_ref();
    ty_of(op, value)
}

/// Retypes `op` in place: `isize` becomes `i64` in its results and constants, and its operands
/// are set to `new`.
fn retype_in_place(
    op: OperationRef,
    new: &[ValueRef],
    rewriter: &mut ConversionPatternRewriter,
) -> Result<(), Report> {
    let old = op
        .borrow()
        .operands()
        .iter()
        .map(|operand| operand.borrow().as_value_ref())
        .collect::<Vec<_>>();
    let context = op.borrow().context_rc();
    let mapper = TypeMapper::new(&context, |context, ty| {
        if ty == context.isize() {
            context.i64()
        } else {
            ty
        }
    });
    let mut rewrite = ElementRewrite::types(mapper).with_int_constants(|value, _, new| {
        let width = Int::width_for_type(new).expect("an integer type");
        Ok(if value.width() == width {
            *value
        } else {
            value.trunc(width)
        })
    });
    rewriter.modify_op_in_place(op, |op| {
        for (old, new) in old.iter().zip(new) {
            op.replaces_uses_of_with(*old, *new);
        }
        replace_elements_in(op, &mut rewrite)
    })??;
    Ok(())
}

fn first_result(op: &Operation) -> ValueRef {
    op.results()[0].borrow().as_value_ref()
}

pub(super) fn make_patterns(
    context: &Rc<Context>,
    config: &Config,
    converters: &[TypeConverter; 2],
    violations: &Seen<String>,
) -> Vec<Rule> {
    let mut patterns = vec![];
    let name = |kind| -> OperationName {
        match kind {
            Kind::Constant => registered::<Constant>(context),
            Kind::Add => registered::<Add>(context),
            Kind::AddXor => registered::<AddXor>(context),
            Kind::Carry => registered::<Carry>(context),
            Kind::Sink => registered::<Sink>(context),
            Kind::Br => registered::<ForwardBr>(context),
            Kind::Ret => registered::<Ret>(context),
            _ => unreachable!(),
        }
    };
    for (kind, (converter, variant)) in config.patterns.iter().map(|(k, v)| (*k, *v)) {
        let converter = &converters[converter];
        let in_place = variant == Variant::InPlace;
        let pattern = match kind {
            Kind::Function => {
                let results = config.convert_results;
                let convert = converter.clone();
                rule::<Function>(
                    context,
                    "function",
                    converter,
                    [],
                    violations,
                    move |op, _, rw| {
                        if !function_uses_converted_types(&op.borrow(), results) {
                            return Ok(false);
                        }
                        let context = op.borrow().context_rc();
                        rw.modify_op_in_place(op, |op| {
                            let function = op.downcast_mut::<Function>().unwrap();
                            let mut signature = function.get_signature_mut();
                            let params = signature
                                .params()
                                .iter()
                                .flat_map(|param| convert.convert_type(&context, param.ty))
                                .map(AbiParam::new)
                                .collect::<Vec<_>>();
                            signature.params_mut().clear();
                            signature.params_mut().extend(params);
                            if results {
                                let declared = signature
                                    .results()
                                    .iter()
                                    .flat_map(|result| convert.convert_type(&context, result.ty))
                                    .map(AbiParam::new)
                                    .collect::<Vec<_>>();
                                signature.declared_results_mut().clear();
                                signature.declared_results_mut().extend(declared);
                            }
                        })?;
                        let body = op.borrow().regions().front().as_pointer().unwrap();
                        rw.convert_region_types(body)?;
                        Ok(true)
                    },
                )
            }
            Kind::Constant => rule::<Constant>(
                context,
                "constant",
                converter,
                [name(Kind::Constant)],
                violations,
                move |op, _, rw| {
                    let constant = op.try_downcast_op::<Constant>().unwrap();
                    let bits = constant.borrow().get_value().unsigned();
                    let span = op.borrow().span();
                    match result_ty(op, 0) {
                        Some(Ty::W) => {
                            let low =
                                rw.create_op::<Constant, _>(span, (Int::from_bits(64, bits),))?;
                            let high = rw.create_op::<Constant, _>(
                                span,
                                (Int::from_bits(64, bits >> 64),),
                            )?;
                            let low = first_result(low.borrow().as_operation());
                            let high = first_result(high.borrow().as_operation());
                            rw.replace_op_with_multiple(op, &[&[low, high]])?;
                        }
                        Some(Ty::P) if in_place => retype_in_place(op, &[], rw)?,
                        Some(Ty::P) => {
                            let new =
                                rw.create_op::<Constant, _>(span, (Int::from_bits(64, bits),))?;
                            let new = first_result(new.borrow().as_operation());
                            rw.replace_op(op, &[new])?;
                        }
                        _ => return Ok(false),
                    }
                    Ok(true)
                },
            ),
            Kind::Mark => rule::<Mark>(context, "mark", converter, [], violations, |op, _, rw| {
                rw.replace_op_with_multiple(op, &[&[]])?;
                Ok(true)
            }),
            Kind::Add => rule::<Add>(
                context,
                "add",
                converter,
                [name(Kind::Add), name(Kind::Carry)],
                violations,
                move |op, operands, rw| {
                    let span = op.borrow().span();
                    match result_ty(op, 0) {
                        Some(Ty::W) => {
                            let ([a_low, a_high], [b_low, b_high]) =
                                (operands.get(0), operands.get(1))
                            else {
                                return Ok(false);
                            };
                            let wrap = Overflow::Wrapping;
                            let low = rw.create_op::<Add, _>(span, (*a_low, *b_low, wrap))?;
                            let carry = rw.create_op::<Carry, _>(span, (*a_low, *b_low))?;
                            let high = rw.create_op::<Add, _>(span, (*a_high, *b_high, wrap))?;
                            let high = first_result(high.borrow().as_operation());
                            let carry = first_result(carry.borrow().as_operation());
                            let high = rw.create_op::<Add, _>(span, (high, carry, wrap))?;
                            let low = first_result(low.borrow().as_operation());
                            let high = first_result(high.borrow().as_operation());
                            rw.replace_op_with_multiple(op, &[&[low, high]])?;
                        }
                        Some(Ty::P) => {
                            let Some(new) = operands.all_one() else {
                                return Ok(false);
                            };
                            if in_place {
                                retype_in_place(op, &new, rw)?;
                            } else {
                                let add = rw.create_op::<Add, _>(
                                    span,
                                    (new[0], new[1], Overflow::Wrapping),
                                )?;
                                let add = first_result(add.borrow().as_operation());
                                rw.replace_op(op, &[add])?;
                            }
                        }
                        _ => return Ok(false),
                    }
                    Ok(true)
                },
            ),
            Kind::AddXor => rule::<AddXor>(
                context,
                "add-xor",
                converter,
                [name(Kind::AddXor), name(Kind::Add), name(Kind::Carry)],
                violations,
                move |op, operands, rw| {
                    let span = op.borrow().span();
                    let both = |op: &Operation| {
                        (
                            op.results()[0].borrow().as_value_ref(),
                            op.results()[1].borrow().as_value_ref(),
                        )
                    };
                    match result_ty(op, 0) {
                        Some(Ty::W) => {
                            let ([a_low, a_high], [b_low, b_high]) =
                                (operands.get(0), operands.get(1))
                            else {
                                return Ok(false);
                            };
                            let low = rw.create_op::<AddXor, _>(span, (*a_low, *b_low))?;
                            let carry = rw.create_op::<Carry, _>(span, (*a_low, *b_low))?;
                            let high = rw.create_op::<AddXor, _>(span, (*a_high, *b_high))?;
                            let (sum_low, xor_low) = both(low.borrow().as_operation());
                            let (sum_high, xor_high) = both(high.borrow().as_operation());
                            let carry = first_result(carry.borrow().as_operation());
                            let sum_high = rw
                                .create_op::<Add, _>(span, (sum_high, carry, Overflow::Wrapping))?;
                            let sum_high = first_result(sum_high.borrow().as_operation());
                            rw.replace_op_with_multiple(
                                op,
                                &[&[sum_low, sum_high], &[xor_low, xor_high]],
                            )?;
                        }
                        Some(Ty::P) => {
                            let Some(new) = operands.all_one() else {
                                return Ok(false);
                            };
                            if in_place {
                                retype_in_place(op, &new, rw)?;
                            } else {
                                let new = rw.create_op::<AddXor, _>(span, (new[0], new[1]))?;
                                let (sum, xor) = both(new.borrow().as_operation());
                                rw.replace_op(op, &[sum, xor])?;
                            }
                        }
                        _ => return Ok(false),
                    }
                    Ok(true)
                },
            ),
            Kind::Join => {
                rule::<Join>(context, "join", converter, [], violations, |op, operands, rw| {
                    let Some(halves) = operands.all_one() else {
                        return Ok(false);
                    };
                    rw.replace_op_with_multiple(op, &[&[halves[0], halves[1]]])?;
                    Ok(true)
                })
            }
            Kind::Low | Kind::High => {
                let high = kind == Kind::High;
                let rewrite =
                    move |op: OperationRef,
                          operands: ConvertedOperands<'_>,
                          rw: &mut ConversionPatternRewriter| {
                        let [low_half, high_half] = operands.get(0) else {
                            return Ok(false);
                        };
                        rw.replace_op(op, &[if high { *high_half } else { *low_half }])?;
                        Ok(true)
                    };
                if high {
                    rule::<High>(context, "high", converter, [], violations, rewrite)
                } else {
                    rule::<Low>(context, "low", converter, [], violations, rewrite)
                }
            }
            Kind::Tag => {
                rule::<Tag>(context, "tag", converter, [], violations, |op, operands, rw| {
                    if !operands.get(0).is_empty() {
                        return Ok(false);
                    }
                    rw.replace_op_with_multiple(op, &[operands.get(1)])?;
                    Ok(true)
                })
            }
            Kind::Sink => rule::<Sink>(
                context,
                "sink",
                converter,
                [name(Kind::Sink)],
                violations,
                |op, operands, rw| {
                    if !uses_converted_types(&op.borrow()) {
                        return Ok(false);
                    }
                    let mut flat = operands.group_flattened(0);
                    flat.extend(operands.group_flattened(1));
                    let span = op.borrow().span();
                    if let Some((first, rest)) = flat.split_first() {
                        rw.set_insertion_point_before(op);
                        rw.create_op::<Sink, _>(span, (*first, rest.to_vec()))?;
                    }
                    rw.erase_op(op)?;
                    Ok(true)
                },
            ),
            Kind::Br => rule::<ForwardBr>(
                context,
                "branch",
                converter,
                [name(Kind::Br)],
                violations,
                move |op, operands, rw| {
                    if !uses_converted_types(&op.borrow()) {
                        return Ok(false);
                    }
                    let flat = operands.iter().flatten().copied().collect::<Vec<_>>();
                    if in_place {
                        // The forwarded operands of the successor are set in place.
                        rw.modify_op_in_place(op, |op| {
                            let owner = op.as_operation_ref();
                            let context = op.context_rc();
                            let group = op.successors()[0].operand_group as usize;
                            op.operands_mut().group_mut(group).set_operands(flat, owner, &context);
                        })?;
                        return Ok(true);
                    }
                    let dest = op.borrow().successor(0).successor();
                    let span = op.borrow().span();
                    rw.set_insertion_point_before(op);
                    rw.create_op::<ForwardBr, _>(span, (dest, flat))?;
                    rw.erase_op(op)?;
                    Ok(true)
                },
            ),
            Kind::Ret => rule::<Ret>(
                context,
                "return",
                converter,
                [name(Kind::Ret)],
                violations,
                |op, operands, rw| {
                    if !uses_converted_types(&op.borrow()) {
                        return Ok(false);
                    }
                    let flat = operands.iter().flatten().copied().collect::<Vec<_>>();
                    let span = op.borrow().span();
                    rw.set_insertion_point_before(op);
                    rw.create_op::<Ret, (Vec<ValueRef>,)>(span, (flat,))?;
                    rw.erase_op(op)?;
                    Ok(true)
                },
            ),
            Kind::Carry | Kind::Mat | Kind::Cast => unreachable!(),
        };
        patterns.push(pattern);
    }
    for (kind, first) in config.decoys.iter().map(|(k, v)| (*k, *v)) {
        let Some((converter, _)) = config.patterns.get(&kind) else {
            continue;
        };
        let converter = &converters[1 - converter];
        let root = match kind {
            Kind::Add => registered::<Add>(context),
            Kind::AddXor => registered::<AddXor>(context),
            Kind::Low => registered::<Low>(context),
            Kind::High => registered::<High>(context),
            Kind::Tag => registered::<Tag>(context),
            _ => registered::<Sink>(context),
        };
        let mut info = PatternInfo::new(
            context.clone(),
            "decoy",
            PatternKind::Operation(root),
            PatternBenefit::new(if first { 9 } else { 0 }),
        );
        info.with_generated_ops([]);
        let decoy = Rule {
            info,
            converter: Some(converter.clone()),
            rewrite: Box::new(|op, _, rw| {
                rw.notify_match_failure(op, Report::msg("a decoy never matches"));
                Ok(false)
            }),
        };
        if first {
            patterns.insert(0, decoy);
        } else {
            patterns.push(decoy);
        }
    }
    patterns
}

// ---------------------------------------------------------------------------------------------
// One run
// ---------------------------------------------------------------------------------------------

pub(super) fn run_conversion_with(
    function: OperationRef,
    context: &Rc<Context>,
    config: &Config,
    violations: &Seen<String>,
    calls: &Rc<core::cell::Cell<usize>>,
    conversion: ConversionConfig,
) -> Result<ConversionResult, Report> {
    let converters = [
        make_converter(context, config.callbacks[0], calls.clone()),
        make_converter(context, config.callbacks[1], calls.clone()),
    ];
    let mut set = ConversionPatternSet::new(context.clone());
    for pattern in make_patterns(context, config, &converters, violations) {
        set.push(pattern);
    }
    let target = make_target(context, config);
    if config.full {
        apply_full_conversion(function, target, set, conversion)
    } else {
        apply_partial_conversion(function, target, set, conversion)
    }
}

pub(super) fn run_conversion(
    function: OperationRef,
    context: &Rc<Context>,
    config: &Config,
    violations: &Seen<String>,
    calls: &Rc<core::cell::Cell<usize>>,
) -> Result<ConversionResult, Report> {
    let converters = [
        make_converter(context, config.callbacks[0], calls.clone()),
        make_converter(context, config.callbacks[1], calls.clone()),
    ];
    let mut set = ConversionPatternSet::new(context.clone());
    for pattern in make_patterns(context, config, &converters, violations) {
        set.push(pattern);
    }
    let mut conversion = ConversionConfig::default();
    conversion.with_reconcile_unrealized_casts(config.reconcile);
    let target = make_target(context, config);
    if config.full {
        apply_full_conversion(function, target, set, conversion)
    } else {
        apply_partial_conversion(function, target, set, conversion)
    }
}

fn all_operations(root: OperationRef) -> Vec<OperationRef> {
    let mut ops = vec![];
    root.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
    ops
}

/// What a run gave: `Ok(true)` a checked success, `Ok(false)` an error of the conversion (with
/// its message in `error`), `Err` a finding.
pub(super) fn check_case(
    rng: &mut Rng,
    complete: bool,
    error: &mut String,
) -> Result<bool, String> {
    let program = Program::random(rng, false);
    let config = if complete {
        Config::complete(rng)
    } else {
        Config::random(rng)
    };
    let inputs = program.inputs(rng);
    let (context, function) = parse_function(&program.signature(), &program.body());
    let source = function.borrow().to_string();
    let describe_case = |output: &str| {
        format!("{config:#?}\ninputs {inputs:x?}\n--- source\n{source}\n--- output\n{output}")
    };
    let expected = evaluate(function, &inputs)
        .map_err(|err| format!("the source does not evaluate: {err}\n{}", describe_case("")))?;
    let violations: Seen<String> = Rc::default();
    let calls = Rc::new(core::cell::Cell::new(0));

    let result = catch_unwind(AssertUnwindSafe(|| {
        run_conversion(function, &context, &config, &violations, &calls)
    }));
    let output = function.borrow().to_string();
    let result = match result {
        Ok(result) => result,
        Err(_) => return Err(format!("the conversion panicked\n{}", describe_case(&output))),
    };
    if let Some(violation) = violations.borrow().first() {
        return Err(format!("operand ranges: {violation}\n{}", describe_case(&output)));
    }
    let result = match result {
        Ok(result) => result,
        Err(err) => {
            *error = format!("{err}");
            if config.must_succeed(&program) {
                return Err(format!(
                    "the conversion must succeed, but: {}\n{}",
                    describe(&err),
                    describe_case(&output)
                ));
            }
            return Ok(false);
        }
    };

    // The result verifies.
    if let Err(err) = function.borrow().recursively_verify() {
        return Err(format!(
            "the result does not verify: {}\n{}",
            describe(&err),
            describe_case(&output)
        ));
    }
    // It computes what the source computed.
    match evaluate(function, &inputs) {
        Ok(actual) if actual == expected => {}
        Ok(actual) => {
            return Err(format!(
                "MISCOMPILE: gave {actual:x?}, expected {expected:x?}\n{}",
                describe_case(&output)
            ));
        }
        Err(err) => {
            return Err(format!("the result does not evaluate: {err}\n{}", describe_case(&output)));
        }
    }
    // No cast where the mode promises none, and no operation the target calls illegal.
    let target = make_target(&context, &config);
    let mut casts = 0;
    for op in all_operations(function) {
        let op = op.borrow();
        let is_cast = op.is::<UnrealizedConversionCast>();
        casts += usize::from(is_cast);
        if is_cast && config.reconcile {
            let may_stay = !config.full
                && target.is_legal(&op)
                && !(config.callbacks[0] && config.callbacks[1]);
            if !may_stay {
                return Err(format!("a cast remains\n{}", describe_case(&output)));
            }
        }
        if is_cast && !config.reconcile {
            continue;
        }
        let accepted = if config.full {
            target.is_legal(&op)
        } else {
            !target.rule_legality(&op).is_illegal()
        };
        if !accepted {
            return Err(format!(
                "'{}' remains, which the target calls illegal\n{}",
                op.name(),
                describe_case(&output)
            ));
        }
    }
    if result.changed() != (output != source) {
        return Err(format!(
            "changed() is {}, but the text {}\n{}",
            result.changed(),
            if output == source {
                "is the same"
            } else {
                "differs"
            },
            describe_case(&output)
        ));
    }

    // The callbacks were called once for each cast that remains after folding:
    // the same conversion without reconciliation, folded by hand, says how many that is.
    if config.reconcile && config.callbacks[0] && config.callbacks[1] {
        let (twin_context, twin) = parse_function(&program.signature(), &program.body());
        let mut unreconciled = config.clone();
        unreconciled.reconcile = false;
        let twin_calls = Rc::new(core::cell::Cell::new(0));
        let ran = run_conversion(twin, &twin_context, &unreconciled, &violations, &twin_calls);
        if let Err(err) = ran {
            return Err(format!(
                "the same conversion without reconciliation fails: {}\n{}",
                describe(&err),
                describe_case(&output)
            ));
        }
        let remaining = crate::conversion::reconcile_unrealized_casts(twin).remaining.len();
        if calls.get() != remaining || twin_calls.get() != 0 {
            return Err(format!(
                "the callbacks were called {} times for {remaining} remaining casts ({} without \
                 reconciliation)\n{}",
                calls.get(),
                twin_calls.get(),
                describe_case(&output)
            ));
        }
        drop(twin_context);
    }

    // The patterns that never match leave nothing behind: the same conversion
    // without them gives the same IR, value numbers aside.
    if !config.decoys.is_empty() {
        let (twin_context, twin) = parse_function(&program.signature(), &program.body());
        let mut plain = config.clone();
        plain.decoys.clear();
        let twin_calls = Rc::new(core::cell::Cell::new(0));
        let ran = run_conversion(twin, &twin_context, &plain, &violations, &twin_calls);
        let twin_output = twin.borrow().to_string();
        // The legalization graph is by name: a pattern without generated operations rooted at a
        // kind makes that kind reachable, and with it the patterns that generate the kind. So
        // the conversion without the patterns that never match may have no path where the one
        // with them has (a kind that is statically illegal and built by its own pattern); such
        // a pair is not compared.
        if ran.is_ok() {
            use crate::testing::text::normalize;
            if normalize(&twin_output) != normalize(&output) {
                return Err(format!(
                    "the patterns that never match changed the result\n{}\n--- without \
                     them\n{twin_output}",
                    describe_case(&output)
                ));
            }
        }
        drop(twin_context);
    }

    // A second run changes nothing.
    if casts == 0 {
        let again = catch_unwind(AssertUnwindSafe(|| {
            run_conversion(function, &context, &config, &violations, &calls)
        }));
        let second = function.borrow().to_string();
        match again {
            Err(_) => return Err(format!("the second run panicked\n{}", describe_case(&output))),
            Ok(Err(err)) => {
                return Err(format!(
                    "the second run failed: {}\n{}\n--- after the second run\n{second}",
                    describe(&err),
                    describe_case(&output)
                ));
            }
            Ok(Ok(result)) => {
                if result.changed() || second != output {
                    return Err(format!(
                        "the second run changed the IR (changed() = {})\n{}\n--- after the second \
                         run\n{second}",
                        result.changed(),
                        describe_case(&output)
                    ));
                }
            }
        }
    }
    Ok(true)
}

/// A short form of an error message, for the histogram: names and numbers removed.
fn bucket(error: &str) -> String {
    let mut short = String::new();
    let mut quoted = false;
    for c in error.chars().take(400) {
        match c {
            '\'' => {
                quoted = !quoted;
                short.push('\'');
            }
            _ if quoted => {}
            '0'..='9' => {}
            '\n' => break,
            c => short.push(c),
        }
    }
    short.truncate(110);
    short
}

/// The default cases per seed of the two harness tests: a few seconds in the dev profile.
const RANDOM_CASES: usize = 200;
const COMPLETE_CASES: usize = 150;

fn run(complete: bool, default_cases: usize) {
    let mut findings = vec![];
    let mut errors = BTreeMap::<String, usize>::new();
    let (mut ok, mut failed) = (0, 0);
    for seed in seeds(&[0x5e3e_2001, 0x5e3e_2002, 0x5e3e_2003, 0x5e3e_2004]) {
        let mut rng = Rng(seed);
        for case in 0..cases(default_cases) {
            let mut error = String::new();
            // Each case draws from a stream of its own, the first from the seed itself, so that
            // a case runs alone with its seed and a count of one.
            let case_seed = if case == 0 { seed } else { rng.next() };
            let mut case_rng = Rng(case_seed);
            match check_case(&mut case_rng, complete, &mut error) {
                Ok(true) => ok += 1,
                Ok(false) => {
                    failed += 1;
                    *errors.entry(bucket(&error)).or_default() += 1;
                }
                Err(finding) => {
                    let head = finding.lines().next().unwrap_or("").to_string();
                    findings.push((bucket(&head), case_seed, finding));
                }
            }
        }
    }
    println!("{ok} conversions succeeded and were checked; {failed} failed with an error:");
    for (error, count) in &errors {
        println!("  {count:6}  {error}");
    }
    let mut kinds = BTreeMap::<String, usize>::new();
    for (kind, ..) in &findings {
        *kinds.entry(kind.clone()).or_default() += 1;
    }
    println!("{} findings:", findings.len());
    for (kind, count) in &kinds {
        println!("  {count:6}  {kind}");
    }
    let mut shown = std::collections::BTreeSet::new();
    for (kind, case_seed, finding) in &findings {
        if shown.insert(kind.clone()) {
            println!(
                "=== RANDOM_CONVERSIONS_TEST_SEED={case_seed} RANDOM_CONVERSIONS_TEST_CASES=1 \
                 runs this case alone\n{finding}\n"
            );
        }
    }
    assert!(
        findings.is_empty(),
        "{} findings (the first of each kind is printed)",
        findings.len()
    );
}

/// Random programs under random pattern sets, targets, converters, modes and configurations.
#[test]
fn random_conversions_keep_what_programs_compute() {
    run(false, RANDOM_CASES);
}

/// Random programs under a pattern for every kind, with callbacks: every run must succeed.
#[test]
fn complete_conversions_succeed_and_keep_what_programs_compute() {
    run(true, COMPLETE_CASES);
}
