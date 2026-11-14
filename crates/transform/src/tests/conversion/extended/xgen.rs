//! The generator of `conversion::random`, pushed where that suite does not go: a marker type that
//! converts to nothing (`i8`), pointers and memory (so that a call does something a pass can
//! lose), declared effects on every kind of parameter and result, function values in locals,
//! block arguments, parameters and results (function types nested in function types), external
//! functions, aliases of aliases, recursion (a fuel parameter), `cf.switch`, two edges of one
//! terminator to one block, and modules of any size.

use alloc::{format, string::String, vec, vec::Vec};

use super::Rng;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum Ty {
    /// `i128`, which becomes `(i64, i64)`.
    X,
    /// `i64`.
    N,
    /// `i1`.
    B,
    /// `i8`, the marker: it becomes nothing.
    M,
    /// `ptr<i64>`.
    P,
    /// The function type of shape `n`.
    F(usize),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum On {
    Nothing,
    Arg(u32),
    Result(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Sig {
    pub(super) params: Vec<Ty>,
    pub(super) results: Vec<Ty>,
    /// The first parameter is a fuel: the function may call itself and its group with less.
    pub(super) fuel: bool,
    /// Per parameter: whether the function may read and write through it (pointers only).
    pub(super) access: Vec<(bool, bool)>,
    /// The declared effects; `None` when the function declares nothing.
    pub(super) effects: Option<Vec<(&'static str, On)>>,
    pub(super) external: bool,
}

/// What the generator knows of a module, for the checks.
#[derive(Debug, Clone)]
pub(super) struct Model {
    pub(super) sigs: Vec<Sig>,
    pub(super) shapes: Vec<Sig>,
    /// The locals of each function.
    pub(super) locals: Vec<Vec<Ty>>,
}

#[derive(Debug, Clone)]
pub(super) enum Arg {
    Int(u128),
    /// A pointer to a fresh cell that holds this.
    Ptr(u128),
    Func(String),
}

#[derive(Debug, Clone)]
pub(super) struct Profile {
    pub(super) front: (usize, usize),
    pub(super) fuel: (usize, usize),
    pub(super) back: (usize, usize),
    pub(super) shapes: (usize, usize),
    pub(super) externs: (usize, usize),
    pub(super) steps: (usize, usize),
    pub(super) max_params: usize,
    pub(super) max_results: usize,
    /// Every function calls the next, and makes no other call.
    pub(super) chain: bool,
    pub(super) calls: usize,
    pub(super) constructs: usize,
    pub(super) markers: bool,
    pub(super) pointers: bool,
    pub(super) inputs: usize,
    /// The most functions that are evaluated.
    pub(super) input_functions: usize,
}

impl Profile {
    pub(super) fn small() -> Self {
        Self {
            front: (1, 4),
            fuel: (0, 3),
            back: (0, 3),
            shapes: (0, 3),
            externs: (0, 3),
            steps: (2, 9),
            max_params: 4,
            max_results: 3,
            chain: false,
            calls: 4,
            constructs: 3,
            markers: true,
            pointers: true,
            inputs: 3,
            input_functions: usize::MAX,
        }
    }

    pub(super) fn wide_signatures() -> Self {
        Self {
            max_params: 14,
            max_results: 7,
            ..Self::small()
        }
    }

    pub(super) fn many_functions() -> Self {
        Self {
            front: (120, 260),
            fuel: (0, 4),
            back: (10, 40),
            shapes: (1, 4),
            externs: (1, 6),
            steps: (1, 5),
            calls: 1,
            constructs: 1,
            inputs: 1,
            input_functions: 60,
            ..Self::small()
        }
    }

    pub(super) fn long_bodies() -> Self {
        Self {
            front: (1, 3),
            fuel: (0, 1),
            back: (0, 2),
            steps: (1500, 4000),
            calls: 6,
            constructs: 24,
            inputs: 2,
            ..Self::small()
        }
    }

    pub(super) fn chain() -> Self {
        Self {
            front: (150, 300),
            fuel: (0, 1),
            back: (0, 1),
            shapes: (0, 1),
            externs: (0, 1),
            steps: (1, 4),
            chain: true,
            calls: 0,
            input_functions: 6,
            constructs: 1,
            pointers: false,
            inputs: 2,
            ..Self::small()
        }
    }
}

fn between(rng: &mut Rng, range: (usize, usize)) -> usize {
    if range.1 <= range.0 {
        range.0
    } else {
        range.0 + rng.below(range.1 - range.0)
    }
}

/// The generator of one module.
struct Gen<'r> {
    rng: &'r mut Rng,
    profile: Profile,
    sigs: Vec<Sig>,
    shapes: Vec<Sig>,
    /// The first function of the fuel group, of the back group, and of the shape functions.
    group: (usize, usize, usize),
    aliases: Vec<(String, String, usize)>,
    me: usize,
    lines: Vec<String>,
    scope: Vec<(String, Ty)>,
    /// The pointers in scope: their name and what may be done through them.
    pointers: Vec<(String, bool, bool)>,
    values: usize,
    blocks: usize,
    locals: Vec<Ty>,
    budget: usize,
    calls: usize,
    /// In the recursive arm of a fuel function: the fuel less one.
    rec_fuel: Option<String>,
}

pub(super) fn type_name(shapes: &[Sig], ty: Ty) -> String {
    match ty {
        Ty::X => String::from("i128"),
        Ty::N => String::from("i64"),
        Ty::B => String::from("i1"),
        Ty::M => String::from("i8"),
        Ty::P => String::from("ptr<i64>"),
        Ty::F(shape) => function_type_text(shapes, &shapes[shape]),
    }
}

pub(super) fn type_list(shapes: &[Sig], types: &[Ty]) -> String {
    types.iter().map(|ty| type_name(shapes, *ty)).collect::<Vec<_>>().join(", ")
}

/// The results of a signature or a function type: one that is itself a function type is
/// parenthesised.
fn results_text(shapes: &[Sig], results: &[Ty], as_type: bool) -> String {
    match results {
        [] if as_type => String::from(" -> ()"),
        [] => String::new(),
        [one] if !matches!(one, Ty::F(_)) => format!(" -> {}", type_name(shapes, *one)),
        _ => format!(" -> ({})", type_list(shapes, results)),
    }
}

/// The signature `(T, U) -> V`, as a call or a function writes it.
pub(super) fn signature_text(shapes: &[Sig], sig: &Sig) -> String {
    format!(
        "({}){}",
        type_list(shapes, &sig.params),
        results_text(shapes, &sig.results, false)
    )
}

/// The function type `(T, U) -> V`.
pub(super) fn function_type_text(shapes: &[Sig], sig: &Sig) -> String {
    format!(
        "({}){}",
        type_list(shapes, &sig.params),
        results_text(shapes, &sig.results, true)
    )
}

fn width(ty: Ty) -> u32 {
    match ty {
        Ty::X => 128,
        Ty::N => 64,
        Ty::B => 1,
        Ty::M => 8,
        Ty::P | Ty::F(_) => 0,
    }
}

impl Gen<'_> {
    fn name(&self, ty: Ty) -> String {
        type_name(&self.shapes, ty)
    }

    fn types(&self, types: &[Ty]) -> String {
        type_list(&self.shapes, types)
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

    fn anonymous(&mut self) -> String {
        let name = format!("%v{}", self.values);
        self.values += 1;
        name
    }

    fn block(&mut self) -> usize {
        self.blocks += 1;
        self.blocks
    }

    fn successor(&self, block: usize, values: &[(String, Ty)]) -> String {
        if values.is_empty() {
            return format!("^b{block}");
        }
        let names = values.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
        let types = values.iter().map(|(_, ty)| *ty).collect::<Vec<_>>();
        format!("^b{block}({names} : {})", self.types(&types))
    }

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

    /// Whether function values may be made here: only ahead of the shape functions.
    fn functions_allowed(&self) -> bool {
        !self.shapes.is_empty() && self.me < self.group.2
    }

    /// A type for a value that is carried around: forwarded, joined, stored in a local.
    fn carried_type(&mut self) -> Ty {
        let roll = self.rng.below(100);
        if roll < 50 {
            Ty::X
        } else if roll < 70 {
            Ty::N
        } else if roll < 85 && self.profile.markers {
            Ty::M
        } else if roll < 100 && self.functions_allowed() {
            Ty::F(self.rng.below(self.shapes.len()))
        } else {
            Ty::X
        }
    }

    fn operand(&mut self, ty: Ty) -> String {
        if self.rng.chance(85)
            && let Some(value) = self.pick(ty)
        {
            return value;
        }
        match ty {
            Ty::B => self.condition(),
            Ty::F(shape) => {
                let target = self.group.2 + shape;
                assert!(self.me < target, "a function value inside a shape function");
                let path = self.path_of(target);
                let name = self.fresh(ty);
                let signature = function_type_text(&self.shapes, &self.shapes[shape]);
                self.emit(format!("{name} = builtin.function_ref @{path} : {signature}"));
                name
            }
            Ty::P => self.pick(ty).expect("a pointer is asked for only when one is in scope"),
            _ => self.constant(ty),
        }
    }

    fn constant(&mut self, ty: Ty) -> String {
        let value = self.rng.value(width(ty));
        let name = self.fresh(ty);
        self.emit(format!("{name} = arith.constant {value} : {}", self.name(ty)));
        name
    }

    fn condition(&mut self) -> String {
        if self.rng.chance(50) {
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

    fn binary(&mut self, ty: Ty) {
        let ops: &[&'static str] = match ty {
            Ty::X => &["add", "band", "bor", "bxor"],
            Ty::M => &["add", "bxor"],
            _ => &["add", "sub", "mul", "band", "bor", "bxor"],
        };
        let op = ops[self.rng.below(ops.len())];
        let (a, b) = (self.operand(ty), self.operand(ty));
        let name = self.fresh(ty);
        self.emit(format!("{name} = arith.{op} {a}, {b}"));
    }

    fn observed(&mut self, ty: Ty) -> String {
        if !matches!(ty, Ty::X | Ty::N) {
            return self.operand(ty);
        }
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

    /// A path that names function `target`: its name, or one of its aliases.
    fn path_of(&mut self, target: usize) -> String {
        let aliases = self
            .aliases
            .iter()
            .filter(|(_, _, t)| *t == target)
            .map(|(name, ..)| name.clone())
            .collect::<Vec<_>>();
        if !aliases.is_empty() && self.rng.chance(50) {
            aliases[self.rng.below(aliases.len())].clone()
        } else {
            format!("f{target}")
        }
    }

    /// The functions `me` may call here.
    fn callees(&self) -> Vec<usize> {
        let count = self.sigs.len();
        let (fuel, back, _) = self.group;
        if self.profile.chain {
            return if self.me + 1 < count {
                vec![self.me + 1]
            } else {
                vec![]
            };
        }
        if self.sigs[self.me].fuel {
            let mut callees = (back..count).collect::<Vec<_>>();
            if self.rec_fuel.is_some() {
                callees.extend(fuel..back);
            }
            callees
        } else {
            (self.me + 1..count).collect()
        }
    }

    fn call(&mut self) {
        if self.calls == 0 {
            return;
        }
        let callees = self.callees();
        if callees.is_empty() {
            return;
        }
        for _ in 0..3 {
            let callee = callees[self.rng.below(callees.len())];
            if self.call_function(callee) {
                self.calls -= 1;
                return;
            }
        }
    }

    /// Emits a call of `callee`, unless a pointer it needs is not in scope.
    fn call_function(&mut self, callee: usize) -> bool {
        let sig = self.sigs[callee].clone();
        let mut pointers = Vec::new();
        for (index, ty) in sig.params.iter().enumerate() {
            if *ty == Ty::P {
                let (read, write) = sig.access[index];
                let fits = self
                    .pointers
                    .iter()
                    .filter(|(_, r, w)| (*r || !read) && (*w || !write))
                    .map(|(name, ..)| name.clone())
                    .collect::<Vec<_>>();
                if fits.is_empty() {
                    return false;
                }
                pointers.push(fits[self.rng.below(fits.len())].clone());
            }
        }
        let mut pointers = pointers.into_iter();
        let mut args = Vec::new();
        for (index, ty) in sig.params.iter().enumerate() {
            if index == 0 && sig.fuel {
                let fuel = match (&self.rec_fuel, self.sigs[self.me].fuel) {
                    (Some(fuel), true) => fuel.clone(),
                    _ => {
                        let name = self.anonymous();
                        let fuel = self.rng.below(4);
                        self.emit(format!("{name} = arith.constant {fuel} : i64"));
                        name
                    }
                };
                args.push(fuel);
            } else if *ty == Ty::P {
                args.push(pointers.next().unwrap());
            } else {
                args.push(self.operand(*ty));
            }
        }
        let args = args.join(", ");
        let path = self.path_of(callee);
        let signature = signature_text(&self.shapes, &sig);
        let callee_text = if self.rng.chance(35) {
            let function = self.anonymous();
            let ty = function_type_text(&self.shapes, &sig);
            self.emit(format!("{function} = builtin.function_ref @{path} : {ty}"));
            format!("builtin.call_indirect {function}({args})")
        } else {
            format!("builtin.call @{path}({args})")
        };
        let results = sig.results.iter().map(|ty| self.fresh(*ty)).collect::<Vec<_>>().join(", ");
        if results.is_empty() {
            self.emit(format!("{callee_text} : extern(\"C\") {signature}"));
        } else {
            self.emit(format!("{results} = {callee_text} : extern(\"C\") {signature}"));
        }
        true
    }

    /// A call of a function value in scope.
    fn call_value(&mut self) {
        if self.calls == 0 || !self.functions_allowed() {
            return;
        }
        let shape = self.rng.below(self.shapes.len());
        let sig = self.shapes[shape].clone();
        let function = self.operand(Ty::F(shape));
        let args = sig.params.iter().map(|ty| self.operand(*ty)).collect::<Vec<_>>().join(", ");
        let signature = signature_text(&self.shapes, &sig);
        let results = sig.results.iter().map(|ty| self.fresh(*ty)).collect::<Vec<_>>().join(", ");
        let call = format!("builtin.call_indirect {function}({args}) : extern(\"C\") {signature}");
        if results.is_empty() {
            self.emit(call);
        } else {
            self.emit(format!("{results} = {call}"));
        }
        self.calls -= 1;
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

    fn memory(&mut self) {
        if self.pointers.is_empty() {
            return;
        }
        let (pointer, read, write) = self.pointers[self.rng.below(self.pointers.len())].clone();
        if write && (self.rng.chance(50) || !read) {
            let value = self.observed(Ty::N);
            self.emit(format!("mem.store {value}, {pointer}"));
        } else if read {
            let name = self.fresh(Ty::N);
            self.emit(format!("{name} = mem.load {pointer}"));
        }
    }

    fn some_values(&mut self, max: usize) -> Vec<(String, Ty)> {
        let count = self.rng.below(max + 1);
        (0..count)
            .map(|_| {
                let ty = self.carried_type();
                (self.operand(ty), ty)
            })
            .collect()
    }

    fn ret(&mut self) {
        let results = self.sigs[self.me].results.clone();
        let immediate =
            results.len() == 1 && matches!(results[0], Ty::N | Ty::X) && self.rng.chance(20);
        if immediate {
            let value = self.rng.value(width(results[0]));
            self.emit(format!("builtin.ret_imm {value} : {}", self.name(results[0])));
            return;
        }
        if results.is_empty() {
            self.emit(String::from("builtin.ret"));
            return;
        }
        let values = results.iter().map(|ty| self.observed(*ty)).collect::<Vec<_>>().join(", ");
        self.emit(format!("builtin.ret {values} : ({})", self.types(&results)));
    }

    /// The body of one arm: its block, some steps, and a branch to the join or a return.
    fn arm(
        &mut self,
        depth: usize,
        block: usize,
        forwarded: &[Ty],
        join: usize,
        joined: &[Ty],
        may_return: bool,
    ) {
        let mark = self.scope.len();
        let params = forwarded.iter().map(|ty| (self.fresh(*ty), *ty)).collect::<Vec<_>>();
        self.label(block, &params);
        let count = 1 + self.rng.below(3);
        self.steps(depth + 1, count);
        if may_return && self.rng.chance(15) {
            self.ret();
        } else {
            let values = joined.iter().map(|ty| (self.observed(*ty), *ty)).collect::<Vec<_>>();
            let branch = format!("cf.br {}", self.successor(join, &values));
            self.emit(branch);
        }
        self.scope.truncate(mark);
    }

    fn diamond(&mut self, depth: usize) {
        let condition = self.operand(Ty::B);
        let joined = (0..self.rng.below(3)).map(|_| self.carried_type()).collect::<Vec<_>>();
        if self.rng.chance(20) {
            // Both edges to one block, with different values of the same types.
            let (arm, join) = (self.block(), self.block());
            let to_then = self.some_values(3);
            let to_else =
                to_then.iter().map(|(_, ty)| (self.operand(*ty), *ty)).collect::<Vec<_>>();
            let branch = format!(
                "cf.cond_br {condition} {}, {} : (i1)",
                self.successor(arm, &to_then),
                self.successor(arm, &to_else)
            );
            self.emit(branch);
            let types = to_then.iter().map(|(_, ty)| *ty).collect::<Vec<_>>();
            self.arm(depth, arm, &types, join, &joined, false);
            let params = joined.iter().map(|ty| (self.fresh(*ty), *ty)).collect::<Vec<_>>();
            self.label(join, &params);
            return;
        }
        let (then_block, else_block, join) = (self.block(), self.block(), self.block());
        let to_then = self.some_values(2);
        let to_else = self.some_values(2);
        let branch = format!(
            "cf.cond_br {condition} {}, {} : (i1)",
            self.successor(then_block, &to_then),
            self.successor(else_block, &to_else)
        );
        self.emit(branch);
        let mut may_return = true;
        for (block, forwarded) in [(then_block, to_then), (else_block, to_else)] {
            let types = forwarded.iter().map(|(_, ty)| *ty).collect::<Vec<_>>();
            let lines = self.lines.len();
            self.arm(depth, block, &types, join, &joined, may_return);
            if self.lines[lines..].iter().any(|line| line.contains("builtin.ret")) {
                may_return = false;
            }
        }
        let params = joined.iter().map(|ty| (self.fresh(*ty), *ty)).collect::<Vec<_>>();
        self.label(join, &params);
    }

    /// A `cf.switch` over the low two bits of an `i64`: keys 0, 1 and -1 (never taken) and a
    /// fallback; two of the keys may share a block.
    fn switch(&mut self, depth: usize) {
        let selector = {
            let value = self.operand(Ty::N);
            let three = self.anonymous();
            self.emit(format!("{three} = arith.constant 3 : i64"));
            let low = self.anonymous();
            self.emit(format!("{low} = arith.band {value}, {three}"));
            let selector = self.anonymous();
            self.emit(format!("{selector} = arith.trunc {low} <{{ ty = #builtin.type<i32> }}>"));
            selector
        };
        let joined = (0..self.rng.below(3)).map(|_| self.carried_type()).collect::<Vec<_>>();
        let shared = self.rng.chance(40);
        let (zero, one, other, join) = (self.block(), self.block(), self.block(), self.block());
        let to_zero = self.some_values(2);
        let to_one = self.some_values(2);
        let to_other = self.some_values(2);
        // A second edge to the block of key 0, with other values of its types.
        let to_minus = to_zero.iter().map(|(_, ty)| (self.operand(*ty), *ty)).collect::<Vec<_>>();
        let mut cases = vec![
            format!("#builtin.i32<0> -> {}", self.successor(zero, &to_zero)),
            format!("#builtin.i32<1> -> {}", self.successor(one, &to_one)),
        ];
        if shared {
            cases.push(format!("#builtin.i32<2> -> {}", self.successor(zero, &to_minus)));
        }
        let branch = format!(
            "cf.switch {selector} [{}], {} : (i32)",
            cases.join(", "),
            self.successor(other, &to_other)
        );
        self.emit(branch);
        for (block, forwarded) in [(zero, to_zero), (one, to_one), (other, to_other)] {
            let types = forwarded.iter().map(|(_, ty)| *ty).collect::<Vec<_>>();
            self.arm(depth, block, &types, join, &joined, false);
        }
        let params = joined.iter().map(|ty| (self.fresh(*ty), *ty)).collect::<Vec<_>>();
        self.label(join, &params);
    }

    fn repeat(&mut self, depth: usize) {
        let counter = Ty::X;
        let start = match self.rng.below(3) {
            0 => 0,
            1 => u128::from(u64::MAX) - 1,
            _ => self.rng.value(128),
        };
        let trips = self.rng.below(4) as u128;
        let end = start.wrapping_add(trips);
        let start_value = self.fresh(counter);
        self.emit(format!("{start_value} = arith.constant {start} : i128"));
        let carried = {
            let mut carried = self.some_values(3);
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
        let bound = self.anonymous();
        self.emit(format!("{bound} = arith.constant {end} : i128"));
        let done = self.fresh(Ty::B);
        self.emit(format!("{done} = arith.cmpi eq {index}, {bound}"));
        self.emit(format!("cf.cond_br {done} ^b{exit}, ^b{body} : (i1)"));
        let head_scope = self.scope.len();

        self.label(body, &[]);
        let count = 1 + self.rng.below(3);
        self.steps(depth + 1, count);
        let one = self.anonymous();
        let next = self.anonymous();
        self.emit(format!("{one} = arith.constant 1 : i128"));
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
            match self.rng.below(20) {
                0..=2 => self.binary(Ty::X),
                3 => self.binary(Ty::N),
                4 if self.profile.markers => self.binary(Ty::M),
                5 => {
                    let ty = self.carried_type();
                    if !matches!(ty, Ty::F(_)) {
                        self.constant(ty);
                    }
                }
                6 => {
                    self.condition();
                }
                7 | 8 => self.call(),
                9 => self.call_value(),
                10 => self.store(),
                11 => self.load(),
                12 | 13 => self.memory(),
                14 if depth < 2 && self.budget > 0 => {
                    self.budget -= 1;
                    self.diamond(depth);
                }
                15 if depth < 2 && self.budget > 0 => {
                    self.budget -= 1;
                    self.repeat(depth);
                }
                16 if depth < 2 && self.budget > 0 => {
                    self.budget -= 1;
                    self.switch(depth);
                }
                _ => self.binary(Ty::X),
            }
        }
    }

    fn effects_text(&self, sig: &Sig) -> String {
        let Some(effects) = &sig.effects else {
            return String::new();
        };
        let effects = effects
            .iter()
            .map(|(kind, on)| match on {
                On::Nothing => String::from(*kind),
                On::Arg(index) => format!("{kind}(arg {index})"),
                On::Result(index) => format!("{kind}(result {index})"),
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(" effects({effects})")
    }

    fn function(&mut self, me: usize) -> String {
        self.me = me;
        self.lines.clear();
        self.scope.clear();
        self.pointers.clear();
        self.values = 0;
        self.blocks = 0;
        self.budget = self.profile.constructs;
        self.calls = self.profile.calls;
        self.rec_fuel = None;
        let sig = self.sigs[me].clone();
        let effects = self.effects_text(&sig);
        let arrow = results_text(&self.shapes, &sig.results, false);
        if sig.external {
            return format!(
                "    builtin.function public extern(\"C\") @f{me}({}){arrow}{effects};",
                self.types(&sig.params)
            );
        }
        let mut fuel = None;
        let params = sig
            .params
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                let name = if index == 0 && sig.fuel {
                    let name = self.anonymous();
                    fuel = Some(name.clone());
                    name
                } else if *ty == Ty::P {
                    let name = self.fresh(*ty);
                    let (read, write) = sig.access[index];
                    self.pointers.push((name.clone(), read, write));
                    name
                } else {
                    self.fresh(*ty)
                };
                format!("{name}: {}", self.name(*ty))
            })
            .collect::<Vec<_>>()
            .join(", ");
        self.locals = (0..self.rng.below(5)).map(|_| self.carried_type()).collect();
        for index in 0..self.locals.len() {
            let value = self.operand(self.locals[index]);
            self.emit(format!("builtin.store_local {index}, {value}"));
        }
        let count = between(self.rng, self.profile.steps);
        if let Some(fuel) = fuel {
            let zero = self.anonymous();
            self.emit(format!("{zero} = arith.constant 0 : i64"));
            let done = self.anonymous();
            self.emit(format!("{done} = arith.cmpi eq {fuel}, {zero}"));
            let (base, rec) = (self.block(), self.block());
            self.emit(format!("cf.cond_br {done} ^b{base}, ^b{rec} : (i1)"));
            let mark = self.scope.len();
            self.label(base, &[]);
            self.steps(1, 1 + count / 2);
            self.ret();
            self.scope.truncate(mark);
            self.label(rec, &[]);
            let one = self.anonymous();
            self.emit(format!("{one} = arith.constant 1 : i64"));
            let less = self.anonymous();
            self.emit(format!("{less} = arith.sub {fuel}, {one}"));
            self.rec_fuel = Some(less);
            self.calls = self.calls.min(2);
            self.steps(1, 1 + count);
            // At least one call that recurses, when a pointer it needs is there.
            let (first, back, _) = self.group;
            let target = first + self.rng.below(back - first);
            self.call_function(target);
            self.ret();
        } else {
            self.steps(0, count);
            if self.profile.chain && me + 1 < self.sigs.len() {
                self.calls = 1;
                self.call_function(me + 1);
            }
            self.ret();
        }
        let locals = if self.locals.is_empty() {
            String::new()
        } else {
            format!(" locals({})", self.types(&self.locals))
        };
        format!(
            "    builtin.function public extern(\"C\") @f{me}({params}){arrow}{effects}{locals} \
             {{\n{}\n    }};",
            self.lines.join("\n")
        )
    }
}

/// A random signature of first-order types.
fn first_order(rng: &mut Rng, profile: &Profile, pointers: bool) -> Sig {
    let ty = |rng: &mut Rng| {
        let roll = rng.below(100);
        if roll < 45 {
            Ty::X
        } else if roll < 65 {
            Ty::N
        } else if roll < 85 && profile.markers {
            Ty::M
        } else if roll < 100 && pointers && profile.pointers {
            Ty::P
        } else {
            Ty::N
        }
    };
    let params = (0..rng.below(profile.max_params + 1)).map(|_| ty(rng)).collect::<Vec<_>>();
    let results = (0..rng.below(profile.max_results + 1))
        .map(|_| match ty(rng) {
            Ty::P => Ty::X,
            ty => ty,
        })
        .collect::<Vec<_>>();
    let access = params
        .iter()
        .map(|ty| {
            if *ty == Ty::P {
                match rng.below(3) {
                    0 => (true, false),
                    1 => (false, true),
                    _ => (true, true),
                }
            } else {
                (false, false)
            }
        })
        .collect();
    Sig {
        params,
        results,
        fuel: false,
        access,
        effects: None,
        external: false,
    }
}

/// The effects a function declares: what its pointers allow, and noise on its other parameters
/// and results (a declaration of more than the function does is still true).
fn declare(rng: &mut Rng, sig: &mut Sig) {
    if rng.chance(30) {
        return;
    }
    let mut effects = Vec::new();
    for (index, (read, write)) in sig.access.iter().enumerate() {
        if *read {
            effects.push(("read", On::Arg(index as u32)));
        }
        if *write {
            effects.push(("write", On::Arg(index as u32)));
        }
    }
    let noise = rng.below(4);
    for _ in 0..noise {
        let kind = ["read", "read", "write", "allocate", "free"][rng.below(5)];
        let on = match rng.below(5) {
            0 | 1 if !sig.params.is_empty() => On::Arg(rng.below(sig.params.len()) as u32),
            2 | 3 if !sig.results.is_empty() => On::Result(rng.below(sig.results.len()) as u32),
            4 if kind == "read" => On::Nothing,
            _ => continue,
        };
        if !effects.contains(&(kind, on)) {
            effects.push((kind, on));
        }
    }
    sig.effects = Some(effects);
}

pub(super) struct Generated {
    pub(super) source: String,
    pub(super) inputs: Vec<(String, Vec<Arg>)>,
    pub(super) model: Model,
}

/// A random module of `profile`.
pub(super) fn module(rng: &mut Rng, profile: &Profile) -> Generated {
    let front = between(rng, profile.front);
    let fuel = between(rng, profile.fuel);
    let back = between(rng, profile.back);
    let shape_count = between(rng, profile.shapes);
    let externs = between(rng, profile.externs);
    let shapes = (0..shape_count)
        .map(|_| {
            let limited = Profile {
                max_params: profile.max_params.min(4),
                max_results: profile.max_results.min(3),
                ..profile.clone()
            };
            first_order(rng, &limited, false)
        })
        .collect::<Vec<_>>();
    let group = (front, front + fuel, front + fuel + back);
    let count = group.2 + shape_count + externs;
    let mut sigs = Vec::new();
    for index in 0..count {
        let mut sig = if index >= group.2 && index < group.2 + shape_count {
            shapes[index - group.2].clone()
        } else {
            let external = index >= group.2 + shape_count;
            let mut sig = first_order(rng, profile, !external);
            sig.external = external;
            // Function values in the signatures of the functions ahead of the shape functions.
            if !external && shape_count > 0 {
                for param in 0..sig.params.len() {
                    if sig.params[param] != Ty::P && rng.chance(12) {
                        sig.params[param] = Ty::F(rng.below(shape_count));
                    }
                }
                for result in 0..sig.results.len() {
                    if rng.chance(12) {
                        sig.results[result] = Ty::F(rng.below(shape_count));
                    }
                }
            }
            if index >= group.0 && index < group.1 {
                sig.fuel = true;
                sig.params.insert(0, Ty::N);
                sig.access.insert(0, (false, false));
            }
            sig
        };
        if profile.chain {
            sig.params.retain(|ty| *ty != Ty::P);
            sig.access = sig.params.iter().map(|_| (false, false)).collect();
        }
        declare(rng, &mut sig);
        sigs.push(sig);
    }
    // The first function takes an `i128`, so that every module holds one.
    if sigs[0].params.is_empty() || !sigs[0].params.contains(&Ty::X) {
        sigs[0].params.push(Ty::X);
        sigs[0].access.push((false, false));
    }
    let mut aliases: Vec<(String, String, usize)> = Vec::new();
    for target in 1..count {
        if rng.chance(if count > 20 { 5 } else { 30 }) {
            aliases.push((format!("a{target}"), format!("f{target}"), target));
            if rng.chance(40) {
                aliases.push((format!("aa{target}"), format!("a{target}"), target));
            }
        }
    }
    let mut inputs = Vec::new();
    let definitions = sigs.iter().filter(|sig| !sig.external).count();
    for (index, sig) in sigs.iter().enumerate() {
        if sig.external {
            continue;
        }
        if index > 2
            && definitions > profile.input_functions
            && rng.below(definitions) >= profile.input_functions
        {
            continue;
        }
        for _ in 0..profile.inputs {
            let args = sig
                .params
                .iter()
                .enumerate()
                .map(|(param, ty)| match ty {
                    Ty::N if param == 0 && sig.fuel => Arg::Int(rng.below(4) as u128),
                    Ty::P => Arg::Ptr(rng.value(64)),
                    Ty::F(shape) => Arg::Func(format!("f{}", group.2 + shape)),
                    ty => Arg::Int(rng.value(width(*ty))),
                })
                .collect();
            inputs.push((format!("f{index}"), args));
        }
    }
    let mut generator = Gen {
        rng,
        profile: profile.clone(),
        sigs,
        shapes,
        group,
        aliases: aliases.clone(),
        me: 0,
        lines: Vec::new(),
        scope: Vec::new(),
        pointers: Vec::new(),
        values: 0,
        blocks: 0,
        locals: Vec::new(),
        budget: 0,
        calls: 0,
        rec_fuel: None,
    };
    let mut text = String::from("builtin.module public @m {\n");
    let mut locals = Vec::new();
    for me in 0..count {
        text.push_str(&generator.function(me));
        text.push('\n');
        locals.push(if generator.sigs[me].external {
            Vec::new()
        } else {
            generator.locals.clone()
        });
    }
    for (alias, target, _) in &aliases {
        text.push_str(&format!("    builtin.function_alias public @{alias} -> @{target};\n"));
    }
    text.push_str("};");
    let model = Model {
        sigs: generator.sigs.clone(),
        shapes: generator.shapes.clone(),
        locals,
    };
    Generated {
        source: text,
        inputs,
        model,
    }
}
