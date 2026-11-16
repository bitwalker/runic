//! The evaluator of `conversion::eval`, with more: memory cells behind `ptr<i64>` values
//! (`mem.load`, `mem.store`), external functions (declarations) with a fixed meaning stated over
//! their original signature, `cf.switch`, a marker type whose values vanish, the structured
//! operations `cfg_to_scf` makes, and bounds that can be set.

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use runic_dialect_arith as arith;
use runic_dialect_cf as cf;
use runic_dialect_mem as mem;
use runic_ir::{
    BlockRef, CallOpInterface, Callable, Context, FxHashMap, Int, Operation, OperationRef,
    Overflow, Symbol, Type, Value, ValueRef,
    dialects::builtin::{
        Call, CallIndirect, Function, FunctionAlias, FunctionReference, LoadLocal, Ret, RetImm,
        StoreLocal,
    },
};

/// A value: the bits of an integer, a function by the name it was referenced with, or a pointer
/// to a memory cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Val {
    Int(u128),
    Func(String),
    Ptr(usize),
}

pub(super) struct Evaluator {
    functions: BTreeMap<String, OperationRef>,
    aliases: BTreeMap<String, String>,
    /// The original parameter and result types of each declaration.
    pub(super) externs: BTreeMap<String, (Vec<Type>, Vec<Type>)>,
    pub(super) memory: Vec<u128>,
    steps: usize,
    pub(super) max_depth: usize,
    pub(super) max_steps: usize,
    /// The number of `builtin.call`/`call_indirect` executed since the last [Self::call].
    pub(super) calls: usize,
}

/// The message of the bound on depth.
pub(super) const TOO_DEEP: &str = "the calls are nested too deeply";
/// The message of the bound on steps.
pub(super) const TOO_LONG: &str = "the evaluation ran too long";

/// The width of a value of type `ty`, if it is an integer: `isize` is 64 bits.
fn width(context: &Context, ty: Type) -> Option<u32> {
    if ty == context.isize() {
        Some(64)
    } else {
        Int::width_for_type(ty)
    }
}

fn mask(value: u128, width: u32) -> u128 {
    super::super::mask(value, width)
}

/// `value`, of `width` bits, read as a signed number.
fn signed(value: u128, width: u32) -> i128 {
    if width >= 128 {
        value as i128
    } else if value >> (width - 1) & 1 == 1 {
        (value | !((1u128 << width) - 1)) as i128
    } else {
        value as i128
    }
}

impl Evaluator {
    /// An evaluator of the functions and aliases under `module`.
    pub(super) fn new(module: OperationRef) -> Self {
        let mut functions = BTreeMap::new();
        let mut aliases = BTreeMap::new();
        module.borrow().prewalk_all(|op: &Operation| {
            if let Some(function) = op.downcast_ref::<Function>() {
                functions.insert(function.name().as_str().to_string(), op.as_operation_ref());
            } else if let Some(alias) = op.downcast_ref::<FunctionAlias>() {
                let target = alias.target().path().name().as_str().to_string();
                aliases.insert(alias.name().as_str().to_string(), target);
            }
        });
        Self {
            functions,
            aliases,
            externs: BTreeMap::new(),
            memory: Vec::new(),
            steps: 0,
            max_depth: 32,
            max_steps: 100_000,
            calls: 0,
        }
    }

    /// The parameter and result types of every declaration, to be given to the evaluator of the
    /// converted module.
    pub(super) fn declarations(&self) -> BTreeMap<String, (Vec<Type>, Vec<Type>)> {
        let mut found = BTreeMap::new();
        for (name, function) in &self.functions {
            let function = function.borrow();
            let function = function.downcast_ref::<Function>().unwrap();
            if function.is_declaration() {
                found.insert(name.clone(), self.signature(name).unwrap());
            }
        }
        found
    }

    /// The function `name` names, through any chain of aliases, and its own name.
    fn resolve(&self, name: &str) -> Result<(OperationRef, String), String> {
        let mut name = name;
        for _ in 0..16 {
            if let Some(function) = self.functions.get(name) {
                return Ok((*function, name.to_string()));
            }
            match self.aliases.get(name) {
                Some(target) => name = target,
                None => return Err(format!("@{name} names no function")),
            }
        }
        Err(format!("the aliases of @{name} do not end"))
    }

    /// The parameter and result types of the function `name`.
    pub(super) fn signature(&self, name: &str) -> Result<(Vec<Type>, Vec<Type>), String> {
        let (function, _) = self.resolve(name)?;
        let function = function.borrow();
        let function = function.downcast_ref::<Function>().unwrap();
        let signature = function.get_signature();
        Ok((
            signature.params().iter().map(|param| param.ty).collect(),
            signature.results().iter().map(|result| result.ty).collect(),
        ))
    }

    /// Calls the function `name` with `args`, starting a new count of steps.
    pub(super) fn call(&mut self, name: &str, args: Vec<Val>) -> Result<Vec<Val>, String> {
        self.steps = 0;
        self.calls = 0;
        self.run(name, args, 0)
    }

    /// What a declaration computes, stated over its original signature: result `j` is a fold of
    /// the integer arguments (an `i128` whole, a marker left out), cut to the result's width.
    fn external(
        &self,
        context: &Context,
        name: &str,
        current: &(Vec<Type>, Vec<Type>),
        args: &[Val],
    ) -> Result<Vec<Val>, String> {
        let Some((params, results)) = self.externs.get(name) else {
            return Err(format!("@{name} is a declaration whose original signature is unknown"));
        };
        let flattened = current.0 != *params || current.1 != *results;
        let (wide, marker) = (context.i128(), context.i8());
        let mut args = args.iter();
        let mut next = || args.next().ok_or_else(|| format!("@{name}: too few arguments"));
        let int = |value: &Val| match value {
            Val::Int(value) => Some(*value),
            _ => None,
        };
        let mut ints = Vec::new();
        for ty in params {
            if *ty == marker {
                if !flattened {
                    next()?;
                }
            } else if *ty == wide && flattened {
                let low = int(next()?).ok_or("a half is no integer")?;
                let high = int(next()?).ok_or("a half is no integer")?;
                ints.push(low | high << 64);
            } else if let Some(value) = int(next()?) {
                ints.push(value);
            }
        }
        if args.next().is_some() {
            return Err(format!("@{name}: too many arguments"));
        }
        let mut values = Vec::new();
        for (index, ty) in results.iter().enumerate() {
            let mut acc = index as u128 + 0x9e37_79b9;
            for value in &ints {
                acc = acc.wrapping_mul(0x1_0000_0000_01b3).wrapping_add(*value) ^ (acc >> 61);
            }
            if *ty == marker {
                if !flattened {
                    values.push(Val::Int(0));
                }
            } else if *ty == wide && flattened {
                values.push(Val::Int(mask(acc, 64)));
                values.push(Val::Int(acc >> 64));
            } else {
                let width = width(context, *ty).ok_or("an external result is no integer")?;
                values.push(Val::Int(mask(acc, width)));
            }
        }
        Ok(values)
    }

    fn run(&mut self, name: &str, args: Vec<Val>, depth: usize) -> Result<Vec<Val>, String> {
        if depth > self.max_depth {
            return Err(String::from(TOO_DEEP));
        }
        let (function, own_name) = self.resolve(name)?;
        let context = function.borrow().context_rc();
        let entry = function
            .borrow()
            .regions()
            .front()
            .get()
            .and_then(|body| body.entry_block_ref());
        let Some(entry) = entry else {
            let current = self.signature(&own_name)?;
            return self.external(&context, &own_name, &current, &args);
        };
        let mut env = FxHashMap::<ValueRef, Val>::default();
        let mut locals = BTreeMap::<u32, Val>::new();
        bind(&context, entry, &args, &mut env)?;
        match self.run_blocks(&context, entry, &mut env, &mut locals, depth)? {
            Flow::Return(values) => Ok(values),
            _ => Err(String::from("a function body ended without a return")),
        }
    }

    /// Runs the blocks of a region from `entry` until an operation leaves the region.
    fn run_blocks(
        &mut self,
        context: &Context,
        entry: BlockRef,
        env: &mut FxHashMap<ValueRef, Val>,
        locals: &mut BTreeMap<u32, Val>,
        depth: usize,
    ) -> Result<Flow, String> {
        let mut block = entry;
        loop {
            let ops =
                block.borrow().body().iter().map(|op| op.as_operation_ref()).collect::<Vec<_>>();
            let mut next = None;
            for op in ops {
                self.steps += 1;
                if self.steps > self.max_steps {
                    return Err(String::from(TOO_LONG));
                }
                match self.step(context, op, env, locals, depth)? {
                    Flow::Continue => {}
                    Flow::Branch(dest, values) => {
                        bind(context, dest, &values, env)?;
                        next = Some(dest);
                        break;
                    }
                    flow => return Ok(flow),
                }
            }
            block = next.ok_or_else(|| String::from("a block does not end in a terminator"))?;
        }
    }

    /// Runs region `index` of `op` with `args` bound to its entry block (`region` is `(index,
    /// args)`).
    fn run_region(
        &mut self,
        context: &Context,
        op: OperationRef,
        (index, args): (usize, &[Val]),
        env: &mut FxHashMap<ValueRef, Val>,
        locals: &mut BTreeMap<u32, Val>,
        depth: usize,
    ) -> Result<Flow, String> {
        let entry = op
            .borrow()
            .regions()
            .iter()
            .nth(index)
            .and_then(|region| region.entry_block_ref())
            .ok_or_else(|| format!("region {index} of '{}' has no block", op.borrow().name()))?;
        bind(context, entry, args, env)?;
        self.run_blocks(context, entry, env, locals, depth)
    }

    /// Executes `op`.
    fn step(
        &mut self,
        context: &Context,
        op_ref: OperationRef,
        env: &mut FxHashMap<ValueRef, Val>,
        locals: &mut BTreeMap<u32, Val>,
        depth: usize,
    ) -> Result<Flow, String> {
        let op = op_ref.borrow();
        let name = op.name();
        let mut operands = Vec::with_capacity(op.num_operands());
        for operand in op.operands().iter() {
            let value = operand.borrow().as_value_ref();
            let Some(held) = env.get(&value) else {
                return Err(format!("'{name}' uses {} before it is defined", value.borrow()));
            };
            operands.push((value.borrow().ty(), held.clone()));
        }
        let int = |index: usize| match &operands[index].1 {
            Val::Int(value) => Ok(*value),
            other => Err(format!("operand {index} of '{name}' is {other:?}")),
        };
        let operand_width = |index: usize| {
            width(context, operands[index].0)
                .ok_or_else(|| format!("operand {index} of '{name}' is not an integer"))
        };
        let result_width = |index: usize| {
            let ty = op.results()[index].borrow().ty();
            width(context, ty)
                .ok_or_else(|| format!("result {index} of '{name}' is not an integer"))
        };
        let wrapping = |overflow: &Overflow| {
            if *overflow == Overflow::Wrapping {
                Ok(())
            } else {
                Err(format!("'{name}' does not wrap"))
            }
        };
        let forwarded = |index: usize| -> Result<Flow, String> {
            let successor = op.successor(index);
            let mut values = Vec::new();
            for argument in successor.arguments.iter() {
                let value = argument.borrow().as_value_ref();
                values.push(env.get(&value).cloned().ok_or_else(|| {
                    format!("'{name}' forwards {} before it is defined", value.borrow())
                })?);
            }
            Ok(Flow::Branch(successor.successor(), values))
        };

        let mut results = Vec::new();
        let dialect_op = name.to_string();
        if let Some(constant) = op.downcast_ref::<arith::Constant>() {
            results.push(Val::Int(mask(constant.get_value().unsigned(), result_width(0)?)));
        } else if let Some(add) = op.downcast_ref::<arith::Add>() {
            wrapping(&add.get_overflow())?;
            results.push(Val::Int(mask(int(0)?.wrapping_add(int(1)?), result_width(0)?)));
        } else if let Some(sub) = op.downcast_ref::<arith::Sub>() {
            wrapping(&sub.get_overflow())?;
            results.push(Val::Int(mask(int(0)?.wrapping_sub(int(1)?), result_width(0)?)));
        } else if let Some(mul) = op.downcast_ref::<arith::Mul>() {
            wrapping(&mul.get_overflow())?;
            results.push(Val::Int(mask(int(0)?.wrapping_mul(int(1)?), result_width(0)?)));
        } else if op.is::<arith::Band>() {
            results.push(Val::Int(int(0)? & int(1)?));
        } else if op.is::<arith::Bor>() {
            results.push(Val::Int(int(0)? | int(1)?));
        } else if op.is::<arith::Bxor>() {
            results.push(Val::Int(int(0)? ^ int(1)?));
        } else if op.is::<arith::Shl>() || op.is::<arith::Shr>() {
            let (value, amount, width) = (int(0)?, int(1)?, operand_width(0)?);
            if amount >= u128::from(width) {
                return Err(format!("'{name}' shifts by {amount}, at least its width"));
            }
            let amount = amount as u32;
            let shifted = if op.is::<arith::Shl>() {
                value << amount
            } else {
                value >> amount
            };
            results.push(Val::Int(mask(shifted, width)));
        } else if let Some(cmpi) = op.downcast_ref::<arith::Cmpi>() {
            let (a, b, width) = (int(0)?, int(1)?, operand_width(0)?);
            let (sa, sb) = (signed(a, width), signed(b, width));
            use arith::CmpPredicate::*;
            let holds = match *cmpi.get_predicate() {
                Eq => a == b,
                Ne => a != b,
                Ult => a < b,
                Ule => a <= b,
                Ugt => a > b,
                Uge => a >= b,
                Slt => sa < sb,
                Sle => sa <= sb,
                Sgt => sa > sb,
                Sge => sa >= sb,
            };
            results.push(Val::Int(u128::from(holds)));
        } else if op.is::<arith::Zext>() || op.is::<arith::Trunc>() || op.is::<arith::ZextOrTrunc>()
        {
            results.push(Val::Int(mask(int(0)?, result_width(0)?)));
        } else if op.is::<arith::Sext>() || op.is::<arith::SextOrTrunc>() {
            let value = signed(int(0)?, operand_width(0)?) as u128;
            results.push(Val::Int(mask(value, result_width(0)?)));
        } else if op.is::<cf::Select>() {
            results.push(if int(0)? != 0 {
                operands[1].1.clone()
            } else {
                operands[2].1.clone()
            });
        } else if op.is::<cf::Br>() || op.is::<cf::CondBr>() {
            let index = if op.is::<cf::CondBr>() && int(0)? == 0 {
                1
            } else {
                0
            };
            return forwarded(index);
        } else if let Some(switch) = op.downcast_ref::<cf::Switch>() {
            let selector = int(0)? as u32;
            let mut index = op.num_successors() - 1;
            for (case, switch_case) in switch.cases().iter().enumerate() {
                if *switch_case.key() as u32 == selector {
                    index = case;
                    break;
                }
            }
            return forwarded(index);
        } else if op.is::<Ret>() {
            return Ok(Flow::Return(operands.into_iter().map(|(_, value)| value).collect()));
        } else if let Some(ret) = op.downcast_ref::<RetImm>() {
            return Ok(Flow::Return(vec![Val::Int(ret.get_value().unsigned())]));
        } else if let Some(reference) = op.downcast_ref::<FunctionReference>() {
            results.push(Val::Func(reference.callee().path().name().as_str().to_string()));
        } else if let Some(call) = op.downcast_ref::<Call>() {
            let Callable::Symbol(path) = call.callable_for_callee() else {
                return Err(format!("'{name}' has no symbol callee"));
            };
            let args = operands.into_iter().map(|(_, value)| value).collect();
            let callee = path.name().as_str().to_string();
            drop(op);
            self.calls += 1;
            results = self.run(&callee, args, depth + 1)?;
            return self.finish(op_ref, results, env);
        } else if op.is::<CallIndirect>() {
            let Val::Func(callee) = &operands[0].1 else {
                return Err(format!("the callee of '{name}' is not a function"));
            };
            let callee = callee.clone();
            let args = operands.into_iter().skip(1).map(|(_, value)| value).collect();
            drop(op);
            self.calls += 1;
            results = self.run(&callee, args, depth + 1)?;
            return self.finish(op_ref, results, env);
        } else if let Some(load) = op.downcast_ref::<LoadLocal>() {
            let index = load.get_local().index();
            let value = locals
                .get(&index)
                .cloned()
                .ok_or_else(|| format!("local {index} is read unset"))?;
            results.push(value);
        } else if let Some(store) = op.downcast_ref::<StoreLocal>() {
            locals.insert(store.get_local().index(), operands[0].1.clone());
        } else if op.is::<mem::Load>() {
            let Val::Ptr(cell) = &operands[0].1 else {
                return Err(format!("'{name}' reads through {:?}", operands[0].1));
            };
            results.push(Val::Int(mask(self.memory[*cell], result_width(0)?)));
        } else if op.is::<mem::Store>() {
            let Val::Ptr(cell) = &operands[1].1 else {
                return Err(format!("'{name}' writes through {:?}", operands[1].1));
            };
            self.memory[*cell] = int(0)?;
        } else if dialect_op == "scf.if" {
            let region = if int(0)? != 0 { 0 } else { 1 };
            drop(op);
            return match self.run_region(context, op_ref, (region, &[]), env, locals, depth)? {
                Flow::Yield(values) => self.finish(op_ref, values, env),
                Flow::Return(values) => Ok(Flow::Return(values)),
                _ => Err(String::from("an scf.if region ended oddly")),
            };
        } else if dialect_op == "scf.index_switch" {
            let selector = int(0)? as u32;
            let (region, count) = {
                let count = op.regions().iter().count();
                let mut region = 0;
                let keys = scf_switch_keys(&op)?;
                for (index, key) in keys.iter().enumerate() {
                    if *key == selector {
                        region = index + 1;
                    }
                }
                (region, count)
            };
            let _ = count;
            drop(op);
            return match self.run_region(context, op_ref, (region, &[]), env, locals, depth)? {
                Flow::Yield(values) => self.finish(op_ref, values, env),
                Flow::Return(values) => Ok(Flow::Return(values)),
                _ => Err(String::from("an scf.index_switch region ended oddly")),
            };
        } else if dialect_op == "scf.while" {
            let mut carried = operands.iter().map(|(_, value)| value.clone()).collect::<Vec<_>>();
            drop(op);
            loop {
                self.steps += 1;
                if self.steps > self.max_steps {
                    return Err(String::from(TOO_LONG));
                }
                match self.run_region(context, op_ref, (0, &carried), env, locals, depth)? {
                    Flow::Condition(true, values) => {
                        match self.run_region(context, op_ref, (1, &values), env, locals, depth)? {
                            Flow::Yield(values) => carried = values,
                            Flow::Return(values) => return Ok(Flow::Return(values)),
                            _ => return Err(String::from("an scf.while body ended oddly")),
                        }
                    }
                    Flow::Condition(false, values) => return self.finish(op_ref, values, env),
                    Flow::Return(values) => return Ok(Flow::Return(values)),
                    _ => return Err(String::from("an scf.while head ended oddly")),
                }
            }
        } else if dialect_op == "scf.yield" {
            return Ok(Flow::Yield(operands.into_iter().map(|(_, value)| value).collect()));
        } else if dialect_op == "scf.condition" {
            let condition = int(0)? != 0;
            return Ok(Flow::Condition(
                condition,
                operands.into_iter().skip(1).map(|(_, value)| value).collect(),
            ));
        } else if dialect_op == "ub.poison" {
            // A poison value is never observed by a program that evaluated before the pass.
            let ty = op.results()[0].borrow().ty();
            results.push(match width(context, ty) {
                Some(_) => Val::Int(0),
                None => Val::Func(String::from("<poison>")),
            });
        } else {
            return Err(format!("the evaluator does not know '{name}'"));
        }
        drop(op);
        self.finish(op_ref, results, env)
    }

    /// Binds the results of `op` to `values`.
    fn finish(
        &self,
        op: OperationRef,
        values: Vec<Val>,
        env: &mut FxHashMap<ValueRef, Val>,
    ) -> Result<Flow, String> {
        let op = op.borrow();
        if values.len() != op.num_results() {
            return Err(format!(
                "'{}' has {} results, but {} values were given",
                op.name(),
                op.num_results(),
                values.len()
            ));
        }
        for (result, value) in op.results().iter().zip(values) {
            env.insert(result.borrow().as_value_ref(), value);
        }
        Ok(Flow::Continue)
    }
}

/// The keys of an `scf.index_switch`, one per region but the last (the default).
fn scf_switch_keys(op: &Operation) -> Result<Vec<u32>, String> {
    let text = op.to_string();
    let _ = text;
    let switch = op
        .downcast_ref::<runic_dialect_scf::IndexSwitch>()
        .ok_or_else(|| String::from("not an scf.index_switch"))?;
    Ok(switch.get_cases().iter().map(|key| *key as u32).collect())
}

enum Flow {
    Continue,
    Branch(BlockRef, Vec<Val>),
    Return(Vec<Val>),
    Yield(Vec<Val>),
    Condition(bool, Vec<Val>),
}

/// Binds the arguments of `block` to `values`, checking that each integer fits its type.
fn bind(
    context: &Context,
    block: BlockRef,
    values: &[Val],
    env: &mut FxHashMap<ValueRef, Val>,
) -> Result<(), String> {
    let arguments = block.borrow().argument_values().collect::<Vec<_>>();
    if arguments.len() != values.len() {
        return Err(format!(
            "a block of {} arguments is given {} values",
            arguments.len(),
            values.len()
        ));
    }
    for (argument, value) in arguments.into_iter().zip(values) {
        let ty = argument.borrow().ty();
        match (width(context, ty), value) {
            (Some(width), Val::Int(bits)) if mask(*bits, width) == *bits => {}
            (None, Val::Func(_) | Val::Ptr(_)) => {}
            _ => return Err(format!("a block argument of type {ty} is given {value:?}")),
        }
        env.insert(argument, value.clone());
    }
    Ok(())
}
