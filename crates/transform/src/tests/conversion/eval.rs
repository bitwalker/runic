//! An evaluator of a module, over the IR itself: integers of up to 128 bits (`isize` is 64 bits),
//! a call stack over the module's functions (through aliases), function values for
//! `builtin.function_ref` and `builtin.call_indirect`, block arguments for `cf.br` and
//! `cf.cond_br`, and locals. A bound on the depth of calls and on the operations executed makes
//! every evaluation end.

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use runic_core::{
    BlockRef, CallOpInterface, Callable, Context, FxHashMap, Int, Operation, OperationRef,
    Overflow, Type, Value, ValueRef,
    dialects::builtin::{
        Call, CallIndirect, Function, FunctionAlias, FunctionReference, LoadLocal, Ret, RetImm,
        StoreLocal,
    },
};
use runic_dialect_arith as arith;
use runic_dialect_cf as cf;

/// The deepest call the evaluator makes.
const MAX_DEPTH: usize = 32;
/// The most operations one evaluation executes.
const MAX_STEPS: usize = 100_000;

/// A value: the bits of an integer, or a function, by the name it was referenced with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Val {
    Int(u128),
    Func(String),
}

pub(super) struct Evaluator {
    functions: BTreeMap<String, OperationRef>,
    aliases: BTreeMap<String, String>,
    steps: usize,
}

/// The width of a value of type `ty`, if it is an integer: `isize` is 64 bits.
fn width(context: &Context, ty: Type) -> Option<u32> {
    if ty == context.isize() {
        Some(64)
    } else {
        Int::width_for_type(ty)
    }
}

fn mask(value: u128, width: u32) -> u128 {
    super::mask(value, width)
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
    /// An evaluator of the functions and aliases directly in `module`.
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
            steps: 0,
        }
    }

    /// The function `name` names, through any chain of aliases.
    fn resolve(&self, name: &str) -> Result<OperationRef, String> {
        let mut name = name;
        for _ in 0..16 {
            if let Some(function) = self.functions.get(name) {
                return Ok(*function);
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
        let function = self.resolve(name)?;
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
        self.run(name, args, 0)
    }

    fn run(&mut self, name: &str, args: Vec<Val>, depth: usize) -> Result<Vec<Val>, String> {
        if depth > MAX_DEPTH {
            return Err(String::from("the calls are nested too deeply"));
        }
        let function = self.resolve(name)?;
        let context = function.borrow().context_rc();
        let entry = function
            .borrow()
            .regions()
            .front()
            .get()
            .and_then(|body| body.entry_block_ref())
            .ok_or_else(|| format!("@{name} is a declaration"))?;
        let mut env = FxHashMap::<ValueRef, Val>::default();
        let mut locals = BTreeMap::<u32, Val>::new();
        bind(&context, entry, &args, &mut env)?;
        let mut block = entry;
        loop {
            let ops =
                block.borrow().body().iter().map(|op| op.as_operation_ref()).collect::<Vec<_>>();
            let mut next = None;
            for op in ops {
                self.steps += 1;
                if self.steps > MAX_STEPS {
                    return Err(String::from("the evaluation ran too long"));
                }
                match self.step(&context, op, &mut env, &mut locals, depth)? {
                    Flow::Continue => {}
                    Flow::Branch(dest, values) => {
                        bind(&context, dest, &values, &mut env)?;
                        next = Some(dest);
                        break;
                    }
                    Flow::Return(values) => return Ok(values),
                }
            }
            block = next.ok_or_else(|| String::from("a block does not end in a terminator"))?;
        }
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
            Val::Func(_) => Err(format!("operand {index} of '{name}' is a function")),
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

        let mut results = Vec::new();
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
        } else if op.is::<cf::Br>() || op.is::<cf::CondBr>() {
            let index = if op.is::<cf::CondBr>() && int(0)? == 0 {
                1
            } else {
                0
            };
            let successor = op.successor(index);
            let mut values = Vec::new();
            for argument in successor.arguments.iter() {
                let value = argument.borrow().as_value_ref();
                values.push(env.get(&value).cloned().ok_or_else(|| {
                    format!("'{name}' forwards {} before it is defined", value.borrow())
                })?);
            }
            return Ok(Flow::Branch(successor.successor(), values));
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
            results = self.run(&callee, args, depth + 1)?;
            return self.finish(op_ref, results, env);
        } else if op.is::<CallIndirect>() {
            let Val::Func(callee) = &operands[0].1 else {
                return Err(format!("the callee of '{name}' is not a function"));
            };
            let callee = callee.clone();
            let args = operands.into_iter().skip(1).map(|(_, value)| value).collect();
            drop(op);
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

enum Flow {
    Continue,
    Branch(BlockRef, Vec<Val>),
    Return(Vec<Val>),
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
            (None, Val::Func(_)) => {}
            _ => return Err(format!("a block argument of type {ty} is given {value:?}")),
        }
        env.insert(argument, value.clone());
    }
    Ok(())
}
