//! An evaluator of lowered functions, written from the WebAssembly specification (release 3.0,
//! "Numerics" and "Execution") and independent of the patterns of the crate.
//!
//! [run] executes one `builtin.function` whose body is `wasm` operations and ends in
//! `builtin.ret`, on argument values, and gives the values it returns or [Outcome::Trap]. A value
//! is its bit pattern in a `u64`, the width of its type (32 or 64) and whether it is a float:
//! [Val].
//!
//! The operations it knows are the integer and float operations of the `wasm` dialect that
//! lowered numeric code uses, the structured control of "Control Instructions" (`wasm.block`,
//! `wasm.loop`, `wasm.if`, `wasm.end`, `wasm.br`, `wasm.br_table`, `wasm.unreachable`) and
//! `builtin.ret` at any depth. An operation it does not know panics with a message that names it:
//! a test never skips one silently.
//!
//! **Control.** A `block`, a `loop` and an `if` run their body (for an `if`, the region the
//! condition picks: the first when it is not zero) on their parameters. Falling off the `end`
//! gives the results. A `br L` unwinds `L` enclosing constructs and then targets the next: a
//! `block` or an `if` is left with the branch's values as its results; a `loop` runs its body
//! again with them as its parameters. `br_table` takes the region of its index (entries in order,
//! and the default for an index at or past their number, the index read unsigned), whose one
//! `br` is the branch taken; the table itself is not a label. A `builtin.ret` at any depth leaves
//! the function. Every operation run spends one unit of fuel ([FUEL]), and running out panics, so
//! an endless loop is a test failure.
//!
//! Floats follow the specification's "Floating-Point Operations": IEEE 754 arithmetic rounding to
//! nearest, ties to even, which Rust's `f32` and `f64` arithmetic and its `as` conversions between
//! integers and floats implement; the saturating truncations are Rust's `as` from a float to an
//! integer, which saturates and gives 0 for a NaN, as `trunc_sat` does. Where the specification
//! leaves a NaN's sign and payload to the engine, the bits given here are Rust's, and a test
//! compares a NaN result only as being a NaN ([Val::is_nan]).

use alloc::{format, string::String, vec::Vec};
use std::collections::HashMap;

use runic_dialect_wasm as wasm;
use runic_ir::{
    BlockRef, OperationRef, RegionRef, Type, Value, ValueRef, dialects::builtin::Function,
};

/// A Wasm number: the bits of an `i32`, an `i64`, an `f32` or an `f64` in the low `width` bits of
/// `bits`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Val {
    /// The bit pattern; the bits at or above `width` are zero.
    pub bits: u64,
    /// 32 or 64.
    pub width: u32,
    /// Whether the value is a float (`f32`, `f64`) rather than an integer.
    pub float: bool,
}

impl Val {
    /// The integer of the low `width` bits of `bits`.
    pub fn new(bits: u64, width: u32) -> Self {
        assert!(width == 32 || width == 64, "a Wasm integer is 32 or 64 bits wide");
        Self {
            bits: bits & mask(width),
            width,
            float: false,
        }
    }

    /// The float of the low `width` bits of `bits`.
    pub fn float_bits(bits: u64, width: u32) -> Self {
        Self {
            float: true,
            ..Self::new(bits, width)
        }
    }

    /// The `i32` `bits`.
    pub fn i32(bits: u32) -> Self {
        Self::new(u64::from(bits), 32)
    }

    /// The `i64` `bits`.
    pub fn i64(bits: u64) -> Self {
        Self::new(bits, 64)
    }

    /// The `f32` `value`.
    pub fn f32(value: f32) -> Self {
        Self::float_bits(u64::from(value.to_bits()), 32)
    }

    /// The `f64` `value`.
    pub fn f64(value: f64) -> Self {
        Self::float_bits(value.to_bits(), 64)
    }

    /// Whether the value is a float NaN.
    pub fn is_nan(self) -> bool {
        self.float
            && if self.width == 32 {
                self.as_f32().is_nan()
            } else {
                self.as_f64().is_nan()
            }
    }

    /// The value read as a signed number.
    fn signed(self) -> i64 {
        if self.width == 32 {
            i64::from(self.bits as u32 as i32)
        } else {
            self.bits as i64
        }
    }

    /// The bits read as an `f32`.
    fn as_f32(self) -> f32 {
        f32::from_bits(self.bits as u32)
    }

    /// The bits read as an `f64`.
    fn as_f64(self) -> f64 {
        f64::from_bits(self.bits)
    }

    /// The float value, widened to an `f64` (exactly, for an `f32`).
    fn as_float(self) -> f64 {
        if self.width == 32 {
            f64::from(self.as_f32())
        } else {
            self.as_f64()
        }
    }
}

/// The width of the Wasm number type `ty` and whether it is a float, or `None` for any other type.
fn number_type(ty: Type) -> Option<(u32, bool)> {
    match ty.integer_width() {
        Some(width @ (32 | 64)) => Some((width, false)),
        Some(_) => None,
        None => match ty.as_float().map(|float| float.bits()) {
            Some(width @ (32 | 64)) => Some((width, true)),
            _ => None,
        },
    }
}

/// What running a function gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    /// The values the function returned.
    Values(Vec<Val>),
    /// A trap: a division by zero, a signed overflow in a division, or `wasm.unreachable`.
    Trap,
}

/// All ones in `width` bits.
fn mask(width: u32) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// The operations a run may execute before it panics: an endless loop is a failure, not a hang.
pub(super) const FUEL: u64 = 1_000_000;

/// What a body gave when it stopped.
enum Flow {
    /// A `wasm.end` with these values.
    End(Vec<Val>),
    /// A `wasm.br` to the label `depth` (0 is the innermost construct around the body that
    /// stopped) with these values.
    Br(u32, Vec<Val>),
    /// A `builtin.ret` with these values.
    Return(Vec<Val>),
    /// A trap.
    Trap,
}

/// The values of the SSA values of the function being run.
struct Env {
    values: HashMap<ValueRef, Val>,
    /// The operations left to run.
    fuel: u64,
}

impl Env {
    fn get(&self, value: ValueRef) -> Val {
        *self.values.get(&value).unwrap_or_else(|| {
            panic!("the evaluator read `{}` before it was defined", value.borrow())
        })
    }

    fn operands(&self, op: OperationRef) -> Vec<Val> {
        let op = op.borrow();
        op.operands()
            .iter()
            .map(|operand| self.get(operand.borrow().as_value_ref()))
            .collect()
    }

    fn define(&mut self, op: OperationRef, values: &[Val]) {
        let op = op.borrow();
        assert_eq!(
            op.results().len(),
            values.len(),
            "`{}` has {} results",
            op.name(),
            values.len()
        );
        for (result, value) in op.results().iter().zip(values) {
            self.values.insert(result.borrow().as_value_ref(), *value);
        }
    }
}

/// Runs the function `function`, a `builtin.function` in Wasm form, on `arguments`.
///
/// Panics for an operation the evaluator does not know, naming it, and when `arguments` do not
/// match the parameters in number or width.
pub(super) fn run(function: OperationRef, arguments: &[Val]) -> Outcome {
    let body = {
        let function = function.borrow();
        let function = function.downcast_ref::<Function>().expect("a builtin.function");
        function.body().entry_block_ref().expect("a function with a body")
    };
    let mut env = Env {
        values: HashMap::new(),
        fuel: FUEL,
    };
    bind_arguments(&mut env, body, arguments);
    match run_block(&mut env, body) {
        Flow::Return(values) => Outcome::Values(values),
        Flow::Trap => Outcome::Trap,
        Flow::End(_) => panic!("the body of a function ended in `wasm.end`"),
        Flow::Br(depth, _) => panic!("a `wasm.br` left the function, {depth} labels past it"),
    }
}

/// Defines the arguments of `block` as `values`.
fn bind_arguments(env: &mut Env, block: BlockRef, values: &[Val]) {
    let arguments = block.borrow().argument_values().collect::<Vec<_>>();
    assert_eq!(arguments.len(), values.len(), "the block takes {} arguments", arguments.len());
    for (argument, value) in arguments.into_iter().zip(values) {
        let ty = argument.borrow().ty();
        assert_eq!(
            number_type(ty),
            Some((value.width, value.float)),
            "the argument `{}` of type `{ty}` was given {value:?}",
            argument.borrow(),
        );
        env.values.insert(argument, *value);
    }
}

/// Runs the operations of `block` in order, until one ends it.
fn run_block(env: &mut Env, block: BlockRef) -> Flow {
    let ops = block.borrow().body().iter().map(|op| op.as_operation_ref()).collect::<Vec<_>>();
    for op in ops {
        if let Some(flow) = step(env, op) {
            return flow;
        }
    }
    panic!("a block ended without a terminator");
}

/// Runs the region `region` on `arguments`.
fn run_region(env: &mut Env, region: RegionRef, arguments: &[Val]) -> Flow {
    let entry = region.borrow().entry_block_ref().expect("a region with a block");
    bind_arguments(env, entry, arguments);
    run_block(env, entry)
}

/// The regions of `op`, in order.
fn regions_of(op: OperationRef) -> Vec<RegionRef> {
    op.borrow().regions().iter().map(|region| region.as_region_ref()).collect()
}

/// What the construct `op` (a `block`, a `loop` or an `if`) does when its body stopped with
/// `flow`: `None` when it is left with results, which are defined; otherwise the flow that
/// continues outward (a branch further out, a return, a trap).
fn leave(env: &mut Env, op: OperationRef, flow: Flow) -> Option<Flow> {
    match flow {
        Flow::End(values) | Flow::Br(0, values) => {
            env.define(op, &values);
            None
        }
        Flow::Br(depth, values) => Some(Flow::Br(depth - 1, values)),
        other => Some(other),
    }
}

/// Runs one operation; `Some` when it ends the block.
fn step(env: &mut Env, op: OperationRef) -> Option<Flow> {
    env.fuel = env.fuel.checked_sub(1).unwrap_or_else(|| {
        panic!("the evaluator ran out of fuel after {FUEL} operations: an endless loop?")
    });
    let name = format!("{}", op.borrow().name());
    let operands = env.operands(op);
    match name.as_str() {
        "wasm.const" => {
            let op_ref = op.borrow();
            let constant = op_ref.downcast_ref::<wasm::Const>().expect("a wasm.const");
            let value = *constant.get_value();
            env.define(op, &[Val::new(value.unsigned() as u64, value.width())]);
        }
        "wasm.unreachable" => return Some(Flow::Trap),
        "builtin.ret" => return Some(Flow::Return(operands)),
        "wasm.end" => return Some(Flow::End(operands)),
        "wasm.br" => {
            let level = *op.borrow().downcast_ref::<wasm::Br>().expect("a wasm.br").get_level();
            let level = u32::try_from(level).expect("a label is at least 0");
            return Some(Flow::Br(level, operands));
        }
        "wasm.block" => {
            let flow = run_region(env, regions_of(op)[0], &operands);
            return leave(env, op, flow);
        }
        "wasm.if" => {
            let regions = regions_of(op);
            let taken = if operands[0].bits != 0 {
                regions[0]
            } else {
                regions[1]
            };
            let flow = run_region(env, taken, &operands[1..]);
            return leave(env, op, flow);
        }
        "wasm.loop" => {
            let body = regions_of(op)[0];
            let mut parameters = operands;
            loop {
                match run_region(env, body, &parameters) {
                    // A branch to the loop runs its body again, with new parameters.
                    Flow::Br(0, values) => parameters = values,
                    flow => return leave(env, op, flow),
                }
            }
        }
        "wasm.br_table" => {
            // The default region is the first; the entries follow it.
            let regions = regions_of(op);
            let entries = regions.len() - 1;
            let index = operands[0].bits as usize;
            let taken = if index < entries {
                regions[index + 1]
            } else {
                regions[0]
            };
            // The table is not a label: the branch of the region is taken as it is.
            return Some(run_region(env, taken, &[]));
        }
        "wasm.constf" => {
            let op_ref = op.borrow();
            let constant = op_ref.downcast_ref::<wasm::ConstF>().expect("a wasm.constf");
            let value = *constant.get_value();
            env.define(op, &[Val::float_bits(value.bits(), value.width())]);
        }
        _ => {
            let result = op
                .borrow()
                .results()
                .iter()
                .next()
                .and_then(|result| number_type(result.borrow().ty()));
            let float = name != "wasm.select"
                && (operands.first().is_some_and(|a| a.float)
                    || result.is_some_and(|(_, float)| float));
            let value = if float {
                compute_float(&name, &operands, result)
            } else {
                compute(&name, &operands)
            };
            match value {
                Some(Some(value)) => env.define(op, &[value]),
                Some(None) => return Some(Flow::Trap),
                None => panic!("the evaluator does not know `{name}`"),
            }
        }
    }
    None
}

/// The result of the float operation `name` on `operands`, or of a conversion from or to a float,
/// whose result has the width and kind `result`: `None` if the evaluator does not know it. No
/// float operation traps.
fn compute_float(name: &str, operands: &[Val], result: Option<(u32, bool)>) -> Option<Option<Val>> {
    let a = *operands.first()?;
    let b = operands.get(1).copied();
    let w = a.width;
    let flag = |value: bool| Some(Some(Val::i32(u32::from(value))));
    // `fadd`, `fsub`, `fmul`, `fdiv`: computed in the operands' width, rounding to nearest, ties
    // to even.
    let arithmetic = |f32_op: fn(f32, f32) -> f32, f64_op: fn(f64, f64) -> f64, b: Val| {
        Some(Some(if w == 32 {
            Val::f32(f32_op(a.as_f32(), b.as_f32()))
        } else {
            Val::f64(f64_op(a.as_f64(), b.as_f64()))
        }))
    };
    match (name, b) {
        ("wasm.add", Some(b)) => arithmetic(|x, y| x + y, |x, y| x + y, b),
        ("wasm.sub", Some(b)) => arithmetic(|x, y| x - y, |x, y| x - y, b),
        ("wasm.mul", Some(b)) => arithmetic(|x, y| x * y, |x, y| x * y, b),
        ("wasm.div", Some(b)) => arithmetic(|x, y| x / y, |x, y| x / y, b),
        // `fneg`: the operand with its sign bit flipped, a NaN included.
        ("wasm.neg", None) => Some(Some(Val::float_bits(a.bits ^ (1u64 << (w - 1)), w))),
        // The comparisons are IEEE 754's: a NaN is unordered with everything, `-0` equals `+0`.
        // Both widths compare as `f64`, which holds an `f32` exactly.
        ("wasm.eq", Some(b)) => flag(a.as_float() == b.as_float()),
        ("wasm.ne", Some(b)) => flag(a.as_float() != b.as_float()),
        ("wasm.lt", Some(b)) => flag(a.as_float() < b.as_float()),
        ("wasm.gt", Some(b)) => flag(a.as_float() > b.as_float()),
        ("wasm.le", Some(b)) => flag(a.as_float() <= b.as_float()),
        ("wasm.ge", Some(b)) => flag(a.as_float() >= b.as_float()),
        // `trunc_sat_s`, `trunc_sat_u`: toward zero, saturating at the bounds of the result, and 0
        // for a NaN; Rust's `as` from a float to an integer is exactly this.
        ("wasm.trunc_sat_s", None) => {
            let (rw, _) = result?;
            let value = a.as_float();
            Some(Some(if rw == 32 {
                Val::i32(value as i32 as u32)
            } else {
                Val::i64(value as i64 as u64)
            }))
        }
        ("wasm.trunc_sat_u", None) => {
            let (rw, _) = result?;
            let value = a.as_float();
            Some(Some(if rw == 32 {
                Val::i32(value as u32)
            } else {
                Val::i64(value as u64)
            }))
        }
        // `convert_s`, `convert_u`: the integer rounded to nearest, ties to even, in one rounding.
        ("wasm.convert_s", None) => {
            let (rw, _) = result?;
            let value = a.signed();
            Some(Some(if rw == 32 {
                Val::f32(value as f32)
            } else {
                Val::f64(value as f64)
            }))
        }
        ("wasm.convert_u", None) => {
            let (rw, _) = result?;
            Some(Some(if rw == 32 {
                Val::f32(a.bits as f32)
            } else {
                Val::f64(a.bits as f64)
            }))
        }
        // `promote` is exact; `demote` rounds to nearest, ties to even.
        ("wasm.promote", None) => Some(Some(Val::f64(f64::from(a.as_f32())))),
        ("wasm.demote", None) => Some(Some(Val::f32(a.as_f64() as f32))),
        // `reinterpret`: the same bits, of the other kind.
        ("wasm.reinterpret", None) => {
            let (rw, float) = result?;
            assert_eq!(rw, w, "`reinterpret` keeps the width");
            Some(Some(Val {
                bits: a.bits,
                width: w,
                float,
            }))
        }
        _ => None,
    }
}

/// The result of the pure operation `name` on `operands`: `None` if the evaluator does not know
/// it; `Some(None)` for a trap.
fn compute(name: &str, operands: &[Val]) -> Option<Option<Val>> {
    let a = operands.first().copied();
    let b = operands.get(1).copied();
    let w = a.map_or(32, |a| a.width);
    let wrap = |bits: u64| Val::new(bits, w);
    let flag = |value: bool| Val::i32(u32::from(value));
    let some = |value: Val| Some(Some(value));
    // The spec: `iadd`, `isub`, `imul` are modulo 2^N.
    match (name, a, b) {
        ("wasm.add", Some(a), Some(b)) => some(wrap(a.bits.wrapping_add(b.bits))),
        ("wasm.sub", Some(a), Some(b)) => some(wrap(a.bits.wrapping_sub(b.bits))),
        ("wasm.mul", Some(a), Some(b)) => some(wrap(a.bits.wrapping_mul(b.bits))),
        // `idiv_u`: a division by zero traps.
        ("wasm.div_u", Some(a), Some(b)) => {
            a.bits.checked_div(b.bits).map_or(Some(None), |quotient| some(wrap(quotient)))
        }
        // `idiv_s`: traps on a zero divisor, and when the quotient 2^(N-1) is not representable.
        ("wasm.div_s", Some(a), Some(b)) => {
            let min = if w == 32 {
                i64::from(i32::MIN)
            } else {
                i64::MIN
            };
            if b.bits == 0 || (a.signed() == min && b.signed() == -1) {
                Some(None)
            } else {
                some(wrap(a.signed().wrapping_div(b.signed()) as u64))
            }
        }
        ("wasm.rem_u", Some(a), Some(b)) => {
            a.bits.checked_rem(b.bits).map_or(Some(None), |remainder| some(wrap(remainder)))
        }
        // `irem_s`: traps on a zero divisor only; the remainder of the minimum by -1 is 0.
        ("wasm.rem_s", Some(a), Some(b)) => {
            if b.bits == 0 {
                Some(None)
            } else if b.signed() == -1 {
                some(wrap(0))
            } else {
                some(wrap((a.signed() % b.signed()) as u64))
            }
        }
        ("wasm.and", Some(a), Some(b)) => some(wrap(a.bits & b.bits)),
        ("wasm.or", Some(a), Some(b)) => some(wrap(a.bits | b.bits)),
        ("wasm.xor", Some(a), Some(b)) => some(wrap(a.bits ^ b.bits)),
        // `ishl`, `ishr_s`, `ishr_u`: the count is taken modulo N.
        ("wasm.shl", Some(a), Some(b)) => some(wrap(a.bits << (b.bits % u64::from(w)))),
        ("wasm.shr_u", Some(a), Some(b)) => some(wrap(a.bits >> (b.bits % u64::from(w)))),
        ("wasm.shr_s", Some(a), Some(b)) => {
            some(wrap((a.signed() >> (b.bits % u64::from(w))) as u64))
        }
        ("wasm.rotl", Some(a), Some(b)) => {
            let k = (b.bits % u64::from(w)) as u32;
            some(wrap(if k == 0 {
                a.bits
            } else {
                (a.bits << k) | (a.bits >> (w - k))
            }))
        }
        ("wasm.rotr", Some(a), Some(b)) => {
            let k = (b.bits % u64::from(w)) as u32;
            some(wrap(if k == 0 {
                a.bits
            } else {
                (a.bits >> k) | (a.bits << (w - k))
            }))
        }
        ("wasm.clz", Some(a), None) => {
            some(wrap(u64::from((a.bits << (64 - w)).leading_zeros().min(w))))
        }
        ("wasm.ctz", Some(a), None) => some(wrap(u64::from(a.bits.trailing_zeros().min(w)))),
        ("wasm.popcnt", Some(a), None) => some(wrap(u64::from(a.bits.count_ones()))),
        ("wasm.eqz", Some(a), None) => some(flag(a.bits == 0)),
        ("wasm.eq", Some(a), Some(b)) => some(flag(a.bits == b.bits)),
        ("wasm.ne", Some(a), Some(b)) => some(flag(a.bits != b.bits)),
        ("wasm.lt_s", Some(a), Some(b)) => some(flag(a.signed() < b.signed())),
        ("wasm.lt_u", Some(a), Some(b)) => some(flag(a.bits < b.bits)),
        ("wasm.gt_s", Some(a), Some(b)) => some(flag(a.signed() > b.signed())),
        ("wasm.gt_u", Some(a), Some(b)) => some(flag(a.bits > b.bits)),
        ("wasm.le_s", Some(a), Some(b)) => some(flag(a.signed() <= b.signed())),
        ("wasm.le_u", Some(a), Some(b)) => some(flag(a.bits <= b.bits)),
        ("wasm.ge_s", Some(a), Some(b)) => some(flag(a.signed() >= b.signed())),
        ("wasm.ge_u", Some(a), Some(b)) => some(flag(a.bits >= b.bits)),
        // `i32.wrap_i64`.
        ("wasm.wrap", Some(a), None) => some(Val::i32(a.bits as u32)),
        // `i64.extend_i32_s`, `i64.extend_i32_u`.
        ("wasm.extend_s", Some(a), None) => some(Val::i64(a.signed() as u64)),
        ("wasm.extend_u", Some(a), None) => some(Val::i64(a.bits)),
        // `iextend8_s`, `iextend16_s`, `iextend32_s`: the low bits read as signed, in place.
        ("wasm.extend8_s", Some(a), None) => some(wrap(i64::from(a.bits as u8 as i8) as u64)),
        ("wasm.extend16_s", Some(a), None) => some(wrap(i64::from(a.bits as u16 as i16) as u64)),
        ("wasm.extend32_s", Some(a), None) => {
            assert_eq!(a.width, 64, "`i32.extend32_s` does not exist");
            some(wrap(i64::from(a.bits as u32 as i32) as u64))
        }
        // `select`: the first value if the condition is not zero.
        ("wasm.select", Some(c), Some(x)) => {
            let y = operands.get(2).copied()?;
            some(if c.bits != 0 { x } else { y })
        }
        _ => None,
    }
}

/// A module with one function `@f` of text `function`, parsed and verified.
fn function_of(
    text: &str,
) -> (
    alloc::rc::Rc<runic_ir::Context>,
    runic_ir::dialects::builtin::ModuleRef,
    OperationRef,
) {
    let module_text = format!("builtin.module public @m {{\n{text}\n}};");
    let (context, module) = super::parse_module(&module_text);
    let function = runic_ir::SymbolTable::symbol_manager(&*module.borrow())
        .lookup_op("f")
        .expect("the function @f");
    (context, module, function)
}

/// Runs the function `@f` of the text `function` on `arguments`.
fn run_text(function: &str, arguments: &[Val]) -> Outcome {
    let (_context, _module, function) = function_of(function);
    run(function, arguments)
}

/// One `i32` function of two parameters returning `op`'s result.
fn binary_text(op: &str) -> String {
    format!(
        "builtin.function public extern(\"wasm\") @f(%a: i32, %b: i32) -> i32 {{\n    %r = \
         wasm.{op} %a, %b;\n    builtin.ret %r : (i32);\n}};"
    )
}

/// The result of `wasm.{op}` on the `i32`s `a` and `b`.
fn i32_binary(op: &str, a: u32, b: u32) -> Outcome {
    run_text(&binary_text(op), &[Val::i32(a), Val::i32(b)])
}

fn ok32(bits: u32) -> Outcome {
    Outcome::Values(alloc::vec![Val::i32(bits)])
}

#[test]
fn rem_s_of_the_minimum_by_minus_one_is_zero() {
    assert_eq!(i32_binary("rem_s", 0x8000_0000, u32::MAX), ok32(0));
}

#[test]
fn div_s_of_the_minimum_by_minus_one_traps() {
    assert_eq!(i32_binary("div_s", 0x8000_0000, u32::MAX), Outcome::Trap);
}

#[test]
fn division_and_remainder_by_zero_trap() {
    for op in ["div_s", "div_u", "rem_s", "rem_u"] {
        assert_eq!(i32_binary(op, 7, 0), Outcome::Trap, "{op}");
    }
}

#[test]
fn signed_division_and_remainder_truncate_toward_zero() {
    // -7 / 2 = -3 and -7 rem 2 = -1; 7 / -2 = -3 and 7 rem -2 = 1.
    assert_eq!(i32_binary("div_s", (-7i32) as u32, 2), ok32((-3i32) as u32));
    assert_eq!(i32_binary("rem_s", (-7i32) as u32, 2), ok32((-1i32) as u32));
    assert_eq!(i32_binary("div_s", 7, (-2i32) as u32), ok32((-3i32) as u32));
    assert_eq!(i32_binary("rem_s", 7, (-2i32) as u32), ok32(1));
}

#[test]
fn shl_by_33_is_a_shift_by_1() {
    assert_eq!(i32_binary("shl", 3, 33), ok32(6));
    assert_eq!(i32_binary("shr_u", 6, 33), ok32(3));
    assert_eq!(i32_binary("shr_s", 0x8000_0000, 33), ok32(0xc000_0000));
}

#[test]
fn rotates_take_the_count_modulo_the_width() {
    assert_eq!(i32_binary("rotl", 0x8000_0001, 1), ok32(3));
    assert_eq!(i32_binary("rotr", 0x8000_0001, 1), ok32(0xc000_0000));
    assert_eq!(i32_binary("rotl", 5, 32), ok32(5));
}

#[test]
fn counts_of_zero_are_the_width() {
    let unary = |op: &str, width: &str, value: Val| {
        run_text(
            &format!(
                "builtin.function public extern(\"wasm\") @f(%a: {width}) -> {width} {{\n    %r = \
                 wasm.{op} %a;\n    builtin.ret %r : ({width});\n}};"
            ),
            &[value],
        )
    };
    assert_eq!(unary("clz", "i32", Val::i32(0)), ok32(32));
    assert_eq!(unary("ctz", "i32", Val::i32(0)), ok32(32));
    assert_eq!(unary("clz", "i64", Val::i64(0)), Outcome::Values(alloc::vec![Val::i64(64)]));
    assert_eq!(unary("clz", "i32", Val::i32(1)), ok32(31));
    assert_eq!(unary("ctz", "i32", Val::i32(8)), ok32(3));
    assert_eq!(
        unary("popcnt", "i64", Val::i64(u64::MAX)),
        Outcome::Values(alloc::vec![Val::i64(64)])
    );
}

#[test]
fn conversions_wrap_and_extend() {
    let convert = |op: &str, from: &str, to: &str, value: Val| {
        run_text(
            &format!(
                "builtin.function public extern(\"wasm\") @f(%a: {from}) -> {to} {{\n    %r = \
                 wasm.{op} %a <{{ ty = {to} }}>;\n    builtin.ret %r : ({to});\n}};"
            ),
            &[value],
        )
    };
    assert_eq!(convert("wrap", "i64", "i32", Val::i64(0x1_ffff_ffff)), ok32(u32::MAX));
    assert_eq!(
        convert("extend_s", "i32", "i64", Val::i32(0x8000_0000)),
        Outcome::Values(alloc::vec![Val::i64(0xffff_ffff_8000_0000)])
    );
    assert_eq!(
        convert("extend_u", "i32", "i64", Val::i32(0x8000_0000)),
        Outcome::Values(alloc::vec![Val::i64(0x8000_0000)])
    );
}

#[test]
fn in_place_sign_extensions_read_the_low_bits() {
    let unary = |op: &str, width: &str, value: Val| {
        run_text(
            &format!(
                "builtin.function public extern(\"wasm\") @f(%a: {width}) -> {width} {{\n    %r = \
                 wasm.{op} %a;\n    builtin.ret %r : ({width});\n}};"
            ),
            &[value],
        )
    };
    assert_eq!(unary("extend8_s", "i32", Val::i32(0x1f0)), ok32(0xffff_fff0));
    assert_eq!(unary("extend8_s", "i32", Val::i32(0x17f)), ok32(0x7f));
    assert_eq!(unary("extend16_s", "i32", Val::i32(0x8000)), ok32(0xffff_8000));
    assert_eq!(
        unary("extend32_s", "i64", Val::i64(0x8000_0000)),
        Outcome::Values(alloc::vec![Val::i64(0xffff_ffff_8000_0000)])
    );
}

#[test]
fn comparisons_read_their_operands_as_signed_or_unsigned() {
    assert_eq!(i32_binary("lt_s", u32::MAX, 0), ok32(1));
    assert_eq!(i32_binary("lt_u", u32::MAX, 0), ok32(0));
    assert_eq!(i32_binary("ge_u", u32::MAX, 0), ok32(1));
    assert_eq!(i32_binary("eq", 4, 4), ok32(1));
    assert_eq!(i32_binary("ne", 4, 4), ok32(0));
}

#[test]
fn select_takes_the_first_value_when_the_condition_is_not_zero() {
    let text =
        "builtin.function public extern(\"wasm\") @f(%c: i32, %a: i32, %b: i32) -> i32 {\n    %r \
         = wasm.select %c, %a, %b;\n    builtin.ret %r : (i32);\n};";
    let run = |c| run_text(text, &[Val::i32(c), Val::i32(10), Val::i32(20)]);
    assert_eq!(run(1), ok32(10));
    assert_eq!(run(0), ok32(20));
    assert_eq!(run(7), ok32(10));
}

#[test]
fn if_runs_the_region_the_condition_picks_and_unreachable_traps() {
    let text = "builtin.function public extern(\"wasm\") @f(%c: i32) -> i32 {\n    %r = wasm.if \
                %c then {\n        %one = wasm.const 1 : i32;\n        wasm.end %one : (i32);\n    \
                } else {\n        wasm.unreachable;\n    } : (i32) -> (i32);\n    builtin.ret %r \
                : (i32);\n};";
    assert_eq!(run_text(text, &[Val::i32(1)]), ok32(1));
    assert_eq!(run_text(text, &[Val::i32(0)]), Outcome::Trap);
}

#[test]
#[should_panic(expected = "the evaluator does not know `wasm.sqrt`")]
fn an_operation_the_evaluator_does_not_know_is_a_failure_that_names_it() {
    run_text(
        "builtin.function public extern(\"wasm\") @f(%a: f32) -> f32 {\n    %r = wasm.sqrt \
         %a;\n    builtin.ret %r : (f32);\n};",
        &[Val::f32(4.0)],
    );
}

/// The result of `wasm.{op}` on the `ty`s `a` and `b`, a value of type `result`.
fn float_binary(op: &str, ty: &str, result: &str, a: Val, b: Val) -> Outcome {
    run_text(
        &format!(
            "builtin.function public extern(\"wasm\") @f(%a: {ty}, %b: {ty}) -> {result} {{\n    \
             %r = wasm.{op} %a, %b;\n    builtin.ret %r : ({result});\n}};"
        ),
        &[a, b],
    )
}

/// The result of the conversion `wasm.{op}` of `value`, of type `from`, to `to`.
fn conversion(op: &str, from: &str, to: &str, value: Val) -> Outcome {
    run_text(
        &format!(
            "builtin.function public extern(\"wasm\") @f(%a: {from}) -> {to} {{\n    %r = \
             wasm.{op} %a <{{ ty = {to} }}>;\n    builtin.ret %r : ({to});\n}};"
        ),
        &[value],
    )
}

/// The one value of `outcome`.
#[track_caller]
fn single(outcome: Outcome) -> Val {
    match outcome {
        Outcome::Values(values) if values.len() == 1 => values[0],
        other => panic!("expected one value, got {other:?}"),
    }
}

#[test]
fn float_arithmetic_rounds_in_the_width_of_its_operands() {
    // `2^24 + 1` is not an `f32`: the sum rounds to `2^24` there, and is exact in an `f64`.
    let sum = float_binary("add", "f32", "f32", Val::f32(16_777_216.0), Val::f32(1.0));
    assert_eq!(single(sum), Val::f32(16_777_216.0));
    let sum = float_binary("add", "f64", "f64", Val::f64(16_777_216.0), Val::f64(1.0));
    assert_eq!(single(sum), Val::f64(16_777_217.0));
    let difference = float_binary("sub", "f64", "f64", Val::f64(0.5), Val::f64(2.0));
    assert_eq!(single(difference), Val::f64(-1.5));
    let product = float_binary("mul", "f32", "f32", Val::f32(1.5), Val::f32(-2.0));
    assert_eq!(single(product), Val::f32(-3.0));
}

#[test]
fn float_division_by_zero_is_an_infinity_or_a_nan_and_does_not_trap() {
    let quotient = float_binary("div", "f64", "f64", Val::f64(1.0), Val::f64(-0.0));
    assert_eq!(single(quotient), Val::f64(f64::NEG_INFINITY));
    let quotient = float_binary("div", "f32", "f32", Val::f32(0.0), Val::f32(0.0));
    assert!(single(quotient).is_nan());
    let difference =
        float_binary("sub", "f64", "f64", Val::f64(f64::INFINITY), Val::f64(f64::INFINITY));
    assert!(single(difference).is_nan());
}

#[test]
fn neg_flips_the_sign_bit_of_every_value_a_nan_included() {
    let neg = |value: Val, ty: &str| {
        single(run_text(
            &format!(
                "builtin.function public extern(\"wasm\") @f(%a: {ty}) -> {ty} {{\n    %r = \
                 wasm.neg %a;\n    builtin.ret %r : ({ty});\n}};"
            ),
            &[value],
        ))
    };
    assert_eq!(neg(Val::f32(0.0), "f32"), Val::f32(-0.0));
    assert_eq!(neg(Val::float_bits(0x7fc0_0001, 32), "f32"), Val::float_bits(0xffc0_0001, 32));
    assert_eq!(neg(Val::f64(-2.5), "f64"), Val::f64(2.5));
}

#[test]
fn float_comparisons_are_unordered_with_a_nan_and_equate_the_zeros() {
    let nan = Val::f64(f64::NAN);
    let compare = |op: &str, a: Val, b: Val| float_binary(op, "f64", "i32", a, b);
    for op in ["eq", "lt", "gt", "le", "ge"] {
        assert_eq!(compare(op, nan, Val::f64(1.0)), ok32(0), "{op}");
        assert_eq!(compare(op, nan, nan), ok32(0), "{op}");
    }
    assert_eq!(compare("ne", nan, nan), ok32(1));
    assert_eq!(compare("eq", Val::f64(-0.0), Val::f64(0.0)), ok32(1));
    assert_eq!(compare("lt", Val::f64(-0.0), Val::f64(0.0)), ok32(0));
    assert_eq!(compare("le", Val::f64(1.0), Val::f64(1.0)), ok32(1));
    let less = float_binary("lt", "f32", "i32", Val::f32(f32::NEG_INFINITY), Val::f32(-3.0e38));
    assert_eq!(less, ok32(1));
}

#[test]
fn saturating_truncation_saturates_and_gives_zero_for_a_nan() {
    let sat = |op: &str, from: &str, to: &str, value: Val| single(conversion(op, from, to, value));
    assert_eq!(sat("trunc_sat_s", "f32", "i32", Val::f32(f32::NAN)), Val::i32(0));
    assert_eq!(sat("trunc_sat_s", "f32", "i32", Val::f32(3.0e9)), Val::i32(0x7fff_ffff));
    assert_eq!(sat("trunc_sat_s", "f64", "i32", Val::f64(-3.0e9)), Val::i32(0x8000_0000));
    assert_eq!(sat("trunc_sat_s", "f64", "i64", Val::f64(-2.5)), Val::i64((-2i64) as u64));
    assert_eq!(sat("trunc_sat_u", "f64", "i32", Val::f64(-1.5)), Val::i32(0));
    assert_eq!(sat("trunc_sat_u", "f64", "i32", Val::f64(4.5e9)), Val::i32(u32::MAX));
    assert_eq!(sat("trunc_sat_u", "f32", "i64", Val::f32(f32::INFINITY)), Val::i64(u64::MAX));
    assert_eq!(sat("trunc_sat_u", "f64", "i64", Val::f64(255.9)), Val::i64(255));
}

#[test]
fn integer_to_float_conversions_read_the_sign_and_round_to_nearest_even() {
    let convert =
        |op: &str, from: &str, to: &str, value: Val| single(conversion(op, from, to, value));
    assert_eq!(convert("convert_s", "i32", "f32", Val::i32(u32::MAX)), Val::f32(-1.0));
    assert_eq!(
        convert("convert_u", "i32", "f32", Val::i32(u32::MAX)),
        Val::f32(4_294_967_296.0)
    );
    // `2^53 + 1` is halfway between two `f64`s: it rounds to the even one, `2^53`.
    let tie = (1u64 << 53) + 1;
    assert_eq!(
        convert("convert_s", "i64", "f64", Val::i64(tie)),
        Val::f64(9_007_199_254_740_992.0)
    );
    // `2^24 + 3` rounds up to `2^24 + 4` in an `f32`, the even neighbour.
    assert_eq!(
        convert("convert_u", "i64", "f32", Val::i64((1 << 24) + 3)),
        Val::f32(16_777_220.0)
    );
    assert_eq!(
        convert("convert_u", "i64", "f64", Val::i64(u64::MAX)),
        Val::f64(18_446_744_073_709_551_616.0)
    );
}

#[test]
fn promotion_is_exact_and_demotion_rounds() {
    let convert =
        |op: &str, from: &str, to: &str, value: Val| single(conversion(op, from, to, value));
    assert_eq!(convert("promote", "f32", "f64", Val::f32(0.1)), Val::f64(f64::from(0.1f32)));
    assert_eq!(convert("demote", "f64", "f32", Val::f64(1.0e300)), Val::f32(f32::INFINITY));
    assert_eq!(convert("demote", "f64", "f32", Val::f64(0.1)), Val::f32(0.1));
    assert!(convert("demote", "f64", "f32", Val::f64(f64::NAN)).is_nan());
}

#[test]
fn reinterpret_keeps_the_bits_and_changes_the_kind() {
    let convert =
        |from: &str, to: &str, value: Val| single(conversion("reinterpret", from, to, value));
    assert_eq!(convert("f32", "i32", Val::f32(1.0)), Val::i32(0x3f80_0000));
    assert_eq!(
        convert("i64", "f64", Val::i64(0x7ff8_0000_0000_0001)),
        Val::float_bits(0x7ff8_0000_0000_0001, 64)
    );
}

#[test]
fn constf_gives_its_bits() {
    let text = "builtin.function public extern(\"wasm\") @f() -> f64 {\n    %r = wasm.constf -1.5 \
                : f64;\n    builtin.ret %r : (f64);\n};";
    assert_eq!(single(run_text(text, &[])), Val::f64(-1.5));
}

#[test]
fn a_function_returns_each_of_its_values_in_order() {
    let text =
        "builtin.function public extern(\"wasm\") @f(%a: i32, %b: f32) -> (f32, i32) {\n    \
         builtin.ret %b, %a : (f32, i32);\n};";
    assert_eq!(
        run_text(text, &[Val::i32(7), Val::f32(2.5)]),
        Outcome::Values(alloc::vec![Val::f32(2.5), Val::i32(7)])
    );
}

/// The `i32` function `@f` of the parameters `params` whose body is `body`, run on `arguments`.
fn run_body(params: &str, body: &str, arguments: &[Val]) -> Outcome {
    run_text(
        &format!("builtin.function public extern(\"wasm\") @f({params}) -> i32 {{\n{body}\n}};"),
        arguments,
    )
}

#[test]
fn a_block_gives_the_values_of_its_end_or_of_a_branch_that_names_it() {
    let body = "%r = wasm.block %x {\n^b(%v: i32):\n    wasm.if %c then {\n        %k = \
                wasm.const 7 : i32;\n        wasm.br 1 %k : (i32);\n    } : (i32) -> ();\n    \
                wasm.end %v : (i32);\n} : (i32) -> (i32);\nbuiltin.ret %r : (i32);";
    let run = |c| run_body("%c: i32, %x: i32", body, &[Val::i32(c), Val::i32(3)]);
    assert_eq!(run(0), ok32(3));
    assert_eq!(run(1), ok32(7));
}

#[test]
fn an_if_with_results_gives_those_of_the_region_it_runs_on_its_parameters() {
    let body = "%r = wasm.if %c, %x then {\n^t(%v: i32):\n    %one = wasm.const 1 : i32;\n    %s \
                = wasm.add %v, %one;\n    wasm.end %s : (i32);\n} else {\n^e(%v: i32):\n    \
                wasm.end %v : (i32);\n} : (i32, i32) -> (i32);\nbuiltin.ret %r : (i32);";
    let run = |c| run_body("%c: i32, %x: i32", body, &[Val::i32(c), Val::i32(10)]);
    assert_eq!(run(5), ok32(11));
    assert_eq!(run(0), ok32(10));
}

#[test]
fn a_branch_to_a_loop_runs_its_body_again_with_new_parameters() {
    // Sums 1..=n: the loop carries (i, sum), branches back while i < n, and ends with the sum.
    let body = "%zero = wasm.const 0 : i32;\n%one = wasm.const 1 : i32;\n%i, %s = wasm.loop \
                %zero, %zero {\n^b(%j: i32, %t: i32):\n    %k = wasm.add %j, %one;\n    %u = \
                wasm.add %t, %k;\n    %more = wasm.lt_u %k, %n;\n    wasm.if %more then {\n        \
                wasm.br 1 %k, %u : (i32, i32);\n    } : (i32) -> ();\n    wasm.end %k, %u : (i32, \
                i32);\n} : (i32, i32) -> (i32, i32);\nbuiltin.ret %s : (i32);";
    let run = |n| run_body("%n: i32", body, &[Val::i32(n)]);
    assert_eq!(run(1), ok32(1));
    assert_eq!(run(4), ok32(10));
    assert_eq!(run(100), ok32(5050));
}

#[test]
fn a_branch_unwinds_the_constructs_it_crosses() {
    // From inside a loop inside a block, `br 2` leaves the outer block; `br 1` leaves the inner.
    let body = "%r = wasm.block {\n    %q = wasm.block {\n        wasm.loop {\n            %k = \
                wasm.const 5 : i32;\n            wasm.if %c then {\n                wasm.br 3 %k \
                : (i32);\n            } : (i32) -> ();\n            wasm.br 1 %k : (i32);\n        \
                } : () -> ();\n        %never = wasm.const 0 : i32;\n        wasm.end %never : \
                (i32);\n    } : () -> (i32);\n    %two = wasm.const 2 : i32;\n    %d = wasm.mul \
                %q, %two;\n    wasm.end %d : (i32);\n} : () -> (i32);\nbuiltin.ret %r : (i32);";
    let run = |c| run_body("%c: i32", body, &[Val::i32(c)]);
    assert_eq!(run(1), ok32(5));
    assert_eq!(run(0), ok32(10));
}

#[test]
fn br_table_takes_the_entry_of_its_index_and_the_default_past_the_entries() {
    let body = "%r = wasm.block {\n    wasm.block {\n        wasm.block {\n            \
                wasm.br_table %i [{\n                wasm.br 0 : ();\n            }, {\n          \
                      wasm.br 1 : ();\n            }] default {\n                wasm.br 1 : \
                ();\n            } : (i32);\n        } : () -> ();\n        %a = wasm.const 10 : \
                i32;\n        wasm.br 1 %a : (i32);\n    } : () -> ();\n    %b = wasm.const 20 : \
                i32;\n    wasm.end %b : (i32);\n} : () -> (i32);\nbuiltin.ret %r : (i32);";
    let run = |i| run_body("%i: i32", body, &[Val::i32(i)]);
    assert_eq!(run(0), ok32(10));
    assert_eq!(run(1), ok32(20));
    assert_eq!(run(2), ok32(20));
    // The index is unsigned: -1 is past the entries.
    assert_eq!(run(u32::MAX), ok32(20));
}

#[test]
fn a_return_nested_in_constructs_leaves_the_function() {
    let body = "wasm.block {\n    wasm.loop {\n        wasm.if %c then {\n            builtin.ret \
                %c : (i32);\n        } : (i32) -> ();\n        wasm.br 1 : ();\n    } : () -> \
                ();\n    wasm.end : ();\n} : () -> ();\n%z = wasm.const 0 : i32;\nbuiltin.ret %z \
                : (i32);";
    let run = |c| run_body("%c: i32", body, &[Val::i32(c)]);
    assert_eq!(run(9), ok32(9));
    assert_eq!(run(0), ok32(0));
}

#[test]
#[should_panic(expected = "the evaluator ran out of fuel")]
fn an_endless_loop_runs_out_of_fuel() {
    let body = "wasm.loop {\n    wasm.br 0 : ();\n} : () -> ();\nbuiltin.ret %c : (i32);";
    run_body("%c: i32", body, &[Val::i32(0)]);
}
