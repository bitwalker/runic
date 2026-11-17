//! An evaluator of lowered functions, written from the WebAssembly specification (release 3.0,
//! "Numerics" and "Execution") and independent of the patterns of the crate.
//!
//! [run] executes one `builtin.function` whose body is `wasm` operations and ends in
//! `builtin.ret`, on argument values, and gives the values it returns or [Outcome::Trap]. A value
//! is its bit pattern in a `u64` and the width of its type (32 or 64): [Val].
//!
//! The operations it knows are the integer operations of the `wasm` dialect that lowered integer
//! code uses, `wasm.if` with `wasm.end`, `wasm.unreachable` and `builtin.ret`. An operation it
//! does not know panics with a message that names it: a test never skips one silently.

use alloc::{format, string::String, vec::Vec};
use std::collections::HashMap;

use runic_dialect_wasm as wasm;
use runic_ir::{BlockRef, OperationRef, RegionRef, ValueRef, dialects::builtin::Function};

use crate::types::carrier_width;

/// A Wasm integer value: the bits of an `i32` or an `i64` in the low `width` bits of `bits`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Val {
    /// The bit pattern; the bits at or above `width` are zero.
    pub bits: u64,
    /// 32 or 64.
    pub width: u32,
}

impl Val {
    /// The value of the low `width` bits of `bits`.
    pub fn new(bits: u64, width: u32) -> Self {
        assert!(width == 32 || width == 64, "a Wasm integer is 32 or 64 bits wide");
        Self {
            bits: bits & mask(width),
            width,
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

    /// The value read as a signed number.
    fn signed(self) -> i64 {
        if self.width == 32 {
            i64::from(self.bits as u32 as i32)
        } else {
            self.bits as i64
        }
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

/// What a body gave when it stopped.
enum Flow {
    /// A `wasm.end` with these values.
    End(Vec<Val>),
    /// A `builtin.ret` with these values.
    Return(Vec<Val>),
    /// A trap.
    Trap,
}

/// The values of the SSA values of the function being run.
struct Env {
    values: HashMap<ValueRef, Val>,
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
    };
    bind_arguments(&mut env, body, arguments);
    match run_block(&mut env, body) {
        Flow::Return(values) => Outcome::Values(values),
        Flow::Trap => Outcome::Trap,
        Flow::End(_) => panic!("the body of a function ended in `wasm.end`"),
    }
}

/// Defines the arguments of `block` as `values`.
fn bind_arguments(env: &mut Env, block: BlockRef, values: &[Val]) {
    let arguments = block.borrow().argument_values().collect::<Vec<_>>();
    assert_eq!(arguments.len(), values.len(), "the block takes {} arguments", arguments.len());
    for (argument, value) in arguments.into_iter().zip(values) {
        let ty = argument.borrow().ty();
        assert_eq!(
            carrier_width(ty),
            Some(value.width),
            "the argument `{}` of type `{ty}` was given a value of {} bits",
            argument.borrow(),
            value.width
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

/// Runs one operation; `Some` when it ends the block.
fn step(env: &mut Env, op: OperationRef) -> Option<Flow> {
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
        "wasm.if" => {
            let regions = op
                .borrow()
                .regions()
                .iter()
                .map(|region| region.as_region_ref())
                .collect::<Vec<_>>();
            let taken = if operands[0].bits != 0 {
                regions[0]
            } else {
                regions[1]
            };
            match run_region(env, taken, &operands[1..]) {
                Flow::End(values) => env.define(op, &values),
                other => return Some(other),
            }
        }
        _ => match compute(&name, &operands) {
            Some(Some(value)) => env.define(op, &[value]),
            Some(None) => return Some(Flow::Trap),
            None => panic!("the evaluator does not know `{name}`"),
        },
    }
    None
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
#[should_panic(expected = "the evaluator does not know `wasm.reinterpret`")]
fn an_operation_the_evaluator_does_not_know_is_a_failure_that_names_it() {
    run_text(
        "builtin.function public extern(\"wasm\") @f(%a: i32) -> f32 {\n    %r = wasm.reinterpret \
         %a <{ ty = f32 }>;\n    builtin.ret %r : (f32);\n};",
        &[Val::i32(0)],
    );
}
