//! Comparisons: operations on two numbers of one type whose result is an `i32`, 1 or 0.

use runic_ir::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    effects::MemoryEffectOpInterface,
    traits::*,
    *,
};

use crate::{FloatType, IntType, NumType, *};

/// Test against zero: `i32.eqz`, `i64.eqz`.
///
/// One operand of an integer type (`i32` or `i64`) and a result that is an `i32`: 1 if the
/// operand is zero and 0 otherwise. It never traps.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = WasmDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Eqz {
    #[operand]
    operand: AnyType,
    #[result]
    result: AnyType,
}

impl InferTypeOpInterface for Eqz {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

impl Verify<dyn InferTypeOpInterface> for Eqz {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        crate::ops::verify_uniform::<IntType>(self.as_operation(), crate::ops::ResultRule::I32)
    }
}

compare_binary! {
    /// Equality: `i32.eq`, `i64.eq`, `f32.eq`, `f64.eq`.
    ///
    /// Two operands of one number type and an `i32` result: 1 if they are equal and 0 otherwise. For
    /// floats it is the IEEE 754 comparison: a NaN is equal to nothing, and `-0.0` equals `0.0`. It never
    /// traps and is commutative.
    Eq, "eq", NumType, [Commutative]
}

compare_binary! {
    /// Inequality: `i32.ne`, `i64.ne`, `f32.ne`, `f64.ne`.
    ///
    /// Two operands of one number type and an `i32` result: 1 if they differ and 0 otherwise. For floats
    /// it is the IEEE 754 comparison: a NaN differs from everything. It never traps and is commutative.
    Ne, "ne", NumType, [Commutative]
}

compare_binary! {
    /// Integer comparison, signed: `i32.lt_s`, `i64.lt_s`.
    ///
    /// Two operands of one integer type and an `i32` result: 1 if the first, read as a signed number, is less than the second and 0 otherwise. It never traps.
    LtS, "lt_s", IntType, []
}

compare_binary! {
    /// Integer comparison, unsigned: `i32.lt_u`, `i64.lt_u`.
    ///
    /// Two operands of one integer type and an `i32` result: 1 if the first, read as a unsigned number, is less than the second and 0 otherwise. It never traps.
    LtU, "lt_u", IntType, []
}

compare_binary! {
    /// Integer comparison, signed: `i32.gt_s`, `i64.gt_s`.
    ///
    /// Two operands of one integer type and an `i32` result: 1 if the first, read as a signed number, is greater than the second and 0 otherwise. It never traps.
    GtS, "gt_s", IntType, []
}

compare_binary! {
    /// Integer comparison, unsigned: `i32.gt_u`, `i64.gt_u`.
    ///
    /// Two operands of one integer type and an `i32` result: 1 if the first, read as a unsigned number, is greater than the second and 0 otherwise. It never traps.
    GtU, "gt_u", IntType, []
}

compare_binary! {
    /// Integer comparison, signed: `i32.le_s`, `i64.le_s`.
    ///
    /// Two operands of one integer type and an `i32` result: 1 if the first, read as a signed number, is less than or equal to the second and 0 otherwise. It never traps.
    LeS, "le_s", IntType, []
}

compare_binary! {
    /// Integer comparison, unsigned: `i32.le_u`, `i64.le_u`.
    ///
    /// Two operands of one integer type and an `i32` result: 1 if the first, read as a unsigned number, is less than or equal to the second and 0 otherwise. It never traps.
    LeU, "le_u", IntType, []
}

compare_binary! {
    /// Integer comparison, signed: `i32.ge_s`, `i64.ge_s`.
    ///
    /// Two operands of one integer type and an `i32` result: 1 if the first, read as a signed number, is greater than or equal to the second and 0 otherwise. It never traps.
    GeS, "ge_s", IntType, []
}

compare_binary! {
    /// Integer comparison, unsigned: `i32.ge_u`, `i64.ge_u`.
    ///
    /// Two operands of one integer type and an `i32` result: 1 if the first, read as a unsigned number, is greater than or equal to the second and 0 otherwise. It never traps.
    GeU, "ge_u", IntType, []
}

compare_binary! {
    /// Float comparison: `f32.lt`, `f64.lt`.
    ///
    /// Two operands of one float type and an `i32` result: 1 if the first is less than the second and 0 otherwise; 0 if either is a NaN. It never traps.
    Lt, "lt", FloatType, []
}

compare_binary! {
    /// Float comparison: `f32.gt`, `f64.gt`.
    ///
    /// Two operands of one float type and an `i32` result: 1 if the first is greater than the second and 0 otherwise; 0 if either is a NaN. It never traps.
    Gt, "gt", FloatType, []
}

compare_binary! {
    /// Float comparison: `f32.le`, `f64.le`.
    ///
    /// Two operands of one float type and an `i32` result: 1 if the first is less than or equal to the second and 0 otherwise; 0 if either is a NaN. It never traps.
    Le, "le", FloatType, []
}

compare_binary! {
    /// Float comparison: `f32.ge`, `f64.ge`.
    ///
    /// Two operands of one float type and an `i32` result: 1 if the first is greater than or equal to the second and 0 otherwise; 0 if either is a NaN. It never traps.
    Ge, "ge", FloatType, []
}
