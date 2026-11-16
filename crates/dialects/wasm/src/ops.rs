//! The operations of the dialect.

use runic_ir::{Operation, Value};

use crate::types::is_int_type;

#[macro_use]
mod macros;
mod call;
mod compare;
mod control;
mod convert;
mod memory;
mod module;
mod numeric;
mod text;
mod variable;
mod verify;

pub(crate) use self::verify::{
    ResultRule, verify_constant, verify_conversion, verify_select, verify_uniform,
};
pub use self::{
    call::*, compare::*, control::*, convert::*, memory::*, module::*, numeric::*, variable::*,
};

/// Marks an operation that may appear in a Wasm constant expression (the initializer of a global,
/// of a table or of a segment offset).
///
/// The marker is on the operation, not on an operation at a type, and `add`, `sub` and `mul` are in
/// constant expressions for integers only (the extended-const feature), while they also take
/// floats. So the bare marker over-approximates: **ask [is_constant_instruction], not
/// `implements::<dyn ConstantInstruction>()`.**
///
/// It is a plain trait, not an `#[operation_trait]`: that macro cannot be used outside `runic-ir`
/// (its blanket implementation of `Verify` breaks the orphan rule), and the marker needs no
/// verifier.
pub trait ConstantInstruction {}

/// Whether `op` is an instruction a Wasm constant expression may hold: `const`, `constf`,
/// `add`, `sub` and `mul` when they compute on integers (`i32` or `i64`), `global_get` of an
/// immutable global, `ref_null`, `ref_func`, and the two address operations `data_addr` and
/// `func_index` (each stands for an `i32.const`).
///
/// This is the precise question; the [ConstantInstruction] marker alone also holds for the float
/// `add`, `sub` and `mul`, which Wasm does not allow in a constant expression, and for a
/// `global_get` of a mutable global. It looks at the operation (and the global it names) only:
/// whether its operands are themselves constant is for the caller to check.
pub fn is_constant_instruction(op: &Operation) -> bool {
    if !op.implements::<dyn ConstantInstruction>() {
        return false;
    }
    if op.is::<Add>() || op.is::<Sub>() || op.is::<Mul>() {
        return op.results().iter().all(|result| is_int_type(result.borrow().ty()));
    }
    if let Some(get) = op.downcast_ref::<GlobalGet>() {
        return get.global_op().is_some_and(|global| !global.borrow().get_mutability().is_mut());
    }
    true
}
