//! The `arith` dialect: constants, arithmetic, bitwise operations, comparisons and casts on
//! signless integers, and constants, arithmetic, comparison and conversion on floating-point
//! numbers.
//!
//! # Floating-point operations
//!
//! `arith.constantf`, `addf`, `subf`, `mulf`, `divf`, `negf` and `cmpf` work on `f32` and `f64`
//! with IEEE 754-2019 semantics, rounding to nearest, ties to even. None of them traps: a division
//! by zero gives an infinity or a NaN. When a result is a NaN, its sign and payload are
//! unspecified, and no fold produces one (which NaN an operation gives differs between hosts and
//! engines), so an operation whose result would be a NaN is left in place. A comparison is
//! deterministic and does fold, NaN operands included. No float operation has an algebraic
//! simplification (`x + 0.0` is not `x`: it changes `-0.0`), and the dialect has no float
//! canonicalization pattern; only folds on constant operands.
//!
//! # Float conversions
//!
//! `arith.fptosi`, `fptoui`, `sitofp`, `uitofp`, `fpext`, `fptrunc` and `bitcast` convert between
//! floats and integers, between `f32` and `f64`, and between the bits of a value and a type of the
//! same width. Each has one operand and one result, never traps and is free of effects, and writes
//! its result type as the property `ty`, as the integer casts do. The integer side of the four
//! integer conversions is any integer type, `isize` included; `fpext` is `f32` to `f64`, `fptrunc`
//! `f64` to `f32`, and `bitcast` is `i32` and `f32`, or `i64` and `f64`, in either direction.
//!
//! `fptosi` and `fptoui` truncate toward zero. Their result is **poison** if the operand is a NaN or
//! the truncated value is not representable in the result type (for `fptoui`, a negative value that
//! truncates to zero gives `0`, and any other negative value is poison). This is LLVM's and MLIR's
//! rule; a target whose conversion traps or saturates (Wasm has both) refines it. `sitofp` and
//! `uitofp` round to nearest, ties to even. A fold never produces a NaN or a poison value: a
//! conversion whose result would be either stays in place. A `bitcast` always folds, a NaN's bits
//! included, and a `bitcast` of a `bitcast` back to the original type folds to the original value.
//! A conversion that involves `isize` folds only when the 32-bit and the 64-bit target agree.
//! Their integer results have the full range of their type.
//!
//! # Traps
//!
//! Some operations trap: a division by zero, a checked operation whose exact result does not fit
//! (`Overflow::Checked`). A trap is not an effect, and every `arith` operation is free of effects;
//! none claims to be speculatable. An operation that would trap may be removed, together with its
//! trap, when its result is unused; a trap is never introduced, and an operation that can trap is
//! never moved to where it would execute more often. A check that must happen is a `cf.assert`.
//!
//! An operation that is defined only for some values of an operand (a divisor that is not zero, a
//! shift amount below the width) says so through
//! [runic_ir::traits::OperandRangeRequirementOpInterface]; one whose requirement relates two
//! operands (a signed division, a checked or unchecked `add`/`sub`/`mul`) reports it there as
//! unexpressed.

#![no_std]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(specialization)]
#![allow(incomplete_features)]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod attributes;
mod builders;
mod canonicalization;
#[macro_use]
mod fold;
#[macro_use]
mod fold_float;
mod ops;
#[cfg(test)]
mod tests;

use runic_ir::{
    AttributeRef, Builder, BuilderExt, Dialect, DialectInfo, Float, Int, OperationRef, ReadyAttr,
    SourceSpan, Type,
    attributes::IntegerLikeAttr,
    derive::DialectRegistration,
    dialects::builtin::attributes::{FloatAttr, IntAttr},
};

pub use self::{
    attributes::{CmpFPredicate, CmpFPredicateAttr, CmpPredicate, CmpPredicateAttr},
    builders::ArithOpBuilder,
    ops::*,
};

#[derive(Debug, DialectRegistration)]
pub struct ArithDialect {
    info: DialectInfo,
}

impl From<DialectInfo> for ArithDialect {
    fn from(info: DialectInfo) -> Self {
        Self { info }
    }
}

impl Dialect for ArithDialect {
    #[inline]
    fn info(&self) -> &DialectInfo {
        &self.info
    }

    /// Materializes an integer-like `attr` of type `ty` as an `arith.constant`, and a
    /// [FloatAttr] of type `ty` as an `arith.constantf`.
    ///
    /// An `IntAttr` is installed as it is; any other integer-like attribute (a `BoolAttr`, say)
    /// gives a new `IntAttr` of its value. A signless value has no default extension, so an
    /// attribute of another type than `ty` is not materialized.
    fn materialize_constant(
        &self,
        builder: &mut dyn Builder,
        attr: AttributeRef,
        ty: Type,
        span: SourceSpan,
    ) -> Option<OperationRef> {
        let mut builder = runic_ir::InsertionGuard::new(builder);

        if let Ok(float) = attr.try_downcast_attr::<FloatAttr>() {
            // An attribute whose type is not a float type of its value's width is not
            // materialized either.
            let borrowed = float.borrow();
            if runic_ir::Attribute::ty(&*borrowed) != ty
                || Float::width_for_type(ty) != Some(borrowed.as_value().width())
            {
                return None;
            }
            drop(borrowed);
            let op_builder = builder.create::<Constantf, _>(span);
            let op = op_builder(ReadyAttr(float)).ok()?;
            return Some(op.as_operation_ref());
        }

        let (value, value_ty, int_attr) = {
            let borrowed = attr.borrow();
            let integer = borrowed.as_attr().as_trait::<dyn IntegerLikeAttr>()?;
            (integer.as_int(), integer.ty(), attr.try_downcast_attr::<IntAttr>().ok())
        };
        // An `IntAttr` whose value is not as wide as a constant of its type is invalid (see
        // `IntAttr`'s verifier); it is not materialized either.
        if value_ty != ty || Int::width_for_type(ty) != Some(value.width()) {
            return None;
        }
        let attr = int_attr.unwrap_or_else(|| builder.context_rc().int_attr(value, ty));
        let op_builder = builder.create::<Constant, _>(span);
        let op = op_builder(ReadyAttr(attr)).ok()?;
        Some(op.as_operation_ref())
    }
}
