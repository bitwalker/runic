//! The `arith` dialect: constants, arithmetic, bitwise operations, comparisons and casts on
//! signless integers.
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
mod ops;
#[cfg(test)]
mod tests;

use runic_ir::{
    AttributeRef, Builder, BuilderExt, Dialect, DialectInfo, Int, OperationRef, ReadyAttr,
    SourceSpan, Type, attributes::IntegerLikeAttr, derive::DialectRegistration,
    dialects::builtin::attributes::IntAttr,
};

pub use self::{
    attributes::{CmpPredicate, CmpPredicateAttr},
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

    /// Materializes an integer-like `attr` of type `ty` as an `arith.constant`.
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
