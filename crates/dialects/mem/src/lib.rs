//! The `mem` dialect: byte-addressed memory.
//!
//! `mem.load` reads a value through a pointer, `mem.store` writes one, and `mem.alloca` allocates
//! storage that lives until the enclosing function returns. Pointers are typed (`ptr<T, a>`) or
//! opaque (`ptr`); a load through an opaque pointer states the type it reads. `mem.gep` computes
//! the address of an element; `mem.sizeof`, `mem.alignof` and `mem.offsetof` give the size,
//! alignment and element offsets of a type as `isize` values.
//!
//! No operation needs a data layout to be built, printed, parsed or type-checked. When a layout is
//! in scope ([DataLayout::closest](runic_core::DataLayout::closest)), the verifiers check that the
//! accessed or measured type has one, and a size, alignment or offset folds to a constant. Loads
//! and stores read and write [MemoryResource]; an invalid address is undefined behaviour of the
//! access, so an unused load is removed like any other read. `mem.gep` computes an address and
//! accesses nothing.

#![no_std]
#![feature(specialization)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![allow(incomplete_features)]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod attributes;
mod builders;
pub mod canonicalization;
mod ops;
#[cfg(test)]
mod tests;

use runic_core::{
    AttributeRef, Builder, Dialect, DialectInfo, OperationRef, SourceSpan, Type,
    attributes::IntegerLikeAttr, derive::DialectRegistration,
};
use runic_dialect_arith::ArithDialect;

pub use self::{
    attributes::{Alignment, AlignmentAttr},
    builders::MemOpBuilder,
    ops::*,
};

#[derive(Debug, DialectRegistration)]
#[dialect(name = "mem")]
pub struct MemDialect {
    info: DialectInfo,
}

impl From<DialectInfo> for MemDialect {
    fn from(info: DialectInfo) -> Self {
        Self { info }
    }
}

impl Dialect for MemDialect {
    #[inline]
    fn info(&self) -> &DialectInfo {
        &self.info
    }

    /// Materializes a constant that a `mem` operation folded to: an integer-like `attr` as an
    /// `arith.constant` (see [ArithDialect]). Any other attribute is not materialized.
    fn materialize_constant(
        &self,
        builder: &mut dyn Builder,
        attr: AttributeRef,
        ty: Type,
        span: SourceSpan,
    ) -> Option<OperationRef> {
        let context = builder.context_rc();
        let is_integer = attr.borrow().as_attr().as_trait::<dyn IntegerLikeAttr>().is_some();
        if !is_integer {
            return None;
        }
        let arith = context.get_or_register_dialect::<ArithDialect>();
        arith.materialize_constant(builder, attr, ty, span)
    }
}
