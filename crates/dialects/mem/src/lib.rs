//! The `mem` dialect: byte-addressed memory.
//!
//! Pointers are typed (`ptr<T, a>`) or opaque (`ptr`), in an integer address space; lengths,
//! sizes, offsets and indices are `isize`. The operations:
//!
//! - [`mem.load`](Load) reads a value through a pointer (through an opaque one it states the type).
//! - [`mem.store`](Store) writes a value through a pointer.
//! - [`mem.alloca`](Alloca) allocates storage that lives until the enclosing function returns.
//! - [`mem.gep`](Gep) computes the address of an element from a typed pointer, an optional index and
//!   a constant path.
//! - [`mem.sizeof`](Sizeof), [`mem.alignof`](Alignof) and [`mem.offsetof`](Offsetof) give the
//!   size, alignment and element offset of a type.
//! - [`mem.memcpy`](MemCpy) copies bytes between regions that must not overlap.
//! - [`mem.memmove`](MemMove) copies bytes between regions that may overlap.
//! - [`mem.memset`](MemSet) fills bytes with an `i8`.
//! - [`mem.ptrtoint`](PtrToInt) and [`mem.inttoptr`](IntToPtr) convert between a pointer and an
//!   `isize`, zero-extending or truncating when the layout gives them different widths.
//! - [`mem.ptrcast`](PtrCast) changes a pointer's type within its address space.
//!
//! **Resources and aliases.** Every access reads or writes [MemoryResource], a child of the
//! default resource: an access is ordered against every effect on the default resource (a call
//! with unknown effects) and against no effect on another of its children (a `cf.assert`). Nothing
//! reasons about which addresses may alias: a write to memory blocks the merging of reads of
//! memory whatever the addresses, and two loads merge only when they have the same operands, so a
//! load through `mem.ptrcast %p` and a load through `%p` are two loads. An invalid address is
//! undefined behaviour of the access, so an unused load is removed like any other read; a store,
//! copy or fill is never removed, except a copy or fill of the constant length 0. `mem.gep`, the
//! size operations and the casts access nothing and have no effects.
//!
//! **What the generic passes do not do.** A store to any address blocks the merging of loads
//! through every other pointer, a store to a fresh `mem.alloca` included. Nothing forwards a stored
//! value to a later load, removes a store that is never read, or promotes an alloca to a value
//! (`Local2Reg` promotes function locals, not memory). Loads merge within one block only. A
//! `cf.assert` between two loads does not block their merging, but it does block the merging of
//! two calls to a function declared `effects(read)`: such a call reads the default resource, of
//! which the assert's resource is a child.
//!
//! **Layouts.** No operation needs a data layout to be built, printed, parsed or type-checked. A
//! layout is read only through [DataLayout::closest](runic_ir::DataLayout::closest), and only by
//! verifiers and folds: when one is in scope, the verifiers check that the accessed or measured
//! type has a layout under it, and a size, alignment or offset folds to a constant; when none is,
//! the checks are skipped and nothing folds.
//!
//! **Folds.** Under a data layout: `mem.sizeof`, `mem.alignof` and `mem.offsetof` fold to `isize`
//! constants when the answer fits the layout's `isize`; `mem.ptrtoint (mem.inttoptr x)` folds to
//! `x` when a pointer of the `mem.inttoptr`'s address space is at least as wide as `isize`, and
//! `mem.inttoptr (mem.ptrtoint p)` to `p` when the `mem.inttoptr`'s type is `p`'s type and `isize`
//! is at least as wide as the pointer. With or without one: a `mem.ptrcast` to its operand's type folds to
//! the operand, and a cast of a cast becomes one cast; a `mem.gep` with neither index nor path
//! folds to its base, and one without an index whose base is a `mem.gep` becomes one gep; a copy
//! or fill of the constant length 0 is erased.
//!
//! **Initializers.** [evaluate_initializer] gives what a `builtin.global_variable`'s initializer
//! denotes under a data layout, for every backend: the global's bytes ([Initializer]: its size,
//! its alignment and its bytes), with a [Relocation] wherever they hold the address of another
//! global variable plus a constant ([Target::Data]) or a function value ([Target::Function], the
//! function an alias leads to). The bytes under a relocation are zero; a backend writes there what
//! the target resolves to (an address, a table slot, a relocation of an object file). It reads
//! constants through the operations' own folds, the zero of a type (`builtin.zero`), structs and
//! arrays built with `builtin.aggregate`, addresses built with `builtin.global_symbol`, `mem.gep`
//! with a constant index, `mem.ptrcast`, and `mem.ptrtoint` and `mem.inttoptr` between a pointer
//! and an integer exactly as wide, and `builtin.function_ref`, which needs the layout's `fn`
//! entry. It does not change the IR. An initializer holds no operation with effects (the verifier
//! of `builtin.global_variable` says so), so nothing it evaluates reads or writes memory. Refused,
//! with an error whose primary label is on the operation that defines the value: arithmetic on an
//! address (an offset is written with `mem.gep`), an operation that does not fold to a constant or
//! that computes with a function value, a value that does not fit its type, a cast between a
//! pointer and an integer of another width, and a type with no layout; a declaration has no
//! initializer to evaluate. A wholly zero global has no bytes, and is recognised without building
//! them. The layout passed places the values; the folds read their own,
//! [DataLayout::closest](runic_ir::DataLayout::closest) of the operation, which is the same when
//! the caller passes the layout in scope of the global.

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
mod initializer;
mod ops;
#[cfg(test)]
mod tests;

use runic_dialect_arith::ArithDialect;
use runic_ir::{
    AttributeRef, Builder, Dialect, DialectInfo, OperationRef, SourceSpan, Type,
    attributes::IntegerLikeAttr, derive::DialectRegistration,
};

pub use self::{
    attributes::{Alignment, AlignmentAttr},
    builders::MemOpBuilder,
    initializer::{Initializer, Relocation, Target, evaluate_initializer},
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
