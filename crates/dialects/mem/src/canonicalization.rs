//! The canonicalization patterns of the `mem` dialect.

use alloc::rc::Rc;

use runic_core::{
    attributes::IntegerLikeAttr,
    dialects::builtin::attributes::Array,
    matchers::Matcher,
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind, RewritePattern},
    *,
};

use crate::*;

/// Makes a `mem.gep` with no index whose base is another `mem.gep` one gep: the inner gep's base
/// and index, and the inner path followed by the outer one.
///
/// `mem.gep (mem.gep %p, %i, [1]), [2]` is `mem.gep %p, %i, [1, 2]`: the outer path selects inside
/// the element the inner gep selected. A gep with an index is left alone: its index steps over
/// values of the inner gep's element type, which one gep on the inner base cannot say in general.
pub struct MergeGepOfGep {
    info: PatternInfo,
}

impl MergeGepOfGep {
    /// Creates the pattern, matching `mem.gep`.
    pub fn new(context: Rc<Context>) -> Self {
        let gep = context.get_or_register_dialect::<MemDialect>().expect_registered_name::<Gep>();
        Self {
            info: PatternInfo::new(
                context,
                "merge-gep-of-gep",
                PatternKind::Operation(gep),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for MergeGepOfGep {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for MergeGepOfGep {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let (base, index, path) = {
            let op = operation.borrow();
            let Some(outer) = op.downcast_ref::<Gep>() else {
                return Ok(false);
            };
            if outer.dynamic_index().is_some() {
                return Ok(false);
            }
            let Some(defining_op) = outer.base().as_value_ref().borrow().get_defining_op() else {
                return Ok(false);
            };
            let defining_op = defining_op.borrow();
            let Some(inner) = defining_op.downcast_ref::<Gep>() else {
                return Ok(false);
            };
            let index = inner.operands().group(1);
            if index.len() > 1 {
                return Ok(false);
            }
            let index = index
                .iter()
                .map(|operand| operand.borrow().as_value_ref())
                .collect::<SmallVec<[ValueRef; 1]>>();
            let mut path: Array<i32> = inner.get_path().clone();
            for step in outer.get_path().iter() {
                path.push(*step);
            }
            (inner.base().as_value_ref(), index, path)
        };

        let _guard = rewriter.modify_op_in_place(operation);
        let context = operation.borrow().context_rc();
        let mut target = operation;
        let mut op = target.borrow_mut();
        let gep = op.downcast_mut::<Gep>().expect("matched a gep");
        gep.base_mut().set(base);
        gep.index_mut().set_operands(index, operation, &context);
        gep.set_path(path);
        Ok(true)
    }
}

/// Makes a `mem.ptrcast` of a `mem.ptrcast` one cast of the inner operand to the outer type:
/// `mem.ptrcast (mem.ptrcast %p : ptr<i8>) : ptr<i16>` is `mem.ptrcast %p : ptr<i16>`.
///
/// Both casts keep the address space, so the merged one does too. When the outer type is the
/// type of `%p`, the cast then folds to `%p`.
pub struct MergePtrCastOfPtrCast {
    info: PatternInfo,
}

impl MergePtrCastOfPtrCast {
    /// Creates the pattern, matching `mem.ptrcast`.
    pub fn new(context: Rc<Context>) -> Self {
        let cast = context
            .get_or_register_dialect::<MemDialect>()
            .expect_registered_name::<PtrCast>();
        Self {
            info: PatternInfo::new(
                context,
                "merge-ptrcast-of-ptrcast",
                PatternKind::Operation(cast),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for MergePtrCastOfPtrCast {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for MergePtrCastOfPtrCast {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let addr = {
            let op = operation.borrow();
            let Some(outer) = op.downcast_ref::<PtrCast>() else {
                return Ok(false);
            };
            let Some(defining_op) = outer.addr().as_value_ref().borrow().get_defining_op() else {
                return Ok(false);
            };
            let defining_op = defining_op.borrow();
            let Some(inner) = defining_op.downcast_ref::<PtrCast>() else {
                return Ok(false);
            };
            inner.addr().as_value_ref()
        };

        let _guard = rewriter.modify_op_in_place(operation);
        let mut target = operation;
        let mut op = target.borrow_mut();
        let cast = op.downcast_mut::<PtrCast>().expect("matched a ptrcast");
        cast.addr_mut().set(addr);
        Ok(true)
    }
}

/// Erases a `mem.memcpy`, `mem.memmove` or `mem.memset` whose length is the constant 0: it
/// accesses no memory.
///
/// The length is read through the constant that defines it, as any integer-like attribute (an
/// `isize` constant holds a 64-bit value). A length that is not a constant, or not 0, is left
/// alone: the operation writes memory, so nothing else removes it.
pub struct EraseZeroLength {
    info: PatternInfo,
}

impl EraseZeroLength {
    /// Creates the pattern, matching the bulk operation `T`, whose last operand is its length.
    pub fn new<T: OpRegistration>(context: Rc<Context>) -> Self {
        let name = context.get_or_register_dialect::<MemDialect>().expect_registered_name::<T>();
        Self {
            info: PatternInfo::new(
                context,
                "erase-zero-length",
                PatternKind::Operation(name),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for EraseZeroLength {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for EraseZeroLength {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let is_empty = {
            let op = operation.borrow();
            let is_bulk = op.is::<MemCpy>() || op.is::<MemMove>() || op.is::<MemSet>();
            let Some(len) = op.operands().iter().last().filter(|_| is_bulk) else {
                return Ok(false);
            };
            let Some(defining_op) = len.borrow().as_value_ref().borrow().get_defining_op() else {
                return Ok(false);
            };
            let Some(attr) = matchers::constant().matches(&*defining_op.borrow()) else {
                return Ok(false);
            };
            let attr = attr.borrow();
            attr.as_attr()
                .as_trait::<dyn IntegerLikeAttr>()
                .is_some_and(|value| value.as_int().is_zero())
        };
        if !is_empty {
            return Ok(false);
        }
        rewriter.erase_op(operation);
        Ok(true)
    }
}
