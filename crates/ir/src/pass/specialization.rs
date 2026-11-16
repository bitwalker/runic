use alloc::format;

use crate::{
    Context, EntityMut, EntityRef, Op, OpRegistration, Operation, OperationName, OperationRef,
    Report, Symbol, SymbolTable, traits::BranchOpInterface,
};

/// What a pass (or an analysis) operates on: a concrete operation type, any [Operation], or an
/// operation that implements some interface.
pub trait PassTarget {
    /// The name of the operation type this target is, or `None` when the target is not one
    /// operation type (any [Operation], `dyn Op`, or an interface).
    ///
    /// A registered operation type answers its name, registering its dialect in `context` first
    /// if that has not been done, as building an operation of that type does.
    fn target_name(context: &Context) -> Option<OperationName>;
    /// Whether an operation named `name` is of this target: one of the type for an operation
    /// type, one that implements the interface for an interface, and any operation for
    /// [Operation].
    fn accepts(name: &OperationName) -> bool;
    /// `op` as this target.
    ///
    /// Panics if `op` is not of this target.
    fn into_target(op: &OperationRef) -> EntityRef<'_, Self>;
    /// `op` as this target, mutably, or an error that names the operation `op` is when it is not
    /// of this target.
    fn into_target_mut(op: &mut OperationRef) -> Result<EntityMut<'_, Self>, Report>;
}

impl<T: 'static> PassTarget for T {
    default fn target_name(_context: &Context) -> Option<OperationName> {
        None
    }

    #[inline]
    default fn accepts(name: &OperationName) -> bool {
        name.is::<T>()
    }

    #[inline]
    #[track_caller]
    default fn into_target(op: &OperationRef) -> EntityRef<'_, T> {
        EntityRef::map(op.borrow(), |t| {
            t.downcast_ref::<T>().unwrap_or_else(|| expected_type::<T>(op))
        })
    }

    #[inline]
    #[track_caller]
    default fn into_target_mut(op: &mut OperationRef) -> Result<EntityMut<'_, T>, Report> {
        into_type_mut::<T>(op, format_args!("a `{}`", core::any::type_name::<T>()))
    }
}
impl<T: OpRegistration> PassTarget for T {
    fn target_name(context: &Context) -> Option<OperationName> {
        context
            .get_or_register_dialect::<<T as OpRegistration>::Dialect>()
            .registered_name::<T>()
    }

    fn into_target_mut(op: &mut OperationRef) -> Result<EntityMut<'_, T>, Report> {
        into_type_mut::<T>(op, format_args!("'{}'", <T as OpRegistration>::full_name()))
    }
}
impl PassTarget for Operation {
    #[inline(always)]
    fn target_name(_context: &Context) -> Option<OperationName> {
        None
    }

    #[inline(always)]
    fn accepts(_name: &OperationName) -> bool {
        true
    }

    #[inline]
    #[track_caller]
    fn into_target(op: &OperationRef) -> EntityRef<'_, Operation> {
        op.borrow()
    }

    #[inline]
    #[track_caller]
    fn into_target_mut(op: &mut OperationRef) -> Result<EntityMut<'_, Operation>, Report> {
        Ok(op.borrow_mut())
    }
}
impl PassTarget for dyn Op {
    #[inline(always)]
    fn target_name(_context: &Context) -> Option<OperationName> {
        None
    }

    #[inline]
    fn accepts(name: &OperationName) -> bool {
        name.implements::<dyn Op>()
    }

    fn into_target(op: &OperationRef) -> EntityRef<'_, dyn Op> {
        EntityRef::map(op.borrow(), |op| op.as_trait::<dyn Op>().unwrap())
    }

    fn into_target_mut(op: &mut OperationRef) -> Result<EntityMut<'_, dyn Op>, Report> {
        into_implementation_mut::<dyn Op>(op)
    }
}
impl PassTarget for dyn BranchOpInterface {
    #[inline(always)]
    fn target_name(_context: &Context) -> Option<OperationName> {
        None
    }

    #[inline]
    fn accepts(name: &OperationName) -> bool {
        name.implements::<dyn BranchOpInterface>()
    }

    #[track_caller]
    fn into_target(op: &OperationRef) -> EntityRef<'_, dyn BranchOpInterface> {
        EntityRef::map(op.borrow(), |t| {
            t.as_trait::<dyn BranchOpInterface>()
                .unwrap_or_else(|| expected_implementation::<dyn BranchOpInterface>(op))
        })
    }

    fn into_target_mut(
        op: &mut OperationRef,
    ) -> Result<EntityMut<'_, dyn BranchOpInterface>, Report> {
        into_implementation_mut::<dyn BranchOpInterface>(op)
    }
}
impl PassTarget for dyn Symbol {
    #[inline(always)]
    fn target_name(_context: &Context) -> Option<OperationName> {
        None
    }

    #[inline]
    fn accepts(name: &OperationName) -> bool {
        name.implements::<dyn Symbol>()
    }

    #[track_caller]
    fn into_target(op: &OperationRef) -> EntityRef<'_, dyn Symbol> {
        EntityRef::map(op.borrow(), |t| {
            t.as_trait::<dyn Symbol>()
                .unwrap_or_else(|| expected_implementation::<dyn Symbol>(op))
        })
    }

    fn into_target_mut(op: &mut OperationRef) -> Result<EntityMut<'_, dyn Symbol>, Report> {
        into_implementation_mut::<dyn Symbol>(op)
    }
}
impl PassTarget for dyn SymbolTable + 'static {
    #[inline(always)]
    fn target_name(_context: &Context) -> Option<OperationName> {
        None
    }

    #[inline]
    fn accepts(name: &OperationName) -> bool {
        name.implements::<dyn SymbolTable>()
    }

    #[track_caller]
    fn into_target(op: &OperationRef) -> EntityRef<'_, dyn SymbolTable + 'static> {
        EntityRef::map(op.borrow(), |t| {
            t.as_trait::<dyn SymbolTable>()
                .unwrap_or_else(|| expected_implementation::<dyn SymbolTable>(op))
        })
    }

    fn into_target_mut(
        op: &mut OperationRef,
    ) -> Result<EntityMut<'_, dyn SymbolTable + 'static>, Report> {
        into_implementation_mut::<dyn SymbolTable>(op)
    }
}

/// `op` as a `T`, mutably, or an error that says it expected `expected` and names the operation
/// `op` is.
fn into_type_mut<'a, T: 'static>(
    op: &'a mut OperationRef,
    expected: core::fmt::Arguments<'_>,
) -> Result<EntityMut<'a, T>, Report> {
    // The name is read, and the error built, before `op` is borrowed mutably.
    let found = {
        let op = op.borrow();
        (!op.is::<T>()).then(|| op.name())
    };
    if let Some(found) = found {
        return Err(Report::msg(format!("expected {expected}, but found '{found}'")));
    }
    Ok(EntityMut::map(op.borrow_mut(), |t| {
        t.downcast_mut::<T>().expect("the type of the operation was checked above")
    }))
}

/// `op` as an implementation of `Trait`, mutably, or an error that names the operation `op` is
/// when it does not implement `Trait`.
fn into_implementation_mut<Trait>(op: &mut OperationRef) -> Result<EntityMut<'_, Trait>, Report>
where
    Trait: ?Sized + core::ptr::Pointee<Metadata = core::ptr::DynMetadata<Trait>> + 'static,
{
    // The name is read, and the error built, before `op` is borrowed mutably.
    let found = {
        let op = op.borrow();
        (!op.implements::<Trait>()).then(|| op.name())
    };
    if let Some(found) = found {
        return Err(Report::msg(format!(
            "expected an operation that implements `{}`, but found '{found}'",
            core::any::type_name::<Trait>()
        )));
    }
    Ok(EntityMut::map(op.borrow_mut(), |t| {
        t.as_trait_mut::<Trait>().expect("the implementation was checked above")
    }))
}

#[cold]
#[inline(never)]
#[track_caller]
fn expected_type<T: 'static>(op: &OperationRef) -> ! {
    panic!(
        "expected operation '{}' to be a `{}`",
        op.borrow().name(),
        core::any::type_name::<T>(),
    )
}

#[cold]
#[inline(never)]
#[track_caller]
fn expected_implementation<Trait: ?Sized + 'static>(op: &OperationRef) -> ! {
    panic!(
        "expected '{}' to implement `{}`, but no vtable was found",
        op.borrow().name(),
        core::any::type_name::<Trait>()
    )
}
