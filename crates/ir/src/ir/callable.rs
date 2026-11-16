mod resolution;

use core::fmt;

pub use self::resolution::{CanonicalCallableRef, ResolvedSymbolCallee, SymbolResolutionError};
use crate::{
    EntityRef, Op, OpOperandRange, OpOperandRangeMut, Operation, RegionRef, Report, Spanned,
    Symbol, SymbolPath, SymbolRef, UnsafeIntrusiveEntityRef, Value, ValueRef,
    dialects::builtin::attributes::{DeclaredEffects, Signature, SymbolRefAttr},
};

/// Checks that the operation `exit`, which leaves its callable (see
/// [Operation::leaves_callable]), reaches `callable` only through operations that forward exits.
///
/// Every operation strictly between `exit` and `callable` (the parent of `exit`, its parent, and so
/// on up to, and not including, `callable`) must implement [PropagateControlFlowBreak]. The first
/// one, from the innermost out, that does not is the error: an `invalid operation` report on
/// `callable`'s name whose primary label is on `exit` and names the operation it crosses, with a
/// second label on that operation and a context label on `callable`.
///
/// The rule is that of `scf.break`, extended to the exits of a callable. An operation that does not
/// forward exits is one whose regions are not known to be left this way, and the analyses that
/// follow control flow through regions (the dead code analysis, the tail call analysis) rely on
/// the declaration to know which operations an exit may cross: an exit under an operation that has
/// not declared it is invalid IR rather than something an analysis must guess at. An operation
/// that is not an exit, or is one that is not nested in `callable`, is accepted: there is nothing
/// between them to check.
///
/// A callable operation with a body, such as `builtin.function`, calls this for each exit its
/// verifier finds in its body.
///
/// [PropagateControlFlowBreak]: crate::PropagateControlFlowBreak
pub fn verify_exit_crosses_only_forwarding_ops(
    exit: &Operation,
    callable: &Operation,
) -> Result<(), Report> {
    use alloc::format;

    use crate::diagnostics::report;

    if !exit.leaves_callable() {
        return Ok(());
    }
    let callable_ref = callable.as_operation_ref();
    let mut parent = exit.parent_op();
    while let Some(crossed_ref) = parent {
        if crossed_ref == callable_ref {
            return Ok(());
        }
        let crossed = crossed_ref.borrow();
        if !crossed.implements::<dyn crate::PropagateControlFlowBreak>() {
            let name = callable.name();
            let kind = name.name();
            let reason = format!(
                "this `{}` leaves the {kind} through `{}`, which has not declared that it \
                 forwards breaking control flow",
                exit.name(),
                crossed.name()
            );
            let crossed_label =
                format!("`{}` does not forward exits of the {kind}", crossed.name());
            let callable_label = format!("in this {kind}");
            return Err(report!(
                message: (format!("invalid operation {name}")),
                labels: [
                    primary(exit.span(), reason),
                    context(crossed.span(), crossed_label),
                    context(callable.span(), callable_label),
                ],
            ));
        }
        parent = crossed.parent_op();
    }
    Ok(())
}

/// A call-like operation is one that transfers control from one function to another.
///
/// These operations may be traditional static calls, e.g. `call @foo`, or indirect calls, e.g.
/// `call_indirect v1`. An operation that uses this interface cannot _also_ implement the
/// `CallableOpInterface`.
pub trait CallOpInterface: Op {
    /// Get the callee of this operation.
    ///
    /// A callee is either a symbol, or a reference to an SSA value.
    fn callable_for_callee(&self) -> Callable;
    /// Sets the callee for this operation.
    fn set_callee(&mut self, callable: Callable);
    /// Get the operands of this operation that are used as arguments for the callee
    fn arguments(&self) -> OpOperandRange<'_>;
    /// Get a mutable reference to the operands of this operation that are used as arguments for the
    /// callee
    fn arguments_mut(&mut self) -> OpOperandRangeMut<'_>;
    /// Resolve the callable operation for the current callee to a `CallableOpInterface`, or `None`
    /// if a valid callable was not resolved. Uses the provided symbol table for the initial
    /// lookup.
    ///
    /// Symbol aliases must be followed to their canonical target, with each alias hop resolved in
    /// that alias's own symbol table rather than the provided one. This does not change the symbol
    /// reference stored on the call.
    ///
    /// This method is used to perform callee resolution using a cached symbol table, rather than
    /// traversing the operation hierarchy looking for symbol tables to try resolving with.
    fn resolve_in_symbol_table(
        &self,
        symbols: &dyn crate::SymbolTable,
    ) -> Option<CanonicalCallableRef> {
        let callee = self.callable_for_callee();
        let path = callee.as_symbol_path()?;
        symbols.resolve_callable(path).ok().map(|callee| callee.target())
    }
    /// Resolve a symbolic call to its named symbol and canonical target.
    fn resolve_symbol_callee(&self) -> Result<ResolvedSymbolCallee, SymbolResolutionError> {
        let table = self
            .as_operation()
            .nearest_symbol_table()
            .ok_or(SymbolResolutionError::NoSymbolTable)?;
        let table = table.borrow();
        let callee = self.callable_for_callee();
        let path = callee.as_symbol_path().ok_or(SymbolResolutionError::NonSymbolCallee)?;
        table
            .as_symbol_table()
            .ok_or(SymbolResolutionError::NoSymbolTable)?
            .resolve_callable(path)
    }
    /// Resolve the stored callee to the symbol it names, *without* following symbol aliases.
    ///
    /// Returns the symbol the call actually references. Unlike [Self::resolve], the result may be a
    /// symbol alias (e.g. [`FunctionAlias`](crate::dialects::builtin::FunctionAlias))
    ///
    /// Returns `None` if the callee is not a symbol, or does not resolve.
    fn resolve_stored_callee(&self) -> Option<SymbolRef> {
        let path = self.callable_for_callee().as_symbol_path()?.clone();
        let symbol_table = self.as_operation().nearest_symbol_table()?;
        let symbol_table = symbol_table.borrow();
        symbol_table.as_symbol_table()?.resolve(&path)
    }
    /// Resolve the callable operation for the current callee to a `CallableOpInterface`, or `None`
    /// if a valid callable was not resolved.
    ///
    /// Symbol aliases must be followed to their canonical target so calls and returns agree on
    /// the operation that owns the callable body. The stored callee remains unchanged.
    fn resolve(&self) -> Option<CanonicalCallableRef> {
        self.resolve_symbol_callee().ok().map(|callee| callee.target())
    }
    /// Enumerate every callable this operation may transfer control to, or `None` if the set of
    /// possible callees is not statically known.
    ///
    /// Direct calls have exactly one possible callee, which the default implementation derives
    /// from [Self::resolve]. Indirect calls whose dispatch set is statically known (e.g. a call of
    /// one of a fixed set of callables, chosen at runtime) override this to enumerate that set, which interprocedural
    /// analyses join over; an empty set means the call can never transfer control (every
    /// dispatch traps).
    ///
    /// Returned targets must be canonical (with symbol aliases resolved) and deduplicated.
    fn possible_callees(&self) -> Option<crate::SmallVec<[CanonicalCallableRef; 2]>> {
        self.resolve().map(|callee| crate::smallvec![callee])
    }
    /// The signature this call site expects of its callee, if one is known.
    ///
    /// The default derives it from the resolved callee, which is what a direct call wants. An
    /// indirect call has no single callee to derive it from but does carry the signature it
    /// dispatches with, and overrides this to return it — so consumers reasoning about a call's
    /// parameter types (taint analysis' external-call sinks, for one) work for both.
    fn callee_signature(&self) -> Option<Signature> {
        Some(self.resolve()?.signature())
    }
}

/// A callable operation is one who represents a potential function, and may be a target for a call-
/// like operation (i.e. implementations of `CallOpInterface`). These operations may be traditional
/// function ops (i.e. `Function`), as well as function reference-producing operations, such as an
/// op that creates closures, or captures a function by reference.
///
/// These operations may only contain a single region.
pub trait CallableOpInterface: Op {
    /// Returns the region executed by this callable.
    ///
    /// This may return `None` in the case of an external callable object, e.g. an externally-
    /// defined function reference.
    fn get_callable_region(&self) -> Option<RegionRef>;
    /// Returns the signature of the callable
    fn signature(&self) -> Signature;
    /// What this callable states about the effects of calling it. `Unknown` by default.
    ///
    /// Nothing is inferred from the callable's body: a callable that states nothing has unknown
    /// effects.
    ///
    /// Declaring no effects, or only reads (and allocations of a result), is the statement that a
    /// call of the callable may be removed when its results are unused, merged with an equal call,
    /// or moved past operations that do not write. A trap or a failure to return is not an effect,
    /// so a callable whose failure to return must be observed declares nothing, or a write. The one
    /// case the IR sees for itself is a signature that never returns (`-> never`): a call made with
    /// such a signature has unknown effects whatever its callee declares.
    fn declared_effects(&self) -> DeclaredEffects {
        DeclaredEffects::Unknown
    }
}

/// A marker trait for symbols that can be named by a call-like operation.
///
/// This does not imply [CallableOpInterface]: aliases name a callable without owning its
/// signature or region. Use [SymbolRef::resolve_callable] to access both identities.
pub trait CallableSymbol: Symbol {}

/// An alias for [`UnsafeIntrusiveEntityRef<dyn CallableSymbol>`]
pub type CallableSymbolRef = UnsafeIntrusiveEntityRef<dyn CallableSymbol>;

#[doc(hidden)]
pub trait AsCallableSymbolRef {
    fn as_callable_symbol_ref(&self) -> SymbolRef;
}
impl AsCallableSymbolRef for CallableSymbolRef {
    #[inline]
    fn as_callable_symbol_ref(&self) -> SymbolRef {
        *self as SymbolRef
    }
}
impl<T: CallableSymbol> AsCallableSymbolRef for T {
    #[inline(always)]
    fn as_callable_symbol_ref(&self) -> SymbolRef {
        self.as_symbol_operation()
            .as_symbol_ref()
            .expect("callable symbols must provide a symbol operation")
    }
}
impl<T: CallableSymbol> AsCallableSymbolRef for UnsafeIntrusiveEntityRef<T> {
    #[inline(always)]
    fn as_callable_symbol_ref(&self) -> SymbolRef {
        *self as SymbolRef
    }
}

/// A [Callable] represents a symbol or a value which can be used as a valid _callee_ for a
/// [CallOpInterface] implementation.
///
/// Symbols are not SSA values, but there are situations where we want to treat them as one, such
/// as indirect calls. Abstracting over whether the callable is a symbol or an SSA value allows us
/// to focus on the call semantics, rather than the difference between the type types of value.
#[derive(Debug, Clone)]
pub enum Callable {
    Symbol(SymbolPath),
    Value(ValueRef),
}
impl fmt::Display for Callable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Symbol(path) => fmt::Display::fmt(path, f),
            Self::Value(value) => fmt::Display::fmt(value, f),
        }
    }
}
impl From<&SymbolRefAttr> for Callable {
    fn from(value: &SymbolRefAttr) -> Self {
        Self::Symbol(value.path().clone())
    }
}
impl From<&SymbolPath> for Callable {
    fn from(value: &SymbolPath) -> Self {
        Self::Symbol(value.clone())
    }
}
impl From<SymbolPath> for Callable {
    fn from(value: SymbolPath) -> Self {
        Self::Symbol(value)
    }
}
impl From<ValueRef> for Callable {
    fn from(value: ValueRef) -> Self {
        Self::Value(value)
    }
}
impl Callable {
    #[inline(always)]
    pub fn new(callable: impl Into<Self>) -> Self {
        callable.into()
    }

    pub fn is_symbol(&self) -> bool {
        matches!(self, Self::Symbol(_))
    }

    pub fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    pub fn as_symbol_path(&self) -> Option<&SymbolPath> {
        match self {
            Self::Symbol(name) => Some(name),
            _ => None,
        }
    }

    pub fn as_value(&self) -> Option<EntityRef<'_, dyn Value>> {
        match self {
            Self::Value(value_ref) => Some(value_ref.borrow()),
            _ => None,
        }
    }

    pub fn unwrap_symbol_path(self) -> SymbolPath {
        match self {
            Self::Symbol(name) => name,
            Self::Value(value_ref) => panic!("expected symbol, got {}", value_ref.borrow().id()),
        }
    }

    pub fn unwrap_value_ref(self) -> ValueRef {
        match self {
            Self::Value(value) => value,
            Self::Symbol(ref name) => panic!("expected value, got {name}"),
        }
    }
}
