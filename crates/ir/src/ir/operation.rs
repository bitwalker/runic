mod builder;
pub mod equivalence;
mod name;
mod state;
#[cfg(test)]
mod verify_tests;

use alloc::rc::Rc;
use core::{
    fmt,
    ptr::{DynMetadata, NonNull, Pointee},
    sync::atomic::AtomicU32,
};

use smallvec::SmallVec;

pub use self::{
    builder::{GenericOperationBuilder, OperationBuilder},
    name::{AttrInfo, OperationName, ParseAssemblyFn},
    state::{OperationState, PendingSuccessorInfo},
};
use super::{
    effects::{
        Effect, EffectInstance, HasRecursiveMemoryEffects, MemoryEffect, MemoryEffectOpInterface,
    },
    *,
};
use crate::{
    Attribute, AttributeRef, AttributeRegistration, Forward, NamedAttribute, ProgramPoint,
    attributes::OpAttribute, dialects::builtin::attributes::SymbolRefAttr,
    patterns::RewritePatternSet,
};

pub type OperationRef = UnsafeIntrusiveEntityRef<Operation>;
pub type OpList = EntityList<Operation>;
pub type OpCursor<'a> = EntityListCursor<'a, Operation>;
pub type OpCursorMut<'a> = EntityListCursorMut<'a, Operation>;

/// The [Operation] struct provides the common foundation for all [Op] implementations.
///
/// It provides:
///
/// * Support for casting between the concrete operation type `T`, `dyn Op`, the underlying
///   `Operation`, and any of the operation traits that the op implements. Not only can the casts
///   be performed, but an [Operation] can be queried to see if it implements a specific trait at
///   runtime to conditionally perform some behavior. This makes working with operations in the IR
///   very flexible and allows for adding or modifying operations without needing to change most of
///   the compiler, which predominately works on operation traits rather than concrete ops.
/// * Storage for all IR entities attached to an operation, e.g. operands, results, nested regions,
///   attributes, etc.
/// * Navigation of the IR graph; navigate up to the containing block/region/op, down to nested
///   regions/blocks/ops, or next/previous sibling operations in the same block. Additionally, you
///   can navigate directly to the definitions of operands used, to users of results produced, and
///   to successor blocks.
/// * Many utility functions related to working with operations, many of which are also accessible
///   via the [Op] trait, so that working with an [Op] or an [Operation] are largely
///   indistinguishable.
///
/// All [Op] implementations can be cast to the underlying [Operation], but most of the
/// fucntionality is re-exported via default implementations of methods on the [Op] trait. The main
/// benefit is avoiding any potential overhead of casting when going through the trait, rather than
/// calling the underlying [Operation] method directly.
///
/// # Safety
///
/// [Operation] is implemented as part of a larger structure that relies on assumptions which depend
/// on IR entities being allocated via [Context], i.e. the arena. Those allocations produce an
/// [UnsafeIntrusiveEntityRef] or [UnsafeEntityRef], which allocate the pointee type inside a struct
/// that provides metadata about the pointee that can be accessed without aliasing the pointee
/// itself - in particular, links for intrusive collections. This is important, because while these
/// pointer types are a bit like raw pointers in that they lack any lifetime information, and are
/// thus unsafe to dereference in general, they _do_ ensure that the pointee can be safely reified
/// as a reference without violating Rust's borrow checking rules, i.e. they are dynamically borrow-
/// checked.
///
/// The reason why we are able to generally treat these "unsafe" references as safe, is because we
/// require that all IR entities be allocated via [Context]. This makes it essential to keep the
/// context around in order to work with the IR, and effectively guarantees that no [RawEntityRef]
/// will be dereferenced after the context is dropped. This is not a guarantee provided by the
/// compiler however, but one that is imposed in practice, as attempting to work with the IR in
/// any capacity without a [Context] is almost impossible. We must ensure however, that we work
/// within this set of rules to uphold the safety guarantees.
///
/// This "fragility" is a tradeoff - we get the performance characteristics of an arena-allocated
/// IR, with the flexibility and power of using pointers rather than indexes as handles, while also
/// maintaining the safety guarantees of Rust's borrowing system. The downside is that we can't just
/// allocate IR entities wherever we want and use them the same way.
#[derive(Spanned)]
pub struct Operation {
    /// The [Context] in which this [Operation] was allocated.
    context: NonNull<Context>,
    /// The dialect and opcode name for this operation, as well as trait implementation metadata
    name: OperationName,
    /// The offset of the field containing this struct inside the concrete [Op] it represents.
    ///
    /// This is required in order to be able to perform casts from [Operation]. An [Operation]
    /// cannot be constructed without providing it to the `uninit` function, and callers of that
    /// function are required to ensure that it is correct.
    offset: usize,
    /// The order of this operation in its containing block
    ///
    /// This is atomic to ensure that even if a mutable reference to this operation is held, loads
    /// of this field cannot be elided, as the value can still be mutated at any time. In practice,
    /// the only time this is ever written, is when all operations in a block have their orders
    /// recomputed, or when a single operation is updating its own order.
    order: AtomicU32,
    #[span]
    pub span: SourceSpan,
    /// The attribute dictionary for this operation (does not include properties)
    pub attrs: EntityMap<OpAttribute>,
    /// The set of operands for this operation
    ///
    /// NOTE: If the op supports immediate operands, the storage for the immediates is handled
    /// by the op, rather than here. Additionally, the semantics of the immediate operands are
    /// determined by the op, e.g. whether the immediate operands are always applied first, or
    /// what they are used for.
    pub operands: OpOperandStorage,
    /// The set of values produced by this operation.
    pub results: OpResultStorage,
    /// If this operation represents control flow, this field stores the set of successors,
    /// and successor operands.
    pub successors: OpSuccessorStorage,
    /// The set of regions belonging to this operation, if any
    pub regions: RegionList,
}

/// Equality over operations is determined by reference identity, i.e. two operations are only equal
/// if they refer to the same address in memory, regardless of the content of the operation itself.
impl Eq for Operation {}
impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::addr_eq(self, other)
    }
}

/// The Hash implementation for operations is defined to match the equality implementation, i.e.
/// the hash of an operation is the hash of its address in memory.
impl core::hash::Hash for Operation {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::ptr::hash(self, state)
    }
}

impl fmt::Debug for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Operation")
            .field_with("name", |f| write!(f, "{}", self.name()))
            .field("offset", &self.offset)
            .field("order", &self.order)
            .field("attrs", &self.attrs)
            .field("block", &self.parent().as_ref().map(|b| b.borrow().id()))
            .field_with("operands", |f| {
                let mut list = f.debug_list();
                for operand in self.operands().all() {
                    list.entry(&operand.borrow());
                }
                list.finish()
            })
            .field("results", &self.results)
            .field("successors", &self.successors)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for OperationRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt::Debug::fmt(&self.borrow(), f)
    }
}

impl fmt::Display for OperationRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl AsRef<dyn Op> for Operation {
    fn as_ref(&self) -> &dyn Op {
        self.name.upcast(self.container()).unwrap()
    }
}

impl AsMut<dyn Op> for Operation {
    fn as_mut(&mut self) -> &mut dyn Op {
        self.name.upcast_mut(self.container().cast_mut()).unwrap()
    }
}

impl Entity for Operation {}
impl EntityWithParent for Operation {
    type Parent = Block;
}
impl EntityListItem for Operation {
    fn on_inserted(this: OperationRef, _cursor: &mut EntityListCursorMut<'_, Self>) {
        // NOTE: We use OperationName, instead of the Operation itself, to avoid borrowing.
        if this.name().implements::<dyn Symbol>() {
            let parent = this.nearest_symbol_table();
            if let Some(mut parent) = parent
                && parent.name().implements::<dyn SymbolTable>()
            {
                // NOTE: We call `unwrap()` here because we are confident that these function calls
                // are valid thanks to the `implements` check above.
                let mut symbol_table = parent.borrow_mut();
                let sym_manager = symbol_table.as_trait_mut::<dyn SymbolTable>().unwrap();
                let mut sym_manager = sym_manager.symbol_manager_mut();

                let symbol_ref = this.borrow().as_symbol_ref().unwrap();

                // Builders check this before creating the op (see `<dyn SymbolTable>::can_define`)
                let is_new = sym_manager.insert_new(symbol_ref, ProgramPoint::Invalid);
                assert!(
                    is_new,
                    "unable to insert {} `{}` in the symbol table of {}: a symbol with the same \
                     name or path is already registered",
                    this.name(),
                    symbol_ref.borrow().name(),
                    parent.name(),
                );
            }
        }
        let order_offset = core::mem::offset_of!(Operation, order);
        unsafe {
            let ptr = UnsafeIntrusiveEntityRef::as_ptr(&this);
            let order_ptr = ptr.byte_add(order_offset).cast::<AtomicU32>();
            (*order_ptr).store(Self::INVALID_ORDER, core::sync::atomic::Ordering::Release);
        }
    }

    fn on_transfer(_this: OperationRef, _from: &mut EntityList<Self>, to: &mut EntityList<Self>) {
        // Invalidate the ordering of the new parent block
        let mut to = to.parent();
        to.borrow_mut().invalidate_op_order();
    }

    /// A symbol taken out of a block leaves the symbol table it was registered in (the nearest one
    /// above the block, as [Self::on_inserted] finds it), which mutably borrows that table.
    ///
    /// `this` is already detached here, so the table is found from the block the list belongs to.
    /// `this` itself may be mutably borrowed (by [Operation::remove] and [Operation::erase]), so
    /// it is not borrowed when that would fail.
    fn on_removed(this: OperationRef, list: &mut EntityListCursorMut<'_, Self>) {
        // NOTE: We use OperationName, instead of the Operation itself, to avoid borrowing.
        if !this.name().implements::<dyn Symbol>() {
            return;
        }
        let Some(parent) = list.parent().and_then(|block| block.grandparent()) else {
            return;
        };
        let symbol_table = if parent.name().implements::<dyn SymbolTable>() {
            Some(parent)
        } else {
            parent.nearest_symbol_table()
        };
        let Some(mut symbol_table) = symbol_table else {
            return;
        };
        let symbol_ref = this.as_trait_ref::<dyn Symbol>().unwrap();
        let name = this.try_borrow().map(|op| op.as_symbol().unwrap().name());

        let mut symbol_table = symbol_table.borrow_mut();
        let sym_manager = symbol_table.as_trait_mut::<dyn SymbolTable>().unwrap();
        let mut sym_manager = sym_manager.symbol_manager_mut();
        sym_manager.symbols_mut().remove_symbol(symbol_ref, name);
    }
}

impl EntityParent<Region> for Operation {
    fn offset() -> usize {
        core::mem::offset_of!(Operation, regions)
    }
}

/// Construction
impl Operation {
    #[doc(hidden)]
    pub unsafe fn uninit<T: Op>(context: Rc<Context>, name: OperationName, offset: usize) -> Self {
        assert!(name.is::<T>());

        Self {
            context: unsafe { NonNull::new_unchecked(Rc::as_ptr(&context).cast_mut()) },
            name,
            offset,
            order: AtomicU32::new(0),
            span: Default::default(),
            attrs: Default::default(),
            operands: Default::default(),
            results: Default::default(),
            successors: Default::default(),
            regions: Default::default(),
        }
    }
}

/// Insertion
impl OperationRef {
    pub fn insert_at_start(self, mut block: BlockRef) {
        assert!(
            self.parent().is_none(),
            "cannot insert operation that is already attached to another block"
        );
        {
            let mut block = block.borrow_mut();
            block.body_mut().push_front(self);
        }
    }

    pub fn insert_at_end(self, mut block: BlockRef) {
        assert!(
            self.parent().is_none(),
            "cannot insert operation that is already attached to another block"
        );
        {
            let mut block = block.borrow_mut();
            block.body_mut().push_back(self);
        }
    }

    pub fn insert_before(self, before: OperationRef) {
        assert!(
            self.parent().is_none(),
            "cannot insert operation that is already attached to another block"
        );
        let mut block = before.parent().expect("'before' block is not attached to a block");
        {
            let mut block = block.borrow_mut();
            let block_body = block.body_mut();
            let mut cursor = unsafe { block_body.cursor_mut_from_ptr(before) };
            cursor.insert_before(self);
        }
    }

    pub fn insert_after(self, after: OperationRef) {
        assert!(
            self.parent().is_none(),
            "cannot insert operation that is already attached to another block"
        );
        let mut block = after.parent().expect("'after' block is not attached to a block");
        {
            let mut block = block.borrow_mut();
            let block_body = block.body_mut();
            let mut cursor = unsafe { block_body.cursor_mut_from_ptr(after) };
            cursor.insert_after(self);
        }
    }
}

/// Read-only Metadata
impl OperationRef {
    pub fn name(&self) -> OperationName {
        let ptr = OperationRef::as_ptr(self);
        // SAFETY: The `name` field of Operation is read-only after an op is allocated, and the
        // safety guarantees of OperationRef require that the allocation never moves for the
        // lifetime of the ref. So it is always safe to read this field via direct pointer, even
        // if a mutable borrow of the containing op exists, because the field is never written to
        // after allocation.
        unsafe {
            let name_ptr = core::ptr::addr_of!((*ptr).name);
            OperationName::clone(&*name_ptr)
        }
    }

    /// Returns this operation as a handle to an implemented trait object, if supported.
    ///
    /// The returned handle preserves the original intrusive allocation identity and swaps only the
    /// pointee metadata for the requested trait object.
    pub fn as_trait_ref<Trait>(self) -> Option<UnsafeIntrusiveEntityRef<Trait>>
    where
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        let ptr = self.name().upcast_raw::<Trait>(OperationRef::into_raw(self).cast())?;
        let (_, metadata) = ptr.to_raw_parts();
        Some(unsafe { self.cast_unsized_unchecked::<Trait>(metadata) })
    }

    /// Attempts to cast this handle to the concrete operation type `T`.
    ///
    /// This reuses the existing intrusive handle rather than reconstructing one from a borrowed
    /// reference.
    pub fn try_downcast_op<T: Op>(self) -> Result<UnsafeIntrusiveEntityRef<T>, Self> {
        if self.name().is::<T>() {
            Ok(unsafe { self.cast_unchecked::<T>() })
        } else {
            Err(self)
        }
    }

    /// Returns a handle to the nearest containing [Operation] of this operation, if it is attached
    /// to one
    pub fn parent_op(&self) -> Option<OperationRef> {
        self.parent_region().and_then(|region| region.parent())
    }

    /// Returns a handle to the containing [Region] of this operation, if it is attached to one
    pub fn parent_region(&self) -> Option<RegionRef> {
        self.grandparent()
    }

    /// Returns the nearest [SymbolTable] from this operation.
    ///
    /// Returns `None` if no parent of this operation is a valid symbol table.
    pub fn nearest_symbol_table(&self) -> Option<OperationRef> {
        let mut parent = self.parent_op();
        while let Some(parent_op) = parent.take() {
            if parent_op.name().implements::<dyn SymbolTable>() {
                return Some(parent_op);
            }
            parent = parent_op.parent_op();
        }
        None
    }
}

/// Metadata
impl Operation {
    /// Get the name of this operation
    ///
    /// An operation name consists of both its dialect, and its opcode.
    pub fn name(&self) -> OperationName {
        self.name.clone()
    }

    /// Get the dialect associated with this operation
    pub fn dialect(&self) -> Rc<dyn Dialect> {
        self.context().get_registered_dialect(self.name.dialect())
    }

    /// Set the source location associated with this operation
    #[inline]
    pub fn set_span(&mut self, span: SourceSpan) {
        self.span = span;
    }

    /// Get a borrowed reference to the owning [Context] of this operation
    #[inline(always)]
    pub fn context(&self) -> &Context {
        // SAFETY: This is safe so long as this operation is allocated in a Context, since the
        // Context by definition outlives the allocation.
        unsafe { self.context.as_ref() }
    }

    /// Get a owned reference to the owning [Context] of this operation
    pub fn context_rc(&self) -> Rc<Context> {
        // SAFETY: This is safe so long as this operation is allocated in a Context, since the
        // Context by definition outlives the allocation.
        //
        // Additionally, constructing the Rc from a raw pointer is safe here, as the pointer was
        // obtained using `Rc::as_ptr`, so the only requirement to call `Rc::from_raw` is to
        // increment the strong count, as `as_ptr` does not preserve the count for the reference
        // held by this operation. Incrementing the count first is required to manufacture new
        // clones of the `Rc` safely.
        unsafe {
            let ptr = self.context.as_ptr().cast_const();
            Rc::increment_strong_count(ptr);
            Rc::from_raw(ptr)
        }
    }
}

/// Verification
impl Operation {
    /// Check the operand group cardinalities declared by this operation's registration.
    ///
    /// This structural check runs before type inference and trait verification, both of which
    /// may access operands by their declared indices. Successor argument groups are independent
    /// of ordinary operands and are checked by the branch interface instead.
    pub fn verify_operand_cardinality(&self) -> Result<(), Report> {
        let declared = self.name().operand_group_sizes();
        let invalid_count = |group, expected, actual| {
            crate::diagnostics::report!(
                message: "invalid operand count",
                labels: [
                    primary(
                        self.span(),
                        alloc::format!(
                            "operand group {group} of '{}' expects {expected} operands, but has \
                             {actual}",
                            self.name()
                        )
                    ),
                ],
            )
        };
        for (group, expected) in declared.iter().copied().enumerate() {
            if group >= self.operands.num_groups() {
                return Err(crate::diagnostics::report!(
                    message: "missing operand group",
                    labels: [
                        primary(
                            self.span(),
                            alloc::format!("'{}' requires operand group {group}", self.name())
                        ),
                    ],
                ));
            }
            let actual = self.operands.group(group).len();
            if let Some(expected) = expected
                && actual != expected
            {
                return Err(invalid_count(group, expected, actual));
            }
        }

        // Unused reserved groups (including implicit group 0) may be empty. Any other populated
        // group must belong to a successor. A set keeps this linear for wide switches.
        let mut extra = self
            .operands
            .groups()
            .enumerate()
            .skip(declared.len())
            .filter_map(|(group, operands)| (!operands.is_empty()).then_some(group))
            .collect::<crate::FxHashSet<_>>();
        for successor in self.successors.iter() {
            extra.remove(&(successor.operand_group as usize));
        }
        if let Some(group) = extra.into_iter().min() {
            return Err(invalid_count(group, 0, self.operands.group(group).len()));
        }
        Ok(())
    }

    /// Check that this operation has at least the regions its registration declares (an
    /// operation may hold more, as `scf.index_switch` holds one per case after the declared
    /// default region), and the number of results it declares when that number is fixed.
    ///
    /// Like [Self::verify_operand_cardinality], this runs before type inference and the
    /// operation's own verifiers, which may access a declared region or result by its index.
    pub fn verify_region_and_result_counts(&self) -> Result<(), Report> {
        let name = self.name();
        let invalid = |what: &str, expected: alloc::string::String, actual: usize| {
            crate::diagnostics::report!(
                message: (alloc::format!("invalid {what} count")),
                labels: [
                    primary(
                        self.span(),
                        alloc::format!("'{name}' has {expected} {what}(s), but this has {actual}")
                    ),
                ],
            )
        };
        if self.regions.len() < name.region_count() {
            let expected = alloc::format!("at least {}", name.region_count());
            return Err(invalid("region", expected, self.regions.len()));
        }
        if let Some(expected) = name.result_count()
            && self.num_results() != expected
        {
            return Err(invalid("result", alloc::format!("{expected}"), self.num_results()));
        }
        Ok(())
    }

    /// Run any verifiers for this operation
    ///
    /// The generic checks run first, for every operation. The first, in one walk of the
    /// attributes, is that every attribute the operation holds, at any depth, is well formed by
    /// itself: its type may be used in IR of this context, and its kind's verifier
    /// ([AttributeRegistration::verify](crate::AttributeRegistration::verify)) accepts it; and
    /// that every symbol reference among them is linked to the symbol its path names (see
    /// [Self::verify_symbol_refs]). The second is that every type the operation holds, at any
    /// depth (see [Self::walk_types]), belongs to this context, is closed, and satisfies its
    /// kind's [TypeInvariants](crate::TypeInvariants) if it has any; a type found valid once, with
    /// everything it holds, is remembered by the context and not checked again. The operation's
    /// own verifiers run after these, so they may assume the operation's attributes, types and
    /// symbol references are valid.
    ///
    /// An attribute can be edited after it is built, and the constructors that skip the checks
    /// (and in-place mutation) do not check anything, so these checks are what catch an invalid
    /// attribute or type in every build.
    pub fn verify(&self) -> Result<(), Report> {
        // One walk of the attributes checks the attributes and the symbol references among them
        self.verify_held_attributes()?;
        self.verify_held_types()?;
        self.verify_dialect_attributes()?;
        let dyn_op: &dyn Op = self.as_ref();
        dyn_op.verify(self.context())
    }

    /// Check that every symbol reference among this operation's properties and attributes is
    /// linked to exactly the symbol its path names, looked up in the nearest symbol table of this
    /// operation (or in this operation, if it is a symbol table and has no symbol table above it),
    /// as the parser looks it up.
    ///
    /// The parser links each reference to the symbol it names once the whole input is parsed, and
    /// leaves one that names no symbol unlinked; the typed builders only accept a symbol that
    /// exists. Removing a symbol from its table, or erasing it, unlinks the references to it. This
    /// check is what turns a reference to a missing symbol into a diagnostic, and it also reports
    /// a reference whose link and path disagree (its symbol was detached and another inserted
    /// under its name, or a symbol table on its path was renamed), so that IR which verifies
    /// prints to text that parses back to the same links.
    ///
    /// Each reference's path is looked up once, which borrows this operation's nearest symbol
    /// table and the symbol tables on the path immutably.
    ///
    /// Every reference the operation holds is checked, at any depth, as [Self::walk_attributes]
    /// visits them: a reference nested in a list, say, is checked too. A reference whose use names
    /// another attribute, or belongs to another operation (a copy that shares the use of the
    /// reference it was copied from), is reported as well.
    pub fn verify_symbol_refs(&self) -> Result<(), Report> {
        self.walk_attributes(&mut |_, attr| match attr.try_downcast_attr::<SymbolRefAttr>() {
            Ok(reference) => self.verify_symbol_ref(reference).into(),
            Err(_) => WalkResult::Continue(()),
        })
        .into_result()
    }

    /// Check one symbol reference this operation holds: see [Self::verify_symbol_refs].
    fn verify_symbol_ref(
        &self,
        attr: UnsafeIntrusiveEntityRef<SymbolRefAttr>,
    ) -> Result<(), Report> {
        let print_path = |path: &crate::SymbolPath| {
            let flags = crate::OpPrintingFlags::default();
            let mut printer = crate::print::AsmPrinter::fragment(self.context_rc(), &flags);
            printer.print_symbol_path(path);
            printer.finish()
        };
        // The symbol `path` names, looked up as the parser does (see `resolve_symbol_uses`)
        let lookup = |path: &crate::SymbolPath| -> Option<SymbolRef> {
            match self.nearest_symbol_table() {
                Some(symbol_table) => symbol_table.borrow().as_symbol_table()?.resolve(path),
                None => self.as_symbol_table()?.resolve(path),
            }
        };
        let symbol_description = |symbol: SymbolRef| {
            let symbol = symbol.borrow();
            alloc::format!(
                "'{}' (a '{}')",
                print_path(&symbol.path()),
                symbol.as_symbol_operation().name()
            )
        };

        let user = attr.borrow().try_user();
        if let Some(user) = user {
            let symbol_use = user.borrow();
            if symbol_use.attr != attr || symbol_use.owner != self.as_operation_ref() {
                let printed = print_path(attr.borrow().path());
                return Err(crate::diagnostics::report!(
                    message: (alloc::format!(
                        "symbol '{printed}' shares the use of another reference"
                    )),
                    labels: [
                        primary(
                            self.span(),
                            alloc::format!(
                                "this reference, held by '{}', is tracked by a use that belongs \
                                 to another attribute or operation",
                                self.name()
                            )
                        ),
                    ],
                ));
            }
        }
        let linked = user.and_then(|user| user.linked_symbol());
        let path = attr.borrow().path().clone();
        let named = lookup(&path);
        if named.is_some() && named == linked {
            return Ok(());
        }
        let printed = print_path(&path);
        let Some(named) = named else {
            return Err(crate::diagnostics::report!(
                message: (alloc::format!("symbol '{printed}' does not resolve")),
                labels: [
                    primary(
                        self.span(),
                        alloc::format!("no symbol visible to '{}' has this path", self.name())
                    ),
                ],
            ));
        };
        let linked = match linked {
            Some(linked) => alloc::format!("linked to {}", symbol_description(linked)),
            None => "linked to no symbol".into(),
        };
        Err(crate::diagnostics::report!(
            message: (alloc::format!(
                "symbol '{printed}' is not linked to the symbol its path names"
            )),
            labels: [
                primary(
                    self.span(),
                    alloc::format!(
                        "this path names {} from '{}', but the reference is {linked}",
                        symbol_description(named),
                        self.name()
                    )
                ),
            ],
        ))
    }

    /// Check every attribute this operation holds, at any depth, in one walk
    /// ([Self::walk_attributes]): the attribute's own type ([Attribute::ty](crate::Attribute::ty))
    /// may be used in IR of this context, and the attribute's kind accepts it
    /// ([Attribute::verify](crate::Attribute::verify)).
    ///
    /// The own type is checked here, not only by [Self::verify_held_types], because a kind that
    /// takes its type from its value (`BoolAttr`'s `i1`) does not list it among the types it holds.
    ///
    /// Each symbol reference is also checked as [Self::verify_symbol_refs] checks it, after its
    /// kind has accepted it, so that one walk does both.
    fn verify_held_attributes(&self) -> Result<(), Report> {
        let context = self.context();
        let invalid = |role: &ElementRole<'_>, problem: &dyn fmt::Display| {
            crate::diagnostics::report!(
                message: "invalid operation",
                labels: [
                    primary(self.span(), alloc::format!("{role} of '{}' {problem}", self.name())),
                ],
            )
        };
        self.walk_attributes_and_depth(&mut |site, attr, nested| {
            let role = ElementRole { site, nested };
            let reference = attr.try_downcast_attr::<SymbolRefAttr>().ok();
            let attr = attr.borrow();
            let ty = attr.ty();
            if let Some(problem) = context.ir_type_problem(ty) {
                return WalkResult::Break(invalid(
                    &role,
                    &format_args!("has type `{ty}`, {problem}"),
                ));
            }
            if let Err(err) = attr.verify(context) {
                return WalkResult::Break(invalid(&role, &format_args!("is {err}")));
            }
            drop(attr);
            match reference {
                Some(reference) => self.verify_symbol_ref(reference).into(),
                None => WalkResult::Continue(()),
            }
        })
        .into_result()
    }

    /// Call [Dialect::verify_operation_attribute](crate::Dialect::verify_operation_attribute) of
    /// the dialect each dictionary entry of this operation is named for, once per entry: an entry
    /// whose key is `<prefix>.<rest>`, where `<prefix>` is a dialect registered in this operation's
    /// context. Properties are not dictionary entries and are not checked here.
    fn verify_dialect_attributes(&self) -> Result<(), Report> {
        let context = self.context();
        for entry in self.attributes() {
            let entry = entry.as_named_attribute();
            let Some((prefix, _)) = entry.name.as_str().split_once('.') else {
                continue;
            };
            // The registry is not borrowed while the hook runs: the dialect is cloned out of it
            let Some(dialect) = context.find_registered_dialect(prefix) else {
                continue;
            };
            dialect.verify_operation_attribute(self, entry.name, &entry.value)?;
        }
        Ok(())
    }

    /// Check every type this operation holds, at any depth, in one walk ([Self::walk_types]): it
    /// was interned by this context, it is not open, and its kind's
    /// [TypeInvariants](crate::TypeInvariants), if it has any, accept it.
    ///
    /// A type may only be used with the context that interned it, and an open type (one holding a
    /// recursion variable) may not be used in IR at all. Debug builds check both where a type is
    /// stored; this check is what catches them in every build, wherever the type is held: as the
    /// type of a result or a block argument, as an attribute's type, or inside an attribute's value
    /// (the type of a type attribute, a parameter of a signature, ..).
    ///
    /// The context remembers the types found valid ([Context::is_verified_type]), each with
    /// everything it holds, so a type held by many operations is checked once.
    fn verify_held_types(&self) -> Result<(), Report> {
        let context = self.context();
        // A type found valid before, with everything it holds, is neither checked nor walked into
        // again. The types checked here are remembered only if all of them are valid: then each
        // was checked with everything it holds (a type the walk skipped is valid already).
        let mut checked = SmallVec::<[crate::Type; 8]>::new();
        let result = self.walk_types(&mut |site, ty| {
            if context.is_verified_type(ty) {
                return WalkResult::Skip;
            }
            checked.push(ty);
            let problem = match context.ir_type_problem(ty) {
                Some(problem) => alloc::borrow::Cow::Borrowed(problem),
                None => match ty.as_trait::<dyn crate::TypeInvariants>() {
                    Some(invariants) => match invariants.check_invariants(context) {
                        Ok(()) => return WalkResult::Continue(()),
                        Err(err) => alloc::format!("which is invalid: {err}").into(),
                    },
                    None => return WalkResult::Continue(()),
                },
            };
            // a result's or block argument's own type keeps the wording of its role
            let (role, held_type) = match site {
                crate::ElementSite::Result(index) => {
                    (alloc::format!("result {index}"), self.results()[index].borrow().ty() != ty)
                }
                crate::ElementSite::BlockArgument {
                    region,
                    block,
                    index,
                } => {
                    let region = self.regions().iter().nth(region).expect("the walk's region");
                    let block = region.body().iter().nth(block).expect("the walk's block");
                    let arg_ty = block.arguments()[index].borrow().ty();
                    (alloc::format!("block argument {index}"), arg_ty != ty)
                }
                crate::ElementSite::Property(_) | crate::ElementSite::Attribute(_) => {
                    (alloc::format!("{site}"), true)
                }
            };
            let verb = if held_type { "holds" } else { "has" };
            WalkResult::Break(crate::diagnostics::report!(
                message: "invalid operation",
                labels: [
                    primary(
                        self.span(),
                        alloc::format!("{role} of '{}' {verb} type `{ty}`, {problem}", self.name())
                    ),
                ],
            ))
        })
        .into_result();
        if result.is_ok() {
            context.remember_verified_types(checked);
        }
        result
    }

    /// Run any verifiers for this operation, and all of its nested operations, recursively, then
    /// check that every value under it is defined before it is read.
    ///
    /// The first phase visits the operations in post-order, so that when the verifier(s) for an
    /// operation run, it is known that all of its children have successfully verified. Each
    /// operation is checked by [Self::verify], then by the structural rules of its operands, its
    /// block and its regions: no operand of an operation is one of its own results, in a region of
    /// any kind; an operation with successors is the last of its block and branches only to blocks
    /// of its own region; a terminator is the last operation of its block; unless the operation
    /// has `NoTerminator`, every block of its regions ends with a terminator; the entry block of a
    /// region has no predecessors; a graph region has at most one block.
    ///
    /// Only if all of that holds does the second phase walk the regions of `self`, because a
    /// malformed block structure would make dominance meaningless: no operand of an operation
    /// under an operation isolated from above is defined outside it, and the definition of every
    /// operand that can be reached dominates its use. A use in a block that cannot be reached from
    /// its region's entry is not checked, nor one that, walked out through the operations that
    /// hold it to the region of its definition, sits in a block that cannot be reached; and in a
    /// graph region every definition dominates every use. The operands of `self` itself are not
    /// checked.
    ///
    /// The first failure is returned.
    pub fn recursively_verify(&self) -> Result<(), Report> {
        self.postwalk(|op: &Operation| {
            op.verify()
                .and_then(|()| crate::verifier::dominance::verify_structure(op))
                .into()
        })
        .into_result()?;
        crate::verifier::dominance::verify_dominance(self)
    }
}

/// What [Operation::is_memory_effect_free] concludes from one operation, before looking at the
/// operations nested in it.
enum OwnMemoryEffects {
    /// The operation has memory effects, or its effects are unknown
    Some,
    /// The operation is memory effect free, whatever it holds
    None,
    /// The operation has no memory effects of its own, and has those of the operations it holds
    NoneOfItsOwn,
}

/// How a check names an attribute it found by [Operation::walk_attributes]: by its property or
/// dictionary entry, or as an attribute held by one.
struct ElementRole<'a> {
    site: crate::ElementSite<'a>,
    nested: bool,
}

impl fmt::Display for ElementRole<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nested {
            write!(f, "an attribute held by {}", self.site)
        } else {
            write!(f, "{}", self.site)
        }
    }
}

/// Traits/Casts
impl Operation {
    pub(super) const fn container(&self) -> *const () {
        unsafe {
            let ptr = self as *const Self;
            ptr.byte_sub(self.offset).cast()
        }
    }

    #[inline(always)]
    pub fn as_operation_ref(&self) -> OperationRef {
        // SAFETY: This is safe under the assumption that we always allocate Operations using the
        // arena, i.e. it is a child of a RawEntityMetadata structure.
        //
        // Additionally, this relies on the fact that Op implementations are #[repr(C)] and ensure
        // that their Operation field is always first in the generated struct
        unsafe { OperationRef::from_raw(self.container().cast()) }
    }

    /// Returns true if the concrete type of this operation is `T`
    #[inline]
    pub fn is<T: 'static>(&self) -> bool {
        self.name.is::<T>()
    }

    /// Returns true if this operation implements `Trait`
    #[inline]
    pub fn implements<Trait>(&self) -> bool
    where
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        self.name.implements::<Trait>()
    }

    /// Attempt to downcast to the concrete [Op] type of this operation
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.name.downcast_ref::<T>(self.container())
    }

    /// Attempt to downcast to the concrete [Op] type of this operation
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.name.downcast_mut::<T>(self.container().cast_mut())
    }

    /// Attempt to cast this operation reference to an implementation of `Trait`
    pub fn as_trait<Trait>(&self) -> Option<&Trait>
    where
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        self.name.upcast(self.container())
    }

    /// Attempt to cast this operation reference to an implementation of `Trait`
    pub fn as_trait_mut<Trait>(&mut self) -> Option<&mut Trait>
    where
        Trait: ?Sized + Pointee<Metadata = DynMetadata<Trait>> + 'static,
    {
        self.name.upcast_mut(self.container().cast_mut())
    }
}

/// Properties
impl Operation {
    #[inline]
    pub fn has_properties(&self) -> bool {
        !self.name.properties().is_empty()
    }

    /// Iterate over the properties (i.e. intrinsic attributes) of this operation
    pub fn properties(&self) -> impl Iterator<Item = NamedAttribute> {
        struct PropertyIter<'a> {
            op: &'a Operation,
            infos: &'a [AttrInfo],
            index: usize,
        }

        impl Iterator for PropertyIter<'_> {
            type Item = NamedAttribute;

            fn next(&mut self) -> Option<Self::Item> {
                let index = self.index;
                if let Some(info) = self.infos.get(index) {
                    self.index += 1;
                    let value = unsafe { (info.get_raw)(self.op.container(), info) };
                    Some(NamedAttribute {
                        name: info.name,
                        value,
                    })
                } else {
                    None
                }
            }
        }

        let infos = self.name.properties();
        PropertyIter {
            op: self,
            infos,
            index: 0,
        }
    }

    /// Iterate over the properties of this operation that are not at their default: a property
    /// without a default is always yielded, one with a default ([AttrInfo::default]) only when
    /// its value differs from it.
    ///
    /// This is what the derived text format prints.
    pub fn non_default_properties(&self) -> impl Iterator<Item = NamedAttribute> {
        let context = self.context_rc();
        self.name.properties().iter().filter_map(move |info| {
            let value = unsafe { (info.get_raw)(self.container(), info) };
            if info.is_default(&context, &value) {
                None
            } else {
                Some(NamedAttribute {
                    name: info.name,
                    value,
                })
            }
        })
    }

    /// Returns true if this operation has property `name`
    #[inline]
    pub fn has_property(&self, name: impl Into<interner::Symbol>) -> bool {
        self.name.has_property(name.into())
    }

    /// Get the value of intrinsic attribute (i.e. property) `name`.
    #[track_caller]
    pub fn get_property<'a>(
        &'a self,
        name: impl Into<interner::Symbol>,
    ) -> EntityRef<'a, dyn Attribute> {
        let name = name.into();
        let Some(prop) = self.name.get_property(name) else {
            panic!("{name} is not a valid property for '{}'", self.name);
        };
        unsafe { (prop.get)(self.container(), prop, core::marker::PhantomData) }
    }

    /// Get the value of intrinsic attribute (i.e. property) `name`, mutably.
    #[track_caller]
    pub fn get_property_mut<'a>(
        &'a mut self,
        name: impl Into<interner::Symbol>,
    ) -> EntityMut<'a, dyn Attribute> {
        let name = name.into();
        let Some(prop) = self.name.get_property(name) else {
            panic!("{name} is not a valid property for '{}'", self.name);
        };
        unsafe { (prop.get_mut)(self.container().cast_mut(), prop, core::marker::PhantomData) }
    }

    /// Set the intrinsic attribute (i.e. property) `name` with `value`.
    ///
    /// Every symbol reference the old value holds, at any depth, whose use belongs to this
    /// operation and that the operation no longer holds (by handle, in any property or entry,
    /// `value` included) is unlinked from its symbol, as erasing the operation would. The
    /// property must hold a value already, as
    /// every property of a built operation does; the builders set the first value without reading
    /// the field.
    ///
    /// This function will panic if `name` is not a valid property of this operation.
    #[track_caller]
    pub fn set_property(
        &mut self,
        name: impl Into<interner::Symbol>,
        value: AttributeRef,
    ) -> Result<(), Report> {
        let name = name.into();
        let Some(prop) = self.name.get_property(name) else {
            panic!("{name} is not a valid property for '{}'", self.name);
        };
        // SAFETY: the property of a built operation holds a valid attribute
        let old = unsafe { (prop.get_raw)(self.container(), prop) };
        unsafe { (prop.try_from)(self.container().cast_mut(), prop, value) }?;
        self.unlink_released_symbol_refs(&old);
        Ok(())
    }

    /// Set the property `name` of an operation being built to its first value: the field holds
    /// no attribute yet, so nothing is read from it and nothing is unlinked.
    #[track_caller]
    pub(crate) fn init_property(
        &mut self,
        name: impl Into<interner::Symbol>,
        value: AttributeRef,
    ) -> Result<(), Report> {
        let name = name.into();
        let Some(prop) = self.name.get_property(name) else {
            panic!("{name} is not a valid property for '{}'", self.name);
        };
        unsafe { (prop.try_from)(self.container().cast_mut(), prop, value) }
    }
}

/// Attributes
impl Operation {
    #[inline]
    pub fn has_attributes(&self) -> bool {
        !self.attrs.is_empty()
    }

    /// Get the underlying attribute set for this operation
    #[inline(always)]
    pub fn attributes(&self) -> &EntityMap<OpAttribute> {
        &self.attrs
    }

    /// Get a mutable reference to the underlying attribute set for this operation
    #[inline(always)]
    pub fn attributes_mut(&mut self) -> &mut EntityMap<OpAttribute> {
        &mut self.attrs
    }

    /// Return the value associated with attribute `name` for this function
    pub fn get_attribute(&self, name: impl Into<interner::Symbol>) -> Option<AttributeRef> {
        let name = name.into();
        if let Some(prop) = self.name.get_property(name) {
            Some(unsafe { (prop.get_raw)(self.container(), prop) })
        } else {
            let cursor = self.attrs.find(&name);
            if let Some(found) = cursor.get() {
                Some(found.as_named_attribute().value)
            } else {
                None
            }
        }
    }

    /// Return the value associated with attribute `name` for this function, as its concrete type
    /// `T`, _if_ the attribute by that name, is of that type.
    pub fn get_typed_attribute<T>(
        &self,
        name: impl Into<interner::Symbol>,
    ) -> Option<UnsafeIntrusiveEntityRef<T>>
    where
        T: AttributeRegistration,
    {
        self.get_attribute(name.into())
            .and_then(|attr_ref| attr_ref.try_downcast_attr::<T>().ok())
    }

    /// Return true if this function has an attributed named `name`
    pub fn has_attribute(&self, name: impl Into<interner::Symbol>) -> bool {
        self.get_attribute(name.into()).is_some()
    }

    /// Set the attribute `name` with `value` for this function.
    ///
    /// Every symbol reference the attribute it replaces holds, at any depth, whose use belongs to
    /// this operation and that the operation no longer holds (by handle, in any property or
    /// entry, `value` included) is unlinked from its symbol, as erasing the operation would.
    pub fn set_attribute(&mut self, name: impl Into<interner::Symbol>, value: AttributeRef) {
        let name = name.into();
        let old = self.attrs.find(&name).get().map(|old| old.as_named_attribute().value);
        let attr = OpAttribute::from(NamedAttribute::new(name, value));
        let attr = self.context().alloc_map_item(attr);
        if old.is_none() {
            self.attrs.insert(attr);
        } else {
            let mut existing = self.attrs.find_mut(&name);
            existing.replace_with(attr).expect("attribute cursor unexpectedly became null");
        }
        if let Some(old) = old {
            self.unlink_released_symbol_refs(&old);
        }
    }

    /// Remove any attribute with the given name from this function
    ///
    /// Every symbol reference the removed attribute holds, at any depth, whose use belongs to this
    /// operation and that the operation no longer holds (by handle, in any property or entry) is
    /// unlinked from its symbol, as erasing the operation would.
    pub fn remove_attribute(&mut self, name: impl Into<interner::Symbol>) {
        let name = name.into();
        let old = self.attrs.find(&name).get().map(|old| old.as_named_attribute().value);
        self.attrs.find_mut(&name).remove();
        if let Some(old) = old {
            self.unlink_released_symbol_refs(&old);
        }
    }

    /// Unlinks the symbol references `old` holds, at any depth, whose use belongs to this
    /// operation and names that reference, except those the operation still holds (by handle, at
    /// any depth of any property or dictionary entry): `old` has just been replaced or removed.
    ///
    /// A use that belongs to another operation (`old` is shared) or names another attribute is
    /// left alone. When `old` holds no such reference, the operation is not walked.
    fn unlink_released_symbol_refs(&mut self, old: &AttributeRef) {
        use crate::walk_nested_attributes;

        let owner = self.as_operation_ref();
        let mut released = SmallVec::<[_; 2]>::new();
        let _ = walk_nested_attributes(old, &mut |attr| {
            if let Ok(reference) = attr.try_downcast_attr::<SymbolRefAttr>()
                && let Some(user) = reference.borrow().try_user()
            {
                let symbol_use = user.borrow();
                if symbol_use.owner == owner && symbol_use.attr == reference {
                    released.push((reference, user));
                }
            }
            WalkResult::Continue(())
        });
        if released.is_empty() {
            return;
        }
        let _ = self.walk_attributes(&mut |_, attr| {
            if !released.is_empty()
                && let Ok(reference) = attr.try_downcast_attr::<SymbolRefAttr>()
            {
                released.retain(|(released, _)| *released != reference);
            }
            WalkResult::Continue(())
        });
        for (_, symbol_use) in released {
            self.unlink_symbol_use(symbol_use);
        }
    }
}

/// Symbol Attributes
impl Operation {
    pub fn set_symbol_attribute(
        &mut self,
        attr_name: impl Into<interner::Symbol>,
        symbol: impl AsSymbolRef,
    ) {
        let attr_name = attr_name.into();
        let symbol = symbol.as_symbol_ref();

        // Do not allow self-references
        //
        // NOTE: We are using this somewhat convoluted way to check identity of the symbol,
        // so that we do not attempt to borrow `self` again if `symbol` and `self` are the
        // same operation. That would fail due to the mutable reference to `self` we are
        // already holding.
        let (data_ptr, _) = SymbolRef::as_ptr(&symbol).to_raw_parts();
        assert!(
            !core::ptr::addr_eq(data_ptr, self.container()),
            "a symbol cannot use itself, except via nested operations"
        );

        // Store the underlying attribute value
        let owner = self.as_operation_ref();
        let context = self.context_rc();
        if self.has_property(attr_name) {
            let mut prop = self.get_property_mut(attr_name);
            let prop = prop.downcast_mut::<SymbolRefAttr>().unwrap();
            prop.set_symbol(symbol);
        } else if let Some(mut attr) = self
            .get_attribute(attr_name)
            .and_then(|attr| attr.try_downcast_attr::<SymbolRefAttr>().ok())
        {
            attr.borrow_mut().set_symbol(symbol);
        } else {
            let path = symbol.borrow().path();
            let symbol_ref = crate::dialects::builtin::attributes::SymbolRef::new(path, None);
            let mut attr = context.create_attribute::<SymbolRefAttr, _>(symbol_ref);
            let user = context.alloc_tracked(SymbolUse::new(owner, attr));
            attr.borrow_mut().set_user(user);
            self.set_attribute(attr_name, attr.as_attribute_ref());
            attr.borrow_mut().link(symbol);
        }
    }

    /// Whether `symbol` is this operation, compared by address, without borrowing `symbol`.
    fn is_self(&self, symbol: &SymbolRef) -> bool {
        let (data_ptr, _) = SymbolRef::as_ptr(symbol).to_raw_parts();
        core::ptr::addr_eq(data_ptr, self.container())
    }

    /// Links `symbol_use` to `symbol`, which may be this operation (already borrowed mutably by
    /// the caller).
    pub(crate) fn link_symbol_use(&mut self, symbol_use: crate::SymbolUseRef, symbol: SymbolRef) {
        if self.is_self(&symbol) {
            let this = self
                .as_trait_mut::<dyn crate::Symbol>()
                .expect("an operation that is a symbol implements Symbol");
            // SAFETY: `symbol` is this operation, whose use list this is
            unsafe { symbol_use.link_into(symbol, this.uses_mut()) };
        } else {
            symbol_use.link_to_symbol(symbol);
        }
    }

    /// Unlinks `symbol_use` from the symbol it is linked to, if any, which may be this operation
    /// (already borrowed mutably by the caller).
    fn unlink_symbol_use(&mut self, symbol_use: crate::SymbolUseRef) {
        let Some(symbol) = symbol_use.linked_symbol() else {
            return;
        };
        if self.is_self(&symbol) {
            let this = self
                .as_trait_mut::<dyn crate::Symbol>()
                .expect("an operation that is a symbol implements Symbol");
            // SAFETY: the use is linked to this operation, so it is in its use list
            unsafe { symbol_use.unlink_from(this.uses_mut()) };
        } else {
            symbol_use.unlink_from_symbol();
        }
    }

    /// This function performs the same change as [`Self::set_symbol_attribute`], but is called
    /// specifically when creating an operation and setting a symbol property of that operation.
    ///
    /// Because properties hold intially-dangling pointers to their attribute impls, we must not
    /// try to dereference those pointers until a valid attribute value has been written.
    fn unsafe_set_symbol_property(
        &mut self,
        attr_name: interner::Symbol,
        symbol: impl AsSymbolRef,
    ) {
        let symbol = symbol.as_symbol_ref();

        // Do not allow self-references
        //
        // NOTE: We are using this somewhat convoluted way to check identity of the symbol,
        // so that we do not attempt to borrow `self` again if `symbol` and `self` are the
        // same operation. That would fail due to the mutable reference to `self` we are
        // already holding.
        let (data_ptr, _) = SymbolRef::as_ptr(&symbol).to_raw_parts();
        assert!(
            !core::ptr::addr_eq(data_ptr, self.container()),
            "a symbol cannot use itself, except via nested operations"
        );

        // Track the usage of `symbol` by `self`
        let context = self.context_rc();
        let owner = self.as_operation_ref();
        let mut attr = {
            let symbol = symbol.borrow();
            let mut attr = context.create_attribute::<SymbolRefAttr, _>(
                crate::dialects::builtin::attributes::SymbolRef::new(symbol.path(), None),
            );

            let user = context.alloc_tracked(SymbolUse::new(owner, attr));
            attr.borrow_mut().set_user(user);
            attr
        };

        // Store the underlying attribute value
        self.init_property(attr_name, attr)
            .expect("not a valid value for this operation property");
        attr.borrow_mut().link(symbol);
    }
}

/// Navigation
impl Operation {
    /// Returns a handle to the containing [Block] of this operation, if it is attached to one
    #[inline]
    pub fn parent(&self) -> Option<BlockRef> {
        self.as_operation_ref().parent()
    }

    /// Returns a handle to the containing [Region] of this operation, if it is attached to one
    pub fn parent_region(&self) -> Option<RegionRef> {
        self.as_operation_ref().parent_region()
    }

    /// Returns a handle to the nearest containing [Operation] of this operation, if it is attached
    /// to one
    pub fn parent_op(&self) -> Option<OperationRef> {
        self.as_operation_ref().parent_op()
    }

    /// Returns a handle to the nearest containing [Operation] of type `T` for this operation, if it
    /// is attached to one
    pub fn nearest_parent_op<T: Op>(&self) -> Option<UnsafeIntrusiveEntityRef<T>> {
        let mut parent = self.parent_op();
        while let Some(op) = parent.take() {
            parent = op.parent_op();
            if let Ok(op) = op.try_downcast_op::<T>() {
                return Some(op);
            }
        }
        None
    }
}

/// Traversal
impl Operation {
    pub fn prewalk_all<F>(&self, callback: F)
    where
        F: FnMut(&Operation),
    {
        Walk::<Operation>::prewalk_all::<Forward, _>(self, callback);
    }

    pub fn prewalk<F, B>(&self, callback: F) -> WalkResult<B>
    where
        F: FnMut(&Operation) -> WalkResult<B>,
    {
        Walk::<Operation>::prewalk::<Forward, _, _>(self, callback)
    }

    pub fn postwalk_all<F>(&self, callback: F)
    where
        F: FnMut(&Operation),
    {
        Walk::<Operation>::postwalk_all::<Forward, _>(self, callback);
    }

    pub fn postwalk<F, B>(&self, callback: F) -> WalkResult<B>
    where
        F: FnMut(&Operation) -> WalkResult<B>,
    {
        Walk::<Operation>::postwalk::<Forward, _, _>(self, callback)
    }
}

/// Regions
impl Operation {
    /// Returns true if this operation has any regions
    #[inline]
    pub fn has_regions(&self) -> bool {
        !self.regions.is_empty()
    }

    /// Returns the number of regions owned by this operation.
    ///
    /// NOTE: This does not include regions of nested operations, just those directly attached
    /// to this operation.
    #[inline]
    pub fn num_regions(&self) -> usize {
        self.regions.len()
    }

    /// Get a reference to the region list for this operation
    #[inline(always)]
    pub fn regions(&self) -> &RegionList {
        &self.regions
    }

    /// Get a mutable reference to the region list for this operation
    #[inline(always)]
    pub fn regions_mut(&mut self) -> &mut RegionList {
        &mut self.regions
    }

    /// Get a reference to a specific region, given its index.
    ///
    /// This function will panic if the index is invalid.
    pub fn region(&self, index: usize) -> EntityRef<'_, Region> {
        let mut cursor = self.regions.front();
        let mut count = 0;
        while !cursor.is_null() {
            if index == count {
                return cursor.into_borrow().unwrap();
            }
            cursor.move_next();
            count += 1;
        }
        panic!("invalid region index {index}: out of bounds");
    }

    /// Get a mutable reference to a specific region, given its index.
    ///
    /// This function will panic if the index is invalid.
    pub fn region_mut(&mut self, index: usize) -> EntityMut<'_, Region> {
        let mut cursor = self.regions.front_mut();
        let mut count = 0;
        while !cursor.is_null() {
            if index == count {
                return cursor.into_borrow_mut().unwrap();
            }
            cursor.move_next();
            count += 1;
        }
        panic!("invalid region index {index}: out of bounds");
    }
}

/// Successors
impl Operation {
    /// Returns true if this operation has any successor blocks
    #[inline]
    pub fn has_successors(&self) -> bool {
        !self.successors.is_empty()
    }

    /// Returns the number of successor blocks this operation may transfer control to
    #[inline]
    pub fn num_successors(&self) -> usize {
        self.successors.len()
    }

    /// Get a reference to the successors of this operation
    #[inline(always)]
    pub fn successors(&self) -> &OpSuccessorStorage {
        &self.successors
    }

    /// Get a mutable reference to the successors of this operation
    #[inline(always)]
    pub fn successors_mut(&mut self) -> &mut OpSuccessorStorage {
        &mut self.successors
    }

    /// Get a reference to the successor group at `index`
    #[inline]
    pub fn successor_group(&self, index: usize) -> OpSuccessorRange<'_> {
        self.successors.group(index)
    }

    /// Get a mutable reference to the successor group at `index`
    #[inline]
    pub fn successor_group_mut(&mut self, index: usize) -> OpSuccessorRangeMut<'_> {
        self.successors.group_mut(index)
    }

    /// Get a reference to the keyed successor group at `index`
    #[inline]
    pub fn keyed_successor_group<T>(&self, index: usize) -> KeyedSuccessorRange<'_, T>
    where
        T: KeyedSuccessor,
    {
        let range = self.successors.group(index);
        KeyedSuccessorRange::new(range, &self.operands)
    }

    /// Get a mutable reference to the keyed successor group at `index`
    #[inline]
    pub fn keyed_successor_group_mut<T>(&mut self, index: usize) -> KeyedSuccessorRangeMut<'_, T>
    where
        T: KeyedSuccessor,
    {
        let range = self.successors.group_mut(index);
        KeyedSuccessorRangeMut::new(range, &mut self.operands)
    }

    /// Get a reference to the successor at `index` in the group at `group_index`
    #[inline]
    pub fn successor_in_group(&self, group_index: usize, index: usize) -> OpSuccessor<'_> {
        let info = &self.successors.group(group_index)[index];
        OpSuccessor {
            dest: info.block,
            arguments: self.operands.group(info.operand_group as usize),
        }
    }

    /// Get a mutable reference to the successor at `index` in the group at `group_index`
    #[inline]
    pub fn successor_in_group_mut(
        &mut self,
        group_index: usize,
        index: usize,
    ) -> OpSuccessorMut<'_> {
        let info = &self.successors.group(group_index)[index];
        OpSuccessorMut {
            dest: info.block,
            arguments: self.operands.group_mut(info.operand_group as usize),
        }
    }

    /// Get a reference to the successor at `index`
    #[inline]
    #[track_caller]
    pub fn successor(&self, index: usize) -> OpSuccessor<'_> {
        let info = &self.successors[index];
        OpSuccessor {
            dest: info.block,
            arguments: self.operands.group(info.operand_group as usize),
        }
    }

    /// Get a mutable reference to the successor at `index`
    #[inline]
    #[track_caller]
    pub fn successor_mut(&mut self, index: usize) -> OpSuccessorMut<'_> {
        let info = self.successors[index];
        OpSuccessorMut {
            dest: info.block,
            arguments: self.operands.group_mut(info.operand_group as usize),
        }
    }

    /// Get an iterator over the successors of this operation
    pub fn successor_iter(&self) -> impl DoubleEndedIterator<Item = OpSuccessor<'_>> + '_ {
        self.successors.iter().map(|info| OpSuccessor {
            dest: info.block,
            arguments: self.operands.group(info.operand_group as usize),
        })
    }
}

/// Operands
impl Operation {
    /// Returns true if this operation has at least one operand
    #[inline]
    pub fn has_operands(&self) -> bool {
        !self.operands.is_empty()
    }

    /// Returns the number of operands given to this operation
    #[inline]
    pub fn num_operands(&self) -> usize {
        self.operands.len()
    }

    /// Get a reference to the operand storage for this operation
    #[inline]
    pub fn operands(&self) -> &OpOperandStorage {
        &self.operands
    }

    /// Get a mutable reference to the operand storage for this operation
    #[inline]
    pub fn operands_mut(&mut self) -> &mut OpOperandStorage {
        &mut self.operands
    }

    /// Replace the current operands of this operation with the ones provided in `operands`.
    pub fn set_operands(&mut self, operands: impl IntoIterator<Item = ValueRef>) {
        self.operands.clear();
        let context = self.context_rc();
        let owner = self.as_operation_ref();
        self.operands.extend(
            operands
                .into_iter()
                .enumerate()
                .map(|(index, value)| context.make_operand(value, owner, index as u16)),
        );
    }

    /// Replace any uses of `from` with `to` within this operation
    pub fn replaces_uses_of_with(&mut self, from: ValueRef, to: ValueRef) {
        if ValueRef::ptr_eq(&from, &to) {
            return;
        }

        for operand in self.operands.iter_mut() {
            debug_assert!(operand.is_linked());
            if ValueRef::ptr_eq(&from, &operand.borrow().value.unwrap()) {
                operand.borrow_mut().set(to);
            }
        }
    }

    /// Replace all uses of this operation's results with `values`
    ///
    /// The number of results and the number of values in `values` must be exactly the same,
    /// otherwise this function will panic.
    pub fn replace_all_uses_with(&mut self, values: impl ExactSizeIterator<Item = ValueRef>) {
        assert_eq!(self.num_results(), values.len());
        for (result, replacement) in self.results.iter_mut().zip(values) {
            if (*result as ValueRef) == replacement {
                continue;
            }
            result.borrow_mut().replace_all_uses_with(replacement);
        }
    }

    /// Replace uses of this operation's results with `values`, for each use which, when provided
    /// to the given callback, returns true.
    ///
    /// The number of results and the number of values in `values` must be exactly the same,
    /// otherwise this function will panic.
    pub fn replace_uses_with_if<F, V>(&mut self, values: V, should_replace: F)
    where
        V: ExactSizeIterator<Item = ValueRef>,
        F: Fn(&OpOperandImpl) -> bool,
    {
        assert_eq!(self.num_results(), values.len());
        for (result, replacement) in self.results.iter_mut().zip(values) {
            let mut result = *result as ValueRef;
            if result == replacement {
                continue;
            }
            result.borrow_mut().replace_uses_with_if(replacement, &should_replace);
        }
    }
}

/// Results
impl Operation {
    /// Returns true if this operation produces any results
    #[inline]
    pub fn has_results(&self) -> bool {
        !self.results.is_empty()
    }

    /// Returns the number of results produced by this operation
    #[inline]
    pub fn num_results(&self) -> usize {
        self.results.len()
    }

    /// Get a reference to the result set of this operation
    #[inline]
    pub fn results(&self) -> &OpResultStorage {
        &self.results
    }

    /// Get a mutable reference to the result set of this operation
    #[inline]
    pub fn results_mut(&mut self) -> &mut OpResultStorage {
        &mut self.results
    }

    /// Get a reference to the result at `index` among all results of this operation
    #[inline]
    pub fn get_result(&self, index: usize) -> &OpResultRef {
        &self.results[index]
    }

    /// Returns true if the results of this operation are used
    pub fn is_used(&self) -> bool {
        self.results.iter().any(|result| result.borrow().is_used())
    }

    /// Returns true if the results of this operation have exactly one user
    pub fn has_exactly_one_use(&self) -> bool {
        let mut used_by = None;
        for result in self.results.iter() {
            let result = result.borrow();
            if !result.is_used() {
                continue;
            }

            for used in result.iter_uses() {
                if used_by.as_ref().is_some_and(|user| !OperationRef::eq(user, &used.owner)) {
                    // We found more than one user
                    return false;
                } else if used_by.is_none() {
                    used_by = Some(used.owner);
                }
            }
        }

        // If we reach here, and we have a `used_by` set, we have exactly one user
        used_by.is_some()
    }

    /// Returns true if the results of this operation are used outside of the given block
    pub fn is_used_outside_of_block(&self, block: &BlockRef) -> bool {
        self.results
            .iter()
            .any(|result| result.borrow().is_used_outside_of_block(block))
    }

    /// Returns true if this operation is directly in a region of an operation that carries
    /// [RegionOperationsAreLive](crate::traits::RegionOperationsAreLive): the region describes
    /// what its parent stands for, so the parent uses this operation, and no pass erases, moves,
    /// merges or folds it.
    pub fn is_in_live_region(&self) -> bool {
        self.parent_op().is_some_and(|parent| {
            parent.borrow().implements::<dyn crate::traits::RegionOperationsAreLive>()
        })
    }

    /// Returns true if this operation is unused and has no side effects that prevent it being erased
    pub fn is_trivially_dead(&self) -> bool {
        !self.is_used() && self.would_be_trivially_dead()
    }

    /// Whether erasing this operation could take away more than what it computes: control that
    /// leaves it by a nested terminator, or a loop that may never end.
    ///
    /// A nested terminator, at any depth, counts unless it is known to keep control inside this
    /// operation:
    ///
    /// * an early exit ([RegionExitTerminatorOpInterface]) whose target is this operation or one
    ///   nested in it;
    /// * any other [RegionBranchTerminatorOpInterface], which goes to a sibling region or back to
    ///   its own parent.
    ///
    /// Every other terminator counts: a return (which leaves the enclosing function from any
    /// depth), an early exit that addresses an ancestor of this operation or does not resolve, a
    /// terminator with block successors (it may close a cycle) and one whose meaning is unknown.
    ///
    /// A loop counts, whether it is this operation or one nested in it
    /// ([RegionBranchOpInterface::has_loop], [LoopLikeOpInterface]): nothing here shows that it
    /// ends, and whether a program ends can be observed.
    ///
    /// [RegionExitTerminatorOpInterface]: crate::RegionExitTerminatorOpInterface
    /// [RegionBranchTerminatorOpInterface]: crate::RegionBranchTerminatorOpInterface
    /// [RegionBranchOpInterface::has_loop]: crate::RegionBranchOpInterface::has_loop
    /// [LoopLikeOpInterface]: crate::LoopLikeOpInterface
    pub fn has_nested_exit_or_loop(&self) -> bool {
        use crate::{
            LoopLikeOpInterface, RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
            RegionExitTerminatorOpInterface,
        };

        fn is_loop(op: &Operation) -> bool {
            op.implements::<dyn LoopLikeOpInterface>()
                || op.as_trait::<dyn RegionBranchOpInterface>().is_some_and(|op| op.has_loop())
        }

        if !self.has_regions() {
            return false;
        }
        if is_loop(self) {
            return true;
        }
        let this = self.as_operation_ref();
        for region in self.regions() {
            let walk = region.prewalk(|op| {
                if is_loop(op) {
                    return WalkResult::Break(());
                }
                if !op.implements::<dyn crate::traits::Terminator>() {
                    return WalkResult::Continue(());
                }
                let stays_inside = match op.as_trait::<dyn RegionExitTerminatorOpInterface>() {
                    Some(exit) => exit.get_exit_target().is_some_and(|target| {
                        target == this || self.is_proper_ancestor_of(&target.borrow())
                    }),
                    None => {
                        op.implements::<dyn RegionBranchTerminatorOpInterface>()
                            && !op.has_successors()
                    }
                };
                if stays_inside {
                    WalkResult::Continue(())
                } else {
                    WalkResult::Break(())
                }
            });
            if walk.was_interrupted() {
                return true;
            }
        }

        false
    }

    /// Returns true if this operation would be dead if unused, and has no side effects that would
    /// prevent erasing it. This is equivalent to checking `is_trivially_dead` if `self` is unused.
    ///
    /// NOTE: Terminators and symbols are never considered to be trivially dead by this function.
    pub fn would_be_trivially_dead(&self) -> bool {
        // An operation in the region of one that stands for a description (a global's
        // initializer) is used by that parent, whatever the uses of its results.
        if self.is_in_live_region() {
            return false;
        }
        if self.implements::<dyn crate::traits::Terminator>() || self.implements::<dyn Symbol>() {
            false
        } else {
            self.would_be_trivially_dead_even_if_terminator()
        }
    }

    /// Implementation of `would_be_trivially_dead` that also considers terminator operations as
    /// dead if they have no side effects. This allows for marking region operations as trivially
    /// dead without always being conservative about terminators.
    pub fn would_be_trivially_dead_even_if_terminator(&self) -> bool {
        use crate::effects::RecursiveEffectIterator;

        if self.is_in_live_region() {
            return false;
        }

        // Control that leaves this operation, and a loop that may not end, are not memory effects:
        // the effects below do not show them.
        if self.has_nested_exit_or_loop() {
            return false;
        }

        let this_op = self.as_operation_ref();
        let effects = RecursiveEffectIterator::<MemoryEffect>::new(this_op);
        for (effecting_op, effect) in effects {
            // The presence of an unknown effect, then we must treat this op as conservatively
            // having effects
            let Some(effect) = effect else {
                return false;
            };

            // We can drop effects if:
            //
            // * the effect is an allocation and the effect is a result of the effecting op.
            // * the effect is a read
            match effect.effect() {
                MemoryEffect::Read => (),
                MemoryEffect::Allocate => match effect.value() {
                    Some(value) => {
                        let is_defined_by_op =
                            value.borrow().get_defining_op().is_some_and(|op| op == effecting_op);
                        let is_droppable =
                            matches!(effect.effect(), MemoryEffect::Allocate) && is_defined_by_op;
                        if !is_droppable {
                            return false;
                        }
                    }
                    None => return false,
                },
                _ => return false,
            }
        }

        // If we get here, none of the operations had effects that prevented marking this operation
        // as dead.
        true
    }

    /// Returns true if this operation, when executed, leaves the callable that encloses it.
    ///
    /// That is the case for an operation that implements [TailCallLike], and for one that
    /// implements [ReturnLike] and is not a terminator of region control flow, i.e. implements
    /// neither [RegionBranchTerminatorOpInterface] nor [RegionExitTerminatorOpInterface]: an
    /// `scf.yield` or `scf.condition` hands values to its parent operation, and an `scf.break`
    /// leaves a loop, but none of them leaves the function.
    ///
    /// [TailCallLike]: crate::traits::TailCallLike
    /// A callable's verifier requires every operation between such an operation and the callable to
    /// implement [PropagateControlFlowBreak](crate::PropagateControlFlowBreak) (see
    /// [verify_exit_crosses_only_forwarding_ops](crate::verify_exit_crosses_only_forwarding_ops)).
    ///
    /// [ReturnLike]: crate::traits::ReturnLike
    /// [RegionBranchTerminatorOpInterface]: crate::RegionBranchTerminatorOpInterface
    /// [RegionExitTerminatorOpInterface]: crate::RegionExitTerminatorOpInterface
    pub fn leaves_callable(&self) -> bool {
        self.implements::<dyn crate::traits::TailCallLike>()
            || (self.implements::<dyn crate::traits::ReturnLike>()
                && !self.implements::<dyn RegionBranchTerminatorOpInterface>()
                && !self.implements::<dyn RegionExitTerminatorOpInterface>())
    }

    /// Returns true if the given operation is free of memory effects.
    ///
    /// An operation is free of memory effects if its implementation of `MemoryEffectOpInterface`
    /// indicates that it has no memory effects. For example, it may implement `NoMemoryEffect`.
    /// Alternatively, if the operation has the `HasRecursiveMemoryEffects` trait, then it is free
    /// of memory effects if all of its nested operations are free of memory effects.
    ///
    /// If the operation has both, then it is free of memory effects if both conditions are
    /// satisfied. An operation whose implementation says its effects are unknown
    /// ([EffectOpInterface::effects_are_known](crate::effects::EffectOpInterface::effects_are_known))
    /// is not free of memory effects.
    pub fn is_memory_effect_free(&self) -> bool {
        match self.own_memory_effects() {
            OwnMemoryEffects::Some => return false,
            OwnMemoryEffects::None => return true,
            OwnMemoryEffects::NoneOfItsOwn => (),
        }

        // Every nested operation, at any depth, is memory effect free by its own effects: then
        // each is memory effect free with what it holds too. The walk visits each nested
        // operation once (asking each the whole question instead would walk each region once
        // per enclosing operation, which doubles the work with each level of nesting).
        for region in self.regions() {
            let walk_result = region.prewalk(|op| match op.own_memory_effects() {
                OwnMemoryEffects::Some => WalkResult::Break(()),
                OwnMemoryEffects::None | OwnMemoryEffects::NoneOfItsOwn => WalkResult::Continue(()),
            });
            if walk_result.was_interrupted() {
                return false;
            }
        }

        true
    }

    /// What [Self::is_memory_effect_free] concludes from this operation alone, before looking at
    /// the operations nested in it.
    fn own_memory_effects(&self) -> OwnMemoryEffects {
        let recursive = self.implements::<dyn HasRecursiveMemoryEffects>();
        if let Some(mem_interface) = self.as_trait::<dyn MemoryEffectOpInterface>() {
            // `has_no_effect` is false when the effects are unknown
            if !mem_interface.has_no_effect() {
                return OwnMemoryEffects::Some;
            }
            // If the op does not have recursive side effects, then it is memory effect free
            if !recursive {
                return OwnMemoryEffects::None;
            }
        } else if !recursive {
            // Otherwise, if the op does not implement the memory effect interface and it does not
            // have recursive side effects, then it cannot be known that the op is moveable.
            return OwnMemoryEffects::Some;
        }
        OwnMemoryEffects::NoneOfItsOwn
    }

    pub fn get_effects_recursively<T: Effect>(&self) -> Option<SmallVec<[EffectInstance<T>; 2]>> {
        use crate::effects::RecursiveEffectIterator;

        let mut effects = SmallVec::<[_; 2]>::default();

        for (_, effect) in RecursiveEffectIterator::new(self.as_operation_ref()) {
            effects.push(effect?);
        }

        Some(effects)
    }

    /// Returns true if this operation is known to have `expected_effect`.
    ///
    /// Returns false if the op has unknown effects, no effects, or does not have `expected_effect`
    /// amongst its effects.
    ///
    /// Only the operation's own effects are consulted, not those of the operations in its
    /// regions: for an operation with `HasRecursiveMemoryEffects`, use
    /// [Self::get_effects_recursively].
    pub fn has_memory_effect(&self, expected_effect: MemoryEffect) -> bool {
        let Some(mem_interface) = self.as_trait::<dyn MemoryEffectOpInterface>() else {
            return false;
        };
        if !mem_interface.effects_are_known() {
            return false;
        }

        mem_interface.effects().any(|effect| effect.effect() == expected_effect)
    }

    /// Returns true if this operation has `expected_effect` and _only_ `expected_effect`.
    ///
    /// Returns false if the op has unknown effects, no effects, or has any effect other than
    /// `expected_effect`.
    ///
    /// Only the operation's own effects are consulted, not those of the operations in its
    /// regions: for an operation with `HasRecursiveMemoryEffects`, use
    /// [Self::get_effects_recursively].
    pub fn has_single_memory_effect(&self, expected_effect: MemoryEffect) -> bool {
        let Some(mem_interface) = self.as_trait::<dyn MemoryEffectOpInterface>() else {
            return false;
        };
        if !mem_interface.effects_are_known() || mem_interface.has_no_effect() {
            return false;
        }
        for effect in mem_interface.effects() {
            if effect.effect() != expected_effect {
                return false;
            }
        }
        true
    }
}

/// Movement
impl Operation {
    /// Remove this operation (and its descendants) from its containing block, and delete them
    ///
    /// The symbol references of this operation and of every operation nested in it are unlinked
    /// from the symbols they name, so an erased operation is no longer a use of any symbol. If this
    /// operation, or an operation nested in it, is a symbol, the references to it from outside the
    /// erased operation are unlinked too: they no longer resolve, and verification reports them.
    ///
    /// # Panics
    ///
    /// Erasing borrows other operations mutably, which panics if the caller holds a borrow of one
    /// of them: the symbol table the operation is registered in, when it is a symbol (it is
    /// removed from that table); every symbol nested in it (their use lists are emptied); and every
    /// symbol a reference held by it or by an operation nested in it is linked to (the use is
    /// unlinked from it). In particular, the caller must not hold a borrow of the enclosing symbol
    /// table when the operation is a symbol or contains one.
    pub fn erase(&mut self) {
        self.unlink_symbol_uses();

        // We don't delete entities currently, so for now this is just an alias for `remove`
        self.remove();

        self.successors.clear();
        self.operands.clear();
    }

    /// Unlinks every symbol reference held by this operation or an operation nested in it from
    /// the symbol it is linked to, and then empties the use list of this operation and of every
    /// operation nested in it that is a symbol, which leaves only the references from outside.
    ///
    /// The uses and symbols are collected first, so no operation is borrowed by the walk when a
    /// symbol's use list is changed. The references held inside are unlinked before the use lists
    /// are emptied, so each use is taken out of a list once. A use list of this operation itself
    /// (a recursive call inside a function being erased) is changed through `self`, which the
    /// caller already borrows mutably; every other symbol is borrowed mutably in turn, so the
    /// caller must not hold a borrow of one.
    fn unlink_symbol_uses(&mut self) {
        let mut uses = SmallVec::<[crate::SymbolUseRef; 4]>::default();
        let mut nested_symbols = SmallVec::<[SymbolRef; 4]>::default();
        // `self` is visited directly: the walkers borrow the operations they visit, and `self` is
        // already borrowed mutably
        let mut collect_uses = |op: &Operation| {
            let _ = Operation::walk_symbol_refs(op, |reference| {
                if let Ok(symbol_use) = reference {
                    uses.push(symbol_use);
                }
                WalkResult::Continue(())
            });
        };
        collect_uses(self);
        for region in self.regions() {
            region.prewalk_all(|op: &Operation| {
                collect_uses(op);
                if let Some(symbol) = op.as_symbol_ref() {
                    nested_symbols.push(symbol);
                }
            });
        }

        for symbol_use in uses {
            self.unlink_symbol_use(symbol_use);
        }

        // What remains in these use lists are the references from outside, as `SymbolTable::remove`
        // leaves them
        if let Some(symbol) = self.as_trait_mut::<dyn crate::Symbol>() {
            symbol.uses_mut().clear();
        }
        for mut symbol in nested_symbols {
            symbol.borrow_mut().uses_mut().clear();
        }
    }

    /// Remove the operation from its parent block, but don't delete it.
    ///
    /// A symbol is removed from the symbol table it was registered in (the nearest one above its
    /// block), which borrows that table mutably: when the operation is a symbol, the caller must
    /// not hold a borrow of the enclosing symbol table, or this panics. The symbols nested in the
    /// operation stay in its own symbol tables, and the references to it are left linked.
    pub fn remove(&mut self) {
        if let Some(mut parent) = self.parent() {
            let mut block = parent.borrow_mut();
            let body = block.body_mut();
            let mut cursor = unsafe { body.cursor_mut_from_ptr(self.as_operation_ref()) };
            cursor.remove();
        }
    }

    /// Unlink this operation from its current block and insert it at `ip`, which may be in the same
    /// or another block in the same function.
    ///
    /// # Panics
    ///
    /// This function will panic if the given program point is unset, or refers to an orphaned op,
    /// i.e. an op that has no parent block.
    ///
    /// It also panics if the operation is a symbol and the caller holds a borrow of the symbol
    /// table it leaves or of the one it enters: moving a symbol removes it from the first and
    /// registers it in the second, each borrowed mutably (see [Self::remove]).
    pub fn move_to(&mut self, mut ip: ProgramPoint) {
        let this = self.as_operation_ref();
        if let Some(op) = ip.operation() {
            if op == this {
                // The move is a no-op
                return;
            }

            assert!(ip.block().is_some(), "cannot insert an operation relative to an orphaned op");
        }

        // Detach `self`
        self.remove();

        {
            // Move `self` to `ip`
            let mut cursor = ip.cursor_mut().expect("insertion point is invalid/unset");
            // NOTE: We use `insert_after` here because the cursor we get is positioned such that
            // insert_after will always insert at the precise point specified.
            cursor.insert_after(self.as_operation_ref());
        }
    }

    /// This drops all operand uses from this operation, which is used to break cyclic dependencies
    /// between references when they are to be deleted
    pub fn drop_all_references(&mut self) {
        self.operands.clear();

        {
            let mut region_cursor = self.regions.front_mut();
            while let Some(mut region) = region_cursor.as_pointer() {
                region.borrow_mut().drop_all_references();
                region_cursor.move_next();
            }
        }

        self.successors.clear();
    }

    /// This drops all uses of any values defined by this operation or its nested regions,
    /// wherever they are located.
    pub fn drop_all_defined_value_uses(&mut self) {
        for result in self.results.iter_mut() {
            let mut res = result.borrow_mut();
            res.uses_mut().clear();
        }

        let mut regions = self.regions.front_mut();
        while let Some(mut region) = regions.as_pointer() {
            let mut region = region.borrow_mut();
            let blocks = region.body_mut();
            let mut cursor = blocks.front_mut();
            while let Some(mut block) = cursor.as_pointer() {
                block.borrow_mut().drop_all_defined_value_uses();
                cursor.move_next();
            }
            regions.move_next();
        }
    }

    /// Drop all uses of results of this operation
    pub fn drop_all_uses(&mut self) {
        for result in self.results.iter_mut() {
            result.borrow_mut().uses_mut().clear();
        }
    }
}

/// Ordering
impl Operation {
    /// This value represents an invalid index ordering for an operation within its containing block
    const INVALID_ORDER: u32 = u32::MAX;
    /// This value represents the stride to use when computing a new order for an operation
    const ORDER_STRIDE: u32 = 5;

    /// Returns true if this operation is an ancestor of `other`.
    ///
    /// An operation is considered its own ancestor, use [Self::is_proper_ancestor_of] if you do not
    /// want this behavior.
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        core::ptr::addr_eq(self, other) || Self::is_a_proper_ancestor_of_b(self, other)
    }

    /// Returns true if this operation is a proper ancestor of `other`
    /// Find the region of `ancestor` containing this operation.
    pub fn region_in_ancestor(&self, ancestor: &OperationRef) -> Option<RegionRef> {
        let mut region = self.parent_region();
        while let Some(current) = region {
            let parent = current.borrow().parent()?;
            if OperationRef::ptr_eq(&parent, ancestor) {
                return Some(current);
            }
            region = parent.borrow().parent_region();
        }
        None
    }

    pub fn is_proper_ancestor_of(&self, other: &Self) -> bool {
        Self::is_a_proper_ancestor_of_b(self, other)
    }

    /// Returns true if operation `a` is a proper ancestor of operation `b`
    fn is_a_proper_ancestor_of_b(a: &Self, b: &Self) -> bool {
        let a = a.as_operation_ref();
        let mut next = b.parent_op();
        while let Some(b) = next.take() {
            if OperationRef::ptr_eq(&a, &b) {
                return true;
            }
            next = b.parent_op();
        }
        false
    }

    /// Given an operation `other` that is within the same parent block, return whether the current
    /// operation is before it in the operation list.
    ///
    /// NOTE: This function has an average complexity of O(1), but worst case may take O(N) where
    /// N is the number of operations within the parent block.
    pub fn is_before_in_block(&self, other: &OperationRef) -> bool {
        use core::sync::atomic::Ordering;

        let block = self.parent().expect("operations without parent blocks have no order");
        let other = other.borrow();
        assert!(
            other
                .parent()
                .as_ref()
                .is_some_and(|other_block| BlockRef::ptr_eq(&block, other_block)),
            "expected both operations to have the same parent block"
        );

        // If the order of the block is already invalid, directly recompute the parent
        if !block.borrow().is_op_order_valid() {
            Self::recompute_block_order(block);
        } else {
            // Update the order of either operation if necessary.
            self.update_order_if_necessary();
            other.update_order_if_necessary();
        }

        self.order.load(Ordering::Relaxed) < other.order.load(Ordering::Relaxed)
    }

    /// Update the order index of this operation of this operation if necessary,
    /// potentially recomputing the order of the parent block.
    fn update_order_if_necessary(&self) {
        use core::sync::atomic::Ordering;

        assert!(self.parent().is_some(), "expected valid parent");

        // If the order is valid for this operation there is nothing to do.
        let block = self.parent().unwrap();
        if self.has_valid_order() || block.borrow().body().iter().count() == 1 {
            return;
        }

        let this = self.as_operation_ref();
        let prev = this.prev();
        let next = this.next();
        assert!(prev.is_some() || next.is_some(), "expected more than one operation in block");

        // If the operation is at the end of the block.
        if next.is_none() {
            let prev = prev.unwrap();
            let prev = prev.borrow();
            let prev_order = prev.order.load(Ordering::Acquire);
            if prev_order == Self::INVALID_ORDER {
                return Self::recompute_block_order(block);
            }

            // Add the stride to the previous operation.
            self.order.store(prev_order + Self::ORDER_STRIDE, Ordering::Release);
            return;
        }

        // If this is the first operation try to use the next operation to compute the
        // ordering.
        if prev.is_none() {
            let next = next.unwrap();
            let next = next.borrow();
            let next_order = next.order.load(Ordering::Acquire);
            match next_order {
                Self::INVALID_ORDER | 0 => {
                    return Self::recompute_block_order(block);
                }
                // If we can't use the stride, just take the middle value left. This is safe
                // because we know there is at least one valid index to assign to.
                order if order <= Self::ORDER_STRIDE => {
                    self.order.store(order / 2, Ordering::Release);
                }
                _ => {
                    self.order.store(Self::ORDER_STRIDE, Ordering::Release);
                }
            }
            return;
        }

        // Otherwise, this operation is between two others. Place this operation in
        // the middle of the previous and next if possible.
        let prev = prev.unwrap().borrow().order.load(Ordering::Acquire);
        let next = next.unwrap().borrow().order.load(Ordering::Acquire);
        if prev == Self::INVALID_ORDER || next == Self::INVALID_ORDER {
            return Self::recompute_block_order(block);
        }

        // Check to see if there is a valid order between the two.
        if prev + 1 == next {
            return Self::recompute_block_order(block);
        }
        self.order.store(prev + ((next - prev) / 2), Ordering::Release);
    }

    fn recompute_block_order(block: BlockRef) {
        use core::sync::atomic::Ordering;

        let block = block.borrow();
        let mut cursor = block.body().front();
        let mut index = 0;
        while let Some(op) = cursor.as_pointer() {
            index += Self::ORDER_STRIDE;
            cursor.move_next();
            let ptr = OperationRef::as_ptr(&op);
            unsafe {
                let order_addr = core::ptr::addr_of!((*ptr).order);
                (*order_addr).store(index, Ordering::Release);
            }
        }

        block.mark_op_order_valid();
    }

    /// Returns `None` if this operation has invalid ordering
    #[inline]
    pub(crate) fn order(&self) -> Option<u32> {
        use core::sync::atomic::Ordering;
        match self.order.load(Ordering::Acquire) {
            Self::INVALID_ORDER => None,
            order => Some(order),
        }
    }

    /// Returns `None` if this operation has invalid ordering
    #[inline]
    #[allow(unused)]
    pub(crate) fn get_or_compute_order(&self) -> u32 {
        use core::sync::atomic::Ordering;

        if let Some(order) = self.order() {
            return order;
        }

        Self::recompute_block_order(
            self.parent().expect("cannot compute block ordering for orphaned operation"),
        );

        self.order.load(Ordering::Acquire)
    }

    /// Returns true if this operation has a valid order
    #[inline(always)]
    pub(super) fn has_valid_order(&self) -> bool {
        self.order().is_some()
    }
}

/// Canonicalization
impl Operation {
    /// Populates `rewrites` with the set of canonicalization patterns registered for this operation
    #[inline]
    pub fn populate_canonicalization_patterns(
        &self,
        rewrites: &mut RewritePatternSet,
        context: Rc<Context>,
    ) {
        self.name.populate_canonicalization_patterns(rewrites, context);
    }
}

impl crate::traits::Foldable for Operation {
    fn fold(&self, results: &mut smallvec::SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        use crate::traits::Foldable;

        if let Some(foldable) = self.as_trait::<dyn Foldable>() {
            foldable.fold(results)
        } else {
            FoldResult::Failed
        }
    }

    fn fold_with<'operands>(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut smallvec::SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        use crate::traits::Foldable;

        if let Some(foldable) = self.as_trait::<dyn Foldable>() {
            foldable.fold_with(operands, results)
        } else {
            FoldResult::Failed
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AsCallableSymbolRef, AsSymbolRef, AttributeRef, Int, Op,
        dialects::builtin::attributes::IntAttr, testing::Test,
    };

    #[test]
    fn set_attribute_replaces_existing_attribute() {
        let test = Test::new("set_attribute_replaces_existing_attribute", |_| ([], []));
        let mut function = test.function();
        let first = test
            .context_rc()
            .create_attribute::<IntAttr, _>(Int::from(0u32))
            .as_attribute_ref();
        let replacement = test
            .context_rc()
            .create_attribute::<IntAttr, _>(Int::from(1u32))
            .as_attribute_ref();

        let mut function = function.borrow_mut();
        let operation = function.as_operation_mut();
        operation.set_attribute("test.attribute", first);
        operation.set_attribute("test.attribute", replacement);

        assert_eq!(operation.attributes().len(), 1);
        let actual = operation.get_attribute("test.attribute").unwrap();
        assert!(AttributeRef::ptr_eq(&actual, &replacement));
    }

    #[test]
    fn borrowed_operation_roundtrips_symbol_and_callable_handles() {
        let mut test = Test::named("borrowed_operation_roundtrips_symbol_and_callable_handles")
            .in_module("test");
        let i32 = test.context().i32();
        let function = test.define_function("callee", &[], &[i32]);
        let op_ref = function.as_operation_ref();

        let symbol_ref = {
            let op = op_ref.borrow();
            op.as_symbol_ref().expect("expected function operation to be a symbol")
        };
        assert_eq!(symbol_ref.borrow().as_operation_ref(), op_ref);

        let callable_ref = {
            let function = function.borrow();
            function.as_callable_symbol_ref()
        };
        assert_eq!(callable_ref.borrow().as_operation_ref(), op_ref);

        let symbol_from_callable = function.as_symbol_ref();
        assert_eq!(symbol_from_callable.borrow().as_operation_ref(), op_ref);
    }

    /// The memory-effect queries on `Operation` for an op that reads, one that writes and one with
    /// no effect.
    #[test]
    fn memory_effect_queries_distinguish_read_write_and_no_effect() {
        use crate::{
            Builder, BuilderExt, SourceSpan, ValueRef,
            dialects::{
                builtin::attributes::Overflow,
                test::{Add, Load, Store},
            },
            effects::MemoryEffect,
        };

        let mut test = Test::named("memory_effect_queries");
        let i32 = test.context().i32();
        let pointer = test.context().ptr_type(i32);
        test.with_function("memory_effect_queries", &[pointer, i32], &[]);
        let (store, load, add) = {
            let mut builder = test.function_builder();
            let entry = builder.entry_block();
            let (addr, value) = {
                let block = entry.borrow();
                (block.arguments()[0] as ValueRef, block.arguments()[1] as ValueRef)
            };
            let builder = builder.builder_mut();
            builder.set_insertion_point_to_end(entry);
            let store = builder.create::<Store, _>(SourceSpan::UNKNOWN)(addr, value).unwrap();
            let load = builder.create::<Load, _>(SourceSpan::UNKNOWN)(addr).unwrap();
            let add = builder.create::<Add, (_, _, _)>(SourceSpan::UNKNOWN)(
                value,
                value,
                Overflow::Wrapping,
            )
            .unwrap();
            (
                store.borrow().as_operation().as_operation_ref(),
                load.borrow().as_operation().as_operation_ref(),
                add.borrow().as_operation().as_operation_ref(),
            )
        };

        let store = store.borrow();
        assert!(store.has_memory_effect(MemoryEffect::Write));
        assert!(!store.has_memory_effect(MemoryEffect::Read));
        assert!(store.has_single_memory_effect(MemoryEffect::Write));
        assert!(!store.has_single_memory_effect(MemoryEffect::Read));
        assert!(!store.is_memory_effect_free());
        assert!(!store.would_be_trivially_dead_even_if_terminator());

        let load = load.borrow();
        assert!(load.has_memory_effect(MemoryEffect::Read));
        assert!(!load.has_memory_effect(MemoryEffect::Write));
        assert!(load.has_single_memory_effect(MemoryEffect::Read));
        assert!(!load.has_single_memory_effect(MemoryEffect::Write));
        assert!(!load.is_memory_effect_free());
        assert!(load.would_be_trivially_dead_even_if_terminator());

        // Both act on the test dialect's memory, through the operand `addr`
        for op in [&store, &load] {
            let effects = op.get_effects_recursively::<MemoryEffect>().unwrap();
            assert_eq!(effects.len(), 1, "{}", op.name());
            let effect = &effects[0];
            assert!(
                effect.resource().dyn_eq(&crate::dialects::test::TestMemoryResource),
                "{}",
                op.name()
            );
            assert_eq!(effect.value(), Some(op.operands()[0].borrow().as_value_ref()));
        }

        let add = add.borrow();
        assert!(!add.has_memory_effect(MemoryEffect::Read));
        assert!(!add.has_memory_effect(MemoryEffect::Write));
        assert!(!add.has_single_memory_effect(MemoryEffect::Read));
        assert!(add.is_memory_effect_free());
        assert!(add.would_be_trivially_dead_even_if_terminator());
    }
}
