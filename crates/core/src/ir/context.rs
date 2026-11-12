use alloc::{rc::Rc, sync::Arc, vec::Vec};
use core::{
    cell::{Cell, RefCell},
    mem::MaybeUninit,
};

use blink_alloc::Blink;
use runic_diagnostics::{
    Diagnostic, DiagnosticCollector, DiagnosticSet, OwnedDiagnostic, PushResult, SourceId,
    SourceMap, SourceMapError, SourceNamespace,
};
use traits::BranchOpInterface;

use super::{traits::BuildableTypeConstraint, *};
use crate::{
    AttributeRef, AttributeRegistration, FxHashMap,
    attributes::{AttributeName, DerivableTypeAttribute, Marker},
    constants::{ConstantData, ConstantId, ConstantPool},
    dialects::builtin::types::{
        ArrayType, FloatType, FunctionType, IntegerType, IsizeType, NeverType, PointerType,
        StructType, UnknownType,
    },
    types::{TypeName, TypeRegistration, TypeStorage, TypeUniquer},
};

/// Represents the shared state of the IR, used during a compilation session.
///
/// The primary purpose(s) of the context are:
///
/// * Provide storage/memory for all allocated IR entities for the lifetime of the session.
/// * Provide unique value and block identifiers for printing the IR
/// * Provide a uniqued constant pool
/// * Provide configuration used during compilation
///
/// # Safety
///
/// The [Context] _must_ live as long as any reference to an IR entity may be dereferenced.
pub struct Context {
    /// Copy-on-write so `sources()` is an O(1) snapshot that reports can retain.
    sources: RefCell<Arc<SourceMap>>,
    /// Non-fatal diagnostics (warnings, notes) emitted while working on this context.
    diagnostics: RefCell<DiagnosticCollector>,
    /// The types interned in this context. Their storage lives in `allocator`; the uniquer holds
    /// only `Copy` handles to it and does not read them when dropped, so the two may be dropped in
    /// either order.
    types: RefCell<TypeUniquer>,
    allocator: Rc<Blink>,
    /// Handles to the commonly used builtin types, interned when the context is created.
    builtin_types: BuiltinTypes,
    registered_dialects: RefCell<FxHashMap<interner::Symbol, Rc<dyn Dialect>>>,
    constants: RefCell<ConstantPool>,
    type_cache: RefCell<FxHashMap<core::any::TypeId, Arc<Type>>>,
    uniqued_attr_cache: RefCell<FxHashMap<AttributeName, AttributeRef>>,
    next_block_id: Cell<u32>,
    next_value_id: Cell<u32>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// The parts of a [Context] that interning a type uses, borrowed separately so that the builtin
/// types can be interned before the context they belong to is assembled.
#[derive(Copy, Clone)]
struct TypeTables<'a> {
    types: &'a RefCell<TypeUniquer>,
    allocator: &'a Blink,
    registered_dialects: &'a RefCell<FxHashMap<interner::Symbol, Rc<dyn Dialect>>>,
}

impl TypeTables<'_> {
    /// See [Context::intern_type].
    fn intern_type<T: TypeRegistration>(self, value: T) -> crate::types::Type {
        let hash = crate::types::hash_of(&value);
        if let Some(ty) = self.types.borrow().find(hash, &value) {
            return ty;
        }
        // Resolved before the uniquer is borrowed again: this may register `T`'s dialect, and so
        // run code that interns types.
        let name = self.get_registered_type_name::<T>();
        // That code may also have interned a value equal to `value`, which `get_or_insert_with`
        // finds again rather than inserting a second type.
        let (ty, rejected) = self.types.borrow_mut().get_or_insert_with(value, hash, |value| {
            let storage = self.allocator.put(TypeStorage::new(name, hash, value));
            // SAFETY: `storage` is an initialized `TypeStorage<T>` in the arena, which keeps it at
            // this address until the context's last `Rc<Blink>` is dropped and is never reset.
            // The `&mut` from `put` is converted to a pointer here and never used again, so
            // nothing can mutate the storage, and the pointer covers the whole storage.
            unsafe { crate::types::Type::from_storage(core::ptr::NonNull::from(storage)) }
        });
        // Dropped only now that the uniquer is released, as its `Drop` may intern types.
        drop(rejected);
        ty
    }

    /// See [Context::get_registered_type_name].
    fn get_registered_type_name<T: TypeRegistration>(self) -> TypeName {
        self.get_or_register_dialect::<T::Dialect>()
            .registered_type_name::<T>()
            .unwrap_or_else(|| {
                panic!(
                    "type `{}` is not registered with dialect '{}'",
                    core::any::type_name::<T>(),
                    <T::Dialect as DialectRegistration>::NAMESPACE
                )
            })
    }

    /// See [Context::get_or_register_dialect].
    fn get_or_register_dialect<T: DialectRegistration>(self) -> Rc<dyn Dialect> {
        let dialect_name = <T as DialectRegistration>::NAMESPACE.into();
        if let Some(dialect) = self.registered_dialects.borrow().get(&dialect_name).cloned() {
            return dialect;
        }

        let dialect = super::dialect::dialect_init::<T>();
        self.registered_dialects.borrow_mut().insert(dialect_name, Rc::clone(&dialect));
        dialect
    }
}

/// Handles to the builtin types a [Context] interns when it is created.
struct BuiltinTypes {
    i1: crate::types::Type,
    i8: crate::types::Type,
    i16: crate::types::Type,
    i32: crate::types::Type,
    i64: crate::types::Type,
    i128: crate::types::Type,
    isize: crate::types::Type,
    f32: crate::types::Type,
    f64: crate::types::Type,
    unknown: crate::types::Type,
    never: crate::types::Type,
    opaque_ptr: crate::types::Type,
}

impl BuiltinTypes {
    fn new(tables: TypeTables<'_>) -> Self {
        let int =
            |width| tables.intern_type(IntegerType::new(width).expect("a valid integer width"));
        Self {
            i1: int(1),
            i8: int(8),
            i16: int(16),
            i32: int(32),
            i64: int(64),
            i128: int(128),
            isize: tables.intern_type(IsizeType),
            f32: tables.intern_type(FloatType::F32),
            f64: tables.intern_type(FloatType::F64),
            unknown: tables.intern_type(UnknownType),
            never: tables.intern_type(NeverType),
            opaque_ptr: tables.intern_type(PointerType::new(None, 0)),
        }
    }
}

/// Collects the diagnostics emitted on a [Context] while it is alive, apart from those already
/// pending. Created by [`Context::diagnostic_scope`].
///
/// [`Self::finish`] restores the diagnostics that were pending and returns the scope's own.
/// Dropping the scope without finishing it restores the pending diagnostics and appends the
/// scope's to them, so nothing emitted in the scope is lost on an early return or unwind.
#[must_use = "a scope collects nothing once dropped"]
pub struct DiagnosticScope<'a> {
    context: &'a Context,
    /// The collector that was active when the scope started; `None` once restored.
    outer: Option<DiagnosticCollector>,
}

impl<'a> DiagnosticScope<'a> {
    /// The context this scope collects from.
    pub fn context(&self) -> &'a Context {
        self.context
    }

    /// End the scope, returning the diagnostics emitted while it was active.
    pub fn finish(mut self) -> DiagnosticCollector {
        let outer = self.outer.take().expect("a diagnostic scope is restored only once");
        core::mem::replace(&mut *self.context.diagnostics.borrow_mut(), outer)
    }
}

impl Drop for DiagnosticScope<'_> {
    fn drop(&mut self) {
        let Some(outer) = self.outer.take() else {
            return;
        };
        // Only a panic inside the collector itself can leave it borrowed here; restoring it then
        // would panic during unwind, so the scope's diagnostics are abandoned instead.
        let Ok(mut diagnostics) = self.context.diagnostics.try_borrow_mut() else {
            return;
        };
        let scoped = core::mem::replace(&mut *diagnostics, outer);
        let _ = diagnostics.merge(scoped.finish());
    }
}

impl Context {
    /// Create a context with its own source namespace.
    pub fn new() -> Self {
        #[cfg(feature = "std")]
        let namespace = SourceNamespace::fresh().expect("source namespaces exhausted");
        #[cfg(not(feature = "std"))]
        let namespace = SourceNamespace::new(core::num::NonZeroU32::MIN);
        Self::with_source_namespace(namespace)
    }

    /// Create a context whose sources live in `namespace`.
    pub fn with_source_namespace(namespace: SourceNamespace) -> Self {
        use hashbrown::hash_map::Entry;

        let mut registered_dialects = FxHashMap::default();
        for dialect in inventory::iter::<DialectRegistrationInfo>() {
            let namespace = interner::Symbol::intern(dialect.namespace());
            match registered_dialects.entry(namespace) {
                Entry::Vacant(entry) => {
                    entry.insert(dialect.create());
                }
                Entry::Occupied(prev) => {
                    panic!(
                        "conflicting dialect registrations for namespace '{namespace}': {} \
                         conflicts with previous registration of {}",
                        dialect.type_name(),
                        prev.key()
                    );
                }
            }
        }

        let allocator = Rc::new(Blink::new());
        let types = RefCell::default();
        let registered_dialects = RefCell::new(registered_dialects);
        // The handles stay valid when `types`, `allocator` and the dialects move into the context:
        // a handle points at its storage in the arena, which the `Rc` keeps where it is.
        let builtin_types = BuiltinTypes::new(TypeTables {
            types: &types,
            allocator: &allocator,
            registered_dialects: &registered_dialects,
        });
        Self {
            sources: RefCell::new(Arc::new(SourceMap::new(namespace))),
            diagnostics: RefCell::new(DiagnosticCollector::new()),
            types,
            allocator,
            builtin_types,
            registered_dialects,
            constants: Default::default(),
            type_cache: Default::default(),
            uniqued_attr_cache: Default::default(),
            next_block_id: Cell::new(0),
            next_value_id: Cell::new(0),
        }
    }

    /// A snapshot of every source registered so far.
    pub fn sources(&self) -> Arc<SourceMap> {
        self.sources.borrow().clone()
    }

    /// Register in-memory source text under `name`.
    pub fn add_source(
        &self,
        name: impl Into<alloc::string::String>,
        text: impl Into<alloc::string::String>,
    ) -> Result<SourceId, SourceMapError> {
        Arc::make_mut(&mut self.sources.borrow_mut()).insert(name, text, None)
    }

    /// Read `path` and register it; a path already registered is not re-read.
    #[cfg(feature = "std")]
    pub fn load_source_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<SourceId> {
        Arc::make_mut(&mut self.sources.borrow_mut()).load_file(path)
    }

    /// Record a non-fatal diagnostic. Errors are returned as `Err(Report)`, not emitted here.
    pub fn emit(&self, diagnostic: impl Diagnostic + Send + Sync + 'static) -> PushResult {
        self.diagnostics.borrow_mut().add(diagnostic)
    }

    /// Record a non-fatal [OwnedDiagnostic]. Errors are returned as `Err(Report)`, not emitted here.
    pub fn emit_owned(&self, diagnostic: OwnedDiagnostic) -> PushResult {
        self.diagnostics.borrow_mut().add_owned(diagnostic)
    }

    /// Drain the diagnostics emitted so far, with this context's sources attached.
    pub fn take_diagnostics(&self) -> DiagnosticSet {
        core::mem::take(&mut *self.diagnostics.borrow_mut())
            .finish()
            .attach_session_sources(self.sources())
    }

    /// Start collecting the diagnostics emitted on this context separately from those already
    /// pending, until the returned scope is finished or dropped.
    ///
    /// Scopes nest, and must end in the reverse of the order they were started.
    pub fn diagnostic_scope(&self) -> DiagnosticScope<'_> {
        let outer = core::mem::take(&mut *self.diagnostics.borrow_mut());
        DiagnosticScope {
            context: self,
            outer: Some(outer),
        }
    }

    pub fn registered_dialects(
        &self,
    ) -> core::cell::Ref<'_, FxHashMap<interner::Symbol, Rc<dyn Dialect>>> {
        self.registered_dialects.borrow()
    }

    pub fn get_registered_dialect(&self, dialect: impl Into<interner::Symbol>) -> Rc<dyn Dialect> {
        let dialect = dialect.into();
        self.registered_dialects.borrow()[&dialect].clone()
    }

    /// Like [`Context::get_registered_dialect`], but returns `None` when the dialect is not
    /// registered, rather than panicking.
    ///
    /// Callers that name a dialect from source text must use this: the name is only known at
    /// runtime, so it cannot be assumed to correspond to a registered dialect.
    pub fn find_registered_dialect(
        &self,
        dialect: impl Into<interner::Symbol>,
    ) -> Option<Rc<dyn Dialect>> {
        let dialect = dialect.into();
        self.registered_dialects.borrow().get(&dialect).cloned()
    }

    pub fn get_registered_name<T>(&self) -> OperationName
    where
        T: OpRegistration,
    {
        self.get_registered_dialect(T::dialect_name()).expect_registered_name::<T>()
    }

    pub fn get_registered_attribute_name<T>(&self) -> AttributeName
    where
        T: AttributeRegistration,
    {
        self.get_registered_dialect(T::dialect_name())
            .expect_registered_attribute_name::<T>()
    }

    /// The parts of this context that interning a type uses.
    #[inline]
    fn type_tables(&self) -> TypeTables<'_> {
        TypeTables {
            types: &self.types,
            allocator: &self.allocator,
            registered_dialects: &self.registered_dialects,
        }
    }

    /// Returns the type denoted by `value`, interning it on first use.
    pub fn intern_type<T: TypeRegistration>(&self, value: T) -> crate::types::Type {
        self.type_tables().intern_type(value)
    }

    /// The registration record of `T`, registering `T`'s dialect first if necessary.
    ///
    /// Panics if `T` is not registered with its dialect, or if another Rust type is registered
    /// under `T`'s name.
    pub fn get_registered_type_name<T: TypeRegistration>(&self) -> TypeName {
        self.type_tables().get_registered_type_name::<T>()
    }

    pub fn get_or_register_dialect<T>(&self) -> Rc<dyn Dialect>
    where
        T: DialectRegistration,
    {
        self.type_tables().get_or_register_dialect::<T>()
    }

    /// The integer type `iN` of `width` bits.
    ///
    /// Panics unless `1 <= width <= IntegerType::MAX_WIDTH`; see [Self::try_int_type].
    pub fn int_type(&self, width: u32) -> crate::types::Type {
        self.try_int_type(width).unwrap_or_else(|| {
            panic!(
                "invalid integer width {width}: an integer type is 1 to {} bits wide",
                IntegerType::MAX_WIDTH
            )
        })
    }

    /// The integer type `iN` of `width` bits, or `None` unless
    /// `1 <= width <= IntegerType::MAX_WIDTH`.
    pub fn try_int_type(&self, width: u32) -> Option<crate::types::Type> {
        let cached = &self.builtin_types;
        Some(match width {
            1 => cached.i1,
            8 => cached.i8,
            16 => cached.i16,
            32 => cached.i32,
            64 => cached.i64,
            128 => cached.i128,
            _ => return IntegerType::new(width).map(|ty| self.intern_type(ty)),
        })
    }

    /// The type `i1`.
    #[inline]
    pub fn i1(&self) -> crate::types::Type {
        self.builtin_types.i1
    }

    /// The type `i8`.
    #[inline]
    pub fn i8(&self) -> crate::types::Type {
        self.builtin_types.i8
    }

    /// The type `i16`.
    #[inline]
    pub fn i16(&self) -> crate::types::Type {
        self.builtin_types.i16
    }

    /// The type `i32`.
    #[inline]
    pub fn i32(&self) -> crate::types::Type {
        self.builtin_types.i32
    }

    /// The type `i64`.
    #[inline]
    pub fn i64(&self) -> crate::types::Type {
        self.builtin_types.i64
    }

    /// The type `i128`.
    #[inline]
    pub fn i128(&self) -> crate::types::Type {
        self.builtin_types.i128
    }

    /// The pointer-width integer type `isize`.
    #[inline]
    pub fn isize_type(&self) -> crate::types::Type {
        self.builtin_types.isize
    }

    /// The type `f32`.
    #[inline]
    pub fn f32_type(&self) -> crate::types::Type {
        self.builtin_types.f32
    }

    /// The type `f64`.
    #[inline]
    pub fn f64_type(&self) -> crate::types::Type {
        self.builtin_types.f64
    }

    /// The unknown type `?`.
    #[inline]
    pub fn unknown_type(&self) -> crate::types::Type {
        self.builtin_types.unknown
    }

    /// The bottom type `never`.
    #[inline]
    pub fn never_type(&self) -> crate::types::Type {
        self.builtin_types.never
    }

    /// The opaque pointer type `ptr`, in address space 0.
    #[inline]
    pub fn opaque_ptr_type(&self) -> crate::types::Type {
        self.builtin_types.opaque_ptr
    }

    /// The pointer type `ptr<T>` to `pointee`, in address space 0.
    pub fn ptr_type(&self, pointee: crate::types::Type) -> crate::types::Type {
        self.ptr_type_in(Some(pointee), 0)
    }

    /// A pointer type in address space `addrspace`; opaque if `pointee` is `None`.
    pub fn ptr_type_in(
        &self,
        pointee: Option<crate::types::Type>,
        addrspace: u32,
    ) -> crate::types::Type {
        self.intern_type(PointerType::new(pointee, addrspace))
    }

    /// The array type `array<T; N>`.
    pub fn array_type(&self, element: crate::types::Type, len: u64) -> crate::types::Type {
        self.intern_type(ArrayType::new(element, len))
    }

    /// The struct type `ty`. Every struct is accepted; its layout is not checked here.
    pub fn struct_type(&self, ty: StructType) -> crate::types::Type {
        self.intern_type(ty)
    }

    /// The function type `(params) -> results`.
    pub fn function_type(
        &self,
        params: impl IntoIterator<Item = crate::types::Type>,
        results: impl IntoIterator<Item = crate::types::Type>,
    ) -> crate::types::Type {
        self.intern_type(FunctionType::new(params, results))
    }

    pub fn get_marker<T>(self: &Rc<Context>) -> UnsafeIntrusiveEntityRef<T>
    where
        T: AttributeRegistration<Value = ()>,
    {
        <T as UniquedAttribute>::get_or_create_uniqued_attribute(self, ())
    }

    pub fn create_attribute<T, V>(self: &Rc<Context>, value: V) -> UnsafeIntrusiveEntityRef<T>
    where
        T: AttributeRegistration,
        <T as AttributeRegistration>::Value: From<V>,
    {
        <T as DerivableTypeAttribute>::create(self, value)
    }

    pub fn create_attribute_with_type<T, V>(
        self: &Rc<Context>,
        value: V,
        ty: Type,
    ) -> UnsafeIntrusiveEntityRef<T>
    where
        T: AttributeRegistration,
        <T as AttributeRegistration>::Value: From<V>,
    {
        <T as AttributeRegistration>::create(self, value, ty)
    }

    pub fn create_constant(&self, data: impl Into<ConstantData>) -> ConstantId {
        let mut constants = self.constants.borrow_mut();
        constants.insert(data.into())
    }

    pub fn get_constant(&self, id: ConstantId) -> Arc<ConstantData> {
        self.constants.borrow().get(id)
    }

    pub fn get_constant_size_in_bytes(&self, id: ConstantId) -> usize {
        self.constants.borrow().get_by_ref(id).len()
    }

    pub fn get_cached_type<T: BuildableTypeConstraint>(&self) -> Option<Arc<Type>> {
        self.type_cache.borrow().get(&core::any::TypeId::of::<T>()).cloned()
    }

    pub fn get_or_insert_type<T: BuildableTypeConstraint>(&self) -> Arc<Type> {
        match self.get_cached_type::<T>() {
            Some(ty) => ty,
            None => {
                let ty = Arc::new(<T as BuildableTypeConstraint>::build(self));
                let mut types = self.type_cache.borrow_mut();
                types.insert(core::any::TypeId::of::<T>(), Arc::clone(&ty));
                ty
            }
        }
    }

    /// Get a new [OpBuilder] for this context
    pub fn builder(self: Rc<Self>) -> OpBuilder {
        OpBuilder::new(Rc::clone(&self))
    }

    /// Create a new, detached and empty [Region]
    pub fn create_region(&self) -> RegionRef {
        self.alloc_tracked(Region::default())
    }

    /// Create a new, detached and empty [Block] with no parameters
    pub fn create_block(self: &Rc<Self>) -> BlockRef {
        let id = self.alloc_block_id();
        let block = Block::new(Rc::clone(self), id);
        self.alloc_tracked(block)
    }

    /// Create a new, detached and empty [Block], with parameters corresponding to the given types
    pub fn create_block_with_params<I>(self: &Rc<Self>, tys: I) -> BlockRef
    where
        I: IntoIterator<Item = Type>,
    {
        let mut block = Rc::clone(self).create_block();
        let owner = block;
        let args = tys.into_iter().enumerate().map(|(index, ty)| {
            let id = self.alloc_value_id();
            let arg = BlockArgument::new(
                SourceSpan::default(),
                id,
                ty,
                owner,
                index.try_into().expect("too many block arguments"),
            );
            self.alloc(arg)
        });
        block.borrow_mut().arguments_mut().extend(args);
        block
    }

    /// Append a new [BlockArgument] to `block`, with the given type and source location
    ///
    /// Returns the block argument as a `dyn Value` reference
    pub fn append_block_argument(
        &self,
        mut block: BlockRef,
        ty: Type,
        span: SourceSpan,
    ) -> ValueRef {
        let next_index = block.borrow().num_arguments();
        let id = self.alloc_value_id();
        let arg = BlockArgument::new(
            span,
            id,
            ty,
            block,
            next_index.try_into().expect("too many block arguments"),
        );
        let arg = self.alloc(arg);
        block.borrow_mut().arguments_mut().push(arg);
        arg.upcast()
    }

    /// Create a new [OpOperand] with the given value, owner, and index.
    ///
    /// NOTE: This inserts the operand as a user of `value`, but does _not_ add the operand to
    /// `owner`'s operand storage, the caller is expected to do that. This makes this function a
    /// more useful primitive.
    pub fn make_operand(&self, mut value: ValueRef, owner: OperationRef, index: u16) -> OpOperand {
        let op_operand = self.alloc_tracked(OpOperandImpl::new(value, owner, index));
        let mut value = value.borrow_mut();
        value.insert_use(op_operand);
        op_operand
    }

    /// Create a new [BlockOperand] with the given block, owner, and index.
    ///
    /// NOTE: This inserts the block operand as a user of `block`, but does _not_ add the block
    /// operand to `owner`'s successor storage, the caller is expected to do that. This makes this
    /// function a more useful primitive.
    pub fn make_block_operand(
        &self,
        mut block: BlockRef,
        owner: OperationRef,
        index: u16,
    ) -> BlockOperandRef {
        let block_operand = self.alloc_tracked(BlockOperand::new(owner, index));
        let mut block = block.borrow_mut();
        block.insert_use(block_operand);
        block_operand
    }

    /// Create a new [OpResult] with the given type, owner, and index
    ///
    /// NOTE: This does not attach the result to the operation, it is expected that the caller will
    /// do so.
    pub fn make_result(
        &self,
        span: SourceSpan,
        ty: Type,
        owner: OperationRef,
        index: u16,
    ) -> OpResultRef {
        let id = self.alloc_value_id();
        self.alloc(OpResult::new(span, id, ty, owner, index))
    }

    /// Create a new [OpResult] named `name`, with the given type, owner, and index
    ///
    /// NOTE: This does not attach the result to the operation, it is expected that the caller will
    /// do so.
    pub fn make_named_result(
        &self,
        span: SourceSpan,
        ty: Type,
        owner: OperationRef,
        name: interner::Symbol,
        index: u8,
    ) -> OpResultRef {
        let id = ValueId::from_symbol(name).with_result_index(index);
        self.alloc(OpResult::new(span, id, ty, owner, index.into()))
    }

    /// Appends `value` as an argument to the `branch_inst` instruction arguments list if the
    /// destination block of the `branch_inst` is `dest`.
    ///
    /// NOTE: Panics if `branch_inst` is not a branch instruction.
    pub fn append_branch_destination_argument(
        &self,
        mut branch_inst: OperationRef,
        dest: BlockRef,
        value: ValueRef,
    ) {
        let mut borrow = branch_inst.borrow_mut();
        let op = borrow.as_mut().as_operation_mut();
        assert!(
            op.as_trait::<dyn BranchOpInterface>().is_some(),
            "expected branch instruction, got {branch_inst:?}"
        );
        let dest_operand_groups: Vec<usize> = op
            .successors()
            .iter()
            .filter(|succ| succ.block.borrow().successor() == dest)
            .map(|succ| succ.operand_group as usize)
            .collect();
        for dest_group in dest_operand_groups {
            // The operand index is assigned when the operand is stored in its group
            let operand = self.make_operand(value, op.as_operation_ref(), 0);
            op.operands_mut().extend_group(dest_group, [operand]);
        }
    }

    /// Allocate a new uninitialized entity of type `T`
    ///
    /// In general, you can probably prefer [Context::alloc] instead, but for use cases where you
    /// need to allocate the space for `T` first, and then perform initialization, this can be
    /// used.
    pub fn alloc_uninit<T: 'static>(&self) -> UnsafeEntityRef<MaybeUninit<T>> {
        UnsafeEntityRef::new_uninit(&self.allocator)
    }

    /// Allocate a new uninitialized entity of type `T`, which needs to be tracked in an intrusive
    /// doubly-linked list.
    ///
    /// In general, you can probably prefer [Context::alloc_tracked] instead, but for use cases
    /// where you need to allocate the space for `T` first, and then perform initialization,
    /// this can be used.
    pub fn alloc_uninit_tracked<T: 'static>(&self) -> UnsafeIntrusiveEntityRef<MaybeUninit<T>> {
        UnsafeIntrusiveEntityRef::<T>::new_uninit_with_metadata(Default::default(), &self.allocator)
    }

    /// Allocate a new `UnsafeEntityRef<T>`.
    ///
    /// [UnsafeEntityRef] is a smart-pointer type for IR entities, which behaves like a ref-counted
    /// pointer with dynamically-checked borrow checking rules. It is designed to play well with
    /// entities allocated from a [Context], and with the somewhat cyclical nature of the IR.
    pub fn alloc<T: 'static>(&self, value: T) -> UnsafeEntityRef<T> {
        UnsafeEntityRef::new(value, &self.allocator)
    }

    /// Allocate a new `UnsafeIntrusiveEntityRef<T>`.
    ///
    /// [UnsafeIntrusiveEntityRef] is like [UnsafeEntityRef], except that it is specially designed
    /// for entities which are meant to be tracked in intrusive linked lists. For example, the
    /// blocks in a region, or the ops in a block. It does this without requiring the entity to know
    /// about the link at all, while still making it possible to access the link from the entity.
    pub fn alloc_tracked<T: 'static>(&self, value: T) -> UnsafeIntrusiveEntityRef<T> {
        UnsafeIntrusiveEntityRef::new_with_metadata(value, Default::default(), &self.allocator)
    }

    /// Allocate a new `UnsafeIntrusiveMapEntityRef<T>`.
    ///
    /// [UnsafeIntrusiveMapEntityRef] is like [UnsafeEntityRef], except that it is specially
    /// designed for entities which are meant to be tracked in intrusive red/black trees. For
    /// example, the attributes of an op. It does this without requiring the entity to know about
    /// the link at all, while still making it possible to access the link from the entity.
    pub fn alloc_map_item<T: EntityMapItem + 'static>(
        &self,
        value: T,
    ) -> UnsafeIntrusiveMapEntityRef<T> {
        UnsafeIntrusiveMapEntityRef::new_with_metadata(value, Default::default(), &self.allocator)
    }

    fn alloc_block_id(&self) -> BlockId {
        let id = self.next_block_id.get();
        self.next_block_id.set(id + 1);
        BlockId::from_u32(id)
    }

    fn alloc_value_id(&self) -> ValueId {
        let id = self.next_value_id.get();
        self.next_value_id.set(id + 1);
        ValueId::from_u32(id)
    }
}

trait UniquedAttribute: AttributeRegistration {
    fn get_or_create_uniqued_attribute(
        context: &Rc<Context>,
        value: <Self as AttributeRegistration>::Value,
    ) -> UnsafeIntrusiveEntityRef<Self>;
    #[allow(unused)]
    fn get_or_create_uniqued_attribute_with_type(
        context: &Rc<Context>,
        value: <Self as AttributeRegistration>::Value,
        ty: Type,
    ) -> UnsafeIntrusiveEntityRef<Self>;
}

impl<T: AttributeRegistration> UniquedAttribute for T {
    default fn get_or_create_uniqued_attribute(
        context: &Rc<Context>,
        value: <Self as AttributeRegistration>::Value,
    ) -> UnsafeIntrusiveEntityRef<Self> {
        context.create_attribute::<T, _>(value)
    }

    default fn get_or_create_uniqued_attribute_with_type(
        context: &Rc<Context>,
        value: <Self as AttributeRegistration>::Value,
        ty: Type,
    ) -> UnsafeIntrusiveEntityRef<Self> {
        context.create_attribute_with_type::<T, _>(value, ty)
    }
}

impl<T: AttributeRegistration + Marker> UniquedAttribute for T {
    fn get_or_create_uniqued_attribute(
        context: &Rc<Context>,
        value: <Self as AttributeRegistration>::Value,
    ) -> UnsafeIntrusiveEntityRef<Self> {
        use hashbrown::hash_map::Entry;
        let name = context.get_registered_attribute_name::<T>();
        let mut cache = context.uniqued_attr_cache.borrow_mut();
        match cache.entry(name) {
            Entry::Vacant(entry) => {
                let attr = context.create_attribute::<T, _>(value);
                entry.insert(attr as AttributeRef);
                attr
            }
            Entry::Occupied(entry) => entry.get().try_downcast_attr().unwrap(),
        }
    }

    fn get_or_create_uniqued_attribute_with_type(
        context: &Rc<Context>,
        value: <Self as AttributeRegistration>::Value,
        ty: Type,
    ) -> UnsafeIntrusiveEntityRef<Self> {
        use hashbrown::hash_map::Entry;
        let name = context.get_registered_attribute_name::<T>();
        let mut cache = context.uniqued_attr_cache.borrow_mut();
        match cache.entry(name) {
            Entry::Vacant(entry) => {
                let attr = context.create_attribute_with_type::<T, _>(value, ty);
                entry.insert(attr as AttributeRef);
                attr
            }
            Entry::Occupied(entry) => entry.get().try_downcast_attr().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    use crate::{
        Context, Type,
        diagnostics::{DiagnosticSet, diagnostic},
        testing::Test,
    };

    /// A block holds more than 255 arguments, each keeping its position in the argument list.
    #[test]
    fn block_with_more_than_255_arguments() {
        const NUM_PARAMS: usize = 300;

        let test = Test::new("wide", &vec![Type::U32; NUM_PARAMS], &[]);
        let function = test.function().borrow();
        let entry = function.entry_block().borrow();
        let args = entry.arguments();

        assert_eq!(args.len(), NUM_PARAMS);
        assert_eq!(args[NUM_PARAMS - 1].borrow().index(), NUM_PARAMS - 1);
    }

    fn messages(diagnostics: &DiagnosticSet) -> Vec<String> {
        diagnostics.iter().map(|entry| entry.diagnostic.to_string()).collect()
    }

    /// Each scope collects only what was emitted while it was the innermost one.
    #[test]
    fn nested_diagnostic_scopes_collect_their_own_diagnostics() {
        let context = Context::default();
        let _ = context.emit(diagnostic!(severity: Warning, message: "pending"));

        let outer = context.diagnostic_scope();
        let _ = context.emit(diagnostic!(severity: Warning, message: "outer"));
        let inner = context.diagnostic_scope();
        let _ = context.emit(diagnostic!(severity: Warning, message: "inner"));
        let inner = inner.finish().finish();
        let _ = context.emit(diagnostic!(severity: Warning, message: "outer again"));
        let outer = outer.finish().finish();

        assert_eq!(messages(&inner), ["inner"]);
        assert_eq!(messages(&outer), ["outer", "outer again"]);
        assert_eq!(messages(&context.take_diagnostics()), ["pending"]);
    }

    /// A scope dropped without being finished hands its diagnostics to the enclosing collector,
    /// after the ones that were already there.
    #[test]
    fn a_dropped_diagnostic_scope_merges_into_the_enclosing_collector() {
        let context = Context::default();
        let _ = context.emit(diagnostic!(severity: Warning, message: "pending"));

        let outer = context.diagnostic_scope();
        let _ = context.emit(diagnostic!(severity: Warning, message: "outer"));
        {
            let _inner = context.diagnostic_scope();
            let _ = context.emit(diagnostic!(severity: Warning, message: "inner"));
        }
        assert_eq!(messages(&outer.finish().finish()), ["outer", "inner"]);

        {
            let _scope = context.diagnostic_scope();
            let _ = context.emit(diagnostic!(severity: Warning, message: "dropped"));
        }
        assert_eq!(messages(&context.take_diagnostics()), ["pending", "dropped"]);
    }
}
