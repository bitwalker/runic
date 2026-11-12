//! The type table of a [Context]: interning, and the builtin types' cached handles and constructors.

use alloc::rc::Rc;
use core::cell::RefCell;

use blink_alloc::Blink;

use super::*;
use crate::{
    FxHashMap,
    dialects::builtin::types::{
        ArrayType, FloatType, FunctionType, IntegerType, IsizeType, NeverType, PointerType,
        StructType, UnknownType,
    },
    types::{TypeName, TypeRegistration, TypeStorage, TypeUniquer},
};

/// The parts of a [Context] that interning a type uses, borrowed separately so that the builtin
/// types can be interned before the context they belong to is assembled.
#[derive(Copy, Clone)]
pub(super) struct TypeTables<'a> {
    /// The id of the context the tables belong to, recorded in every type interned with them.
    pub(super) owner: u32,
    pub(super) types: &'a RefCell<TypeUniquer>,
    pub(super) allocator: &'a Blink,
    pub(super) registered_dialects: &'a RefCell<FxHashMap<interner::Symbol, Rc<dyn Dialect>>>,
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
            let storage = self.allocator.put(TypeStorage::new(name, hash, self.owner, value));
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
pub(super) struct BuiltinTypes {
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
    pub(super) fn new(tables: TypeTables<'_>) -> Self {
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

impl Context {
    /// The parts of this context that interning a type uses.
    #[inline]
    fn type_tables(&self) -> TypeTables<'_> {
        TypeTables {
            owner: self.id,
            types: &self.types,
            allocator: &self.allocator,
            registered_dialects: &self.registered_dialects,
        }
    }

    /// Returns true if `ty` was interned by this context, and so may be used with it.
    #[inline]
    pub fn owns_type(&self, ty: crate::types::Type) -> bool {
        ty.owner_id() == self.id
    }

    /// Panics, in debug builds, unless `ty` was interned by this context. `role` describes where
    /// `ty` is being used, for the panic message.
    #[inline]
    #[track_caller]
    pub(crate) fn debug_assert_owns_type(&self, ty: crate::types::Type, role: &str) {
        debug_assert!(
            self.owns_type(ty),
            "{role} is `{ty}`, a type of another context: a type may only be used with the \
             context that interned it"
        );
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
    pub fn isize(&self) -> crate::types::Type {
        self.builtin_types.isize
    }

    /// The type `f32`.
    #[inline]
    pub fn f32(&self) -> crate::types::Type {
        self.builtin_types.f32
    }

    /// The type `f64`.
    #[inline]
    pub fn f64(&self) -> crate::types::Type {
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
        if let Some(pointee) = pointee {
            self.debug_assert_owns_type(pointee, "the pointee of a pointer type");
        }
        self.intern_type(PointerType::new(pointee, addrspace))
    }

    /// The array type `array<T; N>`.
    pub fn array_type(&self, element: crate::types::Type, len: u64) -> crate::types::Type {
        self.debug_assert_owns_type(element, "the element of an array type");
        self.intern_type(ArrayType::new(element, len))
    }

    /// The struct type `ty`. Every struct is accepted; its layout is not checked here.
    pub fn struct_type(&self, ty: StructType) -> crate::types::Type {
        for field in ty.fields() {
            self.debug_assert_owns_type(field.ty, "a field of a struct type");
        }
        self.intern_type(ty)
    }

    /// The function type `(params) -> results`.
    pub fn function_type(
        &self,
        params: impl IntoIterator<Item = crate::types::Type>,
        results: impl IntoIterator<Item = crate::types::Type>,
    ) -> crate::types::Type {
        let ty = FunctionType::new(params, results);
        for param in ty.params() {
            self.debug_assert_owns_type(*param, "a parameter of a function type");
        }
        for result in ty.results() {
            self.debug_assert_owns_type(*result, "a result of a function type");
        }
        self.intern_type(ty)
    }
}
