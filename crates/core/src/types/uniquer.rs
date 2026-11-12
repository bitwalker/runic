use super::{Type, TypeImpl};

/// The types interned in a context, keyed by the hash stored in each type's header.
///
/// The table holds only `Copy` handles, so dropping it never reads the type storage they point
/// at.
#[derive(Default)]
pub(crate) struct TypeUniquer {
    table: hashbrown::HashTable<Type>,
}

impl TypeUniquer {
    /// Returns the type equal to `value`, whose hash is `hash`, if there is one.
    ///
    /// `hash` must be [hash_of](super::hash_of)`(value)`, computed before the uniquer was
    /// borrowed: only the value's `PartialEq` runs here.
    pub(crate) fn find<T: TypeImpl>(&self, hash: u64, value: &T) -> Option<Type> {
        self.table.find(hash, |ty| ty.downcast_ref::<T>() == Some(value)).copied()
    }

    /// Returns the type equal to `value`, whose hash is `hash`, creating it with `alloc` if there
    /// is none yet.
    ///
    /// `alloc` runs while the uniquer is borrowed, so it must only allocate the storage: it must
    /// not intern types or register dialects. It must return a handle whose stored hash is `hash`.
    /// If an equal type already exists, `value` is handed back instead of being dropped here, so
    /// that the caller can drop it once it has released the uniquer: a value's `Drop` may intern
    /// types itself.
    pub(crate) fn get_or_insert_with<T: TypeImpl>(
        &mut self,
        value: T,
        hash: u64,
        alloc: impl FnOnce(T) -> Type,
    ) -> (Type, Option<T>) {
        if let Some(found) = self.find(hash, &value) {
            return (found, Some(value));
        }
        let ty = alloc(value);
        debug_assert_eq!(ty.stored_hash(), hash);
        self.table.insert_unique(hash, ty, |ty| ty.stored_hash());
        (ty, None)
    }
}
