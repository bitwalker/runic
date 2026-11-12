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
    /// Returns the type equal to `value`, creating it with `alloc` if there is none yet.
    ///
    /// `alloc` is given the value and its hash, and must return a handle whose stored hash is that
    /// hash. If an equal type already exists, `value` is handed back instead of being dropped
    /// here, so that the caller can drop it once it has released the uniquer: a value's `Drop`
    /// may intern types itself.
    pub(crate) fn get_or_insert_with<T: TypeImpl>(
        &mut self,
        value: T,
        alloc: impl FnOnce(T, u64) -> Type,
    ) -> (Type, Option<T>) {
        let hash = super::hash_of(&value);
        if let Some(found) = self.table.find(hash, |ty| ty.downcast_ref::<T>() == Some(&value)) {
            return (*found, Some(value));
        }
        let ty = alloc(value, hash);
        debug_assert_eq!(ty.stored_hash(), hash);
        self.table.insert_unique(hash, ty, |ty| ty.stored_hash());
        (ty, None)
    }
}
