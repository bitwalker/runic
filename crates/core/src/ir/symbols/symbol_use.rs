use alloc::collections::VecDeque;
use core::fmt;

use super::{SymbolPath, SymbolRef};
use crate::{
    Entity, EntityListItem, OperationRef, SymbolTable, UnsafeIntrusiveEntityRef,
    dialects::builtin::attributes::SymbolRefAttr,
};

pub type SymbolUseRef = UnsafeIntrusiveEntityRef<SymbolUse>;
pub type SymbolUseList = crate::EntityList<SymbolUse>;
pub type SymbolUseIter<'a> = crate::EntityListIter<'a, SymbolUse>;
pub type SymbolUseCursor<'a> = crate::EntityListCursor<'a, SymbolUse>;
pub type SymbolUseCursorMut<'a> = crate::EntityListCursorMut<'a, SymbolUse>;

/// A [SymbolUse] represents a use of a [super::Symbol] by an [crate::Operation]
#[derive(Copy, Clone)]
pub struct SymbolUse {
    /// The user of the symbol
    pub owner: OperationRef,
    /// The symbol attribute of the op that stores the symbol
    pub attr: UnsafeIntrusiveEntityRef<SymbolRefAttr>,
}

impl SymbolUse {
    #[inline]
    pub fn new(owner: OperationRef, attr: UnsafeIntrusiveEntityRef<SymbolRefAttr>) -> Self {
        Self { owner, attr }
    }

    /// Returns the symbol attribute that owns this use.
    pub fn symbol(&self) -> UnsafeIntrusiveEntityRef<SymbolRefAttr> {
        self.attr
    }
}

impl UnsafeIntrusiveEntityRef<SymbolUse> {
    /// Resolves the referenced symbol relative to the owning operation of this use.
    #[inline]
    pub fn resolve_symbol(self, path: &SymbolPath) -> Option<SymbolRef> {
        let owner = self.borrow().owner;
        let symbol_table = owner
            .nearest_symbol_table()
            .or_else(|| owner.name().implements::<dyn SymbolTable>().then_some(owner))?;
        let symbol_table = symbol_table.borrow();
        symbol_table.as_symbol_table()?.resolve(path)
    }

    /// Unlinks this use from its current symbol, if any.
    pub fn unlink_from_symbol(self, path: &SymbolPath) {
        if !self.is_linked() {
            return;
        }

        let mut symbol =
            self.resolve_symbol(path).expect("linked symbol uses must resolve to a symbol");
        unsafe {
            symbol.borrow_mut().uses_mut().cursor_mut_from_ptr(self).remove();
        }
    }

    /// Links this use to `symbol`.
    pub fn link_to_symbol(self, mut symbol: SymbolRef) {
        debug_assert!(
            !self.is_linked(),
            "symbol use must be unlinked before it can be linked again"
        );
        symbol.borrow_mut().insert_use(self);
    }
}

impl fmt::Debug for SymbolUse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.attr.borrow();
        f.debug_struct("SymbolUse")
            .field("symbol", &value.path())
            .finish_non_exhaustive()
    }
}

impl Entity for SymbolUse {}
impl EntityListItem for SymbolUse {}

/// An iterator over [SymbolUse] which owns the collection it iterates over.
///
/// This is primarily used in contexts where the set of symbol uses is being gathered from many
/// places, and thus [SymbolUseIter] is not able to be used.
pub struct SymbolUsesIter {
    items: VecDeque<SymbolUse>,
}
impl SymbolUsesIter {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
impl ExactSizeIterator for SymbolUsesIter {
    #[inline(always)]
    fn len(&self) -> usize {
        self.items.len()
    }
}
impl From<VecDeque<SymbolUse>> for SymbolUsesIter {
    fn from(items: VecDeque<SymbolUse>) -> Self {
        Self { items }
    }
}
impl FromIterator<SymbolUseRef> for SymbolUsesIter {
    fn from_iter<T: IntoIterator<Item = SymbolUseRef>>(iter: T) -> Self {
        Self {
            items: iter.into_iter().map(|user| *user.borrow()).collect(),
        }
    }
}
impl core::iter::FusedIterator for SymbolUsesIter {}
impl Iterator for SymbolUsesIter {
    type Item = SymbolUse;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop_front()
    }
}

/// An iterator over [SymbolUseRef] which owns the collection it iterates over.
///
/// This is primarily used in contexts where the set of symbol uses is being gathered from many
/// places, and thus [SymbolUseIter] is not able to be used.
pub struct SymbolUseRefsIter {
    items: VecDeque<SymbolUseRef>,
}
impl SymbolUseRefsIter {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
impl ExactSizeIterator for SymbolUseRefsIter {
    #[inline(always)]
    fn len(&self) -> usize {
        self.items.len()
    }
}
impl From<VecDeque<SymbolUseRef>> for SymbolUseRefsIter {
    fn from(items: VecDeque<SymbolUseRef>) -> Self {
        Self { items }
    }
}
impl FromIterator<SymbolUseRef> for SymbolUseRefsIter {
    fn from_iter<T: IntoIterator<Item = SymbolUseRef>>(iter: T) -> Self {
        Self {
            items: iter.into_iter().collect(),
        }
    }
}
impl core::iter::FusedIterator for SymbolUseRefsIter {}
impl Iterator for SymbolUseRefsIter {
    type Item = SymbolUseRef;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AsSymbolRef, Symbol, SymbolUse, Usable,
        dialects::builtin::attributes::{SymbolRef, SymbolRefAttr},
        testing::Test,
    };

    #[test]
    fn clearing_symbol_use_list_leaves_no_cached_target_state() {
        let mut test =
            Test::named("clearing_symbol_use_list_leaves_no_cached_target_state").in_module("test");
        let mut original = test.define_function("original", &[], &[]);
        let replacement = test.define_function("replacement", &[], &[]);
        let owner = test.define_function("owner", &[], &[]);
        let context = test.context_rc();

        let path = original.borrow().path();
        let mut attr = context.create_attribute::<SymbolRefAttr, _>(SymbolRef::new(path, None));
        let user = context.alloc_tracked(SymbolUse::new(owner.as_operation_ref(), attr));
        attr.borrow_mut().set_user(user);
        attr.borrow_mut().link(original.as_symbol_ref());

        assert!(user.is_linked());
        assert_eq!(original.borrow().iter_uses().count(), 1);

        original.borrow_mut().uses_mut().clear();

        assert!(!user.is_linked());
        assert_eq!(original.borrow().iter_uses().count(), 0);

        attr.borrow_mut().set_symbol(replacement.as_symbol_ref());
        let replacement_path = replacement.borrow().path();

        assert!(user.is_linked());
        assert_eq!(replacement.borrow().iter_uses().count(), 1);
        assert_eq!(attr.borrow().path(), &replacement_path);
    }

    /// An attribute that was never linked to a symbol is reported by the walker, skipped by the
    /// walks over uses, and rejected by the verifier.
    #[test]
    fn an_unlinked_symbol_attribute_is_reported_not_panicked_on() {
        use crate::{Op, Operation, SymbolName, SymbolNameComponent, SymbolPath, WalkResult};

        let mut test = Test::named("an_unlinked_symbol_attribute").in_module("test");
        let mut owner = test.define_function("owner", &[], &[]);
        let context = test.context_rc();

        let path =
            SymbolPath::new([SymbolNameComponent::Leaf(SymbolName::intern("missing"))]).unwrap();
        let attr = context.create_attribute::<SymbolRefAttr, _>(SymbolRef::new(path, None));
        assert!(attr.borrow().try_user().is_none());
        assert!(attr.borrow().resolve().is_none());
        owner.borrow_mut().as_operation_mut().set_attribute("reference", attr);

        let owner = owner.borrow();
        let owner = owner.as_operation();
        let mut linked = 0;
        let mut unresolved = alloc::vec::Vec::new();
        let _ = Operation::walk_symbol_refs(owner, |reference| {
            match reference {
                Ok(_) => linked += 1,
                Err(attr) => unresolved.push(attr),
            }
            WalkResult::Continue(())
        });
        assert_eq!(linked, 0);
        assert_eq!(unresolved, [attr]);

        let mut visited = 0;
        let _ = owner.walk_symbol_uses(|_| {
            visited += 1;
            WalkResult::Continue(())
        });
        assert_eq!(visited, 0);

        let err = owner.verify().expect_err("an unlinked symbol attribute must not verify");
        let message = alloc::format!("{err}");
        assert!(message.contains("symbol '@missing' does not resolve"), "{message}");
    }
}
