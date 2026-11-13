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
///
/// A use is linked when it is in the use list of a symbol; it then records that symbol, so that
/// unlinking it, and resolving the reference that holds it, need no lookup of its path.
#[derive(Copy, Clone)]
pub struct SymbolUse {
    /// The user of the symbol
    pub owner: OperationRef,
    /// The symbol attribute of the op that stores the symbol
    pub attr: UnsafeIntrusiveEntityRef<SymbolRefAttr>,
    /// The symbol in whose use list this use was last put; meaningful only while it is linked
    linked_to: Option<SymbolRef>,
}

impl SymbolUse {
    #[inline]
    pub fn new(owner: OperationRef, attr: UnsafeIntrusiveEntityRef<SymbolRefAttr>) -> Self {
        Self {
            owner,
            attr,
            linked_to: None,
        }
    }

    /// Returns the symbol attribute that owns this use.
    pub fn symbol(&self) -> UnsafeIntrusiveEntityRef<SymbolRefAttr> {
        self.attr
    }
}

impl UnsafeIntrusiveEntityRef<SymbolUse> {
    /// Resolves `path` relative to the owning operation of this use, by looking it up in the
    /// nearest symbol table. Unlike [Self::linked_symbol], this does not consult the link.
    #[inline]
    pub fn resolve_symbol(self, path: &SymbolPath) -> Option<SymbolRef> {
        let owner = self.borrow().owner;
        let symbol_table = owner
            .nearest_symbol_table()
            .or_else(|| owner.name().implements::<dyn SymbolTable>().then_some(owner))?;
        let symbol_table = symbol_table.borrow();
        symbol_table.as_symbol_table()?.resolve(path)
    }

    /// The symbol this use is linked to, or `None` if it is not in the use list of any symbol.
    pub fn linked_symbol(self) -> Option<SymbolRef> {
        if !self.is_linked() {
            return None;
        }
        self.borrow().linked_to
    }

    /// Unlinks this use from the symbol it is linked to, if any.
    ///
    /// This mutably borrows that symbol: the caller must not hold a borrow of it.
    pub fn unlink_from_symbol(mut self) {
        let Some(mut symbol) = self.linked_symbol() else {
            return;
        };
        unsafe {
            symbol.borrow_mut().uses_mut().cursor_mut_from_ptr(self).remove();
        }
        self.borrow_mut().linked_to = None;
    }

    /// Links this use to `symbol`.
    pub fn link_to_symbol(mut self, mut symbol: SymbolRef) {
        debug_assert!(
            !self.is_linked(),
            "symbol use must be unlinked before it can be linked again"
        );
        self.borrow_mut().linked_to = Some(symbol);
        symbol.borrow_mut().insert_use(self);
    }

    /// Removes this use from the use list `uses`, which must be the use list of the symbol it is
    /// linked to, held by the caller: used where that symbol is already borrowed.
    ///
    /// # Safety
    ///
    /// `uses` must be the use list this use is linked into.
    pub(crate) unsafe fn unlink_from(mut self, uses: &mut super::SymbolUseList) {
        unsafe {
            uses.cursor_mut_from_ptr(self).remove();
        }
        self.borrow_mut().linked_to = None;
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

    /// A module whose `@f` calls `@g` and itself, and whose `@h` calls `@f`; `@g` is also named by
    /// a function reference in `@h`.
    const CALLS: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @g(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        %r = builtin.call @g(%a) : extern("C") (i32) -> i32;
        %s = builtin.call @f(%r) : extern("C") (i32) -> i32;
        builtin.ret %s : (i32);
    };
    builtin.function public extern("C") @h(%a: i32) -> i32 {
        %r = builtin.call @f(%a) : extern("C") (i32) -> i32;
        %p = builtin.function_ref @g : (i32) -> i32;
        builtin.ret %r : (i32);
    };
};"#;

    /// [CALLS], parsed, with `@g`, `@f` and `@h`, and the context that owns them.
    fn calls() -> (alloc::rc::Rc<crate::Context>, crate::OperationRef, [crate::OperationRef; 3]) {
        use crate::{
            SymbolTable,
            dialects::builtin::Module,
            parse::{ParserConfig, parse},
        };

        let test = Test::default();
        let module = parse::<Module>(ParserConfig::new(test.context_rc()), "calls", CALLS)
            .into_result()
            .unwrap_or_else(|err| panic!("{}", crate::testing::text::describe(&err)));
        let functions =
            ["g", "f", "h"].map(|name| module.borrow().symbol_manager().lookup_op(name).unwrap());
        (test.context_rc(), module.as_operation_ref(), functions)
    }

    /// The number of uses in the use list of the symbol `op`.
    fn uses(op: crate::OperationRef) -> usize {
        op.borrow().as_symbol_ref().unwrap().borrow().iter_uses().count()
    }

    /// The operations in the body of `function` named `name`.
    fn ops_named(
        function: crate::OperationRef,
        name: &str,
    ) -> alloc::vec::Vec<crate::OperationRef> {
        use alloc::string::ToString;

        let mut found = alloc::vec::Vec::new();
        function.borrow().prewalk_all(|op: &crate::Operation| {
            if op.name().to_string() == name {
                found.push(op.as_operation_ref());
            }
        });
        found
    }

    /// Erasing a call or a reference removes its use from the use list of the symbol it names,
    /// and the reference no longer resolves.
    #[test]
    fn an_erased_operation_leaves_the_use_list_of_its_symbol() {
        use crate::dialects::builtin::FunctionReference;

        let resolves = |op: crate::OperationRef| {
            let op = op.borrow();
            op.downcast_ref::<FunctionReference>().unwrap().callee().resolve().is_some()
        };

        let (_context, _module, [g, f, h]) = calls();
        assert_eq!((uses(g), uses(f), uses(h)), (2, 2, 0));

        let mut reference = ops_named(h, "builtin.function_ref")[0];
        assert!(resolves(reference));
        reference.borrow_mut().erase();
        assert_eq!((uses(g), uses(f)), (1, 2));
        assert!(!resolves(reference), "an erased reference resolves to nothing");

        let mut call = ops_named(h, "builtin.call")[0];
        call.borrow_mut().erase();
        assert_eq!((uses(g), uses(f)), (1, 1));
    }

    /// Erasing a function unlinks the references inside it, among them a call of the function
    /// itself, without borrowing it twice; references to it from outside stay.
    #[test]
    fn erasing_a_function_unlinks_the_calls_inside_it_including_one_of_itself() {
        let (_context, _module, [g, mut f, h]) = calls();
        f.borrow_mut().erase();
        assert_eq!(uses(g), 1, "the reference in @h remains");
        assert_eq!(uses(f), 1, "the recursive call is gone; the call in @h remains");
        assert_eq!(uses(h), 0);
    }

    /// Erasing a module unlinks every reference in it, those between its functions and the
    /// recursive call among them.
    #[test]
    fn erasing_a_module_unlinks_every_reference_inside_it() {
        let (_context, mut module, [g, f, h]) = calls();
        module.borrow_mut().erase();
        assert_eq!((uses(g), uses(f), uses(h)), (0, 0, 0));
    }
}
