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

    /// Links this use to `symbol` through `uses`, the use list of `symbol`, held by the caller:
    /// used where that symbol is already borrowed.
    ///
    /// # Safety
    ///
    /// `uses` must be the use list of `symbol`, and this use must not be linked.
    pub(crate) unsafe fn link_into(mut self, symbol: SymbolRef, uses: &mut super::SymbolUseList) {
        debug_assert!(
            !self.is_linked(),
            "symbol use must be unlinked before it can be linked again"
        );
        self.borrow_mut().linked_to = Some(symbol);
        uses.push_back(self);
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

    /// Whether the callee of the call `op` resolves.
    fn call_resolves(op: crate::OperationRef) -> bool {
        use crate::dialects::builtin::Call;

        let op = op.borrow();
        op.downcast_ref::<Call>().unwrap().callee().resolve().is_some()
    }

    /// `op` verified on its own and recursively, with the errors described.
    fn verify(op: crate::OperationRef) -> [Result<(), alloc::string::String>; 2] {
        let op = op.borrow();
        let describe = |err: crate::Report| crate::testing::text::describe(&err);
        [op.verify().map_err(describe), op.recursively_verify().map_err(describe)]
    }

    /// Erasing a function unlinks the references inside it, among them a call of the function
    /// itself, without borrowing it twice, and the references to it from outside: the call in
    /// `@h` no longer resolves, and the module does not verify.
    #[test]
    fn erasing_a_function_unlinks_every_reference_to_it_and_in_it() {
        let (_context, module, [g, mut f, h]) = calls();
        let call_in_h = ops_named(h, "builtin.call")[0];
        assert!(call_resolves(call_in_h));
        f.borrow_mut().erase();
        assert_eq!(uses(g), 1, "the reference in @h remains");
        assert_eq!(uses(f), 0, "the recursive call and the call in @h are both unlinked");
        assert_eq!(uses(h), 0);
        assert!(!call_resolves(call_in_h));

        for err in verify(call_in_h).into_iter().chain([verify(module)[1].clone()]) {
            let err = err.expect_err("a call of an erased function must not verify");
            assert!(err.contains("symbol '@f' does not resolve"), "{err}");
            assert!(err.contains("no symbol visible to 'builtin.call' has this path"), "{err}");
        }
    }

    /// A module where `@h` calls `@f` directly and `@k` calls it through the alias `@af`.
    const ALIASED_CALLS: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        builtin.ret %a : (i32);
    };
    builtin.function_alias public @af -> @f;
    builtin.function public extern("C") @h(%a: i32) -> i32 {
        %r = builtin.call @f(%a) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);
    };
    builtin.function public extern("C") @k(%a: i32) -> i32 {
        %r = builtin.call @af(%a) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);
    };
};"#;

    /// After an erased function, both its direct call and a call through an alias of it are
    /// diagnosed: the direct call and the alias no longer resolve their reference, and the call
    /// through the alias reaches no callable.
    #[test]
    fn a_direct_call_and_a_call_through_an_alias_of_an_erased_function_are_diagnosed() {
        use crate::{
            SymbolTable,
            dialects::builtin::Module,
            parse::{ParserConfig, parse},
        };

        let test = Test::default();
        let module =
            parse::<Module>(ParserConfig::new(test.context_rc()), "aliased", ALIASED_CALLS)
                .into_result()
                .unwrap_or_else(|err| panic!("{}", crate::testing::text::describe(&err)));
        let [mut f, alias, h, k] = ["f", "af", "h", "k"]
            .map(|name| module.borrow().symbol_manager().lookup_op(name).unwrap());
        let module = module.as_operation_ref();
        let [direct, through_alias] = [h, k].map(|function| ops_named(function, "builtin.call")[0]);
        assert!(verify(module)[1].is_ok());

        f.borrow_mut().erase();

        let [own, _] = verify(direct);
        let err = own.expect_err("the direct call must not verify");
        assert!(err.contains("symbol '@f' does not resolve"), "{err}");

        let [own, _] = verify(alias);
        let err = own.expect_err("the alias of an erased function must not verify");
        assert!(err.contains("symbol '@f' does not resolve"), "{err}");

        // The call's own reference, to the alias, is sound; its verifier finds no callee
        let [own, _] = verify(through_alias);
        own.expect_err("a call through an alias of an erased function must not verify");
        assert!(verify(module)[1].is_err());
    }

    /// A symbol detached without being erased keeps its uses linked; another symbol inserted
    /// under its name is what their paths name now, which verification reports, naming both.
    #[test]
    fn a_reference_linked_to_a_detached_symbol_whose_name_was_reused_is_diagnosed() {
        use crate::{Ident, SymbolName, Visibility, dialects::builtin::ModuleBuilder};

        let (context, module, [mut g, f, h]) = calls();
        g.borrow_mut().remove();
        let mut mb = ModuleBuilder::new(module.try_downcast_op().unwrap());
        mb.define_global_variable(
            Ident::with_empty_span(SymbolName::intern("g")),
            Visibility::Public,
            context.i32(),
        )
        .unwrap();
        assert_eq!(uses(g), 2, "detaching leaves the uses linked");

        let call_of_g = ops_named(f, "builtin.call")[0];
        let reference = ops_named(h, "builtin.function_ref")[0];
        for op in [call_of_g, reference] {
            let [own, _] = verify(op);
            let err = own.expect_err("a reference linked to a detached symbol must not verify");
            assert!(
                err.contains("symbol '@g' is not linked to the symbol its path names"),
                "{err}"
            );
            assert!(
                err.contains("this path names '::@m::@g' (a 'builtin.global_variable')"),
                "{err}"
            );
            assert!(err.contains("but the reference is linked to '"), "{err}");
            assert!(err.contains("(a 'builtin.function')"), "{err}");
        }
    }

    /// The world of the corpus whose `@app::@main` calls `::@lib::@exported` and takes a reference
    /// to `::@lib::@reexported`, an alias of it: the context, the world, `@lib` and `@main`.
    fn world_of_lib_and_app() -> (
        alloc::rc::Rc<crate::Context>,
        crate::OperationRef,
        crate::OperationRef,
        crate::OperationRef,
    ) {
        use crate::{
            SymbolTable,
            dialects::builtin::World,
            parse::{ParserConfig, parse},
        };

        let source = crate::testing::calls::CORPUS
            .iter()
            .find(|source| source.contains("::@lib::@exported"))
            .unwrap();
        let test = Test::default();
        let world = parse::<World>(ParserConfig::new(test.context_rc()), "world", *source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}", crate::testing::text::describe(&err)));
        let lib = world.borrow().symbol_manager().lookup_op("lib").unwrap();
        let app = world.borrow().symbol_manager().lookup_op("app").unwrap();
        let main = app
            .borrow()
            .as_symbol_table()
            .unwrap()
            .symbol_manager()
            .lookup_op("main")
            .unwrap();
        (test.context_rc(), world.as_operation_ref(), lib, main)
    }

    /// References written with an absolute path (the call), with a relative path (the alias's
    /// target), to an alias (the function reference), and stored as a plain attribute through
    /// `set_symbol_attribute`, all verify.
    #[test]
    fn every_form_of_a_sound_reference_verifies() {
        let (_context, world, lib, mut main) = world_of_lib_and_app();
        assert!(verify(world)[1].is_ok(), "{:?}", verify(world)[1]);
        let reexported = lib
            .borrow()
            .as_symbol_table()
            .unwrap()
            .symbol_manager()
            .lookup("reexported")
            .unwrap();
        main.borrow_mut().set_symbol_attribute("note", reexported);
        assert!(verify(world)[1].is_ok(), "{:?}", verify(world)[1]);

        let (_context, module, [g, mut f, _h]) = calls();
        let g = g.borrow().as_symbol_ref().unwrap();
        f.borrow_mut().set_symbol_attribute("note", g);
        assert!(verify(module)[1].is_ok(), "{:?}", verify(module)[1]);
    }

    /// Erasing a module erases the symbols in it: a reference from outside to a symbol nested in
    /// it no longer resolves.
    #[test]
    fn erasing_a_module_unlinks_the_references_to_the_symbols_in_it() {
        let (_context, world, mut lib, main) = world_of_lib_and_app();
        let call = ops_named(main, "builtin.call")[0];
        let reference = ops_named(main, "builtin.function_ref")[0];
        lib.borrow_mut().erase();
        assert!(!call_resolves(call));
        let [own, _] = verify(call);
        let err = own.expect_err("a call into an erased module must not verify");
        assert!(err.contains("symbol '::@lib::@exported' does not resolve"), "{err}");
        let [own, _] = verify(reference);
        let err = own.expect_err("a reference into an erased module must not verify");
        assert!(err.contains("symbol '::@lib::@reexported' does not resolve"), "{err}");
        assert!(verify(world)[1].is_err());
    }

    /// Renaming a symbol rewrites the path of each of its uses, which all still verify; a use
    /// that names a symbol through a renamed symbol table keeps its old path, which names nothing
    /// now, and is diagnosed although it is still linked.
    #[test]
    fn renaming_keeps_the_rewritten_uses_verifying_and_a_stale_nested_path_is_diagnosed() {
        use crate::{SymbolName, SymbolTable};

        let (_context, mut world, mut lib, main) = world_of_lib_and_app();
        let call = ops_named(main, "builtin.call")[0];

        // `@exported` is named by the call (absolute) and by the alias (relative)
        lib.borrow_mut()
            .as_trait_mut::<dyn SymbolTable>()
            .unwrap()
            .rename(SymbolName::intern("exported"), SymbolName::intern("renamed"))
            .unwrap();
        assert!(verify(world)[1].is_ok(), "{:?}", verify(world)[1]);
        assert!(crate::testing::text::print_custom(&call.borrow()).contains("::@lib::@renamed"));

        // Only the uses of `@lib` itself would be rewritten, and it has none
        world
            .borrow_mut()
            .as_trait_mut::<dyn SymbolTable>()
            .unwrap()
            .rename(SymbolName::intern("lib"), SymbolName::intern("library"))
            .unwrap();
        assert!(call_resolves(call), "the call stays linked");
        let [own, _] = verify(call);
        let err = own.expect_err("a stale path must not verify");
        assert!(err.contains("symbol '::@lib::@renamed' does not resolve"), "{err}");
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
