use alloc::vec::Vec;

use super::{SymbolRef, SymbolUse};
use crate::{
    OperationRef, UnsafeIntrusiveEntityRef, WalkResult,
    dialects::builtin::attributes::SymbolRefAttr,
};

/// Links every symbol reference under `root` that is linked to no symbol to the symbol its path
/// names, and returns how many it linked.
///
/// The references are those `root` and every operation nested in it hold among their properties
/// and attributes, at any depth, as [Operation::walk_attributes](crate::Operation::walk_attributes)
/// visits them. A path is looked up as the parser and the verifier look it up: in the nearest
/// symbol table of the operation that holds the reference, or in that operation when it is a
/// symbol table with none above it.
///
/// A reference that is linked already is left as it is, whatever its path now names: a link that
/// disagrees with its path is for the verifier to report, not for this function to repair. A
/// reference whose path names no symbol is left unlinked, and so is a copy of a reference that
/// still shares the use of the reference it was copied from; the verifier reports both.
///
/// This is for IR put together from parts that were built apart: the parser links the references
/// of one input when it has read it, so a reference to a symbol of another input is left
/// unlinked, and stays so when the two are moved under one symbol table. Linking the whole after
/// the move is what makes it verify.
///
/// The operations under `root`, and the symbol tables above it that a path is looked up in, are
/// borrowed while the references are found, and each symbol that gains a use is mutably borrowed
/// while it does: the caller must hold no mutable borrow of an operation under `root` or of a
/// symbol table above it, and no borrow at all of a symbol a reference under `root` may name.
pub fn link_symbol_refs(root: OperationRef) -> usize {
    // What to link is found first, so that no operation is borrowed while a symbol is changed.
    let mut found: Vec<(OperationRef, UnsafeIntrusiveEntityRef<SymbolRefAttr>, SymbolRef)> =
        Vec::new();
    root.borrow().prewalk_all(|op| {
        let _ = op.walk_attributes(&mut |_, attr| {
            let Ok(reference) = attr.try_downcast_attr::<SymbolRefAttr>() else {
                return WalkResult::Continue(());
            };
            let (linked, path) = {
                let reference = reference.borrow();
                (
                    reference.try_user().and_then(|user| user.linked_symbol()).is_some(),
                    reference.path().clone(),
                )
            };
            if linked {
                return WalkResult::Continue(());
            }
            let symbol = match op.nearest_symbol_table() {
                Some(symbol_table) => {
                    symbol_table.borrow().as_symbol_table().and_then(|table| table.resolve(&path))
                }
                None => op.as_symbol_table().and_then(|table| table.resolve(&path)),
            };
            if let Some(symbol) = symbol {
                found.push((op.as_operation_ref(), reference, symbol));
            }
            WalkResult::Continue(())
        });
    });

    let mut linked = 0;
    for (owner, mut reference, symbol) in found {
        let user = reference.borrow().try_user();
        let user = match user {
            Some(user) => {
                // A copy of a reference shares the use of the reference it was copied from until
                // it is given one of its own: a use that tracks another reference, or that was
                // linked a moment ago on behalf of the reference it tracks, is left alone, and the
                // verifier reports the copy.
                let tracks_this = {
                    let tracked = user.borrow();
                    tracked.attr == reference && tracked.owner == owner
                };
                if user.is_linked() || !tracks_this {
                    continue;
                }
                user
            }
            // A reference the parser could not resolve has no use yet: a use records the
            // operation that holds the reference.
            None => {
                let context = owner.borrow().context_rc();
                let user = context.alloc_tracked(SymbolUse::new(owner, reference));
                reference.borrow_mut().set_user(user);
                user
            }
        };
        user.link_to_symbol(symbol);
        linked += 1;
    }
    linked
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::{format, rc::Rc, string::ToString};

    use super::*;
    use crate::{
        AttributeRef, BlockRef, Context, Op, Symbol as _, Usable,
        attributes::AttrList,
        dialects::builtin::{
            Module, World,
            attributes::{List, ListAttr, SymbolRef as SymbolRefValue},
        },
        parse::{self, ParserConfig},
        testing::Test,
    };

    const CALLEE: &str =
        "builtin.module public @a { builtin.function public extern(\"C\") @callee(i32) -> i32; };";
    const CALLER: &str = "builtin.module public @b { builtin.function public extern(\"C\") @f(%x: \
                          i32) -> i32 { %c = builtin.call ::@a::@callee(%x) : extern(\"C\") (i32) \
                          -> i32; builtin.ret %c : (i32); }; };";

    /// Parses `source` without verifying it: a reference that names nothing is left unlinked.
    fn parse(context: &Rc<Context>, name: &str, source: &str) -> OperationRef {
        let config = ParserConfig::new(context.clone()).verify_after_parse(false);
        parse::parse_any(config, name, source)
            .into_result()
            .unwrap_or_else(|report| panic!("{name} should parse: {report}"))
    }

    fn world_block(world: OperationRef) -> BlockRef {
        let world = world.borrow();
        world.downcast_ref::<World>().unwrap().body().entry_block_ref().unwrap()
    }

    fn module_block(module: OperationRef) -> BlockRef {
        let module = module.borrow();
        module.downcast_ref::<Module>().unwrap().body().entry_block_ref().unwrap()
    }

    fn move_to_end(mut op: OperationRef, block: BlockRef) {
        op.borrow_mut().remove();
        op.insert_at_end(block);
    }

    fn verify(root: OperationRef) -> Result<(), std::string::String> {
        root.borrow().recursively_verify().map_err(|report| report.to_string())
    }

    #[test]
    fn a_reference_to_a_symbol_added_after_it_was_parsed_is_linked() {
        for callee_first in [true, false] {
            let context = Rc::new(Context::default());
            let world =
                parse(&context, "world", "builtin.world { builtin.module public @seed { }; };");
            let callee = parse(&context, "a", CALLEE);
            let caller = parse(&context, "b", CALLER);
            let block = world_block(world);
            if callee_first {
                move_to_end(callee, block);
                move_to_end(caller, block);
            } else {
                move_to_end(caller, block);
                move_to_end(callee, block);
            }
            let error = verify(world).expect_err("the reference is not linked yet");
            assert!(error.contains("'::@a::@callee' is not linked"), "{error}");

            assert_eq!(link_symbol_refs(world), 1);
            verify(world).unwrap_or_else(|error| panic!("the world should verify: {error}"));
            assert_eq!(link_symbol_refs(world), 0, "what is linked is not linked again");
        }
    }

    #[test]
    fn a_reference_that_is_linked_already_is_left_as_it_is() {
        let context = Rc::new(Context::default());
        let world = parse(&context, "world", &format!("builtin.world {{ {CALLEE} {CALLER} }};"));
        verify(world).unwrap();
        assert_eq!(link_symbol_refs(world), 0);
        verify(world).unwrap();
    }

    #[test]
    fn a_reference_whose_path_names_nothing_stays_unlinked_for_the_verifier() {
        let context = Rc::new(Context::default());
        let world = parse(&context, "world", &format!("builtin.world {{ {CALLER} }};"));
        assert_eq!(link_symbol_refs(world), 0);
        let error = verify(world).expect_err("the callee does not exist");
        assert!(error.contains("'::@a::@callee' does not resolve"), "{error}");
    }

    #[test]
    fn a_relative_reference_is_looked_up_in_the_nearest_symbol_table() {
        let context = Rc::new(Context::default());
        let world = parse(
            &context,
            "world",
            "builtin.world { builtin.module public @m { builtin.function public extern(\"C\") \
             @f(%x: i32) -> i32 { %c = builtin.call @g(%x) : extern(\"C\") (i32) -> i32; \
             builtin.ret %c : (i32); }; }; };",
        );
        let other = parse(
            &context,
            "other",
            "builtin.module public @other { builtin.function public extern(\"C\") @g(i32) -> i32; \
             };",
        );
        // `@g` is moved beside `@f`, where the relative path finds it.
        let g = module_block(other).borrow().body().front().get().unwrap().as_operation_ref();
        let m = world_block(world).borrow().body().front().get().unwrap().as_operation_ref();
        move_to_end(g, module_block(m));

        assert_eq!(link_symbol_refs(world), 1);
        verify(world).unwrap_or_else(|error| panic!("the world should verify: {error}"));
    }

    #[test]
    fn only_what_is_under_the_root_is_linked() {
        let context = Rc::new(Context::default());
        let world = parse(&context, "world", "builtin.world { builtin.module public @seed { }; };");
        let callee = parse(&context, "a", CALLEE);
        let caller = parse(&context, "b", CALLER);
        let block = world_block(world);
        move_to_end(callee, block);
        move_to_end(caller, block);

        assert_eq!(link_symbol_refs(callee), 0, "the callee's module holds no reference");
        assert!(verify(world).is_err());
        assert_eq!(link_symbol_refs(caller), 1);
        verify(world).unwrap();
    }

    /// A list attribute holding `entries`.
    fn list(
        context: &Rc<Context>,
        entries: impl IntoIterator<Item = AttributeRef>,
    ) -> AttributeRef {
        let mut list = AttrList::new();
        for entry in entries {
            list.push_back(entry.borrow().as_attr().as_attr_ref());
        }
        context.create_attribute::<ListAttr, _>(List::from(list)).as_attribute_ref()
    }

    #[test]
    fn a_reference_nested_in_a_list_attribute_is_linked() {
        let mut test = Test::named("link").in_module("m");
        let mut owner = test.define_function("owner", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        let context = test.context_rc();
        let reference = context
            .create_attribute::<SymbolRefAttr, _>(SymbolRefValue::new(g.borrow().path(), None));
        owner
            .borrow_mut()
            .as_operation_mut()
            .set_attribute("refs", list(&context, [reference.as_attribute_ref()]));
        assert_eq!(g.borrow().iter_uses().count(), 0);

        assert_eq!(link_symbol_refs(owner.as_operation_ref()), 1);
        assert_eq!(g.borrow().iter_uses().count(), 1);
        owner.borrow().as_operation().verify().expect("the operation verifies");
    }

    #[test]
    fn a_copy_that_shares_the_use_of_its_original_is_left_for_the_verifier() {
        let mut test = Test::named("link").in_module("m");
        let mut owner = test.define_function("owner", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        let context = test.context_rc();
        // A reference with a use of its own that is linked to nothing, and a copy of it.
        let mut reference = context
            .create_attribute::<SymbolRefAttr, _>(SymbolRefValue::new(g.borrow().path(), None));
        let user = context.alloc_tracked(SymbolUse::new(owner.as_operation_ref(), reference));
        reference.borrow_mut().set_user(user);
        let copy = reference.borrow().as_attribute_ref().borrow().dyn_clone();
        owner
            .borrow_mut()
            .as_operation_mut()
            .set_attribute("original", reference.as_attribute_ref());
        owner.borrow_mut().as_operation_mut().set_attribute("copy", copy);

        assert_eq!(link_symbol_refs(owner.as_operation_ref()), 1, "the shared use is linked once");
        assert_eq!(g.borrow().iter_uses().count(), 1);
        let error = owner.borrow().as_operation().verify().expect_err("the copy is reported");
        let text = crate::testing::text::describe(&error);
        assert!(text.contains("shares the use of another reference"), "{text}");
    }
}
