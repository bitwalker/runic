//! What a symbol is called, and where it is placed, outside the IR.
//!
//! Two entries of an operation's attribute dictionary have a fixed meaning: [LINK_NAME] and
//! [SECTION]. They are written as any attribute is, `attributes { link_name = "erlang:+/2" }`, and
//! travel with the operation through cloning, printing, parsing and the passes. A backend that
//! produces something a linker or a host reads takes the name it exports a function, an alias or a
//! global under from [effective_link_name], and the data section a global's storage is placed in
//! from [section], so that all globals of one section can be collected together.
//!
//! Only the shape of the entries is verified (see [verify_link_attributes]): a link name is a
//! non-empty string, and only a global variable has a section. That link names are unique is a
//! property of a link, not of one module, and is for a backend to check. This holds for the default
//! name too: it is the symbol's absolute path with its components joined by `::` and not quoted, so
//! two different symbols can have one default name (see [default_link_name]), and a backend checks
//! the uniqueness of the effective names of what it exports.
//!
//! This module also links symbol references that were built apart (see [link_symbol_refs]).

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use super::{SymbolNameComponent, SymbolRef, SymbolUse};
use crate::{
    CompactString, Operation, OperationRef, Report, Spanned, UnsafeIntrusiveEntityRef, WalkResult,
    diagnostics::report,
    dialects::builtin::attributes::{StringAttr, SymbolRefAttr},
};

/// The name of the attribute that gives a symbol the name it has for a linker or a host.
///
/// When present on a function, a function alias or a global variable, its value is a non-empty
/// string that replaces the default name, the symbol's absolute path (see [default_link_name]).
pub const LINK_NAME: &str = "link_name";

/// The name of the attribute that gives a global variable the data section its storage is placed in.
///
/// Its value is a non-empty string. A linker can collect all globals of one section together.
/// Only a global variable may have it.
pub const SECTION: &str = "section";

/// The string value of the entry `name` of the dictionary of `op`, or `None` when `op` has no
/// such entry or the entry is not a string attribute.
fn string_entry(op: &Operation, name: &str) -> Option<CompactString> {
    let handle = op.get_attribute(crate::interner::Symbol::intern(name))?;
    let value = handle.borrow();
    value
        .downcast_ref::<StringAttr>()
        .map(|string| CompactString::from(string.as_str()))
}

/// A symbol whose dictionary may hold the `link_name` entry, and the `section` entry if it is a
/// global variable: `builtin.function`, `builtin.function_alias` and `builtin.global_variable`
/// implement it, and a backend that collects what it links finds those operations with
/// `as_trait::<dyn LinkedSymbol>()`.
///
/// It is a marker that makes an operation's verifier check those entries, once, through
/// `Verify<dyn LinkedSymbol>` (see [verify_link_attributes]).
pub trait LinkedSymbol: super::Symbol {}

/// The value of the `link_name` entry of the dictionary of `op`: the name the symbol has for a
/// linker or a host. It is `None` when `op` has no such entry or the entry is not a string.
///
/// The string is a copy of the attribute's (a short name is copied without allocating).
pub fn link_name(op: &Operation) -> Option<CompactString> {
    string_entry(op, LINK_NAME)
}

/// The value of the `section` entry of the dictionary of `op`: the data section the storage of a
/// global variable is placed in. It is `None` when `op` has no such entry or the entry is not a
/// string.
///
/// The string is a copy of the attribute's (a short name is copied without allocating).
pub fn section(op: &Operation) -> Option<CompactString> {
    string_entry(op, SECTION)
}

/// The name `op` has for a linker or a host when it has no `link_name`: its absolute path, the
/// names of the symbol tables that enclose it and its own joined by `::`, with no leading `::` and
/// no quoting (`lib::math::add` for `::@lib::@math::@add`).
///
/// The default is a convenience, and it is not unique: the components are joined without quoting,
/// so two different symbols can have the same default (`::@a::@"b::c"` and `::@a::@b::@c` are both
/// `a::b::c`). A backend checks that the effective names (see [effective_link_name]) of what it
/// exports are unique, and does not assume the default makes them so.
///
/// It is `None` when `op` is not a symbol, and when it is a symbol that is neither a symbol table
/// nor in one: a symbol that is not attached to a root has no path to name it by.
///
/// The ancestors of `op` are borrowed to find the path, as [Symbol::path](super::Symbol::path) borrows them.
pub fn default_link_name(op: &Operation) -> Option<String> {
    let symbol = op.as_symbol()?;
    if op.parent_op().is_none() && !op.implements::<dyn super::SymbolTable>() {
        return None;
    }
    let mut name = String::new();
    for component in symbol.path().components() {
        let segment = match component {
            SymbolNameComponent::Root => continue,
            SymbolNameComponent::Component(segment) | SymbolNameComponent::Leaf(segment) => segment,
        };
        if !name.is_empty() {
            name.push_str("::");
        }
        name.push_str(segment.as_str());
    }
    Some(name)
}

/// The name a symbol has for a linker or a host, as [effective_link_name] finds it: the value of
/// the `link_name` entry, or the default, which is built. It dereferences to the string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectiveLinkName {
    /// The value of the `link_name` entry.
    Declared(CompactString),
    /// The absolute path of the symbol (see [default_link_name]).
    Default(String),
}

impl core::ops::Deref for EffectiveLinkName {
    type Target = str;

    fn deref(&self) -> &str {
        match self {
            Self::Declared(name) => name,
            Self::Default(name) => name,
        }
    }
}

/// The name `op` has for a linker or a host: its `link_name` when it has one, else its
/// [default_link_name].
pub fn effective_link_name(op: &Operation) -> Option<EffectiveLinkName> {
    match link_name(op) {
        Some(name) => Some(EffectiveLinkName::Declared(name)),
        None => default_link_name(op).map(EffectiveLinkName::Default),
    }
}

/// The text form of the attribute `attr` of `op`: `#builtin.i32<1>`.
fn print_attribute(op: &Operation, attr: &crate::AttributeRef) -> String {
    let flags = Default::default();
    let mut printer = crate::print::AsmPrinter::root(op.context_rc(), &flags);
    printer.print_attribute_value(&*attr.borrow());
    printer.finish().to_string()
}

/// Verifies the `link_name` and `section` entries of the dictionary of `op`, which is a
/// `builtin.function`, a `builtin.function_alias` or a `builtin.global_variable` (`op`'s name is
/// the one the errors report).
///
/// A `link_name` is a string that is not empty, and a `section` is a string that is not empty on a
/// global variable; a function or an alias has no section. The error is an invalid operation whose
/// primary label says what was found.
pub fn verify_link_attributes(op: &Operation) -> Result<(), Report> {
    let invalid = |label: String| {
        report!(
            message: (format!("invalid operation {}", op.name())),
            labels: [primary(op.span(), label)],
        )
    };
    let non_empty_string = |name: &str| -> Option<String> {
        let attr = op.get_attribute(crate::interner::Symbol::intern(name))?;
        match string_entry(op, name) {
            Some(value) if !value.is_empty() => None,
            _ => Some(format!(
                "{name} must be a non-empty string, but this is {}",
                print_attribute(op, &attr)
            )),
        }
    };
    if let Some(label) = non_empty_string(LINK_NAME) {
        return Err(invalid(label));
    }
    if op.has_attribute(crate::interner::Symbol::intern(SECTION)) {
        if op.downcast_ref::<crate::dialects::builtin::GlobalVariable>().is_none() {
            return Err(invalid(format!(
                "only a global variable has a section, not {}",
                op.name()
            )));
        }
        if let Some(label) = non_empty_string(SECTION) {
            return Err(invalid(label));
        }
    }
    Ok(())
}

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

    use alloc::{
        format,
        rc::Rc,
        string::{String, ToString},
        vec::Vec,
    };

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

    /// Parses `source` without verifying it, and returns the operations named `name` in the order
    /// they appear.
    fn ops_named(context: &Rc<Context>, source: &str, name: &str) -> Vec<OperationRef> {
        let root = parse(context, "link.runic", source);
        let mut found = Vec::new();
        root.borrow().prewalk_all(|op| {
            if op.name().to_string() == name {
                found.push(op.as_operation_ref());
            }
        });
        found
    }

    const THREE: &str = "builtin.world { builtin.module public @m { builtin.function public \
                         extern(\"C\") @plain() { builtin.ret; }; builtin.function public \
                         extern(\"C\") @named() attributes { link_name = \"erlang:+/2\" }; \
                         builtin.function_alias public @alias -> @named attributes { link_name = \
                         \"other\" }; builtin.global_variable public @g : ptr<i32> attributes { \
                         link_name = \"g_name\", section = \".data.x\" }; }; };";

    fn text_of(
        op: OperationRef,
        getter: fn(&Operation) -> Option<CompactString>,
    ) -> Option<String> {
        let op = op.borrow();
        getter(&op).map(|name| String::from(&*name))
    }

    #[test]
    fn the_link_name_and_section_of_an_operation_are_read_from_its_dictionary() {
        let context = Rc::new(Context::default());
        let functions = ops_named(&context, THREE, "builtin.function");
        assert_eq!(text_of(functions[0], link_name), None);
        assert_eq!(text_of(functions[1], link_name).as_deref(), Some("erlang:+/2"));
        assert_eq!(text_of(functions[1], section), None);
        let aliases = ops_named(&context, THREE, "builtin.function_alias");
        assert_eq!(text_of(aliases[0], link_name).as_deref(), Some("other"));
        let globals = ops_named(&context, THREE, "builtin.global_variable");
        assert_eq!(text_of(globals[0], link_name).as_deref(), Some("g_name"));
        assert_eq!(text_of(globals[0], section).as_deref(), Some(".data.x"));
    }

    #[test]
    fn an_entry_that_is_not_a_string_has_no_link_name() {
        let context = Rc::new(Context::default());
        let source = "builtin.function public extern(\"C\") @f() attributes { link_name = \
                      #builtin.i32<1> };";
        let functions = ops_named(&context, source, "builtin.function");
        assert_eq!(text_of(functions[0], link_name), None);
    }

    fn default_of(context: &Rc<Context>, source: &str, name: &str, index: usize) -> Option<String> {
        let ops = ops_named(context, source, name);
        default_link_name(&ops[index].borrow())
    }

    #[test]
    fn the_default_link_name_is_the_absolute_path_without_quoting() {
        let context = Rc::new(Context::default());
        let source = "builtin.world { builtin.module public @lib { builtin.module public @math { \
                      builtin.function public extern(\"C\") @add(); builtin.function public \
                      extern(\"C\") @\"a b::c\"(); }; }; builtin.component public @acme::@a { \
                      builtin.function public extern(\"C\") @run(); }; };";
        assert_eq!(
            default_of(&context, source, "builtin.function", 0).as_deref(),
            Some("lib::math::add")
        );
        assert_eq!(
            default_of(&context, source, "builtin.function", 1).as_deref(),
            Some("lib::math::a b::c")
        );
        assert_eq!(
            default_of(&context, source, "builtin.function", 2).as_deref(),
            Some("acme::a::run")
        );
        assert_eq!(
            default_of(&context, source, "builtin.component", 0).as_deref(),
            Some("acme::a")
        );
    }

    /// Two different symbols can have one default name, because the components are joined without
    /// quoting: the function `::@a::@"b::c"` and the function `::@a::@b::@c` are both `a::b::c`.
    /// The default is a convenience, and a backend checks the uniqueness of the effective names it
    /// exports.
    #[test]
    fn two_different_symbols_can_have_the_same_default_link_name() {
        let context = Rc::new(Context::default());
        let source = "builtin.world { builtin.module public @a { builtin.function public \
                      extern(\"C\") @\"b::c\"(); builtin.module public @b { builtin.function \
                      public extern(\"C\") @c(); }; }; };";
        let first = default_of(&context, source, "builtin.function", 0);
        let second = default_of(&context, source, "builtin.function", 1);
        assert_eq!(first.as_deref(), Some("a::b::c"));
        assert_eq!(first, second);
        let functions = ops_named(&context, source, "builtin.function");
        let paths = functions
            .iter()
            .map(|function| function.borrow().as_symbol().unwrap().path().to_string())
            .collect::<Vec<_>>();
        assert_ne!(paths[0], paths[1]);
    }

    /// The function, the alias and the global variable are linked symbols; a module is not.
    #[test]
    fn a_function_an_alias_and_a_global_are_linked_symbols_and_a_module_is_not() {
        let context = Rc::new(Context::default());
        let linked = |name: &str| {
            let ops = ops_named(&context, THREE, name);
            ops.iter().all(|op| op.borrow().as_trait::<dyn LinkedSymbol>().is_some())
        };
        assert!(linked("builtin.function"));
        assert!(linked("builtin.function_alias"));
        assert!(linked("builtin.global_variable"));
        let modules = ops_named(&context, THREE, "builtin.module");
        assert!(modules.iter().all(|op| op.borrow().as_trait::<dyn LinkedSymbol>().is_none()));
    }

    #[test]
    fn an_operation_that_is_not_a_symbol_has_no_default_link_name() {
        let context = Rc::new(Context::default());
        assert_eq!(default_of(&context, THREE, "builtin.ret", 0), None);
    }

    #[test]
    fn the_effective_link_name_is_the_link_name_or_else_the_default() {
        let context = Rc::new(Context::default());
        let functions = ops_named(&context, THREE, "builtin.function");
        let named = functions[1].borrow();
        let name = effective_link_name(&named).unwrap();
        assert!(matches!(name, EffectiveLinkName::Declared(_)));
        assert_eq!(&*name, "erlang:+/2");
        let plain = functions[0].borrow();
        let name = effective_link_name(&plain).unwrap();
        assert!(matches!(name, EffectiveLinkName::Default(_)));
        assert_eq!(&*name, "m::plain");
    }

    fn verify_source(source: &str) -> Result<(), std::string::String> {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context);
        parse::parse_any(config, "link.runic", source)
            .into_result()
            .map(|_| ())
            .map_err(|report| format!("{report:?}"))
    }

    fn rejected(source: &str, message: &str) {
        let error = verify_source(source).expect_err("the entry is invalid");
        assert!(error.contains(message), "expected {message:?} in {error}");
    }

    #[test]
    fn well_formed_entries_verify() {
        verify_source(THREE).unwrap();
    }

    #[test]
    fn a_link_name_that_is_not_a_string_is_rejected_on_each_operation() {
        let attrs = "attributes { link_name = #builtin.i32<1> }";
        let message = "link_name must be a non-empty string, but this is #builtin.i32<1>";
        rejected(&format!("builtin.function public extern(\"C\") @f() {attrs};"), message);
        rejected(
            &format!(
                "builtin.module public @m {{ builtin.function public extern(\"C\") @f(); \
                 builtin.function_alias public @a -> @f {attrs}; }};"
            ),
            message,
        );
        rejected(&format!("builtin.global_variable public @g : ptr<i32> {attrs};"), message);
    }

    #[test]
    fn an_empty_link_name_is_rejected_on_each_operation() {
        let attrs = "attributes { link_name = \"\" }";
        let message = "link_name must be a non-empty string, but this is #builtin.string<\"\">";
        rejected(&format!("builtin.function public extern(\"C\") @f() {attrs};"), message);
        rejected(
            &format!(
                "builtin.module public @m {{ builtin.function public extern(\"C\") @f(); \
                 builtin.function_alias public @a -> @f {attrs}; }};"
            ),
            message,
        );
        rejected(&format!("builtin.global_variable public @g : ptr<i32> {attrs};"), message);
    }

    #[test]
    fn a_section_that_is_not_a_non_empty_string_is_rejected_on_a_global() {
        rejected(
            "builtin.global_variable public @g : ptr<i32> attributes { section = \"\" };",
            "section must be a non-empty string, but this is #builtin.string<\"\">",
        );
        rejected(
            "builtin.global_variable public @g : ptr<i32> attributes { section = #builtin.i32<2> \
             };",
            "section must be a non-empty string, but this is #builtin.i32<2>",
        );
    }

    #[test]
    fn a_section_on_a_function_or_an_alias_is_rejected() {
        let message = "only a global variable has a section, not builtin.function";
        rejected(
            "builtin.function public extern(\"C\") @f() attributes { section = \"s\" };",
            message,
        );
        rejected(
            "builtin.module public @m { builtin.function public extern(\"C\") @f(); \
             builtin.function_alias public @a -> @f attributes { section = \"s\" }; };",
            "only a global variable has a section, not builtin.function_alias",
        );
    }

    #[test]
    fn the_entries_survive_the_text_round_trip() {
        let context = Rc::new(Context::default());
        let root = parse(&context, "link.runic", THREE);
        let printed = root.borrow().to_string();
        for entry in [
            "link_name = #builtin.string<\"erlang:+/2\">",
            "link_name = #builtin.string<\"g_name\">",
            "section = #builtin.string<\".data.x\">",
        ] {
            assert!(printed.contains(entry), "{entry} in {printed}");
        }
        let again = parse(&context, "again.runic", &printed);
        assert_eq!(again.borrow().to_string(), printed);
    }

    #[test]
    fn a_cloned_entry_keeps_its_string() {
        let context = Rc::new(Context::default());
        let functions = ops_named(&context, THREE, "builtin.function");
        let entry = functions[1].borrow().get_attribute(LINK_NAME).unwrap();
        let copy = entry.borrow().dyn_clone();
        let copy = copy.borrow();
        assert_eq!(copy.downcast_ref::<StringAttr>().unwrap().as_str(), "erlang:+/2");
    }
}
