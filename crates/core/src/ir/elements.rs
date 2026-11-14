//! What an operation holds: every attribute and every type, at any depth.
//!
//! An operation holds attributes in its properties and its attribute dictionary, and an attribute
//! holds attributes and types in turn (see [AttributeElements](crate::attributes::AttributeElements)).
//! It holds types as the types of its results and of its regions' block arguments, in its
//! attributes, and inside those types. [Operation::walk_attributes] and [Operation::walk_types]
//! visit all of them, each once, with the place in the operation it was reached from.
//!
//! An operand's type is not something the operation holds: it is the type of the value's
//! definition.

use core::fmt;

use smallvec::{SmallVec, smallvec};

use crate::{
    AttributeRef, FxHashSet, Operation, Type, Value, WalkResult, attributes::ElementVisitor,
};

/// Where an element of an operation was found.
///
/// An element held by another element has the site of the outermost one: an attribute nested in
/// a property has that property's site, and a type held by a result type has that result's site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementSite<'a> {
    /// The property of this name, or something it holds.
    Property(&'a str),
    /// The entry of this name in the attribute dictionary, or something it holds.
    Attribute(&'a str),
    /// The type of the result at this index, or a type it holds.
    Result(usize),
    /// The type of a block argument, or a type it holds: argument `index` of block `block` (in
    /// the order of the region's blocks) of region `region`.
    BlockArgument {
        region: usize,
        block: usize,
        index: usize,
    },
}

impl fmt::Display for ElementSite<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Property(name) => write!(f, "property '{name}'"),
            Self::Attribute(name) => write!(f, "attribute '{name}'"),
            Self::Result(index) => write!(f, "result {index}"),
            Self::BlockArgument {
                region,
                block,
                index,
            } => write!(f, "argument {index} of block {block} of region {region}"),
        }
    }
}

/// The types and attributes one attribute holds directly, in the order it lists them.
#[derive(Default)]
struct Held {
    types: SmallVec<[Type; 4]>,
    attributes: SmallVec<[AttributeRef; 4]>,
}

impl ElementVisitor for Held {
    fn visit_type(&mut self, ty: Type) {
        self.types.push(ty);
    }

    fn visit_attribute(&mut self, attr: &AttributeRef) {
        self.attributes.push(*attr);
    }
}

/// What `attr` holds directly, collected so that no borrow of it is held while the walk goes on.
fn held(attr: &AttributeRef) -> Held {
    let mut held = Held::default();
    attr.borrow().visit_elements(&mut held);
    held
}

/// The identity of an attribute: the address of its storage.
fn identity(attr: &AttributeRef) -> usize {
    AttributeRef::as_ptr(attr).cast::<()>().addr()
}

/// Calls `f` with `root` and each attribute it holds at any depth that is not in `visited`, a
/// holder before what it holds (pre-order, in the order each holder lists them), adding each to
/// `visited`. The flag `f` gets is `false` for `root` and `true` for an attribute nested in it.
///
/// [WalkResult::Skip] from `f` leaves out what that attribute holds, unless it is reached
/// another way; [WalkResult::Break] ends the walk with it.
fn walk_attribute_tree(
    root: &AttributeRef,
    visited: &mut FxHashSet<usize>,
    f: &mut dyn FnMut(&AttributeRef, bool) -> WalkResult,
) -> WalkResult {
    let mut stack: SmallVec<[(AttributeRef, bool); 8]> = smallvec![(*root, false)];
    while let Some((attr, nested)) = stack.pop() {
        if !visited.insert(identity(&attr)) {
            continue;
        }
        match f(&attr, nested) {
            WalkResult::Continue(()) => (),
            WalkResult::Skip => continue,
            WalkResult::Break(err) => return WalkResult::Break(err),
        }
        stack.extend(held(&attr).attributes.into_iter().rev().map(|attr| (attr, true)));
    }
    WalkResult::Continue(())
}

/// Calls `f` with `attr` and every attribute it holds, at any depth: each once, a holder before
/// what it holds.
///
/// This is [Operation::walk_attributes] for an attribute that is not held by an operation, such
/// as the attribute of a memory effect.
pub fn walk_nested_attributes(
    attr: &AttributeRef,
    f: &mut dyn FnMut(&AttributeRef) -> WalkResult,
) -> WalkResult {
    walk_attribute_tree(attr, &mut FxHashSet::default(), &mut |attr, _| f(attr))
}

/// Calls `f` with `root` and each type it holds at any depth that is not in `visited`, a holder
/// before what it holds, adding each to `visited`, which is what ends the walk of a recursive
/// type. [WalkResult::Skip] leaves out what that type holds, unless it is reached another way.
fn walk_type_tree(
    site: ElementSite<'_>,
    root: Type,
    visited: &mut FxHashSet<Type>,
    f: &mut dyn FnMut(ElementSite<'_>, Type) -> WalkResult,
) -> WalkResult {
    let mut stack: SmallVec<[Type; 8]> = smallvec![root];
    while let Some(ty) = stack.pop() {
        if !visited.insert(ty) {
            continue;
        }
        match f(site, ty) {
            WalkResult::Continue(()) => (),
            WalkResult::Skip => continue,
            WalkResult::Break(err) => return WalkResult::Break(err),
        }
        let mut held = SmallVec::<[Type; 4]>::new();
        ty.visit_subtypes(&mut |held_ty| held.push(held_ty));
        stack.extend(held.into_iter().rev());
    }
    WalkResult::Continue(())
}

impl Operation {
    /// Every attribute this operation holds, at any depth: its properties and its attribute
    /// dictionary, and every attribute those hold. Each attribute once, a holder before what it
    /// holds.
    ///
    /// The properties come first, in their declared order, then the dictionary's entries. Each
    /// attribute is reported with the site of the property or entry it was first reached from;
    /// an attribute held in two places (a marker shared by two parameters of a signature, say) is
    /// reported at the first. Attributes are told apart by identity, not by value.
    ///
    /// [WalkResult::Skip] from `f` leaves out what that attribute holds (unless it is reached
    /// another way); [WalkResult::Break] ends the walk and is returned.
    pub fn walk_attributes(
        &self,
        f: &mut dyn FnMut(ElementSite<'_>, &AttributeRef) -> WalkResult,
    ) -> WalkResult {
        self.walk_attributes_and_depth(&mut |site, attr, _| f(site, attr))
    }

    /// [Self::walk_attributes], telling `f` whether the attribute is nested in the property or
    /// entry its site names (`true`) or is that property or entry itself (`false`).
    pub(crate) fn walk_attributes_and_depth(
        &self,
        f: &mut dyn FnMut(ElementSite<'_>, &AttributeRef, bool) -> WalkResult,
    ) -> WalkResult {
        let mut visited = FxHashSet::default();
        for property in self.properties() {
            let site = ElementSite::Property(property.name.as_str());
            walk_attribute_tree(&property.value, &mut visited, &mut |attr, nested| {
                f(site, attr, nested)
            })?;
        }
        for entry in self.attributes() {
            let entry = entry.as_named_attribute();
            let site = ElementSite::Attribute(entry.name.as_str());
            walk_attribute_tree(&entry.value, &mut visited, &mut |attr, nested| {
                f(site, attr, nested)
            })?;
        }
        WalkResult::Continue(())
    }

    /// Every type this operation holds, at any depth: result types, the argument types of its
    /// regions' blocks, each attribute's own type and every type held in an attribute value, and
    /// every type those types hold. Each type once; terminates on recursive types.
    ///
    /// The order is the results, then the block arguments (region by region, block by block),
    /// then the attributes in the order of [Self::walk_attributes], each attribute's types in the
    /// order it lists them (see [Attribute::visit_elements](crate::Attribute::visit_elements));
    /// each type comes before the types it holds. A type is reported with the site it was first
    /// reached from, and a type held by a type with the site of that holder. Operand types are not
    /// visited: they belong to the values' definitions. The operations nested in the regions are
    /// not visited either.
    ///
    /// [WalkResult::Skip] from `f` leaves out the types that type holds (unless they are reached
    /// another way); [WalkResult::Break] ends the walk and is returned.
    pub fn walk_types(&self, f: &mut dyn FnMut(ElementSite<'_>, Type) -> WalkResult) -> WalkResult {
        let mut visited = FxHashSet::default();
        for (index, result) in self.results().iter().enumerate() {
            let ty = result.borrow().ty();
            walk_type_tree(ElementSite::Result(index), ty, &mut visited, f)?;
        }
        for (region, body) in self.regions().iter().enumerate() {
            for (block, body) in body.body().iter().enumerate() {
                for (index, arg) in body.arguments().iter().enumerate() {
                    let site = ElementSite::BlockArgument {
                        region,
                        block,
                        index,
                    };
                    let ty = arg.borrow().ty();
                    walk_type_tree(site, ty, &mut visited, f)?;
                }
            }
        }
        self.walk_attributes(&mut |site, attr| {
            for ty in held(attr).types {
                walk_type_tree(site, ty, &mut visited, f)?;
            }
            WalkResult::Continue(())
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, rc::Rc, string::String, vec, vec::Vec};

    use super::*;
    use crate::{
        AsSymbolRef, BuilderExt, CallConv, Context, Int, Op, SourceSpan, StructType, Symbol as _,
        SymbolUse, Usable,
        attributes::AttrList,
        dialects::{
            builtin::attributes::{
                AbiParam, IntAttr, List, ListAttr, Signature, SignatureAttr, SymbolRef,
                SymbolRefAttr, TypeAttr, ZextAttr,
            },
            test::Load,
        },
        interner::Symbol,
        testing::{Test, label_messages},
    };

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

    /// The labels of the error `op` fails to verify with, on one line.
    fn verify_error(op: &Operation) -> String {
        let err = op.verify().expect_err("the operation must not verify");
        label_messages(err.as_diagnostic()).join(" | ")
    }

    /// A test function `@owner` in a module, which also defines `@g`.
    fn owner_and_g() -> (
        Test,
        crate::dialects::builtin::FunctionRef,
        crate::dialects::builtin::FunctionRef,
    ) {
        let mut test = Test::named("elements").in_module("test");
        let owner = test.define_function("owner", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        (test, owner, g)
    }

    #[test]
    fn a_symbol_reference_nested_in_a_list_is_verified() {
        let (test, mut owner, _) = owner_and_g();
        let context = test.context_rc();
        let path = crate::SymbolPath::new([crate::SymbolNameComponent::Leaf(
            crate::SymbolName::intern("missing"),
        )])
        .unwrap();
        let reference = context
            .create_attribute::<SymbolRefAttr, _>(SymbolRef::new(path, None))
            .as_attribute_ref();
        let entries = list(&context, [reference]);
        owner.borrow_mut().as_operation_mut().set_attribute("entries", entries);
        let owner = owner.borrow();
        let err = owner.as_operation().verify().expect_err("a nested reference is verified");
        let message = format!("{err}");
        assert!(message.contains("symbol '@missing' does not resolve"), "{message}");
    }

    #[test]
    fn an_integer_constant_nested_in_a_list_is_verified() {
        let (test, mut owner, _) = owner_and_g();
        let context = test.context_rc();
        let constant = context
            .create_attribute_with_type::<IntAttr, _>(Int::from(-1i32), context.i8())
            .as_attribute_ref();
        let entries = list(&context, [constant]);
        owner.borrow_mut().as_operation_mut().set_attribute("entries", entries);
        let message = verify_error(owner.borrow().as_operation());
        assert!(
            message.contains(
                "an attribute held by attribute 'entries' of 'builtin.function' is an integer \
                 constant of type `i8` holding a 32-bit value, but a constant of that type is 8 \
                 bits wide"
            ),
            "{message}"
        );
    }

    #[test]
    fn a_signature_nested_in_a_list_is_verified() {
        let (test, mut owner, _) = owner_and_g();
        let context = test.context_rc();
        let signature = Signature::from_parts(
            CallConv::default(),
            vec![AbiParam::new(context.never_type())],
            vec![],
        );
        let signature = context.create_attribute::<SignatureAttr, _>(signature).as_attribute_ref();
        let entries = list(&context, [signature]);
        owner.borrow_mut().as_operation_mut().set_attribute("entries", entries);
        let message = verify_error(owner.borrow().as_operation());
        assert!(
            message.contains(
                "an attribute held by attribute 'entries' of 'builtin.function' is an invalid \
                 signature: parameter 0 has type `never`"
            ),
            "{message}"
        );
    }

    /// A reference nested in a list is a use of its symbol: erasing the operation holding it
    /// takes it out of the symbol's use list.
    #[test]
    fn erasing_an_operation_unlinks_a_nested_symbol_reference() {
        let (test, mut owner, g) = owner_and_g();
        let context = test.context_rc();
        let path = g.borrow().path();
        let mut reference =
            context.create_attribute::<SymbolRefAttr, _>(SymbolRef::new(path, None));
        let user = context.alloc_tracked(SymbolUse::new(owner.as_operation_ref(), reference));
        reference.borrow_mut().set_user(user);
        reference.borrow_mut().link(g.as_symbol_ref());
        let entries = list(&context, [reference.as_attribute_ref()]);
        owner.borrow_mut().as_operation_mut().set_attribute("entries", entries);

        owner
            .borrow()
            .as_operation()
            .verify()
            .expect("a linked nested reference verifies");
        let mut uses = 0;
        let _ = owner.borrow().as_operation().walk_symbol_uses(|_| {
            uses += 1;
            WalkResult::Continue(())
        });
        assert_eq!(uses, 1);
        assert_eq!(g.borrow().iter_uses().count(), 1);

        owner.borrow_mut().as_operation_mut().erase();
        assert_eq!(g.borrow().iter_uses().count(), 0);
        assert!(!user.is_linked());
    }

    /// Each attribute once, a holder before what it holds; a marker shared by two parameters is
    /// reported at its first place.
    #[test]
    fn attributes_are_walked_once_each_holder_first() {
        let (test, mut owner, _) = owner_and_g();
        let context = test.context_rc();
        let zext = context.get_marker::<ZextAttr>().as_attribute_ref();
        let shared_zext = |ty| {
            let mut param = AbiParam::new(ty);
            param.set_attribute(crate::interner::symbols::Extend, zext);
            param
        };
        let signature = Signature::from_parts(
            CallConv::default(),
            vec![shared_zext(context.i8()), shared_zext(context.i16())],
            vec![],
        );
        let signature = context.create_attribute::<SignatureAttr, _>(signature).as_attribute_ref();
        let ty = context.create_attribute::<TypeAttr, _>(context.i64()).as_attribute_ref();
        let entries = list(&context, [signature, ty]);
        // the list's entries are its own
        let (signature, ty) = {
            let held = held(&entries).attributes;
            (held[0], held[1])
        };
        owner.borrow_mut().as_operation_mut().set_attribute("entries", entries);

        let owner = owner.borrow();
        let mut seen = Vec::new();
        let _ = owner.as_operation().walk_attributes(&mut |site, attr| {
            if let ElementSite::Attribute(name) = site {
                seen.push((Symbol::intern(name), *attr));
            }
            WalkResult::Continue(())
        });
        let entries_name = Symbol::intern("entries");
        let expected = [entries, signature, zext, ty];
        assert_eq!(seen.len(), expected.len(), "{seen:?}");
        for ((name, attr), expected) in seen.iter().zip(expected) {
            assert_eq!(*name, entries_name);
            assert!(AttributeRef::ptr_eq(attr, &expected), "{attr:?} is not {expected:?}");
        }

        // Skip leaves out what the list holds
        let mut count = 0;
        let _ = owner.as_operation().walk_attributes(&mut |site, attr| {
            if matches!(site, ElementSite::Attribute(_)) {
                count += 1;
                if AttributeRef::ptr_eq(attr, &entries) {
                    return WalkResult::Skip;
                }
            }
            WalkResult::Continue(())
        });
        assert_eq!(count, 1);
    }

    /// The result type is a member of a two-member recursion group: each member, and each type
    /// they hold, is visited once.
    #[test]
    fn types_are_walked_once_and_a_recursive_type_ends() {
        let mut test = Test::named("rec");
        let context = test.context_rc();
        let (a, b) = {
            let (var0, var1) = (context.rec_var(0), context.rec_var(1));
            let bodies = [
                context.struct_type(StructType::named(
                    Symbol::intern("a"),
                    [context.ptr_type(var1).into()],
                )),
                context.struct_type(StructType::named(
                    Symbol::intern("b"),
                    [context.ptr_type(var0).into()],
                )),
            ];
            let members = context.rec_group(&bodies).expect("a valid group");
            (members[0], members[1])
        };
        let ptr_a = context.ptr_type(a);
        test.with_function("rec", &[ptr_a], &[]);
        let load = {
            let addr = test.entry_block().borrow().arguments()[0] as crate::ValueRef;
            let mut builder = test.function_builder();
            builder.builder_mut().create::<Load, _>(SourceSpan::UNKNOWN)(addr).unwrap()
        };
        let load = load.borrow();
        let op = load.as_operation();
        assert_eq!(op.results()[0].borrow().ty(), a);

        let mut seen = Vec::new();
        let walked = op.walk_types(&mut |site, ty| {
            seen.push((format!("{site}"), ty));
            WalkResult::Continue(())
        });
        assert!(walked.should_continue());
        let ptr_b = context.ptr_type(b);
        let expected = [a, ptr_b, b, ptr_a];
        assert_eq!(seen.len(), expected.len(), "{seen:?}");
        for ((site, ty), expected) in seen.into_iter().zip(expected) {
            assert_eq!(site, "result 0");
            assert_eq!(ty, expected);
        }
    }

    /// Block arguments, then the types in attributes, each type once across the operation: the
    /// parameter type the signature also holds is reported at the block argument.
    #[test]
    fn block_argument_and_attribute_types_are_walked() {
        let test = Test::new("f", |cx| ([cx.ptr_type(cx.i32())], []));
        let context = test.context_rc();
        let mut function = test.function();
        let ty = context.create_attribute::<TypeAttr, _>(context.ptr_type(context.i8()));
        function
            .borrow_mut()
            .as_operation_mut()
            .set_attribute("ty", ty.as_attribute_ref());
        let function = function.borrow();
        let mut seen = Vec::new();
        let _ = function.as_operation().walk_types(&mut |site, ty| {
            seen.push((format!("{site}"), format!("{ty}")));
            WalkResult::Continue(())
        });
        let argument = "argument 0 of block 0 of region 0";
        assert_eq!(
            seen,
            [
                (argument.into(), "ptr<i32>".into()),
                (argument.into(), "i32".into()),
                ("attribute 'ty'".into(), "ptr<i8>".into()),
                ("attribute 'ty'".into(), "i8".into()),
            ] as [(String, String); 4]
        );
    }
}
