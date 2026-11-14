//! What an operation holds: every attribute and every type, at any depth.
//!
//! An operation holds attributes in its properties and its attribute dictionary, and an attribute
//! holds attributes and types in turn (see
//! [AttributeElements](crate::attributes::AttributeElements)). It holds types as the types of its
//! results and of its regions' block arguments, in its attributes, and inside those types.
//! [Operation::walk_attributes] and [Operation::walk_types] visit all of them, each once, with the
//! place in the operation it was reached from.
//!
//! An operand's type is not something the operation holds: it is the type of the value's
//! definition.

use alloc::{boxed::Box, format, rc::Rc, vec::Vec};
use core::fmt;

use smallvec::{SmallVec, smallvec};

use crate::{
    AttributeRef, Context, FxHashMap, FxHashSet, Int, Operation, OperationRef, Report, Spanned,
    Type, TypeMapper, Value, WalkResult,
    attributes::{ElementReplacer, ElementVisitor},
    dialects::builtin::attributes::IntAttr,
    interner::Symbol,
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

/// The elements a walk has seen: an inline list searched in order while there are at most eight,
/// which most operations never outgrow, so their walks allocate nothing; a hash set after that.
struct Visited<T> {
    small: SmallVec<[T; 8]>,
    large: Option<FxHashSet<T>>,
}

impl<T> Default for Visited<T> {
    fn default() -> Self {
        Self {
            small: SmallVec::new(),
            large: None,
        }
    }
}

impl<T: Copy + Eq + core::hash::Hash> Visited<T> {
    /// Adds `value`, answering whether it was not there yet.
    fn insert(&mut self, value: T) -> bool {
        if let Some(large) = self.large.as_mut() {
            return large.insert(value);
        }
        if self.small.contains(&value) {
            return false;
        }
        if self.small.len() == self.small.inline_size() {
            let mut large: FxHashSet<T> = self.small.iter().copied().collect();
            large.insert(value);
            self.large = Some(large);
        } else {
            self.small.push(value);
        }
        true
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
    visited: &mut Visited<usize>,
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
    walk_attribute_tree(attr, &mut Visited::default(), &mut |attr, _| f(attr))
}

/// Calls `f` with `root` and each type it holds at any depth that is not in `visited`, a holder
/// before what it holds, adding each to `visited`, which is what ends the walk of a recursive
/// type. [WalkResult::Skip] leaves out what that type holds, unless it is reached another way.
fn walk_type_tree(
    site: ElementSite<'_>,
    root: Type,
    visited: &mut Visited<Type>,
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
    ///
    /// The key of a keyed successor (a case value of `cf.switch`) is an attribute the operation
    /// holds outside its properties and dictionary: this walk does not reach it, and neither do
    /// attribute verification and [replace_elements_in].
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
        let mut visited = Visited::default();
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
    ///
    /// The types held by the key of a keyed successor are not visited: see
    /// [Self::walk_attributes].
    pub fn walk_types(&self, f: &mut dyn FnMut(ElementSite<'_>, Type) -> WalkResult) -> WalkResult {
        let mut visited = Visited::default();
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

/// The function an [ElementRewrite] carries an integer constant to a new type with.
type IntConstantRewrite<'a> = Box<dyn FnMut(&Int, Type, Type) -> Result<Int, Report> + 'a>;

/// How [replace_elements_in] rewrites the types an operation holds: a [TypeMapper], applied to
/// every type, and optionally a function that carries an integer constant to the type the map
/// gives its type.
///
/// One rewrite may be used for many operations: its mapper remembers the result for each type it
/// has mapped, so each type is mapped once however many operations hold it.
pub struct ElementRewrite<'a> {
    mapper: TypeMapper<'a>,
    int_constants: Option<IntConstantRewrite<'a>>,
}

impl<'a> ElementRewrite<'a> {
    /// A rewrite that maps types with `mapper` and has no rewrite of integer constants: a
    /// constant whose type the map changes is an error.
    pub fn types(mapper: TypeMapper<'a>) -> Self {
        Self {
            mapper,
            int_constants: None,
        }
    }

    /// Called for an integer constant whose type changes: the constant to use at the new type, or
    /// an error. Without it, a constant whose type changes is an error.
    ///
    /// `f` is given the constant's value, its type and the type the map gives that type. It is not
    /// called for a constant whose type the map leaves as it is. The value it answers must be as
    /// wide as a constant of the new type ([Int::width_for_type]); an answer of another width is an
    /// error, and an error from `f` is the error of the rewrite, naming the operation and where
    /// the constant is.
    pub fn with_int_constants(
        mut self,
        f: impl FnMut(&Int, Type, Type) -> Result<Int, Report> + 'a,
    ) -> Self {
        self.int_constants = Some(Box::new(f));
        self
    }
}

/// The [ElementReplacer] that computes the new elements of one operation.
///
/// The replacer cannot fail, and the type map and the rewrite of integer constants can: the first
/// error is kept in `error`, and from then on every element is answered unchanged.
struct OperationReplacer<'r, 'a> {
    rewrite: &'r mut ElementRewrite<'a>,
    op: &'r Operation,
    context: Rc<Context>,
    /// The property or dictionary entry being rewritten (or the result or block argument whose
    /// type is), for errors.
    site: ElementSite<'static>,
    /// How many attributes are being replaced around the current element: 0 at the site itself
    /// (the attribute there, or the type of a result or block argument), 1 inside the attribute at
    /// the site, and so on.
    depth: usize,
    /// The answer for each attribute already asked about, by identity, so an attribute held twice
    /// by the operation is replaced by one new attribute.
    answers: FxHashMap<usize, Option<AttributeRef>>,
    error: Option<Report>,
}

impl OperationReplacer<'_, '_> {
    /// The attribute being replaced as an error names it: the one at the site, or one held by
    /// it.
    fn role(&self) -> alloc::string::String {
        if self.depth > 0 {
            format!("an attribute held by {}", self.site)
        } else {
            format!("{}", self.site)
        }
    }

    /// The type being replaced as an error names it: the type of the result or block argument at
    /// the site, or a type held by the attribute at the site, or by one it holds.
    fn type_role(&self) -> alloc::string::String {
        match self.depth {
            0 => format!("the type of {}", self.site),
            1 => format!("a type held by {}", self.site),
            _ => format!("a type held by an attribute held by {}", self.site),
        }
    }

    fn fail(&mut self, label: alloc::string::String) {
        self.error = Some(crate::diagnostics::report!(
            message: "cannot rewrite the types of an operation",
            labels: [primary(self.op.span(), label)],
        ));
    }

    /// The replacement of the integer constant `value : ty`, when the map changes `ty`.
    fn replace_int_constant(&mut self, value: Int, ty: Type) -> Option<AttributeRef> {
        if ty.is_unknown() {
            return None;
        }
        let new_ty = self.replace_type(ty);
        if new_ty == ty || self.error.is_some() {
            return None;
        }
        let Some(rewrite) = self.rewrite.int_constants.as_mut() else {
            let label = format!(
                "{} of '{}' is the integer constant `{value} : {ty}`, which the type map changes \
                 to type `{new_ty}`, and no rewrite of integer constants was given",
                self.role(),
                self.op.name()
            );
            self.fail(label);
            return None;
        };
        let new_value = match rewrite(&value, ty, new_ty) {
            Ok(new_value) => new_value,
            Err(err) => {
                let label = format!(
                    "{} of '{}' is the integer constant `{value} : {ty}`, which the rewrite of \
                     integer constants could not carry to type `{new_ty}`: {err}",
                    self.role(),
                    self.op.name()
                );
                self.fail(label);
                return None;
            }
        };
        let width = new_value.width();
        let problem = match Int::width_for_type(new_ty) {
            Some(expected) if expected == width => {
                return Some(self.context.int_attr(new_value, new_ty).as_attribute_ref());
            }
            Some(expected) => format!("a constant of that type is {expected} bits wide"),
            None => "no integer constant has that type".into(),
        };
        let label = format!(
            "{} of '{}' is the integer constant `{value} : {ty}`, which the rewrite of integer \
             constants gave the {width}-bit value {new_value} at type `{new_ty}`, but {problem}",
            self.role(),
            self.op.name()
        );
        self.fail(label);
        None
    }
}

impl ElementReplacer for OperationReplacer<'_, '_> {
    fn replace_type(&mut self, ty: Type) -> Type {
        if self.error.is_some() {
            return ty;
        }
        match self.rewrite.mapper.map(ty) {
            Ok(new_ty) => new_ty,
            Err(err) => {
                let label =
                    format!("{} of '{}' cannot be mapped: {err}", self.type_role(), self.op.name());
                self.fail(label);
                ty
            }
        }
    }

    fn replace_attribute(&mut self, attr: &AttributeRef) -> Option<AttributeRef> {
        if self.error.is_some() {
            return None;
        }
        if let Some(answer) = self.answers.get(&identity(attr)) {
            return *answer;
        }
        let constant = attr
            .borrow()
            .downcast_ref::<IntAttr>()
            .map(|constant| (*constant.as_value(), crate::Attribute::ty(constant)));
        let answer = match constant {
            Some((value, ty)) => self.replace_int_constant(value, ty),
            None => {
                self.depth += 1;
                let answer = attr.borrow().replace_elements(self);
                self.depth -= 1;
                answer
            }
        };
        let answer = if self.error.is_some() { None } else { answer };
        self.answers.insert(identity(attr), answer);
        answer
    }
}

/// The new elements of an operation, computed before any is installed.
#[derive(Default)]
struct Replacements {
    results: SmallVec<[(crate::OpResultRef, Type); 2]>,
    arguments: SmallVec<[(crate::BlockArgumentRef, Type); 4]>,
    properties: SmallVec<[(Symbol, AttributeRef); 2]>,
    attributes: SmallVec<[(Symbol, AttributeRef); 2]>,
}

impl Replacements {
    fn is_empty(&self) -> bool {
        self.results.is_empty()
            && self.arguments.is_empty()
            && self.properties.is_empty()
            && self.attributes.is_empty()
    }
}

/// Everything of `op` that `rewrite` changes, or the first error.
fn replacements(op: &Operation, rewrite: &mut ElementRewrite<'_>) -> Result<Replacements, Report> {
    let mut replacer = OperationReplacer {
        rewrite,
        op,
        context: op.context_rc(),
        site: ElementSite::Result(0),
        depth: 0,
        answers: FxHashMap::default(),
        error: None,
    };
    let mut replacements = Replacements::default();
    for (index, result) in op.results().iter().enumerate() {
        replacer.site = ElementSite::Result(index);
        let ty = result.borrow().ty();
        let new_ty = replacer.replace_type(ty);
        if new_ty != ty {
            replacements.results.push((*result, new_ty));
        }
    }
    for (region, body) in op.regions().iter().enumerate() {
        for (block, body) in body.body().iter().enumerate() {
            for (index, arg) in body.arguments().iter().enumerate() {
                replacer.site = ElementSite::BlockArgument {
                    region,
                    block,
                    index,
                };
                let ty = arg.borrow().ty();
                let new_ty = replacer.replace_type(ty);
                if new_ty != ty {
                    replacements.arguments.push((*arg, new_ty));
                }
            }
        }
    }
    for property in op.properties() {
        replacer.site = ElementSite::Property(property.name.as_str());
        if let Some(new) = replacer.replace_attribute(&property.value) {
            replacements.properties.push((property.name, new));
        }
    }
    for entry in op.attributes() {
        let entry = entry.as_named_attribute();
        replacer.site = ElementSite::Attribute(entry.name.as_str());
        if let Some(new) = replacer.replace_attribute(&entry.value) {
            replacements.attributes.push((entry.name, new));
        }
    }
    match replacer.error {
        Some(err) => Err(err),
        None => Ok(replacements),
    }
}

/// Rewrites the operation's properties, attribute dictionary, result types and the argument
/// types of its regions' blocks. Operand types are not touched. Returns whether anything changed.
///
/// Every type the operation holds is mapped by `rewrite`'s [TypeMapper]: a result's type and a
/// block argument's type are set in place on the value, and an attribute that holds a type the
/// map changes, at any depth, is replaced by a new attribute (see
/// [Attribute::replace_elements](crate::Attribute::replace_elements)), installed with
/// [Operation::set_property] or [Operation::set_attribute]. The old attribute is not edited, so
/// another holder of it keeps it as it was. An attribute whose elements the map does not change
/// keeps its handle where it is held directly by the operation or by an attribute that is kept,
/// and the identity map changes nothing and builds no attribute. The entries of a rebuilt list
/// are copies, changed or not; a symbol reference copied this way is given a use of its own,
/// linked to the symbol the original was linked to, and the original's use is unlinked once the
/// operation no longer holds the original. An attribute held twice by the operation is replaced by
/// one new attribute. Because the result type and a type the operation keeps beside it in a
/// property (a hidden `ty`, say) are mapped by one map, they change together.
///
/// What an operation ties to the shape of a type is not revalued: under a map that changes the
/// fields of an aggregate, the index paths of `mem.gep` and `mem.offsetof` keep their numbers and
/// may name other fields. Such a map needs a pattern for the operations that index the aggregate.
/// The key of a keyed successor (a case value of `cf.switch`) is not rewritten either: it is not
/// among the attributes [Operation::walk_attributes] reaches.
///
/// An integer constant ([IntAttr]) whose type the map changes is given its new value by the
/// rewrite of integer constants ([ElementRewrite::with_int_constants]); without one, it is an
/// error naming the operation and where the constant is. The operations nested in the regions
/// are not rewritten (see [recursively_replace_elements_in]).
///
/// Every replacement is computed before any is installed: on an error of the map or of the
/// rewrite of integer constants, nothing of the operation is changed. The properties and
/// dictionary entries are installed before the result and block-argument types are set; a
/// property that refuses its replacement is an internal error (a replacement is of the kind of
/// the attribute it replaces), after which the operation may be partly rewritten. Whether the
/// rewritten operation is still valid (an `arith.zext` whose two types became equal, say) is its
/// verifier's business.
pub fn replace_elements_in(
    op: &mut Operation,
    rewrite: &mut ElementRewrite<'_>,
) -> Result<bool, Report> {
    let replacements = replacements(op, rewrite)?;
    if replacements.is_empty() {
        return Ok(false);
    }
    for (name, new) in replacements.properties {
        give_copied_references_their_own_uses(op, &new);
        // A replacement is of the kind of the attribute it replaces, which the property holds
        if let Err(err) = op.set_property(name, new) {
            return Err(Report::msg(format!(
                "internal error: the replacement of property '{name}' of '{}' is not of the \
                 property's kind, and the operation may be partly rewritten: {err}",
                op.name()
            )));
        }
    }
    for (name, new) in replacements.attributes {
        give_copied_references_their_own_uses(op, &new);
        op.set_attribute(name, new);
    }
    for (mut result, ty) in replacements.results {
        result.borrow_mut().set_type(ty);
    }
    for (mut arg, ty) in replacements.arguments {
        arg.borrow_mut().set_type(ty);
    }
    Ok(true)
}

/// Gives each symbol reference `attr` holds, at any depth, whose use names another attribute (a
/// copy, made when a list holding the reference was rebuilt) a use of its own, owned by `op` and
/// linked to the symbol the shared use is linked to, if any. The shared use stays with the
/// reference it names.
pub(crate) fn give_copied_references_their_own_uses(op: &mut Operation, attr: &AttributeRef) {
    use crate::{
        SymbolUse, UnsafeIntrusiveEntityRef, dialects::builtin::attributes::SymbolRefAttr,
    };

    let mut copies = SmallVec::<[(UnsafeIntrusiveEntityRef<SymbolRefAttr>, _); 2]>::new();
    let _ = walk_nested_attributes(attr, &mut |attr| {
        if let Ok(reference) = attr.try_downcast_attr::<SymbolRefAttr>()
            && let Some(shared) = reference.borrow().try_user()
            && shared.borrow().attr != reference
        {
            copies.push((reference, shared));
        }
        WalkResult::Continue(())
    });
    if copies.is_empty() {
        return;
    }
    let owner = op.as_operation_ref();
    let context = op.context_rc();
    for (mut reference, shared) in copies {
        let own = context.alloc_tracked(SymbolUse::new(owner, reference));
        reference.borrow_mut().replace_shared_user(own);
        if let Some(symbol) = shared.linked_symbol() {
            op.link_symbol_use(own, symbol);
        }
    }
}

/// The same for the operation and everything nested in it.
///
/// [replace_elements_in] is applied to `op` and then to each operation nested in it, in
/// pre-order. On an error the operation that failed is unchanged, and so are the operations after
/// it, but the operations rewritten before it keep their changes.
pub fn recursively_replace_elements_in(
    op: OperationRef,
    rewrite: &mut ElementRewrite<'_>,
) -> Result<bool, Report> {
    let mut ops = Vec::new();
    op.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
    let mut changed = false;
    for mut op in ops {
        changed |= replace_elements_in(&mut op.borrow_mut(), rewrite)?;
    }
    Ok(changed)
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
        // the entries as the list holds them: a list links the attributes it is given (only a
        // replaced list holds copies)
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

#[cfg(test)]
mod rewrite_tests {
    use alloc::{format, vec::Vec};

    use super::*;
    use crate::{
        Context, Op, StructType,
        dialects::{
            builtin::{
                Module,
                attributes::{TypeArrayAttr, TypeAttr},
            },
            test::Constant,
        },
        testing::{Test, text::parse},
    };

    /// `i64` to `i32`, everything else as it is.
    fn i64_to_i32(context: &Context) -> TypeMapper<'_> {
        TypeMapper::new(context, |context, ty| {
            if ty == context.i64() {
                context.i32()
            } else {
                ty
            }
        })
    }

    /// Truncates a constant to the width of its new type.
    fn truncate(value: &Int, _old: Type, new: Type) -> Result<Int, Report> {
        Ok(value.trunc(Int::width_for_type(new).expect("an integer type")))
    }

    const MODULE: &str = r#"builtin.module public @m {
    builtin.global_variable public @var : i64;
    builtin.function private extern("C") @callee(%x: i64) -> i64 locals(i64) {
        builtin.store_local 0, %x;
        %y = builtin.load_local 0 : i64;
        builtin.ret %y : (i64);
    };
    builtin.function public extern("C") @caller(%a: i64) -> i64 {
        %c = test.constant 4294967297 : i64;
        %r = builtin.call @callee(%c) : extern("C") (i64) -> i64;
        %f = builtin.function_ref @callee : (i64) -> i64;
        %t = builtin.call_indirect %f(%r) : extern("C") (i64) -> i64;
        %p = builtin.global_symbol @var : ptr<i64>;
        builtin.ret_imm -1 : i64;
    };
};"#;

    /// Every operation in `root`, in pre-order.
    fn operations(root: OperationRef) -> Vec<OperationRef> {
        let mut ops = Vec::new();
        root.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
        ops
    }

    /// Every attribute every operation in `root` holds, at any depth.
    fn all_attributes(root: OperationRef) -> Vec<AttributeRef> {
        let mut attrs = Vec::new();
        for op in operations(root) {
            let _ = op.borrow().walk_attributes(&mut |_, attr| {
                attrs.push(*attr);
                WalkResult::Continue(())
            });
        }
        attrs
    }

    /// The first operation named `name` in `root`.
    fn find(root: OperationRef, name: &str) -> OperationRef {
        operations(root)
            .into_iter()
            .find(|op| format!("{}", op.borrow().name()) == name)
            .unwrap_or_else(|| panic!("no '{name}'"))
    }

    /// The type a `TypeAttr` property holds.
    fn type_property(op: OperationRef, name: &str) -> Type {
        *op.borrow()
            .get_property(Symbol::intern(name))
            .downcast_ref::<TypeAttr>()
            .unwrap()
            .as_value()
    }

    #[test]
    fn a_whole_module_is_rewritten_from_i64_to_i32() {
        let (context, module) = parse::<Module>(MODULE).expect("the module parses");
        let root = module.as_operation_ref();
        root.borrow().recursively_verify().expect("the module verifies");

        let mut rewrite = ElementRewrite::types(i64_to_i32(&context)).with_int_constants(truncate);
        let changed = recursively_replace_elements_in(root, &mut rewrite).expect("the rewrite");
        assert!(changed);
        drop(rewrite);

        if let Err(err) = root.borrow().recursively_verify() {
            panic!("the rewritten module does not verify: {err:?}");
        }
        let printed = format!("{}", *root.borrow());
        let expected = MODULE.replace("i64", "i32").replace("4294967297", "1");
        let (_expected_context, expected) =
            parse::<Module>(&expected).expect("the expected module parses");
        let expected = format!("{}", *expected.as_operation_ref().borrow());
        assert_eq!(printed, expected);
        assert!(!printed.contains("i64"), "{printed}");

        // the hidden `ty` of `builtin.function_ref` changed with its result
        let function_ref = find(root, "builtin.function_ref");
        let fn_ty = type_property(function_ref, "ty");
        assert_eq!(fn_ty, function_ref.borrow().results()[0].borrow().ty());
        assert_eq!(format!("{fn_ty}"), "(i32) -> i32");
        // and the global's declared type
        assert_eq!(type_property(find(root, "builtin.global_variable"), "ty"), context.i32());
        // `locals`
        let callee = find(root, "builtin.function");
        let callee = callee.borrow();
        let locals = callee.get_property(Symbol::intern("locals"));
        let locals = locals.downcast_ref::<TypeArrayAttr>().unwrap();
        assert_eq!(locals.as_value().iter().copied().collect::<Vec<_>>(), [context.i32()]);
    }

    /// The identity map changes nothing and builds no attribute.
    #[test]
    fn the_identity_changes_nothing() {
        let (context, module) = parse::<Module>(MODULE).expect("the module parses");
        let root = module.as_operation_ref();
        let before = all_attributes(root);
        let printed = format!("{}", *root.borrow());

        let mut rewrite = ElementRewrite::types(TypeMapper::new(&context, |_, ty| ty))
            .with_int_constants(|_, _, _| panic!("no constant changes type"));
        let changed = recursively_replace_elements_in(root, &mut rewrite).expect("the rewrite");
        assert!(!changed);
        drop(rewrite);

        let after = all_attributes(root);
        assert_eq!(before.len(), after.len());
        for (before, after) in before.iter().zip(&after) {
            assert!(AttributeRef::ptr_eq(before, after), "{before:?} was replaced by {after:?}");
        }
        assert_eq!(format!("{}", *root.borrow()), printed);
        root.borrow().recursively_verify().expect("the module still verifies");
    }

    /// Without a rewrite of integer constants, a constant whose type changes is an error naming
    /// the operation and the property, and the operation is left as it was: its result type too,
    /// although results are mapped first.
    #[test]
    fn a_constant_whose_type_changes_needs_the_hook() {
        let (context, module) = parse::<Module>(MODULE).expect("the module parses");
        let root = module.as_operation_ref();
        let mut constant = find(root, "test.constant");
        let value_before = constant.borrow().get_attribute(Symbol::intern("value")).unwrap();

        let mut rewrite = ElementRewrite::types(i64_to_i32(&context));
        let err = replace_elements_in(&mut constant.borrow_mut(), &mut rewrite)
            .expect_err("the constant cannot be retyped");
        let message = crate::testing::label_messages(err.as_diagnostic()).join(" | ");
        assert!(
            message.contains(
                "property 'value' of 'test.constant' is the integer constant `4294967297 : i64`, \
                 which the type map changes to type `i32`, and no rewrite of integer constants \
                 was given"
            ),
            "{message}"
        );
        let constant = constant.borrow();
        assert_eq!(constant.results()[0].borrow().ty(), context.i64());
        let value_after = constant.get_attribute(Symbol::intern("value")).unwrap();
        assert!(AttributeRef::ptr_eq(&value_before, &value_after));
        let typed = constant.downcast_ref::<Constant>().unwrap();
        assert_eq!(crate::Attribute::ty(&*typed.value()), context.i64());
        constant.verify().expect("the constant is as it was");
    }

    /// An error from the rewrite of integer constants names the operation and where the
    /// constant is, and keeps the hook's message.
    #[test]
    fn an_error_from_the_constant_hook_names_the_operation_and_site() {
        let (context, module) = parse::<Module>(MODULE).expect("the module parses");
        let mut constant = find(module.as_operation_ref(), "test.constant");
        let mut rewrite = ElementRewrite::types(i64_to_i32(&context))
            .with_int_constants(|_, _, _| Err(Report::msg("the value does not fit")));
        let err = replace_elements_in(&mut constant.borrow_mut(), &mut rewrite).unwrap_err();
        let text = crate::testing::text::describe(&err);
        assert!(text.contains("the value does not fit"), "{text}");
        assert!(text.contains("property 'value' of 'test.constant'"), "{text}");
        assert_eq!(constant.borrow().results()[0].borrow().ty(), context.i64());
    }

    /// A rewrite of integer constants that answers a value of the wrong width is an error, and
    /// the operation is left as it was.
    #[test]
    fn a_constant_rewritten_to_the_wrong_width_is_an_error() {
        let (context, module) = parse::<Module>(MODULE).expect("the module parses");
        let root = module.as_operation_ref();
        let mut constant = find(root, "test.constant");
        let mut rewrite = ElementRewrite::types(i64_to_i32(&context))
            .with_int_constants(|value, _, _| Ok(*value));
        let err = replace_elements_in(&mut constant.borrow_mut(), &mut rewrite)
            .expect_err("a 64-bit value at type i32");
        let message = crate::testing::label_messages(err.as_diagnostic()).join(" | ");
        assert!(
            message.contains(
                "property 'value' of 'test.constant' is the integer constant `4294967297 : i64`, \
                 which the rewrite of integer constants gave the 64-bit value 4294967297 at type \
                 `i32`, but a constant of that type is 32 bits wide"
            ),
            "{message}"
        );
        assert_eq!(constant.borrow().results()[0].borrow().ty(), context.i64());
    }

    /// A recursive struct type in a signature is mapped to a member of a new group, built from
    /// the mapped body, which refers to itself.
    #[test]
    fn a_recursive_type_in_a_signature_is_mapped_to_its_new_group() {
        let node_of = |context: &Context, field: Type| {
            let body = context.struct_type(StructType::named(
                Symbol::intern("node"),
                [field.into(), context.ptr_type(context.rec_var(0)).into()],
            ));
            context.rec_group(&[body]).expect("a valid group")[0]
        };
        let test = Test::new("f", |cx| ([cx.ptr_type(node_of(cx, cx.i64()))], []));
        let context = test.context_rc();
        let function = test.function();
        let old = context.ptr_type(node_of(&context, context.i64()));

        let mut rewrite = ElementRewrite::types(i64_to_i32(&context));
        let changed = recursively_replace_elements_in(function.as_operation_ref(), &mut rewrite)
            .expect("the rewrite");
        assert!(changed);
        drop(rewrite);

        let function = function.borrow();
        let param = function.signature().params()[0].ty;
        assert_ne!(param, old);
        let node = param.pointee().expect("a pointer");
        assert_eq!(node, node_of(&context, context.i32()));
        assert!(node.rec_group().is_some());
        let fields = node.as_struct().unwrap().fields();
        assert_eq!(fields[0].ty, context.i32());
        assert_eq!(fields[1].ty.pointee(), Some(node));
        let arg = test.entry_block().borrow().arguments()[0].borrow().ty();
        assert_eq!(arg, param);
        function.as_operation().recursively_verify().expect("the function verifies");
    }

    /// One attribute installed on two operations: rewriting one replaces it there, and the other
    /// keeps the attribute, unchanged.
    #[test]
    fn a_shared_attribute_is_replaced_only_where_rewritten() {
        let (context, module) = parse::<Module>(MODULE).expect("the module parses");
        let root = module.as_operation_ref();
        let shared = context.create_attribute::<TypeAttr, _>(context.i64()).as_attribute_ref();
        let mut global = find(root, "builtin.global_variable");
        let mut function = find(root, "builtin.function");
        global.borrow_mut().set_attribute("shared", shared);
        function.borrow_mut().set_attribute("shared", shared);

        let mut rewrite = ElementRewrite::types(i64_to_i32(&context));
        assert!(replace_elements_in(&mut global.borrow_mut(), &mut rewrite).expect("the rewrite"));

        let rewritten = global.borrow().get_attribute(Symbol::intern("shared")).unwrap();
        assert!(!AttributeRef::ptr_eq(&rewritten, &shared));
        assert_eq!(
            *rewritten.borrow().downcast_ref::<TypeAttr>().unwrap().as_value(),
            context.i32()
        );
        let kept = function.borrow().get_attribute(Symbol::intern("shared")).unwrap();
        assert!(AttributeRef::ptr_eq(&kept, &shared));
        assert_eq!(*shared.borrow().downcast_ref::<TypeAttr>().unwrap().as_value(), context.i64());
    }
}

/// Symbol references held inside other attributes: a rebuilt list gives the copy of a reference
/// a use of its own, replacing or removing an attribute unlinks the references it held, and an
/// operation erased afterwards leaves no use behind.
#[cfg(test)]
mod nested_reference_tests {
    use alloc::{format, rc::Rc, vec};

    use super::*;
    use crate::{
        AsSymbolRef, CallConv, Op, Symbol as _, SymbolName, SymbolTable, SymbolUse,
        UnsafeIntrusiveEntityRef, Usable,
        attributes::{AttrList, NamedAttribute},
        dialects::builtin::{
            FunctionRef,
            attributes::{
                AbiParam, List, ListAttr, Signature, SignatureAttr, SymbolRef, SymbolRefAttr,
                TypeAttr,
            },
        },
        testing::Test,
    };

    fn list(cx: &Rc<Context>, entries: impl IntoIterator<Item = AttributeRef>) -> AttributeRef {
        let mut list = AttrList::new();
        for entry in entries {
            list.push_back(entry.borrow().as_attr().as_attr_ref());
        }
        cx.create_attribute::<ListAttr, _>(List::from(list)).as_attribute_ref()
    }

    /// A reference to `to`, linked, whose use belongs to `owner`.
    fn linked(
        cx: &Rc<Context>,
        owner: FunctionRef,
        to: FunctionRef,
    ) -> UnsafeIntrusiveEntityRef<SymbolRefAttr> {
        let path = to.borrow().path();
        let mut reference = cx.create_attribute::<SymbolRefAttr, _>(SymbolRef::new(path, None));
        let user = cx.alloc_tracked(SymbolUse::new(owner.as_operation_ref(), reference));
        reference.borrow_mut().set_user(user);
        reference.borrow_mut().link(to.as_symbol_ref());
        reference
    }

    fn uses(symbol: FunctionRef) -> usize {
        symbol.borrow().iter_uses().count()
    }

    fn i64_to_i32(cx: &Context, ty: Type) -> Type {
        if ty == cx.i64() { cx.i32() } else { ty }
    }

    /// The one reference `owner` holds.
    fn held_reference(owner: FunctionRef) -> UnsafeIntrusiveEntityRef<SymbolRefAttr> {
        let mut found = None;
        let owner = owner.borrow();
        let _ = owner.as_operation().walk_attributes(&mut |_, attr| {
            if let Ok(reference) = attr.try_downcast_attr::<SymbolRefAttr>() {
                assert!(found.is_none(), "one reference");
                found = Some(reference);
            }
            WalkResult::Continue(())
        });
        found.expect("a reference")
    }

    /// `@owner` holding `refs = [@g, type i64]`, rewritten `i64 -> i32`, which rebuilds the list
    /// and copies the reference: the test, `@owner`, `@g`, and the original reference.
    fn rebuilt() -> (Test, FunctionRef, FunctionRef, UnsafeIntrusiveEntityRef<SymbolRefAttr>) {
        let mut test = Test::named("elements").in_module("m");
        let mut owner = test.define_function("owner", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        let cx = test.context_rc();
        let reference = linked(&cx, owner, g);
        let ty = cx.create_attribute::<TypeAttr, _>(cx.i64()).as_attribute_ref();
        let refs = list(&cx, [reference.as_attribute_ref(), ty]);
        owner.borrow_mut().as_operation_mut().set_attribute("refs", refs);
        owner.borrow().as_operation().verify().expect("the operation verifies");

        let mut rewrite = ElementRewrite::types(TypeMapper::new(&cx, i64_to_i32));
        assert!(replace_elements_in(owner.borrow_mut().as_operation_mut(), &mut rewrite).unwrap());
        drop(rewrite);
        let refs_now = owner.borrow().as_operation().get_attribute(Symbol::intern("refs"));
        assert!(!AttributeRef::ptr_eq(&refs_now.unwrap(), &refs), "the list is rebuilt");
        (test, owner, g, reference)
    }

    /// The copy in the rebuilt list has a use of its own, which names it and `@owner` and is the
    /// only use of `@g`; the original's use is unlinked.
    #[test]
    fn a_symbol_reference_in_a_rebuilt_list_stays_linked() {
        let (_test, owner, g, original) = rebuilt();
        let held = held_reference(owner);
        assert!(held != original, "the rebuilt list holds a copy");
        let user = held.borrow().try_user().expect("the copy has a use");
        assert!(user.borrow().attr == held, "the use names the held reference");
        assert!(user.borrow().owner == owner.as_operation_ref());
        assert_eq!(user.linked_symbol(), Some(g.as_symbol_ref()));
        assert_eq!(uses(g), 1);
        assert!(g.borrow().iter_uses().any(|symbol_use| symbol_use.attr == held));
        let original_user = original.borrow().try_user().unwrap();
        assert!(!original_user.is_linked(), "the original's use is unlinked");
        if let Err(err) = owner.borrow().as_operation().verify() {
            panic!("the rewritten operation does not verify: {err:?}");
        }
    }

    /// Renaming the symbol afterwards renames the reference the operation holds.
    #[test]
    fn a_reference_in_a_rebuilt_list_follows_a_rename() {
        let (test, owner, g, _original) = rebuilt();
        let mut module = test.module();
        module
            .borrow_mut()
            .symbol_manager_mut()
            .rename_symbol(g.as_symbol_ref(), SymbolName::intern("renamed"))
            .expect("renamed");
        let held = held_reference(owner);
        let held_path = format!("{}", held.borrow().path());
        assert!(held_path.contains("renamed"), "the held reference names `{held_path}`");
        owner.borrow().as_operation().verify().expect("the operation verifies");
    }

    /// Erasing the operation after the rewrite leaves the use list of the symbol empty.
    #[test]
    fn erasing_after_a_rebuild_leaves_no_use() {
        let (_test, mut owner, g, _original) = rebuilt();
        owner.borrow_mut().as_operation_mut().erase();
        assert_eq!(uses(g), 0);
    }

    /// A copy of a reference installed by hand shares the original's use, and verification
    /// reports it.
    #[test]
    fn a_reference_sharing_another_references_use_is_rejected() {
        let mut test = Test::named("elements").in_module("m");
        let mut owner = test.define_function("owner", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        let cx = test.context_rc();
        let reference = linked(&cx, owner, g);
        let alias = reference.borrow().as_attribute_ref().borrow().dyn_clone();
        owner.borrow_mut().as_operation_mut().set_attribute("alias", alias);
        let err = owner.borrow().as_operation().verify().expect_err("an alias is rejected");
        let text = crate::testing::text::describe(&err);
        assert!(text.contains("shares the use of another reference"), "{text}");
    }

    /// Replacing or removing an attribute unlinks the references it holds, top-level and nested;
    /// replacing it with an attribute that holds the same reference keeps it linked.
    #[test]
    fn replacing_or_removing_an_attribute_unlinks_its_references() {
        for nested in [false, true] {
            let mut test = Test::named("elements").in_module("m");
            let mut owner = test.define_function("owner", &[], &[]);
            let g = test.define_function("g", &[], &[]);
            let cx = test.context_rc();
            let wrap = |reference: AttributeRef| {
                if nested {
                    list(&cx, [reference])
                } else {
                    reference
                }
            };
            let other = || cx.create_attribute::<TypeAttr, _>(cx.i64()).as_attribute_ref();

            // replaced by an attribute without the reference
            let reference = linked(&cx, owner, g).as_attribute_ref();
            owner.borrow_mut().as_operation_mut().set_attribute("refs", wrap(reference));
            assert_eq!(uses(g), 1);
            owner.borrow_mut().as_operation_mut().set_attribute("refs", other());
            assert_eq!(uses(g), 0, "replaced, nested = {nested}");

            // removed
            let reference = linked(&cx, owner, g).as_attribute_ref();
            owner.borrow_mut().as_operation_mut().set_attribute("refs", wrap(reference));
            assert_eq!(uses(g), 1);
            owner.borrow_mut().as_operation_mut().remove_attribute("refs");
            assert_eq!(uses(g), 0, "removed, nested = {nested}");

            // replaced by an attribute that holds the same reference
            let reference = linked(&cx, owner, g).as_attribute_ref();
            owner.borrow_mut().as_operation_mut().set_attribute("refs", reference);
            owner
                .borrow_mut()
                .as_operation_mut()
                .set_attribute("refs", list(&cx, [reference]));
            assert_eq!(uses(g), 1, "the same reference is kept linked");
            owner.borrow().as_operation().verify().expect("the operation verifies");
        }
    }

    /// Replacing a property unlinks the references the old value held.
    #[test]
    fn replacing_a_property_unlinks_its_references() {
        let mut test = Test::named("elements").in_module("m");
        let g = test.define_function("g", &[], &[]);
        let cx = test.context_rc();
        let mut owner = test.define_function("owner", &[], &[]);
        let param = AbiParam::new_with_attrs(
            cx.i32(),
            [NamedAttribute::new("callee", linked(&cx, owner, g).as_attribute_ref())],
        );
        let signature = Signature::from_parts(CallConv::default(), vec![param], vec![]);
        let signature = cx.create_attribute::<SignatureAttr, _>(signature).as_attribute_ref();
        let previous = owner.borrow().as_operation().get_attribute(Symbol::intern("signature"));
        let previous = previous.expect("a function has a signature property");
        owner
            .borrow_mut()
            .as_operation_mut()
            .set_property("signature", signature)
            .unwrap();
        assert_eq!(uses(g), 1);
        owner
            .borrow_mut()
            .as_operation_mut()
            .set_property("signature", previous)
            .unwrap();
        assert_eq!(uses(g), 0);
    }

    /// Erasing an operation unlinks a reference two attributes deep: in a list in a list, and in
    /// a parameter attribute of a signature in a list.
    #[test]
    fn erasing_unlinks_references_at_any_depth() {
        let mut test = Test::named("elements").in_module("m");
        let mut owner = test.define_function("owner", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        let h = test.define_function("h", &[], &[]);
        let cx = test.context_rc();

        let in_list = linked(&cx, owner, g);
        let inner = list(&cx, [in_list.as_attribute_ref()]);
        let outer = list(&cx, [inner]);

        let in_param = linked(&cx, owner, h);
        let param = AbiParam::new_with_attrs(
            cx.i32(),
            [NamedAttribute::new("callee", in_param.as_attribute_ref())],
        );
        let signature = Signature::from_parts(CallConv::default(), vec![param], vec![]);
        let signature = cx.create_attribute::<SignatureAttr, _>(signature).as_attribute_ref();
        let sigs = list(&cx, [signature]);

        {
            let mut owner = owner.borrow_mut();
            owner.as_operation_mut().set_attribute("refs", outer);
            owner.as_operation_mut().set_attribute("sigs", sigs);
        }
        owner.borrow().as_operation().verify().expect("verifies");
        assert_eq!((uses(g), uses(h)), (1, 1));
        let mut count = 0;
        let _ = owner.borrow().as_operation().walk_symbol_uses(|_| {
            count += 1;
            WalkResult::Continue(())
        });
        assert_eq!(count, 2);

        owner.borrow_mut().as_operation_mut().erase();
        assert_eq!((uses(g), uses(h)), (0, 0));
    }

    /// `@owner` holding one reference handle twice: in `refs = [@g, type i64]` and directly at
    /// `direct`, then rewritten `i64 -> i32`, which rebuilds `refs`. Returns the test, `@owner`,
    /// `@g` and the reference.
    fn rebuilt_with_reference_held_twice()
    -> (Test, FunctionRef, FunctionRef, UnsafeIntrusiveEntityRef<SymbolRefAttr>) {
        let mut test = Test::named("elements").in_module("m");
        let mut owner = test.define_function("owner", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        let cx = test.context_rc();
        let reference = linked(&cx, owner, g);
        let ty = cx.create_attribute::<TypeAttr, _>(cx.i64()).as_attribute_ref();
        let refs = list(&cx, [reference.as_attribute_ref(), ty]);
        {
            let mut owner = owner.borrow_mut();
            owner.as_operation_mut().set_attribute("refs", refs);
            owner.as_operation_mut().set_attribute("direct", reference.as_attribute_ref());
        }
        owner.borrow().as_operation().verify().expect("verifies before");
        assert_eq!(uses(g), 1);

        let mut rewrite = ElementRewrite::types(TypeMapper::new(&cx, i64_to_i32));
        assert!(replace_elements_in(owner.borrow_mut().as_operation_mut(), &mut rewrite).unwrap());
        drop(rewrite);
        (test, owner, g, reference)
    }

    /// The same reference handle held directly and in a list the rewrite rebuilds: the reference
    /// still held directly stays linked beside the copy's own use.
    #[test]
    fn rr_a_reference_held_twice_stays_linked_after_a_rebuild() {
        let (_test, owner, g, reference) = rebuilt_with_reference_held_twice();
        let direct_user = reference.borrow().try_user().unwrap();
        assert!(direct_user.is_linked(), "the reference still held at `direct` was unlinked");
        assert_eq!(uses(g), 2, "the copy and the kept reference");
        if let Err(err) = owner.borrow().as_operation().verify() {
            panic!("the rewritten operation does not verify: {err:?}");
        }
    }

    /// The same reference handle held at two dictionary entries: replacing one entry does not
    /// unlink the reference the other still holds.
    #[test]
    fn rr_replacing_one_of_two_holders_keeps_the_reference_linked() {
        let mut test = Test::named("elements").in_module("m");
        let mut owner = test.define_function("owner", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        let cx = test.context_rc();
        let reference = linked(&cx, owner, g).as_attribute_ref();
        {
            let mut owner = owner.borrow_mut();
            owner.as_operation_mut().set_attribute("a", reference);
            owner.as_operation_mut().set_attribute("b", list(&cx, [reference]));
        }
        owner.borrow().as_operation().verify().expect("verifies before");
        let other = cx.create_attribute::<TypeAttr, _>(cx.i64()).as_attribute_ref();
        owner.borrow_mut().as_operation_mut().set_attribute("b", other);
        assert_eq!(uses(g), 1, "`a` still holds the reference");
        if let Err(err) = owner.borrow().as_operation().verify() {
            panic!("the operation does not verify after the replacement: {err:?}");
        }

        // removing the other holder unlinks it
        owner.borrow_mut().as_operation_mut().remove_attribute("a");
        assert_eq!(uses(g), 0, "nothing holds the reference");
    }

    /// A rename after the rebuild renames the reference still held directly.
    #[test]
    fn rr_rename_after_rebuild_with_reference_held_twice() {
        let (test, owner, g, reference) = rebuilt_with_reference_held_twice();
        let mut module = test.module();
        module
            .borrow_mut()
            .symbol_manager_mut()
            .rename_symbol(g.as_symbol_ref(), SymbolName::intern("renamed"))
            .expect("renamed");
        let direct = format!("{}", reference.borrow().path());
        assert!(direct.contains("renamed"), "the directly held reference names `{direct}`");
        owner.borrow().as_operation().verify().expect("the operation verifies");
    }

    /// A list shared by two operations: the one that does not own the use is rejected; rewriting
    /// it gives it its own use and leaves the owner's alone, and both verify.
    #[test]
    fn rr_a_list_shared_by_two_operations() {
        let mut test = Test::named("elements").in_module("m");
        let mut a = test.define_function("a", &[], &[]);
        let mut b = test.define_function("b", &[], &[]);
        let g = test.define_function("g", &[], &[]);
        let cx = test.context_rc();
        let reference = linked(&cx, a, g);
        let ty = cx.create_attribute::<TypeAttr, _>(cx.i64()).as_attribute_ref();
        let refs = list(&cx, [reference.as_attribute_ref(), ty]);
        a.borrow_mut().as_operation_mut().set_attribute("refs", refs);
        b.borrow_mut().as_operation_mut().set_attribute("refs", refs);
        a.borrow().as_operation().verify().expect("the owner verifies");
        b.borrow().as_operation().verify().expect_err("the other is rejected");

        let mut rewrite = ElementRewrite::types(TypeMapper::new(&cx, i64_to_i32));
        assert!(replace_elements_in(b.borrow_mut().as_operation_mut(), &mut rewrite).unwrap());
        drop(rewrite);
        assert_eq!(uses(g), 2);
        a.borrow().as_operation().verify().expect("a still verifies");
        b.borrow().as_operation().verify().expect("b verifies after its rewrite");
    }
}

#[cfg(test)]
mod element_model_tests;
#[cfg(test)]
mod verification_model_tests;
