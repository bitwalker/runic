//! What an attribute holds: the types and attributes in its value, and its own type.
//!
//! An attribute's value is walked through [AttributeElements], which `#[derive(DialectAttribute)]`
//! implements field by field over [AttributeComponent]s, as `#[derive(DialectType)]` walks a
//! type's value over [TypeComponent](crate::types::TypeComponent)s. Through a `dyn Attribute`,
//! [Attribute::visit_elements] and [Attribute::replace_elements] add the attribute's own type,
//! when it is an element (see [AttributeElements]).
//!
//! Nothing here edits an attribute in place: a replacement is a new attribute, and the old one,
//! which another holder may share, is left as it was.

use alloc::{boxed::Box, string::String, vec::Vec};
use core::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize};

use compact_str::CompactString;
use smallvec::SmallVec;

use super::{
    Attribute, AttributeRef, AttributeRegistration, IntoAttributeRef, MaybeInferAttributeType,
};
use crate::{
    Align, CallConv, Int, SymbolPath, SymbolUseRef, Type, UnsafeIntrusiveEntityRef,
    diagnostics::{ColumnNumber, LineNumber, SourceSpan},
    dialects::builtin::FunctionRef,
    effects::MemoryEffect,
    interner::Symbol,
};

/// Receives the types and attributes a value holds, in the order the value holds them.
pub trait ElementVisitor {
    fn visit_type(&mut self, ty: Type);
    fn visit_attribute(&mut self, attr: &AttributeRef);
}

/// Says what a value holds instead of each type and attribute it holds now.
///
/// It is asked about every element, in the order a visit lists them, even after an earlier answer
/// already changed the value.
pub trait ElementReplacer {
    /// The type to hold instead of `ty` (`ty` itself to keep it).
    fn replace_type(&mut self, ty: Type) -> Type;
    /// The attribute to hold instead of `attr`; `None` keeps it.
    ///
    /// The holder may install a copy of the answer rather than the answer itself (a list, whose
    /// entries belong to it, always does), so a replacer must not rely on the identity of the
    /// attribute it returned.
    fn replace_attribute(&mut self, attr: &AttributeRef) -> Option<AttributeRef>;
}

/// Implemented by every attribute value: the types and attributes it holds directly.
///
/// "Directly" means an attribute held by the value is one element: what that attribute holds in
/// turn is its own elements, not this value's. `#[derive(DialectAttribute)]` implements it field
/// by field, so every field of a derived value must be an [AttributeComponent]; a field that is
/// not does not compile:
///
/// ```compile_fail
/// extern crate alloc;
///
/// use runic_core::{derive::DialectAttribute, dialects::test::TestDialect};
///
/// /// Not an `AttributeComponent`.
/// #[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
/// pub struct Opaque;
///
/// #[derive(DialectAttribute, Debug, Default, Clone, PartialEq, Eq, Hash)]
/// #[attribute(dialect = TestDialect)]
/// pub struct HoldsOpaque {
///     field: Opaque,
/// }
/// ```
///
/// A value given with `remote = ".."` implements it by hand, next to the value's type.
///
/// Through a `dyn Attribute`, [Attribute::visit_elements] and [Attribute::replace_elements] list
/// the value's elements and then the attribute's own type, when that is an element:
///
/// - The unknown type is not an element: it means the attribute has no type of its own.
/// - A kind that takes its type from its value's Rust type (`BoolAttr` is `i1`, `I8Attr` to
///   `I128Attr` are `iN`, `I32ArrayAttr` is `i32`) holds no own type as an element: it is not
///   visited, not offered to a replacer, and an attribute rebuilt because its value changed takes
///   the type inferred from the new value.
/// - Otherwise the own type is an element, after the value's: `IntAttr`'s (its width is in its
///   value, but its type may be `isize`), an unbound `LocalVariableAttr`'s.
///
/// Some kinds' choices, as built: a signature's elements are, per parameter, its type then its
/// attributes, then the same per result; a bound `LocalVariableAttr` holds no type (its type is in
/// its function's `locals`); a `SymbolRefAttr` holds nothing, so it is never rebuilt by a
/// replacement and stays linked to its symbol.
///
/// Both methods are pure: they do not create attributes, intern types, or use the context.
pub trait AttributeElements: Sized {
    /// Calls `visitor` with each type and attribute this value holds directly, in a fixed order.
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor);

    /// This value with each type and attribute it holds replaced by `replacer`'s answer, asked in
    /// the order of [Self::visit_elements]; everything else is unchanged.
    ///
    /// `None` when the replacer changed nothing.
    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self>;
}

/// A part of an attribute's value that may hold types and attributes.
///
/// [AttributeElements] values are walked through their components: `#[derive(DialectAttribute)]`
/// visits and replaces each field of a value with these methods, and
/// `#[derive(AttributeComponent)]` implements them for a struct or enum used as such a field.
///
/// A [Type] holds itself, and so does an attribute handle, an [AttributeRef] or a handle to one
/// kind of attribute. Data that holds neither (integers, `bool`, strings, symbols, [Int], plain
/// enums) is a component that holds nothing. So is a handle to an operation or to a symbol use:
/// what it points to is not held by the value. Options, boxes, vectors, arrays and tuples of
/// components hold what their elements hold.
///
/// A handle to one kind of attribute, `UnsafeIntrusiveEntityRef<T>`, can only hold an attribute
/// of kind `T`: replacing it with an attribute of another kind panics.
///
/// Both methods must be pure, as [AttributeElements]'s are.
pub trait AttributeComponent: Sized {
    /// Calls `visitor` with each type and attribute this component holds, in a fixed order.
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor);

    /// This component with each type and attribute replaced by `replacer`'s answer, asked in the
    /// order of [Self::visit_elements]; `None` when the replacer changed nothing.
    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self>;
}

/// The type a kind of attribute takes from its value, whatever the value: `i1` for `BoolAttr`,
/// `iN` for the fixed-width integer kinds, `i32` for `I32ArrayAttr`. `None` for a kind whose type
/// does not follow from its value's Rust type alone: `IntAttr` (whose width the value carries, and
/// which may be `isize`) and every kind that infers nothing.
fn inferred_own_type<A>(context: &crate::Context) -> Option<Type>
where
    A: MaybeInferAttributeType,
{
    A::maybe_infer_type(context).filter(|ty| !ty.is_unknown())
}

/// What [Attribute::visit_elements] does for every derived attribute kind: the elements of its
/// value, then its own type, when that is an element (see [AttributeElements]).
#[doc(hidden)]
pub fn visit_attribute_elements<A>(attr: &A, visitor: &mut dyn ElementVisitor)
where
    A: AttributeRegistration + MaybeInferAttributeType,
{
    A::underlying_value(attr).visit_elements(visitor);
    let ty = Attribute::ty(attr);
    if !ty.is_unknown() && inferred_own_type::<A>(attr.context()).is_none() {
        visitor.visit_type(ty);
    }
}

/// What [Attribute::replace_elements] does for every derived attribute kind: a new attribute of
/// kind `A`, built with `A`'s constructor from the replaced value (or a clone of the value, when
/// only the type changed) and the replaced type, or `None` when nothing changed. `attr` is not
/// touched.
///
/// The own type is offered to the replacer only when it is an element (see [AttributeElements]).
/// For a kind that takes its type from its value, the new attribute's type is the one inferred
/// from the new value.
#[doc(hidden)]
pub fn replace_attribute_elements<A>(
    attr: &A,
    replacer: &mut dyn ElementReplacer,
) -> Option<AttributeRef>
where
    A: AttributeRegistration + MaybeInferAttributeType,
    A::Value: Clone,
{
    let context = attr.context();
    let value = A::underlying_value(attr).replace_elements(replacer);
    let ty = Attribute::ty(attr);
    let inferred = inferred_own_type::<A>(context).is_some();
    let new_ty = if ty.is_unknown() || inferred {
        ty
    } else {
        replacer.replace_type(ty)
    };
    if value.is_none() && new_ty == ty {
        return None;
    }
    let value = value.unwrap_or_else(|| A::underlying_value(attr).clone());
    let new_ty = if inferred {
        A::maybe_infer_type_from_value(context, &value)
            .filter(|ty| !ty.is_unknown())
            .unwrap_or(new_ty)
    } else {
        new_ty
    };
    Some(A::create(&attr.context_rc(), value, new_ty).into_attribute_ref())
}

/// The replacement of a sequence of components: every element is asked, in order; `None` when
/// none changed, or else every element's replacement or clone, collected.
fn replace_each<'a, T, C>(
    elements: impl IntoIterator<Item = &'a T>,
    replacer: &mut dyn ElementReplacer,
) -> Option<C>
where
    T: AttributeComponent + Clone + 'a,
    C: FromIterator<T>,
{
    let elements: Vec<(&T, Option<T>)> = elements
        .into_iter()
        .map(|element| (element, element.replace_elements(replacer)))
        .collect();
    if elements.iter().all(|(_, new)| new.is_none()) {
        return None;
    }
    Some(
        elements
            .into_iter()
            .map(|(element, new)| new.unwrap_or_else(|| element.clone()))
            .collect(),
    )
}

impl AttributeComponent for Type {
    #[inline]
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        visitor.visit_type(*self);
    }

    #[inline]
    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        let ty = replacer.replace_type(*self);
        (ty != *self).then_some(ty)
    }
}

impl AttributeComponent for AttributeRef {
    #[inline]
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        visitor.visit_attribute(self);
    }

    #[inline]
    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        replacer.replace_attribute(self)
    }
}

impl<T: AttributeRegistration> AttributeComponent for UnsafeIntrusiveEntityRef<T> {
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        visitor.visit_attribute(&(*self).into_attribute_ref());
    }

    /// Panics if the replacer answers with an attribute that is not of kind `T`.
    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        let new = replacer.replace_attribute(&(*self).into_attribute_ref())?;
        let name = new.name();
        Some(new.try_downcast_attr::<T>().unwrap_or_else(|_| {
            panic!(
                "a handle to a `{}` attribute cannot hold the `{name}` attribute it was replaced \
                 with",
                T::full_name()
            )
        }))
    }
}

/// Implements [AttributeComponent] for data that holds no types or attributes: it visits nothing
/// and is never replaced.
macro_rules! plain_components {
    ($($ty:ty),* $(,)?) => {
        $(
            impl AttributeComponent for $ty {
                #[inline]
                fn visit_elements(&self, _visitor: &mut dyn ElementVisitor) {}

                #[inline]
                fn replace_elements(&self, _replacer: &mut dyn ElementReplacer) -> Option<Self> {
                    None
                }
            }
        )*
    };
}

plain_components!(
    (),
    bool,
    char,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
    NonZeroU8,
    NonZeroU16,
    NonZeroU32,
    NonZeroU64,
    NonZeroU128,
    NonZeroUsize,
    Symbol,
    CompactString,
    String,
    CallConv,
    Int,
    Align,
    MemoryEffect,
    SymbolPath,
    SourceSpan,
    LineNumber,
    ColumnNumber,
    // The function whose locals table a local variable indexes: the table is the function's.
    FunctionRef,
    // The use that links a symbol reference to its symbol: the use is the reference's link, not
    // something it holds.
    SymbolUseRef,
);

/// Implements [AttributeElements] for a `remote` attribute value that is itself a component: the
/// value holds what the component holds.
macro_rules! component_values {
    ($($ty:ty),* $(,)?) => {
        $(
            impl AttributeElements for $ty {
                #[inline]
                fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
                    AttributeComponent::visit_elements(self, visitor);
                }

                #[inline]
                fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
                    AttributeComponent::replace_elements(self, replacer)
                }
            }
        )*
    };
}

// The `remote` values of `TypeAttr` (and `ub`'s `PoisonAttr`), the marker attributes, `BoolAttr`,
// `StringAttr`, `IntAttr` and the fixed-width integer attributes.
component_values!(Type, (), bool, CompactString, Int, i8, i16, i32, i64, i128);

impl<T: AttributeComponent> AttributeComponent for Option<T> {
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        if let Some(value) = self {
            value.visit_elements(visitor);
        }
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        Some(Some(self.as_ref()?.replace_elements(replacer)?))
    }
}

impl<T: AttributeComponent> AttributeComponent for Box<T> {
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        (**self).visit_elements(visitor);
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        (**self).replace_elements(replacer).map(Box::new)
    }
}

impl<T: AttributeComponent + Clone> AttributeComponent for Box<[T]> {
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        for element in self.iter() {
            element.visit_elements(visitor);
        }
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        replace_each(self.iter(), replacer)
    }
}

impl<T: AttributeComponent + Clone> AttributeComponent for Vec<T> {
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        for element in self {
            element.visit_elements(visitor);
        }
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        replace_each(self, replacer)
    }
}

impl<T: AttributeComponent + Clone, const N: usize> AttributeComponent for SmallVec<[T; N]> {
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        for element in self {
            element.visit_elements(visitor);
        }
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        replace_each(self, replacer)
    }
}

impl<T: AttributeComponent + Clone, const N: usize> AttributeComponent for [T; N] {
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        for element in self {
            element.visit_elements(visitor);
        }
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        let elements: Vec<T> = replace_each(self, replacer)?;
        Some(elements.try_into().unwrap_or_else(|_| unreachable!("one element per element")))
    }
}

impl<A, B> AttributeComponent for (A, B)
where
    A: AttributeComponent + Clone,
    B: AttributeComponent + Clone,
{
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        self.0.visit_elements(visitor);
        self.1.visit_elements(visitor);
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        let a = self.0.replace_elements(replacer);
        let b = self.1.replace_elements(replacer);
        if a.is_none() && b.is_none() {
            return None;
        }
        Some((a.unwrap_or_else(|| self.0.clone()), b.unwrap_or_else(|| self.1.clone())))
    }
}

impl<A, B, C> AttributeComponent for (A, B, C)
where
    A: AttributeComponent + Clone,
    B: AttributeComponent + Clone,
    C: AttributeComponent + Clone,
{
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        self.0.visit_elements(visitor);
        self.1.visit_elements(visitor);
        self.2.visit_elements(visitor);
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        let a = self.0.replace_elements(replacer);
        let b = self.1.replace_elements(replacer);
        let c = self.2.replace_elements(replacer);
        if a.is_none() && b.is_none() && c.is_none() {
            return None;
        }
        Some((
            a.unwrap_or_else(|| self.0.clone()),
            b.unwrap_or_else(|| self.1.clone()),
            c.unwrap_or_else(|| self.2.clone()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, sync::Arc, vec, vec::Vec};

    use super::*;
    use crate::{
        AttributeRef, Context,
        attributes::{AttrList, IntoAttributeRef},
        constants::ConstantData,
        dialects::{
            builtin::attributes::{
                AbiParam, Array, BoolAttr, BytesAttr, DataLayoutAttr, DeclaredEffects,
                DeclaredEffectsAttr, I8Attr, I16Attr, I32ArrayAttr, I32Attr, I64Attr, I128Attr,
                IdentAttr, IntAttr, List, ListAttr, LocalVariable, LocalVariableAttr, Location,
                LocationAttr, MemoryEffectDescriptor, Overflow, OverflowAttr, SextAttr, Signature,
                SignatureAttr, Signedness, SignednessAttr, SretAttr, StringAttr, SymbolRef,
                SymbolRefAttr, TypeArrayAttr, TypeAttr, UnitAttr, Visibility, VisibilityAttr,
                ZextAttr,
            },
            debuginfo::attributes::{
                CompileUnit, CompileUnitAttr, Expression, ExpressionAttr, ExpressionOp,
                InlineCallChain, InlineCallChainAttr, InlineCallFrame, Subprogram, SubprogramAttr,
                Variable, VariableAttr,
            },
        },
        layout::{DataLayoutSpec, Endianness, PointerSpec},
        testing::{
            Test,
            elements::{Element, Identity, Markers, Row, check_every_kind, elements},
        },
    };

    fn ty(ty: Type) -> Element {
        Element::Type(ty)
    }

    fn attr(attr: AttributeRef) -> Element {
        Element::Attribute(attr)
    }

    fn int(context: &Rc<Context>, value: i32) -> AttributeRef {
        context.int_attr(Int::from(value), context.i32()).as_attribute_ref()
    }

    /// One row for each attribute kind of `builtin`, `di` and `test`, and a second where the
    /// value's shape matters (a bound local, a variable without a type).
    fn rows(test: &Test) -> Vec<Row> {
        let context = test.context_rc();
        let cx = &context;
        let (i8, i16, i32, i64) = (cx.i8(), cx.i16(), cx.i32(), cx.i64());
        let symbol = Symbol::intern;

        let zext = cx.get_marker::<ZextAttr>().as_attribute_ref();
        let signature = Signature::from_parts(
            CallConv::default(),
            vec![AbiParam::zext(i32, cx), AbiParam::new(i64)],
            vec![AbiParam::new(i8)],
        );

        let (one, two) = (int(cx, 1), int(cx, 2));
        let mut list = AttrList::new();
        for entry in [one, two] {
            list.push_back(entry.borrow().as_attr().as_attr_ref());
        }

        let mut function = test.function();
        let bound = function.borrow_mut().alloc_local(i16);

        let fn_ty = cx.function_type([i32], [i64]);
        let mut subprogram = Subprogram::new(symbol("f"), symbol("f.rs"), 1, None);
        subprogram.ty = Some(fn_ty);
        let mut variable = Variable::new(symbol("x"), symbol("f.rs"), 2, Some(3));
        variable.ty = Some(i32);
        let layout = DataLayoutSpec::new(Endianness::Little, PointerSpec::sized(32).unwrap())
            .expect("a valid data layout");

        vec![
            Row::new(
                "i32 array: its type follows from its value",
                cx.create_attribute::<I32ArrayAttr, _>(Array::from([1, 2])),
                [],
            ),
            Row::new(
                "type array",
                cx.create_attribute::<TypeArrayAttr, _>(Array::from([i8, i16])),
                [ty(i8), ty(i16)],
            ),
            Row::new("bool", cx.create_attribute::<BoolAttr, _>(true), []),
            Row::new(
                "bytes",
                cx.create_attribute::<BytesAttr, _>(Arc::new(ConstantData::from([1u8, 2]))),
                [],
            ),
            Row::new("data layout", cx.create_attribute::<DataLayoutAttr, _>(layout), []),
            Row::new(
                "declared effects",
                cx.create_attribute::<DeclaredEffectsAttr, _>(DeclaredEffects::Known(
                    [MemoryEffectDescriptor::on_argument(crate::effects::MemoryEffect::Read, 0)]
                        .into_iter()
                        .collect(),
                )),
                [],
            ),
            Row::new("i8", cx.create_attribute::<I8Attr, _>(1i8), []),
            Row::new("i16", cx.create_attribute::<I16Attr, _>(1i16), []),
            Row::new("i32", cx.create_attribute::<I32Attr, _>(1i32), []),
            Row::new("i64", cx.create_attribute::<I64Attr, _>(1i64), []),
            Row::new("i128", cx.create_attribute::<I128Attr, _>(1i128), []),
            Row::new("int: its own type only", cx.int_attr(Int::from(7i64), i64), [ty(i64)]),
            Row::new(
                "list of attributes",
                cx.create_attribute::<ListAttr, _>(List::from(list)),
                [attr(one), attr(two)],
            ),
            Row::new(
                "unbound local: its own type",
                cx.create_attribute_with_type::<LocalVariableAttr, _>(
                    LocalVariable::default(),
                    i32,
                ),
                [ty(i32)],
            ),
            Row::new(
                "bound local: the function holds its type",
                cx.create_attribute::<LocalVariableAttr, _>(bound),
                [],
            ),
            Row::new("location", cx.create_attribute::<LocationAttr, _>(Location::Synthetic), []),
            Row::new("overflow", cx.create_attribute::<OverflowAttr, _>(Overflow::Wrapping), []),
            Row::new(
                "signature: each parameter's type, then its attributes; then each result's",
                cx.create_attribute::<SignatureAttr, _>(signature),
                [ty(i32), attr(zext), ty(i64), ty(i8)],
            ),
            Row::new(
                "signedness",
                cx.create_attribute::<SignednessAttr, _>(Signedness::Signed),
                [],
            ),
            Row::new("string", cx.create_attribute::<StringAttr, _>(CompactString::from("s")), []),
            Row::new(
                "symbol reference",
                cx.create_attribute::<SymbolRefAttr, _>(SymbolRef::default()),
                [],
            ),
            Row::new("type", cx.create_attribute::<TypeAttr, _>(i64), [ty(i64)]),
            Row::new("unit", cx.get_marker::<UnitAttr>(), []),
            Row::new(
                "visibility",
                cx.create_attribute::<VisibilityAttr, _>(Visibility::Public),
                [],
            ),
            Row::new("sret", cx.get_marker::<SretAttr>(), []),
            Row::new("zext", cx.get_marker::<ZextAttr>(), []),
            Row::new("sext", cx.get_marker::<SextAttr>(), []),
            Row::new(
                "ident",
                cx.create_attribute::<IdentAttr, _>(crate::Ident::with_empty_span(symbol("x"))),
                [],
            ),
            Row::new(
                "compile unit",
                cx.create_attribute::<CompileUnitAttr, _>(CompileUnit::new(
                    symbol("rust"),
                    symbol("f.rs"),
                )),
                [],
            ),
            Row::new(
                "expression",
                cx.create_attribute::<ExpressionAttr, _>(Expression::with_ops(vec![
                    ExpressionOp::LocalSlot(0),
                    ExpressionOp::Deref,
                ])),
                [],
            ),
            Row::new(
                "inline call chain",
                cx.create_attribute::<InlineCallChainAttr, _>(InlineCallChain::new(vec![
                    InlineCallFrame {
                        name: symbol("g"),
                        linkage_name: None,
                        file: symbol("g.rs"),
                        line: 1,
                        column: 1,
                        call_file: symbol("f.rs"),
                        call_line: 2,
                        call_column: 3,
                    },
                ])),
                [],
            ),
            Row::new(
                "subprogram: its function type",
                cx.create_attribute::<SubprogramAttr, _>(subprogram),
                [ty(fn_ty)],
            ),
            Row::new(
                "variable: its type",
                cx.create_attribute::<VariableAttr, _>(variable.clone()),
                [ty(i32)],
            ),
            Row::new(
                "variable without a type",
                cx.create_attribute::<VariableAttr, _>(Variable {
                    ty: None,
                    ..variable
                }),
                [],
            ),
        ]
    }

    #[test]
    fn every_attribute_kind_lists_and_replaces_what_it_holds() {
        let test = Test::new("f", |_| ([] as [Type; 0], [] as [Type; 0]));
        check_every_kind(&test.context_rc(), &["builtin", "di", "test"], &rows(&test));
    }

    /// A list's replaced entries are copies of the replacer's answers and of the kept entries;
    /// the original list keeps its entries, linked to it.
    #[test]
    fn a_replaced_list_copies_replacements_and_kept_entries() {
        let context = Rc::new(Context::default());
        let (one, two, three) = (int(&context, 1), int(&context, 2), int(&context, 3));
        let mut entries = AttrList::new();
        for entry in [one, two] {
            entries.push_back(entry.borrow().as_attr().as_attr_ref());
        }
        let list = context.create_attribute::<ListAttr, _>(List::from(entries)).as_attribute_ref();

        struct ReplaceTwo(AttributeRef, AttributeRef);
        impl ElementReplacer for ReplaceTwo {
            fn replace_type(&mut self, ty: Type) -> Type {
                ty
            }

            fn replace_attribute(&mut self, attr: &AttributeRef) -> Option<AttributeRef> {
                (*attr == self.0).then_some(self.1)
            }
        }
        let new = list.borrow().replace_elements(&mut ReplaceTwo(two, three)).unwrap();
        let [Element::Attribute(kept), Element::Attribute(replaced)] = elements(&new)[..] else {
            panic!("two entries");
        };
        assert!(!AttributeRef::ptr_eq(&replaced, &three), "a replacement is copied");
        assert!(replaced == three);
        assert!(!three.borrow().as_attr().as_attr_ref().is_linked(), "the answer is not adopted");
        assert!(!AttributeRef::ptr_eq(&kept, &one), "a kept entry is copied");
        assert!(kept == one);
        assert_eq!(elements(&list), [attr(one), attr(two)]);
        assert!(one.borrow().as_attr().as_attr_ref().is_linked());
    }

    /// A kind that takes its type from its value holds no own type as an element: a replacer
    /// that changes every type leaves it alone, though its own type is a known type. (No kind
    /// holds a type in its value and also takes its own type from that value, so no rebuilt
    /// attribute gets a newly inferred type here.)
    #[test]
    fn an_own_type_inferred_from_the_value_is_not_an_element() {
        let context = Rc::new(Context::default());
        let cases = [
            (context.create_attribute::<BoolAttr, _>(true).as_attribute_ref(), context.i1()),
            (context.create_attribute::<I32Attr, _>(1i32).as_attribute_ref(), context.i32()),
            (
                context
                    .create_attribute::<I32ArrayAttr, _>(Array::from([1, 2]))
                    .as_attribute_ref(),
                context.i32(),
            ),
        ];
        for (attr, own) in cases {
            assert_eq!(attr.borrow().ty(), own);
            assert_eq!(elements(&attr), []);
            let mut markers = Markers::new(&context);
            assert!(attr.borrow().replace_elements(&mut markers).is_none(), "{}", attr.name());
        }
    }

    /// A shared attribute is never edited: replacing what it holds leaves every holder's handle
    /// pointing at the attribute as it was.
    #[test]
    fn replacement_builds_a_new_attribute_and_leaves_the_old_one() {
        let context = Rc::new(Context::default());
        let types = context.create_attribute::<TypeArrayAttr, _>(Array::from([context.i8()]));
        let shared = types.as_attribute_ref();
        let mut markers = Markers::new(&context);
        let new = shared.borrow().replace_elements(&mut markers).unwrap();
        assert_eq!(types.borrow().as_value()[0], context.i8());
        assert_eq!(elements(&new), [ty(markers.ty)]);
        assert!(new.try_downcast_attr::<TypeArrayAttr>().is_ok());
    }

    /// An integer constant whose type changes is replaced as asked, though its value no longer
    /// has its type's width: verification rejects it, not replacement.
    #[test]
    fn an_integer_constant_is_retyped_as_asked() {
        let context = Rc::new(Context::default());
        let constant = int(&context, -1);
        let mut markers = Markers::new(&context);
        let new = constant.borrow().replace_elements(&mut markers).unwrap();
        let new = new.try_downcast_attr::<IntAttr>().unwrap();
        assert_eq!(Attribute::ty(&*new.borrow()), markers.ty);
        assert_eq!(new.borrow().as_value().width(), 32);
    }

    #[test]
    fn a_replacer_is_asked_about_every_element_in_visit_order() {
        let context = Rc::new(Context::default());
        let (i8, i16, i32) = (context.i8(), context.i16(), context.i32());
        let component: (Vec<Option<Type>>, [Type; 2]) =
            (vec![Some(i8), None, Some(i16)], [i32, i8]);
        let mut identity = Identity::default();
        assert!(component.replace_elements(&mut identity).is_none());
        assert_eq!(identity.0, [ty(i8), ty(i16), ty(i32), ty(i8)]);

        struct FirstOnly(Vec<Type>, Type);
        impl ElementReplacer for FirstOnly {
            fn replace_type(&mut self, ty: Type) -> Type {
                self.0.push(ty);
                if self.0.len() == 1 { self.1 } else { ty }
            }

            fn replace_attribute(&mut self, _: &AttributeRef) -> Option<AttributeRef> {
                None
            }
        }
        let mut first = FirstOnly(Vec::new(), context.i64());
        let replaced = component.replace_elements(&mut first).unwrap();
        assert_eq!(first.0, [i8, i16, i32, i8], "asked after the first change too");
        assert_eq!(replaced, (vec![Some(context.i64()), None, Some(i16)], [i32, i8]));
    }

    #[test]
    fn a_typed_attribute_handle_holds_its_attribute() {
        let context = Rc::new(Context::default());
        let constant = context.int_attr(Int::from(1i32), context.i32());
        let other = context.int_attr(Int::from(2i32), context.i32());
        struct Visited(Vec<AttributeRef>);
        impl ElementVisitor for Visited {
            fn visit_type(&mut self, _: Type) {}

            fn visit_attribute(&mut self, attr: &AttributeRef) {
                self.0.push(*attr);
            }
        }
        let mut visited = Visited(Vec::new());
        AttributeComponent::visit_elements(&constant, &mut visited);
        assert!(AttributeRef::ptr_eq(&visited.0[0], &constant.into_attribute_ref()));

        struct To(AttributeRef);
        impl ElementReplacer for To {
            fn replace_type(&mut self, ty: Type) -> Type {
                ty
            }

            fn replace_attribute(&mut self, _: &AttributeRef) -> Option<AttributeRef> {
                Some(self.0)
            }
        }
        let replaced =
            AttributeComponent::replace_elements(&constant, &mut To(other.into_attribute_ref()));
        assert!(
            replaced.is_some_and(|replaced| UnsafeIntrusiveEntityRef::ptr_eq(&replaced, &other))
        );
    }

    #[test]
    #[should_panic(expected = "a handle to a `builtin.int` attribute cannot hold the \
                               `builtin.string` attribute")]
    fn a_typed_attribute_handle_rejects_another_kind() {
        let context = Rc::new(Context::default());
        let constant = context.int_attr(Int::from(1i32), context.i32());
        let markers = &mut Markers::new(&context);
        let _ = AttributeComponent::replace_elements(&constant, markers);
    }
}
