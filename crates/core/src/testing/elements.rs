//! A check of what every attribute kind of some dialects holds, for the dialect crates' tests.
//!
//! [check_every_kind] takes one row per attribute (an instance and the elements it is expected to
//! list) and fails when a kind registered by the dialects has no row, so a kind added later
//! without one fails the test.

use alloc::{collections::BTreeSet, format, rc::Rc, string::String, vec::Vec};
use core::fmt;

use crate::{
    AttributeRef, CompactString, Context, Type, UnsafeIntrusiveEntityRef,
    attributes::{ElementReplacer, ElementVisitor},
    dialects::builtin::attributes::StringAttr,
};

/// One element of an attribute: a type or an attribute it holds.
#[derive(Clone, PartialEq)]
pub enum Element {
    Type(Type),
    Attribute(AttributeRef),
}

impl fmt::Debug for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type(ty) => write!(f, "type {ty}"),
            Self::Attribute(attr) => write!(f, "attribute {:?}", attr.borrow()),
        }
    }
}

impl From<Type> for Element {
    fn from(ty: Type) -> Self {
        Self::Type(ty)
    }
}

impl From<AttributeRef> for Element {
    fn from(attr: AttributeRef) -> Self {
        Self::Attribute(attr)
    }
}

/// Records what it is shown, in order.
#[derive(Default)]
struct Recorder(Vec<Element>);

impl ElementVisitor for Recorder {
    fn visit_type(&mut self, ty: Type) {
        self.0.push(Element::Type(ty));
    }

    fn visit_attribute(&mut self, attr: &AttributeRef) {
        self.0.push(Element::Attribute(*attr));
    }
}

/// The elements `attr` lists, in order.
pub fn elements(attr: &AttributeRef) -> Vec<Element> {
    let mut recorder = Recorder::default();
    attr.borrow().visit_elements(&mut recorder);
    recorder.0
}

/// Keeps every element, and records what it was asked about, in order.
#[derive(Default)]
pub struct Identity(pub Vec<Element>);

impl ElementReplacer for Identity {
    fn replace_type(&mut self, ty: Type) -> Type {
        self.0.push(Element::Type(ty));
        ty
    }

    fn replace_attribute(&mut self, attr: &AttributeRef) -> Option<AttributeRef> {
        self.0.push(Element::Attribute(*attr));
        None
    }
}

/// Replaces every type with one marker type and every attribute with one marker attribute, and
/// records what it was asked about, in order.
pub struct Markers {
    pub ty: Type,
    pub attr: AttributeRef,
    pub asked: Vec<Element>,
}

impl Markers {
    /// The type `i77` and a string attribute `"marker"`, neither of which a row should hold.
    pub fn new(context: &Rc<Context>) -> Self {
        Self {
            ty: context.int_type(77),
            attr: context
                .create_attribute::<StringAttr, _>(CompactString::from("marker"))
                .as_attribute_ref(),
            asked: Vec::new(),
        }
    }

    /// What `element` is replaced with.
    pub fn marker_for(&self, element: &Element) -> Element {
        match element {
            Element::Type(_) => Element::Type(self.ty),
            Element::Attribute(_) => Element::Attribute(self.attr),
        }
    }
}

impl ElementReplacer for Markers {
    fn replace_type(&mut self, ty: Type) -> Type {
        self.asked.push(Element::Type(ty));
        self.ty
    }

    fn replace_attribute(&mut self, attr: &AttributeRef) -> Option<AttributeRef> {
        self.asked.push(Element::Attribute(*attr));
        Some(self.attr)
    }
}

/// An instance of an attribute kind and the elements it is expected to list.
pub struct Row {
    /// What the row is, for failure messages.
    pub case: &'static str,
    pub attr: AttributeRef,
    pub expected: Vec<Element>,
}

impl Row {
    pub fn new<T: crate::Attribute>(
        case: &'static str,
        attr: UnsafeIntrusiveEntityRef<T>,
        expected: impl IntoIterator<Item = Element>,
    ) -> Self {
        Self {
            case,
            attr: crate::attributes::IntoAttributeRef::into_attribute_ref(attr),
            expected: expected.into_iter().collect(),
        }
    }
}

/// The full name, `dialect.name`, of `attr`'s kind.
fn kind_of(attr: &AttributeRef) -> String {
    let name = attr.name();
    format!("{}.{}", name.dialect(), name.name())
}

/// Checks every row, and that the rows cover every attribute kind registered by `dialects` (by
/// namespace) and no other kind:
///
/// * the attribute lists exactly the expected elements, in order;
/// * replacing with the identity answers "unchanged", having been asked about exactly those
///   elements, in order;
/// * replacing every type and every attribute with a marker gives a new attribute of the same
///   kind that lists the markers, or "unchanged" when the attribute holds nothing; the original
///   still lists its own elements.
pub fn check_every_kind(context: &Rc<Context>, dialects: &[&str], rows: &[Row]) {
    let registered: BTreeSet<String> = dialects
        .iter()
        .flat_map(|dialect| {
            context
                .get_registered_dialect(crate::interner::Symbol::intern(dialect))
                .registered_attrs()
                .iter()
                .map(|name| format!("{}.{}", name.dialect(), name.name()))
                .collect::<Vec<_>>()
        })
        .collect();
    let covered: BTreeSet<String> = rows.iter().map(|row| kind_of(&row.attr)).collect();
    let missing: Vec<&String> = registered.difference(&covered).collect();
    assert!(missing.is_empty(), "attribute kinds without a row: {missing:?}");
    let extra: Vec<&String> = covered.difference(&registered).collect();
    assert!(extra.is_empty(), "rows for kinds not registered by {dialects:?}: {extra:?}");

    for row in rows {
        check_row(context, row);
    }
}

fn check_row(context: &Rc<Context>, row: &Row) {
    let case = row.case;
    let kind = kind_of(&row.attr);
    assert_eq!(elements(&row.attr), row.expected, "{case} ({kind}): elements");

    let mut identity = Identity::default();
    let unchanged = row.attr.borrow().replace_elements(&mut identity);
    assert!(unchanged.is_none(), "{case} ({kind}): the identity changed {unchanged:?}");
    assert_eq!(identity.0, row.expected, "{case} ({kind}): the identity was asked");

    let mut markers = Markers::new(context);
    let replaced = row.attr.borrow().replace_elements(&mut markers);
    assert_eq!(markers.asked, row.expected, "{case} ({kind}): the markers were asked");
    let expected: Vec<Element> =
        row.expected.iter().map(|element| markers.marker_for(element)).collect();
    match replaced {
        None => assert!(
            row.expected.is_empty(),
            "{case} ({kind}): replacing {:?} with markers changed nothing",
            row.expected
        ),
        Some(new) => {
            assert!(
                !row.expected.is_empty(),
                "{case} ({kind}): an attribute holding nothing was rebuilt"
            );
            assert_eq!(kind_of(&new), kind, "{case}: the new attribute's kind");
            assert!(!AttributeRef::ptr_eq(&new, &row.attr), "{case} ({kind}): not a new attribute");
            assert_eq!(elements(&new), expected, "{case} ({kind}): the new attribute's elements");
        }
    }
    assert_eq!(elements(&row.attr), row.expected, "{case} ({kind}): the original, after");
}
