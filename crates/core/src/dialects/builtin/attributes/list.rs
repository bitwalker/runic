use alloc::vec::Vec;
use core::fmt;

use crate::{
    AttrPrinter, Attribute, OpPrintingFlags,
    attributes::{AttrList, AttributeComponent, ElementReplacer, ElementVisitor},
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    print::AsmPrinter,
};

#[derive(DialectAttribute, Default)]
#[attribute(dialect = BuiltinDialect, implements(AttrPrinter))]
pub struct List(AttrList);

impl fmt::Debug for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for entry in self.0.iter() {
            let attr = entry.as_trait::<dyn Attribute>().unwrap();
            list.entry_with(|f| write!(f, "{attr:?}"));
        }
        list.finish()
    }
}

impl fmt::Display for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        let flags = OpPrintingFlags::default();
        for entry in self.0.iter() {
            let attr = entry.as_trait::<dyn Attribute>().unwrap();
            list.entry_with(|f| {
                let context = attr.context_rc();
                // Only reached outside an IR print (the `AttrPrinter` of a list is not implemented):
                // the diagnostic text of a value, printed as a fragment, without declarations.
                let mut printer = AsmPrinter::fragment(context, &flags);
                printer.print_attribute_value(attr);
                write!(f, "{}", printer.finish())
            });
        }
        list.finish()
    }
}

impl Eq for List {}
impl PartialEq for List {
    fn eq(&self, other: &Self) -> bool {
        let mut lhs = self.0.front();
        let mut rhs = other.0.front();
        loop {
            match (lhs.get(), rhs.get()) {
                (None, None) => break true,
                (Some(l), Some(r)) => {
                    if !l
                        .as_trait::<dyn Attribute>()
                        .unwrap()
                        .dyn_eq(r.as_trait::<dyn Attribute>().unwrap())
                    {
                        break false;
                    }
                }
                _ => break false,
            }

            lhs.move_next();
            rhs.move_next();
        }
    }
}

/// A list holds its attributes, in order.
///
/// A list owns its entries: an attribute is linked into one list at a time. A replaced list is a
/// new one, holding a copy of each replacement where the replacer gave one, and a copy of the
/// entry where it kept it. The new list never adopts an attribute it was given, so an answer the
/// replacer also gives to another holder is never taken from it.
impl AttributeComponent for AttrList {
    fn visit_elements(&self, visitor: &mut dyn ElementVisitor) {
        for entry in self.iter() {
            visitor.visit_attribute(&entry.as_attribute_ref());
        }
    }

    fn replace_elements(&self, replacer: &mut dyn ElementReplacer) -> Option<Self> {
        let entries: Vec<_> = self
            .iter()
            .map(|entry| {
                let entry = entry.as_attribute_ref();
                let new = replacer.replace_attribute(&entry);
                (entry, new)
            })
            .collect();
        if entries.iter().all(|(_, new)| new.is_none()) {
            return None;
        }
        let mut list = AttrList::new();
        for (entry, new) in entries {
            let attr = new.unwrap_or(entry).borrow().dyn_clone();
            let attr = attr.borrow().as_attr().as_attr_ref();
            list.push_back(attr);
        }
        Some(list)
    }
}

impl Clone for List {
    fn clone(&self) -> Self {
        let mut list = AttrList::new();
        for attr in self.0.iter() {
            let cloned = attr.as_trait::<dyn Attribute>().unwrap().dyn_clone();
            let attr = cloned.borrow().as_attr().as_attr_ref();
            list.push_back(attr);
        }
        Self(list)
    }
}

impl core::hash::Hash for List {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        for attr in self.0.iter() {
            attr.as_trait::<dyn Attribute>().unwrap().dyn_hash(state);
        }
    }
}

impl From<AttrList> for List {
    fn from(value: AttrList) -> Self {
        Self(value)
    }
}

impl From<List> for AttrList {
    fn from(value: List) -> Self {
        value.0
    }
}

impl AsRef<AttrList> for List {
    fn as_ref(&self) -> &AttrList {
        &self.0
    }
}

impl AsMut<AttrList> for List {
    fn as_mut(&mut self) -> &mut AttrList {
        &mut self.0
    }
}

impl AttrPrinter for ListAttr {
    fn print(&self, _printer: &mut AsmPrinter<'_>) {
        todo!()
    }
}
