use alloc::rc::Rc;

use super::{DataLayoutSpec, Endianness, PointerSpec};
use crate::{
    Context, Op, Operation, Report, diagnostics::Spanned,
    dialects::builtin::attributes::DataLayoutAttr,
};

/// The name of the attribute in which an operation states its [DataLayoutSpec]: a
/// [DataLayoutAttr], `data_layout = #builtin.data_layout<..>` in an `attributes { .. }`
/// dictionary.
pub const DATA_LAYOUT_ATTR: &str = "data_layout";

/// Implemented by an operation that can state the [DataLayoutSpec] of the operations it holds:
/// `builtin.world`, `builtin.component` and `builtin.module`. [DataLayout::closest] finds the
/// nearest that states one.
///
/// The verifier rejects an attribute named [DATA_LAYOUT_ATTR] that is not a [DataLayoutAttr].
pub trait DataLayoutOpInterface: Op {
    /// The specification this operation states, if it states one. The default reads the attribute
    /// named [DATA_LAYOUT_ATTR] from the operation's attribute dictionary.
    fn data_layout_spec(&self) -> Option<DataLayoutSpec> {
        let attr = self.as_operation().get_typed_attribute::<DataLayoutAttr>(DATA_LAYOUT_ATTR)?;
        Some(attr.borrow().as_value().clone())
    }
}

impl<T> crate::Verify<dyn DataLayoutOpInterface> for T
where
    T: Op + DataLayoutOpInterface,
{
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let Some(attr) = op.get_attribute(DATA_LAYOUT_ATTR) else {
            return Ok(());
        };
        if attr.borrow().is::<DataLayoutAttr>() {
            return Ok(());
        }
        Err(crate::diagnostics::report!(
            message: (alloc::format!("invalid operation {}", op.name())),
            labels: [
                primary(
                    op.span(),
                    alloc::format!(
                        "the attribute '{DATA_LAYOUT_ATTR}' of this operation is not a \
                         '#builtin.data_layout'"
                    )
                ),
            ],
            notes: [
                help("required due to this operation implementing 'DataLayoutOpInterface'"),
            ],
        ))
    }
}

/// Makes `op` state `spec`: sets its attribute [DATA_LAYOUT_ATTR] to a new [DataLayoutAttr],
/// replacing any it has. An attribute may be shared, so the old one is never edited.
pub fn set_data_layout_spec(op: &mut Operation, spec: DataLayoutSpec) {
    let attr = op.context_rc().create_attribute::<DataLayoutAttr, _>(spec);
    op.set_attribute(DATA_LAYOUT_ATTR, attr.as_attribute_ref());
}

/// What a [Context] keeps for one [DataLayoutSpec]: the specification, shared by every
/// [DataLayout] made for an equal specification in that context.
#[derive(Debug)]
pub(crate) struct LayoutCache {
    spec: DataLayoutSpec,
}

impl LayoutCache {
    pub(crate) fn new(spec: DataLayoutSpec) -> Self {
        Self { spec }
    }
}

/// A [DataLayoutSpec] in effect, as found by [DataLayout::closest] or made by [DataLayout::new].
///
/// It holds its own copy of the specification, never the attribute that stated it, so editing or
/// replacing that attribute does not change it. Cloning it is cheap: every `DataLayout` made in a
/// context for equal specifications shares one copy and what is computed from it.
#[derive(Clone, Debug)]
pub struct DataLayout {
    cache: Rc<LayoutCache>,
}

impl DataLayout {
    /// The layout of `spec` in `context`. A context keeps one copy of each specification it has
    /// been asked for, so a later call for an equal specification allocates nothing.
    pub fn new(context: &Context, spec: &DataLayoutSpec) -> DataLayout {
        DataLayout {
            cache: context.data_layout_cache(spec),
        }
    }

    /// The layout stated by `op` or the nearest of its ancestors that states one, or `None` if
    /// none does.
    ///
    /// Only an operation implementing [DataLayoutOpInterface] states a specification; one that
    /// implements it without stating one is passed over. The nearest specification applies
    /// whole: specifications are never merged.
    pub fn closest(op: &Operation) -> Option<DataLayout> {
        if let Some(spec) = stated_spec(op) {
            return Some(DataLayout::new(op.context(), &spec));
        }
        let mut next = op.parent_op();
        while let Some(current) = next {
            let current = current.borrow();
            if let Some(spec) = stated_spec(&current) {
                return Some(DataLayout::new(current.context(), &spec));
            }
            next = current.parent_op();
        }
        None
    }

    /// True if `self` and `other` share the copy their context keeps for their specification.
    #[cfg(test)]
    pub(crate) fn shares_cache_with(&self, other: &DataLayout) -> bool {
        Rc::ptr_eq(&self.cache, &other.cache)
    }

    pub fn spec(&self) -> &DataLayoutSpec {
        &self.cache.spec
    }

    pub fn endianness(&self) -> Endianness {
        self.spec().endianness()
    }

    /// The pointer of `addrspace`; that of address space 0 if it has no entry of its own.
    pub fn pointer(&self, addrspace: u32) -> PointerSpec {
        self.spec().pointer(addrspace)
    }

    /// The width of `isize`, in bits: the index width of address space 0.
    pub fn isize_width(&self) -> u32 {
        self.spec().isize_width()
    }
}

/// The specification `op` states, if it implements [DataLayoutOpInterface] and states one.
fn stated_spec(op: &Operation) -> Option<DataLayoutSpec> {
    op.as_trait::<dyn DataLayoutOpInterface>()?.data_layout_spec()
}
