mod debug;
mod instance;
mod interface;
mod memory;
mod speculation;

use core::fmt;

pub use self::{debug::*, instance::EffectInstance, interface::*, memory::*, speculation::*};
use crate::{DynPartialEq, any::AsAny, eq::PartialEqable};

/// A kind of side effect an operation can declare, such as [MemoryEffect] or [DebugEffect].
///
/// A dialect can define its own kinds, with their own interface and recursive marker, and walk
/// them with [RecursiveEffectIterator]. The generic passes (dead-code elimination, common
/// subexpression elimination, sinking, ..) ask about memory effects only, so an effect of another
/// kind does not keep an operation alive or stop it from being moved or merged. An effect that
/// those passes must respect is a [MemoryEffect] on a [Resource] of the dialect's own.
pub trait Effect: AsAny + fmt::Debug {
    /// Returns true if `op` declares that its effects of this kind include those of the operations
    /// nested within its regions.
    ///
    /// **THE MARKER AN OP ACTUALLY REGISTERS IS THE CONCRETE ONE.**
    /// `#[operation(traits(HasRecursiveMemoryEffects))]` puts `dyn HasRecursiveMemoryEffects` in
    /// the op's trait table. `dyn HasRecursiveEffects<MemoryEffect>` is a different `TypeId` that
    /// no op registers, even though the blanket impl of [`HasRecursiveEffects`] makes the bound
    /// hold — so asking for the generic form answered `false` for every op in the compiler. This
    /// hook asks for the marker the op really carries.
    fn is_recursive_on(op: &crate::Operation) -> bool
    where
        Self: Sized;
}

/// What an effect acts on: memory, the locals of a function, whether the program goes on, ..
///
/// Resources form a tree rooted at [DefaultResource]. Two effects can interfere only when their
/// resources are not disjoint (`is_disjoint_from`), that is, when one resource is
/// the other or one of its ancestors: an effect on the default resource interferes with every
/// other, and effects on two different children of the default resource never interfere.
///
/// Resources are compared by their type, so a resource is a unit struct.
pub trait Resource: AsAny + PartialEqable + DynPartialEq + fmt::Debug {
    fn name(&self) -> &'static str;

    /// The resource this one is part of. The default resource is the root and has none; every
    /// other resource is, by default, a direct child of the default resource.
    fn parent(&self) -> Option<&'static dyn Resource> {
        Some(&DefaultResource)
    }
}

impl dyn Resource {
    /// Returns true if this resource is `other`, or `other` is one of its ancestors.
    pub fn is_subresource_of(&self, other: &dyn Resource) -> bool {
        let mut current: Option<&dyn Resource> = Some(self);
        while let Some(resource) = current {
            if resource.dyn_eq(other) {
                return true;
            }
            current = resource.parent();
        }
        false
    }

    /// Returns true if neither resource is a subresource of the other: effects on them never
    /// interfere.
    pub fn is_disjoint_from(&self, other: &dyn Resource) -> bool {
        !self.is_subresource_of(other) && !other.is_subresource_of(self)
    }
}

/// The root of the tree of resources: an effect on it interferes with an effect on any resource.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DefaultResource;
impl Resource for DefaultResource {
    fn name(&self) -> &'static str {
        "default"
    }

    fn parent(&self) -> Option<&'static dyn Resource> {
        None
    }
}

/// An automatic allocation-scope resource that is valid in the context of a parent
/// AutomaticAllocationScope trait. A child of [DefaultResource].
#[derive(Debug, Default, PartialEq, Eq)]
pub struct AutomaticAllocationScopeResource;
impl Resource for AutomaticAllocationScopeResource {
    fn name(&self) -> &'static str {
        "automatic-allocation-scope"
    }
}

/// An operation trait for ops that are always speculatable and have no memory effects
pub trait Pure: AlwaysSpeculatable + MemoryEffectOpInterface {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A child of the default resource, for these tests.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Child;
    impl Resource for Child {
        fn name(&self) -> &'static str {
            "test.child"
        }
    }

    /// Another child of the default resource.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Sibling;
    impl Resource for Sibling {
        fn name(&self) -> &'static str {
            "test.sibling"
        }
    }

    /// A child of [Child].
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Grandchild;
    impl Resource for Grandchild {
        fn name(&self) -> &'static str {
            "test.grandchild"
        }

        fn parent(&self) -> Option<&'static dyn Resource> {
            Some(&Child)
        }
    }

    fn r(resource: &dyn Resource) -> &dyn Resource {
        resource
    }

    #[test]
    fn nothing_is_disjoint_from_the_default_resource() {
        for resource in [r(&DefaultResource), r(&Child), r(&Sibling), r(&Grandchild)] {
            assert!(resource.is_subresource_of(&DefaultResource), "{resource:?}");
            assert!(!resource.is_disjoint_from(&DefaultResource), "{resource:?}");
            assert!(!r(&DefaultResource).is_disjoint_from(resource), "{resource:?}");
        }
        assert!(r(&DefaultResource).parent().is_none());
        assert!(!r(&DefaultResource).is_subresource_of(&Child));
    }

    #[test]
    fn a_resource_is_not_disjoint_from_itself() {
        for resource in [r(&DefaultResource), r(&Child), r(&Grandchild)] {
            assert!(resource.is_subresource_of(resource), "{resource:?}");
            assert!(!resource.is_disjoint_from(resource), "{resource:?}");
        }
        assert!(r(&AutomaticAllocationScopeResource).is_subresource_of(&DefaultResource));
    }

    #[test]
    fn siblings_are_disjoint() {
        assert!(r(&Child).is_disjoint_from(&Sibling));
        assert!(r(&Sibling).is_disjoint_from(&Child));
        assert!(r(&Grandchild).is_disjoint_from(&Sibling));
        assert!(r(&Child).is_disjoint_from(&AutomaticAllocationScopeResource));
    }

    #[test]
    fn a_child_and_its_descendant_are_not_disjoint() {
        assert!(r(&Grandchild).is_subresource_of(&Child));
        assert!(!r(&Child).is_subresource_of(&Grandchild));
        assert!(!r(&Grandchild).is_disjoint_from(&Child));
        assert!(!r(&Child).is_disjoint_from(&Grandchild));
    }
}
