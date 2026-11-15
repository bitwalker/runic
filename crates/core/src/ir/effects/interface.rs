use smallvec::SmallVec;

use super::*;
use crate::{OperationRef, SymbolRef, ValueRef};

/// Marker trait for ops with recursive effects of a given [Effect] type.
///
/// Ops with recursive effects are considered to have any of the effects of operations nested within
/// its regions, in addition to any effects it declares on itself. Only when the operation and none
/// of its nested operations carry effects of the given type, can it be assumed that the operation
/// is free of that effect.
pub trait HasRecursiveEffects<T: Effect> {}

impl<T: HasRecursiveMemoryEffects> HasRecursiveEffects<MemoryEffect> for T {}

pub trait EffectOpInterface<T: Effect> {
    /// Return the set all of the operation's effects
    ///
    /// An operation reports its own effects only. The effects of the operations in its regions
    /// are found by the recursive walk ([RecursiveEffectIterator],
    /// [Operation::get_effects_recursively](crate::Operation::get_effects_recursively)) when the
    /// operation has the `HasRecursive*` marker for this kind of effect (for memory effects,
    /// [HasRecursiveMemoryEffects]). Reporting them here as well would count them twice, and,
    /// since the walk asks every operation it visits for its effects, an implementation that
    /// starts the walk at its own operation would recurse without bound.
    fn effects(&self) -> EffectIterator<T>;
    /// False when the operation cannot list its own effects. `effects()` is not consulted.
    ///
    /// The operation's own effects are then unknown, so the operation as a whole is never
    /// effect-free, whatever its regions hold. This differs from an operation without this
    /// interface when the operation has [HasRecursiveMemoryEffects]: the effects of such an
    /// operation without the interface are those of its regions, which may be known.
    ///
    /// The recursive walk ([RecursiveEffectIterator]) reports such an operation as "unknown", and
    /// so do the queries on [Operation](crate::Operation) built on it. A caller that asks for
    /// [Self::effects] directly must check this first.
    fn effects_are_known(&self) -> bool {
        true
    }
    /// Returns true if this operation is known to have no effects: false when its effects are
    /// unknown.
    fn has_no_effect(&self) -> bool {
        self.effects_are_known() && self.effects().is_empty()
    }
    /// Return the set of effect instances that operate on the provided value
    ///
    /// Only meaningful when [Self::effects_are_known] is true. An effect that names no value is an
    /// effect on its resource in general, so on any value: it is not listed here, and code that
    /// reasons per value must treat it as applying to every value.
    fn effects_on_value(&self, value: ValueRef) -> ValueEffectIterator<T> {
        EffectIterator::for_value(self.effects(), value)
    }
    /// Return the set of effect instances that operate on the provided symbol
    ///
    /// Only meaningful when [Self::effects_are_known] is true.
    fn effects_on_symbol(&self, symbol: SymbolRef) -> SymbolEffectIterator<T> {
        EffectIterator::for_symbol(self.effects(), symbol)
    }
    /// Return the set of effect instances that operate on the provided resource
    ///
    /// Only meaningful when [Self::effects_are_known] is true.
    fn effects_on_resource<'a, 'b: 'a>(
        &self,
        resource: &'b dyn Resource,
    ) -> ResourceEffectIterator<'b, T> {
        EffectIterator::for_resource(self.effects(), resource)
    }
}

impl<T: Effect> dyn EffectOpInterface<T> {
    /// Return the set all of the operation's effects that correspond to effect type `T`
    ///
    /// Empty when the operation's effects are unknown: check
    /// [EffectOpInterface::effects_are_known] first.
    pub fn effects_of_type<E>(&self) -> impl Iterator<Item = EffectInstance<T>> + '_
    where
        E: Effect,
    {
        let known = self.effects_are_known();
        self.effects()
            .filter(move |instance| known && instance.effect().as_any().is::<E>())
    }

    /// Returns true if the operation is known to exhibit the given effect: false when its effects
    /// are unknown.
    pub fn has_effect<E>(&self) -> bool
    where
        E: Effect,
    {
        self.effects_are_known()
            && self.effects().any(|instance| instance.effect().as_any().is::<E>())
    }

    /// Returns true if the operation is known to exhibit only the given effect.
    pub fn only_has_effect<E>(&self) -> bool
    where
        E: Effect,
    {
        if !self.effects_are_known() {
            return false;
        }
        let mut effects = self.effects();
        !effects.is_empty() && effects.all(|instance| instance.effect().as_any().is::<E>())
    }
}

pub struct EffectIterator<T> {
    effects: smallvec::IntoIter<[EffectInstance<T>; 4]>,
}
impl<T> EffectIterator<T> {
    pub fn from_smallvec(effects: SmallVec<[EffectInstance<T>; 4]>) -> Self {
        Self {
            effects: effects.into_iter(),
        }
    }

    pub fn new(effects: impl IntoIterator<Item = EffectInstance<T>>) -> Self {
        let effects = effects.into_iter().collect::<SmallVec<[_; 4]>>();
        Self {
            effects: effects.into_iter(),
        }
    }

    pub const fn for_value(effects: Self, value: ValueRef) -> ValueEffectIterator<T> {
        ValueEffectIterator {
            iter: effects,
            value,
        }
    }

    pub const fn for_symbol(effects: Self, symbol: SymbolRef) -> SymbolEffectIterator<T> {
        SymbolEffectIterator {
            iter: effects,
            symbol,
        }
    }

    pub const fn for_resource(
        effects: Self,
        resource: &dyn Resource,
    ) -> ResourceEffectIterator<'_, T> {
        ResourceEffectIterator {
            iter: effects,
            resource,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[EffectInstance<T>] {
        self.effects.as_slice()
    }
}
impl<T> core::iter::FusedIterator for EffectIterator<T> {}
impl<T> ExactSizeIterator for EffectIterator<T> {
    fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    fn len(&self) -> usize {
        self.effects.len()
    }
}
impl<T> Iterator for EffectIterator<T> {
    type Item = EffectInstance<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.effects.next()
    }
}

pub struct ValueEffectIterator<T> {
    iter: EffectIterator<T>,
    value: ValueRef,
}
impl<T> core::iter::FusedIterator for ValueEffectIterator<T> {}
impl<T> Iterator for ValueEffectIterator<T> {
    type Item = EffectInstance<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(instance) = self.iter.next() {
            if instance.value().is_some_and(|v| v == self.value) {
                return Some(instance);
            }
        }

        None
    }
}

pub struct SymbolEffectIterator<T> {
    iter: EffectIterator<T>,
    symbol: SymbolRef,
}
impl<T> core::iter::FusedIterator for SymbolEffectIterator<T> {}
impl<T> Iterator for SymbolEffectIterator<T> {
    type Item = EffectInstance<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(instance) = self.iter.next() {
            if instance.symbol().is_some_and(|s| s == self.symbol) {
                return Some(instance);
            }
        }

        None
    }
}

pub struct ResourceEffectIterator<'a, T> {
    iter: EffectIterator<T>,
    resource: &'a dyn Resource,
}
impl<T> core::iter::FusedIterator for ResourceEffectIterator<'_, T> {}
impl<T> Iterator for ResourceEffectIterator<'_, T> {
    type Item = EffectInstance<T>;

    fn next(&mut self) -> Option<Self::Item> {
        #[allow(clippy::while_let_on_iterator)]
        while let Some(instance) = self.iter.next() {
            if instance.resource().dyn_eq(self.resource) {
                return Some(instance);
            }
        }

        None
    }
}

/// An iterator over the recursive effects of an [crate::Operation].
///
/// The value produced by the iterator is `(OperationRef, Option<EffectInterface<T>>)`, where the
/// operation reference is the effecting op, and the second element is the identified effect:
///
/// * `Some` represents an effect on the operation or one of its nested operations
/// * `None` indicates that we have identified that the given operation has unknown effects, and
///   thus the entire operation could have unknown effects. An operation has unknown effects when
///   it implements neither [EffectOpInterface] nor the recursive marker, or when it implements the
///   interface and [EffectOpInterface::effects_are_known] is false.
///
/// Note that in the case of discovering that an operation has unknown effects, the iterator can
/// continue to visit all effects recursively - it is up to the caller to stop iteration if the
/// presence of unknown effects makes further search wasteful.
pub struct RecursiveEffectIterator<T> {
    buffer: crate::adt::SmallDeque<(OperationRef, Option<EffectInstance<T>>), 2>,
    effecting_ops: SmallVec<[OperationRef; 4]>,
}

impl<T: Effect> RecursiveEffectIterator<T> {
    /// Iterate over the recursive effects of `op`
    pub fn new(op: OperationRef) -> Self {
        Self {
            buffer: Default::default(),
            effecting_ops: SmallVec::from_iter([op]),
        }
    }
}

impl<T: Effect> core::iter::FusedIterator for RecursiveEffectIterator<T> {}

impl<T: Effect> Iterator for RecursiveEffectIterator<T> {
    type Item = (OperationRef, Option<EffectInstance<T>>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(next) = self.buffer.pop_front() {
                return Some(next);
            }

            if let Some(op) = self.effecting_ops.pop() {
                let operation = op.borrow();

                let has_recursive_effects = T::is_recursive_on(&operation)
                    || operation.implements::<dyn HasRecursiveEffects<T>>();
                if has_recursive_effects {
                    for region in operation.regions() {
                        for block in region.body() {
                            let mut next = block.body().front().as_pointer();
                            while let Some(nested) = next.take() {
                                next = nested.next();
                                self.effecting_ops.push(nested);
                            }
                        }
                    }
                }

                match operation.as_trait::<dyn EffectOpInterface<T>>() {
                    Some(effect_interface) if effect_interface.effects_are_known() => {
                        self.buffer.extend(effect_interface.effects().map(|eff| (op, Some(eff))));
                    }
                    // The operation says it cannot list its own effects: they are unknown,
                    // whether or not it also has recursive effects
                    Some(_) => self.buffer.push_back((op, None)),
                    // The operation does not have recursive effects or implement
                    // EffectOpInterface, so its effects are unknown.
                    None if !has_recursive_effects => self.buffer.push_back((op, None)),
                    None => (),
                }
            } else {
                break;
            }
        }

        None
    }
}

#[cfg(test)]
mod derive_tests {
    use alloc::{rc::Rc, vec::Vec};

    use super::*;
    use crate::Context;

    #[derive(crate::derive::EffectOpInterface)]
    #[effects(MemoryEffect(MemoryEffect::Read))]
    #[effects(MemoryEffect(MemoryEffect::Allocate))]
    struct CombinedEffects {
        #[effects(MemoryEffect(MemoryEffect::Write))]
        #[effects(MemoryEffect(MemoryEffect::Free))]
        value: ValueRef,
    }

    impl CombinedEffects {
        fn value(&self) -> ValueRef {
            self.value
        }
    }

    #[derive(crate::derive::EffectOpInterface)]
    #[effects(MemoryEffect())]
    struct EmptyEffects {}

    #[test]
    fn effect_derive_combines_global_field_and_repeated_declarations() {
        let context = Rc::new(Context::default());
        let block = context.create_block_with_params([context.i32()]);
        let value = block.borrow().arguments()[0] as ValueRef;
        let op = CombinedEffects { value };
        let effects = op
            .effects()
            .map(|instance| (*instance.effect(), instance.value()))
            .collect::<Vec<_>>();
        assert!(!op.has_no_effect());
        assert_eq!(effects.len(), 4);
        for expected in [
            (MemoryEffect::Read, None),
            (MemoryEffect::Allocate, None),
            (MemoryEffect::Write, Some(value)),
            (MemoryEffect::Free, Some(value)),
        ] {
            assert!(effects.contains(&expected), "missing declared effect {expected:?}");
        }
    }

    #[test]
    fn effect_derive_reports_empty_declarations_as_effect_free() {
        let op = EmptyEffects {};
        assert_eq!(op.effects().count(), 0);
        assert!(op.has_no_effect());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Builder, BuilderExt, Op, SourceSpan, ValueRef,
        dialects::test::{Store, TestDialect},
        testing::Test,
        traits::SingleBlock,
    };

    /// A second kind of effect, local to this test: the effect machinery is generic over the
    /// effect kind, and dialects define their own.
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum PingEffect {
        Ping,
    }

    impl Effect for PingEffect {
        fn is_recursive_on(op: &crate::Operation) -> bool {
            op.implements::<dyn HasRecursivePingEffects>()
        }
    }

    trait PingEffectOpInterface = EffectOpInterface<PingEffect>;

    trait HasRecursivePingEffects {}
    impl<T: HasRecursivePingEffects> HasRecursiveEffects<PingEffect> for T {}

    #[derive(crate::derive::EffectOpInterface)]
    #[effects(MemoryEffect())]
    #[effects(PingEffect())]
    #[crate::derive::operation(
        dialect = TestDialect,
        traits(SingleBlock, HasRecursiveMemoryEffects, HasRecursivePingEffects),
        implements(MemoryEffectOpInterface, PingEffectOpInterface)
    )]
    pub struct RecursiveRegion {
        #[region]
        body: Region,
    }

    /// An op whose only effect is a `PingEffect`, and which says nothing about memory effects.
    #[derive(crate::derive::EffectOpInterface)]
    #[effects(PingEffect(PingEffect::Ping))]
    #[crate::derive::operation(
        dialect = TestDialect,
        implements(PingEffectOpInterface)
    )]
    pub struct Ping {}

    /// **A DECLARED RECURSIVE-EFFECTS MARKER MUST BE VISIBLE TO THE WALK.**
    ///
    /// `#[operation(traits(HasRecursiveMemoryEffects))]` registers `dyn HasRecursiveMemoryEffects`
    /// in the op's trait table; the walk asked for `dyn HasRecursiveEffects<T>`, which is a
    /// DIFFERENT `TypeId` and is registered by nobody. So the query answered `false` for every op
    /// in the compiler, the walk never descended into a single region, and `scf.if` — whose whole
    /// effect story is its body's — was reported as having UNKNOWN effects instead of its body's.
    #[test]
    fn a_recursive_effects_op_is_walked_into() {
        let mut test = Test::named("recursive_marker_is_visible");
        let i32 = test.context().i32();
        let pointer = test.context().ptr_type(i32);
        test.with_function("recursive_marker_is_visible", &[pointer, i32], &[]);
        let region_op = {
            let mut builder = test.function_builder();
            let entry = builder.entry_block();
            let (addr, value) = {
                let block = entry.borrow();
                (block.arguments()[0] as ValueRef, block.arguments()[1] as ValueRef)
            };
            let builder = builder.builder_mut();
            builder.set_insertion_point_to_end(entry);
            let op = builder.create::<RecursiveRegion, ()>(SourceSpan::UNKNOWN)().unwrap();
            let body = op.borrow().body().as_region_ref();
            let block = builder.create_block(body, None, &[]);
            builder.set_insertion_point_to_end(block);
            builder.create::<Store, _>(SourceSpan::UNKNOWN)(addr, value).unwrap();
            op.borrow().as_operation().as_operation_ref()
        };

        let effects = RecursiveEffectIterator::<MemoryEffect>::new(region_op)
            .map(|(_, effect)| effect)
            .collect::<alloc::vec::Vec<_>>();

        assert_eq!(
            effects.len(),
            1,
            "the walk must report the body's one effect, and never 'unknown': {effects:?}"
        );
        assert_eq!(
            effects[0].as_ref().map(|instance| *instance.effect()),
            Some(MemoryEffect::Write),
            "the nested store's write is the op's recursive effect"
        );
    }

    /// Two effect kinds are walked independently over the same region: each kind sees its own
    /// effects, and an op that says nothing about a kind is "unknown" for that kind only.
    #[test]
    fn the_walk_is_generic_over_the_effect_kind() {
        let mut test = Test::named("two_effect_kinds");
        let i32 = test.context().i32();
        let pointer = test.context().ptr_type(i32);
        test.with_function("two_effect_kinds", &[pointer, i32], &[]);
        let region_op = {
            let mut builder = test.function_builder();
            let entry = builder.entry_block();
            let (addr, value) = {
                let block = entry.borrow();
                (block.arguments()[0] as ValueRef, block.arguments()[1] as ValueRef)
            };
            let builder = builder.builder_mut();
            builder.set_insertion_point_to_end(entry);
            let op = builder.create::<RecursiveRegion, ()>(SourceSpan::UNKNOWN)().unwrap();
            let body = op.borrow().body().as_region_ref();
            let block = builder.create_block(body, None, &[]);
            builder.set_insertion_point_to_end(block);
            builder.create::<Store, _>(SourceSpan::UNKNOWN)(addr, value).unwrap();
            builder.create::<Ping, ()>(SourceSpan::UNKNOWN)().unwrap();
            op.borrow().as_operation().as_operation_ref()
        };

        let memory = RecursiveEffectIterator::<MemoryEffect>::new(region_op)
            .map(|(_, effect)| effect.map(|instance| *instance.effect()))
            .collect::<alloc::vec::Vec<_>>();
        assert_eq!(memory.len(), 2, "{memory:?}");
        assert!(memory.contains(&Some(MemoryEffect::Write)), "{memory:?}");
        assert!(memory.contains(&None), "the ping op's memory effects are unknown: {memory:?}");

        let ping = RecursiveEffectIterator::<PingEffect>::new(region_op)
            .map(|(_, effect)| effect.map(|instance| *instance.effect()))
            .collect::<alloc::vec::Vec<_>>();
        assert_eq!(ping.len(), 2, "{ping:?}");
        assert!(ping.contains(&Some(PingEffect::Ping)), "{ping:?}");
        assert!(ping.contains(&None), "the store's ping effects are unknown: {ping:?}");
    }

    /// A `test.effects <{ known = .. }>` in a new function, unused, directly in the entry block or
    /// inside a [RecursiveRegion]. Returns the operation the queries are asked of: the
    /// `test.effects`, or the region operation that holds it.
    fn effects_op(known: bool, nested: bool) -> (Test, crate::OperationRef) {
        use crate::dialects::test::TestOpBuilder;

        let mut test = Test::named("effects_op");
        test.with_function("effects_op", &[], &[]);
        let op = {
            let mut builder = test.function_builder();
            let entry = builder.entry_block();
            let builder = builder.builder_mut();
            builder.set_insertion_point_to_end(entry);
            if nested {
                let op = builder.create::<RecursiveRegion, ()>(SourceSpan::UNKNOWN)().unwrap();
                let body = op.borrow().body().as_region_ref();
                let block = builder.create_block(body, None, &[]);
                builder.set_insertion_point_to_end(block);
                builder.effects(known, SourceSpan::UNKNOWN).unwrap();
                op.borrow().as_operation().as_operation_ref()
            } else {
                let result = builder.effects(known, SourceSpan::UNKNOWN).unwrap();
                result.borrow().get_defining_op().unwrap()
            }
        };
        (test, op)
    }

    /// An operation that implements the interface and says its effects are unknown is treated as
    /// one without the interface by every query, alone and inside a region with recursive effects.
    #[test]
    fn unknown_effects_from_an_operation_with_the_interface() {
        for nested in [false, true] {
            let (_test, op) = effects_op(false, nested);
            let op = op.borrow();
            assert!(!op.is_memory_effect_free(), "nested: {nested}");
            assert!(!op.is_used());
            assert!(!op.would_be_trivially_dead(), "nested: {nested}");
            assert!(op.get_effects_recursively::<MemoryEffect>().is_none(), "nested: {nested}");
            assert!(!op.has_single_memory_effect(MemoryEffect::Read), "nested: {nested}");
            assert!(!op.has_memory_effect(MemoryEffect::Read), "nested: {nested}");
            let walked = RecursiveEffectIterator::<MemoryEffect>::new(op.as_operation_ref())
                .map(|(_, effect)| effect.map(|instance| *instance.effect()))
                .collect::<alloc::vec::Vec<_>>();
            assert_eq!(walked, [None], "nested: {nested}");
        }
    }

    /// The same operation with known effects (a single `Read`) is dead when unused, and its one
    /// effect is a `Read`.
    #[test]
    fn known_effects_from_the_same_operation() {
        for nested in [false, true] {
            let (_test, op) = effects_op(true, nested);
            let op = op.borrow();
            assert!(!op.is_memory_effect_free(), "nested: {nested}");
            assert!(op.would_be_trivially_dead(), "nested: {nested}");
            let effects = op
                .get_effects_recursively::<MemoryEffect>()
                .map(|effects| effects.iter().map(|e| *e.effect()).collect::<alloc::vec::Vec<_>>());
            assert_eq!(effects, Some(alloc::vec![MemoryEffect::Read]), "nested: {nested}");
            // The single-effect queries consult the operation's own effects only
            assert_eq!(op.has_single_memory_effect(MemoryEffect::Read), !nested);
        }
    }
}
