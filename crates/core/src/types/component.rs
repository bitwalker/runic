//! [TypeComponent]: the parts of a type's value that may hold types.

use alloc::{boxed::Box, string::String, vec::Vec};
use core::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize};

use compact_str::CompactString;
use smallvec::SmallVec;

use super::Type;
use crate::{CallConv, interner::Symbol};

/// A part of a type's value that may hold types.
///
/// [TypeImpl](super::TypeImpl) values are walked through their components:
/// `#[derive(DialectType)]` visits and maps each field of a type with these methods, and
/// `#[derive(TypeComponent)]` implements them for a struct or enum used as such a field.
///
/// Both methods must be pure: they must not intern types, use the context, or panic.
pub trait TypeComponent: Sized {
    /// Calls `visit` with each type this component holds, in a fixed order.
    fn visit_types(&self, visit: &mut dyn FnMut(Type));

    /// A copy of this component in which each type `t` it holds is `map(t)`; `map` is called in
    /// the order of [Self::visit_types], and everything else is unchanged.
    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self;
}

impl TypeComponent for Type {
    #[inline]
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        visit(*self);
    }

    #[inline]
    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        map(*self)
    }
}

/// Implements [TypeComponent] for data that holds no types: it visits nothing and maps to a clone.
macro_rules! plain_components {
    ($($ty:ty),* $(,)?) => {
        $(
            impl TypeComponent for $ty {
                #[inline]
                fn visit_types(&self, _visit: &mut dyn FnMut(Type)) {}

                #[inline]
                fn map_types(&self, _map: &mut dyn FnMut(Type) -> Type) -> Self {
                    self.clone()
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
);

impl<T: TypeComponent> TypeComponent for Option<T> {
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        if let Some(value) = self {
            value.visit_types(visit);
        }
    }

    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        self.as_ref().map(|value| value.map_types(map))
    }
}

impl<T: TypeComponent> TypeComponent for Box<T> {
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        (**self).visit_types(visit);
    }

    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        Box::new((**self).map_types(map))
    }
}

impl<T: TypeComponent> TypeComponent for Box<[T]> {
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        for element in self.iter() {
            element.visit_types(visit);
        }
    }

    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        self.iter().map(|element| element.map_types(map)).collect()
    }
}

impl<T: TypeComponent> TypeComponent for Vec<T> {
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        for element in self {
            element.visit_types(visit);
        }
    }

    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        self.iter().map(|element| element.map_types(map)).collect()
    }
}

impl<T: TypeComponent, const N: usize> TypeComponent for SmallVec<[T; N]> {
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        for element in self {
            element.visit_types(visit);
        }
    }

    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        self.iter().map(|element| element.map_types(map)).collect()
    }
}

impl<T: TypeComponent, const N: usize> TypeComponent for [T; N] {
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        for element in self {
            element.visit_types(visit);
        }
    }

    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        // `from_fn` calls the closure for each index in increasing order.
        core::array::from_fn(|index| self[index].map_types(map))
    }
}

impl<A: TypeComponent, B: TypeComponent> TypeComponent for (A, B) {
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        self.0.visit_types(visit);
        self.1.visit_types(visit);
    }

    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        let a = self.0.map_types(map);
        let b = self.1.map_types(map);
        (a, b)
    }
}

impl<A: TypeComponent, B: TypeComponent, C: TypeComponent> TypeComponent for (A, B, C) {
    fn visit_types(&self, visit: &mut dyn FnMut(Type)) {
        self.0.visit_types(visit);
        self.1.visit_types(visit);
        self.2.visit_types(visit);
    }

    fn map_types(&self, map: &mut dyn FnMut(Type) -> Type) -> Self {
        let a = self.0.map_types(map);
        let b = self.1.map_types(map);
        let c = self.2.map_types(map);
        (a, b, c)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;
    use crate::Context;

    fn visited<C: TypeComponent>(component: &C) -> Vec<Type> {
        let mut types = Vec::new();
        component.visit_types(&mut |ty| types.push(ty));
        types
    }

    #[test]
    fn an_option_visits_only_what_it_holds_in_order() {
        let context = Context::default();
        let (i8, i16, i32) = (context.i8(), context.i16(), context.i32());
        let component = vec![Some(i8), None, Some(i16), None, Some(i32)];
        assert_eq!(visited(&component), [i8, i16, i32]);
    }

    #[test]
    fn mapping_replaces_every_type_position_and_keeps_the_rest() {
        let context = Context::default();
        let (i8, i16, i32, i64) = (context.i8(), context.i16(), context.i32(), context.i64());
        let component: (Type, u32, Box<[Type]>) = (i8, 7, Box::from([i16, i8]));
        let mut seen = Vec::new();
        let mapped = component.map_types(&mut |ty| {
            seen.push(ty);
            if ty == i8 { i32 } else { i64 }
        });
        assert_eq!(seen, [i8, i16, i8], "map is called in visit order");
        assert_eq!(mapped, (i32, 7, Box::from([i64, i32])));
    }

    #[test]
    fn plain_data_visits_nothing_and_maps_to_an_equal_value() {
        let value = (
            Symbol::intern("x"),
            CompactString::from("y"),
            (NonZeroU16::new(4), CallConv::c(), [1u8, 2, 3]),
        );
        assert!(visited(&value).is_empty());
        let mapped = value.map_types(&mut |_| -> Type { unreachable!("nothing to map") });
        assert_eq!(mapped, value);
    }

    #[test]
    fn containers_map_element_by_element() {
        let context = Context::default();
        let (i8, i64) = (context.i8(), context.i64());
        let small: SmallVec<[Type; 2]> = SmallVec::from_slice(&[i8, i8, i8]);
        let mapped = small.map_types(&mut |_| i64);
        assert_eq!(mapped.as_slice(), [i64, i64, i64]);
        let array = [Some(Box::new(i8)), None];
        assert_eq!(visited(&array), [i8]);
        assert_eq!(array.map_types(&mut |_| i64), [Some(Box::new(i64)), None]);
    }
}
