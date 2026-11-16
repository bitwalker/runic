use core::{
    any::{Any, TypeId},
    fmt,
};
use std::{collections::HashMap, sync::Arc};

/// Settings that belong to a frontend or a backend rather than to the harness, held by type.
///
/// Whoever builds the request sets a value of the frontend's or backend's options type, and the
/// frontend or backend reads it back when it runs; the harness never looks inside. There is one value per type: setting a type again replaces its
/// value. Cloning shares the values rather than copying them. The values are plain data: they can be
/// sent to and read from another thread, so that a request can be.
#[derive(Clone, Default)]
pub struct ExtensionOptions {
    values: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl ExtensionOptions {
    /// Options with no entries.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `value`, replacing any earlier value of the same type.
    pub fn set<T: Any + Send + Sync>(&mut self, value: T) {
        self.values.insert(TypeId::of::<T>(), Arc::new(value));
    }

    /// The value of type `T`, if one was set.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.values.get(&TypeId::of::<T>()).and_then(|value| value.downcast_ref::<T>())
    }
}

impl fmt::Debug for ExtensionOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtensionOptions({} entries)", self.values.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Wasm {
        features: u32,
    }

    #[test]
    fn a_value_set_is_read_back_by_its_type() {
        let mut options = ExtensionOptions::new();
        options.set(Wasm { features: 3 });
        options.set(7u8);
        assert_eq!(options.get::<Wasm>(), Some(&Wasm { features: 3 }));
        assert_eq!(options.get::<u8>(), Some(&7));
    }

    #[test]
    fn setting_a_type_again_replaces_its_value() {
        let mut options = ExtensionOptions::new();
        options.set(Wasm { features: 1 });
        options.set(Wasm { features: 2 });
        assert_eq!(options.get::<Wasm>(), Some(&Wasm { features: 2 }));
    }

    #[test]
    fn a_type_never_set_has_no_value() {
        let options = ExtensionOptions::new();
        assert_eq!(options.get::<Wasm>(), None);
    }

    #[test]
    fn a_clone_has_the_same_values_and_later_sets_do_not_cross() {
        let mut options = ExtensionOptions::new();
        options.set(Wasm { features: 1 });
        let mut copy = options.clone();
        copy.set(Wasm { features: 9 });
        assert_eq!(options.get::<Wasm>(), Some(&Wasm { features: 1 }));
        assert_eq!(copy.get::<Wasm>(), Some(&Wasm { features: 9 }));
    }

    #[test]
    fn debug_prints_the_number_of_entries() {
        let mut options = ExtensionOptions::new();
        assert_eq!(format!("{options:?}"), "ExtensionOptions(0 entries)");
        options.set(1u8);
        options.set(Wasm { features: 0 });
        assert_eq!(format!("{options:?}"), "ExtensionOptions(2 entries)");
    }
}
