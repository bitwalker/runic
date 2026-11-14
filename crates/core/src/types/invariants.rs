//! The checks a type kind makes of the types its value holds.

use crate::{Context, Report};

/// Implemented by a type kind whose value can be made invalid by replacing the types it holds.
///
/// [TypeImpl::replace_subtypes](super::TypeImpl::replace_subtypes) does not check the value it
/// builds. Code that rebuilds a type with other held types, such as [TypeMapper](super::TypeMapper),
/// asks the rebuilt type for this interface ([Type::as_trait](super::Type::as_trait)) and checks
/// it. A kind is declared to implement it in the `implements(..)` of `#[derive(DialectType)]`.
pub trait TypeInvariants {
    /// Whether this value is well formed given the types it now holds.
    fn check_invariants(&self, context: &Context) -> Result<(), Report>;
}
