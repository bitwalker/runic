use super::{Attribute, AttributeRef, AttributeRegistration};
use crate::{Int, parse};

/// When implemented on an attribute type, this function will be invoked to parse the contents
/// of the attribute data, if present.
pub trait AttrParser {
    fn parse(parser: &mut dyn parse::Parser<'_>) -> parse::ParseResult<AttributeRef>;
}

/// Implemented on attribute types which are markers, which have the following properties:
///
/// * They have no concrete value, i.e. `()`
/// * The presence of the attribute is what makes it significant
///
/// This is automatically derived for all attributes that have a value type of `()`.
///
/// Marker types are always uniqued.
pub trait Marker {}

impl<T> Marker for T where T: AttributeRegistration<Value = ()> {}

/// Implemented by every attribute whose value is an integer of a fixed width: [crate::IntAttr],
/// [crate::dialects::builtin::attributes::BoolAttr] and the per-width attributes `I8Attr` ...
/// `I128Attr`. It reads and writes the value as an [Int], whatever the attribute stores.
pub trait IntegerLikeAttr: Attribute {
    /// The value of this attribute.
    fn as_int(&self) -> Int;
    /// Replaces the value. Panics if `value` does not have this attribute's width.
    fn set_int(&mut self, value: Int);
}

/// Implemented by any attribute that represents a boolean value
pub trait BoolLikeAttr: Attribute {
    /// Get the boolean value of this attribute
    fn as_bool(&self) -> bool;
    /// Set the underlying boolean value of this attribute to `value`
    fn set_bool(&mut self, value: bool);
}
