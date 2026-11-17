use alloc::{string::ToString, sync::Arc};

use crate::{
    AttrPrinter, AttributeRef,
    attributes::{AttrParser, AttributeElements, ElementReplacer, ElementVisitor},
    constants::ConstantData,
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    parse::{ParseResult, Parser, ParserError},
};

/// A sequence of bytes, written as a string of hexadecimal digits: the data read as one
/// little-endian number, written as a hexadecimal number is written, most significant byte first.
/// The first byte of the data, the one at the lowest address, is the last pair of digits:
/// `#builtin.bytes<"0a0b0c0d">` holds the bytes `0d 0c 0b 0a` in memory order. The empty sequence
/// is `#builtin.bytes<"">`.
#[derive(DialectAttribute)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "Arc<ConstantData>",
    implements(AttrPrinter)
)]
pub struct Bytes;

/// A byte string holds no types or attributes.
impl AttributeElements for Arc<ConstantData> {
    #[inline]
    fn visit_elements(&self, _visitor: &mut dyn ElementVisitor) {}

    #[inline]
    fn replace_elements(&self, _replacer: &mut dyn ElementReplacer) -> Option<Self> {
        None
    }
}

impl AttrPrinter for BytesAttr {
    fn print(&self, printer: &mut crate::print::AsmPrinter<'_>) {
        printer.print_string(self.value.to_string());
    }
}

impl AttrParser for BytesAttr {
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<AttributeRef> {
        let digits = parser.parse_string()?;
        let data = ConstantData::from_str_le(digits.as_str()).map_err(|reason| {
            ParserError::InvalidAttributeValue {
                span: digits.span(),
                reason: reason.to_string(),
            }
        })?;
        Ok(parser.context_rc().create_attribute::<BytesAttr, _>(Arc::new(data)))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dialects::builtin::{Bytes, Module},
        testing::text::{parse, print_generic},
    };

    /// The digits of `#builtin.bytes<..>` are the data as one little-endian number, most
    /// significant byte first: `"0a0b0c0d"` is the bytes `0d 0c 0b 0a` in memory order, and prints
    /// as it was written.
    #[test]
    fn a_byte_string_is_written_most_significant_byte_first() {
        let source = "builtin.module public @m {
    builtin.global_variable public @data : array<i8; 4> {
        %b = \"builtin.bytes\"() <{ value = #builtin.bytes<\"0a0b0c0d\"> }> : () -> array<i8; 4>;
        builtin.yield %b;
    };
};";
        let (_context, module) = parse::<Module>(source).unwrap_or_else(|err| panic!("{err}"));
        let mut found = None;
        module.as_operation_ref().borrow().prewalk_all(|op: &crate::Operation| {
            if let Some(bytes) = op.downcast_ref::<Bytes>() {
                found = Some((bytes.data(), print_generic(op)));
            }
        });
        let (data, generic) = found.expect("a bytes operation");
        assert_eq!(data.as_slice(), [0x0d, 0x0c, 0x0b, 0x0a]);
        assert!(generic.contains("value = #builtin.bytes<\"0a0b0c0d\">"), "{generic}");
    }
}
