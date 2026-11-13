use alloc::{string::ToString, sync::Arc};

use crate::{
    AttrPrinter, AttributeRef,
    attributes::AttrParser,
    constants::ConstantData,
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    parse::{ParseResult, Parser, ParserError},
};

/// A sequence of bytes, written as a string of hexadecimal digits that gives the last byte first:
/// `#builtin.bytes<"0201">` holds the bytes `1, 2`. The empty sequence is `#builtin.bytes<"">`.
#[derive(DialectAttribute)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "Arc<ConstantData>",
    implements(AttrPrinter)
)]
pub struct Bytes;

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
