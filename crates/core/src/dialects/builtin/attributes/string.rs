use crate::{
    AttrPrinter, AttributeRef, CompactString,
    attributes::AttrParser,
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    parse::{ParseResult, Parser, ParserError, unescape},
    print::AsmPrinter,
};

#[derive(DialectAttribute)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "crate::CompactString",
    implements(AttrPrinter),
)]
#[allow(unused)]
struct String;

impl AttrPrinter for StringAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_string(self.value.as_str());
    }
}

impl AttrParser for StringAttr {
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<AttributeRef> {
        let (span, quoted) = parser.parse_string()?.into_parts();
        let value = unescape(&quoted).ok_or_else(|| ParserError::InvalidString {
            span,
            reason: "invalid escape sequence: a string is escaped as by `str::escape_default`"
                .into(),
        })?;
        Ok(parser
            .context_rc()
            .create_attribute::<StringAttr, _>(CompactString::from(value))
            .as_attribute_ref())
    }
}
