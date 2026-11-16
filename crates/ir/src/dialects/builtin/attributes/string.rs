use crate::{
    AttrPrinter, AttributeRef,
    attributes::AttrParser,
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    parse::{ParseResult, Parser},
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
        let value = parser.parse_string()?.into_inner();
        Ok(parser.context_rc().create_attribute::<StringAttr, _>(value).as_attribute_ref())
    }
}
