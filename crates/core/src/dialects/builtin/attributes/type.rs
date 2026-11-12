use crate::{
    AttrPrinter, attributes::AttrParser, derive::DialectAttribute,
    dialects::builtin::BuiltinDialect, print::AsmPrinter,
};

#[derive(DialectAttribute, Debug, Clone, PartialEq, Eq, Hash)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "crate::Type",
    default = "default_type",
    implements(AttrPrinter),
)]
#[allow(unused)]
struct Type;

impl AttrPrinter for TypeAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_type(self.value);
    }
}

impl AttrParser for TypeAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        let ty = parser.parse_type()?.into_inner();

        Ok(parser.context_rc().create_attribute::<TypeAttr, _>(ty))
    }
}

fn default_type(context: &crate::Context) -> crate::Type {
    context.unknown_type()
}
