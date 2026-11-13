use runic_core::{
    AttrPrinter, Context, Type, attributes::AttrParser, derive::DialectAttribute, print::AsmPrinter,
};

use crate::UndefinedBehaviorDialect;

/// The value of a `ub.poison` operation: the type of the poison value it produces.
///
/// Text: `#ub.poison<i32>`; a body-less `#ub.poison` is poison of the unknown type.
#[derive(DialectAttribute, Debug, Clone, PartialEq, Eq, Hash)]
#[attribute(
    dialect = UndefinedBehaviorDialect,
    remote = "Type",
    default = "poison_type",
    implements(AttrPrinter)
)]
#[allow(unused)]
struct Poison;

/// The value of a body-less `#ub.poison`.
fn poison_type(context: &Context) -> Type {
    context.unknown_type()
}

impl AttrPrinter for PoisonAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_type(self.value);
    }
}

impl AttrParser for PoisonAttr {
    fn parse(
        parser: &mut dyn runic_core::parse::Parser<'_>,
    ) -> runic_core::parse::ParseResult<runic_core::AttributeRef> {
        let ty = parser.parse_type()?.into_inner();
        Ok(parser.context_rc().create_attribute::<PoisonAttr, _>(ty))
    }
}
