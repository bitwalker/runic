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

#[cfg(all(test, debug_assertions))]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::Context;

    #[test]
    #[should_panic(expected = "the value of a type attribute is `ptr<rec.0>`, an open type")]
    fn an_open_type_attribute_panics() {
        let context = Rc::new(Context::default());
        let open = context.ptr_type(context.rec_var(0));
        let _ = context.create_attribute::<TypeAttr, _>(open);
    }

    #[test]
    #[should_panic(expected = "the value of a type attribute is `i32`, a type of another context")]
    fn a_type_attribute_of_another_context_panics() {
        let (context, other) = (Rc::new(Context::default()), Context::default());
        let _ = context.create_attribute::<TypeAttr, _>(other.i32());
    }
}
