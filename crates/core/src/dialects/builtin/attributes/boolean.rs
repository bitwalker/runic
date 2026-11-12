use crate::{
    AttrPrinter, Int,
    attributes::{AttrParser, BoolLikeAttr, IntegerLikeAttr},
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    parse::Token,
    print::AsmPrinter,
};

#[derive(DialectAttribute)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "bool",
    implements(BoolLikeAttr, IntegerLikeAttr, AttrPrinter)
)]
#[allow(unused)]
struct Bool;

impl BoolLikeAttr for BoolAttr {
    #[inline(always)]
    fn as_bool(&self) -> bool {
        self.value
    }

    #[inline(always)]
    fn set_bool(&mut self, value: bool) {
        self.value = value;
    }
}

impl IntegerLikeAttr for BoolAttr {
    #[inline]
    fn as_int(&self) -> Int {
        Int::from(self.value)
    }

    fn set_int(&mut self, value: Int) {
        assert_eq!(value.width(), 1, "cannot set a {}-bit value on an i1 attribute", value.width());
        self.value = !value.is_zero();
    }
}

impl AttrPrinter for BoolAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_bool(self.value);
    }
}

impl AttrParser for BoolAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        let value = parser.token_stream_mut().expect_map("'true' or 'false'", |tok| match tok {
            Token::True => Some(true),
            Token::False => Some(false),
            _ => None,
        })?;
        Ok(parser
            .context_rc()
            .create_attribute::<BoolAttr, _>(value.into_inner())
            .as_attribute_ref())
    }
}
