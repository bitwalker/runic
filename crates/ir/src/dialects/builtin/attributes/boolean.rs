use crate::{
    AttrPrinter, AttributeRef, Int,
    attributes::{AttrParser, BoolLikeAttr, IntegerLikeAttr},
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    parse::{ParseResult, Parser, Token},
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
}

impl IntegerLikeAttr for BoolAttr {
    #[inline]
    fn as_int(&self) -> Int {
        Int::from(self.value)
    }
}

impl AttrPrinter for BoolAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_bool(self.value);
    }
}

impl AttrParser for BoolAttr {
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<AttributeRef> {
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
