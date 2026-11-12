use core::fmt::{self, Write};

use crate::{
    TypeFormatter, TypeParser, TypePrinter,
    derive::DialectType,
    dialects::test::TestDialect,
    parse::{ParseResult, Parser, ParserError, ParserExt, Token},
    types::Type,
};

/// A fixed number of lanes of one element type, `!test.vector<T x N>`.
///
/// It exists to exercise dialect types end to end: registration, printing a parameterised body,
/// and parsing that body back.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = TestDialect, name = "vector")]
pub struct TestVectorType {
    element: Type,
    lanes: u32,
}

impl TestVectorType {
    pub fn new(element: Type, lanes: u32) -> Self {
        Self { element, lanes }
    }

    pub fn element(&self) -> Type {
        self.element
    }

    pub fn lanes(&self) -> u32 {
        self.lanes
    }
}

impl TypePrinter for TestVectorType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        printer.ty(self.element)?;
        write!(printer, " x {}", self.lanes)
    }
}

impl TypeParser for TestVectorType {
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<Type> {
        let element = parser.parse_type()?.into_inner();
        if !parser.token_stream_mut().next_if_eq(Token::BareIdent("x"))? {
            let token = parser.token_stream_mut().next()?;
            return Err(match token {
                Some(token) => ParserError::UnexpectedToken {
                    span: token.span(),
                    token: alloc::string::ToString::to_string(&token),
                    expected: Some("'x'".into()),
                },
                None => ParserError::UnexpectedEof {
                    expected: alloc::vec!["'x'".into()],
                },
            });
        }
        let (span, lanes) = parser.parse_decimal_integer::<u32>()?.into_parts();
        if lanes == 0 {
            return Err(ParserError::InvalidIntegerLiteral {
                span,
                reason: "a vector has at least one lane".into(),
            });
        }
        Ok(parser.context().intern_type(Self::new(element, lanes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn a_vector_prints_its_body_after_the_dialect_prefix() {
        let context = Context::default();
        let vector = context.intern_type(TestVectorType::new(context.f32(), 4));
        assert_eq!(alloc::format!("{vector}"), "!test.vector<f32 x 4>");
        assert!(vector.name().parse_fn().is_some());
        assert!(vector.name().default_fn().is_none());
        assert_eq!(vector, context.intern_type(TestVectorType::new(context.f32(), 4)));
    }
}
