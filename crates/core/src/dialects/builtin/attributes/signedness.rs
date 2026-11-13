use core::{fmt, str::FromStr};

use crate::{
    AttrPrinter, SmallVec, attributes::AttrParser, derive::DialectAttribute,
    dialects::builtin::BuiltinDialect, print::AsmPrinter,
};

/// How an operation reads the bits of an integer: as a two's complement signed number or as an
/// unsigned one.
///
/// Integer types are signless, so an operation whose result depends on the interpretation of its
/// operands carries a `Signedness` (or says it in its name, like `sdiv`).
///
/// Text: `signed` or `unsigned`. As an attribute, `#builtin.signedness<signed>`; a body-less
/// `#builtin.signedness` is `signed`.
#[derive(DialectAttribute, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[attribute(dialect = BuiltinDialect, default = "default_signedness", implements(AttrPrinter))]
pub enum Signedness {
    /// The bits are a two's complement signed number.
    Signed,
    /// The bits are an unsigned number.
    Unsigned,
}

/// The value of a body-less `#builtin.signedness`.
fn default_signedness(_context: &crate::Context) -> Signedness {
    Signedness::Signed
}

impl AttrPrinter for SignednessAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_keyword(self.value.as_str());
    }
}

impl AttrParser for SignednessAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        let signedness = parse_signedness(parser)?;
        let attr = parser.context_rc().create_attribute::<SignednessAttr, _>(signedness);
        Ok(attr)
    }
}

/// Parses the keyword `signed` or `unsigned`; any other token is an error at its span.
pub(super) fn parse_signedness(
    parser: &mut dyn crate::parse::Parser<'_>,
) -> crate::parse::ParseResult<Signedness> {
    use crate::parse::Token;

    let keywords = SmallVec::<[Token; 2]>::from_iter(
        [Signedness::Signed, Signedness::Unsigned]
            .iter()
            .map(Signedness::as_str)
            .map(Token::BareIdent),
    );
    let keyword = parser.parse_keyword_from(&keywords)?;
    Ok(keyword.as_str().parse::<Signedness>().unwrap())
}

impl Signedness {
    /// Returns true for [Signedness::Signed]
    #[inline]
    pub const fn is_signed(&self) -> bool {
        matches!(self, Self::Signed)
    }

    /// Returns true for [Signedness::Unsigned]
    #[inline]
    pub const fn is_unsigned(&self) -> bool {
        matches!(self, Self::Unsigned)
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Signed => "signed",
            Self::Unsigned => "unsigned",
        }
    }
}

impl FromStr for Signedness {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "signed" => Ok(Self::Signed),
            "unsigned" => Ok(Self::Unsigned),
            _ => Err("unknown signedness"),
        }
    }
}

impl fmt::Display for Signedness {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl crate::formatter::PrettyPrint for Signedness {
    fn render(&self) -> crate::formatter::Document {
        use crate::formatter::*;
        display(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;

    #[test]
    fn text_round_trips() {
        for value in [Signedness::Signed, Signedness::Unsigned] {
            assert_eq!(value.to_string().parse::<Signedness>(), Ok(value));
        }
        assert!("both".parse::<Signedness>().is_err());
        assert_eq!(Signedness::Signed.to_string(), "signed");
        assert_eq!(Signedness::Unsigned.to_string(), "unsigned");
    }
}
