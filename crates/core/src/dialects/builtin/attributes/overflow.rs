use core::{fmt, str::FromStr};

use smallvec::SmallVec;

use super::signedness::parse_signedness;
use crate::{
    AttrPrinter, Signedness, attributes::AttrParser, derive::DialectAttribute,
    dialects::builtin::BuiltinDialect, print::AsmPrinter,
};

/// What an arithmetic operation does when its exact result does not fit the integer type.
///
/// Integers are signless, so the two modes that care whether the result fits name how it is read:
/// as a signed or as an unsigned number. Always check the documentation of the specific operation
/// involved for any difference in how a variant is interpreted.
///
/// Text: `wrapping`, `checked signed`, `checked unsigned`, `unchecked signed`, `unchecked unsigned`.
#[derive(DialectAttribute, Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
#[attribute(dialect = BuiltinDialect, implements(AttrPrinter))]
pub enum Overflow {
    /// The result is the low bits of the exact result, i.e. the exact result modulo `2^width`.
    #[default]
    Wrapping,
    /// The operation traps if the exact result does not fit the type, with the operands and the
    /// result read as signed or as unsigned numbers.
    Checked(Signedness),
    /// The result is undefined if the exact result does not fit the type, with the operands and the
    /// result read as signed or as unsigned numbers (LLVM's `nsw` and `nuw`).
    Unchecked(Signedness),
}

impl AttrPrinter for OverflowAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let (mode, signedness) = match self.value {
            Overflow::Wrapping => ("wrapping", None),
            Overflow::Checked(signedness) => ("checked", Some(signedness)),
            Overflow::Unchecked(signedness) => ("unchecked", Some(signedness)),
        };
        printer.print_keyword(mode);
        if let Some(signedness) = signedness {
            printer.print_space();
            printer.print_keyword(signedness.as_str());
        }
    }
}

impl AttrParser for OverflowAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        use crate::parse::Token;

        let keywords = SmallVec::<[Token; 3]>::from_iter(
            ["wrapping", "checked", "unchecked"].into_iter().map(Token::BareIdent),
        );

        let mode = parser.parse_keyword_from(&keywords)?;
        let overflow = match mode.as_str() {
            "checked" => Overflow::Checked(parse_signedness(parser)?),
            "unchecked" => Overflow::Unchecked(parse_signedness(parser)?),
            _ => Overflow::Wrapping,
        };

        let attr = parser.context_rc().create_attribute::<OverflowAttr, _>(overflow);
        Ok(attr)
    }
}

impl Overflow {
    /// Returns true if the result is the low bits of the exact result
    pub fn is_wrapping(&self) -> bool {
        matches!(self, Self::Wrapping)
    }

    /// Returns true if overflow will cause a trap
    pub fn is_checked(&self) -> bool {
        matches!(self, Self::Checked(_))
    }

    /// Returns true if overflow makes the result undefined
    pub fn is_unchecked(&self) -> bool {
        matches!(self, Self::Unchecked(_))
    }

    /// How the exact result is read when checking it, or `None` for [Overflow::Wrapping]
    pub fn signedness(&self) -> Option<Signedness> {
        match self {
            Self::Wrapping => None,
            Self::Checked(signedness) | Self::Unchecked(signedness) => Some(*signedness),
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Wrapping => "wrapping",
            Self::Checked(Signedness::Signed) => "checked signed",
            Self::Checked(Signedness::Unsigned) => "checked unsigned",
            Self::Unchecked(Signedness::Signed) => "unchecked signed",
            Self::Unchecked(Signedness::Unsigned) => "unchecked unsigned",
        }
    }
}

impl FromStr for Overflow {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "wrapping" => Ok(Self::Wrapping),
            "checked signed" => Ok(Self::Checked(Signedness::Signed)),
            "checked unsigned" => Ok(Self::Checked(Signedness::Unsigned)),
            "unchecked signed" => Ok(Self::Unchecked(Signedness::Signed)),
            "unchecked unsigned" => Ok(Self::Unchecked(Signedness::Unsigned)),
            _ => Err("unknown overflow type"),
        }
    }
}

impl AsRef<str> for Overflow {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.as_ref())
    }
}

impl crate::formatter::PrettyPrint for Overflow {
    fn render(&self) -> crate::formatter::Document {
        use crate::formatter::*;
        display(self)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn the_default_is_wrapping() {
        assert_eq!(Overflow::default(), Overflow::Wrapping);
        assert!(Overflow::default().is_wrapping());
    }

    #[test]
    fn predicates_and_signedness() {
        let checked = Overflow::Checked(Signedness::Signed);
        let unchecked = Overflow::Unchecked(Signedness::Unsigned);
        assert!(checked.is_checked() && !checked.is_unchecked() && !checked.is_wrapping());
        assert!(unchecked.is_unchecked() && !unchecked.is_checked() && !unchecked.is_wrapping());
        assert_eq!(Overflow::Wrapping.signedness(), None);
        assert_eq!(checked.signedness(), Some(Signedness::Signed));
        assert_eq!(unchecked.signedness(), Some(Signedness::Unsigned));
    }

    #[test]
    fn text_round_trips() {
        for value in [
            Overflow::Wrapping,
            Overflow::Checked(Signedness::Signed),
            Overflow::Checked(Signedness::Unsigned),
            Overflow::Unchecked(Signedness::Signed),
            Overflow::Unchecked(Signedness::Unsigned),
        ] {
            assert_eq!(value.to_string().parse::<Overflow>(), Ok(value));
        }
        assert!("checked".parse::<Overflow>().is_err());
        assert!("overflowing".parse::<Overflow>().is_err());
    }
}
