use core::{cmp::Ordering, fmt, str::FromStr};

use runic_core::{
    AttrPrinter, Int, SmallVec, attributes::AttrParser, derive::DialectAttribute, print::AsmPrinter,
};

use crate::ArithDialect;

/// The relation `arith.cmpi` tests between its operands.
///
/// Integers are signless, so the ordering predicates say how the operands are read: `s` predicates
/// read them as two's complement signed numbers, `u` predicates as unsigned numbers. `eq` and `ne`
/// compare the bits.
///
/// Text: `eq`, `ne`, `slt`, `sle`, `sgt`, `sge`, `ult`, `ule`, `ugt`, `uge`. As an attribute,
/// `#arith.cmp_predicate<slt>`; a body-less `#arith.cmp_predicate` is `eq`.
#[derive(DialectAttribute, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[attribute(dialect = ArithDialect, default = "default_predicate", implements(AttrPrinter))]
pub enum CmpPredicate {
    /// The operands are equal.
    Eq,
    /// The operands are not equal.
    Ne,
    /// Signed less than.
    Slt,
    /// Signed less than or equal.
    Sle,
    /// Signed greater than.
    Sgt,
    /// Signed greater than or equal.
    Sge,
    /// Unsigned less than.
    Ult,
    /// Unsigned less than or equal.
    Ule,
    /// Unsigned greater than.
    Ugt,
    /// Unsigned greater than or equal.
    Uge,
}

/// The value of a body-less `#arith.cmp_predicate`.
fn default_predicate(_context: &runic_core::Context) -> CmpPredicate {
    CmpPredicate::Eq
}

impl CmpPredicate {
    /// Every predicate, in declaration order.
    pub const ALL: [Self; 10] = [
        Self::Eq,
        Self::Ne,
        Self::Slt,
        Self::Sle,
        Self::Sgt,
        Self::Sge,
        Self::Ult,
        Self::Ule,
        Self::Ugt,
        Self::Uge,
    ];

    /// Whether the predicate reads its operands as signed numbers (`slt`, `sle`, `sgt`, `sge`).
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::Slt | Self::Sle | Self::Sgt | Self::Sge)
    }

    /// Whether the predicate reads its operands as unsigned numbers (`ult`, `ule`, `ugt`, `uge`).
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::Ult | Self::Ule | Self::Ugt | Self::Uge)
    }

    /// The predicate that holds exactly when this one does not: `slt` becomes `sge`.
    pub const fn inverted(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Slt => Self::Sge,
            Self::Sle => Self::Sgt,
            Self::Sgt => Self::Sle,
            Self::Sge => Self::Slt,
            Self::Ult => Self::Uge,
            Self::Ule => Self::Ugt,
            Self::Ugt => Self::Ule,
            Self::Uge => Self::Ult,
        }
    }

    /// The predicate that holds for the swapped operands exactly when this one holds for the
    /// operands as given: `slt` becomes `sgt`.
    pub const fn swapped(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Slt => Self::Sgt,
            Self::Sle => Self::Sge,
            Self::Sgt => Self::Slt,
            Self::Sge => Self::Sle,
            Self::Ult => Self::Ugt,
            Self::Ule => Self::Uge,
            Self::Ugt => Self::Ult,
            Self::Uge => Self::Ule,
        }
    }

    /// Whether the predicate holds for `lhs` and `rhs`. Panics if their widths differ.
    #[track_caller]
    pub fn evaluate(self, lhs: Int, rhs: Int) -> bool {
        let ordering = if self.is_signed() {
            lhs.cmp_signed(rhs)
        } else {
            lhs.cmp_unsigned(rhs)
        };
        match self {
            Self::Eq => ordering == Ordering::Equal,
            Self::Ne => ordering != Ordering::Equal,
            Self::Slt | Self::Ult => ordering == Ordering::Less,
            Self::Sle | Self::Ule => ordering != Ordering::Greater,
            Self::Sgt | Self::Ugt => ordering == Ordering::Greater,
            Self::Sge | Self::Uge => ordering != Ordering::Less,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Slt => "slt",
            Self::Sle => "sle",
            Self::Sgt => "sgt",
            Self::Sge => "sge",
            Self::Ult => "ult",
            Self::Ule => "ule",
            Self::Ugt => "ugt",
            Self::Uge => "uge",
        }
    }
}

impl FromStr for CmpPredicate {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|predicate| predicate.as_str() == s)
            .ok_or("unknown predicate")
    }
}

impl fmt::Display for CmpPredicate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl runic_core::formatter::PrettyPrint for CmpPredicate {
    fn render(&self) -> runic_core::formatter::Document {
        use runic_core::formatter::*;
        display(self)
    }
}

impl AttrPrinter for CmpPredicateAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_keyword(self.value.as_str());
    }
}

impl AttrParser for CmpPredicateAttr {
    fn parse(
        parser: &mut dyn runic_core::parse::Parser<'_>,
    ) -> runic_core::parse::ParseResult<runic_core::AttributeRef> {
        let predicate = parse_predicate(parser)?;
        Ok(parser
            .context_rc()
            .create_attribute::<CmpPredicateAttr, _>(predicate)
            .as_attribute_ref())
    }
}

/// Parses one of the ten predicate keywords; any other token is an error at its span.
pub(crate) fn parse_predicate(
    parser: &mut dyn runic_core::parse::Parser<'_>,
) -> runic_core::parse::ParseResult<CmpPredicate> {
    use runic_core::parse::Token;

    let keywords = SmallVec::<[Token; 10]>::from_iter(
        CmpPredicate::ALL.iter().map(|predicate| Token::BareIdent(predicate.as_str())),
    );
    let keyword = parser.parse_keyword_from(&keywords)?;
    Ok(keyword.as_str().parse::<CmpPredicate>().unwrap())
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, string::ToString};

    use runic_core::Context;

    use super::*;

    /// Every attribute kind of `arith` lists and replaces what it holds: a predicate holds
    /// nothing.
    #[test]
    fn every_attribute_kind_lists_and_replaces_what_it_holds() {
        use runic_core::testing::elements::{Row, check_every_kind};

        let context = Rc::new(Context::default());
        let rows = [Row::new(
            "predicate",
            context.create_attribute::<CmpPredicateAttr, _>(CmpPredicate::Ult),
            [],
        )];
        check_every_kind(&context, &["arith"], &rows);
    }

    #[test]
    fn text_round_trips() {
        for predicate in CmpPredicate::ALL {
            assert_eq!(predicate.to_string().parse::<CmpPredicate>(), Ok(predicate));
        }
        assert!("lt".parse::<CmpPredicate>().is_err());
        assert!("neq".parse::<CmpPredicate>().is_err());
    }

    #[test]
    fn evaluate_agrees_with_the_signed_and_unsigned_orderings() {
        for lhs in 0..16u128 {
            for rhs in 0..16u128 {
                let (a, b) = (Int::from_bits(4, lhs), Int::from_bits(4, rhs));
                let signed = a.cmp_signed(b);
                let unsigned = a.cmp_unsigned(b);
                let expected = [
                    (CmpPredicate::Eq, a == b),
                    (CmpPredicate::Ne, a != b),
                    (CmpPredicate::Slt, signed.is_lt()),
                    (CmpPredicate::Sle, signed.is_le()),
                    (CmpPredicate::Sgt, signed.is_gt()),
                    (CmpPredicate::Sge, signed.is_ge()),
                    (CmpPredicate::Ult, unsigned.is_lt()),
                    (CmpPredicate::Ule, unsigned.is_le()),
                    (CmpPredicate::Ugt, unsigned.is_gt()),
                    (CmpPredicate::Uge, unsigned.is_ge()),
                ];
                for (predicate, holds) in expected {
                    assert_eq!(predicate.evaluate(a, b), holds, "{a:?} {predicate} {b:?}");
                }
            }
        }
    }

    #[test]
    fn inverted_and_swapped_are_involutions_that_agree_with_evaluate() {
        for predicate in CmpPredicate::ALL {
            assert_eq!(predicate.inverted().inverted(), predicate);
            assert_eq!(predicate.swapped().swapped(), predicate);
            assert_eq!(predicate.inverted().is_signed(), predicate.is_signed());
            assert_eq!(predicate.swapped().is_signed(), predicate.is_signed());
            for lhs in 0..16u128 {
                for rhs in 0..16u128 {
                    let (a, b) = (Int::from_bits(4, lhs), Int::from_bits(4, rhs));
                    assert_eq!(predicate.inverted().evaluate(a, b), !predicate.evaluate(a, b));
                    assert_eq!(predicate.swapped().evaluate(b, a), predicate.evaluate(a, b));
                }
            }
        }
    }

    #[test]
    fn the_attribute_prints_its_keyword() {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<ArithDialect>();
        let attr = context.create_attribute::<CmpPredicateAttr, _>(CmpPredicate::Slt);
        let flags = Default::default();
        let mut printer = AsmPrinter::root(context.clone(), &flags);
        printer.print_attribute_value(&*attr.borrow());
        assert_eq!(printer.finish().to_string(), "#arith.cmp_predicate<slt>");
    }
}
