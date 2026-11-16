use core::fmt;

use runic_ir::{
    Align, AttrPrinter, attributes::AttrParser, derive::DialectAttribute, parse::ParserError,
    print::AsmPrinter,
};

use crate::MemDialect;

/// The alignment a memory access assumes of its address.
///
/// [Alignment::Natural] is the alignment the data layout gives the accessed type;
/// [Alignment::Bytes] states a power of two of bytes from 1 to 32768 instead. An access whose
/// address is not aligned as it assumes is undefined behaviour.
///
/// Text: `#mem.align<natural>`, `#mem.align<4>`; a body-less `#mem.align` is `natural`. In the text
/// of an operation it is written `, align 4`, and left out when natural.
#[derive(DialectAttribute, Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
#[attribute(name = "align", dialect = MemDialect, implements(AttrPrinter))]
pub enum Alignment {
    /// The alignment the data layout gives the accessed type.
    #[default]
    Natural,
    /// A stated number of bytes.
    Bytes(Align),
}

impl Alignment {
    /// Whether this is [Alignment::Natural].
    pub const fn is_natural(&self) -> bool {
        matches!(self, Self::Natural)
    }

    /// The stated alignment, or `None` when it is natural.
    pub const fn bytes(&self) -> Option<Align> {
        match self {
            Self::Natural => None,
            Self::Bytes(align) => Some(*align),
        }
    }
}

impl From<Align> for Alignment {
    fn from(align: Align) -> Self {
        Self::Bytes(align)
    }
}

impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Natural => f.write_str("natural"),
            Self::Bytes(align) => write!(f, "{align}"),
        }
    }
}

impl runic_ir::formatter::PrettyPrint for Alignment {
    fn render(&self) -> runic_ir::formatter::Document {
        use runic_ir::formatter::*;
        display(self)
    }
}

impl AttrPrinter for AlignmentAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        match self.value {
            Alignment::Natural => printer.print_keyword("natural"),
            Alignment::Bytes(align) => printer.print_decimal_integer(align.bytes()),
        }
    }
}

impl AttrParser for AlignmentAttr {
    fn parse(
        parser: &mut dyn runic_ir::parse::Parser<'_>,
    ) -> runic_ir::parse::ParseResult<runic_ir::AttributeRef> {
        let alignment = if parser.parse_optional_custom_keyword("natural")?.is_some() {
            Alignment::Natural
        } else {
            Alignment::Bytes(parse_align_bytes(parser)?)
        };
        Ok(parser
            .context_rc()
            .create_attribute::<AlignmentAttr, _>(alignment)
            .as_attribute_ref())
    }
}

/// Parses a number of bytes that is a power of two from 1 to 32768; anything else is an error at
/// its span.
pub(crate) fn parse_align_bytes<'input, P>(parser: &mut P) -> runic_ir::parse::ParseResult<Align>
where
    P: ?Sized + runic_ir::parse::Parser<'input>,
{
    use runic_ir::parse::ParserExt;

    let (span, bytes) = parser.parse_decimal_integer::<u64>()?.into_parts();
    Align::from_bytes(bytes).ok_or_else(|| ParserError::InvalidAttributeValue {
        span,
        reason: alloc::format!(
            "an alignment is a power of two from 1 to {}, got {bytes}",
            Align::MAX_BYTES
        ),
    })
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, vec};

    use runic_ir::{
        Context,
        testing::elements::{Row, check_every_kind},
    };

    use super::*;

    /// Every attribute kind of `mem` lists and replaces what it holds: an alignment holds nothing.
    #[test]
    fn every_attribute_kind_lists_and_replaces_what_it_holds() {
        let context = Rc::new(Context::default());
        let rows = vec![
            Row::new(
                "natural",
                context.create_attribute::<AlignmentAttr, _>(Alignment::Natural),
                [],
            ),
            Row::new(
                "bytes",
                context.create_attribute::<AlignmentAttr, _>(Alignment::Bytes(
                    Align::from_bytes(8).unwrap(),
                )),
                [],
            ),
        ];
        check_every_kind(&context, &["mem"], &rows);
    }
}
