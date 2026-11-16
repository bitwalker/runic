//! The attributes of the dialect: limits, the static part of a memory access, mutability.

use alloc::format;
use core::{fmt, str::FromStr};

use runic_ir::{
    AttrPrinter, Context, Report, Type, attributes::AttrParser, derive::DialectAttribute,
    formatter::const_text, parse::ParserExt, print::AsmPrinter,
};

use crate::WasmDialect;

/// The limits of a memory or a table: a minimum and an optional maximum, as Wasm's `limits`.
///
/// The unit is pages for a memory and elements for a table; the attribute does not say which. A
/// well-formed value has no maximum or a maximum that is at least the minimum, which the kind's
/// verifier checks.
///
/// Text: `#wasm.limits<1>` (no maximum), `#wasm.limits<1..16>`. A body-less `#wasm.limits` is the
/// default, `0` (a minimum of 0 and no maximum). The attribute's own type is unknown.
#[derive(DialectAttribute, Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
#[attribute(
    dialect = WasmDialect,
    name = "limits",
    verify = "verify_limits",
    implements(AttrPrinter)
)]
pub struct Limits {
    min: u64,
    max: Option<u64>,
}

impl Limits {
    /// Limits with the minimum `min` and no maximum.
    pub const fn at_least(min: u64) -> Self {
        Self { min, max: None }
    }

    /// Limits with the minimum `min` and the maximum `max`. A `max` below `min` is made, and
    /// rejected by the verifier of the attribute kind.
    pub const fn between(min: u64, max: u64) -> Self {
        Self {
            min,
            max: Some(max),
        }
    }

    /// The minimum.
    #[inline(always)]
    pub const fn min(&self) -> u64 {
        self.min
    }

    /// The maximum, if there is one.
    #[inline(always)]
    pub const fn max(&self) -> Option<u64> {
        self.max
    }
}

impl fmt::Display for Limits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.max {
            None => write!(f, "{}", self.min),
            Some(max) => write!(f, "{}..{max}", self.min),
        }
    }
}

impl runic_ir::formatter::PrettyPrint for Limits {
    fn render(&self) -> runic_ir::formatter::Document {
        runic_ir::formatter::display(self)
    }
}

/// Wasm requires `min <= max`.
fn verify_limits(limits: &Limits, _ty: Type, _context: &Context) -> Result<(), Report> {
    match limits.max {
        Some(max) if max < limits.min => Err(Report::msg(format!(
            "limits `{limits}` with a maximum below the minimum, but Wasm limits satisfy min <= \
             max"
        ))),
        _ => Ok(()),
    }
}

/// `MIN` or `MIN..MAX`, in decimal.
impl AttrPrinter for LimitsAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_decimal_integer(self.value.min);
        if let Some(max) = self.value.max {
            *printer += const_text("..");
            printer.print_decimal_integer(max);
        }
    }
}

/// `MIN` or `MIN..MAX`: decimal integers that fit a `u64`.
impl AttrParser for LimitsAttr {
    fn parse(
        parser: &mut dyn runic_ir::parse::Parser<'_>,
    ) -> runic_ir::parse::ParseResult<runic_ir::AttributeRef> {
        let min = parser.parse_decimal_integer::<u64>()?.into_inner();
        let max = if parser.parse_optional_dot()? {
            parser.parse_dot()?;
            Some(parser.parse_decimal_integer::<u64>()?.into_inner())
        } else {
            None
        };
        Ok(parser
            .context_rc()
            .create_attribute::<LimitsAttr, _>(Limits { min, max })
            .as_attribute_ref())
    }
}

/// The static part of a memory access: an offset added to the address operand, and the alignment
/// the access declares, as Wasm's `memarg`.
///
/// The alignment is in bytes (Wasm's binary format stores its base-2 logarithm) and a power of two
/// of at least 1, which the kind's verifier checks. That it is at most the natural alignment of
/// the access is checked by the load or store that holds the attribute.
///
/// Text: `#wasm.memarg<offset = 8, align = 4>`. A body-less `#wasm.memarg` is the default,
/// `offset = 0, align = 1`. The attribute's own type is unknown.
#[derive(DialectAttribute, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[attribute(
    dialect = WasmDialect,
    name = "memarg",
    default = "default_memarg",
    verify = "verify_memarg",
    implements(AttrPrinter)
)]
pub struct MemArg {
    offset: u64,
    align: u64,
}

impl MemArg {
    /// The offset `offset` and the alignment `align` bytes. An `align` that is not a power of two
    /// is made, and rejected by the verifier of the attribute kind.
    pub const fn new(offset: u64, align: u64) -> Self {
        Self { offset, align }
    }

    /// The offset in bytes, added to the address operand.
    #[inline(always)]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// The declared alignment in bytes.
    #[inline(always)]
    pub const fn align(&self) -> u64 {
        self.align
    }
}

impl Default for MemArg {
    fn default() -> Self {
        Self {
            offset: 0,
            align: 1,
        }
    }
}

impl fmt::Display for MemArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "offset = {}, align = {}", self.offset, self.align)
    }
}

impl runic_ir::formatter::PrettyPrint for MemArg {
    fn render(&self) -> runic_ir::formatter::Document {
        runic_ir::formatter::display(self)
    }
}

/// The value of a body-less `#wasm.memarg`.
fn default_memarg(_context: &Context) -> MemArg {
    MemArg::default()
}

/// Wasm's alignment hint is a power of two.
fn verify_memarg(memarg: &MemArg, _ty: Type, _context: &Context) -> Result<(), Report> {
    if memarg.align.is_power_of_two() {
        Ok(())
    } else {
        Err(Report::msg(format!(
            "an alignment of {} bytes, but a Wasm memarg alignment is a power of two, at least 1",
            memarg.align
        )))
    }
}

/// `offset = OFFSET, align = ALIGN`, both in decimal.
impl AttrPrinter for MemArgAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        *printer += const_text("offset = ");
        printer.print_decimal_integer(self.value.offset);
        *printer += const_text(", align = ");
        printer.print_decimal_integer(self.value.align);
    }
}

/// `offset = OFFSET, align = ALIGN`: decimal integers that fit a `u64`.
impl AttrParser for MemArgAttr {
    fn parse(
        parser: &mut dyn runic_ir::parse::Parser<'_>,
    ) -> runic_ir::parse::ParseResult<runic_ir::AttributeRef> {
        parser.parse_custom_keyword("offset")?;
        parser.parse_equal()?;
        let offset = parser.parse_decimal_integer::<u64>()?.into_inner();
        parser.parse_comma()?;
        parser.parse_custom_keyword("align")?;
        parser.parse_equal()?;
        let align = parser.parse_decimal_integer::<u64>()?.into_inner();
        Ok(parser
            .context_rc()
            .create_attribute::<MemArgAttr, _>(MemArg { offset, align })
            .as_attribute_ref())
    }
}

/// Whether a global may be written: Wasm's `mut`.
///
/// Text: `#wasm.mutability<const>`, `#wasm.mutability<mut>`. A body-less `#wasm.mutability` is
/// `const`.
#[derive(DialectAttribute, Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
#[attribute(dialect = WasmDialect, name = "mutability", implements(AttrPrinter))]
pub enum Mutability {
    /// The global is written only by its initializer.
    #[default]
    Const,
    /// The global may be written by `global.set`.
    Mut,
}

impl Mutability {
    /// Whether this is [Mutability::Mut].
    pub const fn is_mut(self) -> bool {
        matches!(self, Self::Mut)
    }

    /// The keyword of this mutability: `const` or `mut`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Const => "const",
            Self::Mut => "mut",
        }
    }
}

impl FromStr for Mutability {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "const" => Ok(Self::Const),
            "mut" => Ok(Self::Mut),
            _ => Err("unknown mutability"),
        }
    }
}

impl fmt::Display for Mutability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl runic_ir::formatter::PrettyPrint for Mutability {
    fn render(&self) -> runic_ir::formatter::Document {
        runic_ir::formatter::display(self)
    }
}

impl AttrPrinter for MutabilityAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_keyword(self.value.as_str());
    }
}

impl AttrParser for MutabilityAttr {
    fn parse(
        parser: &mut dyn runic_ir::parse::Parser<'_>,
    ) -> runic_ir::parse::ParseResult<runic_ir::AttributeRef> {
        use runic_ir::parse::Token;

        let keyword = parser
            .parse_keyword_from(&[Token::BareIdent("const"), Token::BareIdent("mut")])?
            .into_inner();
        let mutability = keyword.as_str().parse::<Mutability>().unwrap();
        Ok(parser
            .context_rc()
            .create_attribute::<MutabilityAttr, _>(mutability)
            .as_attribute_ref())
    }
}
