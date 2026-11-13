use core::fmt;

use smallvec::SmallVec;

use crate::{
    AttrPrinter,
    attributes::AttrParser,
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    effects::MemoryEffect,
    formatter::const_text,
    parse::{ParseResult, Parser, ParserExt, Token},
    print::AsmPrinter,
};

/// One memory effect a callable declares, optionally on one of its arguments or results.
///
/// Text: `read`, `write`, `allocate` or `free`, optionally followed by `(arg N)` or `(result N)`.
/// At most one of `argument` and `result` is set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryEffectDescriptor {
    pub effect: MemoryEffect,
    pub argument: Option<u8>,
    pub result: Option<u8>,
}

impl MemoryEffectDescriptor {
    /// `effect`, on no particular value
    pub const fn new(effect: MemoryEffect) -> Self {
        Self {
            effect,
            argument: None,
            result: None,
        }
    }

    /// `effect` on argument `index`
    pub const fn on_argument(effect: MemoryEffect, index: u8) -> Self {
        Self {
            effect,
            argument: Some(index),
            result: None,
        }
    }

    /// `effect` on result `index`
    pub const fn on_result(effect: MemoryEffect, index: u8) -> Self {
        Self {
            effect,
            argument: None,
            result: Some(index),
        }
    }
}

impl fmt::Display for MemoryEffectDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.effect.as_ref())?;
        if let Some(index) = self.argument {
            write!(f, "(arg {index})")?;
        }
        if let Some(index) = self.result {
            write!(f, "(result {index})")?;
        }
        Ok(())
    }
}

/// What a callable says about the memory effects of calling it.
///
/// Text: `#builtin.declared_effects<unknown>`, `#builtin.declared_effects<>` (no effects), or a
/// list of effects, `#builtin.declared_effects<read, write(arg 0), allocate(result 0)>`.
#[derive(DialectAttribute, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[attribute(dialect = BuiltinDialect, implements(AttrPrinter))]
pub enum DeclaredEffects {
    /// The callable states nothing: calling it may have any effect.
    #[default]
    Unknown,
    /// Calling the callable has these effects and no others; an empty list means none.
    Known(SmallVec<[MemoryEffectDescriptor; 2]>),
}

impl DeclaredEffects {
    /// The callable is declared to have no memory effects
    pub fn none() -> Self {
        Self::Known(SmallVec::new_const())
    }

    /// Returns true unless the effects are [DeclaredEffects::Unknown]
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// The declared effects, or `None` when they are unknown
    pub fn known(&self) -> Option<&[MemoryEffectDescriptor]> {
        match self {
            Self::Unknown => None,
            Self::Known(effects) => Some(effects.as_slice()),
        }
    }
}

impl<const N: usize> From<[MemoryEffectDescriptor; N]> for DeclaredEffects {
    /// These effects, known
    fn from(effects: [MemoryEffectDescriptor; N]) -> Self {
        Self::Known(SmallVec::from_iter(effects))
    }
}

impl FromIterator<MemoryEffectDescriptor> for DeclaredEffects {
    /// These effects, known
    fn from_iter<I: IntoIterator<Item = MemoryEffectDescriptor>>(iter: I) -> Self {
        Self::Known(SmallVec::from_iter(iter))
    }
}

impl fmt::Display for DeclaredEffects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("unknown"),
            Self::Known(effects) => {
                for (i, effect) in effects.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{effect}")?;
                }
                Ok(())
            }
        }
    }
}

impl crate::formatter::PrettyPrint for DeclaredEffects {
    fn render(&self) -> crate::formatter::Document {
        crate::formatter::display(self)
    }
}

impl AttrPrinter for DeclaredEffectsAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        match &self.value {
            DeclaredEffects::Unknown => printer.print_keyword("unknown"),
            DeclaredEffects::Known(effects) => print_effect_list(printer, effects),
        }
    }
}

impl AttrParser for DeclaredEffectsAttr {
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<crate::AttributeRef> {
        let effects = if parser.parse_optional_custom_keyword("unknown")?.is_some() {
            DeclaredEffects::Unknown
        } else {
            DeclaredEffects::Known(parse_effect_list(parser)?)
        };
        Ok(parser.context_rc().create_attribute::<DeclaredEffectsAttr, _>(effects))
    }
}

/// Prints `effects` separated by commas, each as [MemoryEffectDescriptor]'s text.
pub(crate) fn print_effect_list(printer: &mut AsmPrinter<'_>, effects: &[MemoryEffectDescriptor]) {
    for (i, effect) in effects.iter().enumerate() {
        if i > 0 {
            *printer += const_text(", ");
        }
        printer.print_keyword(match effect.effect {
            MemoryEffect::Read => "read",
            MemoryEffect::Write => "write",
            MemoryEffect::Allocate => "allocate",
            MemoryEffect::Free => "free",
        });
        if let Some(index) = effect.argument {
            printer.print_lparen();
            printer.print_keyword("arg");
            printer.print_space();
            printer.print_decimal_integer(index);
            printer.print_rparen();
        }
        if let Some(index) = effect.result {
            printer.print_lparen();
            printer.print_keyword("result");
            printer.print_space();
            printer.print_decimal_integer(index);
            printer.print_rparen();
        }
    }
}

/// Parses a list of effects separated by commas, possibly empty: the list ends at the first token
/// that does not start an effect.
pub(crate) fn parse_effect_list(
    parser: &mut dyn Parser<'_>,
) -> ParseResult<SmallVec<[MemoryEffectDescriptor; 2]>> {
    const EFFECTS: [Token<'static>; 4] = [
        Token::BareIdent("read"),
        Token::BareIdent("write"),
        Token::BareIdent("allocate"),
        Token::BareIdent("free"),
    ];
    const TARGETS: [Token<'static>; 2] = [Token::BareIdent("arg"), Token::BareIdent("result")];

    let mut effects = SmallVec::new_const();
    if !parser.token_stream_mut().is_next(|tok| EFFECTS.contains(&tok)) {
        return Ok(effects);
    }
    loop {
        let effect = parser.parse_keyword_from(&EFFECTS)?;
        let effect = match effect.as_str() {
            "read" => MemoryEffect::Read,
            "write" => MemoryEffect::Write,
            "allocate" => MemoryEffect::Allocate,
            "free" => MemoryEffect::Free,
            other => {
                unreachable!("`parse_keyword_from` returned `{other}`, which is not an effect")
            }
        };
        let mut descriptor = MemoryEffectDescriptor::new(effect);
        if parser.token_stream_mut().next_if_eq(Token::Lparen)? {
            let target = parser.parse_keyword_from(&TARGETS)?;
            let index = parser.parse_decimal_integer::<u8>()?.into_inner();
            parser.parse_rparen()?;
            if target.as_str() == "arg" {
                descriptor.argument = Some(index);
            } else {
                descriptor.result = Some(index);
            }
        }
        effects.push(descriptor);
        if !parser.parse_optional_comma()? {
            break;
        }
    }
    Ok(effects)
}
