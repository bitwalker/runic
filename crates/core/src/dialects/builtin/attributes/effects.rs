use core::{fmt, ops::Range};

use smallvec::SmallVec;

use crate::{
    AttrPrinter,
    attributes::AttrParser,
    derive::{AttributeComponent, DialectAttribute},
    dialects::builtin::BuiltinDialect,
    effects::MemoryEffect,
    formatter::const_text,
    parse::{ParseResult, Parser, ParserExt, Token},
    print::AsmPrinter,
};

/// One memory effect a callable declares, optionally on one of its arguments or results.
///
/// Text: `read`, `write`, `allocate` or `free`, optionally followed by `(arg N)` or `(result N)`.
/// At most one of `argument` and `result` is set. A descriptor with neither is an effect on no
/// value in particular: it applies to the resource in general, and is what `read` alone means.
/// Such an effect is on any value: code that reasons per value must treat it as applying to every
/// value, though a query for the effects on one value does not list it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, AttributeComponent)]
pub struct MemoryEffectDescriptor {
    pub effect: MemoryEffect,
    pub argument: Option<u32>,
    pub result: Option<u32>,
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
    pub const fn on_argument(effect: MemoryEffect, index: u32) -> Self {
        Self {
            effect,
            argument: Some(index),
            result: None,
        }
    }

    /// `effect` on result `index`
    pub const fn on_result(effect: MemoryEffect, index: u32) -> Self {
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
///
/// Declaring no effects, or only reads (and allocations of a result), is the statement that a call
/// of the callable may be removed when its results are unused, merged with an equal call, or moved
/// past operations that do not write. A trap or a failure to return is not an effect, so a callable
/// whose failure to return must be observed declares nothing, or a write. The one case the IR sees
/// for itself is a signature that never returns (`-> never`): a call made with such a signature has
/// unknown effects whatever its callee declares.
///
/// As an attribute it is valid when each effect names at most one of an argument and a result.
/// Whether the argument or result it names exists is a question for the callable that declares
/// it.
#[derive(DialectAttribute, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[attribute(
    dialect = BuiltinDialect,
    verify = "verify_declared_effects",
    implements(AttrPrinter)
)]
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

/// The declared effects of a signature whose parameters and results changed: `params[i]` and
/// `results[j]` are the new indices old parameter `i` and old result `j` became. An effect on a
/// parameter or result that became several is repeated on each, in order; on one that became none
/// it is an effect on no value. `Unknown` stays `Unknown`.
///
/// An effect on no value is a descriptor with neither `argument` nor `result`: it applies to the
/// resource in general, and it is left as it is. An effect the remap makes equal to one already
/// present is kept once, at its first position; so identity ranges give an equal value for an
/// input without duplicates, and the input with its duplicates removed otherwise.
///
/// Every index a descriptor names must be in `params` (for an argument) or `results` (for a
/// result): that is a precondition, checked with a `debug_assert!`. A function's verifier keeps
/// its declared effects within its signature, so a caller that remaps a function's own effects
/// with that signature's ranges meets it. In a build without debug assertions, an effect that
/// names an index outside the ranges is treated as an effect on no value, which claims at least
/// as much as the effect it replaces.
pub fn remap_declared_effects(
    effects: &DeclaredEffects,
    params: &[Range<u32>],
    results: &[Range<u32>],
) -> DeclaredEffects {
    let DeclaredEffects::Known(effects) = effects else {
        return DeclaredEffects::Unknown;
    };
    let mut remapped = SmallVec::<[MemoryEffectDescriptor; 2]>::new();
    let mut push = |descriptor: MemoryEffectDescriptor| {
        if !remapped.contains(&descriptor) {
            remapped.push(descriptor);
        }
    };
    for descriptor in effects {
        let effect = descriptor.effect;
        match (descriptor.argument, descriptor.result) {
            (Some(index), _) => match params.get(index as usize) {
                Some(new) if !new.is_empty() => {
                    new.clone().for_each(|i| push(MemoryEffectDescriptor::on_argument(effect, i)))
                }
                Some(_) => push(MemoryEffectDescriptor::new(effect)),
                None => {
                    debug_assert!(
                        false,
                        "the effect `{descriptor}` names argument {index}, but only {} \
                         parameter(s) were remapped",
                        params.len()
                    );
                    push(MemoryEffectDescriptor::new(effect))
                }
            },
            (None, Some(index)) => match results.get(index as usize) {
                Some(new) if !new.is_empty() => {
                    new.clone().for_each(|i| push(MemoryEffectDescriptor::on_result(effect, i)))
                }
                Some(_) => push(MemoryEffectDescriptor::new(effect)),
                None => {
                    debug_assert!(
                        false,
                        "the effect `{descriptor}` names result {index}, but only {} result(s) \
                         were remapped",
                        results.len()
                    );
                    push(MemoryEffectDescriptor::new(effect))
                }
            },
            (None, None) => push(MemoryEffectDescriptor::new(effect)),
        }
    }
    DeclaredEffects::Known(remapped)
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

/// Each effect names at most one of an argument and a result.
fn verify_declared_effects(
    effects: &DeclaredEffects,
    _ty: crate::Type,
    _context: &crate::Context,
) -> Result<(), crate::Report> {
    if let Some(effects) = effects.known()
        && let Some(effect) = effects
            .iter()
            .find(|effect| effect.argument.is_some() && effect.result.is_some())
    {
        return Err(crate::Report::msg(alloc::format!(
            "an invalid effect declaration: the effect `{effect}` names both an argument and a \
             result"
        )));
    }
    Ok(())
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
            let index = parser.parse_decimal_integer::<u32>()?.into_inner();
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

#[cfg(test)]
mod tests {
    use super::*;

    const R: MemoryEffect = MemoryEffect::Read;
    const W: MemoryEffect = MemoryEffect::Write;
    const A: MemoryEffect = MemoryEffect::Allocate;

    /// The range `start..end`, built by a call so that a list of one range is not an array of
    /// the one range's length
    fn r(start: u32, end: u32) -> Range<u32> {
        start..end
    }

    fn arg(effect: MemoryEffect, index: u32) -> MemoryEffectDescriptor {
        MemoryEffectDescriptor::on_argument(effect, index)
    }

    fn res(effect: MemoryEffect, index: u32) -> MemoryEffectDescriptor {
        MemoryEffectDescriptor::on_result(effect, index)
    }

    #[test]
    fn identity_ranges_give_an_equal_value() {
        let effects = DeclaredEffects::from([
            MemoryEffectDescriptor::new(R),
            arg(W, 1),
            res(A, 0),
            arg(R, 0),
        ]);
        assert_eq!(remap_declared_effects(&effects, &[0..1, 1..2], &[r(0, 1)]), effects);
    }

    #[test]
    fn a_split_parameter_repeats_its_effect_in_order() {
        let effects = DeclaredEffects::from([arg(R, 0), res(A, 0)]);
        let remapped = remap_declared_effects(&effects, &[r(0, 3)], &[r(0, 2)]);
        assert_eq!(
            remapped,
            DeclaredEffects::from([arg(R, 0), arg(R, 1), arg(R, 2), res(A, 0), res(A, 1)])
        );
    }

    #[test]
    fn a_dropped_parameter_or_result_becomes_an_effect_on_no_value() {
        let effects = DeclaredEffects::from([arg(R, 0), res(A, 0), arg(W, 1)]);
        let remapped = remap_declared_effects(&effects, &[0..0, 0..1], &[r(0, 0)]);
        assert_eq!(
            remapped,
            DeclaredEffects::from([
                MemoryEffectDescriptor::new(R),
                MemoryEffectDescriptor::new(A),
                arg(W, 0),
            ])
        );
    }

    #[test]
    fn later_indices_shift() {
        let effects = DeclaredEffects::from([arg(R, 2), res(W, 1)]);
        let remapped = remap_declared_effects(&effects, &[0..2, 2..3, 3..4], &[0..1, 1..4]);
        assert_eq!(remapped, DeclaredEffects::from([arg(R, 3), res(W, 1), res(W, 2), res(W, 3)]));
    }

    #[test]
    fn duplicates_the_remap_produces_are_kept_once_in_first_occurrence_order() {
        // both parameters became the same value, and a dropped one makes two effects on no value
        let effects = DeclaredEffects::from([
            arg(R, 1),
            arg(R, 0),
            arg(W, 2),
            arg(W, 3),
            MemoryEffectDescriptor::new(W),
        ]);
        let remapped = remap_declared_effects(&effects, &[0..2, 0..2, 2..2, 2..2], &[]);
        assert_eq!(
            remapped,
            DeclaredEffects::from([arg(R, 0), arg(R, 1), MemoryEffectDescriptor::new(W)])
        );
    }

    #[test]
    fn unknown_and_empty_effects_are_unchanged() {
        assert_eq!(
            remap_declared_effects(&DeclaredEffects::Unknown, &[r(0, 2)], &[r(0, 1)]),
            DeclaredEffects::Unknown
        );
        assert_eq!(
            remap_declared_effects(&DeclaredEffects::none(), &[r(0, 2)], &[r(0, 1)]),
            DeclaredEffects::none()
        );
    }

    #[test]
    fn an_effect_on_no_value_stays_on_no_value() {
        let effects = DeclaredEffects::from([MemoryEffectDescriptor::new(R)]);
        assert_eq!(remap_declared_effects(&effects, &[r(0, 3)], &[r(0, 3)]), effects);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "names argument 1")]
    fn an_index_outside_the_ranges_is_a_caller_error_in_debug_builds() {
        remap_declared_effects(&DeclaredEffects::from([arg(R, 1)]), &[r(0, 1)], &[]);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn an_index_outside_the_ranges_is_an_effect_on_no_value_in_release_builds() {
        let effects = DeclaredEffects::from([arg(R, 1), res(W, 0)]);
        assert_eq!(
            remap_declared_effects(&effects, &[r(0, 1)], &[]),
            DeclaredEffects::from([MemoryEffectDescriptor::new(R), MemoryEffectDescriptor::new(W)])
        );
    }
}
