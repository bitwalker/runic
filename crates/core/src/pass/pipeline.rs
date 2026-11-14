//! Pass-pipeline text: a parser, its printed form, and the pass manager it builds.
//!
//! ```text
//! pipeline ::= anchor '(' (element (',' element)*)? ')'
//! element  ::= pipeline | 'recursive' '(' pipeline ')' | pass-name options?
//! options  ::= '{' (key ('=' value)?)* '}'        (pairs separated by spaces)
//! anchor   ::= 'any' | dialect '.' operation
//! ```
//!
//! * An `anchor` names the operations a pipeline runs on: `any`, or an operation such as
//!   `builtin.function`. A nested pipeline runs on the operations of its anchor directly inside
//!   the operation of the pipeline around it; `recursive(..)` runs it on the outermost such
//!   operations at any depth (see [`OpPassManager::nest_recursive`]).
//! * A `pass-name` is the command-line name of a registered pass (see [`super::PassInfo`]), such as
//!   `cse` or `canonicalizer`.
//! * The `options` of a pass are pairs separated by whitespace (a pair is followed by whitespace
//!   or the closing `}`). A key is a bare word given at most once.
//! * A value is either a bare word (a run of characters that are not whitespace, `(){},=` or
//!   `"`, such as `foo-bar`, `3`, `-1` or `a;b`) or a string in double quotes, which may contain
//!   any character but `,`, with the escapes `\"` and `\\`. The value reaches the pass as it was
//!   written, with the quotes and escapes removed: it is text, and nothing converts it (a number
//!   is not read, so `0x10` arrives as `0x10`). The pairs are handed to
//!   [`super::Pass::initialize_options`] joined by `, ` (`key=value` or `key`), so a value must not
//!   contain `,`; a pass prints its options back with [`write_pass_option`].
//! * Whitespace may appear between any two tokens. A pipeline may be empty at any depth
//!   (`builtin.module()`, `builtin.module(builtin.function())`), as a pass manager with no passes
//!   prints itself.
//!
//! The text is the one that [`OpPassManager::print_as_textual_pipeline`] prints: the canonical form
//! ([`PassPipeline`]'s `Display`) separates elements with `,` and no space, and quotes every
//! option value. For example:
//!
//! ```text
//! builtin.module(cse,builtin.function(canonicalizer{enable-patterns="a;b"}),recursive(any(dce)))
//! ```

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use super::{Nesting, OpPassManager, PassInfo, PassManager, registry::RegistryEntry};
use crate::{Context, OperationName, Report, interner::Symbol};

/// How deeply pipelines may be nested in text, so that parsing, loading and printing a pipeline
/// cannot exhaust the stack.
const MAX_NESTING: usize = 64;

/// A pass pipeline read from text: the operations it runs on and what it runs on them.
///
/// See the [module documentation](self) for the grammar. `Display` prints the canonical text, which
/// parses back to an equal pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassPipeline {
    anchor: Anchor,
    elements: Vec<PipelineElement>,
}

/// One element of a [`PassPipeline`], in execution order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineElement {
    /// A pass, run on the operation of the pipeline.
    Pass(SelectedPass),
    /// A pipeline run on the operations of its anchor directly inside the operation of the
    /// pipeline.
    Nested(PassPipeline),
    /// A pipeline run on the outermost operations of its anchor at any depth below the operation
    /// of the pipeline: `recursive(..)`.
    Recursive(PassPipeline),
}

/// A pass named in a pipeline, with the options given for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedPass {
    name: String,
    options: Vec<(String, Option<String>)>,
}

/// The operations a [`PassPipeline`] runs on.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub enum Anchor {
    /// Any operation the passes of the pipeline can be scheduled on.
    #[default]
    Any,
    /// The operations named `dialect.opcode`; the name is resolved when the pipeline is loaded.
    Operation { dialect: Symbol, opcode: Symbol },
}

impl PassPipeline {
    /// Parses `text` as a pipeline (see the [module documentation](self)).
    ///
    /// A malformed text is an error that says what was expected and what was found, at which byte
    /// offset of `text`, with the line of `text` and a caret under the fault.
    pub fn parse(text: &str) -> Result<Self, Report> {
        let mut parser = Parser::new(text);
        let pipeline = parser.pipeline(0)?;
        parser.expect_end()?;
        Ok(pipeline)
    }

    /// Parses `text` as a list of elements with no anchor, such as `cse,canonicalizer`, and anchors
    /// the result on `anchor`. An empty `text` is an empty pipeline.
    pub fn parse_elements(anchor: &OperationName, text: &str) -> Result<Self, Report> {
        let mut parser = Parser::new(text);
        let mut elements = Vec::new();
        if !matches!(parser.peek()?.kind, Kind::End) {
            loop {
                elements.push(parser.element(0)?);
                let token = parser.next()?;
                match token.kind {
                    Kind::Comma => continue,
                    Kind::End => break,
                    _ => return Err(parser.unexpected(&token, "',' or the end of the pipeline")),
                }
            }
        }
        Ok(Self {
            anchor: Anchor::Operation {
                dialect: anchor.dialect(),
                opcode: anchor.name(),
            },
            elements,
        })
    }

    /// The operations this pipeline runs on.
    pub fn anchor(&self) -> &Anchor {
        &self.anchor
    }

    /// The passes and nested pipelines of this pipeline, in execution order.
    pub fn elements(&self) -> &[PipelineElement] {
        &self.elements
    }

    /// Builds the pass manager: every name is looked up in the registry (an unknown one is an
    /// error that lists similar names), options are given to the pass, nested pipelines become
    /// nested managers.
    ///
    /// An anchor that names no registered operation is an error. With [`Nesting::Explicit`], a
    /// pass that runs on one operation type written under an anchor of another is an error; with
    /// [`Nesting::Implicit`] it is nested under a manager for its type.
    ///
    /// A pass may normalise the options it is given (for example, print them in a fixed order), so
    /// the text the loaded manager prints is the canonical form of what was written rather than
    /// the text itself; parsing that text and loading it again prints the same text.
    pub fn load(&self, context: Rc<Context>, nesting: Nesting) -> Result<PassManager, Report> {
        let mut pm = PassManager::new(context, self.anchor.to_string(), nesting)?;
        load_elements(&self.elements, pm.op_pass_manager_mut())?;
        Ok(pm)
    }
}

fn load_elements(elements: &[PipelineElement], pm: &mut OpPassManager) -> Result<(), Report> {
    for element in elements {
        match element {
            PipelineElement::Pass(pass) => pass.add_to(pm)?,
            PipelineElement::Nested(pipeline) => {
                let mut nested = pm.nest_with_type(&pipeline.anchor.to_string())?;
                load_elements(&pipeline.elements, &mut nested)?;
            }
            PipelineElement::Recursive(pipeline) => {
                let mut nested = pm.nest_recursive_with_type(&pipeline.anchor.to_string())?;
                load_elements(&pipeline.elements, &mut nested)?;
            }
        }
    }
    Ok(())
}

impl fmt::Display for PassPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.anchor)?;
        for (i, element) in self.elements.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{element}")?;
        }
        f.write_str(")")
    }
}

impl fmt::Display for PipelineElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass(pass) => write!(f, "{pass}"),
            Self::Nested(pipeline) => write!(f, "{pipeline}"),
            Self::Recursive(pipeline) => write!(f, "recursive({pipeline})"),
        }
    }
}

impl SelectedPass {
    /// The command-line name of the pass.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The options given for the pass, in the order written; keys are unique.
    pub fn options(&self) -> &[(String, Option<String>)] {
        &self.options
    }

    /// The options as [`super::Pass::initialize_options`] receives them: `key=value` or `key`,
    /// joined by `, `.
    pub fn options_string(&self) -> String {
        let mut options = String::new();
        for (i, (key, value)) in self.options.iter().enumerate() {
            if i > 0 {
                options.push_str(", ");
            }
            options.push_str(key);
            if let Some(value) = value {
                options.push('=');
                options.push_str(value);
            }
        }
        options
    }

    fn add_to(&self, pm: &mut OpPassManager) -> Result<(), Report> {
        let Some(info) = PassInfo::lookup(&self.name) else {
            let similar = PassInfo::similar_to(&self.name);
            let hint = match similar.as_slice() {
                [] => String::new(),
                [one] => format!("; did you mean '{one}'?"),
                many => format!(
                    "; did you mean one of {}?",
                    many.iter().map(|name| format!("'{name}'")).collect::<Vec<_>>().join(", ")
                ),
            };
            return Err(Report::msg(format!("unknown pass '{}'{hint}", self.name)));
        };
        info.add_to_pipeline(pm, &self.options_string())
    }
}

impl fmt::Display for SelectedPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)?;
        if self.options.is_empty() {
            return Ok(());
        }
        f.write_str("{")?;
        for (i, (key, value)) in self.options.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write_pass_option(f, key, value.as_deref())?;
        }
        f.write_str("}")
    }
}

/// Writes one option of a pass as pipeline text: `key="value"` with `"` and `\` escaped, or `key`
/// when it has no value.
///
/// This is how a [`PassPipeline`] prints an option; a pass's
/// [`super::Pass::print_as_textual_pipeline`] writes its options with it, separated by a space and
/// enclosed in `{..}`, so that what it prints parses back to the same values.
pub fn write_pass_option<W: fmt::Write + ?Sized>(
    f: &mut W,
    key: &str,
    value: Option<&str>,
) -> fmt::Result {
    f.write_str(key)?;
    if let Some(value) = value {
        f.write_str("=\"")?;
        for c in value.chars() {
            if matches!(c, '"' | '\\') {
                f.write_char('\\')?;
            }
            f.write_char(c)?;
        }
        f.write_char('"')?;
    }
    Ok(())
}

impl Anchor {
    /// Reads `any` or `dialect.opcode`; `None` for anything else.
    fn from_word(word: &str) -> Option<Self> {
        if word == "any" {
            return Some(Self::Any);
        }
        let (dialect, opcode) = word.split_once('.')?;
        let is_ident_start = |c: char| c.is_ascii_alphabetic() || c == '_';
        let valid_dialect = dialect.starts_with(is_ident_start)
            && dialect.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        let valid_opcode = opcode.starts_with(is_ident_start)
            && opcode.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.'))
            && !opcode.ends_with('.');
        (valid_dialect && valid_opcode).then(|| Self::Operation {
            dialect: Symbol::intern(dialect),
            opcode: Symbol::intern(opcode),
        })
    }
}

impl fmt::Display for Anchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => f.write_str(OpPassManager::ANY),
            Self::Operation { dialect, opcode } => write!(f, "{dialect}.{opcode}"),
        }
    }
}

/// Whether `word` can name a pass or an option key: a letter or `_`, then letters, digits, `_`
/// and `-`.
fn is_name(word: &str) -> bool {
    word.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && word.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

#[derive(Debug)]
enum Kind {
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Equal,
    /// A run of characters that are neither whitespace, punctuation nor `"`.
    Word(String),
    /// A string in double quotes, unescaped.
    Quoted(String),
    End,
}

struct Token {
    kind: Kind,
    /// The byte offset of the token in the text.
    offset: usize,
}

struct Parser<'a> {
    text: &'a str,
    position: usize,
    peeked: Option<Token>,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            position: 0,
            peeked: None,
        }
    }

    fn peek(&mut self) -> Result<&Token, Report> {
        if self.peeked.is_none() {
            let token = self.lex()?;
            self.peeked = Some(token);
        }
        Ok(self.peeked.as_ref().unwrap())
    }

    fn next(&mut self) -> Result<Token, Report> {
        match self.peeked.take() {
            Some(token) => Ok(token),
            None => self.lex(),
        }
    }

    fn lex(&mut self) -> Result<Token, Report> {
        let rest = &self.text[self.position..];
        let trimmed = rest.trim_start();
        let offset = self.position + (rest.len() - trimmed.len());
        let mut chars = trimmed.chars();
        let Some(c) = chars.next() else {
            self.position = offset;
            return Ok(Token {
                kind: Kind::End,
                offset,
            });
        };
        let punctuation = match c {
            '(' => Some(Kind::LParen),
            ')' => Some(Kind::RParen),
            '{' => Some(Kind::LBrace),
            '}' => Some(Kind::RBrace),
            ',' => Some(Kind::Comma),
            '=' => Some(Kind::Equal),
            _ => None,
        };
        if let Some(kind) = punctuation {
            self.position = offset + 1;
            return Ok(Token { kind, offset });
        }
        if c == '"' {
            let mut value = String::new();
            let mut end = offset + 1;
            let mut chars = self.text[end..].chars();
            loop {
                match chars.next() {
                    None => {
                        return Err(self.error(
                            offset,
                            "a closing '\"'",
                            "the end of the pipeline",
                        ));
                    }
                    Some('"') => {
                        end += 1;
                        break;
                    }
                    Some('\\') => match chars.next() {
                        Some(escaped @ ('"' | '\\')) => {
                            value.push(escaped);
                            end += 2;
                        }
                        Some(other) => {
                            return Err(self.error(
                                end,
                                "the escape '\\\"' or '\\\\'",
                                &format!("'\\{other}'"),
                            ));
                        }
                        None => {
                            return Err(self.error(
                                offset,
                                "a closing '\"'",
                                "the end of the pipeline",
                            ));
                        }
                    },
                    Some(other) => {
                        value.push(other);
                        end += other.len_utf8();
                    }
                }
            }
            self.position = end;
            return Ok(Token {
                kind: Kind::Quoted(value),
                offset,
            });
        }
        let len = trimmed
            .find(|c: char| c.is_whitespace() || "(){},=\"".contains(c))
            .unwrap_or(trimmed.len());
        self.position = offset + len;
        Ok(Token {
            kind: Kind::Word(trimmed[..len].to_string()),
            offset,
        })
    }

    /// `anchor '(' (element (',' element)*)? ')'`.
    fn pipeline(&mut self, depth: usize) -> Result<PassPipeline, Report> {
        let token = self.next()?;
        let Kind::Word(word) = &token.kind else {
            return Err(self.unexpected(&token, "an anchor ('any' or '<dialect>.<operation>')"));
        };
        let Some(anchor) = Anchor::from_word(word) else {
            return Err(self.unexpected(&token, "an anchor ('any' or '<dialect>.<operation>')"));
        };
        self.expect(|kind| matches!(kind, Kind::LParen), "'(' after the anchor")?;
        self.elements(anchor, depth)
    }

    /// The elements of a pipeline after its `(`, through its `)`.
    fn elements(&mut self, anchor: Anchor, depth: usize) -> Result<PassPipeline, Report> {
        let mut elements = Vec::new();
        if matches!(self.peek()?.kind, Kind::RParen) {
            self.next()?;
        } else {
            loop {
                elements.push(self.element(depth)?);
                let token = self.next()?;
                match token.kind {
                    Kind::Comma => continue,
                    Kind::RParen => break,
                    _ => return Err(self.unexpected(&token, "',' or ')'")),
                }
            }
        }
        Ok(PassPipeline { anchor, elements })
    }

    /// `pipeline | 'recursive' '(' pipeline ')' | pass-name options?`
    fn element(&mut self, depth: usize) -> Result<PipelineElement, Report> {
        const EXPECTED: &str = "a pass name, a nested pipeline or 'recursive(..)'";
        let token = self.next()?;
        let word = match &token.kind {
            Kind::Word(word) => word.clone(),
            _ => return Err(self.unexpected(&token, EXPECTED)),
        };
        if matches!(self.peek()?.kind, Kind::LParen) {
            if depth + 1 >= MAX_NESTING {
                return Err(self.error(
                    token.offset,
                    &format!("at most {MAX_NESTING} levels of nested pipelines"),
                    "more",
                ));
            }
            if word == "recursive" {
                self.next()?;
                let pipeline = self.pipeline(depth + 1)?;
                self.expect(|kind| matches!(kind, Kind::RParen), "')' to close 'recursive('")?;
                return Ok(PipelineElement::Recursive(pipeline));
            }
            let Some(anchor) = Anchor::from_word(&word) else {
                return Err(self.error(
                    token.offset,
                    "an anchor ('any' or '<dialect>.<operation>')",
                    &format!("'{word}'"),
                ));
            };
            self.next()?;
            return Ok(PipelineElement::Nested(self.elements(anchor, depth + 1)?));
        }
        if !is_name(&word) {
            return Err(self.error(token.offset, EXPECTED, &format!("'{word}'")));
        }
        let mut pass = SelectedPass {
            name: word,
            options: Vec::new(),
        };
        if matches!(self.peek()?.kind, Kind::LBrace) {
            self.next()?;
            self.options(&mut pass)?;
        }
        Ok(PipelineElement::Pass(pass))
    }

    /// The options of `pass` after its `{`, through its `}`.
    fn options(&mut self, pass: &mut SelectedPass) -> Result<(), Report> {
        loop {
            let token = self.next()?;
            let key = match token.kind {
                Kind::RBrace => return Ok(()),
                Kind::Word(ref key) if is_name(key) => key.clone(),
                _ => return Err(self.unexpected(&token, "an option key or '}'")),
            };
            if pass.options.iter().any(|(k, _)| *k == key) {
                return Err(self.error(
                    token.offset,
                    &format!("each option of pass '{}' at most once", pass.name),
                    &format!("'{key}' again"),
                ));
            }
            let mut value = None;
            let mut end = token.offset + key.len();
            if matches!(self.peek()?.kind, Kind::Equal) {
                self.next()?;
                let token = self.next()?;
                match token.kind {
                    Kind::Word(word) => {
                        end = self.position;
                        value = Some(word);
                    }
                    Kind::Quoted(quoted) if quoted.contains(',') => {
                        return Err(self.error(
                            token.offset,
                            &format!("a value for option '{key}' without ','"),
                            "a value that contains ','",
                        ));
                    }
                    Kind::Quoted(quoted) => {
                        end = self.position;
                        value = Some(quoted);
                    }
                    _ => {
                        return Err(self.unexpected(
                            &token,
                            &format!(
                                "a value for option '{key}' (a word, number or quoted string)"
                            ),
                        ));
                    }
                }
            }
            // Pairs are separated by whitespace: `{a="x"b="y"}` is not two options.
            let following = self.next()?;
            if following.offset == end && !matches!(following.kind, Kind::RBrace) {
                return Err(self
                    .unexpected(&following, &format!("whitespace or '}}' after option '{key}'")));
            }
            self.peeked = Some(following);
            pass.options.push((key, value));
        }
    }

    fn expect(&mut self, is: impl FnOnce(&Kind) -> bool, expected: &str) -> Result<(), Report> {
        let token = self.next()?;
        if is(&token.kind) {
            Ok(())
        } else {
            Err(self.unexpected(&token, expected))
        }
    }

    fn expect_end(&mut self) -> Result<(), Report> {
        self.expect(|kind| matches!(kind, Kind::End), "the end of the pipeline")
    }

    fn unexpected(&self, token: &Token, expected: &str) -> Report {
        let found = match &token.kind {
            Kind::LParen => "'('".to_string(),
            Kind::RParen => "')'".to_string(),
            Kind::LBrace => "'{'".to_string(),
            Kind::RBrace => "'}'".to_string(),
            Kind::Comma => "','".to_string(),
            Kind::Equal => "'='".to_string(),
            Kind::Word(word) => format!("'{word}'"),
            Kind::Quoted(_) => "a quoted string".to_string(),
            Kind::End => "the end of the pipeline".to_string(),
        };
        self.error(token.offset, expected, &found)
    }

    /// An error at byte `offset` of the text: what was expected and found, the line of the text
    /// that holds the fault and a caret under it.
    fn error(&self, offset: usize, expected: &str, found: &str) -> Report {
        let line_start = self.text[..offset].rfind('\n').map_or(0, |i| i + 1);
        let line_end = self.text[offset..].find('\n').map_or(self.text.len(), |i| offset + i);
        let column = self.text[line_start..offset].chars().count();
        Report::msg(format!(
            "invalid pass pipeline: expected {expected}, found {found} at offset {offset}\n  {}\n  \
             {}^",
            &self.text[line_start..line_end],
            " ".repeat(column)
        ))
    }
}

#[cfg(test)]
mod tests;
