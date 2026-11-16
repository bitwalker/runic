//! `--verify-diagnostics`: the diagnostics a case produces, checked against the expectations
//! written in it.
//!
//! An expectation is written in a `//` comment:
//!
//! ```text
//! // expected-error {{text}}             a diagnostic on the comment's own line
//! // expected-error@+1 {{text}}          on the line below (`@-N` above, `@+N` below)
//! // expected-error@above {{text}}       on the nearest line above, `@below` below, that is not
//!                                        itself only a comment of expectations
//! // expected-error@unknown {{text}}     without a location
//! // expected-error-re {{regex}}         whose text matches a regular expression
//! ```
//!
//! Whitespace may stand between the kind and the `@` (`expected-error @+1 {{text}}`). A location
//! designates a line of the case the comment is in: with `--split-input-file`, a line of another
//! case, or past the first or last line of the case, is an error.
//!
//! with the kinds `error`, `warning`, `info` and `hint`. The text is what lies between `{{` and
//! the last `}}` before the next expectation on the line (or the end of the line), without the
//! whitespace around it; a line may hold several expectations. `expected-` preceded by a letter,
//! a digit, `-` or `_` is not an expectation, nor is text in a string literal.
//!
//! A diagnostic matches an expectation of its kind and line when its message, the text of one of
//! its labels, one of its helps or one of its context frames contains the text (or matches the
//! expression). Its line is that of its first primary label, when the label is in the case's own
//! source: a diagnostic located in another source (written with `loc("other":..)`) matches
//! neither a line nor `@unknown`. Each diagnostic matches at most one expectation and each
//! expectation at most one diagnostic. What does not match is reported by [`mismatches`].

use std::{fmt::Write as _, ops::Range};

use regex::Regex;
use runic_ir::diagnostics::{
    ContextFrame, Diagnostic, DiagnosticSet, Label, LabelStyle, Note, NoteKind, OwnedDiagnostic,
    Report, Severity, SourceId, SourceMap, SourceProvider, SourceSpan, TextRange, VisitDiagnostic,
    diagnostic,
};

/// The name of a kind, as an expectation spells it.
pub fn kind_name(severity: Severity) -> &'static str {
    match severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
        Severity::Hint => "hint",
        _ => "diagnostic",
    }
}

const KINDS: [Severity; 4] = [Severity::Error, Severity::Warning, Severity::Info, Severity::Hint];

/// The kinds other tools have that the diagnostics here do not: an error to expect them.
const OTHER_KINDS: [&str; 2] = ["note", "remark"];

/// One expectation written in the input.
#[derive(Debug)]
pub struct Expectation {
    pub kind: Severity,
    /// The text as written
    pub text: String,
    /// The expression the text is, with `-re`
    regex: Option<Regex>,
    /// The line the diagnostic is expected on (counted from 1), or `None` for `@unknown`
    pub line: Option<u32>,
    /// Where the expectation is written, in the case's text
    pub range: Range<usize>,
}

impl Expectation {
    fn matches(&self, found: &Found) -> bool {
        let at = match (self.line, found.location) {
            (Some(line), Location::Line(at)) => line == at,
            (None, Location::Unknown) => true,
            _ => false,
        };
        self.kind == found.kind
            && at
            && found.texts.iter().any(|text| match &self.regex {
                Some(regex) => regex.is_match(text),
                None => text.contains(self.text.as_str()),
            })
    }
}

/// An expectation that cannot be read, and why.
#[derive(Debug)]
pub struct Malformed {
    pub range: Range<usize>,
    pub message: String,
}

/// The expectations written in `text`, a case whose lines before `first_line` (counted from 0)
/// are padding, or what is wrong with those that cannot be read.
pub fn scan(text: &str, first_line: usize) -> Result<Vec<Expectation>, Vec<Malformed>> {
    let lines = Lines::new(text, first_line);
    let mut expectations = Vec::new();
    let mut malformed = Vec::new();
    for (index, line) in lines.lines.iter().enumerate() {
        let Some(comment) = line.comment else {
            continue;
        };
        if index < first_line {
            continue;
        }
        let content = &text[line.range.clone()];
        let mut from = comment;
        while let Some(token) = next_token(content, from) {
            let start = line.range.start + token.start;
            match read_expectation(content, &token, index, &lines) {
                Ok((expectation, end)) => {
                    expectations.push(Expectation {
                        range: start..line.range.start + end,
                        ..expectation
                    });
                    from = end;
                }
                Err((message, end)) => {
                    malformed.push(Malformed {
                        range: start..line.range.start + end,
                        message,
                    });
                    from = end;
                }
            }
        }
    }
    if malformed.is_empty() {
        Ok(expectations)
    } else {
        Err(malformed)
    }
}

/// The lines of a case's text.
struct Lines {
    lines: Vec<Line>,
    /// The first line of the case: the lines before it are padding
    first: usize,
}

struct Line {
    /// The line's text without its line break
    range: Range<usize>,
    /// Where its `//` comment begins, if it has one
    comment: Option<usize>,
    /// Whether the line holds only a comment with expectations in it
    only_expectations: bool,
}

impl Lines {
    fn new(text: &str, first: usize) -> Self {
        let mut lines = Vec::new();
        let mut offset = 0;
        for content in text.split_inclusive('\n') {
            let line = content.trim_end_matches(['\n', '\r']);
            let comment = comment_start(line);
            let only_expectations = comment.is_some_and(|comment| {
                line[..comment].trim().is_empty() && next_token(line, comment).is_some()
            });
            lines.push(Line {
                range: offset..offset + line.len(),
                comment,
                only_expectations,
            });
            offset += content.len();
        }
        Self { lines, first }
    }
}

/// Where the `//` comment of `line` begins: the first `//` outside a string literal.
fn comment_start(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut in_string = false;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' if in_string => index += 1,
            b'"' => in_string = !in_string,
            b'/' if !in_string && bytes.get(index + 1) == Some(&b'/') => return Some(index),
            _ => {}
        }
        index += 1;
    }
    None
}

/// An `expected-<word>` in a line: where it begins and the word.
struct Token<'a> {
    start: usize,
    /// Where the word ends
    end: usize,
    word: &'a str,
}

/// The first `expected-<kind>` at or after `from` in `line` that is not part of a longer word,
/// with one of the kinds here or of other tools.
fn next_token(line: &str, mut from: usize) -> Option<Token<'_>> {
    const PREFIX: &str = "expected-";
    while let Some(found) = line[from..].find(PREFIX) {
        let start = from + found;
        from = start + PREFIX.len();
        let preceded = line[..start]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if preceded {
            continue;
        }
        let word_len = line[from..]
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(line.len() - from);
        let word = &line[from..from + word_len];
        if KINDS.iter().any(|kind| kind_name(*kind) == word) || OTHER_KINDS.contains(&word) {
            return Some(Token {
                start,
                end: from + word_len,
                word,
            });
        }
    }
    None
}

/// Reads the expectation that `token` begins, on line `index` (counted from 0). Returns it and
/// where it ends in the line, or why it cannot be read and where to go on looking.
fn read_expectation(
    line: &str,
    token: &Token<'_>,
    index: usize,
    lines: &Lines,
) -> Result<(Expectation, usize), (String, usize)> {
    let fail = |message: String, end: usize| Err((message, end));
    let Some(kind) = KINDS.iter().copied().find(|kind| kind_name(*kind) == token.word) else {
        return fail(
            format!(
                "'{}' is not a kind of diagnostic: the kinds are error, warning, info and hint",
                token.word
            ),
            token.end,
        );
    };
    let mut at = token.end;
    let is_regex = line[at..].starts_with("-re")
        && !line[at + 3..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if is_regex {
        at += 3;
    }

    let mut target = Some(index as u32 + 1);
    // `expected-error @+1`: whitespace may stand before the location.
    let spaced = line[at..].trim_start();
    if spaced.starts_with('@') {
        at = line.len() - spaced.len();
    }
    if line[at..].starts_with('@') {
        let len = line[at + 1..]
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '+' && c != '-')
            .unwrap_or(line.len() - at - 1);
        let location = &line[at + 1..at + 1 + len];
        let end = at + 1 + len;
        target = match resolve(location, index, lines) {
            Ok(target) => target,
            Err(message) => return fail(message, end),
        };
        at = end;
    }

    let rest = &line[at..];
    let open = at + (rest.len() - rest.trim_start().len());
    if !line[open..].starts_with("{{") {
        return fail(
            format!(
                "the text of 'expected-{}' must follow it between '{{{{' and '}}}}'",
                token.word
            ),
            at,
        );
    }
    let body = open + 2;
    // The text ends at the last `}}` before the next expectation that follows a `}}`.
    let mut limit = line.len();
    let mut from = body;
    while let Some(next) = next_token(line, from) {
        if line[body..next.start].contains("}}") {
            limit = next.start;
            break;
        }
        from = next.end;
    }
    let Some(close) = line[body..limit].rfind("}}").map(|close| body + close) else {
        return fail(String::from("the text of the expectation has no closing '}}'"), limit);
    };
    let end = close + 2;
    let text = line[body..close].trim();
    if text.is_empty() {
        return fail(String::from("the expectation has no text"), end);
    }
    let regex = if is_regex {
        match Regex::new(text) {
            Ok(regex) => Some(regex),
            Err(error) => {
                return fail(format!("the text is not a regular expression: {error}"), end);
            }
        }
    } else {
        None
    };
    Ok((
        Expectation {
            kind,
            text: String::from(text),
            regex,
            line: target,
            range: 0..0,
        },
        end,
    ))
}

/// The line `@location` on line `index` (counted from 0) designates, counted from 1, or `None`
/// for `@unknown`.
fn resolve(location: &str, index: usize, lines: &Lines) -> Result<Option<u32>, String> {
    let count = lines.lines.len();
    let first = lines.first;
    let target = match location {
        "unknown" => return Ok(None),
        "above" => (first..index).rev().find(|line| !lines.lines[*line].only_expectations),
        "below" => (index + 1..count).find(|line| !lines.lines[*line].only_expectations),
        _ => {
            let offset = location
                .strip_prefix('+')
                .map(|n| (n, true))
                .or_else(|| location.strip_prefix('-').map(|n| (n, false)))
                .filter(|(n, _)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|(n, down)| n.parse::<usize>().ok().map(|n| (n, down)));
            let Some((n, down)) = offset else {
                return Err(format!(
                    "'@{location}' is not a location: write '@+N', '@-N', '@above', '@below' or \
                     '@unknown'"
                ));
            };
            if down {
                index.checked_add(n).filter(|line| *line < count)
            } else {
                index.checked_sub(n).filter(|line| *line >= first)
            }
        }
    };
    match target {
        Some(line) => Ok(Some(line as u32 + 1)),
        None => Err(format!("'@{location}' designates no line of the case it is written in")),
    }
}

/// A diagnostic that was produced, as it is matched.
#[derive(Debug)]
pub struct Found {
    kind: Severity,
    message: String,
    /// The message, the text of each label, each help and each context frame
    texts: Vec<String>,
    /// Its first primary label, with the label's text
    primary: Option<(SourceSpan, Option<String>)>,
    /// Where the first primary label is
    location: Location,
    /// Its notes
    notes: Vec<(NoteKind, String)>,
    /// The messages of its context frames, innermost first
    contexts: Vec<String>,
}

/// Where a diagnostic is, as expectations see it.
#[derive(Debug, Copy, Clone)]
enum Location {
    /// On this line of the case's source, counted from 1
    Line(u32),
    /// Nowhere: it has no primary label with a known span
    Unknown,
    /// In another source than the case's
    Elsewhere,
}

/// Collects, from diagnostics, what is matched against expectations.
pub struct Collector<'a> {
    sources: &'a SourceMap,
    /// The source of the case, whose lines expectations designate
    own: Option<SourceId>,
    found: Vec<Found>,
}

impl<'a> Collector<'a> {
    pub fn new(sources: &'a SourceMap, own: Option<SourceId>) -> Self {
        Self {
            sources,
            own,
            found: Vec::new(),
        }
    }

    pub fn finish(self) -> Vec<Found> {
        self.found
    }

    pub fn set(&mut self, diagnostics: &DiagnosticSet) {
        for entry in diagnostics.iter() {
            self.owned(&entry.diagnostic, entry.effective_severity);
        }
    }

    pub fn owned(&mut self, diagnostic: &OwnedDiagnostic, kind: Severity) {
        self.diagnostic(diagnostic, kind, diagnostic.contexts());
    }

    pub fn report(&mut self, report: &Report) {
        self.diagnostic(report.as_diagnostic(), report.severity(), report.contexts());
    }

    /// Collects `diagnostic` and the diagnostics bundled with it.
    fn diagnostic(
        &mut self,
        diagnostic: &dyn Diagnostic,
        kind: Severity,
        contexts: &[ContextFrame],
    ) {
        let mut message = String::new();
        let _ = diagnostic.message(&mut message);
        let mut visitor = Visitor {
            labels: Vec::new(),
            notes: Vec::new(),
            related: Collector::new(self.sources, self.own),
        };
        diagnostic.visit(&mut visitor);

        let mut texts = vec![message.clone()];
        texts.extend(visitor.labels.iter().filter_map(|(_, _, text)| text.clone()));
        texts.extend(
            visitor
                .notes
                .iter()
                .filter(|(kind, _)| *kind == NoteKind::Help)
                .map(|(_, text)| text.clone()),
        );
        texts.extend(contexts.iter().map(|frame| String::from(frame.message())));
        let primary = visitor
            .labels
            .iter()
            .find(|(span, style, _)| *style == LabelStyle::Primary && !span.is_unknown())
            .map(|(span, _, text)| (*span, text.clone()));
        let location = match &primary {
            None => Location::Unknown,
            Some((span, _)) if Some(span.source().id()) == self.own => self
                .sources
                .line_column(span.source().id(), span.range().start())
                .map_or(Location::Elsewhere, |location| Location::Line(location.line().to_u32())),
            Some(_) => Location::Elsewhere,
        };
        self.found.push(Found {
            kind,
            message,
            texts,
            primary,
            location,
            notes: visitor.notes,
            contexts: contexts.iter().map(|frame| String::from(frame.message())).collect(),
        });
        self.found.extend(visitor.related.finish());
    }
}

struct Visitor<'a> {
    labels: Vec<(SourceSpan, LabelStyle, Option<String>)>,
    notes: Vec<(NoteKind, String)>,
    related: Collector<'a>,
}

impl VisitDiagnostic for Visitor<'_> {
    fn label(&mut self, label: Label<'_>) {
        let text = label.message.map(|message| {
            let mut text = String::new();
            let _ = text.write_fmt(message);
            text
        });
        self.labels.push((label.span, label.style, text));
    }

    fn note(&mut self, note: Note<'_>) {
        let mut text = String::new();
        let _ = text.write_fmt(note.message);
        self.notes.push((note.kind, text));
    }

    fn related(&mut self, diagnostic: &dyn Diagnostic) {
        self.related.diagnostic(diagnostic, diagnostic.severity(), &[]);
    }
}

/// The errors that say what did not match, in the case whose source is `source`: each diagnostic
/// that matched no expectation, rendered as it was (its labels, notes and context frames) under
/// `unexpected <kind>: <message>`, and each expectation that matched no diagnostic, at the
/// comment.
pub fn mismatches(
    expectations: &[Expectation],
    found: &[Found],
    source: Option<SourceId>,
) -> Vec<OwnedDiagnostic> {
    let (unexpected, missing) = unmatched(expectations, found);
    let mut mismatches = Vec::new();
    for found in unexpected {
        let message = format!("unexpected {}: {}", kind_name(found.kind), found.message);
        let mut error = match &found.primary {
            Some((span, Some(label))) => diagnostic!(
                severity: Error,
                message: (message),
                labels: [primary(*span, (label.as_str()))],
            ),
            Some((span, None)) => {
                diagnostic!(severity: Error, message: (message), labels: [primary(*span)],)
            }
            None => diagnostic!(severity: Error, message: (message)),
        };
        for (kind, note) in &found.notes {
            error.push_note(*kind, format_args!("{note}"));
        }
        let mut error = OwnedDiagnostic::new(error);
        for context in &found.contexts {
            error.push_context(context.as_str());
        }
        mismatches.push(error);
    }
    for expectation in missing {
        let message = format!(
            "expected {} \"{}\" was not produced",
            kind_name(expectation.kind),
            expectation.text
        );
        let span = source.and_then(|id| {
            TextRange::try_from_usize(expectation.range.start, expectation.range.end)
                .ok()
                .map(|range| SourceSpan::session(id, range))
        });
        mismatches.push(OwnedDiagnostic::new(match span {
            Some(span) => {
                diagnostic!(severity: Error, message: (message), labels: [primary(span)],)
            }
            None => diagnostic!(severity: Error, message: (message)),
        }));
    }
    mismatches
}

/// The diagnostics and the expectations that matched nothing, in the order they were produced
/// and written.
fn unmatched<'a>(
    expectations: &'a [Expectation],
    found: &'a [Found],
) -> (Vec<&'a Found>, Vec<&'a Expectation>) {
    // The largest matching of diagnostics to expectations, by augmenting paths: an expectation
    // that a later diagnostic needs is given up by an earlier one that has another.
    let mut matched: Vec<Option<usize>> = vec![None; expectations.len()];
    for diagnostic in 0..found.len() {
        let mut seen = vec![false; expectations.len()];
        augment(diagnostic, expectations, found, &mut matched, &mut seen);
    }
    let mut taken = vec![false; found.len()];
    for diagnostic in matched.iter().flatten() {
        taken[*diagnostic] = true;
    }
    let diagnostics = found.iter().zip(taken).filter(|(_, taken)| !taken).map(|(found, _)| found);
    let expectations = expectations
        .iter()
        .zip(&matched)
        .filter(|(_, matched)| matched.is_none())
        .map(|(expectation, _)| expectation);
    (diagnostics.collect(), expectations.collect())
}

fn augment(
    diagnostic: usize,
    expectations: &[Expectation],
    found: &[Found],
    matched: &mut [Option<usize>],
    seen: &mut [bool],
) -> bool {
    for (index, expectation) in expectations.iter().enumerate() {
        if seen[index] || !expectation.matches(&found[diagnostic]) {
            continue;
        }
        seen[index] = true;
        let free = match matched[index] {
            None => true,
            Some(other) => augment(other, expectations, found, matched, seen),
        };
        if free {
            matched[index] = Some(diagnostic);
            return true;
        }
    }
    false
}
