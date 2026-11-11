use core::fmt;

use crate::{
    AttrPrinter, CompactString,
    derive::DialectAttribute,
    diagnostics::{ColumnNumber, LineNumber, SourceId, SourceSpan},
    dialects::builtin::BuiltinDialect,
    print::AsmPrinter,
};

/// This enumeration represents the various ways in which source locations can be encoded as
/// attributes associated with an operation.
///
/// Typically, source locations are given via [SourceSpan], but
/// when parsing the IR, other location types are possible.
#[derive(DialectAttribute, Debug, Clone, Default, PartialEq, Eq, Hash)]
#[attribute(
    name = "loc",
    dialect = BuiltinDialect,
    implements(AttrPrinter)
)]
pub enum Location {
    /// Represents an unknown location, i.e. the information is simply unavailable
    #[default]
    Unknown,
    /// Represents the location associated with compiler-generated IR, for which there is no
    /// sensible corresponding source location to use. This is a special case of `Unknown`, but
    /// tells us that it is intentionally unspecified, rather than missing.
    Synthetic,
    /// An opaque location whose concrete representation is not yet known, or is only meaningful
    /// to something that is processing the IR.
    ///
    /// For us, we use this in parsing for location aliases, to represent references to locations
    /// we haven't parsed yet.
    Opaque(usize),
    /// A reference to the first character of the relevant source code in a file, given as the
    /// line and column where the character can be found.
    FileLineCol {
        file: CompactString,
        line: LineNumber,
        column: ColumnNumber,
    },
    /// A reference to a multi-character range comprising the relevant source code in a file, given
    /// as the starting and ending line and column numbers of the selection.
    FileLineColRange {
        file: CompactString,
        start_line: LineNumber,
        start_column: ColumnNumber,
        end_line: LineNumber,
        end_column: ColumnNumber,
    },
}

impl AttrPrinter for LocationAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_trailing_location_specifier(&self.value);
    }
}

impl Location {
    pub fn from_span(span: SourceSpan, context: &crate::Context) -> Self {
        use crate::diagnostics::SourceProvider;
        if span.is_unknown() {
            return Self::Unknown;
        }
        if span.is_synthetic() {
            return Self::Synthetic;
        }
        if let Some(index) = Self::is_deferred(span) {
            return Self::Opaque(index);
        }
        let sources = context.sources();
        let id = span.source().id();
        let (Some(source), Some(start)) =
            (sources.get(id), sources.line_column(id, span.range().start()))
        else {
            return Self::Unknown;
        };
        let file = CompactString::from(source.display_name);
        if span.is_empty() {
            return Self::FileLineCol {
                file,
                line: start.line(),
                column: start.column(),
            };
        }
        match sources.line_column(id, span.range().end()) {
            Some(end) => Self::FileLineColRange {
                file,
                start_line: start.line(),
                start_column: start.column(),
                end_line: end.line(),
                end_column: end.column(),
            },
            None => Self::FileLineCol {
                file,
                line: start.line(),
                column: start.column(),
            },
        }
    }

    pub fn is_deferred(span: SourceSpan) -> Option<usize> {
        let start = span.range().start();
        (span.source().id().is_unknown() && start < u32::MAX).then_some(start as usize)
    }

    pub fn deferred(index: u32) -> SourceSpan {
        assert!(index < u32::MAX);
        SourceSpan::session(
            SourceId::UNKNOWN,
            crate::diagnostics::TextRange::new(index, index).expect("valid range"),
        )
    }

    pub fn try_into_span(&self, context: &crate::Context) -> Option<SourceSpan> {
        use crate::diagnostics::{LineColumn, TextRange};
        let resolve = |file: &str| {
            let sources = context.sources();
            if let Some(id) = sources.find_by_name(file) {
                return Some(id);
            }
            #[cfg(feature = "std")]
            {
                context.load_source_file(file).ok()
            }
            #[cfg(not(feature = "std"))]
            {
                None
            }
        };
        match self {
            Self::Unknown => Some(SourceSpan::UNKNOWN),
            Self::Synthetic => Some(SourceSpan::SYNTHETIC),
            Self::Opaque(index) => Some(Self::deferred((*index).try_into().unwrap())),
            Self::FileLineCol { file, line, column } => {
                let id = resolve(file)?;
                let at =
                    context.sources().offset_of(id, LineColumn::new_unchecked(*line, *column))?;
                Some(SourceSpan::session(id, TextRange::new(at, at).ok()?))
            }
            Self::FileLineColRange {
                file,
                start_line,
                start_column,
                end_line,
                end_column,
            } => {
                let id = resolve(file)?;
                let sources = context.sources();
                let start =
                    sources.offset_of(id, LineColumn::new_unchecked(*start_line, *start_column))?;
                let end =
                    sources.offset_of(id, LineColumn::new_unchecked(*end_line, *end_column))?;
                Some(SourceSpan::session(id, TextRange::new(start, end).ok()?))
            }
        }
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("unknown"),
            Self::Synthetic => f.write_str("synthetic"),
            Self::Opaque(id) => write!(f, "{id}"),
            Self::FileLineCol { file, line, column } => {
                write!(f, "{file}:{line}:{column}")
            }
            Self::FileLineColRange {
                file,
                start_line,
                start_column,
                end_line,
                end_column,
            } => {
                write!(f, "{file}:{start_line}:{start_column} to {end_line}:{end_column}")
            }
        }
    }
}

impl crate::formatter::PrettyPrint for Location {
    fn render(&self) -> crate::formatter::Document {
        use crate::formatter::*;
        display(self)
    }
}
