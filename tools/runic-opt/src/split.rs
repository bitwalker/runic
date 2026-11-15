//! Splitting an input file into the cases it holds.

/// The marker `--split-input-file` splits at when none is given.
pub const DEFAULT_MARKER: &str = "// -----";

/// One case of a file.
pub struct Case {
    /// The case's text, preceded by a newline for each line of the file before it, so that a
    /// position in it has the line number it has in the file
    pub text: String,
    /// The line (counted from 0) of the file at which the case begins: the lines before it in
    /// `text` are padding
    pub first_line: usize,
    /// Whether the case holds only whitespace and `//` comments: it has no operation
    pub empty: bool,
}

/// Splits `text` at each line that is, once surrounded whitespace is trimmed, exactly `marker`,
/// into its cases.
///
/// The text is not read as IR: a marker line inside a string literal that spans lines splits it
/// too.
pub fn split(text: &str, marker: &str) -> Vec<Case> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut line_of_start = 0;
    let mut offset = 0;
    for (line, content) in text.split_inclusive('\n').enumerate() {
        if content.trim() == marker {
            pieces.push(case(&text[start..offset], line_of_start));
            start = offset + content.len();
            line_of_start = line + 1;
        }
        offset += content.len();
    }
    pieces.push(case(&text[start..], line_of_start));
    pieces
}

/// The whole of `text` as one case.
pub fn whole(text: String) -> Case {
    Case {
        empty: is_empty(&text),
        text,
        first_line: 0,
    }
}

/// The case `text`, which begins at line `line` (counted from 0).
fn case(text: &str, line: usize) -> Case {
    let mut padded = "\n".repeat(line);
    padded.push_str(text);
    Case {
        text: padded,
        first_line: line,
        empty: is_empty(text),
    }
}

/// Whether `text` holds only whitespace and `//` comments.
fn is_empty(text: &str) -> bool {
    text.lines().all(|line| {
        let line = line.trim();
        line.is_empty() || line.starts_with("//")
    })
}
