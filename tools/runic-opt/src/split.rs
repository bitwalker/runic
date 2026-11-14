//! Splitting an input file into the cases it holds.

/// The marker `--split-input-file` splits at when none is given.
pub const DEFAULT_MARKER: &str = "// -----";

/// One case of an input file.
#[derive(Debug, PartialEq, Eq)]
pub struct Piece {
    /// The text of the case, preceded by a newline for each line of the file before it, so that
    /// a position in it has the line number it has in the file.
    pub text: String,
}

/// Splits `text` at each line that is, once surrounded whitespace is trimmed, exactly `marker`.
///
/// The text is not read as IR: a marker line inside a string literal that spans lines splits it
/// too. A case that holds only whitespace and `//` comments is left out.
pub fn split(text: &str, marker: &str) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut line_of_start = 0;
    let mut offset = 0;
    for (line, content) in text.split_inclusive('\n').enumerate() {
        if content.trim() == marker {
            push(&mut pieces, &text[start..offset], line_of_start);
            start = offset + content.len();
            line_of_start = line + 1;
        }
        offset += content.len();
    }
    push(&mut pieces, &text[start..], line_of_start);
    pieces
}

/// Adds the case `text`, which begins at line `line` (counted from 0), unless it is empty.
fn push(pieces: &mut Vec<Piece>, text: &str, line: usize) {
    if text.lines().all(|line| {
        let line = line.trim();
        line.is_empty() || line.starts_with("//")
    }) {
        return;
    }
    let mut padded = "\n".repeat(line);
    padded.push_str(text);
    pieces.push(Piece { text: padded });
}
