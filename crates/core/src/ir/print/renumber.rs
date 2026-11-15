//! Comparing printed texts as programs: the names of values and blocks are renumbered in the order
//! they first appear.

use alloc::string::String;
use core::fmt::Write;

use crate::FxHashMap;

/// `text` with every value and block name replaced by a number given in the order the names first
/// appear: the first value named is `%0`, the next value with another name `%1`, and so on, and
/// blocks likewise (`^0`, `^1`, ..). A result index (`%x#1`) or count (`%x:2`) stays as written.
///
/// Values are numbered by creation, not by their place in the text, so IR parsed from a printed
/// text may print with other numbers than the text had; two texts that differ only so are the same
/// program, and their renumbered texts are equal ([same_program]).
///
/// Names are read as the parser reads them: after `%` or `^`, either a run of digits or a run of
/// letters, digits and `_ $ . -`. Nothing in a string literal (`".."`, in which `\` escapes the
/// character after it) or a comment (`//` to the end of the line) is renamed.
pub fn renumbered(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut values = FxHashMap::<&str, usize>::default();
    let mut blocks = FxHashMap::<&str, usize>::default();
    // The start of the text not yet copied to `out`
    let mut copied = 0;
    let mut at = 0;
    while at < bytes.len() {
        match bytes[at] {
            b'"' => {
                at += 1;
                while at < bytes.len() && bytes[at] != b'"' {
                    at += if bytes[at] == b'\\' { 2 } else { 1 };
                }
                at += 1;
            }
            b'/' if bytes.get(at + 1) == Some(&b'/') => {
                at = text[at..].find('\n').map_or(bytes.len(), |end| at + end);
            }
            sigil @ (b'%' | b'^') => {
                let start = at + 1;
                let end = name_end(bytes, start);
                if end == start {
                    at += 1;
                    continue;
                }
                let names = if sigil == b'%' {
                    &mut values
                } else {
                    &mut blocks
                };
                let next = names.len();
                let number = *names.entry(&text[start..end]).or_insert(next);
                out.push_str(&text[copied..start]);
                write!(out, "{number}").expect("writing to a string does not fail");
                copied = end;
                at = end;
            }
            _ => at += 1,
        }
    }
    out.push_str(&text[copied.min(text.len())..]);
    out
}

/// Whether `a` and `b` are the same program: equal once their value and block names are
/// [renumbered].
pub fn same_program(a: &str, b: &str) -> bool {
    renumbered(a) == renumbered(b)
}

/// The end of the name that starts at `start` (after its sigil): a run of digits, or a run of
/// ASCII letters, digits and `_ $ . -`, as the lexer reads one.
fn name_end(bytes: &[u8], start: usize) -> usize {
    let rest = &bytes[start.min(bytes.len())..];
    let len = match rest.first() {
        Some(c) if c.is_ascii_digit() => rest.iter().take_while(|c| c.is_ascii_digit()).count(),
        Some(_) => rest
            .iter()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'$' | b'.' | b'-'))
            .count(),
        None => 0,
    };
    start + len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_numbered_by_first_appearance() {
        let text = "%7 = op %arg0, %7#1;\n^block9(%3: i32):\n    br ^block2(%3 : i32), ^block9;";
        assert_eq!(renumbered(text), "%0 = op %1, %0#1;\n^0(%2: i32):\n    br ^1(%2 : i32), ^0;");
    }

    #[test]
    fn a_result_count_and_a_digit_run_end_the_name() {
        assert_eq!(renumbered("%5:2 = op; use %5#1, %12abc"), "%0:2 = op; use %0#1, %1abc");
    }

    #[test]
    fn strings_and_comments_are_left_as_they_are() {
        let text = "%4 = op \"%4 ^b \\\" %9\" // %4 ^b\n%9 = op %4;";
        assert_eq!(renumbered(text), "%0 = op \"%4 ^b \\\" %9\" // %4 ^b\n%1 = op %0;");
    }

    #[test]
    fn a_sigil_without_a_name_is_kept() {
        assert_eq!(renumbered("a % b ^ c %"), "a % b ^ c %");
        assert_eq!(renumbered("\"unclosed %1"), "\"unclosed %1");
    }

    #[test]
    fn texts_that_differ_in_numbering_only_are_the_same_program() {
        assert!(same_program("%3 = op %1;\nuse %3;", "%9 = op %0;\nuse %9;"));
        assert!(!same_program("%3 = op %1;\nuse %3;", "%9 = op %0;\nuse %0;"));
        assert!(!same_program("%3 = op;", "%3 = other;"));
    }
}
