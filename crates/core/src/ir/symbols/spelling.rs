//! The textual spelling of symbol paths: segments joined with `::`, a leading `::` for an
//! absolute path, and a segment quoted with `"` when it is not a bare identifier.

use alloc::string::String;

use crate::SmallVec;

/// Returns true when `segment` is not a bare identifier (`[A-Za-z_][A-Za-z0-9_]*`).
pub(crate) fn requires_quoting(segment: &str) -> bool {
    segment.is_empty()
        || segment.starts_with(|c: char| c.is_ascii_digit())
        || !segment.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Splits `path` into whether it is absolute and its raw (unquoted) segments.
///
/// Returns `None` for an invalid spelling: an empty segment, a trailing `::`, an unclosed
/// quote, a quote inside a bare segment, or text after a closing quote.
pub(crate) fn split(path: &str) -> Option<(bool, SmallVec<[&str; 3]>)> {
    let (absolute, mut rest) = match path.strip_prefix("::") {
        Some(rest) => (true, rest),
        None => (false, path),
    };
    let mut segments = SmallVec::new();
    if rest.is_empty() {
        return absolute.then_some((true, segments));
    }
    loop {
        let quoted = rest.strip_prefix('"');
        let (segment, tail) = match quoted {
            Some(inner) => {
                let end = inner.find('"')?;
                (&inner[..end], &inner[end + 1..])
            }
            None => {
                let end = rest.find("::").unwrap_or(rest.len());
                (&rest[..end], &rest[end..])
            }
        };
        if segment.is_empty() || (quoted.is_none() && segment.contains('"')) {
            return None;
        }
        segments.push(segment);
        match tail.strip_prefix("::") {
            Some(next) if !next.is_empty() => rest = next,
            Some(_) => return None,
            None if tail.is_empty() => break,
            None => return None,
        }
    }
    Some((absolute, segments))
}

/// Joins raw `segments` into a path spelling, quoting each segment that needs it.
///
/// A segment already wrapped in quotes loses them if it does not need them; an empty segment
/// is dropped.
pub(crate) fn join<'a>(absolute: bool, segments: impl IntoIterator<Item = &'a str>) -> String {
    let mut out = String::new();
    if absolute {
        out.push_str("::");
    }
    let mut first = true;
    for segment in segments {
        let raw = match segment.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            Some(inner) => inner,
            None => segment,
        };
        if raw.is_empty() {
            continue;
        }
        if !first {
            out.push_str("::");
        }
        first = false;
        if requires_quoting(raw) {
            out.push('"');
            out.push_str(raw);
            out.push('"');
        } else {
            out.push_str(raw);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_handles_roots_quotes_and_rejects_bad_spellings() {
        assert_eq!(split("a::b"), Some((false, SmallVec::from_slice(&["a", "b"]))));
        assert_eq!(split("::a::b"), Some((true, SmallVec::from_slice(&["a", "b"]))));
        assert_eq!(
            split("\"acme::pkg\"::app"),
            Some((false, SmallVec::from_slice(&["acme::pkg", "app"])))
        );
        assert_eq!(split("::"), Some((true, SmallVec::new())));
        for bad in ["", "a::", "a::::b", "\"open", "a\"b", "\"a\"b"] {
            assert_eq!(split(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn join_quotes_only_what_needs_it_and_round_trips() {
        assert_eq!(join(false, ["acme::pkg", "app"]), "\"acme::pkg\"::app");
        assert_eq!(join(true, ["a", "b"]), "::a::b");
        assert_eq!(join(false, ["\"app\"", "", "1st"]), "app::\"1st\"");
        assert_eq!(join(false, ["::"]), "\"::\"");
        let spelled = join(true, ["a:b@1.0.0", "m"]);
        assert_eq!(split(&spelled), Some((true, SmallVec::from_slice(&["a:b@1.0.0", "m"]))));
    }
}
