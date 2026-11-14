use core::{cell::Cell, fmt};

pub use miden_formatting::{
    hex::{DisplayHex, ToHex, to_hex},
    prettier::*,
    pretty_via_display, pretty_via_to_string,
};

pub struct DisplayIndent(pub usize);
impl fmt::Display for DisplayIndent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const INDENT: &str = "  ";
        for _ in 0..self.0 {
            f.write_str(INDENT)?;
        }
        Ok(())
    }
}

/// Render an iterator of `T`, comma-separated
pub struct DisplayValues<T>(Cell<Option<T>>);
impl<T> DisplayValues<T> {
    pub fn new(inner: T) -> Self {
        Self(Cell::new(Some(inner)))
    }
}
impl<T, I> fmt::Display for DisplayValues<I>
where
    T: fmt::Display,
    I: Iterator<Item = T>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iter = self.0.take().unwrap();
        for (i, item) in iter.enumerate() {
            if i == 0 {
                write!(f, "{item}")?;
            } else {
                write!(f, ", {item}")?;
            }
        }
        Ok(())
    }
}

/// Render an iterator of `T`, comma-separated
pub struct DisplayMany<T, U> {
    iterator: Cell<Option<T>>,
    separator: U,
}
impl<T, U> DisplayMany<T, U> {
    pub fn new(inner: T, separator: U) -> Self {
        Self {
            iterator: Cell::new(Some(inner)),
            separator,
        }
    }
}
impl<T, U, I> fmt::Display for DisplayMany<I, U>
where
    T: fmt::Display,
    U: fmt::Display,
    I: Iterator<Item = T>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iter = self.iterator.take().unwrap();
        for (i, item) in iter.enumerate() {
            if i == 0 {
                write!(f, "{item}")?;
            } else {
                write!(f, "{}{item}", self.separator)?;
            }
        }
        Ok(())
    }
}

/// Render an `Option<T>` using the `Display` impl for `T`
pub struct DisplayOptional<'a, T>(pub Option<&'a T>);
impl<T: fmt::Display> fmt::Display for DisplayOptional<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0 {
            None => f.write_str("None"),
            Some(item) => write!(f, "Some({item})"),
        }
    }
}
impl<T: fmt::Display> fmt::Debug for DisplayOptional<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Joins `items` with `separator()` between them as a balanced tree, so that the depth of the
/// document grows with the logarithm of the number of items rather than with the number.
///
/// The text is that of a left fold which starts with the first item and appends a separator and
/// the next item to a non-empty result: a separator follows only a non-empty part.
pub fn join_balanced(
    items: impl IntoIterator<Item = Document>,
    mut separator: impl FnMut() -> Document,
) -> Document {
    let mut parts = alloc::vec::Vec::new();
    let mut has_content = false;
    for item in items {
        if has_content {
            parts.push(separator());
        }
        has_content |= !item.is_empty();
        parts.push(item);
    }
    concat_balanced(&mut parts)
}

/// Concatenates `parts` as a balanced tree, leaving them empty, so that a long list does not make
/// a deep document.
pub fn concat_balanced(parts: &mut [Document]) -> Document {
    match parts {
        [] => Document::Empty,
        [part] => core::mem::take(part),
        _ => {
            let (left, right) = parts.split_at_mut(parts.len() / 2);
            concat_balanced(left) + concat_balanced(right)
        }
    }
}
