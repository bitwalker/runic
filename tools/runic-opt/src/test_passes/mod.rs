//! Passes that exist to test, from text, what has no pass of its own: dialect conversion, a pass
//! that fails, and the integer range analysis. Each is registered like any pass, under a name that
//! begins with `test-`.

mod convert_wide_ints;
mod fail;
mod print_int_ranges;

use runic_core::Report;

/// The `key=value` and `key` pairs of a pass's options, as the pipeline hands them over (joined by
/// `, `), each split into its key and its value.
fn options(options: &str) -> impl Iterator<Item = (&str, Option<&str>)> {
    options
        .split(',')
        .map(str::trim_start)
        .filter(|pair| !pair.trim().is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((key, value)) => (key.trim(), Some(value)),
            None => (pair.trim(), None),
        })
}

/// A switch given as `key`, `key=true` or `key=false`.
fn switch(pass: &str, key: &str, value: Option<&str>) -> Result<bool, Report> {
    match value.map(str::trim) {
        None | Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(value) => Err(Report::msg(format!(
            "invalid value '{value}' of option '{key}' of pass '{pass}': expected 'true' or \
             'false'"
        ))),
    }
}

/// The error for an option `key` that `pass` does not take.
fn unknown(pass: &str, key: &str, known: &str) -> Report {
    Report::msg(format!("pass '{pass}' has no option '{key}'; it takes {known}"))
}
