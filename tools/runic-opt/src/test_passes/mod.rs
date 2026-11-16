//! Passes that exist to test, from text, what has no pass of its own: dialect conversion, a pass
//! that fails, and the integer range analysis. Each is registered like any pass, under a name that
//! begins with `test-`. A fourth, `test-panic`, tests how the tool reports a defect of its own and
//! is not listed.

mod convert_wide_ints;
mod fail;
pub(crate) mod panic;
mod print_int_ranges;

use runic_ir::Report;

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
