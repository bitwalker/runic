//! `--verify-roundtrip`: that both forms of the parsed operation read back to the same program.
//!
//! For each of the custom and the generic form, the operation is printed, the text is parsed in a
//! context of its own (with the parser configured as for the input: verification as the options
//! say, the same region depth limit), and the result is printed in the same form. The two texts
//! are the same program when they are equal after value and block names are renumbered in order
//! of first appearance (see [`runic_core::print::same_program`]). What is parsed lives and dies
//! inside the check.

use std::rc::Rc;

use runic_core::{
    Context, OpPrintingFlags, OperationRef,
    diagnostics::{NoteKind, OwnedDiagnostic, diagnostic},
    parse::{ParserConfig, parse_any},
    print::{renumbered, same_program},
};

use crate::{MAX_REGION_DEPTH, print_operation, render};

/// Checks both forms of `op`; returns the diagnostics of the forms that do not read back, which
/// have no location in the input.
pub fn check(op: OperationRef, verify: bool) -> Vec<OwnedDiagnostic> {
    let mut errors = Vec::new();
    for (form, generic) in [("custom", false), ("generic", true)] {
        if let Err(error) = check_form(op, form, generic, verify) {
            errors.push(error);
        }
    }
    errors
}

fn check_form(
    op: OperationRef,
    form: &str,
    generic: bool,
    verify: bool,
) -> Result<(), OwnedDiagnostic> {
    let flags = OpPrintingFlags {
        generic,
        ..OpPrintingFlags::default()
    };
    let printed = print_operation(op, &flags);

    // Everything parsed from the text is made in this context, and dropped before it is.
    let context = Rc::new(Context::default());
    let config = ParserConfig::new(context.clone())
        .with_max_region_depth(MAX_REGION_DEPTH)
        .verify_after_parse(verify);
    let name = format!("<{form} form>");
    let parsed = parse_any(config, name.as_str(), printed.clone());
    let reparsed = match parsed.result {
        Ok(reparsed) if !parsed.diagnostics.has_errors() => reparsed,
        _ => {
            let mut error = diagnostic!(
                severity: Error,
                message: (format!("the {form} form does not parse back")),
            );
            // Their locations are in the printed text, not in the input: they are rendered into
            // the note, with the printed text's lines.
            for entry in parsed.diagnostics.iter() {
                let rendered = render(entry.diagnostic.display());
                error.push_note(
                    NoteKind::Note,
                    format_args!("parsing it said:\n{}", rendered.trim_end()),
                );
            }
            return Err(OwnedDiagnostic::new(error));
        }
    };
    let again = print_operation(reparsed, &flags);
    drop(parsed.diagnostics);
    if same_program(&printed, &again) {
        return Ok(());
    }
    let (before, after) = (renumbered(&printed), renumbered(&again));
    let (first, second) = first_difference(&before, &after);
    let (first, second) = (first.trim(), second.trim());
    let mut error = diagnostic!(
        severity: Error,
        message: (format!("the {form} form does not print the same program when parsed back")),
    );
    error.push_note(
        NoteKind::Note,
        format_args!("first differing line, printed from the input: {first}"),
    );
    error.push_note(
        NoteKind::Note,
        format_args!("first differing line, printed after parsing it back: {second}"),
    );
    Err(OwnedDiagnostic::new(error))
}

/// The first line of `a` and the first line of `b` at which they differ (an empty line stands for
/// a text that ended).
fn first_difference<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    let mut a = a.lines();
    let mut b = b.lines();
    loop {
        match (a.next(), b.next()) {
            (Some(x), Some(y)) if x == y => continue,
            (x, y) => return (x.unwrap_or(""), y.unwrap_or("")),
        }
    }
}
