//! How a command line renders a diagnostic.
//!
//! [diagnostic] is the text of a diagnostic and the sources it names, as a command line prints it:
//! the lines around a label, not the whole source.

use runic_compile::{Compilation, Emitted};
use runic_ir::diagnostics::{DiagnosticDisplay, RenderConfig};

use crate::worker::SharedBuffer;

/// How the driver renders a diagnostic: the lines around a label, not the whole source.
const fn config(styled: bool) -> RenderConfig {
    RenderConfig {
        styled,
        fold: true,
        ..RenderConfig::DEFAULT
    }
}

/// The text of `display`, a diagnostic with the sources it names, ending with a newline, with
/// colors when `styled`.
pub fn diagnostic(display: DiagnosticDisplay<'_>, styled: bool) -> String {
    let mut text = display.with_config(config(styled)).to_string();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// What a rendered compilation leaves for the thread that wrote its text to standard error.
pub(crate) struct Rendered {
    /// The bytes for standard output: each output a checkpoint delivered to it, in order.
    pub stdout: Vec<u8>,
    /// Whether the compilation failed.
    pub failed: bool,
    /// Whether the compilation failed because the request did not resolve.
    pub unresolved: bool,
}

/// Renders `compilation` to `stderr` and returns what is meant for standard output.
///
/// To `stderr`, in this order: each diagnostic that is not the error, and the error. What the
/// passes printed is already in `stderr`: the compilation sent it there as it was printed, so it
/// comes first. A request that did not resolve is followed by the usage lines of the program `name`.
pub(crate) fn render_compilation(
    compilation: &Compilation,
    styled: bool,
    name: &str,
    stderr: &SharedBuffer,
) -> Rendered {
    let sources = compilation.context().sources();
    for entry in compilation.diagnostics().iter() {
        stderr.push(&diagnostic(entry.diagnostic.display_with_sources(&*sources), styled));
    }
    let unresolved = !compilation.started();
    if let Some(error) = compilation.error() {
        stderr.push(&diagnostic(error.display_with_sources(&*sources), styled));
        if unresolved {
            stderr.push(&format!(
                "usage: {name} [OPTIONS] <INPUT>...\nFor more information, try '--help'.\n"
            ));
        }
    }
    let mut stdout = Vec::new();
    for emitted in compilation.emitted() {
        if let Emitted::Stdout { bytes, .. } = emitted {
            stdout.extend_from_slice(bytes);
        }
    }
    Rendered {
        stdout,
        failed: !compilation.succeeded(),
        unresolved,
    }
}
