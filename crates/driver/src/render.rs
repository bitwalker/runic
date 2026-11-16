//! What a compilation prints: its diagnostics for standard error, its outputs for standard output.

use std::{
    io::{self, Write},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use runic_compile::{Compilation, Emitted};
use runic_ir::diagnostics::{DiagnosticDisplay, RenderConfig};

/// How the driver renders a diagnostic: the lines around a label, not the whole source.
const fn config(styled: bool) -> RenderConfig {
    RenderConfig {
        styled,
        fold: true,
        ..RenderConfig::DEFAULT
    }
}

/// The text of a diagnostic, ending with a newline.
pub(crate) fn render(display: DiagnosticDisplay<'_>, styled: bool) -> String {
    let mut text = display.with_config(config(styled)).to_string();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// Text that two threads share: the worker writes what it renders to standard error here as it
/// renders it, and the thread that started the worker writes the buffer to standard error once the
/// worker has ended, whether it returned or panicked.
#[derive(Clone, Default)]
pub(crate) struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl SharedBuffer {
    /// Appends `text`.
    pub(crate) fn push(&self, text: &str) {
        self.bytes().extend_from_slice(text.as_bytes());
    }

    /// Removes and returns what the buffer holds.
    pub(crate) fn take(&self) -> Vec<u8> {
        core::mem::take(&mut *self.bytes())
    }

    /// The bytes, also after a panic while they were held.
    fn bytes(&self) -> MutexGuard<'_, Vec<u8>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
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
        stderr.push(&render(entry.diagnostic.display_with_sources(&*sources), styled));
    }
    let unresolved = !compilation.started();
    if let Some(error) = compilation.error() {
        stderr.push(&render(error.display_with_sources(&*sources), styled));
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
