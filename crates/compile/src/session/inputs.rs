//! What a session gives the stages for reading inputs and reporting diagnostics.

use std::{borrow::Cow, path::PathBuf};

use runic_core::{
    Report,
    diagnostics::{Diagnostic, PushResult, SourceId},
};

use super::Session;
use crate::{Artifact, CheckpointId, Input, InputSource};

impl Session<'_> {
    /// The artifact kept at the stop point, once the compilation has reached it.
    pub fn stopped(&self) -> Option<&Artifact> {
        self.progress.stopped.as_ref()
    }

    /// The checkpoints reached, in the order they were reached, as of the last time a step
    /// returned. While a step runs, the stages it calls see none here, and nothing in
    /// [Session::stopped], [Session::emitted] or [Session::pass_output]: they learn of progress
    /// from what [Checkpoints::reach](crate::Checkpoints::reach) answers.
    pub fn reached(&self) -> &[CheckpointId] {
        &self.progress.reached
    }

    /// The bytes of `input`: a file is read, and the read is recorded; memory is borrowed.
    pub fn read<'i>(&self, input: &'i Input) -> Result<Cow<'i, [u8]>, Report> {
        match &input.source {
            InputSource::Path(path) => {
                let bytes = std::fs::read(path).map_err(|error| {
                    Report::msg(format!("cannot read input '{}': {error}", path.display()))
                })?;
                self.record_read(path);
                Ok(Cow::Owned(bytes))
            }
            InputSource::Memory { bytes, .. } => Ok(Cow::Borrowed(bytes)),
        }
    }

    /// Reads the text of `input` into the context's source map, so diagnostics can show it.
    ///
    /// A file is read through the context and recorded; memory must be UTF-8 and is registered
    /// under its name.
    pub fn load_source(&self, input: &Input) -> Result<SourceId, Report> {
        match &input.source {
            InputSource::Path(path) => {
                let id = self.context.load_source_file(path).map_err(|error| {
                    Report::msg(format!("cannot read input '{}': {error}", path.display()))
                })?;
                self.record_read(path);
                Ok(id)
            }
            InputSource::Memory { name, bytes } => {
                let text = core::str::from_utf8(bytes).map_err(|error| {
                    Report::msg(format!("input '{name}' is not valid UTF-8: {error}"))
                })?;
                self.context.add_source(name.as_str(), text).map_err(|error| {
                    Report::msg(format!("cannot add input '{name}' to the sources: {error}"))
                })
            }
        }
    }

    /// The files read through the session, in the order each was first read, once each.
    pub fn files_read(&self) -> Vec<PathBuf> {
        self.files_read.borrow().clone()
    }

    /// Records a non-fatal diagnostic in the session's context.
    pub fn emit(&self, diagnostic: impl Diagnostic + Send + Sync + 'static) -> PushResult {
        self.context.emit(diagnostic)
    }

    fn record_read(&self, path: &std::path::Path) {
        let mut files = self.files_read.borrow_mut();
        if !files.iter().any(|read| read == path) {
            files.push(path.to_path_buf());
        }
    }
}

#[cfg(test)]
mod tests {
    use runic_core::diagnostics::SourceProvider;

    use crate::{CompilationRequest, Compiler, Input, Session, testing::TestFrontend};

    fn compiler() -> Compiler {
        Compiler::builder()
            .frontend(TestFrontend::new("alpha", &["a"]))
            .build()
            .unwrap()
    }

    fn session<'c>(compiler: &'c Compiler, inputs: Vec<Input>) -> Session<'c> {
        let mut request = CompilationRequest::new();
        request.name = Some("unit".into());
        request.inputs = inputs;
        compiler.session(request).unwrap()
    }

    #[test]
    fn a_path_is_read_and_recorded_and_memory_is_borrowed_and_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.a");
        std::fs::write(&path, b"from disk").unwrap();
        let from_path = Input::path(&path);
        let from_memory = Input::memory("m.a", "in memory");
        let compiler = compiler();
        let session = session(&compiler, vec![from_path.clone(), from_memory.clone()]);

        assert_eq!(&*session.read(&from_memory).unwrap(), b"in memory");
        assert!(matches!(session.read(&from_memory).unwrap(), std::borrow::Cow::Borrowed(_)));
        assert!(session.files_read().is_empty());
        assert_eq!(&*session.read(&from_path).unwrap(), b"from disk");
        assert_eq!(session.files_read(), [path]);
    }

    #[test]
    fn a_missing_file_is_an_error_naming_the_path_and_the_cause() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.a");
        let input = Input::path(&path);
        let compiler = compiler();
        let session = session(&compiler, vec![Input::memory("m.a", "")]);
        for message in [
            session.read(&input).unwrap_err().to_string(),
            session.load_source(&input).unwrap_err().to_string(),
        ] {
            assert!(message.contains(&*path.to_string_lossy()), "{message}");
            assert!(message.to_lowercase().contains("no such file"), "{message}");
        }
        assert!(session.files_read().is_empty(), "a file that was not read is not recorded");
    }

    #[test]
    fn files_read_lists_each_file_once_in_the_order_first_read() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.a");
        let second = dir.path().join("second.a");
        std::fs::write(&first, "1").unwrap();
        std::fs::write(&second, "2").unwrap();
        let compiler = compiler();
        let session = session(&compiler, vec![Input::memory("m.a", "")]);
        session.read(&Input::path(&second)).unwrap();
        session.load_source(&Input::path(&first)).unwrap();
        session.read(&Input::path(&second)).unwrap();
        session.read(&Input::path(&first)).unwrap();
        assert_eq!(session.files_read(), [second, first]);
    }

    #[test]
    fn load_source_puts_a_path_and_memory_in_the_source_map() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.a");
        std::fs::write(&path, "text on disk").unwrap();
        let compiler = compiler();
        let session = session(&compiler, vec![Input::memory("m.a", "")]);

        let on_disk = session.load_source(&Input::path(&path)).unwrap();
        let in_memory = session.load_source(&Input::memory("mem.a", "text in memory")).unwrap();
        assert_ne!(on_disk, in_memory);
        assert_eq!(session.files_read(), [path]);
        let sources = session.context().sources();
        assert_eq!(sources.get(on_disk).unwrap().text, Some("text on disk"));
        assert_eq!(sources.get(in_memory).unwrap().text, Some("text in memory"));
    }

    #[test]
    fn load_source_of_invalid_utf8_memory_is_an_error_naming_the_input() {
        let compiler = compiler();
        let session = session(&compiler, vec![Input::memory("m.a", "")]);
        let message = session
            .load_source(&Input::memory("bad.a", vec![0xff, 0xfe]))
            .unwrap_err()
            .to_string();
        assert!(message.contains("'bad.a'") && message.contains("UTF-8"), "{message}");
    }

    #[test]
    fn read_returns_bytes_that_are_not_text_for_a_memory_input() {
        let compiler = compiler();
        let session = session(&compiler, vec![Input::memory("m.a", "")]);
        let input = Input::memory("bin.a", vec![0xff, 0x00]);
        assert_eq!(&*session.read(&input).unwrap(), [0xff, 0x00]);
    }
}
