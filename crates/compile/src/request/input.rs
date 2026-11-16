use core::str::FromStr;
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use runic_core::Report;

/// Splits `NAME[=PATH]` at the first `=`.
///
/// `what` names the flag's value in messages (`extern`, `emit request`). An empty name, and a `=`
/// with nothing after it, are errors: the second never means "no path".
fn split_name_and_value<'a>(
    what: &str,
    text: &'a str,
) -> Result<(&'a str, Option<&'a str>), Report> {
    let (name, value) = match text.split_once('=') {
        Some((name, value)) => (name, Some(value)),
        None => (text, None),
    };
    if name.is_empty() {
        return Err(Report::msg(format!("invalid {what} '{text}': the name is empty")));
    }
    if value == Some("") {
        return Err(Report::msg(format!(
            "invalid {what} '{text}': '=' must be followed by a path (write '{name}' alone for \
             none)"
        )));
    }
    Ok((name, value))
}

/// One input to a compilation: where its bytes come from, and optionally the language they are
/// written in.
///
/// The frontend that translates the input is the one named by `language` when it is set
/// (`--language`), and otherwise the one that claims the extension of the input's name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Input {
    /// Where the input's bytes are.
    pub source: InputSource,
    /// The language of the input, when the caller says so rather than leaving it to be inferred.
    pub language: Option<String>,
}

/// Where the bytes of an [Input] come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    /// A file, to be read by the harness. The path is kept as written: it is not made absolute and
    /// not checked to exist.
    Path(PathBuf),
    /// Bytes the caller already holds, under a name that stands in for a path in messages and in
    /// the derivation of the unit's name.
    Memory {
        /// The name of the input.
        name: String,
        /// The input's contents.
        bytes: Vec<u8>,
    },
}

impl Input {
    /// An input read from the file at `path`.
    pub fn path(path: impl Into<PathBuf>) -> Self {
        Self {
            source: InputSource::Path(path.into()),
            language: None,
        }
    }

    /// An input whose contents are `bytes`, called `name`.
    pub fn memory(name: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            source: InputSource::Memory {
                name: name.into(),
                bytes: bytes.into(),
            },
            language: None,
        }
    }

    /// This input, written in `language`.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// The name messages use for this input: the path as written, or the memory input's name.
    pub fn name(&self) -> Cow<'_, str> {
        match &self.source {
            InputSource::Path(path) => path.to_string_lossy(),
            InputSource::Memory { name, .. } => Cow::Borrowed(name),
        }
    }

    fn as_path(&self) -> &Path {
        match &self.source {
            InputSource::Path(path) => path,
            InputSource::Memory { name, .. } => Path::new(name),
        }
    }

    /// The file name without its last extension, or `None` when there is none or it is not UTF-8.
    ///
    /// A memory input's name is treated as a path.
    pub fn stem(&self) -> Option<&str> {
        self.as_path().file_stem().and_then(|stem| stem.to_str())
    }

    /// The file name's last extension, without the dot, or `None` when there is none or it is not
    /// UTF-8. A memory input's name is treated as a path.
    pub fn extension(&self) -> Option<&str> {
        self.as_path().extension().and_then(|ext| ext.to_str())
    }
}

/// A named dependency of the unit being compiled, as `--extern NAME[=PATH]` gives it.
///
/// The harness does not open it. Frontends read it to declare what a source uses from the
/// dependency, and the backend to find it; without a path the name is looked up in the search
/// paths. What the file at the path holds is for the frontend to say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extern {
    /// The name the source uses.
    pub name: String,
    /// Where the dependency is, as written, when the caller says.
    pub path: Option<PathBuf>,
}

impl FromStr for Extern {
    type Err = Report;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (name, path) = split_name_and_value("extern", text)?;
        Ok(Self {
            name: name.to_string(),
            path: path.map(PathBuf::from),
        })
    }
}

/// A request for the artifact of one checkpoint, as `--emit NAME[=PATH]` gives it.
///
/// The harness resolves `checkpoint` (an id or an alias) on the route of the compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitRequest {
    /// The checkpoint's id or alias, as written.
    pub checkpoint: String,
    /// Where the artifact goes.
    pub to: EmitTo,
}

/// Where the artifact of an [EmitRequest] goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitTo {
    /// Wherever the harness's naming rules put it.
    Default,
    /// The file at this path, as written.
    File(PathBuf),
    /// Standard output.
    Stdout,
    /// The caller's memory: the artifact is handed to the caller and no file is written. Only code
    /// can ask for this; no command-line form reaches it.
    Memory,
}

impl FromStr for EmitRequest {
    type Err = Report;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (name, path) = split_name_and_value("emit request", text)?;
        let to = match path {
            None => EmitTo::Default,
            Some("-") => EmitTo::Stdout,
            Some(path) => EmitTo::File(PathBuf::from(path)),
        };
        Ok(Self {
            checkpoint: name.to_string(),
            to,
        })
    }
}

/// The destination `--output-file` names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFile {
    /// The file at this path, as written.
    Path(PathBuf),
    /// Standard output.
    Stdout,
}

impl FromStr for OutputFile {
    type Err = Report;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "" => Err(Report::msg("invalid output file '': the path is empty")),
            "-" => Ok(Self::Stdout),
            path => Ok(Self::Path(PathBuf::from(path))),
        }
    }
}

/// A rewrite of a path prefix in the paths recorded in debug information, as
/// `--remap-path-prefix FROM[=TO]` gives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemapPathPrefix {
    /// The prefix to replace.
    pub from: PathBuf,
    /// What replaces it; `None` stands for `.`.
    pub to: Option<PathBuf>,
}

impl FromStr for RemapPathPrefix {
    type Err = Report;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (from, to) = split_name_and_value("path prefix remapping", text)?;
        Ok(Self {
            from: PathBuf::from(from),
            to: to.map(PathBuf::from),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error<T: FromStr<Err = Report> + core::fmt::Debug>(text: &str) -> String {
        match text.parse::<T>() {
            Ok(value) => panic!("'{text}' should be rejected, but is {value:?}"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn an_extern_is_a_name_with_or_without_a_path() {
        let bare: Extern = "std".parse().unwrap();
        assert_eq!(
            bare,
            Extern {
                name: "std".into(),
                path: None
            }
        );
        let with_path: Extern = "std=libs/std.rlib".parse().unwrap();
        assert_eq!(with_path.path, Some(PathBuf::from("libs/std.rlib")));
    }

    #[test]
    fn an_extern_splits_at_the_first_equals_sign() {
        let ext: Extern = "a=b=c".parse().unwrap();
        assert_eq!(ext.name, "a");
        assert_eq!(ext.path, Some(PathBuf::from("b=c")));
    }

    #[test]
    fn an_extern_with_an_empty_name_is_rejected_naming_what_was_written() {
        let message = error::<Extern>("=lib");
        assert!(message.contains("'=lib'") && message.contains("name is empty"), "{message}");
        assert!(error::<Extern>("").contains("name is empty"));
    }

    #[test]
    fn an_extern_with_an_equals_sign_and_no_path_is_rejected() {
        let message = error::<Extern>("std=");
        assert!(
            message.contains("'std='") && message.contains("followed by a path"),
            "{message}"
        );
    }

    #[test]
    fn an_emit_request_without_a_path_goes_to_the_default_place() {
        let emit: EmitRequest = "ir.optimized".parse().unwrap();
        assert_eq!(
            emit,
            EmitRequest {
                checkpoint: "ir.optimized".into(),
                to: EmitTo::Default
            }
        );
    }

    #[test]
    fn an_emit_request_to_a_dash_goes_to_standard_output() {
        let emit: EmitRequest = "ir=-".parse().unwrap();
        assert_eq!(emit.to, EmitTo::Stdout);
    }

    #[test]
    fn an_emit_request_to_a_path_goes_to_that_file() {
        let emit: EmitRequest = "ir=out/a.runic".parse().unwrap();
        assert_eq!(emit.checkpoint, "ir");
        assert_eq!(emit.to, EmitTo::File(PathBuf::from("out/a.runic")));
    }

    #[test]
    fn an_emit_request_cannot_ask_for_memory_on_a_command_line() {
        let emit: EmitRequest = "ir=memory".parse().unwrap();
        assert_eq!(emit.to, EmitTo::File(PathBuf::from("memory")));
    }

    #[test]
    fn a_malformed_emit_request_is_rejected() {
        assert!(error::<EmitRequest>("=a").contains("invalid emit request '=a'"));
        assert!(error::<EmitRequest>("ir=").contains("invalid emit request 'ir='"));
    }

    #[test]
    fn an_output_file_is_a_path_or_a_dash() {
        assert_eq!("-".parse::<OutputFile>().unwrap(), OutputFile::Stdout);
        assert_eq!("a.out".parse::<OutputFile>().unwrap(), OutputFile::Path("a.out".into()));
        assert!(error::<OutputFile>("").contains("empty"));
    }

    #[test]
    fn a_path_prefix_remapping_has_a_prefix_and_maybe_a_replacement() {
        let remap: RemapPathPrefix = "/home/me=/src".parse().unwrap();
        assert_eq!(remap.from, PathBuf::from("/home/me"));
        assert_eq!(remap.to, Some(PathBuf::from("/src")));
        let bare: RemapPathPrefix = "/home/me".parse().unwrap();
        assert_eq!(bare.to, None);
    }

    #[test]
    fn a_malformed_path_prefix_remapping_is_rejected() {
        assert!(
            error::<RemapPathPrefix>("=/src").contains("invalid path prefix remapping '=/src'")
        );
        assert!(error::<RemapPathPrefix>("/a=").contains("invalid path prefix remapping '/a='"));
    }

    #[test]
    fn a_path_input_is_named_by_its_path_as_written() {
        let input = Input::path("src/../lib/main.wasm");
        assert_eq!(input.name(), "src/../lib/main.wasm");
        assert_eq!(input.stem(), Some("main"));
        assert_eq!(input.extension(), Some("wasm"));
        assert_eq!(input.language, None);
    }

    #[test]
    fn a_memory_input_is_named_by_its_name_and_that_name_is_read_as_a_path() {
        let input = Input::memory("dir/app.wat", b"(module)".to_vec());
        assert_eq!(input.name(), "dir/app.wat");
        assert_eq!(input.stem(), Some("app"));
        assert_eq!(input.extension(), Some("wat"));
    }

    #[test]
    fn an_input_without_an_extension_or_a_file_name_has_neither() {
        assert_eq!(Input::path("Makefile").extension(), None);
        assert_eq!(Input::path("Makefile").stem(), Some("Makefile"));
        assert_eq!(Input::path("..").stem(), None);
        assert_eq!(Input::memory("", Vec::new()).stem(), None);
    }

    #[cfg(unix)]
    #[test]
    fn an_input_whose_path_is_not_utf8_has_no_stem_and_a_lossy_name() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};
        let input = Input::path(PathBuf::from(OsString::from_vec(vec![b'a', 0xff, b'.', b'x'])));
        assert_eq!(input.stem(), None);
        assert_eq!(input.extension(), Some("x"));
        assert_eq!(input.name(), "a\u{fffd}.x");
    }

    #[test]
    fn an_input_can_say_its_language() {
        assert_eq!(Input::path("a").with_language("wasm").language.as_deref(), Some("wasm"));
    }
}
