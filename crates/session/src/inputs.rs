#[cfg(feature = "std")]
use alloc::format;
use alloc::{borrow::Cow, string::String, vec, vec::Vec};
use core::fmt;

use crate::{Path, PathBuf};

#[derive(Clone)]
pub struct FileName {
    name: Cow<'static, str>,
    is_path: bool,
}
impl Eq for FileName {}
impl PartialEq for FileName {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl PartialOrd for FileName {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FileName {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.name.cmp(&other.name)
    }
}
impl fmt::Debug for FileName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
impl fmt::Display for FileName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
impl AsRef<Path> for FileName {
    fn as_ref(&self) -> &Path {
        self.name.as_ref().as_ref()
    }
}
impl From<PathBuf> for FileName {
    fn from(path: PathBuf) -> Self {
        Self {
            name: path.to_string_lossy().into_owned().into(),
            is_path: true,
        }
    }
}
impl From<&'static str> for FileName {
    fn from(name: &'static str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            is_path: false,
        }
    }
}
impl From<String> for FileName {
    fn from(name: String) -> Self {
        Self {
            name: Cow::Owned(name),
            is_path: false,
        }
    }
}
impl AsRef<str> for FileName {
    fn as_ref(&self) -> &str {
        self.name.as_ref()
    }
}
impl FileName {
    pub fn is_path(&self) -> bool {
        self.is_path
    }

    pub fn as_path(&self) -> &Path {
        self.as_ref()
    }

    pub fn as_str(&self) -> &str {
        self.name.as_ref()
    }

    pub fn file_name(&self) -> Option<&str> {
        self.as_path().file_name().and_then(|name| name.to_str())
    }

    pub fn file_stem(&self) -> Option<&str> {
        self.as_path().file_stem().and_then(|name| name.to_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputType {
    Real(PathBuf),
    Stdin { name: FileName, input: Vec<u8> },
}

/// This enum represents the types of raw inputs provided to the compiler
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputFile {
    pub file: InputType,
}
impl InputFile {
    pub fn new(file: InputType) -> Self {
        Self { file }
    }

    /// Returns an [InputFile] representing an empty input
    pub fn empty() -> Self {
        Self::new(InputType::Stdin {
            name: "empty".into(),
            input: vec![],
        })
    }

    /// Get an [InputFile] representing the contents of `path`.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        Self::new(InputType::Real(path.as_ref().to_path_buf()))
    }

    /// Get an [InputFile] representing the contents received from standard input.
    #[cfg(feature = "std")]
    pub fn from_stdin(name: FileName) -> std::io::Result<Self> {
        use std::io::Read;

        let mut input = Vec::with_capacity(1024);
        std::io::stdin().read_to_end(&mut input)?;
        Ok(Self::from_bytes(input, name))
    }

    pub fn from_bytes(bytes: Vec<u8>, name: FileName) -> Self {
        Self::new(InputType::Stdin { name, input: bytes })
    }

    pub fn file_name(&self) -> FileName {
        match &self.file {
            InputType::Real(path) => path.clone().into(),
            InputType::Stdin { name, .. } => name.clone(),
        }
    }

    pub fn as_path(&self) -> Option<&Path> {
        match &self.file {
            InputType::Real(path) => Some(path),
            _ => None,
        }
    }

    pub fn is_real(&self) -> bool {
        matches!(self.file, InputType::Real(_))
    }

    pub fn filestem(&self) -> &str {
        match &self.file {
            InputType::Real(path) => path.file_stem().unwrap().to_str().unwrap(),
            InputType::Stdin { .. } => "noname",
        }
    }
}

#[cfg(feature = "std")]
impl clap::builder::ValueParserFactory for InputFile {
    type Parser = InputFileParser;

    fn value_parser() -> Self::Parser {
        InputFileParser
    }
}

#[doc(hidden)]
#[derive(Clone)]
#[cfg(feature = "std")]
pub struct InputFileParser;

#[cfg(feature = "std")]
impl clap::builder::TypedValueParser for InputFileParser {
    type Value = InputFile;

    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        _arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::error::Error> {
        use clap::error::{Error, ErrorKind};

        let input_file = match value.to_str() {
            Some("-") => InputFile::from_stdin("stdin".into())
                .map_err(|err| Error::raw(ErrorKind::Io, err))?,
            Some(_) | None => InputFile::from_path(PathBuf::from(value)),
        };

        match &input_file.file {
            InputType::Real(path) => {
                if path.exists() {
                    if path.is_file() {
                        Ok(input_file)
                    } else {
                        Err(Error::raw(
                            ErrorKind::ValueValidation,
                            format!("invalid input '{}': not a file", path.display()),
                        ))
                    }
                } else {
                    Err(Error::raw(
                        ErrorKind::ValueValidation,
                        format!("invalid input '{}': file does not exist", path.display()),
                    ))
                }
            }
            InputType::Stdin { .. } => Ok(input_file),
        }
    }
}
