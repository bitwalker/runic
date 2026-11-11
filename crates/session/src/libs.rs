use alloc::borrow::Cow;
#[cfg(feature = "std")]
use alloc::boxed::Box;

use crate::PathBuf;

/// A library requested by the user to be linked against during compilation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkLibrary {
    /// The name of the library.
    ///
    /// If requested by name, e.g. `-l std`, the name is used as given.
    ///
    /// If requested by path, e.g. `-l ./target/libs/foo.a`, then the name of the library
    /// will be the basename of the file specified in the path.
    pub name: Cow<'static, str>,
    /// If specified, the path from which this library should be loaded
    pub path: Option<PathBuf>,
    // How to link against this library
    //pub linkage: Linkage,
}

#[cfg(feature = "std")]
impl clap::builder::ValueParserFactory for LinkLibrary {
    type Parser = LinkLibraryParser;

    fn value_parser() -> Self::Parser {
        LinkLibraryParser
    }
}

#[cfg(feature = "std")]
#[doc(hidden)]
#[derive(Clone)]
pub struct LinkLibraryParser;

#[cfg(feature = "std")]
impl clap::builder::TypedValueParser for LinkLibraryParser {
    type Value = LinkLibrary;

    fn possible_values(
        &self,
    ) -> Option<Box<dyn Iterator<Item = clap::builder::PossibleValue> + '_>> {
        use clap::builder::PossibleValue;

        Some(Box::new([PossibleValue::new("a").help("A static archive")].into_iter()))
    }

    /// Parses the `-l` flag using the following format:
    ///
    /// `-l[KIND[:<LINKAGE>]=]NAME`
    ///
    /// * `KIND` is one of: `masp`; defaults to `masp`
    /// * `LINKAGE` is one of: `static`, `dynamic`; defaults to `dynamic`
    /// * `NAME` is either an absolute path, or a name (without extension)
    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        _arg: Option<&clap::Arg>,
        _value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::error::Error> {
        todo!()
    }
}
