//! How a frontend or backend declares the options a command line can set.

use core::{any::Any, fmt};

use clap::{ArgMatches, Args, Command};

use crate::CompilationRequest;

/// What one declared type can do, held as function pointers so that [OptionsDecl] has no type
/// parameter.
#[derive(Clone, Copy)]
struct Declared {
    augment: fn(Command) -> Command,
    store: fn(&ArgMatches, &mut CompilationRequest) -> Result<(), clap::Error>,
    store_default: fn(&mut CompilationRequest),
}

impl Declared {
    fn of<T>() -> Self
    where
        T: Args + Default + Any + Send + Sync,
    {
        Self {
            augment: T::augment_args,
            store: |matches, request| {
                request.set_options(T::from_arg_matches(matches)?);
                Ok(())
            },
            store_default: |request| {
                if request.options::<T>().is_none() {
                    request.set_options(T::default());
                }
            },
        }
    }
}

/// What a frontend or backend lets a command line set: a type of stable options and, optionally,
/// a type of unstable ones.
///
/// A [Frontend](crate::Frontend) or [Backend](crate::Backend) returns one from its `options`
/// method. The command-line driver reads it to add the stage's flags to its command and, after
/// parsing, to store the values in the [CompilationRequest]; a [Session](crate::Session) stores
/// the defaults of the stages that take part in a compilation, so that a stage always finds a value
/// of each type it declared.
///
/// Each type derives [clap::Args] and [Default]. The declaration does not check the rules the
/// driver holds the types to (no short flags, no positional arguments, no required arguments).
#[derive(Clone)]
pub struct OptionsDecl {
    stable: Option<Declared>,
    unstable: Option<Declared>,
}

impl OptionsDecl {
    /// Declares `T` as the stage's stable options.
    pub fn of<T>() -> Self
    where
        T: Args + Default + Any + Send + Sync,
    {
        Self {
            stable: Some(Declared::of::<T>()),
            unstable: None,
        }
    }

    /// Declares only unstable options, for a stage that has no stable ones.
    pub fn unstable_only<U>() -> Self
    where
        U: Args + Default + Any + Send + Sync,
    {
        Self {
            stable: None,
            unstable: Some(Declared::of::<U>()),
        }
    }

    /// Adds `U` as the stage's unstable options, replacing any declared before.
    pub fn unstable<U>(self) -> Self
    where
        U: Args + Default + Any + Send + Sync,
    {
        Self {
            unstable: Some(Declared::of::<U>()),
            ..self
        }
    }

    /// Whether a type of stable options is declared.
    pub fn has_stable(&self) -> bool {
        self.stable.is_some()
    }

    /// Whether a type of unstable options is declared.
    pub fn has_unstable(&self) -> bool {
        self.unstable.is_some()
    }

    /// `command` with the arguments of the stable type added, as `clap` declares them; `command`
    /// unchanged when no stable type is declared.
    pub fn augment(&self, command: Command) -> Command {
        match &self.stable {
            Some(declared) => (declared.augment)(command),
            None => command,
        }
    }

    /// `command` with the arguments of the unstable type added, as `clap` declares them; `command`
    /// unchanged when no unstable type is declared.
    pub fn augment_unstable(&self, command: Command) -> Command {
        match &self.unstable {
            Some(declared) => (declared.augment)(command),
            None => command,
        }
    }

    /// Reads the stable type from `matches`, the matches of a command [OptionsDecl::augment]
    /// built, and stores it in `request`, replacing a value of that type. Stores nothing when no
    /// stable type is declared.
    pub fn store(
        &self,
        matches: &ArgMatches,
        request: &mut CompilationRequest,
    ) -> Result<(), clap::Error> {
        match &self.stable {
            Some(declared) => (declared.store)(matches, request),
            None => Ok(()),
        }
    }

    /// Reads the unstable type from `matches`, the matches of a command
    /// [OptionsDecl::augment_unstable] built, and stores it in `request`, replacing a value of that
    /// type. Stores nothing when no unstable type is declared.
    pub fn store_unstable(
        &self,
        matches: &ArgMatches,
        request: &mut CompilationRequest,
    ) -> Result<(), clap::Error> {
        match &self.unstable {
            Some(declared) => (declared.store)(matches, request),
            None => Ok(()),
        }
    }

    /// Stores the default of each declared type that `request` holds no value of. A value already
    /// set, in code or from a command line, is kept.
    pub fn store_defaults(&self, request: &mut CompilationRequest) {
        for declared in self.stable.iter().chain(&self.unstable) {
            (declared.store_default)(request);
        }
    }
}

impl fmt::Debug for OptionsDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptionsDecl")
            .field("stable", &self.has_stable())
            .field("unstable", &self.has_unstable())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use clap::{Args, Command};

    use super::*;

    #[derive(Args, Default, Debug, Clone, PartialEq)]
    struct Stable {
        /// Whether to be loud.
        #[arg(long)]
        loud: bool,
        #[arg(long)]
        level: Option<u8>,
    }

    #[derive(Args, Default, Debug, Clone, PartialEq)]
    struct Unstable {
        #[arg(long)]
        quirk: bool,
    }

    fn names(command: &Command) -> Vec<String> {
        command.get_arguments().map(|arg| arg.get_id().to_string()).collect()
    }

    fn parse(decl: &OptionsDecl, line: &[&str]) -> ArgMatches {
        let command =
            decl.augment_unstable(decl.augment(Command::new("test").no_binary_name(true)));
        command.try_get_matches_from(line).unwrap()
    }

    #[test]
    fn a_declared_types_arguments_appear_in_an_augmented_command() {
        let decl = OptionsDecl::of::<Stable>().unstable::<Unstable>();
        assert!(decl.has_stable() && decl.has_unstable());
        assert_eq!(names(&decl.augment(Command::new("test"))), ["loud", "level"]);
        assert_eq!(names(&decl.augment_unstable(Command::new("test"))), ["quirk"]);
    }

    #[test]
    fn values_parsed_from_a_command_line_are_stored_and_read_back_from_the_request() {
        let decl = OptionsDecl::of::<Stable>().unstable::<Unstable>();
        let matches = parse(&decl, &["--loud", "--level", "3", "--quirk"]);
        let mut request = CompilationRequest::new();
        decl.store(&matches, &mut request).unwrap();
        decl.store_unstable(&matches, &mut request).unwrap();
        assert_eq!(
            request.options::<Stable>(),
            Some(&Stable {
                loud: true,
                level: Some(3)
            })
        );
        assert_eq!(request.options::<Unstable>(), Some(&Unstable { quirk: true }));
    }

    #[test]
    fn store_defaults_stores_a_default_when_nothing_was_set_and_keeps_a_value_set_in_code() {
        let decl = OptionsDecl::of::<Stable>().unstable::<Unstable>();
        let mut request = CompilationRequest::new();
        request.set_options(Stable {
            loud: true,
            level: None,
        });
        decl.store_defaults(&mut request);
        assert_eq!(
            request.options::<Stable>(),
            Some(&Stable {
                loud: true,
                level: None
            })
        );
        assert_eq!(request.options::<Unstable>(), Some(&Unstable::default()));
    }

    #[test]
    fn a_decl_with_no_unstable_type_leaves_the_command_unchanged_and_stores_nothing() {
        let decl = OptionsDecl::of::<Stable>();
        assert!(!decl.has_unstable());
        assert!(names(&decl.augment_unstable(Command::new("test"))).is_empty());
        let matches = parse(&decl, &["--loud"]);
        let mut request = CompilationRequest::new();
        decl.store_unstable(&matches, &mut request).unwrap();
        assert_eq!(request.options::<Unstable>(), None);
        decl.store_defaults(&mut request);
        assert_eq!(request.options::<Unstable>(), None);
    }

    #[test]
    fn a_decl_of_only_unstable_options_adds_no_stable_arguments() {
        let decl = OptionsDecl::unstable_only::<Unstable>();
        assert!(!decl.has_stable() && decl.has_unstable());
        assert!(names(&decl.augment(Command::new("test"))).is_empty());
        let matches = parse(&decl, &["--quirk"]);
        let mut request = CompilationRequest::new();
        decl.store(&matches, &mut request).unwrap();
        assert_eq!(request.options::<Stable>(), None);
        decl.store_unstable(&matches, &mut request).unwrap();
        assert_eq!(request.options::<Unstable>(), Some(&Unstable { quirk: true }));
    }
}
