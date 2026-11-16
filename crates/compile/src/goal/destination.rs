use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use runic_ir::Report;

use crate::{CheckpointDecl, EmitTo, OutputFile};

/// Where the artifact of one output goes, once the harness has decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Destination {
    /// The file at this path. It is not created until the artifact is written.
    File(PathBuf),
    /// Standard output.
    Stdout,
    /// The caller's memory: no file is written.
    Memory,
}

/// What the harness names outputs from when a request leaves their destination open.
#[derive(Debug, Clone, Copy)]
pub struct OutputNaming<'a> {
    /// The name of the unit being compiled. Made safe as a file name before it is used as one.
    pub unit: &'a str,
    /// The directory outputs named by the harness go in (`--output-dir`). With none, they go in
    /// the directory the compiler runs in.
    pub output_dir: Option<&'a Path>,
    /// The destination of the one output that has none of its own (`--output-file`).
    pub output_file: Option<&'a OutputFile>,
}

/// Escapes `name` for use as one path component, such as a file stem.
///
/// Anything but ASCII letters, digits, `_` and `-` becomes `_`, and an empty name becomes `_`, so
/// the result is never empty and holds neither a path separator nor a dot: `..` becomes `__`.
fn escape_path_component(name: &str) -> Cow<'_, str> {
    if name.is_empty() {
        return Cow::Borrowed("_");
    }
    let is_safe = name.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'));
    if is_safe {
        return Cow::Borrowed(name);
    }
    let escaped: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    Cow::Owned(escaped)
}

/// Gives each output a destination, in the order of `wanted`: the checkpoint's declaration and
/// where the request says its artifact goes.
///
/// Touches no file system. Paths are built as written and compared as paths: `a/b` and `a//b` are
/// one file, `f` and `./f` are not known to be.
pub(super) fn assign(
    wanted: &[(&CheckpointDecl, &EmitTo)],
    naming: &OutputNaming<'_>,
) -> Result<Vec<Destination>, Report> {
    let defaults: Vec<&CheckpointDecl> = wanted
        .iter()
        .filter(|(_, to)| matches!(to, EmitTo::Default))
        .map(|(decl, _)| *decl)
        .collect();
    let given = match (naming.output_file, defaults.len()) {
        (Some(_), 0) => {
            return Err(Report::msg(
                "--output-file was given, but every output has a destination of its own, so there \
                 is nothing for it to name",
            ));
        }
        (Some(_), count) if count > 1 => {
            let names: Vec<String> = defaults.iter().map(|decl| format!("'{}'", decl.id)).collect();
            return Err(Report::msg(format!(
                "--output-file names one destination, but {count} outputs have none of their own: \
                 {}; give each a destination with --emit NAME=PATH, or use --output-dir",
                names.join(", ")
            )));
        }
        (given, _) => given,
    };

    let stem = escape_path_component(naming.unit);
    let mut destinations = Vec::with_capacity(wanted.len());
    for (decl, to) in wanted {
        destinations.push(match to {
            EmitTo::File(path) => Destination::File(path.clone()),
            EmitTo::Stdout => Destination::Stdout,
            EmitTo::Memory => Destination::Memory,
            EmitTo::Default => match given {
                Some(OutputFile::Path(path)) => Destination::File(path.clone()),
                Some(OutputFile::Stdout) => Destination::Stdout,
                None => {
                    let name = format!("{stem}.{}", decl.extension);
                    Destination::File(match naming.output_dir {
                        Some(dir) => dir.join(name),
                        None => PathBuf::from(name),
                    })
                }
            },
        });
    }

    for (index, destination) in destinations.iter().enumerate() {
        if matches!(destination, Destination::Memory) {
            continue;
        }
        if let Some(first) = destinations[..index].iter().position(|other| other == destination) {
            let (a, b) = (wanted[first].0.id, wanted[index].0.id);
            return Err(Report::msg(match destination {
                Destination::File(path) => {
                    format!("the outputs '{a}' and '{b}' both go to the file '{}'", path.display())
                }
                _ => format!("the outputs '{a}' and '{b}' both go to standard output"),
            }));
        }
    }
    Ok(destinations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CheckpointId;

    const OPT: CheckpointDecl = CheckpointDecl {
        id: CheckpointId::IR_OPTIMIZED,
        aliases: &[],
        extension: "runic",
        render: |_, _, _| Ok(()),
    };
    const TRANS: CheckpointDecl = CheckpointDecl {
        id: CheckpointId::IR_TRANSLATED,
        aliases: &[],
        extension: "translated.runic",
        render: |_, _, _| Ok(()),
    };

    fn naming<'a>(
        unit: &'a str,
        output_dir: Option<&'a Path>,
        output_file: Option<&'a OutputFile>,
    ) -> OutputNaming<'a> {
        OutputNaming {
            unit,
            output_dir,
            output_file,
        }
    }

    fn assigned(
        wanted: &[(&CheckpointDecl, &EmitTo)],
        naming: &OutputNaming<'_>,
    ) -> Vec<Destination> {
        assign(wanted, naming).unwrap()
    }

    fn failed(wanted: &[(&CheckpointDecl, &EmitTo)], naming: &OutputNaming<'_>) -> String {
        assign(wanted, naming).unwrap_err().to_string()
    }

    fn file(path: &str) -> Destination {
        Destination::File(PathBuf::from(path))
    }

    #[test]
    fn a_name_of_safe_characters_is_kept_and_anything_else_becomes_an_underscore() {
        assert_eq!(escape_path_component("my-app_2"), "my-app_2");
        assert_eq!(escape_path_component("my app/v1.0"), "my_app_v1_0");
        assert_eq!(escape_path_component("naïve"), "na_ve");
        assert_eq!(escape_path_component(""), "_");
        assert_eq!(escape_path_component("."), "_");
        assert_eq!(escape_path_component(".."), "__");
        assert_eq!(escape_path_component("..."), "___");
    }

    #[test]
    fn explicit_destinations_are_kept_exactly_as_written() {
        let path = EmitTo::File(PathBuf::from("./x/../a.txt"));
        let out = assigned(
            &[(&OPT, &path), (&TRANS, &EmitTo::Stdout)],
            &naming("app", Some(Path::new("out")), None),
        );
        assert_eq!(out, [file("./x/../a.txt"), Destination::Stdout]);
        assert_eq!(
            assigned(&[(&OPT, &EmitTo::Memory)], &naming("app", None, None)),
            [Destination::Memory]
        );
    }

    #[test]
    fn a_default_output_is_named_from_the_unit_and_the_extension_in_the_current_directory() {
        let out = assigned(&[(&OPT, &EmitTo::Default)], &naming("app", None, None));
        assert_eq!(out, [file("app.runic")]);
    }

    #[test]
    fn a_default_output_goes_in_the_output_directory_when_there_is_one() {
        let dir = Path::new("target/out");
        let out = assigned(
            &[(&TRANS, &EmitTo::Default), (&OPT, &EmitTo::Default)],
            &naming("app", Some(dir), None),
        );
        assert_eq!(out, [file("target/out/app.translated.runic"), file("target/out/app.runic")]);
    }

    #[test]
    fn a_unit_name_that_is_not_a_safe_file_name_is_escaped() {
        let out = assigned(&[(&OPT, &EmitTo::Default)], &naming("my app/../x", None, None));
        assert_eq!(out, [file("my_app____x.runic")]);
        let out = assigned(&[(&OPT, &EmitTo::Default)], &naming("", None, None));
        assert_eq!(out, [file("_.runic")]);
    }

    #[test]
    fn the_output_file_takes_the_one_default_output() {
        let path = OutputFile::Path(PathBuf::from("result.bin"));
        let n = naming("app", Some(Path::new("out")), Some(&path));
        assert_eq!(assigned(&[(&OPT, &EmitTo::Default)], &n), [file("result.bin")]);
        let n = naming("app", None, Some(&OutputFile::Stdout));
        assert_eq!(assigned(&[(&OPT, &EmitTo::Default)], &n), [Destination::Stdout]);
    }

    #[test]
    fn the_output_file_leaves_outputs_with_a_destination_of_their_own_alone() {
        let path = OutputFile::Path(PathBuf::from("result.bin"));
        let out = assigned(
            &[(&TRANS, &EmitTo::Stdout), (&OPT, &EmitTo::Default)],
            &naming("app", None, Some(&path)),
        );
        assert_eq!(out, [Destination::Stdout, file("result.bin")]);
    }

    #[test]
    fn the_output_file_with_several_default_outputs_is_an_error_naming_them() {
        let path = OutputFile::Path(PathBuf::from("result.bin"));
        let message = failed(
            &[(&TRANS, &EmitTo::Default), (&OPT, &EmitTo::Default)],
            &naming("app", None, Some(&path)),
        );
        assert_eq!(
            message,
            "--output-file names one destination, but 2 outputs have none of their own: \
             'ir.translated', 'ir.optimized'; give each a destination with --emit NAME=PATH, or \
             use --output-dir"
        );
    }

    #[test]
    fn the_output_file_with_no_default_output_is_an_error() {
        let message =
            failed(&[(&OPT, &EmitTo::Memory)], &naming("app", None, Some(&OutputFile::Stdout)));
        assert_eq!(
            message,
            "--output-file was given, but every output has a destination of its own, so there is \
             nothing for it to name"
        );
    }

    #[test]
    fn two_outputs_to_one_path_are_an_error_naming_both_and_the_path() {
        let same = EmitTo::File(PathBuf::from("a.txt"));
        let message = failed(&[(&TRANS, &same), (&OPT, &same)], &naming("app", None, None));
        assert_eq!(
            message,
            "the outputs 'ir.translated' and 'ir.optimized' both go to the file 'a.txt'"
        );
    }

    #[test]
    fn an_explicit_path_that_equals_a_named_one_is_a_clash() {
        let same = EmitTo::File(PathBuf::from("out/app.runic"));
        let message = failed(
            &[(&TRANS, &same), (&OPT, &EmitTo::Default)],
            &naming("app", Some(Path::new("out")), None),
        );
        assert!(message.contains("'out/app.runic'"), "{message}");
    }

    #[test]
    fn paths_are_compared_as_paths_without_looking_at_the_file_system() {
        let a = EmitTo::File(PathBuf::from("a.txt"));
        let b = EmitTo::File(PathBuf::from("./a.txt"));
        assert_eq!(
            assigned(&[(&TRANS, &a), (&OPT, &b)], &naming("app", None, None)),
            [file("a.txt"), file("./a.txt")]
        );
        let a = EmitTo::File(PathBuf::from("out/a.txt"));
        let b = EmitTo::File(PathBuf::from("out//a.txt"));
        let message = failed(&[(&TRANS, &a), (&OPT, &b)], &naming("app", None, None));
        assert!(message.contains("both go to the file"), "{message}");
    }

    #[test]
    fn two_outputs_to_standard_output_are_an_error_naming_both() {
        let message = failed(
            &[(&TRANS, &EmitTo::Stdout), (&OPT, &EmitTo::Stdout)],
            &naming("app", None, None),
        );
        assert_eq!(
            message,
            "the outputs 'ir.translated' and 'ir.optimized' both go to standard output"
        );
        let message = failed(
            &[(&TRANS, &EmitTo::Stdout), (&OPT, &EmitTo::Default)],
            &naming("app", None, Some(&OutputFile::Stdout)),
        );
        assert!(message.contains("standard output"), "{message}");
    }

    #[test]
    fn any_number_of_outputs_may_go_to_memory() {
        let out = assigned(
            &[(&TRANS, &EmitTo::Memory), (&OPT, &EmitTo::Memory)],
            &naming("app", None, None),
        );
        assert_eq!(out, [Destination::Memory, Destination::Memory]);
    }
}
