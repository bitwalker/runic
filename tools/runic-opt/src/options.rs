//! The command line.

use std::path::PathBuf;

use clap::Parser;
use runic_core::{
    OpPrintingFlags, Report,
    pass::{IRPrintingConfig, PassInfo, PassSelection},
};

/// Reads textual IR, runs a pass pipeline on it, verifies and prints the result.
///
/// The input holds one top-level operation. It is parsed and verified; the pipeline, if one is
/// given, runs on it with the verifier after each pass; the result is printed to standard output.
/// Diagnostics and IR printed between passes go to standard error. The exit code is 0 when all of
/// this succeeded, 1 otherwise.
#[derive(Debug, Parser)]
#[command(name = "runic-opt", version)]
pub struct Options {
    /// The textual IR to read, or `-` for standard input (the default).
    ///
    /// Regions may be nested at most 1,000 deep; deeper text is an error.
    #[arg(value_name = "FILE")]
    pub input: Option<String>,

    /// Write the result to FILE instead of standard output.
    #[arg(short = 'o', value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// The pass pipeline to run, such as `builtin.module(builtin.function(local2reg),cse)`.
    ///
    /// Text that is one whole pipeline is anchored: its anchor must be `any` or the input's
    /// top-level operation. Any other text, such as `local2reg,cse`, is a list of passes and
    /// nested pipelines run on the input's top-level operation. A pass that runs on one kind of
    /// operation is nested one level, under the operations of its kind directly inside the
    /// operation its pipeline runs on; `recursive(..)`, or the operations in between named,
    /// reach deeper.
    #[arg(short = 'p', long = "pass-pipeline", value_name = "PIPELINE")]
    pub pass_pipeline: Option<String>,

    /// Verify nothing: neither the parsed input, nor the IR after each pass, nor the result.
    #[arg(long)]
    pub no_verify: bool,

    /// Whether the IR is verified after each pass (the default). With `false`, the parsed input
    /// is verified, and the result once, before it is printed.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        require_equals = true,
        default_value_t = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set,
    )]
    pub verify_each: bool,

    /// Split the input at the lines that hold only MARKER (`// -----` when none is given) and
    /// process each part, a case, as an input of its own, in a context of its own.
    ///
    /// The results of the cases that succeeded are printed in order, with a line holding MARKER
    /// between them; a case that fails prints nothing and does not stop the others. Diagnostics
    /// show the line numbers of the whole file. A case that holds only whitespace and `//`
    /// comments is skipped. The value is given with `=`.
    #[arg(
        long,
        value_name = "MARKER",
        num_args = 0..=1,
        require_equals = true,
        value_parser = clap::builder::NonEmptyStringValueParser::new(),
        default_missing_value = crate::split::DEFAULT_MARKER,
    )]
    pub split_input_file: Option<String>,

    /// Check the diagnostics against the expectations written in the input instead of printing
    /// them.
    ///
    /// An expectation is a `//` comment holding `expected-KIND {{TEXT}}`, KIND being `error`,
    /// `warning`, `info` or `hint`: a diagnostic of that kind is expected on the comment's line
    /// whose message, the text of one of its labels, or one of its context frames (such as
    /// "pass 'cse' failed on ..") contains TEXT. `expected-KIND@+N` and `@-N` expect it N lines
    /// below or above, `@above` and `@below` on the nearest line above or below that does not
    /// hold only expectations, and `@unknown` without a location. With `expected-KIND-re`, TEXT
    /// is a regular expression that must match. TEXT runs to the last `}}` before the next
    /// expectation on the line, less the whitespace around it; a line may hold several.
    ///
    /// Each diagnostic matches one expectation at most, and each expectation one diagnostic.
    /// Diagnostics that match none, and expectations that match none, are reported as errors;
    /// the exit code is 1 exactly when there is one, or an expectation cannot be read (its case
    /// is then not processed). A case is printed only when it succeeded. With
    /// `--split-input-file` each case has its own expectations. The warning for a pass that ran
    /// on no operation is a diagnostic like the others (expect it with `@unknown`).
    #[arg(long)]
    pub verify_diagnostics: bool,

    /// Print operations in the generic form. The generic text is not guaranteed to parse back.
    #[arg(long)]
    pub print_generic: bool,

    /// Print the source location of each operation.
    #[arg(long)]
    pub print_locations: bool,

    /// Print the IR before the passes named, or every pass with `all`, to standard error.
    #[arg(long, value_name = "PASS,..|all", value_delimiter = ',')]
    pub print_ir_before: Vec<String>,

    /// Print the IR after the passes named, or every pass with `all`, to standard error.
    #[arg(long, value_name = "PASS,..|all", value_delimiter = ',')]
    pub print_ir_after: Vec<String>,

    /// Print the IR after a pass only when the pass changed it; without `--print-ir-after`, after
    /// every pass that changed it.
    #[arg(long)]
    pub print_ir_after_change: bool,

    /// Print the IR after a pass that failed, as the pass left it, to standard error.
    #[arg(long)]
    pub print_ir_after_failure: bool,

    /// Print the pipeline, as it was loaded, to standard error before running it.
    #[arg(long)]
    pub dump_pass_pipeline: bool,

    /// List the registered passes, with their descriptions, and exit.
    #[arg(long)]
    pub list_passes: bool,

    /// List the registered dialects, with the number of operations of each, and exit.
    #[arg(long)]
    pub list_dialects: bool,
}

/// What is verified (see the command line's `--no-verify` and `--verify-each`).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Verification {
    /// The parser verifies its result, and the pass manager the IR after each pass
    EachPass,
    /// The parser verifies its result, and the result is verified once before it is printed
    Once,
    /// Nothing is verified
    None,
}

impl Options {
    pub fn verification(&self) -> Verification {
        if self.no_verify {
            Verification::None
        } else if self.verify_each {
            Verification::EachPass
        } else {
            Verification::Once
        }
    }

    pub fn printing_flags(&self) -> OpPrintingFlags {
        OpPrintingFlags {
            print_entry_block_headers: false,
            print_source_locations: self.print_locations,
            generic: self.print_generic,
        }
    }

    /// How IR is printed between passes; a pass named that is not registered is an error.
    pub fn ir_printing(&self) -> Result<IRPrintingConfig, Report> {
        let mut after = selection("--print-ir-after", &self.print_ir_after)?;
        if self.print_ir_after_change && after == PassSelection::None {
            after = PassSelection::All;
        }
        let after_failure = if self.print_ir_after_failure {
            PassSelection::All
        } else {
            PassSelection::None
        };
        Ok(IRPrintingConfig {
            before: selection("--print-ir-before", &self.print_ir_before)?,
            after,
            after_failure,
            only_when_modified: self.print_ir_after_change,
            ..Default::default()
        })
    }
}

/// The passes `names` select for `option`: none, `all`, or registered passes.
fn selection(option: &str, names: &[String]) -> Result<PassSelection, Report> {
    if names.is_empty() {
        return Ok(PassSelection::None);
    }
    if names.iter().any(|name| name == "all") {
        return Ok(PassSelection::All);
    }
    for name in names {
        if PassInfo::lookup(name).is_none() {
            let hint = match PassInfo::similar_to(name).as_slice() {
                [] => String::new(),
                [one] => format!("; did you mean '{one}'?"),
                many => format!(
                    "; did you mean one of {}?",
                    many.iter().map(|name| format!("'{name}'")).collect::<Vec<_>>().join(", ")
                ),
            };
            return Err(Report::msg(format!("{option}: unknown pass '{name}'{hint}")));
        }
    }
    Ok(PassSelection::Named(names.to_vec()))
}
