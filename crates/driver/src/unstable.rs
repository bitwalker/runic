//! The driver's unstable settings.

use clap::Args;

/// The unstable settings of the driver itself, which a command line gives as `-Z NAME[=VALUE]`.
///
/// The driver declares them as long arguments of a scratch command that is never the main one, and
/// reads the values of the `-Z` options through it. The settings of a stage are declared the same
/// way, with the stage's name as a prefix.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
#[command(about = None, long_about = None)]
pub struct Unstable {
    /// The pass pipeline to run in place of the default optimization, such as
    /// `builtin.module(cse)`.
    #[arg(long, value_name = "PIPELINE")]
    pub pass_pipeline: Option<String>,

    /// Print the pass pipeline the compilation runs, as text, before it runs.
    #[arg(long)]
    pub dump_pass_pipeline: bool,

    /// Print the IR before each of the named passes, a comma-separated list.
    #[arg(long, value_name = "PASSES", value_delimiter = ',')]
    pub print_ir_before: Vec<String>,

    /// Print the IR after each of the named passes, a comma-separated list.
    #[arg(long, value_name = "PASSES", value_delimiter = ',')]
    pub print_ir_after: Vec<String>,

    /// Print IR in its generic form, without the custom syntax of operations.
    #[arg(long)]
    pub print_generic: bool,

    /// Print the source location of each operation.
    #[arg(long)]
    pub print_locations: bool,

    /// Verify nothing: neither the IR the frontends produce nor the IR after each pass.
    #[arg(long)]
    pub no_verify: bool,

    /// Whether the IR is verified after each pass (the default). With `false`, it is verified once,
    /// after the frontends.
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
}

#[cfg(test)]
mod tests {
    use clap::{Command, FromArgMatches};

    use super::*;

    fn parse(line: &[&str]) -> Unstable {
        let command = Unstable::augment_args(Command::new("driver").no_binary_name(true));
        Unstable::from_arg_matches(&command.try_get_matches_from(line).unwrap()).unwrap()
    }

    #[test]
    fn the_unstable_settings_default_to_changing_nothing() {
        let unstable = parse(&[]);
        assert_eq!(unstable.pass_pipeline, None);
        assert!(unstable.print_ir_before.is_empty() && unstable.print_ir_after.is_empty());
        assert!(!unstable.dump_pass_pipeline && !unstable.print_generic);
        assert!(!unstable.print_locations && !unstable.no_verify);
        assert!(unstable.verify_each);
    }

    #[test]
    fn a_list_setting_splits_at_commas_and_adds_up_across_occurrences() {
        let unstable = parse(&["--print-ir-before=cse,dce", "--print-ir-before", "local2reg"]);
        assert_eq!(unstable.print_ir_before, ["cse", "dce", "local2reg"]);
    }

    #[test]
    fn verify_each_without_a_value_is_true_and_with_one_is_that_value() {
        assert!(parse(&["--verify-each"]).verify_each);
        assert!(!parse(&["--verify-each=false"]).verify_each);
    }
}
