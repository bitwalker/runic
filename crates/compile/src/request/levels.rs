use core::{fmt, str::FromStr};

use runic_core::Report;

fn unknown(what: &str, text: &str, known: &[&str]) -> Report {
    Report::msg(format!(
        "unknown {what} '{text}': the {what}s are {}",
        known.iter().map(|k| format!("'{k}'")).collect::<Vec<_>>().join(", ")
    ))
}

/// How hard the compilation optimizes, as `--opt-level` gives it.
///
/// The optimization pipeline and the backend read it to choose what to run.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum OptLevel {
    /// `0`: no optimization.
    None,
    /// `1`: the optimizations that are cheap.
    Basic,
    /// `2`: the usual balance of speed and size. The default.
    #[default]
    Balanced,
    /// `3`: speed above all.
    Max,
    /// `s`: small code.
    Size,
    /// `z`: the smallest code.
    SizeMin,
}

impl OptLevel {
    const KNOWN: [&'static str; 6] = ["0", "1", "2", "3", "s", "z"];
}

impl FromStr for OptLevel {
    type Err = Report;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Ok(match text {
            "0" => Self::None,
            "1" => Self::Basic,
            "2" => Self::Balanced,
            "3" => Self::Max,
            "s" => Self::Size,
            "z" => Self::SizeMin,
            _ => return Err(unknown("optimization level", text, &Self::KNOWN)),
        })
    }
}

impl fmt::Display for OptLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "0",
            Self::Basic => "1",
            Self::Balanced => "2",
            Self::Max => "3",
            Self::Size => "s",
            Self::SizeMin => "z",
        })
    }
}

/// How much debug information the compilation keeps, as `--debug` gives it.
///
/// Frontends decide whether to record locations, and backends whether to emit them.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum DebugInfo {
    /// `none`: none.
    None,
    /// `line`: source lines only. The default.
    #[default]
    Line,
    /// `full`: everything the frontend knows.
    Full,
}

impl DebugInfo {
    const KNOWN: [&'static str; 3] = ["none", "line", "full"];
}

impl FromStr for DebugInfo {
    type Err = Report;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Ok(match text {
            "none" => Self::None,
            "line" => Self::Line,
            "full" => Self::Full,
            _ => return Err(unknown("debug info level", text, &Self::KNOWN)),
        })
    }
}

impl fmt::Display for DebugInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Line => "line",
            Self::Full => "full",
        })
    }
}

/// How often the IR is verified while passes run.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum Verification {
    /// After every pass. The default.
    #[default]
    EachPass,
    /// Once, when the pipeline has finished.
    Once,
    /// Never.
    None,
}

/// What the compilation prints about the IR while the pass pipeline runs, and how it checks it.
///
/// The pass pipeline reads these.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IrOptions {
    /// Passes before which the IR is printed, by name (`-Z print-ir-before`).
    pub print_before: Vec<String>,
    /// Passes after which the IR is printed, by name (`-Z print-ir-after`).
    pub print_after: Vec<String>,
    /// Print operations in their generic form (`-Z print-generic`).
    pub print_generic: bool,
    /// Print source locations (`-Z print-locations`).
    pub print_locations: bool,
    /// How often to verify (`-Z verify-each`, `-Z no-verify`).
    pub verification: Verification,
    /// Print the pass pipeline before running it (`-Z dump-pass-pipeline`).
    pub dump_pass_pipeline: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_optimization_level_reads_back_as_it_prints() {
        for text in ["0", "1", "2", "3", "s", "z"] {
            let level: OptLevel = text.parse().unwrap();
            assert_eq!(level.to_string(), text);
        }
    }

    #[test]
    fn an_unknown_optimization_level_lists_the_known_ones() {
        let message = "4".parse::<OptLevel>().unwrap_err().to_string();
        assert!(message.contains("'4'"), "{message}");
        for known in ["'0'", "'1'", "'2'", "'3'", "'s'", "'z'"] {
            assert!(message.contains(known), "{message}");
        }
    }

    #[test]
    fn every_debug_level_reads_back_as_it_prints() {
        for text in ["none", "line", "full"] {
            let level: DebugInfo = text.parse().unwrap();
            assert_eq!(level.to_string(), text);
        }
    }

    #[test]
    fn an_unknown_debug_level_lists_the_known_ones() {
        let message = "lots".parse::<DebugInfo>().unwrap_err().to_string();
        assert!(message.contains("'lots'"), "{message}");
        for known in ["'none'", "'line'", "'full'"] {
            assert!(message.contains(known), "{message}");
        }
    }

    #[test]
    fn the_defaults_are_balanced_optimization_line_debug_info_and_a_check_after_each_pass() {
        assert_eq!(OptLevel::default(), OptLevel::Balanced);
        assert_eq!(DebugInfo::default(), DebugInfo::Line);
        let ir = IrOptions::default();
        assert_eq!(ir.verification, Verification::EachPass);
        assert!(ir.print_before.is_empty() && ir.print_after.is_empty());
        assert!(!ir.print_generic && !ir.print_locations && !ir.dump_pass_pipeline);
    }
}
