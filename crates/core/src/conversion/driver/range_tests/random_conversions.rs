//! Random conversions checked by evaluation, and patterns and callbacks that break the contract.
//!
//! - `ops`: the operations these tests declare (a carry, a two-result operation, a marker and an
//!   operation it tags, `isize`/`i64` recasts, and what the materialization callbacks build).
//! - `harness`: random programs, pattern sets, targets, converters and modes; each conversion
//!   that succeeds must verify and compute what the program computed.
//! - `adversarial`: patterns and callbacks that break the contract at each point.
//! - `cost`: termination and cost: chains of uses, the rewrite limit.
//! - `casts`: the text of the cast, fuzzed; reconciliation over random DAGs and cyclic graphs.
//! - `targeted`: recursively legal operations, two converters, legality by type, the partial mode's
//!   reading of the unknown operation policy, where a source cast is placed, and the replacements
//!   and remaps that no placement can satisfy.
//!
//! The randomized tests run fixed seeds; `RANDOM_CONVERSIONS_TEST_SEED` (decimal, or hex with
//! `0x`) runs one seed instead, and `RANDOM_CONVERSIONS_TEST_CASES` sets the number of cases per
//! seed (the long run). Each case of the harness has its own seed: a failing case prints the
//! values of the two variables that run it alone.

use super::*;

mod adversarial;
mod casts;
mod cost;
mod harness;
mod ops;
mod targeted;

/// The seeds a randomized test runs: `RANDOM_CONVERSIONS_TEST_SEED` if set, else `default`.
fn seeds(default: &[u64]) -> Vec<u64> {
    let Ok(seed) = std::env::var("RANDOM_CONVERSIONS_TEST_SEED") else {
        return default.to_vec();
    };
    let seed = seed.trim();
    let seed = match seed.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => seed.parse(),
    };
    vec![seed.expect("RANDOM_CONVERSIONS_TEST_SEED is a u64")]
}

/// The number of cases per seed: `RANDOM_CONVERSIONS_TEST_CASES` if set, else `default`.
fn cases(default: usize) -> usize {
    match std::env::var("RANDOM_CONVERSIONS_TEST_CASES") {
        Ok(n) => n.trim().parse().expect("RANDOM_CONVERSIONS_TEST_CASES is a number"),
        Err(_) => default,
    }
}
