//! The compiler harness of Runic.
//!
//! A compiler built on Runic supplies frontends, which translate sources to IR, and backends,
//! which lower IR and produce code for a target. The harness is what stands between them and
//! around them: it takes a description of one compilation, decides which stages run and where
//! their outputs go, and runs them.
//!
//! This crate holds the vocabulary of a compilation:
//!
//! * a [Triple] names the target.
#![deny(warnings)]

mod triple;

pub use self::triple::Triple;

/// The version of Runic this crate was built from.
pub const RUNIC_VERSION: &str = env!("RUNIC_BUILD_VERSION");

/// The revision of Runic's repository this crate was built from, as `git describe` gives it, or
/// `unknown` when it was built outside a checkout.
pub const RUNIC_REVISION: &str = env!("RUNIC_BUILD_REV");
