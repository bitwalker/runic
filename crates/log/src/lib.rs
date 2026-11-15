//! A configurable logger that can be configured both directly and via environment variables.
//!
//! Designed for use with the [`log` crate][log-crate-url].
//!
//! This crate is inspired by [`env_logger`](https://crates.io/crates/env_logger), but provides
//! additional features for compilers built with Runic that make it more useful when diagnosing
//! issues in a pipeline that has an enormous quantity of verbose trace logs that are emitted for
//! a large number of different entities in the compilation graph. Typically, we're interested in
//! output related to a specific function, possibly refined to specific components of the compiler,
//! or even specific areas of those components.
//!
//! The logger a Runic tool installs ([`runic_logger`]; `runic-opt` installs it) reads four
//! environment variables:
//!
//! - `RUNIC_TRACE`: which records are written, as directives (see [Enabling logging]);
//! - `RUNIC_TRACE_FILTER`: a filter on the key-value pairs of records (see [Filtering by key and
//!   value]);
//! - `RUNIC_TRACE_STYLE`: whether styles are written (see [Colors]);
//! - `RUNIC_TRACE_TIMING`: the precision of a timestamp on each record, one of `s`, `ms`, `us` or
//!   `ns`; without it, records carry no timestamp.
//!
//! There are a few primary use cases we're interested in supporting with this crate:
//!
//! 1. Emit all logs at a given log level, e.g. `RUNIC_TRACE=trace runic-opt input.runic -p sccp`
//! 2. Same as 1, but for a specific component of the compiler, e.g. `RUNIC_TRACE=dataflow=trace`
//! 3. Same as 2, but for a specific topic within that component, e.g.
//!    `RUNIC_TRACE=dataflow:solver=trace`
//! 4. Same as 2, but for any topic in that component which matches a regular expression, e.g.
//!    `RUNIC_TRACE=dataflow:sol.*=trace`
//! 5. Any of the above, but only for records about an operation which matches a given filter:
//!    a. Match the operations of one dialect, e.g. `RUNIC_TRACE=pattern-rewrite-driver=trace RUNIC_TRACE_FILTER=dialect:arith`
//!    b. Match an operation by its name within its dialect, e.g. `RUNIC_TRACE=pattern-rewrite-driver=trace RUNIC_TRACE_FILTER=op:add`
//! 6. Negated directives: emit all logs which match any positive directive and are not also
//!    matched by the negative one, e.g.:
//!    a. Emit everything except the logs of the "dataflow" component: `RUNIC_TRACE=trace,-dataflow=trace`
//!    b. Emit the logs of the "dataflow" component, except its debug-level and more severe logs: `RUNIC_TRACE=dataflow=trace,-dataflow=debug`
//!    c. Emit the logs of the "dataflow" component, except those of its "solver" topic: `RUNIC_TRACE=dataflow=trace,-dataflow:solver=trace`
//!
//! By default, logs are written to `stderr`, but this is configurable.
//!
//! NOTE: The functionality here relies on `log` targets using a structured format that we can
//! parse and extract the relevant pieces from. Any `log` targets which are not in this format
//! will be treated as a single component and emitted according to the default logging rules,
//! while still supporting filtering those components. The format is as follows:
//!
//! ```ignore
//! scope := component (":" topic)*
//!
//! component := [A-Za-z0-9_-]+
//! topic := [^:,=]+
//! ```
//!
//! A record without an explicit target has the path of the module it was logged from as its
//! target (for example `runic_analysis::analyses::callable_uses`).
//!
//! ## Example
//!
//! ```
//! use log::{debug, error, log_enabled, info, Level};
//!
//! runic_log::init();
//!
//! debug!("this is a debug {}", "message");
//! error!("this is printed by default");
//!
//! if log_enabled!(Level::Info) {
//!     let x = 3 * 4; // expensive computation
//!     info!("the answer was: {}", x);
//! }
//! ```
//!
//! [`init`] reads `RUNIC_TRACE`: without it, the program writes only the error record; with
//! `RUNIC_TRACE=info`, the error and the info records; with `RUNIC_TRACE=debug`, all three.
//!
//! See the documentation for the [`log` crate][log-crate-url] for more
//! information about its API.
//!
//! ## Enabling logging
//!
//! **By default all logging is disabled except for the `error` level**
//!
//! The **`RUNIC_TRACE`** environment variable controls logging. Its value is a comma-separated
//! list of directives, optionally followed by `/` and a regular expression (see [Filtering
//! results]). A directive is one of:
//!
//! - `level`: every record at `level` or more severe;
//! - `component[=level]`: the records whose target's component (the part before the first `:`)
//!   is `component`; `component:*` is the same;
//! - `component:topic[=level]`: the records of `component` whose topic (the rest of the target)
//!   matches `topic`, a regular expression;
//! - `path::to::module[=level]`: the records whose target contains `path::to::module`.
//!
//! Without `=level`, a directive selects records of every level. A directive written with a
//! leading `-` and a level (`-component=level`) is negated: a record it selects is not written,
//! whatever other directive selects it.
//!
//! **level** is the maximum [`log::Level`][level-enum] to be shown and includes:
//! - `error`
//! - `warn`
//! - `info`
//! - `debug`
//! - `trace`
//! - `off` (pseudo level to disable all logging for the target)
//!
//! Logging level names are case-insensitive; e.g.,
//! `debug`, `DEBUG`, and `dEbuG` all represent the same logging level. For
//! consistency, our convention is to use the lower case names.
//!
//! Some examples of valid values of `RUNIC_TRACE` are:
//!
//! - `RUNIC_TRACE=trace` turns on all logging
//! - `RUNIC_TRACE=TRACE` turns on all logging (same as previous)
//! - `RUNIC_TRACE=info` turns on all info logging
//! - `RUNIC_TRACE=dataflow=debug` turns on debug logging for the `dataflow` component
//! - `RUNIC_TRACE=runic_analysis::analyses=trace` turns on all logging for the records whose
//!   target is a module path that contains `runic_analysis::analyses`
//! - `RUNIC_TRACE=trace,dataflow=off` turns on all logging, but turns off logging for `dataflow`
//! - `RUNIC_TRACE=off` turns off all logging
//!
//! ## Filtering results
//!
//! A `RUNIC_TRACE` value may end in a regex filter. The syntax is to append `/`
//! followed by a regex. Each message is checked against the regex, and is only
//! logged if it matches. Note that the matching is done after formatting the
//! log string but before adding any logging meta-data. There is a single filter
//! for all modules.
//!
//! Some examples:
//!
//! * `dataflow/fixpoint` turns on all logging for the `dataflow` component where the log
//!   message includes 'fixpoint'.
//! * `info/f.o` turns on all info logging where the log message includes 'foo',
//!   'f1o', 'fao', etc.
//! * `dataflow=debug/foo*foo` turns on debug logging for `dataflow` where the log
//!   message includes 'foofoo' or 'fofoo' or 'fooooooofoo', etc.
//! * `error,dataflow=warn/[0-9]scopes` turn on global error logging and also
//!   warn for `dataflow`. In both cases the log message must include a single digit
//!   number followed by 'scopes'.
//!
//! ## Filtering by key and value
//!
//! The **`RUNIC_TRACE_FILTER`** environment variable filters records by their key-value pairs.
//! Its value is a comma-separated list of `key:value` filters, where `value` is a regular
//! expression; a filter written with a leading `-` (`-key:value`) is negated, and an empty entry
//! (an empty value, a trailing comma) is skipped. A record that carries none of the keys is not
//! affected. A record that carries a key is written only when its value matches one of that key's
//! filters and none of its negated ones, so a key with only negated filters keeps none of its
//! records: `op:.*,-op:constant` keeps the records about every operation but the `constant`
//! operations.
//!
//! The pattern rewrite driver (`RUNIC_TRACE=pattern-rewrite-driver=trace`) logs records with the
//! keys `dialect` (the dialect of the operation, for example `arith`) and `op` (its name within
//! the dialect, for example `add`). A value that is not of the form `key:value`, or whose pattern
//! is not a valid regular expression, is an error: [`runic_logger`] returns it, and
//! [`Builder::parse_env`] panics with it.
//!
//! ## Capturing logs in tests
//!
//! Records logged during `cargo test` will not be captured by the test harness by default.
//! The [`Builder::is_test`] method can be used in unit tests to ensure logs will be captured:
//!
//! ```test_harness
//! #[cfg(test)]
//! mod tests {
//!     use log::info;
//!
//!     fn init() {
//!         let _ = runic_log::builder().is_test(true).try_init();
//!     }
//!
//!     #[test]
//!     fn it_works() {
//!         init();
//!
//!         info!("This record will be captured by `cargo test`");
//!
//!         assert_eq!(2, 1 + 1);
//!     }
//! }
//! ```
//!
//! Enabling test capturing comes at the expense of color and other style support
//! and may have performance implications.
//!
//! ## Colors
//!
//! Outputting of colors and other styles can be controlled by the `RUNIC_TRACE_STYLE`
//! environment variable. It accepts the following [values][fmt::WriteStyle]:
//!
//! * `auto` (default) will attempt to print style characters, but don't force the issue.
//!   If the console isn't available on Windows, or if TERM=dumb, for example, then don't print colors.
//! * `always` will always print style characters even if they aren't supported by the terminal.
//!   This includes emitting ANSI colors on Windows if the console API is unavailable.
//! * `never` will never print style characters.
//!
//! Color may be applied in the logged message or a [custom formatter][fmt].
//!
//! <div class="warning">
//!
//! Logging of untrusted inputs can cause unexpected behavior as they may include ANSI escape codes which
//! will be forwarded to the users terminal as part of "Weaponizing ANSI Escape Sequences".
//!
//! Mitigations include:
//! - Setting `RUNIC_TRACE_STYLE=never` to have all ANSI escape codes stripped
//! - In the application, calling [`Builder::write_style(Never)`][Builder::write_style] to have all ANSI escape codes stripped
//! - In the application, [stripping ANSI escape codes](https://docs.rs/anstream/latest/anstream/adapter/fn.strip_str.html)
//!   from user inputs
//!
//! Note: deactivating the build-time feature `color` is not a mitigation as that removes all ANSI escape code
//! stripping from `runic_log`.
//!
//! </div>
//!
//! ## Tweaking the default format
//!
//! Parts of the default format can be excluded from the log output using the [`Builder`].
//! The following example excludes the timestamp from the log output:
//!
//! ```
//! runic_log::builder()
//!     .format_timestamp(None)
//!     .init();
//! ```
//!
//! ### Stability of the default format
//!
//! The default format won't optimise for long-term stability, and explicitly makes no
//! guarantees about the stability of its output across major, minor or patch version
//! bumps during `0.x`.
//!
//! If you want to capture or interpret the output of `runic_log` programmatically
//! then you should use a custom format.
//!
//! ### Using a custom format
//!
//! Custom formats can be provided as closures to the [`Builder`].
//! These closures take a [`Formatter`][crate::fmt::Formatter] and `log::Record` as arguments:
//!
//! ```
//! use std::io::Write;
//!
//! runic_log::builder()
//!     .format(|buf, record| {
//!         writeln!(buf, "{}: {}", record.level(), record.args())
//!     })
//!     .init();
//! ```
//!
//! See the [`fmt`] module for more details about custom formats.
//!
//! ## Specifying defaults for environment variables
//!
//! `runic_log` can read configuration from environment variables.
//! If these variables aren't present, the default value to use can be tweaked with the [`Env`] type.
//! The following example defaults to log `warn` and above if the `RUNIC_TRACE` environment variable
//! isn't set:
//!
//! ```
//! use runic_log::Env;
//!
//! runic_log::Builder::from_env(Env::default().default_filter_or("warn")).init();
//! ```
//!
//! [level-enum]: https://docs.rs/log/latest/log/enum.Level.html
//! [log-crate-url]: https://docs.rs/log
//! [Enabling logging]: #enabling-logging
//! [Filtering results]: #filtering-results
//! [Filtering by key and value]: #filtering-by-key-and-value
//! [Colors]: #colors

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::print_stderr)]
#![warn(clippy::print_stdout)]
#![allow(clippy::test_attr_in_doctest)]

pub mod filter;
mod logger;
mod writer;

pub mod fmt;

pub use self::{
    fmt::{Target, TimestampPrecision, WriteStyle},
    logger::*,
};

/// The logger a Runic tool installs (`runic-opt` does), and the level it is to be installed at.
///
/// Built, not installed: a caller that embeds the compiler decides when — and whether — the
/// global logger is set. `RUNIC_TRACE` selects what is logged (see the module documentation for
/// its syntax), `RUNIC_TRACE_FILTER` filters records by their key-value pairs, `RUNIC_TRACE_STYLE`
/// selects whether styles are written (`auto`, `always` or `never`), and `RUNIC_TRACE_TIMING`
/// selects the timestamp precision, one of `s`, `ms`, `us` or `ns`; without it, records carry no
/// timestamp. Anything else in `RUNIC_TRACE_TIMING`, and a `RUNIC_TRACE_FILTER` that is not a list
/// of `key:value` filters with valid patterns, is an error, described by the returned message.
///
/// One function rather than one per binary, so that every tool is configured by the same variables
/// and formats its output the same way.
pub fn runic_logger() -> Result<(Box<dyn log::Log>, log::LevelFilter), String> {
    let mut builder = Builder::try_from_default_env()?;
    builder.format_indent(Some(2));
    match std::env::var("RUNIC_TRACE_TIMING") {
        Ok(precision) => match precision.as_str() {
            "s" => builder.format_timestamp_secs(),
            "ms" => builder.format_timestamp_millis(),
            "us" => builder.format_timestamp_micros(),
            "ns" => builder.format_timestamp_nanos(),
            other => {
                return Err(format!(
                    "invalid RUNIC_TRACE_TIMING precision, expected one of [s, ms, us, ns], got \
                     '{other}'"
                ));
            }
        },
        Err(_) => builder.format_timestamp(None),
    };
    let logger = builder.build();
    let filter = logger.filter();
    Ok((Box::new(logger), filter))
}
