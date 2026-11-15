//! The `runic-opt` binary: [`runic_opt::run`] with the process's arguments and streams.
#![deny(warnings)]

use std::{io, process::ExitCode};

fn main() -> ExitCode {
    // `RUNIC_TRACE` selects what is logged.
    match runic_log::runic_logger() {
        Ok((logger, filter)) => {
            if log::set_boxed_logger(logger).is_ok() {
                log::set_max_level(filter);
            }
        }
        Err(message) => {
            eprintln!("error: {message}");
            return ExitCode::FAILURE;
        }
    }
    // A panic is a defect of the tool. `run` reports one on the thread of an input itself, on the
    // standard error it is given; the diagnostics crate's panic hook is not installed, since it
    // would print that panic too and hide the report the panic carries.
    // The streams are not locked for the run: the logger writes to standard error from the thread
    // of an input, and would wait for ever on a lock this thread held while it waits for that one.
    let mut stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut stderr = io::stderr();
    runic_opt::run(std::env::args_os().skip(1), &mut stdin, &mut stdout, &mut stderr)
}
