//! `toyc`: a whole downstream compiler.
//!
//! It compiles `calc`, a toy language, and the textual IR, for two toy stack machines. A compiler
//! built on Runic supplies its frontends, its backends and a `main`; the command line, the
//! flags of each stage and the exit codes come from the driver.
//!
//! `crates/driver/tests/toyc.rs` builds the same compiler and tests it: the two must match.

use std::process::ExitCode;

use runic_compile::{
    Compiler, IrFrontend,
    testing::{AltVmBackend, CalcFrontend, StackVmBackend},
};
use runic_driver::{Driver, EXIT_INTERNAL};

fn main() -> ExitCode {
    let compiler = Compiler::builder()
        .frontend(CalcFrontend)
        .frontend(IrFrontend)
        .backend(StackVmBackend)
        .backend(AltVmBackend)
        .default_target("stackvm-unknown-tiny")
        .build();
    let compiler = match compiler {
        Ok(compiler) => compiler,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(EXIT_INTERNAL);
        }
    };
    let driver = Driver::new(&compiler)
        .name("toyc")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Compiles calc, a toy language, for two toy stack machines")
        .build();
    match driver {
        Ok(driver) => runic_driver::main_with(&driver),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(EXIT_INTERNAL)
        }
    }
}
