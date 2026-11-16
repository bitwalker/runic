//! Stages to run the harness against: a small language with a frontend, and two small machines
//! with backends.
//!
//! The language is `calc` ([CalcFrontend]): definitions of functions over 32-bit integers, such as
//! `fn add(a, b) = a + b;`. The machines are `stackvm` ([StackVmBackend]), which lists each
//! function as instructions of a stack machine, and `altvm` ([AltVmBackend]), which lists the
//! functions and their arities and shares one of the stack machine's targets. They use only what
//! the crate and `runic-ir` export, as the stages of any other compiler would.

pub mod calc;
pub mod stackvm;

pub use self::{
    calc::{CalcAst, CalcFrontend, CalcOptions},
    stackvm::{AltVmBackend, StackVmBackend, StackVmOptions, StackVmUnstable},
};

#[cfg(test)]
mod stages;
#[cfg(test)]
pub(crate) use self::stages::*;
