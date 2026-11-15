//! The tool's behaviour, through [`run`] with in-memory streams.

mod cli;
mod dominance;
mod expectations;
mod pipeline;
mod roundtrip;
mod splitting;
mod test_passes;
mod text;
mod verification;

use std::{ffi::OsString, process::ExitCode};

use super::run;

/// What a run of the tool wrote and returned.
#[derive(Debug)]
struct Output {
    code: ExitCode,
    stdout: String,
    stderr: String,
}

impl Output {
    #[track_caller]
    fn succeeded(&self) -> &Self {
        assert_eq!(self.code, ExitCode::SUCCESS, "the run failed:\n{}", self.stderr);
        self
    }

    #[track_caller]
    fn failed(&self) -> &Self {
        assert_eq!(self.code, ExitCode::FAILURE, "the run succeeded:\n{}", self.stdout);
        self
    }
}

/// Runs the tool with `args` and `stdin`.
fn opt(args: &[&str], stdin: &str) -> Output {
    let mut input = stdin.as_bytes();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let code = run(args.iter().map(OsString::from), &mut input, &mut stdout, &mut stderr);
    Output {
        code,
        stdout: String::from_utf8(stdout).expect("standard output is UTF-8"),
        stderr: String::from_utf8(stderr).expect("standard error is UTF-8"),
    }
}

/// A module whose function computes `a + b` twice and adds the two.
const REPEATED_ADD: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32, %b: i32) -> i32 {
        %x = arith.add %a, %b;
        %y = arith.add %a, %b;
        %z = arith.add %x, %y;
        builtin.ret %z : (i32);
    };
};
"#;

/// A module whose function keeps its argument in a local.
const LOCALS: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 locals(i32) {
        builtin.store_local 0, %a;
        %v = builtin.load_local 0 : i32;
        builtin.ret %v : (i32);
    };
};
"#;

/// A function that says it returns an `i32` and returns nothing: the verifier rejects it.
const INVALID: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f() -> i32 {
        builtin.ret;
    };
};
"#;
