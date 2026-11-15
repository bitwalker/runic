//! The verifier's dominance and block-structure diagnostics, from text, checked by
//! `--verify-diagnostics`: wherever the syntax states an operand's type, text can read a value
//! before its definition.

use super::opt;

/// One case per diagnostic text can express, each with the expectation it must meet, and cases
/// that verify.
pub(super) const CASES: &str = r#"// A return reads a value of a block that cannot be reached.
builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        cf.br ^next;
    ^next:
        // expected-error @+1 {{operand #0 of 'builtin.ret' is not dominated by its definition}}
        builtin.ret %v : (i32);
    ^later:
        %v = arith.add %a, %a;
        cf.br ^next;
    };
};
// -----
// A call reads a value defined after it in its block.
builtin.module public @m {
    builtin.function public extern("C") @g(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        // expected-error @+1 {{it is defined here, later in the same block}}
        %r = builtin.call @g(%v) : extern("C") (i32) -> i32;
        %v = arith.add %a, %a;
        builtin.ret %r : (i32);
    };
};
// -----
// A cast reads a value defined after it in its block.
builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i64 {
        // expected-error @+1 {{operand #0 of 'builtin.unrealized_conversion_cast' is not dominated}}
        %w = builtin.unrealized_conversion_cast %v : i32 to i64;
        %v = arith.add %a, %a;
        builtin.ret %w : (i64);
    };
};
// -----
// A branch forwards a value of the other arm of a diamond.
builtin.module public @m {
    builtin.function public extern("C") @f(%c: i1, %a: i32) -> i32 {
        cf.cond_br %c ^left, ^right : (i1);
    ^left:
        %v = arith.add %a, %a;
        cf.br ^join(%v : i32);
    ^right:
        // expected-error @+1 {{in a block that does not dominate the use}}
        cf.br ^join(%v : i32);
    ^join(%j: i32):
        builtin.ret %j : (i32);
    };
};
// -----
// The exit of a loop reads a value its body defines.
builtin.module public @m {
    builtin.function public extern("C") @f(%c: i1, %a: i32) -> i32 {
        cf.br ^header;
    ^header:
        cf.cond_br %c ^body, ^exit : (i1);
    ^body:
        %v = arith.add %a, %a;
        cf.br ^header;
    ^exit:
        // expected-error @+1 {{operand #0 of 'builtin.ret' is not dominated by its definition}}
        builtin.ret %v : (i32);
    };
};
// -----
// The operand is numbered among all of the operation's operands.
builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> (i32, i32) {
        cf.br ^next;
    ^next:
        // expected-error @+1 {{operand #1 of 'builtin.ret' is not dominated by its definition}}
        builtin.ret %a, %v : (i32, i32);
    ^later:
        %v = arith.add %a, %a;
        cf.br ^next;
    };
};
// -----
// An entry block that a block branches back to.
builtin.module public @m {
    // expected-error @+1 {{the entry block of region #0 of 'builtin.function' has predecessors}}
    builtin.function public extern("C") @f() {
    ^entry:
        cf.br ^next;
    ^next:
        cf.br ^entry;
    };
};
// -----
// A branch that is not the last operation of its block.
builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) {
        // expected-error @+1 {{'cf.br' has successors, but is not the last operation of its block}}
        cf.br ^next;
        %v = arith.add %a, %a;
    ^next:
        builtin.ret;
    };
};
// -----
// Code that cannot be reached is not checked: a block that reads what it defines later, and an
// `scf.if` in it that reads the function's arguments and values of that block.
builtin.module public @m {
    builtin.function public extern("C") @f(%c: i1, %a: i32) -> i32 {
        builtin.ret %a : (i32);
    ^dead:
        %w = builtin.unrealized_conversion_cast %v : i32 to i32;
        %v = arith.add %a, %a;
        %r = scf.if %c then {
            %s = arith.add %a, %w;
            scf.yield %s : (i32);
        } else {
            scf.yield %late : (i32);
        } : (i1) -> (i32);
        %late = arith.add %a, %a;
        builtin.ret %r : (i32);
    };
};
// -----
// Values used where their definitions dominate.
builtin.module public @m {
    builtin.function public extern("C") @f(%c: i1, %a: i32) -> i32 {
        %v = arith.add %a, %a;
        cf.cond_br %c ^left, ^right : (i1);
    ^left:
        %l = arith.add %v, %a;
        cf.br ^join(%l : i32);
    ^right:
        %r = scf.if %c then {
            scf.yield %v : (i32);
        } else {
            scf.yield %a : (i32);
        } : (i1) -> (i32);
        cf.br ^join(%r : i32);
    ^join(%j: i32):
        %k = arith.add %j, %v;
        builtin.ret %k : (i32);
    };
};
"#;

#[test]
fn the_dominance_and_structure_diagnostics_from_text() {
    let output = opt(&["--verify-diagnostics", "--split-input-file"], CASES);
    output.succeeded();
    assert_eq!(output.stderr, "");
}

/// Without the expectations, each invalid case fails with its diagnostic, and the valid cases are
/// printed.
#[test]
fn without_expectations_the_invalid_cases_fail() {
    let output = opt(&["--split-input-file"], CASES);
    output.failed();
    assert_eq!(output.stderr.matches("is not dominated by its definition").count(), 6);
    assert_eq!(output.stderr.matches("has predecessors").count(), 1);
    assert_eq!(output.stderr.matches("is not the last operation of its block").count(), 1);
    assert_eq!(output.stdout.matches("builtin.module").count(), 2, "{}", output.stdout);
}
