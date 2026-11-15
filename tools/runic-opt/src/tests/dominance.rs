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
// An operation after a return.
builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        // expected-error @+1 {{'builtin.ret' is a terminator, but is not the last operation of its block}}
        builtin.ret %a : (i32);
        %v = arith.add %a, %a;
    };
};
// -----
// A function whose body falls off its end.
builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        // expected-error @+1 {{block #0 of region #0 of 'builtin.function' does not end with a terminator}}
        %v = arith.add %a, %a;
    };
};
// -----
// An empty block that is branched to.
builtin.module public @m {
    // expected-error @+1 {{block #1 of region #0 of 'builtin.function' is empty}}
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        cf.br ^empty;
    ^empty:
    };
};
// -----
// A block that falls through to the next.
builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        cf.br ^next;
    ^next:
        // expected-error @+1 {{block #1 of region #0 of 'builtin.function' does not end with a terminator}}
        %v = arith.add %a, %a;
    ^last:
        builtin.ret %a : (i32);
    };
};
// -----
// An arm of `scf.if` that yields nothing.
builtin.module public @m {
    builtin.function public extern("C") @f(%c: i1, %a: i32) -> i32 {
        %r = scf.if %c then {
            // expected-error @+1 {{block #0 of region #0 of 'scf.if' does not end with a terminator}}
            %v = arith.add %a, %a;
        } else {
            scf.yield %a : (i32);
        } : (i1) -> (i32);
        builtin.ret %r : (i32);
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
    assert_eq!(output.stderr.matches("is not the last operation of its block").count(), 2);
    // (the quoted source lines hold the expectations, which end with `}}`)
    assert_eq!(output.stderr.matches("does not end with a terminator\n").count(), 4);
    assert_eq!(output.stdout.matches("builtin.module").count(), 2, "{}", output.stdout);
}

/// Each block that does not end with a terminator, and an operation after one, is a diagnostic,
/// not an internal error, and nothing is printed.
#[test]
fn a_block_without_a_terminator_is_rejected() {
    for source in
        [
            // an operation after `builtin.ret`
            "builtin.function public extern(\"C\") @f(%c: i32) -> i32 {\n    builtin.ret %c : \
             (i32);\n    %x = arith.constant 0 : i32;\n};\n",
            // a function that falls off its end
            "builtin.function public extern(\"C\") @g(%c: i32) -> i32 {\n    %x = arith.constant \
             0 : i32;\n};\n",
            // an empty block
            "builtin.function public extern(\"C\") @h(%c: i32) -> i32 {\n    cf.br ^a;\n^a:\n};\n",
            // a block that falls through to the next
            "builtin.function public extern(\"C\") @i(%c: i32) -> i32 {\n    cf.br ^a;\n^a:\n    \
             %x = arith.constant 0 : i32;\n^b:\n    builtin.ret %c : (i32);\n};\n",
            // an arm of `scf.if` that yields nothing
            "builtin.function public extern(\"C\") @j(%c: i1) -> i32 {\n    %r = scf.if %c then \
             {\n        %z = arith.constant 0 : i32;\n    } else {\n        %y = arith.constant 0 \
             : i32;\n        scf.yield %y : (i32);\n    } : (i1) -> (i32);\n    builtin.ret %r : \
             (i32);\n};\n",
        ]
    {
        let output = opt(&[], source);
        output.failed();
        assert_eq!(output.stdout, "");
        assert!(!output.stderr.contains("internal error"), "{}", output.stderr);
        assert!(
            output.stderr.contains("does not end with a terminator")
                || output.stderr.contains("is a terminator, but is not the last operation"),
            "{}",
            output.stderr
        );
    }
}
