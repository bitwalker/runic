//! The sparse forward framework where the lattice of a value it reads along an edge is one it
//! writes, or one it reads twice: a call that passes one value twice, a recursive call that passes
//! its callee's own entry arguments (in the same order, swapped, or one of them twice), a return of
//! a recursive call's result or of one of the call's own operands, and the same shapes in a CFG
//! (a back edge that passes the header its own argument, or swaps its arguments, and a branch that
//! passes one value twice).
//!
//! Each test runs constant propagation interprocedurally and compares what it knows of every value
//! that matters with the fixed point computed by hand, and has a control: the same program with
//! the edge in question changed. Where a lattice read along an edge is one the edge writes, the
//! program is written so that the lattice read changes after the edge's first visit: the answer
//! must still be the fixed point.

use alloc::{collections::BTreeMap, format, string::String};

use runic_ir::pass::AnalysisManager;

use super::sparse_backward::Program;
use crate::{
    DataFlowConfig, DataFlowSolver, Lattice,
    analyses::{DeadCodeAnalysis, SparseConstantPropagation, constant_propagation::ConstantValue},
};

/// What interprocedural constant propagation knows of each value of `source` that `names` lists:
/// `"?"` when it never reached the value, `"T"` when the value is not one constant (overdefined),
/// and otherwise the constant.
fn constants(source: &str, names: &[&str]) -> BTreeMap<String, String> {
    let program = Program::parse(source);
    let mut config = DataFlowConfig::default();
    config.set_interprocedural(true);
    let mut solver = DataFlowSolver::new(config);
    solver.load::<DeadCodeAnalysis>();
    solver.load::<SparseConstantPropagation>();
    solver
        .initialize_and_run(&program.root.borrow(), AnalysisManager::new(program.root, None))
        .expect("the analyses converge");
    let mut known = BTreeMap::new();
    for value in program.values() {
        let name = program.name_of(value);
        if !names.contains(&name.as_str()) {
            continue;
        }
        let state = match solver.get::<Lattice<ConstantValue>, _>(&value) {
            None => String::from("?"),
            Some(lattice) if lattice.value().is_uninitialized() => String::from("?"),
            Some(lattice) => match lattice.value().constant_value() {
                None => String::from("T"),
                Some(attr) => {
                    let attr = attr.borrow();
                    let int = attr
                        .as_attr()
                        .as_trait::<dyn runic_ir::attributes::IntegerLikeAttr>()
                        .expect("an integer constant")
                        .as_int();
                    format!("{}", int.try_unsigned::<u64>().expect("fits"))
                }
            },
        };
        known.insert(name, state);
    }
    assert_eq!(known.len(), names.len(), "a name is not a value of the program: {names:?}");
    known
}

fn expect(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(name, state)| (String::from(*name), String::from(*state)))
        .collect()
}

/// Assert that `source` gives the states `pairs` lists.
fn check(source: &str, pairs: &[(&str, &str)]) {
    let names = pairs.iter().map(|(name, _)| *name).collect::<alloc::vec::Vec<_>>();
    assert_eq!(constants(source, &names), expect(pairs), "\n{source}");
}

/// A call that passes one value twice gives the callee that constant in both arguments. The
/// control calls it from two sites with two constants: the arguments are not one constant.
#[test]
fn a_call_may_pass_one_value_twice() {
    let source = r#"builtin.module public @m {
    builtin.function private extern("C") @g(%gx: i32, %gy: i32) -> i32 {
        %gs = arith.add %gx, %gy;
        builtin.ret %gs : (i32);
    };
    builtin.function public extern("C") @f() -> i32 {
        %two = arith.constant 2 : i32;
        %three = arith.constant 3 : i32;
        %r = builtin.call @g(%two, %two) : extern("C") (i32, i32) -> i32;
        SECOND
        builtin.ret %r : (i32);
    };
};"#;
    let one_site = source.replace("SECOND", "");
    check(&one_site, &[("%gx", "2"), ("%gy", "2"), ("%gs", "4"), ("%r", "4")]);
    let two_sites = source.replace(
        "SECOND",
        "%q = builtin.call @g(%three, %three) : extern(\"C\") (i32, i32) -> i32;",
    );
    check(
        &two_sites,
        &[("%gx", "T"), ("%gy", "T"), ("%gs", "T"), ("%r", "T"), ("%q", "T")],
    );
}

/// A recursive function, `@h(%hx, %hy)`, whose recursive call passes its own entry arguments.
const RECURSIVE: &str = r#"builtin.module public @m {
    builtin.function private extern("C") @h(%hx: i32, %hy: i32) -> i32 {
        RECURSION
        %hs = arith.add %hx, %hy;
        builtin.ret %hs : (i32);
    };
    builtin.function public extern("C") @f() -> i32 {
        %one = arith.constant 1 : i32;
        %two = arith.constant 2 : i32;
        %three = arith.constant 3 : i32;
        %r = builtin.call @h(OUTER) : extern("C") (i32, i32) -> i32;
        builtin.ret %r : (i32);
    };
};"#;

fn recursive(recursion: &str, outer: &str) -> String {
    let recursion = if recursion.is_empty() {
        String::new()
    } else {
        format!("%hr = builtin.call @h({recursion}) : extern(\"C\") (i32, i32) -> i32;")
    };
    RECURSIVE.replace("RECURSION", &recursion).replace("OUTER", outer)
}

/// The arguments passed back unchanged add nothing: the entry arguments are what the outer call
/// passes, as in the control, which has no recursive call.
#[test]
fn a_recursive_call_may_pass_its_entry_arguments_unchanged() {
    check(
        &recursive("%hx, %hy", "%one, %two"),
        &[("%hx", "1"), ("%hy", "2"), ("%hs", "3"), ("%hr", "3"), ("%r", "3")],
    );
    check(
        &recursive("", "%one, %two"),
        &[("%hx", "1"), ("%hy", "2"), ("%hs", "3"), ("%r", "3")],
    );
}

/// The arguments passed back swapped: each argument is joined with the other, so with 1 and 2
/// from outside neither is a constant, and with 3 and 3 both are 3.
#[test]
fn a_recursive_call_may_swap_its_entry_arguments() {
    check(
        &recursive("%hy, %hx", "%one, %two"),
        &[("%hx", "T"), ("%hy", "T"), ("%hs", "T"), ("%hr", "T"), ("%r", "T")],
    );
    check(
        &recursive("%hy, %hx", "%three, %three"),
        &[("%hx", "3"), ("%hy", "3"), ("%hs", "6"), ("%hr", "6"), ("%r", "6")],
    );
}

/// One entry argument passed back to both: `%hx` is joined with `%hy`, so with 1 and 2 from
/// outside `%hx` is not a constant and `%hy` is still 2; the control passes `%hx` to both.
#[test]
fn a_recursive_call_may_pass_one_entry_argument_twice() {
    check(
        &recursive("%hy, %hy", "%one, %two"),
        &[("%hx", "T"), ("%hy", "2"), ("%hs", "T"), ("%hr", "T"), ("%r", "T")],
    );
    check(
        &recursive("%hx, %hx", "%one, %two"),
        &[("%hx", "1"), ("%hy", "T"), ("%hs", "T"), ("%hr", "T"), ("%r", "T")],
    );
}

/// The recursive call passes `%y` to `%x` and `%x + 1` to `%y`, with 0 and 0 from outside. The
/// sum is known only after the entry block's first visit, so `%y` becomes not a constant after
/// that visit, and `%x`, which reads the lattice of `%y` held by the same visit, must follow:
/// both end not a constant. The control passes the arguments back unchanged: both are 0.
#[test]
fn a_recursive_call_reading_a_lattice_that_changes_later_reaches_the_fixed_point() {
    let source = r#"builtin.module public @m {
    builtin.function private extern("C") @h(%c: i1, %x: i32, %y: i32) -> i32 {
        %one = arith.constant 1 : i32;
        %n = arith.add %x, %one;
        cf.cond_br %c ^again, ^done : (i1);
    ^again:
        %hr = builtin.call @h(%c, PASSED) : extern("C") (i1, i32, i32) -> i32;
        builtin.ret %hr : (i32);
    ^done:
        builtin.ret %y : (i32);
    };
    builtin.function public extern("C") @f(%fc: i1) -> i32 {
        %zero = arith.constant 0 : i32;
        %r = builtin.call @h(%fc, %zero, %zero) : extern("C") (i1, i32, i32) -> i32;
        builtin.ret %r : (i32);
    };
};"#;
    check(
        &source.replace("PASSED", "%y, %n"),
        &[("%x", "T"), ("%y", "T"), ("%n", "T"), ("%hr", "T"), ("%r", "T")],
    );
    check(
        &source.replace("PASSED", "%x, %y"),
        &[("%x", "0"), ("%y", "0"), ("%n", "1"), ("%hr", "0"), ("%r", "0")],
    );
}

/// A return of the result of a recursive call (`%kr`), and a return of an operand of that call
/// (`%kx`): the results of the call are joined from what both returns return. The control returns
/// another constant from the base case: the result is not a constant.
#[test]
fn a_return_may_return_a_recursive_calls_result_or_operand() {
    let source = r#"builtin.module public @m {
    builtin.function private extern("C") @k(%kc: i1, %kx: i32) -> i32 {
        cf.cond_br %kc ^again, ^done : (i1);
    ^again:
        %kr = builtin.call @k(%kc, %kx) : extern("C") (i1, i32) -> i32;
        builtin.ret %kr : (i32);
    ^done:
        %eight = arith.constant 8 : i32;
        builtin.ret RETURNED : (i32);
    };
    builtin.function public extern("C") @f(%fc: i1) -> i32 {
        %seven = arith.constant 7 : i32;
        %r = builtin.call @k(%fc, %seven) : extern("C") (i1, i32) -> i32;
        builtin.ret %r : (i32);
    };
};"#;
    check(&source.replace("RETURNED", "%kx"), &[("%kx", "7"), ("%kr", "7"), ("%r", "7")]);
    check(
        &source.replace("RETURNED", "%eight"),
        &[("%kx", "7"), ("%kr", "8"), ("%r", "8")],
    );
}

/// A loop whose back edge passes the header its own argument: the argument is what enters the
/// loop. The control passes the next value back: the argument is not a constant.
#[test]
fn a_back_edge_may_pass_the_header_its_own_argument() {
    let source = r#"builtin.function public extern("C") @f(%c: i1) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    cf.br ^head(%zero : i32);
^head(%x: i32):
    %n = arith.add %x, %one;
    cf.cond_br %c ^head(BACK : i32), ^exit : (i1);
^exit:
    builtin.ret %x : (i32);
};"#;
    check(&source.replace("BACK", "%x"), &[("%x", "0"), ("%n", "1")]);
    check(&source.replace("BACK", "%n"), &[("%x", "T"), ("%n", "T")]);
}

/// A back edge that passes `%y` to `%x` and `%x + 1` to `%y`: `%y` changes after the header's
/// first visit, and `%x` must follow. The control passes the arguments back unchanged.
#[test]
fn a_back_edge_may_swap_the_headers_arguments() {
    let source = r#"builtin.function public extern("C") @f(%c: i1) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    cf.br ^head(%zero, %zero : i32, i32);
^head(%x: i32, %y: i32):
    %n = arith.add %x, %one;
    cf.cond_br %c ^head(BACK : i32, i32), ^exit : (i1);
^exit:
    builtin.ret %x : (i32);
};"#;
    check(&source.replace("BACK", "%y, %n"), &[("%x", "T"), ("%y", "T"), ("%n", "T")]);
    check(&source.replace("BACK", "%x, %y"), &[("%x", "0"), ("%y", "0"), ("%n", "1")]);
}

/// A conditional branch that passes one value twice to one successor: both arguments are that
/// constant. The control passes two constants.
#[test]
fn a_branch_may_pass_one_value_twice() {
    let source = r#"builtin.function public extern("C") @f(%c: i1) -> i32 {
    %five = arith.constant 5 : i32;
    %six = arith.constant 6 : i32;
    cf.cond_br %c ^join(%five, PASSED : i32, i32), ^other : (i1);
^join(%a: i32, %b: i32):
    %s = arith.add %a, %b;
    builtin.ret %s : (i32);
^other:
    builtin.ret %five : (i32);
};"#;
    check(&source.replace("PASSED", "%five"), &[("%a", "5"), ("%b", "5"), ("%s", "10")]);
    check(&source.replace("PASSED", "%six"), &[("%a", "5"), ("%b", "6"), ("%s", "11")]);
}
