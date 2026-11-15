//! Block merging through the canonicalizer, with a rewrite config that asks for aggressive
//! region simplification: the cases of upstream's `canonicalize-block-merge.mlir` that the `cf`,
//! `arith`, `scf`, `ub` and `test` dialects can express, and `cf.switch` cases merged through
//! their keyed successors.
//!
//! The canonicalizer runs without `split-critical-edges`: that pattern splits the edges of a
//! branch into a block with two predecessors into passthrough blocks, which block merging merges
//! back into one, so with both the canonicalizer does not converge.

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
};

use runic_core::{
    Context, Op,
    dialects::builtin::{Function, FunctionRef},
    parse::{self, ParserConfig},
    pass::{Nesting, Pass, PassManager, PassOptions},
    patterns::{GreedyRewriteConfig, RegionSimplificationLevel},
    print::same_program,
    testing::text::describe,
};

use crate::Canonicalizer;

/// The function `@f` with the parameters `params`, the results `results` (` -> ..` or nothing)
/// and the body `body`.
fn function(params: &str, results: &str, body: &str) -> String {
    format!("builtin.function public extern(\"C\") @f({params}){results} {{\n{body}\n}};")
}

fn parse_function(context: &Rc<Context>, source: &str) -> FunctionRef {
    use runic_dialect_arith as _;
    use runic_dialect_scf as _;
    use runic_dialect_ub as _;
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    parse::parse::<Function>(ParserConfig::new(context.clone()), "f", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}\n{source}", describe(&err)))
}

/// `source` canonicalized to a fixpoint with aggressive region simplification, as printed.
fn canonicalize(source: &str) -> String {
    let context = Rc::new(Context::default());
    let function = parse_function(&context, source);
    let mut config = GreedyRewriteConfig::default();
    config
        .with_top_down_traversal(true)
        .with_region_simplification_level(RegionSimplificationLevel::Aggressive);
    let mut pass = Canonicalizer::new(config, true);
    let options = PassOptions::parse("disable-patterns=split-critical-edges").unwrap();
    pass.initialize_options(&options).unwrap();
    let mut manager = PassManager::on::<Function>(context.clone(), Nesting::Implicit);
    manager.add_pass(Box::new(pass)).unwrap();
    manager.enable_verifier(true);
    manager
        .run(function.as_operation_ref())
        .unwrap_or_else(|err| panic!("{}\n{source}", describe(&err)));
    function.borrow().as_operation().to_string()
}

/// Canonicalizing `input` gives `expected`.
#[track_caller]
fn assert_canonicalizes(input: &str, expected: &str) {
    let output = canonicalize(input);
    let context = Rc::new(Context::default());
    let expected = parse_function(&context, expected).borrow().as_operation().to_string();
    assert!(same_program(&output, &expected), "expected\n{expected}\nfound\n{output}");
}

/// Canonicalizing `input` leaves it as it is.
#[track_caller]
fn assert_unchanged(input: &str) {
    assert_canonicalizes(input, input);
}

#[test]
fn return_blocks() {
    assert_canonicalizes(
        &function(
            "%c: i1",
            "",
            "    cf.cond_br %c ^a, ^b : (i1);
^a:
    builtin.ret;
^b:
    builtin.ret;",
        ),
        &function(
            "%c: i1",
            "",
            "    cf.cond_br %c ^a, ^a : (i1);
^a:
    builtin.ret;",
        ),
    );
}

#[test]
fn matching_arguments() {
    assert_canonicalizes(
        &function(
            "%c: i1, %p: i32, %q: i32",
            " -> i32",
            "    cf.cond_br %c ^a(%p : i32), ^b(%q : i32) : (i1);
^a(%x: i32):
    builtin.ret %x : (i32);
^b(%y: i32):
    builtin.ret %y : (i32);",
        ),
        &function(
            "%c: i1, %p: i32, %q: i32",
            " -> i32",
            "    cf.cond_br %c ^a(%p : i32), ^a(%q : i32) : (i1);
^a(%x: i32):
    builtin.ret %x : (i32);",
        ),
    );
}

/// Upstream folds the merged branch into an `arith.select`; no `cf` pattern does that here.
#[test]
fn mismatch_operands() {
    assert_canonicalizes(
        &function(
            "%c: i1, %p: i32, %q: i32",
            " -> i32",
            "    cf.cond_br %c ^a, ^b : (i1);
^a:
    builtin.ret %p : (i32);
^b:
    builtin.ret %q : (i32);",
        ),
        &function(
            "%c: i1, %p: i32, %q: i32",
            " -> i32",
            "    cf.cond_br %c ^a(%p : i32), ^a(%q : i32) : (i1);
^a(%x: i32):
    builtin.ret %x : (i32);",
        ),
    );
}

#[test]
fn mismatch_operands_matching_arguments() {
    assert_canonicalizes(
        &function(
            "%c: i1, %p: i32, %q: i32",
            " -> (i32, i32)",
            "    cf.cond_br %c ^a(%q : i32), ^b(%p : i32) : (i1);
^a(%x: i32):
    builtin.ret %p, %x : (i32, i32);
^b(%y: i32):
    builtin.ret %q, %y : (i32, i32);",
        ),
        &function(
            "%c: i1, %p: i32, %q: i32",
            " -> (i32, i32)",
            "    cf.cond_br %c ^a(%p, %q : i32, i32), ^a(%q, %p : i32, i32) : (i1);
^a(%x: i32, %y: i32):
    builtin.ret %x, %y : (i32, i32);",
        ),
    );
}

/// The arguments of the two blocks are used differently, but each block has one predecessor,
/// so the canonicalizer replaces them with the values passed, and then the blocks are the same.
#[test]
fn mismatch_argument_uses() {
    assert_canonicalizes(
        &function(
            "%c: i1, %p: i32, %q: i32",
            " -> (i32, i32)",
            "    cf.cond_br %c ^a(%q : i32), ^b(%p : i32) : (i1);
^a(%x: i32):
    builtin.ret %p, %x : (i32, i32);
^b(%y: i32):
    builtin.ret %y, %q : (i32, i32);",
        ),
        &function(
            "%c: i1, %p: i32, %q: i32",
            " -> (i32, i32)",
            "    cf.cond_br %c ^a, ^a : (i1);
^a:
    builtin.ret %p, %q : (i32, i32);",
        ),
    );
}

#[test]
fn mismatch_argument_types() {
    assert_canonicalizes(
        &function(
            "%c: i1, %p: i32, %q: i16, %pp: ptr<i32>, %pq: ptr<i16>",
            "",
            "    cf.cond_br %c ^a(%p : i32), ^b(%q : i16) : (i1);
^a(%x: i32):
    test.store %pp, %x : (ptr<i32>, i32);
    builtin.ret;
^b(%y: i16):
    test.store %pq, %y : (ptr<i16>, i16);
    builtin.ret;",
        ),
        &function(
            "%c: i1, %p: i32, %q: i16, %pp: ptr<i32>, %pq: ptr<i16>",
            "",
            "    cf.cond_br %c ^a, ^b : (i1);
^a:
    test.store %pp, %p : (ptr<i32>, i32);
    builtin.ret;
^b:
    test.store %pq, %q : (ptr<i16>, i16);
    builtin.ret;",
        ),
    );
}

#[test]
fn mismatch_operations() {
    assert_unchanged(&function(
        "%c: i1",
        "",
        "    cf.cond_br %c ^a, ^b : (i1);
^a:
    ub.unreachable;
^b:
    builtin.ret;",
    ));
}

#[test]
fn mismatch_operation_count() {
    assert_unchanged(&function(
        "%c: i1, %p: i32, %pp: ptr<i32>",
        "",
        "    cf.cond_br %c ^a, ^b : (i1);
^a:
    test.store %pp, %p : (ptr<i32>, i32);
    builtin.ret;
^b:
    builtin.ret;",
    ));
}

#[test]
fn contains_regions() {
    assert_unchanged(&function(
        "%c: i1, %p: i32, %pp: ptr<i32>",
        "",
        "    cf.cond_br %c ^a, ^b : (i1);
^a:
    scf.if %c then {
        test.store %pp, %p : (ptr<i32>, i32);
        scf.yield : ();
    } else {
        scf.yield : ();
    } : (i1) -> ();
    builtin.ret;
^b:
    scf.if %c then {
        test.store %pp, %p : (ptr<i32>, i32);
        scf.yield : ();
    } else {
        scf.yield : ();
    } : (i1) -> ();
    builtin.ret;",
    ));
}

/// The merged block's new argument is passed on its back edge too.
#[test]
fn mismatch_loop() {
    assert_canonicalizes(
        &function(
            "%c: i1, %d: i1, %f: ptr<i1>",
            "",
            "    %e = test.load %f;
    cf.cond_br %c ^b, ^exit : (i1);
^a:
    cf.cond_br %e ^a, ^exit : (i1);
^b:
    cf.cond_br %d ^a, ^exit : (i1);
^exit:
    builtin.ret;",
        ),
        &function(
            "%c: i1, %d: i1, %f: ptr<i1>",
            "",
            "    %e = test.load %f;
    cf.cond_br %c ^a(%d : i1), ^exit : (i1);
^a(%x: i1):
    cf.cond_br %x ^a(%e : i1), ^exit : (i1);
^exit:
    builtin.ret;",
        ),
    );
}

#[test]
fn mismatch_operand_types() {
    assert_unchanged(&function(
        "%c: i1, %p: ptr<i32>, %q: ptr<i1>",
        "",
        "    %zero = arith.constant 0 : i32;
    %true = arith.constant 1 : i1;
    cf.br ^head;
^head:
    cf.cond_br %c ^a, ^b : (i1);
^a:
    test.store %p, %zero : (ptr<i32>, i32);
    cf.br ^head;
^b:
    test.store %q, %true : (ptr<i1>, i1);
    cf.br ^head;",
    ));
}

/// `^b` and `^next` branch to `^loop` alike, but `^loop` uses what `^b` computes: merged, it
/// would store what `^next` computes after the back edge.
#[test]
fn nomerge() {
    assert_unchanged(&function(
        "%n: i32, %i: i32, %pp: ptr<i32>",
        "",
        "    %one = arith.constant 1 : i32;
    %lt = arith.cmpi slt %i, %n;
    cf.cond_br %lt ^b, ^exit : (i1);
^b:
    %ip1 = arith.add %i, %one;
    cf.br ^loop(%ip1 : i32);
^loop(%j: i32):
    %jlt = arith.cmpi slt %j, %n;
    test.store %pp, %j : (ptr<i32>, i32);
    test.store %pp, %ip1 : (ptr<i32>, i32);
    cf.cond_br %jlt ^next, ^exit : (i1);
^next:
    %jp1 = arith.add %j, %one;
    cf.br ^loop(%jp1 : i32);
^exit:
    builtin.ret;",
    ));
}

/// An irreducible loop entered in the middle: `^header` uses the argument of `^body_entry`,
/// so `^body_entry` is not merged with `^body_other` (whose argument the canonicalizer replaces
/// with the one value passed to it).
#[test]
fn nested_loop() {
    assert_canonicalizes(
        &function(
            "%a0: i32, %a4: i32, %pp: ptr<i32>",
            "",
            "    cf.br ^body_entry(%a0 : i32);
^header:
    cf.switch %x [#builtin.i32<0> -> ^exit, #builtin.i32<1> -> ^body_entry(%a4 : i32)], \
             ^body_other(%a4 : i32) : (i32);
^body_entry(%x: i32):
    test.store %pp, %x : (ptr<i32>, i32);
    cf.br ^header;
^body_other(%y: i32):
    test.store %pp, %y : (ptr<i32>, i32);
    cf.br ^header;
^exit:
    builtin.ret;",
        ),
        &function(
            "%a0: i32, %a4: i32, %pp: ptr<i32>",
            "",
            "    cf.br ^body_entry(%a0 : i32);
^header:
    cf.switch %x [#builtin.i32<0> -> ^exit, #builtin.i32<1> -> ^body_entry(%a4 : i32)], \
             ^body_other : (i32);
^body_entry(%x: i32):
    test.store %pp, %x : (ptr<i32>, i32);
    cf.br ^header;
^body_other:
    test.store %pp, %a4 : (ptr<i32>, i32);
    cf.br ^header;
^exit:
    builtin.ret;",
        ),
    );
}

/// Two cases of a switch go to blocks that differ in a value from above: the merged block takes
/// it as an argument, which each case passes through its keyed successor.
#[test]
fn switch_cases_are_merged_through_their_keyed_successors() {
    assert_canonicalizes(
        &function(
            "%s: i32, %p: i32, %q: i32",
            " -> i32",
            "    cf.switch %s [#builtin.i32<1> -> ^a, #builtin.i32<2> -> ^b], ^c : (i32);
^a:
    builtin.ret %p : (i32);
^b:
    builtin.ret %q : (i32);
^c:
    %z = arith.add %p, %q;
    builtin.ret %z : (i32);",
        ),
        &function(
            "%s: i32, %p: i32, %q: i32",
            " -> i32",
            "    cf.switch %s [#builtin.i32<1> -> ^a(%p : i32), #builtin.i32<2> -> ^a(%q : i32)], \
             ^c : (i32);
^a(%x: i32):
    builtin.ret %x : (i32);
^c:
    %z = arith.add %p, %q;
    builtin.ret %z : (i32);",
        ),
    );
}

/// A case and the fallback go to the same block once merged; the case that then duplicates the
/// fallback is folded away, and the switch of one remaining case becomes a conditional branch.
#[test]
fn a_switch_whose_targets_merge_is_simplified() {
    assert_canonicalizes(
        &function(
            "%s: i32, %p: i32, %q: i32",
            " -> i32",
            "    cf.switch %s [#builtin.i32<1> -> ^a, #builtin.i32<2> -> ^b], ^c : (i32);
^a:
    builtin.ret %p : (i32);
^b:
    builtin.ret %q : (i32);
^c:
    builtin.ret %p : (i32);",
        ),
        &function(
            "%s: i32, %p: i32, %q: i32",
            " -> i32",
            "    %two = arith.constant 2 : i32;
    %is_two = arith.cmpi eq %s, %two;
    cf.cond_br %is_two ^a(%q : i32), ^a(%p : i32) : (i1);
^a(%x: i32):
    builtin.ret %x : (i32);",
        ),
    );
}

/// A case and the fallback of a switch go to blocks that differ in a value from above: the case,
/// a keyed successor, reaches the leader, and the fallback the block merged into it.
#[test]
fn a_switch_case_and_its_fallback_pass_the_new_argument() {
    assert_canonicalizes(
        &function(
            "%s: i32, %p: i32, %q: i32",
            " -> i32",
            "    cf.switch %s [#builtin.i32<1> -> ^a, #builtin.i32<2> -> ^c], ^b : (i32);
^a:
    builtin.ret %p : (i32);
^b:
    builtin.ret %q : (i32);
^c:
    %z = arith.add %p, %q;
    builtin.ret %z : (i32);",
        ),
        &function(
            "%s: i32, %p: i32, %q: i32",
            " -> i32",
            "    cf.switch %s [#builtin.i32<1> -> ^a(%p : i32), #builtin.i32<2> -> ^c], ^a(%q : \
             i32) : (i32);
^a(%x: i32):
    builtin.ret %x : (i32);
^c:
    %z = arith.add %p, %q;
    builtin.ret %z : (i32);",
        ),
    );
}
