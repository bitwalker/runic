//! `builtin.tail_call` and `builtin.tail_call_indirect` through the passes of this crate: a tail
//! call is a terminator with unknown effects that leaves the function, so no pass removes,
//! merges, copies or moves one, and none reads its operands as the values the function returns.
//!
//! Each test names, beside what must stay, a control that shows the pass does its work on the
//! same shape with a `builtin.ret`.

extern crate std;

use alloc::{format, rc::Rc, string::ToString};

use runic_ir::{
    Context, Report,
    parse::{ParserConfig, parse_any},
    pass::{Nesting, PassPipeline},
};

/// `body`, the text of a module, after `pipeline` ran on it, as text.
fn run(pipeline: &str, body: &str) -> Result<alloc::string::String, Report> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    context.get_or_register_dialect::<runic_dialect_scf::ScfDialect>();
    let text = format!(
        "builtin.module public @m {{\n    builtin.function public extern(\"C\") @g(i32) -> \
         i32;\n    builtin.function public extern(\"C\") @h(i32) -> i32 effects();\n{body}\n}};"
    );
    let module = parse_any(ParserConfig::new(context.clone()), "tail_calls", text).into_result()?;
    let pipeline = PassPipeline::parse_elements(&module.borrow().name(), pipeline)?;
    let mut passes = pipeline.load(context.clone(), Nesting::Implicit)?;
    passes.enable_verifier(true);
    passes.run(module)?;
    let text = module.borrow().to_string();
    drop(passes);
    Ok(text)
}

fn count(text: &str, what: &str) -> usize {
    text.matches(what).count()
}

/// Identical tail calls in two blocks are two exits: common subexpression elimination does not
/// merge terminators, and a tail call has unknown effects besides.
#[test]
fn cse_keeps_two_identical_tail_calls() -> Result<(), Report> {
    let text = run(
        "cse",
        r#"
    builtin.function public extern("C") @f(%c: i1, %a: i32) -> i32 {
        cf.cond_br %c ^x, ^y : (i1);
    ^x:
        builtin.tail_call @g(%a) : extern("C") (i32) -> i32;
    ^y:
        builtin.tail_call @g(%a) : extern("C") (i32) -> i32;
    };"#,
    )?;
    assert_eq!(count(&text, "builtin.tail_call @g"), 2, "{text}");
    Ok(())
}

/// A load before a tail call is neither removed nor moved after it, and the tail call, which
/// has unknown effects, is not dropped for having no results.
#[test]
fn cse_and_dce_keep_a_tail_call_after_a_load() -> Result<(), Report> {
    let body = r#"
    builtin.function public extern("C") @f(%p: ptr<i32>) -> i32 {
        %x = mem.load %p : i32;
        builtin.tail_call @g(%x) : extern("C") (i32) -> i32;
    };"#;
    for pipeline in ["cse", "dce", "canonicalizer,cse,sccp,canonicalizer,dce"] {
        let text = run(pipeline, body)?;
        let load = text.find("mem.load").expect(&text);
        assert!(load < text.find("builtin.tail_call @g").expect(&text), "{pipeline}: {text}");
    }
    Ok(())
}

/// The operands of a tail call are live: an operation whose result a tail call takes stays, and
/// one whose result is unused goes. The function a tail call names is not removed.
#[test]
fn dce_keeps_the_definitions_of_the_arguments_of_a_tail_call() -> Result<(), Report> {
    let text = run(
        "dce",
        r#"
    builtin.function private extern("C") @only_tail_called(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function public extern("C") @f(%a: i32, %b: i32) -> i32 {
        %used = arith.add %a, %b;
        %unused = arith.mul %a, %b;
        builtin.tail_call @only_tail_called(%used) : extern("C") (i32) -> i32;
    };"#,
    )?;
    assert_eq!(count(&text, "arith.add"), 1, "{text}");
    assert_eq!(count(&text, "arith.mul"), 0, "{text}");
    assert_eq!(count(&text, "@only_tail_called"), 2, "{text}");
    Ok(())
}

/// A block that holds only a return is copied into each predecessor that branches to it; a block
/// that holds only a tail call is not: the branches stay, the one tail call stays one. The
/// control is the return.
#[test]
fn canonicalization_does_not_copy_a_tail_call_into_its_predecessors() -> Result<(), Report> {
    let text = run(
        "canonicalizer",
        r#"
    builtin.function public extern("C") @with_tail_call(%c: i1, %a: i32) -> i32 {
        cf.cond_br %c ^x, ^y : (i1);
    ^x:
        cf.br ^t(%a : i32);
    ^y:
        %b = arith.add %a, %a;
        cf.br ^t(%b : i32);
    ^t(%v: i32):
        builtin.tail_call @g(%v) : extern("C") (i32) -> i32;
    };
    builtin.function public extern("C") @with_ret(%c: i1, %a: i32) -> i32 {
        cf.cond_br %c ^x, ^y : (i1);
    ^x:
        cf.br ^t(%a : i32);
    ^y:
        %b = arith.add %a, %a;
        cf.br ^t(%b : i32);
    ^t(%v: i32):
        builtin.ret %v : (i32);
    };"#,
    )?;
    let (with_tail_call, with_ret) = text.split_once("@with_ret").unwrap();
    assert_eq!(count(with_tail_call, "builtin.tail_call"), 1, "{text}");
    assert_eq!(count(with_tail_call, "cf.br"), 2, "{text}");
    assert_eq!(count(with_ret, "builtin.ret"), 2, "the control copies the return: {text}");
    assert_eq!(count(with_ret, "cf.br"), 0, "{text}");
    Ok(())
}

/// A block with one predecessor that holds a tail call is merged into it: the tail call, moved,
/// is the one terminator of the merged block, with the argument the branch passed.
#[test]
fn canonicalization_merges_a_block_that_holds_a_tail_call_into_its_one_predecessor()
-> Result<(), Report> {
    let text = run(
        "canonicalizer",
        r#"
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        cf.br ^t(%a : i32);
    ^t(%v: i32):
        builtin.tail_call @g(%v) : extern("C") (i32) -> i32;
    };"#,
    )?;
    assert_eq!(count(&text, "builtin.tail_call"), 1, "{text}");
    assert_eq!(count(&text, "cf.br"), 0, "{text}");
    assert!(text.contains("builtin.tail_call @g(%0)"), "{text}");
    Ok(())
}

/// Constant propagation through a function whose only exit is a tail call keeps the call, its
/// constant argument, and a call of the function: nothing is known of what it returns.
#[test]
fn sccp_keeps_a_function_that_tail_calls_and_the_calls_of_it() -> Result<(), Report> {
    let body = r#"
    builtin.function private extern("C") @five() -> i32 {
        %c = arith.constant 5 : i32;
        builtin.ret %c : (i32);
    };
    builtin.function private extern("C") @through() -> i32 {
        builtin.tail_call @five() : extern("C") () -> i32;
    };
    builtin.function public extern("C") @user() -> i32 {
        %r = builtin.call @through() : extern("C") () -> i32;
        %one = arith.constant 1 : i32;
        %s = arith.add %r, %one;
        builtin.ret %s : (i32);
    };"#;
    let text = run("sccp", body)?;
    assert_eq!(count(&text, "builtin.tail_call @five"), 1, "{text}");
    assert_eq!(count(&text, "builtin.call @through"), 1, "{text}");
    assert_eq!(count(&text, "arith.add"), 1, "{text}");
    Ok(())
}

/// A load of a local that feeds a tail call is promoted to the value stored, as it would be for
/// a return; a store before a tail call whose local nothing loads is dead, as a function's
/// locals end with it.
#[test]
fn local2reg_promotes_a_local_that_feeds_a_tail_call() -> Result<(), Report> {
    let text = run(
        "local2reg",
        r#"
    builtin.function public extern("C") @promoted(%a: i32) -> i32 locals(i32) {
        builtin.store_local 0, %a;
        %l = builtin.load_local 0 : i32;
        builtin.tail_call @g(%l) : extern("C") (i32) -> i32;
    };
    builtin.function public extern("C") @dead(%a: i32) -> i32 locals(i32) {
        builtin.store_local 0, %a;
        builtin.tail_call @g(%a) : extern("C") (i32) -> i32;
    };"#,
    )?;
    assert_eq!(count(&text, "builtin.store_local"), 0, "{text}");
    assert_eq!(count(&text, "builtin.load_local"), 0, "{text}");
    assert_eq!(count(&text, "builtin.tail_call @g"), 2, "{text}");
    let (promoted, _) = text.split_once("@dead").unwrap();
    assert!(promoted.contains("builtin.tail_call @g(%0)"), "{text}");
    Ok(())
}

/// A tail call between a store and its load, in the arm of an `scf.if`, is control flow between
/// them: the local is not promoted, and the store whose value only another path reads stays.
#[test]
fn local2reg_does_not_promote_across_a_tail_call_in_a_region() -> Result<(), Report> {
    let text = run(
        "local2reg",
        r#"
    builtin.function public extern("C") @f(%c: i1, %a: i32) -> i32 locals(i32) {
        builtin.store_local 0, %a;
        scf.if %c then {
            builtin.tail_call @g(%a) : extern("C") (i32) -> i32;
        } else {
            scf.yield : ();
        } : (i1) -> ();
        %l = builtin.load_local 0 : i32;
        builtin.ret %l : (i32);
    };"#,
    )?;
    assert_eq!(count(&text, "builtin.store_local"), 1, "{text}");
    assert_eq!(count(&text, "builtin.load_local"), 1, "{text}");
    Ok(())
}

/// The pipeline of the lit suite over a module with a tail call of every kind: each stays, the
/// functions they name stay, and what calls a function that tail calls is not folded.
#[test]
fn the_pipeline_keeps_every_tail_call() -> Result<(), Report> {
    let body = r#"
    builtin.function private extern("C") @callee(%x: i32) -> i32 {
        %one = arith.constant 1 : i32;
        %r = arith.add %x, %one;
        builtin.ret %r : (i32);
    };
    builtin.function public extern("C") @direct(%a: i32) -> i32 {
        %two = arith.constant 2 : i32;
        %s = arith.add %a, %two;
        builtin.tail_call @callee(%s) : extern("C") (i32) -> i32;
    };
    builtin.function public extern("C") @indirect(%f: (i32) -> i32, %a: i32) -> i32 {
        builtin.tail_call_indirect %f(%a) : extern("C") (i32) -> i32;
    };
    builtin.function public extern("C") @through_ref(%a: i32) -> i32 {
        %f = builtin.function_ref @callee : (i32) -> i32;
        builtin.tail_call_indirect %f(%a) : extern("C") (i32) -> i32;
    };
    builtin.function public extern("C") @mixed(%c: i1, %a: i32) -> i32 {
        cf.cond_br %c ^x, ^y : (i1);
    ^x:
        builtin.tail_call @callee(%a) : extern("C") (i32) -> i32;
    ^y:
        %seven = arith.constant 7 : i32;
        builtin.ret %seven : (i32);
    };
    builtin.function public extern("C") @user(%c: i1, %a: i32) -> i32 {
        %r = builtin.call @mixed(%c, %a) : extern("C") (i1, i32) -> i32;
        builtin.ret %r : (i32);
    };"#;
    let before = run("dce", body)?;
    let after = run("canonicalizer,cse,sccp,canonicalizer,dce", body)?;
    // The indirect tail call of a function reference becomes a direct one: the count of both
    // kinds together is the same.
    assert_eq!(count(&before, "builtin.tail_call"), 4, "{before}");
    assert_eq!(count(&after, "builtin.tail_call"), 4, "{after}");
    assert_eq!(count(&after, "builtin.tail_call_indirect"), 1, "{after}");
    assert!(after.contains("builtin.function private extern(\"C\") @callee"), "{after}");
    assert_eq!(count(&after, "builtin.call @mixed"), 1, "{after}");
    Ok(())
}

/// `cfg-to-scf` merges the tail calls of one callee from two blocks into one exit that takes its
/// argument from an `scf.if`, and an indirect tail call, whose callee and arguments are two
/// operand groups, keeps both groups when two are merged: the output verifies.
#[test]
fn cfg_to_scf_merges_tail_calls_and_keeps_their_operand_groups() -> Result<(), Report> {
    let text = run(
        "cfg-to-scf",
        r#"
    builtin.function public extern("C") @direct(%c: i1, %a: i32, %b: i32) -> i32 {
        cf.cond_br %c ^x, ^y : (i1);
    ^x:
        builtin.tail_call @g(%a) : extern("C") (i32) -> i32;
    ^y:
        builtin.tail_call @g(%b) : extern("C") (i32) -> i32;
    };
    builtin.function public extern("C") @indirect(%c: i1, %f: (i32) -> i32, %a: i32, %b: i32) -> i32 {
        cf.cond_br %c ^x, ^y : (i1);
    ^x:
        builtin.tail_call_indirect %f(%a) : extern("C") (i32) -> i32;
    ^y:
        builtin.tail_call_indirect %f(%b) : extern("C") (i32) -> i32;
    };
    builtin.function public extern("C") @three_groups(%c: i1, %f: (i32, i32) -> i32, %a: i32, %b: i32) -> i32 {
        cf.cond_br %c ^x, ^y : (i1);
    ^x:
        builtin.tail_call_indirect %f(%a, %b) : extern("C") (i32, i32) -> i32;
    ^y:
        builtin.tail_call_indirect %f(%b, %a) : extern("C") (i32, i32) -> i32;
    };"#,
    )?;
    assert_eq!(count(&text, "builtin.tail_call @g"), 1, "{text}");
    assert_eq!(count(&text, "builtin.tail_call_indirect"), 2, "{text}");
    assert_eq!(count(&text, "scf.if"), 3, "{text}");
    assert!(text.contains("builtin.tail_call_indirect %"), "{text}");
    Ok(())
}
