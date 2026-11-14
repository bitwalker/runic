//! Single decisions of the dialect, each stated by one test: how a typed load is spelled, what a
//! gep needs under a data layout, how large a path step may be, which copies of length zero are
//! erased, which loads merge, and that a forward-referenced address is a diagnostic.

use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
};

use runic_core::{
    Context, SourceSpan, SymbolTable, ValueRef,
    dialects::builtin::{Function, Module},
    parse::{self, ParserConfig},
    pass::{Nesting, OperationPass, PassManager},
    testing::{Test, text::describe},
};
use runic_transform::{Canonicalizer, CommonSubexpressionElimination};

use crate::{Load, MemOpBuilder};

/// `@f(params) -> (results)` with `body`, in a module stating `layout` if given, after `pass`
/// (with verification) when one is given, as text; the error is described with its labels.
fn run(
    layout: Option<&str>,
    params: &str,
    results: &str,
    body: &str,
    pass: Option<Box<dyn OperationPass>>,
) -> Result<String, String> {
    let context = alloc::rc::Rc::new(Context::default());
    let attributes = layout
        .map(|layout| format!(" attributes {{ data_layout = #builtin.data_layout<{layout}> }}"))
        .unwrap_or_default();
    let text = format!(
        "builtin.module public @m{attributes} {{\nbuiltin.function public extern(\"C\") \
         @f({params}) -> ({results}) {{\n{body}\n}};\n}};"
    );
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "mem", text)
        .into_result()
        .map_err(|err| describe(&err))?;
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    if let Some(pass) = pass {
        // the context must outlive every use of `function`: the pass manager gets a clone
        let mut passes = PassManager::on::<Function>(context.clone(), Nesting::Implicit);
        passes.add_pass(pass);
        passes.enable_verifier(true);
        passes.run(function).map_err(|err| describe(&err))?;
    }
    let text = function.borrow().to_string();
    drop(context);
    Ok(text)
}

fn canonicalize(layout: Option<&str>, params: &str, results: &str, body: &str) -> String {
    run(layout, params, results, body, Some(Box::<Canonicalizer>::default())).unwrap()
}

fn cse(params: &str, results: &str, body: &str) -> String {
    run(
        None,
        params,
        results,
        body,
        Some(Box::<CommonSubexpressionElimination>::default()),
    )
    .unwrap()
}

/// A load through a typed pointer that also states its pointee is the load written without it:
/// the parser and the builder drop the stated type, so CSE merges the two.
#[test]
fn a_load_that_states_its_pointee_merges_with_the_plain_form() {
    let text = cse(
        "%p: ptr<i32>",
        "i32, i32",
        "    %a = mem.load %p;
    %b = mem.load %p : i32;
    builtin.ret %a, %b : (i32, i32);",
    );
    assert_eq!(text.matches("mem.load").count(), 1, "{text}");

    let mut test = Test::new("f", |cx| ([cx.ptr_type(cx.i32())], []));
    let context = test.context_rc();
    let p = test.entry_block().borrow().arguments()[0] as ValueRef;
    let loaded = test.function_builder().load_as(p, context.i32(), SourceSpan::UNKNOWN).unwrap();
    let load = loaded.borrow().get_defining_op().unwrap();
    assert!(load.borrow().downcast_ref::<Load>().unwrap().get_ty().is_unknown());
}

/// Under a data layout, a gep with an index needs a pointee with a layout, and a gep with a path
/// needs the offset of the element it selects, as `mem.sizeof` and `mem.offsetof` of the same
/// types do; with no layout in scope nothing is checked against one.
#[test]
fn a_gep_needs_a_layout_when_one_is_in_scope() {
    let layout = Some("little, ptr = 32");
    let cases = [
        (
            "%p: ptr<(i32) -> i32>, %i: isize",
            "    %q = mem.gep %p, %i;",
            "the pointee stepped over, `(i32) -> i32`, has no layout under the data layout in \
             scope",
        ),
        (
            "%p: ptr<struct<i32, (i32) -> i32>>",
            "    %q = mem.gep %p, [1];",
            "the type the element is selected in, `struct<i32, (i32) -> i32>`, has no layout \
             under the data layout in scope",
        ),
        (
            "%p: ptr<struct<i32, (i32) -> i32>>",
            "    %q = mem.gep %p, [0];",
            "the type the element is selected in, `struct<i32, (i32) -> i32>`, has no layout",
        ),
    ];
    for (params, body, expected) in cases {
        let body = format!("{body}\n    builtin.ret;");
        let err = run(layout, params, "", &body, None).unwrap_err();
        assert!(err.contains("invalid operation mem.gep"), "{body}: {err}");
        assert!(err.contains(expected), "{body}: {err}");
        let ok = run(None, params, "", &body, None);
        assert!(ok.is_ok(), "{body}: {ok:?}");
    }
    for body in [
        "    %n = mem.sizeof (i32) -> i32;",
        "    %n = mem.offsetof struct<i32, (i32) -> i32>, [1];",
    ] {
        let err = run(layout, "", "", &format!("{body}\n    builtin.ret;"), None).unwrap_err();
        assert!(err.contains("has no layout under the data layout in scope"), "{err}");
    }
    // a pointee with a layout, and a gep with neither index nor path over one without
    let ok = run(
        layout,
        "%p: ptr<struct<i32, array<i8; 4>>>, %f: ptr<(i32) -> i32>, %i: isize",
        "",
        "    %q = mem.gep %p, %i, [1, 2];\n    %r = mem.gep %f;\n    builtin.ret;",
        None,
    );
    assert!(ok.is_ok(), "{ok:?}");
}

/// A path step above `i32::MAX` is a parse diagnostic that names the step and the limit, in an
/// `offsetof` and in a gep; the largest step parses; a negative step is the verifier's.
#[test]
fn a_path_step_above_i32_max_is_reported_as_too_large() {
    for (params, body) in [
        ("", "    %n = mem.offsetof array<i8; 4294967296>, [3000000000];"),
        ("%p: ptr<array<i8; 4294967296>>", "    %q = mem.gep %p, [3000000000];"),
        ("", "    %n = mem.offsetof array<i8; 4294967296>, [4294967296];"),
    ] {
        let err = run(None, params, "", &format!("{body}\n    builtin.ret;"), None).unwrap_err();
        assert!(!err.contains("negative"), "{err}");
        assert!(err.contains("a path step is at most 2147483647"), "{err}");
    }
    let err = run(
        None,
        "",
        "",
        "    %n = mem.offsetof array<i8; 4294967296>, [3000000000];\n    builtin.ret;",
        None,
    )
    .unwrap_err();
    assert!(err.contains("the step 3000000000 is too large"), "{err}");
    let ok = run(
        None,
        "",
        "",
        "    %n = mem.offsetof array<i8; 4294967296>, [2147483647];\n    builtin.ret;",
        None,
    );
    assert!(ok.is_ok(), "{ok:?}");
    let err = run(
        None,
        "",
        "",
        "    %n = mem.offsetof array<i8; 4>, [-1];\n    builtin.ret;",
        None,
    )
    .unwrap_err();
    assert!(err.contains("the path [-1] has the negative step -1"), "{err}");
}

/// A copy or fill whose constant length is zero only in its low 32 bits (`4294967296 : isize`)
/// is kept, with no layout and under 32-bit and 64-bit layouts; one of length 0 is erased.
#[test]
fn a_length_that_is_zero_only_in_its_low_32_bits_is_kept() {
    for layout in [None, Some("little, ptr = 32"), Some("little, ptr = 64")] {
        let text = canonicalize(
            layout,
            "%p: ptr<i8>, %q: ptr<i8>, %b: i8",
            "",
            "    %n = arith.constant 4294967296 : isize;
    %z = arith.constant 0 : isize;
    mem.memcpy %p, %q, %n;
    mem.memmove %p, %q, %n;
    mem.memset %p, %b, %n;
    mem.memcpy %p, %q, %z;
    mem.memmove %p, %q, %z;
    mem.memset %p, %b, %z;
    builtin.ret;",
        );
        assert_eq!(text.matches("mem.mem").count(), 3, "{layout:?}\n{text}");
        assert!(!text.contains("%z") && text.contains("4294967296"), "{text}");
    }
}

/// Two loads of different types through one opaque pointer do not merge; a store to another
/// address blocks; an alloca between two loads does not; a used alloca survives.
#[test]
fn load_merging_decisions() {
    let text = cse(
        "%q: ptr",
        "i32, i64, i32",
        "    %a = mem.load %q : i32;
    %b = mem.load %q : i64;
    %c = mem.load %q : i32;
    builtin.ret %a, %b, %c : (i32, i64, i32);",
    );
    assert_eq!(text.matches("mem.load").count(), 2, "{text}");
    let text = cse(
        "%p: ptr<i32>, %o: ptr<i32>, %v: i32",
        "i32, i32",
        "    %a = mem.load %p;
    mem.store %v, %o;
    %b = mem.load %p;
    builtin.ret %a, %b : (i32, i32);",
    );
    assert_eq!(text.matches("mem.load").count(), 2, "{text}");
    let text = cse(
        "%p: ptr<i32>, %v: i32",
        "i32, i32, ptr<i32>",
        "    %a = mem.load %p;
    %s = mem.alloca i32;
    %b = mem.load %p;
    builtin.ret %a, %b, %s : (i32, i32, ptr<i32>);",
    );
    assert_eq!(text.matches("mem.load").count(), 1, "{text}");
    assert_eq!(text.matches("mem.alloca").count(), 1, "{text}");
}

/// An operand used before its definition (legal in a block no path reaches) has no type when its
/// user is built, so an operation whose constraint or inference needs the type is a diagnostic,
/// never a panic: a cycle of casts or geps in dead code cannot be written.
#[test]
fn a_forward_referenced_address_is_a_diagnostic() {
    for body in [
        "    builtin.ret;\n^dead:\n    %a = mem.ptrcast %b : ptr<i8>;\n    %b = mem.ptrcast %a : \
         ptr<i32>;\n    cf.br ^dead;",
        "    builtin.ret;\n^dead:\n    %a = mem.gep %a, [0];\n    cf.br ^dead;",
        "    builtin.ret;\n^dead:\n    %v = mem.load %a;\n    %a = mem.alloca i32;\n    cf.br \
         ^dead;",
    ] {
        let context = alloc::rc::Rc::new(Context::default());
        context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
        let text = format!(
            "builtin.module public @m {{\nbuiltin.function public extern(\"C\") @f() \
             {{\n{body}\n}};\n}};"
        );
        let result = parse::parse::<Module>(ParserConfig::new(context.clone()), "mem", text)
            .into_result()
            .map(|_| ())
            .map_err(|err| describe(&err));
        assert!(result.is_err(), "{body}");
    }
}
