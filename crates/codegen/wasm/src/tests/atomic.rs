//! `convert-to-wasm` either converts everything or changes nothing: every diagnostic the pass can
//! raise on input is raised by the check that runs before the conversion of any module starts.
//!
//! Each case provokes one diagnostic in a module that also holds a function that would convert
//! (`@ok`, an `arith.add` of two `i8`, written first), and asserts that the module's text after
//! the failed pass is its text before.

use alloc::boxed::Box;

use runic_ir::pass::{Nesting, PassManager};

use super::*;
use crate::ConvertToWasm;

/// A function that converts.
const OK: &str = "builtin.function public extern(\"wasm\") @ok(%a: i8, %b: i8) -> i8 {\n    %s = \
                  arith.add %a, %b <{ overflow = #builtin.overflow<wrapping> }>;\n    builtin.ret \
                  %s : (i8);\n};";

/// One diagnostic of `convert-to-wasm`: what provokes it.
struct Case {
    /// What the case is.
    what: &'static str,
    /// The functions (and other symbols) of the module, after [OK].
    functions: &'static str,
    /// Whether the module verifies; one that does not is given to the pass unverified.
    verifies: bool,
    /// What the diagnostic says.
    says: &'static str,
}

/// Every diagnostic `convert-to-wasm` raises on the functions of a module.
const CASES: &[Case] = &[
    Case {
        what: "an unknown calling convention on a function",
        functions: "builtin.function public extern(\"fast\") @bad(%x: i32) -> i32 {\n    \
                    builtin.ret %x : (i32);\n};",
        verifies: true,
        says: "`convert-to-wasm` lowers the calling conventions \"wasm\" and \"C\", not \"fast\"",
    },
    Case {
        what: "an unknown calling convention on a declaration",
        functions: "builtin.function public extern(\"fast\") @decl(i8);",
        verifies: true,
        says: "this `builtin.function` uses the calling convention \"fast\"",
    },
    Case {
        what: "an unknown calling convention on a call",
        functions: "builtin.function public extern(\"wasm\") @caller(%x: i8) {\n    builtin.call \
                    @decl(%x) : extern(\"fast\") (i8) -> ();\n    \
                    builtin.ret;\n};\nbuiltin.function public extern(\"fast\") @decl(i8);",
        verifies: true,
        says: "this `builtin.call` uses the calling convention \"fast\"",
    },
    Case {
        what: "an integer wider than 64 bits",
        functions: "builtin.function public extern(\"C\") @wide(%x: i128) {\n    builtin.ret;\n};",
        verifies: true,
        says: "`convert-to-wasm` cannot lower a value of type `i128`: integers wider than 64 bits \
               are not lowered yet",
    },
    Case {
        what: "a wide result of an operation",
        functions: "builtin.function public extern(\"C\") @wide() {\n    %x = arith.constant 1 : \
                    i65;\n    builtin.ret;\n};",
        verifies: true,
        says: "result 0 of `arith.constant` has type `i65`",
    },
    Case {
        what: "an aggregate",
        functions: "builtin.function public extern(\"C\") @aggregate(i32, struct<i32, i8>);",
        verifies: true,
        says: "the type has no Wasm value type to carry it",
    },
    Case {
        what: "a pointer in another address space",
        functions: "builtin.function public extern(\"C\") @space() -> ptr<i8, 1>;",
        verifies: true,
        says: "only pointers in address space 0, into linear memory, have a Wasm carrier",
    },
    Case {
        what: "a mem.alloca",
        functions:
            "builtin.function public extern(\"C\") @frame() {\n    %p = mem.alloca i32;\n    \
             builtin.ret;\n};",
        verifies: true,
        says: "`convert-to-wasm` does not lower `mem.alloca`: run `wasm-lower-frames` first",
    },
    Case {
        what: "a load that one Wasm access cannot make",
        functions: "builtin.function public extern(\"C\") @load(%p: ptr<i128>) {\n    %v = \
                    mem.load %p;\n    builtin.ret;\n};",
        verifies: true,
        says: "`mem.load` of `i128`: only a value that fits one Wasm load is loaded",
    },
    Case {
        what: "a size that does not fit in 32 bits",
        functions: "builtin.function public extern(\"C\") @size() -> isize {\n    %s = mem.sizeof \
                    array<i8; 4294967296>;\n    builtin.ret %s : (isize);\n};",
        verifies: true,
        says: "`convert-to-wasm` cannot lower this `mem.sizeof`: a Wasm address has 32 bits",
    },
    Case {
        what: "a gep over a pointee whose size does not fit in 32 bits",
        functions: "builtin.function public extern(\"C\") @gep(%p: ptr<array<i8; 4294967296>>, \
                    %i: isize) -> ptr<array<i8; 4294967296>> {\n    %q = mem.gep %p, %i;\n    \
                    builtin.ret %q : (ptr<array<i8; 4294967296>>);\n};",
        verifies: true,
        says: "`convert-to-wasm` cannot lower this `mem.gep`: a Wasm address has 32 bits",
    },
    Case {
        what: "unstructured control flow",
        functions:
            "builtin.function public extern(\"C\") @branch() {\n    cf.br ^next;\n^next:\n    \
             builtin.ret;\n};",
        verifies: true,
        says: "unstructured control flow reached `convert-to-wasm`: run \
               `cfg-to-scf{exits=in-place}` first",
    },
    Case {
        what: "a break that names a frontend wasm.loop",
        functions:
            "builtin.function public extern(\"wasm\") @leave(%c: i1, %a: i32) -> i32 {\n    %r = \
             wasm.loop {\n        scf.if %c then {\n            scf.break 1 %a : (i32);\n        } \
             else {\n            scf.yield : ();\n        } : (i1) -> ();\n        wasm.end %a : \
             (i32);\n    } : () -> (i32);\n    builtin.ret %r : (i32);\n};",
        verifies: false,
        says: "`convert-to-wasm` cannot lower this `scf.break`: it leaves a `wasm.loop` that was \
               written before the lowering",
    },
    Case {
        what: "a break that names no loop",
        functions:
            "builtin.function public extern(\"C\") @none(%c: i1) {\n    scf.if %c then {\n  \
             scf.break 1 : ();\n    } else {\n        scf.yield : ();\n    } : (i1) -> ();\n  \
             builtin.ret;\n};",
        verifies: false,
        says: "`convert-to-wasm` cannot lower this `scf.break`: it names enclosing loop 1, but \
               only 0 enclose it",
    },
    Case {
        what: "a continue that names a frontend wasm.block",
        functions: "builtin.function public extern(\"wasm\") @back(%c: i1) {\n    wasm.block {\n  \
                    scf.if %c then {\n            scf.continue 1 : ();\n        } else {\n        \
                    scf.yield : ();\n        } : (i1) -> ();\n        wasm.end : ();\n    } : () \
                    -> ();\n    builtin.ret;\n};",
        verifies: false,
        says: "`convert-to-wasm` cannot lower this `scf.continue`: it names a `wasm.block`, which \
               has no top to go back to",
    },
    Case {
        what: "an exit of level 0",
        functions: "builtin.function public extern(\"C\") @zero(%n: i32) -> i32 {\n    %w = \
                    scf.while %n before {\n    ^b(%i: i32):\n        scf.break 0 %i : (i32);\n    \
                    } after {\n    ^a(%j: i32):\n        scf.yield %j : (i32);\n    } : (i32) -> \
                    (i32);\n    builtin.ret %w : (i32);\n};",
        verifies: false,
        says: "`convert-to-wasm` cannot lower this `scf.break`: its level 0 is not at least 1",
    },
    Case {
        what: "an scf.yield that ends no scf construct",
        functions: "builtin.function public extern(\"C\") @stray() {\n    wasm.block {\n        \
                    scf.yield : ();\n    } : () -> ();\n    builtin.ret;\n};",
        verifies: false,
        says: "`convert-to-wasm` lowers `scf.yield` only with the `scf.if`, `scf.while` or \
               `scf.index_switch` it ends",
    },
    Case {
        what: "an scf construct with an empty body",
        functions: "builtin.function public extern(\"C\") @empty(%c: i1) {\n    scf.if %c then \
                    {\n  scf.yield : ();\n    } else {\n    } : (i1) -> ();\n    builtin.ret;\n};",
        verifies: false,
        says: "`convert-to-wasm` cannot lower `scf.if` with an empty body",
    },
    Case {
        what: "a frontend branch that does not resolve",
        functions:
            "builtin.function public extern(\"wasm\") @branch() {\n    wasm.block {\n        \
             wasm.br 3 : ();\n    } : () -> ();\n    builtin.ret;\n};",
        verifies: false,
        says: "`convert-to-wasm` cannot lower this `wasm.br`: `wasm.br` names label 3",
    },
    Case {
        what: "an isize constant the target cannot hold",
        functions: "builtin.function public extern(\"C\") @constant() -> isize {\n    %c = \
                    arith.constant 1099511627776 : isize;\n    builtin.ret %c : (isize);\n};",
        verifies: true,
        says: "`convert-to-wasm` cannot lower this `arith.constant`: the target's `isize` has 32 \
               bits, and the `isize` constant 1099511627776 does not fit them",
    },
    Case {
        what: "an isize immediate return the target cannot hold",
        functions: "builtin.function public extern(\"C\") @immediate() -> isize {\n    \
                    builtin.ret_imm -2147483649 : isize;\n};",
        verifies: true,
        says: "`convert-to-wasm` cannot lower this `builtin.ret_imm`: the target's `isize` has 32 \
               bits, and the `isize` constant -2147483649 does not fit them",
    },
    Case {
        what: "a cast to a narrow integer written before the conversion",
        functions: "builtin.function public extern(\"wasm\") @cast(%x: i32) -> i8 {\n    %y = \
                    builtin.unrealized_conversion_cast %x : i32 to i8;\n    builtin.ret %y : \
                    (i8);\n};",
        verifies: true,
        says: "`convert-to-wasm` cannot lower a `builtin.unrealized_conversion_cast` to `i8` \
               written before it",
    },
    Case {
        what: "a well-known name held by a symbol of another kind",
        functions: "builtin.function public extern(\"C\") @__linear_memory();\nbuiltin.function \
                    public extern(\"C\") @load(%p: ptr<i32>) -> i32 {\n    %v = mem.load %p;\n    \
                    builtin.ret %v : (i32);\n};",
        verifies: true,
        says: "`@__linear_memory` is a `builtin.function`, not a `wasm.memory`",
    },
    Case {
        what: "an operation no pattern converts",
        functions: "builtin.function public extern(\"C\") @internal() {\n    \
                    \"scf.pending_exit\"() : () -> ();\n    builtin.ret;\n};",
        verifies: false,
        says: "`convert-to-wasm` cannot lower `scf.pending_exit`: no pattern of the pass converts \
               it",
    },
];

/// The module of `case`, parsed in `context` (verified when the case says it verifies).
#[track_caller]
fn module_of_case(context: &Rc<Context>, case: &Case) -> ModuleRef {
    let text = module_of(&format!("{OK}\n{}", case.functions));
    let parsed = if case.verifies {
        parse_module_in(context, &text)
    } else {
        parse_unverified_in(context, &text)
    };
    parsed.unwrap_or_else(|report| panic!("{}: {}", case.what, describe(&report)))
}

#[test]
fn every_diagnostic_is_raised_before_anything_is_converted() {
    for case in CASES {
        let context = context();
        let module = module_of_case(&context, case);
        let before = text_of(module);
        // The pass manager verifies its input: a module that does not verify is given to the
        // conversion directly.
        let error = if case.verifies {
            let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
            passes.add_pass(Box::<ConvertToWasm>::default()).unwrap();
            error_of(passes.run(module.as_operation_ref()))
        } else {
            error_of(crate::convert::convert_module(module))
        };
        assert!(mentions(&error, case.says), "{}: the error is\n{error}", case.what);
        assert_eq!(text_of(module), before, "{}: the module changed", case.what);
    }
}

#[test]
fn every_case_converts_once_its_diagnostic_is_removed() {
    // The function that would convert does: the cases fail for what they hold, not for `@ok`.
    let (_context, module) = parse_module(&module_of(OK));
    crate::convert::convert_module(module).unwrap_or_else(|report| panic!("{}", describe(&report)));
    crate::verify_wasm_form(module.as_operation_ref())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
}

#[test]
fn a_data_layout_the_lowering_does_not_take_is_reported_before_anything_changes() {
    for (layout, says) in [
        ("#builtin.data_layout<big, ptr = 32>", "needs a little-endian data layout"),
        ("#builtin.data_layout<little, ptr = 64>", "needs 32-bit pointers"),
        ("#builtin.data_layout<little, ptr = 32 index 16>", "needs a 32-bit `isize`"),
    ] {
        let context = context();
        let text = format!(
            "builtin.module public @m attributes {{ data_layout = {layout} }} {{\n{OK}\n}};"
        );
        let module = parse_module_in(&context, &text)
            .unwrap_or_else(|report| panic!("{}", describe(&report)));
        let before = text_of(module);
        let error = error_of(run_pipeline(&context, module.as_operation_ref(), "convert-to-wasm"));
        assert!(error.contains(says), "{layout}: {error}");
        assert_eq!(text_of(module), before, "{layout}");
    }
}

#[test]
fn a_module_without_a_data_layout_is_reported_before_anything_changes() {
    let (context, module) = {
        let context = context();
        let module = parse_module_in(&context, &format!("builtin.module public @m {{\n{OK}\n}};"))
            .unwrap_or_else(|report| panic!("{}", describe(&report)));
        (context, module)
    };
    let before = text_of(module);
    let error = error_of(run_pipeline(&context, module.as_operation_ref(), "convert-to-wasm"));
    assert!(error.contains("data layout"), "{error}");
    assert_eq!(text_of(module), before);
}

/// A world of two modules: `@a`, which converts, and `@b`, whose functions are `functions`.
fn world_of(functions: &str) -> String {
    format!(
        "builtin.world {{\n    builtin.module public @a {{\n{OK}\n    }};\n    builtin.module \
         public @b {{\n{functions}\n    }};\n}} {LAYOUT};"
    )
}

/// Runs `convert-to-wasm` on the world of [world_of] `functions`; the error, and whether the
/// world's text changed.
fn convert_world(functions: &str) -> (String, bool) {
    let text = &world_of(functions);
    let context = context();
    let world = parse::parse::<runic_ir::dialects::builtin::World>(
        ParserConfig::new(context.clone()),
        "w",
        text,
    )
    .into_result()
    .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let before = world.borrow().as_operation().to_string();
    let mut passes =
        PassManager::on::<runic_ir::dialects::builtin::World>(context, Nesting::Implicit);
    passes.add_pass(Box::<ConvertToWasm>::default()).unwrap();
    let error = error_of(passes.run(world.as_operation_ref()));
    (error, world.borrow().as_operation().to_string() != before)
}

#[test]
fn a_failure_in_one_module_of_a_world_is_found_before_any_module_changes() {
    // `@a` comes first and would convert; `@b` holds an alloca.
    let (error, changed) = convert_world(
        "builtin.function public extern(\"C\") @frame() {\n    %p = mem.alloca i32;\n    \
         builtin.ret;\n};",
    );
    assert!(error.contains("does not lower `mem.alloca`"), "{error}");
    assert!(!changed, "a module of the world changed");
}

#[test]
fn a_function_named_from_another_module_is_found_before_any_module_changes() {
    // `@b::@g` calls `@a::@ok`, whose signature changes.
    let (error, changed) = convert_world(
        "builtin.function public extern(\"wasm\") @g(%x: i8) -> i8 {\n    %y = builtin.call \
         ::@a::@ok(%x, %x) : extern(\"wasm\") (i8, i8) -> i8;\n    builtin.ret %y : (i8);\n};",
    );
    assert!(
        error.contains(
            "`convert-to-wasm` changes the signature of `@ok`, but a `builtin.call` in `@g`, \
             outside the module `@a` that is converted, names it"
        ),
        "{error}"
    );
    assert!(!changed, "a module of the world changed");
}

#[test]
fn the_passes_report_a_root_that_holds_no_module() {
    // A bare function: each pass runs on modules, and says so.
    let text = "builtin.function public extern(\"wasm\") @f(%a: i8) -> i8 {\n    builtin.ret %a : \
                (i8);\n};";
    for pass in ["convert-to-wasm", "wasm-lower-frames", "wasm-verify-form"] {
        let context = context();
        let function = parse::parse::<runic_ir::dialects::builtin::Function>(
            ParserConfig::new(context.clone()),
            "f",
            text,
        )
        .into_result()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
        let error = error_of(run_pipeline(&context, function.as_operation_ref(), pass));
        assert!(
            error.contains(&format!(
                "`{pass}` runs on a `builtin.module`, or on a world or a component that holds \
                 one, but this `builtin.function` holds none"
            )),
            "{pass}: {error}"
        );
    }
}
