//! The lowering of structured control flow: what the conversion relies on, and differential
//! tests.
//!
//! The differential tests are small programs in `scf` form whose results for several inputs are
//! stated by hand (there is no `scf` interpreter in the tree): each is lowered by
//! `convert-to-wasm`, checked by `wasm-verify-form`, and run by the evaluator, whose fuel turns an
//! endless loop into a failure.

use alloc::{vec, vec::Vec};
use core::cell::RefCell;

use runic_ir::{
    OperationRef, Report,
    conversion::{
        ConversionConfig, ConversionPattern, ConversionPatternRewriter, ConversionPatternSet,
        ConvertedOperands, TypeConverter, apply_full_conversion,
    },
    patterns::{Pattern, PatternInfo},
};

use super::{
    eval::{Outcome, Val, run},
    *,
};
use crate::{
    WasmTarget,
    convert::{
        WellKnownSymbols, calls::pattern_info, conversion_target, convert_module, populate_patterns,
    },
    verify_wasm_form,
};

/// A module of the functions `functions` under the Wasm data layout.
fn module_text(functions: &str) -> String {
    format!(
        "builtin.module public @m attributes {{ data_layout = #builtin.data_layout<little, ptr = \
         32> }} {{\n{functions}\n}};"
    )
}

/// `functions` parsed, lowered by `convert-to-wasm` and checked by `wasm-verify-form`; the module
/// and its function `@f`.
#[track_caller]
fn lowered(functions: &str) -> (Rc<Context>, ModuleRef, OperationRef) {
    let (context, module) = parse_module(&module_text(functions));
    convert_module(module).unwrap_or_else(|report| panic!("{}", describe(&report)));
    module
        .borrow()
        .as_operation()
        .recursively_verify()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    verify_wasm_form(module.as_operation_ref())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let function = runic_ir::SymbolTable::symbol_manager(&*module.borrow())
        .lookup_op("f")
        .expect("the function @f");
    (context, module, function)
}

/// Runs the lowered `@f` of `functions` on each input and compares with the expected outcome.
#[track_caller]
fn check(functions: &str, cases: &[(Vec<Val>, Outcome)]) {
    let (_context, module, function) = lowered(functions);
    for (inputs, expected) in cases {
        let outcome = run(function, inputs);
        assert_eq!(&outcome, expected, "inputs {inputs:?}\n{}", text_of(module));
    }
}

fn one(value: Val) -> Outcome {
    Outcome::Values(vec![value])
}

fn i32s(values: &[u32]) -> Vec<Val> {
    values.iter().map(|value| Val::i32(*value)).collect()
}

// ---------------------------------------------------------------------------------------------
// What the lowering relies on

#[test]
fn returns_and_tail_calls_nested_in_wasm_constructs_pass_the_function_verifier() {
    // The constructs forward exits, so the function verifier accepts a return or a tail call at
    // any depth in them.
    parse_module(
        &module_text(
            "builtin.function public extern(\"C\") @f(%c: i32, %x: i32) -> i32 {\n    %r = \
             wasm.block {\n        %l = wasm.loop %x {\n        ^b(%i: i32):\n            wasm.if \
             %c then {\n    builtin.ret %i : (i32);\n            } else {\n                \
             builtin.tail_call @g(%i) : extern(\"C\") (i32) -> i32;\n            } : (i32) -> \
             ();\n   wasm.end %i : (i32);\n        } : (i32) -> (i32);\n        wasm.end %l : \
             (i32);\n    } : () -> (i32);\n    builtin.ret %r : (i32);\n};\nbuiltin.function \
             public extern(\"C\") @g(i32) -> i32;",
        ),
    );
}

/// A pattern that records the operations it is offered, and whether an `scf` construct encloses
/// each then, and changes nothing.
struct Recorder {
    info: PatternInfo,
    seen: Rc<RefCell<Vec<(String, bool)>>>,
}

impl Pattern for Recorder {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for Recorder {
    fn type_converter(&self) -> Option<&TypeConverter> {
        None
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        _rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let mut enclosed = false;
        let mut parent = op.borrow().parent_op();
        while let Some(ancestor) = parent {
            let ancestor = ancestor.borrow();
            enclosed |= format!("{}", ancestor.name()).starts_with("scf.");
            parent = ancestor.parent_op();
        }
        let name = format!("{}", op.borrow().name());
        self.seen.borrow_mut().push((name, enclosed));
        Ok(false)
    }
}

#[test]
fn the_conversion_offers_an_operation_before_those_nested_in_it_moved_ones_included() {
    // An `scf.break` in an `scf.if` in the body of an `scf.while`, in an `scf.if`. The recorders
    // have no generated operation, so each is tried before the pattern of its operation.
    let (context, module) = parse_module(&module_text(
        "builtin.function public extern(\"wasm\") @f(%c: i1, %x: i32) -> i32 {\n    %r = scf.if \
         %c then {\n        %w = scf.while %x before {\n        ^b(%i: i32):\n            \
         scf.condition %c, %i : (i1, i32);\n        } after {\n        ^a(%j: i32):\n            \
         scf.if %c then {\n                scf.break 1 %j : (i32);\n            } else {\n        \
         scf.yield : ();\n            } : (i1) -> ();\n            scf.yield %j : (i32);\n        \
         } : (i32) -> (i32);\n        scf.yield %w : (i32);\n    } else {\n        scf.yield %x : \
         (i32);\n    } : (i1) -> (i32);\n    builtin.ret %r : (i32);\n};",
    ));
    let module_op = module.as_operation_ref();
    let target = WasmTarget::of(&module_op.borrow()).unwrap();
    let symbols = WellKnownSymbols::of(module).unwrap();
    let converter = crate::types::type_converter();
    let seen = Rc::new(RefCell::new(Vec::new()));
    let mut patterns = ConversionPatternSet::new(context.clone());
    let names = [
        pattern_info::<runic_dialect_scf::If>(&context, "record-if", []),
        pattern_info::<runic_dialect_scf::While>(&context, "record-while", []),
        pattern_info::<runic_dialect_scf::Break>(&context, "record-break", []),
    ];
    for info in names {
        patterns.push(Recorder {
            info,
            seen: seen.clone(),
        });
    }
    populate_patterns(&mut patterns, &converter, &target, &symbols).unwrap();
    apply_full_conversion(
        module_op,
        conversion_target(&context, &converter),
        patterns,
        ConversionConfig::default(),
    )
    .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let seen = seen.borrow();
    let names = seen.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>();
    // The outer `scf.if`, then the `scf.while` moved into its `wasm.if`, then the inner `scf.if`
    // moved into the loop, then the break: each when nothing of `scf` encloses it any more.
    assert_eq!(names, ["scf.if", "scf.while", "scf.if", "scf.break"], "{seen:?}");
    assert!(seen.iter().all(|(_, enclosed)| !enclosed), "{seen:?}");
}

// ---------------------------------------------------------------------------------------------
// Differential tests

#[test]
fn a_counting_loop_sums_its_counter() {
    // The sum of 0..n, with an `i8` counter and an `i32` sum.
    let functions =
        "builtin.function public extern(\"wasm\") @f(%n: i8) -> i32 {\n    %zero8 = \
         arith.constant 0 : i8;\n    %one8 = arith.constant 1 : i8;\n    %zero = arith.constant 0 \
         : i32;\n    %i, %s = scf.while %zero8, %zero before {\n    ^b(%j: i8, %t: i32):\n        \
         %more = arith.cmpi ult %j, %n;\n        scf.condition %more, %j, %t : (i1, i8, i32);\n    \
         } after {\n    ^a(%k: i8, %u: i32):\n        %w = arith.zext %k <{ ty = i32 }>;\n        \
         %v = arith.add %u, %w <{ overflow = #builtin.overflow<wrapping> }>;\n        %l = \
         arith.add %k, %one8 <{ overflow = #builtin.overflow<wrapping> }>;\n        scf.yield %l, \
         %v : (i8, i32);\n    } : (i8, i32) -> (i8, i32);\n    builtin.ret %s : (i32);\n};";
    check(
        functions,
        &[
            (i32s(&[0]), one(Val::i32(0))),
            (i32s(&[1]), one(Val::i32(0))),
            (i32s(&[5]), one(Val::i32(10))),
            (i32s(&[200]), one(Val::i32(19_900))),
            (i32s(&[255]), one(Val::i32(32_385))),
        ],
    );
}

#[test]
fn a_loop_with_a_break_leaves_with_the_break_s_value() {
    // The first i in 0..n with i * i >= k, or 1000 + n when there is none.
    let functions = "builtin.function public extern(\"wasm\") @f(%n: i32, %k: i32) -> i32 {\n    \
                     %zero = arith.constant 0 : i32;\n    %one = arith.constant 1 : i32;\n    \
                     %r = scf.while %zero before {\n    ^b(%i: i32):\n        %more = arith.cmpi \
                     ult %i, %n;\n        scf.condition %more, %i : (i1, i32);\n    } after {\n  \
                       ^a(%j: i32):\n        %sq = arith.mul %j, %j <{ overflow = \
                     #builtin.overflow<wrapping> }>;\n        %hit = arith.cmpi uge %sq, %k;\n   \
                          scf.if %hit then {\n            scf.break 1 %j : (i32);\n        } \
                     else {\n            scf.yield : ();\n        } : (i1) -> ();\n        %next \
                     = arith.add %j, %one <{ overflow = #builtin.overflow<wrapping> }>;\n        \
                     scf.yield %next : (i32);\n    } : (i32) -> (i32);\n    %done = arith.cmpi \
                     eq %r, %n;\n    %base = arith.constant 1000 : i32;\n    %miss = arith.add \
                     %base, %n <{ overflow = #builtin.overflow<wrapping> }>;\n    %out = \
                     cf.select %done, %miss, %r;\n    builtin.ret %out : (i32);\n};";
    check(
        functions,
        &[
            (i32s(&[10, 0]), one(Val::i32(0))),
            (i32s(&[10, 10]), one(Val::i32(4))),
            (i32s(&[10, 49]), one(Val::i32(7))),
            (i32s(&[10, 82]), one(Val::i32(1010))),
            (i32s(&[0, 5]), one(Val::i32(1000))),
        ],
    );
}

#[test]
fn nested_loops_with_a_continue_of_the_outer_one_skip_the_rest_of_the_inner_one() {
    // For each i in 0..n, the inner loop counts j from 0 and continues the outer loop when
    // j == i: the count is the sum of 0..n, n(n-1)/2.
    let functions = "builtin.function public extern(\"wasm\") @f(%n: i32) -> i32 {\n    %zero = \
                     arith.constant 0 : i32;\n    %one = arith.constant 1 : i32;\n    %i, %acc = \
                     scf.while %zero, %zero before {\n    ^ob(%oi: i32, %oa: i32):\n        \
                     %more = arith.cmpi ult %oi, %n;\n        scf.condition %more, %oi, %oa : \
                     (i1, i32, i32);\n    } after {\n    ^oa(%ai: i32, %aa: i32):\n        %j, \
                     %c = scf.while %zero, %aa before {\n        ^ib(%ij: i32, %ic: i32):\n      \
                           %imore = arith.cmpi ult %ij, %n;\n            scf.condition %imore, \
                     %ij, %ic : (i1, i32, i32);\n        } after {\n        ^ia(%jj: i32, %jc: \
                     i32):\n            %same = arith.cmpi eq %jj, %ai;\n            scf.if \
                     %same then {\n                %ni = arith.add %ai, %one <{ overflow = \
                     #builtin.overflow<wrapping> }>;\n                scf.continue 2 %ni, %jc : \
                     (i32, i32);\n            } else {\n                scf.yield : ();\n        \
                         } : (i1) -> ();\n            %nc = arith.add %jc, %one <{ overflow = \
                     #builtin.overflow<wrapping> }>;\n            %nj = arith.add %jj, %one <{ \
                     overflow = #builtin.overflow<wrapping> }>;\n            scf.yield %nj, %nc \
                     : (i32, i32);\n        } : (i32, i32) -> (i32, i32);\n        %next = \
                     arith.add %ai, %one <{ overflow = #builtin.overflow<wrapping> }>;\n        \
                     scf.yield %next, %c : (i32, i32);\n    } : (i32, i32) -> (i32, i32);\n    \
                     builtin.ret %acc : (i32);\n};";
    check(
        functions,
        &[
            (i32s(&[0]), one(Val::i32(0))),
            (i32s(&[1]), one(Val::i32(0))),
            (i32s(&[4]), one(Val::i32(6))),
            (i32s(&[10]), one(Val::i32(45))),
        ],
    );
}

/// A function `@f(%s: i32) -> i8` of an `scf.index_switch` on `%s` whose case `k` gives `v` for
/// each `(k, v)` of `cases`, and whose default gives 99.
fn switch_text(cases: &[(i32, u8)]) -> String {
    let mut text = String::from(
        "builtin.function public extern(\"wasm\") @f(%s: i32) -> i8 {\n    %r = scf.index_switch \
         %s\n",
    );
    for (index, (case, value)) in cases.iter().enumerate() {
        text += &format!(
            "    case {case} {{\n        %c{index} = arith.constant {value} : i8;\n        \
             scf.yield %c{index} : (i8);\n    }}\n"
        );
    }
    text += "    default {\n        %d = arith.constant 99 : i8;\n        scf.yield %d : (i8);\n  \
             } : (i8);\n    builtin.ret %r : (i8);\n};";
    text
}

/// The outcomes of a switch of `cases` (default 99) on each of `selectors`.
fn switch_cases(cases: &[(i32, u8)], selectors: &[i32]) -> Vec<(Vec<Val>, Outcome)> {
    selectors
        .iter()
        .map(|selector| {
            let value =
                cases.iter().find(|(case, _)| case == selector).map_or(99, |(_, value)| *value);
            (vec![Val::i32(*selector as u32)], one(Val::i32(u32::from(value))))
        })
        .collect()
}

#[test]
fn a_dense_switch_takes_the_case_of_its_selector_and_the_default_otherwise() {
    // From 0, with a gap at 3; then from -2, with a gap at 0.
    let cases = [(0, 10), (1, 11), (2, 12), (4, 14)];
    let text = switch_text(&cases);
    assert!(text_of(lowered(&text).1).contains("wasm.br_table"));
    check(&text, &switch_cases(&cases, &[-1, 0, 1, 2, 3, 4, 5, i32::MAX, i32::MIN]));
    let cases = [(-2, 20), (-1, 21), (1, 23), (2, 24)];
    let text = switch_text(&cases);
    check(&text, &switch_cases(&cases, &[-3, -2, -1, 0, 1, 2, 3, i32::MIN]));
}

#[test]
fn a_sparse_switch_compares_its_selector_with_each_case() {
    let cases = [(-7, 1), (100, 2), (5000, 3)];
    let text = switch_text(&cases);
    assert!(!text_of(lowered(&text).1).contains("wasm.br_table"));
    check(&text, &switch_cases(&cases, &[-7, 0, 100, 101, 5000, -1]));
    // Two cases are a chain however close they are.
    let cases = [(0, 5), (1, 6)];
    check(&switch_text(&cases), &switch_cases(&cases, &[0, 1, 2, -1]));
}

#[test]
fn an_if_with_results_gives_those_of_the_branch_it_takes() {
    // (x + 1, y * 2) when c, (x, y) otherwise, with an `i8` x (wrapping at 256).
    let functions = "builtin.function public extern(\"wasm\") @f(%c: i1, %x: i8, %y: i64) -> \
                     (i8, i64) {\n    %a, %b = scf.if %c then {\n        %one = arith.constant \
                     1 : i8;\n        %s = arith.add %x, %one <{ overflow = \
                     #builtin.overflow<wrapping> }>;\n        %two = arith.constant 2 : i64;\n    \
                         %p = arith.mul %y, %two <{ overflow = #builtin.overflow<wrapping> \
                     }>;\n        scf.yield %s, %p : (i8, i64);\n    } else {\n        scf.yield \
                     %x, %y : (i8, i64);\n    } : (i1) -> (i8, i64);\n    builtin.ret %a, %b : \
                     (i8, i64);\n};";
    let case = |c: u32, x: u32, y: u64, a: u32, b: u64| {
        (
            vec![Val::i32(c), Val::i32(x), Val::i64(y)],
            Outcome::Values(vec![Val::i32(a), Val::i64(b)]),
        )
    };
    check(
        functions,
        &[case(1, 5, 21, 6, 42), case(0, 5, 21, 5, 21), case(1, 255, 1 << 62, 0, 1 << 63)],
    );
}

#[test]
fn a_break_from_a_case_of_a_switch_in_a_loop_leaves_the_loop() {
    // Counts i up from 0; at i == k the case of the switch on i - k breaks out with 100 + i.
    let functions = "builtin.function public extern(\"wasm\") @f(%n: i32, %k: i32) -> i32 {\n    \
                     %zero = arith.constant 0 : i32;\n    %one = arith.constant 1 : i32;\n    %r \
                     = scf.while %zero before {\n    ^b(%i: i32):\n        %more = arith.cmpi ult \
                     %i, %n;\n        scf.condition %more, %i : (i1, i32);\n    } after {\n  \
                     ^a(%j: i32):\n        %d = arith.sub %j, %k <{ overflow = \
                     #builtin.overflow<wrapping> }>;\n        scf.index_switch %d\n        case 0 \
                     {\n            %h = arith.constant 100 : i32;\n            %v = arith.add \
                     %h, %j <{ overflow = #builtin.overflow<wrapping> }>;\n            scf.break \
                     1 %v : (i32);\n        }\n        default {\n            scf.yield : ();\n  \
                     } : ();\n        %next = arith.add %j, %one <{ overflow = \
                     #builtin.overflow<wrapping> }>;\n        scf.yield %next : (i32);\n    } : \
                     (i32) -> (i32);\n    builtin.ret %r : (i32);\n};";
    check(
        functions,
        &[
            (i32s(&[10, 3]), one(Val::i32(103))),
            (i32s(&[10, 0]), one(Val::i32(100))),
            (i32s(&[10, 20]), one(Val::i32(10))),
        ],
    );
}

#[test]
fn a_loop_with_a_break_and_a_continue_goes_back_to_the_loop_inside_its_block() {
    // The sum of the even i in 0..n, left early with the sum when it passes `limit`: the odd ones
    // continue, and the break needs the block around the loop.
    let functions = "builtin.function public extern(\"wasm\") @f(%n: i32, %limit: i32) -> i32 {\n \
                        %zero = arith.constant 0 : i32;\n    %one = arith.constant 1 : i32;\n    \
                     %ri, %rs = scf.while %zero, %zero before {\n    ^b(%i: i32, %s: i32):\n     \
                        %more = arith.cmpi ult %i, %n;\n        scf.condition %more, %i, %s : \
                     (i1, i32, i32);\n    } after {\n    ^a(%j: i32, %t: i32):\n        %next = \
                     arith.add %j, %one <{ overflow = #builtin.overflow<wrapping> }>;\n        \
                     %low = arith.band %j, %one;\n        %odd = arith.cmpi eq %low, %one;\n      \
                       scf.if %odd then {\n            scf.continue 1 %next, %t : (i32, \
                     i32);\n        } else {\n            scf.yield : ();\n        } : (i1) -> \
                     ();\n        %u = arith.add %t, %j <{ overflow = #builtin.overflow<wrapping> \
                     }>;\n        %over = arith.cmpi ugt %u, %limit;\n        scf.if %over then \
                     {\n            scf.break 1 %j, %u : (i32, i32);\n        } else {\n         \
                        scf.yield : ();\n        } : (i1) -> ();\n        scf.yield %next, %u : \
                     (i32, i32);\n    } : (i32, i32) -> (i32, i32);\n    builtin.ret %rs : \
                     (i32);\n};";
    check(
        functions,
        &[
            (i32s(&[10, 1000]), one(Val::i32(20))),
            (i32s(&[10, 5]), one(Val::i32(6))),
            (i32s(&[0, 5]), one(Val::i32(0))),
        ],
    );
}
