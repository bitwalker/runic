//! Programs of structured control, frames, memory and calls, lowered and evaluated.
//!
//! Each program is lowered by a pipeline (`convert-to-wasm` alone for programs in `scf` form, the
//! four passes for those in CFG form, and the optimization pipeline in front of them) and run by
//! the evaluator on inputs whose results are stated by hand or computed by a model in Rust. A
//! program with frames also checks that the stack pointer is restored, and that every value it is
//! set to is 16-byte aligned and below where it started.

use alloc::{vec, vec::Vec};

use super::{
    eval::{Machine, Outcome, Val, run_in},
    *,
};

/// `convert-to-wasm` and the check of Wasm form, for programs in `scf` form.
const CONV: &str = "convert-to-wasm,wasm-verify-form";

/// The four passes of the lowering.
const FULL: &str = LOWERING;

/// The default optimization pipeline of a compiler, then the four passes.
const OPT: &str = "canonicalizer,cse,sccp,canonicalizer,dce,cfg-to-scf{exits=in-place},\
                   wasm-lower-frames,convert-to-wasm,wasm-verify-form";

/// The passes of the lowering after `cfg-to-scf`, for programs in `scf` form with frames.
const FR: &str = "wasm-lower-frames,convert-to-wasm,wasm-verify-form";

/// The stack pointer a run starts with, at the end of the machine's memory.
const STACK_TOP: u32 = 1024;

fn i32s(values: &[u32]) -> Vec<Val> {
    values.iter().map(|value| Val::i32(*value)).collect()
}

/// A machine of [STACK_TOP] bytes whose stack pointer is at its end.
fn machine() -> Machine {
    let mut machine = Machine::new(STACK_TOP as usize);
    machine.globals.insert(String::from("__stack_pointer"), Val::i32(STACK_TOP));
    machine
}

/// The text of the module of `functions` lowered by `pipeline`, or the error, described.
fn lowers(functions: &str, pipeline: &str) -> Result<String, String> {
    let (context, module) = parse_module(&module_of(functions));
    run_pipeline(&context, module.as_operation_ref(), pipeline)
        .map(|()| text_of(module))
        .map_err(|report| describe(&report))
}

/// Lowers `functions` by `pipeline`, then checks that `@name` of each case's inputs gives its
/// `i32`s, or traps for `None`.
#[track_caller]
fn check(functions: &str, pipeline: &str, name: &str, cases: &[(&[u32], Option<&[u32]>)]) {
    let (_context, module) = lowered_by(&module_of(functions), pipeline);
    let function = function_named(module, name);
    for (inputs, expected) in cases {
        let outcome = run_in(&mut machine(), function, &i32s(inputs));
        let expected = expected.map_or(Outcome::Trap, |values| Outcome::Values(i32s(values)));
        assert_eq!(outcome, expected, "@{name}{inputs:?}\n{}", text_of(module));
    }
}

// A: a dense switch inside a loop: break 1 from an scf.if in case 1, continue 1 from case 2.
const SW_IN_LOOP: &str = r#"
builtin.function public extern("wasm") @f(%n: i32, %k: i32) -> (i32, i32) {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    %three = arith.constant 3 : i32;
    %ri, %racc = scf.while %zero, %zero before {
    ^b(%i: i32, %acc: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i, %acc : (i1, i32, i32);
    } after {
    ^a(%i2: i32, %acc2: i32):
        %sel = arith.band %i2, %three;
        %na = scf.index_switch %sel
        case 0 {
            %x = arith.add %acc2, %one <{ overflow = #builtin.overflow<wrapping> }>;
            scf.yield %x : (i32);
        }
        case 1 {
            %hit = arith.cmpi eq %i2, %k;
            %two = arith.constant 2 : i32;
            %y = arith.add %acc2, %two <{ overflow = #builtin.overflow<wrapping> }>;
            %z = scf.if %hit then {
                scf.break 1 %i2, %acc2 : (i32, i32);
            } else {
                scf.yield %y : (i32);
            } : (i1) -> (i32);
            scf.yield %z : (i32);
        }
        case 2 {
            %h = arith.constant 100 : i32;
            %w = arith.add %acc2, %h <{ overflow = #builtin.overflow<wrapping> }>;
            %ni = arith.add %i2, %one <{ overflow = #builtin.overflow<wrapping> }>;
            scf.continue 1 %ni, %w : (i32, i32);
        }
        default {
            %t = arith.constant 10 : i32;
            %v = arith.add %acc2, %t <{ overflow = #builtin.overflow<wrapping> }>;
            scf.yield %v : (i32);
        } : (i32);
        %nexti = arith.add %i2, %one <{ overflow = #builtin.overflow<wrapping> }>;
        scf.yield %nexti, %na : (i32, i32);
    } : (i32, i32) -> (i32, i32);
    builtin.ret %ri, %racc : (i32, i32);
};
"#;

#[test]
fn a_switch_in_a_loop_breaks_from_a_case_and_continues_from_another() {
    check(
        SW_IN_LOOP,
        CONV,
        "f",
        &[
            (&[6, 5], Some(&[5, 114])),
            (&[6, 9], Some(&[6, 116])),
            (&[0, 0], Some(&[0, 0])),
            (&[3, 1], Some(&[1, 1])),
        ],
    );
}

// B: continue 2 from inner loop's `before` region (an scf.if yielding the condition), narrow acc.
const CONT_FROM_BEFORE: &str = r#"
builtin.function public extern("wasm") @f(%n: i32) -> i8 {
    %zero = arith.constant 0 : i32;
    %z8 = arith.constant 0 : i8;
    %one = arith.constant 1 : i32;
    %o8 = arith.constant 1 : i8;
    %ri, %racc = scf.while %zero, %z8 before {
    ^b(%i: i32, %acc: i8):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i, %acc : (i1, i32, i8);
    } after {
    ^a(%i2: i32, %acc2: i8):
        %jr, %ar = scf.while %zero, %acc2 before {
        ^ib(%j: i32, %aa: i8):
            %ge = arith.cmpi uge %j, %i2;
            %c = scf.if %ge then {
                %i1 = arith.add %i2, %one <{ overflow = #builtin.overflow<wrapping> }>;
                scf.continue 2 %i1, %aa : (i32, i8);
            } else {
                %hundred = arith.constant 100 : i32;
                %t = arith.cmpi ult %j, %hundred;
                scf.yield %t : (i1);
            } : (i1) -> (i1);
            scf.condition %c, %j, %aa : (i1, i32, i8);
        } after {
        ^ia(%jj: i32, %ab: i8):
            %ab2 = arith.add %ab, %o8 <{ overflow = #builtin.overflow<wrapping> }>;
            %jj2 = arith.add %jj, %one <{ overflow = #builtin.overflow<wrapping> }>;
            scf.yield %jj2, %ab2 : (i32, i8);
        } : (i32, i8) -> (i32, i8);
        %ni = arith.add %i2, %one <{ overflow = #builtin.overflow<wrapping> }>;
        scf.yield %ni, %ar : (i32, i8);
    } : (i32, i8) -> (i32, i8);
    builtin.ret %racc : (i8);
};
"#;

#[test]
fn a_continue_of_the_outer_loop_from_the_before_body_of_the_inner_one_carries_a_narrow_value() {
    check(
        CONT_FROM_BEFORE,
        CONV,
        "f",
        &[(&[5], Some(&[10])), (&[0], Some(&[0])), (&[30], Some(&[179]))],
    );
}

// C: break 2 (outer loop) from inside a dense switch inside an inner loop; continue 2 from the
// inner `after` body.
const BREAK_OUTER_FROM_SWITCH: &str = r#"
builtin.function public extern("wasm") @f(%n: i32, %k: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    %r = scf.while %zero before {
    ^b(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^a(%i2: i32):
        %j = scf.while %zero before {
        ^ib(%jj: i32):
            %m = arith.cmpi ult %jj, %n;
            scf.condition %m, %jj : (i1, i32);
        } after {
        ^ia(%jk: i32):
            %s = arith.sub %jk, %i2 <{ overflow = #builtin.overflow<wrapping> }>;
            scf.index_switch %s
            case 0 {
                %hit = arith.cmpi eq %i2, %k;
                scf.if %hit then {
                    %big = arith.constant 1000 : i32;
                    %v = arith.add %i2, %big <{ overflow = #builtin.overflow<wrapping> }>;
                    scf.break 2 %v : (i32);
                } else {
                    scf.yield : ();
                } : (i1) -> ();
                scf.yield : ();
            }
            case 1 {
                %ni = arith.add %i2, %one <{ overflow = #builtin.overflow<wrapping> }>;
                scf.continue 2 %ni : (i32);
            }
            case 2 {
                scf.yield : ();
            }
            default {
                scf.yield : ();
            } : ();
            %nj = arith.add %jk, %one <{ overflow = #builtin.overflow<wrapping> }>;
            scf.yield %nj : (i32);
        } : (i32) -> (i32);
        %ni2 = arith.add %i2, %one <{ overflow = #builtin.overflow<wrapping> }>;
        scf.yield %ni2 : (i32);
    } : (i32) -> (i32);
    builtin.ret %r : (i32);
};
"#;

#[test]
fn a_break_of_the_outer_loop_from_a_dense_switch_in_the_inner_one_leaves_both() {
    // For each i, the inner loop reaches j == i (case 0) then j == i + 1 (case 1, continue
    // outer with i + 1). When i == k at case 0, break 2 with k + 1000. When i == n - 1, j == i
    // + 1 == n is never reached: the inner loop ends normally and the outer increments.
    check(
        BREAK_OUTER_FROM_SWITCH,
        CONV,
        "f",
        &[(&[5, 3], Some(&[1003])), (&[5, 9], Some(&[5])), (&[0, 0], Some(&[0]))],
    );
}

// D: switch thresholds and extremes.
fn switch_fn(cases: &[i32]) -> String {
    let mut s = String::from(
        "builtin.function public extern(\"wasm\") @f(%s: i32) -> i32 {\n    %r = scf.index_switch \
         %s\n",
    );
    for (j, c) in cases.iter().enumerate() {
        s.push_str(&format!(
            "    case {c} {{\n        %x{j} = arith.constant {} : i32;\n        scf.yield %x{j} : \
             (i32);\n    }}\n",
            j + 1
        ));
    }
    s.push_str(
        "    default {\n        %d = arith.constant 0 : i32;\n        scf.yield %d : (i32);\n    \
         } : (i32);\n    builtin.ret %r : (i32);\n};\n",
    );
    s
}

fn check_switch(cases: &[i32], probes: &[i32]) -> String {
    let body = switch_fn(cases);
    let mut table = Vec::new();
    for p in probes {
        let want = cases.iter().position(|c| c == p).map_or(0, |j| j as u32 + 1);
        table.push(([*p as u32], [want]));
    }
    let cases_ref: Vec<(&[u32], Option<&[u32]>)> =
        table.iter().map(|(a, w)| (&a[..], Some(&w[..]))).collect();
    check(&body, CONV, "f", &cases_ref);
    lowers(&body, CONV).unwrap()
}

#[test]
fn switches_at_the_extremes_of_i32_and_at_the_density_threshold_take_the_right_case() {
    let max = i32::MAX;
    let min = i32::MIN;
    let t = check_switch(&[max - 2, max - 1, max], &[max, max - 1, max - 2, max - 3, min, 0, -1]);
    assert!(t.contains("br_table"), "{t}");
    let t = check_switch(&[min, min + 1, min + 2], &[min, min + 1, min + 2, min + 3, max, 0, -1]);
    assert!(t.contains("br_table"), "{t}");
    let t = check_switch(&[-1, 1, 0], &[-1, 0, 1, 2, -2, min, max]);
    assert!(t.contains("br_table"), "{t}");
    // Boundary: 3 cases over 12 values is dense, 13 is not.
    let t = check_switch(&[0, 5, 11], &[0, 5, 11, 1, 12, -1, 4]);
    assert!(t.contains("br_table"), "{t}");
    let t = check_switch(&[0, 5, 12], &[0, 5, 12, 1, 11, 13, -1]);
    assert!(!t.contains("br_table"), "{t}");
    // Unordered cases and a sparse chain with extremes.
    let t = check_switch(&[min, 0, max], &[min, 0, max, 1, -1]);
    assert!(!t.contains("br_table"), "{t}");
    check_switch(&[7, 3, 5, 4], &[3, 4, 5, 6, 7, 8, 2]);
    check_switch(&[42], &[42, 41, 0]);
    check_switch(&[], &[42, 0]);
}

// E: a dense switch with results whose cases end in different exits, in a loop.
const SWITCH_EXITS: &str = r#"
builtin.function public extern("wasm") @g(%x: i32) -> i32 {
    %c = arith.constant 7 : i32;
    %r = arith.mul %x, %c <{ overflow = #builtin.overflow<wrapping> }>;
    builtin.ret %r : (i32);
};
builtin.function public extern("wasm") @f(%s: i32, %x: i32) -> i32 {
    %r = scf.index_switch %s
    case 0 {
        builtin.ret %x : (i32);
    }
    case 1 {
        builtin.tail_call @g(%x) : extern("wasm") (i32) -> i32;
    }
    case 2 {
        ub.unreachable;
    }
    case 3 {
        %y = arith.constant 3 : i32;
        scf.yield %y : (i32);
    }
    default {
        %z = arith.constant 99 : i32;
        scf.yield %z : (i32);
    } : (i32);
    %one = arith.constant 1 : i32;
    %q = arith.add %r, %one <{ overflow = #builtin.overflow<wrapping> }>;
    builtin.ret %q : (i32);
};
"#;

#[test]
fn the_cases_of_a_switch_may_end_in_a_return_a_tail_call_a_trap_or_a_yield() {
    check(
        SWITCH_EXITS,
        CONV,
        "f",
        &[
            (&[0, 5], Some(&[5])),
            (&[1, 5], Some(&[35])),
            (&[2, 5], None),
            (&[3, 5], Some(&[4])),
            (&[4, 5], Some(&[100])),
            (&[u32::MAX, 5], Some(&[100])),
        ],
    );
}

// F: scf.if with results where one arm breaks, one continues, in a while whose after ends in ret
// on some path; and an `after` that ends in an exit.
const IF_ARMS: &str = r#"
builtin.function public extern("wasm") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    %five = arith.constant 5 : i32;
    %r = scf.while %zero before {
    ^b(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^a(%j: i32):
        %big = arith.cmpi ugt %j, %five;
        %v = scf.if %big then {
            %m = arith.constant 50 : i32;
            %w = arith.add %j, %m <{ overflow = #builtin.overflow<wrapping> }>;
            scf.break 1 %w : (i32);
        } else {
            %low = arith.band %j, %one;
            %odd = arith.cmpi eq %low, %one;
            %u = scf.if %odd then {
                %two = arith.constant 2 : i32;
                %jn = arith.add %j, %two <{ overflow = #builtin.overflow<wrapping> }>;
                scf.continue 1 %jn : (i32);
            } else {
                scf.yield %j : (i32);
            } : (i1) -> (i32);
            scf.yield %u : (i32);
        } : (i1) -> (i32);
        %next = arith.add %v, %one <{ overflow = #builtin.overflow<wrapping> }>;
        scf.yield %next : (i32);
    } : (i32) -> (i32);
    builtin.ret %r : (i32);
};
"#;

#[test]
fn the_arms_of_an_if_with_results_may_break_or_continue() {
    // i: 0 -> 1 -> (odd) 3 -> (odd) 5 -> (odd) 7 -> big: 57.
    check(
        IF_ARMS,
        CONV,
        "f",
        &[(&[100], Some(&[57])), (&[4], Some(&[5])), (&[0], Some(&[0]))],
    );
}

// G: a zero-carried-value loop that counts in a local, with a break.
const ZERO_CARRIED: &str = r#"
builtin.function public extern("wasm") @f(%n: i32) -> i32 locals(i8) {
    %z = arith.constant 0 : i8;
    builtin.store_local 0, %z;
    scf.while before {
        %v = builtin.load_local 0 : i8;
        %n8 = arith.trunc %n <{ ty = i8 }>;
        %more = arith.cmpi ult %v, %n8;
        scf.condition %more : (i1);
    } after {
        %v2 = builtin.load_local 0 : i8;
        %o = arith.constant 3 : i8;
        %v3 = arith.add %v2, %o <{ overflow = #builtin.overflow<wrapping> }>;
        builtin.store_local 0, %v3;
        scf.yield : ();
    } : () -> ();
    %fin = builtin.load_local 0 : i8;
    %w = arith.zext %fin <{ ty = i32 }>;
    builtin.ret %w : (i32);
};
"#;

#[test]
fn a_loop_that_carries_nothing_counts_in_a_narrow_local() {
    // Counts by 3 in an i8 until >= n (as i8). n = 10 -> 12; n = 255 -> wraps: 0,3,...,252,255
    // (255 < 255 false) -> 255. n = 256 -> n8 = 0 -> 0.
    check(
        ZERO_CARRIED,
        CONV,
        "f",
        &[(&[10], Some(&[12])), (&[255], Some(&[255])), (&[256], Some(&[0]))],
    );
}

// H: an `scf.yield` of a value defined in an enclosing region (narrow), and a block argument.
const YIELD_OUTER: &str = r#"
builtin.function public extern("wasm") @f(%c: i1, %d: i1, %x: i8) -> i8 {
    %one = arith.constant 1 : i8;
    %y = arith.add %x, %one <{ overflow = #builtin.overflow<wrapping> }>;
    %r = scf.if %c then {
        %s = scf.if %d then {
            scf.yield %y : (i8);
        } else {
            scf.yield %x : (i8);
        } : (i1) -> (i8);
        scf.yield %s : (i8);
    } else {
        %t = arith.add %y, %y <{ overflow = #builtin.overflow<wrapping> }>;
        scf.yield %t : (i8);
    } : (i1) -> (i8);
    builtin.ret %r : (i8);
};
"#;

#[test]
fn a_yield_of_values_defined_outside_the_construct_gives_them() {
    check(
        YIELD_OUTER,
        CONV,
        "f",
        &[
            (&[1, 1, 255], Some(&[0])),
            (&[1, 0, 255], Some(&[255])),
            (&[0, 0, 200], Some(&[146])),
        ],
    );
}

// I: a loop whose results are the before body's block arguments, narrow (i1, i8, i16), with a break.
const NARROW_LOOP: &str = r#"
builtin.function public extern("wasm") @f(%n: i16) -> (i16, i1) {
    %z = arith.constant 0 : i16;
    %f0 = arith.constant 0 : i1;
    %r, %fl = scf.while %z, %f0 before {
    ^b(%i: i16, %flag: i1):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i, %flag : (i1, i16, i1);
    } after {
    ^a(%j: i16, %g: i1):
        %c = arith.constant 40000 : i16;
        %hit = arith.cmpi ugt %j, %c;
        scf.if %hit then {
            %t = arith.constant 1 : i1;
            scf.break 1 %j, %t : (i16, i1);
        } else {
            scf.yield : ();
        } : (i1) -> ();
        %step = arith.constant 30000 : i16;
        %nj = arith.add %j, %step <{ overflow = #builtin.overflow<wrapping> }>;
        scf.yield %nj, %g : (i16, i1);
    } : (i16, i1) -> (i16, i1);
    builtin.ret %r, %fl : (i16, i1);
};
"#;

#[test]
fn a_loop_of_narrow_values_left_by_a_break_gives_the_break_s_values() {
    // n = 65535: 0 -> 30000 -> 60000 (> 40000: break (60000, 1)).
    // n = 50000: 0 -> 30000 -> 60000 >= n: cond false -> (60000, 0).
    // n = 20000: 0 -> 30000 -> stop (30000, 0).
    check(
        NARROW_LOOP,
        CONV,
        "f",
        &[
            (&[65535], Some(&[60000, 1])),
            (&[50000], Some(&[60000, 0])),
            (&[20000], Some(&[30000, 0])),
        ],
    );
}

// J: a frontend wasm.loop, wasm.if and wasm.block mixed with scf, and break/continue through them.
const FRONTEND_MIX: &str = r#"
builtin.function public extern("wasm") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    %r = wasm.block {
        %w = scf.while %zero before {
        ^b(%i: i32):
            %more = arith.cmpi ult %i, %n;
            scf.condition %more, %i : (i1, i32);
        } after {
        ^a(%j: i32):
            %three = arith.constant 3 : i32;
            %c = arith.cmpi eq %j, %three;
            %cc = arith.zext %c <{ ty = i32 }>;
            wasm.if %cc then {
                %ten = arith.constant 10 : i32;
                %jt = arith.add %j, %ten <{ overflow = #builtin.overflow<wrapping> }>;
                scf.continue 2 %jt : (i32);
            } else {
                wasm.end : ();
            } : (i32) -> ();
            %eleven = arith.constant 15 : i32;
            %d = arith.cmpi eq %j, %eleven;
            scf.if %d then {
                %k = arith.constant 777 : i32;
                scf.break 2 %k : (i32);
            } else {
                scf.yield : ();
            } : (i1) -> ();
            %nj = arith.add %j, %one <{ overflow = #builtin.overflow<wrapping> }>;
            scf.yield %nj : (i32);
        } : (i32) -> (i32);
        wasm.end %w : (i32);
    } : () -> (i32);
    builtin.ret %r : (i32);
};
"#;

#[test]
fn frontend_wasm_constructs_count_as_levels_of_scf_exits() {
    // 0,1,2,3 -> continue 13; 13 >= n? For n = 20: 13..19 -> 20 stop -> 20. For n = 12: 0..3 ->
    // 13 >= 12 -> 13. For n = 3: 3.
    check(
        FRONTEND_MIX,
        CONV,
        "f",
        &[
            (&[20], Some(&[777])),
            (&[15], Some(&[15])),
            (&[12], Some(&[13])),
            (&[3], Some(&[3])),
        ],
    );
}

// ---------------------------------------------------------------------------------------------
// Programs in CFG form, through the four passes and the optimization pipeline

// K: nested loops, an exit from the inner loop straight to the function.
const NESTED_RET: &str = r#"
builtin.function public extern("C") @f(%n: i32, %k: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    cf.br ^outer(%zero, %zero : i32, i32);
^outer(%i: i32, %acc: i32):
    %od = arith.cmpi uge %i, %n;
    cf.cond_br %od ^done, ^inner_pre : (i1);
^inner_pre:
    cf.br ^inner(%zero, %acc : i32, i32);
^inner(%j: i32, %a: i32):
    %id = arith.cmpi uge %j, %i;
    cf.cond_br %id ^inner_done(%a : i32), ^inner_body : (i1);
^inner_body:
    %prod = arith.mul %i, %j;
    %hit = arith.cmpi eq %prod, %k;
    cf.cond_br %hit ^found, ^inner_latch : (i1);
^found:
    %big = arith.constant 100000 : i32;
    %r = arith.add %a, %big;
    builtin.ret %r : (i32);
^inner_latch:
    %a2 = arith.add %a, %j;
    %j2 = arith.add %j, %one;
    cf.br ^inner(%j2, %a2 : i32, i32);
^inner_done(%af: i32):
    %i2 = arith.add %i, %one;
    cf.br ^outer(%i2, %af : i32, i32);
^done:
    builtin.ret %acc : (i32);
};
"#;

fn nested_ret_model(n: u32, k: u32) -> u32 {
    let mut acc = 0u32;
    for i in 0..n {
        for j in 0..i {
            if i.wrapping_mul(j) == k {
                return acc.wrapping_add(100000);
            }
            acc = acc.wrapping_add(j);
        }
    }
    acc
}

#[test]
fn nested_cfg_loops_return_from_the_inner_one() {
    let inputs = [(0, 0), (1, 0), (5, 1000), (5, 6), (6, 0), (7, 12), (10, 81), (10, 80)];
    let table: Vec<([u32; 2], [u32; 1])> =
        inputs.iter().map(|&(n, k)| ([n, k], [nested_ret_model(n, k)])).collect();
    let cases: Vec<(&[u32], Option<&[u32]>)> =
        table.iter().map(|(a, w)| (&a[..], Some(&w[..]))).collect();
    check(NESTED_RET, FULL, "f", &cases);
    check(NESTED_RET, OPT, "f", &cases);
}

// L: a loop of calls, with a tail call exit; mutual recursion.
const CALLS_LOOP: &str = r#"
builtin.function public extern("C") @twice(%x: i32) -> i32 {
    %r = arith.add %x, %x;
    builtin.ret %r : (i32);
};
builtin.function public extern("C") @is_even(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %z = arith.cmpi eq %x, %zero;
    cf.cond_br %z ^yes, ^no : (i1);
^yes:
    %one = arith.constant 1 : i32;
    builtin.ret %one : (i32);
^no:
    %o = arith.constant 1 : i32;
    %y = arith.sub %x, %o;
    %r = builtin.call @is_odd(%y) : extern("C") (i32) -> i32;
    builtin.ret %r : (i32);
};
builtin.function public extern("C") @is_odd(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %z = arith.cmpi eq %x, %zero;
    cf.cond_br %z ^yes, ^no : (i1);
^yes:
    builtin.ret %zero : (i32);
^no:
    %o = arith.constant 1 : i32;
    %y = arith.sub %x, %o;
    builtin.tail_call @is_even(%y) : extern("C") (i32) -> i32;
};
builtin.function public extern("C") @f(%n: i32, %stop: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    cf.br ^head(%zero, %one : i32, i32);
^head(%i: i32, %v: i32):
    %d = arith.cmpi uge %i, %n;
    cf.cond_br %d ^out, ^body : (i1);
^body:
    %w = builtin.call @twice(%v) : extern("C") (i32) -> i32;
    %e = builtin.call @is_even(%i) : extern("C") (i32) -> i32;
    %w2 = arith.add %w, %e;
    %s = arith.cmpi ugt %w2, %stop;
    cf.cond_br %s ^tail, ^latch : (i1);
^tail:
    builtin.tail_call @is_odd(%w2) : extern("C") (i32) -> i32;
^latch:
    %i2 = arith.add %i, %one;
    cf.br ^head(%i2, %w2 : i32, i32);
^out:
    builtin.ret %v : (i32);
};
"#;

fn calls_model(n: u32, stop: u32) -> u32 {
    let mut v = 1u32;
    for i in 0..n {
        let w2 = v.wrapping_mul(2).wrapping_add((i % 2 == 0) as u32);
        if w2 > stop {
            return (w2 % 2 == 1) as u32;
        }
        v = w2;
    }
    v
}

#[test]
fn a_loop_of_calls_with_a_tail_call_exit_and_mutual_recursion_matches_its_model() {
    let inputs = [(0, 100), (3, 100), (5, 1000), (5, 10), (6, 20), (8, 50)];
    let table: Vec<([u32; 2], [u32; 1])> =
        inputs.iter().map(|&(n, k)| ([n, k], [calls_model(n, k)])).collect();
    let cases: Vec<(&[u32], Option<&[u32]>)> =
        table.iter().map(|(a, w)| (&a[..], Some(&w[..]))).collect();
    check(CALLS_LOOP, FULL, "f", &cases);
    check(CALLS_LOOP, OPT, "f", &cases);
}

// M: a cf.switch over an i8 selector (narrow), with a negative key.
const NARROW_SWITCH: &str = r#"
builtin.function public extern("C") @f(%x: i8) -> i32 {
    %w = arith.sext %x <{ ty = i32 }>;
    cf.switch %w [#builtin.i32<0> -> ^a, #builtin.i32<1> -> ^b, #builtin.i32<-1> -> ^c, #builtin.i32<2> -> ^d], ^e : (i32);
^a:
    %r0 = arith.constant 10 : i32;
    builtin.ret %r0 : (i32);
^b:
    %r1 = arith.constant 11 : i32;
    builtin.ret %r1 : (i32);
^c:
    %r2 = arith.constant 12 : i32;
    builtin.ret %r2 : (i32);
^d:
    %r3 = arith.constant 13 : i32;
    builtin.ret %r3 : (i32);
^e:
    %r4 = arith.constant 14 : i32;
    builtin.ret %r4 : (i32);
};
"#;

#[test]
fn a_cf_switch_over_a_sign_extended_narrow_selector_takes_the_right_case() {
    // The carried i8 -1 is 255.
    let cases: &[(&[u32], Option<&[u32]>)] = &[
        (&[0], Some(&[10])),
        (&[1], Some(&[11])),
        (&[255], Some(&[12])),
        (&[2], Some(&[13])),
        (&[3], Some(&[14])),
        (&[254], Some(&[14])),
    ];
    check(NARROW_SWITCH, FULL, "f", cases);
    check(NARROW_SWITCH, OPT, "f", cases);
}

// N: every path ends in ub.unreachable.
const ALL_UNREACHABLE: &str = r#"
builtin.function public extern("C") @f(%c: i1) -> i32 {
    cf.cond_br %c ^a, ^b : (i1);
^a:
    ub.unreachable;
^b:
    ub.unreachable;
};
"#;

#[test]
fn a_function_whose_every_path_is_unreachable_traps() {
    let cases: &[(&[u32], Option<&[u32]>)] = &[(&[0], None), (&[1], None)];
    check(ALL_UNREACHABLE, FULL, "f", cases);
    check(ALL_UNREACHABLE, OPT, "f", cases);
}

// O: narrow locals across a CFG loop.
const LOCALS_LOOP: &str = r#"
builtin.function public extern("C") @f(%n: i32) -> i32 locals(i8, i1) {
    %z = arith.constant 0 : i8;
    %f = arith.constant 0 : i1;
    builtin.store_local 0, %z;
    builtin.store_local 1, %f;
    %zero = arith.constant 0 : i32;
    cf.br ^head(%zero : i32);
^head(%i: i32):
    %d = arith.cmpi uge %i, %n;
    cf.cond_br %d ^out, ^body : (i1);
^body:
    %v = builtin.load_local 0 : i8;
    %step = arith.constant 100 : i8;
    %v2 = arith.add %v, %step <{ overflow = #builtin.overflow<wrapping> }>;
    builtin.store_local 0, %v2;
    %wrapped = arith.cmpi ult %v2, %v;
    %old = builtin.load_local 1 : i1;
    %new = arith.bor %old, %wrapped;
    builtin.store_local 1, %new;
    %one = arith.constant 1 : i32;
    %i2 = arith.add %i, %one;
    cf.br ^head(%i2 : i32);
^out:
    %fin = builtin.load_local 0 : i8;
    %fl = builtin.load_local 1 : i1;
    %a = arith.zext %fin <{ ty = i32 }>;
    %b = arith.zext %fl <{ ty = i32 }>;
    %k = arith.constant 1000 : i32;
    %bk = arith.mul %b, %k;
    %r = arith.add %a, %bk;
    builtin.ret %r : (i32);
};
"#;

#[test]
fn narrow_locals_keep_their_values_across_a_cfg_loop() {
    let model = |n: u32| {
        let (mut v, mut w) = (0u8, false);
        for _ in 0..n {
            let v2 = v.wrapping_add(100);
            w |= v2 < v;
            v = v2;
        }
        u32::from(v) + 1000 * u32::from(w)
    };
    let table: Vec<([u32; 1], [u32; 1])> =
        [0, 1, 2, 3, 5, 7].iter().map(|&n| ([n], [model(n)])).collect();
    let cases: Vec<(&[u32], Option<&[u32]>)> =
        table.iter().map(|(a, w)| (&a[..], Some(&w[..]))).collect();
    check(LOCALS_LOOP, FULL, "f", &cases);
    check(LOCALS_LOOP, OPT, "f", &cases);
}

// ---------------------------------------------------------------------------------------------
// Frames

/// [check], and then that the stack pointer is back at [STACK_TOP] after each run that returns,
/// and that every value it was set to is 16-byte aligned and not above [STACK_TOP].
#[track_caller]
fn check_sp(functions: &str, pipeline: &str, name: &str, cases: &[(&[u32], Option<&[u32]>)]) {
    let (_context, module) = lowered_by(&module_of(functions), pipeline);
    let function = function_named(module, name);
    for (inputs, expected) in cases {
        let mut machine = machine();
        let outcome = run_in(&mut machine, function, &i32s(inputs));
        let expected_outcome =
            expected.map_or(Outcome::Trap, |values| Outcome::Values(i32s(values)));
        assert_eq!(outcome, expected_outcome, "@{name}{inputs:?}\n{}", text_of(module));
        if expected.is_some() {
            assert_eq!(
                machine.globals["__stack_pointer"],
                Val::i32(STACK_TOP),
                "the stack pointer is not restored for {inputs:?}: {:?}\n{}",
                machine.global_sets,
                text_of(module)
            );
        }
        for (_, value) in &machine.global_sets {
            let value = value.bits as u32;
            assert!(
                value.is_multiple_of(16) && value <= STACK_TOP,
                "the stack pointer was set to {value}: {:?}",
                machine.global_sets
            );
        }
    }
}

const FR_RECURSE: &str = r#"
builtin.function public extern("C") @f(%n: i32) -> i32 {
    %p = mem.alloca i32;
    %q = mem.alloca array<i8; 40>, align 64;
    mem.store %n, %p;
    %qa = mem.ptrtoint %q;
    %m = arith.constant 63 : isize;
    %low = arith.band %qa, %m;
    %zero = arith.constant 0 : i32;
    %z = arith.cmpi eq %n, %zero;
    %r = scf.if %z then {
        %one = arith.constant 1 : i32;
        scf.yield %one : (i32);
    } else {
        %o = arith.constant 1 : i32;
        %n1 = arith.sub %n, %o;
        %rr = builtin.call @f(%n1) : extern("C") (i32) -> i32;
        %v = mem.load %p : i32;
        %prod = arith.mul %v, %rr;
        scf.yield %prod : (i32);
    } : (i1) -> (i32);
    %lowi = mem.ptrtoint %q;
    %bad = arith.cmpi ne %low, %m;
    builtin.ret %r : (i32);
};
"#;

#[test]
fn a_recursive_function_with_an_over_aligned_frame_restores_the_stack_pointer() {
    check_sp(
        FR_RECURSE,
        FR,
        "f",
        &[(&[0], Some(&[1])), (&[1], Some(&[1])), (&[5], Some(&[120]))],
    );
}

// Over-aligned frame: the alloca's address is 64-aligned and inside [new sp, old sp).
const FR_ALIGN: &str = r#"
builtin.function public extern("C") @f(%x: i32) -> isize {
    %a = mem.alloca i8;
    %q = mem.alloca array<i8; 40>, align 64;
    %qa = mem.ptrtoint %q;
    builtin.ret %qa : (isize);
};
"#;

#[test]
fn an_over_aligned_slot_is_aligned_and_below_the_stack_pointer_wherever_it_starts() {
    let (_context, module) = lowered_by(&module_of(FR_ALIGN), FR);
    for sp in [1024u32, 1008, 1040, 992] {
        let mut machine = Machine::new(2048);
        machine.globals.insert(String::from("__stack_pointer"), Val::i32(sp));
        let outcome = run_in(&mut machine, function_named(module, "f"), &i32s(&[0]));
        let Outcome::Values(values) = outcome else {
            panic!("@f trapped\n{}", text_of(module));
        };
        let slot = values[0].bits as u32;
        assert_eq!(slot % 64, 0, "{}", text_of(module));
        assert!(slot + 40 <= sp, "the slot {slot} is not below {sp}\n{}", text_of(module));
        assert_eq!(machine.globals["__stack_pointer"], Val::i32(sp));
    }
}

const FR_EXITS: &str = r#"
builtin.function public extern("C") @g(%x: i32) -> i32 {
    %p = mem.alloca i64;
    %r = arith.add %x, %x;
    builtin.ret %r : (i32);
};
builtin.function public extern("C") @f(%s: i32, %x: i32) -> i32 {
    %p = mem.alloca i32;
    mem.store %x, %p;
    %r = scf.index_switch %s
    case 0 {
        %q = mem.alloca i32;
        mem.store %s, %q;
        builtin.tail_call @g(%x) : extern("C") (i32) -> i32;
    }
    case 1 {
        %v = mem.load %p : i32;
        builtin.ret %v : (i32);
    }
    case 2 {
        %fr = builtin.function_ref @g : (i32) -> i32;
        builtin.tail_call_indirect %fr(%x) : extern("C") (i32) -> i32;
    }
    default {
        %c = arith.constant 1 : i1;
        %w = scf.if %c then {
            %pp = mem.alloca i16;
            %seven = arith.constant 7 : i32;
            scf.yield %seven : (i32);
        } else {
            ub.unreachable;
        } : (i1) -> (i32);
        scf.yield %w : (i32);
    } : (i32);
    builtin.ret %r : (i32);
};
"#;

#[test]
fn every_exit_of_a_function_with_a_frame_restores_the_stack_pointer() {
    check_sp(
        FR_EXITS,
        FR,
        "f",
        &[
            (&[0, 5], Some(&[10])),
            (&[1, 5], Some(&[5])),
            (&[2, 6], Some(&[12])),
            (&[3, 6], Some(&[7])),
        ],
    );
}

#[test]
fn an_alloca_in_a_loop_or_in_unstructured_control_is_an_error() {
    let in_loop = |inner: &str| {
        format!(
            "builtin.function public extern(\"C\") @f(%n: i32) -> i32 {{\n{inner}\n    \
             builtin.ret %n : (i32);\n}};"
        )
    };
    // a CFG loop with no cfg-to-scf
    let cfg = r#"
builtin.function public extern("C") @f(%n: i32) -> i32 {
    %p = mem.alloca i32;
    cf.br ^h(%n : i32);
^h(%i: i32):
    %z = arith.constant 0 : i32;
    %d = arith.cmpi eq %i, %z;
    cf.cond_br %d ^o, ^h(%z : i32) : (i1);
^o:
    builtin.ret %i : (i32);
};"#;
    let e = lowers(cfg, FR).unwrap_err();
    assert!(e.contains("structured"), "{e}");
    // an alloca in a before region and in a wasm.loop
    let before = in_loop(
        "    scf.while before {\n        %p = mem.alloca i32;\n        %c = arith.constant 0 : \
         i1;\n        scf.condition %c : (i1);\n    } after {\n        scf.yield : ();\n    } : \
         () -> ();",
    );
    let e = lowers(&before, FR).unwrap_err();
    assert!(e.contains("in a loop"), "{e}");
    let wl = in_loop(
        "    wasm.loop {\n        %p = mem.alloca i32;\n        wasm.end : ();\n    } : () -> ();",
    );
    let e = lowers(&wl, FR).unwrap_err();
    assert!(e.contains("in a loop"), "{e}");
}

// ---------------------------------------------------------------------------------------------
// Memory
/// Lowers `functions` by [FR], writes each of `init` (an address and bytes) in a fresh
/// [machine], and runs `@name` on `inputs`: the outcome, the memory after, and the lowered text.
fn run_mem(
    functions: &str,
    name: &str,
    init: &[(u32, &[u8])],
    inputs: &[u32],
) -> (Outcome, Vec<u8>, String) {
    let (_context, module) = lowered_by(&module_of(functions), FR);
    let mut machine = machine();
    for (address, bytes) in init {
        machine.write(*address, bytes);
    }
    let outcome = run_in(&mut machine, function_named(module, name), &i32s(inputs));
    (outcome, machine.memory.clone(), text_of(module))
}

#[test]
fn narrow_loads_mask_and_narrow_stores_write_the_layout_s_bytes() {
    for (ty, bytes, want) in [
        ("i1", &[0x02u8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff][..], 0u64),
        ("i1", &[0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff][..], 1),
        ("i7", &[0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff][..], 0x7f),
        ("i8", &[0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff][..], 0xff),
        ("i16", &[0x34u8, 0x92, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff][..], 0x9234),
        ("i24", &[0x01u8, 0x02, 0x83, 0xff, 0xff, 0xff, 0xff, 0xff][..], 0x830201),
    ] {
        let body = format!(
            "builtin.function public extern(\"wasm\") @f(%p: ptr<{ty}>) -> {ty} {{\n    %v = \
             mem.load %p : {ty};\n    builtin.ret %v : ({ty});\n}};"
        );
        let (out, _, lowered) = run_mem(&body, "f", &[(16, bytes)], &[16]);
        assert_eq!(out, Outcome::Values(i32s(&[want as u32])), "{ty}\n{lowered}");
    }
    // i40: 8 bytes, masked to 40 bits; i64 result.
    let body = "builtin.function public extern(\"wasm\") @f(%p: ptr<i40>) -> i40 {\n    %v = \
                mem.load %p : i40;\n    builtin.ret %v : (i40);\n};";
    let (out, _, lowered) = run_mem(body, "f", &[(16, &[1, 2, 3, 4, 5, 6, 7, 8])], &[16]);
    assert_eq!(out, Outcome::Values(vec![Val::i64(0x05_0403_0201)]), "{lowered}");
    // Stores: i24 writes how many bytes? i40?
    for (ty, size) in [("i24", 4usize), ("i40", 8), ("i1", 1), ("i7", 1), ("i16", 2)] {
        let body = format!(
            "builtin.function public extern(\"wasm\") @f(%p: ptr<{ty}>, %q: ptr<{ty}>) {{\n    %v \
             = mem.load %p : {ty};\n    mem.store %v, %q;\n    %s = mem.sizeof {ty};\n    \
             builtin.ret;\n}};"
        );
        let (out, mem, lowered) =
            run_mem(&body, "f", &[(16, &[0xff; 8]), (32, &[0xaa; 16])], &[16, 32]);
        assert_eq!(out, Outcome::Values(vec![]), "{lowered}");
        let written = mem[32..48].iter().take_while(|b| **b != 0xaa).count();
        assert_eq!(written, size, "{ty}: {:02x?}\n{lowered}", &mem[32..48]);
    }
}

#[test]
fn geps_and_bulk_operations_address_and_move_the_right_bytes() {
    let body = r#"
builtin.function public extern("wasm") @f(%p: ptr<struct<i8, i64, array<i16; 3>>>, %i: isize) -> isize {
    %q = mem.gep %p, %i, [2, 1];
    %a = mem.ptrtoint %q;
    builtin.ret %a : (isize);
};
builtin.function public extern("wasm") @z(%p: ptr<array<i8; 0>>, %i: isize) -> isize {
    %q = mem.gep %p, %i;
    %a = mem.ptrtoint %q;
    builtin.ret %a : (isize);
};
builtin.function public extern("wasm") @bulk(%d: ptr<i8>, %s: ptr<i8>, %n: isize, %b: i8) {
    mem.memmove %d, %s, %n;
    %d2 = mem.gep %d, %n;
    mem.memset %d2, %b, %n;
    builtin.ret;
};
"#;
    // struct<i8, i64, array<i16;3>>: i8 @0, i64 @8, array @16 (6 bytes), size 24 (align 8).
    let (out, _, lowered) = run_mem(body, "f", &[], &[100, u32::MAX]);
    assert_eq!(out, Outcome::Values(i32s(&[100 - 24 + 18])), "{lowered}");
    let (out, _, lowered) = run_mem(body, "f", &[], &[100, 2]);
    assert_eq!(out, Outcome::Values(i32s(&[100 + 48 + 18])), "{lowered}");
    let (out, _, lowered) = run_mem(body, "z", &[], &[100, 7]);
    assert_eq!(out, Outcome::Values(i32s(&[100])), "{lowered}");
    let (out, mem, lowered) =
        run_mem(body, "bulk", &[(10, &[1, 2, 3, 4, 5, 6])], &[12, 10, 4, 0x1ff]);
    assert_eq!(out, Outcome::Values(vec![]), "{lowered}");
    assert_eq!(&mem[10..20], &[1, 2, 1, 2, 3, 4, 0xff, 0xff, 0xff, 0xff], "{lowered}");
}

#[test]
fn the_builder_of_a_construct_numbers_its_results_in_order() {
    use runic_dialect_wasm::WasmOpBuilder;
    use runic_ir::{Builder, OpBuilder};
    let (context, module) = parse_module(&module_of(
        "builtin.function public extern(\"wasm\") @f(%x: i32) -> i32 {\n builtin.ret %x : \
         (i32);\n};",
    ));
    let f = function_named(module, "f");
    let ret = {
        let f = f.borrow();
        let func = f.downcast_ref::<runic_ir::dialects::builtin::Function>().unwrap();
        func.body().entry_block_ref().unwrap().borrow().terminator().unwrap()
    };
    let mut b = OpBuilder::new(context.clone());
    b.set_insertion_point_before(ret);
    let i32t = context.i32();
    let i64t = context.i64();
    let blk = b
        .block(Vec::<runic_ir::ValueRef>::new(), &[i32t, i64t, i32t], Default::default())
        .unwrap();
    let op = blk.as_operation_ref();
    let idx: Vec<usize> = op.borrow().results().iter().map(|r| r.borrow().index()).collect();
    assert_eq!(idx, [0, 1, 2]);
}

// P: break 2 from the inner loop's before region (naming the wrapped outer loop), continue 2 from
// a dense switch in the inner after body, and a break 1 of the inner loop from its before region.
const BEFORE_BREAKS: &str = r#"
builtin.function public extern("wasm") @f(%n: i32, %k: i32) -> (i32, i32) {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    %ro, %rs = scf.while %zero, %zero before {
    ^b(%i: i32, %s: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i, %s : (i1, i32, i32);
    } after {
    ^a(%i2: i32, %s2: i32):
        %jr, %js = scf.while %zero, %s2 before {
        ^ib(%j: i32, %t: i32):
            %hitk = arith.cmpi eq %t, %k;
            scf.if %hitk then {
                %m = arith.constant 5000 : i32;
                scf.break 2 %m, %t : (i32, i32);
            } else {
                scf.yield : ();
            } : (i1) -> ();
            %four = arith.constant 4 : i32;
            %lim = arith.cmpi uge %j, %four;
            scf.if %lim then {
                scf.break 1 %j, %t : (i32, i32);
            } else {
                scf.yield : ();
            } : (i1) -> ();
            %c = arith.constant 1 : i1;
            scf.condition %c, %j, %t : (i1, i32, i32);
        } after {
        ^ia(%jj: i32, %tt: i32):
            %sel = arith.sub %jj, %i2 <{ overflow = #builtin.overflow<wrapping> }>;
            %tn = arith.add %tt, %one <{ overflow = #builtin.overflow<wrapping> }>;
            scf.index_switch %sel
            case 0 {
                scf.yield : ();
            }
            case 1 {
                %ni = arith.add %i2, %one <{ overflow = #builtin.overflow<wrapping> }>;
                %ten = arith.constant 10 : i32;
                %tx = arith.add %tn, %ten <{ overflow = #builtin.overflow<wrapping> }>;
                scf.continue 2 %ni, %tx : (i32, i32);
            }
            case 3 {
                scf.yield : ();
            }
            default {
                scf.yield : ();
            } : ();
            %jn = arith.add %jj, %one <{ overflow = #builtin.overflow<wrapping> }>;
            scf.yield %jn, %tn : (i32, i32);
        } : (i32, i32) -> (i32, i32);
        %ni2 = arith.add %i2, %one <{ overflow = #builtin.overflow<wrapping> }>;
        %h = arith.constant 100 : i32;
        %sx = arith.add %js, %h <{ overflow = #builtin.overflow<wrapping> }>;
        scf.yield %ni2, %sx : (i32, i32);
    } : (i32, i32) -> (i32, i32);
    builtin.ret %ro, %rs : (i32, i32);
};
"#;

fn before_breaks_model(n: u32, k: u32) -> (u32, u32) {
    let (mut i, mut s) = (0u32, 0u32);
    'outer: while i < n {
        let (mut j, mut t) = (0u32, s);
        let (_jr, js) = loop {
            if t == k {
                return (5000, t);
            }
            if j >= 4 {
                break (j, t);
            }
            let sel = j.wrapping_sub(i);
            let tn = t.wrapping_add(1);
            if sel == 1 {
                i = i.wrapping_add(1);
                s = tn.wrapping_add(10);
                continue 'outer;
            }
            j += 1;
            t = tn;
        };
        i += 1;
        s = js.wrapping_add(100);
    }
    (i, s)
}

#[test]
fn breaks_and_continues_from_the_before_body_of_an_inner_loop_match_their_model() {
    let inputs = [
        (0, 1),
        (1, 999),
        (3, 999),
        (5, 999),
        (6, 999),
        (5, 3),
        (5, 13),
        (5, 30),
        (8, 200),
    ];
    let table: Vec<([u32; 2], [u32; 2])> = inputs
        .iter()
        .map(|&(n, k)| {
            let (a, b) = before_breaks_model(n, k);
            ([n, k], [a, b])
        })
        .collect();
    let cases: Vec<(&[u32], Option<&[u32]>)> =
        table.iter().map(|(a, w)| (&a[..], Some(&w[..]))).collect();
    check(BEFORE_BREAKS, CONV, "f", &cases);
}

#[test]
fn the_programs_give_the_same_results_after_the_optimization_pipeline() {
    check(
        SW_IN_LOOP,
        OPT,
        "f",
        &[
            (&[6, 5], Some(&[5, 114])),
            (&[6, 9], Some(&[6, 116])),
            (&[0, 0], Some(&[0, 0])),
            (&[3, 1], Some(&[1, 1])),
        ],
    );
    check(
        CONT_FROM_BEFORE,
        OPT,
        "f",
        &[(&[5], Some(&[10])), (&[0], Some(&[0])), (&[30], Some(&[179]))],
    );
    check(
        BREAK_OUTER_FROM_SWITCH,
        OPT,
        "f",
        &[(&[5, 3], Some(&[1003])), (&[5, 9], Some(&[5])), (&[0, 0], Some(&[0]))],
    );
    check(
        SWITCH_EXITS,
        OPT,
        "f",
        &[
            (&[0, 5], Some(&[5])),
            (&[1, 5], Some(&[35])),
            (&[2, 5], None),
            (&[3, 5], Some(&[4])),
            (&[4, 5], Some(&[100])),
        ],
    );
    check(
        IF_ARMS,
        OPT,
        "f",
        &[(&[100], Some(&[57])), (&[4], Some(&[5])), (&[0], Some(&[0]))],
    );
    check(
        ZERO_CARRIED,
        OPT,
        "f",
        &[(&[10], Some(&[12])), (&[255], Some(&[255])), (&[256], Some(&[0]))],
    );
    check(
        NARROW_LOOP,
        OPT,
        "f",
        &[
            (&[65535], Some(&[60000, 1])),
            (&[50000], Some(&[60000, 0])),
            (&[20000], Some(&[30000, 0])),
        ],
    );
    check(
        FRONTEND_MIX,
        OPT,
        "f",
        &[
            (&[20], Some(&[777])),
            (&[15], Some(&[15])),
            (&[12], Some(&[13])),
            (&[3], Some(&[3])),
        ],
    );
    check_sp(
        FR_EXITS,
        OPT,
        "f",
        &[
            (&[0, 5], Some(&[10])),
            (&[1, 5], Some(&[5])),
            (&[2, 6], Some(&[12])),
            (&[3, 6], Some(&[7])),
        ],
    );
    check_sp(FR_RECURSE, OPT, "f", &[(&[0], Some(&[1])), (&[5], Some(&[120]))]);
}
