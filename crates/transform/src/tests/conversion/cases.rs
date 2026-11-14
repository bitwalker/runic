//! Modules written by hand, one per feature, converted `i128 -> (i64, i64)` in a full conversion:
//! the converted text is pinned, and the evaluator gives the same answers before and after on a
//! table of inputs.

use alloc::{format, string::String, vec, vec::Vec};

use runic_expect_test::{Expect, expect};

use super::{Family, Input, Mode, Setup, check, patterns::Kind};

/// Inputs that reach the carries and the halves: zero, one, all ones, each half alone, and
/// values whose low halves sum past 2^64.
const WIDE: [u128; 8] = [
    0,
    1,
    u128::MAX,
    u64::MAX as u128,
    (u64::MAX as u128) << 64,
    0x8000_0000_0000_0000,
    0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
    0xffff_ffff_ffff_fffe_ffff_ffff_ffff_ffff,
];

/// Every call of `function` with `arity` parameters drawn from [WIDE], with `extra` appended
/// (the values of the parameters that are not `i128`).
fn table(function: &str, arity: usize, extra: &[u128]) -> Vec<Input> {
    let mut inputs = Vec::new();
    let count = WIDE.len().pow(arity as u32);
    for mut index in 0..count {
        let mut args = Vec::new();
        for _ in 0..arity {
            args.push(WIDE[index % WIDE.len()]);
            index /= WIDE.len();
        }
        args.extend_from_slice(extra);
        inputs.push((String::from(function), args));
    }
    inputs
}

/// [table] with each of `false` and `true` appended for a last `i1` parameter.
fn table_with_flag(function: &str, arity: usize) -> Vec<Input> {
    let mut inputs = table(function, arity, &[0]);
    inputs.extend(table(function, arity, &[1]));
    inputs
}

/// Converts `body`, the body of `builtin.module public @m`, and checks the result against
/// `expected` and the evaluator against `inputs`.
fn case(body: &str, inputs: &[Input], expected: Expect) {
    case_with(&Setup::full(Family::Wide), body, inputs, expected);
}

/// [case] under `setup`.
fn case_with(setup: &Setup, body: &str, inputs: &[Input], expected: Expect) {
    let source = format!("builtin.module public @m {{\n{body}\n}};");
    let converted =
        check(&source, setup, inputs).unwrap_or_else(|err| panic!("{err}\n--- source\n{source}"));
    expected.assert_eq(&converted);
}

/// `@sum`, the callee of most cases.
const SUM: &str = r#"
    builtin.function public extern("C") @sum(%a: i128, %b: i128) -> i128 {
        %s = arith.add %a, %b;
        builtin.ret %s : (i128);
    };"#;

/// Direct calls: `@sum` adds with a carry, `@twice` calls it twice, and the results split.
#[test]
fn calls() {
    let mut inputs = table("sum", 2, &[]);
    inputs.extend(table("twice", 2, &[]));
    case(
        r#"
    builtin.function public extern("C") @sum(%a: i128, %b: i128) -> i128 {
        %s = arith.add %a, %b;
        builtin.ret %s : (i128);
    };
    builtin.function public extern("C") @twice(%a: i128, %b: i128) -> (i128, i128) {
        %x = builtin.call @sum(%a, %b) : extern("C") (i128, i128) -> i128;
        %y = builtin.call @sum(%x, %x) : extern("C") (i128, i128) -> i128;
        builtin.ret %x, %y : (i128, i128);
    };"#,
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @sum(%7: i64, %8: i64, %9: i64, %10: i64) -> (i64, i64) {
                    %13 = arith.add %7, %9;
                    %14 = arith.cmpi ult %13, %7;
                    %15 = arith.zext %14 <{ ty = #builtin.type<i64> }>;
                    %16 = arith.add %8, %10;
                    %17 = arith.add %16, %15;
                    builtin.ret %13, %17 : (i64, i64);
                };
                builtin.function public extern("C") @twice(%19: i64, %20: i64, %21: i64, %22: i64) -> (i64, i64, i64, i64) {
                    %25, %26 = builtin.call @sum(%19, %20, %21, %22) : extern("C") (i64, i64, i64, i64) -> (i64, i64);
                    %28, %29 = builtin.call @sum(%25, %26, %25, %26) : extern("C") (i64, i64, i64, i64) -> (i64, i64);
                    builtin.ret %25, %26, %28, %29 : (i64, i64, i64, i64);
                };
            };"#]],
    );
}

/// `isize` becomes `i32`: the arithmetic is retyped in place (the constant truncated), the call
/// and its callee, the branch and its block arguments, and the local, by the builtin patterns.
#[test]
fn isize_is_retyped_in_place() {
    let values = [0u128, 1, 0xffff_ffff, 0x1_0000_0001, u64::MAX as u128, 0x8000_0000_7fff_ffff];
    let mut inputs = Vec::new();
    for a in values {
        for b in values {
            inputs.push((String::from("caller"), vec![a, b, a & 1]));
        }
    }
    let source = r#"builtin.module public @m {
    builtin.function public extern("C") @mix(%a: isize, %b: isize) -> isize {
        %c = arith.constant 4294967299 : isize;
        %p = arith.mul %a, %c;
        %q = arith.sub %p, %b;
        %r = arith.bxor %q, %a;
        builtin.ret %r : (isize);
    };
    builtin.function public extern("C") @caller(%a: isize, %b: isize, %c: i1) -> isize locals(isize) {
        builtin.store_local 0, %a;
        cf.cond_br %c ^t(%a : isize), ^e : (i1);
    ^t(%x: isize):
        %y = builtin.call @mix(%x, %b) : extern("C") (isize, isize) -> isize;
        builtin.store_local 0, %y;
        cf.br ^j(%y : isize);
    ^e:
        cf.br ^j(%b : isize);
    ^j(%z: isize):
        %l = builtin.load_local 0 : isize;
        %s = arith.add %z, %l;
        builtin.ret %s : (isize);
    };
};"#;
    let converted = check(source, &Setup::full(Family::Size), &inputs)
        .unwrap_or_else(|err| panic!("{err}\n--- source\n{source}"));
    expect![[r#"
        builtin.module public @m {
            builtin.function public extern("C") @mix(%14: i32, %15: i32) -> i32 {
                %2 = arith.constant 3 : i32;
                %3 = arith.mul %14, %2;
                %4 = arith.sub %3, %15;
                %5 = arith.bxor %4, %14;
                builtin.ret %5 : (i32);
            };
            builtin.function public extern("C") @caller(%22: i32, %23: i32, %24: i1) -> i32 locals(i32) {
                builtin.store_local 0, %22;
                cf.cond_br %24 ^block4(%22 : i32), ^block5 : (i1);
            ^block4(%27: i32):
                %31 = builtin.call @mix(%27, %23) : extern("C") (i32, i32) -> i32;
                builtin.store_local 0, %31;
                cf.br ^block6(%31 : i32);
            ^block5:
                cf.br ^block6(%23 : i32);
            ^block6(%29: i32):
                %33 = builtin.load_local 0 : i32;
                %13 = arith.add %29, %33;
                builtin.ret %13 : (i32);
            };
        };"#]].assert_eq(&converted);
}

/// Calls through an alias and through an alias of it: the aliases hold no signature and are left
/// alone, and each call keeps the path it was written with.
#[test]
fn an_alias() {
    let inputs = table("via_aliases", 2, &[]);
    case(
        &format!(
            r#"{SUM}
    builtin.function_alias public @plus -> @sum;
    builtin.function_alias public @plus_again -> @plus;
    builtin.function public extern("C") @via_aliases(%a: i128, %b: i128) -> i128 {{
        %x = builtin.call @plus(%a, %b) : extern("C") (i128, i128) -> i128;
        %y = builtin.call @plus_again(%x, %b) : extern("C") (i128, i128) -> i128;
        builtin.ret %y : (i128);
    }};"#
        ),
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @sum(%7: i64, %8: i64, %9: i64, %10: i64) -> (i64, i64) {
                    %13 = arith.add %7, %9;
                    %14 = arith.cmpi ult %13, %7;
                    %15 = arith.zext %14 <{ ty = #builtin.type<i64> }>;
                    %16 = arith.add %8, %10;
                    %17 = arith.add %16, %15;
                    builtin.ret %13, %17 : (i64, i64);
                };
                builtin.function_alias public @plus -> @sum;
                builtin.function_alias public @plus_again -> @plus;
                builtin.function public extern("C") @via_aliases(%19: i64, %20: i64, %21: i64, %22: i64) -> (i64, i64) {
                    %25, %26 = builtin.call @plus(%19, %20, %21, %22) : extern("C") (i64, i64, i64, i64) -> (i64, i64);
                    %28, %29 = builtin.call @plus_again(%25, %26, %21, %22) : extern("C") (i64, i64, i64, i64) -> (i64, i64);
                    builtin.ret %28, %29 : (i64, i64);
                };
            };"#]],
    );
}

/// An indirect call of a function reference, and of a reference to an alias: the reference and
/// the call both get the converted function type.
#[test]
fn an_indirect_call() {
    let inputs = table("indirect", 2, &[]);
    case(
        &format!(
            r#"{SUM}
    builtin.function_alias public @plus -> @sum;
    builtin.function public extern("C") @indirect(%a: i128, %b: i128) -> i128 {{
        %f = builtin.function_ref @sum : (i128, i128) -> i128;
        %g = builtin.function_ref @plus : (i128, i128) -> i128;
        %x = builtin.call_indirect %f(%a, %b) : extern("C") (i128, i128) -> i128;
        %y = builtin.call_indirect %g(%x, %a) : extern("C") (i128, i128) -> i128;
        builtin.ret %y : (i128);
    }};"#
        ),
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @sum(%9: i64, %10: i64, %11: i64, %12: i64) -> (i64, i64) {
                    %15 = arith.add %9, %11;
                    %16 = arith.cmpi ult %15, %9;
                    %17 = arith.zext %16 <{ ty = #builtin.type<i64> }>;
                    %18 = arith.add %10, %12;
                    %19 = arith.add %18, %17;
                    builtin.ret %15, %19 : (i64, i64);
                };
                builtin.function_alias public @plus -> @sum;
                builtin.function public extern("C") @indirect(%21: i64, %22: i64, %23: i64, %24: i64) -> (i64, i64) {
                    %27 = builtin.function_ref @sum : (i64, i64, i64, i64) -> (i64, i64);
                    %29 = builtin.function_ref @plus : (i64, i64, i64, i64) -> (i64, i64);
                    %31, %32 = builtin.call_indirect %27(%21, %22, %23, %24) : extern("C") (i64, i64, i64, i64) -> (i64, i64);
                    %34, %35 = builtin.call_indirect %29(%31, %32, %21, %22) : extern("C") (i64, i64, i64, i64) -> (i64, i64);
                    builtin.ret %34, %35 : (i64, i64);
                };
            };"#]],
    );
}

/// A diamond of `cf` branches: the conditional branch forwards an `i128` to each arm, each arm
/// forwards two to the join, and the join compares them (`arith.cmpi eq` as both halves equal).
#[test]
fn a_diamond() {
    let inputs = table_with_flag("diamond", 2);
    case(
        r#"
    builtin.function public extern("C") @diamond(%a: i128, %b: i128, %c: i1) -> (i128, i1) {
        cf.cond_br %c ^then(%a : i128), ^else(%b : i128) : (i1);
    ^then(%x: i128):
        %s = arith.add %x, %b;
        cf.br ^join(%s, %x : i128, i128);
    ^else(%y: i128):
        %t = arith.band %y, %a;
        %u = arith.bor %t, %b;
        cf.br ^join(%u, %t : i128, i128);
    ^join(%p: i128, %q: i128):
        %r = arith.bxor %p, %q;
        %e = arith.cmpi eq %p, %q;
        builtin.ret %r, %e : (i128, i1);
    };"#,
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @diamond(%12: i64, %13: i64, %14: i64, %15: i64, %16: i1) -> (i64, i64, i1) {
                    cf.cond_br %16 ^block3(%12, %13 : i64, i64), ^block4(%14, %15 : i64, i64) : (i1);
                ^block3(%19: i64, %20: i64):
                    %31 = arith.add %19, %14;
                    %32 = arith.cmpi ult %31, %19;
                    %33 = arith.zext %32 <{ ty = #builtin.type<i64> }>;
                    %34 = arith.add %20, %15;
                    %35 = arith.add %34, %33;
                    cf.br ^block5(%31, %35, %19, %20 : i64, i64, i64, i64);
                ^block4(%22: i64, %23: i64):
                    %37 = arith.band %22, %12;
                    %38 = arith.band %23, %13;
                    %40 = arith.bor %37, %14;
                    %41 = arith.bor %38, %15;
                    cf.br ^block5(%40, %41, %37, %38 : i64, i64, i64, i64);
                ^block5(%25: i64, %26: i64, %27: i64, %28: i64):
                    %43 = arith.bxor %25, %27;
                    %44 = arith.bxor %26, %28;
                    %46 = arith.cmpi eq %25, %27;
                    %47 = arith.cmpi eq %26, %28;
                    %48 = arith.band %46, %47;
                    builtin.ret %43, %44, %48 : (i64, i64, i1);
                };
            };"#]],
    );
}

/// A loop of `cf` branches whose counter and accumulator are `i128` block arguments: the counter
/// starts just below 2^64, so its increment carries into the high half, and the loop ends when it
/// equals its start plus three.
#[test]
fn a_loop() {
    let inputs = table("repeat", 1, &[]);
    case(
        r#"
    builtin.function public extern("C") @repeat(%a: i128) -> i128 {
        %start = arith.constant 18446744073709551614 : i128;
        %zero = arith.constant 0 : i128;
        cf.br ^head(%start, %zero : i128, i128);
    ^head(%i: i128, %acc: i128):
        %end = arith.constant 18446744073709551617 : i128;
        %done = arith.cmpi eq %i, %end;
        cf.cond_br %done ^exit, ^body : (i1);
    ^body:
        %next_acc = arith.add %acc, %a;
        %mixed = arith.bxor %next_acc, %i;
        %one = arith.constant 1 : i128;
        %next = arith.add %i, %one;
        cf.br ^head(%next, %mixed : i128, i128);
    ^exit:
        builtin.ret %acc : (i128);
    };"#,
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @repeat(%11: i64, %12: i64) -> (i64, i64) {
                    %20 = arith.constant -2 : i64;
                    %21 = arith.constant 0 : i64;
                    %23 = arith.constant 0 : i64;
                    %24 = arith.constant 0 : i64;
                    cf.br ^block3(%20, %21, %23, %24 : i64, i64, i64, i64);
                ^block3(%14: i64, %15: i64, %16: i64, %17: i64):
                    %26 = arith.constant 1 : i64;
                    %27 = arith.constant 1 : i64;
                    %29 = arith.cmpi eq %14, %26;
                    %30 = arith.cmpi eq %15, %27;
                    %31 = arith.band %29, %30;
                    cf.cond_br %31 ^block4, ^block5 : (i1);
                ^block5:
                    %32 = arith.add %16, %11;
                    %33 = arith.cmpi ult %32, %16;
                    %34 = arith.zext %33 <{ ty = #builtin.type<i64> }>;
                    %35 = arith.add %17, %12;
                    %36 = arith.add %35, %34;
                    %38 = arith.bxor %32, %14;
                    %39 = arith.bxor %36, %15;
                    %41 = arith.constant 1 : i64;
                    %42 = arith.constant 0 : i64;
                    %44 = arith.add %14, %41;
                    %45 = arith.cmpi ult %44, %14;
                    %46 = arith.zext %45 <{ ty = #builtin.type<i64> }>;
                    %47 = arith.add %15, %42;
                    %48 = arith.add %47, %46;
                    cf.br ^block3(%44, %48, %38, %39 : i64, i64, i64, i64);
                ^block4:
                    builtin.ret %16, %17 : (i64, i64);
                };
            };"#]],
    );
}

/// Declared effects of a definition whose parameters split: the effect on parameter 1 follows it
/// past the split of parameter 0, and the effect on result 0 is repeated on both halves. A caller
/// calls it.
#[test]
fn effects_of_a_function_whose_parameters_split() {
    let inputs = table("calls_touch", 2, &[]);
    case(
        r#"
    builtin.function public extern("C") @touch(%a: i128, %n: i64, %b: i128) -> i128 effects(read(arg 1), write(result 0)) {
        %x = arith.bxor %a, %b;
        builtin.ret %x : (i128);
    };
    builtin.function public extern("C") @calls_touch(%a: i128, %b: i128) -> i128 {
        %n = arith.constant 7 : i64;
        %x = builtin.call @touch(%a, %n, %b) : extern("C") (i128, i64, i128) -> i128;
        builtin.ret %x : (i128);
    };"#,
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @touch(%8: i64, %9: i64, %10: i64, %11: i64, %12: i64) -> (i64, i64) effects(read(arg 2), write(result 0), write(result 1)) {
                    %15 = arith.bxor %8, %11;
                    %16 = arith.bxor %9, %12;
                    builtin.ret %15, %16 : (i64, i64);
                };
                builtin.function public extern("C") @calls_touch(%18: i64, %19: i64, %20: i64, %21: i64) -> (i64, i64) {
                    %6 = arith.constant 7 : i64;
                    %24, %25 = builtin.call @touch(%18, %19, %6, %20, %21) : extern("C") (i64, i64, i64, i64, i64) -> (i64, i64);
                    builtin.ret %24, %25 : (i64, i64);
                };
            };"#]],
    );
}

/// Locals of `i128` (and one of `i64`, which moves past them) with stores on both sides of a
/// branch and loads after the join.
#[test]
fn locals_on_both_sides_of_a_branch() {
    let inputs = table_with_flag("locals", 2);
    case(
        r#"
    builtin.function public extern("C") @locals(%a: i128, %b: i128, %c: i1) -> (i128, i64, i128) locals(i128, i64, i128) {
        builtin.store_local 0, %a;
        %k = arith.constant 5 : i64;
        builtin.store_local 1, %k;
        builtin.store_local 2, %b;
        cf.cond_br %c ^then, ^else : (i1);
    ^then:
        %s = arith.add %a, %b;
        builtin.store_local 0, %s;
        cf.br ^join;
    ^else:
        %old = builtin.load_local 2 : i128;
        %t = arith.band %old, %a;
        builtin.store_local 2, %t;
        %m = arith.constant 9 : i64;
        builtin.store_local 1, %m;
        cf.br ^join;
    ^join:
        %x = builtin.load_local 0 : i128;
        %y = builtin.load_local 1 : i64;
        %z = builtin.load_local 2 : i128;
        builtin.ret %x, %y, %z : (i128, i64, i128);
    };"#,
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @locals(%11: i64, %12: i64, %13: i64, %14: i64, %15: i1) -> (i64, i64, i64, i64, i64) locals(i64, i64, i64, i64, i64) {
                    builtin.store_local 0, %11;
                    builtin.store_local 1, %12;
                    %3 = arith.constant 5 : i64;
                    builtin.store_local 2, %3;
                    builtin.store_local 3, %13;
                    builtin.store_local 4, %14;
                    cf.cond_br %15 ^block3, ^block4 : (i1);
                ^block3:
                    %18 = arith.add %11, %13;
                    %19 = arith.cmpi ult %18, %11;
                    %20 = arith.zext %19 <{ ty = #builtin.type<i64> }>;
                    %21 = arith.add %12, %14;
                    %22 = arith.add %21, %20;
                    builtin.store_local 0, %18;
                    builtin.store_local 1, %22;
                    cf.br ^block5;
                ^block4:
                    %24 = builtin.load_local 3 : i64;
                    %25 = builtin.load_local 4 : i64;
                    %27 = arith.band %24, %11;
                    %28 = arith.band %25, %12;
                    builtin.store_local 3, %27;
                    builtin.store_local 4, %28;
                    %7 = arith.constant 9 : i64;
                    builtin.store_local 2, %7;
                    cf.br ^block5;
                ^block5:
                    %30 = builtin.load_local 0 : i64;
                    %31 = builtin.load_local 1 : i64;
                    %9 = builtin.load_local 2 : i64;
                    %33 = builtin.load_local 3 : i64;
                    %34 = builtin.load_local 4 : i64;
                    builtin.ret %30, %31, %9, %33, %34 : (i64, i64, i64, i64, i64);
                };
            };"#]],
    );
}

/// `builtin.ret_imm` of an `i128` becomes the constants of its halves and a return of them; a
/// constant of every bit set becomes two of every bit set.
#[test]
fn a_return_of_an_immediate() {
    let inputs = vec![(String::from("immediate"), vec![0]), (String::from("immediate"), vec![1])];
    case(
        r#"
    builtin.function public extern("C") @immediate(%c: i1) -> i128 {
        cf.cond_br %c ^then, ^else : (i1);
    ^then:
        builtin.ret_imm 340282366920938463463374607431768211455 : i128;
    ^else:
        %k = arith.constant 18446744073709551616 : i128;
        builtin.ret %k : (i128);
    };"#,
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @immediate(%0: i1) -> (i64, i64) {
                    cf.cond_br %0 ^block3, ^block4 : (i1);
                ^block3:
                    %2 = arith.constant -1 : i64;
                    %3 = arith.constant -1 : i64;
                    builtin.ret %2, %3 : (i64, i64);
                ^block4:
                    %4 = arith.constant 0 : i64;
                    %5 = arith.constant 1 : i64;
                    builtin.ret %4, %5 : (i64, i64);
                };
            };"#]],
    );
}

/// A partial conversion in which `arith.bor` and `arith.cmpi` have neither a rule nor a pattern:
/// they stay over `i128`, and read their operands through joins (zero-extensions, a shift and an
/// or) of the halves; the add that reads the `bor` reads it through a split (truncations and a
/// shift).
#[test]
fn a_partial_conversion_keeps_what_it_does_not_convert() {
    let inputs = table("partial", 2, &[]);
    let setup = Setup {
        family: Family::Wide,
        mode: Mode::Partial,
        kinds: vec![Kind::Constant, Kind::Add, Kind::And, Kind::Xor],
    };
    case_with(
        &setup,
        r#"
    builtin.function public extern("C") @partial(%a: i128, %b: i128) -> (i128, i1) {
        %s = arith.add %a, %b;
        %o = arith.bor %s, %b;
        %t = arith.add %o, %a;
        %e = arith.cmpi eq %t, %s;
        builtin.ret %t, %e : (i128, i1);
    };"#,
        &inputs,
        expect![[r#"
            builtin.module public @m {
                builtin.function public extern("C") @partial(%6: i64, %7: i64, %8: i64, %9: i64) -> (i64, i64, i1) {
                    %26 = arith.zext %8 <{ ty = #builtin.type<i128> }>;
                    %27 = arith.zext %9 <{ ty = #builtin.type<i128> }>;
                    %28 = arith.constant 64 : i64;
                    %29 = arith.zext %28 <{ ty = #builtin.type<i128> }>;
                    %30 = arith.shl %27, %29;
                    %31 = arith.bor %30, %26;
                    %12 = arith.add %6, %8;
                    %13 = arith.cmpi ult %12, %6;
                    %14 = arith.zext %13 <{ ty = #builtin.type<i64> }>;
                    %15 = arith.add %7, %9;
                    %16 = arith.add %15, %14;
                    %32 = arith.zext %12 <{ ty = #builtin.type<i128> }>;
                    %33 = arith.zext %16 <{ ty = #builtin.type<i128> }>;
                    %34 = arith.constant 64 : i64;
                    %35 = arith.zext %34 <{ ty = #builtin.type<i128> }>;
                    %36 = arith.shl %33, %35;
                    %37 = arith.bor %36, %32;
                    %3 = arith.bor %37, %31;
                    %38 = arith.trunc %3 <{ ty = #builtin.type<i64> }>;
                    %39 = arith.constant 64 : i64;
                    %40 = arith.zext %39 <{ ty = #builtin.type<i128> }>;
                    %41 = arith.shr %3, %40;
                    %42 = arith.trunc %41 <{ ty = #builtin.type<i64> }>;
                    %20 = arith.add %38, %6;
                    %21 = arith.cmpi ult %20, %38;
                    %22 = arith.zext %21 <{ ty = #builtin.type<i64> }>;
                    %23 = arith.add %42, %7;
                    %24 = arith.add %23, %22;
                    %43 = arith.zext %20 <{ ty = #builtin.type<i128> }>;
                    %44 = arith.zext %24 <{ ty = #builtin.type<i128> }>;
                    %45 = arith.constant 64 : i64;
                    %46 = arith.zext %45 <{ ty = #builtin.type<i128> }>;
                    %47 = arith.shl %44, %46;
                    %48 = arith.bor %47, %43;
                    %5 = arith.cmpi eq %48, %37;
                    builtin.ret %20, %24, %5 : (i64, i64, i1);
                };
            };"#]],
    );
}
