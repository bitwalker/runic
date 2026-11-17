//! Every operation [encode](crate::emit::instr::encode) writes, at every type it takes, in a
//! function of one operation built here, run by the evaluator and by `wasmi` on the emitted module
//! ([agree](crate::tests::exec::agree)) on the edge values of 6c's tests. For the operations the
//! evaluator does not interpret (the float operations of the `wasm` dialect that no lowering
//! produces, the trapping truncations, `memory_size`, `memory_grow`, `memory_init`, `data_drop`,
//! the table and reference operations), the expected result is stated here, from the
//! specification, and the emitted module is checked against it.

use alloc::{format, rc::Rc, string::String, vec, vec::Vec};

use runic_ir::{Context, OperationRef, dialects::builtin::ModuleRef};

use crate::tests::{
    LAYOUT,
    eval::{Outcome, Val},
    exec::{Cache, PAGE, Setup, agree, agree_cached, agree_in, emitted, same},
    function_named,
    int::{edges, product},
    parse_module,
};

/// A function `@f` built in a test, with what keeps it alive.
struct Built {
    _context: Rc<Context>,
    module: ModuleRef,
    function: OperationRef,
}

impl Built {
    /// The function `@name` of the module.
    fn named(&self, name: &str) -> OperationRef {
        function_named(self.module, name)
    }
}

/// A module with a one-page memory `@mem`, the declarations `prelude`, and the function `@f` of
/// the signature `signature` (written after the name: `(%a: i32) -> i32`) and the body `body`.
#[track_caller]
fn build(prelude: &str, signature: &str, body: &str) -> Built {
    let text = format!(
        "builtin.module public @m {LAYOUT} {{\nwasm.memory public @mem : \
         1;\n{prelude}\nbuiltin.function public extern(\"wasm\") @f{signature} \
         {{\n{body}\n}};\n}};"
    );
    let (context, module) = parse_module(&text);
    let function = function_named(module, "f");
    Built {
        _context: context,
        module,
        function,
    }
}

/// `@f(%a: a, %b: b, ..) -> result` whose body is `%r = wasm.{op} %a, %b ..{suffix}` and its
/// return.
#[track_caller]
fn op_function(op: &str, params: &[&str], result: &str, suffix: &str) -> Built {
    let names = ["%a", "%b", "%c"];
    let signature = format!(
        "({}) -> {result}",
        params
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("{}: {ty}", names[index]))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let operands = names[..params.len()].join(", ");
    let body = format!("%r = wasm.{op} {operands}{suffix};\nbuiltin.ret %r : ({result});");
    build("", &signature, &body)
}

/// Runs `built` on every one of `cases` under `setup` both ways; the number of cases.
#[track_caller]
fn agree_on(setup: &Setup, built: &Built, cases: &[Vec<Val>]) -> usize {
    let mut cache = Cache::default();
    for case in cases {
        agree_cached(&mut cache, setup, built.function, case);
    }
    cases.len()
}

/// The edge values of the integer type of width `width` (32 or 64), as 6c's tests take them.
fn ints(width: u32) -> Vec<Val> {
    edges(width)
        .into_iter()
        .map(|value| Val::new(value.unsigned() as u64, width))
        .collect()
}

/// The float values tried at width `width`: the zeros, ones, halves, the limits, a subnormal, the
/// infinities, a quiet NaN and values at the bounds of the integer conversions.
fn floats(width: u32) -> Vec<Val> {
    let values: [f64; 21] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        2.5,
        -2.5,
        255.9,
        2_147_483_647.0,
        2_147_483_648.0,
        -2_147_483_649.0,
        4_294_967_296.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
        1.0e300,
        f64::MIN_POSITIVE / 4.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    values
        .iter()
        .map(|value| {
            if width == 32 {
                Val::f32(*value as f32)
            } else {
                Val::f64(*value)
            }
        })
        .collect()
}

/// The values of the Wasm type `ty`.
fn values_of(ty: &str) -> Vec<Val> {
    match ty {
        "i32" => ints(32),
        "i64" => ints(64),
        "f32" => floats(32),
        _ => floats(64),
    }
}

/// The pairs of the values of `ty`.
fn pairs(ty: &str) -> Vec<Vec<Val>> {
    product(&[values_of(ty), values_of(ty)])
}

/// The singles of the values of `ty`.
fn singles(ty: &str) -> Vec<Vec<Val>> {
    values_of(ty).into_iter().map(|value| vec![value]).collect()
}

#[test]
fn integer_arithmetic_agrees_at_both_widths() {
    let mut count = 0;
    for ty in ["i32", "i64"] {
        for op in [
            "add", "sub", "mul", "div_s", "div_u", "rem_s", "rem_u", "and", "or", "xor", "shl",
            "shr_s", "shr_u", "rotl", "rotr",
        ] {
            count += agree_on(&Setup::default(), &op_function(op, &[ty, ty], ty, ""), &pairs(ty));
        }
        for op in ["clz", "ctz", "popcnt", "extend8_s", "extend16_s"] {
            count += agree_on(&Setup::default(), &op_function(op, &[ty], ty, ""), &singles(ty));
        }
    }
    count += agree_on(
        &Setup::default(),
        &op_function("extend32_s", &["i64"], "i64", ""),
        &singles("i64"),
    );
    assert!(count > 2000, "{count}");
}

#[test]
fn comparisons_agree_at_every_type() {
    let mut count = 0;
    for ty in ["i32", "i64"] {
        for op in ["eq", "ne", "lt_s", "lt_u", "gt_s", "gt_u", "le_s", "le_u", "ge_s", "ge_u"] {
            count +=
                agree_on(&Setup::default(), &op_function(op, &[ty, ty], "i32", ""), &pairs(ty));
        }
        count += agree_on(&Setup::default(), &op_function("eqz", &[ty], "i32", ""), &singles(ty));
    }
    for ty in ["f32", "f64"] {
        for op in ["eq", "ne", "lt", "gt", "le", "ge"] {
            count +=
                agree_on(&Setup::default(), &op_function(op, &[ty, ty], "i32", ""), &pairs(ty));
        }
    }
    assert!(count > 2000, "{count}");
}

#[test]
fn float_arithmetic_the_evaluator_knows_agrees() {
    let mut count = 0;
    for ty in ["f32", "f64"] {
        for op in ["add", "sub", "mul", "div"] {
            count += agree_on(&Setup::default(), &op_function(op, &[ty, ty], ty, ""), &pairs(ty));
        }
        count += agree_on(&Setup::default(), &op_function("neg", &[ty], ty, ""), &singles(ty));
    }
    assert!(count > 1500, "{count}");
}

#[test]
fn conversions_agree_for_every_pair_of_types() {
    let mut count = 0;
    let conversions: [(&str, &str, &str); 23] = [
        ("wrap", "i64", "i32"),
        ("extend_s", "i32", "i64"),
        ("extend_u", "i32", "i64"),
        ("trunc_sat_s", "f32", "i32"),
        ("trunc_sat_s", "f64", "i32"),
        ("trunc_sat_s", "f32", "i64"),
        ("trunc_sat_s", "f64", "i64"),
        ("trunc_sat_u", "f32", "i32"),
        ("trunc_sat_u", "f64", "i32"),
        ("trunc_sat_u", "f32", "i64"),
        ("trunc_sat_u", "f64", "i64"),
        ("convert_s", "i32", "f32"),
        ("convert_s", "i64", "f32"),
        ("convert_s", "i32", "f64"),
        ("convert_s", "i64", "f64"),
        ("convert_u", "i32", "f32"),
        ("convert_u", "i64", "f32"),
        ("convert_u", "i32", "f64"),
        ("convert_u", "i64", "f64"),
        ("promote", "f32", "f64"),
        ("demote", "f64", "f32"),
        ("reinterpret", "f32", "i32"),
        ("reinterpret", "i64", "f64"),
    ];
    for (op, from, to) in conversions {
        let suffix = format!(" <{{ ty = {to} }}>");
        count +=
            agree_on(&Setup::default(), &op_function(op, &[from], to, &suffix), &singles(from));
    }
    for (from, to) in [("f64", "i64"), ("i32", "f32")] {
        let suffix = format!(" <{{ ty = {to} }}>");
        count += agree_on(
            &Setup::default(),
            &op_function("reinterpret", &[from], to, &suffix),
            &singles(from),
        );
    }
    assert!(count > 300, "{count}");
}

#[test]
fn constants_and_select_agree_at_every_type() {
    let mut count = 0;
    for value in ints(32) {
        let built = build(
            "",
            "() -> i32",
            &format!("%r = wasm.const {} : i32;\nbuiltin.ret %r : (i32);", value.bits),
        );
        count += agree_on(&Setup::default(), &built, &[vec![]]);
    }
    for value in ints(64) {
        let built = build(
            "",
            "() -> i64",
            &format!("%r = wasm.const {} : i64;\nbuiltin.ret %r : (i64);", value.bits),
        );
        count += agree_on(&Setup::default(), &built, &[vec![]]);
    }
    for (ty, text) in [("f32", "-1.5"), ("f64", "2.5"), ("f64", "0x7ff8000000000001")] {
        let built = build(
            "",
            &format!("() -> {ty}"),
            &format!("%r = wasm.constf {text} : {ty};\nbuiltin.ret %r : ({ty});"),
        );
        count += agree_on(&Setup::default(), &built, &[vec![]]);
    }
    for ty in ["i32", "i64", "f32", "f64"] {
        let built = op_function("select", &["i32", ty, ty], ty, "");
        let mut cases = Vec::new();
        for condition in [0, 1, 7, u32::MAX] {
            for pair in pairs(ty).into_iter().step_by(7) {
                cases.push(vec![Val::i32(condition), pair[0], pair[1]]);
            }
        }
        count += agree_on(&Setup::default(), &built, &cases);
    }
    assert!(count > 100, "{count}");
}

/// The bits of `value` read as a float of its width.
fn as_f64(value: Val) -> f64 {
    if value.width == 32 {
        f64::from(f32::from_bits(value.bits as u32))
    } else {
        f64::from_bits(value.bits)
    }
}

/// The float `value` at the width `width`.
fn float(width: u32, value: f64) -> Val {
    if width == 32 {
        Val::f32(value as f32)
    } else {
        Val::f64(value)
    }
}

/// Checks the emitted `built` against `expected` on every case.
#[track_caller]
fn expect_on(built: &Built, cases: &[Vec<Val>], expected: impl Fn(&[Val]) -> Outcome) -> usize {
    for case in cases {
        let ran = emitted(&Setup::default(), built.function, &[], case);
        let want = expected(case);
        assert!(
            same(&ran.outcome, &want),
            "{case:?}: wasmi gives {:?}, expected {want:?}",
            ran.outcome
        );
    }
    cases.len()
}

/// `fmin` and `fmax` of the specification: a NaN if either operand is one; of two zeros, `-0` for
/// the minimum and `+0` for the maximum if either is that one.
fn min_max(a: f64, b: f64, max: bool) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        let negative = if max {
            a.is_sign_negative() && b.is_sign_negative()
        } else {
            a.is_sign_negative() || b.is_sign_negative()
        };
        if negative { -0.0 } else { 0.0 }
    } else if max {
        a.max(b)
    } else {
        a.min(b)
    }
}

#[test]
fn float_operations_the_evaluator_does_not_interpret_give_the_specification_s_results() {
    let mut count = 0;
    for (ty, width) in [("f32", 32), ("f64", 64)] {
        for op in ["min", "max", "copysign"] {
            let built = op_function(op, &[ty, ty], ty, "");
            count += expect_on(&built, &pairs(ty), |case| {
                let (a, b) = (as_f64(case[0]), as_f64(case[1]));
                Outcome::Values(vec![match op {
                    "min" => float(width, min_max(a, b, false)),
                    "max" => float(width, min_max(a, b, true)),
                    _ => {
                        // The magnitude of the first, the sign bit of the second.
                        let sign = 1u64 << (width - 1);
                        Val::float_bits((case[0].bits & !sign) | (case[1].bits & sign), width)
                    }
                }])
            });
        }
        for op in ["abs", "sqrt", "ceil", "floor", "trunc", "nearest"] {
            let built = op_function(op, &[ty], ty, "");
            count += expect_on(&built, &singles(ty), |case| {
                let a = as_f64(case[0]);
                Outcome::Values(vec![match op {
                    "abs" => Val::float_bits(case[0].bits & !(1u64 << (width - 1)), width),
                    // `sqrt` of an `f32` in `f64` is exact before the rounding to `f32`.
                    "sqrt" => float(width, a.sqrt()),
                    "ceil" => float(width, a.ceil()),
                    "floor" => float(width, a.floor()),
                    "trunc" => float(width, a.trunc()),
                    _ => float(width, a.round_ties_even()),
                }])
            });
        }
    }
    assert!(count > 1000, "{count}");
}

#[test]
fn trapping_truncations_trap_on_a_nan_and_out_of_range_and_truncate_otherwise() {
    let mut count = 0;
    for from in ["f32", "f64"] {
        for (to, width) in [("i32", 32), ("i64", 64)] {
            for (op, signed) in [("trunc_s", true), ("trunc_u", false)] {
                let built = op_function(op, &[from], to, &format!(" <{{ ty = {to} }}>"));
                count += expect_on(&built, &singles(from), |case| {
                    let t = as_f64(case[0]).trunc();
                    let bound = 2f64.powi(width as i32 - i32::from(signed));
                    let low = if signed { -bound } else { 0.0 };
                    if t.is_nan() || t < low || t >= bound {
                        Outcome::Trap
                    } else if signed {
                        Outcome::Values(vec![Val::new(t as i64 as u64, width)])
                    } else {
                        Outcome::Values(vec![Val::new(t as u64, width)])
                    }
                });
            }
        }
    }
    assert!(count > 100, "{count}");
}

/// The setup of the memory tests: a pattern in the first 32 bytes and the last 16 of the page.
fn patterned() -> Setup {
    let low = (0u8..32).map(|byte| byte.wrapping_mul(37) ^ 0x80).collect::<Vec<_>>();
    let high = (0u8..16).map(|byte| 0xf0 | byte).collect::<Vec<_>>();
    Setup::default()
        .write(0, &low)
        .write((PAGE - 16) as u32, &high)
        .comparing_memory()
}

/// The addresses the memory tests try: inside, at the end of the page and past it.
fn addresses() -> Vec<Vec<Val>> {
    let page = PAGE as u32;
    [
        0,
        1,
        3,
        7,
        page - 16,
        page - 9,
        page - 8,
        page - 4,
        page - 2,
        page - 1,
        page,
        u32::MAX,
    ]
    .into_iter()
    .map(|address| vec![Val::i32(address)])
    .collect()
}

#[test]
fn every_load_agrees_at_offsets_and_across_the_end_of_memory() {
    let mut count = 0;
    let loads = [
        ("i32", ""),
        ("i64", ""),
        ("f32", ""),
        ("f64", ""),
        ("i32", " as i8 signed"),
        ("i32", " as i8 unsigned"),
        ("i32", " as i16 signed"),
        ("i32", " as i16 unsigned"),
        ("i64", " as i8 signed"),
        ("i64", " as i8 unsigned"),
        ("i64", " as i16 signed"),
        ("i64", " as i16 unsigned"),
        ("i64", " as i32 signed"),
        ("i64", " as i32 unsigned"),
    ];
    for (ty, access) in loads {
        for memarg in ["", " <{ offset = 8 }>", " <{ offset = 3, align = 1 }>"] {
            let body = format!(
                "%r = wasm.load @mem, %a{access}{memarg} : {ty};\nbuiltin.ret %r : ({ty});"
            );
            let built = build("", &format!("(%a: i32) -> {ty}"), &body);
            count += agree_on(&patterned(), &built, &addresses());
        }
    }
    assert_eq!(count, 14 * 3 * 12);
}

#[test]
fn every_store_agrees_at_offsets_and_across_the_end_of_memory() {
    let mut count = 0;
    let stores = [
        ("i32", "", Val::i32(0x8182_8384)),
        ("i64", "", Val::i64(0x9192_9394_9596_9798)),
        ("f32", "", Val::f32(-1.5)),
        ("f64", "", Val::f64(2.5)),
        ("i32", " as i8", Val::i32(0x8182_8384)),
        ("i32", " as i16", Val::i32(0x8182_8384)),
        ("i64", " as i8", Val::i64(0x9192_9394_9596_9798)),
        ("i64", " as i16", Val::i64(0x9192_9394_9596_9798)),
        ("i64", " as i32", Val::i64(0x9192_9394_9596_9798)),
    ];
    for (ty, access, value) in stores {
        for memarg in ["", " <{ offset = 8 }>"] {
            let body = format!("wasm.store @mem, %a, %v{access}{memarg};\nbuiltin.ret;");
            let built = build("", &format!("(%a: i32, %v: {ty})"), &body);
            let cases = addresses()
                .into_iter()
                .map(|mut case| {
                    case.push(value);
                    case
                })
                .collect::<Vec<_>>();
            count += agree_on(&patterned(), &built, &cases);
        }
    }
    assert_eq!(count, 9 * 2 * 12);
}

#[test]
fn memory_copy_and_fill_agree_on_overlapping_ranges_and_out_of_bounds() {
    let page = PAGE as u32;
    let copy = build(
        "",
        "(%d: i32, %s: i32, %n: i32)",
        "wasm.memory_copy @mem, @mem, %d, %s, %n;\nbuiltin.ret;",
    );
    let fill = build(
        "",
        "(%d: i32, %v: i32, %n: i32)",
        "wasm.memory_fill @mem, %d, %v, %n;\nbuiltin.ret;",
    );
    let i32s = |values: [u32; 3]| values.iter().map(|value| Val::i32(*value)).collect::<Vec<_>>();
    // Forward and backward overlaps, disjoint ranges, empty ranges at and past the end, and
    // ranges that end past it.
    let copies = [
        [4, 0, 16],
        [0, 4, 16],
        [16, 0, 8],
        [page - 8, 0, 8],
        [page, 0, 0],
        [page + 1, 0, 0],
        [page - 8, 0, 9],
        [0, page - 4, 5],
        [0, 0, u32::MAX],
    ];
    let fills = [
        [0, 0x1ab, 16],
        [page - 4, 0xcd, 4],
        [page - 4, 0xcd, 5],
        [page, 0, 0],
        [page + 1, 0, 0],
        [u32::MAX, 1, 1],
    ];
    let copies = copies.map(i32s);
    let fills = fills.map(i32s);
    let count = agree_on(&patterned(), &copy, &copies) + agree_on(&patterned(), &fill, &fills);
    assert_eq!(count, 15);
}

#[test]
fn memory_size_and_grow_give_pages() {
    let built = build(
        "",
        "(%d: i32) -> (i32, i32, i32)",
        "%s = wasm.memory_size @mem;\n%g = wasm.memory_grow @mem, %d;\n%t = wasm.memory_size \
         @mem;\nbuiltin.ret %s, %g, %t : (i32, i32, i32);",
    );
    let run = |delta: u32| emitted(&Setup::default(), built.function, &[], &[Val::i32(delta)]);
    let i32s = |values: [u32; 3]| Outcome::Values(values.map(Val::i32).to_vec());
    assert_eq!(run(2).outcome, i32s([1, 1, 3]));
    assert_eq!(run(0).outcome, i32s([1, 1, 1]));
    // A memory has at most 65536 pages: growing past that fails with -1 and changes nothing.
    assert_eq!(run(65536).outcome, i32s([1, u32::MAX, 1]));
}

#[test]
fn memory_init_copies_passive_data_and_data_drop_empties_it() {
    let prelude = "builtin.global_variable public @seg : i32;";
    let init = build(
        prelude,
        "(%d: i32, %s: i32, %n: i32) -> i32",
        "wasm.memory_init @mem, @seg, %d, %s, %n;\n%r = wasm.load @mem, %d as i8 unsigned : \
         i32;\nbuiltin.ret %r : (i32);",
    );
    let setup = Setup::default().passive("seg", &[10, 20, 30, 40]);
    let run = |values: [u32; 3]| emitted(&setup, init.function, &[], &values.map(Val::i32)).outcome;
    assert_eq!(run([100, 1, 2]), Outcome::Values(vec![Val::i32(20)]));
    // A range past the end of the data traps, an empty one at its end does not.
    assert_eq!(run([100, 3, 2]), Outcome::Trap);
    assert_eq!(run([100, 4, 0]), Outcome::Values(vec![Val::i32(0)]));
    let dropped = build(
        prelude,
        "(%n: i32) -> i32",
        "%z = wasm.const 0 : i32;\nwasm.data_drop @seg;\nwasm.memory_init @mem, @seg, %z, %z, \
         %n;\nbuiltin.ret %n : (i32);",
    );
    let run = |n: u32| emitted(&setup, dropped.function, &[], &[Val::i32(n)]).outcome;
    // A dropped segment is empty: copying nothing from it is allowed, anything else traps.
    assert_eq!(run(0), Outcome::Values(vec![Val::i32(0)]));
    assert_eq!(run(1), Outcome::Trap);
}

#[test]
fn data_addresses_are_the_setup_s_plus_their_offset() {
    let prelude = "builtin.global_variable public @d : i32;";
    let built = build(
        prelude,
        "() -> (i32, i32, i32)",
        "%a = wasm.data_addr @d;\n%b = wasm.data_addr @d + 12;\n%c = wasm.data_addr @d - 4;\n%v = \
         wasm.load @mem, %b : i32;\nbuiltin.ret %a, %c, %v : (i32, i32, i32);",
    );
    let setup = patterned().data("d", 4);
    let outcome = agree_in(&setup, built.function, &[]);
    assert!(
        matches!(outcome, Outcome::Values(ref values) if values[..2] == [Val::i32(4), Val::i32(0)])
    );
}

#[test]
fn globals_are_read_and_written_with_their_final_values_compared() {
    let prelude = "wasm.global private declaration @g : mut i32;\nwasm.global private declaration \
                   @h : mut f64;\nwasm.global private declaration @k : i64;";
    let built = build(
        prelude,
        "(%x: i32, %y: f64) -> (i32, i64)",
        "%old = wasm.global_get @g : i32;\n%new = wasm.add %old, %x;\nwasm.global_set @g, \
         %new;\nwasm.global_set @h, %y;\n%k = wasm.global_get @k : i64;\nbuiltin.ret %old, %k : \
         (i32, i64);",
    );
    let setup = Setup::default()
        .global("g", Val::i32(40))
        .global("h", Val::f64(0.0))
        .global("k", Val::i64(u64::MAX - 1));
    assert_eq!(agree_on(&setup, &built, &[vec![Val::i32(2), Val::f64(-0.5)]]), 1);
}

#[test]
fn locals_start_at_zero_and_keep_what_is_stored() {
    let text = "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, \
                ptr = 32> } {\nbuiltin.function public extern(\"wasm\") @f(%x: i64) -> (i64, i64, \
                f32) locals(i64, f32) {\n%z = builtin.load_local 0 : i64;\nbuiltin.store_local 0, \
                %x;\n%w = builtin.load_local 0 : i64;\n%f = builtin.load_local 1 : \
                f32;\nbuiltin.ret %z, %w, %f : (i64, i64, f32);\n};\n};";
    let (_context, module) = parse_module(text);
    let function = function_named(module, "f");
    let mut cache = Cache::default();
    let outcome = agree_cached(&mut cache, &Setup::default(), function, &[Val::i64(9)]);
    assert_eq!(outcome, Outcome::Values(vec![Val::i64(0), Val::i64(9), Val::f32(0.0)]));
}

/// The callees of the call tests: `@inc` and `@double` of `i32`, `@wide` of `i64`, and `@pair`,
/// which gives two results.
const CALLEES: &str =
    "wasm.table private declaration @t : 4 x !wasm.funcref;\nbuiltin.function public \
     extern(\"wasm\") @inc(%x: i32) -> i32 {\n%one = wasm.const 1 : i32;\n%r = wasm.add %x, \
     %one;\nbuiltin.ret %r : (i32);\n};\nbuiltin.function public extern(\"wasm\") @double(%x: \
     i32) -> i32 {\n%r = wasm.add %x, %x;\nbuiltin.ret %r : (i32);\n};\nbuiltin.function public \
     extern(\"wasm\") @wide(%x: i64) -> i64 {\nbuiltin.ret %x : (i64);\n};\nbuiltin.function \
     public extern(\"wasm\") @pair(%x: i32) -> (i32, i32) {\n%d = wasm.add %x, %x;\nbuiltin.ret \
     %x, %d : (i32, i32);\n};";

#[test]
fn a_call_with_two_results_agrees() {
    let built = build(
        CALLEES,
        "(%x: i32) -> i32",
        "%a, %b = builtin.call @pair(%x) : extern(\"wasm\") (i32) -> (i32, i32);\n%s = wasm.sub \
         %b, %a;\n%t = builtin.call @inc(%s) : extern(\"wasm\") (i32) -> i32;\nbuiltin.ret %t : \
         (i32);",
    );
    assert_eq!(agree_on(&Setup::default(), &built, &singles("i32")), 9);
}

#[test]
fn call_indirect_calls_its_slot_and_traps_past_the_table_on_a_null_slot_and_a_wrong_type() {
    let built = build(
        CALLEES,
        "(%i: i32, %x: i32) -> i32",
        "%k = wasm.func_index @inc;\n%j = wasm.func_index @double;\n%u = wasm.add %k, %j;\n%r = \
         wasm.call_indirect @t, %i(%x) : extern(\"wasm\") (i32) -> i32;\n%s = wasm.add %r, \
         %u;\nbuiltin.ret %s : (i32);",
    );
    // The evaluator gives `@inc` slot 0 and `@double` slot 1; the table has four slots, so 2 and
    // 3 are null (the evaluator, whose table has two, traps past it), and 4 is past the table.
    let cases = [0, 1, 2, 3, 4, u32::MAX]
        .into_iter()
        .map(|i| vec![Val::i32(i), Val::i32(20)])
        .collect::<Vec<_>>();
    agree_on(&Setup::default(), &built, &cases);
    let outcome = |i: u32| {
        emitted(
            &Setup::default(),
            built.function,
            &[built.named("inc"), built.named("double")],
            &[Val::i32(i), Val::i32(20)],
        )
        .outcome
    };
    assert_eq!(outcome(0), Outcome::Values(vec![Val::i32(22)]));
    assert_eq!(outcome(1), Outcome::Values(vec![Val::i32(41)]));
    assert_eq!(outcome(2), Outcome::Trap);
    // A function of another type in the slot: `call_indirect` checks the type and traps.
    let wrong = build(
        CALLEES,
        "(%x: i32) -> i32",
        "%k = wasm.func_index @wide;\n%r = wasm.call_indirect @t, %k(%x) : extern(\"wasm\") (i32) \
         -> i32;\nbuiltin.ret %r : (i32);",
    );
    let ran = emitted(&Setup::default(), wrong.function, &[wrong.named("wide")], &[Val::i32(1)]);
    assert_eq!(ran.outcome, Outcome::Trap);
}

#[test]
fn the_tail_forms_of_calls_agree() {
    let direct = build(
        CALLEES,
        "(%x: i32) -> i32",
        "builtin.tail_call @double(%x) : extern(\"wasm\") (i32) -> i32;",
    );
    let indirect = build(
        CALLEES,
        "(%i: i32, %x: i32) -> i32",
        "%k = wasm.func_index @double;\n%j = wasm.func_index @inc;\n%u = wasm.add %k, \
         %j;\nwasm.return_call_indirect @t, %i(%x) : extern(\"wasm\") (i32) -> i32;",
    );
    let cases = [0, 1, 2, 9]
        .into_iter()
        .map(|i| vec![Val::i32(i), Val::i32(7)])
        .collect::<Vec<_>>();
    let count = agree_on(&Setup::default(), &direct, &singles("i32"))
        + agree_on(&Setup::default(), &indirect, &cases);
    assert_eq!(count, 13);
}

#[test]
fn unreachable_traps_and_an_unused_trapping_result_still_traps() {
    let trap = build("", "() -> i32", "wasm.unreachable;");
    let dropped = build(
        "",
        "(%a: i32, %b: i32) -> i32",
        "%q = wasm.div_u %a, %b;\nbuiltin.ret %a : (i32);",
    );
    assert_eq!(agree(trap.function, &[]), Outcome::Trap);
    let cases = [vec![Val::i32(7), Val::i32(0)], vec![Val::i32(7), Val::i32(2)]];
    assert_eq!(agree_on(&Setup::default(), &dropped, &cases), 2);
    assert_eq!(
        emitted(&Setup::default(), dropped.function, &[], &cases[0]).outcome,
        Outcome::Trap
    );
}

#[test]
fn table_operations_give_the_specification_s_results() {
    let prelude = format!("{CALLEES}\nwasm.table public @tab : 2 x !wasm.funcref;");
    let built = build(
        &prelude,
        "(%i: i32, %x: i32) -> (i32, i32, i32, i32, i32, i32)",
        "%size = wasm.table_size @tab;\n%f = wasm.ref_func @double;\n%null = wasm.ref_null : \
         !wasm.funcref;\n%three = wasm.const 3 : i32;\n%old = wasm.table_grow @tab, %null, \
         %three;\n%grown = wasm.table_size @tab;\nwasm.table_set @tab, %i, %f;\n%e = \
         wasm.table_get @tab, %i : !wasm.funcref;\n%isnull = wasm.ref_is_null %e;\n%one = \
         wasm.const 1 : i32;\n%two = wasm.const 2 : i32;\nwasm.table_fill @tab, %one, %f, \
         %two;\n%g = wasm.table_get @tab, %two : !wasm.funcref;\n%gnull = wasm.ref_is_null \
         %g;\n%last = wasm.const 4 : i32;\n%h = wasm.table_get @tab, %last : \
         !wasm.funcref;\n%hnull = wasm.ref_is_null %h;\nbuiltin.ret %size, %old, %grown, %isnull, \
         %gnull, %hnull : (i32, i32, i32, i32, i32, i32);",
    );
    let run = |i: u32| emitted(&Setup::default(), built.function, &[], &[Val::i32(i), Val::i32(0)]);
    let i32s = |values: [u32; 6]| Outcome::Values(values.map(Val::i32).to_vec());
    // Two slots, grown by three (the old size is 2) to five; slot `i` set, slots 1 and 2 filled,
    // slot 4 still null.
    assert_eq!(run(0).outcome, i32s([2, 2, 5, 0, 0, 1]));
    // Setting slot 5 is past the table: a trap.
    assert_eq!(run(5).outcome, Outcome::Trap);
}

#[test]
fn references_are_null_or_not_and_select_between_them_with_a_typed_select() {
    let built = build(
        CALLEES,
        "(%c: i32) -> (i32, i32, i32)",
        "%f = wasm.ref_func @inc;\n%n = wasm.ref_null : !wasm.funcref;\n%e = wasm.ref_null : \
         !wasm.externref;\n%s = wasm.select %c, %f, %n;\n%a = wasm.ref_is_null %s;\n%b = \
         wasm.ref_is_null %f;\n%x = wasm.ref_is_null %e;\nbuiltin.ret %a, %b, %x : (i32, i32, \
         i32);",
    );
    let run = |c: u32| emitted(&Setup::default(), built.function, &[], &[Val::i32(c)]).outcome;
    assert_eq!(run(1), Outcome::Values(vec![Val::i32(0), Val::i32(0), Val::i32(1)]));
    assert_eq!(run(0), Outcome::Values(vec![Val::i32(1), Val::i32(0), Val::i32(1)]));
}

/// The name of every operation of the `wasm` dialect, from the context's registry.
fn wasm_operations() -> Vec<String> {
    let context = crate::tests::context();
    let mut names = context
        .registered_dialects()
        .values()
        .flat_map(|dialect| {
            dialect.registered_ops().iter().map(|op| format!("{op}")).collect::<Vec<_>>()
        })
        .filter(|name| name.starts_with("wasm."))
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[test]
fn every_operation_of_the_dialect_is_an_instruction_control_or_a_module_entity() {
    use crate::emit::instr::{CONTROL, INSTRUCTIONS, MODULE_LEVEL};
    let names = wasm_operations();
    assert_eq!(names.len(), 96, "{names:?}");
    let unknown = names
        .iter()
        .filter(|name| {
            !INSTRUCTIONS.contains(&name.as_str())
                && !CONTROL.contains(&name.as_str())
                && !MODULE_LEVEL.contains(&name.as_str())
        })
        .collect::<Vec<_>>();
    assert!(unknown.is_empty(), "operations the emitter does not know: {unknown:?}");
    // And every name of the lists is an operation: a list does not name one that is gone.
    for name in INSTRUCTIONS.iter().chain(&CONTROL).chain(&MODULE_LEVEL) {
        assert!(
            names.iter().any(|known| known == name) || name.starts_with("builtin."),
            "`{name}` is not an operation of the dialect"
        );
    }
}
