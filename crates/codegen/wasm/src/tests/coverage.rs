//! Every operation of the source dialects is accounted for: `convert-to-wasm` has a pattern for it,
//! or it is a diagnostic by design, listed here with its reason and what the diagnostic says.
//!
//! The operations are those the context registers, so an operation added to one of the dialects
//! fails these tests until it is lowered or listed. A sweep lowers every source operation, for
//! each class of type its verifier accepts, through the four passes to Wasm form.

use alloc::{boxed::Box, collections::BTreeSet};

use runic_ir::{
    RawWalk,
    conversion::ConversionPatternSet,
    pass::{Nesting, PassManager},
};

use super::*;
use crate::{
    ConvertToWasm, WasmTarget,
    convert::{WellKnownSymbols, populate_patterns},
    type_converter,
};

/// The dialects whose every operation `convert-to-wasm` must account for.
const SOURCE_DIALECTS: [&str; 5] = ["arith", "cf", "mem", "scf", "ub"];

/// The operations of `builtin` that are not function-level: they are modules, symbols and data,
/// which `convert-to-wasm` leaves as they are (a global's initializer is not converted), and the
/// cast the conversion framework itself makes and removes.
const NOT_FUNCTION_LEVEL: [&str; 9] = [
    "builtin.bytes",
    "builtin.component",
    "builtin.function_alias",
    "builtin.global_variable",
    "builtin.module",
    "builtin.segment",
    "builtin.unrealized_conversion_cast",
    "builtin.world",
    "builtin.yield",
];

/// An operation that `convert-to-wasm` reports instead of lowering, and how.
struct ByDesign {
    /// The operation.
    op: &'static str,
    /// A function body with the operation in it, in a module with the 32-bit data layout.
    body: &'static str,
    /// Why the operation is not lowered by `convert-to-wasm`.
    reason: &'static str,
    /// What the diagnostic says.
    says: &'static str,
}

/// The operations that are diagnostics by design when they reach `convert-to-wasm`.
const BY_DESIGN: [ByDesign; 6] = [
    ByDesign {
        op: "builtin.aggregate",
        body: "%a = arith.constant 1 : i32;\n%s = builtin.aggregate %a : \
               struct<i32>;\nbuiltin.ret;",
        reason: "an aggregate value is not one Wasm value: it exists in memory or in a global \
                 variable's initializer",
        says: "`convert-to-wasm` cannot lower `builtin.aggregate`: an aggregate value in a \
               function body is not lowered; it exists in memory or in a global variable's \
               initializer",
    },
    ByDesign {
        op: "mem.alloca",
        body: "%p = mem.alloca i32;\nbuiltin.ret;",
        reason: "`wasm-lower-frames` gives each alloca a slot on the shadow stack first",
        says: "`convert-to-wasm` does not lower `mem.alloca`: run `wasm-lower-frames` first",
    },
    ByDesign {
        op: "cf.br",
        body: "cf.br ^next;\n^next:\nbuiltin.ret;",
        reason: "Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first",
        says: "unstructured control flow reached `convert-to-wasm`: run \
               `cfg-to-scf{exits=in-place}` first",
    },
    ByDesign {
        op: "cf.cond_br",
        body: "%c = arith.constant 1 : i1;\ncf.cond_br %c ^a, ^b : \
               (i1);\n^a:\nbuiltin.ret;\n^b:\nbuiltin.ret;",
        reason: "Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first",
        says: "unstructured control flow reached `convert-to-wasm`: run \
               `cfg-to-scf{exits=in-place}` first",
    },
    ByDesign {
        op: "cf.switch",
        body: "%s = arith.constant 1 : i32;\ncf.switch %s [#builtin.i32<0> -> ^a], ^b : \
               (i32);\n^a:\nbuiltin.ret;\n^b:\nbuiltin.ret;",
        reason: "Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first",
        says: "unstructured control flow reached `convert-to-wasm`: run \
               `cfg-to-scf{exits=in-place}` first",
    },
    ByDesign {
        op: "scf.pending_exit",
        body: "\"scf.pending_exit\"() : () -> ();\nbuiltin.ret;",
        reason: "`cfg-to-scf` makes it while it runs and removes it: it never verifies",
        says: "`scf.pending_exit` is internal to `cfg-to-scf` and never appears in IR",
    },
];

/// The names of the registered operations of `dialect`.
fn operations_of(context: &Context, dialect: &str) -> BTreeSet<String> {
    context
        .registered_dialects()
        .values()
        .flat_map(|d| d.registered_ops().iter().map(|op| op.to_string()).collect::<Vec<_>>())
        .filter(|name| name.split('.').next() == Some(dialect))
        .collect()
}

/// The operations `convert-to-wasm` must account for.
fn required(context: &Context) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for dialect in SOURCE_DIALECTS {
        let found = operations_of(context, dialect);
        assert!(!found.is_empty(), "the dialect `{dialect}` registers no operation");
        names.extend(found);
    }
    let builtin = operations_of(context, "builtin");
    for name in NOT_FUNCTION_LEVEL {
        assert!(builtin.contains(name), "`{name}` is not a registered operation");
    }
    names.extend(builtin.into_iter().filter(|name| !NOT_FUNCTION_LEVEL.contains(&name.as_str())));
    names
}

/// The operations some pattern of `convert-to-wasm` is rooted at.
fn with_patterns(context: &Rc<Context>) -> BTreeSet<String> {
    let module = empty_module(context);
    let target = WasmTarget::of(module.borrow().as_operation()).unwrap();
    let mut patterns = ConversionPatternSet::new(context.clone());
    let symbols = Rc::new(WellKnownSymbols::default());
    let labels = crate::convert::labels::new_labels();
    populate_patterns(&mut patterns, &type_converter(), &target, &symbols, &labels).unwrap();
    patterns
        .patterns()
        .iter()
        .filter_map(|pattern| pattern.info().root_operation())
        .map(|name| name.to_string())
        .collect()
}

/// A module with the 32-bit data layout and nothing in it, in `context`.
fn empty_module(context: &Rc<Context>) -> ModuleRef {
    let text = "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, \
                ptr = 32> } {};";
    parse_module_in(context, text).unwrap_or_else(|report| panic!("{}", describe(&report)))
}

#[test]
fn every_source_operation_has_a_pattern_or_is_a_diagnostic_by_design() {
    let context = context();
    let patterned = with_patterns(&context);
    let by_design: BTreeSet<&str> = BY_DESIGN.iter().map(|entry| entry.op).collect();
    let unaccounted: Vec<String> = required(&context)
        .into_iter()
        .filter(|name| !patterned.contains(name) && !by_design.contains(name.as_str()))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "no pattern of `convert-to-wasm` is rooted at these operations, and none is listed as a \
         diagnostic by design: {unaccounted:?}"
    );
}

#[test]
fn the_diagnostics_by_design_are_registered_operations_without_a_converting_pattern() {
    let context = context();
    let registered = required(&context);
    for entry in &BY_DESIGN {
        assert!(
            registered.contains(entry.op),
            "`{}` is not a registered source operation",
            entry.op
        );
        assert!(!entry.reason.is_empty());
    }
    // `cf.br`, `cf.cond_br` and `cf.switch` have patterns that only report; the others have none.
    let patterned = with_patterns(&context);
    for op in ["mem.alloca", "scf.pending_exit"] {
        assert!(!patterned.contains(op), "`{op}` has a pattern");
    }
}

/// The integer types of the sweep: narrow ones carried in an `i32` and in an `i64`, the widths of
/// the carriers, and `isize`.
const INTS: [&str; 8] = ["i1", "i7", "i8", "i16", "i32", "i33", "i64", "isize"];

/// The float types of the sweep.
const FLOATS: [&str; 2] = ["f32", "f64"];

/// The pointer types of the sweep: typed, opaque and to a pointer.
const POINTERS: [&str; 4] = ["ptr<i8>", "ptr<i64>", "ptr", "ptr<ptr<i8>>"];

/// A function `@f`, under `extern("C")`, of the parameters `params` and the results `results`,
/// whose body is `body`.
fn function(params: &str, results: &str, body: &str) -> String {
    format!("builtin.function public extern(\"C\") @f({params}) -> ({results}) {{\n{body}\n}};")
}

/// The programs of the sweep of the integer type `t`.
fn integer_programs(t: &str, add: &mut impl FnMut(String)) {
    let two = format!("%a: {t}, %b: {t}");
    let binary = [
        "add", "sub", "mul", "udiv", "sdiv", "urem", "srem", "band", "bor", "bxor", "shl", "shr",
        "ashr", "rotl", "rotr",
    ];
    for op in binary {
        add(function(&two, t, &format!("%r = arith.{op} %a, %b;\nbuiltin.ret %r : ({t});")));
    }
    for op in ["add", "sub", "mul"] {
        for mode in ["checked<signed>", "checked<unsigned>", "unchecked"] {
            add(function(
                &two,
                t,
                &format!(
                    "%r = arith.{op} %a, %b <{{ overflow = #builtin.overflow<{mode}> \
                     }}>;\nbuiltin.ret %r : ({t});"
                ),
            ));
        }
    }
    for op in ["add_overflowing", "sub_overflowing", "mul_overflowing", "min", "max"] {
        for signedness in ["signed", "unsigned"] {
            let flag = op.ends_with("overflowing");
            let results = if flag {
                format!("i1, {t}")
            } else {
                String::from(t)
            };
            let values = if flag { "%o, %r" } else { "%r" };
            add(function(
                &two,
                &results,
                &format!(
                    "{values} = arith.{op} %a, %b <{{ signedness = \
                     #builtin.signedness<{signedness}> }}>;\nbuiltin.ret {values} : ({results});"
                ),
            ));
        }
    }
    for op in ["sdivrem", "udivrem"] {
        add(function(
            &two,
            &format!("{t}, {t}"),
            &format!("%q, %m = arith.{op} %a, %b;\nbuiltin.ret %q, %m : ({t}, {t});"),
        ));
    }
    for op in ["bnot", "neg", "clz", "ctz", "clo", "cto", "popcnt"] {
        add(function(
            &format!("%a: {t}"),
            t,
            &format!("%r = arith.{op} %a;\nbuiltin.ret %r : ({t});"),
        ));
    }
    for predicate in ["eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"] {
        add(function(
            &two,
            "i1",
            &format!("%r = arith.cmpi {predicate} %a, %b;\nbuiltin.ret %r : (i1);"),
        ));
    }
    add(function(
        "",
        t,
        &format!("%r = arith.constant 1 : {t};\nbuiltin.ret %r : ({t});"),
    ));
    add(function("", t, &format!("builtin.ret_imm 1 : {t};")));
    for u in INTS {
        for op in ["zext", "sext", "trunc", "zext_or_trunc", "sext_or_trunc", "bitcast"] {
            add(function(
                &format!("%a: {t}"),
                u,
                &format!("%r = arith.{op} %a <{{ ty = {u} }}>;\nbuiltin.ret %r : ({u});"),
            ));
        }
    }
    for f in FLOATS {
        for op in ["sitofp", "uitofp", "bitcast"] {
            add(function(
                &format!("%a: {t}"),
                f,
                &format!("%r = arith.{op} %a <{{ ty = {f} }}>;\nbuiltin.ret %r : ({f});"),
            ));
        }
        for op in ["fptosi", "fptoui", "bitcast"] {
            add(function(
                &format!("%a: {f}"),
                t,
                &format!("%r = arith.{op} %a <{{ ty = {t} }}>;\nbuiltin.ret %r : ({t});"),
            ));
        }
    }
}

/// The programs of the sweep of the float type `f`.
fn float_programs(f: &str, add: &mut impl FnMut(String)) {
    let two = format!("%a: {f}, %b: {f}");
    for op in ["addf", "subf", "mulf", "divf", "remf", "minf", "maxf"] {
        add(function(&two, f, &format!("%r = arith.{op} %a, %b;\nbuiltin.ret %r : ({f});")));
    }
    for op in ["negf", "absf", "sqrtf", "ceilf", "floorf", "truncf", "nearestf"] {
        add(function(
            &format!("%a: {f}"),
            f,
            &format!("%r = arith.{op} %a;\nbuiltin.ret %r : ({f});"),
        ));
    }
    add(function(&two, "i1", "%r = arith.cmpf ult %a, %b;\nbuiltin.ret %r : (i1);"));
    add(function(
        "",
        f,
        &format!("%r = arith.constantf 1.5 : {f};\nbuiltin.ret %r : ({f});"),
    ));
    add(function("", f, &format!("builtin.ret_imm 1.5 : {f};")));
}

/// The programs of the sweep of the type `t` of any class: selects, poison, locals, loads and
/// stores, and comparisons.
fn value_programs(t: &str, add: &mut impl FnMut(String)) {
    add(function(
        &format!("%c: i1, %a: {t}, %b: {t}"),
        t,
        &format!("%r = cf.select %c, %a, %b;\nbuiltin.ret %r : ({t});"),
    ));
    add(function(
        "",
        t,
        &format!("%r = ub.poison <{{ value = #ub.poison<{t}> }}>;\nbuiltin.ret %r : ({t});"),
    ));
    add(format!(
        "builtin.function public extern(\"C\") @f(%a: {t}) -> {t} locals({t}) \
         {{\nbuiltin.store_local 0, %a;\n%r = builtin.load_local 0 : {t};\nbuiltin.ret %r : \
         ({t});\n}};"
    ));
    add(function(
        &format!("%p: ptr<{t}>"),
        t,
        &format!("%r = mem.load %p : {t};\nmem.store %r, %p;\nbuiltin.ret %r : ({t});"),
    ));
    add(function(
        &format!("%a: {t}, %b: {t}"),
        "i1",
        "%r = arith.cmpi eq %a, %b;\nbuiltin.ret %r : (i1);",
    ));
    add(function("", t, &format!("%r = builtin.zero : {t};\nbuiltin.ret %r : ({t});")));
}

/// The programs of the sweep: every source operation, for each class of type its verifier might
/// accept. Each is the functions of a module.
fn sweep_programs() -> Vec<String> {
    let mut programs = Vec::new();
    let mut add = |program: String| programs.push(program);
    for t in INTS {
        integer_programs(t, &mut add);
    }
    for f in FLOATS {
        float_programs(f, &mut add);
    }
    for t in INTS.iter().chain(FLOATS.iter()).chain(POINTERS.iter()) {
        value_programs(t, &mut add);
    }
    add(function("%a: i1", "", "cf.assert %a, \"x\";\nbuiltin.ret;"));
    for t in ["(i8) -> i8", "!wasm.funcref", "!wasm.externref"] {
        add(function("", t, &format!("%r = builtin.zero : {t};\nbuiltin.ret %r : ({t});")));
    }
    add(function(
        "%a: f32",
        "f64",
        "%r = arith.fpext %a <{ ty = f64 }>;\nbuiltin.ret %r : (f64);",
    ));
    add(function(
        "%a: f64",
        "f32",
        "%r = arith.fptrunc %a <{ ty = f32 }>;\nbuiltin.ret %r : (f32);",
    ));
    for t in POINTERS {
        add(function(
            &format!("%a: {t}"),
            "isize",
            "%r = mem.ptrtoint %a;\nbuiltin.ret %r : (isize);",
        ));
        add(function(
            "%a: isize",
            t,
            &format!("%r = mem.inttoptr %a : {t};\nbuiltin.ret %r : ({t});"),
        ));
    }
    add(function(
        "%a: ptr<i8>",
        "ptr<i64>",
        "%r = mem.ptrcast %a : ptr<i64>;\nbuiltin.ret %r : (ptr<i64>);",
    ));
    add(function(
        "",
        "isize, isize, isize",
        "%a = mem.sizeof struct<i8, i64>;\n%b = mem.alignof f64;\n%c = mem.offsetof struct<i8, \
         i64>, [1];\nbuiltin.ret %a, %b, %c : (isize, isize, isize);",
    ));
    add(function(
        "%p: ptr<struct<i32, array<i16; 4>>>, %i: isize",
        "ptr<i16>",
        "%q = mem.gep %p, %i, [1, 2];\nbuiltin.ret %q : (ptr<i16>);",
    ));
    add(function(
        "%d: ptr<i8>, %s: ptr<i8>, %n: isize, %v: i8",
        "",
        "mem.memcpy %d, %s, %n;\nmem.memmove %d, %s, %n;\nmem.memset %d, %v, %n;\nbuiltin.ret;",
    ));
    add(function(
        "",
        "i32",
        "%p = mem.alloca i32;\n%z = arith.constant 0 : i32;\nmem.store %z, %p;\n%v = mem.load \
         %p;\nbuiltin.ret %v : (i32);",
    ));
    add(format!(
        "builtin.global_variable private @data : i32 {{\n%v = arith.constant 7 : \
         i32;\nbuiltin.yield %v;\n}};\n{}",
        function(
            "",
            "i32",
            "%p = builtin.global_symbol @data : ptr<i32>;\n%v = mem.load %p;\nbuiltin.ret %v : \
             (i32);"
        )
    ));
    add(format!(
        "builtin.function public extern(\"C\") @g(%x: i8) -> i8 {{\nbuiltin.ret %x : \
         (i8);\n}};\n{}",
        function(
            "%x: i8",
            "i8",
            "%f = builtin.function_ref @g : (i8) -> i8;\n%y = builtin.call @g(%x) : extern(\"C\") \
             (i8) -> i8;\n%z = builtin.call_indirect %f(%y) : extern(\"C\") (i8) -> \
             i8;\nbuiltin.tail_call_indirect %f(%z) : extern(\"C\") (i8) -> i8;"
        )
    ));
    add(format!(
        "builtin.function public extern(\"C\") @g(%x: i8) -> i8 {{\nbuiltin.ret %x : \
         (i8);\n}};\n{}",
        function("%x: i8", "i8", "builtin.tail_call @g(%x) : extern(\"C\") (i8) -> i8;")
    ));
    // Structured control: `scf.if`, `scf.while` with a break and a continue, `scf.index_switch`
    // and `ub.unreachable`; and unstructured control, which `cfg-to-scf` structures first.
    add(function(
        "%c: i1, %n: i32",
        "i32",
        "%z = arith.constant 0 : i32;\n%one = arith.constant 1 : i32;\n%w = scf.while %z before \
         {\n^b(%i: i32):\n%m = arith.cmpi ult %i, %n;\nscf.condition %m, %i : (i1, i32);\n} after \
         {\n^a(%j: i32):\n%k = arith.add %j, %one <{ overflow = #builtin.overflow<wrapping> \
         }>;\nscf.if %c then {\nscf.break 1 %j : (i32);\n} else {\nscf.yield : ();\n} : (i1) -> \
         ();\n%e = arith.cmpi eq %k, %n;\nscf.if %e then {\nscf.continue 1 %k : (i32);\n} else \
         {\nscf.yield : ();\n} : (i1) -> ();\nscf.yield %k : (i32);\n} : (i32) -> (i32);\n%s = \
         scf.index_switch %w\ncase 0 {\nscf.yield %one : (i32);\n}\ndefault {\nscf.yield %w : \
         (i32);\n} : (i32);\nbuiltin.ret %s : (i32);",
    ));
    add(function(
        "%c: i1",
        "",
        "scf.if %c then {\nub.unreachable;\n} else {\nscf.yield : ();\n} : (i1) -> \
         ();\nbuiltin.ret;",
    ));
    add(function(
        "%c: i1, %s: i32, %a: i32",
        "i32",
        "cf.cond_br %c ^x, ^y : (i1);\n^x:\ncf.switch %s [#builtin.i32<0> -> ^z], ^y : \
         (i32);\n^y:\ncf.br ^z;\n^z:\nbuiltin.ret %a : (i32);",
    ));
    programs
}

/// The diagnostics by design that no pass before `convert-to-wasm` removes, so that no program of
/// the sweep lowers them.
const NEVER_LOWERED: [&str; 2] = ["builtin.aggregate", "scf.pending_exit"];

/// The names of the operations `module` holds.
fn operations_in(module: ModuleRef) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let _ = module.as_operation_ref().raw_prewalk::<runic_ir::Forward, _, _>(
        |op: runic_ir::OperationRef| {
            names.insert(op.borrow().name().to_string());
            runic_ir::WalkResult::<()>::Continue(())
        },
    );
    names
}

#[test]
fn every_source_operation_lowers_for_each_class_of_type_its_verifier_accepts() {
    // Each program that verifies goes through the four passes to Wasm form; one the verifier
    // rejects is of a type the operation does not take. Every source operation, those that are a
    // diagnostic by design only because an earlier pass removes them included, must be in some
    // program that lowered.
    let mut lowered = BTreeSet::new();
    let mut failures = Vec::new();
    for functions in sweep_programs() {
        let context = context();
        let Ok(module) = parse_module_in(&context, &module_of(&functions)) else {
            continue;
        };
        let names = operations_in(module);
        match run_pipeline(&context, module.as_operation_ref(), LOWERING) {
            Ok(()) => lowered.extend(names),
            Err(report) => failures.push(format!("{functions}\n{}", describe(&report))),
        }
    }
    assert!(failures.is_empty(), "these programs did not lower:\n{}", failures.join("\n\n"));
    let missing: Vec<String> = required(&context())
        .into_iter()
        .filter(|name| !NEVER_LOWERED.contains(&name.as_str()) && !lowered.contains(name))
        .collect();
    assert!(
        missing.is_empty(),
        "no program of the sweep lowered these operations: {missing:?}"
    );
}

/// The text of a module with one function holding `body`, with the 32-bit data layout.
fn module_with(body: &str) -> String {
    format!(
        "builtin.module public @m attributes {{ data_layout = #builtin.data_layout<little, ptr = \
         32> }} {{\n    builtin.function public extern(\"wasm\") @f() {{\n{body}\n    }};\n}};"
    )
}

#[test]
fn each_diagnostic_by_design_says_what_it_is_listed_as_saying() {
    for entry in &BY_DESIGN {
        let context = context();
        let text = module_with(entry.body);
        let error = match parse_module_in(&context, &text) {
            // The operation does not even verify.
            Err(report) => describe(&report),
            Ok(module) => {
                let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
                passes.add_pass(Box::<ConvertToWasm>::default()).unwrap();
                passes.enable_verifier(true);
                error_of(passes.run(module.as_operation_ref()))
            }
        };
        assert!(
            mentions(&error, entry.says),
            "`{}` should say\n    {}\nbut the error is\n{error}",
            entry.op,
            entry.says
        );
    }
}

#[test]
fn the_readme_lists_each_diagnostic_by_design_with_its_reason() {
    let readme = include_str!("../../README.md");
    for entry in &BY_DESIGN {
        let row = readme
            .lines()
            .find(|line| line.starts_with(&format!("| `{}` |", entry.op)))
            .unwrap_or_else(|| panic!("the README has no table row for `{}`", entry.op));
        let reason = entry.reason.replace('`', "");
        assert!(
            row.replace('`', "").contains(&reason),
            "the README row of `{}` does not give the reason \"{}\":\n{row}",
            entry.op,
            entry.reason
        );
    }
}
