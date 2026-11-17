//! Every operation of the `wasm` dialect is emitted: each appears in a module of [CORPUS], which
//! [emit_module] emits and validates, or it is an error by design, listed in [BY_DESIGN] with
//! the text its diagnostic must contain.
//!
//! The operations are those the context registers, so an operation added to the dialect fails
//! this test until a module here emits it or it is listed.

use alloc::{collections::BTreeSet, string::String, vec::Vec};

use runic_ir::{Forward, OperationRef, RawWalk, WalkResult};

use super::super::*;
use crate::emit::{EmitOptions, emit_module};

/// Modules in Wasm form that between them hold every operation of the dialect that is not an
/// error by design, each emitted and validated as a module of its own. A function's results that
/// nothing uses are dropped by the emitter.
const CORPUS: [&str; 5] = [
    // Numeric, comparison, conversion and parametric instructions.
    r#"builtin.function public extern("C") @numeric(%a: i32, %b: i32, %x: i64, %y: i64, %p: f32, %q: f32, %m: f64, %n: f64) -> i32 {
        %k = wasm.const 7 : i32;
        %kf = wasm.constf 1.5 : f64;
        %0 = wasm.add %a, %b;
        %1 = wasm.sub %a, %b;
        %2 = wasm.mul %x, %y;
        %3 = wasm.div_s %a, %b;
        %4 = wasm.div_u %x, %y;
        %5 = wasm.rem_s %a, %b;
        %6 = wasm.rem_u %x, %y;
        %7 = wasm.and %a, %b;
        %8 = wasm.or %x, %y;
        %9 = wasm.xor %a, %b;
        %10 = wasm.shl %a, %b;
        %11 = wasm.shr_s %x, %y;
        %12 = wasm.shr_u %a, %b;
        %13 = wasm.rotl %x, %y;
        %14 = wasm.rotr %a, %b;
        %15 = wasm.div %p, %q;
        %16 = wasm.min %m, %n;
        %17 = wasm.max %p, %q;
        %18 = wasm.copysign %m, %n;
        %19 = wasm.clz %a;
        %20 = wasm.ctz %x;
        %21 = wasm.popcnt %a;
        %22 = wasm.extend8_s %a;
        %23 = wasm.extend16_s %x;
        %24 = wasm.extend32_s %x;
        %25 = wasm.abs %p;
        %26 = wasm.neg %m;
        %27 = wasm.sqrt %p;
        %28 = wasm.ceil %m;
        %29 = wasm.floor %p;
        %30 = wasm.trunc %m;
        %31 = wasm.nearest %p;
        %32 = wasm.eq %a, %b;
        %33 = wasm.ne %x, %y;
        %34 = wasm.lt_s %a, %b;
        %35 = wasm.lt_u %x, %y;
        %36 = wasm.gt_s %a, %b;
        %37 = wasm.gt_u %x, %y;
        %38 = wasm.le_s %a, %b;
        %39 = wasm.le_u %x, %y;
        %40 = wasm.ge_s %a, %b;
        %41 = wasm.ge_u %x, %y;
        %42 = wasm.lt %p, %q;
        %43 = wasm.gt %m, %n;
        %44 = wasm.le %p, %q;
        %45 = wasm.ge %m, %n;
        %46 = wasm.eqz %x;
        %47 = wasm.extend_s %a <{ ty = i64 }>;
        %48 = wasm.extend_u %a <{ ty = i64 }>;
        %49 = wasm.wrap %x <{ ty = i32 }>;
        %50 = wasm.convert_s %a <{ ty = f32 }>;
        %51 = wasm.convert_u %x <{ ty = f64 }>;
        %52 = wasm.reinterpret %a <{ ty = f32 }>;
        %53 = wasm.trunc_s %p <{ ty = i32 }>;
        %54 = wasm.trunc_u %m <{ ty = i64 }>;
        %55 = wasm.trunc_sat_s %p <{ ty = i64 }>;
        %56 = wasm.trunc_sat_u %m <{ ty = i32 }>;
        %57 = wasm.promote %p <{ ty = f64 }>;
        %58 = wasm.demote %m <{ ty = f32 }>;
        %59 = wasm.select %a, %x, %y;
        builtin.ret %k : (i32);
    };"#,
    // Linear memory, data and globals.
    r#"wasm.memory public @mem : 1 attributes { wasm.export_name = "memory" };
    builtin.global_variable private @placed : array<i8; 4> {
        %bytes = builtin.bytes "01020304" : array<i8; 4>;
        builtin.yield %bytes;
    };
    builtin.global_variable private @passive : array<i8; 4> {
        %bytes = builtin.bytes "05060708" : array<i8; 4>;
        builtin.yield %bytes;
    };
    wasm.global private @counter : mut i32 { %v = wasm.const 0 : i32; builtin.yield %v; };
    wasm.global private @base : i32 { %a = wasm.data_addr @placed + 2; builtin.yield %a; };
    builtin.function public extern("C") @memory(%p: i32, %l: i64, %f: f32, %n: i32) -> i32 {
        %a = wasm.load @mem, %p : i32;
        %b = wasm.load @mem, %p as i16 signed : i64;
        wasm.store @mem, %p, %f;
        wasm.store @mem, %p, %l as i8;
        %pages = wasm.memory_size @mem;
        %old = wasm.memory_grow @mem, %pages;
        wasm.memory_copy @mem, @mem, %p, %a, %n;
        wasm.memory_fill @mem, %p, %a, %n;
        wasm.memory_init @mem, @passive, %p, %a, %n;
        wasm.data_drop @passive;
        %d = wasm.data_addr @placed;
        %v = wasm.global_get @counter : i32;
        wasm.global_set @counter, %d;
        %w = wasm.global_get @base : i32;
        builtin.ret %v : (i32);
    };"#,
    // Tables, references, element segments and indirect calls.
    r#"wasm.table private @tab : 4 x !wasm.funcref;
    wasm.table private @refs : 4 x !wasm.externref;
    wasm.global private @entry : !wasm.funcref { %f = wasm.ref_func @callee; builtin.yield %f; };
    wasm.global private @slot : i32 { %i = wasm.func_index @callee; builtin.yield %i; };
    builtin.function public extern("C") @callee(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    wasm.elem private @init : @tab at 1 [@callee];
    builtin.function public extern("C") @tables(%i: i32, %r: !wasm.funcref, %x: !wasm.externref) -> i32 {
        %e = wasm.table_get @tab, %i : !wasm.funcref;
        wasm.table_set @tab, %i, %r;
        %n = wasm.table_size @tab;
        %o = wasm.table_grow @refs, %x, %n;
        wasm.table_fill @refs, %i, %x, %n;
        %null = wasm.ref_null : !wasm.funcref;
        %f = wasm.ref_func @callee;
        %z = wasm.ref_is_null %f;
        %s = wasm.func_index @callee;
        %v = wasm.call_indirect @tab, %s(%i) : extern("C") (i32) -> i32;
        wasm.return_call_indirect @tab, %i(%v) : extern("C") (i32) -> i32;
    };"#,
    // Structured control.
    r#"builtin.function public extern("C") @control(%c: i32, %a: i32) -> i32 {
        %r = wasm.block %a {
        ^body(%v: i32):
            %q = wasm.loop %v {
            ^head(%w: i32):
                %t = wasm.if %c, %w then {
                ^then(%x: i32):
                    wasm.br 2 %x : (i32);
                } else {
                ^else(%x: i32):
                    wasm.end %x : (i32);
                } : (i32, i32) -> (i32);
                wasm.br_table %c, %t [{
                    wasm.br 0 %t : (i32);
                }] default {
                    wasm.br 1 %t : (i32);
                } : (i32, i32);
            } : (i32) -> (i32);
            wasm.end %q : (i32);
        } : (i32) -> (i32);
        builtin.ret %r : (i32);
    };"#,
    // A trap.
    r#"builtin.function public extern("C") @trap() {
        wasm.unreachable;
    };"#,
];

/// An operation of the dialect that is an error when it reaches the emitter.
struct ByDesign {
    /// The operation.
    op: &'static str,
    /// A module that holds it.
    module: &'static str,
    /// Why it is not emitted.
    reason: &'static str,
    /// What the diagnostic says.
    says: &'static str,
}

/// The operations of the dialect that are errors by design.
const BY_DESIGN: [ByDesign; 1] = [ByDesign {
    op: "wasm.tag",
    module: "wasm.tag private @exn : (i32, i64);",
    reason: "a tag is an entity of exception handling, which is outside the backend's feature set",
    says: "exceptions are not emitted in this version: a `wasm.tag` needs the exception handling \
           feature, which is outside the backend's feature set",
}];

/// The names of the operations of the `wasm` dialect that the context registers.
fn wasm_operations(context: &Context) -> BTreeSet<String> {
    context
        .registered_dialects()
        .values()
        .flat_map(|d| d.registered_ops().iter().map(|op| op.to_string()).collect::<Vec<_>>())
        .filter(|name| name.split('.').next() == Some("wasm"))
        .collect()
}

/// The names of the operations `root` holds, `root` included.
fn operations_in(root: OperationRef) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let _ = root.raw_prewalk::<Forward, _, _>(|op: OperationRef| {
        names.insert(op.borrow().name().to_string());
        WalkResult::<()>::Continue(())
    });
    names
}

#[test]
fn every_operation_of_the_wasm_dialect_is_emitted_and_validates_or_is_an_error_by_design() {
    let context = context();
    let registered = wasm_operations(&context);
    assert_eq!(registered.len(), 96, "{registered:?}");

    let mut emitted = BTreeSet::new();
    for items in CORPUS {
        let module = parse_module_in(&context, &module_of(items))
            .unwrap_or_else(|report| panic!("{}\n{items}", describe(&report)));
        emit_module(module.as_operation_ref(), &EmitOptions::default())
            .unwrap_or_else(|report| panic!("{}\n{items}", describe(&report)));
        emitted.extend(operations_in(module.as_operation_ref()));
    }

    let mut errors = BTreeSet::new();
    for entry in &BY_DESIGN {
        assert!(
            registered.contains(entry.op),
            "`{}` is not an operation of the dialect",
            entry.op
        );
        assert!(!entry.reason.is_empty());
        let module = parse_module_in(&context, &module_of(entry.module))
            .unwrap_or_else(|report| panic!("{}\n{}", describe(&report), entry.module));
        assert!(operations_in(module.as_operation_ref()).contains(entry.op));
        let error = error_of(emit_module(module.as_operation_ref(), &EmitOptions::default()));
        assert!(mentions(&error, entry.says), "`{}`: {error}", entry.op);
        assert!(mentions(&error, entry.op), "`{}`: {error}", entry.op);
        errors.insert(entry.op.to_string());
    }

    assert!(
        emitted.is_disjoint(&errors),
        "listed as errors and emitted: {:?}",
        emitted.intersection(&errors).collect::<Vec<_>>()
    );
    let missing = registered
        .iter()
        .filter(|name| !emitted.contains(*name) && !errors.contains(*name))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "no module of the corpus emits these operations, and none is listed as an error by \
         design: {missing:?}"
    );
}
