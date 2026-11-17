//! The helpers that keep carried form: each creates as few operations as its case needs.

use runic_dialect_wasm::FuncRefType;
use runic_ir::{
    OperationRef, Type, ValueRef, conversion::ConversionPatternRewriter,
    dialects::builtin::Function,
};

use super::*;
use crate::carried::{const_int, mask, sign_extend, zero};

/// Runs `build` with a rewriter placed before the return of a function `@f(%a: i32, %b: i64)`, given
/// the rewriter and the two parameters; returns what it gave and the operations it created, as
/// text without value numbers. The function must still verify.
fn built<R>(
    build: impl FnOnce(&mut ConversionPatternRewriter, ValueRef, ValueRef) -> Result<R, Report>,
) -> (R, Vec<String>) {
    let (context, module) = parse_module(
        "builtin.module public @m {\n    builtin.function public extern(\"wasm\") @f(%a: i32, %b: \
         i64) {\n        builtin.ret;\n    };\n};",
    );
    let function: OperationRef = runic_ir::SymbolTable::symbol_manager(&*module.borrow())
        .lookup_op("f")
        .expect("the function");
    let (entry, ret) = {
        let function = function.borrow();
        let function = function.downcast_ref::<Function>().expect("a function");
        let entry = function.body().entry_block_ref().expect("a body");
        let ret = entry.borrow().front().expect("the return");
        (entry, ret)
    };
    let (a, b) = {
        let entry = entry.borrow();
        let mut arguments = entry.argument_values();
        (arguments.next().unwrap(), arguments.next().unwrap())
    };
    let mut rewriter = ConversionPatternRewriter::new(context.clone());
    rewriter.set_insertion_point_before(ret);
    let value = build(&mut rewriter, a, b).unwrap_or_else(|report| panic!("{}", describe(&report)));
    module
        .borrow()
        .as_operation()
        .recursively_verify()
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let created = entry
        .borrow()
        .body()
        .iter()
        .filter(|op| op.as_operation_ref() != ret)
        .map(|op| without_numbers(&op.to_string()))
        .collect();
    (value, created)
}

/// The operations [mask] creates for the low `n` bits of the parameter of type `carrier`, and
/// whether it gave the parameter itself.
fn masked(carrier: &str, n: u32) -> (bool, Vec<String>) {
    let ((given, value), created) = built(|rw, a, b| {
        let value = if carrier == "i32" { a } else { b };
        mask(rw, value, n, Default::default()).map(|masked| (masked, value))
    });
    (given == value, created)
}

/// As [masked], for [sign_extend].
fn extended(carrier: &str, n: u32) -> (bool, Vec<String>) {
    let ((given, value), created) = built(|rw, a, b| {
        let value = if carrier == "i32" { a } else { b };
        sign_extend(rw, value, n, Default::default()).map(|extended| (extended, value))
    });
    (given == value, created)
}

/// Owned strings of `lines`.
fn owned(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|line| String::from(*line)).collect()
}

#[test]
fn a_mask_is_an_and_with_the_low_bits_set() {
    for (carrier, n, ones) in [
        ("i32", 1, "1"),
        ("i32", 7, "127"),
        ("i32", 8, "255"),
        ("i32", 16, "65535"),
        ("i32", 31, "2147483647"),
        ("i64", 33, "8589934591"),
        ("i64", 63, "9223372036854775807"),
    ] {
        let (itself, created) = masked(carrier, n);
        assert!(!itself, "{carrier} {n}");
        assert_eq!(
            created,
            owned(&[&format!("% = wasm.const {ones} : {carrier};"), "% = wasm.and %, %;"]),
            "{carrier} {n}"
        );
    }
}

#[test]
fn a_mask_of_the_whole_carrier_creates_nothing() {
    for (carrier, n) in [("i32", 32), ("i64", 64)] {
        let (itself, created) = masked(carrier, n);
        assert!(itself, "{carrier} {n}");
        assert!(created.is_empty(), "{carrier} {n}: {created:?}");
    }
}

#[test]
fn a_sign_extension_of_8_16_or_32_bits_is_one_instruction() {
    for (carrier, n, op) in [
        ("i32", 8, "extend8_s"),
        ("i32", 16, "extend16_s"),
        ("i64", 8, "extend8_s"),
        ("i64", 16, "extend16_s"),
        ("i64", 32, "extend32_s"),
    ] {
        let (itself, created) = extended(carrier, n);
        assert!(!itself, "{carrier} {n}");
        assert_eq!(created, owned(&[&format!("% = wasm.{op} %;")]), "{carrier} {n}");
    }
}

#[test]
fn a_sign_extension_of_another_width_is_a_shift_pair() {
    for (carrier, n, shift) in
        [("i32", 1, 31), ("i32", 7, 25), ("i32", 31, 1), ("i64", 33, 31), ("i64", 63, 1)]
    {
        let (itself, created) = extended(carrier, n);
        assert!(!itself, "{carrier} {n}");
        assert_eq!(
            created,
            owned(&[
                &format!("% = wasm.const {shift} : {carrier};"),
                "% = wasm.shl %, %;",
                "% = wasm.shr_s %, %;",
            ]),
            "{carrier} {n}"
        );
    }
}

#[test]
fn a_sign_extension_of_the_whole_carrier_creates_nothing() {
    for (carrier, n) in [("i32", 32), ("i64", 64)] {
        let (itself, created) = extended(carrier, n);
        assert!(itself, "{carrier} {n}");
        assert!(created.is_empty(), "{carrier} {n}: {created:?}");
    }
}

#[test]
fn an_integer_constant_keeps_the_low_bits_of_its_value() {
    let (_, created) = built(|rw, _, _| {
        let i32 = rw.context().i32();
        let i64 = rw.context().i64();
        const_int(rw, u64::MAX, i32, Default::default())?;
        const_int(rw, 255, i32, Default::default())?;
        const_int(rw, u64::MAX, i64, Default::default())
    });
    assert_eq!(
        created,
        owned(&[
            "% = wasm.const -1 : i32;",
            "% = wasm.const 255 : i32;",
            "% = wasm.const -1 : i64;"
        ])
    );
}

#[test]
fn an_integer_constant_of_a_type_that_is_not_a_wasm_integer_is_an_error() {
    let context = context();
    let mut rewriter = ConversionPatternRewriter::new(context.clone());
    let error = error_of(const_int(&mut rewriter, 1, context.i8(), Default::default()));
    assert!(mentions(&error, "`i8` is not a Wasm integer type"), "{error}");
}

#[test]
fn zero_is_a_constant_of_each_number_type_and_a_null_reference() {
    let (types, created) = built(|rw, _, _| {
        let context = rw.context();
        let types: [Type; 5] = [
            context.i32(),
            context.i64(),
            context.f32(),
            context.f64(),
            FuncRefType::get(context),
        ];
        for ty in types {
            let value = zero(rw, ty, Default::default())?;
            assert_eq!(value.borrow().ty(), ty);
        }
        Ok(types)
    });
    assert_eq!(types.len(), 5);
    assert_eq!(
        created,
        owned(&[
            "% = wasm.const 0 : i32;",
            "% = wasm.const 0 : i64;",
            "% = wasm.constf 0.0 : f32;",
            "% = wasm.constf 0.0 : f64;",
            "% = wasm.ref_null : !wasm.funcref;",
        ])
    );
}
