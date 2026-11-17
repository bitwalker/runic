//! What the body encoder cannot emit is a diagnostic that names the operation; and what it emits
//! records the features it uses.

use alloc::{format, string::String, vec::Vec};

use crate::{
    emit::{
        body::{BodyConfig, encode_function},
        desc::{Feature, ModuleDesc},
    },
    tests::{
        error_of,
        exec::{Setup, assemble},
        function_named, module_of, parse_module,
    },
};

/// The error of emitting the function `@f` of `functions` (with the assembly of the tests).
#[track_caller]
fn emit_error(functions: &str) -> String {
    let (_context, module) = parse_module(&module_of(functions));
    let setup = Setup::default().data("d", 0);
    error_of(assemble(module.as_operation_ref(), &setup, &[], &BodyConfig::default()).map(|_| ()))
}

#[test]
fn an_operation_that_is_not_of_wasm_form_is_an_error_naming_it() {
    let error = emit_error(
        "builtin.function public extern(\"wasm\") @f(%a: i32) -> i32 {\n%r = arith.add %a, \
         %a;\nbuiltin.ret %r : (i32);\n};",
    );
    assert!(
        error.contains("the operation `arith.add` cannot be emitted as WebAssembly")
            && error.contains("`arith.add` is not an operation of Wasm form"),
        "{error}"
    );
}

#[test]
fn control_is_an_internal_error_that_it_is_not_implemented_yet() {
    let error = emit_error(
        "builtin.function public extern(\"wasm\") @f(%a: i32) -> i32 {\n%r = wasm.block %a \
         {\n^b(%v: i32):\nwasm.end %v : (i32);\n} : (i32) -> (i32);\nbuiltin.ret %r : (i32);\n};",
    );
    assert!(
        error.contains(
            "the emission of `wasm.block` is not implemented yet, which is a defect of the backend"
        ) && error.contains("`wasm.block` is not implemented yet"),
        "{error}"
    );
}

#[test]
fn a_value_that_is_not_of_a_wasm_value_type_is_an_error_naming_it() {
    let error = emit_error(
        "builtin.function public extern(\"wasm\") @f(%a: i8) -> i8 {\nbuiltin.ret %a : (i8);\n};",
    );
    assert!(
        error.contains("the operation `builtin.function` cannot be emitted as WebAssembly")
            && error.contains("parameter 0 has type `i8`, which is not a WebAssembly value type"),
        "{error}"
    );
}

#[test]
fn a_declaration_has_no_body_to_encode() {
    let (_context, module) =
        parse_module(&module_of("builtin.function public extern(\"wasm\") @f(i32) -> i32;"));
    let mut desc = ModuleDesc::new();
    struct None_;
    impl crate::emit::instr::Entities for None_ {
        fn function(
            &mut self,
            user: &runic_ir::Operation,
            _: &runic_ir::dialects::builtin::attributes::SymbolRefAttr,
        ) -> Result<crate::emit::desc::FuncId, runic_ir::Report> {
            Err(crate::emit::instr::unsupported(user, "no entities"))
        }

        fn global(
            &mut self,
            user: &runic_ir::Operation,
            _: &runic_ir::dialects::builtin::attributes::SymbolRefAttr,
        ) -> Result<crate::emit::desc::GlobalId, runic_ir::Report> {
            Err(crate::emit::instr::unsupported(user, "no entities"))
        }

        fn table(
            &mut self,
            user: &runic_ir::Operation,
            _: &runic_ir::dialects::builtin::attributes::SymbolRefAttr,
        ) -> Result<crate::emit::desc::TableId, runic_ir::Report> {
            Err(crate::emit::instr::unsupported(user, "no entities"))
        }

        fn data(
            &mut self,
            user: &runic_ir::Operation,
            _: &runic_ir::dialects::builtin::attributes::SymbolRefAttr,
        ) -> Result<crate::emit::desc::DataId, runic_ir::Report> {
            Err(crate::emit::instr::unsupported(user, "no entities"))
        }
    }
    let error = error_of(encode_function(
        function_named(module, "f"),
        &mut desc,
        &mut None_,
        &BodyConfig::default(),
    ));
    assert!(error.contains("`@f` is a declaration, which has no body"), "{error}");
}

/// The features recorded by emitting the function of `functions`.
#[track_caller]
fn features_of(functions: &str) -> (Vec<Feature>, bool) {
    let (_context, module) = parse_module(&module_of(functions));
    let setup = Setup::default().data("d", 0).passive("seg", &[1]);
    let assembled = assemble(module.as_operation_ref(), &setup, &[], &BodyConfig::default())
        .unwrap_or_else(|report| panic!("{}", runic_ir::testing::text::describe(&report)));
    (assembled.features.iter().collect(), assembled.uses_data_count)
}

#[test]
fn each_use_of_a_feature_is_recorded_and_nothing_else() {
    let f = |signature: &str, body: &str| {
        format!("builtin.function public extern(\"wasm\") @f{signature} {{\n{body}\n}};")
    };
    let plain = f("(%a: i32) -> i32", "%r = wasm.add %a, %a;\nbuiltin.ret %r : (i32);");
    assert_eq!(features_of(&plain), (Vec::new(), false));
    let sign = f("(%a: i64) -> i64", "%r = wasm.extend32_s %a;\nbuiltin.ret %r : (i64);");
    assert_eq!(features_of(&sign), (Vec::from([Feature::SignExt]), false));
    let saturating = f(
        "(%a: f32) -> i32",
        "%r = wasm.trunc_sat_u %a <{ ty = i32 }>;\nbuiltin.ret %r : (i32);",
    );
    assert_eq!(features_of(&saturating), (Vec::from([Feature::NontrappingFptoint]), false));
    let bulk = format!(
        "wasm.memory public @mem : 1;\nbuiltin.global_variable public @seg : i32;\n{}",
        f("(%a: i32)", "wasm.memory_init @mem, @seg, %a, %a, %a;\nbuiltin.ret;")
    );
    assert_eq!(features_of(&bulk), (Vec::from([Feature::BulkMemory]), true));
    let pair = f("(%a: i32) -> (i32, i32)", "builtin.ret %a, %a : (i32, i32);");
    assert_eq!(features_of(&pair), (Vec::from([Feature::Multivalue]), false));
    let reference = f(
        "(%a: !wasm.externref) -> i32",
        "%r = wasm.ref_is_null %a;\nbuiltin.ret %r : (i32);",
    );
    assert_eq!(features_of(&reference), (Vec::from([Feature::ReferenceTypes]), false));
    let tail = f("(%a: i32) -> i32", "builtin.tail_call @f(%a) : extern(\"wasm\") (i32) -> i32;");
    assert_eq!(features_of(&tail), (Vec::from([Feature::TailCall]), false));
}
