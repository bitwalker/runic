//! The lowering pipeline and the passes of the crate, run through a pass manager.

use alloc::boxed::Box;
use core::fmt;

use runic_ir::pass::{Nesting, PassManager};

use super::*;
use crate::{ConvertToWasm, VerifyForm, add_lowering_passes};

/// The text of the pipeline of `passes`.
struct Pipeline<'a>(&'a PassManager);

impl fmt::Display for Pipeline<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.print_as_textual_pipeline(f)
    }
}

/// A pass manager on `builtin.module` holding the lowering passes.
fn lowering(context: &Rc<Context>) -> PassManager {
    let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
    add_lowering_passes(passes.op_pass_manager_mut())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    passes.enable_verifier(true);
    passes
}

#[test]
fn the_lowering_passes_are_added_in_order_with_their_options() {
    let context = context();
    let passes = lowering(&context);
    let text = Pipeline(&passes).to_string();
    assert!(
        text.contains(
            "builtin.module(cfg-to-scf{exits=\"in-place\"},wasm-lower-frames,convert-to-wasm,\
             wasm-verify-form)"
        ),
        "{text}"
    );
}

#[test]
fn the_lowering_pipeline_brings_a_module_to_wasm_form() {
    let (context, module) = parse_module(
        "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, ptr = \
         32> } {\n    builtin.function public extern(\"C\") @f(%x: i8) -> (i8 { extend = \
         #builtin.sext }) {\n        cf.br ^next(%x : i8);\n    ^next(%y: i8):\n        \
         builtin.ret %y : (i8);\n    };\n};",
    );
    let mut passes = lowering(&context);
    passes
        .run(module.as_operation_ref())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let text = text_of(module);
    assert_eq!(
        lines(&text)[1..6],
        [
            "builtin.function public extern(\"C\") @f(%: i32) -> (i32 { extend = #builtin.sext }) \
             {",
            "% = wasm.const 255 : i32;",
            "% = wasm.and %, %;",
            "% = wasm.extend8_s %;",
            "builtin.ret % : (i32);",
        ],
        "{text}"
    );
}

/// The passes run on a module directly, without a pipeline.
fn run_pass(pass: Box<dyn runic_ir::pass::OperationPass>, text: &str) -> Result<String, Report> {
    let context = context();
    let module = parse_module_in(&context, text)?;
    let mut passes = PassManager::on::<Module>(context, Nesting::Implicit);
    passes.add_pass(pass)?;
    passes.enable_verifier(true);
    passes.run(module.as_operation_ref())?;
    Ok(text_of(module))
}

#[test]
fn a_module_in_wasm_form_is_left_unchanged_by_both_passes() {
    let text = "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, \
                ptr = 32> } {\n    builtin.function public extern(\"wasm\") @f(%x: i32) -> i32 \
                {\n        %y = wasm.add %x, %x;\n        builtin.ret %y : (i32);\n    };\n};";
    let converted = run_pass(Box::<ConvertToWasm>::default(), text)
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let verified = run_pass(Box::<VerifyForm>::default(), text)
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    assert_eq!(converted, verified);
    assert!(converted.contains("wasm.add"), "{converted}");
}

#[test]
fn a_function_used_outside_its_module_whose_signature_changes_is_refused() {
    // The function `@f` of `@a` is called from `@b`: converting `@a` alone would leave the call
    // in `@b` with the old signature.
    let text = "builtin.world {\n    builtin.module public @a {\n        builtin.function public \
                extern(\"C\") @f(%x: i8) {\n            builtin.ret;\n        };\n    };\n    \
                builtin.module public @b {\n        builtin.function public extern(\"C\") @g(%x: \
                i8) {\n            builtin.call ::@a::@f(%x) : extern(\"C\") (i8) -> ();\n            \
                builtin.ret;\n        };\n    };\n} attributes { data_layout = \
                #builtin.data_layout<little, ptr = 32> };";
    let context = context();
    let world = parse::parse::<runic_ir::dialects::builtin::World>(
        ParserConfig::new(context.clone()),
        "w",
        text,
    )
    .into_result()
    .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let mut passes =
        PassManager::on::<runic_ir::dialects::builtin::World>(context, Nesting::Implicit);
    passes.add_pass(Box::<ConvertToWasm>::default()).unwrap();
    let error = error_of(passes.run(world.as_operation_ref()));
    assert!(
        error.contains(
            "the conversion changes the signature of function 'builtin.function' ::a::f, which is \
             named by 'builtin.call' in ::b::g outside the root of the conversion"
        ),
        "{error}"
    );
}
