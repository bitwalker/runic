//! `wasm-verify-form` checks of what text that verifies cannot hold, reached through IR changed by
//! the API.

use runic_ir::RawWalk;

use super::*;
use crate::verify_wasm_form;

#[test]
fn an_argument_of_a_block_of_a_construct_whose_type_is_not_a_wasm_value_type_is_reported() {
    // In Wasm form, then the unused argument of the loop's body is given the type `i8`: every
    // operation still is in Wasm form, and the block's argument is not.
    let (context, module) =
        parse_module(&module_of(
            "builtin.function public extern(\"wasm\") @f(%x: i32) -> i32 {\n    %r = wasm.loop %x \
             {\n    ^body(%i: i32):\n        wasm.end %x : (i32);\n    } : (i32) -> (i32);\n    \
             builtin.ret %r : (i32);\n};",
        ));
    verify_wasm_form(module.as_operation_ref())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    let mut argument = None;
    let _ = module.as_operation_ref().raw_prewalk::<runic_ir::Forward, _, _>(
        |op: runic_ir::OperationRef| {
            let op = op.borrow();
            if op.is::<runic_dialect_wasm::Loop>() {
                let entry = op.regions().front().get().and_then(|r| r.entry_block_ref());
                argument = entry.and_then(|block| block.borrow().argument_values().next());
            }
            runic_ir::WalkResult::<()>::Continue(())
        },
    );
    let mut argument = argument.expect("the argument of the loop's body");
    argument.borrow_mut().set_type(context.i8());
    let error = error_of(verify_wasm_form(module.as_operation_ref()));
    assert!(
        error.contains(
            "argument 0 of the block of region 0 of `wasm.loop` has type `i8`, which is not a \
             Wasm value type; `convert-to-wasm` should have converted it"
        ),
        "{error}"
    );
}
