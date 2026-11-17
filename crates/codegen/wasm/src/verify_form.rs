//! The pass `wasm-verify-form`, which checks that the functions under the operation it runs on
//! are in Wasm form.

use alloc::{format, string::String};
use core::fmt;

use runic_dialect_mem::Alloca;
use runic_dialect_wasm::{Global, is_val_type};
use runic_ir::{
    EntityMut, Forward, Operation, OperationName, OperationRef, RawWalk, RegionRef, Report,
    SourceSpan, Spanned, Symbol, Type, Value, WalkResult,
    diagnostics::report,
    dialects::builtin::{Call, Function, GlobalVariable, LoadLocal, Ret, StoreLocal, TailCall},
    pass::{Pass, PassExecutionState, PostPassStatus},
    traits::BranchOpInterface,
};

/// The name `wasm-verify-form` is registered under.
pub const NAME: &str = "wasm-verify-form";

const DESCRIPTION: &str =
    "Checks that every function is in Wasm form; changes nothing, and fails on the first violation";

/// Checks that every `builtin.function` under the operation it runs on is in Wasm form, and fails
/// on the first violation, naming the operation and the pass that should have removed it. It
/// changes nothing.
///
/// A function is in Wasm form when:
///
/// * its calling convention is `"wasm"` or `"C"`;
/// * every parameter and result of its signature (a declaration's included; a `never` result is
///   no value) and every local has a Wasm value type ([is_val_type]);
/// * every operation in its body, at any depth, is a `wasm` operation or one of `builtin.call`,
///   `builtin.tail_call`, `builtin.ret`, `builtin.load_local` and `builtin.store_local`;
/// * every operand and result of those operations and every argument of a block in its body has
///   a Wasm value type;
/// * every region in its body, the body included, has one block.
///
/// An operation of `cf` that branches names `cfg-to-scf`, a `mem.alloca` names
/// `wasm-lower-frames`, and every other operation that is not allowed names `convert-to-wasm`.
/// The functions are found through every operation except a `builtin.global_variable` and a
/// `wasm.global`, whose initializers are not code.
#[derive(Default)]
pub struct VerifyForm;

runic_ir::inventory::submit!(runic_ir::pass::PassInfo::new::<VerifyForm>(NAME, DESCRIPTION));

impl Pass for VerifyForm {
    type Target = Operation;

    fn name(&self) -> &'static str {
        NAME
    }

    fn argument(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        DESCRIPTION
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(NAME)
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let root = op.as_operation_ref();
        drop(op);
        state.set_post_pass_status(PostPassStatus::Unchanged);
        state.preserved_analyses_mut().preserve_all();
        verify_wasm_form(root)
    }
}

/// Checks that every function under `root`, `root` included, is in Wasm form (see
/// [VerifyForm]); the error of the first violation.
pub fn verify_wasm_form(root: OperationRef) -> Result<(), Report> {
    root.raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult {
        let (is_function, is_data) = {
            let op = op.borrow();
            (op.is::<Function>(), op.is::<GlobalVariable>() || op.is::<Global>())
        };
        if is_function {
            return match verify_function(op) {
                Ok(()) => WalkResult::Skip,
                Err(error) => WalkResult::Break(error),
            };
        }
        if is_data {
            WalkResult::Skip
        } else {
            WalkResult::Continue(())
        }
    })
    .into_result()
}

/// The error of a function that is not in Wasm form: `label` on `span`.
fn violation(function: &str, span: SourceSpan, label: String) -> Report {
    report!(
        message: (format!("`{NAME}`: the function `@{function}` is not in Wasm form")),
        labels: [primary(span, label)],
    )
}

/// The label of a value of type `ty` that is not a Wasm value type, `what` it is, and the pass
/// that converts it.
fn not_a_value_type(what: String, ty: Type) -> String {
    format!(
        "{what} has type `{ty}`, which is not a Wasm value type; `convert-to-wasm` should have \
         converted it"
    )
}

/// Checks the function `function` (see [VerifyForm]).
fn verify_function(function: OperationRef) -> Result<(), Report> {
    let (name, body) = {
        let op = function.borrow();
        let span = op.span();
        let func = op.downcast_ref::<Function>().expect("a builtin.function");
        let name = String::from(func.get_name().as_str());
        let signature = func.get_signature();
        let convention = signature.calling_convention();
        if !matches!(convention.as_str(), "wasm" | "C") {
            return Err(violation(
                &name,
                span,
                format!(
                    "`@{name}` uses the calling convention \"{}\": Wasm form has \"wasm\" and \
                     \"C\"",
                    convention.as_str()
                ),
            ));
        }
        for (index, param) in signature.params().iter().enumerate() {
            if !is_val_type(param.ty) {
                let what = format!("parameter {index} of `@{name}`");
                return Err(violation(&name, span, not_a_value_type(what, param.ty)));
            }
        }
        for (index, result) in signature.results().iter().enumerate() {
            if !is_val_type(result.ty) {
                let what = format!("result {index} of `@{name}`");
                return Err(violation(&name, span, not_a_value_type(what, result.ty)));
            }
        }
        for (index, ty) in func.get_locals().iter().enumerate() {
            if !is_val_type(*ty) {
                let what = format!("local {index} of `@{name}`");
                return Err(violation(&name, span, not_a_value_type(what, *ty)));
            }
        }
        if func.is_declaration() {
            return Ok(());
        }
        (name, func.body().as_region_ref())
    };
    // The operations first, so that a branch is reported as the branch it is rather than as the
    // blocks it joins; then the regions, the body's first.
    function
        .raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult {
            if op == function {
                return WalkResult::Continue(());
            }
            match verify_operation(&name, op) {
                Ok(()) => WalkResult::Continue(()),
                Err(error) => WalkResult::Break(error),
            }
        })
        .into_result()?;
    verify_region(&name, function, body)?;
    function
        .raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult {
            if op == function {
                return WalkResult::Continue(());
            }
            let regions = op
                .borrow()
                .regions()
                .iter()
                .map(|region| region.as_region_ref())
                .collect::<alloc::vec::Vec<_>>();
            for region in regions {
                if let Err(error) = verify_region(&name, op, region) {
                    return WalkResult::Break(error);
                }
            }
            WalkResult::Continue(())
        })
        .into_result()
}

/// The pass that should have removed `op`, if it is not allowed in a body in Wasm form.
fn remover(op: &Operation) -> Option<&'static str> {
    if op.name().dialect().as_str() == "wasm"
        || op.is::<Call>()
        || op.is::<TailCall>()
        || op.is::<Ret>()
        || op.is::<LoadLocal>()
        || op.is::<StoreLocal>()
    {
        return None;
    }
    if op.is::<Alloca>() {
        return Some(crate::frames::NAME);
    }
    if op.name().dialect().as_str() == "cf" && op.implements::<dyn BranchOpInterface>() {
        return Some("cfg-to-scf");
    }
    Some(crate::convert::NAME)
}

/// Checks one operation of the body of the function `function`: its kind and the types of its
/// operands and results (see [VerifyForm]).
fn verify_operation(function: &str, op: OperationRef) -> Result<(), Report> {
    {
        let op_ref = op.borrow();
        let span = op_ref.span();
        let name = op_ref.name();
        if let Some(pass) = remover(&op_ref) {
            return Err(violation(
                function,
                span,
                format!("`{name}` is not in Wasm form; `{pass}` should have removed it"),
            ));
        }
        for (index, operand) in op_ref.operands().iter().enumerate() {
            let ty = operand.borrow().ty();
            if !is_val_type(ty) {
                let what = format!("operand {index} of `{name}`");
                return Err(violation(function, span, not_a_value_type(what, ty)));
            }
        }
        for (index, result) in op_ref.results().iter().enumerate() {
            let ty = result.borrow().ty();
            if !is_val_type(ty) {
                let what = format!("result {index} of `{name}`");
                return Err(violation(function, span, not_a_value_type(what, ty)));
            }
        }
    }
    Ok(())
}

/// Checks that `region`, a region of `owner` in the function `function`, has one block, and that
/// its arguments have Wasm value types.
fn verify_region(function: &str, owner: OperationRef, region: RegionRef) -> Result<(), Report> {
    let owner = owner.borrow();
    let span = owner.span();
    let owner_name = match owner.downcast_ref::<Function>() {
        Some(_) => format!("`@{function}`"),
        None => format!("`{}`", owner.name()),
    };
    let index = owner
        .regions()
        .iter()
        .position(|candidate| candidate.as_region_ref() == region)
        .unwrap_or_default();
    let region = region.borrow();
    let blocks = region.body().iter().count();
    if blocks != 1 {
        let pass = if blocks > 1 {
            "cfg-to-scf"
        } else {
            crate::convert::NAME
        };
        return Err(violation(
            function,
            span,
            format!(
                "region {index} of {owner_name} has {blocks} blocks, but a region in Wasm form \
                 has one; `{pass}` should have structured it"
            ),
        ));
    }
    for block in region.body().iter() {
        for (argument_index, argument) in block.argument_values().enumerate() {
            let ty = argument.borrow().ty();
            if !is_val_type(ty) {
                let what = format!(
                    "argument {argument_index} of the block of region {index} of {owner_name}"
                );
                return Err(violation(function, span, not_a_value_type(what, ty)));
            }
        }
    }
    Ok(())
}
