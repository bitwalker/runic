//! The pass `convert-to-wasm`: the dialect conversion of generic IR to Wasm form.

pub mod calls;
pub mod int;

use alloc::{format, string::String, vec::Vec};
use core::fmt;

use runic_dialect_arith::ArithDialect;
use runic_dialect_cf::ControlFlowDialect;
use runic_dialect_mem::MemDialect;
use runic_dialect_scf::ScfDialect;
use runic_dialect_ub::UndefinedBehaviorDialect;
use runic_dialect_wasm::{self as wasm, WasmDialect};
use runic_ir::{
    Context, EntityMut, Forward, Operation, OperationName, OperationRef, RawWalk, Report, Spanned,
    Symbol, Type, Value, WalkResult,
    conversion::{
        ConversionConfig, ConversionPatternSet, ConversionTarget, TypeConverter,
        add_builtin_legality_by_type, apply_full_conversion,
    },
    diagnostics::report,
    dialects::builtin::{
        Call, CallIndirect, Component, Function, FunctionAlias, GlobalVariable, Module, ModuleRef,
        RetImm, Segment, TailCall, TailCallIndirect, World, attributes::Signature,
    },
    pass::{Pass, PassExecutionState, PostPassStatus},
};

use crate::{WasmTarget, types::carrier};

/// The name `convert-to-wasm` is registered under.
pub const NAME: &str = "convert-to-wasm";

const DESCRIPTION: &str = "Converts generic operations and types to Wasm form: wasm operations \
                           and Wasm value types (each builtin.module, under a 32-bit \
                           little-endian data layout)";

/// Converts each `builtin.module` under the operation it runs on to Wasm form.
///
/// It runs on a world, a component or a module, or on anything that holds them (it is scheduled
/// as `cfg-to-scf` is); a module nested in a module is converted with the module that holds it.
/// For each module, in order:
///
/// 1. The target ([WasmTarget]) is read from the data layout in scope; a module without one, or
///    with one that is not little-endian with 32-bit pointers and `isize`, is an error.
/// 2. Every function is checked before anything is converted: a calling convention other than
///    `"wasm"` and `"C"` (on a function or on a call) is an error naming it, and so is each value
///    (a parameter, a result, a local, the result of an operation, the argument of a block) whose
///    type has no carrier ([carrier]): an integer wider than 64 bits (not lowered yet), an
///    aggregate, a pointer in an address space other than 0. A value of a function type is refused
///    too: the type converter cannot give it its carrier (see [crate::types::type_converter]).
/// 3. The module is converted by a full conversion ([apply_full_conversion]), verified after,
///    under the target that [populate_patterns] is written for: the `wasm` dialect is legal; the
///    `arith`, `cf`, `mem`, `scf` and `ub` dialects are illegal; the builtin operations the
///    builtin conversion patterns cover are legal exactly when their types are
///    ([add_builtin_legality_by_type]), except `builtin.ret_imm`, which is illegal (Wasm form
///    returns with `builtin.ret`); `builtin.module`, `builtin.function_alias`, `builtin.segment`
///    and `builtin.global_variable` are legal, and a global variable and a `wasm.global` are
///    recursively legal: an initializer is data or a constant expression, not code. Every other
///    operation is illegal, so an operation that no pattern converts is an error that names it.
///
/// A function with a user outside its module whose signature changes is refused by the builtin
/// function pattern, and the error is passed on. Nothing is rolled back after an error.
#[derive(Default)]
pub struct ConvertToWasm;

runic_ir::inventory::submit!(runic_ir::pass::PassInfo::new::<ConvertToWasm>(NAME, DESCRIPTION));

impl Pass for ConvertToWasm {
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

    fn initialize(&mut self, context: alloc::rc::Rc<Context>) -> Result<(), Report> {
        context.get_or_register_dialect::<WasmDialect>();
        Ok(())
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let root = op.as_operation_ref();
        // The conversion borrows the root itself.
        drop(op);
        let mut changed = false;
        for module in modules_under(root) {
            changed |= convert_module(module)?;
        }
        state.set_post_pass_status(PostPassStatus::from(changed));
        if !changed {
            state.preserved_analyses_mut().preserve_all();
        }
        Ok(())
    }
}

/// The outermost `builtin.module`s under `root`, `root` included, found through worlds and
/// components.
fn modules_under(root: OperationRef) -> Vec<ModuleRef> {
    let mut modules = Vec::new();
    let _ = root.raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult {
        if let Ok(module) = op.try_downcast_op::<Module>() {
            modules.push(module);
            return WalkResult::Skip;
        }
        let op = op.borrow();
        if op.is::<World>() || op.is::<Component>() {
            WalkResult::Continue(())
        } else {
            WalkResult::Skip
        }
    });
    modules
}

/// Converts `module` (see [ConvertToWasm]); whether it changed.
pub fn convert_module(module: ModuleRef) -> Result<bool, Report> {
    let module_op = module.as_operation_ref();
    let context = module_op.borrow().context_rc();
    let target = WasmTarget::of(&module_op.borrow())?;
    check_module(module_op)?;
    let converter = crate::types::type_converter();
    let conversion_target = conversion_target(&context, &converter);
    let mut patterns = ConversionPatternSet::new(context.clone());
    populate_patterns(&mut patterns, &converter, &target)?;
    let result =
        apply_full_conversion(module_op, conversion_target, patterns, ConversionConfig::default())?;
    Ok(result.changed())
}

/// The target of the conversion (see [ConvertToWasm]).
pub fn conversion_target(
    context: &alloc::rc::Rc<Context>,
    converter: &TypeConverter,
) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target
        .add_legal_dialect::<WasmDialect>()
        .add_illegal_dialect::<ArithDialect>()
        .add_illegal_dialect::<ControlFlowDialect>()
        .add_illegal_dialect::<MemDialect>()
        .add_illegal_dialect::<ScfDialect>()
        .add_illegal_dialect::<UndefinedBehaviorDialect>();
    add_builtin_legality_by_type(&mut target, converter);
    target
        .add_illegal_op::<RetImm>()
        .add_legal_op::<Module>()
        .add_legal_op::<FunctionAlias>()
        .add_legal_op::<Segment>()
        .add_legal_op::<GlobalVariable>()
        .mark_op_recursively_legal::<GlobalVariable>()
        .mark_op_recursively_legal::<wasm::Global>();
    target
}

/// Adds the patterns of `convert-to-wasm`, all driven by `converter`, for the target `target`.
///
/// Each area of the lowering adds its own from a module of its own under `convert/`.
pub fn populate_patterns(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
    target: &WasmTarget,
) -> Result<(), Report> {
    calls::populate(patterns, converter)?;
    int::populate(patterns, converter, target)?;
    Ok(())
}

/// Checks, before anything is converted, the functions of `module`: their calling conventions
/// and those of their calls, and that every value has a carrier (see [ConvertToWasm]).
fn check_module(module: OperationRef) -> Result<(), Report> {
    let result = module.raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult {
        let is_function = op.borrow().is::<Function>();
        if is_function {
            return match check_function(op) {
                Ok(()) => WalkResult::Skip,
                Err(error) => WalkResult::Break(error),
            };
        }
        let op = op.borrow();
        if op.is::<GlobalVariable>() || op.is::<wasm::Global>() {
            WalkResult::Skip
        } else {
            WalkResult::Continue(())
        }
    });
    result.into_result()
}

/// Checks the function `function` (see [check_module]).
fn check_function(function: OperationRef) -> Result<(), Report> {
    let op = function.borrow();
    let span = op.span();
    let func = op.downcast_ref::<Function>().expect("a builtin.function");
    let name = format!("`@{}`", func.get_name().as_str());
    let signature = func.get_signature().clone();
    calls::Convention::of(&signature, &op)?;
    let context = op.context();
    for (index, param) in signature.params().iter().enumerate() {
        check_carried(context, param.ty, span, || format!("parameter {index} of {name}"))?;
    }
    for (index, result) in signature.results().iter().enumerate() {
        check_carried(context, result.ty, span, || format!("result {index} of {name}"))?;
    }
    for (index, ty) in func.get_locals().iter().enumerate() {
        check_carried(context, *ty, span, || format!("local {index} of {name}"))?;
    }
    if func.is_declaration() {
        return Ok(());
    }
    let body = func.body().as_region_ref();
    drop(op);
    check_region_arguments(context_of(function), function, body, &name)?;
    let result = function.raw_prewalk::<Forward, _, _>(|inner: OperationRef| -> WalkResult {
        if inner == function {
            return WalkResult::Continue(());
        }
        match check_operation(inner) {
            Ok(()) => WalkResult::Continue(()),
            Err(error) => WalkResult::Break(error),
        }
    });
    result.into_result()
}

/// The context of `op`.
fn context_of(op: OperationRef) -> alloc::rc::Rc<Context> {
    op.borrow().context_rc()
}

/// Checks one operation of a function body: the convention of a call, and the carriers of its
/// results and of the arguments of the blocks of its regions.
fn check_operation(op: OperationRef) -> Result<(), Report> {
    let context = context_of(op);
    {
        let op_ref = op.borrow();
        if let Some(signature) = call_signature(&op_ref) {
            calls::Convention::of(&signature, &op_ref)?;
        }
        let span = op_ref.span();
        let name = op_ref.name();
        for (index, result) in op_ref.results().iter().enumerate() {
            let ty = result.borrow().ty();
            check_carried(&context, ty, span, || format!("result {index} of `{name}`"))?;
        }
    }
    let regions = op
        .borrow()
        .regions()
        .iter()
        .map(|region| region.as_region_ref())
        .collect::<Vec<_>>();
    let name = format!("`{}`", op.borrow().name());
    for region in regions {
        check_region_arguments(context.clone(), op, region, &name)?;
    }
    Ok(())
}

/// Checks the carriers of the arguments of the blocks of `region`, a region of `owner`, which the
/// messages call `name`.
fn check_region_arguments(
    context: alloc::rc::Rc<Context>,
    owner: OperationRef,
    region: runic_ir::RegionRef,
    name: &str,
) -> Result<(), Report> {
    let span = owner.borrow().span();
    for (block_index, block) in region.borrow().body().iter().enumerate() {
        for (index, argument) in block.argument_values().enumerate() {
            let ty = argument.borrow().ty();
            check_carried(&context, ty, span, || {
                format!("argument {index} of block {block_index} of {name}")
            })?;
        }
    }
    Ok(())
}

/// The signature of a call operation, if `op` is one.
fn call_signature(op: &Operation) -> Option<Signature> {
    if let Some(call) = op.downcast_ref::<Call>() {
        return Some(call.get_signature().clone());
    }
    if let Some(call) = op.downcast_ref::<TailCall>() {
        return Some(call.get_signature().clone());
    }
    if let Some(call) = op.downcast_ref::<CallIndirect>() {
        return Some(call.get_signature().clone());
    }
    if let Some(call) = op.downcast_ref::<TailCallIndirect>() {
        return Some(call.get_signature().clone());
    }
    None
}

/// An error, with a label on `span` that says `what` (a value) has type `ty`, when `ty` has no
/// carrier, or is a function type.
fn check_carried(
    context: &Context,
    ty: Type,
    span: runic_ir::SourceSpan,
    what: impl FnOnce() -> String,
) -> Result<(), Report> {
    let problem = if ty.is_function() {
        "values of function type are not lowered yet"
    } else if carrier(context, ty).is_some() {
        return Ok(());
    } else if ty.integer_width().is_some() {
        "integers wider than 64 bits are not lowered yet"
    } else if ty.as_pointer().is_some() {
        "only pointers in address space 0, into linear memory, have a Wasm carrier"
    } else {
        "the type has no Wasm value type to carry it"
    };
    Err(report!(
        message: (format!("`{NAME}` cannot lower a value of type `{ty}`: {problem}")),
        labels: [primary(span, format!("{} has type `{ty}`", what()))],
    ))
}
