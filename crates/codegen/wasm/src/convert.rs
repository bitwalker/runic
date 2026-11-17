//! The pass `convert-to-wasm`: the dialect conversion of generic IR to Wasm form.

pub mod calls;
pub mod composed;
pub mod float;
pub mod int;
pub mod memory;
pub mod symbols;

use alloc::{format, string::String, vec::Vec};
use core::fmt;

use runic_dialect_arith::ArithDialect;
use runic_dialect_cf::ControlFlowDialect;
use runic_dialect_mem::{self as mem, MemDialect};
use runic_dialect_scf::ScfDialect;
use runic_dialect_ub::UndefinedBehaviorDialect;
use runic_dialect_wasm::{self as wasm, MemoryRef, TableRef, WasmDialect};
use runic_ir::{
    Context, EntityMut, Forward, Operation, OperationName, OperationRef, RawWalk, Report, Spanned,
    Symbol, Type, Value, WalkResult,
    conversion::{
        ConversionConfig, ConversionPatternSet, ConversionTarget, TypeConverter,
        add_builtin_legality_by_type, apply_full_conversion,
    },
    diagnostics::report,
    dialects::builtin::{
        Call, CallIndirect, Component, Function, FunctionAlias, FunctionReference, GlobalVariable,
        Module, ModuleRef, RetImm, Segment, TailCall, TailCallIndirect, World,
        attributes::Signature,
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
///    aggregate, a pointer in an address space other than 0. A `mem.alloca` is an error that says
///    `wasm-lower-frames` must run first, and a `mem.load` or `mem.store` of a value that one
///    Wasm access cannot read or write is an error naming the operation and the type.
/// 3. The well-known symbols the module's operations name are found or declared
///    ([crate::symbols]), before the conversion starts, so that no pattern creates an operation
///    at the level of the module: the linear memory when an operation accesses memory (a
///    `mem.load`, `mem.store`, `mem.memcpy`, `mem.memmove` or `mem.memset`), the
///    indirect-function table when it makes an indirect call.
/// 4. The module is converted by a full conversion ([apply_full_conversion]), verified after,
///    under the target that [populate_patterns] is written for: the `wasm` dialect is legal; the
///    `arith`, `cf`, `mem`, `scf` and `ub` dialects are illegal; the builtin operations the
///    builtin conversion patterns cover are legal exactly when their types are
///    ([add_builtin_legality_by_type]), except `builtin.ret_imm`, which is illegal (Wasm form
///    returns with `builtin.ret`), and `builtin.call_indirect`, `builtin.tail_call_indirect` and
///    `builtin.function_ref`, which are illegal (Wasm form calls through a table); `builtin.module`, `builtin.function_alias`, `builtin.segment`
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
    check_module(module_op, &target)?;
    let symbols = WellKnownSymbols::of(module)?;
    let converter = crate::types::type_converter();
    let conversion_target = conversion_target(&context, &converter);
    let mut patterns = ConversionPatternSet::new(context.clone());
    populate_patterns(&mut patterns, &converter, &target, &symbols)?;
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
        .add_illegal_op::<CallIndirect>()
        .add_illegal_op::<TailCallIndirect>()
        .add_illegal_op::<FunctionReference>()
        .add_legal_op::<Module>()
        .add_legal_op::<FunctionAlias>()
        .add_legal_op::<Segment>()
        .add_legal_op::<GlobalVariable>()
        .mark_op_recursively_legal::<GlobalVariable>()
        .mark_op_recursively_legal::<wasm::Global>();
    target
}

/// Adds the patterns of `convert-to-wasm`, all driven by `converter`, for the target `target`,
/// naming the well-known symbols `symbols` of the module they convert.
///
/// Each area of the lowering adds its own from a module of its own under `convert/`.
pub fn populate_patterns(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
    target: &WasmTarget,
    symbols: &WellKnownSymbols,
) -> Result<(), Report> {
    calls::populate(patterns, converter)?;
    int::populate(patterns, converter, target)?;
    float::populate(patterns, converter)?;
    composed::populate(patterns, converter)?;
    memory::populate(patterns, converter, target, symbols.memory)?;
    symbols::populate(patterns, converter, symbols.table)?;
    Ok(())
}

/// The well-known symbols ([crate::symbols]) of one module that the patterns of `convert-to-wasm`
/// name, each found or declared before the conversion of the module starts.
#[derive(Clone, Default)]
pub struct WellKnownSymbols {
    /// The linear memory, `@__linear_memory`; `None` when no operation of the module accesses
    /// memory.
    pub memory: Option<MemoryRef>,
    /// The indirect-function table, `@__indirect_function_table`; `None` when no operation of the
    /// module makes an indirect call.
    pub table: Option<TableRef>,
}

impl WellKnownSymbols {
    /// The symbols the operations of `module` need: each found in the module, or declared there
    /// when it is needed and missing. An error when the module holds another symbol of a
    /// well-known name ([crate::symbols]).
    pub fn of(module: ModuleRef) -> Result<Self, Report> {
        let (mut memory, mut table) = (false, false);
        let _ = module.as_operation_ref().raw_prewalk::<Forward, _, _>(|op: OperationRef| {
            let op = op.borrow();
            memory |= op.is::<mem::Load>()
                || op.is::<mem::Store>()
                || op.is::<mem::MemCpy>()
                || op.is::<mem::MemMove>()
                || op.is::<mem::MemSet>();
            table |= op.is::<CallIndirect>() || op.is::<TailCallIndirect>();
            WalkResult::<()>::Continue(())
        });
        Ok(Self {
            memory: memory.then(|| crate::symbols::linear_memory(module)).transpose()?,
            table: table.then(|| crate::symbols::indirect_function_table(module)).transpose()?,
        })
    }
}

/// Checks, before anything is converted, the functions of `module`: their calling conventions
/// and those of their calls, the memory accesses, and that every value has a carrier (see
/// [ConvertToWasm]).
fn check_module(module: OperationRef, target: &WasmTarget) -> Result<(), Report> {
    let result = module.raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult {
        let is_function = op.borrow().is::<Function>();
        if is_function {
            return match check_function(op, target) {
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
fn check_function(function: OperationRef, target: &WasmTarget) -> Result<(), Report> {
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
        match check_operation(inner, target) {
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

/// Checks one operation of a function body: that it is not a `mem.alloca`, what a memory access
/// reads or writes, the convention of a call, and the carriers of its results and of the
/// arguments of the blocks of its regions.
fn check_operation(op: OperationRef, target: &WasmTarget) -> Result<(), Report> {
    let context = context_of(op);
    {
        let op_ref = op.borrow();
        if op_ref.is::<mem::Alloca>() {
            return Err(report!(
                message: (format!(
                    "`{NAME}` does not lower `mem.alloca`: run `{}` first, which gives each one a \
                     slot in its function's frame",
                    crate::frames::NAME
                )),
                labels: [primary(op_ref.span(), "this `mem.alloca` has no slot in a frame")],
            ));
        }
        let accessed = if let Some(load) = op_ref.downcast_ref::<mem::Load>() {
            Some(load.result().ty())
        } else {
            op_ref.downcast_ref::<mem::Store>().map(|store| store.value().ty())
        };
        if let Some(ty) = accessed {
            memory::access(&context, target.layout(), ty, &op_ref)?;
        }
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
/// carrier.
fn check_carried(
    context: &Context,
    ty: Type,
    span: runic_ir::SourceSpan,
    what: impl FnOnce() -> String,
) -> Result<(), Report> {
    let problem = if carrier(context, ty).is_some() {
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
