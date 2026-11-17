//! The pass `convert-to-wasm`: the dialect conversion of generic IR to Wasm form.

pub mod calls;
pub mod composed;
pub mod control;
pub mod float;
pub mod int;
pub mod labels;
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
    Context, EntityMut, Forward, FxHashMap, FxHashSet, Operation, OperationName, OperationRef,
    RawWalk, RegionExitKind, Report, Spanned, Symbol, Type, Value, WalkResult,
    conversion::{
        ConversionConfig, ConversionPatternSet, ConversionTarget, DynamicLegalityResult,
        TypeConverter, add_builtin_legality_by_type, apply_full_conversion,
    },
    diagnostics::report,
    dialects::builtin::{
        Call, CallIndirect, Component, Function, FunctionAlias, FunctionReference, GlobalVariable,
        Module, ModuleRef, RetImm, Segment, TailCall, TailCallIndirect, UnrealizedConversionCast,
        World, attributes::Signature,
    },
    pass::{Pass, PassExecutionState, PostPassStatus},
};

use self::labels::Labels;
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
/// An operation that holds no module is an error that says the pass needs one.
///
/// **Either everything converts or nothing changes.** Every module under the operation is checked
/// before any of them is changed, and every refusal a pattern of the pass can make on IR that
/// verifies is made by this check, through the same function the pattern calls:
///
/// 1. The target ([WasmTarget]) is read from the data layout in scope; a module without one, or
///    with one that is not little-endian with 32-bit pointers and `isize`, is an error.
/// 2. Every function is checked: a calling convention other than `"wasm"` and `"C"` (on a
///    function, a declaration included, or on a call) is an error naming it, and so is each value
///    (a parameter, a result, a local, the result of an operation, the argument of a block) whose
///    type has no carrier ([carrier]): an integer wider than 64 bits (not lowered yet), an
///    aggregate, a pointer in an address space other than 0. A function whose signature changes
///    and that an operation outside the module names is an error naming both (the builtin
///    function pattern refuses it). In a body:
///    * a `mem.alloca` is an error that says `wasm-lower-frames` must run first; a `mem.load` or
///      `mem.store` of a value that one Wasm access cannot read or write, and a `mem.gep`,
///      `mem.sizeof`, `mem.alignof` or `mem.offsetof` whose size or offset the layout does not
///      know or does not fit in 32 bits, are errors naming the operation ([memory]);
///    * `cf.br`, `cf.cond_br` and `cf.switch` are errors that say `cfg-to-scf` must run first;
///    * an `scf.yield` or `scf.condition` that does not end a body of the `scf` construct that
///      lowers it, and an `scf` construct with a body that has no block, are errors
///      ([control]);
///    * an `scf.break` or `scf.continue` whose level does not translate (a break that names a
///      `wasm.loop` the frontend wrote) is an error ([labels::check_exit]);
///    * a `wasm.br` that does not resolve is an error, and the construct each one names is
///      recorded ([labels::record_branches]);
///    * an `isize` constant that the target's 32-bit `isize` does not hold is an error
///      (`calls::check_constant`);
///    * a `builtin.unrealized_conversion_cast` written before the conversion whose result is a
///      narrow integer (one its carrier is wider than) is an error: nothing says its high bits
///      are clear;
///    * an operation that the target does not accept and that no pattern of the pass is rooted
///      at is an error naming it.
///
///    The `after` body of an `scf.while` whose `before` body does not end in `scf.condition` is
///    never entered, and the lowering drops it: it is not checked.
/// 3. The well-known symbols the functions of each module name ([crate::symbols]): a module that
///    holds another symbol of a well-known name that its functions need is an error.
///
/// Then each module is converted:
///
/// 1. The well-known symbols are found or declared ([crate::symbols]) in the module that directly
///    holds the function whose code names them, before the conversion starts, so that no pattern
///    creates an operation at the level of a module: the linear memory when an operation accesses
///    memory (a `mem.load`, `mem.store`, `mem.memcpy`, `mem.memmove` or `mem.memset`), the
///    indirect-function table when it makes an indirect call.
/// 2. The module is converted by a full conversion ([apply_full_conversion]), verified after,
///    under the target that [populate_patterns] is written for: the `wasm` dialect is legal
///    except a `wasm.br` of the input whose label no longer names the construct it named; the
///    `arith`, `cf`, `mem`, `scf` and `ub` dialects are illegal; the builtin operations the
///    builtin conversion patterns cover are legal exactly when their types are
///    ([add_builtin_legality_by_type]), except `builtin.ret_imm`, which is illegal (Wasm form
///    returns with `builtin.ret`), and `builtin.call_indirect`, `builtin.tail_call_indirect` and
///    `builtin.function_ref`, which are illegal (Wasm form calls through a table); a
///    `builtin.unrealized_conversion_cast` is legal exactly when its types are (the patterns of
///    control flow read a value through one, which folds away; see [control]);
///    `builtin.module`, `builtin.function_alias`, `builtin.segment`
///    and `builtin.global_variable` are legal, and a global variable and a `wasm.global` are
///    recursively legal: an initializer is data or a constant expression, not code, and is not
///    converted. Every other operation is illegal.
///
/// An error the conversion itself returns after the check (none is known on IR that verifies) is
/// passed on, and nothing is rolled back after it.
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
        let modules = modules_under(root);
        if modules.is_empty() {
            return Err(crate::no_module_error(NAME, root));
        }
        let changed = convert_modules(&modules)?;
        state.set_post_pass_status(PostPassStatus::from(changed));
        if !changed {
            state.preserved_analyses_mut().preserve_all();
        }
        Ok(())
    }
}

/// The outermost `builtin.module`s under `root`, `root` included, found through worlds and
/// components.
pub(crate) fn modules_under(root: OperationRef) -> Vec<ModuleRef> {
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

/// Converts `module` (see [ConvertToWasm]); whether it changed. When it returns an error from
/// the check, nothing has changed.
pub fn convert_module(module: ModuleRef) -> Result<bool, Report> {
    convert_modules(&[module])
}

/// Checks every one of `modules`, then converts each (see [ConvertToWasm]); whether one changed.
fn convert_modules(modules: &[ModuleRef]) -> Result<bool, Report> {
    let checked = modules
        .iter()
        .map(|module| Checked::of(*module))
        .collect::<Result<Vec<_>, _>>()?;
    let mut changed = false;
    for module in checked {
        changed |= module.convert()?;
    }
    Ok(changed)
}

/// A module that passed the check of [ConvertToWasm], with what the check found.
struct Checked {
    module: ModuleRef,
    target: WasmTarget,
    labels: Labels,
}

impl Checked {
    /// Checks `module` (see [ConvertToWasm]); changes nothing.
    fn of(module: ModuleRef) -> Result<Self, Report> {
        let module_op = module.as_operation_ref();
        let context = module_op.borrow().context_rc();
        let target = WasmTarget::of(&module_op.borrow())?;
        let converter = crate::types::type_converter();
        let labels = labels::new_labels();
        let conversion_target = conversion_target(&context, &converter, &labels);
        let roots = pattern_roots(&context, &converter, &target)?;
        let check = Check {
            root: module_op,
            target: &target,
            converter: &converter,
            conversion_target: &conversion_target,
            roots: &roots,
        };
        check.module()?;
        labels::record_branches(module_op, &labels)?;
        WellKnownSymbols::check(module)?;
        Ok(Self {
            module,
            target,
            labels,
        })
    }

    /// Converts the module (see [ConvertToWasm]); whether it changed.
    fn convert(self) -> Result<bool, Report> {
        let module_op = self.module.as_operation_ref();
        let context = module_op.borrow().context_rc();
        let symbols = alloc::rc::Rc::new(WellKnownSymbols::of(self.module)?);
        let converter = crate::types::type_converter();
        let conversion_target = conversion_target(&context, &converter, &self.labels);
        let mut patterns = ConversionPatternSet::new(context.clone());
        populate_patterns(&mut patterns, &converter, &self.target, &symbols, &self.labels)?;
        let result = apply_full_conversion(
            module_op,
            conversion_target,
            patterns,
            ConversionConfig::default(),
        )?;
        Ok(result.changed() || symbols.declared())
    }
}

/// The names of the operations some pattern of `convert-to-wasm` is rooted at.
fn pattern_roots(
    context: &alloc::rc::Rc<Context>,
    converter: &TypeConverter,
    target: &WasmTarget,
) -> Result<FxHashSet<OperationName>, Report> {
    let mut patterns = ConversionPatternSet::new(context.clone());
    let symbols = alloc::rc::Rc::new(WellKnownSymbols::default());
    populate_patterns(&mut patterns, converter, target, &symbols, &labels::new_labels())?;
    Ok(patterns
        .patterns()
        .iter()
        .filter_map(|pattern| pattern.info().root_operation())
        .collect())
}

/// The target of the conversion (see [ConvertToWasm]), whose legality of a `wasm.br` reads the
/// branches recorded in `labels`.
pub fn conversion_target(
    context: &alloc::rc::Rc<Context>,
    converter: &TypeConverter,
    labels: &Labels,
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
    let by_type = alloc::rc::Rc::new(converter.clone());
    target.add_dynamically_legal_op::<UnrealizedConversionCast, _>(move |op: &Operation| {
        DynamicLegalityResult::legal_if(by_type.is_legal_op(op))
    });
    let branches = labels.clone();
    target.add_dynamically_legal_op::<wasm::Br, _>(move |op: &Operation| {
        DynamicLegalityResult::legal_if(labels::names_its_target(op, &branches))
    });
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
/// naming the well-known symbols `symbols` of the modules they convert, and sharing the table of
/// labels `labels` with the target ([conversion_target]).
///
/// Each area of the lowering adds its own from a module of its own under `convert/`. The patterns
/// of control flow record the constructs they create in `labels` ([labels]), so the patterns of
/// one call convert one module.
pub fn populate_patterns(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
    target: &WasmTarget,
    symbols: &alloc::rc::Rc<WellKnownSymbols>,
    labels: &Labels,
) -> Result<(), Report> {
    calls::populate(patterns, converter)?;
    int::populate(patterns, converter, target)?;
    float::populate(patterns, converter)?;
    composed::populate(patterns, converter)?;
    memory::populate(patterns, converter, target, symbols)?;
    symbols::populate(patterns, converter, symbols)?;
    control::populate(patterns, converter, labels)?;
    Ok(())
}

/// The well-known symbols of one `builtin.module` that the patterns of `convert-to-wasm` name.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct ModuleSymbols {
    /// The linear memory, `@__linear_memory`; `None` when no function the module directly holds
    /// accesses memory.
    pub memory: Option<MemoryRef>,
    /// The indirect-function table, `@__indirect_function_table`; `None` when no function the
    /// module directly holds makes an indirect call.
    pub table: Option<TableRef>,
}

/// The well-known symbols ([crate::symbols]) that the patterns of `convert-to-wasm` name, by the
/// module that directly holds the function whose code names them: each found in that module, or
/// declared there before the conversion starts.
#[derive(Clone, Default)]
pub struct WellKnownSymbols {
    modules: FxHashMap<OperationRef, ModuleSymbols>,
    declared: bool,
}

impl WellKnownSymbols {
    /// For `root` and each module nested in it, outside global initializers, whether a function
    /// it directly holds accesses memory and whether one makes an indirect call.
    fn needs(root: ModuleRef) -> Vec<(ModuleRef, bool, bool)> {
        let mut needs: Vec<(ModuleRef, bool, bool)> = Vec::new();
        let _ = root.as_operation_ref().raw_prewalk::<Forward, _, _>(|op: OperationRef| {
            let op_ref = op.borrow();
            if op_ref.is::<GlobalVariable>() || op_ref.is::<wasm::Global>() {
                return WalkResult::<()>::Skip;
            }
            let memory = op_ref.is::<mem::Load>()
                || op_ref.is::<mem::Store>()
                || op_ref.is::<mem::MemCpy>()
                || op_ref.is::<mem::MemMove>()
                || op_ref.is::<mem::MemSet>();
            let table = op_ref.is::<CallIndirect>() || op_ref.is::<TailCallIndirect>();
            if !(memory || table) {
                return WalkResult::Continue(());
            }
            let Some(module) = op_ref.nearest_parent_op::<Module>() else {
                return WalkResult::Continue(());
            };
            match needs.iter_mut().find(|(found, ..)| *found == module) {
                Some((_, m, t)) => {
                    *m |= memory;
                    *t |= table;
                }
                None => needs.push((module, memory, table)),
            }
            WalkResult::Continue(())
        });
        needs
    }

    /// Checks that each module under `root` (`root` included) that needs a well-known symbol
    /// holds no other symbol of its name ([crate::symbols]); changes nothing.
    pub fn check(root: ModuleRef) -> Result<(), Report> {
        for (module, memory, table) in Self::needs(root) {
            if memory {
                crate::symbols::find_linear_memory(module)?;
            }
            if table {
                crate::symbols::find_indirect_function_table(module)?;
            }
        }
        Ok(())
    }

    /// The symbols the functions of `root` and of the modules nested in it need: each found in the
    /// module that directly holds the function, or declared there when it is needed and missing.
    /// An error when such a module holds another symbol of a well-known name
    /// ([crate::symbols]); [Self::check] finds it without changing anything.
    pub fn of(root: ModuleRef) -> Result<Self, Report> {
        let mut symbols = Self::default();
        for (module, memory, table) in Self::needs(root) {
            let mut found = ModuleSymbols::default();
            if memory {
                symbols.declared |= crate::symbols::find_linear_memory(module)?.is_none();
                found.memory = Some(crate::symbols::linear_memory(module)?);
            }
            if table {
                symbols.declared |= crate::symbols::find_indirect_function_table(module)?.is_none();
                found.table = Some(crate::symbols::indirect_function_table(module)?);
            }
            symbols.modules.insert(module.as_operation_ref(), found);
        }
        Ok(symbols)
    }

    /// The symbols of `module`.
    pub fn of_module(&self, module: ModuleRef) -> ModuleSymbols {
        self.modules.get(&module.as_operation_ref()).copied().unwrap_or_default()
    }

    /// The symbols of the module nearest `op`, which directly holds its function.
    pub fn of_op(&self, op: OperationRef) -> ModuleSymbols {
        match op.borrow().nearest_parent_op::<Module>() {
            Some(module) => self.of_module(module),
            None => ModuleSymbols::default(),
        }
    }

    /// Whether [Self::of] declared a symbol.
    pub fn declared(&self) -> bool {
        self.declared
    }
}

/// The check of [ConvertToWasm] of the modules under one root of the conversion.
struct Check<'a> {
    /// The module the conversion is rooted at.
    root: OperationRef,
    target: &'a WasmTarget,
    converter: &'a TypeConverter,
    conversion_target: &'a ConversionTarget,
    /// The names of the operations some pattern is rooted at.
    roots: &'a FxHashSet<OperationName>,
}

impl Check<'_> {
    /// Checks every function of the root's module, and of the modules nested in it.
    fn module(&self) -> Result<(), Report> {
        let result = self.root.raw_prewalk::<Forward, _, _>(|op: OperationRef| -> WalkResult {
            let is_function = op.borrow().is::<Function>();
            if is_function {
                return match self.function(op) {
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

    /// Checks the function `function` (see [ConvertToWasm]).
    fn function(&self, function: OperationRef) -> Result<(), Report> {
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
        if !self.converter.is_legal_op(&op) {
            check_users_inside(function, self.root)?;
        }
        if func.is_declaration() {
            return Ok(());
        }
        let body = func.body().as_region_ref();
        drop(op);
        check_region_arguments(context_of(function), function, body, &name)?;
        let dropped = dropped_regions(function);
        let result = function.raw_prewalk::<Forward, _, _>(|inner: OperationRef| -> WalkResult {
            if inner == function {
                return WalkResult::Continue(());
            }
            if is_in_dropped_region(inner, function, &dropped) {
                return WalkResult::Skip;
            }
            match self.operation(inner) {
                Ok(()) => WalkResult::Continue(()),
                Err(error) => WalkResult::Break(error),
            }
        });
        result.into_result()
    }

    /// Checks one operation of a function body (see [ConvertToWasm]): what the patterns refuse,
    /// and the carriers of its results and of the arguments of the blocks of its regions.
    fn operation(&self, op: OperationRef) -> Result<(), Report> {
        let context = context_of(op);
        {
            let op_ref = op.borrow();
            if op_ref.is::<mem::Alloca>() {
                return Err(memory::alloca_error(&op_ref));
            }
            if op_ref.is::<runic_dialect_cf::Br>()
                || op_ref.is::<runic_dialect_cf::CondBr>()
                || op_ref.is::<runic_dialect_cf::Switch>()
            {
                return Err(control::unstructured(&op_ref));
            }
            let accessed = if let Some(load) = op_ref.downcast_ref::<mem::Load>() {
                Some(load.result().ty())
            } else {
                op_ref.downcast_ref::<mem::Store>().map(|store| store.value().ty())
            };
            if let Some(ty) = accessed {
                memory::access(&context, self.target.layout(), ty, &op_ref)?;
            }
            memory::gep_layout(self.target.layout(), &op_ref)?;
            memory::measure(self.target.layout(), &op_ref)?;
            if let Some(signature) = call_signature(&op_ref) {
                calls::Convention::of(&signature, &op_ref)?;
            }
            if let Some(exit) = op_ref.downcast_ref::<runic_dialect_scf::Break>() {
                labels::check_exit(&op_ref, *exit.get_levels(), RegionExitKind::Break)?;
            }
            if let Some(exit) = op_ref.downcast_ref::<runic_dialect_scf::Continue>() {
                labels::check_exit(&op_ref, *exit.get_levels(), RegionExitKind::Continue)?;
            }
            if let Some(constant) = op_ref.downcast_ref::<runic_dialect_arith::Constant>() {
                calls::check_constant(&op_ref, *constant.get_value(), constant.result().ty())?;
            }
            if let Some(ret) = op_ref.downcast_ref::<RetImm>() {
                let ty = runic_ir::Attribute::ty(&*ret.value());
                calls::check_constant(&op_ref, *ret.get_value(), ty)?;
            }
            if op_ref.is::<UnrealizedConversionCast>() {
                check_input_cast(&op_ref)?;
            }
            drop(op_ref);
            control::check_structured(op)?;
            let op_ref = op.borrow();
            // A cast written before the conversion (`wasm-lower-frames` writes some) folds away
            // with the casts the conversion leaves; [check_input_cast] refuses one it cannot.
            if !op_ref.is::<UnrealizedConversionCast>()
                && !self.conversion_target.is_legal(&op_ref)
                && !self.roots.contains(&op_ref.name())
            {
                return Err(report!(
                    message: (format!(
                        "`{NAME}` cannot lower `{}`: no pattern of the pass converts it, and it is \
                         not in Wasm form",
                        op_ref.name()
                    )),
                    labels: [primary(
                        op_ref.span(),
                        format!("no pattern of `{NAME}` converts this `{}`", op_ref.name())
                    )],
                ));
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
}

/// The `after` regions of the `scf.while`s of `function` whose `before` body does not end in
/// `scf.condition`: the lowering drops them (see [control]).
fn dropped_regions(function: OperationRef) -> Vec<runic_ir::RegionRef> {
    let mut dropped = Vec::new();
    let _ = function.raw_prewalk::<Forward, _, _>(|op: OperationRef| {
        let op_ref = op.borrow();
        if op_ref.is::<runic_dialect_scf::While>() {
            let regions = op_ref.regions().iter().map(|r| r.as_region_ref()).collect::<Vec<_>>();
            if let [before, after] = regions.as_slice()
                && !control::ends_in_condition(*before)
            {
                dropped.push(*after);
            }
        }
        WalkResult::<()>::Continue(())
    });
    dropped
}

/// Whether `op`, under `function`, is in one of `dropped`.
fn is_in_dropped_region(
    op: OperationRef,
    function: OperationRef,
    dropped: &[runic_ir::RegionRef],
) -> bool {
    if dropped.is_empty() {
        return false;
    }
    let mut region = op.borrow().parent_region();
    while let Some(current) = region {
        if dropped.contains(&current) {
            return true;
        }
        let Some(owner) = current.borrow().parent() else {
            return false;
        };
        if owner == function {
            return false;
        }
        region = owner.borrow().parent_region();
    }
    false
}

/// An error, naming the function and the user, when `function`, whose signature the conversion
/// changes, is named by an operation outside `root`, directly or through an alias of it: the
/// refusal of the builtin function pattern, found before anything changes.
fn check_users_inside(function: OperationRef, root: OperationRef) -> Result<(), Report> {
    let mut pending = alloc::vec![function];
    let mut seen = alloc::vec![function];
    while let Some(symbol) = pending.pop() {
        let owners = {
            let symbol = symbol.borrow();
            let Some(symbol) = symbol.as_trait::<dyn Symbol>() else {
                continue;
            };
            symbol.iter_uses().map(|user| user.owner).collect::<Vec<OperationRef>>()
        };
        for owner in owners {
            if !root.borrow().is_ancestor_of(&owner.borrow()) {
                let function_ref = function.borrow();
                let owner_ref = owner.borrow();
                let name_of = |op: &Operation| {
                    op.as_trait::<dyn Symbol>().map_or_else(
                        || format!("`{}`", op.name()),
                        |symbol| format!("`@{}`", symbol.name()),
                    )
                };
                let holder = owner_ref.nearest_parent_op::<Function>().map_or_else(
                    || String::from("no function"),
                    |f| name_of(&f.as_operation_ref().borrow()),
                );
                return Err(report!(
                    message: (format!(
                        "`{NAME}` changes the signature of {}, but a `{}` in {holder}, outside \
                         the module {} that is converted, names it: the modules that name each \
                         other's functions are converted together, as one module",
                        name_of(&function_ref),
                        owner_ref.name(),
                        name_of(&root.borrow()),
                    )),
                    labels: [
                        primary(owner_ref.span(), "this operation is outside the module"),
                        context(function_ref.span(), "the function whose signature changes"),
                    ],
                ));
            }
            if owner.borrow().is::<FunctionAlias>() && !seen.contains(&owner) {
                seen.push(owner);
                pending.push(owner);
            }
        }
    }
    Ok(())
}

/// An error naming `op`, a `builtin.unrealized_conversion_cast` written before the conversion,
/// when one of its results is a narrow integer: its carrier is wider, and nothing says that the
/// cast's input has the bits above the integer's width clear, which carried form needs.
fn check_input_cast(op: &Operation) -> Result<(), Report> {
    for result in op.results().iter() {
        let ty = result.borrow().ty();
        if calls::narrow(ty).is_some() {
            return Err(report!(
                message: (format!(
                    "`{NAME}` cannot lower a `builtin.unrealized_conversion_cast` to `{ty}` written \
                     before it: nothing says the bits of its input above the width of `{ty}` are \
                     clear"
                )),
                labels: [primary(
                    op.span(),
                    format!("this cast gives a value of type `{ty}`; `arith.trunc` makes one")
                )],
            ));
        }
    }
    Ok(())
}

/// The context of `op`.
fn context_of(op: OperationRef) -> alloc::rc::Rc<Context> {
    op.borrow().context_rc()
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
