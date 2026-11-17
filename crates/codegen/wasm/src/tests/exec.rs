//! Running what the emitter writes: a test-only assembly of the functions of one lowered
//! `builtin.module` into a module description, and `wasmi` to run it, so that the emitted code can
//! be compared with the evaluator ([super::eval]).
//!
//! The assembly mirrors the conventions of the evaluator, which a test states in a [Setup]:
//!
//! - Every function definition of the module is exported under its symbol name; a declaration is
//!   an import from `env` (field: its name) that traps when called.
//! - Every `wasm.memory` of the module, and the memory of a setup that has one, is one memory of
//!   the setup's size, exported as `memory`; its initial bytes are the setup's ([Setup::write]),
//!   as active data at fixed addresses. The evaluator's memory is a byte vector; the setup's size
//!   is a whole number of 64 KiB pages, so that an access traps at the same address in both.
//! - Every `wasm.table` is a table whose slot `i` holds the function the evaluator's table holds
//!   at `i` (the evaluator gives a slot to each function the first time a `wasm.func_index` names
//!   it, from 0, so the module is assembled after the evaluator has run). A function named by a
//!   `wasm.func_index` that the evaluator did not reach gets a slot after the evaluator's, and the
//!   table is as large as the evaluator's: an index past it traps in both.
//! - Every `wasm.global` is a global of its type and mutability whose initial value is the one the
//!   setup gives the evaluator (zero when it gives none); each is exported, so that the final
//!   values can be compared with the evaluator's.
//! - A `builtin.global_variable` is data at the address the setup gives it ([Setup::data]), or
//!   passive data of the bytes it gives ([Setup::passive]).
//!
//! [agree] runs one function both ways, twice for the emitted module (with reuse of locals on and
//! off), and asserts the same [Outcome]: values bit for bit (a NaN as a NaN of its type), a trap
//! for a trap, and with [Setup::compare_memory] the same memory, and always the same final values
//! of the globals the setup names. Every emitted module is validated before it runs.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use runic_dialect_wasm as wasm;
use runic_ir::{
    Forward, FxHashMap, Operation, OperationRef, RawWalk, Report, Spanned, Symbol as _, WalkResult,
    diagnostics::report,
    dialects::builtin::{Function, GlobalVariable, attributes::SymbolRefAttr},
    testing::text::describe,
};

use super::eval::{Machine, Outcome, Val, run_in};
use crate::emit::{
    body::{BodyConfig, encode_function},
    desc::{
        ConstExpr, DataEntry, DataId, DataPlacement, ElemSegment, Entity, Export, ExportTarget,
        FixupKind, FuncId, FuncType, Function as FunctionDesc, FunctionBody, Global, GlobalId,
        Import, Limits, Memory, ModuleDesc, RefType, Table, TableId, ValType,
    },
    instr::{Entities, func_type_of, val_type},
    validate::validate,
    write::{Resolved, write_module},
};

/// A page of linear memory, in bytes.
pub(super) const PAGE: usize = 65_536;

/// What a test gives the evaluator and the emitted module alike.
#[derive(Clone, Debug)]
pub(super) struct Setup {
    /// The size of the memory in bytes, a multiple of [PAGE]; 0 for no memory.
    pub memory: usize,
    /// Bytes written into the memory before the run, at their addresses.
    pub writes: Vec<(u32, Vec<u8>)>,
    /// The address of each data symbol, by name.
    pub data: Vec<(String, u32)>,
    /// The bytes of each passive data symbol, by name (the evaluator does not run
    /// `wasm.memory_init`).
    pub passive: Vec<(String, Vec<u8>)>,
    /// The initial value of each global, by name.
    pub globals: Vec<(String, Val)>,
    /// Whether the memory after the run is compared.
    pub compare_memory: bool,
    /// The functions, by name, whose bodies are not emitted but replaced by `unreachable`:
    /// functions the run does not reach, which hold what the emitter does not write yet.
    pub stubs: Vec<String>,
}

impl Default for Setup {
    /// One page of zeroed memory, no data, no globals; the memory is not compared.
    fn default() -> Self {
        Self {
            memory: PAGE,
            writes: Vec::new(),
            data: Vec::new(),
            passive: Vec::new(),
            globals: Vec::new(),
            compare_memory: false,
            stubs: Vec::new(),
        }
    }
}

impl Setup {
    /// The setup with `bytes` written at `address`.
    pub fn write(mut self, address: u32, bytes: &[u8]) -> Self {
        self.writes.push((address, bytes.to_vec()));
        self
    }

    /// The setup with the data symbol `name` at `address`.
    pub fn data(mut self, name: &str, address: u32) -> Self {
        self.data.push((name.to_string(), address));
        self
    }

    /// The setup with the passive data symbol `name` of the bytes `bytes`.
    pub fn passive(mut self, name: &str, bytes: &[u8]) -> Self {
        self.passive.push((name.to_string(), bytes.to_vec()));
        self
    }

    /// The setup with the global `name` starting at `value`.
    pub fn global(mut self, name: &str, value: Val) -> Self {
        self.globals.push((name.to_string(), value));
        self
    }

    /// The setup whose memory is compared after the run.
    pub fn comparing_memory(mut self) -> Self {
        self.compare_memory = true;
        self
    }

    /// The evaluator's machine of this setup.
    pub fn machine(&self) -> Machine {
        assert_eq!(self.memory % PAGE, 0, "the memory of a setup is a whole number of pages");
        let mut machine = Machine::new(self.memory);
        for (address, bytes) in &self.writes {
            machine.write(*address, bytes);
        }
        for (name, address) in &self.data {
            machine.data.insert(name.clone(), *address);
        }
        for (name, value) in &self.globals {
            machine.globals.insert(name.clone(), *value);
        }
        machine
    }
}

/// The entities of an assembled module, by the name of the symbol that stands for each.
#[derive(Default)]
struct Unit {
    functions: FxHashMap<String, FuncId>,
    globals: FxHashMap<String, GlobalId>,
    tables: FxHashMap<String, TableId>,
    data: FxHashMap<String, DataId>,
}

/// The error of a symbol the assembly has no entity for.
fn missing(user: &Operation, kind: &str, symbol: &SymbolRefAttr) -> Report {
    report!(
        message: (format!("the test assembly has no {kind} `@{}`", symbol.path().name().as_str())),
        labels: [primary(user.span(), String::from("named here"))],
    )
}

/// The name of the symbol `symbol` names: its last component.
fn name_of(symbol: &SymbolRefAttr) -> String {
    symbol.path().name().as_str().to_string()
}

impl Entities for Unit {
    fn function(&mut self, user: &Operation, callee: &SymbolRefAttr) -> Result<FuncId, Report> {
        self.functions
            .get(&name_of(callee))
            .copied()
            .ok_or_else(|| missing(user, "function", callee))
    }

    fn global(&mut self, user: &Operation, global: &SymbolRefAttr) -> Result<GlobalId, Report> {
        self.globals
            .get(&name_of(global))
            .copied()
            .ok_or_else(|| missing(user, "global", global))
    }

    fn table(&mut self, user: &Operation, table: &SymbolRefAttr) -> Result<TableId, Report> {
        self.tables
            .get(&name_of(table))
            .copied()
            .ok_or_else(|| missing(user, "table", table))
    }

    fn data(&mut self, user: &Operation, data: &SymbolRefAttr) -> Result<DataId, Report> {
        self.data
            .get(&name_of(data))
            .copied()
            .ok_or_else(|| missing(user, "data symbol", data))
    }
}

/// The operations a module directly holds.
pub(super) fn ops_of(module: OperationRef) -> Vec<OperationRef> {
    let module = module.borrow();
    let mut ops = Vec::new();
    for region in module.regions().iter() {
        for block in region.body().iter() {
            ops.extend(block.body().iter().map(|op| op.as_operation_ref()));
        }
    }
    ops
}

/// The module that holds `function`.
pub(super) fn module_of(function: OperationRef) -> OperationRef {
    function.borrow().parent_op().expect("a function in a module")
}

/// The value `value` as a constant expression of type `ty`.
fn constant(ty: ValType, value: Option<Val>) -> ConstExpr {
    use wasm_encoder::{Ieee32, Ieee64, Instruction as I};
    let bits = value.map_or(0, |value| value.bits);
    let mut expr = ConstExpr::default();
    let mut writer = expr.writer();
    match ty {
        ValType::I32 => writer.instruction(&I::I32Const(bits as u32 as i32)),
        ValType::I64 => writer.instruction(&I::I64Const(bits as i64)),
        ValType::F32 => writer.instruction(&I::F32Const(Ieee32::new(bits as u32))),
        ValType::F64 => writer.instruction(&I::F64Const(Ieee64::new(bits))),
        ValType::FuncRef => writer.instruction(&I::RefNull(wasm_encoder::HeapType::FUNC)),
        ValType::ExternRef => writer.instruction(&I::RefNull(wasm_encoder::HeapType::EXTERN)),
    };
    expr
}

/// An assembled module: its bytes, and the imported functions with their types (which the
/// runner defines as traps).
pub(super) struct Assembled {
    /// The module, validated.
    pub bytes: Vec<u8>,
    /// The field and type of each imported function.
    pub imports: Vec<(String, FuncType)>,
    /// The name of each global, as exported (`global <name>`).
    pub globals: Vec<String>,
    /// Whether the module has a memory.
    pub has_memory: bool,
    /// The features the description recorded.
    pub features: crate::emit::desc::FeatureSet,
    /// Whether the description asks for a data-count section.
    pub uses_data_count: bool,
}

/// Assembles the functions of `module` under `setup`, with the evaluator's table `table` (the
/// function of each slot), encoding the bodies with `config`; validated.
pub(super) fn assemble(
    module: OperationRef,
    setup: &Setup,
    table: &[OperationRef],
    config: &BodyConfig,
) -> Result<Assembled, Report> {
    let mut desc = ModuleDesc::new();
    desc.names = true;
    let mut unit = Unit::default();
    let mut resolved = Resolved::new();
    let mut defined = Vec::new();
    let mut imports = Vec::new();
    let mut globals = Vec::new();
    let mut ref_funcs = Vec::new();
    let mut function_ops = FxHashMap::default();
    let mut has_memory = false;
    let mut tables = Vec::new();
    for op_ref in ops_of(module) {
        let op = op_ref.borrow();
        if let Some(function) = op.downcast_ref::<Function>() {
            let name = function.get_name().as_str().to_string();
            let ty = func_type_of(&op, &function.get_signature())?;
            let ty = desc.intern_type(ty);
            let entity = if function.is_declaration() {
                imports.push((name.clone(), desc.ty(ty).clone()));
                Entity::Import(Import::new("env", name.clone()))
            } else {
                Entity::Definition(FunctionBody::default())
            };
            let id = desc.add_function(FunctionDesc {
                ty,
                name: name.clone(),
                entity,
            });
            if !function.is_declaration() {
                defined.push((op_ref, id));
                desc.add_export(Export {
                    name: name.clone(),
                    target: ExportTarget::Function(id),
                });
            }
            unit.functions.insert(name, id);
            function_ops.insert(op_ref, id);
        } else if op.downcast_ref::<wasm::Memory>().is_some() {
            has_memory = true;
        } else if let Some(table_op) = op.downcast_ref::<wasm::Table>() {
            let name = table_op.get_name().as_str().to_string();
            let element = if val_type(*table_op.get_element_type()) == Some(ValType::ExternRef) {
                RefType::ExternRef
            } else {
                RefType::FuncRef
            };
            let id = desc.add_table(Table {
                element,
                limits: Limits::at_least(0),
                name: name.clone(),
                entity: Entity::Definition(()),
            });
            tables.push((id, element, table_op.get_limits().min()));
            unit.tables.insert(name, id);
        } else if let Some(global) = op.downcast_ref::<wasm::Global>() {
            let name = global.get_name().as_str().to_string();
            let ty = val_type(*global.get_ty()).expect("a global of a Wasm value type");
            let value = setup.globals.iter().find(|(given, _)| *given == name).map(|(_, v)| *v);
            let id = desc.add_global(Global {
                ty,
                mutable: global.get_mutability().is_mut(),
                name: name.clone(),
                entity: Entity::Definition(constant(ty, value)),
            });
            let export = format!("global {name}");
            desc.add_export(Export {
                name: export.clone(),
                target: ExportTarget::Global(id),
            });
            globals.push(export);
            unit.globals.insert(name, id);
        } else if let Some(variable) = op.downcast_ref::<GlobalVariable>() {
            let name = variable.get_name().as_str().to_string();
            if let Some((_, bytes)) = setup.passive.iter().find(|(given, _)| *given == name) {
                let id = desc.add_data(DataEntry {
                    name: name.clone(),
                    bytes: bytes.clone(),
                    align: 1,
                    section: None,
                    placement: DataPlacement::Passive,
                    fixups: Vec::new(),
                });
                resolved.set_data_address(id, 0);
                unit.data.insert(name, id);
            } else if let Some((_, address)) = setup.data.iter().find(|(given, _)| *given == name) {
                let id = desc.add_data(DataEntry {
                    name: name.clone(),
                    bytes: Vec::new(),
                    align: 1,
                    section: None,
                    placement: DataPlacement::Layout,
                    fixups: Vec::new(),
                });
                resolved.set_data_address(id, *address);
                unit.data.insert(name, id);
            }
        }
    }
    if has_memory || setup.memory > 0 {
        has_memory = true;
        let memory = desc.add_memory(Memory {
            limits: Limits {
                min: (setup.memory / PAGE) as u32,
                max: None,
            },
            name: String::from("memory"),
            entity: Entity::Definition(()),
        });
        desc.add_export(Export {
            name: String::from("memory"),
            target: ExportTarget::Memory(memory),
        });
        for (address, bytes) in &setup.writes {
            desc.add_data(DataEntry {
                name: String::new(),
                bytes: bytes.clone(),
                align: 1,
                section: None,
                placement: DataPlacement::Fixed(*address),
                fixups: Vec::new(),
            });
        }
    }
    for (op, id) in &defined {
        if setup.stubs.contains(&desc.function(*id).name) {
            let mut body = FunctionBody::default();
            body.writer()
                .instruction(&wasm_encoder::Instruction::Unreachable)
                .instruction(&wasm_encoder::Instruction::End);
            desc.function_mut(*id).entity = Entity::Definition(body);
            continue;
        }
        let body = encode_function(*op, &mut desc, &mut unit, config)?;
        desc.function_mut(*id).entity = Entity::Definition(body);
        let _ = op.raw_prewalk::<Forward, _, _>(|nested: OperationRef| -> WalkResult<Report> {
            if let Some(ref_func) = nested.borrow().downcast_ref::<wasm::RefFunc>()
                && let Some(id) = unit.functions.get(&name_of(&ref_func.callee()))
                && !ref_funcs.contains(id)
            {
                ref_funcs.push(*id);
            }
            WalkResult::Continue(())
        });
    }
    // The table: the evaluator's slots, then a slot for each function named by a
    // `wasm.func_index` that the evaluator did not reach.
    let mut slots = table
        .iter()
        .map(|function| *function_ops.get(function).expect("a function of the module"))
        .collect::<Vec<_>>();
    let size = slots.len() as u32;
    let mut named = Vec::new();
    for function in desc.functions() {
        if let Some(body) = function.entity.definition() {
            for fixup in &body.fixups {
                if let FixupKind::TableSlot(target) = fixup.kind {
                    named.push(target);
                }
            }
        }
    }
    for target in named {
        if !slots.contains(&target) {
            slots.push(target);
        }
    }
    for (slot, function) in slots.iter().enumerate() {
        resolved.set_table_slot(*function, slot as u32);
    }
    for (index, (id, element, min)) in tables.iter().enumerate() {
        let min = u32::try_from(*min).unwrap_or(u32::MAX).max(size);
        desc.table_mut(*id).limits = Limits::at_least(min);
        if index == 0 && *element == RefType::FuncRef && size > 0 {
            desc.add_element(ElemSegment::Active {
                table: *id,
                offset: 0,
                functions: slots[..size as usize].to_vec(),
            });
        }
    }
    if !ref_funcs.is_empty() {
        desc.add_element(ElemSegment::Declarative(ref_funcs));
    }
    let bytes = write_module(&desc, &resolved)?;
    validate(&bytes)?;
    Ok(Assembled {
        bytes,
        imports,
        globals,
        has_memory,
        features: desc.features,
        uses_data_count: desc.uses_data_count,
    })
}

/// What a run of the emitted module gave: the outcome, the memory and the final values of the
/// globals.
#[derive(Clone, Debug)]
pub(super) struct Ran {
    /// The values returned, or a trap.
    pub outcome: Outcome,
    /// The bytes of the memory after the run, if the module has one.
    pub memory: Option<Vec<u8>>,
    /// The final value of each global, by name.
    pub globals: Vec<(String, Val)>,
}

/// The `wasmi` value type of `ty`.
fn wasmi_type(ty: ValType) -> wasmi::ValType {
    match ty {
        ValType::I32 => wasmi::ValType::I32,
        ValType::I64 => wasmi::ValType::I64,
        ValType::F32 => wasmi::ValType::F32,
        ValType::F64 => wasmi::ValType::F64,
        ValType::FuncRef => wasmi::ValType::FuncRef,
        ValType::ExternRef => wasmi::ValType::ExternRef,
    }
}

/// `value` for `wasmi`.
fn to_wasmi(value: Val) -> wasmi::Val {
    match (value.width, value.float) {
        (32, false) => wasmi::Val::I32(value.bits as u32 as i32),
        (64, false) => wasmi::Val::I64(value.bits as i64),
        (32, true) => wasmi::Val::F32(wasmi::F32::from_bits(value.bits as u32)),
        _ => wasmi::Val::F64(wasmi::F64::from_bits(value.bits)),
    }
}

/// `value` from `wasmi`; a reference is not a value a test compares.
fn from_wasmi(value: &wasmi::Val) -> Val {
    match value {
        wasmi::Val::I32(value) => Val::i32(*value as u32),
        wasmi::Val::I64(value) => Val::i64(*value as u64),
        wasmi::Val::F32(value) => Val::float_bits(u64::from(value.to_bits()), 32),
        wasmi::Val::F64(value) => Val::float_bits(value.to_bits(), 64),
        other => panic!("a function returned the reference {other:?}, which a test cannot compare"),
    }
}

/// Runs the export `name` of `assembled` on `arguments` with `wasmi`.
pub(super) fn run_wasmi(assembled: &Assembled, name: &str, arguments: &[Val]) -> Ran {
    let engine = wasmi::Engine::default();
    let module = wasmi::Module::new(&engine, &assembled.bytes[..])
        .unwrap_or_else(|error| panic!("wasmi rejects the emitted module: {error}"));
    let mut store = wasmi::Store::new(&engine, ());
    let mut linker = wasmi::Linker::<()>::new(&engine);
    for (field, ty) in &assembled.imports {
        let ty = wasmi::FuncType::new(
            ty.params.iter().map(|ty| wasmi_type(*ty)),
            ty.results.iter().map(|ty| wasmi_type(*ty)),
        );
        linker
            .func_new("env", field, ty, |_, _, _| {
                Err(wasmi::Error::from(wasmi::TrapCode::UnreachableCodeReached))
            })
            .expect("one definition per import");
    }
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .unwrap_or_else(|error| panic!("the emitted module does not instantiate: {error}"));
    let func = instance
        .get_func(&store, name)
        .unwrap_or_else(|| panic!("the emitted module exports no function `{name}`"));
    let ty = func.ty(&store);
    let inputs = arguments.iter().map(|value| to_wasmi(*value)).collect::<Vec<_>>();
    let mut outputs = ty.results().iter().map(|ty| wasmi::Val::default(*ty)).collect::<Vec<_>>();
    let outcome = match func.call(&mut store, &inputs, &mut outputs) {
        Ok(()) => Outcome::Values(outputs.iter().map(from_wasmi).collect()),
        Err(error) if error.as_trap_code().is_some() => Outcome::Trap,
        Err(error) => panic!("running `{name}` failed without a trap: {error}"),
    };
    let memory = assembled.has_memory.then(|| {
        let memory = instance.get_memory(&store, "memory").expect("the memory is exported");
        memory.data(&store).to_vec()
    });
    let globals = assembled
        .globals
        .iter()
        .map(|export| {
            let global = instance.get_global(&store, export).expect("the global is exported");
            let name = export.strip_prefix("global ").unwrap_or(export).to_string();
            (name, from_wasmi(&global.get(&store)))
        })
        .collect();
    Ran {
        outcome,
        memory,
        globals,
    }
}

/// Whether `a` and `b` are the same outcome: values bit for bit, a NaN as a NaN of the same type,
/// a trap as a trap.
pub(super) fn same(a: &Outcome, b: &Outcome) -> bool {
    match (a, b) {
        (Outcome::Trap, Outcome::Trap) => true,
        (Outcome::Values(a), Outcome::Values(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_value(*a, *b))
        }
        _ => false,
    }
}

/// Whether `a` and `b` are the same value: bit for bit, or both NaNs of one type.
pub(super) fn same_value(a: Val, b: Val) -> bool {
    a == b || (a.is_nan() && b.is_nan() && a.width == b.width)
}

/// The name of the function `function`.
fn function_name(function: OperationRef) -> String {
    function
        .borrow()
        .downcast_ref::<Function>()
        .expect("a builtin.function")
        .get_name()
        .as_str()
        .to_string()
}

/// The two configurations every emitted module is run in: with reuse of locals, and without.
pub(super) const CONFIGS: [BodyConfig; 2] = [
    BodyConfig { reuse_locals: true },
    BodyConfig {
        reuse_locals: false,
    },
];

/// The emitted module of `function`'s module under `setup` and `table`, in `config`; a failure
/// to assemble panics with its description.
#[track_caller]
pub(super) fn assembled(
    function: OperationRef,
    setup: &Setup,
    table: &[OperationRef],
    config: &BodyConfig,
) -> Assembled {
    assemble(module_of(function), setup, table, config)
        .unwrap_or_else(|report| panic!("{}", describe(&report)))
}

/// Runs `function` on `arguments` with the evaluator and the emitted module (with reuse of
/// locals on and off), asserts that they agree (see the [module docs](self)), and gives the
/// outcome.
#[track_caller]
pub(super) fn agree(function: OperationRef, arguments: &[Val]) -> Outcome {
    agree_in(&Setup::default(), function, arguments)
}

/// [agree] under `setup`.
#[track_caller]
pub(super) fn agree_in(setup: &Setup, function: OperationRef, arguments: &[Val]) -> Outcome {
    let mut cache = Cache::default();
    agree_cached(&mut cache, setup, function, arguments)
}

/// The emitted modules of one function by the evaluator's table they were assembled with, one
/// per configuration.
#[derive(Default)]
pub(super) struct Cache {
    modules: Vec<(Vec<OperationRef>, Vec<Assembled>)>,
}

/// [agree_in], reusing the modules of `cache` when the evaluator's table is one they were
/// assembled with.
#[track_caller]
pub(super) fn agree_cached(
    cache: &mut Cache,
    setup: &Setup,
    function: OperationRef,
    arguments: &[Val],
) -> Outcome {
    let mut machine = setup.machine();
    let expected = run_in(&mut machine, function, arguments);
    let table = machine.table.clone();
    let position = match cache.modules.iter().position(|(slots, _)| *slots == table) {
        Some(position) => position,
        None => {
            let modules = CONFIGS
                .iter()
                .map(|config| assembled(function, setup, &table, config))
                .collect();
            cache.modules.push((table, modules));
            cache.modules.len() - 1
        }
    };
    let name = function_name(function);
    for (config, module) in CONFIGS.iter().zip(&cache.modules[position].1) {
        let ran = run_wasmi(module, &name, arguments);
        assert!(
            same(&expected, &ran.outcome),
            "`@{name}` on {arguments:?} ({config:?}): the evaluator gives {expected:?}, the \
             emitted module {:?}",
            ran.outcome
        );
        if setup.compare_memory {
            let memory = ran.memory.as_ref().expect("the module has a memory");
            if let Some(at) = memory.iter().zip(&machine.memory).position(|(a, b)| a != b) {
                panic!(
                    "`@{name}` on {arguments:?} ({config:?}): the memory differs first at {at}: \
                     the evaluator has {:#04x}, the emitted module {:#04x}",
                    machine.memory[at], memory[at]
                );
            }
            assert_eq!(memory.len(), machine.memory.len(), "the size of the memory");
        }
        for (global, value) in &ran.globals {
            if let Some(expected) = machine.globals.get(global) {
                assert!(
                    same_value(*expected, *value),
                    "`@{name}` on {arguments:?} ({config:?}): the global `@{global}` ends as \
                     {expected:?} in the evaluator and {value:?} in the emitted module"
                );
            }
        }
    }
    expected
}

/// Runs the emitted module of `function` alone (for the operations the evaluator does not
/// interpret), with reuse of locals on and off, asserting that both give the same, and gives what
/// the first gave. `table` is the table's slots.
#[track_caller]
pub(super) fn emitted(
    setup: &Setup,
    function: OperationRef,
    table: &[OperationRef],
    arguments: &[Val],
) -> Ran {
    let name = function_name(function);
    let runs = CONFIGS
        .iter()
        .map(|config| run_wasmi(&assembled(function, setup, table, config), &name, arguments))
        .collect::<Vec<_>>();
    assert!(same(&runs[0].outcome, &runs[1].outcome), "{:?} and {:?}", runs[0], runs[1]);
    assert_eq!(runs[0].memory, runs[1].memory);
    runs.into_iter().next().expect("two runs")
}

/// The text of the emitted module of `module` under `setup` (reuse of locals on), printed.
#[track_caller]
pub(super) fn text_of_module(module: OperationRef, setup: &Setup) -> String {
    let assembled = assemble(module, setup, &[], &BodyConfig::default())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    crate::text::print(&assembled.bytes).unwrap_or_else(|report| panic!("{report}"))
}
