//! One operation of Wasm form: its stack operands, and the instruction it is written as.
//!
//! [stack_operands] gives the values an operation takes from the Wasm stack, in the order they are
//! pushed (the first pushed first), which is the order of the operation's operands except for
//! five: `wasm.select` (its condition is pushed last), `wasm.call_indirect` and
//! `wasm.return_call_indirect` (their index is pushed after the arguments), `wasm.if` (its
//! condition is pushed after its inputs) and `wasm.br_table` (its index is pushed after the values
//! it carries). [encode] writes the
//! instruction of one operation through a [CodeWriter], chosen by the types of its operands and
//! results; it writes only the instruction, never the operands.
//!
//! The operations it knows are listed in [INSTRUCTIONS]; [CONTROL] lists the structured control of
//! the dialect, which a function body's walk handles itself, and [MODULE_LEVEL] the operations
//! that are entities of a module rather than instructions. Every reference an instruction makes to
//! another entity (a function, a global, a table, a type, a data entry, an address or a table
//! slot) is written as a placeholder with a fix-up ([CodeWriter::fixup_index],
//! [CodeWriter::fixup_i32_const]); the entities are found through [Entities]. The one index written
//! directly is the memory index of the memory instructions, which is 0: a module has at most one
//! memory (decision 9 of the plan), and the tool conventions have no relocation for a memory index.
//!
//! What cannot be written (an operation that is not one of Wasm form, a value that is not of a
//! Wasm value type, a load of a type from a storage that no instruction reads, an offset beyond 32
//! bits) is a [Report] whose label names the operation and says what is unsupported.

use alloc::{format, string::String, vec::Vec};

use runic_dialect_wasm as wasm;
use runic_ir::{
    Operation, Report, SourceSpan, Spanned, Type, Value, ValueRef,
    diagnostics::report,
    dialects::builtin::{
        Call, LoadLocal, StoreLocal, TailCall,
        attributes::{Signature, SymbolRefAttr},
    },
};
use wasm_encoder::{HeapType, Ieee32, Ieee64, Instruction, MemArg, MemArgTarget};

use super::desc::{
    CodeWriter, DataId, Feature, FixupKind, FuncId, FuncType, GlobalId, ModuleDesc, TableId,
    ValType,
};

/// What the body encoder asks of the unit it is encoding for: the entity each symbol an
/// instruction names stands for.
///
/// `user` is the operation that names the symbol, so that an error can point at it; `symbol` is
/// the reference as the operation holds it (the generated symbol accessors return a
/// [SymbolRefAttr]). An implementation resolves the reference and gives the entity's id in the
/// description it builds, or the error of a symbol that names no such entity.
pub trait Entities {
    /// The function `callee` names (`builtin.call`, `builtin.tail_call`, `wasm.ref_func`,
    /// `wasm.func_index`).
    fn function(&mut self, user: &Operation, callee: &SymbolRefAttr) -> Result<FuncId, Report>;
    /// The global `global` names (`wasm.global_get`, `wasm.global_set`).
    fn global(&mut self, user: &Operation, global: &SymbolRefAttr) -> Result<GlobalId, Report>;
    /// The table `table` names (the table operations, `wasm.call_indirect`,
    /// `wasm.return_call_indirect`).
    fn table(&mut self, user: &Operation, table: &SymbolRefAttr) -> Result<TableId, Report>;
    /// The data entry `data` names (`wasm.data_addr`, `wasm.memory_init`, `wasm.data_drop`).
    fn data(&mut self, user: &Operation, data: &SymbolRefAttr) -> Result<DataId, Report>;
}

/// The operations [encode] writes as an instruction: every operation of the `wasm` dialect that
/// is not control ([CONTROL]) or module level ([MODULE_LEVEL]), and the `builtin` operations of
/// Wasm form.
pub const INSTRUCTIONS: [&str; 90] = [
    // Constants and selection.
    "wasm.const",
    "wasm.constf",
    "wasm.select",
    // Numeric: integers and floats.
    "wasm.add",
    "wasm.sub",
    "wasm.mul",
    // Numeric: integers.
    "wasm.div_s",
    "wasm.div_u",
    "wasm.rem_s",
    "wasm.rem_u",
    "wasm.and",
    "wasm.or",
    "wasm.xor",
    "wasm.shl",
    "wasm.shr_s",
    "wasm.shr_u",
    "wasm.rotl",
    "wasm.rotr",
    "wasm.clz",
    "wasm.ctz",
    "wasm.popcnt",
    // Numeric: floats.
    "wasm.div",
    "wasm.min",
    "wasm.max",
    "wasm.copysign",
    "wasm.abs",
    "wasm.neg",
    "wasm.sqrt",
    "wasm.ceil",
    "wasm.floor",
    "wasm.trunc",
    "wasm.nearest",
    // Comparisons.
    "wasm.eqz",
    "wasm.eq",
    "wasm.ne",
    "wasm.lt_s",
    "wasm.lt_u",
    "wasm.gt_s",
    "wasm.gt_u",
    "wasm.le_s",
    "wasm.le_u",
    "wasm.ge_s",
    "wasm.ge_u",
    "wasm.lt",
    "wasm.gt",
    "wasm.le",
    "wasm.ge",
    // Conversions.
    "wasm.wrap",
    "wasm.extend_s",
    "wasm.extend_u",
    "wasm.extend8_s",
    "wasm.extend16_s",
    "wasm.extend32_s",
    "wasm.trunc_s",
    "wasm.trunc_u",
    "wasm.trunc_sat_s",
    "wasm.trunc_sat_u",
    "wasm.convert_s",
    "wasm.convert_u",
    "wasm.promote",
    "wasm.demote",
    "wasm.reinterpret",
    // Memory.
    "wasm.load",
    "wasm.store",
    "wasm.memory_size",
    "wasm.memory_grow",
    "wasm.memory_copy",
    "wasm.memory_fill",
    "wasm.memory_init",
    "wasm.data_drop",
    // Globals, tables, references, addresses.
    "wasm.global_get",
    "wasm.global_set",
    "wasm.table_get",
    "wasm.table_set",
    "wasm.table_size",
    "wasm.table_grow",
    "wasm.table_fill",
    "wasm.ref_null",
    "wasm.ref_func",
    "wasm.ref_is_null",
    "wasm.data_addr",
    "wasm.func_index",
    // Calls, traps and the builtin operations of Wasm form.
    "wasm.call_indirect",
    "wasm.return_call_indirect",
    "wasm.unreachable",
    "builtin.call",
    "builtin.tail_call",
    "builtin.ret",
    "builtin.load_local",
    "builtin.store_local",
];

/// The structured control of the dialect: constructs, their ends and branches. A function body's
/// walk ([body](super::body)) writes them; [encode] does not.
pub const CONTROL: [&str; 6] =
    ["wasm.block", "wasm.loop", "wasm.if", "wasm.end", "wasm.br", "wasm.br_table"];

/// The operations of the dialect that are entities of a module, not instructions.
pub const MODULE_LEVEL: [&str; 5] =
    ["wasm.memory", "wasm.table", "wasm.global", "wasm.elem", "wasm.tag"];

/// The operations that are written again at each use of their result, and nothing at their own
/// position: constants and the placeholders the layout resolves.
pub const REMATERIALIZED: [&str; 5] = [
    "wasm.const",
    "wasm.constf",
    "wasm.ref_null",
    "wasm.data_addr",
    "wasm.func_index",
];

/// The name of `op`, as written: `wasm.add`.
pub fn name_of(op: &Operation) -> String {
    format!("{}", op.name())
}

/// Whether `op` is written again at each use of its result ([REMATERIALIZED]).
pub fn is_rematerialized(op: &Operation) -> bool {
    REMATERIALIZED.contains(&name_of(op).as_str())
}

/// Whether `op` is structured control ([CONTROL]).
pub fn is_control(op: &Operation) -> bool {
    CONTROL.contains(&name_of(op).as_str())
}

/// The values `op` takes from the Wasm stack, in the order they are pushed: the first is pushed
/// first. For every operation but five this is the order of its operands; `wasm.select %c, %a,
/// %b` pushes `a, b, c`, `wasm.call_indirect @t, %i(%args..)` (and its tail form) pushes the
/// arguments, then `i`, `wasm.if %c, %inputs..` pushes the inputs, then `c`, and `wasm.br_table
/// %i, %values..` pushes the values, then `i`.
pub fn stack_operands(op: &Operation) -> Vec<ValueRef> {
    let operands = op
        .operands()
        .iter()
        .map(|operand| operand.borrow().as_value_ref())
        .collect::<Vec<_>>();
    match name_of(op).as_str() {
        "wasm.select" if operands.len() == 3 => Vec::from([operands[1], operands[2], operands[0]]),
        "wasm.call_indirect" | "wasm.return_call_indirect" | "wasm.if" | "wasm.br_table"
            if !operands.is_empty() =>
        {
            let mut pushed = operands[1..].to_vec();
            pushed.push(operands[0]);
            pushed
        }
        _ => operands,
    }
}

/// The Wasm value type of `ty`, or `None` when it is not one.
pub fn val_type(ty: Type) -> Option<ValType> {
    if ty.is::<wasm::FuncRefType>() {
        return Some(ValType::FuncRef);
    }
    if ty.is::<wasm::ExternRefType>() {
        return Some(ValType::ExternRef);
    }
    match ty.integer_width() {
        Some(32) => Some(ValType::I32),
        Some(64) => Some(ValType::I64),
        Some(_) => None,
        None => match ty.as_float().map(|float| float.bits()) {
            Some(32) => Some(ValType::F32),
            Some(64) => Some(ValType::F64),
            _ => None,
        },
    }
}

/// The error of an operation `op` that cannot be emitted, with `label` on its span saying why.
pub fn unsupported(op: &Operation, label: impl Into<String>) -> Report {
    unsupported_at(&name_of(op), op.span(), label.into())
}

/// The error of the operation named `name` at `span` that cannot be emitted, with `label`.
pub fn unsupported_at(name: &str, span: SourceSpan, label: String) -> Report {
    report!(
        message: (format!("the operation `{name}` cannot be emitted as WebAssembly")),
        labels: [primary(span, label)],
    )
}

/// The internal error of a body whose plan the encoder cannot follow, at `op`, with `what` saying
/// what went wrong: a defect of the backend, not of the IR.
pub fn defect(op: &Operation, what: &str) -> Report {
    report!(
        message: (format!(
            "the WebAssembly body of `{}` could not be encoded, which is a defect of the backend",
            name_of(op)
        )),
        labels: [primary(op.span(), String::from(what))],
    )
}

/// The Wasm value type of `ty`, which `what` of `op` has, or the error that it is not one.
pub fn val_type_of(op: &Operation, what: &str, ty: Type) -> Result<ValType, Report> {
    val_type(ty).ok_or_else(|| {
        unsupported(
            op,
            format!(
                "{what} has type `{}`, which is not a WebAssembly value type",
                wasm::val_type_name(ty)
            ),
        )
    })
}

/// The type of operand `index` of `op` as a Wasm value type.
fn operand_type(op: &Operation, index: usize) -> Result<ValType, Report> {
    let operand = op
        .operands()
        .iter()
        .nth(index)
        .map(|operand| operand.borrow().ty())
        .ok_or_else(|| unsupported(op, format!("it has no operand {index}")))?;
    val_type_of(op, &format!("operand {index}"), operand)
}

/// The type of result `index` of `op` as a Wasm value type.
fn result_type(op: &Operation, index: usize) -> Result<ValType, Report> {
    let result = op
        .results()
        .iter()
        .nth(index)
        .map(|result| result.borrow().ty())
        .ok_or_else(|| unsupported(op, format!("it has no result {index}")))?;
    val_type_of(op, &format!("result {index}"), result)
}

/// The function type of `signature`, the signature of `op` (its calling convention is not part
/// of a Wasm type), with every parameter and result a Wasm value type.
pub fn func_type_of(op: &Operation, signature: &Signature) -> Result<FuncType, Report> {
    let params = signature
        .params()
        .iter()
        .enumerate()
        .map(|(index, param)| val_type_of(op, &format!("parameter {index}"), param.ty))
        .collect::<Result<Vec<_>, _>>()?;
    let results = signature
        .results()
        .iter()
        .enumerate()
        .map(|(index, result)| val_type_of(op, &format!("result {index}"), result.ty))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FuncType::new(params, results))
}

/// Records in `desc` the features the function type `ty` needs: multivalue for more than one
/// result, reference types for a reference among its parameters or results.
pub fn record_type_features(desc: &mut ModuleDesc, ty: &FuncType) {
    if ty.results.len() > 1 {
        desc.features.insert(Feature::Multivalue);
    }
    if ty.params.iter().chain(&ty.results).any(|ty| is_reference(*ty)) {
        desc.features.insert(Feature::ReferenceTypes);
    }
}

/// Whether `ty` is a reference type.
pub fn is_reference(ty: ValType) -> bool {
    matches!(ty, ValType::FuncRef | ValType::ExternRef)
}

/// What [encode] needs besides the operation: the description it records features, types and
/// the data-count flag in, the entities symbols stand for, and the number of the function's
/// parameters (the index of its first declared local).
pub struct Encoder<'a> {
    /// The description the function is encoded for.
    pub desc: &'a mut ModuleDesc,
    /// The entities of the unit.
    pub entities: &'a mut dyn Entities,
    /// The number of parameters of the function: `builtin.load_local N` reads local
    /// `params + N`.
    pub params: u32,
}

/// The numeric families of an operation's type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Num {
    I32,
    I64,
    F32,
    F64,
}

fn num(ty: ValType) -> Option<Num> {
    match ty {
        ValType::I32 => Some(Num::I32),
        ValType::I64 => Some(Num::I64),
        ValType::F32 => Some(Num::F32),
        ValType::F64 => Some(Num::F64),
        ValType::FuncRef | ValType::ExternRef => None,
    }
}

/// The instruction of the numeric, comparison or conversion operation `name` (without its
/// `wasm.` prefix) whose first operand has type `a` and whose result has type `r`, or `None` when
/// there is none for those types.
fn numeric(name: &str, a: Num, r: Num) -> Option<Instruction<'static>> {
    use Instruction as I;
    use Num::{F32, F64, I32, I64};
    let by = |i32: I<'static>, i64: I<'static>, f32: I<'static>, f64: I<'static>| match a {
        I32 => i32,
        I64 => i64,
        F32 => f32,
        F64 => f64,
    };
    let int = |i32: I<'static>, i64: I<'static>| match a {
        I32 => Some(i32),
        I64 => Some(i64),
        F32 | F64 => None,
    };
    let float = |f32: I<'static>, f64: I<'static>| match a {
        F32 => Some(f32),
        F64 => Some(f64),
        I32 | I64 => None,
    };
    match name {
        "add" => Some(by(I::I32Add, I::I64Add, I::F32Add, I::F64Add)),
        "sub" => Some(by(I::I32Sub, I::I64Sub, I::F32Sub, I::F64Sub)),
        "mul" => Some(by(I::I32Mul, I::I64Mul, I::F32Mul, I::F64Mul)),
        "div_s" => int(I::I32DivS, I::I64DivS),
        "div_u" => int(I::I32DivU, I::I64DivU),
        "rem_s" => int(I::I32RemS, I::I64RemS),
        "rem_u" => int(I::I32RemU, I::I64RemU),
        "and" => int(I::I32And, I::I64And),
        "or" => int(I::I32Or, I::I64Or),
        "xor" => int(I::I32Xor, I::I64Xor),
        "shl" => int(I::I32Shl, I::I64Shl),
        "shr_s" => int(I::I32ShrS, I::I64ShrS),
        "shr_u" => int(I::I32ShrU, I::I64ShrU),
        "rotl" => int(I::I32Rotl, I::I64Rotl),
        "rotr" => int(I::I32Rotr, I::I64Rotr),
        "clz" => int(I::I32Clz, I::I64Clz),
        "ctz" => int(I::I32Ctz, I::I64Ctz),
        "popcnt" => int(I::I32Popcnt, I::I64Popcnt),
        "div" => float(I::F32Div, I::F64Div),
        "min" => float(I::F32Min, I::F64Min),
        "max" => float(I::F32Max, I::F64Max),
        "copysign" => float(I::F32Copysign, I::F64Copysign),
        "abs" => float(I::F32Abs, I::F64Abs),
        "neg" => float(I::F32Neg, I::F64Neg),
        "sqrt" => float(I::F32Sqrt, I::F64Sqrt),
        "ceil" => float(I::F32Ceil, I::F64Ceil),
        "floor" => float(I::F32Floor, I::F64Floor),
        "trunc" => float(I::F32Trunc, I::F64Trunc),
        "nearest" => float(I::F32Nearest, I::F64Nearest),
        "eqz" => int(I::I32Eqz, I::I64Eqz),
        "eq" => Some(by(I::I32Eq, I::I64Eq, I::F32Eq, I::F64Eq)),
        "ne" => Some(by(I::I32Ne, I::I64Ne, I::F32Ne, I::F64Ne)),
        "lt_s" => int(I::I32LtS, I::I64LtS),
        "lt_u" => int(I::I32LtU, I::I64LtU),
        "gt_s" => int(I::I32GtS, I::I64GtS),
        "gt_u" => int(I::I32GtU, I::I64GtU),
        "le_s" => int(I::I32LeS, I::I64LeS),
        "le_u" => int(I::I32LeU, I::I64LeU),
        "ge_s" => int(I::I32GeS, I::I64GeS),
        "ge_u" => int(I::I32GeU, I::I64GeU),
        "lt" => float(I::F32Lt, I::F64Lt),
        "gt" => float(I::F32Gt, I::F64Gt),
        "le" => float(I::F32Le, I::F64Le),
        "ge" => float(I::F32Ge, I::F64Ge),
        "extend8_s" => int(I::I32Extend8S, I::I64Extend8S),
        "extend16_s" => int(I::I32Extend16S, I::I64Extend16S),
        _ => match (name, a, r) {
            ("wrap", I64, I32) => Some(I::I32WrapI64),
            ("extend_s", I32, I64) => Some(I::I64ExtendI32S),
            ("extend_u", I32, I64) => Some(I::I64ExtendI32U),
            ("extend32_s", I64, I64) => Some(I::I64Extend32S),
            ("trunc_s", F32, I32) => Some(I::I32TruncF32S),
            ("trunc_s", F64, I32) => Some(I::I32TruncF64S),
            ("trunc_s", F32, I64) => Some(I::I64TruncF32S),
            ("trunc_s", F64, I64) => Some(I::I64TruncF64S),
            ("trunc_u", F32, I32) => Some(I::I32TruncF32U),
            ("trunc_u", F64, I32) => Some(I::I32TruncF64U),
            ("trunc_u", F32, I64) => Some(I::I64TruncF32U),
            ("trunc_u", F64, I64) => Some(I::I64TruncF64U),
            ("trunc_sat_s", F32, I32) => Some(I::I32TruncSatF32S),
            ("trunc_sat_s", F64, I32) => Some(I::I32TruncSatF64S),
            ("trunc_sat_s", F32, I64) => Some(I::I64TruncSatF32S),
            ("trunc_sat_s", F64, I64) => Some(I::I64TruncSatF64S),
            ("trunc_sat_u", F32, I32) => Some(I::I32TruncSatF32U),
            ("trunc_sat_u", F64, I32) => Some(I::I32TruncSatF64U),
            ("trunc_sat_u", F32, I64) => Some(I::I64TruncSatF32U),
            ("trunc_sat_u", F64, I64) => Some(I::I64TruncSatF64U),
            ("convert_s", I32, F32) => Some(I::F32ConvertI32S),
            ("convert_s", I64, F32) => Some(I::F32ConvertI64S),
            ("convert_s", I32, F64) => Some(I::F64ConvertI32S),
            ("convert_s", I64, F64) => Some(I::F64ConvertI64S),
            ("convert_u", I32, F32) => Some(I::F32ConvertI32U),
            ("convert_u", I64, F32) => Some(I::F32ConvertI64U),
            ("convert_u", I32, F64) => Some(I::F64ConvertI32U),
            ("convert_u", I64, F64) => Some(I::F64ConvertI64U),
            ("promote", F32, F64) => Some(I::F64PromoteF32),
            ("demote", F64, F32) => Some(I::F32DemoteF64),
            ("reinterpret", F32, I32) => Some(I::I32ReinterpretF32),
            ("reinterpret", F64, I64) => Some(I::I64ReinterpretF64),
            ("reinterpret", I32, F32) => Some(I::F32ReinterpretI32),
            ("reinterpret", I64, F64) => Some(I::F64ReinterpretI64),
            _ => None,
        },
    }
}

/// The feature the numeric operation `name` (without its prefix) needs beyond the MVP.
fn numeric_feature(name: &str) -> Option<Feature> {
    match name {
        "extend8_s" | "extend16_s" | "extend32_s" => Some(Feature::SignExt),
        "trunc_sat_s" | "trunc_sat_u" => Some(Feature::NontrappingFptoint),
        _ => None,
    }
}

/// A Wasm type name for a message.
fn type_name(ty: ValType) -> &'static str {
    match ty {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "f32",
        ValType::F64 => "f64",
        ValType::FuncRef => "funcref",
        ValType::ExternRef => "externref",
    }
}

/// The memory argument of a load or store `op`: its offset, and its alignment as the base-2
/// logarithm of its bytes, on memory 0.
fn memarg(op: &Operation, memarg: &wasm::MemArg) -> Result<MemArg, Report> {
    let offset = memarg.offset();
    if offset > u64::from(u32::MAX) {
        return Err(unsupported(
            op,
            format!("its offset {offset} does not fit the 32-bit address of a memory"),
        ));
    }
    let align = memarg.align();
    if !align.is_power_of_two() {
        return Err(unsupported(op, format!("its alignment {align} is not a power of two")));
    }
    Ok(MemArg {
        offset,
        align: align.trailing_zeros(),
        target: MemArgTarget::Memory(0),
        ordering: None,
    })
}

/// The instruction of the load `op`, by the type of its result, its storage and its extension.
fn load(op: &Operation, load: &wasm::Load) -> Result<Instruction<'static>, Report> {
    use Instruction as I;
    use wasm::{Extension as E, Storage as S};
    let ty = result_type(op, 0)?;
    let arg = memarg(op, &load.get_memarg())?;
    let storage = *load.get_storage();
    let extension = *load.get_extension();
    let instruction = match (ty, storage, extension) {
        (ValType::I32, S::Full, E::Unextended) => I::I32Load(arg),
        (ValType::I64, S::Full, E::Unextended) => I::I64Load(arg),
        (ValType::F32, S::Full, E::Unextended) => I::F32Load(arg),
        (ValType::F64, S::Full, E::Unextended) => I::F64Load(arg),
        (ValType::I32, S::I8, E::Signed) => I::I32Load8S(arg),
        (ValType::I32, S::I8, E::Unsigned) => I::I32Load8U(arg),
        (ValType::I32, S::I16, E::Signed) => I::I32Load16S(arg),
        (ValType::I32, S::I16, E::Unsigned) => I::I32Load16U(arg),
        (ValType::I64, S::I8, E::Signed) => I::I64Load8S(arg),
        (ValType::I64, S::I8, E::Unsigned) => I::I64Load8U(arg),
        (ValType::I64, S::I16, E::Signed) => I::I64Load16S(arg),
        (ValType::I64, S::I16, E::Unsigned) => I::I64Load16U(arg),
        (ValType::I64, S::I32, E::Signed) => I::I64Load32S(arg),
        (ValType::I64, S::I32, E::Unsigned) => I::I64Load32U(arg),
        _ => {
            return Err(unsupported(
                op,
                format!(
                    "a load of `{}` from `{}` storage, {}, is not a WebAssembly instruction",
                    type_name(ty),
                    storage.as_str(),
                    extension.as_str()
                ),
            ));
        }
    };
    Ok(instruction)
}

/// The instruction of the store `op`, by the type of its value and its storage.
fn store(op: &Operation, store: &wasm::Store) -> Result<Instruction<'static>, Report> {
    use Instruction as I;
    use wasm::Storage as S;
    let ty = operand_type(op, 1)?;
    let arg = memarg(op, &store.get_memarg())?;
    let storage = *store.get_storage();
    let instruction = match (ty, storage) {
        (ValType::I32, S::Full) => I::I32Store(arg),
        (ValType::I64, S::Full) => I::I64Store(arg),
        (ValType::F32, S::Full) => I::F32Store(arg),
        (ValType::F64, S::Full) => I::F64Store(arg),
        (ValType::I32, S::I8) => I::I32Store8(arg),
        (ValType::I32, S::I16) => I::I32Store16(arg),
        (ValType::I64, S::I8) => I::I64Store8(arg),
        (ValType::I64, S::I16) => I::I64Store16(arg),
        (ValType::I64, S::I32) => I::I64Store32(arg),
        _ => {
            return Err(unsupported(
                op,
                format!(
                    "a store of `{}` to `{}` storage is not a WebAssembly instruction",
                    type_name(ty),
                    storage.as_str()
                ),
            ));
        }
    };
    Ok(instruction)
}

/// The heap type of the reference type `ty`.
fn heap_type(op: &Operation, ty: ValType) -> Result<HeapType, Report> {
    match ty {
        ValType::FuncRef => Ok(HeapType::FUNC),
        ValType::ExternRef => Ok(HeapType::EXTERN),
        other => Err(unsupported(
            op,
            format!("its result has type `{}`, which is not a reference type", type_name(other)),
        )),
    }
}

/// Downcasts `op` to `T`, which its name says it is.
fn as_op<T: runic_ir::Op>(op: &Operation) -> Result<&T, Report> {
    op.downcast_ref::<T>()
        .ok_or_else(|| unsupported(op, "it is not the operation its name says it is"))
}

/// Writes the instruction of `op` (not its operands) through `writer`, and records in the
/// description the features it uses (and, for `wasm.memory_init` and `wasm.data_drop`, that the
/// module needs a data-count section).
///
/// `builtin.ret` is written `return`; a walk that ends the function with it writes nothing
/// instead. An operation of [CONTROL] is written by the walk of a body
/// ([body](super::body)), not here: given one, this is a [defect]. Any other operation that is not
/// of [INSTRUCTIONS] is an error naming it.
pub fn encode(
    op: &Operation,
    writer: &mut CodeWriter<'_>,
    cx: &mut Encoder<'_>,
) -> Result<(), Report> {
    use Instruction as I;
    let name = name_of(op);
    if CONTROL.contains(&name.as_str()) {
        return Err(defect(
            op,
            "structured control is written by the walk of a body, not as one instruction",
        ));
    }
    if let Some(short) = name.strip_prefix("wasm.")
        && let Some(instruction) = numeric_of(op, short)?
    {
        if let Some(feature) = numeric_feature(short) {
            cx.desc.features.insert(feature);
        }
        writer.instruction(&instruction);
        return Ok(());
    }
    match name.as_str() {
        "wasm.const" => {
            let value = *as_op::<wasm::Const>(op)?.get_value();
            match result_type(op, 0)? {
                ValType::I32 => writer.instruction(&I::I32Const(value.unsigned() as u32 as i32)),
                ValType::I64 => writer.instruction(&I::I64Const(value.unsigned() as u64 as i64)),
                other => {
                    return Err(unsupported(
                        op,
                        format!("an integer constant of type `{}`", type_name(other)),
                    ));
                }
            };
        }
        "wasm.constf" => {
            let value = *as_op::<wasm::ConstF>(op)?.get_value();
            match result_type(op, 0)? {
                ValType::F32 => writer.instruction(&I::F32Const(Ieee32::new(value.bits() as u32))),
                ValType::F64 => writer.instruction(&I::F64Const(Ieee64::new(value.bits()))),
                other => {
                    return Err(unsupported(
                        op,
                        format!("a float constant of type `{}`", type_name(other)),
                    ));
                }
            };
        }
        "wasm.select" => {
            let ty = result_type(op, 0)?;
            if is_reference(ty) {
                cx.desc.features.insert(Feature::ReferenceTypes);
                writer.instruction(&I::TypedSelect(ty.to_encoder()));
            } else {
                writer.instruction(&I::Select);
            }
        }
        "wasm.load" => {
            let instruction = load(op, as_op::<wasm::Load>(op)?)?;
            writer.instruction(&instruction);
        }
        "wasm.store" => {
            let instruction = store(op, as_op::<wasm::Store>(op)?)?;
            writer.instruction(&instruction);
        }
        "wasm.memory_size" => {
            writer.instruction(&I::MemorySize(0));
        }
        "wasm.memory_grow" => {
            writer.instruction(&I::MemoryGrow(0));
        }
        "wasm.memory_copy" => {
            cx.desc.features.insert(Feature::BulkMemory);
            writer.instruction(&I::MemoryCopy {
                src_mem: 0,
                dst_mem: 0,
            });
        }
        "wasm.memory_fill" => {
            cx.desc.features.insert(Feature::BulkMemory);
            writer.instruction(&I::MemoryFill(0));
        }
        "wasm.memory_init" => {
            let data = cx.entities.data(op, &as_op::<wasm::MemoryInit>(op)?.data())?;
            cx.desc.features.insert(Feature::BulkMemory);
            cx.desc.uses_data_count = true;
            // `memory.init` is `0xfc 8`, the data index, then the memory index.
            writer.fixup_index(&[0xfc, 0x08], FixupKind::DataIndex(data)).raw(&[0x00]);
        }
        "wasm.data_drop" => {
            let data = cx.entities.data(op, &as_op::<wasm::DataDrop>(op)?.data())?;
            cx.desc.features.insert(Feature::BulkMemory);
            cx.desc.uses_data_count = true;
            writer.fixup_index(&[0xfc, 0x09], FixupKind::DataIndex(data));
        }
        "wasm.global_get" => {
            let global = cx.entities.global(op, &as_op::<wasm::GlobalGet>(op)?.global())?;
            writer.fixup_index(&[0x23], FixupKind::GlobalIndex(global));
        }
        "wasm.global_set" => {
            let global = cx.entities.global(op, &as_op::<wasm::GlobalSet>(op)?.global())?;
            writer.fixup_index(&[0x24], FixupKind::GlobalIndex(global));
        }
        "wasm.table_get" => {
            let table = cx.entities.table(op, &as_op::<wasm::TableGet>(op)?.table())?;
            cx.desc.features.insert(Feature::ReferenceTypes);
            writer.fixup_index(&[0x25], FixupKind::TableNumber(table));
        }
        "wasm.table_set" => {
            let table = cx.entities.table(op, &as_op::<wasm::TableSet>(op)?.table())?;
            cx.desc.features.insert(Feature::ReferenceTypes);
            writer.fixup_index(&[0x26], FixupKind::TableNumber(table));
        }
        "wasm.table_size" => {
            let table = cx.entities.table(op, &as_op::<wasm::TableSize>(op)?.table())?;
            cx.desc.features.insert(Feature::ReferenceTypes);
            writer.fixup_index(&[0xfc, 0x10], FixupKind::TableNumber(table));
        }
        "wasm.table_grow" => {
            let table = cx.entities.table(op, &as_op::<wasm::TableGrow>(op)?.table())?;
            cx.desc.features.insert(Feature::ReferenceTypes);
            writer.fixup_index(&[0xfc, 0x0f], FixupKind::TableNumber(table));
        }
        "wasm.table_fill" => {
            let table = cx.entities.table(op, &as_op::<wasm::TableFill>(op)?.table())?;
            cx.desc.features.insert(Feature::ReferenceTypes);
            writer.fixup_index(&[0xfc, 0x11], FixupKind::TableNumber(table));
        }
        "wasm.ref_null" => {
            let heap = heap_type(op, result_type(op, 0)?)?;
            cx.desc.features.insert(Feature::ReferenceTypes);
            writer.instruction(&I::RefNull(heap));
        }
        "wasm.ref_func" => {
            let callee = cx.entities.function(op, &as_op::<wasm::RefFunc>(op)?.callee())?;
            cx.desc.features.insert(Feature::ReferenceTypes);
            writer.fixup_index(&[0xd2], FixupKind::FunctionIndex(callee));
        }
        "wasm.ref_is_null" => {
            cx.desc.features.insert(Feature::ReferenceTypes);
            writer.instruction(&I::RefIsNull);
        }
        "wasm.data_addr" => {
            let data_addr = as_op::<wasm::DataAddr>(op)?;
            let offset = data_addr.offset_value();
            let addend = i32::try_from(offset).map_err(|_| {
                unsupported(op, format!("its offset {offset} does not fit an `i32`"))
            })?;
            let data = cx.entities.data(op, &data_addr.data())?;
            writer.fixup_i32_const(FixupKind::MemoryAddr(data), addend);
        }
        "wasm.func_index" => {
            let callee = cx.entities.function(op, &as_op::<wasm::FuncIndex>(op)?.callee())?;
            writer.fixup_i32_const(FixupKind::TableSlot(callee), 0);
        }
        "wasm.call_indirect" => {
            let call = as_op::<wasm::CallIndirect>(op)?;
            let ty = func_type_of(op, &call.get_signature())?;
            let table = cx.entities.table(op, &call.table())?;
            record_type_features(cx.desc, &ty);
            let ty = cx.desc.intern_type(ty);
            writer
                .fixup_index(&[0x11], FixupKind::TypeIndex(ty))
                .fixup_index(&[], FixupKind::TableNumber(table));
        }
        "wasm.return_call_indirect" => {
            let call = as_op::<wasm::ReturnCallIndirect>(op)?;
            let ty = func_type_of(op, &call.get_signature())?;
            let table = cx.entities.table(op, &call.table())?;
            record_type_features(cx.desc, &ty);
            cx.desc.features.insert(Feature::TailCall);
            let ty = cx.desc.intern_type(ty);
            writer
                .fixup_index(&[0x13], FixupKind::TypeIndex(ty))
                .fixup_index(&[], FixupKind::TableNumber(table));
        }
        "wasm.unreachable" => {
            writer.instruction(&I::Unreachable);
        }
        "builtin.call" => {
            let call = as_op::<Call>(op)?;
            let ty = func_type_of(op, &call.get_signature())?;
            let callee = cx.entities.function(op, &call.callee())?;
            record_type_features(cx.desc, &ty);
            writer.fixup_index(&[0x10], FixupKind::FunctionIndex(callee));
        }
        "builtin.tail_call" => {
            let call = as_op::<TailCall>(op)?;
            let ty = func_type_of(op, &call.get_signature())?;
            let callee = cx.entities.function(op, &call.callee())?;
            record_type_features(cx.desc, &ty);
            cx.desc.features.insert(Feature::TailCall);
            writer.fixup_index(&[0x12], FixupKind::FunctionIndex(callee));
        }
        "builtin.ret" => {
            writer.instruction(&I::Return);
        }
        "builtin.load_local" => {
            let index = as_op::<LoadLocal>(op)?.get_local().index();
            writer.instruction(&I::LocalGet(cx.params + index));
        }
        "builtin.store_local" => {
            let index = as_op::<StoreLocal>(op)?.get_local().index();
            writer.instruction(&I::LocalSet(cx.params + index));
        }
        _ if MODULE_LEVEL.contains(&name.as_str()) => {
            return Err(unsupported(
                op,
                format!("`{name}` is an entity of a module, not an instruction of a function body"),
            ));
        }
        _ => {
            return Err(unsupported(op, format!("`{name}` is not an operation of Wasm form")));
        }
    }
    Ok(())
}

/// The names of the numeric, comparison and conversion operations, without their prefix.
const NUMERIC: [&str; 59] = [
    "add",
    "sub",
    "mul",
    "div_s",
    "div_u",
    "rem_s",
    "rem_u",
    "and",
    "or",
    "xor",
    "shl",
    "shr_s",
    "shr_u",
    "rotl",
    "rotr",
    "clz",
    "ctz",
    "popcnt",
    "div",
    "min",
    "max",
    "copysign",
    "abs",
    "neg",
    "sqrt",
    "ceil",
    "floor",
    "trunc",
    "nearest",
    "eqz",
    "eq",
    "ne",
    "lt_s",
    "lt_u",
    "gt_s",
    "gt_u",
    "le_s",
    "le_u",
    "ge_s",
    "ge_u",
    "lt",
    "gt",
    "le",
    "ge",
    "wrap",
    "extend_s",
    "extend_u",
    "extend8_s",
    "extend16_s",
    "extend32_s",
    "trunc_s",
    "trunc_u",
    "trunc_sat_s",
    "trunc_sat_u",
    "convert_s",
    "convert_u",
    "promote",
    "demote",
    "reinterpret",
];

/// The instruction of `op` if it is a numeric, comparison or conversion operation (`short` is
/// its name without the prefix): `None` when it is not one, and an error when its types have no
/// instruction.
fn numeric_of(op: &Operation, short: &str) -> Result<Option<Instruction<'static>>, Report> {
    if !NUMERIC.contains(&short) {
        return Ok(None);
    }
    let a = operand_type(op, 0)?;
    let r = result_type(op, 0)?;
    let types = num(a).zip(num(r));
    match types.and_then(|(a, r)| numeric(short, a, r)) {
        Some(instruction) => Ok(Some(instruction)),
        None => Err(unsupported(
            op,
            format!(
                "`wasm.{short}` of `{}` giving `{}` is not a WebAssembly instruction",
                type_name(a),
                type_name(r)
            ),
        )),
    }
}
