//! A function body: one `builtin.function` in Wasm form becomes a [FunctionBody].
//!
//! [encode_function] plans every block of the body that is written ([plan](super::plan)), gives
//! locals to the values that need them ([locals](super::locals)), and writes the instructions of
//! each plan's roots in order through a [CodeWriter], each operation's instruction by [encode].
//! A region is the unit of the walk: [emit_block] writes the plan of one block. Emitting a root
//! means: for each of its stack operands in order, the tree of the claimed operation that leaves
//! it, a `local.get` of its local, or the instruction of the rematerialized operation that
//! defines it; then the root's instruction; then, for its results from the last to the first, a
//! `local.set` of a used one and a `drop` of an unused one. A claimed operation is followed by
//! nothing, or by `local.tee` when its value has other uses. Between two roots the stack is empty;
//! debug builds check it by counting.
//!
//! **Control** (decision 6 of the plan). A construct is one operation of its block, written where
//! its instruction goes; one with a single result can be a node of a tree like any other.
//!
//! - `wasm.block`, `wasm.loop` and `wasm.if` take their inputs on the stack (an `if` its condition
//!   last). The block type is empty with no inputs and no results, the one value type with no
//!   inputs and one result, and otherwise an interned function type, named through a
//!   [FixupKind::BlockType] fix-up (which needs the multivalue feature). At the top of each region
//!   that is written, the arguments of its block, which are on the stack, are set into their
//!   locals from the last to the first (an unused one is dropped); then the block is written as
//!   above. An `if` whose second region passes its operands through
//!   ([If::else_is_pass_through](wasm::If::else_is_pass_through)) is written without `else`.
//! - An `if` with no inputs and no results, whose second region passes through and whose first
//!   region is one `wasm.br` with no values to a level of at least 1, is written as `br_if` to
//!   that level less one (the `if` itself was a label).
//! - `wasm.end` leaves its values on the stack as the construct's results and writes nothing (the
//!   construct writes its `else` or `end`); `wasm.br` writes `br` after its values;
//!   `wasm.br_table` writes `br_table` after its values and its index, with the level of the one
//!   `wasm.br` of each entry's region and of the default's (the table is no label, so the levels
//!   are used as written). The results of a construct with several go to locals from the last to
//!   the first, after its `end`.
//! - `builtin.ret` writes `return` at any depth, except as the last operation of the body's own
//!   block, where the function's `end` returns the values left on the stack.
//!
//! After an operation that leaves (`br`, `br_table`, `return`, a tail call, `unreachable`) the
//! stack is polymorphic, and what follows (the `else` or `end` of the construct, or operations
//! after an `if` whose arms all leave) is written as it is: Wasm's validation accepts it.
//!
//! The body's block ends in `builtin.ret`, `builtin.tail_call` (`return_call`),
//! `wasm.return_call_indirect` or `wasm.unreachable`.

use alloc::{borrow::Cow, format, vec::Vec};

use runic_dialect_wasm as wasm;
use runic_ir::{
    BlockRef, FxHashMap, Operation, OperationRef, Report, Symbol as _, Value, ValueRef,
    dialects::builtin::Function,
};
use wasm_encoder::{BlockType, Instruction};

use super::{
    desc::{CodeWriter, Feature, FixupKind, FuncType, FunctionBody, ModuleDesc},
    instr::{
        Encoder, Entities, defect, encode, func_type_of, is_reference, name_of,
        record_type_features, unsupported, val_type_of,
    },
    locals::{LiveRange, Locals, Numbering, assign, live_range, numbering},
    plan::{BlockPlan, Role, Source, plan_block},
};

/// How a body is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyConfig {
    /// Whether a local of the emitter is shared by values of one type whose live ranges do not
    /// overlap. Without it, every value that needs a local has its own.
    pub reuse_locals: bool,
}

impl Default for BodyConfig {
    /// Locals are reused.
    fn default() -> Self {
        Self { reuse_locals: true }
    }
}

/// The operations a function's body may end in.
const TERMINATORS: [&str; 4] = [
    "builtin.ret",
    "builtin.tail_call",
    "wasm.return_call_indirect",
    "wasm.unreachable",
];

/// The operations after which control does not continue in their block: the stack is
/// polymorphic.
const LEAVES: [&str; 6] = [
    "wasm.br",
    "wasm.br_table",
    "builtin.ret",
    "builtin.tail_call",
    "wasm.return_call_indirect",
    "wasm.unreachable",
];

/// The plan of a function's body and the locals of its values, with what the encoder records
/// about the function's types.
#[derive(Clone, Debug)]
pub struct FunctionPlan {
    /// The plan of the body's block.
    pub plan: BlockPlan,
    /// The plan of every other block that is written: the blocks of the regions of the
    /// constructs, except those a construct does not write (the second region of an `if` that
    /// passes its operands through, the first of an `if` written as `br_if`, the regions of a
    /// `br_table`).
    pub nested: FxHashMap<BlockRef, BlockPlan>,
    /// The locals: the parameters, the declared locals and the emitter's.
    pub locals: Locals,
    /// The function's type.
    pub ty: FuncType,
    /// Whether a value of the body (a parameter, a declared local, a result, an argument of a
    /// block) is a reference.
    pub uses_references: bool,
}

impl FunctionPlan {
    /// The plan of `block`, the body's block or a nested block that is written.
    pub fn plan_of(&self, block: BlockRef) -> Option<&BlockPlan> {
        self.nested.get(&block).or_else(|| {
            let entry = self.plan.steps.first()?.op.borrow().parent()?;
            (entry == block).then_some(&self.plan)
        })
    }
}

/// The level of the `br_if` that `op` is written as, when it is one: a `wasm.if` with no inputs
/// and no results, whose second region passes through, and whose first region is one `wasm.br`
/// with no values to a level of at least 1 (the `br_if` names that level less one, as the `if` is
/// no label).
pub fn br_if_level(op: &Operation) -> Option<u32> {
    let if_op = op.downcast_ref::<wasm::If>()?;
    if !if_op.inputs().is_empty() || op.num_results() != 0 || !if_op.else_is_pass_through() {
        return None;
    }
    let entry = if_op.then_body().entry_block_ref()?;
    let entry = entry.borrow();
    let mut body = entry.body().iter();
    let (Some(only), None) = (body.next(), body.next()) else {
        return None;
    };
    let br = only.downcast_ref::<wasm::Br>()?;
    if only.num_operands() != 0 {
        return None;
    }
    let level = u32::try_from(*br.get_level()).ok()?;
    level.checked_sub(1)
}

/// The blocks of the regions of `op` that are written, in the order they are written: the body
/// of a `wasm.block` or `wasm.loop`; the first region of a `wasm.if` that is not a `br_if`, and
/// its second unless it passes through. Nothing for any other operation.
fn written_blocks(op: &Operation) -> Vec<BlockRef> {
    let mut blocks = Vec::new();
    if let Some(block) = op.downcast_ref::<wasm::BlockOp>() {
        blocks.extend(block.body().entry_block_ref());
    } else if let Some(loop_op) = op.downcast_ref::<wasm::Loop>() {
        blocks.extend(loop_op.body().entry_block_ref());
    } else if let Some(if_op) = op.downcast_ref::<wasm::If>()
        && br_if_level(op).is_none()
    {
        blocks.extend(if_op.then_body().entry_block_ref());
        if !if_op.else_is_pass_through() {
            blocks.extend(if_op.else_body().entry_block_ref());
        }
    }
    blocks
}

/// The plan and the locals of `function`, a `builtin.function` in Wasm form with a body; an error
/// names a declaration, or a parameter, local, result or argument of a block whose type is not a
/// Wasm value type.
pub fn plan_function(function: OperationRef, config: &BodyConfig) -> Result<FunctionPlan, Report> {
    let (entry, body_region, declared, ty) = {
        let op = function.borrow();
        let Some(func) = op.downcast_ref::<Function>() else {
            return Err(unsupported(&op, "it is not a `builtin.function`"));
        };
        let name = func.get_name();
        if func.is_declaration() {
            return Err(unsupported(
                &op,
                format!("`@{}` is a declaration, which has no body", name.as_str()),
            ));
        }
        let ty = func_type_of(&op, &func.get_signature())?;
        let declared = func
            .get_locals()
            .iter()
            .enumerate()
            .map(|(index, ty)| val_type_of(&op, &format!("local {index}"), *ty))
            .collect::<Result<Vec<_>, _>>()?;
        let body = func.body().as_region_ref();
        let Some(entry) = func.body().entry_block_ref() else {
            return Err(unsupported(&op, "its body has no block"));
        };
        (entry, body, declared, ty)
    };
    let mut uses_references = declared.iter().any(|ty| is_reference(*ty));
    let params = entry.borrow().argument_values().collect::<Vec<_>>();
    for (index, param) in params.iter().enumerate() {
        let ty = param.borrow().ty();
        let ty = val_type_of(&function.borrow(), &format!("parameter {index}"), ty)?;
        uses_references |= is_reference(ty);
    }
    let numbers = numbering(body_region);
    let mut ranges = Vec::new();
    let plan = plan_block(entry);
    uses_references |= collect_ranges(&plan, &numbers, &mut ranges)?;
    // Every block that is written, planned on its own; the arguments of each need locals.
    let mut nested = FxHashMap::default();
    let mut pending = written_by(&plan);
    while let Some((construct, block)) = pending.pop() {
        // The arguments are set at the region's entry: before its first operation, after the
        // construct's instruction (and, for a second region, after the first).
        let first = block.borrow().body().iter().next().map(|op| op.as_operation_ref());
        let def = first.or(Some(construct)).and_then(|op| numbers.of(op)).unwrap_or(0);
        let arguments = block.borrow().argument_values().collect::<Vec<_>>();
        for (index, argument) in arguments.into_iter().enumerate() {
            let ty = val_type_of(
                &construct.borrow(),
                &format!("argument {index} of a region"),
                argument.borrow().ty(),
            )?;
            uses_references |= is_reference(ty);
            if argument.borrow().is_used() {
                ranges.push(live_range(&numbers, argument, ty, def));
            }
        }
        let block_plan = plan_block(block);
        uses_references |= collect_ranges(&block_plan, &numbers, &mut ranges)?;
        pending.extend(written_by(&block_plan));
        nested.insert(block, block_plan);
    }
    let locals = assign(&params, &declared, &ranges, config.reuse_locals);
    Ok(FunctionPlan {
        plan,
        nested,
        locals,
        ty,
        uses_references,
    })
}

/// The blocks the operations of `plan` write, each with the operation whose region holds it.
fn written_by(plan: &BlockPlan) -> Vec<(OperationRef, BlockRef)> {
    plan.steps
        .iter()
        .flat_map(|step| {
            written_blocks(&step.op.borrow())
                .into_iter()
                .map(|block| (step.op, block))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The body of `function`, a `builtin.function` in Wasm form with a body, encoded for `desc`:
/// its locals beyond the parameters and its instructions, with a fix-up for every reference to
/// another entity, which `entities` gives. The features it uses, the types its indirect calls
/// and its constructs name and whether it needs a data-count section are recorded in `desc`; the
/// function's own type is not interned (the caller does that).
pub fn encode_function(
    function: OperationRef,
    desc: &mut ModuleDesc,
    entities: &mut dyn Entities,
    config: &BodyConfig,
) -> Result<FunctionBody, Report> {
    let plan = plan_function(function, config)?;
    record_type_features(desc, &plan.ty);
    if plan.uses_references {
        desc.features.insert(Feature::ReferenceTypes);
    }
    let mut body = FunctionBody::default();
    {
        let mut writer = body.writer();
        let mut cx = Encoder {
            desc,
            entities,
            params: plan.locals.params(),
        };
        emit_block(&plan, &plan.plan, &mut writer, &mut cx, true)?;
        writer.instruction(&Instruction::End);
    }
    body.locals = plan.locals.types();
    Ok(body)
}

/// Adds to `ranges` the live range of every value of `plan` that needs a local; whether a result
/// of an operation is a reference.
fn collect_ranges(
    plan: &BlockPlan,
    numbers: &Numbering,
    ranges: &mut Vec<LiveRange>,
) -> Result<bool, Report> {
    let mut references = false;
    for (index, step) in plan.steps.iter().enumerate() {
        let op = step.op.borrow();
        let def = numbers.of(step.op).unwrap_or(0);
        for (position, result) in op.results().iter().enumerate() {
            let value = result.borrow().as_value_ref();
            let ty = val_type_of(&op, &format!("result {position}"), value.borrow().ty())?;
            references |= is_reference(ty);
            if plan.needs_local(index, value) {
                ranges.push(live_range(numbers, value, ty, def));
            }
        }
    }
    Ok(references)
}

/// A tree being written: the position of its operation in the plan and the next of its stack
/// operands to write.
struct Frame {
    step: usize,
    next: usize,
}

/// Writes `plan`, the plan of one block of `function`: each root's tree, in order, and the
/// constructs among them with their regions. With `ends_function`, the block is the function's
/// body: it must end in one of the operations a body ends in, and a final `builtin.ret` leaves its
/// values on the stack for the function's `end` instead of writing `return`.
pub fn emit_block(
    function: &FunctionPlan,
    plan: &BlockPlan,
    writer: &mut CodeWriter<'_>,
    cx: &mut Encoder<'_>,
    ends_function: bool,
) -> Result<(), Report> {
    let locals = &function.locals;
    let last = plan.steps.len().checked_sub(1);
    if ends_function {
        let ends = last.is_some_and(|last| {
            TERMINATORS.contains(&name_of(&plan.steps[last].op.borrow()).as_str())
        });
        if !ends {
            let op = match last {
                Some(last) => plan.steps[last].op,
                None => return Ok(()),
            };
            let op = op.borrow();
            return Err(unsupported(
                &op,
                format!(
                    "a function body that ends in `{}` is not emitted: it ends in `builtin.ret`, \
                     `builtin.tail_call`, `wasm.return_call_indirect` or `wasm.unreachable`",
                    name_of(&op)
                ),
            ));
        }
    }
    // The number of values on the stack, counted as instructions are written.
    let mut height: usize = 0;
    for root in plan.roots() {
        debug_assert_eq!(height, 0, "the stack is not empty before the root at {root}");
        let root_name = name_of(&plan.steps[root].op.borrow());
        let final_ret = ends_function && Some(root) == last && root_name == "builtin.ret";
        let mut frames = Vec::from([Frame {
            step: root,
            next: 0,
        }]);
        while let Some(frame) = frames.last_mut() {
            let step = &plan.steps[frame.step];
            if let Some(operand) = step.operands.get(frame.next).copied() {
                frame.next += 1;
                match operand.source {
                    Source::Stack(child) => {
                        frames.push(Frame {
                            step: child,
                            next: 0,
                        });
                    }
                    Source::Local => {
                        let Some(local) = locals.of(operand.value) else {
                            return Err(defect(
                                &step.op.borrow(),
                                "an operand is fetched from a local that no value was given",
                            ));
                        };
                        writer.instruction(&Instruction::LocalGet(local));
                        height += 1;
                    }
                    Source::Constant(position) => {
                        let constant = match position {
                            Some(position) => plan.steps[position].op,
                            None => operand.value.borrow().get_defining_op().ok_or_else(|| {
                                defect(&step.op.borrow(), "a constant operand has no definition")
                            })?,
                        };
                        encode(&constant.borrow(), writer, cx)?;
                        height += 1;
                    }
                }
                continue;
            }
            // Every operand is on the stack: the instruction, then what follows it.
            let step = frames.pop().expect("a frame").step;
            let step = &plan.steps[step];
            let op = step.op.borrow();
            if final_ret && frames.is_empty() {
                // The function's `end` returns the values left on the stack.
                debug_assert_eq!(height, op.operands().len());
                break;
            }
            let name = name_of(&op);
            match name.as_str() {
                "wasm.block" | "wasm.loop" | "wasm.if" => {
                    emit_construct(function, &op, writer, cx)?;
                }
                // The values stay on the stack as the construct's results.
                "wasm.end" => {
                    debug_assert_eq!(height, step.operands.len());
                    break;
                }
                "wasm.br" => {
                    let level = level_of(&op)?;
                    writer.instruction(&Instruction::Br(level));
                }
                "wasm.br_table" => {
                    let (labels, default) = table_labels(&op)?;
                    writer.instruction(&Instruction::BrTable(Cow::Owned(labels), default));
                }
                _ => encode(&op, writer, cx)?,
            }
            height -= step.operands.len();
            if LEAVES.contains(&name.as_str()) {
                // The stack is polymorphic; nothing follows in this block.
                height = 0;
                continue;
            }
            let results =
                op.results().iter().map(|r| r.borrow().as_value_ref()).collect::<Vec<_>>();
            height += results.len();
            match step.role {
                Role::Claimed { tee } => {
                    if tee {
                        let Some(local) = locals.of(results[0]) else {
                            return Err(defect(&op, "a teed value was given no local"));
                        };
                        writer.instruction(&Instruction::LocalTee(local));
                    }
                }
                Role::Root => {
                    for result in results.iter().rev() {
                        if result.borrow().is_used() {
                            let Some(local) = locals.of(*result) else {
                                return Err(defect(&op, "a used result was given no local"));
                            };
                            writer.instruction(&Instruction::LocalSet(local));
                        } else {
                            writer.instruction(&Instruction::Drop);
                        }
                        height -= 1;
                    }
                }
                Role::Rematerialized => {
                    return Err(defect(&op, "a rematerialized operation is a node of a tree"));
                }
            }
        }
        if !final_ret && root_name != "wasm.end" {
            debug_assert_eq!(height, 0, "the stack is not empty after the root at {root}");
        }
        // After the `wasm.end` of a region its values are the construct's results.
        height = 0;
    }
    Ok(())
}

/// Writes the construct `op` (`wasm.block`, `wasm.loop` or `wasm.if`), whose stack operands are
/// on the stack: its instruction and block type, each region that is written, and its `end`; or
/// `br_if` for an `if` of that form.
fn emit_construct(
    function: &FunctionPlan,
    op: &Operation,
    writer: &mut CodeWriter<'_>,
    cx: &mut Encoder<'_>,
) -> Result<(), Report> {
    if let Some(level) = br_if_level(op) {
        writer.instruction(&Instruction::BrIf(level));
        return Ok(());
    }
    let (opcode, first_input) = match name_of(op).as_str() {
        "wasm.block" => (0x02, 0),
        "wasm.loop" => (0x03, 0),
        _ => (0x04, 1),
    };
    write_block_type(op, opcode, first_input, writer, cx)?;
    let blocks = written_blocks(op);
    let (first, second) = match blocks.as_slice() {
        [first] => (*first, None),
        [first, second] => (*first, Some(*second)),
        _ => return Err(defect(op, "a construct has no region to write")),
    };
    enter(function, op, first, writer, cx)?;
    if let Some(second) = second {
        writer.instruction(&Instruction::Else);
        enter(function, op, second, writer, cx)?;
    }
    writer.instruction(&Instruction::End);
    Ok(())
}

/// Writes `block`, a block of a region of `construct`: its arguments, on the stack, are set into
/// their locals from the last to the first (an unused one dropped), then its plan.
fn enter(
    function: &FunctionPlan,
    construct: &Operation,
    block: BlockRef,
    writer: &mut CodeWriter<'_>,
    cx: &mut Encoder<'_>,
) -> Result<(), Report> {
    let arguments = block.borrow().argument_values().collect::<Vec<ValueRef>>();
    for argument in arguments.iter().rev() {
        if argument.borrow().is_used() {
            let Some(local) = function.locals.of(*argument) else {
                return Err(defect(construct, "an argument of a region was given no local"));
            };
            writer.instruction(&Instruction::LocalSet(local));
        } else {
            writer.instruction(&Instruction::Drop);
        }
    }
    let Some(plan) = function.plan_of(block) else {
        return Err(defect(construct, "a region that is written was not planned"));
    };
    emit_block(function, plan, writer, cx, false)
}

/// Writes the instruction `opcode` of the construct `op` with its block type, from its inputs
/// (its operands from `first_input` on) and its results.
fn write_block_type(
    op: &Operation,
    opcode: u8,
    first_input: usize,
    writer: &mut CodeWriter<'_>,
    cx: &mut Encoder<'_>,
) -> Result<(), Report> {
    let inputs = op
        .operands()
        .iter()
        .enumerate()
        .skip(first_input)
        .map(|(index, operand)| val_type_of(op, &format!("operand {index}"), operand.borrow().ty()))
        .collect::<Result<Vec<_>, _>>()?;
    let results = op
        .results()
        .iter()
        .enumerate()
        .map(|(index, result)| val_type_of(op, &format!("result {index}"), result.borrow().ty()))
        .collect::<Result<Vec<_>, _>>()?;
    let simple = match (inputs.as_slice(), results.as_slice()) {
        ([], []) => Some(BlockType::Empty),
        ([], [result]) => Some(BlockType::Result(result.to_encoder())),
        _ => None,
    };
    match simple {
        Some(ty) => {
            let instruction = match opcode {
                0x02 => Instruction::Block(ty),
                0x03 => Instruction::Loop(ty),
                _ => Instruction::If(ty),
            };
            writer.instruction(&instruction);
        }
        None => {
            cx.desc.features.insert(Feature::Multivalue);
            let ty = cx.desc.intern_type(FuncType::new(inputs, results));
            writer.fixup_index(&[opcode], FixupKind::BlockType(ty));
        }
    }
    Ok(())
}

/// The level of the `wasm.br` `op`.
fn level_of(op: &Operation) -> Result<u32, Report> {
    let Some(br) = op.downcast_ref::<wasm::Br>() else {
        return Err(unsupported(op, "it is not the operation its name says it is"));
    };
    let level = *br.get_level();
    u32::try_from(level).map_err(|_| unsupported(op, format!("its level {level} is negative")))
}

/// The labels of the `wasm.br_table` `op`: the level of the `wasm.br` of each entry's region, in
/// order, and of the default's.
fn table_labels(op: &Operation) -> Result<(Vec<u32>, u32), Report> {
    let Some(table) = op.downcast_ref::<wasm::BrTable>() else {
        return Err(unsupported(op, "it is not the operation its name says it is"));
    };
    let label = |region: runic_ir::RegionRef| -> Result<u32, Report> {
        let entry = region.borrow().entry_block_ref();
        let br = entry
            .and_then(|entry| entry.borrow().body().iter().next().map(|op| op.as_operation_ref()));
        match br {
            Some(br) if br.borrow().is::<wasm::Br>() => level_of(&br.borrow()),
            _ => Err(unsupported(op, "a region of the table does not hold a `wasm.br`")),
        }
    };
    let mut labels = Vec::with_capacity(table.num_entries());
    for index in 0..table.num_entries() {
        let Some(region) = table.entry_region(index) else {
            return Err(defect(op, "an entry of the table has no region"));
        };
        labels.push(label(region)?);
    }
    let default = label(table.default_region().as_region_ref())?;
    Ok((labels, default))
}
