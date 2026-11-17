//! A function body: one `builtin.function` in Wasm form becomes a [FunctionBody].
//!
//! [encode_function] plans the body's block ([plan](super::plan)), gives locals to the values
//! that need them ([locals](super::locals)), and writes the instructions of the plan's roots in
//! order through a [CodeWriter], each operation's instruction by [encode].
//! A region is the unit of the walk: [emit_block] writes the plan of one block. Emitting a root
//! means: for each of its stack operands in order, the tree of the claimed operation that leaves
//! it, a `local.get` of its local, or the instruction of the rematerialized operation that
//! defines it; then the root's instruction; then, for its results from the last to the first, a
//! `local.set` of a used one and a `drop` of an unused one. A claimed operation is followed by
//! nothing, or by `local.tee` when its value has other uses. Between two roots the stack is empty;
//! debug builds check it by counting.
//!
//! The body's block ends in `builtin.ret` (its values are pushed and the function's `end` returns
//! them: no `return` is written), `builtin.tail_call` (`return_call`),
//! `wasm.return_call_indirect` or `wasm.unreachable`.

use alloc::{format, vec::Vec};

use runic_ir::{
    BlockRef, Operation, OperationRef, Report, Spanned, Symbol as _, diagnostics::report,
    dialects::builtin::Function,
};
use wasm_encoder::Instruction;

use super::{
    desc::{CodeWriter, Feature, FuncType, FunctionBody, ModuleDesc},
    instr::{
        Encoder, Entities, encode, func_type_of, is_reference, name_of, record_type_features,
        unsupported, val_type_of,
    },
    locals::{LiveRange, Locals, assign, live_range, numbering},
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

/// The operations a function's body may end in, in this version of the emitter.
const TERMINATORS: [&str; 4] = [
    "builtin.ret",
    "builtin.tail_call",
    "wasm.return_call_indirect",
    "wasm.unreachable",
];

/// The internal error of a body whose plan the encoder cannot follow: a defect of the backend.
fn defect(op: &Operation, what: &str) -> Report {
    report!(
        message: (format!(
            "the WebAssembly body of `{}` could not be encoded, which is a defect of the backend",
            name_of(op)
        )),
        labels: [primary(op.span(), alloc::string::String::from(what))],
    )
}

/// The plan of a function's body and the locals of its values, with what the encoder records
/// about the function's types.
#[derive(Clone, Debug)]
pub struct FunctionPlan {
    /// The plan of the body's block.
    pub plan: BlockPlan,
    /// The locals: the parameters, the declared locals and the emitter's.
    pub locals: Locals,
    /// The function's type.
    pub ty: FuncType,
    /// Whether a value of the body (a parameter, a declared local, a result) is a reference.
    pub uses_references: bool,
}

/// The plan and the locals of `function`, a `builtin.function` in Wasm form with a body; an error
/// names a declaration, or a parameter, local or result whose type is not a Wasm value type.
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
    let plan = plan_block(entry);
    let numbers = numbering(body_region);
    let mut ranges = Vec::new();
    uses_references |= collect_ranges(&plan, &numbers, &mut ranges)?;
    let locals = assign(&params, &declared, &ranges, config.reuse_locals);
    Ok(FunctionPlan {
        plan,
        locals,
        ty,
        uses_references,
    })
}

/// The body of `function`, a `builtin.function` in Wasm form with a body, encoded for `desc`:
/// its locals beyond the parameters and its instructions, with a fix-up for every reference to
/// another entity, which `entities` gives. The features it uses, the types its indirect calls
/// name and whether it needs a data-count section are recorded in `desc`; the function's own type
/// is not interned (the caller does that).
pub fn encode_function(
    function: OperationRef,
    desc: &mut ModuleDesc,
    entities: &mut dyn Entities,
    config: &BodyConfig,
) -> Result<FunctionBody, Report> {
    let FunctionPlan {
        plan,
        locals,
        ty,
        uses_references,
    } = plan_function(function, config)?;
    record_type_features(desc, &ty);
    if uses_references {
        desc.features.insert(Feature::ReferenceTypes);
    }
    let mut body = FunctionBody::default();
    {
        let mut writer = body.writer();
        let mut cx = Encoder {
            desc,
            entities,
            params: locals.params(),
        };
        emit_block(&plan, &locals, &mut writer, &mut cx, true)?;
        writer.instruction(&Instruction::End);
    }
    body.locals = locals.types();
    Ok(body)
}

/// Adds to `ranges` the live range of every value of `plan` that needs a local; whether a result
/// of an operation is a reference.
fn collect_ranges(
    plan: &BlockPlan,
    numbers: &runic_ir::FxHashMap<OperationRef, u32>,
    ranges: &mut Vec<LiveRange>,
) -> Result<bool, Report> {
    let mut references = false;
    for (index, step) in plan.steps.iter().enumerate() {
        let op = step.op.borrow();
        let def = numbers.get(&step.op).copied().unwrap_or(0);
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

/// Writes the plan of one block: each root's tree, in order. With `ends_function`, the block is
/// the function's body: it must end in one of the operations a body ends in, and a final
/// `builtin.ret` leaves its values on the stack for the function's `end` instead of writing
/// `return`.
pub fn emit_block(
    plan: &BlockPlan,
    locals: &Locals,
    writer: &mut CodeWriter<'_>,
    cx: &mut Encoder<'_>,
    ends_function: bool,
) -> Result<(), Report> {
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
                    "a function body that ends in `{}` is not emitted yet: it ends in \
                     `builtin.ret`, `builtin.tail_call`, `wasm.return_call_indirect` or \
                     `wasm.unreachable`",
                    name_of(&op)
                ),
            ));
        }
    }
    // The number of values on the stack, counted as instructions are written.
    let mut height: usize = 0;
    for root in plan.roots() {
        debug_assert_eq!(height, 0, "the stack is not empty before the root at {root}");
        let final_ret = ends_function
            && Some(root) == last
            && name_of(&plan.steps[root].op.borrow()) == "builtin.ret";
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
            height -= step.operands.len();
            encode(&op, writer, cx)?;
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
        if !final_ret {
            debug_assert_eq!(height, 0, "the stack is not empty after the root at {root}");
        }
    }
    Ok(())
}

/// Whether `block` holds an operation of structured control ([CONTROL](super::instr::CONTROL))
/// at any depth.
pub fn has_control(block: BlockRef) -> bool {
    let block = block.borrow();
    block.body().iter().any(|op| {
        let op = op.as_operation_ref();
        super::instr::is_control(&op.borrow())
            || op
                .borrow()
                .regions()
                .iter()
                .any(|region| region.body().iter().any(|block| has_control(block.as_block_ref())))
    })
}
