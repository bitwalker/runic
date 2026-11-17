//! The pass `wasm-lower-frames`, which gives every `mem.alloca` a slot in a frame on the shadow
//! stack.

use alloc::{format, rc::Rc, vec::Vec};
use core::fmt;

use runic_dialect_mem as mem;
use runic_dialect_wasm::{GlobalGet, GlobalRef, GlobalSet, WasmDialect, WasmOpBuilder};
use runic_ir::{
    Align, Builder, Context, DataLayout, EntityMut, Forward, Int, LoopLikeOpInterface, Op,
    OpBuilder, Operation, OperationName, OperationRef, RawWalk, RegionBranchOpInterface, Report,
    SourceSpan, Spanned, Symbol, ValueRef, WalkResult,
    diagnostics::report,
    dialects::builtin::{BuiltinOpBuilder, Function, Module},
    pass::{Pass, PassExecutionState, PostPassStatus},
};

/// The name `wasm-lower-frames` is registered under.
pub const NAME: &str = "wasm-lower-frames";

const DESCRIPTION: &str = "Gives every mem.alloca of a function a slot in one frame on the shadow \
                           stack in linear memory, addressed from @__stack_pointer";

/// The alignment of the stack pointer at every call, and the least alignment of a frame, in
/// bytes, as the Wasm C ABI has it.
pub const STACK_ALIGN: u64 = 16;

/// Gives every `mem.alloca` of a function a slot in one frame per function on the shadow stack in
/// linear memory, addressed from the stack pointer `@__stack_pointer` ([crate::symbols]).
///
/// It runs on each `builtin.function` with a body under the operation it runs on, nested ones
/// included, after `cfg-to-scf` and before `convert-to-wasm`; the operation is a module, or holds
/// modules through worlds and components, and one that holds none is an error. A function with no `mem.alloca` (in
/// its body, outside the functions nested in it) is left as it is. For one with some:
///
/// * Its control flow must be structured: a region of the function (outside nested functions)
///   with more than one block is an error that says to run `cfg-to-scf` first.
/// * A `mem.alloca` nested in a loop (an operation between it and the function that is a
///   `LoopLikeOpInterface` or a `RegionBranchOpInterface` with a loop: `scf.while`, `wasm.loop`)
///   is an error naming it: each alloca has one slot, and one in a loop would need a slot per
///   iteration.
/// * The frame has one slot per alloca, in the order they appear, each at an offset
///   aligned to its alignment (the operation's `align`, else the layout's alignment of its type)
///   and of the size the layout gives its type; its size is rounded up to 16 bytes, and its
///   alignment is the largest slot alignment, at least 16.
/// * At the start of the entry block, the stack pointer is read (`%sp = wasm.global_get
///   @__stack_pointer : i32`), the frame is taken below it (`%fp = wasm.sub %sp, <size>`, then a
///   `wasm.and` with `-A` when the frame alignment `A` is above 16) and the stack pointer is set to
///   it (`wasm.global_set @__stack_pointer, %fp`). The stack grows down and is 16-byte aligned at
///   every call.
/// * Each alloca is replaced, where it stands, by its slot's address (`%fp + <offset>`, `%fp`
///   itself for offset 0) and a `builtin.unrealized_conversion_cast` of that `i32` to the alloca's
///   pointer type, which `convert-to-wasm` folds away.
/// * Before every operation that leaves the function (`Operation::leaves_callable`: a return or a
///   tail call, at any depth, outside nested functions) the stack pointer is restored
///   (`wasm.global_set @__stack_pointer, %sp`). A tail call's frame is dead when the callee runs.
///
/// The stack pointer is found in the module that directly holds the function, or declared there
/// ([crate::symbols]). The
/// result verifies as it is, with `mem`, `wasm` and the casts mixed, and is the input of
/// `convert-to-wasm`. A data layout must be in scope of a function with an alloca; the
/// frame's size must fit in 32 bits.
#[derive(Default)]
pub struct LowerFrames;

runic_ir::inventory::submit!(runic_ir::pass::PassInfo::new::<LowerFrames>(NAME, DESCRIPTION));

impl Pass for LowerFrames {
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

    fn initialize(&mut self, context: Rc<Context>) -> Result<(), Report> {
        context.get_or_register_dialect::<WasmDialect>();
        Ok(())
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let root = op.as_operation_ref();
        drop(op);
        if crate::convert::modules_under(root).is_empty() {
            return Err(crate::no_module_error(NAME, root));
        }
        let mut functions = Vec::new();
        let _ = root.raw_prewalk::<Forward, _, _>(|op: OperationRef| {
            if let Ok(function) = op.try_downcast_op::<Function>()
                && !Symbol::is_declaration(&*function.borrow())
            {
                functions.push(op);
            }
            WalkResult::<()>::Continue(())
        });
        let mut changed = false;
        for function in functions {
            changed |= lower_function(function)?;
        }
        state.set_post_pass_status(PostPassStatus::from(changed));
        if !changed {
            state.preserved_analyses_mut().preserve_all();
        }
        Ok(())
    }
}

/// The layout of one frame: the offset of each slot, the size and the alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// The offset of each slot from the frame's base, in the order of the slots.
    pub offsets: Vec<u64>,
    /// The size of the frame in bytes, a multiple of 16.
    pub size: u64,
    /// The alignment of the frame's base in bytes: the largest slot alignment, at least 16.
    pub align: u64,
}

impl Frame {
    /// The frame of `slots`, each a size and an alignment in bytes, laid out in order: each at the
    /// first offset after the one before that its alignment divides. `None` when the size does
    /// not fit in 32 bits.
    pub fn of(slots: &[(u64, Align)]) -> Option<Self> {
        let mut offsets = Vec::with_capacity(slots.len());
        let mut end = 0u64;
        let mut align = STACK_ALIGN;
        for (size, slot_align) in slots {
            let offset = slot_align.align_up(end)?;
            offsets.push(offset);
            end = offset.checked_add(*size)?;
            align = align.max(slot_align.bytes());
        }
        let size = Align::from_bytes(STACK_ALIGN)?.align_up(end)?;
        (size <= u64::from(u32::MAX)).then_some(Self {
            offsets,
            size,
            align,
        })
    }
}

/// Lowers the allocas of `function` (see [LowerFrames]); whether it changed anything.
fn lower_function(function: OperationRef) -> Result<bool, Report> {
    let allocas = allocas_of(function);
    if allocas.is_empty() {
        return Ok(false);
    }
    check_structured(function)?;
    for alloca in &allocas {
        check_not_in_loop(*alloca, function)?;
    }
    let context = function.borrow().context_rc();
    let Some(layout) = DataLayout::closest(&function.borrow()) else {
        return Err(report!(
            message: (format!(
                "`{NAME}` needs a data layout to lay out a frame: state `data_layout = \
                 #builtin.data_layout<little, ptr = 32>` on the module, or compile with `--target`"
            )),
            labels: [primary(function.borrow().span(), "no data layout is in scope of this function")],
        ));
    };
    let slots = allocas
        .iter()
        .map(|alloca| slot_of(*alloca, &layout))
        .collect::<Result<Vec<_>, Report>>()?;
    let Some(frame) = Frame::of(&slots) else {
        return Err(report!(
            message: (format!("`{NAME}` cannot lay out this function's frame: a Wasm address has 32 bits")),
            labels: [primary(
                function.borrow().span(),
                "the allocas of this function need a frame of more than 2^32 - 1 bytes"
            )],
        ));
    };
    let Some(module) = function.borrow().nearest_parent_op::<Module>() else {
        return Err(report!(
            message: (format!("`{NAME}` needs the function in a `builtin.module`, to name its stack pointer")),
            labels: [primary(function.borrow().span(), "this function is in no module")],
        ));
    };
    let stack_pointer = crate::symbols::stack_pointer(module)?;

    let mut builder = OpBuilder::new(context.clone());
    let span = function.borrow().span();
    let (sp, fp) = prologue(&mut builder, function, stack_pointer, &frame, span)?;
    for (alloca, offset) in allocas.iter().zip(&frame.offsets) {
        replace_alloca(&mut builder, *alloca, fp, *offset)?;
    }
    for exit in exits_of(function) {
        let span = exit.borrow().span();
        builder.set_insertion_point_before(exit);
        let set = builder.global_set(stack_pointer, sp, span)?;
        name_locally(set, stack_pointer)?;
    }
    Ok(true)
}

/// Makes `op`, a `wasm.global_get` or a `wasm.global_set` of the stack pointer `global`, name it
/// by its bare name when it is in the module nearest `op` ([crate::symbols::local_path]).
fn name_locally(op: OperationRef, global: GlobalRef) -> Result<(), Report> {
    let Some(path) =
        crate::symbols::local_path(op, global.as_operation_ref(), crate::symbols::STACK_POINTER)
    else {
        return Ok(());
    };
    if let Ok(mut get) = op.try_downcast_op::<GlobalGet>() {
        get.borrow_mut().global_mut().set_path(path);
    } else if let Ok(mut set) = op.try_downcast_op::<GlobalSet>() {
        set.borrow_mut().global_mut().set_path(path);
    }
    Ok(())
}

/// The operations under `function` (its own, outside nested functions) for which `select` is
/// true, in pre-order.
fn collect(function: OperationRef, select: impl Fn(&Operation) -> bool) -> Vec<OperationRef> {
    let mut found = Vec::new();
    let _ = function.raw_prewalk::<Forward, _, _>(|op: OperationRef| {
        if op == function {
            return WalkResult::<()>::Continue(());
        }
        let op_ref = op.borrow();
        if op_ref.is::<Function>() {
            return WalkResult::Skip;
        }
        if select(&op_ref) {
            found.push(op);
        }
        WalkResult::Continue(())
    });
    found
}

/// The `mem.alloca`s of `function`, in the order they appear.
fn allocas_of(function: OperationRef) -> Vec<OperationRef> {
    collect(function, |op| op.is::<mem::Alloca>())
}

/// The operations that leave `function`: its returns and tail calls, at any depth.
fn exits_of(function: OperationRef) -> Vec<OperationRef> {
    collect(function, Operation::leaves_callable)
}

/// An error when a region of `function` (outside nested functions) has more than one block.
fn check_structured(function: OperationRef) -> Result<(), Report> {
    let mut owners = collect(function, |op| op.has_regions());
    owners.insert(0, function);
    for owner in owners {
        let owner = owner.borrow();
        for (index, region) in owner.regions().iter().enumerate() {
            let blocks = region.body().iter().count();
            if blocks > 1 {
                return Err(report!(
                    message: (format!(
                        "`{NAME}` needs structured control flow: run `cfg-to-scf` first"
                    )),
                    labels: [primary(
                        owner.span(),
                        format!("region {index} of this `{}` has {blocks} blocks", owner.name())
                    )],
                ));
            }
        }
    }
    Ok(())
}

/// Whether `op` is a loop: a [LoopLikeOpInterface], or a [RegionBranchOpInterface] whose regions
/// form a loop (the test of `Operation::has_nested_exit_or_loop`).
fn is_loop(op: &Operation) -> bool {
    op.implements::<dyn LoopLikeOpInterface>()
        || op.as_trait::<dyn RegionBranchOpInterface>().is_some_and(|op| op.has_loop())
}

/// An error naming `alloca` when an operation between it and `function` is a loop.
fn check_not_in_loop(alloca: OperationRef, function: OperationRef) -> Result<(), Report> {
    let mut parent = alloca.borrow().parent_op();
    while let Some(op) = parent {
        if op == function {
            break;
        }
        let op_ref = op.borrow();
        if is_loop(&op_ref) {
            return Err(report!(
                message: "each `mem.alloca` has one slot in its function's frame; one in a loop \
                          would need a slot per iteration",
                labels: [
                    primary(alloca.borrow().span(), "this `mem.alloca` is in a loop"),
                    context(op_ref.span(), format!("the loop, a `{}`", op_ref.name())),
                ],
            ));
        }
        parent = op_ref.parent_op();
    }
    Ok(())
}

/// The size and alignment of the slot of `alloca` under `layout`.
fn slot_of(alloca: OperationRef, layout: &DataLayout) -> Result<(u64, Align), Report> {
    let op = alloca.borrow();
    let alloca_op = op.downcast_ref::<mem::Alloca>().expect("a mem.alloca");
    let ty = *alloca_op.get_ty();
    let unknown = |error: runic_ir::LayoutError| {
        report!(
            message: (format!("`{NAME}` cannot give this `mem.alloca` a slot")),
            labels: [primary(op.span(), format!("`{ty}` has no layout: {error}"))],
        )
    };
    let size = layout.size_of(ty).map_err(unknown)?;
    let align = match alloca_op.get_align().bytes() {
        Some(align) => align,
        None => layout.align_of(ty).map_err(unknown)?,
    };
    Ok((size, align))
}

/// A `wasm.const` of the `i32` `bits` (its low 32 bits) at the builder's insertion point.
fn const_i32(builder: &mut OpBuilder, bits: u64, span: SourceSpan) -> ValueRef {
    builder.int(
        Int::from_bits(32, u128::from(bits & 0xffff_ffff)),
        builder.context().i32(),
        span,
    )
}

/// Builds, at the start of the entry block of `function`, the reading of the stack pointer, the
/// frame below it and the new stack pointer; the old stack pointer and the frame's base.
fn prologue(
    builder: &mut OpBuilder,
    function: OperationRef,
    stack_pointer: GlobalRef,
    frame: &Frame,
    span: SourceSpan,
) -> Result<(ValueRef, ValueRef), Report> {
    let front = {
        let function = function.borrow();
        let function = function.downcast_ref::<Function>().expect("a builtin.function");
        function.body().entry_block_ref().and_then(|entry| entry.borrow().front())
    };
    let Some(front) = front else {
        return Err(Report::msg("a function with an alloca has an operation in its entry block"));
    };
    builder.set_insertion_point_before(front);
    let sp = builder.global_get(stack_pointer, span)?;
    let get = sp.borrow().get_defining_op().expect("the result of a wasm.global_get");
    name_locally(get, stack_pointer)?;
    let size = const_i32(builder, frame.size, span);
    let mut fp = builder.sub(sp, size, span)?;
    if frame.align > STACK_ALIGN {
        let mask = const_i32(builder, frame.align.wrapping_neg(), span);
        fp = builder.and(fp, mask, span)?;
    }
    let set = builder.global_set(stack_pointer, fp, span)?;
    name_locally(set, stack_pointer)?;
    Ok((sp, fp))
}

/// Replaces `alloca`, where it stands, by the address `fp + offset` and a cast of it to the
/// alloca's pointer type.
fn replace_alloca(
    builder: &mut OpBuilder,
    mut alloca: OperationRef,
    fp: ValueRef,
    offset: u64,
) -> Result<(), Report> {
    let (span, mut result) = {
        let op = alloca.borrow();
        (op.span(), op.results()[0].borrow().as_value_ref())
    };
    let ty = result.borrow().ty();
    builder.set_insertion_point_before(alloca);
    let address = if offset == 0 {
        fp
    } else {
        let offset = const_i32(builder, offset, span);
        builder.add(fp, offset, span)?
    };
    let cast = builder.unrealized_conversion_cast([address], [ty], span)?;
    let pointer = cast.borrow().as_operation().results()[0].borrow().as_value_ref();
    result.borrow_mut().replace_all_uses_with(pointer);
    alloca.borrow_mut().erase();
    Ok(())
}
