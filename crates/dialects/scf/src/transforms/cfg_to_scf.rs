use alloc::{format, rc::Rc, vec::Vec};
use core::fmt;

use runic_dialect_arith::ArithOpBuilder;
use runic_dialect_cf::{self as cf, ControlFlowOpBuilder};
use runic_dialect_ub::UndefinedBehaviorOpBuilder;
use runic_ir::{
    BlockRef, Builder, BuilderExt, Context, EntityMut, Forward, Op, OpBuilder, Operation,
    OperationName, OperationRef, RawWalk, RegionBranchTerminatorOpInterface,
    RegionExitTerminatorOpInterface, RegionRef, Report, SmallVec, Spanned, Usable, ValueRange,
    ValueRef, WalkResult,
    dialects::builtin,
    dominance::DominanceInfo,
    pass::{Pass, PassExecutionState, PassOptions, PostPassStatus, write_pass_option},
    traits::{ConstantLike, ReturnLike, Terminator},
};
use runic_transform::{self as transforms, CFGToSCFInterface};

use crate::*;

/// Lifts unstructured control flow operations (`cf`) to structured control flow operations (`scf`).
///
/// What the pass does with the dead ends of a region (the terminators without successors that do
/// not hand control back to the operation that holds the region: the exits of the function, its
/// `builtin.ret`s, tail calls and every other operation for which [Operation::leaves_callable]
/// holds, and terminators such as `ub.unreachable`) is set by its option `exits` ([Exits]):
///
/// * `exits=merge` (the default) merges the dead ends of a region that are the same operation, with
///   the same attributes and operand types, into one. If a region contains only a single kind of
///   return-like operation, all unstructured control flow ops are replaced. Otherwise a single
///   unstructured switch branching to one block per return-like operation kind remains.
/// * `exits=in-place` leaves every dead end where it is, inside the structured code that holds it,
///   and replaces all unstructured control flow of a function. Every dead end of a region with
///   more than one block is treated alike, whatever its kind: a `ub.unreachable` the user wrote
///   stays where it was, as a `builtin.ret` does. A region in which every path reaches a dead end
///   ends in a `ub.unreachable` after its structured code. A dead end reached from inside a loop is
///   still placed after the loop, in a region of an `scf.if` or `scf.index_switch` on a selector
///   that the loop computes, as the structuring places every block that is not part of a cycle.
///   A terminator that hands control back to the operation holding the region (an `scf.yield`, an
///   `scf.condition`, an `scf.break`) is not a dead end, and is merged as in `exits=merge`.
///
/// A region with a block its entry cannot reach is refused, in either mode, before anything is
/// changed.
///
/// This pass may need to create unreachable terminators in case of infinite loops, which is only
/// supported for 'builtin.func' for now. If you potentially have infinite loops inside CFG regions
/// not belonging to 'builtin.func', consider using the `transform_cfg_to_scf` function directly
/// with a corresponding [CFGToSCFInterface::create_unreachable_terminator] implementation.
#[derive(Default)]
pub struct LiftControlFlowToSCF {
    exits: Exits,
}

/// Where [LiftControlFlowToSCF] puts the exits of a function: the values of its option `exits`.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub enum Exits {
    /// `merge`: the exits of one kind become one exit, after the structured code that computes its
    /// operands; with more than one kind, an unstructured dispatch between them remains.
    #[default]
    Merge,
    /// `in-place`: every exit stays in the block it ends, wherever the structuring puts that block,
    /// and no unstructured control flow remains.
    InPlace,
}

impl Exits {
    /// The value of the option `exits` that selects this placement.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::InPlace => "in-place",
        }
    }
}

impl LiftControlFlowToSCF {
    /// The pass, placing the exits of a function as `exits` says.
    pub const fn new(exits: Exits) -> Self {
        Self { exits }
    }

    /// Where the pass places the exits of a function.
    pub const fn exits(&self) -> Exits {
        self.exits
    }
}

runic_ir::inventory::submit!(::runic_ir::pass::registry::PassInfo::new::<LiftControlFlowToSCF>(
    "cfg-to-scf",
    DESCRIPTION
));

const NAME: &str = "cfg-to-scf";

const DESCRIPTION: &str = "Replaces unstructured control flow (cf) by structured control flow \
                           (scf) where it can; `exits=in-place` leaves returns and tail calls \
                           where they are";

impl Pass for LiftControlFlowToSCF {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "lift-control-flow"
    }

    fn argument(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        DESCRIPTION
    }

    /// Reads `exits` (`merge` or `in-place`, see [Exits]) from the options of the pass pipeline.
    fn initialize_options(&mut self, options: &PassOptions) -> Result<(), Report> {
        options.check_keys(NAME, &["exits"])?;
        if let Some(value) = options.get("exits") {
            self.exits = match value.map(str::trim) {
                Some("merge") => Exits::Merge,
                Some("in-place") => Exits::InPlace,
                Some(value) => {
                    return Err(Report::msg(format!(
                        "invalid value '{value}' of option 'exits' of pass '{NAME}': expected \
                         'merge' or 'in-place'"
                    )));
                }
                None => {
                    return Err(Report::msg(format!(
                        "option 'exits' of pass '{NAME}' needs a value: expected 'exits=merge' or \
                         'exits=in-place'"
                    )));
                }
            };
        }
        Ok(())
    }

    /// Prints the pass, with `{exits=in-place}` when it leaves exits in place; the default,
    /// `merge`, is not printed.
    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(NAME)?;
        if self.exits != Exits::Merge {
            f.write_str("{")?;
            write_pass_option(f, "exits", Some(self.exits.as_str()))?;
            f.write_str("}")?;
        }
        Ok(())
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn initialize(&mut self, context: Rc<runic_ir::Context>) -> Result<(), Report> {
        // Ensure that when this pass is initialized, that the SCF dialect is registered
        context.get_or_register_dialect::<crate::ScfDialect>();

        Ok(())
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let mut transformation = ControlFlowToSCFTransformation;
        let mut changed = false;
        let in_place = self.exits == Exits::InPlace;

        let root = op.as_operation_ref();
        drop(op);

        log::debug!(target: "cfg-to-scf", "applying control flow lifting transformation pass starting from {}", root.borrow());

        // A region with a block its entry cannot reach is refused, and is looked for in every
        // region the pass would transform before any is transformed, so that a refusal leaves
        // the IR as it was.
        let reachable =
            root.raw_prewalk::<Forward, _, _>(|operation: OperationRef| -> WalkResult {
                let op = operation.borrow();
                if op.is::<builtin::Function>() {
                    drop(op);
                    operation.raw_prewalk::<Forward, _, _>(
                        |inner: OperationRef| -> WalkResult {
                            for region in inner.borrow().regions() {
                                if let Err(error) = transforms::check_all_blocks_reachable(&region)
                                {
                                    return WalkResult::Break(error);
                                }
                            }
                            WalkResult::Continue(())
                        },
                    )?;
                    WalkResult::Skip
                } else if op.is::<builtin::World>()
                    || op.is::<builtin::Component>()
                    || op.is::<builtin::Module>()
                {
                    WalkResult::Continue(())
                } else {
                    WalkResult::Skip
                }
            });
        if reachable.was_interrupted() {
            state.set_post_pass_status(PostPassStatus::Unchanged);
            return reachable.into_result();
        }

        let result = root.raw_prewalk::<Forward, _, _>(|operation: OperationRef| -> WalkResult {
            let op = operation.borrow();
            if op.is::<builtin::Function>() {
                if op.regions().is_empty() {
                    return WalkResult::Skip;
                }

                let dominfo = if OperationRef::ptr_eq(&operation, &root) {
                    state.analysis_manager().get_analysis::<DominanceInfo>()
                } else {
                    state.analysis_manager().get_child_analysis::<DominanceInfo>(operation)
                };

                let mut dominfo = match dominfo {
                    Ok(di) => di,
                    Err(err) => return WalkResult::Break(err),
                };
                let dominfo = Rc::make_mut(&mut dominfo);

                let visitor = |inner: OperationRef| -> WalkResult {
                    log::debug!(target: "cfg-to-scf", "applying control flow lifting to {}", inner.borrow());
                    let mut next_region = inner.borrow().regions().front().as_pointer();
                    while let Some(region) = next_region.take() {
                        next_region = region.next();

                        match lift_region(region, &mut transformation, dominfo, in_place) {
                            Ok(did_change) => {
                                log::trace!(
                                    target: "cfg-to-scf",
                                    "control flow lifting completed for region \
                                     (did_change={did_change})"
                                );
                                changed |= did_change;
                            }
                            Err(err) => return WalkResult::Break(err),
                        }
                    }

                    WalkResult::Continue(())
                };

                drop(op);

                operation.raw_postwalk::<Forward, _, _>(visitor)?;

                // Do not enter the function body in the outer walk
                WalkResult::Skip
            } else if op.is::<builtin::World>()
                || op.is::<builtin::Component>()
                || op.is::<builtin::Module>()
            {
                // We only care to recurse into ops that can contain functions
                log::trace!(
                    target: "cfg-to-scf",
                    "looking for functions to apply control flow lifting to in '{}'",
                    op.name()
                );
                WalkResult::Continue(())
            } else {
                // Skip all other ops
                log::trace!("skipping control flow lifting for '{}'", op.name());
                WalkResult::Skip
            }
        });

        if result.was_interrupted() {
            state.set_post_pass_status(PostPassStatus::Unchanged);
            return result.into_result();
        }

        log::debug!(
            target: "cfg-to-scf",
            "control flow lifting transformation pass completed successfully (changed = {changed}"
        );
        if !changed {
            state.preserved_analyses_mut().preserve_all();
        }

        state.set_post_pass_status(changed.into());

        Ok(())
    }
}

/// Lifts the control flow of `region` to structured control flow with `interface`, and answers
/// whether it changed. With `in_place` (`exits=in-place`), the dead ends of the region are taken
/// out first ([PendingExits::take]) and put back afterwards ([PendingExits::restore]). When the
/// structuring fails part-way, its error is returned and the IR is in the unspecified state the
/// structuring left it in ([transforms::transform_cfg_to_scf]): each exit whose placeholder is
/// still in a block is put back in its place, and no placeholder is left in a block, but an exit
/// whose placeholder the structuring moved out of the region goes with it.
fn lift_region(
    region: RegionRef,
    interface: &mut dyn CFGToSCFInterface,
    dominfo: &mut DominanceInfo,
    in_place: bool,
) -> Result<bool, Report> {
    let pending = if in_place {
        PendingExits::take(region)
    } else {
        None
    };
    if pending.is_some() {
        dominfo.invalidate_region(region);
    }
    match transforms::transform_cfg_to_scf(region, interface, dominfo) {
        Ok(did_change) => {
            if let Some(pending) = pending {
                let restored = pending.restore(true);
                dominfo.invalidate_region(region);
                restored?;
            }
            Ok(did_change)
        }
        Err(err) => {
            if let Some(pending) = pending {
                let _ = pending.restore(false);
                dominfo.invalidate_region(region);
            }
            Err(err)
        }
    }
}

/// The exits of a region that `exits=in-place` has taken out while the region is structured.
///
/// [PendingExits::take] replaces each dead end of the region (see [is_dead_end]) by an `scf.pending_exit` with the exit's operands, followed
/// by a `cf.br` to one block, added at the end of the region, that holds only `ub.unreachable`; the
/// exit is detached, and its operands are dropped, so that no detached operation is a user of a
/// value while the structuring runs. The structuring then sees one kind of exit only. Then
/// [PendingExits::restore] puts each exit back where its placeholder is, with the placeholder's
/// operands at that time, and erases the placeholder and every operation after it in its block.
struct PendingExits {
    /// The block every placeholder branches to, which holds only a `ub.unreachable`.
    unreachable: BlockRef,
    /// Each placeholder, the exit it stands for, and the size of each operand group of the exit.
    exits: Vec<(OperationRef, OperationRef, SmallVec<[usize; 2]>)>,
}

impl PendingExits {
    /// Takes the dead ends out of `region`, if it has more than one block and any dead end: see
    /// [PendingExits].
    fn take(region: RegionRef) -> Option<Self> {
        let blocks = region
            .borrow()
            .body()
            .iter()
            .map(|block| block.as_block_ref())
            .collect::<SmallVec<[BlockRef; 8]>>();
        if blocks.len() < 2 {
            return None;
        }
        let exits = blocks
            .iter()
            .filter_map(|block| block.borrow().terminator())
            .filter(|terminator| is_dead_end(&terminator.borrow()))
            .collect::<SmallVec<[OperationRef; 4]>>();
        if exits.is_empty() {
            return None;
        }

        let (context, span) = {
            let parent = region.parent().expect("a region being lifted has a parent");
            let parent = parent.borrow();
            (parent.context_rc(), parent.span())
        };
        let mut builder = OpBuilder::new(context);
        let unreachable = builder.create_block(region, None, &[]);
        builder.set_insertion_point_to_end(unreachable);
        builder.unreachable(span);

        let mut pending = Vec::with_capacity(exits.len());
        for mut exit in exits {
            let block = exit.parent().expect("an exit is in a block");
            let (operands, groups, span) = {
                let op = exit.borrow();
                let operands = op
                    .operands()
                    .all()
                    .iter()
                    .map(|operand| operand.borrow().as_value_ref())
                    .collect::<SmallVec<[ValueRef; 4]>>();
                let groups = op
                    .operands()
                    .groups()
                    .map(|group| group.len())
                    .collect::<SmallVec<[usize; 2]>>();
                (operands, groups, op.span())
            };
            {
                let mut op = exit.borrow_mut();
                op.remove();
                op.operands_mut().clear();
            }
            builder.set_insertion_point_to_end(block);
            let placeholder =
                builder.create::<PendingExit, (SmallVec<[ValueRef; 4]>,)>(span)(operands)
                    .expect("an `scf.pending_exit` takes operands of any type")
                    .as_operation_ref();
            builder
                .br(unreachable, [], span)
                .expect("a `cf.br` to a block without arguments is well formed");
            pending.push((placeholder, exit, groups));
        }

        Some(Self {
            unreachable,
            exits: pending,
        })
    }

    /// Puts every exit back in place of its placeholder, with the placeholder's operands, and
    /// erases the placeholder and what follows it in its block; then erases the block of
    /// `ub.unreachable` the placeholders branched to, if nothing branches to it any more.
    ///
    /// With `check`, an operation after a placeholder whose result is used by an operation that
    /// is not erased with it is an error (the structuring is assumed never to make one), and the
    /// operations after that placeholder are left as they are; without `check` such operations
    /// are left without an error. Every exit is put back either way; the first error is returned.
    fn restore(self, check: bool) -> Result<(), Report> {
        let mut first_error = None;
        for (mut placeholder, exit, groups) in self.exits {
            if let Err(error) = Self::put_back(placeholder, exit, &groups) {
                first_error.get_or_insert(error);
                continue;
            }

            let mut followers = SmallVec::<[OperationRef; 4]>::new();
            let mut next = placeholder.next();
            while let Some(op) = next {
                next = op.next();
                followers.push(op);
            }
            let escaping = followers.iter().copied().find(|follower| {
                follower.borrow().results().all().iter().any(|result| {
                    result.borrow().iter_uses().any(|user| {
                        let user = user.owner.borrow();
                        !followers.iter().any(|erased| erased.borrow().is_ancestor_of(&user))
                    })
                })
            });
            match escaping {
                Some(op) => {
                    if check {
                        let op = op.borrow();
                        first_error.get_or_insert(runic_ir::diagnostics::report!(
                            message: "an internal assumption of `cfg-to-scf` does not hold",
                            labels: [primary(
                                op.span(),
                                format!(
                                    "this `{}` follows an exit that `cfg-to-scf` puts back in \
                                     place, and an operation the pass keeps uses its result",
                                    op.name()
                                )
                            )],
                        ));
                    }
                }
                None => {
                    // The constants the structuring made for these operations (flags and
                    // selectors, created once at the start of the region) may now be unused.
                    let read = followers
                        .iter()
                        .flat_map(|follower| {
                            follower
                                .borrow()
                                .operands()
                                .all()
                                .iter()
                                .map(|operand| operand.borrow().as_value_ref())
                                .collect::<SmallVec<[ValueRef; 4]>>()
                        })
                        .collect::<SmallVec<[ValueRef; 8]>>();
                    for mut follower in followers.into_iter().rev() {
                        follower.borrow_mut().erase();
                    }
                    for value in read {
                        let defining_op = {
                            let value = value.borrow();
                            if value.is_used() {
                                continue;
                            }
                            value.get_defining_op()
                        };
                        if let Some(mut op) = defining_op
                            && op.parent().is_some()
                            && op.borrow().implements::<dyn ConstantLike>()
                        {
                            op.borrow_mut().erase();
                        }
                    }
                }
            }
            placeholder.borrow_mut().erase();
        }

        let mut unreachable = self.unreachable;
        let unused = {
            let block = unreachable.borrow();
            block.parent().is_some() && !block.is_entry_block() && !block.has_predecessors()
        };
        if unused {
            let ops = unreachable
                .borrow()
                .body()
                .iter()
                .map(|op| op.as_operation_ref())
                .collect::<SmallVec<[OperationRef; 1]>>();
            for mut op in ops.into_iter().rev() {
                op.borrow_mut().erase();
            }
            unreachable.borrow_mut().erase();
        }

        first_error.map_or(Ok(()), Err)
    }

    /// Inserts `exit` before `placeholder`, its operands those of `placeholder`, in operand groups
    /// of the sizes `groups` gives.
    fn put_back(
        placeholder: OperationRef,
        mut exit: OperationRef,
        groups: &[usize],
    ) -> Result<(), Report> {
        let (context, values) = {
            let op = placeholder.borrow();
            if op.parent().is_none() {
                return Err(runic_ir::diagnostics::report!(
                    message: "an internal assumption of `cfg-to-scf` does not hold",
                    labels: [primary(
                        exit.borrow().span(),
                        "the structuring removed the place of this exit"
                    )],
                ));
            }
            let values = op
                .operands()
                .all()
                .iter()
                .map(|operand| operand.borrow().as_value_ref())
                .collect::<SmallVec<[ValueRef; 4]>>();
            (op.context_rc(), values)
        };
        debug_assert_eq!(values.len(), groups.iter().sum::<usize>());
        exit.insert_before(placeholder);
        let mut values = values.into_iter().enumerate();
        let mut op = exit.borrow_mut();
        for (group, &len) in groups.iter().enumerate() {
            let operands = values
                .by_ref()
                .take(len)
                .map(|(index, value)| make_operand(&context, value, exit, index))
                .collect::<SmallVec<[_; 4]>>();
            op.operands_mut().extend_group(group, operands);
        }
        Ok(())
    }
}

/// Whether `op` is a dead end of its region under `exits=in-place`: a terminator without
/// successors that leaves the function ([Operation::leaves_callable]), or that is not one of the
/// terminators that hand control back to the operation holding the region (those that are
/// [ReturnLike], or implement [RegionBranchTerminatorOpInterface] or
/// [RegionExitTerminatorOpInterface]), such as `ub.unreachable`.
fn is_dead_end(op: &Operation) -> bool {
    op.implements::<dyn Terminator>()
        && op.num_successors() == 0
        && (op.leaves_callable()
            || !(op.implements::<dyn ReturnLike>()
                || op.implements::<dyn RegionBranchTerminatorOpInterface>()
                || op.implements::<dyn RegionExitTerminatorOpInterface>()))
}

/// An operand of `owner` at position `index` that uses `value`.
fn make_operand(
    context: &Context,
    value: ValueRef,
    owner: OperationRef,
    index: usize,
) -> runic_ir::OpOperand {
    let index = u16::try_from(index).expect("an operation has fewer than 2^16 operands");
    context.make_operand(value, owner, index)
}

/// Implementation of [CFGToSCFInterface] used to lift unstructured control flow operations into
/// the structured control flow operations of this dialect.
struct ControlFlowToSCFTransformation;

impl CFGToSCFInterface for ControlFlowToSCFTransformation {
    /// Creates an `scf.if` op if `control_flow_cond_op` is a `cf.cond_br` op, or an
    /// `scf.index_switch` if it is a `cf.switch`. Otherwise, returns an error.
    fn create_structured_branch_region_op(
        &self,
        builder: &mut runic_ir::OpBuilder,
        control_flow_cond_op: runic_ir::OperationRef,
        result_types: &[runic_ir::Type],
        regions: &mut runic_ir::SmallVec<[runic_ir::RegionRef; 2]>,
    ) -> Result<runic_ir::OperationRef, runic_ir::Report> {
        let cf_op = control_flow_cond_op.borrow();
        if let Some(cond_br) = cf_op.downcast_ref::<cf::CondBr>() {
            assert_eq!(regions.len(), 2);

            let span = cond_br.span();
            let mut if_op = builder.r#if(cond_br.condition().as_value_ref(), result_types, span)?;
            let mut op = if_op.borrow_mut();
            let operation = op.as_operation_ref();

            op.then_body_mut().take_body(regions[0]);
            op.else_body_mut().take_body(regions[1]);

            return Ok(operation);
        }

        if let Some(switch) = cf_op.downcast_ref::<cf::Switch>() {
            let span = switch.span();
            let cases = switch.cases();
            let num_cases = cases.len();
            assert_eq!(regions.len(), num_cases + 1);
            let cases = cases.iter().map(|case| *case.key());
            let mut switch_op = builder.index_switch(
                switch.selector().as_value_ref(),
                cases,
                result_types,
                span,
            )?;
            let mut op = switch_op.borrow_mut();
            let operation = op.as_operation_ref();

            // If any of the case targets are duplicated, we have to duplicate the regions or
            // we will fail to properly lower the input

            // The order of the regions match the original 'cf.switch', hence the fallback region
            // coming last.
            for (index, source_region) in regions.iter().copied().take(num_cases).enumerate() {
                let mut case_region = op.get_case_region(index);
                case_region.borrow_mut().take_body(source_region);
            }
            op.default_region_mut().take_body(*regions.last().unwrap());

            return Ok(operation);
        }

        Err(runic_ir::diagnostics::report!(
            message: "control flow transformation failed",
            labels: [primary(
                cf_op.span(),
                "unknown control flow operation cannot be lifted to structured control flow"
            )],
        ))
    }

    /// Creates an `scf.yield` op returning the given results.
    fn create_structured_branch_region_terminator_op(
        &self,
        span: runic_ir::SourceSpan,
        builder: &mut runic_ir::OpBuilder,
        _branch_region_op: runic_ir::OperationRef,
        _replaced_control_flow_op: Option<runic_ir::OperationRef>,
        results: ValueRange<'_, 2>,
    ) -> Result<(), runic_ir::Report> {
        builder.r#yield(results, span)?;

        Ok(())
    }

    /// Creates an `scf.while` op. The loop body is made the before-region of the
    /// while op and terminated with an `scf.condition` op. The after-region does
    /// nothing but forward the iteration variables.
    fn create_structured_do_while_loop_op(
        &self,
        builder: &mut runic_ir::OpBuilder,
        replaced_op: runic_ir::OperationRef,
        loop_values_init: ValueRange<'_, 2>,
        condition: runic_ir::ValueRef,
        loop_values_next_iter: ValueRange<'_, 2>,
        loop_body: runic_ir::RegionRef,
    ) -> Result<runic_ir::OperationRef, runic_ir::Report> {
        let span = replaced_op.span();

        // Results are derived from the forwarded values given to `scf.condition`
        let result_types = loop_values_next_iter
            .iter()
            .map(|v| v.borrow().ty())
            .collect::<SmallVec<[_; 2]>>();
        let mut while_op = builder.r#while(loop_values_init, &result_types, span)?;
        let mut op = while_op.borrow_mut();
        let operation = op.as_operation_ref();

        op.before_mut().take_body(loop_body);

        builder.set_insertion_point_to_end(op.before().body().back().as_pointer().unwrap());

        builder.condition(condition, loop_values_next_iter, span)?;

        let yielded = op
            .after()
            .entry()
            .arguments()
            .iter()
            .map(|arg| arg.upcast())
            .collect::<SmallVec<[ValueRef; 4]>>();

        builder.set_insertion_point_to_end(op.after().entry().as_block_ref());

        builder.r#yield(yielded, span)?;

        Ok(operation)
    }

    /// Creates an `arith.constant` with an i1 attribute of the given value.
    fn get_cfg_bool_value(
        &self,
        span: runic_ir::SourceSpan,
        builder: &mut runic_ir::OpBuilder,
        value: bool,
    ) -> runic_ir::ValueRef {
        builder.i1(value, span)
    }

    /// Creates an `arith.constant` with an i32 attribute of the given value.
    fn get_cfg_switch_value(
        &self,
        span: runic_ir::SourceSpan,
        builder: &mut runic_ir::OpBuilder,
        value: i32,
    ) -> runic_ir::ValueRef {
        builder.i32(value, span)
    }

    /// Creates a `cf.switch` op with the given cases and flag.
    fn create_cfg_switch_op(
        &self,
        span: runic_ir::SourceSpan,
        builder: &mut runic_ir::OpBuilder,
        flag: runic_ir::ValueRef,
        case_values: &[i32],
        case_destinations: &[runic_ir::BlockRef],
        case_arguments: &[ValueRange<'_, 2>],
        default_dest: runic_ir::BlockRef,
        default_args: ValueRange<'_, 2>,
    ) -> Result<(), Report> {
        let cases = case_values
            .iter()
            .copied()
            .zip(case_destinations.iter().copied().zip(case_arguments))
            .map(|(value, (successor, args))| {
                cf::SwitchCase::create(value, successor, args.to_vec())
            })
            .collect::<SmallVec<[_; 4]>>();

        builder.switch(flag, cases, default_dest, default_args, span)?;

        Ok(())
    }

    fn create_single_destination_branch(
        &self,
        span: runic_ir::SourceSpan,
        builder: &mut runic_ir::OpBuilder,
        _dummy_flag: runic_ir::ValueRef,
        destination: runic_ir::BlockRef,
        arguments: ValueRange<'_, 2>,
    ) -> Result<(), Report> {
        builder.br(destination, arguments, span)?;
        Ok(())
    }

    /// Branches on the `i1` produced by [Self::get_cfg_bool_value] with `cf.cond_br`.
    fn create_conditional_branch(
        &self,
        span: runic_ir::SourceSpan,
        builder: &mut runic_ir::OpBuilder,
        condition: runic_ir::ValueRef,
        true_dest: runic_ir::BlockRef,
        true_args: ValueRange<'_, 2>,
        false_dest: runic_ir::BlockRef,
        false_args: ValueRange<'_, 2>,
    ) -> Result<(), Report> {
        builder.cond_br(condition, true_dest, true_args, false_dest, false_args, span)?;
        Ok(())
    }

    /// Creates a `ub.poison` op of the given type.
    fn get_undef_value(
        &self,
        span: runic_ir::SourceSpan,
        builder: &mut runic_ir::OpBuilder,
        ty: runic_ir::Type,
    ) -> runic_ir::ValueRef {
        builder.poison(ty, span)
    }

    fn create_unreachable_terminator(
        &self,
        span: runic_ir::SourceSpan,
        builder: &mut runic_ir::OpBuilder,
        _region: runic_ir::RegionRef,
    ) -> Result<runic_ir::OperationRef, runic_ir::Report> {
        log::trace!(target: "cfg-to-scf", "creating unreachable terminator at {}", builder.insertion_point());
        let op = builder.unreachable(span);
        Ok(op.as_operation_ref())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, format};

    use builtin::BuiltinOpBuilder;
    use runic_expect_test::expect_file;
    use runic_ir::{
        Report, Signedness, SourceSpan,
        dialects::builtin::{self},
        testing::Test,
    };

    use super::*;

    /// The inputs of the lit tests of cfg-to-scf, each with its name. A file of several cases
    /// (`refused.runic`) is split at its `// -----` lines.
    const LIT_INPUTS: [(&str, &str); 8] = [
        (
            "lift",
            include_str!("../../../../../tests/lit/transforms/cfg-to-scf/lift.runic"),
        ),
        (
            "refused",
            include_str!("../../../../../tests/lit/transforms/cfg-to-scf/refused.runic"),
        ),
        (
            "refusal-changes-nothing",
            include_str!(
                "../../../../../tests/lit/transforms/cfg-to-scf/refusal-changes-nothing.runic"
            ),
        ),
        (
            "tail-call",
            include_str!("../../../../../tests/lit/transforms/cfg-to-scf/tail-call.runic"),
        ),
        (
            "two-entry-loop",
            include_str!("../../../../../tests/lit/transforms/cfg-to-scf/two-entry-loop.runic"),
        ),
        (
            "exits-in-place",
            include_str!("../../../../../tests/lit/transforms/cfg-to-scf/exits-in-place.runic"),
        ),
        (
            "exits-in-place-loops",
            include_str!(
                "../../../../../tests/lit/transforms/cfg-to-scf/exits-in-place-loops.runic"
            ),
        ),
        (
            "exits-in-place-refusal",
            include_str!(
                "../../../../../tests/lit/transforms/cfg-to-scf/exits-in-place-refusal.runic"
            ),
        ),
    ];

    /// The operations under `op`, at any depth, for which [Operation::leaves_callable] holds, and
    /// the dead ends ([is_dead_end]: the exits and terminators such as `ub.unreachable`) under it.
    fn count_exits(op: OperationRef) -> (usize, usize) {
        let mut exits = 0;
        let mut dead_ends = 0;
        op.raw_prewalk_all::<Forward, _>(|op: OperationRef| {
            let op = op.borrow();
            if op.leaves_callable() {
                exits += 1;
            }
            if is_dead_end(&op) {
                dead_ends += 1;
            }
        });
        (exits, dead_ends)
    }

    /// Every function of the inputs of the lit tests of cfg-to-scf, lifted by the pass in either
    /// mode, verifies afterwards, or is refused (a function with a block its entry cannot reach)
    /// and left as it was. With `exits=in-place`, no `cf` branch and no `scf.pending_exit` remain,
    /// the function has exactly as many exits as before, and exactly as many dead ends as before
    /// but for the `ub.unreachable` the pass adds at the end of a body of several blocks: none is
    /// dropped or duplicated.
    #[test]
    fn every_lit_input_lifts_in_both_modes_and_in_place_keeps_every_dead_end() {
        use runic_ir::{
            dialects::builtin::Function,
            parse::{ParserConfig, parse_any},
            pass::{Nesting, PassManager},
        };

        let mut lifted = 0;
        let mut refused = 0;
        for (name, text) in LIT_INPUTS {
            for (index, case) in text.split("\n// -----\n").enumerate() {
                for exits in [Exits::Merge, Exits::InPlace] {
                    let context = Rc::new(Context::default());
                    context.get_or_register_dialect::<crate::ScfDialect>();
                    context.get_or_register_dialect::<cf::ControlFlowDialect>();
                    context.get_or_register_dialect::<runic_dialect_arith::ArithDialect>();
                    context.get_or_register_dialect::<runic_dialect_ub::UndefinedBehaviorDialect>();
                    let root = parse_any(
                        ParserConfig::new(context.clone()),
                        format!("{name}#{index}"),
                        case,
                    )
                    .into_result()
                    .unwrap_or_else(|err| panic!("{name}#{index} does not parse: {err:?}"));

                    let mut functions = alloc::vec::Vec::new();
                    root.raw_prewalk_all::<Forward, _>(|op: OperationRef| {
                        let is_definition = op
                            .borrow()
                            .downcast_ref::<Function>()
                            .is_some_and(|function| !function.body().is_empty());
                        if is_definition {
                            functions.push(op);
                        }
                    });
                    for function in functions {
                        let before = format!("{}", function.borrow());
                        let (exits_before, dead_ends_before) = count_exits(function);
                        let several_blocks = function
                            .borrow()
                            .downcast_ref::<Function>()
                            .is_some_and(|function| function.body().body().len() > 1);
                        let mut passes =
                            PassManager::on::<Function>(context.clone(), Nesting::Implicit);
                        passes.add_pass(Box::new(LiftControlFlowToSCF::new(exits))).unwrap();
                        passes.enable_verifier(true);
                        let label = format!("{name}#{index} in mode {exits:?}:\n{before}");
                        if let Err(err) = passes.run(function) {
                            let err = format!("{err}");
                            assert!(
                                err.contains("does not support unreachable blocks"),
                                "{label}\nfails: {err}"
                            );
                            assert_eq!(format!("{}", function.borrow()), before, "{label}");
                            refused += 1;
                            continue;
                        }
                        lifted += 1;
                        if exits != Exits::InPlace {
                            continue;
                        }
                        let after = format!("{}", function.borrow());
                        let mut leftovers = alloc::vec::Vec::new();
                        function.raw_prewalk_all::<Forward, _>(|op: OperationRef| {
                            let op = op.borrow();
                            if (op.name().dialect().as_str() == "cf" && op.has_successors())
                                || op.is::<PendingExit>()
                            {
                                leftovers.push(format!("{}", op.name()));
                            }
                        });
                        assert!(leftovers.is_empty(), "{label}\nleaves {leftovers:?}:\n{after}");
                        // The `ub.unreachable` the pass leaves after the structured code of a body
                        // in which every path reaches a dead end.
                        let trailing = usize::from(
                            several_blocks
                                && function
                                    .borrow()
                                    .downcast_ref::<Function>()
                                    .and_then(|function| function.body().entry_block_ref())
                                    .and_then(|entry| entry.borrow().terminator())
                                    .is_some_and(|last| {
                                        last.borrow().is::<runic_dialect_ub::Unreachable>()
                                    }),
                        );
                        let (exits_after, dead_ends_after) = count_exits(function);
                        assert_eq!(exits_after, exits_before, "{label}\n{after}");
                        assert_eq!(
                            dead_ends_after,
                            dead_ends_before + trailing,
                            "{label}\n{after}"
                        );
                    }
                }
            }
        }
        assert!(lifted >= 60, "{lifted} lifted");
        assert!(refused >= 4, "{refused} refused");
    }

    /// [ControlFlowToSCFTransformation], except that the `fail_at`-th structured branch region
    /// operation it is asked for (counting from 1) is refused: the structuring fails part-way.
    struct FailingTransformation {
        fail_at: usize,
        calls: core::cell::Cell<usize>,
    }

    impl CFGToSCFInterface for FailingTransformation {
        fn create_structured_branch_region_op(
            &self,
            builder: &mut OpBuilder,
            control_flow_cond_op: OperationRef,
            result_types: &[runic_ir::Type],
            regions: &mut SmallVec<[RegionRef; 2]>,
        ) -> Result<OperationRef, Report> {
            self.calls.set(self.calls.get() + 1);
            if self.calls.get() == self.fail_at {
                return Err(Report::msg("the test refuses this branch"));
            }
            ControlFlowToSCFTransformation.create_structured_branch_region_op(
                builder,
                control_flow_cond_op,
                result_types,
                regions,
            )
        }

        fn create_structured_branch_region_terminator_op(
            &self,
            span: SourceSpan,
            builder: &mut OpBuilder,
            branch_region_op: OperationRef,
            replaced_control_flow_op: Option<OperationRef>,
            results: ValueRange<'_, 2>,
        ) -> Result<(), Report> {
            ControlFlowToSCFTransformation.create_structured_branch_region_terminator_op(
                span,
                builder,
                branch_region_op,
                replaced_control_flow_op,
                results,
            )
        }

        fn create_structured_do_while_loop_op(
            &self,
            builder: &mut OpBuilder,
            replaced_op: OperationRef,
            loop_values_init: ValueRange<'_, 2>,
            condition: ValueRef,
            loop_values_next_iter: ValueRange<'_, 2>,
            loop_body: RegionRef,
        ) -> Result<OperationRef, Report> {
            ControlFlowToSCFTransformation.create_structured_do_while_loop_op(
                builder,
                replaced_op,
                loop_values_init,
                condition,
                loop_values_next_iter,
                loop_body,
            )
        }

        fn get_cfg_bool_value(
            &self,
            span: SourceSpan,
            builder: &mut OpBuilder,
            value: bool,
        ) -> ValueRef {
            ControlFlowToSCFTransformation.get_cfg_bool_value(span, builder, value)
        }

        fn get_cfg_switch_value(
            &self,
            span: SourceSpan,
            builder: &mut OpBuilder,
            value: i32,
        ) -> ValueRef {
            ControlFlowToSCFTransformation.get_cfg_switch_value(span, builder, value)
        }

        fn create_cfg_switch_op(
            &self,
            span: SourceSpan,
            builder: &mut OpBuilder,
            flag: ValueRef,
            case_values: &[i32],
            case_destinations: &[BlockRef],
            case_arguments: &[ValueRange<'_, 2>],
            default_dest: BlockRef,
            default_args: ValueRange<'_, 2>,
        ) -> Result<(), Report> {
            ControlFlowToSCFTransformation.create_cfg_switch_op(
                span,
                builder,
                flag,
                case_values,
                case_destinations,
                case_arguments,
                default_dest,
                default_args,
            )
        }

        fn get_undef_value(
            &self,
            span: SourceSpan,
            builder: &mut OpBuilder,
            ty: runic_ir::Type,
        ) -> ValueRef {
            ControlFlowToSCFTransformation.get_undef_value(span, builder, ty)
        }

        fn create_unreachable_terminator(
            &self,
            span: SourceSpan,
            builder: &mut OpBuilder,
            region: RegionRef,
        ) -> Result<OperationRef, Report> {
            ControlFlowToSCFTransformation.create_unreachable_terminator(span, builder, region)
        }

        fn create_single_destination_branch(
            &self,
            span: SourceSpan,
            builder: &mut OpBuilder,
            dummy_flag: ValueRef,
            destination: BlockRef,
            arguments: ValueRange<'_, 2>,
        ) -> Result<(), Report> {
            ControlFlowToSCFTransformation.create_single_destination_branch(
                span,
                builder,
                dummy_flag,
                destination,
                arguments,
            )
        }

        fn create_conditional_branch(
            &self,
            span: SourceSpan,
            builder: &mut OpBuilder,
            condition: ValueRef,
            true_dest: BlockRef,
            true_args: ValueRange<'_, 2>,
            false_dest: BlockRef,
            false_args: ValueRange<'_, 2>,
        ) -> Result<(), Report> {
            ControlFlowToSCFTransformation.create_conditional_branch(
                span, builder, condition, true_dest, true_args, false_dest, false_args,
            )
        }
    }

    /// When the structuring of `exits=in-place` fails part-way, after the exits were taken out,
    /// the error of the structuring is returned, and the exits are put back where their
    /// placeholders are: no `scf.pending_exit` is left in the function. The rest of the IR is in
    /// the unspecified state the structuring left it in (the blocks it had moved into the regions
    /// of the refused operation are gone, with their exits), so nothing more is checked.
    #[test]
    fn a_structuring_that_fails_part_way_returns_its_error_and_leaves_no_placeholder() {
        use runic_ir::{
            dialects::builtin::Function,
            parse::{ParserConfig, parse},
        };

        let source = "builtin.function public extern(\"C\") @f(%c: i1, %d: i1, %a: i32, %b: i32) \
                      -> i32 {
    cf.cond_br %c ^x, ^y : (i1);
^x:
    cf.cond_br %d ^u, ^v : (i1);
^u:
    builtin.ret %a : (i32);
^v:
    builtin.ret %b : (i32);
^y:
    ub.unreachable;
};";
        for fail_at in [1, 2] {
            let context = Rc::new(Context::default());
            context.get_or_register_dialect::<crate::ScfDialect>();
            context.get_or_register_dialect::<cf::ControlFlowDialect>();
            context.get_or_register_dialect::<runic_dialect_ub::UndefinedBehaviorDialect>();
            let function = parse::<Function>(ParserConfig::new(context.clone()), "f", source)
                .into_result()
                .unwrap_or_else(|err| panic!("does not parse: {err:?}"));
            let function = function.as_operation_ref();
            let region =
                function.borrow().downcast_ref::<Function>().unwrap().body().as_region_ref();
            let mut dominfo = DominanceInfo::new(&function.borrow());
            let mut failing = FailingTransformation {
                fail_at,
                calls: core::cell::Cell::new(0),
            };

            let err = lift_region(region, &mut failing, &mut dominfo, true)
                .expect_err("the structuring fails");

            assert_eq!(format!("{err}"), "the test refuses this branch", "fail at {fail_at}");
            assert_eq!(failing.calls.get(), fail_at, "fail at {fail_at}");
            let mut placeholders = 0;
            function.raw_prewalk_all::<Forward, _>(|op: OperationRef| {
                if op.borrow().is::<PendingExit>() {
                    placeholders += 1;
                }
            });
            let after = format!("{}", function.borrow());
            assert_eq!(placeholders, 0, "fail at {fail_at}:\n{after}");
        }
    }

    /// The exits taken out of a region are put back exactly where they were: taking them out and
    /// restoring them without a check, as the refusal path does, with no structuring in between,
    /// leaves the function as it was, the `ub.unreachable` block the placeholders branched to
    /// erased.
    #[test]
    fn restoring_the_exits_without_a_check_leaves_the_function_as_it_was() {
        use runic_ir::{
            dialects::builtin::Function,
            parse::{ParserConfig, parse},
        };

        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<crate::ScfDialect>();
        context.get_or_register_dialect::<cf::ControlFlowDialect>();
        context.get_or_register_dialect::<runic_dialect_ub::UndefinedBehaviorDialect>();
        let function = parse::<Function>(
            ParserConfig::new(context.clone()),
            "f",
            "builtin.function public extern(\"C\") @f(%c: i1, %a: i32, %b: i32) -> i32 {
    cf.cond_br %c ^x(%a : i32), ^y : (i1);
^x(%v: i32):
    builtin.ret %v : (i32);
^y:
    builtin.tail_call @f(%c, %b, %a) : extern(\"C\") (i1, i32, i32) -> i32;
};",
        )
        .into_result()
        .unwrap_or_else(|err| panic!("does not parse: {err:?}"));
        let function = function.as_operation_ref();
        let before = format!("{}", function.borrow());
        let region = function.borrow().downcast_ref::<Function>().unwrap().body().as_region_ref();

        let pending = PendingExits::take(region).expect("two exits are taken out");
        let taken = format!("{}", function.borrow());
        assert!(taken.contains("scf.pending_exit"), "{taken}");
        pending.restore(false).unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(format!("{}", function.borrow()), before);
    }

    /// The option `exits` reads `merge` and `in-place`, and the pass prints itself with the
    /// option only when it is not the default.
    #[test]
    fn the_exits_option_reads_both_values_and_prints_only_in_place() -> Result<(), Report> {
        use alloc::string::ToString;

        struct Textual<'a>(&'a LiftControlFlowToSCF);
        impl fmt::Display for Textual<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.print_as_textual_pipeline(f)
            }
        }

        for (text, exits, printed) in [
            ("exits=merge", Exits::Merge, "cfg-to-scf"),
            ("exits=in-place", Exits::InPlace, "cfg-to-scf{exits=\"in-place\"}"),
            ("", Exits::Merge, "cfg-to-scf"),
        ] {
            let mut pass = LiftControlFlowToSCF::default();
            pass.initialize_options(&PassOptions::parse(text)?)?;
            assert_eq!(pass.exits(), exits, "{text}");
            assert_eq!(Textual(&pass).to_string(), printed, "{text}");
        }
        for text in ["exits=sideways", "exits", "exit=merge"] {
            let mut pass = LiftControlFlowToSCF::default();
            assert!(pass.initialize_options(&PassOptions::parse(text)?).is_err(), "{text}");
        }
        Ok(())
    }

    /// The result of a `builtin.unrealized_conversion_cast` of `value` to `ty`.
    fn cast(
        builder: &mut builtin::FunctionBuilder<'_, runic_ir::OpBuilder>,
        value: ValueRef,
        ty: runic_ir::Type,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let cast = builder.unrealized_conversion_cast([value], [ty], span)?;
        Ok(cast.borrow().outputs()[0] as ValueRef)
    }

    #[test]
    fn cfg_to_scf_combines_returns_from_distinct_inline_sites() -> Result<(), Report> {
        use runic_ir::dialects::debuginfo::attributes::{
            INLINE_CALL_CHAIN_ATTR_NAME, InlineCallChain, InlineCallChainAttr, InlineCallFrame,
        };

        for second_line in [None, Some(10), Some(20)] {
            let mut test = Test::new("inline_exits", |cx| ([cx.i1(), cx.i32()], [cx.i32()]));
            let (mut first, mut second) = {
                let mut builder = test.function_builder();
                let first_block = builder.create_block();
                let second_block = builder.create_block();
                let entry = builder.current_block();
                let condition = entry.borrow().arguments()[0].upcast();
                let value = entry.borrow().arguments()[1].upcast();
                builder.cond_br(
                    condition,
                    first_block,
                    [],
                    second_block,
                    [],
                    SourceSpan::UNKNOWN,
                )?;
                builder.switch_to_block(first_block);
                builder.ret([value], SourceSpan::UNKNOWN)?;
                builder.switch_to_block(second_block);
                builder.ret([value], SourceSpan::UNKNOWN)?;
                (
                    first_block.borrow().terminator().unwrap(),
                    second_block.borrow().terminator().unwrap(),
                )
            };
            let chain = |call_line| {
                test.context_rc()
                    .create_attribute::<InlineCallChainAttr, _>(InlineCallChain::new(alloc::vec![
                        InlineCallFrame {
                            name: "callee".into(),
                            linkage_name: None,
                            file: "callee.rs".into(),
                            line: 1,
                            column: 1,
                            call_file: "caller.rs".into(),
                            call_line,
                            call_column: 1,
                        }
                    ]))
                    .as_attribute_ref()
            };
            let first_chain = chain(10);
            first.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, first_chain);
            if let Some(line) = second_line {
                second.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, chain(line));
            }
            test.apply_pass::<LiftControlFlowToSCF>(true)?;
            let mut returns = alloc::vec::Vec::new();
            test.function()
                .as_operation_ref()
                .raw_prewalk_all::<Forward, _>(|op: OperationRef| {
                    if op.borrow().is::<builtin::Ret>() {
                        returns.push(op);
                    }
                });
            assert_eq!(returns.len(), 1);
            assert_eq!(
                returns[0].borrow().get_attribute(INLINE_CALL_CHAIN_ATTR_NAME),
                (second_line == Some(10)).then_some(first_chain),
            );
        }
        Ok(())
    }

    #[test]
    fn cfg_to_scf_lift_simple_conditional() -> Result<(), Report> {
        let mut test =
            Test::new("cfg_to_scf_lift_simple_conditional", |cx| ([cx.i32()], [cx.i32()]));

        let span = SourceSpan::default();

        let i32_ty = test.context().i32();

        // Define function body
        let mut builder = test.function_builder();

        let if_is_zero = builder.create_block();
        let if_is_nonzero = builder.create_block();
        let exit_block = builder.create_block();
        let return_val = builder.append_block_param(exit_block, i32_ty, span);

        let block = builder.current_block();
        let input = block.borrow().arguments()[0].upcast();

        let zero = builder.i32(0, span);
        let is_zero = builder.eq(input, zero, span)?;
        builder.cond_br(is_zero, if_is_zero, [], if_is_nonzero, [], span)?;

        builder.switch_to_block(if_is_zero);
        let a_one = builder.i32(1, span);
        let a = builder.add(input, a_one, span)?;
        builder.br(exit_block, [a], span)?;

        builder.switch_to_block(if_is_nonzero);
        let b = builder.mul(input, input, span)?;
        builder.br(exit_block, [b], span)?;

        builder.switch_to_block(exit_block);
        builder.ret(Some(return_val), span)?;

        let operation = test.function().as_operation_ref();

        // Run transformation on function body
        let test_name = test.name();
        let input = format!("{}", operation.borrow());
        let before_path = format!("expected/{test_name}_before.runic");
        expect_file![&before_path].assert_eq(&input);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // an `builtin.return`.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.runic");
        expect_file![&after_path].assert_eq(&output);

        Ok(())
    }

    /// This test ensures that the CF->SCF transformation is correctly applied to unstructured
    /// conditional control flow, where one branch leads to an early exit from the function, while
    /// the other branch performs additional computation before exiting.
    #[test]
    fn cfg_to_scf_lift_conditional_early_exit() -> Result<(), Report> {
        let mut test = Test::new("cfg_to_scf_lift_conditional_early_exit", |cx| {
            ([cx.i32(), cx.i32(), cx.i32(), cx.i32()], [cx.i32()])
        });

        let span = SourceSpan::default();

        let i32_ty = test.context().i32();

        // Define function body
        let mut builder = test.function_builder();

        // The shape this test lifts, a reduction of a reallocation routine: two of its paths call
        // out (here an increment of an argument stands in for each call), join at `^block34`,
        // and the join either returns or ends in `ub.unreachable`; a third path returns early.
        //
        //     ^block32(%v325, %v326, %v327, %v328):
        //         %v330 = arith.constant 0 : i32;
        //         %v331 = arith.cmpi ne %v326, %v330;
        //         cf.cond_br %v331 ^block35, ^block36;
        //     ^block34(%v343: i32):
        //         %v345 = arith.cmpi eq %v343, %v330;
        //         %v346 = arith.zext %v345 : i32;
        //         %v349 = arith.cmpi ne %v346, %v330;
        //         cf.cond_br %v349 ^block39, ^block40;
        //     ^block35:
        //         %v342 = (call with %v325, %v326, %v327, %v328);
        //         cf.br ^block34(%v342);
        //     ^block36:
        //         %v333 = arith.cmpi ne %v328, %v330;
        //         cf.cond_br %v333 ^block37, ^block38;
        //     ^block37:
        //         %v341 = (call with %v328, %v327);
        //         cf.br ^block34(%v341);
        //     ^block38:
        //         builtin.ret %v327;
        //     ^block39:
        //         ub.unreachable;
        //     ^block40:
        //         builtin.ret %v343;

        let block32 = builder.current_block();
        let block34 = builder.create_block();
        let v343 = builder.append_block_param(block34, i32_ty, span);
        let block35 = builder.create_block();
        let block36 = builder.create_block();
        let block37 = builder.create_block();
        let block38 = builder.create_block();
        let block39 = builder.create_block();
        let block40 = builder.create_block();

        let (v325, v326, v327, v328) = {
            let block32 = block32.borrow();
            let args = block32.arguments();
            let arg0: runic_ir::ValueRef = args[0].upcast();
            let arg2: runic_ir::ValueRef = args[2].upcast();
            let arg3: runic_ir::ValueRef = args[3].upcast();
            (arg0, args[1].upcast(), arg2, arg3)
        };

        let v330 = builder.i32(0, span);
        let v331 = builder.ne(v326, v330, span)?;
        builder.cond_br(v331, block35, [], block36, [], span)?;

        builder.switch_to_block(block34);
        let v345 = builder.eq(v343, v330, span)?;
        let v346 = builder.zext(v345, i32_ty, span)?;
        let v349 = builder.ne(v346, v330, span)?;
        builder.cond_br(v349, block39, [], block40, [], span)?;

        builder.switch_to_block(block35);
        let v342_one = builder.i32(1, span);
        let v342 = builder.add(v325, v342_one, span)?;
        builder.br(block34, [v342], span)?;

        builder.switch_to_block(block36);
        let v333 = builder.ne(v328, v330, span)?;
        builder.cond_br(v333, block37, [], block38, [], span)?;

        builder.switch_to_block(block37);
        let v341_one = builder.i32(1, span);
        let v341 = builder.add(v328, v341_one, span)?;
        builder.br(block34, [v341], span)?;

        builder.switch_to_block(block38);
        builder.ret(Some(v327), span)?;

        builder.switch_to_block(block39);
        builder.unreachable(span);

        builder.switch_to_block(block40);
        builder.ret(Some(v343), span)?;

        let operation = test.function().as_operation_ref();

        // Run transformation on function body
        let input = format!("{}", operation.borrow());
        let test_name = test.name();
        let before_path = format!("expected/{test_name}_before.runic");
        expect_file![&before_path].assert_eq(&input);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // a `cf.switch`, which branches to either a return, or an unreachable.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.runic");
        expect_file![&after_path].assert_eq(&output);

        Ok(())
    }

    #[test]
    fn cfg_to_scf_lift_simple_while_loop() -> Result<(), Report> {
        let mut test =
            Test::new("cfg_to_scf_lift_simple_while_loop", |cx| ([cx.i32()], [cx.i32()]));

        let span = SourceSpan::default();

        let i32_ty = test.context().i32();

        // Define function body
        let mut builder = test.function_builder();

        let loop_header = builder.create_block();
        let n = builder.append_block_param(loop_header, i32_ty, span);
        let counter = builder.append_block_param(loop_header, i32_ty, span);
        let if_is_zero = builder.create_block();
        let if_is_nonzero = builder.create_block();

        let block = builder.current_block();
        let input = block.borrow().arguments()[0].upcast();

        let zero = builder.i32(0, span);
        let one = builder.i32(1, span);
        builder.br(loop_header, [input, zero], span)?;

        builder.switch_to_block(loop_header);
        let is_zero = builder.eq(n, zero, span)?;
        builder.cond_br(is_zero, if_is_zero, [], if_is_nonzero, [], span)?;

        builder.switch_to_block(if_is_zero);
        builder.ret(Some(counter), span)?;

        builder.switch_to_block(if_is_nonzero);
        let n_prime = builder.sub_unchecked(n, one, Signedness::Unsigned, span)?;
        let counter_prime_one = builder.i32(1, span);
        let counter_prime = builder.add(counter, counter_prime_one, span)?;
        builder.br(loop_header, [n_prime, counter_prime], span)?;

        let operation = test.function().as_operation_ref();

        // Run transformation on function body
        let input = format!("{}", operation.borrow());
        let test_name = test.name();
        let before_path = format!("expected/{test_name}_before.runic");
        expect_file![&before_path].assert_eq(&input);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // an `builtin.return`.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.runic");
        expect_file![&after_path].assert_eq(&output);

        Ok(())
    }

    #[test]
    fn cfg_to_scf_lift_nested_while_loop() -> Result<(), Report> {
        let mut test = Test::new("cfg_to_scf_lift_nested_while_loop", |cx| {
            ([cx.ptr_type(cx.i32()), cx.i32(), cx.i32()], [cx.i32()])
        });

        let span = SourceSpan::default();

        // Define function body for the following pseudocode:
        //
        // function test(v0: *mut i32, rows: i32, cols: i32) -> i32 {
        //     let row_offset = 0;
        //     let sum = 0;
        //     while row_offset < rows {
        //         let offset = row_offset * cols;
        //         let col_offset = 0;
        //         while col_offset < cols {
        //             let cell = *(v0 + offset + col_offset);
        //             col_offset += 1;
        //             sum += cell;
        //         }
        //         row_offset += 1;
        //     }
        //
        //     return sum;
        // }
        //
        let i32_ty = test.context().i32();
        let ptr_ty = test.context().ptr_type(i32_ty);
        let mut builder = test.function_builder();

        let outer_loop_header = builder.create_block();
        let inner_loop_header = builder.create_block();
        let row_offset = builder.append_block_param(outer_loop_header, i32_ty, span);
        let row_sum = builder.append_block_param(outer_loop_header, i32_ty, span);
        let col_offset = builder.append_block_param(inner_loop_header, i32_ty, span);
        let col_sum = builder.append_block_param(inner_loop_header, i32_ty, span);
        let has_more_rows = builder.create_block();
        let no_more_rows = builder.create_block();
        let has_more_columns = builder.create_block();
        let no_more_columns = builder.create_block();

        let block = builder.current_block();
        let ptr = block.borrow().arguments()[0].upcast();
        let num_rows = block.borrow().arguments()[1].upcast();
        let num_cols = block.borrow().arguments()[2].upcast();

        let zero = builder.i32(0, span);
        builder.br(outer_loop_header, [zero, zero], span)?;

        builder.switch_to_block(outer_loop_header);
        let end_of_rows = builder.ult(row_offset, num_rows, span)?;
        builder.cond_br(end_of_rows, no_more_rows, [], has_more_rows, [], span)?;

        builder.switch_to_block(no_more_rows);
        builder.ret(Some(row_sum), span)?;

        builder.switch_to_block(has_more_rows);
        let offset = builder.mul_unchecked(row_offset, num_cols, Signedness::Unsigned, span)?;
        builder.br(inner_loop_header, [zero, row_sum], span)?;

        builder.switch_to_block(inner_loop_header);
        let end_of_cols = builder.ult(col_offset, num_cols, span)?;
        builder.cond_br(end_of_cols, no_more_columns, [], has_more_columns, [], span)?;

        builder.switch_to_block(no_more_columns);
        let new_row_offset_one = builder.i32(1, span);
        let new_row_offset = builder.add(row_offset, new_row_offset_one, span)?;
        builder.br(outer_loop_header, [new_row_offset, col_sum], span)?;

        builder.switch_to_block(has_more_columns);
        let addr_offset = builder.add_unchecked(offset, col_offset, Signedness::Unsigned, span)?;
        let addr = cast(&mut builder, ptr, i32_ty, span)?;
        let cell_addr = builder.add_unchecked(addr, addr_offset, Signedness::Unsigned, span)?;
        // This represents a bitcast
        let cell_ptr = cast(&mut builder, cell_addr, ptr_ty, span)?;
        // This represents a load
        let cell = cast(&mut builder, cell_ptr, i32_ty, span)?;
        let new_col_offset_one = builder.i32(1, span);
        let new_col_offset = builder.add(col_offset, new_col_offset_one, span)?;
        let new_sum = builder.add_unchecked(col_sum, cell, Signedness::Unsigned, span)?;
        builder.br(inner_loop_header, [new_col_offset, new_sum], span)?;

        let operation = test.function().as_operation_ref();

        // Run transformation on function body
        let input = format!("{}", operation.borrow());
        let test_name = test.name();
        let before_path = format!("expected/{test_name}_before.runic");
        expect_file![&before_path].assert_eq(&input);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // an `builtin.return`.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.runic");
        expect_file![&after_path].assert_eq(&output);

        Ok(())
    }

    #[test]
    fn cfg_to_scf_lift_multiple_exit_nested_while_loop() -> Result<(), Report> {
        let mut test = Test::new("cfg_to_scf_lift_multiple_exit_nested_while_loop", |cx| {
            ([cx.ptr_type(cx.i32()), cx.i32(), cx.i32()], [cx.i32()])
        });

        let span = SourceSpan::default();

        // Define function body for the following pseudocode:
        //
        // function test(v0: *mut i32, rows: i32, cols: i32) -> i32 {
        //     let row_offset = 0;
        //     let sum = 0;
        //     while row_offset < rows {
        //         let offset = row_offset * cols;
        //         let col_offset = 0;
        //         while col_offset < cols {
        //             let cell = *(v0 + offset + col_offset);
        //             col_offset += 1;
        //             let (sum_p, overflowed) = sum.add_overflowing(cell);
        //             if overflowed {
        //                 return -1;
        //             }
        //             sum += cell;
        //         }
        //         row_offset += 1;
        //     }
        //
        //     return sum;
        // }
        //
        let i32_ty = test.context().i32();
        let ptr_ty = test.context().ptr_type(i32_ty);
        let mut builder = test.function_builder();

        let outer_loop_header = builder.create_block();
        let inner_loop_header = builder.create_block();
        let row_offset = builder.append_block_param(outer_loop_header, i32_ty, span);
        let row_sum = builder.append_block_param(outer_loop_header, i32_ty, span);
        let col_offset = builder.append_block_param(inner_loop_header, i32_ty, span);
        let col_sum = builder.append_block_param(inner_loop_header, i32_ty, span);
        let has_more_rows = builder.create_block();
        let no_more_rows = builder.create_block();
        let has_more_columns = builder.create_block();
        let no_more_columns = builder.create_block();
        let has_overflowed = builder.create_block();

        let block = builder.current_block();
        let ptr = block.borrow().arguments()[0].upcast();
        let num_rows = block.borrow().arguments()[1].upcast();
        let num_cols = block.borrow().arguments()[2].upcast();

        let zero = builder.i32(0, span);
        builder.br(outer_loop_header, [zero, zero], span)?;

        builder.switch_to_block(outer_loop_header);
        let more_rows = builder.ult(row_offset, num_rows, span)?;
        builder.cond_br(more_rows, has_more_rows, [], no_more_rows, [], span)?;

        builder.switch_to_block(no_more_rows);
        builder.ret(Some(row_sum), span)?;

        builder.switch_to_block(has_more_rows);
        let offset = builder.mul_unchecked(row_offset, num_cols, Signedness::Unsigned, span)?;
        builder.br(inner_loop_header, [zero, row_sum], span)?;

        builder.switch_to_block(inner_loop_header);
        let more_cols = builder.ult(col_offset, num_cols, span)?;
        builder.cond_br(more_cols, has_more_columns, [], no_more_columns, [], span)?;

        builder.switch_to_block(no_more_columns);
        let new_row_offset_one = builder.i32(1, span);
        let new_row_offset = builder.add(row_offset, new_row_offset_one, span)?;
        builder.br(outer_loop_header, [new_row_offset, col_sum], span)?;

        builder.switch_to_block(has_more_columns);
        let addr_offset = builder.add_unchecked(offset, col_offset, Signedness::Unsigned, span)?;
        let addr = cast(&mut builder, ptr, i32_ty, span)?;
        let cell_addr = builder.add_unchecked(addr, addr_offset, Signedness::Unsigned, span)?;
        // This represents a bitcast
        let cell_ptr = cast(&mut builder, cell_addr, ptr_ty, span)?;
        // This represents a load
        let cell = cast(&mut builder, cell_ptr, i32_ty, span)?;
        let new_col_offset_one = builder.i32(1, span);
        let new_col_offset = builder.add(col_offset, new_col_offset_one, span)?;
        let (overflowed, new_sum) =
            builder.add_overflowing(col_sum, cell, Signedness::Unsigned, span)?;
        builder.cond_br(
            overflowed,
            has_overflowed,
            [],
            inner_loop_header,
            [new_col_offset, new_sum],
            span,
        )?;

        builder.switch_to_block(has_overflowed);
        builder.ret_imm(runic_ir::Int::from(-1i32), span)?;

        let operation = test.function().as_operation_ref();

        // Run transformation on function body
        let input = format!("{}", operation.borrow());
        let test_name = test.name();
        let before_path = format!("expected/{test_name}_before.runic");
        expect_file![&before_path].assert_eq(&input);

        test.apply_passes(
            [Box::<LiftControlFlowToSCF>::default(), transforms::Canonicalizer::create()],
            true,
        )?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // an `builtin.return`.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.runic");
        expect_file![&after_path].assert_eq(&output);

        Ok(())
    }

    /// This test verifies that `debuginfo.debug_value` operations are preserved through the
    /// CF-to-SCF transformation. The key behavior being tested is that `replace_all_uses_with`
    /// (used internally by the transform to replace block arguments with `scf.if` results)
    /// automatically updates the SSA operands of debug value ops.
    #[test]
    fn cfg_to_scf_debug_value_preservation() -> Result<(), Report> {
        use runic_ir::{
            dialects::debuginfo::{DIBuilder, DebugInfoDialect, attributes::Variable},
            interner::Symbol,
        };

        let mut test =
            Test::new("cfg_to_scf_debug_value_preservation", |cx| ([cx.i32()], [cx.i32()]));
        test.context().get_or_register_dialect::<DebugInfoDialect>();

        let span = SourceSpan::default();
        let i32_ty = test.context().i32();
        let mut builder = test.function_builder();

        let if_is_zero = builder.create_block();
        let if_is_nonzero = builder.create_block();
        let exit_block = builder.create_block();
        let return_val = builder.append_block_param(exit_block, i32_ty, span);

        let block = builder.current_block();
        let input = block.borrow().arguments()[0].upcast();

        let input_var =
            Variable::new(Symbol::intern("input"), Symbol::intern("test.rs"), 1, Some(1));
        let result_var =
            Variable::new(Symbol::intern("result"), Symbol::intern("test.rs"), 2, Some(1));

        let zero = builder.i32(0, span);
        let is_zero = builder.eq(input, zero, span)?;
        builder.builder_mut().debug_value(input, input_var.clone(), span)?;
        builder.cond_br(is_zero, if_is_zero, [], if_is_nonzero, [], span)?;

        builder.switch_to_block(if_is_zero);
        let a_one = builder.i32(1, span);
        let a = builder.add(input, a_one, span)?;
        builder.builder_mut().debug_value(a, result_var.clone(), span)?;
        builder.br(exit_block, [a], span)?;

        builder.switch_to_block(if_is_nonzero);
        let b = builder.mul(input, input, span)?;
        builder.builder_mut().debug_value(b, result_var.clone(), span)?;
        builder.br(exit_block, [b], span)?;

        builder.switch_to_block(exit_block);
        // KEY: this debug_value uses the block argument `return_val`, which will be
        // replaced by the scf.if result via replace_all_uses_with
        builder.builder_mut().debug_value(return_val, result_var.clone(), span)?;
        builder.ret(Some(return_val), span)?;

        let operation = test.function().as_operation_ref();

        let input_ir = format!("{}", operation.borrow());
        expect_file!["expected/cfg_to_scf_debug_value_preservation_before.runic"]
            .assert_eq(&input_ir);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        let output = format!("{}", operation.borrow());
        expect_file!["expected/cfg_to_scf_debug_value_preservation_after.runic"].assert_eq(&output);

        Ok(())
    }
}
