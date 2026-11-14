use alloc::rc::Rc;

use runic_core::{
    Builder, EntityMut, Forward, Op, Operation, OperationName, OperationRef, RawWalk, Report,
    SmallVec, Spanned, ValueRange, ValueRef, WalkResult,
    dialects::builtin,
    dominance::DominanceInfo,
    pass::{Pass, PassExecutionState, PostPassStatus},
};
use runic_dialect_arith::ArithOpBuilder;
use runic_dialect_cf::{self as cf, ControlFlowOpBuilder};
use runic_dialect_ub::UndefinedBehaviorOpBuilder;
use runic_transform::{self as transforms, CFGToSCFInterface};

use crate::*;

/// Lifts unstructured control flow operations (`cf`) to structured control flow operations (`scf`).
///
/// This pass is not always guaranteed to replace all unstructured control flow operations. If a
/// region contains only a single kind of return-like operation, all unstructured control flow ops
/// will be replaced successfully. Otherwise a single unstructured switch branching to one block per
/// return-like operation kind remains.
///
/// This pass may need to create unreachable terminators in case of infinite loops, which is only
/// supported for 'builtin.func' for now. If you potentially have infinite loops inside CFG regions
/// not belonging to 'builtin.func', consider using the `transform_cfg_to_scf` function directly
/// with a corresponding [CFGToSCFInterface::create_unreachable_terminator] implementation.
#[derive(Default)]
pub struct LiftControlFlowToSCF;

runic_core::inventory::submit!(
    ::runic_core::pass::registry::PassInfo::new::<LiftControlFlowToSCF>("cfg-to-scf", DESCRIPTION)
);

const DESCRIPTION: &str =
    "Replaces unstructured control flow (cf) by structured control flow (scf) where it can";

impl Pass for LiftControlFlowToSCF {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "lift-control-flow"
    }

    fn argument(&self) -> &'static str {
        "cfg-to-scf"
    }

    fn description(&self) -> &'static str {
        DESCRIPTION
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn initialize(&mut self, context: Rc<runic_core::Context>) -> Result<(), Report> {
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

        let root = op.as_operation_ref();
        drop(op);

        log::debug!(target: "cfg-to-scf", "applying control flow lifting transformation pass starting from {}", root.borrow());

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

                        let result =
                            transforms::transform_cfg_to_scf(region, &mut transformation, dominfo);
                        match result {
                            Ok(did_change) => {
                                log::trace!(
                                    target: "cfg-to-scf",
                                    "control flow lifting completed for region \
                                     (did_change={did_change})"
                                );
                                changed |= did_change;
                            }
                            Err(err) => {
                                return WalkResult::Break(err);
                            }
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

/// Implementation of [CFGToSCFInterface] used to lift unstructured control flow operations into
/// the structured control flow operations of this dialect.
struct ControlFlowToSCFTransformation;

impl CFGToSCFInterface for ControlFlowToSCFTransformation {
    /// Creates an `scf.if` op if `control_flow_cond_op` is a `cf.cond_br` op, or an
    /// `scf.index_switch` if it is a `cf.switch`. Otherwise, returns an error.
    fn create_structured_branch_region_op(
        &self,
        builder: &mut runic_core::OpBuilder,
        control_flow_cond_op: runic_core::OperationRef,
        result_types: &[runic_core::Type],
        regions: &mut runic_core::SmallVec<[runic_core::RegionRef; 2]>,
    ) -> Result<runic_core::OperationRef, runic_core::Report> {
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

        Err(runic_core::diagnostics::report!(
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
        span: runic_core::SourceSpan,
        builder: &mut runic_core::OpBuilder,
        _branch_region_op: runic_core::OperationRef,
        _replaced_control_flow_op: Option<runic_core::OperationRef>,
        results: ValueRange<'_, 2>,
    ) -> Result<(), runic_core::Report> {
        builder.r#yield(results, span)?;

        Ok(())
    }

    /// Creates an `scf.while` op. The loop body is made the before-region of the
    /// while op and terminated with an `scf.condition` op. The after-region does
    /// nothing but forward the iteration variables.
    fn create_structured_do_while_loop_op(
        &self,
        builder: &mut runic_core::OpBuilder,
        replaced_op: runic_core::OperationRef,
        loop_values_init: ValueRange<'_, 2>,
        condition: runic_core::ValueRef,
        loop_values_next_iter: ValueRange<'_, 2>,
        loop_body: runic_core::RegionRef,
    ) -> Result<runic_core::OperationRef, runic_core::Report> {
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
        span: runic_core::SourceSpan,
        builder: &mut runic_core::OpBuilder,
        value: bool,
    ) -> runic_core::ValueRef {
        builder.i1(value, span)
    }

    /// Creates an `arith.constant` with an i32 attribute of the given value.
    fn get_cfg_switch_value(
        &self,
        span: runic_core::SourceSpan,
        builder: &mut runic_core::OpBuilder,
        value: i32,
    ) -> runic_core::ValueRef {
        builder.i32(value, span)
    }

    /// Creates a `cf.switch` op with the given cases and flag.
    fn create_cfg_switch_op(
        &self,
        span: runic_core::SourceSpan,
        builder: &mut runic_core::OpBuilder,
        flag: runic_core::ValueRef,
        case_values: &[i32],
        case_destinations: &[runic_core::BlockRef],
        case_arguments: &[ValueRange<'_, 2>],
        default_dest: runic_core::BlockRef,
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
        span: runic_core::SourceSpan,
        builder: &mut runic_core::OpBuilder,
        _dummy_flag: runic_core::ValueRef,
        destination: runic_core::BlockRef,
        arguments: ValueRange<'_, 2>,
    ) -> Result<(), Report> {
        builder.br(destination, arguments, span)?;
        Ok(())
    }

    /// Branches on the `i1` produced by [Self::get_cfg_bool_value] with `cf.cond_br`.
    fn create_conditional_branch(
        &self,
        span: runic_core::SourceSpan,
        builder: &mut runic_core::OpBuilder,
        condition: runic_core::ValueRef,
        true_dest: runic_core::BlockRef,
        true_args: ValueRange<'_, 2>,
        false_dest: runic_core::BlockRef,
        false_args: ValueRange<'_, 2>,
    ) -> Result<(), Report> {
        builder.cond_br(condition, true_dest, true_args, false_dest, false_args, span)?;
        Ok(())
    }

    /// Creates a `ub.poison` op of the given type.
    fn get_undef_value(
        &self,
        span: runic_core::SourceSpan,
        builder: &mut runic_core::OpBuilder,
        ty: runic_core::Type,
    ) -> runic_core::ValueRef {
        builder.poison(ty, span)
    }

    fn create_unreachable_terminator(
        &self,
        span: runic_core::SourceSpan,
        builder: &mut runic_core::OpBuilder,
        _region: runic_core::RegionRef,
    ) -> Result<runic_core::OperationRef, runic_core::Report> {
        log::trace!(target: "cfg-to-scf", "creating unreachable terminator at {}", builder.insertion_point());
        let op = builder.unreachable(span);
        Ok(op.as_operation_ref())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, format};

    use builtin::BuiltinOpBuilder;
    use runic_core::{
        Report, Signedness, SourceSpan,
        dialects::builtin::{self},
        testing::Test,
    };
    use runic_expect_test::expect_file;

    use super::*;

    /// The result of a `builtin.unrealized_conversion_cast` of `value` to `ty`.
    fn cast(
        builder: &mut builtin::FunctionBuilder<'_, runic_core::OpBuilder>,
        value: ValueRef,
        ty: runic_core::Type,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let cast = builder.unrealized_conversion_cast([value], [ty], span)?;
        Ok(cast.borrow().outputs()[0] as ValueRef)
    }

    #[test]
    fn cfg_to_scf_combines_returns_from_distinct_inline_sites() -> Result<(), Report> {
        use runic_core::dialects::debuginfo::attributes::{
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
        let before_path = format!("expected/{test_name}_before.hir");
        expect_file![&before_path].assert_eq(&input);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // an `builtin.return`.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.hir");
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
            let arg0: runic_core::ValueRef = args[0].upcast();
            let arg2: runic_core::ValueRef = args[2].upcast();
            let arg3: runic_core::ValueRef = args[3].upcast();
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
        let before_path = format!("expected/{test_name}_before.hir");
        expect_file![&before_path].assert_eq(&input);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // a `cf.switch`, which branches to either a return, or an unreachable.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.hir");
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
        let before_path = format!("expected/{test_name}_before.hir");
        expect_file![&before_path].assert_eq(&input);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // an `builtin.return`.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.hir");
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
        let before_path = format!("expected/{test_name}_before.hir");
        expect_file![&before_path].assert_eq(&input);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // an `builtin.return`.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.hir");
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
        builder.ret_imm(runic_core::Int::from(-1i32), span)?;

        let operation = test.function().as_operation_ref();

        // Run transformation on function body
        let input = format!("{}", operation.borrow());
        let test_name = test.name();
        let before_path = format!("expected/{test_name}_before.hir");
        expect_file![&before_path].assert_eq(&input);

        test.apply_passes(
            [Box::new(LiftControlFlowToSCF), transforms::Canonicalizer::create()],
            true,
        )?;

        // Verify that the function body now consists of a single `scf.if` operation, followed by
        // an `builtin.return`.
        let output = format!("{}", operation.borrow());
        let after_path = format!("expected/{test_name}_after.hir");
        expect_file![&after_path].assert_eq(&output);

        Ok(())
    }

    /// This test verifies that `debuginfo.debug_value` operations are preserved through the
    /// CF-to-SCF transformation. The key behavior being tested is that `replace_all_uses_with`
    /// (used internally by the transform to replace block arguments with `scf.if` results)
    /// automatically updates the SSA operands of debug value ops.
    #[test]
    fn cfg_to_scf_debug_value_preservation() -> Result<(), Report> {
        use runic_core::{
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
        expect_file!["expected/cfg_to_scf_debug_value_preservation_before.hir"]
            .assert_eq(&input_ir);

        test.apply_pass::<LiftControlFlowToSCF>(true)?;

        let output = format!("{}", operation.borrow());
        expect_file!["expected/cfg_to_scf_debug_value_preservation_after.hir"].assert_eq(&output);

        Ok(())
    }
}
