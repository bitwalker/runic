use runic_core::{
    Backward, CallOpInterface, EntityMut, Forward, FxHashMap, FxHashSet, Op, Operation,
    OperationName, OperationRef, ProgramPoint, RawWalk, RegionBranchOpInterface, Report, Rewriter,
    SmallVec, Spanned, Symbol, ValueRef,
    dialects::{
        builtin::{Function, LoadLocal, StoreLocal, attributes::LocalVariableAttr},
        debuginfo::{
            DIBuilder, DebugDeclare, DebugValue,
            attributes::{ExpressionOp, FrameBase},
        },
    },
    effects::MemoryEffectOpInterface,
    pass::{Pass, PassExecutionState, PostPassStatus},
    patterns::{RewriterImpl, TracingRewriterListener},
    traits::BranchOpInterface,
};

/// The `log` target of this pass.
const TARGET: &str = "local2reg";

/// The value `store` writes, read from the operation: a promotion earlier in the same run may have
/// replaced it, and the operation, not a copy taken during the scan, holds the current one.
fn stored_value(store: OperationRef) -> ValueRef {
    store
        .borrow()
        .downcast_ref::<StoreLocal>()
        .expect("only `builtin.store_local` is recorded")
        .value()
        .as_value_ref()
}

/// The index, within its function, of the local `operation` names when it is a
/// `builtin.load_local` or a `builtin.store_local`.
///
/// Locals are keyed by index: a local built through the function is bound to it, and one parsed
/// from text is not, so the two do not compare equal even when they name the same slot.
fn local_index(attr: &LocalVariableAttr) -> usize {
    attr.as_value().as_usize()
}

/// Adds to `pinned` the index of every local that `operation` names: in a property or attribute
/// that is a [LocalVariableAttr], or as the attribute of one of its memory effects.
///
/// An operation may have unknown effects, so its effects alone do not show every operation that
/// uses a local; one that holds a local in a property or attribute is taken to use it.
fn pin_locals_named_by(operation: &Operation, pinned: &mut FxHashSet<usize>) {
    let mut pin = |attr: runic_core::AttributeRef| {
        if let Some(local) = attr.borrow().downcast_ref::<LocalVariableAttr>() {
            pinned.insert(local_index(local));
        }
    };
    for property in operation.properties() {
        pin(property.value);
    }
    for attribute in operation.attributes() {
        pin(attribute.as_named_attribute().value);
    }
    // The effects of an operation whose effects are unknown are not read: such an operation pins
    // through its properties and attributes, above
    if let Some(effects) = operation.as_trait::<dyn MemoryEffectOpInterface>()
        && effects.effects_are_known()
    {
        for effect in effects.effects() {
            if let Some(attribute) = effect.attribute() {
                pin(attribute);
            }
        }
    }
}

/// Rewrite debug references to the local at `local_index` so they follow the promoted SSA values.
///
/// When a local is promoted to an SSA value (or its dead stores are erased), the slot the debug
/// operation points at is no longer written, so a debugger following it would read stale or
/// uninitialized data. Declarations become value records at each store, while existing value
/// records retain their SSA operand and drop the obsolete local-location expression, as LLVM's
/// mem2reg converts `dbg.declare` and `dbg.value`.
///
/// Returns `false` when any debug reference to this local cannot be converted safely (its
/// expression is not exactly `[LocalSlot(i)]`). In that case the caller must preserve the stores.
fn convert_debug_references_for_local<R: Rewriter>(
    rewriter: &mut R,
    function_op: OperationRef,
    local_index: usize,
    stores: &[OperationRef],
) -> bool {
    let local_index = local_index as u32;
    let mut declares = SmallVec::<[OperationRef; 2]>::new();
    let mut values = SmallVec::<[OperationRef; 2]>::new();
    function_op.raw_prewalk_all::<Forward, _>(|op: OperationRef| {
        let operation = op.borrow();
        let expression = if let Some(declare) = operation.downcast_ref::<DebugDeclare>() {
            Some((declare.expression(), &mut declares))
        } else {
            operation
                .downcast_ref::<DebugValue>()
                .map(|value| (value.expression(), &mut values))
        };
        if let Some((expr, references)) = expression
            && expr.as_value().operations.iter().any(|op| match op {
                ExpressionOp::LocalSlot(index) => *index == local_index,
                ExpressionOp::FrameBase {
                    base: FrameBase::LocalSlot(index),
                    ..
                } => *index == local_index,
                _ => false,
            })
        {
            references.push(op);
        }
    });

    if declares.is_empty() && values.is_empty() {
        return true;
    }

    let declares_are_safe = declares.iter().all(|declare_op| {
        matches!(
            declare_op
                .borrow()
                .downcast_ref::<DebugDeclare>()
                .unwrap()
                .expression()
                .as_value()
                .operations
                .as_slice(),
            [ExpressionOp::LocalSlot(index)] if *index == local_index
        )
    });
    let values_are_safe = values.iter().all(|value_op| {
        matches!(
            value_op
                .borrow()
                .downcast_ref::<DebugValue>()
                .unwrap()
                .expression()
                .as_value()
                .operations
                .as_slice(),
            [ExpressionOp::LocalSlot(index)] if *index == local_index
        )
    });
    if !declares_are_safe || !values_are_safe {
        return false;
    }

    for declare_op in declares {
        let variable = {
            let op = declare_op.borrow();
            op.downcast_ref::<DebugDeclare>().unwrap().variable().as_value().clone()
        };
        for store in stores {
            let span = store.borrow().span();
            rewriter.set_insertion_point_before(*store);
            let _ = rewriter.debug_value(stored_value(*store), variable.clone(), span);
        }
        rewriter.erase_op(declare_op);
    }

    for value_op in values {
        let (value, variable, span) = {
            let op = value_op.borrow();
            let debug_value = op.downcast_ref::<DebugValue>().unwrap();
            (
                debug_value.value().as_value_ref(),
                debug_value.variable().as_value().clone(),
                op.span(),
            )
        };
        rewriter.set_insertion_point_before(value_op);
        let _ = rewriter.debug_value(value, variable, span);
        rewriter.erase_op(value_op);
    }

    true
}

/// Promotes the local variables of a function that are stored once and loaded once to SSA values,
/// and erases the stores of locals that are never loaded.
///
/// A local is promoted when it has exactly one `builtin.store_local` and exactly one
/// `builtin.load_local`, in one block, the store before the load, and no branch, region-branching
/// operation or call between them: the load's result is replaced by the stored value, and both are
/// erased. Locals are visited in index order, so a chain of locals, each storing the load of the
/// one before, collapses to the first stored value. Every other local is left alone: several loads
/// or stores, a load before its store, a load with no store, a load and store in different blocks.
/// This is not mem2reg: no block arguments are introduced.
///
/// A local that any operation other than its loads and stores names, in a property or attribute,
/// or as the attribute of one of its effects, is never promoted, and its stores are kept.
///
/// Debug information follows a promoted local: a `di.debug_declare` of its slot becomes a
/// `di.debug_value` of each stored value, and a `di.debug_value` that names the slot drops it. When
/// a debug expression does more with the slot than name it (`[LocalSlot(i)]`), the local is left
/// alone.
#[derive(Default)]
pub struct Local2Reg;

runic_core::inventory::submit!(::runic_core::pass::registry::PassInfo::new::<Local2Reg>(
    "local2reg",
    "promote single-use locals to SSA registers"
));

impl Pass for Local2Reg {
    type Target = Function;

    fn name(&self) -> &'static str {
        "local2reg"
    }

    fn argument(&self) -> &'static str {
        "local2reg"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let function = op.into_entity_ref();
        let name = Symbol::name(&*function);

        log::debug!(target: TARGET, "looking for locals to promote in @{name}");

        if function.is_declaration() {
            log::debug!(target: TARGET, "@{name} has no body, nothing to do");
            state.preserved_analyses_mut().preserve_all();
            state.set_post_pass_status(PostPassStatus::Unchanged);
            return Ok(());
        }

        let op = function.as_operation_ref();
        let context = function.as_operation().context_rc();
        drop(function);

        let mut rewriter = RewriterImpl::<TracingRewriterListener>::new(context)
            .with_listener(TracingRewriterListener);

        let mut loaded = FxHashMap::<usize, SmallVec<[OperationRef; 2]>>::default();
        let mut stored = FxHashMap::<usize, SmallVec<[OperationRef; 2]>>::default();
        let mut pinned = FxHashSet::default();
        let function_op = op;
        op.raw_postwalk_all::<Backward, _>(|op: OperationRef| {
            // The function's own properties describe its locals; they do not use them
            if op == function_op {
                return;
            }
            let operation = op.borrow();
            if let Some(load) = operation.downcast_ref::<LoadLocal>() {
                let local = local_index(&load.local());
                log::trace!(
                    target: TARGET,
                    "found load from local {local} @ {}",
                    ProgramPoint::before(op),
                );
                loaded.entry(local).or_default().push(op);
            } else if let Some(store) = operation.downcast_ref::<StoreLocal>() {
                let value = store.value().as_value_ref();
                let local = local_index(&store.local());
                log::trace!(
                    target: TARGET,
                    "found store of {value} to local {local} @ {}",
                    ProgramPoint::before(op),
                );
                stored.entry(local).or_default().push(op);
            } else {
                pin_locals_named_by(&operation, &mut pinned);
            }
        });

        // The candidates are the locals that are loaded or stored: a local that is neither has
        // nothing to rewrite. Sorted by index, so rewrites are applied in a deterministic order.
        let mut locals =
            SmallVec::<[usize; 4]>::from_iter(loaded.keys().chain(stored.keys()).copied());
        locals.sort_unstable();
        locals.dedup();
        // An operation other than a load or a store of the local names it: it may read or write
        // the slot in a way the loads and stores do not show.
        locals.retain(|local| !pinned.contains(local));

        if locals.is_empty() {
            log::debug!(target: TARGET, "@{name} has no promotable locals in use, nothing to do");
            state.preserved_analyses_mut().preserve_all();
            state.set_post_pass_status(PostPassStatus::Unchanged);
            return Ok(());
        }

        let mut changed = PostPassStatus::Unchanged;
        'next_local: for local in locals {
            if let Some(loads) = loaded.get(&local) {
                if loads.len() > 1 {
                    log::trace!(target: TARGET, "ignoring local {local}: loaded more than once");
                    continue;
                }

                // A local loaded once: look for its store in the same block, with no control flow
                // between the two. A load with no store reads an uninitialized local, which is
                // left alone.
                let Some(stores) = stored.get(&local) else {
                    log::trace!(target: TARGET, "ignoring local {local}: never stored");
                    continue;
                };

                // A local stored more than once is not promoted
                if stores.len() > 1 {
                    log::trace!(target: TARGET, "ignoring local {local}: stored more than once");
                    continue;
                }

                // One load of a local stored once is promoted when:
                //
                // 1. The load and store are in the same block
                // 2. The store precedes the load
                // 3. There is no control flow between the store and load, including calls
                let load = loads[0];
                let store = *stores.last().unwrap();

                // 1.
                if load.parent() != store.parent() {
                    log::trace!(
                        target: TARGET,
                        "ignoring local {local}: load and store are in different blocks",
                    );
                    continue;
                }

                // 2. A load that precedes the store reads what the slot held before the store,
                // not the stored value
                if !store.borrow().is_before_in_block(&load) {
                    log::trace!(target: TARGET, "ignoring local {local}: loaded before it is stored");
                    continue;
                }

                // 3.
                let mut next_op = store.next();
                while let Some(current) = next_op.take() {
                    if current == load {
                        break;
                    }

                    next_op = current.next();

                    let curr = current.borrow();
                    if curr.implements::<dyn BranchOpInterface>()
                        || curr.implements::<dyn RegionBranchOpInterface>()
                        || curr.implements::<dyn CallOpInterface>()
                    {
                        log::trace!(
                            target: TARGET,
                            "ignoring local {local}: found control flow between store and load",
                        );
                        continue 'next_local;
                    }
                }

                // Any debug declaration pointing at the promoted slot must follow the value into
                // its SSA register, since the slot will no longer be written.
                if !convert_debug_references_for_local(&mut rewriter, op, local, &[store]) {
                    log::trace!(
                        target: TARGET,
                        "ignoring local {local}: debug references cannot all be converted safely",
                    );
                    continue;
                }
                log::trace!(
                    target: TARGET,
                    "promoting local {local}: erasing the store and replacing the load with the \
                     stored value",
                );
                // Read before erasing: erasing an operation clears its operands
                let value = stored_value(store);
                rewriter.erase_op(store);
                rewriter.replace_all_op_uses_with_values(load, &[Some(value)]);
                rewriter.erase_op(load);
                changed = PostPassStatus::Changed;
            } else if let Some(stores) = stored.get(&local) {
                // A local that is stored but never loaded: its stores are dead. What they stored
                // may now be dead too; other passes remove it.
                if !convert_debug_references_for_local(&mut rewriter, op, local, stores) {
                    log::trace!(
                        target: TARGET,
                        "preserving dead stores of local {local}: debug references cannot all be \
                         converted safely",
                    );
                    continue;
                }
                for store in stores.iter() {
                    changed = PostPassStatus::Changed;
                    log::trace!(target: TARGET, "erasing dead store of local {local}");
                    rewriter.erase_op(*store);
                }
            }
        }

        state.set_post_pass_status(changed);
        if !changed.ir_changed() {
            state.preserved_analyses_mut().preserve_all();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        format,
        string::{String, ToString},
        vec::Vec,
    };

    use litcheck_filecheck::{filecheck, litcheck};
    use runic_core::{
        BuilderExt, Op, SourceSpan, ValueRef,
        dialects::{
            builtin::BuiltinOpBuilder,
            debuginfo::{
                DIBuilder, DebugInfoDialect,
                attributes::{Expression, ExpressionOp, Variable},
            },
        },
        interner::Symbol,
        testing::Test,
    };
    use runic_dialect_arith::ArithOpBuilder;

    use super::Local2Reg;

    /// The test's function, printed.
    fn print(test: &Test) -> String {
        format!("{}", test.function().borrow().as_operation())
    }

    /// A local named by another operation's effect (here a `Free` of it, by `test.touch_local`)
    /// keeps its store, and is not promoted.
    #[test]
    fn preserves_stores_of_a_local_named_by_another_operation() {
        let mut test = Test::new("named_elsewhere", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();
        let store;
        let load;
        {
            let mut builder = test.function_builder();
            let local = builder.alloc_local(i32);
            let input = builder.entry_block().borrow().arguments()[0] as ValueRef;
            store = builder.store_local(local, input, SourceSpan::UNKNOWN).unwrap();
            BuiltinOpBuilder::builder_mut(&mut builder)
                .create::<runic_core::dialects::test::TouchLocal, _>(SourceSpan::UNKNOWN)(
                local
            )
            .unwrap();
            let loaded = builder.load_local(local, SourceSpan::UNKNOWN).unwrap();
            load = loaded.borrow().get_defining_op().unwrap();
            builder.ret([loaded], SourceSpan::UNKNOWN).unwrap();
        }
        test.apply_pass::<Local2Reg>(true).unwrap();
        assert!(
            store.as_operation_ref().parent().is_some(),
            "an operation that names the local keeps the store live"
        );
        assert!(load.parent().is_some(), "and the load is not promoted");
    }

    #[test]
    fn promotes_redundant_load_store_pairs() {
        let mut test = Test::new("promotes_redundant", |cx| ([cx.i32(), cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let local1 = builder.alloc_local(i32);
            let [v0, v1] = *builder.entry_block().borrow().arguments()[0..2].as_array().unwrap();
            let v0 = v0 as ValueRef;
            let v1 = v1 as ValueRef;
            builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local1, v1, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.load_local(local1, SourceSpan::UNKNOWN).unwrap();
            let v4 = builder.add(v2, v3, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v4], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        std::println!("{output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @promotes_redundant
// CHECK-NEXT: [[V4:%\d+]] = arith.add %0, %1;
// CHECK-NEXT: builtin.ret [[V4]] : (i32);
            "#
        );
    }

    #[test]
    fn promotes_a_chain_of_locals_to_the_first_stored_value() {
        // Each chain lists, in program order, the slots the entry argument passes through: it is
        // stored to the first slot, loaded, the load stored to the next slot, and so on. Locals
        // are promoted in index order, so in every order but the descending one, a slot is
        // promoted after the slot whose load it stores: by then its store holds that load's
        // replacement, and that is the value the promotion forwards.
        for chain in [&[1, 0][..], &[0, 1], &[0, 2, 1]] {
            let mut test = Test::new("promotes_chain", |cx| ([cx.i32()], [cx.i32()]));
            let i32 = test.context().i32();
            {
                let mut builder = test.function_builder();
                let locals = chain.iter().map(|_| builder.alloc_local(i32)).collect::<Vec<_>>();
                let mut value = builder.entry_block().borrow().arguments()[0] as ValueRef;
                for &slot in chain {
                    builder.store_local(locals[slot], value, SourceSpan::UNKNOWN).unwrap();
                    value = builder.load_local(locals[slot], SourceSpan::UNKNOWN).unwrap();
                }
                builder.ret([value], SourceSpan::UNKNOWN).unwrap();
            }

            test.apply_pass::<Local2Reg>(true).expect("invalid ir");

            let output = print(&test);
            std::println!("chain {chain:?}: {output}");
            filecheck!(
                output,
                r#"
// CHECK-LABEL: builtin.function public extern("C") @promotes_chain
// CHECK-NEXT: builtin.ret %0 : (i32);
                "#
            );
        }

        // One load stored to two slots, each loaded once
        let mut test = Test::new("promotes_fan_out", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();
        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let local1 = builder.alloc_local(i32);
            let local2 = builder.alloc_local(i32);
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();
            let v1 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local1, v1, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local2, v1, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.load_local(local1, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.load_local(local2, SourceSpan::UNKNOWN).unwrap();
            let v4 = builder.add(v2, v3, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v4], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        std::println!("{output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @promotes_fan_out
// CHECK-NEXT: [[V4:%\d+]] = arith.add %0, %0;
// CHECK-NEXT: builtin.ret [[V4]] : (i32);
            "#
        );
    }

    #[test]
    fn converts_debug_declares_for_promoted_locals() {
        let mut test = Test::new("promotes_with_declare", |cx| ([cx.i32()], [cx.i32()]));
        test.context().get_or_register_dialect::<DebugInfoDialect>();
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();
            let variable = Variable::new(Symbol::intern("x"), Symbol::intern("test.rs"), 1, None);
            let expr = Expression::with_ops(alloc::vec![ExpressionOp::LocalSlot(0)]);
            BuiltinOpBuilder::builder_mut(&mut builder)
                .debug_declare(variable, expr, SourceSpan::UNKNOWN)
                .unwrap();
            let v2 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.add(v2, v2, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v3], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        std::println!("{output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @promotes_with_declare
// CHECK-NEXT: di.debug_value %0 <{ variable = #di.variable<{ name = "x", file = "test.rs", line = 1 }>, expression = #di.expression<[]> }> : (i32);
// CHECK-NEXT: [[V3:%\d+]] = arith.add %0, %0;
// CHECK-NEXT: builtin.ret [[V3]] : (i32);
// CHECK-NOT: di.debug_declare
// CHECK-NOT: builtin.store_local
            "#
        );
    }

    #[test]
    fn converts_debug_declares_along_a_chain_of_locals() {
        // `y` is declared in slot 1, which stores slot 0's load, and promoting slot 0 replaces
        // that load. Slot 1 is promoted when it is loaded, and its store is dead when it is not;
        // either way the declaration follows the value slot 0 stored.
        for loaded in [true, false] {
            let mut test = Test::new("declares_chain", |cx| ([cx.i32()], [cx.i32()]));
            test.context().get_or_register_dialect::<DebugInfoDialect>();
            let i32 = test.context().i32();
            {
                let mut builder = test.function_builder();
                let local0 = builder.alloc_local(i32);
                let local1 = builder.alloc_local(i32);
                let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
                builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();
                let v1 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
                builder.store_local(local1, v1, SourceSpan::UNKNOWN).unwrap();
                let variable =
                    Variable::new(Symbol::intern("y"), Symbol::intern("test.rs"), 2, None);
                let expr = Expression::with_ops(alloc::vec![ExpressionOp::LocalSlot(1)]);
                BuiltinOpBuilder::builder_mut(&mut builder)
                    .debug_declare(variable, expr, SourceSpan::UNKNOWN)
                    .unwrap();
                let v2 = if loaded {
                    builder.load_local(local1, SourceSpan::UNKNOWN).unwrap()
                } else {
                    v0
                };
                let v3 = builder.add(v2, v0, SourceSpan::UNKNOWN).unwrap();
                builder.ret([v3], SourceSpan::UNKNOWN).unwrap();
            }

            test.apply_pass::<Local2Reg>(true).expect("invalid ir");

            let output = print(&test);
            std::println!("loaded = {loaded}: {output}");
            filecheck!(
                output,
                r#"
// CHECK-LABEL: builtin.function public extern("C") @declares_chain
// CHECK-NEXT: di.debug_value %0 <{ variable = #di.variable<{ name = "y", file = "test.rs", line = 2 }>, expression = #di.expression<[]> }> : (i32);
// CHECK-NEXT: [[V3:%\d+]] = arith.add %0, %0;
// CHECK-NEXT: builtin.ret [[V3]] : (i32);
// CHECK-NOT: di.debug_declare
// CHECK-NOT: builtin.store_local
                "#
            );
        }
    }

    #[test]
    fn rewrites_debug_values_for_promoted_locals() {
        let mut test = Test::new("promotes_with_debug_value", |cx| ([cx.i32()], [cx.i32()]));
        test.context().get_or_register_dialect::<DebugInfoDialect>();
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let value = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.store_local(local0, value, SourceSpan::UNKNOWN).unwrap();
            let variable = Variable::new(Symbol::intern("x"), Symbol::intern("test.rs"), 1, None);
            let expression = Expression::with_ops(alloc::vec![ExpressionOp::LocalSlot(0)]);
            BuiltinOpBuilder::builder_mut(&mut builder)
                .debug_value_with_expr(value, variable, Some(expression), SourceSpan::UNKNOWN)
                .unwrap();
            builder.ret([value], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @promotes_with_debug_value
// CHECK-NEXT: di.debug_value %0 <{ variable = #di.variable<{ name = "x", file = "test.rs", line = 1 }>, expression = #di.expression<[]> }> : (i32);
// CHECK-NEXT: builtin.ret %0 : (i32);
// CHECK-NOT: builtin.store_local
// CHECK-NOT: DI_OP_local_slot
            "#
        );
    }

    #[test]
    fn preserves_stores_for_transformed_debug_values() {
        let mut test =
            Test::new("preserves_transformed_debug_value", |cx| ([cx.i32()], [cx.i32()]));
        test.context().get_or_register_dialect::<DebugInfoDialect>();
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let value = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.store_local(local0, value, SourceSpan::UNKNOWN).unwrap();
            let variable = Variable::new(Symbol::intern("x"), Symbol::intern("test.rs"), 1, None);
            let expression =
                Expression::with_ops(alloc::vec![ExpressionOp::LocalSlot(0), ExpressionOp::Deref]);
            BuiltinOpBuilder::builder_mut(&mut builder)
                .debug_value_with_expr(value, variable, Some(expression), SourceSpan::UNKNOWN)
                .unwrap();
            builder.ret([value], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @preserves_transformed_debug_value
// CHECK-NEXT: builtin.store_local 0, %0;
// CHECK-NEXT: di.debug_value %0 <{ variable = #di.variable<{ name = "x", file = "test.rs", line = 1 }>, expression = #di.expression<[DI_OP_local_slot(0), DW_OP_deref]> }> : (i32);
            "#
        );
    }

    #[test]
    fn erases_dead_stores() {
        let mut test = Test::new("erases_dead_stores", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v0], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @erases_dead_stores
// CHECK-NEXT: builtin.ret %0 : (i32);
            "#
        );
    }

    #[test]
    fn does_not_promote_multiply_loaded_locals() {
        let mut test = Test::new("ignores_multiple_loads", |cx| ([cx.i32(), cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let local1 = builder.alloc_local(i32);
            let [v0, v1] = *builder.entry_block().borrow().arguments()[0..2].as_array().unwrap();
            let v0 = v0 as ValueRef;
            let v1 = v1 as ValueRef;
            builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local1, v1, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.load_local(local1, SourceSpan::UNKNOWN).unwrap();
            let v4 = builder.load_local(local1, SourceSpan::UNKNOWN).unwrap();
            let v5 = builder.add(v2, v3, SourceSpan::UNKNOWN).unwrap();
            let v6 = builder.add(v5, v4, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v6], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        std::println!("output: {output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @ignores_multiple_loads
// CHECK-NEXT: builtin.store_local 1, %1;
// CHECK-NEXT: [[V3:%\d+]] = builtin.load_local 1 : i32;
// CHECK-NEXT: [[V4:%\d+]] = builtin.load_local 1 : i32;
// CHECK-NEXT: [[V5:%\d+]] = arith.add %0, [[V3]];
// CHECK-NEXT: [[V6:%\d+]] = arith.add [[V5]], [[V4]]
// CHECK-NEXT: builtin.ret [[V6]] : (i32);
            "#
        );
    }

    #[test]
    fn does_not_promote_multiply_stored_locals() {
        let mut test =
            Test::new("ignores_multiple_stores", |cx| ([cx.i32(), cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let local1 = builder.alloc_local(i32);
            let [v0, v1] = *builder.entry_block().borrow().arguments()[0..2].as_array().unwrap();
            let v0 = v0 as ValueRef;
            let v1 = v1 as ValueRef;
            builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local1, v1, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.load_local(local1, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local1, v1, SourceSpan::UNKNOWN).unwrap();
            let v4 = builder.add(v2, v3, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v4], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        std::println!("output: {output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @ignores_multiple_stores
// CHECK-NEXT: builtin.store_local 1, %1;
// CHECK-NEXT: [[V3:%\d+]] = builtin.load_local 1 : i32;
// CHECK-NEXT: builtin.store_local 1, %1;
// CHECK-NEXT: [[V4:%\d+]] = arith.add %0, [[V3]];
// CHECK-NEXT: builtin.ret [[V4]] : (i32);
            "#
        );
    }

    #[test]
    fn does_not_promote_poison_loads() {
        let mut test = Test::new("ignores_poison_loads", |cx| ([cx.i32(), cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let v2 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.add(v0, v2, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v3], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @ignores_poison_loads
// CHECK-NEXT: [[V2:%\d+]] = builtin.load_local 0 : i32;
// CHECK-NEXT: [[V3:%\d+]] = arith.add %0, [[V2]];
// CHECK-NEXT: builtin.ret [[V3]] : (i32);
            "#
        );
    }

    #[test]
    fn does_not_promote_a_load_that_precedes_its_store() {
        let mut test = Test::new("ignores_load_before_store", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        // The load reads the slot before the store writes it, and its use follows the store
        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let v1 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.add(v0, v0, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local0, v2, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.mul(v1, v0, SourceSpan::UNKNOWN).unwrap();
            let v4 = builder.add(v3, v2, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v4], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        std::println!("{output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @ignores_load_before_store
// CHECK-NEXT: [[V1:%\d+]] = builtin.load_local 0 : i32;
// CHECK-NEXT: [[V2:%\d+]] = arith.add %0, %0;
// CHECK-NEXT: builtin.store_local 0, [[V2]];
// CHECK-NEXT: [[V3:%\d+]] = arith.mul [[V1]], %0;
            "#
        );
    }

    #[test]
    fn does_not_promote_a_load_used_before_its_store() {
        let mut test = Test::new("ignores_use_before_store", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        // The load and its use both precede the store
        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let v1 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.mul(v1, v0, SourceSpan::UNKNOWN).unwrap();
            let v3 = builder.add(v0, v0, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local0, v3, SourceSpan::UNKNOWN).unwrap();
            let v4 = builder.add(v2, v3, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v4], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        std::println!("{output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @ignores_use_before_store
// CHECK-NEXT: [[V1:%\d+]] = builtin.load_local 0 : i32;
// CHECK-NEXT: [[V2:%\d+]] = arith.mul [[V1]], %0;
// CHECK-NEXT: [[V3:%\d+]] = arith.add %0, %0;
// CHECK-NEXT: builtin.store_local 0, [[V3]];
            "#
        );
    }

    #[test]
    fn does_not_promote_a_load_stored_back_to_its_local() {
        let mut test = Test::new("ignores_self_store", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        // The only store writes back the value the only load read
        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let v1 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(local0, v1, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.add(v1, v0, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v2], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        std::println!("{output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @ignores_self_store
// CHECK-NEXT: [[V1:%\d+]] = builtin.load_local 0 : i32;
// CHECK-NEXT: builtin.store_local 0, [[V1]];
// CHECK-NEXT: [[V2:%\d+]] = arith.add [[V1]], %0;
// CHECK-NEXT: builtin.ret [[V2]] : (i32);
            "#
        );
    }

    #[test]
    fn does_not_promote_across_blocks() {
        use runic_dialect_cf::ControlFlowOpBuilder;

        let mut test = Test::new("ignores_inter_block_candidates", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let v0 = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();

            let block1 = builder.create_block();
            builder.br(block1, None, SourceSpan::UNKNOWN).unwrap();

            builder.switch_to_block(block1);

            let v1 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v2 = builder.add(v0, v1, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v2], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = print(&test);
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @ignores_inter_block_candidates
// CHECK-NEXT: builtin.store_local 0, %0;
// CHECK-NEXT: cf.br ^block1;
// CHECK-LABEL: ^block1:
// CHECK-NEXT: [[V1:%\d+]] = builtin.load_local 0 : i32;
// CHECK-NEXT: [[V2:%\d+]] = arith.add %0, [[V1]];
// CHECK-NEXT: builtin.ret [[V2]] : (i32);
            "#
        );
    }

    #[test]
    fn does_not_promote_across_region_control_flow() {
        use runic_dialect_scf::StructuredControlFlowOpBuilder;

        let mut test = Test::new("ignores_intervening_scf", |cx| ([cx.i32(), cx.i1()], [cx.i32()]));
        let i32 = test.context().i32();

        {
            let mut builder = test.function_builder();
            let local0 = builder.alloc_local(i32);
            let [v0, v1] = *builder.entry_block().borrow().arguments()[0..2].as_array().unwrap();
            let v0 = v0 as ValueRef;
            let v1 = v1 as ValueRef;
            builder.store_local(local0, v0, SourceSpan::UNKNOWN).unwrap();

            let if_op = builder.r#if(v1, &[i32], SourceSpan::UNKNOWN).unwrap();
            let v2 = if_op.borrow().results()[0] as ValueRef;
            let entry_block = builder.current_block();

            {
                let then_region = if_op.borrow().then_body().as_region_ref();
                let then_block = builder.create_block_in_region(then_region);
                builder.switch_to_block(then_block);
                let v3 = builder.i32(1, SourceSpan::UNKNOWN);
                builder.r#yield(Some(v3), SourceSpan::UNKNOWN).unwrap();

                let else_region = if_op.borrow().else_body().as_region_ref();
                let else_block = builder.create_block_in_region(else_region);
                builder.switch_to_block(else_block);
                let v4 = builder.i32(2, SourceSpan::UNKNOWN);
                builder.r#yield(Some(v4), SourceSpan::UNKNOWN).unwrap();
            }
            builder.switch_to_block(entry_block);

            let v5 = builder.load_local(local0, SourceSpan::UNKNOWN).unwrap();
            let v6 = builder.add(v5, v2, SourceSpan::UNKNOWN).unwrap();
            builder.ret([v6], SourceSpan::UNKNOWN).unwrap();
        }

        test.apply_pass::<Local2Reg>(true).expect("invalid ir");

        let output = format!(
            "{}",
            <runic_core::dialects::builtin::Function as Op>::print(
                &test.function().borrow(),
                &Default::default()
            )
        );
        std::println!("output: {output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @ignores_intervening_scf
// CHECK-NEXT: builtin.store_local 0, %0;
// CHECK-NEXT: [[V2:%\d+]] = scf.if %1 then {
// CHECK-NEXT: [[V3:%\d+]] = arith.constant 1 : i32;
// CHECK-NEXT: scf.yield [[V3]] : (i32);
// CHECK-NEXT: } else {
// CHECK-NEXT: [[V4:%\d+]] = arith.constant 2 : i32;
// CHECK-NEXT: scf.yield [[V4]] : (i32);
// CHECK-NEXT: } : (i1) -> (i32);
// CHECK-NEXT: [[V5:%\d+]] = builtin.load_local 0 : i32;
// CHECK-NEXT: [[V6:%\d+]] = arith.add [[V5]], [[V2]];
// CHECK-NEXT: builtin.ret [[V6]] : (i32);
            "#
        );
    }

    const HEADER: &str = "builtin.function public extern(\"C\")";

    /// `source` (a function) parsed, `Local2Reg` run on it, printed.
    fn promote_parsed(source: &str) -> String {
        use runic_core::{
            dialects::builtin::Function,
            pass::{Nesting, PassManager},
            testing::text::parse,
        };

        let (context, function) = parse::<Function>(source).unwrap_or_else(|err| panic!("{err}"));
        let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
        passes.add_pass(alloc::boxed::Box::<Local2Reg>::default());
        passes.enable_verifier(true);
        passes
            .run(function.as_operation_ref())
            .expect("the pass runs and the result verifies");
        function.borrow().as_operation().to_string()
    }

    /// An operation whose effects are unknown and that holds the local in an attribute pins it:
    /// its stores are kept and it is not promoted. Without that attribute the same function is
    /// promoted.
    #[test]
    fn an_operation_with_unknown_effects_that_names_a_local_pins_it() {
        let body = |attributes: &str| {
            format!(
                "{HEADER} @f(%0: i32) -> (i32, i32) locals(i32) {{
    builtin.store_local 0, %0;
    %1 = test.effects <{{ known = #builtin.bool<false> }}>{attributes};
    %2 = builtin.load_local 0 : i32;
    builtin.ret %1, %2 : (i32, i32);
}};"
            )
        };

        let pinned = body(" attributes { local = #builtin.local_variable<0, i32> }");
        let output = promote_parsed(&pinned);
        std::println!("{output}");
        assert_eq!(output, pinned);

        let output = promote_parsed(&body(""));
        std::println!("{output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @f
// CHECK-NEXT: [[V1:%\d+]] = test.effects <{ known = #builtin.bool<false> }>;
// CHECK-NEXT: builtin.ret [[V1]], %0 : (i32, i32);
            "#
        );
    }

    /// A pinned local that is only stored keeps its store: pinning stops the erasure of dead
    /// stores as well as promotion. Without `test.touch_local` the same store is erased.
    #[test]
    fn a_pinned_local_that_is_only_stored_keeps_its_store() {
        let body = |touch: &str| {
            format!(
                "{HEADER} @f(%0: i32) -> i32 locals(i32) {{
    builtin.store_local 0, %0;{touch}
    builtin.ret %0 : (i32);
}};"
            )
        };
        let pinned = body("\n    test.touch_local <{ local = #builtin.local_variable<0, i32> }>;");
        let output = promote_parsed(&pinned);
        std::println!("{output}");
        assert_eq!(output, pinned);

        let output = promote_parsed(&body(""));
        std::println!("{output}");
        assert!(!output.contains("builtin.store_local"), "{output}");
    }

    /// An operation nested in a region pins the local it names: `test.touch_local` inside an
    /// `scf.if` before the store keeps the store and the load. With the `scf.if` holding nothing
    /// else, the local is promoted.
    #[test]
    fn an_operation_in_a_region_pins_the_local_it_names() {
        let body = |touch: &str| {
            format!(
                "{HEADER} @f(%0: i32, %1: i1) -> i32 locals(i32) {{
    scf.if %1 then {{{touch}
        scf.yield : ();
    }} else {{
        scf.yield : ();
    }} : (i1) -> ();
    builtin.store_local 0, %0;
    %2 = builtin.load_local 0 : i32;
    builtin.ret %2 : (i32);
}};"
            )
        };
        let pinned =
            body("\n        test.touch_local <{ local = #builtin.local_variable<0, i32> }>;");
        let output = promote_parsed(&pinned);
        std::println!("{output}");
        assert_eq!(output, pinned);

        let output = promote_parsed(&body(""));
        std::println!("{output}");
        assert!(!output.contains("_local"), "{output}");
        assert!(output.contains("builtin.ret %0 : (i32);"), "{output}");
    }

    /// `test.touch_local`, parsed, pins the local it names, and only that local.
    #[test]
    fn a_parsed_operation_that_names_a_local_pins_only_that_local() {
        let source = format!(
            "{HEADER} @f(%0: i32) -> (i32, i32) locals(i32, i32) {{
    builtin.store_local 0, %0;
    builtin.store_local 1, %0;
    test.touch_local <{{ local = #builtin.local_variable<1, i32> }}>;
    %1 = builtin.load_local 0 : i32;
    %2 = builtin.load_local 1 : i32;
    builtin.ret %1, %2 : (i32, i32);
}};"
        );
        let output = promote_parsed(&source);
        std::println!("{output}");
        filecheck!(
            output,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @f
// CHECK-NEXT: builtin.store_local 1, %0;
// CHECK-NEXT: test.touch_local <{ local = #builtin.local_variable<1, i32> }>;
// CHECK-NEXT: [[V2:%\d+]] = builtin.load_local 1 : i32;
// CHECK-NEXT: builtin.ret %0, [[V2]] : (i32, i32);
            "#
        );
    }

    /// A function parsed from text (its locals unbound) promotes exactly as the same function built
    /// through the builder (its locals bound): the two print the same before and after the pass.
    #[test]
    fn a_parsed_function_promotes_as_a_built_one() {
        let mut test = Test::new("f", |cx| ([cx.i32(), cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();
        {
            let mut builder = test.function_builder();
            let locals = [(); 4].map(|_| builder.alloc_local(i32));
            let [v0, v1] = *builder.entry_block().borrow().arguments()[0..2].as_array().unwrap();
            let v0 = v0 as ValueRef;
            let v1 = v1 as ValueRef;
            // local 0: promoted; local 1: loaded twice; local 2: only stored; local 3: a chain
            builder.store_local(locals[0], v0, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(locals[1], v1, SourceSpan::UNKNOWN).unwrap();
            builder.store_local(locals[2], v1, SourceSpan::UNKNOWN).unwrap();
            let a = builder.load_local(locals[0], SourceSpan::UNKNOWN).unwrap();
            builder.store_local(locals[3], a, SourceSpan::UNKNOWN).unwrap();
            let b = builder.load_local(locals[1], SourceSpan::UNKNOWN).unwrap();
            let c = builder.load_local(locals[1], SourceSpan::UNKNOWN).unwrap();
            let d = builder.load_local(locals[3], SourceSpan::UNKNOWN).unwrap();
            let e = builder.add(b, c, SourceSpan::UNKNOWN).unwrap();
            let f = builder.add(e, d, SourceSpan::UNKNOWN).unwrap();
            builder.ret([f], SourceSpan::UNKNOWN).unwrap();
        }
        let before = print(&test);
        let parsed = promote_parsed(&before);
        test.apply_pass::<Local2Reg>(true).expect("invalid ir");
        let built = print(&test);
        std::println!("{before}\n{built}");
        assert_ne!(built, before);
        assert_eq!(parsed, built);
        filecheck!(
            built,
            r#"
// CHECK-LABEL: builtin.function public extern("C") @f
// CHECK-NEXT: builtin.store_local 1, %1;
// CHECK-NEXT: [[B:%\d+]] = builtin.load_local 1 : i32;
// CHECK-NEXT: [[C:%\d+]] = builtin.load_local 1 : i32;
// CHECK-NEXT: [[E:%\d+]] = arith.add [[B]], [[C]];
// CHECK-NEXT: [[F:%\d+]] = arith.add [[E]], %0;
// CHECK-NEXT: builtin.ret [[F]] : (i32);
            "#
        );
    }
}
