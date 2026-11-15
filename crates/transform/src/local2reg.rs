use runic_core::{
    AttributeRef, Backward, CallOpInterface, EntityMut, Forward, FxHashMap, FxHashSet, Op,
    Operation, OperationName, OperationRef, ProgramPoint, RawWalk, RegionBranchOpInterface, Report,
    Rewriter, SmallVec, Spanned, Symbol, ValueRef, WalkResult,
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
    walk_nested_attributes,
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

/// The index, within its function, of the local `attr` names.
///
/// Locals are keyed by index: for operations outside nested callables, the function's verifier
/// ensures that every local of one index in a function has the type the function gives it.
fn local_index(attr: &LocalVariableAttr) -> usize {
    attr.as_value().index() as usize
}

/// Adds to `pinned` the index of every local that `operation` names: a [LocalVariableAttr] it
/// holds at any depth (in a property or attribute, or inside one, such as an entry of a list), or
/// held by the attribute of one of its memory effects.
///
/// An operation may have unknown effects, so its effects alone do not show every operation that
/// uses a local; one that holds a local in a property or attribute is taken to use it.
fn pin_locals_named_by(operation: &Operation, pinned: &mut FxHashSet<usize>) {
    let mut pin = |attr: &AttributeRef| {
        if let Some(local) = attr.borrow().downcast_ref::<LocalVariableAttr>() {
            pinned.insert(local_index(local));
        }
        WalkResult::Continue(())
    };
    let _ = operation.walk_attributes(&mut |_, attr| pin(attr));
    // The effects of an operation whose effects are unknown are not read: such an operation pins
    // through its properties and attributes, above
    if let Some(effects) = operation.as_trait::<dyn MemoryEffectOpInterface>()
        && effects.effects_are_known()
    {
        for effect in effects.effects() {
            if let Some(attribute) = effect.attribute() {
                let _ = walk_nested_attributes(&attribute, &mut pin);
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
    DESCRIPTION
));

const DESCRIPTION: &str = "Replaces a local stored once and loaded once by the stored value, and \
                           erases stores to unread locals";

impl Pass for Local2Reg {
    type Target = Function;

    fn name(&self) -> &'static str {
        "local2reg"
    }

    fn argument(&self) -> &'static str {
        "local2reg"
    }

    fn description(&self) -> &'static str {
        DESCRIPTION
    }

    fn can_schedule_on(&self, name: &OperationName) -> bool {
        name.is::<Function>()
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
    };

    use litcheck_filecheck::{filecheck, litcheck};
    use runic_core::{
        BuilderExt, Op, SourceSpan, ValueRef, dialects::builtin::BuiltinOpBuilder, testing::Test,
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

    /// A local named inside an attribute that another operation holds (here an entry of a list
    /// in the dictionary of the `builtin.ret`) is pinned as if the operation named it directly.
    #[test]
    fn preserves_stores_of_a_local_named_inside_another_attribute() {
        use runic_core::{
            attributes::AttrList,
            dialects::builtin::attributes::{List, ListAttr, LocalVariableAttr},
        };

        let mut test = Test::new("named_in_a_list", |cx| ([cx.i32()], [cx.i32()]));
        let context = test.context_rc();
        let i32 = context.i32();
        let store;
        let load;
        {
            let mut builder = test.function_builder();
            let local = builder.alloc_local(i32);
            let input = builder.entry_block().borrow().arguments()[0] as ValueRef;
            store = builder.store_local(local, input, SourceSpan::UNKNOWN).unwrap();
            let loaded = builder.load_local(local, SourceSpan::UNKNOWN).unwrap();
            load = loaded.borrow().get_defining_op().unwrap();
            let mut ret = builder.ret([loaded], SourceSpan::UNKNOWN).unwrap();

            let local = context.create_attribute::<LocalVariableAttr, _>(local);
            let mut entries = AttrList::new();
            entries.push_back(runic_core::Attribute::as_attr(&*local.borrow()).as_attr_ref());
            let list = context.create_attribute::<ListAttr, _>(List::from(entries));
            ret.borrow_mut().as_operation_mut().set_attribute("locals", list);
        }
        test.apply_pass::<Local2Reg>(true).unwrap();
        assert!(
            store.as_operation_ref().parent().is_some(),
            "an operation that names the local in a list keeps the store live"
        );
        assert!(load.parent().is_some(), "and the load is not promoted");
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
        passes.add_pass(alloc::boxed::Box::<Local2Reg>::default()).unwrap();
        passes.enable_verifier(true);
        passes
            .run(function.as_operation_ref())
            .expect("the pass runs and the result verifies");
        function.borrow().as_operation().to_string()
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

    /// A function parsed from text promotes exactly as the same function built through the
    /// builder: the two print the same before and after the pass.
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
