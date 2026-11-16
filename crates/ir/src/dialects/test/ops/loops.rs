use smallvec::smallvec;

use crate::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::test::TestDialect,
    effects::*,
    traits::*,
    *,
};

/// A loop with one region, which is both its header and its body: `%r = test.loop %init body {
/// ^bb0(%arg: i32): .. } : (i32) -> i32;`.
///
/// The operands enter the region as the arguments of its entry block. Control leaves the loop by
/// the region's [Yield], whose operands are the loop's results, and goes back to the top of the
/// region by a [Continue] nested in it, whose operands are the entry block's next arguments. The
/// loop stands for WebAssembly's `loop`: its back edge is an early exit, not an edge of its region
/// graph, so it overrides the questions the default graph walk answers wrongly
/// ([RegionBranchOpInterface::has_loop], [RegionBranchOpInterface::is_repetitive_region]), says a
/// continue may originate in its header ([HasBreakingControlFlowOpInterface::continue_may_originate_in_header])
/// and is a [LoopLikeOpInterface].
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, HasRecursiveMemoryEffects, PropagateControlFlowBreak),
    implements(
        RegionBranchOpInterface,
        HasBreakingControlFlowOpInterface,
        LoopLikeOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct Loop {
    #[operands]
    inits: AnyType,
    #[region]
    body: Region,
    #[results]
    returns: AnyType,
}

impl Loop {
    /// Whether an operation nested in this loop is a [Continue] addressing it.
    fn has_nested_continue(&self) -> bool {
        let me = self.as_operation().as_operation_ref();
        self.body()
            .prewalk(|op: &Operation| match op.as_trait::<dyn RegionExitTerminatorOpInterface>() {
                Some(exit) if exit.get_exit_target() == Some(me) => WalkResult::<()>::Break(()),
                _ => WalkResult::Continue(()),
            })
            .was_interrupted()
    }
}

impl RegionBranchOpInterface for Loop {
    fn get_entry_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.operands().all())
    }

    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Entering(self.body().as_region_ref())],
            ),
            // The end of the body leaves the loop. The way back is a continue, an exit edge that
            // the continue states itself.
            RegionBranchPoint::Child(_) => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Returning(
                    self.results().all().iter().map(|r| r.borrow().as_value_ref()).collect(),
                )],
            ),
        }
    }

    /// The region is repeated when a continue addresses the loop: the region graph has no edge
    /// back to it.
    fn is_repetitive_region(&self, _index: usize) -> bool {
        self.has_nested_continue()
    }

    fn has_loop(&self) -> bool {
        self.has_nested_continue()
    }
}

impl HasBreakingControlFlowOpInterface for Loop {
    fn has_nested_predecessors(&self) -> bool {
        self.has_nested_continue()
    }

    fn get_continue_target_region(&self) -> Option<RegionRef> {
        Some(self.body().as_region_ref())
    }

    fn continue_may_originate_in_header(&self) -> bool {
        true
    }
}

impl LoopLikeOpInterface for Loop {
    fn get_loop_header_region(&self) -> RegionRef {
        self.body().as_region_ref()
    }

    fn get_loop_regions(&self) -> SmallVec<[RegionRef; 2]> {
        smallvec![self.body().as_region_ref()]
    }

    fn get_region_iter_args(&self) -> Option<EntityRef<'_, [BlockArgumentRef]>> {
        let entry = self.body().entry_block_ref()?;
        Some(EntityRef::map(entry.borrow(), |block| block.arguments()))
    }

    fn get_inits_mut(&mut self) -> OpOperandRangeMut<'_> {
        self.inits_mut()
    }
}

/// An operation that runs its region when its condition is true and forwards early exits:
/// `test.guard %c body { .. } : (i1) -> ();`.
///
/// It has no results; the region ends in a [Yield] of nothing or in an early exit. It stands for
/// an `if` in the path of an exit.
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = TestDialect,
    traits(SingleBlock, NoRegionArguments, HasRecursiveMemoryEffects, PropagateControlFlowBreak),
    implements(RegionBranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Guard {
    #[operand]
    condition: Bool,
    #[region]
    body: Region,
}

impl RegionBranchOpInterface for Guard {
    /// The region may be skipped, so control may go from the entry to after the guard.
    fn get_entry_successor_regions(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> RegionSuccessorIter<'_> {
        let condition = operands[0].as_ref().and_then(|v| v.borrow().as_bool());
        let mut infos = SmallVec::<[RegionSuccessorInfo; 2]>::default();
        if condition.is_none_or(|v| v) {
            infos.push(RegionSuccessorInfo::Entering(self.body().as_region_ref()));
        }
        if condition.is_none_or(|v| !v) {
            infos.push(RegionSuccessorInfo::Returning(SmallVec::new()));
        }
        RegionSuccessorIter::new(self.as_operation(), infos)
    }

    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Entering(self.body().as_region_ref())],
            ),
            RegionBranchPoint::Child(_) => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Returning(SmallVec::new())],
            ),
        }
    }
}

/// Ends a region of an operation that implements [RegionBranchOpInterface] by passing its operands
/// to the results of that operation (the end of the body of a [Loop] leaves it):
/// `test.yield %a : (i32);`.
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = TestDialect,
    traits(Terminator, ReturnLike),
    implements(RegionBranchTerminatorOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Yield {
    #[operands]
    values: AnyType,
}

impl RegionBranchTerminatorOpInterface for Yield {
    fn get_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.values())
    }

    fn get_mutable_successor_operands(
        &mut self,
        _point: RegionBranchPoint,
    ) -> SuccessorOperandRangeMut<'_> {
        SuccessorOperandRangeMut::forward(self.values_mut())
    }
}

/// Goes back to the top of the nearest enclosing operation that receives breaking control flow (a
/// [Loop]), leaving every operation in between, with its operands as the arguments of that
/// operation's header: `test.continue %next : (i32);`.
///
/// It is the minimal early exit of the `test` dialect, the counterpart of `scf.continue` (which
/// the crate cannot use) without a level count.
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = TestDialect,
    traits(Terminator, ReturnLike),
    implements(
        RegionBranchTerminatorOpInterface,
        RegionExitTerminatorOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct Continue {
    #[operands]
    args: AnyType,
}

impl RegionExitTerminatorOpInterface for Continue {
    fn resolve_exit(&self) -> ExitResolution {
        let not_found = || {
            ExitResolution::Unresolved(ExitDefect {
                kind: ExitDefectKind::NotFound,
                detail: "this continue names no enclosing loop: none was found before the \
                         enclosing function body was reached"
                    .into(),
            })
        };
        let Some(mut from_region) = self.as_operation().parent_region() else {
            return not_found();
        };
        loop {
            let Some(parent_ref) = from_region.borrow().parent() else {
                return not_found();
            };
            let parent = parent_ref.borrow();
            if parent.implements::<dyn CallableOpInterface>() {
                return not_found();
            }
            if parent.implements::<dyn HasBreakingControlFlowOpInterface>() {
                return ExitResolution::Resolved(ResolvedExit {
                    target: parent_ref,
                    from_region,
                    callable: None,
                });
            }
            if !parent.implements::<dyn PropagateControlFlowBreak>() {
                return ExitResolution::Unresolved(ExitDefect {
                    detail: alloc::format!(
                        "this continue would have to cross '{}', which has not declared that it \
                         forwards breaking control flow",
                        parent.name()
                    ),
                    kind: ExitDefectKind::Opaque {
                        crossing: parent_ref,
                    },
                });
            }
            let Some(next) = parent.parent_region() else {
                return not_found();
            };
            from_region = next;
        }
    }

    fn get_exit_kind(&self) -> RegionExitKind {
        RegionExitKind::Continue
    }
}

impl RegionBranchTerminatorOpInterface for Continue {
    fn get_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.args())
    }

    fn get_mutable_successor_operands(
        &mut self,
        _point: RegionBranchPoint,
    ) -> SuccessorOperandRangeMut<'_> {
        SuccessorOperandRangeMut::forward(self.args_mut())
    }

    fn get_successor_regions(
        &self,
        _operands: &[Option<AttributeRef>],
    ) -> SmallVec<[RegionSuccessorInfo; 2]> {
        // An exit that does not resolve states no successor: the verifier reports it
        let Some(target) = self.get_exit_target() else {
            return SmallVec::new();
        };
        let target_op = target.borrow();
        let Some(header) = target_op
            .as_trait::<dyn HasBreakingControlFlowOpInterface>()
            .and_then(|receiver| receiver.get_continue_target_region())
        else {
            return SmallVec::new();
        };
        let Some(entry) = header.borrow().entry_block_ref() else {
            return SmallVec::new();
        };
        let values = entry
            .borrow()
            .arguments()
            .iter()
            .map(|arg| arg.borrow().as_value_ref())
            .collect();
        smallvec![RegionSuccessorInfo::Exiting {
            target,
            to: RegionBranchPoint::Child(header),
            values,
        }]
    }
}
