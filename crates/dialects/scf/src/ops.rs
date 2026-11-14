use alloc::{format, rc::Rc, string::String, vec::Vec};

use runic_core::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::{I32ArrayAttr, I32Attr},
    effects::*,
    parse::ParserExt,
    patterns::RewritePatternSet,
    print::AsmPrinter,
    traits::*,
    *,
};

use crate::ScfDialect;

/// [If] is a structured control flow operation representing conditional execution.
///
/// An [If] takes a single condition as an argument, which chooses between one of its two regions
/// based on the condition. If the condition is true, then the `then_body` region is executed,
/// otherwise `else_body`.
///
/// Neither region allows any arguments, and both regions must be terminated with one of:
///
/// * [runic_core::dialects::builtin::Ret] to return from the enclosing function directly
/// * `runic_dialect_ub::Unreachable` to abort execution
/// * [Yield] to return from the enclosing [If]
///
/// The op itself touches no memory — selecting a region is not a read or a write — so it declares
/// no effects OF ITS OWN. What it does to memory is whatever the taken region does, which is what
/// [`HasRecursiveMemoryEffects`] says; the two declarations are complementary, and both are needed.
/// An empty declaration without the marker would claim the whole `scf.if` is effect-free no matter
/// what its regions hold.
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = ScfDialect,
    traits(
        SingleBlock,
        NoRegionArguments,
        HasRecursiveMemoryEffects,
        PropagateControlFlowBreak
    ),
    implements(RegionBranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct If {
    #[operand]
    condition: Bool,
    #[region(name = "then")]
    then_body: Region,
    #[region(name = "else")]
    else_body: Region,
    #[results]
    returns: AnyType,
}

impl If {
    /// The [Yield] terminating the `then` region, if it is terminated by one.
    ///
    /// Returns `None` when the branch is terminated by something else — an `scf.break`, an
    /// `scf.continue`, a `builtin.ret` or a `ub.unreachable`. **This is fallible on purpose**: a
    /// break-terminated branch is legal IR, and this accessor used to `expect` a yield, so every
    /// caller that assumed "the branch yields" is now a caller that has to say what it does when
    /// the branch leaves the `scf.if` entirely instead.
    pub fn then_yield(&self) -> Option<UnsafeIntrusiveEntityRef<Yield>> {
        self.then_body().entry().terminator()?.try_downcast_op::<Yield>().ok()
    }

    /// The [Yield] terminating the `else` region, if it is terminated by one. See
    /// [Self::then_yield].
    pub fn else_yield(&self) -> Option<UnsafeIntrusiveEntityRef<Yield>> {
        self.else_body().entry().terminator()?.try_downcast_op::<Yield>().ok()
    }
}

impl Canonicalizable for If {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(crate::canonicalization::ConvertTrivialIfToSelect::new(context.clone()));
        rewrites.push(crate::canonicalization::IfRemoveUnusedResults::new(context.clone()));
        rewrites.push(crate::canonicalization::FoldRedundantYields::new(context));
    }
}

impl RegionBranchOpInterface for If {
    fn get_entry_successor_regions(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> RegionSuccessorIter<'_> {
        let condition = operands[0].as_ref().and_then(|v| v.borrow().as_bool());
        let has_then = condition.is_none_or(|v| v);
        let else_possible = condition.is_none_or(|v| !v);
        let has_else = else_possible && !self.else_body().is_empty();

        let mut infos = SmallVec::<[RegionSuccessorInfo; 2]>::default();
        if has_then {
            infos.push(RegionSuccessorInfo::Entering(self.then_body().as_region_ref()));
        }

        if else_possible {
            if has_else {
                infos.push(RegionSuccessorInfo::Entering(self.else_body().as_region_ref()));
            } else {
                // Branching back to parent with `then` results
                infos.push(RegionSuccessorInfo::Returning(
                    self.results().all().iter().map(|v| v.borrow().as_value_ref()).collect(),
                ));
            }
        }

        RegionSuccessorIter::new(self.as_operation(), infos)
    }

    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => {
                // Either branch is reachable on entry (unless `else` is empty, as it is optional)
                let mut infos: SmallVec<[_; 2]> =
                    smallvec![RegionSuccessorInfo::Entering(self.then_body().as_region_ref())];
                // Don't consider the else region if it is empty
                if !self.else_body().is_empty() {
                    infos.push(RegionSuccessorInfo::Entering(self.else_body().as_region_ref()));
                }
                RegionSuccessorIter::new(self.as_operation(), infos)
            }
            RegionBranchPoint::Child(_) => {
                // Only the parent If is reachable from then_body/else_body
                RegionSuccessorIter::new(
                    self.as_operation(),
                    [RegionSuccessorInfo::Returning(
                        self.results().all().iter().map(|v| v.borrow().as_value_ref()).collect(),
                    )],
                )
            }
        }
    }

    fn get_region_invocation_bounds(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> SmallVec<[InvocationBounds; 1]> {
        let condition = operands[0].as_ref().and_then(|v| v.borrow().as_bool());

        if let Some(condition) = condition {
            if condition {
                smallvec![InvocationBounds::Exact(1), InvocationBounds::Never]
            } else {
                smallvec![InvocationBounds::Never, InvocationBounds::Exact(1)]
            }
        } else {
            // Only one region is invoked, and no more than a single time
            smallvec![InvocationBounds::NoMoreThan(1); 2]
        }
    }

    #[inline(always)]
    fn is_repetitive_region(&self, _index: usize) -> bool {
        false
    }

    #[inline(always)]
    fn has_loop(&self) -> bool {
        false
    }
}

/// A while is a loop structure composed of two regions: a "before" region, and an "after" region.
///
/// The "before" region's entry block parameters correspond to the operands expected by the
/// operation, and can be used to compute the condition that determines whether the "after" body
/// is executed or not, or simply forwarded to the "after" region. The "before" region must
/// terminate with a [Condition] operation, which will be evaluated to determine whether or not
/// to continue the loop.
///
/// The "after" region corresponds to the loop body, and must terminate with a [Yield] operation,
/// whose operands must be of the same arity and type as the "before" region's argument list. In
/// this way, the "after" body can feed back input to the "before" body to determine whether to
/// continue the loop.
///
/// No effects of its own; its memory effects are its regions', per [`HasRecursiveMemoryEffects`].
/// See [`If`] for why both declarations are required.
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = ScfDialect,
    traits(
        SingleBlock,
        HasRecursiveMemoryEffects,
        PropagateControlFlowBreak
    ),
    implements(
        RegionBranchOpInterface,
        HasBreakingControlFlowOpInterface,
        LoopLikeOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct While {
    #[operands]
    inits: AnyType,
    #[region]
    before: Region,
    #[region]
    after: Region,
    #[results]
    returns: AnyType,
}

impl While {
    /// The [Condition] terminating the `before` region, if it is terminated by one.
    ///
    /// Returns `None` when the header is terminated by an `scf.break` instead — which is legal,
    /// and equivalent to `scf.condition(false, args)`.
    pub fn condition_op(&self) -> Option<UnsafeIntrusiveEntityRef<Condition>> {
        self.before().entry().terminator()?.try_downcast_op::<Condition>().ok()
    }

    /// The [Yield] terminating the `after` region, if it is terminated by one.
    ///
    /// Returns `None` when the body's entry block ends in an `scf.break` or `scf.continue`: such
    /// a body has no yielded values, and its operands are the *target's* results or iter_args,
    /// not this loop's next iteration.
    pub fn yield_op(&self) -> Option<UnsafeIntrusiveEntityRef<Yield>> {
        self.after().entry().terminator()?.try_downcast_op::<Yield>().ok()
    }
}

impl Canonicalizable for While {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(crate::canonicalization::RemoveLoopInvariantArgsFromBeforeBlock::new(
            context.clone(),
        ));
        rewrites.push(crate::canonicalization::WhileConditionTruth::new(context.clone()));
        rewrites.push(crate::canonicalization::WhileUnusedResult::new(context.clone()));
        rewrites.push(crate::canonicalization::WhileRemoveDuplicatedResults::new(context.clone()));
        rewrites.push(crate::canonicalization::WhileRemoveUnusedArgs::new(context));
    }
}

impl LoopLikeOpInterface for While {
    fn get_region_iter_args(&self) -> Option<EntityRef<'_, [BlockArgumentRef]>> {
        let entry = self.before().entry_block_ref()?;
        Some(EntityRef::map(entry.borrow(), |block| block.arguments()))
    }

    fn get_loop_header_region(&self) -> RegionRef {
        self.before().as_region_ref()
    }

    fn get_loop_regions(&self) -> SmallVec<[RegionRef; 2]> {
        smallvec![self.before().as_region_ref(), self.after().as_region_ref()]
    }

    fn get_inits_mut(&mut self) -> OpOperandRangeMut<'_> {
        self.inits_mut()
    }

    fn get_yielded_values_mut(&mut self) -> Option<EntityProjectionMut<'_, OpOperandRangeMut<'_>>> {
        // A body whose entry block ends in `scf.break`/`scf.continue` has NO yielded values: the
        // exit's operands are the target's results or iter_args, and projecting them here would
        // hand a caller an operand group that means something else entirely.
        let mut yield_op = self.yield_op()?.as_operation_ref();

        // The values which are yielded to each iteration
        Some(EntityMut::project(yield_op.borrow_mut(), |op| op.operands_mut().group_mut(0)))
    }
}

impl While {
    /// Returns true if some op nested in this loop addresses it with an early exit of `kind`
    /// (or of any kind, when `kind` is `None`).
    ///
    /// The walk is over the loop's own regions, and resolution is the shared `resolve_exit`, so a
    /// `break [2]` in a nested loop is correctly *not* counted here: it addresses the outer loop,
    /// and the outer loop is the one whose successors change.
    fn has_nested_exit(&self, kind: Option<RegionExitKind>) -> bool {
        let me = self.as_operation().as_operation_ref();
        let region_has_exit = |region: &Region| {
            region
                .prewalk(|op: &Operation| {
                    if let Some(exit) = op.as_trait::<dyn RegionExitTerminatorOpInterface>()
                        && kind.is_none_or(|kind| kind == exit.get_exit_kind())
                        && exit.get_exit_target() == Some(me)
                    {
                        return WalkResult::<()>::Break(());
                    }
                    WalkResult::Continue(())
                })
                .was_interrupted()
        };
        region_has_exit(&self.before()) || region_has_exit(&self.after())
    }
}

impl HasBreakingControlFlowOpInterface for While {
    #[inline]
    fn has_nested_predecessors(&self) -> bool {
        self.has_nested_exit(None)
    }

    #[inline]
    fn get_continue_target_region(&self) -> Option<RegionRef> {
        // A continue re-enters the header, which for `scf.while` is the `before` region — the
        // same region `scf.yield` feeds on the ordinary back edge.
        Some(self.before().as_region_ref())
    }
}

/// The structural rules of [While]:
///
/// * The entry block of the `before` region terminates with [Condition], and the entry block of
///   the `after` region terminates with [Yield] — or either with an exit that leaves the region:
///   a return-like terminator that is neither of the two ([Break] or [Continue], addressing this
///   loop or an enclosing one, or a function return). Such an exit's operands belong to what it
///   addresses, so they are not compared with this loop's lists. `ub.unreachable` is not an
///   exit: the loop asks a region's terminator for the operands of its edge, which an unreachable
///   cannot answer.
/// * The init operands and the `before` block arguments agree in number and, position by
///   position, in type, and so do the [Yield] operands, when the body yields.
/// * The operands forwarded by [Condition], when the header has one, the `after` block arguments
///   and the results agree in number and, position by position, in type.
impl Verify<dyn LoopLikeOpInterface> for While {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let invalid = |label: String| {
            runic_core::diagnostics::report!(
                message: (format!("invalid operation {}", op.name())),
                labels: [primary(op.span(), label)],
            )
        };

        let before = self.before();
        let before_block = before
            .entry_block_ref()
            .ok_or_else(|| invalid("the before region has no block".into()))?;
        let before_block = before_block.borrow();
        let after = self.after();
        let after_block = after
            .entry_block_ref()
            .ok_or_else(|| invalid("the after region has no block".into()))?;
        let after_block = after_block.borrow();

        // An exit leaves the region: return-like, and not the terminator of the other region.
        let is_exit = |op: &Operation| {
            op.implements::<dyn ReturnLike>() && !op.is::<Condition>() && !op.is::<Yield>()
        };
        let before_term = before_block
            .terminator()
            .ok_or_else(|| invalid("the before block has no terminator".into()))?;
        let before_term = before_term.borrow();
        let condition = before_term.downcast_ref::<Condition>();
        if condition.is_none() && !is_exit(&before_term) {
            return Err(invalid(format!(
                "the before block must end in scf.condition or in an exit, but it ends in {}",
                before_term.name()
            )));
        }
        let after_term = after_block
            .terminator()
            .ok_or_else(|| invalid("the after block has no terminator".into()))?;
        let after_term = after_term.borrow();
        let yield_op = after_term.downcast_ref::<Yield>();
        if yield_op.is_none() && !is_exit(&after_term) {
            return Err(invalid(format!(
                "the after block must end in scf.yield or in an exit, but it ends in {}",
                after_term.name()
            )));
        }

        let operand_types =
            |range: OpOperandRange<'_>| range.iter().map(|o| o.borrow().ty()).collect::<Vec<_>>();
        let argument_types =
            |block: &Block| block.arguments().iter().map(|a| a.borrow().ty()).collect::<Vec<_>>();
        let inits = ("init operand", operand_types(self.inits()));
        let before_args = ("before block argument", argument_types(&before_block));
        let yielded =
            yield_op.map(|yield_op| ("scf.yield operand", operand_types(yield_op.yielded())));
        let forwarded = condition.map(|condition| {
            ("scf.condition forwarded operand", operand_types(condition.forwarded()))
        });
        let after_args = ("after block argument", argument_types(&after_block));
        let results = (
            "result",
            self.results().all().iter().map(|r| r.borrow().ty()).collect::<Vec<_>>(),
        );

        let mut pairs = Vec::from([(&inits, &before_args)]);
        if let Some(yielded) = &yielded {
            pairs.push((yielded, &inits));
        }
        if let Some(forwarded) = &forwarded {
            pairs.push((forwarded, &after_args));
            pairs.push((&results, forwarded));
        }
        for ((lhs_name, lhs), (rhs_name, rhs)) in pairs {
            if lhs.len() != rhs.len() {
                return Err(invalid(format!(
                    "{} {lhs_name}(s), but {} {rhs_name}(s)",
                    lhs.len(),
                    rhs.len()
                )));
            }
            if let Some((index, (lhs_ty, rhs_ty))) =
                lhs.iter().zip(rhs.iter()).enumerate().find(|(_, (lhs, rhs))| lhs != rhs)
            {
                return Err(invalid(format!(
                    "{lhs_name} {index} has type '{lhs_ty}', but {rhs_name} {index} has type \
                     '{rhs_ty}'"
                )));
            }
        }

        Ok(())
    }
}

impl RegionBranchOpInterface for While {
    #[inline]
    fn get_entry_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        // Operands being forwarded to the `before` region from outside the loop
        SuccessorOperandRange::forward(self.operands().all())
    }

    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => {
                // The only successor region when branching from outside the While op is the
                // `before` region.
                RegionSuccessorIter::new(
                    self.as_operation(),
                    [RegionSuccessorInfo::Entering(self.before().as_region_ref())],
                )
            }
            RegionBranchPoint::Child(region) => {
                let before_region = self.before().as_region_ref();
                let after_region = self.after().as_region_ref();
                assert!(region == before_region || region == after_region);

                // When branching from `before`, the only successor is `after` or the While itself,
                // otherwise, when branching from `after` the only successor is `before` — UNLESS
                // the body contains a break addressing this loop, in which case the body also
                // exits the loop directly, without passing through `scf.condition`.
                //
                // **This is the edge the early-exit design exists to add, and omitting it is the
                // silent failure.** Nothing crashes without it; every analysis simply believes
                // the only way out of the loop is the condition's false edge, and concludes that
                // code reachable only via the break is dead.
                //
                // The question asked here is narrower than `has_nested_predecessors`: a
                // `continue` is also a nested predecessor, but it re-enters `before`, an edge
                // this arm already reports. Only a `break` adds an exit.
                if region == after_region {
                    let mut infos: SmallVec<[_; 2]> =
                        smallvec![RegionSuccessorInfo::Entering(before_region)];
                    if self.has_nested_exit(Some(RegionExitKind::Break)) {
                        infos.push(RegionSuccessorInfo::Returning(
                            self.results()
                                .all()
                                .iter()
                                .map(|r| r.borrow().as_value_ref())
                                .collect(),
                        ));
                    }
                    RegionSuccessorIter::new(self.as_operation(), infos)
                } else {
                    RegionSuccessorIter::new(
                        self.as_operation(),
                        [
                            RegionSuccessorInfo::Returning(
                                self.results()
                                    .all()
                                    .iter()
                                    .map(|r| r.borrow().as_value_ref())
                                    .collect(),
                            ),
                            RegionSuccessorInfo::Entering(after_region),
                        ],
                    )
                }
            }
        }
    }

    #[inline]
    fn get_region_invocation_bounds(
        &self,
        _operands: &[Option<AttributeRef>],
    ) -> SmallVec<[InvocationBounds; 1]> {
        smallvec![InvocationBounds::Unknown; self.num_regions()]
    }

    #[inline(always)]
    fn is_repetitive_region(&self, _index: usize) -> bool {
        // Both regions are in the loop (`before` -> `after` -> `before` -> `after`)
        true
    }

    #[inline(always)]
    fn has_loop(&self) -> bool {
        true
    }
}

/// `scf.index_switch` is a control-flow operation that branches to one of the given regions
/// based on the values of the argument and the cases. The argument is always an `i32`.
///
/// The operation always has a "default" region and any number of case regions denoted by `i32`
/// constants, written as signed decimals. Control-flow transfers to the case region whose value
/// equals the value of the argument, compared as 32-bit patterns (so a selector `0xffff_ffff`
/// chooses `case -1`). If the argument does not equal any of the case values, control-flow
/// transfers to the "default" region.
///
/// ## Example
///
/// ```text,ignore
/// %0 = scf.index_switch %arg0
/// case -1 {
///   %1 = arith.constant 10 : i32;
///   scf.yield %1 : (i32);
/// }
/// case 5 {
///   %2 = arith.constant 20 : i32;
///   scf.yield %2 : (i32);
/// }
/// default {
///   %3 = arith.constant 30 : i32;
///   scf.yield %3 : (i32);
/// } : (i32);
/// ```
///
/// No effects of its own; its memory effects are its regions', per [`HasRecursiveMemoryEffects`].
/// See [`If`] for why both declarations are required.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = ScfDialect,
    traits(
        SingleBlock,
        HasRecursiveMemoryEffects,
        PropagateControlFlowBreak
    ),
    implements(RegionBranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct IndexSwitch {
    #[operand]
    selector: Int32,
    #[attr]
    cases: I32ArrayAttr,
    #[region]
    default_region: Region,
}

/// The case values are distinct, and there is one region for each after the default region; and
/// the interface's own rule, [verify_region_branch_entry_operands], holds.
impl Verify<dyn RegionBranchOpInterface> for IndexSwitch {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let invalid = |label: String| {
            runic_core::diagnostics::report!(
                message: (format!("invalid operation {}", op.name())),
                labels: [primary(op.span(), label)],
            )
        };
        let cases = self.cases();
        let case_regions = op.num_regions().saturating_sub(1);
        if case_regions != cases.len() {
            return Err(invalid(format!(
                "{} case value(s) need {} region(s) after the default region, but there are \
                 {case_regions}",
                cases.len(),
                cases.len(),
            )));
        }
        for (index, case) in cases.iter().enumerate() {
            if cases.iter().take(index).any(|earlier| earlier == case) {
                return Err(invalid(format!("the case value {case} is used more than once")));
            }
        }
        verify_region_branch_entry_operands(self)
    }
}

impl OpPrinter for IndexSwitch {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use formatter::*;

        printer.print_space();
        printer.print_value_uses(ValueRange::<1>::Operands(&[self.selector().as_operand_ref()]));
        printer.print_space();

        for case in self.cases().iter() {
            let index = self.get_case_index_for_selector(*case).unwrap();
            let region = self.get_case_region(index);
            *printer += nl() + const_text("case ") + display(*case) + const_text(" ");
            printer.print_region(&region.borrow());
        }

        *printer += nl() + const_text("default ");
        printer.print_region(&self.default_region());

        if self.op.has_attributes() {
            printer.print_space();
            printer.print_attribute_dictionary(
                self.op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }

        printer.print_space();
        printer.print_colon_type_list(self.results().iter().map(|r| r.borrow().ty()));
    }
}

impl OpParser for IndexSwitch {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        use runic_core::{dialects::builtin::attributes::Array, parse::ParserError};

        let selector = parser.parse_operand(/*allow_result_number=*/ true)?;
        let selector = parser.resolve_operand(selector, parser.context().i32())?;
        state.add_operand(selector);

        let mut cases = Array::<i32>::default();
        let mut regions = SmallVec::<[RegionRef; 2]>::default();
        while parser.parse_optional_custom_keyword("case")?.is_some() {
            let (span, case_value) = parse_i32(parser)?;
            if cases.contains(&case_value) {
                return Err(ParserError::from(runic_core::diagnostics::report!(
                    message: "invalid scf.index_switch operation",
                    labels: [primary(span, "this case selector has already been used")],
                )));
            }

            let region = parser.context().create_region();
            parser.parse_region(region, &[], false)?;

            cases.push(case_value);
            regions.push(region);
        }

        parser.parse_custom_keyword("default")?;
        let fallback_region = parser.context().create_region();
        parser.parse_region(fallback_region, &[], false)?;

        state
            .add_attribute("cases", parser.context_rc().create_attribute::<I32ArrayAttr, _>(cases));
        // The default region is declared first on the operation; case regions follow it (see
        // `get_case_region`).
        state.add_region(fallback_region);
        for region in regions {
            state.add_region(region);
        }

        parser.parse_optional_attribute_dict(&mut state.attrs)?;
        parser.parse_colon_type_list(&mut state.results)?;

        Ok(())
    }
}

impl IndexSwitch {
    pub fn num_cases(&self) -> usize {
        self.cases().len()
    }

    pub fn get_default_block(&self) -> BlockRef {
        self.default_region().entry_block_ref().expect("default region has no blocks")
    }

    /// The index of the case whose value is `selector`, or `None` when the default region is
    /// taken.
    pub fn get_case_index_for_selector(&self, selector: i32) -> Option<usize> {
        self.cases().iter().position(|case| *case == selector)
    }

    #[track_caller]
    pub fn get_case_block(&self, index: usize) -> BlockRef {
        let block_ref = self.get_case_region(index).borrow().entry_block_ref();
        match block_ref {
            None => panic!("region for case {index} has no blocks"),
            Some(block) => block,
        }
    }

    #[track_caller]
    pub fn get_case_region(&self, mut index: usize) -> RegionRef {
        let mut next_region = self.regions().front().as_pointer();
        let mut current_index = 0;
        // Shift the requested index up by one to account for default region
        index += 1;
        while let Some(region) = next_region.take() {
            if index == current_index {
                return region;
            }
            next_region = region.next();
            current_index += 1;
        }

        panic!("invalid region index `{}`: out of bounds", index - 1)
    }
}

/// The value of a constant `scf.index_switch` selector, read as a 32-bit pattern: any 32-bit
/// integer-like attribute, so a constant `0xffff_ffff : i32` is the case value `-1`. `None` for
/// any other attribute (poison, say).
pub(crate) fn constant_selector(attr: &AttributeRef) -> Option<i32> {
    let value = attr.borrow().as_int()?;
    (value.width() == 32).then(|| value.unsigned() as u32 as i32)
}

/// Parses an `i32` written the way an `i32` constant's literal is written: a signed decimal, or a
/// hex or binary literal standing for its 32-bit pattern.
fn parse_i32(
    parser: &mut dyn OpAsmParser<'_>,
) -> Result<(SourceSpan, i32), runic_core::parse::ParserError> {
    let (span, literal) = parser.parse_integer_literal()?.into_parts();
    let value = literal.to_int(parser.context().i32(), span)?;
    Ok((span, value.signed() as i32))
}

impl RegionBranchOpInterface for IndexSwitch {
    fn get_entry_successor_regions(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> RegionSuccessorIter<'_> {
        let selector = operands[0].as_ref().and_then(constant_selector);
        let selected = selector.map(|s| self.get_case_index_for_selector(s));

        match selected {
            None => {
                // All regions are possible successors
                let infos =
                    self.regions().iter().map(|r| RegionSuccessorInfo::Entering(r.as_region_ref()));
                RegionSuccessorIter::new(self.as_operation(), infos)
            }
            Some(Some(selected)) => {
                // A specific case was selected
                RegionSuccessorIter::new(
                    self.as_operation(),
                    [RegionSuccessorInfo::Entering(self.get_case_region(selected))],
                )
            }
            Some(None) => {
                // The fallback case should be used
                RegionSuccessorIter::new(
                    self.as_operation(),
                    [RegionSuccessorInfo::Entering(self.default_region().as_region_ref())],
                )
            }
        }
    }

    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => {
                // Any region is reachable on entry
                let infos =
                    self.regions().iter().map(|r| RegionSuccessorInfo::Entering(r.as_region_ref()));
                RegionSuccessorIter::new(self.as_operation(), infos)
            }
            RegionBranchPoint::Child(_) => {
                // Only the parent op is reachable from its regions
                RegionSuccessorIter::new(
                    self.as_operation(),
                    [RegionSuccessorInfo::Returning(
                        self.results().all().iter().map(|v| v.borrow().as_value_ref()).collect(),
                    )],
                )
            }
        }
    }

    fn get_region_invocation_bounds(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> SmallVec<[InvocationBounds; 1]> {
        let selector = operands[0].as_ref().and_then(constant_selector);

        if let Some(selector) = selector {
            let mut bounds = smallvec![InvocationBounds::Never; self.num_regions()];
            let selected =
                self.get_case_index_for_selector(selector).map(|idx| idx + 1).unwrap_or(0);
            bounds[selected] = InvocationBounds::Exact(1);
            bounds
        } else {
            // Only one region is invoked, and no more than a single time
            smallvec![InvocationBounds::NoMoreThan(1); self.num_regions()]
        }
    }

    #[inline(always)]
    fn is_repetitive_region(&self, _index: usize) -> bool {
        false
    }

    #[inline(always)]
    fn has_loop(&self) -> bool {
        false
    }
}

impl Canonicalizable for IndexSwitch {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(crate::canonicalization::FoldConstantIndexSwitch::new(context.clone()));
        rewrites.push(crate::canonicalization::FoldRedundantYields::new(context.clone()));
        rewrites.push(crate::canonicalization::IndexSwitchRemoveUnusedResults::new(context));
    }
}

/// The [Condition] op is used in conjunction with [While] as the terminator of its `before` region.
///
/// This op represents a choice between continuing the loop, or exiting the [While] loop and
/// continuing execution after the loop.
///
/// NOTE: Attempting to use this op in any other context than the one described above is invalid,
/// and the implementation of various interfaces by this op will panic if that assumption is
/// violated.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[effects(MemoryEffect())]
#[operation(
    dialect = ScfDialect,
    traits(Terminator, ReturnLike),
    implements(
        RegionBranchTerminatorOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct Condition {
    #[operand]
    condition: Bool,
    #[operands]
    forwarded: AnyType,
}

impl RegionBranchTerminatorOpInterface for Condition {
    #[inline]
    fn get_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.forwarded())
    }

    #[inline]
    fn get_mutable_successor_operands(
        &mut self,
        _point: RegionBranchPoint,
    ) -> SuccessorOperandRangeMut<'_> {
        SuccessorOperandRangeMut::forward(self.forwarded_mut())
    }

    fn get_successor_regions(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> SmallVec<[RegionSuccessorInfo; 2]> {
        // A [While] loop has two regions: `before` (containing this op), and `after`, which this
        // op branches to when the condition is true. If the condition is false, control is
        // transferred back to the parent [While] operation, with the forwarded operands of the
        // condition used as the results of the [While] operation.
        //
        // We can return a single statically-known region if we were given a constant condition
        // value, otherwise we must return both possible regions.
        let cond = operands[0].as_ref().and_then(|v| v.borrow().as_bool());
        let mut regions = SmallVec::<[RegionSuccessorInfo; 2]>::default();

        // Under any other parent this op states no successor; the region-branch terminator
        // verifier reports it.
        let Some(parent_op) = self.parent_op() else {
            return regions;
        };
        let parent_op = parent_op.borrow();
        let Some(while_op) = parent_op.downcast_ref::<While>() else {
            return regions;
        };
        let after_region = while_op.after().as_region_ref();

        // We can't know the condition until runtime, so both the parent `while` op and
        match cond {
            None => {
                regions.push(RegionSuccessorInfo::Entering(after_region));
                regions.push(RegionSuccessorInfo::Returning(
                    while_op.results().all().iter().map(|r| r.borrow().as_value_ref()).collect(),
                ));
            }
            Some(true) => {
                regions.push(RegionSuccessorInfo::Entering(after_region));
            }
            Some(false) => {
                regions.push(RegionSuccessorInfo::Returning(
                    while_op.results().all().iter().map(|r| r.borrow().as_value_ref()).collect(),
                ));
            }
        }

        regions
    }
}

/// The [Yield] op is used in conjunction with [If] and [While] ops as a return-like terminator.
///
/// * With [If], its regions must be terminated with either a [Yield] or an `Unreachable` op.
/// * With [While], a [Yield] is only valid in the `after` region, and the yielded operands must
///   match the region arguments of the `before` region. Thus to return values from the body of a
///   loop, one must first yield them from the `after` region to the `before` region using [Yield],
///   and then yield them from the `before` region by passsing them as forwarded operands of the
///   [Condition] op.
///
/// Any number of operands can be yielded at the same time. However, when [Yield] is used in
/// conjunction with [While], the arity and type of the operands must match the region arguments
/// of the `before` region. When used in conjunction with [If], both the `if_true` and `if_false`
/// regions must yield the same arity and types.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[effects(MemoryEffect())]
#[operation(
    dialect = ScfDialect,
    traits(Terminator, ReturnLike, Pure, AlwaysSpeculatable),
    implements(
        RegionBranchTerminatorOpInterface,
        MemoryEffectOpInterface,
        ConditionallySpeculatable,
        OpPrinter,
    )
)]
pub struct Yield {
    #[operands]
    yielded: AnyType,
}

impl RegionBranchTerminatorOpInterface for Yield {
    #[inline]
    fn get_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.yielded())
    }

    fn get_mutable_successor_operands(
        &mut self,
        _point: RegionBranchPoint,
    ) -> SuccessorOperandRangeMut<'_> {
        SuccessorOperandRangeMut::forward(self.yielded_mut())
    }

    fn get_successor_regions(
        &self,
        _operands: &[Option<AttributeRef>],
    ) -> SmallVec<[RegionSuccessorInfo; 2]> {
        // Depending on the type of operation containing this yield, the set of successor regions
        // is always known.
        //
        // * [While] may only have a yield to its `before` region
        // * [If] may only yield to its parent
        // * [IndexSwitch] may only yield to its parent
        //
        // Under any other parent this op states no successor; the region-branch terminator
        // verifier reports it.
        let Some(parent_op) = self.parent_op() else {
            return SmallVec::new();
        };
        let parent_op = parent_op.borrow();
        if parent_op.is::<If>() || parent_op.is::<IndexSwitch>() {
            smallvec![RegionSuccessorInfo::Returning(
                parent_op.results().all().iter().map(|v| v.borrow().as_value_ref()).collect()
            )]
        } else if let Some(while_op) = parent_op.downcast_ref::<While>() {
            let before_region = while_op.before().as_region_ref();
            smallvec![RegionSuccessorInfo::Entering(before_region)]
        } else {
            SmallVec::new()
        }
    }
}

impl ConditionallySpeculatable for Yield {
    fn speculatability(&self) -> Speculatability {
        Speculatability::Speculatable
    }
}

/// Resolve the operation an early exit in `op` addresses, walking outward and counting receivers.
///
/// This is the whole of the `levels` mechanism, and it is deliberately one function shared by the
/// verifier and the successor-region computations: two answers to "which op does this exit
/// leave?" that disagreed would be worse than one that is wrong. It is also the implementation of
/// [RegionExitTerminatorOpInterface::resolve_exit] for both exit ops, which is what lets the
/// interface's verifier check rules 2-7 over the ANSWER without knowing that `levels` exists.
///
/// # RULE 1 (`levels >= 1`) IS DETECTED HERE AND NOWHERE ELSE
///
/// Every other rule of [RegionExitTerminatorOpInterface]'s contract is verified by the
/// interface's own verifier, for every implementor. Rule 1 cannot be: *"the name is well-formed"*
/// is a property of THIS encoding — a token operand or a label would be malformed in some other
/// way — and the interface has no vocabulary for it. What the interface can do is require the
/// malformedness to be **reported**, which is what [ExitDefectKind::Malformed] below is; what it
/// cannot do is detect one. So a resolver that silently treated `levels = 0` (or a negative
/// `levels`) as "not found" would produce the wrong diagnostic and nothing would catch it. The rule
/// stays here, and it stays stated.
fn resolve_levels(op: &Operation, levels: i32, kind: RegionExitKind) -> ExitResolution {
    // Rule 1. The wording is this encoding's own, and the verifier renders it verbatim.
    if levels < 1 {
        return ExitResolution::Unresolved(ExitDefect {
            kind: ExitDefectKind::Malformed,
            detail: alloc::format!(
                "`levels` is 1-based and counts enclosing loops, so it must be at least 1, but \
                 this {kind} names level {levels}"
            ),
        });
    }
    let levels = levels as u32;

    let not_found = |found: u32| {
        ExitResolution::Unresolved(ExitDefect {
            kind: ExitDefectKind::NotFound,
            detail: alloc::format!(
                "this {kind} names enclosing loop {levels}, but only {found} enclosing loop(s) \
                 were found before the enclosing function body was reached"
            ),
        })
    };

    let mut remaining = levels;
    let mut from_region = match op.parent_region() {
        Some(region) => region,
        None => return not_found(0),
    };
    loop {
        let Some(parent_ref) = from_region.borrow().parent() else {
            return not_found(levels - remaining);
        };
        let parent = parent_ref.borrow();

        // A break may not escape a function: the enclosing callable body is the outer bound of
        // the search, and it is checked BEFORE the receiver test so that a callable which is also
        // somehow loop-like cannot be addressed from outside its own body.
        if parent.implements::<dyn CallableOpInterface>() {
            return not_found(levels - remaining);
        }

        if parent.implements::<dyn HasBreakingControlFlowOpInterface>() {
            remaining -= 1;
            if remaining == 0 {
                return ExitResolution::Resolved(ResolvedExit {
                    target: parent_ref,
                    from_region,
                    // VACUOUS UNDER `levels`, AND `None` IS THE HONEST ANSWER. The search above
                    // stops at the first `CallableOpInterface` ancestor BEFORE testing for a
                    // receiver, so a resolved target is inside this exit's own callable by
                    // construction — there is no fact here the verifier does not already have,
                    // and paying an O(depth) walk on every `get_exit_target()` call to restate a
                    // structural one would be a cost with no check behind it. See
                    // [ResolvedExit::callable] for when the field stops being vacuous.
                    callable: None,
                });
            }
        }

        // Every op the exit passes THROUGH must have opted in to forwarding it. This is the
        // safety property: an op that has not declared transparency makes the IR *invalid* rather
        // than silently mis-analyzed, and it is what makes the set of ops an early exit can cross
        // a closed, declared set rather than whatever happens to be nested at the time.
        if !parent.implements::<dyn PropagateControlFlowBreak>() {
            return ExitResolution::Unresolved(ExitDefect {
                detail: alloc::format!(
                    "this {kind} would have to cross '{}', which has not declared that it \
                     forwards breaking control flow",
                    parent.name()
                ),
                kind: ExitDefectKind::Opaque {
                    crossing: parent_ref,
                },
            });
        }

        let Some(next) = parent.parent_region() else {
            return not_found(levels - remaining);
        };
        from_region = next;
    }
}

/// The target of an exit in `op`, or `None` if it does not resolve — the successor computations'
/// half of [resolve_levels].
fn exit_target(op: &Operation, levels: i32, kind: RegionExitKind) -> Option<OperationRef> {
    match resolve_levels(op, levels, kind) {
        ExitResolution::Resolved(resolved) => Some(resolved.target),
        ExitResolution::Unresolved(_) => None,
    }
}

/// [Break] transfers control out of the `levels`-th enclosing loop, past every intermediate op.
///
/// `levels` is 1-based and counts enclosing ops which receive breaking control flow (i.e.
/// [While]): `levels = 1` names the innermost enclosing loop, `levels = 2` the one outside it. It
/// is an `i32`; zero and negative values are rejected by the verifier.
///
/// The operands become the target loop's **results**, so their types must equal the target's
/// result types.
///
/// # A [Break] IN A LOOP'S `before` REGION
///
/// The verifier accepts it. It is equivalent to `scf.condition(false, args)`, and rule 6 — which
/// rejects a [Continue] in its own target's header — deliberately does not extend to a break.
/// `v6_rejects_continue_in_the_targets_header`'s accept side is the witness.
///
/// Effect-free: unwinding to an enclosing op and carrying values is neither a read nor a write.
/// Where it may appear is a matter of its terminator-ness — see the nested-terminator clause in
/// `Operation::would_be_trivially_dead_even_if_terminator`, without which this declaration
/// deletes the `scf.if` that carries the exit, and the exit with it.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = ScfDialect,
    traits(Terminator, ReturnLike),
    implements(
        RegionBranchTerminatorOpInterface,
        RegionExitTerminatorOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct Break {
    #[attr]
    levels: I32Attr,
    #[operands]
    args: AnyType,
}

/// [Continue] transfers control to the `levels`-th enclosing loop's header, past every
/// intermediate op.
///
/// `levels` is 1-based and counts enclosing loops, exactly as [Break]'s does.
///
/// The operands become the target loop's **next iter_args**, so their types must equal the
/// `before` region's entry-block argument types — i.e. exactly what [Yield] carries in the
/// `after` region.
///
/// Effect-free, for the same reason [`Break`] is.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = ScfDialect,
    traits(Terminator, ReturnLike),
    implements(
        RegionBranchTerminatorOpInterface,
        RegionExitTerminatorOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct Continue {
    #[attr]
    levels: I32Attr,
    #[operands]
    args: AnyType,
}

/// Print `scf.break`/`scf.continue` as `<op> <levels> <operands> : (<types>)`.
///
/// The level count is printed positionally, as a signed decimal, rather than in the attribute
/// dictionary; `scf.index_switch`'s `case <n>` has the same shape.
fn print_exit(op: &Operation, levels: i32, printer: &mut AsmPrinter<'_>) {
    use formatter::*;

    printer.print_space();
    *printer += display(levels);

    let operands = op.operands().all();
    let types = operands.iter().map(|o| o.borrow().ty()).collect::<SmallVec<[_; 2]>>();
    if !operands.is_empty() {
        printer.print_space();
        printer.print_value_uses(ValueRange::<2>::from(operands));
    }
    printer.print_space();
    printer.print_colon_type_list(types);
}

/// Parse `<levels> [<operands>] : (<types>)` into `state`, the inverse of [print_exit]. `levels`
/// is written the way an `i32` constant's literal is written (see [parse_i32]); the verifier
/// rejects a value below 1.
fn parse_exit(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
    let (_, levels) = parse_i32(parser)?;
    state.add_attribute("levels", parser.context_rc().create_attribute::<I32Attr, _>(levels));

    let mut operands = SmallVec::default();
    parser.parse_operand_list(
        &mut operands,
        runic_core::parse::Delimiter::None,
        /*allow_result_number=*/ true,
        None,
    )?;
    let mut types = SmallVec::<[Type; 4]>::default();
    parser.parse_colon_type_list(&mut types)?;
    for (operand, ty) in operands.into_iter().zip(types) {
        state.add_operand(parser.resolve_operand(operand, ty)?);
    }

    Ok(())
}

impl OpPrinter for Break {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        print_exit(self.as_operation(), *self.get_levels(), printer)
    }
}

impl OpParser for Break {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_exit(state, parser)
    }
}

impl OpPrinter for Continue {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        print_exit(self.as_operation(), *self.get_levels(), printer)
    }
}

impl OpParser for Continue {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_exit(state, parser)
    }
}

impl RegionExitTerminatorOpInterface for Break {
    #[inline]
    fn resolve_exit(&self) -> ExitResolution {
        resolve_levels(self.as_operation(), *self.get_levels(), RegionExitKind::Break)
    }

    #[inline(always)]
    fn get_exit_kind(&self) -> RegionExitKind {
        RegionExitKind::Break
    }
}

impl RegionExitTerminatorOpInterface for Continue {
    #[inline]
    fn resolve_exit(&self) -> ExitResolution {
        resolve_levels(self.as_operation(), *self.get_levels(), RegionExitKind::Continue)
    }

    #[inline(always)]
    fn get_exit_kind(&self) -> RegionExitKind {
        RegionExitKind::Continue
    }
}

impl RegionBranchTerminatorOpInterface for Break {
    #[inline]
    fn get_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.args())
    }

    #[inline]
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
        // A break has exactly one successor and it is statically known: control leaves the target
        // loop, whose results receive this op's operands.
        let Some(target) =
            exit_target(self.as_operation(), *self.get_levels(), RegionExitKind::Break)
        else {
            // Malformed IR — the verifier rejects an exit that does not resolve, so this is
            // VERIFIER-GUARDED LATENCY rather than a conservative answer, and it is worth being
            // precise about which.
            //
            // An empty successor set is the BLIND state, not the unknown one: it is exactly the
            // mutation N-1 uses to prove the framework can tell correct from blind, and an
            // analysis reading it concludes that what follows the exit is unreachable — a *wrong*
            // answer, not a safe one. `RegionExitTerminatorOpInterface::get_exit_target`'s own
            // doc says the same thing from the other side: a `None` there means "unknown control
            // flow", never "no control flow". The interface has no representation for unknown
            // successors, so this returns nothing and relies on the verifier having already
            // rejected the program. A guessed target would be silently wrong in a worse way.
            return SmallVec::new();
        };
        let values = target
            .borrow()
            .results()
            .all()
            .iter()
            .map(|r| r.borrow().as_value_ref())
            .collect();
        smallvec![RegionSuccessorInfo::Exiting {
            target,
            to: RegionBranchPoint::Parent,
            values,
        }]
    }
}

impl RegionBranchTerminatorOpInterface for Continue {
    #[inline]
    fn get_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.args())
    }

    #[inline]
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
        let Some(target) =
            exit_target(self.as_operation(), *self.get_levels(), RegionExitKind::Continue)
        else {
            return SmallVec::new();
        };
        let target_op = target.borrow();
        let Some(header) = target_op
            .as_trait::<dyn HasBreakingControlFlowOpInterface>()
            .and_then(|receiver| receiver.get_continue_target_region())
        else {
            // Same verifier-guarded latency as above: a receiver with no continue target cannot
            // be addressed by a `scf.continue` in verified IR.
            return SmallVec::new();
        };
        let values = header
            .borrow()
            .entry()
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

#[cfg(test)]
mod tests {
    use runic_core::{
        diagnostics::Report,
        dialects::{
            builtin::{Function, attributes::Signature},
            test::TestDialect,
        },
        testing::parse_function_fixpoint,
    };
    use runic_expect_test::expect;

    use super::*;

    #[test]
    fn nested_break_can_return_a_loop_result() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = r#"
builtin.function public extern("C") @exit(%flag: i1, %value: i32) -> i32 {
    %result = scf.while %value before {
    ^head(%current: i32):
        scf.condition %flag, %current : (i1, i32);
    } after {
    ^body(%next: i32):
        scf.if %flag then {
            scf.break 1 %next : (i32);
        } else {
            scf.yield : ();
        } : (i1) -> ();
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %result : (i32);
};"#;
        parse_function_fixpoint(&context, "nested_break.hir", source)?;
        Ok(())
    }

    /// An op with a region that has **not** declared [PropagateControlFlowBreak].
    ///
    /// It exists only for V-3, and it has to exist because the tree has no such op today: the
    /// region-bearing ops that can nest inside a function body are exactly `scf.if`,
    /// `scf.index_switch` and `scf.while`, and all three opted in. Rule 3's whole job is to keep
    /// that true as the tree grows, so its witness has to bring its own non-member — otherwise
    /// the rule would be "tested" by a set that cannot violate it.
    #[derive(OpPrinter, OpParser, EffectOpInterface)]
    #[effects(MemoryEffect())]
    #[operation(
        dialect = TestDialect,
        traits(SingleBlock, HasRecursiveMemoryEffects),
        implements(MemoryEffectOpInterface, OpPrinter)
    )]
    pub struct OpaqueRegion {
        #[region]
        body: Region,
    }

    /// [OpaqueRegion]'s twin, identical in every respect except that it declares
    /// [PropagateControlFlowBreak].
    ///
    /// The pair is what makes V-3 a test of the trait rather than of anything else about the two
    /// programs: the accept and reject programs differ in exactly one token, and that token names
    /// an op whose only difference is the marker.
    #[derive(OpPrinter, OpParser, EffectOpInterface)]
    #[effects(MemoryEffect())]
    #[operation(
        dialect = TestDialect,
        traits(
            SingleBlock,
            HasRecursiveMemoryEffects,
            PropagateControlFlowBreak
        ),
        implements(MemoryEffectOpInterface, OpPrinter)
    )]
    pub struct TransparentRegion {
        #[region]
        body: Region,
    }

    /// An exit whose resolver **forgets rule 3**: it counts outward to the first receiver without
    /// ever asking whether the ops it passed through declared [PropagateControlFlowBreak].
    ///
    /// It exists to prove that the interface's blanket verifier CHECKS the resolver rather than
    /// restating it. `scf.break` in the same position reports the crossing itself, as an
    /// [ExitDefectKind::Opaque] defect, and is rejected through the defect arm (V-3). This op
    /// reports a confident `Resolved` and is rejected through the RE-DERIVATION arm, with the
    /// same rule and the same wording — which is the whole claim of variant (B).
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
    pub struct BlindExit {
        #[operands]
        args: AnyType,
    }

    /// An exit whose resolver **forgets rule 2's receiver test**: it names the nearest enclosing
    /// region-bearing op, whatever that op is.
    ///
    /// Placed directly in a loop body its answer is right, and the verifier accepts it; placed one
    /// region deeper it names an op that receives no breaking control flow, and the verifier
    /// rejects it. The pair is what makes this a test of the CHECK rather than of the op.
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
    pub struct StrayExit {
        #[operands]
        args: AnyType,
    }

    /// An exit whose target and originating region are **right**, and whose `callable` is wrong:
    /// it names the receiver it found instead of the callable it lives in.
    ///
    /// This is the witness that the verifier checks `ResolvedExit::callable`. [BlindExit] in the
    /// identical program position reports the identical target and `from_region` with
    /// `callable: None` and is ACCEPTED — so the two resolutions differ in exactly one field, and
    /// that field is what the verifier is reacting to.
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
    pub struct DisownedExit {
        #[operands]
        args: AnyType,
    }

    /// The nearest enclosing region and its parent op, or `None` at the top of the IR.
    fn nearest_enclosing(op: &Operation) -> Option<(RegionRef, OperationRef)> {
        let region = op.parent_region()?;
        let parent = region.borrow().parent()?;
        Some((region, parent))
    }

    /// The honest walk [BlindExit] and [DisownedExit] share: outward to the first receiver, with
    /// no transparency test, which is enough for both of their programs.
    fn nearest_receiver(op: &Operation) -> Option<(RegionRef, OperationRef)> {
        let mut region = op.parent_region();
        while let Some(current) = region {
            let parent_ref = current.borrow().parent()?;
            if parent_ref.borrow().implements::<dyn HasBreakingControlFlowOpInterface>() {
                return Some((current, parent_ref));
            }
            region = parent_ref.borrow().parent_region();
        }
        None
    }

    #[derive(OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator, ReturnLike),
        implements(RegionExitTerminatorOpInterface, RegionBranchTerminatorOpInterface, OpPrinter)
    )]
    pub struct WrongOriginExit {
        #[operands]
        args: AnyType,
    }

    /// Implements the Rust trait but omits its dynamic registration, which analyses require.
    #[derive(OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator, ReturnLike),
        implements(RegionExitTerminatorOpInterface, OpPrinter)
    )]
    pub struct UnregisteredBranchExit {
        #[operands]
        args: AnyType,
    }

    impl RegionExitTerminatorOpInterface for UnregisteredBranchExit {
        fn resolve_exit(&self) -> ExitResolution {
            let (from_region, target) = nearest_receiver(self.as_operation()).unwrap();
            ExitResolution::Resolved(ResolvedExit {
                target,
                from_region,
                callable: None,
            })
        }

        fn get_exit_kind(&self) -> RegionExitKind {
            RegionExitKind::Break
        }
    }

    macro_rules! impl_test_break_successors {
        ($op:ty) => {
            impl RegionBranchTerminatorOpInterface for $op {
                fn get_successor_operands(
                    &self,
                    _point: RegionBranchPoint,
                ) -> SuccessorOperandRange<'_> {
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
                    let target = self.get_exit_target().unwrap();
                    let values = target
                        .borrow()
                        .results()
                        .iter()
                        .map(|result| result.borrow().as_value_ref())
                        .collect();
                    smallvec![RegionSuccessorInfo::Exiting {
                        target,
                        to: RegionBranchPoint::Parent,
                        values,
                    }]
                }
            }
        };
    }

    impl_test_break_successors!(WrongOriginExit);
    impl_test_break_successors!(UnregisteredBranchExit);

    /// A callable that also registers as an exit receiver, to test boundary-check precedence.
    #[derive(OpPrinter, OpParser)]
    #[operation(
        dialect = TestDialect,
        traits(SingleBlock),
        implements(CallableOpInterface, RegionBranchOpInterface, HasBreakingControlFlowOpInterface, OpPrinter)
    )]
    pub struct CallableReceiver {
        #[region]
        body: Region,
    }

    impl CallableOpInterface for CallableReceiver {
        fn get_callable_region(&self) -> Option<RegionRef> {
            Some(self.body().as_region_ref())
        }

        fn signature(&self) -> Signature {
            Signature::new(&self.as_operation().context_rc(), [], [])
        }
    }

    impl RegionBranchOpInterface for CallableReceiver {
        fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
            let successor = match point {
                RegionBranchPoint::Parent => {
                    RegionSuccessorInfo::Entering(self.body().as_region_ref())
                }
                RegionBranchPoint::Child(_) => RegionSuccessorInfo::Returning(SmallVec::new()),
            };
            RegionSuccessorIter::new(self.as_operation(), [successor])
        }
    }

    impl HasBreakingControlFlowOpInterface for CallableReceiver {
        fn has_nested_predecessors(&self) -> bool {
            true
        }

        fn get_continue_target_region(&self) -> Option<RegionRef> {
            None
        }
    }

    #[test]
    fn exit_verifier_requires_registered_branch_terminator_interface() {
        let error = rejection(
            "unregistered_branch_exit.hir",
            &RESOLVER_CHECK_PROGRAM.replace("BODY", "test.unregistered_branch_exit %j : (i32);"),
        );
        assert!(error.contains("RegionBranchTerminatorOpInterface"), "{error}");
    }

    #[test]
    fn exit_verifier_rejects_a_callable_receiver_as_target() {
        let error = rejection(
            "callable_receiver_exit.hir",
            r#"
builtin.function public extern("C") @f() {
    test.callable_receiver {
        test.stray_exit : ();
    };
    builtin.ret;
};"#,
        );
        assert!(error.contains("callable boundary"), "{error}");
    }

    impl RegionExitTerminatorOpInterface for WrongOriginExit {
        fn resolve_exit(&self) -> ExitResolution {
            let (_, target) = nearest_receiver(self.as_operation()).unwrap();
            let from_region = target
                .borrow()
                .as_trait::<dyn HasBreakingControlFlowOpInterface>()
                .unwrap()
                .get_continue_target_region()
                .unwrap();
            ExitResolution::Resolved(ResolvedExit {
                target,
                from_region,
                callable: None,
            })
        }

        fn get_exit_kind(&self) -> RegionExitKind {
            RegionExitKind::Break
        }
    }

    #[test]
    fn exit_verifier_checks_the_resolvers_originating_region() {
        let error = rejection(
            "wrong_exit_origin.hir",
            &RESOLVER_CHECK_PROGRAM.replace("BODY", "test.wrong_origin_exit %j : (i32);"),
        );
        assert!(error.contains("originating region"), "{error}");
    }

    impl RegionExitTerminatorOpInterface for BlindExit {
        fn resolve_exit(&self) -> ExitResolution {
            match nearest_receiver(self.as_operation()) {
                Some((from_region, target)) => ExitResolution::Resolved(ResolvedExit {
                    target,
                    from_region,
                    callable: None,
                }),
                None => ExitResolution::Unresolved(ExitDefect {
                    kind: ExitDefectKind::NotFound,
                    detail: alloc::string::ToString::to_string("no enclosing receiver"),
                }),
            }
        }

        fn get_exit_kind(&self) -> RegionExitKind {
            RegionExitKind::Break
        }
    }

    impl RegionExitTerminatorOpInterface for DisownedExit {
        fn resolve_exit(&self) -> ExitResolution {
            match nearest_receiver(self.as_operation()) {
                Some((from_region, target)) => ExitResolution::Resolved(ResolvedExit {
                    target,
                    from_region,
                    // THE LIE, and the only one: the receiver is not the callable.
                    callable: Some(target),
                }),
                None => ExitResolution::Unresolved(ExitDefect {
                    kind: ExitDefectKind::NotFound,
                    detail: alloc::string::ToString::to_string("no enclosing receiver"),
                }),
            }
        }

        fn get_exit_kind(&self) -> RegionExitKind {
            RegionExitKind::Break
        }
    }

    impl RegionExitTerminatorOpInterface for StrayExit {
        fn resolve_exit(&self) -> ExitResolution {
            match nearest_enclosing(self.as_operation()) {
                Some((from_region, target)) => ExitResolution::Resolved(ResolvedExit {
                    target,
                    from_region,
                    callable: None,
                }),
                None => ExitResolution::Unresolved(ExitDefect {
                    kind: ExitDefectKind::NotFound,
                    detail: alloc::string::ToString::to_string("no enclosing op"),
                }),
            }
        }

        fn get_exit_kind(&self) -> RegionExitKind {
            RegionExitKind::Break
        }
    }

    impl RegionBranchTerminatorOpInterface for BlindExit {
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
            SmallVec::new()
        }
    }

    impl RegionBranchTerminatorOpInterface for StrayExit {
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
            SmallVec::new()
        }
    }

    impl RegionBranchTerminatorOpInterface for DisownedExit {
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
            SmallVec::new()
        }
    }

    /// Parse `source` and return the error it must be rejected with.
    #[track_caller]
    fn rejection(name: &str, source: &str) -> alloc::string::String {
        let context = Rc::new(Context::default());
        match parse_function_fixpoint(&context, name, source) {
            Ok(_) => panic!("expected '{name}' to be rejected, but it was accepted"),
            Err(err) => alloc::format!("{err}"),
        }
    }

    /// Find the first operation of type `T` in the entry block of `function`.
    fn find_op<T: OpRegistration>(function: &Function) -> UnsafeIntrusiveEntityRef<T> {
        function
            .body()
            .entry()
            .body()
            .iter()
            .find_map(|op| op.as_operation_ref().try_downcast_op::<T>().ok())
            .unwrap_or_else(|| {
                panic!("expected a {} op in the function body", <T as OpRegistration>::full_name())
            })
    }

    /// The regions of an operation with operands must parse with no pre-bound entry arguments:
    /// the `^block(...)` header the printer emits declares them. Also covers the variadic
    /// forwarded operands of `scf.condition`, which parse as the trailing comma list following
    /// the condition operand, and the `scf.while` result type signature.
    #[test]
    fn parse_scf_while_round_trips() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @count_to(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %count = scf.while %zero before {
    ^head(%i: i32):
        %continue = arith.cmpi ult %i, %n;
        scf.condition %continue, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %one = arith.constant 1 : i32;
        %next = arith.add %j, %one;
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %count : (i32);
};";
        let (function, printed) = parse_function_fixpoint(&context, "parse_scf_while.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @count_to(%0: i32) -> i32 {
                %1 = arith.constant 0 : i32;
                %7 = scf.while %1 before {
                ^block2(%2: i32):
                    %3 = arith.cmpi ult %2, %0;
                    scf.condition %3, %2 : (i1, i32);
                } after {
                ^block3(%4: i32):
                    %5 = arith.constant 1 : i32;
                    %6 = arith.add %4, %5;
                    scf.yield %6 : (i32);
                } : (i32) -> (i32);
                builtin.ret %7 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let while_op = find_op::<While>(&function);
        let while_op = while_op.borrow();
        assert_eq!(while_op.inits().len(), 1);
        assert_eq!(while_op.num_results(), 1);
        assert_eq!(while_op.before().entry().num_arguments(), 1);
        let condition = while_op.condition_op().unwrap();
        let condition = condition.borrow();
        assert_eq!(condition.forwarded().len(), 1);

        Ok(())
    }

    /// The trailing forwarded-operand list of `scf.condition` may be empty, in which case the
    /// printer omits it entirely; the parser must accept the bare form.
    #[test]
    fn parse_scf_condition_without_forwarded_operands() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @spin(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    scf.while %zero before {
    ^head(%i: i32):
        %continue = arith.cmpi ult %i, %n;
        scf.condition %continue : (i1);
    } after {
    ^body:
        scf.yield %zero : (i32);
    } : (i32) -> ();
    builtin.ret %n : (i32);
};";
        let (function, printed) =
            parse_function_fixpoint(&context, "parse_scf_condition_bare.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @spin(%0: i32) -> i32 {
                %1 = arith.constant 0 : i32;
                scf.while %1 before {
                ^block2(%2: i32):
                    %3 = arith.cmpi ult %2, %0;
                    scf.condition %3 : (i1);
                } after {
                ^block3:
                    scf.yield %1 : (i32);
                } : (i32) -> ();
                builtin.ret %0 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let while_op = find_op::<While>(&function);
        let while_op = while_op.borrow();
        assert_eq!(while_op.num_results(), 0);
        let condition = while_op.condition_op().unwrap();
        let condition = condition.borrow();
        assert_eq!(condition.forwarded().len(), 0);

        Ok(())
    }

    /// Multi-name result bindings (`%x, %y = scf.if ...`) map the parsed names onto the
    /// flattened result list of a single variadic result group.
    #[test]
    fn parse_multi_name_result_bindings() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @pick(%c: i1, %a: i32, %b: i32) -> i32 {
    %x, %y = scf.if %c then {
        scf.yield %a, %b : (i32, i32);
    } else {
        scf.yield %b, %a : (i32, i32);
    } : (i1) -> (i32, i32);
    %sum = arith.add %x, %y <{ overflow = #builtin.overflow<unchecked unsigned> }>;
    builtin.ret %sum : (i32);
};";
        let (function, printed) =
            parse_function_fixpoint(&context, "parse_multi_result.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @pick(%0: i1, %1: i32, %2: i32) -> i32 {
                %3, %4 = scf.if %0 then {
                    scf.yield %1, %2 : (i32, i32);
                } else {
                    scf.yield %2, %1 : (i32, i32);
                } : (i1) -> (i32, i32);
                %5 = arith.add %3, %4 <{ overflow = #builtin.overflow<unchecked unsigned> }>;
                builtin.ret %5 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let if_op = find_op::<If>(&function);
        let if_op = if_op.borrow();
        assert_eq!(if_op.num_results(), 2);
        // Both bound names resolve to the if's results: the adder uses each exactly once.
        for result in if_op.results().all().iter() {
            assert_eq!(result.borrow().iter_uses().count(), 1);
        }

        Ok(())
    }

    /// An op with more than 255 results prints and re-parses, one SSA name per result.
    #[test]
    fn parse_wide_result_list() -> Result<(), Report> {
        const NUM_RESULTS: usize = 256;

        let context = Rc::new(Context::default());
        let names = (0..NUM_RESULTS).map(|i| format!("%r{i}")).collect::<Vec<_>>().join(", ");
        let values = ["%a"; NUM_RESULTS].join(", ");
        let types = ["i32"; NUM_RESULTS].join(", ");
        let source = format!(
            "\
builtin.function public extern(\"C\") @wide(%c: i1, %a: i32) -> i32 {{
    {names} = scf.if %c then {{
        scf.yield {values} : ({types});
    }} else {{
        scf.yield {values} : ({types});
    }} : (i1) -> ({types});
    builtin.ret %r{last} : (i32);
}};",
            last = NUM_RESULTS - 1
        );
        let (function, _printed) =
            parse_function_fixpoint(&context, "parse_wide_result_list.hir", &source)?;

        let function = function.borrow();
        let if_op = find_op::<If>(&function);
        assert_eq!(if_op.borrow().num_results(), NUM_RESULTS);

        Ok(())
    }

    /// `scf.index_switch` regions parse with the default region first, matching the accessors
    /// (`default_region` is region 0, case regions follow in case order).
    #[test]
    fn parse_index_switch_region_order() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @dispatch(%sel: i32, %a: i32, %b: i32) -> i32 {
    %r = scf.index_switch %sel
    case 1 {
        scf.yield %a : (i32);
    }
    default {
        scf.yield %b : (i32);
    } : i32;
    builtin.ret %r : (i32);
};";
        let (function, printed) =
            parse_function_fixpoint(&context, "parse_index_switch.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @dispatch(%0: i32, %1: i32, %2: i32) -> i32 {
                %3 = scf.index_switch %0 
                case 1 {
                    scf.yield %1 : (i32);
                }
                default {
                    scf.yield %2 : (i32);
                } : (i32);
                builtin.ret %3 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let body = function.body();
        let entry = body.entry();
        let arg_a = entry.arguments()[1] as ValueRef;
        let arg_b = entry.arguments()[2] as ValueRef;

        let yielded_value = |region: &Region| -> ValueRef {
            let terminator = region.entry().terminator().unwrap();
            let yield_op = terminator
                .try_downcast_op::<Yield>()
                .expect("expected region to terminate with scf.yield");
            let yield_op = yield_op.borrow();
            let yielded = yield_op.yielded();
            let operand = yielded.iter().next().unwrap();
            operand.borrow().as_value_ref()
        };

        let switch_op = find_op::<IndexSwitch>(&function);
        let switch_op = switch_op.borrow();
        // The default region yields %b and the `case 1` region yields %a; if the parser
        // appended the default region last, the two would come back swapped.
        assert_eq!(yielded_value(&switch_op.default_region()), arg_b);
        let case_region = switch_op.get_case_region(0);
        assert_eq!(yielded_value(&case_region.borrow()), arg_a);

        Ok(())
    }
    /// The round-trip fixture for the new ops: a `break` nested in an `scf.if` inside the loop it
    /// exits, printed and re-parsed to a fixpoint.
    #[test]
    fn parse_scf_break_round_trips() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @first_over(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %found = scf.while %zero before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %stop = arith.cmpi ugt %j, %n;
        %k = scf.if %stop then {
            scf.break 1 %j : (i32);
        } else {
            scf.yield %j : (i32);
        } : (i1) -> (i32);
        %one = arith.constant 1 : i32;
        %next = arith.add %k, %one;
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %found : (i32);
};";
        let (function, printed) = parse_function_fixpoint(&context, "scf_break.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @first_over(%0: i32) -> i32 {
                %1 = arith.constant 0 : i32;
                %9 = scf.while %1 before {
                ^block2(%2: i32):
                    %3 = arith.cmpi ult %2, %0;
                    scf.condition %3, %2 : (i1, i32);
                } after {
                ^block3(%4: i32):
                    %5 = arith.cmpi ugt %4, %0;
                    %6 = scf.if %5 then {
                        scf.break 1 %4 : (i32);
                    } else {
                        scf.yield %4 : (i32);
                    } : (i1) -> (i32);
                    %7 = arith.constant 1 : i32;
                    %8 = arith.add %6, %7;
                    scf.yield %8 : (i32);
                } : (i32) -> (i32);
                builtin.ret %9 : (i32);
            };"#]]
        .assert_eq(&printed);

        // The break resolves through the `scf.if` to the loop, not to the `scf.if`.
        let function = function.borrow();
        let while_op = find_op::<While>(&function);
        let break_op = {
            let while_op = while_op.borrow();
            let if_op = while_op
                .after()
                .entry()
                .body()
                .iter()
                .find_map(|op| op.as_operation_ref().try_downcast_op::<If>().ok())
                .expect("expected an scf.if in the loop body");
            let if_op = if_op.borrow();
            if_op
                .then_body()
                .entry()
                .terminator()
                .unwrap()
                .try_downcast_op::<Break>()
                .unwrap()
        };
        assert_eq!(
            break_op.borrow().get_exit_target(),
            Some(while_op.borrow().as_operation().as_operation_ref())
        );
        assert_eq!(break_op.borrow().get_exit_kind(), RegionExitKind::Break);

        Ok(())
    }

    /// `scf.continue` round-trips, and resolves through an intermediate `scf.if` to the loop's
    /// header rather than to the `scf.if`.
    #[test]
    fn parse_scf_continue_round_trips() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @skip_odds(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %total = scf.while %zero before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %odd = arith.cmpi ugt %j, %n;
        %k = scf.if %odd then {
            %bump_one = arith.constant 1 : i32;
            %bump = arith.add %j, %bump_one;
            scf.continue 1 %bump : (i32);
        } else {
            scf.yield %j : (i32);
        } : (i1) -> (i32);
        %one = arith.constant 1 : i32;
        %next = arith.add %k, %one;
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %total : (i32);
};";
        let (function, printed) = parse_function_fixpoint(&context, "scf_continue.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @skip_odds(%0: i32) -> i32 {
                %1 = arith.constant 0 : i32;
                %11 = scf.while %1 before {
                ^block2(%2: i32):
                    %3 = arith.cmpi ult %2, %0;
                    scf.condition %3, %2 : (i1, i32);
                } after {
                ^block3(%4: i32):
                    %5 = arith.cmpi ugt %4, %0;
                    %8 = scf.if %5 then {
                        %6 = arith.constant 1 : i32;
                        %7 = arith.add %4, %6;
                        scf.continue 1 %7 : (i32);
                    } else {
                        scf.yield %4 : (i32);
                    } : (i1) -> (i32);
                    %9 = arith.constant 1 : i32;
                    %10 = arith.add %8, %9;
                    scf.yield %10 : (i32);
                } : (i32) -> (i32);
                builtin.ret %11 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let while_op = find_op::<While>(&function);
        let while_ref = while_op.borrow().as_operation().as_operation_ref();
        let continue_op = {
            let while_op = while_op.borrow();
            let if_op = while_op
                .after()
                .entry()
                .body()
                .iter()
                .find_map(|op| op.as_operation_ref().try_downcast_op::<If>().ok())
                .expect("expected an scf.if in the loop body");
            let if_op = if_op.borrow();
            if_op
                .then_body()
                .entry()
                .terminator()
                .unwrap()
                .try_downcast_op::<Continue>()
                .unwrap()
        };
        assert_eq!(continue_op.borrow().get_exit_target(), Some(while_ref));
        assert_eq!(continue_op.borrow().get_exit_kind(), RegionExitKind::Continue);

        // The successor names the loop's header, and it is an Exiting edge — not a Returning one
        // to the `scf.if` that lexically contains it.
        let continue_op = continue_op.borrow();
        let successors = <Continue as RegionBranchTerminatorOpInterface>::get_successor_regions(
            &continue_op,
            &[None],
        );
        assert_eq!(successors.len(), 1);
        match &successors[0] {
            RegionSuccessorInfo::Exiting { target, to, .. } => {
                assert_eq!(*target, while_ref);
                assert_eq!(
                    *to,
                    RegionBranchPoint::Child(while_op.borrow().before().as_region_ref())
                );
            }
            other => panic!("expected an Exiting successor, got {other:?}"),
        }

        Ok(())
    }

    /// **V-1** — rule 1: `levels` is 1-based, so zero names no loop.
    #[test]
    fn v1_rejects_levels_zero() {
        let err = rejection(
            "v1.hir",
            "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        scf.break 0 %j : (i32);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};",
        );
        assert!(err.contains("must be at least 1"), "unexpected diagnostic: {err}");
        assert!(err.contains("names level 0"), "unexpected diagnostic: {err}");
    }

    /// **V-2** — rule 2: `levels` must resolve. Two enclosing loops, and the exit names three.
    #[test]
    fn v2_rejects_more_levels_than_enclosing_loops() {
        let err = rejection(
            "v2.hir",
            "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^outer_head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^outer_body(%j: i32):
        %inner = scf.while %j before {
        ^inner_head(%k: i32):
            %go = arith.cmpi ult %k, %n;
            scf.condition %go, %k : (i1, i32);
        } after {
        ^inner_body(%m: i32):
            scf.break 3 %m : (i32);
        } : (i32) -> i32;
        scf.yield %inner : (i32);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};",
        );
        assert!(err.contains("names enclosing loop 3"), "unexpected diagnostic: {err}");
        assert!(
            err.contains("only 2 enclosing loop(s) were found"),
            "unexpected diagnostic: {err}"
        );
    }

    /// **V-3** — rule 3, the opt-in property, and the witness that proves the set is closed.
    ///
    /// The program is the accepted one with a single op swapped: `test.opaque_region` in place of
    /// `scf.if`. Both hold a region, both sit between the break and its target loop, and the ONLY
    /// difference is that one declares [PropagateControlFlowBreak] and the other does not. So the
    /// accept-side control below is not decoration — it is what makes this a test of membership
    /// rather than of nesting depth.
    #[test]
    fn v3_rejects_an_exit_crossing_an_op_that_has_not_opted_in() {
        const PROGRAM: &str = "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        INTERMEDIATE {
            scf.break 1 %j : (i32);
        };
        scf.yield %j : (i32);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};";

        let err = rejection("v3.hir", &PROGRAM.replace("INTERMEDIATE", "test.opaque_region"));
        assert!(
            err.contains("would have to cross 'test.opaque_region'"),
            "unexpected diagnostic: {err}"
        );
        assert!(
            err.contains("has not declared that it forwards breaking control flow"),
            "unexpected diagnostic: {err}"
        );

        // THE ACCEPT SIDE: the same program with one token changed — an op identical to the one
        // above except that it declares the marker.
        let context = Rc::new(Context::default());
        parse_function_fixpoint(
            &context,
            "v3_accept.hir",
            &PROGRAM.replace("INTERMEDIATE", "test.transparent_region"),
        )
        .expect("an exit crossing an op that declares PropagateControlFlowBreak must be accepted");
    }

    /// **V-4** — rule 4: a break's operands become the target loop's results.
    #[test]
    fn v4_rejects_break_operands_that_do_not_match_the_targets_results() {
        let err = rejection(
            "v4.hir",
            "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %flag = arith.cmpi ugt %j, %n;
        scf.break 1 %flag : (i1);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};",
        );
        assert!(
            err.contains("this break carries (i1), but the loop it exits produces (i32)"),
            "unexpected diagnostic: {err}"
        );
    }

    /// **V-5** — rule 5: a continue's operands become the target loop's next iter_args.
    #[test]
    fn v5_rejects_continue_operands_that_do_not_match_the_targets_iter_args() {
        let err = rejection(
            "v5.hir",
            "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %flag = arith.cmpi ugt %j, %n;
        scf.continue 1 %flag : (i1);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};",
        );
        assert!(
            err.contains("this continue carries (i1), but the loop it targets iterates on (i32)"),
            "unexpected diagnostic: {err}"
        );
    }

    /// **V-6** — rule 6: a continue in its own target's header re-enters the region it is in.
    ///
    /// The mirror case is legal and is asserted beside it: a `break` in the `before` region is
    /// equivalent to `scf.condition(false, args)` and costs nothing to allow.
    #[test]
    fn v6_rejects_continue_in_the_targets_header() {
        let err = rejection(
            "v6.hir",
            "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^head(%i: i32):
        scf.continue 1 %i : (i32);
    } after {
    ^body(%j: i32):
        scf.yield %j : (i32);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};",
        );
        assert!(
            err.contains("originates in the header of the loop it targets"),
            "unexpected diagnostic: {err}"
        );

        // A break in the header IS legal.
        let context = Rc::new(Context::default());
        parse_function_fixpoint(
            &context,
            "v6_accept.hir",
            "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^head(%i: i32):
        scf.break 1 %i : (i32);
    } after {
    ^body(%j: i32):
        scf.yield %j : (i32);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};",
        )
        .expect("a break in a loop's header is equivalent to `scf.condition(false, args)`");
    }

    /// **V-7** — rule 2's outer bound: a break may not escape a function.
    #[test]
    fn v7_rejects_a_break_with_no_enclosing_loop() {
        let err = rejection(
            "v7.hir",
            "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    scf.break 1 %n : (i32);
};",
        );
        assert!(err.contains("names enclosing loop 1"), "unexpected diagnostic: {err}");
        assert!(
            err.contains("only 0 enclosing loop(s) were found before the enclosing function body"),
            "unexpected diagnostic: {err}"
        );
    }

    /// V-3's program with the exit left as a hole, so that the resolver-check witnesses run on the
    /// SAME shape the rule's own witness runs on, and the only variable is which op sits in the
    /// hole and how deep.
    const RESOLVER_CHECK_PROGRAM: &str = "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        BODY
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};";

    /// The reject-side body: the exit one region deeper, inside `wrapper`.
    fn nested_body(wrapper: &str, exit: &str) -> alloc::string::String {
        alloc::format!(
            "{wrapper} {{\n            {exit} %j : (i32);\n        }};\n        scf.yield %j : \
             (i32);"
        )
    }

    /// **V-8** — the blanket CHECKS the resolver on rule 3, and the check is not the resolver's.
    ///
    /// [BlindExit] never asks whether it crossed a transparent op, so on the reject side it
    /// returns a confident `Resolved` naming the loop. The rejection therefore cannot come from
    /// the implementor, and it does not: it comes from the interface verifier's own
    /// re-derivation, in the same words rule 3 has always used.
    ///
    /// **The accept side is what makes this a test of the CHECK.** The same op, one region
    /// shallower, resolves correctly and is accepted — so the verifier is discriminating between
    /// two answers rather than refusing the op.
    #[test]
    fn v8_the_blanket_rejects_a_resolver_that_skipped_rule_3() {
        let err = rejection(
            "v8.hir",
            &RESOLVER_CHECK_PROGRAM
                .replace("BODY", &nested_body("test.opaque_region", "test.blind_exit")),
        );
        assert!(
            err.contains("would have to cross 'test.opaque_region'"),
            "unexpected diagnostic: {err}"
        );
        assert!(
            err.contains("has not declared that it forwards breaking control flow"),
            "unexpected diagnostic: {err}"
        );

        // THE ACCEPT SIDE: the same op where its answer happens to be right.
        let context = Rc::new(Context::default());
        parse_function_fixpoint(
            &context,
            "v8_accept.hir",
            &RESOLVER_CHECK_PROGRAM.replace("BODY", "test.blind_exit %j : (i32);"),
        )
        .expect("an exit whose resolver returns the right answer must be accepted");
    }

    /// **V-9** — the blanket CHECKS the resolver on rule 2's receiver half.
    ///
    /// [StrayExit] names the nearest enclosing region-bearing op. Inside a `test.transparent_region`
    /// that op forwards breaking control flow but does not RECEIVE it, so rule 3 cannot be what
    /// fires and rule 2 is. The accept side is the same op one region shallower, where the nearest
    /// enclosing op really is the loop.
    #[test]
    fn v9_the_blanket_rejects_a_resolver_that_named_a_non_receiver() {
        let err = rejection(
            "v9.hir",
            &RESOLVER_CHECK_PROGRAM
                .replace("BODY", &nested_body("test.transparent_region", "test.stray_exit")),
        );
        assert!(
            err.contains("resolved to 'test.transparent_region'"),
            "unexpected diagnostic: {err}"
        );
        assert!(
            err.contains("does not receive breaking control flow"),
            "unexpected diagnostic: {err}"
        );

        let context = Rc::new(Context::default());
        parse_function_fixpoint(
            &context,
            "v9_accept.hir",
            &RESOLVER_CHECK_PROGRAM.replace("BODY", "test.stray_exit %j : (i32);"),
        )
        .expect("an exit whose resolver returns the right answer must be accepted");
    }

    /// **V-10** — `ResolvedExit::callable` is CHECKED.
    ///
    /// [DisownedExit] and [BlindExit] resolve to the same target from the same region in the same
    /// program; the only difference between their two resolutions is that one names a `callable`
    /// and it is the wrong one. **That is what makes this a test of the field rather than of the
    /// op** — and it is why the field is not decoration for the counting encoding that reports
    /// `None`: an encoding that NAMES its target gets this check for free.
    #[test]
    fn v10_the_blanket_rejects_a_resolver_that_claimed_the_wrong_callable() {
        let err = rejection(
            "v10.hir",
            &RESOLVER_CHECK_PROGRAM.replace("BODY", "test.disowned_exit %j : (i32);"),
        );
        assert!(
            err.contains("resolved to a target in 'scf.while'"),
            "unexpected diagnostic: {err}"
        );
        assert!(
            err.contains("which is not the callable it lives in"),
            "unexpected diagnostic: {err}"
        );

        // THE CONTROL, in the same position: same target, same `from_region`, `callable: None`.
        let context = Rc::new(Context::default());
        parse_function_fixpoint(
            &context,
            "v10_control.hir",
            &RESOLVER_CHECK_PROGRAM.replace("BODY", "test.blind_exit %j : (i32);"),
        )
        .expect("an exit that leaves `callable` unreported must be accepted");
    }

    /// `scf.index_switch` cases are `i32`s written as signed decimals: `case -1` parses, prints
    /// back as `case -1`, and is the case value `-1`.
    #[test]
    fn index_switch_with_a_negative_case_round_trips() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @dispatch(%sel: i32, %a: i32, %b: i32) -> i32 {
    %r = scf.index_switch %sel
    case -1 {
        scf.yield %a : (i32);
    }
    case 2147483647 {
        scf.yield %b : (i32);
    }
    default {
        scf.yield %b : (i32);
    } : (i32);
    builtin.ret %r : (i32);
};";
        let (function, printed) =
            parse_function_fixpoint(&context, "parse_index_switch_negative.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @dispatch(%0: i32, %1: i32, %2: i32) -> i32 {
                %3 = scf.index_switch %0 
                case -1 {
                    scf.yield %1 : (i32);
                }
                case 2147483647 {
                    scf.yield %2 : (i32);
                }
                default {
                    scf.yield %2 : (i32);
                } : (i32);
                builtin.ret %3 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let switch_op = find_op::<IndexSwitch>(&function);
        let switch_op = switch_op.borrow();
        assert_eq!(switch_op.cases().iter().copied().collect::<Vec<_>>(), [-1, i32::MAX]);
        assert_eq!(switch_op.get_case_index_for_selector(-1), Some(0));
        assert_eq!(switch_op.get_case_index_for_selector(i32::MAX), Some(1));
        assert_eq!(switch_op.get_case_index_for_selector(0), None);

        Ok(())
    }

    /// A constant selector is compared with the cases as a 32-bit pattern: the `i32` constant
    /// `0xffffffff` is `-1`, so it selects `case -1` (and invokes only that region), while a
    /// selector matching no case selects the default region.
    #[test]
    fn index_switch_compares_a_constant_selector_as_a_32_bit_pattern() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @dispatch(%sel: i32, %a: i32, %b: i32) -> i32 {
    %r = scf.index_switch %sel
    case -1 {
        scf.yield %a : (i32);
    }
    default {
        scf.yield %b : (i32);
    } : (i32);
    builtin.ret %r : (i32);
};";
        let (function, _) =
            parse_function_fixpoint(&context, "index_switch_selector_pattern.hir", source)?;
        let function = function.borrow();
        let switch_op = find_op::<IndexSwitch>(&function);
        let switch_op = switch_op.borrow();
        let case_region = switch_op.get_case_region(0);
        let default_region = switch_op.default_region().as_region_ref();

        let all_ones = context.int_attr(Int::from_bits(32, 0xffff_ffff), context.i32());
        let entered = |selector: AttributeRef| {
            switch_op
                .get_entry_successor_regions(&[Some(selector)])
                .map(|successor| {
                    successor
                        .region_graph_successor()
                        .expect("an index_switch entry successor is a region")
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(entered(all_ones), [case_region]);
        assert_eq!(
            switch_op.get_region_invocation_bounds(&[Some(all_ones)]).as_slice(),
            [InvocationBounds::Never, InvocationBounds::Exact(1)]
        );

        let seven = context.int_attr(Int::from_bits(32, 7), context.i32());
        assert_eq!(entered(seven), [default_region]);
        assert_eq!(
            switch_op.get_region_invocation_bounds(&[Some(seven)]).as_slice(),
            [InvocationBounds::Exact(1), InvocationBounds::Never]
        );

        Ok(())
    }

    /// A case value may appear once; the second `case 1` is rejected.
    #[test]
    fn index_switch_rejects_a_duplicate_case() {
        let err = rejection(
            "index_switch_duplicate_case.hir",
            "\
builtin.function public extern(\"C\") @dispatch(%sel: i32, %a: i32) -> i32 {
    %r = scf.index_switch %sel
    case 1 {
        scf.yield %a : (i32);
    }
    case 1 {
        scf.yield %a : (i32);
    }
    default {
        scf.yield %a : (i32);
    } : (i32);
    builtin.ret %r : (i32);
};",
        );
        assert!(
            err.contains("invalid scf.index_switch operation"),
            "unexpected diagnostic: {err}"
        );
    }

    /// Rule 1 covers a negative `levels` too, with the same message as `levels = 0`.
    #[test]
    fn v1_rejects_a_negative_level() {
        let err = rejection(
            "v1_negative.hir",
            "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        scf.continue -1 %j : (i32);
    } : (i32) -> (i32);
    builtin.ret %r : (i32);
};",
        );
        assert!(err.contains("must be at least 1"), "unexpected diagnostic: {err}");
        assert!(err.contains("names level -1"), "unexpected diagnostic: {err}");
    }

    /// `levels` round-trips through text, and the builders store the `i32` they are given.
    #[test]
    fn exit_levels_round_trip() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @f(%n: i32) -> i32 {
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {
    ^outer_head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^outer_body(%j: i32):
        %inner = scf.while %j before {
        ^inner_head(%k: i32):
            %go = arith.cmpi ult %k, %n;
            scf.condition %go, %k : (i1, i32);
        } after {
        ^inner_body(%m: i32):
            %stop = arith.cmpi ugt %m, %n;
            %p = scf.if %stop then {
                scf.break 2 %m : (i32);
            } else {
                scf.continue 1 %m : (i32);
            } : (i1) -> (i32);
            scf.yield %p : (i32);
        } : (i32) -> (i32);
        scf.yield %inner : (i32);
    } : (i32) -> (i32);
    builtin.ret %r : (i32);
};";
        let (function, printed) = parse_function_fixpoint(&context, "exit_levels.hir", source)?;
        assert!(printed.contains("scf.break 2 %"), "{printed}");
        assert!(printed.contains("scf.continue 1 %"), "{printed}");

        let mut levels = Vec::new();
        function.as_operation_ref().borrow().prewalk_all(|op| {
            if let Some(exit) = op.downcast_ref::<Break>() {
                levels.push(("break", *exit.get_levels()));
            } else if let Some(exit) = op.downcast_ref::<Continue>() {
                levels.push(("continue", *exit.get_levels()));
            }
        });
        assert_eq!(levels, [("break", 2), ("continue", 1)]);

        let mut builder = OpBuilder::new(context.clone());
        let block = context.create_block_with_params([context.i32()]);
        builder.set_insertion_point_to_end(block);
        let arg = block.borrow().arguments()[0].upcast();
        let exit = crate::StructuredControlFlowOpBuilder::r#break(
            &mut builder,
            3,
            [arg],
            SourceSpan::UNKNOWN,
        )?;
        assert_eq!(*exit.borrow().get_levels(), 3);
        let exit = crate::StructuredControlFlowOpBuilder::r#continue(
            &mut builder,
            -2,
            [arg],
            SourceSpan::UNKNOWN,
        )?;
        assert_eq!(*exit.borrow().get_levels(), -2);

        Ok(())
    }

    /// The text of `f(%s: i32) -> i32` returning an `scf.index_switch` on `%s` with one case per
    /// element of `cases`, written as given.
    fn switch_text(cases: &[&str]) -> String {
        let mut text = String::from(
            "builtin.function public extern(\"C\") @f(%s: i32) -> i32 {\n    %r = \
             scf.index_switch %s\n",
        );
        for (index, case) in cases.iter().enumerate() {
            text.push_str(&format!(
                "    case {case} {{\n        %c{index} = arith.constant {index} : i32;\n        \
                 scf.yield %c{index} : (i32);\n    }}\n"
            ));
        }
        text.push_str(
            "    default {\n        %d = arith.constant -7 : i32;\n        scf.yield %d : \
             (i32);\n    } : (i32);\n    builtin.ret %r : (i32);\n};",
        );
        text
    }

    /// In the generic form the `cases` property and the regions are written separately, so they
    /// can disagree: a repeated case value, fewer case values than case regions and more are each
    /// rejected by the verifier, with a diagnostic that names the repeated value or the counts.
    #[test]
    fn index_switch_verifies_its_cases_in_the_generic_form() {
        use runic_core::testing::text::{parse, print_generic};

        let (_cx, function) = parse::<Function>(&switch_text(&["-1", "5"])).unwrap();
        let switch = find_op::<IndexSwitch>(&function.borrow());
        let generic = print_generic(switch.borrow().as_operation());
        // The generic printer separates regions with a newline, the generic parser wants a comma
        let generic = generic.replace("} \n{", "},\n{");
        let text = format!(
            "builtin.function public extern(\"C\") @f(%0: i32) -> i32 {{\n{generic}\n    \
             builtin.ret %4 : (i32);\n}};"
        );
        parse::<Function>(&text).unwrap_or_else(|err| panic!("{err}\n{text}"));

        for (cases, expected) in [
            ("-1, -1", "the case value -1 is used more than once"),
            (
                "-1",
                "1 case value(s) need 1 region(s) after the default region, but there are 2",
            ),
            (
                "-1, 5, 6",
                "3 case value(s) need 3 region(s) after the default region, but there are 2",
            ),
        ] {
            let broken = text.replacen(
                "#builtin.array_i32<-1, 5>",
                &format!("#builtin.array_i32<{cases}>"),
                1,
            );
            assert_ne!(broken, text);
            let err = parse::<Function>(&broken).err().unwrap_or_else(|| panic!("{cases} parsed"));
            assert!(err.contains("invalid operation scf.index_switch"), "{cases}: {err}");
            assert!(err.contains(expected), "{cases}: {err}");
        }
    }

    /// One case value written twice in different spellings is a duplicate, and a value outside
    /// the 32-bit range is rejected in either direction.
    #[test]
    fn a_case_spelled_two_ways_is_a_duplicate() {
        use runic_core::testing::text::parse;

        for pair in [
            ["-1", "0xffffffff"],
            ["-1", "4294967295"],
            ["0x80000000", "-2147483648"],
            ["1", "0b1"],
        ] {
            let err = parse::<Function>(&switch_text(&pair))
                .err()
                .unwrap_or_else(|| panic!("{pair:?} parsed"));
            assert!(err.contains("this case selector has already been used"), "{pair:?}: {err}");
        }
        for case in ["4294967296", "-2147483649"] {
            assert!(parse::<Function>(&switch_text(&[case])).is_err(), "{case}");
        }
    }

    /// Two nested loops whose inner body ends in an `scf.if` with `first` in its then region and
    /// `second` in its else region, each passing `%m`.
    fn loops_text(first: &str, second: &str) -> String {
        format!(
            "builtin.function public extern(\"C\") @f(%n: i32) -> i32 {{
    %zero = arith.constant 0 : i32;
    %r = scf.while %zero before {{
    ^outer_head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    }} after {{
    ^outer_body(%j: i32):
        %inner = scf.while %j before {{
        ^inner_head(%k: i32):
            %go = arith.cmpi ult %k, %n;
            scf.condition %go, %k : (i1, i32);
        }} after {{
        ^inner_body(%m: i32):
            %stop = arith.cmpi ugt %m, %n;
            %p = scf.if %stop then {{
                {first} %m : (i32);
            }} else {{
                {second} %m : (i32);
            }} : (i1) -> (i32);
            scf.yield %p : (i32);
        }} : (i32) -> (i32);
        scf.yield %inner : (i32);
    }} : (i32) -> (i32);
    builtin.ret %r : (i32);
}};"
        )
    }

    /// Levels in every spelling of an `i32` (signed decimal, hex, binary) parse to their value and
    /// round-trip, in the hand-written form and with each exit in the generic form; a level below
    /// 1, beyond the nesting, out of range or missing is rejected, in both forms.
    #[test]
    fn exit_levels_round_trip_in_every_spelling_and_both_forms() {
        use runic_core::testing::text::{parse, round_trip};

        for (first, second, expected) in [
            ("scf.break 2", "scf.continue 1", [2, 1]),
            ("scf.break 1", "scf.continue 2", [1, 2]),
            ("scf.break 0x2", "scf.continue 0b1", [2, 1]),
            ("scf.continue 0x00000001", "scf.break 0b10", [1, 2]),
        ] {
            let text = loops_text(first, second);
            let (_cx, function) =
                parse::<Function>(&text).unwrap_or_else(|err| panic!("{err}\n{text}"));
            let mut levels = Vec::new();
            function.as_operation_ref().borrow().prewalk_all(|op: &Operation| {
                if let Some(exit) = op.downcast_ref::<Break>() {
                    levels.push(*exit.get_levels());
                } else if let Some(exit) = op.downcast_ref::<Continue>() {
                    levels.push(*exit.get_levels());
                }
            });
            assert_eq!(levels, expected, "{first} / {second}");
            let mut skipped = Vec::new();
            // The generic form of an operation with two operand groups (`scf.condition`) does
            // not parse back: the generic printer writes one operand list
            let failures = round_trip::<Function>(function.as_operation_ref(), true, &mut skipped)
                .into_iter()
                .filter(|failure| {
                    !failure.starts_with("generic form of 'scf.condition' does not parse")
                })
                .collect::<Vec<_>>();
            assert!(failures.is_empty(), "{}", failures[0]);
            assert!(skipped.is_empty(), "{skipped:?}");
        }
        for (first, second) in [
            ("scf.break 0", "scf.continue 1"),
            ("scf.break -1", "scf.continue 1"),
            ("scf.break 0xffffffff", "scf.continue 1"),
            ("scf.break 3", "scf.continue 1"),
            ("scf.break 1", "scf.continue 3"),
            ("scf.break 4294967296", "scf.continue 1"),
            ("scf.break", "scf.continue 1"),
        ] {
            assert!(parse::<Function>(&loops_text(first, second)).is_err(), "{first} / {second}");
        }
        // The generic form without `levels` (the rest of the line is commented out)
        let text = loops_text("\"scf.break\"(%m) : (i32) -> ();\n//", "scf.continue 1");
        let err = parse::<Function>(&text).err().expect("levels is required");
        assert!(err.contains("missing required property 'levels'"), "{err}");
    }

    /// The memory effects of `scf` operations: a region operation has the effects of its regions
    /// and none of its own; the terminators have none. (No region operation is trivially dead,
    /// effect-free ones included, because `Operation::carries_a_nested_exit` counts any nested
    /// terminator, `scf.yield` included.)
    #[test]
    fn memory_effect_answers_of_scf_operations() {
        use alloc::string::ToString;

        use runic_core::testing::text::parse;

        let text = "builtin.function public extern(\"C\") @f(%c: i1, %n: i32, %p: ptr<i32>) -> \
                    i32 {
    %pure = scf.if %c then {
        scf.yield %n : (i32);
    } else {
        scf.yield %n : (i32);
    } : (i1) -> (i32);
    %reads = scf.if %c then {
        %x = test.load %p;
        scf.yield %x : (i32);
    } else {
        scf.yield %n : (i32);
    } : (i1) -> (i32);
    %writes = scf.if %c then {
        test.store %p, %n : (ptr<i32>, i32);
        scf.yield %n : (i32);
    } else {
        scf.yield %n : (i32);
    } : (i1) -> (i32);
    %asserts = scf.index_switch %n
    case 1 {
        cf.assert %c, \"m\";
        scf.yield %n : (i32);
    }
    default {
        scf.yield %n : (i32);
    } : (i32);
    %loop = scf.while %n before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        %stop = arith.cmpi ugt %j, %n;
        %q = scf.if %stop then {
            scf.break 1 %j : (i32);
        } else {
            scf.continue 1 %j : (i32);
        } : (i1) -> (i32);
        scf.yield %q : (i32);
    } : (i32) -> (i32);
    builtin.ret %n : (i32);
};";
        let (_cx, function) = parse::<Function>(text).unwrap_or_else(|err| panic!("{err}"));
        let mut answers = Vec::new();
        function.as_operation_ref().borrow().prewalk_all(|op: &Operation| {
            if op.name().to_string().starts_with("scf.") {
                let effects = op.get_effects_recursively::<MemoryEffect>().map(|effects| {
                    effects.iter().map(|effect| effect.effect().to_string()).collect::<Vec<_>>()
                });
                answers.push(format!(
                    "{} effects={effects:?} free={} dead_if_unused={}",
                    op.name(),
                    op.is_memory_effect_free(),
                    op.would_be_trivially_dead(),
                ));
            }
        });
        expect![[r#"
            scf.if effects=Some([]) free=true dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false
            scf.if effects=Some(["read"]) free=false dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false
            scf.if effects=Some(["write"]) free=false dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false
            scf.index_switch effects=Some(["write"]) free=false dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false
            scf.while effects=Some([]) free=true dead_if_unused=false
            scf.condition effects=Some([]) free=true dead_if_unused=false
            scf.if effects=Some([]) free=true dead_if_unused=false
            scf.break effects=Some([]) free=true dead_if_unused=false
            scf.continue effects=Some([]) free=true dead_if_unused=false
            scf.yield effects=Some([]) free=true dead_if_unused=false"#]]
        .assert_eq(&answers.join("\n"));
    }

    #[test]
    fn if_whose_yield_gives_the_wrong_number_of_values_is_rejected() {
        let err = rejection(
            "if_yield_count.hir",
            r#"
builtin.function public extern("C") @f(%c: i1, %x: i32) -> i32 {
    %r = scf.if %c then {
        scf.yield %x, %x : (i32, i32);
    } else {
        scf.yield %x : (i32);
    } : (i1) -> i32;
    builtin.ret %r : (i32);
};"#,
        );
        assert!(
            err.contains(
                "invalid operation scf.yield: successor 0 (the results of scf.if) takes 1 \
                 result(s), but scf.yield gives it 2 operand(s)"
            ),
            "{err}"
        );
    }

    #[test]
    fn if_whose_yield_gives_a_value_of_the_wrong_type_is_rejected() {
        let err = rejection(
            "if_yield_type.hir",
            r#"
builtin.function public extern("C") @f(%c: i1, %x: i32, %w: i64) -> i32 {
    %r = scf.if %c then {
        scf.yield %x : (i32);
    } else {
        scf.yield %w : (i64);
    } : (i1) -> i32;
    builtin.ret %r : (i32);
};"#,
        );
        assert!(
            err.contains(
                "invalid operation scf.yield: operand 0 for successor 0 (the results of scf.if) \
                 has type 'i64', but the result it binds has type 'i32'"
            ),
            "{err}"
        );
    }

    #[test]
    fn index_switch_whose_case_yields_the_wrong_values_is_rejected() {
        let err = rejection(
            "index_switch_yield.hir",
            r#"
builtin.function public extern("C") @f(%s: i32, %x: i32) -> i32 {
    %r = scf.index_switch %s
    case 1 {
        scf.yield : ();
    }
    default {
        scf.yield %x : (i32);
    } : (i32);
    builtin.ret %r : (i32);
};"#,
        );
        assert!(
            err.contains(
                "invalid operation scf.yield: successor 0 (the results of scf.index_switch) takes \
                 1 result(s), but scf.yield gives it 0 operand(s)"
            ),
            "{err}"
        );
    }

    /// A loop whose `scf.condition` forwards `%n` and whose body yields `%j`, both `i32`, with an
    /// `i64` argument `%w` to put in their place.
    const WHILE_TERMINATORS: &str = r#"
builtin.function public extern("C") @f(%n: i32, %w: i64) -> i32 {
    %r = scf.while %n before {
    ^head(%i: i32):
        %more = arith.cmpi ult %i, %n;
        scf.condition %more, %i : (i1, i32);
    } after {
    ^body(%j: i32):
        scf.yield %j : (i32);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};"#;

    /// Parse [WHILE_TERMINATORS], replace operand `operand` of the loop's terminator `T` with the
    /// function's `i64` argument, and verify the terminator alone.
    ///
    /// The loop's own verifier also compares these lists, so parsing the faulty loop would report
    /// the loop; verifying the terminator shows what the terminator's interface reports.
    fn verify_while_terminator_with_an_i64_operand<T: OpRegistration>(operand: usize) -> String {
        let context = Rc::new(Context::default());
        let (function, _) =
            parse_function_fixpoint(&context, "while_terminators.hir", WHILE_TERMINATORS).unwrap();
        let function = function.borrow();
        let wide = function.body().entry().arguments()[1] as ValueRef;
        let while_op = find_op::<While>(&function);
        let while_op = while_op.borrow();
        let terminator = while_op
            .regions()
            .iter()
            .find_map(|region| region.entry().terminator()?.try_downcast_op::<T>().ok())
            .expect("expected the loop to hold the terminator");
        let mut terminator = terminator.as_operation_ref();
        let mut target = terminator.borrow().operands()[operand];
        target.borrow_mut().set(wide);
        let result = terminator.borrow_mut().verify();
        format!("{}", result.unwrap_err())
    }

    #[test]
    fn while_whose_condition_forwards_a_value_of_the_wrong_type_is_rejected() {
        let err = verify_while_terminator_with_an_i64_operand::<Condition>(1);
        assert!(
            err.contains(
                "invalid operation scf.condition: operand 0 for successor 0 (region 1 of \
                 scf.while) has type 'i64', but the block argument it binds has type 'i32'"
            ),
            "{err}"
        );
    }

    #[test]
    fn while_whose_body_yields_a_value_of_the_wrong_type_is_rejected() {
        let err = verify_while_terminator_with_an_i64_operand::<Yield>(0);
        assert!(
            err.contains(
                "invalid operation scf.yield: operand 0 for successor 0 (region 0 of scf.while) \
                 has type 'i64', but the block argument it binds has type 'i32'"
            ),
            "{err}"
        );
    }

    /// A terminator that states no successor under its parent is reported, naming both, instead
    /// of panicking inside its own successor computation.
    #[test]
    fn a_condition_in_an_if_is_reported() {
        let error = rejection(
            "condition_in_if.hir",
            r#"
builtin.function public extern("C") @f(%c: i1, %x: i32) -> i32 {
    %r = scf.if %c then {
        scf.condition %c, %x : (i1, i32);
    } else {
        scf.yield %x : (i32);
    } : (i1) -> i32;
    builtin.ret %r : (i32);
};"#,
        );
        assert!(error.contains("`scf.condition` cannot end a region of `scf.if`"), "{error}");
    }

    #[test]
    fn a_condition_in_an_index_switch_is_reported() {
        let error = rejection(
            "condition_in_index_switch.hir",
            r#"
builtin.function public extern("C") @f(%c: i1, %s: i32, %x: i32) -> i32 {
    %r = scf.index_switch %s
    case 1 {
        scf.condition %c, %x : (i1, i32);
    }
    default {
        scf.yield %x : (i32);
    } : (i32);
    builtin.ret %r : (i32);
};"#,
        );
        assert!(
            error.contains("`scf.condition` cannot end a region of `scf.index_switch`"),
            "{error}"
        );
    }

    /// `test.callable_receiver` is a region-branch operation `scf.yield` does not know.
    #[test]
    fn a_yield_under_a_region_branch_op_of_another_dialect_is_reported() {
        let error = rejection(
            "yield_in_callable_receiver.hir",
            r#"
builtin.function public extern("C") @f() {
    test.callable_receiver {
        scf.yield : ();
    };
    builtin.ret;
};"#,
        );
        assert!(
            error.contains("`scf.yield` cannot end a region of `test.callable_receiver`"),
            "{error}"
        );
    }

    /// A conversion pattern for `scf.while` that converts the types of the arguments of its
    /// `before` region, and nothing else.
    struct ConvertBeforeRegion {
        info: runic_core::patterns::PatternInfo,
        converter: runic_core::conversion::TypeConverter,
    }

    impl runic_core::patterns::Pattern for ConvertBeforeRegion {
        fn info(&self) -> &runic_core::patterns::PatternInfo {
            &self.info
        }
    }

    impl runic_core::conversion::ConversionPattern for ConvertBeforeRegion {
        fn type_converter(&self) -> Option<&runic_core::conversion::TypeConverter> {
            Some(&self.converter)
        }

        fn match_and_rewrite(
            &self,
            op: OperationRef,
            _operands: runic_core::conversion::ConvertedOperands<'_>,
            rewriter: &mut runic_core::conversion::ConversionPatternRewriter,
        ) -> Result<bool, Report> {
            let before = op.borrow().regions().front().as_pointer().unwrap();
            rewriter.convert_region_types(before)
        }
    }

    /// Whether the entry block of the `before` region of the `scf.while` `op` takes no `i32`.
    fn before_region_takes_no_i32(op: &Operation) -> bool {
        let i32 = op.context().i32();
        let before = op.regions().front().as_pointer().unwrap();
        let before = before.borrow();
        !before.entry().argument_types().any(|ty| ty == i32)
    }

    /// Converting the types of a region of a region-branch operation whose entry types change,
    /// without a pattern for what branches to it, fails verification with the region-branch
    /// error: the `scf.yield` of the `after` region forwards an `i32` to the `before` region,
    /// which now takes an `i64` (and so does `scf.while` itself, on entry).
    #[test]
    fn a_converted_region_whose_entry_operands_are_not_converted_fails_verification() {
        use runic_core::{
            conversion::{
                ConversionConfig, ConversionPatternSet, ConversionTarget, DynamicLegalityResult,
                TypeConversion, TypeConverter, UnknownOpPolicy, apply_full_conversion,
            },
            patterns::{PatternBenefit, PatternInfo, PatternKind},
            testing::text::parse,
        };

        let (context, function) = parse::<Function>(
            "builtin.function public extern(\"C\") @f(%flag: i1, %value: i32) -> i32 {
    %r = scf.while %value before {
    ^head(%current: i32):
        scf.condition %flag, %value : (i1, i32);
    } after {
    ^body(%next: i32):
        scf.yield %next : (i32);
    } : (i32) -> i32;
    builtin.ret %r : (i32);
};",
        )
        .unwrap();
        let (i32, i64) = (context.i32(), context.i64());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| (ty == i32).then_some(TypeConversion::One(i64)));
        let scf = context.get_or_register_dialect::<crate::ScfDialect>();
        let info = PatternInfo::new(
            context.clone(),
            "convert-before-region",
            PatternKind::Operation(scf.expect_registered_name::<While>()),
            PatternBenefit::new(1),
        );
        let mut patterns = ConversionPatternSet::new(context.clone());
        patterns.push(ConvertBeforeRegion { info, converter });
        let mut target = ConversionTarget::new(context.clone());
        target
            .set_unknown_op_policy(UnknownOpPolicy::Legal)
            .add_dynamically_legal_op::<While, _>(|op: &Operation| {
                DynamicLegalityResult::legal_if(before_region_takes_no_i32(op))
            });

        let err = apply_full_conversion(
            function.as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap_err();

        assert_eq!(
            format!("{err}"),
            "invalid operation scf.yield: operand 0 for successor 0 (region 0 of scf.while) has \
             type 'i32', but the block argument it binds has type 'i64'"
        );
    }
}
