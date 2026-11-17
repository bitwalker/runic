//! Structured control flow: `block`, `loop`, `if`, `end`, `br`, `br_table` and `unreachable`.
//!
//! Wasm's operand stack is SSA here. The parameters of a `block`, `loop` or `if` are its operands,
//! which become the arguments of the entry block of its body (of both bodies, for an `if`), and
//! its results are the results of the operation. A body ends with [End], which leaves the
//! construct with its operands as the results, or with an early exit, [Br], which leaves the
//! construct it names, with the values that construct expects, or [BrTable], or [Unreachable].
//!
//! # Labels
//!
//! A branch names a *label*: [BlockOp], [Loop] and [If] each introduce one, and only those do
//! (every other operation a branch crosses must forward early exits, and does not count). Label 0
//! is the innermost enclosing one, as in Wasm. A branch to a `block` or an `if` leaves it, with
//! the values of its results, and a branch to a `loop` re-enters its body, with the values of its
//! parameters. Wasm lets `br` name the function itself as the outermost label; here the function
//! is left with `builtin.ret`, and a label that would name the function is an error.
//!
//! # Conditional branches
//!
//! `br_if` is not an operation. It is a [If] whose first region holds a [Br] and whose second
//! passes its operands through, which the text writes without the `else`:
//!
//! ```text,ignore
//! // br_if 0 %v, %c, in a block of one result
//! wasm.if %c, %v then {
//! ^taken(%x: i32):
//!     wasm.br 1 %x : (i32);
//! } : (i32, i32) -> (i32);
//! ```
//!
//! (Label 1 is the construct around the `if`, as the `if` is label 0.) A later stage's emitter
//! writes that shape as `br_if`.
//!
//! # What every analysis sees
//!
//! Each of the three constructs implements `RegionBranchOpInterface`, whose answer for the
//! successors of its body is the fall-through to its parent (the results); a branch states its
//! own edge ([RegionBranchTerminatorOpInterface]): to the results of the block or the `if` it
//! names, or to the entry block of the loop it names. [BlockOp], [Loop] and [If] carry
//! `PropagateControlFlowBreak`, so a branch, a `builtin.ret` or a tail call crosses them.

use alloc::{format, string::String, vec::Vec};

use runic_ir::{
    AttributeRef, BlockArgumentRef, BuilderExt, CallableOpInterface, Context, EntityRef,
    ExitDefect, ExitDefectKind, ExitResolution, HasBreakingControlFlowOpInterface,
    LoopLikeOpInterface, Op, OpAsmParser, OpOperandRangeMut, OpParser, OpPrinter, Operation,
    OperationState, ParseResult, PropagateControlFlowBreak, RegionBranchOpInterface,
    RegionBranchPoint, RegionBranchTerminatorOpInterface, RegionExitKind,
    RegionExitTerminatorOpInterface, RegionRef, RegionSuccessorInfo, RegionSuccessorIter, Report,
    ResolvedExit, SmallVec, SuccessorOperandRange, SuccessorOperandRangeMut, Type, Value, ValueRef,
    Verify, WalkResult,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::I32Attr,
    effects::{HasRecursiveMemoryEffects, MemoryEffect, MemoryEffectOpInterface},
    formatter::const_text,
    parse::{Delimiter, ParserError, ParserExt},
    print::AsmPrinter,
    traits::{AnyType, NoRegionArguments, ReturnLike, SingleBlock, Terminator},
    verify_region_branch_entry_operands,
};

use super::text::*;
use crate::{WasmDialect, is_val_type, val_type_name};

/// A reference to a [BlockOp].
pub type BlockOpRef = runic_ir::UnsafeIntrusiveEntityRef<BlockOp>;
/// A reference to a [Loop].
pub type LoopRef = runic_ir::UnsafeIntrusiveEntityRef<Loop>;
/// A reference to an [If].
pub type IfRef = runic_ir::UnsafeIntrusiveEntityRef<If>;
/// A reference to an [End].
pub type EndRef = runic_ir::UnsafeIntrusiveEntityRef<End>;
/// A reference to a [Br].
pub type BrRef = runic_ir::UnsafeIntrusiveEntityRef<Br>;
/// A reference to a [BrTable].
pub type BrTableRef = runic_ir::UnsafeIntrusiveEntityRef<BrTable>;

/// Whether `op` introduces a label that a [Br] may name: a `wasm.block`, `wasm.loop` or `wasm.if`.
fn is_label(op: &Operation) -> bool {
    op.is::<BlockOp>() || op.is::<Loop>() || op.is::<If>()
}

/// The text of the types of `values`, as a message spells them: `(i32, i64)`.
fn types_of<'a>(values: impl IntoIterator<Item = &'a ValueRef>) -> String {
    let names = values
        .into_iter()
        .map(|value| val_type_name(value.borrow().ty()))
        .collect::<Vec<_>>();
    format!("({})", names.join(", "))
}

/// The text of `types`, as a message spells them: `(i32, i64)`.
fn type_list(types: impl IntoIterator<Item = Type>) -> String {
    let names = types.into_iter().map(val_type_name).collect::<Vec<_>>();
    format!("({})", names.join(", "))
}

/// The results of `op`, as the values a fall-through or a branch to it gives.
fn result_values(op: &Operation) -> SmallVec<[ValueRef; 2]> {
    op.results().all().iter().map(|result| result.borrow().as_value_ref()).collect()
}

/// Whether some operation nested in `op` is an early exit that names `op`, of `kind` (of any kind
/// when `kind` is `None`).
///
/// A branch is resolved by [resolve_level], the one function the verifier and the successor sets
/// share, so a branch that names an enclosing construct further out is not counted here: it is for
/// that construct to count.
fn has_nested_exit(op: &Operation, kind: Option<RegionExitKind>) -> bool {
    let me = op.as_operation_ref();
    op.regions().iter().any(|region| {
        region
            .prewalk(|nested: &Operation| {
                if let Some(exit) = nested.as_trait::<dyn RegionExitTerminatorOpInterface>()
                    && kind.is_none_or(|kind| kind == exit.get_exit_kind())
                    && exit.get_exit_target() == Some(me)
                {
                    return WalkResult::<()>::Break(());
                }
                WalkResult::Continue(())
            })
            .was_interrupted()
    })
}

/// The constant value of the `i32` operand `attr`, when it is one.
fn constant_i32(attr: &Option<AttributeRef>) -> Option<u32> {
    let value = attr.as_ref()?.borrow().as_int()?;
    (value.width() == 32).then(|| value.unsigned() as u32)
}

/// What is wrong with an operand or result of `op` that is not a Wasm value type: the problem, or
/// `None` when every one of them is.
fn first_non_value_type(op: &Operation, first_input: usize) -> Option<String> {
    for (index, operand) in op.operands().iter().enumerate().skip(first_input) {
        let ty = operand.borrow().as_value_ref().borrow().ty();
        if !is_val_type(ty) {
            return Some(format!("its operand {index} has type `{}`", val_type_name(ty)));
        }
    }
    for (index, result) in op.results().iter().enumerate() {
        let ty = result.borrow().ty();
        if !is_val_type(ty) {
            return Some(format!("its result {index} has type `{}`", val_type_name(ty)));
        }
    }
    None
}

/// Checks that `op` has exactly `expected` regions, which the message calls `what`: the generic
/// form can write more, and the custom form would not print them.
fn verify_region_count(op: &Operation, expected: usize, what: &str) -> Result<(), Report> {
    let count = op.num_regions();
    if count == expected {
        return Ok(());
    }
    let plural = if count == 1 { "" } else { "s" };
    Err(rule_error(
        op,
        &format!("takes exactly {what}"),
        format!("it has {count} region{plural}"),
    ))
}

/// Checks that the entry block of `region`, the body called `name` of `op`, takes as its arguments
/// the operands of `op` from `first_input` on.
fn verify_body_arguments(
    op: &Operation,
    rule: &str,
    first_input: usize,
    name: &str,
    region: RegionRef,
) -> Result<(), Report> {
    let Some(entry) = region.borrow().entry_block_ref() else {
        return Err(rule_error(op, rule, format!("{name} has no block")));
    };
    let expected = op
        .operands()
        .iter()
        .skip(first_input)
        .map(|operand| operand.borrow().as_value_ref().borrow().ty())
        .collect::<Vec<_>>();
    let found = entry
        .borrow()
        .arguments()
        .iter()
        .map(|arg| arg.borrow().ty())
        .collect::<Vec<_>>();
    if found != expected {
        return Err(rule_error(
            op,
            rule,
            format!(
                "the entry block of {name} takes {}, not the parameters {}",
                type_list(found),
                type_list(expected)
            ),
        ));
    }
    Ok(())
}

/// Gives `state` the operands `values`, whose first is the one operand of the first group (the
/// condition of an `if`, the index of a table) and whose others are the second group.
fn add_condition_and_values(state: &mut OperationState, values: SmallVec<[ValueRef; 2]>) {
    let (first, rest) = values.split_at(1);
    state.add_operands(first.iter().copied().collect());
    state.add_operands(rest.iter().copied().collect());
}

// ---------------------------------------------------------------------------------------------
// Blocks, loops and ifs

/// A block: `%r = wasm.block %a { ^body(%x: i32): .. wasm.end %y : (i32); } : (i32) -> (i32);`.
/// Wasm's `block` with a block type that has parameters and results.
///
/// The operands are the parameters: they are the arguments of the entry block of the body, in
/// order and of the same types. The results are the values the block leaves with, which are the
/// operands of the [End] that ends the body, or of a [Br] that names the block (label 0 inside it,
/// when no other `wasm.block`, `wasm.loop` or `wasm.if` is in between). The parameters and results
/// are Wasm value types.
///
/// The operation has no effects of its own: its effects are those of what its body holds
/// (`HasRecursiveMemoryEffects`). It forwards early exits.
///
/// The struct is named `BlockOp`, not `Block`, because `Block` is the IR's own type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[effects(MemoryEffect())]
#[operation(
    dialect = WasmDialect,
    name = "block",
    traits(SingleBlock, HasRecursiveMemoryEffects, PropagateControlFlowBreak),
    implements(
        RegionBranchOpInterface,
        HasBreakingControlFlowOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct BlockOp {
    #[operands]
    inputs: AnyType,
    #[region]
    body: Region,
    #[results]
    returns: AnyType,
}

const BLOCK_RULE: &str = "takes parameters and gives results of Wasm value types, and its body's \
                          entry block takes the parameters as its arguments";

/// The parameters and results are Wasm value types, and the entry block of the body takes the
/// parameters; and the interface's own rule, [verify_region_branch_entry_operands], holds.
impl Verify<dyn RegionBranchOpInterface> for BlockOp {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        verify_region_count(op, 1, "one region, its body")?;
        if let Some(problem) = first_non_value_type(op, 0) {
            return Err(rule_error(op, BLOCK_RULE, problem));
        }
        verify_body_arguments(op, BLOCK_RULE, 0, "its body", self.body().as_region_ref())?;
        verify_region_branch_entry_operands(self)
    }
}

impl RegionBranchOpInterface for BlockOp {
    fn get_entry_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.operands().all())
    }

    /// The body is entered from the parent. From the body, control goes to the parent with the
    /// results: by falling off the end of the body ([End]), or by a nested [Br] that names this
    /// block (which states its own edge, and arrives at the same point).
    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Entering(self.body().as_region_ref())],
            ),
            RegionBranchPoint::Child(_) => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Returning(result_values(self.as_operation()))],
            ),
        }
    }

    fn is_repetitive_region(&self, _index: usize) -> bool {
        false
    }

    fn has_loop(&self) -> bool {
        false
    }
}

impl HasBreakingControlFlowOpInterface for BlockOp {
    fn has_nested_predecessors(&self) -> bool {
        has_nested_exit(self.as_operation(), None)
    }

    fn get_continue_target_region(&self) -> Option<RegionRef> {
        None
    }
}

/// A loop: `%r = wasm.loop %a { ^body(%x: i32): .. wasm.br 0 %next : (i32); } : (i32) -> (i32);`.
/// Wasm's `loop`.
///
/// The operands are the parameters: the arguments of the entry block of the body. The one region
/// is both header and body. A [Br] that names the loop (label 0 inside it, when no other
/// `wasm.block`, `wasm.loop` or `wasm.if` is in between) re-enters the body with its operands as
/// the new arguments, so its operands have the types of the parameters. The loop is left by an
/// [End] that ends the body, with the operands of the `end` as the results, or by a [Br] that
/// names a construct around the loop.
///
/// The back edge is an early exit, not an edge of the region graph: the loop answers the questions
/// that a graph walk would answer wrongly (`has_loop`, `is_repetitive_region`: true when a branch
/// names it), allows a continue from its header, and is a [LoopLikeOpInterface].
///
/// The operation has no effects of its own: its effects are those of what its body holds. It
/// forwards early exits.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[effects(MemoryEffect())]
#[operation(
    dialect = WasmDialect,
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
    inputs: AnyType,
    #[region]
    body: Region,
    #[results]
    returns: AnyType,
}

const LOOP_RULE: &str = "takes parameters and gives results of Wasm value types, and its body's \
                         entry block takes the parameters as its arguments";

/// As for [BlockOp].
impl Verify<dyn RegionBranchOpInterface> for Loop {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        verify_region_count(op, 1, "one region, its body")?;
        if let Some(problem) = first_non_value_type(op, 0) {
            return Err(rule_error(op, LOOP_RULE, problem));
        }
        verify_body_arguments(op, LOOP_RULE, 0, "its body", self.body().as_region_ref())?;
        verify_region_branch_entry_operands(self)
    }
}

impl RegionBranchOpInterface for Loop {
    fn get_entry_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.operands().all())
    }

    /// The body is entered from the parent. From the body, control goes to the parent with the
    /// results by falling off the end ([End]); the way back to the top of the body is a nested
    /// [Br], an exit edge that the branch states itself.
    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Entering(self.body().as_region_ref())],
            ),
            RegionBranchPoint::Child(_) => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Returning(result_values(self.as_operation()))],
            ),
        }
    }

    /// The body is repeated when a branch names the loop: the region graph has no edge back to it.
    fn is_repetitive_region(&self, _index: usize) -> bool {
        has_nested_exit(self.as_operation(), Some(RegionExitKind::Continue))
    }

    fn has_loop(&self) -> bool {
        has_nested_exit(self.as_operation(), Some(RegionExitKind::Continue))
    }
}

impl HasBreakingControlFlowOpInterface for Loop {
    fn has_nested_predecessors(&self) -> bool {
        has_nested_exit(self.as_operation(), None)
    }

    fn get_continue_target_region(&self) -> Option<RegionRef> {
        Some(self.body().as_region_ref())
    }

    /// The body is the header, and a branch to the loop is nested in it.
    fn continue_may_originate_in_header(&self) -> bool {
        true
    }
}

impl LoopLikeOpInterface for Loop {
    fn get_loop_header_region(&self) -> RegionRef {
        self.body().as_region_ref()
    }

    fn get_loop_regions(&self) -> SmallVec<[RegionRef; 2]> {
        let mut regions = SmallVec::new();
        regions.push(self.body().as_region_ref());
        regions
    }

    fn get_region_iter_args(&self) -> Option<EntityRef<'_, [BlockArgumentRef]>> {
        let entry = self.body().entry_block_ref()?;
        Some(EntityRef::map(entry.borrow(), |block| block.arguments()))
    }

    fn get_inits_mut(&mut self) -> OpOperandRangeMut<'_> {
        self.inputs_mut()
    }
}

/// A conditional: `%r = wasm.if %c, %a then { ^then(%x: i32): .. } else { ^else(%x: i32): .. } :
/// (i32, i32) -> (i32);`. Wasm's `if` with a block type that has parameters and results.
///
/// The first operand is the condition, an `i32`: when it is not zero the first region runs, and
/// when it is the second runs. The further operands are the parameters: they are the arguments of
/// the entry block of **both** regions. Each region ends as a [BlockOp] does, with an [End] whose
/// operands are the results, or an early exit. A [Br] that names the `if` is label 0 inside either
/// region (when no other `wasm.block`, `wasm.loop` or `wasm.if` is in between), and leaves it with
/// the values of the results.
///
/// Wasm's `if` without an `else` requires that its parameters have the types of its results, and
/// passes them through when the condition is zero. The IR always has the second region, and the
/// text may leave out `else` when the region only passes the operands through (its block holds a
/// single `wasm.end` of its arguments): the parser builds exactly that region, and the printer
/// leaves it out. A `br_if` is such an `if` whose first region holds a [Br].
///
/// The operation has no effects of its own: its effects are those of the region that runs
/// (`HasRecursiveMemoryEffects`). It forwards early exits.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = WasmDialect,
    traits(SingleBlock, HasRecursiveMemoryEffects, PropagateControlFlowBreak),
    implements(
        RegionBranchOpInterface,
        HasBreakingControlFlowOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct If {
    #[operand]
    condition: AnyType,
    #[operands]
    inputs: AnyType,
    #[region(name = "then")]
    then_body: Region,
    #[region(name = "else")]
    else_body: Region,
    #[results]
    returns: AnyType,
}

const IF_RULE: &str = "takes an `i32` condition and parameters, gives results, all of Wasm value \
                       types, and the entry block of each of its two regions takes the parameters \
                       as its arguments";

impl If {
    /// Whether the second region only passes the operands through: its block holds one `wasm.end`
    /// of exactly the block's arguments, which have the types of the results.
    ///
    /// Such a region is what the text leaves out, and what an `if` without `else` stands for.
    pub fn else_is_pass_through(&self) -> bool {
        let region = self.else_body();
        let Some(entry) = region.entry_block_ref() else {
            return false;
        };
        let entry = entry.borrow();
        let mut body = entry.body().iter();
        let (Some(first), None) = (body.next(), body.next()) else {
            return false;
        };
        let Some(end) = first.downcast_ref::<End>() else {
            return false;
        };
        if first.has_attributes() {
            return false;
        }
        let arguments = entry.arguments();
        let values = end.as_operation().operands().all();
        let results = self.results().all();
        values.len() == arguments.len()
            && results.len() == arguments.len()
            && values
                .iter()
                .zip(arguments.iter())
                .all(|(value, argument)| value.borrow().as_value_ref() == (*argument as ValueRef))
            && results
                .iter()
                .zip(arguments.iter())
                .all(|(result, argument)| result.borrow().ty() == argument.borrow().ty())
    }
}

/// The condition is an `i32`, the parameters and results are Wasm value types, and the entry block
/// of each region takes the parameters; and the interface's own rule,
/// [verify_region_branch_entry_operands], holds.
impl Verify<dyn RegionBranchOpInterface> for If {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        verify_region_count(op, 2, "two regions, its `then` and `else` bodies")?;
        let condition = operand_type(op, 0);
        if condition != op.context().i32() {
            return Err(rule_error(
                op,
                IF_RULE,
                format!("its condition has type `{}`", val_type_name(condition)),
            ));
        }
        if let Some(problem) = first_non_value_type(op, 1) {
            return Err(rule_error(op, IF_RULE, problem));
        }
        verify_body_arguments(
            op,
            IF_RULE,
            1,
            "its first region",
            self.then_body().as_region_ref(),
        )?;
        verify_body_arguments(
            op,
            IF_RULE,
            1,
            "its second region",
            self.else_body().as_region_ref(),
        )?;
        verify_region_branch_entry_operands(self)
    }
}

impl RegionBranchOpInterface for If {
    /// The operands after the condition are what the regions receive.
    fn get_entry_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.inputs())
    }

    /// Both regions are entered when the condition is not known, one when it is: a zero selects
    /// the second. (There is no skip edge: the second region always exists, and passes the
    /// operands through when the text has no `else`.)
    fn get_entry_successor_regions(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> RegionSuccessorIter<'_> {
        let condition = operands.first().and_then(constant_i32);
        let mut infos = SmallVec::<[RegionSuccessorInfo; 2]>::default();
        if condition.is_none_or(|value| value != 0) {
            infos.push(RegionSuccessorInfo::Entering(self.then_body().as_region_ref()));
        }
        if condition.is_none_or(|value| value == 0) {
            infos.push(RegionSuccessorInfo::Entering(self.else_body().as_region_ref()));
        }
        RegionSuccessorIter::new(self.as_operation(), infos)
    }

    /// Either region is entered from the parent. From either, control goes to the parent with the
    /// results: by falling off the end ([End]), or by a nested [Br] that names this `if`.
    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => RegionSuccessorIter::new(
                self.as_operation(),
                [
                    RegionSuccessorInfo::Entering(self.then_body().as_region_ref()),
                    RegionSuccessorInfo::Entering(self.else_body().as_region_ref()),
                ],
            ),
            RegionBranchPoint::Child(_) => RegionSuccessorIter::new(
                self.as_operation(),
                [RegionSuccessorInfo::Returning(result_values(self.as_operation()))],
            ),
        }
    }

    fn is_repetitive_region(&self, _index: usize) -> bool {
        false
    }

    fn has_loop(&self) -> bool {
        false
    }
}

impl HasBreakingControlFlowOpInterface for If {
    fn has_nested_predecessors(&self) -> bool {
        has_nested_exit(self.as_operation(), None)
    }

    fn get_continue_target_region(&self) -> Option<RegionRef> {
        None
    }
}

impl OpPrinter for If {
    fn states_operand_types(&self) -> bool {
        true
    }

    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use runic_ir::ValueRange;

        let op = self.as_operation();
        printer.print_space();
        printer.print_value_uses(ValueRange::<2>::from(op.operands().all()));
        printer.print_space();
        printer.print_keyword("then");
        printer.print_space();
        printer.print_region(&self.then_body());
        if !self.else_is_pass_through() {
            printer.print_space();
            printer.print_keyword("else");
            printer.print_space();
            printer.print_region(&self.else_body());
        }
        print_attributes(printer, op);
        printer.print_space();
        printer.print_colon_type_list(op.operands().all().iter().map(|o| o.borrow().ty()));
        printer.print_space();
        printer.print_arrow();
        printer.print_space();
        printer.print_type_list(op.results().all().iter().map(|r| r.borrow().ty()));
    }
}

impl OpParser for If {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        use runic_ir::{Builder, OpBuilder, diagnostics::report};

        let mut operands = SmallVec::<[runic_ir::parse::UnresolvedOperand; 2]>::new();
        operands.push(parser.parse_operand(true)?);
        parser.parse_trailing_operand_list(&mut operands, Delimiter::None)?;

        parser.parse_custom_keyword("then")?;
        let then_region = parser.context().create_region();
        parser.parse_region(then_region, &[], false)?;
        let else_region = if parser.parse_optional_custom_keyword("else")?.is_some() {
            let region = parser.context().create_region();
            parser.parse_region(region, &[], false)?;
            Some(region)
        } else {
            None
        };
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let mut params = SmallVec::<[Type; 4]>::default();
        parser.parse_colon_type_list(&mut params)?;
        parser.parse_arrow_type_list(&mut state.results)?;
        if params.len() != operands.len() {
            return Err(ParserError::OperandAndTypeListMismatch {
                span: state.span,
                num_operands: operands.len(),
                num_types: params.len(),
            });
        }
        let mut values = SmallVec::<[ValueRef; 2]>::new();
        for (operand, ty) in operands.into_iter().zip(params.iter().copied()) {
            values.push(parser.resolve_operand(operand, ty)?);
        }
        add_condition_and_values(state, values);

        let else_region = match else_region {
            Some(region) => region,
            None => {
                // An `if` with no `else` passes its parameters through when the condition is
                // zero, so they have the types of its results.
                let parameters = &params[1..];
                if parameters != state.results.as_slice() {
                    return Err(ParserError::from(report!(
                        message: "invalid wasm.if operation",
                        labels: [primary(
                            state.span,
                            format!(
                                "an `if` with no `else` passes its parameters {} through as its \
                                 results {}, so they have the types of its results, but they \
                                 do not",
                                type_list(parameters.iter().copied()),
                                type_list(state.results.iter().copied())
                            )
                        )],
                    )));
                }
                let context = parser.context_rc();
                let region = context.create_region();
                let mut builder = OpBuilder::new(context);
                let block = builder.create_block(region, None, parameters);
                let values = block
                    .borrow()
                    .arguments()
                    .iter()
                    .map(|argument| *argument as ValueRef)
                    .collect::<Vec<_>>();
                builder.create::<End, (Vec<ValueRef>,)>(state.span)(values)
                    .map_err(ParserError::from)?;
                region
            }
        };
        state.add_region(then_region);
        state.add_region(else_region);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// End

/// The end of a body: `wasm.end %y : (i32);`. Falling off the `end` of a `block`, `loop` or `if`.
///
/// It ends the body of the enclosing [BlockOp], [Loop] or [If] and leaves it: its operands are
/// the results of that construct, in order and of the same types (for a `loop`, falling off the
/// end leaves the loop; the way back to the top is a [Br]). It has no results and no effects. The
/// parent must be one of the three: under any other operation `wasm.end` states no successor, and
/// the verifier rejects it.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[effects(MemoryEffect())]
#[operation(
    dialect = WasmDialect,
    traits(Terminator, ReturnLike),
    implements(RegionBranchTerminatorOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct End {
    #[operands]
    values: AnyType,
}

impl RegionBranchTerminatorOpInterface for End {
    fn get_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.values())
    }

    fn get_mutable_successor_operands(
        &mut self,
        _point: RegionBranchPoint,
    ) -> SuccessorOperandRangeMut<'_> {
        SuccessorOperandRangeMut::forward(self.values_mut())
    }

    fn get_successor_regions(
        &self,
        _operands: &[Option<AttributeRef>],
    ) -> SmallVec<[RegionSuccessorInfo; 2]> {
        // Under anything but a construct of the dialect the terminator states no successor, and
        // the region-branch terminator verifier says so.
        let mut infos = SmallVec::new();
        if let Some(parent) = self.parent_op() {
            let parent = parent.borrow();
            if is_label(&parent) {
                infos.push(RegionSuccessorInfo::Returning(result_values(&parent)));
            }
        }
        infos
    }
}

// ---------------------------------------------------------------------------------------------
// Br

/// A branch: `wasm.br 1 %v : (i32);`. Wasm's `br`.
///
/// The property `level` names the label: 0 is the innermost enclosing [BlockOp], [Loop] or [If],
/// and each further level the next enclosing one of those three (no other operation counts, and
/// each operation crossed on the way must forward early exits, or the exit is opaque). Branching
/// to a `block` or an `if` leaves it, and the operands are its results; branching to a `loop`
/// re-enters its body, and the operands are the new arguments of its entry block. A level that
/// would name the function itself (Wasm lets `br` do so) is an error: this IR leaves a function
/// with `builtin.ret`.
///
/// It is a terminator and an early exit (a [RegionExitTerminatorOpInterface] whose kind is
/// `Continue` when the target is a [Loop] and `Break` otherwise), has no results and no effects,
/// and forwards no further exits itself.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = WasmDialect,
    traits(Terminator, ReturnLike),
    implements(
        RegionBranchTerminatorOpInterface,
        RegionExitTerminatorOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct Br {
    #[attr]
    level: I32Attr,
    #[operands]
    values: AnyType,
}

/// What the label `level` of a branch in `op` names, walking outward from the branch and counting
/// the [BlockOp], [Loop] and [If] it passes.
///
/// This is the one place the level is resolved: the verifier and the sets of successors both use
/// it, so they cannot disagree about which construct a branch leaves.
fn resolve_level(op: &Operation, level: i32) -> ExitResolution {
    if level < 0 {
        return ExitResolution::Unresolved(ExitDefect {
            kind: ExitDefectKind::Malformed,
            detail: format!(
                "`wasm.br` names label {level}, but a label is numbered from 0, the innermost \
                 enclosing `wasm.block`, `wasm.loop` or `wasm.if`"
            ),
        });
    }
    let level = level as u32;
    // The label is not found: either it is the function itself, or beyond it.
    let not_found = |seen: u32| {
        let detail = if level == seen {
            format!(
                "`wasm.br` names label {level}, which is the function itself: Wasm lets `br` name \
                 the function's label, but this IR leaves a function with `builtin.ret`"
            )
        } else {
            format!(
                "`wasm.br` names label {level}, but only {seen} `wasm.block`, `wasm.loop` or \
                 `wasm.if` enclose it (label {seen} would be the function itself, which this IR \
                 leaves with `builtin.ret`)"
            )
        };
        ExitResolution::Unresolved(ExitDefect {
            kind: ExitDefectKind::NotFound,
            detail,
        })
    };

    let mut seen = 0;
    let Some(mut from_region) = op.parent_region() else {
        return not_found(seen);
    };
    loop {
        let Some(parent_ref) = from_region.borrow().parent() else {
            return not_found(seen);
        };
        let parent = parent_ref.borrow();
        // An exit may not escape a function: the callable is the outer bound of the search.
        if parent.implements::<dyn CallableOpInterface>() {
            return not_found(seen);
        }
        if is_label(&parent) {
            if seen == level {
                return ExitResolution::Resolved(ResolvedExit {
                    target: parent_ref,
                    from_region,
                    callable: None,
                });
            }
            seen += 1;
        }
        // Every operation the branch passes through must forward it.
        if !parent.implements::<dyn PropagateControlFlowBreak>() {
            return ExitResolution::Unresolved(ExitDefect {
                kind: ExitDefectKind::Opaque {
                    crossing: parent_ref,
                },
                detail: format!(
                    "`wasm.br` names label {level}, but would have to cross `{}`, which has not \
                     declared that it forwards breaking control flow",
                    parent.name()
                ),
            });
        }
        let Some(next) = parent.parent_region() else {
            return not_found(seen);
        };
        from_region = next;
    }
}

impl Br {
    /// The construct this branch names, when the label resolves.
    pub fn target(&self) -> Option<runic_ir::OperationRef> {
        match resolve_level(self.as_operation(), *self.get_level()) {
            ExitResolution::Resolved(resolved) => Some(resolved.target),
            ExitResolution::Unresolved(_) => None,
        }
    }
}

impl RegionExitTerminatorOpInterface for Br {
    fn resolve_exit(&self) -> ExitResolution {
        resolve_level(self.as_operation(), *self.get_level())
    }

    /// `Continue` when the label is a loop's, `Break` otherwise (and when it does not resolve,
    /// which the verifier rejects).
    fn get_exit_kind(&self) -> RegionExitKind {
        match self.target() {
            Some(target) if target.borrow().is::<Loop>() => RegionExitKind::Continue,
            _ => RegionExitKind::Break,
        }
    }
}

const BR_RULE: &str = "names an enclosing `wasm.block`, `wasm.loop` or `wasm.if` and carries the \
                       values it expects: the results of a block or an if, the parameters of a \
                       loop";

/// The label resolves, and the operands are what the construct it names expects.
///
/// This restates the rules of the exit interface that apply, with the wording of Wasm: the
/// interface's own verifier would call every target a loop.
impl Verify<dyn RegionExitTerminatorOpInterface> for Br {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        if op.num_results() != 0 {
            return Err(rule_error(
                op,
                BR_RULE,
                format!("it declares {} results, and a branch has none", op.num_results()),
            ));
        }
        if op.num_successors() != 0 {
            return Err(rule_error(
                op,
                BR_RULE,
                format!(
                    "it declares {} block successors, and a branch has none",
                    op.num_successors()
                ),
            ));
        }
        let resolved = match self.resolve_exit() {
            ExitResolution::Resolved(resolved) => resolved,
            ExitResolution::Unresolved(defect) => return Err(invalid(op, defect.detail)),
        };
        let target = resolved.target.borrow();
        let found = op
            .operands()
            .iter()
            .map(|operand| operand.borrow().as_value_ref().borrow().ty())
            .collect::<Vec<_>>();
        let (expected, what) = if let Some(target) = target.downcast_ref::<Loop>() {
            let Some(entry) = target.body().entry_block_ref() else {
                // The loop's own verifier reports the body that has no block
                return Ok(());
            };
            (
                entry
                    .borrow()
                    .arguments()
                    .iter()
                    .map(|arg| arg.borrow().ty())
                    .collect::<Vec<_>>(),
                "the parameters of the `wasm.loop` it names (the arguments of its body)",
            )
        } else {
            (
                target.results().iter().map(|result| result.borrow().ty()).collect::<Vec<_>>(),
                "the results of the `wasm.block` or `wasm.if` it names",
            )
        };
        if found != expected {
            return Err(rule_error(
                op,
                BR_RULE,
                format!("it carries {}, and {what} are {}", type_list(found), type_list(expected)),
            ));
        }
        Ok(())
    }
}

impl RegionBranchTerminatorOpInterface for Br {
    fn get_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::forward(self.values())
    }

    fn get_mutable_successor_operands(
        &mut self,
        _point: RegionBranchPoint,
    ) -> SuccessorOperandRangeMut<'_> {
        SuccessorOperandRangeMut::forward(self.values_mut())
    }

    /// One edge, known statically: to the results of the `block` or `if` it names, or to the entry
    /// block of the `loop` it names. A branch that does not resolve states none (the verifier
    /// rejects it): empty is the *blind* answer, not the unknown one, and an analysis that runs
    /// on unverified IR would take what follows as unreachable.
    fn get_successor_regions(
        &self,
        _operands: &[Option<AttributeRef>],
    ) -> SmallVec<[RegionSuccessorInfo; 2]> {
        let mut infos = SmallVec::new();
        let Some(target) = self.target() else {
            return infos;
        };
        let target_op = target.borrow();
        if let Some(target_loop) = target_op.downcast_ref::<Loop>() {
            let header = target_loop.body().as_region_ref();
            let Some(entry) = header.borrow().entry_block_ref() else {
                return infos;
            };
            let values = entry
                .borrow()
                .arguments()
                .iter()
                .map(|argument| *argument as ValueRef)
                .collect();
            infos.push(RegionSuccessorInfo::Exiting {
                target,
                to: RegionBranchPoint::Child(header),
                values,
            });
        } else {
            infos.push(RegionSuccessorInfo::Exiting {
                target,
                to: RegionBranchPoint::Parent,
                values: result_values(&target_op),
            });
        }
        infos
    }
}

impl OpPrinter for Br {
    fn states_operand_types(&self) -> bool {
        true
    }

    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use runic_ir::ValueRange;

        let op = self.as_operation();
        printer.print_space();
        printer.print_decimal_integer(*self.get_level());
        let operands = op.operands().all();
        if !operands.is_empty() {
            printer.print_space();
            printer.print_value_uses(ValueRange::<2>::from(op.operands().all()));
        }
        print_attributes(printer, op);
        printer.print_space();
        printer.print_colon_type_list(operands.iter().map(|o| o.borrow().ty()));
    }
}

impl OpParser for Br {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let (span, literal) = parser.parse_integer_literal()?.into_parts();
        let level = literal.to_int(parser.context().i32(), span)?.signed() as i32;
        state.add_attribute("level", parser.context_rc().create_attribute::<I32Attr, _>(level));

        let mut operands = SmallVec::default();
        parser.parse_optional_operand_list(&mut operands)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        let mut types = SmallVec::<[Type; 4]>::default();
        parser.parse_colon_type_list(&mut types)?;
        let mut values = SmallVec::default();
        parser.resolve_operands(state.span, &operands, &types, &mut values)?;
        for value in values {
            state.add_operand(value);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// BrTable

/// A branch table: `wasm.br_table %i, %v [{ wasm.br 0 %v : (i32); }, { wasm.br 1 %v : (i32); }]
/// default { wasm.br 2 %v : (i32); } : (i32, i32) -> ();`. Wasm's `br_table`.
///
/// The first operand is the index, an `i32`; the others are the values the branch carries. There
/// is one region for each entry of the table, in order, then the default region. Control goes to
/// the region the index selects (an index at or past the number of entries selects the default,
/// and the index is unsigned), and each region holds one block with exactly one [Br], of the
/// values of the table. Each region may name a different label. The operation is a terminator with
/// no block successors, and never falls through: nothing follows it in its block, and control
/// leaves by the branch of the region it selects.
///
/// Why this shape: a branch states one target (an early exit has exactly one), and a table has
/// many. Holding one single-target [Br] per label shows every analysis only single-target exits,
/// which they all handle; and the table is itself a region operation, so the analyses reach the
/// branches. It is not counted as a label, and it forwards early exits.
///
/// The internal layout is that of `scf.index_switch`: the default region is the first region of the
/// operation, and the table entries follow.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = WasmDialect,
    traits(Terminator, SingleBlock, NoRegionArguments, HasRecursiveMemoryEffects, PropagateControlFlowBreak),
    implements(RegionBranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct BrTable {
    #[operand]
    index: AnyType,
    #[operands]
    values: AnyType,
    #[region]
    default_region: Region,
}

const BR_TABLE_RULE: &str = "takes an `i32` index and values, and each of its regions holds one \
                             block with exactly one `wasm.br` of those values";

impl BrTable {
    /// The number of entries of the table: the regions before the default.
    pub fn num_entries(&self) -> usize {
        self.as_operation().num_regions().saturating_sub(1)
    }

    /// The region of entry `index` of the table.
    pub fn entry_region(&self, index: usize) -> Option<RegionRef> {
        (index < self.num_entries())
            .then(|| self.as_operation().regions().iter().nth(index + 1).map(|r| r.as_region_ref()))
            .flatten()
    }

    /// The region that is taken when the index is `index` (entries first, then the default).
    fn selected_region(&self, index: u32) -> RegionRef {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.entry_region(index))
            .unwrap_or(self.default_region().as_region_ref())
    }
}

/// The index is an `i32`, the values are Wasm value types, and each region is one block with one
/// `wasm.br` of the table's values; and the interface's own rule holds.
impl Verify<dyn RegionBranchOpInterface> for BrTable {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let index = operand_type(op, 0);
        if index != op.context().i32() {
            return Err(rule_error(
                op,
                BR_TABLE_RULE,
                format!("its index has type `{}`", val_type_name(index)),
            ));
        }
        for (position, operand) in op.operands().iter().enumerate().skip(1) {
            let ty = operand.borrow().as_value_ref().borrow().ty();
            if !is_val_type(ty) {
                return Err(rule_error(
                    op,
                    BR_TABLE_RULE,
                    format!("its operand {position} has type `{}`", val_type_name(ty)),
                ));
            }
        }
        let table_values = op
            .operands()
            .iter()
            .skip(1)
            .map(|o| o.borrow().as_value_ref())
            .collect::<Vec<_>>();
        let count = op.num_regions();
        for (position, region) in op.regions().iter().enumerate() {
            let name = if position == 0 {
                String::from("its default region")
            } else {
                format!("its entry {}", position - 1)
            };
            let Some(entry) = region.entry_block_ref() else {
                return Err(rule_error(op, BR_TABLE_RULE, format!("{name} has no block")));
            };
            let entry = entry.borrow();
            let ops =
                entry.body().iter().map(|nested| nested.as_operation_ref()).collect::<Vec<_>>();
            if ops.len() != 1 {
                return Err(rule_error(
                    op,
                    BR_TABLE_RULE,
                    format!("{name} holds {} operations", ops.len()),
                ));
            }
            let nested = ops[0].borrow();
            if !nested.is::<Br>() {
                return Err(rule_error(
                    op,
                    BR_TABLE_RULE,
                    format!("{name} holds `{}`, not a `wasm.br`", nested.name()),
                ));
            }
            let carried =
                nested.operands().iter().map(|o| o.borrow().as_value_ref()).collect::<Vec<_>>();
            if carried != table_values {
                return Err(rule_error(
                    op,
                    BR_TABLE_RULE,
                    format!(
                        "the `wasm.br` of {name} carries {}, not the values of the table {}",
                        types_of(&carried),
                        types_of(&table_values)
                    ),
                ));
            }
        }
        debug_assert!(count >= 1 || op.regions().is_empty());
        if count == 0 {
            return Err(rule_error(op, BR_TABLE_RULE, String::from("it has no default region")));
        }
        verify_region_branch_entry_operands(self)
    }
}

impl RegionBranchOpInterface for BrTable {
    /// The regions take no arguments.
    fn get_entry_successor_operands(&self, _point: RegionBranchPoint) -> SuccessorOperandRange<'_> {
        SuccessorOperandRange::empty()
    }

    /// Every region may be entered when the index is not known, and only the selected one when it
    /// is.
    fn get_entry_successor_regions(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> RegionSuccessorIter<'_> {
        let mut infos = SmallVec::<[RegionSuccessorInfo; 2]>::default();
        match operands.first().and_then(constant_i32) {
            Some(index) => infos.push(RegionSuccessorInfo::Entering(self.selected_region(index))),
            None => {
                for region in self.as_operation().regions().iter() {
                    infos.push(RegionSuccessorInfo::Entering(region.as_region_ref()));
                }
            }
        }
        RegionSuccessorIter::new(self.as_operation(), infos)
    }

    /// Every region is entered from the parent. Nothing leaves a region to the table: control
    /// leaves by the `wasm.br` the region holds, which states its own edge.
    fn get_successor_regions(&self, point: RegionBranchPoint) -> RegionSuccessorIter<'_> {
        match point {
            RegionBranchPoint::Parent => RegionSuccessorIter::new(
                self.as_operation(),
                self.as_operation()
                    .regions()
                    .iter()
                    .map(|region| RegionSuccessorInfo::Entering(region.as_region_ref())),
            ),
            RegionBranchPoint::Child(_) => RegionSuccessorIter::new(self.as_operation(), []),
        }
    }

    fn is_repetitive_region(&self, _index: usize) -> bool {
        false
    }

    fn has_loop(&self) -> bool {
        false
    }
}

impl OpPrinter for BrTable {
    fn states_operand_types(&self) -> bool {
        true
    }

    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use runic_ir::ValueRange;

        let op = self.as_operation();
        printer.print_space();
        printer.print_value_uses(ValueRange::<2>::from(op.operands().all()));
        printer.print_space();
        *printer += const_text("[");
        for position in 0..self.num_entries() {
            if position > 0 {
                *printer += const_text(", ");
            }
            if let Some(region) = self.entry_region(position) {
                printer.print_region(&region.borrow());
            }
        }
        *printer += const_text("]");
        printer.print_space();
        printer.print_keyword("default");
        printer.print_space();
        printer.print_region(&self.default_region());
        print_attributes(printer, op);
        printer.print_space();
        printer.print_colon_type_list(op.operands().all().iter().map(|o| o.borrow().ty()));
    }
}

impl OpParser for BrTable {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let mut operands = SmallVec::<[runic_ir::parse::UnresolvedOperand; 2]>::new();
        operands.push(parser.parse_operand(true)?);
        parser.parse_trailing_operand_list(&mut operands, Delimiter::None)?;

        parser.parse_lbracket()?;
        let mut entries = SmallVec::<[RegionRef; 4]>::new();
        if !parser.parse_optional_rbracket()? {
            loop {
                let region = parser.context().create_region();
                parser.parse_region(region, &[], false)?;
                entries.push(region);
                if !parser.parse_optional_comma()? {
                    break;
                }
            }
            parser.parse_rbracket()?;
        }
        parser.parse_custom_keyword("default")?;
        let default = parser.context().create_region();
        parser.parse_region(default, &[], false)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let mut types = SmallVec::<[Type; 4]>::default();
        parser.parse_colon_type_list(&mut types)?;
        let mut values = SmallVec::default();
        parser.resolve_operands(state.span, &operands, &types, &mut values)?;
        add_condition_and_values(state, values);
        // The default region is declared first on the operation; the entries follow it.
        state.add_region(default);
        for region in entries {
            state.add_region(region);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Unreachable

/// A trap: `wasm.unreachable;`. Wasm's `unreachable`.
///
/// It is a terminator: control never continues past it, because it traps. **It differs from
/// `ub.unreachable`, which asserts that the point is never reached and gives the optimizer license
/// to assume so**: executing this one is defined, and traps. As a trap is not an effect, but the
/// operation must not be removed or moved past what it follows, it declares a write on the default
/// memory resource, as `ub.unreachable` does, so no pass erases it, and an operation that holds one
/// is not trivially dead.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[effects(MemoryEffect(MemoryEffect::Write))]
#[operation(
    dialect = WasmDialect,
    traits(Terminator),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct Unreachable {}

/// A reference to an [Unreachable].
pub type UnreachableRef = runic_ir::UnsafeIntrusiveEntityRef<Unreachable>;
