//! The patterns of structured control flow: `scf.if`, `scf.while`, `scf.index_switch`,
//! `scf.break`, `scf.continue` and `ub.unreachable`, lowered to `wasm.block`, `wasm.loop`,
//! `wasm.if`, `wasm.br`, `wasm.br_table` and `wasm.unreachable`; and the errors of unstructured
//! control flow.
//!
//! Each pattern of an `scf` construct builds the `wasm` constructs and moves the bodies into them
//! ([ConversionPatternRewriter::inline_block_at_end]); the moved operations are converted after
//! the constructs that now hold them. A body's `scf.yield` (and a loop's `scf.condition`) is
//! replaced here, by a `wasm.end` or a `wasm.br`; every other terminator stays, and has its own
//! pattern or is legal: a `builtin.ret`, a tail call, `ub.unreachable`, `scf.break`,
//! `scf.continue`. The constructs take no parameters: values from outside are used directly.
//! Conditions are carried `i1`s, `i32`s that are 0 or 1, which `wasm.if` tests for not zero; every
//! result and parameter has the carrier of the `scf` operation's type.
//!
//! * `scf.if %c then { A } else { B } : (i1) -> (R..)` becomes `wasm.if %c then { A' } else
//!   { B' } : () -> (R'..)`, where a body's `scf.yield %v..` becomes `wasm.end %v'..`.
//! * `scf.while %init.. before { ^b(%i..): B; scf.condition %c, %f.. } after { ^a(%j..): A;
//!   scf.yield %n.. }` becomes a `wasm.loop %init'..` whose body is `B'`, then `wasm.if %c' then
//!   { A'; wasm.br 1 %n'.. } else { wasm.end }`, then `wasm.end %f'..` (falling off the end leaves
//!   the loop with its results); in `A'` the arguments `%j..` are the forwarded `%f..`. When some
//!   `scf.break` leaves the loop, the loop is the body of a `wasm.block` of the loop's result
//!   types, which ends in `wasm.end` of the loop's results, and a break leaves that block. A
//!   `before` body that ends in an exit (a break or a function exit) has no `wasm.if`, and its
//!   `after` body is dropped (it is never entered); an `after` body that ends in an exit has no
//!   branch back.
//! * `scf.index_switch %s` of the cases `k_0 .. k_(n-1)` (in the order written) becomes, when it
//!   is dense ([DENSE_SWITCH]), `n + 2` nested `wasm.block`s: the outermost has the switch's
//!   result types; inside it is the block of the default, inside that the block of case `n - 1`,
//!   and so on to the block of case 0, the innermost, which holds a `wasm.br_table` on
//!   `%s - min` (no `wasm.sub` when `min` is 0) with one entry per value from `min` to `max`: a
//!   value that is case `j` branches to label `j` (the block of case `j`), any other to label `n`
//!   (the block of the default), and so does the default of the table. Right after the block of
//!   case `j` ends comes case `j`'s body, whose yield becomes `wasm.br (n - j)` to the outermost
//!   block; after the block of the default comes the default body, whose yield becomes the
//!   `wasm.end` of the outermost block. A switch that is not dense becomes `wasm.if (wasm.eq %s,
//!   k_0)` of the switch's result types, whose first region is case 0's body and whose second
//!   holds the `wasm.if` of the next case and a `wasm.end` of its results; the second region of
//!   the last is the default body. A switch with no case is a `wasm.block` of the default body.
//! * `scf.break N %v..` and `scf.continue N %v..` become `wasm.br L %v'..`, `L` from
//!   [labels](super::labels).
//! * `ub.unreachable` becomes `wasm.unreachable`, which traps.
//! * `cf.br`, `cf.cond_br` and `cf.switch` are errors: `cfg-to-scf{exits=in-place}` structures
//!   them first.
//!
//! A yielded value, a condition or a forwarded value is defined in the moved body and converted
//! after this pattern, so it still has its old type when the `wasm.end` or `wasm.br` is built: the
//! pattern reads it through a `builtin.unrealized_conversion_cast` to its carrier, which folds
//! away with the cast the conversion leaves when it converts the value's producer.

use alloc::{format, rc::Rc, vec::Vec};

use runic_dialect_cf as cf;
use runic_dialect_scf as scf;
use runic_dialect_ub as ub;
use runic_dialect_wasm as wasm;
use runic_ir::{
    BlockRef, Context, Operation, OperationName, OperationRef, RegionExitKind,
    RegionExitTerminatorOpInterface, RegionRef, Report, SmallVec, SourceSpan, Spanned, Type, Value,
    ValueRef, WalkResult,
    conversion::{
        ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConvertedOperands,
        TypeConverter,
    },
    diagnostics::report,
    dialects::builtin::UnrealizedConversionCast,
    patterns::{Pattern, PatternInfo},
};

use super::labels::{Labels, Lowered};
use crate::{
    carried::const_int,
    convert::calls::{pattern_info, registered},
};

/// When an `scf.index_switch` is dense enough to become a branch table: it has at least
/// `min_cases` cases, and the values from its least case to its greatest are at most
/// `entries_per_case` times as many as its cases. A switch that is not dense becomes a chain of
/// `wasm.if`s, one comparison per case.
pub const DENSE_SWITCH: DenseSwitch = DenseSwitch {
    min_cases: 3,
    entries_per_case: 4,
};

/// The threshold of [DENSE_SWITCH].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DenseSwitch {
    /// The fewest cases a branch table is made for.
    pub min_cases: usize,
    /// The most entries of the table per case.
    pub entries_per_case: u64,
}

impl DenseSwitch {
    /// Whether a switch of the case values `cases` is dense.
    pub fn is_dense(&self, cases: &[i32]) -> bool {
        let (Some(min), Some(max)) = (cases.iter().min(), cases.iter().max()) else {
            return false;
        };
        let entries = (i64::from(*max) - i64::from(*min) + 1) as u64;
        cases.len() >= self.min_cases && entries <= self.entries_per_case * cases.len() as u64
    }
}

/// Adds the patterns of this module, which record the constructs they create in `labels`.
pub fn populate(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
    labels: &Labels,
) -> Result<(), Report> {
    let context = patterns.context();
    let rules: [(Rule, &'static str, Root); 11] = [
        (Rule::If, "wasm-scf-if", |c, n, g| pattern_info::<scf::If>(c, n, g)),
        (Rule::While, "wasm-scf-while", |c, n, g| pattern_info::<scf::While>(c, n, g)),
        (Rule::IndexSwitch, "wasm-scf-index-switch", |c, n, g| {
            pattern_info::<scf::IndexSwitch>(c, n, g)
        }),
        (Rule::Exit(RegionExitKind::Break), "wasm-scf-break", |c, n, g| {
            pattern_info::<scf::Break>(c, n, g)
        }),
        (Rule::Exit(RegionExitKind::Continue), "wasm-scf-continue", |c, n, g| {
            pattern_info::<scf::Continue>(c, n, g)
        }),
        (Rule::Unreachable, "wasm-ub-unreachable", |c, n, g| {
            pattern_info::<ub::Unreachable>(c, n, g)
        }),
        (Rule::Unstructured, "wasm-cf-br", |c, n, g| pattern_info::<cf::Br>(c, n, g)),
        (Rule::Unstructured, "wasm-cf-cond-br", |c, n, g| {
            pattern_info::<cf::CondBr>(c, n, g)
        }),
        (Rule::Unstructured, "wasm-cf-switch", |c, n, g| {
            pattern_info::<cf::Switch>(c, n, g)
        }),
        (Rule::Terminator, "wasm-scf-yield", |c, n, g| {
            pattern_info::<scf::Yield>(c, n, g)
        }),
        (Rule::Terminator, "wasm-scf-condition", |c, n, g| {
            pattern_info::<scf::Condition>(c, n, g)
        }),
    ];
    for (rule, name, root) in rules {
        patterns.push(ControlPattern {
            info: root(&context, name, rule.generated(&context)),
            converter: converter.clone(),
            rule,
            labels: labels.clone(),
        });
    }
    Ok(())
}

/// The information of a pattern named by the second argument, rooted at the operation the
/// function is for, that creates the operations of the third.
type Root = fn(&Rc<Context>, &'static str, Vec<OperationName>) -> PatternInfo;

/// What each pattern of this module lowers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Rule {
    /// `scf.if`: a `wasm.if` of the carried results.
    If,
    /// `scf.while`: a `wasm.loop`, in a `wasm.block` when a break leaves it.
    While,
    /// `scf.index_switch`: a branch table or a chain of `wasm.if`s (see the module
    /// documentation).
    IndexSwitch,
    /// `scf.break` or `scf.continue`: a `wasm.br`.
    Exit(RegionExitKind),
    /// `ub.unreachable`: `wasm.unreachable`.
    Unreachable,
    /// `cf.br`, `cf.cond_br`, `cf.switch`: an error.
    Unstructured,
    /// `scf.yield` and `scf.condition`, which the pattern of the construct they end replaces: one
    /// met here is under an operation no pattern of this module lowers, an error.
    Terminator,
}

impl Rule {
    /// The names of the operations a pattern of this rule may create.
    fn generated(self, context: &Rc<Context>) -> Vec<OperationName> {
        macro_rules! names {
            ($($op:ty),*) => { alloc::vec![$(registered::<$op>(context)),*] };
        }
        match self {
            Rule::If | Rule::While | Rule::IndexSwitch => names!(
                wasm::BlockOp,
                wasm::Loop,
                wasm::If,
                wasm::End,
                wasm::Br,
                wasm::BrTable,
                wasm::Const,
                wasm::Sub,
                wasm::Eq,
                UnrealizedConversionCast
            ),
            Rule::Exit(_) => names!(wasm::Br),
            Rule::Unreachable => names!(wasm::Unreachable),
            Rule::Unstructured | Rule::Terminator => Vec::new(),
        }
    }
}

/// The pattern of one [Rule].
struct ControlPattern {
    info: PatternInfo,
    converter: TypeConverter,
    rule: Rule,
    labels: Labels,
}

impl Pattern for ControlPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for ControlPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        match self.rule {
            Rule::Unstructured => return Err(unstructured(&op.borrow())),
            Rule::Terminator => return Err(stray_terminator(&op.borrow())),
            _ => {}
        }
        let Some(values) = operands.all_one() else {
            return Ok(false);
        };
        let values: SmallVec<[ValueRef; 4]> = values;
        let span = op.borrow().span();
        match self.rule {
            Rule::Unstructured | Rule::Terminator => Ok(false),
            Rule::Unreachable => {
                rewriter.set_insertion_point_before(op);
                rewriter.create_op::<wasm::Unreachable, _>(span, ())?;
                rewriter.erase_op(op)?;
                Ok(true)
            }
            Rule::Exit(kind) => {
                let levels = {
                    let op_ref = op.borrow();
                    match kind {
                        RegionExitKind::Break => {
                            op_ref.downcast_ref::<scf::Break>().map(|exit| *exit.get_levels())
                        }
                        RegionExitKind::Continue => {
                            op_ref.downcast_ref::<scf::Continue>().map(|exit| *exit.get_levels())
                        }
                    }
                };
                let Some(levels) = levels else {
                    return Ok(false);
                };
                let level = super::labels::level_of(&op.borrow(), levels, kind, &self.labels)?;
                rewriter.set_insertion_point_before(op);
                rewriter.create_op::<wasm::Br, _>(span, (level, values.to_vec()))?;
                rewriter.erase_op(op)?;
                Ok(true)
            }
            Rule::If | Rule::IndexSwitch if values.is_empty() => Ok(false),
            Rule::If => self.lower_if(rewriter, op, values[0], span),
            Rule::While => self.lower_while(rewriter, op, &values, span),
            Rule::IndexSwitch => self.lower_switch(rewriter, op, values[0], span),
        }
    }
}

/// The error of `cf.br`, `cf.cond_br` or `cf.switch` `op`.
fn unstructured(op: &Operation) -> Report {
    report!(
        message: "unstructured control flow reached `convert-to-wasm`: run \
                  `cfg-to-scf{{exits=in-place}}` first",
        labels: [primary(
            op.span(),
            format!("this `{}` is unstructured control flow", op.name())
        )],
    )
}

/// The error of an `scf.yield` or `scf.condition` `op` that no pattern of a construct replaced.
fn stray_terminator(op: &Operation) -> Report {
    let parent = op.parent_op().map_or_else(
        || alloc::string::String::from("nothing"),
        |parent| format!("`{}`", parent.borrow().name()),
    );
    report!(
        message: (format!(
            "`convert-to-wasm` lowers `{}` only with the `scf.if`, `scf.while` or \
             `scf.index_switch` it ends",
            op.name()
        )),
        labels: [primary(
            op.span(),
            format!("this `{}` ends a region of {parent}", op.name())
        )],
    )
}

/// The entry block of region `index` of `op`; an error naming `op` when there is no such region
/// or it has no block.
fn entry(regions: &[RegionRef], index: usize, op: OperationRef) -> Result<BlockRef, Report> {
    regions.get(index).and_then(|region| region.borrow().entry_block_ref()).ok_or_else(|| {
        let op = op.borrow();
        report!(
            message: (format!("`convert-to-wasm` cannot lower `{}` with an empty body", op.name())),
            labels: [primary(op.span(), "a body of this operation has no block")],
        )
    })
}

/// The values of the operands of `op`.
fn operand_values(op: OperationRef) -> SmallVec<[ValueRef; 4]> {
    op.borrow()
        .operands()
        .iter()
        .map(|operand| operand.borrow().as_value_ref())
        .collect()
}

/// The values of the results of `op`.
fn result_values(op: OperationRef) -> SmallVec<[ValueRef; 4]> {
    op.borrow()
        .results()
        .iter()
        .map(|result| result.borrow().as_value_ref())
        .collect()
}

/// The regions of `op`, in order.
fn regions_of(op: OperationRef) -> SmallVec<[RegionRef; 4]> {
    op.borrow().regions().iter().map(|region| region.as_region_ref()).collect()
}

/// Gives `op`, which was built without results, results of the types `types`.
fn add_results(
    rw: &ConversionPatternRewriter,
    mut op: OperationRef,
    types: &[Type],
    span: SourceSpan,
) {
    for ty in types {
        let index = op.borrow().num_results() as u16;
        let result = rw.context().make_result(span, *ty, op, index);
        op.borrow_mut().results_mut().push(result);
    }
}

/// How a moved body's `scf.yield` is replaced.
#[derive(Copy, Clone)]
enum YieldTo {
    /// By `wasm.end` of the yielded values.
    End,
    /// By `wasm.br` of the yielded values, to this label.
    Br(i32),
}

impl ControlPattern {
    /// The carriers of the result types of `op`.
    fn carried_results(
        &self,
        rw: &ConversionPatternRewriter,
        op: OperationRef,
    ) -> Result<SmallVec<[Type; 4]>, Report> {
        let types = op
            .borrow()
            .results()
            .iter()
            .map(|result| result.borrow().ty())
            .collect::<SmallVec<[Type; 4]>>();
        types
            .into_iter()
            .map(|ty| self.converter.convert_type_1_to_1(rw.context(), ty))
            .collect()
    }

    /// `value` in its carrier: itself when its type is its carrier; the input of the cast that
    /// defines it when that cast is of one value of the carrier (the cast the conversion leaves
    /// for a producer converted earlier); otherwise a new cast of it to its carrier, at the
    /// insertion point.
    fn carried(
        &self,
        rw: &mut ConversionPatternRewriter,
        value: ValueRef,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let ty = value.borrow().ty();
        let carrier = self.converter.convert_type_1_to_1(rw.context(), ty)?;
        if carrier == ty {
            return Ok(value);
        }
        let defining = value.borrow().get_defining_op();
        if let Some(defining) = defining {
            let defining = defining.borrow();
            if defining.is::<UnrealizedConversionCast>()
                && defining.num_operands() == 1
                && defining.num_results() == 1
            {
                let input = defining.operands()[0].borrow().as_value_ref();
                if input.borrow().ty() == carrier {
                    return Ok(input);
                }
            }
        }
        let cast = rw.create_op::<UnrealizedConversionCast, _>(span, (alloc::vec![value],))?;
        let cast = cast.as_operation_ref();
        add_results(rw, cast, &[carrier], span);
        Ok(result_values(cast)[0])
    }

    /// [Self::carried] of each of `values`.
    fn all_carried(
        &self,
        rw: &mut ConversionPatternRewriter,
        values: &[ValueRef],
        span: SourceSpan,
    ) -> Result<Vec<ValueRef>, Report> {
        values.iter().map(|value| self.carried(rw, *value, span)).collect()
    }

    /// Records `op`, a construct this pattern created, as `lowered` in the table of labels.
    fn record(&self, op: OperationRef, lowered: Lowered) {
        self.labels.borrow_mut().insert(op, lowered);
    }

    /// Builds a `wasm.block` of no parameters and the results `results` at the insertion point,
    /// recorded as `lowered`, and puts the insertion point at the end of its body.
    fn block(
        &self,
        rw: &mut ConversionPatternRewriter,
        results: &[Type],
        lowered: Lowered,
        span: SourceSpan,
    ) -> Result<(OperationRef, BlockRef), Report> {
        let op = rw.create_op::<wasm::BlockOp, _>(span, (Vec::<ValueRef>::new(),))?;
        let op = op.as_operation_ref();
        add_results(rw, op, results, span);
        self.record(op, lowered);
        let body = rw.create_block(regions_of(op)[0], &[])?;
        Ok((op, body))
    }

    /// Builds a `wasm.if` on `condition` of no parameters and the results `results` at the
    /// insertion point, recorded as an inner construct; gives each region its block, and answers
    /// the `if` and the two blocks. The insertion point is in the second block.
    fn if_op(
        &self,
        rw: &mut ConversionPatternRewriter,
        condition: ValueRef,
        results: &[Type],
        span: SourceSpan,
    ) -> Result<(OperationRef, BlockRef, BlockRef), Report> {
        let op = rw.create_op::<wasm::If, _>(span, (condition, Vec::<ValueRef>::new()))?;
        let op = op.as_operation_ref();
        add_results(rw, op, results, span);
        self.record(op, Lowered::Inner);
        let regions = regions_of(op);
        let then_block = rw.create_block(regions[0], &[])?;
        let else_block = rw.create_block(regions[1], &[])?;
        Ok((op, then_block, else_block))
    }

    /// Moves the body `source` to the end of `destination` and replaces its terminator when it is
    /// an `scf.yield` (see [YieldTo]); any other terminator stays.
    fn move_body(
        &self,
        rw: &mut ConversionPatternRewriter,
        source: BlockRef,
        destination: BlockRef,
        to: YieldTo,
    ) -> Result<(), Report> {
        rw.inline_block_at_end(source, destination, &[])?;
        self.replace_yield(rw, destination, to)
    }

    /// Replaces the terminator of `block` when it is an `scf.yield` (see [YieldTo]).
    fn replace_yield(
        &self,
        rw: &mut ConversionPatternRewriter,
        block: BlockRef,
        to: YieldTo,
    ) -> Result<(), Report> {
        let Some(terminator) = block.borrow().terminator() else {
            return Ok(());
        };
        if !terminator.borrow().is::<scf::Yield>() {
            return Ok(());
        }
        let span = terminator.borrow().span();
        let values = operand_values(terminator);
        rw.set_insertion_point_before(terminator);
        let values = self.all_carried(rw, &values, span)?;
        match to {
            YieldTo::End => {
                rw.create_op::<wasm::End, _>(span, (values,))?;
            }
            YieldTo::Br(level) => {
                rw.create_op::<wasm::Br, _>(span, (level, values))?;
            }
        }
        rw.erase_op(terminator)
    }

    /// Replaces `op` by the results of `construct`, or erases it when it has none.
    fn replace(
        &self,
        rw: &mut ConversionPatternRewriter,
        op: OperationRef,
        construct: OperationRef,
    ) -> Result<bool, Report> {
        let results = result_values(construct);
        if results.is_empty() {
            rw.erase_op(op)?;
        } else {
            rw.replace_op(op, &results)?;
        }
        Ok(true)
    }

    /// The pattern of `scf.if` (see the module documentation).
    fn lower_if(
        &self,
        rw: &mut ConversionPatternRewriter,
        op: OperationRef,
        condition: ValueRef,
        span: SourceSpan,
    ) -> Result<bool, Report> {
        let results = self.carried_results(rw, op)?;
        let regions = regions_of(op);
        let (then_source, else_source) = (entry(&regions, 0, op)?, entry(&regions, 1, op)?);
        rw.set_insertion_point_before(op);
        let (if_op, then_block, else_block) = self.if_op(rw, condition, &results, span)?;
        self.move_body(rw, then_source, then_block, YieldTo::End)?;
        self.move_body(rw, else_source, else_block, YieldTo::End)?;
        self.replace(rw, op, if_op)
    }

    /// The pattern of `scf.while` (see the module documentation).
    fn lower_while(
        &self,
        rw: &mut ConversionPatternRewriter,
        op: OperationRef,
        inits: &[ValueRef],
        span: SourceSpan,
    ) -> Result<bool, Report> {
        let results = self.carried_results(rw, op)?;
        let regions = regions_of(op);
        let (before, after) = (entry(&regions, 0, op)?, entry(&regions, 1, op)?);
        let wrapped = is_left_by_a_break(op);
        rw.set_insertion_point_before(op);
        let wrapper = if wrapped {
            Some(self.block(rw, &results, Lowered::Loop { wrapped: true }, span)?)
        } else {
            None
        };

        let loop_op = rw.create_op::<wasm::Loop, _>(span, (inits.to_vec(),))?;
        let loop_op = loop_op.as_operation_ref();
        add_results(rw, loop_op, &results, span);
        self.record(
            loop_op,
            if wrapped {
                Lowered::Inner
            } else {
                Lowered::Loop { wrapped: false }
            },
        );
        let parameters =
            inits.iter().map(|value| value.borrow().ty()).collect::<SmallVec<[Type; 4]>>();
        let body = rw.create_block(regions_of(loop_op)[0], &parameters)?;
        let arguments = body.borrow().argument_values().collect::<SmallVec<[ValueRef; 4]>>();
        let ranges = arguments.iter().map(core::slice::from_ref).collect::<Vec<_>>();
        rw.inline_block_at_end(before, body, &ranges)?;

        let terminator = body.borrow().terminator();
        if let Some(condition) = terminator.filter(|term| term.borrow().is::<scf::Condition>()) {
            let term_span = condition.borrow().span();
            let operands = operand_values(condition);
            let Some((&flag, forwarded)) = operands.split_first() else {
                return Err(stray_terminator(&condition.borrow()));
            };
            rw.set_insertion_point_before(condition);
            let flag = self.carried(rw, flag, term_span)?;
            let (_, then_block, else_block) = self.if_op(rw, flag, &[], term_span)?;
            let ranges = forwarded.iter().map(core::slice::from_ref).collect::<Vec<_>>();
            rw.inline_block_at_end(after, then_block, &ranges)?;
            // The `if` is label 0 and the loop label 1.
            self.replace_yield(rw, then_block, YieldTo::Br(1))?;
            rw.set_insertion_point_to_end(else_block);
            rw.create_op::<wasm::End, _>(term_span, (Vec::<ValueRef>::new(),))?;
            rw.set_insertion_point_before(condition);
            let forwarded = self.all_carried(rw, forwarded, term_span)?;
            rw.create_op::<wasm::End, _>(term_span, (forwarded,))?;
            rw.erase_op(condition)?;
        }

        let outer = match wrapper {
            Some((block_op, block_body)) => {
                rw.set_insertion_point_to_end(block_body);
                rw.create_op::<wasm::End, _>(span, (result_values(loop_op).to_vec(),))?;
                block_op
            }
            None => loop_op,
        };
        self.replace(rw, op, outer)
    }

    /// The pattern of `scf.index_switch` (see the module documentation).
    fn lower_switch(
        &self,
        rw: &mut ConversionPatternRewriter,
        op: OperationRef,
        selector: ValueRef,
        span: SourceSpan,
    ) -> Result<bool, Report> {
        let results = self.carried_results(rw, op)?;
        let cases = {
            let op_ref = op.borrow();
            let Some(switch) = op_ref.downcast_ref::<scf::IndexSwitch>() else {
                return Ok(false);
            };
            switch.cases().iter().copied().collect::<Vec<i32>>()
        };
        let regions = regions_of(op);
        let default = entry(&regions, 0, op)?;
        let bodies = (1..regions.len())
            .map(|index| entry(&regions, index, op))
            .collect::<Result<Vec<_>, _>>()?;
        if bodies.len() != cases.len() {
            return Ok(false);
        }
        rw.set_insertion_point_before(op);
        let outer = if cases.is_empty() {
            let (block_op, body) = self.block(rw, &results, Lowered::Inner, span)?;
            self.move_body(rw, default, body, YieldTo::End)?;
            block_op
        } else if DENSE_SWITCH.is_dense(&cases) {
            self.table(rw, selector, &cases, &bodies, default, &results, span)?
        } else {
            self.chain(rw, selector, &cases, &bodies, default, &results, span)?
        };
        self.replace(rw, op, outer)
    }

    /// The branch table of a dense switch (see the module documentation), built at the insertion
    /// point; answers the outermost block.
    #[allow(clippy::too_many_arguments)]
    fn table(
        &self,
        rw: &mut ConversionPatternRewriter,
        selector: ValueRef,
        cases: &[i32],
        bodies: &[BlockRef],
        default: BlockRef,
        results: &[Type],
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let n = cases.len();
        let (outer, outer_body) = self.block(rw, results, Lowered::Inner, span)?;
        let (_, default_body) = self.block(rw, &[], Lowered::Inner, span)?;
        // `inner[j]` is the block of case `j`; each block of a case is built in the next one out.
        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            blocks.push(self.block(rw, &[], Lowered::Inner, span)?);
        }
        blocks.reverse();
        // The insertion point is in the block of case 0.
        let min = cases.iter().copied().min().unwrap_or(0);
        let max = cases.iter().copied().max().unwrap_or(0);
        let i32_type = rw.context().i32();
        let index = if min == 0 {
            selector
        } else {
            let least = const_int(rw, min as u32 as u64, i32_type, span)?;
            let sub = rw.create_op::<wasm::Sub, _>(span, (selector, least))?;
            result_values(sub.as_operation_ref())[0]
        };
        let entries = (i64::from(max) - i64::from(min) + 1) as usize;
        let table = rw.create_op::<wasm::BrTable, _>(span, (index, Vec::<ValueRef>::new()))?;
        let table = table.as_operation_ref();
        let mut owner = table;
        for _ in 0..entries {
            let region = rw.context().create_region();
            owner.borrow_mut().regions_mut().push_back(region);
        }
        let table_regions = regions_of(table);
        let default_label = n as i32;
        let label_of = |value: i64| {
            cases
                .iter()
                .position(|case| i64::from(*case) == value)
                .map_or(default_label, |position| position as i32)
        };
        for (position, region) in table_regions.iter().enumerate() {
            let label = if position == 0 {
                default_label
            } else {
                label_of(i64::from(min) + position as i64 - 1)
            };
            rw.create_block(*region, &[])?;
            rw.create_op::<wasm::Br, _>(span, (label, Vec::<ValueRef>::new()))?;
        }
        // Case `j`'s body follows its block, in the block of case `j + 1` or of the default; it
        // is inside `n - j - 1` blocks of cases, the block of the default and the outermost.
        for (j, body) in bodies.iter().enumerate() {
            let holder = blocks.get(j + 1).map_or(default_body, |(_, block)| *block);
            self.move_body(rw, *body, holder, YieldTo::Br((n - j) as i32))?;
        }
        self.move_body(rw, default, outer_body, YieldTo::End)?;
        Ok(outer)
    }

    /// The chain of `wasm.if`s of a switch that is not dense (see the module documentation), built at
    /// the insertion point; answers the outermost `wasm.if`.
    #[allow(clippy::too_many_arguments)]
    fn chain(
        &self,
        rw: &mut ConversionPatternRewriter,
        selector: ValueRef,
        cases: &[i32],
        bodies: &[BlockRef],
        default: BlockRef,
        results: &[Type],
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let i32_type = rw.context().i32();
        let mut outermost = None;
        let mut holder: Option<BlockRef> = None;
        for (case, body) in cases.iter().zip(bodies.iter()) {
            let constant = const_int(rw, *case as u32 as u64, i32_type, span)?;
            let equal = rw.create_op::<wasm::Eq, _>(span, (selector, constant))?;
            let equal = result_values(equal.as_operation_ref())[0];
            let (if_op, then_block, else_block) = self.if_op(rw, equal, results, span)?;
            self.move_body(rw, *body, then_block, YieldTo::End)?;
            if let Some(enclosing_else) = holder {
                // The `wasm.if` of this case is in the second region of the one before it, which
                // ends with its results.
                rw.set_insertion_point_to_end(enclosing_else);
                rw.create_op::<wasm::End, _>(span, (result_values(if_op).to_vec(),))?;
            }
            outermost.get_or_insert(if_op);
            holder = Some(else_block);
            rw.set_insertion_point_to_end(else_block);
        }
        let (Some(outermost), Some(last_else)) = (outermost, holder) else {
            return Err(Report::msg("a chain of comparisons needs at least one case"));
        };
        self.move_body(rw, default, last_else, YieldTo::End)?;
        Ok(outermost)
    }
}

/// Whether some `scf.break` nested in `op`, an `scf.while`, leaves it.
fn is_left_by_a_break(op: OperationRef) -> bool {
    let op_ref = op.borrow();
    op_ref.regions().iter().any(|region| {
        region
            .prewalk(|nested: &Operation| {
                if let Some(exit) = nested.downcast_ref::<scf::Break>()
                    && exit.get_exit_target() == Some(op)
                {
                    return WalkResult::<()>::Break(());
                }
                WalkResult::Continue(())
            })
            .was_interrupted()
    })
}
