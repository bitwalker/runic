use alloc::{format, rc::Rc, string::String, vec, vec::Vec};

use smallvec::SmallVec;

use super::{
    ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConversionTarget,
    ConvertedOperands, FrozenConversionPatternSet, Legality, LegalizationGraph, TrackedMutations,
    TypeConverter, reconcile_unrealized_casts,
    rewriter::{CastKind, DriverCasts, type_list, types_of},
};
use crate::{
    FxHashSet, OperationName, OperationRef, Report, Spanned, Usable, ValueRef, WalkResult,
    dialects::builtin::UnrealizedConversionCast,
};

/// Dialect conversion mode.
///
/// Only full conversion is implemented today. The additional variants reserve the API shape for
/// the partial and analysis modes described by the design document.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ConversionMode {
    /// Require every operation under the root to be legal after conversion.
    Full,
    /// Reserved for MLIR-style partial conversion, where already-legal operations may remain
    /// without forcing every unknown operation to legalize.
    Partial,
    /// Reserved for analysis-only conversion that reports whether conversion could succeed without
    /// mutating IR.
    Analysis,
}

/// Configuration for the dialect conversion driver.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ConversionConfig {
    mode: ConversionMode,
    reconcile_unrealized_casts: bool,
    verify_after_conversion: bool,
    max_iterations: usize,
}

impl ConversionConfig {
    /// Return the configured conversion mode.
    #[inline]
    pub const fn mode(&self) -> ConversionMode {
        self.mode
    }

    /// Return whether the converted root is recursively verified after legality succeeds.
    #[inline]
    pub const fn verify_after_conversion(&self) -> bool {
        self.verify_after_conversion
    }

    /// Return whether temporary `builtin.unrealized_conversion_cast` operations are reconciled.
    #[inline]
    pub const fn reconcile_unrealized_casts(&self) -> bool {
        self.reconcile_unrealized_casts
    }

    /// Return the maximum number of pattern applications allowed during conversion.
    #[inline]
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Set the conversion mode.
    ///
    /// The current driver accepts only [`ConversionMode::Full`]; other modes are reserved API.
    pub fn with_mode(&mut self, mode: ConversionMode) -> &mut Self {
        self.mode = mode;
        self
    }

    /// Enable or disable recursive IR verification after conversion.
    pub fn with_verify_after_conversion(&mut self, yes: bool) -> &mut Self {
        self.verify_after_conversion = yes;
        self
    }

    /// Enable or disable the folding and realization of the casts the driver inserts.
    ///
    /// When enabled (the default), the casts whose ends agree are folded away when the patterns
    /// have run, and each cast the driver inserted that remains is realized by the type
    /// converter's materialization callbacks; one that none realizes is an error. When disabled,
    /// nothing is folded or realized, and the casts the driver inserted stay in the IR.
    pub fn with_reconcile_unrealized_casts(&mut self, yes: bool) -> &mut Self {
        self.reconcile_unrealized_casts = yes;
        self
    }

    /// Set the maximum number of successful pattern applications.
    ///
    /// This guards against rewrite cycles in conversion patterns.
    pub fn with_max_iterations(&mut self, max_iterations: usize) -> &mut Self {
        self.max_iterations = max_iterations;
        self
    }
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            mode: ConversionMode::Full,
            reconcile_unrealized_casts: true,
            verify_after_conversion: true,
            max_iterations: 1024,
        }
    }
}

/// Result of a conversion driver run.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ConversionResult {
    changed: bool,
    converted_ops: usize,
}

impl ConversionResult {
    /// Return true when at least one pattern application or cast reconciliation changed the IR.
    #[inline]
    pub const fn changed(&self) -> bool {
        self.changed
    }

    /// Return the number of successful conversion pattern applications.
    #[inline]
    pub const fn converted_ops(&self) -> usize {
        self.converted_ops
    }
}

/// Apply full dialect conversion to `root`.
///
/// Full conversion requires every operation under `root` to be legal for `target`, except for
/// operations nested under recursively legal operations.
///
/// Patterns are tried according to the legalization graph derived from the target and each
/// pattern's generated-op metadata. A pattern is given each operand as the values it became
/// ([ConvertedOperands]) and may replace each result by a range of values
/// ([ConversionPatternRewriter::replace_op_with_multiple]); the driver relates a value to the
/// values it became by the `builtin.unrealized_conversion_cast`s it inserts. When the patterns
/// have run, those casts whose ends agree are folded away and each that remains is realized by
/// the materialization callbacks of the type converter that caused it ([TypeConverter]); one that
/// none realizes is an error (see [ConversionConfig::with_reconcile_unrealized_casts]).
///
/// A pattern that returns `Ok(false)` must not mutate IR, and a pattern that returns `Ok(true)`
/// must mutate IR. Violating either contract is reported as a conversion error because this
/// driver intentionally does not roll changes back.
pub fn apply_full_conversion(
    root: OperationRef,
    target: ConversionTarget,
    patterns: ConversionPatternSet,
    config: ConversionConfig,
) -> Result<ConversionResult, Report> {
    if config.mode() != ConversionMode::Full {
        return Err(Report::msg("apply_full_conversion requires ConversionMode::Full"));
    }

    let frozen_patterns = FrozenConversionPatternSet::new(patterns);
    let graph = LegalizationGraph::new(&target, &frozen_patterns);
    let mut driver = FullConversionDriver {
        root,
        target: &target,
        graph,
        context: target.context(),
        casts: DriverCasts::default(),
        config,
        result: ConversionResult {
            changed: false,
            converted_ops: 0,
        },
    };

    driver.run()?;
    Ok(driver.result)
}

struct FullConversionDriver<'a> {
    root: OperationRef,
    target: &'a ConversionTarget,
    graph: LegalizationGraph<'a>,
    context: Rc<crate::Context>,
    /// The casts this conversion inserted that are still in the IR.
    casts: DriverCasts,
    config: ConversionConfig,
    result: ConversionResult,
}

impl FullConversionDriver<'_> {
    fn run(&mut self) -> Result<(), Report> {
        let ops = self.collect_conversion_order()?;
        for op in ops {
            if self.is_live(op) {
                self.legalize_operation(op)?;
            }
        }

        if self.config.reconcile_unrealized_casts() {
            let reconciled = reconcile_unrealized_casts(self.root);
            self.result.changed |= reconciled.changed;
            let remaining = reconciled.remaining;
            // The casts reconciliation erased leave the table.
            let live = remaining.iter().copied().collect::<FxHashSet<_>>();
            self.casts.borrow_mut().retain(|cast, _| live.contains(cast));
            self.result.changed |= self.realize_casts(&remaining)?;
        }
        self.verify_full_conversion_legality()?;

        if self.config.verify_after_conversion() {
            self.root.borrow().recursively_verify()?;
        }

        Ok(())
    }

    fn collect_conversion_order(&self) -> Result<Vec<OperationRef>, Report> {
        let mut ops = vec![];
        self.root
            .borrow()
            .prewalk(|op| {
                if self.target.is_recursively_legal(op) {
                    WalkResult::<Report>::Skip
                } else {
                    ops.push(op.as_operation_ref());
                    WalkResult::<Report>::Continue(())
                }
            })
            .into_result()?;
        Ok(ops)
    }

    fn legalize_operation(&mut self, op: OperationRef) -> Result<(), Report> {
        // The casts the driver inserts are not offered to patterns.
        if self.is_driver_cast(op)
            || (self.config.reconcile_unrealized_casts() && is_unrealized_conversion_cast(op))
        {
            return Ok(());
        }
        if self.target.is_recursively_legal(&op.borrow()) || self.target.is_legal(&op.borrow()) {
            return Ok(());
        }

        let op_name = op.borrow().name();
        if !self.graph.is_legalizable(&op_name) {
            return Err(self.no_legalization_path_error(op));
        }

        let candidates = self.candidate_patterns(&op_name);
        if candidates.is_empty() {
            return Err(self.no_legalization_path_error(op));
        }

        let mut match_failures = vec![];
        for pattern in candidates {
            let type_converter = pattern.type_converter().cloned().map(Rc::new);
            let mut rewriter = ConversionPatternRewriter::new_for_conversion(
                Rc::clone(&self.context),
                type_converter.clone(),
                Rc::clone(&self.casts),
            );
            rewriter.inherit_inline_call_chain(op);
            if op.parent().is_some() {
                rewriter.set_insertion_point_before(op);
            }

            let (ranges, group_sizes) =
                converted_operands_for(op, type_converter.as_deref(), &mut rewriter)?;
            let mutation_count_before_pattern = rewriter.mutation_count();
            let operands = ConvertedOperands::new(&ranges, &group_sizes);
            let applied = pattern.match_and_rewrite(op, operands, &mut rewriter);
            let mutation_count = rewriter.mutation_count();

            match applied {
                Err(err) => {
                    if mutation_count > mutation_count_before_pattern {
                        return Err(pattern_mutated_then_failed_error(
                            pattern.as_ref(),
                            op,
                            mutation_count,
                            err,
                        ));
                    }
                    rewriter.erase_unused_materializations()?;
                    return Err(err);
                }
                Ok(false) => {
                    if mutation_count > mutation_count_before_pattern {
                        return Err(pattern_mutated_without_success_error(
                            pattern.as_ref(),
                            op,
                            mutation_count,
                        ));
                    }
                    rewriter.erase_unused_materializations()?;
                    collect_match_failures(&mut match_failures, &mut rewriter);
                }
                Ok(true) => {
                    if mutation_count == mutation_count_before_pattern {
                        return Err(pattern_succeeded_without_mutation_error(pattern.as_ref(), op));
                    }
                    let mutations = rewriter.take_tracked_mutations();
                    self.note_pattern_application(pattern.as_ref())?;
                    self.validate_generated_ops(pattern.as_ref(), &mutations)?;
                    self.result.changed = true;

                    self.legalize_tracked_ops(&mutations)?;
                    if self.is_live(op) {
                        self.legalize_operation(op)?;
                    }
                    return Ok(());
                }
            }
        }

        Err(no_matching_pattern_error(op, match_failures))
    }

    fn candidate_patterns(&self, op_name: &OperationName) -> Vec<Rc<dyn ConversionPattern>> {
        self.graph
            .legalizer_patterns(op_name)
            .iter()
            .chain(self.graph.any_op_patterns().iter())
            .cloned()
            .collect()
    }

    fn note_pattern_application(&mut self, pattern: &dyn ConversionPattern) -> Result<(), Report> {
        if self.result.converted_ops >= self.config.max_iterations() {
            return Err(Report::msg(format!(
                "dialect conversion exceeded the configured rewrite limit of {} while applying \
                 pattern '{}'",
                self.config.max_iterations(),
                pattern.name()
            )));
        }

        self.result.converted_ops += 1;
        Ok(())
    }

    fn validate_generated_ops(
        &self,
        pattern: &dyn ConversionPattern,
        mutations: &TrackedMutations,
    ) -> Result<(), Report> {
        for op in mutations.inserted_ops() {
            if mutations.materialized_ops().iter().any(|materialized| materialized == op) {
                continue;
            }
            let name = op.borrow().name();
            if !pattern.generated_ops().iter().any(|generated| generated == &name) {
                return Err(Report::msg(format!(
                    "conversion pattern '{}' generated undeclared operation '{}'; add it to the \
                     pattern's generated operation metadata",
                    pattern.name(),
                    name
                )));
            }
        }
        Ok(())
    }

    fn legalize_tracked_ops(&mut self, mutations: &TrackedMutations) -> Result<(), Report> {
        for op in mutations.inserted_ops().iter().chain(mutations.modified_ops().iter()).copied() {
            if self.is_live(op) {
                self.legalize_operation(op)?;
            }
        }
        Ok(())
    }

    /// Realizes each of `casts` (in program order) that the driver inserted, through the
    /// materialization callbacks of its kind of the converter in force when it was inserted;
    /// returns whether any was realized. A cast the driver did not insert is left alone; one that
    /// no callback realizes is an error.
    fn realize_casts(&mut self, casts: &[OperationRef]) -> Result<bool, Report> {
        let mut changed = false;
        for &cast in casts {
            let Some(driver_cast) = self.casts.borrow_mut().remove(&cast) else {
                continue;
            };
            if !self.is_live(cast) {
                continue;
            }
            let (inputs, result_types, span) = {
                let op = cast.borrow();
                let inputs = op
                    .operands()
                    .iter()
                    .map(|operand| operand.borrow().as_value_ref())
                    .collect::<SmallVec<[ValueRef; 2]>>();
                let results = op
                    .results()
                    .iter()
                    .map(|result| result.borrow().as_value_ref())
                    .collect::<SmallVec<[ValueRef; 2]>>();
                (inputs, types_of(&results), op.span())
            };

            let converter = driver_cast.converter;
            let mut rewriter = ConversionPatternRewriter::new_for_conversion(
                Rc::clone(&self.context),
                Some(Rc::clone(&converter)),
                DriverCasts::default(),
            );
            rewriter.inherit_inline_call_chain(cast);
            rewriter.set_insertion_point_before(cast);
            let values = match driver_cast.kind {
                CastKind::Source => converter
                    .materialize_source(&mut rewriter, &inputs, result_types[0], span)?
                    .map(|value| smallvec::smallvec![value]),
                CastKind::Target => {
                    converter.materialize_target(&mut rewriter, inputs[0], &result_types, span)?
                }
            };
            let Some(values) = values else {
                return Err(unrealized_cast_error(cast, driver_cast.kind, &inputs, &result_types));
            };
            if types_of(&values) != result_types {
                return Err(Report::msg(format!(
                    "failed to legalize operation '{}': the {} materialization of the cast of {} \
                     to {} gave values of types {}",
                    cast.borrow().name(),
                    kind_name(driver_cast.kind),
                    type_list(&types_of(&inputs)),
                    type_list(&result_types),
                    type_list(&types_of(&values))
                )));
            }
            rewriter.replace_cast(cast, &values);
            changed = true;
        }
        Ok(changed)
    }

    /// Whether `op` is a cast the driver inserted.
    fn is_driver_cast(&self, op: OperationRef) -> bool {
        self.casts.borrow().contains_key(&op)
    }

    fn verify_full_conversion_legality(&self) -> Result<(), Report> {
        self.root
            .borrow()
            .prewalk(|op| {
                if self.target.is_recursively_legal(op) {
                    return WalkResult::<Report>::Skip;
                }

                // A cast the driver inserted that remains is one reconciliation left alone (off).
                if self.target.is_legal(op) || self.is_driver_cast(op.as_operation_ref()) {
                    WalkResult::<Report>::Continue(())
                } else {
                    let op_ref = op.as_operation_ref();
                    WalkResult::Break(self.no_legalization_path_error(op_ref))
                }
            })
            .into_result()
    }

    fn no_legalization_path_error(&self, op: OperationRef) -> Report {
        let op = op.borrow();
        let reason = legality_failure_reason(self.target, &op);
        if let Some(reason) = reason {
            Report::msg(format!(
                "failed to legalize operation '{}': no legalization path to target; {reason}",
                op.name()
            ))
        } else {
            Report::msg(format!(
                "failed to legalize operation '{}': no legalization path to target",
                op.name()
            ))
        }
    }

    fn is_live(&self, op: OperationRef) -> bool {
        op == self.root || op.parent().is_some()
    }
}

/// One range of values per operand, and the sizes of the operand groups.
type OperandRanges = (Vec<SmallVec<[ValueRef; 2]>>, SmallVec<[usize; 4]>);

/// The values each operand of `op` became, one range per operand, and the sizes of `op`'s operand
/// groups.
///
/// Without a type converter an operand is itself. With one, an operand whose type converts to
/// exactly itself is itself; otherwise, when it is the one result of a cast whose inputs have
/// exactly the converted types, it is that cast's inputs; otherwise it is the results of a new
/// one-to-many cast to the converted types, inserted at the rewriter's insertion point (none when
/// the operand converts to no type: its range is empty).
fn converted_operands_for(
    op: OperationRef,
    type_converter: Option<&TypeConverter>,
    rewriter: &mut ConversionPatternRewriter,
) -> Result<OperandRanges, Report> {
    let (context, span, operands, group_sizes) = {
        let op = op.borrow();
        let operands = op
            .operands()
            .iter()
            .map(|operand| operand.borrow().as_value_ref())
            .collect::<Vec<_>>();
        let group_sizes = op.operands().groups().map(|group| group.len()).collect();
        (op.context_rc(), op.span(), operands, group_sizes)
    };

    let mut ranges = Vec::with_capacity(operands.len());
    for value in operands {
        let Some(type_converter) = type_converter else {
            ranges.push(smallvec::smallvec![value]);
            continue;
        };
        let types = type_converter.convert_value(&context, value);
        if types.as_slice() == [value.borrow().ty()] {
            ranges.push(smallvec::smallvec![value]);
        } else if let Some(inputs) = inputs_of_cast_from(value, &types) {
            ranges.push(inputs);
        } else if types.is_empty() {
            ranges.push(SmallVec::new());
        } else {
            ranges.push(rewriter.insert_cast(CastKind::Target, [value], types, span)?);
        }
    }
    Ok((ranges, group_sizes))
}

/// The inputs of the cast `value` is the one result of, when their types are exactly `types`.
fn inputs_of_cast_from(value: ValueRef, types: &[crate::Type]) -> Option<SmallVec<[ValueRef; 2]>> {
    let defining = value.borrow().get_defining_op()?;
    let cast = defining.try_downcast_op::<UnrealizedConversionCast>().ok()?;
    let cast = cast.borrow();
    if cast.outputs().len() != 1 {
        return None;
    }
    let inputs = cast
        .inputs()
        .iter()
        .map(|input| input.borrow().as_value_ref())
        .collect::<SmallVec<[ValueRef; 2]>>();
    (super::rewriter::types_of(&inputs).as_slice() == types).then_some(inputs)
}

fn collect_match_failures(failures: &mut Vec<String>, rewriter: &mut ConversionPatternRewriter) {
    failures.extend(
        rewriter
            .take_match_failures()
            .into_iter()
            .map(|failure| format!("{}: {}", failure.op().borrow().name(), failure.reason())),
    );
}

fn is_unrealized_conversion_cast(op: OperationRef) -> bool {
    op.try_downcast_op::<UnrealizedConversionCast>().is_ok()
}

fn kind_name(kind: CastKind) -> &'static str {
    match kind {
        CastKind::Source => "source",
        CastKind::Target => "target",
    }
}

/// A cast the driver inserted that no materialization callback realized: names its types and
/// one of its users.
fn unrealized_cast_error(
    cast: OperationRef,
    kind: CastKind,
    inputs: &[ValueRef],
    result_types: &[crate::Type],
) -> Report {
    let user = cast
        .borrow()
        .results()
        .iter()
        .find_map(|result| result.borrow().iter_uses().next().map(|user| user.owner))
        .map(|user| format!("'{}'", user.borrow().name()))
        .unwrap_or_else(|| String::from("nothing"));
    Report::msg(format!(
        "failed to legalize operation '{}': no {} materialization of the type converter realizes \
         the cast of {} to {}, which is used by {user}",
        cast.borrow().name(),
        kind_name(kind),
        type_list(&types_of(inputs)),
        type_list(result_types)
    ))
}

fn legality_failure_reason(target: &ConversionTarget, op: &crate::Operation) -> Option<String> {
    match target.legality(op) {
        Legality::DynamicIllegal {
            reason: Some(reason),
        } => Some(format!("{reason}")),
        Legality::DynamicIllegal { reason: None } => {
            Some(String::from("dynamic legality predicate returned illegal"))
        }
        Legality::Illegal => Some(String::from("operation is explicitly illegal")),
        Legality::Unknown => Some(String::from("operation is unknown to the conversion target")),
        Legality::Legal | Legality::DynamicLegal => None,
    }
}

fn pattern_mutated_then_failed_error(
    pattern: &dyn ConversionPattern,
    op: OperationRef,
    mutation_count: usize,
    err: Report,
) -> Report {
    Report::msg(format!(
        "conversion pattern '{}' failed after mutating operation '{}' ({} mutations observed): {}",
        pattern.name(),
        op.borrow().name(),
        mutation_count,
        err
    ))
}

fn pattern_mutated_without_success_error(
    pattern: &dyn ConversionPattern,
    op: OperationRef,
    mutation_count: usize,
) -> Report {
    Report::msg(format!(
        "conversion pattern '{}' mutated operation '{}' but returned no match ({} mutations \
         observed)",
        pattern.name(),
        op.borrow().name(),
        mutation_count
    ))
}

fn pattern_succeeded_without_mutation_error(
    pattern: &dyn ConversionPattern,
    op: OperationRef,
) -> Report {
    Report::msg(format!(
        "conversion pattern '{}' reported success for operation '{}' without mutating IR",
        pattern.name(),
        op.borrow().name()
    ))
}

fn no_matching_pattern_error(op: OperationRef, failures: Vec<String>) -> Report {
    if failures.is_empty() {
        return Report::msg(format!(
            "failed to legalize operation '{}': no conversion pattern matched",
            op.borrow().name()
        ));
    }

    Report::msg(format!(
        "failed to legalize operation '{}': no conversion pattern matched ({})",
        op.borrow().name(),
        failures.join("; ")
    ))
}

#[cfg(test)]
mod range_tests;

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, string::ToString};

    use pretty_assertions::assert_str_eq;

    use super::*;
    use crate::{
        Context, Int, Op, OperationRef, Overflow, Report, Signedness, SourceSpan, Spanned, Type,
        ValueRef,
        conversion::{
            ConvertedOperands, DynamicLegalityResult, TypeConversion, TypeConverter,
            UnknownOpPolicy,
        },
        dialects::{
            builtin::{BuiltinOpBuilder, UnrealizedConversionCast},
            test::{Add, Constant, Mul, Shl, TestDialect, TestOpBuilder},
        },
        patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind},
        testing::Test,
    };

    struct AddToMul {
        info: PatternInfo,
    }

    impl AddToMul {
        fn new(context: Rc<Context>) -> Self {
            let dialect = context.get_or_register_dialect::<TestDialect>();
            let mut info = PatternInfo::new(
                context,
                "add-to-mul",
                PatternKind::Operation(dialect.expect_registered_name::<Add>()),
                PatternBenefit::new(1),
            );
            info.with_generated_ops([dialect.expect_registered_name::<Mul>()]);
            Self { info }
        }
    }

    impl Pattern for AddToMul {
        fn info(&self) -> &PatternInfo {
            &self.info
        }
    }

    impl ConversionPattern for AddToMul {
        fn match_and_rewrite(
            &self,
            op: OperationRef,
            operands: ConvertedOperands<'_>,
            rewriter: &mut ConversionPatternRewriter,
        ) -> Result<bool, Report> {
            let Some((span, lhs, rhs)) = add_parts(op, operands) else {
                rewriter.notify_match_failure(op, Report::msg("expected test.add"));
                return Ok(false);
            };

            rewriter.replace_op_with_new_op::<Mul, _>(
                op,
                span,
                (lhs, rhs, Overflow::Checked(Signedness::Unsigned)),
            )?;
            Ok(true)
        }
    }

    struct AddToMulWithTypeConversion {
        info: PatternInfo,
        type_converter: TypeConverter,
    }

    impl AddToMulWithTypeConversion {
        fn new(context: Rc<Context>, type_converter: TypeConverter) -> Self {
            let dialect = context.get_or_register_dialect::<TestDialect>();
            let mut info = PatternInfo::new(
                context,
                "add-to-mul-with-type-conversion",
                PatternKind::Operation(dialect.expect_registered_name::<Add>()),
                PatternBenefit::new(1),
            );
            info.with_generated_ops([dialect.expect_registered_name::<Mul>()]);
            Self {
                info,
                type_converter,
            }
        }
    }

    impl Pattern for AddToMulWithTypeConversion {
        fn info(&self) -> &PatternInfo {
            &self.info
        }
    }

    impl ConversionPattern for AddToMulWithTypeConversion {
        fn type_converter(&self) -> Option<&TypeConverter> {
            Some(&self.type_converter)
        }

        fn match_and_rewrite(
            &self,
            op: OperationRef,
            operands: ConvertedOperands<'_>,
            rewriter: &mut ConversionPatternRewriter,
        ) -> Result<bool, Report> {
            let Some((span, lhs, rhs)) = add_parts(op, operands) else {
                return Ok(false);
            };

            rewriter.replace_op_with_new_op::<Mul, _>(
                op,
                span,
                (lhs, rhs, Overflow::Checked(Signedness::Unsigned)),
            )?;
            Ok(true)
        }
    }

    struct MulToShl {
        info: PatternInfo,
    }

    impl MulToShl {
        fn new(context: Rc<Context>) -> Self {
            let dialect = context.get_or_register_dialect::<TestDialect>();
            let mut info = PatternInfo::new(
                context,
                "mul-to-shl",
                PatternKind::Operation(dialect.expect_registered_name::<Mul>()),
                PatternBenefit::new(1),
            );
            info.with_generated_ops([dialect.expect_registered_name::<Shl>()]);
            Self { info }
        }
    }

    impl Pattern for MulToShl {
        fn info(&self) -> &PatternInfo {
            &self.info
        }
    }

    impl ConversionPattern for MulToShl {
        fn match_and_rewrite(
            &self,
            op: OperationRef,
            operands: ConvertedOperands<'_>,
            rewriter: &mut ConversionPatternRewriter,
        ) -> Result<bool, Report> {
            if op.try_downcast_op::<Mul>().is_err() {
                rewriter.notify_match_failure(op, Report::msg("expected test.mul"));
                return Ok(false);
            }

            let Some([lhs, rhs]) = operands.all_one().map(|values| [values[0], values[1]]) else {
                return Ok(false);
            };
            let span = op.borrow().span();

            rewriter.replace_op_with_new_op::<Shl, _>(op, span, (lhs, rhs))?;
            Ok(true)
        }
    }

    struct AddToMulByTwo {
        info: PatternInfo,
    }

    impl AddToMulByTwo {
        fn new(context: Rc<Context>) -> Self {
            let dialect = context.get_or_register_dialect::<TestDialect>();
            let mut info = PatternInfo::new(
                context,
                "add-to-mul-by-two",
                PatternKind::Operation(dialect.expect_registered_name::<Add>()),
                PatternBenefit::new(1),
            );
            info.with_generated_ops([
                dialect.expect_registered_name::<Constant>(),
                dialect.expect_registered_name::<Mul>(),
            ]);
            Self { info }
        }
    }

    impl Pattern for AddToMulByTwo {
        fn info(&self) -> &PatternInfo {
            &self.info
        }
    }

    impl ConversionPattern for AddToMulByTwo {
        fn match_and_rewrite(
            &self,
            op: OperationRef,
            operands: ConvertedOperands<'_>,
            rewriter: &mut ConversionPatternRewriter,
        ) -> Result<bool, Report> {
            if op.try_downcast_op::<Add>().is_err() {
                return Ok(false);
            }

            let span = op.borrow().span();
            let Some(lhs) = operands.one(0) else {
                return Ok(false);
            };
            let constant = rewriter.create_op::<Constant, _>(span, (Int::from(2i32),))?;
            let rhs = constant.borrow().result().as_value_ref();
            let mul = rewriter
                .create_op::<Mul, _>(span, (lhs, rhs, Overflow::Checked(Signedness::Unsigned)))?;
            let result = mul.borrow().result().as_value_ref();
            rewriter.replace_op(op, &[result])?;
            Ok(true)
        }
    }

    struct UndeclaredAddToMul {
        info: PatternInfo,
    }

    impl UndeclaredAddToMul {
        fn new(context: Rc<Context>) -> Self {
            let dialect = context.get_or_register_dialect::<TestDialect>();
            Self {
                info: PatternInfo::new(
                    context,
                    "undeclared-add-to-mul",
                    PatternKind::Operation(dialect.expect_registered_name::<Add>()),
                    PatternBenefit::new(1),
                ),
            }
        }
    }

    impl Pattern for UndeclaredAddToMul {
        fn info(&self) -> &PatternInfo {
            &self.info
        }
    }

    impl ConversionPattern for UndeclaredAddToMul {
        fn match_and_rewrite(
            &self,
            op: OperationRef,
            operands: ConvertedOperands<'_>,
            rewriter: &mut ConversionPatternRewriter,
        ) -> Result<bool, Report> {
            let Some((span, lhs, rhs)) = add_parts(op, operands) else {
                return Ok(false);
            };
            rewriter.replace_op_with_new_op::<Mul, _>(
                op,
                span,
                (lhs, rhs, Overflow::Checked(Signedness::Unsigned)),
            )?;
            Ok(true)
        }
    }

    struct MutatesButFails {
        info: PatternInfo,
    }

    impl MutatesButFails {
        fn new(context: Rc<Context>) -> Self {
            let dialect = context.get_or_register_dialect::<TestDialect>();
            let mut info = PatternInfo::new(
                context,
                "mutates-but-fails",
                PatternKind::Operation(dialect.expect_registered_name::<Add>()),
                PatternBenefit::new(1),
            );
            info.with_generated_ops([dialect.expect_registered_name::<Constant>()]);
            Self { info }
        }
    }

    impl Pattern for MutatesButFails {
        fn info(&self) -> &PatternInfo {
            &self.info
        }
    }

    impl ConversionPattern for MutatesButFails {
        fn match_and_rewrite(
            &self,
            op: OperationRef,
            _operands: ConvertedOperands<'_>,
            rewriter: &mut ConversionPatternRewriter,
        ) -> Result<bool, Report> {
            let span = op.borrow().span();
            let _ = rewriter.create_op::<Constant, _>(span, (Int::from(1i32),))?;
            Ok(false)
        }
    }

    fn add_parts(
        op: OperationRef,
        operands: ConvertedOperands<'_>,
    ) -> Option<(SourceSpan, ValueRef, ValueRef)> {
        op.try_downcast_op::<Add>().ok()?;
        let values = operands.all_one()?;
        Some((op.borrow().span(), values[0], values[1]))
    }

    fn add_function(name: &'static str) -> Test {
        let mut test = Test::named(name);
        let i32 = test.context().i32();
        test.with_function(name, &[i32, i32], &[i32]);
        {
            let mut builder = test.function_builder();
            let block = builder.current_block();
            let lhs = block.borrow().arguments()[0] as ValueRef;
            let rhs = block.borrow().arguments()[1] as ValueRef;
            let result = builder.add(lhs, rhs, SourceSpan::default()).unwrap();
            builder.ret(Some(result), SourceSpan::default()).unwrap();
        }
        test
    }

    fn target_with_test_ops(
        context: Rc<Context>,
        legal_ops: impl FnOnce(&mut ConversionTarget),
    ) -> ConversionTarget {
        let mut target = ConversionTarget::new(context);
        target.set_unknown_op_policy(UnknownOpPolicy::Legal);
        target.add_illegal_op::<Add>();
        legal_ops(&mut target);
        target
    }

    fn i32_to_i64_converter(context: &Context) -> TypeConverter {
        let (from, to) = (context.i32(), context.i64());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| {
            if ty == from {
                Some(TypeConversion::One(to))
            } else {
                None
            }
        });
        converter
    }

    /// Builds a function taking an `i32` that casts it to `middle`, then to `result`, and returns it.
    fn cast_chain_function(
        name: &'static str,
        middle: fn(&Context) -> Type,
        result: fn(&Context) -> Type,
    ) -> Test {
        let mut test = Test::named(name);
        let i32 = test.context().i32();
        let middle = middle(test.context());
        let result = result(test.context());
        test.with_function(name, &[i32], &[result]);
        {
            let mut builder = test.function_builder();
            let block = builder.current_block();
            let arg = block.borrow().arguments()[0] as ValueRef;
            let first = builder
                .builder_mut()
                .unrealized_conversion_cast([arg], [middle], SourceSpan::default())
                .unwrap();
            let first = first.borrow().outputs()[0] as ValueRef;
            let second = builder
                .builder_mut()
                .unrealized_conversion_cast([first], [result], SourceSpan::default())
                .unwrap();
            let second = second.borrow().outputs()[0] as ValueRef;
            builder.ret(Some(second), SourceSpan::default()).unwrap();
        }
        test
    }

    #[test]
    fn full_conversion_rewrites_illegal_op_to_legal_op() {
        let test = add_function("full_conversion_rewrites_illegal_op_to_legal_op");
        let target = target_with_test_ops(test.context_rc(), |target| {
            target.add_legal_op::<Mul>();
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(AddToMul::new(test.context_rc()));

        let result = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap();

        assert!(result.changed());
        assert_eq!(result.converted_ops(), 1);
        let output = test.function().borrow().as_operation().to_string();
        let expected = "\
builtin.function public extern(\"C\") @full_conversion_rewrites_illegal_op_to_legal_op(%0: i32, \
                        %1: i32) -> i32 {
    %3 = test.mul %0, %1 <{ overflow = #builtin.overflow<checked unsigned> }>;
    builtin.ret %3 : (i32);
};";
        assert_str_eq!(output.as_str(), expected);
    }

    #[test]
    fn full_conversion_reports_missing_legalization_path() {
        let test = add_function("full_conversion_reports_missing_legalization_path");
        let target = target_with_test_ops(test.context_rc(), |_| {});
        let patterns = ConversionPatternSet::new(test.context_rc());

        let err = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap_err();

        assert!(format!("{err}").contains("no legalization path"));
    }

    #[test]
    fn full_conversion_legalizes_transitive_rewrites() {
        let test = add_function("full_conversion_legalizes_transitive_rewrites");
        let target = target_with_test_ops(test.context_rc(), |target| {
            target.add_illegal_op::<Mul>().add_legal_op::<Shl>();
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(AddToMul::new(test.context_rc()));
        patterns.push(MulToShl::new(test.context_rc()));

        let result = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap();

        assert_eq!(result.converted_ops(), 2);
        let output = test.function().borrow().as_operation().to_string();
        assert!(output.contains("test.shl"));
        assert!(!output.contains("test.add"));
        assert!(!output.contains("test.mul"));
    }

    #[test]
    fn full_conversion_rewrites_dynamically_illegal_ops() {
        let test = add_function("full_conversion_rewrites_dynamically_illegal_ops");
        let target = target_with_test_ops(test.context_rc(), |target| {
            target
                .add_dynamically_legal_op::<Add, _>(|_| DynamicLegalityResult::illegal())
                .add_legal_op::<Mul>();
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(AddToMul::new(test.context_rc()));

        let result = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap();

        assert_eq!(result.converted_ops(), 1);
        let output = test.function().borrow().as_operation().to_string();
        assert!(output.contains("test.mul"));
        assert!(!output.contains("test.add"));
    }

    /// Without reconciliation, the casts the driver inserted for a 1:1 conversion stay in the IR
    /// and are not an error, even in a target that calls them illegal; they inherit the
    /// converted operation's inline call chain.
    #[test]
    fn without_reconciliation_the_1_to_1_casts_stay() {
        let test = add_function("without_reconciliation_the_1_to_1_casts_stay");
        use crate::dialects::debuginfo::attributes::{
            INLINE_CALL_CHAIN_ATTR_NAME, InlineCallChain, InlineCallChainAttr,
        };
        let chain = test
            .context_rc()
            .create_attribute::<InlineCallChainAttr, _>(InlineCallChain::default())
            .as_attribute_ref();
        let mut add = None;
        test.function().borrow().as_operation().prewalk_all(|op| {
            if op.is::<Add>() {
                add = Some(op.as_operation_ref());
            }
        });
        add.unwrap().borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, chain);
        let target = target_with_test_ops(test.context_rc(), |target| {
            target.add_legal_op::<Mul>().add_illegal_op::<UnrealizedConversionCast>();
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(AddToMulWithTypeConversion::new(
            test.context_rc(),
            i32_to_i64_converter(test.context()),
        ));
        let mut config = ConversionConfig::default();
        config.with_reconcile_unrealized_casts(false);

        let result =
            apply_full_conversion(test.function().as_operation_ref(), target, patterns, config)
                .unwrap();

        assert_eq!(result.converted_ops(), 1);
        let mut inherited = 0;
        test.function().borrow().as_operation().prewalk_all(|op| {
            if op.is::<Mul>() || op.is::<UnrealizedConversionCast>() {
                assert_eq!(op.get_attribute(INLINE_CALL_CHAIN_ATTR_NAME), Some(chain));
                inherited += 1;
            } else if op.is::<Constant>() {
                assert!(!op.has_attribute(INLINE_CALL_CHAIN_ATTR_NAME));
            }
        });
        assert_eq!(inherited, 4, "replacement plus operand and result materializations");
        let output = test.function().borrow().as_operation().to_string();
        assert!(output.contains("builtin.unrealized_conversion_cast"));
        assert!(output.contains(": i32 to i64 "), "{output}");
        assert!(output.contains(": i64 to i32 "), "{output}");
        assert!(output.contains("test.mul"));
        assert!(!output.contains("test.add"));
    }

    /// With reconciliation, a cast the driver inserted that no materialization callback realizes
    /// is an error, even in a target that calls the cast legal; the error names the cast's types
    /// and a user.
    #[test]
    fn a_cast_nobody_realizes_is_an_error() {
        let test = add_function("a_cast_nobody_realizes_is_an_error");
        let target = target_with_test_ops(test.context_rc(), |target| {
            target.add_legal_op::<Mul>().add_legal_op::<UnrealizedConversionCast>();
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(AddToMulWithTypeConversion::new(
            test.context_rc(),
            i32_to_i64_converter(test.context()),
        ));

        let err = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap_err();

        assert_eq!(
            format!("{err}"),
            "failed to legalize operation 'builtin.unrealized_conversion_cast': no target \
             materialization of the type converter realizes the cast of (i32) to (i64), which is \
             used by 'test.mul'"
        );
    }

    /// A pattern written for one value per operand does not match an operand that became two
    /// values: the conversion fails because no pattern matched, and the casts made for the
    /// operands are erased again.
    #[test]
    fn a_pattern_for_one_value_per_operand_does_not_match_a_split_operand() {
        let test =
            add_function("a_pattern_for_one_value_per_operand_does_not_match_a_split_operand");
        let before = test.function().borrow().as_operation().to_string();
        let target = target_with_test_ops(test.context_rc(), |target| {
            target.add_legal_op::<Mul>();
        });
        let (i32, i64) = (test.context().i32(), test.context().i64());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| {
            if ty == i32 {
                Some(TypeConversion::Many(smallvec::smallvec![i64, i64]))
            } else {
                None
            }
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(AddToMulWithTypeConversion::new(test.context_rc(), converter));

        let err = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap_err();

        assert!(format!("{err}").contains("no conversion pattern matched"), "{err}");
        assert_str_eq!(test.function().borrow().as_operation().to_string(), before);
    }

    #[test]
    fn full_conversion_reconciles_identity_unrealized_casts() {
        let test = cast_chain_function(
            "full_conversion_reconciles_identity_unrealized_casts",
            Context::i32,
            Context::i32,
        );
        let mut target = ConversionTarget::new(test.context_rc());
        target
            .set_unknown_op_policy(UnknownOpPolicy::Legal)
            .add_illegal_op::<UnrealizedConversionCast>();

        let result = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            ConversionPatternSet::new(test.context_rc()),
            ConversionConfig::default(),
        )
        .unwrap();

        assert!(result.changed());
        let output = test.function().borrow().as_operation().to_string();
        assert!(!output.contains("builtin.unrealized_conversion_cast"));
    }

    #[test]
    fn full_conversion_reconciles_reversible_unrealized_cast_chains() {
        let test = cast_chain_function(
            "full_conversion_reconciles_reversible_unrealized_cast_chains",
            Context::i64,
            Context::i32,
        );
        let mut target = ConversionTarget::new(test.context_rc());
        target
            .set_unknown_op_policy(UnknownOpPolicy::Legal)
            .add_illegal_op::<UnrealizedConversionCast>();

        let result = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            ConversionPatternSet::new(test.context_rc()),
            ConversionConfig::default(),
        )
        .unwrap();

        assert!(result.changed());
        let output = test.function().borrow().as_operation().to_string();
        assert!(!output.contains("builtin.unrealized_conversion_cast"));
    }

    #[test]
    fn full_conversion_supports_op_to_sequence_rewrites() {
        let test = add_function("full_conversion_supports_op_to_sequence_rewrites");
        let target = target_with_test_ops(test.context_rc(), |target| {
            target.add_legal_op::<Constant>().add_legal_op::<Mul>();
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(AddToMulByTwo::new(test.context_rc()));

        let result = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap();

        assert_eq!(result.converted_ops(), 1);
        let output = test.function().borrow().as_operation().to_string();
        assert!(output.contains("test.constant 2"));
        assert!(output.contains("test.mul"));
        assert!(!output.contains("test.add"));
    }

    #[test]
    fn full_conversion_rejects_undeclared_generated_ops() {
        let test = add_function("full_conversion_rejects_undeclared_generated_ops");
        let target = target_with_test_ops(test.context_rc(), |target| {
            target.add_legal_op::<Mul>();
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(UndeclaredAddToMul::new(test.context_rc()));

        let err = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap_err();

        assert!(format!("{err}").contains("generated undeclared operation"));
    }

    #[test]
    fn full_conversion_rejects_mutation_contract_violations() {
        let test = add_function("full_conversion_rejects_mutation_contract_violations");
        let target = target_with_test_ops(test.context_rc(), |target| {
            target.add_legal_op::<Constant>();
        });
        let mut patterns = ConversionPatternSet::new(test.context_rc());
        patterns.push(MutatesButFails::new(test.context_rc()));

        let err = apply_full_conversion(
            test.function().as_operation_ref(),
            target,
            patterns,
            ConversionConfig::default(),
        )
        .unwrap_err();

        assert!(format!("{err}").contains("mutated operation"));
    }
}
