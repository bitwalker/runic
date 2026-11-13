mod lattice;

use runic_core::{
    AttributeRef, Forward, Operation, ProgramPoint, Report, SmallVec, ValueRef,
    attributes::IntegerLikeAttr,
    traits::{InferIntRangeInterface, operation_operand_range_requirement},
};
pub use runic_core::{ConstantIntRange, IntegerRangeConfidence, IntegerValueRange};

pub use self::lattice::IntegerRangeLattice;
use super::{DeadCodeAnalysis, SparseConstantPropagation, constant_propagation::ConstantValue};
use crate::{
    AnalysisState, AnalysisStateGuard, AnalysisStateGuardMut, BuildableDataFlowAnalysis,
    DataFlowSolver, Lattice, LatticeLike, SparseForwardDataFlowAnalysis, SparseLattice,
    sparse::SparseDataFlowAnalysis,
};

impl LatticeLike for IntegerValueRange {
    fn join(&self, other: &Self) -> Self {
        match (self.range().copied(), other.range().copied()) {
            (None, None) | (Some(_), None) => self.clone(),
            (None, Some(_)) => other.clone(),
            (Some(lhs), Some(rhs)) => Self::with_confidence(
                lhs.union(&rhs),
                self.join_confidence(other).expect("initialized ranges have confidence"),
            ),
        }
    }

    fn meet(&self, _other: &Self) -> Self {
        // This analysis uses a join semilattice. Trust metadata and pending facts do not
        // define an intersection operation; LatticeLike requires unsupported operations to panic.
        unimplemented!("IntegerValueRange is a join semilattice")
    }
}

/// Sparse forward integer range analysis.
#[derive(Default)]
pub struct IntegerRangeAnalysis;

impl BuildableDataFlowAnalysis for IntegerRangeAnalysis {
    type Strategy = SparseDataFlowAnalysis<Self, Forward>;

    fn new(solver: &mut DataFlowSolver) -> Self {
        solver.load::<DeadCodeAnalysis>();
        solver.load::<SparseConstantPropagation>();
        Self
    }
}

impl SparseForwardDataFlowAnalysis for IntegerRangeAnalysis {
    type Lattice = IntegerRangeLattice;

    fn debug_name(&self) -> &'static str {
        "integer-range-analysis"
    }

    fn visit_operation(
        &self,
        op: &Operation,
        operands: &[AnalysisStateGuard<'_, Self::Lattice>],
        results: &mut [AnalysisStateGuardMut<'_, Self::Lattice>],
        solver: &mut DataFlowSolver,
    ) -> Result<(), Report> {
        log::debug!(target: self.debug_name(), "visiting operation {op}");

        if let Some(inferrable) = op.as_trait::<dyn InferIntRangeInterface>() {
            return self
                .visit_range_inferrable_operation(op, inferrable, operands, results, solver);
        }

        for (index, result_lattice) in results.iter_mut().enumerate() {
            let result = op.results()[index].borrow().as_value_ref();

            match constant_range_from_solver(result, op, solver) {
                ConstantRange::Pending => continue,
                ConstantRange::Known(range) => {
                    result_lattice.join(&range);
                    continue;
                }
                ConstantRange::NotConstant => {}
            }

            self.set_to_entry_state(result_lattice);
        }

        Ok(())
    }

    fn set_to_entry_state(&self, lattice: &mut AnalysisStateGuardMut<'_, Self::Lattice>) {
        let Some(value) = lattice.anchor().as_value() else {
            return;
        };
        let ty = value.borrow().ty();
        let entry_state = IntegerValueRange::max_for_type(ty);
        log::trace!(
            target: self.debug_name(),
            "setting integer range for {} to entry state {entry_state}",
            lattice.anchor()
        );
        lattice.join(&entry_state);
    }
}

impl IntegerRangeAnalysis {
    fn visit_range_inferrable_operation(
        &self,
        op: &Operation,
        inferrable: &dyn InferIntRangeInterface,
        operands: &[AnalysisStateGuard<'_, <Self as SparseForwardDataFlowAnalysis>::Lattice>],
        results: &mut [AnalysisStateGuardMut<
            '_,
            <Self as SparseForwardDataFlowAnalysis>::Lattice,
        >],
        solver: &mut DataFlowSolver,
    ) -> Result<(), Report> {
        let operand_ranges = operands
            .iter()
            .map(|operand| operand.value().clone())
            .collect::<SmallVec<[_; 4]>>();
        let has_uninitialized_operand =
            operand_ranges.iter().any(IntegerValueRange::is_uninitialized);
        let mut inferred = SmallVec::<[bool; 4]>::with_capacity(results.len());
        inferred.resize(results.len(), false);
        let mut join_result_range = |value: ValueRef, range: IntegerValueRange| {
            let Some(index) = op
                .results()
                .all()
                .iter()
                .position(|result| result.borrow().as_value_ref() == value)
            else {
                return;
            };
            inferred[index] = true;
            results[index].join(&range);
        };

        inferrable.infer_result_ranges(&operand_ranges, &mut join_result_range);

        for (index, result_lattice) in results.iter_mut().enumerate() {
            if inferred[index] {
                continue;
            }

            let result = op.results()[index].borrow().as_value_ref();
            match constant_range_from_solver(result, op, solver) {
                ConstantRange::Known(range) => {
                    result_lattice.join(&range);
                }
                ConstantRange::NotConstant if !has_uninitialized_operand => {
                    self.set_to_entry_state(result_lattice);
                }
                ConstantRange::Pending | ConstantRange::NotConstant => {}
            }
        }

        Ok(())
    }
}

/// Return the currently known integer range for `value`, if available.
pub fn value_range(solver: &DataFlowSolver, value: ValueRef) -> Option<IntegerValueRange> {
    solver
        .get::<IntegerRangeLattice, _>(&value)
        .map(|range| range.value().clone())
        .filter(|range| !range.is_uninitialized())
}

/// Return true if the range analysis proves every requirement `op` states of an operand
/// ([runic_core::traits::OperandRangeRequirementOpInterface]): for each operand with a
/// requirement, `solver` has an initialized range for it that [ConstantIntRange::is_within] the
/// requirement. An operation that states no requirement trivially has them all proven.
///
/// A requirement on an operand whose ranges are not tracked (`isize`) is the untracked range, which
/// no range is within, so this is false for such an operation. A range of either confidence is
/// accepted: an assumed range is still a fact about a signless integer.
pub fn operand_range_requirements_hold(solver: &DataFlowSolver, op: &Operation) -> bool {
    (0..op.num_operands()).all(|index| {
        let Some(requirement) = operation_operand_range_requirement(op, index) else {
            return true;
        };
        let operand = op.operands()[index].borrow().as_value_ref();
        value_range(solver, operand)
            .and_then(|range| range.range().copied())
            .is_some_and(|range| range.is_within(&requirement))
    })
}

enum ConstantRange {
    Pending,
    Known(IntegerValueRange),
    NotConstant,
}

fn constant_range_from_solver(
    result: ValueRef,
    op: &Operation,
    solver: &mut DataFlowSolver,
) -> ConstantRange {
    let constant = solver.require::<Lattice<ConstantValue>, _>(result, ProgramPoint::after(op));
    if constant.value().is_uninitialized() {
        return ConstantRange::Pending;
    }

    match constant.value().constant_value().and_then(integer_constant_range) {
        Some(range) => ConstantRange::Known(range),
        None => ConstantRange::NotConstant,
    }
}

/// The range of the integer constant `attr`, which knows its type: an `isize` constant has no
/// exact range (see `ConstantIntRange::for_constant`). `None` if `attr` is not an integer.
fn integer_constant_range(attr: AttributeRef) -> Option<IntegerValueRange> {
    let attr = attr.borrow();
    let value = attr.as_attr().as_trait::<dyn IntegerLikeAttr>()?.as_int();
    Some(IntegerValueRange::for_constant(value, attr.ty()))
}

#[cfg(test)]
mod tests {
    use runic_core::{
        Context, Int, Op, OpBuilder, Signedness, SourceSpan, Type,
        dialects::{
            builtin::{BuiltinOpBuilder, FunctionBuilder},
            test::TestOpBuilder,
        },
        pass::AnalysisManager,
        testing::Test,
    };
    use runic_dialect_arith::ArithOpBuilder;
    use runic_dialect_scf::StructuredControlFlowOpBuilder;

    use super::*;
    use crate::{DataFlowConfig, analyses::dce::DeadCodeAnalysis};

    const SPAN: SourceSpan = SourceSpan::UNKNOWN;

    /// One of the context's getters of an integer type, such as [Context::i32].
    type TypeOf = fn(&Context) -> Type;

    /// `test.check_range` of `value` into `umin ..= umax`, both read as unsigned numbers at the
    /// width of `value`'s type.
    fn check_range(
        builder: &mut FunctionBuilder<'_, OpBuilder>,
        value: ValueRef,
        umin: u128,
        umax: u128,
    ) -> Result<ValueRef, Report> {
        let width = Int::width_for_type(value.borrow().ty()).expect("an integer type");
        let umin = Int::from_unsigned(width, umin).expect("umin fits the type");
        let umax = Int::from_unsigned(width, umax).expect("umax fits the type");
        TestOpBuilder::check_range(builder, value, umin, umax, SPAN)
    }

    /// The entry block's argument `index`.
    fn argument(test: &Test, index: usize) -> ValueRef {
        test.entry_block().borrow().arguments()[index].borrow().as_value_ref()
    }

    #[test]
    fn constants_get_exact_ranges() -> Result<(), Report> {
        let mut test = Test::new("constants_get_exact_ranges", |cx| ([], [cx.i32()]));
        let value = {
            let mut builder = test.function_builder();
            let value = builder.i32(42, SPAN);
            builder.ret([value], SPAN)?;
            value
        };

        let solver = run_integer_range_analysis(&test)?;
        let range = range_for(&solver, value);
        assert_eq!(range.umin(), 42);
        assert_eq!(range.umax(), 42);
        assert_eq!(range.smin(), 42);
        assert_eq!(range.smax(), 42);
        assert!(value_range(&solver, value).expect("expected integer range").is_proven());

        Ok(())
    }

    /// A check of a value whose range is only its type's (assumed) proves the checked range, at the
    /// operand's width, in place of the assumed one.
    #[test]
    fn a_check_replaces_an_assumed_range_with_its_own() -> Result<(), Report> {
        let mut test = Test::new("check_replaces_assumed", |cx| ([cx.i64()], [cx.i64()]));
        let checked = {
            let raw = argument(&test, 0);
            let mut builder = test.function_builder();
            let checked = check_range(&mut builder, raw, 0, u32::MAX as u128)?;
            builder.ret([checked], SPAN)?;
            checked
        };

        let solver = run_integer_range_analysis(&test)?;
        let range = range_for(&solver, checked);
        assert_eq!(range.width(), 64);
        assert_eq!(range.umin(), 0);
        assert_eq!(range.umax(), u32::MAX as u128);
        assert_eq!(range.smin(), 0);
        assert_eq!(range.smax(), u32::MAX as i128);
        assert!(value_range(&solver, checked).expect("expected integer range").is_proven());

        Ok(())
    }

    /// A check of a value with a proven range gives the intersection of the two.
    #[test]
    fn a_check_narrows_a_proven_range() -> Result<(), Report> {
        let mut test = Test::new("check_narrows_proven", |cx| ([cx.i1()], [cx.i32()]));
        let checked = {
            let cond = argument(&test, 0);
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let entry = builder.current_block();
            let if_op = builder.r#if(cond, &[i32], SPAN)?;
            let joined = if_op.as_operation_ref().borrow().results()[0] as ValueRef;
            let (then_region, else_region) = {
                let if_op = if_op.borrow();
                (if_op.then_body().as_region_ref(), if_op.else_body().as_region_ref())
            };
            for (region, value) in [(then_region, 1), (else_region, 5)] {
                let block = builder.create_block_in_region(region);
                builder.switch_to_block(block);
                let value = builder.i32(value, SPAN);
                builder.r#yield([value], SPAN)?;
            }
            builder.switch_to_block(entry);
            let checked = check_range(&mut builder, joined, 3, 10)?;
            builder.ret([checked], SPAN)?;
            checked
        };

        let solver = run_integer_range_analysis(&test)?;
        let range = value_range(&solver, checked).expect("expected integer range");
        assert_eq!(range, IntegerValueRange::proven(ConstantIntRange::from_unsigned(32, 3, 5)));

        Ok(())
    }

    /// A check whose `umin` and `umax` are equal establishes that exact value, at the width of the
    /// operand's type.
    #[test]
    fn assertions_establish_exact_values_at_the_result_width() -> Result<(), Report> {
        let types: [(TypeOf, u16); 4] =
            [(Context::i1, 1), (Context::i8, 8), (Context::i32, 32), (Context::i64, 64)];
        for (ty, width) in types {
            let mut test =
                Test::new("assertion_value_domain", |cx| ([ty(cx), ty(cx)], [ty(cx), ty(cx)]));
            let (one, zero) = {
                let one = argument(&test, 0);
                let zero = argument(&test, 1);
                let mut builder = test.function_builder();
                let one = check_range(&mut builder, one, 1, 1)?;
                let zero = check_range(&mut builder, zero, 0, 0)?;
                builder.ret([one, zero], SPAN)?;
                (one, zero)
            };
            let solver = run_integer_range_analysis(&test)?;
            for (result, expected) in [(one, 1), (zero, 0)] {
                let range = value_range(&solver, result).expect("checked result range");
                assert!(range.is_proven());
                let range = range.range().unwrap();
                assert_eq!(
                    (range.width(), range.umin(), range.umax()),
                    (width, expected, expected)
                );
            }
        }
        Ok(())
    }

    /// A check of a constant that lies on a bound of the checked range keeps the constant's exact
    /// range: the check narrows, it never widens a proven value it admits.
    #[test]
    fn checks_keep_valid_boundary_values_exact() -> Result<(), Report> {
        let types: [(TypeOf, u128); 6] = [
            (Context::i1, 1),
            (Context::i8, 127),
            (Context::i16, 32_767),
            (Context::i32, 2_147_483_647),
            (Context::i64, 9_223_372_036_854_775_807),
            (Context::i128, i128::MAX as u128),
        ];
        for (ty, max) in types {
            for value in [0, max] {
                let mut test = Test::new("check_boundary", |cx| ([], [ty(cx)]));
                let checked = {
                    let ty = ty(test.context());
                    let width = Int::width_for_type(ty).unwrap();
                    let mut builder = test.function_builder();
                    let raw = builder.int(Int::from_unsigned(width, value).unwrap(), ty, SPAN);
                    let checked = check_range(&mut builder, raw, 0, max)?;
                    builder.ret([checked], SPAN)?;
                    checked
                };
                let solver = run_integer_range_analysis(&test)?;
                let range = range_for(&solver, checked);
                assert_eq!((range.umin(), range.umax()), (value, value), "{max}");
            }
        }
        Ok(())
    }

    #[test]
    fn unrealized_conversion_cast_preserves_input_range() -> Result<(), Report> {
        let mut test =
            Test::new("unrealized_conversion_cast_preserves_input_range", |cx| ([], [cx.i32()]));
        let cast = {
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let raw = builder.i64(1 << 32, SPAN);
            let cast = builder.unrealized_conversion_cast(raw, i32, SPAN)?;
            builder.ret([cast], SPAN)?;
            cast
        };

        let solver = run_integer_range_analysis(&test)?;
        let range = range_for(&solver, cast);
        assert_eq!(range.umin(), 1 << 32);
        assert_eq!(range.umax(), 1 << 32);

        Ok(())
    }

    #[test]
    fn inferrable_add_uses_operand_ranges() -> Result<(), Report> {
        let mut test = Test::new("inferrable_add_uses_operand_ranges", |cx| ([], [cx.i32()]));
        let sum = {
            let mut builder = test.function_builder();
            let lhs = builder.i32(1, SPAN);
            let rhs = builder.i32(2, SPAN);
            let sum = ArithOpBuilder::add(&mut builder, lhs, rhs, SPAN)?;
            builder.ret([sum], SPAN)?;
            sum
        };

        let solver = run_integer_range_analysis(&test)?;
        let range = range_for(&solver, sum);
        assert_eq!(range.umin(), 3);
        assert_eq!(range.umax(), 3);

        Ok(())
    }

    #[test]
    fn comparisons_infer_exact_true_values() -> Result<(), Report> {
        let mut test = Test::new("comparisons_infer_exact_true_values", |cx| ([], [cx.i1(); 6]));
        let comparisons = {
            let mut builder = test.function_builder();
            let lt_lhs = builder.i32(1, SPAN);
            let lt_rhs = builder.i32(2, SPAN);
            let lt = builder.ult(lt_lhs, lt_rhs, SPAN)?;
            let lte_lhs = builder.i32(2, SPAN);
            let lte_rhs = builder.i32(2, SPAN);
            let lte = builder.ule(lte_lhs, lte_rhs, SPAN)?;
            let gt_lhs = builder.i32(3, SPAN);
            let gt_rhs = builder.i32(2, SPAN);
            let gt = builder.ugt(gt_lhs, gt_rhs, SPAN)?;
            let gte_lhs = builder.i32(2, SPAN);
            let gte_rhs = builder.i32(2, SPAN);
            let gte = builder.uge(gte_lhs, gte_rhs, SPAN)?;
            let eq_lhs = builder.i32(5, SPAN);
            let eq_rhs = builder.i32(5, SPAN);
            let eq = ArithOpBuilder::eq(&mut builder, eq_lhs, eq_rhs, SPAN)?;
            let ne_lhs = builder.i32(5, SPAN);
            let ne_rhs = builder.i32(6, SPAN);
            let ne = builder.ne(ne_lhs, ne_rhs, SPAN)?;
            let comparisons = [lt, lte, gt, gte, eq, ne];
            builder.ret(comparisons, SPAN)?;
            comparisons
        };

        let solver = run_integer_range_analysis(&test)?;
        for comparison in comparisons {
            assert_bool_range(&solver, comparison, true);
        }

        Ok(())
    }

    #[test]
    fn comparisons_infer_exact_false_values() -> Result<(), Report> {
        let mut test = Test::new("comparisons_infer_exact_false_values", |cx| ([], [cx.i1(); 6]));
        let comparisons = {
            let mut builder = test.function_builder();
            let lt_lhs = builder.i32(2, SPAN);
            let lt_rhs = builder.i32(1, SPAN);
            let lt = builder.ult(lt_lhs, lt_rhs, SPAN)?;
            let lte_lhs = builder.i32(3, SPAN);
            let lte_rhs = builder.i32(2, SPAN);
            let lte = builder.ule(lte_lhs, lte_rhs, SPAN)?;
            let gt_lhs = builder.i32(2, SPAN);
            let gt_rhs = builder.i32(3, SPAN);
            let gt = builder.ugt(gt_lhs, gt_rhs, SPAN)?;
            let gte_lhs = builder.i32(2, SPAN);
            let gte_rhs = builder.i32(3, SPAN);
            let gte = builder.uge(gte_lhs, gte_rhs, SPAN)?;
            let eq_lhs = builder.i32(5, SPAN);
            let eq_rhs = builder.i32(6, SPAN);
            let eq = ArithOpBuilder::eq(&mut builder, eq_lhs, eq_rhs, SPAN)?;
            let ne_lhs = builder.i32(5, SPAN);
            let ne_rhs = builder.i32(5, SPAN);
            let ne = builder.ne(ne_lhs, ne_rhs, SPAN)?;
            let comparisons = [lt, lte, gt, gte, eq, ne];
            builder.ret(comparisons, SPAN)?;
            comparisons
        };

        let solver = run_integer_range_analysis(&test)?;
        for comparison in comparisons {
            assert_bool_range(&solver, comparison, false);
        }

        Ok(())
    }

    /// The signed predicates order operands by their signed bounds: `-1 < 1` read as signed.
    #[test]
    fn comparisons_use_signed_bounds_for_signed_predicates() -> Result<(), Report> {
        let mut test = Test::new("comparisons_use_signed_bounds_for_signed_predicates", |cx| {
            ([], [cx.i1(), cx.i1()])
        });
        let (lt, gt) = {
            let mut builder = test.function_builder();
            let minus_one = builder.i32(-1, SPAN);
            let one = builder.i32(1, SPAN);
            let lt = builder.slt(minus_one, one, SPAN)?;
            let minus_one = builder.i32(-1, SPAN);
            let one = builder.i32(1, SPAN);
            let gt = builder.sgt(minus_one, one, SPAN)?;
            builder.ret([lt, gt], SPAN)?;
            (lt, gt)
        };

        let solver = run_integer_range_analysis(&test)?;
        assert_bool_range(&solver, lt, true);
        assert_bool_range(&solver, gt, false);

        Ok(())
    }

    #[test]
    fn comparisons_remain_unknown_when_ranges_overlap() -> Result<(), Report> {
        let mut test = Test::new("comparisons_remain_unknown_when_ranges_overlap", |cx| {
            ([cx.i32()], [cx.i1()])
        });
        let comparison = {
            let arg = argument(&test, 0);
            let mut builder = test.function_builder();
            let limit = builder.i32(10, SPAN);
            let comparison = builder.ult(arg, limit, SPAN)?;
            builder.ret([comparison], SPAN)?;
            comparison
        };

        let solver = run_integer_range_analysis(&test)?;
        let range = range_for(&solver, comparison);
        assert_eq!(range.umin(), 0);
        assert_eq!(range.umax(), 1);
        assert!(value_range(&solver, comparison).expect("expected integer range").is_proven());

        Ok(())
    }

    /// An operation that neither infers ranges nor folds gives its result the assumed range of its
    /// type, even when its operands are constants: `test.mul` has neither.
    #[test]
    fn unknown_ops_fall_back_to_type_range() -> Result<(), Report> {
        let mut test = Test::new("unknown_ops_fall_back_to_type_range", |cx| ([], [cx.i32()]));
        let value = {
            let mut builder = test.function_builder();
            let lhs = builder.i32(2, SPAN);
            let rhs = builder.i32(3, SPAN);
            let value = TestOpBuilder::mul(&mut builder, lhs, rhs, SPAN)?;
            builder.ret([value], SPAN)?;
            value
        };

        let solver = run_integer_range_analysis(&test)?;
        let range = range_for(&solver, value);
        assert_eq!(range.umin(), 0);
        assert_eq!(range.umax(), u32::MAX as u128);
        assert!(!value_range(&solver, value).expect("expected integer range").is_proven());

        Ok(())
    }

    #[test]
    fn region_branch_results_join_ranges() -> Result<(), Report> {
        let mut test = Test::new("region_branch_results_join_ranges", |cx| ([cx.i1()], [cx.i32()]));
        let if_result = {
            let cond = argument(&test, 0);
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let entry = builder.current_block();
            let if_op = builder.r#if(cond, &[i32], SPAN)?;
            let if_result = if_op.as_operation_ref().borrow().results()[0] as ValueRef;

            let (then_region, else_region) = {
                let if_op = if_op.borrow();
                (if_op.then_body().as_region_ref(), if_op.else_body().as_region_ref())
            };

            let then_block = builder.create_block_in_region(then_region);
            builder.switch_to_block(then_block);
            let one = builder.i32(1, SPAN);
            builder.r#yield([one], SPAN)?;

            let else_block = builder.create_block_in_region(else_region);
            builder.switch_to_block(else_block);
            let five = builder.i32(5, SPAN);
            builder.r#yield([five], SPAN)?;

            builder.switch_to_block(entry);
            builder.ret([if_result], SPAN)?;
            if_result
        };

        let solver = run_integer_range_analysis(&test)?;
        let range = range_for(&solver, if_result);
        assert_eq!(range.umin(), 1);
        assert_eq!(range.umax(), 5);
        assert_eq!(range.smin(), 1);
        assert_eq!(range.smax(), 5);

        Ok(())
    }

    /// An `unchecked` addition whose operands are known to overflow has an undefined result, so
    /// the analysis gives it no proven range: only the assumed range of its type.
    #[test]
    fn regression_unchecked_overflow_does_not_prove_the_result_type() -> Result<(), Report> {
        let mut test = Test::new("unchecked_overflow", |cx| ([], [cx.i32()]));
        let sum = {
            let mut builder = test.function_builder();
            let lhs = builder.i32(-1, SPAN);
            let rhs = builder.i32(1, SPAN);
            let sum =
                ArithOpBuilder::add_unchecked(&mut builder, lhs, rhs, Signedness::Unsigned, SPAN)?;
            builder.ret([sum], SPAN)?;
            sum
        };
        let solver = run_integer_range_analysis(&test)?;
        let range = value_range(&solver, sum).expect("expected integer range");
        assert!(!range.is_proven());
        assert_eq!(
            range.range().copied(),
            Some(ConstantIntRange::from_unsigned(32, 0, u32::MAX as u128))
        );
        Ok(())
    }

    /// An unrealized cast hands an `i32` value the range of its `i64` source, whose signed bounds
    /// are those of a 64-bit number. A signed comparison of the `i32` values must not order them
    /// by those bounds.
    #[test]
    fn regression_signed_comparison_does_not_reuse_wider_signed_bounds() -> Result<(), Report> {
        let mut test = Test::new("signed_cast_comparison", |cx| ([], [cx.i1()]));
        let comparison = {
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let lhs = builder.i64(u32::MAX as i64, SPAN);
            let rhs = builder.i64(1, SPAN);
            let lhs = builder.unrealized_conversion_cast(lhs, i32, SPAN)?;
            let rhs = builder.unrealized_conversion_cast(rhs, i32, SPAN)?;
            let comparison = builder.slt(lhs, rhs, SPAN)?;
            builder.ret([comparison], SPAN)?;
            comparison
        };
        let solver = run_integer_range_analysis(&test)?;
        // In i32, 0xffffffff is -1, so a false-only range is unsound.
        assert_eq!(range_for(&solver, comparison).umax(), 1);
        Ok(())
    }

    #[test]
    fn regression_loop_ranges_converge_within_solver_budget() -> Result<(), Report> {
        let mut test = Test::new("incrementing_loop", |cx| ([cx.i1()], [cx.i32()]));
        let loop_result = {
            let cond = argument(&test, 0);
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let entry = builder.current_block();
            let zero = builder.i32(0, SPAN);
            let loop_op = builder.r#while([zero], &[i32], SPAN)?;
            let before = loop_op.borrow().before().entry().as_block_ref();
            let after = loop_op.borrow().after().entry().as_block_ref();
            builder.switch_to_block(before);
            let current = before.borrow().arguments()[0].borrow().as_value_ref();
            builder.condition(cond, [current], SPAN)?;
            builder.switch_to_block(after);
            let current = after.borrow().arguments()[0].borrow().as_value_ref();
            let one = builder.i32(1, SPAN);
            let next = ArithOpBuilder::add(&mut builder, current, one, SPAN)?;
            builder.r#yield([next], SPAN)?;
            builder.switch_to_block(entry);
            let result = loop_op.borrow().results()[0].borrow().as_value_ref();
            builder.ret([result], SPAN)?;
            result
        };
        let solver = run_integer_range_analysis(&test)?;
        assert!(range_for(&solver, loop_result).umax() >= 100);
        Ok(())
    }

    /// Exercise the transfer functions before SCCP initializes its facts. Normal loading runs
    /// SCCP first, which would hide an eager type-top fallback in integer range analysis.
    struct RangesBeforeConstants;

    impl BuildableDataFlowAnalysis for RangesBeforeConstants {
        type Strategy = SparseDataFlowAnalysis<Self, Forward>;

        fn new(_solver: &mut DataFlowSolver) -> Self {
            Self
        }
    }

    impl SparseForwardDataFlowAnalysis for RangesBeforeConstants {
        type Lattice = IntegerRangeLattice;

        fn visit_operation(
            &self,
            op: &Operation,
            operands: &[AnalysisStateGuard<'_, Self::Lattice>],
            results: &mut [AnalysisStateGuardMut<'_, Self::Lattice>],
            solver: &mut DataFlowSolver,
        ) -> Result<(), Report> {
            IntegerRangeAnalysis.visit_operation(op, operands, results, solver)
        }

        fn set_to_entry_state(&self, lattice: &mut AnalysisStateGuardMut<'_, Self::Lattice>) {
            IntegerRangeAnalysis.set_to_entry_state(lattice);
        }
    }

    fn run_ranges_before_constants(test: &Test) -> Result<DataFlowSolver, Report> {
        let mut solver = DataFlowSolver::new(DataFlowConfig::new());
        solver.load::<DeadCodeAnalysis>();
        solver.load::<RangesBeforeConstants>();
        solver.load::<SparseConstantPropagation>();
        let op = test.function().as_operation_ref();
        solver.initialize_and_run(&op.borrow(), AnalysisManager::new(op, None))?;
        Ok(solver)
    }

    /// A constant, and an operation without a range interface that folds to a constant
    /// (`arith.mul`), get exact ranges once SCCP's facts arrive.
    #[test]
    fn delayed_constant_and_fold_facts_keep_exact_ranges() -> Result<(), Report> {
        let mut test = Test::new("delayed_constants", |cx| ([], [cx.i32(), cx.i32()]));
        let (constant, folded) = {
            let mut builder = test.function_builder();
            let constant = builder.i32(42, SPAN);
            let one = builder.i32(1, SPAN);
            let folded = ArithOpBuilder::mul(&mut builder, constant, one, SPAN)?;
            builder.ret([constant, folded], SPAN)?;
            (constant, folded)
        };
        let solver = run_ranges_before_constants(&test)?;
        for value in [constant, folded] {
            let range = value_range(&solver, value).expect("range after delayed SCCP fact");
            assert_eq!(range, IntegerValueRange::constant(Int::from(42i32)));
        }
        Ok(())
    }

    #[test]
    fn delayed_operand_facts_keep_checked_refinements_precise() -> Result<(), Report> {
        let mut test = Test::new("delayed_refinements", |cx| ([cx.i1()], [cx.i64(), cx.i64()]));
        let (raw, wide, narrow) = {
            let condition = argument(&test, 0);
            let i64 = test.context().i64();
            let mut builder = test.function_builder();
            let entry = builder.current_block();
            let branch = builder.r#if(condition, &[i64], SPAN)?;
            let raw = branch.as_operation_ref().borrow().results()[0] as ValueRef;
            let (then_region, else_region) = {
                let branch = branch.borrow();
                (branch.then_body().as_region_ref(), branch.else_body().as_region_ref())
            };
            for region in [then_region, else_region] {
                let block = builder.create_block_in_region(region);
                builder.switch_to_block(block);
                let seven = builder.i64(7, SPAN);
                builder.r#yield([seven], SPAN)?;
            }
            builder.switch_to_block(entry);
            let wide = check_range(&mut builder, raw, 0, u32::MAX as u128)?;
            let narrow = check_range(&mut builder, raw, 0, 255)?;
            builder.ret([wide, narrow], SPAN)?;
            (raw, wide, narrow)
        };
        // Complete SCCP before loading range analysis. The checks therefore have nonconstant facts
        // when range analysis first sees their still-pending region input. No constant fallback
        // can mask an eager full-range publication by refined_by.
        let mut config = DataFlowConfig::new();
        config.set_interprocedural(true);
        let mut solver = DataFlowSolver::new(config);
        solver.load::<DeadCodeAnalysis>();
        solver.load::<SparseConstantPropagation>();
        let op = test.function().as_operation_ref();
        solver.initialize_and_run(&op.borrow(), AnalysisManager::new(op, None))?;
        solver.load::<IntegerRangeAnalysis>();
        solver.initialize_and_run(&op.borrow(), AnalysisManager::new(op, None))?;
        assert_eq!(range_for(&solver, raw), ConstantIntRange::constant(Int::from(7i64)));
        for value in [wide, narrow] {
            let constant = solver.get::<Lattice<ConstantValue>, _>(&value).expect("SCCP fact");
            assert!(!constant.value().is_uninitialized());
            assert!(
                constant.value().constant_value().is_none(),
                "precision must come from checked refinement, not constant fallback"
            );
            let range = value_range(&solver, value).expect("range after delayed operand fact");
            assert_eq!(range, IntegerValueRange::constant(Int::from(7i64)));
        }
        Ok(())
    }

    /// Builds a function of two parameters of type `ty`, `x` and `unknown`, whose body is what
    /// `build` makes of them, then returns whether the range analysis proves every requirement
    /// that the operation defining `build`'s result states of its operands.
    fn requirements_hold(
        ty: TypeOf,
        build: impl FnOnce(
            &mut FunctionBuilder<'_, OpBuilder>,
            ValueRef,
            ValueRef,
        ) -> Result<ValueRef, Report>,
    ) -> Result<bool, Report> {
        let mut test = Test::new("requirements", |cx| ([ty(cx), ty(cx)], [ty(cx)]));
        let result = {
            let (x, unknown) = (argument(&test, 0), argument(&test, 1));
            let mut builder = test.function_builder();
            let result = build(&mut builder, x, unknown)?;
            builder.ret([result], SPAN)?;
            result
        };
        let solver = run_integer_range_analysis(&test)?;
        let op = result.borrow().get_defining_op().expect("an operation result");
        Ok(operand_range_requirements_hold(&solver, &op.borrow()))
    }

    #[test]
    fn a_shift_amount_requirement_holds_only_below_the_width() -> Result<(), Report> {
        let below = requirements_hold(Context::i32, |b, x, _| {
            let amount = b.i32(31, SPAN);
            ArithOpBuilder::shl(b, x, amount, SPAN)
        })?;
        assert!(below, "a constant amount below the width");

        let checked = requirements_hold(Context::i32, |b, x, unknown| {
            let amount = check_range(b, unknown, 0, 31)?;
            ArithOpBuilder::shl(b, x, amount, SPAN)
        })?;
        assert!(checked, "an amount checked into 0 ..= 31");

        let checked_too_wide = requirements_hold(Context::i32, |b, x, unknown| {
            let amount = check_range(b, unknown, 0, 32)?;
            ArithOpBuilder::shl(b, x, amount, SPAN)
        })?;
        assert!(!checked_too_wide, "an amount checked into 0 ..= 32 may be the width");

        let unknown = requirements_hold(Context::i32, |b, x, unknown| {
            ArithOpBuilder::shl(b, x, unknown, SPAN)
        })?;
        assert!(!unknown, "an unknown amount");

        let width = requirements_hold(Context::i32, |b, x, _| {
            let amount = b.i32(32, SPAN);
            ArithOpBuilder::shl(b, x, amount, SPAN)
        })?;
        assert!(!width, "a constant amount equal to the width");

        Ok(())
    }

    #[test]
    fn a_divisor_requirement_holds_only_when_zero_is_excluded() -> Result<(), Report> {
        let zero = requirements_hold(Context::i32, |b, x, _| {
            let divisor = b.i32(0, SPAN);
            b.udiv(x, divisor, SPAN)
        })?;
        assert!(!zero, "a constant divisor of 0");

        let checked = requirements_hold(Context::i32, |b, x, unknown| {
            let divisor = check_range(b, unknown, 1, 5)?;
            b.udiv(x, divisor, SPAN)
        })?;
        assert!(checked, "a divisor checked into 1 ..= 5");

        let unknown = requirements_hold(Context::i32, |b, x, unknown| b.udiv(x, unknown, SPAN))?;
        assert!(!unknown, "an unknown divisor");

        Ok(())
    }

    /// The ranges of `isize` values are not tracked, so a requirement on an `isize` operand is
    /// never proven, not even for a constant shift amount of 1.
    #[test]
    fn an_isize_shift_requirement_is_never_proven() -> Result<(), Report> {
        let holds = requirements_hold(Context::isize, |b, x, _| {
            let amount = b.isize(1, SPAN);
            ArithOpBuilder::shl(b, x, amount, SPAN)
        })?;
        assert!(!holds);
        Ok(())
    }

    /// An operation that states no requirement of its operands has them all proven, whatever its
    /// operands' ranges: an addition, and a rotation by an unknown amount.
    #[test]
    fn operations_without_requirements_trivially_hold() -> Result<(), Report> {
        let add = requirements_hold(Context::i32, |b, x, unknown| {
            ArithOpBuilder::add(b, x, unknown, SPAN)
        })?;
        assert!(add, "arith.add");

        let rotl = requirements_hold(Context::i32, |b, x, unknown| b.rotl(x, unknown, SPAN))?;
        assert!(rotl, "arith.rotl by an unknown amount");

        Ok(())
    }

    fn run_integer_range_analysis(test: &Test) -> Result<DataFlowSolver, Report> {
        let mut config = DataFlowConfig::new();
        config.set_interprocedural(true);
        config.set_max_worklist_iterations(Some(1024));
        let mut solver = DataFlowSolver::new(config);
        solver.load::<DeadCodeAnalysis>();
        solver.load::<IntegerRangeAnalysis>();
        let op = test.function().as_operation_ref();
        solver.initialize_and_run(&op.borrow(), AnalysisManager::new(op, None))?;
        Ok(solver)
    }

    fn range_for(solver: &DataFlowSolver, value: ValueRef) -> ConstantIntRange {
        value_range(solver, value)
            .and_then(|range| range.range().copied())
            .expect("expected integer range")
    }

    fn assert_bool_range(solver: &DataFlowSolver, value: ValueRef, expected: bool) {
        let value_range = value_range(solver, value).expect("expected integer range");
        assert!(value_range.is_proven());
        let range = value_range.range().copied().expect("expected integer range");
        let expected = u128::from(expected);
        assert_eq!(range.umin(), expected);
        assert_eq!(range.umax(), expected);
    }
}
