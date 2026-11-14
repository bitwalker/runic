mod lattice;

use runic_core::{
    AttributeRef, Forward, Operation, ProgramPoint, Report, SmallVec, ValueRef,
    attributes::IntegerLikeAttr,
    traits::{
        InferIntRangeInterface, OperandRangeRequirement, operation_operand_range_requirement,
    },
};
pub use runic_core::{ConstantIntRange, IntegerValueRange};

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
            (Some(lhs), Some(rhs)) => {
                // Every range of a value has the width of the value's type, so two ranges that
                // meet at a join have one width. A range of another width would be read as a
                // fact about a value it does not describe.
                debug_assert!(
                    lhs.is_untracked() || rhs.is_untracked() || lhs.width() == rhs.width(),
                    "joined ranges of different widths: {lhs} and {rhs}"
                );
                Self::new(lhs.union(&rhs))
            }
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

/// Return true if no operand of `op` has a requirement that is
/// [OperandRangeRequirement::Unexpressed] and the range analysis proves every one that is a
/// [OperandRangeRequirement::Range] ([runic_core::traits::OperandRangeRequirementOpInterface]):
/// `solver` has an initialized range for the operand that [ConstantIntRange::is_within] the
/// requirement. An operation that states no requirement trivially has them all proven.
///
/// This is a sound and conservative answer to "this operation cannot trap or be undefined because
/// of the value of an operand", for operations that state what they require of their operands
/// through the interface: when it is true, the values its operands can take neither make it trap
/// nor leave its result undefined; when it is false, they may. A trap that does not depend on an operand's integer
/// value (an invalid pointer in a load, `ub.unreachable`) is outside what it answers. It is never
/// true for a signed division (`MIN / -1` relates the two operands) or for a checked or
/// unchecked addition, subtraction or multiplication (whether the result fits depends on both),
/// nor for an operation with a requirement on an operand whose ranges are not tracked (`isize`).
pub fn operand_range_requirements_hold(solver: &DataFlowSolver, op: &Operation) -> bool {
    (0..op.num_operands()).all(|index| match operation_operand_range_requirement(op, index) {
        OperandRangeRequirement::None => true,
        OperandRangeRequirement::Unexpressed => false,
        OperandRangeRequirement::Range(requirement) => {
            let operand = op.operands()[index].borrow().as_value_ref();
            value_range(solver, operand)
                .and_then(|range| range.range().copied())
                .is_some_and(|range| range.is_within(&requirement))
        }
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
    use alloc::{format, vec::Vec};

    use runic_core::{
        Context, Int, Op, OpBuilder, OpFoldResult, Signedness, SourceSpan, Type,
        dialects::{
            builtin::{BuiltinOpBuilder, FunctionBuilder},
            test::TestOpBuilder,
        },
        pass::AnalysisManager,
        testing::{Test, text::Rng},
        traits::Foldable,
    };
    use runic_dialect_arith::ArithOpBuilder;
    use runic_dialect_cf::ControlFlowOpBuilder;
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

    /// The result of a `builtin.unrealized_conversion_cast` of `value` to `ty`.
    fn cast(
        builder: &mut FunctionBuilder<'_, OpBuilder>,
        value: ValueRef,
        ty: Type,
    ) -> Result<ValueRef, Report> {
        let cast = builder.unrealized_conversion_cast([value], [ty], SPAN)?;
        Ok(cast.borrow().outputs()[0] as ValueRef)
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

        Ok(())
    }

    /// A check of a value whose range is its type's gives the checked range, at the operand's
    /// width.
    #[test]
    fn a_check_of_a_full_range_gives_its_own() -> Result<(), Report> {
        let mut test = Test::new("check_of_full_range", |cx| ([cx.i64()], [cx.i64()]));
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

        Ok(())
    }

    /// A check of a value with a narrower range gives the intersection of the two.
    #[test]
    fn a_check_narrows_a_wider_range() -> Result<(), Report> {
        let mut test = Test::new("check_narrows_range", |cx| ([cx.i1()], [cx.i32()]));
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
        assert_eq!(range, IntegerValueRange::new(ConstantIntRange::from_unsigned(32, 3, 5)));

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
    /// range: the check narrows, it never widens a value it admits.
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

    /// An unrealized cast to an integer type of the same width keeps its source's range.
    #[test]
    fn an_unrealized_cast_to_the_same_width_keeps_the_range() -> Result<(), Report> {
        let mut test = Test::new("cast_same_width", |cx| ([], [cx.i32()]));
        let cast = {
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let raw = builder.i32(7, SPAN);
            let cast = cast(&mut builder, raw, i32)?;
            builder.ret([cast], SPAN)?;
            cast
        };

        let solver = run_integer_range_analysis(&test)?;
        assert_eq!(range_for(&solver, cast), ConstantIntRange::from_unsigned(32, 7, 7));

        Ok(())
    }

    /// An unrealized cast says nothing about how a value of one width becomes a value of another,
    /// so its result has the range of its own type: `1 << 32` cast from `i64` to `i32`, and `5`
    /// cast from `i8` to `i32`.
    #[test]
    fn an_unrealized_cast_to_another_width_has_the_range_of_its_type() -> Result<(), Report> {
        let mut test = Test::new("cast_other_width", |cx| ([], [cx.i32(), cx.i32()]));
        let (narrowed, widened) = {
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let wide = builder.i64(1 << 32, SPAN);
            let narrowed = cast(&mut builder, wide, i32)?;
            let narrow = builder.i8(5, SPAN);
            let widened = cast(&mut builder, narrow, i32)?;
            builder.ret([narrowed, widened], SPAN)?;
            (narrowed, widened)
        };

        let solver = run_integer_range_analysis(&test)?;
        let top = ConstantIntRange::max_for_type(test.context().i32());
        assert_eq!(range_for(&solver, narrowed), top);
        assert_eq!(range_for(&solver, widened), top);

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

    /// The range of a function argument is its type's: every `i32` is `uge 0`.
    #[test]
    fn a_comparison_decided_by_the_type_is_exact() -> Result<(), Report> {
        let mut test = Test::new("comparison_decided_by_type", |cx| ([cx.i32()], [cx.i1()]));
        let comparison = {
            let arg = argument(&test, 0);
            let mut builder = test.function_builder();
            let zero = builder.i32(0, SPAN);
            let comparison = builder.uge(arg, zero, SPAN)?;
            builder.ret([comparison], SPAN)?;
            comparison
        };

        let solver = run_integer_range_analysis(&test)?;
        assert_bool_range(&solver, comparison, true);

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

        Ok(())
    }

    /// An operation that neither infers ranges nor folds gives its result the range of its
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
    /// the analysis gives it only the range of its type.
    #[test]
    fn an_unchecked_overflowing_sum_has_the_range_of_its_type() -> Result<(), Report> {
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
        assert_eq!(
            range.range().copied(),
            Some(ConstantIntRange::from_unsigned(32, 0, u32::MAX as u128))
        );
        Ok(())
    }

    /// `u32::MAX` and `1` as `i64`, each cast to `i32`. The casts give their results the range of
    /// `i32`, since nothing says how the conversion is realized, so the signed comparison of the
    /// `i32` values stays undecided: ordering them by the `i64` bounds would say `false`, though
    /// `0xffffffff` is `-1` as an `i32`.
    #[test]
    fn regression_signed_comparison_does_not_reuse_wider_signed_bounds() -> Result<(), Report> {
        let mut test = Test::new("signed_cast_comparison", |cx| ([], [cx.i1()]));
        let comparison = {
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let lhs = builder.i64(u32::MAX as i64, SPAN);
            let rhs = builder.i64(1, SPAN);
            let lhs = cast(&mut builder, lhs, i32)?;
            let rhs = cast(&mut builder, rhs, i32)?;
            let comparison = builder.slt(lhs, rhs, SPAN)?;
            builder.ret([comparison], SPAN)?;
            comparison
        };
        let solver = run_integer_range_analysis(&test)?;
        assert_eq!(range_for(&solver, comparison).umax(), 1);
        Ok(())
    }

    /// `%x: i8` through an unrealized cast to `i32`, joined with the `i32` constant 5 as the
    /// result of an `scf.if`, compared `ult 256` and checked into `0 ..= u32::MAX`.
    fn cast_joined_with_a_constant() -> Result<(Test, ValueRef, ValueRef, ValueRef), Report> {
        let mut test = Test::new("cast_join", |cx| ([cx.i1(), cx.i8()], [cx.i1()]));
        let (joined, comparison, checked) = {
            let cond = argument(&test, 0);
            let narrow = argument(&test, 1);
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let entry = builder.current_block();
            let cast = cast(&mut builder, narrow, i32)?;
            let if_op = builder.r#if(cond, &[i32], SPAN)?;
            let joined = if_op.as_operation_ref().borrow().results()[0] as ValueRef;
            let (then_region, else_region) = {
                let if_op = if_op.borrow();
                (if_op.then_body().as_region_ref(), if_op.else_body().as_region_ref())
            };
            let then_block = builder.create_block_in_region(then_region);
            builder.switch_to_block(then_block);
            builder.r#yield([cast], SPAN)?;
            let else_block = builder.create_block_in_region(else_region);
            builder.switch_to_block(else_block);
            let five = builder.i32(5, SPAN);
            builder.r#yield([five], SPAN)?;
            builder.switch_to_block(entry);
            let limit = builder.i32(256, SPAN);
            let comparison = builder.ult(joined, limit, SPAN)?;
            let checked = check_range(&mut builder, cast, 0, u32::MAX as u128)?;
            builder.ret([comparison], SPAN)?;
            (joined, comparison, checked)
        };
        Ok((test, joined, comparison, checked))
    }

    /// The cast's result has the range of `i32`, not the `i8` range of its source: joined with a
    /// constant it stays the full range, the comparison is not decided, and the check gives the
    /// checked range. Any of these taken from the source's range would be wrong if the cast were
    /// realized as a sign extension.
    #[test]
    fn a_cast_to_another_width_has_the_range_of_its_type() -> Result<(), Report> {
        let (test, joined, comparison, checked) = cast_joined_with_a_constant()?;
        let solver = run_integer_range_analysis(&test)?;
        let top = ConstantIntRange::max_for_type(test.context().i32());
        assert_eq!(range_for(&solver, joined), top);
        let comparison = range_for(&solver, comparison);
        assert_eq!((comparison.umin(), comparison.umax()), (0, 1));
        assert_eq!(range_for(&solver, checked), top);
        Ok(())
    }

    /// `%y: i32` cast from an `i8` argument, joined with the `i32` constant 5 through a block
    /// argument `%j`: `arith.cmpi ult %j, 256` is not decided, since nothing says how the cast
    /// turns an `i8` into an `i32`.
    #[test]
    fn a_cast_joined_through_a_block_argument_does_not_decide_a_comparison() -> Result<(), Report> {
        let mut test = Test::new("cast_block_join", |cx| ([cx.i1(), cx.i8()], [cx.i1()]));
        let (joined, comparison) = {
            let cond = argument(&test, 0);
            let narrow = argument(&test, 1);
            let i32 = test.context().i32();
            let mut builder = test.function_builder();
            let cast = cast(&mut builder, narrow, i32)?;
            let from_cast = builder.create_block();
            let from_constant = builder.create_block();
            let join = builder.create_block();
            let joined = builder.append_block_param(join, i32, SPAN);
            builder.cond_br(cond, from_cast, [], from_constant, [], SPAN)?;
            builder.switch_to_block(from_cast);
            builder.br(join, [cast], SPAN)?;
            builder.switch_to_block(from_constant);
            let five = builder.i32(5, SPAN);
            builder.br(join, [five], SPAN)?;
            builder.switch_to_block(join);
            let limit = builder.i32(256, SPAN);
            let comparison = builder.ult(joined, limit, SPAN)?;
            builder.ret([comparison], SPAN)?;
            (joined, comparison)
        };

        let solver = run_integer_range_analysis(&test)?;
        let top = ConstantIntRange::max_for_type(test.context().i32());
        assert_eq!(range_for(&solver, joined), top);
        let comparison = range_for(&solver, comparison);
        assert_eq!((comparison.umin(), comparison.umax()), (0, 1));
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

    /// Whether the range analysis proves the requirement of `cf.assert` of the condition `cond`
    /// gives, built in a function of one `i1` parameter (`unknown`).
    fn assert_requirement_holds(
        cond: impl FnOnce(&mut FunctionBuilder<'_, OpBuilder>, ValueRef) -> ValueRef,
    ) -> Result<bool, Report> {
        let mut test = Test::new("assert", |cx| ([cx.i1()], []));
        let assert = {
            let unknown = argument(&test, 0);
            let mut builder = test.function_builder();
            let cond = cond(&mut builder, unknown);
            let assert = builder.assert(cond, "m", SPAN)?;
            builder.ret([], SPAN)?;
            assert
        };
        let solver = run_integer_range_analysis(&test)?;
        let assert = assert.borrow();
        Ok(operand_range_requirements_hold(&solver, assert.as_operation()))
    }

    /// `cf.assert` traps unless its condition is `1`: the requirement holds for a constant `1`,
    /// and not for a constant `0` or a condition that is not known.
    #[test]
    fn an_assert_requirement_holds_only_for_a_condition_of_one() -> Result<(), Report> {
        assert!(assert_requirement_holds(|b, _| b.i1(true, SPAN))?, "an assert of 1");
        assert!(!assert_requirement_holds(|b, _| b.i1(false, SPAN))?, "an assert of 0");
        assert!(
            !assert_requirement_holds(|_, unknown| unknown)?,
            "an assert of an unknown value"
        );
        Ok(())
    }

    /// `test.check_range` requires its operand to lie in the checked range: the requirement holds
    /// when the operand is proven inside it (a constant, or a value checked into a narrower range
    /// first), and not when it may lie outside.
    #[test]
    fn a_check_range_requirement_holds_when_its_operand_is_proven_inside() -> Result<(), Report> {
        let constant = requirements_hold(Context::i32, |b, _, _| {
            let value = b.i32(7, SPAN);
            check_range(b, value, 0, 10)
        })?;
        assert!(constant, "a constant inside the checked range");

        let narrower = requirements_hold(Context::i32, |b, _, unknown| {
            let value = check_range(b, unknown, 2, 5)?;
            check_range(b, value, 0, 10)
        })?;
        assert!(narrower, "a value checked into 2 ..= 5, checked into 0 ..= 10");

        let outside = requirements_hold(Context::i32, |b, _, _| {
            let value = b.i32(11, SPAN);
            check_range(b, value, 0, 10)
        })?;
        assert!(!outside, "a constant outside the checked range");

        let wider = requirements_hold(Context::i32, |b, _, unknown| {
            let value = check_range(b, unknown, 0, 20)?;
            check_range(b, value, 0, 10)
        })?;
        assert!(!wider, "a value checked into 0 ..= 20, checked into 0 ..= 10");

        let unknown =
            requirements_hold(Context::i32, |b, _, unknown| check_range(b, unknown, 0, 10))?;
        assert!(!unknown, "an unknown value");

        let isize = requirements_hold(Context::isize, |b, _, _| {
            let value = b.isize(1, SPAN);
            check_range(b, value, 0, 10)
        })?;
        assert!(!isize, "an isize operand, whose requirement is unexpressed");

        Ok(())
    }

    /// A value checked into `0 ..= signed max` is proven non-negative: its range has the signed
    /// bounds `0 ..= signed max`, and `cmpi slt %checked, 0` is exactly false, at every width.
    #[test]
    fn a_value_checked_into_the_non_negative_half_is_never_signed_negative() -> Result<(), Report> {
        let widths: [(u32, TypeOf); 5] = [
            (8, Context::i8),
            (16, Context::i16),
            (32, Context::i32),
            (64, Context::i64),
            (128, Context::i128),
        ];
        for (width, ty) in widths {
            let signed_max = (1u128 << (width - 1)) - 1;
            let mut test = Test::new("non_negative", move |cx| ([ty(cx)], []));
            let param = argument(&test, 0);
            let int_ty = ty(&test.context_rc());
            let (checked, negative) = {
                let mut builder = test.function_builder();
                let checked = check_range(&mut builder, param, 0, signed_max)?;
                let zero = builder.int(Int::from_bits(width, 0), int_ty, SPAN);
                let negative = builder.slt(checked, zero, SPAN)?;
                builder.ret([], SPAN)?;
                (checked, negative)
            };
            let solver = run_integer_range_analysis(&test)?;
            let range = value_range(&solver, checked).expect("a range for the checked value");
            let range = range.range().copied().unwrap();
            assert_eq!((range.umin(), range.umax()), (0, signed_max), "i{width}");
            assert_eq!((range.smin(), range.smax()), (0, signed_max as i128), "i{width}");
            assert_bool_range(&solver, negative, false);
        }
        Ok(())
    }

    /// A signed division of the minimum value by -1 traps; its divisor is not zero, but the
    /// requirement on the dividend is unexpressed, so the operation is not reported as safe. The
    /// same holds for `sdivrem`.
    #[test]
    fn a_trapping_signed_division_is_not_reported_as_safe() {
        for width in [8u32, 64, 128] {
            let (holds, folds) =
                holds_and_folds(width, Int::signed_min(width), Int::all_ones(width), |b, x, y| {
                    b.sdiv(x, y, SPAN)
                });
            assert!(!folds, "MIN / -1 traps at i{width}");
            assert!(!holds, "MIN / -1 at i{width}");
        }
        let (holds, folds) = holds_and_folds(8, Int::signed_min(8), Int::all_ones(8), |b, x, y| {
            Ok(b.sdivrem(x, y, SPAN)?.0)
        });
        assert!(!folds && !holds, "sdivrem: holds={holds} folds={folds}");
    }

    /// A checked addition of `127 : i8` and `1 : i8` traps and an unchecked unsigned one of
    /// `-1` and `1` is undefined: neither holds. The wrapping addition of the same operands is
    /// defined, and holds.
    #[test]
    fn an_addition_holds_only_when_it_wraps() {
        let (holds, folds) = holds_and_folds(8, Int::from(127i8), Int::from(1i8), |b, x, y| {
            b.add_checked(x, y, Signedness::Signed, SPAN)
        });
        assert!(!holds && !folds, "checked: holds={holds} folds={folds}");
        let (holds, folds) = holds_and_folds(8, Int::from(-1i8), Int::from(1i8), |b, x, y| {
            ArithOpBuilder::add_unchecked(b, x, y, Signedness::Unsigned, SPAN)
        });
        assert!(!holds && !folds, "unchecked: holds={holds} folds={folds}");
        let (holds, folds) = holds_and_folds(8, Int::from(127i8), Int::from(1i8), |b, x, y| {
            ArithOpBuilder::add(b, x, y, SPAN)
        });
        assert!(holds && folds, "wrapping: holds={holds} folds={folds}");
    }

    /// A builder of a binary `arith` operation, for the execution comparisons below.
    type Build =
        fn(&mut FunctionBuilder<'_, OpBuilder>, ValueRef, ValueRef) -> Result<ValueRef, Report>;

    const SHIFTS: [(&str, Build); 3] = [
        ("shl", |b, x, y| ArithOpBuilder::shl(b, x, y, SPAN)),
        ("shr", |b, x, y| b.shr(x, y, SPAN)),
        ("ashr", |b, x, y| b.ashr(x, y, SPAN)),
    ];

    const UNSIGNED_DIVISIONS: [(&str, Build); 4] = [
        ("udiv", |b, x, y| b.udiv(x, y, SPAN)),
        ("urem", |b, x, y| b.urem(x, y, SPAN)),
        ("srem", |b, x, y| b.srem(x, y, SPAN)),
        ("udivrem", |b, x, y| Ok(b.udivrem(x, y, SPAN)?.0)),
    ];

    const SIGNED_DIVISIONS: [(&str, Build); 2] = [
        ("sdiv", |b, x, y| b.sdiv(x, y, SPAN)),
        ("sdivrem", |b, x, y| Ok(b.sdivrem(x, y, SPAN)?.0)),
    ];

    /// `add`, `sub` and `mul` in the wrapping mode, which accept every value.
    const WRAPPING: [(&str, Build); 3] = [
        ("add", |b, x, y| ArithOpBuilder::add(b, x, y, SPAN)),
        ("sub", |b, x, y| ArithOpBuilder::sub(b, x, y, SPAN)),
        ("mul", |b, x, y| ArithOpBuilder::mul(b, x, y, SPAN)),
    ];

    /// `add`, `sub` and `mul` in a checked or unchecked mode, whose requirement is unexpressed.
    const NOT_WRAPPING: [(&str, Build); 6] = [
        ("add checked signed", |b, x, y| b.add_checked(x, y, Signedness::Signed, SPAN)),
        ("add unchecked unsigned", |b, x, y| {
            ArithOpBuilder::add_unchecked(b, x, y, Signedness::Unsigned, SPAN)
        }),
        ("sub checked unsigned", |b, x, y| {
            b.sub_checked(x, y, Signedness::Unsigned, SPAN)
        }),
        ("sub unchecked signed", |b, x, y| {
            ArithOpBuilder::sub_unchecked(b, x, y, Signedness::Signed, SPAN)
        }),
        ("mul checked signed", |b, x, y| b.mul_checked(x, y, Signedness::Signed, SPAN)),
        ("mul unchecked unsigned", |b, x, y| {
            ArithOpBuilder::mul_unchecked(b, x, y, Signedness::Unsigned, SPAN)
        }),
    ];

    /// The type of width `width`, or `isize` for 0.
    fn ty_of(context: &Context, width: u32) -> Type {
        if width == 0 {
            context.isize()
        } else {
            context.int_type(width)
        }
    }

    /// A value of width `width` that is likely to sit on an edge for a shift or a division: 0, 1,
    /// all ones, the sign bit, the width and its neighbours, the largest signed value, or random.
    fn edge(rng: &mut Rng, width: u32) -> Int {
        let mask = if width == 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        let bits = match rng.below(10) {
            0 => 0,
            1 => 1,
            2 => mask,
            3 => 1u128 << (width - 1),
            4 => u128::from(width),
            5 => u128::from(width).wrapping_sub(1),
            6 => u128::from(width) + 1,
            7 => (1u128 << (width - 1)).wrapping_sub(1),
            _ => rng.next_u128(),
        };
        Int::from_bits(width, bits & mask)
    }

    /// Builds `f() { %a = const lhs; %b = const rhs; %r = op %a, %b; }` at `width` (0 for `isize`)
    /// and returns (the operation's requirements hold, the operation folds). `arith`'s folds
    /// decline exactly when the operation traps or is undefined.
    fn holds_and_folds(width: u32, lhs: Int, rhs: Int, build: Build) -> (bool, bool) {
        let mut test = Test::new("f", move |cx| ([], [ty_of(cx, width)]));
        let ty = ty_of(&test.context_rc(), width);
        let result = {
            let mut b = test.function_builder();
            let a = b.int(lhs, ty, SPAN);
            let c = b.int(rhs, ty, SPAN);
            let result = build(&mut b, a, c).unwrap();
            b.ret([result], SPAN).unwrap();
            result
        };
        let solver = run_integer_range_analysis(&test).unwrap();
        let op = result.borrow().get_defining_op().unwrap();
        let op = op.borrow();
        let holds = operand_range_requirements_hold(&solver, &op);
        let mut results = SmallVec::<[OpFoldResult; 1]>::default();
        let folds = op.fold(&mut results).is_ok();
        (holds, folds)
    }

    /// For the shifts, the divisions whose only requirement is on the divisor and the wrapping
    /// `add`, `sub` and `mul`, at widths 1 to 128 with constant operands on the edges: the
    /// requirements hold exactly when the operation folds, that is, when it neither traps nor is
    /// undefined. A signed division, and an `add`, `sub` or `mul` that does not wrap, never holds.
    #[test]
    fn requirements_on_constants_hold_exactly_when_the_operation_is_defined() {
        let mut rng = Rng(0x00b3_5eed);
        let mut disagreements = Vec::new();
        for width in [1u32, 2, 7, 8, 32, 64, 127, 128] {
            for (name, build) in
                SHIFTS.iter().chain(UNSIGNED_DIVISIONS.iter()).chain(WRAPPING.iter())
            {
                for _ in 0..40 {
                    let lhs = edge(&mut rng, width);
                    let rhs = edge(&mut rng, width);
                    let (holds, folds) = holds_and_folds(width, lhs, rhs, *build);
                    if holds != folds {
                        disagreements.push(format!(
                            "{name} i{width} {lhs}, {rhs}: holds={holds} folds={folds}"
                        ));
                    }
                }
            }
            for (name, build) in SIGNED_DIVISIONS.into_iter().chain(NOT_WRAPPING) {
                for _ in 0..40 {
                    let lhs = edge(&mut rng, width);
                    let rhs = edge(&mut rng, width);
                    let (holds, _) = holds_and_folds(width, lhs, rhs, build);
                    if holds {
                        disagreements.push(format!("{name} i{width} {lhs}, {rhs}: holds"));
                    }
                }
            }
        }
        assert!(disagreements.is_empty(), "{disagreements:#?}");
    }

    /// For `isize` a requirement is unexpressed, so it never holds, whatever the constants.
    #[test]
    fn requirements_on_isize_constants_never_hold() {
        let mut rng = Rng(0x0001_512e);
        for (name, build) in
            SHIFTS.iter().chain(UNSIGNED_DIVISIONS.iter()).chain(SIGNED_DIVISIONS.iter())
        {
            for _ in 0..20 {
                let lhs = Int::from(rng.below(100) as i64);
                let rhs = Int::from(1 + rng.below(20) as i64);
                let (holds, _) = holds_and_folds(0, lhs, rhs, *build);
                assert!(!holds, "{name} isize {lhs}, {rhs}");
            }
        }
    }

    /// With a divisor or amount bounded by `test.check_range`, at each width: a shift or an
    /// unsigned division holds exactly when the checked range excludes the trapping values; a
    /// signed division never holds.
    #[test]
    fn requirements_on_checked_operands_hold_exactly_when_the_range_is_safe() {
        let mut rng = Rng(0x0c4e_c4ed);
        let widths: [(u32, TypeOf); 4] =
            [(1, Context::i1), (8, Context::i8), (64, Context::i64), (128, Context::i128)];
        for (width, ty) in widths {
            for _ in 0..60 {
                let a = edge(&mut rng, width).unsigned();
                let c = edge(&mut rng, width).unsigned();
                let (lo, hi) = if a <= c { (a, c) } else { (c, a) };
                for (kind, safe) in
                    [("shl", hi < u128::from(width)), ("udiv", lo >= 1), ("sdiv", false)]
                {
                    let holds = requirements_hold(ty, |b, x, unknown| {
                        let checked = check_range(b, unknown, lo, hi)?;
                        match kind {
                            "shl" => ArithOpBuilder::shl(b, x, checked, SPAN),
                            "udiv" => b.udiv(x, checked, SPAN),
                            _ => b.sdiv(x, checked, SPAN),
                        }
                    })
                    .unwrap();
                    assert_eq!(holds, safe, "{kind} i{width} checked into {lo} ..= {hi}");
                }
            }
        }
    }

    /// Each requirement on an operand that `arith` states as a range, read against a brute-force
    /// model at small widths: every divisor or amount inside the range is one the operation folds
    /// for with every dividend (except `MIN / -1` for `sdiv` and `sdivrem`, whose requirement on the
    /// dividend is unexpressed), and every value outside is one it never folds for.
    #[test]
    fn stated_requirements_match_execution_at_small_widths() {
        for width in [1u32, 2, 3, 5, 8] {
            let n = 1u128 << width;
            for (name, build) in
                SHIFTS.iter().chain(UNSIGNED_DIVISIONS.iter()).chain(SIGNED_DIVISIONS.iter())
            {
                let mut test =
                    Test::new("f", move |cx| ([cx.int_type(width), cx.int_type(width)], []));
                let context = test.context_rc();
                let (x, y) = (argument(&test, 0), argument(&test, 1));
                let result = {
                    let mut b = test.function_builder();
                    let r = build(&mut b, x, y).unwrap();
                    b.ret([], SPAN).unwrap();
                    r
                };
                let op = result.borrow().get_defining_op().unwrap();
                let op = op.borrow();
                let expected_lhs = if name.starts_with("sdiv") {
                    OperandRangeRequirement::Unexpressed
                } else {
                    OperandRangeRequirement::None
                };
                assert_eq!(operation_operand_range_requirement(&op, 0), expected_lhs, "{name}");
                let OperandRangeRequirement::Range(requirement) =
                    operation_operand_range_requirement(&op, 1)
                else {
                    panic!("{name} i{width}: operand 1 has no range requirement");
                };
                // Whether the operation folds, that is, is defined, for constant operands
                let folds = |lhs: u128, rhs: u128| {
                    let ty = context.int_type(width);
                    let operands = [lhs, rhs].map(|bits| {
                        Some(context.int_attr(Int::from_bits(width, bits), ty).as_attribute_ref())
                    });
                    let mut results = SmallVec::<[OpFoldResult; 1]>::default();
                    op.fold_with(&operands, &mut results).is_ok()
                };
                for rhs in 0..n {
                    let inside = ConstantIntRange::constant(Int::from_bits(width, rhs))
                        .is_within(&requirement);
                    let folds_for_all = (0..n).all(|lhs| folds(lhs, rhs));
                    let folds_for_some = (0..n).any(|lhs| folds(lhs, rhs));
                    if !inside {
                        assert!(
                            !folds_for_some,
                            "{name} i{width} rhs={rhs} is outside but defined"
                        );
                    } else if !name.starts_with("sdiv") || rhs != n - 1 {
                        assert!(folds_for_all, "{name} i{width} rhs={rhs} is inside but traps");
                    } else {
                        // -1 divides every dividend but the minimum (at `i1`, -1 itself)
                        assert!(!folds(1 << (width - 1), rhs), "{name} i{width}: MIN / -1");
                        assert!(folds_for_some, "{name} i{width} rhs={rhs}");
                    }
                }
            }
        }
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
        let range = value_range.range().copied().expect("expected integer range");
        let expected = u128::from(expected);
        assert_eq!(range.umin(), expected);
        assert_eq!(range.umax(), expected);
    }
}
