//! Canonicalization patterns for the `arith` dialect.
//!
//! The patterns are:
//!
//! * `simplify-bool-compare`: `cmpi eq`/`cmpi ne` of a widened boolean against 0 or 1 becomes the
//!   boolean or its negation.
//! * `simplify-not-compare`: `bnot` of the `i1` result of a single-use `cmpi` becomes the `cmpi`
//!   with the inverted predicate.
//! * `simplify-trunc-to-bool`: `trunc` to `i1` of a widened boolean becomes the boolean.

use alloc::rc::Rc;

use runic_core::{
    attributes::IntegerLikeAttr,
    matchers::Matcher,
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind, RewritePattern, RewriterExt},
    traits::TransparentCast,
    *,
};

use crate::*;

/// How many widening/transparent-cast operations [widened_bool] looks through at most.
const MAX_WIDENING_DEPTH: usize = 4;

/// An integer that is a boolean `b` widened: zero when `b` is false, nonzero when it is true.
struct WidenedBool {
    /// The `i1` value that was widened.
    value: ValueRef,
    /// Whether the widened value is exactly 1 when `b` is true: every step zero-extends (or is a
    /// transparent cast). A sign extension makes it all ones instead.
    true_is_one: bool,
}

/// Returns the `i1` value that `value` is a widening of.
///
/// Looks through up to [MAX_WIDENING_DEPTH] `arith.zext`, `arith.sext` and transparent casts
/// between integer types; each keeps a false boolean zero and a true one nonzero, and keeps the
/// lowest bit. Returns `value` itself when it is already an `i1`.
fn widened_bool(value: ValueRef) -> Option<WidenedBool> {
    let is_bool = |value: &ValueRef| value.borrow().ty().integer_width() == Some(1);
    let mut current = value;
    let mut true_is_one = true;
    for _ in 0..MAX_WIDENING_DEPTH {
        if is_bool(&current) {
            return Some(WidenedBool {
                value: current,
                true_is_one,
            });
        }
        let defining_op = current.borrow().get_defining_op()?;
        let op = defining_op.borrow();
        let is_sext = op.downcast_ref::<Sext>().is_some();
        let is_widening = is_sext
            || op.downcast_ref::<Zext>().is_some()
            || op.implements::<dyn TransparentCast>();
        if !is_widening || op.num_operands() != 1 {
            return None;
        }
        let operand = op.operands()[0].borrow().as_value_ref();
        if !operand.borrow().ty().is_integer() {
            return None;
        }
        true_is_one &= !is_sext;
        current = operand;
    }
    // The value reached by the last step is a candidate too.
    is_bool(&current).then_some(WidenedBool {
        value: current,
        true_is_one,
    })
}

/// Returns the boolean a constant 0 or 1 integer `value` stands for, or `None` for anything else.
fn constant_bit(value: ValueRef) -> Option<bool> {
    let defining_op = value.borrow().get_defining_op()?;
    let attr = matchers::constant().matches(&*defining_op.borrow())?;
    let attr = attr.borrow();
    let value = attr.as_attr().as_trait::<dyn IntegerLikeAttr>()?.as_int();
    if value.is_zero() {
        Some(false)
    } else if value.is_one() {
        Some(true)
    } else {
        None
    }
}

/// Returns true if `value` has exactly one use.
fn has_single_use(value: ValueRef) -> bool {
    let value = value.borrow();
    let mut uses = value.iter_uses();
    uses.next().is_some() && uses.next().is_none()
}

/// Simplifies `arith.cmpi eq`/`arith.cmpi ne` of a widened boolean against a constant 0 or 1 to
/// the boolean itself or its negation.
///
/// Frontends commonly widen booleans to an integer and test them against 0 again before a branch
/// or select, so these comparisons are common, while the boolean is already at hand:
///
/// * `ne(x, 0)` becomes `b` and `eq(x, 0)` becomes `bnot b`, however `x` widens `b`;
/// * `eq(x, 1)` becomes `b` and `ne(x, 1)` becomes `bnot b`, when `x` widens `b` by zero
///   extension only. A sign-extended true boolean is all ones, which is not 1 at a width above 1,
///   so those comparisons are left alone.
///
/// where `b` is the `i1` that `x` widens (see [widened_bool]).
pub(crate) struct SimplifyBoolCompare {
    info: PatternInfo,
}

impl SimplifyBoolCompare {
    /// Creates the pattern, matching `arith.cmpi`.
    pub fn new(context: Rc<Context>) -> Self {
        let cmpi = context
            .get_or_register_dialect::<ArithDialect>()
            .expect_registered_name::<Cmpi>();
        Self {
            info: PatternInfo::new(
                context,
                "simplify-bool-compare",
                PatternKind::Operation(cmpi),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for SimplifyBoolCompare {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for SimplifyBoolCompare {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let (span, is_eq, lhs, rhs) = {
            let op = operation.borrow();
            let Some(cmpi) = op.downcast_ref::<Cmpi>() else {
                return Ok(false);
            };
            let is_eq = match *cmpi.get_predicate() {
                CmpPredicate::Eq => true,
                CmpPredicate::Ne => false,
                _ => return Ok(false),
            };
            (op.span(), is_eq, cmpi.lhs().as_value_ref(), cmpi.rhs().as_value_ref())
        };

        let (widened, bit) = if let Some(bit) = constant_bit(rhs) {
            (widened_bool(lhs), bit)
        } else if let Some(bit) = constant_bit(lhs) {
            (widened_bool(rhs), bit)
        } else {
            return Ok(false);
        };
        let Some(widened) = widened else {
            return Ok(false);
        };
        if bit && !widened.true_is_one {
            return Ok(false);
        }

        // `eq(x, 1)`/`ne(x, 0)` is `b`; the other two combinations negate it.
        let replacement = if is_eq == bit {
            widened.value
        } else {
            rewriter.set_insertion_point_before(operation);
            rewriter.bnot(widened.value, span)?
        };
        rewriter.replace_op_with_values(operation, &[Some(replacement)]);
        Ok(true)
    }
}

/// Simplifies `arith.bnot` of the `i1` result of a single-use `arith.cmpi` into the `cmpi` with the
/// inverted predicate, so the negation disappears from the generated code. On `i1`, bitwise NOT is
/// logical NOT.
pub(crate) struct SimplifyNotCompare {
    info: PatternInfo,
}

impl SimplifyNotCompare {
    /// Creates the pattern, matching `arith.bnot`.
    pub fn new(context: Rc<Context>) -> Self {
        let bnot = context
            .get_or_register_dialect::<ArithDialect>()
            .expect_registered_name::<Bnot>();
        Self {
            info: PatternInfo::new(
                context,
                "simplify-not-compare",
                PatternKind::Operation(bnot),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for SimplifyNotCompare {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for SimplifyNotCompare {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let operand = {
            let op = operation.borrow();
            let Some(bnot) = op.downcast_ref::<Bnot>() else {
                return Ok(false);
            };
            bnot.operand().as_value_ref()
        };
        if operand.borrow().ty().integer_width() != Some(1) || !has_single_use(operand) {
            return Ok(false);
        }
        let Some(compare) = operand.borrow().get_defining_op() else {
            return Ok(false);
        };
        // The replacement executes where the comparison was, so it carries that span.
        let (span, predicate, lhs, rhs) = {
            let compare = compare.borrow();
            let Some(cmpi) = compare.downcast_ref::<Cmpi>() else {
                return Ok(false);
            };
            (
                compare.span(),
                *cmpi.get_predicate(),
                cmpi.lhs().as_value_ref(),
                cmpi.rhs().as_value_ref(),
            )
        };

        // The inverted comparison takes the place of the original one, not of the `bnot`: the
        // `bnot` may sit in a nested region (a loop body), and rebuilding the comparison there
        // would evaluate it per iteration and keep its operands live across the region. It also
        // belongs to the comparison's inline frame, not to the frame of the `bnot` being rewritten.
        let replacement = {
            let mut scope = rewriter.with_inline_call_chain(compare);
            scope.set_insertion_point_after(compare);
            scope.cmpi(predicate.inverted(), lhs, rhs, span)?
        };
        rewriter.replace_op_with_values(operation, &[Some(replacement)]);
        Ok(true)
    }
}

/// Simplifies `arith.trunc` to `i1` of a widened boolean into the boolean itself: zero and sign
/// extension both keep the lowest bit.
pub(crate) struct SimplifyTruncToBool {
    info: PatternInfo,
}

impl SimplifyTruncToBool {
    /// Creates the pattern, matching `arith.trunc`.
    pub fn new(context: Rc<Context>) -> Self {
        let trunc_op = context
            .get_or_register_dialect::<ArithDialect>()
            .expect_registered_name::<Trunc>();
        Self {
            info: PatternInfo::new(
                context,
                "simplify-trunc-to-bool",
                PatternKind::Operation(trunc_op),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for SimplifyTruncToBool {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for SimplifyTruncToBool {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let operand = {
            let op = operation.borrow();
            let Some(trunc) = op.downcast_ref::<Trunc>() else {
                return Ok(false);
            };
            if trunc.result().ty().integer_width() != Some(1) {
                return Ok(false);
            }
            trunc.operand().as_value_ref()
        };
        let Some(widened) = widened_bool(operand) else {
            return Ok(false);
        };
        rewriter.replace_op_with_values(operation, &[Some(widened.value)]);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, rc::Rc};

    use runic_core::{
        Context, Int, Op, OperationRef, Report, SourceSpan, Type, Value, ValueRef,
        dialects::builtin::{BuiltinOpBuilder, Ret},
        patterns::{
            FrozenRewritePatternSet, GreedyRewriteConfig, RegionSimplificationLevel,
            RewritePatternSet, apply_patterns_and_fold_greedily,
        },
        testing::Test,
        traits::{Canonicalizable, ConstantLike},
    };

    use crate::*;

    const SPAN: SourceSpan = SourceSpan::UNKNOWN;

    /// Applies the canonicalization patterns `register` adds, and folding, to `test`'s function.
    /// Returns whether anything changed.
    fn canonicalize(
        test: &Test,
        register: impl FnOnce(&mut RewritePatternSet, Rc<Context>),
    ) -> bool {
        let context = test.context_rc();
        let mut patterns = RewritePatternSet::new(context.clone());
        register(&mut patterns, context);
        let patterns = Rc::new(FrozenRewritePatternSet::new(patterns));

        let mut config = GreedyRewriteConfig::default();
        config.with_region_simplification_level(RegionSimplificationLevel::None);

        let function = test.function().as_operation_ref();
        match apply_patterns_and_fold_greedily(function, patterns, config) {
            Ok(changed) => changed,
            Err(changed) => panic!("canonicalization failed (changed={changed})"),
        }
    }

    /// The operand of the `builtin.ret` of `test`'s function.
    fn returned(test: &Test) -> ValueRef {
        let mut value = None;
        test.function()
            .borrow()
            .as_operation()
            .prewalk_all(|op: &runic_core::Operation| {
                if op.is::<Ret>() {
                    value = Some(op.operands().all()[0].borrow().as_value_ref());
                }
            });
        value.expect("a builtin.ret")
    }

    /// The operation defining `value`, if any.
    fn defining_op(value: ValueRef) -> Option<OperationRef> {
        value.borrow().get_defining_op()
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    enum Extension {
        Zero,
        Sign,
    }

    /// What `cmpi predicate (ext b), bit` canonicalizes to.
    #[derive(Debug, PartialEq, Eq)]
    enum Outcome {
        /// The boolean `b` itself.
        Bool,
        /// `bnot b`.
        NotBool,
        /// The comparison is left alone.
        Unchanged,
    }

    /// Builds `f(%b: i1) -> i1` returning `cmpi predicate (ext b : i32), bit` (or with the operands
    /// swapped), canonicalizes it, and says what it returns.
    fn simplify_bool_compare(
        extension: Extension,
        predicate: CmpPredicate,
        bit: i32,
        constant_first: bool,
    ) -> Outcome {
        simplify_bool_compare_at(extension, predicate, bit, constant_first, 32)
    }

    /// As [simplify_bool_compare], with the boolean widened to `iN`, `N` being `width`.
    fn simplify_bool_compare_at(
        extension: Extension,
        predicate: CmpPredicate,
        bit: i32,
        constant_first: bool,
        width: u32,
    ) -> Outcome {
        let mut test = Test::new("f", |cx| ([cx.i1()], [cx.i1()]));
        test.context().get_or_register_dialect::<ArithDialect>();
        let flag = test.entry_block().borrow().arguments()[0].borrow().as_value_ref();
        {
            let mut b = test.function_builder();
            let wide = b.builder().context().int_type(width);
            let widened = match extension {
                Extension::Zero => b.zext(flag, wide, SPAN),
                Extension::Sign => b.sext(flag, wide, SPAN),
            }
            .unwrap();
            let constant = b.int(
                if bit == 0 {
                    Int::zero(width)
                } else {
                    Int::one(width)
                },
                wide,
                SPAN,
            );
            let (lhs, rhs) = if constant_first {
                (constant, widened)
            } else {
                (widened, constant)
            };
            let compared = b.cmpi(predicate, lhs, rhs, SPAN).unwrap();
            b.ret(Some(compared), SPAN).unwrap();
        }
        canonicalize(&test, Cmpi::get_canonicalization_patterns);
        test.function()
            .borrow()
            .as_operation()
            .verify()
            .expect("valid after canonicalization");

        let result = returned(&test);
        if result == flag {
            return Outcome::Bool;
        }
        let op = defining_op(result).expect("an operation result");
        let op = op.borrow();
        if let Some(bnot) = op.downcast_ref::<Bnot>() {
            assert_eq!(bnot.operand().as_value_ref(), flag);
            return Outcome::NotBool;
        }
        assert!(op.is::<Cmpi>(), "{}", format!("{}", op));
        Outcome::Unchanged
    }

    #[test]
    fn a_zero_extended_boolean_compared_with_zero_or_one_is_the_boolean() {
        for constant_first in [false, true] {
            let check = |predicate, bit| {
                simplify_bool_compare(Extension::Zero, predicate, bit, constant_first)
            };
            assert_eq!(check(CmpPredicate::Ne, 0), Outcome::Bool);
            assert_eq!(check(CmpPredicate::Eq, 1), Outcome::Bool);
            assert_eq!(check(CmpPredicate::Eq, 0), Outcome::NotBool);
            assert_eq!(check(CmpPredicate::Ne, 1), Outcome::NotBool);
            // Only equality is simplified.
            assert_eq!(check(CmpPredicate::Ult, 1), Outcome::Unchanged);
        }
    }

    #[test]
    fn a_boolean_widened_to_other_widths_is_simplified_the_same_way() {
        for width in [2, 8] {
            for constant_first in [false, true] {
                let zero = |predicate, bit| {
                    simplify_bool_compare_at(Extension::Zero, predicate, bit, constant_first, width)
                };
                assert_eq!(zero(CmpPredicate::Ne, 0), Outcome::Bool, "zext i{width}");
                assert_eq!(zero(CmpPredicate::Eq, 1), Outcome::Bool, "zext i{width}");
                assert_eq!(zero(CmpPredicate::Eq, 0), Outcome::NotBool, "zext i{width}");
                assert_eq!(zero(CmpPredicate::Ne, 1), Outcome::NotBool, "zext i{width}");
                assert_eq!(zero(CmpPredicate::Ult, 1), Outcome::Unchanged, "zext i{width}");
                let sign = |predicate, bit| {
                    simplify_bool_compare_at(Extension::Sign, predicate, bit, constant_first, width)
                };
                assert_eq!(sign(CmpPredicate::Ne, 0), Outcome::Bool, "sext i{width}");
                assert_eq!(sign(CmpPredicate::Eq, 0), Outcome::NotBool, "sext i{width}");
                assert_eq!(sign(CmpPredicate::Eq, 1), Outcome::Unchanged, "sext i{width}");
                assert_eq!(sign(CmpPredicate::Ne, 1), Outcome::Unchanged, "sext i{width}");
            }
        }
    }

    #[test]
    fn a_sign_extended_boolean_is_simplified_only_against_zero() {
        for constant_first in [false, true] {
            let check = |predicate, bit| {
                simplify_bool_compare(Extension::Sign, predicate, bit, constant_first)
            };
            assert_eq!(check(CmpPredicate::Ne, 0), Outcome::Bool);
            assert_eq!(check(CmpPredicate::Eq, 0), Outcome::NotBool);
            // A true boolean sign-extends to -1, never 1: `eq 1` is always false and `ne 1` always
            // true, so neither is the boolean or its negation.
            assert_eq!(check(CmpPredicate::Eq, 1), Outcome::Unchanged);
            assert_eq!(check(CmpPredicate::Ne, 1), Outcome::Unchanged);
        }
    }

    /// Builds `f(%a: i32, %b: i32) -> i1` returning `bnot (cmpi predicate %a, %b)`, with a second
    /// use of the comparison if `shared`, and canonicalizes it.
    fn simplify_not_compare(predicate: CmpPredicate, shared: bool) -> Test {
        let mut test = Test::new("f", |cx| ([cx.i32(), cx.i32()], [cx.i1()]));
        test.context().get_or_register_dialect::<ArithDialect>();
        let args = test.entry_block().borrow().arguments().to_vec();
        let (lhs, rhs) = (args[0].borrow().as_value_ref(), args[1].borrow().as_value_ref());
        {
            let mut b = test.function_builder();
            let compared = b.cmpi(predicate, lhs, rhs, SPAN).unwrap();
            let negated = b.bnot(compared, SPAN).unwrap();
            let result = if shared {
                b.band(negated, compared, SPAN).unwrap()
            } else {
                negated
            };
            b.ret(Some(result), SPAN).unwrap();
        }
        canonicalize(&test, Bnot::get_canonicalization_patterns);
        test.function()
            .borrow()
            .as_operation()
            .verify()
            .expect("valid after canonicalization");
        test
    }

    #[test]
    fn bnot_of_a_comparison_is_the_inverted_comparison() {
        for predicate in CmpPredicate::ALL {
            let test = simplify_not_compare(predicate, false);
            let args = test.entry_block().borrow().arguments().to_vec();
            let op = defining_op(returned(&test)).expect("an operation result");
            let op = op.borrow();
            let cmpi = op.downcast_ref::<Cmpi>().expect("an arith.cmpi");
            assert_eq!(*cmpi.get_predicate(), predicate.inverted());
            assert_eq!(cmpi.lhs().as_value_ref(), args[0].borrow().as_value_ref());
            assert_eq!(cmpi.rhs().as_value_ref(), args[1].borrow().as_value_ref());
        }
    }

    #[test]
    fn bnot_of_a_shared_comparison_is_left_alone() {
        let test = simplify_not_compare(CmpPredicate::Slt, true);
        let op = defining_op(returned(&test)).unwrap();
        let op = op.borrow();
        let band = op.downcast_ref::<Band>().expect("an arith.band");
        let negated = defining_op(band.lhs().as_value_ref()).unwrap();
        assert!(negated.borrow().is::<Bnot>());
    }

    #[test]
    fn trunc_to_i1_of_a_widened_boolean_is_the_boolean() {
        for (extension, width) in [Extension::Zero, Extension::Sign]
            .into_iter()
            .flat_map(|extension| [2, 8, 64].map(|width| (extension, width)))
        {
            let mut test = Test::new("f", |cx| ([cx.i1()], [cx.i1()]));
            test.context().get_or_register_dialect::<ArithDialect>();
            let flag = test.entry_block().borrow().arguments()[0].borrow().as_value_ref();
            {
                let mut b = test.function_builder();
                let (i1, wide) =
                    (b.builder().context().i1(), b.builder().context().int_type(width));
                let widened = match extension {
                    Extension::Zero => b.zext(flag, wide, SPAN),
                    Extension::Sign => b.sext(flag, wide, SPAN),
                }
                .unwrap();
                let truncated = b.trunc(widened, i1, SPAN).unwrap();
                b.ret(Some(truncated), SPAN).unwrap();
            }
            assert!(canonicalize(&test, Trunc::get_canonicalization_patterns));
            assert_eq!(returned(&test), flag, "{extension:?} i{width}");
        }
    }

    /// An inline call chain attribute of one frame named `name`.
    fn chain(test: &Test, name: &'static str) -> runic_core::AttributeRef {
        use runic_core::dialects::debuginfo::attributes::{
            InlineCallChain, InlineCallChainAttr, InlineCallFrame,
        };
        test.context_rc()
            .create_attribute::<InlineCallChainAttr, _>(InlineCallChain::new(alloc::vec![
                InlineCallFrame {
                    name: name.into(),
                    linkage_name: None,
                    file: "source.rs".into(),
                    line: 1,
                    column: 1,
                    call_file: "source.rs".into(),
                    call_line: 2,
                    call_column: 1,
                }
            ]))
            .as_attribute_ref()
    }

    /// The first `T` in `test`'s function.
    fn find_op<T: Op>(test: &Test) -> Option<OperationRef> {
        let mut found = None;
        test.function()
            .borrow()
            .as_operation()
            .prewalk_all(|op: &runic_core::Operation| {
                if found.is_none() && op.is::<T>() {
                    found = Some(op.as_operation_ref());
                }
            });
        found
    }

    #[test]
    fn inline_metadata_survives_a_rewrite_that_creates_an_operation() -> Result<(), Report> {
        check_inline_metadata_on_created_operation(false)
    }

    #[test]
    fn standalone_pattern_application_inherits_inline_metadata() -> Result<(), Report> {
        check_inline_metadata_on_created_operation(true)
    }

    /// `eq (zext %b), 0` rewrites to `bnot %b`: the created `bnot` carries the inline call chain
    /// of the comparison it replaces, whether the rewrite runs in the greedy driver or in a
    /// standalone `PatternApplicator`.
    fn check_inline_metadata_on_created_operation(standalone: bool) -> Result<(), Report> {
        use runic_core::dialects::debuginfo::attributes::INLINE_CALL_CHAIN_ATTR_NAME;
        let mut test = Test::new("f", |cx| ([cx.i1()], [cx.i1()]));
        test.context().get_or_register_dialect::<ArithDialect>();
        let flag = test.entry_block().borrow().arguments()[0].borrow().as_value_ref();
        {
            let mut b = test.function_builder();
            let i32 = b.builder().context().i32();
            let widened = b.zext(flag, i32, SPAN)?;
            let zero = b.i32(0, SPAN);
            let compared = b.cmpi(CmpPredicate::Eq, widened, zero, SPAN)?;
            b.ret(Some(compared), SPAN)?;
        }
        let attr = chain(&test, "inline_compare");
        let mut compare = find_op::<Cmpi>(&test).unwrap();
        compare.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, attr);
        if standalone {
            use runic_core::patterns::{NoopRewriterListener, PatternApplicator, RewriterImpl};
            let context = test.context_rc();
            let mut patterns = RewritePatternSet::new(context.clone());
            Cmpi::get_canonicalization_patterns(&mut patterns, context.clone());
            let mut applicator =
                PatternApplicator::new(Rc::new(FrozenRewritePatternSet::new(patterns)));
            applicator.apply_cost_model(|pattern| *pattern.benefit());
            let mut rewriter = RewriterImpl::<NoopRewriterListener>::new(context);
            assert!(
                applicator
                    .match_and_rewrite(compare, &mut rewriter, |_| true, |_| {}, |_| Ok(()))
                    .is_ok()
            );
        } else {
            assert!(canonicalize(&test, Cmpi::get_canonicalization_patterns));
        }
        let mut created = 0;
        test.function()
            .borrow()
            .as_operation()
            .prewalk_all(|op: &runic_core::Operation| {
                if op.is::<Bnot>() {
                    assert_eq!(op.get_attribute(INLINE_CALL_CHAIN_ATTR_NAME), Some(attr));
                    created += 1;
                }
            });
        assert_eq!(created, 1, "the rewrite must create a bnot with the comparison's frame");
        Ok(())
    }

    /// `bnot (cmpi ..)` rewrites to a `cmpi` that belongs to the inline frame of the comparison it
    /// replaces, not to that of the `bnot`.
    #[test]
    fn simplify_not_compare_creates_the_comparison_in_the_comparisons_frame() -> Result<(), Report>
    {
        use runic_core::dialects::debuginfo::attributes::INLINE_CALL_CHAIN_ATTR_NAME;
        let mut test = Test::new("f", |cx| ([cx.i32(), cx.i32()], [cx.i1()]));
        test.context().get_or_register_dialect::<ArithDialect>();
        let args = test.entry_block().borrow().arguments().to_vec();
        let (lhs, rhs) = (args[0].borrow().as_value_ref(), args[1].borrow().as_value_ref());
        {
            let mut b = test.function_builder();
            let compared = b.cmpi(CmpPredicate::Slt, lhs, rhs, SPAN)?;
            let negated = b.bnot(compared, SPAN)?;
            b.ret(Some(negated), SPAN)?;
        }
        let compare_chain = chain(&test, "compare_frame");
        let not_chain = chain(&test, "not_frame");
        let mut compare = find_op::<Cmpi>(&test).unwrap();
        compare.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, compare_chain);
        let mut negation = find_op::<Bnot>(&test).unwrap();
        negation.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, not_chain);
        assert!(canonicalize(&test, Bnot::get_canonicalization_patterns));
        let created = defining_op(returned(&test)).unwrap();
        let created = created.borrow();
        let cmpi = created.downcast_ref::<Cmpi>().expect("an arith.cmpi");
        assert_eq!(*cmpi.get_predicate(), CmpPredicate::Sge);
        assert_eq!(created.get_attribute(INLINE_CALL_CHAIN_ATTR_NAME), Some(compare_chain));
        Ok(())
    }

    #[test]
    fn inline_metadata_survives_constant_folding() -> Result<(), Report> {
        use runic_core::dialects::debuginfo::attributes::{
            INLINE_CALL_CHAIN_ATTR_NAME, InlineCallChain, InlineCallChainAttr,
        };
        let mut test =
            Test::new("inline_metadata_survives_constant_folding", |cx| ([], [cx.i32()]));
        test.context().get_or_register_dialect::<ArithDialect>();
        let chain = test
            .context_rc()
            .create_attribute::<InlineCallChainAttr, _>(InlineCallChain::default())
            .as_attribute_ref();
        {
            let mut builder = test.function_builder();
            let i32: Type = builder.builder().context().i32();
            let input = builder.i64(33, SPAN);
            let truncated = builder.trunc(input, i32, SPAN)?;
            truncated
                .borrow()
                .get_defining_op()
                .unwrap()
                .borrow_mut()
                .set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, chain);
            builder.ret(Some(truncated), SPAN)?;
        }
        assert!(canonicalize(&test, Trunc::get_canonicalization_patterns));
        let constant = defining_op(returned(&test)).unwrap();
        assert!(constant.borrow().implements::<dyn ConstantLike>());
        assert_eq!(constant.borrow().get_attribute(INLINE_CALL_CHAIN_ATTR_NAME), Some(chain));
        assert_eq!(returned(&test).borrow().ty(), test.context().i32());
        Ok(())
    }

    #[test]
    fn the_folder_replaces_an_addition_of_constants_with_a_constant() -> Result<(), Report> {
        let mut test = Test::new("f", |cx| ([], [cx.i32()]));
        test.context().get_or_register_dialect::<ArithDialect>();
        {
            let mut builder = test.function_builder();
            let (lhs, rhs) = (builder.i32(1, SPAN), builder.i32(2, SPAN));
            let sum = builder.add(lhs, rhs, SPAN)?;
            builder.ret(Some(sum), SPAN)?;
        }
        assert!(canonicalize(&test, |_, _| {}));
        test.function().borrow().as_operation().verify().expect("valid after folding");

        let function = test.function();
        let function = function.borrow();
        let body = function.body();
        let entry = body.entry();
        let ops = entry
            .body()
            .iter()
            .map(|op| op.as_operation_ref())
            .collect::<alloc::vec::Vec<_>>();
        let [constant, ret] = ops[..] else {
            panic!("expected a constant and a return: {}", function.as_operation());
        };
        let constant = constant.borrow();
        let constant = constant.downcast_ref::<Constant>().expect("an arith.constant");
        assert_eq!(*constant.get_value(), Int::from(3i32));
        assert_eq!(constant.result().ty(), test.context().i32());
        assert!(ret.borrow().is::<Ret>());
        assert_eq!(returned(&test), constant.result().as_value_ref());
        assert!(format!("{}", function.as_operation()).contains("arith.constant 3 : i32"));
        Ok(())
    }
}
