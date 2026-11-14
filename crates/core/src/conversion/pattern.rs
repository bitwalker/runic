use super::{ConversionPatternRewriter, TypeConverter};
use crate::{
    Op, OperationRef, Report, SmallVec, UnsafeIntrusiveEntityRef, ValueRef, patterns::Pattern,
};

/// The values each operand of the operation being converted became.
///
/// An operand whose type the pattern's type converter leaves alone is itself; one whose type
/// converts to several types became several values, and one whose type is dropped became none. A
/// pattern without a type converter sees each operand as itself. The operands are numbered as the
/// operation numbers them, and the operation's operand groups remain visible through
/// [Self::group].
#[derive(Copy, Clone)]
pub struct ConvertedOperands<'a> {
    ranges: &'a [SmallVec<[ValueRef; 2]>],
    group_sizes: &'a [usize],
}

impl<'a> ConvertedOperands<'a> {
    /// A view of `ranges`, one per operand, whose operands come in groups of `group_sizes`.
    ///
    /// Panics unless the group sizes add up to the number of ranges.
    pub fn new(ranges: &'a [SmallVec<[ValueRef; 2]>], group_sizes: &'a [usize]) -> Self {
        assert_eq!(
            group_sizes.iter().sum::<usize>(),
            ranges.len(),
            "the operand groups must cover the operands exactly"
        );
        Self {
            ranges,
            group_sizes,
        }
    }

    /// The number of operands of the operation.
    #[inline]
    pub const fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the operation has no operands.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The values operand `operand` became: several, one or none.
    ///
    /// Panics if the operation has no such operand.
    #[inline]
    pub fn get(&self, operand: usize) -> &'a [ValueRef] {
        self.ranges[operand].as_slice()
    }

    /// The value operand `operand` became, when it became exactly one.
    ///
    /// Panics if the operation has no such operand.
    #[inline]
    pub fn one(&self, operand: usize) -> Option<ValueRef> {
        match self.get(operand) {
            [value] => Some(*value),
            _ => None,
        }
    }

    /// One value per operand, when every operand became exactly one value. A pattern written for
    /// one value per operand asks for this and does not match otherwise.
    pub fn all_one(&self) -> Option<SmallVec<[ValueRef; 4]>> {
        (0..self.len()).map(|operand| self.one(operand)).collect()
    }

    /// The ranges, one per operand, in operand order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &'a [ValueRef]> + 'a {
        self.ranges.iter().map(|range| range.as_slice())
    }

    /// The number of operand groups of the operation.
    #[inline]
    pub const fn num_groups(&self) -> usize {
        self.group_sizes.len()
    }

    /// The ranges of the operands of group `group`, one per operand.
    ///
    /// Panics if the operation has no such group.
    pub fn group(&self, group: usize) -> impl Iterator<Item = &'a [ValueRef]> + 'a {
        let start = self.group_sizes[..group].iter().sum::<usize>();
        let end = start + self.group_sizes[group];
        self.ranges[start..end].iter().map(|range| range.as_slice())
    }

    /// The values of every operand of group `group`, in order.
    ///
    /// Panics if the operation has no such group.
    pub fn group_flattened(&self, group: usize) -> SmallVec<[ValueRef; 4]> {
        self.group(group).flatten().copied().collect()
    }
}

/// A rewrite pattern that participates in target-driven dialect conversion.
pub trait ConversionPattern: Pattern {
    /// The type converter that says what this pattern's operands and results become.
    ///
    /// The driver gives the pattern each operand as the values the converter says it becomes, and
    /// checks the values the pattern replaces each result by against what the converter says the
    /// result becomes. A pattern without a converter (the default) sees each operand as itself and
    /// replaces each result by one value of its own type.
    fn type_converter(&self) -> Option<&TypeConverter> {
        None
    }

    /// Try to rewrite `op`, whose operands became `operands`, through `rewriter`.
    ///
    /// Return `Ok(true)` after successfully mutating IR. Return `Ok(false)` when the pattern does
    /// not match; in that case the pattern must not mutate IR, but may call
    /// [`ConversionPatternRewriter::notify_match_failure`] to leave a diagnostic breadcrumb. Return
    /// `Err` for a fatal failure that should abort conversion. The full-conversion driver checks
    /// this mutation contract because it does not provide rollback.
    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report>;
}

/// Typed adapter trait for conversion patterns rooted on a concrete operation.
pub trait OpConversionPattern<T: Op>: ConversionPattern {
    /// Try to rewrite a statically typed root operation.
    ///
    /// This has the same mutation and result contract as
    /// [`ConversionPattern::match_and_rewrite`], but lets pattern implementations work with their
    /// concrete operation type after the caller has performed the root-type check.
    fn match_and_rewrite_typed(
        &self,
        op: UnsafeIntrusiveEntityRef<T>,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report>;
}
