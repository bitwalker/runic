use alloc::{format, rc::Rc, vec::Vec};
use core::cell::RefCell;

use smallvec::SmallVec;

use super::TypeConverter;
use crate::{
    BlockRef, BuildableOp, Builder, BuilderExt, Context, Listener, ListenerType, OperationRef,
    ProgramPoint, RegionRef, Report, SourceSpan, Spanned, Type, UnsafeIntrusiveEntityRef, ValueRef,
    dialects::builtin::BuiltinOpBuilder,
    patterns::{PatternRewriter, Rewriter, RewriterListener},
};

/// A recorded reason that a conversion pattern did not match.
pub struct MatchFailure {
    op: OperationRef,
    reason: Report,
}

impl MatchFailure {
    /// Return the operation that failed to match a conversion pattern.
    #[inline]
    pub const fn op(&self) -> OperationRef {
        self.op
    }

    /// Return the diagnostic reason recorded by the pattern.
    #[inline]
    pub const fn reason(&self) -> &Report {
        &self.reason
    }
}

/// Mutations observed while applying a conversion pattern.
///
/// The conversion driver uses this summary to legalize newly inserted or modified operations after
/// a pattern succeeds. The lists contain each operation at most once, while `mutation_count`
/// counts all listener notifications.
#[derive(Default)]
pub struct TrackedMutations {
    inserted_ops: Vec<OperationRef>,
    modified_ops: Vec<OperationRef>,
    replaced_ops: Vec<OperationRef>,
    erased_ops: Vec<OperationRef>,
    materialized_ops: Vec<OperationRef>,
    block_mutations: usize,
    mutation_count: usize,
}

impl TrackedMutations {
    /// Return operations inserted while the pattern ran.
    #[inline]
    pub fn inserted_ops(&self) -> &[OperationRef] {
        &self.inserted_ops
    }

    /// Return operations modified while the pattern ran.
    #[inline]
    pub fn modified_ops(&self) -> &[OperationRef] {
        &self.modified_ops
    }

    /// Return operations replaced while the pattern ran.
    #[inline]
    pub fn replaced_ops(&self) -> &[OperationRef] {
        &self.replaced_ops
    }

    /// Return operations erased while the pattern ran.
    #[inline]
    pub fn erased_ops(&self) -> &[OperationRef] {
        &self.erased_ops
    }

    /// Return framework-owned materialization operations created while the pattern ran.
    #[inline]
    pub fn materialized_ops(&self) -> &[OperationRef] {
        &self.materialized_ops
    }

    /// Return the number of block insertion/erasure notifications observed.
    #[inline]
    pub const fn block_mutations(&self) -> usize {
        self.block_mutations
    }

    /// Return the total number of mutation notifications observed.
    #[inline]
    pub const fn mutation_count(&self) -> usize {
        self.mutation_count
    }

    /// Return true when any IR mutation notification was observed.
    #[inline]
    pub const fn has_mutations(&self) -> bool {
        self.mutation_count > 0
    }
}

struct ConversionTrackingListener {
    mutations: RefCell<TrackedMutations>,
}

impl ConversionTrackingListener {
    fn new() -> Self {
        Self {
            mutations: RefCell::new(TrackedMutations::default()),
        }
    }

    fn mutation_count(&self) -> usize {
        self.mutations.borrow().mutation_count()
    }

    fn take_mutations(&self) -> TrackedMutations {
        core::mem::take(&mut *self.mutations.borrow_mut())
    }

    fn record_inserted_op(&self, op: OperationRef) {
        let mut mutations = self.mutations.borrow_mut();
        push_unique(&mut mutations.inserted_ops, op);
        mutations.mutation_count += 1;
    }

    fn record_modified_op(&self, op: OperationRef) {
        let mut mutations = self.mutations.borrow_mut();
        push_unique(&mut mutations.modified_ops, op);
        mutations.mutation_count += 1;
    }

    fn record_replaced_op(&self, op: OperationRef) {
        let mut mutations = self.mutations.borrow_mut();
        push_unique(&mut mutations.replaced_ops, op);
        mutations.mutation_count += 1;
    }

    fn record_erased_op(&self, op: OperationRef) {
        let mut mutations = self.mutations.borrow_mut();
        push_unique(&mut mutations.erased_ops, op);
        mutations.mutation_count += 1;
    }

    fn record_materialized_op(&self, op: OperationRef) {
        let mut mutations = self.mutations.borrow_mut();
        push_unique(&mut mutations.materialized_ops, op);
    }

    fn record_block_mutation(&self) {
        let mut mutations = self.mutations.borrow_mut();
        mutations.block_mutations += 1;
        mutations.mutation_count += 1;
    }
}

impl Listener for ConversionTrackingListener {
    fn kind(&self) -> ListenerType {
        ListenerType::Rewriter
    }

    fn notify_operation_inserted(&self, op: OperationRef, _prev: ProgramPoint) {
        self.record_inserted_op(op);
    }

    fn notify_block_inserted(
        &self,
        _block: BlockRef,
        _prev: Option<RegionRef>,
        _ip: Option<BlockRef>,
    ) {
        self.record_block_mutation();
    }
}

impl RewriterListener for ConversionTrackingListener {
    fn notify_block_erased(&self, _block: BlockRef) {
        self.record_block_mutation();
    }

    fn notify_operation_modified(&self, op: OperationRef) {
        self.record_modified_op(op);
    }

    fn notify_operation_replaced(&self, op: OperationRef, _replacement: OperationRef) {
        self.record_replaced_op(op);
    }

    fn notify_operation_replaced_with_values(
        &self,
        op: OperationRef,
        _replacement: &[Option<ValueRef>],
    ) {
        self.record_replaced_op(op);
    }

    fn notify_operation_erased(&self, op: OperationRef) {
        self.record_erased_op(op);
    }
}

fn push_unique(ops: &mut Vec<OperationRef>, op: OperationRef) {
    if !ops.contains(&op) {
        ops.push(op);
    }
}

/// Rewriter used by conversion patterns.
///
/// This wraps the normal [`PatternRewriter`] with mutation tracking and the replacement of a result
/// by a range of values. Conversion patterns should perform all IR changes through this
/// rewriter so the driver can validate the pattern contract and legalize generated operations.
pub struct ConversionPatternRewriter {
    inner: PatternRewriter<Rc<ConversionTrackingListener>>,
    tracking: Rc<ConversionTrackingListener>,
    type_converter: Option<TypeConverter>,
    match_failures: Vec<MatchFailure>,
}

impl ConversionPatternRewriter {
    /// Create a conversion rewriter without type-conversion support.
    #[inline]
    pub fn new(context: Rc<Context>) -> Self {
        Self::new_with_type_converter(context, None)
    }

    /// Create a conversion rewriter with an optional type converter.
    ///
    /// The type converter says what types each result may be replaced by
    /// ([Self::replace_op_with_multiple]).
    #[inline]
    pub fn new_with_type_converter(
        context: Rc<Context>,
        type_converter: Option<TypeConverter>,
    ) -> Self {
        let tracking = Rc::new(ConversionTrackingListener::new());
        let inner = PatternRewriter::new_with_listener(context, Rc::clone(&tracking));
        Self {
            inner,
            tracking,
            type_converter,
            match_failures: Vec::new(),
        }
    }

    /// Set the debug metadata inherited during this conversion attempt.
    pub(super) fn inherit_inline_call_chain(&mut self, op: OperationRef) {
        let chain = op
            .borrow()
            .get_attribute(crate::dialects::debuginfo::attributes::INLINE_CALL_CHAIN_ATTR_NAME);
        self.inner.replace_inline_call_chain(chain);
    }

    /// Set the insertion point before `op`.
    #[inline]
    pub fn set_insertion_point_before(&mut self, op: OperationRef) {
        self.inner.set_insertion_point_before(op);
    }

    /// Set the insertion point after `op`.
    #[inline]
    pub fn set_insertion_point_after(&mut self, op: OperationRef) {
        self.inner.set_insertion_point_after(op);
    }

    /// Create a new operation at the current insertion point.
    pub fn create_op<T, Args>(
        &mut self,
        span: SourceSpan,
        args: Args,
    ) -> Result<UnsafeIntrusiveEntityRef<T>, Report>
    where
        Args: core::marker::Tuple,
        T: BuildableOp<Args>,
    {
        let builder = self.inner.create::<T, Args>(span);
        core::ops::FnOnce::call_once(builder, args)
    }

    /// Insert a `builtin.unrealized_conversion_cast` of `inputs` to values of the types
    /// `results` at the current insertion point, mark it as the driver's, and return its results.
    pub(super) fn insert_cast(
        &mut self,
        inputs: impl IntoIterator<Item = ValueRef>,
        results: impl IntoIterator<Item = Type>,
        span: SourceSpan,
    ) -> Result<SmallVec<[ValueRef; 2]>, Report> {
        let cast =
            BuiltinOpBuilder::unrealized_conversion_cast(&mut *self.inner, inputs, results, span)?;
        self.mark_materialization_op(cast.as_operation_ref());
        let outputs = cast.borrow().outputs().iter().map(|output| *output as ValueRef).collect();
        Ok(outputs)
    }

    /// Mark `op` as a framework-owned materialization operation.
    #[inline]
    pub fn mark_materialization_op(&mut self, op: OperationRef) {
        self.tracking.record_materialized_op(op);
    }

    /// Erase framework materializations that are still unused.
    ///
    /// Drivers call this after a pattern fails to match so temporary casts inserted while building
    /// converted operands do not remain in the IR.
    pub fn erase_unused_materializations(&mut self) -> Result<(), Report> {
        let materialized_ops = self.tracking.mutations.borrow().materialized_ops.clone();
        for op in materialized_ops.into_iter().rev() {
            if op.parent().is_some() && !op.borrow().is_used() {
                self.inner.erase_op(op);
            }
        }
        Ok(())
    }

    /// Replace `op` by one range of values per result of `op`, and erase it.
    ///
    /// A range that is one value of the result's own type replaces the result. Any other range
    /// must have exactly the types the pattern's type converter says the result becomes; when the
    /// result still has users, they are given a many-to-one `builtin.unrealized_conversion_cast`
    /// of the range back to the result's type, inserted before `op`. A range of any other types,
    /// or the wrong number of ranges, is an error raised before anything is changed.
    pub fn replace_op_with_multiple(
        &mut self,
        op: OperationRef,
        values: &[&[ValueRef]],
    ) -> Result<(), Report> {
        let results = op
            .borrow()
            .results()
            .all()
            .iter()
            .map(|result| result.borrow().as_value_ref())
            .collect::<SmallVec<[ValueRef; 2]>>();
        if results.len() != values.len() {
            return Err(Report::msg(format!(
                "operation '{}' has {} results, but it is replaced by {} ranges of values",
                op.borrow().name(),
                results.len(),
                values.len()
            )));
        }

        // Check every range before changing anything.
        let context = self.inner.context_rc();
        let mut casts = SmallVec::<[bool; 2]>::new();
        for (index, (result, range)) in results.iter().zip(values).enumerate() {
            let result_ty = result.borrow().ty();
            if let [value] = range
                && value.borrow().ty() == result_ty
            {
                casts.push(false);
                continue;
            }
            let range_types = types_of(range);
            let Some(converter) = self.type_converter.as_ref() else {
                return Err(Report::msg(format!(
                        "result {index} of '{}' has type '{result_ty}', but it is replaced by \
                         values                      of types {}, and the pattern has no type \
                         converter",
                        op.borrow().name(),
                        type_list(&range_types)
                    )));
            };
            let expected = converter.convert_value(&context, *result);
            if range_types != expected {
                return Err(Report::msg(format!(
                            "result {index} of '{}' has type '{result_ty}', which the type \
                             converter                      converts to {}, but it is replaced by \
                             values of types {}",
                            op.borrow().name(),
                            type_list(&expected),
                            type_list(&range_types)
                        )));
            }
            casts.push(true);
        }

        let span = op.borrow().span();
        let mut replacements = SmallVec::<[Option<ValueRef>; 2]>::new();
        for ((result, range), cast) in results.iter().zip(values).zip(casts) {
            if !cast {
                replacements.push(Some(range[0]));
                continue;
            }
            if !result.borrow().is_used() {
                replacements.push(None);
                continue;
            }
            self.set_insertion_point_before(op);
            let result_ty = result.borrow().ty();
            let outputs = self.insert_cast(range.iter().copied(), [result_ty], span)?;
            replacements.push(Some(outputs[0]));
        }
        self.inner.replace_op_with_values(op, &replacements);
        Ok(())
    }

    /// Replace `op` by one value per result of `op`, and erase it.
    ///
    /// This is [Self::replace_op_with_multiple] with a range of one value per result.
    pub fn replace_op(&mut self, op: OperationRef, values: &[ValueRef]) -> Result<(), Report> {
        let ranges = values.iter().map(core::slice::from_ref).collect::<SmallVec<[_; 2]>>();
        self.replace_op_with_multiple(op, &ranges)
    }

    /// Create a new operation before `op`, then replace `op` with the new operation's results.
    ///
    /// This is a convenience for the common single-operation legalization case. It uses
    /// [`Self::replace_op`] for result replacement, so the same casts are inserted for users.
    pub fn replace_op_with_new_op<T, Args>(
        &mut self,
        op: OperationRef,
        span: SourceSpan,
        args: Args,
    ) -> Result<UnsafeIntrusiveEntityRef<T>, Report>
    where
        Args: core::marker::Tuple,
        T: BuildableOp<Args>,
    {
        self.set_insertion_point_before(op);
        let replacement = self.create_op::<T, Args>(span, args)?;
        let replacement_values = replacement
            .borrow()
            .results()
            .all()
            .iter()
            .map(|result| result.borrow().as_value_ref())
            .collect::<SmallVec<[ValueRef; 2]>>();
        self.replace_op(op, &replacement_values)?;
        Ok(replacement)
    }

    /// Erase an operation with no remaining uses.
    ///
    /// Callers are responsible for ensuring the operation can be erased without leaving dangling
    /// uses or invalid control-flow edges.
    #[inline]
    pub fn erase_op(&mut self, op: OperationRef) -> Result<(), Report> {
        self.inner.erase_op(op);
        Ok(())
    }

    /// Record a non-fatal reason why the current pattern did not match `op`.
    ///
    /// A pattern can call this before returning `Ok(false)`. The driver includes collected match
    /// failures in diagnostics when no candidate pattern can legalize an operation.
    pub fn notify_match_failure(&mut self, op: OperationRef, reason: Report) {
        self.match_failures.push(MatchFailure { op, reason });
    }

    /// Return match failures recorded since the last clear/take.
    #[inline]
    pub fn match_failures(&self) -> &[MatchFailure] {
        &self.match_failures
    }

    /// Remove all recorded match failures.
    #[inline]
    pub fn clear_match_failures(&mut self) {
        self.match_failures.clear();
    }

    /// Take and clear all recorded match failures.
    pub fn take_match_failures(&mut self) -> Vec<MatchFailure> {
        core::mem::take(&mut self.match_failures)
    }

    /// Return the total number of mutation notifications observed by this rewriter.
    #[inline]
    pub fn mutation_count(&self) -> usize {
        self.tracking.mutation_count()
    }

    /// Take and clear tracked mutation state.
    pub fn take_tracked_mutations(&mut self) -> TrackedMutations {
        self.tracking.take_mutations()
    }
}

/// The types of `values`, in order.
pub(super) fn types_of(values: &[ValueRef]) -> SmallVec<[Type; 2]> {
    values.iter().map(|value| value.borrow().ty()).collect()
}

/// `types` written as a parenthesized list: `(i64, i64)`, `(i32)` or `()`.
pub(super) fn type_list(types: &[Type]) -> alloc::string::String {
    let mut list = alloc::string::String::from("(");
    for (index, ty) in types.iter().enumerate() {
        if index > 0 {
            list.push_str(", ");
        }
        list.push_str(&format!("{ty}"));
    }
    list.push(')');
    list
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use crate::{
        Context, OpRegistration, Report, conversion::ConversionPatternRewriter,
        dialects::test::Constant,
    };

    #[test]
    fn records_match_failures() {
        let context = Rc::new(Context::default());
        let op = context
            .get_or_register_dialect::<<Constant as OpRegistration>::Dialect>()
            .expect_registered_name::<Constant>()
            .alloc_default(context.clone());
        let mut rewriter = ConversionPatternRewriter::new(context);

        rewriter.notify_match_failure(op, Report::msg("no match"));

        assert_eq!(rewriter.match_failures().len(), 1);
        rewriter.clear_match_failures();
        assert!(rewriter.match_failures().is_empty());
    }
}
