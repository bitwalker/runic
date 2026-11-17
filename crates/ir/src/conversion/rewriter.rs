use alloc::{format, rc::Rc, string::String, vec::Vec};
use core::cell::RefCell;

use smallvec::SmallVec;

use super::{InputMapping, SignatureConversion, TypeConverter};
use crate::{
    BlockRef, BuildableOp, Builder, BuilderExt, Context, FxHashMap, FxHashSet, Listener,
    ListenerType, Operation, OperationRef, ProgramPoint, RegionRef, Report, SourceSpan, Spanned,
    Type, UnsafeIntrusiveEntityRef, Value, ValueRef,
    dialects::builtin::BuiltinOpBuilder,
    patterns::{PatternRewriter, Rewriter, RewriterExt, RewriterListener},
    traits::Terminator,
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
/// The conversion driver uses this summary to legalize the operations a pattern inserted, changed
/// in place or moved after it succeeds. The lists contain each operation at most once, while
/// `mutation_count` counts all listener notifications.
#[derive(Default)]
pub struct TrackedMutations {
    inserted_ops: Vec<OperationRef>,
    modified_ops: Vec<OperationRef>,
    replaced_ops: Vec<OperationRef>,
    erased_ops: Vec<OperationRef>,
    materialized_ops: Vec<OperationRef>,
    in_place_ops: Vec<OperationRef>,
    moved_ops: Vec<OperationRef>,
    emptied_ops: Vec<OperationRef>,
    block_mutations: usize,
    mutation_count: usize,
}

impl TrackedMutations {
    /// Return operations inserted while the pattern ran.
    #[inline]
    pub fn inserted_ops(&self) -> &[OperationRef] {
        &self.inserted_ops
    }

    /// Return operations modified while the pattern ran: those changed in place, and the users
    /// whose operands a replacement redirected.
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

    /// Return the operations the pattern changed in place: those given to
    /// [ConversionPatternRewriter::modify_op_in_place], and the parent operations of the blocks
    /// given to [ConversionPatternRewriter::apply_signature_conversion]. Unlike
    /// [Self::modified_ops], it leaves out the users a replacement redirected.
    #[inline]
    pub fn in_place_ops(&self) -> &[OperationRef] {
        &self.in_place_ops
    }

    /// Return the operations the pattern moved out of a block it inlined
    /// ([ConversionPatternRewriter::inline_block_before],
    /// [ConversionPatternRewriter::inline_block_at_end]), in the order they were moved. An
    /// operation the pattern created and then moved is an inserted operation
    /// ([Self::inserted_ops]), not a moved one. The operations nested in a moved operation moved
    /// with it and are not listed.
    #[inline]
    pub fn moved_ops(&self) -> &[OperationRef] {
        &self.moved_ops
    }

    /// Return the operations a block of which the pattern inlined, in the order the blocks were
    /// inlined: each has a region the pattern emptied, since an inlined block is the only block
    /// of its region.
    #[inline]
    pub fn emptied_ops(&self) -> &[OperationRef] {
        &self.emptied_ops
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

/// Which way a cast the driver inserted relates values.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum CastKind {
    /// New values to one value of the old type, for a user that expects the old value.
    Source,
    /// One old value to values of the new types, for a pattern that expects the new ones.
    Target,
}

/// A cast the driver inserted: its kind and the type converter of the pattern that caused it.
pub(super) struct DriverCast {
    pub(super) kind: CastKind,
    pub(super) converter: Rc<TypeConverter>,
}

/// The casts the driver inserted during one conversion and that are still in the IR, shared by
/// the rewriters of the conversion. A cast that is erased through a rewriter leaves the table.
pub(super) type DriverCasts = Rc<RefCell<FxHashMap<OperationRef, DriverCast>>>;

/// The blocks whose arguments were converted during one conversion, shared by the rewriters of
/// the conversion. A block is converted at most once per conversion; an erased block leaves the
/// set.
pub(super) type ConvertedBlocks = Rc<RefCell<FxHashSet<BlockRef>>>;

/// What the rewriters of one conversion share: the casts the driver inserted, the blocks whose
/// arguments were converted, and the root of the conversion, which cannot be erased.
#[derive(Clone, Default)]
pub(super) struct ConversionState {
    pub(super) casts: DriverCasts,
    pub(super) converted_blocks: ConvertedBlocks,
    pub(super) root: Option<OperationRef>,
}

struct ConversionTrackingListener {
    mutations: RefCell<TrackedMutations>,
    state: ConversionState,
    /// The operations of the block being inlined: their insertion is a move.
    moving: RefCell<FxHashSet<OperationRef>>,
}

impl ConversionTrackingListener {
    fn new(state: ConversionState) -> Self {
        Self {
            mutations: RefCell::new(TrackedMutations::default()),
            state,
            moving: RefCell::new(FxHashSet::default()),
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

    /// Records the move of `op` out of a block being inlined; an operation the pattern created is
    /// left among the inserted operations.
    fn record_moved_op(&self, op: OperationRef) {
        let mut mutations = self.mutations.borrow_mut();
        if !mutations.inserted_ops.contains(&op) {
            push_unique(&mut mutations.moved_ops, op);
        }
        mutations.mutation_count += 1;
    }

    /// Records that a block of `op` was inlined, which left a region of `op` empty.
    fn record_emptied_op(&self, op: OperationRef) {
        let mut mutations = self.mutations.borrow_mut();
        push_unique(&mut mutations.emptied_ops, op);
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

    fn record_in_place_op(&self, op: OperationRef) {
        let mut mutations = self.mutations.borrow_mut();
        push_unique(&mut mutations.in_place_ops, op);
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
        if self.moving.borrow().contains(&op) {
            self.record_moved_op(op);
        } else {
            self.record_inserted_op(op);
        }
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
    fn notify_block_erased(&self, block: BlockRef) {
        self.state.converted_blocks.borrow_mut().remove(&block);
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
        self.state.casts.borrow_mut().remove(&op);
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
    type_converter: Option<Rc<TypeConverter>>,
    state: ConversionState,
    /// The operation the driver gave the pattern, whose blocks (and those nested in it) the
    /// pattern may inline.
    matched: Option<OperationRef>,
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
        Self::new_for_conversion(context, type_converter.map(Rc::new), ConversionState::default())
    }

    /// Create a conversion rewriter that records the casts it inserts, each under the one shared
    /// handle of `type_converter`, and the blocks it converts, in `state`.
    pub(super) fn new_for_conversion(
        context: Rc<Context>,
        type_converter: Option<Rc<TypeConverter>>,
        state: ConversionState,
    ) -> Self {
        let tracking = Rc::new(ConversionTrackingListener::new(state.clone()));
        let inner = PatternRewriter::new_with_listener(context, Rc::clone(&tracking));
        Self {
            inner,
            tracking,
            type_converter,
            state,
            matched: None,
            match_failures: Vec::new(),
        }
    }

    /// Set the operation the pattern is given, whose blocks the pattern may inline.
    pub(super) fn set_matched_op(&mut self, op: OperationRef) {
        self.matched = Some(op);
    }

    /// Set the debug metadata inherited during this conversion attempt.
    pub(super) fn inherit_inline_call_chain(&mut self, op: OperationRef) {
        let chain = op
            .borrow()
            .get_attribute(crate::dialects::debuginfo::attributes::INLINE_CALL_CHAIN_ATTR_NAME);
        self.inner.replace_inline_call_chain(chain);
    }

    /// The context of the IR being converted.
    #[inline]
    pub fn context(&self) -> &Context {
        self.inner.context()
    }

    /// The root of the conversion: the operation the conversion was applied to, under which every
    /// operation it converts is. `None` for a rewriter not given to a pattern by the driver (one
    /// made with [Self::new_with_type_converter], or the one a materialization callback is given).
    #[inline]
    pub fn root(&self) -> Option<OperationRef> {
        self.state.root
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

    /// Set the insertion point to the start of `block`.
    #[inline]
    pub fn set_insertion_point_to_start(&mut self, block: BlockRef) {
        self.inner.set_insertion_point_to_start(block);
    }

    /// Set the insertion point to the end of `block`.
    #[inline]
    pub fn set_insertion_point_to_end(&mut self, block: BlockRef) {
        self.inner.set_insertion_point_to_end(block);
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
    /// `results` at the current insertion point, record it as the driver's cast of `kind` under
    /// this rewriter's type converter, and return the cast and its results.
    ///
    /// The driver inserts a cast only for a pattern with a type converter.
    pub(super) fn insert_cast(
        &mut self,
        kind: CastKind,
        inputs: impl IntoIterator<Item = ValueRef>,
        results: impl IntoIterator<Item = Type>,
        span: SourceSpan,
    ) -> Result<(OperationRef, SmallVec<[ValueRef; 2]>), Report> {
        let converter = self.type_converter.as_ref().map(Rc::clone).ok_or_else(|| {
            Report::msg("the driver inserts a cast only for a pattern with a type converter")
        })?;
        let cast =
            BuiltinOpBuilder::unrealized_conversion_cast(&mut *self.inner, inputs, results, span)?;
        let cast_op = cast.as_operation_ref();
        self.mark_materialization_op(cast_op);
        self.state.casts.borrow_mut().insert(cast_op, DriverCast { kind, converter });
        let outputs = cast.borrow().outputs().iter().map(|output| *output as ValueRef).collect();
        Ok((cast_op, outputs))
    }

    /// Mark `op` as a framework-owned materialization operation.
    #[inline]
    pub fn mark_materialization_op(&mut self, op: OperationRef) {
        self.tracking.record_materialized_op(op);
    }

    /// Replace the results of `cast` by `values` and erase it.
    pub(super) fn replace_cast(&mut self, cast: OperationRef, values: &[ValueRef]) {
        let values = values.iter().copied().map(Some).collect::<SmallVec<[_; 2]>>();
        self.inner.replace_op_with_values(cast, &values);
    }

    /// Erase framework materializations that are still unused.
    ///
    /// Drivers call this when an attempt to convert an operation ends before its pattern changed
    /// anything, so the casts inserted while building converted operands do not remain in the IR.
    /// A cast the driver inserted that is erased here leaves the table of the driver's casts.
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
    /// of the range back to the result's type. The cast is placed after the value of the range
    /// that is defined last, when that definition is in `op`'s block after `op` (a replacement
    /// built after `op`); otherwise before `op`.
    ///
    /// The values of a range must dominate the users of the result they replace. When the cast
    /// goes after a definition in `op`'s block, a user of the result in that block (or nested in an
    /// operation of it) that is not after that definition is an error; nothing else is checked
    /// (the verifier has no dominance check), so a range defined elsewhere that does not dominate
    /// a user leaves IR that reads a value before its definition.
    ///
    /// An error is raised before anything is changed for the wrong number of ranges, an `op` that
    /// is the root of the conversion, a range of any other types, a range that holds a result of
    /// `op`, a range that holds a value defined inside `op` (in one of its regions, at any depth),
    /// and a user that comes before the definition the cast must follow.
    ///
    /// Afterwards the insertion point is where it was before the call; when it was at `op`, it is
    /// at the position `op` occupied (before the operation that followed it, or at the end of its
    /// block).
    pub fn replace_op_with_multiple(
        &mut self,
        op: OperationRef,
        values: &[&[ValueRef]],
    ) -> Result<(), Report> {
        self.check_not_root(op, "replaced")?;
        check_no_branch_from_outside(op, "replaced")?;
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
        // Per result: `None` when the range replaces it directly, else where its cast goes.
        let mut casts = SmallVec::<[Option<ProgramPoint>; 2]>::new();
        for (index, (result, range)) in results.iter().zip(values).enumerate() {
            if let Some(own) = range.iter().find(|value| results.contains(*value)) {
                return Err(Report::msg(format!(
                    "result {index} of '{}' is replaced by a range that holds '{}', a result of \
                     the operation being replaced",
                    op.borrow().name(),
                    own.borrow()
                )));
            }
            if let Some(inner) = range.iter().find(|value| is_defined_inside(**value, op)) {
                return Err(Report::msg(format!(
                    "result {index} of '{}' is replaced by a range that holds '{}', a value \
                     defined inside the operation being replaced, which is erased with it",
                    op.borrow().name(),
                    inner.borrow()
                )));
            }
            let result_ty = result.borrow().ty();
            if let [value] = range
                && value.borrow().ty() == result_ty
            {
                casts.push(None);
                continue;
            }
            let range_types = types_of(range);
            let Some(converter) = self.type_converter.as_ref() else {
                return Err(Report::msg(format!(
                    "result {index} of '{}' has type '{result_ty}', but it is replaced by values \
                     of types {}, and the pattern has no type converter",
                    op.borrow().name(),
                    type_list(&range_types)
                )));
            };
            let expected = converter.convert_value(&context, *result);
            if range_types != expected {
                return Err(Report::msg(format!(
                    "result {index} of '{}' has type '{result_ty}', which the type converter \
                     converts to {}, but it is replaced by values of types {}",
                    op.borrow().name(),
                    type_list(&expected),
                    type_list(&range_types)
                )));
            }
            let last = last_definition_after(op, range);
            if let Some(last) = last
                && let Some(user) = user_not_after(*result, last)
            {
                return Err(Report::msg(format!(
                    "result {index} of '{}' is replaced by a range defined after it, the last \
                     value by '{}', but its user '{}' comes before that definition, where the \
                     range is not yet defined",
                    op.borrow().name(),
                    last.borrow().name(),
                    user.borrow().name()
                )));
            }
            casts.push(Some(last.map_or_else(|| ProgramPoint::before(op), ProgramPoint::after)));
        }

        // Each cast is inserted before `op`, or after its range when the range is defined after
        // `op`; afterwards the insertion point is where it was, or, when it was at `op`, at the
        // position `op` occupied.
        let saved = *self.inner.insertion_point();
        let restore = match saved {
            ProgramPoint::Op { op: anchor, .. } if anchor == op => match op.next() {
                Some(next) => ProgramPoint::before(next),
                None => op.parent().map_or(saved, ProgramPoint::at_end_of),
            },
            _ => saved,
        };
        let span = op.borrow().span();
        let mut replacements = SmallVec::<[Option<ValueRef>; 2]>::new();
        for ((result, range), cast) in results.iter().zip(values).zip(casts) {
            let Some(point) = cast else {
                replacements.push(Some(range[0]));
                continue;
            };
            if !result.borrow().is_used() {
                replacements.push(None);
                continue;
            }
            self.inner.set_insertion_point(point);
            let result_ty = result.borrow().ty();
            let (_, outputs) =
                self.insert_cast(CastKind::Source, range.iter().copied(), [result_ty], span)?;
            replacements.push(Some(outputs[0]));
        }
        self.inner.replace_op_with_values(op, &replacements);
        self.inner.set_insertion_point(restore);
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

    /// Runs `f` on the operation and records it as modified: the change counts as a mutation and
    /// the operation is legalized again. A result whose type `f` changed and that has users is
    /// replaced, for those users, by a one-to-one cast back to its old type placed after the
    /// operation (the casts of several results follow it in the order of the results); the new
    /// type must be what the converter answers for the result. Changing the type of an argument of
    /// a block of the operation's own regions inside `f` is an error (use a signature conversion).
    ///
    /// What is watched is the operation itself: its own results and the arguments of the blocks of
    /// its own regions. `f` is meant to change the operation itself. A change `f` makes to an
    /// operation or a block nested deeper (for example with `recursively_replace_elements_in`) is
    /// not tracked: it gets no cast, and the nested operation is not legalized again.
    ///
    /// The converter is asked about each result ([TypeConverter::convert_value]) before `f` runs,
    /// while the result still has its old type, and the new type of a changed result must be
    /// exactly the one type it answered; with no converter, any change of a result type is an
    /// error. A changed result without users needs no cast. A changed result with users on a
    /// terminator is an error, since its cast would follow the terminator; no cast is inserted.
    /// These checks are made after `f` has run, so an error leaves the operation as `f` changed it,
    /// and aborts the conversion.
    ///
    /// An `f` that changes nothing still counts as a modification: a pattern that answers
    /// `Ok(true)` after it passes the driver's check that a successful pattern changed the IR, and
    /// when the operation is still illegal it is offered to the patterns again, until the
    /// conversion's limit on the pattern applications that follow from one operation
    /// ([super::ConversionConfig::with_max_rewrites_per_operation]) is reached.
    pub fn modify_op_in_place<R>(
        &mut self,
        mut op: OperationRef,
        f: impl FnOnce(&mut Operation) -> R,
    ) -> Result<R, Report> {
        let context = self.inner.context_rc();
        let results = op
            .borrow()
            .results()
            .all()
            .iter()
            .map(|result| {
                let result = result.borrow();
                (result.as_value_ref(), result.ty())
            })
            .collect::<SmallVec<[(ValueRef, Type); 2]>>();
        // What the converter answers for each result, asked while it has its old type.
        let expected = results
            .iter()
            .map(|(result, _)| {
                self.type_converter
                    .as_ref()
                    .map(|converter| converter.convert_value(&context, *result))
            })
            .collect::<SmallVec<[_; 2]>>();
        let blocks = blocks_of(op)
            .into_iter()
            .map(|block| (block, argument_types(block)))
            .collect::<SmallVec<[_; 2]>>();

        let answer = {
            let _modification = self.inner.modify_op_in_place(op);
            f(&mut op.borrow_mut())
        };
        self.tracking.record_in_place_op(op);

        let name = op.borrow().name();
        for (index, (block, before)) in blocks.iter().enumerate() {
            let after = argument_types(*block);
            if after != *before {
                return Err(Report::msg(format!(
                    "the arguments of block {index} in the regions of '{name}' changed type in \
                     place, from {} to {}; a block's arguments are changed by a signature \
                     conversion",
                    type_list(before),
                    type_list(&after)
                )));
            }
        }

        let mut changed = SmallVec::<[(ValueRef, Type); 2]>::new();
        for (index, ((result, old_ty), expected)) in results.iter().zip(&expected).enumerate() {
            let new_ty = result.borrow().ty();
            if new_ty == *old_ty {
                continue;
            }
            let Some(expected) = expected else {
                return Err(Report::msg(format!(
                    "result {index} of '{name}' changed type in place, from '{old_ty}' to \
                     '{new_ty}', and the pattern has no type converter"
                )));
            };
            if expected.as_slice() != [new_ty] {
                return Err(Report::msg(format!(
                    "result {index} of '{name}' changed type in place, from '{old_ty}' to \
                     '{new_ty}', but the type converter converts '{old_ty}' to {}",
                    type_list(expected)
                )));
            }
            if result.borrow().is_used() {
                if op.borrow().implements::<dyn Terminator>() {
                    return Err(Report::msg(format!(
                        "result {index} of '{name}' changed type in place, from '{old_ty}' to \
                         '{new_ty}', and has users, but '{name}' is a terminator: the cast back \
                         to '{old_ty}' cannot be placed after it"
                    )));
                }
                changed.push((*result, *old_ty));
            }
        }

        if !changed.is_empty() {
            let saved = *self.inner.insertion_point();
            let span = op.borrow().span();
            let mut after = op;
            for (result, old_ty) in changed {
                self.set_insertion_point_after(after);
                let (cast, outputs) =
                    self.insert_cast(CastKind::Source, [result], [old_ty], span)?;
                self.inner.maybe_replace_uses_of_value_with(result, outputs[0], |operand| {
                    operand.owner != cast
                });
                after = cast;
            }
            self.inner.set_insertion_point(saved);
        }
        Ok(answer)
    }

    /// Rewrites the block's arguments. Returns the new arguments. An old argument that still has
    /// users is replaced by a many-to-one cast at the start of the block (of its new arguments, or
    /// of the values it was remapped to; no cast when that is one value of its own type).
    ///
    /// The new arguments, of the types [SignatureConversion::converted_types], are appended to the
    /// block and the old ones erased: the block keeps its identity and its operations. The types an
    /// old argument becomes are taken as the conversion gives them; under a type converter they
    /// must be what the converter answers for the old argument. Values an argument is remapped to
    /// stand for it as given; they must dominate the start of the block (one defined under the
    /// block itself is an error; nothing else is checked). The casts are
    /// recorded like any the driver inserts, and the change counts as a mutation of the block's
    /// parent operation; of a block without a parent operation, as a mutation of the block.
    ///
    /// An error is raised before anything changes when the conversion does not map every old
    /// argument, when it maps one to types the converter does not answer for it, when an argument
    /// is remapped to an argument of the block itself or to a value defined under it (a result of
    /// one of its operations, or an argument or result in a region nested in one, at any depth),
    /// when a cast is needed and the pattern has no
    /// type converter, or when the block was already converted in this conversion.
    pub fn apply_signature_conversion(
        &mut self,
        mut block: BlockRef,
        conversion: &SignatureConversion,
    ) -> Result<SmallVec<[ValueRef; 4]>, Report> {
        let old = block.borrow().argument_values().collect::<SmallVec<[ValueRef; 4]>>();
        let parent = block.borrow().parent_op();
        let describe = || match parent {
            Some(parent) => format!("a block of '{}'", parent.borrow().name()),
            None => String::from("a block"),
        };

        // Check everything before changing anything.
        if self.state.converted_blocks.borrow().contains(&block) {
            return Err(Report::msg(format!(
                "the arguments of {} were already converted in this conversion",
                describe()
            )));
        }
        if conversion.num_inputs() != old.len() {
            return Err(Report::msg(format!(
                "a signature conversion of {} inputs cannot be applied to {}, which has {} \
                 arguments",
                conversion.num_inputs(),
                describe(),
                old.len()
            )));
        }
        let context = self.inner.context_rc();
        let mut needs_cast = SmallVec::<[bool; 4]>::new();
        for (index, argument) in old.iter().enumerate() {
            let old_ty = argument.borrow().ty();
            let Some(mapping) = conversion.mapping(index) else {
                return Err(Report::msg(format!(
                    "the signature conversion of {} does not map argument {index}",
                    describe()
                )));
            };
            let replacement = match mapping {
                InputMapping::Arguments { new_index, count } => {
                    let types = &conversion.converted_types()[*new_index..*new_index + *count];
                    if let Some(converter) = self.type_converter.as_ref() {
                        let expected = converter.convert_value(&context, *argument);
                        if expected.as_slice() != types {
                            return Err(Report::msg(format!(
                                "argument {index} of {} has type '{old_ty}', which the type \
                                 converter converts to {}, but the signature conversion gives it \
                                 types {}",
                                describe(),
                                type_list(&expected),
                                type_list(types)
                            )));
                        }
                    }
                    SmallVec::<[Type; 2]>::from(types)
                }
                InputMapping::Values(values) => {
                    if let Some(value) = values.iter().find(|value| old.contains(value)) {
                        return Err(Report::msg(format!(
                            "argument {index} of {} is remapped to '{}', an argument of the same \
                             block, which the conversion erases",
                            describe(),
                            value.borrow()
                        )));
                    }
                    if let Some(value) = values.iter().find(|value| is_defined_in(**value, block)) {
                        return Err(Report::msg(format!(
                            "argument {index} of {} is remapped to '{}', a value defined in the \
                             block or in a region nested in it, which does not dominate its start",
                            describe(),
                            value.borrow()
                        )));
                    }
                    types_of(values)
                }
            };
            let cast = argument.borrow().is_used() && replacement.as_slice() != [old_ty];
            if cast && self.type_converter.is_none() {
                return Err(Report::msg(format!(
                    "argument {index} of {} of type '{old_ty}' becomes values of types {}, which \
                     needs a cast, and the pattern has no type converter",
                    describe(),
                    type_list(&replacement)
                )));
            }
            needs_cast.push(cast);
        }

        if let Some(parent) = parent {
            self.inner.notify_operation_modification_started(&parent);
        }
        let block_span = parent.map_or_else(SourceSpan::default, |parent| parent.borrow().span());
        let spans = old
            .iter()
            .map(|argument| argument.borrow().span())
            .collect::<SmallVec<[_; 4]>>();
        let mut new_arguments = SmallVec::<[ValueRef; 4]>::new();
        for (new_index, ty) in conversion.converted_types().iter().enumerate() {
            let span =
                conversion.origin(new_index).map_or(block_span, |old_index| spans[old_index]);
            new_arguments.push(context.append_block_argument(block, *ty, span));
        }

        let saved = *self.inner.insertion_point();
        let mut ip = ProgramPoint::at_start_of(block);
        for (index, argument) in old.iter().enumerate() {
            if !argument.borrow().is_used() {
                continue;
            }
            let values = match conversion.mapping(index).expect("checked above") {
                InputMapping::Arguments { new_index, count } => {
                    SmallVec::<[ValueRef; 2]>::from(&new_arguments[*new_index..*new_index + *count])
                }
                InputMapping::Values(values) => values.clone(),
            };
            let replacement = if needs_cast[index] {
                self.inner.set_insertion_point(ip);
                let old_ty = argument.borrow().ty();
                let (cast, outputs) =
                    self.insert_cast(CastKind::Source, values, [old_ty], spans[index])?;
                ip = ProgramPoint::after(cast);
                outputs[0]
            } else {
                values[0]
            };
            self.inner.replace_all_uses_of_value_with(*argument, replacement);
        }
        self.inner.set_insertion_point(saved);
        for _ in 0..old.len() {
            block.borrow_mut().erase_argument(0);
        }
        match parent {
            Some(parent) => {
                self.inner.notify_operation_modified(parent);
                self.tracking.record_in_place_op(parent);
            }
            // No operation is told of the change; it is still a mutation.
            None => self.tracking.record_block_mutation(),
        }
        self.state.converted_blocks.borrow_mut().insert(block);
        Ok(new_arguments)
    }

    /// `SignatureConversion::for_block` applied to every block of the region that is not converted
    /// yet.
    ///
    /// The conversion of each block is [SignatureConversion::for_block] under the pattern's type
    /// converter, and it is applied with [Self::apply_signature_conversion]. A block whose
    /// arguments all convert to themselves is left as it is. Returns whether any block changed.
    /// Without a type converter this is an error.
    pub fn convert_region_types(&mut self, region: RegionRef) -> Result<bool, Report> {
        let converter = self.type_converter.as_ref().map(Rc::clone).ok_or_else(|| {
            Report::msg("converting the types of a region needs a type converter")
        })?;
        let blocks = region
            .borrow()
            .body()
            .iter()
            .map(|block| block.as_block_ref())
            .collect::<SmallVec<[BlockRef; 4]>>();
        let mut changed = false;
        for block in blocks {
            if self.state.converted_blocks.borrow().contains(&block) {
                continue;
            }
            let conversion = SignatureConversion::for_block(block, &converter)?;
            if conversion.is_identity(block) {
                continue;
            }
            self.apply_signature_conversion(block, &conversion)?;
            changed = true;
        }
        Ok(changed)
    }

    /// Create a block at the end of `region`, with arguments of `argument_types`, and set the
    /// insertion point to the end of it. Returns the block.
    ///
    /// `region` must be a region of an operation this pattern created (not a cast the driver
    /// made). The pattern is building converted IR, so the argument types are taken as legal and
    /// are not asked of the type converter. The new block counts as a mutation. A region of any
    /// other operation, or of none, is an error, raised before anything changes.
    pub fn create_block(
        &mut self,
        region: RegionRef,
        argument_types: &[Type],
    ) -> Result<BlockRef, Report> {
        let parent = region.borrow().parent();
        let Some(parent) = parent else {
            return Err(Report::msg(
                "a block can be created only in a region of an operation the pattern created, and \
                 this region belongs to no operation",
            ));
        };
        if !self.created(parent) {
            return Err(Report::msg(format!(
                "a block can be created only in a region of an operation the pattern created, but \
                 this region is of '{}', which the pattern did not create",
                parent.borrow().name()
            )));
        }
        Ok(self.inner.create_block(region, None, argument_types))
    }

    /// Move every operation of `source`, in order, to before `before`, replace each argument of
    /// `source` by the values given for it, and erase `source`.
    ///
    /// This is [Self::inline_block_at_end], with the operations placed before `before` in its
    /// block instead of at the end of a block. The checks and the replacement of the arguments are
    /// those of [Self::inline_block_at_end].
    pub fn inline_block_before(
        &mut self,
        source: BlockRef,
        before: OperationRef,
        argument_values: &[&[ValueRef]],
    ) -> Result<(), Report> {
        let Some(destination) = before.parent() else {
            return Err(Report::msg(format!(
                "a block cannot be inlined before '{}', which is in no block",
                before.borrow().name()
            )));
        };
        self.inline_block(source, destination, Some(before), argument_values)
    }

    /// Move every operation of `source`, in order, to the end of `destination`, replace each
    /// argument of `source` by the values given for it, and erase `source`.
    ///
    /// `argument_values` holds one range per argument of `source`: one value, several (when the
    /// argument's type converts to several types), or none (when it converts to no type). A range
    /// that is one value of the argument's own type replaces the argument. When the pattern has a
    /// type converter, any other range must have exactly the types it gives the argument; without
    /// one, any other range is accepted only for an argument without users. When the argument has
    /// users and such a range, they are given a many-to-one `builtin.unrealized_conversion_cast` of
    /// the range back to the argument's type, placed before the moved operations: the same
    /// relation a signature conversion makes for an argument remapped to values
    /// ([SignatureConversion::remap_input]). So a user converted later is given the values, and
    /// one that stays reads the cast, which is realized by the converter's materializations. The
    /// values must dominate the place the operations move to; one defined in `source` is an
    /// error, and nothing else is checked.
    ///
    /// `source` must be a block of the operation the pattern was given, or of an operation nested
    /// in it; it must be the only block of its region, must not be the successor of a branch, and
    /// must not end in a branch (the branch would leave the region of its successors). The
    /// operation whose region `source` was is left with an empty region, and the pattern erases
    /// or replaces it: one left in the IR when the pattern succeeds is an error the driver
    /// reports. `destination` must be a block of a region of an operation the pattern created
    /// (that operation itself, not one it holds), or a block that holds the operation the pattern
    /// was given (directly or through the operations around it); it must not be `source` or a
    /// block nested in it, and, for [Self::inline_block_at_end], must not end in a branch. These
    /// rules keep the moved operations in the order the driver converts: a moved operation is
    /// offered at most once, and after the operation that now holds it (see the module
    /// documentation of [crate::conversion]).
    ///
    /// The terminator of `source` moves like any other operation; a pattern that does not want it
    /// erases or replaces it afterwards. The moves and the erasure of `source` count as mutations,
    /// so a pattern that inlines a block and then answers `Ok(false)` or `Err` breaks the pattern
    /// contract. An insertion point in `source` moves to the end of the inlined operations; any
    /// other insertion point is kept.
    ///
    /// An error is raised before anything changes when this rewriter was not given an operation
    /// by the driver, when `source` or `destination` breaks the rules above, for the wrong number
    /// of ranges, a range that holds an argument of `source` or a value defined in it, a range of
    /// other types than the converter gives the argument, and a cast that is needed when the
    /// pattern has no type converter.
    pub fn inline_block_at_end(
        &mut self,
        source: BlockRef,
        destination: BlockRef,
        argument_values: &[&[ValueRef]],
    ) -> Result<(), Report> {
        self.inline_block(source, destination, None, argument_values)
    }

    /// [Self::inline_block_before] (`Some(before)`, which is in `destination`) and
    /// [Self::inline_block_at_end] (`None`).
    fn inline_block(
        &mut self,
        source: BlockRef,
        destination: BlockRef,
        before: Option<OperationRef>,
        argument_values: &[&[ValueRef]],
    ) -> Result<(), Report> {
        let Some(matched) = self.matched else {
            return Err(Report::msg(
                "a block can be inlined only by a pattern the conversion driver applies, which is \
                 given the operation whose blocks it may inline",
            ));
        };
        let matched_name = matched.borrow().name();
        let describe = |block: BlockRef| match block.borrow().parent_op() {
            Some(parent) => format!("a block of '{}'", parent.borrow().name()),
            None => String::from("a block of no operation"),
        };
        let source_name = describe(source);

        // Check everything before changing anything.
        let source_parent = source.borrow().parent_op();
        if !source_parent.is_some_and(|parent| is_within(parent, matched)) {
            return Err(Report::msg(format!(
                "a pattern may inline only a block of the operation it is given, \
                 '{matched_name}', or of an operation nested in it, but the block is {source_name}"
            )));
        }
        if source.borrow().has_predecessors() {
            return Err(Report::msg(format!(
                "{source_name} is the successor of a branch and cannot be inlined"
            )));
        }
        if is_block_within(destination, source) {
            return Err(Report::msg(format!(
                "{source_name} cannot be inlined into itself or into a block nested in it"
            )));
        }
        if source.borrow().has_successors() {
            return Err(Report::msg(format!(
                "{source_name} ends in a branch and cannot be inlined: the branch would leave the \
                 region its successors are in"
            )));
        }
        let blocks_in_region =
            source.borrow().parent().map_or(1, |region| region.borrow().body().len());
        if blocks_in_region > 1 {
            return Err(Report::msg(format!(
                "{source_name} is one of the {blocks_in_region} blocks of its region and cannot \
                 be inlined: only the one block of a region can be inlined"
            )));
        }
        let destination_name = describe(destination);
        let in_created = destination.borrow().parent_op().is_some_and(|op| self.created(op));
        let holds_matched =
            matched.parent().is_some_and(|block| is_block_within(block, destination));
        if !in_created && !holds_matched {
            return Err(Report::msg(format!(
                "a pattern may inline a block only into a block of an operation it created or \
                 into a block that holds the operation it is given, '{matched_name}', but the \
                 destination is {destination_name}, which is neither"
            )));
        }
        if before.is_none() && destination.borrow().has_successors() {
            return Err(Report::msg(format!(
                "{source_name} cannot be inlined at the end of {destination_name}, which ends in \
                 a branch"
            )));
        }
        let arguments = source.borrow().argument_values().collect::<SmallVec<[ValueRef; 4]>>();
        if arguments.len() != argument_values.len() {
            return Err(Report::msg(format!(
                "{source_name} has {} arguments, but it is inlined with {} ranges of values for \
                 them",
                arguments.len(),
                argument_values.len()
            )));
        }
        let context = self.inner.context_rc();
        let mut needs_cast = SmallVec::<[bool; 4]>::new();
        for (index, (argument, range)) in arguments.iter().zip(argument_values).enumerate() {
            if let Some(value) = range.iter().find(|value| arguments.contains(value)) {
                return Err(Report::msg(format!(
                    "argument {index} of {source_name} is replaced by '{}', an argument of the \
                     same block, which is erased with it",
                    value.borrow()
                )));
            }
            if let Some(value) = range.iter().find(|value| is_defined_in(**value, source)) {
                return Err(Report::msg(format!(
                    "argument {index} of {source_name} is replaced by '{}', a value defined in \
                     the block or in a region nested in it, which does not dominate the \
                     operations it holds",
                    value.borrow()
                )));
            }
            let old_ty = argument.borrow().ty();
            let used = argument.borrow().is_used();
            if let [value] = range
                && value.borrow().ty() == old_ty
            {
                needs_cast.push(false);
                continue;
            }
            let range_types = types_of(range);
            match self.type_converter.as_ref() {
                Some(converter) => {
                    let expected = converter.convert_value(&context, *argument);
                    if range_types != expected {
                        return Err(Report::msg(format!(
                            "argument {index} of {source_name} has type '{old_ty}', which the \
                             type converter converts to {}, but it is replaced by values of types \
                             {}",
                            type_list(&expected),
                            type_list(&range_types)
                        )));
                    }
                }
                None if used => {
                    return Err(Report::msg(format!(
                        "argument {index} of {source_name} of type '{old_ty}' is replaced by \
                         values of types {}, which needs a cast, and the pattern has no type \
                         converter",
                        type_list(&range_types)
                    )));
                }
                None => {}
            }
            needs_cast.push(used);
        }

        let saved = *self.inner.insertion_point();
        let at = before.map_or_else(|| ProgramPoint::at_end_of(destination), ProgramPoint::before);
        for ((argument, range), cast) in arguments.iter().zip(argument_values).zip(&needs_cast) {
            if !argument.borrow().is_used() {
                continue;
            }
            let replacement = if *cast {
                self.inner.set_insertion_point(at);
                let (old_ty, span) = {
                    let argument = argument.borrow();
                    (argument.ty(), argument.span())
                };
                let (_, outputs) =
                    self.insert_cast(CastKind::Source, range.iter().copied(), [old_ty], span)?;
                outputs[0]
            } else {
                range[0]
            };
            self.inner.replace_all_uses_of_value_with(*argument, replacement);
        }

        *self.tracking.moving.borrow_mut() =
            source.borrow().body().iter().map(|op| op.as_operation_ref()).collect();
        if let Some(parent) = source_parent {
            self.tracking.record_emptied_op(parent);
        }
        let ignored = SmallVec::<[Option<ValueRef>; 4]>::from_elem(None, arguments.len());
        self.inner.inline_block_before(source, destination, before, &ignored);
        self.tracking.moving.borrow_mut().clear();

        let restore = match saved {
            ProgramPoint::Block { block, .. } if block == source => at,
            _ => saved,
        };
        self.inner.set_insertion_point(restore);
        Ok(())
    }

    /// Whether this pattern created `op` (inserted it, not as a cast the driver made) and it is
    /// still in the IR.
    fn created(&self, op: OperationRef) -> bool {
        let mutations = self.tracking.mutations.borrow();
        op.parent().is_some()
            && mutations.inserted_ops.contains(&op)
            && !mutations.materialized_ops.contains(&op)
    }

    /// Erase an operation with no remaining uses.
    ///
    /// Erasing an operation a result of which still has real users (users that are not
    /// `Transparent`), the root of the conversion, or an operation a block of which (at any depth)
    /// is the successor of a branch that is not inside the operation, is an error, raised before
    /// anything changes. Transparent users are erased with the operation.
    pub fn erase_op(&mut self, op: OperationRef) -> Result<(), Report> {
        self.check_not_root(op, "erased")?;
        check_no_branch_from_outside(op, "erased")?;
        if let Some(index) =
            op.borrow().results().iter().position(|result| result.borrow().has_real_uses())
        {
            return Err(Report::msg(format!(
                "'{}' cannot be erased: its result {index} still has users (replace it instead)",
                op.borrow().name()
            )));
        }
        self.inner.erase_op(op);
        Ok(())
    }

    /// An error when `op` is the root of the conversion, which cannot be `what` (erased, replaced).
    fn check_not_root(&self, op: OperationRef, what: &str) -> Result<(), Report> {
        if self.state.root == Some(op) {
            return Err(Report::msg(format!(
                "'{}' is the root of the conversion, which cannot be {what}",
                op.borrow().name()
            )));
        }
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

/// Whether `value` is defined under `block`: an argument of it, a result of one of its operations,
/// or an argument or result in a region nested in one of its operations, at any depth.
fn is_defined_in(value: ValueRef, block: BlockRef) -> bool {
    let mut current = value.borrow().parent_block();
    while let Some(candidate) = current {
        if candidate == block {
            return true;
        }
        current = candidate.borrow().parent_op().and_then(|op| op.parent());
    }
    false
}

/// Whether `op` is `ancestor` or nested in it, at any depth.
fn is_within(op: OperationRef, ancestor: OperationRef) -> bool {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if candidate == ancestor {
            return true;
        }
        current = candidate.parent_op();
    }
    false
}

/// Whether `block` is `ancestor` or nested in an operation of it, at any depth.
fn is_block_within(block: BlockRef, ancestor: BlockRef) -> bool {
    let mut current = Some(block);
    while let Some(candidate) = current {
        if candidate == ancestor {
            return true;
        }
        current = candidate.borrow().parent_op().and_then(|op| op.parent());
    }
    false
}

/// Whether `value` is defined inside `op`: in a block of one of its regions, at any depth.
fn is_defined_inside(value: ValueRef, op: OperationRef) -> bool {
    let mut current = value.borrow().parent_block();
    while let Some(block) = current {
        let Some(parent) = block.borrow().parent_op() else {
            return false;
        };
        if parent == op {
            return true;
        }
        current = parent.parent();
    }
    false
}

/// The operation that defines the value of `range` defined last, when that definition is in `op`'s
/// block after `op`: the source cast of `range`, which stands for a result of `op`, goes after it.
/// `None` when no value of `range` is defined there; the cast then goes before `op`.
fn last_definition_after(op: OperationRef, range: &[ValueRef]) -> Option<OperationRef> {
    let block = op.parent()?;
    let mut last = op;
    for value in range {
        let Some(definition) = value.borrow().get_defining_op() else {
            continue;
        };
        if definition.parent() == Some(block) && last.borrow().is_before_in_block(&definition) {
            last = definition;
        }
    }
    (last != op).then_some(last)
}

/// A user of `result` in the block of `last` (directly, or nested in one of its operations) that is
/// not after `last`, and so cannot read a cast placed after `last`.
fn user_not_after(result: ValueRef, last: OperationRef) -> Option<OperationRef> {
    let block = last.parent()?;
    let result = result.borrow();
    result.iter_uses().find_map(|user| {
        let mut ancestor = user.owner;
        loop {
            let parent = ancestor.parent()?;
            if parent == block {
                break;
            }
            ancestor = parent.borrow().parent_op()?;
        }
        (!last.borrow().is_before_in_block(&ancestor)).then_some(user.owner)
    })
}

/// An error when a block of `op`, at any depth, is the successor of a branch that is not inside
/// `op`: erasing `op` (to be `what`, erased or replaced) would leave that branch without its
/// successor.
fn check_no_branch_from_outside(op: OperationRef, what: &str) -> Result<(), Report> {
    let mut outside = None;
    op.borrow().prewalk_all(|nested: &Operation| {
        if outside.is_some() {
            return;
        }
        for region in nested.regions() {
            for block in region.body() {
                let branch = block
                    .predecessors()
                    .map(|predecessor| predecessor.owner)
                    .find(|owner| !is_within(*owner, op));
                if let Some(branch) = branch {
                    outside = Some(branch);
                    return;
                }
            }
        }
    });
    match outside {
        Some(branch) => Err(Report::msg(format!(
            "'{}' cannot be {what}: a block in it is the successor of '{}', which is not inside \
             it and would be left without its successor",
            op.borrow().name(),
            branch.borrow().name()
        ))),
        None => Ok(()),
    }
}

/// The blocks of the regions of `op`, in order.
fn blocks_of(op: OperationRef) -> SmallVec<[BlockRef; 2]> {
    let op = op.borrow();
    op.regions()
        .iter()
        .flat_map(|region| {
            region
                .body()
                .iter()
                .map(|block| block.as_block_ref())
                .collect::<SmallVec<[_; 2]>>()
        })
        .collect()
}

/// The types of the arguments of `block`, in order.
fn argument_types(block: BlockRef) -> SmallVec<[Type; 4]> {
    block.borrow().argument_types().collect()
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
        let name = context
            .get_or_register_dialect::<<Constant as OpRegistration>::Dialect>()
            .expect_registered_name::<Constant>();
        // SAFETY: the operation is only named in a match failure, never read
        let op = unsafe { name.alloc_default(context.clone()) };
        let mut rewriter = ConversionPatternRewriter::new(context);

        rewriter.notify_match_failure(op, Report::msg("no match"));

        assert_eq!(rewriter.match_failures().len(), 1);
        rewriter.clear_match_failures();
        assert!(rewriter.match_failures().is_empty());
    }
}
