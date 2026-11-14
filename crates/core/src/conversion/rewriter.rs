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

/// What the rewriters of one conversion share: the casts the driver inserted and the blocks whose
/// arguments were converted.
#[derive(Clone, Default)]
pub(super) struct ConversionState {
    pub(super) casts: DriverCasts,
    pub(super) converted_blocks: ConvertedBlocks,
}

struct ConversionTrackingListener {
    mutations: RefCell<TrackedMutations>,
    state: ConversionState,
}

impl ConversionTrackingListener {
    fn new(state: ConversionState) -> Self {
        Self {
            mutations: RefCell::new(TrackedMutations::default()),
            state,
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

    /// The context of the IR being converted.
    #[inline]
    pub fn context(&self) -> &Context {
        self.inner.context()
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
    /// `results` at the current insertion point, record it as the driver's cast of `kind` under
    /// this rewriter's type converter, and return its results.
    ///
    /// The driver inserts a cast only for a pattern with a type converter.
    pub(super) fn insert_cast(
        &mut self,
        kind: CastKind,
        inputs: impl IntoIterator<Item = ValueRef>,
        results: impl IntoIterator<Item = Type>,
        span: SourceSpan,
    ) -> Result<SmallVec<[ValueRef; 2]>, Report> {
        let converter = self.type_converter.as_ref().map(Rc::clone).ok_or_else(|| {
            Report::msg("the driver inserts a cast only for a pattern with a type converter")
        })?;
        let cast =
            BuiltinOpBuilder::unrealized_conversion_cast(&mut *self.inner, inputs, results, span)?;
        let cast_op = cast.as_operation_ref();
        self.mark_materialization_op(cast_op);
        self.state.casts.borrow_mut().insert(cast_op, DriverCast { kind, converter });
        let outputs = cast.borrow().outputs().iter().map(|output| *output as ValueRef).collect();
        Ok(outputs)
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
    ///
    /// Afterwards the insertion point is where it was before the call; when it was at `op`, it is
    /// at the position `op` occupied (before the operation that followed it, or at the end of its
    /// block).
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
            casts.push(true);
        }

        // The casts are inserted before `op`; afterwards the insertion point is where it was, or,
        // when it was at `op`, at the position `op` occupied.
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
            let outputs =
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
    /// operation; the new type must be what the converter answers for the old one. Changing the
    /// type of a block argument inside `f` is an error (use a signature conversion).
    ///
    /// A changed result without users needs no cast. The new type of a changed result must be
    /// exactly the one type the pattern's type converter answers for a value of the old type; with
    /// no converter, any change of a result type is an error. These checks are made after `f` has
    /// run, so an error leaves the operation as `f` changed it, and aborts the conversion.
    pub fn modify_op_in_place<R>(
        &mut self,
        mut op: OperationRef,
        f: impl FnOnce(&mut Operation) -> R,
    ) -> Result<R, Report> {
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
        let blocks = blocks_of(op)
            .into_iter()
            .map(|block| (block, argument_types(block)))
            .collect::<SmallVec<[_; 2]>>();

        let answer = {
            let _modification = self.inner.modify_op_in_place(op);
            f(&mut op.borrow_mut())
        };

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

        let context = self.inner.context_rc();
        let mut changed = SmallVec::<[(ValueRef, Type); 2]>::new();
        for (index, (result, old_ty)) in results.iter().enumerate() {
            let new_ty = result.borrow().ty();
            if new_ty == *old_ty {
                continue;
            }
            let Some(converter) = self.type_converter.as_ref() else {
                return Err(Report::msg(format!(
                    "result {index} of '{name}' changed type in place, from '{old_ty}' to \
                     '{new_ty}', and the pattern has no type converter"
                )));
            };
            let expected = converter.convert_type(&context, *old_ty);
            if expected.as_slice() != [new_ty] {
                return Err(Report::msg(format!(
                    "result {index} of '{name}' changed type in place, from '{old_ty}' to \
                     '{new_ty}', but the type converter converts '{old_ty}' to {}",
                    type_list(&expected)
                )));
            }
            if result.borrow().is_used() {
                changed.push((*result, *old_ty));
            }
        }

        if !changed.is_empty() {
            let saved = *self.inner.insertion_point();
            let span = op.borrow().span();
            let mut after = op;
            for (result, old_ty) in changed {
                self.set_insertion_point_after(after);
                let outputs = self.insert_cast(CastKind::Source, [result], [old_ty], span)?;
                let cast = outputs[0]
                    .borrow()
                    .get_defining_op()
                    .expect("a cast's result is defined by the cast");
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
    /// stand for it as given. The casts are recorded like any the driver inserts, and the change
    /// counts as a mutation of the block's parent operation.
    ///
    /// An error is raised before anything changes when the conversion does not map every old
    /// argument, when it maps one to types the converter does not answer for it, when an argument
    /// is remapped to an argument of the block itself, when a cast is needed and the pattern has no
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
                let outputs = self.insert_cast(CastKind::Source, values, [old_ty], spans[index])?;
                let cast = outputs[0]
                    .borrow()
                    .get_defining_op()
                    .expect("a cast's result is defined by the cast");
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
        if let Some(parent) = parent {
            self.inner.notify_operation_modified(parent);
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
