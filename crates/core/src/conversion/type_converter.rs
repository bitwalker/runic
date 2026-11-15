use alloc::{format, rc::Rc, string::String, vec::Vec};
use core::cell::RefCell;

use smallvec::SmallVec;

use super::ConversionPatternRewriter;
use crate::{Context, FxHashMap, Operation, Report, SourceSpan, Type, ValueRef, WalkResult};

type TypeConversionFn = Rc<dyn Fn(&Context, Type) -> Option<TypeConversion>>;
type SourceMaterializationFn = Rc<
    dyn Fn(
        &mut ConversionPatternRewriter,
        &[ValueRef],
        Type,
        SourceSpan,
    ) -> Result<Option<ValueRef>, Report>,
>;
type TargetMaterializationFn = Rc<
    dyn Fn(
        &mut ConversionPatternRewriter,
        ValueRef,
        &[Type],
        SourceSpan,
    ) -> Result<Option<SmallVec<[ValueRef; 2]>>, Report>,
>;

/// What a conversion callback answers for one type: the types a value of that type becomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeConversion {
    /// The value becomes one value of this type.
    One(Type),
    /// The value becomes one value of each of these types, in order. A list of one type means
    /// the same as [Self::One], and an empty list the same as [Self::Drop].
    Many(SmallVec<[Type; 2]>),
    /// The value becomes no value.
    Drop,
}

impl TypeConversion {
    /// The value becomes one value of type `ty`.
    #[inline]
    pub fn one(ty: Type) -> Self {
        Self::One(ty)
    }

    /// The value becomes one value of each of `types`, in order.
    #[inline]
    pub fn many(types: impl IntoIterator<Item = Type>) -> Self {
        Self::Many(types.into_iter().collect())
    }

    /// The value becomes no value.
    #[inline]
    pub const fn drop() -> Self {
        Self::Drop
    }

    /// The types the value becomes.
    fn into_types(self) -> SmallVec<[Type; 2]> {
        match self {
            Self::One(ty) => smallvec::smallvec![ty],
            Self::Many(types) => types,
            Self::Drop => SmallVec::new(),
        }
    }
}

/// Says what types the values of a conversion become.
///
/// A converter answers for types only: a value (an operand, a result, a block argument) converts
/// as its type does. A converter is flat: it is asked about the type of a value, never about a
/// type held inside another type. A callback that wants a deep map builds a
/// [TypeMapper](crate::types::TypeMapper) in the call.
///
/// Callbacks are tried most recently added first, and the first that answers wins; a type no
/// callback answers for converts to itself. Each callback is given the context of the IR being
/// converted, in which it builds the types it converts to.
///
/// While a conversion runs, a value that became other values is related to them by a
/// `builtin.unrealized_conversion_cast` the driver inserts: a source cast from the new values to
/// the old type, for a user that still expects the old value, or a target cast from the old value
/// to the new types, for a pattern that expects the new ones. When the patterns have run and the
/// casts whose ends agree are folded away, each cast that remains is offered to the materialization
/// callbacks of its kind of the converter of the pattern that caused it, most recently added first.
/// The first callback that answers builds, before the cast, the operations that compute the cast's
/// results; they replace the cast's results and the cast is erased. The operations a callback
/// builds are not offered to patterns: they must be legal in the target. A
/// `builtin.unrealized_conversion_cast` a callback builds is not realized either; only the target
/// judges it. The folding assumes that casts compose (a cast from one type list to a second
/// followed by one from the second to a third means a cast from the first to the third, and a cast
/// from a type list to itself is the identity), so the materializations of a conversion must agree
/// with that.
///
/// The converters of the patterns of one conversion must also agree with each other on what a
/// cast between two type lists means. An operand is read through any cast whose input types are
/// what the pattern's converter answers for it, whoever caused the cast, and chains of casts are
/// folded whoever made their links. Two converters that answer the same type list for a type but
/// mean different things by it (one splits an `i128` into its low and high halves, the other into
/// its high and low halves) miscompile silently.
///
/// The converter remembers what the callbacks answered for a type, so each type is asked about
/// once. The memo is shared by the clones of a converter, and a clone that is given another
/// callback starts a memo of its own.
///
/// A converter made with [Self::with_function_types] answers for a builtin function type itself,
/// before any callback is asked: the function type of the converted parameters and results. This
/// is the one exception to "a converter is flat", and the builtin conversion patterns require it.
#[derive(Clone, Default)]
pub struct TypeConverter {
    function_types: bool,
    conversions: Vec<TypeConversionFn>,
    source_materializations: Vec<SourceMaterializationFn>,
    target_materializations: Vec<TargetMaterializationFn>,
    memo: Rc<RefCell<FxHashMap<Type, SmallVec<[Type; 2]>>>>,
}

impl TypeConverter {
    /// A converter under which every type converts to itself.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a type conversion callback.
    ///
    /// Return `Some` to answer for the type, or `None` to let the callbacks added before it try.
    /// Callbacks are tried most recently added first, so a later callback for a type wins over an
    /// earlier one, and are given the context the type belongs to.
    pub fn add_conversion<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn(&Context, Type) -> Option<TypeConversion> + 'static,
    {
        self.conversions.push(Rc::new(callback));
        self.forget_answers();
        self
    }

    /// Makes this converter answer, for a builtin function type, the function type of its converted
    /// parameters and results (each flattened), before any registered conversion is asked. This
    /// is a case inside [Self::convert_type] (a callback is not given the converter); it starts a
    /// new memo, as adding a conversion does.
    ///
    /// Each parameter and result is converted with [Self::convert_type], so a function type held
    /// by a function type converts too; a `never` result stays `never`, as it does in
    /// [super::convert_signature]. So for every signature `s`, the function type of
    /// `convert_signature(s)` is what this converter answers for the function type of `s`.
    pub fn with_function_types(mut self) -> Self {
        self.function_types = true;
        self.forget_answers();
        self
    }

    /// Whether the converter answers for builtin function types itself
    /// ([Self::with_function_types]).
    #[inline]
    pub fn converts_function_types(&self) -> bool {
        self.function_types
    }

    /// Add a source materialization callback: many new values to one value of the old type.
    ///
    /// It is given the values (none for a value that was dropped), the old type and the span of
    /// the cast, and builds at the rewriter's insertion point the operations that compute a value
    /// of the old type from them. `None`: this callback does not apply. A
    /// `builtin.unrealized_conversion_cast` it builds is not realized; only the target judges it.
    pub fn add_source_materialization<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn(
                &mut ConversionPatternRewriter,
                &[ValueRef],
                Type,
                SourceSpan,
            ) -> Result<Option<ValueRef>, Report>
            + 'static,
    {
        self.source_materializations.push(Rc::new(callback));
        self
    }

    /// Add a target materialization callback: one old value to values of the new types.
    ///
    /// It is given the value, the new types and the span of the cast, and builds at the
    /// rewriter's insertion point the operations that compute one value of each new type from it.
    /// `None`: this callback does not apply. A `builtin.unrealized_conversion_cast` it builds is
    /// not realized; only the target judges it.
    pub fn add_target_materialization<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn(
                &mut ConversionPatternRewriter,
                ValueRef,
                &[Type],
                SourceSpan,
            ) -> Result<Option<SmallVec<[ValueRef; 2]>>, Report>
            + 'static,
    {
        self.target_materializations.push(Rc::new(callback));
        self
    }

    /// The types a value of type `ty` becomes: itself alone when it is legal, several, or none.
    ///
    /// Under [Self::with_function_types], a builtin function type becomes one function type,
    /// whatever the callbacks say about it.
    pub fn convert_type(&self, context: &Context, ty: Type) -> SmallVec<[Type; 2]> {
        if let Some(types) = self.memo.borrow().get(&ty) {
            return types.clone();
        }
        let types = match ty.as_function().filter(|_| self.function_types) {
            Some(function) => {
                let (params, results) = (function.params().to_vec(), function.results().to_vec());
                smallvec::smallvec![self.convert_function_type(context, &params, &results)]
            }
            None => self
                .conversions
                .iter()
                .rev()
                .find_map(|convert| convert(context, ty))
                .map_or_else(|| smallvec::smallvec![ty], TypeConversion::into_types),
        };
        self.memo.borrow_mut().insert(ty, types.clone());
        types
    }

    /// The function type of `params` and `results`, each converted and flattened; a `never`
    /// result stays `never`.
    fn convert_function_type(&self, context: &Context, params: &[Type], results: &[Type]) -> Type {
        let params = params
            .iter()
            .flat_map(|param| self.convert_type(context, *param))
            .collect::<SmallVec<[Type; 4]>>();
        let results = results
            .iter()
            .flat_map(|result| self.convert_result_type(context, *result))
            .collect::<SmallVec<[Type; 2]>>();
        context.function_type(params, results)
    }

    /// The types a declared result of type `ty` becomes: `never` stays `never`, whatever the
    /// callbacks say about it; any other type is [Self::convert_type].
    pub(super) fn convert_result_type(&self, context: &Context, ty: Type) -> SmallVec<[Type; 2]> {
        if ty.is_never() {
            smallvec::smallvec![ty]
        } else {
            self.convert_type(context, ty)
        }
    }

    /// The types `value` becomes: what its type becomes ([Self::convert_type]).
    #[inline]
    pub fn convert_value(&self, context: &Context, value: ValueRef) -> SmallVec<[Type; 2]> {
        self.convert_type(context, value.borrow().ty())
    }

    /// Whether `ty` converts to exactly itself.
    #[inline]
    pub fn is_legal_type(&self, context: &Context, ty: Type) -> bool {
        self.convert_type(context, ty).as_slice() == [ty]
    }

    /// Whether every type the operation has or holds converts to itself: its operands' and results'
    /// types, the argument types of its regions' blocks, and each type held directly by one of its
    /// attributes at any depth of attribute (types held inside other types are not asked about).
    ///
    /// Each type is asked about with [Self::is_legal_type], so through the callbacks and what the
    /// converter remembers of them. The types visited are
    /// those of [Operation::walk_types] without descending into a type, plus the operand types. As
    /// that walk does, it leaves out the type an attribute's kind fixes (the `i1` of a `BoolAttr`,
    /// the `i32` of an `I32Attr`) and the types held by the key of a keyed successor; the operations
    /// nested in the regions are not looked at.
    pub fn is_legal_op(&self, op: &Operation) -> bool {
        let context = op.context();
        let operands_legal = op
            .operands()
            .iter()
            .all(|operand| self.is_legal_type(context, operand.borrow().ty()));
        operands_legal
            && !op
                .walk_types(&mut |_, ty| {
                    if self.is_legal_type(context, ty) {
                        WalkResult::Skip
                    } else {
                        WalkResult::Break(Report::msg("illegal type"))
                    }
                })
                .was_interrupted()
    }

    /// The one type `ty` converts to, or an error when it converts to several or none.
    pub fn convert_type_1_to_1(&self, context: &Context, ty: Type) -> Result<Type, Report> {
        exactly_one(self.convert_type(context, ty), || format!("type '{ty}'"))
    }

    /// The one type `value` converts to ([Self::convert_value]), or an error when it converts to
    /// several or none.
    pub fn convert_value_1_to_1(&self, context: &Context, value: ValueRef) -> Result<Type, Report> {
        exactly_one(self.convert_value(context, value), || {
            format!("value '{}' of type '{}'", value.borrow(), value.borrow().ty())
        })
    }

    /// The value of type `ty` the first source materialization callback that answers, most
    /// recently added first, builds from `inputs`; `None` when none answers.
    pub(super) fn materialize_source(
        &self,
        rewriter: &mut ConversionPatternRewriter,
        inputs: &[ValueRef],
        ty: Type,
        span: SourceSpan,
    ) -> Result<Option<ValueRef>, Report> {
        for materialize in self.source_materializations.iter().rev() {
            if let Some(value) = materialize(rewriter, inputs, ty, span)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// The values of `types` the first target materialization callback that answers, most
    /// recently added first, builds from `input`; `None` when none answers.
    pub(super) fn materialize_target(
        &self,
        rewriter: &mut ConversionPatternRewriter,
        input: ValueRef,
        types: &[Type],
        span: SourceSpan,
    ) -> Result<Option<SmallVec<[ValueRef; 2]>>, Report> {
        for materialize in self.target_materializations.iter().rev() {
            if let Some(values) = materialize(rewriter, input, types, span)? {
                return Ok(Some(values));
            }
        }
        Ok(None)
    }

    /// Start a memo of this converter's own: its callbacks are no longer those of its clones.
    fn forget_answers(&mut self) {
        self.memo = Rc::default();
    }
}

fn exactly_one(
    types: SmallVec<[Type; 2]>,
    describe: impl FnOnce() -> String,
) -> Result<Type, Report> {
    match types.as_slice() {
        [ty] => Ok(*ty),
        [] => Err(Report::msg(format!(
            "{} is dropped by the type converter, but a 1:1 conversion is required",
            describe()
        ))),
        types => Err(Report::msg(format!(
            "{} converts to {} types, but a 1:1 conversion is required",
            describe(),
            types.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use smallvec::smallvec;

    use super::*;
    use crate::{Context, testing::Test};

    #[test]
    fn defaults_to_identity_conversion() {
        let context = Context::default();
        let converter = TypeConverter::new();

        assert_eq!(converter.convert_type_1_to_1(&context, context.i32()).unwrap(), context.i32());
        assert_eq!(converter.convert_type(&context, context.i32()).as_slice(), [context.i32()]);
        assert!(converter.is_legal_type(&context, context.i32()));
    }

    #[test]
    fn applies_registered_type_conversion() {
        let context = Context::default();
        let (i32, i64) = (context.i32(), context.i64());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| {
            if ty == i32 {
                Some(TypeConversion::One(i64))
            } else {
                None
            }
        });

        assert_eq!(converter.convert_type_1_to_1(&context, i32).unwrap(), i64);
        assert!(!converter.is_legal_type(&context, i32));
    }

    #[test]
    fn a_conversion_can_build_types_in_the_context_it_is_given() {
        let context = Context::default();
        let mut converter = TypeConverter::new();
        converter.add_conversion(|context, ty| {
            (!ty.is_pointer()).then(|| TypeConversion::One(context.ptr_type(ty)))
        });

        let i32 = context.i32();
        assert_eq!(converter.convert_type_1_to_1(&context, i32).unwrap(), context.ptr_type(i32));
        assert!(converter.is_legal_type(&context, context.opaque_ptr_type()));
    }

    /// `Many` gives its types in order; `Many` of one type is `One`, and of none `Drop`.
    #[test]
    fn a_type_converts_to_several_one_or_no_types() {
        let context = Context::default();
        let (i8, i16, i32, i64, i128) =
            (context.i8(), context.i16(), context.i32(), context.i64(), context.i128());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| match ty {
            ty if ty == i128 => Some(TypeConversion::many([i64, i32])),
            ty if ty == i16 => Some(TypeConversion::many([i32])),
            ty if ty == i8 => Some(TypeConversion::many([])),
            ty if ty == i64 => Some(TypeConversion::Drop),
            _ => None,
        });

        assert_eq!(converter.convert_type(&context, i128).as_slice(), [i64, i32]);
        assert_eq!(converter.convert_type(&context, i16).as_slice(), [i32]);
        assert_eq!(converter.convert_type_1_to_1(&context, i16).unwrap(), i32);
        assert!(converter.convert_type(&context, i8).is_empty());
        assert!(converter.convert_type(&context, i64).is_empty());
        assert!(!converter.is_legal_type(&context, i8));
        assert!(converter.is_legal_type(&context, i32));
    }

    #[test]
    fn rejects_many_and_drop_for_1_to_1_queries() {
        let context = Context::default();
        let (i32, i64) = (context.i32(), context.i64());
        let mut many = TypeConverter::new();
        many.add_conversion(move |_, ty| {
            if ty == i32 {
                Some(TypeConversion::Many(smallvec![i64, i64]))
            } else {
                None
            }
        });
        assert!(
            format!("{}", many.convert_type_1_to_1(&context, i32).unwrap_err()).contains("1:1")
        );

        let mut drop = TypeConverter::new();
        drop.add_conversion(move |_, ty| {
            if ty == i32 {
                Some(TypeConversion::Drop)
            } else {
                None
            }
        });
        assert!(
            format!("{}", drop.convert_type_1_to_1(&context, i32).unwrap_err()).contains("1:1")
        );
    }

    /// A type callback is asked about a type once, however many values have it; the clones of a
    /// converter share what it answered, and a clone given another callback asks again.
    #[test]
    fn a_type_is_asked_about_once_until_a_conversion_is_added() {
        let test = Test::new("f", |cx| ([cx.i128(), cx.i128(), cx.i128()], []));
        let context = test.context_rc();
        let (i64, i128) = (context.i64(), context.i128());
        let values = {
            let entry = test.entry_block();
            let entry = entry.borrow();
            entry.arguments().iter().map(|arg| *arg as ValueRef).collect::<Vec<_>>()
        };

        let calls = Rc::new(Cell::new(0));
        let mut converter = TypeConverter::new();
        {
            let calls = Rc::clone(&calls);
            converter.add_conversion(move |_, ty| {
                calls.set(calls.get() + 1);
                (ty == i128).then(|| TypeConversion::many([i64, i64]))
            });
        }

        for value in &values {
            assert_eq!(converter.convert_value(&context, *value).as_slice(), [i64, i64]);
        }
        assert_eq!(calls.get(), 1, "one type, asked about once");

        let mut clone = converter.clone();
        assert_eq!(clone.convert_type(&context, i128).as_slice(), [i64, i64]);
        assert_eq!(calls.get(), 1, "a clone shares the answers");

        clone.add_conversion(|_, _| None);
        assert_eq!(clone.convert_type(&context, i128).as_slice(), [i64, i64]);
        assert_eq!(calls.get(), 2, "a clone given another callback asks again");
        assert_eq!(converter.convert_type(&context, i128).as_slice(), [i64, i64]);
        assert_eq!(calls.get(), 2, "the original keeps its answers");
    }

    /// A conversion added later for a type wins over one added before it; a later callback that
    /// declines leaves the type to the earlier ones.
    #[test]
    fn a_later_conversion_for_the_same_type_wins() {
        let context = Context::default();
        let (i16, i32, i64) = (context.i16(), context.i32(), context.i64());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| (ty == i32).then_some(TypeConversion::One(i64)));
        converter.add_conversion(move |_, ty| (ty == i32).then_some(TypeConversion::One(i16)));
        converter.add_conversion(|_, _| None);

        assert_eq!(converter.convert_type(&context, i32).as_slice(), [i16]);
    }

    /// A value converts as its type does: [TypeConverter::convert_value] asks the callbacks about
    /// the value's type, through the memo, so a type is asked about once whatever value has it.
    #[test]
    fn a_value_converts_as_its_type_does() {
        let test = Test::new("f", |cx| ([cx.i32(), cx.i32(), cx.i64()], []));
        let context = test.context_rc();
        let (i32, i64) = (context.i32(), context.i64());
        let [first, second, third] = {
            let entry = test.entry_block();
            let entry = entry.borrow();
            [0, 1, 2].map(|index| entry.arguments()[index] as ValueRef)
        };
        let calls = Rc::new(Cell::new(0));
        let mut converter = TypeConverter::new();
        {
            let calls = Rc::clone(&calls);
            converter.add_conversion(move |_, ty| {
                calls.set(calls.get() + 1);
                (ty == i32).then(TypeConversion::drop)
            });
        }

        assert!(converter.convert_value(&context, first).is_empty());
        assert!(converter.convert_value(&context, second).is_empty());
        assert_eq!(converter.convert_value(&context, third).as_slice(), [i64]);
        assert_eq!(calls.get(), 2, "two types, each asked about once");
        assert_eq!(converter.convert_value_1_to_1(&context, third).unwrap(), i64);
        assert!(converter.convert_value_1_to_1(&context, first).is_err());
        assert_eq!(calls.get(), 2);
    }

    /// `i128` becomes `(i64, i64)`, the marker `array<i8; 0>` becomes nothing, `i16` becomes
    /// `i32`; function types convert by their parts.
    fn function_type_converter(context: &Context) -> TypeConverter {
        let (i16, i32, i64, i128) = (context.i16(), context.i32(), context.i64(), context.i128());
        let marker = context.array_type(context.i8(), 0);
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| match ty {
            ty if ty == i128 => Some(TypeConversion::many([i64, i64])),
            ty if ty == marker => Some(TypeConversion::Drop),
            ty if ty == i16 => Some(TypeConversion::One(i32)),
            _ => None,
        });
        converter.with_function_types()
    }

    /// A function type converts to the function type of its converted parameters and results,
    /// each flattened, and a function type held by a function type converts too. Without
    /// `with_function_types`, a function type no callback answers for is itself.
    #[test]
    fn a_function_type_converts_by_its_parts() {
        let context = Context::default();
        let (i16, i32, i64, i128) = (context.i16(), context.i32(), context.i64(), context.i128());
        let marker = context.array_type(context.i8(), 0);
        let converter = function_type_converter(&context);
        assert!(converter.converts_function_types());

        let function = context.function_type([i128, marker, i32], [i16, i128]);
        let expected = context.function_type([i64, i64, i32], [i32, i64, i64]);
        assert_eq!(converter.convert_type(&context, function).as_slice(), [expected]);
        assert!(!converter.is_legal_type(&context, function));

        let inner = context.function_type([i128], [marker]);
        let outer = context.function_type([inner, i32], [inner]);
        let inner_converted = context.function_type([i64, i64], []);
        let outer_converted = context.function_type([inner_converted, i32], [inner_converted]);
        assert_eq!(converter.convert_type(&context, outer).as_slice(), [outer_converted]);

        let legal = context.function_type([i32], [i64]);
        assert!(converter.is_legal_type(&context, legal));

        let mut flat = TypeConverter::new();
        flat.add_conversion(move |_, ty| (ty == i128).then(|| TypeConversion::many([i64, i64])));
        assert!(!flat.converts_function_types());
        assert!(flat.is_legal_type(&context, function));
    }

    /// `with_function_types` answers for a function type before the callbacks: a callback that
    /// claims function types is not asked about them. It starts a memo of its own.
    #[test]
    fn function_types_are_answered_before_the_callbacks() {
        let context = Context::default();
        let (i32, i64) = (context.i32(), context.i64());
        let function = context.function_type([i64], []);
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| {
            if ty.is_function() {
                Some(TypeConversion::Drop)
            } else {
                (ty == i64).then_some(TypeConversion::One(i32))
            }
        });
        assert!(converter.convert_type(&context, function).is_empty());
        let converter = converter.with_function_types();
        assert_eq!(
            converter.convert_type(&context, function).as_slice(),
            [context.function_type([i32], [])]
        );
    }

    /// For every signature `s`, the function type of `convert_signature(s)` is what the converter
    /// answers for the function type of `s` (one type), `never` included, under a converter that
    /// drops `never` as a value type.
    #[test]
    fn the_function_type_of_a_converted_signature_is_the_converted_function_type() {
        use crate::{
            dialects::builtin::{attributes::Signature, function_type_of},
            testing::{test_seeds, text::Rng},
        };

        for seed in test_seeds(3) {
            let context = Rc::new(Context::default());
            // A converter that drops `never` must leave a `never` result alone on both sides.
            let mut converter = function_type_converter(&context);
            let never = context.never_type();
            converter.add_conversion(move |_, ty| (ty == never).then_some(TypeConversion::Drop));
            let (i16, i32, i64, i128) =
                (context.i16(), context.i32(), context.i64(), context.i128());
            let marker = context.array_type(context.i8(), 0);
            let nested = context.function_type([i128, marker], [i16]);
            let leaves = [i16, i32, i64, i128, marker, nested];
            let mut rng = Rng(seed | 1);
            for case in 0..500 {
                let pick = |rng: &mut Rng| {
                    let ty = rng.pick(&leaves);
                    if rng.below(6) == 0 {
                        let inner = rng.pick(&leaves);
                        context.function_type([inner, ty], [rng.pick(&leaves)])
                    } else {
                        ty
                    }
                };
                let params = (0..rng.below(5)).map(|_| pick(&mut rng)).collect::<Vec<_>>();
                let results = if rng.below(5) == 0 {
                    alloc::vec![context.never_type()]
                } else {
                    (0..rng.below(4)).map(|_| pick(&mut rng)).collect::<Vec<_>>()
                };
                let signature = Signature::new(&context, params, results);
                let converted = super::super::convert_signature(&context, &signature, &converter);
                let of_converted = function_type_of(&context, &converted.signature);
                let converted_of =
                    converter.convert_type(&context, function_type_of(&context, &signature));
                assert_eq!(
                    converted_of.as_slice(),
                    [of_converted],
                    "seed {seed:#x} (set CORE_TEST_SEED to rerun), case {case}: {signature}"
                );
            }
        }
    }

    mod legality {
        use alloc::{format, string::String, vec::Vec};

        use super::*;
        use crate::{
            AttributeRef, OperationRef,
            attributes::AttrList,
            dialects::{
                builtin::{
                    Function, Module,
                    attributes::{BoolAttr, I32Attr, List, ListAttr, TypeAttr},
                },
                test::TestDialect,
            },
            parse::{ParserConfig, parse},
            testing::text::describe,
        };

        /// `i128` becomes `(i64, i64)`; every other type is itself. The converter is flat: it is
        /// not asked about the types inside a type.
        fn wide_converter(context: &Context) -> TypeConverter {
            let (i64, i128) = (context.i64(), context.i128());
            let mut converter = TypeConverter::new();
            converter.add_conversion(move |_, ty| {
                (ty == i128).then(|| TypeConversion::many([i64, i64]))
            });
            converter
        }

        /// Parses `body` as the body of `builtin.module @m`, in a context with the test dialect.
        fn parse_module(body: &str) -> (Rc<Context>, OperationRef) {
            let context = Rc::new(Context::default());
            context.get_or_register_dialect::<TestDialect>();
            let source = format!("builtin.module public @m {{\n{body}\n}};");
            let module = parse::<Module>(ParserConfig::new(context.clone()), "legality", &source)
                .into_result()
                .unwrap_or_else(|err| panic!("{}", describe(&err)));
            (context, module.as_operation_ref())
        }

        /// The operations nested in `root`, in program order, with what each prints as its first
        /// line.
        fn nested_ops(root: OperationRef) -> Vec<(String, OperationRef)> {
            let mut ops = Vec::new();
            root.borrow().prewalk_all(|op| {
                if op.as_operation_ref() != root {
                    let text = format!("{op}");
                    let line = String::from(text.lines().next().unwrap_or_default());
                    ops.push((line, op.as_operation_ref()));
                }
            });
            ops
        }

        /// The one nested operation whose first printed line starts with `prefix`.
        fn op_starting(root: OperationRef, prefix: &str) -> OperationRef {
            let found = nested_ops(root)
                .into_iter()
                .filter(|(line, _)| line.trim_start().starts_with(prefix))
                .map(|(_, op)| op)
                .collect::<Vec<_>>();
            assert_eq!(found.len(), 1, "one operation starting with '{prefix}'");
            found[0]
        }

        /// An operation is illegal when an operand, a result, the argument of a block of one of
        /// its regions, a `TypeAttr` property or a parameter of a signature property has a type
        /// the converter changes; each counterpart over a legal type is legal.
        #[test]
        fn an_operation_is_illegal_when_a_type_it_has_or_holds_is_illegal() {
            let (context, module) = parse_module(
                "    builtin.global_variable public @wide_global : i128;
    builtin.global_variable public @narrow_global : i64;
    builtin.function public extern(\"C\") @wide_decl(i128);
    builtin.function public extern(\"C\") @narrow_decl(i64);
    builtin.function public extern(\"C\") @f(%0: i128, %1: i64, %2: i64) {
        test.sink %0 : (i128);
        test.sink %1 : (i64);
        %3 = test.join %1, %2;
        %4 = test.low %3;
        builtin.ret;
    };
    builtin.function public extern(\"C\") @g() {
        builtin.ret;
    };",
            );
            let converter = wide_converter(&context);
            let legal = |prefix: &str| converter.is_legal_op(&op_starting(module, prefix).borrow());

            assert!(!legal("test.sink %0"), "an illegal operand");
            assert!(legal("test.sink %1"));
            assert!(!legal("%3 = test.join"), "an illegal result");
            assert!(!legal("%4 = test.low"), "an illegal operand, a legal result");
            assert!(!legal("builtin.global_variable public @wide_global"), "a TypeAttr property");
            assert!(legal("builtin.global_variable public @narrow_global"));
            assert!(!legal("builtin.function public extern(\"C\") @wide_decl"), "a signature");
            assert!(legal("builtin.function public extern(\"C\") @narrow_decl"));
            assert!(
                !legal("builtin.function public extern(\"C\") @f"),
                "a signature and arguments"
            );

            // A block argument alone: `@g` takes nothing, and its entry block is given an `i128`.
            let g = op_starting(module, "builtin.function public extern(\"C\") @g");
            assert!(converter.is_legal_op(&g.borrow()));
            let entry = g.try_downcast_op::<Function>().unwrap().borrow().entry_block();
            context.append_block_argument(entry, context.i128(), Default::default());
            assert!(!converter.is_legal_op(&g.borrow()), "an illegal block argument");
        }

        /// A type held by an attribute is asked about at any depth of attribute: a `TypeAttr`
        /// inside a list makes the operation illegal.
        #[test]
        fn a_type_inside_a_list_attribute_is_asked_about() {
            let (context, module) = parse_module(
                "    builtin.function public extern(\"C\") @f(%0: i64) {
        test.sink %0 : (i64);
        builtin.ret;
    };",
            );
            let converter = wide_converter(&context);
            let mut sink = op_starting(module, "test.sink");
            assert!(converter.is_legal_op(&sink.borrow()));

            let narrow = context.create_attribute::<TypeAttr, _>(context.i64()).as_attribute_ref();
            sink.borrow_mut().set_attribute("narrow", list_of(&context, &[narrow]));
            assert!(converter.is_legal_op(&sink.borrow()), "a list of a legal type");

            let wide = context.create_attribute::<TypeAttr, _>(context.i128()).as_attribute_ref();
            sink.borrow_mut().set_attribute("wide", list_of(&context, &[wide]));
            assert!(!converter.is_legal_op(&sink.borrow()), "a list of an illegal type");
        }

        fn list_of(context: &Rc<Context>, entries: &[AttributeRef]) -> AttributeRef {
            let mut list = AttrList::new();
            for entry in entries {
                list.push_back(entry.borrow().as_attr().as_attr_ref());
            }
            context.create_attribute::<ListAttr, _>(List::from(list)).as_attribute_ref()
        }

        /// The converter is flat: `ptr<i128>` converts to itself under a converter that changes
        /// `i128`, so a value of it, a block argument of it and a signature of it are legal.
        #[test]
        fn a_pointer_to_an_illegal_type_is_legal() {
            let (context, module) = parse_module(
                "    builtin.global_variable public @p : ptr<i128>;
    builtin.function public extern(\"C\") @f(%0: ptr<i128>) {
        test.sink %0 : (ptr<i128>);
        builtin.ret;
    };",
            );
            let converter = wide_converter(&context);
            for (line, op) in nested_ops(module) {
                assert!(converter.is_legal_op(&op.borrow()), "{line}");
            }
        }

        /// The type an attribute's kind fixes is not asked about: an operation whose only `i32`
        /// is an `I32Attr`, and whose only `i1` is a `BoolAttr`, is legal under a converter that
        /// changes `i32` and `i1`.
        #[test]
        fn the_type_an_attribute_kind_fixes_is_not_asked_about() {
            let (context, module) = parse_module(
                "    builtin.function public extern(\"C\") @f(%0: i64, %1: i32) {
        test.sink %0 : (i64);
        test.sink %1 : (i32);
        builtin.ret;
    };",
            );
            let (i1, i32, i64) = (context.i1(), context.i32(), context.i64());
            let mut converter = TypeConverter::new();
            converter.add_conversion(move |_, ty| {
                (ty == i32 || ty == i1).then_some(TypeConversion::One(i64))
            });

            let mut narrow = op_starting(module, "test.sink %0");
            let levels = context.create_attribute::<I32Attr, _>(1i32).as_attribute_ref();
            let flag = context.create_attribute::<BoolAttr, _>(true).as_attribute_ref();
            narrow.borrow_mut().set_attribute("levels", levels);
            narrow.borrow_mut().set_attribute("flag", flag);
            assert!(converter.is_legal_op(&narrow.borrow()));
            assert!(
                !converter.is_legal_op(&op_starting(module, "test.sink %1").borrow()),
                "the converter changes i32"
            );
        }
    }
}
