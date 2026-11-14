use alloc::{format, vec::Vec};
use core::ops::Range;

use smallvec::SmallVec;

use super::TypeConverter;
use crate::{
    BlockRef, CallOpInterface, Context, Operation, Report, Type, Value, ValueRef,
    dialects::builtin::attributes::{AbiParam, Signature},
};

/// How the arguments of a block change.
///
/// Each old argument is mapped once: to new arguments ([Self::add_inputs]), or to existing values
/// ([Self::remap_input]; with none, it is dropped). New arguments that replace nothing are added
/// with [Self::add_new_inputs]. The new arguments are in the order of the `add_inputs` and
/// `add_new_inputs` calls. A conversion is applied with
/// [ConversionPatternRewriter::apply_signature_conversion].
///
/// [ConversionPatternRewriter::apply_signature_conversion]:
///     super::ConversionPatternRewriter::apply_signature_conversion
#[derive(Clone, Debug)]
pub struct SignatureConversion {
    converted_types: SmallVec<[Type; 4]>,
    /// For each new argument, the old argument it comes from, if any.
    origins: SmallVec<[Option<usize>; 4]>,
    mappings: Vec<Option<InputMapping>>,
}

/// What an old argument becomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputMapping {
    /// becomes the new arguments `new_index .. new_index + count`
    Arguments {
        /// The first of the new arguments.
        new_index: usize,
        /// How many new arguments, at least one.
        count: usize,
    },
    /// is replaced by existing values (none: dropped)
    Values(SmallVec<[ValueRef; 2]>),
}

impl SignatureConversion {
    /// A conversion of `num_inputs` old arguments, none of them mapped yet.
    pub fn new(num_inputs: usize) -> Self {
        Self {
            converted_types: SmallVec::new(),
            origins: SmallVec::new(),
            mappings: alloc::vec![None; num_inputs],
        }
    }

    /// The number of old arguments.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.mappings.len()
    }

    /// The old argument becomes new arguments of these types (at least one), placed next.
    ///
    /// An error when `old_index` is not an old argument, is already mapped, or `types` is empty
    /// (an argument that becomes no value is dropped with [Self::remap_input]).
    pub fn add_inputs(&mut self, old_index: usize, types: &[Type]) -> Result<(), Report> {
        self.check_unmapped(old_index)?;
        if types.is_empty() {
            return Err(Report::msg(format!(
                "old argument {old_index} must become at least one new argument; an argument that \
                 becomes no value is remapped to none"
            )));
        }
        let new_index = self.converted_types.len();
        self.converted_types.extend_from_slice(types);
        self.origins.extend(core::iter::repeat_n(Some(old_index), types.len()));
        self.mappings[old_index] = Some(InputMapping::Arguments {
            new_index,
            count: types.len(),
        });
        Ok(())
    }

    /// New arguments that replace nothing, placed next.
    pub fn add_new_inputs(&mut self, types: &[Type]) {
        self.converted_types.extend_from_slice(types);
        self.origins.extend(core::iter::repeat_n(None, types.len()));
    }

    /// The old argument is replaced by these values; with none it is dropped.
    ///
    /// The values stand for the argument as given: they are not checked against a type
    /// converter, and they must be available at the start of the block
    /// ([super::ConversionPatternRewriter::apply_signature_conversion] answers an error for a value
    /// defined in the block itself or for one of its arguments). An error when `old_index` is not
    /// an old argument or is already mapped.
    pub fn remap_input(&mut self, old_index: usize, values: &[ValueRef]) -> Result<(), Report> {
        self.check_unmapped(old_index)?;
        self.mappings[old_index] = Some(InputMapping::Values(values.iter().copied().collect()));
        Ok(())
    }

    /// The types of the new arguments, in order.
    #[inline]
    pub fn converted_types(&self) -> &[Type] {
        &self.converted_types
    }

    /// What old argument `old_index` becomes; `None` while it is not mapped, or when there is no
    /// such argument.
    pub fn mapping(&self, old_index: usize) -> Option<&InputMapping> {
        self.mappings.get(old_index).and_then(Option::as_ref)
    }

    /// The old argument new argument `new_index` comes from; `None` for one that replaces nothing.
    pub(super) fn origin(&self, new_index: usize) -> Option<usize> {
        self.origins.get(new_index).copied().flatten()
    }

    /// Each argument of `block` through the converter: several types become `add_inputs`, none
    /// `remap_input(i, &[])`.
    ///
    /// An argument is asked about with [TypeConverter::convert_value]; one that converts to its
    /// own type becomes one new argument of that type.
    pub fn for_block(block: BlockRef, converter: &TypeConverter) -> Result<Self, Report> {
        let context = block.borrow().context_rc();
        let arguments = block.borrow().argument_values().collect::<SmallVec<[ValueRef; 4]>>();
        let mut conversion = Self::new(arguments.len());
        for (index, argument) in arguments.into_iter().enumerate() {
            let types = converter.convert_value(&context, argument);
            if types.is_empty() {
                conversion.remap_input(index, &[])?;
            } else {
                conversion.add_inputs(index, &types)?;
            }
        }
        Ok(conversion)
    }

    /// Whether applying this conversion to `block` would change nothing: each argument becomes
    /// one new argument of its own type, in place, and there are no others.
    pub fn is_identity(&self, block: BlockRef) -> bool {
        let block = block.borrow();
        let types = block.argument_types().collect::<SmallVec<[Type; 4]>>();
        types.len() == self.mappings.len()
            && self.converted_types.as_slice() == types.as_slice()
            && self.mappings.iter().enumerate().all(|(index, mapping)| {
                *mapping
                    == Some(InputMapping::Arguments {
                        new_index: index,
                        count: 1,
                    })
            })
    }

    fn check_unmapped(&self, old_index: usize) -> Result<(), Report> {
        match self.mappings.get(old_index) {
            None => Err(Report::msg(format!(
                "old argument {old_index} does not exist: the conversion has {} old arguments",
                self.mappings.len()
            ))),
            Some(Some(_)) => {
                Err(Report::msg(format!("old argument {old_index} is already mapped")))
            }
            Some(None) => Ok(()),
        }
    }
}

/// A signature converted by type ([convert_signature]): the new signature, and for each old
/// parameter and each old result the indices of the new ones it became.
///
/// `params[i]` is the range of new parameters old parameter `i` became (empty for one that became
/// none), and `results[j]` the same for old result `j`. The results are those a call produces
/// ([Signature::results]): a signature that never returns has none, and neither does its
/// conversion. The ranges are contiguous and in order, as [remap_declared_effects] and
/// [SignatureConversion::add_inputs] take them.
///
/// [remap_declared_effects]: crate::dialects::builtin::attributes::remap_declared_effects
#[derive(Clone, Debug)]
pub struct ConvertedSignature {
    /// The converted signature.
    pub signature: Signature,
    /// For each old parameter, the new parameters it became.
    pub params: Vec<Range<u32>>,
    /// For each old result, the new results it became.
    pub results: Vec<Range<u32>>,
}

/// Each parameter and result becomes one `AbiParam` per type the converter gives for it. One that stays
/// one value keeps its attributes; the parts of one that becomes several carry none. The calling
/// convention is kept.
///
/// Each type is asked about with [TypeConverter::convert_type]: a signature is converted by type,
/// so equal signatures convert to equal signatures, and a callable and every call of it agree. A
/// parameter or result that becomes no type disappears, and one that becomes one type keeps its
/// attributes on the new type. A `never` result (a function that never returns) stays `never`,
/// whatever the converter says about it, as it does in the function types of a converter made
/// with [TypeConverter::with_function_types]; so the function type of the converted signature is
/// what such a converter answers for the function type of `signature`.
pub fn convert_signature(
    context: &Context,
    signature: &Signature,
    converter: &TypeConverter,
) -> ConvertedSignature {
    let (params, param_ranges) =
        convert_abi_params(signature.params(), |ty| converter.convert_type(context, ty));
    let (results, mut result_ranges) = convert_abi_params(signature.declared_results(), |ty| {
        converter.convert_result_type(context, ty)
    });
    if signature.never_returns() {
        // The declared `never` is not a value a call produces.
        result_ranges.clear();
    }
    ConvertedSignature {
        signature: Signature::from_parts(signature.calling_convention(), params, results),
        params: param_ranges,
        results: result_ranges,
    }
}

/// Each of `params` converted with `convert`: the new parameters, and the range of new ones each
/// old one became.
fn convert_abi_params(
    params: &[AbiParam],
    convert: impl Fn(Type) -> SmallVec<[Type; 2]>,
) -> (Vec<AbiParam>, Vec<Range<u32>>) {
    let mut converted = Vec::with_capacity(params.len());
    let mut ranges = Vec::with_capacity(params.len());
    for param in params {
        let start = new_index(converted.len());
        match convert(param.ty).as_slice() {
            [ty] => {
                let mut param = param.clone();
                param.ty = *ty;
                converted.push(param);
            }
            types => converted.extend(types.iter().copied().map(AbiParam::new)),
        }
        ranges.push(start..new_index(converted.len()));
    }
    (converted, ranges)
}

/// `index` as the index of a parameter or result of a signature.
fn new_index(index: usize) -> u32 {
    u32::try_from(index).expect("system limit: a signature has at most u32::MAX parameters")
}

/// Verify that the arguments and results of `op`, a call-like operation, agree with `signature`
/// in count and type: the signature the call carries, or the one of its resolved callee.
///
/// An operation that does not implement [CallOpInterface] is accepted.
pub fn verify_call_signature_operands_and_results(
    op: &Operation,
    signature: &Signature,
) -> Result<(), Report> {
    use crate::{Spanned, diagnostics::report};

    let Some(call) = op.as_trait::<dyn CallOpInterface>() else {
        return Ok(());
    };
    let span = op.span();
    let invalid = |label: alloc::string::String| {
        report!(
            message: (format!("invalid operation {}", op.name())),
            labels: [primary(span, label)],
        )
    };

    let arguments = call.arguments();
    if arguments.len() != signature.params().len() {
        return Err(invalid(format!(
            "the call passes {} argument(s), but its signature has {} parameter(s)",
            arguments.len(),
            signature.params().len()
        )));
    }
    for (index, (argument, param)) in arguments.iter().zip(signature.params()).enumerate() {
        let argument_ty = argument.borrow().ty();
        if argument_ty != param.ty {
            return Err(invalid(format!(
                "argument {index} has type `{argument_ty}`, but parameter {index} of the \
                 signature has type `{}`",
                param.ty
            )));
        }
    }

    let results = signature.results();
    if op.num_results() != results.len() {
        return Err(invalid(format!(
            "the call has {} result(s), but its signature has {}",
            op.num_results(),
            results.len()
        )));
    }
    for (index, (result, expected)) in op.results().iter().zip(results.iter()).enumerate() {
        let result_ty = result.borrow().ty();
        if result_ty != expected.ty {
            return Err(invalid(format!(
                "result {index} has type `{result_ty}`, but result {index} of the signature has \
                 type `{}`",
                expected.ty
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec};

    use super::*;
    use crate::{Context, SourceSpan, conversion::ConversionPatternRewriter, testing::Test};

    fn i32_to_i64_converter(context: &Context) -> TypeConverter {
        let (from, to) = (context.i32(), context.i64());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| {
            if ty == from {
                Some(super::super::TypeConversion::One(to))
            } else {
                None
            }
        });
        converter
    }

    /// Every argument of `block` under the `i32` to `i64` converter, applied by a rewriter.
    fn convert_block(block: BlockRef, context: &Context) -> SmallVec<[ValueRef; 4]> {
        let converter = i32_to_i64_converter(context);
        let conversion = SignatureConversion::for_block(block, &converter).unwrap();
        let mut rewriter = ConversionPatternRewriter::new_with_type_converter(
            block.borrow().context_rc(),
            Some(converter),
        );
        rewriter.apply_signature_conversion(block, &conversion).unwrap()
    }

    #[test]
    fn converts_the_arguments_of_an_entry_block() {
        let mut test = Test::named("converts_the_arguments_of_an_entry_block");
        let (i32, i64) = (test.context().i32(), test.context().i64());
        test.with_function("converts_the_arguments_of_an_entry_block", &[i32, i64], &[]);
        let block = test.entry_block();

        let new = convert_block(block, test.context());

        let types = block.borrow().argument_types().collect::<Vec<_>>();
        assert_eq!(types, vec![i64, i64]);
        assert_eq!(new.as_slice(), block.borrow().argument_values().collect::<Vec<_>>());
    }

    #[test]
    fn converts_the_arguments_of_a_block_after_the_entry() {
        let mut test = Test::new("converts_the_arguments_of_a_block_after_the_entry", |_| ([], []));
        let (i32, i64) = (test.context().i32(), test.context().i64());
        let block = {
            let mut builder = test.function_builder();
            let block = builder.create_block();
            builder.append_block_param(block, i32, SourceSpan::UNKNOWN);
            block
        };

        convert_block(block, test.context());

        let types = block.borrow().argument_types().collect::<Vec<_>>();
        assert_eq!(types, vec![i64]);
    }

    /// An argument whose type converts to several types becomes that many arguments, and one whose
    /// type converts to none is dropped.
    #[test]
    fn an_argument_becomes_several_or_none() {
        let mut test = Test::named("an_argument_becomes_several_or_none");
        let (i16, i32, i64) = (test.context().i16(), test.context().i32(), test.context().i64());
        test.with_function("an_argument_becomes_several_or_none", &[i32, i16], &[]);
        let block = test.entry_block();
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| match ty {
            ty if ty == i32 => Some(super::super::TypeConversion::many([i64, i64])),
            ty if ty == i16 => Some(super::super::TypeConversion::Drop),
            _ => None,
        });

        let mut conversion = SignatureConversion::for_block(block, &converter).unwrap();

        assert_eq!(conversion.converted_types(), [i64, i64]);
        assert_eq!(
            conversion.mapping(0),
            Some(&InputMapping::Arguments {
                new_index: 0,
                count: 2
            })
        );
        assert_eq!(conversion.mapping(1), Some(&InputMapping::Values(SmallVec::new())));
        assert_eq!(
            format!("{}", conversion.add_inputs(0, &[i64]).unwrap_err()),
            "old argument 0 is already mapped"
        );
    }

    /// `i128` becomes `(i64, i64)`, the marker `array<i8; 0>` becomes nothing, `i16` becomes
    /// `i32`, and every other type is itself.
    fn by_type_converter(context: &Context) -> TypeConverter {
        let (i16, i32, i64, i128) = (context.i16(), context.i32(), context.i64(), context.i128());
        let marker = context.array_type(context.i8(), 0);
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| match ty {
            ty if ty == i128 => Some(super::super::TypeConversion::many([i64, i64])),
            ty if ty == marker => Some(super::super::TypeConversion::Drop),
            ty if ty == i16 => Some(super::super::TypeConversion::One(i32)),
            _ => None,
        });
        converter
    }

    fn ranges(ranges: &[Range<u32>]) -> Vec<(u32, u32)> {
        ranges.iter().map(|range| (range.start, range.end)).collect()
    }

    /// A parameter or result that stays keeps its attributes; one that changes one to one keeps
    /// them on its new type; the parts of one that splits carry none; one that vanishes is gone.
    /// The calling convention is kept, and the ranges say where each old one went.
    #[test]
    fn a_signature_is_converted_by_type() {
        let context = alloc::rc::Rc::new(Context::default());
        let (i16, i32, i64, i128) = (context.i16(), context.i32(), context.i64(), context.i128());
        let marker = context.array_type(context.i8(), 0);
        let signature = Signature::from_parts(
            crate::CallConv::new("fast"),
            vec![
                AbiParam::zext(i32, &context),
                AbiParam::sext(i16, &context),
                AbiParam::zext(i128, &context),
                AbiParam::zext(marker, &context),
                AbiParam::new(i32),
            ],
            vec![AbiParam::new(i128), AbiParam::new(marker), AbiParam::sext(i16, &context)],
        );

        let converted = convert_signature(&context, &signature, &by_type_converter(&context));

        let expected = Signature::from_parts(
            crate::CallConv::new("fast"),
            vec![
                AbiParam::zext(i32, &context),
                AbiParam::sext(i32, &context),
                AbiParam::new(i64),
                AbiParam::new(i64),
                AbiParam::new(i32),
            ],
            vec![AbiParam::new(i64), AbiParam::new(i64), AbiParam::sext(i32, &context)],
        );
        assert_eq!(converted.signature, expected);
        assert!(converted.signature.params()[2].attributes().is_empty());
        assert_eq!(ranges(&converted.params), [(0, 1), (1, 2), (2, 4), (4, 4), (4, 5)]);
        assert_eq!(ranges(&converted.results), [(0, 2), (2, 2), (2, 3)]);
    }

    /// Equal signatures convert to equal signatures, and a signature of legal types to itself.
    #[test]
    fn equal_signatures_convert_to_equal_signatures() {
        let context = alloc::rc::Rc::new(Context::default());
        let converter = by_type_converter(&context);
        let make = || Signature::new(&context, [context.i128(), context.i16()], [context.i128()]);
        assert_eq!(
            convert_signature(&context, &make(), &converter).signature,
            convert_signature(&context, &make(), &converter).signature
        );
        let legal = Signature::new(&context, [context.i32()], [context.i64()]);
        assert_eq!(convert_signature(&context, &legal, &converter).signature, legal);
    }

    /// A `never` result stays `never`, even under a converter that drops `never`, and a signature
    /// that never returns has no results to range over.
    #[test]
    fn a_never_result_stays_never() {
        let context = alloc::rc::Rc::new(Context::default());
        let never = context.never_type();
        let mut converter = by_type_converter(&context);
        converter.add_conversion(move |_, ty| {
            (ty == never).then_some(super::super::TypeConversion::Drop)
        });
        let signature = Signature::new(&context, [context.i128()], [never]);

        let converted = convert_signature(&context, &signature, &converter);

        assert_eq!(
            converted.signature,
            Signature::new(&context, [context.i64(), context.i64()], [never])
        );
        assert!(converted.signature.never_returns());
        assert_eq!(ranges(&converted.params), [(0, 2)]);
        assert!(converted.results.is_empty());
    }
}
