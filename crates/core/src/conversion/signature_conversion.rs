use alloc::{format, string::ToString, vec::Vec};

use smallvec::SmallVec;

use super::TypeConverter;
use crate::{
    BlockRef, CallOpInterface, CallableOpInterface, Context, Operation, OperationRef, Report, Type,
    Value, ValueRef, dialects::builtin::attributes::Signature,
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
    /// converter, and they must be available at the start of the block. An error when
    /// `old_index` is not an old argument or is already mapped.
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

/// Convert a builtin ABI signature of `context` 1:1, preserving ABI metadata on each
/// parameter/result.
///
/// This updates only parameter and result types. Calling convention and other ABI metadata remain
/// unchanged.
pub fn convert_signature_1_to_1(
    context: &Context,
    signature: &Signature,
    type_converter: &TypeConverter,
) -> Result<Signature, Report> {
    let mut converted = signature.clone();
    for param in converted.params_mut().iter_mut() {
        param.ty = type_converter.convert_type_1_to_1(context, param.ty)?;
    }
    for result in converted.declared_results_mut().iter_mut() {
        result.ty = type_converter.convert_type_1_to_1(context, result.ty)?;
    }
    Ok(converted)
}

/// Return the converted callable signature for `op`, if it implements `CallableOpInterface`.
///
/// The operation is not mutated. Concrete conversion patterns can use the returned signature to
/// rewrite dialect-specific callable storage.
pub fn converted_callable_signature_1_to_1(
    op: OperationRef,
    type_converter: &TypeConverter,
) -> Result<Option<Signature>, Report> {
    let op = op.borrow();
    let Some(callable) = op.as_trait::<dyn CallableOpInterface>() else {
        return Ok(None);
    };
    convert_signature_1_to_1(op.context(), &callable.signature(), type_converter).map(Some)
}

/// Resolve a call-like operation and return the converted callee signature.
///
/// This helper intentionally does not rewrite call operands, result values, or callee attributes.
/// Different dialects store those pieces differently, so concrete conversion patterns should use
/// this to drive operation-specific rewrites.
pub fn converted_resolved_call_signature_1_to_1(
    op: OperationRef,
    type_converter: &TypeConverter,
) -> Result<Option<Signature>, Report> {
    let op = op.borrow();
    let Some(call) = op.as_trait::<dyn CallOpInterface>() else {
        return Ok(None);
    };
    let signature = resolve_call_signature(op.name().to_string(), call)?;
    convert_signature_1_to_1(op.context(), &signature, type_converter).map(Some)
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

fn resolve_call_signature(
    op_name: alloc::string::String,
    call: &dyn CallOpInterface,
) -> Result<Signature, Report> {
    let callee = call.callable_for_callee();
    let Some(resolved) = call.resolve() else {
        return Err(Report::msg(format!(
            "operation '{op_name}' references unresolved callee '{callee}'"
        )));
    };
    Ok(resolved.signature())
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec};

    use super::*;
    use crate::{
        Context, SourceSpan, conversion::ConversionPatternRewriter,
        dialects::builtin::attributes::Signature, testing::Test,
    };

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

    #[test]
    fn converts_builtin_signature_1_to_1() {
        let context = alloc::rc::Rc::new(Context::default());
        let signature = Signature::new(&context, [context.i32()], [context.i32()]);

        let converted =
            convert_signature_1_to_1(&context, &signature, &i32_to_i64_converter(&context))
                .unwrap();

        assert_eq!(converted.params()[0].ty, context.i64());
        assert_eq!(converted.results()[0].ty, context.i64());
        assert_eq!(converted.calling_convention(), signature.calling_convention());
    }

    #[test]
    fn converts_callable_signature_1_to_1() {
        let mut test = Test::named("converts_callable_signature_1_to_1");
        let i32 = test.context().i32();
        test.with_function("converts_callable_signature_1_to_1", &[i32], &[i32]);

        let converted = converted_callable_signature_1_to_1(
            test.function().as_operation_ref(),
            &i32_to_i64_converter(test.context()),
        )
        .unwrap()
        .expect("function should implement CallableOpInterface");

        assert_eq!(converted.params()[0].ty, test.context().i64());
        assert_eq!(converted.results()[0].ty, test.context().i64());
    }
}
