use alloc::{format, string::ToString, vec::Vec};

use smallvec::SmallVec;

use super::TypeConverter;
use crate::{
    BlockRef, CallOpInterface, CallableOpInterface, Context, Operation, OperationRef, Report, Type,
    Value, ValueRef, dialects::builtin::attributes::Signature,
};

/// Describes how a block/callable signature maps original inputs to converted inputs.
///
/// The current utilities support applying 1:1 block argument type updates in place. The mapping
/// model also records replace/drop decisions for future or dialect-specific signature rewrites,
/// but callers using those mappings must rewrite the affected block predecessors, call operands,
/// and region successor operands themselves.
#[derive(Debug)]
pub struct SignatureConversion {
    converted_types: SmallVec<[Type; 4]>,
    mappings: Vec<InputMapping>,
}

impl SignatureConversion {
    /// Create an empty signature conversion.
    pub fn new() -> Self {
        Self {
            converted_types: SmallVec::new(),
            mappings: Vec::new(),
        }
    }

    /// Build a 1:1 signature conversion for a block's current arguments.
    ///
    /// Each block argument is converted with [`TypeConverter::convert_value_1_to_1`]. The result
    /// can be applied in place with [`Self::apply_to_block_arguments_1_to_1`].
    pub fn for_block_1_to_1(
        block: BlockRef,
        type_converter: &TypeConverter,
    ) -> Result<Self, Report> {
        let context = block.borrow().context_rc();
        let arguments = block
            .borrow()
            .arguments()
            .iter()
            .copied()
            .map(|arg| arg as ValueRef)
            .collect::<SmallVec<[ValueRef; 4]>>();

        let mut conversion = Self::new();
        for (index, argument) in arguments.into_iter().enumerate() {
            conversion.keep_input(index, type_converter.convert_value_1_to_1(&context, argument)?);
        }
        Ok(conversion)
    }

    /// Return the converted input types in new signature order.
    #[inline]
    pub fn converted_types(&self) -> &[Type] {
        &self.converted_types
    }

    /// Return mappings from old inputs to the converted signature.
    #[inline]
    pub fn mappings(&self) -> &[InputMapping] {
        &self.mappings
    }

    /// Keep old input `old_index` as one converted input with `new_type`.
    pub fn keep_input(&mut self, old_index: usize, new_type: Type) -> &mut Self {
        let new_index = self.converted_types.len();
        self.converted_types.push(new_type);
        self.mappings.push(InputMapping::Keep {
            old_index,
            new_index,
            new_count: 1,
        });
        self
    }

    /// Replace old input `old_index` with existing values.
    ///
    /// This records the mapping only. Generic in-place block argument conversion does not support
    /// replacement mappings, so dialect-specific rewrites must update uses and control-flow
    /// operands consistently before relying on this mapping.
    pub fn replace_input(
        &mut self,
        old_index: usize,
        values: impl IntoIterator<Item = ValueRef>,
    ) -> &mut Self {
        self.mappings.push(InputMapping::Replace {
            old_index,
            values: values.into_iter().collect(),
        });
        self
    }

    /// Drop old input `old_index` from the converted signature.
    ///
    /// This records the mapping only. Dropping a live block argument or callable parameter also
    /// requires updating all branch/call/region successor operands that feed it.
    pub fn drop_input(&mut self, old_index: usize) -> &mut Self {
        self.mappings.push(InputMapping::Drop { old_index });
        self
    }

    /// Apply this conversion to block argument types in place.
    ///
    /// This is intentionally limited to 1:1 conversions. 1:N and drop conversions need a
    /// replacement block plus successor rewrites, which should be implemented when a concrete
    /// lowering needs it.
    pub fn apply_to_block_arguments_1_to_1(&self, block: BlockRef) -> Result<bool, Report> {
        let arguments = block.borrow().arguments().iter().copied().collect::<SmallVec<[_; 4]>>();
        if arguments.len() != self.mappings.len() {
            return Err(Report::msg(format!(
                "cannot apply signature conversion to block: expected {} input mappings for the \
                 block arguments, got {}",
                arguments.len(),
                self.mappings.len()
            )));
        }
        if arguments.len() != self.converted_types.len() {
            return Err(Report::msg(format!(
                "cannot apply signature conversion to block: expected {} converted inputs for the \
                 block arguments, got {}",
                arguments.len(),
                self.converted_types.len()
            )));
        }

        for (expected_old_index, mapping) in self.mappings.iter().enumerate() {
            match mapping {
                InputMapping::Keep {
                    old_index,
                    new_index,
                    new_count,
                } if *old_index == expected_old_index
                    && *new_index == expected_old_index
                    && *new_count == 1 => {}
                InputMapping::Keep { .. } => {
                    return Err(Report::msg(
                        "cannot apply non-identity input ordering to block arguments in place",
                    ));
                }
                InputMapping::Replace { .. } | InputMapping::Drop { .. } => {
                    return Err(Report::msg(
                        "cannot apply replace/drop input mapping to block arguments in place",
                    ));
                }
            }
        }

        let mut changed = false;
        for (mut argument, ty) in arguments.into_iter().zip(self.converted_types.iter()) {
            if argument.borrow().ty() != *ty {
                argument.borrow_mut().set_type(*ty);
                changed = true;
            }
        }
        Ok(changed)
    }
}

impl Default for SignatureConversion {
    fn default() -> Self {
        Self::new()
    }
}

/// Mapping from one original input to the converted signature.
#[derive(Clone, Debug)]
pub enum InputMapping {
    /// Preserve one original input as one converted input.
    Keep {
        /// Index of the original input.
        old_index: usize,
        /// First index of this input in the converted signature.
        new_index: usize,
        /// Number of converted inputs produced for this original input.
        new_count: usize,
    },
    /// Replace an original input with existing values.
    Replace {
        /// Index of the original input.
        old_index: usize,
        /// Values that replace the original input.
        values: SmallVec<[ValueRef; 2]>,
    },
    /// Remove an original input from the converted signature.
    Drop {
        /// Index of the original input.
        old_index: usize,
    },
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
    use crate::{Context, SourceSpan, dialects::builtin::attributes::Signature, testing::Test};

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

    #[test]
    fn applies_1_to_1_block_argument_conversion_in_place() {
        let mut test = Test::named("applies_1_to_1_block_argument_conversion_in_place");
        let (i32, i64) = (test.context().i32(), test.context().i64());
        test.with_function("applies_1_to_1_block_argument_conversion_in_place", &[i32, i64], &[]);
        let block = test.entry_block();

        let conversion =
            SignatureConversion::for_block_1_to_1(block, &i32_to_i64_converter(test.context()))
                .expect("signature conversion failed");

        assert!(conversion.apply_to_block_arguments_1_to_1(block).unwrap());
        let types = block.borrow().argument_types().collect::<Vec<_>>();
        assert_eq!(types, vec![i64, i64]);
    }

    #[test]
    fn applies_1_to_1_non_entry_block_argument_conversion_in_place() {
        let mut test =
            Test::new("applies_1_to_1_non_entry_block_argument_conversion_in_place", |_| ([], []));
        let (i32, i64) = (test.context().i32(), test.context().i64());
        let block = {
            let mut builder = test.function_builder();
            let block = builder.create_block();
            builder.append_block_param(block, i32, SourceSpan::UNKNOWN);
            block
        };

        let conversion =
            SignatureConversion::for_block_1_to_1(block, &i32_to_i64_converter(test.context()))
                .expect("signature conversion failed");

        assert!(conversion.apply_to_block_arguments_1_to_1(block).unwrap());
        let types = block.borrow().argument_types().collect::<Vec<_>>();
        assert_eq!(types, vec![i64]);
    }

    #[test]
    fn rejects_non_1_to_1_block_argument_conversion() {
        let mut test = Test::named("rejects_non_1_to_1_block_argument_conversion");
        let i32 = test.context().i32();
        test.with_function("rejects_non_1_to_1_block_argument_conversion", &[i32], &[]);
        let block = test.entry_block();
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| {
            if ty == i32 {
                Some(super::super::TypeConversion::Drop)
            } else {
                None
            }
        });

        let err = SignatureConversion::for_block_1_to_1(block, &converter).unwrap_err();
        assert!(format!("{err}").contains("1:1"));
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
