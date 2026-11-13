use alloc::{format, string::String};

use crate::{
    CallOpInterface, Callable, CallableSymbol, Context, Op, OpOperandRange, OpOperandRangeMut,
    OpParser, OpPrinter, OperationState, Report, SmallVec, Spanned, Type, UnsafeIntrusiveEntityRef,
    derive::operation,
    diagnostics::report,
    dialects::builtin::{
        BuiltinDialect,
        attributes::{AbiParam, DeclaredEffects, Signature, SignatureAttr},
    },
    effects::{
        EffectInstance, EffectIterator, EffectOpInterface, MemoryEffect, MemoryEffectOpInterface,
    },
    parse::{self, Delimiter, ParseResult, ParserError},
    print::AsmPrinter,
    traits::{AnyType, InferTypeOpInterface},
};

pub type CallRef = UnsafeIntrusiveEntityRef<Call>;

/// A direct call of the callable named by `callee`, with the signature the call is made with:
/// `%r = builtin.call @callee(%a, %b) : extern("C") (i32, i32) -> i32;`.
///
/// The call has one argument per parameter of its signature and one result per result of it, of
/// the same types; the results are made from the signature when the call is built or parsed. The
/// callee resolves, through any chain of [FunctionAlias](super::FunctionAlias)es, to a callable
/// whose signature equals the call's: parameter and result types, parameter attributes and calling
/// convention. A callee that names nothing is reported by the check every symbol use gets
/// ([Operation::verify_symbol_refs](crate::Operation::verify_symbol_refs)).
///
/// The call's memory effects are those its canonical callee declares
/// ([CallableOpInterface::declared_effects](crate::CallableOpInterface::declared_effects)): an
/// effect on argument `i` of the callee is an effect on operand `i` of the call, one on result `j`
/// an effect on result `j` of the call. A callee that states no effects, or that does not resolve,
/// makes the call's effects unknown. The callee is resolved on every query.
#[operation(
    dialect = BuiltinDialect,
    implements(CallOpInterface, InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Call {
    /// The callable called: a function, or an alias of one
    #[symbol(callable)]
    callee: crate::SymbolPath,
    /// The signature the call is made with
    #[attr(hidden)]
    signature: SignatureAttr,
    /// The arguments, one per parameter of the signature
    #[operands]
    arguments: AnyType,
    /// The results, one per result of the signature
    #[results]
    returns: AnyType,
}

impl Call {
    /// The effects the canonical callee declares, or [DeclaredEffects::Unknown] when the callee
    /// does not resolve to a callable.
    pub fn callee_declared_effects(&self) -> DeclaredEffects {
        match self.callee().resolve_callable() {
            Ok(callee) => callee.target().borrow().declared_effects(),
            Err(_) => DeclaredEffects::Unknown,
        }
    }
}

/// One result per result of the signature, in order. A call that already has results (the generic
/// form gives them) keeps them, and the verifier checks them against the signature.
impl InferTypeOpInterface for Call {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        if self.op.num_results() != 0 {
            return Ok(());
        }
        let types = self
            .get_signature()
            .results()
            .iter()
            .map(|result| result.ty)
            .collect::<SmallVec<[Type; 2]>>();
        let span = self.op.span();
        let owner = self.op.as_operation_ref();
        for (index, ty) in types.into_iter().enumerate() {
            let result = context.make_result(span, ty, owner, index as u16);
            self.op.results.push(result);
        }
        Ok(())
    }
}

impl CallOpInterface for Call {
    #[inline]
    fn callable_for_callee(&self) -> Callable {
        self.callee().path().into()
    }

    /// Sets the callee to the callable symbol `callable` names in the nearest symbol table.
    ///
    /// # Panics
    ///
    /// Panics if `callable` is not a symbol, if the call is not in a symbol table, or if the path
    /// names no callable symbol there.
    fn set_callee(&mut self, callable: Callable) {
        let path = callable.unwrap_symbol_path();
        let table = self
            .as_operation()
            .nearest_symbol_table()
            .expect("cannot set the callee of a call outside a symbol table");
        let symbol = table
            .borrow()
            .as_symbol_table()
            .expect("the nearest symbol table implements SymbolTable")
            .resolve(&path)
            .expect("the callee must name a symbol");
        let symbol = symbol
            .as_trait_ref::<dyn CallableSymbol>()
            .expect("the callee must name a callable symbol");
        Call::set_callee(self, symbol).expect("the callee is a callable symbol");
    }

    #[inline]
    fn arguments(&self) -> OpOperandRange<'_> {
        self.operands().group(0)
    }

    #[inline]
    fn arguments_mut(&mut self) -> OpOperandRangeMut<'_> {
        self.operands_mut().group_mut(0)
    }
}

impl EffectOpInterface<MemoryEffect> for Call {
    fn effects(&self) -> EffectIterator<MemoryEffect> {
        let DeclaredEffects::Known(declared) = self.callee_declared_effects() else {
            return EffectIterator::new([]);
        };
        let arguments = self.operands().group(0);
        let arguments = arguments.as_slice();
        let results = self.results().group(0);
        let results = results.as_slice();
        EffectIterator::new(declared.into_iter().map(|descriptor| {
            let effect = descriptor.effect;
            // The callee's verifier keeps the indices within its signature; a call whose operands
            // or results disagree with it gets an effect on no value rather than a panic.
            if let Some(index) = descriptor.argument
                && let Some(operand) = arguments.get(usize::from(index))
            {
                EffectInstance::new_for_value(effect, *operand)
            } else if let Some(index) = descriptor.result
                && let Some(result) = results.get(usize::from(index))
            {
                EffectInstance::new_for_value(effect, *result)
            } else {
                EffectInstance::new(effect)
            }
        }))
    }

    fn effects_are_known(&self) -> bool {
        self.callee_declared_effects().is_known()
    }
}

/// A call's arguments and results agree with its signature; its callee resolves, through any chain
/// of aliases, to a callable whose signature equals the call's.
///
/// A callee that is not linked to a symbol is left to
/// [Operation::verify_symbol_refs](crate::Operation::verify_symbol_refs), which reports that it does
/// not resolve.
impl crate::Verify<dyn CallOpInterface> for Call {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let signature = self.get_signature();
        crate::conversion::verify_call_signature_operands_and_results(
            self.as_operation(),
            &signature,
        )?;

        let callee = self.callee();
        if !callee.try_user().is_some_and(|user| user.is_linked()) {
            return Ok(());
        }
        let span = self.as_operation().span();
        let resolved = callee.resolve_callable().map_err(|err| {
            report!(
                message: "invalid operation builtin.call",
                labels: [primary(span, format!("the callee does not resolve to a callable: {err}"))],
            )
        })?;

        let target = resolved.target();
        let target_signature = target.signature();
        if let Some(difference) = signature_difference(&signature, &target_signature) {
            let target_span = target.as_operation_ref().borrow().span();
            return Err(report!(
                message: "invalid operation builtin.call",
                labels: [
                    primary(
                        span,
                        format!(
                            "the call's signature differs from its callee's: {difference}"
                        )
                    ),
                    context(target_span, format!("the callee is declared `{target_signature}`")),
                ],
            ));
        }
        Ok(())
    }
}

/// How the signature of a call, `call`, differs from that of its callee, `callee`: the first
/// difference, in words, or `None` if they are equal.
fn signature_difference(call: &Signature, callee: &Signature) -> Option<String> {
    if call == callee {
        return None;
    }
    if call.calling_convention() != callee.calling_convention() {
        return Some(format!(
            "the call uses the calling convention \"{}\", the callee \"{}\"",
            call.calling_convention().as_str(),
            callee.calling_convention().as_str()
        ));
    }
    if let Some(difference) = list_difference("parameter", call.params(), callee.params()) {
        return Some(difference);
    }
    if let Some(difference) =
        list_difference("result", call.declared_results(), callee.declared_results())
    {
        return Some(difference);
    }
    Some(String::from("the signatures differ"))
}

/// The first difference between the parameters (or results, as `role` says) `call` and `callee`.
fn list_difference(role: &str, call: &[AbiParam], callee: &[AbiParam]) -> Option<String> {
    if call.len() != callee.len() {
        return Some(format!("the call has {} {role}(s), the callee {}", call.len(), callee.len()));
    }
    for (index, (call, callee)) in call.iter().zip(callee).enumerate() {
        if call.ty != callee.ty {
            return Some(format!(
                "{role} {index} has type `{}` in the call, `{}` in the callee",
                call.ty, callee.ty
            ));
        }
        if call.attributes() != callee.attributes() {
            return Some(format!(
                "{role} {index} is `{call}` in the call, `{callee}` in the callee: their \
                 attributes differ"
            ));
        }
    }
    None
}

impl OpPrinter for Call {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::{AttrPrinter, formatter::const_text};

        printer.print_space();
        printer.print_symbol_path(self.callee().path());
        printer.print_operand_list(self.operands().group(0));
        *printer += const_text(" : ");
        self.signature().print(printer);
        let op = self.as_operation();
        if op.has_attributes() {
            printer.print_space();
            printer.print_keyword("attributes");
            printer.print_space();
            printer.print_attribute_dictionary(
                op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }
    }
}

/// Reads `@callee(%a, ..) : signature`, then an optional `attributes {..}`. The arguments take
/// their types from the parameters of the signature.
impl OpParser for Call {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        use crate::attributes::AttrParser;

        let callee = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute("callee", callee);

        let mut arguments = SmallVec::<[parse::UnresolvedOperand; 2]>::default();
        parser.parse_operand_list(&mut arguments, Delimiter::Paren, true, None)?;

        parser.parse_colon()?;
        let signature_start = parser.current_location();
        let signature = <SignatureAttr as AttrParser>::parse(parser)?;
        let params = {
            let signature = signature.borrow();
            let Some(signature) = signature.downcast_ref::<SignatureAttr>() else {
                return Err(ParserError::InvalidAttributeValue {
                    span: signature_start,
                    reason: String::from("expected a signature"),
                });
            };
            signature.params().iter().map(|param| param.ty).collect::<SmallVec<[Type; 2]>>()
        };
        state.add_attribute("signature", signature);

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let mut values = SmallVec::default();
        parser.resolve_operands(state.span, &arguments, &params, &mut values)?;
        state.operands.push(values);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{string::ToString, vec::Vec};

    use super::*;
    use crate::{
        CallableOpInterface, SourceSpan, Usable, Value, ValueRef,
        dialects::builtin::{BuiltinOpBuilder, FunctionRef, attributes::MemoryEffectDescriptor},
        testing::{Test, text::describe},
    };

    /// A module holding `@callee(i32, i32) -> i32`, a declaration, and the test's function
    /// `@caller(i32, i32) -> i32`, whose arguments the tests pass to calls.
    fn caller_and_callee() -> (Test, FunctionRef) {
        let mut test = Test::named("caller").in_module("calls");
        let i32 = test.context().i32();
        let callee = test.define_function("callee", &[i32, i32], &[i32]);
        test.with_function("caller", &[i32, i32], &[i32]);
        (test, callee)
    }

    fn arguments(test: &Test) -> [ValueRef; 2] {
        let entry = test.entry_block();
        let entry = entry.borrow();
        [entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef]
    }

    fn verify(test: &Test) -> Result<(), Report> {
        test.module().borrow().as_operation().recursively_verify()
    }

    #[test]
    fn the_builder_makes_one_result_per_result_of_the_signature() {
        let (mut test, callee) = caller_and_callee();
        let [a, b] = arguments(&test);
        let signature = callee.borrow().get_signature().clone();
        let call = {
            let mut builder = test.function_builder();
            let call =
                builder.call(callee, signature.clone(), [a, b], SourceSpan::UNKNOWN).unwrap();
            let result = call.borrow().returns()[0].borrow().as_value_ref();
            builder.ret([result], SourceSpan::UNKNOWN).unwrap();
            call
        };
        verify(&test).unwrap();

        let call = call.borrow();
        assert_eq!(*call.get_signature(), signature);
        // The builder names the callee by its absolute path
        assert_eq!(call.callee().path().to_string(), "::calls::callee");
        assert_eq!(call.returns().len(), 1);
        assert_eq!(call.returns()[0].borrow().ty(), test.context().i32());
        let interface = call.as_operation().as_trait::<dyn CallOpInterface>().unwrap();
        let passed = interface
            .arguments()
            .iter()
            .map(|operand| operand.borrow().as_value_ref())
            .collect::<Vec<_>>();
        assert_eq!(passed, [a, b]);
        assert!(interface.resolve().and_then(|target| target.as_function()) == Some(callee));
        assert_eq!(interface.callable_for_callee().to_string(), "::calls::callee");
    }

    /// `CallOpInterface::set_callee` relinks the call to the callable the path names.
    #[test]
    fn set_callee_relinks_the_call() {
        let (mut test, callee) = caller_and_callee();
        let i32 = test.context().i32();
        let other = test.define_function("other", &[i32, i32], &[i32]);
        let [a, b] = arguments(&test);
        let signature = callee.borrow().get_signature().clone();
        let mut call = {
            let mut builder = test.function_builder();
            let call = builder.call(callee, signature, [a, b], SourceSpan::UNKNOWN).unwrap();
            let result = call.borrow().returns()[0].borrow().as_value_ref();
            builder.ret([result], SourceSpan::UNKNOWN).unwrap();
            call
        };
        let path = other.borrow().as_operation().as_symbol().unwrap().path();
        CallOpInterface::set_callee(&mut *call.borrow_mut(), path.into());
        verify(&test).unwrap();
        let call = call.borrow();
        assert!(call.resolve().and_then(|target| target.as_function()) == Some(other));
        assert_eq!(callee.borrow().iter_uses().count(), 0);
        assert_eq!(other.borrow().iter_uses().count(), 1);
    }

    #[test]
    fn a_call_to_a_function_with_no_results_has_none() {
        let mut test = Test::named("caller").in_module("calls");
        let unit = test.define_function("unit", &[], &[]);
        test.with_function("caller", &[], &[]);
        let call = {
            let mut builder = test.function_builder();
            let signature = unit.borrow().get_signature().clone();
            let call = builder.call(unit, signature, [], SourceSpan::UNKNOWN).unwrap();
            builder.ret([], SourceSpan::UNKNOWN).unwrap();
            call
        };
        verify(&test).unwrap();
        assert_eq!(call.borrow().as_operation().num_results(), 0);
    }

    /// Builds a call of `@callee` from `@caller` with `signature` and `args`, and returns why the
    /// module does not verify.
    fn call_error(signature: impl FnOnce(&Test) -> Signature, args: usize) -> String {
        let (mut test, callee) = caller_and_callee();
        let signature = signature(&test);
        let args = arguments(&test)[..args].to_vec();
        {
            let mut builder = test.function_builder();
            builder.call(callee, signature, args, SourceSpan::UNKNOWN).unwrap();
            builder.ret_imm(crate::Int::from(0i32), SourceSpan::UNKNOWN).unwrap();
        }
        describe(&verify(&test).expect_err("the call must not verify"))
    }

    fn callee_signature(test: &Test) -> Signature {
        let i32 = test.context().i32();
        Signature::new(&test.context_rc(), [i32, i32], [i32])
    }

    #[test]
    fn an_argument_count_that_disagrees_with_the_signature_is_an_error() {
        let error = call_error(callee_signature, 1);
        assert!(
            error.contains("the call passes 1 argument(s), but its signature has 2 parameter(s)"),
            "{error}"
        );
    }

    #[test]
    fn an_argument_type_that_disagrees_with_the_signature_is_an_error() {
        let error = call_error(
            |test| {
                let (i32, i64) = (test.context().i32(), test.context().i64());
                Signature::new(&test.context_rc(), [i32, i64], [i32])
            },
            2,
        );
        assert!(
            error.contains(
                "argument 1 has type `i32`, but parameter 1 of the signature has type `i64`"
            ),
            "{error}"
        );
    }

    #[test]
    fn a_result_type_that_disagrees_with_the_signature_is_an_error() {
        let (mut test, callee) = caller_and_callee();
        let [a, b] = arguments(&test);
        let signature = callee.borrow().get_signature().clone();
        let call = {
            let mut builder = test.function_builder();
            let call = builder.call(callee, signature, [a, b], SourceSpan::UNKNOWN).unwrap();
            builder.ret_imm(crate::Int::from(0i32), SourceSpan::UNKNOWN).unwrap();
            call
        };
        let i64 = test.context().i64();
        let mut result = call.borrow().returns()[0].borrow().as_value_ref();
        result.borrow_mut().set_type(i64);
        let error = describe(&verify(&test).expect_err("the call must not verify"));
        assert!(
            error.contains("result 0 has type `i64`, but result 0 of the signature has type `i32`"),
            "{error}"
        );
    }

    #[test]
    fn a_call_whose_signature_is_not_its_callees_is_an_error() {
        let error = call_error(
            |test| {
                let i32 = test.context().i32();
                Signature::new(&test.context_rc(), [i32, i32], [])
            },
            2,
        );
        assert!(
            error.contains(
                "the call's signature differs from its callee's: the call has 0 result(s), the \
                 callee 1"
            ),
            "{error}"
        );
    }

    /// Each declared effect becomes an effect on the operand or result it names, or on no value;
    /// an index the call's operands or results do not reach gives an effect on no value.
    #[test]
    fn declared_effects_become_effects_on_the_calls_operands_and_results() {
        let (mut test, mut callee) = caller_and_callee();
        let [a, b] = arguments(&test);
        let signature = callee.borrow().get_signature().clone();
        callee.borrow_mut().set_memory_effects(DeclaredEffects::from([
            MemoryEffectDescriptor::on_argument(MemoryEffect::Read, 1),
            MemoryEffectDescriptor::on_result(MemoryEffect::Allocate, 0),
            MemoryEffectDescriptor::new(MemoryEffect::Write),
            MemoryEffectDescriptor::on_argument(MemoryEffect::Free, 7),
        ]));
        let call = {
            let mut builder = test.function_builder();
            let call = builder.call(callee, signature, [a, b], SourceSpan::UNKNOWN).unwrap();
            let result = call.borrow().returns()[0].borrow().as_value_ref();
            builder.ret([result], SourceSpan::UNKNOWN).unwrap();
            call
        };
        let call = call.borrow();
        let result = call.returns()[0].borrow().as_value_ref();
        assert!(call.effects_are_known());
        let effects = call
            .effects()
            .map(|effect| (*effect.effect(), effect.value()))
            .collect::<Vec<_>>();
        assert_eq!(
            effects,
            [
                (MemoryEffect::Read, Some(b)),
                (MemoryEffect::Allocate, Some(result)),
                (MemoryEffect::Write, None),
                (MemoryEffect::Free, None),
            ]
        );
    }

    #[test]
    fn a_callee_that_states_no_effects_makes_the_calls_effects_unknown() {
        let (mut test, mut callee) = caller_and_callee();
        let [a, b] = arguments(&test);
        let signature = callee.borrow().get_signature().clone();
        let call_ref = {
            let mut builder = test.function_builder();
            builder.call(callee, signature, [a, b], SourceSpan::UNKNOWN).unwrap()
        };
        assert_eq!(callee.borrow().declared_effects(), DeclaredEffects::Unknown);
        let call = call_ref.borrow();
        assert!(!call.effects_are_known());
        assert!(!call.as_operation().is_memory_effect_free());
        assert!(call.as_operation().get_effects_recursively::<MemoryEffect>().is_none());

        drop(call);
        callee.borrow_mut().set_memory_effects(DeclaredEffects::none());
        let call = call_ref.borrow();
        assert!(call.effects_are_known());
        assert!(call.as_operation().is_memory_effect_free());
    }
}
