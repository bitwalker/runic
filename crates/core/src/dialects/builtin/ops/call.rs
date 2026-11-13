use alloc::{format, rc::Rc, string::String};

use crate::{
    CallOpInterface, Callable, CallableSymbol, Context, Op, OpOperandRange, OpOperandRangeMut,
    OpParser, OpPrinter, OperationState, Report, SmallVec, Spanned, Type, UnsafeIntrusiveEntityRef,
    Value,
    derive::{EffectOpInterface, operation},
    diagnostics::report,
    dialects::builtin::{
        BuiltinDialect,
        attributes::{AbiParam, DeclaredEffects, Signature, SignatureAttr, TypeAttr},
    },
    effects::{
        AlwaysSpeculatable, ConditionallySpeculatable, EffectInstance, EffectIterator,
        EffectOpInterface, MemoryEffect, MemoryEffectOpInterface, Pure, Speculatability,
    },
    parse::{self, Delimiter, ParseResult, ParserError},
    patterns::RewritePatternSet,
    print::AsmPrinter,
    traits::{AnyFunction, AnyType, Canonicalizable, InferTypeOpInterface},
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
        let callee = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute("callee", callee);

        let mut arguments = SmallVec::<[parse::UnresolvedOperand; 2]>::default();
        parser.parse_operand_list(&mut arguments, Delimiter::Paren, true, None)?;

        parser.parse_colon()?;
        let (signature, params) = parse_signature(parser)?;
        state.add_attribute(
            "signature",
            parser.context_rc().create_attribute::<SignatureAttr, _>(signature),
        );

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let mut values = SmallVec::default();
        parser.resolve_operands(state.span, &arguments, &params, &mut values)?;
        state.operands.push(values);
        Ok(())
    }
}

/// Reads the signature of a call, as [SignatureAttr] prints it, and returns it with the types of
/// its parameters, which its arguments take.
pub(crate) fn parse_signature(
    parser: &mut dyn crate::OpAsmParser<'_>,
) -> ParseResult<(Signature, SmallVec<[Type; 2]>)> {
    use crate::attributes::AttrParser;

    let start = parser.current_location();
    let signature = <SignatureAttr as AttrParser>::parse(parser)?;
    let signature = signature.borrow();
    let Some(signature) = signature.downcast_ref::<SignatureAttr>() else {
        return Err(ParserError::InvalidAttributeValue {
            span: start,
            reason: String::from("expected a signature"),
        });
    };
    let params = signature.params().iter().map(|param| param.ty).collect();
    Ok((Signature::clone(signature), params))
}

/// The function type of a value that can be called with `signature`: its parameter types and its
/// declared result types, without the calling convention or the parameter attributes.
pub(crate) fn function_type_of(context: &Context, signature: &Signature) -> Type {
    context.function_type(
        signature.params().iter().map(|param| param.ty),
        signature.declared_results().iter().map(|result| result.ty),
    )
}

pub type FunctionReferenceRef = UnsafeIntrusiveEntityRef<FunctionReference>;

/// A value of function type naming the callable `callee`:
/// `%f = builtin.function_ref @callee : (i32, i32) -> i32;`.
///
/// The result's type is written in the text and kept as the hidden property `ty`, from which the
/// result is made. The callee resolves, through any chain of [FunctionAlias](super::FunctionAlias)es,
/// to a callable whose signature has the parameter and result types of that function type. A callee
/// that names nothing is reported by the check every symbol use gets
/// ([Operation::verify_symbol_refs](crate::Operation::verify_symbol_refs)).
///
/// The operation is pure: it has no effects, is removed when unused, and two that name the same
/// symbol with the same type are merged. It is not a constant: the reference is a tracked symbol
/// use, so it is not folded to an attribute. It is not a call, so a callable it names counts as
/// having its address taken. [CallIndirect] calls the value it defines.
#[derive(EffectOpInterface)]
#[operation(
    dialect = BuiltinDialect,
    name = "function_ref",
    traits(Pure, AlwaysSpeculatable),
    implements(
        ConditionallySpeculatable,
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct FunctionReference {
    /// The callable named: a function, or an alias of one
    #[symbol(callable)]
    callee: crate::SymbolPath,
    /// The function type of the value
    #[attr(hidden)]
    ty: TypeAttr,
    /// The function value
    #[result]
    function: AnyFunction,
}

impl ConditionallySpeculatable for FunctionReference {
    fn speculatability(&self) -> Speculatability {
        Speculatability::Speculatable
    }
}

/// The result has the type `ty`.
impl InferTypeOpInterface for FunctionReference {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = *self.get_ty();
        self.function_mut().set_type(ty);
        Ok(())
    }
}

/// The result has the type `ty`, and the callee resolves, through any chain of aliases, to a
/// callable whose signature has the parameter and result types of that function type.
///
/// A callee that is not linked to a symbol is left to
/// [Operation::verify_symbol_refs](crate::Operation::verify_symbol_refs), which reports that it does
/// not resolve.
impl crate::Verify<dyn InferTypeOpInterface> for FunctionReference {
    fn verify(&self, context: &Context) -> Result<(), Report> {
        let span = self.as_operation().span();
        let ty = self.function().ty();
        let written = *self.get_ty();
        if ty != written {
            return Err(report!(
                message: "invalid operation builtin.function_ref",
                labels: [primary(
                    span,
                    format!("the result has type `{ty}`, but the reference is typed `{written}`")
                )],
            ));
        }

        let callee = self.callee();
        if !callee.try_user().is_some_and(|user| user.is_linked()) {
            return Ok(());
        }
        let resolved = callee.resolve_callable().map_err(|err| {
            report!(
                message: "invalid operation builtin.function_ref",
                labels: [primary(span, format!("the callee does not resolve to a callable: {err}"))],
            )
        })?;
        let target = resolved.target();
        let target_signature = target.signature();
        let expected = function_type_of(context, &target_signature);
        if ty == expected {
            return Ok(());
        }
        let target_span = target.as_operation_ref().borrow().span();
        Err(report!(
            message: "invalid operation builtin.function_ref",
            labels: [
                primary(
                    span,
                    format!("the reference has type `{ty}`, but its callee has type `{expected}`")
                ),
                context(target_span, format!("the callee is declared `{target_signature}`")),
            ],
        ))
    }
}

impl OpPrinter for FunctionReference {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::const_text;

        printer.print_space();
        printer.print_symbol_path(self.callee().path());
        *printer += const_text(" : ");
        printer.print_type(self.function().ty());
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

/// Reads `@callee : type`, then an optional `attributes {..}`; the type must be a function type.
impl OpParser for FunctionReference {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let callee = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute("callee", callee);

        let ty = parser.parse_colon_type()?;
        if !ty.is_function() {
            return Err(ParserError::InvalidAttributeValue {
                span: ty.span(),
                reason: format!("expected a function type, got '{ty}'"),
            });
        }

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let ty = ty.into_inner();
        state.add_attribute("ty", parser.context_rc().create_attribute::<TypeAttr, _>(ty));
        state.results.push(ty);
        Ok(())
    }
}

pub type CallIndirectRef = UnsafeIntrusiveEntityRef<CallIndirect>;

/// A call of the function value `callee`, with the signature the call is made with:
/// `%r = builtin.call_indirect %f(%a, %b) : extern("C") (i32, i32) -> i32;`.
///
/// The callee has the function type of the signature: its parameter types and result types. The
/// call has one argument per parameter of the signature and one result per result of it, of the
/// same types; the results are made from the signature when the call is built or parsed.
///
/// The set of callables the call may reach is not known, and neither are its effects: it
/// implements no effect interface, so it is never removed, merged or moved. When the callee is
/// defined by a [FunctionReference] whose callable has the call's signature, canonicalization
/// turns the call into a [Call] of that symbol.
#[operation(
    dialect = BuiltinDialect,
    implements(CallOpInterface, InferTypeOpInterface, OpPrinter)
)]
pub struct CallIndirect {
    /// The function value called
    #[operand]
    callee: AnyFunction,
    /// The arguments, one per parameter of the signature
    #[operands]
    arguments: AnyType,
    /// The signature the call is made with
    #[attr(hidden)]
    signature: SignatureAttr,
    /// The results, one per result of the signature
    #[results]
    returns: AnyType,
}

/// One result per result of the signature, in order. A call that already has results (the generic
/// form gives them) keeps them, and the verifier checks them against the signature.
impl InferTypeOpInterface for CallIndirect {
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

impl CallOpInterface for CallIndirect {
    #[inline]
    fn callable_for_callee(&self) -> Callable {
        Callable::Value(self.callee().as_value_ref())
    }

    /// Makes `callable`, a value, the callee.
    ///
    /// # Panics
    ///
    /// Panics if `callable` is a symbol: an indirect call calls a value, and turning it into a
    /// direct call means replacing it with a [Call]. Nothing in the workspace calls this with a
    /// symbol.
    fn set_callee(&mut self, callable: Callable) {
        match callable {
            Callable::Value(value) => self.callee_mut().set(value),
            Callable::Symbol(path) => panic!(
                "the callee of builtin.call_indirect is a value, not the symbol '{path}': replace \
                 the call with a builtin.call instead"
            ),
        }
    }

    #[inline]
    fn arguments(&self) -> OpOperandRange<'_> {
        self.operands().group(1)
    }

    #[inline]
    fn arguments_mut(&mut self) -> OpOperandRangeMut<'_> {
        self.operands_mut().group_mut(1)
    }

    /// The signature the call carries.
    fn callee_signature(&self) -> Option<Signature> {
        Some(self.get_signature().clone())
    }
}

/// A call of a [FunctionReference] whose callable has the call's signature becomes a [Call]
/// ([CallIndirectOfFunctionRef](crate::dialects::builtin::canonicalization::CallIndirectOfFunctionRef)).
impl Canonicalizable for CallIndirect {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(crate::dialects::builtin::canonicalization::CallIndirectOfFunctionRef::new(
            context,
        ));
    }
}

/// The callee's function type has the parameter and result types of the signature, and the call's
/// arguments and results agree with the signature.
impl crate::Verify<dyn CallOpInterface> for CallIndirect {
    fn verify(&self, context: &Context) -> Result<(), Report> {
        let signature = self.get_signature();
        let expected = function_type_of(context, &signature);
        let ty = self.callee().ty();
        if ty != expected {
            return Err(report!(
                message: "invalid operation builtin.call_indirect",
                labels: [primary(
                    self.as_operation().span(),
                    format!(
                        "the callee has type `{ty}`, but the call's signature gives `{expected}`"
                    )
                )],
            ));
        }
        crate::conversion::verify_call_signature_operands_and_results(
            self.as_operation(),
            &signature,
        )
    }
}

impl OpPrinter for CallIndirect {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::{AsValueRange, AttrPrinter, formatter::const_text};

        printer.print_space();
        printer.print_value_uses(self.operands().group(0).as_value_range());
        printer.print_operand_list(self.operands().group(1));
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

/// Reads `%callee(%a, ..) : signature`, then an optional `attributes {..}`. The callee takes the
/// function type of the signature, and the arguments the types of its parameters.
impl OpParser for CallIndirect {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let callee = parser.parse_operand(false)?;
        let mut arguments = SmallVec::<[parse::UnresolvedOperand; 2]>::default();
        parser.parse_operand_list(&mut arguments, Delimiter::Paren, true, None)?;

        parser.parse_colon()?;
        let (signature, params) = parse_signature(parser)?;
        let callee_ty = function_type_of(parser.context(), &signature);
        state.add_attribute(
            "signature",
            parser.context_rc().create_attribute::<SignatureAttr, _>(signature),
        );

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let callee = parser.resolve_operand(callee, callee_ty)?;
        state.operands.push(SmallVec::from_iter([callee]));
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

    /// `@caller(%f: (i32, i32) -> i32, %a: i32, %b: i32) -> i32` beside `@callee(i32, i32) -> i32`,
    /// with the caller's three arguments.
    fn indirect_caller() -> (Test, FunctionRef, [ValueRef; 3]) {
        let mut test = Test::named("caller").in_module("calls");
        let i32 = test.context().i32();
        let fn_ty = test.context().function_type([i32, i32], [i32]);
        let callee = test.define_function("callee", &[i32, i32], &[i32]);
        test.with_function("caller", &[fn_ty, i32, i32], &[i32]);
        let entry = test.entry_block();
        let args = {
            let entry = entry.borrow();
            [0, 1, 2].map(|index| entry.arguments()[index] as ValueRef)
        };
        (test, callee, args)
    }

    #[test]
    fn the_function_ref_builder_types_the_reference_from_its_callee() {
        let (mut test, callee, [_, a, b]) = indirect_caller();
        let signature = callee.borrow().get_signature().clone();
        let (reference, call) = {
            let mut builder = test.function_builder();
            let reference = builder.function_ref(callee, SourceSpan::UNKNOWN).unwrap();
            let value = reference.borrow().function().as_value_ref();
            let call = builder
                .call_indirect(value, signature.clone(), [a, b], SourceSpan::UNKNOWN)
                .unwrap();
            let result = call.borrow().returns()[0].borrow().as_value_ref();
            builder.ret([result], SourceSpan::UNKNOWN).unwrap();
            (reference, call)
        };
        verify(&test).unwrap();

        let i32 = test.context().i32();
        let fn_ty = test.context().function_type([i32, i32], [i32]);
        let reference = reference.borrow();
        assert_eq!(reference.function().ty(), fn_ty);
        assert!(reference.as_operation().is_memory_effect_free());
        assert!(!reference.as_operation().implements::<dyn CallOpInterface>());
        let call = call.borrow();
        assert_eq!(*call.get_signature(), signature);
        assert_eq!(call.returns().len(), 1);
        assert_eq!(call.returns()[0].borrow().ty(), i32);
    }

    /// The callee is the value; the arguments are the second operand group; the set of callees is
    /// not known; the call has unknown effects; the signature is the call's own.
    #[test]
    fn an_indirect_call_calls_a_value_with_unknown_callees_and_effects() {
        let (mut test, callee, [f, a, b]) = indirect_caller();
        let signature = callee.borrow().get_signature().clone();
        let call = {
            let mut builder = test.function_builder();
            let call = builder
                .call_indirect(f, signature.clone(), [a, b], SourceSpan::UNKNOWN)
                .unwrap();
            let result = call.borrow().returns()[0].borrow().as_value_ref();
            builder.ret([result], SourceSpan::UNKNOWN).unwrap();
            call
        };
        verify(&test).unwrap();

        let call = call.borrow();
        let interface = call.as_operation().as_trait::<dyn CallOpInterface>().unwrap();
        assert!(matches!(interface.callable_for_callee(), Callable::Value(value) if value == f));
        let passed = interface
            .arguments()
            .iter()
            .map(|operand| operand.borrow().as_value_ref())
            .collect::<Vec<_>>();
        assert_eq!(passed, [a, b]);
        assert!(interface.possible_callees().is_none());
        assert!(interface.resolve().is_none());
        assert_eq!(interface.callee_signature(), Some(signature));
        assert!(!call.as_operation().implements::<dyn MemoryEffectOpInterface>());
        assert!(!call.as_operation().is_memory_effect_free());
        assert!(call.as_operation().get_effects_recursively::<MemoryEffect>().is_none());
    }

    #[test]
    fn set_callee_of_an_indirect_call_replaces_the_callee_operand() {
        let (mut test, callee, [f, a, b]) = indirect_caller();
        let signature = callee.borrow().get_signature().clone();
        let (reference, mut call) = {
            let mut builder = test.function_builder();
            let reference = builder.function_ref(callee, SourceSpan::UNKNOWN).unwrap();
            let call = builder.call_indirect(f, signature, [a, b], SourceSpan::UNKNOWN).unwrap();
            let result = call.borrow().returns()[0].borrow().as_value_ref();
            builder.ret([result], SourceSpan::UNKNOWN).unwrap();
            (reference, call)
        };
        let value = reference.borrow().function().as_value_ref();
        CallOpInterface::set_callee(&mut *call.borrow_mut(), Callable::Value(value));
        verify(&test).unwrap();
        assert_eq!(call.borrow().callee().as_value_ref(), value);
        assert!(!f.borrow().is_used());
    }

    #[test]
    #[should_panic(expected = "the callee of builtin.call_indirect is a value")]
    fn set_callee_of_an_indirect_call_to_a_symbol_panics() {
        let (mut test, callee, [f, a, b]) = indirect_caller();
        let signature = callee.borrow().get_signature().clone();
        let mut call = {
            let mut builder = test.function_builder();
            builder.call_indirect(f, signature, [a, b], SourceSpan::UNKNOWN).unwrap()
        };
        let path = callee.borrow().as_operation().as_symbol().unwrap().path();
        CallOpInterface::set_callee(&mut *call.borrow_mut(), path.into());
    }

    #[test]
    fn an_indirect_call_whose_callee_type_disagrees_with_its_signature_is_an_error() {
        let (mut test, _, [f, a, b]) = indirect_caller();
        let i32 = test.context().i32();
        let signature = Signature::new(&test.context_rc(), [i32, i32], []);
        {
            let mut builder = test.function_builder();
            builder.call_indirect(f, signature, [a, b], SourceSpan::UNKNOWN).unwrap();
            builder.ret_imm(crate::Int::from(0i32), SourceSpan::UNKNOWN).unwrap();
        }
        let error = describe(&verify(&test).expect_err("the call must not verify"));
        assert!(error.contains("invalid operation builtin.call_indirect"), "{error}");
        assert!(
            error.contains(
                "the callee has type `(i32, i32) -> i32`, but the call's signature gives `(i32, \
                 i32) -> ()`"
            ),
            "{error}"
        );
    }

    #[test]
    fn an_indirect_call_of_a_value_that_is_not_a_function_is_an_error() {
        let (mut test, callee, [_, a, b]) = indirect_caller();
        let signature = callee.borrow().get_signature().clone();
        let mut builder = test.function_builder();
        let Err(error) = builder.call_indirect(a, signature, [a, b], SourceSpan::UNKNOWN) else {
            panic!("an i32 callee must be rejected");
        };
        let error = describe(&error);
        assert!(error.contains("expected a function type, got i32"), "{error}");
    }

    #[test]
    fn indirect_call_arguments_that_disagree_with_the_signature_are_an_error() {
        let (mut test, callee, [f, a, _]) = indirect_caller();
        let signature = callee.borrow().get_signature().clone();
        {
            let mut builder = test.function_builder();
            builder.call_indirect(f, signature, [a], SourceSpan::UNKNOWN).unwrap();
            builder.ret_imm(crate::Int::from(0i32), SourceSpan::UNKNOWN).unwrap();
        }
        let error = describe(&verify(&test).expect_err("the call must not verify"));
        assert!(
            error.contains("the call passes 1 argument(s), but its signature has 2 parameter(s)"),
            "{error}"
        );
    }

    /// A reference whose result is retyped no longer has its written type; one whose written type
    /// is changed as well no longer matches its callee.
    #[test]
    fn a_function_ref_whose_type_disagrees_with_its_callee_is_an_error() {
        let (mut test, callee, _) = indirect_caller();
        let mut reference = {
            let mut builder = test.function_builder();
            let reference = builder.function_ref(callee, SourceSpan::UNKNOWN).unwrap();
            builder.ret_imm(crate::Int::from(0i32), SourceSpan::UNKNOWN).unwrap();
            reference
        };
        let i32 = test.context().i32();
        let other = test.context().function_type([i32], [i32]);
        reference.borrow_mut().function_mut().set_type(other);
        let error = describe(&verify(&test).expect_err("the reference must not verify"));
        assert!(
            error.contains(
                "the result has type `(i32) -> i32`, but the reference is typed `(i32, i32) -> \
                 i32`"
            ),
            "{error}"
        );

        reference.borrow_mut().set_ty(other);
        let error = describe(&verify(&test).expect_err("the reference must not verify"));
        assert!(error.contains("invalid operation builtin.function_ref"), "{error}");
        assert!(
            error.contains(
                "the reference has type `(i32) -> i32`, but its callee has type `(i32, i32) -> \
                 i32`"
            ),
            "{error}"
        );
    }
}
