use alloc::{format, string::String, vec::Vec};

use crate::{
    AsValueRange, BlockRef, CallConv, CallableOpInterface, CallableSymbol, EntityRef, IdentAttr,
    Op, OpParser, OpPrinter, Operation, RegionKind, RegionKindInterface, RegionRef, SmallVec,
    Symbol, SymbolUse, SymbolUseList, Type, UnsafeIntrusiveEntityRef, Usable, Visibility,
    derive::{EffectOpInterface, operation},
    dialects::builtin::{
        BuiltinDialect,
        attributes::{
            AbiParam, DeclaredEffects, DeclaredEffectsAttr, IntAttr, LocalVariable, Signature,
            SignatureAttr, TypeArrayAttr, VisibilityAttr,
            effects::{parse_effect_list, print_effect_list},
            signature::{
                SignatureWriter, invalid_parameter_list, next_token_span, parse_optional_results,
                parse_params_after_lparen, print_optional_results, print_param_list,
            },
        },
    },
    effects::{
        EffectInstance, EffectIterator, EffectOpInterface, HasRecursiveMemoryEffects, MemoryEffect,
        MemoryEffectOpInterface,
    },
    interner,
    parse::ParserExt,
    print::AsmPrinter,
    traits::{AnyType, IsolatedFromAbove, ReturnLike, SingleRegion, Terminator},
};

trait UsableSymbol = Usable<Use = SymbolUse>;

pub type FunctionRef = UnsafeIntrusiveEntityRef<Function>;

#[operation(
    dialect = BuiltinDialect,
    traits(SingleRegion, IsolatedFromAbove, HasRecursiveMemoryEffects),
    implements(
        UsableSymbol,
        Symbol,
        CallableOpInterface,
        CallableSymbol,
        RegionKindInterface,
        MemoryEffectOpInterface,
        OpPrinter
    )
)]
pub struct Function {
    #[attr]
    name: IdentAttr,
    #[attr]
    linkage: VisibilityAttr,
    #[attr]
    signature: SignatureAttr,
    #[region]
    body: RegionRef,
    /// What the function states about the memory effects of calling it: unknown by default
    #[attr]
    #[default]
    memory_effects: DeclaredEffectsAttr,
    /// The set of local variables allocated within this function
    #[attr]
    #[default]
    locals: TypeArrayAttr,
    /// The uses of this function as a symbol
    #[default]
    uses: SymbolUseList,
}

/// The effects a function declares for itself: those of its `memory_effects` property for a
/// declaration (unknown when the property is [DeclaredEffects::Unknown]), and none for a
/// definition.
///
/// A definition's effects are its body's. That is what [HasRecursiveMemoryEffects] says, and the
/// recursive queries ([Operation::is_memory_effect_free], [Operation::get_effects_recursively],
/// `RecursiveEffectIterator`) report them, "unknown" included. They must not be reported here as
/// well: the recursive walk asks every operation it visits for its own effects, the function
/// included, so reporting the body here would recurse without bound and count each effect twice.
impl EffectOpInterface<MemoryEffect> for Function {
    fn effects(&self) -> EffectIterator<MemoryEffect> {
        if !self.is_declaration() {
            return EffectIterator::new([]);
        }
        match self.get_memory_effects().known() {
            Some(effects) => {
                EffectIterator::new(effects.iter().map(|desc| EffectInstance::new(desc.effect)))
            }
            None => EffectIterator::new([]),
        }
    }

    fn effects_are_known(&self) -> bool {
        !self.is_declaration() || self.get_memory_effects().is_known()
    }
}

impl OpParser for Function {
    fn parse(
        state: &mut crate::OperationState,
        parser: &mut dyn crate::OpAsmParser<'_>,
    ) -> crate::ParseResult {
        use alloc::string::ToString;

        use crate::{
            diagnostics::report,
            parse::{ParserError, Token},
        };

        let visibility = match parser.parse_optional_keyword()? {
            Some(tok) => match tok.inner() {
                Token::BareIdent("public") => Visibility::Public,
                Token::BareIdent("private") => Visibility::Private,
                Token::BareIdent("internal") => Visibility::Internal,
                invalid => {
                    return Err(ParserError::UnexpectedToken {
                        span: tok.span(),
                        token: invalid.to_string(),
                        expected: Some("visibility keyword".to_string()),
                    });
                }
            },
            None => Visibility::Private,
        };

        let extern_keyword = parser.parse_optional_keyword()?;
        match extern_keyword.map(|tok| tok.into_parts()) {
            None => (),
            Some((_, Token::BareIdent("extern"))) => (),
            Some((span, invalid)) => {
                return Err(ParserError::UnexpectedToken {
                    span,
                    token: invalid.to_string(),
                    expected: Some("extern keyword".to_string()),
                });
            }
        }

        parser.parse_lparen()?;
        let cc_string = parser.parse_string()?;
        let cc = CallConv::new(cc_string.as_str());
        parser.parse_rparen()?;

        let name = parser.parse_symbol_name()?;

        // A list whose first element is a value name is the argument list of a definition; any
        // other list is the parameter list of a declaration, unless it is empty, when what follows
        // the results decides (a region makes a definition).
        parser.parse_lparen()?;
        let is_named = |parser: &mut dyn crate::OpAsmParser<'_>| {
            parser.token_stream_mut().is_next(|tok| matches!(tok, Token::PercentIdent(_)))
        };
        let named = is_named(parser);
        let mut args = SmallVec::<[crate::parse::Argument; 4]>::new_const();
        let mut params = Vec::new();
        if named {
            loop {
                if !is_named(parser) {
                    let span = next_token_span(parser)?;
                    return Err(invalid_parameter_list(
                        span,
                        "expected a named parameter, `%name: type`, as the first parameter of \
                         this list is named",
                    ));
                }
                let arg =
                    parser.parse_argument(/*allow_type=*/ true, /*allow_attrs=*/ true)?;
                params.push(AbiParam::new_with_attrs(arg.ty, arg.attrs.iter().copied()));
                args.push(arg);
                if !parser.parse_optional_comma()? {
                    break;
                }
            }
            parser.parse_rparen()?;
        } else {
            parse_params_after_lparen(parser, &mut params)?;
        }

        // Whether the results are a single type without parentheses, which has no dictionary
        let bare_result = parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Rstab))
            && parser.token_stream().peek_second() != Some(Token::Lparen);
        let results = parse_optional_results(parser)?;

        // `effects(..)` and `locals(..)`, each left out at its default
        let memory_effects = if parser.parse_optional_custom_keyword("effects")?.is_some() {
            parser.parse_lparen()?;
            let effects = parse_effect_list(parser)?;
            parser.parse_rparen()?;
            DeclaredEffects::Known(effects)
        } else {
            DeclaredEffects::Unknown
        };
        let mut locals = Vec::new();
        if parser.parse_optional_custom_keyword("locals")?.is_some() {
            parser.parse_lparen()?;
            if !parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Rparen)) {
                loop {
                    locals.push(parser.parse_type()?.into_inner());
                    if !parser.parse_optional_comma()? {
                        break;
                    }
                }
            }
            parser.parse_rparen()?;
        }

        let declaration = !named && !params.is_empty();
        let has_body = parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Lbrace));
        if declaration && has_body {
            let span = next_token_span(parser)?;
            let reason = if bare_result {
                "a function declared without parameter names cannot have a body; the attributes of \
                 a result are written in parentheses, `-> (type { .. })`"
            } else {
                "a function declared without parameter names cannot have a body"
            };
            return Err(ParserError::from(report! {
                message: "invalid function declaration",
                labels: [primary(span, String::from(reason))],
            }));
        }
        if named && !has_body {
            let span = next_token_span(parser)?;
            let reason = "expected the body of a function whose parameters are named";
            return Err(ParserError::from(report! {
                message: "invalid function definition",
                labels: [primary(span, reason)],
            }));
        }

        let sig = Signature::from_parts(cc, params, results);

        let name = parser.context_rc().create_attribute::<IdentAttr, _>(name);
        state.add_attribute("name", name);

        let visibility = parser.context_rc().create_attribute::<VisibilityAttr, _>(visibility);
        state.add_attribute("linkage", visibility);

        let ty = parser.context_rc().create_attribute::<SignatureAttr, _>(sig);
        state.add_attribute("signature", ty);

        let locals = parser.context_rc().create_attribute::<TypeArrayAttr, _>(locals);
        state.add_attribute("locals", locals);

        let memory_effects =
            parser.context_rc().create_attribute::<DeclaredEffectsAttr, _>(memory_effects);
        state.add_attribute("memory_effects", memory_effects);

        // A declaration has one empty region
        let body = match parser.parse_optional_region(&args, false)? {
            Some(body) => body,
            None => parser.context().create_region(),
        };
        state.add_region(body);
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

impl OpPrinter for Function {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::{const_text, display};

        let sig = self.get_signature();

        printer.print_space();
        printer.print_keyword(self.get_linkage().as_str());
        printer.print_space();
        printer.print_keyword("extern");
        printer.print_lparen();
        printer.print_string(sig.calling_convention().as_str());
        printer.print_rparen();
        printer.print_space();

        printer.print_symbol_name(self.get_name().as_symbol());
        if self.is_declaration() {
            print_param_list(printer, sig.params());
            print_optional_results(printer, sig.declared_results());
            self.print_effects_and_locals(printer);
        } else {
            let body = self.body();
            let entry = body.entry();
            printer.print_lparen();
            for (i, arg) in entry.argument_values().enumerate() {
                if i > 0 {
                    *printer += const_text(", ");
                }
                let arg = arg.borrow();
                *printer += display(arg.id()) + const_text(": ");
                printer.print_type(arg.ty());
                if let Some(param) = sig.params().get(i)
                    && !param.attributes().is_empty()
                {
                    printer.print_space();
                    printer.write_attributes(param.attributes());
                }
            }
            printer.print_rparen();
            print_optional_results(printer, sig.declared_results());
            self.print_effects_and_locals(printer);
            printer.print_space();
            printer.print_region(&body);
        }
        if self.op.has_attributes() {
            printer.print_space();
            printer.print_keyword("attributes");
            printer.print_space();
            printer.print_attribute_dictionary(
                self.op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }
    }
}

impl Function {
    /// ` effects(..)` unless the effects are unknown, then ` locals(..)` unless there are none.
    fn print_effects_and_locals(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::const_text;

        if let Some(effects) = self.get_memory_effects().known() {
            printer.print_space();
            printer.print_keyword("effects");
            printer.print_lparen();
            print_effect_list(printer, effects);
            printer.print_rparen();
        }
        let locals = self.get_locals();
        if !locals.is_empty() {
            printer.print_space();
            printer.print_keyword("locals");
            printer.print_lparen();
            for (i, ty) in locals.iter().enumerate() {
                if i > 0 {
                    *printer += const_text(", ");
                }
                printer.print_type(*ty);
            }
            printer.print_rparen();
        }
    }
}

/// Builders
impl Function {
    /// Conver this function from a declaration (no body) to a definition (has a body) by creating
    /// the entry block based on the function signature.
    ///
    /// NOTE: The resulting function is _invalid_ until the block has a terminator inserted into it.
    ///
    /// This function will panic if an entry block has already been created
    pub fn create_entry_block(&mut self) -> BlockRef {
        assert!(self.body().is_empty(), "entry block already exists");
        let signature = self.get_signature();
        let block = self
            .as_operation()
            .context_rc()
            .create_block_with_params(signature.params().iter().map(|p| p.ty));
        drop(signature);
        let mut body = self.body_mut();
        body.push_back(block);
        block
    }
}

/// Accessors
impl Function {
    #[inline]
    pub fn entry_block(&self) -> BlockRef {
        self.body()
            .body()
            .front()
            .as_pointer()
            .expect("cannot get entry block for declaration")
    }

    pub fn last_block(&self) -> BlockRef {
        self.body()
            .body()
            .back()
            .as_pointer()
            .expect("cannot access blocks of a function declaration")
    }

    pub fn num_locals(&self) -> usize {
        self.locals().len()
    }

    pub fn get_local(&self, id: &LocalVariable) -> EntityRef<'_, Type> {
        assert_eq!(
            self.as_operation_ref(),
            id.function().as_operation_ref(),
            "attempted to use local variable reference from different function"
        );
        EntityRef::map(self.get_locals(), |locals| &locals[id.as_usize()])
    }

    pub fn alloc_local(&mut self, ty: Type) -> LocalVariable {
        let mut locals = self.get_locals_mut();
        let id = locals.len();
        locals.push(ty);
        drop(locals);
        LocalVariable::new(self.as_function_ref(), id)
    }

    pub fn iter_locals(&self) -> impl ExactSizeIterator<Item = LocalVariable> {
        let fun = self.as_function_ref();
        (0..self.locals().len()).map(move |i| LocalVariable::new(fun, i))
    }

    #[inline(always)]
    pub fn as_function_ref(&self) -> FunctionRef {
        unsafe { FunctionRef::from_raw(self) }
    }
}

impl RegionKindInterface for Function {
    #[inline(always)]
    fn kind(&self) -> RegionKind {
        RegionKind::SSA
    }
}

impl Usable for Function {
    type Use = SymbolUse;

    #[inline(always)]
    fn uses(&self) -> &SymbolUseList {
        &self.uses
    }

    #[inline(always)]
    fn uses_mut(&mut self) -> &mut SymbolUseList {
        &mut self.uses
    }
}

impl Symbol for Function {
    #[inline(always)]
    fn as_symbol_operation(&self) -> &Operation {
        &self.op
    }

    #[inline(always)]
    fn as_symbol_operation_mut(&mut self) -> &mut Operation {
        &mut self.op
    }

    fn name(&self) -> interner::Symbol {
        self.get_name().as_symbol()
    }

    fn set_name(&mut self, name: interner::Symbol) {
        self.get_name_mut().name = name;
    }

    fn visibility(&self) -> Visibility {
        *self.get_linkage()
    }

    fn set_visibility(&mut self, visibility: Visibility) {
        *self.get_linkage_mut() = visibility;
    }

    /// Returns true if this operation is a declaration, rather than a definition, of a symbol
    ///
    /// The default implementation assumes that all operations are definitions
    #[inline]
    fn is_declaration(&self) -> bool {
        self.body().is_empty()
    }
}

impl CallableOpInterface for Function {
    fn get_callable_region(&self) -> Option<RegionRef> {
        if self.is_declaration() {
            None
        } else {
            self.op.regions().front().as_pointer()
        }
    }

    fn signature(&self) -> Signature {
        self.get_signature().clone()
    }

    /// The function's `memory_effects` property, for a declaration and a definition alike
    fn declared_effects(&self) -> DeclaredEffects {
        self.get_memory_effects().clone()
    }
}

impl CallableSymbol for Function {}

/// A function verifies its body against its signature: the entry block of a definition has one
/// argument per parameter, of the parameter's type, and every `builtin.ret` and `builtin.ret_imm`
/// in the body, outside nested callables, returns exactly the results of the signature (none may
/// appear in a function that never returns).
///
/// Every effect in the function's declared effects names at most one of an argument and a result,
/// and the one it names exists in the signature. A declaration has no locals: its `locals(..)` is
/// empty.
///
/// The signature itself is checked before this runs, by
/// [Operation::verify_signature_attributes], as for every operation that holds one.
impl crate::Verify<dyn CallableSymbol> for Function {
    fn verify(&self, _context: &crate::Context) -> Result<(), crate::Report> {
        use crate::{Spanned, ValueRef, WalkResult, diagnostics::report};

        let span = self.as_operation().span();
        let invalid = |label: String| {
            report!(
                message: "invalid operation builtin.function",
                labels: [primary(span, label)],
            )
        };

        let signature = self.get_signature();
        if let Some(effects) = self.get_memory_effects().known() {
            for effect in effects {
                let target = match (effect.argument, effect.result) {
                    (Some(_), Some(_)) => Some(format!(
                        "the declared effect `{effect}` names both an argument and a result"
                    )),
                    (Some(index), None) if usize::from(index) >= signature.params().len() => {
                        Some(format!(
                            "the declared effect `{effect}` names argument {index}, but the \
                             function has {} parameter(s)",
                            signature.params().len()
                        ))
                    }
                    (None, Some(index)) if usize::from(index) >= signature.results().len() => {
                        Some(format!(
                            "the declared effect `{effect}` names result {index}, but the \
                             function has {} result(s)",
                            signature.results().len()
                        ))
                    }
                    _ => None,
                };
                if let Some(reason) = target {
                    return Err(invalid(reason));
                }
            }
        }
        if self.is_declaration() {
            let locals = self.num_locals();
            if locals != 0 {
                return Err(invalid(format!(
                    "a declaration has no body, so it has no locals, but this one states {locals} \
                     local(s)"
                )));
            }
            return Ok(());
        }

        let entry = self.entry_block();
        let entry = entry.borrow();
        let arguments = entry.arguments();
        if arguments.len() != signature.params().len() {
            return Err(invalid(format!(
                "expected one entry block argument per parameter; parameter count: {}, entry \
                 block argument count: {}",
                signature.params().len(),
                arguments.len()
            )));
        }
        for (index, (argument, param)) in arguments.iter().zip(signature.params()).enumerate() {
            let ty = (*argument as ValueRef).borrow().ty();
            if ty != param.ty {
                return Err(invalid(format!(
                    "argument {index} of the entry block has type `{ty}`, but parameter {index} \
                     has type `{}`",
                    param.ty
                )));
            }
        }

        let expected = signature.results().iter().map(|result| result.ty).collect::<Vec<_>>();
        let never_returns = signature.never_returns();
        let check_return = |op: &Operation, returned: &[Type]| -> Result<(), crate::Report> {
            let reason = if never_returns {
                String::from("this returns from a function that never returns")
            } else if returned != expected.as_slice() {
                format!(
                    "this returns {}, but the function returns {}",
                    type_list(returned),
                    type_list(&expected)
                )
            } else {
                return Ok(());
            };
            Err(report!(
                message: "invalid operation builtin.function",
                labels: [
                    primary(op.span(), reason),
                    context(span, "in this function"),
                ],
            ))
        };

        self.body()
            .prewalk(|op: &Operation| {
                if op.implements::<dyn CallableOpInterface>() {
                    return WalkResult::Skip;
                }
                let checked = if let Some(ret) = op.downcast_ref::<Ret>() {
                    let returned =
                        ret.values().iter().map(|value| value.borrow().ty()).collect::<Vec<_>>();
                    check_return(op, &returned)
                } else if let Some(ret_imm) = op.downcast_ref::<RetImm>() {
                    check_return(op, &[crate::Attribute::ty(&*ret_imm.value())])
                } else {
                    Ok(())
                };
                match checked {
                    Ok(()) => WalkResult::Continue(()),
                    Err(err) => WalkResult::Break(err),
                }
            })
            .into_result()
    }
}

/// `types` as a parenthesised list, `(i32, i64)`.
fn type_list(types: &[Type]) -> String {
    let types = types.iter().map(|ty| format!("{ty}")).collect::<Vec<_>>();
    format!("({})", types.join(", "))
}

/// Returns from the enclosing function with the provided operands as its results.
///
/// Returning touches no memory: the operation declares that it has no memory effects.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = BuiltinDialect,
    traits(Terminator, ReturnLike),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct Ret {
    #[operands]
    values: AnyType,
}

impl OpPrinter for Ret {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let returning = self.values();
        if returning.is_empty() {
            return;
        }

        printer.print_space();
        printer.print_value_uses(returning.as_value_range());
        printer.print_space();
        printer.print_colon_type_list(returning.iter().map(|r| r.borrow().ty()));
    }
}

impl OpParser for Ret {
    fn parse(
        state: &mut crate::OperationState,
        parser: &mut dyn crate::OpAsmParser<'_>,
    ) -> crate::ParseResult {
        use crate::{
            diagnostics::SourceSpan,
            parse::{Delimiter, Token},
        };

        if !parser.token_stream_mut().is_next(|tok| matches!(tok, Token::PercentIdent(_))) {
            return Ok(());
        }

        let start = parser.token_stream().current_position();
        let mut args = SmallVec::default();
        parser.parse_operand_list(
            &mut args,
            Delimiter::None,
            /*allow_result_number=*/ true,
            None,
        )?;

        let mut types = SmallVec::default();
        parser.parse_colon_type_list(&mut types)?;

        let end = parser.token_stream().current_span();
        let span = SourceSpan::session(
            end.source().id(),
            crate::diagnostics::TextRange::new(start, end.range().end()).expect("valid range"),
        );

        let mut operands = SmallVec::default();
        parser.resolve_operands(span, &args, &types, &mut operands)?;

        state.operands.push(operands);

        Ok(())
    }
}

/// Returns from the enclosing function with the provided integer constant as its result.
///
/// Returning touches no memory: the operation declares that it has no memory effects.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = BuiltinDialect,
    traits(Terminator, ReturnLike),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct RetImm {
    #[attr(hidden)]
    value: IntAttr,
}

impl OpPrinter for RetImm {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::{attributes::IntegerLikeAttr, formatter::const_text};

        // Print the constant as `<value> : <type>`, mirroring the grammar accepted by the
        // `OpParser` implementation below, so the output can be re-parsed.
        let value = self.value();
        printer.print_space();
        printer.print_int(value.as_int());
        *printer += const_text(" : ");
        printer.print_type(crate::Attribute::ty(&*value));
    }
}

impl OpParser for RetImm {
    fn parse(
        state: &mut crate::OperationState,
        parser: &mut dyn crate::OpAsmParser<'_>,
    ) -> crate::ParseResult {
        use crate::{diagnostics::SourceSpan, parse::ParserError};

        let (literal_span, literal) = parser.parse_integer_literal()?.into_parts();
        let ty = parser.parse_colon_type()?;
        let span = SourceSpan::session(
            literal_span.source().id(),
            crate::diagnostics::TextRange::new(
                literal_span.range().start(),
                ty.span().range().end(),
            )
            .expect("valid range"),
        );
        if !ty.is_integer() {
            return Err(ParserError::InvalidOperationType {
                span,
                ty_span: ty.span(),
                reason: format!("expected an integer type, got {ty}"),
            });
        }
        let value = literal.to_int(*ty, literal_span)?;
        let attr = parser.context_rc().int_attr(value, ty.into_inner());
        state.add_attribute("value", attr);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Builder, BuilderExt, OperationRef, Overflow, SourceSpan, ValueRef,
        dialects::{
            builtin::attributes::MemoryEffectDescriptor,
            test::{self as test_dialect, Add, Load, Store},
        },
        effects::RecursiveEffectIterator,
        testing::Test,
    };

    fn effects_of(function: &Function) -> Vec<MemoryEffect> {
        EffectOpInterface::<MemoryEffect>::effects(function)
            .map(|instance| *instance.effect())
            .collect()
    }

    /// A declaration states its effects in its `memory_effects` property: unknown when it says
    /// nothing, and exactly the listed effects otherwise, "none" included.
    #[test]
    fn a_declaration_reports_its_memory_effects_property() {
        use crate::effects::EffectOpInterface as _;

        let mut test = Test::named("declaration_effects").in_module("test");
        let mut function = test.define_function("callee", &[], &[]);
        assert!(function.borrow().is_declaration());
        {
            let function = function.borrow();
            assert_eq!(function.declared_effects(), DeclaredEffects::Unknown);
            assert!(!function.effects_are_known());
            assert!(!function.as_operation().is_memory_effect_free());
            assert!(function.as_operation().get_effects_recursively::<MemoryEffect>().is_none());
        }

        function.borrow_mut().set_memory_effects(DeclaredEffects::none());
        {
            let function = function.borrow();
            assert!(function.effects_are_known());
            assert!(effects_of(&function).is_empty());
            assert!(function.as_operation().is_memory_effect_free());
        }

        function.borrow_mut().set_memory_effects(DeclaredEffects::from([
            MemoryEffectDescriptor::new(MemoryEffect::Read),
            MemoryEffectDescriptor::new(MemoryEffect::Write),
        ]));
        let function = function.borrow();
        assert_eq!(effects_of(&function), [MemoryEffect::Read, MemoryEffect::Write]);
        assert!(!function.as_operation().is_memory_effect_free());
    }

    /// A definition's declared effects are its property, as a declaration's are, but it reports
    /// no effects of its own whatever it declares: its body's are found by the recursive walk.
    #[test]
    fn a_definition_declares_effects_without_reporting_them_as_its_own() {
        use crate::effects::EffectOpInterface as _;

        for declared in [
            DeclaredEffects::Unknown,
            DeclaredEffects::none(),
            DeclaredEffects::from([MemoryEffectDescriptor::new(MemoryEffect::Write)]),
        ] {
            let (_test, mut function) = definition(Body::Pure);
            function.borrow_mut().set_memory_effects(declared.clone());
            let function = function.borrow();
            assert_eq!(function.declared_effects(), declared);
            assert!(function.effects_are_known());
            assert!(effects_of(&function).is_empty());
            assert!(function.as_operation().is_memory_effect_free());
        }
    }

    /// What the body of a test function holds.
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Body {
        /// No operation at all
        Empty,
        /// A `test.add`, which declares that it has no memory effects
        Pure,
        /// A `test.load`
        Reads,
        /// A `test.store`
        Writes,
        /// A `test.ret`, which declares nothing about its memory effects
        Unknown,
        /// A `builtin.ret`
        Returns,
        /// A `test.add` and a `builtin.ret`
        PureAndReturns,
    }

    /// A function definition `f(ptr<i32>, i32)` whose body is `body`.
    fn definition(body: Body) -> (Test, FunctionRef) {
        let mut test = Test::named("effects");
        let i32 = test.context().i32();
        let pointer = test.context().ptr_type(i32);
        test.with_function("effects", &[pointer, i32], &[]);
        {
            let mut builder = test.function_builder();
            let entry = builder.entry_block();
            let (addr, value) = {
                let block = entry.borrow();
                (block.arguments()[0] as ValueRef, block.arguments()[1] as ValueRef)
            };
            let builder = builder.builder_mut();
            builder.set_insertion_point_to_end(entry);
            let span = SourceSpan::UNKNOWN;
            if matches!(body, Body::Pure | Body::PureAndReturns) {
                builder.create::<Add, (_, _, _)>(span)(value, value, Overflow::Wrapping).unwrap();
            }
            match body {
                Body::Empty | Body::Pure => (),
                Body::Reads => {
                    builder.create::<Load, _>(span)(addr).unwrap();
                }
                Body::Writes => {
                    builder.create::<Store, _>(span)(addr, value).unwrap();
                }
                Body::Unknown => {
                    builder.create::<test_dialect::Ret, ([ValueRef; 0],)>(span)([]).unwrap();
                }
                Body::Returns | Body::PureAndReturns => {
                    builder.create::<Ret, ([ValueRef; 0],)>(span)([]).unwrap();
                }
            }
        }
        let function = test.function();
        assert!(!function.borrow().is_declaration());
        (test, function)
    }

    fn walk(op: OperationRef) -> Vec<Option<MemoryEffect>> {
        RecursiveEffectIterator::<MemoryEffect>::new(op)
            .map(|(_, effect)| effect.map(|instance| *instance.effect()))
            .collect()
    }

    /// Asking a definition for its effects returns: the function declares none of its own, its
    /// effects are its body's (`HasRecursiveMemoryEffects`).
    #[test]
    fn a_definition_reports_no_effects_of_its_own() {
        for body in [Body::Empty, Body::Pure, Body::Reads, Body::Writes, Body::Unknown] {
            let (_test, function) = definition(body);
            assert!(effects_of(&function.borrow()).is_empty());
        }
    }

    /// `is_memory_effect_free` on a definition answers from the body, and an operation whose
    /// effects are unknown makes the function not free.
    #[test]
    fn a_definition_is_effect_free_exactly_when_its_body_is() {
        for (body, free) in [
            (Body::Empty, true),
            (Body::Pure, true),
            (Body::Reads, false),
            (Body::Writes, false),
            (Body::Unknown, false),
        ] {
            let (_test, function) = definition(body);
            assert_eq!(function.borrow().as_operation().is_memory_effect_free(), free);
        }
    }

    /// The recursive effects of a definition are its body's, each once, and "unknown" stays
    /// distinct from "none": `None` for a body with an operation that declares nothing, `Some([])`
    /// for a body with no effects.
    #[test]
    fn the_recursive_effects_of_a_definition_keep_unknown_apart_from_none() {
        let effects = |body| {
            let (_test, function) = definition(body);
            let function = function.borrow();
            function
                .as_operation()
                .get_effects_recursively::<MemoryEffect>()
                .map(|effects| {
                    effects.iter().map(|instance| *instance.effect()).collect::<Vec<_>>()
                })
        };
        assert_eq!(effects(Body::Empty), Some(alloc::vec![]));
        assert_eq!(effects(Body::Pure), Some(alloc::vec![]));
        assert_eq!(effects(Body::Reads), Some(alloc::vec![MemoryEffect::Read]));
        // Once, not twice: the function must not report its body's effects as its own as well
        assert_eq!(effects(Body::Writes), Some(alloc::vec![MemoryEffect::Write]));
        assert_eq!(effects(Body::Unknown), None);
    }

    /// The walk started at a definition reports the body's effects, and "unknown" for an
    /// operation that declares nothing.
    #[test]
    fn the_walk_started_at_a_definition_reports_its_body() {
        for (body, expected) in [
            (Body::Empty, alloc::vec![]),
            (Body::Pure, alloc::vec![]),
            (Body::Writes, alloc::vec![Some(MemoryEffect::Write)]),
            (Body::Unknown, alloc::vec![None]),
        ] {
            let (_test, function) = definition(body);
            assert_eq!(walk(function.borrow().as_operation().as_operation_ref()), expected);
        }
    }

    /// The other queries on `Operation` return for a definition. They are not recursive (as for
    /// any other operation with regions): a definition has no memory effect of its own.
    #[test]
    fn the_single_effect_queries_return_for_a_definition() {
        let (_test, function) = definition(Body::Writes);
        let function = function.borrow();
        let op = function.as_operation();
        assert!(!op.has_memory_effect(MemoryEffect::Write));
        assert!(!op.has_single_memory_effect(MemoryEffect::Write));
        assert!(!op.would_be_trivially_dead_even_if_terminator());
    }

    /// A function whose body only computes without memory effects and returns is free of memory
    /// effects: returning touches no memory (`builtin.ret` declares that it has none).
    #[test]
    fn a_function_that_only_computes_and_returns_is_free_of_memory_effects() {
        for body in [Body::Returns, Body::PureAndReturns] {
            let (_test, function) = definition(body);
            let function = function.borrow();
            assert!(function.as_operation().is_memory_effect_free());
            let effects = function.as_operation().get_effects_recursively::<MemoryEffect>();
            assert_eq!(effects.map(|effects| effects.len()), Some(0));
        }
    }

    /// `builtin.ret_imm` declares that it has no memory effects, as `builtin.ret` does.
    #[test]
    fn ret_imm_has_no_memory_effects() {
        let mut test = Test::named("ret_imm");
        let i32 = test.context().i32();
        test.with_function("ret_imm", &[], &[i32]);
        let ret = {
            let context = test.context_rc();
            let mut builder = test.function_builder();
            let entry = builder.entry_block();
            let builder = builder.builder_mut();
            builder.set_insertion_point_to_end(entry);
            let value = context.int_attr(crate::Int::from(1i32), i32);
            builder.create::<RetImm, _>(SourceSpan::UNKNOWN)(crate::ReadyAttr(value)).unwrap()
        };
        let ret = ret.borrow();
        assert!(ret.as_operation().is_memory_effect_free());
        assert_eq!(walk(ret.as_operation().as_operation_ref()), []);
    }
}
