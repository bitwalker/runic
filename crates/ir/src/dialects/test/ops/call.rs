use crate::{
    derive::{OpParser, OpPrinter, operation},
    dialects::{
        builtin::attributes::{BoolAttr, Signature, SignatureAttr},
        test::TestDialect,
    },
    effects::*,
    traits::*,
    *,
};

/// An `i32` whose memory effects are a `Read` when `known` is true, and unknown otherwise:
/// `%r = test.effects <{ known = #builtin.bool<true> }>;`.
///
/// It stands for an operation that implements [MemoryEffectOpInterface] and yet cannot always list
/// its effects, as a call to a callee that states none.
#[derive(OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Effects {
    #[attr]
    known: BoolAttr,
    #[result]
    result: Int32,
}

impl InferTypeOpInterface for Effects {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        let i32 = context.i32();
        self.result_mut().set_type(i32);
        Ok(())
    }
}

impl EffectOpInterface<MemoryEffect> for Effects {
    fn effects(&self) -> EffectIterator<MemoryEffect> {
        if *self.get_known() {
            EffectIterator::new([EffectInstance::new(MemoryEffect::Read)])
        } else {
            EffectIterator::new([])
        }
    }

    fn effects_are_known(&self) -> bool {
        *self.get_known()
    }
}

/// A call of one of the two callables `first` and `second`, chosen at runtime by `index`:
/// `%r = test.call_one_of %index, @first, @second(%a) : extern("C") (i32) -> i32;`.
///
/// The set of callables the call may reach is known: the canonical targets of its two symbols, so
/// an analysis that follows calls sees the call as a caller of each. Its callee is the value
/// `index`, not a symbol, so neither symbol is a call use of the callable it names, and a callable
/// either names has its address taken. The call has one argument per parameter of its signature
/// and one result per result of it, of the same types; the results are made from the signature
/// when the call is built or parsed. It implements no effect interface, so its effects are
/// unknown. Its signature is not compared with its callees' signatures: the verifier checks only
/// that its arguments and results agree with its own.
#[operation(
    dialect = TestDialect,
    implements(CallOpInterface, InferTypeOpInterface, OpPrinter)
)]
pub struct CallOneOf {
    /// Which callable is called
    #[operand]
    index: Int32,
    /// The arguments, one per parameter of the signature
    #[operands]
    arguments: AnyType,
    /// One callable the call may reach: a function, or an alias of one
    #[symbol(callable)]
    first: SymbolPath,
    /// The other callable the call may reach: a function, or an alias of one
    #[symbol(callable)]
    second: SymbolPath,
    /// The signature the call is made with
    #[attr(hidden)]
    signature: SignatureAttr,
    /// The results, one per result of the signature
    #[results]
    returns: AnyType,
}

/// One result per result of the signature, in order. A call that already has results (the generic
/// form gives them) keeps them, and the verifier checks them against the signature.
impl InferTypeOpInterface for CallOneOf {
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

impl CallOpInterface for CallOneOf {
    /// The value `index`: the callee is chosen at runtime.
    #[inline]
    fn callable_for_callee(&self) -> Callable {
        Callable::Value(self.index().as_value_ref())
    }

    /// Makes `callable`, a value, the index.
    ///
    /// # Panics
    ///
    /// Panics if `callable` is a symbol: the callee of `test.call_one_of` is its index.
    fn set_callee(&mut self, callable: Callable) {
        match callable {
            Callable::Value(value) => self.index_mut().set(value),
            Callable::Symbol(path) => panic!(
                "the callee of test.call_one_of is its index, a value, not the symbol '{path}'"
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

    /// The canonical targets of `first` and `second`, once each: two aliases of one function give
    /// that function alone. `None` if either symbol does not resolve to a callable.
    fn possible_callees(&self) -> Option<SmallVec<[CanonicalCallableRef; 2]>> {
        let first = self.first().resolve_callable().ok()?.target();
        let second = self.second().resolve_callable().ok()?.target();
        let mut callees = SmallVec::from_iter([first]);
        if second != first {
            callees.push(second);
        }
        Some(callees)
    }

    /// The signature the call carries.
    fn callee_signature(&self) -> Option<Signature> {
        Some(self.get_signature().clone())
    }
}

/// The call's arguments and results agree with its signature.
impl Verify<dyn CallOpInterface> for CallOneOf {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        crate::conversion::verify_call_signature_operands_and_results(
            self.as_operation(),
            &self.get_signature(),
        )
    }
}

impl OpPrinter for CallOneOf {
    /// The index is an `i32`, and the arguments take the types of the parameters of the written
    /// signature.
    fn states_operand_types(&self) -> bool {
        true
    }

    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        use crate::formatter::const_text;

        printer.print_space();
        printer.print_value_uses(self.operands().group(0).as_value_range());
        *printer += const_text(", ");
        printer.print_symbol_path(self.first().path());
        *printer += const_text(", ");
        printer.print_symbol_path(self.second().path());
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

/// Reads `%index, @first, @second(%a, ..) : signature`, then an optional `attributes {..}`. The
/// index is an `i32`, and the arguments take the types of the parameters of the signature.
impl OpParser for CallOneOf {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> parse::ParseResult {
        let index = parser.parse_operand(false)?;
        parser.parse_comma()?;
        let first = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute("first", first);
        parser.parse_comma()?;
        let second = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute("second", second);

        let mut arguments = SmallVec::<[parse::UnresolvedOperand; 2]>::default();
        parser.parse_operand_list(&mut arguments, parse::Delimiter::Paren, true, None)?;

        parser.parse_colon()?;
        let (signature, params) = crate::dialects::builtin::parse_signature(parser)?;
        state.add_attribute(
            "signature",
            parser.context_rc().create_attribute::<SignatureAttr, _>(signature),
        );

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let i32 = parser.context().i32();
        let index = parser.resolve_operand(index, i32)?;
        state.operands.push(SmallVec::from_iter([index]));
        let mut values = SmallVec::default();
        parser.resolve_operands(state.span, &arguments, &params, &mut values)?;
        state.operands.push(values);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, string::ToString, vec::Vec};

    use crate::{
        CallOpInterface, Callable, Op, Report, SourceSpan, ValueRef,
        dialects::{
            builtin::{BuiltinOpBuilder, Module, ModuleBuilder, attributes::Signature},
            test::{CallOneOf, TestOpBuilder},
        },
        effects::MemoryEffect,
        parse::{ParserConfig, parse},
        testing::{Test, text::describe},
    };

    const SOURCE: &str = r#"
builtin.module public @m {
    builtin.function private extern("C") @left(i32) -> i32;
    builtin.function private extern("C") @right(i32) -> i32;
    builtin.function_alias private @alias -> @left;
    builtin.function public extern("C") @caller(%index: i32, %a: i32) -> i32 {
        %r = test.call_one_of %index, @left, @right(%a) : extern("C") (i32) -> i32;
        %s = test.call_one_of %index, @alias, @left(%r) : extern("C") (i32) -> i32;
        builtin.ret %s : (i32);
    };
};
"#;

    /// The `test.call_one_of`s in `@caller` of `module`, in order.
    fn calls(module: crate::dialects::builtin::ModuleRef) -> Vec<crate::OperationRef> {
        ModuleBuilder::new(module)
            .get_function("caller")
            .unwrap()
            .borrow()
            .entry_block()
            .borrow()
            .body()
            .iter()
            .filter_map(|op| op.downcast_ref::<CallOneOf>().map(Op::as_operation_ref))
            .collect()
    }

    /// The text round-trips; the callee is the index, the arguments the second operand group, and
    /// the possible callees the canonical targets of the two symbols, once each.
    #[test]
    fn a_call_one_of_calls_either_symbols_target_through_its_index() {
        let test = Test::default();
        let module =
            parse::<Module>(ParserConfig::new(test.context_rc()), "call_one_of.runic", SOURCE)
                .unwrap();
        let text = format!("{}", module.borrow().as_operation());
        assert!(
            text.contains("test.call_one_of %0, @left, @right(%1) : extern(\"C\") (i32) -> i32;"),
            "{text}"
        );
        // A fresh context numbers the values from zero again
        let other = Test::default();
        let reparsed =
            parse::<Module>(ParserConfig::new(other.context_rc()), "reparsed.runic", &text)
                .unwrap();
        assert_eq!(format!("{}", reparsed.borrow().as_operation()), text);

        let mb = ModuleBuilder::new(module);
        let left = mb.resolve_callable("left").unwrap().target();
        let right = mb.resolve_callable("right").unwrap().target();
        let caller = mb.get_function("caller").unwrap();
        let entry = caller.borrow().entry_block();
        let [index, a] = {
            let entry = entry.borrow();
            [entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef]
        };

        let [both, aliased] = <[_; 2]>::try_from(calls(module)).unwrap();
        let both = both.borrow();
        let both = both.downcast_ref::<CallOneOf>().unwrap();
        assert_eq!(both.callable_for_callee().unwrap_value_ref(), index);
        let passed = both.arguments().iter().map(|o| o.borrow().as_value_ref()).collect::<Vec<_>>();
        assert_eq!(passed, [a]);
        assert_eq!(both.possible_callees().unwrap().as_slice(), &[left, right]);
        assert!(both.resolve().is_none(), "the callee is a value");
        assert_eq!(both.callee_signature().unwrap().to_string(), "extern(\"C\") (i32) -> i32");
        assert!(both.as_operation().get_effects_recursively::<MemoryEffect>().is_none());

        let aliased = aliased.borrow();
        let aliased = aliased.downcast_ref::<CallOneOf>().unwrap();
        assert_eq!(aliased.possible_callees().unwrap().as_slice(), &[left]);
    }

    /// A module holding `@left` and `@right`, both `(i32) -> i32`, and the test's function
    /// `@caller(i32, i32) -> i32`.
    fn caller_of_two() -> (Test, [crate::dialects::builtin::FunctionRef; 2]) {
        let mut test = Test::named("caller").in_module("calls");
        let i32 = test.context().i32();
        let left = test.define_function("left", &[i32], &[i32]);
        let right = test.define_function("right", &[i32], &[i32]);
        test.with_function("caller", &[i32, i32], &[i32]);
        (test, [left, right])
    }

    /// Builds `test.call_one_of` of `left` and `right` with `signature` (theirs when `None`),
    /// passing `argument`, and returns its result.
    fn build(
        test: &mut Test,
        [left, right]: [crate::dialects::builtin::FunctionRef; 2],
        signature: Option<Signature>,
        argument: ValueRef,
    ) -> crate::UnsafeIntrusiveEntityRef<CallOneOf> {
        let index = test.entry_block().borrow().arguments()[0] as ValueRef;
        let signature = signature.unwrap_or_else(|| left.borrow().get_signature().clone());
        let mut builder = test.function_builder();
        let call = builder
            .call_one_of(index, left, right, signature, [argument], SourceSpan::UNKNOWN)
            .unwrap();
        builder.ret_imm(crate::Int::from(0i32), SourceSpan::UNKNOWN).unwrap();
        call
    }

    fn verify(test: &Test) -> Result<(), Report> {
        test.module().borrow().as_operation().recursively_verify()
    }

    #[test]
    fn set_callee_of_a_call_one_of_replaces_the_index() {
        let (mut test, functions) = caller_of_two();
        let argument = test.entry_block().borrow().arguments()[1] as ValueRef;
        let mut call = build(&mut test, functions, None, argument);
        verify(&test).unwrap();
        CallOpInterface::set_callee(&mut *call.borrow_mut(), Callable::Value(argument));
        verify(&test).unwrap();
        assert_eq!(call.borrow().callable_for_callee().unwrap_value_ref(), argument);
        assert_eq!(call.borrow().index().as_value_ref(), argument);
    }

    #[test]
    #[should_panic(expected = "the callee of test.call_one_of is its index")]
    fn set_callee_of_a_call_one_of_to_a_symbol_panics() {
        let (mut test, functions) = caller_of_two();
        let argument = test.entry_block().borrow().arguments()[1] as ValueRef;
        let mut call = build(&mut test, functions, None, argument);
        let path = functions[0].borrow().as_operation().as_symbol().unwrap().path();
        CallOpInterface::set_callee(&mut *call.borrow_mut(), path.into());
    }

    /// The verifier checks the arguments and results against the signature.
    #[test]
    fn a_call_one_of_whose_argument_disagrees_with_its_signature_is_an_error() {
        let (mut test, functions) = caller_of_two();
        let argument = test.entry_block().borrow().arguments()[1] as ValueRef;
        let signature = {
            let (i32, i64) = (test.context().i32(), test.context().i64());
            Signature::new(&test.context_rc(), [i64], [i32])
        };
        build(&mut test, functions, Some(signature), argument);
        let error = describe(&verify(&test).expect_err("the call must not verify"));
        assert!(error.contains("invalid operation test.call_one_of"), "{error}");
        assert!(
            error.contains(
                "argument 0 has type `i32`, but parameter 0 of the signature has type `i64`"
            ),
            "{error}"
        );
    }
}
