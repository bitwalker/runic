use alloc::format;

use crate::{
    AsSymbolRef, Context, IdentAttr, LinkedSymbol, Op, OpParser, OpPrinter, Operation, Report,
    Spanned, Symbol, SymbolName, SymbolRef, SymbolUseList, UnsafeIntrusiveEntityRef, Usable, Value,
    ValueRef, Visibility,
    derive::{EffectOpInterface, operation},
    diagnostics::report,
    dialects::builtin::{
        BuiltinDialect, Yield,
        attributes::{TypeAttr, VisibilityAttr},
    },
    effects::{AlwaysSpeculatable, ConditionallySpeculatable, MemoryEffectOpInterface, Pure},
    parse::ParserError,
    print::AsmPrinter,
    traits::{
        AnyPointer, InferTypeOpInterface, IsolatedFromAbove, NoRegionArguments, RegionYieldsValue,
        SingleBlock, SingleRegion,
    },
};

pub type GlobalVariableRef = UnsafeIntrusiveEntityRef<GlobalVariable>;

/// A [GlobalVariable] represents a named, typed, location in memory.
///
/// Global variables may also specify an initializer, but if not provided, the underlying bytes
/// will be zeroed, which may or may not be a valid instance of the type. It is up to frontends
/// to ensure that an initializer is specified if necessary. The initializer is a region of one
/// block that computes the data, a [Bytes](super::Bytes) for example, and ends with a
/// [Yield](super::Yield) of the value of the global, which has the declared type of the global:
///
/// ```text
/// builtin.global_variable public @data : array<i8; 2> {
///     %b = builtin.bytes "0a0b" : array<i8; 2>;
///     builtin.yield %b;
/// };
/// ```
///
/// The initializer is ordinary code: the yield uses the value, so what computes it is live, and
/// any other operation of the region that nothing uses is dead.
///
/// Global variables, like functions, may also be assigned a visibility. This is only used when
/// resolving symbol uses, and imposes no access restrictions on the code generated from them.
#[operation(
    dialect = BuiltinDialect,
    traits(
        SingleRegion,
        SingleBlock,
        NoRegionArguments,
        IsolatedFromAbove,
        RegionYieldsValue,
    ),
    implements(Symbol, OpPrinter, LinkedSymbol)
)]
pub struct GlobalVariable {
    #[attr]
    name: IdentAttr,
    #[attr]
    visibility: VisibilityAttr,
    #[attr]
    ty: TypeAttr,
    #[region]
    initializer: RegionRef,
    #[default]
    uses: SymbolUseList,
}

impl GlobalVariable {
    #[inline(always)]
    pub fn as_global_var_ref(&self) -> GlobalVariableRef {
        unsafe { GlobalVariableRef::from_raw(self) }
    }

    /// The value the initializer yields: the operand of the `builtin.yield` that ends it, when it
    /// has the declared type of the global.
    ///
    /// It is `None` for a declaration, and for an initializer that does not verify (one that does
    /// not end with a yield, or yields a value of another type).
    pub fn initializer_value(&self) -> Option<ValueRef> {
        let value = self.yielded_value()?;
        (value.borrow().ty() == *self.get_ty()).then_some(value)
    }

    /// The operand of the yield that ends the initializer, whatever its type.
    fn yielded_value(&self) -> Option<ValueRef> {
        let region = self.initializer();
        if region.is_empty() {
            return None;
        }
        let last = region.entry().body().back().as_pointer()?;
        let last = last.borrow();
        let yielded = last.downcast_ref::<Yield>()?;
        if yielded.as_operation().num_operands() != 1 {
            return None;
        }
        Some(yielded.value().as_value_ref())
    }

    /// Checks that a definition's initializer ends with a yield of a value of the declared type.
    fn verify_initializer(&self) -> Result<(), Report> {
        let region = self.initializer();
        if region.is_empty() {
            return Ok(());
        }
        let name = self.get_name().as_str();
        let declared = *self.get_ty();
        let expected =
            "an initializer ends with `builtin.yield`, which gives the value of the global";
        let last = region.entry().body().back().as_pointer();
        let Some(last) = last else {
            return Err(report!(
                message: "invalid operation builtin.global_variable",
                labels: [primary(
                    self.op.span(),
                    format!("the initializer of `@{name}` has no operation, but {expected}")
                )],
            ));
        };
        let last = last.borrow();
        let Some(yielded) = last.downcast_ref::<Yield>() else {
            return Err(report!(
                message: "invalid operation builtin.global_variable",
                labels: [
                    primary(
                        self.op.span(),
                        format!(
                            "the initializer of `@{name}` ends with `{}`, but {expected}",
                            last.name()
                        )
                    ),
                    context(last.span(), "the initializer ends here"),
                ],
            ));
        };
        let ty = yielded.value().ty();
        if ty != declared {
            return Err(report!(
                message: "invalid operation builtin.global_variable",
                labels: [
                    primary(
                        self.op.span(),
                        format!(
                            "the initializer of `@{name}` yields a value of type `{ty}`, but the \
                             global is declared `{declared}`"
                        )
                    ),
                    context(last.span(), "this yields the value"),
                ],
            ));
        }
        Ok(())
    }
}

impl Usable for GlobalVariable {
    type Use = crate::SymbolUse;

    #[inline(always)]
    fn uses(&self) -> &SymbolUseList {
        &self.uses
    }

    #[inline(always)]
    fn uses_mut(&mut self) -> &mut SymbolUseList {
        &mut self.uses
    }
}

impl Symbol for GlobalVariable {
    #[inline(always)]
    fn as_symbol_operation(&self) -> &Operation {
        &self.op
    }

    #[inline(always)]
    fn as_symbol_operation_mut(&mut self) -> &mut Operation {
        &mut self.op
    }

    fn name(&self) -> SymbolName {
        GlobalVariable::name(self).as_symbol()
    }

    fn set_name(&mut self, name: SymbolName) {
        GlobalVariable::set_name(self, name)
    }

    fn visibility(&self) -> Visibility {
        *self.get_visibility()
    }

    fn set_visibility(&mut self, visibility: Visibility) {
        GlobalVariable::set_visibility(self, visibility);
    }

    /// Returns true if this operation is a declaration, rather than a definition, of a symbol
    ///
    /// The default implementation assumes that all operations are definitions
    #[inline]
    fn is_declaration(&self) -> bool {
        self.initializer().is_empty()
    }
}

impl LinkedSymbol for GlobalVariable {}

/// The `link_name` and `section` entries of the dictionary are well formed (see
/// [verify_link_attributes](crate::verify_link_attributes)), and the initializer of a definition
/// ends with a `builtin.yield` of a value of the declared type.
impl crate::Verify<dyn LinkedSymbol> for GlobalVariable {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        crate::verify_link_attributes(&self.op)?;
        self.verify_initializer()
    }
}

impl AsSymbolRef for GlobalVariable {
    fn as_symbol_ref(&self) -> SymbolRef {
        self.as_symbol_operation()
            .as_symbol_ref()
            .expect("global variables must provide a symbol operation")
    }
}

impl OpPrinter for GlobalVariable {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::*;

        printer.print_space();
        printer.print_keyword(self.get_visibility().as_str());
        printer.print_space();
        printer.print_symbol_name(self.get_name().as_symbol());
        *printer += const_text(" : ");
        printer.print_type(*self.get_ty());
        if !self.is_declaration() {
            printer.print_space();
            printer.print_region(&self.initializer());
        }

        if self.op.has_attributes() {
            *printer += const_text(" attributes ");
            printer.print_attribute_dictionary(
                self.op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }
    }
}

impl OpParser for GlobalVariable {
    fn parse(
        state: &mut crate::OperationState,
        parser: &mut dyn crate::OpAsmParser<'_>,
    ) -> crate::ParseResult {
        use crate::parse::Token;

        let visibility = parser
            .parse_keyword_from(&[
                Token::BareIdent("public"),
                Token::BareIdent("private"),
                Token::BareIdent("internal"),
            ])?
            .into_inner()
            .parse::<Visibility>()
            .unwrap();
        state.add_attribute(
            "visibility",
            parser.context_rc().create_attribute::<VisibilityAttr, _>(visibility),
        );

        let name = parser.parse_symbol_name()?;
        state.add_attribute("name", parser.context_rc().create_attribute::<IdentAttr, _>(name));

        let ty = parser.parse_colon_type()?;
        state.add_attribute(
            "ty",
            parser.context_rc().create_attribute::<TypeAttr, _>(ty.into_inner()),
        );

        let initializer =
            parser.parse_optional_region(&[], /*enable_name_shadowing=*/ false)?;
        // We always add the initializer region, even if empty
        state
            .regions
            .push(initializer.unwrap_or_else(|| parser.context().create_region()));

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        Ok(())
    }
}

/// A [GlobalSymbol] reifies the address of a [GlobalVariable] as a value:
/// `%p = builtin.global_symbol @g : ptr<i32>;` for a global `@g : i32`.
///
/// The result is a pointer to the global's declared type, in the address space written in the text
/// (address space 0 when the operation is built). An address inside the global is a `mem.gep` (or a
/// `mem.ptrcast`) of the result. A reference that names nothing is reported by the check every symbol
/// use gets ([Operation::verify_symbol_refs]), which runs before the verifier of the operation.
///
/// The operation is pure: it has no effects, is removed when unused, and two that name the same
/// global with the same type are merged.
#[derive(EffectOpInterface)]
#[operation(
    dialect = BuiltinDialect,
    traits(Pure, AlwaysSpeculatable),
    implements(InferTypeOpInterface, OpPrinter, ConditionallySpeculatable, MemoryEffectOpInterface)
)]
pub struct GlobalSymbol {
    /// The name of the global variable that is referenced
    #[symbol]
    symbol: GlobalVariable,
    /// The address of the global
    #[result]
    addr: AnyPointer,
}

impl GlobalSymbol {
    /// The global variable the reference is linked to, or `None` if it is not linked to one.
    pub fn global(&self) -> Option<GlobalVariableRef> {
        let symbol = self.symbol().resolve()?;
        let symbol = symbol.borrow();
        let global = symbol.as_symbol_operation().downcast_ref::<GlobalVariable>()?;
        Some(global.as_global_var_ref())
    }
}

impl OpPrinter for GlobalSymbol {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::*;

        printer.print_space();
        printer.print_symbol_path(self.get_symbol().path());
        *printer += const_text(" : ");
        printer.print_type(self.addr().ty());

        if self.op.has_attributes() {
            *printer += const_text(" attributes ");
            printer.print_attribute_dictionary(
                self.op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }
    }
}

/// Reads `@symbol : type`, then an optional `attributes {..}`; the type must be a pointer type.
impl OpParser for GlobalSymbol {
    fn parse(
        state: &mut crate::OperationState,
        parser: &mut dyn crate::OpAsmParser<'_>,
    ) -> crate::ParseResult {
        let symbol = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute("symbol", symbol);

        let ty = parser.parse_colon_type()?;

        if !ty.is_pointer() {
            return Err(ParserError::InvalidAttributeValue {
                span: ty.span(),
                reason: format!("expected pointer type, got '{ty}'"),
            });
        }

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        state.results.push(ty.into_inner());

        Ok(())
    }
}

impl ConditionallySpeculatable for GlobalSymbol {
    fn speculatability(&self) -> crate::effects::Speculatability {
        crate::effects::Speculatability::Speculatable
    }
}

/// A result whose type is not yet known gets the type `ptr<T>`, where `T` is the declared type of
/// the global the reference is linked to. A result that already has a type (the text writes one)
/// keeps it, and the verifier checks it against the global.
impl InferTypeOpInterface for GlobalSymbol {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        if !self.addr().ty().is_unknown() {
            return Ok(());
        }
        let Some(global) = self.global() else {
            return Err(report!(
                message: "invalid operation builtin.global_symbol",
                labels: [primary(
                    self.op.span(),
                    format!(
                        "the result type cannot be inferred: '{}' is not linked to a global \
                         variable",
                        self.get_symbol().path()
                    )
                )],
            ));
        };
        let ty = *global.borrow().get_ty();
        self.addr_mut().set_type(context.ptr_type(ty));
        Ok(())
    }
}

/// The result is a pointer to the declared type of the global the reference is linked to.
///
/// A reference that is not linked to a symbol has already been reported by
/// [Operation::verify_symbol_refs], which runs before this.
impl crate::Verify<dyn InferTypeOpInterface> for GlobalSymbol {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let span = self.op.span();
        let Some(global) = self.global() else {
            return Err(report!(
                message: "invalid operation builtin.global_symbol",
                labels: [primary(
                    span,
                    format!(
                        "'{}' is not linked to a global variable",
                        self.get_symbol().path()
                    )
                )],
            ));
        };
        let global = global.borrow();
        let declared = *global.get_ty();
        let ty = self.addr().ty();
        if ty.pointee() == Some(declared) {
            return Ok(());
        }
        Err(report!(
            message: "invalid operation builtin.global_symbol",
            labels: [
                primary(
                    span,
                    format!(
                        "the result has type `{ty}`, but a pointer to the global has type \
                         `ptr<{declared}>` (in any address space)"
                    )
                ),
                context(global.op.span(), format!("the global is declared `{declared}`")),
            ],
        ))
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        rc::Rc,
        string::{String, ToString},
        vec::Vec,
    };

    use super::*;
    use crate::{
        Builder, Ident, Op, OperationRef, SourceSpan,
        dialects::builtin::{
            BuiltinOpBuilder, Module,
            types::{StructField, StructType},
        },
        parse::{ParserConfig, parse_any},
        testing::{
            Test,
            text::{describe, print_generic, round_trip},
        },
    };

    /// A module of a scalar and a struct global, and a function `@f` with `body`.
    fn module(body: &str) -> String {
        format!(
            "builtin.module public @m {{
    builtin.global_variable public @g : i32;
    builtin.global_variable private @s : struct<i32, i8>;
    builtin.function public extern(\"C\") @f() {{
{body}
        builtin.ret;
    }};
}};"
        )
    }

    fn parse_text(source: &str) -> Result<(Rc<Context>, OperationRef), String> {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context.clone());
        match parse_any(config, "globals", source).into_result() {
            Ok(op) => Ok((context, op)),
            Err(report) => Err(describe(&report)),
        }
    }

    const INITIALIZED: &str = "builtin.module public @m {
    builtin.global_variable public @g : array<i8; 2> {
        %b = builtin.bytes \"0a0b\" : array<i8; 2>;
        %unused = builtin.bytes \"0c0d\" : array<i8; 2>;
        builtin.yield %b;
    };
};";

    /// In an initializer, the operation that computes the yielded value is not trivially dead (the
    /// yield uses it) and an unused one is, as the same operation is in a function.
    #[test]
    fn in_an_initializer_the_operation_that_computes_the_value_is_live_and_an_unused_one_is_dead() {
        let (_context, module) = parse_text(INITIALIZED).unwrap_or_else(|err| panic!("{err}"));
        let mut dead = Vec::new();
        module.borrow().prewalk_all(|op: &Operation| {
            if op.is::<crate::dialects::builtin::Bytes>() {
                dead.push((op.is_trivially_dead(), op.would_be_trivially_dead()));
            }
        });
        assert_eq!(dead, [(false, true), (true, true)]);
    }

    /// The initializer's value is the operand of its yield; a declaration has none.
    #[test]
    fn the_initializer_value_is_the_yielded_value() {
        let source = "builtin.module public @m {
    builtin.global_variable public @g : array<i8; 2> {
        %b = builtin.bytes \"0a0b\" : array<i8; 2>;
        builtin.yield %b;
    };
    builtin.global_variable public @d : i32;
};";
        let (_context, module) = parse_text(source).unwrap_or_else(|err| panic!("{err}"));
        let mut values = Vec::new();
        module.borrow().prewalk_all(|op: &Operation| {
            if let Some(global) = op.downcast_ref::<GlobalVariable>() {
                let value = global.initializer_value();
                let defined_by_bytes = value.is_some_and(|value| {
                    value
                        .borrow()
                        .get_defining_op()
                        .is_some_and(|def| def.borrow().is::<crate::dialects::builtin::Bytes>())
                });
                values.push((global.get_name().as_str().to_string(), defined_by_bytes));
            }
        });
        assert_eq!(values, [("g".to_string(), true), ("d".to_string(), false)]);
    }

    /// The initializer is checked in the custom and the generic form: the type of the yielded
    /// value is the declared type of the global.
    #[test]
    fn the_yielded_value_has_the_declared_type_of_the_global() {
        let source = "builtin.module public @m {
    builtin.global_variable public @g : i32 {
        %b = builtin.bytes \"0a0b\" : array<i8; 2>;
        builtin.yield %b;
    };
};";
        let err = parse_text(source).map(|_| ()).expect_err("the types differ");
        assert!(err.contains("invalid operation builtin.global_variable"), "{err}");
        assert!(
            err.contains(
                "the initializer of `@g` yields a value of type `array<i8; 2>`, but the global is \
                 declared `i32`"
            ),
            "{err}"
        );

        // The generic form is checked as well: the global as the generic printer writes it
        let generic = |ty: &str| {
            format!(
                "builtin.module public @m {{
    \"builtin.global_variable\"() <{{ name = #builtin.ident<@g>, visibility = \
                 #builtin.visibility<public>, ty = #builtin.type<{ty}> }}> ({{
        %0 = builtin.bytes \"0a0b\" : array<i8; 2>;
        builtin.yield %0;
    }}) : () -> ();
}};"
            )
        };
        parse_text(&generic("array<i8; 2>")).unwrap_or_else(|err| panic!("{err}"));
        let err = parse_text(&generic("i32"))
            .map(|_| ())
            .expect_err("the generic form is checked");
        assert!(
            err.contains(
                "the initializer of `@g` yields a value of type `array<i8; 2>`, but the global is \
                 declared `i32`"
            ),
            "{err}"
        );
    }

    /// An initializer that ends with another terminator, or has no operation, says what it ends
    /// with and what it should end with.
    #[test]
    fn an_initializer_that_does_not_end_with_a_yield_is_an_error() {
        let source = "builtin.module public @m {
    builtin.global_variable public @g : i32 {
        builtin.ret;
    };
};";
        let err = parse_text(source).map(|_| ()).expect_err("no yield");
        assert!(
            err.contains(
                "the initializer of `@g` ends with `builtin.ret`, but an initializer ends with \
                 `builtin.yield`, which gives the value of the global"
            ),
            "{err}"
        );
        let source = "builtin.module public @m {
    builtin.global_variable public @g : i32 {
    ^bb0:
    };
};";
        let err = parse_text(source).map(|_| ()).expect_err("no operation");
        assert!(
            err.contains(
                "the initializer of `@g` has no operation, but an initializer ends with \
                 `builtin.yield`, which gives the value of the global"
            ),
            "{err}"
        );
    }

    /// The pointer to a global has the global's type, in the address space written; the module
    /// round trips in both forms.
    #[test]
    fn a_global_symbol_points_to_its_globals_type() {
        let source = module(
            "        %p = builtin.global_symbol @g : ptr<i32>;
        %q = builtin.global_symbol @s : ptr<struct<i32, i8>, 3>;
        %r = builtin.global_symbol ::@m::@g : ptr<i32>;",
        );
        let (_context, module) = parse_text(&source).unwrap_or_else(|err| panic!("{err}"));
        let mut types = Vec::new();
        module.borrow().prewalk_all(|op: &Operation| {
            if let Some(symbol) = op.downcast_ref::<GlobalSymbol>() {
                types.push(symbol.addr().ty().to_string());
            }
        });
        assert_eq!(types, ["ptr<i32>", "ptr<struct<i32, i8>, 3>", "ptr<i32>"]);

        let mut skipped = Vec::new();
        let failures = round_trip::<Module>(module, true, &mut skipped);
        assert!(failures.is_empty(), "{}", failures.join("\n\n"));
        assert!(skipped.is_empty(), "skipped: {skipped:#?}");
    }

    /// A `builtin.global_symbol` that names a function, not a global variable, is a diagnostic.
    #[test]
    fn a_global_symbol_of_a_function_is_an_error() {
        let source = "builtin.module public @m {
    builtin.function public extern(\"C\") @h(i32) -> i32;
    builtin.function public extern(\"C\") @f() {
        %p = builtin.global_symbol @h : ptr<i32>;
        builtin.ret;
    };
};";
        let err = parse_text(source).map(|_| ()).expect_err("a function is not a global");
        assert!(err.contains("is not linked to a global variable"), "{err}");
    }

    /// A pointer to another type, an opaque pointer and a type that is not a pointer are each
    /// rejected, in the custom form and in the generic form.
    #[test]
    fn a_global_symbol_of_another_type_is_an_error() {
        let err = parse_text(&module("        %p = builtin.global_symbol @g : ptr<i8>;"))
            .map(|_| ())
            .expect_err("the pointee is not the global's type");
        assert!(err.contains("invalid operation builtin.global_symbol"), "{err}");
        assert!(
            err.contains(
                "the result has type `ptr<i8>`, but a pointer to the global has type `ptr<i32>`"
            ),
            "{err}"
        );
        assert!(err.contains("the global is declared `i32`"), "{err}");

        let err = parse_text(&module("        %p = builtin.global_symbol @s : ptr;"))
            .map(|_| ())
            .expect_err("an opaque pointer has no pointee");
        assert!(err.contains("the result has type `ptr`"), "{err}");

        let err = parse_text(&module("        %p = builtin.global_symbol @g : i32;"))
            .map(|_| ())
            .expect_err("the result is a pointer");
        assert!(err.contains("expected pointer type, got 'i32'"), "{err}");

        // The operation names a global and has no byte offset: `@g+8` does not parse
        let err = parse_text(&module("        %p = builtin.global_symbol @g+8 : ptr<i32>;"))
            .map(|_| ())
            .expect_err("an offset is not accepted");
        assert!(err.contains("unexpected token '+'"), "{err}");

        // The generic form states the result type itself, and the verifier checks it
        let (_context, parsed) =
            parse_text(&module("        %p = builtin.global_symbol @g : ptr<i32>;")).unwrap();
        let mut generic = None;
        parsed.borrow().prewalk_all(|op: &Operation| {
            if op.is::<GlobalSymbol>() {
                generic = Some((op.to_string(), print_generic(op)));
            }
        });
        let (custom, generic) = generic.unwrap();
        assert!(generic.contains("-> ptr<i32>"), "{generic}");
        let text = parsed
            .borrow()
            .to_string()
            .replace(custom.trim(), &generic.replace("-> ptr<i32>", "-> ptr<i8>"));
        let err = parse_text(&text).map(|_| ()).expect_err("the generic form is checked");
        assert!(err.contains("the result has type `ptr<i8>`"), "{err}");
        let text = parsed
            .borrow()
            .to_string()
            .replace(custom.trim(), &generic.replace("-> ptr<i32>", "-> i32"));
        let err = parse_text(&text).map(|_| ()).expect_err("the generic form is checked");
        assert!(err.contains("builtin.global_symbol"), "{err}");
    }

    /// The builder types the result from the global: a pointer to its type in address space 0.
    #[test]
    fn the_builder_types_the_result_from_the_global() {
        let mut test = Test::named("f").in_module("m");
        let context = test.context_rc();
        let s = context
            .struct_type(StructType::new([context.i32(), context.i8()].map(StructField::new)));
        let module = test.module();
        let global = {
            let b = test.builder_mut();
            let body = module.borrow().body().entry_block_ref().unwrap();
            b.set_insertion_point_to_end(body);
            b.create_global_variable(Ident::with_empty_span("s".into()), Visibility::Public, s)
                .unwrap()
        };
        test.with_function("f", &[], &[]);
        let symbol = {
            let mut builder = test.function_builder();
            let symbol = builder.global_symbol(global, SourceSpan::UNKNOWN).unwrap();
            builder.ret([], SourceSpan::UNKNOWN).unwrap();
            symbol
        };
        test.module().borrow().as_operation().recursively_verify().unwrap();
        let symbol = symbol.borrow();
        assert_eq!(symbol.addr().ty(), context.ptr_type(s));
        assert_eq!(symbol.addr().ty().as_pointer().unwrap().addrspace(), 0);
        assert!(symbol.global() == Some(global));
        assert!(symbol.as_operation().is_memory_effect_free());
        assert!(
            symbol
                .as_operation()
                .to_string()
                .contains("builtin.global_symbol ::@m::@s : ptr<struct<i32, i8>>"),
            "{}",
            symbol.as_operation()
        );
    }
}
