use alloc::format;

use crate::{
    CallableSymbol, LinkedSymbol, Op, OpParser, OpPrinter, Operation, Symbol, SymbolName,
    SymbolRef, SymbolUseList, Usable, Visibility,
    derive::operation,
    dialects::builtin::{
        BuiltinDialect, FunctionRef,
        attributes::{IdentAttr, VisibilityAttr},
    },
    print::AsmPrinter,
};

pub type FunctionAliasRef = crate::UnsafeIntrusiveEntityRef<FunctionAlias>;

/// References a function under another name.
///
/// The signature and body belong to the resolved (canonical) target, as an alias has no region of
/// its own. An alias introduces a new name even if its target is only a declaration.
///
/// Alias chains are supported. Each hop is resolved in the symbol table of the alias being followed
/// (see [Self::resolve_target]), meaning an alias can only reference symbols resolvable from
/// its own table. The canonical target must be callable (see the `Verify<dyn CallableSymbol>`
/// implementation), but it does not need to live in the same symbol table as the aliases pointing
/// to it.
#[operation(
    dialect = BuiltinDialect,
    implements(Symbol, CallableSymbol, LinkedSymbol, OpPrinter)
)]
pub struct FunctionAlias {
    #[attr]
    name: IdentAttr,
    #[attr]
    linkage: VisibilityAttr,
    #[symbol(callable)]
    target: crate::SymbolPath,
    #[default]
    uses: SymbolUseList,
}

impl FunctionAlias {
    #[inline(always)]
    pub fn as_function_alias_ref(&self) -> FunctionAliasRef {
        unsafe { FunctionAliasRef::from_raw(self) }
    }

    /// Resolve the target in the alias's own table.
    pub fn resolve_target(&self) -> Option<SymbolRef> {
        let table = self.as_operation().nearest_symbol_table()?;
        let table = table.borrow();
        table.as_symbol_table()?.resolve(self.target().path())
    }

    /// Canonical target of `symbol`, following `FunctionAlias` hops in their own tables.
    ///
    /// Returns `symbol` itself when it is not an alias. Returns `None` on unresolvable
    /// hop, cycle, or excessive depth. Use [SymbolRef::resolve_canonical] to retain the
    /// error cause.
    pub fn canonicalize(symbol: SymbolRef) -> Option<SymbolRef> {
        symbol.resolve_canonical().ok()
    }

    /// Like [`Self::canonicalize`], but downcasts to `Function`.
    pub fn canonicalize_function(symbol: SymbolRef) -> Option<FunctionRef> {
        symbol.resolve_function().ok()
    }
}

impl Usable for FunctionAlias {
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

impl Symbol for FunctionAlias {
    #[inline(always)]
    fn as_symbol_operation(&self) -> &Operation {
        &self.op
    }

    #[inline(always)]
    fn as_symbol_operation_mut(&mut self) -> &mut Operation {
        &mut self.op
    }

    fn name(&self) -> SymbolName {
        self.get_name().as_symbol()
    }

    fn set_name(&mut self, name: SymbolName) {
        self.get_name_mut().name = name;
    }

    fn visibility(&self) -> Visibility {
        *self.get_linkage()
    }

    fn set_visibility(&mut self, visibility: Visibility) {
        *self.get_linkage_mut() = visibility;
    }

    #[inline]
    fn is_declaration(&self) -> bool {
        false
    }
}

impl CallableSymbol for FunctionAlias {}

impl OpParser for FunctionAlias {
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
            .expect("visibilities above are exhaustive");
        state.add_attribute(
            "linkage",
            parser.context_rc().create_attribute::<VisibilityAttr, _>(visibility),
        );

        let name = parser.parse_symbol_name()?;
        state.add_attribute("name", parser.context_rc().create_attribute::<IdentAttr, _>(name));

        parser.parse_arrow()?;
        let target = parser.parse_symbol_ref()?;
        state.attrs.push(crate::NamedAttribute::new("target", target.into_inner()));

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

impl OpPrinter for FunctionAlias {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::const_text;

        printer.print_space();
        printer.print_keyword(self.get_linkage().as_str());
        printer.print_space();
        printer.print_symbol_name(self.get_name().as_symbol());
        printer.print_space();
        *printer += const_text("->");
        printer.print_space();
        printer.print_symbol_path(self.target().path());
        if self.op.has_attributes() {
            printer.print_space();
            *printer += const_text("attributes");
            printer.print_space();
            printer.print_attribute_dictionary(
                self.op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }
    }
}

impl LinkedSymbol for FunctionAlias {}

/// The `link_name` entry of the dictionary is well formed, and there is no `section` entry (see
/// [verify_link_attributes](crate::verify_link_attributes)).
impl crate::Verify<dyn crate::LinkedSymbol> for FunctionAlias {
    fn verify(&self, _context: &crate::Context) -> Result<(), crate::Report> {
        crate::verify_link_attributes(self.as_operation())
    }
}

impl crate::Verify<dyn CallableSymbol> for FunctionAlias {
    fn verify(&self, _context: &crate::Context) -> Result<(), crate::Report> {
        use crate::Spanned;

        let span = self.as_operation().span();
        let callee = self
            .as_operation()
            .as_symbol_ref()
            .expect("function aliases are symbols")
            .resolve_callable()
            .map_err(|err| {
                crate::diagnostics::report!(
                    message: (format!(
                        "invalid builtin.function_alias '{}': {err}",
                        self.get_name().as_str()
                    )),
                    labels: [
                        primary(span, "cannot resolve this alias to a callable"),
                    ],
                )
            })?;

        // Private symbols are only accessible from their defining symbol table.
        let target = self.resolve_target().expect("callable resolution validated the target");
        let target = target.borrow();
        if target.is_private()
            && target.as_symbol_operation().nearest_symbol_table()
                != self.as_operation().nearest_symbol_table()
        {
            return Err(crate::diagnostics::report!(
                message: (format!(
                    "invalid builtin.function_alias '{}': target '{}' is private to another \
                     symbol table",
                    self.get_name().as_str(),
                    target.path()
                )),
                labels: [
                    primary(span, "cannot alias a private symbol in another symbol table"),
                    context(
                        target.as_symbol_operation().span(),
                        "the private target is defined here"
                    ),
                ],
            ));
        }

        // A non-private alias must not expose a private canonical callable.
        let canonical = callee.target().as_symbol_ref();
        let canonical = canonical.borrow();
        if !self.visibility().is_private() && canonical.is_private() {
            return Err(crate::diagnostics::report!(
                message: (format!(
                    "invalid builtin.function_alias '{}': {} alias cannot expose private target \
                     '{}'",
                    self.get_name().as_str(),
                    self.visibility().as_str(),
                    canonical.path()
                )),
                labels: [
                    primary(span, "this alias exposes a private callable"),
                    context(
                        canonical.as_symbol_operation().span(),
                        "the private canonical target is defined here"
                    ),
                ],
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, string::ToString};

    use super::*;
    use crate::{
        CallableOpInterface, Context, Ident, SymbolTable,
        dialects::builtin::{Module, ModuleBuilder, ModuleRef, attributes::Signature},
        parse::{ParserConfig, parse},
        testing::Test,
    };

    fn module_with_alias() -> (Rc<Context>, ModuleRef, FunctionAliasRef) {
        let mut test = Test::default().in_module("test");
        let i32 = test.context().i32();
        let primary = test.define_function("foo", &[], &[i32]);
        let module = test.module();
        let alias = {
            let mut mb = ModuleBuilder::new(module);
            mb.define_function_alias(
                Ident::with_empty_span(SymbolName::intern("bar")),
                Visibility::Public,
                primary,
            )
            .unwrap()
        };
        let ctx = module.borrow().as_operation().context_rc();
        (ctx, module, alias)
    }

    #[test]
    fn alias_resolves_to_primary_and_verifies() {
        let (_ctx, module, alias) = module_with_alias();
        module.borrow().as_operation().recursively_verify().unwrap();
        let mb = ModuleBuilder::new(module);
        // Primary and alias both resolve to the primary definition.
        let foo = mb.get_function("foo").expect("primary should resolve");
        let bar_resolved = mb.resolve_function("bar").expect("alias should resolve to primary");
        assert!(foo == bar_resolved);
        assert!(mb.resolve_function("foo").unwrap() == foo);
        assert!(mb.get_function("bar").is_none());
        assert!(mb.get_function("missing").is_none());
        assert!(mb.resolve_function("missing").is_none());
        let bar_alias = mb.get_function_alias("bar").expect("alias should exist");
        assert!(bar_alias == alias);
        assert!(mb.get_function_alias("foo").is_none());
    }

    #[test]
    fn alias_is_callable_symbol_but_not_callable_op_interface() {
        let (_ctx, module, _alias) = module_with_alias();
        let mb = ModuleBuilder::new(module);
        let alias = mb.get_function_alias("bar").unwrap();
        let alias_borrow = alias.borrow();
        let op = alias_borrow.as_operation();
        assert!(op.implements::<dyn CallableSymbol>());
        assert!(!op.implements::<dyn CallableOpInterface>());
    }

    #[test]
    fn alias_print_parse_roundtrip_with_forward_references() {
        let test = Test::default();
        let source = r#"
builtin.module public @test {
    builtin.function_alias public @bar -> @first;
    builtin.function_alias private @first -> @foo;
    builtin.function public extern("C") @foo(%arg: i32) -> i32 {
        builtin.ret %arg : (i32);
    };
};
"#;
        let module =
            parse::<Module>(ParserConfig::new(test.context_rc()), "alias.runic", source).unwrap();
        let mb = ModuleBuilder::new(module);
        assert!(mb.resolve_function("bar").unwrap() == mb.get_function("foo").unwrap());

        let printed = module.borrow().as_operation().to_string();
        for line in printed.lines().filter(|line| line.contains("builtin.function_alias")) {
            assert!(!line.contains("extern"), "{line}");
            assert!(!line.contains(" : "), "{line}");
        }

        let reparse_test = Test::default();
        let reparsed = parse::<Module>(
            ParserConfig::new(reparse_test.context_rc()),
            "alias.reparsed.runic",
            &printed,
        )
        .unwrap();
        assert_eq!(printed, reparsed.borrow().as_operation().to_string());
    }

    #[test]
    fn alias_missing_target_fails_verification() {
        let mut test = Test::default().in_module("test");
        let i32 = test.context().i32();
        let primary = test.define_function("foo", &[], &[i32]);
        let mut module = test.module();
        {
            let mut mb = ModuleBuilder::new(module);
            mb.define_function_alias(
                Ident::with_empty_span(SymbolName::intern("bar")),
                Visibility::Public,
                primary,
            )
            .unwrap();
        }

        // Delete the target
        {
            module.borrow_mut().remove(SymbolName::intern("foo"));
        }
        let err = module.borrow().as_operation().recursively_verify().unwrap_err();
        assert!(format!("{err}").contains("does not resolve"), "{err}");
    }

    /// A parsed alias whose target names nothing is reported by the check every operation's
    /// symbol uses get, which runs before the alias's own verifier.
    #[test]
    fn alias_to_undeclared_target_is_the_symbol_use_check_error() {
        let err = verify_alias_source(
            "builtin.module public @test { builtin.function_alias public @bar -> @nowhere; };",
        )
        .expect_err("an alias of an undeclared symbol must not verify");
        let message = format!("{err}");
        assert!(message.contains("symbol '@nowhere' does not resolve"), "{message}");
        assert!(!message.contains("invalid builtin.function_alias 'bar'"), "{message}");
    }

    /// An alias of a global variable: the target resolves, to a symbol that is not callable.
    #[test]
    fn alias_to_non_callable_fails_verification() {
        let err = verify_alias_source(
            "builtin.module public @test {\n    builtin.global_variable public @foo : i32;\n    \
             builtin.function_alias public @bar -> @foo;\n};",
        )
        .expect_err("an alias of a global variable must not verify");
        assert!(format!("{err}").contains("not callable"), "{err}");
    }

    /// Removing the target unlinks the alias's use of it; a symbol inserted later under the same
    /// name does not relink it, so the target is reported as not linked to the symbol its path now
    /// names, before the alias's own verifier would look the path up.
    #[test]
    fn alias_whose_target_is_removed_and_replaced_is_not_linked_to_the_replacement() {
        let mut test = Test::default().in_module("test");
        let i32 = test.context().i32();
        let primary = test.define_function("foo", &[], &[i32]);
        let mut module = test.module();
        {
            let mut mb = ModuleBuilder::new(module);
            mb.define_function_alias(
                Ident::with_empty_span(SymbolName::intern("bar")),
                Visibility::Public,
                primary,
            )
            .unwrap();
        }
        // Replace the callable target with a with non-callable global with same name
        {
            module.borrow_mut().remove(SymbolName::intern("foo"));
            let mut mb = ModuleBuilder::new(module);
            mb.define_global_variable(
                Ident::with_empty_span(SymbolName::intern("foo")),
                Visibility::Public,
                i32,
            )
            .unwrap();
            assert!(mb.get_function("foo").is_none());
            assert!(mb.resolve_function("foo").is_none());
            assert!(mb.resolve_function("bar").is_none());
        }
        let err = module.borrow().as_operation().recursively_verify().unwrap_err();
        let err = crate::testing::text::describe(&err);
        assert!(
            err.contains("symbol '::@test::@foo' is not linked to the symbol its path names"),
            "{err}"
        );
        assert!(
            err.contains(
                "this path names '::@test::@foo' (a 'builtin.global_variable') from \
                 'builtin.function_alias', but the reference is linked to no symbol"
            ),
            "{err}"
        );
    }

    #[test]
    fn alias_chain_observes_target_signature_changes() {
        let (ctx, module, _alias) = module_with_alias();
        let mut mb = ModuleBuilder::new(module);
        let alias = mb.get_function_alias("bar").unwrap();
        mb.define_function_alias(Ident::from("chain"), Visibility::Public, alias)
            .unwrap();
        let mut target = mb.get_function("foo").unwrap();
        let updated_signature = Signature::new(&ctx, [ctx.i32()], [ctx.i64()]);
        *target.borrow_mut().get_signature_mut() = updated_signature.clone();

        module.borrow().as_operation().recursively_verify().unwrap();
        assert_eq!(
            mb.resolve_function("chain").unwrap().borrow().get_signature().clone(),
            updated_signature
        );
    }

    /// Parses and verifies a module from its textual IR.
    fn verify_alias_source(source: &str) -> Result<(), crate::Report> {
        let test = Test::default();
        parse::<Module>(ParserConfig::new(test.context_rc()), "alias.runic", source)
            .into_result()
            .map(|_| ())
    }

    fn verify_cross_module_alias(visibility: &str, target: &str) -> Result<(), crate::Report> {
        let source = format!(
            r#"
builtin.module public @test {{
    builtin.module public @api {{
        builtin.function_alias {visibility} @alias -> ::@test::@implementation::@target;
    }};
    builtin.module public @implementation {{
        {target}
    }};
}};
"#
        );
        verify_alias_source(&source)
    }

    fn local_alias_source(
        visibility: &str,
        target_visibility: &str,
        chained: bool,
    ) -> alloc::string::String {
        let target = if chained { "forward" } else { "body" };
        format!(
            r#"
builtin.module public @test {{
    builtin.function_alias {visibility} @alias -> @{target};
    builtin.function_alias private @forward -> @body;
    builtin.function {target_visibility} extern("C") @body() {{ builtin.ret; }};
}};
"#
        )
    }

    #[test]
    fn non_private_aliases_reject_private_canonical_targets() {
        for visibility in ["public", "internal"] {
            for chained in [false, true] {
                let err = verify_alias_source(&local_alias_source(visibility, "private", chained))
                    .expect_err("a non-private alias must not expose a private callable");
                let message = err.to_string();
                assert!(message.contains("builtin.function_alias 'alias'"), "{message}");
                assert!(
                    message.contains(&format!("{visibility} alias cannot expose private target")),
                    "{message}"
                );
                assert!(message.contains("body"), "{message}");
                assert_eq!(crate::testing::label_ranges(err.as_diagnostic()).len(), 2);
            }
        }
    }

    #[test]
    fn private_local_aliases_accept_private_canonical_targets() {
        for chained in [false, true] {
            // TODO instead of `unwrap` use expect which prints values of current loop params
            verify_alias_source(&local_alias_source("private", "private", chained)).unwrap();
        }
    }

    #[test]
    fn aliases_accept_non_private_canonical_targets() {
        for visibility in ["private", "internal", "public"] {
            for target_visibility in ["internal", "public"] {
                for chained in [false, true] {
                    // TODO instead of `unwrap` use expect which prints values of current loop params
                    verify_alias_source(&local_alias_source(
                        visibility,
                        target_visibility,
                        chained,
                    ))
                    .unwrap();
                }
            }
        }
    }

    #[test]
    fn public_alias_rejects_private_canonical_target_through_another_module() {
        for visibility in ["public", "internal"] {
            let err = verify_cross_module_alias(
                "public",
                &format!(
                    r#"
                builtin.function_alias {visibility} @target -> @body;
                builtin.function private extern("C") @body() {{ builtin.ret; }};
                "#,
                ),
            )
            // TODO reformulate message below, `hide` is ambigous here
            .expect_err("a visible intermediate alias must not hide a private canonical target");
            assert!(err.to_string().contains("cannot expose private target"), "{err}");
        }
    }

    #[test]
    fn alias_rejects_private_cross_module_function() {
        for visibility in ["private", "internal", "public"] {
            let err = verify_cross_module_alias(
                visibility,
                r#"builtin.function private extern("C") @target() { builtin.ret; };"#,
            )
            .expect_err("a private function is inaccessible from the alias's table");
            assert!(err.to_string().contains("private to another symbol table"), "{err}");
            assert_eq!(crate::testing::label_ranges(err.as_diagnostic()).len(), 2);
        }
    }

    #[test]
    fn alias_rejects_private_cross_module_alias_to_public_function() {
        for visibility in ["private", "internal", "public"] {
            let err = verify_cross_module_alias(
                visibility,
                r#"
                builtin.function_alias private @target -> @body;
                builtin.function public extern("C") @body() { builtin.ret; };
                "#,
            )
            .expect_err("a public canonical target does not make a private alias accessible");
            assert!(err.to_string().contains("private to another symbol table"), "{err}");
            assert_eq!(crate::testing::label_ranges(err.as_diagnostic()).len(), 2);
        }
    }

    #[test]
    fn alias_accepts_visible_cross_module_callable_symbols() {
        for visibility in ["public", "internal"] {
            verify_cross_module_alias(
                "public",
                &format!(
                    r#"builtin.function {visibility} extern("C") @target() {{ builtin.ret; }};"#,
                ),
            )
            .unwrap();
            verify_cross_module_alias(
                "public",
                &format!(
                    r#"
                builtin.function_alias {visibility} @target -> @body;
                builtin.function {visibility} extern("C") @body() {{ builtin.ret; }};
                "#,
                ),
            )
            .unwrap();
        }
    }

    #[test]
    fn alias_cycles_fail_verification_after_parsing() {
        for aliases in [
            "builtin.function_alias public @bar -> @bar;",
            "builtin.function_alias public @bar -> @first; builtin.function_alias private @first \
             -> @bar;",
        ] {
            let test = Test::default();
            let source = format!("builtin.module public @test {{ {aliases} }};");
            let err =
                parse::<Module>(ParserConfig::new(test.context_rc()), "alias_cycle.runic", &source)
                    .into_result()
                    .err()
                    .expect("alias cycles must fail verification");
            let message = format!("{err}");
            assert!(
                message.contains("cannot reference itself") || message.contains("cycle"),
                "{message}"
            );
        }
    }
}
