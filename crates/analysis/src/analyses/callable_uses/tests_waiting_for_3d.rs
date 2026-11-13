// Tests of `CallableUseSnapshot` that wait for stage 3d, moved here unchanged from the test module
// of `analyses/callable_uses.rs`. This file is not declared as a module and is not compiled. Each
// needs:
//
// - `alias_use_queries_separate_callers_exposure_and_address_taking`: `builtin.call` (written with
//   `hir.exec`), and an operation that takes a function's address (written with
//   `hir.procedure_root`, which no longer exists).
// - `extra_symbol_attribute_on_call_op_is_escaping_use`: `builtin.call`.
// - `extra_symbol_attribute_on_alias_op_is_escaping_use`: `builtin.call` (the fixture's callers).
// - `alias_names_outside_analysis_scope_still_expose_the_target`: `builtin.call` (the fixture's
//   callers).
// - `indirect_alias_calls_are_known_but_table_uses_still_escape`: `builtin.call_indirect` with a
//   statically known dispatch set (written with `builtin.function_table` and
//   `hir.exec_indirect ... tag 1`, which no longer exist; the fixture must be restated).
// - `ambiguous_table_index_is_known_caller_of_every_dispatchable_target`: as the previous test.
//
// The helpers `fixture`, `exec_callers`, `assert_known_callers` and `attach_extra_symbol_attribute`
// are used only by these tests; `fixture` and `exec_callers` use the parked `hir` dialect.

    use alloc::{format, vec::Vec};

    use runic_core::{
        Op, SymbolName,
        diagnostics::Uri,
        dialects::builtin::{Module, ModuleBuilder, ModuleRef},
        parse::{ParserConfig, parse, parse_any},
        testing::Test,
    };

    use super::*;

    fn fixture(visibility: &str, extra_use: &str) -> (Test, ModuleRef) {
        let target_visibility = if visibility == "private" {
            "private"
        } else {
            "internal"
        };
        let test = Test::default();
        test.context().get_or_register_dialect::<runic_dialect_hir::HirDialect>();
        let source = format!(
            r#"
builtin.module public @test {{
    builtin.function {target_visibility} extern("C") @body() {{ builtin.ret; }};
    builtin.function_alias private @first -> @body;
    builtin.function_alias {visibility} @api -> @first;
    builtin.function public extern("C") @caller() {{
        hir.exec @api() : extern("C") () -> ();
        hir.exec @body() : extern("C") () -> ();
        {extra_use}
        builtin.ret;
    }};
}};
"#
        );
        let module =
            parse::<Module>(ParserConfig::new(test.context_rc()), Uri::new("uses.hir"), &source)
                .unwrap();
        (test, module)
    }

    fn exec_callers(module: ModuleRef, function: &str) -> Vec<OperationRef> {
        ModuleBuilder::new(module)
            .get_function(function)
            .unwrap()
            .borrow()
            .entry_block()
            .borrow()
            .body()
            .iter()
            .filter_map(|op| op.downcast_ref::<runic_dialect_hir::Exec>().map(Op::as_operation_ref))
            .collect()
    }

    fn assert_known_callers(info: &CallableUseInfo, expected: &[OperationRef]) {
        assert_eq!(info.known_callers().len(), expected.len());
        for caller in expected {
            assert!(
                info.known_callers().contains(caller),
                "expected {caller} to be a known caller"
            );
        }
    }

    fn attach_extra_symbol_attribute(
        mut owner: OperationRef,
        referenced: SymbolRef,
        path: runic_core::SymbolPath,
    ) {
        owner.borrow_mut().set_symbol_attribute("extra_address", referenced);

        let mut attr = owner
            .borrow()
            .get_attribute("extra_address")
            .unwrap()
            .try_downcast_attr::<runic_core::dialects::builtin::attributes::SymbolRefAttr>()
            .unwrap();
        attr.borrow_mut().set_path(path);
    }

    #[test]
    fn alias_use_queries_separate_callers_exposure_and_address_taking() {
        for (visibility, extra, exposed, address_taken) in [
            ("private", "", false, false),
            ("internal", "", false, false),
            ("public", "", true, false),
            ("private", "%a, %b, %c, %d = hir.procedure_root @api;", false, true),
        ] {
            let (_test, module) = fixture(visibility, extra);
            let body = ModuleBuilder::new(module).resolve_callable("body").unwrap().target();
            let world = module.borrow().as_operation().parent_op().unwrap();
            let uses = CallableUseSnapshot::new(&world.borrow());
            let body_uses = uses.get(body).unwrap();
            let expected_callers = exec_callers(module, "caller");
            assert_eq!(expected_callers.len(), 2);
            assert_known_callers(body_uses, &expected_callers);
            assert_eq!(body_uses.has_external_name(), exposed);
            assert_eq!(body_uses.is_address_taken(), address_taken);
            assert!(!body_uses.has_out_of_scope_uses());
            assert_eq!(body_uses.has_unknown_callers(), exposed || address_taken);

            // The analysis does not rewrite the alias chain
            let alias = ModuleBuilder::new(module).get_function_alias("api").unwrap();
            assert_eq!(alias.borrow().target().path().name(), SymbolName::intern("first"));
            let first = ModuleBuilder::new(module).get_function_alias("first").unwrap();
            assert_eq!(first.borrow().target().path().name(), SymbolName::intern("body"));
        }
    }

    /// Verifies that an extra symbol attribute on a call op is treated as an escaping use
    /// (address taken) and does not change the direct-caller set.
    #[test]
    fn extra_symbol_attribute_on_call_op_is_escaping_use() {
        let (_test, module) = fixture("private", "");
        let mb = ModuleBuilder::new(module);
        let target = mb.resolve_callable("body").unwrap().target();
        let owner = mb
            .get_function("caller")
            .unwrap()
            .borrow()
            .entry_block()
            .borrow()
            .body()
            .iter()
            .find_map(|op| op.downcast_ref::<runic_dialect_hir::Exec>().map(Op::as_operation_ref))
            .unwrap();
        let path = owner
            .borrow()
            .downcast_ref::<runic_dialect_hir::Exec>()
            .unwrap()
            .callee()
            .path()
            .clone();
        let referenced = mb.resolve_callable("api").unwrap().named_symbol() as SymbolRef;
        attach_extra_symbol_attribute(owner, referenced, path);

        let uses = CallableUseSnapshot::new(module.borrow().as_operation());
        let info = uses.get(target).unwrap();
        assert_known_callers(info, &exec_callers(module, "caller"));
        assert!(info.is_address_taken());
        assert!(info.has_unknown_callers());
    }

    /// Verifies that an extra symbol attribute on an alias op is treated as an escaping use
    /// (address taken) and does not change the direct-caller set.
    #[test]
    fn extra_symbol_attribute_on_alias_op_is_escaping_use() {
        let (_test, module) = fixture("private", "");
        let mb = ModuleBuilder::new(module);
        let target = mb.resolve_callable("body").unwrap().target();
        let alias = mb.get_function_alias("api").unwrap();
        let owner = alias.borrow().as_operation_ref();
        let path = alias.borrow().target().path().clone();
        let referenced = mb.resolve_callable("first").unwrap().named_symbol() as SymbolRef;
        attach_extra_symbol_attribute(owner, referenced, path);

        let uses = CallableUseSnapshot::new(module.borrow().as_operation());
        let info = uses.get(target).unwrap();
        assert_known_callers(info, &exec_callers(module, "caller"));
        assert!(info.is_address_taken());
        assert!(info.has_unknown_callers());
    }

    #[test]
    fn alias_names_outside_analysis_scope_still_expose_the_target() {
        let (_test, module) = fixture("public", "");
        let target = ModuleBuilder::new(module).resolve_callable("body").unwrap().target();
        let body = target.as_operation_ref();
        let uses = CallableUseSnapshot::new(&body.borrow());
        let info = uses.get(target).unwrap();
        assert!(
            info.known_callers().is_empty(),
            "callers outside the scope are not known local callers"
        );
        assert!(
            info.has_external_name(),
            "the public alias is outside the walked operation tree"
        );
        assert!(info.has_out_of_scope_uses());
        assert!(info.has_unknown_callers());
    }

    #[test]
    fn indirect_alias_calls_are_known_but_table_uses_still_escape() {
        let test = Test::default();
        test.context().get_or_register_dialect::<runic_dialect_hir::HirDialect>();
        let module = parse::<Module>(
            ParserConfig::new(test.context_rc()),
            Uri::new("table_uses.hir"),
            r#"
builtin.module public @test {
    builtin.function private extern("C") @body() { builtin.ret; };
    builtin.function_alias private @api -> @body;
    builtin.function_table private @table : 2 {
        builtin.function_table_entry 0 @api tag 1;
        builtin.function_table_entry 1 @body tag 1;
    };
    builtin.function public extern("C") @caller(%index: u32) {
        hir.exec_indirect @table[%index]() : extern("C") () -> () tag 1;
        builtin.ret;
    };
};
"#,
        )
        .unwrap();
        let target = ModuleBuilder::new(module).resolve_callable("body").unwrap().target();
        let uses = CallableUseSnapshot::new(module.borrow().as_operation());
        let info = uses.get(target).unwrap();
        assert_eq!(info.known_callers().len(), 1, "should deduplicate canonical table targets");
        assert!(info.is_address_taken());
        assert!(info.has_unknown_callers());
        assert!(!info.has_external_name());
    }

    /// A call through a table with a runtime index may reach any dispatchable entry, and which
    /// one is not known until runtime.
    #[test]
    fn ambiguous_table_index_is_known_caller_of_every_dispatchable_target() {
        let test = Test::default();
        test.context().get_or_register_dialect::<runic_dialect_hir::HirDialect>();
        let module = parse::<Module>(
            ParserConfig::new(test.context_rc()),
            Uri::new("table_two_targets.hir"),
            r#"
builtin.module public @test {
    builtin.function private extern("C") @left() { builtin.ret; };
    builtin.function private extern("C") @right() { builtin.ret; };
    builtin.function_table private @table : 2 {
        builtin.function_table_entry 0 @left tag 1;
        builtin.function_table_entry 1 @right tag 1;
    };
    builtin.function public extern("C") @caller(%index: u32) {
        hir.exec_indirect @table[%index]() : extern("C") () -> () tag 1;
        builtin.ret;
    };
};
"#,
        )
        .unwrap();
        let mb = ModuleBuilder::new(module);
        let call = mb
            .get_function("caller")
            .unwrap()
            .borrow()
            .entry_block()
            .borrow()
            .body()
            .iter()
            .find_map(|op| {
                op.downcast_ref::<runic_dialect_hir::ExecIndirect>().map(Op::as_operation_ref)
            })
            .unwrap();

        let uses = CallableUseSnapshot::new(module.borrow().as_operation());
        for function in ["left", "right"] {
            let target = mb.resolve_callable(function).unwrap().target();
            let info = uses.get(target).unwrap();
            assert_known_callers(info, &[call]);
            assert!(info.is_address_taken());
            assert!(!info.has_external_name());
            assert!(!info.has_out_of_scope_uses());
            assert!(info.has_unknown_callers());
        }
    }
