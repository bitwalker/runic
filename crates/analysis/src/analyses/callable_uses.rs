//! Analysis to track the uses and callers of callables within a given scope.
//!
//! It identifies call sites and flags callables that may be called
//! indirectly or from outside the scope.

use runic_core::{
    CallOpInterface, CanonicalCallableRef, FxHashMap, FxHashSet, Operation, OperationRef, SmallVec,
    SymbolRef, WalkResult, adt::SmallSet, dialects::builtin::FunctionAlias,
};

/// Usage information for a canonical callable within an analysis scope.
///
/// This tracks known call sites and flags that indicate potential
/// unknown callers.
#[derive(Debug, Default)]
pub struct CallableUseInfo {
    /// Call sites within the analysis scope that may transfer control to the callable.
    ///
    /// A call with a statically known dispatch set is listed for each of its possible targets,
    /// even when which target it actually reaches is decided at runtime.
    callers: SmallSet<OperationRef, 4>,
    /// Whether the callable is visible outside the analysis scope.
    has_external_name: bool,
    /// Whether the callable is used in a non-call context.
    has_address_taken: bool,
    /// Whether the callable is referenced by an operation outside the scope.
    is_used_out_of_scope: bool,
}

impl CallableUseInfo {
    /// Returns the identified call sites that may transfer control to the callable.
    pub fn known_callers(&self) -> &[OperationRef] {
        self.callers.as_slice()
    }

    /// Whether the target or any alias exposes a name to callers outside the analysis scope.
    pub fn has_external_name(&self) -> bool {
        self.has_external_name
    }

    /// Whether any name has a non-call, non-alias use (including a function-table entry).
    pub fn is_address_taken(&self) -> bool {
        self.has_address_taken
    }

    /// Returns true if the callable is used outside the analysis scope.
    pub fn has_out_of_scope_uses(&self) -> bool {
        self.is_used_out_of_scope
    }

    /// Returns true if the callable may have callers not listed in `known_callers`.
    pub fn has_unknown_callers(&self) -> bool {
        self.has_external_name || self.has_address_taken || self.is_used_out_of_scope
    }

    fn add_caller(&mut self, caller: OperationRef) {
        self.callers.insert(caller);
    }

    fn collect_symbol_uses(&mut self, target: CanonicalCallableRef, scope: &Operation) {
        let mut worklist = SmallVec::<[SymbolRef; 4]>::from_iter([target.as_symbol_ref()]);
        let mut visited = FxHashSet::default();

        while let Some(symbol) = worklist.pop() {
            if !visited.insert(symbol) {
                continue;
            }

            let symbol = symbol.borrow();
            let outside_scope = !scope.is_ancestor_of(symbol.as_symbol_operation());
            let visibility = symbol.visibility();
            log::trace!(
                target: module_path!(), "found callable symbol '{}' with visibility {visibility}",
                symbol.name()
            );
            let exposed = visibility.is_public()
                || (visibility.is_internal() && (scope.parent().is_some() || outside_scope));
            if exposed {
                log::trace!(
                    target: module_path!(), "marking callable as having unknown callers due to \
                     visibility"
                );
            }
            self.has_external_name |= exposed;
            self.is_used_out_of_scope |= outside_scope;

            let mut call_uses = FxHashSet::default();
            log::trace!(
                target: module_path!(), "looking for non-call uses of callable '{}'",
                symbol.name()
            );
            for symbol_use in symbol.iter_uses() {
                let owner = symbol_use.owner.borrow();
                if let Some(alias) = owner.downcast_ref::<FunctionAlias>()
                    && alias
                        .target()
                        .try_user()
                        .is_some_and(|user| user.borrow().attr == symbol_use.attr)
                {
                    worklist.push(owner.as_symbol_ref().expect("aliases are symbols"));
                    continue;
                }

                if !scope.is_ancestor_of(&owner) {
                    self.is_used_out_of_scope = true;
                }

                if let Some(call) = owner.as_trait::<dyn CallOpInterface>()
                    && call.callable_for_callee().as_symbol_path()
                        == Some(symbol_use.attr.borrow().path())
                {
                    // A call site has at most one callee. Subsequent references to the same symbol
                    // on the same call are treated as non-call uses.
                    if !call_uses.insert(owner.as_operation_ref()) {
                        self.has_address_taken = true;
                    }
                    if scope.is_ancestor_of(&owner) {
                        self.add_caller(owner.as_operation_ref());
                    }
                } else {
                    log::trace!(
                        target: module_path!(), "found symbol use whose user does not implement \
                         CallOpInterface - marking callable as having unknown callers"
                    );
                    self.has_address_taken = true;
                }
            }
        }
    }
}

/// Snapshot of callable usage and call sites within a specific scope.
///
/// This is a snapshot and must be recomputed if the IR is mutated. It conservatively accounts for
/// aliases and visibility across scope boundaries.
#[derive(Debug, Default)]
pub struct CallableUseSnapshot {
    uses: FxHashMap<CanonicalCallableRef, CallableUseInfo>,
}

impl CallableUseSnapshot {
    pub fn new(scope: &Operation) -> Self {
        log::trace!(target: module_path!(), "analyzing callable uses in '{}'", scope.name());
        let mut analysis = Self::default();

        // Walk all ops to ensure all callables are discovered, even if they're not called but
        // only referenced.
        scope.prewalk_all(|op| {
            if let Some(callee) = op.as_symbol_ref().and_then(|s| s.resolve_callable().ok()) {
                analysis.uses.entry(callee.target()).or_default();
            }

            // A reference that does not resolve names no callable, and is skipped.
            let _ = Operation::walk_symbol_refs(op, |reference| {
                if let Ok(symbol_use) = reference
                    && let Ok(callee) = symbol_use.borrow().attr.borrow().resolve_callable()
                {
                    analysis.uses.entry(callee.target()).or_default();
                }
                WalkResult::Continue(())
            });

            if let Some(call) = op.as_trait::<dyn CallOpInterface>()
                && let Some(targets) = call.possible_callees()
            {
                for target in targets {
                    analysis.uses.entry(target).or_default().add_caller(op.as_operation_ref());
                }
            }
        });

        if analysis.uses.is_empty() {
            log::trace!(target: module_path!(), "no callable symbols found in this scope");
        }

        for (target, uses) in analysis.uses.iter_mut() {
            uses.collect_symbol_uses(*target, scope);
        }
        analysis
    }

    pub fn get(&self, target: CanonicalCallableRef) -> Option<&CallableUseInfo> {
        self.uses.get(&target)
    }

    pub fn iter(&self) -> impl Iterator<Item = (CanonicalCallableRef, &CallableUseInfo)> {
        self.uses.iter().map(|(target, uses)| (*target, uses))
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec::Vec};

    use runic_core::{
        Op, SymbolName,
        dialects::{
            builtin::{Call, Module, ModuleBuilder, ModuleRef, attributes::SymbolRefAttr},
            test::CallOneOf,
        },
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
        let source = format!(
            r#"
builtin.module public @test {{
    builtin.function {target_visibility} extern("C") @body() {{ builtin.ret; }};
    builtin.function_alias private @first -> @body;
    builtin.function_alias {visibility} @api -> @first;
    builtin.function public extern("C") @caller() {{
        builtin.call @api() : extern("C") ();
        builtin.call @body() : extern("C") ();
        {extra_use}
        builtin.ret;
    }};
}};
"#
        );
        let module =
            parse::<Module>(ParserConfig::new(test.context_rc()), "uses.runic", &source).unwrap();
        (test, module)
    }

    fn call_callers(module: ModuleRef, function: &str) -> Vec<OperationRef> {
        ModuleBuilder::new(module)
            .get_function(function)
            .unwrap()
            .borrow()
            .entry_block()
            .borrow()
            .body()
            .iter()
            .filter_map(|op| op.downcast_ref::<Call>().map(Op::as_operation_ref))
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
            .try_downcast_attr::<SymbolRefAttr>()
            .unwrap();
        attr.borrow_mut().set_path(path);
    }

    #[test]
    fn alias_use_queries_separate_callers_exposure_and_address_taking() {
        for (visibility, extra, exposed, address_taken) in [
            ("private", "", false, false),
            ("internal", "", false, false),
            ("public", "", true, false),
            ("private", "%a = builtin.function_ref @api : () -> ();", false, true),
        ] {
            let (_test, module) = fixture(visibility, extra);
            let body = ModuleBuilder::new(module).resolve_callable("body").unwrap().target();
            let world = module.borrow().as_operation().parent_op().unwrap();
            let uses = CallableUseSnapshot::new(&world.borrow());
            let body_uses = uses.get(body).unwrap();
            let expected_callers = call_callers(module, "caller");
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
            .find_map(|op| op.downcast_ref::<Call>().map(Op::as_operation_ref))
            .unwrap();
        let path = owner.borrow().downcast_ref::<Call>().unwrap().callee().path().clone();
        let referenced = mb.resolve_callable("api").unwrap().named_symbol() as SymbolRef;
        attach_extra_symbol_attribute(owner, referenced, path);

        let uses = CallableUseSnapshot::new(module.borrow().as_operation());
        let info = uses.get(target).unwrap();
        assert_known_callers(info, &call_callers(module, "caller"));
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
        assert_known_callers(info, &call_callers(module, "caller"));
        assert!(info.is_address_taken());
        assert!(info.has_unknown_callers());
    }

    /// A `test.call_one_of` of an alias and its target is one known caller of the target, and its
    /// symbols, which are not its callee, take the target's address.
    #[test]
    fn indirect_alias_calls_are_known_but_their_symbols_still_escape() {
        let test = Test::default();
        let module = parse::<Module>(
            ParserConfig::new(test.context_rc()),
            "call_one_of_uses.runic",
            r#"
builtin.module public @test {
    builtin.function private extern("C") @body() { builtin.ret; };
    builtin.function_alias private @api -> @body;
    builtin.function public extern("C") @caller(%index: i32) {
        test.call_one_of %index, @api, @body() : extern("C") ();
        builtin.ret;
    };
};
"#,
        )
        .unwrap();
        let target = ModuleBuilder::new(module).resolve_callable("body").unwrap().target();
        let uses = CallableUseSnapshot::new(module.borrow().as_operation());
        let info = uses.get(target).unwrap();
        assert_eq!(info.known_callers().len(), 1, "should deduplicate canonical targets");
        assert!(info.is_address_taken());
        assert!(info.has_unknown_callers());
        assert!(!info.has_external_name());
    }

    /// A `test.call_one_of` with a runtime index may reach either of its callables, and which one
    /// is not known until runtime.
    #[test]
    fn ambiguous_call_one_of_index_is_known_caller_of_every_possible_target() {
        let test = Test::default();
        let module = parse::<Module>(
            ParserConfig::new(test.context_rc()),
            "call_one_of_two_targets.runic",
            r#"
builtin.module public @test {
    builtin.function private extern("C") @left() { builtin.ret; };
    builtin.function private extern("C") @right() { builtin.ret; };
    builtin.function public extern("C") @caller(%index: i32) {
        test.call_one_of %index, @left, @right() : extern("C") ();
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
            .find_map(|op| op.downcast_ref::<CallOneOf>().map(Op::as_operation_ref))
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

    #[test]
    fn internal_alias_visibility_depends_on_analysis_scope() {
        let test = Test::default();
        let world = parse_any(
            ParserConfig::new(test.context_rc()),
            "internal_alias.runic",
            r#"
builtin.world {
    builtin.module public @test {
        builtin.function internal extern("C") @body() { builtin.ret; };
        builtin.function_alias internal @api -> @body;
    };
};
"#,
        )
        .unwrap();
        let module = world
            .borrow()
            .as_symbol_table()
            .unwrap()
            .get(SymbolName::intern("test"))
            .unwrap();
        let module = module.borrow().as_operation_ref().try_downcast_op::<Module>().unwrap();
        let target = ModuleBuilder::new(module).resolve_callable("body").unwrap().target();
        let whole_world = CallableUseSnapshot::new(&world.borrow());
        let info = whole_world.get(target).unwrap();
        // The internal alias stays within the analyzed world, and there are no escaping uses.
        assert!(!info.has_external_name());
        assert!(!info.has_unknown_callers());

        let module_only = CallableUseSnapshot::new(module.borrow().as_operation());
        let info = module_only.get(target).unwrap();
        // The internal alias permits callers elsewhere in the world, outside the analyzed module.
        assert!(info.has_external_name());
        assert!(info.has_unknown_callers());
    }

    /// The analysis scope is `@body` alone, and the aliases that name it lie outside: `@first`
    /// is a use outside the scope, and the public `@api` exposes the target, so it may have
    /// callers that are not known.
    #[test]
    fn alias_names_outside_analysis_scope_still_expose_the_target() {
        let test = Test::default();
        let module = parse::<Module>(
            ParserConfig::new(test.context_rc()),
            "outside_scope.runic",
            r#"
builtin.module public @test {
    builtin.function internal extern("C") @body() { builtin.ret; };
    builtin.function_alias private @first -> @body;
    builtin.function_alias public @api -> @first;
    builtin.function public extern("C") @caller() { builtin.ret; };
};
"#,
        )
        .unwrap();
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

    /// A private function named by a `builtin.function_ref` has its address taken, so it may have
    /// callers that are not known, although nothing outside the module names it; an indirect call
    /// through the reference is not a known caller.
    #[test]
    fn a_function_named_by_a_function_ref_is_address_taken() {
        let test = Test::default();
        let module = parse::<Module>(
            ParserConfig::new(test.context_rc()),
            "function_ref.runic",
            r#"
builtin.module public @test {
    builtin.function private extern("C") @body(i32) -> i32;
    builtin.function private extern("C") @called(i32) -> i32;
    builtin.function public extern("C") @caller(%a: i32) -> i32 {
        %f = builtin.function_ref @body : (i32) -> i32;
        %r = builtin.call_indirect %f(%a) : extern("C") (i32) -> i32;
        %s = builtin.call @called(%r) : extern("C") (i32) -> i32;
        builtin.ret %s : (i32);
    };
};
"#,
        )
        .unwrap();
        let body = ModuleBuilder::new(module).resolve_callable("body").unwrap().target();
        let called = ModuleBuilder::new(module).resolve_callable("called").unwrap().target();
        let uses = CallableUseSnapshot::new(module.borrow().as_operation());

        let info = uses.get(body).unwrap();
        assert!(info.is_address_taken());
        assert!(!info.has_external_name());
        assert!(info.has_unknown_callers());
        assert!(info.known_callers().is_empty());

        let info = uses.get(called).unwrap();
        assert!(!info.is_address_taken());
        assert!(!info.has_unknown_callers());
        assert_eq!(info.known_callers().len(), 1);
    }
}
