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
                    && alias.target().user().borrow().attr == symbol_use.attr
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

            let _ = Operation::walk_symbol_refs(op, |symbol_use| {
                let symbol_use = symbol_use.borrow();
                if let Ok(callee) = symbol_use.attr.borrow().resolve_callable() {
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
    use runic_core::{
        Op, SymbolName,
        dialects::builtin::{Module, ModuleBuilder},
        parse::{ParserConfig, parse_any},
        testing::Test,
    };

    use super::*;

    #[test]
    fn internal_alias_visibility_depends_on_analysis_scope() {
        let test = Test::default();
        let world = parse_any(
            ParserConfig::new(test.context_rc()),
            "internal_alias.hir",
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
}
