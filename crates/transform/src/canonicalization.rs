use alloc::{boxed::Box, format, rc::Rc, string::String, vec::Vec};
use core::fmt;

use runic_ir::{
    Context, EntityMut, Operation, OperationName, Report, Spanned,
    pass::{
        OperationPass, Pass, PassExecutionState, PassOptions, PostPassStatus, write_pass_option,
    },
    patterns::{self, FrozenRewritePatternSet, GreedyRewriteConfig, RewritePatternSet},
};

/// This pass performs various types of canonicalizations over a set of operations by iteratively
/// applying the canonicalization patterns of all loaded dialects until either a fixpoint is reached
/// or the maximum number of iterations/rewrites is exhausted. Canonicalization is best-effort and
/// does not guarantee that the entire IR is in a canonical form after running this pass.
///
/// See the docs for [runic_ir::traits::Canonicalizable] for more details.
///
/// # Options
///
/// The set of patterns applied by the pass can be narrowed from a pass pipeline string with
/// `enable-patterns` (only the listed patterns are applied) and `disable-patterns` (the listed
/// patterns are skipped). Both take a non-empty `;`-separated list of pattern names, e.g.
/// `canonicalizer{enable-patterns="fold-redundant-yields;while-unused-result"}`; spaces around a
/// name are trimmed and empty entries are dropped. A name that matches no registered pattern is
/// an error, and so is a name that appears in both lists. The greedy rewrite driver still folds
/// operations, erases trivially dead ones and, when the rewrite config asks for it, simplifies
/// regions whatever the filter says.
///
/// The pass prints its options in a canonical form: `enable-patterns` before `disable-patterns`
/// whatever order they were written in, each list as its trimmed, non-empty names joined by `;`,
/// and each value quoted and escaped as pipeline text quotes it
/// ([runic_ir::pass::write_pass_option]).
pub struct Canonicalizer {
    config: GreedyRewriteConfig,
    rewrites: Option<Rc<FrozenRewritePatternSet>>,
    require_convergence: bool,
    /// If non-empty, only patterns with these names are applied.
    enabled_patterns: Vec<String>,
    /// Patterns with these names are never applied.
    disabled_patterns: Vec<String>,
}

runic_ir::inventory::submit!(::runic_ir::pass::registry::PassInfo::new::<Canonicalizer>(
    Canonicalizer::NAME,
    Canonicalizer::DESCRIPTION
));

impl Default for Canonicalizer {
    fn default() -> Self {
        let mut config = GreedyRewriteConfig::default();
        config.with_top_down_traversal(true);
        Self {
            config,
            rewrites: None,
            require_convergence: false,
            enabled_patterns: Vec::new(),
            disabled_patterns: Vec::new(),
        }
    }
}

impl Canonicalizer {
    const DESCRIPTION: &str = "Applies the canonicalization patterns of all loaded dialects, and \
                               folds operations, to a fixpoint";
    const NAME: &str = "canonicalizer";

    /// Creates an instance of this pass with the given rewrite config; `require_convergence`
    /// turns a failure to reach a fixpoint into an error.
    pub fn new(config: GreedyRewriteConfig, require_convergence: bool) -> Self {
        Self {
            config,
            require_convergence,
            ..Self::default()
        }
    }

    /// Creates an instance of this pass, configured with default settings.
    pub fn create() -> Box<dyn OperationPass> {
        Box::new(Self::default())
    }

    /// Creates an instance of this pass with the specified config.
    pub fn create_with_config(config: &GreedyRewriteConfig) -> Box<dyn OperationPass> {
        Box::new(Self {
            config: config.clone(),
            ..Self::default()
        })
    }

    /// Returns true if the pattern named `name` passes the `enable-patterns`/`disable-patterns`
    /// filter.
    fn is_pattern_enabled(&self, name: &str) -> bool {
        (self.enabled_patterns.is_empty() || self.enabled_patterns.iter().any(|p| p == name))
            && !self.disabled_patterns.iter().any(|p| p == name)
    }
}

impl Pass for Canonicalizer {
    type Target = Operation;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn argument(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> &'static str {
        Self::DESCRIPTION
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    /// Reads `enable-patterns` and `disable-patterns` (see the [Canonicalizer] docs) from the
    /// options of the pass pipeline.
    fn initialize_options(&mut self, options: &PassOptions) -> Result<(), Report> {
        options.check_keys(Self::NAME, &["enable-patterns", "disable-patterns"])?;
        for (key, value) in options.iter() {
            let Some(value) = value else {
                return Err(Report::msg(format!(
                    "invalid option '{key}' for pass '{}': expected '{key}=<names>'",
                    Self::NAME
                )));
            };
            let names = value
                .split(';')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(String::from)
                .collect::<Vec<_>>();
            let list = match key {
                "enable-patterns" => &mut self.enabled_patterns,
                _ => &mut self.disabled_patterns,
            };
            // An empty list would silently mean "no filter", the opposite of what was asked.
            if names.is_empty() {
                return Err(Report::msg(format!(
                    "option '{key}' of pass '{}' has no pattern names",
                    Self::NAME
                )));
            }
            // The pipeline parser already rejects a repeated key; this covers a direct caller.
            if !list.is_empty() {
                return Err(Report::msg(format!(
                    "option '{key}' of pass '{}' is given more than once",
                    Self::NAME
                )));
            }
            *list = names;
        }
        // A name in both lists is a contradiction rather than a filter.
        if let Some(name) =
            self.enabled_patterns.iter().find(|n| self.disabled_patterns.contains(n))
        {
            return Err(Report::msg(format!(
                "pattern '{name}' is both enabled and disabled in the options of pass '{}'",
                Self::NAME
            )));
        }
        Ok(())
    }

    /// Prints the pass with its pattern filter, the only state settable from pipeline options, in
    /// the canonical form the [Canonicalizer] docs give; the rewrite config given to
    /// [Canonicalizer::new] has no textual form.
    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", Self::NAME)?;
        if self.enabled_patterns.is_empty() && self.disabled_patterns.is_empty() {
            return Ok(());
        }
        f.write_str("{")?;
        let mut sep = "";
        for (key, names) in [
            ("enable-patterns", &self.enabled_patterns),
            ("disable-patterns", &self.disabled_patterns),
        ] {
            if !names.is_empty() {
                f.write_str(sep)?;
                write_pass_option(f, key, Some(&names.join(";")))?;
                sep = " ";
            }
        }
        f.write_str("}")
    }

    fn initialize(&mut self, context: Rc<Context>) -> Result<(), Report> {
        log::trace!(target: Self::NAME, "initializing canonicalizer pass");
        let mut rewrites = RewritePatternSet::new(context.clone());

        for dialect in context.registered_dialects().values() {
            for op in dialect.registered_ops().iter() {
                op.populate_canonicalization_patterns(&mut rewrites, context.clone());
            }
        }

        // A filter naming a pattern that does not exist would silently apply no pattern at all,
        // so reject it.
        let is_registered =
            |name: &String| rewrites.patterns().iter().any(|pattern| pattern.name() == name);
        if let Some(name) = self
            .enabled_patterns
            .iter()
            .chain(&self.disabled_patterns)
            .find(|n| !is_registered(n))
        {
            return Err(Report::msg(format!(
                "unknown canonicalization pattern '{name}' in the options of pass '{}'",
                Self::NAME
            )));
        }
        rewrites.retain(|pattern| self.is_pattern_enabled(pattern.name()));

        self.rewrites = Some(Rc::new(FrozenRewritePatternSet::new(rewrites)));

        Ok(())
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let Some(rewrites) = self.rewrites.as_ref() else {
            log::debug!(target: Self::NAME, "skipping canonicalization as there are no rewrite patterns to apply");
            state.set_post_pass_status(PostPassStatus::Unchanged);
            return Ok(());
        };
        let op = {
            let ptr = op.as_operation_ref();
            drop(op);
            log::debug!(target: Self::NAME, "applying canonicalization to {}", ptr.borrow());
            log::debug!(target: Self::NAME, "  require_convergence = {}", self.require_convergence);
            ptr
        };
        let converged =
            patterns::apply_patterns_and_fold_greedily(op, rewrites.clone(), self.config.clone());
        if self.require_convergence && converged.is_err() {
            log::debug!(target: Self::NAME, "canonicalization could not converge");
            let span = op.borrow().span();
            return Err(runic_ir::diagnostics::report!(
                message: "canonicalization failed",
                labels: [
                    primary(
                        span,
                        format!(
                            "canonicalization did not converge{}",
                            self.config
                                .max_iterations()
                                .map(|max| format!(" after {max} iterations"))
                                .unwrap_or_default()
                        )
                    ),
                ],
            ));
        }

        let op = op.borrow();
        let changed = match converged {
            Ok(changed) => {
                log::debug!(target: Self::NAME, "canonicalization converged for '{}', changed={changed}", op.name());
                changed
            }
            Err(changed) => {
                log::warn!(
                    target: Self::NAME,
                    "canonicalization failed to converge for '{}', changed={changed}",
                    op.name()
                );
                changed
            }
        };
        let ir_changed = changed.into();
        state.set_post_pass_status(ir_changed);

        Ok(())
    }
}

/// Tests of the option parsing and the textual pipeline form of [Canonicalizer].
#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    /// Renders a pass through [Pass::print_as_textual_pipeline].
    struct Textual<'a>(&'a Canonicalizer);

    impl fmt::Display for Textual<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            Pass::print_as_textual_pipeline(self.0, f)
        }
    }

    /// The pattern lists are read from the options and print back quoted.
    #[test]
    fn options_print_as_textual_pipeline() {
        let mut pass = Canonicalizer::default();
        assert_eq!(Textual(&pass).to_string(), "canonicalizer");

        let options = PassOptions::parse("enable-patterns=a;b disable-patterns=c").unwrap();
        Pass::initialize_options(&mut pass, &options).unwrap();
        assert_eq!(
            Textual(&pass).to_string(),
            "canonicalizer{enable-patterns=\"a;b\" disable-patterns=\"c\"}"
        );
        assert_eq!(pass.enabled_patterns, ["a", "b"]);
        assert_eq!(pass.disabled_patterns, ["c"]);
    }

    /// Unknown keys, missing or empty lists, a name in both lists and a repeated key are errors.
    #[test]
    fn invalid_options_are_rejected() {
        let options =
            |pairs: &[(&str, Option<&str>)]| pairs.iter().copied().collect::<PassOptions>();
        for options in [
            options(&[("bogus", Some("1"))]),
            options(&[("enable-patterns", None)]),
            options(&[("enable-patterns", Some(""))]),
            options(&[("enable-patterns", Some(";"))]),
            options(&[("enable-patterns", Some("a")), ("disable-patterns", Some("a"))]),
            options(&[("enable-patterns", Some("a")), ("enable-patterns", Some("b"))]),
        ] {
            let mut pass = Canonicalizer::default();
            assert!(
                Pass::initialize_options(&mut pass, &options).is_err(),
                "expected '{options}' to be rejected"
            );
        }
    }

    /// The canonicalizer keeps a `cf.assert` of a non-constant and of the constant `0`, and erases
    /// one of the constant `1`.
    #[test]
    fn canonicalizer_erases_only_asserts_of_true() -> Result<(), Report> {
        use alloc::{boxed::Box, vec::Vec};

        use runic_ir::{
            dialects::builtin::Function,
            parse::{self, ParserConfig},
            pass::{Nesting, PassManager},
        };

        let context = Rc::new(Context::default());
        let function = parse::parse::<Function>(
            ParserConfig::new(context.clone()),
            "asserts.runic",
            r#"builtin.function public extern("C") @asserts(%c: i1) {
                %true = arith.constant 1 : i1;
                %false = arith.constant 0 : i1;
                cf.assert %c, "unknown";
                cf.assert %true, "true";
                cf.assert %false, "false";
                builtin.ret;
            };"#,
        )
        .into_result()?;
        let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
        passes.add_pass(Box::<Canonicalizer>::default()).unwrap();
        passes.run(function.as_operation_ref())?;
        function.as_operation_ref().borrow().recursively_verify()?;

        let function = function.borrow();
        let messages = function
            .body()
            .entry()
            .body()
            .iter()
            .filter_map(|op| {
                op.downcast_ref::<runic_dialect_cf::Assert>()
                    .map(|assert| assert.message().as_str().to_string())
            })
            .collect::<Vec<_>>();
        assert_eq!(messages, ["unknown", "false"]);
        Ok(())
    }

    /// `@caller` after the canonicalizer ran on it, in a module that also holds `declaration` of
    /// `@callee`; the caller calls `@callee` through a reference with `signature`.
    fn canonicalize_indirect_call(declaration: &str, signature: &str) -> Result<String, Report> {
        use runic_ir::{
            SymbolTable,
            dialects::builtin::{Function, Module},
            parse::{self, ParserConfig},
            pass::{Nesting, PassManager},
        };

        let context = Rc::new(Context::default());
        let source = format!(
            "builtin.module public @m {{
    {declaration}
    builtin.function public extern(\"C\") @caller(%a: i32) -> i32 {{
        %f = builtin.function_ref @callee : (i32) -> i32;
        %r = builtin.call_indirect %f(%a) : {signature};
        builtin.ret %r : (i32);
    }};
}};"
        );
        let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "indirect", source)
            .into_result()?;
        let caller = module.borrow().symbol_manager().lookup_op("caller").unwrap();
        let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
        passes.add_pass(Box::<Canonicalizer>::default()).unwrap();
        passes.enable_verifier(true);
        passes.run(caller)?;
        Ok(caller.borrow().to_string())
    }

    /// The canonicalizer turns an indirect call of a reference into a direct call when the
    /// signatures agree, and the reference, now unused, is erased as a dead operation; a callee
    /// whose calling convention differs keeps the indirect call and its reference.
    #[test]
    fn canonicalizer_turns_an_indirect_call_of_a_function_ref_into_a_call() -> Result<(), Report> {
        let text = canonicalize_indirect_call(
            r#"builtin.function public extern("C") @callee(i32) -> i32;"#,
            r#"extern("C") (i32) -> i32"#,
        )?;
        assert!(!text.contains("builtin.call_indirect"), "{text}");
        assert!(!text.contains("builtin.function_ref"), "{text}");
        assert!(
            text.contains(r#"builtin.call @callee(%0) : extern("C") (i32) -> i32;"#),
            "{text}"
        );

        let text = canonicalize_indirect_call(
            r#"builtin.function public extern("fast") @callee(i32) -> i32;"#,
            r#"extern("C") (i32) -> i32"#,
        )?;
        assert!(
            text.contains(r#"builtin.call_indirect %1(%0) : extern("C") (i32) -> i32;"#),
            "{text}"
        );
        assert!(text.contains("builtin.function_ref @callee : (i32) -> i32;"), "{text}");
        Ok(())
    }
}
