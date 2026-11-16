use alloc::rc::Rc;

use runic_ir::{
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind, RewritePattern},
    *,
};

use crate::*;

/// A canonicalization pattern for [IndexSwitch] that folds away the operation if it has a constant
/// selector value.
pub struct FoldConstantIndexSwitch {
    info: PatternInfo,
}

impl FoldConstantIndexSwitch {
    pub fn new(context: Rc<Context>) -> Self {
        let scf_dialect = context.get_or_register_dialect::<ScfDialect>();
        let switch_op = scf_dialect
            .registered_name::<IndexSwitch>()
            .expect("scf.index_switch is not registered");
        Self {
            info: PatternInfo::new(
                context,
                "fold-constant-index-switch",
                PatternKind::Operation(switch_op),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl FoldConstantIndexSwitch {
    /// The selector's constant value, read as a 32-bit pattern (see `constant_selector`).
    fn match_constant_selector(&self, op: OperationRef) -> Option<i32> {
        use runic_ir::matchers::{self, Matcher};

        let op = op.borrow();
        if let Some(op) = op.downcast_ref::<IndexSwitch>() {
            let selector = op.selector().as_value_ref();
            selector
                .borrow()
                .get_defining_op()
                .and_then(|defined_by| matchers::constant().matches(&*defined_by.borrow()))
                .and_then(|attr| crate::ops::constant_selector(&attr))
        } else {
            None
        }
    }
}

impl Pattern for FoldConstantIndexSwitch {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for FoldConstantIndexSwitch {
    fn match_and_rewrite(
        &self,
        op: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let Some(selector) = self.match_constant_selector(op) else {
            return Ok(false);
        };
        let case_region = {
            let switch_operation = op.borrow();
            let switch_op = switch_operation.downcast_ref::<IndexSwitch>().unwrap();
            let case_index = switch_op.get_case_index_for_selector(selector);
            case_index
                .map(|idx| switch_op.get_case_region(idx))
                .unwrap_or_else(|| switch_op.default_region().as_region_ref())
        };

        let source = case_region.borrow().entry_block_ref().expect("expected non-empty region");
        let terminator =
            source.borrow().terminator().expect("expected region to have a terminator");
        // Only a yield forwards the switch results and can be erased by this fold.
        if !terminator.borrow().is::<Yield>() {
            return Ok(false);
        }
        let results = terminator
            .borrow()
            .operands()
            .iter()
            .copied()
            .map(|o| Some(o.borrow().as_value_ref()))
            .collect::<SmallVec<[Option<ValueRef>; 2]>>();

        let dest = op.parent().unwrap();
        rewriter.inline_block_before(source, dest, Some(op), &[]);
        rewriter.erase_op(terminator);
        // Replace the operation with a potentially empty list of results.
        //
        // The fold mechanism doesn't support the case where the result list is empty
        rewriter.replace_op_with_values(op, &results);

        Ok(true)
    }
}

#[cfg(test)]
mod terminal_region_tests {
    use runic_dialect_arith::ArithOpBuilder;
    use runic_dialect_ub::UndefinedBehaviorOpBuilder;
    use runic_ir::{dialects::builtin::BuiltinOpBuilder, patterns::PatternRewriter, testing::Test};

    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ResultShape {
        Zero,
        Unused,
        One,
        Two,
    }

    #[test]
    fn terminal_regions_preserve_identity_and_result_order() -> Result<(), Report> {
        check_terminal_regions(&[1, 2], true)
    }

    #[test]
    fn yielding_regions_and_unselected_terminals_keep_folding() -> Result<(), Report> {
        check_terminal_regions(&[0, 1], false)
    }

    fn check_terminal_regions(
        terminal_counts: &[usize],
        select_terminal: bool,
    ) -> Result<(), Report> {
        for returns in [false, true] {
            for terminal_first in [false, true] {
                for terminal_count in terminal_counts.iter().copied() {
                    for shape in
                        [ResultShape::Zero, ResultShape::Unused, ResultShape::One, ResultShape::Two]
                    {
                        let mut test = Test::new("terminal_regions", |cx| {
                            let outputs = if shape == ResultShape::Two {
                                alloc::vec![cx.i32(), cx.i64()]
                            } else {
                                alloc::vec![cx.i32()]
                            };
                            ([cx.i32(), cx.i32(), cx.i64(), cx.i32()], outputs)
                        });
                        let (i32_ty, i64_ty) = (test.context().i32(), test.context().i64());
                        let span = SourceSpan::default();
                        let mut builder = test.function_builder();
                        let entry = builder.entry_block();
                        let args = entry
                            .borrow()
                            .arguments()
                            .iter()
                            .map(|arg| arg.upcast())
                            .collect::<SmallVec<[ValueRef; 4]>>();

                        let values = [args[1], args[2], args[3]];
                        let types = if shape == ResultShape::Zero {
                            SmallVec::<[Type; 3]>::new()
                        } else {
                            [i32_ty, i64_ty, i32_ty].into_iter().collect()
                        };
                        let selector = builder.i32(
                            if terminal_first == select_terminal {
                                1
                            } else {
                                0
                            },
                            span,
                        );
                        let structured = builder.index_switch(selector, [1], &types, span)?;
                        let regions = [
                            structured.borrow().get_case_region(0),
                            structured.borrow().default_region().as_region_ref(),
                        ];
                        let mut terminals = SmallVec::<[(usize, OperationRef); 2]>::new();
                        let mut yields = SmallVec::<[OperationRef; 2]>::new();
                        for (index, region) in regions.into_iter().enumerate() {
                            let block = builder.create_block_in_region(region);
                            builder.switch_to_block(block);
                            if terminal_count == 2
                                || (terminal_count == 1 && (index == 0) == terminal_first)
                            {
                                let terminal = if returns {
                                    builder
                                        .ret(
                                            if shape == ResultShape::Two {
                                                alloc::vec![args[3], args[2]]
                                            } else {
                                                alloc::vec![args[3]]
                                            },
                                            span,
                                        )?
                                        .as_operation_ref()
                                } else {
                                    builder.unreachable(span).as_operation_ref()
                                };
                                terminals.push((index, terminal));
                            } else {
                                let yielded = if types.is_empty() {
                                    &values[..0]
                                } else {
                                    &values[..]
                                };
                                yields.push(
                                    builder
                                        .r#yield(yielded.iter().copied(), span)?
                                        .as_operation_ref(),
                                );
                            }
                        }
                        builder.switch_to_block(entry);
                        let result = if matches!(shape, ResultShape::One | ResultShape::Two) {
                            structured.borrow().results()[2].upcast()
                        } else {
                            args[1]
                        };
                        let returned = if shape == ResultShape::Two {
                            alloc::vec![result, structured.borrow().results()[1].upcast()]
                        } else {
                            alloc::vec![result]
                        };
                        let user = builder.ret(returned, span)?.as_operation_ref();
                        let op = structured.as_operation_ref();
                        let function = test.function().as_operation_ref();
                        function.borrow().recursively_verify()?;
                        let before = alloc::format!("{}", function.borrow());
                        let mut rewriter = PatternRewriter::new(test.context_rc());
                        rewriter.set_insertion_point_before(op);
                        let changed = FoldConstantIndexSwitch::new(test.context_rc())
                            .match_and_rewrite(op, &mut *rewriter)?;
                        assert_eq!(
                            changed,
                            terminal_count == 0 || (terminal_count == 1 && !select_terminal)
                        );
                        if !changed {
                            assert_eq!(before, alloc::format!("{}", function.borrow()));
                        }
                        if changed && matches!(shape, ResultShape::One | ResultShape::Two) {
                            assert_eq!(
                                user.borrow().operands()[0].borrow().as_value_ref(),
                                args[3]
                            );
                            if shape == ResultShape::Two {
                                assert_eq!(
                                    user.borrow().operands()[1].borrow().as_value_ref(),
                                    args[2]
                                );
                            }
                        }
                        let returned_values = [args[3], args[2]];
                        for (region_index, terminal) in terminals.into_iter().filter(|_| !changed) {
                            assert_eq!(
                                terminal.parent(),
                                regions[region_index].borrow().entry_block_ref()
                            );
                            let operands = terminal
                                .borrow()
                                .operands()
                                .iter()
                                .map(|o| o.borrow().as_value_ref())
                                .collect::<SmallVec<[ValueRef; 2]>>();
                            assert_eq!(
                                operands.as_slice(),
                                if returns {
                                    if shape == ResultShape::Two {
                                        &returned_values[..]
                                    } else {
                                        &args[3..4]
                                    }
                                } else {
                                    &args[0..0]
                                }
                            );
                        }
                        function.borrow().recursively_verify()?;
                        test.apply_boxed_pass(
                            alloc::boxed::Box::new(runic_transform::Canonicalizer::new(
                                runic_ir::patterns::GreedyRewriteConfig::default(),
                                true,
                            )),
                            true,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod selector_tests {
    use alloc::{boxed::Box, format, rc::Rc};

    use runic_ir::{
        patterns::{self, FrozenRewritePatternSet, GreedyRewriteConfig, RewritePatternSet},
        testing::parse_function_fixpoint,
    };

    use super::*;

    /// The function returned by `source` after [FoldConstantIndexSwitch] ran to a fixpoint.
    fn fold(name: &str, source: &str) -> alloc::string::String {
        let context = Rc::new(Context::default());
        let (function, _) = parse_function_fixpoint(&context, name, source)
            .unwrap_or_else(|err| panic!("'{name}' must parse before the pattern runs: {err}"));
        let pattern_set = RewritePatternSet::from_iter(
            context.clone(),
            [Box::new(FoldConstantIndexSwitch::new(context.clone())) as Box<dyn RewritePattern>],
        );
        let changed = patterns::apply_patterns_and_fold_greedily(
            function.as_operation_ref(),
            Rc::new(FrozenRewritePatternSet::new(pattern_set)),
            GreedyRewriteConfig::default(),
        )
        .expect("expected the canonicalizer to converge");
        assert!(changed, "a constant selector must fold the switch");
        format!("{}", function.as_operation_ref().borrow())
    }

    const PROGRAM: &str = "\
builtin.function public extern(\"C\") @f(%a: i32, %b: i32) -> i32 {
    %s = arith.constant SELECTOR : i32;
    %r = scf.index_switch %s
    case -1 {
        scf.yield %a : (i32);
    }
    default {
        scf.yield %b : (i32);
    } : (i32);
    builtin.ret %r : (i32);
};";

    /// The selector constant `0xffffffff` is the 32-bit pattern of `-1`, so the switch folds to
    /// `case -1`; any other selector folds to the default region.
    #[test]
    fn a_constant_selector_is_compared_as_a_32_bit_pattern() {
        for selector in ["0xffffffff", "-1", "4294967295"] {
            let folded = fold("selector_pattern.runic", &PROGRAM.replace("SELECTOR", selector));
            assert!(!folded.contains("scf.index_switch"), "{folded}");
            assert!(folded.contains("builtin.ret %0 : (i32);"), "selector {selector}:\n{folded}");
        }
        let folded = fold("selector_default.runic", &PROGRAM.replace("SELECTOR", "7"));
        assert!(!folded.contains("scf.index_switch"), "{folded}");
        assert!(folded.contains("builtin.ret %1 : (i32);"), "{folded}");
    }
}
