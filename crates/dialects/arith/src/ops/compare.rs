use alloc::rc::Rc;

use runic_core::{
    derive::{EffectOpInterface, operation},
    effects::*,
    traits::*,
    *,
};

use crate::{ArithDialect, CmpPredicate, CmpPredicateAttr};

/// Integer comparison: `arith.cmpi slt %a, %b` is `1` if `%a` is less than `%b` read as signed
/// numbers, and `0` otherwise.
///
/// The operands have one integer type, the result is an `i1`. The predicate is the property
/// `predicate`, written as a keyword before the operands.
#[derive(EffectOpInterface)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands),
    implements(InferTypeOpInterface, InferIntRangeInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Cmpi {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    result: Bool,
    #[attr(hidden)]
    predicate: CmpPredicateAttr,
}

impl InferTypeOpInterface for Cmpi {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i1());
        Ok(())
    }
}

impl InferIntRangeInterface for Cmpi {
    fn infer_result_ranges(
        &self,
        operands: &[IntegerValueRange],
        set_result_range: &mut dyn FnMut(ValueRef, IntegerValueRange),
    ) {
        let [lhs, rhs] = operands else {
            return;
        };
        let (Some(lhs), Some(rhs)) = (lhs.range(), rhs.range()) else {
            return;
        };
        let range = infer_comparison_range(
            self.lhs().ty(),
            lhs,
            rhs,
            *self.get_predicate(),
            self.result().ty(),
        );
        set_result_range(self.result().as_value_ref(), range);
    }
}

fold_one!(Cmpi, Bool, |op| {
    let predicate = *op.get_predicate();
    move |[a, b]| Some(Int::from(predicate.evaluate(a, b)))
});

impl Canonicalizable for Cmpi {
    fn get_canonicalization_patterns(
        rewrites: &mut patterns::RewritePatternSet,
        context: Rc<Context>,
    ) {
        rewrites.push(crate::canonicalization::SimplifyBoolCompare::new(context));
    }
}

impl OpPrinter for Cmpi {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        let op = self.as_operation();
        printer.print_space();
        printer.print_keyword(self.get_predicate().as_str());
        printer.print_space();
        printer.print_value_uses(op.operands().group(0).as_value_range());
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

impl OpParser for Cmpi {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let predicate = crate::attributes::parse_predicate(parser)?;
        let lhs = parser.parse_operand(/*allow_result_number=*/ true)?;
        parser.parse_comma()?;
        let rhs = parser.parse_operand(/*allow_result_number=*/ true)?;
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

        // The operands are uses of values defined earlier, which carry their types, as in the
        // derived format of the other operations.
        let unknown = parser.context().unknown_type();
        let lhs = parser.resolve_operand(lhs, unknown)?;
        let rhs = parser.resolve_operand(rhs, unknown)?;
        state.add_operand(lhs);
        state.add_operand(rhs);
        state.add_attribute(
            "predicate",
            parser.context_rc().create_attribute::<CmpPredicateAttr, _>(predicate),
        );
        state.attrs.extend(attrs);
        state.results.push(parser.context().i1());

        Ok(())
    }
}

/// The range of the `i1` result of comparing `lhs` and `rhs`, of type `operand_ty`, with
/// `predicate`: an exact boolean when the bounds decide it, the boolean domain otherwise.
pub(crate) fn infer_comparison_range(
    operand_ty: Type,
    lhs: &ConstantIntRange,
    rhs: &ConstantIntRange,
    predicate: CmpPredicate,
    result_ty: Type,
) -> IntegerValueRange {
    let top = || IntegerValueRange::new(ConstantIntRange::max_for_type(result_ty));
    if lhs.is_untracked() || rhs.is_untracked() || lhs.width() != rhs.width() {
        return top();
    }

    if lhs.width() != ConstantIntRange::max_for_type(operand_ty).width() {
        // Unrealized casts preserve the source's bounds. Those signed bounds
        // cannot be used to order values interpreted at a different width.
        return top();
    }

    let outcome = match predicate {
        CmpPredicate::Eq => infer_eq(lhs, rhs),
        CmpPredicate::Ne => infer_eq(lhs, rhs).map(|is_equal| !is_equal),
        predicate if predicate.is_signed() => {
            infer_ordering((lhs.smin(), lhs.smax()), (rhs.smin(), rhs.smax()), predicate)
        }
        predicate => infer_ordering((lhs.umin(), lhs.umax()), (rhs.umin(), rhs.umax()), predicate),
    };

    outcome
        .map(|value| IntegerValueRange::exact_bool(result_ty, value))
        .unwrap_or_else(top)
}

fn infer_eq(lhs: &ConstantIntRange, rhs: &ConstantIntRange) -> Option<bool> {
    if lhs.umin() == lhs.umax() && rhs.umin() == rhs.umax() && lhs.umin() == rhs.umin() {
        Some(true)
    } else if lhs.umax() < rhs.umin() || rhs.umax() < lhs.umin() {
        Some(false)
    } else {
        None
    }
}

/// Decides the ordering `predicate` from the bounds of its operands, read in the predicate's
/// signedness by the caller.
fn infer_ordering<T: Ord>(lhs: (T, T), rhs: (T, T), predicate: CmpPredicate) -> Option<bool> {
    use CmpPredicate::*;

    let ((lmin, lmax), (rmin, rmax), inclusive) = match predicate {
        Slt | Ult => (lhs, rhs, false),
        Sle | Ule => (lhs, rhs, true),
        Sgt | Ugt => (rhs, lhs, false),
        Sge | Uge => (rhs, lhs, true),
        Eq | Ne => unreachable!("handled separately"),
    };
    if inclusive {
        if lmax <= rmin {
            Some(true)
        } else if lmin > rmax {
            Some(false)
        } else {
            None
        }
    } else if lmax < rmin {
        Some(true)
    } else if lmin >= rmax {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDERINGS: [CmpPredicate; 8] = [
        CmpPredicate::Slt,
        CmpPredicate::Sle,
        CmpPredicate::Sgt,
        CmpPredicate::Sge,
        CmpPredicate::Ult,
        CmpPredicate::Ule,
        CmpPredicate::Ugt,
        CmpPredicate::Uge,
    ];

    #[test]
    fn an_untracked_or_mismatched_operand_leaves_the_boolean_domain() {
        let context = Context::default();
        let (i1, i32) = (context.i1(), context.i32());
        let top = ConstantIntRange::max_for_type(i32);
        let zero = ConstantIntRange::constant(Int::from(0i32));
        for (lhs, rhs) in [
            (ConstantIntRange::untracked(), zero),
            (ConstantIntRange::max_for_type(context.i64()), zero),
        ] {
            let result = infer_comparison_range(i32, &lhs, &rhs, CmpPredicate::Uge, i1);
            assert_eq!(result, IntegerValueRange::new(ConstantIntRange::max_for_type(i1)));
        }
        let result = infer_comparison_range(i32, &top, &zero, CmpPredicate::Uge, i1);
        assert_eq!(result, IntegerValueRange::exact_bool(i1, true));
        // Read as signed, the full domain is not decided against zero.
        let result = infer_comparison_range(i32, &top, &zero, CmpPredicate::Sge, i1);
        assert_eq!(result, IntegerValueRange::new(ConstantIntRange::max_for_type(i1)));
    }

    #[test]
    fn the_predicate_chooses_signed_or_unsigned_bounds() {
        let context = Context::default();
        let (i1, i8) = (context.i1(), context.i8());
        let minus_one = ConstantIntRange::constant(Int::from(-1i8));
        let one = ConstantIntRange::constant(Int::from(1i8));
        for (predicate, expected) in [(CmpPredicate::Slt, true), (CmpPredicate::Ult, false)] {
            let result = infer_comparison_range(i8, &minus_one, &one, predicate, i1);
            assert_eq!(result, IntegerValueRange::exact_bool(i1, expected), "{predicate}");
        }
    }

    #[test]
    fn signed_comparison_retains_the_interpretation_width_guard() {
        let context = Context::default();
        let raw = ConstantIntRange::constant(Int::from(u32::MAX as u64));
        let zero = ConstantIntRange::constant(Int::from(0u64));
        let result =
            infer_comparison_range(context.i32(), &raw, &zero, CmpPredicate::Slt, context.i1());
        assert_eq!(result, IntegerValueRange::new(ConstantIntRange::max_for_type(context.i1())));
    }

    #[test]
    fn ordering_decisions_match_every_value_in_small_intervals() {
        for lmin in -3..=3 {
            for lmax in lmin..=3 {
                for rmin in -3..=3 {
                    for rmax in rmin..=3 {
                        for predicate in ORDERINGS {
                            let Some(outcome) =
                                infer_ordering((lmin, lmax), (rmin, rmax), predicate)
                            else {
                                continue;
                            };
                            for lhs in lmin..=lmax {
                                for rhs in rmin..=rmax {
                                    // The bounds are plain numbers here; the caller reads them
                                    // in the predicate's signedness.
                                    let actual = match predicate {
                                        CmpPredicate::Slt | CmpPredicate::Ult => lhs < rhs,
                                        CmpPredicate::Sle | CmpPredicate::Ule => lhs <= rhs,
                                        CmpPredicate::Sgt | CmpPredicate::Ugt => lhs > rhs,
                                        CmpPredicate::Sge | CmpPredicate::Uge => lhs >= rhs,
                                        CmpPredicate::Eq | CmpPredicate::Ne => unreachable!(),
                                    };
                                    assert_eq!(outcome, actual, "{lhs} {predicate} {rhs}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
