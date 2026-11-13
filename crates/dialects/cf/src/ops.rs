use alloc::{rc::Rc, vec::Vec};

use runic_core::{
    attributes::IntegerLikeAttr,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::{I32Attr, StringAttr},
    effects::*,
    matchers::Matcher,
    patterns::RewritePatternSet,
    traits::*,
    *,
};
use runic_dialect_ub::PoisonAttr;

use crate::ControlFlowDialect;

/// An unstructured control flow primitive representing an unconditional branch to `target`
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ControlFlowDialect,
    traits(Terminator),
    implements(
        BranchOpInterface,
        MemoryEffectOpInterface,
        OpPrinter,
    )
)]
pub struct Br {
    #[successor]
    target: Successor,
}

impl Canonicalizable for Br {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites
            .push(crate::canonicalization::SimplifyBrToBlockWithSinglePred::new(context.clone()));
        rewrites.push(crate::canonicalization::SimplifyPassthroughBr::new(context.clone()));
        rewrites.push(crate::canonicalization::SimplifyBrToReturn::new(context));
    }
}

impl BranchOpInterface for Br {
    #[inline]
    fn get_successor_for_operands(
        &self,
        _operands: &[Option<AttributeRef>],
    ) -> Option<SuccessorInfo> {
        Some(self.successors()[0])
    }
}

/// The value of a constant `i1` condition: any integer-like attribute (an `IntAttr` or a
/// `BoolAttr`) read with [IntegerLikeAttr::as_int], or `None` for any other attribute (poison, say).
fn constant_condition(attr: &AttributeRef) -> Option<bool> {
    let attr = attr.borrow();
    let value = attr.as_attr().as_trait::<dyn IntegerLikeAttr>()?.as_int();
    Some(!value.is_zero())
}

/// An unstructured control flow primitive representing a conditional branch to either `then_dest`
/// or `else_dest` depending on the value of `condition`, an `i1`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ControlFlowDialect,
    traits(Terminator),
    implements(
        BranchOpInterface,
        MemoryEffectOpInterface,
        OpPrinter,
    )
)]
pub struct CondBr {
    #[operand]
    condition: Bool,
    #[successor]
    then_dest: Successor,
    #[successor]
    else_dest: Successor,
}

impl Canonicalizable for CondBr {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        let name = context
            .get_or_register_dialect::<ControlFlowDialect>()
            .expect_registered_name::<Self>();
        rewrites.push(crate::canonicalization::SimplifyPassthroughCondBr::new(context.clone()));
        rewrites.push(crate::canonicalization::SplitCriticalEdges::for_op(context.clone(), name));
        rewrites.push(crate::canonicalization::RemoveUnusedSinglePredBlockArgs::new(context));
    }
}

impl BranchOpInterface for CondBr {
    fn get_successor_for_operands(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> Option<SuccessorInfo> {
        let cond = constant_condition(operands[0].as_ref()?)?;
        Some(if cond {
            self.successors()[0]
        } else {
            self.successors()[1]
        })
    }
}

/// An unstructured control flow primitive that represents a multi-way branch to one of multiple
/// branch targets, depending on the value of `selector`.
///
/// If a specific selector value is matched by `cases`, the branch target corresponding to that
/// case is the one to which control is transferred. If no matching case is found for the selector,
/// then the `fallback` target is used instead.
///
/// A `fallback` successor must always be provided.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ControlFlowDialect,
    traits(Terminator),
    implements(
        BranchOpInterface,
        MemoryEffectOpInterface,
        OpPrinter,
    )
)]
pub struct Switch {
    #[operand]
    selector: Int32,
    #[successors(keyed)]
    cases: SwitchCase,
    #[successor]
    fallback: Successor,
}

impl Canonicalizable for Switch {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        let name = context
            .get_or_register_dialect::<ControlFlowDialect>()
            .expect_registered_name::<Self>();
        rewrites.push(crate::canonicalization::SimplifyCondBrLikeSwitch::new(context.clone()));
        rewrites.push(crate::canonicalization::SimplifySwitchFallbackOverlap::new(context.clone()));
        rewrites.push(crate::canonicalization::SplitCriticalEdges::for_op(context.clone(), name));
    }
}

impl BranchOpInterface for Switch {
    #[inline]
    fn get_successor_for_operands(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> Option<SuccessorInfo> {
        // The selector and the keys are compared as 32-bit patterns: a selector constant
        // `0xffff_ffff` selects the key `-1`.
        let selector = {
            let attr = operands[0].as_ref()?.borrow();
            let selector = attr.as_attr().as_trait::<dyn IntegerLikeAttr>()?.as_int();
            if selector.width() != 32 {
                return None;
            }
            selector.unsigned() as u32
        };

        for switch_case in self.cases().iter() {
            let key = *switch_case.key();
            if selector == key as u32 {
                return Some(*switch_case.info());
            }
        }

        // If we reach here, no selector match was found, so use the fallback successor
        Some(self.successors().all().as_slice().last().copied().unwrap())
    }
}

/// Represents a single branch target by matching a specific selector value in a [Switch]
/// operation.
#[derive(Debug, Clone)]
pub struct SwitchCase {
    pub value: UnsafeIntrusiveEntityRef<I32Attr>,
    pub successor: BlockRef,
    pub arguments: Vec<ValueRef>,
}

impl SwitchCase {
    pub fn create(value: i32, successor: BlockRef, arguments: Vec<ValueRef>) -> Self {
        let value = successor.borrow().context_rc().create_attribute::<I32Attr, _>(value);
        Self {
            value,
            successor,
            arguments,
        }
    }
}

#[doc(hidden)]
pub struct SwitchCaseRef<'a> {
    pub value: UnsafeIntrusiveEntityRef<I32Attr>,
    pub successor: BlockOperandRef,
    pub arguments: OpOperandRange<'a>,
}

#[doc(hidden)]
pub struct SwitchCaseMut<'a> {
    pub value: UnsafeIntrusiveEntityRef<I32Attr>,
    pub successor: BlockOperandRef,
    pub arguments: OpOperandRangeMut<'a>,
}

impl KeyedSuccessor for SwitchCase {
    type Key = i32;
    type KeyStorage = I32Attr;
    type Repr<'a> = SwitchCaseRef<'a>;
    type ReprMut<'a> = SwitchCaseMut<'a>;

    fn key(&self) -> i32 {
        *self.value.borrow().as_value()
    }

    fn into_parts(self) -> (UnsafeIntrusiveEntityRef<Self::KeyStorage>, BlockRef, Vec<ValueRef>) {
        (self.value, self.successor, self.arguments)
    }

    fn into_repr(
        key: UnsafeIntrusiveEntityRef<Self::KeyStorage>,
        block: BlockOperandRef,
        operands: OpOperandRange<'_>,
    ) -> Self::Repr<'_> {
        SwitchCaseRef {
            value: key,
            successor: block,
            arguments: operands,
        }
    }

    fn into_repr_mut(
        key: UnsafeIntrusiveEntityRef<Self::KeyStorage>,
        block: BlockOperandRef,
        operands: OpOperandRangeMut<'_>,
    ) -> Self::ReprMut<'_> {
        SwitchCaseMut {
            value: key,
            successor: block,
            arguments: operands,
        }
    }
}

/// Choose one of two integers of one type based on an `i1` condition
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ControlFlowDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Select {
    #[operand]
    cond: Bool,
    #[operand]
    first: AnyInteger,
    #[operand]
    second: AnyInteger,
    #[result]
    result: AnyInteger,
}

impl InferTypeOpInterface for Select {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.first().ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// Both arms and the result have one type: a fold returns an arm, or an arm's constant, as the
/// result.
impl Verify<dyn InferTypeOpInterface> for Select {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let (first, second, result) = (self.first().ty(), self.second().ty(), self.result().ty());
        if first == second && first == result {
            return Ok(());
        }
        Err(runic_core::diagnostics::report!(
            message: "invalid operation",
            labels: [
                primary(
                    self.as_operation().span(),
                    alloc::format!(
                        "the arms and the result of '{}' must have one type, but they have \
                         `{first}`, `{second}` and `{result}`",
                        self.as_operation().name()
                    )
                ),
            ],
        ))
    }
}

impl Canonicalizable for Select {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(crate::canonicalization::SimplifySelectOfBools::new(context));
    }
}

impl Select {
    /// Folds a select whose arms cannot differ: both arms are the same value, or one arm is
    /// poison, in which case choosing the other arm is a valid refinement.
    fn fold_degenerate_arms(
        &self,
        first: &Option<AttributeRef>,
        second: &Option<AttributeRef>,
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        let is_poison = |attr: &Option<AttributeRef>| {
            attr.as_ref()
                .is_some_and(|attr| attr.borrow().as_attr().downcast_ref::<PoisonAttr>().is_some())
        };
        let first_value = self.first().as_value_ref();
        let second_value = self.second().as_value_ref();
        if first_value == second_value || is_poison(second) {
            results.push(
                (*first)
                    .map(OpFoldResult::Attribute)
                    .unwrap_or(OpFoldResult::Value(first_value)),
            );
            return FoldResult::Ok(());
        }
        if is_poison(first) {
            results.push(
                (*second)
                    .map(OpFoldResult::Attribute)
                    .unwrap_or(OpFoldResult::Value(second_value)),
            );
            return FoldResult::Ok(());
        }
        FoldResult::Failed
    }
}

impl Foldable for Select {
    #[inline]
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let first = matchers::foldable_operand().matches(&self.first().as_operand_ref());
        let second = matchers::foldable_operand().matches(&self.second().as_operand_ref());
        if self.fold_degenerate_arms(&first, &second, results).is_ok() {
            return FoldResult::Ok(());
        }
        if let Some(cond) = matchers::foldable_operand()
            .matches(&self.cond().as_operand_ref())
            .and_then(|attr| constant_condition(&attr))
        {
            let maybe_folded = if cond {
                matchers::foldable_operand()
                    .matches(&self.first().as_operand_ref())
                    .map(OpFoldResult::Attribute)
                    .or_else(|| Some(OpFoldResult::Value(self.first().as_value_ref())))
            } else {
                matchers::foldable_operand()
                    .matches(&self.second().as_operand_ref())
                    .map(OpFoldResult::Attribute)
                    .or_else(|| Some(OpFoldResult::Value(self.second().as_value_ref())))
            };

            if let Some(folded) = maybe_folded {
                results.push(folded);
                return FoldResult::Ok(());
            }
        }

        FoldResult::Failed
    }

    #[inline(always)]
    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        if self.fold_degenerate_arms(&operands[1], &operands[2], results).is_ok() {
            return FoldResult::Ok(());
        }
        if let Some(cond) = operands[0].as_ref().and_then(constant_condition) {
            let maybe_folded = if cond {
                operands[1]
                    .map(OpFoldResult::Attribute)
                    .or_else(|| Some(OpFoldResult::Value(self.first().as_value_ref())))
            } else {
                operands[2]
                    .map(OpFoldResult::Attribute)
                    .or_else(|| Some(OpFoldResult::Value(self.second().as_value_ref())))
            };

            if let Some(folded) = maybe_folded {
                results.push(folded);
                return FoldResult::Ok(());
            }
        }
        FoldResult::Failed
    }
}

/// A check that traps with `message` when `cond`, an `i1`, is `0`: `cf.assert %cond, "message";`.
///
/// It has no results and is not a terminator: when `cond` is `1`, execution continues with the
/// next operation. It declares a write effect, so dead-code elimination keeps it and CSE never
/// merges or removes one. Its one canonicalization erases an assert whose condition is the constant
/// `1`; an assert of the constant `0` is the trap, and stays.
#[derive(EffectOpInterface)]
#[operation(
    dialect = ControlFlowDialect,
    implements(MemoryEffectOpInterface, OpPrinter)
)]
#[effects(MemoryEffect(MemoryEffect::Write))]
pub struct Assert {
    #[operand]
    cond: Bool,
    #[attr(hidden)]
    message: StringAttr,
}

impl Canonicalizable for Assert {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(crate::canonicalization::EraseTrueAssert::new(context));
    }
}

impl OpPrinter for Assert {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        let op = self.as_operation();
        printer.print_space();
        printer.print_value_uses(op.operands().group(0).as_value_range());
        *printer += runic_core::formatter::const_text(", ");
        printer.print_string(self.get_message().as_str());
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

impl OpParser for Assert {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let cond = parser.parse_operand(/*allow_result_number=*/ true)?;
        parser.parse_comma()?;
        let message = parser.parse_string()?.into_inner();
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

        // The condition is a use of a value defined earlier, which carries its type, as in the
        // derived format of the other operations.
        let unknown = parser.context().unknown_type();
        let cond = parser.resolve_operand(cond, unknown)?;
        state.add_operand(cond);
        state.add_attribute(
            "message",
            parser.context_rc().create_attribute::<StringAttr, _>(message),
        );
        state.attrs.extend(attrs);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, string::String, vec, vec::Vec};

    use runic_core::{
        diagnostics::Report,
        dialects::builtin::{BuiltinOpBuilder, Ret, attributes::BoolAttr},
        print::AsmPrinter,
        testing::{Test, parse_function_fixpoint},
    };
    use runic_expect_test::expect;

    use super::*;
    use crate::ControlFlowOpBuilder;

    #[test]
    fn switch_building() {
        let mut test = Test::new("foo", |cx| ([cx.i32()], []));
        let context = test.context_rc();
        let selector = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        let mut builder = test.function_builder();
        let block2 = builder.create_block();
        let block3 = builder.create_block();
        builder.append_block_param(block3, context.i32(), SourceSpan::UNKNOWN);
        let fallback = builder.create_block();
        let cases = vec![
            SwitchCase {
                value: context.create_attribute::<I32Attr, _>(1i32),
                successor: block2,
                arguments: vec![],
            },
            SwitchCase {
                value: context.create_attribute::<I32Attr, _>(2i32),
                successor: block3,
                arguments: vec![selector],
            },
        ];
        let op = builder.switch(selector, cases, fallback, [], SourceSpan::UNKNOWN).unwrap();
        let switch_op = op.borrow();

        assert_eq!(switch_op.fallback().successor(), fallback);
        let cases = switch_op.cases();
        let block2_case = cases.get(0).unwrap();
        assert_eq!(block2_case.block(), block2);
        assert_eq!(*block2_case.key(), 1i32);
        let block3_case = cases.get(1).unwrap();
        assert_eq!(block3_case.block(), block3);
        assert_eq!(*block3_case.key(), 2i32);
        assert_eq!(block3_case.arguments().len(), 1);
        assert_eq!(block3_case.arguments()[0].borrow().as_value_ref(), selector);
    }

    /// A `cf.switch` built with an empty `cases` list still places the fallback successor in its
    /// own group (group 1), after the empty keyed-successor group, so `Switch::fallback()` finds
    /// it.
    #[test]
    fn switch_building_with_empty_cases() {
        let mut test = Test::new("foo", |cx| ([cx.i32()], []));
        let selector = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        let mut builder = test.function_builder();
        let fallback = builder.create_block();

        let op = builder.switch(selector, vec![], fallback, [], SourceSpan::UNKNOWN).unwrap();
        let switch_op = op.borrow();

        assert_eq!(switch_op.fallback().successor(), fallback);
        assert_eq!(switch_op.cases().len(), 0);
    }

    /// A `cf.switch` with more than 255 cases, each forwarding an argument to a shared
    /// destination, so that the op holds more than 255 successors, operand groups and operands at
    /// once.
    #[test]
    fn switch_building_with_more_than_255_cases_and_arguments() {
        const NUM_CASES: usize = 300;

        let mut test = Test::new("foo", |cx| ([cx.i32()], []));
        let context = test.context_rc();
        let selector = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        let mut builder = test.function_builder();
        let dest = builder.create_block();
        builder.append_block_param(dest, context.i32(), SourceSpan::UNKNOWN);
        let fallback = builder.create_block();
        let cases = (0..NUM_CASES)
            .map(|i| SwitchCase {
                value: context.create_attribute::<I32Attr, _>(i as i32),
                successor: dest,
                arguments: vec![selector],
            })
            .collect::<Vec<_>>();

        let op = builder.switch(selector, cases, fallback, [], SourceSpan::UNKNOWN).unwrap();
        let switch_op = op.borrow();

        assert_eq!(switch_op.fallback().successor(), fallback);
        let cases = switch_op.cases();
        assert_eq!(cases.len(), NUM_CASES);
        let last_case = cases.get(NUM_CASES - 1).unwrap();
        assert_eq!(last_case.block(), dest);
        assert_eq!(*last_case.key(), (NUM_CASES - 1) as i32);
        assert_eq!(last_case.arguments().len(), 1);
        assert_eq!(last_case.arguments()[0].borrow().as_value_ref(), selector);
        assert_eq!(switch_op.operands().all().len(), NUM_CASES + 1);
    }

    /// The source of a function whose `cf.switch` has `num_cases` zero-argument keyed cases and a
    /// zero-argument fallback, all targeting one block.
    fn wide_switch_source(num_cases: usize) -> String {
        let cases = (0..num_cases)
            .map(|i| format!("#builtin.i32<{i}> -> ^dest"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "\
builtin.function public extern(\"C\") @wide(%s: i32) -> i32 {{
    cf.switch %s [{cases}], ^dest : (i32);
^dest:
    builtin.ret %s : (i32);
}};"
        )
    }

    /// Parses `source` as a function in a fresh context, keeping the context alive with it.
    fn parse_wide_switch(
        name: &str,
        source: &str,
    ) -> Result<(Rc<Context>, runic_core::dialects::builtin::FunctionRef), Report> {
        let context = Rc::new(Context::default());
        let config = runic_core::parse::ParserConfig::new(context.clone());
        let function = runic_core::parse::parse::<runic_core::dialects::builtin::Function>(
            config, name, source,
        )
        .into_result()?;
        Ok((context, function))
    }

    /// The widest keyed `cf.switch` the textual IR accepts: the selector and the fallback take
    /// the two reserved operand groups, and the cases fill the remaining 16-bit group indices
    /// without any case wrapping onto the selector's group.
    #[test]
    fn parse_switch_with_the_most_representable_cases() -> Result<(), Report> {
        const NUM_CASES: usize = u16::MAX as usize - 1;

        let (_context, function) =
            parse_wide_switch("parse_switch_widest.hir", &wide_switch_source(NUM_CASES))?;
        let function = function.borrow();
        let switch_op = function
            .body()
            .entry()
            .terminator()
            .unwrap()
            .try_downcast_op::<Switch>()
            .expect("expected entry block to terminate with cf.switch");
        let switch_op = switch_op.borrow();

        let cases = switch_op.cases();
        assert_eq!(cases.len(), NUM_CASES);
        assert!(cases.get(NUM_CASES - 1).unwrap().arguments().is_empty());
        assert!(switch_op.fallback().arguments.is_empty());
        // The selector is the only operand: no case forwards anything
        assert_eq!(switch_op.operands().all().len(), 1);

        Ok(())
    }

    /// One keyed case past the 16-bit operand-group limit is a parse error, not a case that
    /// silently shares the selector's operand group.
    #[test]
    fn parse_switch_with_too_many_cases_is_rejected() {
        const NUM_CASES: usize = u16::MAX as usize;

        let error = parse_wide_switch("parse_switch_too_wide.hir", &wide_switch_source(NUM_CASES))
            .err()
            .expect("expected the switch to be rejected");
        let message = format!("{error}");
        assert!(message.contains("too many operand groups"), "{message}");
    }

    /// The value returned by the `builtin.ret` terminating `block`.
    fn returned_value(block: BlockRef) -> ValueRef {
        let block = block.borrow();
        let terminator = block.terminator().expect("expected block to have a terminator");
        let ret = terminator
            .try_downcast_op::<Ret>()
            .expect("expected block to terminate with builtin.ret");
        let ret = ret.borrow();
        let values = ret.values();
        let operand = values.iter().next().expect("expected builtin.ret to have an operand");
        operand.borrow().as_value_ref()
    }

    #[test]
    fn parse_generic_br_with_bare_successor() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = r#"
builtin.function public extern("C") @branch(%a: i32) -> i32 {
    "cf.br"()[^dest] : () -> ();
^dest:
    builtin.ret %a : (i32);
};"#;
        let (function, _) = parse_function_fixpoint(&context, "parse_generic_br.hir", source)?;
        let function = function.borrow();
        let body = function.body();
        let entry = body.entry();
        let arg = entry.arguments()[0] as ValueRef;
        let branch = entry.terminator().unwrap().try_downcast_op::<Br>().unwrap();
        let branch = branch.borrow();
        let target = branch.target();
        assert!(target.arguments.is_empty());
        assert_eq!(returned_value(target.successor()), arg);

        let flags = Default::default();
        let mut printer = AsmPrinter::root(context.clone(), &flags);
        printer.print_operation_generic(branch.as_operation());
        expect![[r#""cf.br"() [ ^block2 ] : () -> ();"#]]
            .assert_eq(&format!("{}", printer.finish()));
        Ok(())
    }

    #[test]
    fn parse_generic_cond_br_with_bare_successors() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = r#"
builtin.function public extern("C") @branch(%c: i1, %a: i32, %b: i32) -> i32 {
    "cf.cond_br"(%c)[^yes, ^no] : (i1) -> ();
^yes:
    builtin.ret %a : (i32);
^no:
    builtin.ret %b : (i32);
};"#;
        let (function, _) = parse_function_fixpoint(&context, "parse_generic_cond_br.hir", source)?;
        let function = function.borrow();
        let body = function.body();
        let entry = body.entry();
        let branch = entry.terminator().unwrap().try_downcast_op::<CondBr>().unwrap();
        let branch = branch.borrow();
        assert_eq!(branch.condition().as_value_ref(), entry.arguments()[0] as ValueRef);
        let then_dest = branch.then_dest();
        let else_dest = branch.else_dest();
        assert!(then_dest.arguments.is_empty());
        assert!(else_dest.arguments.is_empty());
        assert_eq!(returned_value(then_dest.successor()), entry.arguments()[1] as ValueRef);
        assert_eq!(returned_value(else_dest.successor()), entry.arguments()[2] as ValueRef);

        let flags = Default::default();
        let mut printer = AsmPrinter::root(context.clone(), &flags);
        printer.print_operation_generic(branch.as_operation());
        expect![[r#""cf.cond_br"(%0) [ ^block2, ^block3 ] : i1 -> ();"#]]
            .assert_eq(&format!("{}", printer.finish()));
        Ok(())
    }

    #[test]
    fn parse_generic_branches_reject_incorrect_successor_counts() {
        use runic_core::{
            dialects::builtin::Function,
            parse::{ParserConfig, parse},
        };

        for (name, operands, types, successors) in [
            ("cf.cond_br", "%c", "i1", "[^dest]"),
            ("cf.cond_br", "%c", "i1", "[^dest, ^dest, ^dest]"),
            ("cf.cond_br", "%c", "i1", ""),
            ("cf.br", "", "", ""),
            ("cf.br", "", "", "[^dest, ^dest]"),
            ("builtin.ret", "", "", "[^dest]"),
        ] {
            let context = Rc::new(Context::default());
            context.get_or_register_dialect::<ControlFlowDialect>();
            let source = format!(
                r#"builtin.function public extern("C") @branch(%c: i1) {{
    "{name}"({operands}){successors} : ({types}) -> ();
^dest:
    builtin.ret;
}};"#
            );
            let error = parse::<Function>(
                ParserConfig::new(context.clone()),
                "parse_generic_successor_count.hir",
                &source,
            )
            .into_result()
            .err()
            .expect("incorrect successor counts must be rejected during parsing");
            let message = format!("{error}");
            assert!(message.contains("invalid successor count"), "{message}");
        }
    }

    #[test]
    fn parse_generic_cond_br_rejects_incorrect_operand_counts() {
        use runic_core::{
            dialects::builtin::Function,
            parse::{ParserConfig, parse},
        };

        for (operands, types) in [("", ""), ("%c, %c", "i1, i1")] {
            let context = Rc::new(Context::default());
            context.get_or_register_dialect::<ControlFlowDialect>();
            let source = format!(
                r#"builtin.function public extern("C") @branch(%c: i1) {{
    "cf.cond_br"({operands})[^yes, ^no] : ({types}) -> ();
^yes:
    builtin.ret;
^no:
    builtin.ret;
}};"#
            );
            let error = parse::<Function>(
                ParserConfig::new(context.clone()),
                "parse_generic_operand_count.hir",
                &source,
            )
            .into_result()
            .err()
            .expect("incorrect operand counts must be rejected");
            let message = format!("{error}");
            assert!(message.contains("invalid operand count"), "{message}");
        }
    }

    #[test]
    fn parse_cond_br_rejects_non_boolean_condition() {
        use runic_core::{
            dialects::builtin::Function,
            parse::{ParserConfig, parse},
        };

        for branch in [
            r#""cf.cond_br"(%c)[^yes, ^no] : (i32) -> ();"#,
            "cf.cond_br %c ^yes, ^no : (i32);",
        ] {
            let context = Rc::new(Context::default());
            context.get_or_register_dialect::<ControlFlowDialect>();
            let source = format!(
                r#"builtin.function public extern("C") @branch(%c: i32) {{
    {branch}
^yes:
    builtin.ret;
^no:
    builtin.ret;
}};"#
            );
            let error = parse::<Function>(
                ParserConfig::new(context.clone()),
                "parse_non_boolean_condition.hir",
                &source,
            )
            .into_result()
            .err()
            .expect("non-boolean branch conditions must be rejected");
            let message = format!("{error}");
            assert!(message.contains("invalid operand"), "{message}");
            assert!(message.contains("1-bit integral type"), "{message}");
            assert!(message.contains("got i32"), "{message}");
        }
    }

    #[test]
    fn parse_generic_switch_rejects_unsupported_successor_layout() {
        use runic_core::{
            dialects::builtin::Function,
            parse::{ParserConfig, parse},
        };

        // A flat successor list cannot distinguish the keyed cases from the fallback. Omitting
        // the list must not bypass the diagnostic and produce missing successor groups either.
        for successors in ["[^dest]", "[^dest, ^dest]", "[]", ""] {
            let context = Rc::new(Context::default());
            context.get_or_register_dialect::<ControlFlowDialect>();
            let source = format!(
                r#"builtin.function public extern("C") @branch(%s: i32) -> i32 {{
    "cf.switch"(%s){successors} : (i32) -> ();
^dest:
    builtin.ret %s : (i32);
}};"#
            );
            let Err(error) = parse::<Function>(
                ParserConfig::new(context.clone()),
                "parse_generic_switch.hir",
                source,
            )
            .into_result() else {
                panic!("expected generic cf.switch with successors '{successors}' to be rejected");
            };
            let message = format!("{error}");
            assert!(
                message.contains(
                    "generic assembly cannot represent the successor layout of 'cf.switch'"
                ),
                "{message}"
            );
        }
    }

    /// Statically-known successor lists (`cf.cond_br %c ^a, ^b`) parse with the `, ` separator
    /// the printer emits between successors, and the condition operand stays attached to the
    /// operation rather than leaking into the first successor's operand group.
    #[test]
    fn parse_cond_br_with_bare_successors() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @branch(%c: i1, %a: i32, %b: i32) -> i32 {
    cf.cond_br %c ^yes, ^no : (i1);
^yes:
    builtin.ret %a : (i32);
^no:
    builtin.ret %b : (i32);
};";
        let (function, printed) = parse_function_fixpoint(&context, "parse_cond_br.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @branch(%0: i1, %1: i32, %2: i32) -> i32 {
                cf.cond_br %0 ^block2, ^block3 : (i1);
            ^block2:
                builtin.ret %1 : (i32);
            ^block3:
                builtin.ret %2 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let body = function.body();
        let entry = body.entry();
        let arg_c = entry.arguments()[0] as ValueRef;
        let arg_a = entry.arguments()[1] as ValueRef;
        let arg_b = entry.arguments()[2] as ValueRef;

        let cond_br = entry
            .terminator()
            .unwrap()
            .try_downcast_op::<CondBr>()
            .expect("expected entry block to terminate with cf.cond_br");
        let cond_br = cond_br.borrow();
        assert_eq!(cond_br.condition().as_value_ref(), arg_c);
        let then_dest = cond_br.then_dest();
        let else_dest = cond_br.else_dest();
        assert!(then_dest.arguments.is_empty());
        assert!(else_dest.arguments.is_empty());
        assert_eq!(returned_value(then_dest.successor()), arg_a);
        assert_eq!(returned_value(else_dest.successor()), arg_b);

        Ok(())
    }

    /// Successor argument lists (`^block(%v, ...)`) parse and attach to the operand group of
    /// the successor they follow, and the target block's own arguments bind as usual.
    #[test]
    fn parse_cond_br_with_successor_arguments() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @branch_args(%c: i1, %a: i32, %b: i32) -> i32 {
    cf.cond_br %c ^one(%a : i32), ^two(%a, %b : i32, i32) : (i1);
^one(%x: i32):
    builtin.ret %x : (i32);
^two(%y: i32, %z: i32):
    builtin.ret %z : (i32);
};";
        let (function, printed) =
            parse_function_fixpoint(&context, "parse_cond_br_args.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @branch_args(%0: i1, %1: i32, %2: i32) -> i32 {
                cf.cond_br %0 ^block2(%1 : i32), ^block3(%1, %2 : i32, i32) : (i1);
            ^block2(%3: i32):
                builtin.ret %3 : (i32);
            ^block3(%4: i32, %5: i32):
                builtin.ret %5 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let body = function.body();
        let entry = body.entry();
        let arg_a = entry.arguments()[1] as ValueRef;
        let arg_b = entry.arguments()[2] as ValueRef;

        let cond_br = entry
            .terminator()
            .unwrap()
            .try_downcast_op::<CondBr>()
            .expect("expected entry block to terminate with cf.cond_br");
        let cond_br = cond_br.borrow();

        let then_dest = cond_br.then_dest();
        let then_args = then_dest
            .arguments
            .iter()
            .map(|o| o.borrow().as_value_ref())
            .collect::<Vec<_>>();
        assert_eq!(then_args, [arg_a]);

        let else_dest = cond_br.else_dest();
        let else_args = else_dest
            .arguments
            .iter()
            .map(|o| o.borrow().as_value_ref())
            .collect::<Vec<_>>();
        assert_eq!(else_args, [arg_a, arg_b]);

        // The successor blocks' own arguments are bound and usable: `^two` returns its second
        // block argument.
        let else_block = else_dest.successor();
        let expected = else_block.borrow().arguments()[1] as ValueRef;
        assert_eq!(returned_value(else_block), expected);

        Ok(())
    }

    /// Block labels register in the parser's block name map when defined: a branch may
    /// reference a block defined earlier in the region (backward reference), and a labeled
    /// definition must unify with targets forward-declared by earlier branches without
    /// dropping the parsed block body.
    #[test]
    fn parse_backward_block_references() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @looped(%start: i1) -> i1 {
    cf.br ^header(%start : i1);
^header(%v: i1):
    cf.cond_br %v ^header(%v : i1), ^exit : (i1);
^exit:
    builtin.ret %v : (i1);
};";
        let (function, printed) =
            parse_function_fixpoint(&context, "parse_backward_refs.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @looped(%0: i1) -> i1 {
                cf.br ^block2(%0 : i1);
            ^block2(%1: i1):
                cf.cond_br %1 ^block2(%1 : i1), ^block3 : (i1);
            ^block3:
                builtin.ret %1 : (i1);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let body = function.body();
        assert_eq!(body.body().iter().count(), 3, "expected all three blocks to be parsed");

        let entry = body.entry();
        let br = entry
            .terminator()
            .unwrap()
            .try_downcast_op::<Br>()
            .expect("expected entry block to terminate with cf.br");
        let header = br.borrow().target().successor();

        // The labeled definition must populate the forward-declared block, not a placeholder.
        let header_block = header.borrow();
        assert_eq!(header_block.num_arguments(), 1);
        assert_eq!(header_block.body().iter().count(), 1, "header block body must not be empty");

        // The backward reference resolves to that same block, forwarding its argument.
        let cond_br = header_block
            .terminator()
            .unwrap()
            .try_downcast_op::<CondBr>()
            .expect("expected header block to terminate with cf.cond_br");
        let cond_br = cond_br.borrow();
        let then_dest = cond_br.then_dest();
        assert_eq!(then_dest.successor(), header);
        let loop_args = then_dest
            .arguments
            .iter()
            .map(|o| o.borrow().as_value_ref())
            .collect::<Vec<_>>();
        assert_eq!(loop_args, [header_block.arguments()[0] as ValueRef]);

        Ok(())
    }

    /// Keyed successor groups (`cf.switch`) parse: each case key maps to its own successor and
    /// operand group, and the trailing successor is the fallback.
    #[test]
    fn parse_switch_with_keyed_successors() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @select(%sel: i32, %a: i32, %b: i32) -> i32 {
    cf.switch %sel [#builtin.i32<1> -> ^one(%a : i32)], ^fallback : (i32);
^one(%x: i32):
    builtin.ret %x : (i32);
^fallback:
    builtin.ret %b : (i32);
};";
        let (function, printed) = parse_function_fixpoint(&context, "parse_switch.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @select(%0: i32, %1: i32, %2: i32) -> i32 {
                cf.switch %0 [#builtin.i32<1> -> ^block2(%1 : i32)], ^block3 : (i32);
            ^block2(%3: i32):
                builtin.ret %3 : (i32);
            ^block3:
                builtin.ret %2 : (i32);
            };"#]]
        .assert_eq(&printed);

        let function = function.borrow();
        let body = function.body();
        let entry = body.entry();
        let arg_a = entry.arguments()[1] as ValueRef;
        let arg_b = entry.arguments()[2] as ValueRef;

        let switch_op = entry
            .terminator()
            .unwrap()
            .try_downcast_op::<Switch>()
            .expect("expected entry block to terminate with cf.switch");
        let switch_op = switch_op.borrow();

        let cases = switch_op.cases();
        assert_eq!(cases.len(), 1);
        let case = cases.iter().next().unwrap();
        assert_eq!(*case.key(), 1);
        let case_args =
            case.arguments().iter().map(|o| o.borrow().as_value_ref()).collect::<Vec<_>>();
        assert_eq!(case_args, [arg_a]);
        // The case target returns its own block argument, fed by the case's operand.
        let case_block = case.block();
        let expected = case_block.borrow().arguments()[0] as ValueRef;
        assert_eq!(returned_value(case_block), expected);

        let fallback = switch_op.fallback();
        assert!(fallback.arguments.is_empty());
        assert_eq!(returned_value(fallback.successor()), arg_b);

        Ok(())
    }

    /// A switch with the keys `-1`, `0` and `i32::MAX` builds, prints each key as a signed
    /// `#builtin.i32<..>` and parses back.
    #[test]
    fn switch_keys_are_i32() -> Result<(), Report> {
        let mut test = Test::new("keys", |cx| ([cx.i32()], [cx.i32()]));
        let selector = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        {
            let mut builder = test.function_builder();
            let negative = builder.create_block();
            let zero = builder.create_block();
            let max = builder.create_block();
            let fallback = builder.create_block();
            let cases = vec![
                SwitchCase::create(-1, negative, vec![]),
                SwitchCase::create(0, zero, vec![]),
                SwitchCase::create(i32::MAX, max, vec![]),
            ];
            builder.switch(selector, cases, fallback, [], SourceSpan::UNKNOWN)?;
            for block in [negative, zero, max, fallback] {
                builder.switch_to_block(block);
                builder.ret([selector], SourceSpan::UNKNOWN)?;
            }
        }
        let function = test.function();
        function.borrow().as_operation().verify()?;
        let text = format!("{}", function.borrow().as_operation());
        expect![[r#"
            builtin.function public extern("C") @keys(%0: i32) -> i32 {
                cf.switch %0 [#builtin.i32<-1> -> ^block1, 
                    #builtin.i32<0> -> ^block2, 
                    #builtin.i32<2147483647> -> ^block3], ^block4 : (i32);
            ^block1:
                builtin.ret %0 : (i32);
            ^block2:
                builtin.ret %0 : (i32);
            ^block3:
                builtin.ret %0 : (i32);
            ^block4:
                builtin.ret %0 : (i32);
            };"#]]
        .assert_eq(&text);

        let context = Rc::new(Context::default());
        // A parsed function numbers its blocks from `^block2`, a built one from `^block1`.
        let (parsed, printed) = parse_function_fixpoint(&context, "switch_keys.hir", &text)?;
        assert!(printed.contains("#builtin.i32<-1> -> ^block2"), "{printed}");
        let parsed = parsed.borrow();
        let switch_op =
            parsed.body().entry().terminator().unwrap().try_downcast_op::<Switch>().unwrap();
        let switch_op = switch_op.borrow();
        let keys = switch_op.cases().iter().map(|case| *case.key()).collect::<Vec<_>>();
        assert_eq!(keys, [-1, 0, i32::MAX]);
        Ok(())
    }

    /// A constant selector is compared with the keys as 32-bit patterns: `0xffff_ffff` selects
    /// the key `-1`; a selector that matches no key selects the fallback.
    #[test]
    fn switch_constant_selector_compares_32_bit_patterns() {
        let mut test = Test::new("select", |cx| ([cx.i32()], []));
        let context = test.context_rc();
        let selector = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        let mut builder = test.function_builder();
        let negative = builder.create_block();
        let max = builder.create_block();
        let fallback = builder.create_block();
        let cases = vec![
            SwitchCase::create(-1, negative, vec![]),
            SwitchCase::create(i32::MAX, max, vec![]),
        ];
        let op = builder.switch(selector, cases, fallback, [], SourceSpan::UNKNOWN).unwrap();
        let op = op.borrow();
        let successor_for = |bits: u128| {
            let attr = context.int_attr(Int::from_bits(32, bits), context.i32());
            op.get_successor_for_operands(&[Some(attr.as_attribute_ref())])
                .expect("a constant selector selects a successor")
                .successor()
        };
        assert_eq!(successor_for(0xffff_ffff), negative);
        assert_eq!(successor_for(0x7fff_ffff), max);
        assert_eq!(successor_for(5), fallback);
        assert_eq!(successor_for(0x8000_0000), fallback);
        assert!(op.get_successor_for_operands(&[None]).is_none());
    }

    /// A constant condition of `cf.cond_br` is read with `as_int`, whether it is an `IntAttr` (an
    /// `i1` constant) or a `BoolAttr`.
    #[test]
    fn cond_br_constant_condition_is_read_as_an_integer() {
        let mut test = Test::new("branch", |cx| ([cx.i1()], []));
        let context = test.context_rc();
        let cond = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        let mut builder = test.function_builder();
        let yes = builder.create_block();
        let no = builder.create_block();
        let op = builder.cond_br(cond, yes, [], no, [], SourceSpan::UNKNOWN).unwrap();
        let op = op.borrow();
        let conditions = [
            (context.int_attr(Int::from(true), context.i1()).as_attribute_ref(), yes),
            (context.int_attr(Int::from(false), context.i1()).as_attribute_ref(), no),
            (context.create_attribute::<BoolAttr, _>(true).as_attribute_ref(), yes),
            (context.create_attribute::<BoolAttr, _>(false).as_attribute_ref(), no),
        ];
        for (attr, expected) in conditions {
            let successor = op.get_successor_for_operands(&[Some(attr)]).unwrap().successor();
            assert_eq!(successor, expected);
        }
        assert!(op.get_successor_for_operands(&[None]).is_none());
    }

    /// `cf.select` with a constant condition folds to the chosen arm, whether the condition is an
    /// `IntAttr` or a `BoolAttr`, and whether it is folded from the IR or from given operands.
    #[test]
    fn select_constant_condition_is_read_as_an_integer() {
        use runic_dialect_arith::ArithOpBuilder;

        let mut test = Test::new("select", |cx| ([cx.i32(), cx.i32()], []));
        let context = test.context_rc();
        let entry = test.function().borrow().entry_block();
        let a = entry.borrow().arguments()[0] as ValueRef;
        let b = entry.borrow().arguments()[1] as ValueRef;
        let mut builder = test.function_builder();
        for value in [true, false] {
            let cond = builder.i1(value, SourceSpan::UNKNOWN);
            let result = builder.select(cond, a, b, SourceSpan::UNKNOWN).unwrap();
            let op = result.borrow().get_defining_op().unwrap();
            let op = op.borrow();
            let select = op.downcast_ref::<Select>().unwrap();
            let mut results = SmallVec::<[OpFoldResult; 1]>::new();
            assert!(select.fold(&mut results).is_ok());
            let expected = if value { a } else { b };
            assert!(matches!(results[0], OpFoldResult::Value(v) if v == expected));

            for attr in [
                context.int_attr(Int::from(value), context.i1()).as_attribute_ref(),
                context.create_attribute::<BoolAttr, _>(value).as_attribute_ref(),
            ] {
                let mut results = SmallVec::<[OpFoldResult; 1]>::new();
                assert!(select.fold_with(&[Some(attr), None, None], &mut results).is_ok());
                assert!(matches!(results[0], OpFoldResult::Value(v) if v == expected));
            }
        }
    }

    /// The message of the first `cf.assert` in the entry block of `function`.
    fn assert_message(function: runic_core::dialects::builtin::FunctionRef) -> String {
        let entry = function.borrow().entry_block();
        let op = entry
            .borrow()
            .body()
            .iter()
            .find(|op| op.is::<Assert>())
            .expect("expected a cf.assert")
            .as_operation_ref();
        let op = op.borrow();
        op.downcast_ref::<Assert>().unwrap().get_message().as_str().into()
    }

    /// `cf.assert` builds, verifies, prints as `cf.assert %c, "message";` with the message
    /// escaped, and parses back to the same message.
    #[test]
    fn assert_round_trips() -> Result<(), Report> {
        let message = "say \"no\" \\ then\nstop";
        let mut test = Test::new("check", |cx| ([cx.i1()], []));
        let cond = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        {
            let mut builder = test.function_builder();
            let op = builder.assert(cond, message, SourceSpan::UNKNOWN)?;
            assert_eq!(op.borrow().get_message().as_str(), message);
            assert_eq!(op.borrow().cond().as_value_ref(), cond);
            builder.ret([], SourceSpan::UNKNOWN)?;
        }
        let function = test.function();
        function.borrow().as_operation().verify()?;
        let text = format!("{}", function.borrow().as_operation());
        expect![[r#"
            builtin.function public extern("C") @check(%0: i1) {
                cf.assert %0, "say \"no\" \\ then\nstop";
                builtin.ret;
            };"#]]
        .assert_eq(&text);

        let context = Rc::new(Context::default());
        let (parsed, _) = parse_function_fixpoint(&context, "assert.hir", &text)?;
        assert_eq!(assert_message(parsed), message);
        Ok(())
    }

    /// Attributes that are not properties are written after the message and survive text.
    #[test]
    fn assert_keeps_other_attributes_in_text() -> Result<(), Report> {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @check(%c: i1) {
    cf.assert %c, \"m\" attributes { note = #builtin.i32<7> };
    builtin.ret;
};";
        let (function, printed) = parse_function_fixpoint(&context, "assert_attrs.hir", source)?;
        expect![[r#"
            builtin.function public extern("C") @check(%0: i1) {
                cf.assert %0, "m" attributes { note = #builtin.i32<7> };
                builtin.ret;
            };"#]]
        .assert_eq(&printed);
        let function = function.borrow();
        let op = function.body().entry().body().front().as_pointer().unwrap();
        let op = op.borrow();
        let note = op.get_attribute("note").expect("the attribute survives");
        assert_eq!(*note.borrow().downcast_ref::<I32Attr>().unwrap().as_value(), 7);
        Ok(())
    }

    /// The condition of `cf.assert` is an `i1`: an `i32` condition is rejected.
    #[test]
    fn assert_rejects_a_non_boolean_condition() {
        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @check(%c: i32) {
    cf.assert %c, \"m\";
    builtin.ret;
};";
        let error = parse_function_fixpoint(&context, "assert_i32.hir", source)
            .err()
            .expect("an i32 condition must be rejected");
        let message = format!("{error}");
        assert!(message.contains("invalid operand"), "{message}");
        assert!(message.contains("1-bit integral type"), "{message}");

        let mut test = Test::new("check", |cx| ([cx.i32()], []));
        let cond = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        let error = test
            .function_builder()
            .assert(cond, "m", SourceSpan::UNKNOWN)
            .err()
            .expect("the builder must reject an i32 condition");
        let message = format!("{error}");
        assert!(message.contains("expected 1-bit integral type, got i32"), "{message}");
    }

    /// An unused `cf.assert` is not trivially dead, and two identical ones each have exactly a
    /// write effect, so neither is removed or merged with the other.
    #[test]
    fn assert_is_kept_and_never_merged() {
        let mut test = Test::new("check", |cx| ([cx.i1()], []));
        let cond = test.function().borrow().entry_block().borrow().arguments()[0] as ValueRef;
        let mut builder = test.function_builder();
        let first = builder.assert(cond, "m", SourceSpan::UNKNOWN).unwrap();
        let second = builder.assert(cond, "m", SourceSpan::UNKNOWN).unwrap();
        for op in [first, second] {
            let op = op.borrow();
            let op = op.as_operation();
            assert!(!op.implements::<dyn Terminator>());
            assert!(!op.implements::<dyn ConstantLike>());
            assert_eq!(op.num_results(), 0);
            assert!(!op.is_trivially_dead());
            assert!(!op.is_memory_effect_free());
            assert!(op.has_memory_effect(MemoryEffect::Write));
            assert!(op.has_single_memory_effect(MemoryEffect::Write));
        }
    }

    /// Canonicalization erases an assert of the constant `1` and keeps an assert of the constant
    /// `0` and of a value that is not constant.
    #[test]
    fn assert_of_true_is_erased() -> Result<(), Report> {
        use alloc::boxed::Box;

        use runic_core::patterns::RewritePattern;

        use crate::canonicalization::{EraseTrueAssert, testing::apply_with_borrowing_listener};

        let context = Rc::new(Context::default());
        let source = "\
builtin.function public extern(\"C\") @check(%c: i1) {
    %t = arith.constant 1 : i1;
    %f = arith.constant 0 : i1;
    cf.assert %t, \"true\";
    cf.assert %f, \"false\";
    cf.assert %c, \"unknown\";
    builtin.ret;
};";
        let (function, _) = parse_function_fixpoint(&context, "erase_true_assert.hir", source)?;
        let pattern: Box<dyn RewritePattern> = Box::new(EraseTrueAssert::new(context.clone()));
        assert!(apply_with_borrowing_listener(&context, function, pattern)?);
        let printed = format!("{}", function.as_operation_ref().borrow());
        expect![[r#"
            builtin.function public extern("C") @check(%0: i1) {
                %2 = arith.constant 0 : i1;
                cf.assert %2, "false";
                cf.assert %0, "unknown";
                builtin.ret;
            };"#]]
        .assert_eq(&printed);
        Ok(())
    }

    /// A `cf.select` whose arms have two types does not verify, and the diagnostic names the
    /// types. (A fold that chose the second arm would hand an `i64` to an `i32` result.)
    #[test]
    fn a_select_of_two_types_does_not_verify() {
        use runic_core::{dialects::builtin::Function, testing::text::parse};

        let text = "builtin.function public extern(\"C\") @f(%c: i1, %a: i32, %b: i64) -> i32 {
    %s = cf.select %c, %a, %b;
    builtin.ret %s : (i32);
};";
        let err = parse::<Function>(text).err().expect("a select of two types must not verify");
        assert!(
            err.contains(
                "the arms and the result of 'cf.select' must have one type, but they have `i32`, \
                 `i64` and `i32`"
            ),
            "{err}"
        );
    }
}
