use crate::{
    AsValueRange, AttributeRef, Context, Op, OpParser, OpPrinter, OperationState, Report, SmallVec,
    Spanned, Type, Value,
    derive::{EffectOpInterface, operation},
    diagnostics::report,
    dialects::builtin::BuiltinDialect,
    effects::MemoryEffectOpInterface,
    formatter::const_text,
    parse::ParseResult,
    print::AsmPrinter,
    traits::{AnyType, FoldResult, Foldable, InferIntRangeInterface, OpFoldResult},
};

/// Values of some types standing for values of others, with nothing said about how one becomes
/// the other: `%lo, %hi = builtin.unrealized_conversion_cast %x : i128 to (i64, i64);`.
///
/// The cast takes any number of inputs and gives at least one result. Its text gives the inputs,
/// their types and the result types; a type list of one type is written bare, unless that type is
/// a function type, and any other list is in parentheses:
///
/// ```text,ignore
/// %y = builtin.unrealized_conversion_cast %x : i32 to i64;
/// %lo, %hi = builtin.unrealized_conversion_cast %x : i128 to (i64, i64);
/// %x = builtin.unrealized_conversion_cast %lo, %hi : (i64, i64) to i128;
/// %x = builtin.unrealized_conversion_cast : () to i128;
/// ```
///
/// An `attributes {..}` dictionary may follow. The input types written must be the types of the
/// inputs. The operation keeps no property: the result types are those of its results, which the
/// generic form states.
///
/// A dialect conversion uses the cast to say that a value now stands for other values, and folds
/// away the chains of casts whose ends agree
/// ([reconcile_unrealized_casts](crate::conversion::reconcile_unrealized_casts)).
/// The cast has no effects, so an unused one is removed. A cast of one value to its own type folds
/// to that value. It is built with
/// [BuiltinOpBuilder::unrealized_conversion_cast](crate::dialects::builtin::BuiltinOpBuilder::unrealized_conversion_cast),
/// which is given the result types; the typed builder that takes only the inputs gives no result,
/// which the verifier rejects.
#[derive(EffectOpInterface)]
#[operation(
    dialect = BuiltinDialect,
    implements(InferIntRangeInterface, Foldable, MemoryEffectOpInterface, OpPrinter)
)]
pub struct UnrealizedConversionCast {
    /// The values cast
    #[operands]
    inputs: AnyType,
    /// The values they stand for
    #[results]
    outputs: AnyType,
}

/// A cast has at least one result.
impl crate::Verify<dyn MemoryEffectOpInterface> for UnrealizedConversionCast {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        if self.as_operation().num_results() != 0 {
            return Ok(());
        }
        Err(report!(
            message: "invalid operation builtin.unrealized_conversion_cast",
            labels: [primary(self.as_operation().span(), "a cast has at least one result")],
        ))
    }
}

/// A cast of one value to its own type is that value.
impl Foldable for UnrealizedConversionCast {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let op = self.as_operation();
        if op.num_operands() != 1 || op.num_results() != 1 {
            return FoldResult::Failed;
        }
        let input = op.operands()[0].borrow().as_value_ref();
        if input.borrow().ty() != op.results()[0].borrow().ty() {
            return FoldResult::Failed;
        }
        results.push(OpFoldResult::Value(input));
        FoldResult::Ok(())
    }

    fn fold_with(
        &self,
        _operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        self.fold(results)
    }
}

impl InferIntRangeInterface for UnrealizedConversionCast {
    /// The cast says nothing about how values of some types become values of others (a zero
    /// extension, a sign extension, a truncation, a split or anything else), so the input's range
    /// is a fact about the result only when there is one input and one result, and both are
    /// integers of one width. A single result of another width has the range of its type; the
    /// results of any other cast are given no range.
    fn infer_result_ranges(
        &self,
        operands: &[crate::IntegerValueRange],
        set_result_range: &mut dyn FnMut(crate::ValueRef, crate::IntegerValueRange),
    ) {
        let op = self.as_operation();
        let ([range], 1) = (operands, op.num_results()) else {
            return;
        };
        let result = op.results()[0].borrow().as_value_ref();
        let top = crate::ConstantIntRange::max_for_type(result.borrow().ty());
        let range = match range.range() {
            None => range.clone(),
            Some(source) if !top.is_untracked() && source.width() == top.width() => range.clone(),
            Some(_) => crate::IntegerValueRange::new(top),
        };
        set_result_range(result, range);
    }
}

/// `types` as the cast writes a type list: one type that is not a function type bare, any other
/// list in parentheses.
fn print_type_list(printer: &mut AsmPrinter<'_>, types: SmallVec<[Type; 4]>) {
    match types.as_slice() {
        [ty] if !ty.is_function() => printer.print_type(*ty),
        _ => printer.print_type_list(types),
    }
}

impl OpPrinter for UnrealizedConversionCast {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        let inputs = op.operands().group(0);
        if !inputs.is_empty() {
            printer.print_space();
            printer.print_value_uses(inputs.as_value_range());
        }
        *printer += const_text(" : ");
        print_type_list(printer, inputs.iter().map(|input| input.borrow().ty()).collect());
        *printer += const_text(" to ");
        print_type_list(printer, op.results().iter().map(|result| result.borrow().ty()).collect());
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

/// Reads `%a, .. : input-types to result-types`, then an optional `attributes {..}`. Each input
/// must have the type written for it.
impl OpParser for UnrealizedConversionCast {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let mut inputs = SmallVec::<[crate::parse::UnresolvedOperand; 2]>::default();
        parser.parse_optional_operand_list(&mut inputs)?;
        parser.parse_colon()?;
        let mut input_types = SmallVec::<[Type; 4]>::default();
        parser.parse_type_list(&mut input_types)?;
        parser.parse_custom_keyword("to")?;
        let mut result_types = SmallVec::<[Type; 4]>::default();
        parser.parse_type_list(&mut result_types)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let mut values = SmallVec::default();
        parser.resolve_operands(state.span, &inputs, &input_types, &mut values)?;
        state.operands.push(values);
        state.results.extend(result_types);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        format,
        rc::Rc,
        string::{String, ToString},
        vec::Vec,
    };

    use super::*;
    use crate::{
        IntegerValueRange, Operation, OperationRef, SourceSpan, ValueRef,
        dialects::builtin::{BuiltinOpBuilder, Module},
        parse::{ParserConfig, parse_any},
        testing::{
            Test,
            text::{describe, print_generic, round_trip},
        },
    };

    /// A module holding a function `f` with the arguments `params` and the body `body`.
    fn module(params: &str, body: &str) -> String {
        format!(
            "builtin.module public @m {{
    builtin.function public extern(\"C\") @f({params}) {{
{body}
        builtin.ret;
    }};
}};"
        )
    }

    fn parse_text(source: &str) -> Result<(Rc<Context>, OperationRef), String> {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context.clone());
        match parse_any(config, "cast", source).into_result() {
            Ok(op) => Ok((context, op)),
            Err(report) => Err(describe(&report)),
        }
    }

    fn casts(root: OperationRef) -> Vec<OperationRef> {
        let mut found = Vec::new();
        root.borrow().prewalk_all(|op: &Operation| {
            if op.is::<UnrealizedConversionCast>() {
                found.push(op.as_operation_ref());
            }
        });
        found
    }

    const FORMS: &str = "        %5 = builtin.unrealized_conversion_cast %0 : i32 to i64;
        %6, %7 = builtin.unrealized_conversion_cast %1 : i128 to (i64, i64);
        %8 = builtin.unrealized_conversion_cast %2, %3 : (i64, i64) to i128;
        %9 = builtin.unrealized_conversion_cast : () to i128;
        %10, %11, %12 = builtin.unrealized_conversion_cast %2, %4 : (i64, i8) to (i1, i16, i32);
        %13 = builtin.unrealized_conversion_cast %0 : i32 to (() -> i32) attributes { note = \
                         #builtin.i32<7> };";
    const PARAMS: &str = "%0: i32, %1: i128, %2: i64, %3: i64, %4: i8";

    /// The four forms, a two-to-three cast and a cast to a function type print as written, and
    /// round trip in the text and the generic form; the generic form states the result types and
    /// no property.
    #[test]
    fn every_form_round_trips_in_text_and_generic_form() {
        let source = module(PARAMS, FORMS);
        let (_context, module) = parse_text(&source).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(module.borrow().to_string(), source);

        let found = casts(module);
        let shapes = found
            .iter()
            .map(|op| (op.borrow().num_operands(), op.borrow().num_results()))
            .collect::<Vec<_>>();
        assert_eq!(shapes, [(1, 1), (1, 2), (2, 1), (0, 1), (2, 3), (1, 1)]);
        assert_eq!(
            print_generic(&found[1].borrow()),
            "%6, %7 = \"builtin.unrealized_conversion_cast\"(%1) : i128 -> (i64, i64);"
        );
        assert_eq!(
            print_generic(&found[3].borrow()),
            "%9 = \"builtin.unrealized_conversion_cast\"() : () -> i128;"
        );

        let mut skipped = Vec::new();
        let failures = round_trip::<Module>(module, true, &mut skipped);
        assert!(failures.is_empty(), "{}", failures.join("\n\n"));
        assert!(skipped.is_empty(), "skipped: {skipped:#?}");
    }

    /// The builder makes the results from the types it is given, and the operation built prints
    /// in the form the parser reads.
    #[test]
    fn the_builder_gives_the_result_types() {
        let mut test = Test::new("f", |cx| ([cx.i128()], []));
        let (i64, i128) = (test.context().i64(), test.context().i128());
        let arg = test.entry_block().borrow().arguments()[0] as ValueRef;
        let mut builder = test.function_builder();
        let split = builder
            .unrealized_conversion_cast([arg], [i64, i64], SourceSpan::UNKNOWN)
            .unwrap();
        let halves = split
            .borrow()
            .outputs()
            .iter()
            .map(|r| r.borrow().as_value_ref())
            .collect::<Vec<_>>();
        let join = builder.unrealized_conversion_cast(halves, [i128], SourceSpan::UNKNOWN).unwrap();
        assert_eq!(
            split.borrow().as_operation().to_string(),
            "%1, %2 = builtin.unrealized_conversion_cast %0 : i128 to (i64, i64);"
        );
        assert_eq!(
            join.borrow().as_operation().to_string(),
            "%3 = builtin.unrealized_conversion_cast %1, %2 : (i64, i64) to i128;"
        );
        assert_eq!(join.borrow().inputs().len(), 2);
        assert!(split.borrow().as_operation().verify().is_ok());

        let err = builder
            .unrealized_conversion_cast([arg], [], SourceSpan::UNKNOWN)
            .map(|_| ())
            .unwrap_err();
        assert!(describe(&err).contains("a cast has at least one result"), "{}", describe(&err));
    }

    /// A cast with no result is rejected by the verifier; input types written that are not the
    /// operands' types are the parser's diagnostic, naming both.
    #[test]
    fn a_cast_without_a_result_and_misstated_input_types_are_errors() {
        let generic = "        \"builtin.unrealized_conversion_cast\"(%0) : (i32) -> ();";
        let err = parse_text(&module("%0: i32", generic)).map(|_| ()).unwrap_err();
        assert!(err.contains("invalid operation builtin.unrealized_conversion_cast"), "{err}");
        assert!(err.contains("a cast has at least one result"), "{err}");

        let text = "        %1 = builtin.unrealized_conversion_cast %0 : i64 to i32;";
        let err = parse_text(&module("%0: i32", text)).map(|_| ()).unwrap_err();
        assert!(err.contains("use here expects i64"), "{err}");
        assert!(err.contains("previous use here expects i32"), "{err}");

        let text = "        %2 = builtin.unrealized_conversion_cast %0, %1 : i32 to i64;";
        let err = parse_text(&module("%0: i32, %1: i32", text)).map(|_| ()).unwrap_err();
        assert!(err.contains("there are 2 operands, but 1 types"), "{err}");
    }

    /// The cast has no effects: dead-code elimination removes an unused one and keeps a used one.
    #[test]
    fn an_unused_cast_is_dead_code() {
        use crate::{
            Region,
            patterns::{NoopRewriterListener, RewriterImpl},
        };

        let mut test = Test::new("f", |cx| ([cx.i32()], [cx.i64()]));
        let (i64, i8) = (test.context().i64(), test.context().i8());
        let arg = test.entry_block().borrow().arguments()[0] as ValueRef;
        {
            let mut builder = test.function_builder();
            builder
                .unrealized_conversion_cast([arg], [i8, i8], SourceSpan::UNKNOWN)
                .unwrap();
            let used =
                builder.unrealized_conversion_cast([arg], [i64], SourceSpan::UNKNOWN).unwrap();
            let used = used.borrow().outputs()[0] as ValueRef;
            builder.ret([used], SourceSpan::UNKNOWN).unwrap();
        }
        let function = test.function().as_operation_ref();
        assert!(casts(function).iter().all(|op| op.borrow().is_memory_effect_free()));
        let region = test.function().borrow().body().as_region_ref();
        let mut rewriter = RewriterImpl::<NoopRewriterListener>::new(test.context_rc());
        Region::dead_code_elimination(&[region], &mut rewriter).unwrap();
        let left = casts(function);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].borrow().num_results(), 1);
    }

    /// One input and one result of the same type fold to the input; nothing else folds.
    #[test]
    fn a_cast_to_its_own_type_folds_to_its_input() {
        let source = module(
            "%0: i32, %1: i64",
            "        %2 = builtin.unrealized_conversion_cast %0 : i32 to i32;
        %3 = builtin.unrealized_conversion_cast %0 : i32 to i64;
        %4, %5 = builtin.unrealized_conversion_cast %0, %1 : (i32, i64) to (i32, i64);
        %6 = builtin.unrealized_conversion_cast : () to i32;",
        );
        let (_context, module) = parse_text(&source).unwrap_or_else(|err| panic!("{err}"));
        let found = casts(module);
        let folds = found
            .iter()
            .map(|op| {
                let mut results = SmallVec::<[OpFoldResult; 1]>::default();
                match op.borrow().fold(&mut results) {
                    FoldResult::Ok(()) => match results.as_slice() {
                        [OpFoldResult::Value(value)] => {
                            Some(*value == op.borrow().operands()[0].borrow().as_value_ref())
                        }
                        _ => panic!("a cast folds to a value"),
                    },
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(folds, [Some(true), None, None, None]);
    }

    /// The input's range is the result's for one input and one result of one width; a result of
    /// another width has the range of its type, and any other cast gives its results no range.
    #[test]
    fn the_range_of_a_cast() {
        use crate::ConstantIntRange;

        let source = module(
            "%0: i32, %1: i64",
            "        %2 = builtin.unrealized_conversion_cast %0 : i32 to i32;
        %3 = builtin.unrealized_conversion_cast %0 : i32 to i64;
        %4, %5 = builtin.unrealized_conversion_cast %1 : i64 to (i32, i32);
        %6 = builtin.unrealized_conversion_cast %0, %0 : (i32, i32) to i64;",
        );
        let (context, module) = parse_text(&source).unwrap_or_else(|err| panic!("{err}"));
        let seven = IntegerValueRange::new(ConstantIntRange::from_unsigned(32, 7, 7));
        let wide = IntegerValueRange::new(ConstantIntRange::from_unsigned(64, 7, 7));
        let ranges = casts(module)
            .iter()
            .map(|op| {
                let op = op.borrow();
                let cast = op.downcast_ref::<UnrealizedConversionCast>().unwrap();
                let operands = match op.num_operands() {
                    1 if op.operands()[0].borrow().ty() == context.i64() => {
                        alloc::vec![wide.clone()]
                    }
                    n => alloc::vec![seven.clone(); n],
                };
                let mut set = Vec::new();
                cast.infer_result_ranges(&operands, &mut |value, range| set.push((value, range)));
                set.into_iter().map(|(_, range)| range).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let top = IntegerValueRange::new(ConstantIntRange::max_for_type(context.i64()));
        assert_eq!(ranges, [alloc::vec![seven.clone()], alloc::vec![top], Vec::new(), Vec::new()]);
    }
}
