//! The guaranteed tail calls: `builtin.tail_call` and `builtin.tail_call_indirect`.

use alloc::{format, rc::Rc, string::String};

use super::{
    ArgumentsAndSignature, function_type_of, parse_arguments_and_signature, print_attributes,
    verify_callee_signature, verify_callee_type,
};
use crate::{
    CallableOpInterface, Context, Op, OpOperandRange, OpParser, OpPrinter, Operation,
    OperationState, Report, SmallVec, Spanned, UnsafeIntrusiveEntityRef,
    derive::operation,
    diagnostics::report,
    dialects::builtin::{
        BuiltinDialect, Function,
        attributes::{Signature, SignatureAttr},
    },
    parse::ParseResult,
    patterns::RewritePatternSet,
    print::AsmPrinter,
    traits::{AnyFunction, AnyType, Canonicalizable, TailCallLike, Terminator},
};

/// The hook the verifiers of the two tail calls hang from. [TailCallLike] cannot carry them: the
/// rules need the operations' signatures, which the marker does not know.
trait TailCallVerification: Op {}

impl TailCallVerification for TailCall {}

impl TailCallVerification for TailCallIndirect {}

pub type TailCallRef = UnsafeIntrusiveEntityRef<TailCall>;

/// A guaranteed tail call of the callable named by `callee`, with the signature the call is made
/// with: `builtin.tail_call @callee(%a, %b) : extern("C") (i32, i32) -> i32;`.
///
/// The operation is a terminator: the enclosing function leaves by calling the callee, and
/// returns whatever the callee returns. The operation has no results of its own, so the
/// signature's declared results are what the enclosing function declares, and so is its calling
/// convention (the function's verifier checks both). The caller's frame does not exist while the
/// callee runs: storage the caller obtained for its own frame (a `mem.alloca`, a local) is dead
/// then, and passing a pointer to it is undefined behaviour. A backend that cannot guarantee the
/// tail call rejects the operation instead of emitting an ordinary call.
///
/// The operation has one argument per parameter of its signature, of the parameter's type. The
/// callee resolves, through any chain of [FunctionAlias](crate::dialects::builtin::FunctionAlias)es,
/// to a callable whose signature equals the operation's, as for [Call](super::Call). A callee
/// that names nothing is reported by the check every symbol use gets
/// ([Operation::verify_symbol_refs](crate::Operation::verify_symbol_refs)). The operation can
/// only appear in a [Function](crate::dialects::builtin::Function).
///
/// The operation implements neither [CallOpInterface](crate::CallOpInterface), nor
/// [InferTypeOpInterface](crate::traits::InferTypeOpInterface), nor a memory effect interface, and
/// is not [ReturnLike](crate::traits::ReturnLike): its effects are unknown, and analyses must not
/// treat it as a call site they understand. The analyses that follow calls match a call's results
/// to the callee's returned values, and a tail call has no results of its own; its operands are
/// arguments, not returned values.
///
/// MLIR has no such operation outside its `llvm` dialect, where a tail call is a flag on
/// `llvm.call` with rules about the `ret` that must follow it; a terminator needs none.
#[operation(
    dialect = BuiltinDialect,
    name = "tail_call",
    traits(Terminator, TailCallLike),
    implements(TailCallVerification, OpPrinter)
)]
pub struct TailCall {
    /// The callable called: a function, or an alias of one
    #[symbol(callable)]
    callee: crate::SymbolPath,
    /// The signature the call is made with
    #[attr(hidden)]
    signature: SignatureAttr,
    /// The arguments, one per parameter of the signature
    #[operands]
    arguments: AnyType,
}

/// The arguments agree with the signature, the operation has no results, it is in a function, and
/// the callee resolves, through any chain of aliases, to a callable whose signature equals the
/// operation's.
///
/// A callee that is not linked to a symbol has already been reported by
/// [Operation::verify_symbol_refs](crate::Operation::verify_symbol_refs), which runs before this.
impl crate::Verify<dyn TailCallVerification> for TailCall {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let signature = self.get_signature();
        let op = self.as_operation();
        verify_tail_call(op, &self.arguments(), &signature)?;
        verify_callee_signature(op, "tail call", self.callee().resolve_callable(), &signature)
    }
}

/// Prints `@callee(%a, ..) : signature`, then `attributes {..}` if there are any, as
/// [Call](super::Call) does.
impl OpPrinter for TailCall {
    /// The arguments take the types of the parameters of the written signature.
    fn states_operand_types(&self) -> bool {
        true
    }

    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::{AttrPrinter, formatter::const_text};

        printer.print_space();
        printer.print_symbol_path(self.callee().path());
        printer.print_operand_list(self.operands().group(0));
        *printer += const_text(" : ");
        self.signature().print(printer);
        print_attributes(printer, self.as_operation());
    }
}

/// Reads `@callee(%a, ..) : signature`, then an optional `attributes {..}`. The arguments take
/// their types from the parameters of the signature.
impl OpParser for TailCall {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let callee = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute("callee", callee);

        let ArgumentsAndSignature {
            arguments, params, ..
        } = parse_arguments_and_signature(state, parser)?;
        let mut values = SmallVec::default();
        parser.resolve_operands(state.span, &arguments, &params, &mut values)?;
        state.operands.push(values);
        Ok(())
    }
}

pub type TailCallIndirectRef = UnsafeIntrusiveEntityRef<TailCallIndirect>;

/// A guaranteed tail call of the function value `callee`, with the signature the call is made
/// with: `builtin.tail_call_indirect %f(%a, %b) : extern("C") (i32, i32) -> i32;`.
///
/// It is to [TailCall] what [CallIndirect](super::CallIndirect) is to [Call](super::Call): the
/// callee has the function type of the signature (its parameter and declared result types), the
/// set of callables the operation may reach is not known, and the semantics, the operation having
/// no results, and what the verifiers check of the enclosing function are those of [TailCall]. The
/// operation implements neither [CallOpInterface](crate::CallOpInterface) nor a memory effect
/// interface, for the reasons [TailCall] gives. When the callee is defined by a
/// [FunctionReference](super::FunctionReference) whose callable has the operation's signature,
/// canonicalization turns the operation into a [TailCall] of that symbol.
#[operation(
    dialect = BuiltinDialect,
    name = "tail_call_indirect",
    traits(Terminator, TailCallLike),
    implements(TailCallVerification, OpPrinter)
)]
pub struct TailCallIndirect {
    /// The function value called
    #[operand]
    callee: AnyFunction,
    /// The arguments, one per parameter of the signature
    #[operands]
    arguments: AnyType,
    /// The signature the call is made with
    #[attr(hidden)]
    signature: SignatureAttr,
}

/// A tail call of a [FunctionReference](super::FunctionReference) whose callable has the call's
/// signature becomes a [TailCall]
/// ([TailCallIndirectOfFunctionRef](crate::dialects::builtin::canonicalization::TailCallIndirectOfFunctionRef)).
impl Canonicalizable for TailCallIndirect {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(
            crate::dialects::builtin::canonicalization::TailCallIndirectOfFunctionRef::new(context),
        );
    }
}

/// The callee's function type has the parameter and result types of the signature, the arguments
/// agree with the signature, the operation has no results, and it is in a function.
impl crate::Verify<dyn TailCallVerification> for TailCallIndirect {
    fn verify(&self, context: &Context) -> Result<(), Report> {
        let signature = self.get_signature();
        let op = self.as_operation();
        verify_callee_type(op, "tail call", context, self.callee().ty(), &signature)?;
        verify_tail_call(op, &self.arguments(), &signature)
    }
}

/// Prints `%callee(%a, ..) : signature`, then `attributes {..}` if there are any, as
/// [CallIndirect](super::CallIndirect) does.
impl OpPrinter for TailCallIndirect {
    /// The callee takes the function type of the written signature, and the arguments the
    /// types of its parameters.
    fn states_operand_types(&self) -> bool {
        true
    }

    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::{AsValueRange, AttrPrinter, formatter::const_text};

        printer.print_space();
        printer.print_value_uses(self.operands().group(0).as_value_range());
        printer.print_operand_list(self.operands().group(1));
        *printer += const_text(" : ");
        self.signature().print(printer);
        print_attributes(printer, self.as_operation());
    }
}

/// Reads `%callee(%a, ..) : signature`, then an optional `attributes {..}`. The callee takes the
/// function type of the signature, and the arguments the types of its parameters.
impl OpParser for TailCallIndirect {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let callee = parser.parse_operand(false)?;
        let ArgumentsAndSignature {
            arguments,
            signature,
            params,
        } = parse_arguments_and_signature(state, parser)?;
        let callee_ty = function_type_of(parser.context(), &signature);

        let callee = parser.resolve_operand(callee, callee_ty)?;
        state.operands.push(SmallVec::from_iter([callee]));
        let mut values = SmallVec::default();
        parser.resolve_operands(state.span, &arguments, &params, &mut values)?;
        state.operands.push(values);
        Ok(())
    }
}

/// The rules the two tail calls share: one argument per parameter of `signature`, of the
/// parameter's type, no results, and a `builtin.function` as the nearest enclosing callable.
fn verify_tail_call(
    op: &Operation,
    arguments: &OpOperandRange<'_>,
    signature: &Signature,
) -> Result<(), Report> {
    let span = op.span();
    let invalid = |label: alloc::string::String| {
        report!(
            message: (format!("invalid operation {}", op.name())),
            labels: [primary(span, label)],
        )
    };

    crate::conversion::verify_call_signature_arguments(op, arguments, signature, "tail call")?;
    if op.num_results() != 0 {
        return Err(invalid(format!(
            "a tail call has no results, because the function returns what the callee returns, \
             but this one has {}",
            op.num_results()
        )));
    }

    let mut parent = op.parent_op();
    while let Some(enclosing) = parent {
        if enclosing.borrow().implements::<dyn CallableOpInterface>() {
            if enclosing.borrow().is::<Function>() {
                return Ok(());
            }
            return Err(invalid(format!(
                "a tail call can only appear in a function, but this one is in `{}`",
                enclosing.borrow().name()
            )));
        }
        parent = enclosing.parent_op();
    }
    Err(invalid(String::from("a tail call can only appear in a function")))
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, string::ToString, vec::Vec};

    use super::*;
    use crate::{
        CallOpInterface, SourceSpan, Usable, ValueRef,
        dialects::builtin::{BuiltinOpBuilder, FunctionRef, Module},
        parse::{ParserConfig, parse_any},
        testing::{
            Test,
            text::{describe, round_trip},
        },
        traits::ReturnLike,
    };

    /// `@caller(i32, i32) -> i32` beside `@callee(i32, i32) -> i32`, and its two arguments.
    fn caller_and_callee() -> (Test, FunctionRef, [ValueRef; 2]) {
        let mut test = Test::named("caller").in_module("calls");
        let i32 = test.context().i32();
        let callee = test.define_function("callee", &[i32, i32], &[i32]);
        test.with_function("caller", &[i32, i32], &[i32]);
        let entry = test.entry_block();
        let args = {
            let entry = entry.borrow();
            [entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef]
        };
        (test, callee, args)
    }

    fn verify(test: &Test) -> Result<(), Report> {
        test.module().borrow().as_operation().recursively_verify()
    }

    #[test]
    fn the_builder_makes_a_terminator_with_no_results() {
        let (mut test, callee, [a, b]) = caller_and_callee();
        let signature = callee.borrow().get_signature().clone();
        let call = {
            let mut builder = test.function_builder();
            builder
                .tail_call(callee, signature.clone(), [a, b], SourceSpan::UNKNOWN)
                .unwrap()
        };
        verify(&test).unwrap();

        let call = call.borrow();
        assert_eq!(*call.get_signature(), signature);
        assert_eq!(call.callee().path().to_string(), "::calls::callee");
        assert_eq!(call.as_operation().num_results(), 0);
        let passed = call
            .arguments()
            .iter()
            .map(|operand| operand.borrow().as_value_ref())
            .collect::<Vec<_>>();
        assert_eq!(passed, [a, b]);
        let op = call.as_operation();
        assert!(op.implements::<dyn Terminator>());
        assert!(op.implements::<dyn TailCallLike>());
        assert!(!op.implements::<dyn ReturnLike>());
        assert!(!op.implements::<dyn CallOpInterface>());
        assert!(!op.is_memory_effect_free());
    }

    #[test]
    fn the_indirect_builder_makes_a_terminator_with_no_results() {
        let mut test = Test::named("caller").in_module("calls");
        let i32 = test.context().i32();
        let fn_ty = test.context().function_type([i32, i32], [i32]);
        test.with_function("caller", &[fn_ty, i32, i32], &[i32]);
        let entry = test.entry_block();
        let [f, a, b] = {
            let entry = entry.borrow();
            [0, 1, 2].map(|index| entry.arguments()[index] as ValueRef)
        };
        let signature = crate::dialects::builtin::attributes::Signature::new(
            &test.context_rc(),
            [i32, i32],
            [i32],
        );
        let call = {
            let mut builder = test.function_builder();
            builder.tail_call_indirect(f, signature, [a, b], SourceSpan::UNKNOWN).unwrap()
        };
        verify(&test).unwrap();
        let call = call.borrow();
        assert_eq!(call.as_operation().num_results(), 0);
        assert_eq!(call.callee().as_value_ref(), f);
        assert!(call.as_operation().implements::<dyn TailCallLike>());
        assert!(!call.as_operation().implements::<dyn CallOpInterface>());
    }

    /// The derived setter relinks the call to another callable.
    #[test]
    fn set_callee_relinks_the_tail_call() {
        let (mut test, callee, [a, b]) = caller_and_callee();
        let i32 = test.context().i32();
        let other = test.define_function("other", &[i32, i32], &[i32]);
        let signature = callee.borrow().get_signature().clone();
        let mut call = {
            let mut builder = test.function_builder();
            builder.tail_call(callee, signature, [a, b], SourceSpan::UNKNOWN).unwrap()
        };
        assert_eq!(callee.borrow().iter_uses().count(), 1);
        call.borrow_mut().set_callee(other).unwrap();
        verify(&test).unwrap();
        assert_eq!(callee.borrow().iter_uses().count(), 0);
        assert_eq!(other.borrow().iter_uses().count(), 1);
    }

    /// A module with callees, and `functions`.
    fn module_with(functions: &str) -> alloc::string::String {
        alloc::format!(
            r#"builtin.module public @m {{
    builtin.function public extern("C") @callee(i32, i32) -> i32;
    builtin.function_alias public @alias -> @callee;
    builtin.function public extern("C") @pair(i32) -> (i32, i64);
    builtin.function public extern("fast") @fast(i32, i32) -> i32;
    builtin.function public extern("C") @abort(i32) -> never;
    {functions}
}};"#
        )
    }

    /// `body` as the body of `@caller(%f: (i32, i32) -> i32, %a: i32, %b: i32, %w: i64, %g: (i32, i32) -> i64) -> i32`.
    fn in_caller(body: &str) -> alloc::string::String {
        module_with(&alloc::format!(
            r#"builtin.function public extern("C") @caller(%f: (i32, i32) -> i32, %a: i32, %b: i32, %w: i64, %g: (i32, i32) -> i64) -> i32 {{
        {body}
    }};"#
        ))
    }

    fn parse(source: &str, verify: bool) -> Result<(Rc<Context>, crate::OperationRef), String> {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context.clone()).verify_after_parse(verify);
        match parse_any(config, "tail_calls", source).into_result() {
            Ok(op) => Ok((context, op)),
            Err(report) => Err(describe(&report)),
        }
    }

    /// Why `source` does not verify.
    fn error_of(source: &str) -> String {
        parse(source, true).map(|_| ()).expect_err("the module must not verify")
    }

    /// Why a module with `body` as the body of `@caller` does not verify.
    fn body_error(body: &str) -> String {
        error_of(&in_caller(body))
    }

    /// The generic form of a direct tail call whose property `signature` and operand and result
    /// types are written as given.
    fn generic(name: &str, operands: &str, signature: &str, types: &str) -> String {
        let callee = if name == "builtin.tail_call" {
            "callee = #builtin.symbol<@callee>, "
        } else {
            ""
        };
        alloc::format!(
            r#""{name}"({operands}) <{{ {callee}signature = #builtin.signature<{signature}> }}> : {types};"#
        )
    }

    #[test]
    fn the_tail_calls_round_trip_in_both_forms() {
        let source = r#"builtin.module public @m {
    builtin.function public extern("C") @callee(i32, i32) -> i32;
    builtin.function public extern("C") @direct(%a: i32, %b: i32) -> i32 {
        builtin.tail_call @callee(%a, %b) : extern("C") (i32, i32) -> i32 attributes { note = #builtin.i32<7> };
    };
    builtin.function public extern("C") @indirect(%f: (i32, i32) -> i32, %a: i32, %b: i32) -> i32 {
        builtin.tail_call_indirect %f(%a, %b) : extern("C") (i32, i32) -> i32;
    };
};"#;
        let (_context, module) = parse(source, true).unwrap_or_else(|err| panic!("{err}"));
        let mut skipped = Vec::new();
        let failures = round_trip::<Module>(module, true, &mut skipped);
        assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n\n"));
        assert!(skipped.is_empty(), "skipped: {skipped:#?}");
        let text = module.borrow().to_string();
        assert!(
            text.contains(
                r#"builtin.tail_call @callee(%0, %1) : extern("C") (i32, i32) -> i32 attributes { note = #builtin.i32<7> };"#
            ),
            "{text}"
        );
        assert!(
            text.contains(
                r#"builtin.tail_call_indirect %2(%3, %4) : extern("C") (i32, i32) -> i32;"#
            ),
            "{text}"
        );
    }

    #[test]
    fn a_tail_call_is_accepted_through_an_alias() {
        parse(
            &in_caller(r#"builtin.tail_call @alias(%a, %b) : extern("C") (i32, i32) -> i32;"#),
            true,
        )
        .unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn a_function_that_never_returns_may_tail_call_a_callee_that_never_returns() {
        let source = module_with(
            r#"builtin.function public extern("C") @diverge(%a: i32) -> never {
        builtin.tail_call @abort(%a) : extern("C") (i32) -> never;
    };"#,
        );
        parse(&source, true).unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn an_argument_count_that_disagrees_with_the_signature_is_an_error() {
        let error = body_error(&generic(
            "builtin.tail_call",
            "%a",
            r#"extern("C") (i32, i32) -> i32"#,
            "(i32) -> ()",
        ));
        assert!(
            error.contains(
                "the tail call passes 1 argument(s), but its signature has 2 parameter(s)"
            ),
            "{error}"
        );
    }

    #[test]
    fn an_argument_type_that_disagrees_with_the_signature_is_an_error() {
        let error = body_error(&generic(
            "builtin.tail_call_indirect",
            "%f, %a, %w",
            r#"extern("C") (i32, i32) -> i32"#,
            "((i32, i32) -> i32 | i32, i64) -> ()",
        ));
        assert!(
            error.contains(
                "argument 1 has type `i64`, but parameter 1 of the signature has type `i32`"
            ),
            "{error}"
        );
    }

    #[test]
    fn a_callee_whose_signature_differs_is_an_error() {
        let error =
            body_error(r#"builtin.tail_call @callee(%a, %b) : extern("C") (i32, i32) -> i64;"#);
        assert!(
            error.contains(
                "the tail call's signature differs from its callee's: result 0 has type `i64` in \
                 the call, `i32` in the callee"
            ),
            "{error}"
        );
        assert!(
            error.contains("the callee is declared `extern(\\\"C\\\") (i32, i32) -> i32`"),
            "{error}"
        );
    }

    #[test]
    fn a_callee_that_names_nothing_is_an_error() {
        let error =
            body_error(r#"builtin.tail_call @missing(%a, %b) : extern("C") (i32, i32) -> i32;"#);
        assert!(error.contains("symbol '@missing' does not resolve"), "{error}");
    }

    #[test]
    fn an_indirect_callee_whose_type_differs_is_an_error() {
        let error = body_error(&generic(
            "builtin.tail_call_indirect",
            "%g, %a, %b",
            r#"extern("C") (i32, i32) -> i32"#,
            "((i32, i32) -> i64 | i32, i32) -> ()",
        ));
        assert!(
            error.contains(
                "the callee has type `(i32, i32) -> i64`, but the tail call's signature gives \
                 `(i32, i32) -> i32`"
            ),
            "{error}"
        );
    }

    #[test]
    fn a_result_given_in_the_generic_form_is_rejected() {
        for (name, operands, types) in [
            ("builtin.tail_call", "%a, %b", "(i32, i32) -> i32"),
            (
                "builtin.tail_call_indirect",
                "%f, %a, %b",
                "((i32, i32) -> i32 | i32, i32) -> i32",
            ),
        ] {
            let text = generic(name, operands, r#"extern("C") (i32, i32) -> i32"#, types);
            let error = body_error(&alloc::format!("%r = {text}"));
            assert!(error.contains("has 0 result(s), but this has 1"), "{name}: {error}");
        }
    }

    #[test]
    fn a_tail_call_outside_a_function_is_an_error() {
        let source = r#"builtin.module public @m {
    builtin.function public extern("C") @callee();
    "builtin.tail_call"() <{ callee = #builtin.symbol<@callee>, signature = #builtin.signature<extern("C") () -> ()> }> : () -> ();
};"#;
        let error = error_of(source);
        assert!(error.contains("a tail call can only appear in a function"), "{error}");
    }

    #[test]
    fn a_tail_call_whose_results_differ_from_the_functions_is_an_error() {
        let error = body_error(r#"builtin.tail_call @pair(%a) : extern("C") (i32) -> (i32, i64);"#);
        assert!(
            error.contains("this tail call returns (i32, i64), but the function returns (i32)"),
            "{error}"
        );
        assert!(error.contains("in this function"), "{error}");

        let error = error_of(&module_with(
            r#"builtin.function public extern("C") @short(%a: i32) -> (i32, i64) {
        builtin.tail_call @callee(%a, %a) : extern("C") (i32, i32) -> i32;
    };"#,
        ));
        assert!(
            error.contains("this tail call returns (i32), but the function returns (i32, i64)"),
            "{error}"
        );
    }

    #[test]
    fn a_returning_function_may_not_tail_call_a_callee_that_never_returns() {
        let error = body_error(r#"builtin.tail_call @abort(%a) : extern("C") (i32) -> never;"#);
        assert!(
            error.contains("this tail call returns (never), but the function returns (i32)"),
            "{error}"
        );
    }

    #[test]
    fn a_function_that_never_returns_may_not_tail_call_a_callee_that_returns() {
        let error = error_of(&module_with(
            r#"builtin.function public extern("C") @diverge(%a: i32) -> never {
        builtin.tail_call @callee(%a, %a) : extern("C") (i32, i32) -> i32;
    };"#,
        ));
        assert!(
            error.contains("this tail call returns (i32), but the function returns (never)"),
            "{error}"
        );
    }

    #[test]
    fn a_tail_call_with_another_calling_convention_than_the_functions_is_an_error() {
        let error =
            body_error(r#"builtin.tail_call @fast(%a, %b) : extern("fast") (i32, i32) -> i32;"#);
        assert!(
            error.contains(
                "this tail call uses the calling convention \\\"fast\\\", but the function uses \
                 \\\"C\\\""
            ),
            "{error}"
        );
        let error = body_error(
            r#"builtin.tail_call_indirect %f(%a, %b) : extern("fast") (i32, i32) -> i32;"#,
        );
        assert!(
            error.contains("the calling convention \\\"fast\\\", but the function uses \\\"C\\\""),
            "{error}"
        );
    }
}
