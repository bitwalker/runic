use alloc::format;

use crate::{
    AsValueRange, CallableOpInterface, Context, Op, OpParser, OpPrinter, OperationState,
    RegionBranchOpInterface, Report, Spanned, UnsafeIntrusiveEntityRef,
    derive::{EffectOpInterface, operation},
    diagnostics::report,
    dialects::builtin::BuiltinDialect,
    effects::{MemoryEffect, MemoryEffectOpInterface},
    parse::ParseResult,
    print::AsmPrinter,
    traits::{AnyType, Terminator},
};

pub type YieldRef = UnsafeIntrusiveEntityRef<Yield>;

/// The end of a region whose value its parent operation takes: `builtin.yield %v;`.
///
/// The operation holds the region, such as the initializer of a
/// [GlobalVariable](super::GlobalVariable), and what the region yields is what it stands for: the
/// value of the global. The operand is a value of any type, defined above the yield; the text does
/// not write its type. An `attributes {..}` dictionary may follow the operand.
///
/// The yield is a terminator that has no results and no effects. It states no successor and
/// leaves nothing: it is not a return, and it is not a terminator of a structured operation, so a
/// pass that looks for the exits of a function or the successors of a region does not see it. Its
/// use of the value keeps the value, and what computes it, alive.
///
/// Under an operation that is a function ([CallableOpInterface]) or that branches between its own
/// regions ([RegionBranchOpInterface]) the yield would state no successor, and the verifier
/// rejects it: those operations have terminators of their own.
#[derive(EffectOpInterface)]
#[effects(MemoryEffect())]
#[operation(
    dialect = BuiltinDialect,
    traits(Terminator),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct Yield {
    /// The value the region yields
    #[operand]
    value: AnyType,
}

impl OpPrinter for Yield {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_space();
        printer.print_value_uses(self.as_operation().operands().group(0).as_value_range());
        let op = self.as_operation();
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

/// Reads `%value`, then an optional `attributes {..}`.
impl OpParser for Yield {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let value = parser.parse_operand(/*allow_result_number=*/ true)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let unknown = parser.context().unknown_type();
        let value = parser.resolve_operand(value, unknown)?;
        state.add_operand(value);
        Ok(())
    }
}

/// The parent is not a function and does not branch between its own regions: a function is left
/// with `builtin.ret`, and a structured operation with a terminator of its own.
impl crate::Verify<dyn MemoryEffectOpInterface> for Yield {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let Some(parent) = op.parent_op() else {
            return Ok(());
        };
        let parent = parent.borrow();
        let reason = if parent.implements::<dyn CallableOpInterface>() {
            format!(
                "`{}` is a function, which is left with `builtin.ret`, not with `builtin.yield`",
                parent.name()
            )
        } else if parent.implements::<dyn RegionBranchOpInterface>() {
            format!(
                "`{}` is a structured operation, which is left with its own terminator, not with \
                 `builtin.yield`",
                parent.name()
            )
        } else {
            return Ok(());
        };
        Err(report!(
            message: "invalid operation builtin.yield",
            labels: [primary(op.span(), reason)],
        ))
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        rc::Rc,
        string::{String, ToString},
        vec::Vec,
    };

    use super::*;
    use crate::{
        Builder, Ident, Operation, OperationRef, SourceSpan, Visibility,
        dialects::builtin::{BuiltinOpBuilder, Module},
        parse::{ParserConfig, parse_any},
        testing::{
            Test,
            text::{describe, print_generic, round_trip},
        },
    };

    fn parse_text(source: &str) -> Result<(Rc<Context>, OperationRef), String> {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context.clone());
        match parse_any(config, "yield", source).into_result() {
            Ok(op) => Ok((context, op)),
            Err(report) => Err(describe(&report)),
        }
    }

    const MODULE: &str = "builtin.module public @m {
    builtin.global_variable public @data : array<i8; 2> {
        %b = builtin.bytes \"0a0b\" : array<i8; 2>;
        builtin.yield %b;
    };
};";

    fn yields(module: &OperationRef) -> Vec<YieldRef> {
        let mut found = Vec::new();
        module.borrow().prewalk_all(|op: &Operation| {
            if let Some(op) = op.downcast_ref::<Yield>() {
                found.push(op.as_yield_ref());
            }
        });
        found
    }

    impl Yield {
        fn as_yield_ref(&self) -> YieldRef {
            unsafe { YieldRef::from_raw(self) }
        }
    }

    /// The yield prints as `builtin.yield %v;` without a type, and the module round trips in the
    /// custom and the generic form.
    #[test]
    fn a_yield_round_trips_in_both_forms() {
        let (_context, module) = parse_text(MODULE).unwrap_or_else(|err| panic!("{err}"));
        let printed = module.borrow().to_string();
        assert!(printed.contains("builtin.yield %0;") || printed.contains("builtin.yield %"));
        assert!(!printed.contains("builtin.yield %0 :"), "{printed}");
        let mut skipped = Vec::new();
        let failures = round_trip::<Module>(module, true, &mut skipped);
        assert!(failures.is_empty(), "{}", failures.join("\n\n"));
        assert!(skipped.is_empty(), "skipped: {skipped:#?}");
    }

    /// The yield is a terminator that is not a return: it has no results and no effects, and a
    /// function is not left by it.
    #[test]
    fn a_yield_is_a_terminator_that_leaves_no_callable_and_has_no_effects() {
        let (_context, module) = parse_text(MODULE).unwrap_or_else(|err| panic!("{err}"));
        let yields = yields(&module);
        assert_eq!(yields.len(), 1);
        let op = yields[0].borrow();
        let op = op.as_operation();
        assert!(op.implements::<dyn crate::traits::Terminator>());
        assert!(!op.implements::<dyn crate::traits::ReturnLike>());
        assert!(!op.leaves_callable());
        assert_eq!(op.num_results(), 0);
        assert_eq!(op.num_operands(), 1);
        assert!(op.is_memory_effect_free());
    }

    /// The attribute dictionary follows the operand, and the generic form states it.
    #[test]
    fn a_yield_keeps_its_attributes() {
        let source = MODULE.replace(
            "builtin.yield %b;",
            "builtin.yield %b attributes { note = #builtin.i32<1> };",
        );
        let (_context, module) = parse_text(&source).unwrap_or_else(|err| panic!("{err}"));
        let printed = module.borrow().to_string();
        assert!(printed.contains("attributes"), "{printed}");
        let yields = yields(&module);
        let generic = print_generic(yields[0].borrow().as_operation());
        assert!(generic.contains("note"), "{generic}");
    }

    /// A function is left with `builtin.ret`: a yield under one is rejected.
    #[test]
    fn a_yield_under_a_function_is_rejected() {
        let source = "builtin.module public @m {
    builtin.function public extern(\"C\") @f(%0: i32) -> i32 {
        builtin.yield %0;
    };
};";
        let err = parse_text(source).map(|_| ()).expect_err("a function does not yield");
        assert!(err.contains("invalid operation builtin.yield"), "{err}");
        assert!(
            err.contains(
                "`builtin.function` is a function, which is left with `builtin.ret`, not with \
                 `builtin.yield`"
            ),
            "{err}"
        );
    }

    /// A structured operation is left with its own terminator: a yield under an `test.guard` is rejected.
    #[test]
    fn a_yield_under_a_structured_operation_is_rejected() {
        let source = "builtin.module public @m {
    builtin.function public extern(\"C\") @f(%0: i1, %1: i32) {
        test.guard %0 {
            builtin.yield %1;
        } : (i1);
        builtin.ret;
    };
};";
        let err = parse_text(source)
            .map(|_| ())
            .expect_err("a structured operation does not yield");
        assert!(err.contains("invalid operation builtin.yield"), "{err}");
        assert!(
            err.contains("is a structured operation, which is left with its own terminator"),
            "{err}"
        );
    }

    /// The builder creates the yield the parser reads: the global built with a `builtin.bytes` and
    /// a yield of it prints as the text and verifies, and its initializer value is the bytes.
    #[test]
    fn the_builder_creates_what_the_parser_reads() {
        let mut test = Test::named("f").in_module("m");
        let module = test.module();
        let context = test.context_rc();
        let array = context.array_type(context.i8(), 2);
        let (global, bytes) = {
            let b = test.builder_mut();
            let body = module.borrow().body().entry_block_ref().unwrap();
            b.set_insertion_point_to_end(body);
            let global = b
                .create_global_variable(
                    Ident::with_empty_span("data".into()),
                    Visibility::Public,
                    array,
                )
                .unwrap();
            let region = global.borrow().initializer().as_region_ref();
            let block = b.create_block(region, None, &[]);
            b.set_insertion_point_to_end(block);
            let bytes = b.bytes(alloc::vec![0x0b_u8, 0x0a], SourceSpan::UNKNOWN).unwrap();
            let value = bytes.borrow().as_operation().results()[0].borrow().as_value_ref();
            b.yield_value(value, SourceSpan::UNKNOWN).unwrap();
            (global, value)
        };
        test.module().borrow().as_operation().recursively_verify().unwrap();
        let global = global.borrow();
        assert!(global.initializer_value() == Some(bytes));
        let text = global.as_operation().to_string();
        assert!(text.contains("builtin.yield %"), "{text}");
        assert!(!text.contains("builtin.yield %0 :"), "{text}");
    }
}
