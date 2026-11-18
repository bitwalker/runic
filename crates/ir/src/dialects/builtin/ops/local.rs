use alloc::format;

use crate::{
    AsValueRange, Context, Op, OpParser, OpPrinter, Operation, OperationState, Report, Spanned,
    Type, UnsafeIntrusiveEntityRef, Value,
    derive::{EffectOpInterface, operation},
    diagnostics::report,
    dialects::builtin::{
        BuiltinDialect,
        attributes::{LocalVariable, LocalVariableAttr},
    },
    effects::{MemoryEffect, MemoryEffectOpInterface, Resource},
    parse::{ParseResult, ParserExt},
    print::AsmPrinter,
    traits::{AnyType, InferTypeOpInterface},
};

pub type LoadLocalRef = UnsafeIntrusiveEntityRef<LoadLocal>;
pub type StoreLocalRef = UnsafeIntrusiveEntityRef<StoreLocal>;

/// The local variables of a function, which `builtin.load_local` reads and `builtin.store_local`
/// writes: a child of the default resource, so an access to a local is ordered against every
/// effect on the default resource (an operation with unknown effects, say) and against no effect
/// on another of its children (memory, an assert).
///
/// It is one resource for every local: a write of one local is ordered against a read of another.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LocalsResource;

impl Resource for LocalsResource {
    fn name(&self) -> &'static str {
        "builtin.locals"
    }
}

/// A read of a local variable of the enclosing function: `%v = builtin.load_local 0 : i32;`.
///
/// The text gives the index of the local and its type, which is the type of the result. An
/// `attributes {..}` dictionary may follow. The generic form states the local as an attribute:
/// `<{ local = #builtin.local_variable<0, i32> }>`.
///
/// The result has the local's type. A local is not memory: it has no address and needs no data
/// layout, so it may hold a value of any type. The load reads `local` in [LocalsResource], and has
/// no other effect: an unused load is removed, and two loads of one local with no write of a local
/// between them are merged.
///
/// A load of a local that has not been stored yields an unspecified value; `Local2Reg` does not
/// promote such a load. A frontend whose source language zero-initializes its locals stores a
/// zero first.
///
/// Inside a function, the local is one of the function's `locals(..)`, of the type the function
/// gives it.
#[derive(EffectOpInterface)]
#[operation(
    dialect = BuiltinDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct LoadLocal {
    /// The local read
    #[attr]
    #[effects(MemoryEffect(MemoryEffect::Read), resource(LocalsResource))]
    local: LocalVariableAttr,
    /// The value read
    #[result]
    result: AnyType,
}

impl OpPrinter for LoadLocal {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::const_text;

        let local = *self.get_local();
        printer.print_space();
        printer.print_decimal_integer(local.index());
        *printer += const_text(" : ");
        printer.print_type(local.ty());
        print_attributes(printer, self.as_operation());
    }
}

/// Reads `INDEX : type`, then an optional `attributes {..}`. The index is a decimal integer that
/// fits a `u32`; the type is the local's type and the result's.
impl OpParser for LoadLocal {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let index = parser.parse_decimal_integer::<u32>()?.into_inner();
        let ty = parser.parse_colon_type()?.into_inner();
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let local = local_attribute(parser, LocalVariable::new(index, ty));
        state.add_attribute("local", local);
        state.results.push(ty);
        Ok(())
    }
}

/// The attribute of `local`.
fn local_attribute(
    parser: &mut dyn crate::OpAsmParser<'_>,
    local: LocalVariable,
) -> crate::AttributeRef {
    parser
        .context_rc()
        .create_attribute::<LocalVariableAttr, _>(local)
        .as_attribute_ref()
}

/// Prints the `attributes {..}` dictionary of `op`, if it has attributes.
fn print_attributes(printer: &mut AsmPrinter<'_>, op: &Operation) {
    if op.has_attributes() {
        printer.print_space();
        printer.print_keyword("attributes");
        printer.print_space();
        printer.print_attribute_dictionary(
            op.attributes().iter().map(|attr| *attr.as_named_attribute()),
        );
    }
}

/// A result whose type is not yet known gets the local's type. A result that already has a type
/// keeps it, and the verifier checks it.
impl InferTypeOpInterface for LoadLocal {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        if !self.result().ty().is_unknown() {
            return Ok(());
        }
        let ty = self.get_local().ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// The result has the local's type. That the local is one of the enclosing function's, of the
/// type the function gives it, is checked by the function's verifier
/// ([Function](crate::dialects::builtin::Function)).
impl crate::Verify<dyn MemoryEffectOpInterface> for LoadLocal {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let ty = self.result().ty();
        verify_local(self.as_operation(), *self.get_local(), ty, "the result")
    }
}

/// A write of `value` to a local variable of the enclosing function: `builtin.store_local 0, %v;`.
///
/// The text gives the index of the local and the value; the local's type is the value's type. An
/// `attributes {..}` dictionary may follow. The generic form states the local as an attribute:
/// `<{ local = #builtin.local_variable<0, i32> }>`.
///
/// The value has the local's type. The store writes `local` in [LocalsResource], so it is never
/// removed by a generic pass, and no read of a local moves or merges across it.
///
/// Inside a function, the local is one of the function's `locals(..)`, of the type the function
/// gives it.
#[derive(EffectOpInterface)]
#[operation(
    dialect = BuiltinDialect,
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct StoreLocal {
    /// The local written
    #[attr]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(LocalsResource))]
    local: LocalVariableAttr,
    /// The value written
    #[operand]
    value: AnyType,
}

impl OpPrinter for StoreLocal {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::const_text;

        printer.print_space();
        printer.print_decimal_integer(self.get_local().index());
        *printer += const_text(", ");
        printer.print_value_uses(self.as_operation().operands().group(0).as_value_range());
        print_attributes(printer, self.as_operation());
    }
}

/// Reads `INDEX, %value`, then an optional `attributes {..}`. The index is a decimal integer that
/// fits a `u32`; the local's type is the value's.
impl OpParser for StoreLocal {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let index = parser.parse_decimal_integer::<u32>()?.into_inner();
        parser.parse_comma()?;
        let value = parser.parse_operand(/*allow_result_number=*/ true)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let unknown = parser.context().unknown_type();
        let value = parser.resolve_operand(value, unknown)?;
        let ty = value.borrow().ty();
        let local = local_attribute(parser, LocalVariable::new(index, ty));
        state.add_attribute("local", local);
        state.add_operand(value);
        Ok(())
    }
}

/// The value has the local's type. That the local is one of the enclosing function's, of the
/// type the function gives it, is checked by the function's verifier
/// ([Function](crate::dialects::builtin::Function)).
impl crate::Verify<dyn MemoryEffectOpInterface> for StoreLocal {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let ty = self.value().ty();
        verify_local(self.as_operation(), *self.get_local(), ty, "the value")
    }
}

/// Checks that the value of `op` that is `role` (its result or its operand), of type `ty`, has
/// the type of `local`.
fn verify_local(op: &Operation, local: LocalVariable, ty: Type, role: &str) -> Result<(), Report> {
    if ty == local.ty() {
        return Ok(());
    }
    let reason =
        format!("{role} has type `{ty}`, but local {} has type `{}`", local.index(), local.ty());
    Err(report!(
        message: (format!("invalid operation {}", op.name())),
        labels: [primary(op.span(), reason)],
    ))
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, string::ToString, vec::Vec};

    use super::*;
    use crate::{
        Context, SourceSpan, UnsafeIntrusiveEntityRef, ValueRef,
        dialects::{
            builtin::{BuiltinOpBuilder, Function},
            test::TestMemoryResource,
        },
        effects::{AutomaticAllocationScopeResource, DefaultResource},
        parse::{self, ParserConfig},
        testing::{Test, label_messages, text::parse as parse_text},
    };

    const HEADER: &str = "builtin.function public extern(\"C\")";

    /// Parses `source` (a function) without verifying it, then verifies it: the error, as one line.
    fn verify_error(source: &str) -> alloc::string::String {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context.clone()).verify_after_parse(false);
        let function = parse::parse::<Function>(config, "locals", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{source}\n{err:?}"));
        let function = function.borrow();
        let err = function
            .as_operation()
            .recursively_verify()
            .expect_err("the function does not verify");
        let labels = label_messages(err.as_diagnostic()).join(" | ");
        let text = alloc::format!("{} {labels}", err.display());
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Parses `source` (a function), which must not parse: the text each label of the error covers.
    fn error_spans(source: &str) -> Vec<alloc::string::String> {
        use crate::diagnostics::{Diagnostic, Label, VisitDiagnostic};

        struct Spans<'a>(&'a str, Vec<alloc::string::String>);
        impl VisitDiagnostic for Spans<'_> {
            fn label(&mut self, label: Label<'_>) {
                let range = label.span.range().into_slice_index();
                self.1.push(self.0.get(range).unwrap_or("<out of range>").into());
            }
        }

        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context);
        let Err(report) = parse::parse::<Function>(config, "locals", source).into_result() else {
            return Vec::new();
        };
        let mut spans = Spans(source, Vec::new());
        let diagnostic: &dyn Diagnostic = report.as_diagnostic();
        diagnostic.visit(&mut spans);
        spans.1
    }

    /// A function built with a local of type `i32`, stored from its argument and loaded back.
    fn built() -> (Test, LocalVariable, UnsafeIntrusiveEntityRef<StoreLocal>, ValueRef) {
        let mut test = Test::new("locals", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();
        let (local, store, loaded) = {
            let mut builder = test.function_builder();
            let local = builder.alloc_local(i32);
            let argument = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let store = builder.store_local(local, argument, SourceSpan::UNKNOWN).unwrap();
            let loaded = builder.load_local(local, SourceSpan::UNKNOWN).unwrap();
            builder.ret([loaded], SourceSpan::UNKNOWN).unwrap();
            (local, store, loaded)
        };
        (test, local, store, loaded)
    }

    /// A load and a store built through the builder print in the derived form, and the function
    /// parses back to the same text: the parsed local has the index and the type written.
    #[test]
    fn locals_print_and_parse_back() {
        let (test, _local, _store, loaded) = built();
        assert_eq!(loaded.borrow().ty(), test.context().i32());
        let function = test.function();
        let printed = function.borrow().as_operation().to_string();
        let expected = alloc::format!(
            "{HEADER} @locals(%0: i32) -> i32 locals(i32) {{
    builtin.store_local 0, %0;
    %1 = builtin.load_local 0 : i32;
    builtin.ret %1 : (i32);
}};"
        );
        assert_eq!(printed, expected);
        function
            .borrow()
            .as_operation()
            .recursively_verify()
            .expect("the built function verifies");

        let (_context, parsed) = parse_text::<Function>(&printed).expect("the text parses");
        assert_eq!(parsed.borrow().as_operation().to_string(), printed);
        let parsed = parsed.borrow();
        let block = parsed.entry_block();
        let block = block.borrow();
        let load = block.body().iter().nth(1).unwrap();
        let load = load.downcast_ref::<LoadLocal>().expect("a load_local");
        assert_eq!(load.get_local().index(), 0);
        assert_eq!(load.get_local().ty(), parsed.as_operation().context().i32());
        assert_eq!(load.result().ty(), parsed.as_operation().context().i32());
    }

    /// A local may hold a value of a type that has no data layout: a function value.
    #[test]
    fn a_local_may_hold_a_function_value() {
        let source = alloc::format!(
            "{HEADER} @f(%0: (i32) -> i32) -> ((i32) -> i32) locals((i32) -> i32) {{
    builtin.store_local 0, %0;
    %1 = builtin.load_local 0 : (i32) -> i32;
    builtin.ret %1 : ((i32) -> i32);
}};"
        );
        let (_context, function) = parse_text::<Function>(&source).expect("the text parses");
        assert_eq!(function.borrow().as_operation().to_string(), source);
    }

    /// A load reads, and a store writes, its local attribute in [LocalsResource], a child of the
    /// default resource disjoint from memory.
    #[test]
    fn locals_are_read_and_written_in_their_own_resource() {
        let (_test, _local, store, loaded) = built();
        let load = loaded.borrow().get_defining_op().unwrap();
        for (op, effect) in
            [(load, MemoryEffect::Read), (store.as_operation_ref(), MemoryEffect::Write)]
        {
            let op = op.borrow();
            let effects = op.as_trait::<dyn MemoryEffectOpInterface>().unwrap();
            assert!(effects.effects_are_known());
            let effects = effects.effects().collect::<Vec<_>>();
            assert_eq!(effects.len(), 1);
            let instance = &effects[0];
            assert_eq!(*instance.effect(), effect);
            assert!(instance.value().is_none());
            let attribute = instance.attribute().expect("the effect is on the local");
            assert!(attribute.borrow().downcast_ref::<LocalVariableAttr>().is_some());
            assert!(instance.resource().dyn_eq(&LocalsResource));
            assert_eq!(instance.resource().parent().unwrap().name(), DefaultResource.name());
            assert!(!instance.resource().is_disjoint_from(&DefaultResource));
            assert!(instance.resource().is_disjoint_from(&TestMemoryResource));
            assert!(instance.resource().is_disjoint_from(&AutomaticAllocationScopeResource));
        }
        assert!(!store.borrow().as_operation().would_be_trivially_dead());
    }

    /// An unused load is removable; a store is not.
    #[test]
    fn an_unused_load_is_trivially_dead() {
        let mut test = Test::new("dead", |cx| ([cx.i32()], [] as [Type; 0]));
        let i32 = test.context().i32();
        let mut builder = test.function_builder();
        let local = builder.alloc_local(i32);
        let loaded = builder.load_local(local, SourceSpan::UNKNOWN).unwrap();
        builder.ret([], SourceSpan::UNKNOWN).unwrap();
        let load = loaded.borrow().get_defining_op().unwrap();
        assert!(load.borrow().would_be_trivially_dead());
    }

    /// Inside a function, a local is one of the function's locals, of the type it gives it, and the
    /// value read or written has that type.
    #[test]
    fn a_local_that_is_not_the_functions_does_not_verify() {
        let load = |local: &str| alloc::format!("    %1 = builtin.load_local {local};");
        for (locals, body, reason) in [
            (
                " locals(i32)",
                load("1 : i32"),
                "this names local 1, but the function has 1 local(s)",
            ),
            ("", load("0 : i32"), "this names local 0, but the function has 0 local(s)"),
            (
                " locals(i32)",
                load("0 : i64"),
                "this states that local 0 has type `i64`, but the function gives it type `i32`",
            ),
            (
                " locals(i32)",
                "    \"builtin.store_local\"(%0) <{ local = #builtin.local_variable<0, i32> }> : \
                 (i64) -> ();"
                    .to_string(),
                "the value has type `i64`, but local 0 has type `i32`",
            ),
            (
                " locals(i32)",
                "    builtin.store_local 0, %0;".to_string(),
                "this states that local 0 has type `i64`, but the function gives it type `i32`",
            ),
        ] {
            let source =
                alloc::format!("{HEADER} @f(%0: i64){locals} {{\n{body}\n    builtin.ret;\n}};");
            let text = verify_error(&source);
            assert!(text.contains(reason), "{source}\n{text}");
        }
    }

    /// A load whose result (stated in the generic form) has another type than its local does not
    /// verify.
    #[test]
    fn a_load_of_another_type_than_its_local_does_not_verify() {
        let source = alloc::format!(
            "{HEADER} @f() locals(i32) {{
    %1 = \"builtin.load_local\"() <{{ local = #builtin.local_variable<0, i32> }}> : () -> i64;
    builtin.ret;
}};"
        );
        let text = verify_error(&source);
        assert!(text.contains("invalid operation builtin.load_local"), "{source}\n{text}");
        assert!(text.contains("the result has type `i64`, but local 0 has type `i32`"), "{text}");
    }

    /// Outside a function, a local is checked only against the type it states: a store of another
    /// type does not verify, and one of that type does.
    #[test]
    fn a_local_outside_a_function_is_checked_against_its_type() {
        let module = |ty: &str| {
            alloc::format!(
                "builtin.module public @m {{
    %c = test.constant 0 : i1;
    test.guard %c {{
        %0 = test.constant 0 : {ty};
        \"builtin.store_local\"(%0) <{{ local = #builtin.local_variable<0, i32> }}> : ({ty}) -> ();
        test.yield : ();
    }} : (i1);
}};"
            )
        };
        let err = parse_text::<crate::dialects::builtin::Module>(&module("i64"))
            .map(|_| ())
            .expect_err("the value is not of the local's type");
        assert!(err.contains("the value has type `i64`, but local 0 has type `i32`"), "{err}");
        parse_text::<crate::dialects::builtin::Module>(&module("i32"))
            .unwrap_or_else(|err| panic!("{err}"));
    }

    /// A load of a local that its function no longer has is built (its type is the local's), and
    /// the function does not verify.
    #[test]
    fn a_load_of_a_local_out_of_range_does_not_verify() {
        let mut test = Test::new("shrunk", |_cx| ([] as [Type; 0], [] as [Type; 0]));
        let i32 = test.context().i32();
        let local = test.function().borrow_mut().alloc_local(i32);
        test.function().borrow_mut().get_locals_mut().remove(0);
        {
            let mut builder = test.function_builder();
            let loaded = builder.load_local(local, SourceSpan::UNKNOWN).unwrap();
            assert_eq!(loaded.borrow().ty(), i32);
            builder.ret([], SourceSpan::UNKNOWN).unwrap();
        }
        let function = test.function();
        let function = function.borrow();
        let err = function
            .as_operation()
            .recursively_verify()
            .expect_err("the function has no local 0");
        let labels = label_messages(err.as_diagnostic()).join(" | ");
        assert!(
            labels.contains("this names local 0, but the function has 0 local(s)"),
            "{labels}"
        );
    }

    /// A local names an index and a type, not a function: one made by `alloc_local` on one
    /// function and used in another that has a local of that index and type verifies there, and
    /// with another type at that index it is rejected.
    #[test]
    fn a_local_made_by_one_function_is_any_functions_local_of_its_index_and_type() {
        for (second_ty, verifies) in [("i32", true), ("i64", false)] {
            let mut test = Test::named("first").in_module("m");
            let (i32, i64) = (test.context().i32(), test.context().i64());
            let module_body = test.module().borrow().body().entry_block_ref().unwrap();
            test.with_function("first", &[i32], &[]);
            let local = test.function().borrow_mut().alloc_local(i32);
            crate::Builder::set_insertion_point_to_end(test.builder_mut(), module_body);
            test.with_function("second", &[i32], &[]);
            let ty = if second_ty == "i32" { i32 } else { i64 };
            let second_local = test.function().borrow_mut().alloc_local(ty);
            assert_eq!(second_local.index(), local.index());
            {
                let mut builder = test.function_builder();
                let argument = builder.entry_block().borrow().arguments()[0] as ValueRef;
                builder.store_local(local, argument, SourceSpan::UNKNOWN).unwrap();
                builder.ret([], SourceSpan::UNKNOWN).unwrap();
            }
            let function = test.function();
            let function = function.borrow();
            let verified = function.as_operation().recursively_verify();
            if verifies {
                verified.unwrap_or_else(|err| panic!("{}", err.display()));
                continue;
            }
            let err = verified.expect_err("local 0 of `@second` has another type");
            let labels = label_messages(err.as_diagnostic()).join(" | ");
            assert!(
                labels.contains(
                    "this states that local 0 has type `i32`, but the function gives it type `i64`"
                ),
                "{labels}"
            );
        }
    }

    /// Both operations print the index first and, for a load, the local's type, for a local built
    /// through the function and one parsed from text alike.
    #[test]
    fn the_short_form_round_trips_for_a_built_and_a_parsed_local() {
        let (test, _local, _store, _loaded) = built();
        let function = test.function();
        let printed = function.borrow().as_operation().to_string();
        assert!(printed.contains("    builtin.store_local 0, %0;\n"), "{printed}");
        assert!(printed.contains("    %1 = builtin.load_local 0 : i32;\n"), "{printed}");

        let (_context, parsed) = parse_text::<Function>(&printed).expect("the text parses");
        let parsed = parsed.borrow();
        for op in parsed.entry_block().borrow().body().iter() {
            if let Some(load) = op.downcast_ref::<LoadLocal>() {
                assert_eq!(load.get_local().ty(), load.result().ty());
            }
            if let Some(store) = op.downcast_ref::<StoreLocal>() {
                assert_eq!(store.get_local().ty(), store.value().ty());
            }
        }
        assert_eq!(parsed.as_operation().to_string(), printed);
    }

    /// An attribute dictionary follows the short form, as it follows `builtin.call`.
    #[test]
    fn the_short_form_takes_an_attribute_dictionary() {
        let source = alloc::format!(
            "{HEADER} @f(%0: i32) locals(i32) {{
    builtin.store_local 0, %0 attributes {{ note = #builtin.i32<1> }};
    %1 = builtin.load_local 0 : i32 attributes {{ note = #builtin.i32<2> }};
    builtin.ret;
}};"
        );
        let (_context, function) = parse_text::<Function>(&source).expect("the text parses");
        assert_eq!(function.borrow().as_operation().to_string(), source);
    }

    /// The generic form of both operations states the local as an attribute; it parses to the same
    /// operations as the short form, and the same attribute.
    #[test]
    fn the_generic_form_parses_to_the_short_form() {
        let generic = alloc::format!(
            "{HEADER} @f(%0: i32) locals(i32) {{
    \"builtin.store_local\"(%0) <{{ local = #builtin.local_variable<0, i32> }}> : (i32) -> ();
    %1 = \"builtin.load_local\"() <{{ local = #builtin.local_variable<0, i32> }}> : () -> i32;
    builtin.ret;
}};"
        );
        let short = alloc::format!(
            "{HEADER} @f(%0: i32) locals(i32) {{
    builtin.store_local 0, %0;
    %1 = builtin.load_local 0 : i32;
    builtin.ret;
}};"
        );
        let (_context, function) = parse_text::<Function>(&generic).expect("the text parses");
        assert_eq!(function.borrow().as_operation().to_string(), short);
    }

    /// Text that is not `load_local INDEX : TYPE` or `store_local INDEX, VALUE` is a diagnostic: a
    /// missing, negative or too large index, a missing type or value.
    #[test]
    fn malformed_short_forms_are_diagnostics() {
        for (body, token) in [
            ("%1 = builtin.load_local : i32;", ":"),
            ("%1 = builtin.load_local 4294967296 : i32;", "4294967296"),
            ("%1 = builtin.load_local -1 : i32;", "-"),
            ("%1 = builtin.load_local 0;", ""),
            ("builtin.store_local 0;", ""),
            ("builtin.store_local %0;", "%0"),
        ] {
            let source = alloc::format!(
                "{HEADER} @f(%0: i32) locals(i32) {{\n    {body}\n    builtin.ret;\n}};"
            );
            let spans = error_spans(&source);
            assert!(!spans.is_empty(), "`{body}` does not parse");
            assert!(spans.iter().any(|span| span == token), "{body}\n{spans:?}");
        }
    }

    /// The `local` property of `op`.
    fn local_property(op: &Operation) -> crate::AttributeRef {
        op.properties()
            .find(|property| property.name.as_str() == "local")
            .expect("a `local` property")
            .value
    }

    /// A local from [Function::alloc_local], the same local parsed from text and one made with
    /// [LocalVariable::new] are equal, as values and as attributes.
    #[test]
    fn a_local_is_the_same_however_it_was_made() {
        let context = Rc::new(Context::default());
        let parse_function = |source: &str| {
            parse::parse::<Function>(ParserConfig::new(context.clone()), "locals", source)
                .into_result()
                .unwrap_or_else(|err| panic!("{source}\n{err:?}"))
        };
        let parsed = parse_function(&alloc::format!(
            "{HEADER} @f() locals(i32) {{
    %0 = builtin.load_local 0 : i32;
    builtin.ret;
}};"
        ));
        let mut other = parse_function(&alloc::format!("{HEADER} @g() {{\n    builtin.ret;\n}};"));
        let allocated = other.borrow_mut().alloc_local(context.i32());
        let made = LocalVariable::new(0, context.i32());

        let parsed = parsed.borrow();
        let entry = parsed.entry_block();
        let entry = entry.borrow();
        let load = entry.body().front().get().expect("a load");
        let parsed_attr = local_property(&load);
        let parsed_local =
            *parsed_attr.borrow().downcast_ref::<LocalVariableAttr>().unwrap().as_value();

        assert_eq!(parsed_local, made);
        assert_eq!(allocated, made);
        assert_eq!(other.borrow().iter_locals().collect::<Vec<_>>(), [made]);
        assert_eq!(other.borrow().get_local(&made), Some(context.i32()));
        for local in [allocated, made] {
            let attr = context.create_attribute::<LocalVariableAttr, _>(local).as_attribute_ref();
            assert!(attr == parsed_attr, "{local:?}");
            assert!(crate::Attribute::ty(&*attr.borrow()).is_unknown());
        }
        let other_type = context
            .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(0, context.i64()))
            .as_attribute_ref();
        let other_index = context
            .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(1, context.i32()))
            .as_attribute_ref();
        assert!(other_type != parsed_attr && other_index != parsed_attr);
    }

    /// A local past the end of the function's locals, and one of another type than the function
    /// gives it, are rejected wherever an operation holds it: in `builtin.load_local`, in
    /// `builtin.store_local`, in the property of another operation, and nested in a list in an
    /// operation's attributes. The error names the operation that holds the local.
    #[test]
    fn a_local_that_is_not_the_functions_is_rejected_wherever_it_is_held() {
        use crate::{
            attributes::AttrList,
            dialects::builtin::attributes::{List, ListAttr},
        };

        for (index, ty, reason) in [
            (1, "i32", "this names local 1, but the function has 1 local(s)"),
            (
                0,
                "i64",
                "this states that local 0 has type `i64`, but the function gives it type `i32`",
            ),
        ] {
            let value = if ty == "i32" { "%0" } else { "%1" };
            for (name, body) in [
                ("builtin.load_local", alloc::format!("%2 = builtin.load_local {index} : {ty};")),
                ("builtin.store_local", alloc::format!("builtin.store_local {index}, {value};")),
                (
                    "test.touch_local",
                    alloc::format!(
                        "test.touch_local <{{ local = #builtin.local_variable<{index}, {ty}> }}>;"
                    ),
                ),
            ] {
                let source = alloc::format!(
                    "{HEADER} @f(%0: i32, %1: i64) locals(i32) {{\n    {body}\n    \
                     builtin.ret;\n}};"
                );
                let text = verify_error(&source);
                assert!(text.contains("invalid operation builtin.function"), "{text}");
                assert!(text.contains(&alloc::format!("`{name}` names a local")), "{text}");
                assert!(text.contains(reason), "{source}\n{text}");
            }

            // nested in a list in the dictionary of the `builtin.ret`
            let mut test = Test::new("listed", |_cx| ([] as [Type; 0], [] as [Type; 0]));
            let context = test.context_rc();
            let local_ty = if ty == "i32" {
                context.i32()
            } else {
                context.i64()
            };
            {
                let mut builder = test.function_builder();
                builder.alloc_local(context.i32());
                let mut ret = builder.ret([], SourceSpan::UNKNOWN).unwrap();
                let local = context
                    .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(index, local_ty));
                let mut entries = AttrList::new();
                entries.push_back(crate::Attribute::as_attr(&*local.borrow()).as_attr_ref());
                let list = context.create_attribute::<ListAttr, _>(List::from(entries));
                ret.borrow_mut().as_operation_mut().set_attribute("locals", list);
            }
            let function = test.function();
            let err = function
                .borrow()
                .as_operation()
                .recursively_verify()
                .expect_err("the listed local is not the function's");
            let text = alloc::format!(
                "{} {}",
                err.display(),
                label_messages(err.as_diagnostic()).join(" | ")
            );
            assert!(text.contains("invalid operation builtin.function"), "{text}");
            assert!(text.contains("`builtin.ret` names a local"), "{text}");
            assert!(text.contains(reason), "{text}");
        }
    }

    /// Outside a function a local is not checked against any function's locals: an operation
    /// that names a local of any index verifies there.
    #[test]
    fn a_local_outside_a_function_names_no_functions_local() {
        let source = "builtin.module public @m {
    %c = test.constant 0 : i1;
    test.guard %c {
        test.touch_local <{ local = #builtin.local_variable<5, i32> }>;
        test.yield : ();
    } : (i1);
};";
        parse_text::<crate::dialects::builtin::Module>(source)
            .unwrap_or_else(|err| panic!("{err}"));
    }

    /// A type map applied to a function rewrites its `locals` and every local its body names
    /// together, and the function still verifies.
    #[test]
    fn a_type_map_rewrites_the_locals_and_the_locals_named_together() {
        use crate::{ElementRewrite, TypeMapper, recursively_replace_elements_in};

        let source = alloc::format!(
            "{HEADER} @f(%0: i64) -> i64 locals(i64, i8) {{
    builtin.store_local 0, %0;
    test.touch_local <{{ local = #builtin.local_variable<0, i64> }}>;
    %1 = builtin.load_local 0 : i64;
    builtin.ret %1 : (i64);
}};"
        );
        let (context, function) = parse_text::<Function>(&source).expect("the text parses");
        let mapper = TypeMapper::new(&context, |context, ty| {
            if ty == context.i64() {
                context.i32()
            } else {
                ty
            }
        });
        let mut rewrite = ElementRewrite::types(mapper);
        let changed = recursively_replace_elements_in(function.as_operation_ref(), &mut rewrite)
            .expect("the rewrite");
        assert!(changed);
        drop(rewrite);

        let function = function.borrow();
        function
            .as_operation()
            .recursively_verify()
            .unwrap_or_else(|err| panic!("{err:?}"));
        assert_eq!(
            function.as_operation().to_string(),
            alloc::format!(
                "{HEADER} @f(%0: i32) -> i32 locals(i32, i8) {{
    builtin.store_local 0, %0;
    test.touch_local <{{ local = #builtin.local_variable<0, i32> }}>;
    %1 = builtin.load_local 0 : i32;
    builtin.ret %1 : (i32);
}};"
            )
        );
    }

    /// An operation moved into another function, which has fewer locals, is rejected by that
    /// function; the one it left verifies.
    #[test]
    fn an_operation_moved_to_a_function_with_fewer_locals_is_rejected() {
        use crate::{ProgramPoint, dialects::builtin::Module};

        let source = "builtin.module public @m {
    builtin.function public extern(\"C\") @f() locals(i32, i32) {
        test.touch_local <{ local = #builtin.local_variable<1, i32> }>;
        builtin.ret;
    };
    builtin.function public extern(\"C\") @g() locals(i32) {
        builtin.ret;
    };
};";
        let (_context, module) = parse_text::<Module>(source).expect("the module parses");
        let mut touch = None;
        let mut ret_of_g = None;
        module.borrow().as_operation().prewalk_all(|op: &Operation| {
            if op.name().to_string() == "test.touch_local" {
                touch = Some(op.as_operation_ref());
            }
            if op.is::<super::super::Ret>()
                && op
                    .nearest_parent_op::<Function>()
                    .is_some_and(|function| function.borrow().name().as_str() == "g")
            {
                ret_of_g = Some(op.as_operation_ref());
            }
        });
        let (mut touch, ret_of_g) = (touch.unwrap(), ret_of_g.unwrap());
        let source_function = touch.borrow().nearest_parent_op::<Function>().unwrap();
        touch.borrow_mut().move_to(ProgramPoint::before(ret_of_g));

        source_function
            .borrow()
            .as_operation()
            .recursively_verify()
            .expect("the function the operation left verifies");
        let err = module
            .borrow()
            .as_operation()
            .recursively_verify()
            .expect_err("`@g` has no local 1");
        let text =
            alloc::format!("{} {}", err.display(), label_messages(err.as_diagnostic()).join(" | "));
        assert!(text.contains("invalid operation builtin.function"), "{text}");
        assert!(text.contains("`test.touch_local` names a local"), "{text}");
        assert!(text.contains("this names local 1, but the function has 1 local(s)"), "{text}");
    }

    /// A local's attribute has no type of its own: one given a type through `set_type` is rejected
    /// by `Operation::verify`, with a message naming the operation that holds it.
    #[test]
    fn a_local_attribute_with_a_type_of_its_own_does_not_verify() {
        let (test, _local, _store, loaded) = built();
        let i32 = test.context().i32();
        let mut load = loaded.borrow().get_defining_op().unwrap();
        load.borrow().verify().expect("the load verifies before");
        load.borrow_mut().get_property_mut("local").set_type(i32);
        let err = load.borrow().verify().expect_err("the local has a type of its own");
        let text =
            alloc::format!("{} {}", err.display(), label_messages(err.as_diagnostic()).join(" | "));
        assert!(text.contains("'builtin.load_local'"), "{text}");
        assert!(text.contains("is a local variable with a type of its own, `i32`"), "{text}");
    }
}
