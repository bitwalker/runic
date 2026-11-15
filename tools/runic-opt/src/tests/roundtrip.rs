//! `--verify-roundtrip`: both forms of the parsed operation are checked to read back.

use std::sync::atomic::{AtomicU32, Ordering};

use runic_core::{
    AttrPrinter, AttributeRef, Context, OpAsmParser, OpParser, OpPrinter, OperationState, Report,
    Value,
    attributes::AttrParser,
    derive::{DialectAttribute, operation},
    dialects::{
        builtin::attributes::{ListAttr, LocalVariable, LocalVariableAttr},
        test::TestDialect,
    },
    parse::{ParseResult, Parser, ParserExt},
    print::AsmPrinter,
    traits::{AnyInteger, InferTypeOpInterface},
};

use super::{INVALID, LOCALS, REPEATED_ADD, opt};

/// One module per dialect of the workspace that uses each of its operations, as the transform
/// crate's tests of the generic form use them.
#[path = "../../../../crates/transform/src/tests/dialect_modules.rs"]
mod dialect_modules;

/// An operation whose custom printer writes its property and whose custom parser does not read
/// it back: `test.roundtrip_lossy_property 3 : i32;` parses with the index 0.
#[operation(dialect = TestDialect, implements(OpPrinter))]
pub struct RoundtripLossyProperty {
    #[attr]
    local: LocalVariableAttr,
}

impl OpPrinter for RoundtripLossyProperty {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let local = *self.get_local();
        printer.print_space();
        printer.print_decimal_integer(local.index());
        printer.print_colon_type(local.ty());
    }
}

impl OpParser for RoundtripLossyProperty {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        // The written index is read and dropped.
        let _ = parser.parse_decimal_integer::<u32>()?;
        let ty = parser.parse_colon_type()?.into_inner();
        let local = parser
            .context_rc()
            .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(0, ty))
            .as_attribute_ref();
        state.add_attribute("local", local);
        Ok(())
    }
}

/// An operation whose custom printer leaves its property out, and whose custom parser gives it
/// the index 0: `test.roundtrip_forgetful_printer;` reads back as itself, whatever the index.
#[operation(dialect = TestDialect, implements(OpPrinter))]
pub struct RoundtripForgetfulPrinter {
    #[attr]
    local: LocalVariableAttr,
}

impl OpPrinter for RoundtripForgetfulPrinter {
    fn print(&self, _printer: &mut AsmPrinter<'_>) {}
}

impl OpParser for RoundtripForgetfulPrinter {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let local = parser
            .context_rc()
            .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(0, parser.context().i32()))
            .as_attribute_ref();
        state.add_attribute("local", local);
        Ok(())
    }
}

/// An attribute whose printer writes one more than it holds: `#test.roundtrip_drifting<3>` prints
/// as `#test.roundtrip_drifting<4>`, which reads back as 4 and prints as 5, in either form of an
/// operation that holds it.
#[derive(DialectAttribute, Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
#[attribute(dialect = TestDialect, implements(AttrPrinter))]
pub struct RoundtripDrifting(u32);

impl AttrPrinter for RoundtripDriftingAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_decimal_integer(self.value.0 + 1);
    }
}

impl AttrParser for RoundtripDriftingAttr {
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<AttributeRef> {
        let value = parser.parse_decimal_integer::<u32>()?.into_inner();
        Ok(parser
            .context_rc()
            .create_attribute::<RoundtripDriftingAttr, _>(RoundtripDrifting(value))
            .as_attribute_ref())
    }
}

/// An operation whose custom printer writes text its custom parser does not read.
#[operation(dialect = TestDialect, implements(OpPrinter))]
pub struct RoundtripUnreadableCustomForm {}

impl OpPrinter for RoundtripUnreadableCustomForm {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_space();
        *printer += runic_core::formatter::const_text("<<unreadable>>");
    }
}

impl OpParser for RoundtripUnreadableCustomForm {
    fn parse(_state: &mut OperationState, _parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        Ok(())
    }
}

/// An operation whose result type is inferred, a different one each time: the generic form states
/// the type it was given, which the parser does not infer again. The custom form states no type,
/// so it parses back and prints the same text, but to an operation of another type, which the
/// generic prints show.
#[operation(dialect = TestDialect, implements(InferTypeOpInterface, OpPrinter))]
pub struct RoundtripUnstableResultType {
    #[result]
    result: AnyInteger,
}

static WIDTH: AtomicU32 = AtomicU32::new(8);

impl InferTypeOpInterface for RoundtripUnstableResultType {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        let width = WIDTH.fetch_add(1, Ordering::Relaxed);
        self.result_mut().set_type(context.int_type(width));
        Ok(())
    }
}

impl OpPrinter for RoundtripUnstableResultType {
    fn print(&self, _printer: &mut AsmPrinter<'_>) {}
}

impl OpParser for RoundtripUnstableResultType {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        // The result type is the inferred one, whatever it is given here.
        state.results.push(parser.context_rc().int_type(32));
        Ok(())
    }
}

/// The lossy operation with `index`, written in the generic form: the custom parser drops the
/// index, so the text of the custom form cannot say it.
fn lossy(index: u32) -> String {
    format!(
        "        \"test.roundtrip_lossy_property\"() <{{ local = #builtin.local_variable<{index}, \
         i32> }}> : () -> ();"
    )
}

/// A function holding `body`.
fn module(body: &str) -> String {
    format!(
        "builtin.module public @m {{\n    builtin.function public extern(\"C\") @f(%a: i32) -> \
         i32 locals(i32, i32, i32, i32) {{\n{body}\n        builtin.ret %a : (i32);\n    \
         }};\n}};\n"
    )
}

#[test]
fn a_module_that_reads_back_passes_silently() {
    for text in [LOCALS, REPEATED_ADD] {
        let plain = opt(&[], text);
        plain.succeeded();
        let checked = opt(&["--verify-roundtrip"], text);
        checked.succeeded();
        assert_eq!(checked.stderr, "");
        assert_eq!(checked.stdout, plain.stdout);
    }
}

#[test]
fn a_custom_form_that_drops_a_property_is_an_error_naming_the_form() {
    let text = module(&lossy(3));
    let output = opt(&["--verify-roundtrip"], &text);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output
            .stderr
            .contains("the custom form does not print the same program when parsed back"),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains("test.roundtrip_lossy_property 3: i32"),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains("test.roundtrip_lossy_property 0: i32"),
        "{}",
        output.stderr
    );
    // The generic form holds the property.
    assert!(!output.stderr.contains("the generic form"), "{}", output.stderr);
}

#[test]
fn a_custom_form_that_does_not_parse_is_an_error_with_the_parsers_diagnostics() {
    let text = module("        test.roundtrip_unreadable_custom_form;");
    let output = opt(&["--verify-roundtrip"], &text);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.contains("the custom form does not parse back"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("parsing it said:"), "{}", output.stderr);
    // The parser's own message, with the printed line it points at
    assert!(output.stderr.contains("unexpected token '<'"), "{}", output.stderr);
    assert!(
        output
            .stderr
            .contains("|         test.roundtrip_unreadable_custom_form <<unreadable>>;"),
        "{}",
        output.stderr
    );
    assert!(!output.stderr.contains("the generic form"), "{}", output.stderr);
    // The diagnostics are located in the printed text, not in the input.
    assert!(!output.stderr.contains("<stdin>"), "{}", output.stderr);
}

#[test]
fn a_generic_form_that_does_not_read_back_is_an_error_naming_the_form() {
    let text = module("        %x = test.roundtrip_unstable_result_type;");
    let output = opt(&["--verify-roundtrip"], &text);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.contains("the generic form does not parse back"),
        "{}",
        output.stderr
    );
    // The custom form reads back to another type
    assert!(
        output.stderr.contains("the custom form loses part of the operation"),
        "{}",
        output.stderr
    );
}

/// A custom form that leaves a part of the operation out prints the same text twice: what it read
/// back to is compared, in the generic form, with the input. The lines are shown as printed,
/// indentation included.
#[test]
fn a_custom_form_that_leaves_out_a_property_loses_part_of_the_operation() {
    let text = module(
        "        \"test.roundtrip_forgetful_printer\"() <{ local = #builtin.local_variable<3, \
         i32> }> : () -> ();",
    );
    let output = opt(&["--verify-roundtrip"], &text);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output.stderr.contains("error: the custom form loses part of the operation"),
        "{}",
        output.stderr
    );
    let line = |index: u32| {
        format!(
            ":         \"test.roundtrip_forgetful_printer\"() <{{ local = \
             #builtin.local_variable<{index}, i32> }}> : () -> ();\n"
        )
    };
    let input = format!("first differing line, printed from the input{}", line(3));
    assert!(output.stderr.contains(&input), "{}", output.stderr);
    let back = format!("first differing line, printed after parsing it back{}", line(0));
    assert!(output.stderr.contains(&back), "{}", output.stderr);
    assert!(!output.stderr.contains("the generic form"), "{}", output.stderr);
}

/// An attribute whose printer drifts makes each form print another program once read back; the
/// generic form's error is its own.
#[test]
fn a_generic_form_that_prints_another_program_is_an_error_naming_the_form() {
    let text = "builtin.function public extern(\"C\") @f(%a: i32) -> i32 {\n    builtin.ret %a \
                attributes { note = #test.roundtrip_drifting<3> } : (i32);\n};\n";
    let output = opt(&["--verify-roundtrip"], text);
    output.failed();
    assert_eq!(output.stdout, "");
    assert!(
        output
            .stderr
            .contains("the generic form does not print the same program when parsed back"),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains(
            "printed from the input:     \"builtin.ret\"(%0) { note = #test.roundtrip_drifting<4> \
             } : i32 -> ();"
        ),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains(
            "printed after parsing it back:     \"builtin.ret\"(%0) { note = \
             #test.roundtrip_drifting<5> } : i32 -> ();"
        ),
        "{}",
        output.stderr
    );
}

/// The attribute dictionary of `builtin.ret` is written by its custom form, so the check passes
/// and the default print keeps it.
#[test]
fn the_check_passes_on_a_return_with_an_attribute_dictionary() {
    let text = "builtin.function public extern(\"C\") @f(%a: i32) -> i32 {\n    \
                \"builtin.ret\"(%a) { note = #builtin.i32<7> } : i32 -> ();\n};\n";
    let generic = opt(&["--print-generic"], text);
    generic.succeeded();
    assert!(generic.stdout.contains("note = #builtin.i32<7>"), "{}", generic.stdout);
    let checked = opt(&["--verify-roundtrip"], text);
    checked.succeeded();
    assert_eq!(checked.stderr, "");
    assert!(
        checked
            .stdout
            .contains("builtin.ret %0 attributes { note = #builtin.i32<7> } : (i32);"),
        "{}",
        checked.stdout
    );
}

/// The terminators with hand-written custom forms (`builtin.ret`, `builtin.ret_imm`,
/// `scf.continue`, `scf.break`) write their attribute dictionary, and read it back.
#[test]
fn the_custom_form_of_every_terminator_holds_its_dictionary() {
    let text = r#"builtin.module public @m {
    builtin.function public extern("C") @a(%a: i32) -> i32 {
        "builtin.ret"(%a) { note = #builtin.i32<1> } : i32 -> ();
    };
    builtin.function public extern("C") @b() -> i7 {
        "builtin.ret_imm"() <{ value = #builtin.int<-1 : i7> }> { note = #builtin.i32<2> } : () -> ();
    };
    builtin.function public extern("C") @c(%n: i32, %c: i1) -> i32 {
        %r = scf.while %n before {
        ^head(%i: i32):
            scf.condition %c, %i : (i1, i32);
        } after {
        ^body(%j: i32):
            %k = scf.if %c then {
                "scf.continue"(%j) <{ levels = #builtin.i32<1> }> { note = #builtin.i32<3> } : i32 -> ();
            } else {
                "scf.break"(%j) <{ levels = #builtin.i32<1> }> { note = #builtin.i32<4> } : i32 -> ();
            } : (i1) -> (i32);
            scf.yield %k : (i32);
        } : (i32) -> (i32);
        builtin.ret %r : (i32);
    };
    builtin.function public extern("C") @d() {
        "builtin.ret"() { note = #builtin.i32<5> } : () -> ();
    };
};
"#;
    let output = opt(&["--verify-roundtrip"], text);
    output.succeeded();
    assert_eq!(output.stderr, "");
    for kept in [
        "builtin.ret %0 attributes { note = #builtin.i32<1> } : (i32);",
        "builtin.ret_imm -1 : i7 attributes { note = #builtin.i32<2> };",
        "scf.continue 1 %4 attributes { note = #builtin.i32<3> } : (i32);",
        "scf.break 1 %4 attributes { note = #builtin.i32<4> } : (i32);",
        "builtin.ret attributes { note = #builtin.i32<5> };",
    ] {
        assert!(output.stdout.contains(kept), "{kept} in\n{}", output.stdout);
    }
}

#[test]
fn both_forms_are_reported_when_neither_reads_back() {
    let text = module(&format!("{}\n        %x = test.roundtrip_unstable_result_type;", lossy(3)));
    let output = opt(&["--verify-roundtrip"], &text);
    output.failed();
    assert!(output.stderr.contains("the custom form does not"), "{}", output.stderr);
    assert!(output.stderr.contains("the generic form does not"), "{}", output.stderr);
}

#[test]
fn the_errors_take_part_in_verify_diagnostics() {
    let text = format!(
        "// expected-error @unknown {{{{the custom form does not parse back}}}}\n{}",
        module("        test.roundtrip_unreadable_custom_form;")
    );
    let output = opt(&["--verify-roundtrip", "--verify-diagnostics"], &text);
    output.succeeded();
    assert_eq!(output.stdout, "");
    assert_eq!(output.stderr, "");

    // Not expected: the diagnostic is reported as unexpected.
    let output = opt(
        &["--verify-roundtrip", "--verify-diagnostics"],
        &module("        test.roundtrip_unreadable_custom_form;"),
    );
    output.failed();
    assert!(
        output.stderr.contains("the custom form does not parse back"),
        "{}",
        output.stderr
    );
}

#[test]
fn the_check_runs_per_case_and_before_the_passes() {
    let good = REPEATED_ADD;
    let bad = module(&lossy(3));
    let text = format!("{good}// -----\n{bad}// -----\n{good}");
    let output = opt(&["--verify-roundtrip", "--split-input-file", "-p", "cse"], &text);
    output.failed();
    // The first and the third cases are printed after the pass; the second prints nothing.
    assert_eq!(output.stdout.matches("// -----").count(), 1, "{}", output.stdout);
    assert_eq!(output.stdout.matches("arith.add").count(), 4, "{}", output.stdout);
    assert_eq!(output.stderr.matches("does not print the same program").count(), 1);
}

#[test]
fn a_failed_check_runs_no_pass() {
    let bad = module(&lossy(3));
    let output = opt(&["--verify-roundtrip", "-p", "test-fail", "--no-verify"], &bad);
    output.failed();
    assert!(!output.stderr.contains("test-fail"), "{}", output.stderr);
    assert!(output.stderr.contains("the custom form"), "{}", output.stderr);
}

#[test]
fn the_check_follows_the_verification_the_options_say() {
    // A function that does not verify, parsed without verification, still reads back.
    let output = opt(&["--verify-roundtrip", "--no-verify"], super::INVALID);
    output.succeeded();
}

/// Every named input of the tool's tests that parses reads back in both forms (an input that does
/// not parse or verify is skipped: it has no operation to print), and so does one module of each
/// dialect.
#[test]
fn the_inputs_of_the_tests_and_a_module_of_each_dialect_read_back() {
    let inputs = [
        LOCALS,
        REPEATED_ADD,
        INVALID,
        super::dominance::CASES,
        super::pipeline::LOCALS_AND_REPEATS,
        super::pipeline::WORLD,
        super::test_passes::WIDE,
        super::test_passes::WIDE_CONVERTED,
        super::test_passes::RANGES,
        super::expectations::MISSING_SEMICOLON,
        super::splitting::BAD,
    ];
    let mut read = 0;
    for text in inputs {
        // Each input is checked whole, and, with the cases of it apart, case by case, with and
        // without verification: what fails to parse is not what is being checked.
        for args in [
            &["--verify-roundtrip"][..],
            &["--verify-roundtrip", "--split-input-file"],
            &["--verify-roundtrip", "--split-input-file", "--no-verify"],
        ] {
            let output = opt(args, text);
            assert!(
                !output.stderr.contains("form does not"),
                "{args:?} on\n{text}\n{}",
                output.stderr
            );
            read += usize::from(output.stdout.contains("builtin."));
        }
    }
    assert!(read >= 15, "only {read} of the runs printed an operation");

    for (dialect, text) in dialect_modules::DIALECT_MODULES {
        let output = opt(&["--verify-roundtrip"], text);
        assert_eq!(output.stderr, "", "the module of {dialect}");
        output.succeeded();
    }
}

/// An operation with a list property, which its derived custom form writes.
#[operation(dialect = TestDialect)]
pub struct RoundtripListProperty {
    #[attr]
    items: ListAttr,
}

/// A module that holds lists in a property and in an attribute dictionary: nested lists, empty
/// lists, and lists of symbol references (to the enclosing function and to a sibling).
const LISTS: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32) -> i32 {
        "test.roundtrip_list_property"() <{ items = [[#builtin.i32<1>, []], [], [#builtin.symbol<@g>]] }> { none = [], refs = [#builtin.symbol<@f>, [#builtin.symbol<@g>, #builtin.string<"s">]] } : () -> ();
        builtin.ret %a : (i32);
    };
    builtin.function public extern("C") @g() {
        builtin.ret;
    };
};
"#;

#[test]
fn lists_in_a_property_and_a_dictionary_read_back() {
    let plain = opt(&[], LISTS);
    plain.succeeded();
    for list in [
        "items = [[#builtin.i32<1>, []], [], [#builtin.symbol<@g>]]",
        "none = []",
        "refs = [#builtin.symbol<@f>, [#builtin.symbol<@g>, #builtin.string<\"s\">]]",
    ] {
        assert!(plain.stdout.contains(list), "{list} in\n{}", plain.stdout);
    }
    let checked = opt(&["--verify-roundtrip"], LISTS);
    checked.succeeded();
    assert_eq!(checked.stderr, "");
    assert_eq!(checked.stdout, plain.stdout);
    let generic = opt(&["--verify-roundtrip", "--print-generic"], LISTS);
    generic.succeeded();
    assert_eq!(generic.stderr, "");
}

/// The symbol references in the lists are linked when the text is read, and unlinked when the
/// operation holding them is erased.
#[test]
fn symbol_references_read_in_a_list_are_linked() {
    use std::rc::Rc;

    use runic_core::{
        Operation,
        parse::{ParserConfig, parse_any},
    };

    let context = Rc::new(Context::default());
    let module = parse_any(ParserConfig::new(context.clone()), "lists.runic", LISTS).unwrap();
    let (f, g) = {
        let module = module.borrow();
        let symbols = module.as_symbol_table().unwrap().symbol_manager();
        (symbols.lookup_op("f").unwrap(), symbols.lookup_op("g").unwrap())
    };
    let uses = |symbol: runic_core::OperationRef| -> usize {
        symbol.borrow().as_symbol().unwrap().iter_uses().count()
    };
    let mut holder = {
        let f = f.borrow();
        let body = f.regions().front().get().unwrap().entry_block_ref().unwrap();
        let holder = body.borrow().front().unwrap();
        assert!(holder.borrow().is::<RoundtripListProperty>());
        holder
    };
    let owners = |symbol: runic_core::OperationRef| {
        symbol
            .borrow()
            .as_symbol()
            .unwrap()
            .iter_uses()
            .all(|usage| usage.owner == holder)
    };
    assert_eq!(uses(g), 2, "one use in the property, one in the dictionary");
    assert_eq!(uses(f), 1, "one use in the dictionary");
    assert!(owners(f) && owners(g));
    assert!(holder.borrow().verify().is_ok());

    Operation::erase(&mut holder.borrow_mut());
    assert_eq!(uses(f), 0);
    assert_eq!(uses(g), 0);
}
