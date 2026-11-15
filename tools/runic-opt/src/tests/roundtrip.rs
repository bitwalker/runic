//! `--verify-roundtrip`: both forms of the parsed operation are checked to read back.

use std::sync::atomic::{AtomicU32, Ordering};

use runic_core::{
    Context, OpAsmParser, OpParser, OpPrinter, OperationState, Report, Value,
    derive::operation,
    dialects::{
        builtin::attributes::{LocalVariable, LocalVariableAttr},
        test::TestDialect,
    },
    parse::{ParseResult, ParserExt},
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
/// so it reads back.
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
    assert!(!output.stderr.contains("the custom form"), "{}", output.stderr);
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
