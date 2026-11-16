//! What the operations that name symbols share: the text of an operation that starts with the
//! symbols it names and its operands, and the report of a rule that is not met.

use alloc::{format, string::String};

use runic_ir::{
    AsValueRange, OpAsmParser, Operation, OperationState, ParseResult, Report, SmallVec, Spanned,
    Type, Value, ValueRef, diagnostics::report, dialects::builtin::attributes::SymbolRefAttr,
    formatter::const_text, parse::UnresolvedOperand, print::AsmPrinter,
};

/// The report of `op` being invalid because of `problem`.
pub(super) fn invalid(op: &Operation, problem: String) -> Report {
    report!(
        message: "invalid operation",
        labels: [primary(op.span(), problem)],
    )
}

/// The report of `op` breaking its rule: "`wasm.load` `rule`, but `problem`".
pub(super) fn rule_error(op: &Operation, rule: &str, problem: String) -> Report {
    invalid(op, format!("`{}` {rule}, but {problem}", op.name()))
}

/// The type of operand `index` of `op`.
pub(super) fn operand_type(op: &Operation, index: usize) -> Type {
    op.operands()[index].borrow().as_value_ref().borrow().ty()
}

/// The type of the one result of `op`.
pub(super) fn result_type(op: &Operation) -> Type {
    op.results()[0].borrow().ty()
}

/// Checks that operand `index` of `op`, which the message calls `what`, has the type `expected`.
pub(super) fn expect_operand(
    op: &Operation,
    rule: &str,
    index: usize,
    what: &str,
    expected: Type,
) -> Result<(), Report> {
    let found = operand_type(op, index);
    if found == expected {
        Ok(())
    } else {
        Err(rule_error(
            op,
            rule,
            format!("{what} has type `{}`", crate::val_type_name(found)),
        ))
    }
}

/// Checks that the one result of `op` has the type `expected`.
pub(super) fn expect_result(op: &Operation, rule: &str, expected: Type) -> Result<(), Report> {
    let found = result_type(op);
    if found == expected {
        Ok(())
    } else {
        Err(rule_error(
            op,
            rule,
            format!(
                "its result has type `{}`, not `{}`",
                crate::val_type_name(found),
                crate::val_type_name(expected)
            ),
        ))
    }
}

/// Prints ` @a, @b, %x, %y`: the symbols the operation names, then all its operands.
pub(super) fn print_symbols_and_operands(
    printer: &mut AsmPrinter<'_>,
    op: &Operation,
    symbols: &[&SymbolRefAttr],
) {
    let mut first = true;
    let mut separate = |printer: &mut AsmPrinter<'_>| {
        if first {
            printer.print_space();
            first = false;
        } else {
            *printer += const_text(", ");
        }
    };
    for symbol in symbols {
        separate(printer);
        printer.print_symbol_path(symbol.path());
    }
    if op.num_operands() > 0 {
        separate(printer);
        printer.print_value_uses(op.operands().group(0).as_value_range());
    }
}

/// Prints ` attributes { .. }` when `op` has discardable attributes.
pub(super) fn print_attributes(printer: &mut AsmPrinter<'_>, op: &Operation) {
    if op.has_attributes() {
        printer.print_space();
        printer.print_keyword("attributes");
        printer.print_space();
        printer.print_attribute_dictionary(
            op.attributes().iter().map(|attr| *attr.as_named_attribute()),
        );
    }
}

/// Reads `@a, @b, %x, %y`: one symbol for each name in `symbols`, which become the properties of
/// those names, then `operands` operands, whose types are those of the values they name. Returns
/// the operands.
pub(super) fn parse_symbols_and_operands(
    state: &mut OperationState,
    parser: &mut dyn OpAsmParser<'_>,
    symbols: &[&'static str],
    operands: usize,
) -> ParseResult<SmallVec<[ValueRef; 4]>> {
    let mut first = true;
    for name in symbols {
        if !first {
            parser.parse_comma()?;
        }
        first = false;
        let symbol = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute(*name, symbol);
    }
    let mut unresolved = SmallVec::<[UnresolvedOperand; 4]>::new();
    for _ in 0..operands {
        if !first {
            parser.parse_comma()?;
        }
        first = false;
        unresolved.push(parser.parse_operand(true)?);
    }
    let unknown = parser.context().unknown_type();
    let mut values = SmallVec::new();
    for operand in unresolved {
        let value = parser.resolve_operand(operand, unknown)?;
        state.add_operand(value);
        values.push(value);
    }
    Ok(values)
}
