//! The generic form reads back what the printer writes, over every corpus of the workspace this
//! crate can reach: the corpora of `corpora.rs` (the fixtures, the golden files, the random
//! programs of `cse_model.rs` and `local2reg_model.rs`), the corpus of `runic_core::testing::calls`,
//! the random modules of the conversion tests, and one module per dialect that uses each of the
//! dialect's operations.

use alloc::{
    collections::BTreeSet,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    Context, OpPrinter, OpPrintingFlags, Operation, OperationRef,
    parse::{self, ParserConfig},
    print::{AsmPrinter, same_program},
    testing::text::describe,
};

use super::{corpora, dialect_modules::DIALECT_MODULES};

/// `op` printed as the root of a print, in the generic form when `generic` is set.
fn print(op: &Operation, generic: bool) -> String {
    let flags = OpPrintingFlags {
        generic,
        ..Default::default()
    };
    let mut printer = AsmPrinter::root(op.context_rc(), &flags);
    OpPrinter::print(op, &mut printer);
    printer.finish().to_string()
}

fn new_context() -> Rc<Context> {
    let context = corpora::new_context();
    context.get_or_register_dialect::<runic_core::dialects::test::TestDialect>();
    context.get_or_register_dialect::<runic_dialect_ub::UndefinedBehaviorDialect>();
    context
}

/// `text` parsed and verified, with the context that owns it: the operation must not outlive it.
fn parse(text: &str) -> Result<(Rc<Context>, OperationRef), String> {
    let context = new_context();
    parse::parse_any(ParserConfig::new(context.clone()), "case", text)
        .into_result()
        .map(|op| (context, op))
        .map_err(|err| describe(&err))
}

/// Every text of the corpora, with its name.
fn sources() -> Vec<(String, String)> {
    let mut sources = corpora::corpus()
        .into_iter()
        .map(|case| (case.name, case.text))
        .collect::<Vec<_>>();
    for (index, text) in runic_core::testing::calls::CORPUS.iter().enumerate() {
        sources.push((format!("calls corpus {index}"), text.to_string()));
    }
    for (index, text) in super::conversion::sample_modules(super::seed(0x6e6e_0001), 40)
        .into_iter()
        .enumerate()
    {
        sources.push((format!("conversion module {index}"), text));
    }
    for (name, text) in DIALECT_MODULES {
        sources.push((format!("dialect module: {name}"), text.to_string()));
    }
    sources
}

/// Parses `text`, prints it in both forms, parses the generic text and prints that in both
/// forms: each print of the second parse is the same program as the print of the first in the
/// same form. Returns the names of the operations the text holds.
fn reads_back(text: &str) -> Result<BTreeSet<String>, String> {
    let (_context, first) = parse(text)?;
    let custom = print(&first.borrow(), false);
    let generic = print(&first.borrow(), true);
    let mut names = BTreeSet::new();
    first.borrow().prewalk_all(|op: &Operation| {
        names.insert(op.name().to_string());
    });

    let (_context, second) = parse(&generic)
        .map_err(|err| format!("the generic text does not parse: {err}\n{generic}"))?;
    let custom_again = print(&second.borrow(), false);
    if !same_program(&custom_again, &custom) {
        return Err(format!(
            "the generic text reads back as another program:\n{custom}\nbecame\n{custom_again}"
        ));
    }
    let generic_again = print(&second.borrow(), true);
    if !same_program(&generic_again, &generic) {
        return Err(format!(
            "the generic text prints differently once read \
             back:\n{generic}\nbecame\n{generic_again}"
        ));
    }
    Ok(names)
}

/// Every case of every corpus parses, prints in the generic form, and that text parses, verifies
/// and prints the same program, in both forms, as the case. Between them the cases use every
/// operation registered here.
#[test]
fn every_case_reads_back_from_the_generic_form() {
    // These golden files hold `test.debug_value`, an operation that only core's own tests
    // register; core's tests of them read their generic text back.
    const UNPARSABLE: [&str; 2] = [
        "golden: transparent_ops_do_not_interfere_with_dce_after.hir",
        "golden: transparent_ops_do_not_interfere_with_dce_before.hir",
    ];

    let mut used = BTreeSet::new();
    let mut failures = Vec::new();
    let mut unparsed = Vec::new();
    let mut checked = 0;
    for (name, text) in sources() {
        if let Err(err) = parse(&text) {
            assert!(UNPARSABLE.contains(&name.as_str()), "{name} does not parse: {err}\n{text}");
            unparsed.push(name);
            continue;
        }
        match reads_back(&text) {
            Ok(names) => used.extend(names),
            Err(err) => failures.push(format!("{name}: {err}")),
        }
        checked += 1;
    }
    assert_eq!(unparsed, UNPARSABLE, "the cases that do not parse");
    assert!(checked > 300, "{checked}");
    assert!(
        failures.is_empty(),
        "{} of {checked} cases:\n{}",
        failures.len(),
        failures.join("\n\n")
    );

    let context = new_context();
    let mut registered = BTreeSet::new();
    for dialect in context.registered_dialects().values() {
        for name in dialect.registered_ops().iter() {
            registered.insert(name.to_string());
        }
    }
    let missing = registered.difference(&used).collect::<Vec<_>>();
    assert!(missing.is_empty(), "operations no case uses: {missing:?}");
}
