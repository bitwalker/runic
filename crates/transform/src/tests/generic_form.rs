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
        "golden: transparent_ops_do_not_interfere_with_dce_after.runic",
        "golden: transparent_ops_do_not_interfere_with_dce_before.runic",
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

/// The lines of `text` that hold an operation in the generic form: a quoted name, alone or after
/// a result list.
fn generic_lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim_start)
        .filter(|line| {
            line.starts_with('"')
                || line.split_once(" = ").is_some_and(|(results, rest)| {
                    results.starts_with('%') && rest.starts_with('"')
                })
        })
        .collect()
}

/// `text` parsed, printed in its default form, and that text parsed and printed again: the same
/// program. Returns the default print, after checking that the text also reads back from the
/// generic form.
fn default_print_reads_back(text: &str) -> String {
    let (_context, first) = parse(text).unwrap_or_else(|err| panic!("{err}\n{text}"));
    let printed = print(&first.borrow(), false);
    let (_context, again) = parse(&printed).unwrap_or_else(|err| panic!("{err}\n{printed}"));
    assert!(same_program(&print(&again.borrow(), false), &printed), "{printed}");
    reads_back(text).unwrap_or_else(|err| panic!("{err}"));
    printed
}

/// A block may be stored, and printed, after a block it dominates, so a use may name a value
/// that the text defines later in another block. The operation whose custom form takes the
/// operand's type from the operand prints in the generic form; every other operation keeps its
/// custom form, those that write their operands' types (`test.sink`, a derived form, and
/// `builtin.ret`, a hand-written one) included, and the default print reads back.
#[test]
fn a_use_before_its_definition_in_a_later_block_prints_in_the_generic_form() {
    let text = "\
builtin.function public extern(\"C\") @f(%a: i32) -> i32 {
    cf.br ^c(%a : i32);
^b:
    %y = \"test.add\"(%x, %w) <{ overflow = #builtin.overflow<wrapping> }> : (i32, i32) -> i32;
    test.sink %x, %w : (i32, i32);
    builtin.ret %x : (i32);
^c(%w: i32):
    %x = test.add %w, %w;
    cf.br ^b;
};";
    let printed = default_print_reads_back(text);
    let generic = generic_lines(&printed);
    assert_eq!(generic.len(), 1, "{printed}");
    assert!(generic[0].contains("= \"test.add\"("), "{printed}");
    assert!(printed.contains("    test.sink %"), "{printed}");
    assert!(printed.contains("    builtin.ret %"), "{printed}");

    // Written with the custom form, the forward use does not parse: the type of `%x` is not
    // known where it is read.
    let custom = text.replace(
        "\"test.add\"(%x, %w) <{ overflow = #builtin.overflow<wrapping> }> : (i32, i32) -> i32",
        "test.add %x, %w",
    );
    assert!(parse(&custom).is_err(), "{custom}");
}

/// The same, with the use nested in an operation of the earlier block: the use's place in the
/// definition's region is that of the `scf.if` that holds it. The `scf.if` and the `scf.yield`
/// write their operands' types and keep their custom forms.
#[test]
fn a_nested_use_before_its_definition_in_a_later_block_prints_in_the_generic_form() {
    let text = "\
builtin.function public extern(\"C\") @f(%a: i32, %c: i1) -> i32 {
    cf.br ^c(%a : i32);
^b:
    %y = scf.if %c then {
        %z = \"test.add\"(%x, %x) <{ overflow = #builtin.overflow<wrapping> }> : (i32, i32) -> i32;
        scf.yield %z : (i32);
    } else {
        scf.yield %x : (i32);
    } : (i1) -> (i32);
    builtin.ret %y : (i32);
^c(%w: i32):
    %x = test.add %w, %w;
    cf.br ^b;
};";
    let printed = default_print_reads_back(text);
    let generic = generic_lines(&printed);
    assert_eq!(generic.len(), 1, "{printed}");
    assert!(generic[0].contains("= \"test.add\"("), "{printed}");
    assert!(printed.contains("= scf.if %"), "{printed}");
    assert_eq!(printed.matches("scf.yield %").count(), 2, "{printed}");
}

/// A use in a later block of a value defined in an earlier one is not forward: every operation
/// keeps its custom form.
#[test]
fn a_use_after_its_definition_in_an_earlier_block_keeps_the_custom_form() {
    let text = "\
builtin.function public extern(\"C\") @f(%a: i32) -> i32 {
    %x = test.add %a, %a;
    cf.br ^b;
^b:
    %y = test.add %x, %x;
    builtin.ret %y : (i32);
};";
    let printed = default_print_reads_back(text);
    assert!(generic_lines(&printed).is_empty(), "{printed}");
}

/// With an entry added to the attribute dictionary of every operation of the dialect modules,
/// each module printed in its default form reads back to what its generic form says: every
/// operation's custom form holds its dictionary.
#[test]
fn every_custom_form_holds_the_attribute_dictionary() {
    use runic_core::dialects::builtin::attributes::I32Attr;

    for (name, text) in DIALECT_MODULES {
        let (context, module) = parse(text).unwrap_or_else(|err| panic!("{name}: {err}"));
        let mut ops = Vec::new();
        module.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
        for (index, mut op) in ops.iter().copied().enumerate() {
            let note = context.create_attribute::<I32Attr, _>(index as i32).as_attribute_ref();
            // Not `note`: the lit files the modules are read from already hold `note` entries, so a
            // key of its own is what lets the count below see that each added entry survives.
            let key = runic_core::interner::Symbol::intern(format!("added{index}").as_str());
            op.borrow_mut().set_attribute(key, note);
        }
        let generic = print(&module.borrow(), true);
        let custom = print(&module.borrow(), false);
        let (_context, read_back) = parse(&custom).unwrap_or_else(|err| {
            panic!("{name}: the custom text does not parse: {err}\n{custom}")
        });
        let generic_again = print(&read_back.borrow(), true);
        assert!(
            same_program(&generic_again, &generic),
            "{name}: the custom form loses part of an \
             operation:\n{generic}\nbecame\n{generic_again}"
        );
        assert_eq!(generic.matches("added").count(), ops.len(), "{name}: {generic}");
    }
}
