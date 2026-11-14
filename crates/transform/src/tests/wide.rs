//! Wide operations print on a small stack: an operation with many successors, regions,
//! parameters, locals or array elements prints, in the custom and the generic form, on a 256 KiB
//! stack, because the printer joins the pieces it is given as a balanced tree whatever the number
//! of pieces a custom printer appends.

use alloc::{format, rc::Rc, string::String, vec::Vec};

use runic_core::{
    Context, OperationRef,
    dialects::builtin::attributes::{Array, I32ArrayAttr},
    formatter::PrettyPrint,
    parse::{ParserConfig, parse_any},
    print::{AsmPrinter, OpPrinter, OpPrintingFlags},
    testing::text::describe,
};

const WIDE: usize = 20_000;

/// The number of cases of the `scf.index_switch`: printing and verifying an operation with `n`
/// regions costs time quadratic in `n` (each region asks its parent for the regions that follow
/// it), so this is fewer than [WIDE], and still several times the width that overflowed a
/// 256 KiB stack before the printer joined its pieces (under 500 cases).
const INDEX_SWITCH_CASES: usize = 2_000;

fn function(signature: &str, body: &str) -> String {
    format!("builtin.function public extern(\"C\") @f{signature} {{\n{body}\n}};")
}

/// One operation with `n` successors.
fn switch(n: usize) -> String {
    let cases = (0..n)
        .map(|i| format!("#builtin.i32<{i}> -> ^b{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let blocks = (0..n).map(|i| format!("^b{i}:\n    builtin.ret;\n")).collect::<String>();
    function(
        "(%s: i32)",
        &format!("    cf.switch %s [{cases}], ^other : (i32);\n{blocks}^other:\n    builtin.ret;"),
    )
}

/// One operation with `n + 1` regions.
fn index_switch(n: usize) -> String {
    let cases = (0..n)
        .map(|i| format!("    case {i} {{\n        scf.yield : ();\n    }}\n"))
        .collect::<String>();
    function(
        "(%s: i32)",
        &format!(
            "    scf.index_switch %s\n{cases}    default {{\n        scf.yield : ();\n    }} : \
             ();\n    builtin.ret;"
        ),
    )
}

/// A function with `n` parameters and `n` results, returning them all.
fn parameters(n: usize) -> String {
    let params = (0..n).map(|i| format!("%a{i}: i32")).collect::<Vec<_>>().join(", ");
    let values = (0..n).map(|i| format!("%a{i}")).collect::<Vec<_>>().join(", ");
    let types = alloc::vec!["i32"; n].join(", ");
    function(
        &format!("({params}) -> ({types})"),
        &format!("    builtin.ret {values} : ({types});"),
    )
}

/// A function with `n` locals.
fn locals(n: usize) -> String {
    let types = alloc::vec!["i32"; n].join(", ");
    function(&format!("() locals({types})"), "    builtin.ret;")
}

/// Prints `op` in the custom or the generic form.
fn print(context: &Rc<Context>, op: OperationRef, generic: bool) -> String {
    let flags = OpPrintingFlags {
        generic,
        ..Default::default()
    };
    let mut printer = AsmPrinter::root(context.clone(), &flags);
    OpPrinter::print(&*op.borrow(), &mut printer);
    let document = printer.finish();
    let text = format!("{document}");
    drop(document);
    text
}

fn on_a_256_kib_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(f)
        .expect("spawn a thread")
        .join()
        .expect("the operation prints without overflowing the stack");
}

/// Parses `text`, prints it in both forms, and checks that each form holds `marker` at least
/// `count` times. (The text is not parsed again: parsing a function of 20,000 blocks takes
/// seconds in an unoptimized build.)
fn prints_in_both_forms(text: String, marker: &'static str, count: usize) {
    on_a_256_kib_stack(move || {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
        let op = parse_any(ParserConfig::new(context.clone()), "wide.runic", text.as_str())
            .into_result()
            .unwrap_or_else(|error| panic!("{}", describe(&error)));
        let custom = print(&context, op, false);
        assert!(custom.matches(marker).count() >= count, "{marker} in the custom form");
        let generic = print(&context, op, true);
        assert!(generic.matches(marker).count() >= count, "{marker} in the generic form");
        drop(context);
    });
}

#[test]
fn a_switch_of_20000_cases_prints() {
    prints_in_both_forms(switch(WIDE), "^b", WIDE);
}

#[test]
fn an_index_switch_of_many_cases_prints() {
    prints_in_both_forms(index_switch(INDEX_SWITCH_CASES), "scf.yield", INDEX_SWITCH_CASES);
}

#[test]
fn a_function_of_20000_parameters_prints() {
    prints_in_both_forms(parameters(WIDE), "i32", WIDE);
}

#[test]
fn a_function_of_20000_locals_prints() {
    prints_in_both_forms(locals(WIDE), "i32", WIDE);
}

#[test]
fn an_array_attribute_of_20000_entries_prints() {
    on_a_256_kib_stack(|| {
        let values = (0..WIDE as i32).collect::<Vec<_>>();
        let rendered = format!("{}", Array::from(values.clone()).render());
        assert_eq!(rendered.matches(", ").count(), WIDE - 1);

        let context = Rc::new(Context::default());
        let text = function("()", "    builtin.ret;");
        let op = parse_any(ParserConfig::new(context.clone()), "wide.runic", text.as_str())
            .into_result()
            .unwrap_or_else(|error| panic!("{}", describe(&error)));
        let mut ret = op.borrow().regions().front().get().unwrap().entry().back().unwrap();
        let array = context.create_attribute::<I32ArrayAttr, _>(Array::from(values));
        ret.borrow_mut().set_attribute("wide", array);
        let generic = print(&context, op, true);
        let expected = format!(
            "wide = #builtin.array_i32<{}>",
            (0..WIDE).map(|i| format!("{i}")).collect::<Vec<_>>().join(", ")
        );
        assert!(generic.contains(&expected), "the generic form prints the array");
        drop(context);
    });
}
