//! Every operation of the source dialects is accounted for: `convert-to-wasm` has a pattern for it,
//! or it is a diagnostic by design, listed here with its reason and what the diagnostic says.
//!
//! The operations are those the context registers, so an operation added to one of the dialects
//! fails these tests until it is lowered or listed.

use alloc::{boxed::Box, collections::BTreeSet};
use std::{fs, path::Path};

use runic_ir::{
    conversion::ConversionPatternSet,
    pass::{Nesting, PassManager},
};

use super::*;
use crate::{
    ConvertToWasm, WasmTarget,
    convert::{WellKnownSymbols, populate_patterns},
    type_converter,
};

/// The dialects whose every operation `convert-to-wasm` must account for.
const SOURCE_DIALECTS: [&str; 5] = ["arith", "cf", "mem", "scf", "ub"];

/// The operations of `builtin` that are not function-level: they are modules, symbols and data,
/// which `convert-to-wasm` leaves as they are (a global's initializer is not converted), and the
/// cast the conversion framework itself makes and removes.
const NOT_FUNCTION_LEVEL: [&str; 9] = [
    "builtin.bytes",
    "builtin.component",
    "builtin.function_alias",
    "builtin.global_variable",
    "builtin.module",
    "builtin.segment",
    "builtin.unrealized_conversion_cast",
    "builtin.world",
    "builtin.yield",
];

/// An operation that `convert-to-wasm` reports instead of lowering, and how.
struct ByDesign {
    /// The operation.
    op: &'static str,
    /// A function body with the operation in it, in a module with the 32-bit data layout.
    body: &'static str,
    /// Why the operation is not lowered by `convert-to-wasm`.
    reason: &'static str,
    /// What the diagnostic says.
    says: &'static str,
}

/// The operations that are diagnostics by design when they reach `convert-to-wasm`.
const BY_DESIGN: [ByDesign; 5] = [
    ByDesign {
        op: "mem.alloca",
        body: "%p = mem.alloca i32;\nbuiltin.ret;",
        reason: "`wasm-lower-frames` gives each alloca a slot on the shadow stack first",
        says: "`convert-to-wasm` does not lower `mem.alloca`: run `wasm-lower-frames` first",
    },
    ByDesign {
        op: "cf.br",
        body: "cf.br ^next;\n^next:\nbuiltin.ret;",
        reason: "Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first",
        says: "unstructured control flow reached `convert-to-wasm`: run \
               `cfg-to-scf{exits=in-place}` first",
    },
    ByDesign {
        op: "cf.cond_br",
        body: "%c = arith.constant 1 : i1;\ncf.cond_br %c ^a, ^b : \
               (i1);\n^a:\nbuiltin.ret;\n^b:\nbuiltin.ret;",
        reason: "Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first",
        says: "unstructured control flow reached `convert-to-wasm`: run \
               `cfg-to-scf{exits=in-place}` first",
    },
    ByDesign {
        op: "cf.switch",
        body: "%s = arith.constant 1 : i32;\ncf.switch %s [#builtin.i32<0> -> ^a], ^b : \
               (i32);\n^a:\nbuiltin.ret;\n^b:\nbuiltin.ret;",
        reason: "Wasm control flow is structured: `cfg-to-scf{exits=in-place}` runs first",
        says: "unstructured control flow reached `convert-to-wasm`: run \
               `cfg-to-scf{exits=in-place}` first",
    },
    ByDesign {
        op: "scf.pending_exit",
        body: "\"scf.pending_exit\"() : () -> ();\nbuiltin.ret;",
        reason: "`cfg-to-scf` makes it while it runs and removes it: it never verifies",
        says: "`scf.pending_exit` is internal to `cfg-to-scf` and never appears in IR",
    },
];

/// The names of the registered operations of `dialect`.
fn operations_of(context: &Context, dialect: &str) -> BTreeSet<String> {
    context
        .registered_dialects()
        .values()
        .flat_map(|d| d.registered_ops().iter().map(|op| op.to_string()).collect::<Vec<_>>())
        .filter(|name| name.split('.').next() == Some(dialect))
        .collect()
}

/// The operations `convert-to-wasm` must account for.
fn required(context: &Context) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for dialect in SOURCE_DIALECTS {
        let found = operations_of(context, dialect);
        assert!(!found.is_empty(), "the dialect `{dialect}` registers no operation");
        names.extend(found);
    }
    let builtin = operations_of(context, "builtin");
    for name in NOT_FUNCTION_LEVEL {
        assert!(builtin.contains(name), "`{name}` is not a registered operation");
    }
    names.extend(builtin.into_iter().filter(|name| !NOT_FUNCTION_LEVEL.contains(&name.as_str())));
    names
}

/// The operations some pattern of `convert-to-wasm` is rooted at.
fn with_patterns(context: &Rc<Context>) -> BTreeSet<String> {
    let module = empty_module(context);
    let target = WasmTarget::of(module.borrow().as_operation()).unwrap();
    let mut patterns = ConversionPatternSet::new(context.clone());
    populate_patterns(&mut patterns, &type_converter(), &target, &WellKnownSymbols::default())
        .unwrap();
    patterns
        .patterns()
        .iter()
        .filter_map(|pattern| pattern.info().root_operation())
        .map(|name| name.to_string())
        .collect()
}

/// A module with the 32-bit data layout and nothing in it, in `context`.
fn empty_module(context: &Rc<Context>) -> ModuleRef {
    let text = "builtin.module public @m attributes { data_layout = #builtin.data_layout<little, \
                ptr = 32> } {};";
    parse_module_in(context, text).unwrap_or_else(|report| panic!("{}", describe(&report)))
}

/// Whether some line that is not a comment, of a lit file of the lowering, holds the operation
/// `name` (as a whole word).
fn in_a_lit_program(files: &[String], name: &str) -> bool {
    files.iter().any(|text| {
        text.lines().filter(|line| !line.trim_start().starts_with("//")).any(|line| {
            line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
                .any(|word| word == name)
        })
    })
}

/// The text of the lit files of `tests/lit/codegen/wasm`.
fn lit_files() -> Vec<String> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/lit/codegen/wasm");
    let mut files = Vec::new();
    for entry in fs::read_dir(&directory).unwrap_or_else(|e| panic!("{}: {e}", directory.display()))
    {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|extension| extension == "runic") {
            files.push(fs::read_to_string(path).unwrap());
        }
    }
    assert!(files.len() >= 20, "found only {} lit files", files.len());
    files
}

#[test]
fn every_source_operation_has_a_pattern_or_is_a_diagnostic_by_design() {
    let context = context();
    let patterned = with_patterns(&context);
    let by_design: BTreeSet<&str> = BY_DESIGN.iter().map(|entry| entry.op).collect();
    let unaccounted: Vec<String> = required(&context)
        .into_iter()
        .filter(|name| !patterned.contains(name) && !by_design.contains(name.as_str()))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "no pattern of `convert-to-wasm` is rooted at these operations, and none is listed as a \
         diagnostic by design: {unaccounted:?}"
    );
}

#[test]
fn the_diagnostics_by_design_are_registered_operations_without_a_converting_pattern() {
    let context = context();
    let registered = required(&context);
    for entry in &BY_DESIGN {
        assert!(
            registered.contains(entry.op),
            "`{}` is not a registered source operation",
            entry.op
        );
        assert!(!entry.reason.is_empty());
    }
    // `cf.br`, `cf.cond_br` and `cf.switch` have patterns that only report; the others have none.
    let patterned = with_patterns(&context);
    for op in ["mem.alloca", "scf.pending_exit"] {
        assert!(!patterned.contains(op), "`{op}` has a pattern");
    }
}

#[test]
fn every_operation_with_a_pattern_is_written_in_a_lit_program_of_the_lowering() {
    let context = context();
    let patterned = with_patterns(&context);
    let files = lit_files();
    let by_design: BTreeSet<&str> = BY_DESIGN.iter().map(|entry| entry.op).collect();
    let missing: Vec<String> = required(&context)
        .into_iter()
        .filter(|name| patterned.contains(name) && !by_design.contains(name.as_str()))
        .filter(|name| !in_a_lit_program(&files, name))
        .collect();
    assert!(
        missing.is_empty(),
        "no program of `tests/lit/codegen/wasm` writes these operations: {missing:?}"
    );
}

/// The text of a module with one function holding `body`, with the 32-bit data layout.
fn module_with(body: &str) -> String {
    format!(
        "builtin.module public @m attributes {{ data_layout = #builtin.data_layout<little, ptr = \
         32> }} {{\n    builtin.function public extern(\"wasm\") @f() {{\n{body}\n    }};\n}};"
    )
}

#[test]
fn each_diagnostic_by_design_says_what_it_is_listed_as_saying() {
    for entry in &BY_DESIGN {
        let context = context();
        let text = module_with(entry.body);
        let error = match parse_module_in(&context, &text) {
            // The operation does not even verify.
            Err(report) => describe(&report),
            Ok(module) => {
                let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
                passes.add_pass(Box::<ConvertToWasm>::default()).unwrap();
                passes.enable_verifier(true);
                error_of(passes.run(module.as_operation_ref()))
            }
        };
        assert!(
            mentions(&error, entry.says),
            "`{}` should say\n    {}\nbut the error is\n{error}",
            entry.op,
            entry.says
        );
    }
}

#[test]
fn the_readme_lists_each_diagnostic_by_design_with_its_reason() {
    let readme = include_str!("../../README.md");
    for entry in &BY_DESIGN {
        let row = readme
            .lines()
            .find(|line| line.starts_with(&format!("| `{}` |", entry.op)))
            .unwrap_or_else(|| panic!("the README has no table row for `{}`", entry.op));
        let reason = entry.reason.replace('`', "");
        assert!(
            row.replace('`', "").contains(&reason),
            "the README row of `{}` does not give the reason \"{}\":\n{row}",
            entry.op,
            entry.reason
        );
    }
}
