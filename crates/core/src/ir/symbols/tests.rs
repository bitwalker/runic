//! Symbol references that do not resolve: the corpus of `testing::calls` with one name left
//! dangling, through the parser, the verifier, and every walker, accessor, printer and query on
//! operations; and symbols removed from built IR.

extern crate std;

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    CallOpInterface, CallableOpInterface, Context, Operation, OperationRef, SymbolTable,
    WalkResult,
    effects::MemoryEffect,
    parse::{self, ParserConfig},
    testing::{
        calls::{CORPUS, mutations, names_symbol},
        text::{describe, print_generic},
    },
};

fn parse_any(source: &str, verify: bool) -> Result<(Rc<Context>, OperationRef), String> {
    let context = Rc::new(Context::default());
    let config = ParserConfig::new(context.clone()).verify_after_parse(verify);
    match parse::parse_any(config, "dangling", source).into_result() {
        Ok(op) => Ok((context, op)),
        Err(report) => Err(describe(&report)),
    }
}

fn core_corpus() -> impl Iterator<Item = &'static str> {
    // `arith`, `cf` and `scf` are not dialects of this crate
    CORPUS.into_iter().filter(|source| !source.contains("arith."))
}

#[test]
fn the_corpus_parses_and_verifies() {
    assert!(core_corpus().count() >= 5);
    for source in core_corpus() {
        parse_any(source, true).unwrap_or_else(|err| panic!("{err}\n{source}"));
    }
}

/// Everything a walker, accessor, printer or query does with an operation, on every operation
/// under `root`. Returns how many references did not resolve.
fn exercise(root: OperationRef) -> usize {
    let mut unresolved = 0;
    let root = root.borrow();
    let _ = root.to_string();
    let _ = print_generic(&root);
    root.prewalk_all(|op: &Operation| {
        let _ = Operation::walk_symbol_refs(op, |reference| {
            match reference {
                Ok(symbol_use) => {
                    let attr = symbol_use.borrow().attr;
                    let attr = attr.borrow();
                    let _ = attr.resolve();
                    let _ = attr.resolve_callable();
                }
                Err(attr) => {
                    unresolved += 1;
                    let attr = attr.borrow();
                    assert!(attr.resolve().is_none() || attr.try_user().is_some());
                    let _ = attr.resolve_callable();
                    let _ = attr.path().to_string();
                }
            }
            WalkResult::Continue(())
        });
        let _ = op.walk_symbol_uses(|_| WalkResult::Continue(()));
        let _ = op.all_symbol_uses().count();
        let _ = op.is_memory_effect_free();
        let _ = op.would_be_trivially_dead();
        let _ = op.is_trivially_dead();
        let _ = op.get_effects_recursively::<MemoryEffect>();
        let _ = op.has_memory_effect(MemoryEffect::Read);
        let _ = op.has_single_memory_effect(MemoryEffect::Read);
        let _ = op.verify();
        let _ = op.to_string();
        if let Some(call) = op.as_trait::<dyn CallOpInterface>() {
            let _ = call.callable_for_callee();
            let _ = call.resolve();
            let _ = call.possible_callees();
            let _ = call.callee_signature();
            let _ = call.arguments().len();
        }
        if let Some(symbol) = op.as_symbol() {
            let _ = symbol.iter_uses().count();
            let _ = symbol.path();
        }
        if let Some(callable) = op.as_trait::<dyn CallableOpInterface>() {
            let _ = callable.declared_effects();
            let _ = callable.signature();
        }
    });
    let _ = root.recursively_verify();
    unresolved
}

/// Each source of the corpus with one declaration deleted or renamed, or one reference renamed:
/// a diagnostic whenever a name is left dangling, and never a panic, with verification on; with
/// verification off, every walker, accessor, printer and query runs without a panic and the
/// verifier then reports the reference.
#[test]
fn fuzz_dangling_symbols_are_diagnosed_and_never_panic() {
    let mut cases = 0;
    let mut must_fail = 0;
    let mut messages = alloc::collections::BTreeMap::<String, usize>::new();
    let mut problems = Vec::new();
    for source in core_corpus() {
        for (what, text, fails) in mutations(source) {
            cases += 1;
            let verified = std::panic::catch_unwind(|| parse_any(&text, true).map(|_| ()));
            match verified {
                Err(_) => problems.push(format!("PANIC (verified) {what}:\n{text}")),
                Ok(Ok(())) if fails => {
                    problems.push(format!("ACCEPTED {what}:\n{text}"));
                }
                Ok(Ok(())) => (),
                Ok(Err(err)) => {
                    let kind = if err.contains("does not resolve") && err.contains("symbol '") {
                        "symbol '..' does not resolve"
                    } else if err.contains("invalid builtin.function_alias") {
                        "invalid builtin.function_alias"
                    } else {
                        problems.push(format!("OTHER ERROR {what}: {err}\n{text}"));
                        "other"
                    };
                    *messages.entry(kind.to_string()).or_default() += 1;
                }
            }
            must_fail += usize::from(fails);

            let unverified = std::panic::catch_unwind(|| {
                let (_context, root) = match parse_any(&text, false) {
                    Ok(parsed) => parsed,
                    Err(err) => return Err(err),
                };
                let unresolved = exercise(root);
                let verifies = root.borrow().recursively_verify().is_ok();
                Ok((unresolved, verifies))
            });
            match unverified {
                Err(_) => problems.push(format!("PANIC (unverified) {what}:\n{text}")),
                Ok(Err(err)) => problems.push(format!("UNVERIFIED PARSE FAILED {what}: {err}")),
                Ok(Ok((_, true))) if fails => {
                    problems.push(format!("VERIFIES AFTER UNVERIFIED PARSE {what}:\n{text}"))
                }
                Ok(Ok(_)) => (),
            }
        }
    }
    assert!(cases > 60 && must_fail > 50, "{cases} cases, {must_fail} must fail");
    assert!(
        problems.is_empty(),
        "{} problems of {cases} cases ({messages:?}):\n{}",
        problems.len(),
        problems.join("\n\n")
    );
    std::eprintln!("{cases} cases, {must_fail} must fail, messages: {messages:?}");
}

/// Built IR: a function that is called, referenced and aliased is removed from its symbol table.
/// Nothing panics, and each operation that named it no longer verifies.
#[test]
fn removing_a_symbol_leaves_references_that_are_diagnosed() {
    for source in core_corpus() {
        let (_context, root) = parse_any(source, true).unwrap();
        // every symbol table under the root, and the names it defines
        let mut tables = Vec::new();
        let mut visit = |op: &Operation| {
            if op.as_symbol_table().is_some() {
                let names = op
                    .regions()
                    .iter()
                    .flat_map(|region| {
                        region
                            .body()
                            .iter()
                            .flat_map(|block| {
                                block
                                    .body()
                                    .iter()
                                    .filter_map(|op| op.as_symbol().map(|s| s.name()))
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                tables.push((op.as_operation_ref(), names));
            }
        };
        visit(&root.borrow());
        root.borrow().prewalk_all(&mut visit);
        for (index, (_, names)) in tables.iter().enumerate() {
            for name in names {
                // a fresh copy for each removal
                let (_context, root) = parse_any(source, true).unwrap();
                let mut found = Vec::new();
                let mut collect = |op: &Operation| {
                    if op.as_symbol_table().is_some() {
                        found.push(op.as_operation_ref());
                    }
                };
                collect(&root.borrow());
                root.borrow().prewalk_all(&mut collect);
                let mut table = found[index];
                let symbol = {
                    let mut table = table.borrow_mut();
                    let table = table.as_trait_mut::<dyn SymbolTable>().unwrap();
                    table.remove(*name)
                };
                let Some(symbol) = symbol else { continue };
                let had_uses = symbol.borrow().iter_uses().count();
                assert_eq!(had_uses, 0, "remove clears the uses");
                // the symbol is no longer in the table, but its operation is still in the block:
                // erase it, as a pass that removes a function would
                let mut symbol_op = symbol.borrow().as_operation_ref();
                symbol_op.borrow_mut().erase();
                let outcome =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exercise(root)));
                let unresolved =
                    outcome.unwrap_or_else(|_| panic!("panic after removing {name}:\n{source}"));
                let named_elsewhere =
                    source.lines().filter(|line| names_symbol(line, &format!("@{name}"))).count()
                        > 1;
                if named_elsewhere && !source.contains("builtin.world") {
                    assert!(unresolved > 0, "no unresolved reference after removing {name}");
                    assert!(root.borrow().recursively_verify().is_err(), "{name}:\n{source}");
                }
            }
        }
    }
}
