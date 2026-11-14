//! The rewrite of every type an operation holds, and the branch and region-terminator verifiers,
//! over the corpora this crate's tests use.
//!
//! Corpora: the fixtures of `drive.rs`, the random programs of `cse_model.rs` and
//! `local2reg_model.rs`, the assert positions of `tests.rs`, and the golden files of `scf`.

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use runic_core::{
    AttributeRef, Context, ElementRewrite, Int, Operation, OperationRef, Report, SymbolTable, Type,
    TypeMapper, Value, WalkResult,
    dialects::builtin::{Function, Module},
    parse::{self, ParserConfig},
    pass::{Nesting, PassManager},
    recursively_replace_elements_in,
    testing::text::{Rng, describe},
};

use super::{cse_model, drive, interp, local2reg_model};

include!("corpora_golden.rs");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Root {
    Module,
    Function,
}

struct Case {
    name: String,
    text: String,
    root: Root,
    /// Argument vectors for the interpreter, when `@f` (or the function) can be run.
    args: Vec<Vec<u64>>,
}

fn corpus() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut rng = Rng(super::seed(0x5e3e_c0de));
    for (name, body) in drive::FIXTURES {
        let addrs = [0x1000u64, 0x1000, 0x2000];
        let args = (0..6)
            .map(|_| {
                vec![
                    rng.below(2),
                    rng.below(2),
                    rng.below(4),
                    rng.below(4),
                    rng.pick(&addrs),
                    rng.pick(&addrs),
                    rng.below(2) * 4,
                    rng.below(256),
                ]
            })
            .collect();
        cases.push(Case {
            name: format!("drive: {name}"),
            text: drive::module(body),
            root: Root::Module,
            args,
        });
    }
    for case in 0..120 {
        let len = 4 + rng.below(28) as usize;
        let program = cse_model::generate(&mut rng, len);
        let addrs = [0x1000u64, 0x1000, 0x2000, 0x3000];
        let args = (0..4)
            .map(|_| {
                vec![
                    rng.pick(&addrs),
                    rng.pick(&addrs),
                    rng.pick(&addrs),
                    rng.below(3),
                    rng.below(3),
                    rng.below(2) * 4,
                    rng.below(256),
                ]
            })
            .collect();
        cases.push(Case {
            name: format!("cse_model {case}"),
            text: program.text,
            root: Root::Module,
            args,
        });
    }
    for case in 0..120 {
        let program = local2reg_model::generate(&mut rng);
        let args = (0..4)
            .map(|_| vec![rng.below(5), rng.below(5), rng.below(2), rng.below(2), 0x1000])
            .collect();
        cases.push(Case {
            name: format!("local2reg_model {case}"),
            text: program.text,
            root: Root::Module,
            args,
        });
    }
    for (name, body) in super::POSITIONS {
        cases.push(Case {
            name: format!("assert position: {name}"),
            text: format!(
                "builtin.function public extern(\"C\") @f(%c: i1, %d: i1, %a: i32, %b: i32, %p: \
                 ptr<i32>) -> i32 {{\n{body}\n}};"
            ),
            root: Root::Function,
            args: Vec::new(),
        });
    }
    for (name, text) in GOLDEN {
        cases.push(Case {
            name: format!("golden: {name}"),
            text: text.to_string(),
            root: Root::Function,
            args: Vec::new(),
        });
    }
    cases
}

fn new_context() -> Rc<Context> {
    use runic_dialect_arith as _;
    use runic_dialect_mem as _;
    use runic_dialect_scf as _;
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    context
}

/// Parses a case; `verify` is whether the parser verifies.
fn parse_case(
    case: &Case,
    text: &str,
    verify: bool,
) -> Result<(Rc<Context>, OperationRef), String> {
    let context = new_context();
    let config = ParserConfig::new(context.clone()).verify_after_parse(verify);
    let root = match case.root {
        Root::Module => parse::parse::<Module>(config, "case", text)
            .into_result()
            .map(|module| module.as_operation_ref()),
        Root::Function => parse::parse::<Function>(config, "case", text)
            .into_result()
            .map(|function| function.as_operation_ref()),
    };
    root.map(|root| (context, root)).map_err(|err| describe(&err))
}

fn all_ops(root: OperationRef) -> Vec<OperationRef> {
    let mut ops = Vec::new();
    root.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
    ops
}

/// Every operation under `root` that does not verify by itself, with its message.
fn rejected(root: OperationRef) -> Vec<(String, String)> {
    all_ops(root)
        .into_iter()
        .filter_map(|op| {
            let op = op.borrow();
            op.verify().err().map(|err| (op.name().to_string(), describe(&err)))
        })
        .collect()
}

fn attribute_handles(root: OperationRef) -> Vec<AttributeRef> {
    let mut handles = Vec::new();
    for op in all_ops(root) {
        let _ = op.borrow().walk_attributes(&mut |_, attr| {
            handles.push(*attr);
            WalkResult::Continue(())
        });
    }
    handles
}

/// The function the interpreter runs: `@f` of a module, or the function itself.
fn function_text(case: &Case, root: OperationRef) -> String {
    match case.root {
        Root::Function => root.borrow().to_string(),
        Root::Module => {
            let module = root.borrow();
            let module = module.downcast_ref::<Module>().unwrap();
            let f = module.symbol_manager().lookup_op("f").unwrap();
            f.borrow().to_string()
        }
    }
}

type Map = fn(&Context, Type) -> Type;

/// The value of a constant at its new type: truncated or sign-extended to the new width.
fn carry(value: &Int, _old: Type, new: Type) -> Result<Int, Report> {
    let width = Int::width_for_type(new).ok_or_else(|| Report::msg("not an integer type"))?;
    Ok(value.sext_or_trunc(width))
}

fn rewrite(context: &Rc<Context>, root: OperationRef, map: Map) -> Result<bool, Report> {
    let mut rewrite =
        ElementRewrite::types(TypeMapper::new(context, map)).with_int_constants(carry);
    recursively_replace_elements_in(root, &mut rewrite)
}

/// The identity changes nothing: it answers `false`, the text is the same, and every
/// attribute handle is the one it was.
#[test]
fn the_identity_keeps_every_handle() {
    let mut parsed = 0;
    for case in corpus() {
        let Ok((context, root)) = parse_case(&case, &case.text, true) else {
            continue;
        };
        parsed += 1;
        let before = root.borrow().to_string();
        let handles = attribute_handles(root);
        let changed = rewrite(&context, root, |_, ty| ty).unwrap();
        assert!(!changed, "{}", case.name);
        assert_eq!(root.borrow().to_string(), before, "{}", case.name);
        let after = attribute_handles(root);
        assert_eq!(handles.len(), after.len());
        assert!(
            handles.iter().zip(&after).all(|(a, b)| AttributeRef::ptr_eq(a, b)),
            "{}: an attribute handle changed",
            case.name
        );
    }
    assert!(parsed > 250, "{parsed}");
}

const MAPS: [(&str, Map); 5] = [
    ("i64 -> i32", |cx, ty| if ty == cx.i64() { cx.i32() } else { ty }),
    ("i32 -> i16", |cx, ty| if ty == cx.i32() { cx.i16() } else { ty }),
    ("i32 -> i64", |cx, ty| if ty == cx.i32() { cx.i64() } else { ty }),
    ("isize -> i64", |cx, ty| if ty == cx.isize() { cx.i64() } else { ty }),
    ("isize -> i32", |cx, ty| if ty == cx.isize() { cx.i32() } else { ty }),
];

/// Every map over every case: the rewrite succeeds; every case that verifies afterwards prints
/// to text that parses back to itself; and the operations rejected afterwards are exactly those
/// whose own constraint fixes the type the map changed (a selector or a result fixed at 32 bits,
/// an operand or a result fixed at `isize`), which only a conversion pattern can rewrite. They
/// are printed with `--nocapture`.
#[test]
fn every_map_over_the_corpora() {
    // operation -> (count, an example)
    let mut by_map: BTreeMap<&str, BTreeMap<String, (usize, String)>> = BTreeMap::new();
    let mut rewrite_errors: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    let mut reparse_failures: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut clean = BTreeMap::<&str, usize>::new();
    let mut total = 0;
    for case in corpus() {
        if parse_case(&case, &case.text, true).is_err() {
            continue;
        }
        total += 1;
        for (map_name, map) in MAPS {
            let (context, root) = parse_case(&case, &case.text, true).unwrap();
            if let Err(err) = rewrite(&context, root, map) {
                let message = describe(&err);
                *rewrite_errors.entry(map_name).or_default().entry(message).or_default() += 1;
                continue;
            }
            let failures = rejected(root);
            if failures.is_empty() {
                *clean.entry(map_name).or_default() += 1;
                // the printed form parses back to itself and verifies
                let text = root.borrow().to_string();
                match parse_case(&case, &text, true) {
                    Ok((_, again)) => {
                        let again = again.borrow().to_string();
                        if again != text {
                            reparse_failures
                                .entry(map_name)
                                .or_default()
                                .push(format!("{}: not a fixpoint", case.name));
                        }
                    }
                    Err(err) => reparse_failures
                        .entry(map_name)
                        .or_default()
                        .push(format!("{}: {err}\n{text}", case.name)),
                }
            }
            for (op, message) in failures {
                let entry = by_map
                    .entry(map_name)
                    .or_default()
                    .entry(op)
                    .or_insert_with(|| (0, format!("[{}] {message}", case.name)));
                entry.0 += 1;
            }
        }
    }
    std::println!("cases: {total}");
    let expected_rejections = |map_name: &str| -> &[&str] {
        match map_name {
            "i32 -> i16" | "i32 -> i64" => &["cf.switch", "scf.index_switch", "test.effects"],
            "isize -> i64" | "isize -> i32" => &[
                "mem.alignof",
                "mem.gep",
                "mem.inttoptr",
                "mem.memcpy",
                "mem.memmove",
                "mem.memset",
                "mem.offsetof",
                "mem.ptrtoint",
                "mem.sizeof",
            ],
            _ => &[],
        }
    };
    for (map_name, _) in MAPS {
        std::println!(
            "== {map_name}: {} cases verify afterwards",
            clean.get(map_name).unwrap_or(&0)
        );
        for (op, (count, example)) in by_map.get(map_name).into_iter().flatten() {
            std::println!("   rejected {op} x{count}: {example}");
        }
        for (message, count) in rewrite_errors.get(map_name).into_iter().flatten() {
            std::println!("   rewrite error x{count}: {message}");
        }
        for failure in reparse_failures.get(map_name).into_iter().flatten().take(3) {
            std::println!("   reparse: {failure}");
        }
    }
    assert!(total > 250, "{total}");
    assert!(rewrite_errors.is_empty(), "rewrite errors: {rewrite_errors:?}");
    assert!(
        reparse_failures.is_empty(),
        "printed forms that do not parse back: {reparse_failures:?}"
    );
    for (map_name, _) in MAPS {
        let rejected: Vec<&str> =
            by_map.get(map_name).into_iter().flatten().map(|(op, _)| op.as_str()).collect();
        assert_eq!(rejected, expected_rejections(map_name), "operations rejected after {map_name}");
    }
}

/// A width-preserving map (`isize -> i64`) does not change what a program computes, where
/// the interpreter can run it, and the rewritten module verifies as a whole.
#[test]
fn a_width_preserving_map_keeps_what_programs_compute() {
    let map: Map = |cx, ty| if ty == cx.isize() { cx.i64() } else { ty };
    let mut interpreted = 0;
    let mut changed_cases = 0;
    let mut unverified = Vec::new();
    for case in corpus() {
        if case.args.is_empty() {
            continue;
        }
        let Ok((context, root)) = parse_case(&case, &case.text, true) else {
            continue;
        };
        let before = function_text(&case, root);
        if rewrite(&context, root, map).unwrap() {
            changed_cases += 1;
        }
        if let Err(err) = root.borrow().recursively_verify() {
            unverified.push(format!("{}: {}", case.name, describe(&err)));
            continue;
        }
        let after = function_text(&case, root);
        let (Ok(f), Ok(g)) = (interp::parse_function(&before), interp::parse_function(&after))
        else {
            continue;
        };
        for args in &case.args {
            match (interp::run(&f, args, 200), interp::run(&g, args, 200)) {
                (Ok(x), Ok(y)) => {
                    interpreted += 1;
                    assert_eq!(x, y, "{}: {args:x?}\n{before}\n{after}", case.name);
                }
                (x, y) => assert_eq!(x.is_err(), y.is_err(), "{}", case.name),
            }
        }
    }
    std::println!(
        "interpreted {interpreted} runs; {changed_cases} cases changed; unverified: {}",
        unverified.len()
    );
    for line in unverified.iter().take(5) {
        std::println!("   {line}");
    }
    assert!(interpreted > 300, "{interpreted}");
    assert!(changed_cases > 100, "{changed_cases}");
}

// ---------------------------------------------------------------------------------------------
// The branch verifiers
// ---------------------------------------------------------------------------------------------

/// Splits `s` at the commas that are not inside `<>`, `()` or `[]`.
fn split_top(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in s.chars() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(current.trim().to_string());
                current.clear();
                continue;
            }
            _ => (),
        }
        current.push(c);
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    parts
}

/// `values : types` with one operand dropped (the last) or added (the first again).
fn mutate_list(
    values: &[String],
    types: &[String],
    add: bool,
) -> Option<(Vec<String>, Vec<String>)> {
    if values.len() != types.len() || values.is_empty() {
        return None;
    }
    let (mut values, mut types) = (values.to_vec(), types.to_vec());
    if add {
        values.insert(0, values[0].clone());
        types.insert(0, types[0].clone());
    } else {
        values.pop();
        types.pop();
    }
    Some((values, types))
}

/// Every text one count mutation away from `text`: (what, mutated text).
fn count_mutations(text: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let with_line = |index: usize, new: String| -> String {
        let mut copy: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        copy[index] = new;
        copy.join("\n")
    };
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('^') {
            // a block header: one more argument, which no branch to it provides. A block no
            // branch goes to (and the entry block of a region, which is entered by its
            // operation) is left alone: nothing disagrees with it.
            let name: String = trimmed
                .chars()
                .take_while(|c| *c == '^' || c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let branched_to = lines.iter().enumerate().any(|(other, l)| {
                other != index
                    && l.trim_start().starts_with("cf.")
                    && l.match_indices(name.as_str()).any(|(at, _)| {
                        !l[at + name.len()..]
                            .starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                    })
            });
            if !branched_to {
                continue;
            }
            let new = if line.contains('(') {
                let close = line.rfind(')').unwrap();
                format!("{}, %zz9: i1{}", &line[..close], &line[close..])
            } else {
                line.replacen(':', "(%zz9: i1):", 1)
            };
            out.push((format!("line {index}: a block argument added"), with_line(index, new)));
            continue;
        }
        if trimmed.starts_with("cf.") {
            // each successor use with arguments: `^name(values : types)`
            let bytes = line.as_bytes();
            let mut at = 0;
            while let Some(found) = line[at..].find('^') {
                let start = at + found;
                let mut end = start + 1;
                while end < bytes.len()
                    && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_')
                {
                    end += 1;
                }
                at = end;
                if end >= bytes.len() || bytes[end] != b'(' {
                    continue;
                }
                let Some(close) = line[end..].find(')') else {
                    continue;
                };
                let close = end + close;
                let inside = &line[end + 1..close];
                let Some((values, types)) = inside.split_once(" : ") else {
                    continue;
                };
                let (values, types) = (split_top(values), split_top(types));
                for add in [false, true] {
                    let Some((values, types)) = mutate_list(&values, &types, add) else {
                        continue;
                    };
                    let args = if values.is_empty() {
                        String::new()
                    } else {
                        format!("({} : {})", values.join(", "), types.join(", "))
                    };
                    let new = format!("{}{}{}", &line[..end], args, &line[close + 1..]);
                    let what = if add { "added" } else { "dropped" };
                    out.push((
                        format!("line {index}: an operand {what} for a successor of `{trimmed}`"),
                        with_line(index, new),
                    ));
                }
                at = close;
            }
        }
        for name in ["scf.yield", "scf.condition"] {
            let Some(rest) = trimmed.strip_prefix(name) else {
                continue;
            };
            let Some((values, types)) = rest.trim_end_matches(';').split_once(" : ") else {
                continue;
            };
            let types = types.trim().trim_start_matches('(').trim_end_matches(')');
            let (values, types) = (split_top(values), split_top(types));
            let indent = &line[..line.len() - trimmed.len()];
            for add in [false, true] {
                if !add && name == "scf.condition" && values.len() < 2 {
                    continue;
                }
                let Some((values, types)) = mutate_list(&values, &types, add) else {
                    continue;
                };
                let new = if values.is_empty() {
                    format!("{indent}{name} : ();")
                } else {
                    format!("{indent}{name} {} : ({});", values.join(", "), types.join(", "))
                };
                let what = if add { "added" } else { "dropped" };
                out.push((
                    format!("line {index}: an operand {what} on `{trimmed}`"),
                    with_line(index, new),
                ));
            }
        }
    }
    out
}

/// Every count mutation of a branch or of a block's arguments is rejected (by the parser or
/// by verification); the ones that are not are listed.
#[test]
fn count_mutations_are_rejected() {
    let (mut total, mut by_parser, mut by_branch_verifier, mut by_other) = (0, 0, 0, 0);
    let mut accepted = Vec::new();
    let mut other_examples = BTreeMap::<String, String>::new();
    for case in corpus() {
        if case.name.starts_with("cse_model") || case.name.starts_with("local2reg_model") {
            // 240 programs with the same few shapes: every eighth is enough
            let n: usize = case.name.rsplit(' ').next().unwrap().parse().unwrap();
            if !n.is_multiple_of(8) {
                continue;
            }
        }
        if parse_case(&case, &case.text, true).is_err() {
            continue;
        }
        for (what, text) in count_mutations(&case.text) {
            total += 1;
            // the context is kept: an operation handle must not outlive its context
            let (_context, root) = match parse_case(&case, &text, false) {
                Ok(parsed) => parsed,
                Err(_) => {
                    by_parser += 1;
                    continue;
                }
            };
            let failures = rejected(root);
            if failures.is_empty() {
                accepted.push(format!("{}: {what}\n{text}", case.name));
            } else if failures.iter().any(|(_, m)| m.contains("successor")) {
                by_branch_verifier += 1;
            } else {
                by_other += 1;
                let (op, message) = &failures[0];
                other_examples
                    .entry(op.clone())
                    .or_insert_with(|| format!("[{}] {what}: {message}", case.name));
            }
        }
    }
    std::println!(
        "count mutations: {total}; rejected by the parser {by_parser}, by a successor check \
         {by_branch_verifier}, by another verifier only {by_other}; accepted {}",
        accepted.len()
    );
    for (op, example) in &other_examples {
        std::println!("   other: {op}: {example}");
    }
    for line in accepted.iter().take(6) {
        std::println!("   ACCEPTED: {line}");
    }
    assert!(by_branch_verifier > 200, "{by_branch_verifier}");
    assert!(accepted.is_empty(), "{} mutations were accepted", accepted.len());
}

/// Changing the type of one block argument makes every branch that forwards an operand to
/// it fail its own verification, with a message naming the successor; changing a result type of
/// a region-branch operation makes a terminator that returns to it fail.
#[test]
fn type_mutations_are_rejected() {
    use runic_core::RegionBranchOpInterface;

    let (mut block_edges, mut result_edges, mut entry_edges) = (0, 0, 0);
    let mut accepted = Vec::new();
    for case in corpus() {
        if case.name.starts_with("cse_model") || case.name.starts_with("local2reg_model") {
            let n: usize = case.name.rsplit(' ').next().unwrap().parse().unwrap();
            if !n.is_multiple_of(8) {
                continue;
            }
        }
        let Ok((context, root)) = parse_case(&case, &case.text, true) else {
            continue;
        };
        let odd = context.int_type(7);
        for op_ref in all_ops(root) {
            // (1) block successors
            let successors = op_ref.borrow().num_successors();
            for index in 0..successors {
                let block = op_ref.borrow().successor(index).successor();
                let args: Vec<_> = block.borrow().arguments().to_vec();
                for mut arg in args {
                    let old = arg.borrow().ty();
                    arg.borrow_mut().set_type(odd);
                    let result = op_ref.borrow().verify();
                    arg.borrow_mut().set_type(old);
                    block_edges += 1;
                    match result {
                        Err(err) if describe(&err).contains(&format!("successor {index}")) => (),
                        Err(err) => accepted.push(format!(
                            "{}: {} rejected without naming successor {index}: {}",
                            case.name,
                            op_ref.borrow().name(),
                            describe(&err)
                        )),
                        Ok(()) => accepted.push(format!(
                            "{}: {} accepts a retyped argument of successor {index}",
                            case.name,
                            op_ref.borrow().name()
                        )),
                    }
                }
            }
            // (2) region branches: results, and the entry arguments of each region
            if !op_ref.borrow().implements::<dyn RegionBranchOpInterface>() {
                continue;
            }
            let terminators: Vec<OperationRef> = {
                let op = op_ref.borrow();
                let mut found = Vec::new();
                for region in op.regions().iter() {
                    for block in region.body().iter() {
                        if let Some(last) = block.body().back().as_pointer() {
                            found.push(last);
                        }
                    }
                }
                found
            };
            let results: Vec<_> = op_ref.borrow().results().iter().copied().collect();
            for mut result in results {
                let old = result.borrow().ty();
                result.borrow_mut().set_type(odd);
                let failing = terminators
                    .iter()
                    .filter(|t| {
                        t.borrow().verify().is_err_and(|err| describe(&err).contains("successor"))
                    })
                    .count();
                let own = op_ref.borrow().verify().is_err();
                result.borrow_mut().set_type(old);
                result_edges += 1;
                if failing == 0 {
                    accepted.push(format!(
                        "{}: no terminator in {} rejects a retyped result (the operation itself \
                         rejects: {own})",
                        case.name,
                        op_ref.borrow().name()
                    ));
                }
            }
            let entry_args: Vec<_> = {
                let op = op_ref.borrow();
                let mut found = Vec::new();
                for region in op.regions().iter() {
                    if let Some(entry) = region.entry_block_ref() {
                        found.extend(entry.borrow().arguments().iter().copied());
                    }
                }
                found
            };
            for mut arg in entry_args {
                let old = arg.borrow().ty();
                arg.borrow_mut().set_type(odd);
                let own = op_ref.borrow().verify();
                let failing = terminators
                    .iter()
                    .filter(|t| {
                        t.borrow().verify().is_err_and(|err| describe(&err).contains("successor"))
                    })
                    .count();
                arg.borrow_mut().set_type(old);
                entry_edges += 1;
                let own_names_successor =
                    own.as_ref().is_err_and(|err| describe(err).contains("successor"));
                if !own_names_successor && failing == 0 {
                    accepted.push(format!(
                        "{}: a retyped entry argument of {} is rejected by no successor check \
                         (own verifier: {:?})",
                        case.name,
                        op_ref.borrow().name(),
                        own.err().map(|err| describe(&err))
                    ));
                }
            }
        }
        // everything was restored
        root.borrow()
            .recursively_verify()
            .unwrap_or_else(|err| panic!("{}", describe(&err)));
    }
    std::println!(
        "type mutations: {block_edges} block arguments, {result_edges} results, {entry_edges} \
         region entry arguments; not rejected by a successor check: {}",
        accepted.len()
    );
    let mut distinct = BTreeSet::new();
    for line in &accepted {
        let key: String = line.split(": ").skip(1).collect::<Vec<_>>().join(": ");
        if distinct.insert(key.chars().take(120).collect::<String>()) {
            std::println!("   {line}");
        }
    }
    assert!(block_edges > 20 && result_edges > 50 && entry_edges > 20);
    assert!(accepted.is_empty(), "{} type mutations escaped", accepted.len());
}

/// Every pass of this crate over the golden `_before` functions, with the verifier on: no
/// pass produces IR the verifier rejects.
#[test]
fn every_pass_over_the_golden_functions_verifies() {
    let mut runs = 0;
    let mut failures = Vec::new();
    for case in corpus() {
        if !case.name.starts_with("golden") && !case.name.starts_with("assert position") {
            continue;
        }
        if parse_case(&case, &case.text, true).is_err() {
            continue;
        }
        for (pass, make) in drive::PASSES {
            let (context, root) = parse_case(&case, &case.text, true).unwrap();
            let mut manager = PassManager::on::<Function>(context.clone(), Nesting::Implicit);
            manager.add_pass(make()).unwrap();
            manager.enable_verifier(true);
            runs += 1;
            if let Err(err) = manager.run(root) {
                failures.push(format!("{pass} on {}: {}", case.name, describe(&err)));
                continue;
            }
            if let Err(err) = root.borrow().recursively_verify() {
                failures.push(format!("{pass} on {}: after: {}", case.name, describe(&err)));
            }
        }
    }
    std::println!("{runs} pass runs; {} failures", failures.len());
    for failure in &failures {
        std::println!("   {failure}");
    }
    assert!(runs > 200, "{runs}");
    assert!(failures.is_empty());
}

// ---------------------------------------------------------------------------------------------
// Region terminators in the wrong place
// ---------------------------------------------------------------------------------------------

/// Every text made from `text` by writing one region terminator as another kind of terminator
/// over the same operands.
fn terminator_swaps(text: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let indent = &line[..line.len() - trimmed.len()];
        let rest = ["scf.yield", "scf.condition", "scf.break 1", "scf.continue 1", "builtin.ret"]
            .into_iter()
            .find_map(|name| trimmed.strip_prefix(name).map(|rest| (name, rest)));
        let Some((name, rest)) = rest else { continue };
        if !rest.starts_with(' ') && !rest.starts_with(';') {
            continue;
        }
        for other in [
            "scf.yield",
            "scf.condition",
            "scf.break 1",
            "scf.break 9",
            "scf.continue 1",
            "scf.continue 9",
            "builtin.ret",
        ] {
            if other == name {
                continue;
            }
            let mut copy: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
            copy[index] = format!("{indent}{other}{rest}");
            out.push((format!("line {index}: `{trimmed}` as {other}"), copy.join("\n")));
        }
        // a well-typed `scf.condition` in this place: the function's `%c: i1` leads the operands
        if name != "scf.condition"
            && text.contains("%c: i1")
            && let Some((values, types)) = rest.trim_end_matches(';').split_once(" : ")
        {
            let types = types.trim().trim_start_matches('(').trim_end_matches(')');
            let (values, types) = (values.trim(), types.trim());
            let new = if values.is_empty() {
                format!("{indent}scf.condition %c : (i1);")
            } else {
                format!("{indent}scf.condition %c, {values} : (i1, {types});")
            };
            let mut copy: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
            copy[index] = new;
            out.push((
                format!("line {index}: `{trimmed}` as a well-typed scf.condition"),
                copy.join("\n"),
            ));
        }
    }
    out
}

/// No text made by writing one region terminator as another makes verification panic: each is
/// rejected, or accepted where it is consistent.
#[test]
fn terminators_swapped_for_one_another_do_not_panic() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut panics = BTreeMap::<String, (usize, String)>::new();
    let (mut total, mut accepted, mut rejected_count) = (0, 0, 0);
    let mut accepted_kinds = BTreeMap::<String, usize>::new();
    for case in corpus() {
        if case.name.starts_with("cse_model") || case.name.starts_with("local2reg_model") {
            let n: usize = case.name.rsplit(' ').next().unwrap().parse().unwrap();
            if !n.is_multiple_of(6) {
                continue;
            }
        }
        for (what, text) in terminator_swaps(&case.text) {
            total += 1;
            let result =
                catch_unwind(AssertUnwindSafe(|| parse_case(&case, &text, true).map(|_| ())));
            match result {
                Ok(Ok(())) => {
                    accepted += 1;
                    let kind = what.split("` as ").nth(1).unwrap_or("").to_string();
                    let from = what
                        .split('`')
                        .nth(1)
                        .unwrap_or("")
                        .split(' ')
                        .next()
                        .unwrap_or("")
                        .to_string();
                    *accepted_kinds.entry(format!("{from} -> {kind}")).or_insert(0usize) += 1;
                }
                Ok(Err(_)) => rejected_count += 1,
                Err(panic) => {
                    let message = panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "a panic".to_string());
                    let entry = panics
                        .entry(message)
                        .or_insert_with(|| (0, format!("{}: {what}", case.name)));
                    entry.0 += 1;
                }
            }
        }
    }
    std::panic::set_hook(hook);
    std::println!(
        "terminator swaps: {total}; rejected {rejected_count}, accepted {accepted}, panicked {}",
        panics.values().map(|(n, _)| n).sum::<usize>()
    );
    for (message, (count, example)) in &panics {
        std::println!("   PANIC x{count}: {message}\n      e.g. {example}");
    }
    std::println!("   accepted: {accepted_kinds:?}");
    assert!(
        panics.is_empty(),
        "verification panicked: {:?}",
        panics.keys().collect::<Vec<_>>()
    );
    assert!(total > 500, "{total}");
}
