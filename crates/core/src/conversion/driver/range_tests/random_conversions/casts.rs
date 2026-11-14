//! The text of the cast, fuzzed; and reconciliation, modelled over random
//! DAGs and (in a graph region) random cyclic graphs of casts over several shapes with one
//! meaning.

use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
    println,
};

use super::{harness::Rng, *};
use crate::{
    conversion::reconcile_unrealized_casts,
    dialects::builtin::Module,
    parse::parse_any,
    testing::text::{print_generic, round_trip},
};

// ---------------------------------------------------------------------------------------------
// The text
// ---------------------------------------------------------------------------------------------

const TYPES: [&str; 9] = [
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "array<i8; 0>",
    "array<i32; 4>",
];
const FUNCTION_TYPES: [&str; 3] = ["() -> i32", "(i64, i64) -> i128", "(i8) -> (i8, i8)"];

fn type_list_text(types: &[String]) -> String {
    match types {
        [ty] if !ty.contains("->") => ty.clone(),
        _ => format!("({})", types.join(", ")),
    }
}

/// A module with one function of random casts: 0 to 3 inputs, 1 to 3 results, function types
/// among them, attributes on some. Returns the text in the form the printer gives.
fn random_casts(rng: &mut Rng) -> String {
    let mut values: Vec<String> = vec![];
    let mut params = vec![];
    for _ in 0..1 + rng.below(4) {
        let ty = TYPES[rng.below(TYPES.len())].to_string();
        params.push(format!("%{}: {ty}", values.len()));
        values.push(ty);
    }
    let mut body = String::new();
    for _ in 0..1 + rng.below(8) {
        let inputs = (0..rng.below(4)).map(|_| rng.below(values.len())).collect::<Vec<_>>();
        let mut results = vec![];
        for _ in 0..1 + rng.below(3) {
            results.push(if rng.chance(15) {
                FUNCTION_TYPES[rng.below(FUNCTION_TYPES.len())].to_string()
            } else {
                TYPES[rng.below(TYPES.len())].to_string()
            });
        }
        let names = (0..results.len())
            .map(|index| format!("%{}", values.len() + index))
            .collect::<Vec<_>>()
            .join(", ");
        let operands = inputs.iter().map(|v| format!("%{v}")).collect::<Vec<_>>().join(", ");
        let input_types = inputs.iter().map(|v| values[*v].clone()).collect::<Vec<_>>();
        let attributes = if rng.chance(15) {
            " attributes { note = #builtin.i32<7> }"
        } else {
            ""
        };
        body.push_str(&format!(
            "        {names} = builtin.unrealized_conversion_cast{}{operands} : {} to \
             {}{attributes};\n",
            if operands.is_empty() { "" } else { " " },
            type_list_text(&input_types),
            type_list_text(&results),
        ));
        values.extend(results);
    }
    format!(
        "builtin.module public @m {{
    builtin.function public extern(\"C\") @f({}) {{
{body}        builtin.ret;
    }};
}};",
        params.join(", ")
    )
}

fn parse_module(source: &str) -> Result<(Rc<Context>, OperationRef), String> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    match parse_any(ParserConfig::new(context.clone()), "cast", source).into_result() {
        Ok(op) => Ok((context, op)),
        Err(report) => Err(describe(&report)),
    }
}

/// Random casts print as written and round trip in the text and the generic form.
#[test]
fn random_cast_text_round_trips() {
    let mut checked = 0;
    for seed in seeds(&[0xca57_0001, 0xca57_0002]) {
        let mut rng = Rng(seed);
        for case in 0..cases(60) {
            let source = random_casts(&mut rng);
            let (context, module) = parse_module(&source)
                .unwrap_or_else(|err| panic!("seed {seed}, case {case}: {err}\n{source}"));
            let printed = module.borrow().to_string();
            assert_eq!(printed, source, "seed {seed}, case {case}");
            module
                .borrow()
                .recursively_verify()
                .unwrap_or_else(|err| panic!("seed {seed}, case {case}: {}", describe(&err)));
            let mut skipped = Vec::new();
            let failures = round_trip::<Module>(module, true, &mut skipped);
            assert!(failures.is_empty(), "seed {seed}, case {case}:\n{}", failures.join("\n\n"));
            // The generic form states result types and no property.
            module.borrow().prewalk_all(|op: &Operation| {
                if op.is::<UnrealizedConversionCast>() {
                    let generic = print_generic(op);
                    assert!(!generic.contains("<{"), "{generic}");
                    checked += 1;
                }
            });
            drop(context);
        }
    }
    println!("{checked} casts round-tripped");
}

/// Mutated cast text never panics the parser; what parses verifies and prints to text that
/// parses to the same text.
#[test]
fn mutated_cast_text_is_an_error_or_parses_to_stable_text() {
    let fragments = [
        " to ",
        " : ",
        "(",
        ")",
        ",",
        "%0",
        "%1",
        "%99",
        "i64",
        "i128",
        "()",
        "() -> i32",
        ";",
        "builtin.unrealized_conversion_cast",
        " attributes { a = #builtin.i32<1> }",
        "=",
        " ",
        "to",
        "(i64, i64)",
        "<",
        ">",
        "{",
        "}",
        "^bb1",
        "-> ",
        "%",
    ];
    let (mut parsed, mut rejected) = (0, 0);
    let mut messages = BTreeMap::<String, usize>::new();
    for seed in seeds(&[0xca57_1001, 0xca57_1002]) {
        let mut rng = Rng(seed);
        for case in 0..cases(400) {
            let source = random_casts(&mut rng);
            // Mutate within the body of the function.
            let start = source.find("{\n        ").map_or(0, |i| i + 2);
            let end = source.rfind("        builtin.ret;").unwrap_or(source.len());
            let mut text = source.clone();
            for _ in 0..1 + rng.below(3) {
                let span = end.min(text.len()) - start;
                if span < 2 {
                    break;
                }
                let at = start + rng.below(span);
                if !text.is_char_boundary(at) {
                    continue;
                }
                match rng.below(4) {
                    0 => {
                        // Delete a few characters.
                        let to = (at + 1 + rng.below(6)).min(text.len());
                        if text.is_char_boundary(to) {
                            text.replace_range(at..to, "");
                        }
                    }
                    1 | 2 => text.insert_str(at, fragments[rng.below(fragments.len())]),
                    _ => {
                        // Replace the token at a fragment boundary.
                        let fragment = fragments[rng.below(fragments.len())];
                        let to = (at + fragment.len()).min(text.len());
                        if text.is_char_boundary(to) {
                            text.replace_range(at..to, fragment);
                        }
                    }
                }
            }
            let outcome = catch_unwind(AssertUnwindSafe(|| parse_module(&text)));
            match outcome {
                Err(_) => panic!("seed {seed}, case {case}: the parser panicked on\n{text}"),
                Ok(Err(message)) => {
                    rejected += 1;
                    let short = message.chars().take(60).collect::<String>();
                    *messages.entry(short).or_default() += 1;
                }
                Ok(Ok((context, module))) => {
                    parsed += 1;
                    module.borrow().recursively_verify().unwrap_or_else(|err| {
                        panic!(
                            "seed {seed}, case {case}: parsed but invalid: {}\n{text}",
                            describe(&err)
                        )
                    });
                    // No forward-reference placeholder survives: a cast without inputs is one
                    // the text wrote.
                    let written = text.matches("builtin.unrealized_conversion_cast :").count()
                        + text.matches("builtin.unrealized_conversion_cast: ").count()
                        + text.matches("\"builtin.unrealized_conversion_cast\"()").count();
                    let mut without_inputs = 0;
                    module.borrow().prewalk_all(|op: &Operation| {
                        if op.is::<UnrealizedConversionCast>() {
                            assert!(
                                op.num_results() > 0,
                                "a cast without a result parsed:\n{text}"
                            );
                            without_inputs += usize::from(op.num_operands() == 0);
                        }
                    });
                    assert!(
                        without_inputs <= written,
                        "seed {seed}, case {case}: {without_inputs} casts without inputs, \
                         {written} written:\n{text}"
                    );
                    let printed = module.borrow().to_string();
                    let (again_context, again) = parse_module(&printed).unwrap_or_else(|err| {
                        panic!(
                            "seed {seed}, case {case}: printed text does not parse: \
                             {err}\n{printed}"
                        )
                    });
                    assert_eq!(again.borrow().to_string(), printed, "seed {seed}, case {case}");
                    drop(again_context);
                    drop(context);
                }
            }
        }
    }
    println!("{parsed} mutated texts parsed, {rejected} were rejected");
    let mut messages = messages.into_iter().collect::<Vec<_>>();
    messages.sort_by_key(|(_, count)| core::cmp::Reverse(*count));
    for (message, count) in messages.iter().take(12) {
        println!("  {count:6}  {message}");
    }
}

// ---------------------------------------------------------------------------------------------
// Reconciliation over shapes with one meaning
// ---------------------------------------------------------------------------------------------

/// The shapes a 128-bit value has: every cast between two shapes means "the same 128 bits, least
/// significant first", so casts compose.
const SHAPES: [&[u32]; 6] = [
    &[128],
    &[64, 64],
    &[64, 32, 32],
    &[32, 32, 64],
    &[32, 32, 32, 32],
    &[32, 64, 32],
];

fn int_name(width: u32) -> String {
    format!("i{width}")
}

struct Node {
    /// The values read: indices into the value table.
    inputs: Vec<usize>,
    /// The values defined.
    outputs: Vec<usize>,
    kind: NodeKind,
}

#[derive(PartialEq)]
enum NodeKind {
    Constant(u128),
    Cast,
    /// A cast of nothing to one value (never folded).
    Source,
    Sink,
}

struct Graph {
    nodes: Vec<Node>,
    /// The width of each value.
    widths: Vec<u32>,
}

impl Graph {
    /// A random graph. With `cyclic`, a cast may read values defined later (the graph is then
    /// printed in a graph region).
    fn random(rng: &mut Rng, cyclic: bool) -> Self {
        let mut graph = Graph {
            nodes: vec![],
            widths: vec![],
        };
        let count = 3 + rng.below(14);
        // Decide every node's outputs first, so a cast can read a later one.
        let mut plans = vec![];
        for index in 0..count {
            let roll = rng.below(10);
            let (kind, shape): (NodeKind, Vec<u32>) = if index < 2 || roll < 2 {
                let width = [32, 64, 128][rng.below(3)];
                (NodeKind::Constant(rng.next() as u128 | (rng.next() as u128) << 64), vec![width])
            } else if roll == 2 {
                (NodeKind::Source, vec![[32, 64, 128][rng.below(3)]])
            } else {
                (NodeKind::Cast, SHAPES[rng.below(SHAPES.len())].to_vec())
            };
            let outputs = shape
                .iter()
                .map(|width| {
                    graph.widths.push(*width);
                    graph.widths.len() - 1
                })
                .collect::<Vec<_>>();
            plans.push((kind, outputs));
        }
        for (index, (kind, outputs)) in plans.iter().enumerate() {
            let mut inputs = vec![];
            if *kind == NodeKind::Cast {
                let visible = |other: usize| cyclic || other < index;
                // Mostly: all the results of one cast, in order (a chain link).
                let links = plans
                    .iter()
                    .enumerate()
                    .filter(|(other, (kind, _))| visible(*other) && *kind == NodeKind::Cast)
                    .map(|(other, _)| other)
                    .collect::<Vec<_>>();
                if !links.is_empty() && rng.chance(65) {
                    let mut link = plans[links[rng.below(links.len())]].1.clone();
                    match rng.below(12) {
                        // Sometimes not a whole link: reordered, or one value from elsewhere.
                        0 => link.reverse(),
                        1 => {
                            let at = rng.below(link.len());
                            let width = graph.widths[link[at]];
                            let others = (0..graph.widths.len())
                                .filter(|v| graph.widths[*v] == width)
                                .filter(|v| {
                                    cyclic || plans[..index].iter().any(|(_, o)| o.contains(v))
                                })
                                .collect::<Vec<_>>();
                            if !others.is_empty() {
                                link[at] = others[rng.below(others.len())];
                            }
                        }
                        _ => {}
                    }
                    inputs = link;
                } else {
                    let shape = SHAPES[rng.below(SHAPES.len())];
                    for width in shape {
                        let candidates = (0..graph.widths.len())
                            .filter(|v| graph.widths[*v] == *width)
                            .filter(|v| cyclic || plans[..index].iter().any(|(_, o)| o.contains(v)))
                            .collect::<Vec<_>>();
                        if candidates.is_empty() {
                            inputs.clear();
                            break;
                        }
                        inputs.push(candidates[rng.below(candidates.len())]);
                    }
                }
            }
            graph.nodes.push(Node {
                inputs,
                outputs: outputs.clone(),
                kind: match kind {
                    NodeKind::Constant(v) => NodeKind::Constant(*v),
                    NodeKind::Cast => NodeKind::Cast,
                    NodeKind::Source => NodeKind::Source,
                    NodeKind::Sink => NodeKind::Sink,
                },
            });
        }
        // A cast for which no inputs were found reads nothing: it is a source.
        for node in &mut graph.nodes {
            if node.kind == NodeKind::Cast && node.inputs.is_empty() {
                node.kind = NodeKind::Source;
            }
        }
        // Sinks: each reads a few values.
        let total = graph.widths.len();
        let defined = graph.nodes.iter().flat_map(|node| node.outputs.clone()).collect::<Vec<_>>();
        for _ in 0..1 + rng.below(5) {
            let inputs = (0..1 + rng.below(3)).map(|_| defined[rng.below(defined.len())]).collect();
            graph.nodes.push(Node {
                inputs,
                outputs: vec![],
                kind: NodeKind::Sink,
            });
        }
        let _ = total;
        graph
    }

    fn text(&self, in_module: bool) -> String {
        let indent = if in_module { "    " } else { "        " };
        let list = |values: &[usize]| {
            values.iter().map(|v| format!("%{v}")).collect::<Vec<_>>().join(", ")
        };
        let types = |values: &[usize]| {
            let names = values.iter().map(|v| int_name(self.widths[*v])).collect::<Vec<_>>();
            type_list_text(&names)
        };
        let mut body = String::new();
        for node in &self.nodes {
            match &node.kind {
                NodeKind::Constant(value) => {
                    let width = self.widths[node.outputs[0]];
                    let value = if width == 128 {
                        *value
                    } else {
                        value & ((1u128 << width) - 1)
                    };
                    body.push_str(&format!(
                        "{indent}%{} = test.constant {value} : {};\n",
                        node.outputs[0],
                        int_name(width)
                    ));
                }
                NodeKind::Source => {
                    // One cast of nothing per value.
                    for output in &node.outputs {
                        body.push_str(&format!(
                            "{indent}%{output} = builtin.unrealized_conversion_cast : () to {};\n",
                            int_name(self.widths[*output])
                        ));
                    }
                }
                NodeKind::Cast => body.push_str(&format!(
                    "{indent}{} = builtin.unrealized_conversion_cast {} : {} to {};\n",
                    list(&node.outputs),
                    list(&node.inputs),
                    types(&node.inputs),
                    types(&node.outputs)
                )),
                NodeKind::Sink => {
                    let names =
                        node.inputs.iter().map(|v| int_name(self.widths[*v])).collect::<Vec<_>>();
                    body.push_str(&format!(
                        "{indent}test.sink {} : ({});\n",
                        list(&node.inputs),
                        names.join(", ")
                    ));
                }
            }
        }
        if in_module {
            format!("builtin.module public @m {{\n{body}}};")
        } else {
            format!(
                "builtin.module public @m {{
    builtin.function public extern(\"C\") @f() {{
{body}        builtin.ret;
    }};
}};"
            )
        }
    }
}

/// What every sink under `root` reads: per operand its type and its number, `None` for a value
/// that depends on a cast of nothing or on itself. A cast means: the bits of its inputs, least
/// significant first, are the bits of its results.
fn observe(root: OperationRef) -> Vec<Vec<(Type, Option<u128>)>> {
    let mut ops = vec![];
    root.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
    let mut env = crate::FxHashMap::<ValueRef, u128>::default();
    let width_of = |value: ValueRef| Int::width_for_type(value.borrow().ty()).unwrap();
    // To a fixpoint: a value is known once everything it is cast from is.
    loop {
        let mut changed = false;
        for op in &ops {
            let op = op.borrow();
            if let Some(constant) = op.downcast_ref::<Constant>() {
                let result = op.results()[0].borrow().as_value_ref();
                changed |= env.insert(result, constant.get_value().unsigned()).is_none();
            } else if op.is::<UnrealizedConversionCast>() && op.num_operands() > 0 {
                let inputs = op
                    .operands()
                    .iter()
                    .map(|operand| operand.borrow().as_value_ref())
                    .collect::<Vec<_>>();
                let results =
                    op.results().iter().map(|r| r.borrow().as_value_ref()).collect::<Vec<_>>();
                if results.iter().all(|r| env.contains_key(r)) {
                    continue;
                }
                let Some(numbers) =
                    inputs.iter().map(|v| env.get(v).copied()).collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                let total: u32 = inputs.iter().map(|v| width_of(*v)).sum();
                let out: u32 = results.iter().map(|v| width_of(*v)).sum();
                assert_eq!(
                    (total, out),
                    (128, 128),
                    "a cast of the model is between shapes of 128 bits"
                );
                let mut bits = 0u128;
                let mut shift = 0;
                for (value, number) in inputs.iter().zip(numbers) {
                    bits |= number << shift;
                    shift += width_of(*value);
                }
                let mut shift = 0;
                for result in results {
                    let width = width_of(result);
                    let mask = if width == 128 {
                        u128::MAX
                    } else {
                        (1u128 << width) - 1
                    };
                    env.insert(result, (bits >> shift) & mask);
                    shift += width;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    ops.iter()
        .filter(|op| op.borrow().is::<Sink>())
        .map(|op| {
            op.borrow()
                .operands()
                .iter()
                .map(|operand| {
                    let value = operand.borrow().as_value_ref();
                    (value.borrow().ty(), env.get(&value).copied())
                })
                .collect()
        })
        .collect()
}

fn check_reconciliation(seed: u64, case: usize, rng: &mut Rng, cyclic: bool) -> (usize, usize) {
    let graph = Graph::random(rng, cyclic);
    let source = graph.text(cyclic);
    let (context, module) = parse_module(&source)
        .unwrap_or_else(|err| panic!("seed {seed}, case {case}: {err}\n{source}"));
    let before = observe(module);
    let source = module.borrow().to_string();
    let casts_before = source.matches("unrealized_conversion_cast").count();
    let reconciled = catch_unwind(AssertUnwindSafe(|| reconcile_unrealized_casts(module)))
        .unwrap_or_else(|_| {
            panic!("seed {seed}, case {case}: reconciliation panicked on\n{source}")
        });
    let output = module.borrow().to_string();
    let fail = |what: &str| -> ! {
        panic!("seed {seed}, case {case}: {what}\n--- source\n{source}\n--- output\n{output}")
    };
    if let Err(err) = module.borrow().recursively_verify() {
        fail(&format!("the result does not verify: {}", describe(&err)));
    }
    let after = observe(module);
    if before.len() != after.len() {
        fail("a sink was lost");
    }
    for (index, (was, is)) in before.iter().zip(&after).enumerate() {
        for (operand, ((was_ty, was_value), (is_ty, is_value))) in was.iter().zip(is).enumerate() {
            if was_ty != is_ty {
                fail(&format!("operand {operand} of sink {index} changed type"));
            }
            // A value that meant a number still means it.
            if was_value.is_some() && was_value != is_value {
                fail(&format!(
                    "operand {operand} of sink {index} was {was_value:x?} and is {is_value:x?}"
                ));
            }
        }
    }
    // What remains is what is in the IR, each used; and changed says whether the text changed.
    let mut remaining = vec![];
    module.borrow().prewalk_all(|op: &Operation| {
        if op.is::<UnrealizedConversionCast>() {
            remaining.push(op.as_operation_ref());
        }
    });
    if remaining.as_slice() != reconciled.remaining.as_slice() {
        fail("the casts returned are not the casts that remain, in order");
    }
    if remaining.iter().any(|cast| !cast.borrow().is_used()) {
        fail("an unused cast remains");
    }
    if reconciled.changed != (output != source) {
        fail(&format!("changed is {}", reconciled.changed));
    }
    let again = reconcile_unrealized_casts(module);
    if again.changed || module.borrow().to_string() != output {
        fail("a second reconciliation changed the IR");
    }
    drop(context);
    (casts_before, remaining.len())
}

/// Random DAGs of casts over shapes with one meaning: what the sinks read is what they read.
#[test]
fn reconciliation_keeps_the_meaning_of_random_dags() {
    let (mut before, mut after) = (0, 0);
    for seed in seeds(&[0x2ec0_0001, 0x2ec0_0002, 0x2ec0_0003]) {
        let mut rng = Rng(seed);
        for case in 0..cases(500) {
            let (b, a) = check_reconciliation(seed, case, &mut rng, false);
            before += b;
            after += a;
        }
    }
    println!("{before} casts before, {after} after");
}

/// Random graphs of casts with cycles, in a graph region: reconciliation ends, and what a sink
/// read that had a meaning it still reads.
#[test]
fn reconciliation_ends_and_keeps_the_meaning_of_random_cyclic_graphs() {
    let (mut before, mut after) = (0, 0);
    for seed in seeds(&[0x2ec1_0001, 0x2ec1_0002, 0x2ec1_0003]) {
        let mut rng = Rng(seed);
        for case in 0..cases(500) {
            let (b, a) = check_reconciliation(seed, case, &mut rng, true);
            before += b;
            after += a;
        }
    }
    println!("{before} casts before, {after} after");
}
