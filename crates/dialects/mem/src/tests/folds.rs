//! Folds against `DataLayout`, gep typing against an independent tree, and the gep
//! merge against an address model, under random specifications.

use alloc::{
    boxed::Box,
    collections::BTreeMap,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use runic_core::{
    Context, DataLayout, OperationRef, SymbolTable, Type, Value, ValueRef,
    dialects::builtin::{Function, Module},
    parse::{self, ParserConfig},
    pass::{Nesting, PassManager},
    testing::text::{Rng, describe},
};
use runic_transform::Canonicalizer;

use crate::*;

/// A type as the generator knows it: its text and its elements.
#[derive(Clone, Debug)]
pub struct Ty {
    pub text: String,
    /// The element at each index of a path step; `None` where the index has no element (an enum
    /// variant without a payload).
    pub elements: Vec<Option<Ty>>,
    /// For an array: the number of elements (each is `elements[0]`).
    pub array_len: Option<u64>,
}

fn scalar(text: &str) -> Ty {
    Ty {
        text: text.to_string(),
        elements: Vec::new(),
        array_len: None,
    }
}

pub fn random_ty(rng: &mut Rng, depth: u32) -> Ty {
    let scalars = [
        "i1",
        "i7",
        "i8",
        "i16",
        "i24",
        "i32",
        "i33",
        "i64",
        "i128",
        "isize",
        "f32",
        "f64",
        "ptr",
        "ptr<i8>",
        "ptr<i64, 1>",
        "ptr<2>",
    ];
    if depth == 0 || rng.below(3) == 0 {
        return scalar(rng.pick(&scalars));
    }
    match rng.below(3) {
        0 => {
            let element = random_ty(rng, depth - 1);
            let len = rng.below(5);
            Ty {
                text: format!("array<{}; {len}>", element.text),
                elements: alloc::vec![Some(element)],
                array_len: Some(len),
            }
        }
        1 => {
            let fields = (0..rng.below(5)).map(|_| random_ty(rng, depth - 1)).collect::<Vec<_>>();
            let repr = match rng.below(6) {
                0 => format!("align({}); ", 1u64 << rng.below(6)),
                1 => format!("packed({}); ", 1u64 << rng.below(3)),
                2 => "packed; ".to_string(),
                _ => String::new(),
            };
            let texts = fields
                .iter()
                .map(|f| {
                    if rng.below(8) == 0 {
                        format!("{} align({})", f.text, 1u64 << rng.below(5))
                    } else {
                        f.text.clone()
                    }
                })
                .collect::<Vec<_>>();
            Ty {
                text: format!("struct<{repr}{}>", texts.join(", ")),
                elements: fields.into_iter().map(Some).collect(),
                array_len: None,
            }
        }
        _ => {
            let tag = rng.pick(&["i8", "i16", "i32", "isize"]);
            let variants = (0..1 + rng.below(4))
                .map(|_| (rng.below(3) != 0).then(|| random_ty(rng, depth - 1)))
                .collect::<Vec<_>>();
            let texts = variants
                .iter()
                .enumerate()
                .map(|(i, v)| match v {
                    Some(payload) => format!("v{i}: {}", payload.text),
                    None => format!("v{i}"),
                })
                .collect::<Vec<_>>();
            let mut elements = alloc::vec![Some(scalar(tag))];
            elements.extend(variants);
            Ty {
                text: format!("enum<{tag}; {}>", texts.join(", ")),
                elements,
                array_len: None,
            }
        }
    }
}

/// A random valid path into `ty`, and the element it selects.
pub fn random_path(rng: &mut Rng, ty: &Ty) -> (Vec<u32>, Ty) {
    let mut path = Vec::new();
    let mut current = ty.clone();
    while rng.below(4) != 0 {
        let next = if let Some(len) = current.array_len {
            if len == 0 {
                break;
            }
            // a path step is an `i32` in the text: see `a_path_step_above_i32_max`
            let index = rng.below(len.min(1 << 31)) as u32;
            path.push(index);
            current.elements[0].clone().unwrap()
        } else {
            let candidates = current
                .elements
                .iter()
                .enumerate()
                .filter_map(|(i, e)| e.clone().map(|e| (i as u32, e)))
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                break;
            }
            let (index, element) = candidates[rng.below(candidates.len() as u64) as usize].clone();
            path.push(index);
            element
        };
        current = next;
    }
    (path, current)
}

pub fn random_layout(rng: &mut Rng) -> String {
    let endian = rng.pick(&["little", "big"]);
    let ptr = rng.pick(&[
        "ptr = 32",
        "ptr = 64",
        "ptr = 16",
        "ptr = 64 align 8 index 32",
        "ptr = 32 align 8",
        "ptr = 24 align 4",
        "ptr = 8",
    ]);
    let mut entries = alloc::vec![endian.to_string(), ptr.to_string()];
    if rng.below(2) == 0 {
        entries
            .push(rng.pick(&["ptr<1> = 64", "ptr<1> = 16", "ptr<1> = 64 align 8 index 32"]).into());
    }
    if rng.below(3) == 0 {
        entries.push(rng.pick(&["ptr<2> = 32", "ptr<2> = 8"]).into());
    }
    if rng.below(3) == 0 {
        entries.push(rng.pick(&["i64 = 4", "i64 = 8", "i32 = 2", "i16 = 1"]).into());
    }
    if rng.below(3) == 0 {
        entries.push(rng.pick(&["i128 = 8", "i128 = 4"]).into());
    }
    if rng.below(3) == 0 {
        entries.push(rng.pick(&["f64 = 4", "f32 = 2"]).into());
    }
    entries.join(", ")
}

fn path_text(path: &[u32]) -> String {
    if path.is_empty() {
        String::new()
    } else {
        format!(", [{}]", path.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
    }
}

pub struct Parsed {
    pub context: Rc<Context>,
    pub function: OperationRef,
}

pub fn parse_module(
    layout: Option<&str>,
    params: &str,
    results: &str,
    body: &str,
) -> Result<Parsed, String> {
    let context = Rc::new(Context::default());
    let attributes = layout
        .map(|layout| format!(" attributes {{ data_layout = #builtin.data_layout<{layout}> }}"))
        .unwrap_or_default();
    let text = format!(
        "builtin.module public @m{attributes} {{\nbuiltin.function public extern(\"C\") \
         @f({params}) -> ({results}) {{\n{body}\n}};\n}};"
    );
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "mem", text.clone())
        .into_result()
        .map_err(|err| format!("{}\n{text}", describe(&err)))?;
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    Ok(Parsed { context, function })
}

pub fn canonicalize(parsed: &Parsed) -> Result<(), String> {
    let mut passes = PassManager::on::<Function>(parsed.context.clone(), Nesting::Implicit);
    passes.add_pass(Box::<Canonicalizer>::default()).unwrap();
    passes.enable_verifier(true);
    passes.run(parsed.function).map_err(|err| describe(&err))
}

fn entry_ops(function: OperationRef) -> Vec<OperationRef> {
    let function = function.borrow();
    let function = function.downcast_ref::<Function>().unwrap();
    let entry = function.entry_block();
    let entry = entry.borrow();
    entry.body().iter().map(|op| op.as_operation_ref()).collect()
}

/// The constant each `%name = arith.constant V : isize;` of `text` defines.
fn constants(text: &str) -> BTreeMap<String, u64> {
    let mut found = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some((name, rest)) = line.split_once(" = arith.constant ") {
            let value = rest.split(' ').next().unwrap();
            found.insert(
                name.to_string(),
                value.parse::<u64>().unwrap_or_else(|_| panic!("{line}")),
            );
        }
    }
    found
}

fn ret_operands(text: &str) -> Vec<String> {
    let ret = text.lines().rev().find(|line| line.contains("builtin.ret")).unwrap();
    let operands = ret.trim().trim_start_matches("builtin.ret").split(" : ").next().unwrap();
    operands
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Under 600 random specifications, 12 random size, alignment and offset operations each fold to
/// exactly what `DataLayout` answers for their type and path, when the answer fits the `isize` of
/// the specification, and stay as they are when it does not. The same module with no
/// specification is left alone.
#[test]
fn size_operations_fold_to_what_the_layout_answers() {
    let mut rng = Rng(super::seed(0x0051_2e0f));
    let (mut folded, mut too_wide, mut rejected) = (0, 0, 0);
    for case in 0..600 {
        let layout = random_layout(&mut rng);
        let mut lines = Vec::new();
        let count = 12;
        for i in 0..count {
            // sometimes an array whose size is beyond a narrow isize
            let ty = if rng.below(12) == 0 {
                let element = random_ty(&mut rng, 1);
                let len = 1u64 << (8 + rng.below(40));
                Ty {
                    text: format!("array<{}; {len}>", element.text),
                    elements: alloc::vec![Some(element)],
                    array_len: Some(len),
                }
            } else {
                random_ty(&mut rng, 3)
            };
            let line = match rng.below(3) {
                0 => format!("    %r{i} = mem.sizeof {};", ty.text),
                1 => format!("    %r{i} = mem.alignof {};", ty.text),
                _ => {
                    let (path, _) = random_path(&mut rng, &ty);
                    format!("    %r{i} = mem.offsetof {}{};", ty.text, path_text(&path))
                }
            };
            lines.push(line);
        }
        let names = (0..count).map(|i| format!("%r{i}")).collect::<Vec<_>>().join(", ");
        let types = (0..count).map(|_| "isize").collect::<Vec<_>>().join(", ");
        let body = format!("{}\n    builtin.ret {names} : ({types});", lines.join("\n"));

        // with no specification: verifies, and nothing folds
        let bare =
            parse_module(None, "", &types, &body).unwrap_or_else(|e| panic!("case {case}: {e}"));
        let before = bare.function.borrow().to_string();
        canonicalize(&bare).unwrap_or_else(|e| panic!("case {case}: {e}"));
        let after = bare.function.borrow().to_string();
        assert!(
            !after.contains("arith.constant"),
            "case {case}: folded without a layout\n{after}"
        );
        // (equal operations may have been merged by nothing: the canonicalizer does not CSE)
        assert_eq!(before.matches("mem.").count(), after.matches("mem.").count(), "case {case}");

        // with the specification: a type that cannot be laid out is a diagnostic (a discriminant
        // too wide, say), never a panic
        let parsed = match parse_module(Some(&layout), "", &types, &body) {
            Ok(parsed) => parsed,
            Err(err) => {
                assert!(
                    err.contains("has no layout under the data layout in scope"),
                    "case {case}: {err}"
                );
                rejected += 1;
                continue;
            }
        };
        let data_layout =
            DataLayout::closest(&parsed.function.borrow()).expect("a layout in scope");
        let width = data_layout.isize_width();
        let expected = entry_ops(parsed.function)
            .into_iter()
            .filter_map(|op| {
                let op = op.borrow();
                if let Some(sizeof) = op.downcast_ref::<Sizeof>() {
                    Some(data_layout.size_of(*sizeof.get_ty()).unwrap())
                } else if let Some(alignof) = op.downcast_ref::<Alignof>() {
                    Some(data_layout.align_of(*alignof.get_ty()).unwrap().bytes())
                } else if let Some(offsetof) = op.downcast_ref::<Offsetof>() {
                    let steps = offsetof.steps().unwrap();
                    Some(data_layout.offset_of(*offsetof.get_ty(), &steps).unwrap().0)
                } else {
                    None
                }
            })
            .collect::<Vec<u64>>();
        assert_eq!(expected.len(), count);
        canonicalize(&parsed).unwrap_or_else(|e| panic!("case {case} under {layout}: {e}\n{body}"));
        let text = parsed.function.borrow().to_string();
        let constants = constants(&text);
        let operands = ret_operands(&text);
        assert_eq!(operands.len(), count);
        for (i, (operand, expected)) in operands.iter().zip(&expected).enumerate() {
            let fits = width >= 64 || *expected >> width == 0;
            match constants.get(operand) {
                Some(value) => {
                    assert!(
                        fits,
                        "case {case} #{i} under {layout}: folded {expected} into a {width}-bit \
                         isize\n{body}\n{text}"
                    );
                    assert_eq!(value, expected, "case {case} #{i} under {layout}\n{body}\n{text}");
                    folded += 1;
                }
                None => {
                    assert!(
                        !fits,
                        "case {case} #{i} under {layout}: {expected} fits {width} bits but did \
                         not fold\n{body}\n{text}"
                    );
                    too_wide += 1;
                }
            }
        }
        assert!(
            text.contains(" : isize;") || expected.iter().all(|e| width < 64 && e >> width != 0)
        );
    }
    std::println!("folded {folded}, too wide {too_wide}, rejected modules {rejected}");
    assert!(folded > 4000 && too_wide > 20, "folded {folded}, too wide {too_wide}");
}

/// A linear form: a constant plus coefficients of leaf values (block arguments).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct Linear {
    constant: u64,
    terms: BTreeMap<usize, u64>,
}

/// The address `value` holds, as a linear form of the function's arguments, under `layout`:
/// a gep is its base plus index times the size of the base's pointee plus the offset of its path.
fn address(layout: &DataLayout, args: &[ValueRef], value: ValueRef) -> Linear {
    if let Some(at) = args.iter().position(|arg| *arg == value) {
        return Linear {
            constant: 0,
            terms: BTreeMap::from([(at, 1)]),
        };
    }
    let defining = value.borrow().get_defining_op().expect("a result");
    let defining = defining.borrow();
    let gep = defining
        .downcast_ref::<Gep>()
        .unwrap_or_else(|| panic!("not a gep: {}", *defining));
    let base = gep.base().as_value_ref();
    let pointee = base.borrow().ty().pointee().expect("a typed base");
    let mut form = address(layout, args, base);
    if let Some(index) = gep.dynamic_index() {
        let size = layout.size_of(pointee).unwrap();
        let at = args.iter().position(|arg| *arg == index).expect("an index is an argument");
        let entry = form.terms.entry(at).or_default();
        *entry = entry.wrapping_add(size);
    }
    let (offset, _) = layout.offset_of(pointee, &gep.steps().unwrap()).unwrap();
    form.constant = form.constant.wrapping_add(offset);
    form.terms.retain(|_, coefficient| *coefficient != 0);
    form
}

fn ret_values(function: OperationRef) -> Vec<ValueRef> {
    let ops = entry_ops(function);
    let ret = ops.last().unwrap().borrow();
    ret.operands().iter().map(|operand| operand.borrow().as_value_ref()).collect()
}

/// Chains of geps under random specifications: the type of every gep is the pointer to the
/// element the generator's own tree selects (and `offset_of` selects the same); after
/// canonicalization every returned pointer has the same type and the same address, as a linear
/// form of the arguments, as before.
#[test]
fn geps_select_the_modelled_element_and_merging_keeps_the_address() {
    let mut rng = Rng(super::seed(0x06e9_c4a1));
    let (mut merged, mut chains, mut rejected) = (0, 0, 0);
    for case in 0..800 {
        let layout = random_layout(&mut rng);
        let root = random_ty(&mut rng, 4);
        let space = rng.pick(&["", ", 1", ", 2"]);
        let params = format!("%p: ptr<{}{space}>, %i: isize, %j: isize", root.text);
        let mut lines = Vec::new();
        // (value name, pointee as the generator knows it)
        let mut pointers = alloc::vec![("%p".to_string(), root.clone())];
        let mut expected_types = Vec::new();
        let count = 2 + rng.below(6) as usize;
        for g in 0..count {
            // mostly chain on the latest pointer
            let (base, pointee) = if rng.below(4) == 0 {
                pointers[rng.below(pointers.len() as u64) as usize].clone()
            } else {
                pointers.last().unwrap().clone()
            };
            let (path, selected) = random_path(&mut rng, &pointee);
            let index = match rng.below(4) {
                0 => ", %i",
                1 => ", %j",
                _ => "",
            };
            let name = format!("%g{g}");
            lines.push(format!("    {name} = mem.gep {base}{index}{};", path_text(&path)));
            expected_types.push(format!("ptr<{}{space}>", selected.text));
            pointers.push((name, selected));
        }
        let names = (0..count).map(|g| format!("%g{g}")).collect::<Vec<_>>().join(", ");
        let types = expected_types.join(", ");
        let body = format!("{}\n    builtin.ret {names} : ({types});", lines.join("\n"));
        // The function's result types are the generator's: parsing and verifying checks that
        // each gep's inferred type is the modelled one (`builtin.ret` checks its operands).
        // a root type without a layout under the specification (a discriminant too wide, say)
        // cannot be measured: a gep with an index or a path over it is a diagnostic, and the
        // chain is skipped
        let parsed = match parse_module(Some(&layout), &params, &types, &body) {
            Ok(parsed) => parsed,
            Err(err) if err.contains("has no layout under the data layout in scope") => {
                rejected += 1;
                continue;
            }
            Err(err) => panic!("case {case} under {layout}: {err}"),
        };
        let data_layout = DataLayout::closest(&parsed.function.borrow()).unwrap();
        let args = {
            let function = parsed.function.borrow();
            let function = function.downcast_ref::<Function>().unwrap();
            let entry = function.entry_block();
            let entry = entry.borrow();
            (0..3).map(|i| entry.arguments()[i] as ValueRef).collect::<Vec<_>>()
        };
        // a root type without a layout (a discriminant too wide) cannot be measured: skip
        let root_ty = args[0].borrow().ty().pointee().unwrap();
        if data_layout.layout_of(root_ty).is_err() {
            continue;
        }
        chains += 1;
        let before_values = ret_values(parsed.function);
        let before_types = before_values.iter().map(|v| v.borrow().ty()).collect::<Vec<Type>>();
        // `offset_of` selects the element the gep's type names
        for value in &before_values {
            let op = value.borrow().get_defining_op().unwrap();
            let op = op.borrow();
            let gep = op.downcast_ref::<Gep>().unwrap();
            let pointee = gep.base().ty().pointee().unwrap();
            let (_, selected) = data_layout.offset_of(pointee, &gep.steps().unwrap()).unwrap();
            assert_eq!(Some(selected), gep.result().ty().pointee(), "case {case}");
        }
        let before = before_values
            .iter()
            .map(|v| address(&data_layout, &args, *v))
            .collect::<Vec<_>>();
        let before_text = parsed.function.borrow().to_string();
        canonicalize(&parsed).unwrap_or_else(|e| panic!("case {case} under {layout}: {e}\n{body}"));
        let after_values = ret_values(parsed.function);
        let after_text = parsed.function.borrow().to_string();
        for (k, value) in after_values.iter().enumerate() {
            assert_eq!(
                value.borrow().ty(),
                before_types[k],
                "case {case} result {k}\n{before_text}\n{after_text}"
            );
            let form = address(&data_layout, &args, *value);
            assert_eq!(
                form, before[k],
                "case {case} result {k} under {layout}: the address changed\n--- \
                 before\n{before_text}\n--- after\n{after_text}"
            );
        }
        if before_text != after_text {
            merged += 1;
        }
    }
    std::println!("{chains} chains, {merged} changed by the canonicalizer, {rejected} rejected");
    assert!(merged > 200, "{merged}");
    // A chain is skipped only when its root type cannot be measured, which the generator makes
    // rarely: a gep check that turned valid chains away would show here.
    assert!(rejected * 20 <= chains, "{rejected} of {chains} chains were rejected");
}
