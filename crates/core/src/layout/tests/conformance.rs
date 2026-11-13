//! Conformance of the data layout: specifications against a model written from the design's
//! table, structs and enums against the host compiler and clang, random aggregates against a
//! reference written here, recursion in every query order, caches and reentrancy.
//!
//! The model, the generators and the reference layouts are written independently of the code
//! under test. The randomized tests run the seeds of [super::seeds] (`LAYOUT_TEST_SEED`
//! overrides them) and print each seed to the captured output, which a failure shows.

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    format,
    string::String,
    vec,
    vec::Vec,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::*;

// The number of cases per seed of each randomized test. Together they run in a few seconds in the
// dev profile.

/// Specifications built, written and compared with another one.
const SPECIFICATIONS: usize = 3600;
/// Damaged specification texts.
const DAMAGED_TEXTS: usize = 12_000;
/// Specifications under which every scalar is laid out.
const SCALAR_SPECIFICATIONS: usize = 450;
/// Specifications under which 25 random aggregates are laid out.
const AGGREGATE_SPECIFICATIONS: usize = 360;
/// Random graphs of recursive types, each asked in every order.
const GRAPHS: usize = 1500;

/// The specification the attribute text `attr` gives, or a description of why it gives none.
fn parse_spec(context: &Rc<Context>, attr: &str) -> Result<DataLayoutSpec, String> {
    let module = parse_module(context, attr)?;
    let module = module.borrow();
    let attr = module
        .get_typed_attribute::<DataLayoutAttr>(DATA_LAYOUT_ATTR)
        .ok_or_else(|| String::from("parsed, but not to a data layout attribute"))?;
    Ok(attr.borrow().as_value().clone())
}

fn round_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

// --- A model of a specification, written from the design's table ------------------------------

/// What a specification says, as the entries given to it: the design's table, nothing canonical.
#[derive(Clone, Debug)]
struct Model {
    big: bool,
    /// By address space: size in bits, alignment in bytes, index width in bits. Has 0.
    pointers: BTreeMap<u32, (u32, u64, u32)>,
    /// By width: alignment in bytes. Has the defaults unless overridden.
    integers: BTreeMap<u32, u64>,
    f32: u64,
    f64: u64,
}

impl Model {
    fn pointer(&self, addrspace: u32) -> (u32, u64, u32) {
        *self.pointers.get(&addrspace).unwrap_or(&self.pointers[&0])
    }

    /// The entry of the next larger listed width, else of the largest.
    fn integer_align(&self, width: u32) -> u64 {
        match self.integers.range(width..).next() {
            Some((_, align)) => *align,
            None => *self.integers.values().next_back().unwrap(),
        }
    }

    fn isize_width(&self) -> u32 {
        self.pointer(0).2
    }

    /// `(size, align)` of an integer of `width` bits.
    fn integer(&self, width: u32) -> (u64, u64) {
        let align = self.integer_align(width);
        (round_up(u64::from(width).div_ceil(8), align), align)
    }

    fn float(&self, bits: u32) -> (u64, u64) {
        let align = if bits == 32 { self.f32 } else { self.f64 };
        (round_up(u64::from(bits / 8), align), align)
    }

    fn pointer_layout(&self, addrspace: u32) -> (u64, u64) {
        let (size, align, _) = self.pointer(addrspace);
        (round_up(u64::from(size).div_ceil(8), align), align)
    }

    /// The widths at which the answer may change, and their neighbours.
    fn probe_widths(&self) -> BTreeSet<u32> {
        let mut probes = BTreeSet::new();
        for &width in self.integers.keys() {
            for probe in [width.saturating_sub(1), width, width + 1] {
                if (1..=IntegerType::MAX_WIDTH).contains(&probe) {
                    probes.insert(probe);
                }
            }
        }
        probes
    }

    /// True if `self` and `other` answer every query alike. The integer answers can only change
    /// at an entry of either, so those widths and their neighbours are compared, with every
    /// width up to 320.
    fn answers_as(&self, other: &Model) -> bool {
        let mut widths: BTreeSet<u32> = (1..=320).collect();
        widths.extend(self.probe_widths());
        widths.extend(other.probe_widths());
        widths.insert(IntegerType::MAX_WIDTH);
        let mut spaces: BTreeSet<u32> = self.pointers.keys().copied().collect();
        spaces.extend(other.pointers.keys().copied());
        spaces.extend([0, 1, 2, 3, 9, 77, u32::MAX - 1, u32::MAX]);
        self.big == other.big
            && self.f32 == other.f32
            && self.f64 == other.f64
            && spaces.iter().all(|space| self.pointer(*space) == other.pointer(*space))
            && widths
                .iter()
                .all(|width| self.integer_align(*width) == other.integer_align(*width))
    }

    /// The specification, built through the API with the entries in a random order.
    fn build(&self, rng: &mut Rng) -> DataLayoutSpec {
        let pointer = |(size_bits, bytes, index_bits): (u32, u64, u32)| PointerSpec {
            size_bits,
            align: align(bytes),
            index_bits,
        };
        let endianness = if self.big {
            Endianness::Big
        } else {
            Endianness::Little
        };
        let mut spec = DataLayoutSpec::new(endianness, pointer(self.pointers[&0])).unwrap();
        let mut steps: Vec<Box<dyn Fn(DataLayoutSpec) -> DataLayoutSpec + '_>> = Vec::new();
        for (&space, &entry) in &self.pointers {
            steps.push(Box::new(move |spec| spec.with_pointer(space, pointer(entry)).unwrap()));
        }
        for (&width, &bytes) in &self.integers {
            steps.push(Box::new(move |spec| spec.with_integer_align(width, align(bytes)).unwrap()));
        }
        steps.push(Box::new(|spec| spec.with_float_align(FloatType::F32, align(self.f32))));
        steps.push(Box::new(|spec| spec.with_float_align(FloatType::F64, align(self.f64))));
        while !steps.is_empty() {
            let step = steps.swap_remove(rng.below(steps.len() as u64) as usize);
            spec = step(spec);
        }
        spec
    }

    /// A text of the specification that is not the canonical one: the entries in a random order,
    /// defaults sometimes written out, `align` and `index` sometimes written when implied.
    fn text(&self, rng: &mut Rng) -> String {
        let mut entries = Vec::new();
        for (&space, &(size, bytes, index)) in &self.pointers {
            let key = if space == 0 && rng.below(4) != 0 {
                String::from("ptr")
            } else {
                format!("ptr<{space}>")
            };
            let implied = size.is_multiple_of(8)
                && u64::from(size / 8).is_power_of_two()
                && u64::from(size / 8) == bytes;
            let mut entry = format!("{key} = {size}");
            if !implied || rng.below(2) == 0 {
                entry.push_str(&format!(" align {bytes}"));
            }
            if index != size || rng.below(2) == 0 {
                entry.push_str(&format!(" index {index}"));
            }
            entries.push(entry);
        }
        for (&width, &bytes) in &self.integers {
            let is_default = DEFAULTS.contains(&(width, bytes));
            if !is_default || rng.below(2) == 0 {
                entries.push(format!("i{width} = {bytes}"));
            }
        }
        if self.f32 != 4 || rng.below(2) == 0 {
            entries.push(format!("f32 = {}", self.f32));
        }
        if self.f64 != 8 || rng.below(2) == 0 {
            entries.push(format!("f64 = {}", self.f64));
        }
        let mut text = String::from(if self.big { "big" } else { "little" });
        while !entries.is_empty() {
            let entry = entries.swap_remove(rng.below(entries.len() as u64) as usize);
            text.push_str(", ");
            text.push_str(&entry);
        }
        format!("#builtin.data_layout<{text}>")
    }
}

const DEFAULTS: [(u32, u64); 5] = [(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)];

fn random_align_bytes(rng: &mut Rng) -> u64 {
    match rng.below(8) {
        // Few values, so that neighbouring entries often agree
        0..=5 => 1 << rng.below(5),
        _ => 1 << rng.below(16),
    }
}

fn random_model_pointer(rng: &mut Rng) -> (u32, u64, u32) {
    let size = 8 * (1 + rng.below(8) as u32);
    let implied = u64::from(size / 8);
    let bytes = if implied.is_power_of_two() && rng.below(3) != 0 {
        implied
    } else {
        random_align_bytes(rng)
    };
    let index = match rng.below(3) {
        0 => 1 + rng.below(u64::from(size)) as u32,
        _ => size,
    };
    (size, bytes, index)
}

fn random_model_width(rng: &mut Rng) -> u32 {
    match rng.below(10) {
        0..=3 => rng.pick(&[1, 7, 8, 9, 16, 24, 32, 33, 64, 65, 96, 128, 129, 200, 256, 300]),
        4..=7 => 1 + rng.below(300) as u32,
        8 => IntegerType::MAX_WIDTH - rng.below(3) as u32,
        _ => 1 + rng.below(u64::from(IntegerType::MAX_WIDTH)) as u32,
    }
}

fn random_model(rng: &mut Rng) -> Model {
    let mut pointers = BTreeMap::new();
    let first = random_model_pointer(rng);
    pointers.insert(0, first);
    for _ in 0..rng.below(4) {
        let space = rng.pick(&[1, 2, 3, 77, u32::MAX]);
        // Often the pointer of address space 0 again: an entry that changes no answer
        let pointer = if rng.below(3) == 0 {
            first
        } else {
            random_model_pointer(rng)
        };
        pointers.insert(space, pointer);
    }
    let mut integers: BTreeMap<u32, u64> = DEFAULTS.into_iter().collect();
    for _ in 0..rng.below(9) {
        integers.insert(random_model_width(rng), random_align_bytes(rng));
    }
    let float = |rng: &mut Rng, default: u64| match rng.below(3) {
        0 => random_align_bytes(rng),
        _ => default,
    };
    Model {
        big: rng.below(2) == 0,
        pointers,
        integers,
        f32: float(rng, 4),
        f64: float(rng, 8),
    }
}

/// `model` with one change, which may or may not change an answer.
fn mutated(rng: &mut Rng, model: &Model) -> Model {
    let mut model = model.clone();
    match rng.below(8) {
        0 => model.big = !model.big,
        1 => model.f32 = random_align_bytes(rng),
        2 => model.f64 = random_align_bytes(rng),
        3 => {
            let space = rng.pick(&[0, 1, 2, 3, 77, u32::MAX]);
            let pointer = if rng.below(2) == 0 {
                model.pointer(space)
            } else {
                random_model_pointer(rng)
            };
            model.pointers.insert(space, pointer);
        }
        4 => {
            // An entry that restates what the table answers
            let width = random_model_width(rng);
            let align = model.integer_align(width);
            model.integers.insert(width, align);
        }
        5 => {
            // Remove a non-default entry, if there is one
            let extra: Vec<u32> = model
                .integers
                .keys()
                .copied()
                .filter(|width| !DEFAULTS.iter().any(|(default, _)| default == width))
                .collect();
            if !extra.is_empty() {
                model.integers.remove(&rng.pick(&extra));
            }
        }
        _ => {
            model.integers.insert(random_model_width(rng), random_align_bytes(rng));
        }
    }
    model
}

/// Checks every query of `spec` against `model`.
fn assert_answers(spec: &DataLayoutSpec, model: &Model, what: &str) {
    let endianness = if model.big {
        Endianness::Big
    } else {
        Endianness::Little
    };
    assert_eq!(spec.endianness(), endianness, "{what}");
    assert_eq!(spec.float_align(FloatType::F32).bytes(), model.f32, "{what}");
    assert_eq!(spec.float_align(FloatType::F64).bytes(), model.f64, "{what}");
    assert_eq!(spec.isize_width(), model.isize_width(), "{what}");
    let mut spaces: Vec<u32> = model.pointers.keys().copied().collect();
    spaces.extend([0, 1, 2, 3, 9, 77, u32::MAX - 1, u32::MAX]);
    for space in spaces {
        let pointer = spec.pointer(space);
        assert_eq!(
            (pointer.size_bits, pointer.align.bytes(), pointer.index_bits),
            model.pointer(space),
            "{what}: ptr<{space}>"
        );
    }
    let mut widths: BTreeSet<u32> = (1..=320).collect();
    widths.extend(model.probe_widths());
    widths.insert(IntegerType::MAX_WIDTH);
    for width in widths {
        assert_eq!(
            spec.integer_align(width).bytes(),
            model.integer_align(width),
            "{what}: i{width}"
        );
    }
}

// --- The specification's text and equality ----------------------------------------------------

#[test]
fn spec_answers_text_and_equality_agree_with_the_model() {
    let mut equal_pairs = 0usize;
    let mut unequal_pairs = 0usize;
    let mut texts = 0usize;
    for seed in seeds() {
        std::eprintln!("seed {seed:#x}");
        let mut rng = Rng(seed);
        let mut context = context();
        for case in 0..SPECIFICATIONS {
            if case % 250 == 0 {
                // Every parse registers a source: start again before they pile up
                context = self::context();
            }
            let what = format!("seed {seed:#x}, case {case}");
            let model = random_model(&mut rng);
            let spec = model.build(&mut rng);
            assert_answers(&spec, &model, &what);

            // A text that is not canonical parses to the same specification
            let written = model.text(&mut rng);
            let parsed = parse_spec(&context, &written)
                .unwrap_or_else(|err| panic!("{what}: {written}: {err}"));
            assert_answers(&parsed, &model, &format!("{what}: {written}"));
            assert_eq!(parsed, spec, "{what}: {written}");
            assert_eq!(hash_of(&parsed), hash_of(&spec), "{what}: {written}");

            // The canonical text parses to an equal one and prints the same again
            let printed = print_spec(&context, &spec);
            assert_eq!(print_spec(&context, &parsed), printed, "{what}: {written}");
            let reparsed = parse_spec(&context, &printed)
                .unwrap_or_else(|err| panic!("{what}: {printed}: {err}"));
            assert_answers(&reparsed, &model, &format!("{what}: {printed}"));
            assert_eq!(reparsed, spec, "{what}: {printed}");
            assert_eq!(hash_of(&reparsed), hash_of(&spec), "{what}: {printed}");
            assert_eq!(print_spec(&context, &reparsed), printed, "{what}");
            texts += 1;

            // Equal exactly when they answer alike; equal ones hash and print alike, and
            // unequal ones print differently
            let other_model = match rng.below(4) {
                0 => random_model(&mut rng),
                1 => {
                    let once = mutated(&mut rng, &model);
                    mutated(&mut rng, &once)
                }
                _ => mutated(&mut rng, &model),
            };
            let other = other_model.build(&mut rng);
            let alike = model.answers_as(&other_model);
            assert_eq!(spec == other, alike, "{what}:\n{model:?}\n{other_model:?}");
            assert_eq!(other == spec, alike, "{what}");
            let other_printed = print_spec(&context, &other);
            if alike {
                assert_eq!(hash_of(&spec), hash_of(&other), "{what}");
                assert_eq!(printed, other_printed, "{what}:\n{model:?}\n{other_model:?}");
                equal_pairs += 1;
            } else {
                assert_ne!(printed, other_printed, "{what}:\n{model:?}\n{other_model:?}");
                unequal_pairs += 1;
            }
        }
    }
    std::eprintln!(
        "specifications: {texts} round trips; {equal_pairs} pairs answering alike, \
         {unequal_pairs} not"
    );
    assert!(equal_pairs > 500 && unequal_pairs > 500);
}

/// Texts that must be diagnostics.
const MALFORMED: &[&str] = &[
    "#builtin.data_layout",
    "#builtin.data_layout<>",
    "#builtin.data_layout<little>",
    "#builtin.data_layout<ptr = 32>",
    "#builtin.data_layout<little,>",
    "#builtin.data_layout<little, ptr = 32,>",
    "#builtin.data_layout<little ptr = 32>",
    "#builtin.data_layout<little, big, ptr = 32>",
    "#builtin.data_layout<middle, ptr = 32>",
    "#builtin.data_layout<little, ptr>",
    "#builtin.data_layout<little, ptr = >",
    "#builtin.data_layout<little, ptr = 32",
    "#builtin.data_layout<little, ptr = 0>",
    "#builtin.data_layout<little, ptr = 7>",
    "#builtin.data_layout<little, ptr = 24>",
    "#builtin.data_layout<little, ptr = 72>",
    "#builtin.data_layout<little, ptr = 128>",
    "#builtin.data_layout<little, ptr = 4294967296>",
    "#builtin.data_layout<little, ptr = 99999999999999999999999999999>",
    "#builtin.data_layout<little, ptr = -32>",
    "#builtin.data_layout<little, ptr = 0x20>",
    "#builtin.data_layout<little, ptr = 32 align 0>",
    "#builtin.data_layout<little, ptr = 32 align 3>",
    "#builtin.data_layout<little, ptr = 32 align 65536>",
    "#builtin.data_layout<little, ptr = 32 align 18446744073709551616>",
    "#builtin.data_layout<little, ptr = 32 index 0>",
    "#builtin.data_layout<little, ptr = 32 index 33>",
    "#builtin.data_layout<little, ptr = 32 index 4 align 4>",
    "#builtin.data_layout<little, ptr = 32 align 4 align 4>",
    "#builtin.data_layout<little, ptr = 32, ptr = 32>",
    "#builtin.data_layout<little, ptr = 32, ptr<0> = 32>",
    "#builtin.data_layout<little, ptr<1> = 32>",
    "#builtin.data_layout<little, ptr = 32, ptr<1> = 12>",
    "#builtin.data_layout<little, ptr = 32, ptr<1> = 32 index 64>",
    "#builtin.data_layout<little, ptr = 32, ptr<4294967296> = 32>",
    "#builtin.data_layout<little, ptr = 32, ptr<i8> = 32>",
    "#builtin.data_layout<little, ptr = 32, ptr<i8, 1> = 32>",
    "#builtin.data_layout<little, ptr = 32, i0 = 1>",
    "#builtin.data_layout<little, ptr = 32, i16777216 = 1>",
    "#builtin.data_layout<little, ptr = 32, i8 = 0>",
    "#builtin.data_layout<little, ptr = 32, i8 = 3>",
    "#builtin.data_layout<little, ptr = 32, i8 = 65536>",
    "#builtin.data_layout<little, ptr = 32, i8 = 1, i8 = 1>",
    "#builtin.data_layout<little, ptr = 32, i8 = 1 align 1>",
    "#builtin.data_layout<little, ptr = 32, f32 = 4, f32 = 4>",
    "#builtin.data_layout<little, ptr = 32, f16 = 2>",
    "#builtin.data_layout<little, ptr = 32, isize = 4>",
    "#builtin.data_layout<little, ptr = 32, struct<i8> = 4>",
    "#builtin.data_layout<little, ptr = 32, array<i8; 2> = 4>",
    "#builtin.data_layout<little, ptr = 32, (i32) -> i32 = 4>",
    "#builtin.data_layout<little, ptr = 32, ? = 4>",
    "#builtin.data_layout<little, ptr = 32, never = 4>",
    "#builtin.data_layout<little, ptr = 32, 8 = 1>",
    "#builtin.data_layout<little, ptr = 32, i8 1>",
    "#builtin.data_layout<little, ptr = 32, i8 == 1>",
    "#builtin.data_layout<little, ptr = 32; i8 = 1>",
    "#builtin.data_layout<little, ptr = 32>>",
    "#builtin.data_layout<<little, ptr = 32>",
    "#builtin.data_layout<little, ptr = 32> : i32",
    "#builtin.data_layout(little, ptr = 32)",
];

/// The outcome of parsing `attr` as a module's `data_layout`: `Err(None)` if the parser
/// panicked.
fn parse_guarded(context: &Rc<Context>, attr: &str) -> Result<DataLayoutSpec, Option<String>> {
    match catch_unwind(AssertUnwindSafe(|| parse_spec(context, attr))) {
        Ok(Ok(spec)) => Ok(spec),
        Ok(Err(diagnostic)) => Err(Some(diagnostic)),
        Err(_) => Err(None),
    }
}

#[test]
fn malformed_specifications_are_diagnostics() {
    let context = context();
    let mut accepted = Vec::new();
    let mut panicked = Vec::new();
    for text in MALFORMED {
        match parse_guarded(&context, text) {
            Ok(spec) => accepted.push(format!("{text} => {}", print_spec(&context, &spec))),
            Err(None) => panicked.push(*text),
            Err(Some(_)) => (),
        }
    }
    assert!(panicked.is_empty(), "panicked on:\n{}", panicked.join("\n"));
    assert!(accepted.is_empty(), "accepted:\n{}", accepted.join("\n"));
}

const FRAGMENTS: &[&str] = &[
    "<",
    ">",
    ",",
    "=",
    " ",
    "ptr",
    "ptr<1>",
    "ptr<",
    "align",
    "index",
    "little",
    "big",
    "i8",
    "i64",
    "i0",
    "i16777215",
    "i16777216",
    "f32",
    "f64",
    "isize",
    "0",
    "1",
    "3",
    "8",
    "32",
    "64",
    "65",
    "4294967295",
    "4294967296",
    "18446744073709551615",
    "18446744073709551616",
    "-",
    "-1",
    "0x10",
    "\"",
    ":",
    ";",
    "(",
    ")",
    "{",
    "}",
    "#",
    "#builtin.data_layout",
    "struct<i8>",
    "?",
    "!",
    "@",
    "%",
    "\n",
    "//",
    "é",
    "align 0",
    "index 0",
    "= 4",
    ", i8 = 1",
    ", ptr = 32",
];

/// Random damage to valid texts: the parser answers with a specification or a diagnostic, never
/// a panic, and whatever it accepts prints and parses back to itself.
#[test]
fn damaged_specifications_never_panic() {
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for seed in seeds() {
        std::eprintln!("seed {seed:#x}");
        let mut rng = Rng(seed);
        let mut context = context();
        for case in 0..DAMAGED_TEXTS {
            if case % 250 == 0 {
                context = self::context();
            }
            let model = random_model(&mut rng);
            let mut text = model.text(&mut rng);
            for _ in 0..1 + rng.below(3) {
                // Damage inside the body, sometimes anywhere
                let from = if rng.below(8) == 0 {
                    0
                } else {
                    "#builtin.data_layout".len().min(text.len())
                };
                let mut at = from + rng.below((text.len() - from + 1) as u64) as usize;
                while !text.is_char_boundary(at) {
                    at -= 1;
                }
                match rng.below(5) {
                    0 => text.truncate(at),
                    1 if at < text.len() => {
                        let mut end = (at + 1 + rng.below(4) as usize).min(text.len());
                        while !text.is_char_boundary(end) {
                            end += 1;
                        }
                        text.replace_range(at..end, "");
                    }
                    2 if at < text.len() => {
                        let mut end = (at + 1 + rng.below(6) as usize).min(text.len());
                        while !text.is_char_boundary(end) {
                            end += 1;
                        }
                        let copy = String::from(&text[at..end]);
                        text.insert_str(at, &copy);
                    }
                    _ => text.insert_str(at, rng.pick(FRAGMENTS)),
                }
            }
            match parse_guarded(&context, &text) {
                Ok(spec) => {
                    let printed = print_spec(&context, &spec);
                    let reparsed = parse_spec(&context, &printed)
                        .unwrap_or_else(|err| panic!("{text} => {printed}: {err}"));
                    assert_eq!(reparsed, spec, "{text} => {printed}");
                    assert_eq!(print_spec(&context, &reparsed), printed, "{text}");
                    accepted += 1;
                }
                Err(Some(_)) => rejected += 1,
                Err(None) => panic!("seed {seed:#x}, case {case}: the parser panicked on:\n{text}"),
            }
        }
    }
    std::eprintln!("damaged specifications: {accepted} accepted, {rejected} rejected");
    assert!(accepted > 100 && rejected > 3000);
}

/// A property marked `#[default]` aside, nothing may create a default data layout: the bodyless
/// form is refused in an attribute dictionary of every operation that states one, and in the
/// generic form.
#[test]
fn the_bodyless_form_is_a_diagnostic_everywhere() {
    let context = context();
    let sources = [
        "builtin.module public @m attributes { data_layout = #builtin.data_layout } {\n};",
        "builtin.component private @c attributes { data_layout = #builtin.data_layout } {\n};",
        "builtin.world attributes { data_layout = #builtin.data_layout } {\n};",
        "builtin.module public @m attributes { other = #builtin.data_layout } {\n};",
    ];
    for source in sources {
        let config = ParserConfig::new(context.clone());
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            parse::parse_any(config, "bodyless.runic", source).into_result().map(|_| ())
        }));
        let err = outcome.expect("no panic").expect_err(source);
        let err = describe(&err);
        assert!(err.contains("has no default value"), "{source}: {err}");
    }
}

// --- closest ------------------------------------------------------------------------------------

fn first_child(op: OperationRef) -> OperationRef {
    let op = op.borrow();
    let region = op.region(0);
    let entry = region.entry();
    entry.front().expect("a non-empty body")
}

fn closest_text(context: &Rc<Context>, op: OperationRef) -> Option<String> {
    DataLayout::closest(&op.borrow()).map(|layout| print_spec(context, layout.spec()))
}

/// `closest` from an operation inside a function body, through a world, a component and a
/// module; an operation that does not implement the interface is passed over even if it carries
/// an attribute named `data_layout`.
#[test]
fn closest_walks_from_an_operation_in_a_body_through_every_level() {
    let l = |body: &str| format!("attributes {{ data_layout = #builtin.data_layout<{body}> }}");
    let levels = ["little, ptr = 16", "little, ptr = 32", "little, ptr = 64"];
    for stated in 0..8u32 {
        let context = context();
        let attrs: Vec<String> = (0..3)
            .map(|level| {
                if stated & (1 << level) != 0 {
                    l(levels[level])
                } else {
                    String::new()
                }
            })
            .collect();
        let source = format!(
            "builtin.world {{\n builtin.component private @c {} {{\n  builtin.module internal @m \
             {} {{\n   builtin.function public extern(\"C\") @f() {{\n    builtin.ret;\n   }};\n  \
             }};\n }};\n}};",
            attrs[1], attrs[2]
        );
        let config = ParserConfig::new(context.clone());
        let mut world = parse::parse_any(config, "levels.runic", source.clone())
            .into_result()
            .unwrap_or_else(|err| panic!("{source}\n{}", describe(&err)));
        if stated & 1 != 0 {
            // A world's dictionary is set through the API; the print below shows where it goes
            let spec = parse_spec(&context, &format!("#builtin.data_layout<{}>", levels[0]));
            set_data_layout_spec(&mut world.borrow_mut(), spec.unwrap());
        }
        let component = first_child(world);
        let module = first_child(component);
        let function = first_child(module);
        let ret = first_child(function);
        let expect = |from: usize| {
            (0..=from)
                .rev()
                .find(|level| stated & (1 << level) != 0)
                .map(|level| format!("#builtin.data_layout<{}>", levels[level]))
        };
        assert_eq!(closest_text(&context, world), expect(0), "{source}");
        assert_eq!(closest_text(&context, component), expect(1), "{source}");
        assert_eq!(closest_text(&context, module), expect(2), "{source}");
        assert_eq!(closest_text(&context, function), expect(2), "{source}");
        assert_eq!(closest_text(&context, ret), expect(2), "{source}");

        // An attribute named `data_layout` on an operation without the interface states nothing
        let mut function = function;
        set_data_layout_spec(
            &mut function.borrow_mut(),
            DataLayoutSpec::new(Endianness::Big, PointerSpec::sized(8).unwrap()).unwrap(),
        );
        assert_eq!(closest_text(&context, function), expect(2), "{source}");
        assert_eq!(closest_text(&context, ret), expect(2), "{source}");

        // The whole tree prints and parses back with what each level states
        let printed = format!("{}", world.borrow());
        let config = ParserConfig::new(context.clone());
        let reparsed = parse::parse_any(config, "levels2.runic", printed.clone())
            .into_result()
            .unwrap_or_else(|err| panic!("{printed}\n{}", describe(&err)));
        let component = first_child(reparsed);
        let module = first_child(component);
        assert_eq!(closest_text(&context, reparsed), expect(0), "{printed}");
        assert_eq!(closest_text(&context, component), expect(1), "{printed}");
        assert_eq!(closest_text(&context, module), expect(2), "{printed}");
    }
}

// --- Scalars ------------------------------------------------------------------------------------

#[test]
fn scalars_follow_the_table_under_random_specifications() {
    let mut checked = 0usize;
    for seed in seeds() {
        std::eprintln!("seed {seed:#x}");
        let mut rng = Rng(seed);
        for _ in 0..SCALAR_SPECIFICATIONS {
            let context = context();
            let cx = &*context;
            let model = random_model(&mut rng);
            let layout = DataLayout::new(cx, &model.build(&mut rng));
            let mut widths: BTreeSet<u32> = (1..=140).collect();
            widths.extend(model.probe_widths());
            widths.extend([255, 256, 257, 1024, IntegerType::MAX_WIDTH]);
            for width in widths {
                let ty = cx.int_type(width);
                assert_eq!(size_align(&layout, ty), Ok(model.integer(width)), "i{width} {model:?}");
                assert_eq!(*layout.layout_of(ty).unwrap().fields(), FieldsShape::Scalar);
                checked += 1;
            }
            assert_eq!(layout.isize_width(), model.isize_width());
            assert_eq!(
                size_align(&layout, cx.isize()),
                Ok(model.integer(model.isize_width())),
                "isize {model:?}"
            );
            assert_eq!(size_align(&layout, cx.f32()), Ok(model.float(32)), "{model:?}");
            assert_eq!(size_align(&layout, cx.f64()), Ok(model.float(64)), "{model:?}");
            let function = cx.function_type([cx.i32()], [cx.i32()]);
            for space in [0, 1, 2, 3, 9, 77, u32::MAX] {
                let expected = Ok(model.pointer_layout(space));
                for pointee in [None, Some(cx.i8()), Some(function), Some(cx.never_type())] {
                    let ty = cx.ptr_type_in(pointee, space);
                    assert_eq!(size_align(&layout, ty), expected, "{ty} {model:?}");
                    checked += 1;
                }
            }
            // An array: stride and alignment of its element
            let element = cx.int_type(1 + rng.below(200) as u32);
            let (size, align) = size_align(&layout, element).unwrap();
            let len = rng.below(5);
            let array = cx.array_type(element, len);
            assert_eq!(size_align(&layout, array), Ok((size * len, align)));
            assert_eq!(
                *layout.layout_of(array).unwrap().fields(),
                FieldsShape::Array {
                    stride: size,
                    count: len
                }
            );
        }
    }
    std::eprintln!("scalars: {checked} layouts against the model");
}

// --- Structs and enums against the host compiler ------------------------------------------------

mod host {

    #[repr(C)]
    pub struct A1 {
        pub a: u8,
        pub b: u32,
    }
    #[repr(C)]
    pub struct A2 {
        pub a: u8,
        pub b: u64,
        pub c: u16,
    }
    #[repr(C)]
    pub struct A3 {
        pub a: u8,
        pub b: u128,
        pub c: u8,
    }
    #[repr(C)]
    pub struct A4 {
        pub a: f32,
        pub b: f64,
        pub c: u8,
    }
    #[repr(C)]
    pub struct A5 {
        pub a: *const u8,
        pub b: u8,
        pub c: [u16; 3],
    }
    #[repr(C)]
    pub struct A6 {
        pub a: u8,
        pub b: A2,
        pub c: [A1; 2],
        pub d: u8,
    }
    #[repr(C)]
    pub struct A7 {
        pub a: [u32; 0],
        pub b: u8,
    }
    #[repr(C)]
    pub struct A8 {}
    #[repr(C)]
    pub struct A9 {
        pub a: u8,
        pub b: A8,
        pub c: u16,
    }
    #[repr(C)]
    pub struct A10 {
        pub a: usize,
        pub b: u8,
        pub c: f32,
        pub d: u8,
        pub e: f64,
        pub f: u16,
        pub g: u128,
    }
    #[repr(C)]
    pub struct A11 {
        pub a: u8,
        pub b: [A3; 3],
        pub c: [[u16; 3]; 2],
        pub d: u8,
    }
    #[repr(C, packed)]
    pub struct P1 {
        pub a: u8,
        pub b: u32,
        pub c: u64,
    }
    #[repr(C, packed(2))]
    pub struct P2 {
        pub a: u8,
        pub b: u32,
        pub c: u64,
        pub d: u16,
    }
    #[repr(C, packed(4))]
    pub struct P4 {
        pub a: u8,
        pub b: u64,
        pub c: u128,
        pub d: f64,
        pub e: u16,
    }
    #[repr(C, packed(8))]
    pub struct P8 {
        pub a: u8,
        pub b: u128,
        pub c: u8,
    }
    #[repr(C, packed(16))]
    pub struct P16 {
        pub a: u8,
        pub b: u128,
        pub c: u8,
    }
    #[repr(C, packed(2))]
    pub struct P2n {
        pub a: u8,
        pub b: A2,
        pub c: [u32; 3],
        pub d: P1,
        pub e: *const u8,
    }
    #[repr(C)]
    pub struct N1 {
        pub a: u8,
        pub b: P2,
        pub c: u64,
        pub d: P1,
        pub e: u16,
    }
    #[repr(C, align(16))]
    pub struct L16 {
        pub a: u8,
    }
    #[repr(C, align(8))]
    pub struct L8 {
        pub a: u8,
        pub b: u16,
    }
    #[repr(C, align(2))]
    pub struct L2 {
        pub a: u64,
        pub b: u8,
    }
    #[repr(C, align(64))]
    pub struct L64 {
        pub a: u8,
        pub b: [u8; 65],
    }
    #[repr(C, align(4))]
    pub struct L4e {}
    #[repr(C)]
    pub struct N2 {
        pub a: u8,
        pub b: L16,
        pub c: u8,
        pub d: L8,
        pub e: [L2; 2],
        pub f: L64,
        pub g: L4e,
        pub h: u8,
    }

    #[repr(C, u8)]
    pub enum E1 {
        A(u32),
        B(u8),
    }
    #[repr(C, u32)]
    pub enum E2 {
        A(u8),
    }
    #[repr(C, u8)]
    pub enum E3 {
        A,
        B(u64),
        C([u8; 3]),
    }
    #[repr(C, u16)]
    pub enum E4 {
        A(P1),
        B,
    }
    #[repr(C, u64)]
    pub enum E5 {
        None,
        Some(L16),
    }
    #[repr(C, u8)]
    pub enum E6 {
        A(A1),
        B(u16),
    }
    #[repr(u8)]
    pub enum E7 {
        A,
        B,
        C,
    }
    #[repr(C, u128)]
    pub enum E8 {
        A(u8),
        B(f64),
    }
}

/// The host's scalar alignments as a specification.
fn host_spec_text() -> String {
    use core::mem::align_of;
    format!(
        "{}, ptr = {} align {}, i8 = {}, i16 = {}, i32 = {}, i64 = {}, i128 = {}, f32 = {}, f64 = \
         {}",
        if cfg!(target_endian = "big") {
            "big"
        } else {
            "little"
        },
        usize::BITS,
        align_of::<usize>(),
        align_of::<u8>(),
        align_of::<u16>(),
        align_of::<u32>(),
        align_of::<u64>(),
        align_of::<u128>(),
        align_of::<f32>(),
        align_of::<f64>(),
    )
}

macro_rules! host_struct {
    ($ty:ident, $text:expr, [$($field:ident),*]) => {
        (
            stringify!($ty),
            String::from($text),
            core::mem::size_of::<host::$ty>() as u64,
            core::mem::align_of::<host::$ty>() as u64,
            vec![$(core::mem::offset_of!(host::$ty, $field) as u64),*],
        )
    };
}

#[test]
fn structs_are_laid_out_as_the_host_compiler_lays_out_repr_c() {
    let context = context();
    let spec = parse_spec(&context, &format!("#builtin.data_layout<{}>", host_spec_text()))
        .expect("the host's specification");
    let layout = DataLayout::new(&context, &spec);

    let a1 = "struct<i8, i32>";
    let a2 = "struct<i8, i64, i16>";
    let a3 = "struct<i8, i128, i8>";
    let a8 = "struct<>";
    let p1 = "struct<packed(1); i8, i32, i64>";
    let p2 = "struct<packed(2); i8, i32, i64, i16>";
    let l16 = "struct<align(16); i8>";
    let l8 = "struct<align(8); i8, i16>";
    let l2 = "struct<align(2); i64, i8>";
    let l64 = "struct<align(64); i8, array<i8; 65>>";
    let l4e = "struct<align(4);>";
    let cases = [
        host_struct!(A1, a1, [a, b]),
        host_struct!(A2, a2, [a, b, c]),
        host_struct!(A3, a3, [a, b, c]),
        host_struct!(A4, "struct<f32, f64, i8>", [a, b, c]),
        host_struct!(A5, "struct<ptr<i8>, i8, array<i16; 3>>", [a, b, c]),
        host_struct!(A6, format!("struct<i8, {a2}, array<{a1}; 2>, i8>"), [a, b, c, d]),
        host_struct!(A7, "struct<array<i32; 0>, i8>", [a, b]),
        host_struct!(A8, a8, []),
        host_struct!(A9, format!("struct<i8, {a8}, i16>"), [a, b, c]),
        host_struct!(A10, "struct<isize, i8, f32, i8, f64, i16, i128>", [a, b, c, d, e, f, g]),
        host_struct!(
            A11,
            format!("struct<i8, array<{a3}; 3>, array<array<i16; 3>; 2>, i8>"),
            [a, b, c, d]
        ),
        host_struct!(P1, p1, [a, b, c]),
        host_struct!(P2, p2, [a, b, c, d]),
        host_struct!(P4, "struct<packed(4); i8, i64, i128, f64, i16>", [a, b, c, d, e]),
        host_struct!(P8, "struct<packed(8); i8, i128, i8>", [a, b, c]),
        host_struct!(P16, "struct<packed(16); i8, i128, i8>", [a, b, c]),
        host_struct!(
            P2n,
            format!("struct<packed(2); i8, {a2}, array<i32; 3>, {p1}, ptr>"),
            [a, b, c, d, e]
        ),
        host_struct!(N1, format!("struct<i8, {p2}, i64, {p1}, i16>"), [a, b, c, d, e]),
        host_struct!(L16, l16, [a]),
        host_struct!(L8, l8, [a, b]),
        host_struct!(L2, l2, [a, b]),
        host_struct!(L64, l64, [a, b]),
        host_struct!(L4e, l4e, []),
        host_struct!(
            N2,
            format!("struct<i8, {l16}, i8, {l8}, array<{l2}; 2>, {l64}, {l4e}, i8>"),
            [a, b, c, d, e, f, g, h]
        ),
    ];
    for (name, text, size, align, offsets) in &cases {
        let ty = parse_ty(&context, text);
        let laid_out = layout.layout_of(ty).unwrap_or_else(|err| panic!("{name}: {err}"));
        let FieldsShape::Offsets(runic_offsets) = laid_out.fields() else {
            panic!("{name}: not offsets");
        };
        assert_eq!(
            (laid_out.size(), laid_out.align().bytes(), runic_offsets.to_vec()),
            (*size, *align, offsets.clone()),
            "{name}: {text}"
        );
        // `offset_of` agrees for every field
        for (index, offset) in offsets.iter().enumerate() {
            let (at, _) = layout.offset_of(ty, &[index as u32]).unwrap();
            assert_eq!(at, *offset, "{name}.{index}");
        }
    }
    std::eprintln!("host structs: {} compared under `{}`", cases.len(), host_spec_text());

    // A path through nested types against the compiler's own sum
    let a6 = parse_ty(&context, &format!("struct<i8, {a2}, array<{a1}; 2>, i8>"));
    let expected = core::mem::offset_of!(host::A6, c)
        + core::mem::size_of::<host::A1>()
        + core::mem::offset_of!(host::A1, b);
    assert_eq!(layout.offset_of(a6, &[2, 1, 1]), Ok((expected as u64, context.i32())));
}

/// The offset of the payload of `value`'s variant from the start of `value`.
fn payload_offset<E, P>(value: &E, payload: &P) -> u64 {
    (payload as *const P as usize - value as *const E as usize) as u64
}

/// A `#[repr(C, uN)]` enum is a tag followed by a union of the payloads (RFC 2195), which is the
/// design's rule for `enum`.
#[test]
fn enums_are_laid_out_as_the_host_compiler_lays_out_repr_c_enums() {
    use core::mem::{align_of, size_of};

    use host::*;

    // Every variant is a value of its enum: the layouts compared below are of these enums
    core::hint::black_box((
        [E1::A(0), E1::B(0)],
        [E3::A, E3::B(0), E3::C([0; 3])],
        [E4::A(P1 { a: 0, b: 0, c: 0 }), E4::B],
        [E5::None, E5::Some(L16 { a: 0 })],
        [E6::A(A1 { a: 0, b: 0 }), E6::B(0)],
        [E7::A, E7::B, E7::C],
        [E8::A(0), E8::B(0.0)],
    ));

    let context = context();
    let spec = parse_spec(&context, &format!("#builtin.data_layout<{}>", host_spec_text()))
        .expect("the host's specification");
    let layout = DataLayout::new(&context, &spec);
    let p1 = "struct<packed(1); i8, i32, i64>";
    let l16 = "struct<align(16); i8>";

    let e1 = E1::A(7);
    let E1::A(e1_payload) = &e1 else {
        unreachable!()
    };
    let e1b = E1::B(7);
    let E1::B(e1b_payload) = &e1b else {
        unreachable!()
    };
    assert_eq!(payload_offset(&e1, e1_payload), payload_offset(&e1b, e1b_payload));
    let e2 = E2::A(1);
    let E2::A(e2_payload) = &e2;
    let e3 = E3::B(1);
    let E3::B(e3_payload) = &e3 else {
        unreachable!()
    };
    let e4 = E4::A(P1 { a: 0, b: 0, c: 0 });
    let E4::A(e4_payload) = &e4 else {
        unreachable!()
    };
    let e5 = E5::Some(L16 { a: 0 });
    let E5::Some(e5_payload) = &e5 else {
        unreachable!()
    };
    let e6 = E6::A(A1 { a: 0, b: 0 });
    let E6::A(e6_payload) = &e6 else {
        unreachable!()
    };
    let e8 = E8::B(0.0);
    let E8::B(e8_payload) = &e8 else {
        unreachable!()
    };
    // Every payload of an enum lies at one offset
    let e3c = E3::C([0; 3]);
    let E3::C(e3c_payload) = &e3c else {
        unreachable!()
    };
    assert_eq!(payload_offset(&e3c, e3c_payload), payload_offset(&e3, e3_payload));
    let e6b = E6::B(0);
    let E6::B(e6b_payload) = &e6b else {
        unreachable!()
    };
    assert_eq!(payload_offset(&e6b, e6b_payload), payload_offset(&e6, e6_payload));
    let e8a = E8::A(0);
    let E8::A(e8a_payload) = &e8a else {
        unreachable!()
    };
    assert_eq!(payload_offset(&e8a, e8a_payload), payload_offset(&e8, e8_payload));

    let cases: [(&str, String, usize, usize, Option<u64>); 8] = [
        (
            "E1",
            "enum<i8; a: i32, b: i8>".into(),
            size_of::<E1>(),
            align_of::<E1>(),
            Some(payload_offset(&e1, e1_payload)),
        ),
        (
            "E2",
            "enum<i32; a: i8>".into(),
            size_of::<E2>(),
            align_of::<E2>(),
            Some(payload_offset(&e2, e2_payload)),
        ),
        (
            "E3",
            "enum<i8; a, b: i64, c: array<i8; 3>>".into(),
            size_of::<E3>(),
            align_of::<E3>(),
            Some(payload_offset(&e3, e3_payload)),
        ),
        (
            "E4",
            format!("enum<i16; a: {p1}, b>"),
            size_of::<E4>(),
            align_of::<E4>(),
            Some(payload_offset(&e4, e4_payload)),
        ),
        (
            "E5",
            format!("enum<i64; none, some: {l16}>"),
            size_of::<E5>(),
            align_of::<E5>(),
            Some(payload_offset(&e5, e5_payload)),
        ),
        (
            "E6",
            "enum<i8; a: struct<i8, i32>, b: i16>".into(),
            size_of::<E6>(),
            align_of::<E6>(),
            Some(payload_offset(&e6, e6_payload)),
        ),
        ("E7", "enum<i8; a, b, c>".into(), size_of::<E7>(), align_of::<E7>(), None),
        (
            "E8",
            "enum<i128; a: i8, b: f64>".into(),
            size_of::<E8>(),
            align_of::<E8>(),
            Some(payload_offset(&e8, e8_payload)),
        ),
    ];
    for (name, text, size, align, payload) in cases {
        let ty = parse_ty(&context, &text);
        let laid_out = layout.layout_of(ty).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(
            (laid_out.size(), laid_out.align().bytes()),
            (size as u64, align as u64),
            "{name}: {text}"
        );
        let FieldsShape::Offsets(offsets) = laid_out.fields() else {
            panic!("{name}: not offsets");
        };
        assert_eq!(offsets[0], 0, "{name}");
        if let Some(payload) = payload {
            assert!(offsets[1..].iter().all(|offset| *offset == payload), "{name}: {offsets:?}");
        }
    }
}

// --- Random aggregates against a reference written here ------------------------------------------

/// A type with a layout, as a tree the reference can lay out itself.
#[derive(Clone, Debug)]
enum T {
    Int(u32),
    Isize,
    F32,
    F64,
    Ptr(u32),
    Array(Box<T>, u64),
    Struct(Repr, Vec<(T, Option<u16>)>),
    /// Discriminant width (0 for `isize`), then each variant's payload.
    Enum(u32, Vec<Option<T>>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Repr {
    Default,
    Align(u16),
    Packed(u16),
    Transparent,
}

/// `(size, align, offsets)`; the offsets are empty for a scalar or an array.
type Shape = (u64, u64, Vec<u64>);

fn reference(model: &Model, ty: &T) -> Shape {
    match ty {
        T::Int(width) => {
            let (size, align) = model.integer(*width);
            (size, align, vec![])
        }
        T::Isize => {
            let (size, align) = model.integer(model.isize_width());
            (size, align, vec![])
        }
        T::F32 => (model.float(32).0, model.float(32).1, vec![]),
        T::F64 => (model.float(64).0, model.float(64).1, vec![]),
        T::Ptr(space) => {
            let (size, align) = model.pointer_layout(*space);
            (size, align, vec![])
        }
        T::Array(element, len) => {
            let (size, align, _) = reference(model, element);
            (size * len, align, vec![])
        }
        T::Struct(Repr::Transparent, fields) => {
            let (size, align, _) = reference(model, &fields[0].0);
            (size, align, vec![0])
        }
        T::Struct(repr, fields) => {
            let mut end = 0;
            let mut struct_align = 1;
            let mut offsets = Vec::new();
            for (field, stated) in fields {
                let (size, natural, _) = reference(model, field);
                let capped = match repr {
                    Repr::Packed(cap) => natural.min(u64::from(*cap)),
                    _ => natural,
                };
                let effective = capped.max(stated.map_or(1, u64::from));
                let offset = round_up(end, effective);
                // The invariants, stated apart from the rule
                assert!(
                    offset >= end && offset - end < effective && offset.is_multiple_of(effective)
                );
                offsets.push(offset);
                end = offset + size;
                struct_align = struct_align.max(effective);
            }
            if let Repr::Align(raised) = repr {
                struct_align = struct_align.max(u64::from(*raised));
            }
            (round_up(end, struct_align), struct_align, offsets)
        }
        T::Enum(width, payloads) => {
            let width = if *width == 0 {
                model.isize_width()
            } else {
                *width
            };
            let (tag_size, tag_align) = model.integer(width);
            let mut payload_align = 1;
            let mut payload_size = 0;
            for payload in payloads.iter().flatten() {
                let (size, align, _) = reference(model, payload);
                payload_align = payload_align.max(align);
                payload_size = payload_size.max(size);
            }
            let offset = round_up(tag_size, payload_align);
            let align = tag_align.max(payload_align);
            let mut offsets = vec![0];
            offsets.extend(payloads.iter().map(|_| offset));
            (round_up(offset + payload_size, align), align, offsets)
        }
    }
}

fn build(cx: &Context, ty: &T) -> Type {
    let stated = |n: u16| core::num::NonZeroU16::new(n).unwrap();
    match ty {
        T::Int(width) => cx.int_type(*width),
        T::Isize => cx.isize(),
        T::F32 => cx.f32(),
        T::F64 => cx.f64(),
        T::Ptr(space) => cx.ptr_type_in(None, *space),
        T::Array(element, len) => cx.array_type(build(cx, element), *len),
        T::Struct(repr, fields) => {
            let fields = fields.iter().map(|(field, align)| {
                let field = StructField::new(build(cx, field));
                match align {
                    Some(align) => field.aligned(stated(*align)),
                    None => field,
                }
            });
            let repr = match repr {
                Repr::Default => StructRepr::Default,
                Repr::Align(n) => StructRepr::Align(stated(*n)),
                Repr::Packed(n) => StructRepr::Packed(stated(*n)),
                Repr::Transparent => StructRepr::Transparent,
            };
            cx.struct_type(StructType::new(fields).with_repr(repr))
        }
        T::Enum(width, payloads) => {
            let discriminant = if *width == 0 {
                cx.isize()
            } else {
                cx.int_type(*width)
            };
            let variants = payloads.iter().enumerate().map(|(index, payload)| {
                let variant = EnumVariant::new(Symbol::intern(format!("v{index}").as_str()));
                match payload {
                    Some(payload) => variant.with_payload(build(cx, payload)),
                    None => variant,
                }
            });
            cx.enum_type(EnumType::new(None, discriminant, variants).expect("a valid enum"))
        }
    }
}

fn random_stated(rng: &mut Rng) -> u16 {
    {
        let end = if rng.below(6) == 0 { 16 } else { 6 };
        1 << rng.below(end)
    }
}

fn random_t(rng: &mut Rng, depth: u32) -> T {
    let kinds = if depth == 0 { 6 } else { 12 };
    match rng.below(kinds) {
        0 | 1 => T::Int(rng.pick(&[1, 7, 8, 16, 24, 32, 33, 64, 65, 128, 129, 300])),
        2 => T::Isize,
        3 => T::F32,
        4 => T::F64,
        5 => T::Ptr(rng.pick(&[0, 1, 2, 77])),
        6 | 7 => T::Array(Box::new(random_t(rng, depth - 1)), rng.below(4)),
        8 => {
            // At most 2 variants for an `i1` discriminant, or an `isize` one, whose width may
            // be 1 bit under a random specification
            let width = rng.pick(&[0, 1, 8, 16, 32, 64, 128]);
            let count = if width <= 1 {
                rng.below(3)
            } else {
                rng.below(5)
            };
            let payloads = (0..count)
                .map(|_| (rng.below(3) != 0).then(|| random_t(rng, depth - 1)))
                .collect();
            T::Enum(width, payloads)
        }
        _ => {
            let repr = match rng.below(7) {
                0 | 1 => Repr::Default,
                2 => Repr::Align(random_stated(rng)),
                3..=5 => Repr::Packed(random_stated(rng)),
                _ => Repr::Transparent,
            };
            if repr == Repr::Transparent {
                return T::Struct(repr, vec![(random_t(rng, depth - 1), None)]);
            }
            let fields = (0..rng.below(6))
                .map(|_| {
                    let align = (rng.below(3) == 0).then(|| random_stated(rng));
                    (random_t(rng, depth - 1), align)
                })
                .collect();
            T::Struct(repr, fields)
        }
    }
}

/// Counts the structs of `ty` that combine `packed(N)` with a field `align(N)`.
fn combined(ty: &T) -> usize {
    match ty {
        T::Array(element, _) => combined(element),
        T::Struct(repr, fields) => {
            let here =
                matches!(repr, Repr::Packed(_)) && fields.iter().any(|(_, align)| align.is_some());
            usize::from(here) + fields.iter().map(|(field, _)| combined(field)).sum::<usize>()
        }
        T::Enum(_, payloads) => payloads.iter().flatten().map(combined).sum(),
        _ => 0,
    }
}

/// Compares the layout of `ty` and of everything inside it with the reference.
fn assert_as_reference(layout: &DataLayout, cx: &Context, model: &Model, ty: &T) {
    let built = build(cx, ty);
    let (size, align, offsets) = reference(model, ty);
    let laid_out = layout.layout_of(built).unwrap_or_else(|err| panic!("{built}: {err}"));
    assert_eq!(
        (laid_out.size(), laid_out.align().bytes()),
        (size, align),
        "{built} under {model:?}"
    );
    assert_eq!(size % align, 0);
    match (ty, laid_out.fields()) {
        (T::Struct(_, fields), FieldsShape::Offsets(got)) => {
            assert_eq!(got.to_vec(), offsets, "{built} under {model:?}");
            // No overlap, in field order, inside the struct
            let mut end = 0;
            for ((field, _), offset) in fields.iter().zip(got.iter()) {
                assert!(*offset >= end, "{built}: overlap");
                end = offset + reference(model, field).0;
                assert_as_reference(layout, cx, model, field);
            }
            assert!(end <= size);
            for index in 0..fields.len() {
                let (at, element) = layout.offset_of(built, &[index as u32]).unwrap();
                assert_eq!((at, element), (offsets[index], build(cx, &fields[index].0)));
            }
            assert!(matches!(
                layout.offset_of(built, &[fields.len() as u32]),
                Err(LayoutError::NoSuchElement { .. })
            ));
        }
        (T::Enum(_, payloads), FieldsShape::Offsets(got)) => {
            assert_eq!(got.to_vec(), offsets, "{built} under {model:?}");
            for (index, payload) in payloads.iter().enumerate() {
                let element = layout.offset_of(built, &[1 + index as u32]);
                match payload {
                    Some(payload) => {
                        assert_eq!(element, Ok((offsets[1 + index], build(cx, payload))));
                        assert!(offsets[1 + index] + reference(model, payload).0 <= size);
                        assert_as_reference(layout, cx, model, payload);
                    }
                    None => assert!(matches!(element, Err(LayoutError::NoSuchElement { .. }))),
                }
            }
        }
        (T::Array(element, len), FieldsShape::Array { stride, count }) => {
            assert_eq!((*stride, *count), (reference(model, element).0, *len));
            assert_as_reference(layout, cx, model, element);
        }
        (T::Struct(..) | T::Enum(..) | T::Array(..), fields) => panic!("{built}: {fields:?}"),
        (_, fields) => assert_eq!(*fields, FieldsShape::Scalar),
    }
}

#[test]
fn random_aggregates_are_laid_out_as_the_reference() {
    let mut cases = 0usize;
    let mut packed_with_field_align = 0usize;
    for seed in seeds() {
        std::eprintln!("seed {seed:#x}");
        let mut rng = Rng(seed);
        for _ in 0..AGGREGATE_SPECIFICATIONS {
            let context = context();
            let cx = &*context;
            let model = random_model(&mut rng);
            let layout = DataLayout::new(cx, &model.build(&mut rng));
            for _ in 0..25 {
                let ty = random_t(&mut rng, 3);
                packed_with_field_align += combined(&ty);
                assert_as_reference(&layout, cx, &model, &ty);
                cases += 1;
            }
            assert!(layout.is_idle());
        }
    }
    std::eprintln!(
        "aggregates: {cases} random types as the reference; {packed_with_field_align} structs \
         combine packed(N) with a field align(N)"
    );
    assert!(packed_with_field_align > 1000);
}

// --- Recursion ----------------------------------------------------------------------------------

/// A field of a struct in a generated graph of types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Scalar,
    /// Type `n` of the graph by value.
    Value(usize),
    /// `array<type n; len>`.
    Array(usize, u64),
    /// `enum<i8; a, b: type n>`.
    Payload(usize),
    /// `ptr<type n>`.
    Pointer(usize),
    /// A function type: no layout.
    Function,
    /// `array<i64; 2^61>`: too large.
    Huge,
    /// `enum<isize; a = 2^40>`: too wide for a 32-bit `isize`.
    WideEnum,
}

/// A graph of struct types: the first `members` are one recursion group, the rest are ordinary
/// structs that may hold members and earlier ordinary structs.
#[derive(Clone, Debug)]
struct Graph {
    members: usize,
    fields: Vec<Vec<Field>>,
}

/// What the layout of a type of a graph is expected to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expected {
    Finite,
    /// Infinite, named after the type asked.
    Infinite,
    NoLayout,
    TooLarge,
    TooWide,
}

impl Graph {
    /// The answer for type `n`, from the definition: the error of the first field, in order,
    /// that has no layout; a type met again while it is being answered is infinite. Computed
    /// with a visiting set of its own for each type asked, so it does not depend on any order.
    fn expected(&self, n: usize) -> Expected {
        self.expect_in(n, &mut vec![false; self.fields.len()])
    }

    fn expect_in(&self, n: usize, visiting: &mut Vec<bool>) -> Expected {
        if visiting[n] {
            return Expected::Infinite;
        }
        visiting[n] = true;
        let mut answer = Expected::Finite;
        for field in &self.fields[n] {
            let of_field = match *field {
                Field::Scalar | Field::Pointer(_) => Expected::Finite,
                Field::Value(m) | Field::Array(m, _) | Field::Payload(m) => {
                    self.expect_in(m, visiting)
                }
                Field::Function => Expected::NoLayout,
                Field::Huge => Expected::TooLarge,
                Field::WideEnum => Expected::TooWide,
            };
            if of_field != Expected::Finite {
                answer = of_field;
                break;
            }
        }
        visiting[n] = false;
        answer
    }

    fn build(&self, cx: &Context) -> Vec<Type> {
        let function = cx.function_type([cx.i32()], [cx.i32()]);
        let huge = cx.array_type(cx.i64(), 1 << 61);
        let wide = cx.enum_type(
            EnumType::new(None, cx.isize(), [EnumVariant::new(sym("a")).with_value(1 << 40)])
                .unwrap(),
        );
        let field = |field: Field, of: &dyn Fn(usize) -> Type| -> StructField {
            let ty = match field {
                Field::Scalar => cx.i32(),
                Field::Value(m) => of(m),
                Field::Array(m, len) => cx.array_type(of(m), len),
                Field::Payload(m) => cx.enum_type(
                    EnumType::new(
                        None,
                        cx.i8(),
                        [
                            EnumVariant::new(sym("a")),
                            EnumVariant::new(sym("b")).with_payload(of(m)),
                        ],
                    )
                    .unwrap(),
                ),
                Field::Pointer(m) => cx.ptr_type(of(m)),
                Field::Function => function,
                Field::Huge => huge,
                Field::WideEnum => wide,
            };
            StructField::new(ty)
        };
        let bodies: Vec<Type> = (0..self.members)
            .map(|n| {
                let fields = self.fields[n]
                    .iter()
                    .map(|f| field(*f, &|m| cx.rec_var(m as u32)))
                    .collect::<Vec<_>>();
                cx.struct_type(StructType::named(sym(&format!("m{n}")), fields))
            })
            .collect();
        let mut types: Vec<Type> = cx.rec_group(&bodies).expect("a valid group").to_vec();
        for (n, of_type) in self.fields.iter().enumerate().skip(self.members) {
            let fields = of_type.iter().map(|f| field(*f, &|m| types[m])).collect::<Vec<_>>();
            types.push(cx.struct_type(StructType::named(sym(&format!("o{n}")), fields)));
        }
        types
    }
}

fn sym(name: &str) -> Symbol {
    Symbol::intern(name)
}

fn random_graph(rng: &mut Rng, members: usize, outsiders: usize) -> Graph {
    let mut fields = Vec::new();
    for n in 0..members + outsiders {
        // A member refers to any member; an outsider to any member or an earlier outsider
        let targets = if n < members { members } else { n };
        let count = 1 + rng.below(3);
        let of_type = (0..count)
            .map(|_| {
                let target = rng.below(targets as u64) as usize;
                match rng.below(14) {
                    0 | 1 => Field::Scalar,
                    2..=5 => Field::Value(target),
                    6 => Field::Array(target, rng.pick(&[0, 1, 3])),
                    7 => Field::Payload(target),
                    8 | 9 => Field::Pointer(target),
                    10 | 11 => Field::Function,
                    12 => Field::Huge,
                    _ => Field::WideEnum,
                }
            })
            .collect();
        fields.push(of_type);
    }
    Graph { members, fields }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut all = Vec::new();
    for rest in permutations(n - 1) {
        for at in 0..=rest.len() {
            let mut order = rest.clone();
            order.insert(at, n - 1);
            all.push(order);
        }
    }
    all
}

/// A specification that answers the builtin types as `little, ptr = 32` does, with a cache of
/// its own: it differs in an address space no type here uses.
fn fresh_little32(cx: &Context, nth: usize) -> DataLayout {
    let spec = DataLayoutSpec::new(Endianness::Little, PointerSpec::sized(32).unwrap())
        .unwrap()
        .with_pointer(
            1_000_000,
            PointerSpec {
                size_bits: 64,
                align: Align::ONE,
                index_bits: 1 + (nth % 64) as u32,
            },
        )
        .unwrap()
        .with_pointer(
            1_000_001 + (nth / 64) as u32,
            PointerSpec {
                size_bits: 8,
                align: Align::ONE,
                index_bits: 8,
            },
        )
        .unwrap();
    DataLayout::new(cx, &spec)
}

/// Checks `graph` in every query order, each order with a cache of its own. Returns how many of
/// its types are infinite and how many have another error.
fn check_every_order(graph: &Graph, case: &str) -> (usize, usize) {
    let context = context();
    let cx = &*context;
    let types = graph.build(cx);
    let expected: Vec<Expected> = (0..types.len()).map(|n| graph.expected(n)).collect();
    let as_answer = |n: usize, answer: &Result<Rc<Layout>, LayoutError>| match answer {
        Ok(_) => Expected::Finite,
        Err(LayoutError::InfiniteSize(ty)) => {
            assert_eq!(*ty, types[n], "{case}: an infinite type names itself");
            Expected::Infinite
        }
        Err(LayoutError::NoLayout(_)) => Expected::NoLayout,
        Err(LayoutError::TooLarge(_)) => Expected::TooLarge,
        Err(LayoutError::DiscriminantTooWide(_)) => Expected::TooWide,
        Err(other) => panic!("{case}: {other:?}"),
    };
    let mut first: Option<Vec<Result<Rc<Layout>, LayoutError>>> = None;
    for (nth, order) in permutations(types.len()).into_iter().enumerate() {
        let layout = fresh_little32(cx, nth);
        let mut answers: Vec<Option<Result<Rc<Layout>, LayoutError>>> = vec![None; types.len()];
        for &n in &order {
            let answer = layout.layout_of(types[n]);
            assert!(layout.is_idle(), "{case}");
            assert_eq!(
                as_answer(n, &answer),
                expected[n],
                "{case}: type {n} asked in order {order:?}: {answer:?}\n{graph:?}"
            );
            answers[n] = Some(answer);
        }
        let answers: Vec<_> = answers.into_iter().map(Option::unwrap).collect();
        // From the cache, and through `offset_of`, the same
        for (n, answer) in answers.iter().enumerate() {
            assert_eq!(&layout.layout_of(types[n]), answer, "{case}: type {n} again");
            if let Err(err) = answer {
                assert_eq!(layout.offset_of(types[n], &[]), Err(err.clone()), "{case}");
            }
        }
        match &first {
            None => first = Some(answers),
            Some(first) => assert_eq!(
                first.iter().map(|a| a.as_ref().map(|l| (**l).clone())).collect::<Vec<_>>(),
                answers.iter().map(|a| a.as_ref().map(|l| (**l).clone())).collect::<Vec<_>>(),
                "{case}: order {order:?} answers differently\n{graph:?}"
            ),
        }
    }
    (
        expected.iter().filter(|e| **e == Expected::Infinite).count(),
        expected
            .iter()
            .filter(|e| !matches!(e, Expected::Infinite | Expected::Finite))
            .count(),
    )
}

/// Every query order of random graphs of two and three mutually recursive types with up to two
/// types that hold them, with fields that have no layout mixed in.
#[test]
fn recursion_answers_do_not_depend_on_the_query_order() {
    let mut graphs = 0usize;
    let mut infinite = 0usize;
    let mut other = 0usize;
    for seed in seeds() {
        std::eprintln!("seed {seed:#x}");
        let mut rng = Rng(seed);
        for case in 0..GRAPHS {
            let members = 2 + rng.below(2) as usize;
            let outsiders = rng.below(3) as usize;
            let graph = random_graph(&mut rng, members, outsiders);
            let (inf, err) = check_every_order(&graph, &format!("seed {seed:#x}, case {case}"));
            infinite += inf;
            other += err;
            graphs += 1;
        }
    }
    std::eprintln!(
        "recursion: {graphs} graphs in every query order; {infinite} infinite types, {other} with \
         another error"
    );
    assert!(infinite > 500 && other > 500);
}

/// Named cases, written out: cycles of two and three, a type holding a member, and
/// a type with both a cycle and a field without a layout, in both field orders.
#[test]
fn recursion_named_cases_in_every_order() {
    use Field::*;
    let cases: [(&str, Graph); 6] = [
        (
            "two by value, a holder",
            Graph {
                members: 2,
                fields: vec![vec![Scalar, Value(1)], vec![Value(0)], vec![Scalar, Value(1)]],
            },
        ),
        (
            "three in a ring, holders of each",
            Graph {
                members: 3,
                fields: vec![
                    vec![Value(1)],
                    vec![Array(2, 2)],
                    vec![Payload(0)],
                    vec![Value(2)],
                    vec![Value(3), Value(0)],
                ],
            },
        ),
        (
            "a cycle, then a field without a layout",
            Graph {
                members: 2,
                fields: vec![vec![Value(1), Function], vec![Value(0)], vec![Value(0), Function]],
            },
        ),
        (
            "a field without a layout, then a cycle",
            Graph {
                members: 2,
                fields: vec![vec![Function, Value(1)], vec![Value(0)], vec![Function, Value(1)]],
            },
        ),
        (
            "a cycle broken by a pointer, and one that is not",
            Graph {
                members: 3,
                fields: vec![vec![Pointer(1), Value(2)], vec![Value(0)], vec![Value(2)]],
            },
        ),
        (
            "a cycle entered from a type too wide for isize",
            Graph {
                members: 3,
                fields: vec![
                    vec![Value(1), WideEnum],
                    vec![WideEnum, Value(2)],
                    vec![Value(0), Huge],
                    vec![Value(0), Value(1)],
                ],
            },
        ),
    ];
    for (name, graph) in cases {
        check_every_order(&graph, name);
    }
}

/// A by-value cycle of `count` structs, and a holder of the first.
fn ring(cx: &Context, count: usize) -> Vec<Type> {
    let bodies: Vec<Type> = (0..count)
        .map(|n| {
            let next = cx.rec_var(((n + 1) % count) as u32);
            cx.struct_type(StructType::named(
                sym(&format!("ring{count}_{n}")),
                [cx.i8().into(), StructField::new(next)],
            ))
        })
        .collect();
    cx.rec_group(&bodies).expect("a valid group").to_vec()
}

#[test]
fn a_cycle_of_several_hundred_types_is_infinite_from_any_start() {
    for count in [300usize, 800] {
        for start in [0, 1, count / 2, count - 1] {
            let context = context();
            let cx = &*context;
            let types = ring(cx, count);
            let layout = fresh_little32(cx, 0);
            assert_eq!(
                layout.layout_of(types[start]),
                Err(LayoutError::InfiniteSize(types[start]))
            );
            assert!(layout.is_idle());
            for ty in &types {
                assert_eq!(layout.layout_of(*ty), Err(LayoutError::InfiniteSize(*ty)));
            }
        }
    }
}

/// `depth` structs, each holding the next by value after an `i8`; the innermost is `i32`.
fn nested(cx: &Context, depth: usize) -> Type {
    let mut ty = cx.i32();
    for _ in 0..depth {
        ty = cx.struct_type(StructType::new([cx.i8().into(), StructField::new(ty)]));
    }
    ty
}

#[test]
fn deep_nesting_without_a_cycle_has_a_layout() {
    let context = context();
    let cx = &*context;
    let layout = fresh_little32(cx, 0);
    // Each level adds 4 bytes: an `i8`, padding, and the 4-aligned level below
    for depth in [10usize, 100, 400] {
        let ty = nested(cx, depth);
        let laid_out = layout.layout_of(ty).unwrap();
        assert_eq!((laid_out.size(), laid_out.align().bytes()), (4 * depth as u64 + 4, 4));
    }
    // Asked from the inside out, any depth is one step at a time
    let mut ty = cx.i32();
    for depth in 1..=50_000u64 {
        ty = cx.struct_type(StructType::new([cx.i8().into(), StructField::new(ty)]));
        assert_eq!(layout.size_of(ty), Ok(4 * depth + 4));
    }
}

// --- Caches and reentrancy -----------------------------------------------------------------------

mod reentrant {
    use super::*;
    use crate::{derive::DialectType, dialects::test::TestDialect};

    /// A dialect type whose layout asks other layouts while it is being computed: of other
    /// types under the same specification, through a second handle to it, and under another
    /// specification of the same context.
    #[derive(Debug, PartialEq, Eq, Hash, DialectType)]
    #[ty(dialect = TestDialect, name = "layout_reentrant", implements(TypeLayout, AggregateType))]
    pub(super) struct ReentrantType {
        pub inner: Type,
        pub context: usize,
    }

    impl TypeLayout for ReentrantType {
        fn layout(&self, _ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
            // SAFETY: the test keeps the context alive for as long as the type is used
            let cx = unsafe { &*(self.context as *const Context) };
            let inner = layout.layout_of(self.inner)?;
            // The same specification through a new handle shares the cache being filled
            let again = DataLayout::new(cx, layout.spec());
            assert!(again.shares_cache_with(layout));
            assert_eq!(again.layout_of(self.inner)?, inner);
            let pair = cx.struct_type(StructType::new([self.inner.into(), cx.i64().into()]));
            let pair = again.layout_of(pair)?;
            // Another specification of the same context, unless this is already the query
            // made under it
            let other = DataLayoutSpec::new(Endianness::Big, PointerSpec::sized(64).unwrap())
                .unwrap()
                .with_integer_align(64, Align::ONE)
                .unwrap();
            let elsewhere = if *layout.spec() == other {
                inner.clone()
            } else {
                let other = DataLayout::new(cx, &other);
                assert!(!other.shares_cache_with(layout));
                other.layout_of(self.inner)?
            };
            let size = inner.size() + pair.size() + elsewhere.size();
            let align = inner.align().max(pair.align());
            Ok(Layout::new(
                align.align_up(size).unwrap(),
                align,
                FieldsShape::Offsets([0].into()),
            ))
        }
    }

    impl crate::types::TypePrinter for ReentrantType {
        fn print(&self, printer: &mut crate::types::TypeFormatter<'_, '_>) -> core::fmt::Result {
            printer.ty(self.inner)
        }
    }

    impl crate::types::TypeParser for ReentrantType {
        fn parse(parser: &mut dyn crate::parse::Parser<'_>) -> crate::parse::ParseResult<Type> {
            let inner = parser.parse_type()?.into_inner();
            let context = parser.context() as *const Context as usize;
            Ok(parser.context().intern_type(Self { inner, context }))
        }
    }

    impl AggregateType for ReentrantType {
        fn element_count(&self) -> u64 {
            1
        }

        fn element_type(&self, index: u32) -> Option<Type> {
            (index < 1).then_some(self.inner)
        }
    }

    /// The layout of an `i32` with one element, at offset 0.
    fn one_element() -> Layout {
        Layout::new(4, Align::from_bytes(4).unwrap(), FieldsShape::Offsets([0].into()))
    }

    /// A type that counts one element but gives a type for a second, which its layout has no
    /// offset for: its interfaces disagree.
    #[derive(Debug, PartialEq, Eq, Hash, DialectType)]
    #[ty(
        dialect = TestDialect,
        name = "layout_uncounted_element",
        implements(TypeLayout, AggregateType)
    )]
    pub(super) struct UncountedElementType {
        pub element: Type,
    }

    impl crate::types::TypePrinter for UncountedElementType {
        fn print(&self, printer: &mut crate::types::TypeFormatter<'_, '_>) -> core::fmt::Result {
            printer.ty(self.element)
        }
    }

    impl crate::types::TypeParser for UncountedElementType {
        fn parse(parser: &mut dyn crate::parse::Parser<'_>) -> crate::parse::ParseResult<Type> {
            let element = parser.parse_type()?.into_inner();
            Ok(parser.context().intern_type(Self { element }))
        }
    }

    impl TypeLayout for UncountedElementType {
        fn layout(&self, _ty: Type, _layout: &DataLayout) -> Result<Layout, LayoutError> {
            Ok(one_element())
        }
    }

    impl AggregateType for UncountedElementType {
        fn element_count(&self) -> u64 {
            1
        }

        fn element_type(&self, index: u32) -> Option<Type> {
            (index < 2).then_some(self.element)
        }
    }

    /// A type that counts two elements, which its layout gives one offset for.
    #[derive(Debug, Default, PartialEq, Eq, Hash, DialectType)]
    #[ty(
        dialect = TestDialect,
        name = "layout_missing_offset",
        implements(TypeLayout, AggregateType)
    )]
    pub(super) struct MissingOffsetType;

    impl TypeLayout for MissingOffsetType {
        fn layout(&self, _ty: Type, _layout: &DataLayout) -> Result<Layout, LayoutError> {
            Ok(one_element())
        }
    }

    impl AggregateType for MissingOffsetType {
        fn element_count(&self) -> u64 {
            2
        }

        fn element_type(&self, _index: u32) -> Option<Type> {
            None
        }
    }
}

#[test]
fn a_type_may_ask_for_other_layouts_while_it_is_laid_out() {
    use reentrant::ReentrantType;

    let context = context();
    let cx = &*context;
    let layout = fresh_little32(cx, 0);
    let at = cx as *const Context as usize;
    let leaf = cx.intern_type(ReentrantType {
        inner: cx.i32(),
        context: at,
    });
    // i32 (4) + struct<i32, i64> (16) + i32 under the other specification (4), aligned to 8
    assert_eq!(size_align(&layout, leaf), Ok((24, 8)));
    let outer = cx.intern_type(ReentrantType {
        inner: cx.array_type(leaf, 2),
        context: at,
    });
    assert!(layout.layout_of(outer).is_ok());
    assert!(layout.is_idle());

    // A reentrant type around a by-value cycle is infinite, and names itself
    let ring = ring(cx, 3);
    let holder = cx.intern_type(ReentrantType {
        inner: ring[1],
        context: at,
    });
    assert_eq!(layout.layout_of(holder), Err(LayoutError::InfiniteSize(holder)));
    assert!(layout.is_idle());
}

/// Different specifications of one context never see each other's answers, for a type whose
/// layout differs and for one whose error differs.
#[test]
fn different_specifications_keep_their_answers_apart() {
    let context = context();
    let cx = &*context;
    let seed = seeds()[0];
    std::eprintln!("seed {seed:#x}");
    let mut rng = Rng(seed);
    let types: Vec<Type> = [
        "struct<i8, i64, ptr, isize>",
        "enum<isize; a = 1099511627776: i64>",
        "array<struct<packed(2); i16, i128>; 3>",
        "struct<i8, (i32) -> i32>",
    ]
    .iter()
    .map(|text| parse_ty(&context, text))
    .collect();
    let mut layouts: Vec<(Model, DataLayoutSpec)> = Vec::new();
    for _ in 0..60 {
        let model = random_model(&mut rng);
        let spec = model.build(&mut rng);
        layouts.push((model, spec));
    }
    // Interleave the queries: each specification answers from its own model
    for round in 0..3 {
        for (model, spec) in &layouts {
            let layout = DataLayout::new(cx, spec);
            let (i8s, i8a) = model.integer(8);
            let (i64s, i64a) = model.integer(64);
            let (ps, pa) = model.pointer_layout(0);
            let (zs, za) = model.integer(model.isize_width());
            let mut end = 0;
            let mut align = 1;
            for (size, a) in [(i8s, i8a), (i64s, i64a), (ps, pa), (zs, za)] {
                end = round_up(end, a) + size;
                align = align.max(a);
            }
            assert_eq!(
                size_align(&layout, types[0]),
                Ok((round_up(end, align), align)),
                "round {round}: {model:?}"
            );
            let wide = layout.layout_of(types[1]);
            if model.isize_width() > 40 {
                assert!(wide.is_ok(), "{model:?}");
            } else {
                assert_eq!(wide, Err(LayoutError::DiscriminantTooWide(types[1])), "{model:?}");
            }
            assert!(layout.layout_of(types[2]).is_ok());
            assert!(matches!(layout.layout_of(types[3]), Err(LayoutError::NoLayout(_))));
        }
    }
}

/// Struct types written as text with every alignment from 0 to 70 and a few large ones, in each
/// position: exactly the powers of two parse, and what parses is what `validate` accepts.
#[test]
fn the_parser_accepts_exactly_the_alignments_that_are_powers_of_two() {
    let context = context();
    let mut values: Vec<u32> = (0..=70).collect();
    values.extend([255, 256, 1000, 1024, 32767, 32768, 32769, 65535, 65536, 100_000]);
    for value in values {
        let valid = value.is_power_of_two() && value <= 32768;
        for text in [
            format!("struct<align({value}); i32>"),
            format!("struct<packed({value}); i32>"),
            format!("struct<i32 align({value})>"),
            format!("struct<packed(2); i8, i32 align({value})>"),
            format!("struct<i8, struct<align({value}); i8>>"),
            format!("ptr<struct<x: i8 align({value})>>"),
        ] {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                parse::parse_type(&context, "align.runic", text.clone()).into_result().is_ok()
            }));
            assert_eq!(outcome.ok(), Some(valid), "{text}");
        }
    }
    for (text, valid) in [
        ("struct<transparent; i32>", true),
        ("struct<transparent; x: i32>", true),
        ("struct<transparent;>", false),
        ("struct<transparent; i32, i8>", false),
        ("struct<transparent; i32 align(1)>", false),
        ("struct<transparent; struct<transparent; i32 align(4)>>", false),
        ("struct<transparent; struct<i32 align(4)>>", true),
        ("struct t<transparent; i32>", true),
        ("struct<packed; i8, i32>", true),
        ("struct<packed(1); i8, i32 align(2)>", true),
    ] {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            parse::parse_type(&context, "transparent.runic", text).into_result().is_ok()
        }));
        assert_eq!(outcome.ok(), Some(valid), "{text}");
    }
}

/// A recursion group is built by replacing the variables of its bodies, which does not run the
/// check again: every body went through it, and a member keeps its body's representation.
#[test]
fn recursion_group_members_keep_a_checked_representation() {
    let context = context();
    let cx = &*context;
    let eight = core::num::NonZeroU16::new(8).unwrap();
    let bodies = [
        cx.struct_type(
            StructType::named(sym("wrap"), [StructField::new(cx.ptr_type(cx.rec_var(1)))])
                .with_repr(StructRepr::Transparent),
        ),
        cx.struct_type(
            StructType::named(
                sym("node"),
                [StructField::new(cx.i8()).aligned(eight), StructField::new(cx.rec_var(0))],
            )
            .with_repr(StructRepr::Packed(core::num::NonZeroU16::new(2).unwrap())),
        ),
    ];
    let members = cx.rec_group(&bodies).unwrap();
    for member in &members {
        let value = member.downcast_ref::<StructType>().unwrap();
        assert!(value.validate().is_ok(), "{member}");
    }
    let layout = fresh_little32(cx, 0);
    assert_eq!(size_align(&layout, members[0]), Ok((4, 4)));
    // i8 at 0 aligned to 8; the pointer wrapper capped to 2: offset 2, end 6, size 8
    let node = layout.layout_of(members[1]).unwrap();
    assert_eq!((node.size(), node.align().bytes()), (8, 8));
    assert_eq!(*node.fields(), FieldsShape::Offsets([0, 2].into()));
}

// --- offset_of ------------------------------------------------------------------------------------

/// The message of the panic `payload` carries.
fn panic_message(payload: &(dyn core::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or_default()
}

/// A type whose `AggregateType` gives a type for an element its layout has no offset for breaks
/// the contract between its interfaces, and `offset_of` panics rather than answer.
#[test]
fn offset_of_panics_when_a_types_interfaces_disagree() {
    use reentrant::UncountedElementType;

    let context = context();
    let cx = &*context;
    let layout = fresh_little32(cx, 0);
    let ty = cx.intern_type(UncountedElementType { element: cx.i32() });
    assert_eq!(layout.offset_of(ty, &[0]), Ok((0, cx.i32())));
    assert!(matches!(layout.offset_of(ty, &[2]), Err(LayoutError::NoSuchElement { .. })));
    // Element 1 has a type but the layout lists one offset
    let panic = catch_unwind(AssertUnwindSafe(|| layout.offset_of(ty, &[1]))).unwrap_err();
    assert!(panic_message(&*panic).contains("has no offset for its element 1"));
    assert!(layout.is_idle());
}

/// A type whose layout has no offset for an element below its count breaks the contract: where
/// debug assertions are on, `layout_of` panics when it stores the layout; elsewhere the layout is
/// stored, and only `offset_of` could meet the missing offset.
#[test]
fn a_layout_without_an_offset_for_a_counted_element_is_a_debug_panic() {
    use reentrant::MissingOffsetType;

    let context = context();
    let cx = &*context;
    let layout = fresh_little32(cx, 0);
    let ty = cx.intern_type(MissingOffsetType);
    let outcome = catch_unwind(AssertUnwindSafe(|| layout.layout_of(ty)));
    if cfg!(debug_assertions) {
        let panic = outcome.unwrap_err();
        assert!(
            panic_message(&*panic).contains("has no offset for its element 1, though it has 2"),
            "{}",
            panic_message(&*panic)
        );
    } else {
        assert!(outcome.unwrap().is_ok());
    }
    assert!(layout.is_idle());
}

/// A random path into `ty`: mostly elements it has, sometimes an index at or past its element
/// count, an index of a variant without a payload, or a step into a scalar.
fn random_path(rng: &mut Rng, ty: Type) -> Vec<u32> {
    let mut path = Vec::new();
    let mut current = Some(ty);
    for _ in 0..rng.below(5) {
        let count = current
            .and_then(|current| current.as_trait::<dyn AggregateType>().map(|a| a.element_count()));
        let index = match count {
            Some(count) if count > 0 && rng.below(6) != 0 => rng.below(count) as u32,
            Some(count) => u32::try_from(count).unwrap().saturating_add(rng.below(3) as u32),
            None => rng.below(3) as u32,
        };
        path.push(index);
        current = current.and_then(|current| element_at(current, &[index]).ok());
    }
    path
}

/// For 500 random pairs of a type with a layout and a path, valid and invalid, `element_at`
/// selects the type `offset_of` selects, or fails as it does.
#[test]
fn element_at_agrees_with_offset_of_on_random_paths() {
    const PAIRS: usize = 500;
    for seed in seeds() {
        std::eprintln!("seed {seed:#x}");
        let mut rng = Rng(seed);
        let context = context();
        let cx = &*context;
        let (mut valid, mut invalid, mut without_layout) = (0usize, 0usize, 0usize);
        while valid + invalid < PAIRS {
            let model = random_model(&mut rng);
            let layout = DataLayout::new(cx, &model.build(&mut rng));
            let ty = build(cx, &random_t(&mut rng, 3));
            let path = random_path(&mut rng, ty);
            let walked = element_at(ty, &path);
            if layout.layout_of(ty).is_err() {
                without_layout += 1;
                continue;
            }
            let laid_out = layout.offset_of(ty, &path).map(|(_, element)| element);
            assert_eq!(
                walked,
                laid_out,
                "seed {seed:#x} (LAYOUT_TEST_SEED={seed}), pair {}: {ty}, path {path:?}",
                valid + invalid
            );
            if walked.is_ok() {
                valid += 1;
            } else {
                invalid += 1;
            }
        }
        std::eprintln!(
            "element_at: {valid} valid and {invalid} invalid paths; {without_layout} types \
             without a layout passed over"
        );
        assert!(
            valid >= 100 && invalid >= 100,
            "seed {seed:#x}: {valid} valid, {invalid} invalid"
        );
    }
}

// --- A field's own alignment, against clang ------------------------------------------------------

/// A field's `align(N)` is C's `__attribute__((aligned(N)))` on a member, and `packed(1)` is
/// `__attribute__((packed))` on the struct. The sizes, alignments and offsets are what Apple
/// clang 17 (arm64) prints for the C structs named in the comments, whose scalar alignments are
/// the defaults of `little, ptr = 64`.
#[test]
fn field_alignments_are_laid_out_as_clang_lays_out_aligned_members() {
    let context = context();
    let spec = parse_spec(&context, "#builtin.data_layout<little, ptr = 64>").unwrap();
    let layout = DataLayout::new(&context, &spec);
    let f1 = "struct<i8, i8 align(4)>";
    let f3 = "struct<i8, i16 align(8), i8, i64 align(16), i8>";
    let g1 = "struct<packed(1); i8, i32 align(2)>";
    let cases: [(String, u64, u64, &[u64]); 10] = [
        // struct F1 { uint8_t a; uint8_t b AL(4); };
        (f1.into(), 8, 4, &[0, 4]),
        // struct F2 { uint8_t a; uint32_t b AL(1); };
        ("struct<i8, i32 align(1)>".into(), 8, 4, &[0, 4]),
        // struct F3 { uint8_t a; uint16_t b AL(8); uint8_t c; uint64_t d AL(16); uint8_t e; };
        (f3.into(), 32, 16, &[0, 8, 10, 16, 24]),
        // struct F4 { uint32_t a AL(2); uint8_t b AL(2); uint8_t c; };
        ("struct<i32 align(2), i8 align(2), i8>".into(), 8, 4, &[0, 4, 5]),
        // struct PK G1 { uint8_t a; uint32_t b AL(2); };
        (g1.into(), 6, 2, &[0, 2]),
        // struct PK G2 { uint8_t a; uint64_t b AL(4); uint8_t c; uint16_t d; uint32_t e AL(8);
        //                uint8_t f; };
        (
            "struct<packed(1); i8, i64 align(4), i8, i16, i32 align(8), i8>".into(),
            24,
            8,
            &[0, 4, 12, 13, 16, 20],
        ),
        // struct PK G3 { uint8_t a; uint8_t b AL(16); };
        ("struct<packed(1); i8, i8 align(16)>".into(), 32, 16, &[0, 16]),
        // struct H1 { uint8_t a; struct F1 b; uint8_t c; struct G1 d; uint8_t e; };
        (format!("struct<i8, {f1}, i8, {g1}, i8>"), 24, 4, &[0, 4, 12, 14, 20]),
        // struct PK H2 { uint8_t a; struct F3 b; uint8_t c; struct F3 d AL(4); };
        (
            format!("struct<packed(1); i8, {f3}, i8, {f3} align(4)>"),
            68,
            4,
            &[0, 1, 33, 36],
        ),
        // struct AL(32) I1 { uint8_t a; uint32_t b AL(8); };
        ("struct<align(32); i8, i32 align(8)>".into(), 32, 32, &[0, 8]),
    ];
    for (text, size, align, offsets) in cases {
        let ty = parse_ty(&context, &text);
        let laid_out = layout.layout_of(ty).unwrap();
        assert_eq!(
            (laid_out.size(), laid_out.align().bytes(), laid_out.fields()),
            (size, align, &FieldsShape::Offsets(offsets.into())),
            "{text}"
        );
    }
}
