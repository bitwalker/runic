use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::hash::BuildHasher;

use super::*;
use crate::{
    Context, OperationRef,
    dialects::builtin::{
        attributes::DataLayoutAttr,
        types::{
            EnumType, EnumVariant, FloatType, IntegerType, StructField, StructRepr, StructType,
        },
    },
    interner::Symbol,
    parse::{self, ParserConfig},
    print::AsmPrinter,
    testing::text::{Rng, describe},
    types::Type,
};

fn align(bytes: u64) -> Align {
    Align::from_bytes(bytes).unwrap()
}

fn hash_of(spec: &DataLayoutSpec) -> u64 {
    rustc_hash::FxBuildHasher.hash_one(spec)
}

/// `little, ptr = 32`.
fn little32() -> DataLayoutSpec {
    DataLayoutSpec::new(Endianness::Little, PointerSpec::sized(32).unwrap()).unwrap()
}

/// Asserts `a == b` and that they hash alike.
fn assert_same(a: &DataLayoutSpec, b: &DataLayoutSpec) {
    assert_eq!(a, b);
    assert_eq!(hash_of(a), hash_of(b), "{a:?}\n{b:?}");
}

// --- Step 1: the value ---------------------------------------------------------------------------

#[test]
fn an_alignment_is_a_power_of_two_from_1_to_32768() {
    assert_eq!(Align::ONE.bytes(), 1);
    assert_eq!(Align::from_bytes(1), Some(Align::ONE));
    for bytes in [1, 2, 4, 8, 16, 1024, 32768] {
        assert_eq!(Align::from_bytes(bytes).map(Align::bytes), Some(bytes));
    }
    for bytes in [0, 3, 6, 24, 65536, u64::MAX] {
        assert_eq!(Align::from_bytes(bytes), None, "{bytes}");
    }
    assert!(align(2) < align(4));
}

#[test]
fn align_up_rounds_to_the_next_multiple_or_overflows() {
    assert_eq!(align(8).align_up(0), Some(0));
    assert_eq!(align(8).align_up(1), Some(8));
    assert_eq!(align(8).align_up(8), Some(8));
    assert_eq!(align(8).align_up(9), Some(16));
    assert_eq!(Align::ONE.align_up(u64::MAX), Some(u64::MAX));
    assert_eq!(align(8).align_up(u64::MAX - 6), None);
    assert_eq!(align(8).align_up(u64::MAX - 7), Some(u64::MAX - 7));
}

#[test]
fn the_defaults() {
    let spec = little32();
    assert_eq!(spec.endianness(), Endianness::Little);
    assert_eq!(
        spec.pointer(0),
        PointerSpec {
            size_bits: 32,
            align: align(4),
            index_bits: 32
        }
    );
    assert_eq!(spec.isize_width(), 32);
    for (width, bytes) in [(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)] {
        assert_eq!(spec.integer_align(width), align(bytes), "i{width}");
    }
    assert_eq!(spec.float_align(FloatType::F32), align(4));
    assert_eq!(spec.float_align(FloatType::F64), align(8));
}

const WIDTHS: [u32; 11] = [1, 7, 8, 9, 24, 33, 64, 65, 128, 129, 16_777_215];

#[test]
fn an_integer_takes_the_next_wider_listed_width_else_the_widest() {
    let spec = little32();
    let got: Vec<u64> = WIDTHS.iter().map(|w| spec.integer_align(*w).bytes()).collect();
    assert_eq!(got, [1, 1, 1, 2, 4, 8, 8, 16, 16, 16, 16]);

    let spec = little32()
        .with_integer_align(64, align(4))
        .unwrap()
        .with_integer_align(256, align(32))
        .unwrap();
    let got: Vec<u64> = WIDTHS.iter().map(|w| spec.integer_align(*w).bytes()).collect();
    assert_eq!(got, [1, 1, 1, 2, 4, 4, 4, 16, 16, 32, 32]);
}

#[test]
fn an_address_space_without_an_entry_answers_as_address_space_0() {
    let wide = PointerSpec {
        size_bits: 64,
        align: align(8),
        index_bits: 32,
    };
    let spec = little32().with_pointer(1, wide).unwrap();
    assert_eq!(spec.pointer(1), wide);
    assert_eq!(spec.pointer(2), spec.pointer(0));
    assert_eq!(spec.pointer(u32::MAX), spec.pointer(0));
    assert_eq!(spec.isize_width(), 32);

    // Address space 0 changes every address space without its own entry, and none with one
    let spec = spec.with_pointer(0, PointerSpec::sized(16).unwrap()).unwrap();
    assert_eq!(spec.pointer(2).size_bits, 16);
    assert_eq!(spec.pointer(1), wide);
    assert_eq!(spec.isize_width(), 16);
}

#[test]
fn equality_ignores_an_entry_equal_to_its_default() {
    let base = little32();
    assert_same(&base, &base.clone().with_integer_align(32, align(4)).unwrap());
    assert_same(&base, &base.clone().with_float_align(FloatType::F64, align(8)));
    assert_same(&base, &base.clone().with_pointer(3, base.pointer(0)).unwrap());
    // Entries that answer as the table already does make no difference either
    assert_same(&base, &base.clone().with_integer_align(24, align(4)).unwrap());
    assert_same(&base, &base.clone().with_integer_align(200, align(16)).unwrap());
    assert_same(
        &base.clone().with_integer_align(64, align(4)).unwrap(),
        &base
            .clone()
            .with_integer_align(64, align(4))
            .unwrap()
            .with_integer_align(48, align(4))
            .unwrap(),
    );

    // Entries that change an answer do
    assert_ne!(base, base.clone().with_integer_align(32, align(8)).unwrap());
    assert_ne!(base, base.clone().with_integer_align(24, align(2)).unwrap());
    assert_ne!(base, base.clone().with_integer_align(256, align(32)).unwrap());
    assert_ne!(base, base.clone().with_float_align(FloatType::F32, align(8)));
    assert_ne!(base, base.clone().with_pointer(1, PointerSpec::sized(64).unwrap()).unwrap());
    let big = DataLayoutSpec::new(Endianness::Big, PointerSpec::sized(32).unwrap()).unwrap();
    assert_ne!(base, big);
}

#[test]
fn entries_inside_the_last_run_make_no_difference() {
    let context = context();
    let base = little32();
    let wide16 = base
        .clone()
        .with_integer_align(64, align(16))
        .unwrap()
        .with_integer_align(128, align(16))
        .unwrap();
    let all_one = [8, 16, 32, 64, 128]
        .into_iter()
        .fold(base.clone(), |spec, width| spec.with_integer_align(width, Align::ONE).unwrap());
    let pairs = [
        // `i96` lies in the last run, which `i128 = 16` already answers for
        (base.clone(), base.clone().with_integer_align(96, align(16)).unwrap()),
        (wide16.clone(), wide16.clone().with_integer_align(48, align(16)).unwrap()),
        // A table with a single run
        (all_one.clone(), all_one.clone().with_integer_align(4, Align::ONE).unwrap()),
    ];
    for (a, b) in pairs {
        assert_same(&a, &b);
        assert_eq!(print_spec(&context, &a), print_spec(&context, &b));
    }
}

#[test]
fn an_invalid_pointer_or_integer_width_is_rejected() {
    let pointer = |size_bits, index_bits| PointerSpec {
        size_bits,
        align: Align::ONE,
        index_bits,
    };
    for bits in [0, 4, 12, 63, 72, 128] {
        assert_eq!(
            DataLayoutSpec::new(Endianness::Little, pointer(bits, 8)),
            Err(InvalidDataLayout::PointerSize { addrspace: 0, bits })
        );
        assert_eq!(
            little32().with_pointer(2, pointer(bits, 8)),
            Err(InvalidDataLayout::PointerSize { addrspace: 2, bits })
        );
    }
    for (size, index) in [(32, 0), (32, 33), (8, 16)] {
        assert_eq!(
            DataLayoutSpec::new(Endianness::Little, pointer(size, index)),
            Err(InvalidDataLayout::IndexWidth {
                addrspace: 0,
                bits: index,
                size
            })
        );
    }
    assert!(DataLayoutSpec::new(Endianness::Little, pointer(8, 1)).is_ok());
    assert!(DataLayoutSpec::new(Endianness::Little, pointer(64, 64)).is_ok());
    for width in [0, IntegerType::MAX_WIDTH + 1, u32::MAX] {
        assert_eq!(
            little32().with_integer_align(width, Align::ONE),
            Err(InvalidDataLayout::IntegerWidth { width })
        );
    }
    assert!(little32().with_integer_align(IntegerType::MAX_WIDTH, Align::ONE).is_ok());
}

// --- Step 2: the attribute -----------------------------------------------------------------------

fn context() -> Rc<Context> {
    Rc::new(Context::default())
}

/// A module stating `layout` in its attribute dictionary.
fn module_source(layout: &str) -> String {
    format!("builtin.module public @m attributes {{ data_layout = {layout} }} {{\n}};")
}

/// Parses a module stating `layout`; the error is described with its labels.
fn parse_module(context: &Rc<Context>, layout: &str) -> Result<OperationRef, String> {
    let config = ParserConfig::new(context.clone());
    parse::parse_any(config, "layout.hir", module_source(layout))
        .into_result()
        .map_err(|err| describe(&err))
}

/// The specification `layout` (the text of an attribute) gives.
fn parse_spec(context: &Rc<Context>, layout: &str) -> Result<DataLayoutSpec, String> {
    let module = parse_module(context, layout)?;
    let module = module.borrow();
    let attr = module
        .get_typed_attribute::<DataLayoutAttr>(DATA_LAYOUT_ATTR)
        .expect("a data layout attribute");
    Ok(attr.borrow().as_value().clone())
}

fn print_spec(context: &Rc<Context>, spec: &DataLayoutSpec) -> String {
    let attr = context.create_attribute::<DataLayoutAttr, _>(spec.clone());
    let flags = Default::default();
    let mut printer = AsmPrinter::fragment(context.clone(), &flags);
    printer.print_attribute_value(&*attr.borrow());
    printer.finish().to_string()
}

#[test]
fn the_examples_round_trip() {
    let context = context();
    let text = "#builtin.data_layout<little, ptr = 32>";
    let spec = parse_spec(&context, text).unwrap();
    assert_same(&spec, &little32());
    assert_eq!(print_spec(&context, &spec), text);

    let text = "#builtin.data_layout<big, ptr = 64, ptr<1> = 64 align 8 index 32, i64 = 4, i128 = \
                8, f64 = 4>";
    let spec = parse_spec(&context, text).unwrap();
    let expected = DataLayoutSpec::new(Endianness::Big, PointerSpec::sized(64).unwrap())
        .unwrap()
        .with_pointer(
            1,
            PointerSpec {
                size_bits: 64,
                align: align(8),
                index_bits: 32,
            },
        )
        .unwrap()
        .with_integer_align(64, align(4))
        .unwrap()
        .with_integer_align(128, align(8))
        .unwrap()
        .with_float_align(FloatType::F64, align(4));
    assert_same(&spec, &expected);
    // `align 8` is what 64 bits implies, so the canonical text leaves it out
    let printed = print_spec(&context, &spec);
    assert_eq!(
        printed,
        "#builtin.data_layout<big, ptr = 64, ptr<1> = 64 index 32, i64 = 4, i128 = 8, f64 = 4>"
    );
    assert_same(&parse_spec(&context, &printed).unwrap(), &spec);
}

#[test]
fn printing_is_canonical() {
    let context = context();
    let cases = [
        // Entries equal to the defaults, or to what the table answers anyway, are left out
        ("little, ptr = 32, i32 = 4, i24 = 4, f32 = 4, ptr<3> = 32", "little, ptr = 32"),
        // Entries print in a fixed order, whatever order they are written in
        (
            "big, f64 = 4, i64 = 4, ptr<2> = 16, f32 = 2, ptr = 64 align 4, i1 = 2",
            "big, ptr = 64 align 4, ptr<2> = 16, i1 = 2, i64 = 4, f32 = 2, f64 = 4",
        ),
        // An entry keeping a width that would otherwise take a neighbour's alignment
        ("little, ptr = 32, i32 = 8", "little, ptr = 32, i32 = 8"),
        // Entries in the last run, which `i128 = 16` answers for, are left out
        ("little, ptr = 32, i96 = 16, i300 = 16", "little, ptr = 32"),
        // A last run past every default width is stated at its first width
        ("little, ptr = 32, i256 = 32", "little, ptr = 32, i129 = 32"),
        (
            "little, ptr = 32, i300 = 32, i256 = 16",
            "little, ptr = 32, i256 = 16, i257 = 32",
        ),
        ("little, ptr = 24 align 4", "little, ptr = 24 align 4"),
        ("little, ptr = 64 align 8 index 64", "little, ptr = 64"),
        ("little, ptr = 32 align 4 index 16", "little, ptr = 32 index 16"),
    ];
    for (written, printed) in cases {
        let spec = parse_spec(&context, &format!("#builtin.data_layout<{written}>")).unwrap();
        assert_eq!(print_spec(&context, &spec), format!("#builtin.data_layout<{printed}>"));
    }
}

/// The seeds the randomized round trip runs: `LAYOUT_TEST_SEED` if set, else a fixed one.
fn seeds() -> Vec<u64> {
    #[cfg(feature = "std")]
    if let Ok(seed) = std::env::var("LAYOUT_TEST_SEED") {
        return alloc::vec![seed.parse().expect("LAYOUT_TEST_SEED is a u64")];
    }
    alloc::vec![0x1a70_0001]
}

/// A random specification, with the address spaces and integer widths its tables list.
struct Built {
    spec: DataLayoutSpec,
    /// The address spaces with an entry, 0 included.
    spaces: Vec<u32>,
    /// The integer widths with an entry, the defaults included.
    widths: Vec<u32>,
    /// Entries for an address space other than 0 that restate the pointer of address space 0.
    restated_pointers: usize,
    /// Integer entries that answer as the table already did.
    redundant_integers: usize,
}

impl Built {
    fn new(spec: DataLayoutSpec) -> Self {
        Self {
            spec,
            spaces: alloc::vec![0],
            widths: alloc::vec![8, 16, 32, 64, 128],
            restated_pointers: 0,
            redundant_integers: 0,
        }
    }

    fn with_pointer(&mut self, addrspace: u32, pointer: PointerSpec) {
        self.spec = self.spec.clone().with_pointer(addrspace, pointer).unwrap();
        self.spaces.push(addrspace);
    }

    fn with_integer_align(&mut self, width: u32, align: Align) {
        self.spec = self.spec.clone().with_integer_align(width, align).unwrap();
        self.widths.push(width);
    }

    /// Adds entries that change no answer: a pointer restating address space 0 (or an address
    /// space restating itself) and integer entries equal to what the table answers.
    fn add_redundant(&mut self, rng: &mut Rng, widths: &[u32]) {
        let addrspace = rng.pick(&[1, 2, 5, 7, u32::MAX]);
        let pointer = if rng.below(2) == 0 {
            self.restated_pointers += 1;
            self.spec.pointer(0)
        } else {
            self.spec.pointer(addrspace)
        };
        self.with_pointer(addrspace, pointer);
        for _ in 0..1 + rng.below(3) {
            let width = random_width(rng, widths);
            let align = self.spec.integer_align(width);
            self.with_integer_align(width, align);
            self.redundant_integers += 1;
        }
    }
}

/// A random alignment of 1 to 2^(`log2_end` - 1) bytes.
fn random_align(rng: &mut Rng, log2_end: u64) -> Align {
    Align::from_bytes(1 << rng.below(log2_end)).unwrap()
}

/// A random valid pointer.
fn random_pointer(rng: &mut Rng) -> PointerSpec {
    let size_bits = 8 * (1 + rng.below(8) as u32);
    let align = match PointerSpec::sized(size_bits) {
        Some(sized) if rng.below(2) == 0 => sized.align,
        _ => random_align(rng, 16),
    };
    let index_bits = if rng.below(2) == 0 {
        size_bits
    } else {
        1 + rng.below(u64::from(size_bits)) as u32
    };
    PointerSpec {
        size_bits,
        align,
        index_bits,
    }
}

/// A random integer width, often one of `widths`.
fn random_width(rng: &mut Rng, widths: &[u32]) -> u32 {
    match rng.below(4) {
        0 | 1 => rng.pick(widths),
        2 => 1 + rng.below(300) as u32,
        _ => 1 + rng.below(u64::from(IntegerType::MAX_WIDTH)) as u32,
    }
}

const SOME_WIDTHS: [u32; 12] =
    [1, 8, 16, 24, 32, 48, 64, 96, 128, 200, 256, IntegerType::MAX_WIDTH];

/// A random valid specification. Some of its entries may be redundant.
fn random_built(rng: &mut Rng) -> Built {
    let endianness = rng.pick(&[Endianness::Little, Endianness::Big]);
    let mut built = Built::new(DataLayoutSpec::new(endianness, random_pointer(rng)).unwrap());
    for _ in 0..rng.below(4) {
        let addrspace = rng.pick(&[0, 1, 2, 7, u32::MAX]);
        built.with_pointer(addrspace, random_pointer(rng));
    }
    for _ in 0..rng.below(8) {
        let width = random_width(rng, &SOME_WIDTHS);
        built.with_integer_align(width, random_align(rng, 16));
    }
    for float in [FloatType::F32, FloatType::F64] {
        if rng.below(3) == 0 {
            built.spec = built.spec.clone().with_float_align(float, random_align(rng, 16));
        }
    }
    if rng.below(3) == 0 {
        built.add_redundant(rng, &SOME_WIDTHS);
    }
    built
}

#[test]
fn random_specifications_round_trip() {
    let context = context();
    for seed in seeds() {
        let mut rng = Rng(seed);
        let (mut restated_pointers, mut redundant_integers) = (0, 0);
        for case in 0..500 {
            let built = random_built(&mut rng);
            restated_pointers += built.restated_pointers;
            redundant_integers += built.redundant_integers;
            let spec = built.spec;
            let printed = print_spec(&context, &spec);
            let parsed = parse_spec(&context, &printed).unwrap_or_else(|err| {
                panic!("seed {seed:#x} (LAYOUT_TEST_SEED={seed}), case {case}: {printed}: {err}")
            });
            assert!(
                parsed == spec && hash_of(&parsed) == hash_of(&spec),
                "seed {seed:#x} (LAYOUT_TEST_SEED={seed}), case {case}: \
                 {printed}\n{spec:?}\n{parsed:?}"
            );
            assert_eq!(print_spec(&context, &parsed), printed, "seed {seed:#x}, case {case}");
        }
        assert!(
            restated_pointers > 0 && redundant_integers > 0,
            "seed {seed:#x} (LAYOUT_TEST_SEED={seed}): the generator made {restated_pointers} \
             restated pointers and {redundant_integers} redundant integer entries"
        );
    }
}

/// A random specification from few choices, so that independent ones are often answer-equal:
/// alignments of 1 to 4 bytes, few widths and address spaces.
fn random_narrow(rng: &mut Rng) -> Built {
    const WIDTHS: [u32; 6] = [1, 8, 16, 32, 64, 128];
    let pointer = PointerSpec::sized(rng.pick(&[32, 64])).unwrap();
    let mut built = Built::new(DataLayoutSpec::new(Endianness::Little, pointer).unwrap());
    for _ in 0..rng.below(2) {
        let addrspace = rng.pick(&[1, 2]);
        built.with_pointer(addrspace, PointerSpec::sized(rng.pick(&[32, 64])).unwrap());
    }
    for _ in 0..rng.below(6) {
        let width = random_width(rng, &WIDTHS);
        built.with_integer_align(width, random_align(rng, 3));
    }
    if rng.below(4) == 0 {
        built.spec = built.spec.clone().with_float_align(FloatType::F64, align(4));
    }
    built
}

/// True if `a` and `b` answer alike at every probe: every width either lists, one above and
/// one below each, 1 and the widest; every address space either lists and one neither does.
/// Compared through the queries, not through `Eq`.
fn answer_alike(a: &Built, b: &Built) -> bool {
    let mut widths: Vec<u32> = alloc::vec![1, IntegerType::MAX_WIDTH];
    for width in a.widths.iter().chain(&b.widths) {
        widths.extend([width - 1, *width, width + 1]);
    }
    widths.retain(|width| (1..=IntegerType::MAX_WIDTH).contains(width));
    let mut spaces: Vec<u32> = a.spaces.iter().chain(&b.spaces).copied().collect();
    spaces.push((0..).find(|space| !spaces.contains(space)).unwrap());
    let (a, b) = (&a.spec, &b.spec);
    a.endianness() == b.endianness()
        && [FloatType::F32, FloatType::F64]
            .into_iter()
            .all(|float| a.float_align(float) == b.float_align(float))
        && spaces.iter().all(|space| a.pointer(*space) == b.pointer(*space))
        && widths.iter().all(|width| a.integer_align(*width) == b.integer_align(*width))
}

#[test]
fn specifications_are_equal_exactly_when_they_answer_alike() {
    let context = context();
    for seed in seeds() {
        let mut rng = Rng(seed);
        let (mut equal, mut unequal) = (0, 0);
        for case in 0..2000 {
            let (a, b) = match case % 4 {
                // Independent, from a wide range of specifications
                0 => (random_built(&mut rng), random_built(&mut rng)),
                // Independent, from few choices
                1 => (random_narrow(&mut rng), random_narrow(&mut rng)),
                // One specification, and it with redundant entries
                2 => {
                    let a = random_built(&mut rng);
                    let mut b = Built {
                        spec: a.spec.clone(),
                        spaces: a.spaces.clone(),
                        widths: a.widths.clone(),
                        restated_pointers: 0,
                        redundant_integers: 0,
                    };
                    b.add_redundant(&mut rng, &SOME_WIDTHS);
                    (a, b)
                }
                // One specification, and it with redundant entries and one entry more
                _ => {
                    let a = random_narrow(&mut rng);
                    let mut b = Built {
                        spec: a.spec.clone(),
                        spaces: a.spaces.clone(),
                        widths: a.widths.clone(),
                        restated_pointers: 0,
                        redundant_integers: 0,
                    };
                    b.add_redundant(&mut rng, &[1, 8, 16, 32, 64, 128]);
                    let width = random_width(&mut rng, &[1, 8, 16, 32, 64, 128]);
                    b.with_integer_align(width, random_align(&mut rng, 3));
                    (a, b)
                }
            };
            let alike = answer_alike(&a, &b);
            let (a, b) = (&a.spec, &b.spec);
            let context_of = || {
                format!(
                    "seed {seed:#x} (LAYOUT_TEST_SEED={seed}), case {case}:\n{a:?}\n{b:?}\n{}\n{}",
                    print_spec(&context, a),
                    print_spec(&context, b)
                )
            };
            assert_eq!(a == b, alike, "{}", context_of());
            if alike {
                equal += 1;
                assert_eq!(hash_of(a), hash_of(b), "{}", context_of());
                assert_eq!(print_spec(&context, a), print_spec(&context, b), "{}", context_of());
            } else {
                unequal += 1;
            }
        }
        assert!(
            equal >= 200 && unequal >= 200,
            "seed {seed:#x}: {equal} equal, {unequal} unequal"
        );
    }
}

#[test]
fn rejected_forms_are_diagnostics() {
    let context = context();
    let cases = [
        ("#builtin.data_layout", "has no default value"),
        ("#builtin.data_layout<ptr = 32>", "starts with its endianness"),
        ("#builtin.data_layout<little>", "gives the pointer of address space 0"),
        ("#builtin.data_layout<little, i32 = 4>", "gives the pointer of address space 0"),
        ("#builtin.data_layout<little, ptr = 32, ptr = 64>", "given more than once"),
        ("#builtin.data_layout<little, ptr = 32, i8 = 1, i8 = 2>", "given more than once"),
        ("#builtin.data_layout<little, ptr = 32, ptr<0> = 64>", "given more than once"),
        ("#builtin.data_layout<little, ptr = 24>", "has no implied alignment"),
        (
            "#builtin.data_layout<little, ptr = 32, ptr<i8> = 32>",
            "`ptr<i8>` is not a data layout key",
        ),
        (
            "#builtin.data_layout<little, ptr = 32, isize = 4>",
            "`isize` is not a data layout key",
        ),
        (
            "#builtin.data_layout<little, ptr = 32, array<i8; 2> = 4>",
            "is not a data layout key",
        ),
        ("#builtin.data_layout<little, ptr = 32, i0 = 1>", "invalid"),
        ("#builtin.data_layout<little, ptr = 32, i32 = 3>", "expected a power of two"),
        ("#builtin.data_layout<little, ptr = 32, i32 = 65536>", "expected a power of two"),
        ("#builtin.data_layout<little, ptr = 32 align 3>", "expected a power of two"),
        (
            "#builtin.data_layout<little, ptr = 12 align 4>",
            "expected a multiple of 8 from 8 to 64",
        ),
        (
            "#builtin.data_layout<little, ptr = 128>",
            "expected a multiple of 8 from 8 to 64",
        ),
        (
            "#builtin.data_layout<little, ptr = 32, ptr<1> = 72>",
            "expected a multiple of 8",
        ),
        (
            "#builtin.data_layout<little, ptr = 32 index 0>",
            "expected 1 to the pointer size",
        ),
        (
            "#builtin.data_layout<little, ptr = 32 index 64>",
            "expected 1 to the pointer size",
        ),
    ];
    for (text, expected) in cases {
        let err = parse_spec(&context, text).expect_err(text);
        assert!(err.contains(expected), "{text}: expected `{expected}` in: {err}");
    }
}

/// A diagnostic about an entry is labelled at the entry.
#[test]
fn an_entry_diagnostic_is_labelled_at_the_entry() {
    let layout = "#builtin.data_layout<little, ptr = 32, ptr<1> = 64 index 72>";
    let source = module_source(layout);
    let config = ParserConfig::new(context());
    let err = parse::parse_any(config, "layout.hir", source.clone())
        .into_result()
        .unwrap_err();
    let ranges = crate::testing::label_ranges(err.as_diagnostic());
    let start = source.find("ptr<1>").unwrap();
    let end = start + "ptr<1> = 64 index 72".len();
    assert_eq!(ranges.iter().map(|r| (r.start, r.end)).collect::<Vec<_>>(), [(start, end)]);
}

// --- Steps 3 and 4: the interface and `DataLayout` ----------------------------------------------

/// The first operation in the single region of `op`.
fn first_child(op: OperationRef) -> OperationRef {
    let op = op.borrow();
    let region = op.region(0);
    let entry = region.entry();
    entry.front().expect("a non-empty body")
}

/// A component holding a module holding a function, each with the attribute dictionary given.
fn nested(context: &Rc<Context>, component: &str, module: &str) -> [OperationRef; 3] {
    let source = format!(
            "builtin.component private @c {component} {{\n    builtin.module internal @m {module} \
             {{\n        builtin.function public extern(\"C\") @f() {{\n            \
             builtin.ret;\n        }};\n    }};\n}};"
        );
    let config = ParserConfig::new(context.clone());
    let component = parse::parse_any(config, "nested.hir", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let module = first_child(component);
    let function = first_child(module);
    [component, module, function]
}

fn closest_spec(op: OperationRef) -> Option<DataLayoutSpec> {
    DataLayout::closest(&op.borrow()).map(|layout| layout.spec().clone())
}

#[test]
fn the_nearest_stated_specification_wins_whole() {
    let context = context();
    let outer = "attributes { data_layout = #builtin.data_layout<little, ptr = 32, i64 = 4> }";
    let inner = "attributes { data_layout = #builtin.data_layout<big, ptr = 64> }";
    let outer_spec = little32().with_integer_align(64, align(4)).unwrap();
    let inner_spec = DataLayoutSpec::new(Endianness::Big, PointerSpec::sized(64).unwrap()).unwrap();

    // (a) neither states one
    let [component, module, function] = nested(&context, "", "");
    assert_eq!(closest_spec(function), None);
    assert_eq!(closest_spec(module), None);
    assert_eq!(closest_spec(component), None);

    // (b) the component: the module implements the interface but states none, and is passed over
    let [_, module, function] = nested(&context, outer, "");
    assert_eq!(closest_spec(function), Some(outer_spec.clone()));
    assert_eq!(closest_spec(module), Some(outer_spec.clone()));

    // (c) the module
    let [component, _, function] = nested(&context, "", inner);
    assert_eq!(closest_spec(function), Some(inner_spec.clone()));
    assert_eq!(closest_spec(component), None);

    // (d) both, different: the module's, not merged with the component's (`i64` is 8 bytes)
    let [component, module, function] = nested(&context, outer, inner);
    let found = closest_spec(function).unwrap();
    assert_eq!(found, inner_spec);
    assert_eq!(found.integer_align(64), align(8));
    assert_eq!(closest_spec(module), Some(inner_spec));
    assert_eq!(closest_spec(component), Some(outer_spec));
}

#[test]
fn a_parsed_module_answers_its_isize_width() {
    let context = context();
    let module = parse_module(&context, "#builtin.data_layout<little, ptr = 64 index 32>").unwrap();
    let layout = DataLayout::closest(&module.borrow()).expect("the module states a layout");
    assert_eq!(layout.isize_width(), 32);
    assert_eq!(layout.pointer(0).size_bits, 64);
    assert_eq!(layout.pointer(9).size_bits, 64);
    assert_eq!(layout.endianness(), Endianness::Little);
}

#[test]
fn the_verifier_rejects_a_data_layout_that_is_not_one() {
    let context = context();
    let err = parse_module(&context, "#builtin.i32<1>").expect_err("a non-layout `data_layout`");
    assert!(err.contains("is not a '#builtin.data_layout'"), "{err}");
}

#[test]
fn equal_specifications_share_one_cache_per_context() {
    let context = context();
    let a = DataLayout::new(&context, &little32());
    let b = DataLayout::new(&context, &little32().with_integer_align(32, align(4)).unwrap());
    assert!(a.shares_cache_with(&b));
    assert!(a.shares_cache_with(&a.clone()));
    let c = DataLayout::new(&context, &little32().with_integer_align(32, align(8)).unwrap());
    assert!(!a.shares_cache_with(&c));

    let other = Context::default();
    assert!(!a.shares_cache_with(&DataLayout::new(&other, &little32())));

    // The layouts found from two modules stating equal specifications share too
    let first = parse_module(&context, "#builtin.data_layout<little, ptr = 32>").unwrap();
    let second = parse_module(&context, "#builtin.data_layout<little, ptr = 32, i8 = 1>").unwrap();
    let first = DataLayout::closest(&first.borrow()).unwrap();
    let second = DataLayout::closest(&second.borrow()).unwrap();
    assert!(first.shares_cache_with(&second) && first.shares_cache_with(&a));
}

#[test]
fn set_data_layout_spec_replaces_the_attribute() {
    let context = context();
    let [mut component, module, function] = nested(&context, "", "");
    let old = DataLayout::closest(&function.borrow());
    assert!(old.is_none());

    set_data_layout_spec(&mut component.borrow_mut(), little32());
    let installed = component.borrow().get_attribute(DATA_LAYOUT_ATTR).unwrap();
    let held = DataLayout::closest(&function.borrow()).unwrap();
    assert_eq!(held.spec(), &little32());

    let big = DataLayoutSpec::new(Endianness::Big, PointerSpec::sized(16).unwrap()).unwrap();
    set_data_layout_spec(&mut component.borrow_mut(), big.clone());
    let replaced = component.borrow().get_attribute(DATA_LAYOUT_ATTR).unwrap();
    assert!(!crate::AttributeRef::ptr_eq(&installed, &replaced));
    // The old attribute is not edited, and a layout already found keeps its own copy
    let old_value = installed.borrow();
    let old_value = old_value.downcast_ref::<DataLayoutAttr>().unwrap();
    assert_eq!(old_value.as_value(), &little32());
    assert_eq!(held.spec(), &little32());
    assert_eq!(closest_spec(function), Some(big.clone()));
    assert_eq!(closest_spec(module), Some(big));
}

#[test]
fn a_world_states_a_specification_through_text() {
    let context = context();
    let source = "builtin.world {\n    builtin.module public @m {\n    };\n};";
    let config = ParserConfig::new(context.clone());
    let mut world = parse::parse_any(config, "world.hir", source).into_result().unwrap();
    set_data_layout_spec(&mut world.borrow_mut(), little32());
    let printed = format!("{}", world.borrow());
    assert!(
        printed.contains("data_layout = #builtin.data_layout<little, ptr = 32>"),
        "{printed}"
    );

    let config = ParserConfig::new(context.clone());
    let world = parse::parse_any(config, "world2.hir", printed.clone())
        .into_result()
        .unwrap_or_else(|err| panic!("{printed}\n{}", describe(&err)));
    let module = first_child(world);
    assert_eq!(closest_spec(module), Some(little32()));
}

/// The specification `op` states in its attribute, if it states one.
fn stated_spec(op: OperationRef) -> Option<DataLayoutSpec> {
    let op = op.borrow();
    let attr = op.get_typed_attribute::<DataLayoutAttr>(DATA_LAYOUT_ATTR)?;
    Some(attr.borrow().as_value().clone())
}

#[test]
fn a_component_and_a_module_state_a_specification_through_text() {
    let context = context();
    let stated = "attributes { data_layout = #builtin.data_layout<little, ptr = 32> }";
    for on_component in [true, false] {
        let (component, module) = if on_component {
            (stated, "")
        } else {
            ("", stated)
        };
        let [component, module, _] = nested(&context, component, module);
        let on_component = on_component.then(little32);
        let on_module = on_component.is_none().then(little32);
        assert_eq!(stated_spec(component), on_component);
        assert_eq!(stated_spec(module), on_module);

        let printed = format!("{}", component.borrow());
        assert!(printed.contains(stated), "{printed}");
        let config = ParserConfig::new(context.clone());
        let reparsed = parse::parse_any(config, "reparsed.hir", printed.clone())
            .into_result()
            .unwrap_or_else(|err| panic!("{printed}\n{}", describe(&err)));
        assert_eq!(stated_spec(reparsed), on_component, "{printed}");
        assert_eq!(stated_spec(first_child(reparsed)), on_module, "{printed}");
    }
}

// --- Task 4: the layout of a type ----------------------------------------------------------------

/// The layout `text` (the body of a `#builtin.data_layout<..>`) states.
fn data_layout(context: &Rc<Context>, text: &str) -> DataLayout {
    let spec = parse_spec(context, &format!("#builtin.data_layout<{text}>")).unwrap();
    DataLayout::new(context, &spec)
}

/// `(size, align)` of `ty` under `layout`, both in bytes.
fn size_align(layout: &DataLayout, ty: Type) -> Result<(u64, u64), LayoutError> {
    let laid_out = layout.layout_of(ty)?;
    assert_eq!(layout.size_of(ty), Ok(laid_out.size()));
    assert_eq!(layout.align_of(ty), Ok(laid_out.align()));
    assert!(laid_out.size().is_multiple_of(laid_out.align().bytes()), "{ty}: {laid_out:?}");
    Ok((laid_out.size(), laid_out.align().bytes()))
}

const LITTLE_32: &str = "little, ptr = 32";
const LITTLE_64: &str = "little, ptr = 64";
const BIG_64: &str = "big, ptr = 64, ptr<1> = 64 align 8 index 32, i64 = 4";

#[test]
fn scalars_and_pointers_under_three_specifications() {
    let context = context();
    let cx = &*context;
    let fn_ty = cx.function_type([cx.i32()], [cx.i32()]);
    // (type, then (size, align) under LITTLE_32, LITTLE_64, BIG_64)
    let cases: [(Type, [(u64, u64); 3]); 20] = [
        (cx.i1(), [(1, 1); 3]),
        (cx.i8(), [(1, 1); 3]),
        (cx.i16(), [(2, 2); 3]),
        (cx.int_type(24), [(4, 4); 3]),
        (cx.i32(), [(4, 4); 3]),
        // The next listed width of i40 is i64, aligned to 4 under BIG_64
        (cx.int_type(40), [(8, 8), (8, 8), (8, 4)]),
        (cx.i64(), [(8, 8), (8, 8), (8, 4)]),
        (cx.int_type(65), [(16, 16); 3]),
        (cx.i128(), [(16, 16); 3]),
        (cx.int_type(256), [(32, 16); 3]),
        (cx.int_type(IntegerType::MAX_WIDTH), [(2_097_152, 16); 3]),
        // isize is the integer of the index width of address space 0
        (cx.isize(), [(4, 4), (8, 8), (8, 4)]),
        (cx.f32(), [(4, 4); 3]),
        (cx.f64(), [(8, 8); 3]),
        (cx.opaque_ptr_type(), [(4, 4), (8, 8), (8, 8)]),
        (cx.ptr_type(cx.i8()), [(4, 4), (8, 8), (8, 8)]),
        (cx.ptr_type(fn_ty), [(4, 4), (8, 8), (8, 8)]),
        (cx.ptr_type(cx.unknown_type()), [(4, 4), (8, 8), (8, 8)]),
        (cx.ptr_type(cx.never_type()), [(4, 4), (8, 8), (8, 8)]),
        (cx.ptr_type_in(Some(cx.i8()), 1), [(4, 4), (8, 8), (8, 8)]),
    ];
    for (index, text) in [LITTLE_32, LITTLE_64, BIG_64].into_iter().enumerate() {
        let layout = data_layout(&context, text);
        for (ty, expected) in &cases {
            assert_eq!(size_align(&layout, *ty), Ok(expected[index]), "{ty} under {text}");
            assert_eq!(*layout.layout_of(*ty).unwrap().fields(), FieldsShape::Scalar, "{ty}");
        }
    }

    // isize follows the index width, not the pointer size
    let layout = data_layout(&context, "little, ptr = 64 align 8 index 32");
    assert_eq!(size_align(&layout, cx.isize()), Ok((4, 4)));
    assert_eq!(size_align(&layout, cx.opaque_ptr_type()), Ok((8, 8)));

    // A pointer is laid out by its own address space; one without an entry by address space 0
    let layout = data_layout(&context, "little, ptr = 64, ptr<1> = 16, ptr<2> = 24 align 4");
    assert_eq!(size_align(&layout, cx.ptr_type_in(Some(cx.i8()), 1)), Ok((2, 2)));
    assert_eq!(size_align(&layout, cx.opaque_ptr_type()), Ok((8, 8)));
    assert_eq!(size_align(&layout, cx.ptr_type_in(Some(cx.i8()), 3)), Ok((8, 8)));
    // A 3-byte pointer aligned to 4 is 4 bytes, so its size stays a multiple of its alignment
    assert_eq!(size_align(&layout, cx.ptr_type_in(Some(cx.i8()), 2)), Ok((4, 4)));
    // So is a float aligned beyond its size
    let layout = data_layout(&context, "little, ptr = 32, f32 = 16, f64 = 2");
    assert_eq!(size_align(&layout, cx.f32()), Ok((16, 16)));
    assert_eq!(size_align(&layout, cx.f64()), Ok((8, 2)));
}

#[test]
fn arrays() {
    let context = context();
    let cx = &*context;
    let layout = data_layout(&context, LITTLE_32);

    let array = cx.array_type(cx.i32(), 3);
    let laid_out = layout.layout_of(array).unwrap();
    assert_eq!((laid_out.size(), laid_out.align()), (12, align(4)));
    assert_eq!(
        *laid_out.fields(),
        FieldsShape::Array {
            stride: 4,
            count: 3
        }
    );
    assert_eq!(laid_out.fields().offset(2), Some(8));
    assert_eq!(laid_out.fields().offset(3), None);

    let empty = cx.array_type(cx.i8(), 0);
    assert_eq!(size_align(&layout, empty), Ok((0, 1)));
    let empty = cx.array_type(cx.i64(), 0);
    assert_eq!(size_align(&layout, empty), Ok((0, 8)));

    // The stride is the element's size, which is already a multiple of its alignment
    let nested = cx.array_type(cx.array_type(cx.int_type(24), 3), 2);
    let laid_out = layout.layout_of(nested).unwrap();
    assert_eq!((laid_out.size(), laid_out.align()), (24, align(4)));
    assert_eq!(
        *laid_out.fields(),
        FieldsShape::Array {
            stride: 12,
            count: 2
        }
    );

    let huge = cx.array_type(cx.i64(), 1 << 61);
    assert_eq!(layout.layout_of(huge), Err(LayoutError::TooLarge(huge)));
    let largest = cx.array_type(cx.i64(), (1 << 61) - 1);
    assert_eq!(layout.size_of(largest), Ok(u64::MAX - 7));
    // Too large inside an array names the inner array, as an element's error is passed on
    let outer = cx.array_type(huge, 1);
    assert_eq!(layout.layout_of(outer), Err(LayoutError::TooLarge(huge)));
    let outer = cx.array_type(largest, 2);
    assert_eq!(layout.layout_of(outer), Err(LayoutError::TooLarge(outer)));
}

#[test]
fn function_types_unknown_and_never_have_no_layout() {
    let context = context();
    let cx = &*context;
    let layout = data_layout(&context, LITTLE_64);
    let fn_ty = cx.function_type([cx.i32()], [cx.i32()]);
    for ty in [fn_ty, cx.unknown_type(), cx.never_type()] {
        assert_eq!(layout.layout_of(ty), Err(LayoutError::NoLayout(ty)), "{ty}");
        assert_eq!(layout.size_of(ty), Err(LayoutError::NoLayout(ty)), "{ty}");
        assert_eq!(layout.align_of(ty), Err(LayoutError::NoLayout(ty)), "{ty}");
        // An array passes its element's error on unchanged, naming the element
        let array = cx.array_type(ty, 4);
        assert_eq!(layout.layout_of(array), Err(LayoutError::NoLayout(ty)), "{array}");
        let nested = cx.array_type(array, 2);
        assert_eq!(layout.layout_of(nested), Err(LayoutError::NoLayout(ty)), "{nested}");
        // Even an empty one: its alignment is its element's
        let empty = cx.array_type(ty, 0);
        assert_eq!(layout.layout_of(empty), Err(LayoutError::NoLayout(ty)), "{empty}");
        // A pointer to it has the pointer's layout
        assert_eq!(size_align(&layout, cx.ptr_type(ty)), Ok((8, 8)), "ptr<{ty}>");
    }
    assert!(layout.is_idle());
}

#[test]
fn a_layout_error_names_the_type_as_it_prints() {
    let context = context();
    let cx = &*context;
    let fn_ty = cx.function_type([cx.i32()], [cx.i32()]);
    let error = data_layout(&context, LITTLE_32).layout_of(fn_ty).unwrap_err();
    assert_eq!(error.to_string(), "type '(i32) -> i32' has no representation in memory");
    let report = crate::Report::from(error.clone());
    assert_eq!(report.to_string(), error.to_string());

    let huge = cx.array_type(cx.i64(), 1 << 61);
    assert_eq!(
        LayoutError::TooLarge(huge).to_string(),
        "type 'array<i64; 2305843009213693952>' is too large: its size in bytes does not fit in \
         64 bits"
    );
    assert_eq!(
        LayoutError::InfiniteSize(cx.i8()).to_string(),
        "type 'i8' has infinite size: it holds itself by value, or a type that does"
    );
    assert_eq!(
        LayoutError::NoSuchElement {
            ty: cx.i8(),
            index: 3
        }
        .to_string(),
        "type 'i8' has no element 3"
    );
    assert_eq!(
        LayoutError::DiscriminantTooWide(cx.i8()).to_string(),
        "type 'i8' has a discriminant that does not fit the 'isize' of the data layout"
    );
}

mod test_types {
    use super::*;
    use crate::{derive::DialectType, dialects::test::TestDialect};

    /// A dialect type with a layout of its own: 24 bytes aligned to 8.
    #[derive(Debug, Default, PartialEq, Eq, Hash, DialectType)]
    #[ty(dialect = TestDialect, name = "layout_fixed", implements(TypeLayout))]
    pub(super) struct FixedType;
    impl TypeLayout for FixedType {
        fn layout(&self, _ty: Type, _layout: &DataLayout) -> Result<Layout, LayoutError> {
            Ok(Layout::scalar(24, align(8)))
        }
    }

    /// A dialect type without [TypeLayout].
    #[derive(Debug, Default, PartialEq, Eq, Hash, DialectType)]
    #[ty(dialect = TestDialect, name = "layout_none")]
    pub(super) struct NoneType;

    /// A dialect type that holds itself by value: its layout asks for its own.
    #[derive(Debug, Default, PartialEq, Eq, Hash, DialectType)]
    #[ty(dialect = TestDialect, name = "layout_self", implements(TypeLayout))]
    pub(super) struct SelfType;
    impl TypeLayout for SelfType {
        fn layout(&self, ty: Type, layout: &DataLayout) -> Result<Layout, LayoutError> {
            let inner = layout.layout_of(ty)?;
            Ok(Layout::scalar(inner.size() + 8, align(8)))
        }
    }
}

#[test]
fn a_dialect_type_lays_itself_out_through_the_interface() {
    use test_types::*;

    let context = context();
    let cx = &*context;
    let layout = data_layout(&context, LITTLE_32);

    let fixed = cx.intern_type(FixedType);
    assert_eq!(size_align(&layout, fixed), Ok((24, 8)));
    let array = cx.array_type(fixed, 2);
    assert_eq!(size_align(&layout, array), Ok((48, 8)));

    let none = cx.intern_type(NoneType);
    assert_eq!(layout.layout_of(none), Err(LayoutError::NoLayout(none)));
    assert_eq!(layout.layout_of(cx.array_type(none, 1)), Err(LayoutError::NoLayout(none)));

    // The guard stops a type that asks for its own layout, and is unwound afterwards
    let looping = cx.intern_type(SelfType);
    assert_eq!(layout.layout_of(looping), Err(LayoutError::InfiniteSize(looping)));
    assert!(layout.is_idle());
    assert_eq!(layout.layout_of(looping), Err(LayoutError::InfiniteSize(looping)));
    // A type holding it by value is infinite too, and says so of itself
    let array = cx.array_type(looping, 3);
    assert_eq!(layout.layout_of(array), Err(LayoutError::InfiniteSize(array)));
    assert!(layout.is_idle());
    // A pointer to it is finite
    assert_eq!(size_align(&layout, cx.ptr_type(looping)), Ok((4, 4)));
}

#[test]
fn layouts_are_shared_by_equal_specifications() {
    let context = context();
    let cx = &*context;
    let a = DataLayout::new(&context, &little32());
    let b = DataLayout::new(&context, &little32().with_integer_align(32, align(4)).unwrap());
    let c = DataLayout::new(&context, &little32().with_integer_align(32, align(8)).unwrap());
    for ty in [cx.i32(), cx.array_type(cx.i16(), 5), cx.opaque_ptr_type()] {
        let first = a.layout_of(ty).unwrap();
        assert!(Rc::ptr_eq(&first, &b.layout_of(ty).unwrap()), "{ty}");
        assert!(Rc::ptr_eq(&first, &a.layout_of(ty).unwrap()), "{ty}");
        assert!(!Rc::ptr_eq(&first, &c.layout_of(ty).unwrap()), "{ty}");
    }
    assert_eq!(c.size_of(cx.i32()), Ok(8));
    // Errors are kept too
    let fn_ty = cx.function_type([], []);
    assert_eq!(a.layout_of(fn_ty), b.layout_of(fn_ty));
}

/// The alignment of an integer of `width` bits under `entries` (the integer entries of a
/// specification, defaults included), stated here apart from `DataLayoutSpec`: that of the
/// narrowest entry at least `width` bits wide, else that of the widest entry.
fn model_integer_align(entries: &alloc::collections::BTreeMap<u32, u64>, width: u32) -> u64 {
    match entries.range(width..).next() {
        Some((_, bytes)) => *bytes,
        None => *entries.values().next_back().unwrap(),
    }
}

#[test]
fn random_integer_layouts_follow_the_rule() {
    let context = context();
    for seed in seeds() {
        let mut rng = Rng(seed);
        for case in 0..2000 {
            let pointer = PointerSpec::sized(8 * (1 << rng.below(4))).unwrap();
            let index_bits = 1 + rng.below(u64::from(pointer.size_bits)) as u32;
            let pointer = PointerSpec {
                index_bits,
                ..pointer
            };
            let mut spec = DataLayoutSpec::new(Endianness::Little, pointer).unwrap();
            let mut entries: alloc::collections::BTreeMap<u32, u64> =
                [(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)].into_iter().collect();
            for _ in 0..rng.below(6) {
                let width = match rng.below(3) {
                    0 => rng.pick(&[1, 8, 16, 24, 32, 64, 128, 256]),
                    1 => 1 + rng.below(300) as u32,
                    _ => 1 + rng.below(u64::from(IntegerType::MAX_WIDTH)) as u32,
                };
                let bytes = 1 << rng.below(16);
                spec = spec.with_integer_align(width, align(bytes)).unwrap();
                entries.insert(width, bytes);
            }
            let width = match rng.below(3) {
                0 => rng.pick(&entries.keys().copied().collect::<Vec<_>>()),
                1 => 1 + rng.below(300) as u32,
                _ => 1 + rng.below(u64::from(IntegerType::MAX_WIDTH)) as u32,
            };
            let layout = DataLayout::new(&context, &spec);
            for (ty, width) in [(context.int_type(width), width), (context.isize(), index_bits)] {
                let model_align = model_integer_align(&entries, width);
                let bytes = u64::from(width).div_ceil(8);
                let model_size = bytes.div_ceil(model_align) * model_align;
                assert_eq!(
                    size_align(&layout, ty),
                    Ok((model_size, model_align)),
                    "seed {seed:#x} (LAYOUT_TEST_SEED={seed}), case {case}: {ty} (i{width}) under \
                     {}",
                    print_spec(&context, &spec)
                );
            }
        }
    }

    // The widest integer under the largest alignment, which also answers for every integer wider
    // than 128 bits
    let spec = little32().with_integer_align(IntegerType::MAX_WIDTH, align(32768)).unwrap();
    let mut entries: alloc::collections::BTreeMap<u32, u64> =
        [(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)].into_iter().collect();
    entries.insert(IntegerType::MAX_WIDTH, 32768);
    let layout = DataLayout::new(&context, &spec);
    for (width, expected) in [(IntegerType::MAX_WIDTH, (2_097_152, 32768)), (129, (32768, 32768))] {
        let model_align = model_integer_align(&entries, width);
        let model_size = u64::from(width).div_ceil(8).div_ceil(model_align) * model_align;
        assert_eq!((model_size, model_align), expected, "i{width}");
        assert_eq!(size_align(&layout, context.int_type(width)), Ok(expected), "i{width}");
    }
}

// --- Task 5: aggregates --------------------------------------------------------------------------

/// The type `text` writes.
fn parse_ty(context: &Rc<Context>, text: &str) -> Type {
    parse::parse_type(context, "type.runic", text)
        .into_result()
        .unwrap_or_else(|err| panic!("{text}\n{}", describe(&err)))
}

/// `(size, align, offsets)` of `ty` under `layout`, for a type whose elements lie at offsets.
fn shape(layout: &DataLayout, ty: Type) -> Result<(u64, u64, Vec<u64>), LayoutError> {
    let (size, align) = size_align(layout, ty)?;
    let FieldsShape::Offsets(offsets) = layout.layout_of(ty)?.fields().clone() else {
        panic!("{ty} has no offsets");
    };
    Ok((size, align, offsets.into_vec()))
}

fn sym(name: &str) -> Symbol {
    Symbol::intern(name)
}

#[test]
fn structs() {
    let context = context();
    let cx = &*context;
    let layout = data_layout(&context, LITTLE_32);
    let cases: [(&str, u64, u64, &[u64]); 15] = [
        ("struct<i8, i32>", 8, 4, &[0, 4]),
        ("struct<>", 0, 1, &[]),
        ("struct<packed(1); i8, i32>", 5, 1, &[0, 1]),
        ("struct<packed(2); i8, i32>", 6, 2, &[0, 2]),
        ("struct<align(8); i8>", 8, 8, &[0]),
        ("struct<i8, i8 align(4)>", 8, 4, &[0, 4]),
        ("struct<packed(1); i8, i32 align(2)>", 6, 2, &[0, 2]),
        ("struct<transparent; i64>", 8, 8, &[0]),
        ("struct<x: i8, y: i16, z: i8>", 6, 2, &[0, 2, 4]),
        // `packed` only caps: a field aligned below the cap keeps its alignment
        ("struct<packed(4); i8, i16, i64>", 12, 4, &[0, 2, 4]),
        // `align` below the fields' alignment changes nothing
        ("struct<align(2); i8, i32>", 8, 4, &[0, 4]),
        // A field's `align` below its natural alignment changes nothing either
        ("struct<i8, i32 align(1)>", 8, 4, &[0, 4]),
        ("struct<align(4);>", 0, 4, &[]),
        (
            "struct<i8, struct<i16, i8>, array<i32; 2>, struct<align(16); i8>>",
            32,
            16,
            &[0, 2, 8, 16],
        ),
        ("struct<array<struct<i8, i32>; 3>, i8>", 28, 4, &[0, 24]),
    ];
    for (text, size, align, offsets) in cases {
        let ty = parse_ty(&context, text);
        assert_eq!(shape(&layout, ty), Ok((size, align, offsets.to_vec())), "{text}");
    }

    // `transparent` is laid out as its field, under any specification
    for text in [LITTLE_32, BIG_64] {
        let layout = data_layout(&context, text);
        for field in ["i64", "struct<i8, i32>", "array<i16; 3>", "struct<>"] {
            let transparent = parse_ty(&context, &format!("struct<transparent; {field}>"));
            let (size, align) = size_align(&layout, parse_ty(&context, field)).unwrap();
            assert_eq!(shape(&layout, transparent), Ok((size, align, alloc::vec![0])), "{field}");
        }
    }

    // Too large: the struct's own arithmetic names the struct, a field's error is passed on
    let largest = cx.array_type(cx.i64(), (1 << 61) - 1);
    let over = cx.struct_type(StructType::new([largest.into(), cx.i64().into()]));
    assert_eq!(layout.layout_of(over), Err(LayoutError::TooLarge(over)));
    let bytes = cx.array_type(cx.i8(), u64::MAX);
    let padded = cx.struct_type(StructType::new([bytes.into(), cx.i16().into()]));
    assert_eq!(layout.layout_of(padded), Err(LayoutError::TooLarge(padded)));
    let rounded = cx.struct_type(
        StructType::new([bytes.into()]).with_repr(StructRepr::Align(2.try_into().unwrap())),
    );
    assert_eq!(layout.layout_of(rounded), Err(LayoutError::TooLarge(rounded)));
    assert_eq!(
        size_align(&layout, cx.struct_type(StructType::new([bytes.into()]))),
        Ok((u64::MAX, 1))
    );
    let huge = cx.array_type(cx.i64(), 1 << 61);
    let holder = cx.struct_type(StructType::new([cx.i8().into(), huge.into()]));
    assert_eq!(layout.layout_of(holder), Err(LayoutError::TooLarge(huge)));
    assert!(layout.is_idle());
}

#[test]
fn enums() {
    let context = context();
    let layout = data_layout(&context, LITTLE_32);
    let cases: [(&str, u64, u64, &[u64]); 8] = [
        ("enum<i8; a, b>", 1, 1, &[0, 1, 1]),
        ("enum<i8; a: i32, b: i8>", 8, 4, &[0, 4, 4]),
        ("enum<i32; a: i8>", 8, 4, &[0, 4]),
        ("enum<i8; a: struct<>>", 1, 1, &[0, 1]),
        ("enum<i32>", 4, 4, &[0]),
        ("enum<i16; a: struct<packed(1); i8, i32>>", 8, 2, &[0, 2]),
        ("enum<i8; a, b: i64, c: array<i8; 3>>", 16, 8, &[0, 8, 8, 8]),
        ("enum option<i64; none, some: struct<align(16); i8>>", 32, 16, &[0, 16, 16]),
    ];
    for (text, size, align, offsets) in cases {
        let ty = parse_ty(&context, text);
        assert_eq!(shape(&layout, ty), Ok((size, align, offsets.to_vec())), "{text}");
    }

    // An `isize` discriminant's values must fit the specification's `isize`
    let wide = parse_ty(&context, "enum<isize; a = 1099511627776>");
    assert_eq!(layout.layout_of(wide), Err(LayoutError::DiscriminantTooWide(wide)));
    assert_eq!(shape(&data_layout(&context, LITTLE_64), wide), Ok((8, 8, alloc::vec![0, 8])));
    let narrow_index = data_layout(&context, "little, ptr = 64 align 8 index 32");
    assert_eq!(narrow_index.layout_of(wide), Err(LayoutError::DiscriminantTooWide(wide)));
    let fits = parse_ty(&context, "enum<isize; a, b = 4294967295: i8>");
    assert_eq!(shape(&layout, fits), Ok((8, 4, alloc::vec![0, 4, 4])));
    let wider = parse_ty(&context, "enum<isize; a, b = 4294967296: i8>");
    assert_eq!(layout.layout_of(wider), Err(LayoutError::DiscriminantTooWide(wider)));
    // An integer discriminant is never checked again
    let i64_enum = parse_ty(&context, "enum<i64; a = 1099511627776>");
    assert_eq!(shape(&layout, i64_enum), Ok((8, 8, alloc::vec![0, 8])));
}

#[test]
fn elements_of_aggregates() {
    let context = context();
    let cx = &*context;
    let elements = |ty: Type, count: u32| -> Vec<Option<Type>> {
        let aggregate = ty.as_trait::<dyn AggregateType>().expect("an aggregate");
        (0..count).map(|index| aggregate.element_type(index)).collect()
    };

    let array = cx.array_type(cx.i16(), 2);
    assert_eq!(elements(array, 3), [Some(cx.i16()), Some(cx.i16()), None]);
    assert_eq!(elements(cx.array_type(cx.i8(), 0), 1), [None]);
    let ty = parse_ty(&context, "struct<i8, f64>");
    assert_eq!(elements(ty, 3), [Some(cx.i8()), Some(cx.f64()), None]);
    let ty = parse_ty(&context, "enum<i8; a: i32, b, c: f32>");
    assert_eq!(elements(ty, 5), [Some(cx.i8()), Some(cx.i32()), None, Some(cx.f32()), None]);
    assert_eq!(
        elements(array, 1)[0],
        array.as_trait::<dyn AggregateType>().unwrap().element_type(0)
    );
    assert_eq!(
        cx.array_type(cx.i8(), 3)
            .as_trait::<dyn AggregateType>()
            .unwrap()
            .element_type(u32::MAX),
        None
    );

    let fn_ty = cx.function_type([cx.i32()], [cx.i32()]);
    for scalar in [cx.i32(), cx.isize(), cx.f32(), cx.opaque_ptr_type(), fn_ty, cx.never_type()] {
        assert!(scalar.as_trait::<dyn AggregateType>().is_none(), "{scalar}");
    }
}

#[test]
fn a_type_holding_one_without_a_layout_names_it() {
    let context = context();
    let cx = &*context;
    let layout = data_layout(&context, LITTLE_32);
    let fn_ty = cx.function_type([cx.i32()], [cx.i32()]);
    let cases = [
        ("struct<i32, (i32) -> i32>", fn_ty),
        ("struct<i8, struct<array<(i32) -> i32; 2>>>", fn_ty),
        ("struct<transparent; never>", cx.never_type()),
        ("enum<i8; a: i32, b: ?>", cx.unknown_type()),
        ("struct<packed(1); i8, enum<i8; a: (i32) -> i32>>", fn_ty),
    ];
    for (text, missing) in cases {
        let ty = parse_ty(&context, text);
        assert_eq!(layout.layout_of(ty), Err(LayoutError::NoLayout(missing)), "{text}");
    }
    assert!(layout.is_idle());
}

/// `struct name<fields>`, made a group of its own, in which `rec.0` is the struct itself.
fn self_group(cx: &Context, ty: Type) -> Type {
    let members = cx.rec_group(&[ty]).expect("a valid group");
    members[0]
}

#[test]
fn recursion_through_a_pointer_is_finite() {
    let context = context();
    let cx = &*context;
    let list = self_group(
        cx,
        cx.struct_type(StructType::named(
            sym("list"),
            [cx.i32().into(), cx.ptr_type(cx.rec_var(0)).into()],
        )),
    );
    assert_eq!(shape(&data_layout(&context, LITTLE_32), list), Ok((8, 4, alloc::vec![0, 4])));
    assert_eq!(shape(&data_layout(&context, LITTLE_64), list), Ok((16, 8, alloc::vec![0, 8])));

    // Two structs pointing to each other, one also holding a pointer in an enum
    let members = cx
        .rec_group(&[
            cx.struct_type(StructType::named(
                sym("a"),
                [cx.i8().into(), cx.ptr_type(cx.rec_var(1)).into()],
            )),
            cx.struct_type(StructType::named(
                sym("b"),
                [cx.enum_type(
                    EnumType::new(
                        None,
                        cx.i8(),
                        [
                            EnumVariant::new(sym("none")),
                            EnumVariant::new(sym("some")).with_payload(cx.ptr_type(cx.rec_var(0))),
                        ],
                    )
                    .unwrap(),
                )
                .into()],
            )),
        ])
        .unwrap();
    let layout = data_layout(&context, LITTLE_32);
    assert_eq!(shape(&layout, members[0]), Ok((8, 4, alloc::vec![0, 4])));
    assert_eq!(shape(&layout, members[1]), Ok((8, 4, alloc::vec![0])));
    assert!(layout.is_idle());
}

#[test]
fn a_type_holding_itself_by_value_is_infinite() {
    let context = context();
    let cx = &*context;
    let layout = data_layout(&context, LITTLE_32);
    let itself = cx.rec_var(0);
    let bodies = [
        cx.struct_type(StructType::named(sym("s"), [cx.i32().into(), itself.into()])),
        cx.struct_type(StructType::named(sym("t"), [cx.array_type(itself, 0).into()])),
        cx.struct_type(
            StructType::named(sym("u"), [itself.into()]).with_repr(StructRepr::Transparent),
        ),
        cx.enum_type(
            EnumType::new(
                Some(sym("e")),
                cx.i8(),
                [EnumVariant::new(sym("a")), EnumVariant::new(sym("b")).with_payload(itself)],
            )
            .unwrap(),
        ),
    ];
    for body in bodies {
        let ty = self_group(cx, body);
        assert_eq!(layout.layout_of(ty), Err(LayoutError::InfiniteSize(ty)), "{ty}");
        assert!(layout.is_idle());
        assert_eq!(layout.layout_of(ty), Err(LayoutError::InfiniteSize(ty)), "{ty}");
        // A type holding it by value is infinite too; a pointer to it is not
        let array = cx.array_type(ty, 2);
        assert_eq!(layout.layout_of(array), Err(LayoutError::InfiniteSize(array)), "{ty}");
        assert_eq!(size_align(&layout, cx.ptr_type(ty)), Ok((4, 4)), "{ty}");
    }
}

/// Two structs that hold each other through an array, and a third that holds the second by
/// value.
fn mutually_holding(cx: &Context) -> [Type; 3] {
    let members = cx
        .rec_group(&[
            cx.struct_type(StructType::named(
                sym("a"),
                [cx.i8().into(), cx.array_type(cx.rec_var(1), 2).into()],
            )),
            cx.struct_type(StructType::named(
                sym("b"),
                [cx.i32().into(), cx.array_type(cx.rec_var(0), 1).into()],
            )),
        ])
        .unwrap();
    let c = cx
        .struct_type(StructType::named(sym("c"), [cx.i64().into(), StructField::new(members[1])]));
    [members[0], members[1], c]
}

#[test]
fn by_value_recursion_is_infinite_whatever_is_asked_first() {
    let orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for order in orders {
        let context = context();
        let types = mutually_holding(&context);
        let layout = data_layout(&context, LITTLE_32);
        let mut answers: [Option<Result<Rc<Layout>, LayoutError>>; 3] = [None, None, None];
        for index in order {
            answers[index] = Some(layout.layout_of(types[index]));
            assert!(layout.is_idle(), "{order:?}");
        }
        for (ty, answer) in types.iter().zip(answers) {
            assert_eq!(answer, Some(Err(LayoutError::InfiniteSize(*ty))), "{ty}, asked {order:?}");
            // Asked again, from the cache
            assert_eq!(layout.layout_of(*ty), Err(LayoutError::InfiniteSize(*ty)), "{ty}");
        }
    }
}

/// A random type with a layout, for the struct generator: an integer, a float, a pointer,
/// `isize`, or, while `depth` is above 0, an array or a struct. Every struct made is pushed onto
/// `structs`.
fn random_field_type(rng: &mut Rng, cx: &Context, depth: u32, structs: &mut Vec<Type>) -> Type {
    let kinds = if depth == 0 { 4 } else { 7 };
    match rng.below(kinds) {
        0 => cx.int_type(match rng.below(2) {
            0 => rng.pick(&SOME_WIDTHS),
            _ => 1 + rng.below(300) as u32,
        }),
        1 => rng.pick(&[cx.f32(), cx.f64()]),
        2 => cx.ptr_type_in(Some(cx.i8()), rng.pick(&[0, 1, 2])),
        3 => cx.isize(),
        4 => cx.array_type(random_field_type(rng, cx, depth - 1, structs), rng.below(4)),
        _ => random_struct(rng, cx, depth - 1, structs),
    }
}

/// A power of two from 1 to 2^(`log2_end` - 1), as a struct states an alignment.
fn random_stated_align(rng: &mut Rng, log2_end: u64) -> core::num::NonZeroU16 {
    core::num::NonZeroU16::new(1 << rng.below(log2_end)).unwrap()
}

/// A random struct of any representation, its fields from [random_field_type].
fn random_struct(rng: &mut Rng, cx: &Context, depth: u32, structs: &mut Vec<Type>) -> Type {
    let repr = match rng.below(8) {
        0..=2 => StructRepr::Default,
        3 | 4 => StructRepr::Align(random_stated_align(rng, 16)),
        5 | 6 => StructRepr::Packed(random_stated_align(rng, 6)),
        _ => StructRepr::Transparent,
    };
    let count = if repr == StructRepr::Transparent {
        1
    } else {
        rng.below(6)
    };
    let aligned_fields = repr != StructRepr::Transparent && rng.below(3) == 0;
    let fields: Vec<StructField> = (0..count)
        .map(|_| {
            let field = StructField::new(random_field_type(rng, cx, depth, structs));
            if aligned_fields && rng.below(2) == 0 {
                field.aligned(random_stated_align(rng, 8))
            } else {
                field
            }
        })
        .collect();
    let ty = cx.struct_type(StructType::new(fields).with_repr(repr));
    structs.push(ty);
    ty
}

/// Which rule [check_struct] checked a struct against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Checked {
    Oracle,
    Invariants,
    Transparent,
}

/// Checks the layout of the struct `ty` under `layout`: against `core::alloc::Layout` for the
/// default and `align(N)` representations without field alignments, else against the invariants
/// of the rule. `case` says where the struct came from, for a failure.
fn check_struct(layout: &DataLayout, ty: Type, case: &str) -> Checked {
    let value = ty.as_struct().unwrap();
    let (size, align, offsets) =
        shape(layout, ty).unwrap_or_else(|err| panic!("{case}: {ty}: {err}"));
    let fields: Vec<Rc<Layout>> =
        value.fields().iter().map(|field| layout.layout_of(field.ty).unwrap()).collect();
    assert_eq!(offsets.len(), fields.len(), "{case}: {ty}");

    if value.repr() == StructRepr::Transparent {
        assert_eq!(
            (size, align, offsets),
            (fields[0].size(), fields[0].align().bytes(), alloc::vec![0]),
            "{case}: {ty}"
        );
        return Checked::Transparent;
    }

    let has_field_align = value.fields().iter().any(|field| field.align.is_some());
    if !has_field_align && let StructRepr::Default | StructRepr::Align(_) = value.repr() {
        let mut oracle = core::alloc::Layout::from_size_align(0, 1).unwrap();
        let mut oracle_offsets = Vec::new();
        for field in &fields {
            let field = core::alloc::Layout::from_size_align(
                usize::try_from(field.size()).unwrap(),
                usize::try_from(field.align().bytes()).unwrap(),
            )
            .unwrap();
            let (next, offset) = oracle.extend(field).unwrap();
            oracle = next;
            oracle_offsets.push(offset as u64);
        }
        if let StructRepr::Align(n) = value.repr() {
            oracle = oracle.align_to(usize::from(n.get())).unwrap();
        }
        let oracle = oracle.pad_to_align();
        assert_eq!(
            (size, align, offsets),
            (oracle.size() as u64, oracle.align() as u64, oracle_offsets),
            "{case}: {ty}"
        );
        return Checked::Oracle;
    }

    // The effective alignment of a field: its own, capped by `packed(P)`, raised by its `align(F)`
    let cap = match value.repr() {
        StructRepr::Packed(cap) => u64::from(cap.get()),
        _ => u64::MAX,
    };
    let mut end = 0;
    let mut struct_align = 1;
    for (index, (field, laid_out)) in value.fields().iter().zip(&fields).enumerate() {
        let stated = field.align.map_or(1, |align| u64::from(align.get()));
        let effective = laid_out.align().bytes().min(cap).max(stated);
        let offset = offsets[index];
        assert!(offset >= end, "{case}: {ty}: field {index} at {offset} overlaps the one before");
        assert!(
            offset.is_multiple_of(effective),
            "{case}: {ty}: field {index} at {offset}, aligned to {effective}"
        );
        assert!(
            offset < end + effective,
            "{case}: {ty}: field {index} at {offset}, not the first multiple of {effective} from \
             {end}"
        );
        end = offset + laid_out.size();
        struct_align = struct_align.max(effective);
    }
    if let StructRepr::Align(n) = value.repr() {
        struct_align = struct_align.max(u64::from(n.get()));
    }
    assert_eq!(align, struct_align, "{case}: {ty}");
    assert_eq!(size, end.div_ceil(struct_align) * struct_align, "{case}: {ty}");
    Checked::Invariants
}

#[test]
fn random_structs_follow_the_rule() {
    for seed in seeds() {
        let context = context();
        let mut rng = Rng(seed);
        let mut counts = [0usize; 3];
        for case in 0..2000 {
            let spec = random_built(&mut rng).spec;
            let layout = DataLayout::new(&context, &spec);
            let mut structs = Vec::new();
            random_struct(&mut rng, &context, 2, &mut structs);
            let case = format!(
                "seed {seed:#x} (LAYOUT_TEST_SEED={seed}), case {case}, under {}",
                print_spec(&context, &spec)
            );
            for ty in structs {
                let checked = check_struct(&layout, ty, &case);
                counts[checked as usize] += 1;
            }
        }
        // Each rule is checked often
        assert!(counts.iter().all(|count| *count >= 200), "seed {seed:#x}: {counts:?}");
    }
}
