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
        types::{FloatType, IntegerType},
    },
    parse::{self, ParserConfig},
    print::AsmPrinter,
    testing::text::{Rng, describe},
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
        .get_typed_attribute::<DataLayoutAttr>("data_layout")
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

/// A valid specification from `rng`.
fn random_spec(rng: &mut Rng) -> DataLayoutSpec {
    let random_align = |rng: &mut Rng| Align::from_bytes(1 << rng.below(16)).unwrap();
    let random_pointer = |rng: &mut Rng| {
        let size_bits = 8 * (1 + rng.below(8) as u32);
        let align = match PointerSpec::sized(size_bits) {
            Some(sized) if rng.below(2) == 0 => sized.align,
            _ => random_align(rng),
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
    };
    let endianness = rng.pick(&[Endianness::Little, Endianness::Big]);
    let mut spec = DataLayoutSpec::new(endianness, random_pointer(rng)).unwrap();
    for _ in 0..rng.below(4) {
        let addrspace = rng.pick(&[0, 1, 2, 7, u32::MAX]);
        spec = spec.with_pointer(addrspace, random_pointer(rng)).unwrap();
    }
    for _ in 0..rng.below(8) {
        let width = match rng.below(3) {
            0 => rng.pick(&[1, 8, 16, 24, 32, 64, 128, 256]),
            1 => 1 + rng.below(300) as u32,
            _ => 1 + rng.below(u64::from(IntegerType::MAX_WIDTH)) as u32,
        };
        spec = spec.with_integer_align(width, random_align(rng)).unwrap();
    }
    for float in [FloatType::F32, FloatType::F64] {
        if rng.below(3) == 0 {
            spec = spec.with_float_align(float, random_align(rng));
        }
    }
    spec
}

#[test]
fn random_specifications_round_trip() {
    let context = context();
    for seed in seeds() {
        let mut rng = Rng(seed);
        for case in 0..500 {
            let spec = random_spec(&mut rng);
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
