//! [evaluate_initializer]: the bytes and relocations of a global variable's initializer, under
//! 32- and 64-bit, little- and big-endian layouts, and each of its errors.

use alloc::{format, rc::Rc, string::String, vec, vec::Vec};

use runic_ir::{
    Context, DataLayout, Op, OperationRef, Report, SymbolTable,
    dialects::builtin::{GlobalVariable, GlobalVariableRef, Module, ModuleRef},
    testing::text::{describe, parse},
};

use crate::{Initializer, Relocation, Target, evaluate_initializer};

/// The layout most tests use.
const LITTLE32: &str = "little, ptr = 32, fn = 32";

/// A module parsed and verified.
struct Parsed {
    _context: Rc<Context>,
    module: ModuleRef,
}

impl Parsed {
    /// The module `@m` stating `#builtin.data_layout<layout>` that holds `items`.
    #[track_caller]
    fn new(layout: &str, items: &str) -> Parsed {
        let text = format!(
            "builtin.module public @m attributes {{ data_layout = #builtin.data_layout<{layout}> \
             }} {{\n{items}\n}};"
        );
        let (context, module) = parse::<Module>(&text).unwrap_or_else(|err| panic!("{err}"));
        Parsed {
            _context: context,
            module,
        }
    }

    /// The operation of the symbol `name`.
    #[track_caller]
    fn symbol(&self, name: &str) -> OperationRef {
        self.module
            .borrow()
            .symbol_manager()
            .lookup_op(runic_ir::SymbolName::intern(name))
            .unwrap_or_else(|| panic!("no symbol @{name}"))
    }

    /// The global variable `name`.
    #[track_caller]
    fn global(&self, name: &str) -> GlobalVariableRef {
        self.symbol(name)
            .borrow()
            .downcast_ref::<GlobalVariable>()
            .expect("a global variable")
            .as_global_var_ref()
    }

    /// [evaluate_initializer] of the global `name` under the module's layout.
    fn evaluate(&self, name: &str) -> Result<Initializer, Report> {
        let global = self.global(name);
        let global = global.borrow();
        let layout = DataLayout::closest(global.as_operation()).expect("a layout");
        evaluate_initializer(&global, &layout)
    }

    /// The initializer of the global `name`, which evaluates.
    #[track_caller]
    fn initializer(&self, name: &str) -> Initializer {
        self.evaluate(name).unwrap_or_else(|report| panic!("{}", describe(&report)))
    }

    /// The error of the global `name`, with its labels.
    #[track_caller]
    fn error(&self, name: &str) -> String {
        match self.evaluate(name) {
            Ok(init) => panic!("@{name} evaluated: {init:?}"),
            Err(report) => describe(&report),
        }
    }

    /// A relocation of `size` bytes at `offset` to the address of the global `name` plus `addend`.
    fn data(&self, offset: u64, size: u8, name: &str, addend: i64) -> Relocation {
        Relocation {
            offset,
            size,
            target: Target::Data(self.global(name)),
            addend,
        }
    }

    /// A relocation of `size` bytes at `offset` to the function `name`.
    fn function(&self, offset: u64, size: u8, name: &str) -> Relocation {
        Relocation {
            offset,
            size,
            target: Target::Function(self.symbol(name)),
            addend: 0,
        }
    }
}

/// A global `@name : ty` whose initializer is `body` (operations ending in a value `%v`) and
/// `builtin.yield %v;`.
fn global(name: &str, ty: &str, body: &str) -> String {
    format!("builtin.global_variable public @{name} : {ty} {{\n{body}\nbuiltin.yield %v;\n}};")
}

/// A global `@g : ty` whose initializer is the constant `value : ty`.
fn constant(ty: &str, value: &str) -> String {
    global("g", ty, &format!("%v = arith.constant {value} : {ty};"))
}

/// The initializer of `@g`, a global of type `ty` initialized by `body`, under `layout`.
#[track_caller]
fn init_of(layout: &str, ty: &str, body: &str) -> Initializer {
    Parsed::new(layout, &global("g", ty, body)).initializer("g")
}

/// The bytes of `@g`, a global of type `ty` initialized by `body`, under `layout`, which has no
/// relocation, with its size and alignment checked.
#[track_caller]
fn bytes_of(layout: &str, ty: &str, body: &str, size: u64, align: u64) -> Vec<u8> {
    let init = init_of(layout, ty, body);
    assert_eq!((init.size, init.align), (size, align), "{init:?}");
    assert!(init.relocations.is_empty(), "{init:?}");
    assert!(init.bytes.is_empty() || init.bytes.len() as u64 == size, "{init:?}");
    init.bytes
}

/// Asserts that `haystack` holds each of `needles`.
#[track_caller]
fn assert_mentions(haystack: &str, needles: &[&str]) {
    for needle in needles {
        assert!(haystack.contains(needle), "no `{needle}` in:\n{haystack}");
    }
}

// ---- Scalars ------------------------------------------------------------------------------------

#[test]
fn each_integer_is_its_size_in_bytes_little_endian_first_byte_lowest() {
    let cases: [(&str, &str, &[u8], u64); 8] = [
        ("i1", "1", &[1], 1),
        ("i8", "-3", &[0xfd], 1),
        ("i16", "-300", &[0xd4, 0xfe], 2),
        ("i32", "123456", &[0x40, 0xe2, 0x01, 0x00], 4),
        ("i64", "-5000000000", &[0x00, 0x0e, 0xfa, 0xd5, 0xfe, 0xff, 0xff, 0xff], 8),
        (
            "i128",
            "-2",
            &[
                0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff,
            ],
            16,
        ),
        ("isize", "-2", &[0xfe, 0xff, 0xff, 0xff], 4),
        ("i24", "65536", &[0x00, 0x00, 0x01, 0x00], 4),
    ];
    for (ty, value, expected, align) in cases {
        let size = expected.len() as u64;
        let bytes = Parsed::new(LITTLE32, &constant(ty, value)).initializer("g");
        assert_eq!((bytes.size, bytes.align), (size, align), "{ty}");
        assert_eq!(bytes.bytes, expected, "{ty} {value}");
        assert!(bytes.relocations.is_empty());
    }
}

#[test]
fn each_integer_is_its_bytes_most_significant_first_under_a_big_endian_layout() {
    let big = "big, ptr = 32, fn = 32";
    let cases: [(&str, &str, &[u8]); 7] = [
        ("i1", "1", &[1]),
        ("i8", "-3", &[0xfd]),
        ("i16", "-300", &[0xfe, 0xd4]),
        ("i32", "123456", &[0x00, 0x01, 0xe2, 0x40]),
        ("i64", "-5000000000", &[0xff, 0xff, 0xff, 0xfe, 0xd5, 0xfa, 0x0e, 0x00]),
        ("i128", "1", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ("isize", "-2", &[0xff, 0xff, 0xff, 0xfe]),
    ];
    for (ty, value, expected) in cases {
        let init = Parsed::new(big, &constant(ty, value)).initializer("g");
        assert_eq!(init.bytes, expected, "{ty} {value}");
    }
}

#[test]
fn an_isize_is_eight_bytes_under_a_64_bit_layout() {
    let init = Parsed::new("little, ptr = 64", &constant("isize", "-2")).initializer("g");
    assert_eq!((init.size, init.align), (8, 8));
    assert_eq!(init.bytes, [0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
}

#[test]
fn each_float_is_its_bits_in_the_layout_s_byte_order() {
    let f32 = global("g", "f32", "%v = arith.constantf 1.5 : f32;");
    let f64 = global("g", "f64", "%v = arith.constantf -2.25 : f64;");
    let little = |text: &str| Parsed::new(LITTLE32, text).initializer("g");
    let big = |text: &str| Parsed::new("big, ptr = 32", text).initializer("g");
    assert_eq!(little(&f32).bytes, 1.5f32.to_bits().to_le_bytes());
    assert_eq!(little(&f64).bytes, (-2.25f64).to_bits().to_le_bytes());
    assert_eq!(big(&f32).bytes, 1.5f32.to_bits().to_be_bytes());
    assert_eq!(big(&f64).bytes, (-2.25f64).to_bits().to_be_bytes());
    assert_eq!((little(&f64).size, little(&f64).align), (8, 8));
}

#[test]
fn bytes_are_written_as_they_are_in_either_byte_order() {
    // The first byte of `builtin.bytes` is the last pair of digits
    let body = "%v = builtin.bytes \"04030201\" : array<i8; 4>;";
    for layout in [LITTLE32, "big, ptr = 32"] {
        assert_eq!(bytes_of(layout, "array<i8; 4>", body, 4, 1), [1, 2, 3, 4]);
    }
}

#[test]
fn an_unfolded_addition_and_multiplication_of_constants_are_folded() {
    let sum = "%a = arith.constant 40 : i32;\n%b = arith.constant 2 : i32;\n%v = arith.add %a, %b;";
    assert_eq!(bytes_of(LITTLE32, "i32", sum, 4, 4), [42, 0, 0, 0]);
    let product = "%a = arith.constant 6 : i64;\n%b = arith.constant 7 : i64;\n%c = arith.mul %a, \
                   %b;\n%v = arith.add %c, %c;";
    assert_eq!(bytes_of(LITTLE32, "i64", product, 8, 8), [84, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn a_constant_zero_has_no_bytes() {
    let init = Parsed::new(LITTLE32, &constant("i64", "0")).initializer("g");
    assert!(init.is_zero(), "{init:?}");
    assert_eq!((init.size, init.align), (8, 8));
}

// ---- Aggregates ---------------------------------------------------------------------------------

#[test]
fn a_long_chain_of_folds_is_evaluated_without_recursion() {
    let mut body = String::from("%one = arith.constant 1 : i32;\n%x0 = arith.constant 0 : i32;\n");
    let count = 20_000;
    for i in 1..=count {
        body.push_str(&format!("%x{i} = arith.add %x{}, %one;\n", i - 1));
    }
    body.push_str(&format!("%v = arith.add %x{count}, %x0;"));
    let bytes = bytes_of(LITTLE32, "i32", &body, 4, 4);
    assert_eq!(bytes, 20_000u32.to_le_bytes());
}

#[test]
fn a_struct_places_its_fields_at_their_offsets_with_zero_padding_between_and_after() {
    let body = "%a = arith.constant 1 : i8;\n%b = arith.constant 16909060 : i32;\n%c = \
                arith.constant 1286 : i16;\n%v = builtin.aggregate %a, %b, %c : struct<i8, i32, \
                i16>;";
    assert_eq!(
        bytes_of(LITTLE32, "struct<i8, i32, i16>", body, 12, 4),
        [1, 0, 0, 0, 4, 3, 2, 1, 6, 5, 0, 0]
    );
    let big = bytes_of("big, ptr = 32", "struct<i8, i32, i16>", body, 12, 4);
    assert_eq!(big, [1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0]);
}

#[test]
fn nested_structs_place_the_inner_struct_at_its_offset() {
    let body = "%a = arith.constant 258 : i16;\n%b = arith.constant 3 : i8;\n%c = arith.constant \
                4 : i32;\n%inner = builtin.aggregate %b, %c : struct<i8, i32>;\n%v = \
                builtin.aggregate %a, %inner : struct<i16, struct<i8, i32>>;";
    assert_eq!(
        bytes_of(LITTLE32, "struct<i16, struct<i8, i32>>", body, 12, 4),
        [2, 1, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
    );
}

#[test]
fn an_array_of_structs_places_each_at_its_stride() {
    let ty = "array<struct<i8, i16>; 2>";
    let body = format!(
        "%a = arith.constant 1 : i8;\n%b = arith.constant 2 : i16;\n%c = arith.constant 3 : \
         i8;\n%d = arith.constant 4 : i16;\n%x = builtin.aggregate %a, %b : struct<i8, i16>;\n%y \
         = builtin.aggregate %c, %d : struct<i8, i16>;\n%v = builtin.aggregate %x, %y : {ty};"
    );
    assert_eq!(bytes_of(LITTLE32, ty, &body, 8, 2), [1, 0, 2, 0, 3, 0, 4, 0]);
}

#[test]
fn an_empty_struct_is_zero_bytes() {
    let init = init_of(LITTLE32, "struct<>", "%v = builtin.aggregate : struct<>;");
    assert_eq!((init.size, init.align), (0, 1));
    assert!(init.is_zero());
    assert!(init.relocations.is_empty());
}

#[test]
fn a_struct_inside_an_array_inside_a_struct_is_placed_at_the_sum_of_the_offsets() {
    let ty = "struct<i8, array<struct<i8, i32>; 2>>";
    let body = format!(
        "%a = arith.constant 9 : i8;\n%b = arith.constant 1 : i8;\n%c = arith.constant 2 : \
         i32;\n%d = arith.constant 3 : i8;\n%e = arith.constant 4 : i32;\n%x = builtin.aggregate \
         %b, %c : struct<i8, i32>;\n%y = builtin.aggregate %d, %e : struct<i8, i32>;\n%arr = \
         builtin.aggregate %x, %y : array<struct<i8, i32>; 2>;\n%v = builtin.aggregate %a, %arr : \
         {ty};"
    );
    assert_eq!(
        bytes_of(LITTLE32, ty, &body, 20, 4),
        [9, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
    );
}

#[test]
fn an_element_used_by_two_aggregates_is_placed_in_each() {
    let ty = "array<struct<i32, array<i8; 2>>; 2>";
    let body = format!(
        "%one = arith.constant 1 : i32;\n%two = arith.constant 2 : i32;\n%bytes = builtin.bytes \
         \"0201\" : array<i8; 2>;\n%first = builtin.aggregate %one, %bytes : struct<i32, \
         array<i8; 2>>;\n%second = builtin.aggregate %two, %bytes : struct<i32, array<i8; \
         2>>;\n%v = builtin.aggregate %first, %second : {ty};"
    );
    assert_eq!(
        bytes_of(LITTLE32, ty, &body, 16, 4),
        [1, 0, 0, 0, 1, 2, 0, 0, 2, 0, 0, 0, 1, 2, 0, 0]
    );
    // An aggregate used twice by another is placed twice
    let ty = "struct<struct<i8, i8>, struct<i8, i8>>";
    let body = format!(
        "%a = arith.constant 5 : i8;\n%pair = builtin.aggregate %a, %a : struct<i8, i8>;\n%v = \
         builtin.aggregate %pair, %pair : {ty};"
    );
    assert_eq!(bytes_of(LITTLE32, ty, &body, 4, 1), [5, 5, 5, 5]);
}

// ---- Zero ---------------------------------------------------------------------------------------

#[test]
fn a_zero_of_a_scalar_or_a_struct_is_its_size_with_no_bytes() {
    for (ty, size, align) in [("i32", 4, 4), ("struct<i8, i64>", 16, 8), ("ptr<i8>", 4, 4)] {
        let init = init_of(LITTLE32, ty, &format!("%v = builtin.zero : {ty};"));
        assert_eq!((init.size, init.align), (size, align), "{ty}");
        assert!(init.is_zero(), "{ty}: {init:?}");
        assert!(init.relocations.is_empty());
    }
}

#[test]
fn a_zero_of_a_large_array_allocates_nothing() {
    let ty = "array<i64; 1048576>";
    let init = init_of(LITTLE32, ty, &format!("%v = builtin.zero : {ty};"));
    assert_eq!(init.size, 8 * 1048576);
    assert!(init.is_zero());
    assert_eq!(init.bytes.capacity(), 0);
}

#[test]
fn an_aggregate_all_of_whose_elements_are_zero_has_no_bytes() {
    let body = "%a = builtin.zero : i32;\n%b = arith.constant 0 : i16;\n%c = builtin.zero : \
                array<i8; 3>;\n%v = builtin.aggregate %a, %b, %c : struct<i32, i16, array<i8; 3>>;";
    let init = init_of(LITTLE32, "struct<i32, i16, array<i8; 3>>", body);
    assert_eq!(init.size, 12);
    assert!(init.is_zero(), "{init:?}");

    // An array of 1024 zeroed arrays of 1024 `i64`s, 8 MiB, allocates nothing
    let ty = "array<array<i64; 1024>; 1024>";
    let operands = vec!["%inner"; 1024].join(", ");
    let body = format!(
        "%inner = builtin.zero : array<i64; 1024>;\n%v = builtin.aggregate {operands} : {ty};"
    );
    let init = init_of(LITTLE32, ty, &body);
    assert_eq!(init.size, 8 * 1024 * 1024);
    assert!(init.is_zero());
    assert_eq!(init.bytes.capacity(), 0);
}

#[test]
fn an_aggregate_with_one_non_zero_byte_at_its_end_has_all_its_bytes() {
    let ty = "struct<array<i8; 1000>, i8>";
    let body = format!(
        "%z = builtin.zero : array<i8; 1000>;\n%one = arith.constant 1 : i8;\n%v = \
         builtin.aggregate %z, %one : {ty};"
    );
    let bytes = bytes_of(LITTLE32, ty, &body, 1001, 1);
    assert_eq!(bytes.len(), 1001);
    assert!(bytes[..1000].iter().all(|byte| *byte == 0));
    assert_eq!(bytes[1000], 1);
}

// ---- Addresses ----------------------------------------------------------------------------------

/// The globals the address tests point into: `@target : i32`, `@bytes : array<i8; 8>` and `@table
/// : struct<i8, array<struct<i16, i32>; 4>>`.
const TARGETS: &str = "\
builtin.global_variable public @target : i32 { %v = arith.constant 7 : i32; builtin.yield %v; };
builtin.global_variable public @bytes : array<i8; 8> { %v = builtin.zero : array<i8; 8>; \
                       builtin.yield %v; };
builtin.global_variable public @table : struct<i8, array<struct<i16, i32>; 4>> { %v = builtin.zero \
                       : struct<i8, array<struct<i16, i32>; 4>>; builtin.yield %v; };
builtin.function public extern(\"C\") @callee(%x: i32) -> i32 { builtin.ret %x : (i32); };
builtin.function_alias public @alias -> @callee;
builtin.function public extern(\"C\") @declared(i32) -> i32;";

/// The module of [TARGETS] and `@g : ty` initialized by `body`, under `layout`.
fn with_targets(layout: &str, ty: &str, body: &str) -> Parsed {
    Parsed::new(layout, &format!("{TARGETS}\n{}", global("g", ty, body)))
}

#[test]
fn a_pointer_global_is_a_relocation_over_zero_bytes() {
    let parsed =
        with_targets(LITTLE32, "ptr<i32>", "%v = builtin.global_symbol @target : ptr<i32>;");
    let init = parsed.initializer("g");
    assert_eq!((init.size, init.align), (4, 4));
    assert_eq!(init.bytes, [0, 0, 0, 0]);
    assert!(!init.is_zero());
    assert_eq!(init.relocations, [parsed.data(0, 4, "target", 0)]);
}

#[test]
fn a_pointer_field_is_a_relocation_at_its_offset() {
    let ty = "struct<i32, ptr<i32>>";
    let body = format!(
        "%a = arith.constant 7 : i32;\n%p = builtin.global_symbol @target : ptr<i32>;\n%v = \
         builtin.aggregate %a, %p : {ty};"
    );
    let parsed = with_targets(LITTLE32, ty, &body);
    let init = parsed.initializer("g");
    assert_eq!(init.bytes, [7, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(init.relocations, [parsed.data(4, 4, "target", 0)]);
}

#[test]
fn a_gep_with_a_path_two_levels_deep_adds_the_element_s_offset() {
    // `@table`: the array at 4, its struct elements 8 bytes apart, the `i32` of each at 4
    let ty = "ptr<i32>";
    let body = "%p = builtin.global_symbol @table : ptr<struct<i8, array<struct<i16, i32>; \
                4>>>;\n%v = mem.gep %p, [1, 2, 1];";
    let parsed = with_targets(LITTLE32, ty, body);
    assert_eq!(parsed.initializer("g").relocations, [parsed.data(0, 4, "table", 24)]);
}

#[test]
fn a_gep_with_a_constant_index_steps_over_whole_pointees() {
    // `@table` is 36 bytes
    let body = "%p = builtin.global_symbol @table : ptr<struct<i8, array<struct<i16, i32>; \
                4>>>;\n%i = arith.constant 2 : isize;\n%v = mem.gep %p, %i, [1, 2, 1];";
    let parsed = with_targets(LITTLE32, "ptr<i32>", body);
    assert_eq!(parsed.initializer("g").relocations, [parsed.data(0, 4, "table", 72 + 24)]);

    // An index computed by a fold, and a gep of a gep
    let body = "%p = builtin.global_symbol @table : ptr<struct<i8, array<struct<i16, i32>; \
                4>>>;\n%a = arith.constant 1 : isize;\n%i = arith.add %a, %a;\n%q = mem.gep %p, \
                %i, [1];\n%v = mem.gep %q, [3];";
    let parsed = with_targets(LITTLE32, "ptr<struct<i16, i32>>", body);
    assert_eq!(parsed.initializer("g").relocations, [parsed.data(0, 4, "table", 72 + 4 + 24)]);
}

#[test]
fn a_gep_with_a_negative_index_gives_a_negative_addend() {
    let body = "%p = builtin.global_symbol @target : ptr<i32>;\n%i = arith.constant -3 : \
                isize;\n%v = mem.gep %p, %i;";
    let parsed = with_targets(LITTLE32, "ptr<i32>", body);
    assert_eq!(parsed.initializer("g").relocations, [parsed.data(0, 4, "target", -12)]);
}

#[test]
fn a_tagged_pointer_is_a_gep_over_bytes_then_ptrtoint() {
    let body = "%p = builtin.global_symbol @bytes : ptr<array<i8; 8>>;\n%q = mem.ptrcast %p : \
                ptr<i8>;\n%tag = arith.constant 3 : isize;\n%r = mem.gep %q, %tag;\n%v = \
                mem.ptrtoint %r;";
    let parsed = with_targets(LITTLE32, "isize", body);
    let init = parsed.initializer("g");
    assert_eq!((init.size, init.align), (4, 4));
    assert_eq!(init.bytes, [0, 0, 0, 0]);
    assert_eq!(init.relocations, [parsed.data(0, 4, "bytes", 3)]);

    // `mem.inttoptr` of the integer gives the address back
    let body = "%p = builtin.global_symbol @bytes : ptr<array<i8; 8>>;\n%q = mem.ptrcast %p : \
                ptr<i8>;\n%tag = arith.constant 3 : isize;\n%r = mem.gep %q, %tag;\n%i = \
                mem.ptrtoint %r;\n%v = mem.inttoptr %i : ptr<i16>;";
    let parsed = with_targets(LITTLE32, "ptr<i16>", body);
    assert_eq!(parsed.initializer("g").relocations, [parsed.data(0, 4, "bytes", 3)]);
}

#[test]
fn a_ptrcast_keeps_the_address() {
    let body = "%p = builtin.global_symbol @target : ptr<i32>;\n%v = mem.ptrcast %p : ptr;";
    let parsed = with_targets(LITTLE32, "ptr", body);
    assert_eq!(parsed.initializer("g").relocations, [parsed.data(0, 4, "target", 0)]);
}

#[test]
fn an_address_under_a_64_bit_layout_is_eight_bytes() {
    let ty = "struct<i8, ptr<i32>>";
    let body = format!(
        "%a = arith.constant 1 : i8;\n%p = builtin.global_symbol @target : ptr<i32>;\n%v = \
         builtin.aggregate %a, %p : {ty};"
    );
    let parsed = with_targets("little, ptr = 64, fn = 32", ty, &body);
    let init = parsed.initializer("g");
    assert_eq!((init.size, init.align), (16, 8));
    assert_eq!(init.bytes, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(init.relocations, [parsed.data(8, 8, "target", 0)]);
}

#[test]
fn two_relocations_in_one_struct_are_in_order_of_offset() {
    let ty = "struct<ptr<i32>, (i32) -> i32, ptr<i8>, i32>";
    let body = format!(
        "%p = builtin.global_symbol @target : ptr<i32>;\n%f = builtin.function_ref @callee : \
         (i32) -> i32;\n%b = builtin.global_symbol @bytes : ptr<array<i8; 8>>;\n%c = mem.ptrcast \
         %b : ptr<i8>;\n%q = mem.gep %c, [];\n%k = arith.constant 9 : i32;\n%v = \
         builtin.aggregate %p, %f, %q, %k : {ty};"
    );
    let parsed = with_targets(LITTLE32, ty, &body);
    let init = parsed.initializer("g");
    assert_eq!(init.bytes, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0]);
    assert_eq!(
        init.relocations,
        [
            parsed.data(0, 4, "target", 0),
            parsed.function(4, 4, "callee"),
            parsed.data(8, 4, "bytes", 0),
        ]
    );
}

// ---- Functions ----------------------------------------------------------------------------------

#[test]
fn a_function_field_is_a_relocation_to_the_function() {
    let ty = "struct<i8, (i32) -> i32>";
    let body = format!(
        "%a = arith.constant 1 : i8;\n%f = builtin.function_ref @callee : (i32) -> i32;\n%v = \
         builtin.aggregate %a, %f : {ty};"
    );
    let parsed = with_targets(LITTLE32, ty, &body);
    let init = parsed.initializer("g");
    assert_eq!((init.size, init.align), (8, 4));
    assert_eq!(init.bytes, [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(init.relocations, [parsed.function(4, 4, "callee")]);
}

#[test]
fn a_function_through_an_alias_names_the_function_it_leads_to() {
    let body = "%v = builtin.function_ref @alias : (i32) -> i32;";
    let parsed = with_targets(LITTLE32, "(i32) -> i32", body);
    assert_eq!(parsed.initializer("g").relocations, [parsed.function(0, 4, "callee")]);
}

#[test]
fn a_function_declaration_is_a_relocation_to_the_declaration() {
    let body = "%v = builtin.function_ref @declared : (i32) -> i32;";
    let parsed = with_targets(LITTLE32, "(i32) -> i32", body);
    assert_eq!(parsed.initializer("g").relocations, [parsed.function(0, 4, "declared")]);
}

#[test]
fn a_function_value_is_eight_bytes_under_fn_64() {
    let ty = "struct<i8, (i32) -> i32>";
    let body = format!(
        "%a = arith.constant 1 : i8;\n%f = builtin.function_ref @callee : (i32) -> i32;\n%v = \
         builtin.aggregate %a, %f : {ty};"
    );
    let parsed = with_targets("little, ptr = 32, fn = 64", ty, &body);
    let init = parsed.initializer("g");
    assert_eq!((init.size, init.align), (16, 8));
    assert_eq!(init.relocations, [parsed.function(8, 8, "callee")]);
}

// ---- Errors -------------------------------------------------------------------------------------

#[test]
fn arithmetic_on_an_address_is_an_error_that_says_to_use_a_gep() {
    let body = "%p = builtin.global_symbol @target : ptr<i32>;\n%i = mem.ptrtoint %p;\n%k = \
                arith.constant 4 : isize;\n%v = arith.add %i, %k;";
    let error = with_targets(LITTLE32, "isize", body).error("g");
    assert_mentions(
        &error,
        &[
            "the initializer of `@g` is not a constant that can be placed",
            "arithmetic on an address cannot be placed: write the offset with `mem.gep`",
            "the global variable it initializes",
        ],
    );
    // Widening the address to an `i64` is arithmetic on it too
    let body = "%p = builtin.global_symbol @target : ptr<i32>;\n%i = mem.ptrtoint %p;\n%v = \
                arith.zext_or_trunc %i <{ ty = i64 }>;";
    let error = with_targets(LITTLE32, "i64", body).error("g");
    assert_mentions(&error, &["arithmetic on an address cannot be placed"]);
}

#[test]
fn a_ptrtoint_to_an_integer_of_another_width_than_the_pointer_is_an_error_naming_the_cast() {
    // `mem.ptrtoint` gives an `isize`; a 32-bit `isize` of a 64-bit pointer is narrower
    let body = "%p = builtin.global_symbol @target : ptr<i32>;\n%v = mem.ptrtoint %p;";
    let error = with_targets("little, ptr = 64 index 32, fn = 32", "isize", body).error("g");
    assert_mentions(
        &error,
        &[
            "the initializer of `@g` is not a constant that can be placed",
            "`mem.ptrtoint` casts an address between `ptr<i32>`, of 64 bits, and `isize` of 32 \
             bits: an address in an integer must be in an integer exactly as wide as the pointer",
        ],
    );
    // A 32-bit `isize` of a 16-bit pointer of address space 1 is wider
    let items = "builtin.global_variable public @small : i8 { %v = arith.constant 1 : i8; \
                 builtin.yield %v; };";
    let body = "%p = builtin.global_symbol @small : ptr<i8, 1>;\n%v = mem.ptrtoint %p;";
    let text = format!("{items}\n{}", global("g", "isize", body));
    let error = Parsed::new("little, ptr = 32, ptr<1> = 16", &text).error("g");
    assert_mentions(
        &error,
        &[
            "`mem.ptrtoint` casts an address between `ptr<i8, 1>`, of 16 bits, and `isize` of 32 \
             bits",
        ],
    );
    // `mem.inttoptr` of a 32-bit address to a 16-bit pointer, likewise
    let body = "%p = builtin.global_symbol @small : ptr<i8>;\n%i = mem.ptrtoint %p;\n%v = \
                mem.inttoptr %i : ptr<i8, 1>;";
    let text = format!("{items}\n{}", global("g", "ptr<i8, 1>", body));
    let error = Parsed::new("little, ptr = 32, ptr<1> = 16", &text).error("g");
    assert_mentions(
        &error,
        &[
            "`mem.inttoptr` casts an address between `ptr<i8, 1>`, of 16 bits, and `isize` of 32 \
             bits",
        ],
    );
    // A `mem.ptrtoint` to `i64` or `i16` cannot be written: its result is an `isize`
    for ty in ["i64", "i16"] {
        let body = format!(
            "%p = builtin.global_symbol @target : ptr<i32>;\n%v = \"mem.ptrtoint\"(%p) : \
             (ptr<i32>) -> {ty};"
        );
        let text = format!("{TARGETS}\n{}", global("g", ty, &body));
        let text = format!(
            "builtin.module public @m attributes {{ data_layout = \
             #builtin.data_layout<{LITTLE32}> }} {{\n{text}\n}};"
        );
        let error = parse::<Module>(&text).err().unwrap_or_else(|| panic!("{ty} parsed"));
        assert_mentions(&error, &["expected the pointer-width integer type `isize`"]);
    }
}

#[test]
fn a_function_value_with_no_fn_entry_is_an_error_that_says_how_to_state_it() {
    let body = "%v = builtin.function_ref @callee : (i32) -> i32;";
    let error = with_targets("little, ptr = 32", "(i32) -> i32", body).error("g");
    assert_mentions(
        &error,
        &[
            "the initializer of `@g` is not a constant that can be placed",
            "a value of type `(i32) -> i32` cannot be placed: type '(i32) -> i32' has no \
             representation in memory: a function value has a layout only where the data layout \
             states one, as in `fn = 32`",
        ],
    );
    // In a struct, the struct has no layout either
    let ty = "struct<i8, (i32) -> i32>";
    let body = format!(
        "%a = arith.constant 1 : i8;\n%f = builtin.function_ref @callee : (i32) -> i32;\n%v = \
         builtin.aggregate %a, %f : {ty};"
    );
    let error = with_targets("little, ptr = 32", ty, &body).error("g");
    assert_mentions(&error, &["as in `fn = 32`"]);
}

#[test]
fn an_operation_that_does_not_fold_is_an_error_naming_it() {
    let body =
        "%a = arith.constant 1 : i32;\n%z = arith.constant 0 : i32;\n%v = arith.udiv %a, %z;";
    let error = Parsed::new(LITTLE32, &global("g", "i32", body)).error("g");
    assert_mentions(
        &error,
        &[
            "the initializer of `@g` is not a constant that can be placed",
            "`arith.udiv` does not fold to a constant",
            "the global variable it initializes",
        ],
    );
}

#[test]
fn a_value_of_a_type_with_no_layout_is_an_error() {
    let body = "%v = builtin.zero : (i32) -> i32;";
    let error = Parsed::new("little, ptr = 32", &global("g", "(i32) -> i32", body)).error("g");
    assert_mentions(
        &error,
        &[
            "the initializer of `@g` is not a constant that can be placed",
            "a value of type `(i32) -> i32` cannot be placed",
        ],
    );
    // An element of an aggregate that the layout places is checked too
    let body = "%z = builtin.zero : never;\n%v = builtin.aggregate %z : struct<never>;";
    let text = global("g", "struct<never>", body);
    let error = Parsed::new(LITTLE32, &text).error("g");
    assert_mentions(&error, &["a value of type `struct<never>` cannot be placed"]);
}

#[test]
fn an_integer_that_does_not_fit_its_type_is_an_error() {
    // An `isize` constant is 64 bits wide; the layout's `isize` is 4 bytes
    let error = Parsed::new(LITTLE32, &constant("isize", "1099511627776")).error("g");
    assert_mentions(
        &error,
        &[
            "the initializer of `@g` is not a constant that can be placed",
            "`arith.constant` gives 1099511627776, which does not fit `isize` of 4 bytes",
        ],
    );
}

#[test]
fn a_declaration_is_an_error() {
    let parsed = Parsed::new(LITTLE32, "builtin.global_variable public @d : i32;");
    let error = parsed.error("d");
    assert_mentions(
        &error,
        &[
            "the global variable `@d` has no initializer to evaluate",
            "this global variable is a declaration: its bytes are defined elsewhere",
        ],
    );
}

#[test]
fn the_target_of_a_relocation_prints_its_symbol() {
    let parsed =
        with_targets(LITTLE32, "ptr<i32>", "%v = builtin.global_symbol @target : ptr<i32>;");
    let init = parsed.initializer("g");
    assert_eq!(format!("{:?}", init.relocations[0].target), "Data(@target)");
    assert_eq!(format!("{:?}", parsed.function(0, 4, "callee").target), "Function(@callee)");
}
