//! Whole programs through the lowering pipeline, evaluated.
//!
//! The programs are those of `tests/lit/codegen/wasm/programs*.runic`, whose text is read from
//! there, so that the lit file and the test cannot drift apart. Each is lowered by the four
//! passes of [add_lowering_passes] and run by the evaluator on inputs whose results are stated by
//! hand in the test.

use alloc::{vec, vec::Vec};

use runic_ir::pass::{Nesting, PassManager, PassPipeline};

use super::{
    eval::{Machine, Outcome, Val},
    exec::{PAGE, run_in},
    *,
};
use crate::add_lowering_passes;

const PROGRAMS: &str = include_str!("../../../../../tests/lit/codegen/wasm/programs.runic");
const MEMORY: &str = include_str!("../../../../../tests/lit/codegen/wasm/programs-memory.runic");
const NARROW: &str = include_str!("../../../../../tests/lit/codegen/wasm/programs-narrow.runic");

/// The text of the module `@name` of `file`, a lit file whose modules are separated by
/// `// -----`.
fn program(file: &str, name: &str) -> String {
    let marker = format!("builtin.module public @{name} ");
    file.split("// -----")
        .find(|chunk| chunk.contains(&marker))
        .unwrap_or_else(|| panic!("no program @{name}"))
        .to_string()
}

/// The module `@name` of `file`, lowered by the passes of [add_lowering_passes].
fn lowered(file: &str, name: &str) -> (Rc<Context>, ModuleRef) {
    let (context, module) = parse_module(&program(file, name));
    let mut passes = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
    add_lowering_passes(passes.op_pass_manager_mut())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    passes.enable_verifier(true);
    passes
        .run(module.as_operation_ref())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    (context, module)
}

/// The function `@name` of `module`.
fn function(module: ModuleRef, name: &'static str) -> runic_ir::OperationRef {
    runic_ir::SymbolTable::symbol_manager(&*module.borrow())
        .lookup_op(name)
        .unwrap_or_else(|| panic!("no function @{name}"))
}

fn i32s(values: &[u32]) -> Vec<Val> {
    values.iter().map(|value| Val::i32(*value)).collect()
}

#[test]
fn the_array_sum_adds_the_words_it_is_given() {
    let (_context, module) = lowered(PROGRAMS, "sum");
    let sum = function(module, "sum");
    let mut machine = Machine::new(PAGE);
    machine.write_words(16, &[1, 2, 3, 4, 5]);
    let mut run = |pointer: u32, count: u32| run_in(&mut machine, sum, &i32s(&[pointer, count]));
    assert_eq!(run(16, 5), Outcome::Values(i32s(&[15])));
    assert_eq!(run(16, 1), Outcome::Values(i32s(&[1])));
    assert_eq!(run(20, 3), Outcome::Values(i32s(&[9])));
    // No element is read: a pointer outside the memory is not touched.
    assert_eq!(run(PAGE as u32 + 1000, 0), Outcome::Values(i32s(&[0])));
    // A read past the end of the memory traps.
    assert_eq!(run(PAGE as u32 - 4, 2), Outcome::Trap);
}

#[test]
fn the_array_sum_wraps_at_32_bits() {
    let (_context, module) = lowered(PROGRAMS, "sum");
    let mut machine = Machine::new(PAGE);
    machine.write_words(0, &[u32::MAX, 2, 7]);
    let outcome = run_in(&mut machine, function(module, "sum"), &i32s(&[0, 3]));
    assert_eq!(outcome, Outcome::Values(i32s(&[8])));
}

#[test]
fn the_state_machine_follows_its_tail_calls_to_a_return_of_two_values() {
    let (_context, module) = lowered(PROGRAMS, "machine");
    let mut machine = Machine::new(0);
    let mut run = |state: &'static str, argument: u64| {
        let arguments = [Val::i32(0), Val::i64(argument)];
        run_in(&mut machine, function(module, state), &arguments)
    };
    let result =
        |flag: u32, argument: u64| Outcome::Values(vec![Val::i32(flag), Val::i64(argument)]);
    // Bit 0 set: returned at once, with `true`.
    assert_eq!(run("state_a", 5), result(1, 5));
    // 0 -> `@state_b`(1) -> `@state_c`(3) -> `@state_a`(7) -> returns `true`.
    assert_eq!(run("state_a", 0), result(1, 7));
    // 2 -> `@state_b`(3), whose bit 1 is set: returns `false`.
    assert_eq!(run("state_a", 2), result(0, 3));
    // 4 -> `@state_b`(5) -> `@state_c`(7), whose bit 2 is set: returns `false`.
    assert_eq!(run("state_a", 4), result(0, 7));
    // The argument is 64 bits wide: the tests and the sums use all of it.
    assert_eq!(run("state_a", 1 << 32), result(1, (1 << 32) + 7));
    // Entering at the other two states.
    assert_eq!(run("state_b", 2), result(0, 2));
    assert_eq!(run("state_c", 0), result(0, 7));
}

#[test]
fn narrow_integers_wrap_in_their_own_width_and_compare_as_signed_or_unsigned() {
    let (_context, module) = lowered(NARROW, "narrow");
    let narrow = function(module, "narrow");
    // (a, b, h) -> (the `i16` result, the `i8` sum overflowed as unsigned, the `i8` sum is
    // negative as signed). The `i8` sum is `(a + b) mod 256`; the result is `h`, or `h + 1000`
    // modulo 65536 when the sum overflowed.
    let cases: [([u32; 3], [u32; 3]); 6] = [
        // 200 + 100 = 300 = 44: overflowed; 65000 + 1000 = 66000 = 464.
        ([200, 100, 65000], [464, 1, 0]),
        ([1, 2, 7], [7, 0, 0]),
        // 100 + 100 = 200, which is -56 as an `i8`: negative, and not overflowed.
        ([100, 100, 1], [1, 0, 1]),
        // 255 + 1 = 0: overflowed.
        ([255, 1, 0], [1000, 1, 0]),
        // 128 + 128 = 0: overflowed; 65535 + 1000 = 66535 = 999.
        ([128, 128, 65535], [999, 1, 0]),
        // 127 + 1 = 128 = -128 as an `i8`: negative, and not overflowed.
        ([127, 1, 5], [5, 0, 1]),
    ];
    let mut machine = Machine::new(0);
    for (arguments, expected) in cases {
        let outcome = run_in(&mut machine, narrow, &i32s(&arguments));
        assert_eq!(outcome, Outcome::Values(i32s(&expected)), "narrow{arguments:?}");
    }
}

#[test]
fn the_global_counter_is_read_and_written_at_the_address_it_is_given() {
    let (_context, module) = lowered(MEMORY, "global");
    let bump = function(module, "bump");
    let mut machine = Machine::new(PAGE);
    machine.data.insert("counter".into(), 8);
    machine.write_words(8, &[5]);
    // The old value and the address come back, and the new value is in memory.
    assert_eq!(run_in(&mut machine, bump, &i32s(&[3])), Outcome::Values(i32s(&[5, 8])));
    assert_eq!(machine.word(8), 8);
    assert_eq!(run_in(&mut machine, bump, &i32s(&[10])), Outcome::Values(i32s(&[8, 8])));
    assert_eq!(machine.word(8), 18);
}

/// The text of the module `name` of `file` after the pass pipeline `pipeline`, read from text and
/// loaded the way `runic-opt -p` does.
fn lowered_by_name(file: &str, name: &str, pipeline: &str) -> String {
    let (context, module) = parse_module(&program(file, name));
    let anchor = module.borrow().as_operation().name();
    let mut passes = PassPipeline::parse_elements(&anchor, pipeline)
        .and_then(|pipeline| pipeline.load(context, Nesting::Implicit))
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    passes.enable_verifier(true);
    passes
        .run(module.as_operation_ref())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    text_of(module)
}

#[test]
fn add_lowering_passes_runs_what_the_four_passes_by_name_run() {
    let pipeline = "cfg-to-scf{exits=in-place},wasm-lower-frames,convert-to-wasm,wasm-verify-form";
    for (file, name) in [
        (PROGRAMS, "sum"),
        (PROGRAMS, "exits"),
        (PROGRAMS, "machine"),
        (MEMORY, "frame"),
        (MEMORY, "global"),
        (MEMORY, "indirect"),
        (NARROW, "narrow"),
    ] {
        let (_context, module) = lowered(file, name);
        assert_eq!(text_of(module), lowered_by_name(file, name, pipeline), "the program @{name}");
    }
}
