//! The whole programs of the run suite, `tests/run/wasm/*.runic`, compiled through the compile
//! harness with the backend `wasm` and run under `wasmi`.
//!
//! The text of each program is read from its file, so that the lit test and this one cannot drift
//! apart. Each program is compiled at the optimization level none, as `runic-opt` compiles it, and
//! at the default level, where the optimization pipeline of the level runs before the lowering and
//! the backend's `cse` and `dce` after it. Every call is made on a fresh instance, as each
//! `wasmtime run --invoke` of the lit test is, and states the results the lit test checks, or the
//! trap whose message it checks. So the programs are checked on a machine with no `wasmtime`, and
//! by two engines where it is installed. Each call has the fuel of [super::super::exec::WASMI_FUEL]: a call that does
//! not terminate fails with a message instead of hanging.

use alloc::{string::String, vec, vec::Vec};

use runic_compile::{
    Compilation, CompilationRequest, Compiler, EmitRequest, EmitTo, Emitted, Input, IrFrontend,
    OptLevel,
};
use runic_ir::testing::text::describe;
use wasmi::TrapCode;

use crate::{
    backend::{TRIPLE, WasmBackend},
    tests::exec::{did_not_terminate, fueled_engine, fueled_store, out_of_fuel},
};

const SUM: &str = include_str!("../../../../../../tests/run/wasm/sum.runic");
const MACHINE: &str = include_str!("../../../../../../tests/run/wasm/machine.runic");
const DISPATCH: &str = include_str!("../../../../../../tests/run/wasm/dispatch.runic");
const NARROW: &str = include_str!("../../../../../../tests/run/wasm/narrow.runic");
const FRAMES: &str = include_str!("../../../../../../tests/run/wasm/frames.runic");
const GLOBALS: &str = include_str!("../../../../../../tests/run/wasm/globals.runic");
const FLOAT: &str = include_str!("../../../../../../tests/run/wasm/float.runic");
const TRAPS: &str = include_str!("../../../../../../tests/run/wasm/traps.runic");
const IMPORTS: &str = include_str!("../../../../../../tests/run/wasm/imports.runic");
const IMPORTS_ENV: &str = include_str!("../../../../../../tests/run/wasm/imports-env.runic");

/// Each program by its file name.
const PROGRAMS: [(&str, &str); 10] = [
    ("sum.runic", SUM),
    ("machine.runic", MACHINE),
    ("dispatch.runic", DISPATCH),
    ("narrow.runic", NARROW),
    ("frames.runic", FRAMES),
    ("globals.runic", GLOBALS),
    ("float.runic", FLOAT),
    ("traps.runic", TRAPS),
    ("imports.runic", IMPORTS),
    ("imports-env.runic", IMPORTS_ENV),
];

/// The two levels each program is compiled at.
const LEVELS: [OptLevel; 2] = [OptLevel::None, OptLevel::Balanced];

/// A value a function takes or returns, as the lit test writes it; a float by its bits, so that
/// values compare exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum V {
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
}

fn i32(value: i32) -> V {
    V::I32(value)
}

fn i64(value: i64) -> V {
    V::I64(value)
}

fn f32(value: f32) -> V {
    V::F32(value.to_bits())
}

fn f64(value: f64) -> V {
    V::F64(value.to_bits())
}

impl V {
    fn to_wasmi(self) -> wasmi::Val {
        match self {
            V::I32(value) => wasmi::Val::I32(value),
            V::I64(value) => wasmi::Val::I64(value),
            V::F32(bits) => wasmi::Val::F32(wasmi::F32::from_bits(bits)),
            V::F64(bits) => wasmi::Val::F64(wasmi::F64::from_bits(bits)),
        }
    }

    fn from_wasmi(value: &wasmi::Val) -> Self {
        match value {
            wasmi::Val::I32(value) => V::I32(*value),
            wasmi::Val::I64(value) => V::I64(*value),
            wasmi::Val::F32(value) => V::F32(value.to_bits()),
            wasmi::Val::F64(value) => V::F64(value.to_bits()),
            other => panic!("a function returned the reference {other:?}"),
        }
    }
}

/// The program `text` compiled for `wasm32-unknown-unknown` at `level`: the binary module.
#[track_caller]
fn compiled(text: &str, level: OptLevel) -> Vec<u8> {
    let compiler = Compiler::builder()
        .frontend(IrFrontend)
        .backend(WasmBackend)
        .build()
        .expect("the IR frontend and the Wasm backend form a valid compiler");
    let mut request = CompilationRequest::new();
    request.name = Some("program".into());
    request.inputs = vec![Input::memory("program.runic", text)];
    request.target = Some(TRIPLE.parse().unwrap());
    request.opt_level = level;
    request.emit.push(EmitRequest {
        checkpoint: "wasm.module".into(),
        to: EmitTo::Memory,
    });
    let compilation: Compilation = compiler.compile(request, &mut ());
    assert!(
        compilation.succeeded(),
        "-O{level}: {}",
        compilation.error().map(describe).unwrap_or_default()
    );
    compilation
        .emitted()
        .iter()
        .find(|emitted| matches!(emitted, Emitted::Memory { .. }))
        .and_then(|emitted| emitted.bytes())
        .expect("the module is delivered to memory")
        .to_vec()
}

/// A program compiled at one level, and the module it imports from `env`, if any.
struct Program {
    level: OptLevel,
    bytes: Vec<u8>,
    env: Option<Vec<u8>>,
}

impl Program {
    /// Calls the export `name` of a fresh instance on `arguments`: the results, or the trap.
    #[track_caller]
    fn call(&self, name: &str, arguments: &[V]) -> Result<Vec<V>, TrapCode> {
        let engine = fueled_engine();
        let mut store = fueled_store(&engine);
        let mut linker = wasmi::Linker::<()>::new(&engine);
        if let Some(env) = &self.env {
            let module = wasmi::Module::new(&engine, &env[..])
                .unwrap_or_else(|error| panic!("wasmi rejects the module `env`: {error}"));
            let instance = wasmi::Linker::<()>::new(&engine)
                .instantiate_and_start(&mut store, &module)
                .unwrap_or_else(|error| panic!("the module `env` does not instantiate: {error}"));
            linker.instance(&mut store, "env", instance).expect("one module `env`");
        }
        let module = wasmi::Module::new(&engine, &self.bytes[..])
            .unwrap_or_else(|error| panic!("wasmi rejects the emitted module: {error}"));
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .unwrap_or_else(|error| panic!("the emitted module does not instantiate: {error}"));
        let func = instance
            .get_func(&store, name)
            .unwrap_or_else(|| panic!("the emitted module exports no function `{name}`"));
        let ty = func.ty(&store);
        let inputs = arguments.iter().map(|value| value.to_wasmi()).collect::<Vec<_>>();
        let mut outputs =
            ty.results().iter().map(|ty| wasmi::Val::default(*ty)).collect::<Vec<_>>();
        match func.call(&mut store, &inputs, &mut outputs) {
            Ok(()) => Ok(outputs.iter().map(V::from_wasmi).collect()),
            Err(error) if out_of_fuel(&store, &error) => did_not_terminate(name),
            Err(error) => Err(error.as_trap_code().unwrap_or_else(|| {
                panic!("-O{}: running `{name}` failed without a trap: {error}", self.level)
            })),
        }
    }

    /// Requires the export `name` to return `results` on `arguments`.
    #[track_caller]
    fn gives(&self, name: &str, arguments: &[V], results: &[V]) {
        assert_eq!(
            self.call(name, arguments),
            Ok(results.to_vec()),
            "-O{}: {name}{arguments:?}",
            self.level
        );
    }

    /// Requires the export `name` to trap with `trap` on `arguments`.
    #[track_caller]
    fn traps(&self, name: &str, arguments: &[V], trap: TrapCode) {
        assert_eq!(self.call(name, arguments), Err(trap), "-O{}: {name}{arguments:?}", self.level);
    }
}

/// The program `text` at each level of [LEVELS].
fn at_each_level(text: &str) -> impl Iterator<Item = Program> + '_ {
    LEVELS.into_iter().map(move |level| Program {
        level,
        bytes: compiled(text, level),
        env: None,
    })
}

#[test]
fn the_sum_over_an_array_in_a_global_variable_adds_the_first_n_words() {
    for program in at_each_level(SUM) {
        for (n, sum) in [(0, 0), (1, 1), (3, 7), (7, 127), (8, -1)] {
            program.gives("sum", &[i32(n)], &[i32(sum)]);
        }
    }
}

#[test]
fn the_state_machine_returns_two_results_and_a_million_tail_calls_do_not_grow_the_stack() {
    for program in at_each_level(MACHINE) {
        for (start, flag, end) in [(0, 1, 7), (1, 1, 1), (2, 0, 3), (4, 0, 7), (-8, 1, -1)] {
            program.gives("machine", &[i32(7), i64(start)], &[i32(flag), i64(end)]);
        }
        program.gives("countdown", &[i32(1_000_000), i64(0)], &[i64(500_000_500_000)]);
        program.traps("count_by_calls", &[i32(1_000_000), i64(0)], TrapCode::StackOverflow);
    }
}

#[test]
fn the_dispatch_takes_each_of_its_three_exits() {
    for program in at_each_level(DISPATCH) {
        program.gives("dispatch", &[i32(0), i32(5)], &[i32(50)]);
        program.gives("dispatch", &[i32(1), i32(5)], &[i32(4)]);
        program.gives("dispatch", &[i32(2), i32(5)], &[i32(24)]);
        program.gives("dispatch", &[i32(-1), i32(-7)], &[i32(0)]);
    }
}

#[test]
fn narrow_integers_wrap_and_extend_under_both_conventions() {
    for program in at_each_level(NARROW) {
        program.gives("narrow", &[i32(200), i32(100), i32(5)], &[i32(1005), i32(1), i32(0)]);
        program.gives("narrow", &[i32(100), i32(50), i32(65000)], &[i32(65000), i32(0), i32(1)]);
        program.gives("narrow", &[i32(200), i32(100), i32(65000)], &[i32(464), i32(1), i32(0)]);
        program.gives("narrow_c", &[i32(-56), i32(100), i32(-536)], &[i32(464), i32(1), i32(0)]);
        program.gives("widen", &[i32(200)], &[i32(-56), i32(200)]);
        program.gives("widen", &[i32(-56)], &[i32(-56), i32(200)]);
        program.gives("bits", &[i32(1), i32(1)], &[i32(0), i32(1), i32(1)]);
        program.gives("bits", &[i32(1), i32(0)], &[i32(1), i32(0), i32(1)]);
        program.gives("calls", &[i32(200), i32(300), i32(1)], &[i32(1), i32(56), i32(65480)]);
        program.gives("calls", &[i32(200), i32(100), i32(0)], &[i32(0), i32(200), i32(200)]);
    }
}

#[test]
fn locals_in_frames_are_given_back_by_every_return() {
    for program in at_each_level(FRAMES) {
        program.gives("locals", &[i32(5), i32(1)], &[i32(10)]);
        program.gives("locals", &[i32(5), i32(0)], &[i32(20)]);
        program.gives("repeat", &[i32(100_000)], &[i32(1_048_560), i32(1_048_560), i32(1_100_000)]);
        program.gives("probe", &[], &[i32(1_048_560)]);
    }
}

#[test]
fn global_variables_are_read_and_written_across_calls_and_a_function_value_in_data_is_called() {
    for program in at_each_level(GLOBALS) {
        // No function has a frame, so there is no stack and the data starts at 1024.
        program.gives("bump", &[i32(3)], &[i32(5), i32(1024)]);
        program.gives("bump_thrice", &[i32(3)], &[i32(5), i32(8), i32(11), i32(14)]);
        program.gives("follow", &[], &[i32(7), i32(8)]);
        program.gives("call_stored", &[i32(21)], &[i32(42)]);
    }
}

#[test]
fn floats_compare_with_nans_and_convert_both_ways_saturating() {
    for program in at_each_level(FLOAT) {
        program.gives(
            "arith",
            &[f64(1.5), f64(-0.25)],
            &[f64(1.25), f64(1.75), f64(-0.375), f64(-6.0), f64(-1.5)],
        );
        program.gives(
            "arith",
            &[f64(1.0), f64(0.0)],
            &[f64(1.0), f64(1.0), f64(0.0), f64(f64::INFINITY), f64(-1.0)],
        );
        let nan = f32(f32::NAN);
        program.gives("compare", &[nan, f32(1.0)], &[i32(0), i32(0), i32(1), i32(1), i32(0)]);
        program.gives("compare", &[f32(1.0), f32(2.0)], &[i32(0), i32(1), i32(1), i32(0), i32(1)]);
        program.gives("to_int", &[f64(-1.5)], &[i32(-1), i32(0), i64(-1), i64(0)]);
        program.gives(
            "to_int",
            &[f64(1e10)],
            &[i32(i32::MAX), i32(-1), i64(10_000_000_000), i64(10_000_000_000)],
        );
        program.gives("to_int", &[f64(f64::NAN)], &[i32(0), i32(0), i64(0), i64(0)]);
        program.gives("from_int", &[i32(-1)], &[f64(-1.0), f64(4_294_967_295.0), f32(-1.0)]);
        program.gives("widths", &[f64(0.1)], &[f32(0.1), f64(f64::from(0.1f32))]);
    }
}

#[test]
fn each_trap_the_lowering_produces_is_the_trap_wasm_has_for_it() {
    for program in at_each_level(TRAPS) {
        program.gives("sdiv_i8", &[i32(-7), i32(2)], &[i32(253)]);
        program.traps("sdiv_i8", &[i32(-128), i32(-1)], TrapCode::UnreachableCodeReached);
        program.gives("sdiv_i32", &[i32(-7), i32(2)], &[i32(-3)]);
        program.traps("sdiv_i32", &[i32(i32::MIN), i32(-1)], TrapCode::IntegerOverflow);
        program.gives("sdiv_i64", &[i64(-7), i64(2)], &[i64(-3)]);
        program.traps("sdiv_i64", &[i64(i64::MIN), i64(-1)], TrapCode::IntegerOverflow);
        program.traps("udiv_i32", &[i32(7), i32(0)], TrapCode::IntegerDivisionByZero);
        program.gives("check", &[i32(3)], &[i32(3)]);
        program.traps("check", &[i32(0)], TrapCode::UnreachableCodeReached);
        program.gives("add_checked", &[i32(i32::MAX - 1), i32(1)], &[i32(i32::MAX)]);
        program.traps("add_checked", &[i32(i32::MAX), i32(1)], TrapCode::UnreachableCodeReached);
        program.gives("peek", &[i32(1024)], &[i32(42)]);
        program.traps("peek", &[i32(-16)], TrapCode::MemoryOutOfBounds);
        program.gives("call_through", &[i32(1), i32(21)], &[i32(42)]);
        program.traps("call_through", &[i32(0), i32(21)], TrapCode::IndirectCallToNull);
        program.traps("call_through", &[i32(2), i32(21)], TrapCode::TableOutOfBounds);
    }
}

#[test]
fn an_import_from_env_is_satisfied_by_a_second_emitted_module() {
    for level in LEVELS {
        let env = compiled(IMPORTS_ENV, level);
        let alone = Program {
            level,
            bytes: env.clone(),
            env: None,
        };
        alone.gives("host_scale", &[i32(14)], &[i32(42)]);
        alone.gives("host_swap", &[i32(1), i64(-2)], &[i64(-2), i32(1)]);
        let program = Program {
            level,
            bytes: compiled(IMPORTS, level),
            env: Some(env),
        };
        program.gives("scaled", &[i32(10)], &[i32(31)]);
        program.gives("swapped", &[i32(5)], &[i64(505)]);
    }
}

#[test]
fn every_program_of_the_run_suite_is_run_here() {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/run/wasm");
    let mut files = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("{directory}: {error}"))
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".runic"))
        .collect::<Vec<String>>();
    files.sort();
    let mut here = PROGRAMS.iter().map(|(name, _)| String::from(*name)).collect::<Vec<_>>();
    here.sort();
    assert_eq!(files, here, "a program of tests/run/wasm has no test here, or one is gone");
}
