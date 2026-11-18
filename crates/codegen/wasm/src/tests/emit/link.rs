//! Objects linked with the linker of the pinned toolchain, `rust-lld -flavor wasm` from its
//! sysroot, and the linked modules run under `wasmi` with fuel: the programs of the run suite,
//! and objects that call each other, share data and functions, contribute to one section, have
//! start functions, and leave symbols for the linker to resolve.
//!
//! Every link is `rust-lld -flavor wasm --no-entry --stack-first -z stack-size=65536
//! --fatal-warnings <extra> 0.o 1.o ... -o linked.wasm` ([link]), in a temporary directory, and the
//! output is validated: a warning of the linker (a call whose signature differs from the
//! definition's, which it replaces with a trap) fails the test. Only the test of a link that
//! fails leaves `--fatal-warnings` out ([link_allowing_warnings]).

use alloc::{format, string::String, vec, vec::Vec};
use std::{path::PathBuf, process::Command};

use runic_compile::{
    Backend, CheckpointDecl, Checkpoints, Compilation, CompilationRequest, Compiler, EmitRequest,
    EmitTo, Emitted, Flow, Input, IrFrontend, OptLevel, OptionsDecl, Session, Triple,
};
use runic_ir::{
    DataLayoutSpec, Report, dialects::builtin::WorldRef, pass::OpPassManager,
    testing::text::describe,
};
use wasmparser::{Parser, Payload, TypeRef};

use super::object::{module, object_of, try_object_of};
use crate::{
    backend::{MODULE, TRIPLE, WasmBackend},
    emit::{EmitOptions, emit_module, emit_object, validate},
    tests::exec::{did_not_terminate, fueled_engine, fueled_store, out_of_fuel},
};

// ---- The linker ---------------------------------------------------------------------------------

/// The size of the stack every link reserves, and the stack size of every complete module the
/// tests compare a linked one with.
const STACK_SIZE: u32 = 65_536;

/// What every link of these tests passes to the linker: no entry point, the stack first, and a
/// stack of [STACK_SIZE] bytes.
const LINK_ARGS: [&str; 4] = ["--no-entry", "--stack-first", "-z", "stack-size=65536"];

/// What turns every warning of the linker into a failed link.
const FATAL_WARNINGS: &str = "--fatal-warnings";

/// The output of `command`, or the panic that says it did not run.
#[track_caller]
fn run(command: &mut Command) -> std::process::Output {
    command
        .output()
        .unwrap_or_else(|error| panic!("could not run {command:?}: {error}"))
}

/// The `rust-lld` of the pinned toolchain's sysroot: `rustc --print sysroot` run in this crate
/// (so that `rust-toolchain.toml` picks the toolchain), then `lib/rustlib/<host>/bin/rust-lld`,
/// the host from `rustc -vV`. A missing linker fails the test with one line that says which path
/// it looked at.
#[track_caller]
fn linker() -> PathBuf {
    let directory = env!("CARGO_MANIFEST_DIR");
    let sysroot = run(Command::new("rustc").args(["--print", "sysroot"]).current_dir(directory));
    let sysroot = String::from_utf8(sysroot.stdout).expect("the sysroot is a path");
    let version = run(Command::new("rustc").arg("-vV").current_dir(directory));
    let version = String::from_utf8(version.stdout).expect("rustc -vV prints text");
    let host = version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .expect("rustc -vV names the host");
    let path = PathBuf::from(sysroot.trim())
        .join("lib/rustlib")
        .join(host)
        .join("bin/rust-lld");
    assert!(path.is_file(), "there is no linker at {}", path.display());
    path
}

/// The objects `objects`, written in that order to a temporary directory (`0.o`, `1.o`, ...),
/// linked with [LINK_ARGS], [FATAL_WARNINGS] and `extra`: the linked module, validated, or the
/// linker's message.
pub(super) fn link(objects: &[&[u8]], extra: &[&str]) -> Result<Vec<u8>, String> {
    let mut args = vec![FATAL_WARNINGS];
    args.extend_from_slice(extra);
    link_allowing_warnings(objects, &args)
}

/// The objects `objects` linked as [link] links them, but without [FATAL_WARNINGS]: for the test
/// of a link that fails.
fn link_allowing_warnings(objects: &[&[u8]], extra: &[&str]) -> Result<Vec<u8>, String> {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let mut command = Command::new(linker());
    command.args(["-flavor", "wasm"]).args(LINK_ARGS).args(extra);
    for (index, object) in objects.iter().enumerate() {
        let path = directory.path().join(format!("{index}.o"));
        std::fs::write(&path, object).expect("the object is written");
        command.arg(path);
    }
    let output = directory.path().join("linked.wasm");
    command.arg("-o").arg(&output);
    let result = run(&mut command);
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).into_owned());
    }
    let bytes = std::fs::read(&output).expect("the linker writes its output");
    validate::validate(&bytes).unwrap_or_else(|report| panic!("the linked module: {report}"));
    Ok(bytes)
}

/// The objects `objects` linked as [link] links them, which must succeed.
#[track_caller]
fn linked(objects: &[&[u8]], extra: &[&str]) -> Vec<u8> {
    link(objects, extra).unwrap_or_else(|message| panic!("the link fails:\n{message}"))
}

// ---- Running ------------------------------------------------------------------------------------

/// A value a function takes or returns.
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

/// What a call gives: its results, or the trap.
type Outcome = Result<Vec<V>, wasmi::TrapCode>;

/// A module instantiated by `wasmi`, with a host module when it imports one.
struct Instance {
    store: wasmi::Store<()>,
    instance: wasmi::Instance,
}

impl Instance {
    /// The module `bytes`, instantiated with no imports.
    #[track_caller]
    fn new(bytes: &[u8]) -> Self {
        Self::with_host(bytes, None)
    }

    /// The module `bytes`, instantiated with the module `host` (a complete module of its own)
    /// under the import module name `name`, when given.
    #[track_caller]
    fn with_host(bytes: &[u8], host: Option<(&str, &[u8])>) -> Self {
        let engine = fueled_engine();
        let mut store = fueled_store(&engine);
        let mut linker = wasmi::Linker::<()>::new(&engine);
        if let Some((name, host)) = host {
            let module = wasmi::Module::new(&engine, host)
                .unwrap_or_else(|error| panic!("wasmi rejects the host module: {error}"));
            let instance = wasmi::Linker::<()>::new(&engine)
                .instantiate_and_start(&mut store, &module)
                .unwrap_or_else(|error| panic!("the host module does not instantiate: {error}"));
            linker.instance(&mut store, name, instance).expect("one host module");
        }
        let module = wasmi::Module::new(&engine, bytes)
            .unwrap_or_else(|error| panic!("wasmi rejects the module: {error}"));
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .unwrap_or_else(|error| panic!("the module does not instantiate: {error}"));
        Self { store, instance }
    }

    /// Calls the export `name` on `arguments`: its results, or the trap.
    #[track_caller]
    fn call(&mut self, name: &str, arguments: &[V]) -> Outcome {
        let func = self
            .instance
            .get_func(&self.store, name)
            .unwrap_or_else(|| panic!("the module exports no function `{name}`"));
        let ty = func.ty(&self.store);
        let inputs = arguments.iter().map(|value| value.to_wasmi()).collect::<Vec<_>>();
        let mut outputs =
            ty.results().iter().map(|ty| wasmi::Val::default(*ty)).collect::<Vec<_>>();
        match func.call(&mut self.store, &inputs, &mut outputs) {
            Ok(()) => Ok(outputs.iter().map(V::from_wasmi).collect()),
            Err(error) if out_of_fuel(&self.store, &error) => did_not_terminate(name),
            Err(error) => Err(error
                .as_trap_code()
                .unwrap_or_else(|| panic!("running `{name}` failed without a trap: {error}"))),
        }
    }

    /// Requires the export `name` to give `results` on `arguments`.
    #[track_caller]
    fn gives(&mut self, name: &str, arguments: &[V], results: &[V]) {
        assert_eq!(self.call(name, arguments), Ok(results.to_vec()), "{name}{arguments:?}");
    }

    /// The value of the exported global `name`, an `i32`.
    #[track_caller]
    fn global(&self, name: &str) -> i32 {
        let global = self
            .instance
            .get_global(&self.store, name)
            .unwrap_or_else(|| panic!("the module exports no global `{name}`"));
        match global.get(&self.store) {
            wasmi::Val::I32(value) => value,
            other => panic!("the global `{name}` holds {other:?}"),
        }
    }
}

/// The names of the exports of the module `bytes`, in order.
fn exports(bytes: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    for payload in Parser::new(0).parse_all(bytes) {
        if let Payload::ExportSection(reader) = payload.expect("the module parses") {
            for export in reader {
                names.push(String::from(export.expect("an export").name));
            }
        }
    }
    names
}

/// The function imports of the module `bytes`: each module and field, in order.
fn function_imports(bytes: &[u8]) -> Vec<(String, String)> {
    let mut imports = Vec::new();
    for payload in Parser::new(0).parse_all(bytes) {
        if let Payload::ImportSection(reader) = payload.expect("the module parses") {
            for import in reader.into_imports() {
                let import = import.expect("an import");
                if matches!(import.ty, TypeRef::Func(_)) {
                    imports.push((String::from(import.module), String::from(import.name)));
                }
            }
        }
    }
    imports
}

/// The sizes of the data segments of the module `bytes`, in order.
fn data_sizes(bytes: &[u8]) -> Vec<usize> {
    let mut sizes = Vec::new();
    for payload in Parser::new(0).parse_all(bytes) {
        if let Payload::DataSection(reader) = payload.expect("the module parses") {
            for data in reader {
                sizes.push(data.expect("a data segment").data.len());
            }
        }
    }
    sizes
}

// ---- The programs of the run suite --------------------------------------------------------------

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
const DISPATCH_TABLE: &str = include_str!("../../../../../../tests/run/wasm/dispatch-table.runic");
const START: &str = include_str!("../../../../../../tests/run/wasm/start.runic");

/// The backend `wasm` with its own lowering, emitting an object or a complete module with a
/// stack of [STACK_SIZE] bytes, delivered at [MODULE]. So a program is compiled for an object as
/// the backend compiles it for a module, at each optimization level.
#[derive(Debug, Clone, Copy)]
struct Emitting {
    object: bool,
}

impl Backend for Emitting {
    fn name(&self) -> &'static str {
        WasmBackend.name()
    }

    fn supports(&self, triple: &Triple) -> bool {
        WasmBackend.supports(triple)
    }

    fn targets(&self) -> Vec<Triple> {
        WasmBackend.targets()
    }

    fn data_layout(&self, triple: &Triple) -> Result<DataLayoutSpec, Report> {
        WasmBackend.data_layout(triple)
    }

    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        WasmBackend.checkpoints()
    }

    fn options(&self) -> Option<OptionsDecl> {
        WasmBackend.options()
    }

    fn add_passes(&self, session: &Session<'_>, passes: &mut OpPassManager) -> Result<(), Report> {
        WasmBackend.add_passes(session, passes)
    }

    fn emit(
        &self,
        _session: &Session<'_>,
        world: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        let options = EmitOptions {
            stack_size: STACK_SIZE,
            ..EmitOptions::default()
        };
        let root = world.as_operation_ref();
        let bytes = if self.object {
            emit_object(root, &options)?
        } else {
            emit_module(root, &options)?
        };
        Ok(match checkpoints.reach(MODULE, bytes)? {
            Flow::Continue(_) => Flow::Continue(()),
            Flow::Break(stopped) => Flow::Break(stopped),
        })
    }
}

/// The program `text` compiled at `level` by [Emitting]: an object, or a complete module.
#[track_caller]
fn compiled(text: &str, level: OptLevel, object: bool) -> Vec<u8> {
    let compiler = Compiler::builder()
        .frontend(IrFrontend)
        .backend(Emitting { object })
        .build()
        .expect("the IR frontend and the backend form a valid compiler");
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
        .expect("the output is delivered to memory")
        .to_vec()
}

/// The two levels each program is compiled at.
const LEVELS: [OptLevel; 2] = [OptLevel::None, OptLevel::Balanced];

/// A program compiled at one level both ways: its complete module and its object linked alone
/// (with `extra`), each run on a fresh instance per call.
struct Both {
    level: OptLevel,
    module: Vec<u8>,
    linked: Vec<u8>,
    host: Option<(&'static str, Vec<u8>)>,
}

impl Both {
    /// The program `text` at `level`, its object linked with `extra`.
    #[track_caller]
    fn new(text: &str, level: OptLevel, extra: &[&str]) -> Self {
        let object = compiled(text, level, true);
        Self {
            level,
            module: compiled(text, level, false),
            linked: linked(&[&object], extra),
            host: None,
        }
    }

    fn instance(&self, bytes: &[u8]) -> Instance {
        let host = self.host.as_ref().map(|(name, bytes)| (*name, bytes.as_slice()));
        Instance::with_host(bytes, host)
    }

    /// The outcome of the export `name` on `arguments`, from the linked module.
    #[track_caller]
    fn linked_call(&self, name: &str, arguments: &[V]) -> Outcome {
        self.instance(&self.linked).call(name, arguments)
    }

    /// Requires the export `name` to give `results` on `arguments` in the linked module and in
    /// the complete module alike.
    #[track_caller]
    fn same(&self, name: &str, arguments: &[V], results: Outcome) {
        let module = self.instance(&self.module).call(name, arguments);
        assert_eq!(module, results, "-O{}: the complete module: {name}{arguments:?}", self.level);
        assert_eq!(
            self.linked_call(name, arguments),
            results,
            "-O{}: the linked object: {name}{arguments:?}",
            self.level
        );
    }

    /// Requires the export `name` to give `results` on `arguments` in both modules.
    #[track_caller]
    fn gives(&self, name: &str, arguments: &[V], results: &[V]) {
        self.same(name, arguments, Ok(results.to_vec()));
    }

    /// Requires the export `name` to trap with `trap` on `arguments` in both modules.
    #[track_caller]
    fn traps(&self, name: &str, arguments: &[V], trap: wasmi::TrapCode) {
        self.same(name, arguments, Err(trap));
    }
}

/// The program `text` at each level of [LEVELS], its object linked with `extra`.
fn at_each_level<'a>(text: &'a str, extra: &'a [&'a str]) -> impl Iterator<Item = Both> + 'a {
    LEVELS.into_iter().map(move |level| Both::new(text, level, extra))
}

#[test]
fn the_sum_program_linked_from_an_object_gives_what_its_module_gives() {
    for program in at_each_level(SUM, &[]) {
        for (n, sum) in [(0, 0), (1, 1), (3, 7), (7, 127), (8, -1)] {
            program.gives("sum", &[i32(n)], &[i32(sum)]);
        }
    }
}

#[test]
fn the_machine_program_linked_from_an_object_gives_what_its_module_gives() {
    for program in at_each_level(MACHINE, &[]) {
        for (start, flag, end) in [(0, 1, 7), (1, 1, 1), (2, 0, 3), (4, 0, 7), (-8, 1, -1)] {
            program.gives("machine", &[i32(7), i64(start)], &[i32(flag), i64(end)]);
        }
        program.gives("countdown", &[i32(1_000_000), i64(0)], &[i64(500_000_500_000)]);
        program.traps("count_by_calls", &[i32(1_000_000), i64(0)], wasmi::TrapCode::StackOverflow);
    }
}

#[test]
fn the_dispatch_program_linked_from_an_object_gives_what_its_module_gives() {
    for program in at_each_level(DISPATCH, &[]) {
        program.gives("dispatch", &[i32(0), i32(5)], &[i32(50)]);
        program.gives("dispatch", &[i32(1), i32(5)], &[i32(4)]);
        program.gives("dispatch", &[i32(2), i32(5)], &[i32(24)]);
        program.gives("dispatch", &[i32(-1), i32(-7)], &[i32(0)]);
    }
}

#[test]
fn the_narrow_program_linked_from_an_object_gives_what_its_module_gives() {
    for program in at_each_level(NARROW, &[]) {
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
fn the_frames_program_linked_from_an_object_uses_the_linkers_stack_as_its_module_uses_its_own() {
    // Both have a stack of 64 KiB from the address 0, so the frame of `@probe` is at 65520 in
    // both.
    for program in at_each_level(FRAMES, &[]) {
        program.gives("locals", &[i32(5), i32(1)], &[i32(10)]);
        program.gives("locals", &[i32(5), i32(0)], &[i32(20)]);
        program.gives("repeat", &[i32(100_000)], &[i32(65_520), i32(65_520), i32(1_100_000)]);
        program.gives("probe", &[], &[i32(65_520)]);
    }
}

#[test]
fn the_globals_program_linked_from_an_object_gives_what_its_module_gives_but_its_addresses() {
    for program in at_each_level(GLOBALS, &[]) {
        program.gives("bump_thrice", &[i32(3)], &[i32(5), i32(8), i32(11), i32(14)]);
        program.gives("follow", &[], &[i32(7), i32(8)]);
        program.gives("call_stored", &[i32(21)], &[i32(42)]);
        // The module has no stack, so its data starts at 1024; the linker reserves its stack
        // first and places the data after it.
        let Ok(module) = program.instance(&program.module).call("bump", &[i32(3)]) else {
            panic!("`bump` traps");
        };
        assert_eq!(module, [i32(5), i32(1024)]);
        let Ok(linked) = program.linked_call("bump", &[i32(3)]) else {
            panic!("`bump` traps");
        };
        assert_eq!(linked[0], i32(5));
        let V::I32(address) = linked[1] else {
            panic!("`bump` returns no address");
        };
        assert!(address >= STACK_SIZE as i32 && address % 4 == 0, "{address}");
    }
}

#[test]
fn the_float_program_linked_from_an_object_gives_what_its_module_gives() {
    for program in at_each_level(FLOAT, &[]) {
        program.gives(
            "arith",
            &[f64(1.5), f64(-0.25)],
            &[f64(1.25), f64(1.75), f64(-0.375), f64(-6.0), f64(-1.5)],
        );
        let nan = f32(f32::NAN);
        program.gives("compare", &[nan, f32(1.0)], &[i32(0), i32(0), i32(1), i32(1), i32(0)]);
        program.gives(
            "to_int",
            &[f64(1e10)],
            &[i32(i32::MAX), i32(-1), i64(10_000_000_000), i64(10_000_000_000)],
        );
        program.gives("from_int", &[i32(-1)], &[f64(-1.0), f64(4_294_967_295.0), f32(-1.0)]);
        program.gives("widths", &[f64(0.1)], &[f32(0.1), f64(f64::from(0.1f32))]);
    }
}

#[test]
fn the_traps_program_linked_from_an_object_traps_where_its_module_traps() {
    use wasmi::TrapCode;
    for program in at_each_level(TRAPS, &[]) {
        program.gives("sdiv_i8", &[i32(-7), i32(2)], &[i32(253)]);
        program.traps("sdiv_i8", &[i32(-128), i32(-1)], TrapCode::UnreachableCodeReached);
        program.traps("sdiv_i32", &[i32(i32::MIN), i32(-1)], TrapCode::IntegerOverflow);
        program.traps("sdiv_i64", &[i64(i64::MIN), i64(-1)], TrapCode::IntegerOverflow);
        program.traps("udiv_i32", &[i32(7), i32(0)], TrapCode::IntegerDivisionByZero);
        program.traps("check", &[i32(0)], TrapCode::UnreachableCodeReached);
        program.traps("add_checked", &[i32(i32::MAX), i32(1)], TrapCode::UnreachableCodeReached);
        program.traps("peek", &[i32(-16)], TrapCode::MemoryOutOfBounds);
        program.gives("call_through", &[i32(1), i32(21)], &[i32(42)]);
        program.traps("call_through", &[i32(0), i32(21)], TrapCode::IndirectCallToNull);
        program.traps("call_through", &[i32(2), i32(21)], TrapCode::TableOutOfBounds);
        // `@answer` (42) is at 1024 in the module, which has no stack and keeps all its data.
        // No symbol names it, so the linker collects its segment: the linked module has no data,
        // and its memory is the stack's one page.
        assert_eq!(program.instance(&program.module).call("peek", &[i32(1024)]), Ok(vec![i32(42)]));
        assert!(data_sizes(&program.linked).is_empty());
        assert_eq!(
            program.linked_call("peek", &[i32(STACK_SIZE as i32)]),
            Err(TrapCode::MemoryOutOfBounds)
        );
    }
}

#[test]
fn the_imports_program_linked_from_an_object_imports_what_nothing_defines_from_env() {
    // Its declarations name no host, so the link resolves them; `--allow-undefined` makes them
    // imports from `env`, which the provider's complete module satisfies, as in the module's
    // test.
    for level in LEVELS {
        let mut program = Both::new(IMPORTS, level, &["--allow-undefined"]);
        program.host = Some(("env", compiled(IMPORTS_ENV, level, false)));
        program.gives("scaled", &[i32(10)], &[i32(31)]);
        program.gives("swapped", &[i32(5)], &[i64(505)]);
        let provider = Both::new(IMPORTS_ENV, level, &[]);
        provider.gives("host_scale", &[i32(14)], &[i32(42)]);
        provider.gives("host_swap", &[i32(1), i64(-2)], &[i64(-2), i32(1)]);
    }
}

#[test]
fn the_dispatch_table_program_linked_from_an_object_finds_its_sections_between_their_bounds() {
    for program in at_each_level(DISPATCH_TABLE, &[]) {
        for (id, result) in [(10, 6), (20, 10), (30, -5), (40, 25), (50, -1)] {
            program.gives("lookup", &[i32(id), i64(5)], &[i64(result)]);
        }
        program.gives("count", &[], &[i32(4)]);
        program.gives("count2", &[], &[i32(1)]);
        program.gives("lookup2", &[i64(7)], &[i64(14)]);
        program.gives("term", &[i32(0), i32(3), i64(5)], &[i32(1), i64(8)]);
        program.gives("term", &[i32(1), i32(0), i64(5)], &[i32(0), i64(0)]);
        program.gives("read", &[i32(0)], &[i64(42)]);
        program.gives("read", &[i32(1)], &[i64(-9)]);
    }
}

#[test]
fn the_start_program_linked_from_an_object_runs_its_start_function_from_the_constructors() {
    for level in LEVELS {
        let module = compiled(START, level, false);
        let mut instance = Instance::new(&module);
        instance.gives("_initialize", &[], &[]);
        instance.gives("read", &[i32(1)], &[i32(43)]);
        // The object's start function is an init function, not the export `_initialize`.
        let object = compiled(START, level, true);
        let linked = linked(&[&object], &["--export=__wasm_call_ctors"]);
        assert!(!exports(&linked).iter().any(|name| name == "_initialize"));
        let mut instance = Instance::new(&linked);
        instance.gives("read", &[i32(1)], &[i32(8)]);
        instance.gives("__wasm_call_ctors", &[], &[]);
        instance.gives("read", &[i32(1)], &[i32(43)]);
    }
}

#[test]
fn every_program_of_the_run_suite_is_linked_here() {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/run/wasm");
    let mut files = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("{directory}: {error}"))
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".runic"))
        .collect::<Vec<String>>();
    files.sort();
    assert_eq!(
        files,
        [
            "dispatch-table.runic",
            "dispatch.runic",
            "float.runic",
            "frames.runic",
            "globals.runic",
            "imports-env.runic",
            "imports.runic",
            "machine.runic",
            "narrow.runic",
            "start.runic",
            "sum.runic",
            "traps.runic",
        ],
        "a program of tests/run/wasm has no test here, or one is gone"
    );
}

// ---- Two objects --------------------------------------------------------------------------------

#[test]
fn two_objects_call_each_others_c_functions_with_pointers_and_bytes_and_share_their_globals() {
    let a = object_of(&[module(
        "a",
        &[
            "builtin.global_variable public @a_value : i32 { %v = arith.constant 7 : i32; \
             builtin.yield %v; } attributes { link_name = \"a_value\" };",
            "wasm.global public @a_counter : mut i32 { %v = wasm.const 100 : i32; builtin.yield \
             %v; } attributes { link_name = \"a_counter\" };",
            "builtin.global_variable public @b_value : i32 attributes { link_name = \"b_value\" };",
            "builtin.function public extern(\"C\") @b_add(ptr<i32>, i8) -> i32 attributes { \
             link_name = \"b_add\" };",
            "builtin.function public extern(\"C\") @a_entry(%k: i8) -> i32 {
            %p = builtin.global_symbol @b_value : ptr<i32>;
            %r = builtin.call @b_add(%p, %k) : extern(\"C\") (ptr<i32>, i8) -> i32;
            builtin.ret %r : (i32);
        } attributes { link_name = \"a_entry\", wasm.export_name = \"a_entry\" };",
            "builtin.function public extern(\"C\") @a_scale(%p: ptr<i32>, %k: i8) -> i32 {
            %v = mem.load %p;
            %w = arith.zext %k <{ ty = i32 }>;
            %r = arith.mul %v, %w;
            builtin.ret %r : (i32);
        } attributes { link_name = \"a_scale\" };",
            "builtin.function public extern(\"C\") @a_read() -> (i32, i32) {
            %p = builtin.global_symbol @a_value : ptr<i32>;
            %v = mem.load %p;
            %c = wasm.global_get @a_counter : i32;
            builtin.ret %v, %c : (i32, i32);
        } attributes { wasm.export_name = \"a_read\" };",
        ],
    )]);
    let b = object_of(&[module(
        "b",
        &[
            "builtin.global_variable public @b_value : i32 { %v = arith.constant 5 : i32; \
             builtin.yield %v; } attributes { link_name = \"b_value\" };",
            "builtin.global_variable public @a_value : i32 attributes { link_name = \"a_value\" };",
            "wasm.global public declaration @a_counter : mut i32 attributes { link_name = \
             \"a_counter\" };",
            "builtin.function public extern(\"C\") @a_scale(ptr<i32>, i8) -> i32 attributes { \
             link_name = \"a_scale\" };",
            "builtin.function public extern(\"C\") @b_add(%p: ptr<i32>, %k: i8) -> i32 {
            %v = mem.load %p;
            %w = arith.zext %k <{ ty = i32 }>;
            %s = arith.add %v, %w;
            %q = builtin.global_symbol @a_value : ptr<i32>;
            mem.store %s, %q;
            %c = wasm.global_get @a_counter : i32;
            %c2 = arith.add %c, %w;
            wasm.global_set @a_counter, %c2;
            %r = builtin.call @a_scale(%q, %k) : extern(\"C\") (ptr<i32>, i8) -> i32;
            builtin.ret %r : (i32);
        } attributes { link_name = \"b_add\" };",
        ],
    )]);
    for order in [[&a[..], &b[..]], [&b[..], &a[..]]] {
        let bytes = linked(&order, &[]);
        let mut instance = Instance::new(&bytes);
        instance.gives("a_read", &[], &[i32(7), i32(100)]);
        // `b_value` is 5: B writes 5 + 3 to `a_value` and returns 8 * 3 from A's `a_scale`.
        instance.gives("a_entry", &[i32(3)], &[i32(24)]);
        instance.gives("a_read", &[], &[i32(8), i32(103)]);
        // The byte 200 is unsigned: 5 + 200 is 205, times 200.
        instance.gives("a_entry", &[i32(200)], &[i32(41_000)]);
        instance.gives("a_read", &[], &[i32(205), i32(303)]);
    }
}

#[test]
fn an_address_with_an_addend_reaches_across_objects_from_code_and_from_data() {
    let a = object_of(&[module(
        "a",
        &["builtin.global_variable public @table : array<i32; 4> {
            %a = arith.constant 10 : i32;
            %b = arith.constant 20 : i32;
            %c = arith.constant 30 : i32;
            %d = arith.constant 40 : i32;
            %t = builtin.aggregate %a, %b, %c, %d : array<i32; 4>;
            builtin.yield %t;
        } attributes { link_name = \"table\" };"],
    )]);
    let b = object_of(&[module(
        "b",
        &[
            "builtin.global_variable public @table : array<i32; 4> attributes { link_name = \
             \"table\" };",
            "builtin.global_variable private @third : ptr<i32> {
            %t = builtin.global_symbol @table : ptr<array<i32; 4>>;
            %q = mem.gep %t, [2];
            builtin.yield %q;
        };",
            "builtin.function public extern(\"wasm\") @from_code() -> i32 {
            %a = wasm.data_addr @table + 4;
            %v = wasm.load @__linear_memory, %a : i32;
            builtin.ret %v : (i32);
        } attributes { wasm.export_name = \"from_code\" };",
            "builtin.function public extern(\"C\") @from_data() -> i32 {
            %pp = builtin.global_symbol @third : ptr<ptr<i32>>;
            %p = mem.load %pp;
            %v = mem.load %p;
            builtin.ret %v : (i32);
        } attributes { wasm.export_name = \"from_data\" };",
            "wasm.memory private declaration @__linear_memory : 0 attributes { link_name = \
             \"__linear_memory\" };",
        ],
    )]);
    let bytes = linked(&[&b, &a], &[]);
    let mut instance = Instance::new(&bytes);
    instance.gives("from_code", &[], &[i32(20)]);
    instance.gives("from_data", &[], &[i32(30)]);
}

#[test]
fn a_function_of_one_object_in_the_data_of_the_other_is_called_through_the_table() {
    let a = object_of(&[module(
        "a",
        &["builtin.function public extern(\"C\") @triple(%x: i8) -> i32 {
            %w = arith.zext %x <{ ty = i32 }>;
            %t = arith.constant 3 : i32;
            %r = arith.mul %w, %t;
            builtin.ret %r : (i32);
        } attributes { link_name = \"triple\" };"],
    )]);
    let b = object_of(&[module(
        "b",
        &[
            "builtin.function public extern(\"C\") @triple(i8) -> i32 attributes { link_name = \
             \"triple\" };",
            "builtin.global_variable private @stored : (i8) -> i32 {
            %f = builtin.function_ref @triple : (i8) -> i32;
            builtin.yield %f;
        };",
            "builtin.function public extern(\"C\") @call_stored(%x: i8) -> i32 {
            %a = builtin.global_symbol @stored : ptr<(i8) -> i32>;
            %f = mem.load %a;
            %r = builtin.call_indirect %f(%x) : extern(\"C\") (i8) -> i32;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"call_stored\" };",
        ],
    )]);
    let bytes = linked(&[&a, &b], &[]);
    Instance::new(&bytes).gives("call_stored", &[i32(200)], &[i32(600)]);
}

/// A module `name` that contributes the entries `entries` (each an `i32`) to the section
/// `dispatch`, with nothing else.
fn contributor(name: &str, entries: &[i32]) -> String {
    let items = entries
        .iter()
        .enumerate()
        .map(|(k, value)| {
            format!(
                "builtin.global_variable private @entry_{k} : i32 {{ %v = arith.constant {value} \
                 : i32; builtin.yield %v; }} attributes {{ section = \"dispatch\" }};"
            )
        })
        .collect::<Vec<_>>();
    module(name, &items.iter().map(String::as_str).collect::<Vec<_>>())
}

#[test]
fn objects_contributing_to_one_section_are_found_between_its_bounds_and_kept_unreferenced() {
    let reader = module(
        "reader",
        &[
            "builtin.global_variable public @start : i32 attributes { link_name = \
             \"__start_dispatch\" };",
            "builtin.global_variable public @stop : i32 attributes { link_name = \
             \"__stop_dispatch\" };",
            "builtin.global_variable private @entry : i32 { %v = arith.constant 1 : i32; \
             builtin.yield %v; } attributes { section = \"dispatch\" };",
            "builtin.function public extern(\"C\") @walk() -> (i32, i32) {
            %s = builtin.global_symbol @start : ptr<i32>;
            %e = builtin.global_symbol @stop : ptr<i32>;
            %si = mem.ptrtoint %s;
            %ei = mem.ptrtoint %e;
            %zero = arith.constant 0 : i32;
            cf.br ^head(%si, %zero, %zero : isize, i32, i32);
        ^head(%at: isize, %n: i32, %sum: i32):
            %done = arith.cmpi uge %at, %ei;
            cf.cond_br %done ^exit, ^body : (i1);
        ^body:
            %p = mem.inttoptr %at : ptr<i32>;
            %v = mem.load %p;
            %one = arith.constant 1 : i32;
            %n2 = arith.add %n, %one;
            %sum2 = arith.add %sum, %v;
            %four = arith.constant 4 : isize;
            %next = arith.add %at, %four;
            cf.br ^head(%next, %n2, %sum2 : isize, i32, i32);
        ^exit:
            builtin.ret %n, %sum : (i32, i32);
        } attributes { wasm.export_name = \"walk\" };",
        ],
    );
    let reader = object_of(&[reader]);
    let first = object_of(&[contributor("first", &[10, 20])]);
    let second = object_of(&[contributor("second", &[300])]);
    let bytes = linked(&[&reader, &first, &second], &[]);
    // One entry of the reader's, two of the first object's and one of the second's, although no
    // symbol of either is named by anything: their segments are retained.
    Instance::new(&bytes).gives("walk", &[], &[i32(4), i32(331)]);
    let bytes = linked(&[&reader], &[]);
    Instance::new(&bytes).gives("walk", &[], &[i32(1), i32(1)]);
}

#[test]
fn functions_of_two_results_tail_call_each_other_across_objects() {
    // `(i32, i64) -> (i1, i64)`: the `i1` is carried as an `i32`. Each counts down and calls the
    // other in tail position, a million times in all, which a stack of calls would not survive.
    let side = |name: &str, other: &str| {
        module(
            name,
            &[
                &format!(
                    "builtin.function public extern(\"wasm\") @{other}(i32, i64) -> (i1, i64) \
                     attributes {{ link_name = \"{other}\" }};"
                ),
                &format!(
                    "builtin.function public extern(\"wasm\") @{name}(%n: i32, %acc: i64) -> (i1, \
                     i64) {{
                    %zero = arith.constant 0 : i32;
                    %done = arith.cmpi eq %n, %zero;
                    cf.cond_br %done ^exit, ^again : (i1);
                ^again:
                    %one = arith.constant 1 : i32;
                    %n2 = arith.sub %n, %one;
                    %wide = arith.zext %n <{{ ty = i64 }}>;
                    %acc2 = arith.add %acc, %wide;
                    builtin.tail_call @{other}(%n2, %acc2) : extern(\"wasm\") (i32, i64) -> (i1, \
                     i64);
                ^exit:
                    %flag = arith.constant {} : i1;
                    builtin.ret %flag, %acc : (i1, i64);
                }} attributes {{ link_name = \"{name}\", wasm.export_name = \"{name}\" }};",
                    if name == "ping" { 1 } else { 0 }
                ),
            ],
        )
    };
    let ping = object_of(&[side("ping", "pong")]);
    let pong = object_of(&[side("pong", "ping")]);
    let bytes = linked(&[&ping, &pong], &[]);
    let mut instance = Instance::new(&bytes);
    instance.gives("ping", &[i32(1_000_000), i64(0)], &[i32(1), i64(500_000_500_000)]);
    instance.gives("ping", &[i32(3), i64(0)], &[i32(0), i64(6)]);
    instance.gives("pong", &[i32(3), i64(0)], &[i32(1), i64(6)]);
}

#[test]
fn private_functions_of_one_name_do_not_collide_and_internal_ones_are_shared_but_not_exported() {
    let object = |name: &str, factor: i32| {
        object_of(&[module(
            name,
            &[
                &format!(
                    "builtin.function private extern(\"C\") @helper(%x: i32) -> i32 {{
                    %k = arith.constant {factor} : i32;
                    %r = arith.mul %x, %k;
                    builtin.ret %r : (i32);
                }} attributes {{ link_name = \"helper\" }};"
                ),
                &format!(
                    "builtin.function public extern(\"C\") @{name}(%x: i32) -> i32 {{
                    %r = builtin.call @helper(%x) : extern(\"C\") (i32) -> i32;
                    builtin.ret %r : (i32);
                }} attributes {{ wasm.export_name = \"{name}\" }};"
                ),
            ],
        )])
    };
    let bytes = linked(&[&object("two", 2), &object("three", 3)], &[]);
    let mut instance = Instance::new(&bytes);
    instance.gives("two", &[i32(7)], &[i32(14)]);
    instance.gives("three", &[i32(7)], &[i32(21)]);

    let provider = object_of(&[module(
        "provider",
        &[
            "builtin.function internal extern(\"C\") @shared(%x: i32) -> i32 {
            %one = arith.constant 1 : i32;
            %r = arith.add %x, %one;
            builtin.ret %r : (i32);
        } attributes { link_name = \"shared\" };",
            "builtin.function public extern(\"C\") @named(%x: i32) -> i32 {
            builtin.ret %x : (i32);
        } attributes { wasm.export_name = \"renamed\" };",
            "builtin.function public extern(\"C\") @plain(%x: i32) -> i32 {
            %r = builtin.call @shared(%x) : extern(\"C\") (i32) -> i32;
            builtin.ret %r : (i32);
        } attributes { link_name = \"plain\" };",
        ],
    )]);
    let user = object_of(&[module(
        "user",
        &[
            "builtin.function public extern(\"C\") @shared(i32) -> i32 attributes { link_name = \
             \"shared\" };",
            "builtin.function public extern(\"C\") @use(%x: i32) -> i32 {
            %r = builtin.call @shared(%x) : extern(\"C\") (i32) -> i32;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"use\" };",
        ],
    )]);
    let bytes = linked(&[&provider, &user], &[]);
    assert_eq!(exports(&bytes), ["memory", "renamed", "use"]);
    let mut instance = Instance::new(&bytes);
    instance.gives("use", &[i32(41)], &[i32(42)]);
    instance.gives("renamed", &[i32(5)], &[i32(5)]);
    // `--export-dynamic` exports every symbol other objects may name and that is not hidden: the
    // public `plain` is exported under its link name, and the internal `shared` still is not.
    let bytes = linked(&[&provider, &user], &["--export-dynamic"]);
    let exported = exports(&bytes);
    assert!(exported.iter().any(|name| name == "plain"), "{exported:?}");
    assert!(!exported.iter().any(|name| name == "shared"), "{exported:?}");
    Instance::new(&bytes).gives("plain", &[i32(1)], &[i32(2)]);
}

#[test]
fn a_private_function_with_an_export_name_is_exported_by_the_linker() {
    // Its symbol is hidden rather than local: the linker exports no local symbol.
    let object = object_of(&[module(
        "m",
        &["builtin.function private extern(\"C\") @f(%x: i32) -> i32 {
            %r = arith.add %x, %x;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"f\" };"],
    )]);
    let bytes = linked(&[&object], &[]);
    assert_eq!(exports(&bytes), ["memory", "f"]);
    Instance::new(&bytes).gives("f", &[i32(4)], &[i32(8)]);
}

#[test]
fn an_undefined_symbol_fails_the_link_unless_it_names_its_host() {
    let object = object_of(&[module(
        "m",
        &[
            "builtin.function public extern(\"C\") @missing(i32) -> i32 attributes { link_name = \
             \"missing\" };",
            "builtin.function public extern(\"C\") @f(%x: i32) -> i32 {
            %r = builtin.call @missing(%x) : extern(\"C\") (i32) -> i32;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"f\" };",
        ],
    )]);
    let message = link_allowing_warnings(&[&object], &[]).expect_err("the link fails");
    assert!(message.contains("undefined symbol: missing"), "{message}");

    let object = object_of(&[module(
        "m",
        &[
            "builtin.function public extern(\"C\") @hosted(i32) -> i32 attributes { link_name = \
             \"hosted\", wasm.import_module = \"host\" };",
            "builtin.function public extern(\"C\") @f(%x: i32) -> i32 {
            %r = builtin.call @hosted(%x) : extern(\"C\") (i32) -> i32;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"f\" };",
        ],
    )]);
    let bytes = linked(&[&object], &[]);
    assert_eq!(function_imports(&bytes), [(String::from("host"), String::from("hosted"))]);
    let (_context, host) = super::object::lowered_world(&[module(
        "h",
        &["builtin.function public extern(\"wasm\") @hosted(%x: i32) -> i32 {
                %r = wasm.add %x, %x;
                builtin.ret %r : (i32);
            } attributes { wasm.export_name = \"hosted\" };"],
    )]);
    let host = emit_module(host, &EmitOptions::default())
        .unwrap_or_else(|report| panic!("{}", describe(&report)));
    Instance::with_host(&bytes, Some(("host", &host))).gives("f", &[i32(21)], &[i32(42)]);
}

#[test]
fn zero_data_has_no_bytes_in_the_linked_module_and_reads_zero() {
    let object = object_of(&[module(
        "m",
        &[
            "builtin.global_variable private @zeros : array<i32; 64> { %z = builtin.zero : \
             array<i32; 64>; builtin.yield %z; };",
            "builtin.global_variable private @word : i32 { %v = arith.constant 9 : i32; \
             builtin.yield %v; };",
            "builtin.function public extern(\"C\") @read(%i: isize) -> i32 {
            %z = builtin.global_symbol @zeros : ptr<array<i32; 64>>;
            %p = mem.ptrcast %z : ptr<i32>;
            %e = mem.gep %p, %i, [];
            %v = mem.load %e;
            %w = builtin.global_symbol @word : ptr<i32>;
            %x = mem.load %w;
            %r = arith.add %v, %x;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"read\" };",
        ],
    )]);
    let bytes = linked(&[&object], &[]);
    // Only the word's four bytes are data.
    assert_eq!(data_sizes(&bytes), [4]);
    let mut instance = Instance::new(&bytes);
    instance.gives("read", &[i32(0)], &[i32(9)]);
    instance.gives("read", &[i32(63)], &[i32(9)]);
}

#[test]
fn the_heap_base_and_the_data_end_are_the_linkers_and_compiled_code_reads_them() {
    let object = object_of(&[module(
        "m",
        &[
            "builtin.global_variable public @heap_base : i8 attributes { link_name = \
             \"__heap_base\" };",
            "builtin.global_variable public @data_end : i8 attributes { link_name = \
             \"__data_end\" };",
            "builtin.global_variable private @word : i32 { %v = arith.constant 9 : i32; \
             builtin.yield %v; };",
            "builtin.function public extern(\"C\") @bounds() -> (isize, isize, isize) {
            %h = builtin.global_symbol @heap_base : ptr<i8>;
            %d = builtin.global_symbol @data_end : ptr<i8>;
            %w = builtin.global_symbol @word : ptr<i32>;
            %hi = mem.ptrtoint %h;
            %di = mem.ptrtoint %d;
            %wi = mem.ptrtoint %w;
            builtin.ret %hi, %di, %wi : (isize, isize, isize);
        } attributes { wasm.export_name = \"bounds\" };",
        ],
    )]);
    let bytes = linked(&[&object], &["--export=__heap_base", "--export=__data_end"]);
    let mut instance = Instance::new(&bytes);
    let (heap_base, data_end) = (instance.global("__heap_base"), instance.global("__data_end"));
    // The data, the word alone, follows the stack.
    assert_eq!(data_end, STACK_SIZE as i32 + 4);
    instance.gives("bounds", &[], &[i32(heap_base), i32(data_end), i32(STACK_SIZE as i32)]);
}

/// A module `name` with a start function that appends `digit` to the decimal number in the
/// global `state` (defined by the module `a`, declared by any other), and for `a`, the export
/// `read` of it.
fn starter(name: &str, digit: i32) -> String {
    let state = if name == "a" {
        "builtin.global_variable public @state : i32 { %v = arith.constant 0 : i32; builtin.yield \
         %v; } attributes { link_name = \"state\" };"
    } else {
        "builtin.global_variable public @state : i32 attributes { link_name = \"state\" };"
    };
    let read = if name == "a" {
        "builtin.function public extern(\"C\") @read() -> i32 {
            %p = builtin.global_symbol @state : ptr<i32>;
            %v = mem.load %p;
            builtin.ret %v : (i32);
        } attributes { wasm.export_name = \"read\" };"
    } else {
        ""
    };
    let init = format!(
        "builtin.function private extern(\"C\") @init() {{
            %p = builtin.global_symbol @state : ptr<i32>;
            %v = mem.load %p;
            %ten = arith.constant 10 : i32;
            %shifted = arith.mul %v, %ten;
            %digit = arith.constant {digit} : i32;
            %w = arith.add %shifted, %digit;
            mem.store %w, %p;
            builtin.ret;
        }} attributes {{ wasm.start = #builtin.unit }};"
    );
    module(name, &[state, read, &init])
}

#[test]
fn start_functions_of_two_objects_run_once_from_the_constructors_or_before_every_export() {
    let (a, b) = (object_of(&[starter("a", 1)]), object_of(&[starter("b", 2)]));
    // Exported, `__wasm_call_ctors` runs each object's start function once, in link order.
    let bytes = linked(&[&a, &b], &["--export=__wasm_call_ctors"]);
    let mut instance = Instance::new(&bytes);
    instance.gives("read", &[], &[i32(0)]);
    instance.gives("__wasm_call_ctors", &[], &[]);
    instance.gives("read", &[], &[i32(12)]);
    instance.gives("read", &[], &[i32(12)]);
    let bytes = linked(&[&b, &a], &["--export=__wasm_call_ctors"]);
    let mut instance = Instance::new(&bytes);
    instance.gives("__wasm_call_ctors", &[], &[]);
    instance.gives("read", &[], &[i32(21)]);
    // Without it, the linker wraps every export to run the constructors first, on every call.
    let bytes = linked(&[&a, &b], &[]);
    let mut instance = Instance::new(&bytes);
    instance.gives("read", &[], &[i32(12)]);
    instance.gives("read", &[], &[i32(1212)]);
}

#[test]
fn the_linker_refuses_nothing_the_object_check_lets_through_in_these_tests() {
    // A unit that defines the stack pointer is refused before the linker could report the
    // duplicate symbol.
    let message = try_object_of(&[module(
        "m",
        &[
            "wasm.global public @sp : mut i32 { %v = wasm.const 4096 : i32; builtin.yield %v; } \
             attributes { link_name = \"__stack_pointer\" };",
        ],
    )])
    .expect_err("the object check refuses it");
    assert!(
        message.contains("cannot define the stack pointer `__stack_pointer`"),
        "{message}"
    );
}

// ---- Aliases and sections ---------------------------------------------------------------------

/// A module `a` that defines the internal `target`, `(i8) -> i32` (twice its byte, unsigned, plus
/// 1), with a public alias `al` and an internal alias `inner` of it, the second through the first.
fn aliased() -> Vec<u8> {
    object_of(&[module(
        "a",
        &[
            "builtin.function internal extern(\"C\") @target(%x: i8) -> i32 {
            %w = arith.zext %x <{ ty = i32 }>;
            %d = arith.add %w, %w;
            %one = arith.constant 1 : i32;
            %r = arith.add %d, %one;
            builtin.ret %r : (i32);
        };",
            "builtin.function_alias public @al -> @target attributes { link_name = \"al\" };",
            "builtin.function_alias internal @inner -> @al attributes { link_name = \"inner\" };",
        ],
    )])
}

#[test]
fn another_object_calls_a_function_by_the_name_of_its_alias() {
    let user = object_of(&[module(
        "b",
        &[
            "builtin.function public extern(\"C\") @al(i8) -> i32 attributes { link_name = \"al\" \
             };",
            "builtin.function public extern(\"C\") @inner(i8) -> i32 attributes { link_name = \
             \"inner\" };",
            "builtin.function public extern(\"C\") @use(%x: i8) -> i32 {
            %a = builtin.call @al(%x) : extern(\"C\") (i8) -> i32;
            %b = builtin.call @inner(%x) : extern(\"C\") (i8) -> i32;
            %k = arith.constant 1000 : i32;
            %s = arith.mul %a, %k;
            %r = arith.add %s, %b;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"use\" };",
        ],
    )]);
    let provider = aliased();
    for order in [[&provider[..], &user[..]], [&user[..], &provider[..]]] {
        let bytes = linked(&order, &[]);
        // Neither alias is exported: an alias's symbol is not, and the target has no export name.
        assert_eq!(exports(&bytes), ["memory", "use"]);
        // 200 is the byte 200, unsigned: 401 through each name.
        Instance::new(&bytes).gives("use", &[i32(200)], &[i32(401_401)]);
    }
}

#[test]
fn a_function_value_in_data_named_by_its_alias_is_called_through_the_table() {
    let user = object_of(&[module(
        "b",
        &[
            "builtin.function public extern(\"C\") @al(i8) -> i32 attributes { link_name = \"al\" \
             };",
            "builtin.global_variable private @stored : (i8) -> i32 {
            %f = builtin.function_ref @al : (i8) -> i32;
            builtin.yield %f;
        };",
            "builtin.function public extern(\"C\") @call_stored(%x: i8) -> i32 {
            %a = builtin.global_symbol @stored : ptr<(i8) -> i32>;
            %f = mem.load %a;
            %r = builtin.call_indirect %f(%x) : extern(\"C\") (i8) -> i32;
            builtin.ret %r : (i32);
        } attributes { wasm.export_name = \"call_stored\" };",
        ],
    )]);
    let bytes = linked(&[&aliased(), &user], &[]);
    Instance::new(&bytes).gives("call_stored", &[i32(5)], &[i32(11)]);
}

#[test]
fn members_of_one_section_of_different_alignments_are_each_read_through_their_own_symbol() {
    // In one segment `mixed`: the byte at 0, the `i64` at 8, the `i16` at 16. Each is read by its
    // own symbol, from the object that defines it and from another.
    let owner = object_of(&[module(
        "a",
        &[
            "builtin.global_variable public @small : i8 { %v = arith.constant 7 : i8; \
             builtin.yield %v; } attributes { link_name = \"small\", section = \"mixed\" };",
            "builtin.global_variable public @wide : i64 { %v = arith.constant 81985529216486895 : \
             i64; builtin.yield %v; } attributes { link_name = \"wide\", section = \"mixed\" };",
            "builtin.global_variable public @half : i16 { %v = arith.constant -3 : i16; \
             builtin.yield %v; } attributes { link_name = \"half\", section = \"mixed\" };",
            "builtin.function public extern(\"C\") @own_wide() -> i64 {
            %p = builtin.global_symbol @wide : ptr<i64>;
            %v = mem.load %p;
            builtin.ret %v : (i64);
        } attributes { wasm.export_name = \"own_wide\" };",
        ],
    )]);
    let reader = object_of(&[module(
        "b",
        &[
            "builtin.global_variable public @small : i8 attributes { link_name = \"small\" };",
            "builtin.global_variable public @wide : i64 attributes { link_name = \"wide\" };",
            "builtin.global_variable public @half : i16 attributes { link_name = \"half\" };",
            "builtin.function public extern(\"C\") @read() -> (i32, i64, i32) {
            %ps = builtin.global_symbol @small : ptr<i8>;
            %s = mem.load %ps;
            %sw = arith.zext %s <{ ty = i32 }>;
            %pw = builtin.global_symbol @wide : ptr<i64>;
            %w = mem.load %pw;
            %ph = builtin.global_symbol @half : ptr<i16>;
            %h = mem.load %ph;
            %hw = arith.sext %h <{ ty = i32 }>;
            builtin.ret %sw, %w, %hw : (i32, i64, i32);
        } attributes { wasm.export_name = \"read\" };",
        ],
    )]);
    for order in [[&owner[..], &reader[..]], [&reader[..], &owner[..]]] {
        let bytes = linked(&order, &[]);
        let mut instance = Instance::new(&bytes);
        instance.gives("read", &[], &[i32(7), i64(0x0123_4567_89ab_cdef), i32(-3)]);
        instance.gives("own_wide", &[], &[i64(0x0123_4567_89ab_cdef)]);
    }
}
