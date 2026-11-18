//! The backend `wasm` of the compile harness, for the target `wasm32-unknown-unknown`.
//!
//! [WasmBackend] lowers the world to Wasm form, cleans the lowered code up when the request
//! optimizes, and emits one complete core WebAssembly module with [emit_module]. The module is
//! delivered at two checkpoints, in this order: [WAT] (`wasm.wat`, alias `wat`), its text
//! rendered by [crate::text::print], and [MODULE] (`wasm.module`, alias `module`),
//! the binary itself in a `.core.wasm` file. Both hold the same bytes, emitted once.
//!
//! The checkpoint [OBJECT] (`wasm.object`, alias `object`) comes before them: a relocatable
//! object for `wasm-ld`, written with [emit_object] only when the request stops at it or asks for
//! an output there.

use std::{boxed::Box, io, string::String, vec, vec::Vec};

use runic_compile::{
    Artifact, Backend, CheckpointDecl, CheckpointId, Checkpoints, DebugInfo, Flow, OptLevel,
    OptionsDecl, Render, Session, Triple,
};
use runic_ir::{
    Align, DataLayoutSpec, Endianness, FunctionSpec, PointerSpec, Report,
    dialects::builtin::{WorldRef, types::FloatType},
    pass::OpPassManager,
};
use runic_transform::{CommonSubexpressionElimination, DeadCodeElimination};

use crate::{
    LOWERING_PASSES, add_lowering_pass,
    emit::{EmitOptions, emit_module, emit_object},
    text, verify_form,
};

/// The one target of the backend: 32-bit WebAssembly, with no operating system.
pub const TRIPLE: &str = "wasm32-unknown-unknown";

/// The checkpoint of the relocatable object, alias `object`, in files with the extension `o`:
/// first on the backend's route. The object is produced only when the request stops here or asks
/// for an output here.
pub const OBJECT: CheckpointId = CheckpointId::new("wasm.object");

/// The checkpoint of the module's text, alias `wat`, in files with the extension `wat`.
pub const WAT: CheckpointId = CheckpointId::new("wasm.wat");

/// The checkpoint of the binary module, alias `module`, in files with the extension `core.wasm`:
/// the end of the backend's route, and so the default stop point.
pub const MODULE: CheckpointId = CheckpointId::new("wasm.module");

/// The size of the shadow stack when `--wasm-stack-size` is not given: 1 MiB, what rustc asks
/// `wasm-ld` for on `wasm32-unknown-unknown`.
pub const DEFAULT_STACK_SIZE: u32 = 1024 * 1024;

const CHECKPOINTS: &[CheckpointDecl] = &[
    CheckpointDecl {
        id: OBJECT,
        aliases: &["object"],
        extension: "o",
        render: render_object as Render,
    },
    CheckpointDecl {
        id: WAT,
        aliases: &["wat"],
        extension: "wat",
        render: render_wat as Render,
    },
    CheckpointDecl {
        id: MODULE,
        aliases: &["module"],
        extension: "core.wasm",
        render: render_module as Render,
    },
];

/// What the options of the backend `wasm` say. The driver prefixes each flag with the backend's
/// name: `stack_size` is `--wasm-stack-size`.
#[derive(clap::Args, Debug, Clone, PartialEq, Eq)]
pub struct WasmOptions {
    /// The size in bytes of the shadow stack, which takes the start of linear memory and grows
    /// down from the initial value of the stack pointer `__stack_pointer`; a positive multiple of
    /// 16. A relocatable object (`--stop-after object`) accepts the option and ignores it: the
    /// linker sets the stack.
    #[arg(long, value_name = "BYTES", default_value_t = DEFAULT_STACK_SIZE)]
    pub stack_size: u32,
}

impl Default for WasmOptions {
    /// A stack of [DEFAULT_STACK_SIZE] bytes.
    fn default() -> Self {
        Self {
            stack_size: DEFAULT_STACK_SIZE,
        }
    }
}

/// The WebAssembly backend, named `wasm`, for the target `wasm32-unknown-unknown`.
///
/// - Its data layout is that of the C ABI of wasm32 (the tool conventions' `BasicCABI.md`):
///   little-endian, pointers and `isize` of 32 bits aligned to 4, `i8`, `i16`, `i32`, `i64` and
///   `i128` aligned to 1, 2, 4, 8 and 16 bytes, `f32` to 4 and `f64` to 8, and a function value
///   (`fn = 32`), its index in the indirect function table, of 32 bits aligned to 4.
/// - It lowers the world with `cfg-to-scf{exits=in-place}`, `wasm-lower-frames` and
///   `convert-to-wasm`; then, when the request's optimization level is above none, `cse` and
///   `dce`, which remove what the lowering leaves dead or repeated; and last `wasm-verify-form`.
/// - It emits every module of the world as one complete core module ([emit_module]), validated,
///   with the stack size of [WasmOptions], a `name` section unless the request asks for no debug
///   information, and the unit's name as the module's name. It then reaches [WAT] and [MODULE]
///   with those bytes. What the emitter cannot emit fails the compilation with its diagnostic.
/// - When the request stops at [OBJECT] or asks for an output there, it first reaches [OBJECT]
///   with a relocatable object ([emit_object]), collected from the world on its own, so that
///   a request for the module alone is never refused for what an object cannot express. A request
///   that stops at [OBJECT] runs no further.
#[derive(Debug, Default, Clone, Copy)]
pub struct WasmBackend;

impl Backend for WasmBackend {
    fn name(&self) -> &'static str {
        "wasm"
    }

    fn supports(&self, triple: &Triple) -> bool {
        triple.as_str() == TRIPLE
    }

    fn targets(&self) -> Vec<Triple> {
        vec![TRIPLE.parse().expect("the triple of the Wasm backend is well formed")]
    }

    fn data_layout(&self, _triple: &Triple) -> Result<DataLayoutSpec, Report> {
        data_layout()
    }

    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        CHECKPOINTS
    }

    fn options(&self) -> Option<OptionsDecl> {
        Some(OptionsDecl::of::<WasmOptions>())
    }

    fn add_passes(&self, session: &Session<'_>, passes: &mut OpPassManager) -> Result<(), Report> {
        let clean_up = session.request().opt_level != OptLevel::None;
        for (name, options) in LOWERING_PASSES {
            if name == verify_form::NAME && clean_up {
                passes.add_pass(Box::<CommonSubexpressionElimination>::default())?;
                passes.add_pass(Box::<DeadCodeElimination>::default())?;
            }
            add_lowering_pass(passes, name, options)?;
        }
        Ok(())
    }

    fn emit(
        &self,
        session: &Session<'_>,
        world: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        let options = EmitOptions {
            stack_size: session
                .options::<WasmOptions>()
                .map_or(DEFAULT_STACK_SIZE, |options| options.stack_size),
            names: session.request().debug != DebugInfo::None,
            module_name: Some(String::from(session.name())),
        };
        let root = world.as_operation_ref();
        let goal = session.goal();
        if goal.stop_after() == OBJECT || goal.output_at(OBJECT).is_some() {
            // The object is written from a unit of its own: a unit collected for it is checked
            // for it, and laying a unit out for a module changes it, so the two do not share one.
            let object = emit_object(root, &options)?;
            if let Flow::Break(stopped) = checkpoints.reach(OBJECT, object)? {
                return Ok(Flow::Break(stopped));
            }
        }
        let bytes = emit_module(root, &options)?;
        let bytes = match checkpoints.reach(WAT, bytes)? {
            Flow::Continue(bytes) => bytes,
            Flow::Break(stopped) => return Ok(Flow::Break(stopped)),
        };
        Ok(match checkpoints.reach(MODULE, bytes)? {
            Flow::Continue(_) => Flow::Continue(()),
            Flow::Break(stopped) => Flow::Break(stopped),
        })
    }
}

/// The data layout of the C ABI of wasm32 (see [WasmBackend]).
fn data_layout() -> Result<DataLayoutSpec, Report> {
    let invalid = |error| Report::msg(std::format!("the data layout is not valid: {error}"));
    let align = |bytes| Align::from_bytes(bytes).expect("a power of two is an alignment");
    let pointer = PointerSpec::sized(32).expect("32 bits is a pointer size");
    let function = FunctionSpec::sized(32).expect("32 bits is a function value's size");
    let mut spec = DataLayoutSpec::new(Endianness::Little, pointer)
        .and_then(|spec| spec.with_function(function))
        .map_err(invalid)?;
    for (width, bytes) in [(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)] {
        spec = spec.with_integer_align(width, align(bytes)).map_err(invalid)?;
    }
    Ok(spec
        .with_float_align(FloatType::F32, align(4))
        .with_float_align(FloatType::F64, align(8)))
}

/// The bytes of the module at `artifact`.
fn module_bytes(artifact: &Artifact) -> Result<&[u8], Report> {
    artifact.downcast_ref::<Vec<u8>>().map(Vec::as_slice).ok_or_else(|| {
        Report::msg(std::format!(
            "the artifact at checkpoint '{}' is not the bytes of a WebAssembly module",
            artifact.checkpoint()
        ))
    })
}

/// Writes the object at `artifact` to `out`, as it is.
fn render_object(
    artifact: &Artifact,
    _session: &Session<'_>,
    out: &mut dyn io::Write,
) -> Result<(), Report> {
    write(artifact, out, module_bytes(artifact)?)
}

/// Writes the text of the module at `artifact` to `out`.
fn render_wat(
    artifact: &Artifact,
    _session: &Session<'_>,
    out: &mut dyn io::Write,
) -> Result<(), Report> {
    let text = text::print(module_bytes(artifact)?)?;
    write(artifact, out, text.as_bytes())
}

/// Writes the module at `artifact` to `out`, as it is.
fn render_module(
    artifact: &Artifact,
    _session: &Session<'_>,
    out: &mut dyn io::Write,
) -> Result<(), Report> {
    write(artifact, out, module_bytes(artifact)?)
}

/// Writes `bytes`, the rendering of `artifact`, to `out`.
fn write(artifact: &Artifact, out: &mut dyn io::Write, bytes: &[u8]) -> Result<(), Report> {
    out.write_all(bytes).map_err(|error| {
        Report::msg(std::format!(
            "could not render checkpoint '{}': {error}",
            artifact.checkpoint()
        ))
    })
}
