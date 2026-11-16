//! The compiler harness of Runic.
//!
//! A compiler built on Runic supplies frontends, which translate sources to IR, and backends,
//! which lower IR and produce code for a target. The harness is what stands between them and
//! around them: it takes a description of one compilation, decides which stages run and where
//! their outputs go, and runs them.
//!
//! What the crate holds:
//!
//! * a [Triple] names the target;
//! * a [CheckpointId] names a point at which a compilation holds an artifact, and an [Observer] is
//!   told of each one reached;
//! * a [CompilationRequest] says everything a caller can ask of one compilation;
//! * a [Route] lists the checkpoints a compilation passes through, in order;
//! * a [Frontend] reads inputs of one language and a [Backend] generates code for targets, and a
//!   [Compiler], built by a [CompilerBuilder] that checks the whole configuration, holds the ones a
//!   compiler offers;
//! * a [Goal] says where a compilation stops and which [Output]s it produces, each with a
//!   [Destination] decided by [resolve_goal] before anything runs;
//! * a [Session] resolves a request against a compiler: which frontend reads each input, the
//!   [Target] and its backend, the route and the goal, and the context and world the stages build
//!   in;
//! * each checkpoint declares how its artifact is rendered, and the outputs a goal names are
//!   rendered and delivered when their checkpoint is reached: a file is written through a
//!   temporary file and a rename, and standard output and memory are kept as bytes in the
//!   session's list of [Emitted] outputs;
//! * the optimization pipeline runs on the world after the frontends: the passes each frontend
//!   adds, then the compiler's optimization (by default the dialect-independent passes of the
//!   optimization level), or the pipeline the request gives instead; [Session::optimize] runs it
//!   and [Session::pass_output] holds what it printed;
//! * the backend of the target lowers the world after that, with passes of its own that a request
//!   cannot replace ([Session::lower]), and then produces its artifacts ([Session::emit_code]),
//!   reaching its checkpoints in order; a compilation that runs to the end stops at the last one;
//! * [Compiler::compile] runs a request through the steps of a session and answers with a
//!   [Compilation]: where it stopped, the artifact there, the outputs delivered, the files read, the
//!   text the passes printed and the diagnostics that were not fatal, whether it succeeded or
//!   failed;
//! * an [IrFrontend] reads Runic's own textual IR: each file holds a module, a component or a
//!   world.
#![deny(warnings)]

mod backend;
mod checkpoint;
mod compilation;
mod compiler;
mod frontend;
mod frontends;
mod goal;
mod output;
mod request;
mod route;
mod session;
mod triple;

#[cfg(test)]
mod testing;

pub use self::{
    backend::Backend,
    checkpoint::{
        Artifact, CheckpointDecl, CheckpointId, Flow, IR_CHECKPOINTS, Observer, RecordingObserver,
        Render, Stopped, render_ir,
    },
    compilation::Compilation,
    compiler::{Compiler, CompilerBuilder},
    frontend::Frontend,
    frontends::IrFrontend,
    goal::{Destination, Goal, Output, OutputNaming, resolve_goal},
    output::{Emitted, write_file},
    request::{
        CompilationRequest, DebugInfo, EmitRequest, EmitTo, ExtensionOptions, Extern, Input,
        InputSource, IrOptions, OptLevel, OutputFile, RemapPathPrefix, Verification,
    },
    route::{Declarer, Route},
    session::{Checkpoints, Session, Target},
    triple::Triple,
};

/// The version of Runic this crate was built from.
pub const RUNIC_VERSION: &str = env!("RUNIC_BUILD_VERSION");

/// The revision of Runic's repository this crate was built from, as `git describe` gives it, or
/// `unknown` when it was built outside a checkout.
pub const RUNIC_REVISION: &str = env!("RUNIC_BUILD_REV");
