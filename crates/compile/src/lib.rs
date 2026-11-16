//! The compiler harness of Runic.
//!
//! A compiler built on Runic supplies frontends, which translate sources to IR, and backends,
//! which lower IR and produce code for a target. The harness is what stands between them and
//! around them: it takes a description of one compilation, decides which stages run and where
//! their outputs go, and runs them.
//!
//! This crate holds the vocabulary of a compilation:
//!
//! * a [Triple] names the target;
//! * a [CheckpointId] names a point at which a compilation holds an artifact, and an [Observer] is
//!   told of each one reached;
//! * a [CompilationRequest] says everything a caller can ask of one compilation.
#![deny(warnings)]

mod checkpoint;
mod request;
mod triple;

pub use self::{
    checkpoint::{
        Artifact, CheckpointDecl, CheckpointId, Flow, IR_CHECKPOINTS, Observer, RecordingObserver,
        Stopped,
    },
    request::{
        CompilationRequest, DebugInfo, EmitRequest, EmitTo, ExtensionOptions, Extern, Input,
        InputSource, IrOptions, OptLevel, OutputFile, RemapPathPrefix, Verification,
    },
    triple::Triple,
};

/// The version of Runic this crate was built from.
pub const RUNIC_VERSION: &str = env!("RUNIC_BUILD_VERSION");

/// The revision of Runic's repository this crate was built from, as `git describe` gives it, or
/// `unknown` when it was built outside a checkout.
pub const RUNIC_REVISION: &str = env!("RUNIC_BUILD_REV");
