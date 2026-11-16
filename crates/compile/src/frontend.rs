use runic_core::{Report, dialects::builtin::WorldRef};

use crate::{CheckpointDecl, Checkpoints, Flow, Input, Session};

/// A source language a compiler reads: it says what it is called, which file extensions are its
/// inputs, and which checkpoints it reaches.
///
/// A compiler registers each frontend once, and that one registered stage serves every session of
/// the compiler, on whatever thread that session runs, so a frontend holds no state of one
/// compilation and is `Send + Sync`. The harness reads the name to
/// match `--language`, the extensions to find the frontend of an input that names no language,
/// and the checkpoints to build the route of a compilation.
pub trait Frontend: Send + Sync {
    /// The name of the frontend: what `--language` takes, and the prefix of each checkpoint id
    /// the frontend declares.
    fn name(&self) -> &'static str;

    /// The extensions, without the dot, of the files this frontend reads. A file whose extension
    /// is listed here and which names no language goes to this frontend.
    fn extensions(&self) -> &'static [&'static str];

    /// The checkpoints this frontend reaches, in the order it reaches them. A frontend that reaches
    /// none declares none.
    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        &[]
    }

    /// Translates `inputs`, all of this frontend's language, into IR in `world`.
    ///
    /// The session says what is being compiled: the world is the session's, and states the data
    /// layout of the target when there is one. The frontend reads its inputs through the session,
    /// which records the files it reads, adds to `world` what the inputs define, and reaches its
    /// own checkpoints through `checkpoints`. When `checkpoints` answers that the compilation
    /// stops, the frontend returns that answer at once as `Ok(Flow::Break(..))`; otherwise it
    /// returns `Ok(Flow::Continue(()))`. An error ends the compilation and is returned to the
    /// caller as it is.
    ///
    /// What the frontend leaves in `world` is valid IR: the harness verifies it after the last
    /// frontend returns, unless the request turns verification off.
    fn translate(
        &self,
        session: &Session<'_>,
        inputs: &[Input],
        world: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report>;
}
