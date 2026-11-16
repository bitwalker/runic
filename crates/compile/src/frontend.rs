use crate::CheckpointDecl;

/// A source language a compiler reads: it says what it is called, which file extensions are its
/// inputs, and which checkpoints it reaches.
///
/// A compiler registers each frontend once, and that one registered stage serves every session of
/// the compiler, so a frontend holds no state of one compilation. The harness reads the name to
/// match `--language`, the extensions to find the frontend of an input that names no language,
/// and the checkpoints to build the route of a compilation.
pub trait Frontend {
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
}
