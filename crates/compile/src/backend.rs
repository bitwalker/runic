use runic_core::{DataLayoutSpec, Report};

use crate::{CheckpointDecl, Triple};

/// A code generator a compiler offers: it says what it is called, which targets it supports, how
/// data is laid out on each, and which checkpoints it reaches.
///
/// A compiler registers each backend once, and that one registered stage serves every session of
/// the compiler, on whatever thread that session runs, so a backend holds no state of one
/// compilation and is `Send + Sync`. The harness reads the name to
/// match `--backend`, `supports` and `targets` to choose the backend for `--target`, the data
/// layout to give the frontends before they run, and the checkpoints to build the route of a
/// compilation.
pub trait Backend: Send + Sync {
    /// The name of the backend: what `--backend` takes, and the prefix of each checkpoint id the
    /// backend declares.
    fn name(&self) -> &'static str;

    /// Whether this backend generates code for `triple`.
    fn supports(&self, triple: &Triple) -> bool;

    /// The triples this backend is known to support, for a message that lists the choices. Every
    /// triple listed is one `supports` accepts.
    fn targets(&self) -> Vec<Triple>;

    /// The data layout of `triple`, which this backend supports.
    fn data_layout(&self, triple: &Triple) -> Result<DataLayoutSpec, Report>;

    /// The checkpoints this backend reaches, in the order it reaches them. A backend reaches at
    /// least one: the artifact it produces.
    fn checkpoints(&self) -> &'static [CheckpointDecl];
}
