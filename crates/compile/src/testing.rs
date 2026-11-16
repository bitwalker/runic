//! Stages for tests: a frontend and a backend whose configuration is given in a line, and that do
//! nothing else.

use runic_core::{DataLayoutSpec, Endianness, PointerSpec, Report};

use crate::{Backend, CheckpointDecl, CheckpointId, Frontend, Triple};

/// A checkpoint declaration, for a test that declares one in a constant.
pub const fn decl(
    id: &'static str,
    aliases: &'static [&'static str],
    extension: &'static str,
) -> CheckpointDecl {
    CheckpointDecl {
        id: CheckpointId::new(id),
        aliases,
        extension,
    }
}

/// A frontend that is only its configuration.
#[derive(Debug, Clone, Copy)]
pub struct TestFrontend {
    /// The name the frontend reports.
    pub name: &'static str,
    /// The extensions the frontend claims.
    pub extensions: &'static [&'static str],
    /// The checkpoints the frontend declares.
    pub checkpoints: &'static [CheckpointDecl],
}

impl TestFrontend {
    /// A frontend with a name and extensions, and no checkpoint.
    pub const fn new(name: &'static str, extensions: &'static [&'static str]) -> Self {
        Self {
            name,
            extensions,
            checkpoints: &[],
        }
    }

    /// The same frontend, declaring `checkpoints`.
    pub const fn with_checkpoints(self, checkpoints: &'static [CheckpointDecl]) -> Self {
        Self {
            checkpoints,
            ..self
        }
    }
}

impl Frontend for TestFrontend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn extensions(&self) -> &'static [&'static str] {
        self.extensions
    }

    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        self.checkpoints
    }
}

/// A backend that is only its configuration: it supports exactly its triples, on a little-endian
/// 32-bit layout.
#[derive(Debug, Clone, Copy)]
pub struct TestBackend {
    /// The name the backend reports.
    pub name: &'static str,
    /// The triples the backend supports.
    pub triples: &'static [&'static str],
    /// The checkpoints the backend declares.
    pub checkpoints: &'static [CheckpointDecl],
}

impl TestBackend {
    /// A backend with a name, triples and checkpoints.
    pub const fn new(
        name: &'static str,
        triples: &'static [&'static str],
        checkpoints: &'static [CheckpointDecl],
    ) -> Self {
        Self {
            name,
            triples,
            checkpoints,
        }
    }
}

impl Backend for TestBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supports(&self, triple: &Triple) -> bool {
        self.triples.contains(&triple.as_str())
    }

    fn targets(&self) -> Vec<Triple> {
        self.triples
            .iter()
            .map(|t| t.parse().expect("a test triple is well formed"))
            .collect()
    }

    fn data_layout(&self, _triple: &Triple) -> Result<DataLayoutSpec, Report> {
        Ok(DataLayoutSpec::new(Endianness::Little, PointerSpec::sized(32).unwrap()).unwrap())
    }

    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        self.checkpoints
    }
}
