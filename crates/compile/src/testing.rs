//! Stages for tests: a frontend and a backend whose configuration is given in a line, and that do
//! nothing else.

use runic_core::{DataLayoutSpec, Endianness, PointerSpec, Report, dialects::builtin::WorldRef};

use crate::{
    Artifact, Backend, CheckpointDecl, CheckpointId, Checkpoints, Flow, Frontend, Input, Session,
    Triple,
};

/// What a test frontend does when it translates.
pub type TranslateFn =
    fn(&Session<'_>, &[Input], WorldRef, &mut Checkpoints<'_>) -> Result<Flow<()>, Report>;

fn translate_nothing(
    _session: &Session<'_>,
    _inputs: &[Input],
    _world: WorldRef,
    _checkpoints: &mut Checkpoints<'_>,
) -> Result<Flow<()>, Report> {
    Ok(Flow::Continue(()))
}

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
        render: render_text,
    }
}

/// A renderer for tests: it writes the artifact's value when that is a `String` or a
/// `&'static str`, and the id of the checkpoint otherwise.
pub fn render_text(
    artifact: &Artifact,
    _session: &Session<'_>,
    out: &mut dyn std::io::Write,
) -> Result<(), Report> {
    let written = if let Some(text) = artifact.downcast_ref::<String>() {
        out.write_all(text.as_bytes())
    } else if let Some(text) = artifact.downcast_ref::<&'static str>() {
        out.write_all(text.as_bytes())
    } else {
        out.write_all(artifact.checkpoint().as_str().as_bytes())
    };
    written.map_err(|error| Report::msg(format!("could not render: {error}")))
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
    /// What the frontend does when it translates: nothing, unless a test says otherwise.
    pub translate: TranslateFn,
}

impl TestFrontend {
    /// A frontend with a name and extensions, and no checkpoint.
    pub const fn new(name: &'static str, extensions: &'static [&'static str]) -> Self {
        Self {
            name,
            extensions,
            checkpoints: &[],
            translate: translate_nothing,
        }
    }

    /// The same frontend, doing `translate` when it translates.
    pub const fn with_translate(self, translate: TranslateFn) -> Self {
        Self { translate, ..self }
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

    fn translate(
        &self,
        session: &Session<'_>,
        inputs: &[Input],
        world: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        (self.translate)(session, inputs, world, checkpoints)
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
