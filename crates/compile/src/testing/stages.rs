//! Stages for unit tests: a frontend and a backend whose configuration is given in a line, and that do
//! nothing else.

use core::fmt;

use runic_core::{
    DataLayoutSpec, Endianness, EntityMut, Ident, Operation, OperationName, PointerSpec, Report,
    Visibility,
    dialects::builtin::{
        FunctionBuilder, ModuleBuilder, World, WorldBuilder, WorldRef, attributes::Signature,
    },
    interner::Symbol,
    pass::{
        OpPassManager, Pass, PassExecutionState, PassInfo, PassOptions, registry::RegistryEntry,
        write_pass_option,
    },
};

use crate::{
    Artifact, Backend, CheckpointDecl, CheckpointId, Checkpoints, Flow, Frontend, Input, Session,
    Triple,
};

/// What a test frontend does when it translates.
pub type TranslateFn =
    fn(&Session<'_>, &[Input], WorldRef, &mut Checkpoints<'_>) -> Result<Flow<()>, Report>;

/// What a test frontend does when it adds passes.
pub type AddPassesFn = fn(&Session<'_>, &mut OpPassManager) -> Result<(), Report>;

fn add_nothing(_session: &Session<'_>, _passes: &mut OpPassManager) -> Result<(), Report> {
    Ok(())
}

/// What a test backend does when it adds passes.
pub type BackendAddPassesFn = AddPassesFn;

/// What a test backend does when it emits.
pub type EmitFn = fn(&Session<'_>, WorldRef, &mut Checkpoints<'_>) -> Result<Flow<()>, Report>;

/// Reaches each checkpoint the backend declares, in order, with a `String` that names it, and
/// returns the first `Break`.
fn emit_each_checkpoint(
    session: &Session<'_>,
    _world: WorldRef,
    checkpoints: &mut Checkpoints<'_>,
) -> Result<Flow<()>, Report> {
    let backend = session.target().expect("a backend emits for a target").backend();
    for decl in backend.checkpoints() {
        if let Flow::Break(stopped) = checkpoints.reach(decl.id, decl.id.as_str().to_owned())? {
            return Ok(Flow::Break(stopped));
        }
    }
    Ok(Flow::Continue(()))
}

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
    /// What the frontend does when it adds passes: nothing, unless a test says otherwise.
    pub add_passes: AddPassesFn,
}

impl TestFrontend {
    /// A frontend with a name and extensions, and no checkpoint.
    pub const fn new(name: &'static str, extensions: &'static [&'static str]) -> Self {
        Self {
            name,
            extensions,
            checkpoints: &[],
            translate: translate_nothing,
            add_passes: add_nothing,
        }
    }

    /// The same frontend, doing `translate` when it translates.
    pub const fn with_translate(self, translate: TranslateFn) -> Self {
        Self { translate, ..self }
    }

    /// The same frontend, doing `add_passes` when it adds passes.
    pub const fn with_add_passes(self, add_passes: AddPassesFn) -> Self {
        Self { add_passes, ..self }
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

    fn add_passes(&self, session: &Session<'_>, passes: &mut OpPassManager) -> Result<(), Report> {
        (self.add_passes)(session, passes)
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
    /// What the backend does when it adds its lowering passes: nothing, unless a test says
    /// otherwise.
    pub add_passes: BackendAddPassesFn,
    /// What the backend does when it emits: reaches each checkpoint it declares, in order, with a
    /// `String` naming it, unless a test says otherwise.
    pub emit: EmitFn,
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
            add_passes: add_nothing,
            emit: emit_each_checkpoint,
        }
    }

    /// The same backend, doing `add_passes` when it adds its lowering passes.
    pub const fn with_add_passes(self, add_passes: BackendAddPassesFn) -> Self {
        Self { add_passes, ..self }
    }

    /// The same backend, doing `emit` when it emits.
    pub const fn with_emit(self, emit: EmitFn) -> Self {
        Self { emit, ..self }
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

    fn add_passes(&self, session: &Session<'_>, passes: &mut OpPassManager) -> Result<(), Report> {
        (self.add_passes)(session, passes)
    }

    fn emit(
        &self,
        session: &Session<'_>,
        world: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        (self.emit)(session, world, checkpoints)
    }
}

/// Adds the pass `test-mark` with the option `name=<name>` to `passes`.
pub fn add_mark(passes: &mut OpPassManager, name: &str) -> Result<(), Report> {
    let mut options = PassOptions::new();
    options.push("name", Some(name));
    let info = PassInfo::lookup(Mark::NAME).expect("test-mark is registered");
    info.add_to_pipeline(passes, &options)
}

/// Adds the pass `test-fail` to `passes`.
pub fn add_fail(passes: &mut OpPassManager) -> Result<(), Report> {
    let info = PassInfo::lookup(Fail::NAME).expect("test-fail is registered");
    info.add_to_pipeline(passes, &PassOptions::new())
}

/// `test-mark`: adds an empty `builtin.module` named by its option `name` ("mark" when none is
/// given) to the world it runs on, so that the order of the modules is the order the passes ran.
#[derive(Default)]
struct Mark {
    name: Option<String>,
}

impl Mark {
    const NAME: &'static str = "test-mark";
}

runic_core::inventory::submit!(PassInfo::new::<Mark>(
    "test-mark",
    "Adds an empty module named by its option `name` to the world"
));

impl Pass for Mark {
    type Target = Operation;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn argument(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> &'static str {
        "Adds an empty module named by its option `name` to the world"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn initialize_options(&mut self, options: &PassOptions) -> Result<(), Report> {
        options.check_keys(Self::NAME, &["name"])?;
        if let Some(value) = options.get("name") {
            let Some(name) = value else {
                return Err(Report::msg("option 'name' of pass 'test-mark' requires a value"));
            };
            self.name = Some(name.to_owned());
        }
        Ok(())
    }

    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(Self::NAME)?;
        if let Some(name) = &self.name {
            f.write_str("{")?;
            write_pass_option(f, "name", Some(name))?;
            f.write_str("}")?;
        }
        Ok(())
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Operation>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let operation = op.as_operation_ref();
        drop(op);
        let world = operation
            .try_downcast_op::<World>()
            .map_err(|_| Report::msg("test-mark runs on the world only"))?;
        let name = self.name.clone().unwrap_or_else(|| "mark".to_owned());
        WorldBuilder::new(world)
            .declare_module(Ident::with_empty_span(Symbol::intern(name.as_str())))?;
        Ok(())
    }
}

/// `test-fail`: fails, with the message `test-fail`.
#[derive(Default)]
struct Fail;

impl Fail {
    const NAME: &'static str = "test-fail";
}

runic_core::inventory::submit!(PassInfo::new::<Fail>("test-fail", "Fails"));

impl Pass for Fail {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "test-fail"
    }

    fn argument(&self) -> &'static str {
        "test-fail"
    }

    fn description(&self) -> &'static str {
        "Fails"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Operation>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        Err(Report::msg("test-fail"))
    }
}

/// `test-break`: adds a module `broken` with a function that has no terminator to the world it
/// runs on, so that the world no longer verifies.
#[derive(Default)]
struct Break;

runic_core::inventory::submit!(PassInfo::new::<Break>("test-break", "Leaves the world invalid"));

impl Pass for Break {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "test-break"
    }

    fn argument(&self) -> &'static str {
        "test-break"
    }

    fn description(&self) -> &'static str {
        "Leaves the world invalid"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Operation>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let operation = op.as_operation_ref();
        let context = op.context_rc();
        drop(op);
        let world = operation
            .try_downcast_op::<World>()
            .map_err(|_| Report::msg("test-break runs on the world only"))?;
        let module =
            WorldBuilder::new(world).declare_module(Ident::with_empty_span("broken".into()))?;
        let mut builder = ModuleBuilder::new(module);
        let signature = Signature::new(&context, [], []);
        let function = builder.define_function(
            Ident::with_empty_span("f".into()),
            Visibility::Public,
            signature,
        )?;
        let _ = FunctionBuilder::new(function, builder.builder());
        Ok(())
    }
}
