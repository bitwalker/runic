use alloc::{
    boxed::Box,
    collections::{BTreeMap, btree_map::Entry},
    format,
    vec::Vec,
};
use core::any::TypeId;

use runic_symbol::sync::{LazyLock, RwLock};

use super::*;
use crate::Report;

static PASS_REGISTRY: LazyLock<PassRegistry> = LazyLock::new(PassRegistry::new);

/// A global, thread-safe pass and pass pipeline registry
///
/// You should generally _not_ need to work with this directly.
pub struct PassRegistry {
    passes: RwLock<BTreeMap<&'static str, PassRegistryEntry>>,
    pipelines: RwLock<BTreeMap<&'static str, PassRegistryEntry>>,
}
impl Default for PassRegistry {
    fn default() -> Self {
        Self::new()
    }
}
impl PassRegistry {
    /// Create a new [PassRegistry] instance from the passes and pipelines submitted with
    /// `inventory`.
    ///
    /// # Panics
    ///
    /// Panics if two submissions share one name and are not the same pass; the message names both
    /// types. A name taken twice is a defect in the linked crates, and the registry has no caller
    /// to return it to, so the first use of the registry reports it.
    pub fn new() -> Self {
        match Self::from_entries(
            inventory::iter::<PassInfo>().map(|pass| pass.0.clone()),
            inventory::iter::<PassPipelineInfo>().map(|pipeline| pipeline.0.clone()),
        ) {
            Ok(registry) => registry,
            Err(error) => panic!("{error}"),
        }
    }

    /// Build a registry from the given passes and pipelines, or report the first name that two of
    /// them share.
    fn from_entries(
        passes: impl IntoIterator<Item = PassRegistryEntry>,
        pipelines: impl IntoIterator<Item = PassRegistryEntry>,
    ) -> Result<Self, Report> {
        let mut registered_passes = BTreeMap::default();
        for pass in passes {
            insert("pass", &mut registered_passes, pass)?;
        }
        let mut registered_pipelines = BTreeMap::default();
        for pipeline in pipelines {
            insert("pass pipeline", &mut registered_pipelines, pipeline)?;
        }
        Ok(Self {
            passes: RwLock::new(registered_passes),
            pipelines: RwLock::new(registered_pipelines),
        })
    }

    /// Get the pass information for the pass whose argument name is `name`
    pub fn get_pass(&self, name: &str) -> Option<PassInfo> {
        self.passes.read().get(name).cloned().map(PassInfo)
    }

    /// Get the pass pipeline information for the pipeline whose argument name is `name`
    pub fn get_pipeline(&self, name: &str) -> Option<PassPipelineInfo> {
        self.pipelines.read().get(name).cloned().map(PassPipelineInfo)
    }

    /// Every registered pass, sorted by its argument name.
    pub fn passes(&self) -> Vec<PassInfo> {
        self.passes.read().values().cloned().map(PassInfo).collect()
    }

    /// Every registered pass pipeline, sorted by its argument name.
    pub fn pipelines(&self) -> Vec<PassPipelineInfo> {
        self.pipelines.read().values().cloned().map(PassPipelineInfo).collect()
    }

    /// Register the given pass.
    ///
    /// Registering the same pass type again does nothing; a name that a pass of another type has
    /// taken is an error.
    pub fn register_pass(&self, info: PassInfo) -> Result<(), Report> {
        insert("pass", &mut self.passes.write(), info.0.dynamic())
    }

    /// Register the given pass pipeline.
    ///
    /// A name that is already taken is an error: a pipeline has no type to tell two registrations
    /// of one pipeline from two pipelines.
    pub fn register_pipeline(&self, info: PassPipelineInfo) -> Result<(), Report> {
        insert("pass pipeline", &mut self.pipelines.write(), info.0.dynamic())
    }
}

/// Inserts `entry`. An entry already there under the same name that is the same pass or pipeline
/// is left alone; one that is another is reported.
fn insert(
    kind: &str,
    entries: &mut BTreeMap<&'static str, PassRegistryEntry>,
    entry: PassRegistryEntry,
) -> Result<(), Report> {
    match entries.entry(entry.arg) {
        Entry::Vacant(slot) => {
            slot.insert(entry);
            Ok(())
        }
        Entry::Occupied(taken) if taken.get().is_same_as(&entry) => Ok(()),
        Entry::Occupied(taken) => Err(duplicate(kind, taken.get(), &entry)),
    }
}

fn duplicate(kind: &str, first: &PassRegistryEntry, second: &PassRegistryEntry) -> Report {
    Report::msg(format!(
        "cannot register {kind} '{}' for '{}': the name is already registered for '{}'",
        first.arg, second.type_name, first.type_name
    ))
}

inventory::collect!(PassInfo);
inventory::collect!(PassPipelineInfo);

/// A type alias for the closure type for registering a pass with a pass manager
pub type PassRegistryFunction = fn(&mut OpPassManager, &PassOptions) -> Result<(), Report>;

/// A type alias for the closure type used for type-erased pass constructors
pub type PassAllocatorFunction = fn() -> Box<dyn OperationPass>;

/// A [RegistryEntry] is a registered pass or pass pipeline.
///
/// This trait provides the common functionality shared by both passes and pipelines.
pub trait RegistryEntry {
    /// Returns the command-line option that may be passed to `runic` that will cause this pass
    /// or pass pipeline to run.
    fn argument(&self) -> &'static str;
    /// Return a description for the pass or pass pipeline.
    fn description(&self) -> &'static str;
    /// Adds this entry to the given pass manager.
    ///
    /// `options` are given to the pass as they are (see [`super::Pass::initialize_options`]).
    ///
    /// Returns `Err` if an error occurred parsing the given options.
    fn add_to_pipeline(&self, pm: &mut OpPassManager, options: &PassOptions) -> Result<(), Report>;
}

/// Information about a pass or pass pipeline in the pass registry
#[derive(Clone)]
struct PassRegistryEntry {
    /// The name of the compiler option for referencing on the command line
    arg: &'static str,
    /// A description of the pass or pass pipeline
    description: &'static str,
    /// The type id of the concrete pass type
    type_id: Option<TypeId>,
    /// The name of the concrete pass type, for error messages
    type_name: &'static str,
    /// Function that registers this entry with a pass manager pipeline
    builder: PassAllocatorFunction,
    /// Whether this entry was registered at run time (`register_pass`, `register_pass_pipeline`),
    /// rather than submitted with `inventory`
    dynamic: bool,
}
impl PassRegistryEntry {
    /// Marks this entry as registered at run time.
    fn dynamic(mut self) -> Self {
        self.dynamic = true;
        self
    }

    /// Whether `self` and `other` register the same pass or pipeline.
    ///
    /// Two passes are the same when their `TypeId`s are equal. Two entries that were both submitted
    /// with `inventory` are also the same when they name one type path (passes) or one name
    /// (pipelines, which record no type): two copies of one crate in a program (a test binary that
    /// links the crate under test through a dependency of its own) submit each of its passes twice,
    /// as two types of one path, and these are one pass. That fallback is a heuristic, since
    /// `core::any::type_name` is best effort and two types may share a path; it does not apply to
    /// an entry registered at run time, whose type name is whatever its pass calls itself.
    fn is_same_as(&self, other: &Self) -> bool {
        let submitted = !self.dynamic && !other.dynamic;
        match (self.type_id, other.type_id) {
            (Some(a), Some(b)) => a == b || (submitted && self.type_name == other.type_name),
            (None, None) => submitted,
            _ => false,
        }
    }
}

impl RegistryEntry for PassRegistryEntry {
    #[inline]
    fn add_to_pipeline(&self, pm: &mut OpPassManager, options: &PassOptions) -> Result<(), Report> {
        default_registration_factory(self.builder)(pm, options)
    }

    #[inline(always)]
    fn argument(&self) -> &'static str {
        self.arg
    }

    #[inline(always)]
    fn description(&self) -> &'static str {
        self.description
    }
}

/// Information about a registered pass pipeline
pub struct PassPipelineInfo(PassRegistryEntry);
impl PassPipelineInfo {
    pub fn new(
        arg: &'static str,
        description: &'static str,
        builder: PassAllocatorFunction,
    ) -> Self {
        Self(PassRegistryEntry {
            arg,
            description,
            type_id: None,
            type_name: arg,
            builder,
            dynamic: false,
        })
    }

    /// Every registered pass pipeline, sorted by its command-line name.
    pub fn registered() -> Vec<PassPipelineInfo> {
        PASS_REGISTRY.pipelines()
    }

    /// Find the [PassInfo] for a registered pass pipeline named `name`
    pub fn lookup(name: &str) -> Option<PassPipelineInfo> {
        PASS_REGISTRY.get_pipeline(name)
    }
}
impl RegistryEntry for PassPipelineInfo {
    fn argument(&self) -> &'static str {
        self.0.argument()
    }

    fn description(&self) -> &'static str {
        self.0.description()
    }

    fn add_to_pipeline(&self, pm: &mut OpPassManager, options: &PassOptions) -> Result<(), Report> {
        self.0.add_to_pipeline(pm, options)
    }
}

/// Information about a registered pass
pub struct PassInfo(PassRegistryEntry);
impl PassInfo {
    /// Create a new [PassInfo] from the given argument name and description, for a default-
    /// constructible pass type `P`.
    pub const fn new<P: Pass + Default>(arg: &'static str, description: &'static str) -> Self {
        let type_id = TypeId::of::<P>();
        Self(PassRegistryEntry {
            arg,
            description,
            type_id: Some(type_id),
            type_name: core::any::type_name::<P>(),
            builder: default_instance::<P>,
            dynamic: false,
        })
    }

    pub const fn new_with_builder<P: Pass>(
        arg: &'static str,
        description: &'static str,
        builder: PassAllocatorFunction,
    ) -> Self {
        let type_id = TypeId::of::<P>();
        Self(PassRegistryEntry {
            arg,
            description,
            type_id: Some(type_id),
            type_name: core::any::type_name::<P>(),
            builder,
            dynamic: false,
        })
    }

    /// Find the [PassInfo] for a registered pass named `name`
    pub fn lookup(name: &str) -> Option<PassInfo> {
        PASS_REGISTRY.get_pass(name)
    }

    /// Every registered pass, sorted by its command-line name.
    pub fn registered() -> Vec<PassInfo> {
        PASS_REGISTRY.passes()
    }

    /// The command-line name of this pass.
    pub fn argument(&self) -> &'static str {
        self.0.arg
    }

    /// A one-line description of what this pass does.
    pub fn description(&self) -> &'static str {
        self.0.description
    }

    /// The registered pass names closest to `name`, for "did you mean" messages: best first, ties
    /// by name, at most three.
    ///
    /// A name is close when its edit distance (insertions, deletions and substitutions of one
    /// character) from `name` is at most a third of the length of `name`, and at least 1.
    pub fn similar_to(name: &str) -> Vec<&'static str> {
        let mut close = Vec::<(usize, &'static str)>::new();
        for pass in PASS_REGISTRY.passes() {
            let candidate = pass.argument();
            let distance = edit_distance(name, candidate);
            if distance <= (name.chars().count() / 3).max(1) {
                close.push((distance, candidate));
            }
        }
        close.sort();
        close.into_iter().take(3).map(|(_, candidate)| candidate).collect()
    }
}
impl RegistryEntry for PassInfo {
    fn argument(&self) -> &'static str {
        self.0.argument()
    }

    fn description(&self) -> &'static str {
        self.0.description()
    }

    fn add_to_pipeline(&self, pm: &mut OpPassManager, options: &PassOptions) -> Result<(), Report> {
        self.0.add_to_pipeline(pm, options)
    }
}

/// Register a specific dialect pipeline registry function with the system.
///
/// # Example
///
/// If your pipeline implements the [Default] trait, you can just do:
///
/// ```text,ignore
/// register_pass_pipeline(
///     "my-pipeline",
///     "A simple test pipeline",
///     default_registration::<MyPipeline>(),
/// )
/// ```
///
/// Otherwise, you need to pass a factor function which will be used to construct fresh instances
/// of the pipeline:
///
/// ```text,ignore
/// register_pass_pipeline(
///     "my-pipeline",
///     "A simple test pipeline",
///     default_dyn_registration(|| MyPipeline::new(MyPipelineOptions::default())),
/// )
/// ```
///
/// NOTE: The functions/closures passed above are required to be `Send + Sync + 'static`, as they
/// are stored in the global registry for the lifetime of the program, and may be accessed from any
/// thread.
///
/// Returns an error if the name is already registered.
pub fn register_pass_pipeline(
    arg: &'static str,
    description: &'static str,
    builder: PassAllocatorFunction,
) -> Result<(), Report> {
    PASS_REGISTRY.register_pipeline(PassPipelineInfo::new(arg, description, builder))
}

/// Register a specific dialect pass allocator function with the system.
///
/// # Example
///
/// ```text,ignore
/// register_pass(|| MyPass::default())
/// ```
///
/// NOTE: The allocator function provided is required to be `Send + Sync + 'static`, as it is
/// stored in the global registry for the lifetime of the program, and may be accessed from any
/// thread.
///
/// Returns an error if the name of the pass is already registered for another type.
pub fn register_pass(builder: PassAllocatorFunction) -> Result<(), Report> {
    let pass = builder();
    let type_id = pass.as_any().type_id();
    let arg = pass.argument();
    assert!(
        !arg.is_empty(),
        "attempted to register pass '{}' without specifying an argument name",
        pass.name()
    );
    let description = pass.description();
    PASS_REGISTRY.register_pass(PassInfo(PassRegistryEntry {
        arg,
        description,
        type_id: Some(type_id),
        type_name: pass.name(),
        builder,
        dynamic: false,
    }))
}

/// The number of insertions, deletions and substitutions of one character that turn `a` into `b`.
fn edit_distance(a: &str, b: &str) -> usize {
    let b = b.chars().collect::<Vec<_>>();
    let mut row = (0..=b.len()).collect::<Vec<_>>();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (diagonal + usize::from(ca != cb)).min(above + 1).min(row[j] + 1);
            diagonal = above;
        }
    }
    row[b.len()]
}

/// A default implementation of a pass pipeline registration function.
///
/// It expects that `P` (the type of the pass or pass pipeline), implements `Default`, so that an
/// instance is default-constructible. It then initializes the pass with the provided options and
/// adds it to `pm` (see [`OpPassManager::add_pass`], which nests it or rejects it when it targets
/// another operation than the anchor of `pm`).
pub fn default_registration<P: Pass + Default>(
    pm: &mut OpPassManager,
    options: &PassOptions,
) -> Result<(), Report> {
    add_with_options(default_instance::<P>(), pm, options)
}

/// Like [default_registration], but takes an arbitrary constructor in the form of a zero-arity
/// closure, rather than relying on [Default]. Thus, this is actually a registration function
/// _factory_, rather than a registration function itself.
pub fn default_registration_factory(
    builder: PassAllocatorFunction,
) -> impl Fn(&mut OpPassManager, &PassOptions) -> Result<(), Report> + Send + Sync + 'static {
    move |pm: &mut OpPassManager, options: &PassOptions| -> Result<(), Report> {
        add_with_options(builder(), pm, options)
    }
}

/// Initializes `pass` with `options` and adds it to `pm`; a pass whose options are invalid is not
/// added.
fn add_with_options(
    mut pass: Box<dyn OperationPass>,
    pm: &mut OpPassManager,
    options: &PassOptions,
) -> Result<(), Report> {
    pass.initialize_options(options)?;
    pm.add_pass(pass)
}

fn default_instance<P: Pass + Default>() -> Box<dyn OperationPass> {
    Box::<P>::default() as Box<dyn OperationPass>
}

#[cfg(test)]
mod tests;
