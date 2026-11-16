use core::{any::Any, fmt, ops::ControlFlow};

/// Identifies a point in a compilation at which an artifact exists.
///
/// An id is a dotted name whose first part is the name of whoever declares it: `ir.optimized` is
/// declared by the harness, `mylang.ast` by a frontend named `mylang`. A user names a checkpoint by its
/// id (or an alias) in `--stop-after` and `--emit`.
///
/// # Equality is by content
///
/// Two ids naming the same string are the same checkpoint, whether or not they were built from
/// the same `&'static str`. An id a frontend builds with [CheckpointId::new] equals the constant
/// the harness declares, and one parsed out of a `--stop-after` value equals both.
///
/// # `Ord` is not route order
///
/// The derived [Ord] exists only so ids can key ordered maps and sets. It is the lexicographic
/// order of the text and has no relation to the order in which a route reaches the checkpoints:
/// `ir.lowered` is less than `ir.optimized`, which is less than `ir.translated`, and a route
/// reaches them the other way around. The route says which of two checkpoints comes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CheckpointId(&'static str);

impl CheckpointId {
    /// The IR after the backend's passes have lowered it to the form it generates code from.
    pub const IR_LOWERED: Self = Self("ir.lowered");
    /// The IR after the optimization pipeline has run.
    pub const IR_OPTIMIZED: Self = Self("ir.optimized");
    /// The IR once every frontend has translated its inputs, before the optimization pipeline.
    pub const IR_TRANSLATED: Self = Self("ir.translated");

    /// An id from its text. Frontends and backends use it to declare their own checkpoints.
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    /// The text of the id.
    pub const fn as_str(&self) -> &'static str {
        self.0
    }

    /// The name of whoever declares the checkpoint: the text before the first `.`, or all of the
    /// id when it has no `.`.
    pub fn owner(&self) -> &'static str {
        match self.0.split_once('.') {
            Some((owner, _)) => owner,
            None => self.0,
        }
    }
}

impl fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// What a frontend or backend declares about one of its checkpoints: the id, the other names a
/// user may write for it, and the extension of a file holding its artifact.
///
/// A route is built from the declarations of its stages. How an artifact is turned into text or
/// bytes is not part of the declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointDecl {
    /// The id of the checkpoint.
    pub id: CheckpointId,
    /// The other names that stand for the checkpoint in `--stop-after` and `--emit`.
    pub aliases: &'static [&'static str],
    /// The extension, without the dot, of a file holding the artifact.
    pub extension: &'static str,
}

/// The checkpoints the harness declares for the IR, in the order a route reaches them.
pub const IR_CHECKPOINTS: [CheckpointDecl; 3] = [
    CheckpointDecl {
        id: CheckpointId::IR_TRANSLATED,
        aliases: &[],
        extension: "translated.runic",
    },
    CheckpointDecl {
        id: CheckpointId::IR_OPTIMIZED,
        aliases: &["ir"],
        extension: "runic",
    },
    CheckpointDecl {
        id: CheckpointId::IR_LOWERED,
        aliases: &[],
        extension: "lowered.runic",
    },
];

/// What a compilation holds at a checkpoint, tagged with the checkpoint it was produced at.
///
/// The value is type-erased so the harness never names the types frontends and backends produce.
/// Whoever knows the type recovers the value with [Artifact::downcast] or [Artifact::downcast_ref].
pub struct Artifact {
    checkpoint: CheckpointId,
    value: Box<dyn Any>,
}

impl Artifact {
    /// Wraps `value` as the artifact produced at `checkpoint`.
    pub fn new<T: Any>(checkpoint: CheckpointId, value: T) -> Self {
        Self {
            checkpoint,
            value: Box::new(value),
        }
    }

    /// The checkpoint the artifact was produced at.
    pub fn checkpoint(&self) -> CheckpointId {
        self.checkpoint
    }

    /// The value, or the artifact unchanged when it is not a `T`.
    pub fn downcast<T: Any>(self) -> Result<T, Self> {
        let Self { checkpoint, value } = self;
        match value.downcast::<T>() {
            Ok(value) => Ok(*value),
            Err(value) => Err(Self { checkpoint, value }),
        }
    }

    /// A borrow of the value, if it is a `T`.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.value.downcast_ref::<T>()
    }
}

impl fmt::Debug for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Artifact")
            .field("checkpoint", &self.checkpoint)
            .finish_non_exhaustive()
    }
}

/// Records that a compilation reached the checkpoint it was asked to stop after.
///
/// The artifact has already been kept by the time a stop is reported: the marker carries only the
/// checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stopped {
    checkpoint: CheckpointId,
}

impl Stopped {
    /// A stop marker for `checkpoint`.
    pub const fn new(checkpoint: CheckpointId) -> Self {
        Self { checkpoint }
    }

    /// The checkpoint the compilation stopped at.
    pub const fn checkpoint(&self) -> CheckpointId {
        self.checkpoint
    }
}

/// The result of publishing an artifact at a checkpoint: `Continue` hands back what the stage
/// keeps working with, and `Break` says the stop point was reached and the stage must return.
pub type Flow<T> = ControlFlow<Stopped, T>;

/// Is told of each artifact as it is produced, without taking it.
///
/// A program that embeds the compiler gives observers to a compilation to watch it. An observer
/// does not write output: writing artifacts is the harness's job.
pub trait Observer {
    /// Called after `artifact` has been produced at `checkpoint`.
    fn on_checkpoint(&mut self, checkpoint: CheckpointId, artifact: &Artifact);
}

/// An [Observer] that keeps the checkpoints it was told of, in order, for tests to read.
#[derive(Debug, Default)]
pub struct RecordingObserver {
    checkpoints: Vec<CheckpointId>,
}

impl RecordingObserver {
    /// The checkpoints reached, in the order they were reached.
    pub fn checkpoints(&self) -> &[CheckpointId] {
        &self.checkpoints
    }
}

impl Observer for RecordingObserver {
    fn on_checkpoint(&mut self, checkpoint: CheckpointId, _artifact: &Artifact) {
        self.checkpoints.push(checkpoint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Ir(u32);

    #[derive(Debug, PartialEq)]
    struct Text(&'static str);

    #[test]
    fn checkpoint_ids_are_stable_strings_that_display_as_themselves() {
        assert_eq!(CheckpointId::IR_TRANSLATED.as_str(), "ir.translated");
        assert_eq!(CheckpointId::IR_OPTIMIZED.as_str(), "ir.optimized");
        assert_eq!(CheckpointId::IR_LOWERED.to_string(), "ir.lowered");
    }

    #[test]
    fn an_id_built_by_a_frontend_equals_the_constant_it_names() {
        assert_eq!(CheckpointId::new("ir.translated"), CheckpointId::IR_TRANSLATED);
        assert_ne!(CheckpointId::new("ir.translated"), CheckpointId::IR_LOWERED);

        // Literals may be merged into one static, so they could not catch a comparison of
        // pointers. This one is built at run time and cannot share a pointer with the constant.
        let built: &'static str = String::from("ir.translated").leak();
        assert_eq!(
            CheckpointId::new(built),
            CheckpointId::IR_TRANSLATED,
            "equality must compare the strings, never the pointers to them"
        );
    }

    #[test]
    fn ord_is_lexicographic_and_is_not_route_order() {
        assert!(CheckpointId::IR_LOWERED < CheckpointId::IR_OPTIMIZED);
        assert!(CheckpointId::IR_OPTIMIZED < CheckpointId::IR_TRANSLATED);
        let route = IR_CHECKPOINTS.map(|decl| decl.id);
        assert_eq!(
            route,
            [
                CheckpointId::IR_TRANSLATED,
                CheckpointId::IR_OPTIMIZED,
                CheckpointId::IR_LOWERED
            ]
        );
    }

    #[test]
    fn the_owner_is_the_text_before_the_first_dot_or_all_of_it() {
        assert_eq!(CheckpointId::IR_TRANSLATED.owner(), "ir");
        assert_eq!(CheckpointId::new("mylang.ast").owner(), "mylang");
        assert_eq!(CheckpointId::new("mylang.ast.typed").owner(), "mylang");
        assert_eq!(CheckpointId::new("bare").owner(), "bare");
    }

    #[test]
    fn the_ir_declares_three_checkpoints_with_their_extensions_and_one_alias() {
        let shape = IR_CHECKPOINTS.map(|decl| (decl.id.as_str(), decl.aliases, decl.extension));
        assert_eq!(
            shape,
            [
                ("ir.translated", &[][..], "translated.runic"),
                ("ir.optimized", &["ir"][..], "runic"),
                ("ir.lowered", &[][..], "lowered.runic"),
            ]
        );
    }

    #[test]
    fn an_artifact_keeps_the_checkpoint_it_was_produced_at() {
        let artifact = Artifact::new(CheckpointId::IR_TRANSLATED, Ir(7));
        assert_eq!(artifact.checkpoint(), CheckpointId::IR_TRANSLATED);
        assert!(format!("{artifact:?}").contains("ir.translated"));
    }

    #[test]
    fn downcast_returns_the_value_when_the_type_matches() {
        let artifact = Artifact::new(CheckpointId::IR_TRANSLATED, Ir(7));
        assert_eq!(artifact.downcast::<Ir>().unwrap(), Ir(7));
    }

    #[test]
    fn downcast_returns_the_artifact_unchanged_when_the_type_does_not_match() {
        let artifact = Artifact::new(CheckpointId::IR_TRANSLATED, Ir(7));
        let returned = artifact.downcast::<Text>().unwrap_err();
        assert_eq!(returned.checkpoint(), CheckpointId::IR_TRANSLATED);
        assert_eq!(returned.downcast::<Ir>().unwrap(), Ir(7));
    }

    #[test]
    fn downcast_ref_borrows_without_consuming() {
        let artifact = Artifact::new(CheckpointId::IR_LOWERED, Text("begin"));
        assert_eq!(artifact.downcast_ref::<Text>(), Some(&Text("begin")));
        assert_eq!(artifact.downcast_ref::<Ir>(), None);
        assert_eq!(artifact.checkpoint(), CheckpointId::IR_LOWERED);
    }

    #[test]
    fn a_stop_marker_names_its_checkpoint() {
        let stopped = Stopped::new(CheckpointId::IR_OPTIMIZED);
        assert_eq!(stopped.checkpoint(), CheckpointId::IR_OPTIMIZED);
        let flow: Flow<u8> = ControlFlow::Break(stopped);
        assert_eq!(flow, ControlFlow::Break(Stopped::new(CheckpointId::IR_OPTIMIZED)));
    }

    #[test]
    fn a_recording_observer_keeps_the_checkpoints_in_the_order_reached() {
        let mut observer = RecordingObserver::default();
        let first = Artifact::new(CheckpointId::IR_TRANSLATED, 1u32);
        let second = Artifact::new(CheckpointId::IR_LOWERED, 2u32);
        observer.on_checkpoint(first.checkpoint(), &first);
        observer.on_checkpoint(second.checkpoint(), &second);
        assert_eq!(
            observer.checkpoints(),
            &[CheckpointId::IR_TRANSLATED, CheckpointId::IR_LOWERED]
        );
    }
}
