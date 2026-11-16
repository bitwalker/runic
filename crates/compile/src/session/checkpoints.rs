//! The hand a stage uses to reach checkpoints.

use core::any::Any;

use runic_core::Report;

use crate::{
    Artifact, CheckpointDecl, CheckpointId, Flow, Frontend, Goal, IR_CHECKPOINTS, Observer, Route,
    Stopped,
};

/// What a session has done so far: the checkpoints reached, the artifact kept at the stop point,
/// and whether the translate step has run.
#[derive(Default)]
pub(super) struct Progress {
    pub(super) reached: Vec<CheckpointId>,
    pub(super) stopped: Option<Artifact>,
    pub(super) translated: bool,
}

/// The stage that is running, and so which checkpoints it may reach.
#[derive(Clone, Copy)]
pub(super) enum Stage<'a> {
    Frontend(&'a dyn Frontend),
    Harness,
}

impl Stage<'_> {
    fn declared(&self) -> &'static [CheckpointDecl] {
        match self {
            Self::Frontend(frontend) => frontend.checkpoints(),
            Self::Harness => &IR_CHECKPOINTS,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Frontend(frontend) => format!("frontend '{}'", frontend.name()),
            Self::Harness => "the harness".to_string(),
        }
    }
}

/// How the stage that is running reaches the checkpoints it declared.
///
/// A frontend is given one by the translate step and calls [Checkpoints::reach] each time it holds
/// an artifact it declared a checkpoint for. The harness checks that the checkpoint is the stage's
/// own and comes after the last one reached, tells the observer, and, at the checkpoint the
/// compilation is to stop after, keeps the artifact and says so.
pub struct Checkpoints<'a> {
    progress: &'a mut Progress,
    observer: &'a mut dyn Observer,
    route: &'a Route,
    goal: &'a Goal,
    stage: Stage<'a>,
}

impl<'a> Checkpoints<'a> {
    pub(super) fn new(
        progress: &'a mut Progress,
        observer: &'a mut dyn Observer,
        route: &'a Route,
        goal: &'a Goal,
        stage: Stage<'a>,
    ) -> Self {
        Self {
            progress,
            observer,
            route,
            goal,
            stage,
        }
    }

    /// Reaches `checkpoint` holding `value`.
    ///
    /// It is an error to reach a checkpoint after the stop point was reached, one the running
    /// stage did not declare, or one that does not come later on the route than the last one
    /// reached (so a checkpoint cannot be reached twice). Otherwise the observer is told, and the
    /// answer is `Continue(value)` to go on, or, when `checkpoint` is where the compilation stops,
    /// `Break` after the artifact has been kept: the stage returns that `Break` at once.
    pub fn reach<T: Any>(&mut self, checkpoint: CheckpointId, value: T) -> Result<Flow<T>, Report> {
        let stage = self.stage.describe();
        let stop = self.goal.stop_after();
        if self.progress.stopped.is_some() {
            return Err(Report::msg(format!(
                "{stage} reached checkpoint '{checkpoint}' after the compilation had stopped at \
                 '{stop}'"
            )));
        }
        let declared = self.stage.declared();
        if !declared.iter().any(|decl| decl.id == checkpoint) {
            let ids: Vec<String> = declared.iter().map(|decl| format!("'{}'", decl.id)).collect();
            let known = if ids.is_empty() {
                "it declares none".to_string()
            } else {
                format!("it declares {}", ids.join(", "))
            };
            return Err(Report::msg(format!(
                "{stage} reached checkpoint '{checkpoint}', which it did not declare ({known})"
            )));
        }
        let Some(position) = self.route.position(checkpoint) else {
            return Err(Report::msg(format!(
                "{stage} reached checkpoint '{checkpoint}', which is not on the route of this \
                 compilation"
            )));
        };
        if let Some(last) = self.progress.reached.last() {
            let after = self.route.position(*last).is_some_and(|last| position > last);
            if !after {
                return Err(Report::msg(format!(
                    "{stage} reached checkpoint '{checkpoint}' after '{last}': checkpoints are \
                     reached in the order of the route, each once"
                )));
            }
        }

        let artifact = Artifact::new(checkpoint, value);
        self.observer.on_checkpoint(checkpoint, &artifact);
        self.progress.reached.push(checkpoint);
        if checkpoint == stop {
            self.progress.stopped = Some(artifact);
            return Ok(Flow::Break(Stopped::new(checkpoint)));
        }
        match artifact.downcast::<T>() {
            Ok(value) => Ok(Flow::Continue(value)),
            Err(_) => unreachable!("the artifact was made from a value of this type"),
        }
    }
}
