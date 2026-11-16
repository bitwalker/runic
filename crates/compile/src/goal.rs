use runic_core::Report;

use crate::{CheckpointId, EmitRequest, EmitTo, Route};

mod destination;

pub use self::destination::{Destination, OutputNaming};

/// One artifact a compilation is asked to produce and where it goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    checkpoint: CheckpointId,
    destination: Destination,
}

impl Output {
    /// The checkpoint whose artifact this output holds.
    pub fn checkpoint(&self) -> CheckpointId {
        self.checkpoint
    }

    /// Where the artifact goes.
    pub fn destination(&self) -> &Destination {
        &self.destination
    }
}

/// What a compilation is asked to do, decided against its route before anything runs: where it
/// stops and which artifacts it produces, each with its destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Goal {
    stop_after: CheckpointId,
    outputs: Vec<Output>,
}

impl Goal {
    /// The checkpoint after which the compilation stops: the one `--stop-after` named, or the
    /// last on the route.
    pub fn stop_after(&self) -> CheckpointId {
        self.stop_after
    }

    /// The outputs, in the order they were asked for. When none was asked for there is one: the
    /// artifact of the stop point, to the default destination.
    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }

    /// The output that holds the artifact of `checkpoint`, if one was asked for. There is at most
    /// one for a checkpoint.
    pub fn output_at(&self, checkpoint: CheckpointId) -> Option<&Output> {
        self.outputs.iter().find(|output| output.checkpoint == checkpoint)
    }
}

fn quoted(names: &[&str]) -> String {
    names.iter().map(|name| format!("'{name}'")).collect::<Vec<_>>().join(", ")
}

/// Decides the goal of a compilation on `route`.
///
/// `stop_after` and the checkpoint of each request in `emit` are resolved with [Route::resolve].
/// Only `stop_after` decides how far the compilation runs; an emit request past it is an error,
/// as is one that names no checkpoint of the route and two that name the same one. `naming` says
/// how outputs without a destination of their own are named. No file is looked at.
pub fn resolve_goal(
    route: &Route,
    stop_after: Option<&str>,
    emit: &[EmitRequest],
    naming: &OutputNaming<'_>,
) -> Result<Goal, Report> {
    let stop = match stop_after {
        None => route.terminal(),
        Some(name) => route.resolve(name).ok_or_else(|| {
            Report::msg(format!(
                "unknown checkpoint '{name}' given to --stop-after: the checkpoints of this \
                 compilation are {}",
                quoted(&route.names())
            ))
        })?,
    };
    let stop_id = route.get(stop).id;

    let mut requests: Vec<(usize, &str, &EmitTo)> = Vec::new();
    for request in emit {
        let name = request.checkpoint.as_str();
        let position = route.resolve(name).ok_or_else(|| {
            Report::msg(format!(
                "cannot emit '{name}': no checkpoint of this compilation has that name; its \
                 checkpoints are {}",
                quoted(&route.names())
            ))
        })?;
        if position > stop {
            return Err(Report::msg(format!(
                "cannot emit '{name}': it is produced after the requested stop point '{stop_id}'"
            )));
        }
        if let Some((_, first, _)) = requests.iter().find(|(other, ..)| *other == position) {
            return Err(Report::msg(format!(
                "cannot emit '{first}' and '{name}': both name the checkpoint '{}'",
                route.get(position).id
            )));
        }
        requests.push((position, name, &request.to));
    }
    if requests.is_empty() {
        requests.push((stop, stop_id.as_str(), &EmitTo::Default));
    }

    let wanted: Vec<_> =
        requests.iter().map(|(position, _, to)| (route.get(*position), *to)).collect();
    let destinations = destination::assign(&wanted, naming)?;
    let outputs = wanted
        .iter()
        .zip(destinations)
        .map(|((decl, _), destination)| Output {
            checkpoint: decl.id,
            destination,
        })
        .collect();
    Ok(Goal {
        stop_after: stop_id,
        outputs,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{CheckpointDecl, OutputFile};

    const MYLANG: &[CheckpointDecl] = &[CheckpointDecl {
        id: CheckpointId::new("mylang.ast"),
        aliases: &["ast"],
        extension: "ast",
        render: |_, _, _| Ok(()),
    }];
    const WASM: &[CheckpointDecl] = &[CheckpointDecl {
        id: CheckpointId::new("wasm.module"),
        aliases: &["wat"],
        extension: "wat",
        render: |_, _, _| Ok(()),
    }];

    fn route() -> Route {
        Route::new(&[("mylang", MYLANG)], Some(("wasm", WASM))).unwrap()
    }

    fn emit(text: &str) -> EmitRequest {
        text.parse().unwrap()
    }

    fn naming(unit: &str) -> OutputNaming<'_> {
        OutputNaming {
            unit,
            output_dir: None,
            output_file: None,
        }
    }

    fn goal(stop: Option<&str>, emits: &[&str]) -> Result<Goal, Report> {
        let emits: Vec<_> = emits.iter().map(|text| emit(text)).collect();
        resolve_goal(&route(), stop, &emits, &naming("app"))
    }

    fn error(stop: Option<&str>, emits: &[&str]) -> String {
        match goal(stop, emits) {
            Ok(goal) => panic!("should be rejected, but is {goal:?}"),
            Err(error) => error.to_string(),
        }
    }

    fn file(path: &str) -> Destination {
        Destination::File(PathBuf::from(path))
    }

    #[test]
    fn with_no_request_the_compilation_runs_to_the_end_and_emits_the_last_artifact() {
        let goal = goal(None, &[]).unwrap();
        assert_eq!(goal.stop_after().as_str(), "wasm.module");
        assert_eq!(goal.outputs().len(), 1);
        assert_eq!(goal.outputs()[0].checkpoint().as_str(), "wasm.module");
        assert_eq!(goal.outputs()[0].destination(), &file("app.wat"));
    }

    #[test]
    fn with_no_request_and_a_stop_point_the_single_output_is_the_stop_points() {
        let goal = goal(Some("ir"), &[]).unwrap();
        assert_eq!(goal.stop_after(), CheckpointId::IR_OPTIMIZED);
        assert_eq!(goal.outputs().len(), 1);
        assert_eq!(goal.outputs()[0].checkpoint(), CheckpointId::IR_OPTIMIZED);
        assert_eq!(goal.outputs()[0].destination(), &file("app.runic"));
    }

    #[test]
    fn a_stop_point_may_be_named_by_its_id_or_an_alias() {
        assert_eq!(goal(Some("mylang.ast"), &[]).unwrap().stop_after().as_str(), "mylang.ast");
        assert_eq!(goal(Some("ast"), &[]).unwrap().stop_after().as_str(), "mylang.ast");
    }

    #[test]
    fn an_unknown_stop_point_lists_every_name_of_the_route() {
        let message = error(Some("codegen"), &[]);
        assert_eq!(
            message,
            "unknown checkpoint 'codegen' given to --stop-after: the checkpoints of this \
             compilation are 'mylang.ast', 'ast', 'ir.translated', 'ir.optimized', 'ir', \
             'ir.lowered', 'wasm.module', 'wat'"
        );
    }

    #[test]
    fn outputs_are_in_the_order_they_were_asked_for() {
        let goal = goal(None, &["wat", "mylang.ast", "ir=-"]).unwrap();
        let checkpoints: Vec<_> = goal.outputs().iter().map(|o| o.checkpoint().as_str()).collect();
        assert_eq!(checkpoints, ["wasm.module", "mylang.ast", "ir.optimized"]);
    }

    #[test]
    fn an_emit_does_not_move_the_stop_point() {
        let goal = goal(None, &["ir.translated"]).unwrap();
        assert_eq!(goal.stop_after().as_str(), "wasm.module");
        assert_eq!(goal.outputs().len(), 1);
        assert_eq!(goal.outputs()[0].checkpoint(), CheckpointId::IR_TRANSLATED);
    }

    #[test]
    fn an_emit_of_the_stop_point_itself_is_accepted() {
        let goal = goal(Some("ir"), &["ir.optimized"]).unwrap();
        assert_eq!(goal.outputs()[0].checkpoint(), CheckpointId::IR_OPTIMIZED);
    }

    #[test]
    fn an_emit_past_the_stop_point_says_so_naming_both() {
        assert_eq!(
            error(Some("ir.translated"), &["ir"]),
            "cannot emit 'ir': it is produced after the requested stop point 'ir.translated'"
        );
        assert_eq!(
            error(Some("ast"), &["wat"]),
            "cannot emit 'wat': it is produced after the requested stop point 'mylang.ast'"
        );
    }

    #[test]
    fn an_emit_no_checkpoint_has_the_name_of_lists_the_names_of_the_route() {
        assert_eq!(
            error(None, &["bytecode"]),
            "cannot emit 'bytecode': no checkpoint of this compilation has that name; its \
             checkpoints are 'mylang.ast', 'ast', 'ir.translated', 'ir.optimized', 'ir', \
             'ir.lowered', 'wasm.module', 'wat'"
        );
    }

    #[test]
    fn a_checkpoint_the_route_does_not_reach_is_unknown_not_late() {
        let route = Route::new(&[], None).unwrap();
        let err = resolve_goal(&route, None, &[emit("ir.lowered")], &naming("app")).unwrap_err();
        assert!(err.to_string().contains("no checkpoint of this compilation has that name"));
    }

    #[test]
    fn two_requests_for_one_checkpoint_are_rejected_naming_both_spellings() {
        assert_eq!(
            error(None, &["ir", "ir.optimized=out.txt"]),
            "cannot emit 'ir' and 'ir.optimized': both name the checkpoint 'ir.optimized'"
        );
        assert!(error(None, &["ir", "ir"]).contains("cannot emit 'ir' and 'ir'"));
    }

    #[test]
    fn a_frontend_declared_checkpoint_resolves_as_stop_point_and_emit() {
        let goal = goal(Some("mylang.ast"), &["ast=tree.txt"]).unwrap();
        assert_eq!(goal.outputs()[0].checkpoint().as_str(), "mylang.ast");
        assert_eq!(goal.outputs()[0].destination(), &file("tree.txt"));
    }

    #[test]
    fn the_naming_of_the_request_reaches_the_outputs() {
        let naming = OutputNaming {
            unit: "app",
            output_dir: Some(std::path::Path::new("out")),
            output_file: Some(&OutputFile::Stdout),
        };
        let goal = resolve_goal(&route(), Some("ir"), &[], &naming).unwrap();
        assert_eq!(goal.outputs()[0].destination(), &Destination::Stdout);
    }
    #[test]
    fn the_output_of_a_checkpoint_is_found_by_its_id() {
        let goal = resolve_goal(
            &route(),
            None,
            &["ir=-".parse().unwrap(), "ast".parse().unwrap()],
            &OutputNaming {
                unit: "app",
                output_dir: None,
                output_file: None,
            },
        )
        .unwrap();
        let optimized = goal.output_at(CheckpointId::IR_OPTIMIZED).unwrap();
        assert_eq!(optimized.destination(), &Destination::Stdout);
        let ast = goal.output_at(CheckpointId::new("mylang.ast")).unwrap();
        assert_eq!(ast.destination(), &Destination::File("app.ast".into()));
        assert!(goal.output_at(CheckpointId::IR_TRANSLATED).is_none());
    }
}
