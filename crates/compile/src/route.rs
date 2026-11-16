use runic_ir::Report;

use crate::{CheckpointDecl, CheckpointId, IR_CHECKPOINTS};

/// A frontend's or backend's name and the checkpoints it declares, in the order its stage reaches
/// them.
pub type Declarer<'a> = (&'a str, &'a [CheckpointDecl]);

/// The checkpoints one compilation passes through, in the order it reaches them.
///
/// A route is each frontend's checkpoints in the order the frontends were given, then the
/// harness's `ir.translated` and `ir.optimized`, and, when there is a backend, `ir.lowered`
/// followed by the backend's checkpoints. A route that has been built has distinct ids, aliases
/// and extensions, and every id is prefixed by the name of the stage that declares it, so a name
/// written by a user stands for at most one checkpoint. The harness reads the route to resolve
/// `--stop-after` and `--emit`, and to name output files.
#[derive(Debug, Clone)]
pub struct Route {
    checkpoints: Vec<CheckpointDecl>,
}

/// Who declares the checkpoints the harness adds to every route.
const HARNESS: &str = "the harness";

impl Route {
    /// The route of a compilation whose frontends and backend are given as `(name, declarations)`.
    ///
    /// Fails, naming the stage at fault, when:
    ///
    /// * a stage's name is empty, is `ir`, is used by two stages, or holds anything but ASCII
    ///   letters, digits, `-` and `_`;
    /// * an id is not `<name>.<stage>` for its declarer's name, with a stage of ASCII letters,
    ///   digits, `-`, `_` and `.`;
    /// * the backend declares no checkpoint;
    /// * an alias or an extension is empty or holds anything but ASCII letters, digits, `-`, `_`
    ///   and `.`;
    /// * an id, an alias or an extension is used twice, or an alias is also an id.
    pub fn new(frontends: &[Declarer<'_>], backend: Option<Declarer<'_>>) -> Result<Self, Report> {
        let mut stages: Vec<(String, &str, &[CheckpointDecl])> = frontends
            .iter()
            .map(|(name, decls)| (format!("frontend '{name}'"), *name, *decls))
            .collect();
        if let Some((name, decls)) = backend {
            stages.push((format!("backend '{name}'"), name, decls));
        }

        for (index, (label, name, decls)) in stages.iter().enumerate() {
            if name.is_empty() {
                return Err(Report::msg(format!("the name of {label} is empty")));
            }
            if let Some(invalid) = invalid_char(name, false) {
                return Err(Report::msg(format!(
                    "the name of {label} holds '{invalid}': a name is made of ASCII letters, \
                     digits, '-' and '_'"
                )));
            }
            if *name == "ir" {
                return Err(Report::msg(format!(
                    "{label} cannot be called 'ir': the name belongs to the checkpoints {HARNESS} \
                     declares"
                )));
            }
            if let Some((first, ..)) = stages[..index].iter().find(|(_, other, _)| other == name) {
                return Err(Report::msg(format!(
                    "{label} has the same name as {first}: the name of a stage must be unique on \
                     a route"
                )));
            }
            validate_declarations(label, name, decls)?;
        }
        if let Some((label, _, decls)) = stages.last()
            && backend.is_some()
            && decls.is_empty()
        {
            return Err(Report::msg(format!(
                "{label} declares no checkpoint: a backend declares at least the checkpoint of \
                 what it produces"
            )));
        }

        // Each checkpoint with who declares it, in route order.
        let mut declared: Vec<(CheckpointDecl, &str)> = Vec::new();
        for (label, _, decls) in &stages[..frontends.len()] {
            declared.extend(decls.iter().map(|decl| (*decl, label.as_str())));
        }
        declared.extend(IR_CHECKPOINTS[..2].iter().map(|decl| (*decl, HARNESS)));
        if let Some((label, _, decls)) = stages.get(frontends.len()) {
            declared.push((IR_CHECKPOINTS[2], HARNESS));
            declared.extend(decls.iter().map(|decl| (*decl, label.as_str())));
        }
        check_distinct(&declared)?;
        Ok(Self {
            checkpoints: declared.into_iter().map(|(decl, _)| decl).collect(),
        })
    }

    /// The checkpoints, in the order the route reaches them.
    pub fn checkpoints(&self) -> &[CheckpointDecl] {
        &self.checkpoints
    }

    /// The declaration at `position` on the route.
    ///
    /// # Panics
    ///
    /// Panics when `position` is not less than the number of checkpoints, which a position
    /// returned by this route never is.
    pub fn get(&self, position: usize) -> &CheckpointDecl {
        &self.checkpoints[position]
    }

    /// The position of the checkpoint a user means by `name`: the one whose id is `name`, or else
    /// the one that has `name` among its aliases.
    pub fn resolve(&self, name: &str) -> Option<usize> {
        self.checkpoints
            .iter()
            .position(|decl| decl.id.as_str() == name)
            .or_else(|| self.checkpoints.iter().position(|decl| decl.aliases.contains(&name)))
    }

    /// The position of the checkpoint whose id is `id`. Aliases are not consulted.
    pub fn position(&self, id: CheckpointId) -> Option<usize> {
        self.checkpoints.iter().position(|decl| decl.id == id)
    }

    /// The position of the last checkpoint, where a compilation that is not asked to stop ends.
    pub fn terminal(&self) -> usize {
        // A route always holds the harness's own checkpoints.
        self.checkpoints.len() - 1
    }

    /// Every name a user may write for a checkpoint on this route: for each checkpoint in route
    /// order, its id and then its aliases.
    pub fn names(&self) -> Vec<&'static str> {
        self.checkpoints
            .iter()
            .flat_map(|decl| core::iter::once(decl.id.as_str()).chain(decl.aliases.iter().copied()))
            .collect()
    }
}

/// The first character of `text` that a name may not hold: anything but ASCII letters, digits,
/// `-` and `_`, and `.` when `dots` allows it.
fn invalid_char(text: &str, dots: bool) -> Option<char> {
    text.chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_') || (dots && *c == '.')))
}

fn validate_declarations(label: &str, name: &str, decls: &[CheckpointDecl]) -> Result<(), Report> {
    for decl in decls {
        let id = decl.id;
        if id.owner() != name {
            return Err(Report::msg(format!(
                "{label} declares the checkpoint '{id}', whose id does not start with the name of \
                 its declarer: write '{name}.<stage>'"
            )));
        }
        let stage = id.as_str().split_once('.').map_or("", |(_, stage)| stage);
        if stage.is_empty() {
            return Err(Report::msg(format!(
                "{label} declares the checkpoint '{id}', which names no stage after '{name}': \
                 write '{name}.<stage>'"
            )));
        }
        if let Some(invalid) = invalid_char(stage, true) {
            return Err(Report::msg(format!(
                "{label} declares the checkpoint '{id}', whose id holds '{invalid}': an id is \
                 made of ASCII letters, digits, '-', '_' and '.'"
            )));
        }
        if decl.extension.is_empty() {
            return Err(Report::msg(format!(
                "{label} declares the checkpoint '{id}' with an empty file extension"
            )));
        }
        if let Some(invalid) = invalid_char(decl.extension, true) {
            return Err(Report::msg(format!(
                "{label} declares the checkpoint '{id}' with the file extension '{}', which holds \
                 '{invalid}': an extension is made of ASCII letters, digits, '-', '_' and '.'",
                decl.extension
            )));
        }
        for alias in decl.aliases {
            if alias.is_empty() {
                return Err(Report::msg(format!(
                    "{label} declares the checkpoint '{id}' with an empty alias"
                )));
            }
            if let Some(invalid) = invalid_char(alias, true) {
                return Err(Report::msg(format!(
                    "{label} declares the checkpoint '{id}' with the alias '{alias}', which holds \
                     '{invalid}': an alias is made of ASCII letters, digits, '-', '_' and '.'"
                )));
            }
        }
    }
    Ok(())
}

/// Checks that no id, alias or extension of `declared` (each checkpoint with who declares it, in
/// route order) is used twice, and that no alias is also an id. An error names both checkpoints
/// and both declarers.
fn check_distinct(declared: &[(CheckpointDecl, &str)]) -> Result<(), Report> {
    for (index, (decl, by)) in declared.iter().enumerate() {
        let earlier = &declared[..index];
        if let Some((_, first_by)) = earlier.iter().find(|(other, _)| other.id == decl.id) {
            return Err(Report::msg(format!(
                "the checkpoint '{}' is declared twice on one route: by {first_by} and by {by}",
                decl.id
            )));
        }
        if let Some((other, other_by)) =
            earlier.iter().find(|(other, _)| other.extension == decl.extension)
        {
            return Err(Report::msg(format!(
                "the file extension '{}' is used by both '{}' (declared by {other_by}) and '{}' \
                 (declared by {by})",
                decl.extension, other.id, decl.id
            )));
        }
    }
    for (index, (decl, by)) in declared.iter().enumerate() {
        for (position, alias) in decl.aliases.iter().enumerate() {
            if let Some((other, other_by)) =
                declared.iter().find(|(other, _)| other.id.as_str() == *alias)
            {
                return Err(Report::msg(format!(
                    "the alias '{alias}' of the checkpoint '{}' (declared by {by}) is also the id \
                     of the checkpoint '{}' (declared by {other_by})",
                    decl.id, other.id
                )));
            }
            let given_before = declared[..index]
                .iter()
                .find(|(other, _)| other.aliases.contains(alias))
                .map(|(other, other_by)| (other.id, *other_by))
                .or_else(|| decl.aliases[..position].contains(alias).then_some((decl.id, *by)));
            if let Some((other, other_by)) = given_before {
                return Err(Report::msg(format!(
                    "the alias '{alias}' is given to both '{other}' (declared by {other_by}) and \
                     '{}' (declared by {by})",
                    decl.id
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn decl(
        id: &'static str,
        aliases: &'static [&'static str],
        extension: &'static str,
    ) -> CheckpointDecl {
        CheckpointDecl {
            id: CheckpointId::new(id),
            aliases,
            extension,
            render: |_, _, _| Ok(()),
        }
    }

    const MYLANG: &[CheckpointDecl] =
        &[decl("mylang.ast", &["ast"], "ast"), decl("mylang.typed", &[], "typed")];
    const WASM: &[CheckpointDecl] = &[decl("wasm.module", &["wat"], "wat")];

    fn ids(route: &Route) -> Vec<&'static str> {
        route.checkpoints().iter().map(|d| d.id.as_str()).collect()
    }

    fn error(frontends: &[Declarer<'_>], backend: Option<Declarer<'_>>) -> String {
        match Route::new(frontends, backend) {
            Ok(route) => panic!("the route should be rejected, but is {:?}", ids(&route)),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn a_route_without_stages_is_the_ir_up_to_the_optimized_checkpoint() {
        let route = Route::new(&[], None).unwrap();
        assert_eq!(ids(&route), ["ir.translated", "ir.optimized"]);
        assert_eq!(route.terminal(), 1);
    }

    #[test]
    fn a_route_is_the_frontends_then_the_ir_then_the_backend() {
        let route = Route::new(&[("mylang", MYLANG)], Some(("wasm", WASM))).unwrap();
        assert_eq!(
            ids(&route),
            [
                "mylang.ast",
                "mylang.typed",
                "ir.translated",
                "ir.optimized",
                "ir.lowered",
                "wasm.module"
            ]
        );
        assert_eq!(route.terminal(), 5);
        assert_eq!(route.get(5).extension, "wat");
    }

    #[test]
    fn frontends_keep_the_order_they_were_given_in() {
        let other: &[CheckpointDecl] = &[decl("other.ast", &[], "oast")];
        let route = Route::new(&[("other", other), ("mylang", MYLANG)], None).unwrap();
        assert_eq!(
            ids(&route),
            ["other.ast", "mylang.ast", "mylang.typed", "ir.translated", "ir.optimized"]
        );
    }

    #[test]
    fn a_frontend_may_declare_no_checkpoint() {
        let route = Route::new(&[("quiet", &[])], Some(("wasm", WASM))).unwrap();
        assert_eq!(ids(&route), ["ir.translated", "ir.optimized", "ir.lowered", "wasm.module"]);
    }

    #[test]
    fn a_name_is_an_id_or_an_alias() {
        let route = Route::new(&[("mylang", MYLANG)], Some(("wasm", WASM))).unwrap();
        assert_eq!(route.resolve("mylang.typed"), Some(1));
        assert_eq!(route.resolve("ast"), Some(0));
        assert_eq!(route.resolve("ir"), Some(3));
        assert_eq!(route.resolve("wat"), Some(5));
        assert_eq!(route.resolve("nothing"), None);
    }

    #[test]
    fn a_position_is_found_by_id_alone() {
        let route = Route::new(&[("mylang", MYLANG)], None).unwrap();
        assert_eq!(route.position(CheckpointId::new("ir.optimized")), Some(3));
        assert_eq!(route.position(CheckpointId::new("ir")), None);
        assert_eq!(route.position(CheckpointId::IR_LOWERED), None);
    }

    #[test]
    fn the_names_are_each_id_followed_by_its_aliases_in_route_order() {
        let route = Route::new(&[("mylang", MYLANG)], None).unwrap();
        assert_eq!(
            route.names(),
            ["mylang.ast", "ast", "mylang.typed", "ir.translated", "ir.optimized", "ir"]
        );
    }

    #[test]
    fn a_stage_with_an_empty_name_is_rejected() {
        assert!(error(&[("", &[])], None).contains("frontend ''"));
        assert!(error(&[], Some(("", WASM))).contains("backend ''"));
    }

    #[test]
    fn a_stage_called_ir_is_rejected() {
        let message = error(&[("ir", &[])], None);
        assert!(message.contains("frontend 'ir'") && message.contains("'ir'"), "{message}");
        assert!(error(&[], Some(("ir", WASM))).contains("backend 'ir'"));
    }

    #[test]
    fn two_stages_with_one_name_are_rejected_naming_both() {
        let message = error(&[("a", &[]), ("a", &[])], None);
        assert!(message.contains("frontend 'a' has the same name as frontend 'a'"), "{message}");
        let message = error(&[("wasm", &[])], Some(("wasm", WASM)));
        assert!(
            message.contains("backend 'wasm' has the same name as frontend 'wasm'"),
            "{message}"
        );
    }

    #[test]
    fn an_id_not_prefixed_by_its_declarer_is_rejected() {
        let wrong: &[CheckpointDecl] = &[decl("other.ast", &[], "x")];
        let message = error(&[("mylang", wrong)], None);
        assert!(
            message.contains("frontend 'mylang'") && message.contains("'other.ast'"),
            "{message}"
        );
    }

    #[test]
    fn an_id_with_no_stage_after_the_owner_is_rejected() {
        for id in ["mylang", "mylang."] {
            let bare: &[CheckpointDecl] = &[decl(id, &[], "x")];
            let message = error(&[("mylang", bare)], None);
            assert!(message.contains("names no stage") && message.contains(id), "{message}");
        }
    }

    #[test]
    fn a_backend_with_no_checkpoint_is_rejected() {
        let message = error(&[], Some(("wasm", &[])));
        assert!(message.contains("backend 'wasm' declares no checkpoint"), "{message}");
    }

    #[test]
    fn an_empty_alias_or_extension_is_rejected() {
        let alias: &[CheckpointDecl] = &[decl("a.x", &[""], "x")];
        assert!(error(&[("a", alias)], None).contains("empty alias"));
        let extension: &[CheckpointDecl] = &[decl("a.x", &[], "")];
        assert!(error(&[("a", extension)], None).contains("empty file extension"));
    }

    #[test]
    fn an_id_used_twice_is_rejected() {
        let twice: &[CheckpointDecl] = &[decl("a.x", &[], "x"), decl("a.x", &[], "y")];
        let message = error(&[("a", twice)], None);
        assert_eq!(
            message,
            "the checkpoint 'a.x' is declared twice on one route: by frontend 'a' and by frontend \
             'a'"
        );
    }

    #[test]
    fn an_alias_equal_to_an_id_or_to_another_alias_is_rejected() {
        let shadows: &[CheckpointDecl] = &[decl("a.x", &["ir.optimized"], "x")];
        let message = error(&[("a", shadows)], None);
        assert!(
            message.contains("alias 'ir.optimized'") && message.contains("also the id"),
            "{message}"
        );

        let own: &[CheckpointDecl] = &[decl("a.x", &[], "x"), decl("a.y", &["a.x"], "y")];
        assert!(error(&[("a", own)], None).contains("alias 'a.x'"));

        let shared: &[CheckpointDecl] = &[decl("a.x", &["s"], "x"), decl("a.y", &["s"], "y")];
        let message = error(&[("a", shared)], None);
        assert_eq!(
            message,
            "the alias 's' is given to both 'a.x' (declared by frontend 'a') and 'a.y' (declared \
             by frontend 'a')"
        );

        let repeated: &[CheckpointDecl] = &[decl("a.x", &["s", "s"], "x")];
        assert!(error(&[("a", repeated)], None).contains("alias 's' is given to both 'a.x'"));
    }

    #[test]
    fn a_clash_with_a_checkpoint_of_the_harness_names_the_stage_at_fault() {
        let ir_alias: &[CheckpointDecl] = &[decl("a.x", &["ir"], "x")];
        assert_eq!(
            error(&[("a", ir_alias)], None),
            "the alias 'ir' is given to both 'a.x' (declared by frontend 'a') and 'ir.optimized' \
             (declared by the harness)"
        );
        let extension: &[CheckpointDecl] = &[decl("a.x", &[], "runic")];
        assert_eq!(
            error(&[("a", extension)], None),
            "the file extension 'runic' is used by both 'a.x' (declared by frontend 'a') and \
             'ir.optimized' (declared by the harness)"
        );
        let backend: &[CheckpointDecl] = &[decl("b.out", &["ir"], "lowered.runic")];
        let message = error(&[], Some(("b", backend)));
        assert!(
            message.contains("'ir.lowered' (declared by the harness)")
                && message.contains("'b.out' (declared by backend 'b')"),
            "{message}"
        );
    }

    #[test]
    fn a_clash_between_two_stages_names_both() {
        let front: &[CheckpointDecl] = &[decl("a.x", &["out"], "x")];
        let back: &[CheckpointDecl] = &[decl("b.y", &["out"], "y")];
        assert_eq!(
            error(&[("a", front)], Some(("b", back))),
            "the alias 'out' is given to both 'a.x' (declared by frontend 'a') and 'b.y' \
             (declared by backend 'b')"
        );
        let back: &[CheckpointDecl] = &[decl("b.y", &["a.x"], "y")];
        assert_eq!(
            error(&[("a", front)], Some(("b", back))),
            "the alias 'a.x' of the checkpoint 'b.y' (declared by backend 'b') is also the id of \
             the checkpoint 'a.x' (declared by frontend 'a')"
        );
        let back: &[CheckpointDecl] = &[decl("b.y", &[], "x")];
        assert!(
            error(&[("a", front)], Some(("b", back))).contains("extension 'x' is used by both")
        );
    }

    #[test]
    fn two_frontends_and_a_backend_make_one_route() {
        let other: &[CheckpointDecl] = &[decl("other.ast", &[], "oast")];
        let route =
            Route::new(&[("mylang", MYLANG), ("other", other)], Some(("wasm", WASM))).unwrap();
        assert_eq!(
            ids(&route),
            [
                "mylang.ast",
                "mylang.typed",
                "other.ast",
                "ir.translated",
                "ir.optimized",
                "ir.lowered",
                "wasm.module"
            ]
        );
    }

    #[test]
    fn a_backend_id_not_prefixed_by_its_name_is_rejected() {
        let wrong: &[CheckpointDecl] = &[decl("mylang.out", &[], "o")];
        let message = error(&[("mylang", MYLANG)], Some(("wasm", wrong)));
        assert!(
            message.contains("backend 'wasm'") && message.contains("'mylang.out'"),
            "{message}"
        );
    }

    #[test]
    fn a_name_an_id_an_alias_or_an_extension_with_a_character_it_cannot_hold_is_rejected() {
        let message = error(&[("my.lang", &[])], None);
        assert!(message.contains("the name of frontend 'my.lang' holds '.'"), "{message}");
        assert!(error(&[], Some(("wasm cm", WASM))).contains("backend 'wasm cm' holds ' '"));
        Route::new(&[], Some(("wasm-cm", &[decl("wasm-cm.module", &[], "wasm")]))).unwrap();

        let id: &[CheckpointDecl] = &[decl("a.x/y", &[], "x")];
        assert!(error(&[("a", id)], None).contains("whose id holds '/'"));
        let alias: &[CheckpointDecl] = &[decl("a.x", &["x=y"], "x")];
        assert!(error(&[("a", alias)], None).contains("the alias 'x=y', which holds '='"));
        let extension: &[CheckpointDecl] = &[decl("a.x", &[], "../x")];
        assert!(
            error(&[("a", extension)], None).contains("the file extension '../x', which holds '/'")
        );
    }

    #[test]
    fn an_extension_used_twice_is_rejected() {
        let twice: &[CheckpointDecl] = &[decl("a.x", &[], "e"), decl("a.y", &[], "e")];
        let message = error(&[("a", twice)], None);
        assert!(
            message.contains("extension 'e' is used by both 'a.x' (declared by frontend 'a')"),
            "{message}"
        );
    }
}
