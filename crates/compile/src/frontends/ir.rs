//! The frontend for Runic's own textual IR.

use runic_core::{
    DATA_LAYOUT_ATTR, DataLayoutOpInterface, DataLayoutSpec, Op, OpPrintingFlags, Operation,
    OperationRef, Report, Symbol, SymbolTable,
    diagnostics::DefaultFailurePolicy,
    dialects::builtin::{Component, Module, World, WorldBuilder, WorldRef},
    parse::{ParserConfig, parse_any},
    print::AsmPrinter,
    set_data_layout_spec,
};

use crate::{Checkpoints, Flow, Frontend, Input, Session};

/// The frontend for files of Runic's textual IR (`.runic`).
///
/// Each input is one `builtin.module`, `builtin.component` or `builtin.world`. A module or a
/// component is added to the session's world; the modules and components of a world file are
/// added to it in order. The frontend declares no checkpoint.
///
/// One world has one data layout. A world file that states a layout must state the one the
/// session's world has (the target's, or an earlier file's), and gives its own to a world that has
/// none. A module or a component that states a layout must state the world's; it may state one
/// of its own only when the world has none.
///
/// Each file is parsed by itself. A reference from one file to a symbol another input defines is
/// linked by the translate step, once every frontend has run.
#[derive(Debug, Default, Clone, Copy)]
pub struct IrFrontend;

impl Frontend for IrFrontend {
    fn name(&self) -> &'static str {
        "runic"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["runic"]
    }

    fn translate(
        &self,
        session: &Session<'_>,
        inputs: &[Input],
        world: WorldRef,
        _checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        for input in inputs {
            let parsed = parse(session, input)?;
            merge(input, parsed, world)?;
        }
        Ok(Flow::Continue(()))
    }
}

fn parse(session: &Session<'_>, input: &Input) -> Result<OperationRef, Report> {
    let bytes = session.read(input)?;
    let text = core::str::from_utf8(&bytes).map_err(|error| {
        Report::msg(format!("input '{}' is not valid UTF-8: {error}", input.name()))
    })?;
    let config = ParserConfig::new(session.context().clone()).verify_after_parse(false);
    let outcome = parse_any(config, input.name(), text);
    if outcome.result.is_err() || outcome.diagnostics.has_errors() {
        return Err(match outcome.into_result_with_policy(&DefaultFailurePolicy) {
            Err(report) => report,
            Ok(_) => Report::msg(format!("input '{}' could not be parsed", input.name())),
        });
    }
    let runic_core::diagnostics::Outcome {
        result,
        diagnostics,
    } = outcome;
    let Ok(parsed) = result else {
        return Err(Report::msg(format!("input '{}' could not be parsed", input.name())));
    };
    // The parser took what it reported out of the context: what is not an error goes back, at
    // the severity it was reported with.
    for entry in diagnostics {
        let severity = entry.effective_severity;
        session.context().emit_owned(entry.diagnostic.with_severity_override(severity));
    }
    Ok(parsed)
}

/// Adds what `parsed` holds to `world`. Nothing of `world` is changed when this fails.
fn merge(input: &Input, parsed: OperationRef, mut world: WorldRef) -> Result<(), Report> {
    let name = input.name();
    // The layout a world file states, when the session's world has none yet and takes it.
    let mut adopted = None;
    let (units, in_world_file) = {
        let parsed = parsed.borrow();
        if let Some(file_world) = parsed.downcast_ref::<World>() {
            adopted = layout_to_adopt(&name, file_world, world)?;
            let units = file_world.body().entry_block_ref().map_or_else(Vec::new, |block| {
                block.borrow().body().iter().map(|op| op.as_operation_ref()).collect()
            });
            (units, true)
        } else {
            // A module or a component is inside the parser's own world.
            (vec![parsed.as_operation_ref()], false)
        }
    };
    let layout = world.borrow().data_layout_spec().or_else(|| adopted.clone());
    for unit in &units {
        let operation = unit.borrow();
        if !operation.is::<Module>() && !operation.is::<Component>() {
            return Err(Report::msg(if in_world_file {
                format!(
                    "the world of '{name}' holds a '{}': a world file holds 'builtin.module' and \
                     'builtin.component' operations",
                    operation.name()
                )
            } else {
                format!(
                    "'{name}' holds a '{}' at its top level: a unit is a 'builtin.module', a \
                     'builtin.component' or a 'builtin.world'",
                    operation.name()
                )
            }));
        }
        check_unit_layout(&name, &operation, layout.as_ref(), world)?;
    }
    for unit in &units {
        check_definable(&name, *unit, world)?;
    }

    if let Some(adopted) = adopted {
        set_data_layout_spec(world.borrow_mut().as_operation_mut(), adopted);
    }
    let block = WorldBuilder::new(world)
        .world
        .borrow()
        .body()
        .entry_block_ref()
        .expect("a world builder gives its world an entry block");
    for mut unit in units {
        unit.borrow_mut().remove();
        unit.insert_at_end(block);
    }
    Ok(())
}

/// The data layout of a world file against the session's world: `Some` when the file states one
/// and the world has none, so the world takes it; an error when both state one and they differ.
fn layout_to_adopt(
    name: &str,
    file: &World,
    world: WorldRef,
) -> Result<Option<DataLayoutSpec>, Report> {
    let Some(stated) = file.data_layout_spec() else {
        return Ok(None);
    };
    let world = world.borrow();
    match world.data_layout_spec() {
        None => Ok(Some(stated)),
        Some(current) if current == stated => Ok(None),
        Some(_) => Err(Report::msg(format!(
            "'{name}' states the data layout {}, but the world's is {}",
            show_layout(file.as_operation()),
            show_layout(world.as_operation())
        ))),
    }
}

/// Fails when `unit`, a module or a component, states a data layout other than `layout`, the one
/// of the world it is added to. A unit may state the world's layout again, and may state one of
/// its own when the world has none.
fn check_unit_layout(
    input: &str,
    unit: &Operation,
    layout: Option<&DataLayoutSpec>,
    world: WorldRef,
) -> Result<(), Report> {
    let stated = unit
        .as_trait::<dyn DataLayoutOpInterface>()
        .and_then(|unit| unit.data_layout_spec());
    match (stated, layout) {
        (Some(stated), Some(layout)) if stated != *layout => {
            let world = world.borrow();
            let of_world = match world.as_operation().get_attribute(DATA_LAYOUT_ATTR) {
                Some(_) => show_layout(world.as_operation()),
                None => String::from("the one this file's world states"),
            };
            Err(Report::msg(format!(
                "a '{}' of '{input}' states the data layout {}, but the world's is {of_world}",
                unit.name(),
                show_layout(unit)
            )))
        }
        _ => Ok(()),
    }
}

/// The layout `op` states, as `#builtin.data_layout<..>` is written.
fn show_layout(op: &Operation) -> String {
    match op.get_attribute(DATA_LAYOUT_ATTR) {
        Some(attr) => {
            let flags = OpPrintingFlags::default();
            let mut printer = AsmPrinter::root(op.context_rc(), &flags);
            printer.print_attribute_value(&*attr.borrow());
            printer.finish().to_string()
        }
        None => String::from("none"),
    }
}

/// Fails when `unit` cannot be defined in `world` beside what it holds.
fn check_definable(input: &str, unit: OperationRef, world: WorldRef) -> Result<(), Report> {
    let operation = unit.borrow();
    let symbol = operation
        .as_trait::<dyn Symbol>()
        .expect("a module and a component are symbols");
    let name = symbol.name();
    let world = world.borrow();
    if operation.is::<Component>() {
        world.reject_component_shadowing(name).map_err(|report| {
            Report::msg(format!(
                "'{input}' defines '{name}', which the world cannot take: {report}"
            ))
        })?;
    }
    (&*world as &dyn SymbolTable)
        .can_define(name, &operation.name())
        .map_err(|conflict| {
            Report::msg(format!(
                "'{input}' defines '{name}', which the world cannot take: {conflict}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use runic_core::Op;

    use crate::{
        CheckpointDecl, CompilationRequest, Compiler, Input, Session,
        frontends::IrFrontend,
        testing::{TestBackend, decl},
    };

    const OUT: &[CheckpointDecl] = &[decl("one.out", &[], "o")];
    const BACKEND: TestBackend = TestBackend::new("one", &["wasm32-wasip2"], OUT);

    fn compiler() -> Compiler {
        Compiler::builder().frontend(IrFrontend).backend(BACKEND).build().unwrap()
    }

    fn session<'c>(compiler: &'c Compiler, inputs: Vec<Input>, target: bool) -> Session<'c> {
        let mut request = CompilationRequest::new();
        request.inputs = inputs;
        if target {
            request.target = Some("wasm32-wasip2".parse().unwrap());
        } else {
            request.stop_after = Some("ir.translated".into());
        }
        compiler.session(request).unwrap()
    }

    fn printed(session: &Session<'_>) -> String {
        session.world().borrow().as_operation().to_string()
    }

    fn failure(inputs: Vec<Input>, target: bool) -> String {
        let compiler = compiler();
        let mut session = session(&compiler, inputs, target);
        session.translate(&mut ()).unwrap_err().to_string()
    }

    fn world_of(inputs: Vec<Input>, target: bool) -> String {
        let compiler = compiler();
        let mut session = session(&compiler, inputs, target);
        let _ = session.translate(&mut ()).unwrap();
        printed(&session)
    }

    fn runic(name: &str, text: &str) -> Input {
        Input::memory(name, text)
    }

    const LAYOUT: &str = "attributes { data_layout = #builtin.data_layout<little, ptr = 32> }";
    const BIG_LAYOUT: &str = "attributes { data_layout = #builtin.data_layout<big, ptr = 64> }";

    fn world_with(children: &str, attributes: &str) -> String {
        format!("builtin.world {{ {children} }} {attributes};")
    }

    fn positions(text: &str, needles: &[&str]) -> Vec<usize> {
        needles
            .iter()
            .map(|needle| text.find(needle).unwrap_or_else(|| panic!("no '{needle}' in\n{text}")))
            .collect()
    }

    #[test]
    fn a_module_is_added_to_the_world() {
        let text = world_of(
            vec![runic(
                "m.runic",
                "builtin.module public @m { builtin.function public extern(\"C\") @f(%x: i32) -> \
                 i32 { builtin.ret %x : (i32); }; };",
            )],
            false,
        );
        assert!(text.contains("builtin.module public @m {"), "{text}");
        assert!(text.contains("builtin.function public extern(\"C\") @f("), "{text}");
    }

    #[test]
    fn a_module_calling_a_function_of_its_own_by_symbol_verifies_in_the_world() {
        let text = world_of(
            vec![runic(
                "m.runic",
                "builtin.module public @m { builtin.function public extern(\"C\") @callee(i32) -> \
                 i32; builtin.function public extern(\"C\") @f(%x: i32) -> i32 { %c = \
                 builtin.call @callee(%x) : extern(\"C\") (i32) -> i32; builtin.ret %c : (i32); \
                 }; };",
            )],
            false,
        );
        assert!(text.contains("builtin.call @callee("), "{text}");
    }

    #[test]
    fn a_component_is_added_to_the_world() {
        let text =
            world_of(vec![runic("c.runic", "builtin.component public @acme::@app { };")], false);
        assert!(text.contains("builtin.component public @acme::@app"), "{text}");
    }

    #[test]
    fn two_files_land_in_one_world_in_input_order() {
        let text = world_of(
            vec![
                runic("second.runic", "builtin.module public @second { };"),
                runic("first.runic", "builtin.component public @first { };"),
            ],
            false,
        );
        let at = positions(&text, &["@second", "@first"]);
        assert!(at[0] < at[1], "{text}");
    }

    #[test]
    fn the_children_of_a_world_file_land_in_the_session_world_in_order() {
        let text = world_of(
            vec![runic(
                "w.runic",
                &world_with(
                    "builtin.module public @z { }; builtin.component public @a { }; \
                     builtin.module public @m { };",
                    "",
                ),
            )],
            false,
        );
        let at = positions(&text, &["@z", "@a", "@m"]);
        assert!(at[0] < at[1] && at[1] < at[2], "{text}");
        assert_eq!(text.matches("builtin.world").count(), 1, "one world, not two:\n{text}");
    }

    #[test]
    fn a_world_file_after_a_module_adds_to_the_same_world() {
        let text = world_of(
            vec![
                runic("m.runic", "builtin.module public @first { };"),
                runic("w.runic", &world_with("builtin.module public @second { };", "")),
            ],
            false,
        );
        let at = positions(&text, &["@first", "@second"]);
        assert!(at[0] < at[1], "{text}");
        assert_eq!(text.matches("builtin.world").count(), 1, "{text}");
    }

    #[test]
    fn a_world_file_whose_data_layout_equals_the_targets_is_accepted() {
        let text = world_of(
            vec![runic("w.runic", &world_with("builtin.module public @m { };", LAYOUT))],
            true,
        );
        assert!(text.contains("#builtin.data_layout<little, ptr = 32>"), "{text}");
        assert!(text.contains("@m"), "{text}");
    }

    #[test]
    fn a_world_file_whose_data_layout_differs_from_the_targets_is_an_error_naming_both() {
        let message = failure(
            vec![runic("w.runic", &world_with("builtin.module public @m { };", BIG_LAYOUT))],
            true,
        );
        assert!(message.contains("'w.runic'"), "{message}");
        assert!(message.contains("#builtin.data_layout<big, ptr = 64>"), "{message}");
        assert!(message.contains("#builtin.data_layout<little, ptr = 32>"), "{message}");
    }

    #[test]
    fn a_world_file_stating_a_layout_gives_it_to_a_world_that_has_none() {
        let text = world_of(
            vec![runic("w.runic", &world_with("builtin.module public @m { };", BIG_LAYOUT))],
            false,
        );
        assert!(text.contains("#builtin.data_layout<big, ptr = 64>"), "{text}");
    }

    #[test]
    fn a_second_world_file_must_agree_with_the_layout_the_first_stated() {
        let first = runic("one.runic", &world_with("builtin.module public @a { };", LAYOUT));
        let same = runic("two.runic", &world_with("builtin.module public @b { };", LAYOUT));
        let text = world_of(vec![first.clone(), same], false);
        assert!(text.contains("#builtin.data_layout<little, ptr = 32>"), "{text}");

        let other = runic("three.runic", &world_with("builtin.module public @c { };", BIG_LAYOUT));
        let message = failure(vec![first, other], false);
        assert!(message.contains("'three.runic'"), "{message}");
        assert!(
            message.contains("big, ptr = 64") && message.contains("little, ptr = 32"),
            "{message}"
        );
    }

    #[test]
    fn a_world_file_without_a_layout_leaves_the_targets_alone() {
        let text = world_of(
            vec![runic("w.runic", &world_with("builtin.module public @m { };", ""))],
            true,
        );
        assert!(text.contains("#builtin.data_layout<little, ptr = 32>"), "{text}");
    }

    #[test]
    fn a_function_at_the_top_level_is_an_error_naming_it_and_the_three_kinds_of_unit() {
        let message = failure(
            vec![runic(
                "f.runic",
                "builtin.function public extern(\"C\") @f(%x: i32) -> i32 { builtin.ret %x : \
                 (i32); };",
            )],
            false,
        );
        assert_eq!(
            message,
            "'f.runic' holds a 'builtin.function' at its top level: a unit is a 'builtin.module', \
             a 'builtin.component' or a 'builtin.world'"
        );
    }

    #[test]
    fn a_function_in_the_world_of_a_world_file_is_an_error_that_says_what_a_world_holds() {
        let message = failure(
            vec![runic(
                "w.runic",
                &world_with("builtin.function public extern(\"C\") @f() { builtin.ret; };", ""),
            )],
            false,
        );
        assert_eq!(
            message,
            "the world of 'w.runic' holds a 'builtin.function': a world file holds \
             'builtin.module' and 'builtin.component' operations"
        );
    }

    #[test]
    fn a_module_that_states_another_layout_than_the_targets_is_an_error_naming_both() {
        let module = format!("builtin.module public @m {BIG_LAYOUT} {{ }};");
        let message = failure(vec![runic("m.runic", &module)], true);
        assert!(message.contains("a 'builtin.module' of 'm.runic'"), "{message}");
        assert!(message.contains("#builtin.data_layout<big, ptr = 64>"), "{message}");
        assert!(message.contains("#builtin.data_layout<little, ptr = 32>"), "{message}");
    }

    #[test]
    fn a_module_may_state_the_targets_layout_again_or_its_own_when_there_is_no_target() {
        let same = format!("builtin.module public @m {LAYOUT} {{ }};");
        let text = world_of(vec![runic("m.runic", &same)], true);
        assert!(text.contains("@m"), "{text}");

        let own = format!("builtin.module public @m {BIG_LAYOUT} {{ }};");
        let text = world_of(vec![runic("m.runic", &own)], false);
        assert!(text.contains("#builtin.data_layout<big, ptr = 64>"), "{text}");
    }

    #[test]
    fn a_module_in_a_world_file_must_agree_with_the_layout_that_file_states() {
        let module = format!("builtin.module public @m {BIG_LAYOUT} {{ }};");
        let message = failure(vec![runic("w.runic", &world_with(&module, LAYOUT))], false);
        assert!(message.contains("a 'builtin.module' of 'w.runic'"), "{message}");
        assert!(message.contains("the one this file's world states"), "{message}");
    }

    #[test]
    fn a_syntax_error_is_the_parsers_report_with_its_source() {
        let compiler = compiler();
        let mut session =
            session(&compiler, vec![runic("bad.runic", "builtin.module public @m {")], false);
        let message = format!("{:?}", session.translate(&mut ()).unwrap_err());
        assert!(message.contains("bad.runic"), "{message}");
        assert!(message.contains("builtin.module public @m {"), "the source line: {message}");
    }

    #[test]
    fn two_files_defining_one_symbol_are_an_error_naming_the_symbol_and_the_second_input() {
        let message = failure(
            vec![
                runic("one.runic", "builtin.module public @m { };"),
                runic("two.runic", "builtin.module public @m { };"),
            ],
            false,
        );
        assert!(message.contains("'two.runic'") && message.contains("'m'"), "{message}");
        assert!(message.contains("redefinition of symbol m"), "{message}");
    }

    #[test]
    fn a_component_that_would_shadow_a_module_tree_is_an_error() {
        let message = failure(
            vec![
                runic(
                    "one.runic",
                    "builtin.module public @acme { builtin.module public @a { }; };",
                ),
                runic("two.runic", "builtin.component public @acme::@a { };"),
            ],
            false,
        );
        assert!(message.contains("'two.runic'") && message.contains("'acme::a'"), "{message}");
    }

    #[test]
    fn a_world_file_defining_a_symbol_the_world_has_is_an_error_and_adds_nothing() {
        let compiler = compiler();
        let mut session = session(
            &compiler,
            vec![
                runic("one.runic", "builtin.module public @m { };"),
                runic(
                    "two.runic",
                    &world_with(
                        "builtin.module public @fresh { }; builtin.module public @m { };",
                        "",
                    ),
                ),
            ],
            false,
        );
        let message = session.translate(&mut ()).unwrap_err().to_string();
        assert!(message.contains("'two.runic'") && message.contains("'m'"), "{message}");
        assert!(!printed(&session).contains("@fresh"), "nothing of a failed file is added");
    }

    #[test]
    fn a_path_input_is_read_through_the_session_and_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unit.runic");
        std::fs::write(&path, "builtin.module public @from_disk { };").unwrap();
        let compiler = compiler();
        let mut session = session(&compiler, vec![Input::path(&path)], false);
        let _ = session.translate(&mut ()).unwrap();
        assert_eq!(session.files_read(), [path]);
        assert!(printed(&session).contains("@from_disk"));
    }

    #[test]
    fn input_that_is_not_utf_8_is_an_error_naming_the_input() {
        let message = failure(vec![Input::memory("bin.runic", vec![0xff, 0xfe])], false);
        assert!(message.contains("'bin.runic'") && message.contains("UTF-8"), "{message}");
    }

    #[test]
    fn a_reference_to_a_symbol_of_another_file_is_linked_whatever_the_order() {
        let a = runic(
            "a.runic",
            "builtin.module public @a { builtin.function public extern(\"C\") @callee(i32) -> \
             i32; };",
        );
        let b = runic(
            "b.runic",
            "builtin.module public @b { builtin.function public extern(\"C\") @f(%x: i32) -> i32 \
             { %c = builtin.call ::@a::@callee(%x) : extern(\"C\") (i32) -> i32; builtin.ret %c : \
             (i32); }; };",
        );
        let alone = failure(vec![b.clone()], false);
        assert!(alone.contains("'::@a::@callee' does not resolve"), "{alone}");
        for inputs in [vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]] {
            let text = world_of(inputs, false);
            assert!(text.contains("@callee") && text.contains("@f"), "{text}");
        }

        // With verification off the reference is linked all the same.
        let compiler = compiler();
        let mut request = CompilationRequest::new();
        request.inputs = vec![b, a];
        request.stop_after = Some("ir.translated".into());
        request.ir.verification = crate::Verification::None;
        let mut session = compiler.session(request).unwrap();
        let _ = session.translate(&mut ()).unwrap();
        session.world().as_operation_ref().borrow().recursively_verify().unwrap();
    }

    #[test]
    fn the_frontend_is_named_runic_and_claims_the_runic_extension_without_checkpoints() {
        use crate::Frontend;
        assert_eq!(IrFrontend.name(), "runic");
        assert_eq!(IrFrontend.extensions(), ["runic"]);
        assert!(IrFrontend.checkpoints().is_empty());
    }
}
