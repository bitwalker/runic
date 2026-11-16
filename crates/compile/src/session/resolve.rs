//! The decisions a session makes before any IR exists: which frontend reads each input, and which
//! target and backend the compilation is for.

use runic_ir::Report;

use super::{Participant, Target};
use crate::{Compiler, Frontend, Input, Triple};

/// The names of `names` as a list for a message: `'a', 'b'`, or `none`.
fn list<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let names: Vec<String> = names.into_iter().map(|name| format!("'{name}'")).collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

/// The names of the compiler's frontends, for a message.
fn languages(compiler: &Compiler) -> String {
    list(compiler.frontends().map(|frontend| frontend.name()))
}

/// The names of the compiler's backends, for a message.
fn backends(compiler: &Compiler) -> String {
    list(compiler.backends().map(|backend| backend.name()))
}

/// The targets of `triples`, for a message.
fn targets(triples: &[Triple]) -> String {
    list(triples.iter().map(Triple::as_str))
}

/// Gives each input to the one frontend that reads it.
///
/// An input that names a language goes to the frontend of that name; any other goes to the
/// frontend that claims its extension. The frontends are returned in the order each first occurs
/// among the inputs, each with its inputs in the order given.
pub(super) fn assign_inputs<'c>(
    compiler: &'c Compiler,
    inputs: &[Input],
) -> Result<Vec<Participant<'c>>, Report> {
    if inputs.is_empty() {
        return Err(Report::msg("no input was given: give at least one input to compile"));
    }
    let mut groups: Vec<Participant<'c>> = Vec::new();
    for input in inputs {
        let frontend = frontend_of(compiler, input)?;
        match groups.iter_mut().find(|(other, _)| other.name() == frontend.name()) {
            Some((_, group)) => group.push(input.clone()),
            None => groups.push((frontend, vec![input.clone()])),
        }
    }
    Ok(groups)
}

/// The frontend that reads `input`.
fn frontend_of<'c>(compiler: &'c Compiler, input: &Input) -> Result<&'c dyn Frontend, Report> {
    let name = input.name();
    if let Some(language) = &input.language {
        return compiler.frontend(language).ok_or_else(|| {
            if compiler.frontends().next().is_none() {
                Report::msg(format!(
                    "the input '{name}' is given as the language '{language}', but this compiler \
                     has no frontend"
                ))
            } else {
                Report::msg(format!(
                    "the input '{name}' is given as the language '{language}', which this \
                     compiler does not read; its languages are {}",
                    languages(compiler)
                ))
            }
        });
    }
    let Some(extension) = input.extension() else {
        return Err(Report::msg(format!(
            "the input '{name}' has no file extension, so its language cannot be inferred: give \
             it with --language; the languages are {}",
            languages(compiler)
        )));
    };
    compiler.frontend_for_extension(extension).ok_or_else(|| {
        let known =
            list(compiler.frontends().flat_map(|frontend| frontend.extensions().iter().copied()));
        Report::msg(format!(
            "the input '{name}' has the extension '{extension}', which no frontend claims; the \
             known extensions are {known}, or give the language with --language"
        ))
    })
}

/// Decides the target of a compilation and the backend that lowers for it.
///
/// The triple is `target`, else the compiler's default target. `None` is a compilation with no
/// backend, which is allowed only when the compiler has none or `stopping` says the compilation
/// stops before a backend is needed. The host is never consulted.
pub(super) fn resolve_target<'c>(
    compiler: &'c Compiler,
    target: Option<&Triple>,
    backend: Option<&str>,
    stopping: bool,
) -> Result<Option<Target<'c>>, Report> {
    let named = match backend {
        None => None,
        Some(name) => Some(compiler.backend(name).ok_or_else(|| {
            Report::msg(format!(
                "unknown backend '{name}' given to --backend; the backends are {}",
                backends(compiler)
            ))
        })?),
    };
    let Some(triple) = target.or_else(|| compiler.default_target()) else {
        if let Some(backend) = named {
            return Err(Report::msg(format!(
                "the backend '{}' was named but no target was given: give one with --target; the \
                 targets of '{}' are {}",
                backend.name(),
                backend.name(),
                targets(&backend.targets())
            )));
        }
        if compiler.backends().next().is_some() && !stopping {
            return Err(Report::msg(format!(
                "no target was given: give one with --target, or stop before the backend with \
                 --stop-after ir.optimized; the targets are {}",
                targets(&compiler.targets())
            )));
        }
        return Ok(None);
    };

    // Said of the triple in a message when the request did not give it.
    let origin = if target.is_some() {
        ""
    } else {
        " (the compiler's default)"
    };
    if compiler.backends().next().is_none() {
        return Err(Report::msg(format!(
            "the target '{triple}' was given, but this compiler has no backend: a target is what \
             a backend generates code for"
        )));
    }
    let backend = match named {
        Some(backend) => {
            if !backend.supports(triple) {
                return Err(Report::msg(format!(
                    "the backend '{}' does not support the target '{triple}'{origin}; its targets \
                     are {}",
                    backend.name(),
                    targets(&backend.targets())
                )));
            }
            backend
        }
        None => {
            let candidates: Vec<_> =
                compiler.backends().filter(|backend| backend.supports(triple)).collect();
            match candidates.as_slice() {
                [] => {
                    return Err(Report::msg(format!(
                        "no backend supports the target '{triple}'{origin}; the targets are {}",
                        targets(&compiler.targets())
                    )));
                }
                [only] => *only,
                several => {
                    return Err(Report::msg(format!(
                        "more than one backend supports the target '{triple}'{origin}: choose one \
                         with --backend; the candidates are {}",
                        list(several.iter().map(|backend| backend.name()))
                    )));
                }
            }
        }
    };
    let data_layout = backend.data_layout(triple).map_err(|report| {
        report.context(format!(
            "the backend '{}' cannot state the data layout of the target '{triple}'",
            backend.name()
        ))
    })?;
    Ok(Some(Target {
        triple: triple.clone(),
        backend,
        data_layout,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Backend,
        testing::{TestBackend, TestFrontend, decl},
    };

    const A: TestFrontend = TestFrontend::new("alpha", &["a", "alp"]);
    const B: TestFrontend = TestFrontend::new("beta", &["b"]);
    const ONE: TestBackend = TestBackend::new(
        "one",
        &["wasm32-unknown-unknown", "wasm32-wasip2"],
        &[decl("one.out", &[], "o")],
    );
    const TWO: TestBackend = TestBackend::new(
        "two",
        &["wasm32-wasip2", "riscv64-unknown-none"],
        &[decl("two.out", &[], "t")],
    );

    fn frontends() -> Compiler {
        Compiler::builder().frontend(A).frontend(B).build().unwrap()
    }

    fn grouped(compiler: &Compiler, inputs: &[Input]) -> Vec<(String, Vec<String>)> {
        assign_inputs(compiler, inputs)
            .unwrap()
            .into_iter()
            .map(|(frontend, inputs)| {
                (
                    frontend.name().to_owned(),
                    inputs.iter().map(|i| i.name().into_owned()).collect(),
                )
            })
            .collect()
    }

    fn input_error(compiler: &Compiler, inputs: &[Input]) -> String {
        match assign_inputs(compiler, inputs) {
            Ok(_) => panic!("the inputs should be rejected"),
            Err(error) => error.to_string(),
        }
    }

    fn triple(text: &str) -> Triple {
        text.parse().unwrap()
    }

    fn resolved(
        compiler: &Compiler,
        target: Option<&str>,
        backend: Option<&str>,
        stopping: bool,
    ) -> Option<(String, String)> {
        let target = target.map(triple);
        resolve_target(compiler, target.as_ref(), backend, stopping)
            .unwrap()
            .map(|t| (t.triple().to_string(), t.backend().name().to_owned()))
    }

    fn target_error(
        compiler: &Compiler,
        target: Option<&str>,
        backend: Option<&str>,
        stopping: bool,
    ) -> String {
        let target = target.map(triple);
        match resolve_target(compiler, target.as_ref(), backend, stopping) {
            Ok(_) => panic!("the target should be rejected"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn no_input_at_all_is_an_error() {
        let message = input_error(&frontends(), &[]);
        assert!(message.contains("no input was given"), "{message}");
    }

    #[test]
    fn an_input_goes_to_the_frontend_that_claims_its_extension() {
        let groups = grouped(&frontends(), &[Input::memory("x.alp", ""), Input::memory("y.b", "")]);
        assert_eq!(
            groups,
            [("alpha".into(), vec!["x.alp".into()]), ("beta".into(), vec!["y.b".into()])]
        );
    }

    #[test]
    fn an_input_that_names_a_language_goes_to_that_frontend_whatever_its_extension() {
        let input = Input::memory("x.b", "").with_language("alpha");
        assert_eq!(grouped(&frontends(), &[input]), [("alpha".into(), vec!["x.b".into()])]);
    }

    #[test]
    fn an_input_with_no_extension_goes_to_the_frontend_it_names() {
        let input = Input::memory("Makefile", "").with_language("beta");
        assert_eq!(grouped(&frontends(), &[input]), [("beta".into(), vec!["Makefile".into()])]);
    }

    #[test]
    fn frontends_come_in_the_order_they_first_occur_and_each_has_its_inputs_in_order() {
        let inputs = [
            Input::memory("1.b", ""),
            Input::memory("2.a", ""),
            Input::memory("3.b", ""),
            Input::memory("4.alp", ""),
        ];
        assert_eq!(
            grouped(&frontends(), &inputs),
            [
                ("beta".into(), vec!["1.b".into(), "3.b".into()]),
                ("alpha".into(), vec!["2.a".into(), "4.alp".into()]),
            ]
        );
    }

    #[test]
    fn a_frontend_with_no_input_does_not_participate() {
        let groups = grouped(&frontends(), &[Input::memory("x.b", "")]);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn an_unknown_language_is_an_error_naming_the_input_and_listing_the_frontends() {
        let message = input_error(&frontends(), &[Input::memory("x.a", "").with_language("gamma")]);
        assert!(message.contains("'x.a'") && message.contains("'gamma'"), "{message}");
        assert!(message.contains("'alpha', 'beta'"), "{message}");
    }

    #[test]
    fn an_unknown_language_on_a_compiler_with_no_frontend_says_it_has_none() {
        let compiler = Compiler::builder().build().unwrap();
        let message = input_error(&compiler, &[Input::memory("x.a", "").with_language("gamma")]);
        assert!(message.contains("'gamma'") && message.contains("has no frontend"), "{message}");
    }

    #[test]
    fn an_input_with_no_extension_and_no_language_asks_for_language() {
        let message = input_error(&frontends(), &[Input::memory("Makefile", "")]);
        assert!(message.contains("'Makefile'") && message.contains("--language"), "{message}");
        assert!(message.contains("'alpha', 'beta'"), "{message}");
    }

    #[test]
    fn an_extension_nobody_claims_is_an_error_naming_it_and_listing_the_known_ones() {
        let message = input_error(&frontends(), &[Input::memory("x.zzz", "")]);
        assert!(message.contains("'x.zzz'") && message.contains("'zzz'"), "{message}");
        assert!(message.contains("'a', 'alp', 'b'"), "{message}");
    }

    #[test]
    fn an_extension_is_matched_exactly_so_a_capital_is_not_claimed() {
        let message = input_error(&frontends(), &[Input::memory("x.A", "")]);
        assert!(message.contains("'A'"), "{message}");
    }

    #[test]
    fn with_a_target_one_backend_that_supports_it_is_chosen() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        let got = resolved(&compiler, Some("wasm32-unknown-unknown"), None, false);
        assert_eq!(got, Some(("wasm32-unknown-unknown".into(), "one".into())));
        let got = resolved(&compiler, Some("riscv64-unknown-none"), None, false);
        assert_eq!(got, Some(("riscv64-unknown-none".into(), "two".into())));
    }

    #[test]
    fn the_default_target_is_used_when_the_request_names_none() {
        let compiler = Compiler::builder()
            .backend(ONE)
            .backend(TWO)
            .default_target("riscv64-unknown-none")
            .build()
            .unwrap();
        let got = resolved(&compiler, None, None, false);
        assert_eq!(got, Some(("riscv64-unknown-none".into(), "two".into())));
        let got = resolved(&compiler, Some("wasm32-unknown-unknown"), None, false);
        assert_eq!(got, Some(("wasm32-unknown-unknown".into(), "one".into())));
    }

    #[test]
    fn a_target_no_backend_supports_is_an_error_listing_the_known_targets() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        let message = target_error(&compiler, Some("mips-unknown-none"), None, false);
        assert!(message.contains("'mips-unknown-none'"), "{message}");
        assert!(
            message.contains("'wasm32-unknown-unknown', 'wasm32-wasip2', 'riscv64-unknown-none'"),
            "{message}"
        );
    }

    #[test]
    fn a_target_several_backends_support_requires_backend_and_lists_the_candidates() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        let message = target_error(&compiler, Some("wasm32-wasip2"), None, false);
        assert!(message.contains("--backend") && message.contains("'one', 'two'"), "{message}");
    }

    #[test]
    fn a_named_backend_settles_a_target_several_backends_support() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        let got = resolved(&compiler, Some("wasm32-wasip2"), Some("two"), false);
        assert_eq!(got, Some(("wasm32-wasip2".into(), "two".into())));
    }

    #[test]
    fn a_named_backend_that_does_not_exist_is_an_error_listing_the_backends() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        let message = target_error(&compiler, Some("wasm32-wasip2"), Some("three"), false);
        assert!(message.contains("'three'") && message.contains("'one', 'two'"), "{message}");
    }

    #[test]
    fn a_named_backend_that_does_not_support_the_target_is_an_error_listing_its_targets() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        let message = target_error(&compiler, Some("riscv64-unknown-none"), Some("one"), false);
        assert!(
            message.contains("'one'") && message.contains("'riscv64-unknown-none'"),
            "{message}"
        );
        assert!(message.contains("'wasm32-unknown-unknown', 'wasm32-wasip2'"), "{message}");
    }

    #[test]
    fn a_backend_named_with_no_target_asks_for_target_and_lists_the_backends_targets() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        let message = target_error(&compiler, None, Some("two"), false);
        assert!(message.contains("--target") && message.contains("'two'"), "{message}");
        assert!(message.contains("'wasm32-wasip2', 'riscv64-unknown-none'"), "{message}");
    }

    #[test]
    fn a_backend_named_with_no_target_and_a_stop_point_is_still_an_error() {
        let compiler = Compiler::builder().backend(ONE).build().unwrap();
        let message = target_error(&compiler, None, Some("one"), true);
        assert!(message.contains("--target"), "{message}");
    }

    #[test]
    fn an_unknown_backend_named_with_no_target_is_the_unknown_backend_error() {
        let compiler = Compiler::builder().backend(ONE).build().unwrap();
        let message = target_error(&compiler, None, Some("nope"), false);
        assert!(
            message.contains("unknown backend 'nope'") && message.contains("'one'"),
            "{message}"
        );
    }

    #[test]
    fn with_no_backend_in_the_compiler_no_target_is_fine() {
        let compiler = frontends();
        assert!(resolved(&compiler, None, None, false).is_none());
        assert!(resolved(&compiler, None, None, true).is_none());
    }

    #[test]
    fn with_backends_and_no_target_and_no_stop_point_the_error_says_how_to_stop_early() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        let message = target_error(&compiler, None, None, false);
        assert!(
            message.contains("no target was given") && message.contains("--target"),
            "{message}"
        );
        assert!(message.contains("--stop-after ir.optimized"), "{message}");
        assert!(message.contains("'riscv64-unknown-none'"), "{message}");
    }

    #[test]
    fn with_backends_and_no_target_and_a_stop_point_there_is_no_target() {
        let compiler = Compiler::builder().backend(ONE).backend(TWO).build().unwrap();
        assert!(resolved(&compiler, None, None, true).is_none());
    }

    #[test]
    fn a_target_on_a_compiler_with_no_backend_is_an_error_that_says_so() {
        let message = target_error(&frontends(), Some("wasm32-wasip2"), None, false);
        assert_eq!(
            message,
            "the target 'wasm32-wasip2' was given, but this compiler has no backend: a target is \
             what a backend generates code for"
        );
    }

    #[test]
    fn the_target_carries_the_backends_data_layout() {
        let compiler = Compiler::builder().backend(ONE).build().unwrap();
        let target = triple("wasm32-wasip2");
        let target = resolve_target(&compiler, Some(&target), None, false).unwrap().unwrap();
        assert_eq!(*target.data_layout(), ONE.data_layout(target.triple()).unwrap());
    }
}
