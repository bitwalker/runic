//! The pass registry as seen by a program that links every pass of the workspace.

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use runic_core::{
    Context,
    pass::{
        Nesting, OpPassManager, OperationPass,
        registry::{PassInfo, PassPipelineInfo, RegistryEntry, register_pass_pipeline},
    },
};
// `cfg-to-scf` is registered by the `scf` crate, which the test links by naming its pass.
use runic_dialect_scf::transforms::LiftControlFlowToSCF;

use crate::{
    Canonicalizer, CommonSubexpressionElimination, ControlFlowSink, DeadCodeElimination, Local2Reg,
    SinkOperandDefs, SparseConditionalConstantPropagation,
};

const EXPECTED: [&str; 8] = [
    "canonicalizer",
    "cfg-to-scf",
    "control-flow-sink",
    "cse",
    "dce",
    "local2reg",
    "sccp",
    "sink-operand-defs",
];

#[test]
fn the_registry_lists_the_passes_of_the_workspace() {
    let registered = PassInfo::registered();
    let names = registered.iter().map(PassInfo::argument).collect::<Vec<_>>();

    // Other tests may register passes of their own: check for the eight, not for only the eight.
    for expected in EXPECTED {
        assert!(names.contains(&expected), "pass '{expected}' is not registered: {names:?}");
    }
    assert!(names.is_sorted(), "{names:?}");
    for pass in &registered {
        assert!(!pass.argument().is_empty());
        assert!(!pass.description().is_empty(), "pass '{}' has no description", pass.argument());
        assert!(
            !pass.description().ends_with('.') && !pass.description().contains('\n'),
            "the description of '{}' is not one line: {}",
            pass.argument(),
            pass.description()
        );
    }
}

#[test]
fn a_description_is_the_same_in_the_registry_and_in_the_pass() {
    let passes: [Box<dyn OperationPass>; 8] = [
        Box::<Canonicalizer>::default(),
        Box::<LiftControlFlowToSCF>::default(),
        Box::<ControlFlowSink>::default(),
        Box::<CommonSubexpressionElimination>::default(),
        Box::<DeadCodeElimination>::default(),
        Box::<Local2Reg>::default(),
        Box::<SparseConditionalConstantPropagation>::default(),
        Box::<SinkOperandDefs>::default(),
    ];
    for pass in passes {
        let info = PassInfo::lookup(pass.argument())
            .unwrap_or_else(|| panic!("pass '{}' is not registered", pass.argument()));
        assert_eq!(info.description(), pass.description(), "pass '{}'", pass.argument());
        assert!(EXPECTED.contains(&pass.argument()));
    }
}

#[test]
fn a_dynamically_registered_pipeline_is_listed() {
    fn build() -> Box<dyn OperationPass> {
        Box::<DeadCodeElimination>::default()
    }
    for name in ["registry-test-pipeline-b", "registry-test-pipeline-a"] {
        register_pass_pipeline(name, "a pipeline of the registry tests", build).unwrap();
    }
    let registered = PassPipelineInfo::registered();
    let names = registered.iter().map(RegistryEntry::argument).collect::<Vec<_>>();
    assert!(names.contains(&"registry-test-pipeline-a"), "{names:?}");
    assert!(names.contains(&"registry-test-pipeline-b"), "{names:?}");
    assert!(names.is_sorted(), "{names:?}");
    let info = PassPipelineInfo::lookup("registry-test-pipeline-a").unwrap();
    assert_eq!(info.description(), "a pipeline of the registry tests");
}

#[test]
fn a_misspelled_name_is_close_to_the_pass_meant() {
    assert_eq!(PassInfo::similar_to("cze").first(), Some(&"cse"));
    assert_eq!(PassInfo::similar_to("dc").first(), Some(&"dce"));
    assert!(PassInfo::similar_to("cse-").contains(&"cse"));
    assert!(PassInfo::similar_to("not-a-pass-at-all").is_empty());
    assert!(PassInfo::similar_to("cze").len() <= 3);
}

struct Pipeline<'a>(&'a OpPassManager);

impl fmt::Display for Pipeline<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.print_as_textual_pipeline(f)
    }
}

#[test]
fn invalid_options_leave_the_pass_manager_unchanged() {
    let mut pm = OpPassManager::any(Nesting::Implicit, Rc::new(Context::default()));
    let canonicalizer = PassInfo::lookup("canonicalizer").unwrap();
    canonicalizer.add_to_pipeline(&mut pm, "").unwrap();
    let before = Pipeline(&pm).to_string();

    // A key without a value, and an unknown key.
    for options in ["enable-patterns", "no-such-option=1"] {
        let error = canonicalizer.add_to_pipeline(&mut pm, options).unwrap_err().to_string();
        assert!(error.contains(options.split('=').next().unwrap()), "{error}");
        assert_eq!(Pipeline(&pm).to_string(), before, "after the options '{options}'");
    }
}

/// Loading pipeline text and printing the manager gives the canonical text, which loads to a
/// manager that prints it again, for every registered pass and for the canonicalizer's options
/// in any order, spaced, with empty entries and with names that need escaping.
#[test]
fn loaded_pipelines_print_their_canonical_text() {
    use runic_core::pass::PassPipeline;

    fn load_and_print(text: &str) -> String {
        let pipeline = PassPipeline::parse(text).unwrap_or_else(|error| panic!("{text}: {error}"));
        let pm = pipeline
            .load(Rc::new(Context::default()), Nesting::Implicit)
            .unwrap_or_else(|error| panic!("{text}: {error}"));
        struct Printed<'a>(&'a runic_core::pass::PassManager);
        impl fmt::Display for Printed<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.print_as_textual_pipeline(f)
            }
        }
        Printed(&pm).to_string()
    }

    let mut cases = PassInfo::registered()
        .iter()
        .map(|pass| (format!("any({})", pass.argument()), None))
        .collect::<Vec<_>>();
    for (text, canonical) in [
        (
            r#"any(canonicalizer{disable-patterns="c" enable-patterns="a;b"})"#,
            r#"any(canonicalizer{enable-patterns="a;b" disable-patterns="c"})"#,
        ),
        (
            r#"any(canonicalizer{enable-patterns="a;b" disable-patterns="c"})"#,
            r#"any(canonicalizer{enable-patterns="a;b" disable-patterns="c"})"#,
        ),
        (
            r#"any(canonicalizer{enable-patterns=" a ;; b ; " disable-patterns=c;})"#,
            r#"any(canonicalizer{enable-patterns="a;b" disable-patterns="c"})"#,
        ),
        (
            r#"any(canonicalizer{disable-patterns="say \"hi\";back\\slash"})"#,
            r#"any(canonicalizer{disable-patterns="say \"hi\";back\\slash"})"#,
        ),
        (
            r#"builtin.module( canonicalizer { enable-patterns = x } , cse )"#,
            r#"builtin.module(canonicalizer{enable-patterns="x"},cse)"#,
        ),
    ] {
        cases.push((text.to_string(), Some(canonical)));
    }
    for (text, canonical) in cases {
        let printed = load_and_print(&text);
        if let Some(canonical) = canonical {
            assert_eq!(printed, canonical, "loading '{text}'");
        }
        assert_eq!(load_and_print(&printed), printed, "loading '{printed}', printed from '{text}'");
    }
}
