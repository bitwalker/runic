use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::{cell::RefCell, fmt};

use super::*;
use crate::{
    EntityMut, Operation, OperationRef,
    dialects::builtin::{Function, Module},
    parse::{ParserConfig, parse_any},
    pass::{
        OperationPass, Pass, PassExecutionState, PassInstrumentation, PassTarget, PostPassStatus,
        registry::register_pass,
    },
};

/// A pass with no options that runs on any operation.
#[derive(Default)]
struct Generic;

/// A second pass with no options that runs on any operation.
#[derive(Default)]
struct Other;

/// A pass that runs on functions only.
#[derive(Default)]
struct OnFunctions;

/// A pass with options in the form the canonicalizer takes them: `alpha`, `beta` and `gamma`
/// take a value and `flag` takes none. It keeps them in the order given and prints them back.
#[derive(Default)]
struct WithOptions {
    options: Vec<(String, Option<String>)>,
}

const KEYS: [&str; 4] = ["alpha", "beta", "gamma", "flag"];

macro_rules! no_op_pass {
    ($ty:ty, $target:ty, $argument:literal, $schedule:expr) => {
        impl Pass for $ty {
            type Target = $target;

            fn name(&self) -> &'static str {
                $argument
            }

            fn argument(&self) -> &'static str {
                $argument
            }

            fn description(&self) -> &'static str {
                "a pass of the pipeline tests"
            }

            fn can_schedule_on(&self, name: &OperationName) -> bool {
                let schedule: fn(&OperationName) -> bool = $schedule;
                schedule(name)
            }

            fn run_on_operation(
                &mut self,
                _op: EntityMut<'_, $target>,
                state: &mut PassExecutionState,
            ) -> Result<(), Report> {
                state.set_post_pass_status(PostPassStatus::Unchanged);
                Ok(())
            }
        }
    };
}

no_op_pass!(Generic, Operation, "pipeline-test-generic", |_| true);
no_op_pass!(Other, Operation, "pipeline-test-other", |_| true);
no_op_pass!(OnFunctions, Function, "pipeline-test-function", |name| name.is::<Function>());

impl Pass for WithOptions {
    type Target = Operation;

    fn name(&self) -> &'static str {
        "pipeline-test-options"
    }

    fn argument(&self) -> &'static str {
        "pipeline-test-options"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    /// The options come joined by `, `; a value never contains `,`, and is kept as it is.
    fn initialize_options(&mut self, options: &str) -> Result<(), Report> {
        if options.is_empty() {
            return Ok(());
        }
        for option in options.split(", ") {
            let (key, value) = match option.split_once('=') {
                Some((key, value)) => (key, Some(value.to_string())),
                None => (option, None),
            };
            if !KEYS.contains(&key) {
                return Err(Report::msg(format!("unknown option '{key}'")));
            }
            if (key == "flag") != value.is_none() {
                return Err(Report::msg(format!("option '{key}' given the wrong way")));
            }
            if self.options.iter().any(|(k, _)| k == key) {
                return Err(Report::msg(format!("option '{key}' given twice")));
            }
            self.options.push((key.to_string(), value));
        }
        Ok(())
    }

    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            SelectedPass {
                name: Pass::argument(self).to_string(),
                options: self.options.clone(),
            }
        )
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        state.set_post_pass_status(PostPassStatus::Unchanged);
        Ok(())
    }
}

fn boxed<P: Pass + Default>() -> Box<dyn OperationPass> {
    Box::<P>::default()
}

/// Registers the passes of these tests; registering them again does nothing.
fn register() {
    register_pass(boxed::<Generic>).unwrap();
    register_pass(boxed::<Other>).unwrap();
    register_pass(boxed::<OnFunctions>).unwrap();
    register_pass(boxed::<WithOptions>).unwrap();
}

/// The pipeline text a pass manager prints for itself.
fn printed(pm: &PassManager) -> String {
    struct Printed<'a>(&'a PassManager);
    impl fmt::Display for Printed<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.print_as_textual_pipeline(f)
        }
    }
    Printed(pm).to_string()
}

fn parse(text: &str) -> PassPipeline {
    PassPipeline::parse(text).unwrap_or_else(|error| panic!("{text}: {error}"))
}

fn parse_error(text: &str) -> String {
    match PassPipeline::parse(text) {
        Ok(pipeline) => panic!("expected '{text}' to be rejected, got '{pipeline}'"),
        Err(error) => error.to_string(),
    }
}

fn load_error(text: &str, nesting: Nesting) -> String {
    match parse(text).load(Rc::new(Context::default()), nesting) {
        Ok(pm) => panic!("expected '{text}' to fail to load, got '{}'", printed(&pm)),
        Err(error) => error.to_string(),
    }
}

#[test]
fn example_pipeline() {
    let text = "builtin.module(cse,builtin.function(canonicalizer{enable-patterns=\"a;b\"}))";
    assert_eq!(parse(text).to_string(), text);
    // Whitespace between tokens is not part of the pipeline.
    let spaced = " builtin.module ( cse , builtin.function( canonicalizer { enable-patterns = \
                  \"a;b\" } ) ) ";
    assert_eq!(parse(spaced), parse(text));
}

/// Bare, numeric and quoted option values print quoted, in source order, and parse back; a key
/// may stand without a value. Repeated keys, quoted keys and values with `,` are rejected.
#[test]
fn pass_options_roundtrip() {
    let source = "builtin.module(canonicalizer{enable-patterns=\"a;b\" disable-patterns=foo-bar \
                  depth=3 verbose})";
    let pipeline = parse(source);
    let printed = pipeline.to_string();
    assert_eq!(
        printed,
        "builtin.module(canonicalizer{enable-patterns=\"a;b\" disable-patterns=\"foo-bar\" \
         depth=\"3\" verbose})"
    );
    assert_eq!(parse(&printed), pipeline);
    let PipelineElement::Pass(pass) = &pipeline.elements()[0] else {
        panic!("expected a pass");
    };
    assert_eq!(
        pass.options_string(),
        "enable-patterns=a;b, disable-patterns=foo-bar, depth=3, verbose"
    );

    for (invalid, expected) in [
        (
            "builtin.module(canonicalizer{disable-patterns=a disable-patterns=b})",
            "expected each option of pass 'canonicalizer' at most once, found 'disable-patterns' \
             again at offset 48",
        ),
        (
            "builtin.module(canonicalizer{enable-patterns=\"a,b\"})",
            "expected a value for option 'enable-patterns' without ',', found a value that \
             contains ',' at offset 45",
        ),
        (
            "builtin.module(canonicalizer{\"a b\"=c})",
            "expected an option key or '}', found a quoted string at offset 29",
        ),
        (
            "builtin.module(canonicalizer{enable-patterns=})",
            "expected a value for option 'enable-patterns' (a word, number or quoted string), \
             found '}' at offset 45",
        ),
        (
            "builtin.module(canonicalizer{enable-patterns=\"a})",
            "expected a closing '\"', found the end of the pipeline at offset 45",
        ),
        (
            "builtin.module(canonicalizer{enable-patterns=\"a\\nb\"})",
            "expected the escape '\\\"' or '\\\\', found '\\n' at offset 47",
        ),
    ] {
        let error = parse_error(invalid);
        assert!(error.contains(expected), "{invalid}: {error}");
    }
}

#[test]
fn quoted_values_escape_quotes_and_backslashes() {
    let text = r#"any(p{a="say \"hi\" \\ {x=(y)}"})"#;
    let pipeline = parse(text);
    let PipelineElement::Pass(pass) = &pipeline.elements()[0] else {
        panic!("expected a pass");
    };
    assert_eq!(pass.options(), [("a".to_string(), Some(r#"say "hi" \ {x=(y)}"#.to_string()))]);
    assert_eq!(pipeline.to_string(), text);
}

#[test]
fn interleaved_pipeline_roundtrip() {
    for source in [
        "builtin.module(canonicalizer,builtin.function(cse))",
        "builtin.module(cse,builtin.function(canonicalizer),cse)",
        "builtin.module(builtin.module(cse,builtin.function(canonicalizer),cse))",
        "any(cse,any(dce),builtin.function(cse))",
    ] {
        assert_eq!(parse(source).to_string(), source);
    }
}

struct RecordPasses(Rc<RefCell<Vec<String>>>);

impl PassInstrumentation for RecordPasses {
    fn run_before_pass(&mut self, pass: &dyn OperationPass, op: &OperationRef) {
        if let Some(name) = pass.argument().strip_prefix("pipeline-test-") {
            self.0.borrow_mut().push(format!("{}:{name}", op.borrow().name()));
        }
    }
}

const TREE: &str = r#"builtin.world {
    builtin.function public extern("C") @root() { builtin.ret; };
    builtin.module public @outer {
        builtin.function public extern("C") @first() { builtin.ret; };
        builtin.module public @inner {
            builtin.function public extern("C") @deep() { builtin.ret; };
        };
    };
};"#;

/// Runs `text` with explicit nesting on `ir` and returns the passes run, as `op:pass`.
fn run(text: &str, ir: &str) -> Vec<String> {
    register();
    let context = Rc::new(Context::default());
    let root = parse_any(ParserConfig::new(context.clone()), "pipeline.hir", ir).unwrap();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let mut pm = parse(text).load(context, Nesting::Explicit).unwrap();
    assert_eq!(printed(&pm), text);
    pm.add_instrumentation(Box::new(RecordPasses(observed.clone())));
    pm.run(root).unwrap();
    observed.take()
}

#[test]
fn interleaved_pipeline_execution_order() {
    let observed = run(
        "builtin.module(pipeline-test-generic,builtin.module(pipeline-test-other),\
         pipeline-test-generic)",
        "builtin.module public @root { builtin.module public @inner {}; };",
    );
    assert_eq!(
        observed,
        ["builtin.module:generic", "builtin.module:other", "builtin.module:generic"]
    );
}

/// `recursive(..)` reaches the functions at every depth; a nested pipeline only those directly
/// inside.
#[test]
fn recursive_pipelines_reach_every_depth() {
    let text = "builtin.world(recursive(builtin.function(pipeline-test-function)))";
    assert_eq!(parse(text).to_string(), text);
    assert_eq!(run(text, TREE), ["builtin.function:function"; 3]);
    let text = "builtin.world(builtin.function(pipeline-test-function))";
    assert_eq!(run(text, TREE), ["builtin.function:function"]);

    let error = parse_error("builtin.world(recursive(builtin.function(cse),cse))");
    assert!(error.contains("expected ')' to close 'recursive(', found ','"), "{error}");
    let error = parse_error("builtin.world(recursive(cse))");
    assert!(
        error.contains("expected an anchor ('any' or '<dialect>.<operation>'), found 'cse'"),
        "{error}"
    );
}

#[test]
fn invalid_pipeline_anchors_return_errors() {
    for (source, expected) in [
        ("nosuch.op(cse)", "invalid pass manager anchor 'nosuch.op'"),
        ("builtin.nosuch(cse)", "invalid pass manager anchor 'builtin.nosuch'"),
        ("builtin.module(nosuch.op(cse))", "invalid pass manager anchor 'nosuch.op'"),
        (
            "builtin.module(builtin.nosuch(cse))",
            "invalid pass manager anchor 'builtin.nosuch'",
        ),
        (
            "builtin.module(builtin.module(nosuch.op(cse)))",
            "invalid pass manager anchor 'nosuch.op'",
        ),
        (
            "builtin.module(recursive(nosuch.op(cse)))",
            "invalid pass manager anchor 'nosuch.op'",
        ),
    ] {
        let error = load_error(source, Nesting::Explicit);
        assert!(error.contains(expected), "{source}: {error}");
    }

    for (source, offset) in [
        ("function(cse)", 0),
        ("builtin.(cse)", 0),
        ("builtin.module(function(cse))", 15),
        ("builtin.module(builtin..x(cse))", 15),
        ("(cse)", 0),
        ("", 0),
    ] {
        let error = parse_error(source);
        assert!(
            error.contains("expected an anchor ('any' or '<dialect>.<operation>')")
                && error.contains(&format!("at offset {offset}")),
            "{source}: {error}"
        );
    }
}

/// A malformed text is reported with its offset, the line of text and a caret under the fault.
#[test]
fn malformed_text_names_the_offset() {
    assert_eq!(
        parse_error("builtin.module(cse})"),
        "invalid pass pipeline: expected ',' or ')', found '}' at offset 18\n  \
         builtin.module(cse})\n                    ^"
    );
    for (source, expected) in [
        (
            "builtin.module(cse",
            "expected ',' or ')', found the end of the pipeline at offset 18",
        ),
        (
            "builtin.module(cse) cse",
            "expected the end of the pipeline, found 'cse' at offset 20",
        ),
        ("builtin.module", "expected '(' after the anchor, found the end of the pipeline"),
        (
            "builtin.module(,cse)",
            "expected a pass name, a nested pipeline or 'recursive(..)', found ','",
        ),
        ("builtin.module(cse.x)", "found 'cse.x' at offset 15"),
        ("builtin.module(cse{a=b}{c=d})", "expected ',' or ')', found '{' at offset 23"),
    ] {
        let error = parse_error(source);
        assert!(error.contains(expected), "{source}: {error}");
    }
}

/// An empty nested pipeline is an error; the top-level pipeline may be empty, as an empty pass
/// manager prints it.
#[test]
fn only_the_top_level_pipeline_may_be_empty() {
    let error = parse_error("builtin.module(builtin.function())");
    assert!(
        error.contains(
            "expected a pass name, a nested pipeline or 'recursive(..)', found ')' at offset 32"
        ),
        "{error}"
    );
    let error = parse_error("builtin.module(recursive(builtin.function()))");
    assert!(error.contains("found ')' at offset 42"), "{error}");

    let empty = parse("builtin.module()");
    assert!(empty.elements().is_empty());
    let pm = empty.load(Rc::new(Context::default()), Nesting::Explicit).unwrap();
    assert_eq!(printed(&pm), "builtin.module()");
}

#[test]
fn elements_are_anchored_on_the_given_operation() {
    let context = Context::default();
    let module = <Module as PassTarget>::target_name(&context).unwrap();
    assert_eq!(
        PassPipeline::parse_elements(&module, "cse, builtin.function(dce{a=1})").unwrap(),
        parse("builtin.module(cse,builtin.function(dce{a=\"1\"}))")
    );
    let empty = PassPipeline::parse_elements(&module, "  ").unwrap();
    assert_eq!(empty.to_string(), "builtin.module()");
    assert_eq!(*empty.anchor(), Anchor::from_word("builtin.module").unwrap());

    let error = PassPipeline::parse_elements(&module, "cse)").unwrap_err().to_string();
    assert!(
        error.contains("expected ',' or the end of the pipeline, found ')' at offset 3"),
        "{error}"
    );
    let error = PassPipeline::parse_elements(&module, "cse,").unwrap_err().to_string();
    assert!(error.contains("found the end of the pipeline at offset 4"), "{error}");
}

#[test]
fn an_unknown_pass_lists_similar_names() {
    register();
    let error = load_error("builtin.module(pipeline-test-generc)", Nesting::Explicit);
    assert!(
        error.starts_with("unknown pass 'pipeline-test-generc'; did you mean"),
        "{error}"
    );
    assert!(error.contains("'pipeline-test-generic'"), "{error}");

    let error =
        load_error("builtin.module(zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz)", Nesting::Explicit);
    assert_eq!(error, "unknown pass 'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz'");
}

#[test]
fn options_reach_the_pass() {
    register();
    let text = "any(pipeline-test-options{gamma=\"x;y\" flag alpha=\"\"})";
    let pm = parse(text).load(Rc::new(Context::default()), Nesting::Explicit).unwrap();
    assert_eq!(printed(&pm), text);

    let error = load_error("any(pipeline-test-options{delta=1})", Nesting::Explicit);
    assert_eq!(error, "unknown option 'delta'");
    let error = load_error("any(pipeline-test-generic{alpha=1})", Nesting::Explicit);
    assert_eq!(error, "pass 'pipeline-test-generic' takes no options");
}

/// A typed pass written under an anchor of another type is an error with explicit nesting, and is
/// nested under its type with implicit nesting.
#[test]
fn a_typed_pass_under_another_anchor() {
    register();
    let text = "builtin.module(pipeline-test-generic,pipeline-test-function)";
    let error = load_error(text, Nesting::Explicit);
    assert!(
        error.starts_with(
            "cannot add pass 'pipeline-test-function', which runs on 'builtin.function', to a \
             pass manager anchored on 'builtin.module'"
        ),
        "{error}"
    );
    let pm = parse(text).load(Rc::new(Context::default()), Nesting::Implicit).unwrap();
    assert_eq!(
        printed(&pm),
        "builtin.module(pipeline-test-generic,builtin.function(pipeline-test-function))"
    );
    // The nesting chosen holds for nested pipelines too.
    let text = "builtin.world(builtin.module(pipeline-test-function))";
    assert!(load_error(text, Nesting::Explicit).starts_with("cannot add pass"));
    let pm = parse(text).load(Rc::new(Context::default()), Nesting::Implicit).unwrap();
    assert_eq!(
        printed(&pm),
        "builtin.world(builtin.module(builtin.function(pipeline-test-function)))"
    );
}

#[test]
fn nesting_is_bounded() {
    let mut text = String::new();
    for _ in 0..10_000 {
        text.push_str("builtin.module(");
    }
    let error = parse_error(&text);
    assert!(error.contains("expected at most 64 levels of nested pipelines"), "{error}");
}

/// A splitmix64 generator: the test is deterministic for a seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// The seeds of the randomized test: `PIPELINE_TEST_SEED` if set, else three fixed ones.
fn seeds() -> Vec<u64> {
    match std::env::var("PIPELINE_TEST_SEED") {
        Ok(seed) => vec![seed.parse().expect("PIPELINE_TEST_SEED is a u64")],
        Err(_) => vec![0x5eed_0001, 0x5eed_0002, 0x5eed_0003],
    }
}

const ANCHORS: [&str; 4] = ["any", "builtin.world", "builtin.module", "builtin.function"];

/// A pipeline on `anchor` that loads with explicit nesting: the function pass appears only under
/// `builtin.function` and `any`.
fn random_pipeline(rng: &mut Rng, anchor: &str, depth: usize) -> PassPipeline {
    let mut elements = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let element = match rng.below(if depth < 3 { 6 } else { 4 }) {
            0 if matches!(anchor, "any" | "builtin.function") => {
                PipelineElement::Pass(SelectedPass {
                    name: "pipeline-test-function".to_string(),
                    options: Vec::new(),
                })
            }
            0 | 1 => PipelineElement::Pass(SelectedPass {
                name: rng.pick(&["pipeline-test-generic", "pipeline-test-other"]).to_string(),
                options: Vec::new(),
            }),
            2 | 3 => PipelineElement::Pass(random_options_pass(rng)),
            4 => {
                let nested = rng.pick(&ANCHORS);
                PipelineElement::Nested(random_pipeline(rng, nested, depth + 1))
            }
            _ => {
                let nested = rng.pick(&ANCHORS);
                PipelineElement::Recursive(random_pipeline(rng, nested, depth + 1))
            }
        };
        elements.push(element);
    }
    PassPipeline {
        anchor: Anchor::from_word(anchor).unwrap(),
        elements,
    }
}

fn random_options_pass(rng: &mut Rng) -> SelectedPass {
    // Every character a quoted value may hold that is special elsewhere, except ','.
    const ALPHABET: &[char] = &[
        'a', 'z', '0', '9', ';', '-', '_', ' ', '"', '\\', '{', '}', '(', ')', '=', '.', 'é',
    ];
    let mut keys = KEYS.to_vec();
    let mut options = Vec::new();
    for _ in 0..rng.below(KEYS.len() + 1) {
        let key = keys.remove(rng.below(keys.len()));
        let value = (key != "flag").then(|| {
            (0..rng.below(6))
                .map(|_| ALPHABET[rng.below(ALPHABET.len())])
                .collect::<String>()
        });
        options.push((key.to_string(), value));
    }
    SelectedPass {
        name: "pipeline-test-options".to_string(),
        options,
    }
}

/// Printing a generated pipeline and parsing the text gives the pipeline back, and the pass
/// manager it loads prints the same text.
#[test]
fn random_pipelines_roundtrip() {
    register();
    for seed in seeds() {
        let mut rng = Rng(seed);
        for case in 0..200 {
            let anchor = rng.pick(&ANCHORS);
            let pipeline = random_pipeline(&mut rng, anchor, 0);
            let text = pipeline.to_string();
            let context =
                || format!("seed {seed} (PIPELINE_TEST_SEED={seed}), case {case}: {text}");
            let parsed =
                PassPipeline::parse(&text).unwrap_or_else(|error| panic!("{}: {error}", context()));
            assert_eq!(parsed, pipeline, "{}", context());
            let pm = pipeline
                .load(Rc::new(Context::default()), Nesting::Explicit)
                .unwrap_or_else(|error| panic!("{}: {error}", context()));
            assert_eq!(printed(&pm), text, "{}", context());
        }
    }
}
