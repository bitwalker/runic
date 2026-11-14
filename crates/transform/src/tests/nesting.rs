//! Regions nested deeply: the parser stops at [`ParserConfig::max_region_depth`] with a diagnostic,
//! and text nested to the default limit parses on the 2 MiB stack of a spawned thread in every
//! profile.

use alloc::{format, rc::Rc, string::String, vec::Vec};

use runic_core::{
    Context,
    parse::{ParserConfig, parse_any},
    testing::text::describe,
};

/// What encloses each level of a nest.
#[derive(Clone, Copy, Debug)]
enum Level {
    Module,
    Function,
    If,
}

/// The text of regions nested as `levels` say, outermost first: a region for each level (an
/// `scf.if` also has an `else` region at its own level). An `scf.if` branches on the `%c` of the
/// function around it.
fn nest(levels: &[Level]) -> String {
    let mut text = String::new();
    let mut closes = Vec::new();
    for (i, level) in levels.iter().enumerate() {
        match level {
            Level::Module => {
                text += &format!("builtin.module public @m{i} {{\n");
                closes.push(String::from("};\n"));
            }
            Level::Function => {
                text += &format!("builtin.function public extern(\"C\") @f{i}(%c: i1) {{\n");
                closes.push(String::from("builtin.ret;\n};\n"));
            }
            Level::If => {
                text += "scf.if %c then {\n";
                closes.push(String::from(
                    "scf.yield : ();\n} else {\nscf.yield : ();\n} : (i1) -> ();\n",
                ));
            }
        }
    }
    for close in closes.iter().rev() {
        text += close;
    }
    text
}

/// `depth` regions: a module, a function, then `scf.if`s.
fn nested_ifs(depth: usize) -> Vec<Level> {
    let mut levels = alloc::vec![Level::Module, Level::Function];
    levels.resize(depth, Level::If);
    levels
}

/// `depth` nested modules.
fn nested_modules(depth: usize) -> Vec<Level> {
    alloc::vec![Level::Module; depth]
}

/// `depth` regions that alternate a module, a function in it and an `scf.if` in that.
fn alternating(depth: usize) -> Vec<Level> {
    [Level::Module, Level::Function, Level::If]
        .into_iter()
        .cycle()
        .take(depth)
        .collect()
}

/// The levels of a nest of a given depth.
type Shape = fn(usize) -> Vec<Level>;

const SHAPES: [(&str, Shape); 3] = [
    ("nested scf.if", nested_ifs),
    ("nested builtin.module", nested_modules),
    ("alternating builtin.module, builtin.function and scf.if", alternating),
];

/// Parses `text` with `config` and verification off (a module in an `scf.if` is not valid IR,
/// and the limit is the parser's), and returns the error, described, if there is one.
fn parse(config: ParserConfig, text: &str) -> Result<(), String> {
    let context = config.context.clone();
    let parsed = parse_any(config.verify_after_parse(false), "nesting.runic", text).into_result();
    let result = parsed.map(drop).map_err(|error| describe(&error));
    drop(context);
    result
}

fn on_a_2_mib_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(f)
        .expect("spawn a thread")
        .join()
        .expect("the nest is parsed or rejected without overflowing the stack");
}

const MESSAGE: &str = "regions are nested more than 64 deep; the limit is set by `ParserConfig`";

#[test]
fn regions_nested_to_the_limit_parse_on_a_2_mib_stack() {
    on_a_2_mib_stack(|| {
        for (shape, levels) in SHAPES {
            let text = nest(&levels(64));
            let config = ParserConfig::new(Rc::new(Context::default()));
            assert_eq!(config.max_region_depth, 64);
            if let Err(error) = parse(config, &text) {
                panic!("{shape}, 64 deep: {error}");
            }
        }
    });
}

#[test]
fn a_nest_of_valid_ir_at_the_limit_parses_and_verifies_on_a_2_mib_stack() {
    on_a_2_mib_stack(|| {
        let context = Rc::new(Context::default());
        let text = nest(&nested_ifs(64));
        let parsed = parse_any(ParserConfig::new(context.clone()), "nesting.runic", text.as_str())
            .into_result();
        if let Err(error) = parsed {
            panic!("{}", describe(&error));
        }
    });
}

#[test]
fn a_region_one_over_the_limit_is_a_diagnostic_at_the_region() {
    on_a_2_mib_stack(|| {
        for (shape, levels) in SHAPES {
            let text = nest(&levels(65));
            let error = parse(ParserConfig::new(Rc::new(Context::default())), &text)
                .expect_err("65 regions are over the limit");
            assert!(error.starts_with(MESSAGE), "{shape}: {error}");
            assert!(error.contains("this region is nested more than 64 deep"), "{shape}: {error}");
        }
    });
}

#[test]
fn the_diagnostic_is_labelled_at_the_brace_that_opens_the_region() {
    let context = Rc::new(Context::default());
    let text = nest(&nested_modules(3));
    let config = ParserConfig::new(context.clone()).with_max_region_depth(2);
    let outcome = parse_any(config, "nesting.runic", text.as_str());
    let error = outcome.into_result().expect_err("three modules are over a limit of two");
    let diagnostic = error.as_diagnostic();
    struct Primary(Option<core::ops::Range<usize>>);
    impl runic_core::diagnostics::VisitDiagnostic for Primary {
        fn label(&mut self, label: runic_core::diagnostics::Label<'_>) {
            self.0.get_or_insert_with(|| {
                let range = label.span.range();
                range.start() as usize..range.end() as usize
            });
        }
    }
    let mut primary = Primary(None);
    diagnostic.visit(&mut primary);
    let range = primary.0.expect("the diagnostic has a label");
    let third = text.match_indices("@m2 {").next().unwrap().0 + "@m2 ".len();
    assert_eq!(range, third..third + 1, "{text}");
}

#[test]
fn a_region_far_over_the_limit_is_a_diagnostic_not_an_overflow() {
    on_a_2_mib_stack(|| {
        for (shape, levels) in SHAPES {
            let text = nest(&levels(10_000));
            let error = parse(ParserConfig::new(Rc::new(Context::default())), &text)
                .expect_err("10,000 regions are over the limit");
            assert!(error.starts_with(MESSAGE), "{shape}: {error}");
        }
    });
}

#[test]
fn the_limit_can_be_raised() {
    std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(|| {
            let text = nest(&nested_ifs(100));
            let config = ParserConfig::new(Rc::new(Context::default()));
            let error = parse(config, &text).expect_err("over the default");
            assert!(error.starts_with(MESSAGE), "{error}");
            let config = ParserConfig::new(Rc::new(Context::default())).with_max_region_depth(100);
            parse(config, &text).expect("within a raised limit");
            let config = ParserConfig::new(Rc::new(Context::default())).with_max_region_depth(99);
            let error = parse(config, &text).expect_err("over a raised limit");
            assert!(error.starts_with("regions are nested more than 99 deep"), "{error}");
        })
        .unwrap()
        .join()
        .unwrap();
}
