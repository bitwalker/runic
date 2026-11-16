//! Pipeline text in both directions, for pipelines over every registered pass: generated
//! pipelines read back to themselves and load to managers that print them; mutated and random
//! text never panics and is a pipeline that reads back or an error with an offset; deep and long
//! text on a small stack. Seeds and case counts follow the crate's convention (`TRANSFORM_TEST_SEED`,
//! `TRANSFORM_TEST_CASES`).

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt;

use runic_ir::{
    Context,
    pass::{Nesting, PassManager, PassPipeline},
    testing::text::Rng,
};

const PASSES: [&str; 8] = [
    "canonicalizer",
    "cfg-to-scf",
    "control-flow-sink",
    "cse",
    "dce",
    "local2reg",
    "sccp",
    "sink-operand-defs",
];

const ANCHORS: [&str; 4] = ["any", "builtin.world", "builtin.module", "builtin.function"];

fn printed(pm: &PassManager) -> String {
    struct Printed<'a>(&'a PassManager);
    impl fmt::Display for Printed<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.print_as_textual_pipeline(f)
        }
    }
    Printed(pm).to_string()
}

fn on_a_small_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(f)
        .expect("spawn a thread")
        .join()
        .expect("the thread finishes");
}

/// Runs `f` on `text`; a panic is reported with the text that caused it.
fn guarded<R>(text: &str, f: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => panic!("panicked on the pipeline text {text:?}"),
    }
}

#[derive(Debug, Clone)]
enum Elem {
    Pass(&'static str, Vec<(String, Option<String>)>),
    Nested(bool, &'static str, Vec<Elem>),
}

fn below(rng: &mut Rng, n: usize) -> usize {
    rng.below(n as u64) as usize
}

fn pattern_name(rng: &mut Rng) -> String {
    const ALPHABET: &[char] = &[
        'a', 'b', 'z', '0', '-', '_', '"', '\\', '{', '}', '(', ')', '=', '.', 'é', '\'', '#',
    ];
    (0..1 + below(rng, 4)).map(|_| ALPHABET[below(rng, ALPHABET.len())]).collect()
}

/// Canonicalizer options in the form the pass prints them: `enable-patterns` first, each list
/// its names joined by `;`, no name in both.
fn canonicalizer_options(rng: &mut Rng) -> Vec<(String, Option<String>)> {
    let mut options = Vec::new();
    let mut taken = Vec::new();
    for key in ["enable-patterns", "disable-patterns"] {
        if below(rng, 3) == 0 {
            continue;
        }
        let mut names = Vec::new();
        for _ in 0..1 + below(rng, 3) {
            let name = pattern_name(rng);
            if !taken.contains(&name) {
                taken.push(name.clone());
                names.push(name);
            }
        }
        if !names.is_empty() {
            options.push((key.to_string(), Some(names.join(";"))));
        }
    }
    options
}

fn random_elems(rng: &mut Rng, depth: usize) -> Vec<Elem> {
    (0..below(rng, 4))
        .map(|_| {
            if depth < 5 && below(rng, 3) == 0 {
                Elem::Nested(
                    below(rng, 3) == 0,
                    ANCHORS[below(rng, 4)],
                    random_elems(rng, depth + 1),
                )
            } else {
                let pass = PASSES[below(rng, PASSES.len())];
                let options = if pass == "canonicalizer" {
                    canonicalizer_options(rng)
                } else {
                    Vec::new()
                };
                Elem::Pass(pass, options)
            }
        })
        .collect()
}

fn quote(value: &str) -> String {
    let mut quoted = String::from("\"");
    for c in value.chars() {
        if matches!(c, '"' | '\\') {
            quoted.push('\\');
        }
        quoted.push(c);
    }
    quoted.push('"');
    quoted
}

fn is_bare(value: &str) -> bool {
    !value.is_empty() && !value.chars().any(|c| c.is_whitespace() || "(){},=\"".contains(c))
}

/// The text of `elems`: canonical, or with whitespace between tokens and bare values where a
/// value can be bare.
fn write_elems(out: &mut String, elems: &[Elem], rng: &mut Option<&mut Rng>) {
    let space = |out: &mut String, rng: &mut Option<&mut Rng>| {
        if let Some(rng) = rng {
            for _ in 0..below(rng, 3) {
                out.push([' ', '\t', '\n'][below(rng, 3)]);
            }
        }
    };
    for (i, elem) in elems.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        space(out, rng);
        match elem {
            Elem::Pass(name, options) => {
                out.push_str(name);
                if !options.is_empty() {
                    space(out, rng);
                    out.push('{');
                    for (i, (key, value)) in options.iter().enumerate() {
                        if i > 0 {
                            out.push(' ');
                        }
                        space(out, rng);
                        out.push_str(key);
                        if let Some(value) = value {
                            space(out, rng);
                            out.push('=');
                            space(out, rng);
                            let bare = is_bare(value)
                                && rng.as_mut().is_some_and(|rng| below(rng, 2) == 0);
                            if bare {
                                out.push_str(value);
                            } else {
                                out.push_str(&quote(value));
                            }
                        }
                    }
                    space(out, rng);
                    out.push('}');
                }
            }
            Elem::Nested(recursive, anchor, elems) => {
                if *recursive {
                    out.push_str("recursive");
                    space(out, rng);
                    out.push('(');
                    space(out, rng);
                }
                out.push_str(anchor);
                space(out, rng);
                out.push('(');
                write_elems(out, elems, rng);
                space(out, rng);
                out.push(')');
                if *recursive {
                    space(out, rng);
                    out.push(')');
                }
            }
        }
        space(out, rng);
    }
}

fn text_of(anchor: &str, elems: &[Elem], mut rng: Option<&mut Rng>) -> String {
    let mut out = String::from(anchor);
    out.push('(');
    write_elems(&mut out, elems, &mut rng);
    out.push(')');
    out
}

/// With explicit nesting, `local2reg` loads only directly under `builtin.function` or `any`.
fn loads_explicitly(anchor: &str, elems: &[Elem]) -> bool {
    elems.iter().all(|elem| match elem {
        Elem::Pass(name, _) => *name != "local2reg" || matches!(anchor, "any" | "builtin.function"),
        Elem::Nested(_, anchor, elems) => loads_explicitly(anchor, elems),
    })
}

/// The text a manager loaded with implicit nesting prints: `local2reg` under another anchor
/// than `builtin.function` or `any` is in a `builtin.function(..)` of its own.
fn implicit(anchor: &str, elems: &[Elem]) -> Vec<Elem> {
    elems
        .iter()
        .map(|elem| match elem {
            Elem::Pass("local2reg", _) if !matches!(anchor, "any" | "builtin.function") => {
                Elem::Nested(false, "builtin.function", vec![elem.clone()])
            }
            Elem::Pass(..) => elem.clone(),
            Elem::Nested(recursive, anchor, elems) => {
                Elem::Nested(*recursive, anchor, implicit(anchor, elems))
            }
        })
        .collect()
}

/// The checks on a text that parses: `print(parse(t))` is canonical (it parses to the same
/// pipeline and prints itself), and a manager it loads prints text that loads to a manager
/// printing the same text. Returns the text each nesting's manager printed.
fn check_parsed(text: &str, pipeline: &PassPipeline) -> [Option<String>; 2] {
    let canonical = pipeline.to_string();
    let reparsed = PassPipeline::parse(&canonical)
        .unwrap_or_else(|error| panic!("{text:?} printed {canonical:?}: {error}"));
    assert_eq!(&reparsed, pipeline, "{text:?} printed {canonical:?}");
    assert_eq!(reparsed.to_string(), canonical, "{text:?}");
    [Nesting::Implicit, Nesting::Explicit].map(|nesting| {
        let pm = pipeline.load(Rc::new(Context::default()), nesting).ok()?;
        let first = printed(&pm);
        let again = PassPipeline::parse(&first)
            .unwrap_or_else(|error| panic!("{text:?} loaded prints {first:?}: {error}"));
        assert_eq!(again.to_string(), first, "{text:?}: the manager's text is not canonical");
        let pm = again
            .load(Rc::new(Context::default()), nesting)
            .unwrap_or_else(|error| panic!("{text:?} loaded prints {first:?}: {error}"));
        assert_eq!(printed(&pm), first, "{text:?}");
        Some(first)
    })
}

/// Generated pipelines over every registered pass, with the canonicalizer's options:
/// canonical text reads back to itself, spaced text reads to the same pipeline, and the manager
/// each loads prints the text the pipeline states.
#[test]
fn generated_pipelines_read_back() {
    on_a_small_stack(|| {
        let mut rng = Rng(super::seed(0x4a_f0_22));
        let (mut loaded, mut rejected) = (0, 0);
        for _ in 0..super::cases(1500) {
            let anchor = ANCHORS[below(&mut rng, 4)];
            let elems = random_elems(&mut rng, 0);
            let canonical = text_of(anchor, &elems, None);
            let spaced = text_of(anchor, &elems, Some(&mut rng));
            let pipeline = guarded(&canonical, || PassPipeline::parse(&canonical))
                .unwrap_or_else(|error| panic!("{canonical:?}: {error}"));
            assert_eq!(pipeline.to_string(), canonical);
            let from_spaced = guarded(&spaced, || PassPipeline::parse(&spaced))
                .unwrap_or_else(|error| panic!("{spaced:?}: {error}"));
            assert_eq!(from_spaced, pipeline, "{spaced:?}");
            let [implicitly, explicitly] =
                guarded(&canonical, || check_parsed(&canonical, &pipeline));
            assert_eq!(
                implicitly.as_deref(),
                Some(text_of(anchor, &implicit(anchor, &elems), None).as_str()),
                "{canonical}"
            );
            if loads_explicitly(anchor, &elems) {
                assert_eq!(explicitly.as_deref(), Some(canonical.as_str()));
                loaded += 1;
            } else {
                assert_eq!(explicitly, None, "{canonical}");
                rejected += 1;
            }
        }
        assert!(loaded > 250 && rejected > 75, "{loaded} {rejected}");
    });
}

// Mutated and random text

const SOUP: &[&str] = &[
    "(",
    ")",
    "{",
    "}",
    ",",
    "=",
    "\"",
    "\\",
    " ",
    "\t",
    "\n",
    ".",
    "-",
    "_",
    ";",
    "a",
    "Z",
    "0",
    "é",
    "\u{0}",
    "\u{2028}",
    "\u{a0}",
    "🦀",
    "any",
    "recursive",
    "builtin",
    "builtin.module",
    "builtin.function",
    "builtin.world",
    "builtin.ret",
    "arith.add",
    "scf.if",
    "nosuch.op",
    "cse",
    "dce",
    "local2reg",
    "canonicalizer",
    "enable-patterns",
    "disable-patterns",
    "x;y",
    "\\\"",
    "\\\\",
    "\\n",
    "any(",
    "recursive(",
    "{a=b}",
    "{a}",
    "{a=\"b\"}",
    "=\"\"",
];

fn mutate(rng: &mut Rng, text: &str) -> String {
    let mut chars = text.chars().collect::<Vec<_>>();
    for _ in 0..1 + below(rng, 3) {
        let at = below(rng, chars.len() + 1);
        match below(rng, 6) {
            0 if !chars.is_empty() => {
                chars.remove(at.min(chars.len() - 1));
            }
            1 if !chars.is_empty() => {
                let end = (at + 1 + below(rng, 8)).min(chars.len());
                chars.drain(at.min(end)..end);
            }
            2 if !chars.is_empty() => {
                let end = (at + 1 + below(rng, 12)).min(chars.len());
                let copy = chars[at.min(end)..end].to_vec();
                let to = below(rng, chars.len() + 1);
                chars.splice(to..to, copy);
            }
            3 => chars.truncate(at),
            _ => {
                let piece = SOUP[below(rng, SOUP.len())].chars().collect::<Vec<_>>();
                chars.splice(at..at, piece);
            }
        }
    }
    chars.into_iter().collect()
}

/// What may be done with any text at all: parse it, and if it parses, everything `check_parsed`
/// checks; `parse_elements` likewise. Returns whether it parsed.
fn exercise(text: &str) -> bool {
    guarded(text, || {
        let function =
            Context::default().get_registered_name::<runic_ir::dialects::builtin::Function>();
        if let Ok(pipeline) = PassPipeline::parse_elements(&function, text) {
            check_parsed(text, &pipeline);
        }
        match PassPipeline::parse(text) {
            Ok(pipeline) => {
                check_parsed(text, &pipeline);
                true
            }
            Err(error) => {
                let message = error.to_string();
                assert!(
                    message.starts_with("invalid pass pipeline: expected "),
                    "{text:?}: {message}"
                );
                assert!(message.contains(" at offset "), "{text:?}: {message}");
                false
            }
        }
    })
}

/// Mutated pipelines and random token soup never panic, loop or overflow a
/// 256 KiB stack; what parses reads back; what does not is an error with an offset.
#[test]
fn mutated_and_random_text_never_panics() {
    on_a_small_stack(|| {
        let mut rng = Rng(super::seed(0x4a_f0_23));
        let (mut parsed, mut failed) = (0, 0);
        for _ in 0..super::cases(3_000) {
            let anchor = ANCHORS[below(&mut rng, 4)];
            let elems = random_elems(&mut rng, 0);
            let valid = if below(&mut rng, 2) == 0 {
                text_of(anchor, &elems, None)
            } else {
                text_of(anchor, &elems, Some(&mut rng))
            };
            let mutated = mutate(&mut rng, &valid);
            if exercise(&mutated) {
                parsed += 1;
            } else {
                failed += 1;
            }
        }
        for _ in 0..super::cases(3_000) {
            let soup = (0..below(&mut rng, 14))
                .map(|_| SOUP[below(&mut rng, SOUP.len())])
                .collect::<String>();
            if exercise(&soup) {
                parsed += 1;
            } else {
                failed += 1;
            }
        }
        std::eprintln!("parsed {parsed}, failed {failed}");
        assert!(parsed > 300 && failed > 3000, "{parsed} {failed}");
    });
}

/// Deep and long texts on a 256 KiB stack. Nesting is bounded at 64 levels, with
/// an error; width is not bounded.
#[test]
fn deep_and_long_text_is_an_error_or_parses() {
    on_a_small_stack(|| {
        for opener in ["any(", "recursive(any(", "builtin.module(", "recursive(", "a.b(", "(", "{"]
        {
            for depth in [1, 31, 32, 62, 63, 64, 65, 1000, 20_000] {
                let unclosed = opener.repeat(depth);
                assert!(!exercise(&unclosed), "{opener} x {depth}");
                let closers = opener.matches('(').count() * depth;
                let closed = format!("{unclosed}{}", ")".repeat(closers));
                let parses = exercise(&closed);
                if opener == "any(" || opener == "builtin.module(" {
                    assert_eq!(parses, depth <= 64, "{opener} x {depth}");
                }
                if opener == "recursive(any(" {
                    // `recursive(..)` is an element, never the whole pipeline.
                    assert!(!parses, "{opener} x {depth}");
                    let inside = exercise(&format!("any({closed})"));
                    assert_eq!(inside, depth <= 63, "any({opener} x {depth})");
                }
            }
        }
        // Width: 20,000 passes, 5,000 nested pipelines, 5,000 options, values of 200 KB and 1 MB.
        assert!(exercise(&format!("any({})", vec!["cse"; 20_000].join(","))));
        assert!(exercise(&format!(
            "builtin.module({})",
            vec!["builtin.function(dce)"; 5_000].join(",")
        )));
        let options = (0..5_000).map(|i| format!("k{i}=v")).collect::<Vec<_>>().join(" ");
        assert!(exercise(&format!("any(cse{{{options}}})")));
        assert!(exercise(&format!(
            "any(canonicalizer{{enable-patterns=\"{}\"}})",
            "x;".repeat(100_000)
        )));
        assert!(!exercise(&format!("any(cse{{a=\"{}", "x".repeat(1_000_000))));
        assert!(!exercise(&")".repeat(1_000_000)));
        assert!(!exercise(&",".repeat(1_000_000)));
    });
}

/// Whether text starts with an anchor does not tell a whole pipeline from a list of elements:
/// `builtin.function(dce),cse` starts with an anchor and is a list of elements, and
/// `builtin.function(dce)` alone parses both ways, to two pipelines.
#[test]
fn text_that_starts_with_an_anchor_may_be_a_list_of_elements() {
    let context = Rc::new(Context::default());
    let module = context.get_registered_name::<runic_ir::dialects::builtin::Module>();
    let error = PassPipeline::parse("builtin.function(dce),cse").unwrap_err().to_string();
    assert!(error.contains("expected the end of the pipeline, found ','"), "{error}");
    let elements = PassPipeline::parse_elements(&module, "builtin.function(dce),cse").unwrap();
    assert_eq!(elements.to_string(), "builtin.module(builtin.function(dce),cse)");
    // The same first element alone parses both ways, to two different pipelines.
    let whole = PassPipeline::parse("builtin.function(dce)").unwrap();
    let elements = PassPipeline::parse_elements(&module, "builtin.function(dce)").unwrap();
    assert_eq!(whole.to_string(), "builtin.function(dce)");
    assert_eq!(elements.to_string(), "builtin.module(builtin.function(dce))");
}
