//! IR printed between the passes of a pipeline loaded from text, as `runic-opt` prints it.

use alloc::{boxed::Box, rc::Rc, string::String, vec, vec::Vec};
use core::cell::RefCell;

use runic_core::{
    Context,
    parse::{ParserConfig, parse_any},
    pass::{IRPrintingConfig, Nesting, PassPipeline, PassSelection},
};

/// A stream whose text the test reads after the pass manager has written to it.
#[derive(Clone, Default)]
struct Buffer(Rc<RefCell<String>>);

impl core::fmt::Write for Buffer {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        self.0.borrow_mut().push_str(text);
        Ok(())
    }
}

const MODULE: &str = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%a: i32, %b: i32) -> i32 {
        %x = arith.add %a, %b;
        %y = arith.add %a, %b;
        %z = arith.add %x, %y;
        builtin.ret %z : (i32);
    };
};"#;

/// Runs `cse,dce` on the module with `config` and returns what was written.
fn run(config: IRPrintingConfig) -> String {
    let context = Rc::new(Context::default());
    let module = parse_any(ParserConfig::new(context.clone()), "printing", MODULE)
        .into_result()
        .expect("a valid module");
    let pipeline = PassPipeline::parse("builtin.module(cse,dce)").unwrap();
    let mut pm = pipeline.load(context.clone(), Nesting::Implicit).unwrap();
    let buffer = Buffer::default();
    pm.enable_ir_printing(config, Box::new(buffer.clone()));
    pm.run(module).unwrap();
    buffer.0.borrow().clone()
}

fn headers(text: &str) -> Vec<&str> {
    text.lines().filter(|line| line.starts_with("// -----")).collect()
}

#[test]
fn ir_is_printed_after_cse_and_not_after_dce() {
    let text = run(IRPrintingConfig {
        after: PassSelection::Named(vec!["cse".into()]),
        ..Default::default()
    });
    assert_eq!(headers(&text), ["// ----- IR after cse on builtin.module -----"], "{text}");
    // After `cse`, `%y` is gone: one `arith.add %a, %b` remains.
    assert_eq!(text.matches("arith.add").count(), 2, "{text}");
    assert!(text.ends_with("};\n\n"), "{text:?}");
}

#[test]
fn ir_is_printed_before_each_pass() {
    let text = run(IRPrintingConfig {
        before: PassSelection::All,
        ..Default::default()
    });
    assert_eq!(
        headers(&text),
        [
            "// ----- IR before cse on builtin.module -----",
            "// ----- IR before dce on builtin.module -----",
        ],
        "{text}"
    );
}
