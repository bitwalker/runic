//! Calls through the passes of this crate: calls in each effect state, calls of a function that
//! never returns.

use alloc::{
    boxed::Box,
    rc::Rc,
    string::{String, ToString},
};

use runic_core::{
    Context, OperationRef, Report, SymbolTable,
    dialects::builtin::{Function, Module},
    parse::{self, ParserConfig},
    pass::{Nesting, OperationPass, PassManager},
    testing::text::describe,
};

use crate::{Canonicalizer, DeadCodeElimination};

type MakePass = fn() -> Box<dyn OperationPass>;

/// Runs `pass` on `function`, verifying after it.
fn run(
    context: &Rc<Context>,
    function: OperationRef,
    pass: Box<dyn OperationPass>,
) -> Result<(), Report> {
    let mut passes = PassManager::on::<Function>(context.clone(), Nesting::Implicit);
    passes.add_pass(pass);
    passes.enable_verifier(true);
    passes.run(function)
}

// ---------------------------------------------------------------------------------------------
// A call of a function that never returns
// ---------------------------------------------------------------------------------------------

const NEVER: &str = r#"
builtin.module public @m {
    builtin.function public extern("C") @abort(i32) -> never EFFECTS;
    builtin.function public extern("C") @f(%c: i1, %a: i32, %g: (i32) -> never) -> i32 {
        scf.if %c then {
            CALL
            scf.yield : ();
        } else {
            scf.yield : ();
        } : (i1) -> ();
        builtin.ret %a : (i32);
    };
};"#;

/// `@f` of [NEVER], with `@abort` declaring `effects` and `call` in the `scf.if`, after `pass`.
fn never_after(pass: MakePass, effects: &str, call: &str) -> String {
    let context = Rc::new(Context::default());
    let source = NEVER.replace("EFFECTS", effects).replace("CALL", call);
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "never", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let function = module.borrow().symbol_manager().lookup_op("f").unwrap();
    run(&context, function, pass()).unwrap();
    module.as_operation_ref().borrow().recursively_verify().unwrap();
    function.borrow().to_string()
}

/// A call whose signature says it never returns is not dead, whatever memory effects its callee
/// declares: no effects, only reads, a write, or nothing at all.
#[test]
fn a_call_that_never_returns_is_kept() {
    let call = r#"builtin.call @abort(%a) : extern("C") (i32) -> never;"#;
    for effects in ["effects()", "effects(read)", "", "effects(write)"] {
        for pass in [(|| Box::<DeadCodeElimination>::default()) as MakePass, || {
            Box::<Canonicalizer>::default()
        }] {
            let text = never_after(pass, effects, call);
            assert!(text.contains("builtin.call @abort"), "{effects}: {text}");
        }
    }
}
