//! Calls through the passes of this crate: calls in each effect state, calls of a function that
//! never returns.

use alloc::{
    boxed::Box,
    format,
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

use crate::{Canonicalizer, CommonSubexpressionElimination, DeadCodeElimination};

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

// ---------------------------------------------------------------------------------------------
// Symbol uses of erased operations
// ---------------------------------------------------------------------------------------------

/// What [uses_of_callee_after] reports.
#[derive(Debug, PartialEq, Eq)]
struct CalleeUses {
    /// The number of uses in `@callee`'s use list
    uses: usize,
    /// The number of known callers `CallableUseSnapshot` reports for `@callee`
    known_callers: usize,
    address_taken: bool,
    unknown_callers: bool,
}

/// A module with a private `@callee`, and `@caller` with `body`; after `pass` ran on the caller:
/// what the use list of `@callee` and `CallableUseSnapshot` of the module say of it, and the text
/// of the caller.
fn uses_of_callee_after(pass: MakePass, body: &str) -> (CalleeUses, String) {
    use runic_analysis::analyses::callable_uses::CallableUseSnapshot;
    use runic_core::{Usable, dialects::builtin::ModuleBuilder};

    let context = Rc::new(Context::default());
    let source = format!(
        "builtin.module public @m {{
    builtin.function private extern(\"C\") @callee(%x: i32) -> i32 effects() {{
        builtin.ret %x : (i32);
    }};
    builtin.function public extern(\"C\") @caller(%a: i32) -> i32 {{
{body}
    }};
}};"
    );
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "uses", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let caller = module.borrow().symbol_manager().lookup_op("caller").unwrap();
    run(&context, caller, pass()).unwrap();
    module.as_operation_ref().borrow().recursively_verify().unwrap();
    let target = ModuleBuilder::new(module).resolve_callable("callee").unwrap().target();
    let callee = ModuleBuilder::new(module).get_function("callee").unwrap();
    let uses = callee.borrow().iter_uses().count();
    let snapshot = CallableUseSnapshot::new(&module.as_operation_ref().borrow());
    let info = snapshot.get(target).unwrap();
    let report = CalleeUses {
        uses,
        known_callers: info.known_callers().len(),
        address_taken: info.is_address_taken(),
        unknown_callers: info.has_unknown_callers(),
    };
    (report, caller.borrow().to_string())
}

const INDIRECT: &str = r#"
        %f = builtin.function_ref @callee : (i32) -> i32;
        %r = builtin.call_indirect %f(%a) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);"#;

const TWO_CALLS: &str = r#"
        %r = builtin.call @callee(%a) : extern("C") (i32) -> i32;
        %dead = builtin.call @callee(%a) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);"#;

/// The use list of a symbol holds exactly the operations that remain: what a pass erased is not
/// a use, so the callee of a canonicalized indirect call is not address-taken, and a callee one of
/// whose calls was erased (`dce`) or merged (`cse`) is not used from outside the module.
#[test]
fn an_erased_operation_leaves_the_use_list_of_its_symbol() {
    let one_known_caller = CalleeUses {
        uses: 1,
        known_callers: 1,
        address_taken: false,
        unknown_callers: false,
    };
    let (report, _) = uses_of_callee_after(|| Box::<Canonicalizer>::default(), INDIRECT);
    assert_eq!(report, one_known_caller);
    for pass in [(|| Box::<DeadCodeElimination>::default()) as MakePass, || {
        Box::<CommonSubexpressionElimination>::default()
    }] {
        let (report, text) = uses_of_callee_after(pass, TWO_CALLS);
        assert_eq!(text.matches("builtin.call").count(), 1, "{text}");
        assert_eq!(report, one_known_caller, "{text}");
    }
}

/// After the canonicalizer rewrites an indirect call of a reference to a private function into a
/// direct call, the function is not address-taken and has exactly that one known caller.
#[test]
fn a_canonicalized_indirect_call_leaves_its_callee_with_one_known_caller() {
    use runic_analysis::analyses::callable_uses::CallableUseSnapshot;
    use runic_core::{CallOpInterface, dialects::builtin::ModuleBuilder};

    let context = Rc::new(Context::default());
    let source = format!(
        "builtin.module public @m {{
    builtin.function private extern(\"C\") @callee(%x: i32) -> i32 {{
        builtin.ret %x : (i32);
    }};
    builtin.function public extern(\"C\") @caller(%a: i32) -> i32 {{
{INDIRECT}
    }};
}};"
    );
    let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "indirect", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let caller = module.borrow().symbol_manager().lookup_op("caller").unwrap();
    run(&context, caller, Box::<Canonicalizer>::default()).unwrap();
    let text = caller.borrow().to_string();
    assert!(
        !text.contains("function_ref") && text.contains("builtin.call @callee"),
        "{text}"
    );

    let mut call = None;
    caller.borrow().prewalk_all(|op: &runic_core::Operation| {
        if op.implements::<dyn CallOpInterface>() {
            call = Some(op.as_operation_ref());
        }
    });
    let call = call.expect("the direct call");
    let target = ModuleBuilder::new(module).resolve_callable("callee").unwrap().target();
    let snapshot = CallableUseSnapshot::new(&module.as_operation_ref().borrow());
    let info = snapshot.get(target).unwrap();
    assert!(!info.is_address_taken(), "{text}");
    assert!(!info.has_unknown_callers(), "{text}");
    assert_eq!(info.known_callers().len(), 1, "{text}");
    assert!(info.known_callers().contains(&call), "{text}");
}
