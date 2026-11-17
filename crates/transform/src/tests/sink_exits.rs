//! The two sink passes leave alone an operation that holds an early exit or a loop that may not
//! end: moved under a condition, or past an effect, the exit or the divergence would not happen
//! where the program wrote it. Each program below has such an operation (`wasm.block` and
//! `scf.if` that hold a `builtin.ret`, a `wasm.loop` and an `scf.while` that may never end)
//! whose result is used under a condition (`control-flow-sink`) or after a store
//! (`sink-operand-defs`), and each pass leaves the module as it was written.

use alloc::{
    rc::Rc,
    string::{String, ToString},
};

use runic_ir::{
    Context, Report,
    parse::{ParserConfig, parse_any},
    pass::{Nesting, PassPipeline},
};

fn run(pipeline: &str, source: &str) -> Result<String, Report> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_wasm::WasmDialect>();
    context.get_or_register_dialect::<runic_dialect_scf::ScfDialect>();
    let module = parse_any(ParserConfig::new(context.clone()), "sink_exits", source.to_string())
        .into_result()?;
    let pipeline = PassPipeline::parse_elements(&module.borrow().name(), pipeline)?;
    let mut passes = pipeline.load(context.clone(), Nesting::Implicit)?;
    passes.enable_verifier(true);
    passes.run(module)?;
    let text = module.borrow().to_string();
    drop(passes);
    Ok(text)
}

fn assert_left_alone(source: &str, what: &str) -> Result<(), Report> {
    let written = run("", source)?;
    for pass in ["control-flow-sink", "sink-operand-defs"] {
        assert_eq!(run(pass, source)?, written, "{pass} moved something in {what}");
    }
    Ok(())
}

/// A `wasm.loop` and a `wasm.block` that holds a `builtin.ret`, before a `wasm.global_set`.
const K1: &str = r#"builtin.module public @m {
wasm.global public @g : mut i32 {
    %z = wasm.const 0 : i32;
    builtin.yield %z;
};
builtin.function public extern("C") @f(%c: i32, %i: i32) -> i32 {
    %r = wasm.loop %i {
    ^l(%a: i32):
        wasm.if %c then {
            wasm.br 1 %a : (i32);
        } : (i32) -> ();
        wasm.end %a : (i32);
    } : (i32) -> (i32);
    wasm.global_set @g, %i;
    %s = arith.add %r, %i;
    builtin.ret %s : (i32);
};
builtin.function public extern("C") @h(%c: i32, %i: i32) -> i32 {
    %r = wasm.block {
        wasm.if %c then {
            builtin.ret %i : (i32);
        } : (i32) -> ();
        wasm.end %i : (i32);
    } : () -> (i32);
    wasm.global_set @g, %i;
    %s = arith.add %r, %i;
    builtin.ret %s : (i32);
};
};"#;

/// A `scf.if` that holds a `builtin.ret` and an `scf.while`, before a `wasm.global_set`.
const K2: &str = r#"builtin.module public @m {
wasm.global public @g : mut i32 {
    %z = wasm.const 0 : i32;
    builtin.yield %z;
};
builtin.function public extern("C") @h(%c: i1, %i: i32) -> i32 {
    %r = scf.if %c then {
        builtin.ret %i : (i32);
    } else {
        scf.yield %i : (i32);
    } : (i1) -> (i32);
    wasm.global_set @g, %i;
    %s = arith.add %r, %i;
    builtin.ret %s : (i32);
};
builtin.function public extern("C") @w(%c: i1, %i: i32) -> i32 {
    %r = scf.while %i before {
    ^b(%a: i32):
        scf.condition %c, %a : (i1, i32);
    } after {
    ^a(%x: i32):
        scf.yield %x : (i32);
    } : (i32) -> (i32);
    wasm.global_set @g, %i;
    %s = arith.add %r, %i;
    builtin.ret %s : (i32);
};
};"#;

/// A `wasm.block` that holds a `builtin.ret` and a `wasm.loop`, used in an arm of an `scf.if`.
const K4: &str = r#"builtin.module public @m {
builtin.function public extern("C") @h(%c: i32, %d: i1, %i: i32) -> i32 {
    %r = wasm.block {
        wasm.if %c then {
            builtin.ret %i : (i32);
        } : (i32) -> ();
        wasm.end %i : (i32);
    } : () -> (i32);
    %l = wasm.loop %i {
    ^l(%a: i32):
        wasm.if %c then {
            wasm.br 1 %a : (i32);
        } : (i32) -> ();
        wasm.end %a : (i32);
    } : (i32) -> (i32);
    %s = scf.if %d then {
        %y = arith.add %r, %l;
        scf.yield %y : (i32);
    } else {
        scf.yield %i : (i32);
    } : (i1) -> (i32);
    builtin.ret %s : (i32);
};
};"#;

/// A `scf.if` that holds a `builtin.ret`, used in an arm of an `scf.if`.
const K5: &str = r#"builtin.module public @m {
builtin.function public extern("C") @h(%c: i1, %d: i1, %i: i32) -> i32 {
    %r = scf.if %c then {
        builtin.ret %i : (i32);
    } else {
        scf.yield %i : (i32);
    } : (i1) -> (i32);
    %s = scf.if %d then {
        %y = arith.add %r, %r;
        scf.yield %y : (i32);
    } else {
        scf.yield %i : (i32);
    } : (i1) -> (i32);
    builtin.ret %s : (i32);
};
};"#;

#[test]
fn a_wasm_loop_and_a_wasm_block_with_a_return_stay_before_a_store() -> Result<(), Report> {
    assert_left_alone(K1, "k1")
}

#[test]
fn an_scf_if_with_a_return_and_an_scf_while_stay_before_a_store() -> Result<(), Report> {
    assert_left_alone(K2, "k2")
}

#[test]
fn a_wasm_block_with_a_return_and_a_wasm_loop_stay_out_of_a_conditional_arm() -> Result<(), Report>
{
    assert_left_alone(K4, "k4")
}

#[test]
fn an_scf_if_with_a_return_stays_out_of_a_conditional_arm() -> Result<(), Report> {
    assert_left_alone(K5, "k5")
}
