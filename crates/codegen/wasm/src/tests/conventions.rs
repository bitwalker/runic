//! The rules of `extern("C")`, evaluated: a caller and callees in one module, for a width carried
//! in an `i32` (`i8`) and one carried in an `i64` (`i40`), through direct, tail and indirect calls.
//!
//! A narrow value whose bits above its width are not all zero is *dirty*. Under `"C"` a function
//! must not trust a parameter that is not marked `zext`, so the test calls the lowered callee
//! directly with a dirty value where the rule says it masks; and a caller must give a `sext`
//! parameter its value sign-extended, which the evaluator's log of calls shows.

use alloc::vec;

use super::{
    eval::{Machine, Outcome, Val, run_in},
    *,
};

/// A narrow type of the test: its name, its width and the width of its carrier.
#[derive(Copy, Clone)]
struct Narrow {
    name: &'static str,
    n: u32,
    carrier: u32,
}

/// The widths of the test: one carried in an `i32`, one in an `i64`.
const NARROW: [Narrow; 2] = [
    Narrow {
        name: "i8",
        n: 8,
        carrier: 32,
    },
    Narrow {
        name: "i40",
        n: 40,
        carrier: 64,
    },
];

impl Narrow {
    /// The value `bits` in the carrier.
    fn val(self, bits: u64) -> Val {
        Val::new(bits, self.carrier)
    }

    /// The least signed value, `2^(N-1)`, in carried form.
    fn min(self) -> u64 {
        1 << (self.n - 1)
    }

    /// The least signed value sign-extended to the carrier: dirty.
    fn min_extended(self) -> u64 {
        let carrier = if self.carrier == 64 {
            u64::MAX
        } else {
            (1 << self.carrier) - 1
        };
        carrier & !((1 << self.n) - 1) | self.min()
    }

    /// The text of `extend = #builtin.<marker>` on a value of this type.
    fn marked(self, marker: &str) -> String {
        format!("{} {{ extend = #builtin.{marker} }}", self.name)
    }
}

/// The functions of the module of `t`.
fn functions(t: Narrow) -> String {
    let n = t.name;
    let sext = t.marked("sext");
    let zext = t.marked("zext");
    let min = t.min();
    [
        // Callees.
        format!("builtin.function public extern(\"C\") @plain(%x: {n}) -> {n} {{\n    builtin.ret %x : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @second(%a: {n}, %b: {n}) -> {n} {{\n    builtin.ret %b : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @sext_param(%x: {sext}) -> {n} {{\n    builtin.ret %x : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @zext_param(%x: {zext}) -> ({zext}) {{\n    builtin.ret %x : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @sext_result() -> ({sext}) {{\n    %c = arith.constant {min} : {n};\n    builtin.ret %c : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @sext_immediate() -> ({sext}) {{\n    builtin.ret_imm {min} : {n};\n}};"),
        format!("builtin.function public extern(\"C\") @plain_immediate() -> {n} {{\n    builtin.ret_imm {min} : {n};\n}};"),
        format!("builtin.function public extern(\"C\") @take(%a: {sext}, %b: {n}) -> {n} {{\n    builtin.ret %b : ({n});\n}};"),
        // Callers.
        format!("builtin.function public extern(\"C\") @call_take(%a: {n}, %b: {n}) -> {n} {{\n    %r = builtin.call @take(%a, %b) : extern(\"C\") ({sext}, {n}) -> {n};\n    builtin.ret %r : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @tail_take(%a: {n}, %b: {n}) -> {n} {{\n    builtin.tail_call @take(%a, %b) : extern(\"C\") ({sext}, {n}) -> {n};\n}};"),
        format!("builtin.function public extern(\"C\") @indirect_take(%a: {n}, %b: {n}) -> {n} {{\n    %f = builtin.function_ref @take : ({n}, {n}) -> {n};\n    %r = builtin.call_indirect %f(%a, %b) : extern(\"C\") ({sext}, {n}) -> {n};\n    builtin.ret %r : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @tail_indirect_take(%a: {n}, %b: {n}) -> {n} {{\n    %f = builtin.function_ref @take : ({n}, {n}) -> {n};\n    builtin.tail_call_indirect %f(%a, %b) : extern(\"C\") ({sext}, {n}) -> {n};\n}};"),
        format!("builtin.function public extern(\"C\") @call_sext_result() -> {n} {{\n    %r = builtin.call @sext_result() : extern(\"C\") () -> ({sext});\n    builtin.ret %r : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @indirect_sext_result() -> {n} {{\n    %f = builtin.function_ref @sext_result : () -> {n};\n    %r = builtin.call_indirect %f() : extern(\"C\") () -> ({sext});\n    builtin.ret %r : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @call_zext_param(%x: {zext}) -> ({zext}) {{\n    %r = builtin.call @zext_param(%x) : extern(\"C\") ({zext}) -> ({zext});\n    builtin.ret %r : ({n});\n}};"),
        format!("builtin.function public extern(\"C\") @indirect_zext_param(%x: {zext}) -> ({zext}) {{\n    %f = builtin.function_ref @zext_param : ({n}) -> {n};\n    %r = builtin.call_indirect %f(%x) : extern(\"C\") ({zext}) -> ({zext});\n    builtin.ret %r : ({n});\n}};"),
    ]
    .join("\n")
}

/// The module of `t`, lowered by `convert-to-wasm` and `wasm-verify-form`, and its context, which
/// must outlive it.
fn lowered(t: Narrow) -> (Rc<Context>, ModuleRef) {
    lowered_by(&module_of(&functions(t)), "convert-to-wasm,wasm-verify-form")
}

/// Runs `@name` of `module` on `arguments`: what it returns, and every call run.
fn run_logged(
    module: ModuleRef,
    name: &str,
    arguments: &[Val],
) -> (Outcome, Vec<(String, Vec<Val>)>) {
    let mut machine = Machine::new(0);
    let outcome = run_in(&mut machine, function_named(module, name), arguments);
    (outcome, machine.calls)
}

/// What `@name` of `module` returns on `arguments`.
fn returns(module: ModuleRef, name: &str, arguments: &[Val]) -> Outcome {
    run_logged(module, name, arguments).0
}

#[test]
fn a_callee_masks_each_narrow_parameter_not_marked_zext_and_trusts_a_zext_one() {
    for t in NARROW {
        let (_context, module) = lowered(t);
        let dirty = t.val(t.min_extended());
        let clean = Outcome::Values(vec![t.val(t.min())]);
        assert_eq!(returns(module, "plain", &[dirty]), clean, "{}\n{}", t.name, text_of(module));
        assert_eq!(returns(module, "sext_param", &[dirty]), clean, "{}", t.name);
        // Every narrow parameter, not only the first.
        assert_eq!(returns(module, "second", &[dirty, dirty]), clean, "{}", t.name);
        // A `zext` parameter is trusted: the value is used as it is.
        assert_eq!(
            returns(module, "zext_param", &[dirty]),
            Outcome::Values(vec![dirty]),
            "{}",
            t.name
        );
    }
}

#[test]
fn a_sext_result_is_sign_extended_at_a_return_and_at_an_immediate_return() {
    for t in NARROW {
        let (_context, module) = lowered(t);
        let extended = Outcome::Values(vec![t.val(t.min_extended())]);
        assert_eq!(
            returns(module, "sext_result", &[]),
            extended,
            "{}\n{}",
            t.name,
            text_of(module)
        );
        assert_eq!(returns(module, "sext_immediate", &[]), extended, "{}", t.name);
        // An unmarked result is zero-extended.
        assert_eq!(
            returns(module, "plain_immediate", &[]),
            Outcome::Values(vec![t.val(t.min())]),
            "{}",
            t.name
        );
    }
}

#[test]
fn a_caller_sign_extends_each_argument_of_a_narrow_sext_parameter() {
    // Through a call, a tail call, an indirect call and an indirect tail call: the callee `@take`
    // is given its `sext` argument sign-extended and its unmarked one as it is.
    for t in NARROW {
        let (_context, module) = lowered(t);
        let min = t.val(t.min());
        for caller in ["call_take", "tail_take", "indirect_take", "tail_indirect_take"] {
            let (outcome, calls) = run_logged(module, caller, &[min, min]);
            assert_eq!(outcome, Outcome::Values(vec![min]), "{} {caller}", t.name);
            let (_, arguments) = calls
                .iter()
                .find(|(name, _)| name == "take")
                .unwrap_or_else(|| panic!("{} {caller} did not call @take: {calls:?}", t.name));
            assert_eq!(
                arguments,
                &vec![t.val(t.min_extended()), min],
                "{} {caller}\n{}",
                t.name,
                text_of(module)
            );
        }
    }
}

#[test]
fn a_caller_masks_each_narrow_result_not_marked_zext_and_trusts_a_zext_one() {
    for t in NARROW {
        let (_context, module) = lowered(t);
        // `@sext_result` returns its value sign-extended; the caller masks it.
        let clean = Outcome::Values(vec![t.val(t.min())]);
        for caller in ["call_sext_result", "indirect_sext_result"] {
            assert_eq!(
                returns(module, caller, &[]),
                clean,
                "{} {caller}\n{}",
                t.name,
                text_of(module)
            );
        }
        // A dirty value through a chain of `zext` parameters and results is never masked.
        let dirty = t.val(t.min_extended());
        for caller in ["call_zext_param", "indirect_zext_param"] {
            assert_eq!(
                returns(module, caller, &[dirty]),
                Outcome::Values(vec![dirty]),
                "{} {caller}\n{}",
                t.name,
                text_of(module)
            );
        }
    }
}
