//! The analyses of this crate over `builtin.tail_call` and `builtin.tail_call_indirect`: a tail
//! call is opaque to the reasoning across calls. The function that holds one has an exit whose
//! returned values are unknown, and its callee has a use that is no known call site.
//!
//! Each test that expects an analysis to give up has a control: the same program with the tail
//! call replaced by what the analysis can follow, whose fact is then known. That keeps the
//! assertion from passing because the analysis never learns anything.

use alloc::{rc::Rc, string::ToString, vec::Vec};

use runic_ir::{
    Context, Operation, OperationRef, ProgramPoint, ValueRef, dialects::builtin::Function,
    pass::AnalysisManager,
};

use super::parse_any;
use crate::{
    DataFlowConfig, DataFlowSolver, Lattice,
    analyses::{
        CallableUseSnapshot, DeadCodeAnalysis, LivenessAnalysis, SparseConstantPropagation,
        constant_propagation::ConstantValue,
        dce::{Executable, PredecessorState},
    },
};

/// What constant propagation knows of a value.
#[derive(Debug, PartialEq, Eq)]
enum Fact {
    /// The analysis has no information: nothing reached the value.
    Unvisited,
    /// The value is not known to be one constant.
    Unknown,
    /// The value is this constant.
    Constant(u128),
}

struct Program {
    #[allow(unused)]
    context: Rc<Context>,
    module: OperationRef,
    solver: DataFlowSolver,
}

impl Program {
    /// Parses `source`, a module, and runs the dead-code and constant analyses over the whole
    /// module, across calls.
    fn solve(source: &str) -> Self {
        let (context, module) = parse_any(source, true).unwrap_or_else(|err| panic!("{err}"));
        let mut config = DataFlowConfig::default();
        config.set_interprocedural(true);
        let mut solver = DataFlowSolver::new(config);
        solver.load::<DeadCodeAnalysis>();
        solver.load::<SparseConstantPropagation>();
        solver
            .initialize_and_run(&module.borrow(), AnalysisManager::new(module, None))
            .expect("the analyses converge");
        Self {
            context,
            module,
            solver,
        }
    }

    fn function(&self, name: &str) -> OperationRef {
        let mut found = None;
        self.module.borrow().prewalk_all(|op: &Operation| {
            if let Some(function) = op.downcast_ref::<Function>()
                && function.name().as_str() == name
            {
                found = Some(op.as_operation_ref());
            }
        });
        found.unwrap_or_else(|| panic!("no function @{name}"))
    }

    /// The `index`th operation in `function` whose name is `name`, in pre-order.
    fn op(&self, function: &str, name: &str, index: usize) -> OperationRef {
        let mut found = Vec::new();
        self.function(function).borrow().prewalk_all(|op: &Operation| {
            if op.name().to_string() == name {
                found.push(op.as_operation_ref());
            }
        });
        *found
            .get(index)
            .unwrap_or_else(|| panic!("no {name} number {index} in @{function}"))
    }

    fn result(&self, function: &str, op: &str, index: usize) -> ValueRef {
        self.op(function, op, index).borrow().results()[0].borrow().as_value_ref()
    }

    fn argument(&self, function: &str, index: usize) -> ValueRef {
        let function = self.function(function);
        let function = function.borrow();
        let function = function.downcast_ref::<Function>().unwrap();
        function.entry_block().borrow().arguments()[index].borrow().as_value_ref()
    }

    fn fact(&self, value: ValueRef) -> Fact {
        let lattice = self
            .solver
            .get::<Lattice<ConstantValue>, _>(&value)
            .expect("the value has a lattice");
        if lattice.value().is_uninitialized() {
            return Fact::Unvisited;
        }
        match lattice.value().constant_value() {
            None => Fact::Unknown,
            Some(constant) => Fact::Constant(
                constant
                    .borrow()
                    .as_attr()
                    .as_trait::<dyn runic_ir::attributes::IntegerLikeAttr>()
                    .expect("an integer constant")
                    .as_int()
                    .try_unsigned::<u64>()
                    .expect("fits") as u128,
            ),
        }
    }

    /// The known predecessors of the point after `op` (a call, or a function), and whether they
    /// are all there is.
    fn predecessors(&self, op: OperationRef) -> (Vec<OperationRef>, bool) {
        let state = self
            .solver
            .get::<PredecessorState, _>(&ProgramPoint::after(op))
            .expect("the point has a predecessor state");
        (state.known_predecessors().to_vec(), state.all_predecessors_known())
    }

    /// Whether the entry block of `function` is known to be executed.
    fn is_entered(&self, function: &str) -> bool {
        let function = self.function(function);
        let function = function.borrow();
        let entry = function.downcast_ref::<Function>().unwrap().entry_block();
        self.solver
            .get::<Executable, _>(&ProgramPoint::at_start_of(entry))
            .is_some_and(|state| state.is_live())
    }
}

/// `@g` is private and returns its argument; `@by_call` calls it with 1 and `@by_tail_call` tail
/// calls it with 2. `@by_tail_call_only` is the control's other half: `@h` is private, tail-called
/// with the constant 3 by its one caller.
const TWO_KINDS_OF_CALLER: &str = r#"
builtin.module public @m {
    builtin.function private extern("C") @g(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function public extern("C") @by_call() -> i32 {
        %one = arith.constant 1 : i32;
        %r = builtin.call @g(%one) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);
    };
    builtin.function public extern("C") @by_tail_call() -> i32 {
        %two = arith.constant 2 : i32;
        builtin.tail_call @g(%two) : extern("C") (i32) -> i32;
    };
    builtin.function private extern("C") @h(%y: i32) -> i32 {
        builtin.ret %y : (i32);
    };
    builtin.function public extern("C") @by_tail_call_only() -> i32 {
        %three = arith.constant 3 : i32;
        builtin.tail_call @h(%three) : extern("C") (i32) -> i32;
    };
};"#;

/// The same module without the tail calls: `@g` has the one caller, and everything is known.
const ONE_KIND_OF_CALLER: &str = r#"
builtin.module public @m {
    builtin.function private extern("C") @g(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function public extern("C") @by_call() -> i32 {
        %one = arith.constant 1 : i32;
        %r = builtin.call @g(%one) : extern("C") (i32) -> i32;
        builtin.ret %r : (i32);
    };
};"#;

/// A callable that is the callee of a tail call does not have all of its uses known: the tail
/// call is a use that is no known call site, so the callee is address-taken and has unknown
/// callers, and the `builtin.call` stays a known caller.
#[test]
fn a_callee_of_a_tail_call_has_a_use_that_is_no_known_call_site() {
    let program = Program::solve(TWO_KINDS_OF_CALLER);
    let module = program.module;
    let snapshot = CallableUseSnapshot::new(&module.borrow());
    let g = program.function("g");
    let call = program.op("by_call", "builtin.call", 0);
    let info = snapshot
        .iter()
        .find(|(target, _)| target.as_operation_ref() == g)
        .map(|(_, info)| info)
        .expect("the callable has a use record");
    assert_eq!(info.known_callers(), &[call]);
    assert!(info.is_address_taken(), "the tail call is a use that is no call site");
    assert!(info.has_unknown_callers());

    // The callee reached only by tail calls has no known caller at all, and unknown callers.
    let h = program.function("h");
    let info = snapshot
        .iter()
        .find(|(target, _)| target.as_operation_ref() == h)
        .map(|(_, info)| info)
        .expect("the callable has a use record");
    assert!(info.known_callers().is_empty());
    assert!(info.has_unknown_callers());
}

/// The control of the above: with only the `builtin.call`, every use of `@g` is known.
#[test]
fn a_callee_with_only_calls_has_all_its_uses_known() {
    let program = Program::solve(ONE_KIND_OF_CALLER);
    let module = program.module;
    let snapshot = CallableUseSnapshot::new(&module.borrow());
    let g = program.function("g");
    let (_, info) = snapshot
        .iter()
        .find(|(target, _)| target.as_operation_ref() == g)
        .expect("the callable has a use record");
    assert!(!info.is_address_taken());
    assert!(!info.has_unknown_callers());
}

/// A private function called with 1 by a call and with 2 by a tail call has no constant
/// argument; called only by the call it has 1.
#[test]
fn an_argument_of_a_function_tail_called_with_another_value_is_not_a_constant() {
    let program = Program::solve(TWO_KINDS_OF_CALLER);
    assert_eq!(program.fact(program.argument("g", 0)), Fact::Unknown);
    // The result of the call is not what was passed in either.
    assert_eq!(program.fact(program.result("by_call", "builtin.call", 0)), Fact::Unknown);

    let control = Program::solve(ONE_KIND_OF_CALLER);
    assert_eq!(control.fact(control.argument("g", 0)), Fact::Constant(1));
    assert_eq!(control.fact(control.result("by_call", "builtin.call", 0)), Fact::Constant(1));
}

/// A function reached only by a tail call is not known to receive the constant the tail call
/// passes: sound, if imprecise, and recorded as the design.
#[test]
fn an_argument_of_a_function_reached_only_by_a_tail_call_is_not_a_constant() {
    let program = Program::solve(TWO_KINDS_OF_CALLER);
    assert_eq!(program.fact(program.argument("h", 0)), Fact::Unknown);
}

const RESULTS_OF_TAIL_CALLERS: &str = r#"
builtin.module public @m {
    builtin.function private extern("C") @five() -> i32 {
        %c = arith.constant 5 : i32;
        builtin.ret %c : (i32);
    };
    builtin.function private extern("C") @nine() -> i32 {
        %c = arith.constant 9 : i32;
        builtin.ret %c : (i32);
    };
    // Returns what @five returns, through a tail call.
    builtin.function private extern("C") @through_tail_call() -> i32 {
        builtin.tail_call @five() : extern("C") () -> i32;
    };
    builtin.function public extern("C") @use_through_tail_call() -> i32 {
        %r = builtin.call @through_tail_call() : extern("C") () -> i32;
        builtin.ret %r : (i32);
    };
    // The control: returns what @five returns, through a call.
    builtin.function private extern("C") @through_call() -> i32 {
        %r = builtin.call @five() : extern("C") () -> i32;
        builtin.ret %r : (i32);
    };
    builtin.function public extern("C") @use_through_call() -> i32 {
        %r = builtin.call @through_call() : extern("C") () -> i32;
        builtin.ret %r : (i32);
    };
    // Two exits: a return of 7 and a tail call of a function that returns 9.
    builtin.function private extern("C") @two_exits(%c: i1) -> i32 {
        cf.cond_br %c ^seven, ^nine : (i1);
    ^seven:
        %s = arith.constant 7 : i32;
        builtin.ret %s : (i32);
    ^nine:
        builtin.tail_call @nine() : extern("C") () -> i32;
    };
    builtin.function public extern("C") @use_two_exits(%c: i1) -> i32 {
        %r = builtin.call @two_exits(%c) : extern("C") (i1) -> i32;
        builtin.ret %r : (i32);
    };
    // The control: both exits return 7.
    builtin.function private extern("C") @both_seven(%c: i1) -> i32 {
        cf.cond_br %c ^a, ^b : (i1);
    ^a:
        %s = arith.constant 7 : i32;
        builtin.ret %s : (i32);
    ^b:
        %t = arith.constant 7 : i32;
        builtin.ret %t : (i32);
    };
    builtin.function public extern("C") @use_both_seven(%c: i1) -> i32 {
        %r = builtin.call @both_seven(%c) : extern("C") (i1) -> i32;
        builtin.ret %r : (i32);
    };
    // A tail call inside an scf.if arm, and a return of 4 after it.
    builtin.function private extern("C") @tail_call_in_arm(%c: i1) -> i32 {
        scf.if %c then {
            builtin.tail_call @nine() : extern("C") () -> i32;
        } else {
            scf.yield : ();
        } : (i1) -> ();
        %n = arith.constant 4 : i32;
        builtin.ret %n : (i32);
    };
    builtin.function public extern("C") @use_tail_call_in_arm(%c: i1) -> i32 {
        %r = builtin.call @tail_call_in_arm(%c) : extern("C") (i1) -> i32;
        builtin.ret %r : (i32);
    };
    builtin.function private extern("C") @ret_in_arm(%c: i1) -> i32 {
        scf.if %c then {
            %m = arith.constant 8 : i32;
            builtin.ret %m : (i32);
        } else {
            scf.yield : ();
        } : (i1) -> ();
        %n = arith.constant 4 : i32;
        builtin.ret %n : (i32);
    };
    builtin.function public extern("C") @use_ret_in_arm(%c: i1) -> i32 {
        %r = builtin.call @ret_in_arm(%c) : extern("C") (i1) -> i32;
        builtin.ret %r : (i32);
    };
    // The control: the same with the arm yielding.
    builtin.function private extern("C") @yield_in_arm(%c: i1) -> i32 {
        scf.if %c then {
            scf.yield : ();
        } else {
            scf.yield : ();
        } : (i1) -> ();
        %n = arith.constant 4 : i32;
        builtin.ret %n : (i32);
    };
    builtin.function public extern("C") @use_yield_in_arm(%c: i1) -> i32 {
        %r = builtin.call @yield_in_arm(%c) : extern("C") (i1) -> i32;
        builtin.ret %r : (i32);
    };
};"#;

/// The result of a call of a function that leaves by a tail call is never folded to anything a
/// function it does not return from could have: the exit's returned values are unknown. The
/// control, which returns what the same callee returns through a call, is the constant.
#[test]
fn the_result_of_a_call_of_a_function_that_tail_calls_is_unknown() {
    let program = Program::solve(RESULTS_OF_TAIL_CALLERS);
    let result = program.result("use_through_tail_call", "builtin.call", 0);
    assert_eq!(program.fact(result), Fact::Unknown);
    let control = program.result("use_through_call", "builtin.call", 0);
    assert_eq!(program.fact(control), Fact::Constant(5));
}

/// A function with a return of 7 and a tail call that ends in a function returning 9: a call of
/// it is not 7, though the one return it has says so. The control with two returns of 7 is 7.
#[test]
fn the_result_of_a_call_of_a_function_with_a_return_and_a_tail_call_is_not_the_return() {
    let program = Program::solve(RESULTS_OF_TAIL_CALLERS);
    let result = program.result("use_two_exits", "builtin.call", 0);
    assert_eq!(program.fact(result), Fact::Unknown);
    let control = program.result("use_both_seven", "builtin.call", 0);
    assert_eq!(program.fact(control), Fact::Constant(7));
}

/// A tail call in the arm of an `scf.if` is an exit of the function, not a yield: the one return
/// after the `scf.if` is not all there is to what the call returns.
#[test]
fn a_tail_call_in_the_arm_of_an_if_is_an_exit_of_the_function() {
    let program = Program::solve(RESULTS_OF_TAIL_CALLERS);
    let result = program.result("use_tail_call_in_arm", "builtin.call", 0);
    assert_eq!(program.fact(result), Fact::Unknown);
    let control = program.result("use_yield_in_arm", "builtin.call", 0);
    assert_eq!(program.fact(control), Fact::Constant(4));
}

/// The dead-code analysis marks the call sites of a function that tail calls as having unknown
/// return points: the tail call is a predecessor of the point after the call that it does not
/// name. The callee of the tail call has unknown predecessors too, and its body is entered.
#[test]
fn the_dead_code_analysis_leaves_the_returns_of_a_tail_calling_function_unknown() {
    let program = Program::solve(RESULTS_OF_TAIL_CALLERS);
    let call = program.op("use_through_tail_call", "builtin.call", 0);
    let (known, all_known) = program.predecessors(call);
    assert!(!all_known, "{known:?}");

    let control = program.op("use_through_call", "builtin.call", 0);
    let (known, all_known) = program.predecessors(control);
    assert!(all_known);
    assert_eq!(known, [program.op("through_call", "builtin.ret", 0)]);

    // The callee of the tail call has callers the analysis does not know, so its body is entered.
    let (known, all_known) = program.predecessors(program.function("five"));
    assert!(!all_known);
    assert_eq!(known, [program.op("through_call", "builtin.call", 0)]);
    assert!(program.is_entered("five"));
}

/// A private function that only a tail call reaches is entered, its returns are all there is to
/// what the tail calling function returns, and the call site of the caller is not resolved to
/// them.
#[test]
fn a_private_function_reached_only_by_a_tail_call_is_entered() {
    let program = Program::solve(
        r#"
builtin.module public @m {
    builtin.function private extern("C") @target(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function private extern("C") @unreferenced(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function public extern("C") @entry(%a: i32) -> i32 {
        builtin.tail_call @target(%a) : extern("C") (i32) -> i32;
    };
};"#,
    );
    assert!(program.is_entered("target"));
    assert!(!program.is_entered("unreferenced"), "the control is not entered");
    assert_eq!(program.fact(program.argument("target", 0)), Fact::Unknown);
    assert!(program.predecessors(program.function("target")).0.is_empty());
}

/// The indirect form too: nothing is known of its callee, and the function has an unknown exit.
#[test]
fn a_tail_call_through_a_function_value_leaves_the_exit_unknown() {
    let program = Program::solve(
        r#"
builtin.module public @m {
    builtin.function private extern("C") @five() -> i32 {
        %c = arith.constant 5 : i32;
        builtin.ret %c : (i32);
    };
    builtin.function private extern("C") @indirect() -> i32 {
        %f = builtin.function_ref @five : () -> i32;
        builtin.tail_call_indirect %f() : extern("C") () -> i32;
    };
    builtin.function public extern("C") @user() -> i32 {
        %r = builtin.call @indirect() : extern("C") () -> i32;
        builtin.ret %r : (i32);
    };
};"#,
    );
    assert_eq!(program.fact(program.result("user", "builtin.call", 0)), Fact::Unknown);
    let call = program.op("user", "builtin.call", 0);
    assert!(!program.predecessors(call).1);
    // `@five` is used by a reference, so its callers are unknown too.
    assert!(program.is_entered("five"));
}

/// Liveness: a value that only a tail call uses is live up to the tail call and not after it, and
/// the arguments of the tail call are not read as the values the function returns, so a value
/// that a `builtin.ret` in another block returns is not made live by it.
#[test]
fn a_value_only_a_tail_call_uses_is_live_up_to_the_tail_call() {
    let program = Program::solve(
        r#"
builtin.module public @m {
    builtin.function private extern("C") @g(%x: i32) -> i32 {
        builtin.ret %x : (i32);
    };
    builtin.function public extern("C") @f(%c: i1, %a: i32, %b: i32) -> i32 {
        %sum = arith.add %a, %b;
        %unused = arith.add %a, %a;
        cf.cond_br %c ^tail, ^ret : (i1);
    ^tail:
        builtin.tail_call @g(%sum) : extern("C") (i32) -> i32;
    ^ret:
        builtin.ret %b : (i32);
    };
};"#,
    );
    let function = program.function("f");
    let liveness = AnalysisManager::new(function, None)
        .get_analysis_for::<LivenessAnalysis, Operation>()
        .expect("liveness runs");
    let tail_call = program.op("f", "builtin.tail_call", 0);
    let ret = program.op("f", "builtin.ret", 0);
    let sum = program.result("f", "arith.add", 0);
    let unused = program.result("f", "arith.add", 1);
    let b = program.argument("f", 2);
    let tail_call = tail_call.borrow();
    assert!(liveness.is_live_before(sum, &tail_call));
    assert!(!liveness.is_live_after(sum, &tail_call), "nothing runs after a tail call");
    assert!(
        !liveness.is_live_before(b, &tail_call),
        "`%b` is returned by the other exit only"
    );
    assert!(liveness.is_live_before(b, &ret.borrow()));
    assert!(!liveness.is_live_before(sum, &ret.borrow()));
    // `%sum` is live from its definition to the tail call, so it is live at the branch.
    let branch = program.op("f", "cf.cond_br", 0);
    assert!(liveness.is_live_before(sum, &branch.borrow()));
    assert!(!liveness.is_live_after(unused, &branch.borrow()));
}

/// A tail call is not a callee's return: the lattice of a block the function never leaves
/// through a return keeps nothing from the tail call's arguments (the constant 1 passed as the
/// argument is no value of the call's result).
#[test]
fn the_arguments_of_a_tail_call_are_not_the_values_the_function_returns() {
    let program = Program::solve(
        r#"
builtin.module public @m {
    builtin.function private extern("C") @g(%x: i32) -> i32 {
        %ten = arith.constant 10 : i32;
        %r = arith.add %x, %ten;
        builtin.ret %r : (i32);
    };
    builtin.function private extern("C") @f() -> i32 {
        %one = arith.constant 1 : i32;
        builtin.tail_call @g(%one) : extern("C") (i32) -> i32;
    };
    builtin.function public extern("C") @user() -> i32 {
        %r = builtin.call @f() : extern("C") () -> i32;
        builtin.ret %r : (i32);
    };
};"#,
    );
    // Were the argument read as the returned value, the result would be the constant 1.
    assert_ne!(program.fact(program.result("user", "builtin.call", 0)), Fact::Constant(1));
    assert_ne!(program.fact(program.result("user", "builtin.call", 0)), Fact::Constant(11));
}

/// The same for a `builtin.ret` in the arm: the dead-code analysis treats an exit nested in
/// region control flow as the tail call is treated, one code path for both.
#[test]
fn a_return_in_the_arm_of_an_if_is_an_exit_of_the_function() {
    let program = Program::solve(RESULTS_OF_TAIL_CALLERS);
    let result = program.result("use_ret_in_arm", "builtin.call", 0);
    assert_eq!(program.fact(result), Fact::Unknown);
}

/// A module whose private function `@k` runs `body` and then returns 4, and whose `@main` calls
/// it. `@g` is a function the body may tail call.
fn k_with_body(body: &str) -> Program {
    Program::solve(&alloc::format!(
        r#"builtin.module public @m {{
    builtin.function public extern("C") @g() -> i32;
    builtin.function private extern("C") @k(%c: i1, %s: i32) -> i32 {{
{body}
        %four = arith.constant 4 : i32;
        builtin.ret %four : (i32);
    }};
    builtin.function public extern("C") @main(%c: i1, %s: i32) -> i32 {{
        %r = builtin.call @k(%c, %s) : extern("C") (i1, i32) -> i32;
        builtin.ret %r : (i32);
    }};
}};"#
    ))
}

/// What a call of `@k` is to constant propagation, and which operations the dead-code analysis
/// registers as its return points, in the order it lists them.
fn call_of_k(program: &Program) -> (Fact, Vec<OperationRef>, bool) {
    let fact = program.fact(program.result("main", "builtin.call", 0));
    let (known, all_known) = program.predecessors(program.op("main", "builtin.call", 0));
    (fact, known, all_known)
}

const RET_NINE: &str = "%n = arith.constant 9 : i32; builtin.ret %n : (i32);";
const TAIL_CALL_G: &str = r#"builtin.tail_call @g() : extern("C") () -> i32;"#;

/// A return that is a direct child of an operation with a region that does not forward control
/// flow is an exit of the function: the call is not 4. The control with no exit is 4.
#[test]
fn a_return_directly_in_an_opaque_region_operation_is_an_exit_of_the_function() {
    let control = k_with_body("");
    assert_eq!(call_of_k(&control).0, Fact::Constant(4));
    let program = k_with_body(&alloc::format!(
        r#""test.reads_locals_and_holds"() ({{ {RET_NINE} }}) : () -> ();"#
    ));
    let (fact, known, _) = call_of_k(&program);
    assert_eq!(fact, Fact::Unknown);
    // The exit is registered once, with the return after it.
    assert_eq!(known.len(), 2, "{known:?}");
    assert!(known.contains(&program.op("k", "builtin.ret", 0)));
    assert!(known.contains(&program.op("k", "builtin.ret", 1)));
}

/// The same for a tail call: its returned values are unknown, so the call is unknown.
#[test]
fn a_tail_call_directly_in_an_opaque_region_operation_is_an_exit_of_the_function() {
    let program = k_with_body(&alloc::format!(
        r#""test.reads_locals_and_holds"() ({{ {TAIL_CALL_G} }}) : () -> ();"#
    ));
    let (fact, _, all_known) = call_of_k(&program);
    assert_eq!(fact, Fact::Unknown);
    assert!(!all_known);
}

/// An exit two region operations down, in an `scf.if` in an `scf.if`, is an exit of the function,
/// registered once however deep it sits.
#[test]
fn a_return_two_ifs_down_is_an_exit_of_the_function_registered_once() {
    let program = k_with_body(&alloc::format!(
        r#"scf.if %c then {{
            scf.if %c then {{ {RET_NINE} }} else {{ scf.yield : (); }} : (i1) -> ();
            scf.yield : ();
        }} else {{ scf.yield : (); }} : (i1) -> ();"#
    ));
    let (fact, known, _) = call_of_k(&program);
    assert_eq!(fact, Fact::Unknown);
    assert_eq!(known.len(), 2, "{known:?}");
}

/// A tail call two region operations down, one of them an opaque one, is an exit of the function.
#[test]
fn a_tail_call_in_an_if_in_an_opaque_region_operation_is_an_exit_of_the_function() {
    let program = k_with_body(&alloc::format!(
        r#""test.reads_locals_and_holds"() ({{
            scf.if %c then {{ {TAIL_CALL_G} }} else {{ scf.yield : (); }} : (i1) -> ();
            "test.ret"() : () -> ();
        }}) : () -> ();"#
    ));
    let (fact, _, all_known) = call_of_k(&program);
    assert_eq!(fact, Fact::Unknown);
    assert!(!all_known);
}

/// A return in the body of an `scf.while` is an exit of the function.
#[test]
fn a_return_in_the_body_of_a_while_is_an_exit_of_the_function() {
    let program = k_with_body(&alloc::format!(
        r#"scf.while before {{ scf.condition %c : (i1); }} after {{
            scf.if %c then {{ {RET_NINE} }} else {{ scf.yield : (); }} : (i1) -> ();
            scf.yield : ();
        }} : () -> ();"#
    ));
    assert_eq!(call_of_k(&program).0, Fact::Unknown);
}

/// A return in a case of an `scf.index_switch` is an exit of the function.
#[test]
fn a_return_in_a_case_of_an_index_switch_is_an_exit_of_the_function() {
    let program = k_with_body(&alloc::format!(
        r#"scf.index_switch %s case 1 {{ {RET_NINE} }} default {{ scf.yield : (); }} : ();"#
    ));
    assert_eq!(call_of_k(&program).0, Fact::Unknown);
}
