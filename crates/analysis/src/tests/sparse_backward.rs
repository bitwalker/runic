//! The sparse backward data-flow framework, through a small analysis built on it in the manner of
//! upstream MLIR's `LivenessAnalysis`: a value is *demanded* when an operation with an effect, a
//! branch condition or a return to an unknown caller uses it, or when an operation one of whose
//! results is demanded uses it.
//!
//! The analysis is a meet semi-lattice of two values: not demanded (the top, where every value
//! starts) and demanded (the bottom). Demand flows from uses to definitions: from the results of
//! an operation to its operands, from the arguments of a block to the operands a branch passes
//! them, from the inputs of a region to the operands its predecessors forward, from the entry
//! arguments of a callee to the operands of its calls, and from the results of the calls of a
//! callee to the operands of its returns.
//!
//! Each test asserts the whole set of demanded values, by the names the program gives them, and
//! most have a control: the same program with the edge in question changed, which shows the answer
//! is because of that edge. The programs where an operand's lattice and the lattice read along an
//! edge are one lattice (a block argument passed back to its own block, a value used twice, a
//! continue carrying its own entry arguments, a recursive call) are written so that the lattice
//! read changes after the edge is first visited: the answer must still be the fixed point.

use alloc::{collections::BTreeSet, rc::Rc, string::String, vec::Vec};

use runic_ir::{
    Backward, CallOpInterface, Context, OpOperandImpl, Operation, OperationRef, ProgramPoint,
    RegionBranchTerminatorOpInterface, Report, Spanned, ValueRef,
    dialects::test::TestDialect,
    parse::{self, ParserConfig},
    pass::AnalysisManager,
    testing::text::describe,
};

use crate::{
    AnalysisState, AnalysisStateGuard, AnalysisStateGuardMut, BuildableDataFlowAnalysis,
    DataFlowConfig, DataFlowSolver, Lattice, LatticeLike, SparseBackwardDataFlowAnalysis,
    SparseLattice,
    analyses::{DeadCodeAnalysis, SparseConstantPropagation},
    sparse::{OperandLattices, SparseDataFlowAnalysis},
};

/// Whether a value is demanded: `true` is the bottom of the lattice, and `meet` moves towards it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Demand(pub(super) bool);

impl LatticeLike for Demand {
    fn join(&self, other: &Self) -> Self {
        Self(self.0 && other.0)
    }

    fn meet(&self, other: &Self) -> Self {
        Self(self.0 || other.0)
    }
}

pub(super) const DEMANDED: Demand = Demand(true);

/// The test analysis.
#[derive(Default)]
pub(super) struct Demanded;

impl BuildableDataFlowAnalysis for Demanded {
    type Strategy = SparseDataFlowAnalysis<Self, Backward>;

    fn new(_solver: &mut DataFlowSolver) -> Self {
        Self
    }
}

impl Demanded {
    pub(super) fn demand(operand: &OpOperandImpl, solver: &mut DataFlowSolver) {
        let mut lattice = solver.get_or_create_mut::<Lattice<Demand>, _>(operand.as_value_ref());
        SparseLattice::meet(&mut lattice, &DEMANDED);
    }
}

impl SparseBackwardDataFlowAnalysis for Demanded {
    type Lattice = Lattice<Demand>;

    fn debug_name(&self) -> &'static str {
        "demanded"
    }

    fn visit_operation(
        &self,
        op: &Operation,
        operands: &mut OperandLattices<'_, Self::Lattice>,
        results: &[AnalysisStateGuard<'_, Self::Lattice>],
        _solver: &mut DataFlowSolver,
    ) -> Result<(), Report> {
        let has_effect = !op.would_be_trivially_dead();
        let result_demanded = results.iter().any(|result| result.value().0);
        if has_effect || result_demanded {
            for index in 0..operands.len() {
                SparseLattice::meet(&mut operands[index], &DEMANDED);
            }
        }
        Ok(())
    }

    fn visit_branch_operand(&self, operand: &OpOperandImpl, solver: &mut DataFlowSolver) {
        Self::demand(operand, solver);
    }

    fn visit_call_operand(&self, operand: &OpOperandImpl, solver: &mut DataFlowSolver) {
        Self::demand(operand, solver);
    }

    fn set_to_exit_state(&self, lattice: &mut AnalysisStateGuardMut<'_, Self::Lattice>) {
        SparseLattice::meet(lattice, &DEMANDED);
    }
}

pub(super) struct Program {
    #[allow(unused)]
    context: Rc<Context>,
    source: String,
    pub(super) root: OperationRef,
}

impl Program {
    pub(super) fn parse(source: &str) -> Self {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<runic_dialect_arith::ArithDialect>();
        context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
        context.get_or_register_dialect::<runic_dialect_scf::ScfDialect>();
        context.get_or_register_dialect::<TestDialect>();
        let config = ParserConfig::new(context.clone());
        let root = parse::parse_any(config, "sparse_backward", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{}\n{source}", describe(&err)));
        Self {
            context,
            source: source.into(),
            root,
        }
    }

    /// The name the source gives `value`: a block argument's span is its name, and a result's
    /// name is read from the text before its operation, `%a, %b = `.
    pub(super) fn name_of(&self, value: ValueRef) -> String {
        let value = value.borrow();
        if let Some(result) = value.downcast_ref::<runic_ir::OpResult>() {
            let owner = result.owner();
            let start = owner.borrow().span().range().start() as usize;
            let line_start = self.source[..start].rfind('\n').map_or(0, |at| at + 1);
            let names = self.source[line_start..start].trim().trim_end_matches('=');
            let name = names.split(',').nth(result.index()).expect("the result has a name");
            name.trim().into()
        } else {
            let range = value.span().range().into_slice_index();
            self.source[range].into()
        }
    }

    /// Every value the program defines, in pre-order.
    pub(super) fn values(&self) -> Vec<ValueRef> {
        let mut values = Vec::new();
        self.root.borrow().prewalk_all(|op: &Operation| {
            for region in op.regions() {
                for block in region.body() {
                    for argument in block.arguments() {
                        values.push(argument.borrow().as_value_ref());
                    }
                }
            }
            for result in op.results().iter() {
                values.push(result.borrow().as_value_ref());
            }
        });
        values
    }

    /// The names of the values the test analysis finds demanded, interprocedurally, with
    /// dead-code analysis and constant propagation (which dead-code analysis needs to follow a
    /// branch whose operands are not constants it knows).
    fn demanded(&self) -> BTreeSet<String> {
        self.demanded_by::<Demanded>()
    }

    /// The names of the values `A`, an analysis over the lattice of the test analysis, finds
    /// demanded, run as [Self::demanded] runs the test analysis.
    pub(super) fn demanded_by<A: BuildableDataFlowAnalysis + 'static>(&self) -> BTreeSet<String> {
        let mut config = DataFlowConfig::default();
        config.set_interprocedural(true);
        let mut solver = DataFlowSolver::new(config);
        solver.load::<DeadCodeAnalysis>();
        solver.load::<SparseConstantPropagation>();
        solver.load::<A>();
        solver
            .initialize_and_run(&self.root.borrow(), AnalysisManager::new(self.root, None))
            .expect("the analyses converge");
        self.values()
            .into_iter()
            .filter(|value| {
                solver.get::<Lattice<Demand>, _>(value).is_some_and(|lattice| lattice.value().0)
            })
            .map(|value| self.name_of(value))
            .collect()
    }
}

/// The demanded values of `source`.
pub(super) fn demanded(source: &str) -> BTreeSet<String> {
    Program::parse(source).demanded()
}

pub(super) fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| String::from(*name)).collect()
}

#[test]
fn the_program_names_its_values() {
    let program = Program::parse(
        r#"builtin.function public extern("C") @f(%p: ptr<i32>, %a: i32) {
    %s, %t = test.loop %a, %a {
    ^head(%x: i32, %y: i32):
        test.yield %x, %y : (i32, i32);
    } : (i32, i32) -> (i32, i32);
    builtin.ret;
};"#,
    );
    let names = program
        .values()
        .into_iter()
        .map(|value| program.name_of(value))
        .collect::<Vec<_>>();
    assert_eq!(names, ["%p", "%a", "%x", "%y", "%s", "%t"]);
}

/// In straight-line code, demand flows from the operand of an operation with an effect to the
/// values it is computed from, and not to a value nothing demanded uses.
#[test]
fn demand_flows_from_an_effect_to_what_its_operands_are_computed_from() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %a: i32, %b: i32) {
    %s = arith.add %a, %b;
    %unused = arith.mul %a, %b;
    %t = arith.add %s, %a;
    STORE
    builtin.ret;
};"#;
    let with = source.replace("STORE", "test.store %p, %t : (ptr<i32>, i32);");
    assert_eq!(demanded(&with), set(&["%p", "%t", "%s", "%a", "%b"]));
    let without = source.replace("STORE", "");
    assert_eq!(demanded(&without), set(&[]));
}

/// An operation that uses one value twice gives the analysis one lattice in two places. The use
/// becomes demanded after the operation is first visited, through the back edge of a loop.
#[test]
fn an_operation_may_use_one_value_twice() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32, %b: i32) {
    cf.br ^head(%a : i32);
^head(%x: i32):
    %d = arith.add %x, %x;
    %e = arith.add %b, %b;
    cf.cond_br %c ^head(%d : i32), ^exit : (i1);
^exit:
    test.store %p, %x : (ptr<i32>, i32);
    builtin.ret;
};"#;
    assert_eq!(demanded(source), set(&["%p", "%c", "%a", "%x", "%d"]));
    // The control: the back edge carries `%x`, not `%d`, so nothing demands `%d`.
    let control = source.replace("^head(%d : i32), ^exit", "^head(%x : i32), ^exit");
    assert_eq!(demanded(&control), set(&["%p", "%c", "%a", "%x"]));
}

/// A branch's operands are demanded when the arguments they are passed to are.
#[test]
fn demand_flows_from_a_block_argument_to_the_operand_a_branch_passes_it() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %a: i32, %b: i32) {
    cf.br ^next(%a : i32);
^next(%x: i32):
    test.store %p, STORED : (ptr<i32>, i32);
    builtin.ret;
};"#;
    assert_eq!(demanded(&source.replace("STORED", "%x")), set(&["%p", "%x", "%a"]));
    assert_eq!(demanded(&source.replace("STORED", "%b")), set(&["%p", "%b"]));
}

/// A conditional branch passing one value to both successors: the value is demanded by the
/// argument of either.
#[test]
fn a_conditional_branch_may_pass_one_value_to_both_successors() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32) {
    cf.cond_br %c ^left(%a : i32), ^right(%a : i32) : (i1);
^left(%x: i32):
    LEFT
    builtin.ret;
^right(%y: i32):
    RIGHT
    builtin.ret;
};"#;
    let store = |value: &str| alloc::format!("test.store %p, {value} : (ptr<i32>, i32);");
    let left = source.replace("LEFT", &store("%x")).replace("RIGHT", "");
    assert_eq!(demanded(&left), set(&["%p", "%c", "%x", "%a"]));
    let right = source.replace("LEFT", "").replace("RIGHT", &store("%y"));
    assert_eq!(demanded(&right), set(&["%p", "%c", "%y", "%a"]));
    let neither = source.replace("LEFT", "").replace("RIGHT", "");
    assert_eq!(demanded(&neither), set(&["%c"]));
}

/// A conditional branch passing one value twice to one successor: the value is demanded by
/// either argument.
#[test]
fn a_conditional_branch_may_pass_one_value_twice_to_one_successor() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32, %b: i32) {
    cf.cond_br %c ^left(%a, %a : i32, i32), ^right(%b : i32) : (i1);
^left(%x: i32, %y: i32):
    test.store %p, STORED : (ptr<i32>, i32);
    builtin.ret;
^right(%z: i32):
    builtin.ret;
};"#;
    assert_eq!(demanded(&source.replace("STORED", "%y")), set(&["%p", "%c", "%y", "%a"]));
    assert_eq!(demanded(&source.replace("STORED", "%x")), set(&["%p", "%c", "%x", "%a"]));
    assert_eq!(demanded(&source.replace("STORED", "%b")), set(&["%p", "%c", "%b"]));
}

/// A loop whose back edge passes the header's own argument back to it. The argument becomes
/// demanded after the back edge is first visited (the use is in the exit block), and the demand
/// still reaches the operand of the entry edge.
#[test]
fn a_back_edge_may_pass_the_header_its_own_argument() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32) {
    cf.br ^head(%a : i32);
^head(%x: i32):
    cf.cond_br %c ^head(%x : i32), ^exit : (i1);
^exit:
    STORE
    builtin.ret;
};"#;
    let with = source.replace("STORE", "test.store %p, %x : (ptr<i32>, i32);");
    assert_eq!(demanded(&with), set(&["%p", "%c", "%x", "%a"]));
    let without = source.replace("STORE", "");
    assert_eq!(demanded(&without), set(&["%c"]));
}

/// A loop whose back edge swaps the header's two arguments: demand of one reaches the other
/// through the back edge, after the back edge is first visited, and then both operands of the
/// entry edge.
#[test]
fn a_back_edge_may_swap_the_headers_arguments() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32, %b: i32) {
    cf.br ^head(%a, %b : i32, i32);
^head(%x: i32, %y: i32):
    cf.cond_br %c ^head(BACK : i32, i32), ^exit : (i1);
^exit:
    test.store %p, %x : (ptr<i32>, i32);
    builtin.ret;
};"#;
    let swapped = source.replace("BACK", "%y, %x");
    assert_eq!(demanded(&swapped), set(&["%p", "%c", "%x", "%y", "%a", "%b"]));
    let same = source.replace("BACK", "%x, %y");
    assert_eq!(demanded(&same), set(&["%p", "%c", "%x", "%a"]));
}

/// The values an `scf.if` yields are demanded when its result is, and its condition is a branch
/// condition.
#[test]
fn demand_flows_through_the_regions_of_an_if() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32, %b: i32) {
    %one = arith.constant 1 : i32;
    %r = scf.if %c then {
        %t = arith.add %a, %one;
        scf.yield %t : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    STORE
    builtin.ret;
};"#;
    let with = source.replace("STORE", "test.store %p, %r : (ptr<i32>, i32);");
    assert_eq!(demanded(&with), set(&["%p", "%c", "%r", "%t", "%a", "%one", "%b"]));
    let without = source.replace("STORE", "");
    assert_eq!(demanded(&without), set(&["%c"]));
}

/// The values of an `scf.while` flow into the arguments of its regions and out through its
/// terminators; one value enters it twice. The sum is demanded only through the loop's result.
#[test]
fn demand_flows_through_the_regions_of_a_while() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %n: i32) {
    %zero = arith.constant 0 : i32;
    %one = arith.constant 1 : i32;
    %ri, %rs = scf.while %zero, %zero before {
    ^head(%i: i32, %s: i32):
        %lt = arith.cmpi ult %i, %n;
        scf.condition %lt, %i, %s : (i1, i32, i32);
    } after {
    ^body(%j: i32, %t: i32):
        %next = arith.add %j, %one;
        %sum = arith.add %t, %j;
        scf.yield %next, %sum : (i32, i32);
    } : (i32, i32) -> (i32, i32);
    STORE
    builtin.ret;
};"#;
    let counter = ["%lt", "%n", "%i", "%zero", "%next", "%j", "%one"];
    let with = source.replace("STORE", "test.store %p, %rs : (ptr<i32>, i32);");
    let mut expected = set(&counter);
    expected.extend(set(&["%p", "%rs", "%s", "%sum", "%t"]));
    assert_eq!(demanded(&with), expected);
    let without = source.replace("STORE", "");
    assert_eq!(demanded(&without), set(&counter));
}

/// A `test.loop` whose `test.continue` carries the entry arguments back: in the same places, and
/// swapped. Swapped, the use of `%y` (above the guard that holds the continue) is visited after
/// the continue, so the continue reads a lattice that changes after its first visit, and the
/// demand reaches `%x`, and through it `%a`.
#[test]
fn a_continue_may_carry_the_entry_arguments_of_its_loop() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32, %b: i32, %k: i32) {
    %r = test.loop %a, %b {
    ^head(%x: i32, %y: i32):
        %u = arith.add %y, %k;
        test.guard %c {
            test.continue CARRIED : (i32, i32);
        } : (i1);
        test.yield %u : (i32);
    } : (i32, i32) -> (i32);
    test.store %p, %r : (ptr<i32>, i32);
    builtin.ret;
};"#;
    let swapped = source.replace("CARRIED", "%y, %x");
    assert_eq!(demanded(&swapped), set(&["%p", "%c", "%r", "%u", "%y", "%k", "%x", "%a", "%b"]));
    let same = source.replace("CARRIED", "%x, %y");
    assert_eq!(demanded(&same), set(&["%p", "%c", "%r", "%u", "%y", "%k", "%b"]));
}

/// A call to a private function all of whose call sites are known: demand flows from the call's
/// result to the callee's return, and from the callee's entry argument to the call's operand.
/// The callee is visited before the caller, so the call's result changes after the return is
/// first visited.
#[test]
fn demand_flows_through_a_call_to_a_function_whose_callers_are_known() {
    let source = r#"builtin.module public @m {
    builtin.function private extern("C") @g(%gx: i32, %gy: i32) -> i32 {
        builtin.ret %gx : (i32);
    };
    builtin.function public extern("C") @f(%p: ptr<i32>, %a: i32, %b: i32) {
        %r = builtin.call @g(%a, %b) : extern("C") (i32, i32) -> i32;
        STORE
        builtin.ret;
    };
};"#;
    let with = source.replace("STORE", "test.store %p, %r : (ptr<i32>, i32);");
    assert_eq!(demanded(&with), set(&["%p", "%r", "%gx", "%a"]));
    let without = source.replace("STORE", "");
    assert_eq!(demanded(&without), set(&[]));
}

/// A recursive call passes the callee's own entry arguments back to it: in the same places, and
/// swapped. Swapped, the demand of `%hx` reaches `%hy` through the recursive call, and then the
/// outer call's `%b`.
#[test]
fn a_recursive_call_may_pass_the_callees_entry_arguments() {
    let source = r#"builtin.module public @m {
    builtin.function private extern("C") @h(%hx: i32, %hy: i32) -> i32 {
        %hr = builtin.call @h(ARGUMENTS) : extern("C") (i32, i32) -> i32;
        %hs = arith.add %hr, %hx;
        builtin.ret %hs : (i32);
    };
    builtin.function public extern("C") @f(%p: ptr<i32>, %a: i32, %b: i32) {
        %fr = builtin.call @h(%a, %b) : extern("C") (i32, i32) -> i32;
        test.store %p, %fr : (ptr<i32>, i32);
        builtin.ret;
    };
};"#;
    let swapped = source.replace("ARGUMENTS", "%hy, %hx");
    assert_eq!(demanded(&swapped), set(&["%p", "%fr", "%hs", "%hr", "%hx", "%hy", "%a", "%b"]));
    let same = source.replace("ARGUMENTS", "%hx, %hy");
    assert_eq!(demanded(&same), set(&["%p", "%fr", "%hs", "%hr", "%hx", "%a"]));
}

/// A recursive function returns the results of its own recursive call, swapped: the return's
/// operands are the call's results, so the return reads lattices it also writes. The demand of
/// the outer call's first result reaches both.
#[test]
fn a_return_may_return_the_results_of_a_recursive_call() {
    let source = r#"builtin.module public @m {
    builtin.function private extern("C") @h(%hx: i32, %hy: i32) -> (i32, i32) {
        %h1, %h2 = builtin.call @h(%hx, %hy) : extern("C") (i32, i32) -> (i32, i32);
        builtin.ret RETURNED : (i32, i32);
    };
    builtin.function public extern("C") @f(%p: ptr<i32>, %a: i32, %b: i32) {
        %f1, %f2 = builtin.call @h(%a, %b) : extern("C") (i32, i32) -> (i32, i32);
        test.store %p, %f1 : (ptr<i32>, i32);
        builtin.ret;
    };
};"#;
    let swapped = source.replace("RETURNED", "%h2, %h1");
    assert_eq!(demanded(&swapped), set(&["%p", "%f1", "%h1", "%h2"]));
    let same = source.replace("RETURNED", "%h1, %h2");
    assert_eq!(demanded(&same), set(&["%p", "%f1", "%h1"]));
}

/// A call to a declaration is a call to an unknown function: its operands are demanded. The
/// control calls a defined function that ignores its argument.
#[test]
fn the_operands_of_a_call_to_a_declaration_are_demanded() {
    let source = r#"builtin.module public @m {
    builtin.function private extern("C") @decl(i32) -> i32;
    builtin.function private extern("C") @ignore(%ix: i32) -> i32 {
        %z = arith.constant 0 : i32;
        builtin.ret %z : (i32);
    };
    builtin.function public extern("C") @f(%a: i32) {
        %r = builtin.call @CALLEE(%a) : extern("C") (i32) -> i32;
        builtin.ret;
    };
};"#;
    assert_eq!(demanded(&source.replace("CALLEE", "decl")), set(&["%a"]));
    assert_eq!(demanded(&source.replace("CALLEE", "ignore")), set(&[]));
}

/// The returns of a public function go to callers that are not known: what they return is
/// demanded. The control is the same function made private, with no callers.
#[test]
fn what_a_public_function_returns_is_demanded() {
    let source = r#"builtin.function VISIBILITY extern("C") @f(%a: i32, %b: i32) -> i32 {
    %one = arith.constant 1 : i32;
    %s = arith.add %a, %one;
    builtin.ret %s : (i32);
};"#;
    assert_eq!(demanded(&source.replace("VISIBILITY", "public")), set(&["%s", "%a", "%one"]));
    let module = alloc::format!(
        "builtin.module public @m {{\n{}\n}};",
        source.replace("VISIBILITY", "private")
    );
    assert_eq!(demanded(&module), set(&[]));
}

/// An operation in a block that dead-code analysis finds unreachable is not visited: what it uses
/// is not demanded. The control branches to the block.
#[test]
fn an_operation_in_an_unreachable_block_demands_nothing() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %b: i32) {
    ENTRY
^dead:
    test.store %p, %b : (ptr<i32>, i32);
    builtin.ret;
};"#;
    assert_eq!(demanded(&source.replace("ENTRY", "builtin.ret;")), set(&[]));
    assert_eq!(demanded(&source.replace("ENTRY", "cf.br ^dead;")), set(&["%p", "%b"]));
}

/// A return may return one value twice, and a call may pass one value twice: the return's two
/// operands are one lattice, demanded through either result of the call, and the value the call
/// passes twice is demanded through either argument.
#[test]
fn a_return_and_a_call_may_use_one_value_twice() {
    let source = r#"builtin.module public @m {
    builtin.function private extern("C") @g(%gx: i32, %gy: i32) -> (i32, i32) {
        builtin.ret %gx, %gx : (i32, i32);
    };
    builtin.function public extern("C") @f(%p: ptr<i32>, %a: i32, %b: i32) {
        %f1, %f2 = builtin.call @g(ARGUMENTS) : extern("C") (i32, i32) -> (i32, i32);
        STORE
        builtin.ret;
    };
};"#;
    let store = |value: &str| alloc::format!("test.store %p, {value} : (ptr<i32>, i32);");
    let program = |arguments: &str, stored: &str| {
        source.replace("ARGUMENTS", arguments).replace("STORE", &store(stored))
    };
    assert_eq!(demanded(&program("%b, %a", "%f2")), set(&["%p", "%f2", "%gx", "%b"]));
    assert_eq!(demanded(&program("%b, %a", "%f1")), set(&["%p", "%f1", "%gx", "%b"]));
    assert_eq!(demanded(&program("%a, %a", "%f2")), set(&["%p", "%f2", "%gx", "%a"]));
    let unused = source.replace("ARGUMENTS", "%a, %a").replace("STORE", "");
    assert_eq!(demanded(&unused), set(&[]));
}

/// The test analysis with the branch-operand hook of upstream's `LivenessAnalysis`: a branch
/// operand that is not forwarded is given to the transfer function, as the only operand of the
/// operation that decides whether it is demanded (the region branch whose regions it chooses, for
/// a region terminator its parent, and otherwise the branch itself), with that operation's
/// results. So the condition of an `scf.if` is demanded only when the `if` has an effect or one of
/// its results is demanded.
#[derive(Default)]
struct DemandedLikeLiveness;

impl BuildableDataFlowAnalysis for DemandedLikeLiveness {
    type Strategy = SparseDataFlowAnalysis<Self, Backward>;

    fn new(_solver: &mut DataFlowSolver) -> Self {
        Self
    }
}

impl SparseBackwardDataFlowAnalysis for DemandedLikeLiveness {
    type Lattice = Lattice<Demand>;

    fn visit_operation(
        &self,
        op: &Operation,
        operands: &mut OperandLattices<'_, Self::Lattice>,
        results: &[AnalysisStateGuard<'_, Self::Lattice>],
        solver: &mut DataFlowSolver,
    ) -> Result<(), Report> {
        Demanded.visit_operation(op, operands, results, solver)
    }

    fn visit_branch_operand(&self, operand: &OpOperandImpl, solver: &mut DataFlowSolver) {
        let owner = operand.owner;
        let owner = owner.borrow();
        let visited = if owner.implements::<dyn RegionBranchTerminatorOpInterface>() {
            owner.parent_op().expect("a region terminator has a parent")
        } else {
            owner.as_operation_ref()
        };
        let visited = visited.borrow();
        let mut operands = OperandLattices::from_values([operand.as_value_ref()], solver);
        let results = visited
            .results()
            .iter()
            .map(|result| {
                let result = result.borrow().as_value_ref();
                solver.require::<Lattice<Demand>, _>(result, ProgramPoint::after(&*owner))
            })
            .collect::<Vec<_>>();
        self.visit_operation(&visited, &mut operands, &results, solver)
            .expect("the transfer function of the test analysis does not fail");
    }

    fn visit_call_operand(&self, operand: &OpOperandImpl, solver: &mut DataFlowSolver) {
        Demanded::demand(operand, solver);
    }

    fn set_to_exit_state(&self, lattice: &mut AnalysisStateGuardMut<'_, Self::Lattice>) {
        SparseLattice::meet(lattice, &DEMANDED);
    }
}

/// A branch-operand hook may give its operand to the transfer function. The `if`'s result is
/// demanded only after the `if` is first visited (through a branch to a later block), so the hook
/// runs again when it changes, and demands the condition; the control stores another value, and
/// the condition, which only the `if` uses, is not demanded. The condition of the `cf.cond_br` is
/// demanded either way, as the operand of a terminator, which has an effect.
#[test]
fn a_branch_operand_hook_may_give_its_operand_to_the_transfer_function() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %d: i1, %a: i32, %b: i32) {
    %r = scf.if %c then {
        scf.yield %a : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    cf.cond_br %d ^next(%r : i32), ^other : (i1);
^next(%x: i32):
    test.store %p, STORED : (ptr<i32>, i32);
    builtin.ret;
^other:
    builtin.ret;
};"#;
    let program = Program::parse(&source.replace("STORED", "%x"));
    assert_eq!(
        program.demanded_by::<DemandedLikeLiveness>(),
        set(&["%p", "%d", "%x", "%r", "%c", "%a", "%b"])
    );
    let control = Program::parse(&source.replace("STORED", "%b"));
    assert_eq!(control.demanded_by::<DemandedLikeLiveness>(), set(&["%p", "%d", "%b"]));
    // The test analysis itself demands every branch condition.
    assert_eq!(control.demanded(), set(&["%p", "%d", "%b", "%c"]));
}

/// The default transfer function for a call to an external function visits every operand of the
/// call, as upstream's does: the arguments, and an operand that is not one (the index that chooses
/// the callee of `test.call_one_of`).
#[test]
fn the_default_external_call_transfer_visits_every_operand_of_the_call() {
    let program = Program::parse(
        r#"builtin.module public @m {
    builtin.function private extern("C") @g(i32) -> i32;
    builtin.function public extern("C") @f(%i: i32, %a: i32, %b: i32) {
        %r = test.call_one_of %i, @g, @g(%a) : extern("C") (i32) -> i32;
        builtin.ret;
    };
};"#,
    );
    let mut call = None;
    program.root.borrow().prewalk_all(|op: &Operation| {
        if op.implements::<dyn CallOpInterface>() {
            call = Some(op.as_operation_ref());
        }
    });
    let call = call.expect("the program has a call");
    let call = call.borrow();
    let mut solver = DataFlowSolver::default();
    Demanded.visit_external_call(call.as_trait::<dyn CallOpInterface>().unwrap(), &mut solver);
    let demanded = program
        .values()
        .into_iter()
        .filter(|value| {
            solver.get::<Lattice<Demand>, _>(value).is_some_and(|lattice| lattice.value().0)
        })
        .map(|value| program.name_of(value))
        .collect::<BTreeSet<_>>();
    assert_eq!(demanded, set(&["%i", "%a"]));
}

/// A branch that gives its successor more operands than the successor has arguments is invalid IR
/// (the verifier refuses it). Parsed without verification, the framework does not skip the extra
/// operand silently: in a debug build, it stops with a message saying what is missing.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "has no argument for forwarded operand 1")]
fn a_branch_operand_without_a_successor_argument_is_a_debug_assertion() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %a: i32, %b: i32) {
    cf.br ^next(%a, %b : i32, i32);
^next(%x: i32):
    test.store %p, %x : (ptr<i32>, i32);
    builtin.ret;
};"#;
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    context.get_or_register_dialect::<TestDialect>();
    let config = ParserConfig::new(context.clone()).verify_after_parse(false);
    let root = parse::parse_any(config, "sparse_backward", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}\n{source}", describe(&err)));
    let mut solver = DataFlowSolver::new(DataFlowConfig::default());
    solver.load::<DeadCodeAnalysis>();
    solver.load::<Demanded>();
    let _ = solver.initialize_and_run(&root.borrow(), AnalysisManager::new(root, None));
}

/// Operand lattices built from values by an analysis have one position per value and one guard
/// per distinct value: [OperandLattices::distinct] lends each guard once, in the order of first
/// use, and what is written through it is read at every position of its value.
#[test]
fn operand_lattices_built_from_values_hold_one_guard_per_distinct_value() {
    let program = Program::parse(
        r#"builtin.function public extern("C") @f(%a: i32, %b: i32) {
    builtin.ret;
};"#,
    );
    let values = program.values();
    let (a, b) = (values[0], values[1]);
    let mut solver = DataFlowSolver::default();
    let mut operands = OperandLattices::<Lattice<Demand>>::from_values([a, b, a], &mut solver);
    assert_eq!(operands.len(), 3);
    {
        let mut distinct = operands.distinct();
        assert_eq!(distinct.len(), 2);
        let first = distinct.next().expect("the lattice of %a");
        assert_eq!(first.anchor().as_value(), Some(a));
        SparseLattice::meet(first, &DEMANDED);
        let second = distinct.next().expect("the lattice of %b");
        assert_eq!(second.anchor().as_value(), Some(b));
        assert!(distinct.next().is_none());
    }
    let demanded = (0..operands.len()).map(|at| operands[at].value().0).collect::<Vec<_>>();
    assert_eq!(demanded, [true, false, true]);
}

/// Two functions that call each other, `@a` passing its arguments to `@b` (swapped, and in the
/// same order) and `@b` passing them back to `@a`; `@a` returns `@b`'s result or its own `%ax`,
/// and its caller, placed before both, stores what it returns. Swapped, the demand of `%ax`
/// reaches `%by` through the call in `@a`, then `%ay` through the call in `@b`, and every argument
/// is demanded; in the same order, only the first arguments are.
#[test]
fn demand_flows_through_mutually_recursive_calls() {
    let source = r#"builtin.module public @m {
    builtin.function public extern("C") @f(%p: ptr<i32>, %fc: i1, %p1: i32, %p2: i32) {
        %r = builtin.call @a(%fc, %p1, %p2) : extern("C") (i1, i32, i32) -> i32;
        test.store %p, %r : (ptr<i32>, i32);
        builtin.ret;
    };
    builtin.function private extern("C") @a(%ac: i1, %ax: i32, %ay: i32) -> i32 {
        cf.cond_br %ac ^rec, ^base : (i1);
    ^rec:
        %ar = builtin.call @b(%ac, PASSED) : extern("C") (i1, i32, i32) -> i32;
        builtin.ret %ar : (i32);
    ^base:
        builtin.ret %ax : (i32);
    };
    builtin.function private extern("C") @b(%bc: i1, %bx: i32, %by: i32) -> i32 {
        %br = builtin.call @a(%bc, %bx, %by) : extern("C") (i1, i32, i32) -> i32;
        builtin.ret %br : (i32);
    };
};"#;
    let swapped = demanded(&source.replace("PASSED", "%ay, %ax"));
    assert_eq!(
        swapped,
        set(&[
            "%p", "%r", "%fc", "%ac", "%bc", "%ar", "%br", "%ax", "%ay", "%bx", "%by", "%p1", "%p2"
        ])
    );
    let same = demanded(&source.replace("PASSED", "%ax, %ay"));
    assert_eq!(same, set(&["%p", "%r", "%fc", "%ac", "%bc", "%ar", "%br", "%ax", "%bx", "%p1"]));
}

/// A conditional branch whose two successors are one block, with a different first operand on
/// each edge and the same second one: the first argument demands both first operands, and the
/// second argument demands only the second operand.
#[test]
fn a_conditional_branch_may_have_one_block_as_both_successors() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32, %b: i32, %d: i32) {
    cf.cond_br %c ^j(%a, %d : i32, i32), ^j(%b, %d : i32, i32) : (i1);
^j(%x: i32, %y: i32):
    test.store %p, STORED : (ptr<i32>, i32);
    builtin.ret;
};"#;
    assert_eq!(demanded(&source.replace("STORED", "%x")), set(&["%p", "%c", "%x", "%a", "%b"]));
    assert_eq!(demanded(&source.replace("STORED", "%y")), set(&["%p", "%c", "%y", "%d"]));
}

/// A loop whose back edge passes the header's first argument to itself and to the second
/// argument, `^head(%x, %x)` into `^head(%x, %y)`, and whose exit block demands `%y`, after the
/// back edge is first visited: the demand of `%y` reaches `%x` through the back edge, and both
/// operands of the entry edge.
#[test]
fn a_back_edge_may_pass_one_argument_to_itself_and_to_another() {
    let source = r#"builtin.function public extern("C") @f(%p: ptr<i32>, %c: i1, %a: i32, %b: i32) {
    cf.br ^head(%a, %b : i32, i32);
^head(%x: i32, %y: i32):
    cf.cond_br %c ^head(%x, %x : i32, i32), ^exit : (i1);
^exit:
    test.store %p, %y : (ptr<i32>, i32);
    builtin.ret;
};"#;
    assert_eq!(demanded(source), set(&["%p", "%c", "%x", "%y", "%a", "%b"]));
}
