//! The analyses of this crate over the corpus of `runic_ir::testing::calls`, and over IR that
//! holds symbol references that do not resolve, parsed with verification off.

use alloc::{format, rc::Rc, string::String, vec::Vec};

use runic_ir::{
    Context, Operation, OperationRef, Symbol,
    dialects::builtin::Function,
    parse::{self, ParserConfig},
    pass::AnalysisManager,
    testing::{
        calls::{CORPUS, mutations},
        text::describe,
    },
};

mod tail_calls;

use crate::{
    DataFlowConfig, DataFlowSolver,
    analyses::{
        CallableUseSnapshot, DeadCodeAnalysis, IntegerRangeAnalysis, SparseConstantPropagation,
        liveness::Liveness,
    },
};

fn parse_any(source: &str, verify: bool) -> Result<(Rc<Context>, OperationRef), String> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<runic_dialect_arith::ArithDialect>();
    context.get_or_register_dialect::<runic_dialect_cf::ControlFlowDialect>();
    context.get_or_register_dialect::<runic_dialect_scf::ScfDialect>();
    let config = ParserConfig::new(context.clone()).verify_after_parse(verify);
    match parse::parse_any(config, "corpus", source).into_result() {
        Ok(op) => Ok((context, op)),
        Err(report) => Err(describe(&report)),
    }
}

/// Every analysis of this crate over `root`. Returns the names of those that returned an error.
fn analyze(root: OperationRef) -> Vec<String> {
    let mut errors = Vec::new();
    let top = root.parent_op().unwrap_or(root);

    // callable uses, over the root and over each nested operation with regions
    let snapshot = CallableUseSnapshot::new(&top.borrow());
    for (_, info) in snapshot.iter() {
        let _ = (info.known_callers().len(), info.is_address_taken(), info.has_unknown_callers());
    }
    let mut functions = Vec::new();
    top.borrow().prewalk_all(|op: &Operation| {
        if op.is::<Function>() {
            functions.push(op.as_operation_ref());
        }
        if op.has_regions() {
            let _ = CallableUseSnapshot::new(op);
        }
    });

    // dead code + constants + ranges, interprocedural and not
    for interprocedural in [true, false] {
        let mut config = DataFlowConfig::default();
        config.set_interprocedural(interprocedural);
        let mut solver = DataFlowSolver::new(config);
        solver.load::<DeadCodeAnalysis>();
        solver.load::<SparseConstantPropagation>();
        if let Err(err) = solver.initialize_and_run(&top.borrow(), AnalysisManager::new(top, None))
        {
            errors.push(format!("dce+sccp ({interprocedural}): {}", describe(&err)));
        }
        solver.load::<IntegerRangeAnalysis>();
        if let Err(err) = solver.initialize_and_run(&top.borrow(), AnalysisManager::new(top, None))
        {
            errors.push(format!("ranges ({interprocedural}): {}", describe(&err)));
        }
    }

    // liveness, per function (it is not interprocedural)
    for function in functions {
        if function.borrow().downcast_ref::<Function>().unwrap().is_declaration() {
            continue;
        }
        let mut solver = DataFlowSolver::new(DataFlowConfig::default());
        solver.load::<DeadCodeAnalysis>();
        solver.load::<SparseConstantPropagation>();
        solver.load::<Liveness>();
        if let Err(err) =
            solver.initialize_and_run(&function.borrow(), AnalysisManager::new(function, None))
        {
            errors.push(format!("liveness: {}", describe(&err)));
        }
    }
    errors
}

#[test]
fn the_analyses_run_over_the_corpus() {
    for source in CORPUS {
        let (_context, root) =
            parse_any(source, true).unwrap_or_else(|err| panic!("{err}\n{source}"));
        let errors = analyze(root);
        assert!(errors.is_empty(), "{errors:#?}\n{source}");
    }
}

/// Each source of the corpus with one declaration deleted or renamed, or one reference renamed,
/// parsed with verification off: no analysis panics.
#[test]
fn fuzz_no_analysis_panics_on_a_dangling_symbol() {
    let mut cases = 0;
    let mut problems = Vec::new();
    let mut errors = alloc::collections::BTreeMap::<String, usize>::new();
    for source in CORPUS {
        for (what, text, _fails) in mutations(source) {
            cases += 1;
            let outcome = std::panic::catch_unwind(|| {
                let (_context, root) = parse_any(&text, false)?;
                Ok::<_, String>(analyze(root))
            });
            match outcome {
                Err(_) => problems.push(format!("PANIC {what}:\n{text}")),
                Ok(Err(err)) => problems.push(format!("PARSE FAILED {what}: {err}")),
                Ok(Ok(failed)) => {
                    for error in failed {
                        *errors.entry(error).or_default() += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 60, "{cases}");
    assert!(
        problems.is_empty(),
        "{} problems of {cases} cases:\n{}",
        problems.len(),
        problems.join("\n\n")
    );
    std::eprintln!("{cases} cases; analysis errors: {errors:#?}");
}

/// Constant propagation folds an operation with the constants its operands are known to hold, not
/// only with the constants its operands are defined by: the product of a sum of two constants is a
/// constant too.
#[test]
fn constants_propagate_through_operations_whose_operands_are_not_constants() {
    use runic_ir::ValueRef;

    use crate::{Lattice, analyses::constant_propagation::ConstantValue};

    let source = "builtin.function public extern(\"C\") @f() -> i32 {
    %two = arith.constant 2 : i32;
    %five = arith.constant 5 : i32;
    %sum = arith.add %two, %five;
    %product = arith.mul %sum, %two;
    builtin.ret %product : (i32);
};";
    let (_context, function) = parse_any(source, true).unwrap();
    let mut solver = DataFlowSolver::default();
    solver.load::<DeadCodeAnalysis>();
    solver.load::<SparseConstantPropagation>();
    solver
        .initialize_and_run(&function.borrow(), AnalysisManager::new(function, None))
        .unwrap();
    let function = function.borrow();
    let block = function.regions().front().get().unwrap().entry().as_block_ref();
    let ret = block.borrow().terminator().unwrap();
    let product: ValueRef = ret.borrow().operands()[0].borrow().as_value_ref();
    let lattice = solver.get::<Lattice<ConstantValue>, _>(&product).expect("a lattice");
    let value = lattice.value().constant_value().expect("a constant");
    assert_eq!(value.borrow().as_int().map(|int| int.unsigned()), Some(14));
}
