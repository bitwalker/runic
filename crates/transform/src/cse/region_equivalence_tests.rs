use alloc::{boxed::Box, rc::Rc};

use runic_core::{
    Context, Report,
    dialects::builtin::{Function, FunctionRef},
    parse::{self, ParserConfig},
    pass::{Nesting, PassManager},
};
use runic_dialect_scf::If;

use super::CommonSubexpressionElimination;

fn parse_function(context: &Rc<Context>, source: &str) -> Result<FunctionRef, Report> {
    parse::parse::<Function>(ParserConfig::new(context.clone()), "regions.hir", source)
        .into_result()
}

#[test]
fn cse_reconciles_inline_locations_in_merged_regions() -> Result<(), Report> {
    use runic_core::{
        Forward, OperationRef, RawWalk,
        dialects::debuginfo::attributes::INLINE_CALL_CHAIN_ATTR_NAME,
    };

    for second_line in [10, 20] {
        let context = Rc::new(Context::default());
        let function = parse_function(
            &context,
            r#"
builtin.function public extern("C") @regions(%c: i1, %a: i32, %b: i32) -> (i32, i32) {
    %x = scf.if %c then {
        %sum = arith.add %a, %b <{ overflow = #builtin.overflow<unchecked unsigned> }>;
        scf.yield %sum : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    %y = scf.if %c then {
        %sum2 = arith.add %a, %b <{ overflow = #builtin.overflow<unchecked unsigned> }>;
        scf.yield %sum2 : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %x, %y : (i32, i32);
};
"#,
        )?;
        let regions = function
            .borrow()
            .body()
            .entry()
            .body()
            .iter()
            .filter(|op| op.is::<If>())
            .map(|op| op.as_operation_ref())
            .collect::<alloc::vec::Vec<_>>();
        let chain = super::tests::inline_chain(&context, 10);
        for (mut root, line) in regions.iter().copied().zip([10, second_line]) {
            let chain = super::tests::inline_chain(&context, line);
            root.raw_prewalk_all::<Forward, _>(|mut op: OperationRef| {
                op.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, chain);
            });
            root.borrow_mut().remove_attribute(INLINE_CALL_CHAIN_ATTR_NAME);
        }
        let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
        passes.add_pass(Box::<CommonSubexpressionElimination>::default());
        passes.run(function.as_operation_ref())?;
        function.as_operation_ref().borrow().recursively_verify()?;
        let function = function.borrow();
        let body = function.body();
        let entry = body.entry();
        assert_eq!(entry.body().iter().filter(|op| op.is::<If>()).count(), 1);
        regions[0].raw_prewalk_all::<Forward, _>(|op: OperationRef| {
            assert_eq!(
                op.borrow().get_attribute(INLINE_CALL_CHAIN_ATTR_NAME),
                (second_line == 10 && !OperationRef::ptr_eq(&op, &regions[0])).then_some(chain),
            );
        });
    }
    Ok(())
}

#[test]
fn cse_eliminates_equivalent_regions() -> Result<(), Report> {
    let context = Rc::new(Context::default());
    let function = parse_function(
        &context,
        r#"
builtin.function public extern("C") @regions(%c: i1, %a: i32, %b: i32) -> (i32, i32) {
    %x = scf.if %c then {
        %sum = arith.add %a, %b <{ overflow = #builtin.overflow<unchecked unsigned> }>;
        scf.yield %sum : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    %y = scf.if %c then {
        %sum2 = arith.add %b, %a <{ overflow = #builtin.overflow<unchecked unsigned> }>;
        scf.yield %sum2 : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %x, %y : (i32, i32);
};
"#,
    )?;
    function.as_operation_ref().borrow().recursively_verify()?;
    let mut passes = PassManager::on::<Function>(context, Nesting::Implicit);
    passes.add_pass(Box::<CommonSubexpressionElimination>::default());
    passes.run(function.as_operation_ref())?;
    function.as_operation_ref().borrow().recursively_verify()?;
    let function = function.borrow();
    let body = function.body();
    let entry = body.entry();
    assert_eq!(entry.body().iter().filter(|op| op.is::<If>()).count(), 1);
    let ret = entry.terminator().unwrap();
    let ret = ret.borrow();
    let operands = ret.operands().all();
    assert_eq!(operands[0].borrow().as_value_ref(), operands[1].borrow().as_value_ref());
    Ok(())
}

#[test]
fn region_equivalence_maps_nested_values_and_preserves_captures() -> Result<(), Report> {
    use runic_core::equivalence::OperationEquivalenceFlags;

    let context = Rc::new(Context::default());
    let source = r#"
builtin.function public extern("C") @nested(%c: i1, %a: i32, %b: i32) -> i32 {
    %first = arith.add %a, %b <{ overflow = #builtin.overflow<unchecked unsigned> }>;
    %second = arith.add %a, %a <{ overflow = #builtin.overflow<unchecked unsigned> }>;
    %x = scf.if %c then {
        %sum = arith.add %first, %second <{ overflow = #builtin.overflow<unchecked unsigned> }>;
        scf.yield %sum : (i32);
    } else {
        scf.yield %b : (i32);
    } : (i1) -> (i32);
    builtin.ret %x : (i32);
};
"#;
    let lhs = parse_function(&context, source)?;
    let equivalent = source
        .replace("%first, %second", "%second, %first")
        .replace("%first", "%renamed");
    let rhs = parse_function(&context, &equivalent)?;
    let different_capture =
        parse_function(&context, &source.replace("scf.yield %b", "scf.yield %a"))?;
    let different_body = parse_function(&context, &source.replace("%a, %a", "%b, %b"))?;
    let lhs = lhs.as_operation_ref();
    let flags = OperationEquivalenceFlags::IGNORE_LOCATIONS;
    assert!(lhs.borrow().is_equivalent(&rhs.as_operation_ref().borrow(), flags));
    assert!(
        !lhs.borrow()
            .is_equivalent(&different_capture.as_operation_ref().borrow(), flags)
    );
    assert!(!lhs.borrow().is_equivalent(&different_body.as_operation_ref().borrow(), flags));
    Ok(())
}

#[test]
fn region_equivalence_preserves_branch_targets_and_case_keys() -> Result<(), Report> {
    use runic_core::equivalence::OperationEquivalenceFlags;

    let context = Rc::new(Context::default());
    let source = r#"
builtin.function public extern("C") @select(%sel: i32, %a: i32, %b: i32) -> i32 {
    cf.switch %sel [#builtin.i32<1> -> ^one(%a : i32)], ^fallback(%b : i32) : (i32);
^one(%x: i32):
    builtin.ret %x : (i32);
^fallback(%y: i32):
    builtin.ret %y : (i32);
};
"#;
    let lhs = parse_function(&context, source)?;
    let rhs =
        parse_function(&context, &source.replace("^one", "^renamed").replace("%x", "%renamed"))?;
    let different_key =
        parse_function(&context, &source.replace("#builtin.i32<1>", "#builtin.i32<2>"))?;
    let different_target =
        parse_function(&context, &source.replace("^one(%a : i32)", "^fallback(%a : i32)"))?;
    let flags = OperationEquivalenceFlags::IGNORE_LOCATIONS;
    let lhs = lhs.as_operation_ref();
    assert!(lhs.borrow().is_equivalent(&rhs.as_operation_ref().borrow(), flags));
    assert!(!lhs.borrow().is_equivalent(&different_key.as_operation_ref().borrow(), flags));
    assert!(!lhs.borrow().is_equivalent(&different_target.as_operation_ref().borrow(), flags));
    Ok(())
}

#[test]
fn region_equivalence_handles_graph_forward_references() {
    use runic_core::{
        Builder, Ident, Signedness, SourceSpan, dialects::builtin::BuiltinOpBuilder,
        equivalence::OperationEquivalenceFlags, testing::Test,
    };
    use runic_dialect_arith::ArithOpBuilder;

    // Both modules are built in one context: types and attributes are compared by their handles,
    // which are unique only within a context.
    let mut test = Test::named("forward_refs");
    let mut modules = alloc::vec::Vec::new();
    for reverse_allocation in [false, true] {
        let builder = test.builder_mut();
        builder.clear_insertion_point();
        let module = builder.create_module(Ident::with_empty_span("forward_refs".into())).unwrap();
        let body = module.borrow().body().as_region_ref();
        let block = builder.create_block(body, None, &[]);
        builder.set_insertion_point_to_end(block);
        // Opposite allocation order ensures address sorting cannot substitute for the SSA map.
        let (seven, nine) = if reverse_allocation {
            let nine = builder.i32(9, SourceSpan::UNKNOWN);
            (builder.i32(7, SourceSpan::UNKNOWN), nine)
        } else {
            let seven = builder.i32(7, SourceSpan::UNKNOWN);
            (seven, builder.i32(9, SourceSpan::UNKNOWN))
        };
        let sum = builder
            .add_unchecked(seven, nine, Signedness::Unsigned, SourceSpan::UNKNOWN)
            .unwrap();
        for value in [sum, seven, nine] {
            let mut op = value.borrow().get_defining_op().unwrap();
            op.borrow_mut().remove();
            op.insert_at_end(block);
        }
        modules.push(module);
    }
    assert!(modules[0].as_operation_ref().borrow().is_equivalent(
        &modules[1].as_operation_ref().borrow(),
        OperationEquivalenceFlags::IGNORE_LOCATIONS,
    ));
}
