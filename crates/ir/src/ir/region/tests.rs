use alloc::rc::Rc;

use crate::{
    Context, Region, SymbolTable,
    diagnostics::Report,
    dialects::builtin::{Function, Module},
    parse::{self, ParserConfig},
};

type TestResult = Result<(), Report>;

/// `Region::find_common_ancestor` returns the innermost region containing all operations.
///
/// Regression coverage: the previous implementation returned `None` for every multi-operation
/// query, and did not terminate when an operation lay outside the first operation's innermost
/// region.
#[test]
fn find_common_ancestor_returns_innermost_common_region() -> TestResult {
    let context = Rc::new(Context::default());
    // The parser anchors parsed operations in a fresh `builtin.world`, so this source produces
    // the region hierarchy: world body > module body > function bodies.
    let source = r#"builtin.module public @test {
    builtin.function public extern("C") @a(%x: i32) -> i64 {
        %v = builtin.unrealized_conversion_cast %x : i32 to i64;
        builtin.ret %v : (i64);
    };
    builtin.function public extern("C") @b() -> i32 {
        builtin.ret_imm 42 : i32;
    };
};"#;
    let config = ParserConfig::new(context.clone());
    let module_op = parse::parse_any(config, "find_common_ancestor.runic", source).into_result()?;

    let (func_a_op, func_b_op) = {
        let module_op = module_op.borrow();
        let module = module_op.downcast_ref::<Module>().expect("expected a module");
        let symbols = module.symbol_manager();
        (
            symbols.lookup_op("a").expect("'a' was not registered in the symbol table"),
            symbols.lookup_op("b").expect("'b' was not registered in the symbol table"),
        )
    };
    let (cast_a, ret_a) = {
        let function =
            func_a_op.try_downcast_op::<Function>().expect("expected '@a' to be a function");
        let function = function.borrow();
        let cast = function.body().entry().front().unwrap();
        let ret = function.body().entry().terminator().unwrap();
        (cast, ret)
    };
    let ret_b = {
        let function =
            func_b_op.try_downcast_op::<Function>().expect("expected '@b' to be a function");
        let function = function.borrow();
        function.body().entry().terminator().unwrap()
    };

    let body_a = cast_a.borrow().parent_region().expect("op must be in a function body");
    let module_region = func_a_op.borrow().parent_region().expect("function must be in a module");
    let world_region = module_op.borrow().parent_region().expect("module must be in a world");
    let world_op = world_region.parent().expect("world region must have an owner");

    assert_eq!(Region::find_common_ancestor(&[]), None);
    assert_eq!(Region::find_common_ancestor(&[cast_a]), Some(body_a));
    assert_eq!(
        Region::find_common_ancestor(&[cast_a, ret_a]),
        Some(body_a),
        "operations in one region share that region"
    );
    assert_eq!(
        Region::find_common_ancestor(&[cast_a, ret_b]),
        Some(module_region),
        "operations in sibling function bodies share the module body region"
    );
    assert_eq!(
        Region::find_common_ancestor(&[cast_a, func_b_op]),
        Some(module_region),
        "nested and outer operations share the outer region"
    );
    assert_eq!(
        Region::find_common_ancestor(&[func_b_op, cast_a]),
        Some(module_region),
        "the result is independent of operation order"
    );
    assert_eq!(
        Region::find_common_ancestor(&[cast_a, ret_a, ret_b]),
        Some(module_region),
        "a region containing only some of the operations is not a common ancestor"
    );
    assert_eq!(
        Region::find_common_ancestor(&[cast_a, module_op]),
        Some(world_region),
        "the candidate walk crosses multiple nesting levels"
    );
    assert_eq!(
        Region::find_common_ancestor(&[world_op, cast_a]),
        None,
        "a top-level operation has no enclosing region, so there is no common ancestor"
    );
    assert_eq!(
        Region::find_common_ancestor(&[cast_a, world_op]),
        None,
        "a top-level operation has no enclosing region, so there is no common ancestor"
    );
    Ok(())
}

mod defined_above {
    //! The helpers that find, replace and capture the values a region uses from above it.

    use alloc::{rc::Rc, string::ToString, vec::Vec};

    use crate::{
        BlockRef, Context, OpOperand, Operation, OperationRef, Region, RegionRef, Report, ValueRef,
        derive::{EffectOpInterface, operation},
        dialects::{
            builtin::{Function, Module},
            test::TestDialect,
        },
        effects::MemoryEffectOpInterface,
        parse::{self, ParserConfig},
        patterns::{NoopRewriterListener, RewriterImpl},
        traits::{AnyType, BranchOpInterface, Terminator},
    };

    type TestResult = Result<(), Report>;

    /// An operation with operands and one region, neither isolated from above nor constraining
    /// the arguments of its region: `"test.ru_holder"(%a) ({ .. }) : i32 -> ();`.
    #[derive(EffectOpInterface)]
    #[operation(dialect = TestDialect, implements(MemoryEffectOpInterface))]
    pub struct RuHolder {
        #[operands]
        inputs: AnyType,
        #[region]
        body: Region,
    }

    /// Ends a block of a [RuHolder] region: `"test.ru_yield"(%a) : i32 -> ();`.
    #[derive(EffectOpInterface)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(MemoryEffectOpInterface)
    )]
    pub struct RuYield {
        #[operands]
        values: AnyType,
    }

    /// A branch that forwards its operands to its one successor:
    /// `"test.ru_br"() [ ^next(%a : i32) ] : () -> ();`.
    #[derive(EffectOpInterface)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(BranchOpInterface, MemoryEffectOpInterface)
    )]
    pub struct RuBr {
        #[successor]
        target: Successor,
    }

    impl BranchOpInterface for RuBr {}

    /// `@f(%a, %b)` defines `%c` and `%s` and holds an outer `test.ru_holder` whose region
    /// defines `%t` and `%k` and holds an inner one, which uses `%t`, `%a` and `%s`.
    const NESTED: &str = r#"builtin.function public extern("C") @f(%a: i32, %b: i32) -> i32 {
    %c = test.constant 1 : i32;
    %s = test.add %a, %c;
    "test.ru_holder"(%b) ({
    ^entry(%x: i32):
        %t = test.add %s, %x;
        %k = test.add %t, %c;
        "test.ru_holder"() ({
            %u = test.add %t, %a;
            %v = test.add %u, %s;
            "test.ru_yield"(%v) : i32 -> ();
        }) : () -> ();
        "test.ru_yield"(%a, %k) : (i32, i32) -> ();
    }) : i32 -> ();
    builtin.ret %s : (i32);
};"#;

    /// The parsed function, and the values and operations of [NESTED] by name.
    struct Nested {
        context: Rc<Context>,
        function: OperationRef,
        a: ValueRef,
        b: ValueRef,
        c: ValueRef,
        s: ValueRef,
        t: ValueRef,
        outer: RegionRef,
        inner: RegionRef,
        /// The operations, in pre-order: `%c`, `%s`, the outer holder, `%t`, `%k`, the inner
        /// holder, `%u`, `%v`, the inner yield, the outer yield, the return.
        ops: Vec<OperationRef>,
    }

    impl Nested {
        fn parse(source: &str) -> Result<Self, Report> {
            let (context, function) = parse_function(source)?;
            let mut ops = Vec::new();
            function.borrow().prewalk_all(|op: &Operation| ops.push(op.as_operation_ref()));
            // The walk starts with the function itself.
            ops.remove(0);
            let [a, b] = function_arguments(function)[..] else {
                panic!("the function takes two arguments");
            };
            let result = |index: usize| ops[index].borrow().results()[0] as ValueRef;
            let region = |index: usize| ops[index].borrow().regions().front().as_pointer().unwrap();
            Ok(Self {
                c: result(0),
                s: result(1),
                t: result(3),
                outer: region(2),
                inner: region(5),
                context,
                function,
                a,
                b,
                ops,
            })
        }

        fn rewriter(&self) -> RewriterImpl<NoopRewriterListener> {
            RewriterImpl::new(self.context.clone())
        }

        /// The values of the operands of `ops[index]`.
        fn operands(&self, index: usize) -> Vec<ValueRef> {
            operand_values(self.ops[index])
        }

        fn verify(&self) -> TestResult {
            self.function.borrow().recursively_verify()
        }
    }

    fn parse_function(source: &str) -> Result<(Rc<Context>, OperationRef), Report> {
        let context = Rc::new(Context::default());
        let function = parse::parse::<Function>(ParserConfig::new(context.clone()), "f", source)
            .into_result()?
            .as_operation_ref();
        Ok((context, function))
    }

    /// The regions of the `test.ru_holder` operations under `op`, in pre-order.
    fn holder_regions(op: OperationRef) -> Vec<RegionRef> {
        let mut regions = Vec::new();
        op.borrow().prewalk_all(|op: &Operation| {
            if op.name().to_string() == "test.ru_holder" {
                regions.push(op.regions().front().as_pointer().unwrap());
            }
        });
        regions
    }

    fn function_arguments(function: OperationRef) -> Vec<ValueRef> {
        entry_arguments(function.borrow().regions().front().as_pointer().unwrap())
    }

    fn operand_values(op: OperationRef) -> Vec<ValueRef> {
        op.borrow()
            .operands()
            .iter()
            .map(|operand| operand.borrow().as_value_ref())
            .collect()
    }

    fn entry_arguments(region: RegionRef) -> Vec<ValueRef> {
        let entry = region.borrow().entry_block_ref().unwrap();
        entry.borrow().arguments().iter().map(|arg| *arg as ValueRef).collect()
    }

    fn block_ops(block: BlockRef) -> Vec<OperationRef> {
        block.borrow().body().iter().map(|op| op.as_operation_ref()).collect()
    }

    #[test]
    fn values_defined_above_a_region_are_found_in_post_order_once_each() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        let outer = ir.outer.borrow();
        let inner = ir.inner.borrow();

        // The inner region is walked before the outer region's operations after it.
        assert_eq!(outer.get_used_values_defined_above(&ir.outer).as_slice(), [ir.s, ir.c, ir.a]);
        assert_eq!(inner.get_used_values_defined_above(&ir.inner).as_slice(), [ir.t, ir.a, ir.s]);
        // With the outer region as the limit, what the outer region defines is not above.
        assert_eq!(inner.get_used_values_defined_above(&ir.outer).as_slice(), [ir.a, ir.s]);
        Ok(())
    }

    #[test]
    fn values_defined_above_several_regions_are_found_once_across_them() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        let values = Region::get_used_values_defined_above_any(&[ir.inner, ir.outer]);
        assert_eq!(values.as_slice(), [ir.t, ir.a, ir.s, ir.c]);
        Ok(())
    }

    #[test]
    fn each_use_of_a_value_defined_above_is_visited() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        let record = |uses: &mut Vec<(OperationRef, ValueRef)>, operand: OpOperand| {
            let operand = operand.borrow();
            uses.push((operand.owner, operand.as_value_ref()));
        };
        let mut uses = Vec::new();
        ir.inner
            .borrow()
            .visit_used_values_defined_above(&ir.outer, |operand| record(&mut uses, operand));
        // `%u` uses `%a`, `%v` uses `%s`; `%u`'s use of `%t` is inside the limit.
        assert_eq!(uses, [(ir.ops[6], ir.a), (ir.ops[7], ir.s)]);

        uses.clear();
        Region::visit_used_values_defined_above_any(&[ir.outer], |operand| {
            record(&mut uses, operand)
        });
        // Every use, the repeated ones included: `%t`, `%k`, `%u`, `%v`, then the outer yield.
        let expected = [
            (ir.ops[3], ir.s),
            (ir.ops[4], ir.c),
            (ir.ops[6], ir.a),
            (ir.ops[7], ir.s),
            (ir.ops[9], ir.a),
        ];
        assert_eq!(uses, expected);
        Ok(())
    }

    #[test]
    fn the_callback_may_change_the_operands_it_visits() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        let b = ir.b;
        ir.outer.borrow().visit_used_values_defined_above(&ir.outer, |mut operand| {
            if operand.borrow().as_value_ref() == ir.a {
                operand.borrow_mut().set(b);
            }
        });
        assert_eq!(ir.operands(6), [ir.t, ir.b]);
        assert_eq!(ir.operands(9), [ir.b, ir.ops[4].borrow().results()[0] as ValueRef]);
        // The use of `%a` above the region is unchanged.
        assert_eq!(ir.operands(1), [ir.a, ir.c]);
        ir.verify()
    }

    #[test]
    fn the_callback_over_several_regions_may_borrow_them_mutably() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        let mut visited = 0;
        Region::visit_used_values_defined_above_any(&[ir.inner, ir.outer], |_| {
            let mut inner = ir.inner;
            let mut outer = ir.outer;
            drop((inner.borrow_mut(), outer.borrow_mut()));
            visited += 1;
        });
        assert_eq!(visited, 8);
        Ok(())
    }

    #[test]
    fn an_empty_region_is_isolated_already() -> TestResult {
        let (context, function) = parse_function(
            r#"builtin.function public extern("C") @f(%a: i32) {
    "test.ru_holder"(%a) ({ }) : i32 -> ();
    builtin.ret;
};"#,
        )?;
        let region = holder_regions(function)[0];
        let mut rewriter = RewriterImpl::<NoopRewriterListener>::new(context);
        let captured = Region::make_isolated_from_above(region, &mut rewriter, |_| true);
        assert!(captured.is_empty());
        assert!(region.borrow().is_empty());
        function.borrow().recursively_verify()
    }

    /// `%a` and `%b`, in the graph region of a module, read each other: only one of them can be
    /// cloned after what it reads. `%a`, placed first without its operand, is captured instead,
    /// and the clone of `%b` reads the argument that replaces it.
    #[test]
    fn operations_that_read_each_other_are_not_all_cloned() -> TestResult {
        let source = r#"builtin.module public @m {
    %a = builtin.unrealized_conversion_cast %b : i32 to i32;
    %b = builtin.unrealized_conversion_cast %a : i32 to i32;
    "test.ru_holder"() ({
        "test.ru_yield"(%a, %b) : (i32, i32) -> ();
    }) : () -> ();
};"#;
        let context = Rc::new(Context::default());
        let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "m", source)
            .into_result()?;
        let module = module.as_operation_ref();
        let region = holder_regions(module)[0];
        let [a, b] = Region::get_used_values_defined_above_any(&[region])[..] else {
            panic!("the region uses `%a` and `%b`");
        };
        let cast_b = b.borrow().get_defining_op().unwrap();
        let mut rewriter = RewriterImpl::<NoopRewriterListener>::new(context);
        let captured = Region::make_isolated_from_above(region, &mut rewriter, |_| true);
        assert_eq!(captured.as_slice(), [a]);

        let entry = region.borrow().entry_block_ref().unwrap();
        let ops = block_ops(entry);
        assert_eq!(ops.len(), 2);
        assert_ne!(ops[0], cast_b);
        let arg = entry_arguments(region)[0];
        let clone = ops[0].borrow().results()[0] as ValueRef;
        assert_eq!(operand_values(ops[0]), [arg]);
        assert_eq!(operand_values(ops[1]), [arg, clone]);
        module.borrow().recursively_verify()
    }

    #[test]
    #[should_panic = "expected the isolation limit to be an ancestor of the region"]
    fn the_limit_must_be_an_ancestor_of_the_region() {
        let ir = Nested::parse(NESTED).unwrap();
        let _ = ir.outer.borrow().get_used_values_defined_above(&ir.inner);
    }

    #[test]
    fn uses_are_replaced_only_within_the_region() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        ir.outer.borrow().replace_all_uses_in_region_with(ir.s, ir.b);
        // `%t` and `%v`, in the outer region and in the inner one, use `%b` now.
        assert_eq!(ir.operands(3), [ir.b, entry_arguments(ir.outer)[0]]);
        assert_eq!(ir.operands(7)[1], ir.b);
        // The return, outside the region, still uses `%s`.
        assert_eq!(ir.operands(10), [ir.s]);

        ir.inner.borrow().replace_all_uses_in_region_with(ir.a, ir.b);
        assert_eq!(ir.operands(6), [ir.t, ir.b]);
        // The outer yield is not in the inner region.
        assert_eq!(ir.operands(9)[0], ir.a);
        ir.verify()
    }

    #[test]
    fn a_region_is_isolated_by_capturing_what_it_uses() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        let x = entry_arguments(ir.outer)[0];
        let captured = Region::make_isolated_from_above(ir.outer, &mut ir.rewriter(), |_| false);
        assert_eq!(captured.as_slice(), [ir.s, ir.c, ir.a]);

        let arguments = entry_arguments(ir.outer);
        assert_eq!(arguments.len(), 4);
        let [x2, s, c, a] = arguments[..] else {
            unreachable!()
        };
        assert_eq!(ir.ops[3].borrow().parent(), ir.outer.borrow().entry_block_ref());
        assert_eq!(ir.operands(3), [s, x2]);
        assert_eq!(ir.operands(4)[1], c);
        assert_eq!(ir.operands(6)[1], a);
        assert_eq!(ir.operands(7)[1], s);
        assert_eq!(ir.operands(9)[0], a);
        assert!(Region::get_used_values_defined_above_any(&[ir.outer]).is_empty());
        // The argument the region had is replaced with the first argument of the new entry
        // block, and the uses above the region are unchanged.
        assert!(!x.borrow().is_used());
        assert_eq!(ir.operands(1), [ir.a, ir.c]);
        assert_eq!(ir.operands(10), [ir.s]);
        ir.verify()
    }

    #[test]
    fn operations_with_no_operands_can_be_cloned_into_the_region() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        let captured =
            Region::make_isolated_from_above(ir.outer, &mut ir.rewriter(), |op| !op.has_operands());
        // `%c` is cloned; `%s` and `%a` are captured.
        assert_eq!(captured.as_slice(), [ir.s, ir.a]);
        let [_, s, a] = entry_arguments(ir.outer)[..] else {
            unreachable!()
        };

        let entry = ir.outer.borrow().entry_block_ref().unwrap();
        let clone = entry.borrow().front().unwrap();
        assert_eq!(clone.name().to_string(), "test.constant");
        assert_ne!(clone, ir.ops[0]);
        let c = clone.borrow().results()[0] as ValueRef;
        assert_eq!(ir.operands(3)[0], s);
        assert_eq!(ir.operands(4)[1], c);
        assert_eq!(ir.operands(6)[1], a);
        // The original `%c` is still used above the region, by `%s`.
        assert_eq!(ir.operands(1), [ir.a, ir.c]);
        ir.verify()
    }

    #[test]
    fn cloned_operations_bring_their_operands_and_are_ordered_by_their_uses() -> TestResult {
        let ir = Nested::parse(NESTED)?;
        let captured = Region::make_isolated_from_above(ir.outer, &mut ir.rewriter(), |_| true);
        // `%s` and `%c` are cloned, `%a`, the operand of `%s`, is captured.
        assert_eq!(captured.as_slice(), [ir.a]);
        let [_, a] = entry_arguments(ir.outer)[..] else {
            unreachable!()
        };

        let entry = ir.outer.borrow().entry_block_ref().unwrap();
        let ops = {
            let entry = entry.borrow();
            entry.body().iter().map(|op| op.as_operation_ref()).collect::<Vec<_>>()
        };
        // `%c` was found after `%s`, and is cloned before it, as `%s` uses it.
        assert_eq!(ops[0].name().to_string(), "test.constant");
        assert_eq!(ops[1].name().to_string(), "test.add");
        assert_eq!(ops[2], ir.ops[3]);
        let c = ops[0].borrow().results()[0] as ValueRef;
        let s = ops[1].borrow().results()[0] as ValueRef;
        assert_eq!(operand_values(ops[1]), [a, c]);
        assert_eq!(ir.operands(3)[0], s);
        assert_eq!(ir.operands(4)[1], c);
        assert_eq!(ir.operands(7)[1], s);
        assert!(Region::get_used_values_defined_above_any(&[ir.outer]).is_empty());
        ir.verify()
    }

    #[test]
    fn a_region_of_several_blocks_is_isolated_through_its_entry_block() -> TestResult {
        let source = r#"builtin.function public extern("C") @f(%a: i32, %b: i32) {
    %c = test.constant 1 : i32;
    "test.ru_holder"(%b) ({
    ^entry(%x: i32):
        "test.ru_br"() [ ^next(%x : i32) ] : () -> ();
    ^next(%y: i32):
        %t = test.add %y, %c;
        "test.ru_yield"(%a, %t) : (i32, i32) -> ();
    }) : i32 -> ();
    builtin.ret;
};"#;
        let (context, function) = parse_function(source)?;
        let region = holder_regions(function)[0];
        let a = function_arguments(function)[0];
        let mut rewriter = RewriterImpl::<NoopRewriterListener>::new(context);
        let captured =
            Region::make_isolated_from_above(region, &mut rewriter, |op| !op.has_operands());
        assert_eq!(captured.as_slice(), [a]);

        let blocks = {
            let region = region.borrow();
            region.body().iter().map(|block| block.as_block_ref()).collect::<Vec<_>>()
        };
        let [entry, next] = blocks[..] else {
            panic!("the region keeps its two blocks");
        };
        let entry_ops = block_ops(entry);
        let names = entry_ops.iter().map(|op| op.name().to_string()).collect::<Vec<_>>();
        assert_eq!(names, ["test.constant", "test.ru_br"]);
        assert_eq!(entry_arguments(region).len(), 2);
        let a2 = entry_arguments(region)[1];
        let clone = entry_ops[0].borrow().results()[0] as ValueRef;
        let next_ops = block_ops(next);
        assert_eq!(operand_values(next_ops[0])[1], clone);
        assert_eq!(operand_values(next_ops[1])[0], a2);
        function.borrow().recursively_verify()
    }

    #[test]
    fn an_operation_holding_a_symbol_reference_is_captured_rather_than_cloned() -> TestResult {
        let source = r#"builtin.module public @m {
    builtin.function public extern("C") @g(%a: i32) -> i32 {
        builtin.ret %a : (i32);
    };
    builtin.function public extern("C") @f(%a: i32) {
        %r = builtin.call @g(%a) : extern("C") (i32) -> i32;
        "test.ru_holder"() ({
            "test.ru_yield"(%r) : i32 -> ();
        }) : () -> ();
        builtin.ret;
    };
};"#;
        let context = Rc::new(Context::default());
        let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "m", source)
            .into_result()?;
        let region = holder_regions(module.as_operation_ref())[0];
        let used = Region::get_used_values_defined_above_any(&[region]);
        let mut rewriter = RewriterImpl::<NoopRewriterListener>::new(context);
        let captured = Region::make_isolated_from_above(region, &mut rewriter, |_| true);
        assert_eq!(captured, used);
        assert_eq!(entry_arguments(region).len(), 1);
        module.as_operation_ref().borrow().recursively_verify()
    }
}
