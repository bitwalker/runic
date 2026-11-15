//! The structural rules and the dominance walk of [Operation::recursively_verify], one case per
//! rule, each with a valid counterpart.
//!
//! Most inputs are parsed without verification and then verified; an input the text cannot
//! express (a value read outside the region it is defined in, or across an operation isolated
//! from above) is parsed valid and then edited through the API.

use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    BlockRef, Context, EntityWithId, OpPrinter, Operation, OperationRef, RegionKind,
    RegionKindInterface, Report, Value, ValueRef,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::test::TestDialect,
    effects::MemoryEffectOpInterface,
    parse::{ParserConfig, parse_any},
    testing::text::describe,
    traits::{
        AnyInteger, BranchOpInterface, HasOnlyGraphRegion, InferTypeOpInterface, NoTerminator,
        Terminator,
    },
};

/// A branch to one block: `test.dom_br ^bb1(%0 : i32);`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(Terminator),
    implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct DomBr {
    #[successor]
    target: Successor,
}

impl BranchOpInterface for DomBr {}

/// A branch to either of two blocks: `test.dom_either ^bb1, ^bb2;`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(Terminator),
    implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct DomEither {
    #[successor]
    first: Successor,
    #[successor]
    second: Successor,
}

impl BranchOpInterface for DomEither {}

/// An operation holding one region with SSA dominance, which may read values from above.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(NoTerminator),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct DomRegion {
    #[region]
    body: RegionRef,
}

/// An operation holding one region with SSA dominance, with a result: `%0 = test.dom_holder {..};`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(NoTerminator),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct DomHolder {
    #[region]
    body: RegionRef,
    #[result]
    result: AnyInteger,
}

impl InferTypeOpInterface for DomHolder {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

/// An operation holding one graph region.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(NoTerminator, HasOnlyGraphRegion),
    implements(RegionKindInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct DomGraph {
    #[region]
    body: RegionRef,
}

impl RegionKindInterface for DomGraph {
    fn kind(&self) -> RegionKind {
        RegionKind::Graph
    }
}

/// A module of the functions `functions`, parsed without verification.
fn parse(functions: &str) -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let source = format!("builtin.module public @m {{\n{functions}\n}};\n");
    let config = ParserConfig::new(context.clone()).verify_after_parse(false);
    let module = parse_any(config, "dominance.runic", source)
        .into_result()
        .unwrap_or_else(|err| panic!("the input parses: {}", describe(&err)));
    (context, module)
}

/// A module of the one function `@f(%a: i32)` whose body is `body`, parsed without verification.
fn parse_body(body: &str) -> (Rc<Context>, OperationRef) {
    parse(&format!("builtin.function public extern(\"C\") @f(%a: i32) {{\n{body}\n}};"))
}

/// The error `module` fails to verify with: its message, then its labels.
fn rejected(module: OperationRef) -> String {
    match module.borrow().recursively_verify() {
        Ok(()) => panic!("the module verifies:\n{}", module.borrow()),
        Err(err) => describe(&err),
    }
}

/// `module` verifies.
#[track_caller]
fn accepted(module: OperationRef) {
    if let Err(err) = module.borrow().recursively_verify() {
        panic!("the module does not verify: {}\n{}", describe(&err), module.borrow());
    }
}

/// The error the module of `body` fails to verify with.
fn body_rejected(body: &str) -> String {
    let (_context, module) = parse_body(body);
    rejected(module)
}

/// The module of `body` verifies.
#[track_caller]
fn body_accepted(body: &str) {
    let (_context, module) = parse_body(body);
    accepted(module);
}

/// Every operation under `root` named `name`, in order.
fn find_all(root: OperationRef, name: &str) -> Vec<OperationRef> {
    let mut found = Vec::new();
    root.borrow().prewalk_all(|op: &Operation| {
        if op.name().to_string() == name {
            found.push(op.as_operation_ref());
        }
    });
    found
}

/// The `index`-th operation under `root` named `name`.
fn find(root: OperationRef, name: &str, index: usize) -> OperationRef {
    find_all(root, name)[index]
}

/// The first result of `op`.
fn result(op: OperationRef) -> ValueRef {
    op.borrow().results()[0].borrow().as_value_ref()
}

/// Sets operand `index` of `op` to `value`.
fn set_operand(mut op: OperationRef, index: usize, value: ValueRef) {
    op.borrow_mut().operands_mut()[index].borrow_mut().set(value);
}

/// The message of a dominance failure of operand `index` of `op`.
fn not_dominated(index: usize, op: &str) -> String {
    format!("operand #{index} of '{op}' is not dominated by its definition")
}

// ----- In one block -----

#[test]
fn a_use_before_its_definition_in_one_block_is_rejected() {
    let error = body_rejected(
        "    test.sink %x : (i32);
    %x = test.constant 1 : i32;
    builtin.ret;",
    );
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(error.contains("operand #0 is read here"), "{error}");
    assert!(error.contains("it is defined here, later in the same block"), "{error}");
}

#[test]
fn a_use_after_its_definition_in_one_block_is_accepted() {
    body_accepted(
        "    %x = test.constant 1 : i32;
    test.sink %x, %a : (i32, i32);
    builtin.ret;",
    );
}

/// The operand is numbered among all of the operation's operands.
#[test]
fn the_failing_operand_is_named_by_its_index() {
    let error = body_rejected(
        "    test.sink %a, %a, %x : (i32, i32, i32);
    %x = test.constant 1 : i32;
    builtin.ret;",
    );
    assert!(error.starts_with(&not_dominated(2, "test.sink")), "{error}");
}

#[test]
fn an_operation_that_reads_its_own_result_is_rejected() {
    let (_context, module) = parse_body(
        "    %c = test.constant 1 : i32;
    %y = test.add %c, %c;
    builtin.ret;",
    );
    let add = find(module, "test.add", 0);
    set_operand(add, 1, result(add));
    let error = rejected(module);
    assert!(error.starts_with(&not_dominated(1, "test.add")), "{error}");
    assert!(error.contains("it is defined by the operation that reads it"), "{error}");
}

// ----- Across blocks -----

/// A function whose entry branches to `^left` or `^right`, which both go to `^join`; `left`,
/// `right` and `join` are their bodies.
fn diamond(left: &str, right: &str, join: &str) -> String {
    format!(
        "    test.dom_either ^left, ^right;
^left:
{left}
    test.dom_br ^join;
^right:
{right}
    test.dom_br ^join;
^join:
{join}
    builtin.ret;"
    )
}

#[test]
fn a_join_that_reads_a_value_of_one_arm_is_rejected() {
    let error =
        body_rejected(&diamond("    %x = test.constant 1 : i32;", "", "    test.sink %x : (i32);"));
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(
        error.contains("it is defined here, in a block that does not dominate the use"),
        "{error}"
    );
}

#[test]
fn an_arm_that_reads_a_value_of_the_other_arm_is_rejected() {
    let error =
        body_rejected(&diamond("    %x = test.constant 1 : i32;", "    test.sink %x : (i32);", ""));
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(error.contains("in a block that does not dominate the use"), "{error}");
}

/// A block argument is defined by its block, and dominates only what its block dominates.
#[test]
fn a_join_that_reads_an_argument_of_one_arm_is_rejected() {
    let error = body_rejected(
        "    test.dom_either ^left(%a : i32), ^right;
^left(%l: i32):
    test.dom_br ^join;
^right:
    test.dom_br ^join;
^join:
    test.sink %l : (i32);
    builtin.ret;",
    );
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(error.contains("in a block that does not dominate the use"), "{error}");
}

#[test]
fn a_join_that_reads_values_of_the_entry_and_its_arguments_is_accepted() {
    body_accepted(&format!(
        "    %x = test.constant 1 : i32;
{}",
        diamond(
            "    test.sink %x : (i32);",
            "    test.sink %x, %a : (i32, i32);",
            "    test.sink %x, %a : (i32, i32);"
        )
    ));
    body_accepted(
        "    test.dom_either ^left(%a : i32), ^right;
^left(%l: i32):
    test.sink %l : (i32);
    test.dom_br ^join(%l : i32);
^right:
    test.dom_br ^join(%a : i32);
^join(%j: i32):
    test.sink %j : (i32);
    builtin.ret;",
    );
}

/// A loop whose header reads a value its body defines.
#[test]
fn a_loop_header_that_reads_a_value_of_its_body_is_rejected() {
    let error = body_rejected(
        "    test.dom_br ^header;
^header:
    test.sink %x : (i32);
    test.dom_either ^body, ^exit;
^body:
    %x = test.constant 1 : i32;
    test.dom_br ^header;
^exit:
    builtin.ret;",
    );
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(error.contains("in a block that does not dominate the use"), "{error}");
}

/// The same loop, carrying the value through an argument of its header.
#[test]
fn a_loop_that_carries_a_value_through_its_header_is_accepted() {
    body_accepted(
        "    test.dom_br ^header(%a : i32);
^header(%h: i32):
    test.sink %h : (i32);
    test.dom_either ^body, ^exit;
^body:
    %x = test.constant 1 : i32;
    test.sink %x, %h : (i32, i32);
    test.dom_br ^header(%x : i32);
^exit:
    test.sink %h : (i32);
    builtin.ret;",
    );
}

// ----- Across regions -----

#[test]
fn a_region_that_reads_values_defined_above_it_is_accepted() {
    body_accepted(
        "    %x = test.constant 1 : i32;
    test.dom_region {
        test.sink %x, %a : (i32, i32);
        test.dom_region {
            test.sink %x, %a : (i32, i32);
        };
    };
    builtin.ret;",
    );
}

/// A region reads a value of its own block that its operation comes before.
#[test]
fn a_region_that_reads_a_value_defined_after_its_operation_is_rejected() {
    let (_context, module) = parse_body(
        "    %x = test.constant 1 : i32;
    test.dom_region {
        test.sink %x : (i32);
    };
    %later = test.constant 2 : i32;
    builtin.ret;",
    );
    let later = result(find(module, "test.constant", 1));
    set_operand(find(module, "test.sink", 0), 0, later);
    let error = rejected(module);
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(error.contains("later in the same block"), "{error}");
}

#[test]
fn a_value_read_inside_the_operation_that_defines_it_is_rejected() {
    let (_context, module) = parse_body(
        "    %h = test.dom_holder {
        test.sink %a : (i32);
    };
    builtin.ret;",
    );
    let holder = find(module, "test.dom_holder", 0);
    set_operand(find(module, "test.sink", 0), 0, result(holder));
    let error = rejected(module);
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(
        error.contains(
            "it is defined here by 'test.dom_holder', which holds the use in one of its regions"
        ),
        "{error}"
    );
}

#[test]
fn a_value_read_outside_the_region_that_defines_it_is_rejected() {
    let (_context, module) = parse_body(
        "    test.dom_region {
        %inner = test.constant 1 : i32;
    };
    %outer = test.constant 2 : i32;
    test.sink %outer : (i32);
    builtin.ret;",
    );
    let inner = result(find(module, "test.constant", 0));
    set_operand(find(module, "test.sink", 0), 0, inner);
    let error = rejected(module);
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(
        error.contains(
            "it is defined here, inside a region nested in an operation of the use's region"
        ),
        "{error}"
    );
}

#[test]
fn a_value_read_in_a_sibling_region_is_rejected() {
    let (_context, module) = parse_body(
        "    test.dom_region {
        %first = test.constant 1 : i32;
    };
    test.dom_region {
        %second = test.constant 2 : i32;
        test.sink %second : (i32);
    };
    builtin.ret;",
    );
    let first = result(find(module, "test.constant", 0));
    set_operand(find(module, "test.sink", 0), 0, first);
    let error = rejected(module);
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(
        error.contains(
            "it is defined here, in a region that is not an ancestor of the use's region"
        ),
        "{error}"
    );
}

/// A use nested in a region, of a value of a block that does not dominate the block holding the
/// region's operation.
#[test]
fn a_nested_use_is_checked_at_the_block_that_holds_it() {
    let error = body_rejected(&diamond(
        "    %x = test.constant 1 : i32;",
        "    test.dom_region {
        test.sink %x : (i32);
    };",
        "",
    ));
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(error.contains("in a block that does not dominate the use"), "{error}");
}

// ----- Isolation -----

#[test]
fn a_value_read_across_an_operation_isolated_from_above_is_rejected() {
    let (_context, module) = parse(
        "builtin.function public extern(\"C\") @f(%a: i32) {
    builtin.ret;
};
builtin.function public extern(\"C\") @g(%b: i32) {
    test.dom_region {
        test.sink %b : (i32);
    };
    builtin.ret;
};",
    );
    accepted(module);
    let f = find(module, "builtin.function", 0);
    let a = f.borrow().regions().front().get().unwrap().entry().arguments()[0];
    let a = a.borrow().as_value_ref();
    set_operand(find(module, "test.sink", 0), 0, a);
    let error = rejected(module);
    assert!(
        error.starts_with(
            "operand #0 of 'test.sink' is defined outside 'builtin.function', which is isolated \
             from above"
        ),
        "{error}"
    );
    assert!(error.contains("inside this 'builtin.function'"), "{error}");
}

/// Isolation is checked in code that cannot be reached, where dominance is not.
#[test]
fn isolation_is_checked_in_an_unreachable_block() {
    let (_context, module) = parse(
        "builtin.function public extern(\"C\") @f(%a: i32) {
    builtin.ret;
};
builtin.function public extern(\"C\") @g(%b: i32) {
    builtin.ret;
^dead:
    test.sink %b : (i32);
    builtin.ret;
};",
    );
    accepted(module);
    let f = find(module, "builtin.function", 0);
    let a = f.borrow().regions().front().get().unwrap().entry().arguments()[0];
    let a = a.borrow().as_value_ref();
    set_operand(find(module, "test.sink", 0), 0, a);
    let error = rejected(module);
    assert!(error.contains("which is isolated from above"), "{error}");
}

// ----- Code that cannot be reached -----

#[test]
fn a_nonsensical_use_in_an_unreachable_block_is_accepted() {
    body_accepted(
        "    builtin.ret;
^dead:
    test.sink %x, %y : (i32, i32);
    %x = test.constant 1 : i32;
    test.dom_br ^dead2;
^dead2:
    %y = test.constant 2 : i32;
    builtin.ret;",
    );
}

/// An operation with a region, in a block that cannot be reached, reading the function's
/// arguments and values of that block, before and after it: what the region holds is reachable in
/// its own region, but walked out to the function's body it sits in dead code.
#[test]
fn a_region_in_an_unreachable_block_is_not_checked_against_the_function() {
    body_accepted(
        "    builtin.ret;
^dead:
    %d = test.constant 1 : i32;
    test.dom_region {
        test.sink %a, %d, %late : (i32, i32, i32);
    };
    %late = test.constant 2 : i32;
    builtin.ret;",
    );
}

/// The region's own uses are still checked: they are reachable in the region.
#[test]
fn a_region_in_an_unreachable_block_is_checked_within_itself() {
    let error = body_rejected(
        "    builtin.ret;
^dead:
    test.dom_region {
        test.sink %x : (i32);
        %x = test.constant 1 : i32;
    };
    builtin.ret;",
    );
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
}

/// A definition in a block that cannot be reached does not dominate a use that can.
#[test]
fn a_reachable_use_of_a_value_of_an_unreachable_block_is_rejected() {
    let error = body_rejected(
        "    test.dom_br ^exit;
^dead:
    %x = test.constant 1 : i32;
    test.dom_br ^exit;
^exit:
    test.sink %x : (i32);
    builtin.ret;",
    );
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
}

// ----- Graph regions -----

#[test]
fn a_use_before_its_definition_in_a_graph_region_is_accepted() {
    body_accepted(
        "    test.dom_graph {
        test.sink %x : (i32);
        %x = test.constant 1 : i32;
    };
    builtin.ret;",
    );
}

#[test]
fn an_operation_that_reads_its_own_result_in_a_graph_region_is_accepted() {
    let (_context, module) = parse_body(
        "    test.dom_graph {
        %c = test.constant 1 : i32;
        %y = test.add %c, %c;
    };
    builtin.ret;",
    );
    let add = find(module, "test.add", 0);
    set_operand(add, 0, result(add));
    accepted(module);
}

/// A graph region orders nothing in the region outside it.
#[test]
fn a_graph_region_that_reads_a_later_value_of_the_function_is_rejected() {
    let (_context, module) = parse_body(
        "    %x = test.constant 1 : i32;
    test.dom_graph {
        test.sink %x : (i32);
    };
    %later = test.constant 2 : i32;
    builtin.ret;",
    );
    let later = result(find(module, "test.constant", 1));
    set_operand(find(module, "test.sink", 0), 0, later);
    let error = rejected(module);
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
}

// ----- Structure -----

#[test]
fn an_operation_with_successors_that_is_not_last_in_its_block_is_rejected() {
    let error = body_rejected(
        "    test.dom_br ^next;
    test.sink %a : (i32);
^next:
    builtin.ret;",
    );
    assert!(
        error.starts_with(
            "'test.dom_br' has successors, but is not the last operation of its block"
        ),
        "{error}"
    );
}

#[test]
fn a_successor_in_another_region_is_rejected() {
    let (_context, module) = parse_body(
        "    test.dom_region {
        test.dom_br ^inner;
    ^inner:
        test.sink %a : (i32);
    };
    test.dom_br ^next;
^next:
    builtin.ret;",
    );
    accepted(module);
    // The inner branch goes to the function's second block instead.
    let outer = find(module, "test.dom_br", 1);
    let next: BlockRef = outer.borrow().successor(0).dest.borrow().successor();
    let mut inner = find(module, "test.dom_br", 0);
    inner.borrow_mut().successor_mut(0).set(next);
    let error = rejected(module);
    assert!(
        error.starts_with("successor #0 of 'test.dom_br' is not a block of its region"),
        "{error}"
    );
    assert!(error.contains("this branches to a block of another region"), "{error}");
}

#[test]
fn an_entry_block_with_predecessors_is_rejected() {
    let (_context, module) = parse(
        "builtin.function public extern(\"C\") @f() {
^entry:
    test.dom_br ^entry;
};",
    );
    let error = rejected(module);
    assert!(
        error.starts_with("the entry block of region #0 of 'builtin.function' has predecessors"),
        "{error}"
    );
    assert!(error.contains("'test.dom_br' branches to it here"), "{error}");
}

#[test]
fn a_graph_region_with_two_blocks_is_rejected() {
    let error = body_rejected(
        "    test.dom_graph {
        test.sink %a : (i32);
    ^second:
        test.sink %a : (i32);
    };
    builtin.ret;",
    );
    assert!(
        error.starts_with("region #0 of 'test.dom_graph' is a graph region with 2 blocks"),
        "{error}"
    );
}

/// Dominance is checked only once everything else verified: a malformed block structure is
/// reported, not the use it makes undominated.
#[test]
fn the_structure_is_reported_before_dominance() {
    let error = body_rejected(
        "    test.dom_br ^next;
    %x = test.constant 1 : i32;
^next:
    test.sink %x : (i32);
    builtin.ret;",
    );
    assert!(error.contains("has successors, but is not the last operation"), "{error}");
}

/// `Operation::verify` checks one operation: it does not look at dominance or structure.
#[test]
fn verify_checks_one_operation_only() {
    let (_context, module) = parse_body(
        "    test.sink %x : (i32);
    %x = test.constant 1 : i32;
    builtin.ret;",
    );
    let function = find(module, "builtin.function", 0);
    assert!(function.borrow().verify().is_ok());
    assert!(find(module, "test.sink", 0).borrow().verify().is_ok());
}

/// The region kinds and dominator trees the walk computes do not outlive it: a module that
/// verified, changed into one that does not, is rejected by the next verification.
#[test]
fn each_verification_computes_dominance_afresh() {
    let (_context, module) = parse_body(
        "    %x = test.constant 1 : i32;
    test.dom_either ^left, ^right;
^left:
    test.sink %x : (i32);
    test.dom_br ^join;
^right:
    test.dom_br ^join;
^join:
    builtin.ret;",
    );
    accepted(module);
    // Move the constant into `^right`.
    let mut constant = find(module, "test.constant", 0);
    let right = find(module, "test.dom_br", 1).borrow().parent().unwrap();
    constant.borrow_mut().move_to(crate::ProgramPoint::at_start_of(right));
    let error = rejected(module);
    assert!(error.contains("in a block that does not dominate the use"), "{error}");
}

/// Verifying an operation nested in a function checks its uses of the function's values too.
#[test]
fn a_nested_operation_verified_alone_is_checked_against_the_values_outside_it() {
    let (_context, module) = parse_body(&format!(
        "    %x = test.constant 1 : i32;
{}",
        diamond(
            "    %y = test.constant 2 : i32;",
            "    test.dom_region {
        test.sink %x : (i32);
    };",
            ""
        )
    ));
    let region_op = find(module, "test.dom_region", 0);
    region_op.borrow().recursively_verify().expect("the region verifies");
    let y = result(find(module, "test.constant", 1));
    set_operand(find(module, "test.sink", 0), 0, y);
    let error = match region_op.borrow().recursively_verify() {
        Ok(()) => panic!("the region verifies"),
        Err(err) => describe(&err),
    };
    assert!(error.starts_with(&not_dominated(0, "test.sink")), "{error}");
    assert!(error.contains("in a block that does not dominate the use"), "{error}");
}
