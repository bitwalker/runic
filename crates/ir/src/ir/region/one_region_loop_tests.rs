//! A loop with one region (`test.loop`) as the target of a continue: the verifier rule that
//! rejects a continue from the header of the loop it targets does not apply to it, and the
//! questions about loops are answered for it.

use alloc::rc::Rc;

use crate::{
    Context, Operation, OperationRef, RegionBranchOpInterface, RegionBranchPoint,
    RegionSuccessorInfo,
    dialects::test::TestDialect,
    parse::{self, ParserConfig},
    testing::text::describe,
};

fn parse_verified(source: &str) -> Result<(Rc<Context>, OperationRef), alloc::string::String> {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let config = ParserConfig::new(context.clone());
    match parse::parse_any(config, "one_region_loop", source).into_result() {
        Ok(op) => Ok((context, op)),
        Err(report) => Err(describe(&report)),
    }
}

/// Why `source` does not verify.
fn rejection(source: &str) -> alloc::string::String {
    match parse_verified(source) {
        Ok(_) => panic!("the source verifies:\n{source}"),
        Err(message) => message,
    }
}

/// The first operation named `name` under `root`, in pre-order.
fn find(root: OperationRef, name: &str) -> OperationRef {
    let mut found = None;
    root.borrow().prewalk_all(|op: &Operation| {
        if found.is_none() && alloc::string::ToString::to_string(&op.name()) == name {
            found = Some(op.as_operation_ref());
        }
    });
    found.unwrap_or_else(|| panic!("no {name}"))
}

/// A loop whose body is `body`, with `%i` its argument, inside a function of `%n` and `%c`.
fn function(body: &str) -> alloc::string::String {
    alloc::format!(
        r#"builtin.function public extern("C") @f(%n: i32, %c: i1) -> i32 {{
    %r = test.loop %n {{
    ^head(%i: i32):
        {body}
    }} : (i32) -> (i32);
    builtin.ret %r : (i32);
}};"#
    )
}

/// The verifier accepts a continue from the region of a loop that has one region, directly and
/// from a forwarding operation nested in it.
#[test]
fn a_continue_from_the_body_of_a_one_region_loop_verifies() {
    parse_verified(&function("test.continue %i : (i32);")).unwrap();
    parse_verified(&function(
        "test.guard %c { test.continue %i : (i32); } : (i1); test.yield %i : (i32);",
    ))
    .unwrap();
}

/// The other rules of a continue still hold in the one-region loop: it carries the header's
/// arguments, and it needs a loop to address.
#[test]
fn a_continue_to_a_one_region_loop_still_carries_the_header_arguments() {
    let err = rejection(&function("test.continue %c : (i1);"));
    assert!(
        err.contains("this continue carries (i1), but the loop it targets iterates on (i32)"),
        "{err}"
    );

    let err = rejection(
        r#"builtin.function public extern("C") @f(%n: i32) {
    test.continue %n : (i32);
};"#,
    );
    assert!(err.contains("this continue names no enclosing loop"), "{err}");
}

/// A loop with one region has a loop in it, and a nested exit, though its region graph has no
/// edge back to its region.
#[test]
fn a_one_region_loop_is_a_loop_and_has_a_nested_exit_or_loop() {
    let (_context, root) = parse_verified(&function("test.continue %i : (i32);")).unwrap();
    let the_loop = find(root, "test.loop");
    assert!(the_loop.borrow().has_nested_exit_or_loop());
    let region_branch = the_loop.borrow();
    let region_branch = region_branch.as_trait::<dyn RegionBranchOpInterface>().unwrap();
    assert!(region_branch.has_loop());
    assert!(region_branch.is_repetitive_region(0));

    // The region graph holds no edge from the region back to itself: the edge is the continue's.
    let region = the_loop.borrow().regions().front().as_pointer().unwrap();
    let successors = region_branch.get_successor_regions(RegionBranchPoint::Child(region));
    let infos = successors.into_successor_infos();
    assert!(matches!(infos.as_slice(), [RegionSuccessorInfo::Returning(_)]), "{infos:?}");
}

/// A forwarding operation that holds the continue of a loop it is nested in has a nested exit.
#[test]
fn an_operation_holding_the_continue_of_its_loop_has_a_nested_exit() {
    let (_context, root) = parse_verified(&function(
        "test.guard %c { test.continue %i : (i32); } : (i1); test.yield %i : (i32);",
    ))
    .unwrap();
    assert!(find(root, "test.guard").borrow().has_nested_exit_or_loop());
}
