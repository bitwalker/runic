use alloc::{format, rc::Rc};

use crate::{
    Context, OperationRef,
    diagnostics::Report,
    dialects::test::TestDialect,
    parse::{self, ParserConfig},
    testing::text::describe,
};

/// Parses `source` without verifying it, in a context with the `test` dialect registered.
fn parse_unverified(source: &str) -> (Rc<Context>, OperationRef) {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let config = ParserConfig::new(context.clone()).verify_after_parse(false);
    let op = parse::parse_any(config, "dialect_attributes", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    (context, op)
}

/// How many times the `test` dialect's hook has been called in `context`.
fn checks(context: &Context) -> usize {
    let dialect = context.get_or_register_dialect::<TestDialect>();
    // SAFETY: the dialect registered under the name `test` is the `TestDialect` registered above.
    unsafe { &*(Rc::as_ptr(&dialect) as *const TestDialect) }.attribute_checks()
}

/// A function whose dictionary is `dictionary`, holding a `test.add` whose dictionary is `inner`.
fn source(dictionary: &str, inner: &str) -> alloc::string::String {
    format!(
        r#"builtin.function public extern("C") @f(%a: i32) -> i32 {{
    %s = test.add %a, %a attributes {{ {inner} }};
    builtin.ret %s : (i32);
}} attributes {{ {dictionary} }};"#
    )
}

fn verify_all(root: OperationRef) -> Result<(), Report> {
    let mut result = Ok(());
    root.borrow().prewalk_all(|op: &crate::Operation| {
        if result.is_ok() {
            result = op.verify();
        }
    });
    result
}

/// An entry the hook accepts is accepted on a `builtin.function` and on a `test` operation.
#[test]
fn a_dialect_checks_its_entry_on_an_operation_of_another_dialect_and_of_its_own() {
    let (_context, root) =
        parse_unverified(&source("test.checked = #builtin.i32<1>", "note = #builtin.i32<2>"));
    verify_all(root).unwrap();
    let (_context, root) =
        parse_unverified(&source("note = #builtin.i32<1>", "test.checked = #builtin.i32<2>"));
    verify_all(root).unwrap();
}

/// A value the hook rejects is the error of the verification, on either kind of operation.
#[test]
fn the_error_of_the_hook_is_the_error_of_the_verification() {
    for (outer, inner, name) in [
        ("test.checked = #builtin.bool<true>", "", "builtin.function"),
        ("", "test.checked = #builtin.bool<true>", "test.add"),
    ] {
        let (_context, root) = parse_unverified(&source(outer, inner));
        let message = describe(&verify_all(root).unwrap_err());
        assert!(
            message.contains(&format!(
                "on {name}: 'test.checked' must be an `i32` constant, but it is a '"
            )),
            "{message}"
        );
    }
}

/// A key of the dialect's own that the hook does not know is rejected with its name.
#[test]
fn a_key_of_the_dialect_that_the_hook_does_not_know_is_rejected() {
    let (_context, root) = parse_unverified(&source("", "test.unknown_key = #builtin.i32<1>"));
    let message = describe(&verify_all(root).unwrap_err());
    assert!(
        message.contains(
            "'test.unknown_key' is not an attribute of the 'test' dialect: it knows 'test.checked'"
        ),
        "{message}"
    );
}

/// A key with no dot, and one whose prefix names no registered dialect, are not checked.
#[test]
fn a_dotless_key_and_a_key_of_no_registered_dialect_are_accepted() {
    let (context, root) = parse_unverified(&source(
        "plain = #builtin.i32<1>, nodialect.x = #builtin.bool<true>",
        "nodialect.y.z = #builtin.i32<1>, also_plain = #builtin.i32<2>",
    ));
    verify_all(root).unwrap();
    assert_eq!(checks(&context), 0);
}

/// The hook is called once per entry named for the dialect and per verification, whatever the
/// operation's own dialect, and not for any other entry.
#[test]
fn the_hook_is_called_once_per_entry_of_its_dialect() {
    let (context, root) = parse_unverified(&source(
        "test.checked = #builtin.i32<1>, nodialect.x = #builtin.i32<1>, plain = #builtin.i32<1>",
        "test.checked = #builtin.i32<2>, test.any = #builtin.bool<true>, nodialect.y = \
         #builtin.i32<2>",
    ));
    assert_eq!(checks(&context), 0, "parsing without verification calls no hook");
    let function = root;
    let add = function.borrow().regions().front().get().unwrap().entry().front().unwrap();

    function.borrow().verify().unwrap();
    assert_eq!(checks(&context), 1);
    add.borrow().verify().unwrap();
    assert_eq!(checks(&context), 3);
    add.borrow().verify().unwrap();
    assert_eq!(checks(&context), 5);
}

/// A property of an operation is not a dictionary entry: it is not passed to the hook.
#[test]
fn a_property_is_not_passed_to_the_hook() {
    let (context, root) = parse_unverified(
        r#"builtin.function public extern("C") @f(%a: i32) -> i32 {
    %s = test.add %a, %a <{ overflow = #builtin.overflow<wrapping> }>;
    builtin.ret %s : (i32);
};"#,
    );
    let function = root;
    let add = function.borrow().regions().front().get().unwrap().entry().front().unwrap();
    assert!(add.borrow().properties().count() > 0);
    assert!(!add.borrow().has_attributes());
    add.borrow().verify().unwrap();
    assert_eq!(checks(&context), 0);
}
