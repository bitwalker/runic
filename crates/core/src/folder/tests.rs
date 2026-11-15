use super::*;
use crate::{
    Listener, ListenerType, Op, OpBuilder, Spanned,
    dialects::{
        builtin::BuiltinOpBuilder,
        debuginfo::attributes::{
            INLINE_CALL_CHAIN_ATTR_NAME, InlineCallChain, InlineCallChainAttr, InlineCallFrame,
        },
        test::TestOpBuilder,
    },
    patterns::{NoopRewriterListener, RewriterListener},
    testing::Test,
};

fn mark_inline(test: &Test, value: ValueRef) -> OperationRef {
    let chain = test
        .context_rc()
        .create_attribute::<InlineCallChainAttr, _>(InlineCallChain::new(alloc::vec![
            InlineCallFrame {
                name: "callee".into(),
                linkage_name: None,
                file: "callee.rs".into(),
                line: 1,
                column: 1,
                call_file: "caller.rs".into(),
                call_line: 7,
                call_column: 1,
            }
        ]))
        .as_attribute_ref();
    let mut op = value.borrow().get_defining_op().unwrap();
    op.borrow_mut().set_attribute(INLINE_CALL_CHAIN_ATTR_NAME, chain);
    op
}

fn assert_location_erased(op: OperationRef) {
    let op = op.borrow();
    assert_eq!(op.span(), SourceSpan::UNKNOWN);
    assert!(!op.has_attribute(INLINE_CALL_CHAIN_ATTR_NAME));
}

#[test]
fn deduplicated_constant_loses_inline_provenance() {
    let mut test = Test::named("deduplicated_constant");
    let i32 = test.context().i32();
    test.with_function("deduplicated_constant", &[], &[i32, i32]);
    let (first, second, ret) = {
        let mut builder = test.function_builder();
        let first = builder.u32(42, SourceSpan::SYNTHETIC).unwrap();
        let second = builder.u32(42, SourceSpan::SYNTHETIC).unwrap();
        let ret = builder.ret([first, second], SourceSpan::UNKNOWN).unwrap();
        (first, second, ret)
    };
    let first_op = mark_inline(&test, first);
    let second_op = mark_inline(&test, second);
    let mut folder = OperationFolder::new(test.context_rc(), NoopRewriterListener);
    assert!(folder.insert_known_constant(first_op, None));
    assert!(first_op.borrow().has_attribute(INLINE_CALL_CHAIN_ATTR_NAME));
    assert!(!folder.insert_known_constant(second_op, None));
    assert_location_erased(first_op);
    for operand in ret.borrow().as_operation().operands().all() {
        assert_eq!(operand.borrow().as_value_ref(), first);
    }
}

#[test]
fn hoisted_constant_loses_inline_provenance() {
    let mut test = Test::named("hoisted_constant");
    let i32 = test.context().i32();
    test.with_function("hoisted_constant", &[], &[i32]);
    let value = {
        let mut builder = test.function_builder();
        builder.u32(1, SourceSpan::UNKNOWN).unwrap();
        let value = builder.u32(42, SourceSpan::SYNTHETIC).unwrap();
        builder.ret([value], SourceSpan::UNKNOWN).unwrap();
        value
    };
    let op = mark_inline(&test, value);
    let mut folder = OperationFolder::new(test.context_rc(), NoopRewriterListener);
    assert!(folder.insert_known_constant(op, None));
    assert_eq!(op.parent().unwrap().borrow().front().unwrap(), op);
    assert_location_erased(op);
}

#[test]
fn rehoisted_constant_loses_inline_provenance() {
    for via_fold in [false, true] {
        let mut test = Test::named("rehoisted_constant");
        let i32 = test.context().i32();
        test.with_function("rehoisted_constant", &[i32], &[i32]);
        let value = {
            let mut builder = test.function_builder();
            let value = builder.u32(42, SourceSpan::SYNTHETIC).unwrap();
            builder.ret([value], SourceSpan::UNKNOWN).unwrap();
            value
        };
        let op = mark_inline(&test, value);
        let mut folder = OperationFolder::new(test.context_rc(), NoopRewriterListener);
        assert!(folder.insert_known_constant(op, None));
        // Registering a constant already in place must preserve its original location.
        assert!(folder.insert_known_constant(op, None));
        assert_eq!(op.borrow().span(), SourceSpan::SYNTHETIC);
        assert!(op.borrow().has_attribute(INLINE_CALL_CHAIN_ATTR_NAME));
        let mut builder = OpBuilder::new(test.context_rc());
        builder.set_insertion_point_before(op);
        let input = op.parent().unwrap().borrow().arguments()[0].upcast();
        builder.add(input, input, SourceSpan::UNKNOWN).unwrap();
        if via_fold {
            assert!(matches!(folder.try_fold(op), FoldResult::Failed));
        } else {
            assert!(folder.insert_known_constant(op, None));
        }
        assert_eq!(op.parent().unwrap().borrow().front().unwrap(), op);
        assert_location_erased(op);
    }
}

#[test]
fn reused_constant_with_unknown_span_loses_inline_provenance() {
    check_reused_constant(false);
}

#[test]
fn cross_dialect_reused_constant_loses_inline_provenance() {
    check_reused_constant(true);
}

fn check_reused_constant(cross_dialect: bool) {
    let mut test = Test::named("reused_constant");
    let i32 = test.context().i32();
    test.with_function("reused_constant", &[], &[i32]);
    let value = {
        let mut builder = test.function_builder();
        let value = builder.u32(42, SourceSpan::UNKNOWN).unwrap();
        builder.ret([value], SourceSpan::UNKNOWN).unwrap();
        value
    };
    let op = mark_inline(&test, value);
    let mut folder = OperationFolder::new(test.context_rc(), NoopRewriterListener);
    assert!(folder.insert_known_constant(op, None));
    let dialect: Rc<dyn Dialect> = if cross_dialect {
        test.context_rc().get_or_register_dialect::<ForwardingDialect>()
    } else {
        op.borrow().dialect()
    };
    let attr = crate::matchers::constant().matches(&op.borrow()).unwrap();
    let reused =
        folder.get_or_create_constant(op.parent().unwrap(), dialect, attr, test.context().i32());
    assert_eq!(reused, Some(value));
    assert_location_erased(op);
}

/// Materializes constants in another dialect to exercise the folder's alternate cache key.
#[derive(crate::derive::DialectRegistration, Debug)]
struct ForwardingDialect {
    #[dialect(info)]
    info: crate::DialectInfo,
}

impl From<crate::DialectInfo> for ForwardingDialect {
    fn from(info: crate::DialectInfo) -> Self {
        Self { info }
    }
}

impl Dialect for ForwardingDialect {
    fn info(&self) -> &crate::DialectInfo {
        &self.info
    }

    fn materialize_constant(
        &self,
        builder: &mut dyn Builder,
        value: AttributeRef,
        ty: Type,
        span: SourceSpan,
    ) -> Option<OperationRef> {
        builder
            .context_rc()
            .get_or_register_dialect::<crate::dialects::test::TestDialect>()
            .materialize_constant(builder, value, ty, span)
    }
}

/// Borrows every op it is notified about, as a verifying listener does.
struct BorrowingListener;

impl Listener for BorrowingListener {
    fn kind(&self) -> ListenerType {
        ListenerType::Rewriter
    }

    fn notify_operation_inserted(&self, op: OperationRef, _prev: ProgramPoint) {
        let _ = op.borrow();
    }
}

impl RewriterListener for BorrowingListener {
    fn notify_operation_modified(&self, op: OperationRef) {
        let _ = op.borrow();
    }

    fn notify_operation_replaced_with_values(
        &self,
        op: OperationRef,
        _replacement: &[Option<ValueRef>],
    ) {
        let _ = op.borrow();
    }

    fn notify_operation_erased(&self, op: OperationRef) {
        let _ = op.borrow();
    }
}

/// Folding an op notifies the listeners only after its borrow of the op is released.
#[test]
fn folding_releases_the_op_before_notifying_listeners() {
    let mut test = Test::named("fold_listener");
    let i32 = test.context().i32();
    test.with_function("fold_listener", &[], &[i32, i32]);
    let (first, second) = {
        let mut builder = test.function_builder();
        let first = builder.u32(42, SourceSpan::UNKNOWN).unwrap();
        let second = builder.u32(42, SourceSpan::UNKNOWN).unwrap();
        builder.ret([first, second], SourceSpan::UNKNOWN).unwrap();
        (first, second)
    };
    let first_op = first.borrow().get_defining_op().unwrap();
    let second_op = second.borrow().get_defining_op().unwrap();
    let mut folder = OperationFolder::new(test.context_rc(), BorrowingListener);
    assert!(folder.insert_known_constant(first_op, None));
    // The duplicate constant folds to the known one, which erases it and replaces its uses
    assert!(matches!(folder.try_fold(second_op), FoldResult::Ok(())));
}

/// In a graph region two casts between equal types read each other. Each folds to its operand,
/// but replacing the result of either by the other's would leave the other reading its own
/// result: the fold is refused and the operations stay. A cast in no cycle folds.
#[test]
fn a_fold_that_would_make_an_operation_read_its_own_result_is_refused() {
    use alloc::string::ToString;

    use crate::{
        dialects::{builtin::UnrealizedConversionCast, test::TestDialect},
        parse::{ParserConfig, parse_any},
        testing::text::describe,
    };

    let source = "builtin.module public @m {
    %0 = builtin.unrealized_conversion_cast %1 : i32 to i32;
    %1 = builtin.unrealized_conversion_cast %0 : i32 to i32;
    %2 = test.constant 1 : i32;
    %3 = builtin.unrealized_conversion_cast %2 : i32 to i32;
    test.sink %0, %3 : (i32, i32);
};";
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let module = parse_any(ParserConfig::new(context.clone()), "folder", source)
        .into_result()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let mut casts = alloc::vec::Vec::new();
    module.borrow().prewalk_all(|op: &crate::Operation| {
        if op.is::<UnrealizedConversionCast>() {
            casts.push(op.as_operation_ref());
        }
    });
    let printed = module.borrow().to_string();
    let mut folder = OperationFolder::new(context.clone(), NoopRewriterListener);
    assert!(matches!(folder.try_fold(casts[0]), FoldResult::Failed));
    assert!(matches!(folder.try_fold(casts[1]), FoldResult::Failed));
    assert_eq!(module.borrow().to_string(), printed);

    assert!(folder.try_fold(casts[2]).is_ok());
    module
        .borrow()
        .recursively_verify()
        .unwrap_or_else(|err| panic!("{}", describe(&err)));
    let after = module.borrow().to_string();
    assert_eq!(after.matches("unrealized_conversion_cast").count(), 2, "{after}");
}
