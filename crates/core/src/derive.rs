pub use runic_macros::{
    AttributeComponent, Dialect, DialectAttribute, DialectRegistration, DialectType,
    EffectOpInterface, OpParser, OpPrinter, TypeComponent, operation, operation_trait,
};

/// What derived code names through `::runic_core`, so that a crate using a derive need not
/// declare these crates itself.
#[doc(hidden)]
pub mod __private {
    pub use alloc::boxed::Box;
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use crate::{
        BuilderExt, Context, Op, Operation, Report, Spanned, Value, ValueRef,
        derive::operation_trait,
        dialects::{
            builtin::attributes::Overflow,
            test::{self, Add},
        },
        pass::{Nesting, PassManager},
        testing::Test,
    };

    /// A marker trait for arithmetic ops
    #[operation_trait]
    trait ArithmeticOp {
        #[verifier]
        fn is_binary_op(op: &Operation, _ctx: &Context) -> Result<(), Report> {
            if op.num_operands() == 2 {
                Ok(())
            } else {
                Err(crate::diagnostics::report!(
                    message: "invalid operation",
                    labels: [
                        primary(
                            op.span(),
                            format!(
                                "incorrect number of operands, expected 2, got {}",
                                op.num_operands()
                            )
                        ),
                    ],
                    notes: [
                        help(
                            "this operator implements 'ArithmeticOp' which requires ops to be \
                             binary"
                        ),
                    ],
                ))
            }
        }
    }

    impl ArithmeticOp for Add {}

    inventory::submit!(crate::DialectRegistrationHookInfo::new::<test::TestDialect>(
        register_arithmetic_op_trait
    ));

    fn register_arithmetic_op_trait(info: &mut crate::DialectInfo) {
        info.register_operation_trait::<Add, dyn ArithmeticOp>();
    }

    #[test]
    fn derived_op_builder_test() {
        use crate::SourceSpan;

        let mut test = Test::named("derived_op_builder");
        let i32 = test.context().i32();
        test.with_function("derived_op_builder", &[i32, i32], &[]);

        let entry = test.entry_block();
        let (lhs, rhs) = {
            let block = entry.borrow();
            let lhs = block.get_argument(0) as ValueRef;
            let rhs = block.get_argument(1) as ValueRef;
            (lhs, rhs)
        };
        let builder = test.builder_mut();
        let add_builder = builder.create::<Add, (_, _, _)>(SourceSpan::default());
        let op = add_builder(lhs, rhs, Overflow::Wrapping);
        let op = op.expect("failed to create AddOp");
        let op = op.borrow();
        assert!(op.as_operation().implements::<dyn ArithmeticOp>());
        assert!(core::hint::black_box(
            !<Add as crate::verifier::Verifier<dyn ArithmeticOp>>::VACUOUS
        ));
    }

    /// `test.add`'s `overflow` is `#[default]`, so it has a builder form without it, which uses
    /// the attribute's registered default, as a parsed `test.add` without it does.
    #[test]
    fn the_short_builder_form_uses_the_registered_default() {
        use crate::SourceSpan;

        let mut test = Test::named("short_builder_form");
        let i32 = test.context().i32();
        test.with_function("short_builder_form", &[i32, i32], &[]);
        let entry = test.entry_block();
        let (lhs, rhs) = {
            let block = entry.borrow();
            (block.get_argument(0) as ValueRef, block.get_argument(1) as ValueRef)
        };
        let add = test.builder_mut().create::<Add, (_, _)>(SourceSpan::default())(lhs, rhs);
        let add = add.expect("failed to create AddOp");
        assert_eq!(*add.borrow().get_overflow(), Overflow::Wrapping);
        assert_eq!(format!("{}", add.borrow().as_operation()), "%2 = test.add %0, %1;");
    }

    #[test]
    #[should_panic = "expected 'i32', got 'i64'"]
    fn derived_op_verifier_test() {
        use crate::SourceSpan;

        let mut test = Test::named("derived_op_verifier");
        let (i32, i64) = (test.context().i32(), test.context().i64());
        test.with_function("derived_op_verifier", &[i32, i64], &[]);

        let entry = test.entry_block();
        let (lhs, invalid_rhs) = {
            let block = entry.borrow();
            let lhs = block.get_argument(0) as ValueRef;
            let rhs = block.get_argument(1) as ValueRef;
            (lhs, rhs)
        };

        // Try to create instance of AddOp with mismatched operand types
        let add_builder = test.builder_mut().create::<Add, (_, _, _)>(SourceSpan::default());
        let op = add_builder(lhs, invalid_rhs, Overflow::Wrapping);
        let op = op.unwrap();

        // Construct a pass manager with the default pass pipeline
        let mut pm = PassManager::on::<Add>(test.context_rc(), Nesting::Implicit);
        // Run pass pipeline
        pm.run(op.as_operation_ref()).unwrap();
    }

    /// Fails if [`InvalidOpsWithReturn`] is created successfully. [`InvalidOpsWithReturn`] is a
    /// struct that has differing types in its result and arguments, despite implementing the
    /// [`SameOperandsAndResultType`] trait.
    #[test]
    #[should_panic = "expected 'i32', got 'i64'"]
    fn same_operands_and_result_type_verifier_test() {
        use crate::SourceSpan;

        let mut test = Test::named("same_operands_and_result_type_verifier");
        let (i32, i64) = (test.context().i32(), test.context().i64());
        test.with_function("same_operands_and_result_type_verifier", &[i32, i32], &[]);
        let block = test.entry_block();
        let (lhs, rhs) = {
            let block = block.borrow();
            let lhs = block.get_argument(0) as ValueRef;
            let rhs = block.get_argument(1) as ValueRef;
            (lhs, rhs)
        };

        let add_builder = test.builder_mut().create::<Add, (_, _, _)>(SourceSpan::default());
        let op = add_builder(lhs, rhs, Overflow::Wrapping);
        let mut op = op.unwrap();

        // NOTE: We override the result's type in order to force the SameOperandsAndResultType
        // verification function to trigger an error
        {
            let mut binding = op.borrow_mut();
            let mut result = binding.result_mut();
            result.set_type(i64);
        }

        // Construct a pass manager with the default pass pipeline
        let mut pm = PassManager::on::<Add>(test.context_rc(), Nesting::Implicit);
        // Run pass pipeline
        pm.run(op.as_operation_ref()).unwrap();
    }
}
