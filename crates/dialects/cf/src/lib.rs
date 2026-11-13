//! The `cf` dialect: unstructured control flow (`cf.br`, `cf.cond_br`, `cf.switch`), `cf.select`,
//! and `cf.assert`, a check that traps with a message when its condition is `0`.

#![no_std]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(specialization)]
#![allow(incomplete_features)]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod builders;
mod canonicalization;
mod ops;

use runic_core::{
    AttributeRef, Builder, Dialect, DialectInfo, OperationRef, SourceSpan, Type,
    attributes::IntegerLikeAttr, derive::DialectRegistration,
};
use runic_dialect_arith::ArithDialect;
use runic_dialect_ub::{PoisonAttr, UndefinedBehaviorDialect};

pub use self::{builders::ControlFlowOpBuilder, ops::*};

#[derive(Debug, DialectRegistration)]
#[dialect(name = "cf")]
pub struct ControlFlowDialect {
    info: DialectInfo,
}

impl From<DialectInfo> for ControlFlowDialect {
    fn from(info: DialectInfo) -> Self {
        Self { info }
    }
}

impl Dialect for ControlFlowDialect {
    #[inline]
    fn info(&self) -> &DialectInfo {
        &self.info
    }

    /// Materializes a constant that a `cf` operation folded to: an integer-like `attr` as an
    /// `arith.constant` (see [ArithDialect]), a [PoisonAttr] as a `ub.poison` (see
    /// [UndefinedBehaviorDialect]). Any other attribute is not materialized.
    fn materialize_constant(
        &self,
        builder: &mut dyn Builder,
        attr: AttributeRef,
        ty: Type,
        span: SourceSpan,
    ) -> Option<OperationRef> {
        let context = builder.context_rc();
        let is_integer = attr.borrow().as_attr().as_trait::<dyn IntegerLikeAttr>().is_some();
        if is_integer {
            let arith = context.get_or_register_dialect::<ArithDialect>();
            return arith.materialize_constant(builder, attr, ty, span);
        }
        if attr.try_downcast_attr::<PoisonAttr>().is_ok() {
            let ub = context.get_or_register_dialect::<UndefinedBehaviorDialect>();
            return ub.materialize_constant(builder, attr, ty, span);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, format};

    use runic_core::{
        Int, Op, SourceSpan, Value, ValueRef,
        dialects::builtin::{BuiltinOpBuilder, attributes::StringAttr},
        patterns::RewritePattern,
        testing::Test,
    };
    use runic_dialect_arith::{ArithOpBuilder, Constant};
    use runic_dialect_ub::{Poison, UndefinedBehaviorOpBuilder};
    use runic_expect_test::{Expect, expect};

    use super::*;
    use crate::canonicalization::{SimplifySelectOfBools, testing::apply_with_borrowing_listener};

    /// The arms a test's `cf.select` chooses between, made from the function's `i32` argument.
    #[derive(Clone, Copy)]
    enum Arm {
        Argument,
        Constant(i32),
        Poison,
    }

    /// Builds `f(%c: i1, %a: i32) -> i32` that returns `cf.select cond, first, second`, with
    /// `cond` the constant `value` when given and `%c` otherwise, canonicalizes it and checks
    /// its text against `expected`.
    fn canonicalize_select(cond: Option<bool>, first: Arm, second: Arm, expected: Expect) {
        let mut test = Test::new("f", |cx| ([cx.i1(), cx.i32()], [cx.i32()]));
        let context = test.context_rc();
        let entry = test.function().borrow().entry_block();
        let c = entry.borrow().arguments()[0] as ValueRef;
        let a = entry.borrow().arguments()[1] as ValueRef;
        {
            let mut builder = test.function_builder();
            let mut arm = |arm: Arm| match arm {
                Arm::Argument => a,
                Arm::Constant(value) => builder.i32(value, SourceSpan::UNKNOWN),
                Arm::Poison => builder.poison(context.i32(), SourceSpan::UNKNOWN),
            };
            let (first, second) = (arm(first), arm(second));
            let cond = cond.map(|value| builder.i1(value, SourceSpan::UNKNOWN)).unwrap_or(c);
            let result = builder.select(cond, first, second, SourceSpan::UNKNOWN).unwrap();
            builder.ret([result], SourceSpan::UNKNOWN).unwrap();
        }
        let function = test.function();
        let pattern: Box<dyn RewritePattern> =
            Box::new(SimplifySelectOfBools::new(context.clone()));
        apply_with_borrowing_listener(&context, function, pattern).unwrap();
        function.borrow().as_operation().recursively_verify().unwrap();
        expected.assert_eq(&format!("{}", function.borrow().as_operation()));
    }

    /// A select with a constant condition and constant arms folds to the chosen arm, materialized
    /// as an `arith.constant`.
    #[test]
    fn select_of_constants_folds_to_an_arith_constant() {
        canonicalize_select(
            Some(true),
            Arm::Constant(7),
            Arm::Constant(9),
            expect![[r#"
            builtin.function public extern("C") @f(%0: i1, %1: i32) -> i32 {
                %6 = arith.constant 7 : i32;
                builtin.ret %6 : (i32);
            };"#]],
        );
        canonicalize_select(
            Some(false),
            Arm::Constant(7),
            Arm::Constant(9),
            expect![[r#"
            builtin.function public extern("C") @f(%0: i1, %1: i32) -> i32 {
                %6 = arith.constant 9 : i32;
                builtin.ret %6 : (i32);
            };"#]],
        );
    }

    /// A select with a poison arm folds to the other arm; when both arms are poison, it folds to
    /// poison, materialized as a `ub.poison`.
    #[test]
    fn select_with_a_poison_arm_folds_to_the_other_arm() {
        canonicalize_select(
            None,
            Arm::Poison,
            Arm::Constant(9),
            expect![[r#"
            builtin.function public extern("C") @f(%0: i1, %1: i32) -> i32 {
                %5 = arith.constant 9 : i32;
                builtin.ret %5 : (i32);
            };"#]],
        );
        canonicalize_select(
            None,
            Arm::Argument,
            Arm::Poison,
            expect![[r#"
            builtin.function public extern("C") @f(%0: i1, %1: i32) -> i32 {
                builtin.ret %1 : (i32);
            };"#]],
        );
        canonicalize_select(
            None,
            Arm::Poison,
            Arm::Poison,
            expect![[r#"
            builtin.function public extern("C") @f(%0: i1, %1: i32) -> i32 {
                %5 = ub.poison <{ value = #ub.poison<i32> }>;
                builtin.ret %5 : (i32);
            };"#]],
        );
        canonicalize_select(
            Some(true),
            Arm::Poison,
            Arm::Argument,
            expect![[r#"
            builtin.function public extern("C") @f(%0: i1, %1: i32) -> i32 {
                builtin.ret %1 : (i32);
            };"#]],
        );
    }

    /// The dialect materializes an integer-like attribute through `arith`, poison through `ub`,
    /// and nothing else.
    #[test]
    fn materialize_constant_delegates_to_arith_and_ub() {
        let mut test = Test::new("f", |_| ([], []));
        let context = test.context_rc();
        let dialect = context.get_or_register_dialect::<ControlFlowDialect>();

        let int = context.int_attr(Int::from(5i32), context.i32()).as_attribute_ref();
        let op = dialect
            .materialize_constant(test.builder_mut(), int, context.i32(), SourceSpan::UNKNOWN)
            .expect("an integer materializes");
        assert!(op.borrow().is::<Constant>());

        let poison = context.create_attribute::<PoisonAttr, _>(context.i32()).as_attribute_ref();
        let op = dialect
            .materialize_constant(test.builder_mut(), poison, context.i64(), SourceSpan::UNKNOWN)
            .expect("poison materializes");
        assert!(op.borrow().is::<Poison>());
        assert_eq!(op.borrow().results()[0].borrow().ty(), context.i64());

        let string = context.create_attribute::<StringAttr, _>("no").as_attribute_ref();
        assert!(
            dialect
                .materialize_constant(
                    test.builder_mut(),
                    string,
                    context.i32(),
                    SourceSpan::UNKNOWN
                )
                .is_none()
        );
        // An integer of another type than the one asked for is not materialized either.
        let int = context.int_attr(Int::from(5i32), context.i32()).as_attribute_ref();
        assert!(
            dialect
                .materialize_constant(test.builder_mut(), int, context.i64(), SourceSpan::UNKNOWN)
                .is_none()
        );
    }
}
