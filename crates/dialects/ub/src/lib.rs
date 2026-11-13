#![no_std]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![allow(incomplete_features)]
#![deny(warnings)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod attributes;
mod builders;
mod ops;

use runic_core::{
    AttributeRef, Builder, BuilderExt, Dialect, DialectInfo, OperationRef, SourceSpan, Type,
    derive::DialectRegistration,
};

pub use self::{attributes::PoisonAttr, builders::UndefinedBehaviorOpBuilder, ops::*};

#[derive(Debug, DialectRegistration)]
#[dialect(name = "ub")]
pub struct UndefinedBehaviorDialect {
    info: DialectInfo,
}

impl From<DialectInfo> for UndefinedBehaviorDialect {
    fn from(info: DialectInfo) -> Self {
        Self { info }
    }
}

impl Dialect for UndefinedBehaviorDialect {
    #[inline]
    fn info(&self) -> &DialectInfo {
        &self.info
    }

    fn materialize_constant(
        &self,
        builder: &mut dyn Builder,
        attr: AttributeRef,
        ty: Type,
        span: SourceSpan,
    ) -> Option<OperationRef> {
        // Poison of any type is the same value, so the constant takes the requested type.
        attr.try_downcast_attr::<PoisonAttr>().ok()?;
        let op_builder = builder.create::<Poison, _>(span);
        op_builder(ty).ok().map(|op| op.as_operation_ref())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        format,
        rc::Rc,
        string::{String, ToString},
    };

    use runic_core::{
        Context, Int, Op, OpFoldResult, OperationRef, SmallVec, SourceSpan, Type, Value,
        dialects::builtin::BuiltinOpBuilder,
        effects::MemoryEffect,
        print::AsmPrinter,
        testing::{Test, parse_function_fixpoint},
        traits::{ConstantLike, Foldable, Terminator},
    };

    use super::*;

    type TypeFn = fn(&Context) -> Type;

    /// The first operation of the entry block of `test`'s function.
    fn first_op(test: &Test) -> OperationRef {
        test.entry_block()
            .borrow()
            .body()
            .front()
            .as_pointer()
            .expect("a first operation")
    }

    /// Builds `f`, which returns poison of the type `ty` makes, verifies it and returns its text.
    fn build(ty: TypeFn) -> (Test, String) {
        let mut test = Test::new("f", |cx| ([], [ty(cx)]));
        let context = test.context_rc();
        context.get_or_register_dialect::<UndefinedBehaviorDialect>();
        let poison_ty = ty(&context);
        {
            let mut builder = test.function_builder();
            let poison = builder.poison(poison_ty, SourceSpan::UNKNOWN);
            assert_eq!(poison.borrow().ty(), poison_ty);
            builder.ret([poison], SourceSpan::UNKNOWN).unwrap();
        }
        let function = test.function();
        function.borrow().as_operation().verify().expect("valid function");
        let text = format!("{}", function.borrow().as_operation());
        (test, text)
    }

    #[test]
    fn poison_round_trips() {
        let types: [(TypeFn, &str); 2] =
            [(Context::i32, "i32"), (|cx| cx.ptr_type(cx.i32()), "ptr<i32>")];
        for (ty, spelling) in types {
            let (_test, text) = build(ty);
            assert!(text.contains("ub.poison"), "{text}");
            assert!(text.contains(spelling), "{text}");

            // A fresh context numbers values from zero, as the printed text does.
            let context = Rc::new(Context::default());
            context.get_or_register_dialect::<UndefinedBehaviorDialect>();
            let (function, printed) =
                parse_function_fixpoint(&context, "f", &text).expect("parses");
            assert_eq!(printed, text);
            let op = function
                .borrow()
                .body()
                .entry()
                .body()
                .front()
                .as_pointer()
                .expect("a first operation");
            let op = op.borrow();
            let poison = op.downcast_ref::<Poison>().expect("a ub.poison");
            assert_eq!(*poison.get_value(), ty(&context));
            assert_eq!(poison.result().ty(), ty(&context));
        }
    }

    #[test]
    fn poison_result_is_the_attribute_type() {
        let (test, _) = build(|cx| cx.ptr_type(cx.i32()));
        let context = test.context_rc();
        let op = first_op(&test);
        let op = op.borrow();
        let poison = op.downcast_ref::<Poison>().unwrap();
        let expected = context.ptr_type(context.i32());
        assert_eq!(*poison.get_value(), expected);
        assert_eq!(poison.result().ty(), expected);
        assert!(op.implements::<dyn ConstantLike>());
    }

    #[test]
    fn poison_folds_and_materializes() {
        let (mut test, _) = build(Context::i32);
        let context = test.context_rc();
        let dialect = context.get_or_register_dialect::<UndefinedBehaviorDialect>();

        let attr = {
            let op = first_op(&test);
            let op = op.borrow();
            let poison = op.downcast_ref::<Poison>().unwrap();
            let mut results = SmallVec::<[OpFoldResult; 1]>::new();
            assert!(poison.fold(&mut results).is_ok());
            assert_eq!(results.len(), 1);
            match results.pop().unwrap() {
                OpFoldResult::Attribute(attr) => attr,
                _ => panic!("poison folds to its attribute"),
            }
        };
        assert_eq!(*attr.borrow().downcast_ref::<PoisonAttr>().unwrap().as_value(), context.i32());

        let ty = context.i64();
        let op = dialect
            .materialize_constant(test.builder_mut(), attr, ty, SourceSpan::UNKNOWN)
            .expect("materialized");
        let op = op.borrow();
        let poison = op.downcast_ref::<Poison>().expect("a ub.poison");
        assert_eq!(*poison.get_value(), ty);
        assert_eq!(poison.result().ty(), ty);

        let other = context.int_attr(Int::from(1i32), context.i32()).as_attribute_ref();
        assert!(
            dialect
                .materialize_constant(test.builder_mut(), other, context.i32(), SourceSpan::UNKNOWN)
                .is_none()
        );
    }

    #[test]
    fn unreachable_is_a_terminator_that_writes() {
        let mut test = Test::new("f", |_| ([], []));
        test.context().get_or_register_dialect::<UndefinedBehaviorDialect>();
        let op = test.function_builder().unreachable(SourceSpan::UNKNOWN);
        {
            let op = op.borrow();
            assert!(op.as_operation().implements::<dyn Terminator>());
            assert!(op.as_operation().has_memory_effect(MemoryEffect::Write));
        }
        let function = test.function();
        function.borrow().as_operation().verify().expect("valid function");
        let text = format!("{}", function.borrow().as_operation());
        assert!(text.contains("ub.unreachable"), "{text}");
        let (_, printed) = parse_function_fixpoint(&test.context_rc(), "f", &text).expect("parses");
        assert_eq!(printed, text);
    }

    #[test]
    fn poison_attribute_prints_its_type() {
        let context = Rc::new(Context::default());
        context.get_or_register_dialect::<UndefinedBehaviorDialect>();
        let attr = context.create_attribute::<PoisonAttr, _>(context.i32());
        let flags = Default::default();
        let mut printer = AsmPrinter::root(context.clone(), &flags);
        printer.print_attribute_value(&*attr.borrow());
        assert_eq!(printer.finish().to_string(), "#ub.poison<i32>");
    }
}
