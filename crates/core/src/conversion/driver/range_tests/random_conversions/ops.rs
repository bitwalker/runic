//! The operations the random conversions and the adversarial tests declare.

use super::*;
use crate::{
    OpPrinter,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    effects::MemoryEffectOpInterface,
    traits::{AnyInteger, AnyType, InferTypeOpInterface, Int64, SizedInt},
};

/// 1 when the unsigned sum of two `i64`s does not fit in 64 bits, else 0.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Carry {
    #[operand]
    lhs: Int64,
    #[operand]
    rhs: Int64,
    #[result]
    result: Int64,
}

impl InferTypeOpInterface for Carry {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i64());
        Ok(())
    }
}

/// The wrapping sum and the exclusive or of two integers of one type: two results.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct AddXor {
    #[operand]
    lhs: AnyInteger,
    #[operand]
    rhs: AnyInteger,
    #[result]
    sum: AnyInteger,
    #[result]
    xor: AnyInteger,
}

impl InferTypeOpInterface for AddXor {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.lhs().ty();
        self.sum_mut().set_type(ty);
        self.xor_mut().set_type(ty);
        Ok(())
    }
}

/// A value of the marker type.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Mark {
    #[result]
    result: AnyType,
}

impl InferTypeOpInterface for Mark {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        let ty = context.array_type(context.i8(), 0);
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// `value`, tagged by a marker: the result is `value`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Tag {
    #[operand]
    marker: AnyType,
    #[operand]
    value: AnyInteger,
    #[result]
    result: AnyInteger,
}

impl InferTypeOpInterface for Tag {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.value().ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// An `isize` as an `i64` (the same bits).
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct ToWord {
    #[operand]
    value: AnyInteger,
    #[result]
    result: Int64,
}

impl InferTypeOpInterface for ToWord {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i64());
        Ok(())
    }
}

/// An `i64` as an `isize` (the same bits).
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct ToSize {
    #[operand]
    value: Int64,
    #[result]
    result: AnyInteger,
}

impl InferTypeOpInterface for ToSize {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.isize());
        Ok(())
    }
}

/// What a source materialization builds for two halves: as `test.join`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct MJoin {
    #[operand]
    low: Int64,
    #[operand]
    high: Int64,
    #[result]
    result: SizedInt<128>,
}

impl InferTypeOpInterface for MJoin {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i128());
        Ok(())
    }
}

/// What a target materialization builds for the low half: as `test.low`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct MLow {
    #[operand]
    value: SizedInt<128>,
    #[result]
    result: Int64,
}

impl InferTypeOpInterface for MLow {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i64());
        Ok(())
    }
}

/// What a target materialization builds for the high half: as `test.high`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct MHigh {
    #[operand]
    value: SizedInt<128>,
    #[result]
    result: Int64,
}

impl InferTypeOpInterface for MHigh {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i64());
        Ok(())
    }
}

/// What a source materialization builds for a dropped marker: as `test.mark`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct MMark {
    #[result]
    result: AnyType,
}

impl InferTypeOpInterface for MMark {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        let ty = context.array_type(context.i8(), 0);
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// Each operation builds, prints and verifies.
#[test]
fn the_operations_build_and_verify() {
    use crate::{BuilderExt, Op, SourceSpan, testing::Test};

    let mut test = Test::new("f", |cx| ([cx.i64(), cx.i64(), cx.i128(), cx.isize()], []));
    let args = test.entry_block().borrow().argument_values().collect::<Vec<_>>();
    {
        let mut builder = test.function_builder();
        let b = builder.builder_mut();
        const S: SourceSpan = SourceSpan::UNKNOWN;
        b.create::<Carry, _>(S)(args[0], args[1]).unwrap();
        b.create::<AddXor, _>(S)(args[2], args[2]).unwrap();
        let m = b.create::<Mark, _>(S)().unwrap();
        let m = m.borrow().result().as_value_ref();
        b.create::<Tag, _>(S)(m, args[2]).unwrap();
        b.create::<ToWord, _>(S)(args[3]).unwrap();
        b.create::<ToSize, _>(S)(args[0]).unwrap();
        b.create::<MJoin, _>(S)(args[0], args[1]).unwrap();
        b.create::<MLow, _>(S)(args[2]).unwrap();
        b.create::<MHigh, _>(S)(args[2]).unwrap();
        b.create::<MMark, _>(S)().unwrap();
        b.create::<crate::dialects::builtin::Ret, (Vec<ValueRef>,)>(S)(vec![]).unwrap();
    }
    let text = test.function().borrow().as_operation().to_string();
    for name in [
        "test.carry",
        "test.add_xor",
        "test.mark",
        "test.tag",
        "test.to_word",
        "test.to_size",
        "test.m_join",
        "test.m_low",
        "test.m_high",
        "test.m_mark",
    ] {
        assert!(text.contains(name), "{name} in\n{text}");
    }
    test.function().borrow().as_operation().recursively_verify().unwrap();
}
