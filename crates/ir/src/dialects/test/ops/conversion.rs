//! Operations for the dialect-conversion tests, which split an `i128` into two `i64` halves.

use crate::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::test::TestDialect,
    effects::*,
    traits::*,
    *,
};

/// The `i128` whose low 64 bits are `low` and whose high 64 bits are `high`:
/// `%r = test.join %low, %high;`.
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Join {
    #[operand]
    low: Int64,
    #[operand]
    high: Int64,
    #[result]
    result: SizedInt<128>,
}

impl InferTypeOpInterface for Join {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i128());
        Ok(())
    }
}

/// The low 64 bits of an `i128`: `%r = test.low %value;`.
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Low {
    #[operand]
    value: SizedInt<128>,
    #[result]
    result: Int64,
}

impl InferTypeOpInterface for Low {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i64());
        Ok(())
    }
}

/// The high 64 bits of an `i128`: `%r = test.high %value;`.
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct High {
    #[operand]
    value: SizedInt<128>,
    #[result]
    result: Int64,
}

impl InferTypeOpInterface for High {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i64());
        Ok(())
    }
}

/// Uses a value, then any number of further values, of any types, and gives nothing:
/// `test.sink %first, %rest0, %rest1 : (i32, i64, i64);`. Its operands are in two groups, `first`
/// and `rest`.
///
/// It stands for an operation whose operands come in more than one group, and for a user that
/// does nothing but use its operands.
#[derive(OpParser, OpPrinter, EffectOpInterface)]
#[operation(dialect = TestDialect, implements(MemoryEffectOpInterface, OpPrinter))]
pub struct Sink {
    #[operand]
    first: AnyType,
    #[operands]
    rest: AnyType,
}
