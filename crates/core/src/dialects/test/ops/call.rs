use crate::{
    derive::{OpParser, OpPrinter, operation},
    dialects::{builtin::attributes::BoolAttr, test::TestDialect},
    effects::*,
    traits::*,
    *,
};

/// An `i32` whose memory effects are a `Read` when `known` is true, and unknown otherwise:
/// `%r = test.effects <{ known = #builtin.bool<true> }>;`.
///
/// It stands for an operation that implements [MemoryEffectOpInterface] and yet cannot always list
/// its effects, as a call to a callee that states none.
#[derive(OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Effects {
    #[attr]
    known: BoolAttr,
    #[result]
    result: Int32,
}

impl InferTypeOpInterface for Effects {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        let i32 = context.i32();
        self.result_mut().set_type(i32);
        Ok(())
    }
}

impl EffectOpInterface<MemoryEffect> for Effects {
    fn effects(&self) -> EffectIterator<MemoryEffect> {
        if *self.get_known() {
            EffectIterator::new([EffectInstance::new(MemoryEffect::Read)])
        } else {
            EffectIterator::new([])
        }
    }

    fn effects_are_known(&self) -> bool {
        *self.get_known()
    }
}
