use runic_core::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    effects::*,
    traits::*,
    *,
};

use crate::*;

/// This operation represents a value produced by undefined behavior, i.e. control reaching a
/// program point that is not supposed to be reachable.
///
/// Any operation performed on a poison value, itself produces poison, and can be folded as such.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = UndefinedBehaviorDialect,
    traits(ConstantLike),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Poison {
    #[attr(hidden)]
    value: PoisonAttr,
    #[result]
    result: AnyType,
}

impl Foldable for Poison {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        results.push(OpFoldResult::Attribute(self.value));
        FoldResult::Ok(())
    }

    fn fold_with(
        &self,
        _operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        results.push(OpFoldResult::Attribute(self.value));
        FoldResult::Ok(())
    }
}

impl InferTypeOpInterface for Poison {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let poison_ty = *self.get_value();
        self.result_mut().set_type(poison_ty);
        Ok(())
    }
}

/// A poison value has the type its attribute names. The generated setters do not maintain this:
/// `set_value` replaces the attribute and leaves the result as it was.
///
/// The check hangs on [InferTypeOpInterface] because [ConstantLike] cannot carry it: the
/// `#[operation_trait]` macro gives every such trait a blanket `Verify` impl that an operation
/// cannot specialize.
impl Verify<dyn InferTypeOpInterface> for Poison {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let attr_ty = *self.get_value();
        let result_ty = self.result().ty();
        if attr_ty == result_ty {
            return Ok(());
        }
        Err(runic_core::diagnostics::report!(
            message: "invalid operation",
            labels: [
                primary(
                    self.as_operation().span(),
                    alloc::format!(
                        "the result of '{}' has type `{result_ty}`, but its value has type \
                         `{attr_ty}`",
                        self.as_operation().name()
                    )
                ),
            ],
        ))
    }
}

/// This operation represents an assertion that a specific program point should never be dynamically
/// reachable.
///
/// The specific way this gets lowered is up to the codegen backend and optimization choices.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = UndefinedBehaviorDialect,
    traits(Terminator),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
#[effects(MemoryEffect(MemoryEffect::Write))]
pub struct Unreachable {}
