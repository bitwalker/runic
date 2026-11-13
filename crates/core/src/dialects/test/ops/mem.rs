use crate::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::test::*,
    effects::*,
    traits::*,
    *,
};

/// The memory `test.load` reads and `test.store` writes: a child of the default resource.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TestMemoryResource;

impl Resource for TestMemoryResource {
    fn name(&self) -> &'static str {
        "test.memory"
    }
}

/// Store `value` on the heap at `addr`, a write of [TestMemoryResource]
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct Store {
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(TestMemoryResource))]
    addr: AnyPointer,
    #[operand]
    value: AnyType,
}

/// Load `result` from the heap at `addr`, a read of [TestMemoryResource]
///
/// The type of load is determined by the pointer operand type - cast the pointer to the type you
/// wish to load, so long as such a load is safe according to the semantics of your high-level
/// language.
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Load {
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Read), resource(TestMemoryResource))]
    addr: AnyPointer,
    #[result]
    result: AnyType,
}

impl InferTypeOpInterface for Load {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let _span = self.span();
        let pointee = {
            let addr = self.addr();
            let addr_value = addr.value();
            addr_value.ty().pointee()
        };
        match pointee {
            Some(pointee) => {
                self.result_mut().set_type(pointee);
                Ok(())
            }
            None => {
                // let addr = self.addr();
                // let addr_value = addr.value();
                // let addr_ty = addr_value.ty();
                // Err(context
                //     .session
                //     .diagnostics
                //     .diagnostic(runic_session::diagnostics::Severity::Error)
                //     .with_message("invalid operand for 'load'")
                //     .with_primary_label(
                //         span,
                //         format!("invalid 'addr' operand, expected pointer, got '{addr_ty}'"),
                //     )
                //     .into_report())
                Ok(())
            }
        }
    }
}

/// An operation that names a local variable of the enclosing function in its `local` property and
/// frees it: `test.touch_local <{ local = #builtin.local_variable<0, i32> }>;`.
///
/// It stands for an operation, other than a load or store of the local, whose effect reaches the
/// local: a pass that promotes locals must leave this one alone.
#[derive(OpPrinter, OpParser, EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct TouchLocal {
    #[attr]
    #[effects(
        MemoryEffect(MemoryEffect::Free),
        resource(crate::dialects::builtin::LocalsResource)
    )]
    local: crate::dialects::builtin::attributes::LocalVariableAttr,
}
