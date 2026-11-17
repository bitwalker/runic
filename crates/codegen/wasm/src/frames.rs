//! The pass `wasm-lower-frames`, which gives every `mem.alloca` a slot in a frame on the shadow
//! stack.

use core::fmt;

use runic_ir::{
    EntityMut, Operation, OperationName, Report,
    pass::{Pass, PassExecutionState, PostPassStatus},
};

/// The name `wasm-lower-frames` is registered under.
pub const NAME: &str = "wasm-lower-frames";

const DESCRIPTION: &str = "Gives every mem.alloca of a function a slot in one frame on the shadow \
                           stack in linear memory (not written yet: changes nothing)";

/// Gives every `mem.alloca` of a function a fixed slot in one frame per function on the shadow
/// stack in linear memory, addressed from `@__stack_pointer`; it is to save the stack pointer on
/// entry and restore it before every return and tail call, keep the frame 16-byte aligned and
/// growing down, and refuse an alloca in a loop.
///
/// The pass is registered so that the lowering pipeline is whole, and it changes nothing yet: it
/// leaves every `mem.alloca` where it is, and `convert-to-wasm` reports one.
#[derive(Default)]
pub struct LowerFrames;

runic_ir::inventory::submit!(runic_ir::pass::PassInfo::new::<LowerFrames>(NAME, DESCRIPTION));

impl Pass for LowerFrames {
    type Target = Operation;

    fn name(&self) -> &'static str {
        NAME
    }

    fn argument(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        DESCRIPTION
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(NAME)
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Self::Target>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        state.set_post_pass_status(PostPassStatus::Unchanged);
        state.preserved_analyses_mut().preserve_all();
        Ok(())
    }
}
