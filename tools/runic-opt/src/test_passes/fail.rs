//! `test-fail`: a pass that fails.

use core::fmt;

use runic_core::{
    EntityMut, Operation, OperationName, Report,
    diagnostics::report,
    pass::{Pass, PassExecutionState, PassInfo, PostPassStatus, write_pass_option},
};

/// Fails at the operation it runs on, with the message of its option `message` ("test-fail" when
/// none is given). With `after-change`, it first reports that it changed the IR (it changes
/// nothing), so that what is done after a pass that changed the IR and then failed can be tested.
pub struct Fail {
    message: String,
    after_change: bool,
}

const NAME: &str = "test-fail";
const DEFAULT_MESSAGE: &str = "test-fail";

runic_core::inventory::submit!(PassInfo::new::<Fail>(
    NAME,
    "Fails at the operation it runs on, with the message of its option `message`"
));

impl Default for Fail {
    fn default() -> Self {
        Self {
            message: String::from(DEFAULT_MESSAGE),
            after_change: false,
        }
    }
}

impl Pass for Fail {
    type Target = Operation;

    fn name(&self) -> &'static str {
        NAME
    }

    fn argument(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "Fails at the operation it runs on, with the message of its option `message`"
    }

    fn can_schedule_on(&self, _name: &OperationName) -> bool {
        true
    }

    fn initialize_options(&mut self, options: &str) -> Result<(), Report> {
        for (key, value) in super::options(options) {
            match (key, value) {
                ("message", Some(value)) => self.message = String::from(value),
                ("after-change", value) => {
                    self.after_change = super::switch(NAME, key, value)?;
                }
                _ => return Err(super::unknown(NAME, key, "'message=..' and 'after-change'")),
            }
        }
        Ok(())
    }

    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(NAME)?;
        let message = (self.message != DEFAULT_MESSAGE).then_some(self.message.as_str());
        if message.is_none() && !self.after_change {
            return Ok(());
        }
        f.write_str("{")?;
        if let Some(message) = message {
            write_pass_option(f, "message", Some(message))?;
            if self.after_change {
                f.write_str(" ")?;
            }
        }
        if self.after_change {
            write_pass_option(f, "after-change", None)?;
        }
        f.write_str("}")
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let span = op.span;
        drop(op);
        if self.after_change {
            state.set_post_pass_status(PostPassStatus::Changed);
        }
        Err(report!(message: (self.message.clone()), labels: [primary(span)],))
    }
}
