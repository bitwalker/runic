//! `test-panic`: a pass that panics, for testing how the tool reports a defect of its own. It is
//! registered like any pass but not listed by `--list-passes`.

use core::fmt;

use runic_core::{
    EntityMut, Operation, OperationName, Report,
    pass::{Pass, PassExecutionState, PassInfo, PassOptions, write_pass_option},
};

/// Panics at the operation it runs on. With `cannot-unwind`, it panics where a panic cannot
/// unwind, so that the process aborts.
#[derive(Default)]
pub struct Panic {
    cannot_unwind: bool,
}

pub const NAME: &str = "test-panic";
const DESCRIPTION: &str = "Panics at the operation it runs on";
/// The message of the panic.
pub const MESSAGE: &str = "the test-panic pass panicked";

runic_core::inventory::submit!(PassInfo::new::<Panic>(NAME, DESCRIPTION));

impl Pass for Panic {
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

    fn initialize_options(&mut self, options: &PassOptions) -> Result<(), Report> {
        options.check_keys(NAME, &["cannot-unwind"])?;
        if let Some(value) = options.get("cannot-unwind") {
            self.cannot_unwind = super::switch(NAME, "cannot-unwind", value)?;
        }
        Ok(())
    }

    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(NAME)?;
        if self.cannot_unwind {
            f.write_str("{")?;
            write_pass_option(f, "cannot-unwind", None)?;
            f.write_str("}")?;
        }
        Ok(())
    }

    fn run_on_operation(
        &mut self,
        _op: EntityMut<'_, Operation>,
        _state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        if self.cannot_unwind {
            cannot_unwind();
        }
        panic!("{MESSAGE}")
    }
}

/// Panics in a function that cannot unwind: the panic aborts the process.
extern "C" fn cannot_unwind() {
    panic!("{MESSAGE}")
}
