//! The pass pipeline over a converted module: every pass of the crate, run on the module.

use alloc::{boxed::Box, rc::Rc, vec::Vec};

use runic_core::{
    Context, OperationRef, Report,
    dialects::builtin::{Function, Module},
    pass::{Nesting, OperationPass, PassManager},
};

use super::Rng;

type MakePass = fn() -> Box<dyn OperationPass>;

/// Every pass of the crate, `cfg-to-scf` last and the others in a random order.
pub(super) fn shuffled(rng: &mut Rng) -> Vec<(&'static str, MakePass)> {
    let mut order = crate::tests::drive::PASSES.to_vec();
    let lift = order.iter().position(|(name, _)| *name == "cfg-to-scf").unwrap();
    let lift = order.remove(lift);
    for index in (1..order.len()).rev() {
        order.swap(index, rng.below(index + 1));
    }
    order.push(lift);
    order
}

/// Runs `pass` over `module` on a pass manager anchored on the module, verifying after it.
pub(super) fn run_pass(
    context: &Rc<Context>,
    module: OperationRef,
    pass: Box<dyn OperationPass>,
) -> Result<(), Report> {
    let mut manager = PassManager::on::<Module>(context.clone(), Nesting::Implicit);
    // A pass whose target is `Function` is not nested implicitly (no concrete operation type
    // states its name as a pass target), and panics when it is run on the module: it is nested
    // by hand (see `probes::wanted_a_function_pass_on_a_module_is_nested_implicitly`).
    if pass.name() == "local2reg" {
        manager.add_nested_pass::<Function>(pass);
    } else {
        manager.add_pass(pass);
    }
    manager.enable_verifier(true);
    manager.run(module)
}
