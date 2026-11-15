#![no_std]
#![deny(warnings)]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod canonicalization;
mod cfg_to_scf;
mod cse;
mod dce;
mod local2reg;
//mod inliner;
mod sccp;
mod sink;
#[cfg(test)]
mod tests;

pub use self::dce::DeadCodeElimination;
//pub use self::inliner::Inliner;
pub use self::{
    canonicalization::Canonicalizer,
    cfg_to_scf::{CFGToSCFInterface, check_all_blocks_reachable, transform_cfg_to_scf},
    cse::CommonSubexpressionElimination,
    local2reg::Local2Reg,
    sccp::SparseConditionalConstantPropagation,
    sink::{ControlFlowSink, SinkOperandDefs},
};
