mod convert_trivial_if_to_select;
mod fold_constant_index_switch;
mod fold_redundant_yields;
mod if_remove_unused_results;
mod index_switch_remove_unused_results;
mod nested_exit;
mod remove_loop_invariant_args_from_before_block;
mod while_condition_truth;
mod while_rebuild;
mod while_remove_duplicated_results;
mod while_remove_unused_args;
mod while_unused_result;

pub use self::{
    convert_trivial_if_to_select::ConvertTrivialIfToSelect,
    fold_constant_index_switch::FoldConstantIndexSwitch,
    fold_redundant_yields::FoldRedundantYields, if_remove_unused_results::IfRemoveUnusedResults,
    index_switch_remove_unused_results::IndexSwitchRemoveUnusedResults,
    remove_loop_invariant_args_from_before_block::RemoveLoopInvariantArgsFromBeforeBlock,
    while_condition_truth::WhileConditionTruth,
    while_remove_duplicated_results::WhileRemoveDuplicatedResults,
    while_remove_unused_args::WhileRemoveUnusedArgs, while_unused_result::WhileUnusedResult,
};

/// Strips trailing whitespace from every line and ends the text with a single newline: the printed
/// IR the tests of the canonicalization patterns compare.
#[cfg(test)]
fn normalize_ir(input: &str) -> alloc::string::String {
    use alloc::vec::Vec;

    let mut normalized = input.lines().map(str::trim_end).collect::<Vec<_>>().join("\n");
    normalized.push('\n');
    normalized
}
