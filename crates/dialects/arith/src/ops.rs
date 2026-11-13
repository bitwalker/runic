mod binary;
mod coercions;
mod compare;
mod constants;
mod unary;

pub use self::{binary::*, coercions::*, compare::*, constants::*, unary::*};

/// Checks that the result `name` of `op`, of type `ty`, is an `i1`.
///
/// A result's type constraint is checked when the operation is built, not when it is verified, so
/// an operation whose `i1` result (a comparison, an overflow flag) must stay one checks it here.
pub(crate) fn verify_bool_result(
    op: &runic_core::Operation,
    name: &str,
    ty: runic_core::Type,
    context: &runic_core::Context,
) -> Result<(), runic_core::Report> {
    use runic_core::Spanned;

    if ty == context.i1() {
        return Ok(());
    }
    Err(runic_core::diagnostics::report!(
        message: "invalid operation",
        labels: [
            primary(
                op.span(),
                alloc::format!("the result `{name}` of '{}' has type `{ty}`, not `i1`", op.name())
            ),
        ],
    ))
}
