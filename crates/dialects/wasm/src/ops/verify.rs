//! The checks the numeric operations share, and their messages.
//!
//! The operands and results of a numeric operation are declared with no type constraint of their
//! own: the check is made here, so that the message names the Wasm rule (what the instruction
//! takes) and says what the user wrote, where a constraint's message would only name the type
//! that was expected.

use alloc::{format, string::String};

use runic_ir::{
    Context, Operation, Report, Type, Value, diagnostics::Spanned, traits::TypeConstraint,
};

use crate::val_type_name;

/// What the result of an operation that takes numbers of one type is.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum ResultRule {
    /// A value of the type of the operands.
    SameType,
    /// An `i32`: the result of a comparison.
    I32,
}

/// The report of `op` being invalid because of `problem`, in terms of what it takes (`rule`):
/// "`wasm.shl` takes two operands of one type, `i32` or `i64`, but ...".
fn invalid(op: &Operation, rule: &str, problem: String) -> Report {
    runic_ir::diagnostics::report!(
        message: "invalid operation",
        labels: [
            primary(op.span(), format!("`{}` {rule}, but {problem}", op.name())),
        ],
    )
}

/// The type of operand `index` of `op`.
fn operand_type(op: &Operation, index: usize) -> Type {
    op.operands()[index].borrow().as_value_ref().borrow().ty()
}

/// The type of the one result of `op`.
fn result_type(op: &Operation) -> Type {
    op.results()[0].borrow().ty()
}

/// Checks that the operands of `op`, one or two, have one type that `C` accepts, and that the
/// result is what `result` says.
pub(crate) fn verify_uniform<C: TypeConstraint>(
    op: &Operation,
    result: ResultRule,
) -> Result<(), Report> {
    let description = C::get().description();
    let count = op.num_operands();
    let first = operand_type(op, 0);
    let rule = match (count, result) {
        (1, ResultRule::SameType) => {
            format!("takes one operand, {description}, and gives a result of its type")
        }
        (1, ResultRule::I32) => format!("takes one operand, {description}, and gives an `i32`"),
        (_, ResultRule::SameType) => format!(
            "takes two operands of one type, {description}, and gives a result of that type"
        ),
        (_, ResultRule::I32) => {
            format!("takes two operands of one type, {description}, and gives an `i32`")
        }
    };
    if !C::get().matches(first) {
        let which = if count == 1 {
            "its operand"
        } else {
            "its first operand"
        };
        return Err(invalid(op, &rule, format!("{which} has type `{}`", val_type_name(first))));
    }
    if count == 2 {
        let second = operand_type(op, 1);
        if second != first {
            let problem = if C::get().matches(second) {
                format!(
                    "its operands have the types `{}` and `{}`",
                    val_type_name(first),
                    val_type_name(second)
                )
            } else {
                format!("its second operand has type `{}`", val_type_name(second))
            };
            return Err(invalid(op, &rule, problem));
        }
    }
    let actual = result_type(op);
    let expected = match result {
        ResultRule::SameType => first,
        ResultRule::I32 => op.context().i32(),
    };
    if actual != expected {
        return Err(invalid(
            op,
            &rule,
            format!(
                "its result has type `{}`, not `{}`",
                val_type_name(actual),
                val_type_name(expected)
            ),
        ));
    }
    Ok(())
}

/// Checks the conversion `op`: its operand matches `From`, its result is the type `ty` it names
/// and matches `To`, and the pair of the two is one that `rule` accepts (`pairs` says which in
/// words).
pub(crate) fn verify_conversion<From: TypeConstraint, To: TypeConstraint>(
    op: &Operation,
    context: &Context,
    ty: Type,
    rule: fn(&Context, Type, Type) -> bool,
    pairs: &str,
) -> Result<(), Report> {
    let from = operand_type(op, 0);
    let to = result_type(op);
    let shape = format!("converts {pairs}");
    if !From::get().matches(from) {
        return Err(invalid(op, &shape, format!("its operand has type `{}`", val_type_name(from))));
    }
    if to != ty {
        return Err(invalid(
            op,
            &shape,
            format!(
                "its result has type `{}`, not its `ty` property `{}`",
                val_type_name(to),
                val_type_name(ty)
            ),
        ));
    }
    if !To::get().matches(to) || !rule(context, from, to) {
        return Err(invalid(
            op,
            &shape,
            format!("`{}` to `{}` is not one of them", val_type_name(from), val_type_name(to)),
        ));
    }
    Ok(())
}

/// Checks the constant `op`, whose value has the type `value`: `C` accepts it, and the result has
/// it.
pub(crate) fn verify_constant<C: TypeConstraint>(
    op: &Operation,
    value: Type,
) -> Result<(), Report> {
    let rule = format!(
        "takes a value of type {} and gives a result of that type",
        C::get().description()
    );
    if !C::get().matches(value) {
        return Err(invalid(op, &rule, format!("its value has type `{}`", val_type_name(value))));
    }
    let result = result_type(op);
    if result != value {
        return Err(invalid(
            op,
            &rule,
            format!(
                "its result has type `{}`, not `{}`",
                val_type_name(result),
                val_type_name(value)
            ),
        ));
    }
    Ok(())
}

/// Checks `select`: the condition is an `i32`, the two values have one Wasm value type, and the
/// result has it.
pub(crate) fn verify_select(op: &Operation) -> Result<(), Report> {
    use crate::{ValType, is_val_type};

    let rule = format!(
        "takes an `i32` condition and two values of one type, {}, and gives a result of that type",
        ValType::get().description()
    );
    let condition = operand_type(op, 0);
    if condition != op.context().i32() {
        return Err(invalid(
            op,
            &rule,
            format!("its condition has type `{}`", val_type_name(condition)),
        ));
    }
    let (first, second) = (operand_type(op, 1), operand_type(op, 2));
    if !is_val_type(first) {
        return Err(invalid(
            op,
            &rule,
            format!("its first value has type `{}`", val_type_name(first)),
        ));
    }
    if second != first {
        let problem = if is_val_type(second) {
            format!(
                "its values have the types `{}` and `{}`",
                val_type_name(first),
                val_type_name(second)
            )
        } else {
            format!("its second value has type `{}`", val_type_name(second))
        };
        return Err(invalid(op, &rule, problem));
    }
    let result = result_type(op);
    if result != first {
        return Err(invalid(
            op,
            &rule,
            format!(
                "its result has type `{}`, not `{}`",
                val_type_name(result),
                val_type_name(first)
            ),
        ));
    }
    Ok(())
}
