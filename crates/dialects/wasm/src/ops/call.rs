//! The indirect calls: `call_indirect` and `return_call_indirect`.
//!
//! Direct calls are `builtin.call` and `builtin.tail_call`, which the dialect does not repeat. An
//! indirect call goes through a table of `funcref` and an `i32` index into it. Like
//! `builtin.call_indirect`, it has unknown effects: it implements no effect interface, so it is
//! never removed, merged or moved, and no effect-aware pass moves anything across it.

use alloc::{format, string::String};

use runic_ir::{
    AttrPrinter, Context, Op, OpAsmParser, OpParser, OpPrinter, Operation, OperationState,
    ParseResult, Report, SmallVec, Spanned, TailCallOpInterface, Type, Value, Verify,
    attributes::AttrParser,
    derive::operation,
    dialects::builtin::attributes::{Signature, SignatureAttr},
    formatter::const_text,
    parse::{Delimiter, ParserError, UnresolvedOperand},
    print::AsmPrinter,
    traits::{AnyType, InferTypeOpInterface, TailCallLike, Terminator, TypeConstraint},
};

use super::text::*;
use crate::{
    FuncRefType, Table, ValType, WasmDialect, is_val_type, referenced_entity, val_type_name,
};

/// The hook the verifiers of the two indirect calls hang from: the rules need the operations'
/// signatures, which no trait that exists carries.
trait IndirectCall: Op {}

/// A reference to a [CallIndirect].
pub type CallIndirectRef = runic_ir::UnsafeIntrusiveEntityRef<CallIndirect>;

/// A reference to a [ReturnCallIndirect].
pub type ReturnCallIndirectRef = runic_ir::UnsafeIntrusiveEntityRef<ReturnCallIndirect>;

const CALL_RULE: &str = "calls a function through a `funcref` table and an `i32` index, passes \
                         the arguments of its signature and gives its results, all of Wasm value \
                         types";

/// Checks what the two indirect calls share: the table is a `wasm.table` of `funcref`, the index
/// is an `i32`, the signature has Wasm value types, and the arguments agree with it.
fn verify_indirect_call(
    op: &Operation,
    table: &runic_ir::dialects::builtin::attributes::SymbolRefAttr,
    signature: &Signature,
) -> Result<(), Report> {
    let referenced = referenced_entity::<Table>(op, table, "wasm.table")?;
    let element = *referenced.borrow().get_element_type();
    if element != FuncRefType::get(op.context()) {
        return Err(rule_error(
            op,
            CALL_RULE,
            format!(
                "the table `{}` holds `{}`, and an indirect call goes through a table of `funcref`",
                table.path(),
                val_type_name(element)
            ),
        ));
    }
    expect_operand(op, CALL_RULE, 0, "its index", op.context().i32())?;
    for (kind, params) in
        [("parameter", signature.params()), ("result", signature.declared_results())]
    {
        for (index, param) in params.iter().enumerate() {
            if !is_val_type(param.ty) {
                return Err(rule_error(
                    op,
                    CALL_RULE,
                    format!(
                        "{kind} {index} of its signature has type `{}`, and a Wasm value type is \
                         {}",
                        val_type_name(param.ty),
                        ValType::get().description()
                    ),
                ));
            }
        }
    }
    let arguments = op.operands().group(1);
    if arguments.len() != signature.params().len() {
        return Err(rule_error(
            op,
            CALL_RULE,
            format!(
                "it passes {} argument(s) and its signature has {} parameter(s)",
                arguments.len(),
                signature.params().len()
            ),
        ));
    }
    for (index, (argument, param)) in arguments.iter().zip(signature.params()).enumerate() {
        let found = argument.borrow().ty();
        if found != param.ty {
            return Err(rule_error(
                op,
                CALL_RULE,
                format!(
                    "argument {index} has type `{}` and parameter {index} of its signature has \
                     type `{}`",
                    val_type_name(found),
                    val_type_name(param.ty)
                ),
            ));
        }
    }
    Ok(())
}

/// Prints `@t, %i(%a, ..) : signature`, then the discardable attributes.
fn print_indirect_call(
    printer: &mut AsmPrinter<'_>,
    op: &Operation,
    table: &runic_ir::dialects::builtin::attributes::SymbolRefAttr,
    signature: &SignatureAttr,
) {
    use runic_ir::AsValueRange;

    printer.print_space();
    printer.print_symbol_path(table.path());
    *printer += const_text(", ");
    printer.print_value_uses(op.operands().group(0).as_value_range());
    printer.print_operand_list(op.operands().group(1));
    *printer += const_text(" : ");
    signature.print(printer);
    print_attributes(printer, op);
}

/// Reads `@t, %i(%a, ..) : signature`, then an optional `attributes {..}`. The index is an `i32`,
/// and the arguments take the types of the parameters of the signature.
fn parse_indirect_call(
    state: &mut OperationState,
    parser: &mut dyn OpAsmParser<'_>,
) -> ParseResult {
    let table = parser.parse_symbol_ref()?.into_inner();
    state.add_attribute("table", table);
    parser.parse_comma()?;
    let index = parser.parse_operand(false)?;
    let mut arguments = SmallVec::<[UnresolvedOperand; 2]>::default();
    parser.parse_operand_list(&mut arguments, Delimiter::Paren, true, None)?;
    parser.parse_colon()?;

    let start = parser.current_location();
    let parsed = <SignatureAttr as AttrParser>::parse(parser)?;
    let parsed = parsed.borrow();
    let Some(signature) = parsed.downcast_ref::<SignatureAttr>() else {
        return Err(ParserError::InvalidAttributeValue {
            span: start,
            reason: String::from("expected a signature"),
        });
    };
    let signature = Signature::clone(signature);
    let params = signature.params().iter().map(|param| param.ty).collect::<SmallVec<[Type; 2]>>();
    state.add_attribute(
        "signature",
        parser.context_rc().create_attribute::<SignatureAttr, _>(signature),
    );
    parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

    let i32 = parser.context().i32();
    let index = parser.resolve_operand(index, i32)?;
    state.operands.push(SmallVec::from_iter([index]));
    let mut values = SmallVec::default();
    parser.resolve_operands(state.span, &arguments, &params, &mut values)?;
    state.operands.push(values);
    Ok(())
}

/// An indirect call: `%r = wasm.call_indirect @t, %i(%a, %b) : extern("C") (i32, i32) -> i32;`.
/// `call_indirect`.
///
/// It calls the function in entry `%i` (an `i32`) of the table `@t`, a `wasm.table` of `funcref`,
/// with the arguments and the signature written: one argument per parameter and one result per
/// result, each of a Wasm value type (the signature's calling convention is carried and is not
/// otherwise looked at). It traps when the entry is null, when the index is out of bounds, and
/// when the function's type is not the signature (the checks of Wasm's `call_indirect`, which no
/// verifier can make: a trap is not an effect).
///
/// The set of functions the call may reach is not known, and neither are its effects: it
/// implements no effect interface, so it is never removed, merged or moved, as
/// `builtin.call_indirect`. It is not a `builtin` call operation either: its callee is a table
/// and an index, not a value or a symbol.
#[operation(
    dialect = WasmDialect,
    implements(IndirectCall, InferTypeOpInterface, OpPrinter)
)]
pub struct CallIndirect {
    /// The table called through
    #[symbol]
    table: Table,
    /// The index into the table, an `i32`
    #[operand]
    index: AnyType,
    /// The arguments, one per parameter of the signature
    #[operands]
    arguments: AnyType,
    /// The signature the call is made with
    #[attr]
    signature: SignatureAttr,
    /// The results, one per result of the signature
    #[results]
    returns: AnyType,
}

impl IndirectCall for CallIndirect {}

/// One result per result of the signature, in order. A call that already has results (the generic
/// form gives them) keeps them, and the verifier checks them against the signature.
impl InferTypeOpInterface for CallIndirect {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        if self.op.num_results() != 0 {
            return Ok(());
        }
        let types = self
            .get_signature()
            .results()
            .iter()
            .map(|result| result.ty)
            .collect::<SmallVec<[Type; 2]>>();
        let span = self.op.span();
        let owner = self.op.as_operation_ref();
        for (index, ty) in types.into_iter().enumerate() {
            let result = context.make_result(span, ty, owner, index as u16);
            self.op.results.push(result);
        }
        Ok(())
    }
}

impl OpPrinter for CallIndirect {
    /// The arguments take the types of the parameters of the written signature.
    fn states_operand_types(&self) -> bool {
        true
    }

    fn print(&self, printer: &mut AsmPrinter<'_>) {
        print_indirect_call(printer, self.as_operation(), &self.table(), &self.signature());
    }
}

impl OpParser for CallIndirect {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_indirect_call(state, parser)
    }
}

/// The table is a `wasm.table` of `funcref`, the index is an `i32`, and the arguments and results
/// agree with a signature of Wasm value types.
impl Verify<dyn IndirectCall> for CallIndirect {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        verify_indirect_call(op, &self.table(), &self.get_signature())?;
        let results = self.get_signature().results().to_vec();
        if op.num_results() != results.len() {
            return Err(rule_error(
                op,
                CALL_RULE,
                format!(
                    "it has {} result(s) and its signature has {}",
                    op.num_results(),
                    results.len()
                ),
            ));
        }
        for (index, (result, expected)) in op.results().iter().zip(&results).enumerate() {
            let found = result.borrow().ty();
            if found != expected.ty {
                return Err(rule_error(
                    op,
                    CALL_RULE,
                    format!(
                        "result {index} has type `{}` and result {index} of its signature has \
                         type `{}`",
                        val_type_name(found),
                        val_type_name(expected.ty)
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// An indirect tail call: `wasm.return_call_indirect @t, %i(%a) : extern("C") (i32) -> i32;`.
/// `return_call_indirect`.
///
/// It is [CallIndirect] as a terminator: the enclosing function leaves by calling the function in
/// entry `%i` of the table `@t`, and returns whatever that function returns, so the operation has
/// no results of its own. It obeys the rule `builtin.tail_call` obeys: the declared results of its
/// signature are the results of the enclosing `builtin.function`, and its calling convention is
/// the function's. The function verifier applies the rule about the operations an exit may
/// cross (only those that forward exits) to it, since it is [TailCallLike], and the rule about the
/// results and the calling convention, through [TailCallOpInterface], which it implements.
///
/// Its effects are unknown, as those of [CallIndirect] are.
#[operation(
    dialect = WasmDialect,
    name = "return_call_indirect",
    traits(Terminator, TailCallLike),
    implements(IndirectCall, TailCallOpInterface, OpPrinter)
)]
pub struct ReturnCallIndirect {
    /// The table called through
    #[symbol]
    table: Table,
    /// The index into the table, an `i32`
    #[operand]
    index: AnyType,
    /// The arguments, one per parameter of the signature
    #[operands]
    arguments: AnyType,
    /// The signature the call is made with
    #[attr]
    signature: SignatureAttr,
}

impl IndirectCall for ReturnCallIndirect {}

impl OpPrinter for ReturnCallIndirect {
    /// The arguments take the types of the parameters of the written signature.
    fn states_operand_types(&self) -> bool {
        true
    }

    fn print(&self, printer: &mut AsmPrinter<'_>) {
        print_indirect_call(printer, self.as_operation(), &self.table(), &self.signature());
    }
}

impl OpParser for ReturnCallIndirect {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_indirect_call(state, parser)
    }
}

/// The checks of [CallIndirect]. That the declared results and the calling convention of the
/// signature are those of the enclosing function is the function verifier's, which finds this
/// operation through [TailCallOpInterface].
impl Verify<dyn IndirectCall> for ReturnCallIndirect {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        verify_indirect_call(self.as_operation(), &self.table(), &self.get_signature())?;
        runic_ir::verify_tail_call_is_in_function(self.as_operation())
    }
}

/// The signature the call is made with, which the function verifier holds to the function's
/// declared results and calling convention.
impl TailCallOpInterface for ReturnCallIndirect {
    fn tail_signature(&self) -> Signature {
        self.get_signature().clone()
    }
}
