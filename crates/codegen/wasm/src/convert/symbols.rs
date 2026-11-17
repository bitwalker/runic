//! The patterns of addresses and function values: `builtin.global_symbol`,
//! `builtin.function_ref`, `builtin.call_indirect` and `builtin.tail_call_indirect`.
//!
//! * `builtin.global_symbol @g` becomes `wasm.data_addr @g + 0`, the address of the global
//!   variable in linear memory.
//! * A value of a function type is carried in an `i32`, the index of the function in the
//!   module's indirect-function table `@__indirect_function_table`: `builtin.function_ref @f`
//!   becomes `wasm.func_index @f`, by the path it was written with.
//! * `builtin.call_indirect %f(args) : sig` becomes `wasm.call_indirect
//!   @__indirect_function_table, %f(args) : sig'`, and `builtin.tail_call_indirect` becomes
//!   `wasm.return_call_indirect`, where `sig'` is the converted signature
//!   ([convert_signature]). The rules of the calling conventions are those of a call and a tail
//!   call, through the wrapper of `calls` that adds them.
//!
//! A `builtin.unrealized_conversion_cast` that `wasm-lower-frames` leaves, from the `i32` address
//! of a slot to the pointer type of its `mem.alloca`, needs no pattern: a cast that was in the IR
//! before the conversion is not offered to the patterns, and once its result's users are
//! converted it is the first link of a chain of casts from `i32` back to `i32`, which the driver
//! folds away.

use alloc::{format, rc::Rc, vec::Vec};

use runic_dialect_wasm as wasm;
use runic_ir::{
    Int, Op, OperationRef, Report, Spanned, ValueRef,
    conversion::{
        ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConvertedOperands,
        TypeConverter, convert_signature,
    },
    dialects::builtin::{
        CallIndirect, FunctionReference, GlobalSymbol, GlobalVariableRef, TailCallIndirect,
        attributes::Signature,
    },
    patterns::{Pattern, PatternInfo},
};

use crate::{
    convert::{
        WellKnownSymbols,
        calls::{self, Role, pattern_info, registered},
    },
    symbols::{INDIRECT_FUNCTION_TABLE, local_path},
};

/// Adds the patterns of this module. `symbols` holds the indirect-function table of each module
/// whose functions make an indirect call, found or declared before the conversion starts; a
/// pattern that needs one its module has not got is an error.
pub fn populate(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
    symbols: &Rc<WellKnownSymbols>,
) -> Result<(), Report> {
    let context = patterns.context();
    patterns.push(GlobalSymbolPattern {
        info: pattern_info::<GlobalSymbol>(
            &context,
            "wasm-global-symbol",
            [registered::<wasm::DataAddr>(&context)],
        ),
        converter: converter.clone(),
    });
    patterns.push(FunctionRefPattern {
        info: pattern_info::<FunctionReference>(
            &context,
            "wasm-function-ref",
            [registered::<wasm::FuncIndex>(&context)],
        ),
        converter: converter.clone(),
    });
    for tail in [false, true] {
        let (role, info) = if tail {
            (
                Role::TailCallIndirect,
                pattern_info::<TailCallIndirect>(
                    &context,
                    "wasm-tail-call-indirect-by-type",
                    [registered::<wasm::ReturnCallIndirect>(&context)],
                ),
            )
        } else {
            (
                Role::CallIndirect,
                pattern_info::<CallIndirect>(
                    &context,
                    "wasm-call-indirect-by-type",
                    [registered::<wasm::CallIndirect>(&context)],
                ),
            )
        };
        let inner: Rc<dyn ConversionPattern> = Rc::new(IndirectCallPattern {
            info,
            converter: converter.clone(),
            symbols: symbols.clone(),
            tail,
        });
        patterns.push(calls::with_convention(&context, role, inner, converter));
    }
    Ok(())
}

/// `builtin.global_symbol @g`: `wasm.data_addr @g + 0`.
struct GlobalSymbolPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for GlobalSymbolPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for GlobalSymbolPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let (global, path, span) = {
            let op_ref = op.borrow();
            let symbol = op_ref.downcast_ref::<GlobalSymbol>().expect("rooted at global_symbol");
            (symbol.global(), symbol.symbol().path().clone(), op_ref.span())
        };
        let Some(global) = global else {
            rewriter.notify_match_failure(
                op,
                Report::msg("the symbol does not resolve to a `builtin.global_variable`"),
            );
            return Ok(false);
        };
        rewriter.set_insertion_point_before(op);
        let mut address = rewriter.create_op::<wasm::DataAddr, (GlobalVariableRef, Int)>(
            span,
            (global, Int::from(0i32)),
        )?;
        address.borrow_mut().data_mut().set_path(path);
        let value = address.borrow().result().as_value_ref();
        rewriter.replace_op(op, &[value])?;
        Ok(true)
    }
}

/// `builtin.function_ref @f`: `wasm.func_index @f`, by the path it was written with.
struct FunctionRefPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for FunctionRefPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for FunctionRefPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let (resolved, path, span) = {
            let op_ref = op.borrow();
            let reference =
                op_ref.downcast_ref::<FunctionReference>().expect("rooted at function_ref");
            let callee = reference.callee();
            (callee.resolve_callable(), callee.path().clone(), op_ref.span())
        };
        let callee = match resolved {
            Ok(resolved) => resolved.named_symbol(),
            Err(error) => {
                rewriter.notify_match_failure(
                    op,
                    Report::msg(format!("the callee does not resolve to a callable: {error}")),
                );
                return Ok(false);
            }
        };
        rewriter.set_insertion_point_before(op);
        let mut index = rewriter.create_op::<wasm::FuncIndex, _>(span, (callee,))?;
        index.borrow_mut().callee_mut().set_path(path);
        let value = index.borrow().result().as_value_ref();
        rewriter.replace_op(op, &[value])?;
        Ok(true)
    }
}

/// `builtin.call_indirect` (or, with `tail`, `builtin.tail_call_indirect`) through the table of
/// its module in `symbols`, by type: the callee is the `i32` index its operand became, the arguments the values
/// the others became, and the signature the converted one. The calling convention's rules are
/// added around it by `calls::with_convention`.
struct IndirectCallPattern {
    info: PatternInfo,
    converter: TypeConverter,
    symbols: Rc<WellKnownSymbols>,
    tail: bool,
}

impl Pattern for IndirectCallPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for IndirectCallPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let (signature, span) = {
            let op_ref = op.borrow();
            let signature: Option<Signature> = if self.tail {
                op_ref.downcast_ref::<TailCallIndirect>().map(|op| op.get_signature().clone())
            } else {
                op_ref.downcast_ref::<CallIndirect>().map(|op| op.get_signature().clone())
            };
            (signature.expect("rooted at an indirect call"), op_ref.span())
        };
        let Some(table) = self.symbols.of_op(op).table else {
            return Err(Report::msg(format!(
                "`convert-to-wasm` declared no indirect-function table for the module of `{}`",
                op.borrow().name()
            )));
        };
        let converted = convert_signature(rewriter.context(), &signature, &self.converter);
        let Some(index) = operands.one(0) else {
            rewriter.notify_match_failure(
                op,
                Report::msg("the callee, a function value, did not become exactly one value"),
            );
            return Ok(false);
        };
        let arguments = operands.group_flattened(1);
        rewriter.set_insertion_point_before(op);
        let path = local_path(op, table.as_operation_ref(), INDIRECT_FUNCTION_TABLE);
        if self.tail {
            let mut call = rewriter.create_op::<wasm::ReturnCallIndirect, _>(
                span,
                (table, index, arguments, converted.signature),
            )?;
            if let Some(path) = path {
                call.borrow_mut().table_mut().set_path(path);
            }
            rewriter.erase_op(op)?;
            return Ok(true);
        }
        let mut call = rewriter.create_op::<wasm::CallIndirect, _>(
            span,
            (table, index, arguments, converted.signature),
        )?;
        if let Some(path) = path {
            call.borrow_mut().table_mut().set_path(path);
        }
        let results = call
            .borrow()
            .as_operation()
            .results()
            .iter()
            .map(|result| result.borrow().as_value_ref())
            .collect::<Vec<ValueRef>>();
        let ranges = converted
            .results
            .iter()
            .map(|range| &results[range.start as usize..range.end as usize])
            .collect::<Vec<_>>();
        rewriter.replace_op_with_multiple(op, &ranges)?;
        Ok(true)
    }
}
