//! `test-convert-wide-ints`: a dialect conversion of `i128` to `(i64, i64)`.

use core::fmt;
use std::rc::Rc;

use runic_core::{
    Context, EntityMut, Int, Op, Operation, OperationName, Report, SourceSpan, Type,
    UnsafeIntrusiveEntityRef, ValueRef,
    conversion::{
        ConversionConfig, ConversionPatternRewriter, ConversionPatternSet, ConversionTarget,
        DynamicLegalityResult, TypeConversion, TypeConverter, UnknownOpPolicy,
        add_builtin_legality_by_type, apply_full_conversion, apply_partial_conversion,
        populate_builtin_conversion_patterns,
    },
    dialects::builtin::{FunctionAlias, Module, World},
    pass::{Pass, PassExecutionState, PassInfo, PostPassStatus, write_pass_option},
};
use runic_dialect_arith as arith;

/// Converts `i128` to `(i64, i64)`, low half first, under the operation it runs on, the root of
/// the conversion (a module or a world: the root must hold every user of a function whose
/// signature changes).
///
/// Core's builtin patterns convert functions, calls, function references, returns, locals and
/// the operands every branch forwards; no `arith` operation has a pattern. In a partial
/// conversion (the default) an `arith` operation over `i128` stays, and reads a join of the two
/// halves (zero-extend both, shift the high half left by 64, or); a value of it that a converted
/// operation reads is split (truncate, and shift right by 64 and truncate). With the option
/// `full`, every operation must end up free of `i128`, so an `arith` operation over it is an
/// error.
#[derive(Default)]
pub struct ConvertWideInts {
    full: bool,
}

const NAME: &str = "test-convert-wide-ints";
const DESCRIPTION: &str =
    "Converts i128 to (i64, i64) under the operation it runs on; `full` for a full conversion";

runic_core::inventory::submit!(PassInfo::new::<ConvertWideInts>(NAME, DESCRIPTION));

impl Pass for ConvertWideInts {
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

    fn initialize_options(&mut self, options: &str) -> Result<(), Report> {
        for (key, value) in super::options(options) {
            match key {
                "full" => self.full = super::switch(NAME, key, value)?,
                _ => return Err(super::unknown(NAME, key, "'full'")),
            }
        }
        Ok(())
    }

    fn print_as_textual_pipeline(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(NAME)?;
        if self.full {
            f.write_str("{")?;
            write_pass_option(f, "full", None)?;
            f.write_str("}")?;
        }
        Ok(())
    }

    fn run_on_operation(
        &mut self,
        op: EntityMut<'_, Operation>,
        state: &mut PassExecutionState,
    ) -> Result<(), Report> {
        let root = op.as_operation_ref();
        let context = op.context_rc();
        // The conversion borrows the root itself.
        drop(op);

        let converter = converter(&context, !self.full);
        let mut target = ConversionTarget::new(context.clone());
        target
            .add_legal_op::<Module>()
            .add_legal_op::<World>()
            .add_legal_op::<FunctionAlias>();
        add_builtin_legality_by_type(&mut target, &converter);
        if self.full {
            // Every other operation is legal exactly when it holds no `i128`.
            let check = converter.clone();
            target.set_unknown_op_policy(UnknownOpPolicy::Dynamic(Rc::new(move |op| {
                DynamicLegalityResult::legal_if(check.is_legal_op(op))
            })));
        }
        let mut patterns = ConversionPatternSet::new(context.clone());
        populate_builtin_conversion_patterns(&mut patterns, &converter)?;
        let config = ConversionConfig::default();
        let result = if self.full {
            apply_full_conversion(root, target, patterns, config)?
        } else {
            apply_partial_conversion(root, target, patterns, config)?
        };
        state.set_post_pass_status(PostPassStatus::from(result.changed()));
        Ok(())
    }
}

/// `i128` becomes `(i64, i64)`, and function types convert by their parts. With `materialize`,
/// the halves are joined and split with `arith` operations.
fn converter(context: &Rc<Context>, materialize: bool) -> TypeConverter {
    let (i64, i128) = (context.i64(), context.i128());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == i128).then(|| TypeConversion::many([i64, i64])));
    if materialize {
        // Zero-extend both halves, shift the high one left by 64, or them.
        converter.add_source_materialization(move |rewriter, inputs, ty, span| {
            let [low, high] = inputs else {
                return Ok(None);
            };
            if ty != i128 {
                return Ok(None);
            }
            let low = result(&rewriter.create_op::<arith::Zext, _>(span, (*low, i128))?);
            let high = result(&rewriter.create_op::<arith::Zext, _>(span, (*high, i128))?);
            let amount = sixty_four(rewriter, span, i128)?;
            let high = result(&rewriter.create_op::<arith::Shl, _>(span, (high, amount))?);
            Ok(Some(result(&rewriter.create_op::<arith::Bor, _>(span, (high, low))?)))
        });
        // The low half is a truncation; the high half a shift right by 64 and a truncation.
        converter.add_target_materialization(move |rewriter, input, types, span| {
            if types != [i64, i64] || input.borrow().ty() != i128 {
                return Ok(None);
            }
            let low = result(&rewriter.create_op::<arith::Trunc, _>(span, (input, i64))?);
            let amount = sixty_four(rewriter, span, i128)?;
            let high = result(&rewriter.create_op::<arith::Shr, _>(span, (input, amount))?);
            let high = result(&rewriter.create_op::<arith::Trunc, _>(span, (high, i64))?);
            Ok(Some(runic_core::smallvec![low, high]))
        });
    }
    converter.with_function_types()
}

/// The `i64` constant 64 zero-extended to `ty`, so that no `i128` constant is built.
fn sixty_four(
    rewriter: &mut ConversionPatternRewriter,
    span: SourceSpan,
    ty: Type,
) -> Result<ValueRef, Report> {
    let amount = result(&rewriter.create_op::<arith::Constant, _>(span, (Int::from(64i64),))?);
    Ok(result(&rewriter.create_op::<arith::Zext, _>(span, (amount, ty))?))
}

/// The one result of a new operation.
fn result<T: Op>(op: &UnsafeIntrusiveEntityRef<T>) -> ValueRef {
    op.borrow().as_operation().results()[0].borrow().as_value_ref()
}
