//! The converters, targets and pattern sets of the two families, and the patterns the test owns:
//! those `arith` would own for `i128 -> (i64, i64)`, the `builtin.ret_imm` pattern, and the
//! in-place retyping pattern for `isize -> i32`.

use alloc::{boxed::Box, rc::Rc, vec, vec::Vec};

use runic_core::{
    Context, ElementRewrite, Int, Op, OpRegistration, Operation, OperationName, OperationRef,
    Overflow, Report, SourceSpan, Spanned, Type, Value, ValueRef,
    conversion::{
        ConversionConfig, ConversionPattern, ConversionPatternRewriter, ConversionPatternSet,
        ConversionResult, ConversionTarget, ConvertedOperands, DynamicLegalityResult,
        TypeConversion, TypeConverter, add_builtin_legality_by_type, apply_full_conversion,
        apply_partial_conversion, populate_builtin_conversion_patterns,
    },
    dialects::builtin::{FunctionAlias, Module, Ret, RetImm},
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind},
    replace_elements_in,
    types::TypeMapper,
};
use runic_dialect_arith::{self as arith, ArithDialect, CmpPredicate};

use super::{Family, Mode, Setup};

/// The `arith` operations the patterns of a family convert.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kind {
    Constant,
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    /// `arith.cmpi eq`.
    Eq,
}

impl Kind {
    /// The kinds a family converts.
    pub(super) fn of(family: Family) -> &'static [Kind] {
        match family {
            Family::Wide => &[Kind::Constant, Kind::Add, Kind::And, Kind::Or, Kind::Xor, Kind::Eq],
            Family::Size => {
                &[Kind::Constant, Kind::Add, Kind::Sub, Kind::Mul, Kind::And, Kind::Or, Kind::Xor]
            }
        }
    }

    fn name(self, context: &Rc<Context>) -> OperationName {
        match self {
            Kind::Constant => registered::<arith::Constant>(context),
            Kind::Add => registered::<arith::Add>(context),
            Kind::Sub => registered::<arith::Sub>(context),
            Kind::Mul => registered::<arith::Mul>(context),
            Kind::And => registered::<arith::Band>(context),
            Kind::Or => registered::<arith::Bor>(context),
            Kind::Xor => registered::<arith::Bxor>(context),
            Kind::Eq => registered::<arith::Cmpi>(context),
        }
    }

    /// Adds a rule to `target` that calls an operation of this kind legal when `converter` calls
    /// it legal by type.
    fn add_rule(self, target: &mut ConversionTarget, converter: &TypeConverter) {
        let converter = converter.clone();
        let by_type =
            move |op: &Operation| DynamicLegalityResult::legal_if(converter.is_legal_op(op));
        match self {
            Kind::Constant => target.add_dynamically_legal_op::<arith::Constant, _>(by_type),
            Kind::Add => target.add_dynamically_legal_op::<arith::Add, _>(by_type),
            Kind::Sub => target.add_dynamically_legal_op::<arith::Sub, _>(by_type),
            Kind::Mul => target.add_dynamically_legal_op::<arith::Mul, _>(by_type),
            Kind::And => target.add_dynamically_legal_op::<arith::Band, _>(by_type),
            Kind::Or => target.add_dynamically_legal_op::<arith::Bor, _>(by_type),
            Kind::Xor => target.add_dynamically_legal_op::<arith::Bxor, _>(by_type),
            Kind::Eq => target.add_dynamically_legal_op::<arith::Cmpi, _>(by_type),
        };
    }
}

fn registered<T: OpRegistration>(context: &Rc<Context>) -> OperationName {
    context
        .get_or_register_dialect::<<T as OpRegistration>::Dialect>()
        .expect_registered_name::<T>()
}

// ---------------------------------------------------------------------------------------------
// Converters and targets
// ---------------------------------------------------------------------------------------------

/// The converter of `setup`: with materializations in a partial conversion only.
pub(super) fn converter(context: &Context, setup: &Setup) -> TypeConverter {
    let materialize = setup.mode == Mode::Partial;
    match setup.family {
        Family::Wide => wide_converter(context, materialize),
        Family::Size => size_converter(context, materialize),
    }
}

/// `i128` becomes `(i64, i64)`, low half first, and function types convert by their parts. With
/// `materialize`, the halves are joined by zero-extending both, shifting the high half left by 64
/// and or-ing them, and split by a truncation (the low half) and a shift right by 64 and a
/// truncation (the high half). The shift amount is an `i64` constant zero-extended to `i128`, so
/// that no `i128` constant is built.
fn wide_converter(context: &Context, materialize: bool) -> TypeConverter {
    let (i64, i128) = (context.i64(), context.i128());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == i128).then(|| TypeConversion::many([i64, i64])));
    if materialize {
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
            let joined = result(&rewriter.create_op::<arith::Bor, _>(span, (high, low))?);
            Ok(Some(joined))
        });
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

/// The `i64` constant 64, zero-extended to `ty`.
fn sixty_four(
    rewriter: &mut ConversionPatternRewriter,
    span: SourceSpan,
    ty: Type,
) -> Result<ValueRef, Report> {
    let amount = result(&rewriter.create_op::<arith::Constant, _>(span, (Int::from(64i64),))?);
    Ok(result(&rewriter.create_op::<arith::Zext, _>(span, (amount, ty))?))
}

/// `isize` becomes `i32`, and function types convert by their parts. With `materialize`, an `i32`
/// becomes an `isize` and an `isize` an `i32` by `arith.sext_or_trunc`.
fn size_converter(context: &Context, materialize: bool) -> TypeConverter {
    let (i32, isize) = (context.i32(), context.isize());
    let mut converter = TypeConverter::new();
    converter.add_conversion(move |_, ty| (ty == isize).then(|| TypeConversion::one(i32)));
    if materialize {
        converter.add_source_materialization(move |rewriter, inputs, ty, span| {
            let [value] = inputs else {
                return Ok(None);
            };
            if ty != isize || value.borrow().ty() != i32 {
                return Ok(None);
            }
            let value = rewriter.create_op::<arith::SextOrTrunc, _>(span, (*value, isize))?;
            Ok(Some(result(&value)))
        });
        converter.add_target_materialization(move |rewriter, input, types, span| {
            if types != [i32] || input.borrow().ty() != isize {
                return Ok(None);
            }
            let value = rewriter.create_op::<arith::SextOrTrunc, _>(span, (input, i32))?;
            Ok(Some(runic_core::smallvec![result(&value)]))
        });
    }
    converter.with_function_types()
}

/// The target of `setup`: the module and aliases are legal; the builtin operations and every
/// branch are judged by type; in a full conversion the whole `arith` dialect is judged by type, in
/// a partial one only the kinds of `setup.kinds` (the others have no rule). The unknown-operation
/// policy is the default.
pub(super) fn target(
    context: &Rc<Context>,
    setup: &Setup,
    converter: &TypeConverter,
) -> ConversionTarget {
    let mut target = ConversionTarget::new(context.clone());
    target.add_legal_op::<Module>().add_legal_op::<FunctionAlias>();
    add_builtin_legality_by_type(&mut target, converter);
    match setup.mode {
        Mode::Full => {
            let converter = converter.clone();
            target.add_dynamically_legal_dialect::<ArithDialect, _>(move |op| {
                DynamicLegalityResult::legal_if(converter.is_legal_op(op))
            });
        }
        Mode::Partial => {
            for kind in &setup.kinds {
                kind.add_rule(&mut target, converter);
            }
        }
    }
    target
}

/// Converts `module` as `setup` says: the builtin patterns and the family's patterns for
/// `setup.kinds`, under [target] and [converter].
pub(super) fn convert(
    context: &Rc<Context>,
    module: OperationRef,
    setup: &Setup,
) -> Result<ConversionResult, Report> {
    let converter = converter(context, setup);
    let target = target(context, setup, &converter);
    let mut patterns = ConversionPatternSet::new(context.clone());
    populate_builtin_conversion_patterns(&mut patterns, &converter)?;
    let rules = match setup.family {
        Family::Wide => wide_patterns(context, &converter, &setup.kinds),
        Family::Size => vec![retype_pattern(context, &converter, &setup.kinds)],
    };
    for rule in rules {
        patterns.push(rule);
    }
    match setup.mode {
        Mode::Full => apply_full_conversion(module, target, patterns, ConversionConfig::default()),
        Mode::Partial => {
            apply_partial_conversion(module, target, patterns, ConversionConfig::default())
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------------------------

type Rewrite = dyn Fn(
    OperationRef,
    ConvertedOperands<'_>,
    &mut ConversionPatternRewriter,
) -> Result<bool, Report>;

/// A conversion pattern made of a closure, under a type converter.
pub(super) struct Rule {
    info: PatternInfo,
    converter: TypeConverter,
    rewrite: Box<Rewrite>,
}

impl Rule {
    fn new(
        context: &Rc<Context>,
        name: &'static str,
        kind: PatternKind,
        converter: &TypeConverter,
        generated: impl IntoIterator<Item = OperationName>,
        rewrite: impl Fn(
            OperationRef,
            ConvertedOperands<'_>,
            &mut ConversionPatternRewriter,
        ) -> Result<bool, Report>
        + 'static,
    ) -> Self {
        let mut info = PatternInfo::new(context.clone(), name, kind, PatternBenefit::new(1));
        info.with_generated_ops(generated);
        Self {
            info,
            converter: converter.clone(),
            rewrite: Box::new(rewrite),
        }
    }
}

impl Pattern for Rule {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for Rule {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        (self.rewrite)(op, operands, rewriter)
    }
}

/// The one result of a new operation.
fn result<T: Op>(op: &runic_core::UnsafeIntrusiveEntityRef<T>) -> ValueRef {
    op.borrow().as_operation().results()[0].borrow().as_value_ref()
}

/// Whether the first result of `op` is an `i128`.
fn wide_result(op: OperationRef) -> bool {
    let op = op.borrow();
    op.results()[0].borrow().ty() == op.context().i128()
}

/// The `i64` constant holding `bits`.
fn half(
    rewriter: &mut ConversionPatternRewriter,
    span: SourceSpan,
    bits: u128,
) -> Result<ValueRef, Report> {
    let value = Int::from_bits(64, bits);
    Ok(result(&rewriter.create_op::<arith::Constant, _>(span, (value,))?))
}

/// The two halves each of two operands, when both became two values.
fn halves<'a>(operands: &ConvertedOperands<'a>) -> Option<([ValueRef; 2], [ValueRef; 2])> {
    match (operands.get(0), operands.get(1)) {
        ([a_low, a_high], [b_low, b_high]) => Some(([*a_low, *a_high], [*b_low, *b_high])),
        _ => None,
    }
}

/// The patterns `arith` would own for `i128 -> (i64, i64)`, for the kinds of `kinds`, and the
/// pattern of `builtin.ret_imm`.
fn wide_patterns(context: &Rc<Context>, converter: &TypeConverter, kinds: &[Kind]) -> Vec<Rule> {
    let operation = |kind: Kind| PatternKind::Operation(kind.name(context));
    let (constant, add, cmpi, zext, band) = (
        Kind::Constant.name(context),
        Kind::Add.name(context),
        Kind::Eq.name(context),
        registered::<arith::Zext>(context),
        Kind::And.name(context),
    );
    let mut rules = Vec::new();
    for kind in kinds {
        let rule = match kind {
            Kind::Constant => Rule::new(
                context,
                "i128-constant",
                operation(Kind::Constant),
                converter,
                [constant.clone()],
                |op, _, rw| {
                    if !wide_result(op) {
                        return Ok(false);
                    }
                    let bits = op
                        .borrow()
                        .downcast_ref::<arith::Constant>()
                        .unwrap()
                        .get_value()
                        .unsigned();
                    let span = op.borrow().span();
                    rw.set_insertion_point_before(op);
                    let low = half(rw, span, bits)?;
                    let high = half(rw, span, bits >> 64)?;
                    rw.replace_op_with_multiple(op, &[&[low, high]])?;
                    Ok(true)
                },
            ),
            // The low sum, a carry out of it (the low sum is below an operand exactly when the
            // sum wrapped), and the high sum plus the carry.
            Kind::Add => Rule::new(
                context,
                "i128-add",
                operation(Kind::Add),
                converter,
                [add.clone(), cmpi.clone(), zext.clone()],
                |op, operands, rw| {
                    let wraps = *op.borrow().downcast_ref::<arith::Add>().unwrap().get_overflow()
                        == Overflow::Wrapping;
                    let Some(([a_low, a_high], [b_low, b_high])) = halves(&operands) else {
                        return Ok(false);
                    };
                    if !wide_result(op) || !wraps {
                        return Ok(false);
                    }
                    let span = op.borrow().span();
                    let i64 = rw.context().i64();
                    let wrap = Overflow::Wrapping;
                    rw.set_insertion_point_before(op);
                    let low = result(&rw.create_op::<arith::Add, _>(span, (a_low, b_low, wrap))?);
                    let carry =
                        rw.create_op::<arith::Cmpi, _>(span, (low, a_low, CmpPredicate::Ult))?;
                    let carry =
                        result(&rw.create_op::<arith::Zext, _>(span, (result(&carry), i64))?);
                    let high =
                        result(&rw.create_op::<arith::Add, _>(span, (a_high, b_high, wrap))?);
                    let high = result(&rw.create_op::<arith::Add, _>(span, (high, carry, wrap))?);
                    rw.replace_op_with_multiple(op, &[&[low, high]])?;
                    Ok(true)
                },
            ),
            Kind::And => bitwise::<arith::Band>(context, "i128-and", converter),
            Kind::Or => bitwise::<arith::Bor>(context, "i128-or", converter),
            Kind::Xor => bitwise::<arith::Bxor>(context, "i128-xor", converter),
            // Both halves equal.
            Kind::Eq => Rule::new(
                context,
                "i128-eq",
                operation(Kind::Eq),
                converter,
                [cmpi.clone(), band.clone()],
                |op, operands, rw| {
                    let predicate =
                        *op.borrow().downcast_ref::<arith::Cmpi>().unwrap().get_predicate();
                    let Some(([a_low, a_high], [b_low, b_high])) = halves(&operands) else {
                        return Ok(false);
                    };
                    if predicate != CmpPredicate::Eq {
                        return Ok(false);
                    }
                    let span = op.borrow().span();
                    rw.set_insertion_point_before(op);
                    let low =
                        rw.create_op::<arith::Cmpi, _>(span, (a_low, b_low, CmpPredicate::Eq))?;
                    let high =
                        rw.create_op::<arith::Cmpi, _>(span, (a_high, b_high, CmpPredicate::Eq))?;
                    let both =
                        rw.create_op::<arith::Band, _>(span, (result(&low), result(&high)))?;
                    rw.replace_op(op, &[result(&both)])?;
                    Ok(true)
                },
            ),
            Kind::Sub | Kind::Mul => continue,
        };
        rules.push(rule);
    }
    // `builtin.ret_imm` of an `i128`: its halves as constants, and a return of them.
    rules.push(Rule::new(
        context,
        "i128-ret-imm",
        PatternKind::Operation(registered::<RetImm>(context)),
        converter,
        [constant, registered::<Ret>(context)],
        |op, _, rw| {
            let (value, span) = {
                let op = op.borrow();
                let ret = op.downcast_ref::<RetImm>().unwrap();
                (ret.get_value().unsigned(), op.span())
            };
            let i128 = rw.context().i128();
            if runic_core::Attribute::ty(&*op.borrow().downcast_ref::<RetImm>().unwrap().value())
                != i128
            {
                return Ok(false);
            }
            rw.set_insertion_point_before(op);
            let low = half(rw, span, value)?;
            let high = half(rw, span, value >> 64)?;
            rw.create_op::<Ret, (Vec<ValueRef>,)>(span, (vec![low, high],))?;
            rw.erase_op(op)?;
            Ok(true)
        },
    ));
    rules
}

/// A bitwise operation over `i128`, as the same operation over each half.
fn bitwise<T>(context: &Rc<Context>, name: &'static str, converter: &TypeConverter) -> Rule
where
    T: OpRegistration + runic_core::BuildableOp<(ValueRef, ValueRef)>,
{
    let root = registered::<T>(context);
    Rule::new(
        context,
        name,
        PatternKind::Operation(root.clone()),
        converter,
        [root],
        |op, operands, rw| {
            let Some(([a_low, a_high], [b_low, b_high])) = halves(&operands) else {
                return Ok(false);
            };
            if !wide_result(op) {
                return Ok(false);
            }
            let span = op.borrow().span();
            rw.set_insertion_point_before(op);
            let low = result(&rw.create_op::<T, _>(span, (a_low, b_low))?);
            let high = result(&rw.create_op::<T, _>(span, (a_high, b_high))?);
            rw.replace_op_with_multiple(op, &[&[low, high]])?;
            Ok(true)
        },
    )
}

/// One pattern for any operation: an operation of the kinds of `kinds` that uses `isize` is
/// retyped in place, its operands set to the values they became and every type it holds mapped
/// by `isize -> i32` ([replace_elements_in]), a constant truncated to its new width. Truncation
/// commutes with each of these kinds.
fn retype_pattern(context: &Rc<Context>, converter: &TypeConverter, kinds: &[Kind]) -> Rule {
    let names = kinds.iter().map(|kind| kind.name(context)).collect::<Vec<_>>();
    let check = converter.clone();
    Rule::new(
        context,
        "isize-retype-in-place",
        PatternKind::Any,
        converter,
        [],
        move |op, operands, rw| {
            if !names.contains(&op.borrow().name()) || check.is_legal_op(&op.borrow()) {
                return Ok(false);
            }
            let Some(new) = operands.all_one() else {
                return Ok(false);
            };
            let old = op
                .borrow()
                .operands()
                .iter()
                .map(|operand| operand.borrow().as_value_ref())
                .collect::<Vec<_>>();
            let context = op.borrow().context_rc();
            let mapper = TypeMapper::new(&context, |context, ty| {
                if ty == context.isize() {
                    context.i32()
                } else {
                    ty
                }
            });
            let mut rewrite = ElementRewrite::types(mapper).with_int_constants(|value, _, new| {
                let width = Int::width_for_type(new).expect("an integer type");
                Ok(if value.width() == width {
                    *value
                } else {
                    value.trunc(width)
                })
            });
            rw.modify_op_in_place(op, |op| {
                for (old, new) in old.iter().zip(&new) {
                    op.replaces_uses_of_with(*old, *new);
                }
                replace_elements_in(op, &mut rewrite)
            })??;
            Ok(true)
        },
    )
}
