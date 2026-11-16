use runic_ir::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::FloatAttr,
    effects::MemoryEffectOpInterface,
    traits::*,
    *,
};

use crate::{ArithDialect, CmpFPredicate, CmpFPredicateAttr, fold_float::*};

/// A floating-point constant, `arith.constantf 1.5 : f64`, whose result has the type of its value.
/// A NaN or an infinity is written as its bits, `arith.constantf 0x7ff8000000000000 : f64`.
///
/// This is its own operation, where MLIR has one `arith.constant` for integers and floats: an
/// operation's attribute field has one concrete attribute type, and [crate::Constant] keeps
/// [IntAttr](runic_ir::dialects::builtin::attributes::IntAttr). The constant is free of effects
/// and folds to its attribute.
#[derive(EffectOpInterface)]
#[operation(
    dialect = ArithDialect,
    traits(ConstantLike),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Constantf {
    #[attr(hidden)]
    value: FloatAttr,
    #[result]
    result: AnyFloat,
}

impl InferTypeOpInterface for Constantf {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.value().ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// A constant's result has the type of its value, as [InferTypeOpInterface] gives it, as
/// [crate::Constant]'s verifier checks it.
impl Verify<dyn InferTypeOpInterface> for Constantf {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let attr_ty = self.value().ty();
        let result_ty = self.result().ty();
        if attr_ty == result_ty {
            return Ok(());
        }
        Err(runic_ir::diagnostics::report!(
            message: "invalid operation",
            labels: [
                primary(
                    self.as_operation().span(),
                    alloc::format!(
                        "the result of '{}' has type `{result_ty}`, but its value has type \
                         `{attr_ty}`",
                        self.as_operation().name()
                    )
                ),
            ],
        ))
    }
}

impl Foldable for Constantf {
    #[inline]
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        results.push(OpFoldResult::Attribute(self.value));
        FoldResult::Ok(())
    }

    #[inline(always)]
    fn fold_with(
        &self,
        _operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        self.fold(results)
    }
}

impl OpPrinter for Constantf {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        printer.print_space();
        printer.print_float(*self.get_value());
        printer.print_space();
        printer.print_colon_type(self.result().ty());
        // The attributes that are not properties, as `arith.constant` prints them.
        let op = self.as_operation();
        if op.has_attributes() {
            printer.print_space();
            printer.print_keyword("attributes");
            printer.print_space();
            printer.print_attribute_dictionary(
                op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }
    }
}

impl OpParser for Constantf {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        use alloc::format;

        use runic_ir::parse::{ParserError, ParserExt};

        let (literal_span, literal) = parser.parse_float_literal_text()?.into_parts();
        let (ty_span, ty) = parser.parse_colon_type()?.into_parts();
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;
        let span = SourceSpan::session(
            literal_span.source().id(),
            runic_ir::diagnostics::TextRange::new(
                literal_span.range().start(),
                ty_span.range().end(),
            )
            .expect("valid range"),
        );

        if !ty.is_float() {
            return Err(ParserError::InvalidOperationType {
                span,
                ty_span,
                reason: format!("expected floating-point type, got '{ty}'"),
            });
        }
        let value = literal.to_float(ty, literal_span)?;
        let attr = parser.context_rc().float_attr(value);

        state.add_attribute("value", attr);
        state.attrs.extend(attrs);
        state.results.push(ty);

        Ok(())
    }
}

/// Implements [InferTypeOpInterface] for a float operation whose result has the type of `$operand`.
macro_rules! infer_float_return_ty {
    ($Op:ty, $operand:ident) => {
        impl InferTypeOpInterface for $Op {
            fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
                let ty = self.$operand().ty();
                self.result_mut().set_type(ty);
                Ok(())
            }
        }
    };
}

/// Floating-point addition: IEEE 754-2019 binary arithmetic on two operands of one float type,
/// rounding to nearest, ties to even.
///
/// It is free of effects and never traps. When the result is a NaN, its sign and payload are
/// unspecified. The operation is commutative in value; either operand order may give any NaN, so
/// marking it [Commutative] (which only lets operand order be normalized) is sound. It has no
/// algebraic simplification: `x + 0.0` is not `x`, as it changes `-0.0`. It folds when both
/// operands are constants and the result is not a NaN.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Addf {
    #[operand]
    lhs: AnyFloat,
    #[operand]
    rhs: AnyFloat,
    #[result]
    result: AnyFloat,
}

infer_float_return_ty!(Addf, lhs);
fold_float_one!(Addf, |_op| |[a, b]| FloatBinary::Add.apply(a, b));

/// Floating-point subtraction: IEEE 754-2019 binary arithmetic on two operands of one float type,
/// rounding to nearest, ties to even.
///
/// It is free of effects and never traps. When the result is a NaN (`inf - inf`), its sign and
/// payload are unspecified. It folds when both operands are constants and the result is not a NaN.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Subf {
    #[operand]
    lhs: AnyFloat,
    #[operand]
    rhs: AnyFloat,
    #[result]
    result: AnyFloat,
}

infer_float_return_ty!(Subf, lhs);
fold_float_one!(Subf, |_op| |[a, b]| FloatBinary::Sub.apply(a, b));

/// Floating-point multiplication: IEEE 754-2019 binary arithmetic on two operands of one float
/// type, rounding to nearest, ties to even.
///
/// It is free of effects and never traps. When the result is a NaN (`0.0 * inf`), its sign and
/// payload are unspecified. Like [Addf] it is [Commutative] and has no algebraic simplification.
/// It folds when both operands are constants and the result is not a NaN.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, Commutative, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Mulf {
    #[operand]
    lhs: AnyFloat,
    #[operand]
    rhs: AnyFloat,
    #[result]
    result: AnyFloat,
}

infer_float_return_ty!(Mulf, lhs);
fold_float_one!(Mulf, |_op| |[a, b]| FloatBinary::Mul.apply(a, b));

/// Floating-point division: IEEE 754-2019 binary arithmetic on two operands of one float type,
/// rounding to nearest, ties to even.
///
/// It is free of effects. A division by zero does not trap: it gives an infinity, or a NaN for
/// `0.0 / 0.0`, whose sign and payload are unspecified. It folds when both operands are constants
/// and the result is not a NaN, so `1.0 / 0.0` folds to an infinity and `0.0 / 0.0` does not.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Divf {
    #[operand]
    lhs: AnyFloat,
    #[operand]
    rhs: AnyFloat,
    #[result]
    result: AnyFloat,
}

infer_float_return_ty!(Divf, lhs);
fold_float_one!(Divf, |_op| |[a, b]| FloatBinary::Div.apply(a, b));

/// Floating-point negation: flips the sign bit and nothing else, for every bit pattern, NaNs
/// included (a NaN keeps its payload).
///
/// It is free of effects and never traps. It folds on a constant that is not a NaN.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Negf {
    #[operand]
    operand: AnyFloat,
    #[result]
    result: AnyFloat,
}

infer_float_return_ty!(Negf, operand);
fold_float_one!(Negf, |_op| |[a]| (!a.is_nan()).then(|| negated(a)));

/// Floating-point comparison: `arith.cmpf olt %a, %b` is `1` if `%a` is ordered and less than
/// `%b`, and `0` otherwise.
///
/// The operands have one float type, the result is an `i1`. The predicate is the property
/// `predicate`, one of MLIR's sixteen (see [CmpFPredicate]), written as a keyword before the
/// operands. It is free of effects and never traps.
///
/// It folds whenever both operands are constants, NaNs included, since a comparison is
/// deterministic, and for the predicates `false` and `true` with any operands.
#[derive(EffectOpInterface)]
#[operation(
    dialect = ArithDialect,
    traits(BinaryOp, SameTypeOperands),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Cmpf {
    #[operand]
    lhs: AnyFloat,
    #[operand]
    rhs: AnyFloat,
    #[result]
    result: Bool,
    #[attr(hidden)]
    predicate: CmpFPredicateAttr,
}

impl InferTypeOpInterface for Cmpf {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i1());
        Ok(())
    }
}

impl Foldable for Cmpf {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let operands = crate::fold::operand_constants(self.as_operation());
        self.fold_with(&operands, results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        let op = self.as_operation();
        let predicate = *self.get_predicate();
        let operand_ty = crate::fold::operand_type(op);
        let folded = crate::fold_float::fold_float_compare(
            &op.context_rc(),
            // `false` and `true` need no constants: the operands are not read.
            operands,
            operand_ty,
            crate::fold::result_type(op, 0),
            |[a, b]: [Float; 2]| Some(predicate.evaluate(a, b)),
        )
        .or_else(|| match predicate {
            CmpFPredicate::False | CmpFPredicate::True => {
                let context = op.context_rc();
                let result_ty = crate::fold::result_type(op, 0);
                (result_ty == context.i1() && operand_ty.is_float()).then(|| {
                    let value = Int::from(predicate == CmpFPredicate::True);
                    context.int_attr(value, result_ty).as_attribute_ref()
                })
            }
            _ => None,
        });
        crate::fold::push_folded(results, folded.map(|attr| [attr]))
    }
}

impl OpPrinter for Cmpf {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        let op = self.as_operation();
        printer.print_space();
        printer.print_keyword(self.get_predicate().as_str());
        printer.print_space();
        printer.print_value_uses(op.operands().group(0).as_value_range());
        if op.has_attributes() {
            printer.print_space();
            printer.print_keyword("attributes");
            printer.print_space();
            printer.print_attribute_dictionary(
                op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }
    }
}

impl OpParser for Cmpf {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let predicate = crate::attributes::parse_float_predicate(parser)?;
        let lhs = parser.parse_operand(/*allow_result_number=*/ true)?;
        parser.parse_comma()?;
        let rhs = parser.parse_operand(/*allow_result_number=*/ true)?;
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

        let unknown = parser.context().unknown_type();
        let lhs = parser.resolve_operand(lhs, unknown)?;
        let rhs = parser.resolve_operand(rhs, unknown)?;
        state.add_operand(lhs);
        state.add_operand(rhs);
        state.add_attribute(
            "predicate",
            parser.context_rc().create_attribute::<CmpFPredicateAttr, _>(predicate),
        );
        state.attrs.extend(attrs);
        state.results.push(parser.context().i1());

        Ok(())
    }
}
