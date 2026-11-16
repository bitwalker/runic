//! Constants, `select` and the arithmetic of the numeric types.

use runic_ir::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::{FloatAttr, IntAttr},
    effects::MemoryEffectOpInterface,
    traits::*,
    *,
};

use crate::{FloatType, IntType, NumType, *};

/// An integer constant, `wasm.const 5 : i32`: `i32.const` and `i64.const`.
///
/// The value is an [IntAttr] of type `i32` or `i64`, and the result has the type of the value.
/// The text writes the value as a signed decimal and the type after a colon; the literal may also
/// be written in hexadecimal or binary. The operation is free of effects, folds to its attribute
/// (as every constant does) and may appear in a Wasm constant expression
/// ([ConstantInstruction]).
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    traits(ConstantLike, ConstantInstruction),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Const {
    #[attr(hidden)]
    value: IntAttr,
    #[result]
    result: AnyType,
}

impl InferTypeOpInterface for Const {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.value().ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// The value has a type `IntType` accepts, and the result has it.
impl Verify<dyn InferTypeOpInterface> for Const {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        crate::ops::verify_constant::<IntType>(self.as_operation(), self.value().ty())
    }
}

impl Foldable for Const {
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

impl OpPrinter for Const {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        printer.print_space();
        printer.print_int(*self.get_value());
        printer.print_space();
        printer.print_colon_type(self.result().ty());
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

impl OpParser for Const {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        use alloc::format;

        use runic_ir::parse::{ParserError, ParserExt};

        let (literal_span, literal) = parser.parse_integer_literal()?.into_parts();
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

        if !is_int_type(ty) {
            return Err(ParserError::InvalidOperationType {
                span,
                ty_span,
                reason: format!("expected `i32` or `i64`, got '{ty}'"),
            });
        }
        let value = literal.to_int(ty, literal_span)?;
        let attr = parser.context_rc().int_attr(value, ty);

        state.add_attribute("value", attr);
        state.attrs.extend(attrs);
        state.results.push(ty);

        Ok(())
    }
}

/// A floating-point constant, `wasm.constf 1.5 : f64`: `f32.const` and `f64.const`.
///
/// The value is a [FloatAttr] and the result has its type, `f32` or `f64`. A NaN or an infinity is
/// written as its bits, `wasm.constf 0x7ff8000000000000 : f64`, so every bit pattern, a NaN's
/// payload included, is a constant. The operation is free of effects, folds to its attribute and
/// may appear in a Wasm constant expression ([ConstantInstruction]).
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    name = "constf",
    traits(ConstantLike, ConstantInstruction),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct ConstF {
    #[attr(hidden)]
    value: FloatAttr,
    #[result]
    result: AnyType,
}

impl InferTypeOpInterface for ConstF {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.value().ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// The value has a type `FloatType` accepts, and the result has it.
impl Verify<dyn InferTypeOpInterface> for ConstF {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        crate::ops::verify_constant::<FloatType>(self.as_operation(), self.value().ty())
    }
}

impl Foldable for ConstF {
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

impl OpPrinter for ConstF {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        printer.print_space();
        printer.print_float(*self.get_value());
        printer.print_space();
        printer.print_colon_type(self.result().ty());
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

impl OpParser for ConstF {
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

        if !is_float_type(ty) {
            return Err(ParserError::InvalidOperationType {
                span,
                ty_span,
                reason: format!("expected `f32` or `f64`, got '{ty}'"),
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

/// Selects one of two values: `select` and `select t`.
///
/// `wasm.select %c, %a, %b` has the value `%a` when `%c`, an `i32`, is not zero, and `%b` when it
/// is zero. Both values are evaluated. `%a` and `%b` have one Wasm value type, numeric or
/// reference, and the result has that type. It is free of effects.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = WasmDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Select {
    #[operand]
    condition: AnyType,
    #[operand]
    if_true: AnyType,
    #[operand]
    if_false: AnyType,
    #[result]
    result: AnyType,
}

impl InferTypeOpInterface for Select {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.if_true().ty();
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// The condition is an `i32`, and the two values and the result have one Wasm value type.
impl Verify<dyn InferTypeOpInterface> for Select {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        crate::ops::verify_select(self.as_operation())
    }
}

same_type_binary! {
    /// Addition: `i32.add`, `i64.add`, `f32.add`, `f64.add`.
    ///
    /// Two operands of one number type (`i32`, `i64`, `f32`, `f64`) and a result of that type. Integer
    /// addition wraps modulo 2^N; float addition is IEEE 754 binary arithmetic, rounding to nearest, ties
    /// to even. It never traps. It is commutative (in value: a float NaN's payload is unspecified either
    /// way), and, on integers, may appear in a Wasm constant expression ([ConstantInstruction]).
    Add, "add", NumType, [Commutative, ConstantInstruction]
}

same_type_binary! {
    /// Subtraction: `i32.sub`, `i64.sub`, `f32.sub`, `f64.sub`.
    ///
    /// Two operands of one number type and a result of that type. Integer subtraction wraps modulo
    /// 2^N; float subtraction is IEEE 754 binary arithmetic, rounding to nearest, ties to even. It never
    /// traps. On integers it may appear in a Wasm constant expression ([ConstantInstruction]).
    Sub, "sub", NumType, [ConstantInstruction]
}

same_type_binary! {
    /// Multiplication: `i32.mul`, `i64.mul`, `f32.mul`, `f64.mul`.
    ///
    /// Two operands of one number type and a result of that type. Integer multiplication wraps modulo
    /// 2^N; float multiplication is IEEE 754 binary arithmetic, rounding to nearest, ties to even. It
    /// never traps. It is commutative (in value: a float NaN's payload is unspecified either way), and,
    /// on integers, may appear in a Wasm constant expression ([ConstantInstruction]).
    Mul, "mul", NumType, [Commutative, ConstantInstruction]
}

same_type_binary! {
    /// Signed integer division: `i32.div_s`, `i64.div_s`.
    ///
    /// Two operands of one integer type (`i32` or `i64`) and a result of that type; the operands are
    /// read as signed numbers and the quotient is truncated toward zero. It traps when the divisor is
    /// zero and when the dividend is the minimum value and the divisor is -1 (the quotient does not
    /// fit). A trap is not an effect: an unused result removes the operation.
    DivS, "div_s", IntType, []
}

same_type_binary! {
    /// Unsigned integer division: `i32.div_u`, `i64.div_u`.
    ///
    /// Two operands of one integer type and a result of that type; the operands are read as unsigned
    /// numbers. It traps when the divisor is zero. A trap is not an effect: an unused result removes the
    /// operation.
    DivU, "div_u", IntType, []
}

same_type_binary! {
    /// Signed integer remainder: `i32.rem_s`, `i64.rem_s`.
    ///
    /// Two operands of one integer type and a result of that type; the operands are read as signed
    /// numbers and the remainder has the sign of the dividend. It traps when the divisor is zero; the
    /// minimum value remainder -1 is 0 and does not trap. A trap is not an effect: an unused result
    /// removes the operation.
    RemS, "rem_s", IntType, []
}

same_type_binary! {
    /// Unsigned integer remainder: `i32.rem_u`, `i64.rem_u`.
    ///
    /// Two operands of one integer type and a result of that type; the operands are read as unsigned
    /// numbers. It traps when the divisor is zero. A trap is not an effect: an unused result removes the
    /// operation.
    RemU, "rem_u", IntType, []
}

same_type_binary! {
    /// Bitwise and: `i32.and`, `i64.and`.
    ///
    /// Two operands of one integer type and a result of that type. It never traps and is commutative.
    And, "and", IntType, [Commutative]
}

same_type_binary! {
    /// Bitwise or: `i32.or`, `i64.or`.
    ///
    /// Two operands of one integer type and a result of that type. It never traps and is commutative.
    Or, "or", IntType, [Commutative]
}

same_type_binary! {
    /// Bitwise exclusive or: `i32.xor`, `i64.xor`.
    ///
    /// Two operands of one integer type and a result of that type. It never traps and is commutative.
    Xor, "xor", IntType, [Commutative]
}

same_type_binary! {
    /// Shift left: `i32.shl`, `i64.shl`.
    ///
    /// Two operands of one integer type and a result of that type: the value and the shift count. The
    /// count is taken modulo the width (32 or 64), so a count of the width or more does not give zero.
    /// It never traps.
    Shl, "shl", IntType, []
}

same_type_binary! {
    /// Arithmetic (sign-extending) shift right: `i32.shr_s`, `i64.shr_s`.
    ///
    /// Two operands of one integer type and a result of that type: the value, read as signed, and the
    /// shift count, which is taken modulo the width. It never traps.
    ShrS, "shr_s", IntType, []
}

same_type_binary! {
    /// Logical (zero-filling) shift right: `i32.shr_u`, `i64.shr_u`.
    ///
    /// Two operands of one integer type and a result of that type: the value, read as unsigned, and the
    /// shift count, which is taken modulo the width. It never traps.
    ShrU, "shr_u", IntType, []
}

same_type_binary! {
    /// Rotate left: `i32.rotl`, `i64.rotl`.
    ///
    /// Two operands of one integer type and a result of that type: the value and the rotation count,
    /// which is taken modulo the width. It never traps.
    Rotl, "rotl", IntType, []
}

same_type_binary! {
    /// Rotate right: `i32.rotr`, `i64.rotr`.
    ///
    /// Two operands of one integer type and a result of that type: the value and the rotation count,
    /// which is taken modulo the width. It never traps.
    Rotr, "rotr", IntType, []
}

same_type_unary! {
    /// Count of leading zero bits: `i32.clz`, `i64.clz`.
    ///
    /// One operand of an integer type and a result of that type; the count of an all-zero value is the
    /// width. It never traps.
    Clz, "clz", IntType, []
}

same_type_unary! {
    /// Count of trailing zero bits: `i32.ctz`, `i64.ctz`.
    ///
    /// One operand of an integer type and a result of that type; the count of an all-zero value is the
    /// width. It never traps.
    Ctz, "ctz", IntType, []
}

same_type_unary! {
    /// Count of set bits: `i32.popcnt`, `i64.popcnt`.
    ///
    /// One operand of an integer type and a result of that type. It never traps.
    Popcnt, "popcnt", IntType, []
}

same_type_binary! {
    /// Floating-point division: `f32.div`, `f64.div`.
    ///
    /// Two operands of one float type (`f32` or `f64`) and a result of that type; IEEE 754 division,
    /// rounding to nearest, ties to even. It never traps: a zero divisor gives an infinity or a NaN.
    Div, "div", FloatType, []
}

same_type_binary! {
    /// Floating-point minimum: `f32.min`, `f64.min`.
    ///
    /// Two operands of one float type and a result of that type. The result is a NaN if either operand
    /// is a NaN, and `-0.0` is below `0.0`. It never traps.
    Min, "min", FloatType, []
}

same_type_binary! {
    /// Floating-point maximum: `f32.max`, `f64.max`.
    ///
    /// Two operands of one float type and a result of that type. The result is a NaN if either operand
    /// is a NaN, and `-0.0` is below `0.0`. It never traps.
    Max, "max", FloatType, []
}

same_type_binary! {
    /// Copy of a sign: `f32.copysign`, `f64.copysign`.
    ///
    /// Two operands of one float type and a result of that type: the first operand with the sign of
    /// the second, bit for bit (a NaN included). It never traps.
    Copysign, "copysign", FloatType, []
}

same_type_unary! {
    /// Absolute value: `f32.abs`, `f64.abs`.
    ///
    /// One operand of a float type and a result of that type: the operand with its sign bit cleared
    /// (a NaN included). It never traps.
    Abs, "abs", FloatType, []
}

same_type_unary! {
    /// Negation: `f32.neg`, `f64.neg`.
    ///
    /// One operand of a float type and a result of that type: the operand with its sign bit flipped
    /// (a NaN included). It never traps.
    Neg, "neg", FloatType, []
}

same_type_unary! {
    /// Square root: `f32.sqrt`, `f64.sqrt`.
    ///
    /// One operand of a float type and a result of that type; IEEE 754 square root, a NaN for a
    /// negative operand other than `-0.0`. It never traps.
    Sqrt, "sqrt", FloatType, []
}

same_type_unary! {
    /// Round up: `f32.ceil`, `f64.ceil`.
    ///
    /// One operand of a float type and a result of that type: the smallest integral value that is not
    /// below the operand. It never traps.
    Ceil, "ceil", FloatType, []
}

same_type_unary! {
    /// Round down: `f32.floor`, `f64.floor`.
    ///
    /// One operand of a float type and a result of that type: the largest integral value that is not
    /// above the operand. It never traps.
    Floor, "floor", FloatType, []
}

same_type_unary! {
    /// Round toward zero: `f32.trunc`, `f64.trunc`.
    ///
    /// One operand of a float type and a result of that type: the integral value nearest the operand
    /// that is not larger in magnitude. This is the float-to-float rounding; the conversion of a float
    /// to an integer is [TruncS] and [TruncU]. It never traps.
    Trunc, "trunc", FloatType, []
}

same_type_unary! {
    /// Round to nearest, ties to even: `f32.nearest`, `f64.nearest`.
    ///
    /// One operand of a float type and a result of that type: the integral value nearest the operand,
    /// the even one of two when the operand is halfway between. It never traps.
    Nearest, "nearest", FloatType, []
}
