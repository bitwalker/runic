//! The builder trait of the dialect.

use runic_ir::{
    Builder, BuilderExt, Float, Int, OpBuilder, ReadyAttr, Report, SourceSpan, Type, ValueRef,
    dialects::builtin::FunctionBuilder,
};

use crate::*;

/// Defines the builder methods of the binary operations (two operands of one type).
macro_rules! binary_builders {
    ($($method:ident, $Op:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, ".")]
        fn $method(
            &mut self,
            lhs: ValueRef,
            rhs: ValueRef,
            span: SourceSpan,
        ) -> Result<ValueRef, Report> {
            let op_builder = self.builder_mut().create::<$Op, _>(span);
            let op = op_builder(lhs, rhs)?;
            Ok(op.borrow().result().as_value_ref())
        }
    )*};
}

/// Defines the builder methods of the operations of one operand and a result that follows from it.
macro_rules! unary_builders {
    ($($method:ident, $Op:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, ".")]
        fn $method(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
            let op_builder = self.builder_mut().create::<$Op, _>(span);
            let op = op_builder(operand)?;
            Ok(op.borrow().result().as_value_ref())
        }
    )*};
}

/// Defines the builder methods of the conversions whose result type the caller names.
macro_rules! typed_conversion_builders {
    ($($method:ident, $Op:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, ".")]
        fn $method(
            &mut self,
            operand: ValueRef,
            ty: Type,
            span: SourceSpan,
        ) -> Result<ValueRef, Report> {
            let op_builder = self.builder_mut().create::<$Op, _>(span);
            let op = op_builder(operand, ty)?;
            Ok(op.borrow().result().as_value_ref())
        }
    )*};
}

/// Defines the builder methods of the conversions whose result type is fixed.
macro_rules! fixed_conversion_builders {
    ($($method:ident, $Op:ident, $result:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, ".")]
        fn $method(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
            let ty = self.builder().context().$result();
            let op_builder = self.builder_mut().create::<$Op, _>(span);
            let op = op_builder(operand, ty)?;
            Ok(op.borrow().result().as_value_ref())
        }
    )*};
}

/// Builds the operations of the `wasm` dialect, returning their results.
///
/// A method returns the error of the operation's type inference, which is none for these
/// operations; the operand and result types are checked when the function that holds the
/// operation is verified.
pub trait WasmOpBuilder<'f, B: ?Sized + Builder> {
    /// The integer constant `value` of type `ty`, an `i32` or an `i64` (any other type does not
    /// verify).
    fn int(&mut self, value: Int, ty: Type, span: SourceSpan) -> ValueRef {
        let attr = self.builder().context_rc().int_attr(value, ty);
        let op_builder = self.builder_mut().create::<Const, _>(span);
        let constant = op_builder(ReadyAttr(attr)).unwrap();
        constant.borrow().result().as_value_ref()
    }

    /// The `i32` constant holding the bits of `value`.
    fn i32(&mut self, value: i32, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().i32();
        self.int(Int::from(value), ty, span)
    }

    /// The `i64` constant holding the bits of `value`.
    fn i64(&mut self, value: i64, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().i64();
        self.int(Int::from(value), ty, span)
    }

    /// The float constant `value`, of the type its width gives: `f32` for a 32-bit value, `f64`
    /// for a 64-bit one.
    fn float(&mut self, value: Float, span: SourceSpan) -> ValueRef {
        let attr = self.builder().context_rc().float_attr(value);
        let op_builder = self.builder_mut().create::<ConstF, _>(span);
        let constant = op_builder(ReadyAttr(attr)).unwrap();
        constant.borrow().result().as_value_ref()
    }

    /// The `f32` constant `value`, bit for bit (a NaN keeps its payload).
    fn f32(&mut self, value: f32, span: SourceSpan) -> ValueRef {
        self.float(Float::from_f32(value), span)
    }

    /// The `f64` constant `value`, bit for bit (a NaN keeps its payload).
    fn f64(&mut self, value: f64, span: SourceSpan) -> ValueRef {
        self.float(Float::from_f64(value), span)
    }

    /// `if_true` if `condition`, an `i32`, is not zero, and `if_false` otherwise: `select`.
    fn select(
        &mut self,
        condition: ValueRef,
        if_true: ValueRef,
        if_false: ValueRef,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Select, _>(span);
        let op = op_builder(condition, if_true, if_false)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The `i32` 1 if `operand`, an integer, is zero, and 0 otherwise.
    fn eqz(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Eqz, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    binary_builders! {
        add, Add, "sum";
        sub, Sub, "difference";
        mul, Mul, "product";
        div_s, DivS, "signed quotient; traps on a zero divisor and on the minimum value divided by -1";
        div_u, DivU, "unsigned quotient; traps on a zero divisor";
        rem_s, RemS, "signed remainder; traps on a zero divisor";
        rem_u, RemU, "unsigned remainder; traps on a zero divisor";
        and, And, "bitwise and";
        or, Or, "bitwise or";
        xor, Xor, "bitwise exclusive or";
        shl, Shl, "value shifted left by the count modulo the width";
        shr_s, ShrS, "value shifted right, sign-filling, by the count modulo the width";
        shr_u, ShrU, "value shifted right, zero-filling, by the count modulo the width";
        rotl, Rotl, "value rotated left by the count modulo the width";
        rotr, Rotr, "value rotated right by the count modulo the width";
        div, Div, "float quotient; never traps";
        min, Min, "float minimum; a NaN if either operand is one";
        max, Max, "float maximum; a NaN if either operand is one";
        copysign, Copysign, "first float with the sign of the second";
    }

    unary_builders! {
        clz, Clz, "count of leading zero bits";
        ctz, Ctz, "count of trailing zero bits";
        popcnt, Popcnt, "count of set bits";
        abs, Abs, "float absolute value";
        neg, Neg, "float negation";
        sqrt, Sqrt, "float square root";
        ceil, Ceil, "float rounded up";
        floor, Floor, "float rounded down";
        trunc, Trunc, "float rounded toward zero";
        nearest, Nearest, "float rounded to nearest, ties to even";
        extend8_s, Extend8S, "sign extension of the low 8 bits";
        extend16_s, Extend16S, "sign extension of the low 16 bits";
        extend32_s, Extend32S, "sign extension of the low 32 bits of an `i64`";
    }

    binary_builders! {
        eq, Eq, "`i32` 1 if the operands are equal, else 0";
        ne, Ne, "`i32` 1 if the operands differ, else 0";
        lt_s, LtS, "`i32` 1 if the first is signed less than the second, else 0";
        lt_u, LtU, "`i32` 1 if the first is unsigned less than the second, else 0";
        gt_s, GtS, "`i32` 1 if the first is signed greater than the second, else 0";
        gt_u, GtU, "`i32` 1 if the first is unsigned greater than the second, else 0";
        le_s, LeS, "`i32` 1 if the first is signed less than or equal to the second, else 0";
        le_u, LeU, "`i32` 1 if the first is unsigned less than or equal to the second, else 0";
        ge_s, GeS, "`i32` 1 if the first is signed greater than or equal to the second, else 0";
        ge_u, GeU, "`i32` 1 if the first is unsigned greater than or equal to the second, else 0";
        lt, Lt, "`i32` 1 if the first float is less than the second, else 0";
        gt, Gt, "`i32` 1 if the first float is greater than the second, else 0";
        le, Le, "`i32` 1 if the first float is less than or equal to the second, else 0";
        ge, Ge, "`i32` 1 if the first float is greater than or equal to the second, else 0";
    }

    typed_conversion_builders! {
        trunc_s, TruncS, "signed integer of type `ty` from a float, truncating toward zero; traps on a NaN and on a value out of range";
        trunc_u, TruncU, "unsigned integer of type `ty` from a float, truncating toward zero; traps on a NaN and on a value out of range";
        trunc_sat_s, TruncSatS, "signed integer of type `ty` from a float, truncating toward zero and saturating; 0 for a NaN";
        trunc_sat_u, TruncSatU, "unsigned integer of type `ty` from a float, truncating toward zero and saturating; 0 for a NaN";
        convert_s, ConvertS, "float of type `ty` from a signed integer";
        convert_u, ConvertU, "float of type `ty` from an unsigned integer";
        reinterpret, Reinterpret, "same bits as the operand, as the number type `ty` of the same width";
    }

    fixed_conversion_builders! {
        wrap, Wrap, i32, "the low 32 bits of an `i64`, as an `i32`";
        extend_s, ExtendS, i64, "an `i32` sign-extended to an `i64`";
        extend_u, ExtendU, i64, "an `i32` zero-extended to an `i64`";
        promote, Promote, f64, "an `f32` as an `f64`";
        demote, Demote, f32, "an `f64` rounded to an `f32`";
    }

    /// The builder this builder wraps.
    fn builder(&self) -> &B;

    /// The builder this builder wraps, mutably.
    fn builder_mut(&mut self) -> &mut B;
}

impl<'f, B: ?Sized + Builder> WasmOpBuilder<'f, B> for FunctionBuilder<'f, B> {
    #[inline(always)]
    fn builder(&self) -> &B {
        FunctionBuilder::builder(self)
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        FunctionBuilder::builder_mut(self)
    }
}

impl<'f> WasmOpBuilder<'f, OpBuilder> for &'f mut OpBuilder {
    #[inline(always)]
    fn builder(&self) -> &OpBuilder {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut OpBuilder {
        self
    }
}

impl<B: ?Sized + Builder> WasmOpBuilder<'_, B> for B {
    #[inline(always)]
    fn builder(&self) -> &B {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        self
    }
}
