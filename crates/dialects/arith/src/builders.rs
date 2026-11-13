use runic_core::{
    Builder, BuilderExt, Int, OpBuilder, Overflow, ReadyAttr, Report, Signedness, SourceSpan, Type,
    ValueRef, dialects::builtin::FunctionBuilder,
};

use crate::*;

/// Builds the operations of the `arith` dialect, returning their results.
pub trait ArithOpBuilder<'f, B: ?Sized + Builder> {
    /// The constant `value` of type `ty`.
    ///
    /// Panics unless `ty` is the integer type of `value`'s width, or `isize` and `value` is 64
    /// bits wide.
    fn int(&mut self, value: Int, ty: Type, span: SourceSpan) -> ValueRef {
        let attr = self.builder().context_rc().int_attr(value, ty);
        let op_builder = self.builder_mut().create::<Constant, _>(span);
        let constant = op_builder(ReadyAttr(attr)).unwrap();
        constant.borrow().result().as_value_ref()
    }

    /// The `i1` constant `value`: `1` for `true`.
    fn i1(&mut self, value: bool, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().i1();
        self.int(Int::from(value), ty, span)
    }

    /// The `i8` constant holding the bits of `value`.
    fn i8(&mut self, value: i8, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().i8();
        self.int(Int::from(value), ty, span)
    }

    /// The `i16` constant holding the bits of `value`.
    fn i16(&mut self, value: i16, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().i16();
        self.int(Int::from(value), ty, span)
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

    /// The `i128` constant holding the bits of `value`.
    fn i128(&mut self, value: i128, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().i128();
        self.int(Int::from(value), ty, span)
    }

    /// The `isize` constant `value`. On a 32-bit target it stands for its low 32 bits.
    fn isize(&mut self, value: i64, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().isize();
        self.int(Int::from(value), ty, span)
    }

    /// The wrapping two's complement sum.
    fn add(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Add, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Wrapping)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The two's complement sum, trapping if the exact sum does not fit, read as
    /// `signedness` says.
    fn add_checked(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Add, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Checked(signedness))?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The two's complement sum, undefined if the exact sum does not fit, read as
    /// `signedness` says.
    fn add_unchecked(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Add, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Unchecked(signedness))?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The wrapping two's complement sum with an overflow flag: returns the flag, which says
    /// whether the exact sum does not fit read as `signedness` says, then the wrapped sum.
    fn add_overflowing(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<(ValueRef, ValueRef), Report> {
        let op_builder = self.builder_mut().create::<AddOverflowing, _>(span);
        let op = op_builder(lhs, rhs, signedness)?;
        let op = op.borrow();
        Ok((op.overflowed().as_value_ref(), op.result().as_value_ref()))
    }

    /// The wrapping two's complement difference.
    fn sub(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Sub, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Wrapping)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The two's complement difference, trapping if the exact difference does not fit, read as
    /// `signedness` says.
    fn sub_checked(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Sub, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Checked(signedness))?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The two's complement difference, undefined if the exact difference does not fit, read as
    /// `signedness` says.
    fn sub_unchecked(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Sub, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Unchecked(signedness))?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The wrapping two's complement difference with an overflow flag: returns the flag, which says
    /// whether the exact difference does not fit read as `signedness` says, then the wrapped difference.
    fn sub_overflowing(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<(ValueRef, ValueRef), Report> {
        let op_builder = self.builder_mut().create::<SubOverflowing, _>(span);
        let op = op_builder(lhs, rhs, signedness)?;
        let op = op.borrow();
        Ok((op.overflowed().as_value_ref(), op.result().as_value_ref()))
    }

    /// The wrapping two's complement product.
    fn mul(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Mul, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Wrapping)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The two's complement product, trapping if the exact product does not fit, read as
    /// `signedness` says.
    fn mul_checked(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Mul, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Checked(signedness))?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The two's complement product, undefined if the exact product does not fit, read as
    /// `signedness` says.
    fn mul_unchecked(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Mul, (_, _, _)>(span);
        let op = op_builder(lhs, rhs, Overflow::Unchecked(signedness))?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The wrapping two's complement product with an overflow flag: returns the flag, which says
    /// whether the exact product does not fit read as `signedness` says, then the wrapped product.
    fn mul_overflowing(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<(ValueRef, ValueRef), Report> {
        let op_builder = self.builder_mut().create::<MulOverflowing, _>(span);
        let op = op_builder(lhs, rhs, signedness)?;
        let op = op.borrow();
        Ok((op.overflowed().as_value_ref(), op.result().as_value_ref()))
    }

    /// Unsigned division, rounding toward zero. Traps on a zero divisor.
    ///
    /// A trap is not an effect: the operation may be removed with its trap when its result is
    /// unused, and is never moved to where it would execute more often (see the crate docs).
    fn udiv(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Udiv, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Signed division, rounding toward zero. Traps on a zero divisor and on `MIN / -1`.
    ///
    /// A trap is not an effect: the operation may be removed with its trap when its result is
    /// unused, and is never moved to where it would execute more often (see the crate docs).
    fn sdiv(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Sdiv, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Unsigned remainder. Traps on a zero divisor.
    ///
    /// A trap is not an effect: the operation may be removed with its trap when its result is
    /// unused, and is never moved to where it would execute more often (see the crate docs).
    fn urem(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Urem, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Signed remainder, with the sign of `lhs`. Traps on a zero divisor; `MIN % -1` is `0`.
    ///
    /// A trap is not an effect: the operation may be removed with its trap when its result is
    /// unused, and is never moved to where it would execute more often (see the crate docs).
    fn srem(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Srem, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Unsigned division and remainder: returns the quotient, then the remainder, in the order of
    /// the operation's results. Traps on a zero divisor.
    ///
    /// A trap is not an effect: the operation may be removed with its trap when its result is
    /// unused, and is never moved to where it would execute more often (see the crate docs).
    fn udivrem(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        span: SourceSpan,
    ) -> Result<(ValueRef, ValueRef), Report> {
        let op_builder = self.builder_mut().create::<Udivrem, _>(span);
        let op = op_builder(lhs, rhs)?;
        let op = op.borrow();
        Ok((op.quotient().as_value_ref(), op.remainder().as_value_ref()))
    }

    /// Signed division and remainder: returns the quotient, then the remainder, in the order of
    /// the operation's results. Traps on a zero divisor, and on `MIN / -1` because of the quotient.
    ///
    /// A trap is not an effect: the operation may be removed with its trap when its result is
    /// unused, and is never moved to where it would execute more often (see the crate docs).
    fn sdivrem(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        span: SourceSpan,
    ) -> Result<(ValueRef, ValueRef), Report> {
        let op_builder = self.builder_mut().create::<Sdivrem, _>(span);
        let op = op_builder(lhs, rhs)?;
        let op = op.borrow();
        Ok((op.quotient().as_value_ref(), op.remainder().as_value_ref()))
    }

    /// Bitwise AND.
    fn band(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Band, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Bitwise OR.
    fn bor(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Bor, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Bitwise XOR.
    fn bxor(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Bxor, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Bitwise NOT.
    fn bnot(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Bnot, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `lhs` shifted left by `rhs`, which has the type of `lhs`.
    fn shl(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Shl, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `lhs` shifted right by `rhs`, filling with zeros.
    fn shr(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Shr, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `lhs` shifted right by `rhs`, filling with copies of the sign bit.
    fn ashr(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Ashr, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `lhs` rotated left by `rhs` modulo the width.
    fn rotl(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Rotl, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `lhs` rotated right by `rhs` modulo the width.
    fn rotr(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Rotr, _>(span);
        let op = op_builder(lhs, rhs)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The `i1` result of comparing `lhs` and `rhs` with `predicate`.
    fn cmpi(
        &mut self,
        predicate: CmpPredicate,
        lhs: ValueRef,
        rhs: ValueRef,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Cmpi, _>(span);
        let op = op_builder(lhs, rhs, predicate)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `lhs == rhs`: `arith.cmpi eq`.
    fn eq(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Eq, lhs, rhs, span)
    }

    /// `lhs != rhs`: `arith.cmpi ne`.
    fn ne(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Ne, lhs, rhs, span)
    }

    /// `lhs < rhs`, signed: `arith.cmpi slt`.
    fn slt(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Slt, lhs, rhs, span)
    }

    /// `lhs <= rhs`, signed: `arith.cmpi sle`.
    fn sle(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Sle, lhs, rhs, span)
    }

    /// `lhs > rhs`, signed: `arith.cmpi sgt`.
    fn sgt(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Sgt, lhs, rhs, span)
    }

    /// `lhs >= rhs`, signed: `arith.cmpi sge`.
    fn sge(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Sge, lhs, rhs, span)
    }

    /// `lhs < rhs`, unsigned: `arith.cmpi ult`.
    fn ult(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Ult, lhs, rhs, span)
    }

    /// `lhs <= rhs`, unsigned: `arith.cmpi ule`.
    fn ule(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Ule, lhs, rhs, span)
    }

    /// `lhs > rhs`, unsigned: `arith.cmpi ugt`.
    fn ugt(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Ugt, lhs, rhs, span)
    }

    /// `lhs >= rhs`, unsigned: `arith.cmpi uge`.
    fn uge(&mut self, lhs: ValueRef, rhs: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.cmpi(CmpPredicate::Uge, lhs, rhs, span)
    }

    /// The smaller of `lhs` and `rhs`, read as `signedness` says.
    fn min(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Min, _>(span);
        let op = op_builder(lhs, rhs, signedness)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The larger of `lhs` and `rhs`, read as `signedness` says.
    fn max(
        &mut self,
        lhs: ValueRef,
        rhs: ValueRef,
        signedness: Signedness,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Max, _>(span);
        let op = op_builder(lhs, rhs, signedness)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Two's complement negation.
    fn neg(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Neg, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The number of set bits, as a value of the operand's type.
    fn popcnt(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Popcnt, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The number of leading zero bits, as a value of the operand's type.
    fn clz(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Clz, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The number of trailing zero bits, as a value of the operand's type.
    fn ctz(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Ctz, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The number of leading one bits, as a value of the operand's type.
    fn clo(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Clo, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The number of trailing one bits, as a value of the operand's type.
    fn cto(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Cto, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `operand` truncated to the narrower fixed-width type `ty`.
    fn trunc(&mut self, operand: ValueRef, ty: Type, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Trunc, _>(span);
        let op = op_builder(operand, ty)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `operand` zero-extended to the wider fixed-width type `ty`.
    fn zext(&mut self, operand: ValueRef, ty: Type, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Zext, _>(span);
        let op = op_builder(operand, ty)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `operand` sign-extended to the wider fixed-width type `ty`.
    fn sext(&mut self, operand: ValueRef, ty: Type, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Sext, _>(span);
        let op = op_builder(operand, ty)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `operand` converted between `isize` and the fixed-width type `ty`, sign-extending or
    /// truncating as the target requires.
    fn sext_or_trunc(
        &mut self,
        operand: ValueRef,
        ty: Type,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<SextOrTrunc, _>(span);
        let op = op_builder(operand, ty)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// `operand` converted between `isize` and the fixed-width type `ty`, zero-extending or
    /// truncating as the target requires.
    fn zext_or_trunc(
        &mut self,
        operand: ValueRef,
        ty: Type,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<ZextOrTrunc, _>(span);
        let op = op_builder(operand, ty)?;
        Ok(op.borrow().result().as_value_ref())
    }

    fn builder(&self) -> &B;
    fn builder_mut(&mut self) -> &mut B;
}

impl<'f, B: ?Sized + Builder> ArithOpBuilder<'f, B> for FunctionBuilder<'f, B> {
    #[inline(always)]
    fn builder(&self) -> &B {
        FunctionBuilder::builder(self)
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        FunctionBuilder::builder_mut(self)
    }
}

impl<'f> ArithOpBuilder<'f, OpBuilder> for &'f mut OpBuilder {
    #[inline(always)]
    fn builder(&self) -> &OpBuilder {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut OpBuilder {
        self
    }
}

impl<B: ?Sized + Builder> ArithOpBuilder<'_, B> for B {
    #[inline(always)]
    fn builder(&self) -> &B {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        self
    }
}
