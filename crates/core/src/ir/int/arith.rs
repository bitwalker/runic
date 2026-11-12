//! Arithmetic on [Int].
//!
//! Every method takes the operands by value and panics if two operands differ in width. Each is
//! computed on `unsigned()` or `signed()` in `u128`/`i128` and rebuilt with `Int::from_bits`, which
//! keeps the invariant that the bits at or above the width are zero. No shift here has an amount of
//! 128 or more: the amount is checked against the width first.

use core::ops::{BitAnd, BitOr, BitXor, Not, Shl};

use super::{Int, mask};

impl Int {
    #[track_caller]
    fn same_width(self, rhs: Self) {
        assert_eq!(self.width, rhs.width, "cannot combine integers of different widths");
    }

    /// Whether `value` is outside this width's signed range.
    fn signed_overflows(self, value: i128) -> bool {
        Self::from_signed(self.width(), value).is_none()
    }

    /// Whether the value is the minimum signed value: only the sign bit set.
    fn is_signed_min(self) -> bool {
        self == Self::signed_min(self.width())
    }

    /// The value as an amount for `<<`, [Self::lshr] or [Self::ashr]: the unsigned value, clamped
    /// to `u32::MAX`. Any amount of `width` or more behaves alike for these, so the clamp is exact.
    ///
    /// Not for a rotation: see [Self::rotate_amount].
    pub const fn shift_amount(self) -> u32 {
        if self.bits > u32::MAX as u128 {
            u32::MAX
        } else {
            self.bits as u32
        }
    }

    /// The value as an amount for [Self::rotl] or [Self::rotr] on a `width`-bit value: the
    /// unsigned value modulo `width`.
    ///
    /// Panics if `width` is 0.
    pub const fn rotate_amount(self, width: u32) -> u32 {
        assert!(width != 0, "cannot rotate a zero-width integer");
        (self.bits % width as u128) as u32
    }

    /// Wrapping addition, modulo `2^width`. The same bits for a signed or an unsigned reading.
    #[track_caller]
    pub fn wrapping_add(self, rhs: Self) -> Self {
        self.same_width(rhs);
        Self::from_bits(self.width(), self.bits.wrapping_add(rhs.bits))
    }

    /// Wrapping subtraction, modulo `2^width`. The same bits for a signed or an unsigned reading.
    #[track_caller]
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        self.same_width(rhs);
        Self::from_bits(self.width(), self.bits.wrapping_sub(rhs.bits))
    }

    /// Wrapping multiplication, modulo `2^width`. The same bits for a signed or an unsigned
    /// reading.
    #[track_caller]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        self.same_width(rhs);
        Self::from_bits(self.width(), self.bits.wrapping_mul(rhs.bits))
    }

    /// Two's complement negation, modulo `2^width`: the minimum signed value negates to itself.
    pub fn wrapping_neg(self) -> Self {
        Self::from_bits(self.width(), self.bits.wrapping_neg())
    }

    /// The wrapped sum, and whether the sum of the values read as signed numbers is out of range.
    #[track_caller]
    pub fn overflowing_add_signed(self, rhs: Self) -> (Self, bool) {
        self.same_width(rhs);
        let (value, wrapped) = self.signed().overflowing_add(rhs.signed());
        (self.wrapping_add(rhs), wrapped || self.signed_overflows(value))
    }

    /// The wrapped difference, and whether the difference of the values read as signed numbers is
    /// out of range.
    #[track_caller]
    pub fn overflowing_sub_signed(self, rhs: Self) -> (Self, bool) {
        self.same_width(rhs);
        let (value, wrapped) = self.signed().overflowing_sub(rhs.signed());
        (self.wrapping_sub(rhs), wrapped || self.signed_overflows(value))
    }

    /// The wrapped product, and whether the product of the values read as signed numbers is out
    /// of range.
    #[track_caller]
    pub fn overflowing_mul_signed(self, rhs: Self) -> (Self, bool) {
        self.same_width(rhs);
        let (value, wrapped) = self.signed().overflowing_mul(rhs.signed());
        (self.wrapping_mul(rhs), wrapped || self.signed_overflows(value))
    }

    /// The wrapped sum, and whether the sum of the values read as unsigned numbers is `2^width` or
    /// more.
    #[track_caller]
    pub fn overflowing_add_unsigned(self, rhs: Self) -> (Self, bool) {
        self.same_width(rhs);
        let (value, wrapped) = self.bits.overflowing_add(rhs.bits);
        (self.wrapping_add(rhs), wrapped || value > mask(self.width()))
    }

    /// The wrapped difference, and whether the values read as unsigned numbers give a negative
    /// difference.
    #[track_caller]
    pub fn overflowing_sub_unsigned(self, rhs: Self) -> (Self, bool) {
        self.same_width(rhs);
        (self.wrapping_sub(rhs), self.bits < rhs.bits)
    }

    /// The wrapped product, and whether the product of the values read as unsigned numbers is
    /// `2^width` or more.
    #[track_caller]
    pub fn overflowing_mul_unsigned(self, rhs: Self) -> (Self, bool) {
        self.same_width(rhs);
        let (value, wrapped) = self.bits.overflowing_mul(rhs.bits);
        (self.wrapping_mul(rhs), wrapped || value > mask(self.width()))
    }

    /// The sum, or `None` if it overflows reading both values as signed numbers.
    #[track_caller]
    pub fn checked_add_signed(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_add_signed(rhs);
        (!overflow).then_some(value)
    }

    /// The difference, or `None` if it overflows reading both values as signed numbers.
    #[track_caller]
    pub fn checked_sub_signed(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_sub_signed(rhs);
        (!overflow).then_some(value)
    }

    /// The product, or `None` if it overflows reading both values as signed numbers.
    #[track_caller]
    pub fn checked_mul_signed(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_mul_signed(rhs);
        (!overflow).then_some(value)
    }

    /// The sum, or `None` if it overflows reading both values as unsigned numbers.
    #[track_caller]
    pub fn checked_add_unsigned(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_add_unsigned(rhs);
        (!overflow).then_some(value)
    }

    /// The difference, or `None` if it overflows reading both values as unsigned numbers.
    #[track_caller]
    pub fn checked_sub_unsigned(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_sub_unsigned(rhs);
        (!overflow).then_some(value)
    }

    /// The product, or `None` if it overflows reading both values as unsigned numbers.
    #[track_caller]
    pub fn checked_mul_unsigned(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_mul_unsigned(rhs);
        (!overflow).then_some(value)
    }

    /// The quotient of the values read as unsigned numbers, or `None` if `rhs` is zero.
    #[track_caller]
    pub fn div_unsigned(self, rhs: Self) -> Option<Self> {
        self.same_width(rhs);
        (rhs.bits != 0).then(|| Self::from_bits(self.width(), self.bits / rhs.bits))
    }

    /// The remainder of the values read as unsigned numbers, or `None` if `rhs` is zero.
    #[track_caller]
    pub fn rem_unsigned(self, rhs: Self) -> Option<Self> {
        self.same_width(rhs);
        (rhs.bits != 0).then(|| Self::from_bits(self.width(), self.bits % rhs.bits))
    }

    /// The quotient of the values read as signed numbers, rounded toward zero. `None` if `rhs` is
    /// zero, or for the minimum value divided by `-1`, whose quotient is out of range.
    #[track_caller]
    pub fn div_signed(self, rhs: Self) -> Option<Self> {
        self.same_width(rhs);
        let (a, b) = (self.signed(), rhs.signed());
        if b == 0 || (b == -1 && self.is_signed_min()) {
            return None;
        }
        Some(Self::from_bits(self.width(), a.wrapping_div(b) as u128))
    }

    /// The remainder of the values read as signed numbers, with the sign of the dividend. `None`
    /// if `rhs` is zero. The minimum value modulo `-1` is zero.
    #[track_caller]
    pub fn rem_signed(self, rhs: Self) -> Option<Self> {
        self.same_width(rhs);
        let (a, b) = (self.signed(), rhs.signed());
        (b != 0).then(|| Self::from_bits(self.width(), a.wrapping_rem(b) as u128))
    }

    /// Shifts right, filling with zeros. An amount of `width` or more gives zero.
    pub fn lshr(self, amount: u32) -> Self {
        if amount >= self.width() {
            return Self::zero(self.width());
        }
        Self::from_bits(self.width(), self.bits >> amount)
    }

    /// Shifts right, filling with the sign bit. An amount of `width` or more gives every bit equal
    /// to the sign bit.
    pub fn ashr(self, amount: u32) -> Self {
        let amount = amount.min(self.width() - 1);
        Self::from_bits(self.width(), (self.signed() >> amount) as u128)
    }

    /// Rotates left by `amount` modulo the width.
    ///
    /// A rotation depends on the amount modulo the width, so an amount held as an `Int` must not be
    /// clamped with [Self::shift_amount]: use [Self::rotate_amount].
    pub fn rotl(self, amount: u32) -> Self {
        let width = self.width();
        let amount = amount % width;
        if amount == 0 {
            return self;
        }
        Self::from_bits(width, (self.bits << amount) | (self.bits >> (width - amount)))
    }

    /// Rotates right by `amount` modulo the width.
    ///
    /// A rotation depends on the amount modulo the width, so an amount held as an `Int` must not be
    /// clamped with [Self::shift_amount]: use [Self::rotate_amount].
    pub fn rotr(self, amount: u32) -> Self {
        let width = self.width();
        self.rotl(width - amount % width)
    }

    /// The smaller of the two values read as signed numbers.
    #[track_caller]
    pub fn min_signed(self, rhs: Self) -> Self {
        if self.cmp_signed(rhs).is_le() {
            self
        } else {
            rhs
        }
    }

    /// The larger of the two values read as signed numbers.
    #[track_caller]
    pub fn max_signed(self, rhs: Self) -> Self {
        if self.cmp_signed(rhs).is_ge() {
            self
        } else {
            rhs
        }
    }

    /// The smaller of the two values read as unsigned numbers.
    #[track_caller]
    pub fn min_unsigned(self, rhs: Self) -> Self {
        if self.cmp_unsigned(rhs).is_le() {
            self
        } else {
            rhs
        }
    }

    /// The larger of the two values read as unsigned numbers.
    #[track_caller]
    pub fn max_unsigned(self, rhs: Self) -> Self {
        if self.cmp_unsigned(rhs).is_ge() {
            self
        } else {
            rhs
        }
    }

    /// The number of set bits.
    pub const fn count_ones(self) -> u32 {
        self.bits.count_ones()
    }

    /// The number of clear bits above the highest set bit, within the width: the width for zero.
    pub const fn leading_zeros(self) -> u32 {
        self.bits.leading_zeros() - (Self::MAX_WIDTH - self.width as u32)
    }

    /// The number of clear bits below the lowest set bit, within the width: the width for zero.
    pub const fn trailing_zeros(self) -> u32 {
        if self.bits == 0 {
            self.width as u32
        } else {
            self.bits.trailing_zeros()
        }
    }

    /// The number of set bits from the top of the width down to the first clear bit: the width
    /// for all ones.
    pub fn leading_ones(self) -> u32 {
        (!self).leading_zeros()
    }

    /// The number of set bits from the bottom up to the first clear bit: the width for all ones.
    pub fn trailing_ones(self) -> u32 {
        (!self).trailing_zeros()
    }
}

macro_rules! bitwise_operator {
    ($trait:ident, $method:ident, $op:tt) => {
        /// Bitwise operation on two integers of the same width; panics if the widths differ.
        impl $trait for Int {
            type Output = Int;

            #[track_caller]
            fn $method(self, rhs: Int) -> Int {
                self.same_width(rhs);
                Int::from_bits(self.width(), self.bits $op rhs.bits)
            }
        }
    };
}

bitwise_operator!(BitAnd, bitand, &);
bitwise_operator!(BitOr, bitor, |);
bitwise_operator!(BitXor, bitxor, ^);

/// Shifts left, filling with zeros. An amount of `width` or more gives zero.
///
/// Left shift has one meaning for a bit pattern, like the bitwise operators, so it is an operator
/// (an inherent method named `shl` trips clippy's `should_implement_trait`). Call it as
/// `x << amount`, or `x.shl(amount)` with `core::ops::Shl` in scope.
///
/// Unlike Rust's `<<` on a primitive, an amount of `width` or more is not an overflow: it gives
/// zero.
impl Shl<u32> for Int {
    type Output = Int;

    fn shl(self, amount: u32) -> Int {
        if amount >= self.width() {
            return Int::zero(self.width());
        }
        Int::from_bits(self.width(), self.bits << amount)
    }
}

/// The bitwise complement within the width.
impl Not for Int {
    type Output = Int;

    fn not(self) -> Int {
        Int::from_bits(self.width(), !self.bits)
    }
}

/// Folds an operation whose operands are `isize` constants (64-bit values) so that the result is
/// right on a 32-bit target as well as a 64-bit one.
///
/// `f` is called with the operands as they are, and again with each truncated to 32 bits. The fold
/// succeeds when both calls do and agree. Whether the result is an `isize` is decided by both
/// result widths: if the first is 64 bits wide and the second 32, the result is pointer-width and
/// the first, truncated to 32 bits, must equal the second. Otherwise the two results must have the
/// same width and be equal (a comparison's `i1`, or an extension to `i64` or `i128`, whose result
/// does not change width with the target). Results of any other pair of widths do not fold. It
/// returns the first call's result.
///
/// Panics if an operand is not 64 bits wide.
pub fn fold_isize<const N: usize>(
    operands: [Int; N],
    f: impl Fn([Int; N]) -> Option<Int>,
) -> Option<Int> {
    for operand in operands {
        assert_eq!(operand.width(), 64, "isize constants are 64 bits wide");
    }
    let wide = f(operands)?;
    let narrow = f(operands.map(|operand| operand.trunc(32)))?;
    let agree = if wide.width() == 64 && narrow.width() == 32 {
        wide.trunc(32) == narrow
    } else {
        wide == narrow
    };
    agree.then_some(wide)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::{cmp::Ordering, ops::Shl};

    use crate::Int;

    // The reference model: plain i128/u128 arithmetic on the sign- or zero-extended operands.

    fn low_mask(w: u32) -> u128 {
        if w == 128 {
            u128::MAX
        } else {
            (1u128 << w) - 1
        }
    }

    fn reduce(w: u32, v: i128) -> Int {
        Int::from_bits(w, (v as u128) & low_mask(w))
    }

    fn model_signed(w: u32, b: u128) -> i128 {
        if (b >> (w - 1)) & 1 == 1 {
            b as i128 - (1i128 << w)
        } else {
            b as i128
        }
    }

    fn bit_at(b: u128, i: u32) -> bool {
        (b >> i) & 1 == 1
    }

    fn from_bit_fn(w: u32, f: impl Fn(u32) -> bool) -> Int {
        let mut bits = 0u128;
        for i in 0..w {
            if f(i) {
                bits |= 1 << i;
            }
        }
        Int::from_bits(w, bits)
    }

    #[test]
    fn small_widths_against_the_model() {
        for w in 1..=8u32 {
            let n = 1u128 << w;
            let lo = -(1i128 << (w - 1));
            let hi = 1i128 << (w - 1);
            let signed_out = |v: i128| v < lo || v >= hi;
            let unsigned_out = |v: i128| v < 0 || v >= (1i128 << w);
            for a in 0..n {
                let x = Int::from_bits(w, a);
                let (sa, ua) = (model_signed(w, a), a as i128);

                assert_eq!(x.wrapping_neg(), reduce(w, -sa));
                assert_eq!(!x, reduce(w, !sa));
                let bits: Vec<bool> = (0..w).map(|i| bit_at(a, i)).collect();
                assert_eq!(x.count_ones(), bits.iter().filter(|b| **b).count() as u32);
                assert_eq!(
                    x.leading_zeros(),
                    bits.iter().rev().take_while(|b| !**b).count() as u32
                );
                assert_eq!(x.trailing_zeros(), bits.iter().take_while(|b| !**b).count() as u32);
                assert_eq!(x.leading_ones(), bits.iter().rev().take_while(|b| **b).count() as u32);
                assert_eq!(x.trailing_ones(), bits.iter().take_while(|b| **b).count() as u32);

                for amt in 0..=w + 2 {
                    let shl = if amt >= w { 0 } else { ua << amt };
                    assert_eq!(x.shl(amt), reduce(w, shl), "shl {x:?} {amt}");
                    let lshr = if amt >= w { 0 } else { ua >> amt };
                    assert_eq!(x.lshr(amt), reduce(w, lshr), "lshr {x:?} {amt}");
                    assert_eq!(x.ashr(amt), reduce(w, sa >> amt), "ashr {x:?} {amt}");
                }
                for amt in 0..=2 * w + 1 {
                    let k = amt % w;
                    assert_eq!(
                        x.rotl(amt),
                        from_bit_fn(w, |i| bit_at(a, (i + w - k) % w)),
                        "rotl {x:?} {amt}"
                    );
                    assert_eq!(
                        x.rotr(amt),
                        from_bit_fn(w, |i| bit_at(a, (i + k) % w)),
                        "rotr {x:?} {amt}"
                    );
                }

                for b in 0..n {
                    let y = Int::from_bits(w, b);
                    let (sb, ub) = (model_signed(w, b), b as i128);

                    assert_eq!(x.wrapping_add(y), reduce(w, sa + sb));
                    assert_eq!(x.wrapping_sub(y), reduce(w, sa - sb));
                    assert_eq!(x.wrapping_mul(y), reduce(w, sa * sb));

                    let cases = [
                        ("add", sa + sb, ua + ub),
                        ("sub", sa - sb, ua - ub),
                        ("mul", sa * sb, ua * ub),
                    ];
                    let signed_results = [
                        (x.overflowing_add_signed(y), x.checked_add_signed(y)),
                        (x.overflowing_sub_signed(y), x.checked_sub_signed(y)),
                        (x.overflowing_mul_signed(y), x.checked_mul_signed(y)),
                    ];
                    let unsigned_results = [
                        (x.overflowing_add_unsigned(y), x.checked_add_unsigned(y)),
                        (x.overflowing_sub_unsigned(y), x.checked_sub_unsigned(y)),
                        (x.overflowing_mul_unsigned(y), x.checked_mul_unsigned(y)),
                    ];
                    for (i, (name, se, ue)) in cases.into_iter().enumerate() {
                        let (r, flag) = signed_results[i].0;
                        assert_eq!(
                            (r, flag),
                            (reduce(w, se), signed_out(se)),
                            "{name} signed {x:?} {y:?}"
                        );
                        assert_eq!(signed_results[i].1, (!flag).then_some(r));
                        let (r, flag) = unsigned_results[i].0;
                        assert_eq!(
                            (r, flag),
                            (reduce(w, ue), unsigned_out(ue)),
                            "{name} unsigned {x:?} {y:?}"
                        );
                        assert_eq!(unsigned_results[i].1, (!flag).then_some(r));
                    }

                    assert_eq!(x.div_unsigned(y), (ub != 0).then(|| reduce(w, ua / ub)));
                    assert_eq!(x.rem_unsigned(y), (ub != 0).then(|| reduce(w, ua % ub)));
                    let min_by_neg_one = sa == lo && sb == -1;
                    assert_eq!(
                        x.div_signed(y),
                        (sb != 0 && !min_by_neg_one).then(|| reduce(w, sa / sb))
                    );
                    assert_eq!(x.rem_signed(y), (sb != 0).then(|| reduce(w, sa % sb)));

                    assert_eq!(x & y, reduce(w, sa & sb));
                    assert_eq!(x | y, reduce(w, sa | sb));
                    assert_eq!(x ^ y, reduce(w, sa ^ sb));
                    let (min_s, max_s) = if sa <= sb { (x, y) } else { (y, x) };
                    assert_eq!((x.min_signed(y), x.max_signed(y)), (min_s, max_s));
                    let (min_u, max_u) = if ua <= ub { (x, y) } else { (y, x) };
                    assert_eq!((x.min_unsigned(y), x.max_unsigned(y)), (min_u, max_u));
                }
            }
        }
    }

    #[test]
    fn counts_at_the_extremes() {
        for w in 1..=128u32 {
            assert_eq!(Int::zero(w).leading_zeros(), w);
            assert_eq!(Int::zero(w).trailing_zeros(), w);
            assert_eq!(Int::zero(w).count_ones(), 0);
            assert_eq!(Int::zero(w).leading_ones(), 0);
            assert_eq!(Int::zero(w).trailing_ones(), 0);
            assert_eq!(Int::all_ones(w).leading_ones(), w);
            assert_eq!(Int::all_ones(w).trailing_ones(), w);
            assert_eq!(Int::all_ones(w).count_ones(), w);
            assert_eq!(Int::all_ones(w).leading_zeros(), 0);
            assert_eq!(Int::all_ones(w).trailing_zeros(), 0);
            assert_eq!(Int::from_bits(w, 1).leading_zeros(), w - 1);
        }
    }

    #[test]
    fn shifts_and_rotates_at_every_width() {
        for w in 1..=128u32 {
            let x =
                Int::from_bits(w, 0x8000_0000_0000_0001_u128.wrapping_mul(0x1234_5678_9abc_def1));
            for amt in [0, 1, w - 1, w, w + 1, 2 * w, u32::MAX] {
                let k = amt % w;
                assert_eq!(x.rotl(amt), from_bit_fn(w, |i| bit_at(x.unsigned(), (i + w - k) % w)));
                assert_eq!(x.rotr(amt), from_bit_fn(w, |i| bit_at(x.unsigned(), (i + k) % w)));
                if amt >= w {
                    assert_eq!(x.shl(amt), Int::zero(w));
                    assert_eq!(x.lshr(amt), Int::zero(w));
                    let fill = if x.is_negative() {
                        Int::all_ones(w)
                    } else {
                        Int::zero(w)
                    };
                    assert_eq!(x.ashr(amt), fill);
                }
            }
        }
    }

    macro_rules! wide_test {
        ($name:ident, $w:expr, $i:ty, $u:ty) => {
            #[test]
            fn $name() {
                let w: u32 = $w;
                let samples: [i128; 11] = [
                    0,
                    1,
                    2,
                    -1,
                    <$i>::MIN as i128,
                    <$i>::MIN as i128 + 1,
                    <$i>::MAX as i128,
                    <$i>::MAX as i128 - 1,
                    1 << 32,
                    (1 << 32) - 1,
                    1 << 31,
                ];
                let int = |v: $i| Int::from_bits(w, v as $u as u128);
                let sint = |v: $u| Int::from_bits(w, v as u128);
                for &a in &samples {
                    let (a, ua) = (a as $i, a as $i as $u);
                    let x = int(a);
                    assert_eq!(x.wrapping_neg(), int(a.wrapping_neg()));
                    assert_eq!(!x, int(!a));
                    assert_eq!(x.count_ones(), a.count_ones());
                    assert_eq!(x.leading_zeros(), a.leading_zeros());
                    assert_eq!(x.trailing_zeros(), a.trailing_zeros());
                    assert_eq!(x.leading_ones(), a.leading_ones());
                    assert_eq!(x.trailing_ones(), a.trailing_ones());
                    for amt in [0u32, 1, 2, 31, 32, 33, w / 2, w - 2, w - 1] {
                        assert_eq!(x.shl(amt), sint(ua.wrapping_shl(amt)));
                        assert_eq!(x.lshr(amt), sint(ua.wrapping_shr(amt)));
                        assert_eq!(x.ashr(amt), int(a >> amt));
                        assert_eq!(x.rotl(amt), sint(ua.rotate_left(amt)));
                        assert_eq!(x.rotr(amt), sint(ua.rotate_right(amt)));
                    }
                    assert_eq!(x.rotl(w + 3), sint(ua.rotate_left(3)));
                    for &b in &samples {
                        let (b, ub) = (b as $i, b as $i as $u);
                        let y = int(b);
                        assert_eq!(x.wrapping_add(y), int(a.wrapping_add(b)));
                        assert_eq!(x.wrapping_sub(y), int(a.wrapping_sub(b)));
                        assert_eq!(x.wrapping_mul(y), int(a.wrapping_mul(b)));

                        let check = |s: (Int, bool),
                                     sc: Option<Int>,
                                     u: (Int, bool),
                                     uc: Option<Int>,
                                     se: ($i, bool),
                                     ue: ($u, bool),
                                     sce: Option<$i>,
                                     uce: Option<$u>| {
                            assert_eq!(s, (int(se.0), se.1), "signed {a} {b}");
                            assert_eq!(sc, sce.map(int));
                            assert_eq!(u, (sint(ue.0), ue.1), "unsigned {ua} {ub}");
                            assert_eq!(uc, uce.map(sint));
                        };
                        check(
                            x.overflowing_add_signed(y),
                            x.checked_add_signed(y),
                            x.overflowing_add_unsigned(y),
                            x.checked_add_unsigned(y),
                            a.overflowing_add(b),
                            ua.overflowing_add(ub),
                            a.checked_add(b),
                            ua.checked_add(ub),
                        );
                        check(
                            x.overflowing_sub_signed(y),
                            x.checked_sub_signed(y),
                            x.overflowing_sub_unsigned(y),
                            x.checked_sub_unsigned(y),
                            a.overflowing_sub(b),
                            ua.overflowing_sub(ub),
                            a.checked_sub(b),
                            ua.checked_sub(ub),
                        );
                        check(
                            x.overflowing_mul_signed(y),
                            x.checked_mul_signed(y),
                            x.overflowing_mul_unsigned(y),
                            x.checked_mul_unsigned(y),
                            a.overflowing_mul(b),
                            ua.overflowing_mul(ub),
                            a.checked_mul(b),
                            ua.checked_mul(ub),
                        );

                        assert_eq!(x.div_unsigned(y), ua.checked_div(ub).map(sint));
                        assert_eq!(x.rem_unsigned(y), ua.checked_rem(ub).map(sint));
                        assert_eq!(x.div_signed(y), a.checked_div(b).map(int));
                        // `i64::checked_rem` is `None` for `MIN % -1`; `rem_signed` gives zero there.
                        assert_eq!(x.rem_signed(y), (b != 0).then(|| int(a.wrapping_rem(b))));

                        assert_eq!(x & y, int(a & b));
                        assert_eq!(x | y, int(a | b));
                        assert_eq!(x ^ y, int(a ^ b));
                        assert_eq!(x.min_signed(y), int(a.min(b)));
                        assert_eq!(x.max_signed(y), int(a.max(b)));
                        assert_eq!(x.min_unsigned(y), sint(ua.min(ub)));
                        assert_eq!(x.max_unsigned(y), sint(ua.max(ub)));
                    }
                }
            }
        };
    }

    wide_test!(sixty_four_bits_against_i64_and_u64, 64, i64, u64);
    wide_test!(one_hundred_twenty_eight_bits_against_i128_and_u128, 128, i128, u128);

    fn s(w: u32, v: i128) -> Int {
        Int::from_signed(w, v).unwrap()
    }

    fn u(w: u32, v: u128) -> Int {
        Int::from_unsigned(w, v).unwrap()
    }

    #[test]
    fn multiplication_overflow_at_65_bits() {
        // Signed range is [-2^64, 2^64), unsigned is [0, 2^65).
        let x = u(65, 1 << 32);
        assert_eq!(x.overflowing_mul_signed(x), (Int::from_bits(65, 1 << 64), true));
        assert_eq!(x.overflowing_mul_unsigned(x), (Int::from_bits(65, 1 << 64), false));
        let y = u(65, 1 << 33);
        assert_eq!(y.overflowing_mul_unsigned(x), (Int::zero(65), true));
        let min = s(65, -(1i128 << 64));
        assert_eq!(min.overflowing_mul_signed(s(65, -1)), (min, true));
        assert_eq!(min.checked_mul_signed(s(65, 1)), Some(min));
        assert_eq!(s(65, 1 << 63).checked_mul_signed(s(65, 1)), Some(s(65, 1 << 63)));
        assert_eq!(s(65, 1 << 31).checked_mul_signed(s(65, 1 << 32)), Some(s(65, 1 << 63)));
        assert_eq!(s(65, 1 << 32).checked_mul_signed(s(65, 1 << 32)), None);
        assert_eq!(min.div_signed(s(65, -1)), None);
        assert_eq!(min.rem_signed(s(65, -1)), Some(Int::zero(65)));
    }

    #[test]
    fn edges_at_127_bits() {
        let min = s(127, -(1i128 << 126));
        let max = s(127, (1i128 << 126) - 1);
        assert_eq!(min.div_signed(s(127, -1)), None);
        assert_eq!(min.rem_signed(s(127, -1)), Some(Int::zero(127)));
        assert_eq!(min.div_signed(s(127, 1)), Some(min));
        assert_eq!(min.overflowing_mul_signed(s(127, -1)), (min, true));
        assert_eq!(max.overflowing_add_signed(s(127, 1)), (min, true));
        assert_eq!(max.overflowing_add_unsigned(s(127, 1)), (Int::from_bits(127, 1 << 126), false));
        assert_eq!(
            Int::all_ones(127).overflowing_add_unsigned(Int::from_bits(127, 1)),
            (Int::zero(127), true)
        );
        assert_eq!(
            Int::zero(127).overflowing_sub_unsigned(Int::from_bits(127, 1)),
            (Int::all_ones(127), true)
        );
        assert_eq!(min.overflowing_sub_signed(s(127, 1)), (max, true));
        assert_eq!(u(127, 1 << 62).checked_mul_unsigned(u(127, 1 << 63)), Some(u(127, 1 << 125)));
        assert_eq!(u(127, 1 << 63).checked_mul_unsigned(u(127, 1 << 64)), None);
        assert_eq!(s(127, 1 << 62).checked_mul_signed(s(127, 1 << 63)), Some(s(127, 1 << 125)));
        assert_eq!(s(127, 1 << 63).checked_mul_signed(s(127, 1 << 63)), None);
        assert_eq!(min.ashr(200), Int::all_ones(127));
        assert_eq!(min.lshr(126), Int::from_bits(127, 1));
        assert_eq!(min.cmp_signed(max), Ordering::Less);
    }

    #[test]
    #[should_panic]
    fn arithmetic_on_different_widths_panics() {
        let _ = Int::from(1u8).wrapping_add(Int::from(1u16));
    }

    #[test]
    #[should_panic]
    fn operator_on_different_widths_panics() {
        let _ = Int::from(1u8) & Int::from(1u16);
    }

    // fold_isize

    fn isize_const(v: i64) -> Int {
        Int::from(v)
    }

    fn shl_amount(b: Int) -> u32 {
        b.unsigned().min(u32::MAX as u128) as u32
    }

    #[test]
    fn fold_isize_commuting_operations_fold() {
        let values = [
            0i64,
            1,
            -1,
            2,
            i64::MIN,
            i64::MAX,
            1 << 32,
            (1 << 32) - 1,
            0x1234_5678_9abc_def0,
        ];
        for &a in &values {
            for &b in &values {
                let (x, y) = (isize_const(a), isize_const(b));
                let fold =
                    |f: fn(Int, Int) -> Int| super::fold_isize([x, y], |[p, q]| Some(f(p, q)));
                assert_eq!(fold(Int::wrapping_add), Some(isize_const(a.wrapping_add(b))));
                assert_eq!(fold(Int::wrapping_sub), Some(isize_const(a.wrapping_sub(b))));
                assert_eq!(fold(Int::wrapping_mul), Some(isize_const(a.wrapping_mul(b))));
                assert_eq!(fold(|p, q| p & q), Some(isize_const(a & b)));
                assert_eq!(fold(|p, q| p | q), Some(isize_const(a | b)));
                assert_eq!(fold(|p, q| p ^ q), Some(isize_const(a ^ b)));
            }
        }
    }

    #[test]
    fn fold_isize_shl() {
        let shl = |a: i64, amount: i64| {
            super::fold_isize([isize_const(a), isize_const(amount)], |[p, q]| {
                Some(p.shl(shl_amount(q)))
            })
        };
        for amount in [0i64, 1, 31, 32, 40, 63, 64, 100] {
            assert_eq!(
                shl(0x1234_5678_9abc_def1, amount),
                Some(isize_const(0x1234_5678_9abc_def1).shl(amount as u32))
            );
        }
        assert_eq!(shl(1, (1 << 32) + 1), None);
    }

    #[test]
    fn fold_isize_operations_that_depend_on_the_width() {
        let div = |a: i64, b: i64| {
            super::fold_isize([isize_const(a), isize_const(b)], |[p, q]| p.div_unsigned(q))
        };
        assert_eq!(div(1 << 32, 2), None);
        assert_eq!(div(100, 7), Some(isize_const(14)));

        let ult = |a: i64, b: i64| {
            super::fold_isize([isize_const(a), isize_const(b)], |[p, q]| {
                Some(Int::from(p.cmp_unsigned(q) == Ordering::Less))
            })
        };
        assert_eq!(ult(1 << 32, 1), None);
        assert_eq!(ult(5, 7), Some(Int::from(true)));
        assert_eq!(ult(7, 5), Some(Int::from(false)));

        let lshr = |a: i64, n: i64| {
            super::fold_isize([isize_const(a), isize_const(n)], |[p, q]| {
                Some(p.lshr(shl_amount(q)))
            })
        };
        assert_eq!(lshr(-1, 1), None);
        assert_eq!(lshr(8, 1), Some(isize_const(4)));
    }

    #[test]
    fn fold_isize_propagates_none() {
        assert_eq!(super::fold_isize([isize_const(1)], |_| None), None);
        // Fails only on the 32-bit call: the operand is 2^32, which truncates to zero.
        let div = super::fold_isize([isize_const(1 << 32)], |[p]| {
            Int::from_bits(p.width(), 1).div_unsigned(p)
        });
        assert_eq!(div, None);
    }

    #[test]
    fn fold_isize_extensions() {
        let sext = |a: i64| super::fold_isize([isize_const(a)], |[p]| Some(p.sext(64)));
        assert_eq!(sext(5), Some(isize_const(5)));
        assert_eq!(sext(1 << 32), None);
        assert_eq!(sext(-1), Some(isize_const(-1)));
        let zext = |a: i64| super::fold_isize([isize_const(a)], |[p]| Some(p.zext(64)));
        assert_eq!(zext(5), Some(isize_const(5)));
        assert_eq!(zext(1 << 32), None);
        assert_eq!(zext(-1), None);
        let wide = super::fold_isize([isize_const(-1)], |[p]| Some(p.sext(128)));
        assert_eq!(wide, Some(Int::from(-1i128)));
        // Models `sext isize -> i64` as written in terms of the operand's own width.
        let modeled = super::fold_isize([isize_const(7)], |[p]| Some(p.trunc(p.width()).sext(64)));
        assert_eq!(modeled, Some(isize_const(7)));
    }

    #[test]
    fn fold_isize_other_result_widths() {
        let same32 = super::fold_isize([isize_const(1 << 40)], |_| Some(Int::from(7u32)));
        assert_eq!(same32, Some(Int::from(7u32)));
        let same_width_32 =
            super::fold_isize([isize_const(1)], |[p]| Some(Int::from_bits(32, p.unsigned())));
        assert_eq!(same_width_32, Some(Int::from(1u32)));
        let odd = super::fold_isize([isize_const(1)], |[p]| {
            Some(Int::zero(if p.width() == 64 { 16 } else { 8 }))
        });
        assert_eq!(odd, None);
    }

    #[test]
    fn shift_and_rotate_amounts() {
        let big = Int::from((1u64 << 32) + 1);
        assert_eq!(big.shift_amount(), u32::MAX);
        assert_eq!(big.rotate_amount(64), 1);
        assert_eq!(big.rotate_amount(7), (((1u64 << 32) + 1) % 7) as u32);
        assert_eq!(Int::from(5u64).shift_amount(), 5);
        assert_eq!(Int::from(u32::MAX as u64).shift_amount(), u32::MAX);
        assert_eq!(Int::from(u128::MAX).shift_amount(), u32::MAX);
        assert_eq!(Int::from(70u8).rotate_amount(64), 6);
        assert_eq!(Int::from_bits(3, 7).shift_amount(), 7);
    }

    #[test]
    #[should_panic]
    fn rotate_amount_of_width_zero_panics() {
        Int::from(1u8).rotate_amount(0);
    }

    #[test]
    fn signed_bounds() {
        assert_eq!(Int::signed_min(8).signed(), i8::MIN as i128);
        assert_eq!(Int::signed_max(8).signed(), i8::MAX as i128);
        assert_eq!(Int::signed_min(64).signed(), i64::MIN as i128);
        assert_eq!(Int::signed_max(64).signed(), i64::MAX as i128);
        assert_eq!(Int::signed_min(128).signed(), i128::MIN);
        assert_eq!(Int::signed_max(128).signed(), i128::MAX);
        assert_eq!(Int::signed_min(127).signed(), -(1i128 << 126));
        assert_eq!(Int::signed_max(127).signed(), (1i128 << 126) - 1);
        assert_eq!(Int::signed_min(1).signed(), -1);
        assert_eq!(Int::signed_max(1).signed(), 0);
        for w in 1..=128u32 {
            assert_eq!(Int::signed_min(w).width(), w);
            assert_eq!(Int::signed_max(w).width(), w);
            assert!(Int::signed_min(w).is_signed_min());
            assert_eq!(Int::signed_max(w), !Int::signed_min(w));
        }
    }

    #[test]
    fn sixty_three_bits_against_the_model() {
        let w = 63u32;
        let (lo, hi) = (-(1i128 << 62), (1i128 << 62) - 1);
        let samples: [i128; 11] =
            [0, 1, 2, -1, lo, lo + 1, hi, hi - 1, 1 << 32, (1 << 32) - 1, 1 << 31];
        let unsigned_of = |v: i128| (v as u128) & ((1u128 << w) - 1);
        let umax = (1i128 << w) - 1;
        for &a in &samples {
            for &b in &samples {
                let (x, y) = (s(w, a), s(w, b));
                let (ua, ub) = (unsigned_of(a) as i128, unsigned_of(b) as i128);
                let signed = [(a + b, 0), (a - b, 1), (a * b, 2)];
                let unsigned = [(ua + ub, 0), (ua - ub, 1), (ua * ub, 2)];
                let so = [
                    (x.overflowing_add_signed(y), x.checked_add_signed(y)),
                    (x.overflowing_sub_signed(y), x.checked_sub_signed(y)),
                    (x.overflowing_mul_signed(y), x.checked_mul_signed(y)),
                ];
                let uo = [
                    (x.overflowing_add_unsigned(y), x.checked_add_unsigned(y)),
                    (x.overflowing_sub_unsigned(y), x.checked_sub_unsigned(y)),
                    (x.overflowing_mul_unsigned(y), x.checked_mul_unsigned(y)),
                ];
                let wrapped = [x.wrapping_add(y), x.wrapping_sub(y), x.wrapping_mul(y)];
                for ((v, i), (uv, _)) in signed.into_iter().zip(unsigned) {
                    let out = v < lo || v > hi;
                    assert_eq!(so[i].0, (reduce(w, v), out), "signed {a} {b} op {i}");
                    assert_eq!(so[i].1, (!out).then(|| reduce(w, v)));
                    let uout = uv < 0 || uv > umax;
                    assert_eq!(uo[i].0, (reduce(w, uv), uout), "unsigned {ua} {ub} op {i}");
                    assert_eq!(uo[i].1, (!uout).then(|| reduce(w, uv)));
                    assert_eq!(wrapped[i], reduce(w, v));
                }
            }
        }
        let min = s(w, lo);
        assert_eq!(min.div_signed(s(w, -1)), None);
        assert_eq!(min.rem_signed(s(w, -1)), Some(Int::zero(w)));
    }

    #[test]
    #[should_panic(expected = "different widths")]
    fn min_with_different_widths_panics() {
        let _ = Int::from(1u8).min_signed(Int::from(1u16));
    }

    #[test]
    #[should_panic]
    fn fold_isize_requires_64_bit_operands() {
        let _ = super::fold_isize([Int::from(1u32)], |[p]| Some(p));
    }
}
