//! An independent model of the `arith` operations: the oracle of the differential test of the
//! folds and of the pattern tests.
//!
//! Nothing here calls an arithmetic method of `Int`: a value is a width and a `u128` of bits, and
//! every operation is computed bit by bit or on 256-bit limbs (`[u64; 4]`), where the exact result
//! of an addition, subtraction or multiplication of two 128-bit numbers fits, so "does it fit" is
//! a comparison and not a flag. [M::of_int] and [M::to_int] only move the bits.

use alloc::vec::Vec;

use runic_core::Int;

/// A `w`-bit pattern, `1 <= w <= 128`; the bits at or above `w` are zero.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct M {
    pub w: u32,
    pub bits: u128,
}

pub fn mask(w: u32) -> u128 {
    assert!((1..=128).contains(&w));
    if w == 128 {
        u128::MAX
    } else {
        (1u128 << w) - 1
    }
}

impl M {
    pub fn new(w: u32, bits: u128) -> Self {
        Self {
            w,
            bits: bits & mask(w),
        }
    }

    pub fn zero(w: u32) -> Self {
        Self::new(w, 0)
    }

    pub fn ones(w: u32) -> Self {
        Self::new(w, u128::MAX)
    }

    pub fn smin(w: u32) -> Self {
        Self::new(w, 1u128 << (w - 1))
    }

    pub fn smax(w: u32) -> Self {
        Self::new(w, mask(w) >> 1)
    }

    pub fn flag(value: bool) -> Self {
        Self::new(1, value as u128)
    }

    /// The bits of `value`.
    pub fn of_int(value: Int) -> Self {
        Self::new(value.width(), value.unsigned())
    }

    /// An `Int` of these bits.
    pub fn to_int(self) -> Int {
        Int::from_bits(self.w, self.bits)
    }

    /// Bit `i`; clear for `i >= w`.
    pub fn bit(self, i: u32) -> bool {
        i < self.w && (self.bits >> i) & 1 == 1
    }

    pub fn sign(self) -> bool {
        self.bit(self.w - 1)
    }

    /// The `w`-bit pattern whose bit `i` is `f(i)`.
    pub fn from_fn(w: u32, f: impl Fn(u32) -> bool) -> Self {
        let mut bits = 0u128;
        for i in 0..w {
            if f(i) {
                bits |= 1u128 << i;
            }
        }
        Self::new(w, bits)
    }

    pub fn trunc(self, w: u32) -> Self {
        assert!(w <= self.w);
        Self::from_fn(w, |i| self.bit(i))
    }

    pub fn zext(self, w: u32) -> Self {
        assert!(w >= self.w);
        Self::from_fn(w, |i| self.bit(i))
    }

    pub fn sext(self, w: u32) -> Self {
        assert!(w >= self.w);
        Self::from_fn(w, |i| if i < self.w { self.bit(i) } else { self.sign() })
    }

    pub fn sext_or_trunc(self, w: u32) -> Self {
        if w >= self.w {
            self.sext(w)
        } else {
            self.trunc(w)
        }
    }

    pub fn zext_or_trunc(self, w: u32) -> Self {
        if w >= self.w {
            self.zext(w)
        } else {
            self.trunc(w)
        }
    }
}

/// A 256-bit two's complement number, least significant limb first.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Wide([u64; 4]);

impl Wide {
    fn zext(value: M) -> Self {
        Self([value.bits as u64, (value.bits >> 64) as u64, 0, 0])
    }

    fn sext(value: M) -> Self {
        let extended = value.sext(128).bits;
        let fill = if value.sign() { u64::MAX } else { 0 };
        Self([extended as u64, (extended >> 64) as u64, fill, fill])
    }

    fn low(self, w: u32) -> M {
        M::new(w, self.0[0] as u128 | ((self.0[1] as u128) << 64))
    }

    fn add(self, rhs: Self) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for i in 0..4 {
            let sum = self.0[i] as u128 + rhs.0[i] as u128 + carry;
            out[i] = sum as u64;
            carry = sum >> 64;
        }
        Self(out)
    }

    fn not(self) -> Self {
        Self(self.0.map(|limb| !limb))
    }

    fn neg(self) -> Self {
        self.not().add(Self([1, 0, 0, 0]))
    }

    fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    /// The product modulo `2^256`: exact for two sign- or zero-extended 128-bit numbers.
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0u64; 4];
        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..(4 - i) {
                let cur = out[i + j] as u128 + (self.0[i] as u128) * (rhs.0[j] as u128) + carry;
                out[i + j] = cur as u64;
                carry = cur >> 64;
            }
        }
        Self(out)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Sign {
    Signed,
    Unsigned,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Arith {
    Add,
    Sub,
    Mul,
}

fn widen(value: M, sign: Sign) -> Wide {
    match sign {
        Sign::Signed => Wide::sext(value),
        Sign::Unsigned => Wide::zext(value),
    }
}

/// The wrapped result of `op`, and whether the exact result does not fit read as `sign` says.
pub fn overflowing(op: Arith, a: M, b: M, sign: Sign) -> (M, bool) {
    assert_eq!(a.w, b.w);
    let (x, y) = (widen(a, sign), widen(b, sign));
    let exact = match op {
        Arith::Add => x.add(y),
        Arith::Sub => x.sub(y),
        Arith::Mul => x.mul(y),
    };
    let wrapped = exact.low(a.w);
    (wrapped, widen(wrapped, sign) != exact)
}

pub fn wrapping(op: Arith, a: M, b: M) -> M {
    // The low bits do not depend on how the operands are read.
    let signed = overflowing(op, a, b, Sign::Signed).0;
    assert_eq!(signed, overflowing(op, a, b, Sign::Unsigned).0);
    signed
}

pub fn neg(a: M) -> M {
    Wide::zext(a).neg().low(a.w)
}

/// Restoring long division of two unsigned numbers; `b != 0`.
fn udivrem_bits(a: u128, b: u128) -> (u128, u128) {
    assert!(b != 0);
    let (mut q, mut r) = (0u128, 0u128);
    for i in (0..128).rev() {
        let carry = r >> 127;
        r = (r << 1) | ((a >> i) & 1);
        if carry == 1 || r >= b {
            r = r.wrapping_sub(b);
            q |= 1u128 << i;
        }
    }
    (q, r)
}

/// Quotient and remainder read as unsigned; `None` for a zero divisor.
pub fn udivrem(a: M, b: M) -> Option<(M, M)> {
    assert_eq!(a.w, b.w);
    if b.bits == 0 {
        return None;
    }
    let (q, r) = udivrem_bits(a.bits, b.bits);
    Some((M::new(a.w, q), M::new(a.w, r)))
}

fn magnitude(a: M) -> u128 {
    if a.sign() { neg(a).bits } else { a.bits }
}

/// Quotient (rounded toward zero) and remainder (sign of the dividend) read as signed. `None` for
/// a zero divisor. For `MIN / -1` the quotient is `None` (it does not fit: `sdiv` and `sdivrem`
/// trap) and the remainder is zero (`srem` does not trap).
pub fn sdivrem(a: M, b: M) -> Option<(Option<M>, M)> {
    assert_eq!(a.w, b.w);
    if b.bits == 0 {
        return None;
    }
    if a == M::smin(a.w) && b == M::ones(a.w) {
        return Some((None, M::zero(a.w)));
    }
    let (q, r) = udivrem_bits(magnitude(a), magnitude(b));
    let (q, r) = (M::new(a.w, q), M::new(a.w, r));
    let q = if a.sign() != b.sign() { neg(q) } else { q };
    let r = if a.sign() { neg(r) } else { r };
    Some((Some(q), r))
}

/// The shift amount `b` as a number, if it is less than the width.
fn amount(a: M, b: M) -> Option<u32> {
    assert_eq!(a.w, b.w);
    (b.bits < a.w as u128).then_some(b.bits as u32)
}

pub fn shl(a: M, b: M) -> Option<M> {
    let n = amount(a, b)?;
    Some(M::from_fn(a.w, |i| i >= n && a.bit(i - n)))
}

pub fn lshr(a: M, b: M) -> Option<M> {
    let n = amount(a, b)?;
    Some(M::from_fn(a.w, |i| a.bit(i + n)))
}

pub fn ashr(a: M, b: M) -> Option<M> {
    let n = amount(a, b)?;
    Some(M::from_fn(a.w, |i| a.bit((i + n).min(a.w - 1))))
}

pub fn rotl(a: M, b: M) -> M {
    assert_eq!(a.w, b.w);
    let n = (b.bits % a.w as u128) as u32;
    // Bit `i` of the result is bit `i - n` of the operand, modulo the width.
    M::from_fn(a.w, |i| a.bit((i + a.w - n) % a.w))
}

pub fn rotr(a: M, b: M) -> M {
    assert_eq!(a.w, b.w);
    let n = (b.bits % a.w as u128) as u32;
    M::from_fn(a.w, |i| a.bit((i + n) % a.w))
}

pub fn bitwise(a: M, b: M, f: impl Fn(bool, bool) -> bool) -> M {
    assert_eq!(a.w, b.w);
    M::from_fn(a.w, |i| f(a.bit(i), b.bit(i)))
}

pub fn bnot(a: M) -> M {
    M::from_fn(a.w, |i| !a.bit(i))
}

pub fn ult(a: M, b: M) -> bool {
    assert_eq!(a.w, b.w);
    // The highest bit that differs decides.
    for i in (0..a.w).rev() {
        if a.bit(i) != b.bit(i) {
            return b.bit(i);
        }
    }
    false
}

pub fn slt(a: M, b: M) -> bool {
    if a.sign() != b.sign() {
        a.sign()
    } else {
        ult(a, b)
    }
}

/// Whether `a pred b`, for the ten predicate names.
pub fn compare(pred: &str, a: M, b: M) -> bool {
    match pred {
        "eq" => a == b,
        "ne" => a != b,
        "slt" => slt(a, b),
        "sle" => !slt(b, a),
        "sgt" => slt(b, a),
        "sge" => !slt(a, b),
        "ult" => ult(a, b),
        "ule" => !ult(b, a),
        "ugt" => ult(b, a),
        "uge" => !ult(a, b),
        _ => panic!("unknown predicate {pred}"),
    }
}

pub fn min(a: M, b: M, sign: Sign) -> M {
    let less = match sign {
        Sign::Signed => slt(b, a),
        Sign::Unsigned => ult(b, a),
    };
    if less { b } else { a }
}

pub fn max(a: M, b: M, sign: Sign) -> M {
    let greater = match sign {
        Sign::Signed => slt(a, b),
        Sign::Unsigned => ult(a, b),
    };
    if greater { b } else { a }
}

pub fn popcnt(a: M) -> M {
    M::new(a.w, (0..a.w).filter(|i| a.bit(*i)).count() as u128)
}

/// The number of bits equal to `bit` from the top (`from_top`) or the bottom, up to the first that
/// differs.
pub fn run(a: M, bit: bool, from_top: bool) -> M {
    let mut n = 0u32;
    let order: Vec<u32> = if from_top {
        (0..a.w).rev().collect()
    } else {
        (0..a.w).collect()
    };
    for i in order {
        if a.bit(i) != bit {
            break;
        }
        n += 1;
    }
    M::new(a.w, n as u128)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model against Rust's own integers at 8 bits, exhaustively: the model is the oracle of
    /// the differential test, so it is checked against a second oracle where one exists.
    #[test]
    fn the_model_agrees_with_rust_at_eight_bits() {
        for x in 0..=255u8 {
            let a = M::new(8, x as u128);
            assert_eq!(neg(a).bits as u8, x.wrapping_neg());
            assert_eq!(bnot(a).bits as u8, !x);
            assert_eq!(popcnt(a).bits as u32, x.count_ones());
            assert_eq!(run(a, false, true).bits as u32, x.leading_zeros());
            assert_eq!(run(a, false, false).bits as u32, x.trailing_zeros());
            assert_eq!(run(a, true, true).bits as u32, x.leading_ones());
            assert_eq!(run(a, true, false).bits as u32, x.trailing_ones());
            assert_eq!(a.sext(16).bits as u16, x as i8 as i16 as u16);
            assert_eq!(a.zext(16).bits as u16, x as u16);
            assert_eq!(a.trunc(3).bits as u8, x & 7);
            for y in 0..=255u8 {
                let b = M::new(8, y as u128);
                let (sx, sy) = (x as i8, y as i8);
                for (op, u, s) in [
                    (Arith::Add, x.overflowing_add(y), sx.overflowing_add(sy)),
                    (Arith::Sub, x.overflowing_sub(y), sx.overflowing_sub(sy)),
                    (Arith::Mul, x.overflowing_mul(y), sx.overflowing_mul(sy)),
                ] {
                    let (value, flag) = overflowing(op, a, b, Sign::Unsigned);
                    assert_eq!((value.bits as u8, flag), u, "{op:?} {x} {y}");
                    let (value, flag) = overflowing(op, a, b, Sign::Signed);
                    assert_eq!((value.bits as u8 as i8, flag), s, "{op:?} {sx} {sy}");
                }
                assert_eq!(
                    udivrem(a, b).map(|(q, r)| (q.bits as u8, r.bits as u8)),
                    (y != 0).then(|| (x / y, x % y))
                );
                let signed = sdivrem(a, b);
                if y == 0 {
                    assert_eq!(signed, None);
                } else {
                    let (q, r) = signed.unwrap();
                    assert_eq!(q.map(|q| q.bits as u8 as i8), sx.checked_div(sy), "{sx} / {sy}");
                    assert_eq!(r.bits as u8 as i8, sx.wrapping_rem(sy), "{sx} % {sy}");
                }
                assert_eq!(shl(a, b).map(|v| v.bits as u8), (y < 8).then(|| x << (y & 7)));
                assert_eq!(lshr(a, b).map(|v| v.bits as u8), (y < 8).then(|| x >> (y & 7)));
                assert_eq!(ashr(a, b).map(|v| v.bits as u8 as i8), (y < 8).then(|| sx >> (y & 7)));
                assert_eq!(rotl(a, b).bits as u8, x.rotate_left(y as u32 % 8));
                assert_eq!(rotr(a, b).bits as u8, x.rotate_right(y as u32 % 8));
                assert_eq!(compare("slt", a, b), sx < sy);
                assert_eq!(compare("sle", a, b), sx <= sy);
                assert_eq!(compare("sgt", a, b), sx > sy);
                assert_eq!(compare("sge", a, b), sx >= sy);
                assert_eq!(compare("ult", a, b), x < y);
                assert_eq!(compare("ule", a, b), x <= y);
                assert_eq!(compare("ugt", a, b), x > y);
                assert_eq!(compare("uge", a, b), x >= y);
                assert_eq!(min(a, b, Sign::Signed).bits as u8 as i8, sx.min(sy));
                assert_eq!(max(a, b, Sign::Signed).bits as u8 as i8, sx.max(sy));
                assert_eq!(min(a, b, Sign::Unsigned).bits as u8, x.min(y));
                assert_eq!(max(a, b, Sign::Unsigned).bits as u8, x.max(y));
            }
        }
    }

    /// And against `u128`/`i128` on edge values, where the 256-bit limbs carry.
    #[test]
    fn the_model_agrees_with_rust_at_128_bits() {
        let edges = [
            0u128,
            1,
            2,
            u128::MAX,
            u128::MAX - 1,
            1 << 127,
            (1 << 127) - 1,
            (1 << 127) + 1,
            1 << 64,
            (1 << 64) - 1,
            0xdead_beef_0123_4567_89ab_cdef_0f1e_2d3c,
            127,
            128,
            129,
        ];
        for x in edges {
            for y in edges {
                let (a, b) = (M::new(128, x), M::new(128, y));
                let (sx, sy) = (x as i128, y as i128);
                for (op, u, s) in [
                    (Arith::Add, x.overflowing_add(y), sx.overflowing_add(sy)),
                    (Arith::Sub, x.overflowing_sub(y), sx.overflowing_sub(sy)),
                    (Arith::Mul, x.overflowing_mul(y), sx.overflowing_mul(sy)),
                ] {
                    let (value, flag) = overflowing(op, a, b, Sign::Unsigned);
                    assert_eq!((value.bits, flag), u, "{op:?} {x} {y}");
                    let (value, flag) = overflowing(op, a, b, Sign::Signed);
                    assert_eq!((value.bits as i128, flag), s, "{op:?} {sx} {sy}");
                }
                assert_eq!(
                    udivrem(a, b).map(|(q, r)| (q.bits, r.bits)),
                    (y != 0).then(|| (x / y, x % y))
                );
                if y != 0 {
                    let (q, r) = sdivrem(a, b).unwrap();
                    assert_eq!(q.map(|q| q.bits as i128), sx.checked_div(sy), "{sx} / {sy}");
                    assert_eq!(r.bits as i128, sx.wrapping_rem(sy), "{sx} % {sy}");
                }
                assert_eq!(compare("slt", a, b), sx < sy);
                assert_eq!(compare("ult", a, b), x < y);
            }
        }
    }
}
