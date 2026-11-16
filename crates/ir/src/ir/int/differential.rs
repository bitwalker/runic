//! `Int` arithmetic, and [fold_isize], against a big-integer model.
//!
//! The model is schoolbook arithmetic on 256-bit two's complement numbers held as `[u64; 4]`
//! (little-endian limbs). It shares no code with `ir/int.rs` or `ir/int/arith.rs`, and it never
//! uses `u128`/`i128` arithmetic except to move bits in and out. Every exact result of an operation
//! on two values of at most 128 bits fits in 256 bits, so the model computes the mathematical
//! result and then reduces it, which is the definition each `Int` method is checked against.

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::{Int, IsizeFold, fold_isize};

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Big([u64; 4]);

impl Big {
    const ONE: Big = Big([1, 0, 0, 0]);
    const ZERO: Big = Big([0; 4]);

    fn from_u128(v: u128) -> Big {
        Big([v as u64, (v >> 64) as u64, 0, 0])
    }

    /// The low 128 bits.
    fn low_u128(self) -> u128 {
        (self.0[0] as u128) | ((self.0[1] as u128) << 64)
    }

    fn bit(self, i: u32) -> bool {
        (self.0[(i / 64) as usize] >> (i % 64)) & 1 == 1
    }

    fn set_bit(&mut self, i: u32) {
        self.0[(i / 64) as usize] |= 1u64 << (i % 64);
    }

    fn is_negative(self) -> bool {
        self.bit(255)
    }

    fn not(self) -> Big {
        Big([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }

    fn and(self, o: Big) -> Big {
        Big([self.0[0] & o.0[0], self.0[1] & o.0[1], self.0[2] & o.0[2], self.0[3] & o.0[3]])
    }

    fn or(self, o: Big) -> Big {
        Big([self.0[0] | o.0[0], self.0[1] | o.0[1], self.0[2] | o.0[2], self.0[3] | o.0[3]])
    }

    fn xor(self, o: Big) -> Big {
        Big([self.0[0] ^ o.0[0], self.0[1] ^ o.0[1], self.0[2] ^ o.0[2], self.0[3] ^ o.0[3]])
    }

    /// Addition modulo 2^256, limb by limb with a carry.
    fn add(self, o: Big) -> Big {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            let (s1, c1) = self.0[i].overflowing_add(o.0[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            out[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }
        Big(out)
    }

    fn neg(self) -> Big {
        self.not().add(Big::ONE)
    }

    fn sub(self, o: Big) -> Big {
        self.add(o.neg())
    }

    /// Multiplication modulo 2^256, schoolbook on 32-bit digits so that no partial product needs
    /// more than 64 bits.
    fn mul(self, o: Big) -> Big {
        let digits = |b: Big| {
            let mut d = [0u64; 8];
            for i in 0..4 {
                d[2 * i] = b.0[i] & 0xffff_ffff;
                d[2 * i + 1] = b.0[i] >> 32;
            }
            d
        };
        let (a, b) = (digits(self), digits(o));
        let mut acc = [0u64; 8];
        for i in 0..8 {
            let mut carry = 0u64;
            for j in 0..(8 - i) {
                let cur = acc[i + j] + a[i] * b[j] + carry;
                acc[i + j] = cur & 0xffff_ffff;
                carry = cur >> 32;
            }
        }
        let mut out = [0u64; 4];
        for i in 0..4 {
            out[i] = acc[2 * i] | (acc[2 * i + 1] << 32);
        }
        Big(out)
    }

    fn shl1(self) -> Big {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            out[i] = (self.0[i] << 1) | carry;
            carry = self.0[i] >> 63;
        }
        Big(out)
    }

    /// Logical shift right by one.
    fn shr1(self) -> Big {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for i in (0..4).rev() {
            out[i] = (self.0[i] >> 1) | (carry << 63);
            carry = self.0[i] & 1;
        }
        Big(out)
    }

    fn shl(self, n: u32) -> Big {
        let mut out = self;
        for _ in 0..n.min(256) {
            out = out.shl1();
        }
        out
    }

    fn lshr(self, n: u32) -> Big {
        let mut out = self;
        for _ in 0..n.min(256) {
            out = out.shr1();
        }
        out
    }

    /// Arithmetic shift right: floor division by 2^n.
    fn ashr(self, n: u32) -> Big {
        if self.is_negative() {
            self.not().lshr(n).not()
        } else {
            self.lshr(n)
        }
    }

    fn cmp_unsigned(self, o: Big) -> Ordering {
        for i in (0..4).rev() {
            match self.0[i].cmp(&o.0[i]) {
                Ordering::Equal => continue,
                other => return other,
            }
        }
        Ordering::Equal
    }

    fn cmp_signed(self, o: Big) -> Ordering {
        match (self.is_negative(), o.is_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => self.cmp_unsigned(o),
        }
    }

    /// Unsigned long division, one bit at a time. `d` is not zero.
    fn divrem_unsigned(self, d: Big) -> (Big, Big) {
        assert!(d != Big::ZERO);
        let mut q = Big::ZERO;
        let mut r = Big::ZERO;
        for i in (0..256).rev() {
            r = r.shl1();
            if self.bit(i) {
                r.0[0] |= 1;
            }
            if r.cmp_unsigned(d) != Ordering::Less {
                r = r.sub(d);
                q.set_bit(i);
            }
        }
        (q, r)
    }

    fn abs(self) -> Big {
        if self.is_negative() { self.neg() } else { self }
    }

    /// Signed division rounding toward zero; the remainder has the dividend's sign.
    fn divrem_signed(self, d: Big) -> (Big, Big) {
        let (q, r) = self.abs().divrem_unsigned(d.abs());
        let q = if self.is_negative() != d.is_negative() {
            q.neg()
        } else {
            q
        };
        let r = if self.is_negative() { r.neg() } else { r };
        (q, r)
    }

    /// The low `w` bits set.
    fn mask(w: u32) -> Big {
        let mut m = Big::ZERO;
        for i in 0..w {
            m.set_bit(i);
        }
        m
    }

    /// The value modulo 2^w, as a non-negative number.
    fn low(self, w: u32) -> Big {
        self.and(Big::mask(w))
    }

    /// The low `w` bits read as a two's complement number.
    fn low_signed(self, w: u32) -> Big {
        let low = self.low(w);
        if low.bit(w - 1) {
            low.or(Big::mask(w).not())
        } else {
            low
        }
    }

    /// 2^n
    fn pow2(n: u32) -> Big {
        let mut b = Big::ZERO;
        b.set_bit(n);
        b
    }
}

/// The value of `x` read as an unsigned number.
fn unsigned_of(x: Int) -> Big {
    Big::from_u128(x.unsigned())
}

/// The value of `x` read as a two's complement number. Built from the bits, not from
/// `Int::signed`.
fn signed_of(x: Int) -> Big {
    Big::from_u128(x.unsigned()).low_signed(x.width())
}

/// The `w`-bit `Int` holding `v` modulo 2^w.
fn reduce(w: u32, v: Big) -> Int {
    Int::from_bits(w, v.low(w).low_u128())
}

/// Whether `v` is outside `[-2^(w-1), 2^(w-1))`.
fn signed_out_of_range(w: u32, v: Big) -> bool {
    let lo = Big::pow2(w - 1).neg();
    let hi = Big::pow2(w - 1);
    v.cmp_signed(lo) == Ordering::Less || v.cmp_signed(hi) != Ordering::Less
}

/// Whether `v` is outside `[0, 2^w)`.
fn unsigned_out_of_range(w: u32, v: Big) -> bool {
    v.is_negative() || v.cmp_signed(Big::pow2(w)) != Ordering::Less
}

// ---------------------------------------------------------------------------------------------
// Random inputs
// ---------------------------------------------------------------------------------------------

/// splitmix64
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn u128(&mut self) -> u128 {
        ((self.next() as u128) << 64) | self.next() as u128
    }

    fn width(&mut self) -> u32 {
        const EDGES: [u32; 12] = [1, 2, 3, 7, 8, 31, 32, 33, 63, 64, 65, 127];
        match self.below(4) {
            0 => EDGES[self.below(EDGES.len() as u64) as usize],
            1 => 128,
            _ => 1 + self.below(128) as u32,
        }
    }

    /// A `w`-bit value, biased toward the values where arithmetic goes wrong.
    fn value(&mut self, w: u32) -> Int {
        let all = if w == 128 {
            u128::MAX
        } else {
            (1u128 << w) - 1
        };
        let min = 1u128 << (w - 1);
        let bits = match self.below(16) {
            0 => 0,
            1 => 1,
            2 => all,
            3 => min,
            4 => min - 1,
            5 => min + 1,
            6 => all - 1,
            7 => 2,
            8 => 1u128 << self.below(w as u64),
            9 => (1u128 << self.below(w as u64)).wrapping_sub(1),
            10 => all ^ (1u128 << self.below(w as u64)),
            // A value that fits in the low half, so products are sometimes exact
            11 => self.u128() >> (128 - w.div_ceil(2)),
            12 => all ^ (self.u128() >> (128 - w.div_ceil(2))),
            _ => self.u128(),
        };
        Int::from_bits(w, bits)
    }

    fn amount(&mut self, w: u32) -> u32 {
        match self.below(8) {
            0 => 0,
            1 => w - 1,
            2 => w,
            3 => w + 1,
            4 => u32::MAX,
            5 => 127 + self.below(4) as u32,
            6 => self.next() as u32,
            _ => self.below(2 * w as u64 + 2) as u32,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The differential check
// ---------------------------------------------------------------------------------------------

fn check_unary(x: Int) {
    let w = x.width();
    let (u, s) = (unsigned_of(x), signed_of(x));

    assert_eq!(x.wrapping_neg(), reduce(w, s.neg()), "wrapping_neg {x:?}");
    assert_eq!(!x, reduce(w, u.not()), "not {x:?}");

    let bits: Vec<bool> = (0..w).map(|i| u.bit(i)).collect();
    assert_eq!(x.count_ones(), bits.iter().filter(|b| **b).count() as u32, "count_ones {x:?}");
    assert_eq!(
        x.leading_zeros(),
        bits.iter().rev().take_while(|b| !**b).count() as u32,
        "leading_zeros {x:?}"
    );
    assert_eq!(
        x.trailing_zeros(),
        bits.iter().take_while(|b| !**b).count() as u32,
        "trailing_zeros {x:?}"
    );
    assert_eq!(
        x.leading_ones(),
        bits.iter().rev().take_while(|b| **b).count() as u32,
        "leading_ones {x:?}"
    );
    assert_eq!(
        x.trailing_ones(),
        bits.iter().take_while(|b| **b).count() as u32,
        "trailing_ones {x:?}"
    );

    // The existing accessors, against the model
    assert_eq!(Big::from_u128(x.signed() as u128).low_signed(128), s.low_signed(128));
    assert_eq!(x.is_negative(), s.is_negative());
    assert_eq!(x.is_zero(), u == Big::ZERO);
    assert_eq!(x.is_all_ones(), u == Big::mask(w));
    assert_eq!(Int::signed_min(w), reduce(w, Big::pow2(w - 1)));
    assert_eq!(Int::signed_max(w), reduce(w, Big::pow2(w - 1).sub(Big::ONE)));

    // An amount held as an `Int`
    let clamped = if u.cmp_unsigned(Big::from_u128(u32::MAX as u128)) == Ordering::Greater {
        u32::MAX
    } else {
        u.low_u128() as u32
    };
    assert_eq!(x.shift_amount_unsigned(), clamped, "shift_amount_unsigned {x:?}");
    for rotated in [1u32, 2, 7, 32, 63, 64, 65, 127, 128, w] {
        let (_, r) = u.divrem_unsigned(Big::from_u128(rotated as u128));
        assert_eq!(
            x.rotate_amount_unsigned(rotated),
            r.low_u128() as u32,
            "rotate_amount_unsigned {x:?} {rotated}"
        );
    }
}

fn check_shift(x: Int, amount: u32) {
    let w = x.width();
    let (u, s) = (unsigned_of(x), signed_of(x));

    let shl = if amount >= w {
        Big::ZERO
    } else {
        u.shl(amount)
    };
    assert_eq!(x << amount, reduce(w, shl), "shl {x:?} {amount}");
    let lshr = if amount >= w {
        Big::ZERO
    } else {
        u.lshr(amount)
    };
    assert_eq!(x.lshr(amount), reduce(w, lshr), "lshr {x:?} {amount}");
    // Floor division of the signed value by 2^amount; a shift of 256 or more is all sign bits
    assert_eq!(x.ashr(amount), reduce(w, s.ashr(amount)), "ashr {x:?} {amount}");

    let k = amount % w;
    let rotl = u.shl(k).or(u.lshr(w - k));
    assert_eq!(x.rotl(amount), reduce(w, rotl), "rotl {x:?} {amount}");
    let rotr = u.lshr(k).or(u.shl(w - k));
    assert_eq!(x.rotr(amount), reduce(w, rotr), "rotr {x:?} {amount}");
}

fn check_binary(x: Int, y: Int) {
    let w = x.width();
    let (ux, sx) = (unsigned_of(x), signed_of(x));
    let (uy, sy) = (unsigned_of(y), signed_of(y));

    // Wrapping: the same bits for either reading
    assert_eq!(x.wrapping_add(y), reduce(w, sx.add(sy)), "wrapping_add {x:?} {y:?}");
    assert_eq!(x.wrapping_add(y), reduce(w, ux.add(uy)));
    assert_eq!(x.wrapping_sub(y), reduce(w, sx.sub(sy)), "wrapping_sub {x:?} {y:?}");
    assert_eq!(x.wrapping_sub(y), reduce(w, ux.sub(uy)));
    assert_eq!(x.wrapping_mul(y), reduce(w, sx.mul(sy)), "wrapping_mul {x:?} {y:?}");
    assert_eq!(x.wrapping_mul(y), reduce(w, ux.mul(uy)));

    // Overflow flags and checked forms, from the exact result
    type Overflowing = fn(Int, Int) -> (Int, bool);
    type Checked = fn(Int, Int) -> Option<Int>;
    let signed: [(&str, Big, Overflowing, Checked); 3] = [
        ("add", sx.add(sy), Int::overflowing_add_signed, Int::checked_add_signed),
        ("sub", sx.sub(sy), Int::overflowing_sub_signed, Int::checked_sub_signed),
        ("mul", sx.mul(sy), Int::overflowing_mul_signed, Int::checked_mul_signed),
    ];
    for (name, exact, overflowing, checked) in signed {
        let out = signed_out_of_range(w, exact);
        assert_eq!(overflowing(x, y), (reduce(w, exact), out), "{name} signed {x:?} {y:?}");
        assert_eq!(checked(x, y), (!out).then(|| reduce(w, exact)), "{name} signed {x:?} {y:?}");
    }
    let unsigned: [(&str, Big, Overflowing, Checked); 3] = [
        ("add", ux.add(uy), Int::overflowing_add_unsigned, Int::checked_add_unsigned),
        ("sub", ux.sub(uy), Int::overflowing_sub_unsigned, Int::checked_sub_unsigned),
        ("mul", ux.mul(uy), Int::overflowing_mul_unsigned, Int::checked_mul_unsigned),
    ];
    for (name, exact, overflowing, checked) in unsigned {
        let out = unsigned_out_of_range(w, exact);
        assert_eq!(overflowing(x, y), (reduce(w, exact), out), "{name} unsigned {x:?} {y:?}");
        assert_eq!(checked(x, y), (!out).then(|| reduce(w, exact)), "{name} unsigned {x:?} {y:?}");
    }

    // Division
    if uy == Big::ZERO {
        assert_eq!(x.div_unsigned(y), None);
        assert_eq!(x.rem_unsigned(y), None);
        assert_eq!(x.div_signed(y), None);
        assert_eq!(x.rem_signed(y), None);
    } else {
        let (q, r) = ux.divrem_unsigned(uy);
        assert_eq!(x.div_unsigned(y), Some(reduce(w, q)), "div_unsigned {x:?} {y:?}");
        assert_eq!(x.rem_unsigned(y), Some(reduce(w, r)), "rem_unsigned {x:?} {y:?}");
        let (q, r) = sx.divrem_signed(sy);
        // The quotient is out of range only for MIN / -1
        let out = signed_out_of_range(w, q);
        assert_eq!(out, sx == Big::pow2(w - 1).neg() && sy == Big::ONE.neg());
        assert_eq!(x.div_signed(y), (!out).then(|| reduce(w, q)), "div_signed {x:?} {y:?}");
        assert_eq!(x.rem_signed(y), Some(reduce(w, r)), "rem_signed {x:?} {y:?}");
        // q * y + r == x, exactly
        assert_eq!(q.mul(sy).add(r), sx);
    }

    // Bitwise
    assert_eq!(x & y, reduce(w, ux.and(uy)));
    assert_eq!(x | y, reduce(w, ux.or(uy)));
    assert_eq!(x ^ y, reduce(w, ux.xor(uy)));

    // Comparison, minimum and maximum
    assert_eq!(x.cmp_signed(y), sx.cmp_signed(sy), "cmp_signed {x:?} {y:?}");
    assert_eq!(x.cmp_unsigned(y), ux.cmp_unsigned(uy), "cmp_unsigned {x:?} {y:?}");
    let (lo, hi) = if sx.cmp_signed(sy) == Ordering::Greater {
        (y, x)
    } else {
        (x, y)
    };
    assert_eq!((x.min_signed(y), x.max_signed(y)), (lo, hi), "signed min/max {x:?} {y:?}");
    let (lo, hi) = if ux.cmp_unsigned(uy) == Ordering::Greater {
        (y, x)
    } else {
        (x, y)
    };
    assert_eq!((x.min_unsigned(y), x.max_unsigned(y)), (lo, hi), "unsigned min/max {x:?} {y:?}");
}

fn check_resize(x: Int, to: u32) {
    let w = x.width();
    if to >= w {
        assert_eq!(x.zext(to), reduce(to, unsigned_of(x)));
        assert_eq!(x.sext(to), reduce(to, signed_of(x)));
    }
    if to <= w {
        assert_eq!(x.trunc(to), reduce(to, unsigned_of(x)));
    }
}

fn run(seed: u64, rounds: u32) {
    let mut rng = Rng(seed);
    for _ in 0..rounds {
        let w = rng.width();
        let (x, y) = (rng.value(w), rng.value(w));
        check_unary(x);
        check_binary(x, y);
        check_binary(y, x);
        check_binary(x, x);
        check_shift(x, rng.amount(w));
        check_shift(x, rng.amount(w));
        check_resize(x, rng.width());
    }
}

/// Every arithmetic method of `Int`, at random widths (a quarter of them from the edge list, a
/// quarter 128) and edge-biased values, against the model.
#[test]
fn int_arithmetic_agrees_with_a_big_integer_model() {
    run(0x5eed_0003, 2_000);
}

/// The widths the plan names, each on its own with every pair of edge values and random ones.
#[test]
fn int_arithmetic_at_the_named_widths() {
    let mut rng = Rng(0x5eed_0004);
    for w in [1u32, 2, 63, 64, 65, 127, 128] {
        let all = Int::all_ones(w);
        let min = Int::signed_min(w);
        let mut values = alloc::vec![
            Int::zero(w),
            Int::from_bits(w, 1),
            Int::from_bits(w, 2),
            all,
            all.wrapping_sub(Int::from_bits(w, 1)),
            min,
            min.wrapping_add(Int::from_bits(w, 1)),
            Int::signed_max(w),
            Int::signed_max(w).wrapping_sub(Int::from_bits(w, 1)),
            Int::from_bits(w, 1u128 << (w / 2)),
            Int::from_bits(w, (1u128 << (w / 2)).wrapping_sub(1)),
            Int::from_bits(w, 1u128 << (w.div_ceil(2) - 1).min(w - 1)),
        ];
        for _ in 0..30 {
            values.push(rng.value(w));
        }
        for &x in &values {
            check_unary(x);
            for amount in [0, 1, w / 2, w - 1, w, w + 1, 2 * w, 2 * w + 1, 127, 128, 129, u32::MAX]
            {
                check_shift(x, amount);
            }
            for to in [1u32, 2, 63, 64, 65, 127, 128] {
                check_resize(x, to);
            }
            for &y in &values {
                check_binary(x, y);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// fold_isize, against its definition
// ---------------------------------------------------------------------------------------------

type Op = (&'static str, IsizeFold, fn([Int; 2]) -> Option<Int>);

fn amount_of(x: Int) -> u32 {
    x.shift_amount_unsigned()
}

fn count(x: Int, n: u32) -> Option<Int> {
    Some(Int::from_bits(x.width(), n as u128))
}

/// Operations written the way an `arith` fold would write them: on operands of whatever width
/// they are handed.
const OPS: &[Op] = &[
    ("add", IsizeFold::PointerWidth, |[a, b]| Some(a.wrapping_add(b))),
    ("sub", IsizeFold::PointerWidth, |[a, b]| Some(a.wrapping_sub(b))),
    ("mul", IsizeFold::PointerWidth, |[a, b]| Some(a.wrapping_mul(b))),
    ("and", IsizeFold::PointerWidth, |[a, b]| Some(a & b)),
    ("or", IsizeFold::PointerWidth, |[a, b]| Some(a | b)),
    ("xor", IsizeFold::PointerWidth, |[a, b]| Some(a ^ b)),
    ("udiv", IsizeFold::PointerWidth, |[a, b]| a.div_unsigned(b)),
    ("urem", IsizeFold::PointerWidth, |[a, b]| a.rem_unsigned(b)),
    ("sdiv", IsizeFold::PointerWidth, |[a, b]| a.div_signed(b)),
    ("srem", IsizeFold::PointerWidth, |[a, b]| a.rem_signed(b)),
    ("add checked signed", IsizeFold::PointerWidth, |[a, b]| a.checked_add_signed(b)),
    ("add checked unsigned", IsizeFold::PointerWidth, |[a, b]| {
        a.checked_add_unsigned(b)
    }),
    ("mul checked signed", IsizeFold::PointerWidth, |[a, b]| a.checked_mul_signed(b)),
    ("mul checked unsigned", IsizeFold::PointerWidth, |[a, b]| {
        a.checked_mul_unsigned(b)
    }),
    ("shl", IsizeFold::PointerWidth, |[a, b]| Some(a << amount_of(b))),
    ("lshr", IsizeFold::PointerWidth, |[a, b]| Some(a.lshr(amount_of(b)))),
    ("ashr", IsizeFold::PointerWidth, |[a, b]| Some(a.ashr(amount_of(b)))),
    ("rotl", IsizeFold::PointerWidth, |[a, b]| {
        Some(a.rotl(b.rotate_amount_unsigned(a.width())))
    }),
    ("rotr", IsizeFold::PointerWidth, |[a, b]| {
        Some(a.rotr(b.rotate_amount_unsigned(a.width())))
    }),
    ("smin", IsizeFold::PointerWidth, |[a, b]| Some(a.min_signed(b))),
    ("umax", IsizeFold::PointerWidth, |[a, b]| Some(a.max_unsigned(b))),
    ("popcnt", IsizeFold::PointerWidth, |[a, _]| count(a, a.count_ones())),
    ("clz", IsizeFold::PointerWidth, |[a, _]| count(a, a.leading_zeros())),
    ("ctz", IsizeFold::PointerWidth, |[a, _]| count(a, a.trailing_zeros())),
    ("eq", IsizeFold::Fixed(1), |[a, b]| Some(Int::from(a == b))),
    ("slt", IsizeFold::Fixed(1), |[a, b]| Some(Int::from(a.cmp_signed(b).is_lt()))),
    ("ult", IsizeFold::Fixed(1), |[a, b]| Some(Int::from(a.cmp_unsigned(b).is_lt()))),
    ("add overflows signed", IsizeFold::Fixed(1), |[a, b]| {
        Some(Int::from(a.overflowing_add_signed(b).1))
    }),
    ("mul overflows unsigned", IsizeFold::Fixed(1), |[a, b]| {
        Some(Int::from(a.overflowing_mul_unsigned(b).1))
    }),
    ("zext to i64", IsizeFold::Fixed(64), |[a, _]| Some(a.zext(64))),
    ("sext to i64", IsizeFold::Fixed(64), |[a, _]| Some(a.sext(64))),
    ("sext to i128", IsizeFold::Fixed(128), |[a, _]| Some(a.sext(128))),
    ("trunc to i32", IsizeFold::Fixed(32), |[a, _]| Some(a.trunc(32))),
    ("trunc to i8", IsizeFold::Fixed(8), |[a, _]| Some(a.trunc(8))),
];

/// An `isize` constant: a 64-bit value, biased toward the ones that differ between targets.
fn isize_value(rng: &mut Rng) -> Int {
    let bits = match rng.below(12) {
        0 => 0,
        1 => 1,
        2 => u64::MAX,
        3 => 1 << 32,
        4 => (1 << 32) - 1,
        5 => 1 << 31,
        6 => (1 << 31) - 1,
        7 => i64::MIN as u64,
        8 => rng.next() & 0xffff_ffff,
        9 => rng.next() & 0x3f,
        10 => (rng.next() & 0xffff_ffff) | 0xffff_ffff_0000_0000,
        _ => rng.next(),
    };
    Int::from(bits)
}

/// `fold_isize` folds exactly when a 64-bit and a 32-bit target compute the same thing, and then
/// returns the 64-bit target's result.
///
/// The two targets are evaluated here directly: the 64-bit one on the operands, the 32-bit one on
/// their low 32 bits. For a pointer-width result "the same thing" is: the 64-bit result, held as
/// the `isize` constant it would become, reads on a 32-bit target (its low 32 bits) as the 32-bit
/// result. For a result of a fixed width it is equality.
#[test]
fn fold_isize_folds_exactly_when_the_two_targets_agree() {
    let mut rng = Rng(0x5eed_0007);
    let mut folded = 0u32;
    let mut refused = 0u32;
    for _ in 0..20_000 {
        let operands = [isize_value(&mut rng), isize_value(&mut rng)];
        for &(name, kind, op) in OPS {
            let on_64 = op(operands);
            let on_32 = op(operands.map(|operand| operand.trunc(32)));
            let expected = match (on_64, on_32) {
                (Some(wide), Some(narrow)) => {
                    let agree = match kind {
                        IsizeFold::PointerWidth => {
                            assert_eq!((wide.width(), narrow.width()), (64, 32), "{name}");
                            wide.trunc(32) == narrow
                        }
                        IsizeFold::Fixed(width) => {
                            assert_eq!((wide.width(), narrow.width()), (width, width), "{name}");
                            wide == narrow
                        }
                    };
                    agree.then_some(wide)
                }
                _ => None,
            };
            let actual = fold_isize(operands, kind, op);
            assert_eq!(actual, expected, "{name} {operands:?}");
            if actual.is_some() {
                folded += 1;
            } else {
                refused += 1;
            }
        }
    }
    // Both outcomes are exercised
    assert!(folded > 50_000 && refused > 50_000, "folded {folded}, refused {refused}");
}
