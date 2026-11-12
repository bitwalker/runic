use core::{cmp::Ordering, fmt};

/// An integer of a fixed width between 1 and 128 bits.
///
/// `Int` is signless: it is a bit pattern, and carries no interpretation as a signed or an unsigned
/// number. The operations whose result depends on the interpretation say which one in their name
/// (`signed`, `unsigned`, `sext`, `zext`), and there is deliberately no `Ord`.
///
/// Invariant: `1 <= width <= 128`, and every bit of `bits` at or above `width` is zero. Every
/// constructor establishes it, and nothing else writes the fields.
///
/// Two values are equal only if their widths and their bits are.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Int {
    bits: u128,
    width: u8,
}

/// The mask of the low `width` bits. `width` is between 1 and 128.
const fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

const fn check_width(width: u32) {
    assert!(
        width >= 1 && width <= Int::MAX_WIDTH,
        "invalid integer width: an Int is 1 to 128 bits wide"
    );
}

impl Int {
    /// The widest integer value, in bits.
    pub const MAX_WIDTH: u32 = 128;

    /// The `width`-bit integer whose bits are the low `width` bits of `bits`; the rest are dropped.
    ///
    /// Panics if `width` is 0 or more than [Self::MAX_WIDTH].
    pub const fn from_bits(width: u32, bits: u128) -> Self {
        check_width(width);
        Self {
            bits: bits & mask(width),
            width: width as u8,
        }
    }

    /// The `width`-bit integer with the unsigned value `value`, or `None` if `value >= 2^width`.
    ///
    /// Panics if `width` is 0 or more than [Self::MAX_WIDTH].
    pub const fn from_unsigned(width: u32, value: u128) -> Option<Self> {
        check_width(width);
        if value > mask(width) {
            None
        } else {
            Some(Self::from_bits(width, value))
        }
    }

    /// The `width`-bit integer with the signed value `value`, or `None` if `value` is outside
    /// `[-2^(width-1), 2^(width-1))`.
    ///
    /// Panics if `width` is 0 or more than [Self::MAX_WIDTH].
    pub const fn from_signed(width: u32, value: i128) -> Option<Self> {
        check_width(width);
        if width < 128 {
            let limit = 1i128 << (width - 1);
            if value < -limit || value >= limit {
                return None;
            }
        }
        Some(Self::from_bits(width, value as u128))
    }

    /// The `width`-bit integer with every bit clear.
    ///
    /// Panics if `width` is 0 or more than [Self::MAX_WIDTH].
    pub const fn zero(width: u32) -> Self {
        Self::from_bits(width, 0)
    }

    /// The `width`-bit integer with every bit set.
    ///
    /// Panics if `width` is 0 or more than [Self::MAX_WIDTH].
    pub const fn all_ones(width: u32) -> Self {
        Self::from_bits(width, u128::MAX)
    }

    /// The width in bits of a constant of the type `ty`: `N` for `iN` with `N <= 128`, 64 for
    /// `isize`, and `None` for any other type (a wider integer type has no constants).
    ///
    /// This is the one statement of the rule: an integer constant of type `ty` holds an `Int` of
    /// exactly this width.
    pub fn width_for_type(ty: crate::Type) -> Option<u32> {
        match ty.integer_width() {
            Some(width) => (width <= Self::MAX_WIDTH).then_some(width),
            None => ty.is_isize().then_some(64),
        }
    }

    /// The width of this integer in bits.
    pub const fn width(self) -> u32 {
        self.width as u32
    }

    /// The bits as an unsigned number: the value, zero-extended.
    pub const fn unsigned(self) -> u128 {
        self.bits
    }

    /// The bits as a two's complement number: the value, sign-extended from [Self::width].
    pub const fn signed(self) -> i128 {
        let shift = 128 - self.width as u32;
        ((self.bits << shift) as i128) >> shift
    }

    /// The unsigned value as a `T`, or `None` if it does not fit.
    pub fn try_unsigned<T: TryFrom<u128>>(self) -> Option<T> {
        T::try_from(self.unsigned()).ok()
    }

    /// The signed value as a `T`, or `None` if it does not fit.
    pub fn try_signed<T: TryFrom<i128>>(self) -> Option<T> {
        T::try_from(self.signed()).ok()
    }

    /// Whether every bit is clear.
    pub const fn is_zero(self) -> bool {
        self.bits == 0
    }

    /// Whether the value is 1: the lowest bit is set and no other.
    pub const fn is_one(self) -> bool {
        self.bits == 1
    }

    /// Whether every bit is set.
    pub const fn is_all_ones(self) -> bool {
        self.bits == mask(self.width as u32)
    }

    /// Whether the sign bit (the highest bit) is set.
    pub const fn is_negative(self) -> bool {
        (self.bits >> (self.width as u32 - 1)) & 1 == 1
    }

    /// This value zero-extended to `width` bits.
    ///
    /// Panics if `width < self.width()` or `width` is more than [Self::MAX_WIDTH].
    pub const fn zext(self, width: u32) -> Self {
        check_width(width);
        assert!(width >= self.width as u32, "cannot zero-extend to a narrower width");
        Self {
            bits: self.bits,
            width: width as u8,
        }
    }

    /// This value sign-extended to `width` bits.
    ///
    /// Panics if `width < self.width()` or `width` is more than [Self::MAX_WIDTH].
    pub const fn sext(self, width: u32) -> Self {
        check_width(width);
        assert!(width >= self.width as u32, "cannot sign-extend to a narrower width");
        Self::from_bits(width, self.signed() as u128)
    }

    /// This value truncated to its low `width` bits.
    ///
    /// Panics if `width > self.width()` or `width` is 0.
    pub const fn trunc(self, width: u32) -> Self {
        check_width(width);
        assert!(width <= self.width as u32, "cannot truncate to a wider width");
        Self::from_bits(width, self.bits)
    }

    /// Compares the values as unsigned numbers. Panics if the widths differ.
    pub fn cmp_unsigned(self, other: Self) -> Ordering {
        assert_eq!(self.width, other.width, "cannot compare integers of different widths");
        self.bits.cmp(&other.bits)
    }

    /// Compares the values as two's complement numbers. Panics if the widths differ.
    pub fn cmp_signed(self, other: Self) -> Ordering {
        assert_eq!(self.width, other.width, "cannot compare integers of different widths");
        self.signed().cmp(&other.signed())
    }
}

impl From<bool> for Int {
    fn from(value: bool) -> Self {
        Self::from_bits(1, value as u128)
    }
}

macro_rules! from_primitive {
    ($($ty:ty => $width:expr),* $(,)?) => {
        $(
            impl From<$ty> for Int {
                fn from(value: $ty) -> Self {
                    Self::from_bits($width, value as u128)
                }
            }
        )*
    };
}

from_primitive! {
    u8 => 8, i8 => 8, u16 => 16, i16 => 16, u32 => 32, i32 => 32,
    u64 => 64, i64 => 64, u128 => 128, i128 => 128,
}

/// The signed decimal at the width; an `i1` is `0` or `1`.
impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.width == 1 {
            fmt::Display::fmt(&self.bits, f)
        } else {
            fmt::Display::fmt(&self.signed(), f)
        }
    }
}

/// The display form followed by the width, such as `-1i8`.
impl fmt::Debug for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}i{}", self.width)
    }
}

impl fmt::LowerHex for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.bits, f)
    }
}

impl fmt::UpperHex for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.bits, f)
    }
}

impl fmt::Binary for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&self.bits, f)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, string::ToString};

    use super::*;

    fn pattern_signed(w: u32, b: u128) -> i128 {
        if b >> (w - 1) == 1 {
            b as i128 - (1i128 << w)
        } else {
            b as i128
        }
    }

    #[test]
    fn bits_round_trip_for_small_widths() {
        for w in 1..=10u32 {
            for b in 0..(1u128 << w) {
                let x = Int::from_bits(w, b);
                assert_eq!(x.width(), w);
                assert_eq!(x.unsigned(), b);
                assert_eq!(x.signed(), pattern_signed(w, b));
                assert_eq!(Int::from_signed(w, x.signed()), Some(x));
                assert_eq!(Int::from_unsigned(w, x.unsigned()), Some(x));
                assert_eq!(x.is_negative(), b >> (w - 1) == 1);
            }
        }
    }

    #[test]
    fn range_checks_of_the_checked_constructors() {
        for w in 1..=8u32 {
            for v in -300i128..=300 {
                let in_signed = -(1i128 << (w - 1)) <= v && v < (1i128 << (w - 1));
                assert_eq!(Int::from_signed(w, v).is_some(), in_signed, "signed {v} in {w}");
                if v >= 0 {
                    let in_unsigned = v < (1i128 << w);
                    assert_eq!(
                        Int::from_unsigned(w, v as u128).is_some(),
                        in_unsigned,
                        "unsigned {v} in {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn from_bits_truncates() {
        assert_eq!(Int::from_bits(8, 0x1ff).unsigned(), 0xff);
    }

    #[test]
    fn wide_widths() {
        assert_eq!(Int::from_bits(128, u128::MAX).signed(), -1);
        assert_eq!(Int::from_signed(128, i128::MIN).unwrap().unsigned(), 1u128 << 127);
        assert!(Int::from_unsigned(128, u128::MAX).is_some());
        assert!(Int::from_signed(127, i128::MIN).is_none());
        assert_eq!(Int::all_ones(127).signed(), -1);
        assert_eq!(Int::all_ones(128).unsigned(), u128::MAX);
        assert_eq!(Int::all_ones(127).unsigned(), u128::MAX >> 1);
    }

    #[test]
    fn edges_at_127_bits() {
        assert_eq!(Int::from_signed(127, 1 << 126), None);
        let min = Int::from_signed(127, -(1 << 126)).unwrap();
        assert_eq!(min.signed(), -(1 << 126));
        assert_eq!(Int::from_unsigned(127, 1 << 127), None);
        assert_eq!(Int::from_bits(127, 1 << 126).signed(), -(1 << 126));
    }

    #[test]
    fn resizing() {
        for w1 in 1..=9u32 {
            for w2 in w1..=9u32 {
                for b in 0..(1u128 << w1) {
                    let x = Int::from_bits(w1, b);
                    assert_eq!(x.zext(w2).unsigned(), x.unsigned());
                    assert_eq!(x.zext(w2).width(), w2);
                    assert_eq!(x.sext(w2).signed(), x.signed());
                    assert_eq!(x.sext(w2).width(), w2);
                    assert_eq!(x.zext(w2).trunc(w1), x);
                    assert_eq!(x.sext(w2).trunc(w1), x);
                    assert_eq!(x.sext(128).signed(), x.signed());
                    assert_eq!(x.zext(128).unsigned(), x.unsigned());
                }
            }
            for b in 0..(1u128 << w1) {
                let x = Int::from_bits(w1, b);
                assert_eq!(x.zext(w1), x);
                assert_eq!(x.sext(w1), x);
                assert_eq!(x.trunc(w1), x);
            }
        }
        let x = Int::from_bits(128, u128::MAX);
        assert_eq!(x.sext(128), x);
        assert_eq!(x.trunc(1), Int::from_bits(1, 1));
    }

    #[test]
    #[should_panic]
    fn zero_width_panics() {
        Int::from_bits(0, 0);
    }

    #[test]
    #[should_panic]
    fn too_wide_panics() {
        Int::from_bits(129, 0);
    }

    #[test]
    #[should_panic]
    fn zext_to_a_narrower_width_panics() {
        Int::from(1u16).zext(8);
    }

    #[test]
    #[should_panic]
    fn sext_to_a_narrower_width_panics() {
        Int::from(1u16).sext(8);
    }

    #[test]
    #[should_panic]
    fn trunc_to_a_wider_width_panics() {
        Int::from(1u8).trunc(16);
    }

    #[test]
    #[should_panic]
    fn comparing_different_widths_panics() {
        Int::from(1u8).cmp_signed(Int::from(1u16));
    }

    #[test]
    #[should_panic]
    fn comparing_unsigned_different_widths_panics() {
        Int::from(1u8).cmp_unsigned(Int::from(1u16));
    }

    #[test]
    fn comparison() {
        for w in 1..=6u32 {
            for a in 0..(1u128 << w) {
                for b in 0..(1u128 << w) {
                    let (x, y) = (Int::from_bits(w, a), Int::from_bits(w, b));
                    assert_eq!(x.cmp_unsigned(y), x.unsigned().cmp(&y.unsigned()));
                    assert_eq!(x.cmp_signed(y), x.signed().cmp(&y.signed()));
                }
            }
        }
    }

    #[test]
    fn formatting() {
        for w in 2..=9u32 {
            for b in 0..(1u128 << w) {
                let x = Int::from_bits(w, b);
                assert_eq!(format!("{x}"), x.signed().to_string());
            }
        }
        assert_eq!(format!("{}", Int::from_bits(1, 0)), "0");
        assert_eq!(format!("{}", Int::from_bits(1, 1)), "1");
        assert_eq!(format!("{}", Int::from(i128::MIN)), i128::MIN.to_string());
        assert_eq!(format!("{:#x}", Int::from(-1i8)), "0xff");
        assert_eq!(format!("{:x}", Int::from(255u16)), "ff");
        assert_eq!(format!("{:X}", Int::from(255u16)), "FF");
        assert_eq!(format!("{:b}", Int::from_bits(3, 5)), "101");
        assert_eq!(format!("{:?}", Int::from(-1i8)), "-1i8");
        assert_eq!(format!("{:?}", Int::from(true)), "1i1");
    }

    #[test]
    fn conversions() {
        assert_eq!(Int::from(-1i8), Int::from(255u8));
        assert_ne!(Int::from(1u8), Int::from(1u16));
        assert_eq!(Int::from(true), Int::from_bits(1, 1));
        assert_eq!(Int::from(false).width(), 1);
        assert_eq!(Int::from(0u8).width(), 8);
        assert_eq!(Int::from(0i8).width(), 8);
        assert_eq!(Int::from(0u16).width(), 16);
        assert_eq!(Int::from(0i16).width(), 16);
        assert_eq!(Int::from(0u32).width(), 32);
        assert_eq!(Int::from(0i32).width(), 32);
        assert_eq!(Int::from(0u64).width(), 64);
        assert_eq!(Int::from(0i64).width(), 64);
        assert_eq!(Int::from(0u128).width(), 128);
        assert_eq!(Int::from(0i128).width(), 128);
        assert_eq!(Int::from(-1i32).try_unsigned::<u32>(), Some(u32::MAX));
        assert_eq!(Int::from(-1i32).try_signed::<i8>(), Some(-1));
        assert_eq!(Int::from(300u16).try_unsigned::<u8>(), None);
        assert_eq!(Int::from(-1i8).try_signed::<u8>(), None);
    }

    #[test]
    fn predicates() {
        for w in 1..=128u32 {
            assert!(Int::zero(w).is_zero());
            assert!(Int::all_ones(w).is_all_ones());
            assert!(Int::from_bits(w, 1).is_one());
            assert!(!Int::zero(w).is_one());
            assert!(!Int::all_ones(w).is_zero());
        }
        let one = Int::from_bits(1, 1);
        assert!(one.is_one() && one.is_all_ones());
        assert!(!Int::from_bits(2, 1).is_all_ones());
    }

    #[test]
    fn width_for_type() {
        use crate::Context;

        let context = Context::default();
        for w in [1u32, 7, 8, 32, 64, 127, 128] {
            assert_eq!(Int::width_for_type(context.int_type(w)), Some(w), "i{w}");
        }
        assert_eq!(Int::width_for_type(context.isize()), Some(64));
        assert_eq!(Int::width_for_type(context.int_type(129)), None);
        assert_eq!(Int::width_for_type(context.int_type(256)), None);
        assert_eq!(Int::width_for_type(context.opaque_ptr_type()), None);
        assert_eq!(Int::width_for_type(context.unknown_type()), None);
        assert_eq!(Int::width_for_type(context.f64()), None);
    }
}
