use core::fmt;

use crate::{Context, Type};

/// A floating-point constant: an IEEE 754 binary32 (`f32`) or binary64 (`f64`) value, held as its
/// bit pattern.
///
/// Holding the bits makes every value, a NaN with any payload included, a distinct and exact
/// constant that survives printing, parsing and copying unchanged.
///
/// `Eq`, `Ord` and `Hash` compare the width and then the bits. This is deliberate, and it is not
/// the IEEE 754 comparison: `0.0` and `-0.0` are different values, and a NaN equals itself and
/// orders after the finite values of its sign. Attributes are uniqued by value and constants are
/// the keys of the constant folder's map, so they need an equality that is reflexive and that
/// tells apart values whose behavior differs. A comparison of two floats as numbers is an
/// operation of the `arith` dialect, not this equality.
///
/// Invariant: `width` is 32 or 64, and for 32 every bit of `bits` at or above bit 32 is zero.
/// Every constructor establishes it, and nothing else writes the fields.
///
/// The text form is [fmt::Display]'s, which is what the printer writes (without the type); the
/// parser reads it back to the same bits:
///
/// - A finite value is its shortest decimal digits that read back to the same bits, with a `.`
///   in the mantissa so it is never an integer literal: `1.0`, `-0.0`, `0.0025`, `1.0e100`,
///   `5.0e-324`. The exponent, when there is one, is a lowercase `e` followed by an optionally
///   negative decimal exponent, with no `+` and no leading zeros.
/// - A NaN or an infinity is its bits in hexadecimal with every digit: `0x7ff8000000000000` for
///   an `f64`, `0x7f800000` for an `f32`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Float {
    width: u8,
    bits: u64,
}

impl Float {
    /// The 32-bit float with the value of `value`, bit for bit (a NaN keeps its payload).
    pub const fn from_f32(value: f32) -> Self {
        Self::from_bits32(value.to_bits())
    }

    /// The 64-bit float with the value of `value`, bit for bit (a NaN keeps its payload).
    pub const fn from_f64(value: f64) -> Self {
        Self::from_bits64(value.to_bits())
    }

    /// The 32-bit float whose IEEE 754 binary32 encoding is `bits`.
    pub const fn from_bits32(bits: u32) -> Self {
        Self {
            width: 32,
            bits: bits as u64,
        }
    }

    /// The 64-bit float whose IEEE 754 binary64 encoding is `bits`.
    pub const fn from_bits64(bits: u64) -> Self {
        Self { width: 64, bits }
    }

    /// The width of this float in bits: 32 or 64.
    pub const fn width(self) -> u32 {
        self.width as u32
    }

    /// The encoding of this float, zero-extended to 64 bits for a 32-bit float.
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// The value as an `f32`, exactly as held, or `None` for a 64-bit float.
    pub const fn to_f32(self) -> Option<f32> {
        if self.width == 32 {
            Some(f32::from_bits(self.bits as u32))
        } else {
            None
        }
    }

    /// The value as an `f64`, exactly as held, or `None` for a 32-bit float.
    pub const fn to_f64(self) -> Option<f64> {
        if self.width == 64 {
            Some(f64::from_bits(self.bits))
        } else {
            None
        }
    }

    /// True if this value is a NaN of any payload and sign.
    pub fn is_nan(self) -> bool {
        match (self.to_f32(), self.to_f64()) {
            (Some(value), _) => value.is_nan(),
            (_, Some(value)) => value.is_nan(),
            _ => unreachable!("a float is 32 or 64 bits wide"),
        }
    }

    /// True if this value is neither a NaN nor an infinity.
    pub fn is_finite(self) -> bool {
        match (self.to_f32(), self.to_f64()) {
            (Some(value), _) => value.is_finite(),
            (_, Some(value)) => value.is_finite(),
            _ => unreachable!("a float is 32 or 64 bits wide"),
        }
    }

    /// True if the sign bit is set, for any value including `-0.0` and a NaN.
    pub const fn is_sign_negative(self) -> bool {
        (self.bits >> (self.width as u32 - 1)) & 1 == 1
    }

    /// The type of a constant of this value: `f32` for a 32-bit float, `f64` for a 64-bit one.
    pub fn ty(self, context: &Context) -> Type {
        if self.width == 32 {
            context.f32()
        } else {
            context.f64()
        }
    }

    /// The width in bits of a constant of the type `ty`: 32 for `f32`, 64 for `f64`, and `None`
    /// for any other type.
    ///
    /// This is the one statement of the rule: a float constant of type `ty` holds a `Float` of
    /// exactly this width.
    pub fn width_for_type(ty: Type) -> Option<u32> {
        ty.as_float().map(|float| float.bits())
    }
}

/// The text form of the value, as the printer writes it, without the type.
impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.is_finite() {
            return if self.width == 32 {
                write!(f, "{:#010x}", self.bits)
            } else {
                write!(f, "{:#018x}", self.bits)
            };
        }
        // `{:?}` is the shortest text that reads back to the same bits. It writes a `.` unless it
        // writes an exponent, and then the mantissa may be a bare integer (`1e100`), which gets a
        // `.0` so that the mantissa of every float with an exponent has a point.
        let text = match (self.to_f32(), self.to_f64()) {
            (Some(value), _) => alloc::format!("{value:?}"),
            (_, Some(value)) => alloc::format!("{value:?}"),
            _ => unreachable!("a float is 32 or 64 bits wide"),
        };
        match text.split_once('e') {
            Some((mantissa, exponent)) if !mantissa.contains('.') => {
                write!(f, "{mantissa}.0e{exponent}")
            }
            _ => f.write_str(&text),
        }
    }
}

/// The same text as [fmt::Display].
impl fmt::Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, rc::Rc, string::ToString, vec::Vec};
    use core::{cmp::Ordering, hash::Hasher};

    use super::*;

    #[test]
    fn a_float_holds_its_width_and_bits() {
        let single = Float::from_f32(1.5);
        assert_eq!((single.width(), single.bits()), (32, 0x3fc0_0000));
        let double = Float::from_f64(1.5);
        assert_eq!((double.width(), double.bits()), (64, 0x3ff8_0000_0000_0000));
        assert_eq!(Float::from_bits32(0x3fc0_0000), single);
        assert_eq!(Float::from_bits64(0x3ff8_0000_0000_0000), double);
    }

    #[test]
    fn a_float_converts_only_to_its_own_width() {
        let single = Float::from_f32(0.25);
        assert_eq!((single.to_f32(), single.to_f64()), (Some(0.25), None));
        let double = Float::from_f64(0.25);
        assert_eq!((double.to_f32(), double.to_f64()), (None, Some(0.25)));
    }

    #[test]
    fn a_nan_converts_with_its_payload() {
        let bits = 0xfff0_0000_0000_1234u64;
        let nan = Float::from_bits64(bits);
        assert_eq!(nan.to_f64().unwrap().to_bits(), bits);
        let bits = 0xffc0_1234u32;
        assert_eq!(Float::from_bits32(bits).to_f32().unwrap().to_bits(), bits);
    }

    #[test]
    fn classification_follows_the_value() {
        for float in [Float::from_f32(f32::NAN), Float::from_f64(f64::NAN)] {
            assert!(float.is_nan() && !float.is_finite());
        }
        for float in [Float::from_f32(f32::INFINITY), Float::from_f64(f64::NEG_INFINITY)] {
            assert!(!float.is_nan() && !float.is_finite());
        }
        for float in [Float::from_f32(1.0), Float::from_f64(-0.0), Float::from_bits64(1)] {
            assert!(!float.is_nan() && float.is_finite());
        }
        assert!(Float::from_f32(-0.0).is_sign_negative());
        assert!(Float::from_f64(-1.0).is_sign_negative());
        assert!(Float::from_bits32(0xffc0_0000).is_sign_negative());
        assert!(!Float::from_f64(0.0).is_sign_negative());
        assert!(!Float::from_f32(f32::INFINITY).is_sign_negative());
    }

    #[test]
    fn the_type_of_a_float_follows_its_width() {
        let context = Rc::new(Context::default());
        assert_eq!(Float::from_f32(1.0).ty(&context), context.f32());
        assert_eq!(Float::from_f64(1.0).ty(&context), context.f64());
        assert_eq!(Float::width_for_type(context.f32()), Some(32));
        assert_eq!(Float::width_for_type(context.f64()), Some(64));
        assert_eq!(Float::width_for_type(context.i32()), None);
        assert_eq!(Float::width_for_type(context.unknown_type()), None);
    }

    #[test]
    fn equality_is_by_width_and_bits() {
        assert_ne!(Float::from_f64(0.0), Float::from_f64(-0.0));
        assert_ne!(Float::from_f32(0.0), Float::from_f64(0.0));
        let nan = Float::from_f64(f64::NAN);
        assert_eq!(nan, nan);
        assert_ne!(nan, Float::from_bits64(nan.bits() | 1));
        assert_eq!(Float::from_f32(1.0), Float::from_bits32(0x3f80_0000));
    }

    #[test]
    fn ordering_is_by_width_then_bits() {
        assert_eq!(Float::from_f64(0.0).cmp(&Float::from_f64(-0.0)), Ordering::Less);
        assert_eq!(Float::from_f32(1.0).cmp(&Float::from_f32(1.0)), Ordering::Equal);
        assert_eq!(Float::from_f32(1.0).cmp(&Float::from_f64(0.0)), Ordering::Less);
        let nan = Float::from_f64(f64::NAN);
        assert_eq!(nan.cmp(&nan), Ordering::Equal);
        let mut sorted = [Float::from_f64(-0.0), Float::from_f64(0.0), Float::from_f32(1.0)];
        sorted.sort();
        assert_eq!(sorted, [Float::from_f32(1.0), Float::from_f64(0.0), Float::from_f64(-0.0)]);
    }

    /// A hasher that records its input, so two hashes can be compared without `std`.
    #[derive(Default)]
    struct Recorder(Vec<u8>);

    impl Hasher for Recorder {
        fn finish(&self) -> u64 {
            0
        }

        fn write(&mut self, bytes: &[u8]) {
            self.0.extend_from_slice(bytes);
        }
    }

    fn hash_of(float: Float) -> Vec<u8> {
        use core::hash::Hash;

        let mut recorder = Recorder::default();
        float.hash(&mut recorder);
        recorder.0
    }

    #[test]
    fn the_hash_is_by_width_and_bits() {
        assert_eq!(hash_of(Float::from_f64(f64::NAN)), hash_of(Float::from_f64(f64::NAN)));
        assert_eq!(hash_of(Float::from_f32(1.0)), hash_of(Float::from_bits32(0x3f80_0000)));
        assert_ne!(hash_of(Float::from_f64(0.0)), hash_of(Float::from_f64(-0.0)));
        assert_ne!(hash_of(Float::from_f32(0.0)), hash_of(Float::from_f64(0.0)));
    }

    #[test]
    fn the_text_form_is_decimal_with_a_point_for_a_finite_value() {
        for (float, text) in [
            (Float::from_f64(1.0), "1.0"),
            (Float::from_f64(-0.0), "-0.0"),
            (Float::from_f64(0.0), "0.0"),
            (Float::from_f64(1.5), "1.5"),
            (Float::from_f64(2.5e-3), "0.0025"),
            (Float::from_f64(1.0e100), "1.0e100"),
            (Float::from_f64(1.5e-7), "1.5e-7"),
            (Float::from_f64(-2.5e300), "-2.5e300"),
            (Float::from_f64(f64::from_bits(1)), "5.0e-324"),
            (Float::from_f64(f64::MAX), "1.7976931348623157e308"),
            (Float::from_f32(1.0), "1.0"),
            (Float::from_f32(0.1), "0.1"),
            (Float::from_f32(f32::from_bits(1)), "1.0e-45"),
            (Float::from_f32(f32::MAX), "3.4028235e38"),
        ] {
            assert_eq!(float.to_string(), text);
            assert_eq!(format!("{float:?}"), text);
        }
    }

    #[test]
    fn the_text_form_is_every_hex_digit_of_the_bits_for_a_nan_or_an_infinity() {
        for (float, text) in [
            (Float::from_f64(f64::NAN), "0x7ff8000000000000"),
            (Float::from_f64(f64::INFINITY), "0x7ff0000000000000"),
            (Float::from_f64(f64::NEG_INFINITY), "0xfff0000000000000"),
            (Float::from_f32(f32::INFINITY), "0x7f800000"),
            (Float::from_f32(f32::NEG_INFINITY), "0xff800000"),
            (Float::from_bits32(0x7fc0_0001), "0x7fc00001"),
            (Float::from_bits64(0x7ff0_0000_0000_0001), "0x7ff0000000000001"),
        ] {
            assert_eq!(float.to_string(), text);
        }
    }
}
