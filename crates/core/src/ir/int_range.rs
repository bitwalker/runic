use core::fmt;

use super::{Int, Type};

/// Bounds on an integer value, tracked using both signed and unsigned interpretations.
///
/// Unsigned bounds describe the stored raw representation: an N-bit integer, including a signed
/// integer, is zero-extended to its storage width. Signed bounds interpret the same bit pattern
/// at `width`; they are only meaningful for signed ordering at that width.
///
/// This mirrors MLIR's `ConstantIntRanges`, but is intentionally bounded to the integer widths an
/// [Int] can hold (1 to 128 bits). Values with unsupported or non-integer types are represented by
/// `width == 0`.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct ConstantIntRange {
    width: u16,
    umin: u128,
    umax: u128,
    smin: i128,
    smax: i128,
}

impl ConstantIntRange {
    /// Return an untracked/non-integer range.
    #[inline]
    pub const fn untracked() -> Self {
        Self {
            width: 0,
            umin: 0,
            umax: 0,
            smin: 0,
            smax: 0,
        }
    }

    /// Return true if this range represents a value that cannot be tracked precisely.
    #[inline]
    pub const fn is_untracked(&self) -> bool {
        self.width == 0
    }

    /// Return the storage width used for this range, or 0 if untracked.
    #[inline]
    pub const fn width(&self) -> u16 {
        self.width
    }

    /// The minimum unsigned bound.
    #[inline]
    pub const fn umin(&self) -> u128 {
        self.umin
    }

    /// The maximum unsigned bound.
    #[inline]
    pub const fn umax(&self) -> u128 {
        self.umax
    }

    /// The minimum signed bound.
    #[inline]
    pub const fn smin(&self) -> i128 {
        self.smin
    }

    /// The maximum signed bound.
    #[inline]
    pub const fn smax(&self) -> i128 {
        self.smax
    }

    /// Return the conservative range for all valid values of `ty`.
    ///
    /// Integer types are signless, so every bit pattern of the width is valid: neither the
    /// signed nor the unsigned bounds are constrained. Integers wider than 128 bits, `isize`,
    /// and non-integer types (including pointers and functions) are untracked.
    pub fn max_for_type(ty: Type) -> Self {
        match ty.integer_width() {
            Some(width @ 1..=128) => {
                let width = width as u16;
                Self::from_unsigned(width, 0, unsigned_max(width))
            }
            _ => Self::untracked(),
        }
    }

    /// Return the exact range for an integer value: its bit pattern, from which both the signed
    /// and the unsigned bounds follow.
    ///
    /// This is for a value of a fixed-width type, `iN`: the value's width is the type's. A constant
    /// read from an attribute goes through [Self::for_constant] with the attribute's type, which
    /// knows that an `isize` constant has no exact range.
    pub fn constant(value: Int) -> Self {
        Self::from_unsigned(value.width() as u16, value.unsigned(), value.unsigned())
    }

    /// Return the range of a constant `value` of type `ty`.
    ///
    /// An `isize` constant holds a 64-bit value, but on a 32-bit target it stands for its low 32
    /// bits, so it has no exact range: it is untracked, as [Self::max_for_type] of `isize` is. So is
    /// a constant of a type that is not an integer type of at most 128 bits. Otherwise the range is
    /// exact, as [Self::constant] gives it.
    ///
    /// In a debug build, panics if `value` is not as wide as a constant of `ty`
    /// ([Int::width_for_type]).
    pub fn for_constant(value: Int, ty: Type) -> Self {
        match ty.integer_width() {
            Some(1..=128) => {
                debug_assert_eq!(
                    Int::width_for_type(ty),
                    Some(value.width()),
                    "a constant of type `{ty}` must be as wide as the type"
                );
                Self::constant(value)
            }
            _ => Self::untracked(),
        }
    }

    /// Return an exact boolean in the raw representation of `ty`.
    pub fn exact_bool(ty: Type, value: bool) -> Self {
        let width = Self::max_for_type(ty).width();
        let value = u128::from(value);
        Self::from_unsigned(width, value, value)
    }

    /// Return a range with unsigned semantic bounds.
    pub fn from_unsigned(width: u16, umin: u128, umax: u128) -> Self {
        if width == 0 || width > 128 {
            return Self::untracked();
        }

        let max = unsigned_max(width);
        let umin = umin.min(max);
        let umax = umax.min(max);
        let signed_max = signed_max(width);
        let (smin, smax) = if umax <= signed_max as u128 {
            (umin as i128, umax as i128)
        } else if umin > signed_max as u128 {
            (unsigned_to_signed(width, umin), unsigned_to_signed(width, umax))
        } else {
            (signed_min(width), signed_max)
        };

        Self {
            width,
            umin,
            umax,
            smin,
            smax,
        }
    }

    /// Return a range with signed semantic bounds.
    pub fn from_signed(width: u16, smin: i128, smax: i128) -> Self {
        if width == 0 || width > 128 {
            return Self::untracked();
        }

        let min = signed_min(width);
        let max = signed_max(width);
        let smin = smin.max(min);
        let smax = smax.min(max);
        let (umin, umax) = if smin >= 0 {
            (smin as u128, smax as u128)
        } else if smax < 0 {
            (signed_to_unsigned(width, smin), signed_to_unsigned(width, smax))
        } else {
            (0, unsigned_max(width))
        };

        Self {
            width,
            umin,
            umax,
            smin,
            smax,
        }
    }

    /// Compute the least upper bound of two ranges, zero-extending their raw representations
    /// to the larger width. This changes the signed interpretation without changing raw values.
    pub fn union(&self, other: &Self) -> Self {
        if self.is_untracked() {
            return *self;
        }
        if other.is_untracked() {
            return *other;
        }
        let width = self.width.max(other.width);
        let lhs = self.zero_extended_to(width);
        let rhs = other.zero_extended_to(width);

        Self {
            width,
            umin: lhs.umin.min(rhs.umin),
            umax: lhs.umax.max(rhs.umax),
            smin: lhs.smin.min(rhs.smin),
            smax: lhs.smax.max(rhs.smax),
        }
    }

    /// Interpret a tracked range at a width at least as large as its current width.
    fn zero_extended_to(&self, width: u16) -> Self {
        debug_assert!(!self.is_untracked() && self.width <= width && width <= 128);
        if self.width == width {
            return *self;
        }
        // Every raw value from a smaller representation has a clear sign bit at this width.
        // Keep the unsigned bounds intact, including any redundant bounds left by widening.
        Self {
            width,
            smin: self.umin as i128,
            smax: self.umax as i128,
            ..*self
        }
    }

    /// Widen expanding bounds to the limits of their representation, ensuring that
    /// repeated loop-carried joins reach a fixed point in a bounded number of steps.
    pub fn widen(&self, other: &Self) -> Self {
        let joined = self.union(other);
        if joined.is_untracked() {
            return joined;
        }
        let previous = self.zero_extended_to(joined.width);
        Self {
            width: joined.width,
            umin: if joined.umin < previous.umin {
                0
            } else {
                joined.umin
            },
            umax: if joined.umax > previous.umax {
                unsigned_max(joined.width)
            } else {
                joined.umax
            },
            smin: if joined.smin < previous.smin {
                signed_min(joined.width)
            } else {
                joined.smin
            },
            smax: if joined.smax > previous.smax {
                signed_max(joined.width)
            } else {
                joined.smax
            },
        }
    }

    /// Intersect two ranges, returning `None` for an empty intersection or incompatible widths.
    /// An untracked range imposes no bounds, so it is the identity for intersection.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if self.is_untracked() {
            return Some(*other);
        }
        if other.is_untracked() {
            return Some(*self);
        }
        if self.width != other.width {
            return None;
        }

        let umin = self.umin.max(other.umin);
        let umax = self.umax.min(other.umax);
        let smin = self.smin.max(other.smin);
        let smax = self.smax.min(other.smax);
        if umin > umax || smin > smax {
            return None;
        }

        // Intersect each sign's raw representation separately. Overlapping unsigned and signed
        // bounds alone do not guarantee a common bit pattern.
        let positive = (smax >= 0).then(|| (umin.max(smin.max(0) as u128), umax.min(smax as u128)));
        let negative = (smin < 0).then(|| {
            (
                umin.max(signed_to_unsigned(self.width, smin)),
                umax.min(signed_to_unsigned(self.width, smax.min(-1))),
            )
        });
        [positive, negative]
            .into_iter()
            .flatten()
            .filter(|(min, max)| min <= max)
            .map(|(min, max)| Self::from_unsigned(self.width, min, max))
            .reduce(|lhs, rhs| lhs.union(&rhs))
    }

    /// Infer addition when the raw unsigned representation cannot overflow.
    pub fn add_unsigned_without_overflow(&self, other: &Self) -> Option<Self> {
        if self.is_untracked() || other.is_untracked() || self.width != other.width {
            return None;
        }
        let umin = self.umin.checked_add(other.umin)?;
        let umax = self.umax.checked_add(other.umax)?;
        (umax <= unsigned_max(self.width)).then(|| Self::from_unsigned(self.width, umin, umax))
    }

    /// Infer addition when the signed interpretation cannot overflow.
    pub fn add_signed_without_overflow(&self, other: &Self) -> Option<Self> {
        if self.is_untracked() || other.is_untracked() || self.width != other.width {
            return None;
        }
        let smin = self.smin.checked_add(other.smin)?;
        let smax = self.smax.checked_add(other.smax)?;
        (smin >= signed_min(self.width) && smax <= signed_max(self.width))
            .then(|| Self::from_signed(self.width, smin, smax))
    }

    /// Infer bitwise wrapping addition from either non-overflowing interpretation.
    /// Each interpretation can independently constrain the same resulting bit pattern.
    pub fn add_wrapping(&self, other: &Self) -> Option<Self> {
        match (
            self.add_unsigned_without_overflow(other),
            self.add_signed_without_overflow(other),
        ) {
            (Some(unsigned), Some(signed)) => unsigned.intersection(&signed),
            (range, None) | (None, range) => range,
        }
    }

    /// Return true if every value of this range lies in `other`: each bound, unsigned and signed,
    /// is inside `other`'s. False when either range is untracked or the widths differ.
    pub fn is_within(&self, other: &ConstantIntRange) -> bool {
        if self.is_untracked() || other.is_untracked() || self.width != other.width {
            return false;
        }
        self.umin >= other.umin
            && self.umax <= other.umax
            && self.smin >= other.smin
            && self.smax <= other.smax
    }
}

impl fmt::Debug for ConstantIntRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ConstantIntRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_untracked() {
            f.write_str("untracked")
        } else {
            write!(
                f,
                "u[{}, {}] s[{}, {}] width {}",
                self.umin, self.umax, self.smin, self.smax, self.width
            )
        }
    }
}

/// The integer range lattice value attached to an SSA value.
#[derive(Default, Clone, PartialEq, Eq)]
pub struct IntegerValueRange {
    value: Option<ConstantIntRange>,
}

impl IntegerValueRange {
    /// Create an uninitialized lattice value.
    #[inline]
    pub const fn uninitialized() -> Self {
        Self { value: None }
    }

    /// Create an initialized lattice value from `range`.
    #[inline]
    pub const fn new(range: ConstantIntRange) -> Self {
        Self { value: Some(range) }
    }

    /// Create the top value for the range of `ty`.
    #[inline]
    pub fn max_for_type(ty: Type) -> Self {
        Self::new(ConstantIntRange::max_for_type(ty))
    }

    /// Create an exact range for `value`.
    ///
    /// This is for a value of a fixed-width type, `iN`. A constant read from an attribute goes
    /// through [Self::for_constant] with the attribute's type.
    #[inline]
    pub fn constant(value: Int) -> Self {
        Self::new(ConstantIntRange::constant(value))
    }

    /// Create the range of a constant `value` of type `ty`: an exact range, or, for an
    /// `isize` constant (or one of a type that is not an integer type of at most 128 bits), the
    /// untracked range of every value of the type, as [Self::max_for_type] gives it. See
    /// [ConstantIntRange::for_constant].
    pub fn for_constant(value: Int, ty: Type) -> Self {
        let range = ConstantIntRange::for_constant(value, ty);
        if range.is_untracked() {
            Self::max_for_type(ty)
        } else {
            Self::new(range)
        }
    }

    /// Return true if this lattice value has not been initialized.
    #[inline]
    pub const fn is_uninitialized(&self) -> bool {
        self.value.is_none()
    }

    /// Return the underlying range if initialized.
    #[inline]
    pub const fn range(&self) -> Option<&ConstantIntRange> {
        self.value.as_ref()
    }

    /// Create an exact boolean in the raw representation of `ty`.
    pub fn exact_bool(ty: Type, value: bool) -> Self {
        Self::new(ConstantIntRange::exact_bool(ty, value))
    }

    /// Refine with an explicit runtime-checked range in the result's representation.
    ///
    /// A range of another width says nothing about the value being checked, so the answer is
    /// then the checked range.
    pub fn refined_by(&self, range: &ConstantIntRange) -> Self {
        let Some(current) = self.value else {
            return Self::uninitialized();
        };
        let narrowed = if current.width == range.width {
            current.intersection(range)
        } else {
            None
        };
        // Disjoint inputs trap. Any result that exists still satisfies the checked range.
        Self::new(narrowed.unwrap_or(*range))
    }
}

impl fmt::Debug for IntegerValueRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for IntegerValueRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            None => f.write_str("uninitialized"),
            Some(range) => fmt::Display::fmt(&range, f),
        }
    }
}

#[inline]
const fn unsigned_max(width: u16) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

#[inline]
const fn signed_min(width: u16) -> i128 {
    if width == 0 {
        0
    } else if width >= 128 {
        i128::MIN
    } else {
        -(1i128 << (width - 1))
    }
}

#[inline]
const fn signed_max(width: u16) -> i128 {
    if width == 0 {
        0
    } else if width >= 128 {
        i128::MAX
    } else {
        (1i128 << (width - 1)) - 1
    }
}

#[inline]
const fn unsigned_to_signed(width: u16, value: u128) -> i128 {
    if width >= 128 {
        value as i128
    } else {
        let shift = 128 - width;
        ((value << shift) as i128) >> shift
    }
}

#[inline]
const fn signed_to_unsigned(width: u16, value: i128) -> u128 {
    if width >= 128 || value >= 0 {
        value as u128
    } else {
        (1u128 << width) - value.unsigned_abs()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::Context;

    #[test]
    fn mixed_width_joins_preserve_raw_values() {
        let narrow = ConstantIntRange::constant(Int::from(true));
        let wide = ConstantIntRange::constant(Int::from(7u32));
        assert_eq!(narrow.union(&wide), ConstantIntRange::from_unsigned(32, 1, 7));
        assert_eq!(wide.union(&narrow), ConstantIntRange::from_unsigned(32, 1, 7));

        // The raw representation of -1i8 is 255; widening its interpretation cannot sign-extend it.
        let negative = ConstantIntRange::constant(Int::from(-1i8));
        let zero = ConstantIntRange::constant(Int::from(0u16));
        assert_eq!(negative.union(&zero), ConstantIntRange::from_unsigned(16, 0, 255));

        // Keep the wider input's signed bounds, even when its unsigned interval crosses the sign bit.
        let signed = ConstantIntRange::from_signed(16, -5, 5);
        assert_eq!(narrow.union(&signed), signed);
        assert_eq!(narrow.union(&ConstantIntRange::untracked()), ConstantIntRange::untracked());
    }

    #[test]
    fn mixed_width_widening_compares_the_same_signed_interpretation() {
        let byte = ConstantIntRange::constant(Int::from(-1i8));
        let word = ConstantIntRange::constant(Int::from(255u16));
        assert_eq!(byte.widen(&word), word);
        assert_eq!(word.widen(&byte), word);

        let negative_word = ConstantIntRange::constant(Int::from(-1i16));
        let widened = byte.widen(&negative_word);
        assert_eq!(widened.width(), 16);
        assert_eq!((widened.umin(), widened.umax()), (255, 65535));
        assert_eq!((widened.smin(), widened.smax()), (-32768, 255));
    }

    #[test]
    fn mixed_width_joins_keep_every_supported_raw_representation() {
        for narrow_width in 1..=128 {
            let raw_max = unsigned_max(narrow_width);
            let narrow = ConstantIntRange::from_unsigned(narrow_width, raw_max, raw_max);
            assert_eq!(narrow.union(&narrow), narrow);
            for wide_width in narrow_width + 1..=128 {
                let zero = ConstantIntRange::from_unsigned(wide_width, 0, 0);
                let joined = narrow.union(&zero);
                assert_eq!(joined, ConstantIntRange::from_unsigned(wide_width, 0, raw_max));
            }
        }
    }

    #[test]
    fn small_mixed_width_joins_obey_lattice_laws_and_contain_inputs() {
        fn contains(range: &ConstantIntRange, raw: u128) -> bool {
            if range.is_untracked() {
                return true;
            }
            if raw > unsigned_max(range.width) {
                return false;
            }
            let signed = if raw < 1 << (range.width - 1) {
                raw as i128
            } else {
                raw as i128 - (1 << range.width)
            };
            range.umin <= raw && raw <= range.umax && range.smin <= signed && signed <= range.smax
        }

        // Enumerate every nonempty pair of signed/unsigned intervals for widths one and two,
        // including redundant bounds like those produced by widening. Add all three-bit
        // constructor ranges so associativity also crosses three interpretation widths.
        let mut ranges = vec![ConstantIntRange::untracked()];
        for width in 1..=2 {
            for umin in 0..=unsigned_max(width) {
                for umax in umin..=unsigned_max(width) {
                    for smin in signed_min(width)..=signed_max(width) {
                        for smax in smin..=signed_max(width) {
                            let range = ConstantIntRange {
                                width,
                                umin,
                                umax,
                                smin,
                                smax,
                            };
                            if (0..=unsigned_max(width)).any(|raw| contains(&range, raw)) {
                                ranges.push(range);
                            }
                        }
                    }
                }
            }
        }
        for min in 0..8 {
            for max in min..8 {
                ranges.push(ConstantIntRange::from_unsigned(3, min, max));
                ranges.push(ConstantIntRange::from_signed(3, min as i128 - 4, max as i128 - 4));
            }
        }

        for lhs in &ranges {
            assert_eq!(lhs.union(lhs), *lhs);
            for rhs in &ranges {
                let joined = lhs.union(rhs);
                assert_eq!(joined, rhs.union(lhs));
                for raw in 0..8 {
                    if contains(lhs, raw) || contains(rhs, raw) {
                        assert!(contains(&joined, raw), "{lhs} union {rhs} excludes {raw}");
                    }
                }
                // Widening may independently expand signed and unsigned intervals. Its result
                // must still be a valid join operand, even when the bounds are not reduced.
                let widened = lhs.widen(rhs);
                assert_eq!(widened.union(&widened), widened);
                for third in &ranges {
                    assert_eq!(lhs.union(rhs).union(third), lhs.union(&rhs.union(third)));
                    assert_eq!(widened.union(third), third.union(&widened));
                    assert_eq!(widened.union(lhs).union(third), widened.union(&lhs.union(third)));
                }
            }
        }
    }

    /// A range of another width says nothing about the value being checked: the full range of
    /// `i8` refined by the full range of `i32` is the full range of `i32`.
    #[test]
    fn a_full_width_range_says_nothing_about_a_wider_checked_range() {
        let context = Context::default();
        let full = IntegerValueRange::max_for_type(context.i8());
        let checked = ConstantIntRange::max_for_type(context.i32());
        assert_eq!(full.refined_by(&checked).range(), Some(&checked));
    }

    #[test]
    fn regression_type_range_contains_all_constants_of_its_width() {
        let context = Context::default();
        let top = ConstantIntRange::max_for_type(context.i64());
        let constant = ConstantIntRange::constant(Int::from(u64::MAX - 1));
        assert_eq!(top.union(&constant), top);
    }

    #[test]
    fn wrapping_add_uses_each_interpretation_independently() {
        let unsigned = ConstantIntRange::constant(Int::from(0x7fff_ffffu32));
        let one = ConstantIntRange::constant(Int::from(1u32));
        assert_eq!(
            unsigned.add_wrapping(&one),
            Some(ConstantIntRange::constant(Int::from(0x8000_0000u32)))
        );
        let signed = ConstantIntRange::constant(Int::from(-1i32));
        assert_eq!(signed.add_wrapping(&one), Some(ConstantIntRange::constant(Int::from(0i32))));
        let straddling = ConstantIntRange::from_unsigned(32, 0, 0x8000_0000);
        assert_eq!(
            straddling.add_wrapping(&one),
            Some(ConstantIntRange::from_unsigned(32, 1, 0x8000_0001))
        );
    }

    #[test]
    fn checked_add_uses_the_selected_interpretation() {
        let max = ConstantIntRange::constant(Int::from(0x7fff_ffffu32));
        let one = ConstantIntRange::constant(Int::from(1u32));
        assert!(max.add_unsigned_without_overflow(&one).is_some());
        assert!(max.add_signed_without_overflow(&one).is_none());
        let minus_one = ConstantIntRange::constant(Int::from(-1i32));
        assert!(minus_one.add_unsigned_without_overflow(&one).is_none());
        assert_eq!(
            minus_one.add_signed_without_overflow(&one),
            Some(ConstantIntRange::constant(Int::from(0i32)))
        );
    }

    #[test]
    fn intersection_respects_top_empty_and_incompatible_ranges() {
        let range = ConstantIntRange::from_unsigned(8, 0, 5);
        assert_eq!(range.intersection(&ConstantIntRange::untracked()), Some(range));
        assert_eq!(ConstantIntRange::untracked().intersection(&range), Some(range));
        assert_eq!(range.intersection(&ConstantIntRange::from_unsigned(8, 6, 10)), None);
        let context = Context::default();
        assert_eq!(range.intersection(&ConstantIntRange::max_for_type(context.i16())), None);
        let lhs = ConstantIntRange::from_unsigned(8, 0, 50)
            .union(&ConstantIntRange::from_unsigned(8, 128, 200));
        let rhs = ConstantIntRange::from_unsigned(8, 60, 127)
            .union(&ConstantIntRange::from_unsigned(8, 206, 255));
        assert_eq!(lhs.intersection(&rhs), None);
    }

    /// Refinement waits for its input, keeps the bounds of an input of the checked width, and
    /// gives the checked range for an input of another width, however precise.
    #[test]
    fn refinement_waits_for_input_and_preserves_bounds_of_its_width() {
        let context = Context::default();
        let range = ConstantIntRange::max_for_type(context.i32());
        assert!(IntegerValueRange::uninitialized().refined_by(&range).is_uninitialized());
        let precise = IntegerValueRange::new(ConstantIntRange::from_unsigned(32, 0, 10));
        assert_eq!(
            precise.refined_by(&range).range(),
            Some(&ConstantIntRange::from_unsigned(32, 0, 10))
        );
        let wider = IntegerValueRange::new(ConstantIntRange::from_unsigned(64, 0, 10));
        assert_eq!(wider.refined_by(&range).range(), Some(&range));
        let disjoint = IntegerValueRange::constant(Int::from(u64::MAX));
        assert_eq!(
            disjoint.refined_by(&range),
            IntegerValueRange::new(ConstantIntRange::max_for_type(context.i32()))
        );
    }

    /// An input of the checked width keeps its bounds; an input of another width, narrower or
    /// wider, gives the checked range.
    #[test]
    fn explicit_refinement_keeps_the_input_bounds_only_at_the_checked_width() {
        let checked = ConstantIntRange::from_unsigned(32, 0, 2_147_483_647);
        for (input, expected) in [
            (ConstantIntRange::from_unsigned(8, 0, 3), checked),
            (
                ConstantIntRange::from_unsigned(32, 0, 3),
                ConstantIntRange::from_unsigned(32, 0, 3),
            ),
            (ConstantIntRange::from_unsigned(64, 0, 3), checked),
        ] {
            assert_eq!(
                IntegerValueRange::new(input).refined_by(&checked),
                IntegerValueRange::new(expected)
            );
        }
        assert!(IntegerValueRange::uninitialized().refined_by(&checked).is_uninitialized());
    }

    #[test]
    fn refinement_preserves_declared_signed_bounds() {
        // The declared raw interval spans all eight-bit values, but its signed bounds exclude
        // 6..=250. Refinement must not erase that part of the runtime check: an input of the
        // checked width is narrowed by both, and an input of another width gives the check.
        let checked = ConstantIntRange::from_signed(8, -5, 5);
        for (min, max, expected) in [
            (0, 200, ConstantIntRange::from_unsigned(8, 0, 5)),
            (128, 255, ConstantIntRange::from_signed(8, -5, -1)),
        ] {
            let input = IntegerValueRange::new(ConstantIntRange::from_unsigned(8, min, max));
            assert_eq!(input.refined_by(&checked), IntegerValueRange::new(expected));
            let wider = IntegerValueRange::new(ConstantIntRange::from_unsigned(64, min, max));
            assert_eq!(wider.refined_by(&checked), IntegerValueRange::new(checked));
        }
    }

    #[test]
    fn a_range_is_within_another_when_each_bound_is_inside() {
        let outer = ConstantIntRange::from_unsigned(8, 10, 20);
        // equal
        assert!(outer.is_within(&outer));
        // inside
        assert!(ConstantIntRange::from_unsigned(8, 11, 19).is_within(&outer));
        assert!(ConstantIntRange::constant(Int::from(10u8)).is_within(&outer));
        assert!(ConstantIntRange::constant(Int::from(20u8)).is_within(&outer));
        // straddling the lower end, the upper end, and both
        assert!(!ConstantIntRange::from_unsigned(8, 9, 15).is_within(&outer));
        assert!(!ConstantIntRange::from_unsigned(8, 15, 21).is_within(&outer));
        assert!(!ConstantIntRange::from_unsigned(8, 0, 255).is_within(&outer));
        // the containing range is not within the one it contains
        assert!(!outer.is_within(&ConstantIntRange::from_unsigned(8, 11, 19)));
    }

    #[test]
    fn a_range_is_within_another_only_at_the_same_width() {
        let narrow = ConstantIntRange::from_unsigned(8, 0, 5);
        let wide = ConstantIntRange::from_unsigned(16, 0, 100);
        assert!(!narrow.is_within(&wide));
        assert!(!ConstantIntRange::from_unsigned(16, 0, 5).is_within(&narrow));
        // A 64-bit range whose values fit 32 bits is not within the 32-bit type's range.
        let context = Context::default();
        let fits = ConstantIntRange::constant(Int::from(1u64 << 31));
        assert!(!fits.is_within(&ConstantIntRange::max_for_type(context.i32())));
        assert!(fits.is_within(&ConstantIntRange::max_for_type(context.i64())));
    }

    #[test]
    fn an_untracked_range_is_never_within_and_contains_nothing() {
        let untracked = ConstantIntRange::untracked();
        let range = ConstantIntRange::from_unsigned(8, 0, 5);
        assert!(!untracked.is_within(&untracked));
        assert!(!untracked.is_within(&range));
        assert!(!range.is_within(&untracked));
    }

    #[test]
    fn a_range_is_within_another_only_if_its_signed_bounds_are_inside() {
        // Unsigned 0..=255 at width 8 with signed bounds -5..=5: the values 0..=5 and 251..=255.
        let signed = ConstantIntRange::from_signed(8, -5, 5);
        assert!(ConstantIntRange::from_unsigned(8, 0, 5).is_within(&signed));
        assert!(ConstantIntRange::from_signed(8, -5, -1).is_within(&signed));
        assert!(!ConstantIntRange::from_unsigned(8, 0, 6).is_within(&signed));
        assert!(!ConstantIntRange::max_for_type(Context::default().i8()).is_within(&signed));
    }

    #[test]
    fn small_intersections_match_concrete_bit_patterns() {
        fn contains(range: &ConstantIntRange, value: u128) -> bool {
            let signed = unsigned_to_signed(4, value);
            range.umin <= value
                && value <= range.umax
                && range.smin <= signed
                && signed <= range.smax
        }
        for umin in 0..16 {
            for umax in umin..16 {
                let unsigned = ConstantIntRange::from_unsigned(4, umin, umax);
                for smin in -8..8 {
                    for smax in smin..8 {
                        let signed = ConstantIntRange::from_signed(4, smin, smax);
                        let intersection = unsigned.intersection(&signed);
                        for value in 0..16 {
                            assert_eq!(
                                intersection.as_ref().is_some_and(|range| contains(range, value)),
                                contains(&unsigned, value) && contains(&signed, value),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn small_wrapping_ranges_contain_every_concrete_result() {
        for lmin in 0..16 {
            for lmax in lmin..16 {
                let lhs = ConstantIntRange::from_unsigned(4, lmin, lmax);
                for rmin in 0..16 {
                    for rmax in rmin..16 {
                        let rhs = ConstantIntRange::from_unsigned(4, rmin, rmax);
                        if let Some(result) = lhs.add_wrapping(&rhs) {
                            for lhs in lmin..=lmax {
                                for rhs in rmin..=rmax {
                                    let value = (lhs + rhs) & 15;
                                    let signed = unsigned_to_signed(4, value);
                                    assert!(result.umin <= value && value <= result.umax);
                                    assert!(result.smin <= signed && signed <= result.smax);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// A constant of a fixed-width type has the exact range of its value; an `isize` constant is
    /// untracked, because its value on a 32-bit target is its low 32 bits.
    #[test]
    fn the_range_of_a_constant_knows_its_type() {
        let context = Context::default();

        let i32 = ConstantIntRange::for_constant(Int::from(-5i32), context.i32());
        assert_eq!(i32, ConstantIntRange::constant(Int::from(-5i32)));
        assert_eq!((i32.width(), i32.smin(), i32.smax()), (32, -5, -5));

        let i1 = ConstantIntRange::for_constant(Int::from(true), context.i1());
        assert_eq!((i1.width(), i1.umin(), i1.umax()), (1, 1, 1));

        let i128 = ConstantIntRange::for_constant(Int::from(i128::MIN), context.i128());
        assert_eq!((i128.width(), i128.smin(), i128.smax()), (128, i128::MIN, i128::MIN));

        let isize = ConstantIntRange::for_constant(Int::from(1u64 << 32), context.isize());
        assert!(isize.is_untracked());
        assert_eq!(isize, ConstantIntRange::max_for_type(context.isize()));

        let exact = IntegerValueRange::for_constant(Int::from(7u32), context.i32());
        assert_eq!(exact, IntegerValueRange::constant(Int::from(7u32)));
        let isize = IntegerValueRange::for_constant(Int::from(1u64 << 32), context.isize());
        assert!(isize.range().unwrap().is_untracked());
        assert_eq!(isize, IntegerValueRange::max_for_type(context.isize()));
    }

    /// In a debug build, a constant whose value is not as wide as its type is a bug.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "must be as wide as the type")]
    fn the_range_of_a_constant_of_the_wrong_width_panics_in_a_debug_build() {
        let context = Context::default();
        let _ = ConstantIntRange::for_constant(Int::from(5i64), context.i32());
    }
}
