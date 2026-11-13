use alloc::boxed::Box;
use core::fmt::{self, Write};

use super::print_name;
use crate::{
    FxHashMap, FxHashSet, Int,
    derive::{DialectType, TypeComponent},
    dialects::builtin::BuiltinDialect,
    interner::Symbol,
    layout::{AggregateType, TypeLayout},
    types::{Type, TypeAliasName, TypeFormatter, TypePrinter},
};

/// A variant of an [EnumType]: a name, a discriminant value and an optional payload type.
///
/// A variant is built without a value, or with `with_value`; the enum assigns the others when it
/// is constructed, and its variants then all hold their resolved values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeComponent)]
pub struct EnumVariant {
    name: Symbol,
    /// The raw bits of the value: explicit if set by the user, resolved once in an [EnumType].
    value: Option<u128>,
    payload: Option<Type>,
}

impl EnumVariant {
    /// A variant with no explicit value and no payload.
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            value: None,
            payload: None,
        }
    }

    /// Gives this variant the explicit discriminant value `bits`, as raw bits of the
    /// discriminant type.
    pub fn with_value(mut self, bits: u128) -> Self {
        self.value = Some(bits);
        self
    }

    /// Gives this variant a payload of type `ty`.
    pub fn with_payload(mut self, ty: Type) -> Self {
        self.payload = Some(ty);
        self
    }

    pub fn name(&self) -> Symbol {
        self.name
    }

    /// The raw bits of this variant's discriminant value.
    ///
    /// The value is resolved once the variant is part of an [EnumType]; only then is the result
    /// meaningful. Before that it is the explicit value, or 0 for a variant without one, whatever
    /// value the enum will give it.
    pub fn value(&self) -> u128 {
        self.value.unwrap_or(0)
    }

    pub fn payload(&self) -> Option<Type> {
        self.payload
    }
}

/// Why an [EnumType] cannot be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InvalidEnumType {
    #[error("the discriminant of an enum is an integer type of at most 128 bits, or `isize`")]
    InvalidDiscriminantType,
    #[error("the value of variant `{variant}` does not fit the discriminant type")]
    ValueOutOfRange { variant: Symbol },
    #[error("the implicit value of variant `{variant}` does not fit the discriminant type")]
    ImplicitValueOverflow { variant: Symbol },
    #[error("the enum has more than one variant named `{variant}`")]
    DuplicateVariant { variant: Symbol },
    #[error("variants `{other}` and `{variant}` have the same value")]
    DuplicateValue { variant: Symbol, other: Symbol },
}

/// A tagged union, `enum name<discriminant; variant, ...>`.
///
/// The discriminant is an `iN` of at most 128 bits, or `isize`, whose values are 64 bits wide.
/// An `isize` discriminant's values are checked against 64 bits here; laying the enum out checks
/// them again against the `isize` of the data layout.
/// A variant's value is the raw bits of the discriminant: it is the one given explicitly, or else
/// the previous variant's value plus one, or 0 for the first. Values are resolved when the enum
/// is constructed, so `enum<i8; a, b>` and `enum<i8; a = 0, b = 1>` are one type. The enum's name
/// is part of its identity.
///
/// `replace_subtypes` replaces the discriminant and payload types and keeps the variants as they
/// are, with their resolved values; it does not run the checks of [Self::new], so the caller must
/// replace the discriminant only with a valid one of the same width.
#[derive(Debug, PartialEq, Eq, Hash, DialectType)]
#[ty(dialect = BuiltinDialect, name = "enum", implements(TypeLayout, AggregateType))]
pub struct EnumType {
    name: Option<Symbol>,
    discriminant: Type,
    variants: Box<[EnumVariant]>,
}

impl EnumType {
    /// The width in bits of values of the discriminant type `ty`, if it is a valid one: `N` for
    /// `iN` with `N <= 128`, and 64 for `isize`.
    /// This is the width of a constant of that type, [Int::width_for_type](crate::Int::width_for_type).
    pub fn discriminant_width(ty: Type) -> Option<u32> {
        crate::Int::width_for_type(ty)
    }

    /// True if the raw bits `bits` of a variant's value fit a discriminant of `width` bits: no bit
    /// at or above `width` is set.
    pub fn value_fits(bits: u128, width: u32) -> bool {
        width >= u128::BITS || bits >> width == 0
    }

    /// Creates an enum, checking that `discriminant` is a valid discriminant type, that every
    /// variant's value fits it, and that names and values are distinct.
    pub fn new(
        name: Option<Symbol>,
        discriminant: Type,
        variants: impl IntoIterator<Item = EnumVariant>,
    ) -> Result<Self, InvalidEnumType> {
        let width = Self::discriminant_width(discriminant)
            .ok_or(InvalidEnumType::InvalidDiscriminantType)?;
        let fits = |bits: u128| Self::value_fits(bits, width);

        let mut variants: Box<[EnumVariant]> = variants.into_iter().collect();
        let mut previous = None::<u128>;
        for variant in variants.iter_mut() {
            let bits = match (variant.value, previous) {
                (Some(bits), _) if fits(bits) => bits,
                (Some(_), _) => {
                    return Err(InvalidEnumType::ValueOutOfRange {
                        variant: variant.name,
                    });
                }
                (None, None) => 0,
                (None, Some(previous)) => match previous.checked_add(1) {
                    Some(bits) if fits(bits) => bits,
                    _ => {
                        return Err(InvalidEnumType::ImplicitValueOverflow {
                            variant: variant.name,
                        });
                    }
                },
            };
            variant.value = Some(bits);
            previous = Some(bits);
        }

        let mut names = FxHashSet::default();
        for variant in variants.iter() {
            if !names.insert(variant.name) {
                return Err(InvalidEnumType::DuplicateVariant {
                    variant: variant.name,
                });
            }
        }
        let mut values = FxHashMap::default();
        for variant in variants.iter() {
            if let Some(other) = values.insert(variant.value(), variant.name) {
                return Err(InvalidEnumType::DuplicateValue {
                    variant: variant.name,
                    other,
                });
            }
        }

        Ok(Self {
            name,
            discriminant,
            variants,
        })
    }

    pub fn name(&self) -> Option<Symbol> {
        self.name
    }

    pub fn discriminant(&self) -> Type {
        self.discriminant
    }

    /// The variants in declaration order, each with its resolved value.
    pub fn variants(&self) -> &[EnumVariant] {
        &self.variants
    }
}

/// A recursive enum is referred to by its name.
impl TypeAliasName for EnumType {
    fn alias_name(&self) -> Option<Symbol> {
        self.name
    }
}

/// Writes the `width`-bit pattern `bits` as a signed decimal; an `i1` is written as `0` or `1`.
fn write_signed(printer: &mut TypeFormatter<'_, '_>, bits: u128, width: u32) -> fmt::Result {
    write!(printer, "{}", Int::from_bits(width, bits))
}

impl TypePrinter for EnumType {
    fn print(&self, printer: &mut TypeFormatter<'_, '_>) -> fmt::Result {
        let width = Self::discriminant_width(self.discriminant).unwrap_or(64);
        printer.write_str("enum")?;
        if let Some(name) = self.name {
            printer.write_str(" ")?;
            print_name(printer, name)?;
        }
        printer.write_str("<")?;
        printer.ty(self.discriminant)?;
        // A value is left out only while the variants count up from 0, so that
        // `enum<i8; a = 0, b = 1>` prints as `enum<i8; a, b>`. From the first variant that does
        // not, every value is written.
        let mut counting = true;
        for (i, variant) in self.variants.iter().enumerate() {
            printer.write_str(if i == 0 { "; " } else { ", " })?;
            print_name(printer, variant.name)?;
            counting &= variant.value() == i as u128;
            if !counting {
                printer.write_str(" = ")?;
                write_signed(printer, variant.value(), width)?;
            }
            if let Some(payload) = variant.payload {
                printer.write_str(": ")?;
                printer.ty(payload)?;
            }
        }
        printer.write_str(">")
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec::Vec};

    use super::*;
    use crate::{Context, TypeImpl};

    fn sym(name: &str) -> Symbol {
        Symbol::intern(name)
    }

    fn variants(names: &[&str]) -> Vec<EnumVariant> {
        names.iter().map(|name| EnumVariant::new(sym(name))).collect()
    }

    fn values(ty: Type) -> Vec<u128> {
        ty.as_enum().unwrap().variants().iter().map(EnumVariant::value).collect()
    }

    #[test]
    fn implicit_values_count_up_from_zero_or_the_previous_value() {
        let context = Context::default();
        let ty = context
            .enum_type(EnumType::new(None, context.i8(), variants(&["a", "b", "c"])).unwrap());
        assert_eq!(values(ty), [0, 1, 2]);
        let ty = context.enum_type(
            EnumType::new(
                None,
                context.i8(),
                [
                    EnumVariant::new(sym("a")).with_value(5),
                    EnumVariant::new(sym("b")),
                    EnumVariant::new(sym("c")).with_value(0),
                    EnumVariant::new(sym("d")),
                ],
            )
            .unwrap(),
        );
        assert_eq!(values(ty), [5, 6, 0, 1]);
        assert!(ty.is_enum() && !context.i8().is_enum());
        assert_eq!(EnumVariant::new(sym("x")).value(), 0);
        assert_eq!(EnumVariant::new(sym("x")).with_value(3).value(), 3);
    }

    #[test]
    fn implicit_and_equal_explicit_values_are_one_type() {
        let context = Context::default();
        let implicit = EnumType::new(None, context.i8(), variants(&["a", "b"])).unwrap();
        let explicit = EnumType::new(
            None,
            context.i8(),
            [
                EnumVariant::new(sym("a")).with_value(0),
                EnumVariant::new(sym("b")).with_value(1),
            ],
        )
        .unwrap();
        assert_eq!(context.enum_type(implicit), context.enum_type(explicit));
        let named = EnumType::new(Some(sym("e")), context.i8(), variants(&["a", "b"])).unwrap();
        assert_ne!(
            context.enum_type(named),
            context.enum_type(EnumType::new(None, context.i8(), variants(&["a", "b"])).unwrap())
        );
        let wider = EnumType::new(None, context.i16(), variants(&["a", "b"])).unwrap();
        assert_ne!(
            context.enum_type(wider),
            context.enum_type(EnumType::new(None, context.i8(), variants(&["a", "b"])).unwrap())
        );
    }

    #[test]
    fn invalid_enums_are_rejected_in_order() {
        let context = Context::default();
        let a = EnumVariant::new(sym("a"));
        let b = EnumVariant::new(sym("b"));
        for ty in [context.f32(), context.opaque_ptr_type(), context.int_type(129)] {
            assert_eq!(
                EnumType::new(None, ty, [a]).unwrap_err(),
                InvalidEnumType::InvalidDiscriminantType
            );
        }
        assert_eq!(
            EnumType::new(None, context.i8(), [a, b.with_value(256)]).unwrap_err(),
            InvalidEnumType::ValueOutOfRange { variant: sym("b") }
        );
        assert_eq!(
            EnumType::new(None, context.i1(), [a, b, EnumVariant::new(sym("c"))]).unwrap_err(),
            InvalidEnumType::ImplicitValueOverflow { variant: sym("c") }
        );
        assert_eq!(
            EnumType::new(None, context.i8(), [a, b, a.with_value(7)]).unwrap_err(),
            InvalidEnumType::DuplicateVariant { variant: sym("a") }
        );
        assert_eq!(
            EnumType::new(None, context.i8(), [a, b.with_value(0)]).unwrap_err(),
            InvalidEnumType::DuplicateValue {
                variant: sym("b"),
                other: sym("a")
            }
        );
        // A range error is found before a duplicate.
        assert_eq!(
            EnumType::new(None, context.i8(), [a, a, b.with_value(256)]).unwrap_err(),
            InvalidEnumType::ValueOutOfRange { variant: sym("b") }
        );
        assert_eq!(
            format!(
                "{}",
                InvalidEnumType::DuplicateValue {
                    variant: sym("b"),
                    other: sym("a")
                }
            ),
            "variants `a` and `b` have the same value"
        );
    }

    #[test]
    fn a_payload_may_be_open_but_a_discriminant_may_not() {
        let context = Context::default();
        let var = context.rec_var(0);
        for payload in [context.ptr_type(var), var] {
            let ty = EnumType::new(
                None,
                context.i8(),
                [EnumVariant::new(sym("a")).with_payload(payload)],
            )
            .unwrap();
            assert!(context.enum_type(ty).is_open(), "{payload}");
        }
        assert_eq!(
            EnumType::new(None, var, variants(&["a"])).unwrap_err(),
            InvalidEnumType::InvalidDiscriminantType
        );
    }

    #[test]
    fn the_implicit_value_range_is_the_width_of_the_discriminant() {
        let context = Context::default();
        let many = |n: usize| {
            (0..n)
                .map(|i| EnumVariant::new(Symbol::intern(format!("v{i}"))))
                .collect::<Vec<_>>()
        };
        assert!(EnumType::new(None, context.i8(), many(256)).is_ok());
        assert_eq!(
            EnumType::new(None, context.i8(), many(257)).unwrap_err(),
            InvalidEnumType::ImplicitValueOverflow {
                variant: sym("v256")
            }
        );
        let top = EnumVariant::new(sym("top")).with_value(u128::MAX);
        let ty = EnumType::new(None, context.int_type(128), [top]).unwrap();
        assert_eq!(ty.variants()[0].value(), u128::MAX);
        assert_eq!(
            EnumType::new(None, context.int_type(128), [top, EnumVariant::new(sym("next"))])
                .unwrap_err(),
            InvalidEnumType::ImplicitValueOverflow {
                variant: sym("next")
            }
        );
        assert_eq!(
            EnumType::new(None, context.int_type(127), [top]).unwrap_err(),
            InvalidEnumType::ValueOutOfRange {
                variant: sym("top")
            }
        );
    }

    #[test]
    fn isize_is_a_64_bit_discriminant_and_empty_enums_are_accepted() {
        let context = Context::default();
        let big = EnumVariant::new(sym("a")).with_value(u64::MAX as u128);
        assert!(EnumType::new(None, context.isize(), [big]).is_ok());
        assert_eq!(
            EnumType::new(None, context.isize(), [big.with_value(1 << 64)]).unwrap_err(),
            InvalidEnumType::ValueOutOfRange { variant: sym("a") }
        );
        let empty = context.enum_type(EnumType::new(None, context.i32(), []).unwrap());
        assert!(empty.as_enum().unwrap().variants().is_empty());
    }

    #[test]
    fn enums_print_in_the_documented_syntax() {
        let context = Context::default();
        let i8 = context.i8();
        let some = EnumVariant::new(sym("some")).with_payload(context.i32());
        let cases = [
            (EnumType::new(None, i8, []).unwrap(), "enum<i8>"),
            (EnumType::new(None, i8, variants(&["a", "b"])).unwrap(), "enum<i8; a, b>"),
            (
                EnumType::new(Some(sym("option")), i8, [EnumVariant::new(sym("none")), some])
                    .unwrap(),
                "enum option<i8; none, some: i32>",
            ),
            (
                EnumType::new(
                    Some(sym("color")),
                    i8,
                    [
                        EnumVariant::new(sym("red")).with_value(1),
                        EnumVariant::new(sym("green")).with_value(2),
                        EnumVariant::new(sym("blue")).with_value(4),
                    ],
                )
                .unwrap(),
                "enum color<i8; red = 1, green = 2, blue = 4>",
            ),
            (
                EnumType::new(None, i8, [EnumVariant::new(sym("a")).with_value(255)]).unwrap(),
                "enum<i8; a = -1>",
            ),
            (
                EnumType::new(
                    None,
                    i8,
                    [
                        EnumVariant::new(sym("a")).with_value(127),
                        EnumVariant::new(sym("b")),
                        EnumVariant::new(sym("c")),
                    ],
                )
                .unwrap(),
                "enum<i8; a = 127, b = -128, c = -127>",
            ),
            (
                EnumType::new(None, context.i1(), [EnumVariant::new(sym("a")).with_value(1)])
                    .unwrap(),
                "enum<i1; a = 1>",
            ),
            (
                EnumType::new(
                    None,
                    context.int_type(128),
                    [EnumVariant::new(sym("a")).with_value(u128::MAX)],
                )
                .unwrap(),
                "enum<i128; a = -1>",
            ),
            (
                EnumType::new(
                    None,
                    context.isize(),
                    [EnumVariant::new(sym("a")).with_value(1 << 63)],
                )
                .unwrap(),
                "enum<isize; a = -9223372036854775808>",
            ),
            (
                EnumType::new(
                    Some(sym("my enum")),
                    context.i32(),
                    [EnumVariant::new(sym("a b")), EnumVariant::new(sym("struct"))],
                )
                .unwrap(),
                "enum \"my enum\"<i32; \"a b\", struct>",
            ),
        ];
        for (ty, text) in cases {
            assert_eq!(format!("{}", context.enum_type(ty)), text);
        }
    }

    #[test]
    fn subtypes_are_the_discriminant_then_the_payloads_in_order() {
        let context = Context::default();
        let ty = context.enum_type(
            EnumType::new(
                None,
                context.i8(),
                [
                    EnumVariant::new(sym("a")).with_payload(context.i64()),
                    EnumVariant::new(sym("b")),
                    EnumVariant::new(sym("c")).with_payload(context.f32()),
                ],
            )
            .unwrap(),
        );
        let mut seen = Vec::new();
        ty.visit_subtypes(&mut |subtype| seen.push(subtype));
        assert_eq!(seen, [context.i8(), context.i64(), context.f32()]);
        let replaced =
            context.enum_type(ty.as_enum().unwrap().replace_subtypes(&mut |_| context.i16()));
        assert_eq!(format!("{replaced}"), "enum<i16; a: i16, b, c: i16>");
    }
}
