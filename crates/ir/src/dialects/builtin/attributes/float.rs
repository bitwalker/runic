use crate::{
    AttrPrinter, attributes::AttrParser, derive::DialectAttribute,
    dialects::builtin::BuiltinDialect, parse::ParserExt, print::AsmPrinter,
};

/// A floating-point constant: `#builtin.float<1.5 : f64>`.
///
/// Its type is `f32` for a 32-bit value and `f64` for a 64-bit one; [crate::Context::float_attr]
/// creates one, and takes no type because the value's width gives it. The value is held as its
/// bits, so a NaN keeps its payload and `-0.0` is not `0.0`; two attributes of the same bits are
/// the same attribute in a context ([crate::Float] says how the values compare).
///
/// A `FloatAttr` whose type is not `f32` or `f64` of its value's width is invalid
/// ([crate::Float::width_for_type] states the rule). It is checked in two places: by
/// `float_attr`, which cannot create one, and by the kind's verifier
/// ([crate::AttributeRegistration::verify]), which [crate::Operation::verify] runs for every
/// constant an operation holds, at any depth. `create_attribute_with_type::<FloatAttr, _>` and
/// mutation in place do not check it.
///
/// The text between the angle brackets is a float literal (see [crate::Float] for how one is
/// written, and [crate::parse::ParserExt::parse_float_literal] for the ones it reads), a colon
/// and the type. A body-less `#builtin.float` is the 64-bit `0.0`.
#[derive(DialectAttribute)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "crate::Float",
    default = "default_float",
    verify = "verify_float",
    implements(AttrPrinter),
)]
#[allow(unused)]
struct Float;

/// A float constant is exactly as wide as a constant of its type
/// ([crate::Float::width_for_type]).
fn verify_float(
    value: &crate::Float,
    ty: crate::Type,
    _context: &crate::Context,
) -> Result<(), crate::Report> {
    let width = value.width();
    let problem = match crate::Float::width_for_type(ty) {
        Some(expected) if expected == width => return Ok(()),
        Some(expected) => alloc::format!("a constant of that type is {expected} bits wide"),
        None => "no float constant has that type".into(),
    };
    Err(crate::Report::msg(alloc::format!(
        "a float constant of type `{ty}` holding a {width}-bit value, but {problem}"
    )))
}

/// The value of a body-less `#builtin.float`: a 64-bit zero.
fn default_float(_context: &crate::Context) -> crate::Float {
    crate::Float::from_f64(0.0)
}

impl AttrPrinter for FloatAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::const_text;

        printer.print_float(self.value);
        *printer += const_text(" : ");
        printer.print_type(crate::Attribute::ty(self));
    }
}

impl AttrParser for FloatAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        // The literal's text is read before the type that says its width, so find the type first:
        // the literal is `<literal> : <type>`.
        let (span, literal) = parser.parse_float_literal_text()?.into_parts();
        let ty = parser.parse_colon_type()?.into_inner();
        let value = literal.to_float(ty, span)?;
        Ok(parser.context_rc().float_attr(value).as_attribute_ref())
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::{Attribute, Context, Float, testing::text::describe};

    #[test]
    fn float_attr_takes_its_type_from_the_width() {
        let context = Rc::new(Context::default());
        let single = context.float_attr(Float::from_f32(1.5));
        assert_eq!(single.borrow().ty(), context.f32());
        assert_eq!(*single.borrow().as_value(), Float::from_f32(1.5));
        let double = context.float_attr(Float::from_f64(1.5));
        assert_eq!(double.borrow().ty(), context.f64());
        assert!(single.borrow().verify(&context).is_ok());
        assert!(double.borrow().verify(&context).is_ok());
    }

    #[test]
    fn create_attribute_infers_the_type_from_the_value() {
        let context = Rc::new(Context::default());
        let attr = context.create_attribute::<FloatAttr, _>(Float::from_f32(2.0));
        assert_eq!(attr.borrow().ty(), context.f32());
    }

    #[test]
    fn the_verifier_rejects_a_value_of_the_other_width() {
        let context = Rc::new(Context::default());
        let attr =
            context.create_attribute_with_type::<FloatAttr, _>(Float::from_f32(1.0), context.f64());
        let err = attr.borrow().verify(&context).unwrap_err();
        assert!(
            describe(&err).contains(
                "a float constant of type `f64` holding a 32-bit value, but a constant of that \
                 type is 64 bits wide"
            ),
            "{}",
            describe(&err)
        );
    }

    #[test]
    fn the_verifier_rejects_a_value_typed_as_an_integer() {
        let context = Rc::new(Context::default());
        let attr =
            context.create_attribute_with_type::<FloatAttr, _>(Float::from_f32(1.0), context.i32());
        let err = attr.borrow().verify(&context).unwrap_err();
        assert!(
            describe(&err).contains(
                "a float constant of type `i32` holding a 32-bit value, but no float constant has \
                 that type"
            ),
            "{}",
            describe(&err)
        );
    }

    #[test]
    fn attributes_of_equal_bits_are_the_same_attribute() {
        let context = Rc::new(Context::default());
        let a = context.float_attr(Float::from_f64(0.1)).as_attribute_ref();
        let b = context.float_attr(Float::from_bits64(0.1f64.to_bits())).as_attribute_ref();
        assert!(a == b, "the same bits must give the same attribute");
        let zero = context.float_attr(Float::from_f64(0.0)).as_attribute_ref();
        let negative_zero = context.float_attr(Float::from_f64(-0.0)).as_attribute_ref();
        assert!(zero != negative_zero);
        let nan = context.float_attr(Float::from_f64(f64::NAN)).as_attribute_ref();
        let again = context.float_attr(Float::from_f64(f64::NAN)).as_attribute_ref();
        assert!(nan == again);
        let single = context.float_attr(Float::from_f32(0.0)).as_attribute_ref();
        assert!(single != zero);
    }

    #[test]
    fn the_default_is_a_64_bit_zero() {
        let context = Rc::new(Context::default());
        assert_eq!(default_float(&context), Float::from_bits64(0));
    }
}
