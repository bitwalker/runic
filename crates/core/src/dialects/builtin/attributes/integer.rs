use crate::{
    AttrPrinter,
    attributes::{AttrParser, IntegerLikeAttr},
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    parse::ParserExt,
    print::AsmPrinter,
};

/// Defines the attribute `$name Attr` of the signless integer type of `$width` bits, held in the
/// signed Rust integer `$t` of that width. It reads the literals an integer constant of that type
/// does, and prints its value as a signed decimal.
macro_rules! define_integer_attr {
    ($name:ident, $t:ident, $width:literal) => {
        #[derive(DialectAttribute)]
        #[attribute(dialect = BuiltinDialect, remote = $t, implements(IntegerLikeAttr, AttrPrinter))]
        #[repr(transparent)]
        pub struct $name;

        paste::paste! {
            impl IntegerLikeAttr for [<$name Attr>] {
                #[inline]
                fn as_int(&self) -> crate::Int {
                    crate::Int::from(self.value)
                }
            }

            impl AttrPrinter for [<$name Attr>] {
                fn print(&self, printer: &mut AsmPrinter<'_>) {
                    printer.print_int(self.as_int());
                }
            }

            impl AttrParser for [<$name Attr>] {
                fn parse(
                    parser: &mut dyn crate::parse::Parser<'_>,
                ) -> crate::parse::ParseResult<crate::AttributeRef> {
                    let (span, literal) = parser.parse_integer_literal()?.into_parts();
                    let ty = parser.context().[<i $width>]();
                    let value = literal.to_int(ty, span)?.signed() as $t;
                    Ok(parser
                        .context_rc()
                        .create_attribute::<[<$name Attr>], _>(value)
                        .as_attribute_ref())
                }
            }
        }
    };
}

define_integer_attr!(I8, i8, 8);
define_integer_attr!(I16, i16, 16);
define_integer_attr!(I32, i32, 32);
define_integer_attr!(I64, i64, 64);
define_integer_attr!(I128, i128, 128);

/// An integer constant of any width up to 128 bits: `#builtin.int<-1 : i7>`.
///
/// Its type is the integer type of its value's width, or `isize` for a 64-bit value. Every
/// integer constant is an `IntAttr`; [crate::Context::int_attr] creates one of a given type.
///
/// An `IntAttr` whose type is not `iN` of its value's width (or `isize` with 64 bits) is invalid;
/// [crate::Int::width_for_type] states the rule. It is checked in two places: by `int_attr`, which
/// panics rather than create an invalid constant, and by the verifier
/// ([crate::Operation::verify_int_attributes]), which rejects an operation holding one among its
/// properties and attributes. `create_attribute_with_type::<IntAttr, _>`, the setters generated for
/// an operation's `IntAttr` field, and mutation in place do not check it.
#[derive(DialectAttribute)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "crate::Int",
    default = "default_int",
    implements(IntegerLikeAttr, AttrPrinter),
)]
#[allow(unused)]
struct Int;

/// The value of a body-less `#builtin.int`: a 32-bit zero.
fn default_int(_context: &crate::Context) -> crate::Int {
    crate::Int::zero(32)
}

impl IntegerLikeAttr for IntAttr {
    #[inline]
    fn as_int(&self) -> crate::Int {
        self.value
    }
}

impl AttrPrinter for IntAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::const_text;

        printer.print_int(self.value);
        *printer += const_text(" : ");
        printer.print_type(crate::Attribute::ty(self));
    }
}

impl AttrParser for IntAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        let (span, literal) = parser.parse_integer_literal()?.into_parts();
        let ty = parser.parse_colon_type()?.into_inner();
        let value = literal.to_int(ty, span)?;
        Ok(parser.context_rc().int_attr(value, ty).as_attribute_ref())
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::{Attribute, Context, Int};

    #[test]
    #[should_panic(expected = "an integer constant of type i64 cannot hold the 32-bit value 1")]
    fn int_attr_of_a_mismatched_width_panics() {
        let context = Rc::new(Context::default());
        context.int_attr(Int::from(1i32), context.i64());
    }

    #[test]
    fn int_attr_of_isize_holds_a_64_bit_value() {
        let context = Rc::new(Context::default());
        let attr = context.int_attr(Int::from(1i64), context.isize());
        assert_eq!(attr.borrow().ty(), context.isize());
        assert_eq!(attr.borrow().as_int(), Int::from(1i64));
    }

    #[test]
    fn int_attr_infers_its_type_from_the_width() {
        let context = Rc::new(Context::default());
        let attr = context.create_attribute::<IntAttr, _>(Int::from_bits(7, 1));
        assert_eq!(attr.borrow().ty(), context.int_type(7));
    }
}
