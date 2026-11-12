use crate::{
    AttrPrinter, Immediate,
    attributes::{AttrParser, IntegerLikeAttr},
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    parse::ParserExt,
    print::AsmPrinter,
};

macro_rules! define_integer_attr {
    ($name:ident, $t:ident) => {
        __define_integer_attr!($name, $t, $t, |parser| {
            Ok(parser.parse_decimal_integer::<$t>()?.into_inner())
        });
    };
}

/// Defines an integer attribute of the signless integer type of `$width` bits, held in the signed
/// storage `$t`. It reads the literals an integer constant of that type does.
macro_rules! define_signless_integer_attr {
    ($name:ident, $t:ident, $width:literal) => {
        paste::paste! {
            __define_integer_attr!($name, $t, $t, |parser| {
                let (span, literal) = parser.parse_integer_literal()?.into_parts();
                let ty = parser.context().[<i $width>]();
                Ok(literal.to_immediate(ty, span)?.[<as_ $t>]().expect("held in signed storage"))
            });
        }
    };
}

macro_rules! __define_integer_attr {
    ($name:ident, $t:ty, $t_id:ident, |$parser:ident| $parse:expr) => {
        #[derive(DialectAttribute)]
        #[attribute(dialect = BuiltinDialect, remote = $t_id, implements(IntegerLikeAttr, AttrPrinter))]
        #[repr(transparent)]
        pub struct $name;

        paste::paste! {
            impl IntegerLikeAttr for [<$name Attr>] {
                #[inline]
                fn as_immediate(&self) -> Immediate {
                    Immediate::from(self.value)
                }

                fn set_from_immediate_lossy(&mut self, value: Immediate) {
                    if let Some(value) = value.[<as_ $t_id>]() {
                        self.value = value;
                    } else {
                        self.value = value.as_u128().unwrap() as $t;
                    }
                }
            }

            impl AttrPrinter for [<$name Attr>] {
                fn print(&self, printer: &mut AsmPrinter<'_>) {
                    printer.print_decimal_integer(self.as_immediate());
                }
            }

            impl AttrParser for [<$name Attr>] {
                fn parse(
                    $parser: &mut dyn crate::parse::Parser<'_>,
                ) -> crate::parse::ParseResult<crate::AttributeRef> {
                    let value = {
                        let parsed: crate::parse::ParseResult<$t> = $parse;
                        parsed?
                    };
                    Ok($parser
                        .context_rc()
                        .create_attribute::<[<$name Attr>], _>(value)
                        .as_attribute_ref())
                }
            }
        }
    };
}

define_signless_integer_attr!(I8, i8, 8);
define_integer_attr!(U8, u8);
define_signless_integer_attr!(I16, i16, 16);
define_integer_attr!(U16, u16);
define_signless_integer_attr!(I32, i32, 32);
define_integer_attr!(U32, u32);
define_signless_integer_attr!(I64, i64, 64);
define_integer_attr!(U64, u64);
define_signless_integer_attr!(I128, i128, 128);
define_integer_attr!(U128, u128);
