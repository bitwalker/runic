use crate::{
    AttrPrinter, Type, attributes::AttrParser, derive::DialectAttribute,
    dialects::builtin::BuiltinDialect, parse::ParserExt,
};

/// A local variable of a function: its index in the function's `locals` and its type.
///
/// Two locals of the same index and type are equal, however they were made: by
/// [Function::alloc_local](crate::dialects::builtin::Function::alloc_local), by [Self::new], or
/// parsed from text. The value names no function: the function an operation is in is the one
/// whose local it names, and that function's verifier checks the index and the type against its
/// `locals`.
///
/// As an attribute ([LocalVariableAttr]), the type is the value's only element: it is visited,
/// rewritten by a type map together with the function's `locals`, and verified like any type an
/// attribute holds. The attribute's own type is unknown. It has no default: the text always gives
/// the index and the type, `#builtin.local_variable<0, i32>`.
#[derive(DialectAttribute, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[attribute(
    dialect = BuiltinDialect,
    no_default,
    implements(AttrPrinter)
)]
pub struct LocalVariable {
    index: u32,
    ty: Type,
}

impl LocalVariable {
    /// The local of index `index` and type `ty`.
    pub fn new(index: u32, ty: Type) -> Self {
        Self { index, ty }
    }

    /// The local's index in its function's `locals`.
    #[inline(always)]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The local's type.
    #[inline(always)]
    pub fn ty(&self) -> Type {
        self.ty
    }
}

impl core::fmt::Display for LocalVariable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "local{}", self.index)
    }
}

/// `INDEX, TYPE`: the index in decimal, then the type.
impl AttrPrinter for LocalVariableAttr {
    fn print(&self, printer: &mut crate::print::AsmPrinter<'_>) {
        use crate::formatter::*;

        printer.print_decimal_integer(self.value.index);
        *printer += const_text(", ");
        printer.print_type(self.value.ty);
    }
}

/// `INDEX, TYPE`: a decimal index that fits a `u32`, then the local's type.
impl AttrParser for LocalVariableAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        let index = parser.parse_decimal_integer::<u32>()?.into_inner();
        parser.parse_comma()?;
        let ty = parser.parse_type()?.into_inner();

        Ok(parser
            .context_rc()
            .create_attribute::<LocalVariableAttr, _>(LocalVariable::new(index, ty))
            .as_attribute_ref())
    }
}
