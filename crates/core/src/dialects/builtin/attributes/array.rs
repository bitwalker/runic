use core::fmt;

use smallvec::SmallVec;

use crate::{
    AttrPrinter,
    attributes::{AttrParser, InferAttributeValueType},
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    formatter::DisplayValues,
    parse::ParserExt,
    print::AsmPrinter,
};

/// An array of `i32` values, `#builtin.array_i32<-1, 2, 3>` (`#builtin.array_i32<>` when empty).
/// Each element is written as an `i32` constant is and printed as a signed decimal.
#[derive(DialectAttribute)]
#[attribute(
    name = "array_i32",
    dialect = BuiltinDialect,
    remote = "Array<i32>",
    implements(AttrPrinter),
)]
#[allow(unused)]
struct I32Array;

/// An array of types, `#builtin.type_array<i32, ptr<i8>>` (`#builtin.type_array<>` when empty).
#[derive(DialectAttribute)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "Array<crate::Type>",
    implements(AttrPrinter),
)]
#[allow(unused)]
struct TypeArray;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Array<T>(SmallVec<[T; 2]>);

impl<T> Array<T> {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub fn push(&mut self, value: T) {
        self.0.push(value);
    }

    pub fn remove(&mut self, index: usize) -> T {
        self.0.remove(index)
    }
}

default impl<I: IntoIterator> From<I> for Array<<I as IntoIterator>::Item> {
    #[inline(always)]
    fn from(iter: I) -> Self {
        Self::from_iter(iter)
    }
}

impl<T, const N: usize> From<[T; N]> for Array<T> {
    #[inline]
    fn from(value: [T; N]) -> Self {
        Self(SmallVec::from_iter(value))
    }
}

impl<T> From<SmallVec<[T; 2]>> for Array<T> {
    fn from(value: SmallVec<[T; 2]>) -> Self {
        Self(value)
    }
}

impl<T> From<alloc::vec::Vec<T>> for Array<T> {
    fn from(value: alloc::vec::Vec<T>) -> Self {
        Self(SmallVec::from_vec(value))
    }
}

impl<T> FromIterator<T> for Array<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(SmallVec::from_iter(iter))
    }
}

impl<T> Default for Array<T> {
    #[inline]
    fn default() -> Self {
        Self(SmallVec::new_const())
    }
}

impl<T> Array<T>
where
    T: Eq,
{
    pub fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }
}

impl<T> fmt::Debug for Array<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl<T> fmt::Display for Array<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", DisplayValues::new(self.0.iter()))
    }
}

impl<T> crate::formatter::PrettyPrint for Array<T>
where
    T: crate::formatter::PrettyPrint,
{
    fn render(&self) -> crate::formatter::Document {
        use crate::formatter::*;

        let entries = self.iter().fold(Document::Empty, |acc, v| match acc {
            Document::Empty => v.render(),
            _ => acc + const_text(", ") + v.render(),
        });
        if self.is_empty() {
            const_text("[]")
        } else {
            const_text("[") + entries + const_text("]")
        }
    }
}

impl<T> core::ops::Index<usize> for Array<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> core::ops::Index<core::ops::Range<usize>> for Array<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: core::ops::Range<usize>) -> &Self::Output {
        &self.0[range]
    }
}

impl<T> core::ops::Index<core::ops::RangeTo<usize>> for Array<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: core::ops::RangeTo<usize>) -> &Self::Output {
        &self.0[range]
    }
}

impl<T> core::ops::Index<core::ops::RangeToInclusive<usize>> for Array<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: core::ops::RangeToInclusive<usize>) -> &Self::Output {
        &self.0[range]
    }
}

impl<T> core::ops::Index<core::ops::RangeFrom<usize>> for Array<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: core::ops::RangeFrom<usize>) -> &Self::Output {
        &self.0[range]
    }
}

impl<T> core::ops::Index<core::ops::RangeFull> for Array<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _range: core::ops::RangeFull) -> &Self::Output {
        self.0.as_slice()
    }
}

impl<T> core::ops::IndexMut<usize> for Array<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl crate::attributes::DebugAssertIrValue for Array<crate::Type> {
    #[track_caller]
    fn debug_assert_ir_value(&self, context: &crate::Context) {
        for ty in self.iter() {
            context.debug_assert_ir_type(*ty, "an element of a type array attribute");
        }
    }
}

impl<T: Clone + InferAttributeValueType> InferAttributeValueType for Array<T> {
    fn infer_type(context: &crate::Context) -> crate::Type {
        <T as InferAttributeValueType>::infer_type(context)
    }

    fn infer_type_from_value(&self, context: &crate::Context) -> crate::Type {
        if self.0.is_empty() {
            Self::infer_type(context)
        } else {
            <T as InferAttributeValueType>::infer_type_from_value(&self.0[0], context)
        }
    }
}

impl AttrParser for I32ArrayAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        use crate::parse::Token;

        let ty = parser.context().i32();
        let mut values = Array::from_iter([]);
        let starts_element = |tok: Token<'_>| {
            matches!(tok, Token::Minus | Token::Int(_) | Token::Hex(_) | Token::Binary(_))
        };
        if parser.token_stream_mut().is_next(starts_element) {
            loop {
                let (span, literal) = parser.parse_integer_literal()?.into_parts();
                values.push(literal.to_int(ty, span)?.signed() as i32);
                if !parser.parse_optional_comma()? {
                    break;
                }
            }
        }
        Ok(parser.context_rc().create_attribute::<Self, _>(values).as_attribute_ref())
    }
}

impl crate::print::AttrPrinter for I32ArrayAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::*;

        for (i, value) in self.0.iter().enumerate() {
            if i > 0 {
                *printer += const_text(", ");
            }
            printer.print_int(crate::Int::from(*value));
        }
    }
}

impl crate::print::AttrPrinter for TypeArrayAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::*;

        for (i, value) in self.0.iter().enumerate() {
            if i > 0 {
                *printer += const_text(", ");
            }
            printer.print_type(*value);
        }
    }
}

impl AttrParser for TypeArrayAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        use crate::parse::Token;

        let mut types = Array::<crate::Type>::default();
        if !parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Rangle)) {
            loop {
                types.push(parser.parse_type()?.into_inner());
                if !parser.parse_optional_comma()? {
                    break;
                }
            }
        }
        Ok(parser.context_rc().create_attribute::<Self, _>(types).as_attribute_ref())
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::Context;

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "an element of a type array attribute is `ptr<rec.0>`, an open type")]
    fn an_open_type_in_a_type_array_attribute_panics() {
        let context = Rc::new(Context::default());
        let open = context.ptr_type(context.rec_var(0));
        let _ = context.create_attribute::<TypeArrayAttr, _>([context.i32(), open]);
    }

    #[test]
    fn closed_types_make_a_type_array_attribute() {
        let context = Rc::new(Context::default());
        let attr = context.create_attribute::<TypeArrayAttr, _>([context.i32(), context.i64()]);
        assert_eq!(attr.borrow().len(), 2);
    }
}
