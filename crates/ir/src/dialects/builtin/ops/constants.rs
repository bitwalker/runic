use alloc::{format, string::ToString, sync::Arc};

use crate::{
    AggregateType, AsValueRange, AttributeRef, Context, Op, OpParser, OpPrinter, OperationState,
    Report, SmallVec, Spanned, UnsafeIntrusiveEntityRef, Value,
    constants::ConstantData,
    derive::{EffectOpInterface, operation},
    diagnostics::report,
    dialects::builtin::{BuiltinDialect, attributes::BytesAttr},
    effects::{
        AlwaysSpeculatable, ConditionallySpeculatable, MemoryEffectOpInterface, Pure,
        Speculatability,
    },
    formatter::const_text,
    parse::{ParseResult, ParserError},
    print::AsmPrinter,
    traits::{
        AnyArrayOf, AnyType, ConstantLike, FoldResult, Foldable, InferTypeOpInterface, Int8,
        OpFoldResult,
    },
};

pub type BytesRef = UnsafeIntrusiveEntityRef<Bytes>;

/// A constant array of bytes: `%b = builtin.bytes "0a0b0c0d" : array<i8; 4>;`.
///
/// The text is the data read as one little-endian number, written as a hexadecimal number is
/// written, most significant byte first (as [BytesAttr] writes it): the first byte of the data, the
/// one at the lowest address, is the last pair of digits. The example holds the bytes `0d 0c 0b 0a`
/// in memory order. The result is an `array<i8; N>`, where `N` is the number of bytes. It is the
/// data of the initializer region of a [GlobalVariable](super::GlobalVariable), which yields it
/// with a [Yield](super::Yield): the contents are placed in memory at the address of the global.
///
/// The operation is a constant: it has no effects and folds to its value.
#[derive(EffectOpInterface)]
#[operation(
    dialect = BuiltinDialect,
    traits(ConstantLike),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Bytes {
    /// The bytes
    #[attr(hidden)]
    value: BytesAttr,
    /// The array of the bytes
    #[result]
    result: AnyArrayOf<Int8>,
}

impl Bytes {
    /// The number of bytes.
    pub fn len(&self) -> usize {
        self.get_value().len()
    }

    /// Whether there are no bytes.
    pub fn is_empty(&self) -> bool {
        self.get_value().is_empty()
    }

    /// The bytes, first byte first.
    pub fn data(&self) -> Arc<ConstantData> {
        self.get_value().clone()
    }
}

/// A result whose type is not yet known gets the type `array<i8; N>`, `N` the number of bytes. A
/// result that already has a type (the text writes one) keeps it, and the verifier checks it.
impl InferTypeOpInterface for Bytes {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        if !self.result().ty().is_unknown() {
            return Ok(());
        }
        let ty = context.array_type(context.i8(), self.len() as u64);
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// The result is an `array<i8; N>`, `N` the number of bytes.
impl crate::Verify<dyn InferTypeOpInterface> for Bytes {
    fn verify(&self, context: &Context) -> Result<(), Report> {
        let expected = context.array_type(context.i8(), self.len() as u64);
        let ty = self.result().ty();
        if ty == expected {
            return Ok(());
        }
        Err(report!(
            message: "invalid operation builtin.bytes",
            labels: [primary(
                self.as_operation().span(),
                format!("the result has type `{ty}`, but {} byte(s) make a `{expected}`", self.len())
            )],
        ))
    }
}

impl Foldable for Bytes {
    #[inline]
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        results.push(OpFoldResult::Attribute(self.value));
        FoldResult::Ok(())
    }

    #[inline(always)]
    fn fold_with(
        &self,
        _operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        self.fold(results)
    }
}

impl OpPrinter for Bytes {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_space();
        printer.print_string(self.get_value().to_string());
        printer.print_space();
        printer.print_colon_type(self.result().ty());
        let op = self.as_operation();
        if op.has_attributes() {
            printer.print_space();
            printer.print_keyword("attributes");
            printer.print_space();
            printer.print_attribute_dictionary(
                op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }
    }
}

/// Reads `"digits" : type`, then an optional `attributes {..}`; the digits are the data written
/// most significant byte first, as [BytesAttr] reads them.
impl OpParser for Bytes {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let digits = parser.parse_string()?;
        let data = ConstantData::from_str_le(digits.as_str()).map_err(|reason| {
            ParserError::InvalidAttributeValue {
                span: digits.span(),
                reason: reason.to_string(),
            }
        })?;
        let ty = parser.parse_colon_type()?.into_inner();
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let id = parser.context().create_constant(data);
        let data = parser.context().get_constant(id);
        state.add_attribute("value", parser.context_rc().create_attribute::<BytesAttr, _>(data));
        state.results.push(ty);
        Ok(())
    }
}

pub type AggregateRef = UnsafeIntrusiveEntityRef<Aggregate>;

/// A struct or an array value built from its elements, one operand per element, in order:
///
/// ```text,ignore
/// %e = builtin.aggregate %name, %arity, %f : struct<ptr<i8>, i32, i64>;
/// %a = builtin.aggregate %e0, %e1 : array<i32; 2>;
/// %u = builtin.aggregate : struct<>;
/// ```
///
/// The text is the operands, a colon and the result type, then an optional `attributes {..}`. The
/// result type is always written: a struct and an array of the same elements differ. It is any
/// type that implements [AggregateType](crate::AggregateType), a dialect's own aggregate type too,
/// except an enum. Operand `i` has the type of element `i` of the result type, and there is one
/// operand per element. A nested aggregate is a `builtin.aggregate` that is an operand of another.
/// The elements of the type do not overlap: under a data layout each takes bytes of its own (no
/// builtin type's elements overlap), and a value whose type's elements overlap cannot be placed in
/// a global's data.
///
/// The operation is pure: it has no effects, is removed when unused, and two with the same
/// operands and type are merged. It does not fold: there is no attribute of an aggregate value.
#[derive(EffectOpInterface)]
#[operation(
    dialect = BuiltinDialect,
    traits(Pure, AlwaysSpeculatable),
    implements(ConditionallySpeculatable, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Aggregate {
    /// The elements, in order
    #[operands]
    elements: AnyType,
    /// The value built from them
    #[result]
    result: AnyType,
}

impl ConditionallySpeculatable for Aggregate {
    fn speculatability(&self) -> Speculatability {
        Speculatability::Speculatable
    }
}

/// The result type is an aggregate type that is not an enum, with one element per operand, each of
/// the type of its operand.
// An op's checks run only as the verifier of a listed trait or interface.
impl crate::Verify<dyn MemoryEffectOpInterface> for Aggregate {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let span = op.span();
        let invalid = |label: alloc::string::String| {
            report!(
                message: "invalid operation builtin.aggregate",
                labels: [primary(span, label)],
            )
        };
        let ty = self.result().ty();
        if ty.is_enum() {
            return Err(invalid(format!(
                "the result has type `{ty}`, but an enum value cannot be built with \
                 `builtin.aggregate` yet"
            )));
        }
        let Some(aggregate) = ty.as_trait::<dyn AggregateType>() else {
            return Err(invalid(format!(
                "the result has type `{ty}`, which is not an aggregate type: `builtin.aggregate` \
                 builds a struct or an array"
            )));
        };
        let given = op.num_operands();
        let count = aggregate.element_count();
        if u64::try_from(given).ok() != Some(count) {
            return Err(invalid(format!(
                "`{ty}` has {count} element(s), but {given} operand(s) are given"
            )));
        }
        for (index, operand) in op.operands().iter().enumerate() {
            let operand_ty = operand.borrow().ty();
            let element = u32::try_from(index).ok().and_then(|i| aggregate.element_type(i));
            match element {
                Some(element) if element == operand_ty => (),
                Some(element) => {
                    return Err(invalid(format!(
                        "operand {index} has type `{operand_ty}`, but element {index} of `{ty}` \
                         has type `{element}`"
                    )));
                }
                None => {
                    return Err(invalid(format!(
                        "operand {index} has type `{operand_ty}`, but element {index} of `{ty}` \
                         has no type"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Prints an optional `attributes {..}` after the rest of the operation.
fn print_attributes(printer: &mut AsmPrinter<'_>, op: &crate::Operation) {
    if op.has_attributes() {
        printer.print_space();
        printer.print_keyword("attributes");
        printer.print_space();
        printer.print_attribute_dictionary(
            op.attributes().iter().map(|attr| *attr.as_named_attribute()),
        );
    }
}

impl OpPrinter for Aggregate {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        let elements = op.operands().group(0);
        if !elements.is_empty() {
            printer.print_space();
            printer.print_value_uses(elements.as_value_range());
        }
        *printer += const_text(" : ");
        printer.print_type(self.result().ty());
        print_attributes(printer, op);
    }
}

/// Reads `%a, .. : type`, then an optional `attributes {..}`. The operands are resolved with the
/// types they were defined with; the verifier checks them against the elements of the type.
impl OpParser for Aggregate {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let mut elements = SmallVec::<[crate::parse::UnresolvedOperand; 2]>::default();
        parser.parse_optional_operand_list(&mut elements)?;
        let ty = parser.parse_colon_type()?.into_inner();
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let unknown = parser.context().unknown_type();
        let mut values = SmallVec::default();
        parser.resolve_operands_of_uniform_type(&elements, unknown, &mut values)?;
        state.operands.push(values);
        state.results.push(ty);
        Ok(())
    }
}

pub type ZeroRef = UnsafeIntrusiveEntityRef<Zero>;

/// The zero of a type: `%z = builtin.zero : array<i64; 4096>;`.
///
/// The value is the one whose every byte is zero: the integer and the float zero, the null pointer,
/// an aggregate of zeros. The operation takes any type, written after the colon; whether the type
/// has a layout is asked where the value is used, since the operation has no data layout to ask.
/// An optional `attributes {..}` may follow the type.
///
/// The operation is pure: it has no effects, is removed when unused, and two of the same type are
/// merged. It does not fold.
#[derive(EffectOpInterface)]
#[operation(
    dialect = BuiltinDialect,
    traits(Pure, AlwaysSpeculatable),
    implements(ConditionallySpeculatable, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Zero {
    /// The zero
    #[result]
    result: AnyType,
}

impl ConditionallySpeculatable for Zero {
    fn speculatability(&self) -> Speculatability {
        Speculatability::Speculatable
    }
}

impl OpPrinter for Zero {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        *printer += const_text(" : ");
        printer.print_type(self.result().ty());
        print_attributes(printer, self.as_operation());
    }
}

/// Reads `: type`, then an optional `attributes {..}`.
impl OpParser for Zero {
    fn parse(state: &mut OperationState, parser: &mut dyn crate::OpAsmParser<'_>) -> ParseResult {
        let ty = parser.parse_colon_type()?.into_inner();
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        state.results.push(ty);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, string::String, vec::Vec};

    use super::*;
    use crate::{
        Operation, OperationRef, SourceSpan,
        dialects::builtin::{BuiltinDialect, BuiltinOpBuilder, Module},
        parse::{ParserConfig, parse_any},
        testing::{
            Test,
            text::{describe, round_trip},
        },
    };

    /// A module with a global of `ty` initialized by `init`, and a function holding `body`.
    fn module(ty: &str, init: &str, body: &str) -> String {
        format!(
            "builtin.module public @m {{
    builtin.global_variable public @data : {ty} {{
        {init}
    }};
    builtin.function public extern(\"C\") @f() {{
{body}
        builtin.ret;
    }};
}};"
        )
    }

    fn parse_text(source: &str) -> Result<(Rc<Context>, OperationRef), String> {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context.clone());
        match parse_any(config, "bytes", source).into_result() {
            Ok(op) => Ok((context, op)),
            Err(report) => Err(describe(&report)),
        }
    }

    fn all_bytes(root: OperationRef) -> Vec<OperationRef> {
        let mut found = Vec::new();
        root.borrow().prewalk_all(|op: &Operation| {
            if op.is::<Bytes>() {
                found.push(op.as_operation_ref());
            }
        });
        found
    }

    /// The bytes are written most significant byte first, as `#builtin.bytes<..>` writes them; the
    /// result is an `array<i8; N>`; the module round trips in both forms.
    #[test]
    fn bytes_round_trip_in_both_forms() {
        let source = module(
            "array<i8; 2>",
            "%b = builtin.bytes \"0201\" : array<i8; 2>;\n        builtin.yield %b;",
            "        %e = builtin.bytes \"\" : array<i8; 0>;
        %c = builtin.bytes \"ff00a1\" : array<i8; 3> attributes { note = #builtin.i32<7> };",
        );
        let (_context, module) = parse_text(&source).unwrap_or_else(|err| panic!("{err}"));
        let found = all_bytes(module);
        let data = found
            .iter()
            .map(|op| {
                let op = op.borrow();
                let bytes = op.downcast_ref::<Bytes>().unwrap();
                (bytes.data().as_slice().to_vec(), bytes.result().ty().to_string())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            data,
            [
                (alloc::vec![1u8, 2], String::from("array<i8; 2>")),
                (alloc::vec![], String::from("array<i8; 0>")),
                (alloc::vec![0xa1, 0x00, 0xff], String::from("array<i8; 3>")),
            ]
        );
        let generic = crate::testing::text::print_generic(&found[0].borrow());
        assert!(generic.contains("value = #builtin.bytes<\"0201\">"), "{generic}");

        let mut skipped = Vec::new();
        let failures = round_trip::<Module>(module, true, &mut skipped);
        assert!(failures.is_empty(), "{}", failures.join("\n\n"));
        assert!(skipped.is_empty(), "skipped: {skipped:#?}");
    }

    /// The digits of `builtin.bytes` are the data as one little-endian number, most significant byte
    /// first: `"0a0b0c0d"` is the bytes `0d 0c 0b 0a` in memory order, and prints as it was written.
    #[test]
    fn a_byte_string_is_written_most_significant_byte_first() {
        let text = "%0 = builtin.bytes \"0a0b0c0d\" : array<i8; 4>;";
        let (_context, module) =
            parse_text(&module("array<i8; 4>", &format!("{text}\n        builtin.yield %0;"), ""))
                .unwrap_or_else(|err| panic!("{err}"));
        let found = all_bytes(module);
        assert_eq!(found.len(), 1);
        let op = found[0].borrow();
        let bytes = op.downcast_ref::<Bytes>().unwrap();
        assert_eq!(bytes.data().as_slice(), [0x0d, 0x0c, 0x0b, 0x0a]);
        assert_eq!(op.to_string(), text);
    }

    /// A result type that is not the array of the bytes is a diagnostic, in either form; so are
    /// digits that do not make bytes: an odd number of them, or digits that are not hex.
    #[test]
    fn bytes_of_another_type_are_an_error() {
        let err =
            parse_text(&module("i8", "", "        %b = builtin.bytes \"0201\" : array<i8; 3>;"))
                .map(|_| ())
                .expect_err("three bytes are not two");
        assert!(err.contains("invalid operation builtin.bytes"), "{err}");
        assert!(
            err.contains("the result has type `array<i8; 3>`, but 2 byte(s) make a `array<i8; 2>`"),
            "{err}"
        );

        let err = parse_text(&module("i8", "", "        %b = builtin.bytes \"0201\" : i16;"))
            .map(|_| ())
            .expect_err("the result is an array");
        assert!(err.contains("builtin.bytes"), "{err}");
        assert!(err.contains("expected an array of 8-bit integral type, got i16"), "{err}");

        let err =
            parse_text(&module("i8", "", "        %b = builtin.bytes \"021\" : array<i8; 2>;"))
                .map(|_| ())
                .expect_err("an odd number of digits");
        assert!(err.contains("even number of hex digits"), "{err}");

        let err =
            parse_text(&module("i8", "", "        %b = builtin.bytes \"zz\" : array<i8; 1>;"))
                .map(|_| ())
                .expect_err("digits that are not hex");
        assert!(err.contains("invalid hex-encoded data: contains invalid hex digits"), "{err}");

        let generic = "        %b = \"builtin.bytes\"() <{ value = #builtin.bytes<\"0201\"> }> : \
                       () -> array<i8; 1>;";
        let err = parse_text(&module("i8", "", generic))
            .map(|_| ())
            .expect_err("the generic form is checked");
        assert!(err.contains("2 byte(s) make a `array<i8; 2>`"), "{err}");
    }

    /// The builders give the result the type they are given; both operations are pure (no
    /// effects, dead when unused), neither folds, and an aggregate whose operands do not match its
    /// elements is rejected by the verifier, not by the builder.
    #[test]
    fn the_builders_make_pure_operations_of_the_type_given() {
        use crate::{
            ValueRef,
            dialects::builtin::types::{StructField, StructType},
            traits::Foldable,
        };

        let mut test = Test::new("f", |cx| ([cx.i32(), cx.i64()], []));
        let context = test.context_rc();
        let (i32, i64) = (context.i32(), context.i64());
        let pair = context.struct_type(StructType::new([i32, i64].map(StructField::new)));
        let arguments = test
            .entry_block()
            .borrow()
            .arguments()
            .iter()
            .map(|argument| *argument as ValueRef)
            .collect::<Vec<_>>();
        let mut builder = test.function_builder();
        let zero = builder.zero(i64, SourceSpan::UNKNOWN).unwrap();
        let built = builder.aggregate(arguments.clone(), pair, SourceSpan::UNKNOWN).unwrap();
        let wrong = builder.aggregate([arguments[0]], pair, SourceSpan::UNKNOWN).unwrap();
        {
            let zero = zero.borrow();
            let built = built.borrow();
            assert_eq!(zero.result().ty(), i64);
            assert_eq!(built.result().ty(), pair);
            assert_eq!(zero.as_operation().to_string(), "%2 = builtin.zero : i64;");
            assert_eq!(
                built.as_operation().to_string(),
                "%3 = builtin.aggregate %0, %1 : struct<i32, i64>;"
            );
            for op in [zero.as_operation(), built.as_operation()] {
                assert!(op.verify().is_ok());
                assert!(op.is_memory_effect_free(), "{op}");
                assert!(op.is_trivially_dead(), "{op}");
            }
            assert!(!zero.as_operation().implements::<dyn Foldable>());
            assert!(!built.as_operation().implements::<dyn Foldable>());
        }
        let err = wrong.borrow().as_operation().verify().unwrap_err();
        assert!(
            describe(&err)
                .contains("`struct<i32, i64>` has 2 element(s), but 1 operand(s) are given"),
            "{}",
            describe(&err)
        );
    }

    /// The builder makes the array type from the data; the constant folds to its attribute, and
    /// the builtin dialect materializes a bytes attribute only at the array type of its bytes.
    #[test]
    fn bytes_fold_to_their_attribute_and_materialize_from_it() {
        use crate::traits::{FoldResult, Foldable, OpFoldResult};

        let mut test = Test::new("f", |_| ([], []));
        let context = test.context_rc();
        let dialect = context.get_or_register_dialect::<BuiltinDialect>();
        let mut builder = test.function_builder();
        let built = builder.bytes([1u8, 2, 3], SourceSpan::UNKNOWN).unwrap();
        let array = context.array_type(context.i8(), 3);
        {
            let built = built.borrow();
            assert_eq!(built.result().ty(), array);
            assert_eq!(built.data().as_slice(), [1, 2, 3]);
            assert!(built.as_operation().verify().is_ok());
            assert!(built.as_operation().is_memory_effect_free());
            assert!(
                built
                    .as_operation()
                    .to_string()
                    .contains("builtin.bytes \"030201\" : array<i8; 3>")
            );
        }

        let attr = built.borrow().value.as_attribute_ref();
        let mut results = SmallVec::<[OpFoldResult; 1]>::default();
        assert!(matches!(built.borrow().fold(&mut results), FoldResult::Ok(())));
        match results.as_slice() {
            [OpFoldResult::Attribute(folded)] => assert!(AttributeRef::ptr_eq(folded, &attr)),
            other => panic!("expected the attribute, got {other:?}"),
        }

        let builder = builder.builder_mut();
        let op = dialect
            .materialize_constant(builder, attr, array, SourceSpan::UNKNOWN)
            .expect("the bytes are materialized at their array type");
        let op = op.borrow();
        let bytes = op.downcast_ref::<Bytes>().unwrap();
        assert_eq!(bytes.result().ty(), array);
        assert!(AttributeRef::ptr_eq(&op.get_attribute("value").unwrap(), &attr));

        let other = context.array_type(context.i8(), 2);
        assert!(
            dialect
                .materialize_constant(builder, attr, other, SourceSpan::UNKNOWN)
                .is_none()
        );
        let integer = context.int_attr(crate::Int::from(1i8), context.i8()).as_attribute_ref();
        assert!(
            dialect
                .materialize_constant(builder, integer, context.i8(), SourceSpan::UNKNOWN)
                .is_none()
        );
    }
}
