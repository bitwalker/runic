use alloc::{format, string::String};

use smallvec::SmallVec;

use super::signature::next_token_span;
use crate::{
    AttrPrinter, AttributeRef, Type,
    attributes::{AttrParser, AttributeElements, ElementReplacer, ElementVisitor},
    derive::DialectAttribute,
    diagnostics::{SourceSpan, TextRange},
    dialects::builtin::{
        BuiltinDialect,
        types::{FloatType, IntegerType, PointerType},
    },
    formatter::const_text,
    layout::{Align, DataLayoutSpec, Endianness, InvalidDataLayout, PointerSpec},
    parse::{ParseResult, Parser, ParserError, ParserExt},
    print::AsmPrinter,
};

/// A [DataLayoutSpec], as an operation states it in its attribute
/// [DATA_LAYOUT_ATTR](crate::layout::DATA_LAYOUT_ATTR).
///
/// Text: the endianness, then entries keyed by a type, written as the type is. Widths are in bits,
/// alignments in bytes:
///
/// ```text
/// #builtin.data_layout<little, ptr = 32>
/// #builtin.data_layout<big, ptr = 64, ptr<1> = 64 align 8 index 32, i64 = 4, i128 = 8, f64 = 4>
/// ```
///
/// * `ptr = S` (or `ptr<N> = S` for address space `N`) is a pointer of `S` bits, aligned to `S / 8`
///   bytes, with an index of `S` bits; `align A` and `index I` after the size state the alignment
///   and the index width. The pointer of address space 0 is required.
/// * `iN = A` aligns integers of `N` bits to `A` bytes; `f32 = A` and `f64 = A` align the floats.
///
/// The printed text is canonical: specifications that answer every query alike (and so are
/// equal) print the same text, and the text parses back to an equal specification. It lists the
/// endianness; the pointers by address space (leaving out an address space other than 0 that has
/// the pointer of 0, and an `align` or `index` equal to what the size implies); integer entries by
/// ascending width; then `f32` and `f64` if they differ from their defaults. The integer entries
/// are the specification's alignment at the last width of each run of widths with one alignment
/// but the last run, at each default width, and at the first width of the last run when no
/// default width lies in it, leaving out every entry equal to the default for its width. Parsed,
/// they are added to the defaults and give the same alignment at every width.
///
/// The attribute has no default: `#builtin.data_layout` without a body is an error.
#[derive(DialectAttribute)]
#[attribute(
    dialect = BuiltinDialect,
    remote = "DataLayoutSpec",
    no_default,
    implements(AttrPrinter)
)]
#[allow(dead_code)]
// Exists only to drive the derive, which names the attribute `<marker>Attr` and its text after the
// marker: it is not `crate::layout::DataLayout`, and is never used as a value.
struct DataLayout;

/// A data layout holds no types or attributes: its entries are keyed by kinds of type (a pointer
/// of an address space, an integer width, a float kind), not by types.
impl AttributeElements for DataLayoutSpec {
    #[inline]
    fn visit_elements(&self, _visitor: &mut dyn ElementVisitor) {}

    #[inline]
    fn replace_elements(&self, _replacer: &mut dyn ElementReplacer) -> Option<Self> {
        None
    }
}

impl AttrPrinter for DataLayoutAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let spec = self.as_value();
        printer.print_keyword(spec.endianness().as_str());
        for (addrspace, pointer) in spec.distinct_pointers() {
            let key = printer.context().ptr_type_in(None, addrspace);
            print_key(printer, key);
            printer.print_decimal_integer(pointer.size_bits);
            if pointer.shorthand_align() != Some(pointer.align) {
                *printer += const_text(" align ");
                printer.print_decimal_integer(pointer.align.bytes());
            }
            if pointer.index_bits != pointer.size_bits {
                *printer += const_text(" index ");
                printer.print_decimal_integer(pointer.index_bits);
            }
        }
        for (width, align) in spec.printed_integers() {
            let key = printer.context().int_type(width);
            print_key(printer, key);
            printer.print_decimal_integer(align.bytes());
        }
        for (float, align) in spec.printed_floats() {
            let key = match float {
                FloatType::F32 => printer.context().f32(),
                FloatType::F64 => printer.context().f64(),
            };
            print_key(printer, key);
            printer.print_decimal_integer(align.bytes());
        }
    }
}

/// Writes `, key = `.
fn print_key(printer: &mut AsmPrinter<'_>, key: Type) {
    *printer += const_text(", ");
    printer.print_type(key);
    *printer += const_text(" = ");
}

/// What an entry of the text is keyed by.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Key {
    Pointer(u32),
    Integer(u32),
    Float(FloatType),
}

/// An entry of the text, with its span.
struct Entry {
    span: SourceSpan,
    key: Key,
    value: EntryValue,
}

enum EntryValue {
    Pointer(PointerSpec),
    Align(Align),
}

impl AttrParser for DataLayoutAttr {
    /// Reads `endianness (',' key '=' value)*`.
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<AttributeRef> {
        let start = next_token_span(parser)?.range().start();
        let endianness = parse_endianness(parser)?;
        let mut entries = SmallVec::<[Entry; 8]>::new();
        while parser.parse_optional_comma()? {
            let entry = parse_entry(parser)?;
            if entries.iter().any(|seen: &Entry| seen.key == entry.key) {
                return Err(invalid(entry.span, "this key is given more than once"));
            }
            entries.push(entry);
        }
        let body = span_from(parser, start);

        let Some(pointer) = entries.iter().find_map(|entry| match (entry.key, &entry.value) {
            (Key::Pointer(0), EntryValue::Pointer(pointer)) => Some((entry.span, *pointer)),
            _ => None,
        }) else {
            return Err(invalid(
                body,
                "a data layout gives the pointer of address space 0, as in `ptr = 32`",
            ));
        };
        let mut spec = DataLayoutSpec::new(endianness, pointer.1)
            .map_err(|err| invalid_layout(pointer.0, err))?;
        for entry in entries {
            spec = match (entry.key, entry.value) {
                (Key::Pointer(0), _) => Ok(spec),
                (Key::Pointer(addrspace), EntryValue::Pointer(pointer)) => {
                    spec.with_pointer(addrspace, pointer)
                }
                (Key::Integer(width), EntryValue::Align(align)) => {
                    spec.with_integer_align(width, align)
                }
                (Key::Float(float), EntryValue::Align(align)) => {
                    Ok(spec.with_float_align(float, align))
                }
                _ => unreachable!("an entry's value is parsed by its key"),
            }
            .map_err(|err| invalid_layout(entry.span, err))?;
        }
        Ok(parser
            .context_rc()
            .create_attribute::<DataLayoutAttr, _>(spec)
            .as_attribute_ref())
    }
}

/// Reads `little` or `big`.
fn parse_endianness(parser: &mut dyn Parser<'_>) -> ParseResult<Endianness> {
    if parser.parse_optional_custom_keyword("little")?.is_some() {
        return Ok(Endianness::Little);
    }
    if parser.parse_optional_custom_keyword("big")?.is_some() {
        return Ok(Endianness::Big);
    }
    let span = next_token_span(parser)?;
    Err(invalid(span, "a data layout starts with its endianness: `little` or `big`"))
}

/// Reads `key '=' value`.
fn parse_entry(parser: &mut dyn Parser<'_>) -> ParseResult<Entry> {
    let start = next_token_span(parser)?.range().start();
    let ty = parser.parse_type()?.into_inner();
    let key = entry_key(ty).ok_or_else(|| {
        invalid(
            span_from(parser, start),
            &format!(
                "`{ty}` is not a data layout key: expected `ptr`, `ptr<N>`, an integer type or a \
                 float type"
            ),
        )
    })?;
    parser.parse_equal()?;
    let value = match key {
        Key::Pointer(_) => EntryValue::Pointer(parse_pointer(parser, start)?),
        Key::Integer(_) | Key::Float(_) => EntryValue::Align(parse_align(parser)?),
    };
    Ok(Entry {
        span: span_from(parser, start),
        key,
        value,
    })
}

/// The key `ty` is, if it is one: an opaque pointer, an integer type or a float type.
fn entry_key(ty: Type) -> Option<Key> {
    if let Some(pointer) = ty.downcast_ref::<PointerType>() {
        pointer.is_opaque().then(|| Key::Pointer(pointer.addrspace()))
    } else if let Some(integer) = ty.downcast_ref::<IntegerType>() {
        Some(Key::Integer(integer.width()))
    } else {
        ty.downcast_ref::<FloatType>().map(|float| Key::Float(*float))
    }
}

/// Reads `size ('align' bytes)? ('index' bits)?`, of the pointer entry starting at `start`.
fn parse_pointer(parser: &mut dyn Parser<'_>, start: u32) -> ParseResult<PointerSpec> {
    let size_bits = parser.parse_decimal_integer::<u32>()?.into_inner();
    let align = if parser.parse_optional_custom_keyword("align")?.is_some() {
        Some(parse_align(parser)?)
    } else {
        None
    };
    let index_bits = if parser.parse_optional_custom_keyword("index")?.is_some() {
        parser.parse_decimal_integer::<u32>()?.into_inner()
    } else {
        size_bits
    };
    let align = match align.or_else(|| PointerSpec::sized(size_bits).map(|sized| sized.align)) {
        Some(align) => align,
        // The size is invalid: the specification reports it
        None if !PointerSpec::is_valid_size(size_bits) => Align::ONE,
        None => {
            return Err(invalid(
                span_from(parser, start),
                &format!(
                    "a pointer of {size_bits} bits has no implied alignment ({size_bits} / 8 \
                     bytes is not a power of two): give one with `align`"
                ),
            ));
        }
    };
    Ok(PointerSpec {
        size_bits,
        align,
        index_bits,
    })
}

/// Reads an alignment in bytes: a power of two from 1 to 32768.
fn parse_align(parser: &mut dyn Parser<'_>) -> ParseResult<Align> {
    let (span, bytes) = parser.parse_decimal_integer::<u64>()?.into_parts();
    Align::from_bytes(bytes).ok_or_else(|| {
        invalid(
            span,
            &format!(
                "an alignment of {bytes} bytes: expected a power of two from 1 to {}",
                Align::MAX_BYTES
            ),
        )
    })
}

/// The span from `start` to the end of the last token read.
fn span_from(parser: &mut dyn Parser<'_>, start: u32) -> SourceSpan {
    let end = parser.token_stream().previous_end().max(start);
    SourceSpan::session(
        parser.source_id(),
        TextRange::new(start, end).expect("the span ends after it starts"),
    )
}

fn invalid(span: SourceSpan, reason: &str) -> ParserError {
    ParserError::InvalidAttributeValue {
        span,
        reason: String::from(reason),
    }
}

fn invalid_layout(span: SourceSpan, err: InvalidDataLayout) -> ParserError {
    invalid(span, &format!("{err}"))
}
