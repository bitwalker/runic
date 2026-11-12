//! The type parser: the syntax of the builtin types, type aliases and dialect types.
//!
//! ```text
//! type          := '?' | 'never' | integer-type | 'isize' | 'f32' | 'f64'
//!                | pointer-type | array-type | struct-type | function-type
//!                | type-alias | dialect-type
//! integer-type  := 'i' [1-9][0-9]*
//! pointer-type  := 'ptr' ('<' (address-space | type (',' address-space)?) '>')?
//! address-space := decimal-literal
//! array-type    := 'array' '<' type ';' decimal-literal '>'
//! struct-type   := 'struct' struct-name? '<' (struct-repr ';')? (struct-field (',' struct-field)*)? '>'
//! struct-name   := bare-identifier | string
//! struct-repr   := 'align' '(' N ')' | 'packed' ('(' N ')')? | 'transparent'
//! struct-field  := (field-name ':')? type ('align' '(' N ')')?
//! field-name    := bare-identifier | string
//! function-type := '(' (type (',' type)*)? ')' '->' (type | '(' (type (',' type)*)? ')')
//! type-alias    := '!' alias-name
//! dialect-type  := '!' dialect-namespace '.' type-name ('<' type-body '>')?
//! ```
//!
//! A function type's results are a parenthesised list whenever they start with `(`, so a single
//! result that is itself a function type is written in parentheses: `() -> (() -> i32)`. A
//! one-element list `(T)` is the single result `T`.
//!
//! The body of a dialect type is parsed by the type's [TypeParser](crate::TypeParser), which is
//! handed the parser positioned just after the `<` and must stop before the closing `>`.

use alloc::{
    format,
    string::{String, ToString},
};
use core::num::NonZeroU16;

use super::{ParseResult, Parser, ParserError, ParserExt, Token};
use crate::{
    IntegerType, SmallVec, StructField, StructRepr, StructType, Type,
    diagnostics::{SourceSpan, Span},
    interner::Symbol,
};

/// The number of types a type may be nested in. Each level of nesting is a few recursive calls,
/// so the limit keeps deeply nested input from overflowing the stack.
pub(super) const MAX_TYPE_NESTING: u32 = 256;

/// Parses a type if the next token can start one, and returns `None` otherwise.
///
/// Fails if the type would be nested in more than [MAX_TYPE_NESTING] types.
pub(super) fn parse_optional_type<'input, P: Parser<'input>>(
    parser: &mut P,
) -> ParseResult<Option<Span<Type>>> {
    let enclosing = parser.state().type_nesting;
    if enclosing > MAX_TYPE_NESTING {
        return Err(type_nesting_too_deep(parser));
    }
    // Restored on every return from the nested parse, successful or not, so that the count
    // always matches the types actually being parsed.
    parser.state_mut().type_nesting = enclosing + 1;
    let result = parse_optional_type_nested(parser);
    parser.state_mut().type_nesting = enclosing;
    result
}

/// The error for a type nested in more than [MAX_TYPE_NESTING] types, at the next token.
#[cold]
#[inline(never)]
fn type_nesting_too_deep<'input, P: Parser<'input>>(parser: &mut P) -> ParserError {
    let span = match parser.token_stream_mut().peek() {
        Ok(Some((start, _, end))) => span!(parser.source_id(), start, end),
        Ok(None) => parser.current_location(),
        Err(err) => return err,
    };
    ParserError::InvalidType {
        span,
        reason: format!("type nesting is too deep (limit {MAX_TYPE_NESTING})"),
    }
}

/// [parse_optional_type], once the nesting has been accounted for.
///
/// Each kind of type is parsed by a function of its own, so that this frame, which is live at
/// every level of a nested type, stays small.
fn parse_optional_type_nested<'input, P: Parser<'input>>(
    parser: &mut P,
) -> ParseResult<Option<Span<Type>>> {
    let (start, token, end) = match parser.token_stream_mut().peek() {
        Ok(Some(next)) => next,
        Ok(None) => return Ok(None),
        Err(err) => return Err(err),
    };
    let ty = match token {
        Token::Ptr => parse_pointer_type(parser),
        Token::Array => parse_array_type(parser),
        Token::Struct => parse_struct_type(parser),
        Token::Lparen => parse_function_type(parser),
        Token::BangIdent(id) => parse_alias_or_dialect_type(parser, id, start, end),
        token => match parse_scalar_type(parser, token, start, end) {
            Some(ty) => ty,
            None => return Ok(None),
        },
    };
    match ty {
        Ok(ty) => Ok(Some(spanned_from(parser, start, ty))),
        Err(err) => Err(err),
    }
}

/// `ty`, spanning the source from `start` to the end of the last token consumed.
fn spanned_from<'input, P: Parser<'input>>(parser: &mut P, start: u32, ty: Type) -> Span<Type> {
    let end = parser.token_stream().previous_end();
    Span::new(span!(parser.source_id(), start, end), ty)
}

/// Parses a type that is a single token, if `token`, the next one, is such a type.
fn parse_scalar_type<'input, P: Parser<'input>>(
    parser: &mut P,
    token: Token<'input>,
    start: u32,
    end: u32,
) -> Option<ParseResult<Type>> {
    let span = span!(parser.source_id(), start, end);
    let context = parser.context();
    let ty = match token {
        Token::Question => Ok(context.unknown_type()),
        Token::Never => Ok(context.never_type()),
        Token::Isize => Ok(context.isize()),
        Token::F32 => Ok(context.f32()),
        Token::F64 => Ok(context.f64()),
        Token::IntType(width) => {
            context.try_int_type(width).ok_or_else(|| ParserError::InvalidType {
                span,
                reason: format!("an integer type is 1 to {} bits wide", IntegerType::MAX_WIDTH),
            })
        }
        _ => return None,
    };
    Some(parser.token_stream_mut().next().and(ty))
}

/// Parses a type, failing if the next token cannot start one.
fn parse_type<'input, P: Parser<'input>>(parser: &mut P) -> ParseResult<Type> {
    if let Some(ty) = parse_optional_type(parser)? {
        return Ok(ty.into_inner());
    }
    match parser.token_stream_mut().next()? {
        Some(token) => Err(ParserError::UnexpectedToken {
            span: token.span(),
            token: token.to_string(),
            expected: Some("type".into()),
        }),
        None => Err(ParserError::UnexpectedEof {
            expected: alloc::vec!["type".into()],
        }),
    }
}

/// Parses `ptr`, followed by nothing, or by `<N>`, `<T>` or `<T, N>`.
fn parse_pointer_type<'input, P: Parser<'input>>(parser: &mut P) -> ParseResult<Type> {
    parser.token_stream_mut().next()?;
    let context = parser.context_rc();
    if !parser.parse_optional_langle()? {
        return Ok(context.opaque_ptr_type());
    }
    let (pointee, addrspace) = match parser.parse_optional_decimal_integer::<u32>()? {
        Some(addrspace) => (None, addrspace.into_inner()),
        None => {
            let pointee = parse_type(parser)?;
            let addrspace = if parser.parse_optional_comma()? {
                parser.parse_decimal_integer::<u32>()?.into_inner()
            } else {
                0
            };
            (Some(pointee), addrspace)
        }
    };
    parser.parse_rangle()?;
    Ok(context.ptr_type_in(pointee, addrspace))
}

/// Parses `array<T; N>`.
fn parse_array_type<'input, P: Parser<'input>>(parser: &mut P) -> ParseResult<Type> {
    parser.token_stream_mut().next()?;
    parser.parse_langle()?;
    let element = parse_type(parser)?;
    parser.parse_semicolon()?;
    let len = parser.parse_decimal_integer::<u64>()?.into_inner();
    parser.parse_rangle()?;
    Ok(parser.context().array_type(element, len))
}

/// Parses `struct`, an optional name, then the representation and fields in angle brackets.
fn parse_struct_type<'input, P: Parser<'input>>(parser: &mut P) -> ParseResult<Type> {
    parser.token_stream_mut().next()?;
    let name = if parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Langle)) {
        None
    } else {
        Some(parse_name(parser, "struct name or '<'")?)
    };
    parser.parse_langle()?;

    let repr = parse_optional_struct_repr(parser)?;
    if repr.is_some() {
        parser.parse_semicolon()?;
    }

    let mut fields = SmallVec::<[StructField; 4]>::default();
    if !parser.parse_optional_rangle()? {
        loop {
            fields.push(parse_struct_field(parser)?);
            if !parser.parse_optional_comma()? {
                break;
            }
        }
        parser.parse_rangle()?;
    }

    let ty = match name {
        Some(name) => StructType::named(name, fields),
        None => StructType::new(fields),
    };
    Ok(parser.context().struct_type(ty.with_repr(repr.unwrap_or_default())))
}

/// Parses a struct representation, if the next token starts one.
fn parse_optional_struct_repr<'input, P: Parser<'input>>(
    parser: &mut P,
) -> ParseResult<Option<StructRepr>> {
    // `align: i32` is a field named `align`, not a representation.
    if is_next_token_followed_by_a_colon(parser) {
        return Ok(None);
    }
    let Some(repr) = parser.token_stream_mut().next_if_map(|tok| match tok {
        Token::BareIdent(repr @ ("align" | "packed" | "transparent")) => Some(repr),
        _ => None,
    })?
    else {
        return Ok(None);
    };
    Ok(Some(match repr.into_inner() {
        "align" => StructRepr::Align(parse_alignment(parser)?),
        "packed" if parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Lparen)) => {
            StructRepr::Packed(parse_alignment(parser)?)
        }
        "packed" => StructRepr::Packed(NonZeroU16::MIN),
        _ => StructRepr::Transparent,
    }))
}

/// Parses a struct field: an optional name, its type and an optional alignment.
fn parse_struct_field<'input, P: Parser<'input>>(parser: &mut P) -> ParseResult<StructField> {
    let name = if is_next_field_name(parser) {
        let name = parse_name(parser, "field name")?;
        parser.parse_colon()?;
        Some(name)
    } else {
        None
    };
    let mut field = StructField::new(parse_type(parser)?);
    field.name = name;
    if parser.token_stream_mut().next_if_eq(Token::BareIdent("align"))? {
        field = field.aligned(parse_alignment(parser)?);
    }
    Ok(field)
}

/// Returns true if the next token names a struct field rather than starting its type.
///
/// A string or a bare identifier can only be a name. A keyword such as `ptr` can be either, so it
/// is a name when the next character after it is a single `:`.
fn is_next_field_name<'input, P: Parser<'input>>(parser: &mut P) -> bool {
    let Ok(Some((_, token, _))) = parser.token_stream_mut().peek() else {
        return false;
    };
    match token {
        Token::String(_) | Token::BareIdent(_) => true,
        token if token.is_keyword() => is_next_token_followed_by_a_colon(parser),
        _ => false,
    }
}

/// Returns true if the source just after the upcoming token starts with a single `:`.
fn is_next_token_followed_by_a_colon<'input, P: Parser<'input>>(parser: &mut P) -> bool {
    if !matches!(parser.token_stream_mut().peek(), Ok(Some(_))) {
        return false;
    }
    // The remaining source starts just after the upcoming token.
    let rest = parser.token_stream().remaining_source().trim_start();
    rest.starts_with(':') && !rest.starts_with("::")
}

/// Parses a struct or field name: a bare identifier (which may be spelled like a keyword) or a
/// quoted string, unescaped as the printer escapes it.
fn parse_name<'input, P: Parser<'input>>(parser: &mut P, expected: &str) -> ParseResult<Symbol> {
    let Some((start, token, end)) = parser.token_stream_mut().peek()? else {
        return Err(ParserError::UnexpectedEof {
            expected: alloc::vec![expected.into()],
        });
    };
    // A keyword is named by its own spelling, which is read before the token is consumed.
    let keyword = token
        .is_keyword()
        .then(|| Symbol::intern(parser.token_stream().slice(start, end)));
    parser.token_stream_mut().next()?;
    let span = span!(parser.source_id(), start, end);
    match token {
        Token::BareIdent(name) => Ok(Symbol::intern(name)),
        Token::String(quoted) => {
            unescape(quoted).map(Symbol::intern).ok_or_else(|| ParserError::InvalidString {
                span,
                reason: "invalid escape sequence: a quoted name is escaped as by \
                         `str::escape_default`"
                    .into(),
            })
        }
        token => keyword.ok_or_else(|| ParserError::UnexpectedToken {
            span,
            token: token.to_string(),
            expected: Some(expected.into()),
        }),
    }
}

/// Undoes [str::escape_default], which the printer uses to quote a name.
fn unescape(quoted: &str) -> Option<String> {
    let mut unescaped = String::with_capacity(quoted.len());
    let mut chars = quoted.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            unescaped.push(c);
            continue;
        }
        unescaped.push(match chars.next()? {
            't' => '\t',
            'r' => '\r',
            'n' => '\n',
            c @ ('\'' | '"' | '\\') => c,
            'u' => {
                if chars.next()? != '{' {
                    return None;
                }
                let mut code = 0u32;
                let mut digits = 0;
                loop {
                    match chars.next()? {
                        '}' if digits > 0 => break,
                        c => {
                            code = code.checked_mul(16)?.checked_add(c.to_digit(16)?)?;
                            digits += 1;
                        }
                    }
                }
                char::from_u32(code)?
            }
            _ => return None,
        });
    }
    Some(unescaped)
}

/// Parses `(N)`, a non-zero alignment.
fn parse_alignment<'input, P: Parser<'input>>(parser: &mut P) -> ParseResult<NonZeroU16> {
    parser.parse_lparen()?;
    let (span, align) = parser.parse_decimal_integer::<u16>()?.into_parts();
    parser.parse_rparen()?;
    NonZeroU16::new(align).ok_or(ParserError::InvalidIntegerLiteral {
        span,
        reason: "expected a non-zero alignment".into(),
    })
}

/// Parses a function type, `(params) -> results`.
fn parse_function_type<'input, P: Parser<'input>>(parser: &mut P) -> ParseResult<Type> {
    let start = parser.token_stream_mut().peek()?.map_or(0, |(start, ..)| start);
    let params = parse_parenthesised_types(parser)?;
    if !parser.parse_optional_arrow()? {
        let end = parser.token_stream().previous_end();
        return Err(ParserError::InvalidType {
            span: span!(parser.source_id(), start, end),
            reason: "expected '->': a parenthesised type list is only a type as the parameters of \
                     a function type (use `struct<..>` for an aggregate)"
                .into(),
        });
    }
    let results = if parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Lparen)) {
        parse_parenthesised_types(parser)?
    } else {
        SmallVec::from_iter([parse_type(parser)?])
    };
    Ok(parser.context().function_type(params, results))
}

/// Parses `(T, U, ..)`, a parenthesised and possibly empty list of types.
fn parse_parenthesised_types<'input, P: Parser<'input>>(
    parser: &mut P,
) -> ParseResult<SmallVec<[Type; 4]>> {
    parser.parse_lparen()?;
    let mut types = SmallVec::default();
    if !parser.parse_optional_rparen()? {
        loop {
            types.push(parse_type(parser)?);
            if !parser.parse_optional_comma()? {
                break;
            }
        }
        parser.parse_rparen()?;
    }
    Ok(types)
}

/// How the builtin type registered as `name` is written, if there is such a type.
pub(super) fn bare_builtin_spelling(name: &str) -> Option<&'static str> {
    Some(match name {
        "int" => "`iN`, such as `i32`",
        "isize" => "`isize`",
        "float" => "`f32` or `f64`",
        "ptr" => "`ptr`, `ptr<N>`, `ptr<T>` or `ptr<T, N>`",
        "array" => "`array<T; N>`",
        "struct" => "`struct<..>`",
        "function" => "`(..) -> ..`",
        "unknown" => "`?`",
        "never" => "`never`",
        _ => return None,
    })
}

/// Parses `!id`, which names a type alias, or a dialect type if it contains a `.`.
fn parse_alias_or_dialect_type<'input, P: Parser<'input>>(
    parser: &mut P,
    id: &str,
    start: u32,
    end: u32,
) -> ParseResult<Type> {
    let span = span!(parser.source_id(), start, end);
    parser.token_stream_mut().next()?;
    let Some((dialect, name)) = id.split_once('.') else {
        let alias = Symbol::intern(id);
        return parser
            .state()
            .symbols
            .type_alias_definitions
            .get(&alias)
            .map(|ty| *ty.inner())
            .ok_or(ParserError::UndefinedTypeAlias { span, id: alias });
    };

    if dialect
        == <crate::dialects::builtin::BuiltinDialect as crate::DialectRegistration>::NAMESPACE
    {
        let mut reason = String::from("builtin types are written without the '!builtin.' prefix");
        if let Some(spelling) = bare_builtin_spelling(name) {
            reason.push_str(": write ");
            reason.push_str(spelling);
        }
        return Err(ParserError::InvalidType { span, reason });
    }
    let dialect = Symbol::intern(dialect);
    let Some(registered) = parser.context().find_registered_dialect(dialect) else {
        return Err(ParserError::UnknownDialect { span, dialect });
    };
    // The registered types are sorted by name.
    let types = registered.registered_types();
    let Ok(index) = types.binary_search_by_key(&Symbol::intern(name), |ty| ty.name()) else {
        return Err(ParserError::UnknownType {
            span,
            dialect,
            name: name.into(),
        });
    };
    let type_name = types[index].clone();

    if parser.parse_optional_langle()? {
        let Some(parse) = type_name.parse_fn() else {
            return Err(ParserError::TypeTakesNoParameters {
                span,
                name: type_name.to_string(),
            });
        };
        let ty = parse(parser)?;
        if ty.name() != &type_name {
            return Err(ParserError::InvalidType {
                span,
                reason: format!(
                    "the parser of type '{type_name}' returned `{ty}`, a type of another kind"
                ),
            });
        }
        parser.parse_rangle()?;
        Ok(ty)
    } else {
        let Some(default) = type_name.default_fn() else {
            return Err(ParserError::TypeRequiresParameters {
                span,
                name: type_name.to_string(),
            });
        };
        Ok(default(parser.context()))
    }
}
