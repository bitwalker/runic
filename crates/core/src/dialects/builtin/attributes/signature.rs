use alloc::{rc::Rc, string::String, vec::Vec};
use core::{
    cell::RefCell,
    fmt::{self, Write},
};

use crate::{
    AttrPrinter, CallConv, Context, NamedAttribute, OpPrintingFlags, Type,
    attributes::{AttrParser, AttributeDict, Marker},
    derive::DialectAttribute,
    diagnostics::{SourceSpan, TextRange, report},
    dialects::builtin::BuiltinDialect,
    formatter, interner,
    parse::{ParseResult, ParsedAttrs, Parser, ParserError, Token},
    print::AsmPrinter,
    types::{Aliased, TypeAliases},
};

/// The value of the marker attributes of a parameter: they hold nothing, and print as their name
/// alone (`#builtin.zext`).
type MarkerValue = ();

/// A marker attribute for "struct return" parameters of a function.
///
/// An sret parameter is a parameter introduced when the ABI of a function requires the caller to
/// allocate memory to hold the function results, and then pass a pointer to that allocation to
/// the callee at a given parameter position (typically the first or last parameter). It is up to
/// the caller to ensure the given pointer is of sufficient size and alignment to hold the results.
#[derive(DialectAttribute)]
#[attribute(dialect = BuiltinDialect, remote = "MarkerValue", implements(AttrPrinter, Marker))]
#[allow(dead_code)]
struct Sret;

impl AttrPrinter for SretAttr {
    fn print(&self, _printer: &mut AsmPrinter<'_>) {}
}

/// A marker attribute for argument parameters or results which should be zero-extended to the
/// target architecture's native machine word size, _if and only if_ the parameter type is smaller
/// than the native machine word size.
#[derive(DialectAttribute)]
#[attribute(dialect = BuiltinDialect, remote = "MarkerValue", implements(AttrPrinter, Marker))]
#[allow(dead_code)]
struct Zext;

impl AttrPrinter for ZextAttr {
    fn print(&self, _printer: &mut AsmPrinter<'_>) {}
}

/// A marker attribute for argument parameters or results which should be sign-extended to the
/// target architecture's native machine word size, _if and only if_ the parameter type is smaller
/// than the native machine word size.
#[derive(DialectAttribute)]
#[attribute(dialect = BuiltinDialect, remote = "MarkerValue", implements(AttrPrinter, Marker))]
#[allow(dead_code)]
struct Sext;

impl AttrPrinter for SextAttr {
    fn print(&self, _printer: &mut AsmPrinter<'_>) {}
}

/// Represents how to extend an integer argument or result narrower than the target's registers
/// or stack slots to their full width.
///
/// Integer types are signless, so the extension is a property of the parameter, chosen by
/// whoever builds the signature, not of the type.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum ArgumentExtension {
    /// Do not perform any extension, high bits have undefined contents
    #[default]
    None,
    /// Zero-extend the value
    Zext,
    /// Sign-extend the value
    Sext,
}
impl fmt::Display for ArgumentExtension {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Zext => f.write_str("zext"),
            Self::Sext => f.write_str("sext"),
        }
    }
}

/// Describes a function parameter or result.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AbiParam {
    /// The type associated with this value
    pub ty: Type,
    /// The attributes associated with this value
    pub attrs: AttributeDict,
}

impl Clone for AbiParam {
    fn clone(&self) -> Self {
        let mut attrs = AttributeDict::new();
        for attr in self.attrs.iter() {
            let value = attr.value.borrow();
            let new_value = value.dyn_clone();
            let new_attr = value.context_rc().alloc_map_item(NamedAttribute {
                name: attr.name,
                value: new_value,
            });
            attrs.insert(new_attr);
        }
        Self { ty: self.ty, attrs }
    }
}

impl AbiParam {
    /// Panics, in debug builds, if `ty` is open: a parameter's type enters IR.
    #[track_caller]
    pub fn new(ty: Type) -> Self {
        Self::new_with_attribute_dict(ty, AttributeDict::new())
    }

    /// Panics, in debug builds, if `ty` is open: a parameter's type enters IR.
    #[track_caller]
    pub fn new_with_attribute_dict(ty: Type, attrs: AttributeDict) -> Self {
        crate::types::debug_assert_closed(ty, "the type of a parameter");
        Self { ty, attrs }
    }

    pub fn new_with_attrs(ty: Type, attributes: impl IntoIterator<Item = NamedAttribute>) -> Self {
        let mut attrs = AttributeDict::new();
        for attr in attributes {
            let context = attr.value.borrow().context_rc();
            context.debug_assert_ir_type(ty, "the type of a parameter");
            attrs.insert(context.alloc_map_item(attr));
        }
        Self::new_with_attribute_dict(ty, attrs)
    }

    pub fn zext(ty: Type, context: &Rc<Context>) -> Self {
        let zext = context.create_attribute::<ZextAttr, _>(());
        Self::new_with_attrs(ty, [NamedAttribute::new(interner::symbols::Extend, zext)])
    }

    pub fn sext(ty: Type, context: &Rc<Context>) -> Self {
        let sext = context.create_attribute::<SextAttr, _>(());
        Self::new_with_attrs(ty, [NamedAttribute::new(interner::symbols::Extend, sext)])
    }

    pub fn sret(ty: Type, context: &Rc<Context>) -> Self {
        assert!(ty.is_pointer(), "sret parameters must be pointers");
        let sret = context.create_attribute::<SretAttr, _>(());
        Self::new_with_attrs(ty, [NamedAttribute::new(interner::symbols::Sret, sret)])
    }

    pub fn mark_sret(&mut self, context: &Rc<Context>) {
        let sret = context.create_attribute::<SretAttr, _>(());
        let attr = context.alloc_map_item(NamedAttribute {
            name: interner::symbols::Sret,
            value: sret,
        });
        self.attrs.insert(attr);
    }

    pub fn is_sret_param(&self) -> bool {
        self.attrs.contains(&interner::symbols::Sret)
    }

    /// How the parameter is extended: the value of its `extend` attribute, `#builtin.zext` or
    /// `#builtin.sext`.
    pub fn extension(&self) -> ArgumentExtension {
        match self.attrs.find(&interner::symbols::Extend).get() {
            None => ArgumentExtension::None,
            Some(attr) => {
                let value = attr.value.borrow();
                if value.is::<ZextAttr>() {
                    ArgumentExtension::Zext
                } else if value.is::<SextAttr>() {
                    ArgumentExtension::Sext
                } else {
                    ArgumentExtension::None
                }
            }
        }
    }

    pub fn should_zero_extend(&self) -> bool {
        matches!(self.extension(), ArgumentExtension::Zext)
    }

    pub fn should_sign_extend(&self) -> bool {
        matches!(self.extension(), ArgumentExtension::Sext)
    }
}

impl fmt::Display for AbiParam {
    /// The IR text of the parameter, `type dictionary-attribute?`, as a fragment.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_with(f, [self], |writer| write_param(writer, self))
    }
}

/// A [Signature] represents the type, ABI, and linkage of a function.
///
/// A function signature provides us with all of the necessary detail to correctly
/// validate and emit code for a function, whether from the perspective of a caller,
/// or the callee.
#[derive(DialectAttribute, Default, Debug, Clone, PartialEq, Eq, Hash)]
#[attribute(
    dialect = BuiltinDialect,
    implements(AttrPrinter)
)]
pub struct Signature {
    /// The arguments expected by this function
    pub params: Vec<AbiParam>,
    /// The results returned by this function
    pub results: Vec<AbiParam>,
    /// The calling convention that applies to this function
    pub cc: CallConv,
}

impl fmt::Display for Signature {
    /// The IR text of the signature, `extern("cc") (params) -> results`, as a fragment.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_with(f, self.params.iter().chain(&self.results), |writer| {
            print_signature(writer, self)
        })
    }
}

impl Signature {
    /// Create a new signature for a function with the given calling convention and types.
    ///
    /// Each type becomes an [AbiParam] without attributes: integer types are signless, so no
    /// extension is chosen for them. `results` are the declared results: `[never]` for a function
    /// that never returns.
    pub fn with_convention<P, R>(context: &Rc<Context>, cc: CallConv, params: P, results: R) -> Self
    where
        P: IntoIterator<Item = Type>,
        R: IntoIterator<Item = Type>,
    {
        let param = |ty: Type| {
            context.debug_assert_ir_type(ty, "the type of a parameter");
            AbiParam::new(ty)
        };
        Self {
            params: params.into_iter().map(param).collect(),
            results: results.into_iter().map(param).collect(),
            cc,
        }
    }

    /// Create a new signature for a function with the default calling convention and given types.
    ///
    /// See [Self::with_convention].
    pub fn new<P, R>(context: &Rc<Context>, params: P, results: R) -> Self
    where
        P: IntoIterator<Item = Type>,
        R: IntoIterator<Item = Type>,
    {
        Self::with_convention(context, Default::default(), params, results)
    }

    /// Get the calling convention of this function
    pub fn calling_convention(&self) -> CallConv {
        self.cc
    }

    /// Returns the number of arguments expected by this function
    pub fn arity(&self) -> usize {
        self.params().len()
    }

    /// Returns a slice containing the parameters for this function
    pub fn params(&self) -> &[AbiParam] {
        self.params.as_slice()
    }

    /// Returns the parameter at `index`, if present
    #[inline]
    pub fn param(&self, index: usize) -> Option<&AbiParam> {
        self.params.get(index)
    }

    /// Returns the parameter at `index`, mutably, if present
    #[inline]
    pub fn param_mut(&mut self, index: usize) -> Option<&mut AbiParam> {
        self.params.get_mut(index)
    }

    /// The values a call produces: empty for a function that never returns.
    pub fn results(&self) -> &[AbiParam] {
        if self.never_returns() {
            &[]
        } else {
            self.results.as_slice()
        }
    }

    /// The results as declared: `[never]` for a function that never returns.
    #[inline]
    pub fn declared_results(&self) -> &[AbiParam] {
        self.results.as_slice()
    }

    /// True if the single declared result is `never`.
    pub fn never_returns(&self) -> bool {
        matches!(self.results.as_slice(), [AbiParam { ty, .. }] if ty.is_never())
    }

    /// Returns the result at `index` of [Self::results], if present
    #[inline]
    pub fn result(&self, index: usize) -> Option<&AbiParam> {
        self.results().get(index)
    }

    /// Returns the result at `index` of [Self::results], mutably, if present
    #[inline]
    pub fn result_mut(&mut self, index: usize) -> Option<&mut AbiParam> {
        if self.never_returns() {
            None
        } else {
            self.results.get_mut(index)
        }
    }
}

impl AttrPrinter for SignatureAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        print_signature(printer, self);
    }
}

impl AttrParser for SignatureAttr {
    /// Reads `'extern' '(' string ')' '(' (param (',' param)*)? ')' ('->' result-list)?`.
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<crate::AttributeRef> {
        parser.parse_custom_keyword("extern")?;
        parser.parse_lparen()?;
        let cc = CallConv::new(parser.parse_string()?.as_str());
        parser.parse_rparen()?;
        let params = parse_param_list(parser)?;
        let results = parse_optional_results(parser)?;
        let signature = Signature {
            params,
            results,
            cc,
        };
        Ok(parser.context_rc().create_attribute::<SignatureAttr, _>(signature))
    }
}

/// What the signature helpers below write to: an [AsmPrinter], or [PlainText] for the `Display` of
/// a signature or parameter without attributes, which has no context to make a printer with.
pub(crate) trait SignatureWriter {
    /// Writes `text` as it is.
    fn write_text(&mut self, text: &'static str);
    /// Writes `ty`, a recursion-group member as its alias.
    fn write_type(&mut self, ty: Type);
    /// Writes `string` quoted, escaped as by `str::escape_default`.
    fn write_string(&mut self, string: &str);
    /// Writes `attrs`, which is not empty, as a dictionary attribute.
    fn write_attributes(&mut self, attrs: &AttributeDict);
}

impl SignatureWriter for AsmPrinter<'_> {
    fn write_text(&mut self, text: &'static str) {
        *self += formatter::const_text(text);
    }

    fn write_type(&mut self, ty: Type) {
        self.print_type(ty);
    }

    fn write_string(&mut self, string: &str) {
        self.print_string(string);
    }

    fn write_attributes(&mut self, attrs: &AttributeDict) {
        self.print_attribute_dictionary(attrs.iter().map(|attr| *attr));
    }
}

/// The text an [AsmPrinter] made with [AsmPrinter::fragment] writes for a signature without
/// attributes, written without one.
#[derive(Default)]
struct PlainText {
    text: String,
    aliases: RefCell<TypeAliases>,
}

impl SignatureWriter for PlainText {
    fn write_text(&mut self, text: &'static str) {
        self.text.push_str(text);
    }

    fn write_type(&mut self, ty: Type) {
        let aliased = Aliased {
            ty,
            aliases: &self.aliases,
            definition: false,
        };
        // Writing to a `String` does not fail
        let _ = write!(self.text, "{aliased}");
    }

    fn write_string(&mut self, string: &str) {
        let _ = write!(self.text, "\"{}\"", string.escape_default());
    }

    fn write_attributes(&mut self, _attrs: &AttributeDict) {
        unreachable!("a signature with attributes is displayed with a printer")
    }
}

/// Writes to `f` what `write` writes: with a fragment printer made with the context of the first
/// attribute of `params`, or as [PlainText] if they have none.
fn display_with<'a>(
    f: &mut fmt::Formatter<'_>,
    params: impl IntoIterator<Item = &'a AbiParam>,
    write: impl FnOnce(&mut dyn SignatureWriter),
) -> fmt::Result {
    let context = params
        .into_iter()
        .find_map(|param| param.attrs.iter().next().map(|attr| attr.value.borrow().context_rc()));
    match context {
        Some(context) => {
            let flags = OpPrintingFlags::default();
            let mut printer = AsmPrinter::fragment(context, &flags);
            write(&mut printer);
            write!(f, "{}", printer.finish())
        }
        None => {
            let mut text = PlainText::default();
            write(&mut text);
            f.write_str(&text.text)
        }
    }
}

/// Writes `extern("cc") (param, ..)`, then ` -> result-list` if the signature declares results.
fn print_signature<W: SignatureWriter + ?Sized>(writer: &mut W, signature: &Signature) {
    writer.write_text("extern(");
    writer.write_string(signature.cc.as_str());
    writer.write_text(") ");
    print_param_list(writer, signature.params());
    print_optional_results(writer, signature.declared_results());
}

/// Writes a `param`: its type, then its attributes as a dictionary if it has any.
fn write_param<W: SignatureWriter + ?Sized>(writer: &mut W, param: &AbiParam) {
    writer.write_type(param.ty);
    if !param.attrs.is_empty() {
        writer.write_text(" ");
        writer.write_attributes(&param.attrs);
    }
}

/// Writes a parenthesised list of `param`: `(ptr<i32>, i8 { extend = #builtin.zext })`.
pub(crate) fn print_param_list<W: SignatureWriter + ?Sized>(writer: &mut W, params: &[AbiParam]) {
    writer.write_text("(");
    for (i, param) in params.iter().enumerate() {
        if i > 0 {
            writer.write_text(", ");
        }
        write_param(writer, param);
    }
    writer.write_text(")");
}

/// Writes a `result-list`: the type alone when it is one result without attributes whose type does
/// not start with `(` (is not a function type), a parenthesised list of `param` otherwise. So an
/// unparenthesised result never has a dictionary, and a `{` after it is not one.
pub(crate) fn print_result_list<W: SignatureWriter + ?Sized>(writer: &mut W, results: &[AbiParam]) {
    match results {
        [result] if result.attrs.is_empty() && !result.ty.is_function() => {
            writer.write_type(result.ty)
        }
        results => print_param_list(writer, results),
    }
}

/// Writes ` -> result-list` if there are `results`, and nothing otherwise.
pub(crate) fn print_optional_results<W: SignatureWriter + ?Sized>(
    writer: &mut W,
    results: &[AbiParam],
) {
    if !results.is_empty() {
        writer.write_text(" -> ");
        print_result_list(writer, results);
    }
}

/// The span of the next token (with the `%` of a value name), or the current location at the end
/// of the input.
pub(crate) fn next_token_span<'input, P: ?Sized + Parser<'input>>(
    parser: &mut P,
) -> ParseResult<SourceSpan> {
    match parser.token_stream_mut().peek()? {
        Some((start, token, end)) => {
            // The token of a value name is the name after its `%`
            let start = if matches!(token, Token::PercentIdent(_)) {
                start - 1
            } else {
                start
            };
            Ok(SourceSpan::session(
                parser.source_id(),
                TextRange::new(start, end).expect("the lexer produces ordered ranges"),
            ))
        }
        None => Ok(parser.current_location()),
    }
}

/// An "invalid parameter list" error labelled `reason` at `span`.
pub(crate) fn invalid_parameter_list(span: SourceSpan, reason: &str) -> ParserError {
    ParserError::from(report! {
        message: "invalid parameter list",
        labels: [primary(span, String::from(reason))],
    })
}

/// Reads a `param`: a type, then an optional dictionary of its attributes.
///
/// A value name is an error: a list that does not name its first parameter names none.
fn parse_param<'input, P: ?Sized + Parser<'input>>(parser: &mut P) -> ParseResult<AbiParam> {
    if parser.token_stream_mut().is_next(|tok| matches!(tok, Token::PercentIdent(_))) {
        let span = next_token_span(parser)?;
        return Err(invalid_parameter_list(
            span,
            "expected a type: the parameters of this list are not named",
        ));
    }
    let ty = parser.parse_type()?.into_inner();
    let mut attrs = ParsedAttrs::default();
    parser.parse_optional_attribute_dict(&mut attrs)?;
    Ok(AbiParam::new_with_attrs(ty, attrs))
}

/// Reads the rest of a parenthesised list of `param` whose `(` has been read, up to and including
/// its `)`, into `params`.
pub(crate) fn parse_params_after_lparen<'input, P: ?Sized + Parser<'input>>(
    parser: &mut P,
    params: &mut Vec<AbiParam>,
) -> ParseResult {
    if parser.parse_optional_rparen()? {
        return Ok(());
    }
    loop {
        params.push(parse_param(parser)?);
        if !parser.parse_optional_comma()? {
            break;
        }
    }
    parser.parse_rparen()
}

/// Reads a parenthesised list of `param`.
pub(crate) fn parse_param_list<'input, P: ?Sized + Parser<'input>>(
    parser: &mut P,
) -> ParseResult<Vec<AbiParam>> {
    parser.parse_lparen()?;
    let mut params = Vec::new();
    parse_params_after_lparen(parser, &mut params)?;
    Ok(params)
}

/// Reads a `result-list`: a parenthesised list of `param`, or a single type, which has no
/// dictionary (a `{` after it is left for the caller).
pub(crate) fn parse_result_list<'input, P: ?Sized + Parser<'input>>(
    parser: &mut P,
) -> ParseResult<Vec<AbiParam>> {
    if parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Lparen)) {
        parse_param_list(parser)
    } else {
        let ty = parser.parse_type()?.into_inner();
        Ok(alloc::vec![AbiParam::new(ty)])
    }
}

/// Reads `'->' result-list` if the next token is `->`; no results otherwise.
pub(crate) fn parse_optional_results<'input, P: ?Sized + Parser<'input>>(
    parser: &mut P,
) -> ParseResult<Vec<AbiParam>> {
    if parser.parse_optional_arrow()? {
        parse_result_list(parser)
    } else {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use super::*;

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "the type of a parameter is `ptr<rec.0>`, an open type")]
    fn an_open_parameter_type_panics() {
        let context = crate::Context::default();
        let _ = AbiParam::new(context.ptr_type(context.rec_var(0)));
    }

    /// A function that never returns declares the single result `never`, and produces no values.
    #[test]
    fn a_never_returning_signature_has_three_views() {
        let context = Rc::new(crate::Context::default());
        let mut signature = Signature::new(&context, [context.i32()], [context.never_type()]);
        assert!(signature.never_returns());
        assert!(signature.results().is_empty());
        assert_eq!(signature.declared_results().len(), 1);
        assert!(signature.declared_results()[0].ty.is_never());
        assert!(signature.result(0).is_none());
        assert!(signature.result_mut(0).is_none());
    }

    /// An ordinary signature declares the results it produces.
    #[test]
    fn an_ordinary_signature_declares_what_it_produces() {
        let context = Rc::new(crate::Context::default());
        let mut signature =
            Signature::new(&context, [context.i32()], [context.i64(), context.i8()]);
        assert!(!signature.never_returns());
        assert_eq!(signature.results(), signature.declared_results());
        assert_eq!(signature.results().len(), 2);
        assert_eq!(signature.result(1).map(|r| r.ty), Some(context.i8()));
        assert_eq!(signature.result_mut(0).map(|r| r.ty), Some(context.i64()));
        assert!(signature.result(2).is_none());

        // `never` among other results is not a never-returning function
        let mixed = Signature::new(&context, [], [context.never_type(), context.i32()]);
        assert!(!mixed.never_returns());
        assert_eq!(mixed.results().len(), 2);

        let none = Signature::new(&context, [], []);
        assert!(!none.never_returns());
        assert!(none.results().is_empty() && none.declared_results().is_empty());
    }

    /// A signature's `Display` output must render its results, not a second copy of its params.
    #[test]
    fn display_renders_params_and_results_separately() {
        let context = crate::Context::default();
        let signature = Signature {
            params: alloc::vec![AbiParam::new(context.i32())],
            results: alloc::vec![AbiParam::new(context.i64())],
            cc: CallConv::default(),
        };

        let rendered = format!("{signature}");
        assert_eq!(rendered, "extern(\"C\") (i32) -> i64");
        let (params, results) = rendered
            .split_once("->")
            .unwrap_or_else(|| panic!("expected a '->' in {rendered}"));
        assert!(params.contains("i32") && !params.contains("i64"), "{rendered}");
        assert!(results.contains("i64") && !results.contains("i32"), "{rendered}");
    }

    /// `Display` writes the IR text of a parameter and of a signature, attributes and all.
    #[test]
    fn display_writes_the_ir_text() {
        let context = Rc::new(crate::Context::default());
        let zext = AbiParam::zext(context.i8(), &context);
        assert_eq!(format!("{zext}"), "i8 { extend = #builtin.zext }");
        assert_eq!(format!("{}", AbiParam::new(context.i32())), "i32");

        let signature = Signature {
            params: alloc::vec![zext, AbiParam::sret(context.ptr_type(context.i32()), &context)],
            results: alloc::vec![AbiParam::sext(context.i16(), &context)],
            cc: CallConv::new("fast"),
        };
        assert_eq!(
            format!("{signature}"),
            "extern(\"fast\") (i8 { extend = #builtin.zext }, ptr<i32> { sret = #builtin.sret }) \
             -> (i16 { extend = #builtin.sext })"
        );

        let never = Signature::new(&context, [], [context.never_type()]);
        assert_eq!(format!("{never}"), "extern(\"C\") () -> never");
        let none = Signature::new(&context, [context.i32()], []);
        assert_eq!(format!("{none}"), "extern(\"C\") (i32)");
        let function = context.function_type([context.i32()], [context.i32()]);
        let returns_fn = Signature::new(&context, [function], [function]);
        assert_eq!(format!("{returns_fn}"), "extern(\"C\") ((i32) -> i32) -> ((i32) -> i32)");
    }

    /// A recursion-group member in a signature is written as its alias, with or without
    /// attributes, and without declarations.
    #[test]
    fn display_writes_a_recursion_group_member_as_its_alias() {
        let context = Rc::new(crate::Context::default());
        let fields = [context.i32().into(), context.ptr_type(context.rec_var(0)).into()];
        let body = context
            .struct_type(crate::StructType::named(crate::interner::Symbol::intern("list"), fields));
        let list = context.rec_group(&[body]).unwrap()[0];
        let ptr = context.ptr_type(list);

        let plain = Signature::new(&context, [ptr], [ptr]);
        assert_eq!(format!("{plain}"), "extern(\"C\") (ptr<!list>) -> ptr<!list>");
        let marked = Signature {
            params: alloc::vec![AbiParam::sret(ptr, &context)],
            results: alloc::vec![],
            cc: CallConv::default(),
        };
        assert_eq!(format!("{marked}"), "extern(\"C\") (ptr<!list> { sret = #builtin.sret })");
    }
}
