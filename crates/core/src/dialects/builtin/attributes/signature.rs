use alloc::{rc::Rc, string::String, vec::Vec};
use core::{
    cell::RefCell,
    fmt::{self, Write},
};

use crate::{
    AttrPrinter, AttributeRef, CallConv, Context, NamedAttribute, OpPrintingFlags, SmallVec, Type,
    attributes::{AttrParser, Marker},
    derive::{AttributeComponent, DialectAttribute},
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
///
/// Its attributes are shared handles, sorted by name with at most one entry per name. A copy of
/// a parameter (or of a [Signature]) shares them with the original, so an attribute held by a
/// parameter is replaced with [Self::set_attribute], never mutated in place: a change made through
/// a shared handle would change every copy.
///
/// Two parameters are equal, and hash alike, when their types and the names and values of their
/// attributes are; the handles need not be the same.
#[derive(Debug, Clone, PartialEq, Eq, Hash, AttributeComponent)]
pub struct AbiParam {
    /// The type associated with this value
    pub ty: Type,
    /// The attributes associated with this value, sorted by name, one per name
    attrs: SmallVec<[NamedAttribute; 1]>,
}

impl AbiParam {
    /// Panics, in debug builds, if `ty` is open: a parameter's type enters IR.
    #[track_caller]
    pub fn new(ty: Type) -> Self {
        crate::types::debug_assert_closed(ty, "the type of a parameter");
        Self {
            ty,
            attrs: SmallVec::new_const(),
        }
    }

    /// A parameter of type `ty` holding `attributes`. A name given more than once keeps its last
    /// value, as [Self::set_attribute] does.
    ///
    /// Panics, in debug builds, if `ty` is open or belongs to another context than an attribute.
    #[track_caller]
    pub fn new_with_attrs(ty: Type, attributes: impl IntoIterator<Item = NamedAttribute>) -> Self {
        let mut param = Self::new(ty);
        for attr in attributes {
            param.set_attribute(attr.name, attr.value);
        }
        param
    }

    /// The attributes of the parameter, sorted by name.
    ///
    /// The handles are shared with every copy of the parameter: replace an attribute with
    /// [Self::set_attribute], do not mutate it.
    #[inline]
    pub fn attributes(&self) -> &[NamedAttribute] {
        self.attrs.as_slice()
    }

    /// The attribute `name`, if the parameter has it.
    ///
    /// The handle is shared with every copy of the parameter: replace the attribute with
    /// [Self::set_attribute], do not mutate it.
    pub fn attribute(&self, name: impl Into<interner::Symbol>) -> Option<AttributeRef> {
        let name = name.into();
        self.find(name).ok().map(|index| self.attrs[index].value)
    }

    /// Sets the attribute `name` to `value`, replacing any value it already has.
    ///
    /// Panics, in debug builds, if the parameter's type belongs to another context than `value`.
    #[track_caller]
    pub fn set_attribute(&mut self, name: impl Into<interner::Symbol>, value: AttributeRef) {
        if cfg!(debug_assertions) {
            value
                .borrow()
                .context()
                .debug_assert_ir_type(self.ty, "the type of a parameter");
        }
        let name = name.into();
        match self.find(name) {
            Ok(index) => self.attrs[index].value = value,
            Err(index) => self.attrs.insert(index, NamedAttribute::new(name, value)),
        }
    }

    /// Removes the attribute `name`, returning its value if the parameter had it.
    pub fn remove_attribute(&mut self, name: impl Into<interner::Symbol>) -> Option<AttributeRef> {
        let name = name.into();
        self.find(name).ok().map(|index| self.attrs.remove(index).value)
    }

    /// The index of the attribute `name`, or where it would be inserted to keep the order.
    fn find(&self, name: interner::Symbol) -> Result<usize, usize> {
        self.attrs.binary_search_by(|attr| attr.name.cmp(&name))
    }

    pub fn zext(ty: Type, context: &Rc<Context>) -> Self {
        let mut param = Self::new(ty);
        context.debug_assert_ir_type(ty, "the type of a parameter");
        param.set_attribute(interner::symbols::Extend, context.create_attribute::<ZextAttr, _>(()));
        param
    }

    pub fn sext(ty: Type, context: &Rc<Context>) -> Self {
        let mut param = Self::new(ty);
        context.debug_assert_ir_type(ty, "the type of a parameter");
        param.set_attribute(interner::symbols::Extend, context.create_attribute::<SextAttr, _>(()));
        param
    }

    pub fn sret(ty: Type, context: &Rc<Context>) -> Self {
        assert!(ty.is_pointer(), "sret parameters must be pointers");
        let mut param = Self::new(ty);
        context.debug_assert_ir_type(ty, "the type of a parameter");
        param.mark_sret(context);
        param
    }

    /// Marks the parameter `sret`; a parameter already so marked is left as it is, in effect.
    pub fn mark_sret(&mut self, context: &Rc<Context>) {
        self.set_attribute(interner::symbols::Sret, context.create_attribute::<SretAttr, _>(()));
    }

    pub fn is_sret_param(&self) -> bool {
        self.find(interner::symbols::Sret).is_ok()
    }

    /// How the parameter is extended: the value of its `extend` attribute, `#builtin.zext` or
    /// `#builtin.sext`.
    pub fn extension(&self) -> ArgumentExtension {
        match self.attribute(interner::symbols::Extend) {
            None => ArgumentExtension::None,
            Some(attr) => {
                let value = attr.borrow();
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

/// Why a [Signature] is invalid: where `never` appears other than as the single result of a
/// function that never returns.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidSignature {
    /// The parameter at `index` has type `never`.
    NeverParameter { index: usize },
    /// The result at `index` has type `never`, and is not the only result.
    NeverAmongResults { index: usize },
}

impl fmt::Display for InvalidSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NeverParameter { index } => write!(
                f,
                "parameter {index} has type `never`, which is only valid as the single result of \
                 a function that never returns"
            ),
            Self::NeverAmongResults { index } => write!(
                f,
                "result {index} has type `never`, which is only valid as the single result of a \
                 function that never returns"
            ),
        }
    }
}

impl core::error::Error for InvalidSignature {}

/// A [Signature] represents the type, ABI, and linkage of a function.
///
/// A function signature provides us with all of the necessary detail to correctly
/// validate and emit code for a function, whether from the perspective of a caller,
/// or the callee.
///
/// As an attribute ([SignatureAttr]), its elements ([crate::Attribute::visit_elements]) are, for
/// each parameter, its type then its attributes, then the same for each result.
///
/// A signature attribute is valid when [Signature::validate] accepts it and the type of each
/// parameter and declared result belongs to the context and is closed; [crate::Operation::verify]
/// checks it wherever an operation holds one.
#[derive(DialectAttribute, Default, Debug, Clone, PartialEq, Eq, Hash)]
#[attribute(
    dialect = BuiltinDialect,
    verify = "verify_signature",
    implements(AttrPrinter)
)]
pub struct Signature {
    /// The arguments expected by this function
    params: Vec<AbiParam>,
    /// The results returned by this function, as declared
    results: Vec<AbiParam>,
    /// The calling convention that applies to this function
    cc: CallConv,
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

    /// A signature with the calling convention `cc`, the parameters `params` and the declared
    /// results `results` (`[never]` for a function that never returns), as they are.
    pub fn from_parts(cc: CallConv, params: Vec<AbiParam>, results: Vec<AbiParam>) -> Self {
        Self {
            params,
            results,
            cc,
        }
    }

    /// Get the calling convention of this function
    pub fn calling_convention(&self) -> CallConv {
        self.cc
    }

    /// Sets the calling convention of this function to `cc`.
    pub fn set_calling_convention(&mut self, cc: CallConv) {
        self.cc = cc;
    }

    /// Returns the number of arguments expected by this function
    pub fn arity(&self) -> usize {
        self.params().len()
    }

    /// Returns a slice containing the parameters for this function
    pub fn params(&self) -> &[AbiParam] {
        self.params.as_slice()
    }

    /// The parameters of this function, to change in place.
    #[inline]
    pub fn params_mut(&mut self) -> &mut Vec<AbiParam> {
        &mut self.params
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

    /// The results as declared, to change in place: `[never]` for a function that never returns.
    #[inline]
    pub fn declared_results_mut(&mut self) -> &mut Vec<AbiParam> {
        &mut self.results
    }

    /// True if the single declared result is `never`.
    pub fn never_returns(&self) -> bool {
        matches!(self.results.as_slice(), [AbiParam { ty, .. }] if ty.is_never())
    }

    /// `Err` if `never` appears as a parameter, or as a result that is not the only one: a function
    /// that never returns declares the single result `never`, and `never` means nothing elsewhere.
    pub fn validate(&self) -> Result<(), InvalidSignature> {
        if let Some(index) = self.params.iter().position(|param| param.ty.is_never()) {
            return Err(InvalidSignature::NeverParameter { index });
        }
        if !self.never_returns()
            && let Some(index) = self.results.iter().position(|result| result.ty.is_never())
        {
            return Err(InvalidSignature::NeverAmongResults { index });
        }
        Ok(())
    }
}

/// A signature is valid ([Signature::validate]), and the type of each of its parameters and
/// declared results may be used in IR of `context`.
///
/// The verifier checks every type an operation holds; the parameter types are checked here too so
/// that the error says which parameter or result is the problem.
fn verify_signature(
    signature: &Signature,
    _ty: Type,
    context: &Context,
) -> Result<(), crate::Report> {
    let invalid = |problem: &dyn fmt::Display| {
        crate::Report::msg(alloc::format!("an invalid signature: {problem}"))
    };
    signature.validate().map_err(|err| invalid(&err))?;
    let params = signature.params().iter().enumerate().map(|(i, p)| ("parameter", i, p));
    let results = signature.declared_results().iter().enumerate().map(|(i, r)| ("result", i, r));
    for (kind, index, param) in params.chain(results) {
        let ty = param.ty;
        if let Some(problem) = context.ir_type_problem(ty) {
            return Err(invalid(&format_args!("{kind} {index} has type `{ty}`, {problem}")));
        }
    }
    Ok(())
}

impl AttrPrinter for SignatureAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        print_signature(printer, self);
    }
}

impl AttrParser for SignatureAttr {
    /// Reads `'extern' '(' string ')' '(' (param (',' param)*)? ')' ('->' result-list)?`.
    fn parse(parser: &mut dyn Parser<'_>) -> ParseResult<AttributeRef> {
        parser.parse_custom_keyword("extern")?;
        parser.parse_lparen()?;
        let cc = CallConv::new(parser.parse_string()?.as_str());
        parser.parse_rparen()?;
        let params = parse_param_list(parser)?;
        let results = parse_optional_results(parser)?;
        let signature = Signature::from_parts(cc, params, results);
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
    fn write_attributes(&mut self, attrs: &[NamedAttribute]);
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

    fn write_attributes(&mut self, attrs: &[NamedAttribute]) {
        self.print_attribute_dictionary(attrs.iter().copied());
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

    fn write_attributes(&mut self, _attrs: &[NamedAttribute]) {
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
        .find_map(|param| param.attributes().first().map(|attr| attr.value.borrow().context_rc()));
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
    writer.write_string(signature.calling_convention().as_str());
    writer.write_text(") ");
    print_param_list(writer, signature.params());
    print_optional_results(writer, signature.declared_results());
}

/// Writes a `param`: its type, then its attributes as a dictionary if it has any.
fn write_param<W: SignatureWriter + ?Sized>(writer: &mut W, param: &AbiParam) {
    writer.write_type(param.ty);
    if !param.attributes().is_empty() {
        writer.write_text(" ");
        writer.write_attributes(param.attributes());
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
        [result] if result.attributes().is_empty() && !result.ty.is_function() => {
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

    /// In a debug build, `set_attribute` checks the attribute's context as `new_with_attrs` does.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a type of another context")]
    fn set_attribute_of_another_context_panics_in_a_debug_build() {
        let context = Rc::new(crate::Context::default());
        let other = Rc::new(crate::Context::default());
        let foreign = other.create_attribute::<ZextAttr, _>(()).as_attribute_ref();
        let mut param = AbiParam::new(context.i8());
        param.set_attribute(interner::symbols::Extend, foreign);
    }

    /// The hash of `value` under one fixed hasher.
    fn hash_of<T: core::hash::Hash>(value: &T) -> u64 {
        use core::hash::{BuildHasher, BuildHasherDefault};
        BuildHasherDefault::<rustc_hash::FxHasher>::default().hash_one(value)
    }

    /// A copy of a parameter holds the same attribute handles as the original: cloning allocates
    /// no attribute.
    #[test]
    fn a_cloned_parameter_shares_its_attributes() {
        let context = Rc::new(crate::Context::default());
        let param = AbiParam::zext(context.i8(), &context);
        let copy = param.clone();
        let original = param.attribute(interner::symbols::Extend).unwrap();
        let shared = copy.attribute(interner::symbols::Extend).unwrap();
        assert!(AttributeRef::ptr_eq(&original, &shared));
        assert_eq!(copy, param);
    }

    /// The order attributes are set in does not matter: the parameters are equal, hash alike,
    /// print alike, and list their attributes sorted by name.
    #[test]
    fn a_parameter_does_not_depend_on_the_order_its_attributes_are_set_in() {
        let context = Rc::new(crate::Context::default());
        let ty = context.ptr_type(context.i8());
        let sret = context.create_attribute::<SretAttr, _>(()).as_attribute_ref();
        let zext = context.create_attribute::<ZextAttr, _>(()).as_attribute_ref();

        let mut first = AbiParam::new(ty);
        first.set_attribute(interner::symbols::Sret, sret);
        first.set_attribute(interner::symbols::Extend, zext);
        let mut second = AbiParam::new(ty);
        second.set_attribute(interner::symbols::Extend, zext);
        second.set_attribute(interner::symbols::Sret, sret);

        assert_eq!(first, second);
        assert_eq!(hash_of(&first), hash_of(&second));
        assert_eq!(format!("{first}"), format!("{second}"));
        assert_eq!(format!("{first}"), "ptr<i8> { extend = #builtin.zext, sret = #builtin.sret }");
        let names = |param: &AbiParam| {
            param.attributes().iter().map(|attr| attr.name.as_str()).collect::<Vec<_>>()
        };
        assert_eq!(names(&first), ["extend", "sret"]);
        assert_eq!(names(&second), ["extend", "sret"]);

        // Equality compares attribute values, not handles
        let mut third = AbiParam::new(ty);
        third.set_attribute(
            interner::symbols::Extend,
            context.create_attribute::<ZextAttr, _>(()).as_attribute_ref(),
        );
        third.set_attribute(
            interner::symbols::Sret,
            context.create_attribute::<SretAttr, _>(()).as_attribute_ref(),
        );
        assert_eq!(first, third);
        assert_eq!(hash_of(&first), hash_of(&third));
    }

    /// `set_attribute` on a name the parameter has replaces its value, and `remove_attribute`
    /// takes it out.
    #[test]
    fn setting_an_attribute_replaces_and_removing_takes_it_out() {
        let context = Rc::new(crate::Context::default());
        let zext = context.create_attribute::<ZextAttr, _>(()).as_attribute_ref();
        let sext = context.create_attribute::<SextAttr, _>(()).as_attribute_ref();
        let key = interner::symbols::Extend;

        let mut param = AbiParam::new(context.i8());
        assert!(param.remove_attribute(key).is_none());
        param.set_attribute(key, zext);
        param.set_attribute(key, sext);
        assert_eq!(param.attributes().len(), 1, "{param}");
        assert!(AttributeRef::ptr_eq(&param.attribute(key).unwrap(), &sext));
        assert_eq!(param.extension(), ArgumentExtension::Sext);

        let removed = param.remove_attribute(key).expect("an `extend` attribute");
        assert!(AttributeRef::ptr_eq(&removed, &sext));
        assert!(param.attribute(key).is_none());
        assert!(param.attributes().is_empty());
        assert_eq!(param.extension(), ArgumentExtension::None);
        assert_eq!(param, AbiParam::new(context.i8()));
    }

    /// A name given more than once to `new_with_attrs` keeps its last value.
    #[test]
    fn a_parameter_built_with_a_repeated_name_keeps_the_last() {
        let context = Rc::new(crate::Context::default());
        let zext = context.create_attribute::<ZextAttr, _>(()).as_attribute_ref();
        let sext = context.create_attribute::<SextAttr, _>(()).as_attribute_ref();
        let sret = context.create_attribute::<SretAttr, _>(()).as_attribute_ref();
        let key = interner::symbols::Extend;
        let param = AbiParam::new_with_attrs(
            context.ptr_type(context.i8()),
            [
                NamedAttribute::new(key, zext),
                NamedAttribute::new(interner::symbols::Sret, sret),
                NamedAttribute::new(key, sext),
            ],
        );
        assert_eq!(param.attributes().len(), 2, "{param}");
        assert!(AttributeRef::ptr_eq(&param.attribute(key).unwrap(), &sext));
        assert!(param.is_sret_param());
    }

    /// Names given to `new_with_attrs` in reverse order come out of `attributes()` sorted.
    #[test]
    fn a_parameter_built_from_names_in_reverse_order_lists_them_sorted() {
        let context = Rc::new(crate::Context::default());
        let zext = context.create_attribute::<ZextAttr, _>(()).as_attribute_ref();
        let sret = context.create_attribute::<SretAttr, _>(()).as_attribute_ref();
        let param = AbiParam::new_with_attrs(
            context.ptr_type(context.i8()),
            [
                NamedAttribute::new(interner::symbols::Sret, sret),
                NamedAttribute::new(interner::symbols::Extend, zext),
            ],
        );
        let names = param.attributes().iter().map(|attr| attr.name.as_str()).collect::<Vec<_>>();
        assert_eq!(names, ["extend", "sret"]);
        assert!(AttributeRef::ptr_eq(
            &param.attribute(interner::symbols::Extend).unwrap(),
            &zext
        ));
        assert!(AttributeRef::ptr_eq(&param.attribute(interner::symbols::Sret).unwrap(), &sret));
    }

    /// `from_parts` holds what it is given; the `_mut` accessors and `set_calling_convention`
    /// change the signature in place.
    #[test]
    fn a_signature_is_built_from_parts_and_changed_through_its_accessors() {
        let context = Rc::new(crate::Context::default());
        let zext = AbiParam::zext(context.i8(), &context);
        let mut signature = Signature::from_parts(
            CallConv::new("fast"),
            alloc::vec![zext.clone()],
            alloc::vec![AbiParam::new(context.i32())],
        );
        assert_eq!(signature.calling_convention(), CallConv::new("fast"));
        assert_eq!(signature.params(), core::slice::from_ref(&zext));
        assert_eq!(signature.declared_results(), [AbiParam::new(context.i32())]);
        assert_eq!(
            signature,
            Signature::from_parts(
                CallConv::new("fast"),
                alloc::vec![AbiParam::zext(context.i8(), &context)],
                alloc::vec![AbiParam::new(context.i32())],
            )
        );

        signature.set_calling_convention(CallConv::default());
        assert_eq!(signature.calling_convention(), CallConv::default());

        signature.params_mut().push(AbiParam::new(context.i64()));
        signature.params_mut()[0].ty = context.i16();
        assert_eq!(signature.arity(), 2);
        assert_eq!(signature.params()[0].ty, context.i16());
        assert_eq!(signature.params()[0].extension(), ArgumentExtension::Zext);
        assert_eq!(signature.params()[1], AbiParam::new(context.i64()));

        signature.declared_results_mut()[0] = AbiParam::new(context.never_type());
        assert!(signature.never_returns());
        assert!(signature.results().is_empty());
        assert_eq!(signature.declared_results().len(), 1);
        signature.declared_results_mut().clear();
        assert!(!signature.never_returns());
        assert!(signature.results().is_empty() && signature.declared_results().is_empty());

        assert_eq!(format!("{signature}"), "extern(\"C\") (i16 { extend = #builtin.zext }, i64)");
    }

    /// A function that never returns declares the single result `never`, and produces no values.
    #[test]
    fn a_never_returning_signature_has_three_views() {
        let context = Rc::new(crate::Context::default());
        let signature = Signature::new(&context, [context.i32()], [context.never_type()]);
        assert!(signature.never_returns());
        assert!(signature.results().is_empty());
        assert_eq!(signature.declared_results().len(), 1);
        assert!(signature.declared_results()[0].ty.is_never());
    }

    /// An ordinary signature declares the results it produces.
    #[test]
    fn an_ordinary_signature_declares_what_it_produces() {
        let context = Rc::new(crate::Context::default());
        let signature = Signature::new(&context, [context.i32()], [context.i64(), context.i8()]);
        assert!(!signature.never_returns());
        assert_eq!(signature.results(), signature.declared_results());
        assert_eq!(signature.results().len(), 2);

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
        let signature = Signature::from_parts(
            CallConv::default(),
            alloc::vec![AbiParam::new(context.i32())],
            alloc::vec![AbiParam::new(context.i64())],
        );

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

        let signature = Signature::from_parts(
            CallConv::new("fast"),
            alloc::vec![zext, AbiParam::sret(context.ptr_type(context.i32()), &context)],
            alloc::vec![AbiParam::sext(context.i16(), &context)],
        );
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
        let marked = Signature::from_parts(
            CallConv::default(),
            alloc::vec![AbiParam::sret(ptr, &context)],
            alloc::vec![],
        );
        assert_eq!(format!("{marked}"), "extern(\"C\") (ptr<!list> { sret = #builtin.sret })");
    }
}
