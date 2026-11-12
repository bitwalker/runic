use alloc::{format, rc::Rc, vec::Vec};
use core::fmt;

use crate::{
    AttrPrinter, CallConv, Context, NamedAttribute, OpPrintingFlags, Type,
    attributes::{AttrParser, AttributeDict},
    derive::DialectAttribute,
    dialects::builtin::BuiltinDialect,
    formatter,
    print::AsmPrinter,
};

/// A marker attribute for "struct return" parameters of a function.
///
/// An sret parameter is a parameter introduced when the ABI of a function requires the caller to
/// allocate memory to hold the function results, and then pass a pointer to that allocation to
/// the callee at a given parameter position (typically the first or last parameter). It is up to
/// the caller to ensure the given pointer is of sufficient size and alignment to hold the results.
#[derive(DialectAttribute, Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
#[attribute(dialect = BuiltinDialect, implements(AttrPrinter))]
#[allow(dead_code)]
pub struct Sret;

impl From<()> for Sret {
    fn from(_value: ()) -> Self {
        Self
    }
}

impl fmt::Display for Sret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("sret")
    }
}

impl AttrPrinter for SretAttr {
    fn print(&self, _printer: &mut crate::print::AsmPrinter<'_>) {}
}

/// A marker attribute for argument parameters or results which should be zero-extended to the
/// target architecture's native machine word size, _if and only if_ the parameter type is smaller
/// than the native machine word size.
#[derive(DialectAttribute, Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
#[attribute(dialect = BuiltinDialect, implements(AttrPrinter))]
#[allow(dead_code)]
pub struct Zext;

impl From<()> for Zext {
    fn from(_value: ()) -> Self {
        Self
    }
}

impl fmt::Display for Zext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("zext")
    }
}

impl AttrPrinter for ZextAttr {
    fn print(&self, _printer: &mut crate::print::AsmPrinter<'_>) {}
}

/// A marker attribute for argument parameters or results which should be sign-extended to the
/// target architecture's native machine word size, _if and only if_ the parameter type is smaller
/// than the native machine word size.
#[derive(DialectAttribute, Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
#[attribute(dialect = BuiltinDialect, implements(AttrPrinter))]
#[allow(dead_code)]
pub struct Sext;

impl From<()> for Sext {
    fn from(_value: ()) -> Self {
        Self
    }
}

impl fmt::Display for Sext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("sext")
    }
}

impl AttrPrinter for SextAttr {
    fn print(&self, _printer: &mut crate::print::AsmPrinter<'_>) {}
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
        Self::new_with_attrs(ty, [NamedAttribute::new(crate::interner::symbols::Extension, zext)])
    }

    pub fn sext(ty: Type, context: &Rc<Context>) -> Self {
        let sext = context.create_attribute::<SextAttr, _>(());
        Self::new_with_attrs(ty, [NamedAttribute::new(crate::interner::symbols::Extension, sext)])
    }

    pub fn sret(ty: Type, context: &Rc<Context>) -> Self {
        assert!(ty.is_pointer(), "sret parameters must be pointers");
        let sret = context.create_attribute::<SretAttr, _>(());
        Self::new_with_attrs(ty, [NamedAttribute::new("sret", sret)])
    }

    pub fn mark_sret(&mut self, context: &Rc<Context>) {
        let sret = context.create_attribute::<SretAttr, _>(());
        let attr = context.alloc_map_item(NamedAttribute {
            name: crate::interner::symbols::Sret,
            value: sret,
        });
        self.attrs.insert(attr);
    }

    pub fn is_sret_param(&self) -> bool {
        self.attrs.contains(&crate::interner::symbols::Sret)
    }

    pub fn extension(&self) -> ArgumentExtension {
        match self.attrs.find(&crate::interner::symbols::Extension).get() {
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

impl formatter::PrettyPrint for AbiParam {
    fn render(&self) -> formatter::Document {
        use formatter::*;

        // Only reached outside an IR print, through the `Display` of `AbiParam` and `Signature`
        // (the `AttrPrinter` of a signature writes its types with the printer), so the type is
        // written with its `Display` form and the attributes below are printed as fragments,
        // without the declarations of the recursion groups they refer to.
        let ty = text(format!("{}", self.ty));

        let mut doc = Document::Empty;
        let flags = OpPrintingFlags::default();
        for (i, attr) in self.attrs.iter().enumerate() {
            let (key, value) = match attr.name.as_str() {
                "sret" => (const_text("sret"), Document::Empty),
                "extend" => {
                    let value = attr.value.borrow();
                    if value.is::<ZextAttr>() {
                        (const_text("zext"), Document::Empty)
                    } else if value.is::<SextAttr>() {
                        (const_text("sext"), Document::Empty)
                    } else {
                        let mut printer = AsmPrinter::fragment(value.context_rc(), &flags);
                        printer.print_attribute_value(&*value);
                        let value_pp = printer.finish();
                        (const_text("extend"), value_pp)
                    }
                }
                other => {
                    let value = attr.value.borrow();
                    let mut printer = AsmPrinter::fragment(value.context_rc(), &flags);
                    printer.print_attribute_value(&*value);
                    (const_text(other), printer.finish())
                }
            };

            if i == 0 {
                doc += const_text(" { ");
            } else {
                doc += const_text(", ");
            }
            if value.is_empty() {
                doc += key;
            } else {
                doc += key + const_text(" = ") + value;
            }
        }

        if doc.is_empty() {
            ty
        } else {
            ty + doc + const_text(" }")
        }
    }
}

impl fmt::Display for AbiParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::formatter::PrettyPrint;
        write!(f, "{}", self.render())
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .key(&"params")
            .value_with(|f| {
                let mut builder = f.debug_list();
                for param in self.params.iter() {
                    builder.entry(&format_args!("{param}"));
                }
                builder.finish()
            })
            .key(&"results")
            .value_with(|f| {
                let mut builder = f.debug_list();
                for result in self.results.iter() {
                    builder.entry(&format_args!("{result}"));
                }
                builder.finish()
            })
            .entry(&"cc", &format_args!("{}", self.cc))
            .finish()
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
        printer.print_keyword("extern");
        printer.print_lparen();
        printer.print_string(self.cc.as_str());
        printer.print_rparen();
        printer.print_space();
        printer.print_function_type_parts(
            self.params().iter().map(|p| p.ty),
            self.results().iter().map(|p| p.ty),
        );
    }
}

impl AttrParser for SignatureAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        use crate::parse::ParserError;

        parser.parse_custom_keyword("extern")?;
        parser.parse_lparen()?;
        let cc_string = parser.parse_string()?;
        parser.parse_rparen()?;
        let (ty_span, ty) = parser.parse_type()?.into_parts();
        let Some(fn_ty) = ty.as_function() else {
            return Err(ParserError::InvalidAttributeValue {
                span: ty_span,
                reason: format!("expected a function type, got '{ty}'"),
            });
        };

        let cc = CallConv::new(cc_string.as_str());
        let context = parser.context_rc();
        let signature = Signature::with_convention(
            &context,
            cc,
            fn_ty.params().iter().copied(),
            fn_ty.results().iter().copied(),
        );
        let attr = context.create_attribute::<SignatureAttr, _>(signature);
        Ok(attr)
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
        let (params, results) = rendered
            .split_once("\"results\"")
            .unwrap_or_else(|| panic!("expected a 'results' key in {rendered}"));

        assert!(params.contains("i32"), "expected 'i32' under params in {rendered}");
        assert!(!params.contains("i64"), "unexpected 'i64' under params in {rendered}");
        assert!(results.contains("i64"), "expected 'i64' under results in {rendered}");
        assert!(!results.contains("i32"), "unexpected 'i32' under results in {rendered}");
    }
}
