use darling::FromDeriveInput;
use heck::ToSnakeCase;
use quote::{format_ident, quote, quote_spanned};
use syn::{Ident, parse_quote, parse_quote_spanned, spanned::Spanned};

/// Represents the parsed struct definition for the operation we wish to define
///
/// Only named structs are allowed at this time.
#[derive(Debug, FromDeriveInput)]
#[darling(
    attributes(attribute),
    forward_attrs(doc, cfg, allow, derive),
    supports(any)
)]
pub struct Attribute {
    ident: Ident,
    #[allow(unused)]
    vis: syn::Visibility,
    generics: syn::Generics,
    #[allow(unused)]
    attrs: Vec<syn::Attribute>,
    dialect: Ident,
    #[darling(default)]
    name: Option<Ident>,
    #[darling(default)]
    remote: Option<syn::Path>,
    /// A function `fn(&Context) -> Value` giving the attribute's default value.
    #[darling(default)]
    default: Option<syn::Path>,
    /// The attribute has no default value: it is always written with a body, and its
    /// `create_default` panics.
    #[darling(default)]
    no_default: bool,
    #[darling(default)]
    bounds: Option<Vec<syn::WherePredicate>>,
    #[darling(default)]
    traits: darling::util::PathList,
    #[darling(default)]
    implements: darling::util::PathList,
}

impl Attribute {
    #[allow(unused)]
    pub fn value_type(&self) -> syn::Type {
        syn::Type::Path(syn::TypePath {
            qself: None,
            path: self.value_type_path(),
        })
    }

    pub fn value_type_path(&self) -> syn::Path {
        if let Some(remote) = self.remote.clone() {
            remote
        } else {
            syn::Path {
                leading_colon: None,
                segments: syn::punctuated::Punctuated::from_iter([syn::PathSegment {
                    ident: self.ident.clone(),
                    arguments: syn::PathArguments::None,
                }]),
            }
        }
    }
}

impl quote::ToTokens for Attribute {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let value_type_name = self.value_type_path();
        let attr_name = self.name.clone().unwrap_or_else(|| {
            let name = self.ident.to_string().to_snake_case();
            format_ident!("{name}", span = self.ident.span())
        });
        let attr_name_base = &self.ident;
        let attr_struct_name = format_ident!("{attr_name_base}Attr", span = self.ident.span());

        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();

        let value_type_name_str = value_type_name.clone().into_token_stream().to_string();
        let struct_doc = syn::Lit::Str(syn::LitStr::new(
            &format!("An attribute which represents `{value_type_name_str}` values."),
            self.ident.span(),
        ));

        let mut debug_generics = self.generics.clone();
        for tp in debug_generics.type_params_mut() {
            tp.bounds.push(syn::TypeParamBound::Trait(syn::TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: parse_quote!(::core::fmt::Debug),
            }));
        }
        let (debug_impl_generics, debug_ty_generics, debug_where_clause) =
            debug_generics.split_for_impl();

        let mut display_generics = self.generics.clone();
        for tp in display_generics.type_params_mut() {
            tp.bounds.push(syn::TypeParamBound::Trait(syn::TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: parse_quote!(::core::fmt::Display),
            }));
        }

        let mut attr_impl_generics = self.generics.clone();
        if let Some(bounds) = self.bounds.as_ref() {
            let where_clause = attr_impl_generics.make_where_clause();
            where_clause.predicates.extend(bounds.iter().cloned());
        }
        let (attr_impl_generics, attr_ty_generics, attr_where_clause) =
            attr_impl_generics.split_for_impl();

        // struct $Attr
        tokens.extend(quote_spanned! { attr_name.span() =>
            #[doc = #struct_doc]
            ///
            /// It is not possible to construct values of this type directly, instead you must
            /// allocate them using a [`::runic_core::Context`] via `Self::create`.
            #[derive(Clone, PartialEq, Eq, Hash)]
            #[repr(C)]
            pub struct #attr_struct_name #ty_generics {
                _attr: ::runic_core::attributes::Attr,
                value: #value_type_name #ty_generics,
            }

            impl #impl_generics core::ops::Deref for #attr_struct_name #ty_generics #where_clause {
                type Target = #value_type_name #ty_generics;

                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    &self.value
                }
            }

            impl #impl_generics core::ops::DerefMut for #attr_struct_name #ty_generics #where_clause {
                #[inline(always)]
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.value
                }
            }

            impl #debug_impl_generics ::core::fmt::Debug for #attr_struct_name #debug_ty_generics #debug_where_clause {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    f.debug_struct(stringify!(#attr_struct_name))
                        .field("attr", &self._attr)
                        .field("value", &self.value)
                        .finish()
                }
            }
        });

        // impl AsRef<T>/AsMut<T>
        // impl AsRef<dyn Attr>/AsMut<dyn Attr>
        tokens.extend(quote_spanned! { attr_name.span() =>
            impl #impl_generics AsRef<#value_type_name #ty_generics> for #attr_struct_name #ty_generics #where_clause {
                #[inline(always)]
                fn as_ref(&self) -> &#value_type_name #ty_generics {
                    &self.value
                }
            }

            impl #impl_generics AsMut<#value_type_name #ty_generics> for #attr_struct_name #ty_generics #where_clause {
                #[inline(always)]
                fn as_mut(&mut self) -> &mut #value_type_name #ty_generics {
                    &mut self.value
                }
            }

            impl #impl_generics AsRef<::runic_core::attributes::Attr> for #attr_struct_name #ty_generics #where_clause {
                #[inline(always)]
                fn as_ref(&self) -> &::runic_core::attributes::Attr {
                    &self._attr
                }
            }

            impl #impl_generics AsMut<::runic_core::attributes::Attr> for #attr_struct_name #ty_generics #where_clause {
                #[inline(always)]
                fn as_mut(&mut self) -> &mut ::runic_core::attributes::Attr {
                    &mut self._attr
                }
            }
        });

        // impl Attribute
        // impl AttributeRegistration

        let create_default_impl: syn::ImplItemFn = if self.no_default {
            parse_quote_spanned! { attr_name.span() =>
                fn create_default(_context: &::alloc::rc::Rc<::runic_core::Context>) -> ::runic_core::UnsafeIntrusiveEntityRef<Self> {
                    panic!("the attribute `{}` has no default value", <Self as ::runic_core::AttributeRegistration>::full_name())
                }
            }
        } else if let Some(default_ctor) = self.default.as_ref() {
            parse_quote_spanned! { default_ctor.span() =>
                #[allow(clippy::let_unit_value)]
                fn create_default(context: &::alloc::rc::Rc<::runic_core::Context>) -> ::runic_core::UnsafeIntrusiveEntityRef<Self> {
                    let value: <Self as ::runic_core::AttributeRegistration>::Value = #default_ctor(context);
                    let ty = <Self as ::runic_core::attributes::MaybeInferAttributeType>::maybe_infer_type_from_value(context, &value);
                    Self::create(context, value, ty.unwrap_or_else(|| context.unknown_type()))
                }
            }
        } else {
            parse_quote_spanned! { attr_name.span() =>
                #[allow(clippy::let_unit_value)]
                fn create_default(context: &::alloc::rc::Rc<::runic_core::Context>) -> ::runic_core::UnsafeIntrusiveEntityRef<Self> {
                    let value = <<Self as ::runic_core::AttributeRegistration>::Value>::default();
                    let ty = <Self as ::runic_core::attributes::MaybeInferAttributeType>::maybe_infer_type_from_value(context, &value);
                    Self::create(context, value, ty.unwrap_or_else(|| context.unknown_type()))
                }
            }
        };

        let no_default = self.no_default;
        let dialect = &self.dialect;
        let attr_name_str =
            syn::Lit::Str(syn::LitStr::new(&attr_name.to_string(), attr_name.span()));
        let traits = &self.traits;
        let implements = &self.implements;
        tokens.extend(quote! {
            impl #attr_impl_generics ::runic_core::Attribute for #attr_struct_name #attr_ty_generics #attr_where_clause {
                #[inline(always)]
                fn type_name(&self) -> &'static str {
                    ::core::any::type_name::<Self>()
                }

                #[inline(always)]
                fn as_any(&self) -> &dyn ::core::any::Any {
                    self
                }

                #[inline(always)]
                fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any {
                    self
                }

                #[inline]
                fn context(&self) -> &::runic_core::Context {
                    self._attr.context()
                }

                #[inline]
                fn context_rc(&self) -> ::alloc::rc::Rc<::runic_core::Context> {
                    self._attr.context_rc()
                }

                #[inline]
                fn name(&self) -> &::runic_core::AttributeName {
                    self._attr.name()
                }

                #[inline(always)]
                fn as_attr(&self) -> &::runic_core::attributes::Attr {
                    &self._attr
                }

                #[inline(always)]
                fn as_attr_mut(&mut self) -> &mut ::runic_core::attributes::Attr {
                    &mut self._attr
                }
                #[inline(always)]
                fn value(&self) -> &dyn ::runic_core::attributes::AttributeValue {
                    &self.value
                }
                #[inline(always)]
                fn value_mut(&mut self) -> &mut dyn ::runic_core::attributes::AttributeValue {
                    &mut self.value
                }
                fn ty(&self) -> ::runic_core::Type {
                    self._attr.ty()
                }
                fn set_type(&mut self, ty: ::runic_core::Type) {
                    self._attr.set_type(ty);
                }
            }

            impl #attr_impl_generics #attr_struct_name #attr_ty_generics #attr_where_clause {
                #[inline]
                pub const fn as_value(&self) -> &#value_type_name #ty_generics {
                    &self.value
                }

                #[inline]
                pub fn as_value_mut(&mut self) -> &mut #value_type_name #ty_generics {
                    &mut self.value
                }
            }

            impl #attr_impl_generics ::runic_core::AttributeRegistration for #attr_struct_name #attr_ty_generics #attr_where_clause {
                type Dialect = #dialect;
                type Value = #value_type_name #ty_generics;
                const HAS_DEFAULT: bool = !#no_default;

                fn name() -> ::runic_core::interner::Symbol {
                    ::runic_core::interner::Symbol::intern(#attr_name_str)
                }

                fn traits() -> ::alloc::boxed::Box<[::runic_core::traits::TraitInfo]> {
                    ::alloc::boxed::Box::from([
                        ::runic_core::traits::TraitInfo::new::<Self, dyn core::any::Any>(),
                        ::runic_core::traits::TraitInfo::new::<Self, dyn ::runic_core::Attribute>(),
                        #(
                            ::runic_core::traits::TraitInfo::new::<Self, dyn #traits>(),
                        )*
                        #(
                            ::runic_core::traits::TraitInfo::new::<Self, dyn #implements>(),
                        )*
                    ])
                }

                #[allow(clippy::let_unit_value)]
                fn create<AttrValue>(
                    context: &::alloc::rc::Rc<::runic_core::Context>,
                    value: AttrValue,
                    ty: ::runic_core::Type
                ) -> ::runic_core::UnsafeIntrusiveEntityRef<Self>
                where
                    #value_type_name #ty_generics: From<AttrValue>,
                {
                    let name = context.get_registered_attribute_name::<#attr_struct_name #ty_generics>();
                    let value = <#value_type_name #ty_generics>::from(value);
                    <#value_type_name #ty_generics as ::runic_core::attributes::DebugAssertIrValue>::debug_assert_ir_value(&value, context);
                    // `Attr::as_attr_ref` relies on this layout to find the entity header
                    const {
                        assert!(
                            ::core::mem::offset_of!(Self, _attr) == 0
                                && ::core::mem::align_of::<Self>()
                                    == ::core::mem::align_of::<::runic_core::attributes::Attr>()
                        )
                    }
                    unsafe {
                        let offset = ::core::mem::offset_of!(#attr_struct_name #ty_generics, _attr);
                        let _attr = ::runic_core::attributes::Attr::uninit::<Self>(context, name, ty, offset);
                        context.alloc_tracked(Self {
                            _attr,
                            value,
                        })
                    }
                }

                #create_default_impl

                #[inline(always)]
                fn underlying_value(attr: &Self) -> &Self::Value {
                    &attr.value
                }

                #[inline(always)]
                fn underlying_value_mut(attr: &mut Self) -> &mut Self::Value {
                    &mut attr.value
                }
            }

            ::runic_core::inventory::submit!(::runic_core::DialectAttributeRegistrationInfo::new::<#attr_struct_name #attr_ty_generics>());
        });

        // impl $DerivedTrait
        for derived_trait in self.traits.iter() {
            tokens.extend(quote_spanned! { derived_trait.span() =>
                impl #attr_impl_generics #derived_trait for #attr_struct_name #attr_ty_generics #attr_where_clause {}
            });
        }
    }
}

#[cfg(test)]
mod naming_tests {
    use quote::{ToTokens, quote};

    /// Inspect the registration method consumers use, rather than testing the case
    /// conversion dependency independently of the generated attribute.
    fn registered_name(type_name: &str) -> String {
        let input = syn::parse_str(&format!(
            "#[attribute(dialect = TestDialect)] struct {type_name}(bool);"
        ))
        .unwrap();
        let output = crate::dialect::derive_attribute(&input).unwrap().into_token_stream();
        let output: syn::File = syn::parse2(output).unwrap();
        let registration = output
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Impl(item)
                    if item.trait_.as_ref().is_some_and(|(_, path, _)| {
                        path.segments.last().unwrap().ident == "AttributeRegistration"
                    }) =>
                {
                    Some(item)
                }
                _ => None,
            })
            .unwrap();
        let name = registration
            .items
            .iter()
            .find_map(|item| match item {
                syn::ImplItem::Fn(method) if method.sig.ident == "name" => Some(method),
                _ => None,
            })
            .unwrap();
        let [syn::Stmt::Expr(syn::Expr::Call(call), None)] = name.block.stmts.as_slice() else {
            panic!("registration name must return the interned literal");
        };
        assert_eq!(
            call.func.to_token_stream().to_string(),
            quote!(::runic_core::interner::Symbol::intern).to_string()
        );
        let [
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(name),
                ..
            }),
        ] = call.args.iter().collect::<Vec<_>>().as_slice()
        else {
            panic!("registration name must contain exactly one string literal");
        };
        name.value()
    }

    #[test]
    fn generated_attribute_names_split_acronyms() {
        assert_eq!(registered_name("HTTPRequest"), "http_request");
        assert_eq!(registered_name("XMLHttpRequest"), "xml_http_request");
    }

    #[test]
    fn generated_attribute_names_lowercase_unicode() {
        assert_eq!(registered_name("ÉclairValue"), "éclair_value");
        assert_eq!(registered_name("ΔeltaValue"), "δelta_value");
        assert_eq!(registered_name("HTTPÉclair"), "http_éclair");
    }
}
