use darling::FromDeriveInput;
use heck::ToSnakeCase;
use quote::quote;
use syn::{Ident, spanned::Spanned};

/// The parsed definition of a dialect type: a struct or enum whose values are the type's
/// parameters, and which is interned as-is.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(ty), supports(struct_any, enum_any))]
pub struct DialectType {
    ident: Ident,
    generics: syn::Generics,
    dialect: syn::Path,
    #[darling(default)]
    name: Option<syn::LitStr>,
    #[darling(default)]
    implements: darling::util::PathList,
}

impl DialectType {
    /// The name of the type within its dialect: the `name` argument, or else the Rust identifier
    /// without a trailing `Type`, snake-cased.
    fn type_name(&self) -> syn::LitStr {
        self.name.clone().unwrap_or_else(|| {
            let ident = self.ident.to_string();
            let base = ident.strip_suffix("Type").filter(|base| !base.is_empty()).unwrap_or(&ident);
            syn::LitStr::new(&base.to_snake_case(), self.ident.span())
        })
    }
}

impl quote::ToTokens for DialectType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        if !self.generics.params.is_empty() {
            tokens.extend(
                syn::Error::new(
                    self.generics.span(),
                    "`DialectType` cannot be derived for a generic type: each type is registered \
                     once with its dialect",
                )
                .into_compile_error(),
            );
            return;
        }

        for (hook, reason) in [
            ("TypePrinter", "a type that implements it is printed with it automatically"),
            (
                "TypeParser",
                "a type that implements it is parsed with it automatically, and it cannot be \
                 looked up as an interface",
            ),
        ] {
            if let Some(path) = self
                .implements
                .iter()
                .find(|path| path.segments.last().is_some_and(|segment| segment.ident == hook))
            {
                tokens.extend(
                    syn::Error::new(
                        path.span(),
                        format!("`{hook}` must not be listed in `implements(..)`: {reason}"),
                    )
                    .into_compile_error(),
                );
                return;
            }
        }

        let ident = &self.ident;
        let dialect = &self.dialect;
        let name = self.type_name();
        let implements = &self.implements;
        tokens.extend(quote! {
            impl ::runic_core::types::TypeImpl for #ident {}

            impl ::runic_core::types::TypeRegistration for #ident {
                type Dialect = #dialect;

                fn name() -> ::runic_core::interner::Symbol {
                    ::runic_core::interner::Symbol::intern(#name)
                }

                fn traits() -> ::runic_core::derive::__private::Box<[::runic_core::traits::TraitInfo]> {
                    ::runic_core::derive::__private::Box::from([
                        ::runic_core::traits::TraitInfo::new::<Self, dyn ::core::any::Any>(),
                        #(
                            ::runic_core::traits::TraitInfo::new::<Self, dyn #implements>(),
                        )*
                    ])
                }
            }

            ::runic_core::inventory::submit!(
                ::runic_core::types::DialectTypeRegistrationInfo::new::<#ident>()
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use quote::{ToTokens, quote};

    fn derive(input: proc_macro2::TokenStream) -> darling::Result<syn::File> {
        let input: syn::DeriveInput = syn::parse2(input).unwrap();
        let output = crate::dialect::derive_type(&input)?.into_token_stream();
        Ok(syn::parse2(output).unwrap())
    }

    /// The literal returned by the generated `TypeRegistration::name`.
    fn registered_name(file: &syn::File) -> String {
        let registration = file
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Impl(item)
                    if item.trait_.as_ref().is_some_and(|(_, path, _)| {
                        path.segments.last().unwrap().ident == "TypeRegistration"
                    }) =>
                {
                    Some(item)
                }
                _ => None,
            })
            .expect("a TypeRegistration impl");
        let name = registration
            .items
            .iter()
            .find_map(|item| match item {
                syn::ImplItem::Fn(method) if method.sig.ident == "name" => Some(&method.block),
                _ => None,
            })
            .expect("a name method");
        let [syn::Stmt::Expr(syn::Expr::Call(call), None)] = name.stmts.as_slice() else {
            panic!("registration name must return the interned literal");
        };
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

    fn error(input: proc_macro2::TokenStream) -> String {
        derive(input).expect_err("the derive rejects this input").to_string()
    }

    #[test]
    fn the_default_name_drops_a_trailing_type_and_is_snake_cased() {
        let file = derive(quote! { #[ty(dialect = D)] struct FloatVectorType(u32); }).unwrap();
        assert_eq!(registered_name(&file), "float_vector");
        let file = derive(quote! { #[ty(dialect = D)] enum Type { A } }).unwrap();
        assert_eq!(registered_name(&file), "type");
        let file = derive(quote! { #[ty(dialect = D, name = "vec")] struct Vector; }).unwrap();
        assert_eq!(registered_name(&file), "vec");
    }

    #[test]
    fn the_value_is_registered_as_written() {
        let file = derive(quote! { #[ty(dialect = D)] struct PairType { a: u8, b: u8 } }).unwrap();
        assert!(
            !file.items.iter().any(|item| matches!(item, syn::Item::Struct(_))),
            "the derive must not define a wrapper struct"
        );
        let output = file.into_token_stream().to_string();
        assert!(
            output.contains(
                &quote!(::runic_core::types::DialectTypeRegistrationInfo::new::<PairType>())
                    .to_string()
            )
        );
    }

    #[test]
    fn generated_paths_go_through_runic_core() {
        let file = derive(quote! { #[ty(dialect = D)] struct XType; }).unwrap();
        let output = file.into_token_stream().to_string();
        assert!(!output.contains(":: alloc ::"), "a consumer need not declare `alloc`: {output}");
        assert!(output.contains(&quote!(::runic_core::derive::__private::Box).to_string()));
    }

    #[test]
    fn malformed_arguments_are_reported() {
        assert!(error(quote! { #[ty(name = "x")] struct XType; }).contains("dialect"));
        assert!(error(quote! { #[ty(dialect = D, nmae = "x")] struct XType; }).contains("nmae"));
        assert!(!error(quote! { #[ty(dialect = D)] union XType { a: u8 } }).is_empty());
    }

    #[test]
    fn listing_the_printer_is_rejected() {
        let file = derive(quote! {
            #[ty(dialect = D, implements(Describe, crate::types::TypePrinter))]
            struct XType;
        })
        .unwrap();
        let output = file.into_token_stream().to_string();
        assert!(output.contains("compile_error") && output.contains("TypePrinter"));
        assert!(!output.contains("TypeRegistration"));
    }

    #[test]
    fn listing_the_parser_is_rejected() {
        let file = derive(quote! {
            #[ty(dialect = D, implements(TypeParser, Describe))]
            struct XType;
        })
        .unwrap();
        let output = file.into_token_stream().to_string();
        assert!(output.contains("compile_error") && output.contains("TypeParser"));
        assert!(!output.contains("TypeRegistration"));
    }

    #[test]
    fn generic_types_are_rejected() {
        let file = derive(quote! { #[ty(dialect = D)] struct XType<T>(T); }).unwrap();
        let output = file.into_token_stream().to_string();
        assert!(output.contains("compile_error"));
        assert!(!output.contains("TypeRegistration"));
    }
}
