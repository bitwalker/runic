//! `#[derive(TypeComponent)]`, and the field-by-field walk it shares with
//! `#[derive(DialectType)]`.

use darling::ast::{Data, Fields, Style};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::{Ident, spanned::Spanned};

/// The bodies of a field-by-field walk over a struct or enum: `visit` calls
/// `TypeComponent::visit_types` on each field in declaration order, passing it the closure named
/// `visit`; `map` builds `Self` from `TypeComponent::map_types` of each field, in the same order,
/// passing it the closure named `map`. Both read the value through `self`.
pub struct Fieldwise {
    pub visit: TokenStream,
    pub map: TokenStream,
}

impl Fieldwise {
    pub fn new(data: &Data<syn::Variant, syn::Field>) -> Self {
        let visit = Ident::new("visit", Span::call_site());
        let map = Ident::new("map", Span::call_site());
        match data {
            Data::Struct(fields) => Self::of_struct(fields, &visit, &map),
            Data::Enum(variants) => Self::of_enum(variants, &visit, &map),
        }
    }

    fn of_struct(fields: &Fields<syn::Field>, visit: &Ident, map: &Ident) -> Self {
        let members = members(&fields.fields);
        let places: Vec<TokenStream> = fields
            .fields
            .iter()
            .zip(&members)
            .map(|(field, member)| quote_spanned!(field.ty.span() => &self.#member))
            .collect();
        let visits = visit_calls(&fields.fields, &places, visit);
        let maps = map_calls(&fields.fields, &places, map);
        let built = construct(quote!(Self), fields.style, &members, &maps);
        let (unused_visit, unused_map) = unused(fields.fields.is_empty(), visit, map);
        Self {
            visit: quote! {
                #unused_visit
                #(#visits)*
            },
            map: quote! {
                #unused_map
                #built
            },
        }
    }

    fn of_enum(variants: &[syn::Variant], visit: &Ident, map: &Ident) -> Self {
        let mut visit_arms = Vec::with_capacity(variants.len());
        let mut map_arms = Vec::with_capacity(variants.len());
        for variant in variants {
            let name = &variant.ident;
            let fields: Vec<syn::Field> = variant.fields.iter().cloned().collect();
            let style = Style::from(&variant.fields);
            let members = members(&fields);
            // Fields are bound to fresh names, so that a field named like a parameter does not
            // shadow it.
            let bindings: Vec<Ident> = fields
                .iter()
                .enumerate()
                .map(|(index, field)| format_ident!("__field_{index}", span = field.ty.span()))
                .collect();
            let places: Vec<TokenStream> =
                bindings.iter().map(|binding| quote!(#binding)).collect();
            let pattern = construct(quote!(Self::#name), style, &members, &places);
            let visits = visit_calls(&fields, &places, visit);
            let maps = map_calls(&fields, &places, map);
            let built = construct(quote!(Self::#name), style, &members, &maps);
            visit_arms.push(quote!(#pattern => { #(#visits)* }));
            map_arms.push(quote!(#pattern => #built));
        }
        let no_fields = variants.iter().all(|variant| variant.fields.is_empty());
        let (unused_visit, unused_map) = unused(no_fields, visit, map);
        Self {
            visit: quote! {
                #unused_visit
                match self {
                    #(#visit_arms,)*
                }
            },
            map: quote! {
                #unused_map
                match self {
                    #(#map_arms,)*
                }
            },
        }
    }
}

/// How each of `fields` is named: by its identifier, or by its index in a tuple.
fn members(fields: &[syn::Field]) -> Vec<syn::Member> {
    fields
        .iter()
        .enumerate()
        .map(|(index, field)| match &field.ident {
            Some(ident) => syn::Member::Named(ident.clone()),
            None => syn::Member::Unnamed(syn::Index {
                index: index as u32,
                span: field.ty.span(),
            }),
        })
        .collect()
}

/// Statements marking the closures as used, for a value with no fields to pass them to.
fn unused(no_fields: bool, visit: &Ident, map: &Ident) -> (TokenStream, TokenStream) {
    if no_fields {
        (quote!(let _ = #visit;), quote!(let _ = #map;))
    } else {
        (TokenStream::new(), TokenStream::new())
    }
}

/// `TypeComponent::visit_types(place, visit);` for each field. The call and its `place` are
/// spanned at the field's type, so that a field type without the impl is reported there.
fn visit_calls(fields: &[syn::Field], places: &[TokenStream], visit: &Ident) -> Vec<TokenStream> {
    fields
        .iter()
        .zip(places)
        .map(|(field, place)| {
            quote_spanned! { field.ty.span() =>
                ::runic_core::types::TypeComponent::visit_types(#place, #visit);
            }
        })
        .collect()
}

/// `TypeComponent::map_types(place, map)` for each field, spanned at the field's type.
fn map_calls(fields: &[syn::Field], places: &[TokenStream], map: &Ident) -> Vec<TokenStream> {
    fields
        .iter()
        .zip(places)
        .map(|(field, place)| {
            quote_spanned! { field.ty.span() =>
                ::runic_core::types::TypeComponent::map_types(#place, #map)
            }
        })
        .collect()
}

/// `path { member: value, .. }`, `path(value, ..)` or `path`, as `style` says; used both as an
/// expression and as a pattern.
fn construct(
    path: TokenStream,
    style: Style,
    members: &[syn::Member],
    values: &[TokenStream],
) -> TokenStream {
    match style {
        Style::Struct => quote!(#path { #(#members: #values),* }),
        Style::Tuple => quote!(#path(#(#values),*)),
        Style::Unit => path,
    }
}

/// Derives `TypeComponent` for a non-generic struct or enum, field by field.
pub fn derive_type_component(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    if let syn::Data::Union(data) = &input.data {
        return Err(syn::Error::new(
            data.union_token.span(),
            "`TypeComponent` cannot be derived for a union: its fields cannot be walked safely",
        ));
    }
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new(
            input.generics.span(),
            "`TypeComponent` cannot be derived for a generic type",
        ));
    }
    let data = Data::try_from(&input.data)
        .map_err(|error| syn::Error::new(input.ident.span(), error.to_string()))?;
    let Fieldwise { visit, map } = Fieldwise::new(&data);
    let ident = &input.ident;
    Ok(quote! {
        #[automatically_derived]
        impl ::runic_core::types::TypeComponent for #ident {
            fn visit_types(&self, visit: &mut dyn FnMut(::runic_core::types::Type)) {
                #visit
            }

            fn map_types(
                &self,
                map: &mut dyn FnMut(::runic_core::types::Type) -> ::runic_core::types::Type,
            ) -> Self {
                #map
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use quote::{ToTokens, quote};

    fn derive(input: proc_macro2::TokenStream) -> syn::Result<String> {
        let input: syn::DeriveInput = syn::parse2(input).unwrap();
        let output = super::derive_type_component(&input)?;
        // Parsed back, so that the generated code is known to be well formed.
        let file: syn::File = syn::parse2(output).unwrap();
        Ok(file.into_token_stream().to_string())
    }

    fn call(method: &str, place: proc_macro2::TokenStream, closure: &str) -> String {
        let method = syn::Ident::new(method, proc_macro2::Span::call_site());
        let closure = syn::Ident::new(closure, proc_macro2::Span::call_site());
        quote!(::runic_core::types::TypeComponent::#method(#place, #closure)).to_string()
    }

    /// The positions of `needles` in `haystack`, which must all occur.
    fn positions(haystack: &str, needles: &[String]) -> Vec<usize> {
        needles
            .iter()
            .map(|needle| {
                haystack
                    .find(needle.as_str())
                    .unwrap_or_else(|| panic!("{needle} in {haystack}"))
            })
            .collect()
    }

    #[test]
    fn fields_are_walked_in_declaration_order() {
        let output = derive(quote! { struct Field { b: Ty, a: Option<Ty>, c: u8 } }).unwrap();
        for method in [("visit_types", "visit"), ("map_types", "map")] {
            let calls = ["b", "a", "c"].map(|field| {
                let field = syn::Ident::new(field, proc_macro2::Span::call_site());
                call(method.0, quote!(&self.#field), method.1)
            });
            let at = positions(&output, &calls);
            assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "{method:?}: {output}");
        }
        assert!(output.contains("Self { b :"), "{output}");
    }

    #[test]
    fn tuple_and_unit_structs_are_supported() {
        let output = derive(quote! { struct Pair(Ty, u8); }).unwrap();
        let at = positions(
            &output,
            &[
                call("map_types", quote!(&self.0), "map"),
                call("map_types", quote!(&self.1), "map"),
            ],
        );
        assert!(at[0] < at[1]);
        assert!(output.contains("Self ("), "{output}");
        let output = derive(quote! { struct Unit; }).unwrap();
        assert!(!output.contains("TypeComponent :: visit_types"), "{output}");
    }

    #[test]
    fn an_enum_has_one_arm_per_variant() {
        let output = derive(quote! {
            enum Shape {
                Unit,
                Tuple(Ty, Ty),
                Named { visit: Ty },
            }
        })
        .unwrap();
        for arm in [
            quote!(Self::Unit => Self::Unit),
            quote!(Self::Tuple(__field_0, __field_1) => Self::Tuple(
                ::runic_core::types::TypeComponent::map_types(__field_0, map),
                ::runic_core::types::TypeComponent::map_types(__field_1, map)
            )),
            quote!(Self::Named { visit: __field_0 } => Self::Named {
                visit: ::runic_core::types::TypeComponent::map_types(__field_0, map)
            }),
        ] {
            assert!(output.contains(&arm.to_string()), "{arm} in {output}");
        }
        assert_eq!(output.matches("=>").count(), 6, "three arms in each method: {output}");
        let output = derive(quote! { enum Empty {} }).unwrap();
        assert!(output.contains("match self { }"), "{output}");
    }

    #[test]
    fn generic_types_and_unions_are_rejected() {
        let error = derive(quote! { struct Wrapper<T>(T); }).unwrap_err();
        assert!(error.to_string().contains("generic"), "{error}");
        let error = derive(quote! { union Bits { a: u8, b: i8 } }).unwrap_err();
        assert!(error.to_string().contains("union"), "{error}");
    }
}
