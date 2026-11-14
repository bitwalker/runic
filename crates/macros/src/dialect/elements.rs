//! `#[derive(AttributeComponent)]`, and the field-by-field walk it shares with
//! `#[derive(DialectAttribute)]`.

use darling::ast::{Data, Fields, Style};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{Ident, spanned::Spanned};

use super::type_component::{construct, members};

/// The bodies of a field-by-field walk over a struct or enum, in the shape of
/// `AttributeComponent` (and `AttributeElements`): `visit` calls
/// `AttributeComponent::visit_elements` on each field in declaration order, passing it the
/// parameter named `visitor`; `replace` calls `AttributeComponent::replace_elements` on each field
/// in the same order, passing it the parameter named `replacer`, and answers `None` when every
/// field did, or else `Some` of the value rebuilt from the new fields and clones of the unchanged
/// ones. Both read the value through `self`.
pub struct ElementsFieldwise {
    pub visit: TokenStream,
    pub replace: TokenStream,
}

impl ElementsFieldwise {
    pub fn new(data: &Data<syn::Variant, syn::Field>) -> Self {
        match data {
            Data::Struct(fields) => Self::of_struct(fields),
            Data::Enum(variants) => Self::of_enum(variants),
        }
    }

    fn of_struct(fields: &Fields<syn::Field>) -> Self {
        let members = members(&fields.fields);
        let places: Vec<TokenStream> = fields
            .fields
            .iter()
            .zip(&members)
            .map(|(field, member)| quote_spanned!(field.ty.span() => &self.#member))
            .collect();
        let visits = visit_calls(&fields.fields, &places);
        let replaced = rebuild(quote!(Self), fields.style, &fields.fields, &members, &places);
        let (unused_visitor, unused_replacer) = unused(fields.fields.is_empty());
        Self {
            visit: quote! {
                #unused_visitor
                #(#visits)*
            },
            replace: quote! {
                #unused_replacer
                #replaced
            },
        }
    }

    fn of_enum(variants: &[syn::Variant]) -> Self {
        let mut visit_arms = Vec::with_capacity(variants.len());
        let mut replace_arms = Vec::with_capacity(variants.len());
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
            let visits = visit_calls(&fields, &places);
            let replaced = rebuild(quote!(Self::#name), style, &fields, &members, &places);
            visit_arms.push(quote!(#pattern => { #(#visits)* }));
            replace_arms.push(quote!(#pattern => { #replaced }));
        }
        let no_fields = variants.iter().all(|variant| variant.fields.is_empty());
        let (unused_visitor, unused_replacer) = unused(no_fields);
        // A reference to an empty enum is not treated as uninhabited, so an enum without variants
        // matches on the value itself, which has no arms to bind anything by move.
        let scrutinee = if variants.is_empty() {
            quote!(*self)
        } else {
            quote!(self)
        };
        Self {
            visit: quote! {
                #unused_visitor
                match #scrutinee {
                    #(#visit_arms,)*
                }
            },
            replace: quote! {
                #unused_replacer
                match #scrutinee {
                    #(#replace_arms,)*
                }
            },
        }
    }
}

/// Statements marking the parameters as used, for a value with no fields to pass them to.
fn unused(no_fields: bool) -> (TokenStream, TokenStream) {
    if no_fields {
        (quote!(let _ = visitor;), quote!(let _ = replacer;))
    } else {
        (TokenStream::new(), TokenStream::new())
    }
}

/// `AttributeComponent::visit_elements(place, visitor);` for each field. The call and its `place`
/// are spanned at the field's type, so that a field type without the impl is reported there.
fn visit_calls(fields: &[syn::Field], places: &[TokenStream]) -> Vec<TokenStream> {
    fields
        .iter()
        .zip(places)
        .map(|(field, place)| {
            quote_spanned! { field.ty.span() =>
                ::runic_core::attributes::AttributeComponent::visit_elements(#place, visitor);
            }
        })
        .collect()
}

/// The replacement of a value built by `path` from `fields` (read at `places`): `None` when no
/// field changed, or else `Some` of the value with each changed field replaced and each unchanged
/// one cloned. Every field's replacement is asked for, in order, before the answer is known.
fn rebuild(
    path: TokenStream,
    style: Style,
    fields: &[syn::Field],
    members: &[syn::Member],
    places: &[TokenStream],
) -> TokenStream {
    let replacements: Vec<TokenStream> = fields
        .iter()
        .zip(places)
        .map(|(field, place)| {
            quote_spanned! { field.ty.span() =>
                ::runic_core::attributes::AttributeComponent::replace_elements(#place, replacer)
            }
        })
        .collect();
    match fields.len() {
        0 => quote!(::core::option::Option::None),
        // The one field's answer is the value's.
        1 => {
            let replacement = &replacements[0];
            let values = [quote!(#replacement?)];
            let built = construct(path, style, members, &values);
            quote!(::core::option::Option::Some(#built))
        }
        _ => {
            let news: Vec<Ident> = fields
                .iter()
                .enumerate()
                .map(|(index, field)| format_ident!("__new_{index}", span = field.ty.span()))
                .collect();
            let values: Vec<TokenStream> = fields
                .iter()
                .zip(&news)
                .zip(places)
                // Not spanned at the field: the clone of a `Copy` field is generated code, not
                // the user's.
                .map(|((_, new), place)| {
                    quote! {
                        match #new {
                            ::core::option::Option::Some(value) => value,
                            ::core::option::Option::None => ::core::clone::Clone::clone(#place),
                        }
                    }
                })
                .collect();
            let built = construct(path, style, members, &values);
            let unchanged = news
                .iter()
                .map(|new| quote!(#new.is_none()))
                .reduce(|all, next| quote!(#all && #next));
            quote! {
                #(let #news = #replacements;)*
                if #unchanged {
                    ::core::option::Option::None
                } else {
                    ::core::option::Option::Some(#built)
                }
            }
        }
    }
}

/// Derives `AttributeComponent` for a non-generic struct or enum, field by field.
pub fn derive_attribute_component(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    if let syn::Data::Union(data) = &input.data {
        return Err(syn::Error::new(
            data.union_token.span(),
            "`AttributeComponent` cannot be derived for a union: its fields cannot be walked \
             safely",
        ));
    }
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new(
            input.generics.span(),
            "`AttributeComponent` cannot be derived for a generic type",
        ));
    }
    let data = Data::try_from(&input.data)
        .map_err(|error| syn::Error::new(input.ident.span(), error.to_string()))?;
    let ElementsFieldwise { visit, replace } = ElementsFieldwise::new(&data);
    let ident = &input.ident;
    Ok(quote! {
        #[automatically_derived]
        impl ::runic_core::attributes::AttributeComponent for #ident {
            fn visit_elements(
                &self,
                visitor: &mut dyn ::runic_core::attributes::ElementVisitor,
            ) {
                #visit
            }

            fn replace_elements(
                &self,
                replacer: &mut dyn ::runic_core::attributes::ElementReplacer,
            ) -> ::core::option::Option<Self> {
                #replace
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use quote::{ToTokens, quote};

    fn derive(input: proc_macro2::TokenStream) -> syn::Result<String> {
        let input: syn::DeriveInput = syn::parse2(input).unwrap();
        let output = super::derive_attribute_component(&input)?;
        // Parsed back, so that the generated code is known to be well formed.
        let file: syn::File = syn::parse2(output).unwrap();
        Ok(file.into_token_stream().to_string())
    }

    fn call(method: &str, place: proc_macro2::TokenStream, parameter: &str) -> String {
        let method = syn::Ident::new(method, proc_macro2::Span::call_site());
        let parameter = syn::Ident::new(parameter, proc_macro2::Span::call_site());
        quote!(::runic_core::attributes::AttributeComponent::#method(#place, #parameter))
            .to_string()
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
        for method in [("visit_elements", "visitor"), ("replace_elements", "replacer")] {
            let calls = ["b", "a", "c"].map(|field| {
                let field = syn::Ident::new(field, proc_macro2::Span::call_site());
                call(method.0, quote!(&self.#field), method.1)
            });
            let at = positions(&output, &calls);
            assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "{method:?}: {output}");
        }
        assert!(
            output.contains(
                &quote!(__new_0.is_none() && __new_1.is_none() && __new_2.is_none()).to_string()
            ),
            "unchanged only when no field changed: {output}"
        );
        assert!(
            output.contains(&quote!(::core::clone::Clone::clone(&self.a)).to_string()),
            "an unchanged field is cloned: {output}"
        );
    }

    #[test]
    fn a_single_field_answers_for_the_value() {
        let output = derive(quote! { struct List(Items); }).unwrap();
        let replaced = call("replace_elements", quote!(&self.0), "replacer");
        assert!(output.contains(&format!("Self ({replaced} ?)")), "{replaced} ? in {output}");
        assert!(!output.contains("Clone"), "nothing to clone: {output}");
    }

    #[test]
    fn a_value_without_fields_holds_nothing() {
        let output = derive(quote! { struct Unit; }).unwrap();
        assert!(!output.contains("AttributeComponent :: visit_elements"), "{output}");
        assert!(output.contains("let _ = replacer ;"), "{output}");
        assert!(output.contains(":: core :: option :: Option :: None"), "{output}");
    }

    #[test]
    fn an_enum_has_one_arm_per_variant() {
        let output = derive(quote! {
            enum Shape {
                Unit,
                Tuple(Ty, Ty),
                Named { visitor: Ty },
            }
        })
        .unwrap();
        for arm in [
            quote!(Self::Unit => { ::core::option::Option::None }),
            quote!(Self::Named { visitor: __field_0 } => {
                ::core::option::Option::Some(Self::Named {
                    visitor: ::runic_core::attributes::AttributeComponent::replace_elements(
                        __field_0, replacer
                    )?
                })
            }),
        ] {
            assert!(output.contains(&arm.to_string()), "{arm} in {output}");
        }
        assert!(output.contains(&quote!(Self::Tuple(__field_0, __field_1)).to_string()));
        let output = derive(quote! { enum Empty {} }).unwrap();
        assert!(output.contains("match * self { }"), "{output}");
    }

    #[test]
    fn generic_types_and_unions_are_rejected() {
        let error = derive(quote! { struct Wrapper<T>(T); }).unwrap_err();
        assert!(error.to_string().contains("generic"), "{error}");
        let error = derive(quote! { union Bits { a: u8, b: i8 } }).unwrap_err();
        assert!(error.to_string().contains("union"), "{error}");
    }
}
