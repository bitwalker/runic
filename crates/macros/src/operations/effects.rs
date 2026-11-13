use std::collections::BTreeMap;

use darling::{FromDeriveInput, FromField};
use quote::{ToTokens, quote};
use syn::{Ident, Token, parse_quote, punctuated::Punctuated};

pub fn derive_effect_op_interface(input: &syn::DeriveInput) -> darling::Result<EffectOpInterface> {
    let input = EffectOpInterfaceInput::from_derive_input(input)?;
    let fields = input.data.take_struct().unwrap().fields;
    let mut effects = BTreeMap::<String, EffectGroup>::new();
    let declarations = std::iter::once((None, input.attrs.as_slice()))
        .chain(fields.iter().map(|field| (field.ident.clone(), field.attrs.as_slice())));
    for (field, attrs) in declarations {
        for Effect {
            kind,
            values,
            resource,
        } in parse_effects(attrs)?
        {
            let group =
                effects
                    .entry(kind.to_token_stream().to_string())
                    .or_insert_with(|| EffectGroup {
                        kind,
                        instances: Vec::new(),
                    });
            group.instances.extend(values.into_iter().map(|value| EffectInstance {
                field: field.clone(),
                value,
                resource: resource.clone(),
            }));
        }
    }
    if effects.is_empty() {
        let kind: syn::Path = parse_quote!(::runic_core::effects::MemoryEffect);
        effects.insert(
            kind.to_token_stream().to_string(),
            EffectGroup {
                kind,
                instances: Vec::new(),
            },
        );
    }

    Ok(EffectOpInterface {
        ident: input.ident,
        generics: input.generics,
        effects,
    })
}

#[derive(Debug, FromDeriveInput)]
#[darling(
    forward_attrs(doc, cfg, allow, derive, effects),
    supports(struct_named)
)]
struct EffectOpInterfaceInput {
    ident: Ident,
    generics: syn::Generics,
    attrs: Vec<syn::Attribute>,
    data: darling::ast::Data<(), FieldEffect>,
}

pub struct EffectOpInterface {
    ident: Ident,
    generics: syn::Generics,
    effects: BTreeMap<String, EffectGroup>,
}

struct EffectGroup {
    kind: syn::Path,
    instances: Vec<EffectInstance>,
}

struct EffectInstance {
    field: Option<Ident>,
    value: syn::Expr,
    /// The resource the effect acts on, or `None` for the default resource.
    resource: Option<syn::Path>,
}

/// The effects of one kind named in an `#[effects(..)]` attribute, with the resource named by the
/// attribute's `resource(..)`, if any.
struct Effect {
    kind: syn::Path,
    values: Punctuated<syn::Expr, Token![,]>,
    resource: Option<syn::Path>,
}

/// One item of an `#[effects(..)]` attribute: `Kind(value, ..)`, or `resource(Path)`, the
/// resource of every effect of that attribute.
enum EffectItem {
    Effect(Effect),
    Resource(syn::Path),
}

impl syn::parse::Parse for EffectItem {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kind = input.parse::<syn::Path>()?;
        let values;
        let _paren = syn::parenthesized!(values in input);
        if kind.is_ident("resource") {
            let resource = values.parse::<syn::Path>()?;
            if !values.is_empty() {
                return Err(values.error("expected one path to a resource"));
            }
            return Ok(Self::Resource(resource));
        }
        Ok(Self::Effect(Effect {
            kind,
            values: values.parse_terminated(syn::Expr::parse, Token![,])?,
            resource: None,
        }))
    }
}

#[derive(Debug, FromField)]
#[darling(forward_attrs(doc, cfg, allow, effects))]
struct FieldEffect {
    ident: Option<Ident>,
    attrs: Vec<syn::Attribute>,
}

impl ToTokens for EffectOpInterface {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let op_type = &self.ident;
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();
        for EffectGroup { kind, instances } in self.effects.values() {
            let no_effects = instances.is_empty();
            let values = instances.iter().map(
                |EffectInstance {
                     field,
                     value,
                     resource,
                 }| {
                    let resource = match resource {
                        Some(resource) => quote! {
                            <#resource as ::core::default::Default>::default()
                        },
                        None => quote! { ::runic_core::effects::DefaultResource },
                    };
                    match field {
                        Some(field) => quote! {
                            ::runic_core::effects::EffectInstance::new_for_value_with_resource(
                                #value,
                                self.#field(),
                                #resource,
                            )
                        },
                        None => quote! {
                            ::runic_core::effects::EffectInstance::new_with_resource(
                                #value,
                                #resource,
                            )
                        },
                    }
                },
            );
            tokens.extend(quote! {
                impl #impl_generics ::runic_core::effects::EffectOpInterface<#kind> for #op_type #ty_generics #where_clause {
                    #[inline(always)]
                    fn has_no_effect(&self) -> bool {
                        #no_effects
                    }

                    fn effects(&self) -> ::runic_core::effects::EffectIterator<#kind> {
                        ::runic_core::effects::EffectIterator::from_smallvec(
                            ::runic_core::smallvec![#(#values),*]
                        )
                    }
                }
            });
        }
    }
}

/// The effects named by the `#[effects(..)]` attributes in `attrs`. A `resource(Path)` in an
/// attribute is the resource of every effect named in that attribute; without one, the effects act
/// on the default resource.
fn parse_effects(attrs: &[syn::Attribute]) -> syn::Result<Vec<Effect>> {
    let mut effects = Vec::new();
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("effects")) {
        let list = attr.meta.require_list()?;
        let items = list.parse_args_with(Punctuated::<EffectItem, Token![,]>::parse_terminated)?;
        let mut resource = None;
        let mut named = Vec::new();
        for item in items {
            match item {
                EffectItem::Effect(effect) => named.push(effect),
                EffectItem::Resource(path) => {
                    if resource.is_some() {
                        return Err(syn::Error::new_spanned(
                            path,
                            "an `effects` attribute names at most one resource",
                        ));
                    }
                    resource = Some(path);
                }
            }
        }
        if named.is_empty()
            && let Some(resource) = resource
        {
            return Err(syn::Error::new_spanned(
                resource,
                "a `resource(..)` needs an effect in the same `effects` attribute",
            ));
        }
        effects.extend(named.into_iter().map(|effect| Effect {
            resource: resource.clone(),
            ..effect
        }));
    }
    Ok(effects)
}
