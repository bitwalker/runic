#![allow(dead_code)]

use quote::ToTokens;

#[test]
fn operation_names_with_digits_support_explicit_registration_spelling() {
    for (override_name, expected) in [(None, "i32_load8_s"), (Some("i32_load_8s"), "i32_load_8s")] {
        let name_option = override_name.map(|name| quote::quote! { name = #name, });
        let input = syn::parse_quote! {
            #[operation(dialect = WasmDialect, #name_option)]
            pub struct I32Load8S {}
        };
        let output = crate::operation::derive_operation(input).unwrap();
        let file: syn::File = syn::parse2(output).unwrap();
        let registration = file
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Impl(item)
                    if item.trait_.as_ref().is_some_and(|(_, path, _)| {
                        path.segments.last().unwrap().ident == "OpRegistration"
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
                syn::ImplItem::Fn(method) if method.sig.ident == "name" => Some(&method.block),
                _ => None,
            })
            .unwrap();
        let expected: syn::Block = syn::parse_quote! {
            { ::runic_core::interner::Symbol::intern(#expected) }
        };
        assert_eq!(name, &expected);
    }
}

#[test]
fn derive_attribute_test() {
    let item_input: syn::DeriveInput = syn::parse_quote! {
        /// A simple boolean attribute
        #[derive(DialectAttribute, Debug, Copy, Clone, PartialEq, Eq)]
        #[attribute(
            dialect = TestDialect,
            traits(BooleanLikeAttr),
            implements(AttrPrinter)
        )]
        pub struct Bool(bool);
    };

    let output = crate::dialect::derive_attribute(&item_input);
    match output {
        Ok(output) => {
            let formatted = format_output(&output.into_token_stream().to_string());
            println!("{formatted}");
        }
        Err(err) => {
            panic!("{err}");
        }
    }
}

#[test]
fn derive_remote_attribute_test() {
    let item_input: syn::DeriveInput = syn::parse_quote! {
        /// A simple boolean attribute
        #[derive(DialectAttribute, Debug, Copy, Clone, PartialEq, Eq)]
        #[attribute(
            dialect = TestDialect,
            remote = "bool",
            default = "bool::default",
            traits(BooleanLikeAttr),
            implements(AttrPrinter)
        )]
        pub struct Bool;
    };

    let output = crate::dialect::derive_attribute(&item_input);
    match output {
        Ok(output) => {
            let formatted = format_output(&output.into_token_stream().to_string());
            println!("{formatted}");
        }
        Err(err) => {
            panic!("{err}");
        }
    }
}

#[test]
fn derive_effect_op_interface_test() {
    let item_input: syn::DeriveInput = syn::parse_quote! {
        /// A simple boolean attribute
        #[derive(EffectOpInterface)]
        #[effects(MemoryEffect(MemoryEffect::Write))]
        pub struct Load {
            ptr: Ptr
        }
    };

    let output = crate::operations::derive_effect_op_interface(&item_input);
    match output {
        Ok(output) => {
            let formatted = format_output(&output.into_token_stream().to_string());
            println!("{formatted}");
        }
        Err(err) => {
            panic!("{err}");
        }
    }
}

#[test]
fn derive_effect_op_interface_fields_test() {
    let item_input: syn::DeriveInput = syn::parse_quote! {
        /// A simple boolean attribute
        #[derive(EffectOpInterface)]
        pub struct Load {
            #[effects(MemoryEffect(MemoryEffect::Read))]
            ptr: Ptr
        }
    };

    let output = crate::operations::derive_effect_op_interface(&item_input);
    match output {
        Ok(output) => {
            let formatted = format_output(&output.into_token_stream().to_string());
            println!("{formatted}");
        }
        Err(err) => {
            panic!("{err}");
        }
    }
}

#[test]
fn operation_trait_no_verifier_test() {
    let meta = Vec::default();
    let item_input: syn::ItemTrait = syn::parse_quote! {
        #[operation_trait]
        pub trait SameOperandsAndResultType : SameTypeOperands {}
    };

    let output = crate::operations::derive_operation_trait(meta, item_input);
    match output {
        Ok(output) => {
            let formatted = format_output(&output.into_token_stream().to_string());
            println!("{formatted}");
        }
        Err(err) => {
            panic!("{err}");
        }
    }
}

#[test]
fn operation_trait_with_verifiers_test() {
    let meta = Vec::default();
    let item_input: syn::ItemTrait = syn::parse_quote! {
        #[operation_trait]
        pub trait SameOperandsAndResultType : SameTypeOperands {
            #[verifier]
            fn has_same_operands_and_result_type(_op: &::runic_core::Operation, _context: &::runic_core::Context) -> Result<(), ::runic_core::diagnostics::Report> {
                true
            }

            #[verifier]
            fn additional_check(_op: &::runic_core::Operation, _context: &::runic_core::Context) -> Result<(), ::runic_core::diagnostics::Report> {
                true
            }
        }
    };

    let output = crate::operations::derive_operation_trait(meta, item_input);
    match output {
        Ok(output) => {
            let formatted = format_output(&output.into_token_stream().to_string());
            println!("{formatted}");
        }
        Err(err) => {
            panic!("{err}");
        }
    }
}

#[test]
fn operation_trait_with_generics_test() {
    let meta = Vec::default();
    let item_input: syn::ItemTrait = syn::parse_quote! {
        #[operation_trait]
        pub trait SameOperandsAndResultType<T: AttributeRegistration> : SameTypeOperands<T> + UnusedGenerics {
            #[verifier]
            fn verify_it<T: AttributeRegistration>(_op: &::runic_core::Operation, _context: &::runic_core::Context) -> Result<(), ::runic_core::diagnostics::Report> {
                true
            }
        }
    };

    let output = crate::operations::derive_operation_trait(meta, item_input);
    match output {
        Ok(output) => {
            let formatted = format_output(&output.into_token_stream().to_string());
            println!("{formatted}");
        }
        Err(err) => {
            panic!("{err}");
        }
    }
}

#[test]
fn operation_impl_test() {
    let item_input: syn::DeriveInput = syn::parse_quote! {
        /// Two's complement sum
        #[operation(
            dialect = TestDialect,
            traits(BinaryOp, Commutative, SameTypeOperands),
            implements(InferTypeOpInterface),
        )]
        pub struct Add {
            /// The left-hand operand
            #[operand]
            lhs: AnyInteger,
            #[operand]
            rhs: AnyInteger,
            #[result]
            result: AnyInteger,
            #[attr]
            overflow: OverflowAttr,
        }
    };

    let output = crate::operation::derive_operation(item_input);
    match output {
        Ok(output) => {
            let formatted = format_output(&output.to_string());
            println!("{formatted}");
        }
        Err(err) => {
            panic!("{err}");
        }
    }
}

/// The generated items of `#[operation]` on `input`, as a file.
fn expand_operation(input: syn::DeriveInput) -> syn::File {
    let output = crate::operation::derive_operation(input).unwrap();
    syn::parse2(output).unwrap()
}

/// The `impl` blocks in `file` of the trait whose last path segment is `name`.
fn trait_impls<'a>(file: &'a syn::File, name: &str) -> Vec<&'a syn::ItemImpl> {
    file.items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Impl(item)
                if item
                    .trait_
                    .as_ref()
                    .is_some_and(|(_, path, _)| path.segments.last().unwrap().ident == name) =>
            {
                Some(item)
            }
            _ => None,
        })
        .collect()
}

/// The generic argument of the trait `impl`, e.g. the argument tuple of `FnOnce<(..)>`.
fn trait_argument(item: &syn::ItemImpl) -> String {
    let (_, path, _) = item.trait_.as_ref().unwrap();
    match &path.segments.last().unwrap().arguments {
        syn::PathArguments::AngleBracketed(args) => args.args.to_token_stream().to_string(),
        _ => String::new(),
    }
}

/// The method `name` defined in an inherent `impl` block of `file`.
fn inherent_method<'a>(file: &'a syn::File, name: &str) -> &'a syn::ImplItemFn {
    file.items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Impl(item) if item.trait_.is_none() => Some(item),
            _ => None,
        })
        .flat_map(|item| item.items.iter())
        .find_map(|item| match item {
            syn::ImplItem::Fn(method) if method.sig.ident == name => Some(method),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no method `{name}`"))
}

/// An attribute parameter of the builder and the setter accepts anything that is
/// `IntoAttribute` the attribute: a value or a ready-made attribute. The `BuildableOp` impl that
/// `create::<Op, _>` selects keeps the concrete value type.
#[test]
fn attribute_parameters_are_bounded_by_into_attribute() {
    let file = expand_operation(syn::parse_quote! {
        #[operation(dialect = TestDialect)]
        pub struct Constant {
            #[attr]
            value: IntAttr,
            #[result]
            result: AnyInteger,
        }
    });
    let text = file.to_token_stream().to_string();
    assert!(!text.contains("From <"), "{text}");

    let fn_once = trait_impls(&file, "FnOnce");
    assert_eq!(fn_once.len(), 1);
    let where_clause = fn_once[0].generics.where_clause.to_token_stream().to_string();
    assert!(
        where_clause.contains("TValueValue : :: runic_core :: IntoAttribute < IntAttr >"),
        "{where_clause}"
    );

    let create = inherent_method(&file, "create");
    let where_clause = create.sig.generics.where_clause.to_token_stream().to_string();
    assert!(
        where_clause.contains("TValueValue : :: runic_core :: IntoAttribute < IntAttr >"),
        "{where_clause}"
    );

    let buildable = trait_impls(&file, "BuildableOp");
    assert_eq!(buildable.len(), 1);
    assert_eq!(
        trait_argument(buildable[0]),
        "(< IntAttr as :: runic_core :: AttributeRegistration > :: Value ,)"
    );

    let setter = inherent_method(&file, "set_value");
    assert!(setter.sig.generics.params.is_empty(), "{}", setter.sig.to_token_stream());
    let input = setter.sig.inputs.iter().nth(1).unwrap().to_token_stream().to_string();
    assert_eq!(input, "value : impl :: runic_core :: IntoAttribute < IntAttr >");
}

/// An op with a `#[default]` attribute still has a minimal builder, which takes only the
/// required parameters.
#[test]
fn a_defaulted_attribute_keeps_the_minimal_builder() {
    let file = expand_operation(syn::parse_quote! {
        #[operation(dialect = TestDialect)]
        pub struct AddrOf {
            #[operand]
            base: AnyPointer,
            #[attr]
            #[default]
            offset: I32Attr,
            #[result]
            result: AnyPointer,
        }
    });
    let mut buildable = trait_impls(&file, "BuildableOp")
        .into_iter()
        .map(trait_argument)
        .collect::<Vec<_>>();
    buildable.sort();
    assert_eq!(
        buildable,
        [
            "(:: runic_core :: ValueRef , < I32Attr as :: runic_core :: AttributeRegistration > \
             :: Value)",
            "(:: runic_core :: ValueRef ,)",
        ]
    );

    let mut fn_once =
        trait_impls(&file, "FnOnce").into_iter().map(trait_argument).collect::<Vec<_>>();
    fn_once.sort();
    assert_eq!(
        fn_once,
        ["(:: runic_core :: ValueRef , TOffsetValue)", "(:: runic_core :: ValueRef ,)"]
    );
    let text = file.to_token_stream().to_string();
    assert!(!text.contains("From <"), "{text}");
}

fn format_output(input: &str) -> String {
    use std::{
        io::{Read, Write},
        process::{Command, Stdio},
    };

    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024"])
        .args([
            "--config",
            "unstable_features=true,normalize_doc_attributes=true,use_field_init_shorthand=true,\
             condense_wildcard_suffixes=true,format_strings=true,group_imports=StdExternalCrate,\
             imports_granularity=Crate",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn 'rustfmt'");

    {
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(input.as_bytes()).expect("failed to write input to 'rustfmt'");
    }
    let mut buf = String::new();
    let mut stdout = child.stdout.take().unwrap();
    stdout.read_to_string(&mut buf).expect("failed to read output from 'rustfmt'");
    match child.wait() {
        Ok(status) => {
            if status.success() {
                buf
            } else {
                let mut stderr = child.stderr.take().unwrap();
                let mut err_buf = String::new();
                let _ = stderr.read_to_string(&mut err_buf).ok();
                panic!(
                    "command 'rustfmt' failed with status {:?}\n\nReason: {}",
                    status.code(),
                    if err_buf.is_empty() {
                        "<no output>"
                    } else {
                        err_buf.as_str()
                    },
                );
            }
        }
        Err(err) => panic!("command 'rustfmt' failed with {err}"),
    }
}

#[test]
fn effect_derive_rejects_malformed_fields_before_expansion() {
    let input: syn::DeriveInput = syn::parse_quote! {
        #[effects(MemoryEffect(MemoryEffect::Read))]
        struct InvalidEffects {
            #[effects(MemoryEffect)]
            value: ValueRef,
        }
    };
    assert!(crate::operations::derive_effect_op_interface(&input).is_err());
}

#[test]
fn an_attribute_cannot_both_have_no_default_and_give_one() {
    let input: syn::DeriveInput = syn::parse_quote! {
        #[attribute(dialect = TestDialect, no_default, default = make_default)]
        struct Contradictory(bool);
    };
    let Err(err) = crate::dialect::derive_attribute(&input) else {
        panic!("a contradiction is an error");
    };
    assert!(err.to_string().contains("`no_default` cannot also give a `default`"), "{err}");

    // Either alone is accepted
    for input in [
        syn::parse_quote! { #[attribute(dialect = TestDialect, no_default)] struct A(bool); },
        syn::parse_quote! { #[attribute(dialect = TestDialect, default = make_default)] struct B(bool); },
    ] {
        assert!(crate::dialect::derive_attribute(&input).is_ok());
    }
}

#[test]
fn a_defaulted_property_asserts_that_its_attribute_has_a_default() {
    let input = syn::parse_quote! {
        #[operation(dialect = TestDialect)]
        pub struct Add {
            #[operand]
            lhs: AnyInteger,
            #[attr]
            #[default]
            overflow: OverflowAttr,
            #[attr]
            required: OverflowAttr,
        }
    };
    let output = crate::operation::derive_operation(input).unwrap().to_string();
    // One check, for the defaulted property, as an item so that it is always evaluated
    let condition = quote::quote! {
        <OverflowAttr as ::runic_core::AttributeRegistration>::HAS_DEFAULT
    };
    let check = format!("{} ({condition} ,", quote::quote! { const _: () = assert! });
    assert_eq!(output.matches("HAS_DEFAULT").count(), 1, "{output}");
    assert!(output.contains(&check), "{check}\n{output}");
    assert!(output.contains("the property `overflow` is marked `#[default]`"), "{output}");
}
