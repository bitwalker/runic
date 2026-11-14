use alloc::{format, string::String};

use crate::{
    Context, Op, OpPrinter, Operation, Report, Spanned, Type, UnsafeIntrusiveEntityRef, Value,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    diagnostics::report,
    dialects::builtin::{BuiltinDialect, Function, attributes::LocalVariableAttr},
    effects::{MemoryEffect, MemoryEffectOpInterface, Resource},
    traits::{AnyType, InferTypeOpInterface},
};

pub type LoadLocalRef = UnsafeIntrusiveEntityRef<LoadLocal>;
pub type StoreLocalRef = UnsafeIntrusiveEntityRef<StoreLocal>;

/// The local variables of a function, which `builtin.load_local` reads and `builtin.store_local`
/// writes: a child of the default resource, so an access to a local is ordered against every
/// effect on the default resource (an operation with unknown effects, say) and against no effect
/// on another of its children (memory, an assert).
///
/// It is one resource for every local: a write of one local is ordered against a read of another.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LocalsResource;

impl Resource for LocalsResource {
    fn name(&self) -> &'static str {
        "builtin.locals"
    }
}

/// A read of a local variable of the enclosing function:
/// `%v = builtin.load_local <{ local = #builtin.local_variable<0, i32> }>;`.
///
/// The result has the local's type. A local is not memory: it has no address and needs no data
/// layout, so it may hold a value of any type. The load reads `local` in [LocalsResource], and has
/// no other effect: an unused load is removed, and two loads of one local with no write of a local
/// between them are merged.
///
/// A load of a local that has not been stored yields an unspecified value; `Local2Reg` does not
/// promote such a load. A frontend whose source language zero-initializes its locals stores a
/// zero first.
///
/// Inside a function, the local is one of the function's `locals(..)`, of the type the function
/// gives it.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = BuiltinDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct LoadLocal {
    /// The local read
    #[attr]
    #[effects(MemoryEffect(MemoryEffect::Read), resource(LocalsResource))]
    local: LocalVariableAttr,
    /// The value read
    #[result]
    result: AnyType,
}

/// A result whose type is not yet known gets the local's type. A result that already has a type
/// keeps it, and the verifier checks it. A local bound to a function that has no local of its
/// index is an error.
impl InferTypeOpInterface for LoadLocal {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        if !self.result().ty().is_unknown() {
            return Ok(());
        }
        let local = self.local();
        let ty = if local.as_value().is_bound() {
            let index = local.as_value().as_usize();
            let function = local.as_value().function();
            let function = function.borrow();
            let locals = function.get_locals();
            match locals.iter().nth(index) {
                Some(ty) => *ty,
                None => {
                    return Err(report!(
                        message: "invalid operation builtin.load_local",
                        labels: [primary(
                            self.span(),
                            format!(
                                "this names local {index}, but its function has {} local(s)",
                                locals.len()
                            )
                        )],
                    ));
                }
            }
        } else {
            local.local_type()
        };
        drop(local);
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// The local is one of the enclosing function's, and the result has its type.
impl crate::Verify<dyn MemoryEffectOpInterface> for LoadLocal {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let ty = self.result().ty();
        verify_local(self.as_operation(), &self.local(), ty, "the result")
    }
}

/// A write of `value` to a local variable of the enclosing function:
/// `builtin.store_local %v <{ local = #builtin.local_variable<0, i32> }> : (i32);`.
///
/// The value has the local's type. The store writes `local` in [LocalsResource], so it is never
/// removed by a generic pass, and no read of a local moves or merges across it.
///
/// Inside a function, the local is one of the function's `locals(..)`, of the type the function
/// gives it.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = BuiltinDialect,
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct StoreLocal {
    /// The local written
    #[attr]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(LocalsResource))]
    local: LocalVariableAttr,
    /// The value written
    #[operand]
    value: AnyType,
}

/// The local is one of the enclosing function's, and the value has its type.
impl crate::Verify<dyn MemoryEffectOpInterface> for StoreLocal {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let ty = self.value().ty();
        verify_local(self.as_operation(), &self.local(), ty, "the value")
    }
}

/// Checks the local named by `op`, whose value (`role`) has type `ty`.
///
/// A local built through a function ([Function::alloc_local]) is bound to that function, and must
/// be inside it. Inside a function, the local's index is one of the function's locals, and a local
/// parsed from text (unbound, its type on the attribute) has the type the function gives that
/// local. Outside a function only the type of an unbound local is checked against `ty`.
fn verify_local(
    op: &Operation,
    attr: &LocalVariableAttr,
    ty: Type,
    role: &str,
) -> Result<(), Report> {
    let invalid = |reason: String| {
        if op.is::<LoadLocal>() {
            report!(
                message: "invalid operation builtin.load_local",
                labels: [primary(op.span(), reason)],
            )
        } else {
            report!(
                message: "invalid operation builtin.store_local",
                labels: [primary(op.span(), reason)],
            )
        }
    };

    let local = attr.as_value();
    let index = local.as_usize();
    let function = op.nearest_parent_op::<Function>();
    if local.is_bound()
        && function.is_none_or(|function| {
            function.as_operation_ref() != local.function().as_operation_ref()
        })
    {
        return Err(invalid(format!(
            "local {index} is a local of another function than the one this operation is in"
        )));
    }

    let local_ty = match function {
        Some(function) => {
            let function = function.borrow();
            let locals = function.get_locals();
            if index >= locals.len() {
                return Err(invalid(format!(
                    "this names local {index}, but the function has {} local(s)",
                    locals.len()
                )));
            }
            let declared = locals[index];
            let stated = crate::Attribute::ty(attr);
            if !local.is_bound() && stated != declared {
                return Err(invalid(format!(
                    "this states that local {index} has type `{stated}`, but the function gives \
                     it type `{declared}`"
                )));
            }
            declared
        }
        None => crate::Attribute::ty(attr),
    };
    if ty != local_ty {
        return Err(invalid(format!(
            "{role} has type `{ty}`, but local {index} has type `{local_ty}`"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, string::ToString, vec::Vec};

    use super::*;
    use crate::{
        Context, SourceSpan, UnsafeIntrusiveEntityRef, ValueRef,
        dialects::{
            builtin::{BuiltinOpBuilder, attributes::LocalVariable},
            test::TestMemoryResource,
        },
        effects::{AutomaticAllocationScopeResource, DefaultResource},
        parse::{self, ParserConfig},
        testing::{Test, label_messages, text::parse as parse_text},
    };

    const HEADER: &str = "builtin.function public extern(\"C\")";

    /// Parses `source` (a function) without verifying it, then verifies it: the error, as one line.
    fn verify_error(source: &str) -> alloc::string::String {
        let context = Rc::new(Context::default());
        let config = ParserConfig::new(context.clone()).verify_after_parse(false);
        let function = parse::parse::<Function>(config, "locals", source)
            .into_result()
            .unwrap_or_else(|err| panic!("{source}\n{err:?}"));
        let function = function.borrow();
        let err = function
            .as_operation()
            .recursively_verify()
            .expect_err("the function does not verify");
        let labels = label_messages(err.as_diagnostic()).join(" | ");
        let text = alloc::format!("{} {labels}", err.display());
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// A function built with a local of type `i32`, stored from its argument and loaded back.
    fn built() -> (Test, LocalVariable, UnsafeIntrusiveEntityRef<StoreLocal>, ValueRef) {
        let mut test = Test::new("locals", |cx| ([cx.i32()], [cx.i32()]));
        let i32 = test.context().i32();
        let (local, store, loaded) = {
            let mut builder = test.function_builder();
            let local = builder.alloc_local(i32);
            let argument = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let store = builder.store_local(local, argument, SourceSpan::UNKNOWN).unwrap();
            let loaded = builder.load_local(local, SourceSpan::UNKNOWN).unwrap();
            builder.ret([loaded], SourceSpan::UNKNOWN).unwrap();
            (local, store, loaded)
        };
        (test, local, store, loaded)
    }

    /// A load and a store built through the builder print in the derived form, and the function
    /// parses back to the same text: the parsed locals are unbound, and their type is the one
    /// written on the attribute.
    #[test]
    fn locals_print_and_parse_back() {
        let (test, _local, _store, loaded) = built();
        assert_eq!(loaded.borrow().ty(), test.context().i32());
        let function = test.function();
        let printed = function.borrow().as_operation().to_string();
        let expected = alloc::format!(
            "{HEADER} @locals(%0: i32) -> i32 locals(i32) {{
    builtin.store_local %0 <{{ local = #builtin.local_variable<0, i32> }}> : (i32);
    %1 = builtin.load_local <{{ local = #builtin.local_variable<0, i32> }}>;
    builtin.ret %1 : (i32);
}};"
        );
        assert_eq!(printed, expected);
        function
            .borrow()
            .as_operation()
            .recursively_verify()
            .expect("the built function verifies");

        let (_context, parsed) = parse_text::<Function>(&printed).expect("the text parses");
        assert_eq!(parsed.borrow().as_operation().to_string(), printed);
        let parsed = parsed.borrow();
        let block = parsed.entry_block();
        let block = block.borrow();
        let load = block.body().iter().nth(1).unwrap();
        let load = load.downcast_ref::<LoadLocal>().expect("a load_local");
        assert!(!load.get_local().is_bound());
        assert_eq!(load.local().local_type(), parsed.as_operation().context().i32());
        assert_eq!(load.result().ty(), parsed.as_operation().context().i32());
    }

    /// A local may hold a value of a type that has no data layout: a function value.
    #[test]
    fn a_local_may_hold_a_function_value() {
        let source = alloc::format!(
            "{HEADER} @f(%0: (i32) -> i32) -> ((i32) -> i32) locals((i32) -> i32) {{
    builtin.store_local %0 <{{ local = #builtin.local_variable<0, (i32) -> i32> }}> : ((i32) -> \
             i32);
    %1 = builtin.load_local <{{ local = #builtin.local_variable<0, (i32) -> i32> }}>;
    builtin.ret %1 : ((i32) -> i32);
}};"
        );
        let (_context, function) = parse_text::<Function>(&source).expect("the text parses");
        assert_eq!(function.borrow().as_operation().to_string(), source);
    }

    /// A load reads, and a store writes, its local attribute in [LocalsResource], a child of the
    /// default resource disjoint from memory.
    #[test]
    fn locals_are_read_and_written_in_their_own_resource() {
        let (_test, _local, store, loaded) = built();
        let load = loaded.borrow().get_defining_op().unwrap();
        for (op, effect) in
            [(load, MemoryEffect::Read), (store.as_operation_ref(), MemoryEffect::Write)]
        {
            let op = op.borrow();
            let effects = op.as_trait::<dyn MemoryEffectOpInterface>().unwrap();
            assert!(effects.effects_are_known());
            let effects = effects.effects().collect::<Vec<_>>();
            assert_eq!(effects.len(), 1);
            let instance = &effects[0];
            assert_eq!(*instance.effect(), effect);
            assert!(instance.value().is_none());
            let attribute = instance.attribute().expect("the effect is on the local");
            assert!(attribute.borrow().downcast_ref::<LocalVariableAttr>().is_some());
            assert!(instance.resource().dyn_eq(&LocalsResource));
            assert_eq!(instance.resource().parent().unwrap().name(), DefaultResource.name());
            assert!(!instance.resource().is_disjoint_from(&DefaultResource));
            assert!(instance.resource().is_disjoint_from(&TestMemoryResource));
            assert!(instance.resource().is_disjoint_from(&AutomaticAllocationScopeResource));
        }
        assert!(!store.borrow().as_operation().would_be_trivially_dead());
    }

    /// An unused load is removable; a store is not.
    #[test]
    fn an_unused_load_is_trivially_dead() {
        let mut test = Test::new("dead", |cx| ([cx.i32()], [] as [Type; 0]));
        let i32 = test.context().i32();
        let mut builder = test.function_builder();
        let local = builder.alloc_local(i32);
        let loaded = builder.load_local(local, SourceSpan::UNKNOWN).unwrap();
        builder.ret([], SourceSpan::UNKNOWN).unwrap();
        let load = loaded.borrow().get_defining_op().unwrap();
        assert!(load.borrow().would_be_trivially_dead());
    }

    /// Inside a function, a local is one of the function's locals, of the type it gives it, and the
    /// value read or written has that type.
    #[test]
    fn a_local_that_is_not_the_functions_does_not_verify() {
        let load = |local: &str| {
            alloc::format!(
                "    %1 = builtin.load_local <{{ local = #builtin.local_variable<{local}> }}>;"
            )
        };
        for (locals, body, reason) in [
            (
                " locals(i32)",
                load("1, i32"),
                "this names local 1, but the function has 1 local(s)",
            ),
            ("", load("0, i32"), "this names local 0, but the function has 0 local(s)"),
            (
                " locals(i32)",
                load("0, i64"),
                "this states that local 0 has type `i64`, but the function gives it type `i32`",
            ),
            (
                " locals(i32)",
                "    builtin.store_local %0 <{ local = #builtin.local_variable<0, i32> }> : (i64);"
                    .to_string(),
                "the value has type `i64`, but local 0 has type `i32`",
            ),
            (
                " locals(i32)",
                "    builtin.store_local %0 <{ local = #builtin.local_variable<0, i64> }> : (i64);"
                    .to_string(),
                "this states that local 0 has type `i64`, but the function gives it type `i32`",
            ),
        ] {
            let source =
                alloc::format!("{HEADER} @f(%0: i64){locals} {{\n{body}\n    builtin.ret;\n}};");
            let text = verify_error(&source);
            assert!(text.contains(reason), "{source}\n{text}");
        }
    }

    /// A load whose result (stated in the generic form) has another type than its local does not
    /// verify.
    #[test]
    fn a_load_of_another_type_than_its_local_does_not_verify() {
        let source = alloc::format!(
            "{HEADER} @f() locals(i32) {{
    %1 = \"builtin.load_local\"() <{{ local = #builtin.local_variable<0, i32> }}> : () -> i64;
    builtin.ret;
}};"
        );
        let text = verify_error(&source);
        assert!(text.contains("invalid operation builtin.load_local"), "{source}\n{text}");
        assert!(text.contains("the result has type `i64`, but local 0 has type `i32`"), "{text}");
    }

    /// Outside a function, an unbound local is checked only against the type it states: a store
    /// of another type does not verify, and one of that type does.
    #[test]
    fn an_unbound_local_outside_a_function_is_checked_against_its_type() {
        let module = |ty: &str| {
            alloc::format!(
                "builtin.module public @m {{
    builtin.global_variable public @g : i32 {{
        %0 = test.constant 0 : {ty};
        builtin.store_local %0 <{{ local = #builtin.local_variable<0, i32> }}> : ({ty});
    }};
}};"
            )
        };
        let err = parse_text::<crate::dialects::builtin::Module>(&module("i64"))
            .map(|_| ())
            .expect_err("the value is not of the local's type");
        assert!(err.contains("the value has type `i64`, but local 0 has type `i32`"), "{err}");
        parse_text::<crate::dialects::builtin::Module>(&module("i32"))
            .unwrap_or_else(|err| panic!("{err}"));
    }

    /// The type of a load of a local bound to a function that no longer has a local of its index
    /// is an error, not a panic.
    #[test]
    fn a_load_of_a_bound_local_out_of_range_is_an_error() {
        let mut test = Test::new("shrunk", |_cx| ([] as [Type; 0], [] as [Type; 0]));
        let i32 = test.context().i32();
        let local = test.function().borrow_mut().alloc_local(i32);
        test.function().borrow_mut().get_locals_mut().remove(0);
        let mut builder = test.function_builder();
        let err = builder
            .load_local(local, SourceSpan::UNKNOWN)
            .expect_err("the function has no local 0");
        let labels = label_messages(err.as_diagnostic()).join(" | ");
        assert!(
            labels.contains("this names local 0, but its function has 0 local(s)"),
            "{labels}"
        );
    }

    /// A local built through one function and used in another does not verify.
    #[test]
    fn a_local_of_another_function_does_not_verify() {
        let mut test = Test::named("first").in_module("m");
        let i32 = test.context().i32();
        let module_body = test.module().borrow().body().entry_block_ref().unwrap();
        test.with_function("first", &[i32], &[]);
        let local = test.function().borrow_mut().alloc_local(i32);
        crate::Builder::set_insertion_point_to_end(test.builder_mut(), module_body);
        test.with_function("second", &[i32], &[]);
        {
            let mut builder = test.function_builder();
            let argument = builder.entry_block().borrow().arguments()[0] as ValueRef;
            builder.store_local(local, argument, SourceSpan::UNKNOWN).unwrap();
            builder.ret([], SourceSpan::UNKNOWN).unwrap();
        }
        let function = test.function();
        let function = function.borrow();
        let err = function
            .as_operation()
            .recursively_verify()
            .expect_err("the function does not verify");
        let labels = label_messages(err.as_diagnostic()).join(" | ");
        assert!(
            labels.contains(
                "local 0 is a local of another function than the one this operation is in"
            ),
            "{labels}"
        );
    }
}
