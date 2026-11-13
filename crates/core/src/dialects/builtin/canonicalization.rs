//! The canonicalization patterns of the builtin dialect.

use alloc::rc::Rc;

use super::{BuiltinDialect, BuiltinOpBuilder, CallIndirect, FunctionReference};
use crate::{
    Context, OperationRef, Report, SmallVec, Spanned, ValueRef,
    dialects::builtin::attributes::SymbolRefAttr,
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind, RewritePattern, Rewriter},
};

/// Turns a `builtin.call_indirect` whose callee is defined by a `builtin.function_ref` into a
/// `builtin.call` of the symbol the reference names, by the path the reference names it with, with
/// the same arguments, signature, span and attribute dictionary.
///
/// The rewrite is made only when the signature of the callable the symbol resolves to equals the
/// call's, calling convention and parameter attributes included; otherwise the call is left alone.
/// The reference itself is left in place: once unused, it is removed as any dead operation is.
pub struct CallIndirectOfFunctionRef {
    info: PatternInfo,
}

impl CallIndirectOfFunctionRef {
    /// Creates the pattern, matching `builtin.call_indirect`.
    pub fn new(context: Rc<Context>) -> Self {
        let call_indirect = context
            .get_or_register_dialect::<BuiltinDialect>()
            .expect_registered_name::<CallIndirect>();
        Self {
            info: PatternInfo::new(
                context,
                "call-indirect-of-function-ref",
                PatternKind::Operation(call_indirect),
                PatternBenefit::MAX,
            ),
        }
    }
}

impl Pattern for CallIndirectOfFunctionRef {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl RewritePattern for CallIndirectOfFunctionRef {
    fn match_and_rewrite(
        &self,
        operation: OperationRef,
        rewriter: &mut dyn Rewriter,
    ) -> Result<bool, Report> {
        let (callee, path, signature, arguments, span, attributes) = {
            let op = operation.borrow();
            let Some(call) = op.downcast_ref::<CallIndirect>() else {
                return Ok(false);
            };
            let Some(defining_op) = call.callee().as_value_ref().borrow().get_defining_op() else {
                return Ok(false);
            };
            let defining_op = defining_op.borrow();
            let Some(reference) = defining_op.downcast_ref::<FunctionReference>() else {
                return Ok(false);
            };
            let Ok(resolved) = reference.callee().resolve_callable() else {
                return Ok(false);
            };
            let signature = call.get_signature().clone();
            if resolved.signature() != signature {
                return Ok(false);
            }
            let arguments = call
                .arguments()
                .iter()
                .map(|operand| operand.borrow().as_value_ref())
                .collect::<SmallVec<[ValueRef; 4]>>();
            let path = reference.callee().path().clone();
            // A symbol reference in the dictionary gets a use of its own on the new call; the
            // indirect call's use goes with it when it is erased
            let attributes = op
                .attributes()
                .iter()
                .map(|attr| {
                    let attr = *attr.as_named_attribute();
                    let symbol = attr.value.try_downcast_attr::<SymbolRefAttr>().ok().and_then(
                        |reference| {
                            let reference = reference.borrow();
                            Some((reference.resolve()?, reference.path().clone()))
                        },
                    );
                    (attr, symbol)
                })
                .collect::<SmallVec<[_; 2]>>();
            (resolved.named_symbol(), path, signature, arguments, op.span(), attributes)
        };

        rewriter.set_insertion_point_before(operation);
        let mut call = rewriter.call(callee, signature, arguments, span)?;
        // The builder names the callee by its absolute path. The reference and the call are in one
        // function, so the path the reference was written with names the same symbol from the
        // call; keeping it keeps the text, and calls that CSE compares by path, as written.
        call.borrow_mut().callee_mut().set_path(path);
        {
            let mut call = call.borrow_mut();
            let call = crate::Op::as_operation_mut(&mut *call);
            for (attr, symbol) in attributes {
                let Some((symbol, path)) = symbol else {
                    call.set_attribute(attr.name, attr.value);
                    continue;
                };
                // Linked to the same symbol, by the path it was written with, as the callee is
                call.set_symbol_attribute(attr.name, symbol);
                let reference = call.get_attribute(attr.name).unwrap();
                let mut reference = reference.try_downcast_attr::<SymbolRefAttr>().ok().unwrap();
                reference.borrow_mut().set_path(path);
            }
        }
        rewriter.replace_op(operation, call.as_operation_ref());
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        format,
        string::{String, ToString},
        vec::Vec,
    };

    use super::*;
    use crate::{
        CallOpInterface, Operation, SymbolTable,
        dialects::builtin::{Call, Module},
        parse::{self, ParserConfig},
        patterns::RewriterImpl,
    };

    /// Parses a module holding `declarations` and `@caller(%a: i32, %b: i32) -> i32`, which calls
    /// `@callee` through a `builtin.function_ref` with `signature`; applies the pattern to the
    /// indirect call, and returns whether it applied, and the module with the context that owns it.
    fn rewrite(
        declarations: &str,
        signature: &str,
    ) -> Result<(bool, OperationRef, Rc<Context>), Report> {
        let context = Rc::new(Context::default());
        let source = format!(
            "builtin.module public @m {{
    {declarations}
    builtin.function public extern(\"C\") @caller(%a: i32, %b: i32) -> i32 {{
        %f = builtin.function_ref @callee : (i32, i32) -> i32;
        %r = builtin.call_indirect %f(%a, %b) : {signature};
        builtin.ret %r : (i32);
    }};
}};"
        );
        let module = parse::parse::<Module>(ParserConfig::new(context.clone()), "indirect", source)
            .into_result()?;
        let module = module.as_operation_ref();
        let mut call = None;
        module.borrow().prewalk_all(|op: &Operation| {
            if op.is::<CallIndirect>() {
                call = Some(op.as_operation_ref());
            }
        });
        let mut rewriter: RewriterImpl = RewriterImpl::new(context.clone());
        let applied = CallIndirectOfFunctionRef::new(context.clone())
            .match_and_rewrite(call.unwrap(), &mut rewriter)?;
        module.borrow().recursively_verify()?;
        Ok((applied, module, context))
    }

    /// The text of `@caller` in `module`.
    fn caller_text(module: OperationRef) -> String {
        let module = module.try_downcast_op::<Module>().unwrap();
        let caller = module.borrow().symbol_manager().lookup_op("caller").unwrap();
        caller.borrow().to_string()
    }

    /// The indirect call becomes a call with its arguments and signature; the reference, now
    /// unused, is left in place.
    #[test]
    fn an_indirect_call_of_a_function_ref_becomes_a_direct_call() -> Result<(), Report> {
        let (applied, module, _context) = rewrite(
            r#"builtin.function public extern("C") @callee(i32, i32) -> i32;"#,
            r#"extern("C") (i32, i32) -> i32"#,
        )?;
        assert!(applied);
        let text = caller_text(module);
        assert!(!text.contains("builtin.call_indirect"), "{text}");
        assert!(text.contains("builtin.function_ref @callee : (i32, i32) -> i32;"), "{text}");
        let lines = text.lines().map(str::trim).collect::<Vec<_>>();
        let call = lines.iter().find(|line| line.contains("builtin.call ")).expect(&text);
        assert!(
            call.ends_with(r#"= builtin.call @callee(%0, %1) : extern("C") (i32, i32) -> i32;"#),
            "{text}"
        );
        Ok(())
    }

    /// The new call names the symbol the reference names, an alias here, not the alias's target.
    #[test]
    fn the_direct_call_names_the_symbol_the_reference_names() -> Result<(), Report> {
        let (applied, module, _context) = rewrite(
            r#"builtin.function public extern("C") @target(i32, i32) -> i32;
    builtin.function_alias public @callee -> @target;"#,
            r#"extern("C") (i32, i32) -> i32"#,
        )?;
        assert!(applied);
        let mut call = None;
        module.borrow().prewalk_all(|op: &Operation| {
            if op.is::<Call>() {
                call = Some(op.as_operation_ref());
            }
        });
        let call = call.unwrap().try_downcast_op::<Call>().unwrap();
        let call = call.borrow();
        assert_eq!(call.resolve_stored_callee().unwrap().borrow().name().as_str(), "callee");
        assert_eq!(
            call.resolve().unwrap().as_function().unwrap().borrow().name().as_str(),
            "target"
        );
        Ok(())
    }

    /// A callee whose calling convention differs from the call's keeps the indirect call.
    #[test]
    fn an_indirect_call_whose_signature_differs_from_the_callees_is_kept() -> Result<(), Report> {
        let (applied, module, _context) = rewrite(
            r#"builtin.function public extern("fast") @callee(i32, i32) -> i32;"#,
            r#"extern("C") (i32, i32) -> i32"#,
        )?;
        assert!(!applied);
        let text = caller_text(module);
        assert!(
            text.contains(r#"builtin.call_indirect %2(%0, %1) : extern("C") (i32, i32) -> i32;"#),
            "{text}"
        );
        assert!(!text.contains("builtin.call "), "{text}");
        Ok(())
    }

    /// The new call keeps the indirect call's attribute dictionary. A symbol reference in it is a
    /// use of its own on the new call, so it still resolves once the indirect call is erased.
    #[test]
    fn the_rewrite_keeps_the_attribute_dictionary() -> Result<(), Report> {
        let (applied, module, _context) = rewrite(
            r#"builtin.function public extern("C") @callee(i32, i32) -> i32;
    builtin.function public extern("C") @other();"#,
            r#"extern("C") (i32, i32) -> i32 attributes { note = #builtin.i32<7>, target = #builtin.symbol<@other> }"#,
        )?;
        assert!(applied);
        let text = caller_text(module);
        let lines = text.lines().map(str::trim).collect::<Vec<_>>();
        let call = lines.iter().find(|line| line.contains("builtin.call ")).expect(&text);
        assert!(call.contains("note = #builtin.i32<7>"), "{text}");
        assert!(call.contains("target = #builtin.symbol<@other>"), "{text}");

        let mut call = None;
        module.borrow().prewalk_all(|op: &Operation| {
            if op.is::<Call>() {
                call = Some(op.as_operation_ref());
            }
        });
        let call = call.unwrap();
        let target = call.borrow().get_attribute("target").unwrap();
        let target = target.try_downcast_attr::<SymbolRefAttr>().ok().unwrap();
        let other = target.borrow().resolve().expect("the copied reference resolves");
        assert_eq!(other.borrow().name().as_str(), "other");
        assert_eq!(other.borrow().iter_uses().count(), 1, "the erased call's use is gone");
        Ok(())
    }
}
