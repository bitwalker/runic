//! The declarative macros that define the repeated shapes of the numeric operations.
//!
//! Each macro defines one operation: a struct with its doc comment, the `#[operation]`
//! attribute, the derived text form, the effect-free memory interface and its type inference.
//! Every operation of the numeric families is free of effects (a trap is not an effect).

/// Defines `$Op`, printed `wasm.$name`, a binary operation of two operands of one type `$C` and
/// a result of that type. The text form is `%r = wasm.$name %a, %b;`, with no type.
macro_rules! same_type_binary {
    (
        $(#[$doc:meta])*
        $Op:ident, $name:literal, $C:ty, [$($trait:ident),*]
    ) => { paste::paste! {
        $(#[$doc])*
        #[derive(
            EffectOpInterface,
            OpPrinter,
            OpParser
        )]
        #[operation(
            dialect = WasmDialect,
            name = $name,
            traits(BinaryOp $(, $trait)*),
            implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
        )]
        pub struct [<$Op>] {
            #[operand]
            lhs: AnyType,
            #[operand]
            rhs: AnyType,
            #[result]
            result: AnyType,
        }

        impl InferTypeOpInterface for $Op {
            fn infer_return_types(
                &mut self,
                _context: &Context,
            ) -> Result<(), Report> {
                let ty = self.lhs().ty();
                self.result_mut().set_type(ty);
                Ok(())
            }
        }

        impl Verify<dyn InferTypeOpInterface> for $Op {
            fn verify(&self, _context: &Context) -> Result<(), Report> {
                $crate::ops::verify_uniform::<$C>(self.as_operation(), $crate::ops::ResultRule::SameType)
            }
        }
    } };
}

/// Defines `$Op`, printed `wasm.$name`, a unary operation of one operand of type `$C` and a result
/// of that type. The text form is `%r = wasm.$name %a;`, with no type.
macro_rules! same_type_unary {
    (
        $(#[$doc:meta])*
        $Op:ident, $name:literal, $C:ty, [$($trait:ident),*]
    ) => { paste::paste! {
        $(#[$doc])*
        #[derive(
            EffectOpInterface,
            OpPrinter,
            OpParser
        )]
        #[operation(
            dialect = WasmDialect,
            name = $name,
            traits(UnaryOp $(, $trait)*),
            implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
        )]
        pub struct [<$Op>] {
            #[operand]
            operand: AnyType,
            #[result]
            result: AnyType,
        }

        impl InferTypeOpInterface for $Op {
            fn infer_return_types(
                &mut self,
                _context: &Context,
            ) -> Result<(), Report> {
                let ty = self.operand().ty();
                self.result_mut().set_type(ty);
                Ok(())
            }
        }

        impl Verify<dyn InferTypeOpInterface> for $Op {
            fn verify(&self, _context: &Context) -> Result<(), Report> {
                $crate::ops::verify_uniform::<$C>(self.as_operation(), $crate::ops::ResultRule::SameType)
            }
        }
    } };
}

/// Defines `$Op`, printed `wasm.$name`, a comparison of two operands of one type `$C` whose result
/// is an `i32`: 1 for true, 0 for false. The text form is `%r = wasm.$name %a, %b;`.
macro_rules! compare_binary {
    (
        $(#[$doc:meta])*
        $Op:ident, $name:literal, $C:ty, [$($trait:ident),*]
    ) => { paste::paste! {
        $(#[$doc])*
        #[derive(
            EffectOpInterface,
            OpPrinter,
            OpParser
        )]
        #[operation(
            dialect = WasmDialect,
            name = $name,
            traits(BinaryOp $(, $trait)*),
            implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
        )]
        pub struct [<$Op>] {
            #[operand]
            lhs: AnyType,
            #[operand]
            rhs: AnyType,
            #[result]
            result: AnyType,
        }

        impl InferTypeOpInterface for $Op {
            fn infer_return_types(
                &mut self,
                context: &Context,
            ) -> Result<(), Report> {
                self.result_mut().set_type(context.i32());
                Ok(())
            }
        }

        impl Verify<dyn InferTypeOpInterface> for $Op {
            fn verify(&self, _context: &Context) -> Result<(), Report> {
                $crate::ops::verify_uniform::<$C>(self.as_operation(), $crate::ops::ResultRule::I32)
            }
        }
    } };
}

/// Defines `$Op`, printed `wasm.$name`, a conversion from an operand of type `$From` to a result
/// of type `$To`, which the text writes as the property `ty`: `%r = wasm.$name %a <{ ty = i64 }>;`.
///
/// `$rule` is a `fn(&Context, Type, Type) -> bool` that says which pairs the instruction has, and
/// `$pairs` says them in words for the message of a pair it does not have.
macro_rules! conversion {
    (
        $(#[$doc:meta])*
        $Op:ident, $name:literal, $From:ty => $To:ty, $rule:expr, $pairs:literal
    ) => { paste::paste! {
        $(#[$doc])*
        #[derive(
            EffectOpInterface,
            OpPrinter,
            OpParser
        )]
        #[operation(
            dialect = WasmDialect,
            name = $name,
            traits(UnaryOp),
            implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
        )]
        pub struct [<$Op>] {
            #[operand]
            operand: AnyType,
            #[attr(hidden)]
            ty: TypeAttr,
            #[result]
            result: AnyType,
        }

        impl InferTypeOpInterface for $Op {
            fn infer_return_types(
                &mut self,
                _context: &Context,
            ) -> Result<(), Report> {
                let ty = *self.get_ty();
                self.result_mut().set_type(ty);
                Ok(())
            }
        }

        impl Verify<dyn InferTypeOpInterface> for $Op {
            fn verify(&self, context: &Context) -> Result<(), Report> {
                $crate::ops::verify_conversion::<$From, $To>(
                    self.as_operation(),
                    context,
                    *self.get_ty(),
                    $rule,
                    $pairs,
                )
            }
        }
    } };
}
