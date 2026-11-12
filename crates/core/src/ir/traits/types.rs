use alloc::{format, string::ToString};
use core::fmt;

use crate::{
    ArrayType, CompactString, ConstantIntRange, Context, IntegerValueRange, Op, Operation, Report,
    Type, ValueRef, derive::operation_trait, ir::value::Value,
};

/// OpInterface to compute the return type(s) of an operation.
pub trait InferTypeOpInterface: Op {
    /// Run type inference for this op's results, using the current state, and apply any changes.
    ///
    /// Returns an error if unable to infer types, or if some type constraint is violated.
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report>;

    /// Return whether the set sets of types are compatible
    fn are_compatible_return_types(&self, lhs: &[Type], rhs: &[Type]) -> bool {
        lhs == rhs
    }
}

/// A value-range constraint implied by an operation operand.
///
/// This describes semantic constraints beyond the raw Miden field element representation. For
/// example, a `u32`-typed operand implies that a raw field element must have been checked to fit
/// in the `u32` range before it reaches that operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueRangeConstraint {
    /// The value must be a valid member of this HIR type's value range.
    Type(Type),
}

impl ValueRangeConstraint {
    /// Return the implicit value-range constraint for `ty`, if values of that type must be more
    /// constrained than an arbitrary field element.
    pub fn from_type(ty: Type) -> Option<Self> {
        if is_unconstrained_value_type(ty) {
            None
        } else {
            Some(Self::Type(ty))
        }
    }

    pub fn ty(&self) -> Type {
        match self {
            Self::Type(ty) => *ty,
        }
    }
}

/// The range requirement imposed on an operation operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandRangeRequirement {
    /// The operand does not need to be range-constrained before being consumed by this operation.
    None,
    /// The operand must satisfy the given range constraint before being consumed.
    Required(ValueRangeConstraint),
}

impl OperandRangeRequirement {
    #[inline]
    pub fn into_constraint(self) -> Option<ValueRangeConstraint> {
        match self {
            Self::None => None,
            Self::Required(constraint) => Some(constraint),
        }
    }
}

/// Return the implicit value-range requirement for `op` operand `operand_index`.
///
/// Operations may override the default type-derived behavior with
/// [OperandRangeRequirementOpInterface]. This is important for checked operations, which consume a
/// value precisely in order to establish a value-range contract.
pub fn operation_operand_range_requirement(
    op: &Operation,
    operand_index: usize,
) -> Option<ValueRangeConstraint> {
    if operand_index >= op.num_operands() {
        return None;
    }

    if let Some(interface) = op.as_trait::<dyn OperandRangeRequirementOpInterface>() {
        return interface.operand_range_requirement(operand_index).into_constraint();
    }

    default_operand_range_requirement(op, operand_index)
}

/// Return the default type-derived range requirement for `op` operand `operand_index`.
pub fn default_operand_range_requirement(
    op: &Operation,
    operand_index: usize,
) -> Option<ValueRangeConstraint> {
    let operand = &op.operands()[operand_index];
    let value = operand.borrow().as_value_ref();
    let value = value.borrow();
    ValueRangeConstraint::from_type(value.ty())
}

/// Return true if values of `ty` are unconstrained by type alone.
pub fn is_unconstrained_value_type(ty: Type) -> bool {
    ty.is_unknown() || ty.is_never()
}

/// OpInterface for operations with operand-specific value-range requirements.
pub trait OperandRangeRequirementOpInterface: Op {
    /// Returns the [OperandRangeRequirement] for the operand at `operand_index`
    fn operand_range_requirement(&self, operand_index: usize) -> OperandRangeRequirement;
}

/// Describes an operation result whose value range was refined from an input value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueRangeRefinement {
    pub input: ValueRef,
    pub result: ValueRef,
    /// The range checked at runtime, expressed at the result's storage width.
    pub range: ConstantIntRange,
}

/// OpInterface for checked cast operations.
///
/// Implementing this interface indicates that the operation result is produced by checking the
/// input value against the output range before materializing the cast result.
pub trait CheckedCastOpInterface: Op {
    /// Return a [ValueRangeRefinement], if possible, for `result`
    fn checked_cast_refinement(&self, result: ValueRef) -> Option<ValueRangeRefinement>;
}

/// OpInterface for assertion operations that refine a value's range.
///
/// Implementing this interface indicates that the operation result is a checked/refined version of
/// one of the operation's input values.
pub trait ValueRangeAssertionOpInterface: Op {
    /// Return a [ValueRangeRefinement], if possible, for `result`
    fn value_range_assertion(&self, result: ValueRef) -> Option<ValueRangeRefinement>;
}

/// OpInterface for operations that can infer integer ranges for their results.
///
/// This is the HIR equivalent of MLIR's `InferIntRangeInterface`. Implementations receive the
/// current operand lattice values and may call `set_result_range` for any result whose range can
/// be inferred. The sparse dataflow analysis owns joining those inferred ranges into the result
/// lattices.
pub trait InferIntRangeInterface: Op {
    fn infer_result_ranges(
        &self,
        operands: &[IntegerValueRange],
        set_result_range: &mut dyn FnMut(ValueRef, IntegerValueRange),
    );
}

/// Return the value-range refinement that produces `result`, if `op` has one.
pub fn operation_result_value_range_refinement(
    op: &Operation,
    result: ValueRef,
) -> Option<ValueRangeRefinement> {
    op.as_trait::<dyn ValueRangeAssertionOpInterface>()
        .and_then(|interface| interface.value_range_assertion(result))
        .or_else(|| {
            op.as_trait::<dyn CheckedCastOpInterface>()
                .and_then(|interface| interface.checked_cast_refinement(result))
        })
}

/// Op expects all operands to be of the same type
#[operation_trait]
pub trait SameTypeOperands {
    #[verifier]
    fn operands_are_the_same_type(op: &Operation, _context: &Context) -> Result<(), Report> {
        let mut operands = op.operands().iter();
        // If there are no operands, then it is trivially true that operands agree on type
        let Some(first_operand) = operands.next() else {
            return Ok(());
        };
        let (expected_ty, set_by) = {
            let operand = first_operand.borrow();
            let value = operand.value();
            (value.ty(), value.span())
        };

        for operand in operands {
            let operand = operand.borrow();
            let value = operand.value();
            let value_ty = value.ty();
            if value_ty != expected_ty {
                return Err(crate::diagnostics::report!(
                    message: (::alloc::format!("invalid operation {}", op.name())),
                    labels: [
                        primary(
                            op.span,
                            "this operation expects all operands to be of the same type"
                        ),
                        context(set_by, "inferred the expected type from this value"),
                        context(value.span(), "which differs from this value's type"),
                    ],
                    notes: [
                        help(format!("expected '{expected_ty}', got '{value_ty}'")),
                    ],
                ));
            }
        }

        Ok(())
    }
}

/// Op expects all operands and results to be of the same type.
///
/// **NOTE:** Operations that implement this trait must also explicitly implement
/// [`SameTypeOperands`]. This can be achieved by using the "traits" field in the [#operation]
/// macro, as shown below:
///
/// ```rust,ignore
/// #[operation (
///     dialect = ArithDialect,
///     traits(UnaryOp, SameTypeOperands, SameOperandsAndResultType),
///     implements(InferTypeOpInterface, MemoryEffectOpInterface)
/// )]
/// pub struct SomeOp {
///     # ...
/// }
/// ```
#[operation_trait]
pub trait SameOperandsAndResultType: SameTypeOperands {
    #[verifier]
    fn operands_and_result_are_the_same_type(
        op: &Operation,
        _context: &Context,
    ) -> Result<(), Report> {
        let mut operands = op.operands().iter();
        // If there are no operands, then it is trivially true that operands and results agree
        // on type
        let Some(first_operand) = operands.next() else {
            return Ok(());
        };
        let (expected_ty, set_by) = {
            let operand = first_operand.borrow();
            let value = operand.value();
            (value.ty(), value.span())
        };

        let results = op.results();
        if results.is_empty() {
            return Err(crate::diagnostics::report!(
                message: (format!("invalid operation result {}", op.name())),
                labels: [
                    primary(
                        op.span,
                        "this operation has operands, so it requires at least one result of the \
                         same type"
                    ),
                    context(set_by, "inferred the expected type from this value"),
                ],
            ));
        }

        for result in results.iter() {
            let result = result.borrow();
            let value = result.as_value_ref().borrow();
            let result_ty = result.ty();

            if result_ty != expected_ty {
                return Err(crate::diagnostics::report!(
                    message: (::alloc::format!("invalid operation result {}", op.name())),
                    labels: [
                        primary(
                            op.span,
                            "this operation expects the operands and the results to be of the \
                             same type"
                        ),
                        context(set_by, "inferred the expected type from this value"),
                        context(value.span(), "which differs from this value's type"),
                    ],
                    notes: [
                        help(format!("expected '{expected_ty}', got '{result_ty}'")),
                    ],
                ));
            }
        }

        Ok(())
    }
}

/// An operation trait that indicates it expects a variable number of operands, matching the given
/// type constraint, i.e. zero or more of the base type.
#[operation_trait]
pub trait Variadic<T: TypeConstraint> {
    #[verifier]
    fn all_operands_match_constraint<T: TypeConstraint>(
        op: &Operation,
        _context: &Context,
    ) -> Result<(), Report> {
        for operand in op.operands().iter() {
            let operand = operand.borrow();
            let value = operand.value();
            let ty = value.ty();
            let constraint = <T as TypeConstraint>::get();
            if constraint.matches(ty) {
                continue;
            } else {
                let description = constraint.description();
                return Err(crate::diagnostics::report!(
                    message: "invalid operand",
                    labels: [
                        primary(
                            value.span(),
                            format!("expected operand type to be {description}, but got {ty}")
                        ),
                    ],
                ));
            }
        }

        Ok(())
    }
}

pub trait TypeConstraint: 'static {
    fn get() -> Self
    where
        Self: Sized;
    fn description(&self) -> CompactString;
    fn matches(&self, ty: crate::Type) -> bool;
}

/// A type that can be constructed as a [crate::Type]
pub trait BuildableTypeConstraint: TypeConstraint {
    fn build(context: &Context) -> crate::Type;
}

macro_rules! type_constraint {
    ($Constraint:ident, $description:literal, $matcher:literal) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub struct $Constraint;
        impl TypeConstraint for $Constraint {
            #[inline(always)]
            fn get() -> Self {
                Self
            }

            #[inline(always)]
            fn description(&self) -> $crate::CompactString {
                $crate::CompactString::const_new($description)
            }

            #[inline]
            fn matches(&self, _ty: $crate::Type) -> bool {
                $matcher
            }
        }
    };

    ($Constraint:ident, $description:literal, $matcher:path) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub struct $Constraint;
        impl TypeConstraint for $Constraint {
            #[inline(always)]
            fn get() -> Self {
                Self
            }

            #[inline(always)]
            fn description(&self) -> $crate::CompactString {
                $crate::CompactString::const_new($description)
            }

            #[inline]
            fn matches(&self, ty: $crate::Type) -> bool {
                $matcher(&ty)
            }
        }
    };

    ($Constraint:ident, $description:literal, |$matcher_input:ident| $matcher:expr) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub struct $Constraint;
        impl TypeConstraint for $Constraint {
            #[inline(always)]
            fn get() -> Self {
                Self
            }

            #[inline(always)]
            fn description(&self) -> $crate::CompactString {
                $crate::CompactString::const_new($description)
            }

            #[inline]
            fn matches(&self, $matcher_input: $crate::Type) -> bool {
                $matcher
            }
        }
    };
}

type_constraint!(AnyType, "any type", true);

// TODO(pauls): Extend Type with new Function variant, we'll use that to represent function handles
//type_constraint!(AnyFunction, "a function type", crate::Type::is_function);

type_constraint!(AnyArray, "any array type", crate::Type::is_array);

type_constraint!(AnyStruct, "any struct type", crate::Type::is_struct);

type_constraint!(AnyPointer, "a pointer type", crate::Type::is_pointer);

type_constraint!(AnyInteger, "an integral type", crate::Type::is_integer);

type_constraint!(AnyPointerOrInteger, "an integral or pointer type", |ty| ty.is_pointer()
    || ty.is_integer());

/// A boolean
pub type Bool = SizedInt<1>;

/// A signless 8-bit integer
pub type Int8 = SizedInt<8>;

/// A signless 16-bit integer
pub type Int16 = SizedInt<16>;

/// A signless 32-bit integer
pub type Int32 = SizedInt<32>;

/// A signless 64-bit integer
pub type Int64 = SizedInt<64>;

/// A signless 128-bit integer
pub type Int128 = SizedInt<128>;

/// Represents a fixed-width integer of `N` bits
pub struct SizedInt<const N: usize>(core::marker::PhantomData<[(); N]>);

impl<const N: usize> Copy for SizedInt<N> {}

impl<const N: usize> Clone for SizedInt<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const N: usize> fmt::Debug for SizedInt<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(core::any::type_name::<Self>())
    }
}

impl<const N: usize> fmt::Display for SizedInt<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{N}-bit integral type")
    }
}

impl<const N: usize> TypeConstraint for SizedInt<N> {
    #[inline(always)]
    fn get() -> Self {
        Self(core::marker::PhantomData)
    }

    fn description(&self) -> CompactString {
        CompactString::from(self.to_string())
    }

    fn matches(&self, ty: crate::Type) -> bool {
        ty.integer_width().is_some_and(|width| width as usize == N)
    }
}

impl<const N: usize> BuildableTypeConstraint for SizedInt<N> {
    fn build(context: &Context) -> crate::Type {
        let width = u32::try_from(N).expect("an integer width fits in 32 bits");
        context.int_type(width)
    }
}

/// A type constraint for pointer values
pub struct PointerOf<T>(core::marker::PhantomData<T>);

impl<T> Copy for PointerOf<T> {}

impl<T> Clone for PointerOf<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for PointerOf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(core::any::type_name::<Self>())
    }
}

impl<T: TypeConstraint> fmt::Display for PointerOf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pointee = <T as TypeConstraint>::get().description();
        write!(f, "a pointer to {pointee}")
    }
}

impl<T: TypeConstraint> TypeConstraint for PointerOf<T> {
    #[inline(always)]
    fn get() -> Self {
        Self(core::marker::PhantomData)
    }

    fn description(&self) -> CompactString {
        CompactString::from(self.to_string())
    }

    fn matches(&self, ty: crate::Type) -> bool {
        ty.pointee()
            .is_some_and(|pointee| <T as TypeConstraint>::get().matches(pointee))
    }
}

impl<T: BuildableTypeConstraint> BuildableTypeConstraint for PointerOf<T> {
    fn build(context: &Context) -> crate::Type {
        context.ptr_type(<T as BuildableTypeConstraint>::build(context))
    }
}

/// A type constraint for array values
pub struct AnyArrayOf<T>(core::marker::PhantomData<T>);

impl<T> Copy for AnyArrayOf<T> {}

impl<T> Clone for AnyArrayOf<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for AnyArrayOf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(core::any::type_name::<Self>())
    }
}

impl<T: TypeConstraint> fmt::Display for AnyArrayOf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let element = <T as TypeConstraint>::get().description();
        write!(f, "an array of {element}")
    }
}

impl<T: TypeConstraint> TypeConstraint for AnyArrayOf<T> {
    #[inline(always)]
    fn get() -> Self {
        Self(core::marker::PhantomData)
    }

    fn description(&self) -> CompactString {
        CompactString::from(self.to_string())
    }

    fn matches(&self, ty: crate::Type) -> bool {
        ty.downcast_ref::<ArrayType>()
            .is_some_and(|array| <T as TypeConstraint>::get().matches(array.element()))
    }
}

/// A type constraint for array values
pub struct ArrayOf<const N: usize, T>(core::marker::PhantomData<[T; N]>);

impl<const N: usize, T> Copy for ArrayOf<N, T> {}

impl<const N: usize, T> Clone for ArrayOf<N, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const N: usize, T> fmt::Debug for ArrayOf<N, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(core::any::type_name::<Self>())
    }
}

impl<const N: usize, T: TypeConstraint> fmt::Display for ArrayOf<N, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let element = <T as TypeConstraint>::get().description();
        write!(f, "an array of {N} {element}")
    }
}

impl<const N: usize, T: TypeConstraint> TypeConstraint for ArrayOf<N, T> {
    #[inline(always)]
    fn get() -> Self {
        Self(core::marker::PhantomData)
    }

    fn description(&self) -> CompactString {
        CompactString::from(self.to_string())
    }

    fn matches(&self, ty: crate::Type) -> bool {
        ty.downcast_ref::<ArrayType>().is_some_and(|array| {
            array.len() == N as u64 && <T as TypeConstraint>::get().matches(array.element())
        })
    }
}

impl<const N: usize, T: BuildableTypeConstraint> BuildableTypeConstraint for ArrayOf<N, T> {
    fn build(context: &Context) -> crate::Type {
        let element = <T as BuildableTypeConstraint>::build(context);
        context.array_type(element, N as u64)
    }
}

/// Represents a conjunction of two constraints as a concrete value
pub struct And<T, U> {
    _left: core::marker::PhantomData<T>,
    _right: core::marker::PhantomData<U>,
}

impl<T, U> Copy for And<T, U> {}

impl<T, U> Clone for And<T, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, U> fmt::Debug for And<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(core::any::type_name::<Self>())
    }
}

impl<T: TypeConstraint, U: TypeConstraint> TypeConstraint for And<T, U> {
    #[inline(always)]
    fn get() -> Self {
        Self {
            _left: core::marker::PhantomData,
            _right: core::marker::PhantomData,
        }
    }

    fn description(&self) -> CompactString {
        let left = <T as TypeConstraint>::get().description();
        let right = <U as TypeConstraint>::get().description();
        CompactString::from(format!("both {left} and {right}"))
    }

    #[inline]
    fn matches(&self, ty: crate::Type) -> bool {
        <T as TypeConstraint>::get().matches(ty) && <U as TypeConstraint>::get().matches(ty)
    }
}

/// Represents a disjunction of two constraints as a concrete value
pub struct Or<T, U> {
    _left: core::marker::PhantomData<T>,
    _right: core::marker::PhantomData<U>,
}

impl<T, U> Copy for Or<T, U> {}

impl<T, U> Clone for Or<T, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, U> fmt::Debug for Or<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(core::any::type_name::<Self>())
    }
}

impl<T: TypeConstraint, U: TypeConstraint> TypeConstraint for Or<T, U> {
    #[inline(always)]
    fn get() -> Self {
        Self {
            _left: core::marker::PhantomData,
            _right: core::marker::PhantomData,
        }
    }

    fn description(&self) -> CompactString {
        let left = <T as TypeConstraint>::get().description();
        let right = <U as TypeConstraint>::get().description();
        CompactString::from(format!("either {left} or {right}"))
    }

    #[inline]
    fn matches(&self, ty: crate::Type) -> bool {
        <T as TypeConstraint>::get().matches(ty) || <U as TypeConstraint>::get().matches(ty)
    }
}

#[cfg(test)]
mod tests {
    use super::{Bool, Int32, TypeConstraint};
    use crate::Type;

    #[test]
    fn sized_integer_constraints_require_the_declared_width() {
        let boolean = Bool::get();
        assert!(boolean.matches(&Type::I1));
        assert!(!boolean.matches(&Type::U32));

        let int32 = Int32::get();
        assert!(int32.matches(&Type::I32));
        assert!(int32.matches(&Type::U32));
        assert!(!int32.matches(&Type::I1));
    }
}
