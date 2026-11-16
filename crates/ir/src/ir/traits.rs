mod canonicalization;
mod foldable;
mod info;
mod types;

use alloc::format;

use smallvec::SmallVec;

pub(crate) use self::info::assert_trait_table_built_for;
pub use self::{
    canonicalization::Canonicalizable,
    foldable::{FoldResult, Foldable, OpFoldResult},
    info::TraitInfo,
    types::*,
};
use super::BlockRef;
use crate::{
    AttributeRef, Context, Operation,
    derive::operation_trait,
    diagnostics::{Report, Spanned},
};

/// Marker trait for commutative ops, e.g. `X op Y == Y op X`
#[operation_trait]
pub trait Commutative {}

/// Marker trait for constant-like ops
#[operation_trait]
pub trait ConstantLike {}

/// Marker trait for return-like ops
#[operation_trait]
pub trait ReturnLike {}

/// Marker trait for tail-call-like ops: a terminator without successors that leaves the enclosing
/// callable by calling another one.
///
/// The caller's frame does not exist while the callee runs, and what the callee returns is what
/// the enclosing callable returns. This is the contrast with [ReturnLike]: the operands of a
/// return-like operation are the values the enclosing callable returns, whereas the operands of a
/// tail-call-like operation are the arguments of the call it makes. An analysis that reads the
/// operands of a [ReturnLike] operation as returned values must not read those of a tail call
/// that way, which is why a tail call is never [ReturnLike].
#[operation_trait]
pub trait TailCallLike {}

/// Op is a terminator (i.e. it can be used to terminate a block)
#[operation_trait]
pub trait Terminator {}

/// Op's regions do not require blocks to end with a [Terminator]
#[operation_trait]
pub trait NoTerminator {}

/// Marker trait for idemptoent ops, i.e. `op op X == op X (unary) / X op X == X (binary)`
#[operation_trait]
pub trait Idempotent {}

/// Marker trait for ops that exhibit the property `op op X == X`
#[operation_trait]
pub trait Involution {}

/// Marker trait for ops which are not permitted to access values defined above them
#[operation_trait]
pub trait IsolatedFromAbove {}

/// Marker trait for ops which have only regions of [`crate::RegionKind::Graph`]
#[operation_trait]
pub trait HasOnlyGraphRegion {}

/// Op's regions are all single-block graph regions, that not require a terminator
///
/// This trait _cannot_ be derived via `derive!`
#[operation_trait]
pub trait GraphRegionNoTerminator:
    NoTerminator + SingleBlock + crate::RegionKindInterface + HasOnlyGraphRegion
{
}

/// This interface provides information for branching terminator operations, i.e. terminator
/// operations with successors.
///
/// This interface is meant to model well-defined cases of control-flow of value propagation, where
/// what occurs along control-flow edges is assumed to be side-effect free. For example,
/// corresponding successor operands and successor block arguments may have different types. In such
/// cases, `are_types_compatible` can be implemented to compare types along control-flow edges. By
/// default, type equality is used.
///
/// # Verification
///
/// The verifier checks, for each successor, that the operands [Self::get_successor_operands]
/// states for it (produced and forwarded) are as many as the successor block's arguments, and that
/// each forwarded operand's type is compatible, by [Self::are_types_compatible], with the type of
/// the block argument it binds. A produced operand has no value to compare, so only its position
/// is counted.
pub trait BranchOpInterface: crate::Op {
    /// Returns the operands that correspond to the arguments of the successor at `index`.
    ///
    /// It consists of a number of operands that are internally produced by the operation, followed
    /// by a range of operands that are forwarded. An example operation making use of produced
    /// operands would be:
    ///
    /// ```text,ignore
    /// invoke %function(%0)
    ///     label ^success ^error(%1 : i32)
    ///
    /// ^error(%e: !error, %arg0: i32):
    ///     ...
    ///```
    ///
    /// The operand that would map to the `^error`s `%e` operand is produced by the `invoke`
    /// operation, while `%1` is a forwarded operand that maps to `%arg0` in the successor.
    ///
    /// Produced operands always map to the first few block arguments of the successor, followed by
    /// the forwarded operands. Mapping them in any other order is not supported by the interface.
    ///
    /// By having the forwarded operands last allows users of the interface to append more forwarded
    /// operands to the branch operation without interfering with other successor operands.
    fn get_successor_operands(&self, index: usize) -> crate::SuccessorOperandRange<'_> {
        let op = <Self as crate::Op>::as_operation(self);
        let operand_group = op.successors()[index].operand_group as usize;
        crate::SuccessorOperandRange::forward(op.operands().group(operand_group))
    }
    /// The mutable version of [Self::get_successor_operands].
    fn get_successor_operands_mut(&mut self, index: usize) -> crate::SuccessorOperandRangeMut<'_> {
        let op = <Self as crate::Op>::as_operation_mut(self);
        let operand_group = op.successors()[index].operand_group as usize;
        crate::SuccessorOperandRangeMut::forward(op.operands_mut().group_mut(operand_group))
    }
    /// Returns the block argument of the successor corresponding to the operand at `operand_index`.
    ///
    /// Returns `None` if the specified operand is not a successor operand.
    fn get_successor_block_argument(
        &self,
        operand_index: usize,
    ) -> Option<crate::BlockArgumentRef> {
        let op = <Self as crate::Op>::as_operation(self);
        let operand_groups = op.operands().num_groups();
        let mut next_index = 0usize;
        for operand_group in 0..operand_groups {
            let group_size = op.operands().group(operand_group).len();
            if (next_index..(next_index + group_size)).contains(&operand_index) {
                let arg_index = operand_index - next_index;
                // We found the operand group, now map that to a successor
                let succ_info =
                    op.successors().iter().find(|s| operand_group == s.operand_group as usize)?;
                return succ_info
                    .block
                    .borrow()
                    .successor()
                    .borrow()
                    .arguments()
                    .get(arg_index)
                    .cloned();
            }

            next_index += group_size;
        }

        None
    }
    /// Returns the successor that would be chosen with the given constant operands.
    ///
    /// Each operand of this op has an entry in the `operands` slice. If the operand is non-constant,
    /// the corresponding entry will be `None`.
    ///
    /// Returns `None` if a single successor could not be chosen.
    #[inline]
    #[allow(unused_variables)]
    fn get_successor_for_operands(
        &self,
        operands: &[Option<AttributeRef>],
    ) -> Option<crate::SuccessorInfo> {
        None
    }
    /// This is called to compare types along control-flow edges.
    ///
    /// By default, types must be exactly equal to be compatible.
    fn are_types_compatible(&self, lhs: crate::Type, rhs: crate::Type) -> bool {
        lhs == rhs
    }

    /// Changes the destination to `new_dest` if the current destination is `old_dest`.
    fn change_branch_destination(&mut self, old_dest: BlockRef, new_dest: BlockRef) {
        let op = <Self as crate::Op>::as_operation_mut(self);
        assert_eq!(old_dest.borrow().num_arguments(), new_dest.borrow().num_arguments());
        for successor_info in op.successors_mut().iter_mut() {
            if successor_info.successor() == old_dest {
                successor_info.block.borrow_mut().set(new_dest);
            }
        }
    }
}

impl<T> crate::Verify<dyn BranchOpInterface> for T
where
    T: crate::Op + BranchOpInterface,
{
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        verify_successor_keys_are_distinct(self.as_operation())?;
        verify_branch_successor_operands(self)
    }
}

/// Check that no two keyed successors of `op` have the same key: a key selects one successor
/// (see [crate::KeyedSuccessor]). Keys are compared as attributes, so two spellings of one value
/// are one key.
fn verify_successor_keys_are_distinct(op: &Operation) -> Result<(), Report> {
    struct Key(AttributeRef);
    impl Eq for Key {}
    impl PartialEq for Key {
        fn eq(&self, other: &Self) -> bool {
            self.0.borrow().dyn_eq(&other.0.borrow())
        }
    }
    impl core::hash::Hash for Key {
        fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
            self.0.borrow().hash(state);
        }
    }

    let mut seen = crate::FxHashMap::<Key, usize>::default();
    for (index, successor) in op.successors().all().iter().enumerate() {
        let Some(key) = successor.key else {
            continue;
        };
        if let Some(earlier) = seen.insert(Key(key), index) {
            let flags = Default::default();
            let mut printer = crate::print::AsmPrinter::root(op.context_rc(), &flags);
            printer.print_attribute_value(&*key.borrow());
            let key = printer.finish();
            let name = op.name();
            return Err(crate::diagnostics::report!(
                message: (format!(
                    "invalid operation {name}: successor {index} has the key {key}, which \
                     successor {earlier} has too"
                )),
                labels: [
                    primary(op.span(), format!("the key {key} selects two successors")),
                ],
            ));
        }
    }
    Ok(())
}

/// Check every successor of `branch` against the arguments of the block it goes to.
fn verify_branch_successor_operands(branch: &dyn BranchOpInterface) -> Result<(), Report> {
    let op = branch.as_operation();
    for index in 0..op.num_successors() {
        let block = op.successor(index).successor();
        let block = block.borrow();
        let arguments = block.argument_types().collect::<SmallVec<[_; 4]>>();
        verify_successor_operands(
            op,
            &format!("successor {index} ({block})"),
            &branch.get_successor_operands(index),
            "block argument",
            &arguments,
            |lhs, rhs| branch.are_types_compatible(lhs, rhs),
        )?;
    }
    Ok(())
}

/// Check the operands `op` states for one control-flow edge against the types of the values they
/// bind at its destination, `inputs`: there are as many operands as inputs, and each forwarded
/// operand's type is compatible with its input's type by `are_types_compatible`.
///
/// `successor` names the edge in the diagnostic, and `input_kind` names what an input is there (a
/// block argument, or a result).
pub(crate) fn verify_successor_operands(
    op: &Operation,
    successor: &str,
    operands: &crate::SuccessorOperandRange<'_>,
    input_kind: &str,
    inputs: &[crate::Type],
    are_types_compatible: impl Fn(crate::Type, crate::Type) -> bool,
) -> Result<(), Report> {
    use crate::SuccessorOperands;

    let name = op.name();
    let num_produced = operands.num_produced();
    if operands.len() != inputs.len() {
        let given = if num_produced == 0 {
            format!("{} operand(s)", operands.len())
        } else {
            format!(
                "{} operand(s) ({num_produced} produced, {} forwarded)",
                operands.len(),
                operands.len() - num_produced
            )
        };
        return Err(crate::diagnostics::report!(
            message: (format!(
                "invalid operation {name}: {successor} takes {} {input_kind}(s), but {name} gives \
                 it {given}",
                inputs.len()
            )),
            labels: [
                primary(op.span(), format!("this gives {successor} {given}")),
            ],
        ));
    }

    let forwarded = operands.forwarded();
    for (offset, (operand, &input_ty)) in
        forwarded.iter().zip(inputs.iter().skip(num_produced)).enumerate()
    {
        let index = num_produced + offset;
        let operand_ty = operand.borrow().ty();
        if !are_types_compatible(operand_ty, input_ty) {
            return Err(crate::diagnostics::report!(
                message: (format!(
                    "invalid operation {name}: operand {index} for {successor} has type \
                     '{operand_ty}', but the {input_kind} it binds has type '{input_ty}'"
                )),
                labels: [
                    primary(
                        op.span(),
                        format!("operand {index} for {successor} has the wrong type")
                    ),
                ],
            ));
        }
    }
    Ok(())
}

/// This interface provides information for select-like operations, i.e., operations that forward
/// specific operands to the output, depending on a binary condition.
///
/// If the value of the condition is 1, then the `true` operand is returned, and the third operand
/// is ignored, even if it was poison.
///
/// If the value of the condition is 0, then the `false` operand is returned, and the second operand
/// is ignored, even if it was poison.
///
/// If the condition is poison, then poison is returned.
///
/// Implementing operations can also accept shaped conditions, in which case the operation works
/// element-wise.
pub trait SelectLikeOpInterface {
    /// Returns the operand that represents the boolean condition for this select-like op.
    fn get_condition(&self) -> crate::ValueRef;
    /// Returns the operand that would be chosen for a true condition.
    fn get_true_value(&self) -> crate::ValueRef;
    /// Returns the operand that would be chosen for a false condition.
    fn get_false_value(&self) -> crate::ValueRef;
}

/// Marker trait for unary ops, i.e. those which take a single operand
#[operation_trait]
pub trait UnaryOp {
    #[verifier]
    fn is_unary_op(op: &Operation, _context: &Context) -> Result<(), Report> {
        if op.num_operands() == 1 {
            Ok(())
        } else {
            Err(crate::diagnostics::report!(
                message: (::alloc::format!("invalid operation {}", op.name())),
                labels: [
                    primary(
                        op.span(),
                        format!(
                            "incorrect number of operands, expected 1, got {}",
                            op.num_operands()
                        )
                    ),
                ],
                notes: [
                    help(
                        "this operator implements 'UnaryOp', which requires it to have exactly \
                         one operand"
                    ),
                ],
            ))
        }
    }
}

/// Marker trait for binary ops, i.e. those which take two operands
#[operation_trait]
pub trait BinaryOp {
    #[verifier]
    fn is_binary_op(op: &Operation, _context: &Context) -> Result<(), Report> {
        if op.num_operands() == 2 {
            Ok(())
        } else {
            Err(crate::diagnostics::report!(
                message: (::alloc::format!("invalid operation {}", op.name())),
                labels: [
                    primary(
                        op.span(),
                        format!(
                            "incorrect number of operands, expected 2, got {}",
                            op.num_operands()
                        )
                    ),
                ],
                notes: [
                    help(
                        "this operator implements 'BinaryOp', which requires it to have exactly \
                         two operands"
                    ),
                ],
            ))
        }
    }
}

/// Op's regions have no arguments
#[operation_trait]
pub trait NoRegionArguments {
    #[verifier]
    fn no_region_arguments(op: &Operation, _context: &Context) -> Result<(), Report> {
        for region in op.regions().iter() {
            if region.is_empty() {
                continue;
            }
            if region.entry().has_arguments() {
                return Err(crate::diagnostics::report!(
                    message: (::alloc::format!("invalid operation {}", op.name())),
                    labels: [
                        primary(
                            op.span(),
                            "this operation does not permit regions with arguments, but one was \
                             found"
                        ),
                    ],
                ));
            }
        }

        Ok(())
    }
}

/// Op's regions have a single block
#[operation_trait]
pub trait SingleBlock {
    #[verifier]
    fn has_only_single_block_regions(op: &Operation, _context: &Context) -> Result<(), Report> {
        for region in op.regions().iter() {
            if region.body().iter().count() > 1 {
                return Err(crate::diagnostics::report!(
                    message: (::alloc::format!("invalid operation {}", op.name())),
                    labels: [
                        primary(
                            op.span(),
                            "this operation requires single-block regions, but regions with \
                             multiple blocks were found"
                        ),
                    ],
                ));
            }
        }

        Ok(())
    }
}

// pub trait SingleBlockImplicitTerminator<T: Op + Default> {}

/// Op has a single region
#[operation_trait]
pub trait SingleRegion {
    #[verifier]
    fn has_exactly_one_region(op: &Operation, _context: &Context) -> Result<(), Report> {
        let num_regions = op.num_regions();
        if num_regions != 1 {
            return Err(crate::diagnostics::report!(
                message: (::alloc::format!("invalid operation {}", op.name())),
                labels: [
                    primary(
                        op.span(),
                        format!(
                            "this operation requires exactly one region, but got {num_regions}"
                        )
                    ),
                ],
            ));
        }

        Ok(())
    }
}

/// Marker trait for a cast whose result is its operand: the same bits, under a different type.
///
/// Codegen emits nothing for such an op itself: the stack slot that holds the operand becomes the
/// result. Only an operand still live after the cast is copied first (`dup`), as for any other
/// op, and the copy becomes the result. A cast that checks or changes a value, such as `arith.zext`
/// or `arith.trunc`, or a cast between signednesses, must not implement this.
///
/// The verifier admits one operand and one result of the same type, or that are both pointers in
/// the same address space (the pointee may differ, and either may be opaque): a pointer in
/// another address space need not have the same representation. Integer types are signless and
/// uniqued, so an integer cast is transparent only between identical types.
#[operation_trait]
pub trait TransparentCast {
    #[verifier]
    fn is_transparent_cast(op: &Operation, _context: &Context) -> Result<(), Report> {
        use crate::Value;

        let invalid = |label: alloc::string::String, help: &str| {
            crate::diagnostics::report!(
                message: (::alloc::format!("invalid operation {}", op.name())),
                labels: [
                    primary(op.span(), label),
                ],
                notes: [
                    help(::alloc::format!(
                        "this operator implements 'TransparentCast', which {help}"
                    )),
                ],
            )
        };

        if op.num_operands() != 1 || op.num_results() != 1 {
            return Err(invalid(
                format!(
                    "expected one operand and one result, got {} and {}",
                    op.num_operands(),
                    op.num_results()
                ),
                "requires it to have exactly one operand and one result",
            ));
        }
        let from = op.operands()[0].borrow().value().ty();
        let to = op.results()[0].borrow().ty();

        if from == to {
            return Ok(());
        }
        match (from.integer_width(), to.integer_width()) {
            (Some(from_width), Some(to_width)) => Err(invalid(
                format!("{from} is {from_width} bits wide, but {to} is {to_width}"),
                "requires an integer operand and result to be of the same type",
            )),
            _ => match (from.as_pointer(), to.as_pointer()) {
                (Some(from_ptr), Some(to_ptr)) if from_ptr.addrspace() == to_ptr.addrspace() => {
                    Ok(())
                }
                (Some(from_ptr), Some(to_ptr)) => Err(invalid(
                    format!(
                        "{from} is in address space {}, but {to} is in address space {}",
                        from_ptr.addrspace(),
                        to_ptr.addrspace()
                    ),
                    "requires pointers to be in the same address space",
                )),
                _ => Err(invalid(
                    format!("cannot reinterpret {from} as {to}"),
                    "converts only pointers to pointers, or a value to a value of its own type",
                )),
            },
        }
    }
}

// pub trait HasParent<T> {}
// pub trait ParentOneOf<(T,...)> {}

/// Marker trait for ops which:
///
/// * Represent the attachment of metadata to values in the IR
/// * Should not be considered as a "real" user for purposes of determining liveness of its operands
/// * Should not be considered dead unless all of its operands are also dead
/// * Does not result in any code being emitted during codegen
///
/// The goal of such operations is to attach important metadata, such as debug information, to
/// values in the IR, ensuring that the metadata is preserved through transformations, while not
/// interfering with optimizations that may make the original value dead except for the uses by
/// transparent ops.
#[operation_trait]
pub trait Transparent {
    #[verifier]
    fn has_no_results(op: &Operation, _context: &Context) -> Result<(), Report> {
        if op.results().is_empty() {
            Ok(())
        } else {
            Err(crate::diagnostics::report!(
                message: (::alloc::format!("invalid operation {}", op.name())),
                labels: [
                    primary(op.span(), "expected operation to have no results"),
                ],
                notes: [
                    help(
                        "this operator implements 'Transparent', which requires it to have no \
                         results"
                    ),
                ],
            ))
        }
    }

    #[verifier]
    fn has_no_more_than_one_operand(op: &Operation, _context: &Context) -> Result<(), Report> {
        if op.num_operands() > 1 {
            Err(crate::diagnostics::report!(
                message: (::alloc::format!("invalid operation {}", op.name())),
                labels: [
                    primary(op.span(), "expected operation to have no more than one operand"),
                ],
                notes: [
                    help(
                        "this operator implements 'Transparent', which requires it to have an \
                         arity < 2"
                    ),
                ],
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use super::*;
    use crate::{
        BlockRef, BuilderExt, Op, Report, SourceSpan, StructField, StructType, Type, Value,
        ValueRef,
        derive::{EffectOpInterface, operation},
        dialects::{
            builtin::{BuiltinOpBuilder, attributes::TypeAttr},
            test::TestDialect,
        },
        effects::MemoryEffectOpInterface,
        testing::Test,
        traits::InferTypeOpInterface,
    };

    /// A cast of any type to any type, to put the `TransparentCast` verifier to every pair.
    #[operation(
        dialect = TestDialect,
        traits(TransparentCast),
        implements(InferTypeOpInterface)
    )]
    pub struct AnyCast {
        #[operand]
        operand: AnyType,
        #[attr(hidden)]
        ty: TypeAttr,
        #[result]
        result: AnyType,
    }

    impl InferTypeOpInterface for AnyCast {
        fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
            let ty = *self.get_ty();
            self.result_mut().set_type(ty);
            Ok(())
        }
    }

    /// The operand and result types of a cast, made in the test's context.
    type CastTypes = fn(&Context) -> (Type, Type);

    /// Verify a function that casts its argument, of type `from`, to `to`.
    fn verify_cast(types: CastTypes) -> Result<(), Report> {
        let span = SourceSpan::UNKNOWN;
        let mut test = Test::named("verify_cast");
        let (from, to) = types(test.context());
        test.with_function("verify_cast", &[from], &[]);
        {
            let mut builder = test.function_builder();
            let arg = builder.entry_block().borrow().arguments()[0] as ValueRef;
            let cast = builder.builder_mut().create::<AnyCast, _>(span);
            cast(arg, to).unwrap();
            builder.ret(None, span).unwrap();
        }
        test.function().borrow().as_operation().recursively_verify()
    }

    #[test]
    fn a_transparent_cast_between_pointers_ignores_the_pointee() {
        let cases: [CastTypes; 4] = [
            |context| (context.ptr_type(context.i8()), context.ptr_type(context.i64())),
            |context| (context.ptr_type(context.i8()), context.opaque_ptr_type()),
            |context| (context.opaque_ptr_type(), context.ptr_type(context.i8())),
            |context| (context.opaque_ptr_type(), context.opaque_ptr_type()),
        ];
        for types in cases {
            verify_cast(types).unwrap_or_else(|err| {
                let context = Context::default();
                let (from, to) = types(&context);
                panic!("{from} to {to} should be transparent: {err}")
            });
        }
    }

    #[test]
    fn a_transparent_cast_between_pointers_keeps_the_address_space() {
        let typed_in_one: CastTypes = |context| {
            (
                context.ptr_type_in(Some(context.i8()), 1),
                context.ptr_type_in(Some(context.i64()), 1),
            )
        };
        let opaque_in_one: CastTypes =
            |context| (context.ptr_type_in(None, 1), context.ptr_type_in(Some(context.i8()), 1));
        for types in [typed_in_one, opaque_in_one] {
            verify_cast(types).unwrap_or_else(|err| panic!("a cast within address space 1: {err}"));
        }

        let cases: [CastTypes; 3] = [
            |context| (context.ptr_type(context.i8()), context.ptr_type_in(Some(context.i8()), 1)),
            |context| (context.ptr_type_in(None, 2), context.opaque_ptr_type()),
            |context| (context.ptr_type_in(None, 1), context.ptr_type_in(Some(context.i32()), 2)),
        ];
        for types in cases {
            let message = rejection(types);
            assert!(
                message.contains("requires pointers to be in the same address space"),
                "{message}"
            );
        }
    }

    #[test]
    fn a_cast_between_identical_types_is_always_transparent() {
        let cases: [CastTypes; 5] = [
            |context| (context.isize(), context.isize()),
            |context| (context.int_type(24), context.int_type(24)),
            |context| (context.f64(), context.f64()),
            |context| (context.array_type(context.i8(), 4), context.array_type(context.i8(), 4)),
            |context| {
                let pair = || {
                    StructType::new([
                        StructField::new(context.i32()),
                        StructField::new(context.isize()),
                    ])
                };
                (context.struct_type(pair()), context.struct_type(pair()))
            },
        ];
        for types in cases {
            verify_cast(types).unwrap_or_else(|err| {
                let context = Context::default();
                let (from, to) = types(&context);
                panic!("{from} to {to} should be transparent: {err}")
            });
        }
    }

    #[test]
    fn a_transparent_cast_never_reinterprets_a_pointer_as_an_integer() {
        let cases: [CastTypes; 4] = [
            |context| (context.ptr_type(context.i8()), context.i32()),
            |context| (context.i32(), context.ptr_type(context.i8())),
            |context| (context.ptr_type(context.i8()), context.isize()),
            |context| (context.isize(), context.i64()),
        ];
        for types in cases {
            let message = rejection(types);
            assert!(message.contains("converts only pointers to pointers"), "{message}");
        }
    }

    /// The help of the verifier's error for a cast, on one line.
    fn rejection(types: CastTypes) -> alloc::string::String {
        let err = verify_cast(types).unwrap_err();
        format!("{err:?}").split_whitespace().collect::<alloc::vec::Vec<_>>().join(" ")
    }

    #[test]
    fn a_transparent_cast_never_changes_the_bit_width() {
        let cases: [CastTypes; 2] = [
            |context| (context.i32(), context.i64()),
            |context| (context.i64(), context.i128()),
        ];
        for types in cases {
            let message = rejection(types);
            assert!(message.contains("to be of the same type"), "{message}");
        }
    }

    #[test]
    fn a_transparent_cast_never_changes_the_kind_of_value() {
        let cases: [CastTypes; 2] = [
            |context| (context.f64(), context.i64()),
            |context| (context.i64(), context.f64()),
        ];
        for types in cases {
            let message = rejection(types);
            assert!(message.contains("converts only pointers to pointers"), "{message}");
        }
    }

    /// A branch that forwards its operands to its one successor.
    #[derive(EffectOpInterface)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(BranchOpInterface, MemoryEffectOpInterface)
    )]
    pub struct BranchTestBr {
        #[successor]
        target: Successor,
    }

    impl BranchOpInterface for BranchTestBr {}

    /// A branch that produces the first argument of its one successor itself, and forwards its
    /// operands to the rest.
    #[derive(EffectOpInterface)]
    #[operation(
        dialect = TestDialect,
        traits(Terminator),
        implements(BranchOpInterface, MemoryEffectOpInterface)
    )]
    pub struct BranchTestProducingBr {
        #[successor]
        target: Successor,
    }

    impl BranchOpInterface for BranchTestProducingBr {
        fn get_successor_operands(&self, index: usize) -> crate::SuccessorOperandRange<'_> {
            let op = self.as_operation();
            let group = op.successors()[index].operand_group as usize;
            crate::SuccessorOperandRange::new(1, op.operands().group(group))
        }
    }

    /// Build a branch from the entry block of a function taking an `i32` and an `i64` to a block
    /// whose arguments have the types `dest` returns, forwarding the function's arguments at the
    /// positions in `forwarded`, and verify the branch. The branch is a [BranchTestProducingBr]
    /// when `producing` is set, and a [BranchTestBr] otherwise.
    fn verify_branch(
        producing: bool,
        dest: fn(&Context) -> alloc::vec::Vec<Type>,
        forwarded: &[usize],
    ) -> Result<(), Report> {
        type Args = (BlockRef, alloc::vec::Vec<ValueRef>);

        let span = SourceSpan::UNKNOWN;
        let mut test = Test::named("verify_branch");
        let (i32, i64) = (test.context().i32(), test.context().i64());
        test.with_function("verify_branch", &[i32, i64], &[]);
        let dest_types = dest(test.context());
        let mut builder = test.function_builder();
        let arguments = builder
            .entry_block()
            .borrow()
            .arguments()
            .iter()
            .map(|arg| *arg as ValueRef)
            .collect::<alloc::vec::Vec<_>>();
        let block = builder.create_block();
        for ty in dest_types {
            builder.append_block_param(block, ty, span);
        }
        let operands =
            forwarded.iter().map(|&index| arguments[index]).collect::<alloc::vec::Vec<_>>();
        let branch = if producing {
            let create = builder.builder_mut().create::<BranchTestProducingBr, Args>(span);
            create(block, operands).unwrap().as_operation_ref()
        } else {
            let create = builder.builder_mut().create::<BranchTestBr, Args>(span);
            create(block, operands).unwrap().as_operation_ref()
        };
        branch.borrow().verify()
    }

    #[test]
    fn a_branch_forwards_as_many_operands_as_its_successor_takes() {
        let err = verify_branch(false, |context| [context.i32()].into(), &[]).unwrap_err();
        let err = format!("{err}");
        assert!(err.contains("successor 0 (^block"), "{err}");
        assert!(
            err.contains("takes 1 block argument(s), but test.branch_test_br gives it 0"),
            "{err}"
        );

        verify_branch(false, |context| [context.i32()].into(), &[0]).unwrap();
    }

    #[test]
    fn a_branch_forwards_operands_of_its_successors_argument_types() {
        let err = verify_branch(false, |context| [context.i32()].into(), &[1]).unwrap_err();
        let err = format!("{err}");
        assert!(err.contains("operand 0 for successor 0 (^block"), "{err}");
        assert!(
            err.contains("has type 'i64', but the block argument it binds has type 'i32'"),
            "{err}"
        );
    }

    #[test]
    fn a_produced_operand_counts_toward_the_successors_arguments_but_has_no_type_to_compare() {
        // The produced first argument may have any type; the forwarded `i32` binds the second.
        verify_branch(true, |context| [context.i1(), context.i32()].into(), &[0]).unwrap();

        let err = verify_branch(true, |context| [context.i32()].into(), &[0]).unwrap_err();
        let err = format!("{err}");
        assert!(err.contains("gives it 2 operand(s) (1 produced, 1 forwarded)"), "{err}");

        let err =
            verify_branch(true, |context| [context.i1(), context.i32()].into(), &[1]).unwrap_err();
        let err = format!("{err}");
        assert!(err.contains("operand 1 for successor 0"), "{err}");
        assert!(
            err.contains("has type 'i64', but the block argument it binds has type 'i32'"),
            "{err}"
        );
    }
}
