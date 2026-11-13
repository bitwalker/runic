//! Constant folding of integer operations.
//!
//! Every fold of an `arith` operation on constants goes through [fold_ints] (one result) or
//! [fold_int_pair] (two results). Each fold states what its results are ([FoldsTo]: a value of
//! the operand type, an `i1`, or a cast's target), and the helpers decline unless the operation's
//! result types are those: an operation that does not verify does not fold. They read the operand
//! constants, compute the result with a closure over [Int]s, and create a new
//! [IntAttr](runic_core::dialects::builtin::attributes::IntAttr) for it. For `isize` operands the
//! computation goes through [fold_isize], with the result kind that the result type gives, so no
//! fold chooses it for itself and no fold computes on the 64-bit value of an `isize` constant
//! directly.

use alloc::rc::Rc;

use runic_core::{
    AttributeRef, Context, FoldResult, Int, IsizeFold, OpFoldResult, Operation, SmallVec, Type,
    Value,
    attributes::IntegerLikeAttr,
    fold_isize,
    matchers::{self, Matcher},
};

/// The value of the constant `operand`, if it is an integer-like attribute of type `ty` holding a
/// value as wide as a constant of `ty`.
///
/// Anything else is not a constant this dialect folds: a missing operand (not a constant), a
/// constant of another type, or an attribute that is not integer-like, such as `ub.poison`'s.
pub(crate) fn constant_operand(operand: &Option<AttributeRef>, ty: Type) -> Option<Int> {
    let operand = operand.as_ref()?.borrow();
    let attr = operand.as_attr().as_trait::<dyn IntegerLikeAttr>()?;
    let value = attr.as_int();
    (attr.ty() == ty && Int::width_for_type(ty) == Some(value.width())).then_some(value)
}

/// The constant attribute of each operand of `op`, `None` for an operand that is not a constant:
/// the operands [Foldable::fold](runic_core::traits::Foldable::fold) hands to `fold_with`.
pub(crate) fn operand_constants(op: &Operation) -> SmallVec<[Option<AttributeRef>; 2]> {
    op.operands()
        .iter()
        .map(|operand| matchers::foldable_operand().matches(operand))
        .collect()
}

/// The type of the first operand of `op`.
pub(crate) fn operand_type(op: &Operation) -> Type {
    op.operands()[0].borrow().as_value_ref().borrow().ty()
}

/// The type of the result `index` of `op`.
pub(crate) fn result_type(op: &Operation, index: usize) -> Type {
    op.results()[index].borrow().ty()
}

/// The values of `operands`, if there are `N` of them and each is a constant of type `ty`.
fn constant_operands<const N: usize>(
    operands: &[Option<AttributeRef>],
    ty: Type,
) -> Option<[Int; N]> {
    let operands: &[Option<AttributeRef>; N] = operands.try_into().ok()?;
    let mut values = [Int::zero(1); N];
    for (value, operand) in values.iter_mut().zip(operands) {
        *value = constant_operand(operand, ty)?;
    }
    Some(values)
}

/// What a result of a folded operation is, as its operation's verifier requires it.
#[derive(Copy, Clone, Debug)]
pub(crate) enum FoldsTo {
    /// A value of the operand type.
    Operand,
    /// An `i1`: a comparison or an overflow flag.
    Bool,
    /// A value of this type: a cast's target.
    Type(Type),
}

impl FoldsTo {
    /// The type of the result, for operands of type `operand_ty`.
    fn ty(self, context: &Context, operand_ty: Type) -> Type {
        match self {
            FoldsTo::Operand => operand_ty,
            FoldsTo::Bool => context.i1(),
            FoldsTo::Type(ty) => ty,
        }
    }
}

/// Whether a result of type `result_ty` is what `folds_to` says for operands of type
/// `operand_ty`, and is one these helpers compute.
///
/// A fixed-width operand with an `isize` result is refused: computing it once, at the 64 bits an
/// `isize` constant holds, is right only for a resize, which folds it itself (see
/// `ops::coercions::fold_resize`).
fn result_is(context: &Context, operand_ty: Type, result_ty: Type, folds_to: FoldsTo) -> bool {
    result_ty == folds_to.ty(context, operand_ty)
        && (operand_ty.is_isize() || !result_ty.is_isize())
}

/// `f` on `values`, constants of type `operand_ty`, for a result of type `result_ty`, which
/// [result_is] has accepted.
///
/// A fixed-width `operand_ty` computes directly. An `isize` one computes through [fold_isize], with
/// [IsizeFold::PointerWidth] for an `isize` result and [IsizeFold::Fixed] at the result's width for
/// a fixed-width one (a comparison's or an overflow flag's `i1`, a conversion's target).
///
/// `f` is written against the width of the operands it is given (`a.width()`), which for `isize`
/// is 64 on one call and 32 on the other. It returns a value as wide as a constant of
/// `result_ty`, or `None` when the operation does not fold.
fn compute<const N: usize>(
    values: [Int; N],
    operand_ty: Type,
    result_ty: Type,
    f: impl Fn([Int; N]) -> Option<Int>,
) -> Option<Int> {
    let result_width = Int::width_for_type(result_ty)?;
    if operand_ty.is_isize() {
        let kind = if result_ty.is_isize() {
            IsizeFold::PointerWidth
        } else {
            IsizeFold::Fixed(result_width)
        };
        fold_isize(values, kind, f)
    } else {
        f(values)
    }
}

/// Folds an operation whose operands are constants of the integer type `operand_ty` to a constant
/// of type `result_ty`: the value `f` computes on the operand values (see [compute]).
///
/// Returns `None`, and creates nothing, unless `result_ty` is what `folds_to` says, every operand
/// is an integer-like attribute of type `operand_ty`, and `f` folds. The result is a new
/// attribute: an operand's attribute is never changed.
pub(crate) fn fold_ints<const N: usize>(
    context: &Rc<Context>,
    operands: &[Option<AttributeRef>],
    operand_ty: Type,
    result_ty: Type,
    folds_to: FoldsTo,
    f: impl Fn([Int; N]) -> Option<Int>,
) -> Option<AttributeRef> {
    if !result_is(context, operand_ty, result_ty, folds_to) {
        return None;
    }
    let values = constant_operands(operands, operand_ty)?;
    let value = compute(values, operand_ty, result_ty, f)?;
    Some(context.int_attr(value, result_ty).as_attribute_ref())
}

/// As [fold_ints], for an operation of two results: `first` computes the first, of type
/// `result_tys[0]`, which must be what `folds_to[0]` says, and `second` the second.
///
/// The two fold together or not at all: no attribute is created unless both do.
pub(crate) fn fold_int_pair<const N: usize>(
    context: &Rc<Context>,
    operands: &[Option<AttributeRef>],
    operand_ty: Type,
    result_tys: [Type; 2],
    folds_to: [FoldsTo; 2],
    first: impl Fn([Int; N]) -> Option<Int>,
    second: impl Fn([Int; N]) -> Option<Int>,
) -> Option<[AttributeRef; 2]> {
    if !result_tys
        .iter()
        .zip(folds_to)
        .all(|(result_ty, folds_to)| result_is(context, operand_ty, *result_ty, folds_to))
    {
        return None;
    }
    let values = constant_operands(operands, operand_ty)?;
    let first = compute(values, operand_ty, result_tys[0], first)?;
    let second = compute(values, operand_ty, result_tys[1], second)?;
    Some([
        context.int_attr(first, result_tys[0]).as_attribute_ref(),
        context.int_attr(second, result_tys[1]).as_attribute_ref(),
    ])
}

/// Pushes the attributes of a fold onto `results`: [FoldResult::Ok] if there are any, and
/// [FoldResult::Failed], with nothing pushed, for `None`.
pub(crate) fn push_folded<const K: usize>(
    results: &mut SmallVec<[OpFoldResult; 1]>,
    folded: Option<[AttributeRef; K]>,
) -> FoldResult {
    match folded {
        Some(attrs) => {
            results.extend(attrs.map(OpFoldResult::Attribute));
            FoldResult::Ok(())
        }
        None => FoldResult::Failed,
    }
}

/// Implements [Foldable](runic_core::traits::Foldable) for an operation of one result, which is
/// what the [FoldsTo] variant `$to` says.
///
/// `$f` is an expression, which may read the operation through `$op`, giving the closure
/// [fold_ints] computes the result with; the operand type is the first operand's. `fold` folds
/// with the operands' constants.
macro_rules! fold_one {
    ($Op:ty, $to:ident, |$op:ident| $f:expr) => {
        impl runic_core::traits::Foldable for $Op {
            fn fold(
                &self,
                results: &mut runic_core::SmallVec<[runic_core::OpFoldResult; 1]>,
            ) -> runic_core::FoldResult {
                let operands = $crate::fold::operand_constants(self.as_operation());
                self.fold_with(&operands, results)
            }

            fn fold_with(
                &self,
                operands: &[Option<runic_core::AttributeRef>],
                results: &mut runic_core::SmallVec<[runic_core::OpFoldResult; 1]>,
            ) -> runic_core::FoldResult {
                let $op = self;
                let op = self.as_operation();
                let folded = $crate::fold::fold_ints(
                    &op.context_rc(),
                    operands,
                    $crate::fold::operand_type(op),
                    $crate::fold::result_type(op, 0),
                    $crate::fold::FoldsTo::$to,
                    $f,
                );
                $crate::fold::push_folded(results, folded.map(|attr| [attr]))
            }
        }
    };
}

/// As [fold_one], for an operation of two results, which are what `$first` and `$second` say:
/// `$f` gives a pair of closures, for the first result and the second, which fold together or not
/// at all (see [fold_int_pair]).
macro_rules! fold_two {
    ($Op:ty, [$first:ident, $second:ident], |$op:ident| $f:expr) => {
        impl runic_core::traits::Foldable for $Op {
            fn fold(
                &self,
                results: &mut runic_core::SmallVec<[runic_core::OpFoldResult; 1]>,
            ) -> runic_core::FoldResult {
                let operands = $crate::fold::operand_constants(self.as_operation());
                self.fold_with(&operands, results)
            }

            fn fold_with(
                &self,
                operands: &[Option<runic_core::AttributeRef>],
                results: &mut runic_core::SmallVec<[runic_core::OpFoldResult; 1]>,
            ) -> runic_core::FoldResult {
                let $op = self;
                let op = self.as_operation();
                let (first, second) = $f;
                let folded = $crate::fold::fold_int_pair(
                    &op.context_rc(),
                    operands,
                    $crate::fold::operand_type(op),
                    [$crate::fold::result_type(op, 0), $crate::fold::result_type(op, 1)],
                    [$crate::fold::FoldsTo::$first, $crate::fold::FoldsTo::$second],
                    first,
                    second,
                );
                $crate::fold::push_folded(results, folded)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use runic_core::dialects::builtin::attributes::TypeAttr;

    use super::*;

    /// The constant `value` of type `ty`, as an operand.
    fn constant(context: &Rc<Context>, value: Int, ty: Type) -> Option<AttributeRef> {
        Some(context.int_attr(value, ty).as_attribute_ref())
    }

    /// The value and type of the integer attribute `attr`.
    fn read(attr: Option<AttributeRef>) -> Option<(Int, Type)> {
        let attr = attr?;
        let attr = attr.borrow();
        let int = attr.as_attr().as_trait::<dyn IntegerLikeAttr>().unwrap();
        Some((int.as_int(), int.ty()))
    }

    fn add([a, b]: [Int; 2]) -> Option<Int> {
        Some(a.wrapping_add(b))
    }

    #[test]
    fn a_fixed_width_fold_computes_directly() {
        let context = Rc::new(Context::default());
        let i8 = context.i8();
        let operands =
            [constant(&context, Int::from(127i8), i8), constant(&context, Int::from(1i8), i8)];
        let folded = fold_ints(&context, &operands, i8, i8, FoldsTo::Operand, add);
        assert_eq!(read(folded), Some((Int::from(-128i8), i8)));
        // `None` from the computation is no fold.
        assert_eq!(
            fold_ints(&context, &operands, i8, i8, FoldsTo::Operand, |[a, b]: [Int; 2]| {
                a.checked_add_signed(b)
            }),
            None
        );
    }

    #[test]
    fn an_isize_fold_needs_both_targets_to_agree() {
        let context = Rc::new(Context::default());
        let isize = context.isize();
        let big = constant(&context, Int::from(1i64 << 32), isize);
        let one = constant(&context, Int::from(1i64), isize);
        let two = constant(&context, Int::from(2i64), isize);
        // The low 32 bits of the sum are the same on both targets.
        let folded = fold_ints(&context, &[big, one], isize, isize, FoldsTo::Operand, add);
        assert_eq!(read(folded), Some((Int::from((1i64 << 32) + 1), isize)));
        // `2^32 / 2` is `2^31` on a 64-bit target and `0 / 2` on a 32-bit one.
        let divide = |[a, b]: [Int; 2]| a.div_unsigned(b);
        assert_eq!(fold_ints(&context, &[big, two], isize, isize, FoldsTo::Operand, divide), None);
        // A comparison has an `i1` result on both targets: `2^32 > 1` on one, `0 > 1` on the other.
        let i1 = context.i1();
        let greater = |[a, b]: [Int; 2]| Some(Int::from(a.cmp_unsigned(b).is_gt()));
        assert_eq!(fold_ints(&context, &[big, one], isize, i1, FoldsTo::Bool, greater), None);
        assert_eq!(
            read(fold_ints(&context, &[two, one], isize, i1, FoldsTo::Bool, greater)),
            Some((Int::from(true), i1))
        );
    }

    #[test]
    fn a_fold_declines_unless_the_result_type_is_what_it_says() {
        let context = Rc::new(Context::default());
        let (i1, i16, i32, isize) = (context.i1(), context.i16(), context.i32(), context.isize());
        let three = constant(&context, Int::from(3i32), i32);
        let one = constant(&context, Int::from(1i32), i32);
        let operands = [three, one];
        // A result of another type than the operands', or than `i1` for a flag.
        assert_eq!(fold_ints(&context, &operands, i32, i16, FoldsTo::Operand, add), None);
        let flag = |[a, b]: [Int; 2]| Some(Int::from(a == b));
        assert_eq!(fold_ints(&context, &operands, i32, i32, FoldsTo::Bool, flag), None);
        assert_eq!(fold_ints(&context, &operands, i32, i1, FoldsTo::Operand, add), None);
        // A pair declines if either result is not what it says.
        assert_eq!(
            fold_int_pair(
                &context,
                &operands,
                i32,
                [i1, i16],
                [FoldsTo::Bool, FoldsTo::Operand],
                flag,
                add
            ),
            None
        );
        // `isize` operands with a fixed-width result that is not the stated one.
        let big = constant(&context, Int::from(3i64), isize);
        assert_eq!(fold_ints(&context, &[big, big], isize, i16, FoldsTo::Operand, add), None);
        // A fixed-width operand with an `isize` result is a resize's own case.
        let resize = |[a]: [Int; 1]| Some(a.zext_or_trunc(64));
        assert_eq!(fold_ints(&context, &[three], i32, isize, FoldsTo::Type(isize), resize), None);
    }

    #[test]
    fn only_constants_of_the_operand_type_fold() {
        let context = Rc::new(Context::default());
        let (i32, i64) = (context.i32(), context.i64());
        let one = constant(&context, Int::from(1i32), i32);
        // A non-constant operand.
        assert_eq!(fold_ints(&context, &[one, None], i32, i32, FoldsTo::Operand, add), None);
        // A constant of another type.
        let wide = constant(&context, Int::from(1i64), i64);
        assert_eq!(fold_ints(&context, &[one, wide], i32, i32, FoldsTo::Operand, add), None);
        // An attribute that is not integer-like.
        let ty = Some(context.create_attribute::<TypeAttr, _>(i32).as_attribute_ref());
        assert_eq!(fold_ints(&context, &[one, ty], i32, i32, FoldsTo::Operand, add), None);
        // Another number of operands than the computation takes.
        assert_eq!(fold_ints(&context, &[one], i32, i32, FoldsTo::Operand, add), None);
        assert_eq!(
            read(fold_ints(&context, &[one, one], i32, i32, FoldsTo::Operand, add)),
            Some((Int::from(2i32), i32))
        );
    }
}
