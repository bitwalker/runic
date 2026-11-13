use alloc::{format, rc::Rc, string::String};

use runic_core::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::TypeAttr,
    effects::MemoryEffectOpInterface,
    matchers::Matcher,
    traits::*,
    *,
};

use crate::*;

/// The value of the constant `operand`, if it is an integer-like attribute of type `ty` holding a
/// value as wide as a constant of `ty`.
pub(super) fn constant_operand(operand: &Option<AttributeRef>, ty: Type) -> Option<Int> {
    let operand = operand.as_ref()?.borrow();
    let attr = operand.as_attr().as_trait::<dyn IntegerLikeAttr>()?;
    let value = attr.as_int();
    (attr.ty() == ty && Int::width_for_type(ty) == Some(value.width())).then_some(value)
}

/// Folds a resize of the constant operand of `op` to the attribute `resize` maps its value and the
/// result width to, at the result type; `resize` returns `None` for a cast that is not valid.
///
/// The fold creates a new attribute: the operand's attribute may be shared by other uses.
fn fold_resize(
    op: &Operation,
    operand: &Option<AttributeRef>,
    results: &mut SmallVec<[OpFoldResult; 1]>,
    resize: impl FnOnce(Int, u32) -> Option<Int>,
) -> FoldResult {
    let operand_ty = op.operands()[0].borrow().as_value_ref().borrow().ty();
    let result_ty = op.results()[0].borrow().ty();
    let (Some(value), Some(width)) = (
        constant_operand(operand, operand_ty),
        result_ty.integer_width().filter(|width| *width <= Int::MAX_WIDTH),
    ) else {
        return FoldResult::Failed;
    };
    // An invalid cast (one its verifier rejects) is not folded.
    let Some(value) = resize(value, width) else {
        return FoldResult::Failed;
    };
    let result = op.context_rc().int_attr(value, result_ty);
    results.push(OpFoldResult::Attribute(result.as_attribute_ref()));
    FoldResult::Ok(())
}

/// What a cast operation requires of its operand type and its result type.
#[derive(Copy, Clone)]
enum CastRule {
    /// Both are fixed-width, and the result is narrower.
    Narrower,
    /// Both are fixed-width, and the result is wider.
    Wider,
    /// Exactly one of the two is `isize`, the other is fixed-width.
    OneIsize,
}

/// Checks the operand and result types of the cast `op` against `rule`.
fn verify_cast(op: &Operation, rule: CastRule) -> Result<(), Report> {
    let from = op.operands()[0].borrow().as_value_ref().borrow().ty();
    let to = op.results()[0].borrow().ty();
    let problem: Option<String> = match rule {
        CastRule::Narrower | CastRule::Wider => match (from.integer_width(), to.integer_width()) {
            (Some(from_width), Some(to_width)) => match rule {
                CastRule::Narrower if to_width >= from_width => {
                    Some(format!("`{to}` is not narrower than `{from}`"))
                }
                CastRule::Wider if to_width <= from_width => {
                    Some(format!("`{to}` is not wider than `{from}`"))
                }
                _ => None,
            },
            _ => Some(format!(
                "it resizes between fixed-width integer types, and `{from}` to `{to}` is not; use \
                 `arith.sext_or_trunc` or `arith.zext_or_trunc` for `isize`"
            )),
        },
        CastRule::OneIsize => {
            let fixed = |ty: Type| ty.integer_width().is_some();
            let valid = (from.is_isize() && fixed(to)) || (fixed(from) && to.is_isize());
            (!valid).then(|| {
                format!(
                    "exactly one of its types must be `isize` and the other fixed-width, but it \
                     converts `{from}` to `{to}`"
                )
            })
        }
    };
    let Some(problem) = problem else {
        return Ok(());
    };
    Err(runic_core::diagnostics::report!(
        message: "invalid operation",
        labels: [
            primary(op.span(), format!("invalid '{}' from `{from}` to `{to}`: {problem}", op.name())),
        ],
    ))
}

/// Implements [InferTypeOpInterface] for a cast whose result has the type `ty`, and checks the cast
/// against its [CastRule].
///
/// The check hangs on [InferTypeOpInterface] because [UnaryOp] cannot carry it: the
/// `#[operation_trait]` macro gives every such trait a blanket `Verify` impl that an operation
/// cannot specialize.
macro_rules! cast_type_rules {
    ($Op:ty, $rule:expr) => {
        impl InferTypeOpInterface for $Op {
            fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
                let ty = *self.get_ty();
                self.result_mut().set_type(ty);
                Ok(())
            }
        }

        impl Verify<dyn InferTypeOpInterface> for $Op {
            fn verify(&self, _context: &Context) -> Result<(), Report> {
                verify_cast(self.as_operation(), $rule)
            }
        }
    };
}

/// Truncation to the low bits of a narrower fixed-width integer type `ty`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Trunc {
    #[operand]
    operand: AnyInteger,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyInteger,
}

cast_type_rules!(Trunc, CastRule::Narrower);

impl Canonicalizable for Trunc {
    fn get_canonicalization_patterns(
        rewrites: &mut patterns::RewritePatternSet,
        context: Rc<Context>,
    ) {
        rewrites.push(crate::canonicalization::SimplifyTruncToBool::new(context));
    }
}

impl Foldable for Trunc {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let operand = matchers::foldable_operand().matches(&self.operand().as_operand_ref());
        self.fold_with(&[operand], results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        fold_resize(self.as_operation(), &operands[0], results, |value, width| {
            (width < value.width()).then(|| value.trunc(width))
        })
    }
}

/// Zero extension to a wider fixed-width integer type `ty`: the new high bits are zero.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Zext {
    #[operand]
    operand: AnyInteger,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyInteger,
}

cast_type_rules!(Zext, CastRule::Wider);

impl Foldable for Zext {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let operand = matchers::foldable_operand().matches(&self.operand().as_operand_ref());
        self.fold_with(&[operand], results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        fold_resize(self.as_operation(), &operands[0], results, |value, width| {
            (width > value.width()).then(|| value.zext(width))
        })
    }
}

/// Sign extension to a wider fixed-width integer type `ty`: the new high bits are copies of the
/// sign bit.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Sext {
    #[operand]
    operand: AnyInteger,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyInteger,
}

cast_type_rules!(Sext, CastRule::Wider);

impl Foldable for Sext {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let operand = matchers::foldable_operand().matches(&self.operand().as_operand_ref());
        self.fold_with(&[operand], results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        fold_resize(self.as_operation(), &operands[0], results, |value, width| {
            (width > value.width()).then(|| value.sext(width))
        })
    }
}

/// Conversion between `isize` and a fixed-width integer type `ty`, in either direction: the value
/// is sign-extended if the target is wider than the source on the target machine, and truncated if
/// it is narrower.
///
/// Exactly one of the operand type and `ty` is `isize`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct SextOrTrunc {
    #[operand]
    operand: AnyInteger,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyInteger,
}

cast_type_rules!(SextOrTrunc, CastRule::OneIsize);

/// Conversion between `isize` and a fixed-width integer type `ty`, in either direction: the value
/// is zero-extended if the target is wider than the source on the target machine, and truncated if
/// it is narrower.
///
/// Exactly one of the operand type and `ty` is `isize`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = ArithDialect,
    traits(UnaryOp),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct ZextOrTrunc {
    #[operand]
    operand: AnyInteger,
    #[attr(hidden)]
    ty: TypeAttr,
    #[result]
    result: AnyInteger,
}

cast_type_rules!(ZextOrTrunc, CastRule::OneIsize);
