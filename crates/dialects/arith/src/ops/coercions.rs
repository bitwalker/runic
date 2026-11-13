use alloc::{format, rc::Rc, string::String};

use runic_core::{
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::TypeAttr,
    effects::MemoryEffectOpInterface,
    traits::*,
    *,
};

use crate::*;

/// Folds a resize of the constant operand of `op` to the value `resize` maps it and the result
/// width to, at the result type.
///
/// Only a cast that satisfies `rule`, the rule its verifier checks, folds: a cast that does not
/// verify (an `isize` operand of `trunc`, the wrong width order) is left alone.
fn fold_resize(
    op: &Operation,
    operands: &[Option<AttributeRef>],
    results: &mut SmallVec<[OpFoldResult; 1]>,
    rule: CastRule,
    resize: impl Fn(Int, u32) -> Int,
) -> FoldResult {
    let operand_ty = crate::fold::operand_type(op);
    let result_ty = crate::fold::result_type(op, 0);
    if rule.check(operand_ty, result_ty).is_some() {
        return FoldResult::Failed;
    }
    let Some(width) = Int::width_for_type(result_ty) else {
        return FoldResult::Failed;
    };
    let folded =
        crate::fold::fold_ints(&op.context_rc(), operands, operand_ty, result_ty, |[a]| {
            Some(resize(a, width))
        });
    crate::fold::push_folded(results, folded.map(|attr| [attr]))
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

impl CastRule {
    /// What is wrong with a cast from `from` to `to` under this rule, or `None` if it is valid.
    fn check(self, from: Type, to: Type) -> Option<String> {
        match self {
            CastRule::Narrower | CastRule::Wider => {
                match (from.integer_width(), to.integer_width()) {
                    (Some(from_width), Some(to_width)) => match self {
                        CastRule::Narrower if to_width >= from_width => {
                            Some(format!("`{to}` is not narrower than `{from}`"))
                        }
                        CastRule::Wider if to_width <= from_width => {
                            Some(format!("`{to}` is not wider than `{from}`"))
                        }
                        _ => None,
                    },
                    _ => Some(format!(
                        "it resizes between fixed-width integer types, and `{from}` to `{to}` is \
                         not; use `arith.sext_or_trunc` or `arith.zext_or_trunc` for `isize`"
                    )),
                }
            }
            CastRule::OneIsize => {
                let fixed = |ty: Type| ty.integer_width().is_some();
                let valid = (from.is_isize() && fixed(to)) || (fixed(from) && to.is_isize());
                (!valid).then(|| {
                    format!(
                        "exactly one of its types must be `isize` and the other fixed-width, but \
                         it converts `{from}` to `{to}`"
                    )
                })
            }
        }
    }
}

/// Checks the operand and result types of the cast `op` against `rule`, and its result type
/// against its `ty` attribute.
fn verify_cast(op: &Operation, rule: CastRule, ty: Type) -> Result<(), Report> {
    let from = op.operands()[0].borrow().as_value_ref().borrow().ty();
    let to = op.results()[0].borrow().ty();
    let problem = if to != ty {
        Some(format!("its result type `{to}` is not its `ty` attribute `{ty}`"))
    } else {
        rule.check(from, to)
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
                verify_cast(self.as_operation(), $rule, *self.get_ty())
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
        let operands = crate::fold::operand_constants(self.as_operation());
        self.fold_with(&operands, results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        fold_resize(self.as_operation(), operands, results, CastRule::Narrower, |value, width| {
            value.trunc(width)
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
        let operands = crate::fold::operand_constants(self.as_operation());
        self.fold_with(&operands, results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        fold_resize(self.as_operation(), operands, results, CastRule::Wider, |value, width| {
            value.zext(width)
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
        let operands = crate::fold::operand_constants(self.as_operation());
        self.fold_with(&operands, results)
    }

    fn fold_with(
        &self,
        operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        fold_resize(self.as_operation(), operands, results, CastRule::Wider, |value, width| {
            value.sext(width)
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
