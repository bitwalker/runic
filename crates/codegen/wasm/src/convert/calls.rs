//! The patterns of functions, calls, returns, locals and constants, and the calling conventions.
//!
//! The types change through core's builtin conversion patterns
//! ([populate_builtin_conversion_patterns]): this module adds them, and wraps four of them (those
//! of `builtin.function`, `builtin.call`, `builtin.tail_call` and `builtin.ret`) with what the
//! calling convention needs on top, in carried form. The builtin patterns of
//! `builtin.call_indirect`, `builtin.tail_call_indirect` and `builtin.function_ref` are left out:
//! a function value is an `i32` here, a table index, and [symbols](super::symbols) lowers those
//! three, its indirect calls under the same rules:
//!
//! * `extern("wasm")`: nothing. Both sides of a call rely on carried form.
//! * `extern("C")`: nothing is assumed that the signature does not say. A function masks each
//!   narrow parameter (a carried integer narrower than its carrier) that is not marked `zext` at
//!   its entry, and its body reads the masked value; it sign-extends a narrow result marked `sext`
//!   at each `builtin.ret` and `builtin.ret_imm`. A call sign-extends each argument whose
//!   parameter is narrow and marked `sext`, and masks each narrow result that is not marked
//!   `zext`. A tail call sign-extends its arguments as a call does; its results are the callee's,
//!   and the function verifier already requires that the callee's results, markers included, are
//!   the function's, so nothing is done to them. An indirect call and an indirect tail call follow
//!   the rules of a call and a tail call.
//! * any other convention is an error naming the convention and the operation.
//!
//! `builtin.ret_imm` becomes a `wasm.const` and a `builtin.ret`; `arith.constant` becomes a
//! `wasm.const` of its value zero-extended in the carrier, and `arith.constantf` a `wasm.constf`.

use alloc::{format, rc::Rc, vec::Vec};

use runic_dialect_arith as arith;
use runic_dialect_wasm as wasm;
use runic_ir::{
    Context, Op, OpRegistration, Operation, OperationName, OperationRef, Report, SmallVec, Spanned,
    Symbol, Type, Value, ValueRef,
    conversion::{
        ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConvertedOperands,
        FrozenConversionPatternSet, TypeConverter, populate_builtin_conversion_patterns,
    },
    diagnostics::report,
    dialects::builtin::{
        Call, CallIndirect, Function, Ret, RetImm, TailCall, TailCallIndirect,
        UnrealizedConversionCast,
        attributes::{ArgumentExtension, Signature},
    },
    patterns::{Pattern, PatternBenefit, PatternInfo, PatternKind},
};

use crate::{
    carried::{self, const_int, mask, sign_extend},
    types::{carrier, carrier_width, width},
};

/// The calling conventions Wasm form has.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Convention {
    /// `extern("wasm")`: both sides of a call rely on carried form.
    Wasm,
    /// `extern("C")`: nothing is assumed that the signature does not say.
    C,
}

impl Convention {
    /// The convention of `signature`, held by `op`; an error, with a label on `op`, for a
    /// convention other than `"wasm"` and `"C"`.
    pub fn of(signature: &Signature, op: &Operation) -> Result<Self, Report> {
        match signature.calling_convention().as_str() {
            "wasm" => Ok(Self::Wasm),
            "C" => Ok(Self::C),
            other => Err(report!(
                message: (format!(
                    "`convert-to-wasm` lowers the calling conventions \"wasm\" and \"C\", not \
                     \"{other}\""
                )),
                labels: [primary(
                    op.span(),
                    format!("this `{}` uses the calling convention \"{other}\"", op.name())
                )],
            )),
        }
    }
}

/// The width `N` of `ty` when it is a narrow integer: a carried integer narrower than its
/// carrier, whose bits above `N` carried form keeps clear.
pub fn narrow(ty: Type) -> Option<u32> {
    let n = width(ty)?;
    let carrier_bits = if n <= 32 { 32 } else { 64 };
    (n < carrier_bits).then_some(n)
}

/// The names of the `wasm` operations the convention rules may create.
fn extension_ops(context: &Rc<Context>) -> [OperationName; 7] {
    [
        registered::<wasm::Const>(context),
        registered::<wasm::And>(context),
        registered::<wasm::Extend8S>(context),
        registered::<wasm::Extend16S>(context),
        registered::<wasm::Extend32S>(context),
        registered::<wasm::Shl>(context),
        registered::<wasm::ShrS>(context),
    ]
}

/// The name of the registered operation `T`.
pub(crate) fn registered<T: OpRegistration>(context: &Rc<Context>) -> OperationName {
    context
        .get_or_register_dialect::<<T as OpRegistration>::Dialect>()
        .expect_registered_name::<T>()
}

/// The information of a pattern of `convert-to-wasm` named `name`, rooted at `T`, that creates
/// `generated`.
pub(crate) fn pattern_info<T: OpRegistration>(
    context: &Rc<Context>,
    name: &'static str,
    generated: impl IntoIterator<Item = OperationName>,
) -> PatternInfo {
    let mut info = PatternInfo::new(
        context.clone(),
        name,
        PatternKind::Operation(registered::<T>(context)),
        PatternBenefit::new(1),
    );
    info.with_generated_ops(generated);
    info
}

/// Adds the builtin conversion patterns driven by `converter`, with the calling-convention rules
/// on those of functions, calls, tail calls and returns, and the patterns of `builtin.ret_imm`,
/// `arith.constant` and `arith.constantf`.
pub fn populate(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
) -> Result<(), Report> {
    let context = patterns.context();
    let mut builtin = ConversionPatternSet::new(context.clone());
    populate_builtin_conversion_patterns(&mut builtin, converter)?;
    let builtin = FrozenConversionPatternSet::new(builtin);
    let mut wrapped = 0;
    for pattern in builtin.patterns() {
        let inner = Rc::clone(pattern);
        let role = match inner.name() {
            "builtin-function-by-type" => Role::Function,
            "builtin-call-by-type" => Role::Call,
            "builtin-tail-call-by-type" => Role::TailCall,
            "builtin-ret-by-type" => Role::Ret,
            // A function value is a table index here, not a function value: lowered by
            // `symbols`.
            "builtin-call-indirect-by-type"
            | "builtin-tail-call-indirect-by-type"
            | "builtin-function-ref-by-type" => continue,
            _ => {
                patterns.push(Shared(inner));
                continue;
            }
        };
        wrapped += 1;
        patterns.push(WithConvention::new(&context, role, inner, converter));
    }
    if wrapped != 4 {
        return Err(Report::msg(
            "`convert-to-wasm` did not find the builtin conversion patterns of functions, calls, \
             tail calls and returns that it adds the calling conventions to",
        ));
    }
    patterns.push(RetImmPattern {
        info: pattern_info::<RetImm>(
            &context,
            "wasm-ret-imm",
            [registered::<wasm::Const>(&context), registered::<Ret>(&context)],
        ),
        converter: converter.clone(),
    });
    patterns.push(ConstantPattern {
        info: pattern_info::<arith::Constant>(
            &context,
            "wasm-arith-constant",
            [registered::<wasm::Const>(&context)],
        ),
        converter: converter.clone(),
    });
    patterns.push(ConstantFPattern {
        info: pattern_info::<arith::Constantf>(
            &context,
            "wasm-arith-constantf",
            [registered::<wasm::ConstF>(&context)],
        ),
        converter: converter.clone(),
    });
    Ok(())
}

/// A builtin conversion pattern added as it is.
struct Shared(Rc<dyn ConversionPattern>);

impl Pattern for Shared {
    fn info(&self) -> &PatternInfo {
        self.0.info()
    }
}

impl ConversionPattern for Shared {
    fn type_converter(&self) -> Option<&TypeConverter> {
        self.0.type_converter()
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        self.0.match_and_rewrite(op, operands, rewriter)
    }
}

/// Which pattern a [WithConvention] wraps: the operation it is rooted at.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Role {
    /// `builtin.function`.
    Function,
    /// `builtin.call`.
    Call,
    /// `builtin.tail_call`.
    TailCall,
    /// `builtin.ret`.
    Ret,
    /// `builtin.call_indirect`, whose operand 0 is the callee and whose new call is a
    /// `wasm.call_indirect`.
    CallIndirect,
    /// `builtin.tail_call_indirect`, whose operand 0 is the callee.
    TailCallIndirect,
}

/// `inner`, a pattern rooted at the operation `role` names that converts it by type, with the
/// rules of the calling convention on top (see the module documentation), as one pattern: the
/// rules of functions, calls and returns, shared by every pattern that lowers one.
///
/// For a call (`Role::Call`, `Role::CallIndirect`), `inner` must leave the new call, a
/// `builtin.call` or a `wasm.call_indirect`, after the operations that preceded the old one, and
/// its results read only through the casts the conversion leaves.
pub(crate) fn with_convention(
    context: &Rc<Context>,
    role: Role,
    inner: Rc<dyn ConversionPattern>,
    converter: &TypeConverter,
) -> impl ConversionPattern + 'static {
    WithConvention::new(context, role, inner, converter)
}

/// A builtin conversion pattern of a function, a call, a tail call or a return, and the rules of
/// the calling convention on top of it.
struct WithConvention {
    info: PatternInfo,
    role: Role,
    inner: Rc<dyn ConversionPattern>,
    converter: TypeConverter,
}

impl WithConvention {
    fn new(
        context: &Rc<Context>,
        role: Role,
        inner: Rc<dyn ConversionPattern>,
        converter: &TypeConverter,
    ) -> Self {
        let generated = inner.generated_ops().iter().cloned().chain(extension_ops(context));
        let info = match role {
            Role::Function => pattern_info::<Function>(context, "wasm-function", generated),
            Role::Call => pattern_info::<Call>(context, "wasm-call", generated),
            Role::TailCall => pattern_info::<TailCall>(context, "wasm-tail-call", generated),
            Role::Ret => pattern_info::<Ret>(context, "wasm-ret", generated),
            Role::CallIndirect => {
                pattern_info::<CallIndirect>(context, "wasm-call-indirect", generated)
            }
            Role::TailCallIndirect => {
                pattern_info::<TailCallIndirect>(context, "wasm-tail-call-indirect", generated)
            }
        };
        Self {
            info,
            role,
            inner,
            converter: converter.clone(),
        }
    }
}

impl Pattern for WithConvention {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for WithConvention {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        match self.role {
            Role::Function => self.function(op, operands, rewriter),
            Role::Call | Role::TailCall | Role::CallIndirect | Role::TailCallIndirect => {
                self.call(op, operands, rewriter)
            }
            Role::Ret => self.ret(op, operands, rewriter),
        }
    }
}

impl WithConvention {
    /// A function: converted by the builtin pattern, then, under `"C"`, each narrow parameter
    /// not marked `zext` masked at the entry, its users reading the masked value.
    fn function(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let (signature, entry, span) = {
            let op_ref = op.borrow();
            let function = op_ref.downcast_ref::<Function>().expect("rooted at builtin.function");
            let entry = (!function.is_declaration())
                .then(|| function.body().entry_block_ref())
                .flatten();
            (function.get_signature().clone(), entry, op_ref.span())
        };
        let convention = Convention::of(&signature, &op.borrow())?;
        let masks = match (convention, entry) {
            (Convention::C, Some(_)) => signature
                .params()
                .iter()
                .enumerate()
                .filter(|(_, param)| param.extension() != ArgumentExtension::Zext)
                .filter_map(|(index, param)| narrow(param.ty).map(|n| (index, n)))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        if !self.inner.match_and_rewrite(op, operands, rewriter)? {
            return Ok(false);
        }
        let Some(entry) = entry else {
            return Ok(true);
        };
        if masks.is_empty() {
            return Ok(true);
        }
        // Every parameter has one carrier, so parameter `i` is still argument `i`. Its users are
        // the casts back to its old type that the signature conversion left at the start of the
        // block; the masks go before them.
        let arguments = entry.borrow().argument_values().collect::<SmallVec<[ValueRef; 4]>>();
        let front = entry.borrow().front();
        let Some(front) = front else {
            return Ok(true);
        };
        rewriter.set_insertion_point_before(front);
        for (index, n) in masks {
            let argument = arguments[index];
            let users = casts_reading(argument, &op.borrow())?;
            if users.is_empty() {
                continue;
            }
            let masked = mask(rewriter, argument, n, span)?;
            for cast in users {
                rewriter.replace_op_with_multiple(cast, &[&[masked]])?;
            }
        }
        Ok(true)
    }

    /// A call or a tail call, direct or indirect: under `"C"`, each argument whose parameter is
    /// narrow and marked `sext` sign-extended before the inner pattern converts the call, and each
    /// narrow result of a call that is not marked `zext` masked after it, its users reading the
    /// masked value.
    fn call(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let (signature, span) = {
            let op_ref = op.borrow();
            let signature = match self.role {
                Role::Call => op_ref.downcast_ref::<Call>().map(|op| op.get_signature().clone()),
                Role::TailCall => {
                    op_ref.downcast_ref::<TailCall>().map(|op| op.get_signature().clone())
                }
                Role::CallIndirect => {
                    op_ref.downcast_ref::<CallIndirect>().map(|op| op.get_signature().clone())
                }
                _ => op_ref.downcast_ref::<TailCallIndirect>().map(|op| op.get_signature().clone()),
            };
            (signature.expect("rooted at a call"), op_ref.span())
        };
        // The callee of an indirect call is its operand 0, before the arguments.
        let first_argument = match self.role {
            Role::CallIndirect | Role::TailCallIndirect => 1,
            _ => 0,
        };
        let convention = Convention::of(&signature, &op.borrow())?;
        if convention == Convention::Wasm {
            return self.inner.match_and_rewrite(op, operands, rewriter);
        }
        let extend = signature
            .params()
            .iter()
            .enumerate()
            .filter(|(_, param)| param.extension() == ArgumentExtension::Sext)
            .filter_map(|(index, param)| narrow(param.ty).map(|n| (first_argument + index, n)))
            .collect::<Vec<_>>();
        let masks = match self.role {
            Role::Call | Role::CallIndirect => signature
                .results()
                .iter()
                .enumerate()
                .filter(|(_, result)| result.extension() != ArgumentExtension::Zext)
                .filter_map(|(index, result)| narrow(result.ty).map(|n| (index, n)))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        if extend.is_empty() && masks.is_empty() {
            return self.inner.match_and_rewrite(op, operands, rewriter);
        }

        let before = op.prev();
        let block = op.parent().expect("a call is in a block");
        rewriter.set_insertion_point_before(op);
        let (ranges, groups) = extended_operands(rewriter, &operands, &extend, span)?;
        let extended = ConvertedOperands::new(&ranges, &groups);
        if !self.inner.match_and_rewrite(op, extended, rewriter)? {
            return Err(unconverted(&op.borrow()));
        }
        if masks.is_empty() {
            return Ok(true);
        }
        // The new call is after what preceded the old one and the sign extensions.
        let mut next = match before {
            Some(before) => before.next(),
            None => block.borrow().front(),
        };
        let call = loop {
            let Some(candidate) = next else {
                return Err(Report::msg(
                    "`convert-to-wasm` did not find the call the call pattern built",
                ));
            };
            let found = match self.role {
                Role::CallIndirect => candidate.borrow().is::<wasm::CallIndirect>(),
                _ => candidate.borrow().is::<Call>(),
            };
            if found {
                break candidate;
            }
            next = candidate.next();
        };
        let results = call
            .borrow()
            .results()
            .iter()
            .map(|result| result.borrow().as_value_ref())
            .collect::<SmallVec<[ValueRef; 2]>>();
        rewriter.set_insertion_point_after(call);
        for (index, n) in masks {
            let result = results[index];
            let users = casts_reading(result, &call.borrow())?;
            if users.is_empty() {
                continue;
            }
            let masked = mask(rewriter, result, n, span)?;
            for cast in users {
                rewriter.replace_op_with_multiple(cast, &[&[masked]])?;
            }
        }
        Ok(true)
    }

    /// A return: under `"C"`, each narrow result of the function marked `sext` sign-extended
    /// before the builtin pattern converts the return.
    fn ret(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let extend = sign_extended_results(&op.borrow())?;
        let extend = match extend {
            Some(extend) if !extend.is_empty() => extend,
            _ => return self.inner.match_and_rewrite(op, operands, rewriter),
        };
        let span = op.borrow().span();
        rewriter.set_insertion_point_before(op);
        let (ranges, groups) = extended_operands(rewriter, &operands, &extend, span)?;
        let extended = ConvertedOperands::new(&ranges, &groups);
        if !self.inner.match_and_rewrite(op, extended, rewriter)? {
            return Err(unconverted(&op.borrow()));
        }
        Ok(true)
    }
}

/// For a return `op` (a `builtin.ret` or a `builtin.ret_imm`, whose values have their old types):
/// the index and width of each value that is narrow and that the enclosing function, under `"C"`,
/// marks `sext`. `None` when the function is not under `"C"` or `op` is in no function.
fn sign_extended_results(op: &Operation) -> Result<Option<Vec<(usize, u32)>>, Report> {
    let Some(function) = op.nearest_parent_op::<Function>() else {
        return Ok(None);
    };
    let signature = function.borrow().get_signature().clone();
    if Convention::of(&signature, function.borrow().as_operation())? != Convention::C {
        return Ok(None);
    }
    let types: SmallVec<[Type; 2]> = match op.downcast_ref::<RetImm>() {
        Some(ret) => SmallVec::from_slice(&[runic_ir::Attribute::ty(&*ret.value())]),
        None => op.operands().iter().map(|operand| operand.borrow().ty()).collect(),
    };
    Ok(Some(
        types
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                signature
                    .results()
                    .get(*index)
                    .is_some_and(|result| result.extension() == ArgumentExtension::Sext)
            })
            .filter_map(|(index, ty)| narrow(*ty).map(|n| (index, n)))
            .collect(),
    ))
}

/// One range of values per operand.
type OperandRanges = Vec<SmallVec<[ValueRef; 2]>>;

/// The operands as `operands` gives them, with operand `i` sign-extended from its low `n` bits for
/// each `(i, n)` of `extend`; and the sizes of the operand groups.
fn extended_operands(
    rewriter: &mut ConversionPatternRewriter,
    operands: &ConvertedOperands<'_>,
    extend: &[(usize, u32)],
    span: runic_ir::SourceSpan,
) -> Result<(OperandRanges, SmallVec<[usize; 4]>), Report> {
    let mut ranges = operands.iter().map(SmallVec::<[ValueRef; 2]>::from_slice).collect::<Vec<_>>();
    for (index, n) in extend {
        let Some(value) = operands.one(*index) else {
            return Err(Report::msg(format!(
                "operand {index} became {} values, but a narrow integer is carried in one",
                operands.get(*index).len()
            )));
        };
        ranges[*index] = SmallVec::from_slice(&[sign_extend(rewriter, value, *n, span)?]);
    }
    let groups = (0..operands.num_groups()).map(|group| operands.group(group).count()).collect();
    Ok((ranges, groups))
}

/// The users of `value`, which must all be `builtin.unrealized_conversion_cast`s (the casts back
/// to its old type that the conversion left); `owner` is what defines it, for the error.
fn casts_reading(
    value: ValueRef,
    owner: &Operation,
) -> Result<SmallVec<[OperationRef; 2]>, Report> {
    let mut casts = SmallVec::new();
    for user in value.borrow().iter_uses() {
        let user = user.owner;
        if !user.borrow().is::<UnrealizedConversionCast>() {
            return Err(Report::msg(format!(
                "`convert-to-wasm` expected only casts to read a value of `{}` it converted, but \
                 `{}` reads it",
                owner.name(),
                user.borrow().name()
            )));
        }
        if !casts.contains(&user) {
            casts.push(user);
        }
    }
    Ok(casts)
}

/// The error of a pattern that changed the IR (the extensions) and whose builtin pattern then did
/// not convert `op`.
fn unconverted(op: &Operation) -> Report {
    Report::msg(format!(
        "`convert-to-wasm` extended the operands of `{}`, but the builtin pattern did not convert \
         it",
        op.name()
    ))
}

/// `builtin.ret_imm V : T`: a `wasm.const` of `V` in `T`'s carrier (sign-extended where the
/// function is under `"C"` and marks a narrow result `sext`, zero-extended otherwise) and a
/// `builtin.ret` of it.
struct RetImmPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for RetImmPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for RetImmPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let (value, ty, span) = {
            let op_ref = op.borrow();
            let ret = op_ref.downcast_ref::<RetImm>().expect("rooted at builtin.ret_imm");
            (*ret.get_value(), runic_ir::Attribute::ty(&*ret.value()), op_ref.span())
        };
        let Some(carrier) = carrier(rewriter.context(), ty) else {
            return Ok(false);
        };
        let Some(carrier_bits) = carrier_width(carrier) else {
            return Ok(false);
        };
        let sign_extended = sign_extended_results(&op.borrow())?
            .is_some_and(|extend| extend.iter().any(|(index, _)| *index == 0));
        let bits = if sign_extended {
            value.sext(carrier_bits).unsigned()
        } else {
            value.unsigned()
        };
        rewriter.set_insertion_point_before(op);
        let constant = const_int(rewriter, bits as u64, carrier, span)?;
        rewriter.create_op::<Ret, (Vec<ValueRef>,)>(span, (alloc::vec![constant],))?;
        rewriter.erase_op(op)?;
        Ok(true)
    }
}

/// `arith.constant V : T`: a `wasm.const` of `V` zero-extended in `T`'s carrier.
struct ConstantPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for ConstantPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for ConstantPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let (value, ty, span) = {
            let op_ref = op.borrow();
            let constant =
                op_ref.downcast_ref::<arith::Constant>().expect("rooted at arith.constant");
            (*constant.get_value(), constant.result().ty(), op_ref.span())
        };
        let Some(carrier) = carrier(rewriter.context(), ty) else {
            return Ok(false);
        };
        rewriter.set_insertion_point_before(op);
        let constant = const_int(rewriter, value.unsigned() as u64, carrier, span)?;
        rewriter.replace_op(op, &[constant])?;
        Ok(true)
    }
}

/// `arith.constantf V : T`: a `wasm.constf` of `V`.
struct ConstantFPattern {
    info: PatternInfo,
    converter: TypeConverter,
}

impl Pattern for ConstantFPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for ConstantFPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        _operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let (value, span) = {
            let op_ref = op.borrow();
            let constant =
                op_ref.downcast_ref::<arith::Constantf>().expect("rooted at arith.constantf");
            (*constant.get_value(), op_ref.span())
        };
        rewriter.set_insertion_point_before(op);
        let constant = carried::result(&rewriter.create_op::<wasm::ConstF, _>(span, (value,))?);
        rewriter.replace_op(op, &[constant])?;
        Ok(true)
    }
}
