use alloc::{format, rc::Rc};

use runic_ir::{
    derive::{EffectOpInterface, operation},
    dialects::builtin::attributes::TypeAttr,
    effects::*,
    formatter::const_text,
    parse::ParseResult,
    patterns::RewritePatternSet,
    traits::*,
    *,
};

use super::access::{invalid, print_attributes};
use crate::MemDialect;

/// The address `addr` as an `isize`: `%a = mem.ptrtoint %p;`.
///
/// The pointer may have any pointer type, typed or opaque, in any address space. Other integer
/// widths are reached through `arith`. The widths are those of the data layout: the pointer size
/// of the pointer's address space and the layout's `isize`. The address is zero-extended when the
/// `isize` is wider than the pointer and truncated when it is narrower. The operation has no
/// effects.
///
/// `mem.ptrtoint` of a `mem.inttoptr` of `x` folds to `x` when a data layout is in scope
/// ([DataLayout::closest]) and its pointer of that address space is at least as wide as its
/// `isize`; otherwise the pair is kept.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    name = "ptrtoint",
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct PtrToInt {
    /// The pointer converted
    #[operand]
    addr: AnyPointer,
    /// The address as an integer
    #[result]
    result: Isize,
}

/// The `isize` `value` as a pointer of the type `ty`: `%p = mem.inttoptr %a : ptr<i32>;`.
///
/// The type is written in the text and kept in the hidden property `ty`, from which the result is
/// made; the verifier rejects a `ty` that is not a pointer type and a result whose type is not
/// `ty`. The widths are those of the data layout: the layout's `isize` and the pointer size of
/// `ty`'s address space. The value is zero-extended when the pointer is wider than the `isize` and
/// truncated when it is narrower. The operation has no effects.
///
/// `mem.inttoptr` of a `mem.ptrtoint` of `p` folds to `p` when the result has `p`'s type, a data
/// layout is in scope ([DataLayout::closest]) and its `isize` is at least as wide as its pointer
/// of that address space; otherwise, and to another pointer type, the pair is kept.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    name = "inttoptr",
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct IntToPtr {
    /// The address converted
    #[operand]
    value: Isize,
    /// The pointer type of the result
    #[attr(hidden)]
    ty: TypeAttr,
    /// The address as a pointer. Any type to the constraint, so that a `ty` that is not a pointer
    /// type reaches the verifier, which says so
    #[result]
    result: AnyType,
}

/// The pointer `addr` as a pointer of the type `ty`, in the same address space:
/// `%q = mem.ptrcast %p : ptr<i8>;`.
///
/// Either pointer may be typed or opaque: a cast between a typed and an opaque pointer of one
/// address space is allowed in both directions. The type is written in the text and kept in the
/// hidden property `ty`, from which the result is made; the verifier rejects a `ty` that is not a
/// pointer type, a `ty` in another address space than `addr`, and a result whose type is not `ty`.
///
/// The cast has no effects and does not change the address. A cast to `addr`'s own type folds to
/// `addr`, and a cast of a cast becomes one cast
/// ([MergePtrCastOfPtrCast](crate::canonicalization::MergePtrCastOfPtrCast)). Nothing reasons
/// about aliases: a load through the cast and a load through `addr` are loads of two values.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    name = "ptrcast",
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct PtrCast {
    /// The pointer cast
    #[operand]
    addr: AnyPointer,
    /// The pointer type of the result
    #[attr(hidden)]
    ty: TypeAttr,
    /// The pointer as the type `ty`. Any type to the constraint, so that a `ty` that is not a
    /// pointer type reaches the verifier, which says so
    #[result]
    result: AnyType,
}

/// A result whose type is not yet known is an `isize`.
impl InferTypeOpInterface for PtrToInt {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        if self.result().ty().is_unknown() {
            self.result_mut().set_type(context.isize());
        }
        Ok(())
    }
}

/// `mem.ptrtoint (mem.inttoptr %a)` is `%a` when the layout's pointer holds every bit of its
/// `isize`.
impl Foldable for PtrToInt {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let Some(defining_op) = self.addr().as_value_ref().borrow().get_defining_op() else {
            return FoldResult::Failed;
        };
        let defining_op = defining_op.borrow();
        let Some(inner) = defining_op.downcast_ref::<IntToPtr>() else {
            return FoldResult::Failed;
        };
        let value = inner.value().as_value_ref();
        if value.borrow().ty() != self.result().ty() {
            return FoldResult::Failed;
        }
        // The pointer between the two casts must hold every bit of an `isize`
        match cast_widths(self.as_operation(), self.addr().ty()) {
            Some((pointer, isize)) if pointer >= isize => (),
            _ => return FoldResult::Failed,
        }
        results.push(OpFoldResult::Value(value));
        FoldResult::Ok(())
    }

    fn fold_with(
        &self,
        _operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        self.fold(results)
    }
}

/// A result whose type is not yet known gets the type `ty`. A result that already has a type (the
/// generic form writes one) keeps it, and the verifier checks it.
impl InferTypeOpInterface for IntToPtr {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        if self.result().ty().is_unknown() {
            let ty = *self.get_ty();
            self.result_mut().set_type(ty);
        }
        Ok(())
    }
}

/// `ty` is a pointer type, and the result has the type `ty`.
impl Verify<dyn InferTypeOpInterface> for IntToPtr {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let stated = *self.get_ty();
        verify_stated_pointer(op, stated, self.result().ty())
    }
}

/// `mem.inttoptr (mem.ptrtoint %p)` is `%p` when the result has `%p`'s type and the layout's
/// `isize` holds every bit of the pointer.
impl Foldable for IntToPtr {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let Some(defining_op) = self.value().as_value_ref().borrow().get_defining_op() else {
            return FoldResult::Failed;
        };
        let defining_op = defining_op.borrow();
        let Some(inner) = defining_op.downcast_ref::<PtrToInt>() else {
            return FoldResult::Failed;
        };
        let addr = inner.addr().as_value_ref();
        if addr.borrow().ty() != self.result().ty() {
            return FoldResult::Failed;
        }
        // The `isize` between the two casts must hold every bit of the pointer
        match cast_widths(self.as_operation(), self.result().ty()) {
            Some((pointer, isize)) if isize >= pointer => (),
            _ => return FoldResult::Failed,
        }
        results.push(OpFoldResult::Value(addr));
        FoldResult::Ok(())
    }

    fn fold_with(
        &self,
        _operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        self.fold(results)
    }
}

/// A result whose type is not yet known gets the type `ty`. A result that already has a type (the
/// generic form writes one) keeps it, and the verifier checks it.
impl InferTypeOpInterface for PtrCast {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        if self.result().ty().is_unknown() {
            let ty = *self.get_ty();
            self.result_mut().set_type(ty);
        }
        Ok(())
    }
}

/// `ty` is a pointer type in the address space of the operand, and the result has the type `ty`.
impl Verify<dyn InferTypeOpInterface> for PtrCast {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let stated = *self.get_ty();
        verify_stated_pointer(op, stated, self.result().ty())?;
        let from = self.addr().ty();
        let (Some(source), Some(target)) = (from.as_pointer(), stated.as_pointer()) else {
            return Ok(());
        };
        if source.addrspace() != target.addrspace() {
            return Err(invalid(
                op,
                format!(
                    "the operand `{from}` is in address space {}, but the cast to `{stated}` is \
                     in address space {}: a cast keeps the address space",
                    source.addrspace(),
                    target.addrspace()
                ),
            ));
        }
        Ok(())
    }
}

/// `mem.ptrcast %p` to `%p`'s own type is `%p`.
impl Foldable for PtrCast {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let addr = self.addr().as_value_ref();
        if addr.borrow().ty() != self.result().ty() {
            return FoldResult::Failed;
        }
        results.push(OpFoldResult::Value(addr));
        FoldResult::Ok(())
    }

    fn fold_with(
        &self,
        _operands: &[Option<AttributeRef>],
        results: &mut SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        self.fold(results)
    }
}

/// A cast of a cast becomes one cast
/// ([MergePtrCastOfPtrCast](crate::canonicalization::MergePtrCastOfPtrCast)).
impl Canonicalizable for PtrCast {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(crate::canonicalization::MergePtrCastOfPtrCast::new(context));
    }
}

/// The widths in bits of a pointer of the type `pointer` and of an `isize`, under the data layout
/// in scope at `op`. `None` with no layout in scope, or if `pointer` is not a pointer type: a fold
/// that depends on the widths then does not apply.
fn cast_widths(op: &Operation, pointer: Type) -> Option<(u32, u32)> {
    let layout = DataLayout::closest(op)?;
    let addrspace = pointer.as_pointer()?.addrspace();
    Some((layout.pointer(addrspace).size_bits, layout.isize_width()))
}

/// The stated type `stated` of a cast is a pointer type, and the result has it.
fn verify_stated_pointer(op: &Operation, stated: Type, result: Type) -> Result<(), Report> {
    if !stated.is_pointer() {
        return Err(invalid(
            op,
            format!("the result type of a cast to a pointer is a pointer type, not `{stated}`"),
        ));
    }
    if result != stated {
        return Err(invalid(
            op,
            format!("the result has type `{result}`, but the cast is to `{stated}`"),
        ));
    }
    Ok(())
}

impl OpPrinter for PtrToInt {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        let op = self.as_operation();
        printer.print_space();
        printer.print_value_uses(op.operands().group(0).as_value_range());
        print_attributes(printer, op);
    }
}

/// Reads `%p` and an optional `attributes {..}`. The result is an `isize`.
impl OpParser for PtrToInt {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let addr = parser.parse_operand(/*allow_result_number=*/ true)?;
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

        let unknown = parser.context().unknown_type();
        let addr = parser.resolve_operand(addr, unknown)?;
        state.add_operand(addr);
        state.attrs.extend(attrs);
        state.results.push(parser.context().isize());
        Ok(())
    }
}

impl OpPrinter for IntToPtr {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        print_cast(printer, self.as_operation(), *self.get_ty());
    }
}

/// Reads `%a : type` and an optional `attributes {..}`. The result has the written type.
impl OpParser for IntToPtr {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_cast(state, parser)
    }
}

impl OpPrinter for PtrCast {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        print_cast(printer, self.as_operation(), *self.get_ty());
    }
}

/// Reads `%p : type` and an optional `attributes {..}`. The result has the written type.
impl OpParser for PtrCast {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_cast(state, parser)
    }
}

/// ` %v : ty`, then the attributes of `op`.
fn print_cast(printer: &mut print::AsmPrinter<'_>, op: &Operation, ty: Type) {
    printer.print_space();
    printer.print_value_uses(op.operands().group(0).as_value_range());
    *printer += const_text(" : ");
    printer.print_type(ty);
    print_attributes(printer, op);
}

/// Reads `%v : type` and an optional `attributes {..}`; the type is the property `ty`, and the
/// result's type is inferred from it.
fn parse_cast(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
    let operand = parser.parse_operand(/*allow_result_number=*/ true)?;
    parser.parse_colon()?;
    let ty = parser.parse_type()?.into_inner();
    let mut attrs = SmallVec::default();
    parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

    let unknown = parser.context().unknown_type();
    let operand = parser.resolve_operand(operand, unknown)?;
    state.add_operand(operand);
    state.add_attribute("ty", parser.context_rc().create_attribute::<TypeAttr, _>(ty));
    state.attrs.extend(attrs);
    state.results.push(unknown);
    Ok(())
}
