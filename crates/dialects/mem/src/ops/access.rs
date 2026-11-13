use alloc::{format, string::String};

use runic_core::{
    derive::{EffectOpInterface, operation},
    diagnostics::report,
    dialects::builtin::attributes::TypeAttr,
    effects::*,
    formatter::const_text,
    parse::ParseResult,
    traits::*,
    *,
};

use crate::{Alignment, AlignmentAttr, MemDialect, attributes::parse_align_bytes};

/// The byte-addressed memory that `mem.load` reads and `mem.store` writes: a child of the default
/// resource, so an access is ordered against every effect on the default resource (a call with
/// unknown effects, say) and against no effect on another of its children (a `cf.assert`).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MemoryResource;

impl Resource for MemoryResource {
    fn name(&self) -> &'static str {
        "mem.memory"
    }
}

/// A read of a value of the type the pointer `addr` points to: `%v = mem.load %p;`.
///
/// Through an opaque pointer the type read is written, and kept in the hidden property `ty`:
/// `%v = mem.load %p : i32;`. Through a typed pointer `ty` is left unknown, or, when a type is
/// written anyway, must be the pointee. The result has the type read. `, align N` states the
/// alignment of the address in bytes; without it the access assumes the alignment the data layout
/// gives the type read.
///
/// The load reads [MemoryResource] at `addr`, and has no other effect: an address that is invalid,
/// or not aligned as the load assumes, is undefined behaviour of the access, so an unused load is
/// removed and two equal loads with no write between them are merged. When a data layout is in
/// scope ([DataLayout::closest]), the type read must have a layout under it.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    implements(
        InferTypeOpInterface,
        MemoryEffectOpInterface,
        OperandRangeRequirementOpInterface,
        OpPrinter
    )
)]
pub struct Load {
    /// The address read
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Read), resource(MemoryResource))]
    addr: AnyPointer,
    /// The type read through an opaque pointer; unknown when the pointer is typed
    #[attr(hidden)]
    #[default]
    ty: TypeAttr,
    /// The alignment the load assumes of `addr`
    #[attr(hidden)]
    #[default]
    align: AlignmentAttr,
    /// The value read
    #[result]
    result: AnyType,
}

impl Load {
    /// The type the load reads: the pointee of a typed pointer, or `ty` for an opaque one. `None`
    /// for an opaque pointer whose `ty` is unknown.
    pub fn loaded_type(&self) -> Option<Type> {
        self.addr().ty().pointee().or_else(|| {
            let ty = *self.get_ty();
            (!ty.is_unknown()).then_some(ty)
        })
    }
}

/// A result whose type is not yet known gets the type read: the pointee of a typed pointer, or
/// `ty` for an opaque one. A result that already has a type (the generic form writes one) keeps
/// it, and the verifier checks it.
impl InferTypeOpInterface for Load {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        if !self.result().ty().is_unknown() {
            return Ok(());
        }
        if let Some(ty) = self.loaded_type() {
            self.result_mut().set_type(ty);
        }
        Ok(())
    }
}

/// A load through an opaque pointer states the type it reads; through a typed pointer a stated
/// type is the pointee; the result has the type read.
impl Verify<dyn InferTypeOpInterface> for Load {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let pointer = self.addr().ty();
        let stated = *self.get_ty();
        let result = self.result().ty();
        let problem = match pointer.pointee() {
            Some(pointee) if !stated.is_unknown() && stated != pointee => Some(format!(
                "the pointer points to `{pointee}`, but the load states that it reads `{stated}`"
            )),
            None if stated.is_unknown() => Some(format!(
                "a load through the opaque pointer type `{pointer}` must state the type it reads \
                 (`mem.load %p : i32`)"
            )),
            _ => None,
        };
        let problem = problem.or_else(|| {
            let read = self.loaded_type()?;
            (result != read)
                .then(|| format!("the result has type `{result}`, but the load reads `{read}`"))
        });
        match problem {
            None => Ok(()),
            Some(problem) => Err(invalid(op, problem)),
        }
    }
}

/// When a data layout is in scope, the type read has a layout under it.
impl Verify<dyn MemoryEffectOpInterface> for Load {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        verify_layout(self.as_operation(), self.result().ty(), "the type read")
    }
}

/// Whether an address is valid is not a range of it.
impl OperandRangeRequirementOpInterface for Load {
    fn operand_range_requirement(&self, index: usize) -> OperandRangeRequirement {
        match index {
            0 => OperandRangeRequirement::Unexpressed,
            _ => OperandRangeRequirement::None,
        }
    }
}

impl OpPrinter for Load {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        let op = self.as_operation();
        printer.print_space();
        printer.print_value_uses(op.operands().group(0).as_value_range());
        print_alignment(printer, *self.get_align());
        let stated = *self.get_ty();
        if self.addr().ty().pointee().is_none() && !stated.is_unknown() {
            *printer += const_text(" : ");
            printer.print_type(stated);
        }
        print_attributes(printer, op);
    }
}

/// Reads `%p`, then an optional `, align N`, an optional `: type` and an optional
/// `attributes {..}`. The result's type is inferred.
impl OpParser for Load {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let addr = parser.parse_operand(/*allow_result_number=*/ true)?;
        let align = parse_optional_alignment(parser)?;
        let ty = if parser.parse_optional_colon()? {
            Some(parser.parse_type()?.into_inner())
        } else {
            None
        };
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

        // The address is a use of a value defined earlier, which carries its type, as in the
        // derived format of the other operations.
        let unknown = parser.context().unknown_type();
        let addr = parser.resolve_operand(addr, unknown)?;
        state.add_operand(addr);
        if let Some(ty) = ty {
            state.add_attribute("ty", parser.context_rc().create_attribute::<TypeAttr, _>(ty));
        }
        add_alignment(state, parser, align);
        state.attrs.extend(attrs);
        state.results.push(unknown);
        Ok(())
    }
}

/// A write of `value` to the pointer `addr`: `mem.store %v, %p;`.
///
/// Through a typed pointer, `value` has the pointee's type; through an opaque pointer it may have
/// any type. `, align N` states the alignment of the address in bytes; without it the access
/// assumes the alignment the data layout gives the value's type.
///
/// The store writes [MemoryResource] at `addr`, so it is never removed, and no read of that memory
/// moves or merges across it. When a data layout is in scope ([DataLayout::closest]), the value's
/// type must have a layout under it.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    implements(MemoryEffectOpInterface, OperandRangeRequirementOpInterface, OpPrinter)
)]
pub struct Store {
    /// The value written
    #[operand]
    value: AnyType,
    /// The address written
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(MemoryResource))]
    addr: AnyPointer,
    /// The alignment the store assumes of `addr`
    #[attr(hidden)]
    #[default]
    align: AlignmentAttr,
}

/// Through a typed pointer the value has the pointee's type; when a data layout is in scope, the
/// value's type has a layout under it.
impl Verify<dyn MemoryEffectOpInterface> for Store {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let value = self.value().ty();
        if let Some(pointee) = self.addr().ty().pointee()
            && pointee != value
        {
            return Err(invalid(
                op,
                format!("the value has type `{value}`, but the pointer points to `{pointee}`"),
            ));
        }
        verify_layout(op, value, "the type written")
    }
}

/// Whether an address is valid is not a range of it; every value may be stored.
impl OperandRangeRequirementOpInterface for Store {
    fn operand_range_requirement(&self, index: usize) -> OperandRangeRequirement {
        match index {
            1 => OperandRangeRequirement::Unexpressed,
            _ => OperandRangeRequirement::None,
        }
    }
}

impl OpPrinter for Store {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        let op = self.as_operation();
        printer.print_space();
        printer.print_value_uses(op.operands().group(0).as_value_range());
        print_alignment(printer, *self.get_align());
        print_attributes(printer, op);
    }
}

/// Reads `%v, %p`, then an optional `, align N` and an optional `attributes {..}`.
impl OpParser for Store {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let value = parser.parse_operand(/*allow_result_number=*/ true)?;
        parser.parse_comma()?;
        let addr = parser.parse_operand(/*allow_result_number=*/ true)?;
        let align = parse_optional_alignment(parser)?;
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

        let unknown = parser.context().unknown_type();
        let value = parser.resolve_operand(value, unknown)?;
        let addr = parser.resolve_operand(addr, unknown)?;
        state.add_operand(value);
        state.add_operand(addr);
        add_alignment(state, parser, align);
        state.attrs.extend(attrs);
        Ok(())
    }
}

/// Storage for a value of type `ty`, which lives until the enclosing function returns:
/// `%p = mem.alloca i32;`. The result is a `ptr<ty>` in address space 0. `, align N` states the
/// alignment of the storage in bytes; without it the storage has the alignment the data layout
/// gives `ty`.
///
/// The storage is uninitialized. The operation allocates its result in
/// [AutomaticAllocationScopeResource] and has no other effect, so an unused one is removed. When a
/// data layout is in scope ([DataLayout::closest]), `ty` must have a layout under it.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Alloca {
    /// The type of the value the storage holds
    #[attr(hidden)]
    ty: TypeAttr,
    /// The alignment of the storage
    #[attr(hidden)]
    #[default]
    align: AlignmentAttr,
    /// The address of the storage
    #[result]
    #[effects(
        MemoryEffect(MemoryEffect::Allocate),
        resource(AutomaticAllocationScopeResource)
    )]
    result: AnyPointer,
}

/// A result whose type is not yet known gets the type `ptr<ty>`. A result that already has a type
/// (the generic form writes one) keeps it, and the verifier checks it.
impl InferTypeOpInterface for Alloca {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        if !self.result().ty().is_unknown() {
            return Ok(());
        }
        let ty = context.ptr_type(*self.get_ty());
        self.result_mut().set_type(ty);
        Ok(())
    }
}

/// The allocated type is known, and the result is a pointer to it in address space 0.
impl Verify<dyn InferTypeOpInterface> for Alloca {
    fn verify(&self, context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let ty = *self.get_ty();
        if ty.is_unknown() {
            return Err(invalid(op, String::from("the allocated type is not known")));
        }
        let expected = context.ptr_type(ty);
        let result = self.result().ty();
        if result != expected {
            return Err(invalid(
                op,
                format!(
                    "the result has type `{result}`, but an alloca of `{ty}` gives `{expected}`"
                ),
            ));
        }
        Ok(())
    }
}

/// When a data layout is in scope, the allocated type has a layout under it.
impl Verify<dyn MemoryEffectOpInterface> for Alloca {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        verify_layout(self.as_operation(), *self.get_ty(), "the allocated type")
    }
}

impl OpPrinter for Alloca {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        printer.print_space();
        printer.print_type(*self.get_ty());
        print_alignment(printer, *self.get_align());
        print_attributes(printer, self.as_operation());
    }
}

/// Reads a type, then an optional `, align N` and an optional `attributes {..}`.
impl OpParser for Alloca {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let ty = parser.parse_type()?.into_inner();
        let align = parse_optional_alignment(parser)?;
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

        state.add_attribute("ty", parser.context_rc().create_attribute::<TypeAttr, _>(ty));
        add_alignment(state, parser, align);
        state.attrs.extend(attrs);
        state.results.push(parser.context().unknown_type());
        Ok(())
    }
}

/// The diagnostic of an invalid `op`, with `problem` as the label of the operation.
pub(crate) fn invalid(op: &Operation, problem: String) -> Report {
    report!(
        message: (format!("invalid operation {}", op.name())),
        labels: [primary(op.span(), problem)],
    )
}

/// When a data layout is in scope at `op`, `ty` (described as `what`) has a layout under it.
pub(crate) fn verify_layout(op: &Operation, ty: Type, what: &str) -> Result<(), Report> {
    let Some(layout) = DataLayout::closest(op) else {
        return Ok(());
    };
    layout.layout_of(ty).map(|_| ()).map_err(|error| {
        invalid(
            op,
            format!("{what}, `{ty}`, has no layout under the data layout in scope: {error}"),
        )
    })
}

/// `, align N` when `align` is not natural.
fn print_alignment(printer: &mut print::AsmPrinter<'_>, align: Alignment) {
    if let Alignment::Bytes(align) = align {
        *printer += const_text(", ");
        printer.print_keyword("align");
        printer.print_space();
        printer.print_decimal_integer(align.bytes());
    }
}

/// ` attributes {..}` when `op` has attributes that are not properties.
pub(crate) fn print_attributes(printer: &mut print::AsmPrinter<'_>, op: &Operation) {
    if op.has_attributes() {
        printer.print_space();
        printer.print_keyword("attributes");
        printer.print_space();
        printer.print_attribute_dictionary(
            op.attributes().iter().map(|attr| *attr.as_named_attribute()),
        );
    }
}

/// Reads an optional `, align N`; without one the alignment is natural.
fn parse_optional_alignment(parser: &mut dyn OpAsmParser<'_>) -> ParseResult<Alignment> {
    if !parser.parse_optional_comma()? {
        return Ok(Alignment::Natural);
    }
    parser.parse_custom_keyword("align")?;
    Ok(Alignment::Bytes(parse_align_bytes(parser)?))
}

/// Gives `state` the property `align` when `align` is not natural, the property's default.
fn add_alignment(state: &mut OperationState, parser: &dyn OpAsmParser<'_>, align: Alignment) {
    if !align.is_natural() {
        state.add_attribute(
            "align",
            parser.context_rc().create_attribute::<AlignmentAttr, _>(align),
        );
    }
}
