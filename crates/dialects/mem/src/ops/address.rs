use alloc::{format, rc::Rc, string::String};

use runic_core::{
    derive::{EffectOpInterface, operation},
    dialects::builtin::attributes::{Array, I32ArrayAttr},
    effects::*,
    formatter::const_text,
    parse::{Delimiter, ParseResult, ParserExt},
    patterns::RewritePatternSet,
    traits::*,
    *,
};

use super::access::{invalid, print_attributes};
use crate::MemDialect;

/// The address of an element of the memory the typed pointer `base` points to:
/// `%q = mem.gep %p, %i, [1, 2];` is `&p[i].1.2`.
///
/// The optional `index`, an `isize`, steps over whole values of the pointee `T` of `base`, as
/// `p + i` does in C; then the constant `path` selects an element inside the value reached, one
/// step per aggregate, as [AggregateType] numbers its elements (an enum's element 0 is its
/// discriminant and element `1 + v` the payload of variant `v`). Either may be left out:
/// `mem.gep %p, [1]`, `mem.gep %p, %i`, and `mem.gep %p`, which is `%p`. The result is a pointer to
/// the element selected ([element_at] of `T` and `path`) in the address space of `base`.
///
/// The address is computed, never accessed: the gep has no effects, and address arithmetic wraps.
/// An opaque base must be cast to a typed pointer first. Every step of the path is
/// non-negative.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Gep {
    /// The pointer the address is computed from
    #[operand]
    base: AnyPointer,
    /// The number of whole pointees to step over: none, or one `isize`
    #[operands]
    index: Isize,
    /// The elements selected inside the pointee, outermost first
    #[attr(hidden)]
    #[default]
    path: I32ArrayAttr,
    /// The address of the element selected. Any type to the constraint, so that a gep whose type
    /// cannot be inferred reaches its verifier, which says why
    #[result]
    result: AnyType,
}

impl Gep {
    /// The dynamic index, if the gep has one.
    pub fn dynamic_index(&self) -> Option<ValueRef> {
        self.operands()
            .group(1)
            .iter()
            .next()
            .map(|operand| operand.borrow().as_value_ref())
    }

    /// The steps of the path, or `None` if one is negative.
    pub fn steps(&self) -> Option<SmallVec<[u32; 4]>> {
        self.get_path().iter().map(|step| u32::try_from(*step).ok()).collect()
    }

    /// The type of the result: a pointer to the element the path selects in the pointee of the
    /// base, in the base's address space. `None` if the base is opaque, a step is negative, or the
    /// path selects no element.
    pub fn selected_type(&self, context: &Context) -> Option<Type> {
        let base = self.base().ty();
        let pointer = base.as_pointer()?;
        let element = element_at(pointer.pointee()?, &self.steps()?).ok()?;
        Some(context.ptr_type_in(Some(element), pointer.addrspace()))
    }
}

/// A result whose type is not yet known gets a pointer to the element selected, when there is
/// one. A result that already has a type (the generic form writes one) keeps it, and the verifier
/// checks it.
impl InferTypeOpInterface for Gep {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        if !self.result().ty().is_unknown() {
            return Ok(());
        }
        if let Some(ty) = self.selected_type(context) {
            self.result_mut().set_type(ty);
        }
        Ok(())
    }
}

/// At most one index; a typed base; a path of non-negative steps that selects an element of the
/// pointee; a result that points to it in the base's address space.
impl Verify<dyn InferTypeOpInterface> for Gep {
    fn verify(&self, context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let indices = self.operands().group(1).len();
        if indices > 1 {
            return Err(invalid(
                op,
                format!("a gep has at most one index, but this one has {indices}"),
            ));
        }
        let base = self.base().ty();
        let Some(pointee) = base.pointee() else {
            return Err(invalid(
                op,
                format!(
                    "the base has the opaque pointer type `{base}`, but a gep steps through a \
                     typed pointer: cast the base to one first"
                ),
            ));
        };
        let path = self.get_path();
        if let Some(step) = path.iter().find(|step| **step < 0) {
            return Err(invalid(
                op,
                format!("the path {} has the negative step {step}", path_text(path.iter())),
            ));
        }
        let steps = self.steps().expect("no step is negative");
        if let Err(error) = element_at(pointee, &steps) {
            return Err(invalid(
                op,
                format!(
                    "the path {} selects no element of `{pointee}`: {error}",
                    path_text(path.iter())
                ),
            ));
        }
        let expected = self.selected_type(context).expect("the path selects an element");
        let result = self.result().ty();
        if result != expected {
            return Err(invalid(
                op,
                format!("the result has type `{result}`, but the gep gives `{expected}`"),
            ));
        }
        Ok(())
    }
}

/// A gep with no index and an empty path is its base.
impl Foldable for Gep {
    fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        let base = self.base().as_value_ref();
        let is_identity = self.operands().group(1).is_empty()
            && self.get_path().is_empty()
            && self.result().ty() == base.borrow().ty();
        if !is_identity {
            return FoldResult::Failed;
        }
        results.push(OpFoldResult::Value(base));
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

/// A gep with no index whose base is another gep becomes one gep
/// ([MergeGepOfGep](crate::canonicalization::MergeGepOfGep)).
impl Canonicalizable for Gep {
    fn get_canonicalization_patterns(rewrites: &mut RewritePatternSet, context: Rc<Context>) {
        rewrites.push(crate::canonicalization::MergeGepOfGep::new(context));
    }
}

impl OpPrinter for Gep {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        let op = self.as_operation();
        printer.print_space();
        printer.print_value_uses(op.operands().group(0).as_value_range());
        if !op.operands().group(1).is_empty() {
            *printer += const_text(", ");
            printer.print_value_uses(op.operands().group(1).as_value_range());
        }
        print_path(printer, &self.get_path());
        print_attributes(printer, op);
    }
}

/// Reads `%p`, then an optional `, %i`, an optional `, [n, ..]` and an optional
/// `attributes {..}`. The result's type is inferred.
impl OpParser for Gep {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let base = parser.parse_operand(/*allow_result_number=*/ true)?;
        let mut index = None;
        let mut path = None;
        if parser.parse_optional_comma()? {
            match parser.parse_optional_operand(/*allow_result_number=*/ true)? {
                Some(operand) => {
                    index = Some(operand);
                    if parser.parse_optional_comma()? {
                        path = Some(parse_path(parser)?);
                    }
                }
                None => path = Some(parse_path(parser)?),
            }
        }
        let mut attrs = SmallVec::default();
        parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

        // The operands are uses of values defined earlier, which carry their types, as in the
        // derived format of the other operations.
        let unknown = parser.context().unknown_type();
        let base = parser.resolve_operand(base, unknown)?;
        state.operands.push(SmallVec::from_iter([base]));
        let mut indices = SmallVec::default();
        if let Some(index) = index {
            indices.push(parser.resolve_operand(index, unknown)?);
        }
        state.operands.push(indices);
        add_path(state, parser, path);
        state.attrs.extend(attrs);
        state.results.push(unknown);
        Ok(())
    }
}

/// `[1, 2]`, the text of a path.
pub(crate) fn path_text<'a>(path: impl Iterator<Item = &'a i32>) -> String {
    let steps = path.map(|step| format!("{step}")).collect::<alloc::vec::Vec<_>>();
    format!("[{}]", steps.join(", "))
}

/// `, [n, ..]` when `path` is not empty.
pub(crate) fn print_path(printer: &mut print::AsmPrinter<'_>, path: &Array<i32>) {
    if path.is_empty() {
        return;
    }
    *printer += const_text(", [");
    for (i, step) in path.iter().enumerate() {
        if i > 0 {
            *printer += const_text(", ");
        }
        printer.print_int(Int::from(*step));
    }
    *printer += const_text("]");
}

/// Reads `[n, ..]`, each step written as an `i32` constant is. A negative step parses, and the
/// verifier reports it.
pub(crate) fn parse_path(parser: &mut dyn OpAsmParser<'_>) -> ParseResult<Array<i32>> {
    let ty = parser.context().i32();
    let mut path = Array::default();
    parser.parse_comma_separated_list(Delimiter::Bracket, Some("path"), |parser| {
        let (span, literal) = parser.parse_integer_literal()?.into_parts();
        path.push(literal.to_int(ty, span)?.signed() as i32);
        Ok(true)
    })?;
    Ok(path)
}

/// Gives `state` the property `path` when `path` is given and not empty, the property's default.
pub(crate) fn add_path(
    state: &mut OperationState,
    parser: &dyn OpAsmParser<'_>,
    path: Option<Array<i32>>,
) {
    if let Some(path) = path.filter(|path| !path.is_empty()) {
        state.add_attribute("path", parser.context_rc().create_attribute::<I32ArrayAttr, _>(path));
    }
}
