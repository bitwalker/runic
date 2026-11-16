use alloc::{format, string::String};

use runic_ir::{
    derive::{EffectOpInterface, operation},
    dialects::builtin::attributes::{I32ArrayAttr, TypeAttr},
    effects::*,
    parse::ParseResult,
    traits::*,
    *,
};

use super::{
    access::{invalid, print_attributes, verify_layout},
    address::{add_path, parse_path, path_text, print_path},
};
use crate::MemDialect;

/// The size in bytes of a value of type `ty`, an `isize`: `%n = mem.sizeof struct<i32, i8>;`.
///
/// The size is a multiple of the alignment, so it is also the distance between neighbouring
/// elements of an array of `ty`. When a data layout is in scope ([DataLayout::closest]), `ty` must
/// have a layout under it, and the operation folds to the size when it fits in an `isize` of that
/// layout. It has no effects.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Sizeof {
    /// The type measured
    #[attr(hidden)]
    ty: TypeAttr,
    /// The size in bytes
    #[result]
    result: Isize,
}

/// The alignment in bytes of a value of type `ty`, an `isize`: `%n = mem.alignof i64;`.
///
/// When a data layout is in scope ([DataLayout::closest]), `ty` must have a layout under it, and
/// the operation folds to the alignment when it fits in an `isize` of that layout. It has no
/// effects.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Alignof {
    /// The type measured
    #[attr(hidden)]
    ty: TypeAttr,
    /// The alignment in bytes
    #[result]
    result: Isize,
}

/// The byte offset of the element `path` selects inside a value of type `ty`, an `isize`:
/// `%n = mem.offsetof struct<i32, i8>, [1];`.
///
/// The path selects elements as [AggregateType] numbers them, as a `mem.gep` path does; it must
/// select an element of `ty` ([element_at]) whether or not a data layout is in scope. When one is
/// ([DataLayout::closest]), `ty` must have a layout under it, and the operation folds to the offset
/// ([DataLayout::offset_of]) when it fits in an `isize` of that layout. An empty path selects `ty`
/// itself, at offset 0, and is left out of the text. It has no effects.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, Foldable, OpPrinter)
)]
pub struct Offsetof {
    /// The type the element is selected in
    #[attr(hidden)]
    ty: TypeAttr,
    /// The elements selected, outermost first
    #[attr(hidden)]
    #[default]
    path: I32ArrayAttr,
    /// The offset in bytes
    #[result]
    result: Isize,
}

impl Offsetof {
    /// The steps of the path, or `None` if one is negative.
    pub fn steps(&self) -> Option<SmallVec<[u32; 4]>> {
        self.get_path().iter().map(|step| u32::try_from(*step).ok()).collect()
    }
}

/// The type inference, verifiers and fold of an operation that measures its property `ty`:
/// `$what` describes `ty` in a diagnostic, `$check` verifies the rest of the operation, and
/// `$measure` gives the constant under a data layout.
macro_rules! measure_op {
    ($Op:ident, $what:literal, $check:path, $measure:path) => {
        /// A result whose type is not yet known is an `isize`.
        impl InferTypeOpInterface for $Op {
            fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
                if self.result().ty().is_unknown() {
                    self.result_mut().set_type(context.isize());
                }
                Ok(())
            }
        }

        /// The measured type is known.
        impl Verify<dyn InferTypeOpInterface> for $Op {
            fn verify(&self, _context: &Context) -> Result<(), Report> {
                verify_known(self.as_operation(), *self.get_ty())?;
                $check(self)
            }
        }

        /// When a data layout is in scope, the measured type has a layout under it.
        impl Verify<dyn MemoryEffectOpInterface> for $Op {
            fn verify(&self, _context: &Context) -> Result<(), Report> {
                verify_layout(self.as_operation(), *self.get_ty(), $what)
            }
        }

        /// Under a data layout in scope, the constant the operation measures, when it fits in an
        /// `isize` of that layout.
        impl Foldable for $Op {
            fn fold(&self, results: &mut SmallVec<[OpFoldResult; 1]>) -> FoldResult {
                fold_to_isize(self.as_operation(), results, |layout| $measure(self, layout))
            }

            fn fold_with(
                &self,
                _operands: &[Option<AttributeRef>],
                results: &mut SmallVec<[OpFoldResult; 1]>,
            ) -> FoldResult {
                self.fold(results)
            }
        }
    };
}

measure_op!(Sizeof, "the type measured", nothing_else, measure_size);
measure_op!(Alignof, "the type measured", nothing_else, measure_align);
measure_op!(
    Offsetof,
    "the type the element is selected in",
    verify_offsetof_path,
    measure_offset
);

/// The size of `ty` under `layout`.
fn measure_size(op: &Sizeof, layout: &DataLayout) -> Option<u64> {
    layout.size_of(*op.get_ty()).ok()
}

/// The alignment of `ty` under `layout`, in bytes.
fn measure_align(op: &Alignof, layout: &DataLayout) -> Option<u64> {
    layout.align_of(*op.get_ty()).ok().map(|align| align.bytes())
}

/// The offset of the element the path selects in `ty` under `layout`.
fn measure_offset(op: &Offsetof, layout: &DataLayout) -> Option<u64> {
    let (offset, _) = layout.offset_of(*op.get_ty(), &op.steps()?).ok()?;
    Some(offset)
}

/// A `sizeof` or `alignof` has nothing to verify besides its type.
fn nothing_else<T>(_op: &T) -> Result<(), Report> {
    Ok(())
}

/// The path of an `offsetof` has non-negative steps and selects an element of `ty`.
fn verify_offsetof_path(op: &Offsetof) -> Result<(), Report> {
    let path = op.get_path();
    let operation = op.as_operation();
    if let Some(step) = path.iter().find(|step| **step < 0) {
        return Err(invalid(
            operation,
            format!("the path {} has the negative step {step}", path_text(path.iter())),
        ));
    }
    let ty = *op.get_ty();
    let steps = op.steps().expect("no step is negative");
    element_at(ty, &steps).map(|_| ()).map_err(|error| {
        invalid(
            operation,
            format!("the path {} selects no element of `{ty}`: {error}", path_text(path.iter())),
        )
    })
}

/// `ty` is not the unknown type.
fn verify_known(op: &Operation, ty: Type) -> Result<(), Report> {
    if ty.is_unknown() {
        return Err(invalid(op, String::from("the type measured is not known")));
    }
    Ok(())
}

/// Folds `op` to the `isize` constant `measure` gives under the data layout in scope at `op`.
/// Nothing folds with no layout in scope, when `measure` gives nothing (the verifier reports
/// why), or when the value does not fit in the `isize` of the layout.
fn fold_to_isize(
    op: &Operation,
    results: &mut SmallVec<[OpFoldResult; 1]>,
    measure: impl FnOnce(&DataLayout) -> Option<u64>,
) -> FoldResult {
    let Some(layout) = DataLayout::closest(op) else {
        return FoldResult::Failed;
    };
    let Some(value) = measure(&layout) else {
        return FoldResult::Failed;
    };
    let width = layout.isize_width();
    if width < u64::BITS && value >> width != 0 {
        return FoldResult::Failed;
    }
    let context = op.context_rc();
    // An `isize` constant holds a 64-bit value, which stands for its low bits on a narrower target
    let attr = context.int_attr(Int::from_bits(64, u128::from(value)), context.isize());
    results.push(OpFoldResult::Attribute(attr.as_attribute_ref()));
    FoldResult::Ok(())
}

impl OpPrinter for Sizeof {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        printer.print_space();
        printer.print_type(*self.get_ty());
        print_attributes(printer, self.as_operation());
    }
}

impl OpPrinter for Alignof {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        printer.print_space();
        printer.print_type(*self.get_ty());
        print_attributes(printer, self.as_operation());
    }
}

impl OpPrinter for Offsetof {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        printer.print_space();
        printer.print_type(*self.get_ty());
        print_path(printer, &self.get_path());
        print_attributes(printer, self.as_operation());
    }
}

/// Reads a type and an optional `attributes {..}`.
impl OpParser for Sizeof {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_measured(state, parser, false)
    }
}

/// Reads a type and an optional `attributes {..}`.
impl OpParser for Alignof {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_measured(state, parser, false)
    }
}

/// Reads a type, an optional `, [n, ..]` and an optional `attributes {..}`.
impl OpParser for Offsetof {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_measured(state, parser, true)
    }
}

/// Reads a type, then, when `with_path`, an optional `, [n, ..]`, then an optional
/// `attributes {..}`.
fn parse_measured(
    state: &mut OperationState,
    parser: &mut dyn OpAsmParser<'_>,
    with_path: bool,
) -> ParseResult {
    let ty = parser.parse_type()?.into_inner();
    let path = if with_path && parser.parse_optional_comma()? {
        Some(parse_path(parser)?)
    } else {
        None
    };
    let mut attrs = SmallVec::default();
    parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

    state.add_attribute("ty", parser.context_rc().create_attribute::<TypeAttr, _>(ty));
    add_path(state, parser, path);
    state.attrs.extend(attrs);
    state.results.push(parser.context().isize());
    Ok(())
}
