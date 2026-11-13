use alloc::rc::Rc;

use runic_core::{
    derive::{EffectOpInterface, operation},
    effects::*,
    parse::ParseResult,
    patterns::RewritePatternSet,
    traits::*,
    *,
};

use super::access::{MemoryResource, print_attributes};
use crate::MemDialect;

/// A copy of `len` bytes from `src` to `dst`: `mem.memcpy %dst, %src, %len;`.
///
/// `len` is an `isize` count of bytes; the pointers may have any pointer types, typed or opaque,
/// in any address spaces. The two regions must not overlap: a copy between overlapping regions is
/// undefined behaviour, which no verifier can check (`mem.memmove` is the copy that allows it).
///
/// The copy reads [MemoryResource] at `src` and writes it at `dst`, so it is never removed, and no
/// read of memory moves or merges across it. A copy of the constant length 0 does nothing and is
/// erased by the canonicalizer.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    name = "memcpy",
    implements(MemoryEffectOpInterface, OperandRangeRequirementOpInterface, OpPrinter)
)]
pub struct MemCpy {
    /// The address of the first byte written
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(MemoryResource))]
    dst: AnyPointer,
    /// The address of the first byte read
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Read), resource(MemoryResource))]
    src: AnyPointer,
    /// The number of bytes copied
    #[operand]
    len: Isize,
}

/// A copy of `len` bytes from `src` to `dst`, where the regions may overlap:
/// `mem.memmove %dst, %src, %len;`.
///
/// The bytes are copied as if through a temporary buffer, so an overlap is defined. Otherwise it is
/// `mem.memcpy`: `len` is an `isize` count of bytes, the pointers may have any pointer types, it
/// reads [MemoryResource] at `src` and writes it at `dst`, so it is never removed and no read of
/// memory moves or merges across it, and a move of the constant length 0 is erased by the
/// canonicalizer.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    name = "memmove",
    implements(MemoryEffectOpInterface, OperandRangeRequirementOpInterface, OpPrinter)
)]
pub struct MemMove {
    /// The address of the first byte written
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(MemoryResource))]
    dst: AnyPointer,
    /// The address of the first byte read
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Read), resource(MemoryResource))]
    src: AnyPointer,
    /// The number of bytes copied
    #[operand]
    len: Isize,
}

/// A fill of `len` bytes at `dst` with the byte `byte`: `mem.memset %dst, %byte, %len;`.
///
/// `byte` is an `i8`, `len` an `isize` count of bytes; the pointer may have any pointer type. The
/// fill writes [MemoryResource] at `dst`, so it is never removed, and no read of memory moves or
/// merges across it. A fill of the constant length 0 does nothing and is erased by the
/// canonicalizer.
#[derive(EffectOpInterface)]
#[operation(
    dialect = MemDialect,
    name = "memset",
    implements(MemoryEffectOpInterface, OperandRangeRequirementOpInterface, OpPrinter)
)]
pub struct MemSet {
    /// The address of the first byte written
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(MemoryResource))]
    dst: AnyPointer,
    /// The byte written
    #[operand]
    byte: Int8,
    /// The number of bytes written
    #[operand]
    len: Isize,
}

/// The requirements, printer, parser and canonicalization patterns of a bulk operation whose
/// operands are `$pointers` pointers followed by other values, the last of them the length.
macro_rules! bulk_op {
    ($Op:ident, $pointers:literal) => {
        /// Whether an address is valid is not a range of it; every length and byte is allowed.
        impl OperandRangeRequirementOpInterface for $Op {
            fn operand_range_requirement(&self, index: usize) -> OperandRangeRequirement {
                if index < $pointers {
                    OperandRangeRequirement::Unexpressed
                } else {
                    OperandRangeRequirement::None
                }
            }
        }

        /// An operation of the constant length 0 is erased
        /// ([EraseZeroLength](crate::canonicalization::EraseZeroLength)).
        impl Canonicalizable for $Op {
            fn get_canonicalization_patterns(
                rewrites: &mut RewritePatternSet,
                context: Rc<Context>,
            ) {
                rewrites.push(crate::canonicalization::EraseZeroLength::new::<$Op>(context));
            }
        }

        impl OpPrinter for $Op {
            fn print(&self, printer: &mut print::AsmPrinter<'_>) {
                let op = self.as_operation();
                printer.print_space();
                printer.print_value_uses(op.operands().group(0).as_value_range());
                print_attributes(printer, op);
            }
        }

        /// Reads three comma-separated operands and an optional `attributes {..}`.
        impl OpParser for $Op {
            fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
                parse_three_operands(state, parser)
            }
        }
    };
}

bulk_op!(MemCpy, 2);
bulk_op!(MemMove, 2);
bulk_op!(MemSet, 1);

/// Reads `%a, %b, %c` and an optional `attributes {..}`.
fn parse_three_operands(
    state: &mut OperationState,
    parser: &mut dyn OpAsmParser<'_>,
) -> ParseResult {
    let first = parser.parse_operand(/*allow_result_number=*/ true)?;
    parser.parse_comma()?;
    let second = parser.parse_operand(/*allow_result_number=*/ true)?;
    parser.parse_comma()?;
    let third = parser.parse_operand(/*allow_result_number=*/ true)?;
    let mut attrs = SmallVec::default();
    parser.parse_optional_attribute_dict_with_keyword(&mut attrs)?;

    // The operands are uses of values defined earlier, which carry their types, as in the derived
    // format of the other operations.
    let unknown = parser.context().unknown_type();
    for operand in [first, second, third] {
        let operand = parser.resolve_operand(operand, unknown)?;
        state.add_operand(operand);
    }
    state.attrs.extend(attrs);
    Ok(())
}
