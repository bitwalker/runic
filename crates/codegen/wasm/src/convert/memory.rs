//! The patterns of the `mem` dialect: loads and stores, addresses, the measures of a type, the
//! pointer casts and the bulk operations, all on the linear memory `@__linear_memory`.
//!
//! A pointer and an `isize` are carried in an `i32`, the address of a byte in linear memory, and
//! every access names the module's linear memory ([crate::symbols::linear_memory]).
//!
//! * **`mem.load`** of a value of type `T` (the pointee, or the type written for an opaque
//!   pointer) whose size `S` under the layout is 1, 2, 4 or 8 bytes and whose type has a carrier:
//!   one `wasm.load` of the carrier, which reads `S` bytes (a narrow load, zero-extended, when `S`
//!   is less than the carrier's size), then the mask of carried form when `T` is an integer `iN`
//!   with `N < 8 * S` (an `i1` is one byte, masked to one bit). A function value is an `i32`, its
//!   table slot, loaded as one where the data layout gives it 4 bytes (`fn = 32`); with no `fn`
//!   entry it has no layout, and the error says to state one. Any other `T` is an error (see the
//!   pre-check of `convert-to-wasm`).
//! * **`mem.store`**: one `wasm.store` of the carried value, of `S` bytes. The carried value is
//!   zero-extended, so the bytes written above `N` bits are zero.
//! * The alignment of an access is the operation's `align` when it states one, else the layout's
//!   alignment of `T`; in either case at most `S`. Wasm's alignment is a hint bounded by the
//!   width of the access, so a larger promise is dropped, not an error. The offset is 0.
//! * **`mem.gep %p, %i, [path]`**: `%p + %i * size_of(T) + offset_of(T, path)` in `i32`, with only
//!   what is needed: no multiplication by 1, no addition of 0, and `%p` itself when there is no
//!   index and the offset is 0. A size or an offset that does not fit in 32 bits is an error. The
//!   arithmetic wraps, as `mem.gep` says it does.
//! * **`mem.sizeof`**, **`mem.alignof`**, **`mem.offsetof`**: a `wasm.const` of the layout's
//!   answer; an answer that does not fit in 32 bits is an error.
//! * **`mem.ptrtoint`**, **`mem.inttoptr`**, **`mem.ptrcast`**: the operand itself (in address
//!   space 0 every pointer and `isize` are the same `i32`).
//! * **`mem.memcpy`** and **`mem.memmove`**: `wasm.memory_copy` within the linear memory (Wasm's
//!   copy allows the ranges to overlap); **`mem.memset`**: `wasm.memory_fill`.
//! * **`mem.alloca`** has no pattern: `wasm-lower-frames` gives each one a slot in a frame, and
//!   `convert-to-wasm` reports one it meets before it converts anything (`alloca_error`).
//!
//! Every refusal of these patterns on IR that verifies is one function that the check before the
//! conversion calls too (`access`, `gep_layout`, `measure`), so the check finds it first.

use alloc::{format, rc::Rc, string::String};

use runic_dialect_mem as mem;
use runic_dialect_wasm::{self as wasm, Extension, MemArg, MemoryRef, Storage};
use runic_ir::{
    Context, DataLayout, Operation, OperationName, OperationRef, Report, SourceSpan, Spanned, Type,
    Value, ValueRef,
    conversion::{
        ConversionPattern, ConversionPatternRewriter, ConversionPatternSet, ConvertedOperands,
        TypeConverter,
    },
    diagnostics::report,
    patterns::{Pattern, PatternInfo},
};

use crate::{
    WasmTarget,
    carried::{self, const_int, mask},
    convert::{
        WellKnownSymbols,
        calls::{pattern_info, registered},
    },
    symbols::{LINEAR_MEMORY, local_path},
    types::{carrier, carrier_width, width},
};

/// Adds the patterns of this module. `symbols` holds the linear memory of each module whose
/// functions access memory, found or declared before the conversion starts; a pattern that needs
/// one its module has not got is an error.
pub fn populate(
    patterns: &mut ConversionPatternSet,
    converter: &TypeConverter,
    target: &WasmTarget,
    symbols: &Rc<WellKnownSymbols>,
) -> Result<(), Report> {
    let context = patterns.context();
    let rules = [
        Rule::Load,
        Rule::Store,
        Rule::Gep,
        Rule::Sizeof,
        Rule::Alignof,
        Rule::Offsetof,
        Rule::PtrToInt,
        Rule::IntToPtr,
        Rule::PtrCast,
        Rule::MemCpy,
        Rule::MemMove,
        Rule::MemSet,
    ];
    for rule in rules {
        patterns.push(MemPattern {
            info: rule.info(&context),
            converter: converter.clone(),
            layout: target.layout().clone(),
            symbols: symbols.clone(),
            rule,
        });
    }
    Ok(())
}

/// What one Wasm access of a value of a type reads or writes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Access {
    /// The carrier of the type: the type of the value loaded or stored.
    carrier: Type,
    /// The number of bytes accessed, the size of the type: 1, 2, 4 or 8.
    size: u64,
    /// How much of the carrier is accessed.
    storage: Storage,
    /// The width `N` of the type when it is an integer `iN` with `N < 8 * size`: a load masks to
    /// it.
    mask: Option<u32>,
}

/// What one Wasm load or store of a value of type `ty` accesses under `layout`; an error naming
/// `op` (a `mem.load` or a `mem.store`) and the type when the type has no carrier or its size is
/// not 1, 2, 4 or 8 bytes.
pub(crate) fn access(
    context: &Context,
    layout: &DataLayout,
    ty: Type,
    op: &Operation,
) -> Result<Access, Report> {
    let size = layout.size_of(ty).ok();
    let carried = carrier(context, ty);
    let (Some(carrier), Some(size @ (1 | 2 | 4 | 8))) = (carried, size) else {
        let (verb, noun) = if op.is::<mem::Store>() {
            ("stored", "store")
        } else {
            ("loaded", "load")
        };
        let label = match (carried, size) {
            (None, _) => format!("`{ty}` has no Wasm value type to carry it"),
            (Some(_), Some(size)) => {
                format!("`{ty}` is {size} bytes under the data layout in scope")
            }
            (Some(_), None) => format!("`{ty}` has no layout under the data layout in scope"),
        };
        if ty.is_function() && size.is_none() {
            return Err(report!(
                message: (format!(
                    "`{}` of `{ty}`: a function value is {verb} only where the data layout gives \
                     it a size: state `fn = 32` in the data layout",
                    op.name()
                )),
                labels: [primary(op.span(), label)],
            ));
        }
        return Err(report!(
            message: (format!(
                "`{}` of `{ty}`: only a value that fits one Wasm {noun} is {verb}; {noun} its \
                 fields",
                op.name()
            )),
            labels: [primary(op.span(), label)],
        ));
    };
    let carrier_bytes = match carrier_width(carrier) {
        Some(bits) => u64::from(bits / 8),
        None => size,
    };
    let storage = match (size, size < carrier_bytes) {
        (1, true) => Storage::I8,
        (2, true) => Storage::I16,
        (4, true) => Storage::I32,
        _ => Storage::Full,
    };
    let mask = width(ty)
        .filter(|_| ty.integer_width().is_some())
        .filter(|n| u64::from(*n) < 8 * size);
    Ok(Access {
        carrier,
        size,
        storage,
        mask,
    })
}

/// The error of a `mem.alloca` that reaches `convert-to-wasm`, which has no pattern for it.
pub(crate) fn alloca_error(op: &Operation) -> Report {
    report!(
        message: (format!(
            "`{}` does not lower `mem.alloca`: run `{}` first, which gives each one a slot in its \
             function's frame",
            super::NAME,
            crate::frames::NAME
        )),
        labels: [primary(op.span(), "this `mem.alloca` has no slot in a frame")],
    )
}

/// What the `mem.gep` `op` adds to its base under `layout`: the size of its pointee when it has
/// an index, and the offset of its path (0 for an empty path); `None` when `op` is not a
/// `mem.gep` of a typed pointer with a path. An error naming `op` when the layout does not know
/// the size or the offset, or when one does not fit in 32 bits.
pub(crate) fn gep_layout(
    layout: &DataLayout,
    op: &Operation,
) -> Result<Option<(Option<u64>, u64)>, Report> {
    let Some(gep) = op.downcast_ref::<mem::Gep>() else {
        return Ok(None);
    };
    let (Some(pointee), Some(path)) = (gep.base().ty().pointee(), gep.steps()) else {
        return Ok(None);
    };
    let size = if gep.dynamic_index().is_none() {
        None
    } else {
        let size = layout.size_of(pointee).map_err(|error| {
            layout_error(op, format!("the size of `{pointee}` is not known: {error}"))
        })?;
        Some(fits_32(op, size, || format!("the size of `{pointee}`"))?)
    };
    let offset = if path.is_empty() {
        0
    } else {
        let (offset, _) = layout.offset_of(pointee, &path).map_err(|error| {
            layout_error(op, format!("the offset of {path:?} in `{pointee}` is not known: {error}"))
        })?;
        fits_32(op, offset, || format!("the offset of {path:?} in `{pointee}`"))?
    };
    Ok(Some((size, offset)))
}

/// The answer of `layout` for `op`, a `mem.sizeof`, `mem.alignof` or `mem.offsetof`, in 32 bits;
/// `None` for any other operation. An error naming `op` when the layout does not know it or it
/// does not fit in 32 bits.
pub(crate) fn measure(layout: &DataLayout, op: &Operation) -> Result<Option<u64>, Report> {
    let (value, what) = if let Some(sizeof) = op.downcast_ref::<mem::Sizeof>() {
        let ty = *sizeof.get_ty();
        let size = layout.size_of(ty).map_err(|error| {
            layout_error(op, format!("the size of `{ty}` is not known: {error}"))
        })?;
        (size, format!("the size of `{ty}`"))
    } else if let Some(alignof) = op.downcast_ref::<mem::Alignof>() {
        let ty = *alignof.get_ty();
        let align = layout.align_of(ty).map_err(|error| {
            layout_error(op, format!("the alignment of `{ty}` is not known: {error}"))
        })?;
        (align.bytes(), format!("the alignment of `{ty}`"))
    } else if let Some(offsetof) = op.downcast_ref::<mem::Offsetof>() {
        let ty = *offsetof.get_ty();
        let path = offsetof.steps().unwrap_or_default();
        let (offset, _) = layout.offset_of(ty, &path).map_err(|error| {
            layout_error(op, format!("the offset of {path:?} in `{ty}` is not known: {error}"))
        })?;
        (offset, format!("the offset of {path:?} in `{ty}`"))
    } else {
        return Ok(None);
    };
    fits_32(op, value, || what).map(Some)
}

/// The path by which an operation that replaces `op` names `memory` ([local_path]).
fn memory_path(op: OperationRef, memory: MemoryRef) -> Option<runic_ir::SymbolPath> {
    local_path(op, memory.as_operation_ref(), LINEAR_MEMORY)
}

/// The Wasm alignment of an access of `size` bytes of a value of type `ty`: `stated` when the
/// operation states one, else the layout's alignment of `ty`; in either case at most `size`.
fn alignment(layout: &DataLayout, ty: Type, stated: mem::Alignment, size: u64) -> u64 {
    let align = match stated.bytes() {
        Some(align) => align.bytes(),
        None => layout.align_of(ty).map_or(1, |align| align.bytes()),
    };
    align.min(size)
}

/// What each pattern of this module lowers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Rule {
    /// `mem.load`: `wasm.load`, then the mask.
    Load,
    /// `mem.store`: `wasm.store`.
    Store,
    /// `mem.gep`: `wasm.mul` and `wasm.add` of constants from the layout.
    Gep,
    /// `mem.sizeof`: a `wasm.const`.
    Sizeof,
    /// `mem.alignof`: a `wasm.const`.
    Alignof,
    /// `mem.offsetof`: a `wasm.const`.
    Offsetof,
    /// `mem.ptrtoint`: the operand.
    PtrToInt,
    /// `mem.inttoptr`: the operand.
    IntToPtr,
    /// `mem.ptrcast`: the operand.
    PtrCast,
    /// `mem.memcpy`: `wasm.memory_copy`.
    MemCpy,
    /// `mem.memmove`: `wasm.memory_copy`.
    MemMove,
    /// `mem.memset`: `wasm.memory_fill`.
    MemSet,
}

impl Rule {
    /// The information of the pattern of this rule.
    fn info(self, context: &Rc<Context>) -> PatternInfo {
        let none: [OperationName; 0] = [];
        let constants = [registered::<wasm::Const>(context)];
        let address = [
            registered::<wasm::Const>(context),
            registered::<wasm::Mul>(context),
            registered::<wasm::Add>(context),
        ];
        let load = [
            registered::<wasm::Load>(context),
            registered::<wasm::Const>(context),
            registered::<wasm::And>(context),
        ];
        match self {
            Rule::Load => pattern_info::<mem::Load>(context, "wasm-mem-load", load),
            Rule::Store => pattern_info::<mem::Store>(
                context,
                "wasm-mem-store",
                [registered::<wasm::Store>(context)],
            ),
            Rule::Gep => pattern_info::<mem::Gep>(context, "wasm-mem-gep", address),
            Rule::Sizeof => pattern_info::<mem::Sizeof>(context, "wasm-mem-sizeof", constants),
            Rule::Alignof => pattern_info::<mem::Alignof>(context, "wasm-mem-alignof", constants),
            Rule::Offsetof => {
                pattern_info::<mem::Offsetof>(context, "wasm-mem-offsetof", constants)
            }
            Rule::PtrToInt => pattern_info::<mem::PtrToInt>(context, "wasm-mem-ptrtoint", none),
            Rule::IntToPtr => pattern_info::<mem::IntToPtr>(context, "wasm-mem-inttoptr", none),
            Rule::PtrCast => pattern_info::<mem::PtrCast>(context, "wasm-mem-ptrcast", none),
            Rule::MemCpy => pattern_info::<mem::MemCpy>(
                context,
                "wasm-mem-memcpy",
                [registered::<wasm::MemoryCopy>(context)],
            ),
            Rule::MemMove => pattern_info::<mem::MemMove>(
                context,
                "wasm-mem-memmove",
                [registered::<wasm::MemoryCopy>(context)],
            ),
            Rule::MemSet => pattern_info::<mem::MemSet>(
                context,
                "wasm-mem-memset",
                [registered::<wasm::MemoryFill>(context)],
            ),
        }
    }
}

/// The pattern of one [Rule].
struct MemPattern {
    info: PatternInfo,
    converter: TypeConverter,
    layout: DataLayout,
    symbols: Rc<WellKnownSymbols>,
    rule: Rule,
}

impl Pattern for MemPattern {
    fn info(&self) -> &PatternInfo {
        &self.info
    }
}

impl ConversionPattern for MemPattern {
    fn type_converter(&self) -> Option<&TypeConverter> {
        Some(&self.converter)
    }

    fn match_and_rewrite(
        &self,
        op: OperationRef,
        operands: ConvertedOperands<'_>,
        rewriter: &mut ConversionPatternRewriter,
    ) -> Result<bool, Report> {
        let Some(values) = operands.all_one() else {
            return Ok(false);
        };
        let span = op.borrow().span();
        rewriter.set_insertion_point_before(op);
        match self.rule {
            Rule::Load => self.load(op, &values, rewriter, span),
            Rule::Store => self.store(op, &values, rewriter, span),
            Rule::Gep => self.gep(op, &values, rewriter, span),
            Rule::Sizeof | Rule::Alignof | Rule::Offsetof => {
                let Some(value) = measure(&self.layout, &op.borrow())? else {
                    return Ok(false);
                };
                let constant = const_int(rewriter, value, rewriter.context().i32(), span)?;
                rewriter.replace_op(op, &[constant])?;
                Ok(true)
            }
            Rule::PtrToInt | Rule::IntToPtr | Rule::PtrCast => {
                let [value] = values.as_slice() else {
                    return Ok(false);
                };
                rewriter.replace_op(op, &[*value])?;
                Ok(true)
            }
            Rule::MemCpy | Rule::MemMove => {
                let [dst, src, len] = values.as_slice() else {
                    return Ok(false);
                };
                let memory = self.memory(op)?;
                let mut copy = rewriter
                    .create_op::<wasm::MemoryCopy, _>(span, (memory, memory, *dst, *src, *len))?;
                if let Some(path) = memory_path(op, memory) {
                    let mut copy = copy.borrow_mut();
                    copy.dst_mut().set_path(path.clone());
                    copy.src_mut().set_path(path);
                }
                rewriter.erase_op(op)?;
                Ok(true)
            }
            Rule::MemSet => {
                let [dst, byte, len] = values.as_slice() else {
                    return Ok(false);
                };
                let memory = self.memory(op)?;
                let mut fill =
                    rewriter.create_op::<wasm::MemoryFill, _>(span, (memory, *dst, *byte, *len))?;
                if let Some(path) = memory_path(op, memory) {
                    fill.borrow_mut().memory_mut().set_path(path);
                }
                rewriter.erase_op(op)?;
                Ok(true)
            }
        }
    }
}

impl MemPattern {
    /// The linear memory of the module of `op`; an error when none was found or declared for it.
    fn memory(&self, op: OperationRef) -> Result<MemoryRef, Report> {
        self.symbols.of_op(op).memory.ok_or_else(|| {
            Report::msg(format!(
                "`convert-to-wasm` declared no linear memory for the module of `{}`",
                op.borrow().name()
            ))
        })
    }

    /// `mem.load`: a `wasm.load` of the carrier, then the mask.
    fn load(
        &self,
        op: OperationRef,
        values: &[ValueRef],
        rewriter: &mut ConversionPatternRewriter,
        span: SourceSpan,
    ) -> Result<bool, Report> {
        let [addr] = values else {
            return Ok(false);
        };
        let (access, align) = {
            let op_ref = op.borrow();
            let load = op_ref.downcast_ref::<mem::Load>().expect("rooted at mem.load");
            let ty = load.result().ty();
            let access = access(rewriter.context(), &self.layout, ty, &op_ref)?;
            (access, alignment(&self.layout, ty, *load.get_align(), access.size))
        };
        let memory = self.memory(op)?;
        let mut load =
            rewriter.create_op::<wasm::Load, _>(span, (memory, *addr, MemArg::new(0, align)))?;
        {
            let mut load = load.borrow_mut();
            if let Some(path) = memory_path(op, memory) {
                load.memory_mut().set_path(path);
            }
            load.set_storage(access.storage);
            if access.storage != Storage::Full {
                load.set_extension(Extension::Unsigned);
            }
            load.result_mut().set_type(access.carrier);
        }
        let mut value = carried::result(&load);
        if let Some(n) = access.mask {
            value = mask(rewriter, value, n, span)?;
        }
        rewriter.replace_op(op, &[value])?;
        Ok(true)
    }

    /// `mem.store`: a `wasm.store` of the carried value.
    fn store(
        &self,
        op: OperationRef,
        values: &[ValueRef],
        rewriter: &mut ConversionPatternRewriter,
        span: SourceSpan,
    ) -> Result<bool, Report> {
        let [value, addr] = values else {
            return Ok(false);
        };
        let (access, align) = {
            let op_ref = op.borrow();
            let store = op_ref.downcast_ref::<mem::Store>().expect("rooted at mem.store");
            let ty = store.value().ty();
            let access = access(rewriter.context(), &self.layout, ty, &op_ref)?;
            (access, alignment(&self.layout, ty, *store.get_align(), access.size))
        };
        let memory = self.memory(op)?;
        let mut store = rewriter
            .create_op::<wasm::Store, _>(span, (memory, *addr, *value, MemArg::new(0, align)))?;
        {
            let mut store = store.borrow_mut();
            if let Some(path) = memory_path(op, memory) {
                store.memory_mut().set_path(path);
            }
            store.set_storage(access.storage);
        }
        rewriter.erase_op(op)?;
        Ok(true)
    }

    /// `mem.gep %p, %i, [path]`: `%p + %i * size + offset`, emitting only what is needed.
    fn gep(
        &self,
        op: OperationRef,
        values: &[ValueRef],
        rewriter: &mut ConversionPatternRewriter,
        span: SourceSpan,
    ) -> Result<bool, Report> {
        let Some((&base, index)) = values.split_first() else {
            return Ok(false);
        };
        let Some((size, offset)) = gep_layout(&self.layout, &op.borrow())? else {
            return Ok(false);
        };
        let i32 = rewriter.context().i32();
        let mut address = base;
        if let (Some(&index), Some(size)) = (index.first(), size) {
            let step = match size {
                0 => None,
                1 => Some(index),
                size => {
                    let size = const_int(rewriter, size, i32, span)?;
                    Some(carried::result(&rewriter.create_op::<wasm::Mul, _>(span, (index, size))?))
                }
            };
            if let Some(step) = step {
                address =
                    carried::result(&rewriter.create_op::<wasm::Add, _>(span, (address, step))?);
            }
        }
        if offset != 0 {
            let offset = const_int(rewriter, offset, i32, span)?;
            address =
                carried::result(&rewriter.create_op::<wasm::Add, _>(span, (address, offset))?);
        }
        rewriter.replace_op(op, &[address])?;
        Ok(true)
    }
}

/// `value`, when it fits in 32 bits; otherwise an error naming `op` and `what` the value is.
fn fits_32(op: &Operation, value: u64, what: impl FnOnce() -> String) -> Result<u64, Report> {
    if value <= u64::from(u32::MAX) {
        return Ok(value);
    }
    Err(report!(
        message: (format!(
            "`convert-to-wasm` cannot lower this `{}`: a Wasm address has 32 bits",
            op.name()
        )),
        labels: [primary(op.span(), format!("{} is {value} bytes, more than 32 bits hold", what()))],
    ))
}

/// An error naming `op` that the layout does not answer what `problem` says.
fn layout_error(op: &Operation, problem: String) -> Report {
    report!(
        message: (format!("`convert-to-wasm` cannot lower this `{}`", op.name())),
        labels: [primary(op.span(), problem)],
    )
}
