//! A `builtin.global_variable`'s initializer as the bytes of a data entry, with fix-ups (decision
//! 4 of the plan).
//!
//! The initializer is generic code that nothing lowers, ending in `builtin.yield %v`. [evaluate]
//! computes `%v` from the operation that defines it, without changing the IR:
//!
//! - an operation that is a constant (`arith.constant`, `arith.constantf`, `builtin.bytes`: what
//!   [matchers::constant] matches) gives its attribute;
//! - `builtin.global_symbol @g` gives the address of the data entry of `g` (no offset), and
//!   `builtin.function_ref @f` the slot of the function `f` in the indirect function table;
//! - any other operation is folded through its own fold ([Foldable::fold_with]) with the values
//!   of its operands that are attributes (an address or a slot is not one, so it is given as
//!   unknown). A fold that gives an attribute gives the value; one that gives one of the
//!   operation's operands gives the value of that operand. Anything else is the error that the
//!   initializer is not a constant the backend can place, naming the operation and the global.
//!
//! The value becomes bytes by the global's type under the data layout in scope
//! ([DataLayout::closest]):
//!
//! - an integer is `size_of` the type bytes, little-endian, zero-extended from its width; a value
//!   wider than the bytes (an `isize` constant is 64 bits wide) must fit them as a signed or an
//!   unsigned number;
//! - a float is its 4 or 8 bytes, which must be `size_of` the type;
//! - bytes must be exactly `size_of` the type long;
//! - an address or a table slot is four placeholder bytes with a [FixupKind::MemoryAddr] or
//!   [FixupKind::TableSlot] fix-up, for a global of a pointer type, a function type or `isize`
//!   whose size is 4 bytes. A function type has no layout of its own: a value of one is a table
//!   slot, carried in an `i32`, so it is given the size and alignment of a pointer in address
//!   space 0.
//!
//! The entry's alignment is `align_of` the type. A value that does not fit the type, or a type
//! whose size the layout cannot give, is an error naming the global.

use alloc::{format, string::String, vec::Vec};

use runic_ir::{
    AttributeRef, DataLayout, FoldResult, OpFoldResult, Operation, Report, SmallVec, Spanned, Type,
    ValueRef,
    attributes::IntegerLikeAttr,
    diagnostics::report,
    dialects::builtin::{
        FunctionReference, GlobalSymbol, GlobalVariable,
        attributes::{BytesAttr, FloatAttr},
    },
    matchers::{self, Matcher},
    traits::Foldable,
};

use super::{
    desc::{DATA_PLACEHOLDER, DataId, Fixup, FixupKind, FuncId},
    instr::Entities,
};

/// The bytes of a global variable, how they are aligned, and the fix-ups in them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DataInit {
    /// The bytes, `size_of` the global's type long.
    pub bytes: Vec<u8>,
    /// The alignment of the global's address, in bytes: `align_of` its type.
    pub align: u32,
    /// The fix-ups in `bytes`: at most one, at offset 0, for an address or a table slot.
    pub fixups: Vec<Fixup>,
}

/// What a value of an initializer is.
#[derive(Clone)]
enum Evaluated {
    /// A constant.
    Attribute(AttributeRef),
    /// The address of a data entry.
    Address(DataId),
    /// The slot of a function in the indirect function table.
    Slot(FuncId),
}

/// The bytes of `global`, a `builtin.global_variable` definition, with the entities its
/// initializer names found through `entities` (see the module docs).
pub fn evaluate(global: &Operation, entities: &mut dyn Entities) -> Result<DataInit, Report> {
    let Some(variable) = global.downcast_ref::<GlobalVariable>() else {
        return Err(cannot_lay_out(global, String::from("it is not a `builtin.global_variable`")));
    };
    let ty = *variable.get_ty();
    let Some(layout) = DataLayout::closest(global) else {
        return Err(cannot_lay_out(global, String::from("no data layout is in scope of it")));
    };
    let Some(value) = variable.initializer_value() else {
        return Err(cannot_lay_out(
            global,
            String::from("its initializer yields no value of its type"),
        ));
    };
    let mut evaluator = Evaluator {
        global,
        entities,
        memo: Vec::new(),
    };
    let evaluated = evaluator.value(value)?;
    let (size, align) = size_and_align(global, &layout, ty)?;
    let mut init = DataInit {
        bytes: Vec::new(),
        align,
        fixups: Vec::new(),
    };
    match evaluated {
        Evaluated::Attribute(attribute) => {
            init.bytes = attribute_bytes(global, ty, size, &attribute)?;
        }
        Evaluated::Address(data) => {
            reference(global, ty, size, "an address", &mut init, FixupKind::MemoryAddr(data))?;
        }
        Evaluated::Slot(function) => {
            reference(global, ty, size, "a table slot", &mut init, FixupKind::TableSlot(function))?;
        }
    }
    Ok(init)
}

/// The size and alignment of `ty` under `layout`, in bytes; a function type has those of a
/// pointer in address space 0. An error naming `global` when the layout cannot give them.
fn size_and_align(global: &Operation, layout: &DataLayout, ty: Type) -> Result<(u64, u32), Report> {
    if ty.is_function() {
        let pointer = layout.pointer(0);
        return Ok((u64::from(pointer.size_bits / 8), pointer.align.bytes() as u32));
    }
    let laid_out = layout.layout_of(ty).map_err(|error| {
        cannot_lay_out(
            global,
            format!("its type `{ty}` has no size under the data layout in scope: {error}"),
        )
    })?;
    Ok((laid_out.size(), laid_out.align().bytes() as u32))
}

/// Four placeholder bytes and a fix-up of `kind` for `init`, when `ty` is a pointer, a function
/// type or `isize` of 4 bytes; otherwise the error that the value (`what`) does not fit.
fn reference(
    global: &Operation,
    ty: Type,
    size: u64,
    what: &str,
    init: &mut DataInit,
    kind: FixupKind,
) -> Result<(), Report> {
    let holds = ty.is_pointer() || ty.is_function() || ty.is_isize();
    if !holds || size != DATA_PLACEHOLDER.len() as u64 {
        return Err(does_not_fit(
            global,
            format!(
                "its initializer gives {what}, which is 4 bytes and is held only by a pointer, a \
                 function type or `isize` of 4 bytes, and its type is `{ty}` of {size} bytes"
            ),
        ));
    }
    init.bytes = DATA_PLACEHOLDER.to_vec();
    init.fixups.push(Fixup {
        offset: 0,
        kind,
        addend: 0,
    });
    Ok(())
}

/// The `size` bytes of the constant `attribute` for a global of type `ty`.
fn attribute_bytes(
    global: &Operation,
    ty: Type,
    size: u64,
    attribute: &AttributeRef,
) -> Result<Vec<u8>, Report> {
    let attribute = attribute.borrow();
    if let Some(int) = attribute.as_attr().as_trait::<dyn IntegerLikeAttr>() {
        let int = int.as_int();
        let bits = size.saturating_mul(8);
        let fits = u64::from(int.width()) <= bits
            || (bits < 128 && {
                let value = int.signed();
                let low = -(1i128 << (bits - 1).min(126));
                let high = 1i128 << bits;
                value >= low && value < high
            });
        if !fits {
            return Err(does_not_fit(
                global,
                format!(
                    "its initializer gives {}, and its type `{ty}` is {size} bytes",
                    int.signed()
                ),
            ));
        }
        // An integer no wider than the bytes is written zero-extended; a wider one that fits is
        // written as its low bytes.
        let pattern = if u64::from(int.width()) <= bits {
            int.unsigned()
        } else {
            int.signed() as u128
        };
        let mut bytes = Vec::with_capacity(size as usize);
        for i in 0..size {
            bytes.push(if i < 16 {
                (pattern >> (8 * i)) as u8
            } else {
                0
            });
        }
        return Ok(bytes);
    }
    if let Some(float) = attribute.downcast_ref::<FloatAttr>() {
        let float = *float.as_value();
        let width = u64::from(float.width() / 8);
        if width != size {
            return Err(does_not_fit(
                global,
                format!(
                    "its initializer gives a {}-bit float, and its type `{ty}` is {size} bytes",
                    float.width()
                ),
            ));
        }
        return Ok(float.bits().to_le_bytes()[..width as usize].to_vec());
    }
    if let Some(bytes) = attribute.downcast_ref::<BytesAttr>() {
        let bytes = bytes.as_value().as_slice();
        if bytes.len() as u64 != size {
            return Err(does_not_fit(
                global,
                format!(
                    "its initializer gives {} bytes, and its type `{ty}` is {size} bytes",
                    bytes.len()
                ),
            ));
        }
        return Ok(bytes.to_vec());
    }
    Err(does_not_fit(
        global,
        format!(
            "its initializer gives `{}`, which is not bytes, an integer or a float",
            OpFoldResult::Attribute(attribute.as_attribute_ref())
        ),
    ))
}

/// Evaluates the values of one initializer.
struct Evaluator<'a> {
    global: &'a Operation,
    entities: &'a mut dyn Entities,
    /// The values evaluated so far.
    memo: Vec<(ValueRef, Evaluated)>,
}

impl Evaluator<'_> {
    /// What `value` is (see the module docs).
    fn value(&mut self, value: ValueRef) -> Result<Evaluated, Report> {
        if let Some((_, known)) = self.memo.iter().find(|(seen, _)| ValueRef::ptr_eq(seen, &value))
        {
            return Ok(known.clone());
        }
        let evaluated = self.compute(value)?;
        self.memo.push((value, evaluated.clone()));
        Ok(evaluated)
    }

    fn compute(&mut self, value: ValueRef) -> Result<Evaluated, Report> {
        let Some(defining) = value.borrow().get_defining_op() else {
            return Err(cannot_lay_out(
                self.global,
                String::from("its initializer uses a value no operation defines"),
            ));
        };
        let op = defining.borrow();
        if let Some(symbol) = op.downcast_ref::<GlobalSymbol>() {
            return self.entities.data(&op, &symbol.symbol()).map(Evaluated::Address);
        }
        if let Some(reference) = op.downcast_ref::<FunctionReference>() {
            return self.entities.function(&op, &reference.callee()).map(Evaluated::Slot);
        }
        if let Some(attribute) = matchers::constant().matches(&op) {
            return Ok(Evaluated::Attribute(attribute));
        }
        let operands = op
            .operands()
            .iter()
            .map(|operand| operand.borrow().as_value_ref())
            .collect::<Vec<_>>();
        let mut constants = Vec::with_capacity(operands.len());
        for operand in &operands {
            constants.push(match self.value(*operand)? {
                Evaluated::Attribute(attribute) => Some(attribute),
                Evaluated::Address(_) | Evaluated::Slot(_) => None,
            });
        }
        let index = op
            .results()
            .iter()
            .position(|result| ValueRef::ptr_eq(&result.borrow().as_value_ref(), &value));
        let mut results = SmallVec::<[OpFoldResult; 1]>::new();
        let folded = match op.fold_with(&constants, &mut results) {
            FoldResult::Ok(()) => index.and_then(|index| results.into_iter().nth(index)),
            FoldResult::InPlace | FoldResult::Failed => None,
        };
        match folded {
            Some(OpFoldResult::Attribute(attribute)) => Ok(Evaluated::Attribute(attribute)),
            Some(OpFoldResult::Value(folded))
                if !ValueRef::ptr_eq(&folded, &value)
                    && operands.iter().any(|operand| ValueRef::ptr_eq(operand, &folded)) =>
            {
                drop(op);
                self.value(folded)
            }
            _ => Err(not_constant(&op, self.global)),
        }
    }
}

/// The name of the global variable `global` for a message: `@g`.
fn name(global: &Operation) -> String {
    match global.downcast_ref::<GlobalVariable>() {
        Some(variable) => format!("@{}", variable.get_name().as_str()),
        None => format!("`{}`", global.name()),
    }
}

/// The error of an initializer whose value `op` defines and does not fold to a constant.
fn not_constant(op: &Operation, global: &Operation) -> Report {
    report!(
        message: (format!(
            "the initializer of the global variable `{}` is not a constant the backend can place",
            name(global)
        )),
        labels: [
            primary(
                op.span(),
                format!(
                    "`{}` does not fold to a constant: the value of a global variable is bytes, \
                     an integer, a float, the address of a global variable or a function",
                    op.name()
                )
            ),
            context(global.span(), "the global variable it initializes"),
        ],
    )
}

/// The error of a global variable `global` whose value does not fit its type, with `label`.
fn does_not_fit(global: &Operation, label: String) -> Report {
    report!(
        message: (format!("the value of the global variable `{}` does not fit its type", name(global))),
        labels: [primary(global.span(), label)],
    )
}

/// The error of a global variable `global` that cannot be laid out, with `label`.
fn cannot_lay_out(global: &Operation, label: String) -> Report {
    report!(
        message: (format!("the global variable `{}` cannot be laid out in linear memory", name(global))),
        labels: [primary(global.span(), label)],
    )
}
