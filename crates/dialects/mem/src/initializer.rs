//! What a global variable's initializer denotes: the bytes of the global and the addresses in
//! them, under a data layout ([evaluate_initializer]).
//!
//! The initializer of a `builtin.global_variable` is a region of ordinary operations ending in
//! `builtin.yield %v`. [evaluate_initializer] computes `%v` from the operation that defines it,
//! without changing the IR, and places it in the bytes of the global:
//!
//! 1. **A constant**, what [matchers::constant] matches, or any other operation that folds to an
//!    attribute through its own fold ([Foldable::fold_with]) given the values of its operands that
//!    are constants (an operand that is an address, a function, a zero or an aggregate is unknown
//!    to the fold). A fold that gives one of the operation's operands gives that operand's value.
//!    An integer has the width of its type (`N` for `iN`, the layout's `isize` width for `isize`)
//!    and is a number of `size_of` its type bytes in the layout's byte order, zero-extended from
//!    that width; a constant wider than its type (an `isize` constant is 64 bits wide) must fit
//!    the type's width as a signed or an unsigned number, and is truncated to it. A float is its
//!    bytes, zero-extended to `size_of` its type in the same way. Bytes are written as they are,
//!    and must be `size_of` the type long.
//! 2. **`builtin.zero`**: `size_of` its type zero bytes.
//! 3. **An address**, the address of a global variable plus a constant: `builtin.global_symbol @g`
//!    is `@g` plus 0; `mem.gep` of an address adds `index * size_of(pointee)` and the offset of the
//!    element its path selects, where the index, if there is one, is a constant integer (signed;
//!    the sum must fit an `i64`); `mem.ptrcast` of an address is the address; `mem.ptrtoint` of an
//!    address is the address as an integer, which must be exactly as wide as the pointer; and
//!    `mem.inttoptr` of such an integer is the address, as a pointer exactly as wide as the
//!    integer. An address is a [Relocation] of the pointer's (or the integer's) size, over zero
//!    bytes. `mem.inttoptr` of an integer constant (or of a zero) exactly as wide as the pointer
//!    is the pointer with that value: bytes, as a constant integer of the pointer's width, with no
//!    relocation; so a null pointer and an absolute address can be written.
//! 4. **A function**: `builtin.function_ref @f` is a [Relocation] of the size the layout's `fn`
//!    entry gives a function value, naming the function the reference resolves to through
//!    aliases.
//! 5. **`builtin.aggregate`**: element `i` is placed at the offset the layout gives element `i` of
//!    the aggregate's type; the bytes no element covers are zero. The elements of the type must
//!    not overlap under the layout (no builtin type's do; a dialect's own aggregate type could). An
//!    element value used by two aggregates is evaluated once and placed twice.
//!
//! Anything else is the error that the initializer is not a constant that can be placed, with a
//! label on the operation that says why. Among what is refused:
//!
//! - a constant that does not fit the width of its type, and a float wider than `size_of` its type;
//! - `builtin.bytes` whose length is not `size_of` its type, as when the layout pads the elements
//!   of `array<i8; N>` (`i8 = 2`): bytes are not laid over padded elements;
//! - arithmetic on an address, and an operation that does not fold;
//! - an address cast to or from an integer of another width than the pointer's;
//! - an aggregate whose type's elements overlap under the layout;
//! - a value of a type that has no layout.
//!
//! Each value is evaluated once (the evaluation is memoised by value) and without recursion over
//! the operation graph. A value that is known to be wholly zero (a `builtin.zero`, a constant whose
//! bytes are all zero, a value of a type of size 0, and an aggregate of such values) is recognised
//! without building its bytes and is not walked when it is placed, however often it is shared, so
//! `builtin.zero` of a megabyte array allocates nothing.

use alloc::{format, rc::Rc, string::String, vec::Vec};
use core::fmt;

use runic_ir::{
    AggregateType, AttributeRef, DataLayout, Endianness, FoldResult, FxHashMap, FxHashSet,
    LayoutError, Op, OpFoldResult, Operation, OperationRef, Report, SmallVec, Spanned, Symbol,
    Type, ValueRef,
    attributes::IntegerLikeAttr,
    diagnostics::report,
    dialects::builtin::{
        Aggregate, FunctionReference, GlobalSymbol, GlobalVariable, GlobalVariableRef, Zero,
        attributes::{BytesAttr, FloatAttr},
    },
    matchers::{self, Matcher},
    traits::Foldable,
};

use crate::{Gep, IntToPtr, PtrCast, PtrToInt};

/// What a global variable's initializer denotes: the bytes of the global and the addresses in them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Initializer {
    /// `size_of` the global's type, in bytes.
    pub size: u64,
    /// `align_of` the global's type, in bytes.
    pub align: u64,
    /// Empty when every byte is zero and there is no relocation; otherwise exactly `size` bytes.
    /// The bytes under a relocation are zero.
    pub bytes: Vec<u8>,
    /// The addresses in the bytes, in ascending order of offset; they do not overlap.
    pub relocations: Vec<Relocation>,
}

impl Initializer {
    /// True if every byte of the global is zero and there is no relocation: `bytes` is empty.
    pub fn is_zero(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// An address in the bytes of a global variable, which the consumer of an [Initializer] writes
/// over the zero bytes it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relocation {
    /// Where in the bytes.
    pub offset: u64,
    /// How many bytes: the size of a pointer, or of a function value.
    pub size: u8,
    /// What the address is the address of.
    pub target: Target,
    /// Added to the address of a `Data` target; always 0 for a `Function`.
    pub addend: i64,
}

/// What a [Relocation] is the address of.
#[derive(Clone, PartialEq, Eq)]
pub enum Target {
    /// The address of a global variable.
    Data(GlobalVariableRef),
    /// The function (the definition or declaration an alias leads to): the operation that
    /// implements the callable, a `builtin.function` in IR that verifies.
    Function(OperationRef),
}

/// `Data(@g)` or `Function(@f)`: the symbol named, or the operation's name if it is not a symbol.
impl fmt::Debug for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(global) => write!(f, "Data(@{})", global.borrow().get_name().as_str()),
            Self::Function(op) => {
                let op = op.borrow();
                match op.as_trait::<dyn Symbol>() {
                    Some(symbol) => write!(f, "Function(@{})", symbol.name()),
                    None => write!(f, "Function({})", op.name()),
                }
            }
        }
    }
}

/// The bytes and relocations of `global`, a `builtin.global_variable` definition, under `layout`
/// (see the module docs). The IR is not changed.
///
/// A declaration (a global with no initializer) is an error, as is an initializer that does not
/// end in `builtin.yield` of a value of the global's type, a value that cannot be placed, and a
/// type with no layout under `layout`. Every error names the global; one about a value has its
/// primary label on the operation that defines the value.
pub fn evaluate_initializer(
    global: &GlobalVariable,
    layout: &DataLayout,
) -> Result<Initializer, Report> {
    let op = global.as_operation();
    let name = String::from(global.get_name().as_str());
    if global.initializer().is_empty() {
        return Err(report!(
            message: (format!("the global variable `@{name}` has no initializer to evaluate")),
            labels: [primary(
                op.span(),
                "this global variable is a declaration: its bytes are defined elsewhere"
            )],
        ));
    }
    let ty = *global.get_ty();
    let Some(value) = global.initializer_value() else {
        return Err(report!(
            message: (not_placeable(&name)),
            labels: [primary(
                op.span(),
                format!("its initializer does not end in `builtin.yield` of a value of its type `{ty}`")
            )],
        ));
    };
    let mut evaluator = Evaluator {
        global: op,
        name,
        layout,
        entries: Vec::new(),
        by_value: FxHashMap::default(),
    };
    let root = evaluator.evaluate(value)?;
    let defining = evaluator.entries[root].op;
    let laid_out = layout
        .layout_of(ty)
        .map_err(|error| evaluator.no_layout(defining, ty, &error))?;
    let mut out = Output {
        size: laid_out.size(),
        bytes: Vec::new(),
        relocations: Vec::new(),
    };
    evaluator.place(root, &mut out)?;
    out.relocations.sort_by_key(|relocation| relocation.offset);
    Ok(Initializer {
        size: out.size,
        align: laid_out.align().bytes(),
        bytes: out.bytes,
        relocations: out.relocations,
    })
}

/// The message of every error about a value of the initializer of `@name`.
fn not_placeable(name: &str) -> String {
    format!("the initializer of `@{name}` is not a constant that can be placed")
}

/// The bytes `start..end` that element `index` of an aggregate takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ElementRange {
    pub index: u32,
    pub start: u64,
    pub end: u64,
}

/// Two of `ranges` that share a byte, the one that starts first first; `None` when no two do.
/// An empty range shares no byte. `ranges` is sorted by where each starts.
pub(crate) fn first_overlap(ranges: &mut [ElementRange]) -> Option<(ElementRange, ElementRange)> {
    ranges.sort_by_key(|range| (range.start, range.index));
    let mut widest: Option<ElementRange> = None;
    for range in ranges.iter().filter(|range| range.start < range.end) {
        if let Some(previous) = widest {
            if range.start < previous.end {
                return Some((previous, *range));
            }
            if range.end > previous.end {
                widest = Some(*range);
            }
        } else {
            widest = Some(*range);
        }
    }
    None
}

/// What a value of an initializer is.
#[derive(Clone)]
enum Kind {
    /// A constant.
    Constant(AttributeRef),
    /// Zero bytes, `size_of` the value's type.
    Zero,
    /// The address of `target` plus `addend`.
    Address {
        target: GlobalVariableRef,
        addend: i64,
    },
    /// A function value: the callable it names, aliases resolved.
    Function(OperationRef),
    /// A struct or an array: the entries of its elements.
    Aggregate { elements: Rc<[usize]> },
}

/// An evaluated value.
struct Entry {
    /// The value's type, which decides how it is placed.
    ty: Type,
    /// The operation that defines the value, which the errors about it name.
    op: OperationRef,
    kind: Kind,
    /// True if the value is known to be zero bytes with no relocation without placing it: a value
    /// whose type has size 0, a zero, a constant whose bytes are all zero, and an aggregate of
    /// such values. Placing it walks nothing.
    zero: bool,
}

/// How a constant is written: what [Evaluator::encode] gives.
enum Encoding {
    /// The bytes of a number (an integer or a float), least significant first, at most `size_of`
    /// the type; the rest of `size_of` is zero, on the side of the most significant byte.
    Number(SmallVec<[u8; 16]>),
    /// The bytes of a `builtin.bytes` constant, exactly `size_of` the type, written as they are.
    Bytes,
}

/// The bytes of the global being built: allocated at the first non-zero byte or relocation.
struct Output {
    size: u64,
    bytes: Vec<u8>,
    relocations: Vec<Relocation>,
}

/// Evaluates the values of one initializer.
struct Evaluator<'a> {
    /// The global variable, which every error names.
    global: &'a Operation,
    /// Its name, without the `@`.
    name: String,
    layout: &'a DataLayout,
    /// The values evaluated so far.
    entries: Vec<Entry>,
    /// The entry of each value evaluated so far.
    by_value: FxHashMap<ValueRef, usize>,
}

impl Evaluator<'_> {
    /// The entry of `value`, evaluating it and every value it depends on, in post-order with an
    /// explicit stack.
    fn evaluate(&mut self, value: ValueRef) -> Result<usize, Report> {
        let mut stack = alloc::vec![(value, false)];
        let mut in_progress = FxHashSet::<ValueRef>::default();
        while let Some((current, expanded)) = stack.pop() {
            if self.by_value.contains_key(&current) {
                continue;
            }
            let Some(defining) = current.borrow().get_defining_op() else {
                return Err(report!(
                    message: (not_placeable(&self.name)),
                    labels: [primary(
                        self.global.span(),
                        "its initializer uses a value that no operation defines"
                    )],
                ));
            };
            if expanded {
                let index = self.compute(current, defining)?;
                self.by_value.insert(current, index);
                in_progress.remove(&current);
                continue;
            }
            if !in_progress.insert(current) {
                return Err(self.error(defining, "the value of this operation depends on itself"));
            }
            stack.push((current, true));
            let op = defining.borrow();
            for operand in op.operands().iter() {
                let operand = operand.borrow().as_value_ref();
                if !self.by_value.contains_key(&operand) {
                    stack.push((operand, false));
                }
            }
        }
        Ok(self.by_value[&value])
    }

    /// The entry of an operand of an operation being computed, which [Self::evaluate] evaluated
    /// first.
    fn operand(&self, op: OperationRef, value: ValueRef) -> Result<usize, Report> {
        self.by_value
            .get(&value)
            .copied()
            .ok_or_else(|| self.error(op, "the value of this operation depends on itself"))
    }

    /// Adds an entry for a value of type `ty` defined by `op`.
    fn push(&mut self, ty: Type, op: OperationRef, kind: Kind) -> usize {
        let zero = matches!(self.layout.size_of(ty), Ok(0))
            || match &kind {
                Kind::Zero => true,
                Kind::Aggregate { elements } => {
                    elements.iter().all(|element| self.entries[*element].zero)
                }
                Kind::Constant(attribute) => self.is_zero_constant(ty, attribute),
                Kind::Address { .. } | Kind::Function(_) => false,
            };
        self.entries.push(Entry { ty, op, kind, zero });
        self.entries.len() - 1
    }

    /// True if the constant `attribute` of type `ty` is placed as zero bytes: it can be placed
    /// (the errors [Self::encode] gives are left to placement) and every byte it writes is zero.
    fn is_zero_constant(&self, ty: Type, attribute: &AttributeRef) -> bool {
        let Ok(size) = self.layout.size_of(ty) else {
            return false;
        };
        match self.encode(ty, attribute, size) {
            Ok(Encoding::Number(bytes)) => bytes.iter().all(|byte| *byte == 0),
            Ok(Encoding::Bytes) => attribute
                .borrow()
                .downcast_ref::<BytesAttr>()
                .is_some_and(|bytes| bytes.as_value().as_slice().iter().all(|byte| *byte == 0)),
            Err(_) => false,
        }
    }

    /// Evaluates `value`, defined by `defining`, whose operands are evaluated.
    fn compute(&mut self, value: ValueRef, defining: OperationRef) -> Result<usize, Report> {
        let ty = value.borrow().ty();
        let op = defining.borrow();
        let operands = op
            .operands()
            .iter()
            .map(|operand| operand.borrow().as_value_ref())
            .collect::<SmallVec<[ValueRef; 4]>>();
        let mut entries = SmallVec::<[usize; 4]>::with_capacity(operands.len());
        for operand in &operands {
            entries.push(self.operand(defining, *operand)?);
        }

        if let Some(symbol) = op.downcast_ref::<GlobalSymbol>() {
            let Some(target) = symbol.global() else {
                return Err(self.error(defining, "the symbol it names is not a global variable"));
            };
            return Ok(self.push(ty, defining, Kind::Address { target, addend: 0 }));
        }
        if let Some(reference) = op.downcast_ref::<FunctionReference>() {
            let resolved = reference.callee().resolve_callable().map_err(|error| {
                self.error(defining, format!("the callee does not resolve to a callable: {error}"))
            })?;
            let target = resolved.target().as_operation_ref();
            return Ok(self.push(ty, defining, Kind::Function(target)));
        }
        if op.is::<Zero>() {
            return Ok(self.push(ty, defining, Kind::Zero));
        }
        if op.is::<Aggregate>() {
            self.check_disjoint(defining, ty)?;
            let elements = Rc::from(entries.as_slice());
            return Ok(self.push(ty, defining, Kind::Aggregate { elements }));
        }
        if let Some(attribute) = matchers::constant().matches(&op) {
            return Ok(self.push(ty, defining, Kind::Constant(attribute)));
        }

        // An address through `mem`'s address operations
        let base = entries.first().map(|entry| &self.entries[*entry].kind);
        if let Some(Kind::Address { target, addend }) = base.cloned() {
            if let Some(gep) = op.downcast_ref::<Gep>() {
                let addend = self.gep(defining, gep, &entries, &target, addend)?;
                return Ok(self.push(ty, defining, Kind::Address { target, addend }));
            }
            if op.is::<PtrCast>() {
                return Ok(self.push(ty, defining, Kind::Address { target, addend }));
            }
            if op.is::<PtrToInt>() || op.is::<IntToPtr>() {
                let from = self.entries[entries[0]].ty;
                self.check_cast(defining, from, ty)?;
                return Ok(self.push(ty, defining, Kind::Address { target, addend }));
            }
        }

        // `mem.inttoptr` of an integer constant is the pointer with that value
        if op.is::<IntToPtr>()
            && let Some(&operand) = entries.first()
            && matches!(self.entries[operand].kind, Kind::Constant(_) | Kind::Zero)
        {
            let from = self.entries[operand].ty;
            self.check_cast(defining, from, ty)?;
            let kind = self.entries[operand].kind.clone();
            return Ok(self.push(ty, defining, kind));
        }

        // Anything else folds to a constant through its own fold
        let constants = entries
            .iter()
            .map(|entry| match &self.entries[*entry].kind {
                Kind::Constant(attribute) => Some(*attribute),
                _ => None,
            })
            .collect::<SmallVec<[Option<AttributeRef>; 4]>>();
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
            Some(OpFoldResult::Attribute(attribute)) => {
                Ok(self.push(ty, defining, Kind::Constant(attribute)))
            }
            Some(OpFoldResult::Value(folded)) => {
                match operands.iter().position(|operand| ValueRef::ptr_eq(operand, &folded)) {
                    Some(position) => Ok(entries[position]),
                    None => Err(self.not_folded(defining, &op, &entries)),
                }
            }
            None => Err(self.not_folded(defining, &op, &entries)),
        }
    }

    /// The addend of `gep`, an address of `target` plus `addend` (see the module docs).
    fn gep(
        &self,
        defining: OperationRef,
        gep: &Gep,
        entries: &[usize],
        target: &GlobalVariableRef,
        addend: i64,
    ) -> Result<i64, Report> {
        let base = self.entries[entries[0]].ty;
        let Some(pointee) = base.pointee() else {
            return Err(self.error(
                defining,
                format!("the base has the opaque pointer type `{base}`, which a gep cannot step"),
            ));
        };
        let mut delta = 0i128;
        if let Some(&index) = entries.get(1) {
            let index = self.constant_index(defining, index)?;
            let size = self
                .layout
                .size_of(pointee)
                .map_err(|error| self.no_layout(defining, pointee, &error))?;
            delta = index.saturating_mul(i128::from(size));
        }
        let Some(steps) = gep.steps() else {
            return Err(self.error(defining, "the path of this gep has a negative step"));
        };
        if !steps.is_empty() {
            let (offset, _) = self
                .layout
                .offset_of(pointee, &steps)
                .map_err(|error| self.no_layout(defining, pointee, &error))?;
            delta = delta.saturating_add(i128::from(offset));
        }
        i64::try_from(i128::from(addend).saturating_add(delta)).map_err(|_| {
            let target = target.borrow();
            self.error(
                defining,
                format!(
                    "`mem.gep` moves the address of `@{}` plus {addend} by {delta} bytes, which \
                     does not fit a signed 64-bit offset",
                    target.get_name().as_str()
                ),
            )
        })
    }

    /// The value of the entry `index`, the index of a gep, as a signed number.
    fn constant_index(&self, defining: OperationRef, index: usize) -> Result<i128, Report> {
        let entry = &self.entries[index];
        match &entry.kind {
            Kind::Zero => return Ok(0),
            Kind::Constant(attribute) => {
                let attribute = attribute.borrow();
                if let Some(int) = attribute.as_attr().as_trait::<dyn IntegerLikeAttr>() {
                    return Ok(int.as_int().signed());
                }
            }
            _ => {}
        }
        Err(self.error(defining, "the index of this gep is not a constant integer"))
    }

    /// Checks that a `mem.ptrtoint` or `mem.inttoptr` (`defining`) from `from` to `to` keeps
    /// every bit of an address: the integer is exactly as wide as the pointer (decision 4).
    fn check_cast(&self, defining: OperationRef, from: Type, to: Type) -> Result<(), Report> {
        let (pointer, integer) = if from.is_pointer() {
            (from, to)
        } else {
            (to, from)
        };
        let Some(pointer_type) = pointer.as_pointer() else {
            return Err(self.error(defining, "this cast is not between a pointer and an integer"));
        };
        let addrspace = pointer_type.addrspace();
        let pointer_bits = self.layout.pointer(addrspace).size_bits;
        let integer_bits = self.integer_bits(integer);
        if integer_bits == Some(pointer_bits) {
            return Ok(());
        }
        let name = defining.borrow().name();
        let integer_text = match integer_bits {
            Some(bits) => format!("`{integer}` of {bits} bits"),
            None => format!("`{integer}`"),
        };
        Err(self.error(
            defining,
            format!(
                "`{name}` casts an address between `{pointer}`, of {pointer_bits} bits, and \
                 {integer_text}: an address in an integer must be in an integer exactly as wide \
                 as the pointer"
            ),
        ))
    }

    /// The width in bits of the integer type `ty` under the layout: `iN` is `N`, `isize` the
    /// layout's `isize`.
    fn integer_bits(&self, ty: Type) -> Option<u32> {
        if ty.is_isize() {
            Some(self.layout.isize_width())
        } else {
            ty.integer_width()
        }
    }

    /// The error of `op`, `defining`, which does not fold to a constant, with the reason the
    /// kinds of its operands (`entries`) give.
    fn not_folded(&self, defining: OperationRef, op: &Operation, entries: &[usize]) -> Report {
        let name = op.name();
        let kinds = entries.iter().map(|entry| &self.entries[*entry].kind);
        let mut label = format!("`{name}` does not fold to a constant");
        for kind in kinds {
            match kind {
                Kind::Address { .. } => {
                    label = String::from(
                        "arithmetic on an address cannot be placed: write the offset with \
                         `mem.gep`",
                    );
                    break;
                }
                Kind::Function(_) => {
                    label = format!(
                        "`{name}` does not fold to a constant: an operation on a function value \
                         cannot be placed"
                    );
                    break;
                }
                Kind::Constant(_) | Kind::Zero | Kind::Aggregate { .. } => {}
            }
        }
        self.error(defining, label)
    }

    /// Places the value of the entry `root` at offset 0 of `out`, with an explicit stack.
    fn place(&self, root: usize, out: &mut Output) -> Result<(), Report> {
        let mut stack = alloc::vec![(root, 0u64)];
        while let Some((index, offset)) = stack.pop() {
            let entry = &self.entries[index];
            if entry.zero {
                // The type must still have a layout: a zero of a type without one is an error
                self.layout
                    .layout_of(entry.ty)
                    .map_err(|error| self.no_layout(entry.op, entry.ty, &error))?;
                continue;
            }
            match &entry.kind {
                Kind::Zero => {}
                Kind::Aggregate { elements, .. } => {
                    for (i, element) in elements.iter().enumerate().rev() {
                        let at = u32::try_from(i)
                            .ok()
                            .and_then(|i| self.layout.offset_of(entry.ty, &[i]).ok())
                            .map(|(at, _)| at);
                        let Some(at) = at else {
                            let error = self.layout.layout_of(entry.ty).err();
                            return Err(match error {
                                Some(error) => self.no_layout(entry.op, entry.ty, &error),
                                None => self
                                    .error(entry.op, format!("`{}` has no element {i}", entry.ty)),
                            });
                        };
                        stack.push((*element, offset + at));
                    }
                }
                Kind::Constant(attribute) => {
                    let size = self
                        .layout
                        .size_of(entry.ty)
                        .map_err(|error| self.no_layout(entry.op, entry.ty, &error))?;
                    self.place_constant(entry, attribute, size, offset, out)?;
                }
                Kind::Address { target, addend } => {
                    let bits = match entry.ty.as_pointer() {
                        Some(pointer) => Some(self.layout.pointer(pointer.addrspace()).size_bits),
                        None => self.integer_bits(entry.ty),
                    };
                    let Some(bits) = bits else {
                        return Err(self.error(
                            entry.op,
                            format!("an address of type `{}` cannot be placed", entry.ty),
                        ));
                    };
                    self.layout
                        .size_of(entry.ty)
                        .map_err(|error| self.no_layout(entry.op, entry.ty, &error))?;
                    let relocation = Relocation {
                        offset,
                        size: (bits / 8) as u8,
                        target: Target::Data(*target),
                        addend: *addend,
                    };
                    self.relocate(entry.op, relocation, out)?;
                }
                Kind::Function(target) => {
                    self.layout
                        .size_of(entry.ty)
                        .map_err(|error| self.no_layout(entry.op, entry.ty, &error))?;
                    let Some(function) = self.layout.function() else {
                        // A function type has a layout only with a function entry
                        return Err(self.error(entry.op, "a function value has no layout"));
                    };
                    let relocation = Relocation {
                        offset,
                        size: (function.size_bits / 8) as u8,
                        target: Target::Function(*target),
                        addend: 0,
                    };
                    self.relocate(entry.op, relocation, out)?;
                }
            }
        }
        Ok(())
    }

    /// Places the constant `attribute` of `entry`, `size` bytes, at `offset` of `out`.
    fn place_constant(
        &self,
        entry: &Entry,
        attribute: &AttributeRef,
        size: u64,
        offset: u64,
        out: &mut Output,
    ) -> Result<(), Report> {
        let encoding = self.encode(entry.ty, attribute, size).map_err(|label| {
            let name = entry.op.borrow().name();
            self.error(entry.op, format!("`{name}` gives {label}"))
        })?;
        match encoding {
            Encoding::Number(mut bytes) => {
                if bytes.iter().all(|byte| *byte == 0) {
                    return Ok(());
                }
                // A number of `size` bytes in the layout's byte order: its significant bytes are
                // the low addresses under little-endian and the high ones under big-endian
                let at = if self.layout.endianness() == Endianness::Big {
                    bytes.reverse();
                    offset + (size - bytes.len() as u64)
                } else {
                    offset
                };
                self.write(entry.op, at, &bytes, out)
            }
            Encoding::Bytes => {
                let attribute = attribute.borrow();
                let bytes = attribute.downcast_ref::<BytesAttr>().expect("a bytes constant");
                self.write(entry.op, offset, bytes.as_value().as_slice(), out)
            }
        }
    }

    /// How the constant `attribute` of type `ty`, `size` bytes, is written (see the module docs),
    /// or what it gives and why that cannot be placed, for a label that begins "`op` gives".
    ///
    /// An integer is truncated to the width of `ty` (`N` for `iN`, the layout's `isize` width,
    /// the pointer's width under `mem.inttoptr`), which it must fit as a signed or an unsigned
    /// number, and zero-extended to `size` bytes. A float is its bytes, zero-extended likewise.
    fn encode(&self, ty: Type, attribute: &AttributeRef, size: u64) -> Result<Encoding, String> {
        let attribute = attribute.borrow();
        let size_bits = size.saturating_mul(8);
        if let Some(int) = attribute.as_attr().as_trait::<dyn IntegerLikeAttr>() {
            let int = int.as_int();
            let width = self
                .constant_width(ty)
                .map_or(size_bits, |width| u64::from(width).min(size_bits));
            let pattern = if u64::from(int.width()) <= width {
                int.unsigned()
            } else {
                let value = int.signed();
                // `width` is below the integer's own width, which is at most 128: a signed
                // number of `width` bits, or an unsigned one
                let high = if width >= 127 {
                    i128::MAX
                } else {
                    (1i128 << width) - 1
                };
                let fits = width > 0 && value >= -(1i128 << (width - 1)) && value <= high;
                if !fits {
                    return Err(format!("{value}, which does not fit `{ty}` of {width} bits"));
                }
                (value as u128) & ((1u128 << width) - 1)
            };
            let len = width.div_ceil(8).min(16) as usize;
            return Ok(Encoding::Number(SmallVec::from_slice(&pattern.to_le_bytes()[..len])));
        }
        if let Some(float) = attribute.downcast_ref::<FloatAttr>() {
            let float = *float.as_value();
            let width = u64::from(float.width() / 8);
            if width > size {
                return Err(format!(
                    "a {}-bit float, which does not fit `{ty}` of {size} bytes",
                    float.width()
                ));
            }
            let bits = float.bits().to_le_bytes();
            return Ok(Encoding::Number(SmallVec::from_slice(&bits[..width as usize])));
        }
        if let Some(bytes) = attribute.downcast_ref::<BytesAttr>() {
            let len = bytes.as_value().as_slice().len() as u64;
            if len == size {
                return Ok(Encoding::Bytes);
            }
            let padded = ty.as_array().and_then(|array| {
                let element = array.element();
                let stride = self.layout.size_of(element).ok()?;
                (stride > 1).then_some((element, stride))
            });
            return Err(match padded {
                Some((element, stride)) => {
                    format!(
                        "{len} bytes, but `{ty}` is {size} bytes: the layout pads its elements \
                         (each `{element}` takes {stride} bytes), and bytes cannot be laid over \
                         padded elements"
                    )
                }
                None => format!("{len} bytes, but `{ty}` is {size} bytes"),
            });
        }
        Err(format!(
            "`{}`, which is not bytes, an integer or a float",
            OpFoldResult::Attribute(attribute.as_attribute_ref())
        ))
    }

    /// The width in bits of an integer constant of type `ty` under the layout: `iN` is `N`,
    /// `isize` the layout's `isize`, and a pointer (a constant given by `mem.inttoptr`) the
    /// pointer's width. `None` for any other type.
    fn constant_width(&self, ty: Type) -> Option<u32> {
        match ty.as_pointer() {
            Some(pointer) => Some(self.layout.pointer(pointer.addrspace()).size_bits),
            None => self.integer_bits(ty),
        }
    }

    /// An error naming `ty` when the elements of `ty`, an aggregate type `defining` builds, overlap
    /// under the layout: placing them would write one element's bytes over another's. The
    /// elements of an array never overlap; a type whose layout is not known is reported when it
    /// is placed.
    fn check_disjoint(&self, defining: OperationRef, ty: Type) -> Result<(), Report> {
        if ty.as_array().is_some() {
            return Ok(());
        }
        let Some(count) = ty.as_trait::<dyn AggregateType>().map(|ty| ty.element_count()) else {
            return Ok(());
        };
        let mut ranges = Vec::new();
        for index in 0..u32::try_from(count).unwrap_or(u32::MAX) {
            let Ok((at, element)) = self.layout.offset_of(ty, &[index]) else {
                return Ok(());
            };
            let Ok(size) = self.layout.size_of(element) else {
                return Ok(());
            };
            ranges.push(ElementRange {
                index,
                start: at,
                end: at.saturating_add(size),
            });
        }
        let Some((first, second)) = first_overlap(&mut ranges) else {
            return Ok(());
        };
        Err(self.error(
            defining,
            format!(
                "the elements of `{ty}` overlap under the data layout: element {} takes bytes \
                 {}..{} and element {} bytes {}..{}, so they cannot both be placed",
                first.index, first.start, first.end, second.index, second.start, second.end
            ),
        ))
    }

    /// Writes `bytes` at `offset` of `out`, unless every one of them is zero.
    fn write(
        &self,
        op: OperationRef,
        offset: u64,
        bytes: &[u8],
        out: &mut Output,
    ) -> Result<(), Report> {
        if bytes.iter().all(|byte| *byte == 0) {
            return Ok(());
        }
        self.materialize(out)?;
        let start = usize::try_from(offset).ok();
        let range = start.and_then(|start| Some(start..start.checked_add(bytes.len())?));
        match range.and_then(|range| out.bytes.get_mut(range)) {
            Some(slot) => {
                slot.copy_from_slice(bytes);
                Ok(())
            }
            None => Err(self.error(op, "this value lies outside the bytes of the global")),
        }
    }

    /// Adds `relocation` to `out`.
    fn relocate(
        &self,
        op: OperationRef,
        relocation: Relocation,
        out: &mut Output,
    ) -> Result<(), Report> {
        let end = relocation.offset.checked_add(u64::from(relocation.size));
        if end.is_none_or(|end| end > out.size) {
            return Err(self.error(op, "this value lies outside the bytes of the global"));
        }
        self.materialize(out)?;
        out.relocations.push(relocation);
        Ok(())
    }

    /// Makes the bytes of `out` `size` zero bytes, if they are not yet.
    fn materialize(&self, out: &mut Output) -> Result<(), Report> {
        if !out.bytes.is_empty() || out.size == 0 {
            return Ok(());
        }
        let size = usize::try_from(out.size).ok();
        let reserved = size.is_some_and(|size| out.bytes.try_reserve_exact(size).is_ok());
        let Some(size) = size.filter(|_| reserved) else {
            return Err(report!(
                message: (not_placeable(&self.name)),
                labels: [primary(
                    self.global.span(),
                    format!("its {} bytes cannot be held in memory", out.size)
                )],
            ));
        };
        out.bytes.resize(size, 0);
        Ok(())
    }

    /// The error that the value of type `ty` that `op` defines cannot be placed: `error`.
    fn no_layout(&self, op: OperationRef, ty: Type, error: &LayoutError) -> Report {
        self.error(op, format!("a value of type `{ty}` cannot be placed: {error}"))
    }

    /// The error about the value `op` defines, with `label` on it.
    fn error(&self, op: OperationRef, label: impl Into<String>) -> Report {
        let span = op.borrow().span();
        let label: String = label.into();
        report!(
            message: (not_placeable(&self.name)),
            labels: [
                primary(span, label),
                context(self.global.span(), "the global variable it initializes"),
            ],
        )
    }
}
