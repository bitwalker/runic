//! The builder trait of the dialect.

use runic_ir::{
    AsCallableSymbolRef, Builder, BuilderExt, Float, Ident, InsertionGuard, Int, Op, OpBuilder,
    OpRegistration, OperationRef, ReadyAttr, Report, SourceSpan, Spanned, SymbolName, SymbolTable,
    Type, Value, ValueRef, Visibility,
    attributes::AttrList,
    dialects::builtin::{
        FunctionBuilder, GlobalVariableRef,
        attributes::{Array, List, Signature, SymbolRef, SymbolRefAttr},
    },
};

use crate::*;

/// Gives the operation `owner` a result of each of the types `results`.
fn add_results(
    context: &runic_ir::Context,
    mut owner: runic_ir::OperationRef,
    results: &[Type],
    span: SourceSpan,
) {
    for ty in results {
        let result = context.make_result(span, *ty, owner, 0);
        owner.borrow_mut().results_mut().push(result);
    }
}

/// The types of the operands of `op`.
fn operand_types(op: runic_ir::OperationRef) -> alloc::vec::Vec<Type> {
    op.borrow()
        .operands()
        .iter()
        .map(|o| o.borrow().as_value_ref().borrow().ty())
        .collect()
}

/// Defines the builder methods of the binary operations (two operands of one type).
macro_rules! binary_builders {
    ($($method:ident, $Op:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, ".")]
        fn $method(
            &mut self,
            lhs: ValueRef,
            rhs: ValueRef,
            span: SourceSpan,
        ) -> Result<ValueRef, Report> {
            let op_builder = self.builder_mut().create::<$Op, _>(span);
            let op = op_builder(lhs, rhs)?;
            Ok(op.borrow().result().as_value_ref())
        }
    )*};
}

/// Defines the builder methods of the operations of one operand and a result that follows from it.
macro_rules! unary_builders {
    ($($method:ident, $Op:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, ".")]
        fn $method(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
            let op_builder = self.builder_mut().create::<$Op, _>(span);
            let op = op_builder(operand)?;
            Ok(op.borrow().result().as_value_ref())
        }
    )*};
}

/// Defines the builder methods of the conversions whose result type the caller names.
macro_rules! typed_conversion_builders {
    ($($method:ident, $Op:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, ".")]
        fn $method(
            &mut self,
            operand: ValueRef,
            ty: Type,
            span: SourceSpan,
        ) -> Result<ValueRef, Report> {
            let op_builder = self.builder_mut().create::<$Op, _>(span);
            let op = op_builder(operand, ty)?;
            Ok(op.borrow().result().as_value_ref())
        }
    )*};
}

/// Defines the builder methods of the conversions whose result type is fixed.
macro_rules! fixed_conversion_builders {
    ($($method:ident, $Op:ident, $result:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, ".")]
        fn $method(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
            let ty = self.builder().context().$result();
            let op_builder = self.builder_mut().create::<$Op, _>(span);
            let op = op_builder(operand, ty)?;
            Ok(op.borrow().result().as_value_ref())
        }
    )*};
}

/// Checks that `builder` can define the symbol `name` of the kind `T` where it inserts: a symbol
/// whose name is taken in the enclosing symbol table is an error, where inserting it would fail an
/// assertion. A builder with no symbol table at its insertion point defines anything.
fn ensure_definable<T: OpRegistration, B: ?Sized + Builder>(
    builder: &B,
    name: SymbolName,
) -> Result<(), Report> {
    let Some(owner) = builder.insertion_block().and_then(|block| block.borrow().parent_op()) else {
        return Ok(());
    };
    let table = if owner.name().implements::<dyn SymbolTable>() {
        owner
    } else {
        let Some(table) = owner.borrow().nearest_symbol_table() else {
            return Ok(());
        };
        table
    };
    let kind = builder
        .context()
        .get_or_register_dialect::<T::Dialect>()
        .expect_registered_name::<T>();
    table
        .borrow()
        .as_trait::<dyn SymbolTable>()
        .expect("a symbol table")
        .can_define(name, &kind)
        .map_err(Report::msg)
}

/// Builds the operations of the `wasm` dialect, returning their results.
///
/// A method returns the error of the operation's type inference, which is none for these
/// operations; the operand and result types are checked when the function that holds the
/// operation is verified.
pub trait WasmOpBuilder<'f, B: ?Sized + Builder> {
    /// The integer constant `value` of type `ty`, an `i32` or an `i64` (any other type does not
    /// verify).
    fn int(&mut self, value: Int, ty: Type, span: SourceSpan) -> ValueRef {
        let attr = self.builder().context_rc().int_attr(value, ty);
        let op_builder = self.builder_mut().create::<Const, _>(span);
        let constant = op_builder(ReadyAttr(attr)).unwrap();
        constant.borrow().result().as_value_ref()
    }

    /// The `i32` constant holding the bits of `value`.
    fn i32(&mut self, value: i32, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().i32();
        self.int(Int::from(value), ty, span)
    }

    /// The `i64` constant holding the bits of `value`.
    fn i64(&mut self, value: i64, span: SourceSpan) -> ValueRef {
        let ty = self.builder().context().i64();
        self.int(Int::from(value), ty, span)
    }

    /// The float constant `value`, of the type its width gives: `f32` for a 32-bit value, `f64`
    /// for a 64-bit one.
    fn float(&mut self, value: Float, span: SourceSpan) -> ValueRef {
        let attr = self.builder().context_rc().float_attr(value);
        let op_builder = self.builder_mut().create::<ConstF, _>(span);
        let constant = op_builder(ReadyAttr(attr)).unwrap();
        constant.borrow().result().as_value_ref()
    }

    /// The `f32` constant `value`, bit for bit (a NaN keeps its payload).
    fn f32(&mut self, value: f32, span: SourceSpan) -> ValueRef {
        self.float(Float::from_f32(value), span)
    }

    /// The `f64` constant `value`, bit for bit (a NaN keeps its payload).
    fn f64(&mut self, value: f64, span: SourceSpan) -> ValueRef {
        self.float(Float::from_f64(value), span)
    }

    /// `if_true` if `condition`, an `i32`, is not zero, and `if_false` otherwise: `select`.
    fn select(
        &mut self,
        condition: ValueRef,
        if_true: ValueRef,
        if_false: ValueRef,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Select, _>(span);
        let op = op_builder(condition, if_true, if_false)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// The `i32` 1 if `operand`, an integer, is zero, and 0 otherwise.
    fn eqz(&mut self, operand: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Eqz, _>(span);
        let op = op_builder(operand)?;
        Ok(op.borrow().result().as_value_ref())
    }

    binary_builders! {
        add, Add, "sum";
        sub, Sub, "difference";
        mul, Mul, "product";
        div_s, DivS, "signed quotient; traps on a zero divisor and on the minimum value divided by -1";
        div_u, DivU, "unsigned quotient; traps on a zero divisor";
        rem_s, RemS, "signed remainder; traps on a zero divisor";
        rem_u, RemU, "unsigned remainder; traps on a zero divisor";
        and, And, "bitwise and";
        or, Or, "bitwise or";
        xor, Xor, "bitwise exclusive or";
        shl, Shl, "value shifted left by the count modulo the width";
        shr_s, ShrS, "value shifted right, sign-filling, by the count modulo the width";
        shr_u, ShrU, "value shifted right, zero-filling, by the count modulo the width";
        rotl, Rotl, "value rotated left by the count modulo the width";
        rotr, Rotr, "value rotated right by the count modulo the width";
        div, Div, "float quotient; never traps";
        min, Min, "float minimum; a NaN if either operand is one";
        max, Max, "float maximum; a NaN if either operand is one";
        copysign, Copysign, "first float with the sign of the second";
    }

    unary_builders! {
        clz, Clz, "count of leading zero bits";
        ctz, Ctz, "count of trailing zero bits";
        popcnt, Popcnt, "count of set bits";
        abs, Abs, "float absolute value";
        neg, Neg, "float negation";
        sqrt, Sqrt, "float square root";
        ceil, Ceil, "float rounded up";
        floor, Floor, "float rounded down";
        trunc, Trunc, "float rounded toward zero";
        nearest, Nearest, "float rounded to nearest, ties to even";
        extend8_s, Extend8S, "sign extension of the low 8 bits";
        extend16_s, Extend16S, "sign extension of the low 16 bits";
        extend32_s, Extend32S, "sign extension of the low 32 bits of an `i64`";
    }

    binary_builders! {
        eq, Eq, "`i32` 1 if the operands are equal, else 0";
        ne, Ne, "`i32` 1 if the operands differ, else 0";
        lt_s, LtS, "`i32` 1 if the first is signed less than the second, else 0";
        lt_u, LtU, "`i32` 1 if the first is unsigned less than the second, else 0";
        gt_s, GtS, "`i32` 1 if the first is signed greater than the second, else 0";
        gt_u, GtU, "`i32` 1 if the first is unsigned greater than the second, else 0";
        le_s, LeS, "`i32` 1 if the first is signed less than or equal to the second, else 0";
        le_u, LeU, "`i32` 1 if the first is unsigned less than or equal to the second, else 0";
        ge_s, GeS, "`i32` 1 if the first is signed greater than or equal to the second, else 0";
        ge_u, GeU, "`i32` 1 if the first is unsigned greater than or equal to the second, else 0";
        lt, Lt, "`i32` 1 if the first float is less than the second, else 0";
        gt, Gt, "`i32` 1 if the first float is greater than the second, else 0";
        le, Le, "`i32` 1 if the first float is less than or equal to the second, else 0";
        ge, Ge, "`i32` 1 if the first float is greater than or equal to the second, else 0";
    }

    typed_conversion_builders! {
        trunc_s, TruncS, "signed integer of type `ty` from a float, truncating toward zero; traps on a NaN and on a value out of range";
        trunc_u, TruncU, "unsigned integer of type `ty` from a float, truncating toward zero; traps on a NaN and on a value out of range";
        trunc_sat_s, TruncSatS, "signed integer of type `ty` from a float, truncating toward zero and saturating; 0 for a NaN";
        trunc_sat_u, TruncSatU, "unsigned integer of type `ty` from a float, truncating toward zero and saturating; 0 for a NaN";
        convert_s, ConvertS, "float of type `ty` from a signed integer";
        convert_u, ConvertU, "float of type `ty` from an unsigned integer";
        reinterpret, Reinterpret, "same bits as the operand, as the number type `ty` of the same width";
    }

    fixed_conversion_builders! {
        wrap, Wrap, i32, "the low 32 bits of an `i64`, as an `i32`";
        extend_s, ExtendS, i64, "an `i32` sign-extended to an `i64`";
        extend_u, ExtendU, i64, "an `i32` zero-extended to an `i64`";
        promote, Promote, f64, "an `f32` as an `f64`";
        demote, Demote, f32, "an `f64` rounded to an `f32`";
    }

    /// Creates a [Memory] named `name` with `limits` in pages: a definition.
    ///
    /// Returns an error when the name is taken in the enclosing symbol table.
    fn create_memory(
        &mut self,
        name: Ident,
        visibility: Visibility,
        limits: Limits,
    ) -> Result<MemoryRef, Report> {
        ensure_definable::<Memory, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Memory, (_, _, _)>(name.span());
        op_builder(name, visibility, limits)
    }

    /// Creates a [Memory] declaration, an import: as [Self::create_memory], with the property
    /// `declaration`.
    fn create_memory_declaration(
        &mut self,
        name: Ident,
        visibility: Visibility,
        limits: Limits,
    ) -> Result<MemoryRef, Report> {
        ensure_definable::<Memory, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Memory, (_, _, _)>(name.span());
        let mut op = op_builder(name, visibility, limits)?;
        op.borrow_mut().set_declaration(true);
        Ok(op)
    }

    /// Creates a [Table] named `name` with `limits` in elements and the reference type
    /// `element_type`: a definition. Any other element type does not verify.
    ///
    /// Returns an error when the name is taken in the enclosing symbol table.
    fn create_table(
        &mut self,
        name: Ident,
        visibility: Visibility,
        limits: Limits,
        element_type: Type,
    ) -> Result<TableRef, Report> {
        ensure_definable::<Table, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Table, (_, _, _, _)>(name.span());
        op_builder(name, visibility, limits, element_type)
    }

    /// Creates a [Table] declaration, an import: as [Self::create_table], with the property
    /// `declaration`.
    fn create_table_declaration(
        &mut self,
        name: Ident,
        visibility: Visibility,
        limits: Limits,
        element_type: Type,
    ) -> Result<TableRef, Report> {
        ensure_definable::<Table, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Table, (_, _, _, _)>(name.span());
        let mut op = op_builder(name, visibility, limits, element_type)?;
        op.borrow_mut().set_declaration(true);
        Ok(op)
    }

    /// Creates a [Global] declaration, an import, of the type `ty`: its region is empty. A
    /// definition is made by [Self::create_global].
    fn create_global_declaration(
        &mut self,
        name: Ident,
        visibility: Visibility,
        ty: Type,
        mutability: Mutability,
    ) -> Result<GlobalRef, Report> {
        ensure_definable::<Global, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Global, (_, _, _)>(name.span());
        let mut global = op_builder(name, visibility, ty)?;
        global.borrow_mut().set_mutability(mutability);
        Ok(global)
    }

    /// Creates a [Global] of the type `ty` whose initializer `init` builds: a definition.
    ///
    /// The initializer is a region of one block; `init` is called with the builder positioned in
    /// it, builds the constant expression (`wasm.const` and the like), and returns the value the
    /// global has. The value is not checked here: an expression that is not constant, or has
    /// another type, does not verify. The insertion point is as it was before the call when it
    /// returns.
    ///
    /// Returns an error when the name is taken in the enclosing symbol table, or when `init` fails.
    fn create_global(
        &mut self,
        name: Ident,
        visibility: Visibility,
        ty: Type,
        mutability: Mutability,
        init: impl FnOnce(&mut B) -> Result<ValueRef, Report>,
    ) -> Result<GlobalRef, Report> {
        let global = self.create_global_declaration(name, visibility, ty, mutability)?;
        let region = global.borrow().initializer().as_region_ref();
        let mut guard = InsertionGuard::new(self.builder_mut());
        let block = guard.create_block(region, None, &[]);
        guard.set_insertion_point_to_end(block);
        init(&mut *guard).map(|_| global)
    }

    /// Creates an active [Elem] named `name` that puts `functions` into `table` from the element
    /// `offset`.
    ///
    /// Returns an error when the name is taken in the enclosing symbol table. The table and the
    /// functions are checked when the segment is verified.
    fn create_elem<C: AsCallableSymbolRef>(
        &mut self,
        name: Ident,
        visibility: Visibility,
        table: TableRef,
        offset: i32,
        functions: impl IntoIterator<Item = C>,
    ) -> Result<ElemRef, Report> {
        ensure_definable::<Elem, _>(self.builder(), name.name)?;
        let context = self.builder().context_rc();
        let i32 = context.i32();
        let offset = context.int_attr(Int::from(offset), i32);
        let mut list = AttrList::new();
        for function in functions {
            let path = function.as_callable_symbol_ref().borrow().path();
            let reference =
                context.create_attribute::<SymbolRefAttr, _>(SymbolRef::new(path, None));
            list.push_back(runic_ir::Attribute::as_attr(&*reference.borrow()).as_attr_ref());
        }
        let functions = List::from(list);
        let op_builder = self.builder_mut().create::<Elem, (_, _, _, _, _)>(name.span());
        let elem = op_builder(name, visibility, table, ReadyAttr(offset), functions)?;
        // The references of the list are linked once the segment is in its table
        runic_ir::link_symbol_refs(runic_ir::Op::as_operation_ref(&*elem.borrow()));
        Ok(elem)
    }

    /// Creates a [Tag] named `name` with the parameter types `params`: a definition.
    ///
    /// Returns an error when the name is taken in the enclosing symbol table.
    fn create_tag(
        &mut self,
        name: Ident,
        visibility: Visibility,
        params: &[Type],
    ) -> Result<TagRef, Report> {
        ensure_definable::<Tag, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Tag, (_, _, _)>(name.span());
        op_builder(name, visibility, Array::from(params.to_vec()))
    }

    /// Creates a [Tag] declaration, an import: as [Self::create_tag], with the property
    /// `declaration`.
    fn create_tag_declaration(
        &mut self,
        name: Ident,
        visibility: Visibility,
        params: &[Type],
    ) -> Result<TagRef, Report> {
        ensure_definable::<Tag, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Tag, (_, _, _)>(name.span());
        let mut op = op_builder(name, visibility, Array::from(params.to_vec()))?;
        op.borrow_mut().set_declaration(true);
        Ok(op)
    }

    /// Loads a value of the number type `ty` from `memory` at the `i32` address `addr` plus the
    /// offset of `memarg`, reading its whole type: `i32.load` and the others of their own type.
    /// The result is the value loaded.
    fn load(
        &mut self,
        memory: MemoryRef,
        addr: ValueRef,
        ty: Type,
        memarg: MemArg,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        self.load_narrow(memory, addr, ty, Storage::Full, Extension::Unextended, memarg, span)
    }

    /// Loads the integer `ty` from `memory`, reading only `storage` bytes and extending the value
    /// as `extension` says: `i32.load8_s`, `i64.load32_u` and the other narrow loads.
    #[allow(clippy::too_many_arguments)]
    fn load_narrow(
        &mut self,
        memory: MemoryRef,
        addr: ValueRef,
        ty: Type,
        storage: Storage,
        extension: Extension,
        memarg: MemArg,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Load, (_, _, _)>(span);
        let mut op = op_builder(memory, addr, memarg)?;
        {
            let mut op = op.borrow_mut();
            op.set_storage(storage);
            op.set_extension(extension);
            op.result_mut().set_type(ty);
        }
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// Stores `value`, a number, in `memory` at the `i32` address `addr` plus the offset of
    /// `memarg`, writing its whole type: `i32.store` and the others of their own type.
    fn store(
        &mut self,
        memory: MemoryRef,
        addr: ValueRef,
        value: ValueRef,
        memarg: MemArg,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        self.store_narrow(memory, addr, value, Storage::Full, memarg, span)
    }

    /// Stores the low `storage` bytes of the integer `value`: `i32.store8`, `i64.store32` and the
    /// other narrow stores.
    fn store_narrow(
        &mut self,
        memory: MemoryRef,
        addr: ValueRef,
        value: ValueRef,
        storage: Storage,
        memarg: MemArg,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let op_builder = self.builder_mut().create::<Store, (_, _, _, _)>(span);
        let mut op = op_builder(memory, addr, value, memarg)?;
        op.borrow_mut().set_storage(storage);
        let op = Op::as_operation_ref(&*op.borrow());
        Ok(op)
    }

    /// The size of `memory` in pages, an `i32`: `memory.size`.
    fn memory_size(&mut self, memory: MemoryRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<MemorySize, _>(span);
        let op = op_builder(memory)?;
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// Grows `memory` by the `i32` number of pages `delta`, giving the previous size or -1:
    /// `memory.grow`.
    fn memory_grow(
        &mut self,
        memory: MemoryRef,
        delta: ValueRef,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<MemoryGrow, _>(span);
        let op = op_builder(memory, delta)?;
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// Copies `len` bytes from the address `src_addr` of `src` to the address `dst_addr` of
    /// `dst`: `memory.copy`.
    fn memory_copy(
        &mut self,
        dst: MemoryRef,
        src: MemoryRef,
        dst_addr: ValueRef,
        src_addr: ValueRef,
        len: ValueRef,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let op_builder = self.builder_mut().create::<MemoryCopy, _>(span);
        let op = op_builder(dst, src, dst_addr, src_addr, len)?;
        let op = Op::as_operation_ref(&*op.borrow());
        Ok(op)
    }

    /// Sets `len` bytes from the address `dst_addr` of `memory` to the low byte of `value`:
    /// `memory.fill`.
    fn memory_fill(
        &mut self,
        memory: MemoryRef,
        dst_addr: ValueRef,
        value: ValueRef,
        len: ValueRef,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let op_builder = self.builder_mut().create::<MemoryFill, _>(span);
        let op = op_builder(memory, dst_addr, value, len)?;
        let op = Op::as_operation_ref(&*op.borrow());
        Ok(op)
    }

    /// Copies `len` bytes from the offset `src_offset` of the data `data` to the address
    /// `dst_addr` of `memory`: `memory.init`.
    fn memory_init(
        &mut self,
        memory: MemoryRef,
        data: GlobalVariableRef,
        dst_addr: ValueRef,
        src_offset: ValueRef,
        len: ValueRef,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let op_builder = self.builder_mut().create::<MemoryInit, _>(span);
        let op = op_builder(memory, data, dst_addr, src_offset, len)?;
        let op = Op::as_operation_ref(&*op.borrow());
        Ok(op)
    }

    /// Drops the data `data`: `data.drop`.
    fn data_drop(
        &mut self,
        data: GlobalVariableRef,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let op_builder = self.builder_mut().create::<DataDrop, _>(span);
        let op = op_builder(data)?;
        let op = Op::as_operation_ref(&*op.borrow());
        Ok(op)
    }

    /// The value of `global`, with the type the global has: `global.get`.
    fn global_get(&mut self, global: GlobalRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let ty = *global.borrow().get_ty();
        let op_builder = self.builder_mut().create::<GlobalGet, _>(span);
        let mut op = op_builder(global)?;
        op.borrow_mut().result_mut().set_type(ty);
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// Writes `value` to the mutable `global`: `global.set`.
    fn global_set(
        &mut self,
        global: GlobalRef,
        value: ValueRef,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let op_builder = self.builder_mut().create::<GlobalSet, _>(span);
        let op = op_builder(global, value)?;
        let op = Op::as_operation_ref(&*op.borrow());
        Ok(op)
    }

    /// The element `index`, an `i32`, of `table`, with the element type of the table:
    /// `table.get`.
    fn table_get(
        &mut self,
        table: TableRef,
        index: ValueRef,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let ty = *table.borrow().get_element_type();
        let op_builder = self.builder_mut().create::<TableGet, _>(span);
        let mut op = op_builder(table, index)?;
        op.borrow_mut().result_mut().set_type(ty);
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// Sets the element `index` of `table` to `value`: `table.set`.
    fn table_set(
        &mut self,
        table: TableRef,
        index: ValueRef,
        value: ValueRef,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let op_builder = self.builder_mut().create::<TableSet, _>(span);
        let op = op_builder(table, index, value)?;
        let op = Op::as_operation_ref(&*op.borrow());
        Ok(op)
    }

    /// The size of `table` in elements, an `i32`: `table.size`.
    fn table_size(&mut self, table: TableRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<TableSize, _>(span);
        let op = op_builder(table)?;
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// Grows `table` by `delta` elements, each `init`, giving the previous size or -1:
    /// `table.grow`.
    fn table_grow(
        &mut self,
        table: TableRef,
        init: ValueRef,
        delta: ValueRef,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<TableGrow, _>(span);
        let op = op_builder(table, init, delta)?;
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// Sets `len` elements of `table` from `index` to `value`: `table.fill`.
    fn table_fill(
        &mut self,
        table: TableRef,
        index: ValueRef,
        value: ValueRef,
        len: ValueRef,
        span: SourceSpan,
    ) -> Result<OperationRef, Report> {
        let op_builder = self.builder_mut().create::<TableFill, _>(span);
        let op = op_builder(table, index, value, len)?;
        let op = Op::as_operation_ref(&*op.borrow());
        Ok(op)
    }

    /// The null reference of the reference type `ty`: `ref.null`.
    fn ref_null(&mut self, ty: Type, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<RefNull, _>(span);
        let mut op = op_builder()?;
        op.borrow_mut().result_mut().set_type(ty);
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// A `funcref` to `callee`: `ref.func`.
    fn ref_func<C: AsCallableSymbolRef>(
        &mut self,
        callee: C,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<RefFunc, (C,)>(span);
        let op = op_builder(callee)?;
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// 1 if the reference `reference` is null, an `i32`: `ref.is_null`.
    fn ref_is_null(&mut self, reference: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<RefIsNull, _>(span);
        let op = op_builder(reference)?;
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// The address of `data`, plus `offset` bytes, as an `i32` that a layout or a linker
    /// resolves (see [DataAddr]).
    fn data_addr(
        &mut self,
        data: GlobalVariableRef,
        offset: i32,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let context = self.builder().context_rc();
        let i32 = context.i32();
        let offset = context.int_attr(Int::from(offset), i32);
        let op_builder = self.builder_mut().create::<DataAddr, (_, _)>(span);
        let op = op_builder(data, ReadyAttr(offset))?;
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// The index of `callee` in the indirect-call table, an `i32` that a layout or a linker
    /// resolves (see [FuncIndex]).
    fn func_index<C: AsCallableSymbolRef>(
        &mut self,
        callee: C,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<FuncIndex, (C,)>(span);
        let op = op_builder(callee)?;
        let result = op.borrow().result().as_value_ref();
        Ok(result)
    }

    /// Builds a `wasm.block` with the parameters `inputs` and results of the types `results`. The
    /// body has one block, with an argument for each parameter, which the caller fills and ends
    /// with a `wasm.end` or an early exit.
    fn block<T: IntoIterator<Item = ValueRef>>(
        &mut self,
        inputs: T,
        results: &[Type],
        span: SourceSpan,
    ) -> Result<BlockOpRef, Report> {
        let op_builder = self.builder_mut().create::<BlockOp, (T,)>(span);
        let mut block_op = op_builder(inputs)?;
        add_results(self.builder().context(), block_op.as_operation_ref(), results, span);
        let types = operand_types(block_op.as_operation_ref());
        let entry = self.builder().context_rc().create_block_with_params(types);
        block_op.borrow_mut().body_mut().body_mut().push_back(entry);
        Ok(block_op)
    }

    /// Builds a `wasm.loop` with the parameters `inputs` and results of the types `results`. The
    /// body has one block, with an argument for each parameter, which the caller fills and ends
    /// with a `wasm.end` (which leaves the loop) or an early exit (a `wasm.br` that names the loop
    /// goes back to the top).
    fn r#loop<T: IntoIterator<Item = ValueRef>>(
        &mut self,
        inputs: T,
        results: &[Type],
        span: SourceSpan,
    ) -> Result<LoopRef, Report> {
        let op_builder = self.builder_mut().create::<Loop, (T,)>(span);
        let mut loop_op = op_builder(inputs)?;
        add_results(self.builder().context(), loop_op.as_operation_ref(), results, span);
        let types = operand_types(loop_op.as_operation_ref());
        let entry = self.builder().context_rc().create_block_with_params(types);
        loop_op.borrow_mut().body_mut().body_mut().push_back(entry);
        Ok(loop_op)
    }

    /// Builds a `wasm.if` on the `i32` `condition` with the parameters `inputs` and results of the
    /// types `results`. Each of the two regions has one block, with an argument for each
    /// parameter, which the caller fills and ends with a `wasm.end` or an early exit.
    fn r#if<T: IntoIterator<Item = ValueRef>>(
        &mut self,
        condition: ValueRef,
        inputs: T,
        results: &[Type],
        span: SourceSpan,
    ) -> Result<IfRef, Report> {
        let op_builder = self.builder_mut().create::<If, (_, T)>(span);
        let mut if_op = op_builder(condition, inputs)?;
        add_results(self.builder().context(), if_op.as_operation_ref(), results, span);
        let mut types = operand_types(if_op.as_operation_ref());
        types.remove(0);
        let then_entry = self.builder().context_rc().create_block_with_params(types.clone());
        let else_entry = self.builder().context_rc().create_block_with_params(types);
        let mut op = if_op.borrow_mut();
        op.then_body_mut().body_mut().push_back(then_entry);
        op.else_body_mut().body_mut().push_back(else_entry);
        drop(op);
        Ok(if_op)
    }

    /// Builds a `wasm.end` of `values`: falling off the end of a block, loop or if.
    fn end<T: IntoIterator<Item = ValueRef>>(
        &mut self,
        values: T,
        span: SourceSpan,
    ) -> Result<EndRef, Report> {
        let op_builder = self.builder_mut().create::<End, (T,)>(span);
        op_builder(values)
    }

    /// Builds a `wasm.br` to label `level` (0 is the innermost enclosing block, loop or if)
    /// carrying `values`. The label is resolved when the operation is verified, with its
    /// enclosing operations in place.
    fn br<T: IntoIterator<Item = ValueRef>>(
        &mut self,
        level: i32,
        values: T,
        span: SourceSpan,
    ) -> Result<BrRef, Report> {
        let op_builder = self.builder_mut().create::<Br, (_, T)>(span);
        op_builder(level, values)
    }

    /// Builds a `wasm.br_table` on the `i32` `index`, carrying `values`, with `entries` empty
    /// entry regions after the (empty) default region. The caller gives each region a block that
    /// holds one `wasm.br` of `values`.
    fn br_table<T: IntoIterator<Item = ValueRef>>(
        &mut self,
        index: ValueRef,
        values: T,
        entries: usize,
        span: SourceSpan,
    ) -> Result<BrTableRef, Report> {
        let op_builder = self.builder_mut().create::<BrTable, (_, T)>(span);
        let table = op_builder(index, values)?;
        let mut owner = table.as_operation_ref();
        for _ in 0..entries {
            let region = self.builder().context().create_region();
            owner.borrow_mut().regions_mut().push_back(region);
        }
        Ok(table)
    }

    /// Builds a `wasm.unreachable`, which traps. (The name is not `unreachable`, which the builder
    /// of `ub`, whose operation does not trap, has.)
    fn trap(&mut self, span: SourceSpan) -> Result<UnreachableRef, Report> {
        let op_builder = self.builder_mut().create::<Unreachable, _>(span);
        op_builder()
    }

    /// Calls the function in entry `index`, an `i32`, of `table` with `arguments` and the
    /// signature `signature`: `call_indirect`. The call has one result per result of the
    /// signature. (The name is not `call_indirect`, which `builtin`'s builder trait has.)
    fn call_through_table<A: IntoIterator<Item = ValueRef>>(
        &mut self,
        table: TableRef,
        index: ValueRef,
        signature: Signature,
        arguments: A,
        span: SourceSpan,
    ) -> Result<CallIndirectRef, Report> {
        let op_builder = self.builder_mut().create::<CallIndirect, (_, _, A, _)>(span);
        op_builder(table, index, arguments, signature)
    }

    /// Tail-calls the function in entry `index` of `table` with `arguments` and the signature
    /// `signature`: `return_call_indirect`. (The name is not `return_call_indirect`, to go with
    /// [Self::call_through_table].)
    fn return_call_through_table<A: IntoIterator<Item = ValueRef>>(
        &mut self,
        table: TableRef,
        index: ValueRef,
        signature: Signature,
        arguments: A,
        span: SourceSpan,
    ) -> Result<ReturnCallIndirectRef, Report> {
        let op_builder = self.builder_mut().create::<ReturnCallIndirect, (_, _, A, _)>(span);
        op_builder(table, index, arguments, signature)
    }

    /// The builder this builder wraps.
    fn builder(&self) -> &B;

    /// The builder this builder wraps, mutably.
    fn builder_mut(&mut self) -> &mut B;
}

impl<'f, B: ?Sized + Builder> WasmOpBuilder<'f, B> for FunctionBuilder<'f, B> {
    #[inline(always)]
    fn builder(&self) -> &B {
        FunctionBuilder::builder(self)
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        FunctionBuilder::builder_mut(self)
    }
}

impl<'f> WasmOpBuilder<'f, OpBuilder> for &'f mut OpBuilder {
    #[inline(always)]
    fn builder(&self) -> &OpBuilder {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut OpBuilder {
        self
    }
}

impl<B: ?Sized + Builder> WasmOpBuilder<'_, B> for B {
    #[inline(always)]
    fn builder(&self) -> &B {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        self
    }
}
