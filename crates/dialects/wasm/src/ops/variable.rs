//! The operations on globals and tables, the reference operations, and the two address operations.
//!
//! A global and a table are named by symbol (`@g`, `@t`), as a memory is. Their effects are on
//! [GlobalsResource] and [TablesResource]; the reference operations and the address operations are
//! free of effects.

use alloc::format;

use runic_ir::{
    Context, Int, Op, OpAsmParser, OpParser, OpPrinter, Operation, OperationState, ParseResult,
    Report, Spanned, Value, Verify,
    derive::{EffectOpInterface, operation},
    dialects::builtin::{
        GlobalVariable,
        attributes::{IntAttr, SymbolRefAttr},
    },
    effects::{MemoryEffect, MemoryEffectOpInterface},
    formatter::const_text,
    parse::{ParserError, ParserExt},
    print::AsmPrinter,
    traits::{AnyType, InferTypeOpInterface, TypeConstraint},
};

use super::text::*;
use crate::{
    ConstantInstruction, FuncRefType, Global, GlobalRef, GlobalsResource, RefType, Table,
    TablesResource, WasmDialect, is_ref_type, referenced_entity, val_type_name,
};

/// The operation named by the global symbol of `op`, if it is a [Global].
fn global_of(reference: &SymbolRefAttr) -> Option<GlobalRef> {
    let symbol = reference.resolve()?;
    let op = symbol.borrow().as_symbol_operation().as_operation_ref();
    op.try_downcast_op::<Global>().ok()
}

/// Reads an optional `attributes { .. }` and records the type `ty` as the one result.
fn finish_with_result(
    state: &mut OperationState,
    parser: &mut dyn OpAsmParser<'_>,
    ty: runic_ir::Type,
) -> ParseResult {
    parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
    state.results.push(ty);
    Ok(())
}

/// Reads the optional `attributes { .. }` that ends an operation with no result.
fn finish(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
    parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Globals

/// Reading a global: `%v = wasm.global_get @g : i32;`. `global.get`.
///
/// The result has the type of the global, which the text states. The operation reads
/// [GlobalsResource], so two reads of one global merge only when no write of a global, and no
/// call that may write one, lies between them. It may appear in a Wasm constant expression when
/// the global is immutable ([ConstantInstruction]; ask [is_constant_instruction](crate::is_constant_instruction),
/// which looks at the global).
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    traits(ConstantInstruction),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
#[effects(MemoryEffect(MemoryEffect::Read), resource(GlobalsResource))]
pub struct GlobalGet {
    /// The global read
    #[symbol]
    global: Global,
    /// The value of the global
    #[result]
    result: AnyType,
}

impl GlobalGet {
    /// The global this operation reads, or `None` if its reference is not linked to a [Global].
    pub fn global_op(&self) -> Option<GlobalRef> {
        global_of(&self.global())
    }
}

const GLOBAL_GET_RULE: &str = "takes the name of a global and gives a value of its type";

impl OpPrinter for GlobalGet {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.global()]);
        printer.print_space();
        printer.print_colon_type(self.result().ty());
        print_attributes(printer, op);
    }
}

impl OpParser for GlobalGet {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["global"], 0)?;
        let ty = parser.parse_colon_type()?.into_inner();
        finish_with_result(state, parser, ty)
    }
}

/// The name is a `wasm.global` and the result has its type.
impl Verify<dyn MemoryEffectOpInterface> for GlobalGet {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let global = referenced_entity::<Global>(op, &self.global(), "wasm.global")?;
        let ty = *global.borrow().get_ty();
        let found = result_type(op);
        if found != ty {
            return Err(rule_error(
                op,
                GLOBAL_GET_RULE,
                format!(
                    "its result has type `{}` and the global `{}` has type `{}`",
                    val_type_name(found),
                    self.global().path(),
                    val_type_name(ty)
                ),
            ));
        }
        Ok(())
    }
}

/// Writing a global: `wasm.global_set @g, %v;`. `global.set`.
///
/// The global is mutable (`mut`) and `%v` has its type. The operation writes [GlobalsResource], so
/// it is never removed, and no read of a global is merged or moved across it.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
#[effects(MemoryEffect(MemoryEffect::Write), resource(GlobalsResource))]
pub struct GlobalSet {
    /// The global written
    #[symbol]
    global: Global,
    /// The value written
    #[operand]
    value: AnyType,
}

const GLOBAL_SET_RULE: &str = "takes the name of a mutable global and a value of its type";

impl OpPrinter for GlobalSet {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.global()]);
        print_attributes(printer, op);
    }
}

impl OpParser for GlobalSet {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["global"], 1)?;
        finish(state, parser)
    }
}

/// The name is a `wasm.global`, the global is mutable, and the value has its type.
impl Verify<dyn MemoryEffectOpInterface> for GlobalSet {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let global = referenced_entity::<Global>(op, &self.global(), "wasm.global")?;
        let (ty, mutable) = {
            let global = global.borrow();
            (*global.get_ty(), global.get_mutability().is_mut())
        };
        if !mutable {
            return Err(rule_error(
                op,
                GLOBAL_SET_RULE,
                format!(
                    "the global `{}` is immutable, and `global.set` writes only a `mut` global",
                    self.global().path()
                ),
            ));
        }
        let found = operand_type(op, 0);
        if found != ty {
            return Err(rule_error(
                op,
                GLOBAL_SET_RULE,
                format!(
                    "its value has type `{}` and the global `{}` has type `{}`",
                    val_type_name(found),
                    self.global().path(),
                    val_type_name(ty)
                ),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Tables

/// The element type of the table `reference` names, which must be a `wasm.table`.
fn element_type(op: &Operation, reference: &SymbolRefAttr) -> Result<runic_ir::Type, Report> {
    let table = referenced_entity::<Table>(op, reference, "wasm.table")?;
    Ok(*table.borrow().get_element_type())
}

/// Checks that operand `index` of `op` has the element type `element` of the table `reference`
/// names.
fn expect_element(
    op: &Operation,
    rule: &str,
    reference: &SymbolRefAttr,
    index: usize,
    what: &str,
    element: runic_ir::Type,
) -> Result<(), Report> {
    let found = operand_type(op, index);
    if found == element {
        return Ok(());
    }
    Err(rule_error(
        op,
        rule,
        format!(
            "{what} has type `{}` and the table `{}` holds `{}`",
            val_type_name(found),
            reference.path(),
            val_type_name(element)
        ),
    ))
}

/// Reading a table: `%r = wasm.table_get @t, %i : !wasm.funcref;`. `table.get`.
///
/// The index is an `i32`; the result has the element type of the table, which the text states. It
/// traps when the index is out of bounds. It reads [TablesResource].
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
#[effects(MemoryEffect(MemoryEffect::Read), resource(TablesResource))]
pub struct TableGet {
    /// The table read
    #[symbol]
    table: Table,
    /// The index, an `i32`
    #[operand]
    index: AnyType,
    /// The element
    #[result]
    result: AnyType,
}

const TABLE_GET_RULE: &str =
    "takes a table and an `i32` index and gives an element, a reference of the table's type";

impl OpPrinter for TableGet {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.table()]);
        printer.print_space();
        printer.print_colon_type(self.result().ty());
        print_attributes(printer, op);
    }
}

impl OpParser for TableGet {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["table"], 1)?;
        let ty = parser.parse_colon_type()?.into_inner();
        finish_with_result(state, parser, ty)
    }
}

/// The name is a `wasm.table`, the index is an `i32`, and the result has the element type.
impl Verify<dyn MemoryEffectOpInterface> for TableGet {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let element = element_type(op, &self.table())?;
        expect_operand(op, TABLE_GET_RULE, 0, "its index", op.context().i32())?;
        let found = result_type(op);
        if found != element {
            return Err(rule_error(
                op,
                TABLE_GET_RULE,
                format!(
                    "its result has type `{}` and the table `{}` holds `{}`",
                    val_type_name(found),
                    self.table().path(),
                    val_type_name(element)
                ),
            ));
        }
        Ok(())
    }
}

/// Writing a table: `wasm.table_set @t, %i, %r;`. `table.set`.
///
/// The index is an `i32` and `%r` has the element type of the table. It traps when the index is
/// out of bounds. It writes [TablesResource], so it is never removed.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
#[effects(MemoryEffect(MemoryEffect::Write), resource(TablesResource))]
pub struct TableSet {
    /// The table written
    #[symbol]
    table: Table,
    /// The index, an `i32`
    #[operand]
    index: AnyType,
    /// The reference written
    #[operand]
    value: AnyType,
}

const TABLE_SET_RULE: &str =
    "takes a table, an `i32` index and a reference of the table's element type";

impl OpPrinter for TableSet {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.table()]);
        print_attributes(printer, op);
    }
}

impl OpParser for TableSet {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["table"], 2)?;
        finish(state, parser)
    }
}

/// The name is a `wasm.table`, the index is an `i32`, and the value has the element type.
impl Verify<dyn MemoryEffectOpInterface> for TableSet {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let element = element_type(op, &self.table())?;
        expect_operand(op, TABLE_SET_RULE, 0, "its index", op.context().i32())?;
        expect_element(op, TABLE_SET_RULE, &self.table(), 1, "its value", element)
    }
}

/// The size of a table: `%n = wasm.table_size @t;`. `table.size`.
///
/// The result is an `i32`, the number of elements. It reads [TablesResource], since
/// `wasm.table_grow` changes it.
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
#[effects(MemoryEffect(MemoryEffect::Read), resource(TablesResource))]
pub struct TableSize {
    /// The table measured
    #[symbol]
    table: Table,
    /// The number of elements
    #[result]
    result: AnyType,
}

const TABLE_SIZE_RULE: &str = "takes a table and gives its size in elements, an `i32`";

impl InferTypeOpInterface for TableSize {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

impl OpPrinter for TableSize {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.table()]);
        print_attributes(printer, op);
    }
}

impl OpParser for TableSize {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["table"], 0)?;
        let i32 = parser.context().i32();
        finish_with_result(state, parser, i32)
    }
}

/// The name is a `wasm.table` and the result is an `i32`.
impl Verify<dyn MemoryEffectOpInterface> for TableSize {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        element_type(op, &self.table())?;
        expect_result(op, TABLE_SIZE_RULE, op.context().i32())
    }
}

/// Growing a table: `%old = wasm.table_grow @t, %init, %delta;`. `table.grow`.
///
/// It adds `%delta` elements (an `i32`), each `%init` (a reference of the element type), and gives
/// the previous size, or -1 (all ones) when the table cannot grow that far. It writes
/// [TablesResource], so it is never removed.
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
#[effects(MemoryEffect(MemoryEffect::Write), resource(TablesResource))]
pub struct TableGrow {
    /// The table grown
    #[symbol]
    table: Table,
    /// The value of each new element
    #[operand]
    init: AnyType,
    /// The number of elements added, an `i32`
    #[operand]
    delta: AnyType,
    /// The previous size, or -1
    #[result]
    result: AnyType,
}

const TABLE_GROW_RULE: &str = "takes a table, a reference of the table's element type and an \
                               `i32` number of elements, and gives an `i32`";

impl InferTypeOpInterface for TableGrow {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

impl OpPrinter for TableGrow {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.table()]);
        print_attributes(printer, op);
    }
}

impl OpParser for TableGrow {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["table"], 2)?;
        let i32 = parser.context().i32();
        finish_with_result(state, parser, i32)
    }
}

/// The name is a `wasm.table`, the initial value has the element type, and the delta and the
/// result are `i32`s.
impl Verify<dyn MemoryEffectOpInterface> for TableGrow {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let element = element_type(op, &self.table())?;
        expect_element(op, TABLE_GROW_RULE, &self.table(), 0, "its initial value", element)?;
        expect_operand(op, TABLE_GROW_RULE, 1, "its delta", op.context().i32())?;
        expect_result(op, TABLE_GROW_RULE, op.context().i32())
    }
}

/// Filling a range of a table: `wasm.table_fill @t, %i, %r, %n;`. `table.fill`.
///
/// It sets `%n` elements (an `i32`) from the index `%i` (an `i32`) to `%r`, a reference of the
/// element type. It traps when the range is out of bounds. It writes [TablesResource], so it is
/// never removed.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
#[effects(MemoryEffect(MemoryEffect::Write), resource(TablesResource))]
pub struct TableFill {
    /// The table written
    #[symbol]
    table: Table,
    /// The first index, an `i32`
    #[operand]
    index: AnyType,
    /// The reference written
    #[operand]
    value: AnyType,
    /// The number of elements, an `i32`
    #[operand]
    len: AnyType,
}

const TABLE_FILL_RULE: &str = "takes a table, an `i32` index, a reference of the table's element \
                               type and an `i32` number of elements";

impl OpPrinter for TableFill {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.table()]);
        print_attributes(printer, op);
    }
}

impl OpParser for TableFill {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["table"], 3)?;
        finish(state, parser)
    }
}

/// The name is a `wasm.table`, the index and the length are `i32`s, and the value has the element
/// type.
impl Verify<dyn MemoryEffectOpInterface> for TableFill {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let element = element_type(op, &self.table())?;
        let i32 = op.context().i32();
        expect_operand(op, TABLE_FILL_RULE, 0, "its index", i32)?;
        expect_element(op, TABLE_FILL_RULE, &self.table(), 1, "its value", element)?;
        expect_operand(op, TABLE_FILL_RULE, 2, "its length", i32)
    }
}

// ---------------------------------------------------------------------------------------------
// References

/// A null reference: `%r = wasm.ref_null : !wasm.funcref;`. `ref.null`.
///
/// The result type, `!wasm.funcref` or `!wasm.externref`, is the one the text states. It is free of
/// effects and may appear in a Wasm constant expression ([ConstantInstruction]).
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    traits(ConstantInstruction),
    implements(MemoryEffectOpInterface, OpPrinter)
)]
pub struct RefNull {
    /// The null reference
    #[result]
    result: AnyType,
}

const REF_NULL_RULE: &str = "gives a null reference of a reference type, `funcref` or `externref`";

impl OpPrinter for RefNull {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        printer.print_space();
        printer.print_colon_type(self.result().ty());
        print_attributes(printer, op);
    }
}

impl OpParser for RefNull {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let ty = parser.parse_colon_type()?.into_inner();
        finish_with_result(state, parser, ty)
    }
}

/// The result is a reference type.
impl Verify<dyn MemoryEffectOpInterface> for RefNull {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let found = result_type(op);
        if !is_ref_type(found) {
            return Err(rule_error(
                op,
                REF_NULL_RULE,
                format!(
                    "its result has type `{}`, not {}",
                    val_type_name(found),
                    RefType::get().description()
                ),
            ));
        }
        Ok(())
    }
}

/// A reference to a function: `%r = wasm.ref_func @f;`. `ref.func`.
///
/// `@f` is a callable (a function, or an alias of one); the result is a `!wasm.funcref`. It is
/// free of effects and may appear in a Wasm constant expression ([ConstantInstruction]).
///
/// Wasm validates `ref.func` in a function body only for a function that the module *declares*:
/// one named by an element segment, by an export, or by the initializer of a global. The dialect
/// accepts `wasm.ref_func` of any function, as it is an IR and not a binary, and does not check
/// that rule. An emitter must satisfy it, by writing a declarative element segment
/// (`(elem declare func $f)`) for each function that a `ref_func` in a function body names and
/// that nothing else declares.
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    traits(ConstantInstruction),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct RefFunc {
    /// The function referred to
    #[symbol(callable)]
    callee: runic_ir::SymbolPath,
    /// The reference, a `funcref`
    #[result]
    result: AnyType,
}

const REF_FUNC_RULE: &str = "takes the name of a function and gives a `funcref`";

impl InferTypeOpInterface for RefFunc {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(FuncRefType::get(context));
        Ok(())
    }
}

impl OpPrinter for RefFunc {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.callee()]);
        print_attributes(printer, op);
    }
}

impl OpParser for RefFunc {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["callee"], 0)?;
        let ty = FuncRefType::get(parser.context());
        finish_with_result(state, parser, ty)
    }
}

/// Checks that `reference`, held by `op`, names a callable.
fn verify_callable(op: &Operation, rule: &str, reference: &SymbolRefAttr) -> Result<(), Report> {
    match reference.resolve_callable() {
        Ok(_) => Ok(()),
        Err(error) => Err(rule_error(
            op,
            rule,
            format!("`{}` does not name a function: {error}", reference.path()),
        )),
    }
}

/// The name is a callable and the result is a `funcref`.
impl Verify<dyn MemoryEffectOpInterface> for RefFunc {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        verify_callable(op, REF_FUNC_RULE, &self.callee())?;
        expect_result(op, REF_FUNC_RULE, FuncRefType::get(op.context()))
    }
}

/// Testing a reference for null: `%b = wasm.ref_is_null %r;`. `ref.is_null`.
///
/// The operand is a `!wasm.funcref` or a `!wasm.externref` and the result is an `i32`: 1 when the
/// reference is null, 0 otherwise. It is free of effects.
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct RefIsNull {
    /// The reference tested
    #[operand]
    operand: AnyType,
    /// 1 if it is null
    #[result]
    result: AnyType,
}

const REF_IS_NULL_RULE: &str = "takes a reference, `funcref` or `externref`, and gives an `i32`";

impl InferTypeOpInterface for RefIsNull {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

impl OpPrinter for RefIsNull {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[]);
        print_attributes(printer, op);
    }
}

impl OpParser for RefIsNull {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &[], 1)?;
        let i32 = parser.context().i32();
        finish_with_result(state, parser, i32)
    }
}

/// The operand is a reference and the result is an `i32`.
impl Verify<dyn MemoryEffectOpInterface> for RefIsNull {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        let found = operand_type(op, 0);
        if !is_ref_type(found) {
            return Err(rule_error(
                op,
                REF_IS_NULL_RULE,
                format!("its operand has type `{}`", val_type_name(found)),
            ));
        }
        expect_result(op, REF_IS_NULL_RULE, op.context().i32())
    }
}

// ---------------------------------------------------------------------------------------------
// Addresses

/// The address of data in linear memory: `%a = wasm.data_addr @data + 16;`.
///
/// This is **not a Wasm instruction**. It stands for an `i32.const` whose value only a layout or a
/// linker knows: the address of the byte `offset` bytes into the `builtin.global_variable`
/// `@data` in linear memory. The backend that lays the data out (a backend with its own layout)
/// resolves it, or, in an object file, the linker does through a relocation. The offset is a
/// signed constant, written `+ N` or `- N`, and left out when 0. The result is an `i32`. The
/// operation is free of effects and may appear in a Wasm constant expression
/// ([ConstantInstruction]), as the `i32.const` it stands for does.
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    traits(ConstantInstruction),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct DataAddr {
    /// The data whose address is taken
    #[symbol]
    data: GlobalVariable,
    /// The signed offset in bytes, an `i32` constant
    #[attr]
    offset: IntAttr,
    /// The address, an `i32`
    #[result]
    result: AnyType,
}

const DATA_ADDR_RULE: &str = "takes the name of data (`builtin.global_variable`) and a constant \
                              offset and gives an address, an `i32`";

impl DataAddr {
    /// The offset in bytes, as a signed number.
    pub fn offset_value(&self) -> i64 {
        (*self.get_offset()).try_signed::<i64>().unwrap_or(0)
    }
}

impl InferTypeOpInterface for DataAddr {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

impl OpPrinter for DataAddr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.data()]);
        let offset = self.offset_value();
        if offset != 0 {
            *printer += const_text(if offset < 0 { " - " } else { " + " });
            printer.print_decimal_integer(offset.unsigned_abs());
        }
        print_attributes(printer, op);
    }
}

impl OpParser for DataAddr {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["data"], 0)?;
        let negative = if parser.parse_optional_plus()? {
            Some(false)
        } else if parser.parse_optional_minus()? {
            Some(true)
        } else {
            None
        };
        let mut offset = 0i64;
        if let Some(negative) = negative {
            let magnitude = parser.parse_decimal_integer::<u64>()?;
            let span = magnitude.span();
            let magnitude = magnitude.into_inner();
            let signed = if negative {
                -(magnitude as i128)
            } else {
                magnitude as i128
            };
            offset = i32::try_from(signed).map_err(|_| ParserError::InvalidIntegerLiteral {
                span,
                reason: format!("{signed} is not a 32-bit signed offset"),
            })? as i64;
        }
        let i32 = parser.context().i32();
        let attr = parser.context_rc().int_attr(Int::from(offset as i32), i32);
        state.add_attribute("offset", attr);
        finish_with_result(state, parser, i32)
    }
}

/// The name is a `builtin.global_variable`, the offset is an `i32` and the result is an `i32`.
impl Verify<dyn MemoryEffectOpInterface> for DataAddr {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<GlobalVariable>(op, &self.data(), "builtin.global_variable")?;
        let i32 = op.context().i32();
        let offset = runic_ir::Attribute::ty(&*self.offset());
        if offset != i32 {
            return Err(rule_error(
                op,
                DATA_ADDR_RULE,
                format!("its offset has type `{}`", val_type_name(offset)),
            ));
        }
        expect_result(op, DATA_ADDR_RULE, i32)
    }
}

/// The index of a function in the indirect-call table: `%i = wasm.func_index @f;`.
///
/// This is **not a Wasm instruction**. It stands for an `i32.const` whose value only a layout or
/// a linker knows: the position of the function `@f` in the module's indirect-function table.
/// The operation names no table, and the dialect assumes none: which `funcref` table that is (a
/// module may have several, and `wasm.call_indirect` names the one it calls through) and where in
/// it `@f` sits are decided by the layout of the backend that builds the table, or, in an object
/// file, by the linker through a relocation. `@f` is a callable and the result
/// is an `i32`. The operation is free of effects and may appear in a Wasm constant expression
/// ([ConstantInstruction]), as the `i32.const` it stands for does.
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    traits(ConstantInstruction),
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct FuncIndex {
    /// The function whose index is taken
    #[symbol(callable)]
    callee: runic_ir::SymbolPath,
    /// The index, an `i32`
    #[result]
    result: AnyType,
}

const FUNC_INDEX_RULE: &str =
    "takes the name of a function and gives its index in the indirect-call table, an `i32`";

impl InferTypeOpInterface for FuncIndex {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

impl OpPrinter for FuncIndex {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.callee()]);
        print_attributes(printer, op);
    }
}

impl OpParser for FuncIndex {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["callee"], 0)?;
        let i32 = parser.context().i32();
        finish_with_result(state, parser, i32)
    }
}

/// The name is a callable and the result is an `i32`.
impl Verify<dyn MemoryEffectOpInterface> for FuncIndex {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        verify_callable(op, FUNC_INDEX_RULE, &self.callee())?;
        expect_result(op, FUNC_INDEX_RULE, op.context().i32())
    }
}
