//! The operations on a named linear memory: loads, stores, and the bulk and size operations.
//!
//! Every one of them names the [Memory] it works on (`@mem`). The memory is 32-bit: an address is
//! an `i32`, and an access traps when it is out of bounds (a trap is not an effect). Their effects
//! are on [LinearMemoryResource], which the effect system orders against the `mem` dialect's own
//! memory.

use alloc::format;
use core::{fmt, str::FromStr};

use runic_ir::{
    AttrPrinter, Context, Op, OpAsmParser, OpParser, OpPrinter, OperationState, ParseResult,
    Report, Spanned, Type, Value, Verify,
    attributes::AttrParser,
    derive::{DialectAttribute, EffectOpInterface, operation},
    dialects::builtin::GlobalVariable,
    effects::{MemoryEffect, MemoryEffectOpInterface},
    formatter::const_text,
    parse::{ParserError, ParserExt, Token},
    print::AsmPrinter,
    traits::{AnyType, InferTypeOpInterface},
};

use super::text::*;
use crate::{
    LinearMemoryResource, MemArg, MemArgAttr, Memory, WasmDialect, is_int_type, is_num_type,
    referenced_entity, val_type_name,
};

/// How many bytes of its value a load reads or a store writes: the whole value, or the low 8, 16
/// or 32 bits of an integer.
///
/// `Full` is the access of `i32.load`, `f64.store` and the like, `I8`, `I16` and `I32` are the
/// storage of `i32.load8_s`, `i64.load32_u`, `i64.store16` and the other narrow forms (see
/// [Load] and [Store] for which are valid). Text: `#wasm.storage<full>`, `#wasm.storage<i8>`,
/// `#wasm.storage<i16>`, `#wasm.storage<i32>`; a body-less `#wasm.storage` is `full`.
#[derive(DialectAttribute, Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
#[attribute(dialect = WasmDialect, name = "storage", implements(AttrPrinter))]
pub enum Storage {
    /// The whole value: a load of its own type.
    #[default]
    Full,
    /// The low 8 bits.
    I8,
    /// The low 16 bits.
    I16,
    /// The low 32 bits (of an `i64`).
    I32,
}

impl Storage {
    /// The width in bytes of narrow storage, and `None` for [Storage::Full].
    pub const fn bytes(self) -> Option<u64> {
        match self {
            Self::Full => None,
            Self::I8 => Some(1),
            Self::I16 => Some(2),
            Self::I32 => Some(4),
        }
    }

    /// Whether this is narrow storage: anything but [Storage::Full].
    pub const fn is_narrow(self) -> bool {
        !matches!(self, Self::Full)
    }

    /// The keyword of this storage: `full`, `i8`, `i16` or `i32`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
        }
    }
}

impl FromStr for Storage {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full" => Ok(Self::Full),
            "i8" => Ok(Self::I8),
            "i16" => Ok(Self::I16),
            "i32" => Ok(Self::I32),
            _ => Err("unknown storage"),
        }
    }
}

impl fmt::Display for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl runic_ir::formatter::PrettyPrint for Storage {
    fn render(&self) -> runic_ir::formatter::Document {
        runic_ir::formatter::display(self)
    }
}

impl AttrPrinter for StorageAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_keyword(self.value.as_str());
    }
}

impl AttrParser for StorageAttr {
    fn parse(
        parser: &mut dyn runic_ir::parse::Parser<'_>,
    ) -> runic_ir::parse::ParseResult<runic_ir::AttributeRef> {
        let keyword = parser
            .parse_keyword_from(&[
                Token::BareIdent("full"),
                Token::IntType(8),
                Token::IntType(16),
                Token::IntType(32),
            ])?
            .into_inner();
        let storage = keyword.as_str().parse::<Storage>().unwrap();
        Ok(parser
            .context_rc()
            .create_attribute::<StorageAttr, _>(storage)
            .as_attribute_ref())
    }
}

/// How a narrow load fills the bits above the storage it read: with the sign bit, or with zeros.
///
/// `Signed` and `Unsigned` are the `_s` and `_u` of `i32.load8_s` and `i64.load32_u`. A load of
/// its whole type, and every store, extends nothing: `Unextended`. Text: `#wasm.extension<signed>`,
/// `#wasm.extension<unsigned>`, `#wasm.extension<unextended>`; a body-less `#wasm.extension` is
/// `unextended`.
#[derive(DialectAttribute, Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
#[attribute(dialect = WasmDialect, name = "extension", implements(AttrPrinter))]
pub enum Extension {
    /// Nothing is extended: the access has its whole type.
    #[default]
    Unextended,
    /// The value read is sign-extended: the `_s` loads.
    Signed,
    /// The value read is zero-extended: the `_u` loads.
    Unsigned,
}

impl Extension {
    /// The keyword of this extension: `unextended`, `signed` or `unsigned`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unextended => "unextended",
            Self::Signed => "signed",
            Self::Unsigned => "unsigned",
        }
    }
}

impl FromStr for Extension {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "unextended" => Ok(Self::Unextended),
            "signed" => Ok(Self::Signed),
            "unsigned" => Ok(Self::Unsigned),
            _ => Err("unknown extension"),
        }
    }
}

impl fmt::Display for Extension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl runic_ir::formatter::PrettyPrint for Extension {
    fn render(&self) -> runic_ir::formatter::Document {
        runic_ir::formatter::display(self)
    }
}

impl AttrPrinter for ExtensionAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_keyword(self.value.as_str());
    }
}

impl AttrParser for ExtensionAttr {
    fn parse(
        parser: &mut dyn runic_ir::parse::Parser<'_>,
    ) -> runic_ir::parse::ParseResult<runic_ir::AttributeRef> {
        let keyword = parser
            .parse_keyword_from(&[
                Token::BareIdent("unextended"),
                Token::BareIdent("signed"),
                Token::BareIdent("unsigned"),
            ])?
            .into_inner();
        let extension = keyword.as_str().parse::<Extension>().unwrap();
        Ok(parser
            .context_rc()
            .create_attribute::<ExtensionAttr, _>(extension)
            .as_attribute_ref())
    }
}

/// The width in bytes of a value of `ty`, a Wasm number type; 1 for any other type, which no access
/// has (the verifier rejects it, and a text that wrote one still has a valid alignment).
fn value_width(context: &Context, ty: Type) -> u64 {
    if is_int_type(ty) {
        u64::from(ty.integer_width().unwrap_or(0)) / 8
    } else if ty == context.f32() {
        4
    } else if ty.is_float() {
        8
    } else {
        1
    }
}

/// The width in bytes of an access that moves a value of `ty` through `storage`: the storage when
/// it is narrow, the value otherwise. This is the natural alignment of the access, and the most an
/// alignment may be.
fn access_width(context: &Context, ty: Type, storage: Storage) -> u64 {
    storage.bytes().unwrap_or_else(|| value_width(context, ty))
}

/// Prints ` as i8 signed`: the narrow storage and, when `extension` is stated, how it is extended.
/// Nothing for an access of the whole value.
fn print_storage(printer: &mut AsmPrinter<'_>, storage: Storage, extension: Extension) {
    if !storage.is_narrow() {
        return;
    }
    *printer += const_text(" as ");
    printer.print_keyword(storage.as_str());
    if extension != Extension::Unextended {
        printer.print_space();
        printer.print_keyword(extension.as_str());
    }
}

/// Prints ` <{ offset = N, align = N }>`, leaving out the offset when it is 0 and the alignment
/// when it is `natural`, and everything when both are left out.
fn print_memarg(printer: &mut AsmPrinter<'_>, memarg: MemArg, natural: u64) {
    let offset = memarg.offset() != 0;
    let align = memarg.align() != natural;
    if !offset && !align {
        return;
    }
    *printer += const_text(" <{ ");
    if offset {
        *printer += const_text("offset = ");
        printer.print_decimal_integer(memarg.offset());
    }
    if align {
        if offset {
            *printer += const_text(", ");
        }
        *printer += const_text("align = ");
        printer.print_decimal_integer(memarg.align());
    }
    *printer += const_text(" }>");
}

/// Reads an optional `<{ offset = N, align = N }>`: each entry is optional, and the order is free.
fn parse_memarg(parser: &mut dyn OpAsmParser<'_>) -> ParseResult<(Option<u64>, Option<u64>)> {
    let (mut offset, mut align) = (None, None);
    if !parser.parse_optional_langle()? {
        return Ok((offset, align));
    }
    parser.parse_lbrace()?;
    loop {
        let key =
            parser.parse_keyword_from(&[Token::BareIdent("offset"), Token::BareIdent("align")])?;
        parser.parse_equal()?;
        let value = parser.parse_decimal_integer::<u64>()?.into_inner();
        let slot = if key.inner().as_str() == "offset" {
            &mut offset
        } else {
            &mut align
        };
        if slot.replace(value).is_some() {
            return Err(ParserError::InvalidAttributeValue {
                span: key.span(),
                reason: format!("`{}` is given twice", key.inner()),
            });
        }
        if !parser.parse_optional_comma()? {
            break;
        }
    }
    parser.parse_rbrace()?;
    parser.parse_rangle()?;
    Ok((offset, align))
}

/// Reads an optional `as i8`, `as i16` or `as i32`.
fn parse_storage(parser: &mut dyn OpAsmParser<'_>) -> ParseResult<Option<Storage>> {
    if parser.parse_optional_custom_keyword("as")?.is_none() {
        return Ok(None);
    }
    let keyword = parser
        .parse_keyword_from(&[Token::IntType(8), Token::IntType(16), Token::IntType(32)])?
        .into_inner();
    Ok(Some(
        keyword.as_str().parse::<Storage>().expect("the keywords above are storage"),
    ))
}

/// The static part of an access, as the text wrote it, with what it left out filled in: the
/// offset is 0 and the alignment the width of the access.
fn memarg_attr(
    parser: &mut dyn OpAsmParser<'_>,
    written: (Option<u64>, Option<u64>),
    width: u64,
) -> runic_ir::AttributeRef {
    let memarg = MemArg::new(written.0.unwrap_or(0), written.1.unwrap_or(width));
    parser.context_rc().create_attribute::<MemArgAttr, _>(memarg).as_attribute_ref()
}

/// Checks the access of `op`, a load or a store of `ty` through `storage` with `memarg`; `value`
/// is what the message calls the value moved (`its result`, `its value`), `extension` is the
/// extension of a load (a store passes `None`), and `rule` is the rule of the operation.
fn verify_access(
    op: &runic_ir::Operation,
    rule: &str,
    value: &str,
    ty: Type,
    storage: Storage,
    extension: Option<Extension>,
    memarg: MemArg,
) -> Result<(), Report> {
    if !is_num_type(ty) {
        return Err(rule_error(op, rule, format!("{value} has type `{}`", val_type_name(ty))));
    }
    let context = op.context();
    let verb = if extension.is_some() {
        "read"
    } else {
        "written"
    };
    if let Some(narrow) = storage.bytes() {
        if !is_int_type(ty) {
            return Err(rule_error(
                op,
                rule,
                format!(
                    "{value} has type `{}` and its storage is `{storage}`, but only an integer is \
                     {verb} through narrower storage (Wasm has no narrow float access)",
                    val_type_name(ty)
                ),
            ));
        }
        if narrow >= value_width(context, ty) {
            return Err(rule_error(
                op,
                rule,
                format!(
                    "{value} has type `{}` and its storage is `{storage}`, but narrow storage is \
                     narrower than the value (`i8` or `i16` for an `i32`; `i8`, `i16` or `i32` \
                     for an `i64`), and an access of the whole value names no storage",
                    val_type_name(ty)
                ),
            ));
        }
    }
    if let Some(extension) = extension {
        match (storage.is_narrow(), extension) {
            (true, Extension::Unextended) => {
                return Err(rule_error(
                    op,
                    rule,
                    format!(
                        "its storage is `{storage}`, but a narrow load says whether it extends \
                         the value read with `signed` or `unsigned`, and this one says neither"
                    ),
                ));
            }
            (false, Extension::Signed | Extension::Unsigned) => {
                return Err(rule_error(
                    op,
                    rule,
                    format!(
                        "it is `{extension}`, but a load of the whole value extends nothing (only \
                         a narrow load is `signed` or `unsigned`)"
                    ),
                ));
            }
            _ => (),
        }
    }
    let width = access_width(context, ty, storage);
    if memarg.align() > width {
        return Err(rule_error(
            op,
            rule,
            format!(
                "its alignment is {} bytes, but the access is {width} bytes wide and the \
                 alignment of a Wasm access is at most its width",
                memarg.align()
            ),
        ));
    }
    if memarg.offset() > u64::from(u32::MAX) {
        return Err(rule_error(
            op,
            rule,
            format!(
                "its offset is {}, but the offset of a Wasm access is an unsigned 32-bit integer",
                memarg.offset()
            ),
        ));
    }
    Ok(())
}

const LOAD_RULE: &str = "takes an `i32` address in a memory and gives a number, `i32`, `i64`, \
                         `f32` or `f64`, read from it";
const STORE_RULE: &str =
    "takes an `i32` address in a memory and a number, `i32`, `i64`, `f32` or `f64`, to write there";

/// A load from a named memory: `%v = wasm.load @mem, %p as i8 signed <{ offset = 8, align = 1 }> : i32;`.
///
/// It mirrors the fourteen load instructions: `i32.load`, `i64.load`, `f32.load` and `f64.load`
/// read a value of their own type, and `i32.load8_s`, `i32.load8_u`, `i32.load16_s`,
/// `i32.load16_u`, `i64.load8_s`, `i64.load8_u`, `i64.load16_s`, `i64.load16_u`, `i64.load32_s`
/// and `i64.load32_u` read a narrow integer and extend it. The address `%p` is an `i32`, the
/// result has the type written after the colon, a number type.
///
/// The properties say which instruction it is. `storage` ([Storage]) is `full` (the default) for
/// the four loads of a whole value and `i8`, `i16` or `i32` for a narrow load, which only an
/// integer result has and whose storage is narrower than the result (`i32` has `i8` and `i16`,
/// `i64` has all three). `extension` ([Extension]) is `signed` or `unsigned` for exactly the narrow
/// loads and `unextended` (the default) for the others. `memarg` ([MemArg]) is the static offset,
/// added to the address and at most 2^32 - 1, and the alignment in bytes, at most the width of the
/// access (the storage when narrow, the type otherwise). Nothing else verifies: no float has a
/// narrow load, no narrow load lacks an extension, and a load of a whole value has none.
///
/// The text writes `as STORAGE [signed | unsigned]` for a narrow load, then
/// `<{ offset = N, align = N }>` when the offset is not 0 or the alignment is not the width of the
/// access (each entry is optional), and the result type last. The alignment of the text is the
/// width of the access when absent.
///
/// The load reads [LinearMemoryResource] at its address. It traps when the access is out of
/// bounds, which is not an effect: an unused load is removed, and two loads of one address with no
/// write between them are merged.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
pub struct Load {
    /// The memory read
    #[symbol]
    memory: Memory,
    /// The address, an `i32`
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Read), resource(LinearMemoryResource))]
    addr: AnyType,
    /// The static offset and the alignment
    #[attr]
    memarg: MemArgAttr,
    /// How much is read
    #[attr]
    #[default]
    storage: StorageAttr,
    /// How narrow storage is extended
    #[attr]
    #[default]
    extension: ExtensionAttr,
    /// The value read
    #[result]
    result: AnyType,
}

impl OpPrinter for Load {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.memory()]);
        let storage = *self.get_storage();
        print_storage(printer, storage, *self.get_extension());
        let natural = access_width(op.context(), self.result().ty(), storage);
        print_memarg(printer, *self.get_memarg(), natural);
        printer.print_space();
        printer.print_colon_type(self.result().ty());
        print_attributes(printer, op);
    }
}

impl OpParser for Load {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["memory"], 1)?;
        let storage = parse_storage(parser)?;
        let mut extension = Extension::Unextended;
        if storage.is_some() {
            if parser.parse_optional_custom_keyword("signed")?.is_some() {
                extension = Extension::Signed;
            } else if parser.parse_optional_custom_keyword("unsigned")?.is_some() {
                extension = Extension::Unsigned;
            }
        }
        let written = parse_memarg(parser)?;
        let ty = parser.parse_colon_type()?.into_inner();
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let storage = storage.unwrap_or_default();
        let width = access_width(parser.context(), ty, storage);
        state.add_attribute("memarg", memarg_attr(parser, written, width));
        let context = parser.context_rc();
        state.add_attribute("storage", context.create_attribute::<StorageAttr, _>(storage));
        state.add_attribute("extension", context.create_attribute::<ExtensionAttr, _>(extension));
        state.results.push(ty);
        Ok(())
    }
}

/// The memory is a `wasm.memory`, the address is an `i32`, and the access is one of the fourteen
/// loads (see [Load]).
impl Verify<dyn MemoryEffectOpInterface> for Load {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<Memory>(op, &self.memory(), "wasm.memory")?;
        expect_operand(op, LOAD_RULE, 0, "its address", op.context().i32())?;
        verify_access(
            op,
            LOAD_RULE,
            "its result",
            self.result().ty(),
            *self.get_storage(),
            Some(*self.get_extension()),
            *self.get_memarg(),
        )
    }
}

/// A store into a named memory: `wasm.store @mem, %p, %v as i8 <{ offset = 8 }>;`.
///
/// It mirrors the nine store instructions: `i32.store`, `i64.store`, `f32.store` and `f64.store`
/// write a value of their own type, and `i32.store8`, `i32.store16`, `i64.store8`, `i64.store16`
/// and `i64.store32` write the low bits of an integer. The address `%p` is an `i32` and `%v` is
/// the value, a number type. `storage` and `memarg` are those of [Load]; a store has no extension
/// (the bits above the storage are dropped), so its text cannot say `signed` or `unsigned` and
/// nothing else about it is optional.
///
/// The text writes `as STORAGE` for a narrow store, then `<{ offset = N, align = N }>` as [Load]
/// does; the value has the type of the operand.
///
/// The store writes [LinearMemoryResource] at its address, so it is never removed, and no read of
/// memory is merged or moved across it. It traps when the access is out of bounds.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
pub struct Store {
    /// The memory written
    #[symbol]
    memory: Memory,
    /// The address, an `i32`
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(LinearMemoryResource))]
    addr: AnyType,
    /// The value written
    #[operand]
    value: AnyType,
    /// The static offset and the alignment
    #[attr]
    memarg: MemArgAttr,
    /// How much is written
    #[attr]
    #[default]
    storage: StorageAttr,
}

impl OpPrinter for Store {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.memory()]);
        let storage = *self.get_storage();
        print_storage(printer, storage, Extension::Unextended);
        let natural = access_width(op.context(), self.value().ty(), storage);
        print_memarg(printer, *self.get_memarg(), natural);
        print_attributes(printer, op);
    }
}

impl OpParser for Store {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let operands = parse_symbols_and_operands(state, parser, &["memory"], 2)?;
        let ty = operands[1].borrow().ty();
        let storage = parse_storage(parser)?;
        for keyword in ["signed", "unsigned"] {
            if let Some(span) = parser.parse_optional_custom_keyword(keyword)? {
                return Err(ParserError::InvalidAttributeValue {
                    span,
                    reason: format!(
                        "`{keyword}` is not part of a store: it keeps the low bits of its value \
                         and has no signedness"
                    ),
                });
            }
        }
        let written = parse_memarg(parser)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let storage = storage.unwrap_or_default();
        let width = access_width(parser.context(), ty, storage);
        state.add_attribute("memarg", memarg_attr(parser, written, width));
        state.add_attribute(
            "storage",
            parser.context_rc().create_attribute::<StorageAttr, _>(storage),
        );
        Ok(())
    }
}

/// The memory is a `wasm.memory`, the address is an `i32`, and the access is one of the nine
/// stores (see [Store]).
impl Verify<dyn MemoryEffectOpInterface> for Store {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<Memory>(op, &self.memory(), "wasm.memory")?;
        expect_operand(op, STORE_RULE, 0, "its address", op.context().i32())?;
        verify_access(
            op,
            STORE_RULE,
            "its value",
            self.value().ty(),
            *self.get_storage(),
            None,
            *self.get_memarg(),
        )
    }
}

/// The size of a memory in pages: `%pages = wasm.memory_size @mem;`. `memory.size`.
///
/// The result is an `i32`, the number of 64 KiB pages. It reads [LinearMemoryResource], since
/// `wasm.memory_grow` changes it.
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
#[effects(MemoryEffect(MemoryEffect::Read), resource(LinearMemoryResource))]
pub struct MemorySize {
    /// The memory measured
    #[symbol]
    memory: Memory,
    /// The number of pages
    #[result]
    result: AnyType,
}

const SIZE_RULE: &str = "takes a memory and gives its size in pages, an `i32`";

impl InferTypeOpInterface for MemorySize {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

impl OpPrinter for MemorySize {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.memory()]);
        print_attributes(printer, op);
    }
}

impl OpParser for MemorySize {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["memory"], 0)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        state.results.push(parser.context().i32());
        Ok(())
    }
}

/// The memory is a `wasm.memory` and the result is an `i32`.
impl Verify<dyn MemoryEffectOpInterface> for MemorySize {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<Memory>(op, &self.memory(), "wasm.memory")?;
        expect_result(op, SIZE_RULE, op.context().i32())
    }
}

/// Growing a memory: `%old = wasm.memory_grow @mem, %delta;`. `memory.grow`.
///
/// It adds `%delta` pages (an `i32`, unsigned) and gives the previous size in pages, or -1 (all
/// ones) when the memory cannot grow that far. It writes [LinearMemoryResource]: the new pages are
/// valid addresses, and `wasm.memory_size` changes, so no read of memory is merged or moved across
/// it, and it is never removed.
#[derive(EffectOpInterface)]
#[operation(
    dialect = WasmDialect,
    implements(InferTypeOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
#[effects(MemoryEffect(MemoryEffect::Write), resource(LinearMemoryResource))]
pub struct MemoryGrow {
    /// The memory grown
    #[symbol]
    memory: Memory,
    /// The number of pages added, an `i32`
    #[operand]
    delta: AnyType,
    /// The previous size in pages, or -1
    #[result]
    result: AnyType,
}

const GROW_RULE: &str = "takes a memory and an `i32` number of pages and gives an `i32`";

impl InferTypeOpInterface for MemoryGrow {
    fn infer_return_types(&mut self, context: &Context) -> Result<(), Report> {
        self.result_mut().set_type(context.i32());
        Ok(())
    }
}

impl OpPrinter for MemoryGrow {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.memory()]);
        print_attributes(printer, op);
    }
}

impl OpParser for MemoryGrow {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["memory"], 1)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        state.results.push(parser.context().i32());
        Ok(())
    }
}

/// The memory is a `wasm.memory`, the delta and the result are `i32`s.
impl Verify<dyn MemoryEffectOpInterface> for MemoryGrow {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<Memory>(op, &self.memory(), "wasm.memory")?;
        expect_operand(op, GROW_RULE, 0, "its delta", op.context().i32())?;
        expect_result(op, GROW_RULE, op.context().i32())
    }
}

/// Copying within or between memories: `wasm.memory_copy @dst, @src, %d, %s, %n;`. `memory.copy`.
///
/// It copies `%n` bytes (all `i32`) from the address `%s` of `@src` to the address `%d` of
/// `@dst`, as if through a buffer, so the ranges may overlap. It traps when either range is out of
/// bounds. It reads and writes [LinearMemoryResource], so it is never removed.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
pub struct MemoryCopy {
    /// The memory written
    #[symbol]
    dst: Memory,
    /// The memory read
    #[symbol]
    src: Memory,
    /// The first address written
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(LinearMemoryResource))]
    dst_addr: AnyType,
    /// The first address read
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Read), resource(LinearMemoryResource))]
    src_addr: AnyType,
    /// The number of bytes
    #[operand]
    len: AnyType,
}

const COPY_RULE: &str = "takes two memories and three `i32`s: a destination address, a source \
                         address and a number of bytes";

impl OpPrinter for MemoryCopy {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.dst(), &*self.src()]);
        print_attributes(printer, op);
    }
}

impl OpParser for MemoryCopy {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["dst", "src"], 3)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

/// Both memories are `wasm.memory` and the three operands are `i32`s.
impl Verify<dyn MemoryEffectOpInterface> for MemoryCopy {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<Memory>(op, &self.dst(), "wasm.memory")?;
        referenced_entity::<Memory>(op, &self.src(), "wasm.memory")?;
        let i32 = op.context().i32();
        expect_operand(op, COPY_RULE, 0, "its destination address", i32)?;
        expect_operand(op, COPY_RULE, 1, "its source address", i32)?;
        expect_operand(op, COPY_RULE, 2, "its length", i32)
    }
}

/// Filling a range of a memory: `wasm.memory_fill @mem, %d, %value, %n;`. `memory.fill`.
///
/// It sets `%n` bytes from the address `%d` to the low byte of `%value` (all three `i32`). It traps
/// when the range is out of bounds. It writes [LinearMemoryResource], so it is never removed.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
pub struct MemoryFill {
    /// The memory written
    #[symbol]
    memory: Memory,
    /// The first address written
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(LinearMemoryResource))]
    dst_addr: AnyType,
    /// The byte, in the low bits of an `i32`
    #[operand]
    value: AnyType,
    /// The number of bytes
    #[operand]
    len: AnyType,
}

const FILL_RULE: &str =
    "takes a memory and three `i32`s: a destination address, a byte value and a number of bytes";

impl OpPrinter for MemoryFill {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.memory()]);
        print_attributes(printer, op);
    }
}

impl OpParser for MemoryFill {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["memory"], 3)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

/// The memory is a `wasm.memory` and the three operands are `i32`s.
impl Verify<dyn MemoryEffectOpInterface> for MemoryFill {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<Memory>(op, &self.memory(), "wasm.memory")?;
        let i32 = op.context().i32();
        expect_operand(op, FILL_RULE, 0, "its destination address", i32)?;
        expect_operand(op, FILL_RULE, 1, "its value", i32)?;
        expect_operand(op, FILL_RULE, 2, "its length", i32)
    }
}

/// Copying passive data into a memory: `wasm.memory_init @mem, @data, %d, %s, %n;`. `memory.init`.
///
/// It copies `%n` bytes from the offset `%s` of the passive data segment `@data`, a
/// `builtin.global_variable`, to the address `%d` of `@mem` (all three `i32`). It traps when
/// either range is out of bounds, and when the segment was dropped (`wasm.data_drop`) and `%n` or
/// `%s` is not 0. It reads and writes [LinearMemoryResource], so it is never removed.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
#[effects(MemoryEffect(MemoryEffect::Read), resource(LinearMemoryResource))]
pub struct MemoryInit {
    /// The memory written
    #[symbol]
    memory: Memory,
    /// The data copied
    #[symbol]
    data: GlobalVariable,
    /// The first address written
    #[operand]
    #[effects(MemoryEffect(MemoryEffect::Write), resource(LinearMemoryResource))]
    dst_addr: AnyType,
    /// The first offset read in the data
    #[operand]
    src_offset: AnyType,
    /// The number of bytes
    #[operand]
    len: AnyType,
}

const INIT_RULE: &str = "takes a memory, a data segment (`builtin.global_variable`) and three \
                         `i32`s: a destination address, a source offset and a number of bytes";

impl OpPrinter for MemoryInit {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.memory(), &*self.data()]);
        print_attributes(printer, op);
    }
}

impl OpParser for MemoryInit {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["memory", "data"], 3)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

/// The memory is a `wasm.memory`, the data a `builtin.global_variable`, and the three operands are
/// `i32`s.
impl Verify<dyn MemoryEffectOpInterface> for MemoryInit {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<Memory>(op, &self.memory(), "wasm.memory")?;
        referenced_entity::<GlobalVariable>(op, &self.data(), "builtin.global_variable")?;
        let i32 = op.context().i32();
        expect_operand(op, INIT_RULE, 0, "its destination address", i32)?;
        expect_operand(op, INIT_RULE, 1, "its source offset", i32)?;
        expect_operand(op, INIT_RULE, 2, "its length", i32)
    }
}

/// Dropping a passive data segment: `wasm.data_drop @data;`. `data.drop`.
///
/// After it, `wasm.memory_init` of `@data` copies nothing (a range that is not empty traps). The
/// operation writes [LinearMemoryResource], which is what `memory_init` reads, so it is never
/// removed and a `memory_init` is not moved across it.
#[derive(EffectOpInterface)]
#[operation(dialect = WasmDialect, implements(MemoryEffectOpInterface, OpPrinter))]
#[effects(MemoryEffect(MemoryEffect::Write), resource(LinearMemoryResource))]
pub struct DataDrop {
    /// The data dropped
    #[symbol]
    data: GlobalVariable,
}

impl OpPrinter for DataDrop {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        let op = self.as_operation();
        print_symbols_and_operands(printer, op, &[&*self.data()]);
        print_attributes(printer, op);
    }
}

impl OpParser for DataDrop {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_symbols_and_operands(state, parser, &["data"], 0)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

/// The data is a `builtin.global_variable`.
impl Verify<dyn MemoryEffectOpInterface> for DataDrop {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let op = self.as_operation();
        referenced_entity::<GlobalVariable>(op, &self.data(), "builtin.global_variable")?;
        Ok(())
    }
}
