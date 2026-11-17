//! The module-level operations: memories, tables, globals, element segments and tags.
//!
//! Each is a symbol that sits beside the functions of a `builtin.module`, in the symbol table's one
//! name space (Wasm has an index space for each kind; here a `wasm.global @g` and a
//! `builtin.function @g` of one module collide). What each says by itself is verified here; the
//! rules that need the whole module (how many memories there may be, the order of the index
//! spaces) are not.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use runic_ir::{
    AsSymbolRef, Context, Ident, Op, OpAsmParser, OpParser, OpPrinter, Operation, OperationRef,
    OperationState, ParseResult, Report, SmallVec, Spanned, Symbol, SymbolName, SymbolRef,
    SymbolUseList, UnsafeIntrusiveEntityRef, Usable, Value, Visibility,
    attributes::AttrList,
    derive::operation,
    diagnostics::report,
    dialects::builtin::attributes::{
        Array, BoolAttr, IdentAttr, IntAttr, List, ListAttr, SymbolRefAttr, TypeArrayAttr,
        TypeAttr, VisibilityAttr,
    },
    parse::{Delimiter, ParserError, ParserExt, Token},
    print::AsmPrinter,
    traits::{
        IsolatedFromAbove, NoRegionArguments, NoTerminator, RegionOperationsAreLive, SingleBlock,
        SingleRegion, TypeConstraint,
    },
};

use crate::{
    Limits, LimitsAttr, Mutability, MutabilityAttr, RefType, ValType, WasmDialect,
    is_constant_instruction, is_ref_type, is_val_type, val_type_name,
};

/// The most pages a 32-bit memory has: 65536 pages of 64 KiB, 4 GiB.
pub const MAX_MEMORY_PAGES: u64 = 65536;

/// The most elements a table has: Wasm limits are 32-bit.
pub const MAX_TABLE_ELEMENTS: u64 = u32::MAX as u64;

/// A module-level operation of the dialect: a [Memory], a [Table], a [Global], an [Elem] or a
/// [Tag].
///
/// Every one of them is a [Symbol]. The trait marks them, and the verifier of each (the checks of
/// what the operation states by itself) runs through it. An operation that names one of them by
/// symbol checks the kind with [referenced_entity].
pub trait ModuleEntity: Symbol {}

/// The report of `op` being invalid because of `problem`.
fn invalid(op: &Operation, problem: String) -> Report {
    report!(
        message: "invalid operation",
        labels: [primary(op.span(), problem)],
    )
}

/// The operation that `reference`, a symbol reference held by `op`, names, if it is a `T`.
///
/// `expected` is the name of `T` as the error spells it (`wasm.memory`). A reference that is linked
/// to no symbol, or to an operation that is not a `T`, is an error whose label says what the
/// reference names. This is the check an operation that takes a memory, a table, a global or a tag
/// makes in its verifier (the framework checks the kind of a symbol when it is set, not when the
/// IR is verified): `referenced_entity::<Memory>(op, &self.memory(), "wasm.memory")?`.
pub fn referenced_entity<T: Op>(
    op: &Operation,
    reference: &SymbolRefAttr,
    expected: &str,
) -> Result<UnsafeIntrusiveEntityRef<T>, Report> {
    let path = reference.path();
    let Some(symbol) = reference.resolve() else {
        return Err(invalid(
            op,
            format!(
                "`{}` names `@{path}`, which resolves to no symbol, not a `{expected}`",
                op.name()
            ),
        ));
    };
    let target = symbol.borrow().as_symbol_operation().as_operation_ref();
    let found = target.borrow().name();
    target.try_downcast_op::<T>().map_err(|_| {
        invalid(
            op,
            format!("`{}` names `@{path}`, which is a `{found}`, not a `{expected}`", op.name()),
        )
    })
}

/// Implements what every module-level symbol shares: the use list, the symbol interface and
/// `AsSymbolRef`. `$declaration` says whether the operation is a declaration.
macro_rules! module_symbol {
    ($Op:ident, |$this:ident| $declaration:expr) => {
        impl Usable for $Op {
            type Use = runic_ir::SymbolUse;

            #[inline(always)]
            fn uses(&self) -> &SymbolUseList {
                &self.uses
            }

            #[inline(always)]
            fn uses_mut(&mut self) -> &mut SymbolUseList {
                &mut self.uses
            }
        }

        impl Symbol for $Op {
            #[inline(always)]
            fn as_symbol_operation(&self) -> &Operation {
                &self.op
            }

            #[inline(always)]
            fn as_symbol_operation_mut(&mut self) -> &mut Operation {
                &mut self.op
            }

            fn name(&self) -> SymbolName {
                self.get_name().as_symbol()
            }

            fn set_name(&mut self, name: SymbolName) {
                self.get_name_mut().name = name;
            }

            fn visibility(&self) -> Visibility {
                *self.get_visibility()
            }

            fn set_visibility(&mut self, visibility: Visibility) {
                *self.get_visibility_mut() = visibility;
            }

            fn is_declaration(&self) -> bool {
                let $this = self;
                $declaration
            }
        }

        impl ModuleEntity for $Op {}

        impl AsSymbolRef for $Op {
            fn as_symbol_ref(&self) -> SymbolRef {
                self.as_symbol_operation()
                    .as_symbol_ref()
                    .expect("module-level operations are symbols")
            }
        }
    };
}

/// Prints ` public @name` or ` public declaration @name`.
fn print_header(
    printer: &mut AsmPrinter<'_>,
    visibility: Visibility,
    declaration: bool,
    name: Ident,
) {
    printer.print_space();
    printer.print_keyword(visibility.as_str());
    if declaration {
        printer.print_space();
        printer.print_keyword("declaration");
    }
    printer.print_space();
    printer.print_symbol_name(name.as_symbol());
}

/// Prints ` attributes { .. }` when the operation has discardable attributes.
fn print_attributes(printer: &mut AsmPrinter<'_>, op: &Operation) {
    use runic_ir::formatter::const_text;

    if op.has_attributes() {
        printer.print_space();
        *printer += const_text("attributes");
        printer.print_space();
        printer.print_attribute_dictionary(
            op.attributes().iter().map(|attr| *attr.as_named_attribute()),
        );
    }
}

/// Reads `public @name` or `public declaration @name`, and records the three properties. Returns
/// whether the operation is a declaration.
fn parse_header(
    state: &mut OperationState,
    parser: &mut dyn OpAsmParser<'_>,
    declarable: bool,
) -> Result<bool, ParserError> {
    let visibility = parser
        .parse_keyword_from(&[
            Token::BareIdent("public"),
            Token::BareIdent("private"),
            Token::BareIdent("internal"),
        ])?
        .into_inner()
        .parse::<Visibility>()
        .expect("the visibilities above are exhaustive");
    state.add_attribute(
        "visibility",
        parser.context_rc().create_attribute::<VisibilityAttr, _>(visibility),
    );
    let declaration = declarable && parser.parse_optional_custom_keyword("declaration")?.is_some();
    if declarable {
        state.add_attribute(
            "declaration",
            parser.context_rc().create_attribute::<BoolAttr, _>(declaration),
        );
    }
    let name = parser.parse_symbol_name()?;
    state.add_attribute("name", parser.context_rc().create_attribute::<IdentAttr, _>(name));
    Ok(declaration)
}

/// Prints limits as `MIN` or `MIN..MAX`.
fn print_limits(printer: &mut AsmPrinter<'_>, limits: Limits) {
    use runic_ir::formatter::const_text;

    printer.print_decimal_integer(limits.min());
    if let Some(max) = limits.max() {
        *printer += const_text("..");
        printer.print_decimal_integer(max);
    }
}

/// Reads limits written `MIN` or `MIN..MAX` and records them as the property `limits`.
fn parse_limits(
    state: &mut OperationState,
    parser: &mut dyn OpAsmParser<'_>,
) -> Result<(), ParserError> {
    let min = parser.parse_decimal_integer::<u64>()?.into_inner();
    let max = if parser.parse_optional_dot()? {
        parser.parse_dot()?;
        Some(parser.parse_decimal_integer::<u64>()?.into_inner())
    } else {
        None
    };
    let limits = match max {
        Some(max) => Limits::between(min, max),
        None => Limits::at_least(min),
    };
    state.add_attribute("limits", parser.context_rc().create_attribute::<LimitsAttr, _>(limits));
    Ok(())
}

/// A memory: `wasm.memory public @mem : 1..16;`.
///
/// It mirrors Wasm's `mem` (a memory definition) and `import` of a memory. The limits are in pages
/// of 64 KiB: a minimum, and a maximum when there is one. The memory is 32-bit, so each limit is at
/// most 65536 pages, and the maximum is at least the minimum. Where the text writes `declaration`
/// (`wasm.memory public declaration @mem : 1..16;`), the memory is an import: the property
/// `declaration` is true, and `wasm.import_module` says which module it comes from (`env` when the
/// entry is absent). A memory that is not a declaration is a definition. `wasm.export_name`
/// exports it, whether it is a definition or an import (Wasm exports what it imports).
///
/// The text form is `visibility [declaration] @name : MIN[..MAX]`, then an optional
/// `attributes { .. }`. The operation is a [Symbol] and has no region.
#[operation(dialect = WasmDialect, implements(Symbol, OpPrinter, ModuleEntity))]
pub struct Memory {
    #[attr]
    name: IdentAttr,
    #[attr]
    visibility: VisibilityAttr,
    #[attr]
    limits: LimitsAttr,
    #[attr]
    #[default]
    declaration: BoolAttr,
    #[default]
    uses: SymbolUseList,
}

/// A reference to a [Memory].
pub type MemoryRef = UnsafeIntrusiveEntityRef<Memory>;

module_symbol!(Memory, |this| *this.get_declaration());

impl OpPrinter for Memory {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        print_header(printer, *self.get_visibility(), *self.get_declaration(), *self.get_name());
        *printer += runic_ir::formatter::const_text(" : ");
        print_limits(printer, *self.get_limits());
        print_attributes(printer, &self.op);
    }
}

impl OpParser for Memory {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_header(state, parser, true)?;
        parser.parse_colon()?;
        parse_limits(state, parser)?;
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

/// The limits are at most `bound`; `unit` and `what` complete the message.
fn verify_limit_bounds(
    op: &Operation,
    name: &str,
    limits: Limits,
    bound: u64,
    rule: &str,
) -> Result<(), Report> {
    let what = |kind: &str, value: u64| {
        invalid(op, format!("`{}` `@{name}` has {kind} {value}, but {rule} {bound}", op.name()))
    };
    if limits.min() > bound {
        return Err(what("a minimum of", limits.min()));
    }
    match limits.max() {
        Some(max) if max > bound => Err(what("a maximum of", max)),
        _ => Ok(()),
    }
}

/// The limits are at most 65536 pages.
impl runic_ir::Verify<dyn ModuleEntity> for Memory {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        verify_limit_bounds(
            &self.op,
            self.get_name().as_str(),
            *self.get_limits(),
            MAX_MEMORY_PAGES,
            "a 32-bit memory has at most",
        )
    }
}

/// A table: `wasm.table private @tab : 0..8 x !wasm.funcref;`.
///
/// It mirrors Wasm's `table` and `import` of a table. The limits are in elements: a minimum, and a
/// maximum when there is one; each is at most 2^32 - 1, and the maximum is at least the minimum.
/// The element type is a reference type, `!wasm.funcref` or `!wasm.externref`. A `declaration` is
/// an import, as for [Memory].
///
/// The text form is `visibility [declaration] @name : MIN[..MAX] x TYPE`, then an optional
/// `attributes { .. }`.
#[operation(dialect = WasmDialect, implements(Symbol, OpPrinter, ModuleEntity))]
pub struct Table {
    #[attr]
    name: IdentAttr,
    #[attr]
    visibility: VisibilityAttr,
    #[attr]
    limits: LimitsAttr,
    #[attr]
    element_type: TypeAttr,
    #[attr]
    #[default]
    declaration: BoolAttr,
    #[default]
    uses: SymbolUseList,
}

/// A reference to a [Table].
pub type TableRef = UnsafeIntrusiveEntityRef<Table>;

module_symbol!(Table, |this| *this.get_declaration());

impl OpPrinter for Table {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use runic_ir::formatter::const_text;

        print_header(printer, *self.get_visibility(), *self.get_declaration(), *self.get_name());
        *printer += runic_ir::formatter::const_text(" : ");
        print_limits(printer, *self.get_limits());
        *printer += const_text(" x ");
        printer.print_type(*self.get_element_type());
        print_attributes(printer, &self.op);
    }
}

impl OpParser for Table {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_header(state, parser, true)?;
        parser.parse_colon()?;
        parse_limits(state, parser)?;
        parser.parse_custom_keyword("x")?;
        let ty = parser.parse_type()?.into_inner();
        state
            .add_attribute("element_type", parser.context_rc().create_attribute::<TypeAttr, _>(ty));
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

/// The limits are at most 2^32 - 1 elements, and the element type is a reference type.
impl runic_ir::Verify<dyn ModuleEntity> for Table {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let name = self.get_name().as_str();
        let ty = *self.get_element_type();
        if !is_ref_type(ty) {
            return Err(invalid(
                &self.op,
                format!(
                    "`wasm.table` `@{name}` has the element type `{}`, but a Wasm table holds \
                     references: {}",
                    val_type_name(ty),
                    RefType::get().description()
                ),
            ));
        }
        verify_limit_bounds(
            &self.op,
            name,
            *self.get_limits(),
            MAX_TABLE_ELEMENTS,
            "a Wasm table has at most",
        )
    }
}

/// A global: `wasm.global private @sp : mut i32 { %v = wasm.const 65536 : i32; };`.
///
/// It mirrors Wasm's `global` and `import` of a global. The type is a Wasm value type, and the
/// mutability says whether `global.set` may write it: `mut` is written before the type, and a
/// constant global writes nothing. The initializer is the region: one block, with no terminator,
/// holding a Wasm constant expression, and the value of the global is the one result of the last
/// operation, which has the global's type. Every operation of the initializer is a constant
/// instruction (see [is_constant_instruction]) and uses only values defined in the region (the
/// operation is isolated from above). A `global_get` in it names an immutable global that is an
/// import, or is defined before this global in the same block (Wasm types the initializer of a
/// global in a context of the globals before it, so a global cannot read itself, a later one, or
/// take part in a cycle).
///
/// A declaration, an import, is written with the keyword `declaration` and has no region, as for
/// [Memory]: `wasm.global private declaration @g : i32;`. A definition has its region, and a
/// region with no operation is an error, not an import.
///
/// The text form is `visibility [declaration] @name : [mut] TYPE`, then the region `{ .. }` of a
/// definition, then an optional `attributes { .. }`. The region form is the only one: a global with
/// one `wasm.const` is not abbreviated.
#[operation(
    dialect = WasmDialect,
    traits(SingleRegion, SingleBlock, NoRegionArguments, NoTerminator, IsolatedFromAbove, RegionOperationsAreLive),
    implements(Symbol, OpPrinter, ModuleEntity)
)]
pub struct Global {
    #[attr]
    name: IdentAttr,
    #[attr]
    visibility: VisibilityAttr,
    #[attr]
    ty: TypeAttr,
    #[attr]
    #[default]
    mutability: MutabilityAttr,
    #[attr]
    #[default]
    declaration: BoolAttr,
    #[region]
    initializer: RegionRef,
    #[default]
    uses: SymbolUseList,
}

/// A reference to a [Global].
pub type GlobalRef = UnsafeIntrusiveEntityRef<Global>;

module_symbol!(Global, |this| *this.get_declaration());

impl OpPrinter for Global {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use runic_ir::formatter::const_text;

        print_header(printer, *self.get_visibility(), *self.get_declaration(), *self.get_name());
        *printer += runic_ir::formatter::const_text(" : ");
        if self.get_mutability().is_mut() {
            printer.print_keyword("mut");
            printer.print_space();
        }
        printer.print_type(*self.get_ty());
        if !self.is_declaration() {
            *printer += const_text(" ");
            printer.print_region(&self.initializer());
        }
        print_attributes(printer, &self.op);
    }
}

impl OpParser for Global {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        let declaration = parse_header(state, parser, true)?;
        parser.parse_colon()?;
        let mutability = if parser.parse_optional_custom_keyword("mut")?.is_some() {
            Mutability::Mut
        } else {
            Mutability::Const
        };
        state.add_attribute(
            "mutability",
            parser.context_rc().create_attribute::<MutabilityAttr, _>(mutability),
        );
        let ty = parser.parse_type()?.into_inner();
        state.add_attribute("ty", parser.context_rc().create_attribute::<TypeAttr, _>(ty));
        // The region is always added, empty for a declaration. A definition must write one: a
        // missing region is not an import.
        let initializer = if declaration {
            parser.context().create_region()
        } else {
            match parser.parse_optional_region(&[], false)? {
                Some(region) => region,
                None => {
                    return Err(ParserError::from(report!(
                        message: "invalid wasm.global operation",
                        labels: [primary(
                            state.span,
                            "a global that is not a `declaration` has an initializer, a region \
                             `{{ .. }}` that holds a constant expression, but none is written"
                        )],
                    )));
                }
            }
        };
        state.regions.push(initializer);
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

impl Global {
    /// The operations of the initializer, in order. Empty for a declaration.
    fn initializer_ops(&self) -> Vec<OperationRef> {
        let region = self.initializer();
        if region.is_empty() {
            return Vec::new();
        }
        region.entry().body().iter().map(|op| op.as_operation_ref()).collect()
    }
}

impl Global {
    /// Checks the `global_get` `op` of this global's initializer: it reads an immutable global that
    /// is an import or is defined before this global in its block.
    fn verify_global_read(
        &self,
        name: &str,
        op: &Operation,
        get: &crate::GlobalGet,
    ) -> Result<(), Report> {
        // A reference that names no global is the business of the read's own verifier
        let Some(read) = get.global_op() else {
            return Ok(());
        };
        let (read_name, read_ref) = (read.borrow().get_name().as_str().to_string(), read);
        let invalid_read = |problem: String| {
            report!(
                message: "invalid operation",
                labels: [
                    primary(self.op.span(), problem),
                    context(op.span(), "this reads the global"),
                ],
            )
        };
        if read_ref.borrow().get_mutability().is_mut() {
            return Err(invalid_read(format!(
                "the initializer of `wasm.global` `@{name}` reads the mutable global \
                 `@{read_name}`, but a Wasm constant expression reads only immutable globals"
            )));
        }
        if read_ref.borrow().is_declaration() {
            return Ok(());
        }
        let read_op = read_ref.as_operation_ref();
        let mut earlier = self.op.as_operation_ref().prev();
        while let Some(candidate) = earlier {
            if candidate == read_op {
                return Ok(());
            }
            earlier = candidate.prev();
        }
        let which = if read_op == self.op.as_operation_ref() {
            "the global itself"
        } else {
            "a global that is not defined before this one"
        };
        Err(invalid_read(format!(
            "the initializer of `wasm.global` `@{name}` reads `@{read_name}`, which is {which}, \
             but a Wasm constant expression reads only an imported global or a global defined \
             before this one in the module"
        )))
    }
}

/// The type is a Wasm value type; a definition has an initializer of constant instructions whose
/// last operation gives one result of the type.
impl runic_ir::Verify<dyn ModuleEntity> for Global {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let name = self.get_name().as_str();
        let ty = *self.get_ty();
        if !is_val_type(ty) {
            return Err(invalid(
                &self.op,
                format!(
                    "`wasm.global` `@{name}` has the type `{}`, but a Wasm global holds a value: \
                     {}",
                    val_type_name(ty),
                    ValType::get().description()
                ),
            ));
        }
        let ops = self.initializer_ops();
        if self.is_declaration() {
            if !self.initializer().is_empty() {
                return Err(invalid(
                    &self.op,
                    format!(
                        "`wasm.global` `@{name}` is a declaration, but it has an initializer: an \
                         import has no region"
                    ),
                ));
            }
            return Ok(());
        }
        let Some(last) = ops.last() else {
            return Err(invalid(
                &self.op,
                format!(
                    "`wasm.global` `@{name}` has an initializer with no operation, but the \
                     initializer of a definition computes the value of the global (an import is \
                     written with `declaration` and has no region)"
                ),
            ));
        };
        for op in &ops {
            let op = op.borrow();
            if let Some(get) = op.downcast_ref::<crate::GlobalGet>() {
                self.verify_global_read(name, &op, get)?;
            }
            if !is_constant_instruction(&op) {
                return Err(report!(
                    message: "invalid operation",
                    labels: [
                        primary(
                            self.op.span(),
                            format!(
                                "the initializer of `wasm.global` `@{name}` holds `{}`, but a Wasm \
                                 constant expression holds only constant instructions",
                                op.name()
                            )
                        ),
                        context(op.span(), "this is not a constant instruction"),
                    ],
                ));
            }
        }
        let last = last.borrow();
        if last.num_results() != 1 {
            return Err(invalid(
                &self.op,
                format!(
                    "the initializer of `wasm.global` `@{name}` ends in `{}`, which has {} \
                     results, but the value of a constant expression is one result",
                    last.name(),
                    last.num_results()
                ),
            ));
        }
        let value = last.results()[0].borrow().ty();
        if value != ty {
            return Err(invalid(
                &self.op,
                format!(
                    "the initializer of `wasm.global` `@{name}` has the value type `{}`, but the \
                     global has the type `{}`",
                    val_type_name(value),
                    val_type_name(ty)
                ),
            ));
        }
        Ok(())
    }
}

/// An active element segment: `wasm.elem private @init : @tab at 0 [@f, @g];`.
///
/// It mirrors Wasm's `elem` of the active form with a table and a constant offset: the functions
/// are put into the table `@tab` from the element `at` the offset. The table is a [Table] whose
/// element type is `funcref`; the offset is an `i32` constant, written as `wasm.const` writes it;
/// each function resolves to a callable. A segment is always a definition: Wasm does not import one,
/// so it has no `declaration` form. Passive and declarative segments, and segments whose elements
/// are expressions, are not modelled.
///
/// The text form is `visibility @name : @TABLE at OFFSET [@f, @g, ..]`, then an optional
/// `attributes { .. }`.
#[operation(dialect = WasmDialect, implements(Symbol, OpPrinter, ModuleEntity))]
pub struct Elem {
    #[attr]
    name: IdentAttr,
    #[attr]
    visibility: VisibilityAttr,
    #[symbol]
    table: Table,
    #[attr]
    offset: IntAttr,
    #[attr]
    functions: ListAttr,
    #[default]
    uses: SymbolUseList,
}

/// A reference to an [Elem].
pub type ElemRef = UnsafeIntrusiveEntityRef<Elem>;

module_symbol!(Elem, |_this| false);

impl Elem {
    /// The entries of the function list, in order, whatever they are. A verified segment has only
    /// symbol references among them.
    fn function_entries(&self) -> Vec<runic_ir::AttributeRef> {
        let functions = self.get_functions();
        let entries: &AttrList = AsRef::<AttrList>::as_ref(&*functions);
        entries.iter().map(|entry| entry.as_attribute_ref()).collect()
    }

    /// The references to the functions of the segment, in order, one for each entry of the list:
    /// each is a symbol reference held by this operation.
    ///
    /// The verifier rejects a segment with an entry that is not a symbol reference, so for a
    /// verified segment this is the whole list, and no entry is skipped. For one that is not
    /// verified, the entries that are not symbol references are left out; ask
    /// [Self::function_count] to compare.
    pub fn function_refs(&self) -> Vec<UnsafeIntrusiveEntityRef<SymbolRefAttr>> {
        self.function_entries()
            .into_iter()
            .filter_map(|entry| entry.try_downcast_attr::<SymbolRefAttr>().ok())
            .collect()
    }

    /// The number of entries of the function list, symbol references or not.
    pub fn function_count(&self) -> usize {
        self.function_entries().len()
    }

    /// The table this segment initializes, or `None` if the reference is not linked to a [Table].
    pub fn table_op(&self) -> Option<TableRef> {
        let symbol = self.table().resolve()?;
        let op = symbol.borrow().as_symbol_operation().as_operation_ref();
        op.try_downcast_op::<Table>().ok()
    }

    /// The offset of the first element in the table, as the unsigned 32-bit value its bits are.
    pub fn offset_value(&self) -> u32 {
        (*self.get_offset()).unsigned() as u32
    }
}

impl OpPrinter for Elem {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use runic_ir::formatter::const_text;

        print_header(printer, *self.get_visibility(), false, *self.get_name());
        *printer += runic_ir::formatter::const_text(" : ");
        printer.print_symbol_path(self.table().path());
        *printer += const_text(" at ");
        printer.print_int(*self.get_offset());
        printer.print_space();
        *printer += const_text("[");
        for (index, entry) in self.function_entries().into_iter().enumerate() {
            if index > 0 {
                *printer += const_text(", ");
            }
            // An entry that is not a symbol is invalid, and printed as the attribute it is, so
            // that the text of invalid IR does not look like valid IR.
            match entry.try_downcast_attr::<SymbolRefAttr>() {
                Ok(reference) => printer.print_symbol_path(reference.borrow().path()),
                Err(entry) => printer.print_attribute_value(&*entry.borrow()),
            }
        }
        *printer += const_text("]");
        print_attributes(printer, &self.op);
    }
}

impl OpParser for Elem {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_header(state, parser, false)?;
        parser.parse_colon()?;
        let table = parser.parse_symbol_ref()?.into_inner();
        state.add_attribute("table", table);

        parser.parse_custom_keyword("at")?;
        let (span, literal) = parser.parse_integer_literal()?.into_parts();
        let i32 = parser.context().i32();
        let offset = literal.to_int(i32, span)?;
        state.add_attribute("offset", parser.context_rc().int_attr(offset, i32));

        let mut list = AttrList::new();
        parser.parse_comma_separated_list(Delimiter::Bracket, Some("function list"), |parser| {
            let reference = parser.parse_symbol_ref()?.into_inner();
            list.push_back(runic_ir::Attribute::as_attr(&*reference.borrow()).as_attr_ref());
            Ok(true)
        })?;
        state.add_attribute(
            "functions",
            parser.context_rc().create_attribute::<ListAttr, _>(List::from(list)),
        );
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

/// The table is a `wasm.table` of `funcref`, the offset is an `i32`, and each entry of the
/// function list is a symbol that names a callable.
impl runic_ir::Verify<dyn ModuleEntity> for Elem {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let name = self.get_name().as_str();
        let table = referenced_entity::<Table>(&self.op, &self.table(), "wasm.table")?;
        let element = *table.borrow().get_element_type();
        if element != crate::FuncRefType::get(self.op.context()) {
            return Err(invalid(
                &self.op,
                format!(
                    "`wasm.elem` `@{name}` fills the table `{}` of `{}`, but a segment of \
                     functions fills a `funcref` table",
                    self.table().path(),
                    val_type_name(element)
                ),
            ));
        }
        let offset = runic_ir::Attribute::ty(&*self.offset());
        if offset != self.op.context().i32() {
            return Err(invalid(
                &self.op,
                format!(
                    "`wasm.elem` `@{name}` has an offset of type `{}`, but the offset of a \
                     segment is an `i32` constant",
                    val_type_name(offset)
                ),
            ));
        }
        for (index, entry) in self.function_entries().into_iter().enumerate() {
            let Ok(reference) = entry.try_downcast_attr::<SymbolRefAttr>() else {
                return Err(invalid(
                    &self.op,
                    format!(
                        "`wasm.elem` `@{name}` has entry {index} of its function list that is not \
                         a function symbol, but every element of a segment of functions is a \
                         function"
                    ),
                ));
            };
            let reference = reference.borrow();
            let path = reference.path();
            let Some(symbol) = reference.resolve() else {
                return Err(invalid(
                    &self.op,
                    format!(
                        "`wasm.elem` `@{name}` lists `{path}`, which resolves to no symbol, but \
                         every element of a segment of functions is a function"
                    ),
                ));
            };
            if reference.resolve_callable().is_err() {
                let found = symbol.borrow().as_symbol_operation().name();
                return Err(invalid(
                    &self.op,
                    format!(
                        "`wasm.elem` `@{name}` lists `{path}`, which is a `{found}`, but every \
                         element of a segment of functions is a function"
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// A tag: `wasm.tag private @exn : (i32, i64);`.
///
/// It mirrors Wasm's `tag` (the exception-handling proposal) and `import` of a tag. The tag has a
/// list of parameter types, each a Wasm value type, and no results. A `declaration` is an import,
/// as for [Memory]. Nothing uses a tag yet: the operations that throw and catch are not part of the
/// dialect.
///
/// The text form is `visibility [declaration] @name : (TYPE, ..)`, then an optional
/// `attributes { .. }`.
#[operation(dialect = WasmDialect, implements(Symbol, OpPrinter, ModuleEntity))]
pub struct Tag {
    #[attr]
    name: IdentAttr,
    #[attr]
    visibility: VisibilityAttr,
    #[attr]
    params: TypeArrayAttr,
    #[attr]
    #[default]
    declaration: BoolAttr,
    #[default]
    uses: SymbolUseList,
}

/// A reference to a [Tag].
pub type TagRef = UnsafeIntrusiveEntityRef<Tag>;

module_symbol!(Tag, |this| *this.get_declaration());

impl Tag {
    /// The parameter types of the tag.
    pub fn param_types(&self) -> SmallVec<[runic_ir::Type; 4]> {
        self.get_params().iter().copied().collect()
    }
}

impl OpPrinter for Tag {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use runic_ir::formatter::const_text;

        print_header(printer, *self.get_visibility(), *self.get_declaration(), *self.get_name());
        *printer += runic_ir::formatter::const_text(" : ");
        *printer += const_text("(");
        for (index, ty) in self.get_params().iter().enumerate() {
            if index > 0 {
                *printer += const_text(", ");
            }
            printer.print_type(*ty);
        }
        *printer += const_text(")");
        print_attributes(printer, &self.op);
    }
}

impl OpParser for Tag {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        parse_header(state, parser, true)?;
        parser.parse_colon()?;
        let mut types = Vec::new();
        parser.parse_comma_separated_list(Delimiter::Paren, Some("type list"), |parser| {
            types.push(parser.parse_type()?.into_inner());
            Ok(true)
        })?;
        state.add_attribute(
            "params",
            parser.context_rc().create_attribute::<TypeArrayAttr, _>(Array::from(types)),
        );
        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;
        Ok(())
    }
}

/// Each parameter is a Wasm value type.
impl runic_ir::Verify<dyn ModuleEntity> for Tag {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        let name = self.get_name().as_str();
        for (index, ty) in self.get_params().iter().enumerate() {
            if !is_val_type(*ty) {
                return Err(invalid(
                    &self.op,
                    format!(
                        "`wasm.tag` `@{name}` has parameter {index} of type `{}`, but a Wasm tag \
                         takes values: {}",
                        val_type_name(*ty),
                        ValType::get().description()
                    ),
                ));
            }
        }
        Ok(())
    }
}
