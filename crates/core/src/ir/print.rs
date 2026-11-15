//! ## Syntax
//!
//! The grammar for the printed assembly format is given below:
//!
//! ```text
//! # File
//! top-level := (type-alias-def | rec-group-def)* operation
//!
//! A root operation is printed after the declarations of the recursion groups it refers to, one
//! `rec` block per group, each after the groups it refers to, then a blank line.
//!
//! # Operations
//! operation         := op-results? (generic-operation | custom-operation) trailing-location? ';'
//! custom-operation  := custom-op-name custom-operation-format
//! generic-operation := generic-op-name '(' value-uses? ')' successors? properties?
//!                        regions? attributes? ':' generic-op-signature
//! generic-op-signature     := generic-op-operand-types '->' generic-op-types
//! generic-op-operand-types := '(' operand-group ('|' operand-group)* ')' | type
//! operand-group            := type-list-no-parens?
//! generic-op-types         := type-list-parens | type
//!
//! The generic form holds the whole operation, and the parser builds the operation from it alone,
//! never with the operation's custom parser, then verifies it as it does any operation:
//!
//! * The operand list holds the operands of every operand group that no successor forwards, in
//!   group order. The signature lists their types and, when there is more than one such group,
//!   divides them into the groups with `|`; a group may be empty (`(ptr<i8> |)` is a group of one
//!   operand and an empty one). The division is written in the signature, where each operand is
//!   matched with its type, rather than as a property in the manner of MLIR's
//!   `operandSegmentSizes`: an entry of `properties` must be a property of the operation, and the
//!   group sizes are not one.
//! * Each successor gives the arguments it forwards, with their types; see `successors`.
//! * `properties` holds every property of the operation, a property at its default included. A
//!   parser also takes a property from the trailing `attributes` dictionary; a property given in
//!   both is an error, and so is an entry of `properties` that is not a property of the
//!   operation.
//! * The regions are written in the order the operation stores them, which a custom form need
//!   not follow (`scf.index_switch` writes its default region last).
//! * The result types are those the operation has: a type the text gives a result that differs
//!   from the type the operation gives it once built (inferring it from its operands, say) is an
//!   error at the type written.
//! * The location follows the signature after a space, as in the custom form.
//!
//! A generic operation's signature lists its operand types and its result types. It is not
//! itself a type: a lone operand or result type may be written bare (`i32 -> ()`), and `(i32)`
//! alone is not a type at all, since tuples are not a type form.
//!
//! An operation prints in the generic form when it has no custom printer; every operation does
//! when [OpPrintingFlags::generic] is set; and an operation with a custom printer does when one of
//! its operands is defined after it in the text, unless its custom form writes every operand's
//! type ([OpPrinter::states_operand_types]): a custom form that takes an operand's type from the
//! operand could not be read back, since the type is not known where the use is read. An operand
//! is defined after the operation when its definition is the operation itself or one that holds
//! it, or comes later in the region that holds the definition: later in the same block as the
//! operation (or the operation that holds it), which a graph region allows; or in a block stored
//! after that block, which a region with control flow also allows, since a block may be stored
//! after a block it dominates (`^a: br ^c; ^b: %y = test.add %x, %x; ..; ^c: %x = ..; br ^b`).
//!
//! # Locations
//!
//! trailing-location := 'loc' '(' location ')'
//! location := 'unknown'
//!           | 'synthetic'
//!           | 'opaque' '<' decimal-literal '>'
//!           | file-line-col
//!           | file-line-col-range
//! file-line-col       := file-uri ':' line-number ':' column-number
//! file-line-col-range := file-line-col 'to' line-number? ':' column-number
//! file-uri := string
//! line-number   := nonzero-decimal-literal
//! column-number := nonzero-decimal-literal
//!
//! With [OpPrintingFlags::print_source_locations], every operation is followed by its location,
//! and so is every block argument where it is defined: in a block label, or in the parameter
//! list of a function's custom form; both forms read the locations back. A location alias
//! (`loc(#name)`) is an error, since nothing defines one.
//!
//! # Numbers
//!
//! decimal-literal := [0-9]+
//! nonzero-decimal-literal := [1-9][0-9]*
//! hex-literal := '0x' [A-Fa-f0-9]+
//! binary-literal := '0b' [0-1]+
//!
//! # Identifiers
//!
//! bare-id    := [A-Za-z_][A-Za-z0-9_$.]*
//! string     := '"' [^"]* '"'        (in which '\' escapes the character after it)
//! alias-name := bare-id
//! value-id   := '%' suffix-id
//! caret-id   := '^' suffix-id
//! block-id   := caret-id
//! suffix-id  := [0-9]+
//!             | [A-Za-z$._-][A-Za-z0-9$._-]*
//! symbol-ref-id   := '@' (bare-id | string)
//! generic-op-name := string
//! custom-op-name  := bare-id
//! dialect-namespace := bare-id
//!
//! # Values
//!
//! op-results := op-result (',' op-result)* '='
//! op-result  := value-id (':' decimal-literal)?
//! value-uses := value-use (',' value-use)*
//! value-use  := value-id ('#' decimal-literal)
//! value-use-and-type      := value-use ':' type
//! value-use-and-type-list := value-use-and-type (',' value-use-and-type)*
//! value-id-and-type-list  := value-id-and-type (',' value-id-and-type)*
//! value-id-and-type       := value-id ':' type
//!
//! # Successors
//!
//! successors          := '[' successor-group ('|' successor-group)* ']'
//! successor-group     := (successor (',' successor)*)?
//! successor           := (attribute-value ':')? block-id successor-arguments?
//! successor-arguments := '(' value-uses ':' type-list-no-parens ')'
//!
//! The successors are written in the groups the operation stores them in, separated by `|` when
//! it has more than one (`cf.switch` holds its keyed cases, then its fallback:
//! `[ #builtin.i32<0>: ^a, #builtin.i32<1>: ^b | ^c ]`); the text gives as many groups as the
//! operation declares, and as many successors in each as a group of fixed size declares. A
//! successor in a keyed group is written after its key, an attribute of the kind the group
//! stores; a successor in any other group has no key. A list holds at least one successor.
//!
//! # Regions
//!
//! regions := '(' region (',' region)* ')'
//! region  := '{' entry-block? block* '}'
//!
//! # Blocks
//!
//! entry-block     := operation+
//! block           := block-label operation+
//! block-label     := block-id block-arguments? ':'
//! block-arguments := '(' (block-argument (',' block-argument)*)? ')'
//! block-argument  := value-id ':' type trailing-location?
//!
//! The label of a region's entry block is left out when nothing needs it: in the generic form,
//! when the block has no arguments; in a custom form, also when the form gives the arguments
//! itself (a function's parameters, say). An entry block that a branch targets is labelled. A
//! region whose one block has no arguments and no operations is written `{ }`, as a region with
//! no block is.
//!
//! # Properties and Attributes
//!
//! properties := '<' dictionary-attribute '>'
//! attributes := dictionary-attribute
//! dictionary-attribute := '{' (attribute-entry (',' attribute-entry)*)? '}'
//! attribute-entry := attribute-name '=' attribute-value
//! attribute-name  := bare-id | string
//! attribute-value := dialect-attribute | builtin-attribute
//!
//! A dictionary holds attributes, but is not itself an attribute value: `{ a = {} }` is an error.
//!
//! # Functions
//!
//! function      := 'builtin.function' visibility? 'extern'? '(' string ')' symbol-ref-id
//!                  (definition | declaration) ('attributes' dictionary-attribute)?
//! definition    := '(' (argument (',' argument)*)? ')' ('->' result-list)? region
//! declaration   := '(' (param (',' param)*)? ')' ('->' result-list)?
//! argument      := value-id ':' type dictionary-attribute? trailing-location?
//! param         := type dictionary-attribute?
//! result-list   := type | '(' (param (',' param)*)? ')'
//!
//! The dictionary of a parameter or result holds its attributes, such as
//! `{ extend = #builtin.zext }` or `{ sret = #builtin.sret }`, and is left out when it is empty. A
//! result list is written without parentheses only when it is one result without attributes whose
//! type does not start with `(`, so the `{` after `-> i32` is always a body. An empty parameter
//! list followed by a region is a definition, and followed by anything else a declaration; a
//! non-empty list is a definition exactly when its first element starts with a value id, and an
//! element of the other form is an error. A declaration has no body. A function that never returns
//! has the single result `never`.
//!
//! # Builtin Attributes
//!
//! builtin-attribute := type-attribute
//!                    | integer-attribute
//!                    | int-attribute
//!                    | boolean-attribute
//!                    | string-attribute
//!                    | symbol-ref-attribute
//!                    | location-attribute
//!                    | signature-attribute
//!
//! type-attribute := type
//! integer-attribute := '-'? decimal-literal | hex-literal | binary-literal
//! int-attribute := '#builtin.int' '<' integer-literal ':' type '>'
//! boolean-attribute := 'true' | 'false'
//! string-attribute := string
//!     (every quoted string is written with the escapes of `str::escape_default` and read back by
//!     undoing them; an invalid escape is a parse error. The two exceptions are the quoted
//!     operation name of the generic form and a base64 byte string, which are taken as written)
//! symbol-ref-attribute := symbol-ref-id ('::' symbol-ref-id)*
//! location-attribute := 'loc' '(' location ')'
//! signature-attribute := '#builtin.signature' '<' 'extern' '(' string ')'
//!                        '(' (param (',' param)*)? ')' ('->' result-list)? '>'
//!     (`param` and `result-list` as for functions, read and written by the same code)
//!
//! No bare attribute is followed by a `: type` suffix. An integer literal is an attribute only
//! where the syntax that reads it expects an integer type, such as an operation's own format: an
//! entry of a dictionary has no expected type, so `{ a = 1 }` is an error. An integer literal
//! read where an integer type is expected is a `#builtin.int` attribute of that type, and one
//! written as `#builtin.int<-1 : i7>` carries its type. Any other builtin attribute is written
//! as a dialect attribute, `#builtin.name<..>`; the integer, bool and string attributes
//! (`#builtin.i32<-1>`, `#builtin.bool<true>`, `#builtin.string<"..">`) all parse back from what
//! they print, a string being escaped as by `str::escape_default`.
//!
//! In attribute position a type is a type attribute, so `array<`, `ptr`, `struct` and `(` start a
//! type. `[` starts no attribute value (it is kept for list attributes), and neither does `unit`.
//!
//! # Dialect Attributes
//!
//! dialect-attribute := '#' (opaque-dialect-attr | pretty-dialect-attr)
//! opaque-dialect-attr := dialect-namespace dialect-attr-body
//! pretty-dialect-attr := dialect-namespace '.' pretty-dialect-attr-lead-id dialect-attr-body?
//! pretty-dialect-attr-lead-id := [A-Za-z][A-Za-z0-9._]*
//! dialect-attr-body := '<' dialect-attr-contents+ '>'
//! dialect-attr-contents := dialect-attr-body
//!                        | '(' dialect-attr-contents+ ')'
//!                        | '[' dialect-attr-contents+ ']'
//!                        | '{' dialect-attr-contents+ '}'
//!                        | [^\[<({\]>)}\0]+
//!
//! # Types
//!
//! type := type-alias | dialect-type | builtin-type
//! type-list-no-parens := type (',' type)*
//! type-list-parens := '(' (type-list-no-parens)? ')'
//!
//! # Type Aliases
//!
//! type-alias := '!' alias-name
//! type-alias-def := 'type' '!' alias-name '=' type ';'
//! rec-group-def  := 'rec' '{' type-alias-def+ '}'
//!
//! `type` and `rec` are keywords only here: `type` followed by `!`, and `rec` followed by `{`.
//! A `rec` block declares a recursion group, whose members are its aliases in the order they are
//! declared. Inside the block, an alias that is not defined before the block names a member of the
//! block, whether or not it has been declared yet; after the block, each alias names its member.
//! Blocks do not nest. The printer names the members itself: after the name of a named struct or
//! enum (`!list`, then `!list_0`, `!list_1`, .. for others with that name), and `!t0`, `!t1`, ..
//! otherwise.
//!
//! # Builtin Types
//!
//! builtin-type := unknown-type
//!               | never-type
//!               | integer-type
//!               | index-type
//!               | float-type
//!               | pointer-type
//!               | array-type
//!               | struct-type
//!               | enum-type
//!               | function-type
//!
//! unknown-type := '?'
//! never-type := 'never'
//!
//! integer-type := 'i' [1-9][0-9]*            (a signless integer of 1 to 16777215 bits)
//! index-type := 'isize'
//! float-type := 'f32' | 'f64'
//!
//! pointer-type := 'ptr' ('<' (address-space | type (',' address-space)?) '>')?
//! address-space := decimal-literal
//!
//! array-type := 'array' '<' type ';' decimal-literal '>'
//!
//! struct-type  := 'struct' struct-name? '<' (struct-repr ';')? (struct-field (',' struct-field)*)? '>'
//! struct-name  := bare-id | string
//! struct-repr  := 'align' '(' nonzero-decimal-literal ')'
//!               | 'packed' ('(' nonzero-decimal-literal ')')?
//!               | 'transparent'
//! struct-field := (field-name ':')? type ('align' '(' nonzero-decimal-literal ')')?
//! field-name   := bare-id | string
//!
//! enum-type    := 'enum' enum-name? '<' type (';' enum-variant (',' enum-variant)*)? '>'
//! enum-name    := bare-id | string
//! enum-variant := variant-name ('=' integer-literal)? (':' type)?
//! variant-name := bare-id | string
//!
//! The type of an enum is its discriminant, an integer type of at most 128 bits or `isize`. A
//! variant without `= value` has the previous variant's value plus one, or 0 if it is the
//! first. The printer leaves values out only while the variants count up from 0; from the first
//! variant whose value is not its position, it writes every value, as a signed decimal:
//! `enum<i8; a, b = 5, c = 6>`.
//!
//! function-type := type-list-parens '->' function-results
//! function-results := type-list-parens | type
//!
//! A function type's results start with '(' exactly when they are a parenthesised list, so a
//! single result that is itself a function type is parenthesised: `() -> (() -> i32)`.
//!
//! # Dialect Types
//!
//! dialect-type := '!' dialect-namespace '.' dialect-type-name dialect-type-body?
//! dialect-type-body := '<' dialect-type-contents '>'
//!
//! The contents of a dialect type's body are whatever the type's own printer prints, and are
//! parsed by its own parser.
//!
//! ```
//!
//! ## Examples
//!
//! A function in the generic form, with these features:
//!
//! * The entry block of the function's body, labelled with the function's arguments
//! * A result pack (`%sum:2`) and the use of one of its values (`%sum#1`)
//! * An operation with two operand groups (`test.sink`: one operand, then the rest)
//! * Successors that forward arguments, and the keyed cases and the fallback of a `cf.switch`
//!
//! ```text
//! "builtin.function"() <{ name = #builtin.ident<@f>, linkage = #builtin.visibility<public>,
//!         signature = #builtin.signature<extern("C") (i32, i64) -> i32>,
//!         memory_effects = #builtin.declared_effects<unknown>, locals = #builtin.type_array<> }> ({
//! ^entry(%a: i32, %b: i64):
//!     %sum:2 = "arith.add_overflowing"(%a, %a) <{ signedness = #builtin.signedness<signed> }>
//!         : (i32, i32) -> (i1, i32);
//!     "test.sink"(%a, %b, %a) : (i32 | i64, i32) -> ();
//!     "cf.cond_br"(%sum#0) [ ^overflowed, ^done(%sum#1 : i32) ] : i1 -> ();
//! ^overflowed:
//!     "cf.switch"(%a) [ #builtin.i32<0>: ^done(%a : i32), #builtin.i32<1>: ^overflowed
//!         | ^done(%sum#1 : i32) ] : i32 -> ();
//! ^done(%r: i32):
//!     "builtin.ret"(%r) : i32 -> ();
//! }) : () -> () loc("file.runic":1:1);
//! ```
//!
//! Additional notes about the printed format:
//!
//! * If an operation has properties or attributes, they must all be printed - or in the case of
//!   the custom format, present in the printed output. If not, then it will not be possible to
//!   round-trip the operation through the printed form and back.
//! * The one exception is a property whose field is marked `#[default]`: an operation whose text
//!   does not give it gets the attribute's registered default, so the derived custom format does
//!   not print it while it has that value, and leaves out the `<{ .. }>` dictionary when no
//!   property remains: `test.add %0, %1` is the wrapping addition. The generic form prints every
//!   property.

mod asm_printer;
mod renumber;
#[cfg(test)]
mod tests;

use alloc::format;
use core::fmt;

pub use self::{
    asm_printer::{AsmPrinter, PrimitiveInteger},
    renumber::{renumbered, same_program},
};
use super::{OpOperandRange, OpResultRange, Operation, Region, ValueRange};
use crate::{EntityWithId, Location, Value, formatter::Document};

/// Options which configure how IR entities are printed to IR assembly
#[derive(Default, Debug, Clone, Copy)]
pub struct OpPrintingFlags {
    /// When `true`, forces printing of entry block headers for all regions
    pub print_entry_block_headers: bool,
    /// When `true`, prints trailing location specifiers after all operations, i.e. `loc(..)`
    pub print_source_locations: bool,
    /// When `true`, every operation, at every depth and including the one printing starts from,
    /// prints in the generic form ([AsmPrinter::print_operation_generic]) rather than its custom
    /// form.
    pub generic: bool,
}

/// A trait which must be implemented by all attribute types, for printing the attribute in
/// assembly format
///
/// A type must be written with [AsmPrinter::print_type] or one of the printer's other type
/// methods, never with its `Display` form: only the printer writes recursion-group members as
/// the aliases its declarations define. A printer created to print a part separately must be
/// made with [AsmPrinter::nested].
pub trait AttrPrinter {
    fn print(&self, printer: &mut AsmPrinter<'_>);
}

/// The `OpPrinter` trait is expected to be implemented by all [crate::Op] impls as a prequisite.
///
/// The actual implementation is typically generated as part of deriving [crate::Op].
///
/// A type must be written with [AsmPrinter::print_type] or one of the printer's other type
/// methods, never with its `Display` form: only the printer writes recursion-group members as
/// the aliases its declarations define. A printer created to print a part separately must be
/// made with [AsmPrinter::nested].
pub trait OpPrinter {
    /// Prints this operation with the given `flags`
    fn print(&self, printer: &mut AsmPrinter<'_>);

    /// Whether this custom form writes the type of every operand it reads, so that the parser
    /// knows an operand's type where it reads the use even when the operand is defined later in
    /// the text. An operation whose form does not, and which has such an operand, is printed in
    /// the generic form instead (see the module docs).
    ///
    /// The derived printer answers `true` when its format writes a type for every operand: the
    /// operation does not infer its types, so its signature lists its operands' types, or it has
    /// no operands but those its successors forward, which are written with their types. A
    /// hand-written printer answers `false` unless it says otherwise.
    fn states_operand_types(&self) -> bool {
        false
    }
}

impl OpPrinter for Operation {
    #[inline]
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::*;

        if !printer.flags().generic
            && let Some(custom_printer) = self.as_trait::<dyn OpPrinter>()
            && (custom_printer.states_operand_types() || !has_forward_operand(self, printer))
        {
            printer.print_results(self.results().all());
            *printer += display(self.name());
            custom_printer.print(printer);
            // Add source location if requested
            if printer.flags().print_source_locations {
                let loc = Location::from_span(self.span, self.context());
                printer.print_space();
                printer.print_trailing_location_specifier(&loc);
            }
            *printer += const_text(";");
        } else {
            printer.print_operation_generic(self);
        }
    }
}

/// Whether an operand of `op` is defined after `op` in the printed text: by `op` itself, by an
/// operation that holds `op`, or at a later place in the region that holds the definition.
///
/// The place of the use in the definition's region is that of `op`, or of the operation that
/// holds `op` and sits in a block of that region. In the definition's block, the use is forward
/// when the definition is that operation or comes after it (a block argument never does). In
/// another block of the region, it is forward when the definition's block is stored, and so
/// printed, after the block of the use: a graph region allows that, and so does a region with
/// control flow, in which a block may be stored after a block it dominates.
///
/// A custom form that infers its operands' types cannot name such a value, since its type is
/// not known where the parser reads the use, so `op` is printed in the generic form, which
/// states every operand's type.
fn has_forward_operand(op: &Operation, printer: &AsmPrinter<'_>) -> bool {
    let this = op.as_operation_ref();
    op.operands().iter().any(|operand| {
        let value = operand.borrow().as_value_ref();
        let (def, def_block) = {
            let value = value.borrow();
            (value.get_defining_op(), value.parent_block())
        };
        let Some(def_block) = def_block else {
            return false;
        };
        let Some(def_region) = def_block.borrow().parent() else {
            return false;
        };
        // Find the operation in a block of `def_region` that is `op` or holds it.
        let mut ancestor = this;
        let use_block = loop {
            if def == Some(ancestor) {
                return true;
            }
            let (parent_block, parent_op) = {
                let ancestor = ancestor.borrow();
                (ancestor.parent(), ancestor.parent_op())
            };
            let Some(parent_block) = parent_block else {
                return false;
            };
            if parent_block.borrow().parent() == Some(def_region) {
                break parent_block;
            }
            match parent_op {
                Some(parent) => ancestor = parent,
                None => return false,
            }
        };
        if use_block == def_block {
            def.is_some_and(|def| !def.borrow().is_before_in_block(&ancestor))
        } else {
            printer.block_position(def_block) > printer.block_position(use_block)
        }
    })
}

/// Renders an operation as IR assembly.
impl crate::formatter::PrettyPrint for Operation {
    /// Prints this operation.
    ///
    /// An operation without a parent operation is the root of what is printed, and its text is
    /// preceded by the declarations of the recursion groups it refers to (see
    /// [AsmPrinter::rec_group_declarations]) and a blank line, so that it parses on its own. An
    /// operation with a parent is printed as a fragment ([AsmPrinter::fragment]), with no
    /// declarations: its text is part of its root's, and its recursion-group members are written
    /// as aliases declared there.
    fn render(&self) -> Document {
        let flags = OpPrintingFlags::default();
        let context = self.context_rc();
        let mut printer = if self.parent_op().is_none() {
            AsmPrinter::root(context, &flags)
        } else {
            AsmPrinter::fragment(context, &flags)
        };
        self.print(&mut printer);
        printer.finish()
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use crate::formatter::PrettyPrint;

        write!(f, "{}", self.render())
    }
}
