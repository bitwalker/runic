//! ## Syntax
//!
//! The grammar for the printed assembly format is given below:
//!
//! ```text
//! # File
//! top-level := (operation | attribute-alias-def | type-alias-def | rec-group-def)+
//!
//! A root operation is printed after the declarations of the recursion groups it refers to, one
//! `rec` block per group, each after the groups it refers to, then a blank line.
//!
//! # Operations
//! operation         := op-results? (generic-operation | custom-operation) trailing-location? ';'
//! custom-operation  := custom-op-name custom-operation-format
//! generic-operation := generic-op-name '(' value-uses? ')' successors? properties?
//!                        regions? attributes? ':' generic-op-signature
//! generic-op-signature := generic-op-types '->' generic-op-types
//! generic-op-types     := type-list-parens | type
//!
//! A generic operation's signature lists its operand types and its result types. It is not
//! itself a type: a lone operand or result type may be written bare (`i32 -> ()`), and `(i32)`
//! alone is not a type at all, since tuples are not a type form.
//!
//! # Locations
//!
//! trailing-location := 'loc' '(' location ')'
//! location := '?'
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
//! successors := '[' successor (',' successor)* ']'
//! successor  := block-id (':' block-arguments)?
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
//! block-arguments := '(' value-id-and-type-list? ')'
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
//! # Attribute Aliases
//!
//! attribute-alias-def := '#' alias-name '=' attribute-value ';'
//!
//! # Builtin Attributes
//!
//! builtin-attribute := type-attribute
//!                    | integer-attribute
//!                    | boolean-attribute
//!                    | string-attribute
//!                    | symbol-ref-attribute
//!                    | location-attribute
//!
//! type-attribute := type
//! integer-attribute := '-'? decimal-literal | hex-literal | binary-literal
//! boolean-attribute := 'true' | 'false'
//! string-attribute := string
//! symbol-ref-attribute := symbol-ref-id ('::' symbol-ref-id)*
//! location-attribute := 'loc' '(' location ')'
//!
//! No attribute is followed by a `: type` suffix. An integer literal is an attribute only where
//! the syntax that reads it expects an integer type, such as an operation's own format: an entry
//! of a dictionary has no expected type, so `{ a = 1 }` is an error. Any other builtin attribute
//! is written as a dialect attribute, `#builtin.name<..>`.
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
//! The following example demonstrates a few things:
//!
//! * A generic operation with results, a multi-block region, properties and attributes
//! * A few operations with custom formats
//! * Named values (e.g. `%flag`) and result packs (i.e. `%out:2`)
//! * An example location specifier
//!
//! ```text
//! %flag, %out:2 = "dialect.op"(%arg0, %0) <{ prop = true }> ({
//!     builtin.br ^after(%arg0, %0);
//! ^after(%1: i32, %2 : ptr<i8>):
//!     %3 = arith.constant 0 : i32;
//!     %4 = arith.eq %1, %3 : i1;
//!     builtin.ret %4, %1, %3;
//! }) { attr = "string" } : (i32, ptr<i8>) -> (i1, i32, i32) loc<"file.hir":1:1 to :32>;
//! ```
//!
//! This example shows the `cf.cond_br` operation in generic form, with some other notable features:
//!
//! * The operation has no results, regions, properties, attributes, or location specifier
//! * The operand and successor arguments both show how individual values of a result pack produced
//!   by a previous operation can be referenced.
//!
//! ```text
//! %overflowing_add:2 = arith.add %lhs, %rhs <{ overflow = overflowing }>;
//! "cf.cond_br"(%overflowing_add#0) [
//!     ^overflowed,
//!     ^didnt:(%overflowing_add#1)
//! ] : (i1, i32) -> ();
//! ```
//!
//! Additional notes about the printed format:
//!
//! * If an operation has properties or attributes, they must all be printed - or in the case of
//!   the custom format, present in the printed output. If not, then it will not be possible to
//!   round-trip the operation through the printed form and back.

mod asm_printer;
#[cfg(test)]
mod tests;

use alloc::format;
use core::fmt;

pub use self::asm_printer::AsmPrinter;
use super::{OpOperandRange, OpResultRange, Operation, Region, RegionList, ValueRange};
use crate::{EntityWithId, Location, Value, formatter::Document};

/// Options which configure how IR entities are printed to IR assembly
#[derive(Default, Debug)]
pub struct OpPrintingFlags {
    /// When `true`, forces printing of entry block headers for all regions
    pub print_entry_block_headers: bool,
    /// When `true`, prints trailing location specifiers after all operations, i.e. `loc(..)`
    pub print_source_locations: bool,
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
}

impl OpPrinter for Operation {
    #[inline]
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        use crate::formatter::*;

        if let Some(custom_printer) = self.as_trait::<dyn OpPrinter>() {
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
