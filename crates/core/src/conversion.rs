//! Target-driven dialect conversion.
//!
//! A conversion rewrites the operations under a root operation from one set of operations and
//! types to another. It is driven by a [ConversionTarget], which says which operations are legal;
//! a set of [ConversionPattern]s, each rooted at an operation (or at any operation), which rewrite
//! an operation the target does not accept; and, per pattern, an optional [TypeConverter], which
//! says what type each value becomes: one type, several, or none. Concrete lowerings own their
//! targets, patterns and converters; this module owns the driver.
//!
//! # The pattern contract
//!
//! A pattern is given the operation and its converted operands, and makes every change through
//! the [ConversionPatternRewriter] it is given. It answers `Ok(false)` when it changed nothing,
//! `Ok(true)` when it changed the IR, and `Err` to abort the conversion. Changes are applied at
//! once and nothing is rolled back: a pattern that changes the IR and then answers `Ok(false)` or
//! `Err`, or answers `Ok(true)` without a change, is an error, and the IR is left as the pattern
//! left it. An attempt that ends before its pattern changed anything leaves the IR as it was: the
//! casts the driver made for its operands are erased. A pattern declares the operations it
//! creates (its generated operations); creating any other is an error. The operations a pattern
//! creates or changes are legalized after it succeeds, and the operation it was given is offered
//! to the patterns again while it is still in the IR and not accepted.
//!
//! # Casts carry "this value is now these values"
//!
//! There is no map from old values to new ones. A value that became other values is related to
//! them by a `builtin.unrealized_conversion_cast` in the IR, which takes any number of values and
//! gives any number. The driver inserts two kinds, each between type lists that differ:
//!
//! - a **target** cast, of one value to the values of its converted types, made for an operand
//!   that a pattern is given;
//! - a **source** cast, of several values (or none) to one value of an old type, made for the
//!   users that still expect a value that was replaced, retyped or dropped.
//!
//! Each cast is recorded with the type converter of the pattern that caused it.
//!
//! # Operands are ranges
//!
//! A pattern is given each operand as the values it became ([ConvertedOperands]), with the
//! operation's operand groups still visible. Without a type converter an operand is itself. With
//! one ([TypeConverter::convert_value]), an operand whose type converts to exactly itself is
//! itself; one defined by a cast whose inputs have exactly the converted types is those inputs
//! (this is how a user converted after its producer reads the values the producer became); an
//! operand that converts to no type is an empty range; any other is the results of a new target
//! cast, placed before the operation. A pattern written for one value per operand asks for that
//! ([ConvertedOperands::one], [ConvertedOperands::all_one]) and does not match otherwise.
//!
//! # Results are replaced by ranges
//!
//! [ConversionPatternRewriter::replace_op_with_multiple] replaces each result by a range of values:
//! one value of the result's own type, or values of exactly the types the converter gives the
//! result. A result that still has users is replaced, for them, by a source cast of the range back
//! to the result's type, placed before the operation. [ConversionPatternRewriter::replace_op] and
//! [ConversionPatternRewriter::replace_op_with_new_op] are the one-value forms.
//!
//! # Materializations
//!
//! When the patterns have run, chains of casts whose ends agree are folded away and casts without
//! users are erased ([reconcile_unrealized_casts]). Each cast the driver inserted that remains is
//! handed to the materialization callbacks of its kind of the converter recorded with it, most
//! recently added first, until one answers:
//!
//! - a source callback ([TypeConverter::add_source_materialization]) is given the old type and the
//!   values, and builds one value of that type;
//! - a target callback ([TypeConverter::add_target_materialization]) is given the value and the new
//!   types, and builds values of those types.
//!
//! The cast is replaced by what the callback built.
//! A cast no callback realizes is an error in a full conversion; in a partial one it stays when the
//! target calls `builtin.unrealized_conversion_cast` legal. A cast in the IR that the driver did
//! not insert is never handed to a callback. With
//! [ConversionConfig::with_reconcile_unrealized_casts] off, nothing is folded or realized and the
//! casts the driver inserted stay.
//!
//! # Block arguments
//!
//! A [SignatureConversion] says what each argument of a block becomes: new arguments of given
//! types ([SignatureConversion::add_inputs]), existing values ([SignatureConversion::remap_input]),
//! or nothing; and which new arguments replace nothing ([SignatureConversion::add_new_inputs]).
//! [ConversionPatternRewriter::apply_signature_conversion] appends the new arguments, in the order
//! of those calls, erases the old ones, and replaces an old argument that still has users by a
//! source cast at the start of the block; the block keeps its identity and operations, and is
//! converted at most once per conversion. [ConversionPatternRewriter::convert_region_types] does
//! this for every block of a region under the pattern's converter. The terminators that branch to
//! a converted block are not converted with it: unless a pattern converts them, the result fails
//! verification, which names the edge.
//!
//! # Changes in place
//!
//! [ConversionPatternRewriter::modify_op_in_place] runs a closure on the operation and records it
//! as changed, so a pattern that changes only a property or a type succeeds. A result whose type
//! the closure changed is replaced, for its users, by a source cast back to its old type, placed
//! after the operation; the new type must be what the converter answers for the result. Only the
//! operation itself is watched: a change to a nested operation is neither cast nor legalized
//! again.
//!
//! # Full and partial conversion
//!
//! The entry point is the mode; both share one driver.
//!
//! - [apply_full_conversion]: every operation under the root must end up legal. An operation the
//!   target does not call legal ([ConversionTarget::legality], the unknown operation policy
//!   included) is converted, and one no pattern converts is an error.
//! - [apply_partial_conversion]: an operation a rule of the target calls illegal must be converted;
//!   one the target calls legal is left as it is; one no rule matches
//!   ([ConversionTarget::rule_legality]) and the policy does not call legal is offered to the
//!   patterns and stays as it is when none applies. The final check reports only operations a rule
//!   calls illegal.
//!
//! In both, the operations nested under a recursively legal operation are not visited, and a
//! conversion that succeeds leaves IR that verifies (unless
//! [ConversionConfig::with_verify_after_conversion] is off). [TypeConverter::is_legal_op] can serve
//! as a dynamic legality rule: an operation is legal when every type it uses converts to itself.
//!
//! # The rewrite limit
//!
//! The driver visits the operations under the root in the order it collected them (a pre-order
//! walk that does not enter a recursively legal operation). The pattern applications that follow
//! from one of them, to the operation itself, to what the patterns insert or change in place while
//! it is legalized, and to the operation again while it stays illegal, are counted, and the count
//! starts again at the next operation. More than
//! [ConversionConfig::with_max_rewrites_per_operation] (1,024 by default) is an error that names
//! the operation and the pattern: the limit stops a pattern that never ends, not a large program.
//! A pattern may not erase or replace the root of the conversion; the rewriter answers an error.
//!
//! # What a pattern may rely on
//!
//! A pattern may rely on: being offered only an operation the target does not call legal; its
//! operands being given as described above, under its own type converter; the casts it causes
//! being realized by its own converter's callbacks; the operations it creates being legalized
//! after it succeeds; and the patterns rooted at one operation being tried in order of the
//! estimated length of their path to legal operations, then benefit.
//!
//! A pattern may not rely on: the order in which operations are offered (it is a pre-order walk,
//! but the operations a pattern creates or changes in place are legalized at once, before the
//! operations that follow; a user redirected to a cast waits for its turn); the operations around it being
//! converted yet (an operand may be an old value read through a cast); a cast it sees in the IR
//! staying there (casts are folded and realized at the end); a failed attempt's changes being
//! undone; the operation it is given being offered only once; or a change it makes to a nested
//! operation being tracked.

mod diagnostics;
mod driver;
mod legalization_graph;
mod pattern;
mod pattern_set;
mod reconcile;
mod rewriter;
mod signature_conversion;
mod target;
mod type_converter;

pub use self::{
    diagnostics::*, driver::*, legalization_graph::*, pattern::*, pattern_set::*, reconcile::*,
    rewriter::*, signature_conversion::*, target::*, type_converter::*,
};
