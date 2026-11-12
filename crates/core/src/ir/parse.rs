#![allow(unused)]

macro_rules! span {
    ($source_id:expr, $at:expr) => {{
        let at = $at;
        span!($source_id, at, at)
    }};

    ($source_id:expr, $start:expr, $end:expr) => {
        SourceSpan::session(
            $source_id,
            crate::diagnostics::TextRange::new($start, $end)
                .expect("lexer produced an inverted range"),
        )
    };
}

macro_rules! spanned {
    ($source_id:expr, $at:expr, $value:expr) => {
        Span::new(span!($source_id, $at), $value)
    };

    ($source_id:expr, $start:expr, $end:expr, $value:expr) => {
        Span::new(span!($source_id, $start, $end), $value)
    };
}

mod asm_parser;
mod delimiter;
mod error;
mod from_str_radix;
pub mod lexer;
mod operation;
mod parser;
pub mod scanner;
#[cfg(test)]
mod tests;
mod token;
mod types;

use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::{
    num::NonZeroU8,
    ops::{Deref, DerefMut},
};

use self::{
    asm_parser::AsmParserState,
    lexer::{Lexed, Lexer, TokenStream},
    parser::DefaultParser,
    scanner::Scanner,
};
pub use self::{
    delimiter::Delimiter,
    error::ParserError,
    from_str_radix::FromStrRadix,
    parser::{IntegerLiteral, Parser, ParserExt},
    token::Token,
};
use super::{
    AsCallableSymbolRef, BlockArgumentRef, BlockId, BlockRef, Context, DiagnosticScope,
    GenericOperationBuilder, Ident, OpBuilder, OpRegistration, OpResultRef, Operation,
    OperationName, OperationRef, OperationState, RawEntityRef, RegionRef, SuccessorInfo, SymbolRef,
    Type, UnsafeIntrusiveEntityRef, ValueId, ValueRef, interner::Symbol,
    operation::ParseAssemblyFn,
};
use crate::{
    Attribute, AttributeRef, Builder, CallableOpInterface, CompactString, FxHashMap,
    NamedAttribute, ProgramPoint, SmallVec, ToCompactString,
    adt::{SmallOrdMap, smallmap::SmallMap},
    diagnostics::{
        ColumnNumber, LineNumber, Outcome, Report, SourceId, SourceProvider, SourceSpan, Span,
    },
    dialects::builtin::{
        WorldBuilder, WorldRef,
        attributes::{Location, LocationAttr},
    },
    formatter::DisplayValues,
    interner, smallvec,
};

/// The interface for parsing operations printed to IR assembly
pub trait OpParser {
    /// Parse the current operation into `state` using `parser`.
    ///
    /// The following may be assumed by the parser implementation:
    ///
    /// 1. The parser has already consumed all operation results and the operation name, and it is
    ///    up to this trait function to parse everything after that point.
    /// 2. The trailing location specifier and terminating `;` will be handled by the parser, and
    ///    do not need to be parsed by this function.
    ///
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult;
}

pub type ParseResult<T = ()> = Result<T, ParserError>;

/// This struct contains configuration for the IR assembly parser.
pub struct ParserConfig {
    /// The context in which IR entities should be constructed
    pub context: Rc<Context>,
    /// Set to true if the parser should verify after parsing
    pub verify: bool,
}

impl ParserConfig {
    pub fn new(context: Rc<Context>) -> Self {
        Self {
            context,
            verify: true,
        }
    }

    /// Set the flag that determines whether verification will be run after parsing
    pub fn verify_after_parse(mut self, yes: bool) -> Self {
        self.verify = yes;
        self
    }

    #[inline]
    pub const fn should_verify_after_parse(&self) -> bool {
        self.verify
    }
}

/// This struct contains the state of the parser relevant to high-level parser implementations
pub struct ParserState<'input> {
    pub config: ParserConfig,
    pub token_stream: TokenStream<'input>,
    /// The current state for symbol parsing
    pub symbols: SymbolState,
    /// Optional high-level parser state to be populated during parsing
    pub asm_state: Option<Box<AsmParserState>>,
    /// Contains the stack of default dialects to use when parsing regions.
    ///
    /// A new dialect gets pushed to the stack before parsing regions nested under an operation
    /// implementing `OpAsmOpInterface`, and popped when done. At the top-level we start with
    /// "builtin" as the default, so that the top-level builtin operations parse as-is.
    pub default_dialect_stack: SmallVec<[interner::Symbol; 1]>,
    /// The number of types enclosing the one being parsed, which the type parser limits.
    type_nesting: u32,
}

impl<'input> ParserState<'input> {
    pub fn new(config: ParserConfig, token_stream: TokenStream<'input>) -> Self {
        Self {
            config,
            token_stream,
            symbols: Default::default(),
            asm_state: None,
            default_dialect_stack: smallvec![interner::Symbol::intern("builtin")],
            type_nesting: 0,
        }
    }

    #[inline]
    pub fn context(&self) -> &Context {
        &self.config.context
    }

    #[inline]
    pub fn context_rc(&self) -> Rc<Context> {
        self.config.context.clone()
    }
}

/// This struct records all parsed top-level symbols
#[derive(Default)]
pub struct SymbolState {
    /// A map from attribute alias identifier to Attribute
    pub attribute_alias_definitions: FxHashMap<interner::Symbol, Span<AttributeRef>>,
    /// A map from type alias identifier to Type
    pub type_alias_definitions: FxHashMap<interner::Symbol, Span<Type>>,
    /// The aliases of the `rec` block being parsed, if one is.
    pub(crate) rec_scope: Option<RecScope>,
}

/// The aliases of a `rec` block while it is parsed.
///
/// An alias that is not defined outside the block names a member of the block, whether or not it
/// has been declared yet. While the block is parsed it stands for a recursion variable whose index
/// is the alias's position in order of first mention; the variables are renumbered to the order of
/// declaration, which is the order of the members, when the block ends.
#[derive(Default)]
pub(crate) struct RecScope {
    /// The aliases of the block, in order of first mention.
    aliases: Vec<RecScopeAlias>,
    /// The position of each alias in `aliases`.
    positions: FxHashMap<interner::Symbol, u32>,
    /// The positions of the declared aliases, in order of declaration.
    declared: Vec<u32>,
}

/// An alias of a `rec` block.
pub(crate) struct RecScopeAlias {
    pub id: interner::Symbol,
    /// Where the alias was first mentioned.
    pub first_use: SourceSpan,
    /// The span and body of its declaration, once seen.
    pub declaration: Option<Span<Type>>,
}

impl RecScope {
    /// The index of the recursion variable that `id` stands for, which is mentioned at `span`.
    pub fn mention(&mut self, id: interner::Symbol, span: SourceSpan) -> u32 {
        if let Some(position) = self.positions.get(&id) {
            return *position;
        }
        let position = u32::try_from(self.aliases.len()).expect("fewer than 2^32 aliases");
        self.aliases.push(RecScopeAlias {
            id,
            first_use: span,
            declaration: None,
        });
        self.positions.insert(id, position);
        position
    }

    /// Whether `id` has been declared in the block.
    pub fn is_declared(&self, id: interner::Symbol) -> bool {
        self.positions
            .get(&id)
            .is_some_and(|position| self.aliases[*position as usize].declaration.is_some())
    }

    /// Records the declaration of `id` at `span` as `body`; `id` must not be declared yet.
    pub fn declare(&mut self, id: interner::Symbol, span: SourceSpan, body: Type) {
        let position = self.mention(id, span);
        let alias = &mut self.aliases[position as usize];
        debug_assert!(alias.declaration.is_none(), "`!{id}` is declared twice");
        alias.declaration = Some(Span::new(span, body));
        self.declared.push(position);
    }

    /// The aliases in order of first mention.
    pub fn aliases(&self) -> &[RecScopeAlias] {
        &self.aliases
    }

    /// The positions of the declared aliases, in order of declaration.
    pub fn declared(&self) -> &[u32] {
        &self.declared
    }
}

/// Parse IR assembly in the generic format from `source`, registered with the context under `name`
pub fn parse_generic(
    config: ParserConfig,
    name: impl Into<String>,
    source: impl Into<String>,
) -> Outcome<WorldRef> {
    match config.context.add_source(name, source) {
        Ok(id) => parse_source_generic(config, id),
        Err(err) => Outcome::from_report(Report::from_error(err)),
    }
}

/// Parse IR assembly in the generic format from the file at `path`
#[cfg(feature = "std")]
pub fn parse_file_generic(
    config: ParserConfig,
    path: impl AsRef<std::path::Path>,
) -> Outcome<WorldRef> {
    match config.context.load_source_file(path) {
        Ok(id) => parse_source_generic(config, id),
        Err(err) => Outcome::from_report(Report::from_error(err)),
    }
}

/// This parses the given source and appends parsed operations to a new
/// [World](crate::dialects::builtin::World).
///
/// If parsing is successful, the populated [World](crate::dialects::builtin::World) is returned.
/// Otherwise, the error that caused parsing to fail is recorded in the outcome's diagnostics.
///
/// **This wraps unconditionally, unlike [`parse_anchored_source`]**, which returns a parsed
/// `builtin.world` as the root rather than nesting it in the world it created to parse into. A
/// generic-format source that is itself a world therefore still yields a world within a world,
/// which is invalid IR — see [`parse_anchored_source`] for why. Left as it is because these entry
/// points have no callers outside this module's tests, and the one test covering them is
/// `#[ignore]`d; revive the generic format and this needs fixing first.
fn parse_source_generic(config: ParserConfig, source_id: SourceId) -> Outcome<WorldRef> {
    let context = config.context.clone();
    let scope = context.diagnostic_scope();
    let sources = context.sources();
    let Some(source) = sources.get(source_id).and_then(|source| source.text) else {
        return parse_outcome(
            scope,
            Err(Report::msg("source is not registered with this context")),
        );
    };
    let scanner = Scanner::new(source);
    let token_stream = TokenStream::new(source_id, scanner);
    let mut parser = DefaultParser::new(ParserState::new(config, token_stream));
    parser.state_mut().asm_state = Some(Default::default());

    let span = parser.current_location();
    let operation_parser = operation::TopLevelOperationParser::new(parser);
    let result = operation_parser.parse(span).map_err(Report::from);
    parse_outcome(scope, result)
}

/// End the parse's diagnostic `scope` and build the [Outcome] of the parse from what it collected
/// and `result`, attaching the context's sources as they are *now*.
///
/// The snapshot taken to read the input is not enough: resolving `loc("file":line:col)` loads
/// `file` into the context mid-parse, and diagnostics labelled with that span need it to render.
fn parse_outcome<T>(scope: DiagnosticScope<'_>, result: Result<T, Report>) -> Outcome<T> {
    let context = scope.context();
    let mut diagnostics = scope.finish();
    let result = result.map_err(|report| {
        diagnostics.add_report(report);
    });
    Outcome {
        result,
        diagnostics: diagnostics.finish().attach_session_sources(context.sources()),
    }
}

/// Parse one type from `source`, registered with `context` under `name`.
///
/// The whole of `source` must be the type: anything after it is an error.
pub fn parse_type(
    context: &Rc<Context>,
    name: impl Into<String>,
    source: impl Into<String>,
) -> Outcome<Type> {
    match context.add_source(name, source) {
        Ok(id) => parse_type_source(context, id),
        Err(err) => Outcome::from_report(Report::from_error(err)),
    }
}

/// Parse the single type that is the whole of the source `source_id`.
fn parse_type_source(context: &Rc<Context>, source_id: SourceId) -> Outcome<Type> {
    let scope = context.diagnostic_scope();
    let sources = context.sources();
    let Some(source) = sources.get(source_id).and_then(|source| source.text) else {
        return parse_outcome(
            scope,
            Err(Report::msg("source is not registered with this context")),
        );
    };
    let scanner = Scanner::new(source);
    let token_stream = TokenStream::new(source_id, scanner);
    let config = ParserConfig::new(context.clone());
    let mut parser = DefaultParser::new(ParserState::new(config, token_stream));
    let result = parser.parse_type().and_then(|ty| match parser.token_stream_mut().next()? {
        None => Ok(ty.into_inner()),
        Some(token) if matches!(token.inner(), Token::Eof) => Ok(ty.into_inner()),
        Some(token) => Err(ParserError::UnexpectedToken {
            span: token.span(),
            token: token.to_string(),
            expected: Some("end of input".into()),
        }),
    });
    parse_outcome(scope, result.map_err(Report::from))
}

/// Parse an operation of type `T` from `source`, registered with the context under `name`
pub fn parse<T: OpParser + OpRegistration>(
    config: ParserConfig,
    name: impl Into<String>,
    source: impl Into<String>,
) -> Outcome<UnsafeIntrusiveEntityRef<T>> {
    let anchor = config.context.get_registered_name::<T>();
    parse_anchored(anchor, config, name, source).map(downcast_anchored::<T>)
}

/// Parse an operation of type `T` from the file at `path`
#[cfg(feature = "std")]
pub fn parse_file<T: OpParser + OpRegistration>(
    config: ParserConfig,
    path: impl AsRef<std::path::Path>,
) -> Outcome<UnsafeIntrusiveEntityRef<T>> {
    let anchor = config.context.get_registered_name::<T>();
    parse_file_anchored(anchor, config, path).map(downcast_anchored::<T>)
}

fn downcast_anchored<T: OpRegistration>(op: OperationRef) -> UnsafeIntrusiveEntityRef<T> {
    match op.try_downcast_op::<T>() {
        Ok(op) => op,
        Err(_) => unreachable!("anchored parser returned the wrong operation type"),
    }
}

/// Parse any operation from `source`, registered with the context under `name`.
pub fn parse_any(
    config: ParserConfig,
    name: impl Into<String>,
    source: impl Into<String>,
) -> Outcome<OperationRef> {
    match config.context.add_source(name, source) {
        Ok(id) => parse_anchored_source(None, config, id),
        Err(err) => Outcome::from_report(Report::from_error(err)),
    }
}

/// Parse IR assembly anchored at an operation `anchor`, from `source` registered with the
/// context under `name`.
pub fn parse_anchored(
    anchor: OperationName,
    config: ParserConfig,
    name: impl Into<String>,
    source: impl Into<String>,
) -> Outcome<OperationRef> {
    match config.context.add_source(name, source) {
        Ok(id) => parse_anchored_source(Some(anchor), config, id),
        Err(err) => Outcome::from_report(Report::from_error(err)),
    }
}

/// Parse any operation from the file at `path`.
#[cfg(feature = "std")]
pub fn parse_file_any(
    config: ParserConfig,
    path: impl AsRef<std::path::Path>,
) -> Outcome<OperationRef> {
    match config.context.load_source_file(path) {
        Ok(id) => parse_anchored_source(None, config, id),
        Err(err) => Outcome::from_report(Report::from_error(err)),
    }
}

/// Parse IR assembly anchored at an operation `anchor`, from the file at `path`.
#[cfg(feature = "std")]
pub fn parse_file_anchored(
    anchor: OperationName,
    config: ParserConfig,
    path: impl AsRef<std::path::Path>,
) -> Outcome<OperationRef> {
    match config.context.load_source_file(path) {
        Ok(id) => parse_anchored_source(Some(anchor), config, id),
        Err(err) => Outcome::from_report(Report::from_error(err)),
    }
}

/// Parse the single top-level operation in the source `source_id`, and return it.
///
/// The operation may be preceded by type declarations (`type !a = ..;` and `rec { .. }`), which is
/// how the printer writes the recursion groups a root operation refers to.
///
/// Symbol resolution is rooted at a [World](crate::dialects::builtin::World), so parsing needs one
/// to anchor the operation it is reading — `OperationParser` takes a
/// [World](crate::dialects::builtin::World) and parses into its body. For a source whose top-level
/// operation is a module, a component or anything else, that world is a genuine addition: the
/// operation had none, and callers rely on getting one (see `runic_compile`'s
/// `extract_miden_component_or_bail`).
///
/// For a source that *is* a `builtin.world`, it is not. Leaving the parsed world nested inside
/// the anchor produces a world within a world, which is not merely redundant — it is invalid
/// IR. A world is an **anonymous** symbol table, and [`crate::Symbol::path`] asserts that an
/// anonymous symbol table has no parent, so the first `path()` on any symbol inside the inner
/// world panics rather than returning a wrong answer.
///
/// So when the parsed operation is itself a [World](crate::dialects::builtin::World), it is
/// detached from the anchor and returned as the root. The now-empty anchor is left to the context's
/// arena, which is how every other discarded entity is disposed of.
///
/// # Why finalization and verification run after the detach
///
/// Both symbol-use linking and verification need the final operation hierarchy. While the
/// parsed world is still nested under the temporary anchor, an absolute path like
/// `::@ns:pkg@1.0.0::@m::@tbl` resolves against the anchor and cannot find its target. Linking
/// would leave the use unresolved, and a verifier such as `hir.exec_indirect`'s would reject IR
/// that is valid once detached.
///
/// After establishing the root, this function passes it to
/// [`operation::OperationParser::finalize_without_verifying`]. Finalization then resolves deferred
/// locations and links symbol uses using that root, so its location walk still reaches the parsed
/// IR after detachment. This function then verifies the root if the parser configuration requests
/// it. Verification failures are reported as [`ParserError::Report`], exactly as when `finalize`
/// verified.
///
/// **Note that [`parse_source_generic`] does none of this.** The generic-format entry points —
/// [`parse_generic`] and [`parse_file_generic`] — go through `TopLevelOperationParser` instead,
/// return the anchor world itself rather than what was parsed into it, and would still nest a
/// world inside a world. That is deliberate rather than overlooked: those functions have no
/// callers outside this module's tests, and the one test covering them is `#[ignore]`d. If the
/// generic format is ever revived, it needs the same treatment.
fn parse_anchored_source(
    anchor_name: Option<OperationName>,
    config: ParserConfig,
    source_id: SourceId,
) -> Outcome<OperationRef> {
    use crate::{BuilderExt, dialects::builtin::World};

    let context = config.context.clone();
    let scope = context.diagnostic_scope();
    let sources = context.sources();
    let Some(source) = sources.get(source_id).and_then(|source| source.text) else {
        return parse_outcome(
            scope,
            Err(Report::msg("source is not registered with this context")),
        );
    };
    let scanner = Scanner::new(source);
    let token_stream = TokenStream::new(source_id, scanner);
    let should_verify = config.should_verify_after_parse();
    let mut parser = DefaultParser::new(ParserState::new(config, token_stream));
    let span = parser.current_location();
    let world = match parser.builder_mut().create::<World, ()>(span)() {
        Ok(world) => world,
        Err(report) => return parse_outcome(scope, Err(report)),
    };
    let anchor = world.as_operation_ref();
    parser.state_mut().asm_state = Some(Box::<AsmParserState>::default());
    let mut operation_parser = operation::OperationParser::new(parser, world);

    let result = (|| -> Result<OperationRef, Report> {
        operation::parse_type_declarations(&mut operation_parser).map_err(Report::from)?;
        let op = operation_parser.parse_operation().map_err(Report::from)?;
        // See the note above: a parsed world is the root, not something to be rooted.
        //
        // The mutable borrow held across `remove` is safe *because* the operation is a `World`, and
        // that is not incidental. `remove` reaches `EntityListItem::on_removed`
        // (`ir/operation.rs:234`), which re-borrows the operation immutably to unregister it from
        // its parent's symbol table — but only inside `if this.name().implements::<dyn Symbol>()`.
        // A `World` implements `SymbolTable` and not `Symbol`, so that branch is skipped and no
        // second borrow is taken. If `World` ever gains `Symbol`, this line becomes a runtime
        // aliasing panic with no compile error to warn you, and it would need to drop the borrow
        // first.
        let root = if op.borrow().is::<World>() {
            let mut parsed_world = op;
            parsed_world.borrow_mut().remove();
            op
        } else {
            // The parsed operation is still rooted at the anchor, and the anchor is what a verifier
            // resolving an absolute path will see, so it is the anchor that must be verified.
            anchor
        };

        // Symbol uses must be linked after normalizing the root. Absolute paths inside an explicit
        // world cannot resolve while that world is still nested under the temporary anchor.
        operation_parser
            .finalize_without_verifying(root.try_downcast_op::<World>().expect("root is a world"))
            .map_err(Report::from)?;

        // Verification is the parser's, so its failures must look like the parser's: the same
        // `ParserError::Report` wrapper that `finalize` produced when it ran this pass itself.
        if should_verify {
            root.borrow()
                .recursively_verify()
                .map_err(|err| Report::from(ParserError::from(err)))?;
        }

        match anchor_name {
            None => Ok(op),
            Some(anchor) => {
                let operation = op.borrow();
                if operation.name() != anchor {
                    return Err(Report::msg(format!(
                        "expected operation '{anchor}', got '{}'",
                        operation.name()
                    )));
                }

                Ok(op)
            }
        }
    })();

    parse_outcome(scope, result)
}

/// This trait is implemented by parsers which can parse operations from IR assembly, and provides
/// higher-level parser functions for that purpose.
pub trait OpAsmParser<'input>: Parser<'input> {
    /// Parse a `loc(...)` specifier if present.
    ///
    /// Location for BlockArgument and Operation may be deferred with an alias, in
    /// which case an OpaqueLoc is set and will be resolved when parsing
    /// completes.
    fn parse_optional_location_specifier(&mut self) -> ParseResult<Option<Location>>;

    /// Return the name of the specified result in the specified syntax, as well
    /// as the sub-element in the name.  It returns an empty string and ~0U for
    /// invalid result numbers.  For example, in this operation:
    ///
    ///  %x, %y:2, %z = foo.op
    ///
    ///    getResultName(0) == {"x", 0 }
    ///    getResultName(1) == {"y", 0 }
    ///    getResultName(2) == {"y", 1 }
    ///    getResultName(3) == {"z", 0 }
    ///    getResultName(4) == {"", ~0U }
    fn get_result_name(&self, result_num: usize) -> Option<(interner::Symbol, usize)>;

    /// Returns the number of declared SSA results.
    fn get_num_results(&self) -> usize;

    /// Parse an operation in its generic form.
    ///
    /// The parsed operation is parsed in the current context, and inserted in the provided block
    /// and insertion point. The results produced by this operation aren't mapped to any named value
    /// in the parser.
    fn parse_generic_operation(&mut self, ip: Option<ProgramPoint>) -> ParseResult<OperationRef>;

    /// Parse the name of an operation, in the custom form.
    fn parse_custom_operation_name(&mut self) -> ParseResult<Span<OperationName>>;

    // /// Parse the name of an operation.
    //fn parse_operation_name(&mut self) -> ParseResult<OperationName>;

    /// Parse different components, e.g. operand, successors, regions, attribute and function
    /// signature - of the generic form of an operation instance and populate the input
    /// [OperationState] 'result' with those components.
    ///
    /// If any of the components is explicitly provided, then skip parsing that component.
    fn parse_generic_operation_after_name(
        &mut self,
        state: &mut OperationState,
        operands: Option<&[UnresolvedOperand]>,
        successors: Option<&[BlockRef]>,
        regions: Option<&[RegionRef]>,
        attrs: Option<ParsedAttrs>,
        signature: Option<Type>,
    ) -> ParseResult;

    /// Parse a single SSA value operand name.
    fn parse_operand(&mut self, allow_result_number: bool) -> ParseResult<UnresolvedOperand>;

    /// Parse a single SSA value operand name, if present.
    fn parse_optional_operand(
        &mut self,
        allow_result_number: bool,
    ) -> ParseResult<Option<UnresolvedOperand>>;

    /// Parse zero or more comma-separated SSA operand references with a specified delimiter, and
    /// an optional required operand count.
    fn parse_operand_list(
        &mut self,
        result: &mut SmallVec<[UnresolvedOperand; 2]>,
        delimiter: Delimiter,
        allow_result_number: bool,
        required_operand_count: Option<NonZeroU8>,
    ) -> ParseResult {
        let start = self.current_location();
        let base_len = result.len();
        self.parse_comma_separated_list(delimiter, Some("operand list"), |parser| {
            let operand = parser.parse_operand(allow_result_number)?;
            result.push(operand);

            // Stop after the required number of operands, leaving the trailing comma and any
            // remaining items to be consumed by a following (e.g. variadic) operand group.
            Ok(required_operand_count
                .is_none_or(|required| result.len() - base_len < required.get() as usize))
        })?;

        if let Some(required) = required_operand_count
            && result.len() - base_len != required.get() as usize
        {
            return Err(ParserError::InvalidOperandCount {
                span: start,
                expected: required.get() as usize,
                actual: result.len() - base_len,
            });
        }

        Ok(())
    }

    /// Parse zero or more trailing SSA comma-separated trailing operand references with a specified
    /// surrounding delimiter, and an optional required operand count.
    ///
    /// A leading comma is expected before the operands.
    fn parse_trailing_operand_list(
        &mut self,
        result: &mut SmallVec<[UnresolvedOperand; 2]>,
        delimiter: Delimiter,
    ) -> ParseResult {
        if self.parse_optional_comma()? {
            self.parse_operand_list(result, delimiter, true, None)
        } else {
            Ok(())
        }
    }

    /// Parse a comma-separated list of SSA operand references without a surrounding delimiter,
    /// if one is present.
    ///
    /// Without a delimiter, [Self::parse_operand_list] requires at least one operand; this method
    /// consumes nothing when there is none.
    fn parse_optional_operand_list(
        &mut self,
        result: &mut SmallVec<[UnresolvedOperand; 2]>,
    ) -> ParseResult {
        let Some(first) = self.parse_optional_operand(/* allow_result_number= */ true)? else {
            return Ok(());
        };
        result.push(first);
        self.parse_trailing_operand_list(result, Delimiter::None)
    }

    /// Resolve an operand to an SSA value, emitting an error on failure.
    fn resolve_operand(&mut self, operand: UnresolvedOperand, ty: Type) -> ParseResult<ValueRef>;

    /// Resolve an operand to an SSA value, emitting an error on failure.
    fn resolve_operands_of_uniform_type(
        &mut self,
        operands: &[UnresolvedOperand],
        ty: Type,
        result: &mut SmallVec<[ValueRef; 2]>,
    ) -> ParseResult {
        for operand in operands.iter().copied() {
            result.push(self.resolve_operand(operand, ty)?);
        }
        Ok(())
    }

    /// Resolve an operand to an SSA value, emitting an error on failure.
    fn resolve_operands(
        &mut self,
        span: SourceSpan,
        operands: &[UnresolvedOperand],
        tys: &[Type],
        result: &mut SmallVec<[ValueRef; 2]>,
    ) -> ParseResult {
        if operands.len() != tys.len() {
            return Err(ParserError::OperandAndTypeListMismatch {
                span,
                num_operands: operands.len(),
                num_types: tys.len(),
            });
        }
        for (operand, ty) in operands.iter().copied().zip(tys.iter().copied()) {
            result.push(self.resolve_operand(operand, ty)?);
        }
        Ok(())
    }

    /// Parse a single argument with the following syntax:
    ///
    ///   `%ssaName : !type { optionalAttrDict} loc(optionalSourceLoc)`
    ///
    /// If `allow_type` is false or `allow_attrs` are false then the respective parts of the grammar
    /// are not parsed.
    fn parse_argument(&mut self, allow_type: bool, allow_attrs: bool) -> ParseResult<Argument>;

    /// Parse a single argument, if present, with the following syntax:
    ///
    ///   `%ssaName : !type { optionalAttrDict} loc(optionalSourceLoc)`
    ///
    /// If `allow_type` is false or `allow_attrs` are false then the respective parts of the grammar
    /// are not parsed.
    fn parse_optional_argument(
        &mut self,
        allow_type: bool,
        allow_attrs: bool,
    ) -> ParseResult<Option<Argument>>;

    /// Parse zero or more arguments with a specified surrounding delimiter.
    fn parse_argument_list(
        &mut self,
        delimiter: Delimiter,
        allow_type: bool,
        allow_attrs: bool,
        result: &mut SmallVec<[Argument; 4]>,
    ) -> ParseResult {
        self.parse_comma_separated_list(delimiter, Some("argument list"), |parser| {
            let arg = parser.parse_optional_argument(allow_type, allow_attrs)?;
            if let Some(arg) = arg {
                result.push(arg);
                Ok(true)
            } else {
                Ok(false)
            }
        })
    }

    /// Parses a region. Any parsed blocks are appended to 'region' and must be moved to the op
    /// regions after the op is created. The first block of the region takes 'arguments'.
    ///
    /// If 'enable_name_shadowing' is set to true, the argument names are allowed to shadow the
    /// names of other existing SSA values defined above the region scope. 'enable_name_shadowing'
    /// can only be set to true for regions attached to operations that are 'IsolatedFromAbove'.
    fn parse_region(
        &mut self,
        region: RegionRef,
        arguments: &[Argument],
        enable_name_shadowing: bool,
    ) -> ParseResult;

    /// Parses a region if present.
    fn parse_optional_region(
        &mut self,
        arguments: &[Argument],
        enable_name_shadowing: bool,
    ) -> ParseResult<Option<RegionRef>>;

    /// Parses a region preceded by the given keyword token, if the keyword is the next token
    fn parse_optional_region_with_token(
        &mut self,
        keyword: &str,
        arguments: &[Argument],
        enable_name_shadowing: bool,
    ) -> ParseResult<Option<RegionRef>> {
        if self.token_stream_mut().next_if_eq(Token::BareIdent(keyword))? {
            let region = self.context().create_region();
            self.parse_region(region, arguments, enable_name_shadowing)?;
            Ok(Some(region))
        } else {
            Ok(None)
        }
    }

    /// Parse a single operation successor.
    fn parse_successor(&mut self) -> ParseResult<Span<BlockRef>>;

    /// Parse an optional operation successor.
    fn parse_optional_successor(&mut self) -> ParseResult<Option<Span<BlockRef>>>;

    /// Parse a single operation successor and its operand list.
    fn parse_successor_and_use_list(
        &mut self,
        operands: &mut SmallVec<[ValueRef; 2]>,
    ) -> ParseResult<Span<BlockRef>>;
}

/// This trait is implemented by dialects which hook into IR assembly parsing in specific ways.
///
/// NOTE: This is currently an empty trait until we add support for dialect parsing hooks.
pub trait OpAsmDialectInterface: crate::Dialect {}

/// This trait is implemented by operations which hook into IR assembly parsing in specific ways.
pub trait OpAsmOpInterface: crate::Op {
    /// Get the special names to use when printing the results of this operation.
    ///
    /// Each result value in the returned vector starts a result "pack" starting at that result,
    /// giving the name to that pack. To signal that a result pack should use the default naming
    /// scheme, `None` can be provided instead of a name.
    ///
    /// For example, if you have an operation that has four results and you want to split these into
    /// three distinct groups you could do the following:
    ///
    /// ```text
    /// let results = self.results().all();
    /// smallvec::smallvec![
    ///     (results[0], Some("first_result".into()),
    ///     (results[1], Some("middle_results".into()),
    ///     (results[3], None /* use the default numbering */)
    /// ]
    /// ```
    ///
    /// This would print the operation as follows:
    ///
    /// ```hir
    /// %first_result, %middle_results:2, %0 = "my.op" ...
    /// ```
    fn get_asm_result_names(&self) -> SmallVec<[(OpResultRef, Option<CompactString>); 2]> {
        SmallVec::new_const()
    }
    /// Get the names to use when printing the block arguments for a region immediately nested
    /// under this operation.
    ///
    /// If no entry is present in the returned map for a given argument, then the default is used
    ///
    /// The default implementation returns an empty map
    fn get_asm_block_argument_names(
        &self,
        region: RegionRef,
    ) -> SmallOrdMap<BlockArgumentRef, CompactString, 4> {
        SmallOrdMap::new()
    }
    /// Get the names to use for the each block inside a region attached to this operation.
    ///
    /// If no entry is present in the returned map for a given block, then the default name is used
    ///
    /// The default implementation returns an empty map
    fn get_asm_block_names(&self) -> SmallOrdMap<BlockRef, CompactString, 1> {
        SmallOrdMap::new()
    }
    /// Return the default dialect used when printing/parsing operations in regions nested under
    /// this operation.
    ///
    /// This allows for eliding the dialect prefix from the operation name, for example it would be
    /// possible to omit the `scf.` prefix from all operations within a `scf.while`  if this method
    /// returned `scf`.
    ///
    /// The default implementation returns `None`.
    fn get_default_dialect(&self) -> Option<interner::Symbol> {
        None
    }
}

/// Represents a value parsed from an argument list, with an associated type and attributes.
///
/// This is a richer version of [UnresolvedOperand], as most operands cannot have associated
/// attributes.
#[derive(Debug)]
pub struct Argument {
    pub name: UnresolvedOperand,
    pub ty: Type,
    pub attrs: ParsedAttrs,
    pub loc: Location,
}

impl Argument {
    pub fn has_attribute(&self, name: impl Into<interner::Symbol>) -> bool {
        let name = name.into();
        self.attrs.iter().any(|attr| attr.name == name)
    }
}

/// Represents a value parsed from IR assembly that has not been resolved to a concrete [ValueRef]
/// yet.
#[derive(Copy, Clone)]
pub struct UnresolvedOperand {
    pub loc: SourceSpan,
    pub name: ValueId,
}

impl core::fmt::Debug for UnresolvedOperand {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UnresolvedOperand")
            .field_with("name", |f| write!(f, "{}", self.name))
            .finish()
    }
}

/// A type alias for a vector of [NamedAttribute]
pub type ParsedAttrs = SmallVec<[NamedAttribute; 1]>;
