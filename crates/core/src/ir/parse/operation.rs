use compact_str::format_compact;

use super::*;
use crate::{
    BlockArgument, Builder, EntityWithId, Forward, FunctionType, FxHashSet, InsertionGuard, Op,
    PendingSuccessorInfo, RawWalk, SuccessorGroupShape, Value, WalkResult,
    adt::{SmallDenseMap, SmallSet},
    dialects::builtin::{
        WorldRef,
        attributes::{Location, LocationAttr},
    },
    traits::{IsolatedFromAbove, Terminator},
};

/// This class provides support for parsing operations and regions of operations.
pub struct OperationParser<P> {
    parser: P,
    /// The top level operation that holds all of the parsed operations.
    top_level: WorldRef,

    /// A list of isolated name scopes.
    isolated_name_scopes: SmallVec<[IsolatedSSANameScope; 2]>,

    /// This keeps track of the block names as well as the location of the first reference for each
    /// nested name scope.
    ///
    /// This is used to diagnose invalid block references and memorize them.
    blocks_by_name: SmallVec<[SmallDenseMap<BlockId, Span<BlockRef>>; 2]>,
    forward_ref: SmallVec<[SmallDenseMap<BlockRef, SourceSpan>; 2]>,

    /// These are all of the placeholders we've made along with the location of their first
    /// reference, to allow checking for use of undefined values.
    forward_ref_placeholders: SmallDenseMap<ValueRef, SourceSpan>,

    /// Operations that define the placeholders.
    ///
    /// These are kept until the end of of the lifetime of the parser because some custom parsers
    /// may store references to them in local state and use them after forward references
    /// have been resolved.
    forward_ref_ops: SmallSet<OperationRef, 2>,
}

impl<'input, P> Parser<'input> for OperationParser<P>
where
    P: Parser<'input>,
{
    #[inline(always)]
    fn builder(&self) -> &OpBuilder {
        self.parser.builder()
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut OpBuilder {
        self.parser.builder_mut()
    }

    #[inline(always)]
    fn state(&self) -> &ParserState<'input> {
        self.parser.state()
    }

    #[inline(always)]
    fn state_mut(&mut self) -> &mut ParserState<'input> {
        self.parser.state_mut()
    }

    #[inline(always)]
    fn token_stream(&self) -> &TokenStream<'input> {
        self.parser.token_stream()
    }

    #[inline(always)]
    fn token_stream_mut(&mut self) -> &mut TokenStream<'input> {
        self.parser.token_stream_mut()
    }

    #[inline]
    fn parse_extended_attribute(&mut self, ty: Type) -> ParseResult<Span<AttributeRef>> {
        super::parser::parse_extended_attribute(self, ty)
    }

    #[inline]
    fn parse_optional_type(&mut self) -> ParseResult<Option<Span<Type>>> {
        super::types::parse_optional_type(self)
    }
}

/// This type is used to keep track of things that are either an Operation or a BlockArgument.
///
/// We cannot use Value for this, because not all Operations have results.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OpOrArgument {
    Op(OperationRef),
    Arg(BlockArgumentRef),
}

impl OpOrArgument {
    pub fn span(&self) -> SourceSpan {
        use crate::diagnostics::Spanned;
        match self {
            Self::Op(op) => op.borrow().span,
            Self::Arg(arg) => arg.borrow().span(),
        }
    }

    pub fn set_span(self, span: SourceSpan) {
        match self {
            Self::Op(mut op) => op.borrow_mut().set_span(span),
            Self::Arg(mut arg) => arg.borrow_mut().set_span(span),
        }
    }
}

/// A successor of an operation in the generic form, as the text gives it.
struct GenericSuccessor {
    /// The destination, where the text names it
    block: Span<BlockRef>,
    /// The key the text gives the successor, where it is written
    key: Option<Span<AttributeRef>>,
    /// The index of the successor group it is in
    group: usize,
    /// The arguments forwarded to `block`
    operands: SmallVec<[ValueRef; 2]>,
}

#[derive(Debug, Copy, Clone)]
pub struct ResultRecord {
    pub loc: SourceSpan,
    pub id: interner::Symbol,
    /// The number of consecutive results bound to the name, at most [ValueId::MAX_NAMED_RESULTS]
    pub count: usize,
}

/// This struct represents an isolated SSA name scope.
///
/// This scope may contain other nested non-isolated scopes. These scopes are used for operations
/// that are known to be isolated to allow for reusing names within their regions, even if those
/// names are used above.
#[derive(Default)]
struct IsolatedSSANameScope {
    /// This keeps track of all of the SSA values we are tracking for each name scope, indexed by
    /// their name.
    ///
    /// This has one entry per result number.
    values: FxHashMap<ValueId, SmallVec<[Option<Span<ValueRef>>; 1]>>,
    /// This keeps track of all of the values defined by a specific name scope.
    definitions_per_scope: SmallVec<[FxHashSet<ValueId>; 2]>,
}

impl IsolatedSSANameScope {
    /// Record that a definition was added at the current scope.
    pub fn record_definition(&mut self, def: ValueId) {
        self.definitions_per_scope.last_mut().unwrap().insert(def);
    }

    /// Push a nested name scope.
    pub fn push_ssa_name_scope(&mut self) {
        self.definitions_per_scope.push(Default::default());
    }

    /// Pop a nested name scope.
    pub fn pop_ssa_name_scope(&mut self) {
        for def in self.definitions_per_scope.pop().unwrap() {
            self.values.remove(&def);
        }
    }
}

impl<'input, P> OperationParser<P>
where
    P: Parser<'input>,
{
    pub fn new(mut parser: P, top_level: WorldRef) -> Self {
        // Ensure the world's body region is populated
        {
            let world_body = { top_level.borrow().body().as_region_ref() };
            if world_body.borrow().is_empty() {
                parser.builder_mut().create_block(world_body, None, &[]);
            } else {
                parser
                    .builder_mut()
                    .set_insertion_point_to_end(world_body.borrow().entry_block_ref().unwrap());
            }
        }

        let mut this = Self {
            parser,
            top_level,
            isolated_name_scopes: Default::default(),
            blocks_by_name: Default::default(),
            forward_ref: Default::default(),
            forward_ref_placeholders: Default::default(),
            forward_ref_ops: Default::default(),
        };

        // The top level operation starts a new name scope.
        this.push_ssa_name_scope(true);

        // If we are populating the parser state, prepare it for parsing.
        if let Some(state) = this.parser.state_mut().asm_state.as_deref_mut() {
            state.initialize();
        }

        this
    }

    /// After parsing is finished, this function must be called to see if there are any remaining
    /// issues.
    ///
    /// Verification runs as the last step, if [`ParserConfig::should_verify_after_parse`] is set.
    pub fn finalize(self) -> ParseResult {
        let verify = self.parser.state().config.should_verify_after_parse();
        self.finalize_impl(verify)
    }

    /// Like [`Self::finalize`], but never verifies, regardless of the parser configuration.
    ///
    /// The caller supplies the root after reparenting the parsed operation. The caller
    /// is responsible for verifying that root when [`ParserConfig::should_verify_after_parse`]
    /// is set; see `parse_anchored_source`.
    pub fn finalize_without_verifying(mut self, root: WorldRef) -> ParseResult {
        self.top_level = root;
        self.finalize_impl(false)
    }

    fn finalize_impl(mut self, verify: bool) -> ParseResult {
        // Check for any forward references that are left.  If we find any, error out.
        if !self.forward_ref_placeholders.is_empty() {
            let mut labels = Vec::with_capacity(self.forward_ref_placeholders.len());
            for (i, (_, span)) in self.forward_ref_placeholders.iter().enumerate() {
                labels.push(crate::diagnostics::OwnedLabel {
                    span: *span,
                    style: if i == 0 {
                        crate::diagnostics::LabelStyle::Primary
                    } else {
                        crate::diagnostics::LabelStyle::Context
                    },
                    message: None,
                });
            }
            return Err(ParserError::UndeclaredValueUses { labels });
        }

        // Pop the top level name scope.
        self.pop_ssa_name_scope()?;

        // Verify that the parsed operations are valid.
        if verify {
            self.top_level.borrow().as_operation().recursively_verify()?;
        }

        // If we are populating the parser state, finalize the top-level operation.
        if let Some(asm_state) = self.parser.state_mut().asm_state.as_deref_mut() {
            asm_state.finalize(self.top_level.as_operation_ref());
        }

        Ok(())
    }
}

/// SSA Value Handling
impl<'input, P> OperationParser<P>
where
    P: Parser<'input>,
{
    /// Push a new SSA name scope to the parser.
    fn push_ssa_name_scope(&mut self, isolated: bool) {
        self.blocks_by_name.push(Default::default());
        self.forward_ref.push(Default::default());

        // Push back a new name definition scope.
        if isolated {
            self.isolated_name_scopes.push(Default::default());
        }
        self.isolated_name_scopes.last_mut().unwrap().push_ssa_name_scope();
    }

    /// Pop the last SSA name scope from the parser.
    fn pop_ssa_name_scope(&mut self) -> ParseResult {
        let forward_ref_current_scope = self.forward_ref.pop().unwrap();

        // Verify that all referenced blocks were defined.
        if !forward_ref_current_scope.is_empty() {
            let mut labels = Vec::default();
            for (block, span) in forward_ref_current_scope {
                labels.push(crate::diagnostics::OwnedLabel {
                    span,
                    style: if labels.is_empty() {
                        crate::diagnostics::LabelStyle::Primary
                    } else {
                        crate::diagnostics::LabelStyle::Context
                    },
                    message: None,
                });
                // Add this block to the top-level region to allow for automatic cleanup.
                self.top_level.borrow_mut().body_mut().body_mut().push_back(block);
            }
            return Err(ParserError::UndefinedBlocks { labels });
        }

        // Pop the next nested namescope. If there is only one internal namescope, just pop the
        // isolated scope.
        let current_name_scope = self.isolated_name_scopes.last_mut().unwrap();
        if current_name_scope.definitions_per_scope.len() == 1 {
            self.isolated_name_scopes.pop();
        } else {
            current_name_scope.pop_ssa_name_scope();
        }

        self.blocks_by_name.pop();

        Ok(())
    }

    /// Register a definition of a value with the symbol table.
    fn add_definition(&mut self, use_info: UnresolvedOperand, value: ValueRef) -> ParseResult {
        let entries = self.get_ssa_value_entry(use_info.name);

        // Make sure there is a slot for this value.
        let result_index = use_info.name.result_index().unwrap_or(0) as usize;
        if entries.len() <= result_index {
            entries.resize(result_index + 1, None);
        }

        // If we already have an entry for this, check to see if it was a definition or a forward
        // reference.
        if let Some(mut existing) = entries[result_index] {
            if !self.forward_ref_placeholders.contains_key(existing.inner()) {
                // Named as written: `%p`, not `%p#0`, for a name bound to several results.
                return Err(ParserError::ValueRedefinition {
                    span: use_info.loc,
                    prev_span: existing.span(),
                    name: use_info.name.without_result_index(),
                });
            }

            if existing.borrow().ty() != value.borrow().ty() {
                return Err(ParserError::ValueDefinitionTypeMismatch {
                    span: use_info.loc,
                    prev_span: existing.span(),
                    ty: format_compact!("{}", value.borrow().ty()),
                    prev_ty: format_compact!("{}", existing.borrow().ty()),
                });
            }

            // If it was a forward reference, update everything that used it to use the actual
            // definition instead, delete the forward ref, and remove it from our set of forward
            // references we track.
            existing.borrow_mut().replace_all_uses_with(value);
            self.forward_ref_placeholders.remove(&existing);

            // If a definition of the value already exists, replace it in the assembly
            // state.
            if let Some(asm_state) = self.parser.state_mut().asm_state.as_deref_mut() {
                asm_state.refine_definition(existing.into_inner(), value);
            }
        }

        // Record this definition for the current scope. The scope records the name without its
        // result index, as `values` is keyed, so that popping the scope removes every result
        // bound to the name.
        let entries = self.get_ssa_value_entry(use_info.name);
        entries[result_index] = Some(Span::new(use_info.loc, value));
        self.record_definition(use_info.name.without_result_index());

        Ok(())
    }

    /// Parse an optional list of SSA uses into 'results'.
    ///
    ///   ssa-use-list ::= ssa-use (`,` ssa-use)*
    ///   ssa-use-list-opt ::= ssa-use-list?
    ///
    fn parse_optional_ssa_use_list<const N: usize>(
        &mut self,
        results: &mut SmallVec<[UnresolvedOperand; N]>,
    ) -> ParseResult {
        if !self
            .parser
            .token_stream_mut()
            .is_next(|tok| matches!(tok, Token::PercentIdent(_)))
        {
            return Ok(());
        }

        self.parse_comma_separated_list(Delimiter::None, Some("SSA use list"), |parser| {
            let result = parser.parse_ssa_use(/*allow_result_number*/ true)?;
            results.push(result);
            Ok(true)
        })
    }

    /// Parse a single SSA use into 'result'.
    ///
    /// If 'allow_result_number' is true then we allow `#42` syntax.
    ///
    ///   ssa-use ::= ssa-id
    ///
    fn parse_ssa_use(&mut self, allow_result_number: bool) -> ParseResult<UnresolvedOperand> {
        let (span, id) = self
            .parser
            .token_stream_mut()
            .expect_map("SSA value", |tok| match tok {
                Token::PercentIdent(id) => Some(ValueId::from_symbol(interner::Symbol::intern(id))),
                _ => None,
            })?
            .into_parts();

        // If we have an attribute ID, it is a result number.
        let result_num = self.parser.token_stream_mut().next_if_map(|tok| match tok {
            Token::HashIdent(num) => Some(num),
            _ => None,
        })?;
        if let Some(result_num) = result_num {
            let (result_num_span, result_num) = result_num.into_parts();
            if !allow_result_number {
                return Err(ParserError::ResultNumberUsedInArgumentList {
                    span: result_num_span,
                    value_span: span,
                });
            }
            let index =
                result_num.parse::<u8>().map_err(|err| ParserError::InvalidResultIndex {
                    span: result_num_span,
                    value_span: span,
                    reason: err.to_string(),
                })?;

            Ok(UnresolvedOperand {
                loc: span,
                name: id.with_result_index(index),
            })
        } else {
            Ok(UnresolvedOperand {
                loc: span,
                name: id,
            })
        }
    }

    /// Given a reference to an SSA value and its type, return a reference.
    ///
    /// This returns `None` on failure.
    fn resolve_ssa_use(&mut self, use_info: UnresolvedOperand, ty: Type) -> ParseResult<ValueRef> {
        let entries = self.get_ssa_value_entry(use_info.name);

        // If we have already seen a value of this name, return it.
        let result_index = use_info.name.result_index().unwrap_or(0) as usize;
        if result_index < entries.len()
            && let Some(value_ref) = entries[result_index]
        {
            // Check that the type matches the other uses.
            let (span, value_ref) = value_ref.into_parts();
            let value = value_ref.borrow();
            let prev_ty = value.ty();
            if prev_ty == ty || ty.is_unknown() {
                if let Some(asm_state) = self.parser.state_mut().asm_state.as_deref_mut() {
                    asm_state.add_uses(value_ref, &[use_info.loc]);
                }
                return Ok(value_ref);
            }

            return Err(ParserError::ValueUseTypeMismatch {
                span: use_info.loc,
                prev_span: span,
                ty: format_compact!("{ty}"),
                prev_ty: format_compact!("{prev_ty}"),
            });
        }

        // Make sure we have enough slots for this.
        if entries.len() <= result_index {
            entries.resize(result_index + 1, None);
        }

        // If the value has already been defined and this is an overly large result number,
        // diagnose that.
        if entries[0].is_some_and(|v| !self.is_forward_ref_placeholder(v.into_inner())) {
            return Err(ParserError::InvalidResultIndex {
                span: use_info.loc,
                value_span: SourceSpan::UNKNOWN,
                reason: format!("{} has index {result_index}", use_info.name),
            });
        }

        // Otherwise, this is a forward reference.
        //
        // Create a placeholder and remember that we did so.
        let result = self.create_forward_ref_placeholder(use_info.loc, ty);
        let entries = self.get_ssa_value_entry(use_info.name);
        entries[result_index] = Some(Span::new(use_info.loc, result));

        if let Some(asm_state) = self.parser.state_mut().asm_state.as_deref_mut() {
            asm_state.add_uses(result, &[use_info.loc]);
        }
        Ok(result)
    }

    /// Parse an SSA use with an associated type.
    ///
    ///   ssa-use-and-type ::= ssa-use `:` type
    fn parse_ssa_def_or_use_and_type<F>(&mut self, mut action: F) -> ParseResult
    where
        F: FnMut(&mut Self, UnresolvedOperand, Type) -> ParseResult,
    {
        let use_info = self.parse_ssa_use(true)?;
        let ty = self.parser.parse_colon_type()?;

        action(self, use_info, ty.into_inner())
    }

    /// Parse a (possibly empty) list of SSA operands, followed by a colon, then
    /// followed by a type list.
    ///
    ///   ssa-use-and-type-list ::= ssa-use-list ':' type-list-no-parens
    ///
    fn parse_optional_ssa_use_and_type_list<const N: usize>(
        &mut self,
        results: &mut SmallVec<[ValueRef; N]>,
    ) -> ParseResult {
        let mut value_ids = SmallVec::<[UnresolvedOperand; 4]>::new_const();
        self.parse_optional_ssa_use_list(&mut value_ids)?;

        // If there were no operands, then there is no colon or type lists.
        if value_ids.is_empty() {
            return Ok(());
        }

        let mut types = SmallVec::<[Type; 4]>::new_const();
        self.parser.token_stream_mut().expect(Token::Colon)?;
        self.parser.parse_type_list_no_parens(&mut types)?;

        if value_ids.len() != types.len() {
            let start = value_ids[0].loc;
            let end = self.parser.token_stream().current_position();
            return Err(ParserError::MismatchedValueAndTypeLists {
                span: span!(start.source().id(), start.range().start(), end),
                num_values: value_ids.len(),
                num_types: types.len(),
            });
        }

        results.reserve(value_ids.len());
        for (unresolved, ty) in value_ids.into_iter().zip(types) {
            results.push(self.resolve_ssa_use(unresolved, ty)?);
        }

        Ok(())
    }

    /// Return the location of the value identified by its name and number if it has been already
    /// referenced.
    fn get_reference_loc(&mut self, id: ValueId) -> Option<SourceSpan> {
        let values = &self.isolated_name_scopes.last().unwrap().values;
        let entry = values.get(&id.without_result_index())?;
        let result_index = id.result_index().unwrap_or(0) as usize;
        entry.get(result_index).and_then(|v| v.map(|v| v.span()))
    }

    /// Record that a definition was added at the current scope.
    fn record_definition(&mut self, id: ValueId) {
        self.isolated_name_scopes.last_mut().unwrap().record_definition(id);
    }

    /// Get the value entry for the given SSA name.
    fn get_ssa_value_entry(&mut self, id: ValueId) -> &mut SmallVec<[Option<Span<ValueRef>>; 1]> {
        self.isolated_name_scopes
            .last_mut()
            .unwrap()
            .values
            .entry(id.without_result_index())
            .or_default()
    }

    /// Create and remember a new placeholder for a forward reference.
    ///
    /// The placeholder is the one result, of type `ty`, of a `builtin.unrealized_conversion_cast`
    /// with no inputs, built outside any block: it is only something with a def/use chain. The
    /// text can write a cast of the same shape, so a placeholder is told apart by being in
    /// `forward_ref_placeholders` (its result) and `forward_ref_ops` (the operation), never by its
    /// shape. When the value is defined its uses move to the definition; a placeholder never
    /// defined is reported by [Self::finalize], and every placeholder is erased when the parser is
    /// dropped, so none survives a parse.
    fn create_forward_ref_placeholder(&mut self, loc: SourceSpan, ty: Type) -> ValueRef {
        use crate::dialects::builtin::BuiltinOpBuilder;

        let mut builder = crate::OpBuilder::new(self.parser.context_rc());
        let op_ref = builder
            .unrealized_conversion_cast([], [ty], loc)
            .unwrap_or_else(|err| unreachable!("a cast with one result is valid: {err}"));
        let result = op_ref.borrow().outputs()[0] as ValueRef;
        self.forward_ref_placeholders.insert(result, loc);
        self.forward_ref_ops.insert(op_ref.as_operation_ref());
        result
    }
}

/// Operation Parsing
impl<'input, P> OperationParser<P>
where
    P: Parser<'input>,
{
    /// Parse an operation.
    ///
    ///  operation         ::= op-result-list?
    ///                        (generic-operation | custom-operation)
    ///                        trailing-location?
    ///  generic-operation ::= string-literal `(` ssa-use-list? `)`
    ///                        successor-list? (`(` region-list `)`)?
    ///                        attribute-dict? `:` function-type
    ///  custom-operation  ::= bare-id custom-operation-format
    ///  op-result-list    ::= op-result (`,` op-result)* `=`
    ///  op-result         ::= ssa-id (`:` integer-literal)
    ///
    pub fn parse_operation(&mut self) -> ParseResult<OperationRef> {
        let start = self.parser.token_stream().current_position();

        let mut result_ids = SmallVec::<[ResultRecord; 1]>::new_const();
        let mut num_expected_results = 0usize;
        if self
            .parser
            .token_stream_mut()
            .is_next(|tok| matches!(tok, Token::PercentIdent(_)))
        {
            // Parse the group of result ids.
            self.parse_comma_separated_list(Delimiter::None, Some("ssa identifiers"), |parser| {
                // Parse the next result id.
                let (name_span, name) = parser
                    .token_stream_mut()
                    .expect_map("SSA identifier", |tok| match tok {
                        Token::PercentIdent(id) => Some(id),
                        _ => None,
                    })?
                    .into_parts();

                // If the next token is a ':', we parse the expected result count.
                let mut expected_sub_results = 1;
                let mut end = name_span.range().end();
                if parser.token_stream_mut().next_if_eq(Token::Colon)? {
                    // Check that the next token is an integer.
                    let (count_span, count_str) = parser
                        .token_stream_mut()
                        .expect_map("integer number of results", |tok| match tok {
                            Token::Int(n) => Some(n),
                            _ => None,
                        })?
                        .into_parts();
                    let count = count_str.parse::<usize>().map_err(|err| {
                        ParserError::InvalidIntegerLiteral {
                            span: count_span,
                            reason: err.to_string(),
                        }
                    })?;
                    // A shared name encodes the result index in the value id, which bounds it.
                    if count == 0 || count > ValueId::MAX_NAMED_RESULTS {
                        return Err(ParserError::InvalidResultCount {
                            span: count_span,
                            limit: ValueId::MAX_NAMED_RESULTS,
                        });
                    }
                    end = count_span.range().end();
                    expected_sub_results = count;
                }

                let span = span!(name_span.source().id(), name_span.range().start(), end);
                result_ids.push(ResultRecord {
                    loc: span,
                    id: interner::Symbol::intern(name),
                    count: expected_sub_results,
                });
                num_expected_results += expected_sub_results;
                Ok(true)
            })?;

            self.parser.token_stream_mut().expect(Token::Equal)?;
        }

        let source_id = self.source_id();
        let Some(name_token) = self
            .parser
            .token_stream_mut()
            .peek()?
            .map(|(start, tok, end)| spanned!(source_id, start, end, tok))
        else {
            return Err(ParserError::UnexpectedEof {
                expected: vec!["operation name".to_string()],
            });
        };

        let (name_span, name_token) = name_token.into_parts();
        let op = match name_token {
            Token::BareIdent(_) => self.parse_custom_operation(&result_ids)?,
            Token::String(_) => self.parse_generic_operation_internal(None, false)?,
            invalid => {
                return Err(ParserError::UnexpectedToken {
                    span: name_span,
                    token: invalid.to_string(),
                    expected: "an operation name".into(),
                });
            }
        };

        // If the operation had a name, register it.
        let end = self.current_location();
        if !result_ids.is_empty() {
            let op = op.borrow();
            match op.num_results() {
                0 => return Err(ParserError::NamedOpWithNoResults { span: name_span }),
                n if n != num_expected_results => {
                    return Err(ParserError::ResultCountMismatch {
                        span: name_span,
                        count: n,
                        expected: num_expected_results,
                    });
                }
                _ => (),
            }

            // Add this operation to the assembly state if it was provided to populate.
            if let Some(asm_state) = self.state_mut().asm_state.as_deref_mut() {
                let mut asm_result_groups = SmallVec::<[_; 4]>::new_const();
                asm_result_groups.reserve(result_ids.len());
                let mut result_index = 0usize;
                for record in result_ids.iter() {
                    asm_result_groups.push((result_index, record.loc));
                    result_index += record.count;
                }
                asm_state.finalize_operation_definition(
                    op.as_operation_ref(),
                    name_span,
                    end,
                    &asm_result_groups,
                );
            }

            // Bind each parsed result name to a result of the operation, in order. The printer
            // emits one SSA name per result (e.g. `%a, %b, %c = scf.if`) - the canonical form for
            // an op with a single result group - so the names map directly onto the flattened
            // result list; the `%name:count` spelling instead binds `count` consecutive results to
            // `%name#0..#count`. The previous code indexed `results().group(i)` by the i-th *name*,
            // which panicked once a single variadic group was bound by more than one name.
            let results = op.results().all();
            let mut result_offset = 0usize;
            for result_record in result_ids.iter() {
                for result_index in 0..result_record.count {
                    let name = ValueId::from_symbol(result_record.id);
                    let name = if result_record.count == 1 {
                        name
                    } else {
                        // Bounded by the check on `count` when the record was parsed
                        name.with_result_index(result_index as u8)
                    };
                    let use_info = UnresolvedOperand {
                        loc: result_record.loc,
                        name,
                    };
                    let value = results[result_offset] as ValueRef;
                    result_offset += 1;
                    self.add_definition(use_info, value)?;
                }
            }
        } else if let Some(asm_state) = self.state_mut().asm_state.as_deref_mut() {
            asm_state.finalize_operation_definition(op, name_span, end, &[])?;
        }

        Ok(op)
    }

    /// Parse a single operation successor.
    ///
    ///   successor ::= block-id
    ///
    pub fn parse_successor(&mut self) -> ParseResult<Span<BlockRef>> {
        // Verify branch is identifier and get the matching block.
        let (span, id) = self
            .parser
            .token_stream_mut()
            .expect_map("block name", |tok| match tok {
                Token::CaretIdent(id) => Some(BlockId::from_symbol(interner::Symbol::intern(id))),
                _ => None,
            })?
            .into_parts();

        Ok(self.get_block_named(id, span))
    }

    /// Parse the successor list of an operation in the generic form, appending each successor to
    /// `successors` with the index of its group.
    ///
    ///   successors      ::= `[` successor-group (`|` successor-group)* `]`
    ///   successor-group ::= (successor (`,` successor)*)?
    ///   successor       ::= (attribute-value `:`)? block-id successor-arguments?
    ///   successor-arguments ::= `(` value-use-list `:` type-list-no-parens `)`
    ///
    /// Returns the number of groups the list has. A list without a successor is an error.
    fn parse_generic_successors(
        &mut self,
        successors: &mut SmallVec<[GenericSuccessor; 2]>,
    ) -> ParseResult<usize> {
        let start = self.parser.current_location();
        self.parser.parse_lbracket()?;
        let mut group = 0usize;
        loop {
            if !self
                .token_stream_mut()
                .is_next(|tok| matches!(tok, Token::Pipe | Token::Rbracket))
            {
                loop {
                    let successor = self.parse_generic_successor(group)?;
                    successors.push(successor);
                    if !self.token_stream_mut().next_if_eq(Token::Comma)? {
                        break;
                    }
                }
            }
            if !self.token_stream_mut().next_if_eq(Token::Pipe)? {
                break;
            }
            group += 1;
        }
        self.parser.parse_rbracket()?;

        if successors.is_empty() {
            return Err(ParserError::InvalidEmptySuccessorList { span: start });
        }
        Ok(group + 1)
    }

    /// Parse one successor of the generic form, in the successor group `group`.
    fn parse_generic_successor(&mut self, group: usize) -> ParseResult<GenericSuccessor> {
        let key = if self.token_stream_mut().is_next(|tok| matches!(tok, Token::CaretIdent(_))) {
            None
        } else {
            let unknown = self.context().unknown_type();
            let key = self.parser.parse_attribute(unknown)?;
            self.parser.parse_colon()?;
            Some(key)
        };
        let block = self.parse_successor()?;
        let mut operands = SmallVec::new_const();
        if self.token_stream_mut().next_if_eq(Token::Lparen)? {
            self.parse_optional_ssa_use_and_type_list(&mut operands)?;
            self.parser.parse_rparen()?;
        }
        Ok(GenericSuccessor {
            block,
            key,
            group,
            operands,
        })
    }

    /// Check the successors of the generic form against the successor groups `name` declares:
    /// as many groups, the number of successors each declares, and a key of the kind a keyed
    /// group stores on each of its successors and on no other.
    ///
    /// `groups` is the number of groups the text gives, `None` when it gives no successor list.
    fn check_generic_successors(
        name: &OperationName,
        span: SourceSpan,
        successors: &[GenericSuccessor],
        groups: Option<usize>,
    ) -> ParseResult {
        let declared = name.successor_groups();
        if declared.is_empty() && !successors.is_empty() {
            return Err(ParserError::InvalidSuccessorCount {
                span,
                expected: 0,
                actual: successors.len(),
            });
        }
        if let Some(groups) = groups
            && groups != declared.len()
        {
            return Err(ParserError::InvalidSuccessorGroupCount {
                span,
                name: name.to_string(),
                expected: declared.len(),
                actual: groups,
            });
        }
        if let Some(extra) = successors.iter().find(|successor| successor.group >= declared.len()) {
            return Err(ParserError::InvalidSuccessorCount {
                span: extra.block.span(),
                expected: 0,
                actual: successors.len(),
            });
        }
        for (index, shape) in declared.iter().enumerate() {
            let members = || successors.iter().filter(move |successor| successor.group == index);
            let actual = members().count();
            if let Some(expected) = shape.size
                && actual != expected
            {
                return Err(ParserError::InvalidSuccessorCount {
                    span,
                    expected,
                    actual,
                });
            }
            for successor in members() {
                match (shape.key, successor.key) {
                    (None, None) => (),
                    (Some(is_key), Some(key)) if is_key(key.inner()) => (),
                    (Some(_), Some(key)) => {
                        return Err(ParserError::InvalidSuccessorKey {
                            span: key.span(),
                            reason: format!(
                                "this is not a key of the kind successor group {index} of \
                                 '{name}' takes"
                            ),
                        });
                    }
                    (Some(_), None) => {
                        return Err(ParserError::InvalidSuccessorKey {
                            span: successor.block.span(),
                            reason: format!(
                                "successor group {index} of '{name}' takes a key before each \
                                 successor"
                            ),
                        });
                    }
                    (None, Some(key)) => {
                        return Err(ParserError::InvalidSuccessorKey {
                            span: key.span(),
                            reason: format!("successor group {index} of '{name}' takes no key"),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Parse an operation instance that is in the generic form.
    ///
    /// If `ip` is provided, operation is inserted at that point.
    pub fn parse_generic_operation(
        &mut self,
        ip: Option<ProgramPoint>,
    ) -> ParseResult<OperationRef> {
        self.parse_generic_operation_internal(ip, true)
    }

    /// Parse a generic operation, optionally leaving source-state finalization to the caller.
    ///
    /// The operation is built from the text alone, never with its custom parser. A result type
    /// the text states that differs from the type the operation gives the result once built (by
    /// inferring it) is an error at the stated type.
    fn parse_generic_operation_internal(
        &mut self,
        ip: Option<ProgramPoint>,
        finalize_definition: bool,
    ) -> ParseResult<OperationRef> {
        let (span, name) = self
            .token_stream_mut()
            .expect_map("operation name", |tok| match tok {
                Token::String(name) => Some(CompactString::from(name)),
                _ => None,
            })?
            .into_parts();

        if name.is_empty() {
            return Err(ParserError::InvalidOperationName {
                span,
                reason: "operation names cannot be empty".to_string(),
            });
        }

        let Some((dialect_name, opcode)) = name.split_once('.') else {
            return Err(ParserError::InvalidOperationName {
                span,
                reason: "operation names must be fully-qualified, e.g. <dialect>.<opcode>"
                    .to_string(),
            });
        };

        // The dialect name comes from the source, so it may name nothing at all. This is a
        // different failure from an unregistered operation inside a dialect that does exist.
        let dialect_symbol = interner::Symbol::intern(dialect_name);
        let Some(dialect) = self.parser.context().find_registered_dialect(dialect_symbol) else {
            return Err(ParserError::UnknownDialect {
                span,
                dialect: dialect_symbol,
            });
        };
        let opcode = interner::Symbol::intern(opcode);
        let Some(op_name) =
            dialect.registered_ops().iter().find(|name| name.name() == opcode).cloned()
        else {
            return Err(ParserError::InvalidOperationName {
                span,
                reason: "unable to parse unregistered operations".to_string(),
            });
        };

        // If we are populating the parser state, start a new operation definition.
        if let Some(asm_state) = self.state_mut().asm_state.as_deref_mut() {
            asm_state.start_operation_definition();
        }

        let mut result = OperationState::new(span, op_name.clone());
        let mut guard = CleanupOpStateRegions { state: &mut result };

        let stated_results =
            self.parse_generic_operation_parts(&mut guard, None, None, None, None, None)?;

        // Create the operation and try to parse a location for it.
        let op = if let Some(ip) = ip {
            let mut builder = InsertionGuard::new(self.builder_mut());
            builder.set_insertion_point(ip);
            builder.create_operation(&mut guard)?
        } else {
            self.builder_mut().create_operation(&mut guard)?
        };
        {
            let built = op.borrow();
            if stated_results.len() != built.num_results() {
                return Err(ParserError::StatedResultCountMismatch {
                    span: stated_results.first().map_or(span, |stated| stated.span()),
                    stated: stated_results.len(),
                    built: built.num_results(),
                    op: op_name.to_string(),
                });
            }
            for (stated, result) in stated_results.iter().zip(built.results().all().iter()) {
                let inferred = result.borrow().ty();
                if inferred != *stated.inner() {
                    return Err(ParserError::ResultTypeMismatch {
                        span: stated.span(),
                        stated: stated.inner().to_string(),
                        inferred: inferred.to_string(),
                        op: op_name.to_string(),
                    });
                }
            }
        }
        self.parse_trailing_location_specifier(OpOrArgument::Op(op))?;

        self.parse_semicolon()?;

        if finalize_definition {
            let end = self.current_location();
            if let Some(asm_state) = self.state_mut().asm_state.as_deref_mut() {
                asm_state.finalize_operation_definition(op, span, end, &[]);
            }
        }

        Ok(op)
    }

    /// Parse different components, viz., use-info of operand(s), successor(s), region(s),
    /// attribute(s) and function-type, of the generic form of an operation instance and populate
    /// the input operation-state 'result' with those components.
    ///
    /// If any of the components is explicitly provided, then skip parsing that component. Given
    /// operands are one operand group; given successors are one group of successors without
    /// keys or arguments, so they cannot be given for an operation whose successors are keyed or
    /// in more than one group.
    pub fn parse_generic_operation_after_name(
        &mut self,
        result: &mut OperationState,
        parsed_operand_use_info: Option<&[UnresolvedOperand]>,
        parsed_successors: Option<&[BlockRef]>,
        parsed_regions: Option<&[RegionRef]>,
        parsed_attributes: Option<ParsedAttrs>,
        parsed_fn_type: Option<Type>,
    ) -> ParseResult {
        self.parse_generic_operation_parts(
            result,
            parsed_operand_use_info,
            parsed_successors,
            parsed_regions,
            parsed_attributes,
            parsed_fn_type,
        )
        .map(drop)
    }

    /// [Self::parse_generic_operation_after_name], returning the result types the text states,
    /// each with where it is written (the operation's own span for a given type).
    ///
    /// The operands are put in the operation's own groups, in order, as the signature divides
    /// them, and the arguments of each successor in a group of its own after those.
    fn parse_generic_operation_parts(
        &mut self,
        result: &mut OperationState,
        parsed_operand_use_info: Option<&[UnresolvedOperand]>,
        parsed_successors: Option<&[BlockRef]>,
        parsed_regions: Option<&[RegionRef]>,
        parsed_attributes: Option<ParsedAttrs>,
        parsed_fn_type: Option<Type>,
    ) -> ParseResult<SmallVec<[Span<Type>; 4]>> {
        let name = result.name.clone();

        // Parse the operand list, if not explicitly provided.
        let mut operand_info = SmallVec::<[UnresolvedOperand; 8]>::default();
        if let Some(provided_use_info) = parsed_operand_use_info {
            operand_info.extend_from_slice(provided_use_info);
        } else {
            self.parser.parse_lparen()?;
            self.parse_optional_ssa_use_list(&mut operand_info)?;
            self.parser.parse_rparen()?;
        }

        // Parse the successor list, if not explicitly provided.
        let mut successors = SmallVec::<[GenericSuccessor; 2]>::default();
        let groups = if let Some(provided_succs) = parsed_successors {
            if !matches!(name.successor_groups(), [] | [SuccessorGroupShape { key: None, .. }]) {
                return Err(ParserError::UngroupedSuccessors {
                    span: result.span,
                    name: name.to_string(),
                });
            }
            successors.extend(provided_succs.iter().map(|block| GenericSuccessor {
                block: Span::new(result.span, *block),
                key: None,
                group: 0,
                operands: SmallVec::new_const(),
            }));
            None
        } else if self.parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Lbracket)) {
            // Check if the operation is not a known terminator.
            if !name.implements::<dyn Terminator>() {
                return Err(ParserError::NonTerminatorWithSuccessors { span: result.span });
            }
            Some(self.parse_generic_successors(&mut successors)?)
        } else {
            None
        };
        Self::check_generic_successors(&name, result.span, &successors, groups)?;

        // Parse the properties, `<{ .. }>`, where the printer writes them. A property may also
        // be given in the trailing attribute dictionary, but not in both. An entry that is not a
        // property of the operation is an error.
        let properties_start = result.attrs.len();
        if self.token_stream_mut().next_if_eq(Token::Langle)? {
            let op_name = &name;
            self.parse_attribute_dict_checking_keys(&mut result.attrs, &mut |key| {
                if op_name.has_property(*key.inner()) {
                    Ok(())
                } else {
                    Err(ParserError::NotAProperty {
                        span: key.span(),
                        name: key.into_inner(),
                        op: op_name.to_string(),
                    })
                }
            })?;
            self.parser.parse_rangle()?;
        }
        let properties_end = result.attrs.len();

        // Parse the region list, if not explicitly provided. The regions of an operation that is
        // isolated from above are their own naming scopes, as in its custom form.
        if let Some(provided_regions) = parsed_regions {
            result.regions.extend_from_slice(provided_regions);
        } else if self.token_stream_mut().is_next(|tok| matches!(tok, Token::Lparen)) {
            self.parser.parse_lparen()?;
            let isolated = name.implements::<dyn IsolatedFromAbove>();
            loop {
                let region = self.builder().context().create_region();
                self.parse_region(region, &[], isolated)?;
                result.regions.push(region);
                if !self.token_stream_mut().next_if_eq(Token::Comma)? {
                    break;
                }
            }
            self.parser.parse_rparen()?;
        }

        // Parse the attributes, if not explicitly provided.
        if let Some(provided_attrs) = parsed_attributes {
            result.attrs.extend(provided_attrs);
        } else if self.token_stream_mut().is_next(|tok| matches!(tok, Token::Lbrace)) {
            let start = self.parser.token_stream_mut().peek()?.map(|(start, ..)| start);
            self.parse_attribute_dict(&mut result.attrs)?;
            let end = self.parser.token_stream().previous_end();
            let (properties, attributes) = result.attrs.split_at(properties_end);
            let properties = &properties[properties_start..];
            if let Some(duplicate) =
                attributes.iter().find(|attr| properties.iter().any(|p| p.name == attr.name))
            {
                return Err(ParserError::DuplicateAttribute {
                    span: span!(self.parser.source_id(), start.unwrap_or(end), end),
                    name: duplicate.name,
                });
            }
        }

        // Parse the operation type, if not explicitly provided.
        let mut params = SmallVec::<[Type; 4]>::default();
        let mut group_sizes = SmallVec::<[usize; 4]>::default();
        let mut results = SmallVec::<[Span<Type>; 4]>::default();
        let ty_span = if let Some(provided_fn_ty) = parsed_fn_type {
            let Some(fn_ty) = provided_fn_ty.as_function() else {
                return Err(ParserError::InvalidOperationType {
                    span: result.span,
                    ty_span: result.span,
                    reason: format!("expected a function type, got '{provided_fn_ty}'"),
                });
            };
            params.extend_from_slice(fn_ty.params());
            group_sizes.push(params.len());
            results.extend(fn_ty.results().iter().map(|ty| Span::new(result.span, *ty)));
            result.span
        } else {
            self.parser.parse_colon()?;
            let start = self.parser.token_stream_mut().peek()?.map(|(start, ..)| start);
            self.parse_generic_operand_types(&mut params, &mut group_sizes)?;
            self.parser.parse_arrow()?;
            self.parse_generic_result_types(&mut results)?;
            let end = self.parser.token_stream().previous_end();
            span!(self.parser.source_id(), start.unwrap_or(end), end)
        };
        result.results.extend(results.iter().map(|ty| *ty.inner()));

        // Check that we have the right number of types for the operands.
        if operand_info.len() != params.len() {
            return Err(ParserError::InvalidOperationType {
                span: result.span,
                ty_span,
                reason: format!(
                    "expected {} operand type(s), but got {}",
                    operand_info.len(),
                    params.len()
                ),
            });
        }

        // The operation's own groups come first, then one group per successor. These groups
        // must exist even when they have no operands, and the index of the last must fit the
        // 16 bits an operation stores it in.
        let own_groups = group_sizes.len();
        let total_groups = own_groups + successors.len();
        if total_groups > u16::MAX as usize + 1 {
            return Err(ParserError::TooManyOperandGroups {
                span: result.span,
                limit: u16::MAX as usize,
            });
        }
        result.operands.resize_with(total_groups, Default::default);

        // Resolve all of the operands, into the groups the signature divides them into.
        let mut operands = operand_info.iter().zip(params);
        for (group, size) in group_sizes.iter().copied().enumerate() {
            for (use_info, ty) in operands.by_ref().take(size) {
                let value = self.resolve_ssa_use(*use_info, ty)?;
                result.operands[group].push(value);
            }
        }

        for (index, successor) in successors.into_iter().enumerate() {
            let operand_group = own_groups + index;
            result.operands[operand_group] = successor.operands;
            result.successors.push(PendingSuccessorInfo {
                block: successor.block.into_inner(),
                key: successor.key.map(Span::into_inner),
                // Both fit: the number of operand groups is checked above, and the successor
                // group is one of those the operation declares
                operand_group: operand_group as u16,
                successor_group: successor.group as u16,
            });
        }

        Ok(results)
    }

    /// Parse the operand types of the signature of an operation in the generic form, appending
    /// them to `types` and the size of each operand group to `group_sizes`.
    ///
    ///   generic-op-operand-types ::= `(` operand-group (`|` operand-group)* `)` | type
    ///   operand-group ::= type-list-no-parens?
    ///
    /// A lone type is one group of one operand.
    fn parse_generic_operand_types(
        &mut self,
        types: &mut SmallVec<[Type; 4]>,
        group_sizes: &mut SmallVec<[usize; 4]>,
    ) -> ParseResult {
        if !self.token_stream_mut().next_if_eq(Token::Lparen)? {
            types.push(self.parser.parse_type()?.into_inner());
            group_sizes.push(1);
            return Ok(());
        }
        loop {
            let mut size = 0;
            if !self
                .token_stream_mut()
                .is_next(|tok| matches!(tok, Token::Pipe | Token::Rparen))
            {
                loop {
                    types.push(self.parser.parse_type()?.into_inner());
                    size += 1;
                    if !self.token_stream_mut().next_if_eq(Token::Comma)? {
                        break;
                    }
                }
            }
            group_sizes.push(size);
            if !self.token_stream_mut().next_if_eq(Token::Pipe)? {
                break;
            }
        }
        self.parser.parse_rparen()
    }

    /// Parse the result types of the signature of an operation in the generic form, each with
    /// where it is written.
    ///
    ///   generic-op-types ::= `(` type-list-no-parens? `)` | type
    fn parse_generic_result_types(&mut self, types: &mut SmallVec<[Span<Type>; 4]>) -> ParseResult {
        if !self.token_stream_mut().next_if_eq(Token::Lparen)? {
            types.push(self.parser.parse_type()?);
            return Ok(());
        }
        if !self.token_stream_mut().is_next(|tok| matches!(tok, Token::Rparen)) {
            loop {
                types.push(self.parser.parse_type()?);
                if !self.token_stream_mut().next_if_eq(Token::Comma)? {
                    break;
                }
            }
        }
        self.parser.parse_rparen()
    }

    /// Parse an optional trailing location and give it to the operation or block argument.
    ///
    ///   trailing-location ::= (`loc` `(` location `)`)?
    ///
    /// A location alias (`loc(#loc42)`) is an error: nothing defines one.
    fn parse_trailing_location_specifier(&mut self, op_or_argument: OpOrArgument) -> ParseResult {
        // If there is a 'loc' we parse a trailing location.
        if !self.token_stream_mut().next_if_eq(Token::Loc)? {
            return Ok(());
        }

        self.parser.parse_lparen()?;
        self.reject_location_alias()?;
        let loc = self.parser.parse_location_instance()?;
        self.parser.parse_rparen()?;

        if let Some(span) = loc.try_into_span(self.context()) {
            op_or_argument.set_span(span);
        }

        Ok(())
    }

    /// An error if the next token is a location alias, a hash identifier without a dot (with
    /// one, it starts a dialect attribute): the text cannot define an alias, so none resolves.
    fn reject_location_alias(&mut self) -> ParseResult {
        if self
            .token_stream_mut()
            .is_next(|tok| matches!(tok, Token::HashIdent(id) if !id.contains('.')))
        {
            let alias = self.token_stream_mut().expect_map("location alias", |tok| match tok {
                Token::HashIdent(_) => Some(()),
                _ => None,
            })?;
            return Err(ParserError::LocationAlias { span: alias.span() });
        }
        Ok(())
    }

    /// Parse an operation instance that is in the op-defined custom form.
    ///
    /// `results` specifies information about the "%name =" specifiers.
    pub fn parse_custom_operation(
        &mut self,
        results: &[ResultRecord],
    ) -> ParseResult<OperationRef> {
        let (name_span, name) = self.parse_custom_operation_name()?.into_parts();

        // This is the actual hook for the custom op parsing, usually implemented by the op itself
        // (`OpParser::parse()`). We retrieve it either from the OperationName or from the Dialect.
        let Some(parse_assembly_fn) = name.parse_assembly_fn() else {
            return Err(ParserError::InvalidCustomOperation {
                span: name_span,
                reason: format!(
                    "'{name}' has no custom form: write it in the generic form, \"{name}\"(..)"
                ),
            });
        };
        let isolated_from_above = name.implements::<dyn IsolatedFromAbove>();
        //let default_dialect = name.default_dialect();
        // let guard = DefaultDialectStackScope::new(&mut self.default_dialect_stack);
        // guard.push(default_dialect);
        //

        let mut op_state = OperationState::new(name_span, name);

        // If we are populating the parser state, start a new operation definition.
        if let Some(asm_state) = self.state_mut().asm_state.as_deref_mut() {
            asm_state.start_operation_definition();
        }

        // Have the op implementation take a crack and parsing this.
        let span = op_state.span;
        let name = op_state.name.clone();
        let mut guard = CleanupOpStateRegions {
            state: &mut op_state,
        };
        let mut custom_parser = CustomOpAsmParser::new(
            span,
            results,
            parse_assembly_fn,
            name,
            isolated_from_above,
            self,
        );
        custom_parser.parse_operation(&mut guard)?;

        // Otherwise, create the operation and try to parse a location for it.
        let op = self.builder_mut().create_operation(&mut guard)?;

        self.parse_trailing_location_specifier(OpOrArgument::Op(op))?;

        self.parse_semicolon()?;

        Ok(op)
    }

    /// Parse the name of an operation, in the custom form.
    pub fn parse_custom_operation_name(&mut self) -> ParseResult<Span<OperationName>> {
        let (name_span, name) = self
            .token_stream_mut()
            .expect_map("operation name", |tok| match tok {
                Token::BareIdent(id) => Some(id),
                _ => None,
            })?
            .into_parts();

        // If the operation doesn't have a dialect prefix try using the default dialect.
        let (dialect, opcode) = name.split_once('.').unwrap_or_else(|| {
            (self.state_mut().default_dialect_stack.last().unwrap().as_str(), name)
        });

        let dialect_symbol = interner::Symbol::intern(dialect);
        let Some(dialect) = self.context().find_registered_dialect(dialect_symbol) else {
            return Err(ParserError::UnknownDialect {
                span: name_span,
                dialect: dialect_symbol,
            });
        };
        dialect
            .registered_ops()
            .iter()
            .find(|name| name.name() == opcode)
            .cloned()
            .map(|name| Span::new(name_span, name))
            .ok_or(ParserError::UnknownOperation { span: name_span })
    }
}

// Region Parsing
impl<'input, P> OperationParser<P>
where
    P: Parser<'input>,
{
    /// Parse a region into 'region' with the provided entry block arguments.
    ///
    /// 'isolated' indicates if the naming scope of this region is isolated from those above.
    ///
    /// Fails if the region would be nested in more regions than
    /// [ParserConfig::max_region_depth](super::ParserConfig::max_region_depth) allows.
    pub fn parse_region(
        &mut self,
        region: RegionRef,
        entry_arguments: &[Argument],
        isolated: bool,
    ) -> ParseResult {
        let enclosing = self.parser.state().region_nesting;
        let limit = self.parser.state().config.max_region_depth;
        if enclosing >= limit {
            return Err(self.region_nesting_too_deep(limit));
        }
        // Restored on every return from the nested parse, successful or not, so that the count
        // always matches the regions actually being parsed.
        self.parser.state_mut().region_nesting = enclosing + 1;
        let result = self.parse_region_nested(region, entry_arguments, isolated);
        self.parser.state_mut().region_nesting = enclosing;
        result
    }

    /// The error for a region nested in more than `limit` regions, at the `{` that opens it: it
    /// is the region `limit + 1` deep (parsing stops there, so deeper regions are not seen).
    #[cold]
    #[inline(never)]
    fn region_nesting_too_deep(&mut self, limit: u32) -> ParserError {
        let span = match self.parser.token_stream_mut().peek() {
            Ok(Some((start, _, end))) => span!(self.parser.source_id(), start, end),
            Ok(None) => self.parser.current_location(),
            Err(err) => return err,
        };
        ParserError::RegionNestingTooDeep {
            span,
            limit,
            depth: limit.saturating_add(1),
        }
    }

    /// [Self::parse_region], once the nesting has been accounted for.
    fn parse_region_nested(
        &mut self,
        region: RegionRef,
        entry_arguments: &[Argument],
        isolated: bool,
    ) -> ParseResult {
        // Parse the '{'.
        self.parser.parse_lbrace()?;

        // Parse the region body.
        if !entry_arguments.is_empty()
            || self.token_stream_mut().is_next(|tok| !matches!(tok, Token::Rbrace))
        {
            let start = self.parser.current_location();
            self.parse_region_body(region, start, entry_arguments, isolated)?;
        }

        self.parser.parse_rbrace()?;

        Ok(())
    }

    /// Parse a region body into 'region'.
    pub fn parse_region_body(
        &mut self,
        mut region: RegionRef,
        start: SourceSpan,
        entry_arguments: &[Argument],
        isolated: bool,
    ) -> ParseResult {
        let ip = *self.builder().insertion_point();

        // Push a new named value scope.
        self.push_ssa_name_scope(isolated);

        // Parse the first block directly to allow for it to be unnamed.
        let owning_block = self.builder().context_rc().create_block();

        // If this block is not defined in the source file, add a definition for it now in the
        // assembly state. Blocks with a name will be defined when the name is parsed.
        if !self
            .parser
            .token_stream_mut()
            .is_next(|tok| matches!(tok, Token::CaretIdent(_)))
            && let Some(asm_state) = self.parser.state_mut().asm_state.as_deref_mut()
        {
            asm_state.add_block_definition(owning_block, start);
        }

        // Add arguments to the entry block if we had the form with explicit names.
        if !entry_arguments.is_empty() && entry_arguments[0].name.name.is_user_defined() {
            // If we had named arguments, then don't allow a block name.
            if let Some((start, Token::CaretIdent(_), end)) = self.token_stream_mut().peek()? {
                return Err(ParserError::BlockNameInRegionWithNamedArgs {
                    span: span!(self.source_id(), start, end),
                });
            }

            for arg in entry_arguments {
                let arg_info = arg.name;

                // Ensure that the argument was not already defined.
                if let Some(def_loc) = self.get_reference_loc(arg_info.name) {
                    return Err(ParserError::ValueRedefinition {
                        name: arg_info.name,
                        span: arg_info.loc,
                        prev_span: def_loc,
                    });
                }
                let argument = self
                    .context()
                    .append_block_argument(owning_block, arg.ty, arg_info.loc)
                    .borrow()
                    .downcast_ref::<BlockArgument>()
                    .unwrap()
                    .as_block_argument_ref();
                // An argument written with a location takes it.
                if let Some(span) =
                    arg.loc.as_ref().and_then(|loc| loc.try_into_span(self.context()))
                {
                    OpOrArgument::Arg(argument).set_span(span);
                }
                // Add a definition of this arg to the assembly state if provided.
                if let Some(asm_state) = self.state_mut().asm_state.as_deref_mut() {
                    asm_state.add_block_argument_definition(argument, arg_info.loc);
                }

                // Record the definition for this argument.
                self.add_definition(arg_info, argument as ValueRef)?;
            }
        }

        self.parse_block(Some(owning_block))?;

        // Verify that no other arguments were parsed.
        if !entry_arguments.is_empty()
            && owning_block.borrow().num_arguments() > entry_arguments.len()
        {
            return Err(ParserError::EntryBlockArgumentsAlreadyDefined { span: start });
        }

        // Parse the rest of the region.
        region.borrow_mut().body_mut().push_back(owning_block);

        while self.token_stream_mut().is_next(|tok| !matches!(tok, Token::Rbrace)) {
            // Each subsequent block starts with a label; `parse_block` resolves it to the
            // forward-declared block when the label was already referenced as a successor.
            let block = self.parse_block(None)?;
            region.borrow_mut().push_back(block);
        }

        // Pop the SSA value scope for this region.
        self.pop_ssa_name_scope()?;

        // Reset the original insertion point.
        self.builder_mut().restore_insertion_point(ip);

        Ok(())
    }
}

// Block Parsing
impl<'input, P> OperationParser<P>
where
    P: Parser<'input>,
{
    /// Parse a new block, returning the block that was defined.
    ///
    /// When `block` is `Some`, that block is the one populated and returned - unless the parsed
    /// label was already forward-referenced as a successor, in which case this panics: branch
    /// targets already point at the forward-declared block, so the caller's block cannot be used.
    /// Callers that may encounter forward-declared labels must pass `None` and attach the returned
    /// block to the region.
    ///
    ///   block ::= block-label? operation*
    ///   block-label    ::= block-id block-arg-list? `:`
    ///   block-id       ::= caret-id
    ///   block-arg-list ::= `(` ssa-id-and-type-list? `)`
    ///
    pub fn parse_block(&mut self, block: Option<BlockRef>) -> ParseResult<BlockRef> {
        // The first block of a region may already exist, if it does the caret identifier is
        // optional.
        if let Some(block) = block
            && self.token_stream_mut().is_next(|tok| !matches!(tok, Token::CaretIdent(_)))
        {
            self.parse_block_body(block)?;
            return Ok(block);
        }

        let (name_span, written) = self
            .token_stream_mut()
            .expect_map("a block name", |tok| match tok {
                Token::CaretIdent(name) => Some(interner::Symbol::intern(name)),
                _ => None,
            })?
            .into_parts();
        let name = BlockId::from_symbol(written);

        // Define the block with the specified name.
        //
        // If a block has yet to be set, this is a new definition. If the caller provided a block,
        // use it. Otherwise create a new one.
        let block_and_loc = self.get_block_info_by_name(name);

        let block = if let Some(forward_declared) = block_and_loc {
            // The block has a forward declaration. Forward declarations are removed once defined,
            // so if this name already resolves to a non-forward-declared block, it is a
            // redeclaration.
            if !self.erase_forward_ref(forward_declared.into_inner()) {
                return Err(ParserError::BlockAlreadyDefined {
                    span: name_span,
                    name: alloc::format!("^{written}"),
                });
            }
            // A caller that provided a block expects it to be the one populated, but branch
            // targets already point at the forward-declared block, so we cannot honor that here.
            // Such callers must pass `None` and use the returned block instead.
            assert!(
                block.is_none(),
                "parse_block was given an explicit block, but `{name}` was forward-referenced as \
                 a successor; pass `None` so the forward-declared block can be used"
            );
            Span::new(name_span, forward_declared.into_inner())
        } else {
            let block =
                Span::new(name_span, block.unwrap_or_else(|| self.context_rc().create_block()));
            // Register the definition so later references (e.g. backwards branches) resolve to
            // this block, and duplicate definitions are detected.
            self.blocks_by_name.last_mut().unwrap().insert(name, block);
            block
        };

        // Populate the high level assembly state if necessary.
        if let Some(asm_state) = self.state_mut().asm_state.as_deref_mut() {
            asm_state.add_block_definition(block.into_inner(), name_span);
        }

        // If an argument list is present, parse it.
        if self.token_stream_mut().is_next(|tok| matches!(tok, Token::Lparen)) {
            self.parse_optional_block_arg_list(block.into_inner())?;
        }
        self.parse_colon()?;

        // Parse the body of the block.
        self.parse_block_body(block.into_inner())?;
        Ok(block.into_inner())
    }

    /// Parse a list of operations into 'block'.
    pub fn parse_block_body(&mut self, block: BlockRef) -> ParseResult {
        // Set the insertion point to the end of the block to parse.
        self.builder_mut().set_insertion_point_to_end(block);

        // Parse the list of operations that make up the body of the block.
        while self
            .token_stream_mut()
            .is_next(|tok| !matches!(tok, Token::CaretIdent(_) | Token::Rbrace))
        {
            self.parse_operation()?;
        }

        Ok(())
    }

    /// Parse a (possibly empty) list of SSA operands with types as block arguments
    /// enclosed in parentheses.
    ///
    ///   value-id-and-type-list ::= value-id-and-type (`,` ssa-id-and-type)*
    ///   block-arg-list ::= `(` value-id-and-type-list? `)`
    ///
    pub fn parse_optional_block_arg_list(&mut self, owner: BlockRef) -> ParseResult {
        if self.token_stream_mut().is_next(|tok| matches!(tok, Token::Rbrace)) {
            return Ok(());
        }

        // If the block already has arguments, then we're handling the entry block.
        // Parse and register the names for the arguments, but do not add them.
        let defining_existing_args = owner.borrow().has_arguments();
        let mut next_argument = 0usize;

        let context = self.context_rc();
        self.parse_comma_separated_list(Delimiter::Paren, Some("block argument list"), |parser| {
            parser.parse_ssa_def_or_use_and_type(|parser, use_info, ty| {
                // If we are defining existing arguments, ensure that the argument has already been
                // created with the right type.
                let arg = if defining_existing_args {
                    // Otherwise, ensure that this argument has already been created.
                    let owner = owner.borrow();
                    if next_argument >= owner.num_arguments() {
                        return Err(ParserError::TooManyBlockArguments { span: use_info.loc });
                    }

                    // Finally, make sure the existing argument has the correct type.
                    let arg = owner.arguments()[next_argument];
                    next_argument += 1;
                    let arg_borrowed = arg.borrow();
                    if arg_borrowed.ty() != ty {
                        return Err(ParserError::BlockArgumentTypeMismatch {
                            span: use_info.loc,
                            arg: arg_borrowed.id(),
                            ty: format_compact!("{}", arg_borrowed.ty()),
                            expected: format_compact!("{ty}"),
                        });
                    }
                    arg
                } else {
                    let arg = context.append_block_argument(owner, ty, use_info.loc);
                    arg.downcast_value::<BlockArgument>()
                };

                // If the argument has an explicit loc(...) specifier, parse and apply it.
                parser.parse_trailing_location_specifier(OpOrArgument::Arg(arg))?;

                // Mark this block argument definition in the parser state if it was provided.
                if let Some(asm_state) = parser.state_mut().asm_state.as_deref_mut() {
                    asm_state.add_block_argument_definition(arg, use_info.loc);
                }

                parser.add_definition(use_info, arg)
            })?;

            Ok(true)
        })
    }

    /// Get the block with the specified name, creating it if it doesn't already exist.
    ///
    /// The location specified is the point of use, which allows us to diagnose references to blocks
    /// that are not defined precisely.
    pub fn get_block_named(&mut self, name: BlockId, loc: SourceSpan) -> Span<BlockRef> {
        let block = match self.get_block_info_by_name(name) {
            Some(block) => block,
            None => {
                let block = self.context_rc().create_block();
                self.blocks_by_name.last_mut().unwrap().insert(name, Span::new(loc, block));
                self.insert_forward_ref(block, loc);
                Span::new(loc, block)
            }
        };

        if let Some(asm_state) = self.state_mut().asm_state.as_deref_mut() {
            asm_state.add_block_uses(block.into_inner(), &[loc]);
        }

        block
    }
}

impl<'input, P> OperationParser<P>
where
    P: Parser<'input>,
{
    /// Returns the info for a block at the current scope for the given name.
    fn get_block_info_by_name(&self, name: BlockId) -> Option<Span<BlockRef>> {
        self.blocks_by_name.last()?.get(&name).copied()
    }

    /// Insert a new forward reference to the given block.
    fn insert_forward_ref(&mut self, block: BlockRef, loc: SourceSpan) {
        self.forward_ref.last_mut().unwrap().insert_new(block, loc);
    }

    /// Erase any forward reference to the given block.
    fn erase_forward_ref(&mut self, block: BlockRef) -> bool {
        self.forward_ref.last_mut().unwrap().remove(&block).is_some()
    }

    /// Return true if this is a forward reference.
    fn is_forward_ref_placeholder(&self, value: ValueRef) -> bool {
        self.forward_ref_placeholders.contains_key(&value)
    }
}

impl<P> Drop for OperationParser<P> {
    fn drop(&mut self) {
        for mut op in self.forward_ref_ops.drain(..) {
            let mut op = op.borrow_mut();
            // Drop all uses of undefined forward declared reference and destroy defining operation.
            op.drop_all_uses();
            op.erase();
        }
        for scope in self.forward_ref.drain(..) {
            for (mut fwd, _) in scope.into_iter() {
                // Delete all blocks that were created as forward references but never
                // included into a region.
                fwd.borrow_mut().drop_all_uses();
            }
        }
    }
}

// RAII-style guard for cleaning up the regions in the operation state before deleting them.
//
// Within the parser, regions may get deleted if parsing failed, and other errors may be present, in
// particular undominated uses.  This makes sure such uses are deleted.
struct CleanupOpStateRegions<'a> {
    state: &'a mut OperationState,
}

impl<'a> Deref for CleanupOpStateRegions<'a> {
    type Target = OperationState;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &*self.state
    }
}

impl<'a> DerefMut for CleanupOpStateRegions<'a> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.state
    }
}

impl<'a> Drop for CleanupOpStateRegions<'a> {
    fn drop(&mut self) {
        for mut region in self.state.regions.iter_mut() {
            let mut region = region.borrow_mut();
            let mut body = region.body_mut().front_mut();
            loop {
                if let Some(mut block) = body.get_mut() {
                    block.drop_all_defined_value_uses();
                } else {
                    break;
                }
                body.move_next();
            }
        }
    }
}

struct CustomOpAsmParser<'a, P> {
    span: SourceSpan,
    /// The operation name
    op_name: OperationName,
    /// The backing operation parser.
    parser: &'a mut OperationParser<P>,
    /// Information about the result name specifiers.
    result_ids: &'a [ResultRecord],
    /// The abstract information of the operation.
    parse_assembly: ParseAssemblyFn,
    isolated_from_above: bool,
}

impl<'a, 'input: 'a, P> CustomOpAsmParser<'a, P>
where
    P: Parser<'input>,
{
    pub fn new(
        span: SourceSpan,
        result_ids: &'a [ResultRecord],
        parse_assembly: ParseAssemblyFn,
        op_name: OperationName,
        isolated_from_above: bool,
        parser: &'a mut OperationParser<P>,
    ) -> Self {
        Self {
            span,
            op_name,
            parser,
            result_ids,
            parse_assembly,
            isolated_from_above,
        }
    }

    /// Parse an instance of the operation described by 'opDefinition' into the provided operation
    /// state.
    pub fn parse_operation(&mut self, state: &mut OperationState) -> ParseResult {
        let parse_assembly = self.parse_assembly;
        parse_assembly(state, self)?;

        // Verify that the parsed attributes does not have duplicate attributes.
        //
        // This can happen if an attribute set during parsing is also specified in the attribute
        // dictionary in the assembly, or the attribute is set multiple during parsing.
        state.attrs.sort_by_key(|attr| attr.name);
        for (i, attr) in state.attrs.iter().enumerate() {
            if state.attrs.get(i + 1).is_some_and(|attr2| attr.name == attr2.name) {
                return Err(ParserError::DuplicateAttribute {
                    span: self.span,
                    name: attr.name,
                });
            }
        }
        Ok(())
    }
}

impl<'a, 'input: 'a, P> Parser<'input> for CustomOpAsmParser<'a, P>
where
    P: Parser<'input>,
{
    fn builder(&self) -> &OpBuilder {
        self.parser.builder()
    }

    fn builder_mut(&mut self) -> &mut OpBuilder {
        self.parser.builder_mut()
    }

    fn state(&self) -> &ParserState<'input> {
        self.parser.state()
    }

    fn state_mut(&mut self) -> &mut ParserState<'input> {
        self.parser.state_mut()
    }

    fn token_stream(&self) -> &TokenStream<'input> {
        self.parser.token_stream()
    }

    fn token_stream_mut(&mut self) -> &mut TokenStream<'input> {
        self.parser.token_stream_mut()
    }

    fn context<'p>(&'p self) -> &'p Context
    where
        'input: 'p,
    {
        self.parser.context()
    }

    fn context_rc(&self) -> Rc<Context> {
        self.parser.context_rc()
    }

    fn source_id(&self) -> SourceId {
        self.parser.source_id()
    }

    fn current_location(&self) -> SourceSpan {
        self.parser.current_location()
    }

    fn parse_attribute(&mut self, ty: Type) -> ParseResult<Span<AttributeRef>> {
        self.parser.parse_attribute(ty)
    }

    fn parse_extended_attribute(&mut self, ty: Type) -> ParseResult<Span<AttributeRef>> {
        self.parser.parse_extended_attribute(ty)
    }

    fn parse_optional_attribute(&mut self, ty: Type) -> ParseResult<Option<Span<AttributeRef>>> {
        self.parser.parse_optional_attribute(ty)
    }

    fn parse_type(&mut self) -> ParseResult<Span<Type>> {
        self.parser.parse_type()
    }

    fn parse_optional_type(&mut self) -> ParseResult<Option<Span<Type>>> {
        self.parser.parse_optional_type()
    }

    fn parse_type_list(&mut self, result: &mut SmallVec<[Type; 4]>) -> ParseResult {
        self.parser.parse_type_list(result)
    }

    fn parse_type_list_no_parens(&mut self, result: &mut SmallVec<[Type; 4]>) -> ParseResult {
        self.parser.parse_type_list_no_parens(result)
    }

    fn parse_dialect_symbol_body(&mut self) -> ParseResult<Span<CompactString>> {
        self.parser.parse_dialect_symbol_body()
    }

    fn parse_attribute_dict(&mut self, attrs: &mut ParsedAttrs) -> ParseResult {
        self.parser.parse_attribute_dict(attrs)
    }

    fn parse_optional_attribute_dict(&mut self, attrs: &mut ParsedAttrs) -> ParseResult {
        self.parser.parse_optional_attribute_dict(attrs)
    }

    fn parse_optional_attribute_dict_with_keyword(
        &mut self,
        attrs: &mut ParsedAttrs,
    ) -> ParseResult {
        self.parser.parse_optional_attribute_dict_with_keyword(attrs)
    }
}

impl<'a, 'input: 'a, P> OpAsmParser<'input> for CustomOpAsmParser<'a, P>
where
    P: Parser<'input>,
{
    #[inline]
    fn parse_generic_operation(&mut self, ip: Option<ProgramPoint>) -> ParseResult<OperationRef> {
        self.parser.parse_generic_operation(ip)
    }

    #[inline]
    fn parse_generic_operation_after_name(
        &mut self,
        result: &mut OperationState,
        parsed_operand_use_info: Option<&[UnresolvedOperand]>,
        parsed_successors: Option<&[BlockRef]>,
        parsed_regions: Option<&[RegionRef]>,
        parsed_attributes: Option<ParsedAttrs>,
        parsed_fn_type: Option<Type>,
    ) -> ParseResult {
        self.parser.parse_generic_operation_after_name(
            result,
            parsed_operand_use_info,
            parsed_successors,
            parsed_regions,
            parsed_attributes,
            parsed_fn_type,
        )
    }

    #[inline]
    fn parse_custom_operation_name(&mut self) -> ParseResult<Span<OperationName>> {
        self.parser.parse_custom_operation_name()
    }

    /// Return the name of the specified result in the specified syntax, as well
    /// as the subelement in the name.  For example, in this operation:
    ///
    ///  %x, %y:2, %z = foo.op
    ///
    ///    getResultName(0) == {"x", 0 }
    ///    getResultName(1) == {"y", 0 }
    ///    getResultName(2) == {"y", 1 }
    ///    getResultName(3) == {"z", 0 }
    fn get_result_name(&self, mut result_num: usize) -> Option<(interner::Symbol, usize)> {
        // Scan for the resultID that contains this result number.
        for entry in self.result_ids {
            if result_num < entry.count {
                return Some((entry.id, result_num));
            }
            result_num -= entry.count;
        }

        // Invalid result number
        None
    }

    /// Return the number of declared SSA results.  This returns 4 for the foo.op example in the
    /// comment for [Self::get_result_name].
    fn get_num_results(&self) -> usize {
        self.result_ids.iter().map(|entry| entry.count).sum()
    }

    /// Parse a single operand.
    fn parse_operand(&mut self, allow_result_number: bool) -> ParseResult<UnresolvedOperand> {
        self.parser.parse_ssa_use(allow_result_number)
    }

    /// Parse a single operand if present.
    fn parse_optional_operand(
        &mut self,
        allow_result_number: bool,
    ) -> ParseResult<Option<UnresolvedOperand>> {
        if self
            .parser
            .token_stream_mut()
            .is_next(|tok| matches!(tok, Token::PercentIdent(_)))
        {
            self.parse_operand(allow_result_number).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Resolve an operand to an SSA value, emitting an error on failure.
    fn resolve_operand(&mut self, operand: UnresolvedOperand, ty: Type) -> ParseResult<ValueRef> {
        self.parser.resolve_ssa_use(operand, ty)
    }

    /// Parse a single argument with the following syntax:
    ///
    ///   `%ssaname : !type { optionalAttrDict} loc(optionalSourceLoc)`
    ///
    /// If `allowType` is false or `allowAttrs` are false then the respective
    /// parts of the grammar are not parsed.
    fn parse_argument(&mut self, allow_type: bool, allow_attrs: bool) -> ParseResult<Argument> {
        let name = self.parse_operand(/*allow_result_number=*/ false)?;
        let ty = if allow_type {
            Some(self.parse_colon_type()?)
        } else {
            None
        };
        let mut attrs = SmallVec::new_const();
        if allow_attrs {
            self.parse_optional_attribute_dict(&mut attrs)?;
        }
        let loc = self.parse_optional_location_specifier()?;
        Ok(Argument {
            name,
            ty: ty.map_or_else(|| self.context().unknown_type(), |t| t.into_inner()),
            attrs,
            loc,
        })
    }

    fn parse_optional_argument(
        &mut self,
        allow_type: bool,
        allow_attrs: bool,
    ) -> ParseResult<Option<Argument>> {
        if self
            .parser
            .token_stream_mut()
            .is_next(|tok| matches!(tok, Token::PercentIdent(_)))
        {
            self.parse_argument(allow_type, allow_attrs).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Parse a region that takes `arguments` of `argTypes` types.  This
    /// effectively defines the SSA values of `arguments` and assigns their type.
    fn parse_region(
        &mut self,
        region: RegionRef,
        arguments: &[Argument],
        enable_name_shadowing: bool,
    ) -> ParseResult {
        // Try to parse the region.
        assert!(
            !enable_name_shadowing || self.isolated_from_above,
            "name shadowing is only allowed on isolated regions"
        );
        self.parser.parse_region(region, arguments, enable_name_shadowing)
    }

    /// Parses a region if present. If the region is present, a new region is
    /// allocated and placed in `region`. If no region is present, `region`
    /// remains untouched.
    fn parse_optional_region(
        &mut self,
        arguments: &[Argument],
        enable_name_shadowing: bool,
    ) -> ParseResult<Option<RegionRef>> {
        if self.parser.token_stream_mut().is_next(|tok| matches!(tok, Token::Lbrace)) {
            let new_region = self.context().create_region();
            self.parse_region(new_region, arguments, enable_name_shadowing)?;
            Ok(Some(new_region))
        } else {
            Ok(None)
        }
    }

    fn parse_successor(&mut self) -> ParseResult<Span<BlockRef>> {
        self.parser.parse_successor()
    }

    fn parse_optional_successor(&mut self) -> ParseResult<Option<Span<BlockRef>>> {
        if self
            .parser
            .token_stream_mut()
            .is_next(|tok| matches!(tok, Token::CaretIdent(_)))
        {
            self.parse_successor().map(Some)
        } else {
            Ok(None)
        }
    }

    fn parse_successor_and_use_list(
        &mut self,
        operands: &mut SmallVec<[ValueRef; 2]>,
    ) -> ParseResult<Span<BlockRef>> {
        let dest = self.parse_successor()?;

        // Handle optional arguments.
        if self.parse_optional_lparen()? {
            self.parser.parse_optional_ssa_use_and_type_list(operands)?;
            self.parse_rparen()?;
        }

        Ok(dest)
    }

    /// Parse a loc(...) specifier if present, filling in result if so.
    fn parse_optional_location_specifier(&mut self) -> ParseResult<Option<Location>> {
        // If there is a 'loc' we parse a trailing location.
        if !self.parser.token_stream_mut().next_if_eq(Token::Loc)? {
            return Ok(None);
        }

        self.parser.parse_lparen()?;
        self.parser.reject_location_alias()?;
        let loc = self.parser.parse_location_instance()?;
        self.parser.parse_rparen()?;

        Ok(Some(loc))
    }
}

/// Parses the type declarations that come next at the top level, if any.
///
///   type-declarations ::= (type-alias-def | rec-group-def)*
///
pub(super) fn parse_type_declarations<'input, P>(parser: &mut OperationParser<P>) -> ParseResult
where
    P: Parser<'input>,
{
    while parse_optional_type_declaration(parser)? {}
    Ok(())
}

/// Parses a type declaration if one comes next at the top level, and returns whether it did.
///
/// The keyword `type` is recognised only where it is followed by a `!`-identifier, and `rec` only
/// where it is followed by `{`, so both remain usable as operation and attribute names. A `!`-identifier at the top level is the removed
/// `!alias = type` form, and is an error.
fn parse_optional_type_declaration<'input, P>(parser: &mut OperationParser<P>) -> ParseResult<bool>
where
    P: Parser<'input>,
{
    let Some((start, token, end)) = parser.token_stream_mut().peek()? else {
        return Ok(false);
    };
    match token {
        Token::BareIdent("type")
            if matches!(parser.token_stream().peek_second(), Some(Token::BangIdent(_))) =>
        {
            parse_type_alias_def(parser)?;
            Ok(true)
        }
        Token::BareIdent("rec")
            if matches!(parser.token_stream().peek_second(), Some(Token::Lbrace)) =>
        {
            parse_rec_group_def(parser)?;
            Ok(true)
        }
        Token::BangIdent(_) => Err(ParserError::BareTypeAliasDefinition {
            span: span!(parser.source_id(), start, end),
        }),
        _ => Ok(false),
    }
}

/// Parse a type alias declaration.
///
///   type-alias-def ::= 'type' '!' alias-name `=` type `;`
///
fn parse_type_alias_def<'input, P>(parser: &mut OperationParser<P>) -> ParseResult
where
    P: Parser<'input>,
{
    let start = parser.token_stream().current_position();
    let (span, id) = parse_type_alias_def_head(parser)?;
    if parser.state().symbols.type_alias_definitions.contains_key(&id) {
        return Err(ParserError::TypeAliasAlreadyDefined { span, id });
    }
    let ty = parser.parse_type()?.into_inner();
    parser.parse_semicolon()?;
    let span = span!(parser.source_id(), start, parser.token_stream().previous_end());
    define_type_alias(parser, id, span, ty);
    Ok(())
}

/// Parse the declaration of a recursion group.
///
///   rec-group-def ::= 'rec' '{' type-alias-def+ '}'
///
/// Inside the block, an alias that is not defined before the block names a member of the block,
/// whether or not it has been declared yet; the members are in the order of their declarations.
/// After the block, each alias names its member.
fn parse_rec_group_def<'input, P>(parser: &mut OperationParser<P>) -> ParseResult
where
    P: Parser<'input>,
{
    let rec_span = parser
        .token_stream_mut()
        .expect_if("'rec'", |tok| tok == Token::BareIdent("rec"))?
        .span();
    parser.parse_lbrace()?;
    parser.state_mut().symbols.rec_scope = Some(Default::default());
    let declarations = parse_rec_group_body(parser);
    let scope = parser.state_mut().symbols.rec_scope.take().expect("the scope of the block");
    declarations?;

    // Every alias of the block is declared in it.
    if let Some(alias) = scope.aliases().iter().find(|alias| alias.declaration.is_none()) {
        return Err(ParserError::UndefinedTypeAlias {
            span: alias.first_use,
            id: alias.id,
        });
    }
    let declared = scope.declared();
    if declared.is_empty() {
        return Err(ParserError::InvalidRecGroup {
            span: rec_span,
            reason: "a `rec` block declares at least one type".to_string(),
        });
    }

    // Renumber the variables from the order of first mention to the order of declaration.
    let mut renumbered = alloc::vec![0u32; declared.len()];
    for (member, position) in declared.iter().enumerate() {
        renumbered[*position as usize] = member as u32;
    }
    let context = parser.context_rc();
    let in_order = renumbered
        .iter()
        .enumerate()
        .all(|(position, member)| position as u32 == *member);
    let declarations = declared
        .iter()
        .map(|position| {
            let alias = &scope.aliases()[*position as usize];
            let declaration = alias.declaration.expect("every alias is declared");
            (alias.id, declaration.span(), declaration.into_inner())
        })
        .collect::<Vec<_>>();
    let bodies = declarations
        .iter()
        .map(|(_, _, body)| {
            if in_order {
                *body
            } else {
                context.substitute_rec_vars(*body, &mut |index| {
                    context.rec_var(renumbered[index as usize])
                })
            }
        })
        .collect::<SmallVec<[Type; 4]>>();

    let members = context.rec_group(&bodies).map_err(|err| {
        use crate::types::RecGroupError;
        let span = match err {
            RecGroupError::Empty => rec_span,
            RecGroupError::BareVariable { member }
            | RecGroupError::ScalarMember { member }
            | RecGroupError::MemberAsBody { member }
            | RecGroupError::VariableOutOfRange { member, .. } => declarations[member].1,
        };
        ParserError::InvalidType {
            span,
            reason: err.to_string(),
        }
    })?;
    for ((id, span, _), member) in declarations.into_iter().zip(members) {
        define_type_alias(parser, id, span, member);
    }
    Ok(())
}

/// Parse the declarations of a `rec` block, through its closing `}`, into the parser's scope.
fn parse_rec_group_body<'input, P>(parser: &mut OperationParser<P>) -> ParseResult
where
    P: Parser<'input>,
{
    let unexpected_eof = || ParserError::UnexpectedEof {
        expected: alloc::vec!["'type'".to_string(), "'}'".to_string()],
    };
    loop {
        let Some((start, token, end)) = parser.token_stream_mut().peek()? else {
            return Err(unexpected_eof());
        };
        match token {
            Token::Eof => return Err(unexpected_eof()),
            Token::Rbrace => {
                parser.parse_rbrace()?;
                return Ok(());
            }
            Token::BareIdent("rec") => {
                return Err(ParserError::InvalidRecGroup {
                    span: span!(parser.source_id(), start, end),
                    reason: "`rec` blocks do not nest".to_string(),
                });
            }
            Token::BareIdent("type") => {
                let (span, id) = parse_type_alias_def_head(parser)?;
                let symbols = &mut parser.state_mut().symbols;
                let scope = symbols.rec_scope.as_ref().expect("the scope of the block");
                if symbols.type_alias_definitions.contains_key(&id) || scope.is_declared(id) {
                    return Err(ParserError::TypeAliasAlreadyDefined { span, id });
                }
                symbols.rec_scope.as_mut().expect("the scope of the block").mention(id, span);
                let body = parser.parse_type()?.into_inner();
                parser.parse_semicolon()?;
                let span = span!(parser.source_id(), start, parser.token_stream().previous_end());
                parser
                    .state_mut()
                    .symbols
                    .rec_scope
                    .as_mut()
                    .expect("the scope of the block")
                    .declare(id, span, body);
            }
            token => {
                return Err(ParserError::UnexpectedToken {
                    span: span!(parser.source_id(), start, end),
                    token: token.to_string(),
                    expected: "a type declaration, 'type !name = ..;', or '}'".into(),
                });
            }
        }
    }
}

/// Parse the start of a type alias declaration, `'type' '!' alias-name '='`, and return the alias
/// and its span.
fn parse_type_alias_def_head<'input, P>(
    parser: &mut OperationParser<P>,
) -> ParseResult<(SourceSpan, interner::Symbol)>
where
    P: Parser<'input>,
{
    parser.token_stream_mut().expect(Token::BareIdent("type"))?;
    let (span, id) = parser
        .token_stream_mut()
        .expect_map("!-identifier", |tok| match tok {
            Token::BangIdent(id) => Some(interner::Symbol::intern(id)),
            _ => None,
        })?
        .into_parts();

    // Make sure this isn't invading the dialect type namespace.
    if id.as_str().contains('.') {
        return Err(ParserError::InvalidTypeAliasName {
            span,
            id,
            reason: "type names with a '.' are reserved for dialect-defined names".to_string(),
        });
    }

    parser.parse_equal()?;
    Ok((span, id))
}

/// Define the type alias `id` as `ty`, declared at `span`.
fn define_type_alias<'input, P>(
    parser: &mut OperationParser<P>,
    id: interner::Symbol,
    span: SourceSpan,
    ty: Type,
) where
    P: Parser<'input>,
{
    let state = parser.state_mut();
    if let Some(asm_state) = state.asm_state.as_deref_mut() {
        asm_state.add_type_alias_definition(id, span, ty);
    }
    state.symbols.type_alias_definitions.insert(id, Span::new(span, ty));
}
