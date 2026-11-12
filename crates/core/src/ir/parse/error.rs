use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    BlockId, CompactString, ValueId,
    diagnostics::{Diagnostic, OwnedDiagnostic, OwnedLabel, Report, SourceSpan},
    formatter::DisplayValues,
    interner,
};

#[derive(Debug, thiserror::Error, Diagnostic)]
pub enum ParserError {
    #[error("unexpected end of input: expected one of {}", DisplayValues::new(expected.iter()))]
    UnexpectedEof { expected: Vec<String> },
    #[error("invalid syntax")]
    UnexpectedToken {
        #[label(primary, "unexpected token '{token}'")]
        span: SourceSpan,
        token: String,
        #[help]
        expected: Option<String>,
    },
    #[error("invalid syntax")]
    InvalidCharacter {
        #[label(primary, "unexpected character '{character}'")]
        span: SourceSpan,
        character: char,
    },
    #[error("invalid syntax")]
    UnclosedQuote {
        #[label(primary, "missing closing quote for string starting here")]
        span: SourceSpan,
    },
    #[error("invalid syntax")]
    UnclosedDelimiter {
        #[label(primary, "missing closing delimiter '{expected}'")]
        span: SourceSpan,
        expected: char,
    },
    #[error("invalid syntax")]
    #[diagnostic(help = "Only '\"', '\\r', '\\n', and '\\t' may be escaped in strings")]
    InvalidEscapeSequence {
        #[label(primary, "invalid escape sequence")]
        span: SourceSpan,
    },
    #[error("invalid integer literal")]
    InvalidIntegerLiteral {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        reason: String,
    },
    #[error("invalid string")]
    InvalidString {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        reason: String,
    },
    #[error("invalid attribute value")]
    InvalidAttributeValue {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        reason: String,
    },
    #[error("invalid operand and type list")]
    OperandAndTypeListMismatch {
        #[label(
            primary,
            "there are {num_operands} operands, but {num_types} types: expected the same number \
             of both"
        )]
        span: SourceSpan,
        num_operands: usize,
        num_types: usize,
    },
    #[error("invalid operand list")]
    InvalidOperandCount {
        #[label(
            primary,
            "expected {expected} operands in this list, but found {actual}"
        )]
        span: SourceSpan,
        expected: usize,
        actual: usize,
    },
    #[error("use of undeclared values")]
    UndeclaredValueUses {
        #[labels]
        labels: Vec<OwnedLabel>,
    },
    #[error("operation location alias was never defined")]
    UnresolvedLocationAlias {
        #[label(primary, "occurs here")]
        span: SourceSpan,
    },
    #[error("invalid location alias value")]
    InvalidLocationAlias {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        reason: String,
    },
    #[error("use of undefined blocks")]
    UndefinedBlocks {
        #[labels]
        labels: Vec<OwnedLabel>,
    },
    #[error("redefinition of SSA value")]
    ValueRedefinition {
        #[label(primary, "occurs here")]
        span: SourceSpan,
        #[label("previously defined here")]
        prev_span: SourceSpan,
    },
    #[error("definition of SSA value does not match expected type")]
    ValueDefinitionTypeMismatch {
        #[label(primary, "definition here has type {ty}")]
        span: SourceSpan,
        #[label("previously used here with type {prev_ty}")]
        prev_span: SourceSpan,
        ty: CompactString,
        prev_ty: CompactString,
    },
    #[error("invalid use of result number in argument list")]
    ResultNumberUsedInArgumentList {
        #[label(primary)]
        span: SourceSpan,
        #[label]
        value_span: SourceSpan,
    },
    #[error("invalid result index")]
    InvalidResultIndex {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        #[label]
        value_span: SourceSpan,
        reason: String,
    },
    #[error("use of value expects different type than prior uses")]
    ValueUseTypeMismatch {
        #[label(primary, "use here expects {ty}")]
        span: SourceSpan,
        #[label("previous use here expects {prev_ty}")]
        prev_span: SourceSpan,
        ty: CompactString,
        prev_ty: CompactString,
    },
    #[error("mismatched value and type lists")]
    MismatchedValueAndTypeLists {
        #[label(primary, "there are {num_values} values, but {num_types} types")]
        span: SourceSpan,
        num_values: usize,
        num_types: usize,
    },
    #[error("invalid result count")]
    InvalidResultCount { span: SourceSpan },
    #[error("named op with no results")]
    NamedOpWithNoResults { span: SourceSpan },
    #[error("result count mismatch")]
    ResultCountMismatch {
        span: SourceSpan,
        count: usize,
        expected: usize,
    },
    #[error("invalid successor list")]
    InvalidEmptySuccessorList {
        #[label(primary, "expected at least one successor")]
        span: SourceSpan,
    },
    #[error("invalid operation name")]
    InvalidOperationName {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        reason: String,
    },
    #[error("unknown dialect '{dialect}'")]
    #[diagnostic(help = "only dialects registered in this context can be named here")]
    UnknownDialect {
        #[label(primary, "this dialect is not registered")]
        span: SourceSpan,
        dialect: interner::Symbol,
    },
    #[error("unknown type '{name}' in dialect '{dialect}'")]
    #[diagnostic(help = "only types registered with the dialect can be named here")]
    UnknownType {
        #[label(primary, "this type is not registered with '{dialect}'")]
        span: SourceSpan,
        dialect: interner::Symbol,
        name: CompactString,
    },
    #[error("invalid type")]
    TypeTakesNoParameters {
        #[label(primary, "type '{name}' takes no parameters, but '<' follows it")]
        span: SourceSpan,
        name: String,
    },
    #[error("invalid type")]
    TypeRequiresParameters {
        #[label(primary, "type '{name}' requires parameters in '<..>'")]
        span: SourceSpan,
        name: String,
    },
    #[error("invalid type")]
    InvalidType {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        reason: String,
    },
    #[error("use of undefined type alias '!{id}'")]
    UndefinedTypeAlias {
        #[label(primary, "no type alias with this name is defined before this point")]
        span: SourceSpan,
        id: interner::Symbol,
    },
    #[error("invalid operation")]
    NonTerminatorWithSuccessors {
        #[label(primary, "non-terminator operations cannot have successors")]
        span: SourceSpan,
    },
    #[error("generic assembly cannot represent the successor layout of '{name}'")]
    #[diagnostic(help = "use the operation's custom assembly syntax instead")]
    UnsupportedGenericSuccessorLayout {
        #[label(
            primary,
            "generic successors cannot encode keys or multiple successor groups"
        )]
        span: SourceSpan,
        name: String,
    },
    #[error("invalid successor count")]
    InvalidSuccessorCount {
        #[label(primary, "expected {expected} successors, but found {actual}")]
        span: SourceSpan,
        expected: usize,
        actual: usize,
    },
    #[error("too many operand groups")]
    TooManyOperandGroups {
        #[label(
            primary,
            "this successor needs an operand group past the limit of {limit}"
        )]
        span: SourceSpan,
        limit: usize,
    },
    #[error("invalid operation type")]
    InvalidOperationType {
        #[label(primary)]
        span: SourceSpan,
        #[label("{reason}")]
        ty_span: SourceSpan,
        reason: String,
    },
    #[error("invalid operation")]
    UnknownOperation {
        #[label(primary, "unknown/unregistered operation")]
        span: SourceSpan,
    },
    #[error("invalid attribute")]
    UnknownAttribute {
        #[label(primary, "unknown/unregistered attribute")]
        span: SourceSpan,
    },
    #[error("invalid custom operation")]
    InvalidCustomOperation {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        reason: String,
    },
    #[error("invalid block name in region with named arguments")]
    BlockNameInRegionWithNamedArgs {
        #[label(primary)]
        span: SourceSpan,
    },
    #[error("region entry argument is already in use")]
    RegionArgumentAlreadyDefined {
        #[label(primary, "attempted to redefine {arg} here")]
        span: SourceSpan,
        #[label("previously defined here")]
        prev_span: SourceSpan,
        arg: ValueId,
    },
    #[error("entry block arguments were already defined")]
    EntryBlockArgumentsAlreadyDefined {
        #[label(primary)]
        span: SourceSpan,
    },
    #[error("redefinition of block {name}")]
    BlockAlreadyDefined {
        #[label(primary)]
        span: SourceSpan,
        name: BlockId,
    },
    /// A symbol table defines two symbols with one name, or with one path (see
    /// [crate::SymbolConflict])
    #[error("{conflict}")]
    SymbolConflict {
        #[label(primary)]
        span: SourceSpan,
        conflict: crate::SymbolConflict,
    },
    #[error("too many arguments specified in argument list")]
    TooManyBlockArguments {
        #[label(primary)]
        span: SourceSpan,
    },
    #[error("block argument type mismatch")]
    BlockArgumentTypeMismatch {
        #[label(primary, "expected {expected}, got {ty}")]
        span: SourceSpan,
        arg: ValueId,
        ty: CompactString,
        expected: CompactString,
    },
    #[error("attribute '{name}' occurs more than once in the attribute list")]
    DuplicateAttribute {
        #[label(primary)]
        span: SourceSpan,
        name: interner::Symbol,
    },
    #[error("invalid attribute alias definition")]
    AttributeAliasAlreadyDefined {
        #[label(primary, "alias '{id}' is already defined")]
        span: SourceSpan,
        id: interner::Symbol,
    },
    #[error("invalid attribute alias name '{id}'")]
    InvalidAttributeAliasName {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        id: interner::Symbol,
        reason: String,
    },
    #[error("invalid type alias definition")]
    TypeAliasAlreadyDefined {
        #[label(primary, "alias '{id}' is already defined")]
        span: SourceSpan,
        id: interner::Symbol,
    },
    #[error("invalid type alias definition")]
    BareTypeAliasDefinition {
        #[label(primary, "a type alias is declared with `type !name = ..;`")]
        span: SourceSpan,
    },
    #[error("invalid type alias name '{id}'")]
    InvalidTypeAliasName {
        #[label(primary, "{reason}")]
        span: SourceSpan,
        id: interner::Symbol,
        reason: String,
    },
    #[error("invalid file metadata")]
    UnknownFileMetadata {
        #[label(primary, "'{key}' is not a recognized metadata key")]
        span: SourceSpan,
        key: CompactString,
    },
    /// The name of a component or module cannot name it, e.g. one of its segments is not
    /// valid.
    #[error("{reason}")]
    InvalidSymbolTableName {
        /// The span of the name
        #[label(primary, "this name is invalid")]
        span: SourceSpan,
        /// Why the name is invalid
        reason: String,
    },
    #[error("{0}")]
    #[diagnostic(transparent)]
    Report(#[diagnostic_source] OwnedDiagnostic),
}

impl ParserError {
    /// The name at `span` cannot name a symbol-table op, for the reason `error` gives.
    pub fn invalid_symbol_table_name(span: SourceSpan, error: Report) -> Self {
        Self::InvalidSymbolTableName {
            span,
            reason: error.to_string(),
        }
    }
}

impl From<Report> for ParserError {
    #[inline]
    fn from(value: Report) -> Self {
        Self::Report(value.into())
    }
}
