mod block;
mod builder;
mod callable;
pub mod callconv;
pub mod cfg;
mod context;
mod dialect;
pub mod dominance;
pub mod effects;
pub mod entity;
mod ident;
mod int;
mod int_range;
pub mod loops;
mod op;
mod operands;
mod operation;
pub mod parse;
pub mod print;
mod region;
mod successor;
pub(crate) mod symbols;
pub mod traits;
mod usable;
mod value;
pub mod verifier;
mod visit;

pub use runic_diagnostics::{Report, SourceSpan, Span, Spanned};
pub use runic_symbol as interner;

pub use self::{
    block::{
        Block, BlockCursor, BlockCursorMut, BlockId, BlockList, BlockOperand, BlockOperandRef,
        BlockRef, PostOrderBlockIter, PreOrderBlockIter,
    },
    builder::{Builder, BuilderExt, InsertionGuard, Listener, ListenerType, OpBuilder},
    callable::*,
    callconv::CallConv,
    context::{Context, DiagnosticScope},
    dialect::{
        Dialect, DialectAttributeRegistrationInfo, DialectInfo, DialectOpRegistrationInfo,
        DialectRegistration, DialectRegistrationHook, DialectRegistrationHookInfo,
        DialectRegistrationInfo,
    },
    entity::{
        Entity, EntityGroup, EntityId, EntityList, EntityListCursor, EntityListCursorMut,
        EntityListItem, EntityListIter, EntityMap, EntityMapCursor, EntityMapCursorMut,
        EntityMapItem, EntityMapIter, EntityMut, EntityParent, EntityProjection,
        EntityProjectionMut, EntityRange, EntityRangeMut, EntityRef, EntityStorage, EntityWithId,
        EntityWithKey, EntityWithParent, MaybeDefaultEntityListIter, MaybeDefaultEntityMapIter,
        RawEntityRef, StorableEntity, UnsafeEntityRef, UnsafeIntrusiveEntityRef,
        UnsafeIntrusiveMapEntityRef,
    },
    ident::{FunctionIdent, Ident, IdentAttr},
    int::Int,
    int_range::{ConstantIntRange, IntegerRangeConfidence, IntegerValueRange},
    op::{BuildableOp, GenericSuccessorLayout, Op, OpExt, OpRegistration},
    operands::{
        OpOperand, OpOperandImpl, OpOperandList, OpOperandRange, OpOperandRangeMut,
        OpOperandStorage,
    },
    operation::{
        AttrInfo, GenericOperationBuilder, OpCursor, OpCursorMut, OpList, Operation,
        OperationBuilder, OperationName, OperationRef, OperationState, PendingSuccessorInfo,
        equivalence,
    },
    parse::{OpAsmParser, OpParser, ParseResult},
    print::{AttrPrinter, OpPrinter, OpPrintingFlags},
    region::{
        ExitDefect, ExitDefectKind, ExitResolution, HasBreakingControlFlowOpInterface,
        InvocationBounds, LoopLikeOpInterface, PropagateControlFlowBreak, Region,
        RegionBranchOpInterface, RegionBranchPoint, RegionBranchTerminatorOpInterface,
        RegionCursor, RegionCursorMut, RegionExitKind, RegionExitTerminatorOpInterface, RegionKind,
        RegionKindInterface, RegionList, RegionRef, RegionSuccessor, RegionSuccessorInfo,
        RegionSuccessorIter, RegionSuccessorKind, RegionTransformFailed, ResolvedExit,
    },
    successor::{
        KeyedSuccessor, KeyedSuccessorRange, KeyedSuccessorRangeMut, OpSuccessor, OpSuccessorMut,
        OpSuccessorRange, OpSuccessorRangeMut, OpSuccessorStorage, SuccessorInfo, SuccessorOperand,
        SuccessorOperandRange, SuccessorOperandRangeMut, SuccessorOperands, SuccessorWithKey,
        SuccessorWithKeyMut,
    },
    symbols::*,
    traits::{FoldResult, OpFoldResult},
    usable::Usable,
    value::{
        AsValueRange, BlockArgument, BlockArgumentRange, BlockArgumentRangeMut, BlockArgumentRef,
        OpResult, OpResultRange, OpResultRangeMut, OpResultRef, OpResultStorage, Value, ValueId,
        ValueRange, ValueRef,
    },
    verifier::{OpVerifier, Verify},
    visit::{
        OpVisitor, OperationVisitor, RawWalk, ReverseBlock, Searcher, SymbolVisitor, Visitor, Walk,
        WalkMut, WalkOrder, WalkResult, WalkStage,
    },
};
pub use crate::{
    dialects::builtin::types::{
        ArrayType, EnumType, EnumVariant, FloatType, FunctionType, IntegerType, InvalidEnumType,
        IsizeType, NeverType, PointerType, RecVarType, StructField, StructRepr, StructType,
        UnknownType,
    },
    types::{
        RecGroup, RecGroupError, Type, TypeAliasName, TypeComponent, TypeFormatter, TypeImpl,
        TypeName, TypeParser, TypePrinter, TypeRegistration,
    },
};
