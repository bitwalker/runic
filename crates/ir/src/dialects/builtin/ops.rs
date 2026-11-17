mod alias;
mod call;
mod cast;
mod component;
mod constants;
mod function;
mod global_variable;
mod local;
mod module;
mod segment;
mod world;
mod yield_op;

// The op builders of components and modules are not exported: they would create ops
// with names that skip the spelling and the checks of the builtin builders
pub use self::{
    alias::{FunctionAlias, FunctionAliasRef},
    call::{
        Call, CallIndirect, CallIndirectRef, CallRef, FunctionReference, FunctionReferenceRef,
        TailCall, TailCallIndirect, TailCallIndirectRef, TailCallRef,
    },
    cast::UnrealizedConversionCast,
    component::{Component, ComponentRef},
    constants::{Bytes, BytesRef},
    function::{Function, FunctionBuilder as PrimFunctionBuilder, FunctionRef, Ret, RetImm},
    global_variable::*,
    local::{LoadLocal, LoadLocalRef, LocalsResource, StoreLocal, StoreLocalRef},
    module::{Module, ModuleRef},
    segment::*,
    world::{World, WorldRef},
    yield_op::{Yield, YieldRef},
};
pub(crate) use self::{
    call::{CopiedAttributes, function_type_of, parse_signature, tail_call_mismatch},
    world::shadowing_error,
};

/// Parses the raw segment naming a module, a symbol table of `kind`, and adds its
/// canonical spelling (see [crate::SymbolPath::segments_of]) to `state` as the `name`
/// attribute.
fn parse_segment_name(
    state: &mut crate::OperationState,
    parser: &mut dyn crate::OpAsmParser<'_>,
    kind: crate::SymbolTableKind,
) -> crate::ParseResult {
    let name = parser.parse_symbol_name()?;
    let spelled = crate::SymbolPath::table_name(kind, name.name)
        .map_err(|err| crate::parse::ParserError::invalid_symbol_table_name(name.span, err))?;
    let name = crate::Ident::new(spelled, name.span);
    state.add_attribute(
        "name",
        parser.context_rc().create_attribute::<super::attributes::IdentAttr, _>(name),
    );
    Ok(())
}
